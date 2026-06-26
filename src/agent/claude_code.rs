//! Claude Code subprocess engine.
//!
//! Drives the `claude` CLI in headless mode (`-p --output-format stream-json`)
//! so a remote daemon can serve "act as if I'm running `claude` in this
//! shell" UX over a WS without piping a TTY. Each `run_turn`:
//!
//!   1. Spawns `claude -p <prompt>` in the workspace dir, with
//!      `--session-id <uuid>` on the first turn and `--resume <uuid>` on
//!      every subsequent turn so context threads through.
//!   2. Reads `stream-json` JSON-Lines from stdout, translates the
//!      `assistant` / `user(tool_result)` / `result` event flavors into the
//!      same [`Event`] enum the ReAct loop emits.
//!   3. Awaits the child exit; returns the `result.result` string as the
//!      final answer.
//!
//! Permission model: by design Claude Code's headless mode cannot show a
//! TTY confirmation, so any tool that would normally ask the operator
//! fails. We pass `--dangerously-skip-permissions` so worktree runs (the
//! default) execute end-to-end. Routing those approvals back through the
//! kotonia-cli `ApprovalHandler` requires a custom MCP server attached via
//! `--permission-prompt-tool` — that's a future stage; today we trust the
//! worktree boundary.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::execution::host::ExecutionResult;

use super::agent::{Event, EventSink};
use super::history::HistoryStore;
use super::provider::ChatMsg;

/// Claude Code requires `--session-id` to be a valid UUID. Host-side
/// session ids (kotonia-cli's `<YYYYMMDD-HHMMSS>-<4hex>` form) don't
/// qualify, so derive a stable UUID v5 from the host id when the input
/// isn't already a UUID. Same input → same UUID, so subsequent `--resume`
/// against the host id keeps working.
pub fn claude_code_session_id(host_session_id: &str) -> String {
    if uuid::Uuid::parse_str(host_session_id).is_ok() {
        return host_session_id.to_string();
    }
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, host_session_id.as_bytes())
        .to_string()
}

/// Where the `claude` binary lives. The PATH lookup happens lazily on
/// spawn, so we just remember the name (or an override).
#[derive(Clone, Debug)]
pub struct ClaudeCodeConfig {
    /// Binary name or absolute path. Defaults to `"claude"`.
    pub binary: PathBuf,
    /// Workspace cwd the subprocess runs in.
    pub workspace_root: PathBuf,
    /// Session id used for `--session-id` (first turn) and `--resume`
    /// (subsequent turns). Caller supplies — usually the same id the host
    /// uses for its own session log.
    pub session_id: String,
}

pub struct ClaudeCodeEngine {
    config: ClaudeCodeConfig,
    /// True until we've successfully started one session. Drives the
    /// `--session-id` vs `--resume` switch.
    first_turn: bool,
}

#[derive(Debug)]
pub enum ClaudeCodeError {
    Spawn(String),
    Io(String),
    NonZeroExit { code: Option<i32>, stderr: String },
    NoFinalResult,
}

impl std::fmt::Display for ClaudeCodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaudeCodeError::Spawn(m) => write!(f, "spawn claude: {m}"),
            ClaudeCodeError::Io(m) => write!(f, "claude io: {m}"),
            ClaudeCodeError::NonZeroExit { code, stderr } => {
                let code_str = code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string());
                if stderr.is_empty() {
                    write!(f, "claude exited with status {code_str}")
                } else {
                    write!(f, "claude exited with status {code_str}: {stderr}")
                }
            }
            ClaudeCodeError::NoFinalResult => {
                write!(f, "claude produced no `result` event — see stderr")
            }
        }
    }
}

impl std::error::Error for ClaudeCodeError {}

impl ClaudeCodeEngine {
    pub fn new(workspace_root: &Path, session_id: String) -> Self {
        Self {
            config: ClaudeCodeConfig {
                binary: PathBuf::from("claude"),
                workspace_root: workspace_root.to_path_buf(),
                session_id,
            },
            first_turn: true,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.config.session_id
    }

    /// Run one turn against the Claude Code subprocess. Streams events into
    /// `sink` as they arrive on stdout; returns the final assistant text.
    pub async fn run_turn(
        &mut self,
        prompt: &str,
        sink: &mut dyn EventSink,
    ) -> Result<String, ClaudeCodeError> {
        let mut cmd = Command::new(&self.config.binary);
        cmd.arg("-p")
            .arg(prompt)
            .arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("--dangerously-skip-permissions")
            .current_dir(&self.config.workspace_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if self.first_turn {
            cmd.arg("--session-id").arg(&self.config.session_id);
        } else {
            cmd.arg("--resume").arg(&self.config.session_id);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| ClaudeCodeError::Spawn(e.to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ClaudeCodeError::Io("stdout pipe missing".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| ClaudeCodeError::Io("stderr pipe missing".into()))?;

        // Drain stderr in the background so the child can never block on a
        // full pipe; capture for the error report.
        let stderr_task = tokio::spawn(async move {
            let mut buf = String::new();
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                buf.push_str(&line);
                buf.push('\n');
            }
            buf
        });

        let mut reader = BufReader::new(stdout).lines();
        let mut iteration = 0u32;
        let mut final_text: Option<String> = None;
        // The `result` event carries the last assistant text, but it
        // arrives only at the end. Track the in-progress text so a caller
        // who's just streaming events still sees content immediately.
        // (We don't currently emit incremental text — we batch one
        // `LlmThinking` per assistant turn — but the field is here so
        // future stages can emit prefix tokens.)
        let _ = &mut iteration;

        while let Some(line) = reader
            .next_line()
            .await
            .map_err(|e| ClaudeCodeError::Io(e.to_string()))?
        {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    sink.emit(Event::Malformed {
                        excerpt: format!(
                            "claude stream-json parse error: {e} (line: {})",
                            truncate(&line, 240)
                        ),
                    });
                    continue;
                }
            };
            match value.get("type").and_then(|v| v.as_str()) {
                Some("system") => {
                    if value.get("subtype").and_then(|v| v.as_str()) == Some("init") {
                        iteration += 1;
                        // Treat each Claude Code session start as one
                        // "iteration" for the host's progress display.
                        sink.emit(Event::IterationStart {
                            iteration,
                            max: 0, // Claude Code self-paces; no host cap.
                        });
                    }
                }
                Some("assistant") => {
                    let blocks = value
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let mut had_text = false;
                    for block in blocks {
                        match block.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                let text = block
                                    .get("text")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if !text.trim().is_empty() && !had_text {
                                    sink.emit(Event::LlmThinking);
                                    had_text = true;
                                }
                                // The `result` event below carries the
                                // canonical final answer. Emitting Final
                                // here too would deliver it twice (the
                                // earlier draft did this to stream prose
                                // to the web UI as it arrives — that's a
                                // job for a dedicated streaming event,
                                // not Final).
                            }
                            Some("tool_use") => {
                                let name = block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let input = block.get("input").cloned().unwrap_or(Value::Null);
                                let display = render_tool_invocation(&name, &input);
                                sink.emit(Event::Bash { command: display });
                            }
                            _ => {}
                        }
                    }
                }
                Some("user") => {
                    // Claude Code emits a synthetic user message whose
                    // content is `tool_result` blocks — the executed tool's
                    // output. Map them to `Observation`.
                    let blocks = value
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_array())
                        .cloned()
                        .unwrap_or_default();
                    for block in blocks {
                        if block.get("type").and_then(|v| v.as_str()) != Some("tool_result") {
                            continue;
                        }
                        let combined = block
                            .get("content")
                            .map(stringify_tool_result_content)
                            .unwrap_or_default();
                        let is_error = block
                            .get("is_error")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let result = ExecutionResult {
                            combined,
                            exit_code: if is_error { 1 } else { 0 },
                            timed_out: false,
                            truncated: false,
                        };
                        sink.emit(Event::Observation { result });
                    }
                }
                Some("result") => {
                    let is_error = value
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let result_text = value
                        .get("result")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if is_error {
                        sink.emit(Event::Error {
                            message: if result_text.is_empty() {
                                "claude returned an error result".into()
                            } else {
                                result_text.clone()
                            },
                        });
                    } else {
                        sink.emit(Event::Final {
                            answer: result_text.clone(),
                        });
                    }
                    final_text = Some(result_text);
                }
                Some("error") => {
                    let message = value
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("claude reported an error event")
                        .to_string();
                    sink.emit(Event::Error { message });
                }
                // Quietly-ignored housekeeping events Claude Code emits.
                // Surfacing these as "malformed" buries the real signal.
                Some("rate_limit_event")
                | Some("stream_event")
                | Some("compact_boundary") => {}
                _ => {
                    // Unknown event type — surface as malformed so the
                    // operator can investigate, but keep going.
                    sink.emit(Event::Malformed {
                        excerpt: format!(
                            "unknown claude stream event: {}",
                            truncate(&line, 240)
                        ),
                    });
                }
            }
        }

        let status = child
            .wait()
            .await
            .map_err(|e| ClaudeCodeError::Io(e.to_string()))?;
        let stderr_buf = stderr_task.await.unwrap_or_default();

        sink.emit(Event::Done {
            iterations: iteration.max(1),
            success: status.success() && final_text.is_some(),
        });

        if !status.success() {
            return Err(ClaudeCodeError::NonZeroExit {
                code: status.code(),
                stderr: stderr_buf.trim().to_string(),
            });
        }
        let answer = final_text.ok_or(ClaudeCodeError::NoFinalResult)?;
        // After the first successful turn, future turns must use --resume.
        self.first_turn = false;
        Ok(answer)
    }
}

/// Render a Claude tool invocation as a single shell-style line so the
/// host's `Event::Bash` rendering ("$ <cmd>") still makes sense. Bash gets
/// passed through verbatim; other tools get a `[<name>] <args-json>`
/// summary so the operator can see what's happening.
fn render_tool_invocation(name: &str, input: &Value) -> String {
    if name == "Bash" {
        if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
            return cmd.to_string();
        }
    }
    let args = serde_json::to_string(input).unwrap_or_else(|_| "{}".into());
    format!("[{name}] {args}")
}

/// Tool result `content` can be either a plain string or an array of
/// `{type:"text", text:"..."}` blocks. Flatten into one string so the
/// `Event::Observation::result.combined` stays the same shape as the ReAct
/// path's `ExecutionResult`.
fn stringify_tool_result_content(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_string();
    }
    if let Some(arr) = value.as_array() {
        let mut out = String::new();
        for block in arr {
            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                out.push_str(text);
                out.push('\n');
            }
        }
        return out;
    }
    serde_json::to_string(value).unwrap_or_default()
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push('…');
        out
    }
}

/// Host-side wrapper bundling [`ClaudeCodeEngine`] with the kotonia history
/// store + light metadata. Mirrors enough of the [`super::agent::Agent`]
/// surface that `main.rs` / `daemon.rs` can dispatch through a small enum
/// without case-splitting at every call site.
pub struct ClaudeCodeAgent {
    engine: ClaudeCodeEngine,
    history: Option<HistoryStore>,
    in_place: bool,
}

impl ClaudeCodeAgent {
    pub fn new(workspace_root: &Path, session_id: String, in_place: bool) -> Self {
        Self {
            engine: ClaudeCodeEngine::new(workspace_root, session_id),
            history: None,
            in_place,
        }
    }

    pub fn with_history(mut self, history: HistoryStore) -> Self {
        self.history = Some(history);
        self
    }

    /// Seed the on-disk history from a resumed session. Claude Code itself
    /// owns its conversation state; we mirror messages into the JSONL log
    /// so `--list-sessions` / `--resume` work the same way for both
    /// engines.
    pub fn seed_messages(&mut self, _prior: Vec<ChatMsg>) {
        // No-op: Claude Code is authoritative for its own context window
        // via the `--resume <session_id>` flag. The host-side messages
        // vector exists only for display; future stages can hydrate it
        // when we add a transcript-rebuild path.
    }

    pub fn log_initial_system(&mut self) {
        if let Some(h) = &mut self.history {
            let _ = h.append_message(
                &super::provider::ChatRole::System,
                "[claude-code engine — Claude Code owns its own conversation state]",
            );
        }
    }

    pub fn provider_label(&self) -> String {
        format!("claude-code ({})", self.engine.session_id())
    }

    pub fn backend_label(&self) -> &'static str {
        "claude-code"
    }

    pub fn model_id(&self) -> &'static str {
        "claude-code"
    }

    pub fn session_id(&self) -> Option<&str> {
        self.history.as_ref().map(|h| h.session_id.as_str())
    }

    /// Claude Code drives its own tool catalog (read/edit/bash/web/…), so
    /// the "native_mode" surface used by the ReAct banner is meaningless
    /// here. Report `true` so the banner doesn't suggest the delimiter
    /// fallback is in play.
    pub fn native_mode(&self) -> bool {
        true
    }

    pub fn in_place(&self) -> bool {
        self.in_place
    }

    pub async fn run_turn(
        &mut self,
        task: &str,
        sink: &mut dyn EventSink,
    ) -> Result<String, ClaudeCodeError> {
        if let Some(h) = &mut self.history {
            let _ = h.append_turn_start();
            let _ = h.append_message(&super::provider::ChatRole::User, task);
        }
        let result = self.engine.run_turn(task, sink).await;
        if let Some(h) = &mut self.history {
            match &result {
                Ok(answer) => {
                    let _ = h.append_message(&super::provider::ChatRole::Assistant, answer);
                    let _ = h.append_turn_end(1, true);
                }
                Err(e) => {
                    let _ = h.append_message(
                        &super::provider::ChatRole::Assistant,
                        &format!("[claude-code error: {e}]"),
                    );
                    let _ = h.append_turn_end(1, false);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_bash_passes_command_through() {
        let input = serde_json::json!({"command": "ls -la"});
        assert_eq!(render_tool_invocation("Bash", &input), "ls -la");
    }

    #[test]
    fn render_other_tool_summary() {
        let input = serde_json::json!({"path": "/tmp/x"});
        let out = render_tool_invocation("Read", &input);
        assert!(out.starts_with("[Read]"));
        assert!(out.contains("/tmp/x"));
    }

    #[test]
    fn stringify_tool_result_handles_text_blocks() {
        let v = serde_json::json!([{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]);
        let s = stringify_tool_result_content(&v);
        assert!(s.contains("hello"));
        assert!(s.contains("world"));
    }

    #[test]
    fn stringify_tool_result_handles_plain_string() {
        let v = serde_json::json!("plain");
        assert_eq!(stringify_tool_result_content(&v), "plain");
    }
}
