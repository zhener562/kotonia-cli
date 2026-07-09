//! kotonia-cli agent loop.
//!
//! Two execution modes, picked automatically based on the backend:
//!
//! 1. **Native tool calling** (Gemma 4 26B on vLLM with `--tool-call-parser`,
//!    DeepSeek API, kotonia hosted `/api/v1/chat/completions`). The model
//!    emits `tool_calls` directly; the agent dispatches `bash` /
//!    `web_search`. No delimiter parsing.
//! 2. **Delimiter fallback** (V4-Flash on llama.cpp — no tool-call parser).
//!    The model emits a `<<<BASH>>>` block; we execute it; the stdout
//!    becomes the next turn's user message. Loop until the model emits
//!    `<<<FINAL_ANSWER>>>` or hits the iteration cap.
//!
//! Approval policy gates every bash invocation in both modes.
//!
//! The agent owns nothing transactional — `AgentWorkspace` (worktree or
//! in-place) is passed in by the caller, so the binary controls cleanup.
//! The conversation history persists across `run_turn` calls so the CLI's
//! REPL mode keeps full context.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::Serialize;

use crate::ai::{AiCallOptions, AiContent, AiMessage, AiRole, AiStopReason, AiTool};
use crate::execution::host::{ExecutionResult, HostExecutor};

use super::approval::{decide, ApprovalMode, Decision};
use super::history::HistoryStore;
use super::parse::{parse, Action};
use super::prompt::{system_prompt, system_prompt_native};
use super::provider::{ChatMsg, ChatRole, Provider};

const DEFAULT_MAX_ITERATIONS: u32 = 30;
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// What an `ApprovalHandler` returns when the policy demands a human.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    Approve,
    Deny,
}

/// Caller-supplied prompt-for-approval. The CLI prints to stderr and reads
/// stdin; tests / web frontends can implement their own.
pub trait ApprovalHandler: Send {
    fn ask(&mut self, command: &str, reason: &str) -> ApprovalOutcome;
}

/// Events the agent emits as it runs. The CLI renders these to stdout;
/// the `serve` JSON protocol serializes them one-per-line (the `type` tag
/// is the snake_case variant name, so `IterationStart` → `iteration_start`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    IterationStart { iteration: u32, max: u32 },
    LlmThinking,
    Bash { command: String },
    BashSkipped { command: String, reason: String },
    Observation { result: ExecutionResult },
    Final { answer: String },
    Malformed { excerpt: String },
    Error { message: String },
    Done { iterations: u32, success: bool },
}

pub trait EventSink: Send {
    fn emit(&mut self, event: Event);
}

pub struct AgentConfig {
    pub approval: ApprovalMode,
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub in_place: bool,
    /// When set, the system prompt advertises the kotonia /api/v1 image /
    /// audio / video endpoints (using `$KOTONIA_API_KEY` from the inherited
    /// shell env). Leave None to hide the capability entirely.
    pub kotonia_api_base: Option<String>,
    /// Force the delimiter-based ReAct loop even when the backend supports
    /// native `tools` calling. Useful for diffing / debugging when the
    /// native path misbehaves.
    pub force_delimiter: bool,
    /// Persona-level system prompt prefix. Prepended verbatim to the
    /// agent's tool-aware base prompt, separated by a `---` block, so
    /// downstream consumers (kotonia-desktop's Iris character) can give
    /// the agent a voice / personality without forking the base prompt.
    /// Leave `None` for the plain CLI/daemon experience.
    pub persona_prefix: Option<String>,
}

impl AgentConfig {
    pub fn new(approval: ApprovalMode, in_place: bool) -> Self {
        Self {
            approval,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_tokens: DEFAULT_MAX_TOKENS,
            in_place,
            kotonia_api_base: None,
            force_delimiter: false,
            persona_prefix: None,
        }
    }
}

pub struct Agent {
    provider: Provider,
    executor: HostExecutor,
    config: AgentConfig,
    /// Whether to drive the agent through the native `tools` surface or the
    /// legacy delimiter parser. Decided at construction time from the
    /// backend's capability + the user's force flag.
    native_mode: bool,
    /// Tool catalog advertised to the native path. Empty in delimiter mode.
    tool_catalog: Vec<AiTool>,
    /// Cached system prompt text. Kept separately from the message vec so
    /// the native path can pass it via `AiCallOptions::with_system` without
    /// shipping it as a `role=system` AiMessage on every call.
    system_prompt: String,
    /// Delimiter-mode conversation. The first entry is always the system
    /// message; subsequent entries are user/assistant turns.
    messages: Vec<ChatMsg>,
    /// Native-mode conversation, excluding the system message. Lazily
    /// hydrated from `messages` on first `run_turn` so resumed sessions
    /// keep their text-only context.
    native_messages: Vec<AiMessage>,
    history: Option<HistoryStore>,
    /// Coarse cancellation flag. Checked at each iteration boundary so a
    /// frontend can stop a runaway turn between tool calls (in-flight bash /
    /// LLM calls still complete first — that's the "coarse" tradeoff). Reset
    /// to `false` at the start of every `run_turn`.
    cancel: Arc<AtomicBool>,
}

impl Agent {
    pub fn new(workspace_root: &Path, provider: Provider, config: AgentConfig) -> Self {
        let executor = HostExecutor::new(workspace_root.to_path_buf());
        let native_mode = provider.supports_native_tools() && !config.force_delimiter;
        let kotonia_base = config.kotonia_api_base.as_deref();
        let base_prompt = if native_mode {
            system_prompt_native(workspace_root, config.in_place, kotonia_base)
        } else {
            system_prompt(workspace_root, config.in_place, kotonia_base)
        };
        let system_prompt_text = match config.persona_prefix.as_deref() {
            Some(prefix) if !prefix.trim().is_empty() => {
                format!("{}\n\n---\n\n{}", prefix.trim(), base_prompt)
            }
            _ => base_prompt,
        };
        let messages = vec![ChatMsg::system(system_prompt_text.clone())];
        let tool_catalog = if native_mode { build_tool_catalog() } else { Vec::new() };
        Self {
            provider,
            executor,
            config,
            native_mode,
            tool_catalog,
            system_prompt: system_prompt_text,
            messages,
            native_messages: Vec::new(),
            history: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A shared handle to the cancellation flag. A frontend (e.g. the `serve`
    /// stdin reader) sets it to `true` to request that the current turn stop
    /// at the next iteration boundary.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    /// Attach an on-disk session log. Every message pushed into the
    /// conversation (system / user / assistant / observation) is appended
    /// best-effort; write failures are logged to stderr but do not abort
    /// the run.
    pub fn with_history(mut self, history: HistoryStore) -> Self {
        self.history = Some(history);
        self
    }

    /// Pre-load conversation history from a resumed session. The system
    /// prompt that `new()` installed stays in place — only the previous
    /// user/assistant exchange is appended after it.
    pub fn seed_messages(&mut self, prior: Vec<ChatMsg>) {
        self.messages.extend(prior);
    }

    /// Log the freshly-installed system prompt to history. Call once after
    /// `with_history` on a new (non-resumed) session.
    pub fn log_initial_system(&mut self) {
        if let Some(msg) = self.messages.first().cloned() {
            self.history_append(&msg);
        }
    }

    pub fn provider_label(&self) -> String {
        format!("{} ({})", self.provider.model_id(), self.provider.backend_label())
    }

    /// Short tag of the provider backend (e.g. `kotonia` / `deepseek` /
    /// a user-defined provider name). Use this when persisting metadata —
    /// parsing `provider_label()` is fragile.
    pub fn backend_label(&self) -> &str {
        self.provider.backend_label()
    }

    /// Public model id of the active backend (e.g. `kotonia-gemma4-26b`,
    /// `deepseek-chat:thinking`). For history headers / observability.
    pub fn model_id(&self) -> &str {
        self.provider.model_id()
    }

    pub fn session_id(&self) -> Option<&str> {
        self.history.as_ref().map(|h| h.session_id.as_str())
    }

    fn history_append(&mut self, msg: &ChatMsg) {
        if let Some(h) = &mut self.history {
            if let Err(e) = h.append_message(&msg.role, &msg.content) {
                eprintln!("kotonia-cli: history append failed: {e}");
            }
        }
    }

    fn history_append_observation(&mut self, result: &ExecutionResult) {
        if let Some(h) = &mut self.history {
            if let Err(e) =
                h.append_observation(result.exit_code, result.timed_out, result.truncated)
            {
                eprintln!("kotonia-cli: history append failed: {e}");
            }
        }
    }

    fn history_turn_start(&mut self) {
        if let Some(h) = &mut self.history {
            let _ = h.append_turn_start();
        }
    }

    fn history_turn_end(&mut self, iterations: u32, success: bool) {
        if let Some(h) = &mut self.history {
            let _ = h.append_turn_end(iterations, success);
        }
    }

    /// Whether the agent is driving the model via native `tools` calling.
    /// Exposed for the CLI banner so the operator can confirm which loop
    /// is active.
    pub fn native_mode(&self) -> bool {
        self.native_mode
    }

    /// One user turn: append the task, loop the LLM/tool exchange until the
    /// model returns a final answer or hits the iteration cap. Dispatches to
    /// the native tool-calling path when the backend supports it; otherwise
    /// runs the legacy `<<<BASH>>>` delimiter loop.
    pub async fn run_turn(
        &mut self,
        task: &str,
        approval: &mut dyn ApprovalHandler,
        sink: &mut dyn EventSink,
    ) -> Result<String, AgentError> {
        if self.native_mode {
            self.run_turn_native(task, approval, sink).await
        } else {
            self.run_turn_delimiter(task, approval, sink).await
        }
    }

    async fn run_turn_delimiter(
        &mut self,
        task: &str,
        approval: &mut dyn ApprovalHandler,
        sink: &mut dyn EventSink,
    ) -> Result<String, AgentError> {
        self.history_turn_start();
        self.cancel.store(false, Ordering::SeqCst);
        self.push_logged(ChatMsg::user(task.to_string()));

        for iter in 1..=self.config.max_iterations {
            if let Some(e) = self.check_cancelled(iter, sink) {
                return Err(e);
            }
            sink.emit(Event::IterationStart {
                iteration: iter,
                max: self.config.max_iterations,
            });

            sink.emit(Event::LlmThinking);
            let assistant_text = match self
                .provider
                .complete(self.messages.clone(), self.config.max_tokens)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    let msg = e.to_string();
                    sink.emit(Event::Error {
                        message: msg.clone(),
                    });
                    self.history_turn_end(iter, false);
                    return Err(AgentError::Llm(msg));
                }
            };
            self.push_logged(ChatMsg::assistant(assistant_text.clone()));

            match parse(&assistant_text) {
                Action::Final(answer) => {
                    sink.emit(Event::Final {
                        answer: answer.clone(),
                    });
                    sink.emit(Event::Done {
                        iterations: iter,
                        success: true,
                    });
                    self.history_turn_end(iter, true);
                    return Ok(answer);
                }
                Action::Bash(command) => {
                    let decision = decide(self.config.approval, &command);
                    let allow = match decision {
                        Decision::Allow => true,
                        Decision::AskUser { reason } => {
                            let outcome = approval.ask(&command, &reason);
                            if outcome == ApprovalOutcome::Approve {
                                true
                            } else {
                                sink.emit(Event::BashSkipped {
                                    command: command.clone(),
                                    reason: format!("operator denied ({reason})"),
                                });
                                self.push_logged(ChatMsg::user(format!(
                                    "Operator DENIED that command (reason: {reason}). \
                                     Try a different approach or ask in a FINAL_ANSWER \
                                     why you're stuck."
                                )));
                                false
                            }
                        }
                    };
                    if allow {
                        sink.emit(Event::Bash {
                            command: command.clone(),
                        });
                        let result = match self.executor.bash(&command).await {
                            Ok(r) => r,
                            Err(e) => {
                                let msg = e.to_string();
                                sink.emit(Event::Error {
                                    message: msg.clone(),
                                });
                                self.history_turn_end(iter, false);
                                return Err(AgentError::Executor(msg));
                            }
                        };
                        sink.emit(Event::Observation {
                            result: result.clone(),
                        });
                        self.history_append_observation(&result);
                        self.push_logged(ChatMsg::user(result.as_observation()));
                    }
                }
                Action::Malformed { excerpt } => {
                    sink.emit(Event::Malformed {
                        excerpt: excerpt.clone(),
                    });
                    self.push_logged(ChatMsg::user(
                        "Your previous response did not contain a <<<BASH>>> or \
                         <<<FINAL_ANSWER>>> block. Re-read the response format and \
                         emit exactly one block, then stop."
                            .to_string(),
                    ));
                }
            }
        }

        sink.emit(Event::Done {
            iterations: self.config.max_iterations,
            success: false,
        });
        self.history_turn_end(self.config.max_iterations, false);
        Err(AgentError::IterationLimit(self.config.max_iterations))
    }

    fn push_logged(&mut self, msg: ChatMsg) {
        self.history_append(&msg);
        self.messages.push(msg);
    }

    /// If a cancellation was requested, emit a terminal `Done{success:false}`,
    /// close the history turn, and return the error to bail the loop. `iter`
    /// is the iteration we were about to start, so completed iterations is
    /// `iter - 1`. Returns `None` when no cancellation is pending.
    fn check_cancelled(&mut self, iter: u32, sink: &mut dyn EventSink) -> Option<AgentError> {
        if !self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        let done = iter.saturating_sub(1);
        sink.emit(Event::Done {
            iterations: done,
            success: false,
        });
        self.history_turn_end(done, false);
        Some(AgentError::Cancelled)
    }

    async fn run_turn_native(
        &mut self,
        task: &str,
        approval: &mut dyn ApprovalHandler,
        sink: &mut dyn EventSink,
    ) -> Result<String, AgentError> {
        self.history_turn_start();
        self.cancel.store(false, Ordering::SeqCst);
        self.hydrate_native_from_messages();
        self.native_messages
            .push(AiMessage::user_text(task.to_string()));
        // Keep the on-disk transcript readable for resume: every native
        // turn writes the equivalent ChatMsg so `--list-sessions` /
        // `--resume` continue to work without a separate native log
        // format.
        let user_log = ChatMsg::user(task.to_string());
        self.history_append(&user_log);
        self.messages.push(user_log);

        for iter in 1..=self.config.max_iterations {
            if let Some(e) = self.check_cancelled(iter, sink) {
                return Err(e);
            }
            sink.emit(Event::IterationStart {
                iteration: iter,
                max: self.config.max_iterations,
            });
            sink.emit(Event::LlmThinking);

            let options = AiCallOptions::new(
                self.provider.model_id().to_string(),
                self.config.max_tokens,
            )
            .with_system(self.system_prompt.clone())
            .with_tools(self.tool_catalog.clone());

            let response = match self
                .provider
                .complete_with_tools(self.native_messages.clone(), options)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    sink.emit(Event::Error {
                        message: msg.clone(),
                    });
                    self.history_turn_end(iter, false);
                    return Err(AgentError::Llm(msg));
                }
            };

            // Record the assistant turn in both the native log (full
            // content blocks) and the legacy ChatMsg log (text summary)
            // so resume keeps working.
            self.native_messages.push(AiMessage {
                role: AiRole::Assistant,
                content: response.content.clone(),
            });
            let assistant_log = ChatMsg::assistant(summarise_assistant_response(&response.content));
            self.history_append(&assistant_log);
            self.messages.push(assistant_log);

            let tool_uses: Vec<(String, String, serde_json::Value)> = response
                .tool_uses()
                .into_iter()
                .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                .collect();

            if tool_uses.is_empty() {
                // No tool calls — this is the final answer. Accept any
                // stop reason here: MaxTokens still gets the text we
                // have, EndTurn is the happy path.
                let answer = response.text();
                if let AiStopReason::MaxTokens = response.stop_reason {
                    sink.emit(Event::Error {
                        message: "model hit max_tokens before issuing tool_call or finishing"
                            .into(),
                    });
                }
                sink.emit(Event::Final {
                    answer: answer.clone(),
                });
                sink.emit(Event::Done {
                    iterations: iter,
                    success: true,
                });
                self.history_turn_end(iter, true);
                return Ok(answer);
            }

            // Dispatch every tool the model asked for, in order. Collect
            // results and feed them back as a single Tool-role message so
            // the model sees the whole batch on the next call.
            let mut tool_result_blocks: Vec<AiContent> = Vec::new();
            for (id, name, input) in tool_uses {
                let result = self
                    .dispatch_tool(&id, &name, &input, approval, sink)
                    .await;
                match result {
                    Ok((content, is_error)) => {
                        // Also mirror the result into the legacy log so a
                        // delimiter-mode resume can see what happened.
                        let legacy_text =
                            format!("[tool {name} (id={id})]\n{}", content.trim_end());
                        let legacy = ChatMsg::user(legacy_text);
                        self.history_append(&legacy);
                        self.messages.push(legacy);
                        tool_result_blocks.push(AiContent::ToolResult {
                            tool_use_id: id,
                            content,
                            is_error,
                        });
                    }
                    Err(e) => {
                        sink.emit(Event::Error {
                            message: e.to_string(),
                        });
                        self.history_turn_end(iter, false);
                        return Err(e);
                    }
                }
            }
            self.native_messages.push(AiMessage {
                role: AiRole::Tool,
                content: tool_result_blocks,
            });
        }

        sink.emit(Event::Done {
            iterations: self.config.max_iterations,
            success: false,
        });
        self.history_turn_end(self.config.max_iterations, false);
        Err(AgentError::IterationLimit(self.config.max_iterations))
    }

    /// On the first native turn (or first turn after a `--resume`), seed
    /// `native_messages` from the legacy text-only `messages`. Skip the
    /// system message — it's already in `self.system_prompt`.
    fn hydrate_native_from_messages(&mut self) {
        if !self.native_messages.is_empty() {
            return;
        }
        for msg in self.messages.iter().skip(1) {
            // skip system at index 0
            let role = match msg.role {
                ChatRole::User => AiRole::User,
                ChatRole::Assistant => AiRole::Assistant,
                ChatRole::System => continue,
            };
            self.native_messages.push(AiMessage {
                role,
                content: vec![AiContent::Text {
                    text: msg.content.clone(),
                }],
            });
        }
    }

    /// Execute a tool call. Returns `(observation_text, is_error)` so the
    /// native loop can package the result as a `ToolResult` block.
    ///
    /// Error semantics: agent-loop errors (executor crash, missing tool)
    /// bubble as `AgentError`; tool-level errors (non-zero exit, denied by
    /// approval) come back as Ok with `is_error=true` so the model can see
    /// them and adapt.
    async fn dispatch_tool(
        &mut self,
        _id: &str,
        name: &str,
        input: &serde_json::Value,
        approval: &mut dyn ApprovalHandler,
        sink: &mut dyn EventSink,
    ) -> Result<(String, bool), AgentError> {
        match name {
            "bash" => {
                let command = input
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if command.trim().is_empty() {
                    return Ok(("bash tool called with empty `command`".into(), true));
                }
                let decision = decide(self.config.approval, &command);
                let allow = match decision {
                    Decision::Allow => true,
                    Decision::AskUser { reason } => {
                        let outcome = approval.ask(&command, &reason);
                        if outcome == ApprovalOutcome::Approve {
                            true
                        } else {
                            sink.emit(Event::BashSkipped {
                                command: command.clone(),
                                reason: format!("operator denied ({reason})"),
                            });
                            return Ok((
                                format!(
                                    "Operator DENIED that command (reason: {reason}). \
                                     Try a different approach or stop with a final answer \
                                     explaining what you're stuck on."
                                ),
                                true,
                            ));
                        }
                    }
                };
                if !allow {
                    unreachable!();
                }
                sink.emit(Event::Bash {
                    command: command.clone(),
                });
                let result = match self.executor.bash(&command).await {
                    Ok(r) => r,
                    Err(e) => return Err(AgentError::Executor(e.to_string())),
                };
                sink.emit(Event::Observation {
                    result: result.clone(),
                });
                self.history_append_observation(&result);
                let is_error = result.exit_code != 0 || result.timed_out;
                Ok((result.as_observation(), is_error))
            }
            "web_search" => {
                let query = input
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if query.trim().is_empty() {
                    return Ok(("web_search called with empty `query`".into(), true));
                }
                let max_results = input
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5)
                    .clamp(1, 30);
                // Shell out to the local `web-search` CLI (searxng wrapper).
                // Quoting via single-quotes + ' → '"'"' escape so an
                // adversarial query can't break out of the argument.
                let escaped = shell_single_quote(&query);
                let command = format!("web-search {escaped} {max_results}");
                sink.emit(Event::Bash {
                    command: command.clone(),
                });
                let result = match self.executor.bash(&command).await {
                    Ok(r) => r,
                    Err(e) => return Err(AgentError::Executor(e.to_string())),
                };
                sink.emit(Event::Observation {
                    result: result.clone(),
                });
                self.history_append_observation(&result);
                let is_error = result.exit_code != 0 || result.timed_out;
                Ok((result.as_observation(), is_error))
            }
            "fetch_url" => {
                let url = input
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if url.trim().is_empty() {
                    return Ok(("fetch_url called with empty `url`".into(), true));
                }
                // Cheap input shape check — refuse anything that's not
                // http(s) so the model can't try to read `file:///etc/passwd`
                // even if the workspace approval would let it.
                if !(url.starts_with("http://") || url.starts_with("https://")) {
                    return Ok((
                        format!("fetch_url: only http(s) URLs are supported, got `{url}`"),
                        true,
                    ));
                }
                let max_chars = input
                    .get("max_chars")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let escaped = shell_single_quote(&url);
                let command = if max_chars > 0 {
                    format!("fetch-url {escaped} {max_chars}")
                } else {
                    format!("fetch-url {escaped}")
                };
                sink.emit(Event::Bash {
                    command: command.clone(),
                });
                let result = match self.executor.bash(&command).await {
                    Ok(r) => r,
                    Err(e) => return Err(AgentError::Executor(e.to_string())),
                };
                sink.emit(Event::Observation {
                    result: result.clone(),
                });
                self.history_append_observation(&result);
                let is_error = result.exit_code != 0 || result.timed_out;
                Ok((result.as_observation(), is_error))
            }
            other => Ok((
                format!(
                    "Unknown tool: `{other}`. Available tools: bash, web_search, fetch_url."
                ),
                true,
            )),
        }
    }
}

/// Wrap an arbitrary string in shell single quotes so it survives `bash -c`
/// intact. The standard POSIX trick: every internal `'` becomes `'\''`.
/// Returns the wrapped form (already includes the surrounding quotes).
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Tool catalog advertised over the OpenAI-compatible `tools` array.
/// Keep the schema inside the OpenAPI 3.0 subset shared by every backend
/// (no `$ref`, no `oneOf`/`anyOf`, no `additionalProperties`).
fn build_tool_catalog() -> Vec<AiTool> {
    vec![
        AiTool {
            name: "bash".to_string(),
            description: "Run a single shell command inside the workspace cwd. \
                          Pipes, redirects, &&, ; are allowed. Returns \
                          combined stdout+stderr and the exit code."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    }
                },
                "required": ["command"]
            }),
        },
        AiTool {
            name: "web_search".to_string(),
            description: "Search the web via the local Searxng instance \
                          (aggregates Google / Bing / DuckDuckGo, no API \
                          key needed). Returns the top results as `title / \
                          URL / snippet (≤240 chars)` rows — this is the \
                          SERP preview only, NOT the article body. Use \
                          `fetch_url` to read the body of a result that \
                          looks promising."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "description": "1..30. Defaults to 5."
                    }
                },
                "required": ["query"]
            }),
        },
        AiTool {
            name: "fetch_url".to_string(),
            description: "Download an http(s) URL and return its main \
                          article body as clean Markdown. Boilerplate \
                          (nav, footer, ads, comments) is stripped via \
                          trafilatura. JS-only SPAs and paywalled pages \
                          may fail with `no extractable content` — try a \
                          different result in that case. Pass `max_chars` \
                          to truncate long articles before they fill the \
                          context window."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http(s) URL to fetch."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional. Cut the output at this \
                                        many code points (UTF-8 safe). \
                                        Omit or pass 0 for the full body."
                    }
                },
                "required": ["url"]
            }),
        },
    ]
}

/// Best-effort one-line summary of an assistant turn for the legacy log.
/// The native log (`native_messages`) preserves full content; this is just
/// so `--resume` produces something readable when loaded by a delimiter
/// session.
fn summarise_assistant_response(blocks: &[AiContent]) -> String {
    let mut parts = Vec::new();
    for b in blocks {
        match b {
            AiContent::Text { text } => {
                if !text.trim().is_empty() {
                    parts.push(text.clone());
                }
            }
            AiContent::ToolUse { name, input, .. } => {
                parts.push(format!(
                    "[tool_call {name}({})]",
                    serde_json::to_string(input).unwrap_or_else(|_| "?".into())
                ));
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        "[empty assistant turn]".into()
    } else {
        parts.join("\n")
    }
}

#[derive(Debug)]
pub enum AgentError {
    Llm(String),
    Executor(String),
    IterationLimit(u32),
    /// A frontend requested cancellation and the loop stopped at an
    /// iteration boundary. A terminal `Done{success:false}` was already
    /// emitted before this bubbles.
    Cancelled,
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Llm(msg) => write!(f, "LLM error: {msg}"),
            AgentError::Executor(msg) => write!(f, "executor error: {msg}"),
            AgentError::IterationLimit(n) => {
                write!(f, "agent hit iteration cap ({n}) without final answer")
            }
            AgentError::Cancelled => write!(f, "turn cancelled by operator"),
        }
    }
}

impl std::error::Error for AgentError {}
