//! On-disk session log for kotonia-cli.
//!
//! Layout: `~/.kotonia/sessions/<id>.jsonl`. One JSON line per event —
//! session metadata, every ChatMsg pushed into the conversation, and an
//! observation marker after each bash call so post-mortem inspection can
//! see the exit code / truncation state without re-deriving from the
//! user-role text dump.
//!
//! Append-only. Resume reads every `message` line in order and seeds a
//! new Agent's `messages` vec. The system prompt is regenerated for the
//! current workspace at resume time (it's path-dependent), so resuming
//! into a different worktree is supported — the model sees a fresh
//! system prompt + the previous conversation history.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::provider::{ChatMsg, ChatRole};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TranscriptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMetadata {
    pub id: String,
    pub workspace: PathBuf,
    pub in_place: bool,
}

#[derive(Debug)]
pub enum HistoryError {
    Io(std::io::Error),
    Serde(serde_json::Error),
    NoHome,
    SessionNotFound(String),
}

impl std::fmt::Display for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HistoryError::Io(e) => write!(f, "history I/O error: {e}"),
            HistoryError::Serde(e) => write!(f, "history serialization error: {e}"),
            HistoryError::NoHome => write!(f, "cannot locate $HOME for ~/.kotonia/sessions"),
            HistoryError::SessionNotFound(id) => write!(f, "no session log for id `{id}`"),
        }
    }
}

impl std::error::Error for HistoryError {}

impl From<std::io::Error> for HistoryError {
    fn from(e: std::io::Error) -> Self {
        HistoryError::Io(e)
    }
}
impl From<serde_json::Error> for HistoryError {
    fn from(e: serde_json::Error) -> Self {
        HistoryError::Serde(e)
    }
}

/// Append-only writer for one session. Holds the file handle so each
/// `append_*` is one syscall; lines are flushed on drop.
pub struct HistoryStore {
    pub session_id: String,
    pub path: PathBuf,
    writer: File,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LogEntry {
    Session {
        id: String,
        model: String,
        backend: String,
        approval: String,
        workspace: String,
        in_place: bool,
        started_at: String,
    },
    Message {
        role: String,
        content: String,
        ts: String,
    },
    /// A frontend-visible chat bubble. The regular `message` entries remain
    /// the model/resume transcript and can contain tool summaries,
    /// observations, or editor context. UI clients should prefer these rows.
    UiMessage {
        role: String,
        content: String,
        turn_id: u64,
        ts: String,
    },
    Observation {
        exit_code: i32,
        timed_out: bool,
        truncated: bool,
        ts: String,
    },
    TurnStart {
        ts: String,
    },
    TurnEnd {
        iterations: u32,
        success: bool,
        ts: String,
    },
}

impl HistoryStore {
    /// Default session directory (`~/.kotonia/sessions/`).
    pub fn default_dir() -> Result<PathBuf, HistoryError> {
        let home = dirs::home_dir().ok_or(HistoryError::NoHome)?;
        let p = home.join(".kotonia").join("sessions");
        std::fs::create_dir_all(&p)?;
        Ok(p)
    }

    /// Open (creating) the log under the default directory.
    pub fn open(session_id: impl Into<String>) -> Result<Self, HistoryError> {
        Self::open_in(Self::default_dir()?, session_id)
    }

    /// Open under an explicit directory. Used by tests to avoid touching
    /// the operator's real ~/.kotonia.
    pub fn open_in(
        dir: impl AsRef<Path>,
        session_id: impl Into<String>,
    ) -> Result<Self, HistoryError> {
        let session_id = session_id.into();
        std::fs::create_dir_all(dir.as_ref())?;
        let path = dir.as_ref().join(format!("{session_id}.jsonl"));
        let writer = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            session_id,
            path,
            writer,
        })
    }

    /// Write a session-header line. Call once on fresh-session start; safe
    /// to skip on resume (the original header remains at the top of the file).
    pub fn write_header(
        &mut self,
        model: &str,
        backend: &str,
        approval: &str,
        workspace: &Path,
        in_place: bool,
    ) -> Result<(), HistoryError> {
        let entry = LogEntry::Session {
            id: self.session_id.clone(),
            model: model.to_string(),
            backend: backend.to_string(),
            approval: approval.to_string(),
            workspace: workspace.display().to_string(),
            in_place,
            started_at: now_iso(),
        };
        self.write_line(&entry)
    }

    pub fn append_message(&mut self, role: &ChatRole, content: &str) -> Result<(), HistoryError> {
        let role_str = match role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        let entry = LogEntry::Message {
            role: role_str.to_string(),
            content: content.to_string(),
            ts: now_iso(),
        };
        self.write_line(&entry)
    }

    pub fn append_ui_message(
        &mut self,
        role: &str,
        content: &str,
        turn_id: u64,
    ) -> Result<(), HistoryError> {
        let entry = LogEntry::UiMessage {
            role: role.to_string(),
            content: content.to_string(),
            turn_id,
            ts: now_iso(),
        };
        self.write_line(&entry)
    }

    pub fn append_observation(
        &mut self,
        exit_code: i32,
        timed_out: bool,
        truncated: bool,
    ) -> Result<(), HistoryError> {
        let entry = LogEntry::Observation {
            exit_code,
            timed_out,
            truncated,
            ts: now_iso(),
        };
        self.write_line(&entry)
    }

    pub fn append_turn_start(&mut self) -> Result<(), HistoryError> {
        self.write_line(&LogEntry::TurnStart { ts: now_iso() })
    }

    pub fn append_turn_end(&mut self, iterations: u32, success: bool) -> Result<(), HistoryError> {
        self.write_line(&LogEntry::TurnEnd {
            iterations,
            success,
            ts: now_iso(),
        })
    }

    fn write_line(&mut self, entry: &LogEntry) -> Result<(), HistoryError> {
        let line = serde_json::to_string(entry)?;
        writeln!(self.writer, "{line}")?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Read all `message` entries from a session log in arrival order. The
/// system message (if any was logged) is dropped — the caller regenerates
/// a workspace-appropriate system prompt and prepends it.
pub fn load_session_messages(session_id: &str) -> Result<Vec<ChatMsg>, HistoryError> {
    load_session_messages_in(HistoryStore::default_dir()?, session_id)
}

pub fn load_session_messages_in(
    dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<Vec<ChatMsg>, HistoryError> {
    let path = dir.as_ref().join(format!("{session_id}.jsonl"));
    if !path.exists() {
        return Err(HistoryError::SessionNotFound(session_id.to_string()));
    }
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let mut msgs = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LogEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue, // skip malformed lines rather than aborting
        };
        if let LogEntry::Message { role, content, .. } = entry {
            let r = match role.as_str() {
                "system" => continue, // re-derived from workspace
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                _ => continue,
            };
            msgs.push(ChatMsg {
                role: r,
                content,
            });
        }
    }
    Ok(msgs)
}

pub fn load_session_metadata(session_id: &str) -> Result<SessionMetadata, HistoryError> {
    load_session_metadata_in(HistoryStore::default_dir()?, session_id)
}

pub fn load_session_metadata_in(
    dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<SessionMetadata, HistoryError> {
    let path = dir.as_ref().join(format!("{session_id}.jsonl"));
    if !path.exists() {
        return Err(HistoryError::SessionNotFound(session_id.to_string()));
    }
    let file = File::open(&path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let entry: LogEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if let LogEntry::Session {
            id,
            workspace,
            in_place,
            ..
        } = entry
        {
            return Ok(SessionMetadata {
                id,
                workspace: PathBuf::from(workspace),
                in_place,
            });
        }
    }
    Err(HistoryError::SessionNotFound(session_id.to_string()))
}

/// Build the operator-visible chat transcript for history UIs.
///
/// Newer sessions contain explicit `ui_message` rows. Older logs are
/// reconstructed best-effort from the model transcript, filtering tool
/// observations and extracting `final_answer` tool calls. When both formats
/// exist, matching legacy rows are de-duplicated in favour of the explicit UI
/// row while preserving older turns from before protocol v2.
pub fn load_session_transcript(
    session_id: &str,
) -> Result<Vec<TranscriptMessage>, HistoryError> {
    load_session_transcript_in(HistoryStore::default_dir()?, session_id)
}

pub fn load_session_transcript_in(
    dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<Vec<TranscriptMessage>, HistoryError> {
    let path = dir.as_ref().join(format!("{session_id}.jsonl"));
    if !path.exists() {
        return Err(HistoryError::SessionNotFound(session_id.to_string()));
    }
    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    #[derive(Clone)]
    struct Candidate {
        line: usize,
        explicit_ui: bool,
        message: TranscriptMessage,
        removed: bool,
    }

    let mut candidates = Vec::<Candidate>::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LogEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let message = match entry {
            LogEntry::UiMessage { role, content, .. }
                if (role == "user" || role == "assistant") && !content.trim().is_empty() =>
            {
                Some((true, TranscriptMessage { role, content }))
            }
            LogEntry::Message { role, content, .. } if role == "user" => {
                legacy_user_message(&content).map(|content| {
                    (
                        false,
                        TranscriptMessage {
                            role: "user".to_string(),
                            content,
                        },
                    )
                })
            }
            LogEntry::Message { role, content, .. } if role == "assistant" => {
                legacy_assistant_message(&content).map(|content| {
                    (
                        false,
                        TranscriptMessage {
                            role: "assistant".to_string(),
                            content,
                        },
                    )
                })
            }
            _ => None,
        };
        if let Some((explicit_ui, message)) = message {
            candidates.push(Candidate {
                line: line_index,
                explicit_ui,
                message,
                removed: false,
            });
        }
    }

    // The model transcript and the explicit UI transcript are written near
    // each other but not in a fixed order: user UI precedes `run_turn`, while
    // assistant UI follows it. Match by role/content and remove the nearest
    // legacy duplicate.
    let ui_indexes: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.explicit_ui.then_some(i))
        .collect();
    for ui_index in ui_indexes {
        let ui = &candidates[ui_index];
        let best = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                !c.explicit_ui
                    && !c.removed
                    && c.message.role == ui.message.role
                    && c.message.content == ui.message.content
            })
            .min_by_key(|(_, c)| c.line.abs_diff(ui.line))
            .map(|(i, _)| i);
        if let Some(i) = best {
            candidates[i].removed = true;
        }
    }

    candidates.sort_by_key(|c| c.line);
    Ok(candidates
        .into_iter()
        .filter(|c| !c.removed)
        .map(|c| c.message)
        .collect())
}

fn legacy_user_message(content: &str) -> Option<String> {
    let mut text = content.trim();
    if let Some((before, _)) = text.split_once("\n\n<!-- KOTONIA_EDITOR_CONTEXT_START -->") {
        text = before.trim();
    }
    if text.is_empty()
        || text.starts_with("[tool ")
        || text.starts_with("[exit ")
        || text.starts_with("Operator DENIED")
        || text.starts_with("Your previous response did not contain")
    {
        return None;
    }

    // Protocol v1 injected language/persona instructions into the first user
    // message. The language instruction was the final prefix immediately
    // before the actual request, so trim through it when recognisable.
    const JA_LANGUAGE: &str =
        "デフォルトでは日本語で回答してください。ユーザーが他の言語で書いた場合は、その言語で回答してください。";
    if let Some(pos) = text.rfind(JA_LANGUAGE) {
        let after = text[pos + JA_LANGUAGE.len()..].trim();
        if !after.is_empty() {
            return Some(after.to_string());
        }
    }
    if let Some(pos) = text.rfind("Reply in \"") {
        let tail = &text[pos..];
        const END: &str =
            "If the user writes in another language, reply in that language instead.";
        if let Some(end) = tail.find(END) {
            let after = tail[end + END.len()..].trim();
            if !after.is_empty() {
                return Some(after.to_string());
            }
        }
    }
    Some(text.to_string())
}

fn legacy_assistant_message(content: &str) -> Option<String> {
    let text = content.trim();
    if text.is_empty() || text == "[empty assistant turn]" {
        return None;
    }

    if let Some(start) = text.find("[tool_call final_answer(") {
        let json_start = start + "[tool_call final_answer(".len();
        let tail = &text[json_start..];
        if let Some(json_end) = tail.find(")]") {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&tail[..json_end]) {
                if let Some(answer) = value.get("answer").and_then(|v| v.as_str()) {
                    if !answer.trim().is_empty() {
                        return Some(answer.trim().to_string());
                    }
                }
            }
        }
    }

    if let Some(start) = text.find("<<<FINAL_ANSWER>>>") {
        let tail = &text[start + "<<<FINAL_ANSWER>>>".len()..];
        let answer = tail
            .split("<<<END>>>")
            .next()
            .unwrap_or(tail)
            .trim();
        if !answer.is_empty() {
            return Some(answer.to_string());
        }
    }

    // Native tool-call summaries are internal reasoning/actions, not chat
    // bubbles. Plain assistant text is retained for delimiter/legacy agents.
    if text.contains("[tool_call ") {
        None
    } else {
        Some(text.to_string())
    }
}

/// List session ids in `~/.kotonia/sessions/`, newest-modified first. Used
/// by `kotonia-cli --list-sessions` (future) and useful for debugging.
pub fn list_sessions() -> Result<Vec<SessionListing>, HistoryError> {
    list_sessions_in(HistoryStore::default_dir()?)
}

pub fn list_sessions_in(dir: impl AsRef<Path>) -> Result<Vec<SessionListing>, HistoryError> {
    let dir = dir.as_ref();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_default();
        let meta = entry.metadata()?;
        let modified = meta.modified()?;
        out.push(SessionListing { id, modified });
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(out)
}

#[derive(Debug)]
pub struct SessionListing {
    pub id: String,
    pub modified: std::time::SystemTime,
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = HistoryStore::open_in(dir.path(), "test-session").unwrap();
        store
            .write_header("gemma4", "local", "auto", Path::new("/tmp/foo"), false)
            .unwrap();
        store.append_message(&ChatRole::User, "こんにちは").unwrap();
        store.append_message(&ChatRole::Assistant, "hello back").unwrap();
        store.append_observation(0, false, false).unwrap();
        store.append_message(&ChatRole::User, "[exit 0]").unwrap();

        let msgs = load_session_messages_in(dir.path(), "test-session").unwrap();
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0].role, ChatRole::User));
        assert_eq!(msgs[0].content, "こんにちは");
        assert!(matches!(msgs[1].role, ChatRole::Assistant));
        assert_eq!(msgs[1].content, "hello back");
        assert!(matches!(msgs[2].role, ChatRole::User));
        assert!(msgs[2].content.contains("exit 0"));
    }

    #[test]
    fn load_missing_session_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_session_messages_in(dir.path(), "no-such-id").unwrap_err();
        assert!(matches!(err, HistoryError::SessionNotFound(_)));
    }

    #[test]
    fn list_sessions_sorts_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        for id in &["a", "b", "c"] {
            let mut s = HistoryStore::open_in(dir.path(), *id).unwrap();
            s.append_message(&ChatRole::User, id).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let listed = list_sessions_in(dir.path()).unwrap();
        assert_eq!(listed.len(), 3);
        assert_eq!(listed[0].id, "c");
        assert_eq!(listed[2].id, "a");
    }

    #[test]
    fn transcript_prefers_ui_rows_and_filters_tool_noise() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = HistoryStore::open_in(dir.path(), "ui-session").unwrap();
        store.append_ui_message("user", "直して", 1).unwrap();
        store
            .append_message(
                &ChatRole::User,
                "直して\n\n<!-- KOTONIA_EDITOR_CONTEXT_START -->\n{\"active_file\":\"src/main.rs\"}",
            )
            .unwrap();
        store
            .append_message(
                &ChatRole::Assistant,
                r#"[tool_call bash({"command":"cargo test"})]"#,
            )
            .unwrap();
        store
            .append_message(
                &ChatRole::Assistant,
                r#"[tool_call final_answer({"answer":"修正したよ"})]"#,
            )
            .unwrap();
        store
            .append_ui_message("assistant", "修正したよ", 1)
            .unwrap();

        let transcript = load_session_transcript_in(dir.path(), "ui-session").unwrap();
        assert_eq!(
            transcript,
            vec![
                TranscriptMessage {
                    role: "user".into(),
                    content: "直して".into(),
                },
                TranscriptMessage {
                    role: "assistant".into(),
                    content: "修正したよ".into(),
                },
            ]
        );
    }

    #[test]
    fn transcript_recovers_protocol_v1_first_user_message() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = HistoryStore::open_in(dir.path(), "legacy-session").unwrap();
        store
            .append_message(
                &ChatRole::User,
                "キャラ設定\n\nデフォルトでは日本語で回答してください。ユーザーが他の言語で書いた場合は、その言語で回答してください。\n\n本題です",
            )
            .unwrap();
        let transcript = load_session_transcript_in(dir.path(), "legacy-session").unwrap();
        assert_eq!(transcript[0].content, "本題です");
    }

    #[test]
    fn session_metadata_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = HistoryStore::open_in(dir.path(), "meta-session").unwrap();
        store
            .write_header(
                "gemma",
                "kotonia",
                "allowlist",
                Path::new("/tmp/kotonia-agent-abcd"),
                false,
            )
            .unwrap();
        let meta = load_session_metadata_in(dir.path(), "meta-session").unwrap();
        assert_eq!(meta.id, "meta-session");
        assert_eq!(meta.workspace, PathBuf::from("/tmp/kotonia-agent-abcd"));
        assert!(!meta.in_place);
    }
}
