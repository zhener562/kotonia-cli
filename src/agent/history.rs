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
        let home = std::env::var_os("HOME").ok_or(HistoryError::NoHome)?;
        let p = PathBuf::from(home).join(".kotonia").join("sessions");
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
}
