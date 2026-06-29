//! Wire types for the daemon ⇄ kotonia.ai WS round-trip.
//!
//! Kept in sync (by hand) with `backend/src/handlers/agent_runtime_ws.rs`.
//! The two crates do not share a workspace dep so the type definitions
//! live in both — if you change one, change the other.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServerMsg {
    RunTask {
        task_id: String,
        session_id: String,
        prompt: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Stable per-browser-tab id. Used as the trust key for the
        /// notifier-gated approval flow. `None` means the caller (web UI or
        /// other) didn't supply one, which the gate treats as "untrusted
        /// origin" and requires fresh approval.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        browser_session_id: Option<String>,
        /// Client IP the backend extracted (CF-Connecting-IP only).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin_ip: Option<String>,
        /// First ~120 chars of the User-Agent header.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_agent: Option<String>,
    },
    ApprovalResult {
        approval_id: String,
        approved: bool,
    },
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeviceMsg {
    AgentEvent {
        task_id: String,
        event: WireEvent,
    },
    ApprovalRequest {
        approval_id: String,
        task_id: String,
        command: String,
        reason: String,
    },
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireEvent {
    IterationStart {
        iteration: u32,
        max: u32,
    },
    LlmThinking,
    /// Streaming prose chunk from the model (Claude Code engine today).
    /// Renderers should append, not replace; `Final` will NOT replay the
    /// same content when it's already been streamed via `Text`.
    Text {
        text: String,
    },
    Bash {
        command: String,
    },
    BashSkipped {
        command: String,
        reason: String,
    },
    Observation {
        exit_code: i32,
        timed_out: bool,
        truncated: bool,
        combined: String,
    },
    InspectImage {
        path: String,
        size_bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    Final {
        answer: String,
    },
    Malformed {
        excerpt: String,
    },
    Error {
        message: String,
    },
    Done {
        iterations: u32,
        success: bool,
    },
}

impl WireEvent {
    /// Convert a `super::agent::Event` into a wire-ready form.
    pub fn from_event(event: super::agent::Event) -> Self {
        use super::agent::Event;
        match event {
            Event::IterationStart { iteration, max } => Self::IterationStart { iteration, max },
            Event::LlmThinking => Self::LlmThinking,
            Event::Text { text } => Self::Text { text },
            Event::Bash { command } => Self::Bash { command },
            Event::BashSkipped { command, reason } => Self::BashSkipped { command, reason },
            Event::Observation { result } => Self::Observation {
                exit_code: result.exit_code,
                timed_out: result.timed_out,
                truncated: result.truncated,
                combined: result.combined,
            },
            Event::InspectImage {
                path,
                size_bytes,
                error,
            } => Self::InspectImage {
                path,
                size_bytes,
                error,
            },
            Event::Final { answer } => Self::Final { answer },
            Event::Malformed { excerpt } => Self::Malformed { excerpt },
            Event::Error { message } => Self::Error { message },
            Event::Done {
                iterations,
                success,
            } => Self::Done {
                iterations,
                success,
            },
        }
    }
}
