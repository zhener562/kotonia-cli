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
            Event::Bash { command } => Self::Bash { command },
            Event::BashSkipped { command, reason } => Self::BashSkipped { command, reason },
            Event::Observation { result } => Self::Observation {
                exit_code: result.exit_code,
                timed_out: result.timed_out,
                truncated: result.truncated,
                combined: result.combined,
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
