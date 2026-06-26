//! Tiny engine-dispatch layer.
//!
//! `main.rs` and `daemon.rs` both want to drive "the active agent" without
//! caring whether it's a ReAct loop or a Claude Code subprocess. Wrapping
//! the two concrete agents in a small enum keeps both call sites
//! straightforward — no `dyn AgentLike` dance, no async trait.

use crate::agent::agent::{
    Agent, AgentError, ApprovalHandler, Event, EventSink,
};
use crate::agent::claude_code::{ClaudeCodeAgent, ClaudeCodeError};
use crate::agent::history::HistoryStore;
use crate::agent::provider::ChatMsg;

pub enum DispatchAgent {
    ReAct(Agent),
    ClaudeCode(ClaudeCodeAgent),
}

#[derive(Debug)]
pub enum DispatchError {
    ReAct(AgentError),
    ClaudeCode(ClaudeCodeError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::ReAct(e) => write!(f, "{e}"),
            DispatchError::ClaudeCode(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl DispatchAgent {
    pub fn with_history(self, history: HistoryStore) -> Self {
        match self {
            Self::ReAct(a) => Self::ReAct(a.with_history(history)),
            Self::ClaudeCode(c) => Self::ClaudeCode(c.with_history(history)),
        }
    }

    pub fn log_initial_system(&mut self) {
        match self {
            Self::ReAct(a) => a.log_initial_system(),
            Self::ClaudeCode(c) => c.log_initial_system(),
        }
    }

    pub fn seed_messages(&mut self, prior: Vec<ChatMsg>) {
        match self {
            Self::ReAct(a) => a.seed_messages(prior),
            Self::ClaudeCode(c) => c.seed_messages(prior),
        }
    }

    pub fn provider_label(&self) -> String {
        match self {
            Self::ReAct(a) => a.provider_label(),
            Self::ClaudeCode(c) => c.provider_label(),
        }
    }

    pub fn backend_label(&self) -> &str {
        match self {
            Self::ReAct(a) => a.backend_label(),
            Self::ClaudeCode(c) => c.backend_label(),
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::ReAct(a) => a.model_id(),
            Self::ClaudeCode(c) => c.model_id(),
        }
    }

    pub fn native_mode(&self) -> bool {
        match self {
            Self::ReAct(a) => a.native_mode(),
            Self::ClaudeCode(c) => c.native_mode(),
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::ReAct(a) => a.session_id(),
            Self::ClaudeCode(c) => c.session_id(),
        }
    }

    /// Run one user turn. `approval` is ignored by the Claude Code engine —
    /// that subprocess runs with `--dangerously-skip-permissions` so the
    /// host's approval policy doesn't gate it. (Future stages can hand
    /// the prompt through a custom MCP server.)
    pub async fn run_turn(
        &mut self,
        task: &str,
        approval: &mut dyn ApprovalHandler,
        sink: &mut dyn EventSink,
    ) -> Result<String, DispatchError> {
        match self {
            Self::ReAct(a) => a
                .run_turn(task, approval, sink)
                .await
                .map_err(DispatchError::ReAct),
            Self::ClaudeCode(c) => c
                .run_turn(task, sink)
                .await
                .map_err(DispatchError::ClaudeCode),
        }
    }

    /// Emit Claude Code's bypass-permissions banner once at startup so the
    /// operator knows the subprocess is unsandboxed. No-op for ReAct.
    pub fn emit_engine_banner(&self, sink: &mut dyn EventSink) {
        if let Self::ClaudeCode(_) = self {
            sink.emit(Event::IterationStart {
                iteration: 0,
                max: 0,
            });
        }
    }
}
