//! Two tiny events per agent run — `start` and `end` — sent to kotonia.ai so
//! we can see where people get stuck: a run that fails before its first model
//! call (no `claude` binary, no API key, worktree trouble) never reaches the
//! server otherwise.
//!
//! What is sent: a random per-run id, the engine (`react` / `claude-code`),
//! whether the model is kotonia's or another provider's, how the run was
//! driven (one-shot / REPL / pipe / serve), the outcome, a fixed failure label
//! and the duration. The `x-kotonia-client` header adds version, OS and WSL.
//! **Never** prompts, replies, file paths, model output or error text.
//!
//! Only when logged in to kotonia (a token is present), only to the kotonia
//! server, and never when `DO_NOT_TRACK=1` or `KOTONIA_TELEMETRY=0`.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::agent::dispatch::DispatchError;
use crate::client_info;

const TIMEOUT: Duration = Duration::from_secs(2);

/// One agent run. The CLI has one per process (the free functions below);
/// embedders such as kotonia-desktop make one per task with [`Run::begin`].
pub struct Run {
    id: String,
    started: Instant,
    base: String,
    token: String,
    engine: String,
    provider: &'static str,
    mode: Mutex<&'static str>,
    fail: Mutex<Option<&'static str>>,
    /// `start` runs in the background; `finish` waits for it so a run that
    /// fails at once does not exit before `start` is out.
    start_send: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

static RUN: OnceLock<Run> = OnceLock::new();

/// Opted out via `DO_NOT_TRACK` / `KOTONIA_TELEMETRY`.
pub fn opted_out() -> bool {
    let set = |name: &str| {
        std::env::var(name)
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_default()
    };
    let dnt = set("DO_NOT_TRACK");
    let knob = set("KOTONIA_TELEMETRY");
    (!dnt.is_empty() && dnt != "0" && dnt != "false")
        || matches!(knob.as_str(), "0" | "false" | "off" | "no")
}

/// Begin a run and send `start` in the background. No-op when opted out or
/// not logged in to kotonia.
pub fn start(engine: &str, model: &str) {
    let Some(run) = Run::new(engine, model) else {
        return;
    };
    if RUN.set(run).is_err() {
        return;
    }
    if let Some(run) = RUN.get() {
        let body = run.payload("start", None);
        let handle = tokio::spawn(async move {
            send_to(body).await;
        });
        *run.start_send.lock().unwrap() = Some(handle);
    }
}

impl Run {
    /// None when opted out or not logged in to kotonia.
    fn new(engine: &str, model: &str) -> Option<Self> {
        if opted_out() {
            return None;
        }
        let token = std::env::var("KOTONIA_API_KEY")
            .ok()
            .filter(|t| !t.trim().is_empty())?;
        let base = std::env::var("KOTONIA_API_BASE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://kotonia.ai".to_string());
        Some(Run {
            id: uuid::Uuid::new_v4().to_string(),
            started: Instant::now(),
            base: base.trim_end_matches('/').to_string(),
            token,
            engine: label(engine).to_string(),
            provider: if engine == "claude-code" {
                "claude"
            } else if model.starts_with("kotonia") {
                "kotonia"
            } else {
                "other"
            },
            mode: Mutex::new("unknown"),
            fail: Mutex::new(None),
            start_send: Mutex::new(None),
        })
    }

    /// Begin a run for an embedder and send `start` in the background.
    /// `mode` is a fixed label such as `desktop`.
    pub fn begin(engine: &str, model: &str, mode: &'static str) -> Option<std::sync::Arc<Self>> {
        let run = std::sync::Arc::new(Self::new(engine, model)?);
        *run.mode.lock().unwrap() = mode;
        let body = run.payload("start", None);
        let (base, token) = (run.base.clone(), run.token.clone());
        let handle = tokio::spawn(async move {
            post(&base, &token, body).await;
        });
        *run.start_send.lock().unwrap() = Some(handle);
        Some(run)
    }

    /// Record why the run failed. The first label wins.
    pub fn fail(&self, kind: &'static str) {
        let mut fail = self.fail.lock().unwrap();
        if fail.is_none() {
            *fail = Some(kind);
        }
    }

    /// Send `end` (bounded by a short timeout), after `start` is out.
    pub async fn finish(&self, ok: bool) {
        let start_send = self.start_send.lock().unwrap().take();
        if let Some(handle) = start_send {
            let _ = tokio::time::timeout(TIMEOUT, handle).await;
        }
        post(&self.base, &self.token, self.payload("end", Some(ok))).await;
    }

    fn payload(&self, phase: &str, ok: Option<bool>) -> serde_json::Value {
        let mut body = json!({
            "phase": phase,
            "run_id": self.id,
            "engine": self.engine,
            "provider": self.provider,
        });
        if let Some(ok) = ok {
            body["mode"] = json!(*self.mode.lock().unwrap());
            body["outcome"] = json!(if ok { "ok" } else { "error" });
            body["fail"] = json!(*self.fail.lock().unwrap());
            body["duration_ms"] = json!(self.started.elapsed().as_millis() as u64);
        }
        body
    }
}

/// How the run is being driven: `oneshot`, `repl`, `pipe` or `serve`.
pub fn set_mode(mode: &'static str) {
    if let Some(run) = RUN.get() {
        *run.mode.lock().unwrap() = mode;
    }
}

/// Record why the run failed. The first label wins (the root cause).
pub fn fail(kind: &'static str) {
    if let Some(run) = RUN.get() {
        run.fail(kind);
    }
}

/// Send `end` and wait for it (bounded by a short timeout).
pub async fn finish(ok: bool) {
    if let Some(run) = RUN.get() {
        run.finish(ok).await;
    }
}

/// Fixed label for an agent error. The error text itself is never sent.
pub fn failure_kind(error: &DispatchError) -> &'static str {
    use crate::agent::agent::AgentError;
    use crate::agent::claude_code::ClaudeCodeError;
    match error {
        DispatchError::ReAct(AgentError::Llm(msg)) => llm_failure_kind(msg),
        DispatchError::ReAct(AgentError::Executor(msg)) if msg.contains("no shell found") => {
            "no_shell"
        }
        DispatchError::ReAct(AgentError::Executor(_)) => "executor",
        DispatchError::ReAct(AgentError::IterationLimit(_)) => "iteration_limit",
        DispatchError::ReAct(AgentError::Cancelled) => "cancelled",
        DispatchError::ClaudeCode(ClaudeCodeError::Spawn(_)) => "claude_spawn",
        DispatchError::ClaudeCode(ClaudeCodeError::Io(_)) => "claude_io",
        DispatchError::ClaudeCode(ClaudeCodeError::NonZeroExit { .. }) => "claude_exit",
        DispatchError::ClaudeCode(ClaudeCodeError::NoFinalResult) => "claude_no_result",
    }
}

fn llm_failure_kind(msg: &str) -> &'static str {
    if msg.contains("has no API key") {
        return "llm_no_key";
    }
    if msg.contains("transport error") {
        return "llm_transport";
    }
    let status = msg
        .find("API error (")
        .and_then(|i| msg.get(i + "API error (".len()..i + "API error (".len() + 3));
    match status {
        Some("401") | Some("403") => "llm_auth",
        Some("402") => "llm_payment",
        Some("429") => "llm_rate_limit",
        Some(s) if s.starts_with('5') => "llm_server",
        Some(s) if s.starts_with('4') => "llm_request",
        _ => "llm_other",
    }
}

fn label(raw: &str) -> &'static str {
    match raw {
        "react" => "react",
        "claude-code" => "claude-code",
        _ => "other",
    }
}

async fn send_to(body: serde_json::Value) {
    if let Some(run) = RUN.get() {
        post(&run.base, &run.token, body).await;
    }
}

async fn post(base: &str, token: &str, body: serde_json::Value) {
    let Ok(client) = reqwest::Client::builder().timeout(TIMEOUT).build() else {
        return;
    };
    let _ = client
        .post(format!("{base}/api/v1/client-events"))
        .bearer_auth(token)
        .header(client_info::HEADER, client_info::header_value())
        .json(&body)
        .send()
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_errors_map_to_fixed_labels() {
        assert_eq!(
            llm_failure_kind("API error (401): unauthorized"),
            "llm_auth"
        );
        assert_eq!(
            llm_failure_kind("API error (402): out of points"),
            "llm_payment"
        );
        assert_eq!(
            llm_failure_kind("API error (429): slow down"),
            "llm_rate_limit"
        );
        assert_eq!(llm_failure_kind("API error (503): busy"), "llm_server");
        assert_eq!(
            llm_failure_kind("transport error: connection refused"),
            "llm_transport"
        );
        assert_eq!(
            llm_failure_kind("provider `x` has no API key. set X"),
            "llm_no_key"
        );
        assert_eq!(llm_failure_kind("something else"), "llm_other");
    }

    #[test]
    fn engine_label_never_echoes_user_input() {
        assert_eq!(label("react"), "react");
        assert_eq!(label("/home/me/secret"), "other");
    }
}
