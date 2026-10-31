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

struct Run {
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
    if opted_out() {
        return;
    }
    let Some(token) = std::env::var("KOTONIA_API_KEY")
        .ok()
        .filter(|t| !t.trim().is_empty())
    else {
        return;
    };
    let base = std::env::var("KOTONIA_API_BASE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://kotonia.ai".to_string());
    let run = Run {
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
    };
    if RUN.set(run).is_err() {
        return;
    }
    let body = payload("start", None);
    let handle = tokio::spawn(async move {
        send(body).await;
    });
    if let Some(run) = RUN.get() {
        *run.start_send.lock().unwrap() = Some(handle);
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
        let mut fail = run.fail.lock().unwrap();
        if fail.is_none() {
            *fail = Some(kind);
        }
    }
}

/// Send `end` and wait for it (bounded by a short timeout).
pub async fn finish(ok: bool) {
    let Some(run) = RUN.get() else {
        return;
    };
    let start_send = run.start_send.lock().unwrap().take();
    if let Some(handle) = start_send {
        let _ = tokio::time::timeout(TIMEOUT, handle).await;
    }
    send(payload("end", Some(ok))).await;
}

/// Fixed label for an agent error. The error text itself is never sent.
pub fn failure_kind(error: &DispatchError) -> &'static str {
    use crate::agent::agent::AgentError;
    use crate::agent::claude_code::ClaudeCodeError;
    match error {
        DispatchError::ReAct(AgentError::Llm(msg)) => llm_failure_kind(msg),
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

fn payload(phase: &str, ok: Option<bool>) -> serde_json::Value {
    let Some(run) = RUN.get() else {
        return json!({});
    };
    let mut body = json!({
        "phase": phase,
        "run_id": run.id,
        "engine": run.engine,
        "provider": run.provider,
    });
    if let Some(ok) = ok {
        body["mode"] = json!(*run.mode.lock().unwrap());
        body["outcome"] = json!(if ok { "ok" } else { "error" });
        body["fail"] = json!(*run.fail.lock().unwrap());
        body["duration_ms"] = json!(run.started.elapsed().as_millis() as u64);
    }
    body
}

async fn send(body: serde_json::Value) {
    let Some(run) = RUN.get() else {
        return;
    };
    let Ok(client) = reqwest::Client::builder().timeout(TIMEOUT).build() else {
        return;
    };
    let _ = client
        .post(format!("{}/api/v1/client-events", run.base))
        .bearer_auth(&run.token)
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
