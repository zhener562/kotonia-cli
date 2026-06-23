//! `kotonia-cli daemon` — long-lived WS client that connects to a
//! kotonia.ai backend, waits for [`ServerMsg::RunTask`] commands, and
//! drives [`crate::agent::agent::Agent`] for each task.
//!
//! Per task we:
//! 1. Create a fresh `AgentWorkspace::create_worktree` off the current cwd
//!    (so the operator's working copy is untouched until they merge).
//! 2. Build an `Agent` with two custom impls:
//!    - [`WsEventSink`] — forwards every `Event` as a [`DeviceMsg::AgentEvent`]
//!    - [`WsApprovalHandler`] — emits [`DeviceMsg::ApprovalRequest`] then
//!      blocks until the WS reader observes a matching
//!      [`ServerMsg::ApprovalResult`]
//! 3. Run `agent.run_turn`, then clean up the worktree.
//!
//! The sync `ApprovalHandler::ask` is bridged to the async WS reader via
//! `tokio::task::block_in_place` + `std::sync::mpsc`. This requires the
//! multi-thread runtime (`#[tokio::main]` default).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::task::block_in_place;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    http::{header::AUTHORIZATION, HeaderValue},
    Message as WsMessage,
};

use crate::agent::agent::{
    Agent, AgentConfig, ApprovalHandler, ApprovalOutcome, Event, EventSink,
};
use crate::agent::approval::ApprovalMode;
use crate::agent::provider::Provider;
use crate::agent::wire::{DeviceMsg, ServerMsg, WireEvent};
use crate::agent::worktree::AgentWorkspace;

const RECONNECT_DELAY: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// HTTP(S) base URL of the kotonia.ai backend.
    pub server: String,
    pub device_id: String,
    pub device_token: String,
    /// Model id passed through to [`Provider::for_model`] when a RunTask
    /// arrives. Default matches the one-shot CLI's `--model` default.
    pub model: String,
    /// Approval policy applied to every agent task this daemon runs.
    pub approval: ApprovalMode,
    /// If true, agent runs in the operator's cwd; otherwise a fresh
    /// git worktree under /tmp/ is created per task (default).
    pub in_place: bool,
}

/// Map of `approval_id → sync sender`. Inserted by [`WsApprovalHandler::ask`]
/// before it blocks; drained by the WS reader when an `ApprovalResult`
/// arrives. `std::sync::Mutex` (not tokio) because the handler is sync.
type PendingApprovals = Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<bool>>>>;

/// Drive the daemon forever, reconnecting on transient errors. Returns
/// only on unrecoverable config errors.
pub async fn run(config: DaemonConfig) -> Result<(), String> {
    let ws_url = http_to_ws_url(&config.server, &config.device_id)
        .ok_or_else(|| format!("invalid server URL: {}", config.server))?;
    let bearer = format!("Bearer {}", config.device_token);
    HeaderValue::from_str(&bearer).map_err(|e| format!("invalid bearer token: {e}"))?;

    eprintln!(
        "[daemon] connecting to {ws_url}\n  device={}\n  model={}\n  approval={}\n  in_place={}",
        config.device_id, config.model, config.approval, config.in_place
    );

    loop {
        match connect_and_pump(&ws_url, &bearer, &config).await {
            Ok(()) => {
                eprintln!("[daemon] connection closed cleanly, reconnecting in 5s");
            }
            Err(e) => {
                eprintln!("[daemon] connection error: {e}; reconnecting in 5s");
            }
        }
        sleep(RECONNECT_DELAY).await;
    }
}

async fn connect_and_pump(
    ws_url: &str,
    bearer: &str,
    config: &DaemonConfig,
) -> Result<(), String> {
    let mut request = ws_url
        .into_client_request()
        .map_err(|e| format!("ws request build: {e}"))?;
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(bearer).map_err(|e| format!("bearer header: {e}"))?,
    );

    let (ws_stream, _resp) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| format!("ws connect: {e}"))?;
    eprintln!("[daemon] connected");

    let (mut write, mut read) = ws_stream.split();

    // Per-connection outbound channel: agent task and reader both push
    // DeviceMsg here; one writer task drains.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<DeviceMsg>();
    let pending: PendingApprovals = Arc::new(Mutex::new(HashMap::new()));

    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            let json = match serde_json::to_string(&msg) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("[daemon] skip non-serializable DeviceMsg: {e}");
                    continue;
                }
            };
            if write.send(WsMessage::Text(json.into())).await.is_err() {
                break;
            }
        }
        let _ = write.send(WsMessage::Close(None)).await;
    });

    while let Some(frame) = read.next().await {
        let msg = frame.map_err(|e| format!("ws read: {e}"))?;
        let text = match msg {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Binary(_) | WsMessage::Frame(_) => {
                // tokio-tungstenite auto-replies to protocol-level Pings
                // for us. The application-level keepalive uses
                // ServerMsg::Ping / DeviceMsg::Pong over JSON frames
                // (handled above as Text).
                continue;
            }
        };

        let server_msg: ServerMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[daemon] dropped malformed frame: {e}");
                continue;
            }
        };

        match server_msg {
            ServerMsg::Ping => {
                let _ = out_tx.send(DeviceMsg::Pong);
            }
            ServerMsg::ApprovalResult {
                approval_id,
                approved,
            } => {
                if let Some(tx) = pending.lock().unwrap().remove(&approval_id) {
                    let _ = tx.send(approved);
                } else {
                    eprintln!(
                        "[daemon] stale ApprovalResult for {approval_id} (no pending waiter)"
                    );
                }
            }
            ServerMsg::RunTask { task_id, prompt } => {
                let pending = pending.clone();
                let out_tx = out_tx.clone();
                let config = config.clone();
                tokio::spawn(async move {
                    run_agent_task(task_id, prompt, config, out_tx, pending).await;
                });
            }
        }
    }

    // Reader exited → close out_tx so the writer task can flush + exit.
    drop(out_tx);
    let _ = writer.await;
    Ok(())
}

async fn run_agent_task(
    task_id: String,
    prompt: String,
    config: DaemonConfig,
    out_tx: mpsc::UnboundedSender<DeviceMsg>,
    pending: PendingApprovals,
) {
    // Provider build is the cheap, sync part — if it fails the operator's
    // local model server is missing / mistyped, so surface that and bail.
    let provider = match Provider::for_model(&config.model) {
        Ok(p) => p,
        Err(e) => {
            emit_error(&out_tx, &task_id, format!("provider `{}`: {e}", config.model));
            return;
        }
    };

    let launch_cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            emit_error(&out_tx, &task_id, format!("read cwd: {e}"));
            return;
        }
    };

    let workspace = if config.in_place {
        AgentWorkspace::in_place(launch_cwd)
    } else {
        match AgentWorkspace::create_worktree(&launch_cwd, None).await {
            Ok(w) => w,
            Err(e) => {
                emit_error(&out_tx, &task_id, format!("worktree create: {e}"));
                return;
            }
        }
    };

    let agent_config = AgentConfig::new(config.approval, config.in_place);
    let mut agent = Agent::new(&workspace.root, provider, agent_config);

    let mut sink = WsEventSink {
        task_id: task_id.clone(),
        out_tx: out_tx.clone(),
    };
    let mut approval = WsApprovalHandler {
        task_id: task_id.clone(),
        out_tx: out_tx.clone(),
        pending,
    };

    // Agent emits Final + Done events via the sink, so we don't need to
    // wrap the result here. We do log the AgentError variant for the
    // daemon's stderr / journalctl trail.
    if let Err(e) = agent.run_turn(&prompt, &mut approval, &mut sink).await {
        emit_error(&out_tx, &task_id, format!("agent: {e}"));
    }

    if let Err(e) = workspace.cleanup(false).await {
        eprintln!("[daemon] worktree cleanup for task {task_id} failed: {e}");
    }
}

fn emit_error(out_tx: &mpsc::UnboundedSender<DeviceMsg>, task_id: &str, message: String) {
    eprintln!("[daemon] task {task_id} error: {message}");
    let _ = out_tx.send(DeviceMsg::AgentEvent {
        task_id: task_id.to_string(),
        event: WireEvent::Error { message },
    });
}

// ── Sink + ApprovalHandler impls ────────────────────────────────────────

struct WsEventSink {
    task_id: String,
    out_tx: mpsc::UnboundedSender<DeviceMsg>,
}

impl EventSink for WsEventSink {
    fn emit(&mut self, event: Event) {
        let _ = self.out_tx.send(DeviceMsg::AgentEvent {
            task_id: self.task_id.clone(),
            event: WireEvent::from_event(event),
        });
    }
}

struct WsApprovalHandler {
    task_id: String,
    out_tx: mpsc::UnboundedSender<DeviceMsg>,
    pending: PendingApprovals,
}

impl ApprovalHandler for WsApprovalHandler {
    fn ask(&mut self, command: &str, reason: &str) -> ApprovalOutcome {
        let approval_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = std::sync::mpsc::channel::<bool>();

        // Register the waiter BEFORE sending the request — otherwise the
        // reader could race in with the result and drop it on the floor.
        self.pending.lock().unwrap().insert(approval_id.clone(), tx);

        let send_ok = self
            .out_tx
            .send(DeviceMsg::ApprovalRequest {
                approval_id: approval_id.clone(),
                task_id: self.task_id.clone(),
                command: command.to_string(),
                reason: reason.to_string(),
            })
            .is_ok();
        if !send_ok {
            // WS gone — bail safe (deny). Also clean up the pending entry
            // so it doesn't leak.
            self.pending.lock().unwrap().remove(&approval_id);
            return ApprovalOutcome::Deny;
        }

        // Block this worker thread (block_in_place lets other tasks keep
        // running on other workers) until the reader hands us the answer.
        let approved = block_in_place(|| rx.recv().unwrap_or(false));
        if approved {
            ApprovalOutcome::Approve
        } else {
            ApprovalOutcome::Deny
        }
    }
}

// ── URL helper ───────────────────────────────────────────────────────────

fn http_to_ws_url(server: &str, device_id: &str) -> Option<String> {
    let server = server.trim_end_matches('/');
    let (scheme, rest) = if let Some(rest) = server.strip_prefix("https://") {
        ("wss", rest)
    } else if let Some(rest) = server.strip_prefix("http://") {
        ("ws", rest)
    } else {
        return None;
    };
    Some(format!("{scheme}://{rest}/api/agent-runtime/ws/{device_id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_to_ws_handles_both_schemes() {
        assert_eq!(
            http_to_ws_url("https://kotonia.ai", "dev1"),
            Some("wss://kotonia.ai/api/agent-runtime/ws/dev1".to_string())
        );
        assert_eq!(
            http_to_ws_url("http://localhost:8001/", "dev1"),
            Some("ws://localhost:8001/api/agent-runtime/ws/dev1".to_string())
        );
        assert_eq!(http_to_ws_url("kotonia.ai", "dev1"), None);
    }
}
