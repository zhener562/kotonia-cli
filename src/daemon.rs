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
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex, RwLock};
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
use crate::agent::claude_code::ClaudeCodeAgent;
use crate::agent::dispatch::DispatchAgent;
use crate::agent::history::{load_session_messages, HistoryStore};
use crate::agent::provider::Provider;
use crate::agent::wire::{DeviceMsg, ServerMsg, WireEvent};
use crate::agent::worktree::AgentWorkspace;

const RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Sessions idle longer than this are GC'd: their worktree is removed and
/// their conversation history is forgotten. 30 min matches a typical
/// "stepped away from the chat" interval.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// How often the GC task sweeps idle sessions.
const SESSION_GC_INTERVAL: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// HTTP(S) base URL of the kotonia.ai backend.
    pub server: String,
    pub device_id: String,
    pub device_token: String,
    /// Model id passed through to [`Provider::resolve`] when a RunTask
    /// arrives. Default matches the one-shot CLI's `--model` default.
    pub model: String,
    /// Optional provider override; None means infer from model id.
    pub provider: Option<String>,
    /// Agent engine string: `"react"` or `"claude-code"`. Same surface as
    /// the one-shot CLI's `--engine` — `"claude-code"` makes the daemon
    /// drive the local `claude` binary as a subprocess per task.
    pub engine: String,
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

/// One running multi-turn session. Holds the live agent (whose internal
/// state accumulates across `run_turn` calls) and the worktree it operates
/// in. `DispatchAgent` is an engine-agnostic envelope around either a
/// kotonia ReAct loop or a Claude Code subprocess.
struct SessionState {
    agent: DispatchAgent,
    workspace: AgentWorkspace,
    last_active: Instant,
}

/// Per-daemon-process session map. Survives WS reconnects so a flaky
/// network doesn't lose conversation context. New sessions get a fresh
/// worktree + provider; existing ones reuse them so follow-up tasks
/// thread the prior context.
struct SessionRegistry {
    inner: RwLock<HashMap<String, Arc<AsyncMutex<SessionState>>>>,
}

impl SessionRegistry {
    fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    /// Look up or build a session. Returns the per-session async mutex so
    /// the caller can lock it for the duration of `run_turn`.
    ///
    /// `model_override` lets the operator pick a different provider for a
    /// *new* session via `/model` in the web console. Ignored on existing
    /// sessions (the console regenerates session_id so this lands fresh).
    /// Auto-resumes if a JSONL log already exists for the same session_id
    /// in ~/.kotonia/sessions — operator can /resume across daemon restarts.
    async fn get_or_create(
        &self,
        session_id: &str,
        config: &DaemonConfig,
        model_override: Option<&str>,
    ) -> Result<Arc<AsyncMutex<SessionState>>, String> {
        {
            let map = self.inner.read().await;
            if let Some(s) = map.get(session_id) {
                return Ok(s.clone());
            }
        }
        let model = model_override.unwrap_or(&config.model);
        let use_claude_code = config.engine == "claude-code" || model == "claude-code";
        let launch_cwd =
            std::env::current_dir().map_err(|e| format!("read cwd: {e}"))?;
        let workspace = if config.in_place {
            AgentWorkspace::in_place(launch_cwd)
        } else {
            AgentWorkspace::create_worktree(&launch_cwd, None)
                .await
                .map_err(|e| format!("worktree create: {e}"))?
        };
        let mut agent = if use_claude_code {
            let cc_session_id =
                crate::agent::claude_code::claude_code_session_id(session_id);
            DispatchAgent::ClaudeCode(ClaudeCodeAgent::new(
                &workspace.root,
                cc_session_id,
                config.in_place,
            ))
        } else {
            let provider = Provider::resolve(config.provider.as_deref(), model)
                .map_err(|e| format!("provider `{model}`: {e}"))?;
            let agent_config = AgentConfig::new(config.approval, config.in_place);
            DispatchAgent::ReAct(Agent::new(&workspace.root, provider, agent_config))
        };

        // Attach history persistence + auto-resume. The HistoryStore opens
        // (or creates) ~/.kotonia/sessions/{session_id}.jsonl. If prior
        // messages exist on disk, we seed the agent so /resume works
        // transparently across daemon restarts.
        match HistoryStore::open(session_id) {
            Ok(mut store) => {
                let prior = load_session_messages(session_id).unwrap_or_default();
                let resuming = !prior.is_empty();
                if !resuming {
                    let _ = store.write_header(
                        agent.model_id(),
                        agent.backend_label(),
                        &config.approval.to_string(),
                        &workspace.root,
                        config.in_place,
                    );
                }
                agent = agent.with_history(store);
                if resuming {
                    eprintln!(
                        "[daemon] session {} resumed from disk ({} prior msgs)",
                        short(session_id),
                        prior.len()
                    );
                    agent.seed_messages(prior);
                } else {
                    agent.log_initial_system();
                }
            }
            Err(e) => {
                eprintln!("[daemon] session {} history disabled: {e}", short(session_id));
            }
        }

        let state = Arc::new(AsyncMutex::new(SessionState {
            agent,
            workspace,
            last_active: Instant::now(),
        }));

        // Write-lock + check-and-insert: if another task raced us, return
        // their session and drop ours (its workspace is a fresh /tmp dir
        // that's harmless to leak transiently).
        let mut map = self.inner.write().await;
        if let Some(existing) = map.get(session_id) {
            return Ok(existing.clone());
        }
        map.insert(session_id.to_string(), state.clone());
        eprintln!("[daemon] session {} opened (model={})", short(session_id), model);
        Ok(state)
    }

    /// Drop sessions idle longer than `idle`. Sessions currently in flight
    /// (mutex locked) are skipped this round and retried next sweep.
    async fn gc_idle(&self, idle: Duration) {
        let now = Instant::now();

        let candidates: Vec<String> = {
            let map = self.inner.read().await;
            map.iter()
                .filter_map(|(id, state)| {
                    state.try_lock().ok().and_then(|s| {
                        if now.duration_since(s.last_active) > idle {
                            Some(id.clone())
                        } else {
                            None
                        }
                    })
                })
                .collect()
        };

        if candidates.is_empty() {
            return;
        }

        let mut map = self.inner.write().await;
        for id in candidates {
            let Some(state_arc) = map.remove(&id) else {
                continue;
            };
            match Arc::try_unwrap(state_arc) {
                Ok(mutex) => {
                    let state = mutex.into_inner();
                    eprintln!("[daemon] session {} GC", short(&id));
                    tokio::spawn(async move {
                        if let Err(e) = state.workspace.cleanup(false).await {
                            eprintln!("[daemon] worktree cleanup failed: {e}");
                        }
                    });
                }
                Err(arc) => {
                    // Another task picked up a reference between our scan
                    // and the write lock — put it back and retry next sweep.
                    map.insert(id, arc);
                }
            }
        }
    }
}

fn short(id: &str) -> &str {
    let n = id.char_indices().nth(8).map(|(i, _)| i).unwrap_or(id.len());
    &id[..n]
}

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

    let registry = Arc::new(SessionRegistry::new());

    // Background GC for idle sessions. Survives reconnects (lives at run()
    // scope, not per-connection).
    let gc_registry = registry.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(SESSION_GC_INTERVAL);
        tick.tick().await; // skip immediate first tick
        loop {
            tick.tick().await;
            gc_registry.gc_idle(SESSION_IDLE_TIMEOUT).await;
        }
    });

    loop {
        match connect_and_pump(&ws_url, &bearer, &config, registry.clone()).await {
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
    registry: Arc<SessionRegistry>,
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
            ServerMsg::RunTask {
                task_id,
                session_id,
                prompt,
                model,
            } => {
                let pending = pending.clone();
                let out_tx = out_tx.clone();
                let config = config.clone();
                let registry = registry.clone();
                tokio::spawn(async move {
                    run_agent_task(
                        task_id, session_id, prompt, model, config, registry, out_tx, pending,
                    )
                    .await;
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
    session_id: String,
    prompt: String,
    model: Option<String>,
    config: DaemonConfig,
    registry: Arc<SessionRegistry>,
    out_tx: mpsc::UnboundedSender<DeviceMsg>,
    pending: PendingApprovals,
) {
    let state_arc = match registry
        .get_or_create(&session_id, &config, model.as_deref())
        .await
    {
        Ok(s) => s,
        Err(e) => {
            emit_error(&out_tx, &task_id, e);
            return;
        }
    };

    // Lock the session for the duration of this turn — sequential per
    // session. Concurrent RunTasks for the same session queue up here.
    let mut state = state_arc.lock().await;

    let mut sink = WsEventSink {
        task_id: task_id.clone(),
        out_tx: out_tx.clone(),
    };
    let mut approval = WsApprovalHandler {
        task_id: task_id.clone(),
        out_tx: out_tx.clone(),
        pending,
    };

    if let Err(e) = state.agent.run_turn(&prompt, &mut approval, &mut sink).await {
        emit_error(&out_tx, &task_id, format!("agent: {e}"));
    }
    state.last_active = Instant::now();
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
