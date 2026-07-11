//! JSON stdio protocol (`--serve`) for machine-readable frontends — the
//! VS Code extension in particular.
//!
//! The agent engine is untouched: we just swap the CLI's human `StdoutSink`
//! / `StdioApproval` for JSON implementations of the same `EventSink` /
//! `ApprovalHandler` traits, and drive `run_turn` from stdin instead of a
//! TTY REPL.
//!
//! ## Wire contract (protocol v1)
//!
//! Newline-delimited JSON (JSONL). **stdout carries the protocol only**;
//! human/diagnostic logs stay on stderr. Every message is a single JSON
//! object with a `"type"` tag.
//!
//! Outbound (engine → frontend):
//! - `hello`            — handshake, emitted once at startup (see [`HelloInfo`]).
//! - the `Event` enum   — `iteration_start` / `llm_thinking` / `bash` /
//!                        `bash_skipped` / `observation` / `final` /
//!                        `malformed` / `error` / `done`. Each also carries
//!                        the current `turn_id`.
//! - `approval_request` — `{turn_id, approval_id, command, reason}`.
//!
//! Inbound (frontend → engine, on stdin):
//! - `user_turn`         — `{text, context?}`. One user turn. Rejected while a
//!                         turn is running.
//! - `approval_response` — `{approval_id, approve, remember?}`. `remember` is
//!                         an extension-side concern and ignored here.
//! - `cancel`            — `{turn_id?}`. Requests a coarse stop of the running
//!                         turn at the next iteration boundary.
//!
//! `resume` is NOT a protocol message — it's passed as the `--resume <id>`
//! spawn argument, mirroring the CLI.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::agent::agent::{Agent, ApprovalHandler, ApprovalOutcome, Event, EventSink};

/// Bumped when the wire contract changes incompatibly. The frontend compares
/// against its own expectation and refuses to attach on a major mismatch.
pub const PROTOCOL_VERSION: u32 = 1;

/// Static facts announced in the one-shot `hello` handshake. Built by the
/// binary from the resolved provider + workspace so the frontend can render
/// its status bar without guessing.
pub struct HelloInfo {
    pub model: String,
    pub backend: String,
    pub tool_mode: &'static str,
    pub approval_mode: String,
    pub workspace_root: String,
    pub is_worktree: bool,
    pub session_id: Option<String>,
    pub kotonia_api: bool,
}

/// Messages the frontend sends us on stdin.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Inbound {
    UserTurn {
        text: String,
        /// Optional editor context (active file / selection). Accepted so the
        /// frontend can send it, but not yet folded into the prompt in v1.
        #[serde(default)]
        #[allow(dead_code)]
        context: Option<Value>,
    },
    ApprovalResponse {
        #[serde(default)]
        approval_id: u64,
        approve: bool,
        /// Extension-side "allow for session" hint. The engine's allowlist is
        /// unchanged; we parse-and-ignore so the frontend owns that policy.
        #[serde(default)]
        #[allow(dead_code)]
        remember: bool,
    },
    Cancel {
        #[serde(default)]
        #[allow(dead_code)]
        turn_id: Option<u64>,
    },
}

/// An approval verdict relayed from the stdin reader to the blocking
/// `JsonApproval::ask` waiting inside `run_turn`.
struct ApprovalMsg {
    approval_id: u64,
    approve: bool,
}

/// Serialize a JSON value as one compact line on stdout and flush. This is
/// the ONLY writer to stdout in serve mode; the reader thread never touches
/// it, and `ask` / `emit` run on the same (executor) thread, so no locking
/// races. Flushing per-line keeps the frontend responsive on a pipe.
fn emit_value(v: &Value) {
    let mut out = std::io::stdout().lock();
    if serde_json::to_writer(&mut out, v).is_ok() {
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}

/// `EventSink` that renders each `Event` as `{type, turn_id, ...fields}`.
///
/// Also tracks whether a terminal `done` was emitted for the current turn so
/// the serve loop can uphold the invariant "every turn ends with exactly one
/// `done`" — the engine's error paths (LLM / executor failure) return `Err`
/// after only an `error` event, and `error` is not always terminal (the
/// native `max_tokens` path emits it mid-turn), so `done` is the frontend's
/// only reliable end-of-turn signal.
struct JsonSink {
    turn_id: u64,
    saw_done: bool,
}

impl JsonSink {
    fn new() -> Self {
        Self {
            turn_id: 0,
            saw_done: false,
        }
    }
    fn begin_turn(&mut self, id: u64) {
        self.turn_id = id;
        self.saw_done = false;
    }
    fn saw_done(&self) -> bool {
        self.saw_done
    }
    /// Emit a synthetic terminal `done` (used when the engine bailed via
    /// `Err` without emitting one).
    fn emit_synthetic_done(&mut self) {
        emit_value(&json!({
            "type": "done",
            "turn_id": self.turn_id,
            "iterations": 0,
            "success": false,
        }));
        self.saw_done = true;
    }
}

impl EventSink for JsonSink {
    fn emit(&mut self, event: Event) {
        if matches!(event, Event::Done { .. }) {
            self.saw_done = true;
        }
        // The Event derive already produces `{"type": "...", ...}`; splice in
        // the turn id so the frontend can group events per turn.
        let mut v = serde_json::to_value(&event).unwrap_or_else(|e| {
            json!({"type": "error", "message": format!("event serialize failed: {e}")})
        });
        if let Some(obj) = v.as_object_mut() {
            obj.insert("turn_id".to_string(), json!(self.turn_id));
        }
        emit_value(&v);
    }
}

/// `ApprovalHandler` that emits an `approval_request` and blocks (on a std
/// channel fed by the stdin reader) until the matching `approval_response`
/// arrives. Blocking is fine: the current-thread runtime has only the one
/// `run_turn` task, and it's the thing waiting.
struct JsonApproval {
    rx: mpsc::Receiver<ApprovalMsg>,
    turn_id: u64,
    next_approval_id: u64,
}

impl JsonApproval {
    fn new(rx: mpsc::Receiver<ApprovalMsg>) -> Self {
        Self {
            rx,
            turn_id: 0,
            next_approval_id: 0,
        }
    }
    fn set_turn(&mut self, id: u64) {
        self.turn_id = id;
    }
}

impl ApprovalHandler for JsonApproval {
    fn ask(&mut self, command: &str, reason: &str) -> ApprovalOutcome {
        self.next_approval_id += 1;
        let approval_id = self.next_approval_id;
        emit_value(&json!({
            "type": "approval_request",
            "turn_id": self.turn_id,
            "approval_id": approval_id,
            "command": command,
            "reason": reason,
        }));

        // Wait for the response that matches this request. Turns are serial so
        // at most one approval is ever outstanding, but we still drop stale
        // ids defensively. `approval_id == 0` means the frontend didn't echo
        // one — accept it leniently.
        loop {
            match self.rx.recv() {
                Ok(msg) if msg.approval_id == approval_id || msg.approval_id == 0 => {
                    return if msg.approve {
                        ApprovalOutcome::Approve
                    } else {
                        ApprovalOutcome::Deny
                    };
                }
                Ok(_) => continue, // stale response for an earlier approval
                Err(_) => return ApprovalOutcome::Deny, // stdin closed → deny
            }
        }
    }
}

/// Read stdin line-by-line on a dedicated OS thread and demux by `type`:
/// user turns flow to the async serve loop, approvals to `JsonApproval`, and
/// cancels flip the shared flag directly. When stdin hits EOF the senders
/// drop, which unblocks both consumers.
fn read_stdin(
    turn_tx: tokio::sync::mpsc::UnboundedSender<String>,
    appr_tx: mpsc::Sender<ApprovalMsg>,
    cancel: Arc<AtomicBool>,
) {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Inbound>(&line) {
            Ok(Inbound::UserTurn { text, .. }) => {
                if turn_tx.send(text).is_err() {
                    break; // serve loop gone
                }
            }
            Ok(Inbound::Cancel { .. }) => {
                cancel.store(true, Ordering::SeqCst);
            }
            Ok(Inbound::ApprovalResponse {
                approval_id,
                approve,
                ..
            }) => {
                let _ = appr_tx.send(ApprovalMsg {
                    approval_id,
                    approve,
                });
            }
            Err(e) => {
                eprintln!("kotonia-cli serve: ignoring unparseable line: {e}");
            }
        }
    }
}

/// Run the JSON stdio protocol against `agent` until stdin closes.
///
/// Consumes `agent` (like the CLI's one-shot / REPL branches). The caller
/// still owns the workspace and performs cleanup afterwards.
pub async fn serve(mut agent: Agent, hello: HelloInfo) {
    emit_value(&json!({
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "model": hello.model,
        "backend": hello.backend,
        "tool_mode": hello.tool_mode,
        "approval_mode": hello.approval_mode,
        "workspace_root": hello.workspace_root,
        "is_worktree": hello.is_worktree,
        "session_id": hello.session_id,
        "kotonia_api": hello.kotonia_api,
    }));

    let (turn_tx, mut turn_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (appr_tx, appr_rx) = mpsc::channel::<ApprovalMsg>();
    let cancel = agent.cancel_handle();

    // Dedicated OS thread so stdin keeps draining even while a turn blocks the
    // async executor (e.g. waiting on an approval).
    std::thread::spawn(move || read_stdin(turn_tx, appr_tx, cancel));

    let mut approval = JsonApproval::new(appr_rx);
    let mut sink = JsonSink::new();
    let mut turn_id: u64 = 0;

    // Serial: one turn at a time. New `user_turn`s that arrive mid-turn queue
    // in the channel and run after the current one finishes.
    while let Some(text) = turn_rx.recv().await {
        turn_id += 1;
        approval.set_turn(turn_id);
        sink.begin_turn(turn_id);
        // The agent emits its own terminal `done` on success / iteration-limit
        // / cancel. On a hard `Err` (LLM or executor failure) it emits only an
        // `error`, so we synthesize the closing `done` to keep the per-turn
        // contract. The Rust error is logged to stderr; serving continues.
        if let Err(e) = agent.run_turn(&text, &mut approval, &mut sink).await {
            eprintln!("kotonia-cli serve: turn {turn_id} ended with error: {e}");
            if !sink.saw_done() {
                sink.emit_synthetic_done();
            }
        }
    }
}
