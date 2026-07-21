# Kotonia Agent — VS Code extension

A thin VS Code client over the [`kotonia-cli`](../) Rust engine. The engine
does all the work (ReAct loop, tool execution, git-worktree isolation,
approval policy, history); this extension is just a UI that speaks the engine's
JSON stdio protocol (`kotonia-cli --serve`).

## Architecture

```
VS Code (extension host)                     kotonia-cli --serve (child)
┌───────────────────────────┐                ┌──────────────────────────┐
│ extension.ts  (lifecycle)  │ ── stdin ───▶  │ stdin reader → turn/appr │
│ engine.ts     (spawn+JSONL)│ ◀─ stdout ───  │ JsonSink (events)        │
│ panel.ts      (webview)    │                │ JsonApproval             │
│ media/main.js (render UI)  │                │ agent loop (untouched)   │
└───────────────────────────┘                └──────────────────────────┘
```

Protocol wire types live in [`src/protocol.ts`](src/protocol.ts), mirroring the
Rust side (`../src/serve.rs` + the `Event` enum). Monorepo layout keeps both in
lockstep — see [`../docs/VSCODE_EXTENSION_DESIGN_QUESTIONS.md`](../docs/VSCODE_EXTENSION_DESIGN_QUESTIONS.md).

## Where things run (important)

The engine needs `bash`, `git`, and (for local models) the LLM servers — all on
Linux. The Windows dev machine can't build native Rust (Smart App Control). So
run VS Code **connected to Linux**:

- **Remote-SSH** to the GPU box (production-like: local models reachable), or
- **Remote-WSL** locally (use a hosted model: `kotonia-llm-basic` + API key).

The extension host — and therefore the spawned engine — runs on that remote.
`kotonia.enginePath` resolves there.

## Setup

1. Build the engine on the target host: `cargo build` in the repo root →
   `target/debug/kotonia-cli`.
2. `cd vscode-extension && npm install && npm run compile`.
3. Point `kotonia.enginePath` at the binary (e.g.
   `${workspaceFolder}/target/debug/kotonia-cli`, or leave `kotonia-cli` if on PATH).
4. For hosted models, run **Kotonia: Set Kotonia API Key** (stored in
   VS Code SecretStorage, injected as `KOTONIA_API_KEY` into the engine).
5. **Kotonia: Open Agent** — the panel starts the engine and shows the handshake.

Dev-run the extension: open this folder in VS Code and press **F5** (Extension
Development Host).

## Settings

| Setting | Default | Notes |
|---|---|---|
| `kotonia.enginePath` | `kotonia-cli` | Engine binary (supports `${workspaceFolder}`). |
| `kotonia.model` | `kotonia-llm-basic` | Hosted (GPU-free) default. |
| `kotonia.approvalMode` | `allowlist` | `all` / `allowlist` / `auto`. |
| `kotonia.workspaceMode` | `worktree` | `worktree` (isolated) or `in-place`. |
| `kotonia.extraArgs` | `[]` | Extra engine CLI args. |

## Status (Phase 2)

Done: engine spawn + env injection, JSONL parsing, event rendering, inline
approval UI, cancel, editor-selection context in `user_turn`, protocol-version
check, crash → restart.

Not yet (Phase 3): worktree diff view + Merge button, `file:line` jump, session
list / resume UI, token streaming. The `remember` approval flag is sent but
session-scoped auto-approve memory is not yet applied on the extension side.
