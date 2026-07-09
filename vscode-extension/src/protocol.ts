// TypeScript mirror of the kotonia-cli `--serve` JSON stdio protocol.
//
// Source of truth is the Rust side (`src/serve.rs` + the `Event` enum in
// `src/agent/agent.rs`). Keep this in lockstep — the monorepo layout exists
// precisely so both move together. See
// `docs/VSCODE_EXTENSION_DESIGN_QUESTIONS.md`.

export const PROTOCOL_VERSION = 1;

// ---- Outbound: engine -> extension (stdout) --------------------------------

export interface Hello {
  type: "hello";
  protocol_version: number;
  model: string;
  backend: "local" | "deepseek-api" | "kotonia-api" | string;
  tool_mode: "native" | "delimiter";
  approval_mode: string;
  workspace_root: string;
  is_worktree: boolean;
  session_id: string | null;
  kotonia_api: boolean;
}

// Every event below also carries `turn_id` (spliced in by the JsonSink).
interface WithTurn {
  turn_id: number;
}

export interface ExecutionResult {
  exit_code: number;
  timed_out: boolean;
  truncated: boolean;
  combined: string;
}

export type EngineEvent =
  | ({ type: "iteration_start"; iteration: number; max: number } & WithTurn)
  | ({ type: "llm_thinking" } & WithTurn)
  | ({ type: "bash"; command: string } & WithTurn)
  | ({ type: "bash_skipped"; command: string; reason: string } & WithTurn)
  | ({ type: "observation"; result: ExecutionResult } & WithTurn)
  | ({ type: "final"; answer: string } & WithTurn)
  | ({ type: "malformed"; excerpt: string } & WithTurn)
  | ({ type: "error"; message: string } & WithTurn)
  | ({ type: "done"; iterations: number; success: boolean } & WithTurn);

export interface ApprovalRequest extends WithTurn {
  type: "approval_request";
  approval_id: number;
  command: string;
  reason: string;
}

export type Outbound = Hello | ApprovalRequest | EngineEvent;

// ---- Inbound: extension -> engine (stdin) ----------------------------------

export interface EditorContext {
  active_file?: string;
  selection?: { start_line: number; end_line: number };
  selection_text?: string;
}

export interface UserTurn {
  type: "user_turn";
  text: string;
  context?: EditorContext;
}

export interface ApprovalResponse {
  type: "approval_response";
  approval_id: number;
  approve: boolean;
  remember?: boolean;
}

export interface Cancel {
  type: "cancel";
  turn_id?: number;
}

export type Inbound = UserTurn | ApprovalResponse | Cancel;

// Narrowing helpers --------------------------------------------------------

export function isHello(m: Outbound): m is Hello {
  return m.type === "hello";
}

export function isApprovalRequest(m: Outbound): m is ApprovalRequest {
  return m.type === "approval_request";
}
