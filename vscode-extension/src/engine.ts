// Owns the kotonia-cli engine child process and the JSON stdio protocol.
//
// The engine is spawned once with `--serve`; the extension is a thin client.
// This class handles: spawn + env injection, line-buffered JSONL parsing of
// stdout, forwarding parsed messages to a listener, and sending inbound
// messages (user_turn / approval_response / cancel) on stdin.

import * as vscode from "vscode";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import {
  Inbound,
  Outbound,
  PROTOCOL_VERSION,
  EditorContext,
} from "./protocol";

export interface EngineOptions {
  binary: string;
  cwd: string;
  model: string;
  approvalMode: string;
  workspaceMode: "worktree" | "in-place";
  extraArgs: string[];
  resumeSessionId?: string;
  /** Extra env vars (API keys) injected into the child. */
  env: Record<string, string>;
}

export type MessageListener = (msg: Outbound) => void;

export class KotoniaEngine {
  private proc: ChildProcessWithoutNullStreams | undefined;
  private stdoutBuf = "";
  private disposed = false;

  constructor(
    private readonly opts: EngineOptions,
    private readonly onMessage: MessageListener,
    private readonly log: vscode.OutputChannel,
    private readonly onExit: (code: number | null) => void,
  ) {}

  start(): void {
    const args = ["--serve", "--model", this.opts.model, "--approval", this.opts.approvalMode];
    if (this.opts.workspaceMode === "in-place") {
      args.push("--in-place");
    }
    if (this.opts.resumeSessionId) {
      args.push("--resume", this.opts.resumeSessionId);
    }
    args.push(...this.opts.extraArgs);

    this.log.appendLine(`[engine] spawn: ${this.opts.binary} ${args.join(" ")}`);
    this.log.appendLine(`[engine] cwd: ${this.opts.cwd}`);

    let proc: ChildProcessWithoutNullStreams;
    try {
      proc = spawn(this.opts.binary, args, {
        cwd: this.opts.cwd,
        env: { ...process.env, ...this.opts.env },
      });
    } catch (e) {
      this.log.appendLine(`[engine] spawn failed: ${e}`);
      vscode.window.showErrorMessage(`Kotonia: failed to spawn engine (${this.opts.binary}): ${e}`);
      this.onExit(null);
      return;
    }
    this.proc = proc;

    proc.stdout.setEncoding("utf8");
    proc.stdout.on("data", (chunk: string) => this.onStdout(chunk));

    proc.stderr.setEncoding("utf8");
    proc.stderr.on("data", (chunk: string) => {
      // stderr is human/diagnostic logs by contract — surface in the channel.
      this.log.append(chunk);
    });

    proc.on("error", (err) => {
      this.log.appendLine(`[engine] process error: ${err.message}`);
      vscode.window.showErrorMessage(`Kotonia engine error: ${err.message}`);
    });

    proc.on("exit", (code) => {
      this.log.appendLine(`[engine] exited with code ${code}`);
      if (!this.disposed) {
        this.onExit(code);
      }
    });
  }

  private onStdout(chunk: string): void {
    this.stdoutBuf += chunk;
    let idx: number;
    while ((idx = this.stdoutBuf.indexOf("\n")) >= 0) {
      const line = this.stdoutBuf.slice(0, idx).trim();
      this.stdoutBuf = this.stdoutBuf.slice(idx + 1);
      if (!line) {
        continue;
      }
      let msg: Outbound;
      try {
        msg = JSON.parse(line) as Outbound;
      } catch (e) {
        this.log.appendLine(`[engine] non-JSON stdout line: ${line}`);
        continue;
      }
      if (msg.type === "hello" && msg.protocol_version !== PROTOCOL_VERSION) {
        vscode.window.showWarningMessage(
          `Kotonia: engine protocol v${msg.protocol_version} != extension v${PROTOCOL_VERSION}. Update one side.`,
        );
      }
      this.onMessage(msg);
    }
  }

  private send(msg: Inbound): void {
    if (!this.proc || this.proc.exitCode !== null) {
      this.log.appendLine(`[engine] cannot send, process not running: ${JSON.stringify(msg)}`);
      return;
    }
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
  }

  sendUserTurn(text: string, context?: EditorContext): void {
    this.send({ type: "user_turn", text, context });
  }

  sendApproval(approvalId: number, approve: boolean, remember = false): void {
    this.send({ type: "approval_response", approval_id: approvalId, approve, remember });
  }

  cancel(): void {
    this.send({ type: "cancel" });
  }

  isRunning(): boolean {
    return !!this.proc && this.proc.exitCode === null;
  }

  dispose(): void {
    this.disposed = true;
    if (this.proc && this.proc.exitCode === null) {
      // Closing stdin makes the serve loop exit cleanly (EOF), which lets the
      // engine tear down its worktree. Fall back to SIGTERM if it lingers.
      try {
        this.proc.stdin.end();
      } catch {
        /* ignore */
      }
      const proc = this.proc;
      setTimeout(() => {
        if (proc.exitCode === null) {
          proc.kill("SIGTERM");
        }
      }, 3000);
    }
  }
}
