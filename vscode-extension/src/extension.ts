// Extension entry point. Owns the engine + panel lifecycle and bridges panel
// actions to the engine and engine messages back to the panel.

import * as vscode from "vscode";
import { KotoniaEngine } from "./engine";
import { ChatPanel, PanelAction } from "./panel";
import { EditorContext, Outbound } from "./protocol";

const SECRET_KOTONIA = "kotonia.apiKey";
const SECRET_DEEPSEEK = "kotonia.deepseekKey";
const SELECTION_CAP = 2000;

interface Session {
  engine: KotoniaEngine;
  panel: ChatPanel;
}

let session: Session | undefined;
let output: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("Kotonia Agent");
  context.subscriptions.push(output);

  context.subscriptions.push(
    vscode.commands.registerCommand("kotonia.start", () => start(context)),
    vscode.commands.registerCommand("kotonia.newSession", () => restart(context)),
    vscode.commands.registerCommand("kotonia.cancel", () => session?.engine.cancel()),
    vscode.commands.registerCommand("kotonia.setKotoniaApiKey", () =>
      promptSecret(context, SECRET_KOTONIA, "Kotonia API key (KOTONIA_API_KEY)"),
    ),
    vscode.commands.registerCommand("kotonia.setDeepSeekApiKey", () =>
      promptSecret(context, SECRET_DEEPSEEK, "DeepSeek API key (DEEPSEEK_API_KEY)"),
    ),
  );
}

export function deactivate(): void {
  session?.engine.dispose();
  session = undefined;
}

async function start(context: vscode.ExtensionContext): Promise<void> {
  if (session) {
    session.panel.reveal();
    return;
  }
  const folder = pickWorkspaceFolder();
  if (!folder) {
    vscode.window.showErrorMessage("Kotonia: open a folder first — the agent needs a workspace.");
    return;
  }

  const cfg = vscode.workspace.getConfiguration("kotonia");
  const binary = resolveBinary(cfg.get<string>("enginePath", "kotonia-cli"), folder);
  const env = await gatherEnv(context);

  const panel = new ChatPanel(
    context.extensionUri,
    (a) => handlePanelAction(a),
    () => {
      // Panel closed by the user → tear down the engine.
      session?.engine.dispose();
      session = undefined;
    },
  );

  const engine = new KotoniaEngine(
    {
      binary,
      cwd: folder.uri.fsPath,
      model: cfg.get<string>("model", "kotonia-gemma4-26b"),
      approvalMode: cfg.get<string>("approvalMode", "allowlist"),
      workspaceMode: cfg.get<"worktree" | "in-place">("workspaceMode", "worktree"),
      extraArgs: cfg.get<string[]>("extraArgs", []),
      env,
    },
    (msg: Outbound) => onEngineMessage(msg),
    output,
    (code) => onEngineExit(context, code),
  );

  session = { engine, panel };
  engine.start();
}

async function restart(context: vscode.ExtensionContext): Promise<void> {
  session?.engine.dispose();
  session?.panel.dispose();
  session = undefined;
  await start(context);
}

function handlePanelAction(a: PanelAction): void {
  if (!session) {
    return;
  }
  switch (a.kind) {
    case "ready":
      break;
    case "send":
      session.engine.sendUserTurn(a.text, currentEditorContext());
      break;
    case "approval":
      session.engine.sendApproval(a.approvalId, a.approve, a.remember);
      break;
    case "cancel":
      session.engine.cancel();
      break;
  }
}

function onEngineMessage(msg: Outbound): void {
  session?.panel.postEngineMessage(msg);
}

function onEngineExit(context: vscode.ExtensionContext, code: number | null): void {
  if (!session) {
    return;
  }
  // Surface the death in the panel and offer a restart. The session log is on
  // disk, so `New Session` can resume context later.
  session.panel.postEngineMessage({
    type: "error",
    message: `engine exited (code ${code}). Use "Kotonia: New Session" to restart.`,
    turn_id: 0,
  } as Outbound);
  const dead = session;
  session = undefined;
  dead.engine.dispose();
  vscode.window
    .showWarningMessage("Kotonia engine exited.", "Restart")
    .then((choice) => {
      if (choice === "Restart") {
        dead.panel.dispose();
        void start(context);
      }
    });
}

// ---- helpers ---------------------------------------------------------------

function pickWorkspaceFolder(): vscode.WorkspaceFolder | undefined {
  const active = vscode.window.activeTextEditor?.document.uri;
  if (active) {
    const f = vscode.workspace.getWorkspaceFolder(active);
    if (f) {
      return f;
    }
  }
  return vscode.workspace.workspaceFolders?.[0];
}

function resolveBinary(raw: string, folder: vscode.WorkspaceFolder): string {
  return raw.replace(/\$\{workspaceFolder\}/g, folder.uri.fsPath);
}

async function gatherEnv(context: vscode.ExtensionContext): Promise<Record<string, string>> {
  const env: Record<string, string> = {};
  const kotonia = await context.secrets.get(SECRET_KOTONIA);
  if (kotonia) {
    env.KOTONIA_API_KEY = kotonia;
  }
  const deepseek = await context.secrets.get(SECRET_DEEPSEEK);
  if (deepseek) {
    env.DEEPSEEK_API_KEY = deepseek;
  }
  return env;
}

function currentEditorContext(): EditorContext | undefined {
  const ed = vscode.window.activeTextEditor;
  if (!ed) {
    return undefined;
  }
  const ctx: EditorContext = { active_file: vscode.workspace.asRelativePath(ed.document.uri) };
  const sel = ed.selection;
  if (!sel.isEmpty) {
    ctx.selection = { start_line: sel.start.line + 1, end_line: sel.end.line + 1 };
    ctx.selection_text = ed.document.getText(sel).slice(0, SELECTION_CAP);
  }
  return ctx;
}

async function promptSecret(
  context: vscode.ExtensionContext,
  key: string,
  label: string,
): Promise<void> {
  const value = await vscode.window.showInputBox({
    prompt: `Enter your ${label}`,
    password: true,
    ignoreFocusOut: true,
  });
  if (value === undefined) {
    return;
  }
  if (value === "") {
    await context.secrets.delete(key);
    vscode.window.showInformationMessage(`Kotonia: cleared ${label}.`);
  } else {
    await context.secrets.store(key, value);
    vscode.window.showInformationMessage(`Kotonia: stored ${label}. Restart the session to apply.`);
  }
}
