// Extension entry point. Layout (Claude-Code-style):
//   - Sidebar "Sessions" tree = history / session management.
//   - Editor-area panel = the live chat (conversation + avatar + input).
// The engine's events flow to the chat panel only; the sidebar just lists
// on-disk sessions and resumes them into the panel.

import * as vscode from "vscode";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";
import { execFile, spawn } from "child_process";
import { KotoniaEngine } from "./engine";
import { ChatPanel } from "./panel";
import { SessionTreeProvider } from "./sessionTree";
import { PanelAction } from "./webview";
import { EditorContext, Hello, Outbound } from "./protocol";

const SECRET_KOTONIA = "kotonia.apiKey";
const SECRET_DEEPSEEK = "kotonia.deepseekKey";
const SELECTION_CAP = 2000;

interface EngineState {
  engine: KotoniaEngine;
  hello?: Hello;
  remembered: Set<string>;
  lastTurnId: number;
  greeted: boolean;
  firstTurn: boolean;
}

let chatPanel: ChatPanel | undefined;
let engineState: EngineState | undefined;
let sessionTree: SessionTreeProvider;
let output: vscode.OutputChannel;
let extContext: vscode.ExtensionContext;
let speakAbort: AbortController | undefined;

/** Post to the chat panel (the only chat surface). No-op if it's closed. */
const ui = {
  postEngineMessage(m: Outbound): void {
    chatPanel?.postEngineMessage(m);
  },
  note(text: string, turnId?: number): void {
    chatPanel?.note(text, turnId);
  },
  setBusy(busy: boolean): void {
    chatPanel?.setBusy(busy);
  },
  reset(): void {
    chatPanel?.reset();
  },
  avatarBegin(): void {
    chatPanel?.avatarBegin();
  },
  avatarChunk(chunkType: number, data: string): void {
    chatPanel?.avatarChunk(chunkType, data);
  },
  avatarEnd(): void {
    chatPanel?.avatarEnd();
  },
  avatarStop(): void {
    chatPanel?.avatarStop();
  },
};

export function activate(context: vscode.ExtensionContext): void {
  extContext = context;
  output = vscode.window.createOutputChannel("Kotonia Agent");
  context.subscriptions.push(output);

  sessionTree = new SessionTreeProvider();
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("kotonia.sessions", sessionTree),
    vscode.commands.registerCommand("kotonia.start", () => newChat()),
    vscode.commands.registerCommand("kotonia.newChat", () => newChat()),
    vscode.commands.registerCommand("kotonia.openSession", (id: string) => openSession(id)),
    vscode.commands.registerCommand("kotonia.refreshSessions", () => sessionTree.refresh()),
    vscode.commands.registerCommand("kotonia.login", () => runLogin()),
    vscode.commands.registerCommand("kotonia.toggleAvatar", () => toggleAvatar()),
    vscode.commands.registerCommand("kotonia.cancel", () => engineState?.engine.cancel()),
    vscode.commands.registerCommand("kotonia.reviewChanges", () => reviewChanges()),
    vscode.commands.registerCommand("kotonia.applyChanges", () => applyChanges()),
    vscode.commands.registerCommand("kotonia.setKotoniaApiKey", () =>
      promptSecret(SECRET_KOTONIA, "Kotonia API key (KOTONIA_API_KEY)"),
    ),
    vscode.commands.registerCommand("kotonia.setDeepSeekApiKey", () =>
      promptSecret(SECRET_DEEPSEEK, "DeepSeek API key (DEEPSEEK_API_KEY)"),
    ),
  );
}

export function deactivate(): void {
  engineState?.engine.dispose();
  engineState = undefined;
}

// ---- chat panel + engine lifecycle -----------------------------------------

/** Create the center chat panel if needed, then reveal it. */
function ensureChatPanel(): void {
  if (chatPanel) {
    chatPanel.reveal();
    return;
  }
  chatPanel = new ChatPanel(
    extContext.extensionUri,
    (a) => handlePanelAction(a),
    () => {
      // Panel closed → end the session (it's saved to disk and resumable).
      chatPanel = undefined;
      engineState?.engine.dispose();
      engineState = undefined;
      speakAbort?.abort();
      sessionTree.refresh();
    },
  );
}

/** Start a brand-new chat session in the center panel. */
function newChat(): void {
  ensureChatPanel();
  restart();
}

/** Resume an existing session (by id) in the center panel. */
function openSession(id: string): void {
  ensureChatPanel();
  restart(id);
}

function restart(resumeSessionId?: string): void {
  engineState?.engine.dispose();
  engineState = undefined;
  speakAbort?.abort();
  ui.reset();
  startEngine(resumeSessionId);
}

function startEngine(resumeSessionId?: string): void {
  const folder = pickWorkspaceFolder();
  if (!folder) {
    ui.note("open a folder first — the agent needs a workspace.");
    return;
  }
  const cfg = vscode.workspace.getConfiguration("kotonia");
  const binary = resolveBinary(cfg.get<string>("enginePath", "kotonia-cli"), folder);
  const model = cfg.get<string>("model", "kotonia-gemma4-26b");

  void gatherEnv().then((env) => {
    if (model.startsWith("kotonia") && !env.KOTONIA_API_KEY && !process.env.KOTONIA_API_KEY) {
      const daemonJson = path.join(os.homedir(), ".kotonia", "daemon.json");
      if (!fs.existsSync(daemonJson)) {
        ui.note("not signed in — run “Kotonia: Login” (device-code), then start a new chat.");
      }
    }
    const engine = new KotoniaEngine(
      {
        binary,
        cwd: folder.uri.fsPath,
        model,
        approvalMode: cfg.get<string>("approvalMode", "allowlist"),
        workspaceMode: cfg.get<"worktree" | "in-place">("workspaceMode", "worktree"),
        extraArgs: cfg.get<string[]>("extraArgs", []),
        resumeSessionId,
        env,
      },
      (msg: Outbound) => onEngineMessage(msg),
      output,
      (code) => onEngineExit(code),
    );
    engineState = { engine, remembered: new Set(), lastTurnId: 0, greeted: false, firstTurn: true };
    engine.start();
    setTimeout(() => sessionTree.refresh(), 800);
  });
}

function handlePanelAction(a: PanelAction): void {
  switch (a.kind) {
    case "ready":
      // The chat webview (re)loaded — recover its status line if a session is
      // already live. Engine start is explicit (New Chat / open session).
      if (engineState?.hello) {
        ui.postEngineMessage(engineState.hello);
      }
      break;
    case "send": {
      if (!engineState) {
        ui.note("no active session — press New Chat (＋) in the Sessions sidebar.");
        break;
      }
      let text = a.text;
      if (engineState.firstTurn) {
        engineState.firstTurn = false;
        const instr = languageInstruction();
        if (instr) {
          text = `${instr}\n\n${text}`;
        }
      }
      ui.setBusy(true);
      engineState.engine.sendUserTurn(text, currentEditorContext());
      break;
    }
    case "approval":
      if (a.remember && engineState) {
        engineState.remembered.add(leadingToken(a.command));
      }
      engineState?.engine.sendApproval(a.approvalId, a.approve, a.remember);
      break;
    case "cancel":
      engineState?.engine.cancel();
      break;
    case "open":
      void openFileAtLine(a.file, a.line);
      break;
  }
}

function onEngineMessage(msg: Outbound): void {
  if (!engineState) {
    return;
  }
  if (msg.type === "hello") {
    engineState.hello = msg;
    if (!engineState.greeted) {
      engineState.greeted = true;
      ui.note("session ready — type your request below and press Enter (Shift+Enter for a newline).");
    }
  } else if ("turn_id" in msg && typeof msg.turn_id === "number") {
    engineState.lastTurnId = msg.turn_id;
  }

  if (msg.type === "final") {
    void speak(msg.answer);
  }
  if (msg.type === "done") {
    sessionTree.refresh();
  }

  if (msg.type === "approval_request") {
    const tok = leadingToken(msg.command);
    if (engineState.remembered.has(tok)) {
      engineState.engine.sendApproval(msg.approval_id, true, true);
      ui.note(`auto-approved (remembered \`${tok}\`): ${msg.command}`, msg.turn_id);
      return;
    }
  }

  ui.postEngineMessage(msg);
}

function onEngineExit(code: number | null): void {
  if (!engineState) {
    return;
  }
  ui.postEngineMessage({
    type: "error",
    message: `engine exited (code ${code}). Press New Chat (＋) to restart.`,
    turn_id: 0,
  } as Outbound);
  engineState.engine.dispose();
  engineState = undefined;
  vscode.window.showWarningMessage("Kotonia engine exited.", "New Chat").then((choice) => {
    if (choice === "New Chat") {
      newChat();
    }
  });
}

// ---- talking avatar (#3) ---------------------------------------------------

function toggleAvatar(): void {
  const cfg = vscode.workspace.getConfiguration("kotonia");
  const next = !cfg.get<boolean>("avatar.enabled", false);
  void cfg.update("avatar.enabled", next, vscode.ConfigurationTarget.Workspace);
  if (next && !cfg.get<string>("avatar.id", "").trim()) {
    vscode.window.showInformationMessage(
      "Kotonia: avatar on — set a registered avatar id in kotonia.avatar.id to see it speak.",
    );
  }
  ui.note(`talking avatar ${next ? "ON" : "OFF"}.`);
  if (!next) {
    speakAbort?.abort();
    ui.avatarStop();
  }
}

function readAvatarAuth(cfg: vscode.WorkspaceConfiguration): { base: string; token: string } | undefined {
  let token: string | undefined;
  let base: string | undefined;
  try {
    const raw = fs.readFileSync(path.join(os.homedir(), ".kotonia", "daemon.json"), "utf8");
    const dj = JSON.parse(raw) as { server?: string; device_token?: string };
    token = dj.device_token;
    base = dj.server;
  } catch {
    /* not logged in */
  }
  if (!token && process.env.KOTONIA_API_KEY) {
    token = process.env.KOTONIA_API_KEY;
  }
  const override = cfg.get<string>("avatar.apiBase", "").trim();
  base = (override || base || "https://kotonia.ai").replace(/\/+$/, "");
  return token ? { base, token } : undefined;
}

async function speak(text: string): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("kotonia");
  if (!cfg.get<boolean>("avatar.enabled", false)) {
    return;
  }
  const avatarId = cfg.get<string>("avatar.id", "").trim();
  if (!avatarId || !text.trim()) {
    return;
  }
  const auth = readAvatarAuth(cfg);
  if (!auth) {
    ui.note("avatar: not signed in — run “Kotonia: Login”.");
    return;
  }

  speakAbort?.abort();
  const abort = new AbortController();
  speakAbort = abort;

  const body = {
    text,
    avatar_id: avatarId,
    tts_backend: cfg.get<string>("avatar.ttsBackend", "qwen3"),
    language: cfg.get<string>("avatar.language", "ja"),
    speaker: cfg.get<string>("avatar.speaker", "") || undefined,
    speed: 1.0,
    fps: 25,
  };

  try {
    const res = await fetch(`${auth.base}/api/voice/ditto/tts/stream/avatar`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${auth.token}` },
      body: JSON.stringify(body),
      signal: abort.signal,
    });
    if (!res.ok || !res.body) {
      ui.note(`avatar: HTTP ${res.status} from ditto stream`);
      return;
    }
    ui.avatarBegin();
    const reader = res.body.getReader();
    let acc = new Uint8Array(0);
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value && value.length) {
        const merged = new Uint8Array(acc.length + value.length);
        merged.set(acc);
        merged.set(value, acc.length);
        acc = merged;
      }
      while (acc.length >= 5) {
        const type = acc[0];
        const len = acc[1] * 16777216 + acc[2] * 65536 + acc[3] * 256 + acc[4];
        if (acc.length < 5 + len) {
          break;
        }
        const payload = acc.subarray(5, 5 + len);
        acc = acc.slice(5 + len);
        ui.avatarChunk(type, Buffer.from(payload).toString("base64"));
      }
    }
    ui.avatarEnd();
  } catch (e) {
    const err = e as { name?: string; message?: string };
    if (err?.name !== "AbortError") {
      ui.note(`avatar error: ${err?.message ?? String(e)}`);
    }
  }
}

// ---- worktree review / apply (#13) -----------------------------------------

function worktreePath(): string | undefined {
  if (!engineState) {
    vscode.window.showWarningMessage("Kotonia: no active session.");
    return undefined;
  }
  if (!engineState.hello?.is_worktree) {
    vscode.window.showInformationMessage(
      "Kotonia: in-place mode — the agent's edits are already in your workspace (nothing to merge).",
    );
    return undefined;
  }
  return engineState.hello.workspace_root;
}

async function reviewChanges(): Promise<void> {
  const wt = worktreePath();
  if (!wt) {
    return;
  }
  await runExec("git", ["add", "-A", "-N"], wt);
  const { stdout } = await runExec("git", ["diff", "HEAD"], wt);
  if (!stdout.trim()) {
    vscode.window.showInformationMessage("Kotonia: the agent hasn't changed any files yet.");
    return;
  }
  const doc = await vscode.workspace.openTextDocument({ content: stdout, language: "diff" });
  await vscode.window.showTextDocument(doc, { preview: false });
}

async function applyChanges(): Promise<void> {
  const wt = worktreePath();
  if (!wt) {
    return;
  }
  const folder = pickWorkspaceFolder();
  if (!folder) {
    vscode.window.showErrorMessage("Kotonia: no workspace folder to apply into.");
    return;
  }
  await runExec("git", ["add", "-A", "-N"], wt);
  const { stdout: patch } = await runExec("git", ["diff", "HEAD", "--binary"], wt);
  if (!patch.trim()) {
    vscode.window.showInformationMessage("Kotonia: no agent changes to apply.");
    return;
  }
  const confirm = await vscode.window.showWarningMessage(
    "Apply the agent's changes to your working copy? Review the diff first (Kotonia: Review Agent Changes).",
    { modal: true },
    "Apply",
  );
  if (confirm !== "Apply") {
    return;
  }
  const tmp = path.join(os.tmpdir(), `kotonia-apply-${Date.now()}.patch`);
  fs.writeFileSync(tmp, patch);
  const { code, stderr } = await runExec("git", ["apply", "--3way", tmp], folder.uri.fsPath);
  try {
    fs.unlinkSync(tmp);
  } catch {
    /* ignore */
  }
  if (code === 0) {
    vscode.window.showInformationMessage(
      "Kotonia: applied the agent's changes to your workspace. Review & commit as usual.",
    );
    ui.note("applied agent changes to workspace", engineState?.lastTurnId);
  } else {
    vscode.window.showErrorMessage(`Kotonia: git apply failed — ${stderr.trim() || "conflict"}`);
  }
}

// ---- login (device-code) ---------------------------------------------------

function runLogin(): void {
  const folder = pickWorkspaceFolder();
  if (!folder) {
    vscode.window.showErrorMessage("Kotonia: open a folder first.");
    return;
  }
  const cfg = vscode.workspace.getConfiguration("kotonia");
  const binary = resolveBinary(cfg.get<string>("enginePath", "kotonia-cli"), folder);
  output.appendLine("[login] starting device-code flow");

  const child = spawn(binary, ["login"], { cwd: folder.uri.fsPath, env: process.env });
  let verifyUri: string | undefined;
  let sawCodeLabel = false;
  let buf = "";

  const handleLine = (line: string): void => {
    output.appendLine(`[login] ${line}`);
    const url = line.match(/https?:\/\/\S+/);
    if (!verifyUri && url) {
      // Hold the verification URL until we also have the code, then open the
      // page with the code prefilled (`?code=…`) so the user just clicks
      // Approve — no manual typing.
      verifyUri = url[0];
      return;
    }
    if (/enter this code/i.test(line)) {
      sawCodeLabel = true;
      return;
    }
    if (sawCodeLabel && line.trim()) {
      const code = line.trim();
      sawCodeLabel = false;
      const openUrl = verifyUri
        ? `${verifyUri}${verifyUri.includes("?") ? "&" : "?"}code=${encodeURIComponent(code)}`
        : undefined;
      void vscode.env.clipboard.writeText(code);
      if (openUrl) {
        void vscode.env.openExternal(vscode.Uri.parse(openUrl));
      }
      promptLoginApproval(code, openUrl);
    }
  };

  child.stdout.on("data", (d: Buffer) => {
    buf += d.toString();
    let idx: number;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      handleLine(line);
    }
  });
  child.stderr.on("data", (d: Buffer) => output.append(d.toString()));
  child.on("error", (e) =>
    vscode.window.showErrorMessage(`Kotonia: login failed to start — ${e.message}`),
  );
  child.on("exit", (code) => {
    if (code === 0) {
      vscode.window.showInformationMessage("Kotonia: signed in. Press New Chat (＋) to start.");
    } else {
      vscode.window.showErrorMessage(
        "Kotonia: login did not complete. See the “Kotonia Agent” output channel.",
      );
    }
  });
}

/** A sticky (button-bearing) notification showing the login code, so it stays
 * put while the user switches to the browser. With the prefilled URL the user
 * usually only needs to click Approve; the buttons are a fallback. */
function promptLoginApproval(code: string, url?: string): void {
  void vscode.window
    .showInformationMessage(
      `Kotonia ログイン: ブラウザで「Approve」を押すとサインイン完了です。コード ${code} は自動入力＆クリップボードにコピー済み（未対応環境では貼り付けてください）。`,
      "コードをコピー",
      "ログインページを開く",
    )
    .then((choice) => {
      if (choice === "コードをコピー") {
        void vscode.env.clipboard.writeText(code);
      } else if (choice === "ログインページを開く" && url) {
        void vscode.env.openExternal(vscode.Uri.parse(url));
      }
    });
}

// ---- helpers ---------------------------------------------------------------

function languageInstruction(): string {
  const lang = vscode.workspace.getConfiguration("kotonia").get<string>("language", "ja").trim();
  if (!lang) {
    return "";
  }
  if (lang === "ja") {
    return "デフォルトでは日本語で回答してください。ユーザーが他の言語で書いた場合は、その言語で回答してください。";
  }
  return `Reply in "${lang}" by default. If the user writes in another language, reply in that language instead.`;
}

function leadingToken(command: string): string {
  const toks = command.trim().split(/\s+/);
  for (const t of toks) {
    if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(t)) {
      continue;
    }
    return t;
  }
  return toks[0] || "";
}

async function openFileAtLine(file: string, line: number): Promise<void> {
  const folder = pickWorkspaceFolder();
  let uri: vscode.Uri;
  if (path.isAbsolute(file)) {
    uri = vscode.Uri.file(file);
  } else if (folder) {
    uri = vscode.Uri.joinPath(folder.uri, file);
  } else {
    return;
  }
  try {
    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc, { preview: true });
    const pos = new vscode.Position(Math.max(0, line - 1), 0);
    editor.selection = new vscode.Selection(pos, pos);
    editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
  } catch {
    vscode.window.showWarningMessage(`Kotonia: could not open ${file}`);
  }
}

function runExec(
  cmd: string,
  args: string[],
  cwd: string,
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    execFile(cmd, args, { cwd, maxBuffer: 32 * 1024 * 1024 }, (err, stdout, stderr) => {
      const code =
        err && typeof (err as unknown as { code?: number }).code === "number"
          ? ((err as unknown as { code: number }).code as number)
          : err
            ? 1
            : 0;
      resolve({ code, stdout: stdout || "", stderr: stderr || "" });
    });
  });
}

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

async function gatherEnv(): Promise<Record<string, string>> {
  const env: Record<string, string> = {};
  const kotonia = await extContext.secrets.get(SECRET_KOTONIA);
  if (kotonia) {
    env.KOTONIA_API_KEY = kotonia;
  }
  const deepseek = await extContext.secrets.get(SECRET_DEEPSEEK);
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

async function promptSecret(key: string, label: string): Promise<void> {
  const value = await vscode.window.showInputBox({
    prompt: `Enter your ${label}`,
    password: true,
    ignoreFocusOut: true,
  });
  if (value === undefined) {
    return;
  }
  if (value === "") {
    await extContext.secrets.delete(key);
    vscode.window.showInformationMessage(`Kotonia: cleared ${label}.`);
  } else {
    await extContext.secrets.store(key, value);
    vscode.window.showInformationMessage(`Kotonia: stored ${label}. Start a new chat to apply.`);
  }
}
