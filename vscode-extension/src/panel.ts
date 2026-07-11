// Editor-area webview panel — an optional wider surface for the chat, opened
// via "Kotonia: Open Agent in Editor". Mirrors the sidebar view: same HTML,
// same action protocol. extension.ts broadcasts engine messages to both.

import * as vscode from "vscode";
import { PanelAction, renderChatHtml } from "./webview";
import { Outbound } from "./protocol";

export class ChatPanel {
  private readonly panel: vscode.WebviewPanel;

  constructor(
    extensionUri: vscode.Uri,
    onAction: (a: PanelAction) => void,
    onDispose: () => void,
  ) {
    this.panel = vscode.window.createWebviewPanel(
      "kotoniaAgentEditor",
      "Kotonia Agent",
      vscode.ViewColumn.Active,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      },
    );
    this.panel.iconPath = vscode.Uri.joinPath(extensionUri, "media", "icon.png");
    this.panel.webview.html = renderChatHtml(this.panel.webview, extensionUri);
    this.panel.webview.onDidReceiveMessage((a: PanelAction) => onAction(a));
    this.panel.onDidDispose(() => onDispose());
  }

  postEngineMessage(msg: Outbound): void {
    this.panel.webview.postMessage({ kind: "engine", msg });
  }
  note(text: string, turnId?: number): void {
    this.panel.webview.postMessage({ kind: "note", text, turnId });
  }
  setBusy(busy: boolean): void {
    this.panel.webview.postMessage({ kind: "busy", busy });
  }
  reset(): void {
    this.panel.webview.postMessage({ kind: "reset" });
  }
  avatarBegin(): void {
    this.panel.webview.postMessage({ kind: "avatarBegin" });
  }
  avatarChunk(chunkType: number, data: string): void {
    this.panel.webview.postMessage({ kind: "avatarChunk", chunkType, data });
  }
  avatarEnd(): void {
    this.panel.webview.postMessage({ kind: "avatarEnd" });
  }
  avatarStop(): void {
    this.panel.webview.postMessage({ kind: "avatarStop" });
  }
  reveal(): void {
    this.panel.reveal();
  }
  dispose(): void {
    this.panel.dispose();
  }
}
