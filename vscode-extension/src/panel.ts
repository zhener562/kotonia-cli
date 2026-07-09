// Webview panel: renders the engine's event stream and collects user input.
//
// The panel is pure view + input plumbing. It knows nothing about the child
// process — `extension.ts` bridges panel actions to the `KotoniaEngine` and
// engine messages back to the panel.

import * as vscode from "vscode";
import { Outbound } from "./protocol";

/** Actions the webview sends up to the extension host. */
export type PanelAction =
  | { kind: "send"; text: string }
  | { kind: "approval"; approvalId: number; approve: boolean; remember: boolean }
  | { kind: "cancel" }
  | { kind: "ready" };

export class ChatPanel {
  public readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];

  constructor(
    extensionUri: vscode.Uri,
    private readonly onAction: (a: PanelAction) => void,
    onDispose: () => void,
  ) {
    this.panel = vscode.window.createWebviewPanel(
      "kotoniaAgent",
      "Kotonia Agent",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      },
    );

    this.panel.webview.html = this.html(extensionUri, this.panel.webview);

    this.panel.webview.onDidReceiveMessage(
      (a: PanelAction) => this.onAction(a),
      undefined,
      this.disposables,
    );

    this.panel.onDidDispose(
      () => {
        onDispose();
        this.dispose();
      },
      undefined,
      this.disposables,
    );
  }

  /** Forward an engine protocol message into the webview for rendering. */
  postEngineMessage(msg: Outbound): void {
    this.panel.webview.postMessage({ kind: "engine", msg });
  }

  /** Tell the webview to enable/disable the input (serial turn gating). */
  setBusy(busy: boolean): void {
    this.panel.webview.postMessage({ kind: "busy", busy });
  }

  reveal(): void {
    this.panel.reveal();
  }

  dispose(): void {
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
  }

  private html(extensionUri: vscode.Uri, webview: vscode.Webview): string {
    const nonce = getNonce();
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, "media", "main.js"),
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, "media", "main.css"),
    );
    const csp = [
      `default-src 'none'`,
      `img-src ${webview.cspSource}`,
      `style-src ${webview.cspSource}`,
      `script-src 'nonce-${nonce}'`,
    ].join("; ");

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${styleUri}" rel="stylesheet" />
  <title>Kotonia Agent</title>
</head>
<body>
  <div id="status" class="status">engine starting…</div>
  <div id="log" class="log"></div>
  <div id="composer" class="composer">
    <textarea id="input" rows="2" placeholder="Ask the agent to do something…"></textarea>
    <div class="composer-buttons">
      <button id="send" title="Send (Ctrl/Cmd+Enter)">Send</button>
      <button id="cancel" class="secondary" disabled>Cancel</button>
    </div>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < 32; i++) {
    out += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return out;
}
