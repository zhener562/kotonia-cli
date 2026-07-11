// Shared webview scaffolding used by the sidebar view (view.ts). Keeps the
// HTML shell, CSP/nonce, and the action protocol in one place.

import * as vscode from "vscode";

/** Actions the webview sends up to the extension host. */
export type PanelAction =
  | { kind: "send"; text: string }
  | {
      kind: "approval";
      approvalId: number;
      approve: boolean;
      remember: boolean;
      command: string;
    }
  | { kind: "cancel" }
  | { kind: "open"; file: string; line: number }
  | { kind: "ready" };

export function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < 32; i++) {
    out += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return out;
}

/** Build the chat webview HTML. Loads media/main.js + main.css via
 * webview-safe URIs and locks a CSP with a per-render nonce. */
export function renderChatHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const nonce = getNonce();
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "main.js"));
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "main.css"));
  const csp = [
    `default-src 'none'`,
    `img-src ${webview.cspSource} blob:`,
    `style-src ${webview.cspSource}`,
    `script-src 'nonce-${nonce}'`,
    `media-src blob:`,
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
  <div id="avatar-wrap" class="avatar-wrap">
    <canvas id="avatar" class="avatar"></canvas>
  </div>
  <div id="log" class="log"></div>
  <div id="composer" class="composer">
    <textarea id="input" rows="2" placeholder="Ask the agent to do something…  (Enter to send · Shift+Enter for newline)"></textarea>
    <div class="composer-buttons">
      <button id="send" title="Send (Enter)">Send</button>
      <button id="cancel" class="secondary" disabled>Cancel</button>
    </div>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
}
