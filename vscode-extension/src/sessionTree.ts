// Sidebar session list (activity-bar "Sessions" view). Reads the engine's
// on-disk transcripts under ~/.kotonia/sessions/*.jsonl and shows them newest
// first; clicking one resumes it in the center chat panel. Claude-Code-style:
// the sidebar is history/management, the editor panel is the live chat.

import * as vscode from "vscode";
import * as os from "os";
import * as fs from "fs";
import * as path from "path";

export class SessionItem extends vscode.TreeItem {
  constructor(
    public readonly id: string,
    label: string,
    when: string,
    tooltip: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = when;
    this.tooltip = tooltip;
    this.contextValue = "kotoniaSession";
    this.iconPath = new vscode.ThemeIcon("comment-discussion");
    this.command = {
      command: "kotonia.openSession",
      title: "Resume Session",
      arguments: [id],
    };
  }
}

export class SessionTreeProvider implements vscode.TreeDataProvider<SessionItem> {
  private readonly changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;

  refresh(): void {
    this.changed.fire();
  }

  getTreeItem(item: SessionItem): vscode.TreeItem {
    return item;
  }

  getChildren(): SessionItem[] {
    const dir = path.join(os.homedir(), ".kotonia", "sessions");
    let files: string[];
    try {
      files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    } catch {
      return [];
    }
    const rows = files
      .map((f) => {
        const full = path.join(dir, f);
        let mtime = 0;
        try {
          mtime = fs.statSync(full).mtimeMs;
        } catch {
          /* ignore */
        }
        return { id: f.replace(/\.jsonl$/, ""), full, mtime };
      })
      .sort((a, b) => b.mtime - a.mtime);

    return rows.map((r) => {
      const meta = readSessionMeta(r.full);
      const label = meta.firstUser || r.id;
      const when = meta.when || new Date(r.mtime).toLocaleString();
      const tip = [r.id, meta.workspace].filter(Boolean).join("\n");
      return new SessionItem(r.id, label, when, tip);
    });
  }
}

/** Peek a transcript for a friendly label (first real user message, minus the
 * injected language hint) + started_at + workspace. Best-effort. */
function readSessionMeta(file: string): {
  firstUser?: string;
  when?: string;
  workspace?: string;
} {
  const out: { firstUser?: string; when?: string; workspace?: string } = {};
  let raw: string;
  try {
    raw = fs.readFileSync(file, "utf8");
  } catch {
    return out;
  }
  for (const line of raw.split("\n")) {
    if (!line.trim()) {
      continue;
    }
    let o: {
      kind?: string;
      role?: string;
      content?: string;
      started_at?: string;
      workspace?: string;
    };
    try {
      o = JSON.parse(line);
    } catch {
      continue;
    }
    if (o.kind === "session") {
      out.workspace = o.workspace;
      if (o.started_at) {
        const d = new Date(o.started_at);
        out.when = isNaN(d.getTime()) ? o.started_at : d.toLocaleString();
      }
    } else if (o.kind === "message" && o.role === "user" && typeof o.content === "string") {
      // Strip the leading one-line language instruction we inject (it's
      // separated from the real text by a blank line).
      const parts = o.content.split("\n\n");
      const text = (parts.length > 1 ? parts.slice(1).join(" ") : o.content).trim();
      if (text) {
        out.firstUser = text.slice(0, 72).replace(/\s+/g, " ");
        break;
      }
    }
  }
  return out;
}
