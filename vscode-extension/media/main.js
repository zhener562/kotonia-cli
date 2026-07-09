// @ts-check
// Webview frontend for the Kotonia Agent panel. Framework-free: it renders the
// engine event stream into #log and posts user actions back to the extension.
(function () {
  const vscode = acquireVsCodeApi();

  const statusEl = document.getElementById("status");
  const logEl = document.getElementById("log");
  const inputEl = /** @type {HTMLTextAreaElement} */ (document.getElementById("input"));
  const sendBtn = /** @type {HTMLButtonElement} */ (document.getElementById("send"));
  const cancelBtn = /** @type {HTMLButtonElement} */ (document.getElementById("cancel"));

  let busy = false;
  /** @type {HTMLElement | null} */
  let currentTurnEl = null;
  let currentTurnId = -1;

  function scrollToEnd() {
    logEl.scrollTop = logEl.scrollHeight;
  }

  /** Append a fresh element to the current turn group (creating it if needed). */
  function turnGroup(turnId) {
    if (turnId !== currentTurnId || !currentTurnEl) {
      currentTurnId = turnId;
      currentTurnEl = document.createElement("div");
      currentTurnEl.className = "turn";
      logEl.appendChild(currentTurnEl);
    }
    return currentTurnEl;
  }

  function addRow(turnId, cls, html) {
    const row = document.createElement("div");
    row.className = "row " + cls;
    row.innerHTML = html;
    turnGroup(turnId).appendChild(row);
    scrollToEnd();
    return row;
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function addUserBubble(text) {
    // User turns render before the engine assigns a turn_id; put them in their
    // own group so the following engine events attach to the next group.
    currentTurnEl = null;
    currentTurnId = -1;
    const g = document.createElement("div");
    g.className = "turn user-turn";
    const row = document.createElement("div");
    row.className = "row user";
    row.textContent = text;
    g.appendChild(row);
    logEl.appendChild(g);
    scrollToEnd();
  }

  function setBusy(next) {
    busy = next;
    sendBtn.disabled = next;
    cancelBtn.disabled = !next;
    inputEl.disabled = next;
  }

  function renderEngine(msg) {
    switch (msg.type) {
      case "hello": {
        const tag = msg.is_worktree ? "worktree" : "in-place";
        statusEl.textContent =
          `${msg.model} · ${msg.backend} · ${msg.tool_mode} · ${msg.approval_mode} · ${tag}` +
          (msg.session_id ? ` · session ${msg.session_id}` : "");
        statusEl.classList.add("ready");
        break;
      }
      case "iteration_start":
        addRow(msg.turn_id, "iter", `<span class="dim">iter ${msg.iteration}/${msg.max}</span>`);
        break;
      case "llm_thinking":
        addRow(msg.turn_id, "thinking", `<span class="dim">· thinking…</span>`);
        break;
      case "bash":
        addRow(msg.turn_id, "bash", `<span class="prompt">$</span> <code>${esc(msg.command)}</code>`);
        break;
      case "bash_skipped":
        addRow(
          msg.turn_id,
          "skipped",
          `<span class="badge warn">skipped</span> ${esc(msg.reason)}<br/><code>${esc(msg.command)}</code>`,
        );
        break;
      case "observation": {
        const r = msg.result || {};
        const flags = [];
        if (r.timed_out) flags.push(`<span class="badge err">timed out</span>`);
        if (r.truncated) flags.push(`<span class="badge warn">truncated</span>`);
        const codeBadge = `<span class="badge ${r.exit_code === 0 ? "ok" : "err"}">exit ${r.exit_code}</span>`;
        const body = (r.combined || "").trim();
        const long = body.split("\n").length > 20;
        const pre = `<pre class="${long ? "collapsible" : ""}">${esc(body)}</pre>`;
        addRow(msg.turn_id, "obs", `${codeBadge} ${flags.join(" ")}${body ? pre : ""}`);
        break;
      }
      case "final":
        addRow(msg.turn_id, "final", `<div class="final-answer">${esc(msg.answer)}</div>`);
        break;
      case "malformed":
        addRow(msg.turn_id, "malformed", `<span class="dim">malformed output — retrying</span>`);
        break;
      case "error":
        addRow(msg.turn_id, "error", `<span class="badge err">error</span> ${esc(msg.message)}`);
        break;
      case "done":
        addRow(
          msg.turn_id,
          "done",
          `<span class="dim">— done after ${msg.iterations} iter · ${msg.success ? "✓" : "✗"}</span>`,
        );
        setBusy(false);
        break;
      case "approval_request":
        renderApproval(msg);
        break;
    }
  }

  function renderApproval(msg) {
    const row = addRow(
      msg.turn_id,
      "approval",
      `<div class="approval-head"><span class="badge warn">approval</span> ${esc(msg.reason)}</div>` +
        `<code>${esc(msg.command)}</code>` +
        `<div class="approval-actions">` +
        `<label class="remember"><input type="checkbox" class="remember-cb" /> remember for session</label>` +
        `<button class="approve">Approve</button>` +
        `<button class="deny secondary">Deny</button>` +
        `</div>`,
    );
    const cb = row.querySelector(".remember-cb");
    const finish = (approve) => {
      const remember = !!(cb && cb.checked);
      row.querySelectorAll("button").forEach((b) => (b.disabled = true));
      row.querySelector(".approval-actions").insertAdjacentHTML(
        "beforeend",
        `<span class="dim decided">${approve ? "approved" : "denied"}${remember ? " (remembered)" : ""}</span>`,
      );
      vscode.postMessage({ kind: "approval", approvalId: msg.approval_id, approve, remember });
    };
    row.querySelector(".approve").addEventListener("click", () => finish(true));
    row.querySelector(".deny").addEventListener("click", () => finish(false));
  }

  function submit() {
    const text = inputEl.value.trim();
    if (!text || busy) return;
    addUserBubble(text);
    inputEl.value = "";
    setBusy(true);
    vscode.postMessage({ kind: "send", text });
  }

  sendBtn.addEventListener("click", submit);
  cancelBtn.addEventListener("click", () => vscode.postMessage({ kind: "cancel" }));
  inputEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  });

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (data.kind === "engine") {
      renderEngine(data.msg);
    } else if (data.kind === "busy") {
      setBusy(data.busy);
    }
  });

  vscode.postMessage({ kind: "ready" });
})();
