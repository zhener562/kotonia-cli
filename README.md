# kotonia-cli

A local shell agent driven by hosted or self-hosted LLMs. Two engines:

- **ReAct** — kotonia-cli's own loop over any OpenAI-compatible
  `/chat/completions` endpoint. Native tool calling (`bash` /
  `web_search` / `fetch_url`). Built-in shortcuts for the
  [kotonia.ai](https://kotonia.ai) hosted API and the DeepSeek API;
  custom providers via `~/.kotonia/providers.json`.
- **Claude Code** — drives the local `claude` binary as a subprocess in
  headless `stream-json` mode. Lets a daemon on your machine serve "act
  as if I'm running `claude` from this shell" UX over a WS to the
  kotonia.ai `/agent` web console.

```
─────────────────────────────────────────────
kotonia-cli
  model     : kotonia-llm-basic (kotonia)
  tools     : native (bash + web_search + fetch_url)
  approval  : allowlist
  workspace : /tmp/kotonia-agent-xyz  (worktree)
─────────────────────────────────────────────
```

## Install

Prebuilt binaries for Linux (x64 / arm64), macOS (Apple Silicon) and
Windows (x64) are on the
[Releases page](https://github.com/zhener562/kotonia-cli/releases/latest):
download the archive for your platform, extract `kotonia-cli`
(`kotonia-cli.exe` on Windows) and put it somewhere on your `PATH`.
Under WSL, use the Linux build.

Or build from source:

```sh
git clone https://github.com/zhener562/kotonia-cli
cd kotonia-cli
cargo install --path .
```

The binary lands in `~/.cargo/bin/kotonia-cli`.

### Web search + page fetch CLIs

The agent's `web_search` and `fetch_url` tools shell out to two small
Python wrappers shipped in `scripts/`. Put them on your `PATH`:

```sh
ln -s "$(pwd)/scripts/web-search"  ~/.local/bin/web-search
ln -s "$(pwd)/scripts/fetch-url"   ~/.local/bin/fetch-url
```

Dependencies:

- **web-search** requires a running Searxng instance at
  `http://127.0.0.1:8888` (overridable via `SEARXNG_URL`). The simplest
  recipe is the Searxng docker compose at
  <https://github.com/searxng/searxng-docker>.
- **fetch-url** requires the `trafilatura` CLI on your `PATH`:

  ```sh
  uv tool install trafilatura
  # or: pipx install trafilatura
  ```

Both tools degrade gracefully — `kotonia-cli` will keep working without
them; only the matching tool call will return an error to the model.

## Authentication

`kotonia-cli login` pairs your machine with a kotonia.ai account via an
OAuth-style device-code flow:

```sh
kotonia-cli login
# prints a URL + code; approve from a logged-in browser tab.
# Writes ~/.kotonia/daemon.json {server, device_id, device_token}.

kotonia-cli auth-status --validate
# verifies that a present token is still accepted by the server.

kotonia-cli logout
# removes the shared device credential.
```

The stored `device_token` is reused as the bearer for **both** the
daemon WS and the public `/api/v1/*` API — one login covers both
surfaces. You can still mint a separate `kotonia_…` API key at
<https://kotonia.ai/api-manager> if you want a key that's not bound to a
paired device.

For the DeepSeek-hosted API, set `DEEPSEEK_API_KEY` in your env.

## Approval channel (phone confirm)

When you run the daemon, kotonia.ai can drive your machine through it.
That's the point — but it also means a future compromise of kotonia.ai
would let an attacker run arbitrary commands here. To make that
structurally impossible, `kotonia-cli daemon` gates every first task
from a new browser session through an **independent push-notification
channel** (Telegram today; Discord planned). The bot token lives on
*your* machine; kotonia.ai never sees it; an attacker who owns the
backend still can't forge an Approve.

```sh
kotonia-cli pair-notifier telegram
```

The flow:

1. In Telegram, message `@BotFather` → `/newbot` → follow prompts → copy the
   bot token.
2. Paste the token at the `kotonia-cli pair-notifier` prompt.
3. The CLI prints a one-time pairing code. Open your new bot in Telegram
   and send `/start <CODE>`.
4. Done — `~/.kotonia/notifier.json` now holds `{bot_token, chat_id}` at
   0600 perms.

Once paired, every first task from a new browser tab sends an Approve /
Deny prompt to your phone. Approving extends 24h of silent trust for
that tab; subsequent tasks run with no prompt. The daemon **refuses to
start** without a paired notifier — pass `--no-notifier` only if you
fully trust both kotonia.ai and every browser session that drives this
daemon.

## Model providers

`--model` picks the model id; `--provider` (optional) forces a specific
provider. When `--provider` is omitted the provider is inferred from the
model id.

Built-in providers:

| `--provider` | Endpoint                       | Default model         | Auth                              |
| ------------ | ------------------------------ | --------------------- | --------------------------------- |
| `kotonia`    | kotonia.ai `/api/v1`           | `kotonia-llm-basic`  | `KOTONIA_API_KEY` or `daemon.json` device_token |
| `deepseek`   | api.deepseek.com               | `deepseek-chat`       | `DEEPSEEK_API_KEY`                |

DeepSeek's `:thinking` suffix on `deepseek-chat:thinking` /
`deepseek-reasoner:thinking` is forwarded as the upstream `thinking` +
`reasoning_effort` body knob.

### Custom providers (`~/.kotonia/providers.json`)

Any OpenAI-compatible endpoint can be added without code changes. Example:

```json
{
  "providers": {
    "openai": {
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "default_model": "gpt-5",
      "max_tokens_param": "max_completion_tokens",
      "models": ["gpt-5", "gpt-4.1"]
    },
    "local-llama": {
      "base_url": "http://127.0.0.1:8080/v1",
      "default_model": "llama-3.3-70b"
    }
  }
}
```

Then:

```sh
kotonia-cli --provider openai --model gpt-5 "summarise main.rs"
kotonia-cli --provider local-llama "what does router.rs do?"
```

The `models` array lets you skip `--provider` for those ids
(`kotonia-cli --model gpt-5 ...` infers `openai`).

## Usage

```sh
# One-shot ReAct (defaults to kotonia-llm-basic)
kotonia-cli "explain the http error handling in src/router.rs"

# Interactive REPL
kotonia-cli

# Switch model / provider
kotonia-cli --model deepseek-reasoner:thinking "design a rate limiter"
kotonia-cli --provider openai --model gpt-5 "summarise the README"

# Claude Code engine — drive the local `claude` binary headlessly
kotonia-cli --engine claude-code "explain main.rs"

# Resume a prior session
kotonia-cli --list-sessions
kotonia-cli --resume 20260621-205141-9c4a
```

### Daemon mode

After `kotonia-cli login` **and** `kotonia-cli pair-notifier telegram`
(see [Approval channel](#approval-channel-phone-confirm)), run the
daemon to expose your machine to the kotonia.ai `/agent` web console:

```sh
kotonia-cli daemon                       # default model + ReAct
kotonia-cli daemon --engine claude-code  # remote Claude Code
kotonia-cli daemon --in-place            # don't create a worktree per task
kotonia-cli daemon --no-notifier         # opt out of phone confirm (dangerous)
```

Tasks issued from the web UI stream `Event`s back over WS (iteration
ticks, tool invocations, observations, final answers, errors). The
first task from each new browser tab triggers a Telegram push asking
you to approve the session for 24h.

### Home (resident agent, experimental)

A persistent workspace of Markdown files the agent reads at the start of every
conversation, plus a heartbeat and reminders that keep working across the
machine sleeping.

```bash
kotonia-cli home init                       # ~/.kotonia/home (or $KOTONIA_HOME_DIR)
kotonia-cli --home                          # REPL with SOUL / USER / MEMORY … loaded
kotonia-cli home remind 15:00 "send the invoice"
kotonia-cli home heartbeat --every 30       # runs the `- ` items in HEARTBEAT.md
kotonia-cli home hours 09:00-21:30          # heartbeat stays quiet outside these
kotonia-cli home list
kotonia-cli home run                        # stay resident (Ctrl-C to stop)
kotonia-cli home tick --dry-run             # what would fire right now
```

- Due times are wall-clock and checked every minute, so a sleeping machine
  catches up on wake: missed heartbeats collapse into one run, a reminder more
  than 6 hours late is recorded but not pushed. Nothing wakes the machine.
- An empty `HEARTBEAT.md` means no model call at all.
- Heartbeat runs use the `allowlist` approval policy with nobody to approve:
  anything that needs a human is refused and listed in `feed/<date>.jsonl`.
  The agent is told not to send, buy, delete or share during a heartbeat.
- Messages are written to `feed/<date>.jsonl`, printed, and pushed to the
  paired phone notifier (`pair-notifier`) unless `--no-notifier`.
- One `home run` per home directory (file lock). Running it on two machines
  against the same account will double up; there is no cross-device lease yet.

### Connectors (mail, API keys)

The resident agent (`--home`, heartbeat) can ask for an account **in the
conversation, at the moment it needs one**: it calls `request_connector`, and
the terminal shows a one-screen card (what it is for, what connecting means, a
link to create an app password) with a hidden prompt for the secret.

```bash
kotonia-cli connector add gmail          # same card, from the shell
kotonia-cli connector list | check <id> | remove <id>
kotonia-cli mail recent [--new | --days N] [--limit N]
kotonia-cli mail read <uid>
kotonia-cli secret run <id> -- 'curl -H "Authorization: Bearer $KOTONIA_TTS_API_KEY" …'
```

- Mail is IMAP over TLS, **read-only**: `EXAMINE` + `BODY.PEEK`, so nothing is
  sent, deleted or marked as read. Gmail and iCloud need an app password.
  Japanese charsets (ISO-2022-JP etc.) are decoded.
- Secrets live in `~/.kotonia/vault/` (0600 files in a 0700 dir) and are typed
  at a hidden prompt, never into the chat. Any stored secret that appears in a
  command's output is replaced with `[redacted secret]` before the agent, the
  UI or the session log sees it, and commands touching the vault path need
  approval even in `allowlist` mode. This is exact-match redaction against a
  shell agent running as you — a value transformed on the way out (base64 …)
  is not caught.
- Mail content is passed to the model between "untrusted content" markers, and
  the prompt forbids acting on instructions found in it.
- Cards are shown by the terminal REPL / one-shot only. `--serve` (VS Code),
  the daemon and heartbeats answer "nobody can fill a card right now", and the
  agent tells the user to run `connector add`.

### Web access in background runs

A heartbeat runs with nobody watching, so text it reads (mail, pages) could
try to make it carry data out in a URL. By default every network tool is
blocked there — `curl`/`wget` were already gated by the allowlist, and
`fetch_url` / `web_search` are gated too in unattended runs.

```bash
kotonia-cli web allow wttr.in        # approve a site (subdomains included)
kotonia-cli web list | remove <site>
```

- During a heartbeat, `fetch_url` may open an approved site with **GET, no
  `?query`, no `#fragment`, path ≤ 160 chars**. The page is fetched by
  kotonia-cli itself: redirects are followed only to other approved,
  query-less URLs, no cookies are kept, and the text comes back fenced as
  untrusted web content. Everything else is refused and recorded in the feed
  with the `web allow` command to run if it should have been allowed.
- `web_search` stays blocked in background runs: the query itself would leave.
- In `--home` conversations, opening a site that is not approved asks
  `[y] once / [a] always / [N] no`; `a` adds it to the list.
- Data packed into the *path* of an approved site still gets out; it only
  matters if someone else can read that site's request logs. Approve sites
  you trust, not "the web".

### Approval modes

`--approval` controls how `bash` commands are gated in the **ReAct**
engine. The Claude Code engine ignores this and runs with
`--dangerously-skip-permissions` (trust the worktree boundary):

- **`all`** — every command pops a `[y/N]` prompt before running.
- **`allowlist`** (default) — read-only / build / test families run
  silently; anything destructive (`rm`, `git push --force`, `curl`, …)
  asks first.
- **`auto`** — never ask. Dangerous; use only with `--in-place=false`
  (the default — see below).

### Workspace isolation

By default `kotonia-cli` creates a fresh `git worktree` under `/tmp/` and
runs everything there. Your real working copy is untouched until you
explicitly merge.

Pass `--in-place` to disable that and run inside your launch `cwd`.
Machine frontends can combine `--keep-worktree` with `--resume`; protocol
v2 reattaches to the saved worktree when it still exists, so closing and
reopening a UI does not discard uncommitted agent edits.

## The ReAct loop

Each iteration is `provider call → tool dispatch → observation` until
the model returns a final answer or hits `--max-iterations`. Native
tool calling is the default; pass `--force-delimiter` to drive a model
through the legacy `<<<BASH>>>` / `<<<FINAL_ANSWER>>>` text loop (debug
hook).

The tool catalogue is intentionally tiny:

- **`bash(command)`** — single shell command, captured into the next
  observation.
- **`web_search(query, max_results=5)`** — Searxng SERP preview.
- **`fetch_url(url, max_chars?)`** — main-article extraction via
  `trafilatura`, returned as Markdown.

## Telemetry

When you are logged in to kotonia.ai (`kotonia-cli login`, or
`KOTONIA_API_KEY` is set), each agent run sends two small events — `start`
and `end` — to the kotonia.ai server only, so we can see where runs fail
before they ever reach a model (no `claude` binary, missing API key,
worktree trouble). Requests to kotonia.ai also carry an
`x-kotonia-client` header like `cli/0.1.14 (linux-x86_64; wsl)`.

What is sent: a random per-run id, the engine (`react` / `claude-code`),
whether the model is kotonia's, how the run was driven (one-shot / REPL /
pipe / serve), the outcome, a fixed failure label (e.g. `claude_spawn`,
`llm_auth`), the duration, and the version / OS / WSL from the header.
**Never** prompts, replies, file paths or error text. Third-party
providers (OpenAI, DeepSeek, …) never receive the header. See
[`src/telemetry.rs`](src/telemetry.rs).

Turn it off with `DO_NOT_TRACK=1` or `KOTONIA_TELEMETRY=0`.

## License

MIT — see [`LICENSE`](LICENSE).
