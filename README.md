# kotonia-cli

A local shell agent driven by self-hosted or hosted LLMs. ReAct-style loop
with **native tool calling** on models that support it (Gemma 4 26B
Uncensored, DeepSeek API, kotonia hosted API) and a delimiter-based
fallback on models that don't (DeepSeek-V4-Flash on `llama.cpp`).

It runs commands on your machine through `bash`, searches the web through
a local Searxng instance, and extracts article bodies with `trafilatura`.

```
─────────────────────────────────────────────
kotonia-cli
  model     : gemma4-26b-uncensored (local)
  tools     : native (bash + web_search + fetch_url)
  approval  : allowlist
  workspace : /tmp/kotonia-agent-xyz  (worktree)
─────────────────────────────────────────────
```

## Install

### Build from source

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

## Model backends

`kotonia-cli` picks the backend from the `--model` id:

| `--model`                  | Backend                              | Tool calling |
| -------------------------- | ------------------------------------ | ------------ |
| `deepseek-v4-flash`        | local llama.cpp on `:8898`           | delimiter    |
| `gemma4-26b-uncensored`    | local vLLM on `:8899`                | native       |
| `deepseek-chat`            | DeepSeek API (V4-Flash class)        | native       |
| `deepseek-reasoner`        | DeepSeek API (V4-Pro reasoning)      | native       |
| `kotonia-v4-flash`         | hosted — kotonia.ai `/api/v1/chat`   | delimiter    |
| `kotonia-gemma4-26b`       | hosted — kotonia.ai `/api/v1/chat`   | native       |

The hosted route requires `KOTONIA_API_KEY` (mint at
<https://kotonia.ai/api-manager>). The DeepSeek API route requires
`DEEPSEEK_API_KEY`. The local routes assume the matching server is up on
the listed port — there's a smoke note in [`docs/local-servers.md`].

## Usage

```sh
# One-shot
kotonia-cli "explain the http error handling in src/router.rs"

# Interactive REPL (no prompt argument)
kotonia-cli

# Switch backend
kotonia-cli --model kotonia-gemma4-26b "summarise the README"

# Resume a prior session
kotonia-cli --list-sessions
kotonia-cli --resume 20260621-205141-9c4a
```

### Approval modes

`--approval` controls how `bash` commands are gated:

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

## Building the agent

The agent loop runs `iter ← provider call → tool dispatch → observation`
until the model returns a final answer or hits `--max-iterations`. On
native-tool backends each iteration is a single `tools` round-trip; on
delimiter backends each iteration parses one `<<<BASH>>>` or
`<<<FINAL_ANSWER>>>` block out of the model's free text.

The tool catalogue is intentionally tiny:

- **`bash(command)`** — single shell command, captured into the next
  observation.
- **`web_search(query, max_results=5)`** — Searxng SERP preview.
- **`fetch_url(url, max_chars?)`** — main-article extraction via
  `trafilatura`, returned as Markdown.

## License

MIT — see [`LICENSE`](LICENSE).
