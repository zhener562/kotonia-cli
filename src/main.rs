//! kotonia-cli — local shell agent backed by a hosted or self-hosted LLM.
//!
//! Built-in providers (resolved by model id):
//!   - `kotonia-gemma4-26b` (default) — kotonia.ai /api/v1 chat (native tools)
//!   - `deepseek-chat`                — DeepSeek API (native tools)
//!   - `deepseek-reasoner`            — DeepSeek API (reasoning, native tools)
//!
//! Custom providers can be added via `~/.kotonia/providers.json` and selected
//! with `--provider <name> --model <id>` (any OpenAI-compatible endpoint).
//!
//! All built-in backends drive the model via native OpenAI-compatible tool
//! calling (`bash`, `web_search`, `fetch_url`). The legacy `<<<BASH>>>`
//! delimiter loop is reachable through `--force-delimiter` for debugging.
//!
//! When called with a prompt argument the CLI runs one task and exits.
//! When called with no prompt it drops into an interactive REPL — each
//! line is one user turn; the agent's conversation history persists across
//! turns so follow-ups land in context.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use kotonia_cli::agent::agent::{
    Agent, AgentConfig, ApprovalHandler, ApprovalOutcome, Event, EventSink,
};
use kotonia_cli::agent::approval::ApprovalMode;
use kotonia_cli::agent::claude_code::ClaudeCodeAgent;
use kotonia_cli::agent::dispatch::DispatchAgent;
use kotonia_cli::agent::history::{list_sessions, load_session_messages, HistoryStore};
use kotonia_cli::agent::provider::Provider;
use kotonia_cli::agent::worktree::AgentWorkspace;
use kotonia_cli::config as daemon_config;
use kotonia_cli::daemon::{self, DaemonConfig};
use kotonia_cli::login;
use kotonia_cli::notifier;
use kotonia_cli::serve;

#[derive(Parser, Debug)]
#[command(
    name = "kotonia-cli",
    about = "Local shell agent backed by a self-hosted LLM or the DeepSeek API.",
    version
)]
struct Cli {
    /// Optional subcommand. With no subcommand the rest of the args are
    /// parsed as a one-shot / REPL task (the default kotonia-cli UX).
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Args used when no subcommand is given.
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Pair this machine with a kotonia.ai account via the OAuth-style
    /// device-code flow: print a code, wait for the user to approve it from
    /// a logged-in browser tab, then persist device_id + device_token to
    /// ~/.kotonia/daemon.json so subsequent `daemon` runs need no flags.
    Login(LoginArgs),

    /// Run as a long-lived WS daemon that streams agent tasks issued from
    /// the paired kotonia.ai web UI. Reads credentials from
    /// ~/.kotonia/daemon.json (written by `login`) if not supplied via
    /// env / flag.
    Daemon(DaemonArgs),

    /// Pair this machine with a third-party messaging app (Telegram for
    /// now) so the daemon can ask for phone confirmation before trusting a
    /// new browser session. The approval channel is independent of
    /// kotonia.ai — a compromised backend cannot forge an approve signal.
    PairNotifier(PairNotifierArgs),
}

#[derive(Args, Debug)]
struct LoginArgs {
    /// HTTP(S) base of the kotonia.ai backend to pair with.
    #[arg(long, default_value = "https://kotonia.ai", env = "KOTONIA_API_BASE")]
    server: String,
}

#[derive(Args, Debug)]
struct PairNotifierArgs {
    /// Notifier kind. `telegram` is the only supported value today; `discord`
    /// is planned in a follow-up.
    kind: String,
}

#[derive(Args, Debug)]
struct DaemonArgs {
    /// HTTP(S) base of the kotonia.ai backend. WS endpoint is derived
    /// (https → wss, http → ws) and the path is fixed. Falls back to the
    /// `server` field of ~/.kotonia/daemon.json if not set.
    #[arg(long, env = "KOTONIA_API_BASE")]
    server: Option<String>,

    /// Device id the daemon was paired as. Falls back to
    /// ~/.kotonia/daemon.json.
    #[arg(long, env = "KOTONIA_DEVICE_ID")]
    device_id: Option<String>,

    /// Bearer token issued at pairing time. Falls back to
    /// ~/.kotonia/daemon.json. Sent as `Authorization: Bearer <token>`
    /// on the WS upgrade.
    #[arg(long, env = "KOTONIA_DEVICE_TOKEN", hide_env_values = true)]
    device_token: Option<String>,

    /// Model id for every task this daemon runs. Same surface as the
    /// one-shot CLI's `--model`. Default matches the one-shot default.
    #[arg(long, default_value = "kotonia-gemma4-26b")]
    model: String,

    /// Explicit provider name (`kotonia`, `deepseek`, or any entry from
    /// `~/.kotonia/providers.json`). When omitted the provider is inferred
    /// from the model id.
    #[arg(long)]
    provider: Option<String>,

    /// Agent engine: `react` (default) or `claude-code`. Also selected
    /// automatically when `--model claude-code` is passed.
    #[arg(long, default_value = "react")]
    engine: String,

    /// Approval policy. `all` / `allowlist` (default) / `auto`.
    #[arg(long, default_value = "allowlist")]
    approval: String,

    /// Run agent tasks inside the daemon's cwd. Default is to create a
    /// fresh git worktree per task (matches the one-shot CLI).
    #[arg(long)]
    in_place: bool,

    /// Opt out of the phone-side approval channel (Telegram/Discord) for
    /// new browser sessions. Without this flag the daemon refuses to start
    /// unless `kotonia-cli pair-notifier <telegram|discord>` has been run.
    /// Only safe if you fully trust the kotonia.ai backend AND every web
    /// session that drives this daemon — i.e. local-only experiments.
    #[arg(long)]
    no_notifier: bool,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// One-shot task description. Omit to enter the interactive REPL.
    prompt: Option<String>,

    /// Model id. Defaults to the hosted `kotonia-gemma4-26b` (requires
    /// `kotonia-cli login`). DeepSeek API: `deepseek-chat`,
    /// `deepseek-reasoner` (with optional `:thinking` suffix, needs
    /// `DEEPSEEK_API_KEY`). Custom providers come from
    /// `~/.kotonia/providers.json` and are selected via `--provider`.
    #[arg(short, long, default_value = "kotonia-gemma4-26b")]
    model: String,

    /// Explicit provider name (`kotonia`, `deepseek`, or any entry from
    /// `~/.kotonia/providers.json`). When omitted the provider is inferred
    /// from the model id; falls back to the default provider for unknown
    /// model ids.
    #[arg(long)]
    provider: Option<String>,

    /// Agent engine: `react` (default — kotonia-cli's own ReAct loop over a
    /// provider) or `claude-code` (drive the local `claude` binary as a
    /// subprocess in headless stream-json mode). Also selected automatically
    /// when `--model claude-code` is passed.
    #[arg(long, default_value = "react")]
    engine: String,

    /// Approval mode: `all` gates every command, `allowlist` (default) auto-runs
    /// read-only / build / test families and gates anything destructive,
    /// `auto` runs everything without asking.
    #[arg(short, long, default_value = "allowlist")]
    approval: String,

    /// Run inside the caller's cwd directly. Default is to create a git
    /// worktree off the current branch in /tmp/kotonia-agent-* and operate
    /// there so the working copy is untouched until the operator merges.
    #[arg(long)]
    in_place: bool,

    /// Base branch / ref the worktree is forked from. Defaults to HEAD.
    #[arg(long)]
    base_ref: Option<String>,

    /// Cap on agent loop iterations per turn.
    #[arg(long, default_value_t = 30)]
    max_iterations: u32,

    /// Don't print the workspace path on shutdown.
    #[arg(long)]
    quiet_shutdown: bool,

    /// Keep the worktree on disk after shutdown (so you can `git merge` it
    /// manually). Default is to `git worktree remove --force` it.
    #[arg(long)]
    keep_worktree: bool,

    /// Resume an earlier session by id (loads message history from
    /// ~/.kotonia/sessions/<id>.jsonl).
    #[arg(long)]
    resume: Option<String>,

    /// Override the session id for the new log file (default: random UUID).
    #[arg(long)]
    session: Option<String>,

    /// Don't write the session log to disk. Use for ephemeral / sensitive
    /// runs where you don't want a transcript on disk.
    #[arg(long)]
    no_history: bool,

    /// List known sessions (newest first) and exit.
    #[arg(long)]
    list_sessions: bool,

    /// Force the legacy `<<<BASH>>>` delimiter loop even when the backend
    /// supports native tool calling. Useful for diffing/debugging.
    #[arg(long)]
    force_delimiter: bool,

    /// Speak the JSON stdio protocol (JSONL) instead of the human TTY UI.
    /// For the VS Code extension / machine-readable frontends: stdout carries
    /// the protocol, stderr stays logs. Only the `react` engine is supported.
    /// See `src/serve.rs` for the wire contract. Combine with `--resume <id>`.
    #[arg(long)]
    serve: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.cmd {
        Some(Cmd::Daemon(args)) => return run_daemon(args).await,
        Some(Cmd::Login(args)) => return run_login(args).await,
        Some(Cmd::PairNotifier(args)) => return run_pair_notifier(args).await,
        None => {}
    }

    let cli = cli.run;

    if cli.list_sessions {
        return print_sessions_and_exit();
    }

    let approval_mode = match ApprovalMode::parse(&cli.approval) {
        Some(m) => m,
        None => {
            eprintln!(
                "kotonia-cli: unknown approval mode `{}` (expected all|allowlist|auto)",
                cli.approval
            );
            return ExitCode::from(2);
        }
    };

    // Decide engine: explicit `--engine` wins, otherwise `--model claude-code`
    // also selects the Claude Code subprocess engine for ergonomics.
    let engine_choice = if cli.engine == "claude-code" || cli.model == "claude-code" {
        EngineChoice::ClaudeCode
    } else if cli.engine == "react" {
        EngineChoice::ReAct
    } else {
        eprintln!(
            "kotonia-cli: unknown engine `{}` (expected react|claude-code)",
            cli.engine
        );
        return ExitCode::from(2);
    };

    let launch_cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kotonia-cli: cannot read cwd: {e}");
            return ExitCode::from(1);
        }
    };

    let workspace = if cli.in_place {
        AgentWorkspace::in_place(launch_cwd)
    } else {
        match AgentWorkspace::create_worktree(&launch_cwd, cli.base_ref.as_deref()).await {
            Ok(w) => w,
            Err(e) => {
                eprintln!("kotonia-cli: failed to create worktree: {e}");
                eprintln!("   (hint: pass --in-place to operate on the cwd directly)");
                return ExitCode::from(1);
            }
        }
    };

    // Resolve session_id up front; the ClaudeCode engine needs it for
    // `--session-id` / `--resume`, and the ReAct path uses it for history.
    let session_id = cli
        .session
        .clone()
        .or_else(|| cli.resume.clone())
        .unwrap_or_else(new_session_id);

    // The kotonia /api/v1 helper banner only applies to the ReAct prompt
    // (the model is told it can shell out to the API). ClaudeCode runs its
    // own tool surface, so we suppress this for that engine.
    let kotonia_api_enabled = matches!(engine_choice, EngineChoice::ReAct)
        && std::env::var("KOTONIA_API_KEY").is_ok();

    let mut agent = match engine_choice {
        EngineChoice::ReAct => {
            let provider = match Provider::resolve(cli.provider.as_deref(), &cli.model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("kotonia-cli: cannot use model `{}`: {e}", cli.model);
                    return ExitCode::from(2);
                }
            };
            let mut config = AgentConfig::new(approval_mode, cli.in_place);
            config.max_iterations = cli.max_iterations;
            config.force_delimiter = cli.force_delimiter;
            config.kotonia_api_base = if std::env::var("KOTONIA_API_KEY").is_ok() {
                Some(
                    std::env::var("KOTONIA_API_BASE")
                        .unwrap_or_else(|_| "https://kotonia.ai".to_string()),
                )
            } else {
                None
            };
            DispatchAgent::ReAct(Agent::new(&workspace.root, provider, config))
        }
        EngineChoice::ClaudeCode => {
            // The Claude Code session id must be a UUID; the host's compact
            // timestamp ids don't qualify, so coerce on first use.
            let claude_session_id =
                kotonia_cli::agent::claude_code::claude_code_session_id(&session_id);
            DispatchAgent::ClaudeCode(ClaudeCodeAgent::new(
                &workspace.root,
                claude_session_id,
                cli.in_place,
            ))
        }
    };

    // Wire history persistence + resume.
    if !cli.no_history {
        match HistoryStore::open(&session_id) {
            Ok(mut store) => {
                let is_resume = cli.resume.is_some();
                if !is_resume {
                    let _ = store.write_header(
                        agent.model_id(),
                        agent.backend_label(),
                        &approval_mode.to_string(),
                        &workspace.root,
                        cli.in_place,
                    );
                }
                agent = agent.with_history(store);
                if is_resume {
                    if let Some(id) = cli.resume.as_deref() {
                        match load_session_messages(id) {
                            Ok(prior) => {
                                eprintln!(
                                    "resumed session `{id}` ({} prior messages)",
                                    prior.len()
                                );
                                agent.seed_messages(prior);
                            }
                            Err(e) => {
                                eprintln!("kotonia-cli: cannot resume `{id}`: {e}");
                                return ExitCode::from(1);
                            }
                        }
                    }
                } else {
                    agent.log_initial_system();
                }
            }
            Err(e) => {
                eprintln!("kotonia-cli: history disabled ({e})");
            }
        }
    }

    // JSON stdio protocol: skip the human banner/REPL and drive the agent from
    // stdin. Only the ReAct engine exposes the event/approval surface the
    // protocol needs; claude-code runs its own tool loop, so reject it here.
    if cli.serve {
        match agent {
            DispatchAgent::ReAct(react_agent) => {
                let hello = serve::HelloInfo {
                    model: react_agent.model_id().to_string(),
                    backend: react_agent.backend_label().to_string(),
                    tool_mode: if react_agent.native_mode() {
                        "native"
                    } else {
                        "delimiter"
                    },
                    approval_mode: approval_mode.to_string(),
                    workspace_root: workspace.root.to_string_lossy().to_string(),
                    is_worktree: workspace.is_worktree(),
                    session_id: react_agent.session_id().map(|s| s.to_string()),
                    kotonia_api: kotonia_api_enabled,
                };
                serve::serve(react_agent, hello).await;
                cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
                return ExitCode::SUCCESS;
            }
            DispatchAgent::ClaudeCode(_) => {
                eprintln!(
                    "kotonia-cli: --serve is only supported with the `react` engine \
                     (not claude-code)"
                );
                cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
                return ExitCode::from(2);
            }
        }
    }

    print_banner(&workspace, approval_mode, &agent, kotonia_api_enabled);

    let interactive = cli.prompt.is_none() && atty_stdin();
    let mut approval_handler = StdioApproval::new();
    let mut sink = StdoutSink::new();

    let outcome = if let Some(task) = cli.prompt.clone() {
        // One-shot.
        agent.run_turn(&task, &mut approval_handler, &mut sink).await
    } else if interactive {
        // REPL mode.
        repl(&mut agent, &mut approval_handler, &mut sink).await
    } else {
        // Pipe / non-TTY stdin: treat the entire stdin as a single prompt.
        match read_stdin_to_end() {
            Ok(task) if !task.trim().is_empty() => {
                agent.run_turn(&task, &mut approval_handler, &mut sink).await
            }
            _ => {
                eprintln!("kotonia-cli: no prompt (pass as argument or pipe via stdin)");
                cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
                return ExitCode::from(2);
            }
        }
    };

    let exit_code = match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nkotonia-cli: {e}");
            ExitCode::from(1)
        }
    };

    cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
    exit_code
}

async fn repl(
    agent: &mut DispatchAgent,
    approval: &mut StdioApproval,
    sink: &mut StdoutSink,
) -> Result<String, kotonia_cli::agent::dispatch::DispatchError> {
    eprintln!("(interactive — empty line / `exit` / Ctrl-D to quit)");
    let stdin = io::stdin();
    let mut last_answer = String::new();
    loop {
        eprint!("\nkotonia> ");
        let _ = io::stderr().flush();
        let mut buf = String::new();
        let n = match stdin.lock().read_line(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("\nkotonia-cli: stdin error: {e}");
                break;
            }
        };
        if n == 0 {
            // EOF (Ctrl-D)
            eprintln!();
            break;
        }
        let task = buf.trim();
        if task.is_empty() || matches!(task, "exit" | "quit" | ":q") {
            break;
        }
        match agent.run_turn(task, approval, sink).await {
            Ok(ans) => last_answer = ans,
            Err(e) => {
                eprintln!("\nkotonia-cli: turn failed: {e}");
                // Don't tear down — let the operator try again with new input.
            }
        }
    }
    Ok(last_answer)
}

fn atty_stdin() -> bool {
    // Lightweight TTY detection without an extra dep: check if stdin is a terminal
    // via the `isatty` syscall through std::io::IsTerminal (Rust 1.70+).
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}

async fn cleanup_workspace(workspace: AgentWorkspace, keep: bool, quiet: bool) {
    if !workspace.is_worktree() {
        return;
    }
    let path = workspace.root.clone();
    let branch = workspace.branch().map(|s| s.to_string());
    if !quiet {
        if keep {
            eprintln!(
                "\nkept worktree: {}{}",
                path.display(),
                branch
                    .as_deref()
                    .map(|b| format!(" (branch {b})"))
                    .unwrap_or_default()
            );
        } else {
            eprintln!("\ncleaning up worktree: {}", path.display());
        }
    }
    if let Err(e) = workspace.cleanup(keep).await {
        eprintln!("kotonia-cli: worktree cleanup failed: {e}");
    }
}

fn print_banner(
    workspace: &AgentWorkspace,
    approval: ApprovalMode,
    agent: &DispatchAgent,
    kotonia_api: bool,
) {
    eprintln!("─────────────────────────────────────────────");
    eprintln!("kotonia-cli");
    eprintln!("  model     : {}", agent.provider_label());
    eprintln!(
        "  tools     : {}",
        if agent.native_mode() {
            "native (bash + web_search + fetch_url)"
        } else {
            "delimiter (<<<BASH>>> + web-search + fetch-url CLI)"
        }
    );
    eprintln!("  approval  : {approval}");
    eprintln!(
        "  workspace : {} ({})",
        workspace.root.display(),
        if workspace.is_worktree() {
            "worktree"
        } else {
            "in-place"
        }
    );
    if let Some(id) = agent.session_id() {
        eprintln!("  session   : {id}  (resume with --resume {id})");
    }
    if kotonia_api {
        let base =
            std::env::var("KOTONIA_API_BASE").unwrap_or_else(|_| "https://kotonia.ai".to_string());
        eprintln!("  kotonia   : {base}/api/v1  (image/audio/video tools enabled)");
    }
    eprintln!("─────────────────────────────────────────────");
}

async fn run_daemon(args: DaemonArgs) -> ExitCode {
    let approval = match ApprovalMode::parse(&args.approval) {
        Some(m) => m,
        None => {
            eprintln!(
                "kotonia-cli daemon: unknown approval mode `{}` (expected all|allowlist|auto)",
                args.approval
            );
            return ExitCode::from(2);
        }
    };

    // Resolve creds: flag/env > ~/.kotonia/daemon.json.
    let stored = daemon_config::load();
    let server = args
        .server
        .or_else(|| stored.as_ref().map(|c| c.server.clone()))
        .unwrap_or_else(|| "https://kotonia.ai".to_string());
    let device_id = match args
        .device_id
        .or_else(|| stored.as_ref().map(|c| c.device_id.clone()))
    {
        Some(v) => v,
        None => {
            eprintln!("kotonia-cli daemon: no device_id. Run `kotonia-cli login` first, or pass --device-id / KOTONIA_DEVICE_ID.");
            return ExitCode::from(2);
        }
    };
    let device_token = match args
        .device_token
        .or_else(|| stored.as_ref().map(|c| c.device_token.clone()))
    {
        Some(v) => v,
        None => {
            eprintln!("kotonia-cli daemon: no device_token. Run `kotonia-cli login` first, or pass --device-token / KOTONIA_DEVICE_TOKEN.");
            return ExitCode::from(2);
        }
    };

    // ── Notifier resolve ──────────────────────────────────────────────
    // Foolproof default: refuse to start without a paired notifier. The
    // operator must either run `pair-notifier` or explicitly opt out.
    let notifier = match (notifier::load_notifier_config(), args.no_notifier) {
        (Some(stored_notifier), _) => {
            let http = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("kotonia-cli daemon: http client: {e}");
                    return ExitCode::from(1);
                }
            };
            let n = notifier::build_notifier(&stored_notifier, http);
            match n.ping().await {
                Ok(name) => {
                    eprintln!("[daemon] notifier verified: @{name}");
                }
                Err(e) => {
                    eprintln!(
                        "kotonia-cli daemon: notifier ping failed ({e}). Fix the bot token \
                         in ~/.kotonia/notifier.json or re-run `kotonia-cli pair-notifier`."
                    );
                    return ExitCode::from(1);
                }
            }
            Some(n)
        }
        (None, true) => {
            eprintln!(
                "[daemon] WARNING: --no-notifier set; every task runs without phone \
                 confirmation. If the kotonia.ai backend is compromised the attacker \
                 can drive this daemon at will."
            );
            None
        }
        (None, false) => {
            eprintln!(
                "kotonia-cli daemon: no approval channel configured.\n\n\
                 For safety the daemon refuses to start without one — if our backend is \
                 compromised, an attacker could otherwise run arbitrary commands on this \
                 machine.\n\n\
                 Set one up with:\n\
                     kotonia-cli pair-notifier telegram\n\n\
                 Or, only if you fully trust both kotonia.ai and every browser that drives \
                 this daemon, opt out explicitly:\n\
                     kotonia-cli daemon --no-notifier"
            );
            return ExitCode::from(2);
        }
    };

    let config = DaemonConfig {
        server,
        device_id,
        device_token,
        model: args.model,
        provider: args.provider,
        engine: args.engine,
        approval,
        in_place: args.in_place,
        notifier,
    };
    match daemon::run(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kotonia-cli daemon: {e}");
            ExitCode::from(1)
        }
    }
}

async fn run_login(args: LoginArgs) -> ExitCode {
    match login::run(&args.server).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kotonia-cli login: {e}");
            ExitCode::from(1)
        }
    }
}

async fn run_pair_notifier(args: PairNotifierArgs) -> ExitCode {
    let http = match reqwest::Client::builder()
        // Default timeout. Long-poll calls supply their own per-request timeouts.
        .timeout(std::time::Duration::from_secs(60))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kotonia-cli pair-notifier: http client: {e}");
            return ExitCode::from(1);
        }
    };
    let cfg = match args.kind.to_ascii_lowercase().as_str() {
        "telegram" | "tg" => match notifier::telegram::pair(http).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("kotonia-cli pair-notifier: {e}");
                return ExitCode::from(1);
            }
        },
        "discord" => {
            eprintln!("kotonia-cli pair-notifier: discord support is planned but not implemented yet.");
            return ExitCode::from(2);
        }
        other => {
            eprintln!(
                "kotonia-cli pair-notifier: unknown notifier kind `{other}` (expected telegram)"
            );
            return ExitCode::from(2);
        }
    };
    match notifier::save_notifier_config(&cfg) {
        Ok(path) => {
            eprintln!();
            eprintln!("Saved notifier config to {}", path.display());
            eprintln!();
            eprintln!("Next: start the daemon with `kotonia-cli daemon`. It will gate every");
            eprintln!("new browser session through your Telegram bot for 24h trust.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kotonia-cli pair-notifier: save config: {e}");
            ExitCode::from(1)
        }
    }
}

fn new_session_id() -> String {
    // Short readable id: <YYYYMMDD-HHMMSS>-<4 hex>. Sortable by start time,
    // unique enough for a single operator on one machine.
    let now = chrono::Utc::now();
    let stamp = now.format("%Y%m%d-%H%M%S");
    let rnd: String = uuid::Uuid::new_v4().to_string().chars().take(4).collect();
    format!("{stamp}-{rnd}")
}

#[derive(Clone, Copy, Debug)]
enum EngineChoice {
    ReAct,
    ClaudeCode,
}

fn print_sessions_and_exit() -> ExitCode {
    match list_sessions() {
        Ok(sessions) if sessions.is_empty() => {
            eprintln!("(no sessions in ~/.kotonia/sessions/)");
            ExitCode::SUCCESS
        }
        Ok(sessions) => {
            for s in sessions {
                let when = match s
                    .modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .and_then(|d| chrono::DateTime::<chrono::Utc>::from_timestamp(d.as_secs() as i64, 0))
                {
                    Some(dt) => dt.format("%Y-%m-%d %H:%M UTC").to_string(),
                    None => "?".to_string(),
                };
                println!("{}  {}", when, s.id);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kotonia-cli: cannot list sessions: {e}");
            ExitCode::from(1)
        }
    }
}

fn read_stdin_to_end() -> io::Result<String> {
    use std::io::Read;
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

struct StdoutSink;

impl StdoutSink {
    fn new() -> Self {
        Self
    }
}

impl EventSink for StdoutSink {
    fn emit(&mut self, event: Event) {
        match event {
            Event::IterationStart { iteration, max } => {
                println!("\n── iter {iteration}/{max} ──");
            }
            Event::LlmThinking => {
                print!("· thinking ");
                let _ = io::stdout().flush();
            }
            Event::Bash { command } => {
                println!("\n$ {command}");
            }
            Event::BashSkipped { command, reason } => {
                println!("\n[skipped] {reason}");
                println!("$ {command}");
            }
            Event::Observation { result } => {
                let header = if result.timed_out {
                    format!("[exit {} • TIMED OUT]", result.exit_code)
                } else if result.truncated {
                    format!("[exit {} • truncated]", result.exit_code)
                } else {
                    format!("[exit {}]", result.exit_code)
                };
                println!("{header}");
                if !result.combined.is_empty() {
                    println!("{}", result.combined.trim_end());
                }
            }
            Event::Final { answer } => {
                println!("\n══ final answer ══");
                println!("{answer}");
            }
            Event::Malformed { excerpt } => {
                println!("\n[malformed model output — retrying]");
                println!("{}…", excerpt.trim());
            }
            Event::Error { message } => {
                eprintln!("\n[error] {message}");
            }
            Event::Done {
                iterations,
                success,
            } => {
                eprintln!(
                    "\n── done after {iterations} iter{} ── {}",
                    if iterations == 1 { "" } else { "s" },
                    if success { "✓" } else { "✗" }
                );
            }
        }
    }
}

struct StdioApproval {
    stdin: io::Stdin,
}

impl StdioApproval {
    fn new() -> Self {
        Self {
            stdin: io::stdin(),
        }
    }
}

impl ApprovalHandler for StdioApproval {
    fn ask(&mut self, command: &str, reason: &str) -> ApprovalOutcome {
        eprintln!("\n────── approval required ({reason}) ──────");
        eprintln!("$ {command}");
        eprint!("approve? [y/N] ");
        let _ = io::stderr().flush();
        let mut buf = String::new();
        let mut handle = self.stdin.lock();
        if handle.read_line(&mut buf).is_err() {
            return ApprovalOutcome::Deny;
        }
        let answer = buf.trim().to_ascii_lowercase();
        if matches!(answer.as_str(), "y" | "yes") {
            ApprovalOutcome::Approve
        } else {
            ApprovalOutcome::Deny
        }
    }
}
