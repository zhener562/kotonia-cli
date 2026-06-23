//! kotonia-cli — local shell agent backed by a self-hosted LLM.
//!
//! Model backends:
//!   - `deepseek-v4-flash`        (default) — local llama.cpp on :8898
//!   - `gemma4-26b-uncensored`    local vLLM on :8899 (native tool calling)
//!   - `deepseek-chat`            DeepSeek API (V4-Flash class, native tools)
//!   - `deepseek-reasoner`        DeepSeek API (V4-Pro reasoning, native tools)
//!   - `kotonia-v4-flash`         remote — hits kotonia.ai /api/v1 chat
//!   - `kotonia-gemma4-26b`       remote — kotonia.ai /api/v1 chat (native tools)
//!
//! Backends that advertise OpenAI-compatible `tools` are driven via native
//! tool calling (`bash`, `web_search`). V4-Flash on llama.cpp falls back to
//! the legacy `<<<BASH>>>` delimiter loop because the build has no
//! `--tool-call-parser`.
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
use kotonia_cli::agent::history::{list_sessions, load_session_messages, HistoryStore};
use kotonia_cli::agent::provider::Provider;
use kotonia_cli::agent::worktree::AgentWorkspace;
use kotonia_cli::config as daemon_config;
use kotonia_cli::daemon::{self, DaemonConfig};
use kotonia_cli::login;

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
}

#[derive(Args, Debug)]
struct LoginArgs {
    /// HTTP(S) base of the kotonia.ai backend to pair with.
    #[arg(long, default_value = "https://kotonia.ai", env = "KOTONIA_API_BASE")]
    server: String,
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
    #[arg(long, default_value = "deepseek-v4-flash")]
    model: String,

    /// Approval policy. `all` / `allowlist` (default) / `auto`.
    #[arg(long, default_value = "allowlist")]
    approval: String,

    /// Run agent tasks inside the daemon's cwd. Default is to create a
    /// fresh git worktree per task (matches the one-shot CLI).
    #[arg(long)]
    in_place: bool,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// One-shot task description. Omit to enter the interactive REPL.
    prompt: Option<String>,

    /// Model id. Local: `deepseek-v4-flash`, `gemma4-26b-uncensored`.
    /// DeepSeek API: `deepseek-chat`, `deepseek-reasoner` (with optional
    /// `:thinking` suffix). Requires DEEPSEEK_API_KEY for the API routes.
    #[arg(short, long, default_value = "deepseek-v4-flash")]
    model: String,

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
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.cmd {
        Some(Cmd::Daemon(args)) => return run_daemon(args).await,
        Some(Cmd::Login(args)) => return run_login(args).await,
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

    let provider = match Provider::for_model(&cli.model) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kotonia-cli: cannot use model `{}`: {e}", cli.model);
            return ExitCode::from(2);
        }
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

    let mut config = AgentConfig::new(approval_mode, cli.in_place);
    config.max_iterations = cli.max_iterations;
    config.force_delimiter = cli.force_delimiter;
    // Surface kotonia /api/v1 (image/audio/video) only when the operator
    // exported KOTONIA_API_KEY. The key passes through bash naturally
    // (Command inherits env); we just gate the prompt section so the
    // model doesn't try to call an API it has no credentials for.
    config.kotonia_api_base = if std::env::var("KOTONIA_API_KEY").is_ok() {
        Some(
            std::env::var("KOTONIA_API_BASE")
                .unwrap_or_else(|_| "https://kotonia.ai".to_string()),
        )
    } else {
        None
    };
    let kotonia_api_enabled = config.kotonia_api_base.is_some();
    let mut agent = Agent::new(&workspace.root, provider, config);

    // Wire history persistence + resume.
    let session_id = cli
        .session
        .clone()
        .or_else(|| cli.resume.clone())
        .unwrap_or_else(new_session_id);

    if !cli.no_history {
        match HistoryStore::open(&session_id) {
            Ok(mut store) => {
                let is_resume = cli.resume.is_some();
                if !is_resume {
                    let label = agent.provider_label();
                    let backend = if label.contains("(deepseek-api)") {
                        "deepseek-api"
                    } else if label.contains("(kotonia-api)") {
                        "kotonia-api"
                    } else {
                        "local"
                    };
                    let _ = store.write_header(
                        label.split_whitespace().next().unwrap_or(""),
                        backend,
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
    agent: &mut Agent,
    approval: &mut StdioApproval,
    sink: &mut StdoutSink,
) -> Result<String, kotonia_cli::agent::agent::AgentError> {
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
    agent: &Agent,
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

    let config = DaemonConfig {
        server,
        device_id,
        device_token,
        model: args.model,
        approval,
        in_place: args.in_place,
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

fn new_session_id() -> String {
    // Short readable id: <YYYYMMDD-HHMMSS>-<4 hex>. Sortable by start time,
    // unique enough for a single operator on one machine.
    let now = chrono::Utc::now();
    let stamp = now.format("%Y%m%d-%H%M%S");
    let rnd: String = uuid::Uuid::new_v4().to_string().chars().take(4).collect();
    format!("{stamp}-{rnd}")
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
