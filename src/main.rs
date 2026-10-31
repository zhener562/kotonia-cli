//! kotonia-cli — local shell agent backed by a hosted or self-hosted LLM.
//!
//! Built-in providers (resolved by model id):
//!   - `kotonia-llm-basic` (default)  — kotonia.ai /api/v1 chat (free local, native tools)
//!   - `kotonia-llm-standard`         — kotonia.ai /api/v1 chat (metered cloud tier)
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
use serde::Serialize;

use kotonia_cli::agent::agent::{
    Agent, AgentConfig, ApprovalHandler, ApprovalOutcome, ConnectionReply, Event, EventSink,
};
use kotonia_cli::agent::approval::ApprovalMode;
use kotonia_cli::agent::claude_code::ClaudeCodeAgent;
use kotonia_cli::agent::dispatch::DispatchAgent;
use kotonia_cli::agent::history::{
    list_sessions, load_session_messages, load_session_metadata, load_session_transcript,
    HistoryStore,
};
use kotonia_cli::agent::provider::Provider;
use kotonia_cli::agent::worktree::AgentWorkspace;
use kotonia_cli::config as daemon_config;
use kotonia_cli::connectors;
use kotonia_cli::daemon::{self, DaemonConfig};
use kotonia_cli::home::feed::Delivery as HomeDelivery;
use kotonia_cli::home::heartbeat as home_heartbeat;
use kotonia_cli::home::runner as home_runner;
use kotonia_cli::home::schedule as home_schedule;
use kotonia_cli::home::workspace::{self as home_workspace, HomeDir};
use kotonia_cli::login;
use kotonia_cli::notifier;
use kotonia_cli::serve;
use kotonia_cli::telemetry;
use kotonia_cli::vault::Vault;

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

    /// Report the shared kotonia.ai credential state. `--validate` performs
    /// a non-rotating server check so stale/expired device tokens do not look
    /// logged in merely because daemon.json still exists.
    AuthStatus(AuthStatusArgs),

    /// Remove the shared device credentials from ~/.kotonia/daemon.json.
    Logout(LogoutArgs),

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

    /// Resident agent ("home"): a persistent workspace of memory files plus
    /// a heartbeat and reminders that survive the machine sleeping.
    /// `kotonia-cli home init` to start, `kotonia-cli home run` to stay resident.
    Home(HomeArgs),

    /// Accounts the agent may use (mail, API keys). Secrets go to the local
    /// vault; the agent never sees them.
    Connector(ConnectorArgs),

    /// Read-only mail through a connected IMAP account.
    Mail(MailArgs),

    /// Secrets in the local vault (~/.kotonia/vault).
    Secret(SecretArgs),
}

#[derive(Args, Debug)]
struct ConnectorArgs {
    #[command(subcommand)]
    cmd: ConnectorCmd,
}

#[derive(Subcommand, Debug)]
enum ConnectorCmd {
    /// List connected accounts.
    List,
    /// Connect an account: gmail / icloud / imap / api_key.
    Add {
        kind: String,
        /// Email address to prefill (mail) or service name (api_key).
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        name: Option<String>,
    },
    /// Test a connector's stored credentials.
    Check { id: String },
    /// Disconnect: delete the secret and the record.
    Remove { id: String },
}

#[derive(Args, Debug)]
struct MailArgs {
    #[command(subcommand)]
    cmd: MailCmd,
}

#[derive(Subcommand, Debug)]
enum MailCmd {
    /// Newest messages in INBOX (headers + snippet).
    Recent {
        /// Connector id (default: the first mail connector).
        #[arg(long)]
        connector: Option<String>,
        /// Look back this many days.
        #[arg(long, default_value_t = 2)]
        days: i64,
        /// Only messages that arrived since the previous `--new` call.
        #[arg(long)]
        new: bool,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Print one message.
    Read {
        uid: u32,
        #[arg(long)]
        connector: Option<String>,
        #[arg(long, default_value_t = 12000)]
        max_chars: usize,
    },
}

#[derive(Args, Debug)]
struct SecretArgs {
    #[command(subcommand)]
    cmd: SecretCmd,
}

#[derive(Subcommand, Debug)]
enum SecretCmd {
    /// List secret names (never values).
    List,
    /// Store a secret, typed at a hidden prompt.
    Set {
        name: String,
    },
    Remove {
        name: String,
    },
    /// Run a command with a secret in its environment. Output is scrubbed
    /// of the secret before it is printed.
    Run {
        name: String,
        /// Environment variable name (default: the connector's, else SECRET).
        #[arg(long)]
        env: Option<String>,
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
}

#[derive(Args, Debug)]
struct HomeArgs {
    #[command(subcommand)]
    cmd: HomeCmd,
}

#[derive(Subcommand, Debug)]
enum HomeCmd {
    /// Create ~/.kotonia/home (or $KOTONIA_HOME_DIR) with template files.
    /// Never overwrites an existing file.
    Init,
    /// Print the block that is prepended to every conversation.
    Prompt,
    /// Turn the heartbeat on (`--every 30`, minutes) or off (`--off`).
    Heartbeat {
        #[arg(long, default_value_t = 30)]
        every: i64,
        #[arg(long)]
        off: bool,
    },
    /// Local hours in which the heartbeat may reach out, `HH:MM-HH:MM`
    /// (default 09:00-21:30). Reminders ignore this.
    Hours { range: String },
    /// Add a one-off reminder. WHEN is `+30m` / `+2h` / `HH:MM` /
    /// `YYYY-MM-DD HH:MM` (local) or RFC 3339.
    Remind { when: String, text: Vec<String> },
    /// List scheduled jobs.
    List,
    /// Remove a job by id.
    Remove { id: String },
    /// Evaluate the schedule once (what `run` does every minute).
    Tick {
        #[command(flatten)]
        run: HomeRunArgs,
    },
    /// Stay resident: evaluate the schedule every minute until killed.
    Run {
        #[command(flatten)]
        run: HomeRunArgs,
    },
}

#[derive(Args, Debug)]
struct HomeRunArgs {
    /// Model for heartbeat runs.
    #[arg(short, long, default_value = "kotonia-llm-basic")]
    model: String,
    #[arg(long)]
    provider: Option<String>,
    /// Seconds between schedule checks.
    #[arg(long, default_value_t = 60)]
    poll_secs: u64,
    /// Show what would fire without running or advancing anything.
    #[arg(long)]
    dry_run: bool,
    /// Do not push to the paired phone notifier, even if one is set up.
    #[arg(long)]
    no_notifier: bool,
}

#[derive(Args, Debug)]
struct LoginArgs {
    /// HTTP(S) base of the kotonia.ai backend to pair with.
    #[arg(long, default_value = "https://kotonia.ai", env = "KOTONIA_API_BASE")]
    server: String,
}

#[derive(Args, Debug)]
struct AuthStatusArgs {
    /// Emit one machine-readable JSON object.
    #[arg(long)]
    json: bool,

    /// Validate the stored device token against the server.
    #[arg(long)]
    validate: bool,
}

#[derive(Args, Debug)]
struct LogoutArgs {
    /// Emit one machine-readable JSON object.
    #[arg(long)]
    json: bool,
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
    #[arg(long, default_value = "kotonia-llm-basic")]
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

    /// Model id. Defaults to the hosted `kotonia-llm-basic` (free local;
    /// requires `kotonia-cli login`). `kotonia-llm-standard` is the metered
    /// cloud tier; append `:think` to `kotonia-llm-basic` for the reasoning
    /// pass. DeepSeek API: `deepseek-chat`, `deepseek-reasoner` (optional
    /// `:thinking` suffix, needs `DEEPSEEK_API_KEY`). Custom providers come
    /// from `~/.kotonia/providers.json` and are selected via `--provider`.
    #[arg(short, long, default_value = "kotonia-llm-basic")]
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

    /// Load the resident agent's memory files (~/.kotonia/home, see
    /// `kotonia-cli home init`) into the system prompt.
    #[arg(long)]
    home: bool,

    /// Speak the JSON stdio protocol (JSONL) instead of the human TTY UI.
    /// For the VS Code extension / machine-readable frontends: stdout carries
    /// the protocol, stderr stays logs. Only the `react` engine is supported.
    /// See `src/serve.rs` for the wire contract. Combine with `--resume <id>`.
    #[arg(long)]
    serve: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let code = real_main().await;
    // No-op unless an agent run started telemetry (see `telemetry`).
    telemetry::finish(code == ExitCode::SUCCESS).await;
    code
}

async fn real_main() -> ExitCode {
    // Propagate the paired device_token into KOTONIA_API_KEY so (a) the
    // one-shot/REPL run path's `kotonia_api_base` gate below (which only
    // checks the env var) actually fires for a plain `kotonia-cli login`
    // session, and (b) the ReAct agent's bash subprocess inherits a
    // working key for the curl examples in its system prompt
    // (`Authorization: Bearer $KOTONIA_API_KEY`). Without this, a logged-in
    // user who never separately exported KOTONIA_API_KEY gets a working
    // chat model (the LLM call itself falls back to the device_token) but
    // an agent that has no idea the images/audio/video API exists at all —
    // it just hallucinates a made-up local endpoint instead. Mirrors
    // kotonia-desktop's `main.rs`, which does the same bridge.
    if let Some(cfg) = daemon_config::load() {
        if std::env::var_os("KOTONIA_API_KEY").is_none() {
            std::env::set_var("KOTONIA_API_KEY", &cfg.device_token);
        }
        if std::env::var_os("KOTONIA_API_BASE").is_none() {
            std::env::set_var("KOTONIA_API_BASE", &cfg.server);
        }
    }

    let cli = Cli::parse();

    match cli.cmd {
        Some(Cmd::Daemon(args)) => return run_daemon(args).await,
        Some(Cmd::Login(args)) => return run_login(args).await,
        Some(Cmd::AuthStatus(args)) => return run_auth_status(args).await,
        Some(Cmd::Logout(args)) => return run_logout(args),
        Some(Cmd::PairNotifier(args)) => return run_pair_notifier(args).await,
        Some(Cmd::Home(args)) => return run_home(args).await,
        Some(Cmd::Connector(args)) => return run_connector(args).await,
        Some(Cmd::Mail(args)) => return run_mail(args).await,
        Some(Cmd::Secret(args)) => return run_secret(args).await,
        None => {}
    }

    let cli = cli.run;

    if cli.list_sessions {
        return print_sessions_and_exit();
    }

    if cli.serve {
        kotonia_cli::client_info::set_mode("serve");
    }
    let engine_label = if cli.model == "claude-code" {
        "claude-code"
    } else {
        cli.engine.as_str()
    };
    telemetry::start(engine_label, &cli.model);

    let approval_mode = match ApprovalMode::parse(&cli.approval) {
        Some(m) => m,
        None => {
            eprintln!(
                "kotonia-cli: unknown approval mode `{}` (expected all|allowlist|auto)",
                cli.approval
            );
            telemetry::fail("bad_args");
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
        telemetry::fail("bad_args");
        return ExitCode::from(2);
    };

    let launch_cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kotonia-cli: cannot read cwd: {e}");
            telemetry::fail("cwd");
            return ExitCode::from(1);
        }
    };

    // Resolve session_id up front; the ClaudeCode engine needs it for
    // `--session-id` / `--resume`, and the ReAct path uses it for history.
    let session_id = cli
        .session
        .clone()
        .or_else(|| cli.resume.clone())
        .unwrap_or_else(new_session_id);

    let resume_meta = cli
        .resume
        .as_deref()
        .and_then(|id| load_session_metadata(id).ok());
    let workspace = if cli.in_place {
        AgentWorkspace::in_place(launch_cwd.clone())
    } else if let Some(meta) = resume_meta.as_ref().filter(|m| !m.in_place) {
        match AgentWorkspace::resume_worktree(&launch_cwd, &meta.workspace).await {
            Ok(w) => {
                eprintln!(
                    "resumed saved worktree for session `{}`: {}",
                    meta.id,
                    w.root.display()
                );
                w
            }
            Err(e) => {
                eprintln!(
                    "kotonia-cli: saved worktree for `{}` is unavailable ({e}); \
                     creating a fresh worktree",
                    meta.id
                );
                match AgentWorkspace::create_worktree(&launch_cwd, cli.base_ref.as_deref()).await {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("kotonia-cli: failed to create worktree: {e}");
                        eprintln!("   (hint: pass --in-place to operate on the cwd directly)");
                        telemetry::fail("worktree");
                        return ExitCode::from(1);
                    }
                }
            }
        }
    } else {
        match AgentWorkspace::create_worktree(&launch_cwd, cli.base_ref.as_deref()).await {
            Ok(w) => w,
            Err(e) => {
                eprintln!("kotonia-cli: failed to create worktree: {e}");
                eprintln!("   (hint: pass --in-place to operate on the cwd directly)");
                telemetry::fail("worktree");
                return ExitCode::from(1);
            }
        }
    };
    let in_place = !workspace.is_worktree();

    // The kotonia /api/v1 helper banner only applies to the ReAct prompt
    // (the model is told it can shell out to the API). ClaudeCode runs its
    // own tool surface, so we suppress this for that engine.
    let kotonia_api_enabled =
        matches!(engine_choice, EngineChoice::ReAct) && std::env::var("KOTONIA_API_KEY").is_ok();

    let mut agent = match engine_choice {
        EngineChoice::ReAct => {
            let provider = match Provider::resolve(cli.provider.as_deref(), &cli.model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("kotonia-cli: cannot use model `{}`: {e}", cli.model);
                    telemetry::fail("provider");
                    return ExitCode::from(2);
                }
            };
            let mut config = AgentConfig::new(approval_mode, in_place);
            config.max_iterations = cli.max_iterations;
            config.force_delimiter = cli.force_delimiter;
            config.connectors = cli.home;
            config.persona_prefix = match home_prompt_prefix(cli.home) {
                Ok(home) => join_prefixes(home, frontend_system_prompt_prefix()),
                Err(e) => {
                    eprintln!("kotonia-cli: --home: {e}");
                    telemetry::fail("bad_args");
                    return ExitCode::from(2);
                }
            };
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
                in_place,
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
                        in_place,
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
                                telemetry::fail("resume");
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
        telemetry::set_mode("serve");
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
                    history: cli
                        .resume
                        .as_deref()
                        .and_then(|id| load_session_transcript(id).ok())
                        .unwrap_or_default(),
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
                telemetry::fail("bad_args");
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
        telemetry::set_mode("oneshot");
        agent
            .run_turn(&task, &mut approval_handler, &mut sink)
            .await
    } else if interactive {
        // REPL mode.
        telemetry::set_mode("repl");
        repl(&mut agent, &mut approval_handler, &mut sink).await
    } else {
        // Pipe / non-TTY stdin: treat the entire stdin as a single prompt.
        telemetry::set_mode("pipe");
        match read_stdin_to_end() {
            Ok(task) if !task.trim().is_empty() => {
                agent
                    .run_turn(&task, &mut approval_handler, &mut sink)
                    .await
            }
            _ => {
                eprintln!("kotonia-cli: no prompt (pass as argument or pipe via stdin)");
                telemetry::fail("no_prompt");
                cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
                return ExitCode::from(2);
            }
        }
    };

    let exit_code = match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nkotonia-cli: {e}");
            telemetry::fail(telemetry::failure_kind(&e));
            ExitCode::from(1)
        }
    };

    cleanup_workspace(workspace, cli.keep_worktree, cli.quiet_shutdown).await;
    exit_code
}

fn home_prompt_prefix(enabled: bool) -> Result<Option<String>, String> {
    if !enabled {
        return Ok(None);
    }
    let home = HomeDir::resolve()?;
    if !home.exists() {
        return Err(format!(
            "{} is not set up yet; run `kotonia-cli home init` first",
            home.root().display()
        ));
    }
    let mut block = home.prompt_block(
        home_workspace::DEFAULT_MAX_FILE_CHARS,
        home_workspace::DEFAULT_MAX_TOTAL_CHARS,
    );
    block.push_str(&connectors::prompt_section(&self_exe()));
    Ok(Some(block))
}

fn self_exe() -> String {
    std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "kotonia-cli".to_string())
}

fn find_connector(
    registry: &connectors::Registry,
    id: Option<&str>,
    want_mail: bool,
) -> Result<connectors::ConnectorRecord, String> {
    let found = match id {
        Some(id) => registry.connectors.get(id).cloned(),
        None if want_mail => registry.first_mail().cloned(),
        None => None,
    };
    found.ok_or_else(|| match id {
        Some(id) => format!("no connector `{id}` (see `kotonia-cli connector list`)"),
        None => "no mail account connected (run `kotonia-cli connector add gmail`)".to_string(),
    })
}

fn connector_secret(id: &str) -> Result<String, String> {
    Vault::resolve()?
        .get(id)?
        .ok_or_else(|| format!("the secret for `{id}` is missing; connect it again"))
}

async fn run_connector(args: ConnectorArgs) -> ExitCode {
    let result: Result<(), String> = async {
        match args.cmd {
            ConnectorCmd::List => {
                let r = connectors::Registry::load()?;
                if r.connectors.is_empty() {
                    println!("nothing connected");
                }
                for line in r.prompt_lines() {
                    println!("{line}");
                }
                Ok(())
            }
            ConnectorCmd::Add {
                kind,
                account,
                name,
            } => {
                let cat = connectors::Catalog::parse(&kind).ok_or_else(|| {
                    format!(
                        "unknown connector `{kind}` (one of {})",
                        connectors::Catalog::NAMES.join(", ")
                    )
                })?;
                let name = match (cat, name) {
                    (connectors::Catalog::ApiKey, None) => Some(
                        kotonia_cli::tty::read_visible_line("サービス名", None)
                            .map_err(|e| e.to_string())?,
                    ),
                    (_, n) => n,
                };
                let card = connectors::card_for(cat, "", account.as_deref(), name.as_deref());
                let values = kotonia_cli::tty::fill_connector_card(&card, false)
                    .map_err(|e| e.to_string())?
                    .ok_or("cancelled")?;
                let c = connectors::connect_from_card(cat, &card, &values).await?;
                println!("{}", c.for_human());
                if c.check.ok {
                    Ok(())
                } else {
                    Err("connection test failed".to_string())
                }
            }
            ConnectorCmd::Check { id } => {
                let mut r = connectors::Registry::load()?;
                let rec = find_connector(&r, Some(&id), false)?;
                let check = connectors::check_connector(&rec, &connector_secret(&id)?).await;
                println!(
                    "{}: {}",
                    if check.ok { "ok" } else { "failed" },
                    check.message
                );
                if let Some(c) = r.connectors.get_mut(&id) {
                    c.last_check = Some(check);
                }
                r.save()
            }
            ConnectorCmd::Remove { id } => {
                let mut r = connectors::Registry::load()?;
                let had = r.connectors.remove(&id).is_some();
                let had_secret = Vault::resolve()?.remove(&id)?;
                if !had && !had_secret {
                    return Err(format!("no connector `{id}`"));
                }
                r.save()?;
                println!("removed {id}");
                Ok(())
            }
        }
    }
    .await;
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kotonia-cli connector: {e}");
            ExitCode::from(1)
        }
    }
}

async fn run_mail(args: MailArgs) -> ExitCode {
    let offset = home_schedule::local_offset_now();
    let result: Result<String, String> = async {
        let registry = connectors::Registry::load()?;
        match args.cmd {
            MailCmd::Recent {
                connector,
                days,
                new,
                limit,
            } => {
                let rec = find_connector(&registry, connector.as_deref(), true)?;
                let secret = connector_secret(&rec.id)?;
                let window = if new {
                    connectors::mail::Window::New
                } else {
                    connectors::mail::Window::SinceDays(days)
                };
                let items = connectors::mail::recent(&rec, &secret, window, limit).await?;
                Ok(connectors::mail::format_summaries(&items, offset))
            }
            MailCmd::Read {
                uid,
                connector,
                max_chars,
            } => {
                let rec = find_connector(&registry, connector.as_deref(), true)?;
                let secret = connector_secret(&rec.id)?;
                let text = connectors::mail::read(&rec, &secret, uid).await?;
                let text = match text.char_indices().nth(max_chars) {
                    Some((i, _)) => format!("{}\n… (cut at {max_chars} characters)", &text[..i]),
                    None => text,
                };
                Ok(connectors::mail::wrap_untrusted(&text))
            }
        }
    }
    .await;
    match result {
        Ok(out) => {
            println!("{}", kotonia_cli::vault::redact(&out));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kotonia-cli mail: {}", kotonia_cli::vault::redact(&e));
            ExitCode::from(1)
        }
    }
}

async fn run_secret(args: SecretArgs) -> ExitCode {
    let result: Result<i32, String> = async {
        let vault = Vault::resolve()?;
        match args.cmd {
            SecretCmd::List => {
                for n in vault.names() {
                    println!("{n}");
                }
                Ok(0)
            }
            SecretCmd::Set { name } => {
                kotonia_cli::vault::validate_name(&name)?;
                let v =
                    kotonia_cli::tty::read_hidden_line(&format!("{name} (入力は表示されません): "))
                        .map_err(|e| e.to_string())?;
                vault.set(&name, v.trim())?;
                println!("stored {name}");
                Ok(0)
            }
            SecretCmd::Remove { name } => {
                if vault.remove(&name)? {
                    println!("removed {name}");
                    Ok(0)
                } else {
                    Err(format!("no secret `{name}`"))
                }
            }
            SecretCmd::Run { name, env, command } => {
                let value = vault
                    .get(&name)?
                    .ok_or_else(|| format!("no secret `{name}`"))?;
                let env = env
                    .or_else(|| {
                        connectors::Registry::load().ok().and_then(|r| {
                            r.connectors.get(&name).and_then(|c| match &c.kind {
                                connectors::ConnectorKind::ApiKey { env } => Some(env.clone()),
                                _ => None,
                            })
                        })
                    })
                    .unwrap_or_else(|| "SECRET".to_string());
                let out = tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(command.join(" "))
                    .env(&env, value.trim())
                    .stdin(std::process::Stdio::null())
                    .output()
                    .await
                    .map_err(|e| format!("spawn: {e}"))?;
                let scrub = |b: &[u8]| {
                    kotonia_cli::vault::redact_with(
                        &String::from_utf8_lossy(b),
                        &[value.trim().to_string()],
                    )
                };
                print!("{}", scrub(&out.stdout));
                eprint!("{}", scrub(&out.stderr));
                Ok(out.status.code().unwrap_or(1))
            }
        }
    }
    .await;
    match result {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(code.clamp(1, 255) as u8),
        Err(e) => {
            eprintln!("kotonia-cli secret: {e}");
            ExitCode::from(1)
        }
    }
}

/// Home block first (who you are, what you remember), then the frontend's
/// persona / language instructions.
fn join_prefixes(a: Option<String>, b: Option<String>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a}\n\n{b}")),
        (a, b) => a.or(b),
    }
}

async fn run_home(args: HomeArgs) -> ExitCode {
    let home = match HomeDir::resolve() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("kotonia-cli home: {e}");
            return ExitCode::from(1);
        }
    };
    let needs_init = !matches!(args.cmd, HomeCmd::Init);
    if needs_init && !home.exists() {
        eprintln!(
            "kotonia-cli home: {} is not set up yet; run `kotonia-cli home init` first",
            home.root().display()
        );
        return ExitCode::from(1);
    }
    let now = chrono::Utc::now();
    let offset = home_schedule::local_offset_now();
    let load = || home_schedule::Schedule::load(home.root());
    let result: Result<(), String> = match args.cmd {
        HomeCmd::Init => home.ensure_initialized().map(|created| {
            println!("home: {}", home.root().display());
            if created.is_empty() {
                println!("already set up; nothing changed");
            } else {
                println!("created: {}", created.join(", "));
            }
        }),
        HomeCmd::Prompt => {
            // Exactly what `--home` and the heartbeat prepend.
            match home_prompt_prefix(true) {
                Ok(p) => println!("{}", p.unwrap_or_default()),
                Err(e) => eprintln!("{e}"),
            }
            Ok(())
        }
        HomeCmd::Heartbeat { every, off } => load().and_then(|mut s| {
            if off {
                s.clear_heartbeat();
                println!("heartbeat off");
            } else {
                s.set_heartbeat(every * 60, now);
                let checks = home_heartbeat::checklist(
                    &home.read(home_workspace::HEARTBEAT_FILE).unwrap_or_default(),
                );
                println!("heartbeat every {every} min ({} check(s) in HEARTBEAT.md)", checks.len());
                if checks.is_empty() {
                    println!("HEARTBEAT.md has no `- ` items yet, so ticks will do nothing (and cost nothing)");
                }
            }
            s.save(home.root())
        }),
        HomeCmd::Hours { range } => match home_schedule::ActiveHours::parse(&range) {
            Some(h) => load().and_then(|mut s| {
                s.active_hours = Some(h);
                println!("active hours {}-{}", h.start.format("%H:%M"), h.end.format("%H:%M"));
                s.save(home.root())
            }),
            None => Err(format!("expected HH:MM-HH:MM, got `{range}`")),
        },
        HomeCmd::Remind { when, text } => {
            let text = text.join(" ");
            if text.trim().is_empty() {
                Err("reminder text is empty".to_string())
            } else {
                match home_schedule::parse_when(&when, now, offset) {
                    Some(at) => load().and_then(|mut s| {
                        let id = s.add_reminder(at, text, now);
                        println!(
                            "{id}: {}",
                            at.with_timezone(&offset).format("%Y-%m-%d %H:%M")
                        );
                        s.save(home.root())
                    }),
                    None => Err(format!(
                        "cannot read `{when}` (try +30m, 15:00, \"2026-11-02 09:00\")"
                    )),
                }
            }
        }
        HomeCmd::List => load().map(|s| {
            let h = s.active_hours();
            println!(
                "active hours {}-{}",
                h.start.format("%H:%M"),
                h.end.format("%H:%M")
            );
            if s.jobs.is_empty() {
                println!("no jobs");
            }
            for j in &s.jobs {
                let due = j.next_due.with_timezone(&offset).format("%m-%d %H:%M");
                match &j.kind {
                    home_schedule::JobKind::Heartbeat { every_secs } => {
                        println!("{}  heartbeat every {} min, next {due}", j.id, every_secs / 60)
                    }
                    home_schedule::JobKind::Reminder { text, .. } => {
                        println!("{}  reminder {due}  {text}", j.id)
                    }
                }
            }
        }),
        HomeCmd::Remove { id } => load().and_then(|mut s| {
            if s.remove(&id) {
                println!("removed {id}");
                s.save(home.root())
            } else {
                Err(format!("no job `{id}`"))
            }
        }),
        HomeCmd::Tick { run } => {
            let (cfg, notifier) = home_runner_parts(&run);
            let delivery = HomeDelivery::new(home.clone(), notifier, true);
            home_runner::tick(&home, &cfg, &delivery, now, offset)
                .await
                .map(|_| ())
        }
        HomeCmd::Run { run } => {
            let (cfg, notifier) = home_runner_parts(&run);
            home_runner::run_forever(home.clone(), cfg, notifier).await
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kotonia-cli home: {e}");
            ExitCode::from(1)
        }
    }
}

fn home_runner_parts(
    run: &HomeRunArgs,
) -> (
    home_runner::RunnerConfig,
    Option<std::sync::Arc<dyn notifier::Notifier>>,
) {
    let cfg = home_runner::RunnerConfig {
        model: run.model.clone(),
        provider: run.provider.clone(),
        poll_secs: run.poll_secs,
        max_iterations: 15,
        heartbeat_timeout: std::time::Duration::from_secs(600),
        dry_run: run.dry_run,
    };
    let notifier = if run.no_notifier || run.dry_run {
        None
    } else {
        notifier::load_notifier_config()
            .map(|c| notifier::build_notifier(&c, reqwest::Client::new()))
    };
    (cfg, notifier)
}

fn frontend_system_prompt_prefix() -> Option<String> {
    let mut blocks = Vec::new();
    if let Ok(persona) = std::env::var("KOTONIA_PERSONA_PREFIX") {
        if !persona.trim().is_empty() {
            blocks.push(persona.trim().to_string());
        }
    }
    if let Ok(language) = std::env::var("KOTONIA_REPLY_LANGUAGE") {
        let language = language.trim();
        if !language.is_empty() {
            let instruction = if language == "ja" {
                "デフォルトでは日本語で回答してください。ユーザーが他の言語で書いた場合は、その言語で回答してください。".to_string()
            } else {
                format!(
                    "Reply in \"{language}\" by default. If the user writes in another language, reply in that language instead."
                )
            };
            blocks.push(instruction);
        }
    }
    (!blocks.is_empty()).then(|| blocks.join("\n\n"))
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

#[derive(Serialize)]
struct AuthStatusOutput {
    logged_in: bool,
    source: &'static str,
    valid: Option<bool>,
    server: Option<String>,
    device_id_prefix: Option<String>,
    error: Option<String>,
}

async fn run_auth_status(args: AuthStatusArgs) -> ExitCode {
    let stored = daemon_config::load();
    let env_key = std::env::var("KOTONIA_API_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty());

    let mut output = if let Some(cfg) = stored.as_ref() {
        AuthStatusOutput {
            logged_in: true,
            source: "device",
            valid: None,
            server: Some(cfg.server.clone()),
            device_id_prefix: Some(cfg.device_id.chars().take(8).collect()),
            error: None,
        }
    } else if env_key.is_some() {
        AuthStatusOutput {
            logged_in: true,
            source: "api_key",
            valid: None,
            server: std::env::var("KOTONIA_API_BASE").ok(),
            device_id_prefix: None,
            error: None,
        }
    } else {
        AuthStatusOutput {
            logged_in: false,
            source: "none",
            valid: None,
            server: None,
            device_id_prefix: None,
            error: None,
        }
    };

    if args.validate {
        if let Some(cfg) = stored.as_ref() {
            match validate_device_auth(cfg).await {
                Ok(valid) => {
                    output.valid = Some(valid);
                    output.logged_in = valid;
                }
                Err(e) => output.error = Some(e),
            }
        }
    }

    if args.json {
        println!(
            "{}",
            serde_json::to_string(&output).unwrap_or_else(|_| {
                r#"{"logged_in":false,"source":"none","valid":null}"#.to_string()
            })
        );
    } else if output.logged_in {
        let validity = match output.valid {
            Some(true) => "validated",
            Some(false) => "invalid",
            None => "not validated",
        };
        println!(
            "logged in via {} ({validity}){}",
            output.source,
            output
                .device_id_prefix
                .as_deref()
                .map(|id| format!(", device {id}"))
                .unwrap_or_default()
        );
        if let Some(error) = output.error.as_deref() {
            eprintln!("validation unavailable: {error}");
        }
    } else {
        println!("not logged in");
    }

    if output.valid == Some(false) || !output.logged_in {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn validate_device_auth(cfg: &daemon_config::DaemonStoredConfig) -> Result<bool, String> {
    let url = format!(
        "{}/api/agent-runtime/auth-status/{}",
        cfg.server.trim_end_matches('/'),
        cfg.device_id
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let response = client
        .get(url)
        .bearer_auth(&cfg.device_token)
        .header(
            kotonia_cli::client_info::HEADER,
            kotonia_cli::client_info::header_value(),
        )
        .send()
        .await
        .map_err(|e| format!("auth status request: {e}"))?;
    match response.status() {
        status if status.is_success() => Ok(true),
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => Ok(false),
        reqwest::StatusCode::NOT_FOUND => Err(
            "server does not expose device auth-status yet; local credentials are present"
                .to_string(),
        ),
        status => {
            let body = response.text().await.unwrap_or_default();
            Err(format!("auth status returned {status}: {body}"))
        }
    }
}

fn run_logout(args: LogoutArgs) -> ExitCode {
    match daemon_config::remove() {
        Ok(path) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "logged_in": false,
                        "removed": path.is_some(),
                    })
                );
            } else {
                println!("logged out");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "logged_in": false,
                        "removed": false,
                        "error": e,
                    })
                );
            } else {
                eprintln!("kotonia-cli logout: {e}");
            }
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
            eprintln!(
                "kotonia-cli pair-notifier: discord support is planned but not implemented yet."
            );
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
                    .and_then(|d| {
                        chrono::DateTime::<chrono::Utc>::from_timestamp(d.as_secs() as i64, 0)
                    }) {
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
            Event::Text { text } => {
                // Streamed prose from Claude Code (between tool calls).
                // Print on its own line so it doesn't run into the
                // "· thinking" marker that precedes it.
                println!("\n{}", text.trim_end());
            }
            Event::Bash { command } => {
                println!("\n$ {command}");
            }
            Event::ConnectorRequested { title, .. } => {
                println!("\n[connector] {title} への接続を求めています");
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
            Event::InspectImage {
                path,
                size_bytes,
                error,
            } => match error {
                None => println!("\n[inspect {path} ({size_bytes} bytes)]"),
                Some(msg) => println!("\n[inspect {path} failed: {msg}]"),
            },
            Event::Final { answer } => {
                println!("\n══ final answer ══");
                // Claude Code path emits an empty `answer` when the text
                // was already streamed via `Text`. Skip the blank line in
                // that case so the separator stays clean.
                if !answer.is_empty() {
                    println!("{answer}");
                }
            }
            Event::ContextCompacted {
                before_tokens,
                after_tokens,
                stubbed,
                summarised,
                summary_failed,
            } => {
                println!(
                    "\n[context compacted: ~{before_tokens} → ~{after_tokens} tokens{}]",
                    compaction_detail(stubbed, summarised, summary_failed)
                );
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
        Self { stdin: io::stdin() }
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

    fn request_connection(&mut self, card: &connectors::ConnectorCard) -> ConnectionReply {
        if !atty_stdin() {
            return ConnectionReply::Unavailable;
        }
        match kotonia_cli::tty::fill_connector_card(card, true) {
            Ok(Some(values)) => ConnectionReply::Provided(values),
            Ok(None) => ConnectionReply::Declined,
            Err(_) => ConnectionReply::Unavailable,
        }
    }
}

/// ", 3 old outputs stubbed, 12 older messages summarised" for the TTY line.
fn compaction_detail(stubbed: u64, summarised: u64, summary_failed: bool) -> String {
    let mut parts = Vec::new();
    if stubbed > 0 {
        parts.push(format!("{stubbed} old outputs stubbed"));
    }
    if summarised > 0 {
        parts.push(format!("{summarised} older messages summarised"));
    }
    if summary_failed {
        parts.push("summary failed, outputs stubbed instead".to_string());
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}
