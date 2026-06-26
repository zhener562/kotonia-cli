//! Independent approval channel for the daemon.
//!
//! When the daemon receives a `RunTask` from kotonia.ai for a browser
//! session it doesn't yet trust, it asks a `Notifier` (Telegram or
//! Discord) to push an Approve/Deny prompt to the operator's phone.
//! The notifier's transport never touches kotonia.ai, so a compromised
//! backend can't forge an approve signal — the attacker would have to
//! separately steal the operator's Telegram/Discord account.
//!
//! Persisted to `~/.kotonia/notifier.json` (mode 0600) once paired.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod telegram;

/// Approval channel kind. Selected at pair time; baked into the stored config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotifierKind {
    Telegram,
    // `Discord` lands in a follow-up commit.
}

/// What `kotonia-cli pair-notifier` writes to `~/.kotonia/notifier.json`.
/// The token is a long-lived bearer for the third-party platform, so the
/// file is locked down to 0600.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotifierStoredConfig {
    pub kind: NotifierKind,
    /// Bot token (Telegram BotFather token / Discord application bot token).
    pub bot_token: String,
    /// Telegram `chat_id` (as string for forwards-compat with Discord's
    /// snowflake-shaped ids).
    pub chat_id: String,
    /// Display name captured at pair time. Echoed back in logs so the
    /// operator can confirm the right account is wired up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

/// What the daemon asks the operator to approve.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The browser session whose first task we're gating. `None` means the
    /// web frontend didn't include one — typically because the operator is
    /// hitting an older `/agent` page that predates the localStorage
    /// integration. The notifier surfaces that explicitly so the operator
    /// knows to redeploy/reload before approval can persist.
    pub browser_session_id: Option<String>,
    /// First ~200 chars of the prompt — enough for the operator to decide
    /// "yeah, that's me" vs "what is this?". Truncated upstream.
    pub prompt_excerpt: String,
    /// Origin IP of the web request, if backend forwarded one. Helps catch
    /// "someone else is driving my account" from a foreign IP.
    pub origin_ip: Option<String>,
    /// Browser User-Agent excerpt, again for the "is this me?" check.
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Trust this browser session for `trust_ttl` (24h default).
    Approve,
    /// Block this session and surface an error back to the web caller.
    Deny,
    /// No reply before the deadline — treated as Deny but logged separately.
    Timeout,
}

#[derive(Debug)]
pub enum NotifierError {
    Network(String),
    Auth(String),
    Other(String),
}

impl std::fmt::Display for NotifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotifierError::Network(m) => write!(f, "notifier network: {m}"),
            NotifierError::Auth(m) => write!(f, "notifier auth: {m}"),
            NotifierError::Other(m) => write!(f, "notifier: {m}"),
        }
    }
}

impl std::error::Error for NotifierError {}

#[async_trait]
pub trait Notifier: Send + Sync {
    /// Verify the channel is reachable and the credentials work. Used at
    /// pair time and by the daemon at startup. Returns a friendly identifier
    /// for the channel (e.g. Telegram bot username) for log lines.
    async fn ping(&self) -> Result<String, NotifierError>;

    /// Send an approval request and block until the operator decides or
    /// `timeout` elapses. Implementations are responsible for editing the
    /// pushed message after the fact so the operator can see the outcome.
    async fn request_approval(
        &self,
        req: ApprovalRequest,
        timeout: Duration,
    ) -> Result<ApprovalDecision, NotifierError>;
}

// ── persistence ─────────────────────────────────────────────────────────

pub fn notifier_config_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".kotonia").join("notifier.json"))
}

pub fn load_notifier_config() -> Option<NotifierStoredConfig> {
    let path = notifier_config_path()?;
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn save_notifier_config(cfg: &NotifierStoredConfig) -> Result<PathBuf, String> {
    let path = notifier_config_path().ok_or_else(|| "HOME is not set".to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&path, perms)
            .map_err(|e| format!("chmod {}: {e}", path.display()))?;
    }
    Ok(path)
}

/// Build a live notifier from a stored config. Returns an `Arc` so the
/// daemon can clone the handle into per-task spawned futures without
/// touching the underlying transport state.
pub fn build_notifier(
    cfg: &NotifierStoredConfig,
    http: reqwest::Client,
) -> Arc<dyn Notifier> {
    match cfg.kind {
        NotifierKind::Telegram => Arc::new(telegram::TelegramNotifier::new(
            cfg.bot_token.clone(),
            cfg.chat_id.clone(),
            http,
        )),
    }
}
