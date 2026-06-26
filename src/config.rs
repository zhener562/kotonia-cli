//! Persisted CLI state under `~/.kotonia/`:
//!
//!   - `daemon.json` — credentials written by `kotonia-cli login` (device-code
//!     flow). Loaded by `daemon` runs and by the `kotonia` built-in provider
//!     as a `device_token` fallback for `/api/v1/*`.
//!   - `providers.json` — optional user-supplied provider catalog merged into
//!     the built-in registry. Never written by the CLI; humans edit it.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStoredConfig {
    pub server: String,
    pub device_id: String,
    pub device_token: String,
}

/// Returns `$HOME/.kotonia/daemon.json`, or None if $HOME is unset.
pub fn config_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".kotonia").join("daemon.json"))
}

pub fn load() -> Option<DaemonStoredConfig> {
    let path = config_path()?;
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn save(cfg: &DaemonStoredConfig) -> Result<PathBuf, String> {
    let path = config_path().ok_or_else(|| "HOME is not set".to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg)
        .map_err(|e| format!("serialize config: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    // Tighten perms: contains a long-lived bearer token. 0600 on unix.
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

/// Returns `$HOME/.kotonia/providers.json`, or None if $HOME is unset.
pub fn providers_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".kotonia").join("providers.json"))
}

/// One entry in `providers.json`. Every field is optional so the file can
/// stay minimal; the registry fills in built-in defaults for missing pieces.
///
/// Auth resolution: `api_key_env` first (read env at startup), then literal
/// `api_key`, then None.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProviderFileEntry {
    /// Full base URL of the OpenAI-compatible endpoint (e.g.
    /// `"https://api.openai.com/v1"`). Required.
    pub base_url: String,
    /// Env var holding the bearer token. Checked first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Literal bearer token. Use only for local/testing — checked-in tokens
    /// are a leak waiting to happen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Default model id when `--model` is omitted and this provider is selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// `"max_tokens"` (default), `"max_completion_tokens"` (GPT-5/4.1/o-series),
    /// `"max_output_tokens"` (xAI Grok responses endpoint, etc).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_param: Option<String>,
    /// Cap on the per-request `max_tokens`. Useful when a backend chokes on
    /// big completions even though the protocol allows them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_cap: Option<u32>,
    /// Extra request headers (e.g. `OpenAI-Organization`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_headers: Vec<[String; 2]>,
    /// Extra fields merged into the request body. Lets you set things like
    /// `temperature` or vendor-specific knobs without code changes.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra_body: Map<String, Value>,
    /// Models served by this provider, used by the registry to infer a
    /// provider when `--provider` is omitted but `--model <id>` matches.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProvidersFile {
    #[serde(default)]
    pub providers: HashMap<String, ProviderFileEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_provider: Option<String>,
}

/// Returns `None` if the file is missing (the common case for new installs).
/// Parse errors bubble — a malformed JSON file should fail loud, not be
/// silently ignored.
pub fn load_providers() -> Result<Option<ProvidersFile>, String> {
    let Some(path) = providers_path() else {
        return Ok(None);
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let parsed: ProvidersFile = serde_json::from_str(&raw)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    Ok(Some(parsed))
}
