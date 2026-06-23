//! Persisted daemon credentials (~/.kotonia/daemon.json).
//!
//! `kotonia-cli login` writes here after a successful device-code flow.
//! `kotonia-cli daemon` reads here when no env / flag overrides are set.
//!
//! Format is JSON to avoid adding a TOML parser dependency — the file is
//! never edited by humans, only by the CLI.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
