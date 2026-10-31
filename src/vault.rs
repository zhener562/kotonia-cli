//! Secret storage for connectors: one file per secret under
//! `~/.kotonia/vault/` (mode 0600 in a 0700 directory), the same protection
//! `daemon.json` and `notifier.json` already rely on.
//!
//! The point of the vault is that **secret values never enter the model's
//! context**: they are typed by the user into a hidden prompt (not into the
//! chat), read by Rust code that needs them (the IMAP client), and scrubbed
//! from every command output before the agent sees it ([`redact`]).
//!
//! What this does *not* do: stop a process running as the same user from
//! reading the files. The agent's bash runs as the user, so two further
//! layers sit in front of it — the approval policy treats any command that
//! touches the vault path as needing a human, and [`redact`] replaces any
//! secret value that still shows up in output. Redaction is exact-match:
//! a value that was transformed (base64, split, reversed) on the way out is
//! not caught. An OS keychain would not change that picture on Linux
//! (`secret-tool lookup` needs no prompt either), so a file vault was chosen
//! for portability; the API is small enough to swap later.

use std::path::{Path, PathBuf};

pub const REDACTED: &str = "[redacted secret]";

/// Shorter values are too likely to collide with ordinary output ("1234",
/// "true") to redact safely, and too weak to be worth protecting this way.
pub const MIN_REDACT_LEN: usize = 8;

#[derive(Debug, Clone)]
pub struct Vault {
    dir: PathBuf,
}

impl Vault {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// `$KOTONIA_VAULT_DIR`, else `~/.kotonia/vault`.
    pub fn resolve() -> Result<Self, String> {
        if let Some(dir) = std::env::var_os("KOTONIA_VAULT_DIR") {
            if !dir.is_empty() {
                return Ok(Self::new(PathBuf::from(dir)));
            }
        }
        let home = dirs::home_dir().ok_or_else(|| "HOME is not set".to_string())?;
        Ok(Self::new(home.join(".kotonia").join("vault")))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, name: &str) -> Result<PathBuf, String> {
        validate_name(name)?;
        Ok(self.dir.join(name))
    }

    pub fn set(&self, name: &str, value: &str) -> Result<(), String> {
        if value.is_empty() {
            return Err("secret value is empty".to_string());
        }
        let path = self.path_for(name)?;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| format!("create {}: {e}", self.dir.display()))?;
        restrict(&self.dir, 0o700)?;
        // Write to a temp file that is 0600 from birth, then rename, so the
        // value is never briefly world-readable and never half-written.
        let tmp = self.dir.join(format!(".{name}.tmp"));
        {
            use std::io::Write;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            let mut f = opts
                .open(&tmp)
                .map_err(|e| format!("write {}: {e}", tmp.display()))?;
            f.write_all(value.as_bytes())
                .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))
    }

    pub fn get(&self, name: &str) -> Result<Option<String>, String> {
        let path = self.path_for(name)?;
        match std::fs::read_to_string(&path) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    pub fn remove(&self, name: &str) -> Result<bool, String> {
        let path = self.path_for(name)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(format!("remove {}: {e}", path.display())),
        }
    }

    pub fn names(&self) -> Vec<String> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| validate_name(n).is_ok())
            .collect();
        out.sort();
        out
    }

    /// Every stored value long enough to redact. Read fresh on each call so a
    /// secret added mid-session is covered from the next command on.
    pub fn values_for_redaction(&self) -> Vec<String> {
        self.names()
            .iter()
            .filter_map(|n| self.get(n).ok().flatten())
            .map(|v| v.trim().to_string())
            .filter(|v| v.chars().count() >= MIN_REDACT_LEN)
            .collect()
    }
}

/// Secret names are file names: keep them boring.
pub fn validate_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid secret name `{name}` (letters, digits, `_ - .`, not starting with `.`)"
        ))
    }
}

fn restrict(path: &Path, mode: u32) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
            .map_err(|e| format!("chmod {}: {e}", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = (path, mode);
    Ok(())
}

/// Replace every occurrence of any secret in `text`. Longest first, so a
/// secret that contains another is not half-redacted.
pub fn redact_with(text: &str, secrets: &[String]) -> String {
    let mut secrets: Vec<&String> = secrets
        .iter()
        .filter(|s| s.chars().count() >= MIN_REDACT_LEN)
        .collect();
    if secrets.is_empty() {
        return text.to_string();
    }
    secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let mut out = text.to_string();
    for s in secrets {
        if out.contains(s.as_str()) {
            out = out.replace(s.as_str(), REDACTED);
        }
    }
    out
}

/// Redact using the default vault. Never fails: with no vault there is
/// nothing to hide.
pub fn redact(text: &str) -> String {
    match Vault::resolve() {
        Ok(v) => redact_with(text, &v.values_for_redaction()),
        Err(_) => text.to_string(),
    }
}

/// True when a shell command refers to the vault directory. The approval
/// policy gates these even in `allowlist` mode.
pub fn command_touches_vault(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if lower.contains(".kotonia/vault") || lower.contains("kotonia_vault_dir") {
        return true;
    }
    match Vault::resolve() {
        Ok(v) => {
            let dir = v.dir().to_string_lossy().to_ascii_lowercase();
            !dir.is_empty() && lower.contains(&dir)
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_remove_and_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let v = Vault::new(dir.path().join("vault"));
        v.set("imap.gmail", "abcd efgh ijkl mnop").unwrap();
        assert_eq!(
            v.get("imap.gmail").unwrap().as_deref(),
            Some("abcd efgh ijkl mnop")
        );
        assert_eq!(v.names(), vec!["imap.gmail"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&v.dir().join("imap.gmail")), 0o600);
            assert_eq!(mode(v.dir()), 0o700);
        }
        assert!(v.remove("imap.gmail").unwrap());
        assert!(!v.remove("imap.gmail").unwrap());
        assert_eq!(v.get("imap.gmail").unwrap(), None);
    }

    #[test]
    fn names_reject_path_tricks() {
        for bad in ["", "../x", "a/b", ".hidden", "with space"] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
        assert!(validate_name("kotonia_tts-2.key").is_ok());
    }

    #[test]
    fn redaction_is_exact_longest_first_and_ignores_short_values() {
        let secrets = vec![
            "sk-live-abcdef123456".to_string(),
            "abcdef123456".to_string(),
            "1234".to_string(),
        ];
        let out = redact_with(
            "key=sk-live-abcdef123456 other=abcdef123456 pin=1234",
            &secrets,
        );
        assert_eq!(out, format!("key={REDACTED} other={REDACTED} pin=1234"));
    }

    #[test]
    fn vault_paths_are_detected() {
        assert!(command_touches_vault("cat ~/.kotonia/vault/imap.gmail"));
        assert!(command_touches_vault("ls $KOTONIA_VAULT_DIR"));
        assert!(!command_touches_vault("cat ~/.kotonia/home/MEMORY.md"));
    }
}
