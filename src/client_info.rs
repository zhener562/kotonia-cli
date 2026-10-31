//! Who is calling: the `x-kotonia-client` header sent on every request to
//! kotonia.ai (and only there — third-party providers never see it).
//!
//! ```text
//! cli/0.1.11 (linux-x86_64; wsl)
//! cli/0.1.11 (windows-x86_64) serve
//! ```
//!
//! Lets the server tell a CLI that stopped working on WSL from one on native
//! Linux or Windows. Embedders (kotonia-desktop) call [`set_client_name`]
//! before the first request so their traffic is not counted as the CLI.

use std::sync::OnceLock;

pub const HEADER: &str = "x-kotonia-client";

static NAME: OnceLock<&'static str> = OnceLock::new();
static MODE: OnceLock<&'static str> = OnceLock::new();
static VALUE: OnceLock<String> = OnceLock::new();

/// Override the product name (default `cli`). First call wins; must run
/// before anything reads [`header_value`].
pub fn set_client_name(name: &'static str) {
    let _ = NAME.set(name);
}

/// Mark how the binary was driven, e.g. `serve` when the VS Code extension
/// runs it over the JSON stdio protocol. First call wins.
pub fn set_mode(mode: &'static str) {
    let _ = MODE.set(mode);
}

pub fn header_value() -> &'static str {
    VALUE.get_or_init(|| {
        let name = NAME.get().copied().unwrap_or("cli");
        let version = env!("CARGO_PKG_VERSION");
        let platform = platform();
        let wsl = if is_wsl() { "; wsl" } else { "" };
        match MODE.get() {
            Some(mode) => format!("{name}/{version} ({platform}{wsl}) {mode}"),
            None => format!("{name}/{version} ({platform}{wsl})"),
        }
    })
}

pub fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// True inside Windows Subsystem for Linux (1 or 2).
pub fn is_wsl() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    if std::env::var_os("WSL_DISTRO_NAME").is_some() {
        return true;
    }
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|release| release.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_names_the_product_version_and_platform() {
        let value = header_value();
        assert!(value.starts_with("cli/"), "{value}");
        assert!(value.contains(env!("CARGO_PKG_VERSION")), "{value}");
        assert!(value.contains(&platform()), "{value}");
        assert!(value.is_ascii());
    }
}
