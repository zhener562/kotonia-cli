//! `kotonia-cli login` — device-code pair flow against kotonia.ai.
//!
//! Prints the verification URL + user code, polls until the user approves
//! it from a logged-in browser tab, then persists the issued
//! device_id / device_token to ~/.kotonia/daemon.json so subsequent
//! `kotonia-cli daemon` invocations need no flags.
//!
//! `create_device_code` / `poll_once` are exposed publicly (not just used by
//! `run`'s CLI loop below) so kotonia-desktop can drive the same flow from
//! its own GUI — one poll call per tick, driven by the frontend's timer,
//! instead of a blocking `sleep` loop.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::config::{save as save_config, DaemonStoredConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeSession {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: i64,
    pub interval: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PollOutcome {
    Pending,
    Approved {
        device_id: String,
        device_token: String,
    },
}

#[derive(Debug, Serialize)]
struct EmptyBody {}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("http client: {e}"))
}

/// POST /api/agent-runtime/device-codes — start a new pairing session.
pub async fn create_device_code(server: &str) -> Result<DeviceCodeSession, String> {
    let server = server.trim_end_matches('/');
    let http = http_client()?;
    let resp = http
        .post(format!("{server}/api/agent-runtime/device-codes"))
        .json(&EmptyBody {})
        .send()
        .await
        .map_err(|e| format!("POST device-codes: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("POST device-codes returned {status}: {body}"));
    }
    resp.json()
        .await
        .map_err(|e| format!("parse device-code response: {e}"))
}

/// GET /api/agent-runtime/device-codes/{device_code} — one poll attempt.
/// A `410 Gone` (expired or already consumed) is reported as an `Err`,
/// same as any other non-success status — there's no `Expired` variant to
/// match on separately.
pub async fn poll_once(server: &str, device_code: &str) -> Result<PollOutcome, String> {
    let server = server.trim_end_matches('/');
    let http = http_client()?;
    let resp = http
        .get(format!("{server}/api/agent-runtime/device-codes/{device_code}"))
        .send()
        .await
        .map_err(|e| format!("poll: {e}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::GONE {
        return Err("device code expired or already consumed".to_string());
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("poll returned {status}: {body}"));
    }
    resp.json()
        .await
        .map_err(|e| format!("parse poll response: {e}"))
}

/// Persist an approved device pairing to `~/.kotonia/daemon.json`.
pub fn save_pairing(server: &str, device_id: String, device_token: String) -> Result<std::path::PathBuf, String> {
    save_config(&DaemonStoredConfig {
        server: server.trim_end_matches('/').to_string(),
        device_id,
        device_token,
    })
}

pub async fn run(server: &str) -> Result<(), String> {
    let code = create_device_code(server).await?;

    println!();
    println!("─────────────────────────────────────────────");
    println!("  Open this URL in a logged-in browser tab:");
    println!();
    println!("     {}", code.verification_uri);
    println!();
    println!("  Then enter this code:");
    println!();
    println!("     {}", code.user_code);
    println!();
    println!("  (expires in {} minutes)", code.expires_in / 60);
    println!("─────────────────────────────────────────────");
    println!();

    let interval = Duration::from_secs(code.interval.max(1) as u64);

    print!("Waiting for approval");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();

    loop {
        sleep(interval).await;
        print!(".");
        let _ = std::io::stdout().flush();

        match poll_once(server, &code.device_code).await {
            Ok(PollOutcome::Pending) => continue,
            Ok(PollOutcome::Approved {
                device_id,
                device_token,
            }) => {
                println!(" approved!");
                let path = save_pairing(server, device_id.clone(), device_token)?;
                println!();
                println!("Paired as device {}.", &device_id[..8.min(device_id.len())]);
                println!("Saved to {}", path.display());
                println!();
                println!("Run `kotonia-cli daemon` to connect.");
                return Ok(());
            }
            Err(e) => return Err(format!("\n{e}")),
        }
    }
}
