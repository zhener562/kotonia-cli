//! `kotonia-cli login` — device-code pair flow against kotonia.ai.
//!
//! Prints the verification URL + user code, polls until the user approves
//! it from a logged-in browser tab, then persists the issued
//! device_id / device_token to ~/.kotonia/daemon.json so subsequent
//! `kotonia-cli daemon` invocations need no flags.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::config::{save as save_config, DaemonStoredConfig};

#[derive(Debug, Deserialize)]
struct CreateDeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: i64,
    interval: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum PollResponse {
    Pending,
    Approved {
        device_id: String,
        device_token: String,
    },
}

#[derive(Debug, Serialize)]
struct EmptyBody {}

pub async fn run(server: &str) -> Result<(), String> {
    let server = server.trim_end_matches('/');
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let create_url = format!("{server}/api/agent-runtime/device-codes");
    let resp = http
        .post(&create_url)
        .json(&EmptyBody {})
        .send()
        .await
        .map_err(|e| format!("POST device-codes: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("POST device-codes returned {status}: {body}"));
    }
    let code: CreateDeviceCodeResponse = resp
        .json()
        .await
        .map_err(|e| format!("parse device-code response: {e}"))?;

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

    let poll_url = format!("{server}/api/agent-runtime/device-codes/{}", code.device_code);
    let interval = Duration::from_secs(code.interval.max(1) as u64);

    print!("Waiting for approval");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();

    loop {
        sleep(interval).await;
        print!(".");
        let _ = std::io::stdout().flush();

        let resp = http
            .get(&poll_url)
            .send()
            .await
            .map_err(|e| format!("\npoll: {e}"))?;
        let status = resp.status();
        if status == reqwest::StatusCode::GONE {
            return Err("\ndevice code expired or already consumed".to_string());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("\npoll returned {status}: {body}"));
        }
        let poll: PollResponse = resp
            .json()
            .await
            .map_err(|e| format!("\nparse poll response: {e}"))?;
        match poll {
            PollResponse::Pending => continue,
            PollResponse::Approved {
                device_id,
                device_token,
            } => {
                println!(" approved!");
                let cfg = DaemonStoredConfig {
                    server: server.to_string(),
                    device_id: device_id.clone(),
                    device_token,
                };
                let path = save_config(&cfg)?;
                println!();
                println!("Paired as device {}.", &device_id[..8.min(device_id.len())]);
                println!("Saved to {}", path.display());
                println!();
                println!("Run `kotonia-cli daemon` to connect.");
                return Ok(());
            }
        }
    }
}
