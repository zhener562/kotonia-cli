//! Telegram bot transport for [`super::Notifier`].
//!
//! Wire path:
//! ```text
//!   daemon ─HTTPS─> api.telegram.org ─push─> operator's Telegram app
//!                                              ↓ inline button tap
//!   daemon <─long-poll getUpdates── api.telegram.org
//! ```
//!
//! kotonia.ai backend is **not** in this path. The bot token is created by
//! the operator with @BotFather and stored locally; an attacker who owns
//! the kotonia.ai backend still cannot forge an Approve.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use super::{ApprovalDecision, ApprovalRequest, Notifier, NotifierError, NotifierKind, NotifierStoredConfig};

const API_BASE: &str = "https://api.telegram.org";
/// Prefix for our `callback_data` strings — guards against unrelated bot
/// chatter accidentally satisfying an approval poll.
const CALLBACK_PREFIX: &str = "kotonia_approval";

pub struct TelegramNotifier {
    bot_token: String,
    chat_id: String,
    http: reqwest::Client,
}

impl TelegramNotifier {
    pub fn new(bot_token: String, chat_id: String, http: reqwest::Client) -> Self {
        Self {
            bot_token,
            chat_id,
            http,
        }
    }

    fn url(&self, method: &str) -> String {
        format!("{API_BASE}/bot{}/{method}", self.bot_token)
    }
}

// ── Telegram Bot API response types (just the subset we read) ───────────

#[derive(Debug, Deserialize)]
struct TgResp<T> {
    ok: bool,
    // serde maps missing `Option<T>` to None without needing `#[serde(default)]`
    // (which would force T: Default and bite us on generics).
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TgUser {
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    first_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TgMessage {
    message_id: i64,
}

#[derive(Debug, Deserialize)]
struct TgUpdate {
    update_id: i64,
    #[serde(default)]
    callback_query: Option<TgCallbackQuery>,
    #[serde(default)]
    message: Option<TgUpdateMessage>,
}

#[derive(Debug, Deserialize)]
struct TgCallbackQuery {
    id: String,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    from: Option<TgUser>,
}

#[derive(Debug, Deserialize)]
struct TgUpdateMessage {
    chat: TgChat,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    from: Option<TgUser>,
}

#[derive(Debug, Deserialize)]
struct TgChat {
    id: i64,
}

#[async_trait]
impl Notifier for TelegramNotifier {
    async fn ping(&self) -> Result<String, NotifierError> {
        let resp: TgResp<TgUser> = self
            .http
            .get(self.url("getMe"))
            .send()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?
            .json()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?;
        if !resp.ok {
            return Err(NotifierError::Auth(
                resp.description.unwrap_or_else(|| "getMe failed".into()),
            ));
        }
        let user = resp
            .result
            .ok_or_else(|| NotifierError::Other("getMe returned no result".into()))?;
        Ok(user
            .username
            .or(user.first_name)
            .unwrap_or_else(|| "unknown".into()))
    }

    async fn request_approval(
        &self,
        req: ApprovalRequest,
        timeout: Duration,
    ) -> Result<ApprovalDecision, NotifierError> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let approve_data = format!("{CALLBACK_PREFIX}:approve:{request_id}");
        let deny_data = format!("{CALLBACK_PREFIX}:deny:{request_id}");

        let text = compose_request_text(&req);

        let body = serde_json::json!({
            "chat_id": self.chat_id,
            "text": text,
            // HTML mode is more forgiving than Markdown — only `<`, `>`, `&`
            // need escaping in user-supplied substrings (handled in
            // `compose_request_text`), and stray backticks / asterisks in
            // the operator's prompt won't break parsing the way they do in
            // Markdown mode.
            "parse_mode": "HTML",
            "reply_markup": {
                "inline_keyboard": [[
                    {"text": "✅ Approve 24h", "callback_data": approve_data.clone()},
                    {"text": "❌ Deny",       "callback_data": deny_data.clone()},
                ]]
            }
        });

        let resp: TgResp<TgMessage> = self
            .http
            .post(self.url("sendMessage"))
            .json(&body)
            .send()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?
            .json()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?;
        if !resp.ok {
            return Err(NotifierError::Other(
                resp.description
                    .unwrap_or_else(|| "sendMessage failed".into()),
            ));
        }
        let message_id = resp
            .result
            .ok_or_else(|| NotifierError::Other("sendMessage returned no result".into()))?
            .message_id;

        // Long-poll for callback_query until timeout or match.
        let started = std::time::Instant::now();
        let mut last_update_id: i64 = 0;
        loop {
            if started.elapsed() >= timeout {
                let _ = self
                    .edit_message(
                        message_id,
                        &format!(
                            "⏱️ Approval timed out for browser session {}.",
                            session_label(&req.browser_session_id)
                        ),
                    )
                    .await;
                return Ok(ApprovalDecision::Timeout);
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            // Long-poll for at most 25s per round; honor remaining budget.
            let poll_secs = remaining.as_secs().min(25).max(1) as u32;
            let updates = self.get_updates(last_update_id + 1, poll_secs).await?;
            for upd in updates {
                last_update_id = last_update_id.max(upd.update_id);
                let Some(cb) = upd.callback_query else {
                    continue;
                };
                let Some(data) = cb.data else {
                    continue;
                };
                if !data.starts_with(CALLBACK_PREFIX) {
                    continue;
                }
                let decision = if data == approve_data {
                    Some(ApprovalDecision::Approve)
                } else if data == deny_data {
                    Some(ApprovalDecision::Deny)
                } else {
                    None
                };
                // Always ack so the operator's button spinner clears.
                let _ = self.answer_callback(&cb.id).await;
                let Some(d) = decision else { continue };
                let who = cb
                    .from
                    .and_then(|u| u.username.or(u.first_name))
                    .unwrap_or_else(|| "?".into());
                let who_html = html_escape(&who);
                let final_text = match d {
                    ApprovalDecision::Approve => format!(
                        "✅ Approved by @{who_html} — browser session {} trusted for 24h.",
                        session_label(&req.browser_session_id)
                    ),
                    ApprovalDecision::Deny => format!(
                        "❌ Denied by @{who_html} — browser session {} blocked.",
                        session_label(&req.browser_session_id)
                    ),
                    ApprovalDecision::Timeout => unreachable!(),
                };
                let _ = self.edit_message(message_id, &final_text).await;
                return Ok(d);
            }
        }
    }
}

impl TelegramNotifier {
    async fn get_updates(
        &self,
        offset: i64,
        timeout_secs: u32,
    ) -> Result<Vec<TgUpdate>, NotifierError> {
        let body = serde_json::json!({
            "offset": offset,
            "timeout": timeout_secs,
            // Filter at source — we don't care about regular messages here.
            "allowed_updates": ["callback_query"],
        });
        // HTTP timeout has to outlast Telegram's long-poll timeout, with
        // some buffer so we don't kill an in-flight response.
        let resp: TgResp<Vec<TgUpdate>> = self
            .http
            .post(self.url("getUpdates"))
            .timeout(Duration::from_secs((timeout_secs as u64) + 5))
            .json(&body)
            .send()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?
            .json()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?;
        if !resp.ok {
            return Err(NotifierError::Other(
                resp.description.unwrap_or_else(|| "getUpdates failed".into()),
            ));
        }
        Ok(resp.result.unwrap_or_default())
    }

    async fn answer_callback(&self, callback_id: &str) -> Result<(), NotifierError> {
        let body = serde_json::json!({ "callback_query_id": callback_id });
        let _ = self
            .http
            .post(self.url("answerCallbackQuery"))
            .json(&body)
            .send()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?;
        Ok(())
    }

    async fn edit_message(&self, message_id: i64, text: &str) -> Result<(), NotifierError> {
        let body = serde_json::json!({
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "Markdown",
        });
        let _ = self
            .http
            .post(self.url("editMessageText"))
            .json(&body)
            .send()
            .await
            .map_err(|e| NotifierError::Network(e.to_string()))?;
        Ok(())
    }

    /// Long-poll for `/start <code>` (case-sensitive) from any chat the
    /// bot is in. Returns `(chat_id, display_name)` when matched.
    async fn wait_for_start(
        &self,
        code: &str,
        timeout: Duration,
    ) -> Result<(i64, Option<String>), NotifierError> {
        let started = std::time::Instant::now();
        let mut last_update_id: i64 = 0;
        let expected = format!("/start {code}");
        loop {
            if started.elapsed() >= timeout {
                return Err(NotifierError::Other(
                    "timed out waiting for /start message".into(),
                ));
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            let poll_secs = remaining.as_secs().min(25).max(1) as u32;
            let body = serde_json::json!({
                "offset": last_update_id + 1,
                "timeout": poll_secs,
                "allowed_updates": ["message"],
            });
            let resp: TgResp<Vec<TgUpdate>> = self
                .http
                .post(self.url("getUpdates"))
                .timeout(Duration::from_secs((poll_secs as u64) + 5))
                .json(&body)
                .send()
                .await
                .map_err(|e| NotifierError::Network(e.to_string()))?
                .json()
                .await
                .map_err(|e| NotifierError::Network(e.to_string()))?;
            if !resp.ok {
                return Err(NotifierError::Other(
                    resp.description.unwrap_or_else(|| "getUpdates failed".into()),
                ));
            }
            for upd in resp.result.unwrap_or_default() {
                last_update_id = last_update_id.max(upd.update_id);
                let Some(msg) = upd.message else { continue };
                let text = msg.text.unwrap_or_default();
                if text.trim() == expected {
                    let name = msg.from.and_then(|u| u.username.or(u.first_name));
                    return Ok((msg.chat.id, name));
                }
            }
        }
    }
}

/// Run the interactive `kotonia-cli pair-notifier telegram` flow.
/// Returns the validated config; caller writes it via
/// [`super::save_notifier_config`].
pub async fn pair(http: reqwest::Client) -> Result<NotifierStoredConfig, String> {
    use std::io::{BufRead, Write};

    eprintln!();
    eprintln!("─── Telegram bot setup ─────────────────────────");
    eprintln!();
    eprintln!("1. Open Telegram and message @BotFather");
    eprintln!("2. Send: /newbot");
    eprintln!("3. Follow the prompts (any name; username must end in 'bot')");
    eprintln!("4. Copy the bot token BotFather replies with");
    eprintln!();
    eprintln!("Why your own bot? It keeps the approval channel out of");
    eprintln!("kotonia.ai entirely — even if our backend is ever compromised,");
    eprintln!("the attacker still cannot forge an approve signal.");
    eprintln!();

    eprint!("Bot token: ");
    let _ = std::io::stderr().flush();
    let stdin = std::io::stdin();
    let mut buf = String::new();
    stdin
        .lock()
        .read_line(&mut buf)
        .map_err(|e| format!("read token: {e}"))?;
    let token = buf.trim().to_string();
    if token.is_empty() {
        return Err("empty token".into());
    }

    let probe = TelegramNotifier::new(token.clone(), "0".into(), http);
    let bot_name = probe.ping().await.map_err(|e| format!("verify bot: {e}"))?;
    eprintln!("✓ verified bot @{bot_name}");

    // Short uppercase code; long enough to make collision irrelevant in the
    // 5-minute window we wait, short enough to type comfortably.
    let code: String = uuid::Uuid::new_v4()
        .to_string()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(6)
        .collect::<String>()
        .to_ascii_uppercase();

    eprintln!();
    eprintln!("Now open your bot @{bot_name} in Telegram and send:");
    eprintln!();
    eprintln!("    /start {code}");
    eprintln!();
    eprintln!("Waiting up to 5 minutes...");

    let (chat_id, display_name) = probe
        .wait_for_start(&code, Duration::from_secs(300))
        .await
        .map_err(|e| format!("waiting for /start: {e}"))?;

    eprintln!();
    eprintln!(
        "✓ paired with chat_id {chat_id}{}",
        display_name
            .as_ref()
            .map(|n| format!(" (@{n})"))
            .unwrap_or_default()
    );

    Ok(NotifierStoredConfig {
        kind: NotifierKind::Telegram,
        bot_token: token,
        chat_id: chat_id.to_string(),
        display_name,
    })
}

fn compose_request_text(req: &ApprovalRequest) -> String {
    let mut text = String::new();
    text.push_str("🛡️ <b>Kotonia agent approval</b>\n\n");
    match &req.browser_session_id {
        Some(id) => {
            text.push_str(&format!(
                "New browser session: <code>{}</code>\n",
                html_escape(&short(id, 12))
            ));
        }
        None => {
            // No id means the operator's frontend predates the localStorage
            // integration. Approving still lets *this* task through, but
            // there's no key to remember, so the gate will fire again next
            // task. Call that out so the operator knows to redeploy/reload.
            text.push_str(
                "<i>⚠️ Web frontend did not send a browser session id — \
                 approving this will <b>not</b> persist. Redeploy /agent or \
                 hard-reload the page to get 24h-trust working.</i>\n",
            );
        }
    }
    if let Some(ip) = &req.origin_ip {
        text.push_str(&format!(
            "Origin IP: <code>{}</code>\n",
            html_escape(ip)
        ));
    }
    if let Some(ua) = &req.user_agent {
        text.push_str(&format!("Browser: {}\n", html_escape(&short(ua, 80))));
    }
    text.push_str("\nFirst prompt:\n");
    text.push_str("<pre>");
    text.push_str(&html_escape(&short(&req.prompt_excerpt, 400)));
    text.push_str("</pre>\n");
    text.push_str("\nApprove this caller for the next 24h?");
    text
}

/// Render a session id (or its absence) for the Telegram message — used in
/// the request, timeout, and final-edit lines.
fn session_label(id: &Option<String>) -> String {
    match id {
        Some(s) => format!("<code>{}</code>", html_escape(&short(s, 12))),
        None => "<i>(unknown — frontend missing browser_session_id)</i>".to_string(),
    }
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_includes_session_and_prompt() {
        let req = ApprovalRequest {
            browser_session_id: Some("abc12345-67890-very-long".into()),
            prompt_excerpt: "list files".into(),
            origin_ip: Some("203.0.113.5".into()),
            user_agent: Some("Mozilla/5.0".into()),
        };
        let text = compose_request_text(&req);
        assert!(text.contains("abc12345"));
        assert!(text.contains("list files"));
        assert!(text.contains("203.0.113.5"));
    }

    #[test]
    fn compose_warns_when_session_id_missing() {
        // Old `/agent` frontends predate the localStorage integration and
        // don't send a browser_session_id. The notifier message has to
        // surface that so the operator knows approval won't persist.
        let req = ApprovalRequest {
            browser_session_id: None,
            prompt_excerpt: "list files".into(),
            origin_ip: None,
            user_agent: None,
        };
        let text = compose_request_text(&req);
        assert!(text.to_ascii_lowercase().contains("did not send"));
        assert!(text.contains("Redeploy") || text.contains("redeploy") || text.contains("reload"));
    }

    #[test]
    fn compose_escapes_html_in_prompt() {
        // A prompt full of HTML metachars must not break the message. The
        // earlier Markdown mode crashed on stray backticks; HTML mode only
        // needs `<`, `>`, `&` escaped — verify that's actually happening.
        let req = ApprovalRequest {
            browser_session_id: Some("s".into()),
            prompt_excerpt: "</pre><script>alert('x')</script> & co".into(),
            origin_ip: None,
            user_agent: None,
        };
        let text = compose_request_text(&req);
        assert!(!text.contains("<script>"));
        assert!(text.contains("&lt;script&gt;"));
        assert!(text.contains("&amp;"));
    }

    #[test]
    fn short_truncates_with_ellipsis() {
        assert_eq!(short("short", 10), "short");
        assert!(short("0123456789abcdef", 8).ends_with('…'));
    }
}
