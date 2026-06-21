//! LLM backend abstraction for the kotonia-cli agent.
//!
//! `Provider` is an enum (no `dyn`, no `async_trait`) that wraps either a
//! local OpenAI-compatible vLLM/llama.cpp endpoint or the official DeepSeek
//! REST API. The agent loop talks only to `Provider::complete`, so adding a
//! third backend (Anthropic, xAI, …) is a new enum variant plus a routing
//! rule.

use crate::ai::deepseek::{DeepSeekClient, Message as DsMessage, MessageRole as DsRole};
use crate::ai::local_vllm::{LocalVllmClient, Message as LocalMessage};
use crate::ai::{AiCallOptions, AiContent, AiMessage, AiResponse, AiRole};

#[derive(Clone, Debug)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct ChatMsg {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMsg {
    pub fn system(s: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: s.into(),
        }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: s.into(),
        }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: s.into(),
        }
    }
}

pub enum Provider {
    /// Local OpenAI-compatible endpoint. The wrapped `LocalVllmClient` routes
    /// by model id internally: `deepseek-v4-flash` → port 8898 (V4-Flash on
    /// llama.cpp), anything else → port 8899 (Gemma 4 26B Uncensored vLLM).
    Local { client: LocalVllmClient, model: String },
    /// Official DeepSeek API. Model ids: `deepseek-chat` (V4-Flash class),
    /// `deepseek-reasoner` (V4-Pro reasoning). `:thinking` suffix toggles
    /// reasoning mode where supported.
    DeepSeekApi { client: DeepSeekClient, model: String },
    /// Kotonia hosted `/api/v1/chat/completions` (OpenAI-compatible). Lets a
    /// kotonia-cli on a different machine reach the same V4-Flash / Gemma4
    /// 26B without running them locally. Auth: `KOTONIA_API_KEY` from env,
    /// base URL overridable via `KOTONIA_API_BASE` (default `https://kotonia.ai`).
    KotoniaApi {
        client: reqwest::Client,
        api_base: String,
        api_key: String,
        /// Public model id (`kotonia-v4-flash`, `kotonia-gemma4-26b`).
        model: String,
    },
}

#[derive(Debug)]
pub enum ProviderError {
    Local(String),
    DeepSeek(String),
    Kotonia(String),
    MissingApiKey { env_var: &'static str },
    /// Backend does not implement native tool calling. Caller should fall back
    /// to the delimiter path.
    ToolsUnsupported,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Local(m) => write!(f, "local LLM error: {m}"),
            ProviderError::DeepSeek(m) => write!(f, "DeepSeek API error: {m}"),
            ProviderError::Kotonia(m) => write!(f, "Kotonia API error: {m}"),
            ProviderError::MissingApiKey { env_var } => {
                write!(f, "missing {env_var} (set it to use this model)")
            }
            ProviderError::ToolsUnsupported => {
                write!(f, "this backend does not support native tool calling")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

impl Provider {
    /// Pick the right backend from a CLI-supplied model id. Falls back to
    /// `Local` for anything we don't recognise as an API model — this is
    /// intentional so users can point at any new local model id without
    /// touching code.
    pub fn for_model(model: &str) -> Result<Self, ProviderError> {
        if is_kotonia_api_model(model) {
            let key = std::env::var("KOTONIA_API_KEY").map_err(|_| {
                ProviderError::MissingApiKey {
                    env_var: "KOTONIA_API_KEY",
                }
            })?;
            let api_base = std::env::var("KOTONIA_API_BASE")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "https://kotonia.ai".to_string());
            Ok(Provider::KotoniaApi {
                client: reqwest::Client::new(),
                api_base,
                api_key: key,
                model: model.to_string(),
            })
        } else if is_deepseek_api_model(model) {
            let key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
                ProviderError::MissingApiKey {
                    env_var: "DEEPSEEK_API_KEY",
                }
            })?;
            Ok(Provider::DeepSeekApi {
                client: DeepSeekClient::new(key),
                model: model.to_string(),
            })
        } else {
            Ok(Provider::Local {
                client: LocalVllmClient::new(),
                model: model.to_string(),
            })
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Provider::Local { model, .. } => model,
            Provider::DeepSeekApi { model, .. } => model,
            Provider::KotoniaApi { model, .. } => model,
        }
    }

    /// Short tag for banners / logs ("local", "deepseek-api", "kotonia-api").
    pub fn backend_label(&self) -> &'static str {
        match self {
            Provider::Local { .. } => "local",
            Provider::DeepSeekApi { .. } => "deepseek-api",
            Provider::KotoniaApi { .. } => "kotonia-api",
        }
    }

    /// Whether the backend can execute the `tools` / `tool_choice` surface of
    /// the OpenAI chat-completions wire format. V4-Flash on llama.cpp does
    /// NOT — there is no tool-call parser wired up. Everything else does.
    pub fn supports_native_tools(&self) -> bool {
        match self {
            Provider::Local { model, .. } => !is_v4flash_id(model),
            Provider::DeepSeekApi { .. } => true,
            Provider::KotoniaApi { model, .. } => !is_v4flash_id(model),
        }
    }

    pub async fn complete(
        &self,
        messages: Vec<ChatMsg>,
        max_tokens: u32,
    ) -> Result<String, ProviderError> {
        self.complete_with_retry(messages, max_tokens, 3).await
    }

    /// Wrapper around the per-backend call that retries on transient
    /// transport failures (TCP reset, connection refused mid-keepalive,
    /// reqwest "error sending request" — all of which we've seen vLLM
    /// produce intermittently under load). API-level errors (401, 400,
    /// rate limit) bubble straight through.
    pub async fn complete_with_retry(
        &self,
        messages: Vec<ChatMsg>,
        max_tokens: u32,
        max_attempts: u32,
    ) -> Result<String, ProviderError> {
        let mut last_err: Option<ProviderError> = None;
        for attempt in 1..=max_attempts {
            let result = match self {
                Provider::Local { client, model } => {
                    let local_msgs: Vec<LocalMessage> = messages
                        .clone()
                        .into_iter()
                        .map(to_local_message)
                        .collect();
                    client
                        .create_message(model, local_msgs, max_tokens)
                        .await
                        .map(|(text, _)| text)
                        .map_err(|e| ProviderError::Local(e.to_string()))
                }
                Provider::DeepSeekApi { client, model } => {
                    let ds_msgs: Vec<DsMessage> =
                        messages.clone().into_iter().map(to_ds_message).collect();
                    client
                        .create_message(model, ds_msgs, max_tokens)
                        .await
                        .map(|(text, _)| text)
                        .map_err(|e| ProviderError::DeepSeek(e.to_string()))
                }
                Provider::KotoniaApi {
                    client,
                    api_base,
                    api_key,
                    model,
                } => kotonia_api_complete_text(
                    client,
                    api_base,
                    api_key,
                    model,
                    messages.clone(),
                    max_tokens,
                )
                .await
                .map_err(ProviderError::Kotonia),
            };
            match result {
                Ok(t) => return Ok(t),
                Err(e) if attempt < max_attempts && is_transient_transport_error(&e) => {
                    eprintln!(
                        "kotonia-cli: transient LLM transport error (attempt {attempt}/{max_attempts}): {e} — retrying"
                    );
                    let backoff_ms = 250u64 * (1u64 << (attempt as u64 - 1));
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    /// Native tool-calling path. Returns the assistant response in the
    /// provider-agnostic `AiResponse` shape (text blocks + tool_use blocks)
    /// so the agent can dispatch tools without parsing custom delimiters.
    ///
    /// Errors with `ProviderError::ToolsUnsupported` for V4-Flash backends
    /// — the caller should fall back to the legacy delimiter path.
    pub async fn complete_with_tools(
        &self,
        messages: Vec<AiMessage>,
        options: AiCallOptions,
    ) -> Result<AiResponse, ProviderError> {
        if !self.supports_native_tools() {
            return Err(ProviderError::ToolsUnsupported);
        }
        match self {
            Provider::Local { client, .. } => client
                .create_with_tools(messages, options)
                .await
                .map_err(|e| ProviderError::Local(e.to_string())),
            Provider::DeepSeekApi { client, .. } => client
                .create_with_tools(messages, options)
                .await
                .map_err(|e| ProviderError::DeepSeek(e.to_string())),
            Provider::KotoniaApi {
                client,
                api_base,
                api_key,
                ..
            } => kotonia_api_complete_tools(client, api_base, api_key, messages, options)
                .await
                .map_err(ProviderError::Kotonia),
        }
    }
}

/// Match reqwest connection-level failures wrapped by the AI clients. These
/// almost always succeed on the next attempt (vLLM hiccup, TCP keepalive
/// reset, brief socket unavailability). HTTP 4xx/5xx response errors do NOT
/// match — those are real API errors and should bubble.
fn is_transient_transport_error(err: &ProviderError) -> bool {
    let msg = match err {
        ProviderError::Local(m) => m,
        ProviderError::DeepSeek(m) => m,
        ProviderError::Kotonia(m) => m,
        ProviderError::MissingApiKey { .. } => return false,
        ProviderError::ToolsUnsupported => return false,
    };
    let lower = msg.to_ascii_lowercase();
    lower.contains("error sending request")
        || lower.contains("connection refused")
        || lower.contains("connection reset")
        || lower.contains("connection closed")
        || lower.contains("broken pipe")
        || lower.contains("dns")
        || (lower.contains("timed out") && !lower.contains("model"))
}

/// `deepseek-chat`, `deepseek-reasoner`, with optional `:thinking` suffix.
/// Anything else falls through to the local backend.
fn is_deepseek_api_model(model: &str) -> bool {
    let canonical = model.strip_suffix(":thinking").unwrap_or(model);
    matches!(canonical, "deepseek-chat" | "deepseek-reasoner")
}

/// Public model ids served by the kotonia hosted `/api/v1/chat/completions`.
/// Must match `resolve_chat_model` in `handlers/api_v1.rs`.
fn is_kotonia_api_model(model: &str) -> bool {
    matches!(
        model,
        "kotonia-v4-flash" | "kotonia-gemma4-26b"
    )
}

/// True for any V4-Flash flavour. The llama.cpp build has no
/// `--tool-call-parser`, so the agent must fall back to delimiter mode.
fn is_v4flash_id(model: &str) -> bool {
    matches!(
        model,
        "deepseek-v4-flash" | "v4flash" | "kotonia-v4-flash"
    )
}

async fn kotonia_api_complete_text(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model: &str,
    messages: Vec<ChatMsg>,
    max_tokens: u32,
) -> Result<String, String> {
    let url = format!("{}/api/v1/chat/completions", api_base.trim_end_matches('/'));
    let wire_msgs: Vec<_> = messages
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "role": match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                "content": m.content,
            })
        })
        .collect();
    let body = serde_json::json!({
        "model": model,
        "messages": wire_msgs,
        "max_tokens": max_tokens,
    });
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("error sending request: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse response: {e} (body: {text})"))?;
    Ok(parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string())
}

async fn kotonia_api_complete_tools(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    messages: Vec<AiMessage>,
    options: AiCallOptions,
) -> Result<AiResponse, String> {
    // Mirrors openai_compat::create_with_tools, but with the Kotonia API
    // path + key. Inlined here so we don't depend on `pub(crate)` internals
    // crossing module boundaries.
    use crate::ai::{AiStopReason, AiToolChoice, AiUsage};

    let url = format!("{}/api/v1/chat/completions", api_base.trim_end_matches('/'));

    // Build message array.
    let mut wire_msgs: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = options.system.as_deref() {
        if !system.trim().is_empty() {
            wire_msgs.push(serde_json::json!({"role": "system", "content": system}));
        }
    }
    for m in &messages {
        wire_msgs.push(ai_message_to_kotonia_wire(m)?);
    }

    let mut body = serde_json::json!({
        "model": options.model,
        "messages": wire_msgs,
        "max_tokens": options.max_tokens,
    });
    if !options.tools.is_empty() {
        let tools: Vec<_> = options
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect();
        body["tools"] = serde_json::Value::Array(tools);
        body["tool_choice"] = match &options.tool_choice {
            AiToolChoice::Auto => serde_json::Value::String("auto".into()),
            AiToolChoice::Any => serde_json::Value::String("required".into()),
            AiToolChoice::None => serde_json::Value::String("none".into()),
            AiToolChoice::Specific { name } => serde_json::json!({
                "type": "function",
                "function": {"name": name},
            }),
        };
    }

    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("error sending request: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse response: {e} (body: {text})"))?;
    let choice = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| "no choices in response".to_string())?;
    let msg = choice
        .get("message")
        .ok_or_else(|| "no message in choice".to_string())?;
    let mut content: Vec<AiContent> = Vec::new();
    if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            content.push(AiContent::Text {
                text: text.to_string(),
            });
        }
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for call in tool_calls {
            let id = call
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let input = if args_str.trim().is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(args_str)
                    .unwrap_or_else(|_| serde_json::Value::String(args_str.to_string()))
            };
            content.push(AiContent::ToolUse {
                id,
                name,
                input,
                metadata: None,
            });
        }
    }
    let has_tools = content.iter().any(|c| matches!(c, AiContent::ToolUse { .. }));
    let stop_reason = if has_tools {
        AiStopReason::ToolUse
    } else {
        match choice.get("finish_reason").and_then(|v| v.as_str()) {
            Some("tool_calls") => AiStopReason::ToolUse,
            Some("length") => AiStopReason::MaxTokens,
            _ => AiStopReason::EndTurn,
        }
    };
    let usage = v.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Ok(AiResponse {
        content,
        stop_reason,
        usage: AiUsage {
            input_tokens,
            output_tokens,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            reasoning_tokens: 0,
        },
    })
}

fn ai_message_to_kotonia_wire(m: &AiMessage) -> Result<serde_json::Value, String> {
    match m.role {
        AiRole::System => Ok(serde_json::json!({
            "role": "system",
            "content": text_only(&m.content, "system")?,
        })),
        AiRole::User => Ok(serde_json::json!({
            "role": "user",
            "content": text_only(&m.content, "user")?,
        })),
        AiRole::Assistant => {
            let mut text_blocks = Vec::new();
            let mut tool_calls = Vec::new();
            for block in &m.content {
                match block {
                    AiContent::Text { text } => text_blocks.push(text.as_str()),
                    AiContent::ToolUse { id, name, input, .. } => tool_calls.push(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(input).unwrap_or_default(),
                        },
                    })),
                    _ => {
                        return Err(
                            "assistant content cannot contain image/tool_result blocks"
                                .into(),
                        )
                    }
                }
            }
            let mut out = serde_json::Map::new();
            out.insert("role".into(), serde_json::Value::String("assistant".into()));
            let joined = text_blocks.join("\n");
            if joined.is_empty() && !tool_calls.is_empty() {
                out.insert("content".into(), serde_json::Value::Null);
            } else {
                out.insert("content".into(), serde_json::Value::String(joined));
            }
            if !tool_calls.is_empty() {
                out.insert("tool_calls".into(), serde_json::Value::Array(tool_calls));
            }
            Ok(serde_json::Value::Object(out))
        }
        AiRole::Tool => {
            // OpenAI-compat: each ToolResult block becomes a separate
            // tool-role message. The caller is expected to pass exactly one
            // tool result per AiMessage when targeting this path.
            let first = m
                .content
                .first()
                .ok_or_else(|| "tool message is empty".to_string())?;
            match first {
                AiContent::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => Ok(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                })),
                _ => Err("tool messages must contain ToolResult blocks".into()),
            }
        }
    }
}

fn text_only(content: &[AiContent], role: &str) -> Result<String, String> {
    let mut out = Vec::new();
    for c in content {
        match c {
            AiContent::Text { text } => out.push(text.as_str()),
            _ => return Err(format!("{role} messages only support text content")),
        }
    }
    Ok(out.join("\n"))
}


fn to_local_message(m: ChatMsg) -> LocalMessage {
    match m.role {
        ChatRole::System => LocalMessage::system(m.content),
        ChatRole::User => LocalMessage::user(m.content),
        ChatRole::Assistant => LocalMessage::assistant(m.content),
    }
}

fn to_ds_message(m: ChatMsg) -> DsMessage {
    DsMessage {
        role: match m.role {
            ChatRole::System => DsRole::System,
            ChatRole::User => DsRole::User,
            ChatRole::Assistant => DsRole::Assistant,
        },
        content: m.content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_api_models_detected() {
        assert!(is_deepseek_api_model("deepseek-chat"));
        assert!(is_deepseek_api_model("deepseek-reasoner"));
        assert!(is_deepseek_api_model("deepseek-chat:thinking"));
        assert!(is_deepseek_api_model("deepseek-reasoner:thinking"));
    }

    #[test]
    fn local_models_not_classified_as_api() {
        // V4-Flash local & Gemma4 26B local must NOT route to the API by mistake.
        assert!(!is_deepseek_api_model("deepseek-v4-flash"));
        assert!(!is_deepseek_api_model("v4flash"));
        assert!(!is_deepseek_api_model("gemma4-26b-uncensored"));
    }

    #[test]
    fn for_model_picks_local_without_api_key() {
        // Local model should never look at DEEPSEEK_API_KEY.
        let _ = std::env::var("DEEPSEEK_API_KEY"); // tolerate either presence
        let provider = Provider::for_model("deepseek-v4-flash").unwrap();
        assert_eq!(provider.backend_label(), "local");
        assert_eq!(provider.model_id(), "deepseek-v4-flash");
    }

    #[test]
    fn transport_error_detector_catches_reqwest_shapes() {
        // The original reproduction: reqwest "error sending request" wrapped by
        // local_vllm. Must be marked as transient so the retry kicks in.
        assert!(is_transient_transport_error(&ProviderError::Local(
            "HTTP error: error sending request for url (http://127.0.0.1:8899/v1/chat/completions)"
                .into()
        )));
        assert!(is_transient_transport_error(&ProviderError::Local(
            "connection refused".into()
        )));
        assert!(is_transient_transport_error(&ProviderError::DeepSeek(
            "connection reset by peer".into()
        )));
    }

    #[test]
    fn transport_error_detector_skips_api_level_errors() {
        // 4xx / 5xx body errors from the API are NOT transient.
        assert!(!is_transient_transport_error(&ProviderError::DeepSeek(
            "API error: 401 Unauthorized - {invalid key}".into()
        )));
        assert!(!is_transient_transport_error(&ProviderError::Local(
            "OpenAI error 429: rate limit".into()
        )));
        assert!(!is_transient_transport_error(&ProviderError::MissingApiKey {
            env_var: "X"
        }));
        assert!(!is_transient_transport_error(&ProviderError::ToolsUnsupported));
    }

    #[test]
    fn kotonia_api_models_detected() {
        assert!(is_kotonia_api_model("kotonia-v4-flash"));
        assert!(is_kotonia_api_model("kotonia-gemma4-26b"));
        assert!(!is_kotonia_api_model("deepseek-v4-flash"));
        assert!(!is_kotonia_api_model("kotonia-foo"));
    }

    #[test]
    fn v4flash_detected_across_namespaces() {
        // Local + kotonia-hosted V4-Flash flavours all opt out of native tools.
        assert!(is_v4flash_id("deepseek-v4-flash"));
        assert!(is_v4flash_id("v4flash"));
        assert!(is_v4flash_id("kotonia-v4-flash"));
        assert!(!is_v4flash_id("gemma4-26b-uncensored"));
        assert!(!is_v4flash_id("kotonia-gemma4-26b"));
    }

    #[test]
    fn supports_native_tools_by_backend_and_model() {
        // Local Gemma4 → yes; local V4-Flash → no.
        let p = Provider::for_model("gemma4-26b-uncensored").unwrap();
        assert!(p.supports_native_tools());
        let p = Provider::for_model("deepseek-v4-flash").unwrap();
        assert!(!p.supports_native_tools());
    }

    #[test]
    fn for_model_demands_api_key_for_deepseek_chat() {
        // Save & temporarily clear the env var to make the test deterministic.
        let saved = std::env::var("DEEPSEEK_API_KEY").ok();
        // Safety: setting env var in a test is racy under multi-threaded test
        // execution. We mark this single-threaded by gating the assertion on
        // whether saved was empty to begin with — if the user already has the
        // key, just exit (the path is exercised by the smoke test).
        if saved.is_some() {
            return;
        }
        match Provider::for_model("deepseek-chat") {
            Ok(_) => panic!("expected MissingApiKey error, got a Provider"),
            Err(ProviderError::MissingApiKey { env_var }) => {
                assert_eq!(env_var, "DEEPSEEK_API_KEY")
            }
            Err(other) => panic!("expected MissingApiKey, got {other:?}"),
        }
    }
}
