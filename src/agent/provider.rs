//! LLM backend handle for the kotonia-cli agent.
//!
//! Every shipping backend speaks the OpenAI `/chat/completions` shape. The
//! per-vendor knobs (DeepSeek `:thinking`, max-tokens flavour, custom
//! headers) live in [`crate::providers::ProviderSpec`]; this type just
//! bundles a spec + a model id + per-request extras, and exposes a tiny
//! API the agent loop uses (`complete` + `complete_with_tools`).
//!
//! Adding a new provider is now a `~/.kotonia/providers.json` entry or a
//! new built-in inside [`crate::providers`] — no enum variants here.

use serde_json::{Map, Value};

use crate::ai::openai_compat::{
    create_with_tools as oai_create_with_tools, OpenAiCompatConfig,
};
use crate::ai::{
    AiCallOptions, AiContent, AiMessage, AiResponse, AiRole, AiStopReason, AiUsage,
};
use crate::providers::{ProviderRegistry, ProviderSpec};

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

pub struct Provider {
    spec: ProviderSpec,
    requested_model: String,
    canonical_model: String,
    extra_body: Map<String, Value>,
    client: reqwest::Client,
}

#[derive(Debug)]
pub enum ProviderError {
    Config(String),
    MissingApiKey { provider: String, hint: String },
    Transport(String),
    Api(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Config(m) => write!(f, "provider config error: {m}"),
            ProviderError::MissingApiKey { provider, hint } => {
                write!(f, "provider `{provider}` has no API key. {hint}")
            }
            ProviderError::Transport(m) => write!(f, "transport error: {m}"),
            ProviderError::Api(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ProviderError {}

impl Provider {
    /// Back-compat shim: pick a provider purely from a model id (no
    /// `--provider`). Defers to [`ProviderRegistry::resolve`] with
    /// `provider = None`.
    pub fn for_model(model: &str) -> Result<Self, ProviderError> {
        Self::resolve(None, model)
    }

    /// Resolve an explicit `(provider, model)` pair through the registry.
    /// `provider = None` means "infer from model id".
    pub fn resolve(provider: Option<&str>, model: &str) -> Result<Self, ProviderError> {
        let registry = ProviderRegistry::load().map_err(ProviderError::Config)?;
        let (spec, resolved) = registry
            .resolve(provider, model)
            .map_err(ProviderError::Config)?;
        Ok(Self {
            spec,
            requested_model: model.to_string(),
            canonical_model: resolved.canonical_model,
            extra_body: resolved.extra_body,
            client: reqwest::Client::new(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.requested_model
    }

    pub fn provider_name(&self) -> &str {
        &self.spec.name
    }

    /// Short tag for banners / logs. Matches the previous backend_label()
    /// surface so callers don't have to change.
    pub fn backend_label(&self) -> &str {
        &self.spec.name
    }

    /// True for every shipping backend (all speak OpenAI-compatible tool
    /// calling). Kept as a hook so `--force-delimiter` can gate cleanly if
    /// a future provider lacks the surface.
    pub fn supports_native_tools(&self) -> bool {
        true
    }

    fn require_api_key(&self) -> Result<String, ProviderError> {
        self.spec.api_key.clone().ok_or_else(|| {
            let hint = match self.spec.name.as_str() {
                "kotonia" => {
                    "Run `kotonia-cli login`, or export KOTONIA_API_KEY.".to_string()
                }
                "deepseek" => "Set DEEPSEEK_API_KEY in your environment.".to_string(),
                other => format!(
                    "Set its api_key_env in ~/.kotonia/providers.json, or export the \
                     appropriate variable for provider `{other}`."
                ),
            };
            ProviderError::MissingApiKey {
                provider: self.spec.name.clone(),
                hint,
            }
        })
    }

    fn build_config(&self) -> Result<OpenAiCompatConfig, ProviderError> {
        let api_key = self.require_api_key()?;
        let mut config = OpenAiCompatConfig::new(
            self.spec.base_url.clone(),
            api_key,
            self.spec.max_tokens_param,
        );
        config.model = Some(self.canonical_model.clone());
        config.max_tokens_cap = self.spec.max_tokens_cap;
        config.extra_headers = self.spec.extra_headers.clone();
        config.extra_body = self.extra_body.clone();
        Ok(config)
    }

    /// Text-only completion (delimiter loop path). Wraps `complete_with_tools`
    /// internally because the underlying transport is the same — the agent
    /// just discards any tool_use blocks the model produces.
    pub async fn complete(
        &self,
        messages: Vec<ChatMsg>,
        max_tokens: u32,
    ) -> Result<String, ProviderError> {
        self.complete_with_retry(messages, max_tokens, 3).await
    }

    /// Wrapper around the per-backend call that retries on transient
    /// transport failures (TCP reset, connection refused mid-keepalive,
    /// reqwest "error sending request"). API-level errors (401, 400, rate
    /// limit) bubble straight through.
    pub async fn complete_with_retry(
        &self,
        messages: Vec<ChatMsg>,
        max_tokens: u32,
        max_attempts: u32,
    ) -> Result<String, ProviderError> {
        let mut last_err: Option<ProviderError> = None;
        for attempt in 1..=max_attempts {
            let ai_msgs: Vec<AiMessage> =
                messages.iter().cloned().map(chatmsg_to_ai_message).collect();
            let options = AiCallOptions::new(self.canonical_model.clone(), max_tokens);
            let config = self.build_config()?;
            let result = oai_create_with_tools(&self.client, config, ai_msgs, options)
                .await
                .map(|r| join_text(&r.content))
                .map_err(|e| match e {
                    crate::ai::AiError::Http(m) | crate::ai::AiError::Parse(m) => {
                        ProviderError::Transport(m)
                    }
                    crate::ai::AiError::Api { status, message } => ProviderError::Api(
                        format!("API error{}: {message}", status.map(|s| format!(" ({s})")).unwrap_or_default()),
                    ),
                    crate::ai::AiError::Invalid(m) => ProviderError::Api(m),
                });
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
    pub async fn complete_with_tools(
        &self,
        messages: Vec<AiMessage>,
        mut options: AiCallOptions,
    ) -> Result<AiResponse, ProviderError> {
        options.model = self.canonical_model.clone();
        let config = self.build_config()?;
        oai_create_with_tools(&self.client, config, messages, options)
            .await
            .map_err(|e| match e {
                crate::ai::AiError::Http(m) | crate::ai::AiError::Parse(m) => {
                    ProviderError::Transport(m)
                }
                crate::ai::AiError::Api { status, message } => ProviderError::Api(format!(
                    "API error{}: {message}",
                    status.map(|s| format!(" ({s})")).unwrap_or_default()
                )),
                crate::ai::AiError::Invalid(m) => ProviderError::Api(m),
            })
    }
}

/// Match reqwest connection-level failures wrapped by the AI clients. These
/// almost always succeed on the next attempt (server hiccup, TCP keepalive
/// reset, brief socket unavailability). HTTP 4xx/5xx response errors do NOT
/// match — those are real API errors and should bubble.
pub(crate) fn is_transient_transport_error(err: &ProviderError) -> bool {
    let msg = match err {
        ProviderError::Transport(m) => m,
        ProviderError::Api(_) => return false,
        ProviderError::Config(_) => return false,
        ProviderError::MissingApiKey { .. } => return false,
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

fn chatmsg_to_ai_message(m: ChatMsg) -> AiMessage {
    let role = match m.role {
        ChatRole::System => AiRole::System,
        ChatRole::User => AiRole::User,
        ChatRole::Assistant => AiRole::Assistant,
    };
    AiMessage {
        role,
        content: vec![AiContent::Text { text: m.content }],
    }
}

fn join_text(content: &[AiContent]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            AiContent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// Silence dead-code warnings for items kept around so the agent loop's
// tool-result handling stays self-contained.
#[allow(dead_code)]
fn _keep_imports_used(_u: AiUsage, _s: AiStopReason) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_model_resolves_kotonia_builtin() {
        // Doesn't require KOTONIA_API_KEY at construction — we only validate
        // a key exists when a request actually runs.
        let p = Provider::for_model("kotonia-llm-basic").unwrap();
        assert_eq!(p.provider_name(), "kotonia");
        assert_eq!(p.model_id(), "kotonia-llm-basic");
    }

    #[test]
    fn for_model_resolves_deepseek_builtin() {
        let p = Provider::for_model("deepseek-chat").unwrap();
        assert_eq!(p.provider_name(), "deepseek");
        assert_eq!(p.model_id(), "deepseek-chat");
    }

    #[test]
    fn explicit_provider_overrides_inference() {
        let p = Provider::resolve(Some("deepseek"), "custom-model").unwrap();
        assert_eq!(p.provider_name(), "deepseek");
        assert_eq!(p.model_id(), "custom-model");
    }

    #[test]
    fn supports_native_tools_uniformly() {
        let p = Provider::for_model("kotonia-llm-basic").unwrap();
        assert!(p.supports_native_tools());
    }

    #[test]
    fn transport_error_detector_catches_reqwest_shapes() {
        assert!(is_transient_transport_error(&ProviderError::Transport(
            "HTTP error: error sending request for url (http://127.0.0.1:8899/v1/chat/completions)"
                .into()
        )));
        assert!(is_transient_transport_error(&ProviderError::Transport(
            "connection refused".into()
        )));
    }

    #[test]
    fn transport_error_detector_skips_api_level_errors() {
        assert!(!is_transient_transport_error(&ProviderError::Api(
            "API error (401): unauthorized".into()
        )));
        assert!(!is_transient_transport_error(&ProviderError::MissingApiKey {
            provider: "x".into(),
            hint: "y".into(),
        }));
        assert!(!is_transient_transport_error(&ProviderError::Config(
            "bad json".into()
        )));
    }
}
