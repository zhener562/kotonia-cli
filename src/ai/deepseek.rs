#![allow(dead_code)]

/// DeepSeek API Client
///
/// REST API implementation for DeepSeek chat completions using the
/// OpenAI-compatible `/chat/completions` surface.
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

const DEEPSEEK_API_BASE: &str = "https://api.deepseek.com";
const THINKING_SUFFIX: &str = ":thinking";

#[derive(Debug)]
pub enum DeepSeekError {
    HttpError(reqwest::Error),
    ApiError(String),
}

impl fmt::Display for DeepSeekError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeepSeekError::HttpError(e) => write!(f, "HTTP error: {}", e),
            DeepSeekError::ApiError(msg) => write!(f, "API error: {}", msg),
        }
    }
}

impl Error for DeepSeekError {}

impl From<reqwest::Error> for DeepSeekError {
    fn from(err: reqwest::Error) -> Self {
        DeepSeekError::HttpError(err)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
}

impl ThinkingConfig {
    fn enabled() -> Self {
        Self {
            thinking_type: "enabled".to_string(),
        }
    }

    fn disabled() -> Self {
        Self {
            thinking_type: "disabled".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekOptions {
    pub thinking_enabled: bool,
    pub reasoning_effort: Option<String>,
}

impl DeepSeekOptions {
    pub fn thinking() -> Self {
        Self {
            thinking_enabled: true,
            reasoning_effort: Some("high".to_string()),
        }
    }

    pub fn non_thinking() -> Self {
        Self {
            thinking_enabled: false,
            reasoning_effort: None,
        }
    }
}

#[derive(Serialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    thinking: ThinkingConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: i32,
    #[serde(default)]
    pub completion_tokens: i32,
    #[serde(default)]
    pub total_tokens: i32,
    #[serde(default)]
    pub prompt_cache_hit_tokens: i32,
    #[serde(default)]
    pub prompt_cache_miss_tokens: i32,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize, Debug)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
}

pub fn is_deepseek_model(model: &str) -> bool {
    model.starts_with("deepseek-")
}

pub fn canonical_model_name(model: &str) -> &str {
    model.strip_suffix(THINKING_SUFFIX).unwrap_or(model)
}

pub fn default_options_for_model(model: &str) -> DeepSeekOptions {
    match canonical_model_name(model) {
        "deepseek-reasoner" => DeepSeekOptions::thinking(),
        "deepseek-chat" => DeepSeekOptions::non_thinking(),
        _ if model.ends_with(THINKING_SUFFIX) => DeepSeekOptions::thinking(),
        _ => DeepSeekOptions::non_thinking(),
    }
}

/// DeepSeek API client
pub struct DeepSeekClient {
    api_key: String,
    client: Client,
}

impl DeepSeekClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
        }
    }

    pub async fn create_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
    ) -> Result<(String, Usage), DeepSeekError> {
        self.create_message_with_options(
            canonical_model_name(model),
            messages,
            max_tokens,
            default_options_for_model(model),
        )
        .await
    }

    pub async fn create_message_with_options(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        options: DeepSeekOptions,
    ) -> Result<(String, Usage), DeepSeekError> {
        let url = format!("{}/chat/completions", DEEPSEEK_API_BASE);

        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            max_tokens,
            thinking: if options.thinking_enabled {
                ThinkingConfig::enabled()
            } else {
                ThinkingConfig::disabled()
            },
            reasoning_effort: if options.thinking_enabled {
                options.reasoning_effort
            } else {
                None
            },
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(DeepSeekError::ApiError(format!(
                "Failed to create message: {} - {}",
                status, error_text
            )));
        }

        let resp: ChatCompletionResponse = response.json().await?;
        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        let usage = resp.usage.unwrap_or_default();

        Ok((content, usage))
    }

    /// Native tool calling through DeepSeek's OpenAI-compatible chat surface.
    pub async fn create_with_tools(
        &self,
        messages: Vec<super::AiMessage>,
        options: super::AiCallOptions,
    ) -> Result<super::AiResponse, super::AiError> {
        let deepseek_options = default_options_for_model(&options.model);
        let mut config = super::openai_compat::OpenAiCompatConfig::new(
            DEEPSEEK_API_BASE,
            self.api_key.clone(),
            super::openai_compat::MaxTokensParam::MaxTokens,
        );
        config.model = Some(canonical_model_name(&options.model).to_string());
        config.extra_body.insert(
            "thinking".into(),
            serde_json::json!({
                "type": if deepseek_options.thinking_enabled {
                    "enabled"
                } else {
                    "disabled"
                }
            }),
        );
        if deepseek_options.thinking_enabled {
            if let Some(reasoning_effort) = deepseek_options.reasoning_effort {
                config.extra_body.insert(
                    "reasoning_effort".into(),
                    serde_json::json!(reasoning_effort),
                );
            }
        }

        super::openai_compat::create_with_tools(&self.client, config, messages, options).await
    }
}
