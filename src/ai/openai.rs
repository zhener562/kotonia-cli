#![allow(dead_code)]

/// OpenAI API Client for ReAct Agent
///
/// Simple REST API implementation for OpenAI chat completions
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

fn uses_max_completion_tokens(model: &str) -> bool {
    model.starts_with("gpt-5")
        || model.starts_with("gpt-4.1")
        || model.starts_with("o4")
        || model.starts_with("o3")
}

#[derive(Debug)]
pub enum OpenAIError {
    HttpError(reqwest::Error),
    ApiError(String),
}

impl fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpenAIError::HttpError(e) => write!(f, "HTTP error: {}", e),
            OpenAIError::ApiError(msg) => write!(f, "API error: {}", msg),
        }
    }
}

impl Error for OpenAIError {}

impl From<reqwest::Error> for OpenAIError {
    fn from(err: reqwest::Error) -> Self {
        OpenAIError::HttpError(err)
    }
}

/// Message role in a conversation
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Image URL with optional detail parameter
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
}

/// Content part for multimodal messages
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// Message content - can be simple text or array of parts (multimodal)
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Message in a conversation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
}

impl Message {
    /// Create a user message with text content
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a user message with multimodal content (text and/or images)
    pub fn user_with_content(parts: Vec<ContentPart>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Parts(parts),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Get text content if this is a text message
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(parts) => {
                // Get first text part
                parts.iter().find_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            }
        }
    }
}

impl ContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// Create an image content part from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an image content part from URL with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail.into()),
            },
        }
    }

    /// Create an image content part from base64 data
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        let data_url = format!("data:{};base64,{}", media_type.into(), data.into());
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: data_url,
                detail: None,
            },
        }
    }
}

/// Request to create a chat completion
#[derive(Serialize, Debug)]
struct CreateChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
}

/// Token usage information in response
#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(default)]
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i32,
}

/// Response from creating a chat completion
#[derive(Deserialize, Debug)]
pub struct CreateChatCompletionResponse {
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: ResponseMessage,
}

#[derive(Deserialize, Debug)]
pub struct ResponseMessage {
    pub content: String,
}

/// OpenAI API client
pub struct OpenAIClient {
    api_key: String,
    client: Client,
}

impl OpenAIClient {
    /// Create a new OpenAI client with the given API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
        }
    }

    /// Create a chat completion (returns content and usage)
    pub async fn create_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
    ) -> Result<(String, Usage), OpenAIError> {
        let url = format!("{}/chat/completions", OPENAI_API_BASE);

        // Determine which token parameter to use based on model
        // Newer models (GPT-5, GPT-4.1, o4, o3) use max_completion_tokens
        let use_max_completion_tokens = uses_max_completion_tokens(model);

        let request_body = CreateChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature: 1.0,
            max_tokens: if use_max_completion_tokens {
                None
            } else {
                Some(max_tokens)
            },
            max_completion_tokens: if use_max_completion_tokens {
                Some(max_tokens)
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
            return Err(OpenAIError::ApiError(format!(
                "Failed to create message: {} - {}",
                status, error_text
            )));
        }

        let completion_response: CreateChatCompletionResponse = response.json().await?;

        let content = completion_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .unwrap_or_default();

        Ok((content, completion_response.usage))
    }

    /// Native tool calling via the OpenAI Chat Completions `tools` surface.
    pub async fn create_with_tools(
        &self,
        messages: Vec<super::AiMessage>,
        options: super::AiCallOptions,
    ) -> Result<super::AiResponse, super::AiError> {
        let max_tokens_param = if uses_max_completion_tokens(&options.model) {
            super::openai_compat::MaxTokensParam::MaxCompletionTokens
        } else {
            super::openai_compat::MaxTokensParam::MaxTokens
        };

        let mut config = super::openai_compat::OpenAiCompatConfig::new(
            OPENAI_API_BASE,
            self.api_key.clone(),
            max_tokens_param,
        );
        config.temperature = Some(1.0);

        super::openai_compat::create_with_tools(&self.client, config, messages, options).await
    }
}
