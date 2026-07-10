//! Provider-agnostic types for native tool calling.
//!
//! Each provider client adapts these into and out of its native wire format
//! (OpenAI tool_calls / tool messages, DeepSeek's thinking-tagged variant).
//! Vendored from the hage backend's `infrastructure::ai` module, trimmed to
//! the subset kotonia-cli uses (text + tool calling — no images, no
//! Anthropic/Gemini adapters).

pub mod openai_compat;

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Role of a message in a conversation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AiRole {
    System,
    User,
    Assistant,
    /// Tool result message. OpenAI-compat APIs require a dedicated `tool`
    /// role for results.
    Tool,
}

/// A single content block within a message.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AiContent {
    Text {
        text: String,
    },
    /// Inline image. Encoded as base64 (no `data:` prefix). On the OpenAI
    /// wire this turns into an `{"type":"image_url","image_url":{"url":
    /// "data:<media_type>;base64,<data>"}}` part inside a user message.
    /// Only carried inside `AiRole::User` messages — multimodal tool
    /// results land here too (the dispatcher injects an extra user
    /// message after a `tool` role text acknowledgement).
    Image {
        /// e.g. "image/png", "image/jpeg", "image/webp"
        media_type: String,
        /// Base64-encoded bytes (no `data:...;base64,` prefix).
        data: String,
    },
    /// Assistant requests a tool invocation.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// Result returned for a previous ToolUse.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl AiContent {
    pub fn text(s: impl Into<String>) -> Self {
        AiContent::Text { text: s.into() }
    }
}

/// One message in a conversation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AiMessage {
    pub role: AiRole,
    pub content: Vec<AiContent>,
}

impl AiMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: AiRole::User,
            content: vec![AiContent::text(text)],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: AiRole::Assistant,
            content: vec![AiContent::text(text)],
        }
    }

    /// Concatenate all text blocks in this message.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                AiContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Tool definition advertised to the model. `input_schema` is JSON Schema —
/// keep within the OpenAPI 3.0 subset (no `$ref`, no `oneOf`/`anyOf`).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AiTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AiToolChoice {
    Auto,
    Any,
    None,
    Specific { name: String },
}

impl Default for AiToolChoice {
    fn default() -> Self {
        AiToolChoice::Auto
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum AiStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AiUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_tokens: u32,
    #[serde(default)]
    pub cache_read_tokens: u32,
    #[serde(default)]
    pub reasoning_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AiResponse {
    pub content: Vec<AiContent>,
    pub stop_reason: AiStopReason,
    pub usage: AiUsage,
}

impl AiResponse {
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                AiContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|c| match c {
                AiContent::ToolUse {
                    id, name, input, ..
                } => Some((id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect()
    }
}

/// Per-call options for `create_with_tools`.
#[derive(Debug, Clone)]
pub struct AiCallOptions {
    pub model: String,
    pub max_tokens: u32,
    pub system: Option<String>,
    pub tools: Vec<AiTool>,
    pub tool_choice: AiToolChoice,
}

impl AiCallOptions {
    pub fn new(model: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            model: model.into(),
            max_tokens,
            system: None,
            tools: Vec::new(),
            tool_choice: AiToolChoice::default(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<AiTool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, choice: AiToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }
}

#[derive(Debug)]
pub enum AiError {
    Http(String),
    Api { status: Option<u16>, message: String },
    Parse(String),
    Invalid(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AiError::Http(msg) => write!(f, "HTTP error: {}", msg),
            AiError::Api { status, message } => match status {
                Some(s) => write!(f, "API error ({}): {}", s, message),
                None => write!(f, "API error: {}", message),
            },
            AiError::Parse(msg) => write!(f, "Parse error: {}", msg),
            AiError::Invalid(msg) => write!(f, "Invalid request: {}", msg),
        }
    }
}

impl Error for AiError {}

impl From<reqwest::Error> for AiError {
    fn from(err: reqwest::Error) -> Self {
        AiError::Http(err.to_string())
    }
}
