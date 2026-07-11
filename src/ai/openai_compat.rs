use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{
    AiCallOptions, AiContent, AiError, AiMessage, AiResponse, AiRole, AiStopReason, AiToolChoice,
    AiUsage,
};

#[derive(Debug, Clone, Copy)]
pub enum MaxTokensParam {
    MaxTokens,
    MaxCompletionTokens,
    MaxOutputTokens,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: Option<String>,
    pub max_tokens_param: MaxTokensParam,
    pub max_tokens_cap: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub extra_body: Map<String, Value>,
    pub extra_headers: Vec<(String, String)>,
}

impl OpenAiCompatConfig {
    pub fn new(
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        max_tokens_param: MaxTokensParam,
    ) -> Self {
        Self {
            api_base: api_base.into(),
            api_key: api_key.into(),
            model: None,
            max_tokens_param,
            max_tokens_cap: None,
            temperature: None,
            top_p: None,
            extra_body: Map::new(),
            extra_headers: Vec::new(),
        }
    }
}

pub async fn create_with_tools(
    client: &Client,
    config: OpenAiCompatConfig,
    messages: Vec<AiMessage>,
    options: AiCallOptions,
) -> Result<AiResponse, AiError> {
    let url = format!("{}/chat/completions", config.api_base.trim_end_matches('/'));
    let body = build_request_body(config.clone(), messages, options)?;

    let mut req = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json");
    for (name, value) in config.extra_headers {
        req = req.header(name, value);
    }

    let response = req.json(&body).send().await.map_err(AiError::from)?;
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(AiError::Api {
            status: Some(status.as_u16()),
            message: error_text,
        });
    }

    let response_text = response.text().await.map_err(AiError::from)?;
    let parsed: CompatResponse = serde_json::from_str(&response_text)
        .map_err(|e| AiError::Parse(format!("{} - Body: {}", e, response_text)))?;
    wire_response_to_ai(parsed)
}

fn build_request_body(
    config: OpenAiCompatConfig,
    messages: Vec<AiMessage>,
    options: AiCallOptions,
) -> Result<Value, AiError> {
    let model = config.model.unwrap_or(options.model);
    let max_tokens = options
        .max_tokens
        .min(config.max_tokens_cap.unwrap_or(options.max_tokens));

    let mut body = Map::new();
    body.insert("model".into(), Value::String(model));
    body.insert(
        "messages".into(),
        Value::Array(ai_messages_to_wire(&messages, options.system.as_deref())?),
    );
    match config.max_tokens_param {
        MaxTokensParam::MaxTokens => {
            body.insert("max_tokens".into(), json!(max_tokens));
        }
        MaxTokensParam::MaxCompletionTokens => {
            body.insert("max_completion_tokens".into(), json!(max_tokens));
        }
        MaxTokensParam::MaxOutputTokens => {
            body.insert("max_output_tokens".into(), json!(max_tokens));
        }
    }
    if let Some(temperature) = config.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = config.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if !options.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(options.tools.iter().map(tool_to_wire).collect()),
        );
        body.insert(
            "tool_choice".into(),
            tool_choice_to_wire(&options.tool_choice),
        );
    }
    for (key, value) in config.extra_body {
        body.insert(key, value);
    }
    Ok(Value::Object(body))
}

fn ai_messages_to_wire(
    messages: &[AiMessage],
    system: Option<&str>,
) -> Result<Vec<Value>, AiError> {
    let mut out = Vec::new();
    if let Some(system) = system {
        if !system.trim().is_empty() {
            out.push(json!({
                "role": "system",
                "content": system,
            }));
        }
    }

    for message in messages {
        match &message.role {
            AiRole::System => out.push(json!({
                "role": "system",
                "content": content_text_only(&message.content, "system")?,
            })),
            AiRole::User => out.push(user_message_to_wire(&message.content)?),
            AiRole::Assistant => out.push(assistant_message_to_wire(&message.content)?),
            AiRole::Tool => {
                for block in &message.content {
                    match block {
                        AiContent::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => out.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content,
                        })),
                        _ => {
                            return Err(AiError::Invalid(
                                "OpenAI-compatible tool messages must contain only ToolResult blocks"
                                    .into(),
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

fn content_text_only(content: &[AiContent], role: &str) -> Result<String, AiError> {
    let mut text = Vec::new();
    for block in content {
        match block {
            AiContent::Text { text: t } => text.push(t.as_str()),
            _ => {
                return Err(AiError::Invalid(format!(
                    "{} messages only support text content in OpenAI-compatible adapter",
                    role
                )));
            }
        }
    }
    Ok(text.join("\n"))
}

fn user_message_to_wire(content: &[AiContent]) -> Result<Value, AiError> {
    let has_non_text = content.iter().any(|c| !matches!(c, AiContent::Text { .. }));
    if !has_non_text {
        return Ok(json!({
            "role": "user",
            "content": content_text_only(content, "user")?,
        }));
    }

    // Multimodal: emit the parts-array form. Text and image blocks both
    // belong in user messages; other block kinds are invalid here.
    let mut parts = Vec::new();
    for block in content {
        match block {
            AiContent::Text { text } => parts.push(json!({
                "type": "text",
                "text": text,
            })),
            AiContent::Image { media_type, data } => parts.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{media_type};base64,{data}")
                }
            })),
            _ => {
                return Err(AiError::Invalid(
                    "user messages only support text and image content in OpenAI-compatible adapter"
                        .into(),
                ));
            }
        }
    }

    Ok(json!({
        "role": "user",
        "content": parts,
    }))
}

fn assistant_message_to_wire(content: &[AiContent]) -> Result<Value, AiError> {
    let mut text_blocks = Vec::new();
    let mut tool_calls = Vec::new();

    for block in content {
        match block {
            AiContent::Text { text } => text_blocks.push(text.as_str()),
            AiContent::ToolUse {
                id, name, input, ..
            } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": serde_json::to_string(input).map_err(|e| {
                        AiError::Invalid(format!("failed to serialize tool arguments: {}", e))
                    })?,
                },
            })),
            _ => {
                return Err(AiError::Invalid(
                    "assistant messages cannot contain image or tool result blocks".into(),
                ));
            }
        }
    }

    let text = text_blocks.join("\n");
    let mut message = Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    if text.is_empty() && !tool_calls.is_empty() {
        message.insert("content".into(), Value::Null);
    } else {
        message.insert("content".into(), Value::String(text));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }

    Ok(Value::Object(message))
}

fn tool_to_wire(tool: &super::AiTool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    })
}

fn tool_choice_to_wire(choice: &AiToolChoice) -> Value {
    match choice {
        AiToolChoice::Auto => Value::String("auto".into()),
        AiToolChoice::Any => Value::String("required".into()),
        AiToolChoice::None => Value::String("none".into()),
        AiToolChoice::Specific { name } => json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

#[derive(Deserialize, Debug)]
struct CompatResponse {
    choices: Vec<CompatChoice>,
    #[serde(default)]
    usage: Option<CompatUsage>,
}

#[derive(Deserialize, Debug)]
struct CompatChoice {
    message: CompatResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct CompatResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<CompatToolCall>,
}

#[derive(Deserialize, Debug)]
struct CompatToolCall {
    id: String,
    #[serde(default)]
    function: CompatFunctionCall,
}

#[derive(Deserialize, Debug, Default)]
struct CompatFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize, Debug, Default)]
struct CompatUsage {
    #[serde(default)]
    prompt_tokens: i32,
    #[serde(default)]
    completion_tokens: i32,
    #[serde(default)]
    prompt_tokens_details: Option<CompatPromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompatCompletionTokensDetails>,
    #[serde(default)]
    prompt_cache_hit_tokens: i32,
}

#[derive(Deserialize, Debug, Default)]
struct CompatPromptTokensDetails {
    #[serde(default)]
    cached_tokens: i32,
}

#[derive(Deserialize, Debug, Default)]
struct CompatCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: i32,
}

fn wire_response_to_ai(resp: CompatResponse) -> Result<AiResponse, AiError> {
    let Some(choice) = resp.choices.into_iter().next() else {
        return Err(AiError::Parse(
            "OpenAI-compatible response contained no choices".into(),
        ));
    };

    let mut content = Vec::new();
    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            content.push(AiContent::Text { text });
        }
    }

    for call in choice.message.tool_calls {
        let input = if call.function.arguments.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| Value::String(call.function.arguments.clone()))
        };
        content.push(AiContent::ToolUse {
            id: call.id,
            name: call.function.name,
            input,
            metadata: None,
        });
    }

    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, AiContent::ToolUse { .. }));
    let stop_reason = if has_tool_calls {
        AiStopReason::ToolUse
    } else {
        match choice.finish_reason.as_deref() {
            Some("stop") | None => AiStopReason::EndTurn,
            Some("tool_calls") => AiStopReason::ToolUse,
            Some("length") => AiStopReason::MaxTokens,
            Some(other) => AiStopReason::Other(other.to_string()),
        }
    };

    let usage = resp.usage.unwrap_or_default();
    Ok(AiResponse {
        content,
        stop_reason,
        usage: AiUsage {
            input_tokens: usage.prompt_tokens.max(0) as u32,
            output_tokens: usage.completion_tokens.max(0) as u32,
            cache_creation_tokens: 0,
            cache_read_tokens: (usage
                .prompt_tokens_details
                .map(|details| details.cached_tokens)
                .unwrap_or_default()
                .max(usage.prompt_cache_hit_tokens)
                .max(0)) as u32,
            reasoning_tokens: usage
                .completion_tokens_details
                .map(|details| details.reasoning_tokens)
                .unwrap_or_default()
                .max(0) as u32,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_assistant_tool_calls_and_results() {
        let messages = vec![
            AiMessage {
                role: AiRole::Assistant,
                content: vec![
                    AiContent::Text {
                        text: "I'll run Python.".into(),
                    },
                    AiContent::ToolUse {
                        id: "call_1".into(),
                        name: "python_execute".into(),
                        input: json!({"code": "print(1)"}),
                        metadata: None,
                    },
                ],
            },
            AiMessage {
                role: AiRole::Tool,
                content: vec![AiContent::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "1\n".into(),
                    is_error: false,
                }],
            },
        ];

        let wire = ai_messages_to_wire(&messages, Some("system prompt")).unwrap();
        assert_eq!(
            wire[0],
            json!({"role": "system", "content": "system prompt"})
        );
        assert_eq!(wire[1]["role"], "assistant");
        assert_eq!(
            wire[1]["tool_calls"][0]["function"]["name"],
            "python_execute"
        );
        assert_eq!(
            wire[2],
            json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "1\n",
            })
        );
    }

    #[test]
    fn parses_tool_call_response() {
        let response = CompatResponse {
            choices: vec![CompatChoice {
                finish_reason: Some("tool_calls".into()),
                message: CompatResponseMessage {
                    content: Some("Need computation.".into()),
                    tool_calls: vec![CompatToolCall {
                        id: "call_1".into(),
                        function: CompatFunctionCall {
                            name: "python_execute".into(),
                            arguments: r#"{"code":"print(2)"}"#.into(),
                        },
                    }],
                },
            }],
            usage: Some(CompatUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                prompt_tokens_details: Some(CompatPromptTokensDetails { cached_tokens: 3 }),
                completion_tokens_details: Some(CompatCompletionTokensDetails {
                    reasoning_tokens: 2,
                }),
                prompt_cache_hit_tokens: 0,
            }),
        };

        let ai = wire_response_to_ai(response).unwrap();
        assert!(matches!(ai.stop_reason, AiStopReason::ToolUse));
        assert_eq!(ai.text(), "Need computation.");
        let calls = ai.tool_uses();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "python_execute");
        assert_eq!(ai.usage.cache_read_tokens, 3);
        assert_eq!(ai.usage.reasoning_tokens, 2);
    }

    #[test]
    fn parses_null_usage_details() {
        let response: CompatResponse = serde_json::from_value(json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"ok\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": null,
                "completion_tokens_details": null
            }
        }))
        .unwrap();

        let ai = wire_response_to_ai(response).unwrap();
        assert_eq!(ai.tool_uses().len(), 1);
        assert_eq!(ai.usage.cache_read_tokens, 0);
        assert_eq!(ai.usage.reasoning_tokens, 0);
    }

    #[test]
    fn serializes_specific_tool_choice() {
        let value = tool_choice_to_wire(&AiToolChoice::Specific {
            name: "python_execute".into(),
        });
        assert_eq!(
            value,
            json!({
                "type": "function",
                "function": {"name": "python_execute"},
            })
        );
    }

    #[test]
    fn serializes_user_image_parts() {
        let value = user_message_to_wire(&[
            AiContent::Text {
                text: "look".into(),
            },
            AiContent::Image {
                media_type: "image/png".into(),
                data: "abc123".into(),
            },
        ])
        .unwrap();

        assert_eq!(
            value,
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"}
                    }
                ]
            })
        );
    }
}
