#![allow(dead_code)]

/// Local vLLM client (OpenAI-compatible REST surface) with xAI fallback.
///
/// Hits a self-hosted vLLM instance — typically `llm_server/start_gemma4_26b_uncensored.sh`
/// on port 8899. Uses the same wire types as `super::openai` so callers in
/// `voice_chat.rs` / `handlers::ai` can mirror the OpenAI dispatch path.
///
/// Multimodal: the underlying Gemma 4 26B Uncensored MAX model supports vision,
/// and the vLLM server is launched with `--limit-mm-per-prompt` to accept
/// images. Callers wanting to send images build multimodal messages with
/// [`Message::user_with_content`] + [`ContentPart::image_base64`].
///
/// Cost tracking: local inference is free, so callers should skip the
/// `pricing::*::calculate_cost` path or record zero usage.
///
/// ## xAI fallback
///
/// If `XAI_API_KEY` is set in the environment at client construction, requests
/// that fail against the local server (connection refused, timeout, HTTP 5xx)
/// transparently retry against xAI's Grok 4.1 fast non-reasoning. This lets us
/// stop the local vLLM (e.g. during GPU-heavy experiments) without taking down
/// chat for end users. Fallback model name is overridable via
/// `LOCAL_VLLM_FALLBACK_MODEL`. Successful fallbacks log at WARN.
use std::time::Duration;

use reqwest::Client;
use serde::Serialize;

pub use super::openai::{ContentPart, Message, Usage};
use super::openai::{CreateChatCompletionResponse, OpenAIError};

// The vLLM launcher binds IPv4 only. `localhost` may resolve to `::1` first,
// which makes reqwest fail before reaching the healthy server.
const LOCAL_VLLM_API_BASE: &str = "http://127.0.0.1:8899/v1";
// DeepSeek-V4-Flash on llama.cpp (cchuter/feat/v4-port-cuda fork) — CPU MoE
// offload, ~5-12 tok/s, 256K ctx. Latency-tolerant only (TRPG / story / ReAct).
const LOCAL_V4FLASH_API_BASE: &str = "http://127.0.0.1:8898/v1";
const XAI_API_BASE: &str = "https://api.x.ai/v1";
const DEFAULT_FALLBACK_MODEL: &str = "grok-4-1-fast-non-reasoning";

/// Resolve the local OpenAI-compatible endpoint for a given served model id.
/// Lets us route slow latency-tolerant models (V4-Flash) and the fast voice
/// model (Gemma 4 26B Uncensored) to different inference processes without
/// touching the dozen call sites that pass `model` straight through.
fn base_url_for_model(model: &str) -> &'static str {
    if is_v4flash_model(model) {
        LOCAL_V4FLASH_API_BASE
    } else {
        LOCAL_VLLM_API_BASE
    }
}

fn is_v4flash_model(model: &str) -> bool {
    matches!(model, "deepseek-v4-flash" | "v4flash")
}

/// Short connect timeout — when the local server is down, `connect()` fails
/// almost immediately (connection refused). When it is up, normal generation
/// usually starts ~22ms after connect, so this never trips. Set short so
/// fallback fires fast instead of blocking voice-chat callers on a TCP backoff.
const LOCAL_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);

/// vLLM's uvicorn front-end closes idle keep-alive sockets after ~5 seconds
/// (its `--timeout-keep-alive` default). reqwest's default `pool_idle_timeout`
/// is 90 seconds, so a paused REPL session reliably hands the next request a
/// half-closed socket → "error sending request". Capping our pool to 2 s
/// means reqwest evicts the connection before uvicorn does, eliminating the
/// predictable first-call failure after any idle gap.
const LOCAL_POOL_IDLE: Duration = Duration::from_secs(2);

#[derive(Serialize)]
struct LocalChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
}

/// Sampling knobs passed per-request to the local vLLM.
///
/// `default_with_max` mirrors the original `create_message(..., max_tokens)`
/// behavior — temperature 1.0, no penalty — so legacy call sites keep their
/// shape. `voice()` is the curated config for character-chat replies, where
/// onomatopoeia runaway ("あああああ…" extended to thousands of tokens) was
/// blowing through max_tokens and overloading Ditto with mouth-shape frames.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub max_tokens: u32,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl SamplingConfig {
    pub fn default_with_max(max_tokens: u32) -> Self {
        Self {
            temperature: 1.0,
            max_tokens,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }

    /// Voice / character chat. frequency_penalty 0.5 + presence_penalty 0.3
    /// curb the long-onomatopoeia / repeated-phrase cascades that bypass
    /// `max_tokens=4096` and pile up Ditto frames. 768 caps the worst case
    /// to ~10s of TTS audio so a single bad reply can't dominate the GPU.
    pub fn voice() -> Self {
        Self {
            temperature: 0.85,
            max_tokens: 768,
            top_p: Some(0.92),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
        }
    }
}

#[derive(Serialize)]
struct XaiChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    max_output_tokens: u32,
}

#[derive(Clone)]
struct XaiFallback {
    api_key: String,
    text_model: String,
}

pub struct LocalVllmClient {
    client: Client,
    fallback: Option<XaiFallback>,
}

impl Default for LocalVllmClient {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalVllmClient {
    pub fn new() -> Self {
        let client = Client::builder()
            .connect_timeout(LOCAL_CONNECT_TIMEOUT)
            .pool_idle_timeout(LOCAL_POOL_IDLE)
            .build()
            .unwrap_or_else(|_| Client::new());
        let fallback = std::env::var("XAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|api_key| XaiFallback {
                api_key,
                text_model: std::env::var("LOCAL_VLLM_FALLBACK_MODEL")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| DEFAULT_FALLBACK_MODEL.to_string()),
            });
        Self { client, fallback }
    }

    /// Mirror of `OpenAIClient::create_message`. Returns `(content, usage)`.
    /// On local failure, retries against xAI if fallback is configured.
    pub async fn create_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
    ) -> Result<(String, Usage), OpenAIError> {
        self.create_message_with_sampling(
            model,
            messages,
            SamplingConfig::default_with_max(max_tokens),
        )
        .await
    }

    /// Like `create_message` but takes a full sampling config (temperature,
    /// frequency_penalty, top_p, …). xAI fallback only forwards `max_tokens`
    /// because the fallback struct hasn't been extended — penalties are a
    /// local-only knob today.
    pub async fn create_message_with_sampling(
        &self,
        model: &str,
        messages: Vec<Message>,
        sampling: SamplingConfig,
    ) -> Result<(String, Usage), OpenAIError> {
        match self.call_local_chat(model, &messages, &sampling).await {
            Ok(result) => Ok(result),
            Err(local_err) => {
                let Some(fb) = &self.fallback else {
                    return Err(local_err);
                };
                eprintln!(
                    "[local_vllm] unavailable ({}), falling back to xAI ({})",
                    local_err, fb.text_model
                );
                self.call_xai_chat(fb, &messages, sampling.max_tokens)
                    .await
                    .map_err(|xai_err| {
                        OpenAIError::ApiError(format!(
                            "local vLLM failed and xAI fallback also failed: \
                             local={local_err} xai={xai_err}",
                        ))
                    })
            }
        }
    }

    /// Native tool calling via OpenAI-compatible `tools` surface.
    /// vLLM was launched with `--enable-auto-tool-choice --tool-call-parser gemma4`
    /// so the standard OpenAI tool wire format works. On failure, retries
    /// against xAI Grok with the fallback model substituted in.
    pub async fn create_with_tools(
        &self,
        messages: Vec<super::AiMessage>,
        options: super::AiCallOptions,
    ) -> Result<super::AiResponse, super::AiError> {
        let mut local_config = super::openai_compat::OpenAiCompatConfig::new(
            base_url_for_model(&options.model),
            "EMPTY",
            super::openai_compat::MaxTokensParam::MaxTokens,
        );
        // V4-Flash supports 256K ctx; Gemma 4 26B is 32K. Cap conservatively
        // so a stray giant max_tokens request doesn't OOM the smaller model.
        local_config.max_tokens_cap = Some(if is_v4flash_model(&options.model) {
            8192
        } else {
            32768
        });

        // Clone inputs so we can retry on fallback. Both AiMessage and
        // AiCallOptions are Clone.
        let messages_for_local = messages.clone();
        let options_for_local = options.clone();

        match super::openai_compat::create_with_tools(
            &self.client,
            local_config,
            messages_for_local,
            options_for_local,
        )
        .await
        {
            Ok(resp) => Ok(resp),
            Err(local_err) => {
                let Some(fb) = &self.fallback else {
                    return Err(local_err);
                };
                eprintln!(
                    "[local_vllm tools] unavailable ({}), falling back to xAI ({})",
                    local_err, fb.text_model
                );

                let xai_config = super::openai_compat::OpenAiCompatConfig::new(
                    XAI_API_BASE,
                    fb.api_key.clone(),
                    super::openai_compat::MaxTokensParam::MaxOutputTokens,
                );
                let mut fb_options = options;
                fb_options.model = fb.text_model.clone();

                super::openai_compat::create_with_tools(
                    &self.client,
                    xai_config,
                    messages,
                    fb_options,
                )
                .await
            }
        }
    }

    async fn call_local_chat(
        &self,
        model: &str,
        messages: &[Message],
        sampling: &SamplingConfig,
    ) -> Result<(String, Usage), OpenAIError> {
        let url = format!("{}/chat/completions", base_url_for_model(model));
        let body = LocalChatRequest {
            model,
            messages,
            temperature: sampling.temperature,
            max_tokens: sampling.max_tokens,
            top_p: sampling.top_p,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", "Bearer EMPTY")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(OpenAIError::ApiError(format!(
                "local vLLM error: {} - {}",
                status, error_text
            )));
        }

        let parsed: CreateChatCompletionResponse = response.json().await?;
        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        Ok((content, parsed.usage))
    }

    async fn call_xai_chat(
        &self,
        fb: &XaiFallback,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<(String, Usage), OpenAIError> {
        let url = format!("{}/chat/completions", XAI_API_BASE);
        let body = XaiChatRequest {
            model: &fb.text_model,
            messages,
            temperature: 1.0,
            max_output_tokens: max_tokens,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", fb.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(OpenAIError::ApiError(format!(
                "xAI fallback error: {} - {}",
                status, error_text
            )));
        }

        let parsed: CreateChatCompletionResponse = response.json().await?;
        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        Ok((content, parsed.usage))
    }
}

/// Model-name based check. Returns true if the model is served by the local
/// vLLM endpoint and should be dispatched to [`LocalVllmClient`].
///
/// Listed explicitly so adding new local models is intentional (avoids
/// accidentally routing remote APIs through localhost).
pub fn is_local_model(model: &str) -> bool {
    matches!(
        model,
        "gemma4-26b-uncensored" | "deepseek-v4-flash" | "v4flash"
    )
}

/// Whether the given local model accepts image inputs. Gemma 4 26B Uncensored
/// MAX is multimodal, but DeepSeek-V4-Flash served via llama.cpp is text-only
/// (no vision adapter built into this GGUF). Add new local models here as
/// needed.
pub fn local_supports_vision(model: &str) -> bool {
    matches!(model, "gemma4-26b-uncensored")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai::Message;
    use crate::ai::{AiCallOptions, AiMessage, AiTool};

    /// End-to-end smoke against the running vLLM on :8899. Requires
    /// `start_gemma4_26b_uncensored.sh` to be up. Run manually:
    ///   cd backend && cargo test smoke_local_vllm -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn smoke_local_vllm() {
        let client = LocalVllmClient::new();
        let msgs = vec![
            Message::system("Reply with one word.".to_string()),
            Message::user("Say 'pong'.".to_string()),
        ];
        let (text, usage) = client
            .create_message("gemma4-26b-uncensored", msgs, 30)
            .await
            .expect("vLLM call failed — is :8899 up?");
        println!("local vLLM reply: {:?}", text);
        println!(
            "usage: prompt={} completion={}",
            usage.prompt_tokens, usage.completion_tokens
        );
        assert!(!text.is_empty(), "got empty reply");
    }

    /// End-to-end native tool-call smoke against the running vLLM on :8899.
    /// Run manually:
    ///   cd backend && cargo test smoke_local_vllm_tools -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn smoke_local_vllm_tools() {
        let client = LocalVllmClient::new();
        let tool = AiTool {
            name: "echo".to_string(),
            description: "Echo text".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }),
        };
        let response = client
            .create_with_tools(
                vec![AiMessage::user_text(
                    "Call the echo tool with text local-react-ok. Do not answer directly.",
                )],
                AiCallOptions::new("gemma4-26b-uncensored", 128).with_tools(vec![tool]),
            )
            .await
            .expect("vLLM native tool call failed - is :8899 up?");

        let calls = response.tool_uses();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "echo");
        assert_eq!(calls[0].2["text"], "local-react-ok");
    }

    /// Force fallback by aiming the local client at a closed port, then verify
    /// xAI returns a non-empty reply. Requires XAI_API_KEY in env. Manual:
    ///   XAI_API_KEY=... cargo test smoke_xai_fallback -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn smoke_xai_fallback() {
        // Simulate local-down by temporarily pointing the URL at a closed port.
        // We test via the live path: call_xai_chat directly with a real key.
        let client = LocalVllmClient::new();
        let Some(fb) = client.fallback.as_ref() else {
            panic!("XAI_API_KEY must be set for this test");
        };
        let msgs = vec![
            Message::system("Reply with exactly one word.".to_string()),
            Message::user("Say 'pong'.".to_string()),
        ];
        let (text, _usage) = client
            .call_xai_chat(fb, &msgs, 30)
            .await
            .expect("xAI fallback failed");
        println!("xAI fallback reply: {:?}", text);
        assert!(!text.is_empty(), "got empty reply from xAI");
    }

    #[test]
    fn is_local_model_matches() {
        assert!(is_local_model("gemma4-26b-uncensored"));
        assert!(!is_local_model("gemini-3.1-flash-lite"));
        assert!(!is_local_model("gpt-5"));
        assert!(!is_local_model("gemma-4-26b-a4b-it"));
    }
}
