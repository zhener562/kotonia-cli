//! Provider registry — built-in shortcuts + `~/.kotonia/providers.json`.
//!
//! Everything kotonia-cli talks to is an OpenAI-compatible
//! `/chat/completions` endpoint. A *provider* bundles the base URL, the
//! bearer token, the `max_tokens` flavour, and any per-model massaging
//! (the only one that ships today is DeepSeek's `:thinking` suffix → an
//! injected `thinking` + `reasoning_effort` body).
//!
//! Resolution:
//!   1. `--provider <name>` if given → use that spec
//!   2. else look up the requested model id in `model_index` to pick a
//!      built-in provider (`kotonia-llm-basic` → `kotonia`,
//!      `deepseek-*` → `deepseek`, `providers.json::models[]` → that one)
//!   3. else error with a hint
//!
//! Auth fallback for the `kotonia` provider: `KOTONIA_API_KEY` env →
//! `~/.kotonia/daemon.json::device_token` → None.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::ai::openai_compat::MaxTokensParam;
use crate::config::{self, ProviderFileEntry};

/// Per-model hook for vendor-specific request shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderHook {
    /// No transformation.
    None,
    /// DeepSeek: strip `:thinking` suffix, set `thinking.type` and
    /// `reasoning_effort` accordingly. `deepseek-reasoner` defaults to
    /// thinking; `deepseek-chat` defaults to non-thinking.
    DeepSeekThinking,
    /// Kotonia-hosted models. `kotonia-llm-basic` (== `kotonia-llm`) is the
    /// local model; it gets the base model's official sampling
    /// (temp 1.0 / top_p 0.95 / top_k 20). Default is no-think (fast); the
    /// `:think` suffix opts into the reasoning pass, which also sets a
    /// `max_tokens` floor of 8192 — the model card requires ≥4096
    /// (preferably 8192+) or the reasoning budget truncates to blank output.
    /// `kotonia-llm-standard` is DeepSeek-backed server-side and takes no
    /// local shaping.
    Kotonia,
}

#[derive(Debug, Clone)]
pub struct ProviderSpec {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub default_model: String,
    pub max_tokens_param: MaxTokensParam,
    pub max_tokens_cap: Option<u32>,
    pub extra_headers: Vec<(String, String)>,
    pub extra_body_base: Map<String, Value>,
    pub hook: ProviderHook,
}

/// Per-request result after applying the provider hook.
#[derive(Debug, Clone)]
pub struct ResolvedRequest {
    /// Model id sent on the wire (e.g. `deepseek-reasoner` without `:thinking`).
    pub canonical_model: String,
    /// Body fields to merge into the request (e.g. DeepSeek thinking knobs).
    pub extra_body: Map<String, Value>,
}

impl ProviderSpec {
    pub fn resolve_request(&self, requested_model: &str) -> ResolvedRequest {
        let mut extra_body = self.extra_body_base.clone();
        let canonical = match self.hook {
            ProviderHook::None => requested_model.to_string(),
            ProviderHook::Kotonia => {
                // Default no-think (fast); `:think` opts into the reasoning
                // pass. `:nothink` is still accepted as an explicit no-think
                // for callers migrating off the old suffix.
                let (canon, thinking) = match requested_model.strip_suffix(":think") {
                    Some(stripped) => (stripped.to_string(), true),
                    None => (
                        requested_model
                            .strip_suffix(":nothink")
                            .unwrap_or(requested_model)
                            .to_string(),
                        false,
                    ),
                };
                // `kotonia-llm-basic` (and its bare alias `kotonia-llm`) is the
                // local ThinkCap model. `kotonia-llm-standard` is DeepSeek-
                // backed server-side and takes no local shaping.
                let is_basic = matches!(canon.as_str(), "kotonia-llm" | "kotonia-llm-basic");
                if is_basic {
                    extra_body.insert("temperature".into(), json!(1.0));
                    extra_body.insert("top_p".into(), json!(0.95));
                    extra_body.insert("top_k".into(), json!(20));
                    if thinking {
                        extra_body.insert("max_tokens".into(), json!(8192));
                        extra_body.insert(
                            "chat_template_kwargs".into(),
                            json!({"enable_thinking": true}),
                        );
                    } else {
                        extra_body.insert(
                            "chat_template_kwargs".into(),
                            json!({"enable_thinking": false}),
                        );
                    }
                }
                canon
            }
            ProviderHook::DeepSeekThinking => {
                let (canon, thinking_enabled) =
                    match requested_model.strip_suffix(":thinking") {
                        Some(stripped) => (stripped.to_string(), true),
                        None => (
                            requested_model.to_string(),
                            requested_model == "deepseek-reasoner",
                        ),
                    };
                extra_body.insert(
                    "thinking".into(),
                    json!({
                        "type": if thinking_enabled { "enabled" } else { "disabled" }
                    }),
                );
                if thinking_enabled {
                    extra_body.insert("reasoning_effort".into(), json!("high"));
                }
                canon
            }
        };
        ResolvedRequest {
            canonical_model: canonical,
            extra_body,
        }
    }
}

pub struct ProviderRegistry {
    specs: HashMap<String, ProviderSpec>,
    /// model id → provider name. Lets `--provider` be omitted for any model
    /// id we recognise.
    model_index: HashMap<String, String>,
    /// First-pass default if neither `--provider` nor `--model` resolves.
    pub default_provider: String,
}

impl ProviderRegistry {
    pub fn load() -> Result<Self, String> {
        let mut specs: HashMap<String, ProviderSpec> = HashMap::new();
        let mut model_index: HashMap<String, String> = HashMap::new();

        // ── Built-in: kotonia (hosted /api/v1) ─────────────────────────────
        let kotonia = kotonia_builtin();
        for m in ["kotonia-llm", "kotonia-llm-basic", "kotonia-llm-standard"] {
            model_index.insert(m.into(), kotonia.name.clone());
        }
        specs.insert(kotonia.name.clone(), kotonia);

        // ── Built-in: deepseek (api.deepseek.com) ──────────────────────────
        let deepseek = deepseek_builtin();
        for m in [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-chat:thinking",
            "deepseek-reasoner:thinking",
        ] {
            model_index.insert(m.into(), deepseek.name.clone());
        }
        specs.insert(deepseek.name.clone(), deepseek);

        let mut default_provider = "kotonia".to_string();

        // ── User-supplied providers.json (optional) ────────────────────────
        if let Some(file) = config::load_providers()? {
            if let Some(dp) = file.default_provider {
                default_provider = dp;
            }
            for (name, entry) in file.providers {
                let spec = file_entry_to_spec(&name, &entry)?;
                for m in &entry.models {
                    model_index.insert(m.clone(), name.clone());
                }
                if let Some(default_model) = entry.default_model.as_deref() {
                    model_index.entry(default_model.into()).or_insert_with(|| name.clone());
                }
                specs.insert(name, spec);
            }
        }

        Ok(Self {
            specs,
            model_index,
            default_provider,
        })
    }

    /// Resolve `(provider, model)` → spec + per-request body shaping.
    ///
    /// If `provider` is None, the registry infers it from the model id; if
    /// no inference matches it falls back to `default_provider`.
    pub fn resolve(
        &self,
        provider: Option<&str>,
        model: &str,
    ) -> Result<(ProviderSpec, ResolvedRequest), String> {
        let name = match provider {
            Some(p) => p.to_string(),
            None => self
                .model_index
                .get(model)
                .cloned()
                .unwrap_or_else(|| self.default_provider.clone()),
        };
        let spec = self.specs.get(&name).cloned().ok_or_else(|| {
            let mut known: Vec<&String> = self.specs.keys().collect();
            known.sort();
            format!(
                "unknown provider `{name}`. Known: {}. Add new ones at ~/.kotonia/providers.json.",
                known
                    .into_iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let resolved = spec.resolve_request(model);
        Ok((spec, resolved))
    }

    pub fn known_providers(&self) -> Vec<String> {
        let mut out: Vec<String> = self.specs.keys().cloned().collect();
        out.sort();
        out
    }
}

fn kotonia_builtin() -> ProviderSpec {
    let base = std::env::var("KOTONIA_API_BASE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://kotonia.ai".to_string());
    let base_url = format!("{}/api/v1", base.trim_end_matches('/'));
    let api_key = std::env::var("KOTONIA_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(daemon_device_token);
    ProviderSpec {
        name: "kotonia".into(),
        base_url,
        api_key,
        default_model: "kotonia-llm-basic".into(),
        max_tokens_param: MaxTokensParam::MaxTokens,
        max_tokens_cap: None,
        extra_headers: Vec::new(),
        extra_body_base: Map::new(),
        hook: ProviderHook::Kotonia,
    }
}

fn deepseek_builtin() -> ProviderSpec {
    ProviderSpec {
        name: "deepseek".into(),
        base_url: "https://api.deepseek.com".into(),
        api_key: std::env::var("DEEPSEEK_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        default_model: "deepseek-chat".into(),
        max_tokens_param: MaxTokensParam::MaxTokens,
        max_tokens_cap: None,
        extra_headers: Vec::new(),
        extra_body_base: Map::new(),
        hook: ProviderHook::DeepSeekThinking,
    }
}

fn daemon_device_token() -> Option<String> {
    config::load().map(|cfg| cfg.device_token)
}

fn file_entry_to_spec(name: &str, entry: &ProviderFileEntry) -> Result<ProviderSpec, String> {
    if entry.base_url.trim().is_empty() {
        return Err(format!(
            "providers.json provider `{name}`: `base_url` is required"
        ));
    }
    let max_tokens_param = match entry
        .max_tokens_param
        .as_deref()
        .unwrap_or("max_tokens")
    {
        "max_tokens" => MaxTokensParam::MaxTokens,
        "max_completion_tokens" => MaxTokensParam::MaxCompletionTokens,
        "max_output_tokens" => MaxTokensParam::MaxOutputTokens,
        other => {
            return Err(format!(
                "providers.json provider `{name}`: unknown max_tokens_param `{other}` \
                 (expected max_tokens / max_completion_tokens / max_output_tokens)"
            ));
        }
    };
    let api_key = entry
        .api_key_env
        .as_deref()
        .and_then(|env| std::env::var(env).ok().filter(|s| !s.is_empty()))
        .or_else(|| entry.api_key.clone().filter(|s| !s.is_empty()));
    let extra_headers = entry
        .extra_headers
        .iter()
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect();
    Ok(ProviderSpec {
        name: name.to_string(),
        base_url: entry.base_url.clone(),
        api_key,
        default_model: entry
            .default_model
            .clone()
            .unwrap_or_else(|| "<unset>".into()),
        max_tokens_param,
        max_tokens_cap: entry.max_tokens_cap,
        extra_headers,
        extra_body_base: entry.extra_body.clone(),
        hook: ProviderHook::None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_thinking_suffix_strips_and_injects() {
        let spec = deepseek_builtin();
        let r = spec.resolve_request("deepseek-chat:thinking");
        assert_eq!(r.canonical_model, "deepseek-chat");
        assert_eq!(r.extra_body["thinking"]["type"], "enabled");
        assert_eq!(r.extra_body["reasoning_effort"], "high");
    }

    #[test]
    fn deepseek_reasoner_defaults_to_thinking() {
        let spec = deepseek_builtin();
        let r = spec.resolve_request("deepseek-reasoner");
        assert_eq!(r.canonical_model, "deepseek-reasoner");
        assert_eq!(r.extra_body["thinking"]["type"], "enabled");
    }

    #[test]
    fn deepseek_chat_disables_thinking_by_default() {
        let spec = deepseek_builtin();
        let r = spec.resolve_request("deepseek-chat");
        assert_eq!(r.canonical_model, "deepseek-chat");
        assert_eq!(r.extra_body["thinking"]["type"], "disabled");
        assert!(r.extra_body.get("reasoning_effort").is_none());
    }

    #[test]
    fn kotonia_standard_passes_through_unshaped() {
        // `kotonia-llm-standard` is DeepSeek-backed server-side; the client
        // sends it as-is with no local shaping.
        let spec = kotonia_builtin();
        let r = spec.resolve_request("kotonia-llm-standard");
        assert_eq!(r.canonical_model, "kotonia-llm-standard");
        assert!(r.extra_body.is_empty());
    }

    #[test]
    fn basic_think_suffix_gets_sampling_and_token_floor() {
        let spec = kotonia_builtin();
        let r = spec.resolve_request("kotonia-llm-basic:think");
        assert_eq!(r.canonical_model, "kotonia-llm-basic");
        assert_eq!(r.extra_body["temperature"], json!(1.0));
        assert_eq!(r.extra_body["top_p"], json!(0.95));
        assert_eq!(r.extra_body["top_k"], json!(20));
        // Thinking needs >=4096 (preferably 8192+) headroom or the reasoning
        // budget truncates to blank output.
        assert_eq!(r.extra_body["max_tokens"], json!(8192));
        assert_eq!(
            r.extra_body["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
    }

    #[test]
    fn basic_defaults_to_no_think() {
        let spec = kotonia_builtin();
        // Bare basic and the `kotonia-llm` alias default to no-think, and the
        // explicit `:nothink` suffix is still honoured — none set a token floor.
        for id in ["kotonia-llm-basic", "kotonia-llm", "kotonia-llm-basic:nothink"] {
            let r = spec.resolve_request(id);
            assert_eq!(
                r.extra_body["chat_template_kwargs"],
                json!({"enable_thinking": false}),
                "{id}"
            );
            assert!(!r.extra_body.contains_key("max_tokens"), "{id}");
        }
    }

    #[test]
    fn kotonia_llm_models_route_to_kotonia_provider() {
        let reg = ProviderRegistry::load().unwrap();
        for m in ["kotonia-llm", "kotonia-llm-basic", "kotonia-llm-standard"] {
            let (spec, _) = reg.resolve(None, m).unwrap();
            assert_eq!(spec.name, "kotonia");
        }
    }

    #[test]
    fn registry_infers_provider_from_model() {
        let reg = ProviderRegistry::load().unwrap();
        let (spec, _) = reg.resolve(None, "kotonia-llm-basic").unwrap();
        assert_eq!(spec.name, "kotonia");
        let (spec, _) = reg.resolve(None, "deepseek-chat").unwrap();
        assert_eq!(spec.name, "deepseek");
    }

    #[test]
    fn registry_explicit_provider_wins() {
        let reg = ProviderRegistry::load().unwrap();
        let (spec, _) =
            reg.resolve(Some("deepseek"), "some-custom-model").unwrap();
        assert_eq!(spec.name, "deepseek");
    }

    #[test]
    fn registry_rejects_unknown_provider() {
        let reg = ProviderRegistry::load().unwrap();
        let err = reg.resolve(Some("nope"), "x").unwrap_err();
        assert!(err.contains("unknown provider"));
    }
}
