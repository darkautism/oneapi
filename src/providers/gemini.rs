use async_trait::async_trait;
use crate::types::*;
use crate::providers::{ModelBackend, BaseBackend, StreamResponse};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Instant, Duration};
use futures::stream;

pub struct GeminiBackend {
    pub base: BaseBackend,
    pub client: Client,
    pub api_key: String,
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: Option<GeminiConfig>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiConfig {
    temperature: Option<f32>,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    total_token_count: u32,
}

impl GeminiBackend {
    pub fn new(models: Vec<String>, account: &str, max_context: u32, api_key: &str) -> Self {
        Self {
            base: BaseBackend::new(models, account, max_context),
            client: Client::new(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl ModelBackend for GeminiBackend {
    fn info(&self) -> &ProviderInfo { &self.base.info }
    async fn status(&self) -> ProviderStatus { self.base.status.read().await.clone() }
    async fn set_status(&self, status: ProviderStatus) { *self.base.status.write().await = status; }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        // Strip common prefixes if they're not supposed to be in the URL
        let model_name = req.model.strip_prefix("google/").unwrap_or(&req.model);
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", model_name, self.api_key);
        
        let contents = req.messages.iter().map(|m| {
            let text = if m.content.is_string() {
                m.content.as_str().unwrap().to_string()
            } else {
                m.content.to_string()
            };
            GeminiContent {
                role: if m.role == "assistant" { "model".into() } else { "user".into() },
                parts: vec![GeminiPart { text }],
            }
        }).collect();

        let gemini_req = GeminiRequest {
            contents,
            generation_config: Some(GeminiConfig {
                temperature: req.temperature,
                max_output_tokens: req.max_tokens.or(Some(1000)),
            }),
        };

        let response = self.client.post(&url)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(e.to_string()))?;

        if response.status().as_u16() == 429 {
            self.set_status(ProviderStatus::RateLimited(Instant::now() + Duration::from_secs(60))).await;
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            return Err(ProviderError::ApiError(response.text().await.unwrap_or_default()));
        }

        let resp_data = response.json::<GeminiResponse>().await
            .map_err(|e| ProviderError::ApiError(e.to_string()))?;

        let candidate = resp_data.candidates.get(0).ok_or(ProviderError::ApiError("No candidates".into()))?;
        let text = candidate.content.parts.get(0).map(|p| p.text.clone()).unwrap_or_default();

        Ok(ChatResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String(text),
                },
                finish_reason: candidate.finish_reason.clone(),
            }],
            usage: resp_data.usage_metadata.map(|u| Usage {
                prompt_tokens: u.prompt_token_count,
                completion_tokens: u.candidates_token_count,
                total_tokens: u.total_token_count,
            }),
        })
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<StreamResponse, ProviderError> {
        let model_name = req.model.strip_prefix("google/").unwrap_or(&req.model);
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", model_name, self.api_key);
        let tag = self.base.info.account_tag.clone();

        let contents = req.messages.iter().map(|m| {
            let text = if m.content.is_string() {
                m.content.as_str().unwrap().to_string()
            } else {
                m.content.to_string()
            };
            GeminiContent {
                role: if m.role == "assistant" { "model".into() } else { "user".into() },
                parts: vec![GeminiPart { text }],
            }
        }).collect::<Vec<_>>();

        let gemini_req = GeminiRequest {
            contents,
            generation_config: Some(GeminiConfig {
                temperature: req.temperature,
                max_output_tokens: req.max_tokens.or(Some(1000)),
            }),
        };

        // Eagerly perform the request to catch status errors for fallback
        let response = self.client.post(&url)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("[{}] {}", tag, e)))?;

        if response.status().as_u16() == 429 {
            self.set_status(ProviderStatus::RateLimited(Instant::now() + Duration::from_secs(60))).await;
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Could not read error body".into());
            return Err(ProviderError::ApiError(format!("[{}] Status {}: {}", tag, status, body)));
        }

        let resp_data = response.json::<GeminiResponse>().await
            .map_err(|e| ProviderError::ApiError(format!("[{}] {}", tag, e)))?;

        let candidate = resp_data.candidates.get(0).ok_or(ProviderError::ApiError(format!("[{}] No candidates", tag)))?;
        let text = candidate.content.parts.get(0).map(|p| p.text.clone()).unwrap_or_default();
        let model_id = req.model.clone();
        
        let fut = async move {
            let chunk = ChatStreamChunk::new(model_id, text, Some("stop".into()));
            Ok(serde_json::to_string(&chunk).unwrap())
        };

        Ok(Box::pin(stream::once(fut)))
    }
}
