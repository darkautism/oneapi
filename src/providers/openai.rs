use async_trait::async_trait;
use crate::types::*;
use crate::providers::{ModelBackend, BaseBackend, StreamResponse};
use reqwest::Client;
use std::time::{Instant, Duration};
use futures::StreamExt;

pub struct OpenAIBackend {
    pub base: BaseBackend,
    pub client: Client,
    pub api_key: String,
    pub base_url: String,
}

impl OpenAIBackend {
    pub fn new(models: Vec<String>, account: &str, max_context: u32, api_key: &str, base_url: &str) -> Self {
        Self {
            base: BaseBackend::new(models, account, max_context),
            client: Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }
}

#[async_trait]
impl ModelBackend for OpenAIBackend {
    fn info(&self) -> &ProviderInfo {
        &self.base.info
    }

    async fn status(&self) -> ProviderStatus {
        self.base.status.read().await.clone()
    }

    async fn set_status(&self, status: ProviderStatus) {
        *self.base.status.write().await = status;
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let tag = self.base.info.account_tag.clone();
        
        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("[{}] {}", tag, e)))?;

        if response.status().as_u16() == 429 || response.status().as_u16() == 423 {
            let reset_after = response.headers()
                .get("x-ratelimit-reset-requests")
                .or_else(|| response.headers().get("retry-after"))
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);

            self.set_status(ProviderStatus::RateLimited(Instant::now() + Duration::from_secs(reset_after))).await;
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Could not read error body".into());
            if body.contains("insufficient_quota") {
                self.set_status(ProviderStatus::QuotaExceeded).await;
                return Err(ProviderError::QuotaExceeded);
            }
            return Err(ProviderError::ApiError(format!("[{}] Status {}: {}", tag, status, body)));
        }

        let resp_data = response.json::<ChatResponse>().await
            .map_err(|e| ProviderError::ApiError(format!("[{}] {}", tag, e)))?;

        Ok(resp_data)
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<StreamResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let tag = self.base.info.account_tag.clone();
        let original_model = req.model.clone();
        let mut stream_req = req.clone();
        stream_req.stream = Some(true);

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&stream_req)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("[{}] {}", tag, e)))?;

        if response.status().as_u16() == 429 || response.status().as_u16() == 423 {
            let reset_after = response.headers()
                .get("x-ratelimit-reset-requests")
                .or_else(|| response.headers().get("retry-after"))
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);

            self.set_status(ProviderStatus::RateLimited(Instant::now() + Duration::from_secs(reset_after))).await;
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Could not read error body".into());
            return Err(ProviderError::ApiError(format!("[{}] Status {}: {}", tag, status, body)));
        }

        let stream = response.bytes_stream().map(move |item| {
            match item {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let mut results = Vec::new();
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        
                        if line.starts_with("data: ") {
                            let content = line.trim_start_matches("data: ").trim();
                            if content != "[DONE]" && !content.is_empty() {
                                // Try to fix the model name in the chunk so client doesn't get confused
                                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(content) {
                                    if let Some(m) = val.get_mut("model") {
                                        *m = serde_json::Value::String(original_model.clone());
                                    }
                                    if let Ok(serialized) = serde_json::to_string(&val) {
                                        results.push(Ok(serialized));
                                    } else {
                                        results.push(Ok(content.to_string()));
                                    }
                                } else {
                                    results.push(Ok(content.to_string()));
                                }
                            }
                        } else if line.starts_with("{") && line.contains("\"error\"") {
                            results.push(Err(ProviderError::ApiError(format!("[{}] Mid-stream Error: {}", tag, line))));
                        }
                    }
                    results
                }
                Err(e) => vec![Err(ProviderError::ApiError(format!("[{}] Network/Stream Error: {}", tag, e)))]
            }
        }).flat_map(|list| {
            futures::stream::iter(list)
        });

        Ok(Box::pin(stream))
    }
}
