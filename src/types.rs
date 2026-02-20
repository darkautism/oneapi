use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<OpenAITool>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAITool {
    pub r#type: String, // "function"
    pub function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatStreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatMessageDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

impl ChatStreamChunk {
    pub fn new(model: String, content: String, finish_reason: Option<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion.chunk".into(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: Some(content),
                },
                finish_reason,
            }],
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProviderStatus {
    Healthy,
    RateLimited(Instant),
    QuotaExceeded,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub models: Vec<String>, // Changed from name: String
    pub account_tag: String,
    pub max_context: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("Rate limit reached")]
    RateLimit,
    #[error("Quota exceeded")]
    QuotaExceeded,
    #[error("Provider is offline: {0}")]
    Offline(String),
    #[error("API Error: {0}")]
    ApiError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub intelligence_rank: u8, // 1-10
    pub price_per_1k_input: f64,
    pub price_per_1k_output: f64,
}

use clap::ValueEnum;

#[derive(Debug, Clone, Serialize, Deserialize, ValueEnum, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum FallbackStrategy {
    Performance,
    Cost,
    Chain,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<serde_json::Value>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<AnthropicTool>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value, // Can be string or array of content blocks
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessageResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicContentBlock {
    pub r#type: String,
    pub text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
