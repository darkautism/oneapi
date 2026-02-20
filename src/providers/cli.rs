use async_trait::async_trait;
use crate::types::*;
use crate::providers::{ModelBackend, BaseBackend, StreamResponse};
use std::process::Stdio;
use tokio::process::Command;
use tiktoken_rs::cl100k_base;

pub struct CliBackend {
    pub base: BaseBackend,
    pub command_template: String,
    pub json_path: Option<String>,
}

impl CliBackend {
    pub fn new(models: Vec<String>, account: &str, max_context: u32, template: &str, json_path: Option<String>) -> Self {
        Self {
            base: BaseBackend::new(models, account, max_context),
            command_template: template.to_string(),
            json_path,
        }
    }
}

#[async_trait]
impl ModelBackend for CliBackend {
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
        let messages_str = req.messages.iter()
            .map(|m| {
                let content_str = if m.content.is_string() {
                    m.content.as_str().unwrap().to_string()
                } else {
                    m.content.to_string()
                };
                format!("{}: {}", m.role, content_str)
            })
            .collect::<Vec<_>>()
            .join("\n");
        
        let cmd_str = self.command_template
            .replace("{model}", &req.model)
            .replace("{messages}", &messages_str);

        let output = Command::new("sh")
            .arg("-c")
            .arg(&cmd_str)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ProviderError::Offline(e.to_string()))?;

        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            if err_msg.contains("quota") || err_msg.contains("exhausted") {
                self.set_status(ProviderStatus::QuotaExceeded).await;
                return Err(ProviderError::QuotaExceeded);
            }
            return Err(ProviderError::ApiError(err_msg.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        
        // Strip ANSI escape codes (colors, etc.) to ensure clean output
        let clean_bytes = strip_ansi_escapes::strip(&stdout);
        let clean_stdout = String::from_utf8_lossy(&clean_bytes).to_string();

        let content = if let Some(path) = &self.json_path {
            let v: serde_json::Value = serde_json::from_str(&clean_stdout)
                .map_err(|e| ProviderError::ApiError(format!("JSON parse error: {}", e)))?;
            
            v.pointer(path)
                .and_then(|val| val.as_str())
                .unwrap_or(&clean_stdout)
                .to_string()
        } else {
            clean_stdout.trim().to_string()
        };

        // Simulate token counting for CLI
        let bpe = cl100k_base().unwrap();
        let prompt_tokens = bpe.encode_with_special_tokens(&messages_str).len() as u32;
        let completion_tokens = bpe.encode_with_special_tokens(&content).len() as u32;

        Ok(ChatResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        })
    }

    async fn chat_stream(&self, _req: ChatRequest) -> Result<StreamResponse, ProviderError> {
        Err(ProviderError::ApiError("CLI does not support streaming".to_string()))
    }
}
