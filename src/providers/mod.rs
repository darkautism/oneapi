use async_trait::async_trait;
use crate::types::*;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod openai;
pub mod cli;
pub mod gemini;
pub mod gemini_code_assist;

pub type StreamResponse = Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    fn info(&self) -> &ProviderInfo;
    async fn status(&self) -> ProviderStatus;
    async fn set_status(&self, status: ProviderStatus);
    
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError>;
    async fn chat_stream(&self, req: ChatRequest) -> Result<StreamResponse, ProviderError>;
}

pub struct BaseBackend {
    pub info: ProviderInfo,
    pub status: Arc<RwLock<ProviderStatus>>,
}

impl BaseBackend {
    pub fn new(models: Vec<String>, account_tag: &str, max_context: u32) -> Self {
        Self {
            info: ProviderInfo {
                models,
                account_tag: account_tag.to_string(),
                max_context,
            },
            status: Arc::new(RwLock::new(ProviderStatus::Healthy)),
        }
    }
}
