use tokio::time::{sleep, Duration};
use crate::types::*;
use crate::router::Router;
use std::sync::Arc;

pub struct HealthManager {
    pub router_state: Arc<tokio::sync::RwLock<Arc<Router>>>,
    pub max_interval: Duration,
}

impl HealthManager {
    pub fn new(router_state: Arc<tokio::sync::RwLock<Arc<Router>>>, max_interval_secs: u64) -> Self {
        Self { 
            router_state, 
            max_interval: Duration::from_secs(max_interval_secs) 
        }
    }

    pub async fn run(&self) {
        let mut backoffs: std::collections::HashMap<String, Duration> = std::collections::HashMap::new();
        let mut next_probes: std::collections::HashMap<String, std::time::Instant> = std::collections::HashMap::new();
        let min_interval = Duration::from_secs(60);

        loop {
            // Get the latest backends from the shared router state
            let backends = {
                let r = self.router_state.read().await;
                r.backends.clone()
            };

            for backend in backends {
                let account = &backend.info().account_tag;
                let status = backend.status().await;
                
                let is_failed = match status {
                    ProviderStatus::RateLimited(reset_at) => std::time::Instant::now() >= reset_at,
                    ProviderStatus::QuotaExceeded | ProviderStatus::Offline => true,
                    ProviderStatus::Healthy => {
                        backoffs.remove(account);
                        next_probes.remove(account);
                        false
                    },
                };

                if is_failed {
                    let now = std::time::Instant::now();
                    let next_probe = next_probes.get(account).cloned().unwrap_or(now);

                    if now >= next_probe {
                            let model_display = backend.info().models.get(0).cloned().unwrap_or_else(|| "unknown".to_string());
                            tracing::info!("Active probing backend: {} ({})", model_display, account);

                            let ping_req = ChatRequest {
                                model: model_display.clone(),
                            messages: vec![ChatMessage {
                                role: "user".to_string(),
                                content: serde_json::Value::String("This is a health-check ping. Please respond with 'pong'.".to_string()),
                            }],
                            temperature: Some(0.0),
                            max_tokens: Some(5),
                            stream: Some(false),
                            tools: None,
                        };

                        match backend.chat(ping_req).await {
                            Ok(_) => {
                                tracing::info!("Backend {} recovered!", model_display);
                                backend.set_status(ProviderStatus::Healthy).await;
                                backoffs.remove(account);
                                next_probes.remove(account);
                            }
                            Err(_) => {
                                // Exponential backoff: double the interval
                                let current_backoff = backoffs.get(account).cloned().unwrap_or(min_interval);
                                let next_backoff = std::cmp::min(current_backoff * 2, self.max_interval);
                                
                                backoffs.insert(account.clone(), next_backoff);
                                next_probes.insert(account.clone(), now + next_backoff);
                                
                                tracing::warn!("Backend {} probe failed. Next probe in {}s", model_display, next_backoff.as_secs());
                            }
                        }
                    }
                }
            }
            sleep(Duration::from_secs(10)).await;
        }
    }
}
