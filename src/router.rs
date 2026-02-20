use crate::types::*;
use crate::providers::{ModelBackend, StreamResponse};
use crate::quota::QuotaManager;
use crate::registry::IntelligenceRegistry;
use crate::config::GroupChain;
use std::sync::Arc;
use rand::seq::SliceRandom;
use futures::StreamExt;

pub struct Router {
    pub backends: Vec<Arc<dyn ModelBackend>>,
    pub quota_manager: Arc<QuotaManager>,
    pub registry: Arc<IntelligenceRegistry>,
    pub aliases: std::collections::HashMap<String, Vec<String>>,
    pub chains: Vec<GroupChain>,
}

impl Router {
    pub fn new(
        backends: Vec<Arc<dyn ModelBackend>>, 
        quota_manager: Arc<QuotaManager>, 
        registry: Arc<IntelligenceRegistry>,
        aliases: std::collections::HashMap<String, Vec<String>>,
        chains: Vec<GroupChain>,
    ) -> Self {
        Self { backends, quota_manager, registry, aliases, chains }
    }

    fn get_backend(&self, tag: &str) -> Option<Arc<dyn ModelBackend>> {
        self.backends.iter().find(|b| b.info().account_tag == tag).cloned()
    }

    pub async fn chat(&self, req: ChatRequest, global_strategy: FallbackStrategy) -> Result<ChatResponse, ProviderError> {
        let mut strategy = global_strategy;
        let model_id = req.model.clone();

        if model_id == "cost-priority" {
            strategy = FallbackStrategy::Cost;
        } else if model_id == "intelligence-priority" {
            strategy = FallbackStrategy::Performance;
        }

        if strategy == FallbackStrategy::Chain {
            if let Some(chain) = self.chains.iter().find(|c| c.name == model_id) {
                return self.execute_chain(req, chain).await;
            }
        }
        
        // Non-chain mode or specific strategy requested
        let candidates = self.select_candidates_v2(&model_id, strategy);
        self.execute_balanced_group_v2(req, candidates).await
    }

    fn select_candidates_v2(&self, requested_model: &str, strategy: FallbackStrategy) -> Vec<(Arc<dyn ModelBackend>, Option<String>)> {
        let mut candidates = Vec::new();

        // 1. Check if it's an alias
        if let Some(targets) = self.aliases.get(requested_model) {
            for target in targets {
                if let Some(b) = self.get_backend(target) {
                    candidates.push((b, None));
                }
                if let Some(chain) = self.chains.iter().find(|c| c.name == *target) {
                    for group in &chain.groups {
                        for item in group {
                            let (tag, model_override) = if item.contains("::") {
                                let parts: Vec<&str> = item.splitn(2, "::").collect();
                                (parts[0], Some(parts[1].to_string()))
                            } else {
                                (item.as_str(), None)
                            };
                            if let Some(backend) = self.get_backend(tag) {
                                candidates.push((backend, model_override));
                            }
                        }
                    }
                }
            }
            if !candidates.is_empty() { return candidates; }
        }

        // 2. Check for exact model match in backends
        let mut exact_matches = Vec::new();
        for b in &self.backends {
            if b.info().models.contains(&requested_model.to_string()) {
                exact_matches.push((b.clone(), Some(requested_model.to_string())));
            }
        }
        if !exact_matches.is_empty() {
            return exact_matches;
        }

        // 3. Fallback: If it's a "claude-" model, try to use the 'fakegpt' chain if it exists
        if requested_model.starts_with("claude-") {
            if let Some(chain) = self.chains.iter().find(|c| c.name == "fakegpt") {
                for group in &chain.groups {
                    for item in group {
                        let (tag, model_override) = if item.contains("::") {
                            let parts: Vec<&str> = item.splitn(2, "::").collect();
                            (parts[0], Some(parts[1].to_string()))
                        } else {
                            (item.as_str(), None)
                        };
                        if let Some(backend) = self.get_backend(tag) {
                            candidates.push((backend, model_override));
                        }
                    }
                }
                if !candidates.is_empty() { return candidates; }
            }
        }

        // 4. Strategy based global selection
        let mut all_pairs = Vec::new();
        for b in &self.backends {
            for m in &b.info().models {
                all_pairs.push((b.clone(), m.clone()));
            }
        }

        match strategy {
            FallbackStrategy::Cost => {
                all_pairs.sort_by(|(_, m1), (_, m2)| {
                    let s1 = self.registry.get_spec(m1);
                    let s2 = self.registry.get_spec(m2);
                    s1.price_per_1k_output.partial_cmp(&s2.price_per_1k_output).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            FallbackStrategy::Performance => {
                all_pairs.sort_by(|(_, m1), (_, m2)| {
                    let s1 = self.registry.get_spec(m1);
                    let s2 = self.registry.get_spec(m2);
                    s2.intelligence_rank.cmp(&s1.intelligence_rank) // Higher rank first
                });
            }
            _ => {}
        }

        all_pairs.into_iter().map(|(b, m)| (b, Some(m))).collect()
    }

    async fn execute_chain(&self, req: ChatRequest, chain: &GroupChain) -> Result<ChatResponse, ProviderError> {
        for group_items in &chain.groups {
            let mut candidates = Vec::new();
            for item in group_items {
                let (tag, model_override) = if item.contains("::") {
                    let parts: Vec<&str> = item.splitn(2, "::").collect();
                    (parts[0], Some(parts[1].to_string()))
                } else {
                    (item.as_str(), None)
                };

                if let Some(backend) = self.get_backend(tag) {
                    candidates.push((backend, model_override));
                }
            }
            
            if candidates.is_empty() { continue; }

            match self.execute_balanced_group_v2(req.clone(), candidates).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    tracing::warn!("Chain '{}' group failed: {}. Falling back...", chain.name, e);
                    continue;
                }
            }
        }
        Err(ProviderError::Offline(format!("All groups in chain '{}' failed", chain.name)))
    }

    async fn execute_balanced_group_v2(&self, req: ChatRequest, candidates: Vec<(Arc<dyn ModelBackend>, Option<String>)>) -> Result<ChatResponse, ProviderError> {
        let mut healthy = Vec::new();
        for (backend, ovr) in candidates {
            if matches!(backend.status().await, ProviderStatus::Healthy) {
                healthy.push((backend, ovr));
            }
        }
        if healthy.is_empty() { return Err(ProviderError::QuotaExceeded); }

        {
            let mut rng = rand::rng();
            healthy.shuffle(&mut rng);
        }

        let mut last_error: Option<ProviderError> = None;
        for (backend, model_override) in healthy {
            let account = backend.info().account_tag.clone();
            
            // Priority: 1. Manual override from Chain, 2. Exact match, 3. Default models[0]
            let target_model = if let Some(m) = model_override {
                m
            } else if backend.info().models.contains(&req.model) {
                req.model.clone()
            } else {
                backend.info().models[0].clone()
            };

            if !self.quota_manager.check_budget_for(&account, &target_model).await {
                continue; 
            }

            let mut backend_req = req.clone();
            backend_req.model = target_model.clone();

            match backend.chat(backend_req).await {
                Ok(resp) => {
                    if let Some(usage) = &resp.usage {
                        let spec = self.registry.get_spec(&target_model);
                        self.quota_manager.record_usage(&account, &spec, usage).await;
                    }
                    // resp.model = format!("{} ({})", req.model, account);
                    println!(
                        "\x1b[1;34m[Router]\x1b[0m Resolved: \x1b[1;32m{}\x1b[0m @ \x1b[1;33m{}\x1b[0m",
                        resp.model,
                        account
                    );
                    return Ok(resp);
                }
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            }
        }
        if let Some(e) = last_error {
            return Err(e);
        }
        Err(ProviderError::Offline("Balanced group failed".into()))
    }


    pub async fn chat_stream(&self, req: ChatRequest, global_strategy: FallbackStrategy) -> Result<StreamResponse, ProviderError> {
        let mut strategy = global_strategy;
        let model_id = req.model.clone();

        if model_id == "cost-priority" {
            strategy = FallbackStrategy::Cost;
        } else if model_id == "intelligence-priority" {
            strategy = FallbackStrategy::Performance;
        }

        // Group Chain Stream Support
        if strategy == FallbackStrategy::Chain {
            if let Some(chain) = self.chains.iter().find(|c| c.name == model_id) {
                for group_items in &chain.groups {
                    let mut candidates = Vec::new();
                    for item in group_items {
                        let (tag, model_override) = if item.contains("::") {
                            let parts: Vec<&str> = item.splitn(2, "::").collect();
                            (parts[0], Some(parts[1].to_string()))
                        } else {
                            (item.as_str(), None)
                        };
                        if let Some(backend) = self.get_backend(tag) {
                            candidates.push((backend, model_override));
                        }
                    }

                    {
                        let mut rng = rand::rng();
                        candidates.shuffle(&mut rng);
                    }

                    for (backend, model_override) in candidates {
                        let account = backend.info().account_tag.clone();
                        let target_model = model_override.unwrap_or_else(|| {
                            if backend.info().models.contains(&model_id) { model_id.clone() } else { backend.info().models[0].clone() }
                        });

                        if matches!(backend.status().await, ProviderStatus::Healthy) && self.quota_manager.check_budget_for(&account, &target_model).await {
                             let spec = self.registry.get_spec(&target_model);
                             let mut backend_req = req.clone();
                             backend_req.model = target_model;
                             
                             println!("\x1b[1;34m[Router/Stream]\x1b[0m Chain '{}' -> Backend: \x1b[1;33m{}\x1b[0m (Model: {})", chain.name, account, backend_req.model);
                             
                             match backend.chat_stream(backend_req).await {
                                Ok(stream_res) => {
                                    let prompt_tokens = tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(&format!("{:?}", req.messages)).len() as u32;
                                    self.quota_manager.record_usage(&account, &spec, &Usage { prompt_tokens, completion_tokens: 0, total_tokens: prompt_tokens }).await;
                                    
                                    let mapped_stream = stream_res.map(move |res| {
                                        match res {
                                            Ok(content) => Ok(content),
                                            Err(e) => Err(e)
                                        }
                                    });
                                    return Ok(Box::pin(mapped_stream));
                                },
                                Err(e) => {
                                    tracing::warn!("Backend stream failed, trying next: {}", e);
                                    continue;
                                }
                             }
                        }
                    }
                }
                return Err(ProviderError::Offline(format!("Chain '{}' all backends failed to start stream", chain.name)));
            }
        }

        let candidates = self.select_candidates_v2(&model_id, strategy);
        let mut last_error: Option<ProviderError> = None;
        for (backend, model_override) in candidates {
            let account = backend.info().account_tag.clone();
            let target_model = model_override.unwrap_or_else(|| {
                if backend.info().models.contains(&model_id) { model_id.clone() } else { backend.info().models[0].clone() }
            });

            if matches!(backend.status().await, ProviderStatus::Healthy) && self.quota_manager.check_budget_for(&account, &target_model).await {
                println!("\x1b[1;34m[Router/Stream]\x1b[0m Request: \x1b[1;32m{}\x1b[0m -> Backend: \x1b[1;33m{}\x1b[0m (Model: {})", model_id, account, target_model);

                let spec = self.registry.get_spec(&target_model);
                let mut backend_req = req.clone();
                backend_req.model = target_model;
                match backend.chat_stream(backend_req).await {
                    Ok(stream_res) => {
                        let prompt_tokens = tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(&format!("{:?}", req.messages)).len() as u32;
                        self.quota_manager.record_usage(&account, &spec, &Usage { prompt_tokens, completion_tokens: 0, total_tokens: prompt_tokens }).await;
                        
                        let mapped_stream = stream_res.map(move |res| {
                            match res {
                                Ok(content) => Ok(content),
                                Err(e) => Err(e)
                            }
                        });
                        return Ok(Box::pin(mapped_stream));
                    },
                    Err(e) => {
                        tracing::warn!("Stream failover: {}", e);
                        last_error = Some(e);
                        continue;
                    }
                }
            }
        }
        if let Some(e) = last_error {
            return Err(e);
        }
        Err(ProviderError::Offline("No streamable backends".into()))
    }
}
