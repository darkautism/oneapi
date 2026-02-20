use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::*;

pub struct QuotaManager {
    // Key: account_tag
    pub spends: Arc<RwLock<HashMap<String, f64>>>,
    pub limits: HashMap<String, f64>,
}

impl QuotaManager {
    pub fn new(limits: HashMap<String, f64>) -> Self {
        let path = crate::paths::get_path("usage.json");
        let spends = match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => HashMap::new(),
        };
        Self {
            spends: Arc::new(RwLock::new(spends)),
            limits,
        }
    }

    // Helper: create a QuotaManager with given spends without touching filesystem
    pub fn with_spends_and_limits(spends: HashMap<String, f64>, limits: HashMap<String, f64>) -> Self {
        Self { spends: Arc::new(RwLock::new(spends)), limits }
    }

    pub async fn save(&self) -> anyhow::Result<()> {
        let spends = self.spends.read().await;
        let content = serde_json::to_string_pretty(&*spends)?;
        crate::paths::write_atomic("usage.json", &content)?;
        Ok(())
    }

    pub async fn check_budget(&self, account_tag: &str) -> bool {
        // Backwards compatible check: if a per-account+model limit exists it should be
        // checked by callers using `check_budget(account, model)`. Keep this signature
        // for compatibility internally but prefer using the new two-arg version.
        if let Some(limit) = self.limits.get(account_tag) {
            let spends = self.spends.read().await;
            let current = spends.get(account_tag).unwrap_or(&0.0);
            return current < limit;
        }
        true
    }

    pub async fn record_usage(&self, account_tag: &str, spec: &ModelSpec, usage: &Usage) {
        let cost = (usage.prompt_tokens as f64 / 1000.0 * spec.price_per_1k_input) +
                   (usage.completion_tokens as f64 / 1000.0 * spec.price_per_1k_output);

        let key = format!("{}::{}", account_tag, spec.model_id);
        let mut spends = self.spends.write().await;
        let entry = spends.entry(key.clone()).or_insert(0.0);
        *entry += cost;

        tracing::info!("Account {} model {} cost: ${:.6}, total spend: ${:.6}", account_tag, spec.model_id, cost, *entry);

        // Immediate Flush & Atomic Write
        if let Ok(content) = serde_json::to_string_pretty(&*spends) {
            let _ = crate::paths::write_atomic("usage.json", &content);
        }
    }

    // New API: check budget for a specific account+model. Falls back to per-account limit
    // if a per-model limit isn't set (for backward compatibility with configs).
    pub async fn check_budget_for(&self, account_tag: &str, model_id: &str) -> bool {
        let key = format!("{}::{}", account_tag, model_id);
        if let Some(limit) = self.limits.get(&key) {
            let spends = self.spends.read().await;
            let current = spends.get(&key).unwrap_or(&0.0);
            return current < limit;
        }

        // Fallback: check per-account limit if present
        if let Some(limit) = self.limits.get(account_tag) {
            let spends = self.spends.read().await;
            let current = spends.get(account_tag).unwrap_or(&0.0);
            return current < limit;
        }

        true
    }

    // Migrate legacy spends keyed by account -> distribute evenly into account::model
    // using the provided backends to find models for each account. Writes migrated
    // spends back to disk atomically.
    pub async fn migrate_spends(&self, backends: &Vec<std::sync::Arc<dyn crate::providers::ModelBackend>>) -> anyhow::Result<()> {
        let mut spends_map = self.spends.write().await;
        // collect legacy keys
        let legacy_keys: Vec<String> = spends_map.keys().filter(|k| !k.contains("::")).cloned().collect();
        if legacy_keys.is_empty() { return Ok(()); }

        for acct in legacy_keys {
            if let Some(amount) = spends_map.remove(&acct) {
                // find models for this account
                let mut models: Vec<String> = Vec::new();
                for b in backends.iter() {
                    if b.info().account_tag == acct {
                        for m in b.info().models.iter() { models.push(m.clone()); }
                    }
                }

                if models.is_empty() {
                    // no models found; keep as-is under a fallback key
                    let key = format!("{}::{}", acct, "unknown");
                    let entry = spends_map.entry(key).or_insert(0.0);
                    *entry += amount;
                } else {
                    let per = amount / (models.len() as f64);
                    for m in models.iter() {
                        let key = format!("{}::{}", acct, m);
                        let entry = spends_map.entry(key).or_insert(0.0);
                        *entry += per;
                    }
                }
            }
        }

        // flush to disk
        if let Ok(content) = serde_json::to_string_pretty(&*spends_map) {
            let _ = crate::paths::write_atomic("usage.json", &content);
        }
        Ok(())
    }
}
