use std::collections::HashMap;
use crate::types::*;

pub struct IntelligenceRegistry {
    pub models: HashMap<String, ModelSpec>,
}

impl IntelligenceRegistry {
    pub fn new() -> Self {
        let path = crate::paths::get_path("models.json");
        let mut models: HashMap<String, ModelSpec> = match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => HashMap::new(),
        };

        if models.is_empty() {
            // Provide some sensible defaults on first run
            models.insert("gpt-4o".to_string(), ModelSpec {
                model_id: "gpt-4o".to_string(),
                intelligence_rank: 9,
                price_per_1k_input: 0.005,
                price_per_1k_output: 0.015,
            });
            models.insert("gemini-1.5-pro".to_string(), ModelSpec {
                model_id: "gemini-1.5-pro".to_string(),
                intelligence_rank: 9,
                price_per_1k_input: 0.0035,
                price_per_1k_output: 0.0105,
            });
            // Try to save them immediately
            let _ = serde_json::to_string_pretty(&models).map(|content| {
                let _ = crate::paths::write_atomic("models.json", &content);
            });
        }

        Self { models }
    }

    pub fn get_spec(&self, model_id: &str) -> ModelSpec {
        self.models.get(model_id).cloned().unwrap_or(ModelSpec {
            model_id: model_id.to_string(),
            intelligence_rank: 5, // Unknown models get a neutral rank
            price_per_1k_input: 0.01,
            price_per_1k_output: 0.03,
        })
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(&self.models)?;
        crate::paths::write_atomic("models.json", &content)?;
        Ok(())
    }
}
