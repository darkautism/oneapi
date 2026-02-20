use serde::{Deserialize, Serialize};
use std::fs;
use crate::types::*;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub client: ClientConfig,
    pub admin: AdminConfig,
    pub fallback_strategy: FallbackStrategy,
    pub backends: Vec<BackendConfig>,
    #[serde(default)]
    pub aliases: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub chains: Vec<GroupChain>, // New: Group Chain Fallback
    #[serde(default = "default_max_probe")]
    pub max_probe_interval_secs: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GroupChain {
    pub name: String,
    pub groups: Vec<Vec<String>>, // List of groups, each group is a list of account_tags
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BackendConfig {
    Openai {
        models: Vec<String>, // Changed from name: String
        account_tag: String,
        api_key: String,
        base_url: String,
        max_context: u32,
        budget_limit: Option<f64>,
    },
    Cli {
        models: Vec<String>, // Changed from name: String
        account_tag: String,
        command: String,
        max_context: u32,
        budget_limit: Option<f64>,
        json_path: Option<String>,
    },
    Gemini {
        models: Vec<String>,
        account_tag: String,
        api_key: String,
        max_context: u32,
        budget_limit: Option<f64>,
    },
}

impl BackendConfig {
    pub fn account_tag(&self) -> &str {
        match self {
            BackendConfig::Openai { account_tag, .. }
            | BackendConfig::Cli { account_tag, .. }
            | BackendConfig::Gemini { account_tag, .. } => account_tag,
        }
    }
}

pub fn duplicate_account_tags(backends: &[BackendConfig]) -> Vec<String> {
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for backend in backends {
        *counts.entry(backend.account_tag().to_string()).or_insert(0) += 1;
    }
    let mut duplicates: Vec<String> = counts
        .into_iter()
        .filter_map(|(tag, count)| if count > 1 { Some(tag) } else { None })
        .collect();
    duplicates.sort();
    duplicates
}

pub fn ensure_unique_account_tags(backends: &[BackendConfig]) -> anyhow::Result<()> {
    let duplicates = duplicate_account_tags(backends);
    if !duplicates.is_empty() {
        anyhow::bail!(
            "Duplicate account_tag detected: {}. account_tag must be unique.",
            duplicates.join(", ")
        );
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientConfig {
    pub bind_addr: String,
    pub api_key: Option<String>,
    pub auth_enabled: bool,
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AdminConfig {
    pub enabled: bool,
    pub bind_addr: String,
    pub api_key: Option<String>,
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
}

fn default_max_probe() -> u64 { 14400 }

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let actual_path = if path == "config.yaml" {
            crate::paths::get_path("config.yaml")
        } else {
            path.to_string()
        };
        
        if !std::path::Path::new(&actual_path).exists() {
            return Ok(Config {
                client: ClientConfig {
                    bind_addr: "0.0.0.0:3000".into(),
                    api_key: None,
                    auth_enabled: false,
                    tls: None,
                },
                admin: AdminConfig {
                    enabled: true,
                    bind_addr: "127.0.0.1:3001".into(),
                    api_key: None,
                    tls: None,
                },
                fallback_strategy: FallbackStrategy::Performance,
                backends: vec![],
                aliases: std::collections::HashMap::new(),
                chains: vec![],
                max_probe_interval_secs: 14400,
            });
        }
        let content = fs::read_to_string(&actual_path)?;

        // Load as YAML Value first so we can migrate legacy `name` -> `models` fields
        let mut value: serde_yaml::Value = serde_yaml::from_str(&content)?;
        let mut migrated = false;
        if let serde_yaml::Value::Mapping(ref mut m) = value {
            if let Some(backends_val) = m.get_mut(&serde_yaml::Value::String("backends".into())) {
                if let serde_yaml::Value::Sequence(seq) = backends_val {
                    for item in seq.iter_mut() {
                        if let serde_yaml::Value::Mapping(map) = item {
                            // If legacy `name` exists and `models` not present, migrate
                            let has_models = map.keys().any(|k| matches!(k, serde_yaml::Value::String(s) if s == "models"));
                            if !has_models {
                                if let Some(name_val) = map.remove(&serde_yaml::Value::String("name".into())) {
                                    map.insert(serde_yaml::Value::String("models".into()), serde_yaml::Value::Sequence(vec![name_val]));
                                    migrated = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if migrated {
            // write back migrated YAML
            let new_content = serde_yaml::to_string(&value)?;
            crate::paths::write_atomic("config.yaml", &new_content)?;
        }

        let config: Config = serde_yaml::from_value(value)?;
        ensure_unique_account_tags(&config.backends)?;
        Ok(config)
    }

    pub fn save(&self, _path: &str) -> anyhow::Result<()> {
        ensure_unique_account_tags(&self.backends)?;
        let content = serde_yaml::to_string(self)?;
        crate::paths::write_atomic("config.yaml", &content)?;
        Ok(())
    }
}
