use oneapi::config::{
    ensure_unique_account_tags, AdminConfig, BackendConfig, ClientConfig, Config,
};
use oneapi::types::FallbackStrategy;
use std::collections::HashMap;
use std::fs;
use tempfile::tempdir;

#[test]
fn ensure_unique_account_tags_rejects_duplicates() {
    let backends = vec![
        BackendConfig::Openai {
            models: vec!["gpt-4o".into()],
            account_tag: "dup".into(),
            api_key: "sk-a".into(),
            base_url: "https://api.openai.com/v1".into(),
            max_context: 128000,
            budget_limit: None,
        },
        BackendConfig::Cli {
            models: vec!["gemini-1.5-flash".into()],
            account_tag: "dup".into(),
            command: "echo ok".into(),
            max_context: 1000000,
            budget_limit: None,
            json_path: None,
        },
    ];

    let err = ensure_unique_account_tags(&backends).unwrap_err();
    assert!(err.to_string().contains("Duplicate account_tag"));
}

#[test]
fn config_load_rejects_duplicate_account_tags() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.yaml");
    let conf = Config {
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
        backends: vec![
            BackendConfig::Gemini {
                models: vec!["gemini-1.5-flash".into()],
                account_tag: "same-tag".into(),
                api_key: "k1".into(),
                max_context: 1000000,
                budget_limit: None,
            },
            BackendConfig::Cli {
                models: vec!["gemini-1.5-pro".into()],
                account_tag: "same-tag".into(),
                command: "echo ok".into(),
                max_context: 1000000,
                budget_limit: None,
                json_path: None,
            },
        ],
        aliases: HashMap::new(),
        chains: vec![],
        max_probe_interval_secs: 14400,
    };

    fs::write(&config_path, serde_yaml::to_string(&conf).unwrap()).unwrap();

    let err = Config::load(config_path.to_str().unwrap()).unwrap_err();
    assert!(err.to_string().contains("Duplicate account_tag"));
}
