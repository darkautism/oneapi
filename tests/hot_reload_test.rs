use oneapi::types::FallbackStrategy;
use oneapi::config::{Config, BackendConfig};
use oneapi::router::Router;
use oneapi::quota::QuotaManager;
use oneapi::registry::IntelligenceRegistry;
use std::sync::Arc;
use std::collections::HashMap;
use std::fs;
use tempfile::tempdir;

#[tokio::test]
async fn test_config_hot_reload() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.yaml");
    
    // Initial config: 0 backends
    let initial_conf = Config {
        client: oneapi::config::ClientConfig { bind_addr: "0.0.0.0:3000".into(), api_key: None, auth_enabled: false, tls: None },
        admin: oneapi::config::AdminConfig { enabled: true, bind_addr: "127.0.0.1:3001".into(), api_key: None, tls: None },
        fallback_strategy: FallbackStrategy::Performance,
        backends: vec![],
        aliases: HashMap::new(),
        chains: vec![],
        max_probe_interval_secs: 14400,
    };
    let yaml = serde_yaml::to_string(&initial_conf).unwrap();
    fs::write(&config_path, yaml).unwrap();

    let registry = Arc::new(IntelligenceRegistry::new());
    let quota_manager = Arc::new(QuotaManager::new(HashMap::new()));
    let router = Arc::new(tokio::sync::RwLock::new(Arc::new(Router::new(vec![], quota_manager, registry, HashMap::new(), vec![]))));
    
    // Simulate AppState
    let router_clone = router.clone();
    let config_path_str = config_path.to_str().unwrap().to_string();
    
    // Check initial state
    {
        let r = router.read().await;
        assert_eq!(r.backends.len(), 0);
    }

    // Update config file
    let mut updated_conf = initial_conf.clone();
    updated_conf.backends.push(BackendConfig::Openai {
        models: vec!["gpt-4o".into()],
        account_tag: "test-account".into(),
        api_key: "sk-123".into(),
        base_url: "https://api.openai.com/v1".into(),
        max_context: 128000,
        budget_limit: None,
    });
    let yaml_upd = serde_yaml::to_string(&updated_conf).unwrap();
    fs::write(&config_path, yaml_upd).unwrap();

    // Trigger reload logic (mimicking AppState::reload_router)
    let conf = Config::load(&config_path_str).unwrap();
    let mut backends: Vec<Arc<dyn oneapi::providers::ModelBackend>> = Vec::new();
    for b in &conf.backends {
        match b {
            BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, .. } => {
                backends.push(Arc::new(oneapi::providers::openai::OpenAIBackend::new(models.clone(), account_tag, *max_context, api_key, base_url)));
            }
            _ => {}
        }
    }
    let new_router = Router::new(backends, Arc::new(QuotaManager::new(HashMap::new())), Arc::new(IntelligenceRegistry::new()), conf.aliases.clone(), conf.chains.clone());
    *router_clone.write().await = Arc::new(new_router);

    // Verify reload
    {
        let r = router.read().await;
        assert_eq!(r.backends.len(), 1);
        assert_eq!(r.backends[0].info().account_tag, "test-account");
    }
}
