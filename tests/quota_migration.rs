use oneapi::quota::QuotaManager;
use oneapi::providers::ModelBackend;
use std::sync::Arc;
use std::collections::HashMap;

struct DummyBackend { info: oneapi::types::ProviderInfo }

#[async_trait::async_trait]
impl ModelBackend for DummyBackend {
    fn info(&self) -> &oneapi::types::ProviderInfo { &self.info }
    async fn status(&self) -> oneapi::types::ProviderStatus { oneapi::types::ProviderStatus::Healthy }
    async fn set_status(&self, _s: oneapi::types::ProviderStatus) {}
    async fn chat(&self, _req: oneapi::types::ChatRequest) -> Result<oneapi::types::ChatResponse, oneapi::types::ProviderError> { Err(oneapi::types::ProviderError::Offline("no".into())) }
    async fn chat_stream(&self, _req: oneapi::types::ChatRequest) -> Result<oneapi::providers::StreamResponse, oneapi::types::ProviderError> { Err(oneapi::types::ProviderError::ApiError("no".into())) }
}

#[tokio::test]
async fn migrate_legacy_spends_even_split() {
    let mut spends = HashMap::new();
    spends.insert("acct1".to_string(), 10.0);

    let qm = QuotaManager::with_spends_and_limits(spends, HashMap::new());

    let backend = DummyBackend { info: oneapi::types::ProviderInfo { models: vec!["m1".into(), "m2".into()], account_tag: "acct1".into(), max_context: 1000 } };
    let backends: Vec<Arc<dyn ModelBackend>> = vec![Arc::new(backend)];

    qm.migrate_spends(&backends).await.unwrap();

    let spends_map = qm.spends.read().await;
    assert!(spends_map.get("acct1::m1").is_some());
    assert_eq!(*spends_map.get("acct1::m1").unwrap(), 5.0);
    assert_eq!(*spends_map.get("acct1::m2").unwrap(), 5.0);
}
