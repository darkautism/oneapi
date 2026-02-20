use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};
use oneapi::types::*;
use oneapi::providers::openai::OpenAIBackend;
use oneapi::router::Router;
use oneapi::quota::QuotaManager;
use oneapi::registry::IntelligenceRegistry;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::test]
async fn test_fallback_on_429() {
    let server_a = MockServer::start().await;
    let server_b = MockServer::start().await;

    // Server A always returns 429
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server_a)
        .await;

    // Server B returns success
    let success_resp = ChatResponse {
        id: "test-b".into(),
        object: "chat.completion".into(),
        created: 12345,
        model: "gpt-4o".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage { role: "assistant".into(), content: serde_json::json!("I am B") },
            finish_reason: Some("stop".into()),
        }],
        usage: Some(Usage { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }),
    };

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_resp))
        .mount(&server_b)
        .await;

    let backend_a = Arc::new(OpenAIBackend::new(vec!["gpt-4o".to_string()], "acc-a", 1000, "key", &server_a.uri()));
    let backend_b = Arc::new(OpenAIBackend::new(vec!["gpt-4o".to_string()], "acc-b", 1000, "key", &server_b.uri()));

    let registry = Arc::new(IntelligenceRegistry::new());
    let quota_manager = Arc::new(QuotaManager::new(HashMap::new()));
    let router = Router::new(vec![backend_a, backend_b], quota_manager, registry, std::collections::HashMap::new(), vec![]);

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage { role: "user".into(), content: serde_json::json!("hi") }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
    };

    let res = router.chat(req, FallbackStrategy::Performance).await.unwrap();
    assert_eq!(res.id, "test-b");
    assert_eq!(res.choices[0].message.content, serde_json::json!("I am B"));
}
