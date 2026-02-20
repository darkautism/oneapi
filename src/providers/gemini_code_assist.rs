use async_trait::async_trait;
use crate::providers::{BaseBackend, ModelBackend, StreamResponse};
use crate::types::*;
use futures::stream;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION: &str = "v1internal";
const PROJECT_CACHE_FILE: &str = "oneapi_code_assist.json";

pub struct GeminiCodeAssistBackend {
    pub base: BaseBackend,
    pub client: Client,
    pub gemini_home: PathBuf,
}

#[derive(Deserialize)]
struct RefreshTokenResponse {
    access_token: String,
    expires_in: i64,
    refresh_token: Option<String>,
    token_type: Option<String>,
}

impl GeminiCodeAssistBackend {
    pub fn new(models: Vec<String>, account: &str, max_context: u32, gemini_home: PathBuf) -> Self {
        Self {
            base: BaseBackend::new(models, account, max_context),
            client: Client::new(),
            gemini_home,
        }
    }

    fn oauth_creds_path(&self) -> PathBuf {
        self.gemini_home.join(".gemini").join("oauth_creds.json")
    }

    fn project_cache_path(&self) -> PathBuf {
        self.gemini_home.join(".gemini").join(PROJECT_CACHE_FILE)
    }

    fn as_i64(value: Option<&Value>) -> Option<i64> {
        match value {
            Some(Value::Number(n)) => n.as_i64(),
            Some(Value::String(s)) => s.parse::<i64>().ok(),
            _ => None,
        }
    }

    fn read_json_file(path: &Path) -> Result<Value, ProviderError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| ProviderError::Offline(format!("Failed reading {}: {}", path.display(), e)))?;
        serde_json::from_str(&raw)
            .map_err(|e| ProviderError::ApiError(format!("Invalid JSON at {}: {}", path.display(), e)))
    }

    fn write_json_file_secure(path: &Path, value: &Value) -> Result<(), ProviderError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ProviderError::Offline(format!("Failed creating {}: {}", parent.display(), e)))?;
        }
        let content = format!(
            "{}\n",
            serde_json::to_string_pretty(value)
                .map_err(|e| ProviderError::ApiError(format!("JSON serialize failed: {}", e)))?
        );
        std::fs::write(path, content)
            .map_err(|e| ProviderError::Offline(format!("Failed writing {}: {}", path.display(), e)))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| ProviderError::Offline(format!("Failed chmod {}: {}", path.display(), e)))?;
        }
        Ok(())
    }

    fn extract_project_id(value: &Value) -> Option<String> {
        if let Some(s) = value.get("cloudaicompanionProject").and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
        if let Some(s) = value
            .get("cloudaicompanionProject")
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
        {
            return Some(s.to_string());
        }
        if let Some(s) = value.get("project_id").and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
        if let Some(response) = value.get("response") {
            return Self::extract_project_id(response);
        }
        None
    }

    fn metadata_body(project: Option<&str>) -> Value {
        serde_json::json!({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": project,
        })
    }

    fn request_headers(access_token: &str) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        let auth_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", access_token))
            .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("Bearer"));
        headers.insert(
            reqwest::header::AUTHORIZATION,
            auth_value,
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("google-cloud-sdk vscode_cloudshelleditor/0.1"),
        );
        headers.insert(
            "X-Goog-Api-Client",
            reqwest::header::HeaderValue::from_static("gl-node/22.17.0"),
        );
        headers.insert(
            "Client-Metadata",
            reqwest::header::HeaderValue::from_static("{\"ideType\":\"IDE_UNSPECIFIED\",\"platform\":\"PLATFORM_UNSPECIFIED\",\"pluginType\":\"GEMINI\"}"),
        );
        headers
    }

    async fn refresh_access_token_if_needed(&self) -> Result<String, ProviderError> {
        let creds_path = self.oauth_creds_path();
        let mut creds = Self::read_json_file(&creds_path)?;
        let now_ms = chrono::Utc::now().timestamp_millis();
        let expiry = Self::as_i64(creds.get("expiry_date")).unwrap_or(0);
        let current_access = creds
            .get("access_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(access_token) = current_access {
            if expiry > now_ms + 60_000 {
                return Ok(access_token);
            }
        }

        let refresh_token = creds
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProviderError::ApiError("Gemini OAuth refresh_token missing; re-login required.".to_string()))?
            .to_string();
        let (oauth_client_id, oauth_client_secret) =
            crate::gemini_oauth::oauth_client_credentials().ok_or_else(|| {
                ProviderError::ApiError(
                    "Gemini OAuth client credentials not found. Set ONEAPI_GEMINI_OAUTH_CLIENT_ID/SECRET or ONEAPI_GEMINI_OAUTH_SOURCE_FILE."
                        .to_string(),
                )
            })?;

        let form_body = reqwest::Url::parse_with_params(
            "https://dummy.local",
            &[
                ("client_id", oauth_client_id.as_str()),
                ("client_secret", oauth_client_secret.as_str()),
                ("refresh_token", refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ],
        )
        .map_err(|e| ProviderError::ApiError(format!("Failed building refresh request: {}", e)))?
        .query()
        .unwrap_or_default()
        .to_string();

        let token_resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(form_body)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("Token refresh request failed: {}", e)))?;

        if !token_resp.status().is_success() {
            let body = token_resp.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError(format!("Gemini OAuth refresh failed: {}", body)));
        }

        let token = token_resp
            .json::<RefreshTokenResponse>()
            .await
            .map_err(|e| ProviderError::ApiError(format!("Invalid refresh token response: {}", e)))?;

        let new_expiry = chrono::Utc::now().timestamp_millis() + token.expires_in * 1000;
        let obj = creds.as_object_mut().ok_or_else(|| {
            ProviderError::ApiError(format!("{} is not a JSON object", creds_path.display()))
        })?;
        obj.insert(
            "access_token".to_string(),
            Value::String(token.access_token.clone()),
        );
        if let Some(refresh) = token.refresh_token {
            obj.insert("refresh_token".to_string(), Value::String(refresh));
        }
        obj.insert("expiry_date".to_string(), Value::Number(new_expiry.into()));
        if let Some(token_type) = token.token_type {
            obj.insert("token_type".to_string(), Value::String(token_type));
        }
        Self::write_json_file_secure(&creds_path, &creds)?;
        Ok(token.access_token)
    }

    async fn post_code_assist(&self, method: &str, access_token: &str, body: &Value) -> Result<Value, ProviderError> {
        let url = format!("{}/{}:{}", CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION, method);
        let response = self
            .client
            .post(url)
            .headers(Self::request_headers(access_token))
            .json(body)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("Code Assist request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError(format!(
                "Code Assist {} failed ({}): {}",
                method, status, text
            )));
        }
        response
            .json::<Value>()
            .await
            .map_err(|e| ProviderError::ApiError(format!("Invalid Code Assist {} response: {}", method, e)))
    }

    async fn get_code_assist_operation(&self, operation: &str, access_token: &str) -> Result<Value, ProviderError> {
        let url = format!("{}/{}/{}", CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION, operation);
        let response = self
            .client
            .get(url)
            .headers(Self::request_headers(access_token))
            .send()
            .await
            .map_err(|e| ProviderError::Offline(format!("Code Assist operation poll failed: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError(format!(
                "Code Assist operation poll failed ({}): {}",
                status, text
            )));
        }
        response
            .json::<Value>()
            .await
            .map_err(|e| ProviderError::ApiError(format!("Invalid Code Assist poll response: {}", e)))
    }

    async fn discover_or_onboard_project(&self, access_token: &str) -> Result<String, ProviderError> {
        let env_project = std::env::var("GOOGLE_CLOUD_PROJECT")
            .ok()
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok());

        let load_body = serde_json::json!({
            "cloudaicompanionProject": env_project.clone(),
            "metadata": Self::metadata_body(env_project.as_deref()),
        });
        let load_data = self.post_code_assist("loadCodeAssist", access_token, &load_body).await?;

        if let Some(project_id) = Self::extract_project_id(&load_data) {
            return Ok(project_id);
        }
        if load_data.get("currentTier").is_some() {
            if let Some(project_id) = env_project {
                return Ok(project_id);
            }
            return Err(ProviderError::ApiError(
                "This account requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID.".to_string(),
            ));
        }

        let default_tier = load_data
            .get("allowedTiers")
            .and_then(|v| v.as_array())
            .and_then(|tiers| {
                tiers
                    .iter()
                    .find(|tier| tier.get("isDefault").and_then(|v| v.as_bool()) == Some(true))
            })
            .and_then(|tier| tier.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("legacy-tier");

        if default_tier != "free-tier" && env_project.is_none() {
            return Err(ProviderError::ApiError(
                "This account requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID.".to_string(),
            ));
        }

        let mut onboard_body = serde_json::json!({
            "tierId": default_tier,
            "metadata": Self::metadata_body(if default_tier == "free-tier" { None } else { env_project.as_deref() }),
        });
        if default_tier != "free-tier" {
            if let Some(project_id) = &env_project {
                onboard_body["cloudaicompanionProject"] = Value::String(project_id.clone());
            }
        }

        let mut lro_data = self.post_code_assist("onboardUser", access_token, &onboard_body).await?;
        let mut attempts = 0usize;
        while lro_data.get("done").and_then(|v| v.as_bool()) != Some(true) {
            let op_name = lro_data
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::ApiError("onboardUser did not return operation name".to_string()))?;
            if attempts >= 24 {
                return Err(ProviderError::ApiError(
                    "Timed out while waiting for Code Assist onboarding.".to_string(),
                ));
            }
            attempts += 1;
            tokio::time::sleep(Duration::from_secs(5)).await;
            lro_data = self.get_code_assist_operation(op_name, access_token).await?;
        }

        if let Some(project_id) = Self::extract_project_id(&lro_data) {
            return Ok(project_id);
        }
        if let Some(project_id) = env_project {
            return Ok(project_id);
        }
        Err(ProviderError::ApiError(
            "Unable to determine Cloud Code Assist project ID.".to_string(),
        ))
    }

    async fn ensure_project_id(&self, access_token: &str) -> Result<String, ProviderError> {
        let project_cache_path = self.project_cache_path();
        if project_cache_path.exists() {
            let cached = Self::read_json_file(&project_cache_path)?;
            if let Some(project_id) = Self::extract_project_id(&cached) {
                return Ok(project_id);
            }
        }

        let creds_path = self.oauth_creds_path();
        if creds_path.exists() {
            let creds = Self::read_json_file(&creds_path)?;
            if let Some(project_id) = Self::extract_project_id(&creds) {
                return Ok(project_id);
            }
        }

        let project_id = self.discover_or_onboard_project(access_token).await?;
        let cache_value = serde_json::json!({ "project_id": project_id });
        Self::write_json_file_secure(&project_cache_path, &cache_value)?;

        if creds_path.exists() {
            let mut creds = Self::read_json_file(&creds_path)?;
            if let Some(obj) = creds.as_object_mut() {
                obj.insert("project_id".to_string(), Value::String(project_id.clone()));
                let _ = Self::write_json_file_secure(&creds_path, &creds);
            }
        }
        Ok(project_id)
    }

    fn extract_message_text(content: &Value) -> String {
        if let Some(text) = content.as_str() {
            text.to_string()
        } else {
            content.to_string()
        }
    }

    fn to_code_assist_contents(messages: &[ChatMessage]) -> Vec<Value> {
        messages
            .iter()
            .map(|m| {
                let role = if m.role == "assistant" { "model" } else { "user" };
                serde_json::json!({
                    "role": role,
                    "parts": [{ "text": Self::extract_message_text(&m.content) }],
                })
            })
            .collect()
    }

    fn to_code_assist_tools(tools: Option<&Vec<OpenAITool>>) -> Option<Value> {
        let tools = tools?;
        if tools.is_empty() {
            return None;
        }
        let function_declarations: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                if tool.r#type != "function" {
                    return None;
                }
                Some(serde_json::json!({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parametersJsonSchema": tool.function.parameters,
                }))
            })
            .collect();
        if function_declarations.is_empty() {
            return None;
        }
        Some(serde_json::json!([{
            "functionDeclarations": function_declarations
        }]))
    }

    fn usage_from_response(value: &Value) -> Option<Usage> {
        let usage = value
            .get("response")
            .and_then(|v| v.get("usageMetadata"))
            .cloned()?;
        let prompt = usage
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion = usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let total = usage
            .get("totalTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or((prompt + completion) as u64) as u32;
        Some(Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
        })
    }

    fn first_candidate(value: &Value) -> Option<(String, Option<String>)> {
        let candidate = value
            .get("response")
            .and_then(|v| v.get("candidates"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())?;
        let finish_reason = candidate
            .get("finishReason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let parts = candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = String::new();
        for part in parts {
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                out.push_str(text);
            }
        }
        Some((out, finish_reason))
    }
}

#[async_trait]
impl ModelBackend for GeminiCodeAssistBackend {
    fn info(&self) -> &ProviderInfo {
        &self.base.info
    }

    async fn status(&self) -> ProviderStatus {
        self.base.status.read().await.clone()
    }

    async fn set_status(&self, status: ProviderStatus) {
        *self.base.status.write().await = status;
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let access_token = self.refresh_access_token_if_needed().await?;
        let project_id = self.ensure_project_id(&access_token).await?;
        let model_name = req.model.strip_prefix("google/").unwrap_or(&req.model).to_string();
        let code_assist_tools = Self::to_code_assist_tools(req.tools.as_ref());

        let mut generation_config = serde_json::Map::new();
        if let Some(temperature) = req.temperature {
            generation_config.insert("temperature".to_string(), serde_json::json!(temperature));
        }
        if let Some(max_output_tokens) = req.max_tokens {
            generation_config.insert("maxOutputTokens".to_string(), serde_json::json!(max_output_tokens));
        }

        let mut request = serde_json::json!({
            "contents": Self::to_code_assist_contents(&req.messages)
        });
        if !generation_config.is_empty() {
            request["generationConfig"] = Value::Object(generation_config);
        }
        if let Some(tools) = code_assist_tools {
            request["tools"] = tools;
            request["toolConfig"] = serde_json::json!({
                "functionCallingConfig": {
                    "mode": "AUTO"
                }
            });
        }

        let request_body = serde_json::json!({
            "project": project_id,
            "model": model_name,
            "request": request,
            "userAgent": "oneapi",
            "requestId": format!("oneapi-{}", uuid::Uuid::new_v4()),
        });

        let url = format!("{}/{}:generateContent", CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION);
        let response = self
            .client
            .post(url)
            .headers(Self::request_headers(&access_token))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ProviderError::Offline(e.to_string()))?;

        if response.status().as_u16() == 429 {
            self.set_status(ProviderStatus::RateLimited(Instant::now() + Duration::from_secs(60))).await;
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            if body.to_lowercase().contains("quota") || body.to_lowercase().contains("exhausted") {
                self.set_status(ProviderStatus::QuotaExceeded).await;
                return Err(ProviderError::QuotaExceeded);
            }
            return Err(ProviderError::ApiError(body));
        }

        let payload = response
            .json::<Value>()
            .await
            .map_err(|e| ProviderError::ApiError(format!("Invalid generateContent response: {}", e)))?;

        let (text, finish_reason) = Self::first_candidate(&payload)
            .ok_or_else(|| ProviderError::ApiError("No candidates in Code Assist response".to_string()))?;

        Ok(ChatResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String(text),
                },
                finish_reason,
            }],
            usage: Self::usage_from_response(&payload),
        })
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<StreamResponse, ProviderError> {
        let model_id = req.model.clone();
        let response = self.chat(req).await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_str())
            .unwrap_or_default()
            .to_string();

        let fut = async move {
            let chunk = ChatStreamChunk::new(model_id, text, Some("stop".into()));
            Ok(serde_json::to_string(&chunk).unwrap())
        };
        Ok(Box::pin(stream::once(fut)))
    }
}
