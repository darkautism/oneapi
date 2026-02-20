use oneapi::paths;
use oneapi::types::*;
use oneapi::providers::{openai::OpenAIBackend, cli::CliBackend, gemini_code_assist::GeminiCodeAssistBackend, ModelBackend};
use oneapi::router::Router;
use oneapi::quota::QuotaManager;
use oneapi::registry::IntelligenceRegistry;
use oneapi::config::{Config, BackendConfig, TlsConfig, ensure_unique_account_tags};
use clap::{Parser, Subcommand};
use dialoguer::{Input, MultiSelect, theme::ColorfulTheme};
use serde::Deserialize;
use std::sync::Arc;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use axum::{
    routing::{post, get}, Json, Router as AxumRouter, extract::State, 
    http::{StatusCode, Request}, middleware::{self, Next},
    response::{Response, IntoResponse}
};
use axum::response::sse::{Event, Sse};
use tokio_stream::StreamExt as _;
use axum_server::tls_rustls::RustlsConfig;

const GEMINI_OAUTH_AUTH_TYPE: &str = "oauth-personal";
const GEMINI_OAUTH_REDIRECT_URI: &str = "https://codeassist.google.com/authcode";
const GEMINI_OAUTH_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
];

fn parse_model_ids(raw: &str) -> anyhow::Result<Vec<String>> {
    let models: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if models.is_empty() {
        anyhow::bail!("model_ids must include at least one model");
    }
    Ok(models)
}

fn default_gemini_model_ids() -> Vec<String> {
    vec![
        "gemini-2.0-flash".to_string(),
        "gemini-2.5-flash-lite".to_string(),
        "gemini-2.5-flash".to_string(),
        "gemini-2.5-pro".to_string(),
        "gemini-3-flash-preview".to_string(),
        "gemini-3-pro-preview".to_string(),
    ]
}

fn is_code_assist_gemini_model(model_id: &str) -> bool {
    model_id.starts_with("gemini-2.") || model_id.starts_with("gemini-3.")
}

fn oauth_credentials_exist(gemini_home: &Path) -> bool {
    gemini_home.join(".gemini").join("oauth_creds.json").exists()
        || gemini_home.join(".gemini").join("google_accounts.json").exists()
}

fn extract_gemini_model_ids(text: &str) -> Vec<String> {
    let mut models = Vec::new();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '.' && c != '/');
        if token.is_empty() {
            continue;
        }
        let candidate = token.rsplit('/').next().unwrap_or(token);
        if candidate.starts_with("gemini-")
            && candidate
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
        {
            let model = candidate.to_string();
            if !models.contains(&model) {
                models.push(model);
            }
        }
    }
    models
}

fn fetch_gemini_models_from_cli(gemini_home: &Path) -> Vec<String> {
    let cmd = format!(
        "timeout 12 env GEMINI_CLI_HOME={} GEMINI_DEFAULT_AUTH_TYPE={} NO_BROWSER=true gemini -p '/models' --output-format text 2>&1",
        shell_single_quote(&gemini_home.to_string_lossy()),
        GEMINI_OAUTH_AUTH_TYPE
    );
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output();
    match output {
        Ok(out) => {
            let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
            if !out.stderr.is_empty() {
                combined.push('\n');
                combined.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            extract_gemini_model_ids(&combined)
        }
        Err(_) => Vec::new(),
    }
}

fn merge_model_candidates(base: Vec<String>, extra: &[String]) -> Vec<String> {
    let mut merged = base;
    for item in extra {
        if !merged.contains(item) {
            merged.push(item.clone());
        }
    }
    merged
}

fn select_gemini_models_interactive(
    account_tag: &str,
    choices: &[String],
    defaults: &[String],
) -> anyhow::Result<Vec<String>> {
    if choices.is_empty() {
        anyhow::bail!("No Gemini models available; pass --model-ids explicitly");
    }

    let theme = ColorfulTheme::default();
    let default_flags: Vec<bool> = choices.iter().map(|m| defaults.contains(m)).collect();
    let selected = MultiSelect::with_theme(&theme)
        .with_prompt(format!("Select models for '{}'", account_tag))
        .items(choices)
        .defaults(&default_flags)
        .interact()?;
    if selected.is_empty() {
        anyhow::bail!("At least one model must be selected");
    }
    Ok(selected.into_iter().map(|idx| choices[idx].clone()).collect())
}

fn resolve_gemini_model_ids(
    raw: &str,
    account_tag: &str,
    existing_models: Option<&[String]>,
) -> anyhow::Result<Vec<String>> {
    if !raw.eq_ignore_ascii_case("auto") {
        return parse_model_ids(raw);
    }

    let registry_models = default_gemini_model_ids();
    let defaults = if let Some(existing) = existing_models {
        if !existing.is_empty() && existing.iter().any(|m| is_code_assist_gemini_model(m)) {
            existing.to_vec()
        } else {
            registry_models.clone()
        }
    } else {
        registry_models.clone()
    };

    let mut candidates = merge_model_candidates(registry_models, &defaults);
    if candidates.is_empty() {
        anyhow::bail!("No Gemini model candidates available; pass --model-ids explicitly");
    }

    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Ok(defaults);
    }

    let gemini_home = gemini_oauth_home(account_tag)?;
    if oauth_credentials_exist(&gemini_home) {
        let cli_models = fetch_gemini_models_from_cli(&gemini_home);
        candidates = merge_model_candidates(candidates, &cli_models);
    }

    select_gemini_models_interactive(account_tag, &candidates, &defaults)
}

fn validate_account_tag_for_oauth(account_tag: &str) -> anyhow::Result<()> {
    if account_tag.trim().is_empty() {
        anyhow::bail!("account_tag cannot be empty");
    }
    if account_tag == "." || account_tag == ".." || account_tag.contains('/') || account_tag.contains('\\') {
        anyhow::bail!("account_tag cannot contain path separators or reserved values '.'/'..'");
    }
    Ok(())
}

fn gemini_oauth_home(account_tag: &str) -> anyhow::Result<PathBuf> {
    validate_account_tag_for_oauth(account_tag)?;
    Ok(paths::get_config_dir().join("gemini-oauth").join(account_tag))
}

fn ensure_gemini_oauth_settings(gemini_home: &Path) -> anyhow::Result<()> {
    let gemini_dir = gemini_home.join(".gemini");
    std::fs::create_dir_all(&gemini_dir)?;
    let settings_path = gemini_dir.join("settings.json");

    let mut settings = if settings_path.exists() {
        let raw = std::fs::read_to_string(&settings_path)?;
        serde_json::from_str::<serde_json::Value>(&raw)?
    } else {
        serde_json::json!({})
    };

    let root = settings
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{} must be a JSON object", settings_path.display()))?;

    if !root.contains_key("security") {
        root.insert("security".to_string(), serde_json::json!({}));
    }
    let security = root
        .get_mut("security")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| anyhow::anyhow!("{}.security must be an object", settings_path.display()))?;

    if !security.contains_key("auth") {
        security.insert("auth".to_string(), serde_json::json!({}));
    }
    let auth = security
        .get_mut("auth")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| anyhow::anyhow!("{}.security.auth must be an object", settings_path.display()))?;

    auth.insert(
        "selectedType".to_string(),
        serde_json::Value::String(GEMINI_OAUTH_AUTH_TYPE.to_string()),
    );

    let content = format!("{}\n", serde_json::to_string_pretty(&settings)?);
    std::fs::write(&settings_path, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&settings_path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn read_active_google_account(gemini_home: &Path) -> anyhow::Result<Option<String>> {
    let account_path = gemini_home.join(".gemini").join("google_accounts.json");
    if !account_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&account_path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(value.get("active").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

fn shell_single_quote(input: &str) -> String {
    format!("'{}'", input.replace('\'', "'\"'\"'"))
}

fn gemini_oauth_command(gemini_home: &Path) -> String {
    format!(
        "GEMINI_CLI_HOME={} gemini --model {{model}} -p '{{messages}}' --output-format text",
        shell_single_quote(&gemini_home.to_string_lossy())
    )
}

fn is_gemini_oauth_cli_command(command: &str) -> bool {
    command.contains("GEMINI_CLI_HOME=") && command.contains(" gemini ")
}

fn extract_gemini_oauth_home_from_command(command: &str) -> Option<PathBuf> {
    let prefix = "GEMINI_CLI_HOME=";
    let start = command.find(prefix)?;
    let rest = &command[start + prefix.len()..];
    if rest.is_empty() {
        return None;
    }

    if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        return Some(PathBuf::from(&stripped[..end]));
    }
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(PathBuf::from(&stripped[..end]));
    }

    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    Some(PathBuf::from(&rest[..end]))
}

fn gemini_oauth_manual_login_command(gemini_home: &Path) -> String {
    format!(
        "GEMINI_CLI_HOME={} NO_BROWSER=true gemini",
        shell_single_quote(&gemini_home.to_string_lossy())
    )
}

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: i64,
    refresh_token: Option<String>,
    token_type: Option<String>,
    scope: Option<String>,
}

#[derive(Deserialize)]
struct GoogleUserInfoResponse {
    email: Option<String>,
}

fn generate_oauth_state() -> String {
    use rand::RngExt;
    let mut bytes = [0u8; 16];
    rand::rng().fill(&mut bytes);
    hex::encode(bytes)
}

fn generate_pkce_verifier() -> String {
    use base64::Engine as _;
    use rand::RngExt;
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn generate_pkce_challenge(verifier: &str) -> String {
    use base64::Engine as _;
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

fn parse_authorization_input(raw: &str) -> anyhow::Result<(String, Option<String>)> {
    let input = raw.trim();
    if input.is_empty() {
        anyhow::bail!("authorization input is empty");
    }

    if let Ok(url) = reqwest::Url::parse(input) {
        let code = url
            .query_pairs()
            .find(|(k, _)| k == "code")
            .map(|(_, v)| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("redirect URL does not contain 'code' parameter"))?;
        let state = url
            .query_pairs()
            .find(|(k, _)| k == "state")
            .map(|(_, v)| v.to_string());
        return Ok((code, state));
    }

    if input.contains("code=") {
        let query_like = input.trim_start_matches('?');
        if let Ok(url) = reqwest::Url::parse(&format!("https://dummy.local/?{}", query_like)) {
            if let Some(code) = url
                .query_pairs()
                .find(|(k, _)| k == "code")
                .map(|(_, v)| v.to_string())
            {
                let state = url
                    .query_pairs()
                    .find(|(k, _)| k == "state")
                    .map(|(_, v)| v.to_string());
                return Ok((code, state));
            }
        }
    }

    Ok((input.to_string(), None))
}

fn write_json_file_secure(path: &Path, value: &serde_json::Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = format!("{}\n", serde_json::to_string_pretty(value)?);
    std::fs::write(path, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn write_active_google_account(gemini_home: &Path, email: &str) -> anyhow::Result<()> {
    let accounts_path = gemini_home.join(".gemini").join("google_accounts.json");
    let mut old: Vec<String> = Vec::new();

    if accounts_path.exists() {
        let raw = std::fs::read_to_string(&accounts_path)?;
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(prev_active) = value.get("active").and_then(|v| v.as_str()) {
                if prev_active != email {
                    old.push(prev_active.to_string());
                }
            }
            if let Some(prev_old) = value.get("old").and_then(|v| v.as_array()) {
                for item in prev_old {
                    if let Some(s) = item.as_str() {
                        old.push(s.to_string());
                    }
                }
            }
        }
    }

    old.retain(|s| s != email);
    old.sort();
    old.dedup();

    write_json_file_secure(
        &accounts_path,
        &serde_json::json!({
            "active": email,
            "old": old,
        }),
    )
}

async fn run_native_headless_oauth_login(gemini_home: &Path) -> anyhow::Result<Option<String>> {
    let (oauth_client_id, oauth_client_secret) = oneapi::gemini_oauth::oauth_client_credentials()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Gemini OAuth client credentials not found. Set ONEAPI_GEMINI_OAUTH_CLIENT_ID/SECRET or ONEAPI_GEMINI_OAUTH_SOURCE_FILE."
            )
        })?;
    let state = generate_oauth_state();
    let code_verifier = generate_pkce_verifier();
    let code_challenge = generate_pkce_challenge(&code_verifier);
    let scope = GEMINI_OAUTH_SCOPES.join(" ");

    let mut auth_url = reqwest::Url::parse("https://accounts.google.com/o/oauth2/v2/auth")?;
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", &oauth_client_id)
        .append_pair("redirect_uri", GEMINI_OAUTH_REDIRECT_URI)
        .append_pair("response_type", "code")
        .append_pair("scope", &scope)
        .append_pair("code_challenge_method", "S256")
        .append_pair("code_challenge", &code_challenge)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent")
        .append_pair("state", &state);

    println!("[Gemini OAuth] Open this URL and complete authentication:");
    println!("{}", auth_url);
    println!("[Gemini OAuth] Paste the authorization code (or full redirect URL).");

    let input: String = Input::new()
        .with_prompt("Authorization code / redirect URL")
        .interact_text()?;
    let (code, returned_state) = parse_authorization_input(&input)?;
    if let Some(returned_state) = returned_state {
        if returned_state != state {
            anyhow::bail!("OAuth state mismatch; please retry login");
        }
    }

    let client = reqwest::Client::new();
    let form_body = reqwest::Url::parse_with_params(
        "https://dummy.local",
        &[
            ("client_id", oauth_client_id.as_str()),
            ("client_secret", oauth_client_secret.as_str()),
            ("code", code.as_str()),
            ("code_verifier", code_verifier.as_str()),
            ("redirect_uri", GEMINI_OAUTH_REDIRECT_URI),
            ("grant_type", "authorization_code"),
        ],
    )?
    .query()
    .unwrap_or_default()
    .to_string();
    let token_resp = client
        .post("https://oauth2.googleapis.com/token")
        .header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(form_body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("token exchange request failed: {}", e))?;

    if !token_resp.status().is_success() {
        let body = token_resp.text().await.unwrap_or_default();
        anyhow::bail!("token exchange failed: {}", body);
    }

    let token = token_resp
        .json::<OAuthTokenResponse>()
        .await
        .map_err(|e| anyhow::anyhow!("invalid token response: {}", e))?;

    let access_token = token.access_token.clone();
    let expiry_date = chrono::Utc::now().timestamp_millis() + token.expires_in * 1000;
    let oauth_creds_path = gemini_home.join(".gemini").join("oauth_creds.json");
    write_json_file_secure(
        &oauth_creds_path,
        &serde_json::json!({
            "access_token": token.access_token,
            "refresh_token": token.refresh_token,
            "token_type": token.token_type.unwrap_or_else(|| "Bearer".to_string()),
            "scope": token.scope.unwrap_or(scope),
            "expiry_date": expiry_date,
        }),
    )?;

    let user_email = match client
        .get("https://www.googleapis.com/oauth2/v2/userinfo")
        .bearer_auth(access_token)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp
            .json::<GoogleUserInfoResponse>()
            .await
            .ok()
            .and_then(|u| u.email),
        _ => None,
    };

    if let Some(email) = &user_email {
        let _ = write_active_google_account(gemini_home, email);
    }

    Ok(user_email)
}

async fn run_gemini_oauth_login(
    account_tag: &str,
    headless: bool,
    no_browser: bool,
    prepare_only: bool,
    run_gemini: bool,
) -> anyhow::Result<Option<(PathBuf, Option<String>)>> {
    let gemini_home = gemini_oauth_home(account_tag)?;
    ensure_gemini_oauth_settings(&gemini_home)?;
    let oauth_file = gemini_home.join(".gemini").join("oauth_creds.json");
    let existing_active = read_active_google_account(&gemini_home)?;
    let has_existing_auth = oauth_file.exists() || existing_active.is_some();

    if headless {
        let manual_cmd = gemini_oauth_manual_login_command(&gemini_home);
        println!("[Gemini OAuth] Headless mode for '{}' (SSH-friendly).", account_tag);

        if has_existing_auth {
            println!("[Gemini OAuth] Existing Gemini OAuth credentials found; finalizing now.");
            return Ok(Some((gemini_home, existing_active)));
        }
        if prepare_only {
            println!("[Gemini OAuth] Login helper command:");
            println!("  {}", manual_cmd);
            println!("[Gemini OAuth] prepare-only mode: oneapi will NOT enter Gemini TUI.");
            println!("[Gemini OAuth] Step 1: Run the helper command above.");
            println!("[Gemini OAuth] Step 2: Complete login (copy URL to another machine if needed), then run /quit.");
            println!("[Gemini OAuth] Step 3: Rerun this same oneapi command to finalize backend registration.");
            return Ok(None);
        }

        if !run_gemini {
            let user_email = run_native_headless_oauth_login(&gemini_home).await?;
            return Ok(Some((gemini_home, user_email)));
        }

        println!("[Gemini OAuth] Login helper command:");
        println!("  {}", manual_cmd);
        println!("[Gemini OAuth] --run-gemini set: handing off terminal to Gemini TUI.");
    }

    let use_no_browser = headless || no_browser;
    if use_no_browser {
        println!("[Gemini OAuth] Gemini session for '{}' is launching now.", account_tag);
    } else {
        println!(
            "[Gemini OAuth] Browser login for '{}' starting. This command will wait (not stuck).",
            account_tag
        );
    }
    println!("[Gemini OAuth] Terminal control is now handed to Gemini temporarily.");
    println!("[Gemini OAuth] If this feels stuck, cancel and rerun with --headless --prepare-only.");
    println!("[Gemini OAuth] After authentication, run /quit in Gemini to return to oneapi.");
    use std::io::Write as _;
    std::io::stdout().flush()?;

    let mut cmd = std::process::Command::new("gemini");
    cmd.env("GEMINI_CLI_HOME", &gemini_home)
        .arg("--prompt-interactive")
        .arg("Complete Google OAuth login, then run /quit.");
    if headless || no_browser {
        cmd.env("NO_BROWSER", "true");
    }

    let status = cmd.status()?;
    if !status.success() {
        println!(
            "[Gemini OAuth] Login was cancelled/interrupted (status: {}). No backend changes were made.",
            status
        );
        println!("[Gemini OAuth] Retry with --headless for manual SSH flow.");
        return Ok(None);
    }

    let active_account = read_active_google_account(&gemini_home)?;
    if !oauth_file.exists() && active_account.is_none() {
        println!(
            "[Gemini OAuth] Login ended but no credentials detected at {}.",
            oauth_file.display()
        );
        println!("[Gemini OAuth] No backend changes were made. Please retry login.");
        return Ok(None);
    }

    Ok(Some((gemini_home, active_account)))
}

#[derive(Parser)]
#[command(name = "oneapi", version = "1.0", about = "OpenAI API but minimalist.")]
struct CliArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the server
    Run {
        #[arg(short, long, default_value = "config.yaml")]
        config: String,
        #[arg(short, long)]
        daemon: bool,
    },
    /// Stop background process
    Stop,
    /// Client service configuration
    Client {
        #[command(subcommand)]
        sub: ClientCommands,
    },
    /// Admin management configuration
    Admin {
        #[command(subcommand)]
        sub: AdminCommands,
    },
    /// Backend model management
    Backend {
        #[command(subcommand)]
        sub: BackendCommands,
    },
    /// Global router & model configuration
    Config {
        #[command(subcommand)]
        sub: ConfigCommands,
    },
    /// Manage fallback chains interactively
    Chain,
    /// System service management
    Service {
        #[command(subcommand)]
        sub: ServiceCommands,
    },
}

#[derive(Subcommand)]
enum ClientCommands {
    /// Set API key
    Key { key: String },
    /// Regenerate random API key
    Regen,
    /// Toggle auth (on/off)
    Auth { state: String },
    /// Set bind address
    Bind { addr: String },
    /// Enable TLS
    Tls { cert: String, key: String },
    /// Disable TLS
    TlsOff,
}

#[derive(Subcommand)]
enum AdminCommands {
    /// Set API key
    Key { key: String },
    /// Regenerate random API key
    Regen,
    /// Toggle admin API (on/off)
    State { state: String },
    /// Set bind address
    Bind { addr: String },
    /// Enable TLS
    Tls { cert: String, key: String },
    /// Disable TLS
    TlsOff,
}

#[derive(Subcommand)]
enum BackendCommands {
    /// Add a new backend
    Add {
        #[arg(long)] backend_type: Option<String>,
        #[arg(long)] model_ids: Option<String>,
        #[arg(long)] account: Option<String>,
        #[arg(long)] key_or_cmd: Option<String>,
    },
    /// Remove a backend
    Remove { account: String },
    /// Edit an existing backend (Interactive)
    Edit { account: String },
    /// Login Gemini CLI OAuth and register as backend
    GeminiOauthLogin {
        account: String,
        /// Comma-separated model IDs, or 'auto' to open interactive picker (TTY) with Gemini/registry candidates
        #[arg(long, default_value = "auto")]
        model_ids: String,
        #[arg(long, default_value_t = 1000000)]
        max_context: u32,
        #[arg(long)]
        budget_limit: Option<f64>,
        /// Headless OAuth mode (print URL, then paste code/redirect URL)
        #[arg(long)]
        headless: bool,
        /// Only print manual headless steps, do not perform OAuth
        #[arg(long, requires = "headless")]
        prepare_only: bool,
        /// Force legacy Gemini TUI handoff in headless mode
        #[arg(long, requires = "headless", conflicts_with = "prepare_only")]
        run_gemini: bool,
        /// Force Gemini not to open local browser
        #[arg(long)]
        no_browser: bool,
    },
    /// List all backends
    List,
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Set routing strategy
    Strategy { mode: Option<String> },
    /// Set model alias/group
    Alias { name: String, #[arg(long, value_delimiter = ',')] accounts: Vec<String> },
    /// Set probe limit
    Probe { seconds: u64 },
    /// Update model registry
    Registry {
        model: String,
        #[arg(long)] rank: u8,
        #[arg(long)] in_price: f64,
        #[arg(long)] out_price: f64,
    },
    /// List configuration
    List,
}

#[derive(Subcommand)]
enum ServiceCommands {
    /// Install systemd service
    Install,
    /// Uninstall systemd service
    Uninstall,
}

struct AppState {
    router: Arc<tokio::sync::RwLock<Arc<Router>>>,
    strategy: Arc<tokio::sync::RwLock<FallbackStrategy>>,
    config_path: String,
    config: Arc<tokio::sync::RwLock<Config>>,
}

impl AppState {
    async fn reload_router(&self) -> anyhow::Result<()> {
        let conf = Config::load(&self.config_path)?;
        let mut backends: Vec<Arc<dyn ModelBackend>> = Vec::new();
        let mut limits = HashMap::new();
        for b in &conf.backends {
            match b {
                BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, budget_limit } => {
                    backends.push(Arc::new(OpenAIBackend::new(models.clone(), account_tag, *max_context, api_key, base_url)));
                    if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                }
                BackendConfig::Cli { models, account_tag, command, max_context, budget_limit, json_path } => {
                    if is_gemini_oauth_cli_command(command) {
                        if let Some(gemini_home) = extract_gemini_oauth_home_from_command(command) {
                            backends.push(Arc::new(GeminiCodeAssistBackend::new(
                                models.clone(),
                                account_tag,
                                *max_context,
                                gemini_home,
                            )));
                        } else {
                            backends.push(Arc::new(CliBackend::new(
                                models.clone(),
                                account_tag,
                                *max_context,
                                command,
                                json_path.clone(),
                            )));
                        }
                    } else {
                        backends.push(Arc::new(CliBackend::new(
                            models.clone(),
                            account_tag,
                            *max_context,
                            command,
                            json_path.clone(),
                        )));
                    }
                    if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                }
                BackendConfig::Gemini { models, account_tag, api_key, max_context, budget_limit } => {
                    backends.push(Arc::new(oneapi::providers::gemini::GeminiBackend::new(models.clone(), account_tag, *max_context, api_key)));
                    if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                }
            }
        }
        let registry = Arc::new(IntelligenceRegistry::new());
        let quota_manager = Arc::new(QuotaManager::new(limits));
        let new_router = Router::new(backends, quota_manager, registry, conf.aliases.clone(), conf.chains.clone());
        *self.router.write().await = Arc::new(new_router);
        *self.config.write().await = conf;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    match args.command {
        Commands::Run { config: config_path, daemon } => {
            let _guard = if daemon {
                use daemonize::Daemonize;
                let log_dir = paths::get_config_dir();
                let file_appender = tracing_appender::rolling::daily(&log_dir, "oneapi.log");
                let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                
                tracing_subscriber::fmt()
                    .with_writer(non_blocking)
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_level(true)
                    .without_time()
                    .init();

                let pid_path = paths::get_path("oneapi.pid");
                Daemonize::new()
                    .pid_file(pid_path)
                    .working_directory(paths::get_config_dir())
                    .start()?;
                Some(guard)
            } else {
                tracing_subscriber::fmt()
                    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_level(true)
                    .without_time()
                    .init();
                None
            };

            let conf = Config::load(&config_path)?;
            
            let mut backends: Vec<Arc<dyn ModelBackend>> = Vec::new();
            let mut limits = HashMap::new();
            for b in &conf.backends {
                match b {
                    BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, budget_limit } => {
                        backends.push(Arc::new(OpenAIBackend::new(models.clone(), account_tag, *max_context, api_key, base_url)));
                        if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                    }
                    BackendConfig::Cli { models, account_tag, command, max_context, budget_limit, json_path } => {
                        if is_gemini_oauth_cli_command(command) {
                            if let Some(gemini_home) = extract_gemini_oauth_home_from_command(command) {
                                backends.push(Arc::new(GeminiCodeAssistBackend::new(
                                    models.clone(),
                                    account_tag,
                                    *max_context,
                                    gemini_home,
                                )));
                            } else {
                                backends.push(Arc::new(CliBackend::new(
                                    models.clone(),
                                    account_tag,
                                    *max_context,
                                    command,
                                    json_path.clone(),
                                )));
                            }
                        } else {
                            backends.push(Arc::new(CliBackend::new(
                                models.clone(),
                                account_tag,
                                *max_context,
                                command,
                                json_path.clone(),
                            )));
                        }
                        if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                    }
                    BackendConfig::Gemini { models, account_tag, api_key, max_context, budget_limit } => {
                        backends.push(Arc::new(oneapi::providers::gemini::GeminiBackend::new(models.clone(), account_tag, *max_context, api_key)));
                        if let Some(l) = budget_limit { for m in models.iter() { limits.insert(format!("{}::{}", account_tag, m), *l); } }
                    }
                }
            }

            let registry = Arc::new(IntelligenceRegistry::new());
            let quota_manager = Arc::new(QuotaManager::new(limits));
            let router = Router::new(backends, quota_manager, registry, conf.aliases.clone(), conf.chains.clone());
            let router_arc: Arc<Router> = Arc::new(router);
            let router_state: Arc<tokio::sync::RwLock<Arc<Router>>> = Arc::new(tokio::sync::RwLock::new(router_arc));
            
            tokio::spawn({
                let rs = router_state.clone();
                let limit = conf.max_probe_interval_secs;
                async move { oneapi::health::HealthManager::new(rs, limit).run().await; }
            });

            let state = Arc::new(AppState { 
                router: router_state, 
                strategy: Arc::new(tokio::sync::RwLock::new(conf.fallback_strategy.clone())),
                config_path: config_path.clone(),
                config: Arc::new(tokio::sync::RwLock::new(conf.clone())),
            });

async fn ping_handler() -> &'static str {
    "pong"
}

            let client_app = AxumRouter::new()
                .route("/ping", get(ping_handler))
                .route("/chat/completions", post(chat_handler))
                .route("/messages", post(anthropic_handler))
                .route("/messages/count_tokens", post(anthropic_count_tokens_handler))
                .route("/models", get(models_handler))
                .route("/models/{id}", get(model_detail_handler))
                .route("/organizations", get(anthropic_orgs_handler))
                .route("/organizations/{id}", get(anthropic_org_detail_handler))
                .route("/organizations/{id}/features", get(anthropic_features_handler))
                .route("/organizations/{id}/billing/status", get(anthropic_billing_handler))
                .route("/organizations/{id}/members/me", get(anthropic_membership_handler))
                .route("/organizations/{id}/members", get(anthropic_members_handler))
                .route("/organizations/{id}/subscription", get(anthropic_subscription_handler))
                .route("/organizations/{id}/api_keys", get(anthropic_org_keys_handler))
                .route("/account", get(anthropic_account_handler))
                .route("/auth/status", get(anthropic_auth_status_handler))
                .route("/users/current", get(anthropic_user_handler))
                .route("/users/current/stats", get(anthropic_user_stats_handler))
                .route("/check_completion", get(anthropic_generic_ok))
                .route("/v1/chat/completions", post(chat_handler))
                .route("/v1/messages", post(anthropic_handler))
                .route("/v1/v1/messages", post(anthropic_handler))
                .route("/v1/messages/count_tokens", post(anthropic_count_tokens_handler))
                .route("/v1/models", get(models_handler))
                .route("/v1/models/{id}", get(model_detail_handler))
                .route("/v1/organizations", get(anthropic_orgs_handler))
                .route("/v1/organizations/{id}", get(anthropic_org_detail_handler))
                .route("/v1/organizations/{id}/features", get(anthropic_features_handler))
                .route("/v1/organizations/{id}/billing/status", get(anthropic_billing_handler))
                .route("/v1/organizations/{id}/members/me", get(anthropic_membership_handler))
                .route("/v1/organizations/{id}/members/user_fake_gpt", get(anthropic_membership_handler))
                .route("/v1/organizations/{id}/members", get(anthropic_members_handler))
                .route("/v1/organizations/{id}/subscription", get(anthropic_subscription_handler))
                .route("/v1/organizations/{id}/api_keys", get(anthropic_org_keys_handler))
                .route("/v1/account", get(anthropic_account_handler))
                .route("/v1/auth/status", get(anthropic_auth_status_handler))
                .route("/v1/users/current", get(anthropic_user_handler))
                .route("/v1/users/current/stats", get(anthropic_user_stats_handler))
                .route("/v1/check_completion", get(anthropic_generic_ok))
                .route("/v1/v1/models", get(models_handler))
                .route("/api/organizations", get(anthropic_orgs_handler))
                .route("/api/organizations/{id}", get(anthropic_org_detail_handler))
                .route("/api/organizations/{id}/features", get(anthropic_features_handler))
                .route("/api/organizations/{id}/billing/status", get(anthropic_billing_handler))
                .route("/api/organization/claude_code_first_token_date", get(anthropic_generic_ok))
                .route("/api/users/current", get(anthropic_user_handler))
                .route("/oauth2/v1/token", post(anthropic_oauth_token_handler))
                .route("/v1/metadata", get(anthropic_generic_ok))
                .fallback(fallback_handler)
                .layer(middleware::from_fn(log_middleware))
                .layer(middleware::from_fn_with_state(state.clone(), client_auth_middleware))
                .with_state(state.clone());
            let admin_app = AxumRouter::new()
                .route("/v1/router/status", get(status_handler))
                .route("/v1/router/usage", get(usage_handler))
                .route("/v1/router/config", get(get_config_handler))
                .route("/v1/router/config/backend", post(add_backend_api))
                .route("/v1/router/config/backend/{id}", axum::routing::delete(remove_backend_api))
                .route("/v1/router/config/strategy", post(set_strategy_api))
                .route("/v1/router/config/alias", post(set_alias_api))
                .layer(middleware::from_fn_with_state(state.clone(), admin_auth_middleware))
                .with_state(state.clone());

            let client_scheme = if conf.client.tls.is_some() { "https" } else { "http" };
            println!("[Startup] Client API binding: {}://{}", client_scheme, conf.client.bind_addr);
            if conf.admin.enabled {
                let admin_scheme = if conf.admin.tls.is_some() { "https" } else { "http" };
                println!("[Startup] Admin API binding: {}://{}", admin_scheme, conf.admin.bind_addr);
            } else {
                println!("[Startup] Admin API binding: disabled");
            }

            let client_addr: std::net::SocketAddr = conf.client.bind_addr.parse()?;
            let client_handle = if let Some(tls) = &conf.client.tls {
                let rustls_config = RustlsConfig::from_pem_file(&tls.cert_path, &tls.key_path).await?;
                let handle = axum_server::Handle::new();
                tokio::spawn(shutdown_watcher(handle.clone()));
                tracing::info!("Client API (HTTPS) listening on {}", conf.client.bind_addr);
                tokio::spawn(async move { axum_server::bind_rustls(client_addr, rustls_config).handle(handle).serve(client_app.into_make_service()).await.unwrap(); })
            } else {
                let handle = axum_server::Handle::new();
                tokio::spawn(shutdown_watcher(handle.clone()));
                tracing::info!("Client API (HTTP) listening on {}", conf.client.bind_addr);
                tokio::spawn(async move { axum_server::bind(client_addr).handle(handle).serve(client_app.into_make_service()).await.unwrap(); })
            };

            if conf.admin.enabled {
                let admin_addr: std::net::SocketAddr = conf.admin.bind_addr.parse()?;
                let handle = axum_server::Handle::new();
                tokio::spawn(shutdown_watcher(handle.clone()));
                if let Some(tls) = &conf.admin.tls {
                    let rustls_config = RustlsConfig::from_pem_file(&tls.cert_path, &tls.key_path).await?;
                    tracing::info!("Admin API (HTTPS) listening on {}", conf.admin.bind_addr);
                    tokio::spawn(async move { axum_server::bind_rustls(admin_addr, rustls_config).handle(handle).serve(admin_app.into_make_service()).await.unwrap(); });
                } else {
                    tracing::info!("Admin API (HTTP) listening on {}", conf.admin.bind_addr);
                    tokio::spawn(async move { axum_server::bind(admin_addr).handle(handle).serve(admin_app.into_make_service()).await.unwrap(); });
                }
            }
            let _ = client_handle.await;
        }
        Commands::Stop => {
            let pid_path = paths::get_path("oneapi.pid");
            if std::path::Path::new(&pid_path).exists() {
                let pid = std::fs::read_to_string(&pid_path)?.trim().parse::<i32>()?;
                unsafe { libc::kill(pid, libc::SIGTERM); }
                let _ = std::fs::remove_file(&pid_path);
                println!("OneAPI stopped.");
            }
        }
        Commands::Client { sub } => {
            let mut conf = Config::load("config.yaml")?;
            match sub {
                ClientCommands::Key { key } => conf.client.api_key = Some(key),
                ClientCommands::Regen => {
                    use rand::RngExt; let mut b = [0u8; 16]; rand::rng().fill(&mut b);
                    let k = hex::encode(b); conf.client.api_key = Some(k.clone());
                    println!("New Client Key: {}", k);
                }
                ClientCommands::Auth { state } => conf.client.auth_enabled = state == "on",
                ClientCommands::Bind { addr } => conf.client.bind_addr = addr,
                ClientCommands::Tls { cert, key } => conf.client.tls = Some(TlsConfig { cert_path: cert, key_path: key }),
                ClientCommands::TlsOff => conf.client.tls = None,
            }
            conf.save("config.yaml")?;
        }
        Commands::Admin { sub } => {
            let mut conf = Config::load("config.yaml")?;
            match sub {
                AdminCommands::Key { key } => conf.admin.api_key = Some(key),
                AdminCommands::Regen => {
                    use rand::RngExt; let mut b = [0u8; 16]; rand::rng().fill(&mut b);
                    let k = hex::encode(b); conf.admin.api_key = Some(k.clone());
                    println!("New Admin Key: {}", k);
                }
                AdminCommands::State { state } => conf.admin.enabled = state == "on",
                AdminCommands::Bind { addr } => conf.admin.bind_addr = addr,
                AdminCommands::Tls { cert, key } => conf.admin.tls = Some(TlsConfig { cert_path: cert, key_path: key }),
                AdminCommands::TlsOff => conf.admin.tls = None,
            }
            conf.save("config.yaml")?;
        }
        Commands::Backend { sub } => {
            let mut conf = Config::load("config.yaml")?;
            match sub {
                BackendCommands::Add { backend_type, model_ids, account, key_or_cmd } => {
                    let new_backend = if let (Some(bt), Some(mids), Some(acc), Some(kc)) = (backend_type, model_ids, account, key_or_cmd) {
                        let models = parse_model_ids(&mids)?;
                        if bt == "openai" { BackendConfig::Openai { models, account_tag: acc, api_key: kc, base_url: "https://api.openai.com/v1".into(), max_context: 128000, budget_limit: None } }
                        else if bt == "gemini" { BackendConfig::Gemini { models, account_tag: acc, api_key: kc, max_context: 1000000, budget_limit: None } }
                        else { BackendConfig::Cli { models, account_tag: acc, command: kc, max_context: 1000000, budget_limit: None, json_path: None } }
                    } else { oneapi::cli::interactive_add() };
                    let new_tag = new_backend.account_tag().to_string();
                    if conf.backends.iter().any(|b| b.account_tag() == new_tag) {
                        anyhow::bail!("account_tag '{}' already exists. account_tag must be unique.", new_tag);
                    }
                    conf.backends.push(new_backend);
                    ensure_unique_account_tags(&conf.backends)?;
                }
                BackendCommands::Remove { account } => {
                    conf.backends.retain(|b| match b {
                        BackendConfig::Openai { account_tag, .. } => account_tag != &account,
                        BackendConfig::Cli { account_tag, .. } => account_tag != &account,
                        BackendConfig::Gemini { account_tag, .. } => account_tag != &account,
                    });
                }
                BackendCommands::Edit { account } => {
                    let idx = conf.backends.iter().position(|b| match b {
                        BackendConfig::Openai { account_tag, .. } => account_tag == &account,
                        BackendConfig::Cli { account_tag, .. } => account_tag == &account,
                        BackendConfig::Gemini { account_tag, .. } => account_tag == &account,
                    });
                    if let Some(i) = idx {
                        let edited = oneapi::cli::interactive_edit(&conf.backends[i]);
                        let edited_tag = edited.account_tag().to_string();
                        if conf
                            .backends
                            .iter()
                            .enumerate()
                            .any(|(j, b)| j != i && b.account_tag() == edited_tag)
                        {
                            anyhow::bail!("account_tag '{}' already exists. account_tag must be unique.", edited_tag);
                        }
                        conf.backends[i] = edited;
                        ensure_unique_account_tags(&conf.backends)?;
                    }
                }
                BackendCommands::GeminiOauthLogin { account, model_ids, max_context, budget_limit, headless, prepare_only, run_gemini, no_browser } => {
                    if let Some(existing_idx) = conf.backends.iter().position(|b| b.account_tag() == account) {
                        let existing_models_snapshot = match &conf.backends[existing_idx] {
                            BackendConfig::Cli { models: existing_models, command, .. }
                                if is_gemini_oauth_cli_command(command) =>
                            {
                                existing_models.clone()
                            }
                            _ => {
                                anyhow::bail!(
                                    "account_tag '{}' already exists and is not a Gemini OAuth CLI backend.",
                                    account
                                );
                            }
                        };
                        let models = resolve_gemini_model_ids(&model_ids, &account, Some(&existing_models_snapshot))?;
                        let gemini_home = gemini_oauth_home(&account)?;
                        if let BackendConfig::Cli {
                            models: existing_models,
                            command,
                            max_context: existing_max_context,
                            budget_limit: existing_budget,
                            ..
                        } = &mut conf.backends[existing_idx]
                        {
                            *existing_models = models.clone();
                            *existing_max_context = max_context;
                            *existing_budget = budget_limit;
                            *command = gemini_oauth_command(&gemini_home);
                            ensure_unique_account_tags(&conf.backends)?;
                            println!(
                                "Updated Gemini OAuth backend '{}'. Models: {}",
                                account,
                                models.join(",")
                            );
                        }
                    } else {
                        if let Some((gemini_home, active_account)) = run_gemini_oauth_login(&account, headless, no_browser, prepare_only, run_gemini).await? {
                            let models = resolve_gemini_model_ids(&model_ids, &account, None)?;
                            let command = gemini_oauth_command(&gemini_home);
                            conf.backends.push(BackendConfig::Cli {
                                models: models.clone(),
                                account_tag: account.clone(),
                                command,
                                max_context,
                                budget_limit,
                                json_path: None,
                            });
                            ensure_unique_account_tags(&conf.backends)?;
                            if let Some(email) = active_account {
                                println!(
                                    "Gemini OAuth login success: {} ({}) models={}",
                                    account,
                                    email,
                                    models.join(",")
                                );
                            } else {
                                println!("Gemini OAuth login success: {} models={}", account, models.join(","));
                            }
                        } else {
                            println!("Gemini OAuth login not finalized yet; backend was not added.");
                        }
                    }
                }
                BackendCommands::List => {
                    println!("{:<15} {:<15} {:<20} {}", "Account", "Type", "Models", "Budget/Context");
                    println!("{}", "-".repeat(70));
                    for b in &conf.backends {
                        match b {
                            BackendConfig::Openai { account_tag, models, max_context, budget_limit, .. } => {
                                println!("{:<15} {:<15} {:<20} {}/{}", account_tag, "OpenAI", models.join(","), budget_limit.unwrap_or(0.0), max_context);
                            }
                            BackendConfig::Gemini { account_tag, models, max_context, budget_limit, .. } => {
                                println!("{:<15} {:<15} {:<20} {}/{}", account_tag, "Gemini", models.join(","), budget_limit.unwrap_or(0.0), max_context);
                            }
                            BackendConfig::Cli { account_tag, models, command, max_context, budget_limit, .. } => {
                                let backend_type = if is_gemini_oauth_cli_command(command) {
                                    "GeminiOAuth"
                                } else {
                                    "CLI"
                                };
                                println!("{:<15} {:<15} {:<20} {}/{}", account_tag, backend_type, models.join(","), budget_limit.unwrap_or(0.0), max_context);
                            }
                        }
                    }
                }
            }
            conf.save("config.yaml")?;
        }
        Commands::Config { sub } => {
            let mut conf = Config::load("config.yaml")?;
            match sub {
                ConfigCommands::Strategy { mode } => {
                    let st = if let Some(m) = mode {
                        match m.to_lowercase().as_str() {
                            "performance" => FallbackStrategy::Performance,
                            "cost" => FallbackStrategy::Cost,
                            "chain" => FallbackStrategy::Chain,
                            _ => {
                                println!("Unknown mode: {}. Available: performance, cost, chain", m);
                                return Ok(());
                            }
                        }
                    } else {
                        let theme = dialoguer::theme::ColorfulTheme::default();
                        let items = vec!["Performance (Lowest Latency)", "Cost (Lowest Price)", "Chain (Group-Tiered Fallback)"];
                        let selection = dialoguer::Select::with_theme(&theme)
                            .with_prompt("Choose Global Routing Strategy")
                            .items(&items)
                            .default(0)
                            .interact()
                            .unwrap();
                        match selection {
                            0 => FallbackStrategy::Performance,
                            1 => FallbackStrategy::Cost,
                            _ => FallbackStrategy::Chain,
                        }
                    };
                    conf.fallback_strategy = st;
                    println!("Strategy set to: {:?}", conf.fallback_strategy);
                }
                ConfigCommands::Alias { name, accounts } => { conf.aliases.insert(name, accounts); },
                ConfigCommands::Probe { seconds } => conf.max_probe_interval_secs = seconds,
                ConfigCommands::Registry { model, rank, in_price, out_price } => {
                    let mut reg = IntelligenceRegistry::new();
                    reg.models.insert(model.clone(), ModelSpec { model_id: model, intelligence_rank: rank, price_per_1k_input: in_price, price_per_1k_output: out_price });
                    reg.save()?;
                }
                ConfigCommands::List => {
                    println!("Strategy: {:?}", conf.fallback_strategy);
                    println!("Probe Interval: {}s", conf.max_probe_interval_secs);
                    println!("\nAliases:");
                    for (name, accs) in &conf.aliases { println!("  {:<15} -> {}", name, accs.join(", ")); }
                    println!("\nChains:");
                    for chain in &conf.chains { println!("  {:<15} -> {:?}", chain.name, chain.groups); }
                }
            }
            conf.save("config.yaml")?;
        }
        Commands::Chain => {
            let conf = Config::load("config.yaml")?;
            let nc = oneapi::cli::interactive_chain_manage(conf);
            nc.save("config.yaml")?;
        }
        Commands::Service { sub } => {
            match sub {
                ServiceCommands::Install => {
                    let exe = std::env::current_exe()?; let dir = paths::get_config_dir(); let user = std::env::var("USER").unwrap_or_else(|_| "root".into());
                    let service = format!("[Unit]\nDescription=OneAPI\n[Service]\nType=simple\nUser={}\nWorkingDirectory={}\nExecStart={} run\nRestart=always\n[Install]\nWantedBy=default.target", user, dir.display(), exe.display());
                    let systemd_user_dir = home::home_dir().unwrap().join(".config/systemd/user");
                    if !systemd_user_dir.exists() { std::fs::create_dir_all(&systemd_user_dir)?; }
                    let service_path = systemd_user_dir.join("oneapi.service");
                    std::fs::write(&service_path, service)?;
                    println!("User-space service generated at: {}", service_path.display());
                    println!("Run:\n  systemctl --user daemon-reload\n  systemctl --user enable oneapi\n  systemctl --user start oneapi");
                }
                ServiceCommands::Uninstall => {
                    println!("Run:\n  systemctl --user stop oneapi\n  systemctl --user disable oneapi\n  rm ~/.config/systemd/user/oneapi.service\n  systemctl --user daemon-reload");
                }
            }
        }
    }
    Ok(())
}

async fn shutdown_watcher(handle: axum_server::Handle) {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutting down OneAPI...");
    handle.graceful_shutdown(Some(std::time::Duration::from_secs(10)));
}

async fn client_auth_middleware(State(state): State<Arc<AppState>>, req: Request<axum::body::Body>, next: Next) -> Result<Response, StatusCode> {
    let conf = state.config.read().await;
    if conf.client.auth_enabled {
        if let Some(key) = &conf.client.api_key {
            let auth_header = req.headers().get("Authorization").and_then(|v| v.to_str().ok());
            let x_api_key = req.headers().get("x-api-key").and_then(|v| v.to_str().ok());
            
            let authorized = (auth_header == Some(&format!("Bearer {}", key))) || (x_api_key == Some(key.as_str()));
            
            if !authorized {
                tracing::debug!("Unauthorized client request");
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    }
    Ok(next.run(req).await)
}

async fn admin_auth_middleware(State(state): State<Arc<AppState>>, req: Request<axum::body::Body>, next: Next) -> Result<Response, StatusCode> {
    let conf = state.config.read().await;
    if let Some(key) = &conf.admin.api_key {
        if req.headers().get("Authorization").and_then(|v| v.to_str().ok()) != Some(&format!("Bearer {}", key)) { return Err(StatusCode::UNAUTHORIZED); }
    }
    Ok(next.run(req).await)
}

async fn log_middleware(req: Request<axum::body::Body>, next: Next) -> Response {
    let debug_enabled = tracing::enabled!(tracing::Level::DEBUG);
    if debug_enabled {
        let method = req.method().clone();
        let uri = req.uri().clone();
        tracing::debug!("--> {} {}", method, uri);
        let (parts, body) = req.into_parts();
        let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap_or_default();
        if let Ok(text) = std::str::from_utf8(&bytes) {
            tracing::debug!("Req Body: {}", text);
        }
        let req = Request::from_parts(parts, axum::body::Body::from(bytes));
        let mut res = next.run(req).await;

        // Add Anthropic specific headers to appease the CLI
        res.headers_mut().insert("anthropic-organization", axum::http::HeaderValue::from_static("org_fake_gpt"));

        tracing::debug!("<-- {}", res.status());
        let (parts, body) = res.into_parts();
        let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap_or_default();
        if let Ok(text) = std::str::from_utf8(&bytes) {
            tracing::debug!("Res Body: {}", text);
        }
        return Response::from_parts(parts, axum::body::Body::from(bytes));
    }

    let mut res = next.run(req).await;

    // Add Anthropic specific headers to appease the CLI
    res.headers_mut().insert("anthropic-organization", axum::http::HeaderValue::from_static("org_fake_gpt"));
    res
}

async fn fallback_handler(method: axum::http::Method, uri: axum::http::Uri, headers: axum::http::HeaderMap, body: axum::body::Bytes) -> impl IntoResponse {
    let body_str = String::from_utf8_lossy(&body);
    tracing::debug!("!!! UNIVERSAL FALLBACK !!!");
    tracing::debug!("Method: {}", method);
    tracing::debug!("URI: {}", uri);
    for (name, value) in headers.iter() {
        tracing::debug!("Header: {}: {:?}", name, value);
    }
    tracing::debug!("Body: {}", body_str);
    
    if uri.path().contains("token") {
        return Json(serde_json::json!({
            "access_token": "ya29.fake-token",
            "expires_in": 3600,
            "token_type": "Bearer"
        })).into_response();
    }

    (StatusCode::OK, "OK").into_response()
}

async fn anthropic_count_tokens_handler(
    Json(_req): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "input_tokens": 100
    }))
}

async fn anthropic_orgs_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "data": [{
            "id": "org_fake_gpt",
            "name": "Fake GPT Organization",
            "created_at": "2024-01-01T00:00:00Z",
            "role": "admin",
            "capabilities": ["can_use_claude_code", "fast_mode", "projects"],
            "tier": "build"
        }]
    }))
}

async fn anthropic_org_detail_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "id": "org_fake_gpt",
        "name": "Fake GPT Organization",
        "created_at": "2024-01-01T00:00:00Z",
        "role": "admin",
        "capabilities": ["can_use_claude_code", "fast_mode", "projects"],
        "tier": "build",
        "is_paid": true
    }))
}

async fn anthropic_features_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "data": ["fast_mode", "projects", "artifacts"]
    }))
}

async fn anthropic_billing_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "active",
        "billing_type": "prepaid",
        "current_period_end": "2099-01-01T00:00:00Z"
    }))
}

async fn anthropic_membership_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "role": "admin",
        "user": {
            "id": "user_fake_gpt",
            "name": "Fake User",
            "email": "fake@example.com"
        }
    }))
}

async fn anthropic_members_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "data": [{
            "role": "admin",
            "user": {
                "id": "user_fake_gpt",
                "name": "Fake User",
                "email": "fake@example.com"
            }
        }]
    }))
}

async fn anthropic_subscription_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "active",
        "plan": "pro"
    }))
}

async fn anthropic_org_keys_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "data": []
    }))
}

async fn anthropic_user_stats_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "total_tokens_used": 0,
        "total_tokens_limit": 1000000
    }))
}

async fn anthropic_generic_ok() -> impl IntoResponse {
    StatusCode::OK
}

async fn anthropic_oauth_token_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "access_token": "ya29.fake-token",
        "expires_in": 3600,
        "token_type": "Bearer"
    }))
}

async fn anthropic_account_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "account_id": "acc_fake_gpt",
        "email": "fake@example.com",
        "name": "Fake User"
    }))
}

async fn anthropic_auth_status_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "logged_in": true,
        "method": "api_key",
        "user_id": "user_fake_gpt",
        "org_id": "org_fake_gpt"
    }))
}

async fn anthropic_user_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "id": "user_fake_gpt",
        "email": "fake@example.com",
        "name": "Fake User"
    }))
}

fn clean_content(value: &mut serde_json::Value) {
    if let Some(arr) = value.as_array_mut() {
        for block in arr {
            if let Some(obj) = block.as_object_mut() {
                obj.remove("cache_control");
            }
        }
    } else if let Some(obj) = value.as_object_mut() {
        obj.remove("cache_control");
    }
}

fn system_to_string(system: serde_json::Value) -> String {
    if let Some(s) = system.as_str() {
        return s.to_string();
    }
    if let Some(arr) = system.as_array() {
        return arr.iter().filter_map(|block| {
            if let Some(s) = block.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = block.as_object() {
                obj.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
            } else {
                None
            }
        }).collect::<Vec<_>>().join("\n\n");
    }
    system.to_string()
}

async fn anthropic_handler(
    State(state): State<Arc<AppState>>,
    Json(anthropic_req): Json<AnthropicMessageRequest>,
) -> impl IntoResponse {
    let mut messages = Vec::new();
    if let Some(system_val) = anthropic_req.system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: serde_json::Value::String(system_to_string(system_val)),
        });
    }
    for msg in anthropic_req.messages {
        let mut content = msg.content.clone();
        clean_content(&mut content);
        messages.push(ChatMessage {
            role: msg.role,
            content,
        });
    }
    let tools = anthropic_req.tools.map(|tools| {
        tools.into_iter().map(|t| OpenAITool {
            r#type: "function".into(),
            function: OpenAIFunction {
                name: t.name,
                description: t.description,
                parameters: t.input_schema,
            },
        }).collect()
    });

    let chat_req = ChatRequest {
        model: anthropic_req.model.clone(),
        messages,
        temperature: anthropic_req.temperature,
        max_tokens: Some(anthropic_req.max_tokens),
        stream: anthropic_req.stream,
        tools,
    };

    let s = state.strategy.read().await.clone();
    let r = state.router.read().await.clone();
    let model_name = anthropic_req.model.clone();

    if chat_req.stream.unwrap_or(false) {
        match r.chat_stream(chat_req, s).await {
            Ok(st) => {
                let stream = st.map(move |res| {
                    match res {
                        Ok(json_str) => {
                            if let Ok(chunk) = serde_json::from_str::<ChatStreamChunk>(&json_str) {
                                if let Some(content) = chunk.choices.get(0).and_then(|c| c.delta.content.as_ref()) {
                                    let data = serde_json::json!({
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": { "type": "text_delta", "text": content }
                                    });
                                    return Ok::<Event, std::convert::Infallible>(Event::default().event("content_block_delta").data(data.to_string()));
                                }
                                if chunk.choices.get(0).map(|c| c.finish_reason.is_some()).unwrap_or(false) {
                                    let data = serde_json::json!({
                                        "type": "message_delta",
                                        "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                                        "usage": { "output_tokens": 1 }
                                    });
                                    return Ok::<Event, std::convert::Infallible>(Event::default().event("message_delta").data(data.to_string()));
                                }
                            }
                            Ok::<Event, std::convert::Infallible>(Event::default().data(""))
                        }
                        Err(e) => {
                            tracing::debug!("Stream error: {:?}", e);
                            Ok::<Event, std::convert::Infallible>(Event::default().data(""))
                        },
                    }
                });
                
                // Prepend message_start and content_block_start
                let start_events = vec![
                    Ok(Event::default().event("message_start").data(serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": uuid::Uuid::new_v4().to_string(),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model_name,
                            "usage": { "input_tokens": 1, "output_tokens": 1 }
                        }
                    }).to_string())),
                    Ok(Event::default().event("content_block_start").data(serde_json::json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": { "type": "text", "text": "" }
                    }).to_string()))
                ];
                
                let end_events = vec![
                    Ok(Event::default().event("content_block_stop").data(serde_json::json!({ "type": "content_block_stop", "index": 0 }).to_string())),
                    Ok(Event::default().event("message_stop").data(serde_json::json!({ "type": "message_stop" }).to_string()))
                ];

                let full_stream = tokio_stream::iter(start_events)
                    .chain(stream)
                    .chain(tokio_stream::iter(end_events));

                Sse::new(full_stream).into_response()
            }
            Err(e) => {
                tracing::debug!("Chat stream error: {:?}", e);
                (StatusCode::SERVICE_UNAVAILABLE, format!("Stream Error: {:?}", e)).into_response()
            }
        }
    } else {
        match r.chat(chat_req, s).await {
            Ok(resp) => {
                let content_text = resp.choices.get(0).and_then(|c| {
                    match &c.message.content {
                        serde_json::Value::String(s) => Some(s.clone()),
                        val => Some(val.to_string()),
                    }
                }).unwrap_or_else(|| "".into());

                let anthropic_resp = AnthropicMessageResponse {
                    id: resp.id,
                    r#type: "message".into(),
                    role: "assistant".into(),
                    content: vec![AnthropicContentBlock {
                        r#type: "text".into(),
                        text: Some(content_text),
                    }],
                    model: resp.model,
                    stop_reason: Some("end_turn".into()),
                    stop_sequence: None,
                    usage: AnthropicUsage {
                        input_tokens: resp.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(1),
                        output_tokens: resp.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(1),
                    },
                };
                Json(anthropic_resp).into_response()
            }
            Err(e) => {
                tracing::debug!("Chat error: {:?}", e);
                (StatusCode::SERVICE_UNAVAILABLE, format!("Chat Error: {:?}", e)).into_response()
            }
        }
    }
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let s = state.strategy.read().await.clone();
    let r = state.router.read().await.clone();
    
    let error_to_json = |e: String| {
        serde_json::json!({
            "error": {
                "message": e,
                "type": "oneapi_error",
                "param": null,
                "code": "service_unavailable"
            }
        })
    };

    if req.stream.unwrap_or(false) {
        match r.chat_stream(req, s).await {
            Ok(st) => Sse::new(st.map(move |res| {
                Ok::<Event, std::convert::Infallible>(match res { 
                    Ok(d) => Event::default().data(d), 
                    Err(e) => {
                        let err_msg = format!("{:?}", e); // Use Debug for full context
                        tracing::error!("Stream chunk yield error: {}", err_msg);
                        tracing::debug!("Stream chunk yield error detail: {}", err_msg);
                        Event::default().data(serde_json::to_string(&error_to_json(err_msg)).unwrap())
                    }
                })
            })).into_response(),
            Err(e) => {
                let err_msg = format!("{:?}", e);
                tracing::error!("Chat stream init failure: {}", err_msg);
                tracing::debug!("Chat stream init failure detail: {}", err_msg);
                (StatusCode::SERVICE_UNAVAILABLE, Json(error_to_json(err_msg))).into_response()
            },
        }
    } else {
        match r.chat(req, s).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => {
                let err_msg = e.to_string();
                tracing::error!("Chat error: {}", err_msg);
                (StatusCode::SERVICE_UNAVAILABLE, Json(error_to_json(err_msg))).into_response()
            },
        }
    }
}

async fn status_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let r = state.router.read().await;
    let mut bks = Vec::new();
    for b in &r.backends { bks.push(serde_json::json!({"name": b.info().models[0], "account": b.info().account_tag, "status": format!("{:?}", b.status().await)})); }
    Json(serde_json::json!( { "strategy": format!("{:?}", state.strategy.read().await), "backends": bks, "aliases": r.aliases }))
}

async fn usage_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let r = state.router.read().await;
    Json(serde_json::json!({"spends": *r.quota_manager.spends.read().await, "limits": r.quota_manager.limits}))
}

async fn get_config_handler(State(state): State<Arc<AppState>>) -> Json<Config> { Json(Config::load(&state.config_path).unwrap()) }

async fn models_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let r = state.router.read().await;
    let mut data = Vec::new();
    
    // Helper to add models
    let mut add_model = |id: &str, display_name: &str| {
        data.push(serde_json::json!({
            "type": "model",
            "id": id,
            "display_name": display_name,
            "created_at": "2024-01-01T00:00:00Z"
        }));
    };

    // Add explicit Sonnet IDs that claudecode might look for
    add_model("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet");
    add_model("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet");
    add_model("claude-3-5-haiku-20241022", "Claude 3.5 Haiku");
    add_model("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet (Latest)");
    add_model("claude-3-5-haiku-latest", "Claude 3.5 Haiku (Latest)");
    add_model("claude-haiku-4-5-20251001", "Claude 4.5 Haiku");
    
    // Add fakegpt
    add_model("fakegpt", "Fake GPT");
    
    // Add priority virtual models
    add_model("cost-priority", "Cost Priority (Cheapest First)");
    add_model("intelligence-priority", "Intelligence Priority (Strongest First)");

    // Add models from backends
    for b in &r.backends {
        for m in &b.info().models {
            add_model(m, m);
        }
    }
    
    // Add models from chains
    for c in &r.chains {
        if c.name != "fakegpt" {
            add_model(&c.name, &c.name);
        }
    }

    // Add models from aliases
    for name in r.aliases.keys() {
        if name != "claude-3-5-sonnet-20241022" && name != "claude-3-5-haiku-20241022" {
            add_model(name, name);
        }
    }

    Json(serde_json::json!({
        "data": data,
        "has_more": false,
        "first_id": data.first().map(|m| m["id"].as_str().unwrap_or("")).unwrap_or(""),
        "last_id": data.last().map(|m| m["id"].as_str().unwrap_or("")).unwrap_or("")
    }))
}

async fn model_detail_handler(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "type": "model",
        "id": id,
        "display_name": id,
        "created_at": "2024-01-01T00:00:00Z"
    }))
}

async fn set_strategy_api(State(state): State<Arc<AppState>>, Json(payload): Json<serde_json::Value>) -> StatusCode {
    if let Some(mode) = payload.get("mode").and_then(|v| v.as_str()) {
        let st = if mode == "performance" { FallbackStrategy::Performance } else { FallbackStrategy::Cost };
        *state.strategy.write().await = st.clone();
        { let mut conf = state.config.write().await; conf.fallback_strategy = st; let _ = conf.save("config.yaml"); }
        return StatusCode::OK;
    }
    StatusCode::BAD_REQUEST
}

async fn add_backend_api(State(state): State<Arc<AppState>>, Json(new_backend): Json<BackendConfig>) -> StatusCode {
    let new_tag = new_backend.account_tag().to_string();
    {
        let mut conf = state.config.write().await;
        if conf.backends.iter().any(|b| b.account_tag() == new_tag) {
            return StatusCode::CONFLICT;
        }
        conf.backends.push(new_backend);
        if conf.save("config.yaml").is_err() {
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    }
    if state.reload_router().await.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR;
    }
    StatusCode::ACCEPTED
}

async fn remove_backend_api(State(state): State<Arc<AppState>>, axum::extract::Path(target_id): axum::extract::Path<String>) -> StatusCode {
    {
        let mut conf = state.config.write().await;
        conf.backends.retain(|b| match b {
            BackendConfig::Openai { account_tag, .. } => account_tag != &target_id,
            BackendConfig::Cli { account_tag, .. } => account_tag != &target_id,
            BackendConfig::Gemini { account_tag, .. } => account_tag != &target_id,
        });
        let _ = conf.save("config.yaml");
    }
    let _ = state.reload_router().await;
    StatusCode::OK
}

async fn set_alias_api(State(state): State<Arc<AppState>>, Json(payload): Json<serde_json::Value>) -> StatusCode {
    if let (Some(n), Some(accs)) = (
        payload.get("name").and_then(|v| v.as_str()),
        payload.get("accounts").and_then(|v| v.as_array())
    ) {
        let list: Vec<String> = accs.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
        {
            let mut conf = state.config.write().await;
            conf.aliases.insert(n.to_string(), list);
            let _ = conf.save("config.yaml");
        }
        let _ = state.reload_router().await;
        return StatusCode::OK;
    }
    StatusCode::BAD_REQUEST
}
