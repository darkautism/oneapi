use base64::Engine as _;
use std::path::PathBuf;

fn env_non_empty(keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Ok(value) = std::env::var(key) {
            if !value.trim().is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn normalize_value(raw: String) -> String {
    let trimmed = raw.trim().to_string();
    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&trimmed) {
        if bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || b"-._@:/.+ ".contains(b))
        {
            if let Ok(decoded) = String::from_utf8(bytes) {
                if !decoded.trim().is_empty() {
                    return decoded;
                }
            }
        }
    }
    trimmed
}

fn extract_between(content: &str, prefix: &str, suffix: char) -> Option<String> {
    let start = content.find(prefix)?;
    let rest = &content[start + prefix.len()..];
    let end = rest.find(suffix)?;
    Some(rest[..end].to_string())
}

fn extract_by_prefixes(content: &str, prefixes: &[(&str, char)]) -> Option<String> {
    for (prefix, quote) in prefixes {
        if let Some(v) = extract_between(content, prefix, *quote) {
            return Some(normalize_value(v));
        }
    }
    None
}

fn parse_source_credentials(content: &str) -> Option<(String, String)> {
    let id_prefixes = [
        ("CLIENT_ID = decode(\"", '"'),
        ("CLIENT_ID=decode(\"", '"'),
        ("CLIENT_ID = atob(\"", '"'),
        ("CLIENT_ID=atob(\"", '"'),
        ("OAUTH_CLIENT_ID = \"", '"'),
        ("OAUTH_CLIENT_ID=\"", '"'),
        ("OAUTH_CLIENT_ID = '", '\''),
        ("OAUTH_CLIENT_ID='", '\''),
    ];
    let secret_prefixes = [
        ("CLIENT_SECRET = decode(\"", '"'),
        ("CLIENT_SECRET=decode(\"", '"'),
        ("CLIENT_SECRET = atob(\"", '"'),
        ("CLIENT_SECRET=atob(\"", '"'),
        ("OAUTH_CLIENT_SECRET = \"", '"'),
        ("OAUTH_CLIENT_SECRET=\"", '"'),
        ("OAUTH_CLIENT_SECRET = '", '\''),
        ("OAUTH_CLIENT_SECRET='", '\''),
    ];

    let client_id = extract_by_prefixes(content, &id_prefixes)?;
    let client_secret = extract_by_prefixes(content, &secret_prefixes)?;
    if client_id.is_empty() || client_secret.is_empty() {
        return None;
    }
    Some((client_id, client_secret))
}

fn source_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = env_non_empty(&[
        "ONEAPI_GEMINI_OAUTH_SOURCE_FILE",
        "GEMINI_OAUTH_SOURCE_FILE",
    ]) {
        paths.push(PathBuf::from(p));
    }

    if let Some(home) = home::home_dir() {
        paths.push(
            home.join(".npm-global/lib/node_modules/@mariozechner/pi-coding-agent/packages/ai/src/utils/oauth/google-gemini-cli.ts"),
        );
        paths.push(
            home.join(".npm-global/lib/node_modules/@mariozechner/pi-coding-agent/node_modules/@mariozechner/pi-ai/dist/utils/oauth/google-gemini-cli.js"),
        );
        paths.push(
            home.join(".npm-global/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js"),
        );
    }
    paths
}

fn discover_from_sources() -> Option<(String, String)> {
    for path in source_candidates() {
        if !path.exists() {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Some(creds) = parse_source_credentials(&content) {
                return Some(creds);
            }
        }
    }
    None
}

pub fn oauth_client_credentials() -> Option<(String, String)> {
    let env_id = env_non_empty(&["ONEAPI_GEMINI_OAUTH_CLIENT_ID", "GEMINI_OAUTH_CLIENT_ID"]);
    let env_secret = env_non_empty(&[
        "ONEAPI_GEMINI_OAUTH_CLIENT_SECRET",
        "GEMINI_OAUTH_CLIENT_SECRET",
    ]);
    match (env_id, env_secret) {
        (Some(id), Some(secret)) => Some((id, secret)),
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => discover_from_sources(),
    }
}
