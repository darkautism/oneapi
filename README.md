# OneAPI

> **"The market is heavy with machinery—gears that grind, features that shout, interfaces that clutter the soul. Some are grand but hollow; some are functional but hideous. OneAPI is born from a desire for the essential. It is a clean blade in a world of scrap metal. It is mine—minimalist, autonomous, invisible. I have focused on the pulse, not the skin. The 'face' of this tool—its beauty, its light—is yours to determine. I provide the heartbeat; you provide the eyes."**

OneAPI is a high-performance LLM gateway designed for absolute transparency and autonomy. It unifies multiple providers and local CLI tools into a single, intelligent heartbeat.

---

## Key Capabilities

- **Service & Admin Decoupling**: Completely separate network listeners and security layers for the service API (OpenAI) and the management API.
- **Model Agnostic Transparency**: OneAPI spoofs any requested model, dynamically routing traffic based on internal intelligence/cost strategies. The world sees one; you have many.
- **Group Chain Fallback**: Sophisticated multi-stage routing. Define tiered groups of models; if an entire group fails (429/Quota), the engine seamlessly shifts to the next tier.
- **Active Exponential Probing**: Intelligent self-healing via "ping-pong" requests to locked or rate-limited accounts with dynamic backoff.
- **Zero-Data-Loss Architecture**: Every configuration change and token spend is written atomically and flushed to hardware, ensuring total reliability.
- **CLI Inversion**: Wrap any local tool (`gcloud`, `ollama`, etc.) as a standard cloud API with local token estimation and budget tracking.
- **Dynamic Hot-Reloading**: Full management parity between CLI and API with immediate state propagation. Silence the noise without restarting the engine.

---

## CLI Management Reference

### 1. Client (LLM Service)
| Command | Usage | Description |
| :--- | :--- | :--- |
| **`key`** | `oneapi client key <KEY>` | Manually set the client API key. |
| **`regen`** | `oneapi client regen` | Regenerate a random client API key. |
| **`auth`** | `oneapi client auth <on\|off>` | Toggle authentication requirements. |
| **`bind`** | `oneapi client bind <ADDR>` | Set listen address (e.g., `0.0.0.0:3000`). |
| **`tls`** | `oneapi client tls <CERT> <KEY>` | Enable TLS for the client service. |

### 2. Admin (Management API)
| Command | Usage | Description |
| :--- | :--- | :--- |
| **`key`** | `oneapi admin key <KEY>` | Set the admin API key. |
| **`regen`** | `oneapi admin regen` | Regenerate a random admin API key. |
| **`state`** | `oneapi admin state <on\|off>` | Enable or disable the Admin API listener. |
| **`bind`** | `oneapi admin bind <ADDR>` | Set listen address (e.g., `127.0.0.1:3001`). |
| **`tls`** | `oneapi admin tls <CERT> <KEY>` | Enable TLS for the admin management. |

### 3. Backend & Routing
| Command | Usage | Description |
| :--- | :--- | :--- |
| **`backend add`** | `oneapi backend add` | Add a new provider (Interactive). |
| **`backend gemini-oauth-login`** | `oneapi backend gemini-oauth-login <ACCOUNT> [--model-ids auto\|m1,m2] [--headless] [--prepare-only] [--run-gemini]` | Launch Gemini CLI OAuth login in an isolated `~/.oneapi/gemini-oauth/<ACCOUNT>` profile, then register or update that backend (`--model-ids auto` opens an interactive selector on TTY). |
| **`backend list`**| `oneapi backend list` | Show all configured backends and their models. |
| **`config list`** | `oneapi config list` | View current strategy, aliases, and chains. |
| **`config strategy`** | `oneapi config strategy <mode>` | Set global mode (`performance` \| `cost`). |
| **`chain`** | `oneapi chain` | **Interactive manager** for multi-group fallback tiers. |

- `account_tag` must be unique across all backends. Duplicate tags are rejected on load/save and API/CLI updates.
- `--headless` now runs oneapi-native OAuth (prints auth URL, then asks for code/redirect URL) without entering Gemini TUI.  
- Use `--headless --prepare-only` to only print manual steps; use `--headless --run-gemini` to force old Gemini TUI handoff.
- With `--model-ids auto`, oneapi shows a model picker (when TTY is available), seeded with Gemini Code Assist defaults (`gemini-2.0/2.5/3` family), then merged with Gemini `/models` output (if available).
- Gemini OAuth runtime requests are sent to Cloud Code Assist (`cloudcode-pa.googleapis.com`) with OAuth token + project context, not the Gemini API key endpoint.
- OAuth client credentials can be provided via env (`ONEAPI_GEMINI_OAUTH_CLIENT_ID`, `ONEAPI_GEMINI_OAUTH_CLIENT_SECRET`; aliases: `GEMINI_OAUTH_CLIENT_ID`, `GEMINI_OAUTH_CLIENT_SECRET`).
- If env vars are absent, oneapi can parse credentials from upstream source files (`ONEAPI_GEMINI_OAUTH_SOURCE_FILE` or common local pi-mono/gemini-cli install paths).
- Re-running `backend gemini-oauth-login` on an existing Gemini OAuth account updates its model list/context/budget in-place instead of failing.
- For accounts that only allow Standard tier, set `GOOGLE_CLOUD_PROJECT` (or `GOOGLE_CLOUD_PROJECT_ID`) before using that backend.
- HTTP request/response body logging is DEBUG-level only (`RUST_LOG=debug` to enable verbose protocol logs).
- oneapi does not locally clamp Gemini OAuth `maxOutputTokens`; when `max_tokens` is provided it is forwarded as-is, otherwise no explicit output cap is injected.
- Different models/context windows/tool behaviors can still yield non-intuitive truncation or `MAX_TOKENS`; tune prompts and `max_tokens` per chain/backend.

---

## API Architecture

OneAPI runs two distinct server instances for maximum isolation:

### 1. Client Service API (Default `port 3000`)
- Standard OpenAI `/v1/chat/completions` endpoint.
- Routes based on Global Strategy, Model Aliases, or Group Chains.

### 2. Admin Management API (Default `port 3001`)
- Restricted endpoints for dynamic configuration:
    - `GET /v1/router/status`: Live health and backend states.
    - `GET /v1/router/usage`: Real-time spend tracking per account/model.
    - `GET /v1/router/config`: Full configuration management.
    - `POST/DELETE`: Hot-reload backends, strategies, and aliases.

---

## Deployment

```bash
# Build and install as a user-space systemd service
cargo install --path .
oneapi service install
```

All data persists securely in `~/.oneapi/`.
