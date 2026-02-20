use dialoguer::{Input, Select, MultiSelect, theme::ColorfulTheme};
use crate::config::{BackendConfig, Config, GroupChain};

pub fn interactive_add() -> BackendConfig {
    let theme = ColorfulTheme::default();
    let types = vec!["OpenAI-Compatible API", "CLI Command (Stealing CLI)", "Google Gemini API"];
    
    let selection = Select::with_theme(&theme)
        .with_prompt("Choose backend type")
        .items(&types)
        .default(0)
        .interact()
        .unwrap();

    let models_raw: String = Input::with_theme(&theme)
        .with_prompt("Supported Model IDs (comma separated, e.g. gpt-4o,gpt-4o-mini)")
        .interact_text()
        .unwrap();
    let models: Vec<String> = models_raw.split(',').map(|s| s.trim().to_string()).collect();

    let account_tag: String = Input::with_theme(&theme)
        .with_prompt("Account tag (unique identifier for this account)")
        .interact_text()
        .unwrap();

    let max_context: u32 = Input::with_theme(&theme)
        .with_prompt("Max context window")
        .default(128000)
        .interact_text()
        .unwrap();

    let budget: String = Input::with_theme(&theme)
        .with_prompt("Monthly budget limit (USD, empty for unlimited)")
        .allow_empty(true)
        .interact_text()
        .unwrap();
    
    let budget_limit = if budget.is_empty() { None } else { budget.parse::<f64>().ok() };

    match selection {
        0 => {
            let api_key: String = Input::with_theme(&theme).with_prompt("API Key").interact_text().unwrap();
            let base_url: String = Input::with_theme(&theme).with_prompt("Base URL").default("https://api.openai.com/v1".into()).interact_text().unwrap();
            BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, budget_limit }
        },
        1 => {
            let command: String = Input::with_theme(&theme).with_prompt("CLI Command (use {model} and {messages} as placeholders)").default("gcloud ai models generate-content --model={model} --prompt='{messages}'".into()).interact_text().unwrap();
            let json_path: String = Input::with_theme(&theme).with_prompt("JSON pointer to content (e.g. /candidates/0/content, leave empty for raw)").allow_empty(true).interact_text().unwrap();
            let json_path_opt = if json_path.is_empty() { None } else { Some(json_path) };
            BackendConfig::Cli { models, account_tag, command, max_context, budget_limit, json_path: json_path_opt }
        },
        2 => {
            let api_key: String = Input::with_theme(&theme).with_prompt("API Key").interact_text().unwrap();
            BackendConfig::Gemini { models, account_tag, api_key, max_context, budget_limit }
        },
        _ => unreachable!()
    }
}

pub fn interactive_edit(existing: &BackendConfig) -> BackendConfig {
    let theme = ColorfulTheme::default();
    let types = vec!["OpenAI-Compatible API", "CLI Command (Stealing CLI)", "Google Gemini API"];

    let default_sel = match existing {
        BackendConfig::Openai { .. } => 0,
        BackendConfig::Cli { .. } => 1,
        BackendConfig::Gemini { .. } => 2,
    };

    let selection = Select::with_theme(&theme)
        .with_prompt("Choose backend type")
        .items(&types)
        .default(default_sel)
        .interact()
        .unwrap();

    match (selection, existing) {
        (0, BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, budget_limit }) => {
            let models_raw: String = Input::with_theme(&theme)
                .with_prompt("Supported Model IDs (comma separated)")
                .default(models.join(","))
                .interact_text()
                .unwrap();
            let models: Vec<String> = models_raw.split(',').map(|s| s.trim().to_string()).collect();

            let account_tag: String = Input::with_theme(&theme)
                .with_prompt("Account tag (unique identifier for this account)")
                .default(account_tag.clone())
                .interact_text()
                .unwrap();

            let max_context: u32 = Input::with_theme(&theme)
                .with_prompt("Max context window")
                .default(*max_context)
                .interact_text()
                .unwrap();

            let budget: String = Input::with_theme(&theme)
                .with_prompt("Monthly budget limit (USD, empty for unlimited)")
                .allow_empty(true)
                .default(budget_limit.map(|v| v.to_string()).unwrap_or_default())
                .interact_text()
                .unwrap();
            let budget_limit = if budget.is_empty() { None } else { budget.parse::<f64>().ok() };

            let api_key: String = Input::with_theme(&theme).with_prompt("API Key").default(api_key.clone()).interact_text().unwrap();
            let base_url: String = Input::with_theme(&theme).with_prompt("Base URL").default(base_url.clone()).interact_text().unwrap();
            BackendConfig::Openai { models, account_tag, api_key, base_url, max_context, budget_limit }
        }
        (1, BackendConfig::Cli { models, account_tag, command, max_context, budget_limit, json_path }) => {
            let models_raw: String = Input::with_theme(&theme)
                .with_prompt("Supported Model IDs (comma separated)")
                .default(models.join(","))
                .interact_text()
                .unwrap();
            let models: Vec<String> = models_raw.split(',').map(|s| s.trim().to_string()).collect();

            let account_tag: String = Input::with_theme(&theme)
                .with_prompt("Account tag (unique identifier for this account)")
                .default(account_tag.clone())
                .interact_text()
                .unwrap();

            let max_context: u32 = Input::with_theme(&theme)
                .with_prompt("Max context window")
                .default(*max_context)
                .interact_text()
                .unwrap();

            let budget: String = Input::with_theme(&theme)
                .with_prompt("Monthly budget limit (USD, empty for unlimited)")
                .allow_empty(true)
                .default(budget_limit.map(|v| v.to_string()).unwrap_or_default())
                .interact_text()
                .unwrap();
            let budget_limit = if budget.is_empty() { None } else { budget.parse::<f64>().ok() };

            let command: String = Input::with_theme(&theme).with_prompt("CLI Command (use {model} and {messages})").default(command.clone()).interact_text().unwrap();
            let json_path_str: String = Input::with_theme(&theme).with_prompt("JSON pointer to content (empty for raw)").allow_empty(true).default(json_path.clone().unwrap_or_default()).interact_text().unwrap();
            let json_path_opt = if json_path_str.is_empty() { None } else { Some(json_path_str) };
            BackendConfig::Cli { models, account_tag, command, max_context, budget_limit, json_path: json_path_opt }
        }
        (2, BackendConfig::Gemini { models, account_tag, api_key, max_context, budget_limit }) => {
            let models_raw: String = Input::with_theme(&theme)
                .with_prompt("Supported Model IDs (comma separated)")
                .default(models.join(","))
                .interact_text()
                .unwrap();
            let models: Vec<String> = models_raw.split(',').map(|s| s.trim().to_string()).collect();

            let account_tag: String = Input::with_theme(&theme)
                .with_prompt("Account tag (unique identifier for this account)")
                .default(account_tag.clone())
                .interact_text()
                .unwrap();

            let max_context: u32 = Input::with_theme(&theme)
                .with_prompt("Max context window")
                .default(*max_context)
                .interact_text()
                .unwrap();

            let budget: String = Input::with_theme(&theme)
                .with_prompt("Monthly budget limit (USD, empty for unlimited)")
                .allow_empty(true)
                .default(budget_limit.map(|v| v.to_string()).unwrap_or_default())
                .interact_text()
                .unwrap();
            let budget_limit = if budget.is_empty() { None } else { budget.parse::<f64>().ok() };

            let api_key: String = Input::with_theme(&theme).with_prompt("API Key").default(api_key.clone()).interact_text().unwrap();
            BackendConfig::Gemini { models, account_tag, api_key, max_context, budget_limit }
        }
        _ => interactive_add()
    }
}

pub fn interactive_chain_manage(mut config: Config) -> Config {
    let theme = ColorfulTheme::default();
    
    // Create a list of "Account::Model" pairs for all models in all backends
    let mut available_choices = Vec::new();
    for b in &config.backends {
        match b {
            BackendConfig::Openai { account_tag, models, .. } |
            BackendConfig::Cli { account_tag, models, .. } |
            BackendConfig::Gemini { account_tag, models, .. } => {
                for m in models {
                    available_choices.push(format!("{}::{}", account_tag, m));
                }
            }
        }
    }

    loop {
        let options = vec!["List Chains", "Create Chain", "Edit Chain", "Delete Chain", "Save & Exit"];
        let selection = Select::with_theme(&theme)
            .with_prompt("Group Chain Manager")
            .items(&options)
            .default(0)
            .interact()
            .unwrap();

        match selection {
            0 => { // List
                if config.chains.is_empty() { println!("No chains defined."); }
                for c in &config.chains {
                    println!("\nChain: \x1b[1;32m{}\x1b[0m", c.name);
                    for (i, g) in c.groups.iter().enumerate() {
                        println!("  \x1b[1;34mGroup {}\x1b[0m: {}", i + 1, g.join(", "));
                    }
                }
            },
            1 => { // Create
                let name: String = Input::with_theme(&theme).with_prompt("Chain Name").interact_text().unwrap();
                let mut groups = Vec::new();
                manage_groups_interactive(&mut groups, &available_choices, &theme);
                if !groups.is_empty() {
                    config.chains.push(GroupChain { name, groups });
                }
            },
            2 => { // Edit
                let names: Vec<_> = config.chains.iter().map(|c| c.name.as_str()).collect();
                if names.is_empty() { println!("No chains to edit."); continue; }
                let idx = Select::with_theme(&theme).with_prompt("Select chain to edit").items(&names).interact().unwrap();
                manage_groups_interactive(&mut config.chains[idx].groups, &available_choices, &theme);
            }
            3 => { // Delete
                let names: Vec<_> = config.chains.iter().map(|c| c.name.as_str()).collect();
                if names.is_empty() { println!("No chains to delete."); continue; }
                let idx = Select::with_theme(&theme).with_prompt("Select chain to delete").items(&names).interact().unwrap();
                config.chains.remove(idx);
            },
            _ => break,
        }
    }
    config
}

fn manage_groups_interactive(groups: &mut Vec<Vec<String>>, available_choices: &Vec<String>, theme: &ColorfulTheme) {
    loop {
        println!("\n\x1b[1;33mCurrent Hierarchy (Fallback Order):\x1b[0m");
        if groups.is_empty() { println!("  (Empty - At least one group required)"); }
        for (i, g) in groups.iter().enumerate() {
            println!("  {}. \x1b[1;34mGroup\x1b[0m: {}", i + 1, g.join(", "));
        }

        let opts = vec!["Add Group", "Delete Group", "Move Group Up", "Move Group Down", "Edit Group Content", "Done"];
        let action = Select::with_theme(theme).with_prompt("Manage Groups").items(&opts).default(opts.len() - 1).interact().unwrap();

        match action {
            0 => { // Add
                if available_choices.is_empty() { println!("No models available."); continue; }
                let selected = MultiSelect::with_theme(theme).with_prompt("Select models (Space to toggle)").items(available_choices).interact().unwrap();
                let group: Vec<String> = selected.iter().map(|i| available_choices[*i].clone()).collect();
                if !group.is_empty() { groups.push(group); }
            },
            1 => { // Delete
                if groups.is_empty() { continue; }
                let items: Vec<_> = (1..=groups.len()).map(|i| i.to_string()).collect();
                let idx = Select::with_theme(theme).with_prompt("Select group index to delete").items(&items).interact().unwrap();
                groups.remove(idx);
            },
            2 => { // Up
                if groups.len() < 2 { continue; }
                let items: Vec<_> = (1..=groups.len()).map(|i| i.to_string()).collect();
                let idx = Select::with_theme(theme).with_prompt("Move which group up?").items(&items).interact().unwrap();
                if idx > 0 { groups.swap(idx, idx - 1); }
            },
            3 => { // Down
                if groups.len() < 2 { continue; }
                let items: Vec<_> = (1..=groups.len()).map(|i| i.to_string()).collect();
                let idx = Select::with_theme(theme).with_prompt("Move which group down?").items(&items).interact().unwrap();
                if idx < groups.len() - 1 { groups.swap(idx, idx + 1); }
            },
            4 => { // Edit Group Content
                if groups.is_empty() { continue; }
                let items: Vec<_> = (1..=groups.len()).map(|i| i.to_string()).collect();
                let idx = Select::with_theme(theme).with_prompt("Select group index to edit").items(&items).interact().unwrap();
                let defaults: Vec<bool> = available_choices.iter().map(|a| groups[idx].contains(a)).collect();
                let selected = MultiSelect::with_theme(theme).with_prompt("Select models (Space to toggle)").items(available_choices).defaults(&defaults).interact().unwrap();
                let new_group: Vec<String> = selected.iter().map(|i| available_choices[*i].clone()).collect();
                if !new_group.is_empty() { groups[idx] = new_group; }
            }
            _ => {
                if groups.is_empty() { println!("Chain must have at least one group."); continue; }
                break;
            }
        }
    }
}
