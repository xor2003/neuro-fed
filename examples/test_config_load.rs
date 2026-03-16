use neuro_fed_node::config::NodeConfig;

fn main() {
    println!("Testing configuration loading...");

    // Load configuration from file
    match NodeConfig::load_from_file("config.toml") {
        Ok(config) => {
            println!("Successfully loaded configuration from config.toml");
            println!(
                "Proxy config - OpenAI Base URL: {}",
                config.proxy_config.openai_base_url
            );
            println!(
                "Proxy config - OpenAI API Key present: {}",
                config.proxy_config.openai_api_key.is_some()
            );
            println!("Full config: {:?}", config);
        }
        Err(e) => {
            println!("Error loading config from file: {}", e);
            println!("Loading default configuration...");
            let config = NodeConfig::load_or_default();
            println!(
                "Proxy config - OpenAI Base URL: {}",
                config.proxy_config.openai_base_url
            );
            println!(
                "Proxy config - OpenAI API Key present: {}",
                config.proxy_config.openai_api_key.is_some()
            );
        }
    }
}
