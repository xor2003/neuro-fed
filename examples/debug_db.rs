use neuro_fed_node::persistence::PCPersistence;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing database connection with create_if_missing=true fix...");
    
    // Clean up any existing files
    let _ = std::fs::remove_file("neurofed.db");
    let _ = std::fs::remove_file("neurofed.db-shm");
    let _ = std::fs::remove_file("neurofed.db-wal");
    
    // Test with fixed implementation using create_if_missing(true)
    println!("\n=== Test: Fixed implementation with create_if_missing(true) ===");
    println!("Checking if file exists before test...");
    if std::path::Path::new("neurofed.db").exists() {
        println!("File exists before test (should not happen after cleanup)");
    } else {
        println!("File does not exist before test (expected)");
    }
    
    match PCPersistence::new("neurofed.db").await {
        Ok(persistence) => {
            println!("✓ Success with create_if_missing(true)!");
            
            // Verify the file was created
            if std::path::Path::new("neurofed.db").exists() {
                println!("✓ Database file was created successfully");
                
                // Test basic operations
                println!("\n=== Testing basic database operations ===");
                
                // Test saving and loading a peer
                use neuro_fed_node::persistence::Peer;
                let test_peer = Peer {
                    pubkey: "test_pubkey_123".to_string(),
                    reputation_score: 0.8,
                    zaps_received: 5,
                    last_seen: chrono::Utc::now().timestamp(),
                };
                
                match persistence.save_peer(&test_peer).await {
                    Ok(_) => println!("✓ Successfully saved peer to database"),
                    Err(e) => println!("✗ Failed to save peer: {}", e),
                }
                
                match persistence.load_peer("test_pubkey_123").await {
                    Ok(Some(loaded_peer)) => {
                        println!("✓ Successfully loaded peer from database");
                        println!("  Pubkey: {}", loaded_peer.pubkey);
                        println!("  Reputation: {}", loaded_peer.reputation_score);
                    }
                    Ok(None) => println!("✗ Peer not found in database"),
                    Err(e) => println!("✗ Failed to load peer: {}", e),
                }
                
                // Test WAL mode is enabled
                println!("\n=== Checking WAL mode ===");
                if std::path::Path::new("neurofed.db-wal").exists() {
                    println!("✓ WAL file exists (WAL mode is active)");
                } else {
                    println!("⚠ WAL file doesn't exist yet (may appear after writes)");
                }
                
            } else {
                println!("✗ Database file was NOT created");
            }
        }
        Err(e) => println!("✗ Failed: {}", e),
    }
    
    println!("\n=== Testing absolute path handling ===");
    let current_dir = std::env::current_dir()?;
    let absolute_path = current_dir.join("test_absolute.db");
    let absolute_path_str = absolute_path.to_string_lossy().to_string();
    
    println!("Testing with absolute path: {}", absolute_path_str);
    match PCPersistence::new(&absolute_path_str).await {
        Ok(_) => {
            println!("✓ Absolute path handled correctly");
            if absolute_path.exists() {
                println!("✓ Absolute path database file created");
            }
        }
        Err(e) => println!("✗ Failed with absolute path: {}", e),
    }
    
    // Clean up
    let _ = std::fs::remove_file(&absolute_path);
    let _ = std::fs::remove_file(format!("{}-shm", absolute_path_str));
    let _ = std::fs::remove_file(format!("{}-wal", absolute_path_str));
    
    println!("\n=== Final file check ===");
    let files = ["neurofed.db", "neurofed.db-shm", "neurofed.db-wal"];
    for file in &files {
        if std::path::Path::new(file).exists() {
            println!("✓ {} exists", file);
        } else {
            println!("✗ {} does not exist", file);
        }
    }
    
    Ok(())
}