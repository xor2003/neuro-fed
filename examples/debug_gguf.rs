//! Debug GGUF file structure
use candle_core::{Device, Tensor};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf";
    
    let mut file = File::open(model_path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    println!("GGUF Metadata:");
    for (key, value) in &content.metadata {
        println!("  {}: {:?}", key, value);
    }
    
    println!("\nAvailable tensors (first 30):");
    let device = Device::Cpu;
    for (i, (name, _)) in content.tensor_infos.iter().enumerate().take(30) {
        println!("  {}. {}", i, name);
    }
    
    // Try to load specific tensors
    println!("\nTrying to load key tensors:");
    
    let tensor_names = vec![
        "output.weight",
        "token_embd.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
    ];
    
    for name in tensor_names {
        match content.tensor(&mut file, name, &device) {
            Ok(tensor) => {
                let shape = tensor.shape();
                println!("  {}: Shape {:?}", name, shape);
            }
            Err(e) => {
                println!("  {}: Not found ({})", name, e);
            }
        }
    }
    
    Ok(())
}