//! Basic example of using the Diffusion Prefill system

use diffusion_prefill::DiffusionPrefill;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the Diffusion Prefill system with default configuration
    let diffusion = DiffusionPrefill::default()?;
    
    // Example text to prefill
    let text = "The quick brown fox jumps over the lazy dog";
    
    // Prefill the cache with the example text
    println!("Prefilling cache with text: {}", text);
    diffusion.prefill(text)?;
    
    // Generate some text using the prefill cache
    let prompt = "The quick brown";
    let max_length = 50;
    println!("\nGenerating text with prompt: {}", prompt);
    let generated = diffusion.generate(prompt, max_length).await?;
    
    println!("\nGenerated text: {}", generated);
    
    Ok(())
}
