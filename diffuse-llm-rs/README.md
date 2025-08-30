# Diffuse-LLM-RS

A high-performance Rust implementation of diffusion models for language modeling.

## Features

- Multiple beta scheduling strategies (Linear, Quadratic, Cosine)
- Flexible model architecture through the `DiffusionModel` trait
- Efficient batched operations using ndarray
- Async/await support for I/O-bound operations
- CUDA support (optional)

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
diffuse-llm-rs = { path = "../diffuse-llm-rs" }
```

## Basic Usage

```rust
use diffuse_llm_rs::diffuse_llm::*;
use ndarray::Array2;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Configure the model
    let config = DiffusionConfig {
        num_timesteps: 1000,
        hidden_size: 128,
        num_layers: 4,
        num_attention_heads: 4,
        vocab_size: 1000,
        max_sequence_length: 32,
        beta_start: 0.0001,
        beta_end: 0.02,
        beta_schedule: BetaSchedule::Cosine,
    };

    // 2. Initialize the diffusion model
    let diffuse_llm = DiffuseLLM::new(config).await?;
    
    // 3. Create and use your model (implementing DiffusionModel trait)
    // ...
    
    Ok(())
}
```

## Running Examples

```bash
# Run the diffusion example
cargo run --example diffusion_example --release
```

## Features

- `cuda`: Enables CUDA acceleration (requires CUDA toolchain)

```toml
[dependencies]
diffuse-llm-rs = { path = "../diffuse-llm-rs", features = ["cuda"] }
```

## License

Licensed under either of:

 * Apache License, Version 2.0
 * MIT license

at your option.
