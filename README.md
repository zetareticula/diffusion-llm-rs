# Diffusion-LLM-RS

A high-performance Rust implementation of diffusion models for language modeling, featuring advanced quantization, memory management, and tokenization.

## Features

- **Diffusion Models for Language**: Efficient implementation of diffusion-based language models
- **Advanced Quantization**: Support for 1, 2, 4, and 8-bit quantization with mixed precision
- **Memory Management**: Optimized HBM and host memory management with smart swapping
- **Tokenization**: Fast BPE tokenizer with configurable vocabulary and special tokens
- **Modular Architecture**: Clean separation of concerns with independent, reusable components

## Project Structure

- `diffuse-llm-rs/`: Core diffusion model implementation
- `fusion-anns/`: Approximate nearest neighbor search with GPU acceleration
- `io-dedup/`: Input/output deduplication for efficient data processing
- `memory_manager/`: Advanced memory management for HBM and host memory
- `ns-router-rs/`: Neuro-symbolic routing for model parallelism
- `prefill-kvquant-rs/`: KV cache quantization for efficient prefilling
- `quantization/`: Advanced quantization strategies (1, 2, 4, 8-bit)
- `salience-engine/`: Salience scoring and cache optimization
- `tokenizer/`: Fast BPE tokenizer implementation

## Getting Started

### Prerequisites

- Rust 1.70 or later
- CUDA 11.8+ (for GPU acceleration)

### Building

```bash
cargo build --release
```

### Running Examples

```bash
cargo run --example hello_workspace
```

## License

MIT
