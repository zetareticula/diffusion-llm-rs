//! Example usage of the diffusion model for a simple 1D regression task.

// This example demonstrates a simple 1D diffusion model
// It shows how to create a diffusion model and use it for training and sampling

use diffuse_llm_rs::diffuse_llm::*;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

// Simple implementation of the DiffusionModel trait for our example
struct SimpleModel {
    weight: f32,
    bias: f32,
}

impl DiffusionModel for SimpleModel {
    fn forward(&self, x: &Array2<f32>, _t: &Array1<usize>) -> Array2<f32> {
        x * self.weight + self.bias
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the model configuration
    let config = DiffusionConfig {
        num_timesteps: 100,
        hidden_size: 32,
        num_layers: 2,
        num_attention_heads: 2,
        vocab_size: 1,  // Not used in this example
        max_sequence_length: 1,  // Single value prediction
        beta_start: 0.0001,
        beta_end: 0.02,
        beta_schedule: BetaSchedule::Linear,
    };

    // 2. Create a simple diffusion model

    let mut model = SimpleModel {
        weight: 0.5,  // These would be learned in a real scenario
        bias: 0.1,
    };

    // 3. Create a simple dataset (sine wave)
    let num_points = 100;
    let x: Vec<f32> = (0..num_points)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    
    // Convert to 2D array (batch_size=num_points, seq_len=1)
    let x_start = Array2::from_shape_vec((num_points, 1), x).unwrap();

    // 4. Training loop (simplified)
    let num_epochs = 5;
    let batch_size = 10;
    let mut rng = ChaChaRng::seed_from_u64(42);

    for epoch in 0..num_epochs {
        // Sample a random batch
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..num_points))
            .collect();
        
        let batch = Array2::from_shape_fn(
            (batch_size, 1),
            |(i, _)| x_start[[indices[i], 0]]
        );
        
        // Sample timesteps
        let t = Array1::from_shape_fn(batch_size, |_| rng.gen_range(0..config.num_timesteps));
        
        // Compute loss (in a real implementation, you'd update model parameters here)
        let loss = config.p_losses(&model, &batch, &t, None);
        
        println!("Epoch {}, Loss: {:.4}", epoch + 1, loss);
    }

    // 5. Generate samples
    println!("\nGenerating samples...");
    let num_samples = 5;
    let samples = config.sample(
        &mut model,
        (num_samples, 1),  // Generate 5 points
        Some(50)  // Use 50 sampling steps
    );
    
    println!("Generated samples:");
    for i in 0..num_samples {
        println!("  Sample {}: {:.4}", i + 1, samples[[i, 0]]);
    }
    
    Ok(())
}

// Custom model implementation example
struct CustomDiffusionModel {
    // Your model parameters here
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl DiffusionModel for CustomDiffusionModel {
    fn forward(&self, x: &Array2<f32>, _t: &Array1<usize>) -> Array2<f32> {
        // Implement your custom forward pass
        x.dot(&self.weights) + &self.bias
    }
}
