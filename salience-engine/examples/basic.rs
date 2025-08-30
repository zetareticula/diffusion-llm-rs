//! # Zeta Reticula Apache 2.0 License
//! Copyright (c) 2025-present The Zeta Reticula Authors.
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!     http://www.apache.org/licenses/LICENSE-2.0
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.


//! Basic example of using the Salience Engine

use salience_engine::salience::{SalienceEngine, SystemConfig};
use anyhow::Result;
use prefill_kvquant_rs::kvquant::CompressedVector;

fn main() -> Result<()> {
    // Initialize the Salience Engine with a cache size of 1000
    let config = SystemConfig { cache_size: 1000 };
    let engine = SalienceEngine::new(&config)?;
    
    // Create a sample compressed vector (in a real scenario, this would come from your model)
    let vector = CompressedVector {
        id: "test_vector".to_string(),
        data: vec![1, 2, 3, 4],  // Example quantized data
        bits: 8,  // Using 8 bits per value
        original_shape: vec![4],  // Original shape of the vector
    };
    
    // Update the engine with the vector
    engine.update(&vector);
    
    // Get the importance score for the vector
    let score = engine.score(&vector);
    println!("Importance score: {:.4}", score);
    
    // Optimize a set of vectors (in this case, just one)
    let optimized = engine.optimize(&[vector.clone()])?;
    println!("Optimized to {} vectors", optimized.len());
    
    // Make a prediction
    let prediction = engine.predict(&vector);
    println!("Prediction: {:.4}", prediction);
    
    Ok(())
}
