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


use anyhow::Result;
use fusion_anns::fusion_anns::{FusionANNS, SystemConfig};
use prefill_kvquant_rs::kvquant::CompressedVector;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize system configuration
    let config = SystemConfig {
        num_gpu_blocks: 1024,
        block_size: 4096,
        parallel_queries: 4,
    };

    // Create a new FusionANNS instance
    let fusion_anns = FusionANNS::new(&config)?;
    
    // Original shape for our vectors (3-dimensional)
    let original_shape = vec![3];
    
    // Create sample vectors (in a real application, these would be properly quantized)
    // For demonstration, we're using simple u8 values
    let vectors = vec![
        CompressedVector {
            id: "vec1".to_string(),
            data: vec![1, 2, 3],  // Quantized data (u8)
            bits: 8,              // 8 bits per value
            original_shape: original_shape.clone(),
        },
        CompressedVector {
            id: "vec2".to_string(),
            data: vec![4, 5, 6],  // Quantized data (u8)
            bits: 8,              // 8 bits per value
            original_shape: original_shape.clone(),
        },
        CompressedVector {
            id: "vec3".to_string(),
            data: vec![7, 8, 9],  // Quantized data (u8)
            bits: 8,              // 8 bits per value
            original_shape: original_shape.clone(),
        },
    ];

    // Index the vectors
    println!("Indexing vectors...");
    fusion_anns.index_vectors(&vectors).await?;
    
    // Create a query vector (must be f32)
    let query = vec![1.0, 2.0, 2.0];  // Similar to vec1
    
    // Perform a query to find the 2 nearest neighbors
    println!("Querying nearest neighbors...");
    let results = fusion_anns.query(&query, 2).await?;
    
    // Print the results
    println!("\nQuery results (top 2 nearest neighbors):");
    for (id, distance) in results {
        println!("Vector {}: distance = {}", id, distance);
    }
    
    Ok(())
}
