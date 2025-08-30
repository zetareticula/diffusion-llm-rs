//! Prefill Key-Value Cache with Quantization Support

use std::sync::Arc;
use dashmap::DashMap;
use half::f16;
use ndarray::{Array1, Array2};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use super::Config;

/// Compressed vector representation with quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    pub id: String,
    pub data: Vec<u8>,
    pub bits: u8,
    pub original_shape: Vec<usize>,
    pub quant_scale: f32,
    pub quant_zero_point: f32,
}

/// Key-Value Cache with support for quantized vectors
pub struct KVCache {
    store: DashMap<String, CompressedVector>,
    config: Config,
}

/// Trait for quantization operations
pub trait Quantizer: Send + Sync {
    fn quantize(&self, input: &[f32], bits: u8) -> Vec<u8>;
    fn dequantize(&self, input: &[u8], bits: u8) -> Vec<f32>;
}

/// Implementation of a simple bit quantizer
pub struct BitQuantizer {
    scale: f32,
    zero_point: f32,
}

impl Quantizer for BitQuantizer {
    fn quantize(&self, input: &[f32], bits: u8) -> Vec<u8> {
        let max_val = (1u32 << bits) - 1;
        input.iter().map(|&x| {
            let scaled = ((x - self.zero_point) / self.scale).clamp(0.0, max_val as f32);
            scaled as u8
        }).collect()
    }
    
    fn dequantize(&self, input: &[u8], bits: u8) -> Vec<f32> {
        input.iter().map(|&x| {
            x as f32 * self.scale + self.zero_point
        }).collect()
    }
}

impl KVCache {
    /// Create a new KVCache with the given configuration
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            store: DashMap::new(),
            config: config.clone(),
        })
    }
    
    /// Insert a batch of vectors into the cache
    pub async fn insert_batch(&self, vectors: &[CompressedVector]) -> Result<()> {
        for vector in vectors {
            self.store.insert(vector.id.clone(), vector.clone());
        }
        Ok(())
    }
    
    /// Get a batch of vectors by their IDs
    pub async fn get_batch(&self, ids: &[u32]) -> Result<Vec<Array1<f32>>> {
        let mut result = Vec::with_capacity(ids.len());
        
        for &id in ids {
            if let Some(vector) = self.store.get(&id.to_string()) {
                let decompressed = self.decompress_vector(vector.value())?;
                result.push(decompressed);
            } else {
                // Return zero vector if not found
                result.push(Array1::zeros(self.config.embedding_dim));
            }
        }
        
        Ok(result)
    }
    
    /// Compress a vector using the specified number of bits
    pub fn compress_vector(&self, id: &str, vector: &Array1<f32>, bits: u8) -> CompressedVector {
        let min_val = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (max_val - min_val) / ((1u32 << bits) - 1) as f32;
        let zero_point = min_val;
        
        let quantizer = BitQuantizer { scale, zero_point };
        let quantized = quantizer.quantize(vector.as_slice().unwrap(), bits);
        
        CompressedVector {
            id: id.to_string(),
            data: quantized,
            bits,
            original_shape: vec![vector.len()],
            quant_scale: scale,
            quant_zero_point: zero_point,
        }
    }
    
    /// Decompress a vector back to its original form
    pub fn decompress_vector(&self, vector: &CompressedVector) -> Result<Array1<f32>> {
        let quantizer = BitQuantizer {
            scale: vector.quant_scale,
            zero_point: vector.quant_zero_point,
        };
        
        let decompressed = quantizer.dequantize(&vector.data, vector.bits);
        Ok(Array1::from_vec(decompressed))
    }
    
    /// Get the current cache size in bytes
    pub fn size_bytes(&self) -> usize {
        self.store.iter()
            .map(|entry| entry.data.len() + std::mem::size_of_val(&*entry.id))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_quantization() {
        let config = Config::default();
        let kv_cache = KVCache::new(&config).unwrap();
        
        let vector = array![0.1, 0.5, 1.0, 0.0];
        let compressed = kv_cache.compress_vector("test", &vector, 4);
        let decompressed = kv_cache.decompress_vector(&compressed).unwrap();
        
        // Check if decompressed values are close to original
        for (orig, dec) in vector.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 0.1, "Decompressed value too different from original");
        }
    }
}
