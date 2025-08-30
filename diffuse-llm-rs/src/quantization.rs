//! Quantization utilities for the diffusion model
//! 
//! This module provides functionality for quantizing and dequantizing tensors
//! to reduce memory usage and improve performance.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use quantiles::ckms::CKMS;
use std::sync::Arc;
use std::sync::Mutex;

/// Quantizes a tensor to the specified number of bits
/// 
/// # Arguments
/// * `data` - Input tensor to quantize
/// * `bits` - Number of bits to use for quantization (1-8)
/// 
/// # Returns
/// A tuple containing:
/// - Quantized data as bytes
/// - Scale factor
/// - Zero point
pub fn quantize_tensor(data: &[f32], bits: u8) -> (Vec<u8>, f32, f32) {
    assert!((1..=8).contains(&bits), "Bits must be between 1 and 8");
    
    let max_val = data
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = data
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));
    
    // Calculate scale and zero point
    let q_min = 0.0;
    let q_max = (1u32 << bits) as f32 - 1.0;
    
    let scale = (max_val - min_val) / (q_max - q_min);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    
    let zero_point = q_min - min_val / scale;
    let zero_point = zero_point.clamp(q_min, q_max).round() as u8;
    
    // Quantize the data
    let quantized: Vec<u8> = data
        .iter()
        .map(|&x| {
            let x = ((x / scale) + zero_point as f32).round() as i32;
            x.clamp(0, (1 << bits) - 1) as u8
        })
        .collect();
    
    (quantized, scale, zero_point as f32)
}

/// Dequantizes a tensor back to f32
/// 
/// # Arguments
/// * `data` - Quantized data
/// * `scale` - Scale factor used for quantization
/// * `zero_point` - Zero point used for quantization
/// 
/// # Returns
/// Dequantized tensor as f32
pub fn dequantize_tensor(data: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter()
        .map(|&x| (x as f32 - zero_point) * scale)
        .collect()
}

/// Quantized tensor storage
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    data: Vec<u8>,
    /// Original shape of the tensor
    shape: Vec<usize>,
    /// Scale factor
    scale: f32,
    /// Zero point
    zero_point: f32,
    /// Number of bits used for quantization
    bits: u8,
}

impl QuantizedTensor {
    /// Creates a new quantized tensor
    pub fn new(data: Vec<u8>, shape: Vec<usize>, scale: f32, zero_point: f32, bits: u8) -> Self {
        Self {
            data,
            shape,
            scale,
            zero_point,
            bits,
        }
    }
    
    /// Dequantizes the tensor back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        dequantize_tensor(&self.data, self.scale, self.zero_point)
    }
    
    /// Gets the compression ratio compared to f32
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * 4; // 4 bytes per f32
        let compressed_size = (self.data.len() * self.bits as usize + 7) / 8; // Round up to nearest byte
        original_size as f32 / compressed_size as f32
    }
}

/// Quantized key-value cache entry
#[derive(Debug, Clone)]
pub struct QuantizedKVCacheEntry {
    /// Quantized keys
    pub keys: QuantizedTensor,
    /// Quantized values
    pub values: QuantizedTensor,
    /// Number of tokens currently cached
    pub seq_len: usize,
}

impl QuantizedKVCacheEntry {
    /// Creates a new quantized KV cache entry
    pub fn new(keys: Array3<f32>, values: Array3<f32>, bits: u8) -> Self {
        let shape = keys.shape().to_vec();
        let (keys_data, keys_scale, keys_zp) = quantize_tensor(
            keys.as_slice().unwrap(),
            bits
        );
        
        let (values_data, values_scale, values_zp) = quantize_tensor(
            values.as_slice().unwrap(),
            bits
        );
        
        Self {
            keys: QuantizedTensor::new(keys_data, shape.clone(), keys_scale, keys_zp, bits),
            values: QuantizedTensor::new(values_data, shape, values_scale, values_zp, bits),
            seq_len: keys.shape()[1],
        }
    }
    
    /// Dequantizes the keys
    pub fn dequantize_keys(&self) -> Array3<f32> {
        let data = self.keys.dequantize();
        Array3::from_shape_vec(
            (self.keys.shape[0], self.keys.shape[1], self.keys.shape[2]),
            data
        ).unwrap()
    }
    
    /// Dequantizes the values
    pub fn dequantize_values(&self) -> Array3<f32> {
        let data = self.values.dequantize();
        Array3::from_shape_vec(
            (self.values.shape[0], self.values.shape[1], self.values.shape[2]),
            data
        ).unwrap()
    }
}

/// Adaptive quantizer that adjusts based on the input distribution
pub struct AdaptiveQuantizer {
    /// Number of bits to use for quantization
    bits: u8,
    /// Target compression ratio
    target_ratio: f32,
    /// Statistics collector for adaptive quantization
    stats: Arc<Mutex<CKMS<f32>>>,
}

impl AdaptiveQuantizer {
    /// Creates a new adaptive quantizer
    pub fn new(bits: u8, target_ratio: f32) -> Self {
        Self {
            bits,
            target_ratio,
            stats: Arc::new(Mutex::new(CKMS::new(0.01))), // 1% error bound
        }
    }
    
    /// Updates the statistics with new data
    pub fn update_stats(&self, data: &[f32]) {
        let mut stats = self.stats.lock().unwrap();
        for &x in data {
            stats.insert(x);
        }
    }
    
    /// Computes the optimal quantization parameters
    pub fn compute_params(&self) -> (f32, f32) {
        let stats = self.stats.lock().unwrap();
        let min = stats.query(0.0).unwrap_or(0.0);
        let max = stats.query(1.0).unwrap_or(1.0);
        
        // Adaptive scaling based on distribution
        let q_max = (1u32 << self.bits) as f32 - 1.0;
        let scale = (max - min) / q_max;
        let zero_point = (-min / scale).round().clamp(0.0, q_max);
        
        (scale, zero_point)
    }
    
    /// Quantizes the input data using adaptive parameters
    pub fn quantize(&self, data: &[f32]) -> (Vec<u8>, f32, f32) {
        let (scale, zero_point) = self.compute_params();
        let q_max = (1u32 << self.bits) as f32 - 1.0;
        
        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| {
                let x = ((x / scale) + zero_point).round() as i32;
                x.clamp(0, q_max as i32) as u8
            })
            .collect();
            
        (quantized, scale, zero_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_quantization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (quantized, scale, zero_point) = quantize_tensor(&data, 4);
        let dequantized = dequantize_tensor(&quantized, scale, zero_point);
        
        // Check that dequantized values are close to original
        for (orig, deq) in data.iter().zip(dequantized) {
            assert!((orig - deq).abs() < 0.1, "Dequantized value too far from original");
        }
    }
    
    #[test]
    fn test_quantized_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let (quantized, scale, zero_point) = quantize_tensor(&data, 4);
        
        let qt = QuantizedTensor::new(quantized, shape, scale, zero_point, 4);
        let dequantized = qt.dequantize();
        
        assert_eq!(dequantized.len(), 4);
        assert!(qt.compression_ratio() > 4.0, "Compression ratio should be > 4x");
    }
    
    #[test]
    fn test_adaptive_quantizer() {
        let quantizer = AdaptiveQuantizer::new(4, 4.0);
        let data = (0..1000).map(|x| x as f32 / 1000.0).collect::<Vec<_>>();
        
        quantizer.update_stats(&data);
        let (scale, zero_point) = quantizer.compute_params();
        
        assert!(scale > 0.0, "Scale should be positive");
        assert!(zero_point >= 0.0, "Zero point should be non-negative");
    }
}
