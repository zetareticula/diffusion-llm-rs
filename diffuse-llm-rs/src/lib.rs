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

//! A Rust implementation of diffusion models for text generation.

#![allow(dead_code)]
#![feature(f16)]

// Enable f16 feature for half-precision floating point support

use std::f32::consts::PI;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use serde::{
    de::{self, Deserializer, SeqAccess, Visitor},
    ser::{SerializeSeq, Serializer},
    Deserialize, Serialize,
};
use std::fmt;
use tokenizers::Tokenizer;
use async_trait::async_trait;

/// Diffusion model implementation
/// This module provides a diffusion model implementation for training and sampling.
/// It includes a simple diffusion model interface and a diffusion configuration struct.

/// Quantization utilities for the diffusion model
pub mod quantization;

pub mod diffuse_llm {
    use super::*;
    
    /// Configuration for the diffusion model
    /// 
    /// This struct contains all the hyperparameters needed to configure
    /// a diffusion model for training and inference.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DiffusionConfig {
        /// Number of diffusion timesteps
        pub num_timesteps: usize,
        /// Whether to use phase-aware quantization (different precision for prefill/decoding)
        pub use_phase_aware_quant: bool,
        /// Quantization configuration for different phases
        pub quant_config: QuantizationConfig,
        /// Size of the hidden layers
        pub hidden_size: usize,
        /// Number of transformer layers
        pub num_layers: usize,
        /// Number of attention heads in each layer
        pub num_attention_heads: usize,
        /// Size of the vocabulary
        pub vocab_size: usize,
        /// Maximum sequence length
        pub max_sequence_length: usize,
        /// Starting value for the noise schedule
        pub beta_start: f32,
        /// Ending value for the noise schedule
        pub beta_end: f32,
        /// Type of noise schedule to use
        pub beta_schedule: BetaSchedule,
        /// Whether to use KV cache for attention
        pub use_kv_cache: bool,
        /// Number of bits for key/value quantization (0 for no quantization)
        pub kv_quant_bits: u8,
        /// Maximum cache size in bytes
        pub max_cache_size: usize,
    }

    /// Quantization configuration for different phases of generation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QuantizationConfig {
        /// Bit width for prefill phase (higher precision)
        pub prefill_bits: u8,
        /// Bit width for decoding phase (can be lower precision)
        pub decode_bits: u8,
        /// Whether to enable progressive precision reduction during decoding
        pub progressive_precision: bool,
        /// Minimum bits to use in progressive precision mode
        pub min_decode_bits: u8,
    }
    
    impl Default for QuantizationConfig {
        fn default() -> Self {
            Self {
                prefill_bits: 8,
                decode_bits: 4,
                progressive_precision: true,
                min_decode_bits: 2,
            }
        }
    }

    /// Type of noise schedule for the diffusion process
    ///
    /// Different schedules control how the noise is added during the forward process.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum BetaSchedule {
        /// Linear interpolation between beta_start and beta_end
        Linear,
        /// Quadratic interpolation (slower increase at start)
        Quadratic,
        /// Cosine-based schedule as proposed in https://arxiv.org/abs/2102.09672
        Cosine,
    }

    /// Key-Value cache entry for attention layers with phase-aware quantization
    #[derive(Debug, Clone)]
    pub struct KVCacheEntry {
        /// Cached keys for each layer and head
        keys: Array3<f32>,
        /// Cached values for each layer and head
        values: Array3<f32>,
        /// Quantized versions for different phases
        prefill_quantized: Option<super::quantization::QuantizedKVCacheEntry>,
        decode_quantized: Option<super::quantization::QuantizedKVCacheEntry>,
        /// Current quantization bits for each phase
        prefill_quant_bits: u8,
        decode_quant_bits: u8,
        /// Current phase (prefill or decode)
        is_prefill_phase: bool,
        /// Number of tokens currently cached
        seq_len: usize,
    }
    
    impl KVCacheEntry {
        /// Creates a new KV cache entry with phase-aware quantization
        pub fn new(keys: Array3<f32>, values: Array3<f32>, prefill_bits: u8, decode_bits: u8) -> Self {
            let seq_len = keys.shape()[1];
            
            // Initialize with empty quantized entries - they'll be populated on demand
            let prefill_quantized = if prefill_bits > 0 {
                Some(super::quantization::QuantizedKVCacheEntry::new(
                    &keys,
                    &values,
                    prefill_bits
                ))
            } else {
                None
            };
            
            let decode_quantized = if decode_bits > 0 {
                Some(super::quantization::QuantizedKVCacheEntry::new(
                    &keys,
                    &values,
                    decode_bits
                ))
            } else {
                None
            };
            
            Self {
                keys,
                values,
                prefill_quantized,
                decode_quantized,
                prefill_quant_bits: prefill_bits,
                decode_quant_bits: decode_bits,
                is_prefill_phase: true, // Start in prefill phase by default
                seq_len,
            }
        }
        
        /// Gets the keys, dequantizing if necessary based on current phase
        pub fn get_keys(&self) -> Array3<f32> {
            if self.is_prefill_phase {
                match &self.prefill_quantized {
                    Some(quantized) => quantized.dequantize_keys(),
                    None => self.keys.clone(),
                }
            } else {
                match &self.decode_quantized {
                    Some(quantized) => quantized.dequantize_keys(),
                    None => self.keys.clone(),
                }
            }
        }
        
        /// Gets the values, dequantizing if necessary based on current phase
        pub fn get_values(&self) -> Array3<f32> {
            if self.is_prefill_phase {
                match &self.prefill_quantized {
                    Some(quantized) => quantized.dequantize_values(),
                    None => self.values.clone(),
                }
            } else {
                match &self.decode_quantized {
                    Some(quantized) => quantized.dequantize_values(),
                    None => self.values.clone(),
                }
            }
        }
        
        /// Switches between prefill and decode phases
        /// 
        /// This is a convenience method that calls `transition_phase`.
        /// Prefer using `transition_phase` directly for more control.
        pub fn set_phase(&mut self, is_prefill: bool) {
            self.transition_phase(is_prefill);
        }
        
        /// Gets the current quantization bits based on phase
        pub fn get_current_quant_bits(&self) -> u8 {
            if self.is_prefill_phase {
                self.prefill_quant_bits
            } else {
                self.decode_quant_bits
            }
        }
        
        /// Handles the transition between prefill and decode phases
        /// 
        /// This method should be called when switching from prefill to decode phase.
        /// It ensures the cache is properly quantized for the new phase.
        pub fn transition_phase(&mut self, is_prefill: bool) {
            if self.is_prefill_phase == is_prefill {
                return; // No transition needed
            }
            
            self.is_prefill_phase = is_prefill;
            
            // If we're transitioning to decode phase, ensure we have a decode quantized cache
            if !is_prefill && self.decode_quant_bits > 0 && self.decode_quantized.is_none() {
                self.decode_quantized = Some(super::quantization::QuantizedKVCacheEntry::new(
                    &self.keys,
                    &self.values,
                    self.decode_quant_bits
                ));
            }
        }
        
        /// Updates the cache with new keys and values
        pub fn update(&mut self, new_keys: Array3<f32>, new_values: Array3<f32>) {
            self.keys = new_keys;
            self.values = new_values;
            self.seq_len = self.keys.shape()[1];
            
            // Update prefill quantized cache if enabled
            if self.prefill_quant_bits > 0 {
                if let Some(quantized) = &mut self.prefill_quantized {
                    quantized.update(&self.keys, &self.values);
                } else {
                    self.prefill_quantized = Some(super::quantization::QuantizedKVCacheEntry::new(
                        &self.keys,
                        &self.values,
                        self.prefill_quant_bits
                    ));
                }
            }
            
            // Update decode quantized cache if enabled
            if self.decode_quant_bits > 0 {
                if let Some(quantized) = &mut self.decode_quantized {
                    quantized.update(&self.keys, &self.values);
                } else {
                    self.decode_quantized = Some(super::quantization::QuantizedKVCacheEntry::new(
                        &self.keys,
                        &self.values,
                        self.decode_quant_bits
                    ));
                }
            }
        }
        
        /// Gets the memory usage in bytes
        pub fn memory_usage(&self) -> usize {
            let mut total = 0;
            
            // Calculate prefill quantized size if enabled
            if let Some(quantized) = &self.prefill_quantized {
                let keys_size = (quantized.keys.data.len() * quantized.keys.bits as usize + 7) / 8;
                let values_size = (quantized.values.data.len() * quantized.values.bits as usize + 7) / 8;
                total += keys_size + values_size;
            }
            
            // Calculate decode quantized size if enabled
            if let Some(quantized) = &self.decode_quantized {
                let keys_size = (quantized.keys.data.len() * quantized.keys.bits as usize + 7) / 8;
                let values_size = (quantized.values.data.len() * quantized.values.bits as usize + 7) / 8;
                total += keys_size + values_size;
            }
            
            // If no quantization is used, return full precision size
            if total == 0 {
                self.keys.len() * 4 + self.values.len() * 4
            } else {
                total
            }
        }
        
        /// Gets the number of tokens in the cache
        pub fn len(&self) -> usize {
            self.seq_len
        }
        
        /// Checks if the cache is empty
        pub fn is_empty(&self) -> bool {
            self.seq_len == 0
        }
    }

    /// Main diffusion model implementation with KV caching and quantization support
    ///
    /// This struct contains all the components needed for training and inference
    /// with a diffusion-based language model, including support for efficient
    /// KV caching and quantization to reduce memory usage.
    pub struct DiffuseLLM {
        /// Model configuration
        config: DiffusionConfig,
        /// Tokenizer for text processing
        tokenizer: Arc<RwLock<Tokenizer>>,
        /// Normalization component for input data
        normalizer: Normalizer,
        /// Manages model weights and parameters
        weight_manager: WeightManager,
        /// Handles distributed training synchronization
        sync_manager: SyncManager,
        /// Cache for attention masks to avoid recomputation
        attention_masks: DashMap<String, Array2<f32>>,
        /// KV Cache for attention layers with quantization support
        kv_cache: DashMap<String, KVCacheEntry>,
        /// Adaptive quantizer for dynamic quantization
        quantizer: Option<super::quantization::AdaptiveQuantizer>,
        /// Noise schedule parameters (betas)
        betas: Array1<f32>,
        /// Cumulative product of (1 - beta_t)
        alphas: Array1<f32>,
        /// Cumulative product of alpha_t (used in the forward process)
        alpha_bars: Array1<f32>,
        /// Current memory usage of the KV cache in bytes
        cache_memory_usage: std::sync::atomic::AtomicUsize,
    }
    
    /// Handles input normalization for the diffusion model
    /// 
    /// This component is responsible for normalizing input data to have
    /// zero mean and unit variance, which helps with training stability.
    /// It's calculated from the training data statistics.
    pub struct Normalizer {
        /// Mean value used for normalization
        mean: f32,
        /// Standard deviation used for normalization
        std: f32,
    }
    
    /// Manages model weights and their updates
    ///
    /// This struct is responsible for storing and updating the model's weights
    /// in a thread-safe manner using DashMap. It supports distributed training
    /// by managing weight synchronization across multiple processes.
    ///
    /// # Type Parameters
    ///
    /// * `T`: The numeric type used for weights (typically `f32` or `f16`)
    ///
    /// # Fields
    ///
    /// * `weights` - Thread-safe storage for model weights
    /// * `gradients` - Thread-safe storage for weight gradients
    ///
    /// # Methods
    ///
    /// * `new()` - Creates a new WeightManager instance
    /// * `get_weight()` - Retrieves a weight tensor by name
    /// * `get_gradient()` - Retrieves a gradient tensor by name
    /// * `update_gradient()` - Updates a gradient value
    /// * `apply_gradients()` - Applies accumulated gradients to update weights
    ///
    /// # Implementation Details
    ///
    /// Uses `DashMap` for thread-safe, concurrent access to weights and gradients.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ndarray::Array2;
    /// use diffuse_llm_rs::diffuse_llm::WeightManager;
    ///
    /// // Create a new weight manager
    /// let mut weight_manager = WeightManager::new();
    /// 
    pub struct WeightManager {
        /// Model weights stored in a thread-safe map
        weights: DashMap<String, Array2<f16>>,
        /// Gradients for each weight parameter
        gradients: DashMap<String, Array2<f16>>,
    }
    
    /// Handles synchronization across distributed training processes
    ///
    /// This struct manages the coordination of model updates across multiple
    /// processes during distributed training. It ensures that all processes
    /// stay in sync during the training process.
    ///
    /// # Fields
    ///
    /// * `barrier` - Synchronization barrier for process coordination
    /// * `rank` - Unique identifier for the current process
    /// * `world_size` - Total number of processes in the training cluster
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use tokio::sync::Barrier;
    /// use diffuse_llm_rs::diffuse_llm::SyncManager;
    ///
    /// // Create a new sync manager
    /// let barrier = Arc::new(Barrier::new(2)); // For 2 processes
    /// let sync_manager = SyncManager::new(barrier, 0, 2);
    ///
    /// // Wait for all processes to reach this point
    /// sync_manager.barrier.wait();
    /// ```
    pub struct SyncManager {
        /// Points in the model where synchronization should occur
        sync_points: Vec<SyncPoint>,
        /// Barrier for coordinating between processes
        barrier: Arc<tokio::sync::Barrier>,
    }
    
    /// Represents a synchronization point in the model architecture
    ///
    /// Used to coordinate gradient updates across distributed training processes
    /// at specific layers of the model.
    #[derive(Clone, Debug)]
    pub struct SyncPoint {
        /// Identifier for the layer where synchronization occurs
        pub layer_id: usize,
        /// Flag indicating if this layer participates in gradient updates
        pub requires_grad: bool,
    }
    
    /// Configuration for the system architecture
    ///
    /// This struct defines the architectural parameters of the diffusion model,
    /// including the number of layers, hidden size, attention heads, and other
    /// model dimensions.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SystemConfig {
        /// Number of transformer layers in the model
        pub num_layers: usize,
        /// Dimensionality of the hidden states
        pub hidden_size: usize,
        /// Number of attention heads in each layer
        pub num_attention_heads: usize,
        /// Size of the vocabulary
        pub vocab_size: usize,
        /// Maximum sequence length the model can process
        pub max_sequence_length: usize,
    }
    
    /// Provides default configuration values for the diffusion model
    ///
    /// This implementation sets up a configuration similar to GPT-2 medium:
    /// - 1000 diffusion timesteps
    /// - 768 hidden units
    /// - 12 transformer layers
    /// - 12 attention heads
    /// - 50,257 token vocabulary (matching GPT-2)
    /// - 1024 maximum sequence length
    /// - Linear beta schedule from 1e-4 to 0.02
    impl Default for DiffusionConfig {
        fn default() -> Self {
            Self {
                num_timesteps: 1000,
                hidden_size: 768,
                num_layers: 12,
                num_attention_heads: 12,
                vocab_size: 50257, // GPT-2 vocab size
                max_sequence_length: 1024,
                beta_start: 0.0001, // Starting value for the noise schedule
                beta_end: 0.02,     // Ending value for the noise schedule
                beta_schedule: BetaSchedule::Linear, // Default to linear schedule
                use_kv_cache: true, // Enable KV cache by default
                kv_quant_bits: 4,   // 4-bit quantization for KV cache
                max_cache_size: 2 * 1024 * 1024 * 1024, // 2GB default cache size
            }
        }
    }

    /// Creates a beta schedule based on the configuration
    ///
    /// The beta schedule determines how noise is added during the forward diffusion process.
    /// Different schedules can affect model performance and training stability.
    ///
    /// # Arguments
    ///
    /// * `self` - Reference to the diffusion configuration
    ///
    /// # Returns
    ///
    /// * `Array1<f32>` - A vector of beta values for each timestep
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::Array1;
    /// use diffuse_llm_rs::diffuse_llm::{DiffusionConfig, BetaSchedule};
    ///
    /// // Create a configuration with a linear schedule
    /// let config = DiffusionConfig {
    ///     num_timesteps: 1000,
    ///     beta_start: 0.0001,
    ///     beta_end: 0.02,
    ///     beta_schedule: BetaSchedule::Linear,
    ///     ..Default::default()
    /// };
    ///
    /// // Generate the beta schedule
    /// let betas = config.create_beta_schedule();
    /// 
    /// // Verify the schedule properties
    /// assert_eq!(betas.len(), 1000);
    /// assert!(betas[0] >= 0.0001);
    /// assert!(betas[999] <= 0.02);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if `num_timesteps` is zero.
    impl DiffusionConfig {
        /// Creates a beta schedule based on the configuration
        ///
        /// The beta schedule determines how noise is added during the forward diffusion process.
        /// Different schedules can affect model performance and training stability.
        ///
        /// # Returns
        ///
        /// * `Array1<f32>` - A vector of beta values for each timestep
        ///
        /// # Panics
        ///
        /// This function will panic if `num_timesteps` is zero.
        ///
        /// # Schedule Types
        ///
        /// - `Linear`: Linear interpolation between `beta_start` and `beta_end`
        /// - `Quadratic`: Quadratic interpolation (slower increase at start)
        /// - `Cosine`: Cosine-based schedule from the paper "Improved Denoising Diffusion Probabilistic Models"
        pub fn create_beta_schedule(&self) -> Array1<f32> {
            let mut betas = Array1::zeros(self.num_timesteps);
            
            //
            match self.beta_schedule {
                BetaSchedule::Linear => {
                    for t in 0..self.num_timesteps {
                        let beta = self.beta_start + (self.beta_end - self.beta_start) * 
                            (t as f32) / ((self.num_timesteps - 1) as f32);
                        betas[t] = beta;
                    }
                },
                // Quadratic schedule: slower increase in noise at the beginning
                // This can help with early training stability by adding less noise initially
                BetaSchedule::Quadratic => {
                    for t in 0..self.num_timesteps {
                        let t_norm = (t as f32) / ((self.num_timesteps - 1) as f32);
                        // Quadratic interpolation for slower initial increase
                        let beta = self.beta_start + (self.beta_end - self.beta_start) * t_norm * t_norm;
                        betas[t] = beta;
                    }
                },
                // Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
                // This schedule adds noise more gradually and can lead to better sample quality
                BetaSchedule::Cosine => {
                    let s = 0.008;  // Small offset to prevent beta from being too small near t=0
                    for t in 0..self.num_timesteps {
                        let t_norm = (t as f32) / (self.num_timesteps as f32);
                        // Map to cosine space with offset
                        let f_t = ((t_norm + s) / (1.0 + s) * PI / 2.0).cos().powi(2);
                        let f_0 = (s / (1.0 + s) * PI / 2.0).cos().powi(2);
                        // Ensure beta is in valid range
                        let beta = (1.0 - (f_t / f_0)).min(0.999);
                        betas[t] = beta;
                    }
                },
            }
            
            betas
        }
        
        /// Computes the prediction loss for a given model and data
        ///
        /// This method implements the training objective for diffusion models, which is to
        /// predict the noise added during the forward process. The loss is computed as the
        /// mean squared error between the predicted and actual noise.
        ///
        /// # Arguments
        ///
        /// * `model` - The diffusion model implementing the `DiffusionModel` trait
        /// * `x_start` - The input data tensor of shape `[batch_size, feature_dim]`
        /// * `t` - The timesteps for each sample in the batch, shape `[batch_size]`
        /// * `noise` - Optional noise tensor. If not provided, standard normal noise will be used
        ///
        /// # Returns
        ///
        /// * `Array1<f32>` - The loss value for each sample in the batch
        ///
        /// # Panics
        ///
        /// This function will panic if the shapes of the inputs are incompatible.
        pub fn p_losses<M: DiffusionModel>(
            &self,
            model: &M,
            x_start: &Array2<f32>,
            t: &Array1<usize>,  // Changed from i64 to usize
            noise: Option<Array2<f32>>,
        ) -> Array1<f32> {
            // Generate beta, alpha, and alpha_bar schedules
            let betas = self.create_beta_schedule();
            let alphas: Array1<f32> = 1.0 - &betas;
            
            // Compute cumulative product of alphas (alpha_bar)
            let alpha_bars: Array1<f32> = alphas.iter().scan(1.0, |state, &a| {
                *state *= a;
                Some(*state)
            }).collect();
            
            // Sample noise if not provided
            let noise: Array2<f32> = match noise {
                Some(n) => n,
                None => Array2::random_using(x_start.dim(), StandardNormal, &mut rand::thread_rng()),
            };
            
            // Get batch size and ensure shapes match
            let batch_size = x_start.shape()[0];
            assert_eq!(noise.shape(), x_start.shape(), "Noise shape must match input shape");
            assert_eq!(t.len(), batch_size, "Timesteps must match batch size");
            
            // Add noise to the input
            // Reshape alpha_bars[t] to match the shape of x_start for broadcasting 
            let alpha_bars_t = t.mapv(|t| alpha_bars[t]).into_shape((x_start.shape()[0], 1)).unwrap();
            // Reshape 1.0 - alpha_bars[t] to match the shape of x_start for broadcasting
            //
            let noisy = x_start * &alpha_bars_t.mapv(f32::sqrt) 
                + &noise * (1.0 - &alpha_bars_t).mapv(f32::sqrt);
                
            // Predict noise
            let predicted_noise = model.forward(&noisy, t);
            
                }
            },
            // Quadratic schedule: slower increase in noise at the beginning
            // This can help with early training stability by adding less noise initially
            BetaSchedule::Quadratic => {
                for t in 0..self.num_timesteps {
                    let t_norm = (t as f32) / ((self.num_timesteps - 1) as f32);
                    // Quadratic interpolation for slower initial increase
                    let beta = self.beta_start + (self.beta_end - self.beta_start) * t_norm * t_norm;
                    betas[t] = beta;
    pub struct TokenizedVector {
        pub id: String,
        pub tokens: Vec<u32>,
        #[serde(serialize_with = "serialize_array2", deserialize_with = "deserialize_array2")]
        pub embeddings: Array2<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(serialize_with = "serialize_option_array2", deserialize_with = "deserialize_option_array2")]
        pub attention_mask: Option<Array2<f32>>,
        pub offset: usize,
    }
    
    // Serialization/deserialization helpers for Array2<f32>
    fn serialize_array2<S>(array: &Array2<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        
        let mut seq = serializer.serialize_seq(Some(array.nrows() * array.ncols()))?;
        for &val in array.iter() {
            seq.serialize_element(&val)?;
        }
        seq.end()
    }
    
    fn deserialize_array2<'de, D>(deserializer: D) -> Result<Array2<f32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<f32>::deserialize(deserializer)?;
        let len = vec.len();
        // Default to a column vector if we can't determine dimensions
        Array2::from_shape_vec((len, 1), vec).map_err(serde::de::Error::custom)
    }
    
    fn serialize_option_array2<S>(option: &Option<Array2<f32>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match option {
            Some(array) => serialize_array2(array, serializer),
            None => serializer.serialize_none(),
        }
    }
    
    fn deserialize_option_array2<'de, D>(deserializer: D) -> Result<Option<Array2<f32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Some(deserialize_array2(deserializer)?))
    }
    
    impl TokenizedVector {
        pub fn new(id: String, tokens: Vec<u32>, embeddings: Array2<f32>) -> Self {
            Self {
                id,
                tokens,
                embeddings,
                attention_mask: None,
                offset: 0,
            }
        }
    }
    
    impl TokenizedVector {
        pub fn apply_mask(&mut self, mask: Array2<f32>, offset: usize) {
            // Apply mask to embeddings
            for (i, mut emb) in self.embeddings.rows_mut().into_iter().enumerate() {
                if i >= offset && i < offset + mask.nrows() {
                    let mask_row = mask.row(i - offset);
                    for (e, &m) in emb.iter_mut().zip(mask_row.iter()) {
                        *e *= m;
                    }
                }
            }
            self.attention_mask = Some(mask);
            self.offset = offset;
        }
    }
    
    /// Core trait for diffusion models with KV caching support
    /// 
    /// This trait defines the interface that all diffusion models must implement.
    /// It includes methods for both standard forward passes and cached inference.
    #[async_trait]
    pub trait DiffusionModel: Send + Sync {
        /// Performs the forward pass of the diffusion model
        /// 
        /// # Arguments
        /// * `x` - Input tensor of shape [batch_size, feature_dim]
        /// * `t` - Timesteps for each sample in the batch, shape [batch_size]
        /// 
        /// # Returns
        /// * `Array2<f32>` - Predicted noise tensor of same shape as input
        fn forward(&self, x: &Array2<f32>, t: &Array1<usize>) -> Array2<f32>;
        
        /// Performs forward pass with KV caching
        /// 
        /// # Arguments
        /// * `x` - Input tensor of shape [batch_size, feature_dim]
        /// * `t` - Timesteps for each sample in the batch, shape [batch_size]
        /// * `keys` - Cached keys from previous steps, shape [num_layers, seq_len, hidden_size]
        /// * `values` - Cached values from previous steps, shape [num_layers, seq_len, hidden_size]
        /// 
        /// # Returns
        /// * `Result<Array2<f32>>` - Predicted noise tensor and updated KV cache
        fn forward_with_cache(
            &self,
            x: &Array2<f32>,
    /// It's primarily used for testing the diffusion pipeline with a straightforward model.
    #[derive(Debug, Clone)]
    pub struct SimpleDiffusionModel {
        /// Weight matrix of shape [input_dim, output_dim]
        pub weights: Array2<f32>,
        /// Bias vector of shape [output_dim]
        pub bias: Array1<f32>,
    }

    impl SimpleDiffusionModel {
        /// Creates a new SimpleDiffusionModel with the given weights and bias
        /// 
        /// # Arguments
        /// * `input_dim` - Dimensionality of the input features
        /// * `output_dim` - Dimensionality of the output features
        /// 
        /// # Returns
        /// A new instance of `SimpleDiffusionModel` with randomly initialized weights and zero bias
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let weights = Array2::random_using(
                (input_dim, output_dim),
                StandardNormal,
                &mut rand::thread_rng()
            ) * 0.02;  // Small initialization
            
            let bias = Array1::zeros(output_dim);
            
            Self { weights, bias }
        }
    }

    #[async_trait::async_trait]
    impl DiffusionModel for SimpleDiffusionModel {
        fn forward(&self, x: &Array2<f32>, _t: &Array1<usize>) -> Array2<f32> {
            // Simple linear transformation: output = x * weights + bias
            // where x is [batch_size, input_dim]
            // weights is [input_dim, output_dim]
            // bias is [output_dim]
            // result is [batch_size, output_dim]
            x.dot(&self.weights) + &self.bias
        }

        fn forward_with_cache(
            &self,
            x: &Array2<f32>,
            t: &Array1<usize>,
            _keys: &Array3<f32>,
            _values: &Array3<f32>,
        ) -> Result<Array2<f32>, anyhow::Error> {
            // For the simple model, we just ignore the cache
            Ok(self.forward(x, t))
        }

        fn update_kv_cache(
            &self,
            _x: &Array2<f32>,
            _t: &Array1<usize>,
            cache: &KVCacheEntry,
        ) -> Result<(Array3<f32>, Array3<f32>), anyhow::Error> {
            // For the simple model, we just return the existing cache
            // In a real implementation, this would update the cache with new keys/values
            Ok((cache.keys.clone(), cache.values.clone()))
        }
    }
    
    impl DiffuseLLM {
        /// Samples from the diffusion model with phase-aware quantization
        /// 
        /// This method implements the full sampling process with phase-aware quantization,
        /// using higher precision during the prefill phase and potentially lower precision
        /// during the decoding phase.
        ///
        /// # Arguments
        /// * `model` - The diffusion model implementing the `DiffusionModel` trait
        /// * `shape` - Shape of the output tensor (batch_size, sequence_length)
        /// * `num_steps` - Optional number of diffusion steps (defaults to num_timesteps from config)
        /// * `cache_id` - Optional cache ID for KV caching
        /// 
        /// # Returns
        /// A tensor of shape [batch_size, sequence_length] containing the sampled tokens
        pub fn sample<M: DiffusionModel>(
            &self,
            model: &M,
            shape: (usize, usize),
            num_steps: Option<usize>,
            cache_id: Option<&str>,
        ) -> Result<Array2<f32>, anyhow::Error> {
            let (batch_size, seq_len) = shape;
            let num_steps = num_steps.unwrap_or(self.config.num_timesteps);
            
            // Initialize the KV cache if enabled and cache_id is provided
            let mut kv_cache = if self.config.use_kv_cache && cache_id.is_some() {
                let mut cache = self.get_or_init_cache(cache_id.unwrap(), batch_size);
                
                // Set initial phase to prefill
                cache.set_phase(true);
                Some(cache)
            } else {
                None
            };
            
            // Initialize with random noise
            let mut x = Array2::random(
                (batch_size, self.config.hidden_size * seq_len),
                rand::distributions::StandardNormal
            );
            
            // Sample loop
            for t in (0..num_steps).rev() {
                let t_array = Array1::from_elem(batch_size, t);
                
                // Check if we should switch from prefill to decode phase
                if let Some(ref mut cache) = &mut kv_cache {
                    let is_prefill_phase = t > num_steps / 2; // First half is prefill
                    cache.set_phase(is_prefill_phase);
                    
                    // Apply progressive precision reduction if enabled
                    if self.config.use_phase_aware_quant 
                        && self.config.quant_config.progressive_precision 
                        && !is_prefill_phase {
                        
                        // Calculate target bits based on progress through decode phase
                        let progress = (num_steps - t) as f32 / (num_steps / 2) as f32;
                        let target_bits = (self.config.quant_config.decode_bits as f32 * (1.0 - progress) 
                            + self.config.quant_config.min_decode_bits as f32 * progress) as u8;
                        
                        // Update decode quantization if needed
                        if target_bits != cache.decode_quant_bits {
                            cache.decode_quant_bits = target_bits;
                            cache.decode_quantized = None; // Will be recreated on next update
                        }
                    }
                    
                    // Get updated keys and values for this timestep
                    let (new_keys, new_values) = model.update_kv_cache(&x, &t_array, cache)?;
                    
                    // Forward pass with cached keys/values
                    let noise_pred = model.forward_with_cache(
                        &x,
                        &t_array,
                        &cache.get_keys(),
                        &cache.get_values()
                    )?;
                    
                    // Update cache with new keys/values
                    cache.update(new_keys, new_values);
                    
                    // Update the sample
                    x = self.p_sample(model, &x, &t_array, &noise_pred)?;
                } else {
                    // Standard forward pass without cache
                    let noise_pred = model.forward(&x, &t_array);
                    x = self.p_sample(model, &x, &t_array, &noise_pred)?;
                }
            }
            
            // Save the final state to cache if enabled
            if let (Some(cache_id), Some(cache)) = (cache_id, kv_cache) {
                self.update_kv_cache(
                    cache_id,
                    cache.get_keys(),
                    cache.get_values()
                )?;
            }
            
            Ok(x)
                    // Standard forward pass without cache
                    let noise_pred = model.forward(&x, &t_array);
                    x = self.p_sample(model, &x, &t_array, &noise_pred)?;
                }
            }
            
            // Save the final state to cache if enabled
            if let (Some(cache_id), Some(cache)) = (cache_id, kv_cache) {
                self.update_kv_cache(
                    cache_id,
                    cache.get_keys(),
                    cache.get_values()
                )?;
            }
            
            Ok(x)
        }
        
        /// Initializes the KV cache with phase-aware quantization
        fn init_kv_cache(&self, batch_size: usize) -> KVCacheEntry {
            let num_layers = self.config.num_layers;
            let num_heads = self.config.num_attention_heads;
            let head_dim = self.config.hidden_size / num_heads;
            
            // Initialize empty keys and values
            let shape = (num_layers, 0, num_heads * head_dim);
            let keys = Array3::zeros(shape);
            let values = Array3::zeros(shape);
            
            // Get quantization bits for each phase
            let (prefill_bits, decode_bits) = if self.config.use_phase_aware_quant {
                (
                    self.config.quant_config.prefill_bits,
                    self.config.quant_config.decode_bits
                )
            } else {
                // Fallback to uniform quantization if phase-aware is disabled
                (self.config.kv_quant_bits, self.config.kv_quant_bits)
            };
            
            KVCacheEntry::new(keys, values, prefill_bits, decode_bits)
        }
        
        /// Gets or initializes a KV cache entry
        pub fn get_or_init_cache(&self, cache_id: &str, batch_size: usize) -> KVCacheEntry {
            if let Some(entry) = self.kv_cache.get(cache_id) {
                entry.value().clone()
            } else {
                let entry = self.init_kv_cache(batch_size);
                self.kv_cache.insert(cache_id.to_string(), entry.clone());
                entry
            }
        }
        
        /// Updates the KV cache for a given cache ID
        pub fn update_kv_cache(
            &self, 
            cache_id: &str, 
            keys: Array3<f32>,
            values: Array3<f32>,
        ) -> Result<(), anyhow::Error> {
            if !self.config.use_kv_cache {
                return Ok(());
            }
            
            // Check if we need to evict entries to free up memory
            let entry_size = keys.len() * 4 * 2; // 4 bytes per f32, 2 arrays (keys + values)
            let current_usage = self.cache_memory_usage.load(std::sync::atomic::Ordering::Relaxed);
            let new_usage = current_usage + entry_size;
            
            if new_usage > self.config.max_cache_size {
                self.evict_oldest_entries(new_usage - self.config.max_cache_size)?;
            }
            
            // Update the cache entry
            if let Some(mut entry) = self.kv_cache.get_mut(cache_id) {
                // Update memory usage
                let old_size = entry.memory_usage();
                let new_size = keys.len() * 4 * 2; // Approximate size
                
                entry.update(keys, values);
                
                // Update total memory usage
                self.cache_memory_usage.fetch_add(
                    new_size.saturating_sub(old_size),
                    std::sync::atomic::Ordering::Relaxed
                );
            } else {
                // Create new entry if it doesn't exist
                let entry = KVCacheEntry::new(
                    keys,
                    values,
                    self.config.kv_quant_bits
                );
                let entry_size = entry.memory_usage();
                
                self.kv_cache.insert(cache_id.to_string(), entry);
                self.cache_memory_usage.fetch_add(
                    entry_size,
                    std::sync::atomic::Ordering::Relaxed
                );
            }
            
            Ok(())
        }
        
        /// Evicts the oldest cache entries to free up memory
        fn evict_oldest_entries(&self, bytes_to_free: usize) -> Result<(), anyhow::Error> {
            let mut entries: Vec<_> = self.kv_cache
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().memory_usage()))
                .collect();
            
            // Sort by memory usage (descending)
            entries.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
            
            let mut bytes_freed = 0;
            for (key, size) in entries {
                if bytes_freed >= bytes_to_free {
                    break;
                }
                
                if self.kv_cache.remove(&key).is_some() {
                    bytes_freed += size;
                }
            }
            
            // Update memory usage
            self.cache_memory_usage.fetch_sub(
                bytes_freed,
                std::sync::atomic::Ordering::Relaxed
            );
            
            Ok(())
        }
        
        /// Clears the KV cache
        pub fn clear_kv_cache(&self) {
            self.kv_cache.clear();
            self.cache_memory_usage.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        
        /// Gets the current KV cache memory usage in bytes
        pub fn kv_cache_memory_usage(&self) -> usize {
            self.cache_memory_usage.load(std::sync::atomic::Ordering::Relaxed)
        }
        
        /// Adds noise to the input according to the diffusion process
        /// 
        /// This method implements the forward process of the diffusion model,
        /// which gradually adds noise to the input data according to the schedule.
        ///
        /// # Arguments
        /// * `x_start` - The input data tensor of shape [batch_size, feature_dim]
        /// * `t` - The timestep for each sample in the batch, shape [batch_size]
        /// * `noise` - Optional noise tensor. If not provided, standard normal noise will be used
        ///
        /// # Returns
        /// A tuple containing:
        /// - The noisy version of the input
        /// - The noise that was added (or generated)
        pub fn add_noise(
            &self,
            x_start: &Array2<f32>,
            t: &Array1<usize>,
            noise: Option<Array2<f32>>,
        ) -> (Array2<f32>, Array2<f32>) {
            // Generate noise if not provided
            let noise = noise.unwrap_or_else(|| {
                Array2::random(x_start.raw_dim(), rand::distributions::StandardNormal)
            });
            
            // Get the beta and alpha_bar values for the current timestep
            let betas = self.config.create_beta_schedule();
            let alphas: Array1<f32> = 1.0 - &betas;
            
            // Compute cumulative product of alphas (alpha_bar)
            let mut alpha_bars = Array1::ones(alphas.len());
            for i in 1..alphas.len() {
                alpha_bars[i] = alpha_bars[i-1] * alphas[i-1];
            }
            
            // Get the alpha_bar values for the current timestep for each sample in the batch
            let mut alpha_bar_t = Array1::zeros(t.len());
            for (i, &ti) in t.iter().enumerate() {
                let ti = ti.min(alpha_bars.len() - 1); // Clamp to valid range
                alpha_bar_t[i] = alpha_bars[ti];
            }
            
            // Reshape alpha_bar_t for broadcasting
            let alpha_bar_t = alpha_bar_t.insert_axis(ndarray::Axis(1));
            
            // Apply the forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
            let mean = x_start * alpha_bar_t.mapv(f32::sqrt);
            let std = (1.0 - &alpha_bar_t).mapv(f32::sqrt);
            let noisy = mean + &noise * &std;
            
            (noisy, noise)
        }
        
        /// Performs a single sampling step in the reverse diffusion process
        /// 
        /// This method implements the reverse process of the diffusion model,
        /// which gradually denoises the input by predicting and removing noise.
        ///
        /// # Arguments
        /// * `model` - The diffusion model implementing the `DiffusionModel` trait
        /// * `x_t` - The current noisy input tensor of shape [batch_size, feature_dim]
        /// * `t` - The current timestep for each sample in the batch, shape [batch_size]
        /// * `noise_pred` - The predicted noise for the current timestep
        ///
        /// # Returns
        /// A tensor of the same shape as `x_t` with one step of noise removed
        fn p_sample<M: DiffusionModel>(
            &self,
            model: &M,
            x_t: &Array2<f32>,
            t: &Array1<usize>,
            noise_pred: &Array2<f32>,
        ) -> Result<Array2<f32>, anyhow:: Error> {
            // Get the beta, alpha, and alpha_bar values for the current timestep
            let betas = self.config.create_beta_schedule();
            let alphas: Array1<f32> = 1.0 - &betas;
            let mut alpha_bars = Array1::ones(alphas.len());
            for i in 1..alphas.len() {
                alpha_bars[i] = alpha_bars[i-1] * alphas[i-1];
            }
            
            // Get the values for the current timestep for each sample in the batch
            let mut alpha_bar_t = Array1::zeros(t.len());
            let mut alpha_bar_t_prev = Array1::zeros(t.len());
            let mut beta_t = Array1::zeros(t.len());
            let mut alpha_t = Array1::zeros(t.len());
            
            for (i, &ti) in t.iter().enumerate() {
                let ti = ti.min(betas.len() - 1); // Clamp to valid range
                alpha_bar_t[i] = alpha_bars[ti];
                beta_t[i] = betas[ti];
                alpha_t[i] = alphas[ti];
                
                // Handle the case when t = 0 (last step)
                if ti > 0 {
                    alpha_bar_t_prev[i] = alpha_bars[ti - 1];
                } else {
                    alpha_bar_t_prev[i] = 1.0; // alpha_bar_0 = 1.0
                }
            }
            
            // Calculate the mean of the reverse process posterior
            // This is the "denoised" version of x_t
            let mean_coeff1 = (alpha_bar_t_prev.mapv(f32::sqrt) * beta_t.clone()) / 
                (1.0 - &alpha_bar_t);
            let mean_coeff2 = (alphas.mapv(f32::sqrt) * (1.0 - alpha_bar_t_prev)) / 
                (1.0 - &alpha_bar_t);
                
            // Calculate the mean of the posterior distribution
            let mean = mean_coeff1.insert_axis(ndarray::Axis(1)) * x_t + 
                      mean_coeff2.insert_axis(ndarray::Axis(1)) * noise_pred;
            
            // For t > 0, add random noise (Langevin dynamics)
            let noise = if t[0] > 0 {
                // Generate random noise with the same shape as x_t
                Array2::random(x_t.raw_dim(), rand::distributions::StandardNormal)
            } else {
                // No noise for the last step (t=0)
                Array2::zeros(x_t.raw_dim())
            };
            
            // Calculate the variance of the posterior
            let variance = ((1.0 - &alpha_bar_t_prev) / (1.0 - &alpha_bar_t)) * beta_t;
            let std = variance.mapv(|v| v.sqrt());
            
            // Add the noise to the mean
            let x_prev = mean + std.insert_axis(ndarray::Axis(1)) * noise;
            
            Ok(x_prev)
        }
        
        /// Tokenizes a model and returns a vector of tokenized vectors
        /// 
        /// # Arguments
        /// * `model_path` - Path to the model to tokenize
        /// 
        /// # Returns
        /// A vector of `TokenizedVector` containing the tokenized model
        pub async fn tokenize_model(&self, model_path: &str) -> Result<Vec<TokenizedVector>, anyhow::Error> {
            let tokenizer = self.tokenizer.read().await;
            let mut results = Vec::new();
            
            // In a real implementation, you would load and process the model weights here
            // For now, we'll create a dummy tokenized vector
            let encoding = tokenizer.encode(EncodeInput::Single(model_path.to_string()), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            
            // Create a tokenized vector with dummy embeddings
            let embeddings = Array2::zeros((encoding.len(), 768)); // 768 is a common embedding size
            
            results.push(TokenizedVector {
                id: model_path.to_string(),
                tokens: encoding.get_ids().to_vec(),
                embeddings,
                attention_mask: None,
                offset: 0,
            });
            
            Ok(results)
        }
        
        pub fn apply_attention_masks(&self, tokens: &mut [TokenizedVector], offsets: &[usize]) -> Result<(), anyhow::Error> {
            // Implementation would apply attention masks
            // This is a placeholder implementation
            for (token, &offset) in tokens.iter_mut().zip(offsets) {
                if let Some(mask) = self.attention_masks.get(&token.id) {
                    token.apply_mask(mask.value().clone(), offset);
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;
        
        #[tokio::test]
        async fn test_diffusion_process() {
            // Create a default config
            let config = DiffusionConfig::default();
            
            // Initialize the model
            let model = DiffuseLLM::new(config).await.unwrap();
            
            // Test beta schedule
            let betas = model.config.create_beta_schedule();
            assert_eq!(betas.len(), 1000);
            assert!(betas[0] > 0.0 && betas[0] < 1.0);
            
            // Test noise addition
            let x = Array2::zeros((2, 10)); // Batch of 2, sequence length 10
            let t = Array1::from_vec(vec![10, 20]);
            let (noisy, noise) = model.add_noise(&x, &t, None);
            
            // Check shapes
            assert_eq!(noisy.shape(), &[2, 10]);
            assert_eq!(noise.shape(), &[2, 10]);
            
            // Test sampling
            let mut simple_model = SimpleDiffusionModel {
                weights: Array2::eye(10),
                bias: Array1::zeros(10),
            };
            
            let samples = model.sample(&mut simple_model, (2, 10), Some(10));
            assert_eq!(samples.shape(), &[2, 10]);
        }
    }
}
