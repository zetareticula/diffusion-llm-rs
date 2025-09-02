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


//! Salience engine for the Diffusion Prefill system

use ndarray::Array1;
use std::collections::HashMap;
use anyhow::Result;

/// Configuration for the SalienceEngine
#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub cache_size: usize,
}

/// Main salience engine
#[derive(Debug)]
pub struct SalienceEngine {
    cache_size: usize,
    cache: HashMap<String, f32>,
}

impl SalienceEngine {
    /// Create a new SalienceEngine with the given configuration
    pub fn new(config: &SystemConfig) -> Result<Self> {
        Ok(Self {
            cache_size: config.cache_size,
            cache: HashMap::with_capacity(config.cache_size),
        })
    }

    /// Calculate salience scores for the given text
    pub fn calculate(&mut self, text: &str) -> HashMap<String, f32> {
        let mut scores = HashMap::new();
        
        // Simple word frequency as a placeholder
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            *scores.entry(word).or_insert(0.0) += 1.0;
        }
        
        // Update cache
        for (word, score) in &scores {
            self.cache.insert(word.clone(), *score);
            
            // Enforce cache size limit
            if self.cache.len() > self.cache_size {
                let key = self.cache.keys().next().cloned().unwrap();
                self.cache.remove(&key);
            }
        }
        
        scores
    }
    
    /// Get the cached score for a word
    pub fn get_cached_score(&self, word: &str) -> Option<f32> {
        self.cache.get(&word.to_lowercase()).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_salience_engine() -> Result<()> {
        let config = SystemConfig { cache_size: 100 };
        let mut engine = SalienceEngine::new(&config)?;
        
        let text = "the quick brown fox jumps over the lazy dog";
        let scores = engine.calculate(text);
        
        assert!(scores.get("the").is_some());
        assert_eq!(*scores.get("the").unwrap(), 2.0);
        
        Ok(())
    }
}
