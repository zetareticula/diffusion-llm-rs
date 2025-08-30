pub mod salience {
    use std::sync::Arc;
    use dashmap::DashMap;
    use ndarray::{Array1, Array2};
    use anyhow::Result;
    use prefill_kvquant_rs::kvquant;
    
    #[derive(Debug, Clone)]
    pub struct SystemConfig {
        pub cache_size: usize,
    }
    
    pub struct SalienceEngine {
        scorer: ImportanceScorer,
        cache_optimizer: CacheOptimizer,
        prefill_predictor: PrefillPredictor,
    }
    
    pub struct ImportanceScorer {
        weights: DashMap<String, f32>,
        access_patterns: DashMap<String, AccessPattern>,
    }
    
    #[derive(Debug, Clone)]
    pub struct AccessPattern {
        pub frequency: usize,
        pub recency: std::time::Instant,
        pub importance_score: f32,
    }
    
    pub struct CacheOptimizer {
        cache_size: usize,
        eviction_policy: String,
    }
    
    pub struct PrefillPredictor {
        model: PredictionModel,
        accuracy: f32,
    }
    
    pub struct PredictionModel {
        weights: Array2<f32>,
        bias: Array1<f32>,
    }
    
    impl SalienceEngine {
        pub fn new(config: &SystemConfig) -> Result<Self> {
            Ok(Self {
                scorer: ImportanceScorer {
                    weights: DashMap::new(),
                    access_patterns: DashMap::new(),
                },
                cache_optimizer: CacheOptimizer {
                    cache_size: config.cache_size,
                    eviction_policy: "lru".to_string(),
                },
                prefill_predictor: PrefillPredictor {
                    model: PredictionModel {
                        weights: Array2::zeros((1, 1)),
                        bias: Array1::zeros(1),
                    },
                    accuracy: 0.0,
                },
            })
        }
    }

    impl ImportanceScorer {
        pub fn score(&self, vector: &kvquant::CompressedVector) -> f32 {
            let mut score = 0.0;
            
            // Simple scoring based on access patterns
            if let Some(pattern) = self.access_patterns.get(&vector.id) {
                score += pattern.importance_score;
            }
            
            score
        }

        pub fn update(&self, vector: &kvquant::CompressedVector) {
            // Update access patterns
            let mut entry = self.access_patterns.entry(vector.id.clone()).or_insert_with(|| {
                AccessPattern::new(0, std::time::Instant::now(), 0.0)
            });
            
            entry.frequency += 1;
            entry.recency = std::time::Instant::now();
            entry.importance_score = self.score(vector);
        }
    }

    impl CacheOptimizer {
        pub fn optimize(&self, vectors: &[kvquant::CompressedVector]) -> Result<Vec<kvquant::CompressedVector>> {
            // Simple LRU implementation for now
            let mut sorted = vectors.to_vec();
            sorted.sort_by_key(|v| v.id.clone());
            sorted.truncate(self.cache_size);
            Ok(sorted)
        }
    }

    impl PrefillPredictor {
        pub fn predict(&self, vector: &kvquant::CompressedVector) -> f32 {
            // Simple dot product for now
            self.model.predict(vector)
        }
    }

    impl PredictionModel {
        pub fn new(weights: Array2<f32>, bias: Array1<f32>) -> Self {
            Self { weights, bias }
        }
        
        pub fn predict(&self, vector: &kvquant::CompressedVector) -> f32 {
            // Dummy implementation - would perform actual prediction
            0.0
        }
    }

    impl AccessPattern {
        pub fn new(frequency: usize, recency: std::time::Instant, importance_score: f32) -> Self {
            Self {
                frequency,
                recency,
                importance_score,
            }
        }
    }

    impl SalienceEngine {
        pub fn update(&self, vector: &kvquant::CompressedVector) {
            self.scorer.update(vector);
        }
        
        pub fn optimize(&self, vectors: &[kvquant::CompressedVector]) -> Result<Vec<kvquant::CompressedVector>> {
            self.cache_optimizer.optimize(vectors)
        }
        
        pub fn predict(&self, vector: &kvquant::CompressedVector) -> f32 {
            self.prefill_predictor.predict(vector)
        }
        
        pub fn score(&self, vector: &kvquant::CompressedVector) -> f32 {
            // Combine scores from different components
            let base_score = self.scorer.score(vector);
            let prediction = self.prefill_predictor.predict(vector);
            
            // Simple weighted average for now
            base_score * 0.7 + prediction * 0.3
        }
    }
    
}