pub mod core;
pub mod diffuse_llm;
pub mod kvquant;
pub mod ns_router;
pub mod salience;
pub mod fusion_anns;
pub mod io_dedup;
pub mod memory_manager;
pub mod quantization;
pub mod tokenizer;

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;

/// Main Zeta-Reticula system coordinator
pub struct ZetaReticula {
    /// Diffuse-LLM tokenization engine
    pub diffuse_llm: Arc<diffuse_llm::DiffuseLLM>,
    /// KV-Cache quantization manager
    pub kvquant: Arc<kvquant::PrefillKVQuant>,
    /// Neurosymbolic router
    pub ns_router: Arc<ns_router::NsRouter>,
    /// Salience engine for importance scoring
    pub salience: Arc<salience::SalienceEngine>,
    /// FusionANNS system
    pub fusion_anns: Arc<fusion_anns::FusionANNS>,
    /// I/O deduplication engine
    pub io_dedup: Arc<io_dedup::IODedupEngine>,
    /// System configuration
    pub config: SystemConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SystemConfig {
    pub quantization_bits: Vec<u8>, // 1, 2, 4, 8 bit options
    pub ssd_path: String,
    pub hbm_size_gb: usize,
    pub batch_size: usize,
    pub num_gpu_blocks: usize,
    pub enable_direct_io: bool,
    pub dedup_buffer_size_mb: usize,
}

impl ZetaReticula {
    pub async fn new(config: SystemConfig) -> Result<Self> {
        let diffuse_llm = Arc::new(diffuse_llm::DiffuseLLM::new(&config).await?);
        let kvquant = Arc::new(kvquant::PrefillKVQuant::new(&config)?);
        let ns_router = Arc::new(ns_router::NsRouter::new(&config)?);
        let salience = Arc::new(salience::SalienceEngine::new(&config)?);
        let fusion_anns = Arc::new(fusion_anns::FusionANNS::new(&config)?);
        let io_dedup = Arc::new(io_dedup::IODedupEngine::new(&config)?);

        Ok(Self {
            diffuse_llm,
            kvquant,
            ns_router,
            salience,
            fusion_anns,
            io_dedup,
            config,
        })
    }

    pub async fn process_model(&self, model_path: &str) -> Result<()> {
        // Main processing pipeline
        tracing::info!("Processing model: {}", model_path);
        
        // Step 1: Tokenization with diffuse-llm
        let tokens = self.diffuse_llm.tokenize_model(model_path).await?;
        
        // Step 2: Quantization
        let quantized = self.kvquant.quantize_vectors(&tokens, &self.config.quantization_bits)?;
        
        // Step 3: Store with deduplication
        self.io_dedup.store_vectors(&quantized).await?;
        
        // Step 4: Build navigation graph
        self.ns_router.build_graph(&quantized)?;
        
        // Step 5: Initialize FusionANNS
        self.fusion_anns.index_vectors(&quantized).await?;
        
        Ok(())
    }
}
