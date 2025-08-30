pub mod io_dedup {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::unix::io::AsRawFd;
    use memmap2::MmapOptions;
    use prefill_kvquant_rs::kvquant::CompressedVector;
    
    pub struct IODedupEngine {
        ssd_storage: SSDStorage,
        dedup_buffer: Arc<RwLock<DedupBuffer>>,
        io_merger: IOMerger,
        read_amplification_monitor: ReadAmplificationMonitor,
    }
    
    pub struct SSDStorage {
        file: File,
        page_size: usize,
        direct_io_enabled: bool,
        spatial_locality_optimizer: SpatialLocalityOptimizer,
    }
    
    pub struct DedupBuffer {
        buffer: Vec<u8>,
        seen_hashes: DashMap<u64, Vec<u8>>,
        capacity: usize,
    }
    
    pub struct IOMerger {
        pending_ios: DashMap<String, Vec<IORequest>>,
        batch_size: usize,
    }
    
    #[derive(Debug, Clone)]
    pub struct IORequest {
        pub id: String,
        pub offset: usize,
        pub length: usize,
        pub data: Vec<u8>,
    }
    
    pub struct ReadAmplificationMonitor {
        reads_requested: usize,
        reads_performed: usize,
        amplification_ratio: f32,
    }
    
    pub struct SpatialLocalityOptimizer {
        similarity_threshold: f32,
        clustering_algorithm: String,
    }
    
    impl IODedupEngine {
        pub fn new(config: &SystemConfig) -> Result<Self> {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&config.ssd_path)?;
            
            // Enable Direct I/O if configured
            if config.enable_direct_io {
                #[cfg(target_os = "linux")]
                unsafe {
                    use libc::{O_DIRECT, fcntl, F_GETFL, F_SETFL};
                    let fd = file.as_raw_fd();
                    let flags = fcntl(fd, F_GETFL, 0);
                    fcntl(fd, F_SETFL, flags | O_DIRECT);
                }
            }
            
            Ok(Self {
                ssd_storage: SSDStorage {
                    file,
                    page_size: 4096,
                    direct_io_enabled: config.enable_direct_io,
                    spatial_locality_optimizer: SpatialLocalityOptimizer {
                        similarity_threshold: 0.85,
                        clustering_algorithm: "kmeans".to_string(),
                    },
                },
                dedup_buffer: Arc::new(RwLock::new(DedupBuffer {
                    buffer: Vec::with_capacity(config.dedup_buffer_size_mb * 1024 * 1024),
                    seen_hashes: DashMap::new(),
                    capacity: config.dedup_buffer_size_mb * 1024 * 1024,
                })),
                io_merger: IOMerger {
                    pending_ios: DashMap::new(),
                    batch_size: config.batch_size,
                },
                read_amplification_monitor: ReadAmplificationMonitor {
                    reads_requested: 0,
                    reads_performed: 0,
                    amplification_ratio: 1.0,
                },
            })
        }
        
        pub async fn store_vectors(&self, vectors: &[kvquant::CompressedVector]) -> Result<()> {
            // Group similar vectors for spatial locality
            let grouped = self.group_similar_vectors(vectors)?;
            
            for group in grouped {
                // Check for deduplication
                let unique_vectors = self.deduplicate(&group)?;
                
                // Merge I/Os within mini-batch
                let merged_requests = self.io_merger.merge_requests(&unique_vectors)?;
                
                // Write to SSD sequentially
                self.write_sequential(&merged_requests).await?;
            }
            
            Ok(())
        }
        
        fn group_similar_vectors(&self, 
                                vectors: &[kvquant::CompressedVector]) 
                                -> Result<Vec<Vec<kvquant::CompressedVector>>> {
            // Group vectors by similarity for better spatial locality
            // This would use clustering algorithms
            Ok(vec![vectors.to_vec()])
        }
        
        fn deduplicate(&self, vectors: &[kvquant::CompressedVector]) -> Result<Vec<kvquant::CompressedVector>> {
            let mut dedup_buffer = self.dedup_buffer.write();
            let mut unique = Vec::new();
            
            for vector in vectors {
                let hash = self.compute_hash(&vector.data);
                
                if !dedup_buffer.seen_hashes.contains_key(&hash) {
                    dedup_buffer.seen_hashes.insert(hash, vector.data.clone());
                    unique.push(vector.clone());
                }
            }
            
            Ok(unique)
        }
        
        fn compute_hash(&self, data: &[u8]) -> u64 {
            // Simple hash function, replace with xxhash or similar
            data.iter().fold(0u64, |acc, &byte| {
                acc.wrapping_mul(31).wrapping_add(byte as u64)
            })
        }
        
        async fn write_sequential(&self, requests: &[IORequest]) -> Result<()> {
            use std::io::Write;
            
            for request in requests {
                self.ssd_storage.file.write_all(&request.data)?;
            }
            
            self.ssd_storage.file.sync_all()?;
            Ok(())
        }
    }
    
    impl IOMerger {
        fn merge_requests(&self, vectors: &[kvquant::CompressedVector]) -> Result<Vec<IORequest>> {
            let mut merged = Vec::new();
            let mut current_batch = Vec::new();
            let mut current_size = 0;
            
            for vector in vectors {
                current_batch.push(vector.data.clone());
                current_size += vector.data.len();
                
                if current_batch.len() >= self.batch_size {
                    merged.push(IORequest {
                        id: format!("batch_{}", merged.len()),
                        offset: 0,
                        length: current_size,
                        data: current_batch.concat(),
                    });
                    current_batch.clear();
                    current_size = 0;
                }
            }
            
            if !current_batch.is_empty() {
                merged.push(IORequest {
                    id: format!("batch_{}", merged.len()),
                    offset: 0,
                    length: current_size,
                    data: current_batch.concat(),
                });
            }
            
            Ok(merged)
        }
    }
}