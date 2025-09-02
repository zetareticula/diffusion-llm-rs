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
//! 
//! Memory management utilities for the zeta-reticula project
//! 
//! Currently only implements a simple memory manager that tracks allocations.
//! In the future, this will be extended to support more advanced memory management strategies.
//! 
//! # Usage
//! ```
//! use zeta_memory_manager::MemoryManager;
//! 
//! let manager = MemoryManager::new();
//! manager.allocate("test", 1024).unwrap();
//! 
//! ```

pub mod memory_manager;

pub use memory_manager::MemoryManager;

#[derive(Debug, thiserror::Error)]
pub enum MemoryManagerError {
    #[error("Memory already allocated for key: {0}")]
    AlreadyAllocated(String),
}

impl From<MemoryManagerError> for anyhow::Error {
    fn from(err: MemoryManagerError) -> Self {
        anyhow::Error::msg(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, MemoryManagerError>;

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub key: String,
    pub size: usize,
}

impl MemoryAllocation {
    pub fn new(key: String, size: usize) -> Self {
        Self { key, size }
    }
}

impl fmt::Display for MemoryAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} bytes", self.key, self.size)
    }
}

impl Serialize for MemoryAllocation {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("MemoryAllocation", 2)?;
        state.serialize_field("key", &self.key)?;
        state.serialize_field("size", &self.size)?;
        state.end()
    }
}

impl Deserialize for MemoryAllocation {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'_>,
    {
        #[derive(Deserialize)]
        struct MemoryAllocationHelper {
            key: String,
            size: usize,
        }

        let helper = MemoryAllocationHelper::deserialize(deserializer)?;
        Ok(MemoryAllocation::new(helper.key, helper.size))
    }
}

/// A simple memory manager that tracks allocations
#[derive(Debug, Default)]
pub struct MemoryManager {
    allocations: DashMap<String, Vec<u8>>,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            allocations: DashMap::new(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager() {
        let manager = MemoryManager::new();
        manager.allocate("test", 1024).unwrap();
        assert!(manager.get("test").is_some());
        manager.deallocate("test").unwrap();
        assert!(manager.get("test").is_none());
        
        assert!(manager.allocate("test", 1024).is_err());
        manager.allocate("other", 1024).unwrap();
        assert!(manager.get("other").is_some());
        manager.deallocate("other").unwrap();
        assert!(manager.get("other").is_none());
        
        assert!(manager.allocate("other", 1024).is_ok());
    }
}
