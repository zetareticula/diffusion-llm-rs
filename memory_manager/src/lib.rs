//! Memory management utilities for the zeta-reticula project

use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;

/// A simple memory manager that tracks allocations
#[derive(Debug, Default)]
pub struct MemoryManager {
    allocations: DashMap<String, Vec<u8>>,
}

impl MemoryManager {
    /// Create a new MemoryManager
    pub fn new() -> Self {
        Self {
            allocations: DashMap::new(),
        }
    }

    /// Allocate memory with the given key and size
    pub fn allocate(&self, key: &str, size: usize) -> anyhow::Result<()> {
        self.allocations.insert(key.to_string(), vec![0u8; size]);
        Ok(())
    }

    /// Deallocate memory with the given key
    pub fn deallocate(&self, key: &str) -> anyhow::Result<()> {
        self.allocations.remove(key);
        Ok(())
    }

    /// Get a reference to the allocated memory
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.allocations.get(key).map(|v| v.clone())
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
    }
}
