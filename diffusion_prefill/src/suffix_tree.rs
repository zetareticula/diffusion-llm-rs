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


//! Suffix tree implementation for efficient string operations

use std::collections::HashMap;

/// A node in the suffix tree
#[derive(Debug, Default)]
struct Node {
    children: HashMap<u8, Node>,
    start: usize,
    end: Option<usize>,
}

/// A suffix tree for efficient string operations
#[derive(Debug)]
pub struct SuffixTree {
    root: Node,
    text: Vec<u8>,
}

impl SuffixTree {
    /// Create a new suffix tree from the given text
    pub fn new(text: &str) -> Self {
        let bytes = text.as_bytes().to_vec();
        let mut tree = SuffixTree {
            root: Node::default(),
            text: bytes,
        };
        
        // Build the suffix tree
        for i in 0..tree.text.len() {
            tree.insert_suffix(i);
        }
        
        tree
    }
    
    /// Insert a suffix into the tree
    fn insert_suffix(&mut self, start: usize) {
        let mut current = &mut self.root;
        
        for i in start..self.text.len() {
            let c = self.text[i];
            current = current.children.entry(c).or_insert_with(|| Node {
                children: HashMap::new(),
                start: start,
                end: Some(i),
            });
        }
    }
    
    /// Search for a pattern in the tree
    pub fn search(&self, pattern: &str) -> bool {
        let pattern_bytes = pattern.as_bytes();
        let mut current = &self.root;
        
        for &c in pattern_bytes {
            if let Some(node) = current.children.get(&c) {
                current = node;
            } else {
                return false;
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_suffix_tree() {
        let text = "banana";
        let tree = SuffixTree::new(text);
        
        assert!(tree.search("banana"));
        assert!(tree.search("ana"));
        assert!(tree.search("na"));
        assert!(!tree.search("apple"));
    }
}
