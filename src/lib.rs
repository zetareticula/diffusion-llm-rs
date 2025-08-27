//! Zeta Reticula - A high-performance LLM inference and serving system

// Re-export all workspace crates
pub use diffuse_llm_rs;
pub use fusion_anns;
pub use io_dedup;
pub use ns_router_rs;
pub use prefill_kvquant_rs;
pub use salience_engine;

// Add any shared types or utilities here

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
