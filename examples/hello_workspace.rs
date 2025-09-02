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


//! Example showing how to use the zeta-reticula workspace components

// Import workspace crates
use zeta_reticula::diffuse_llm_rs;
use zeta_reticula::fusion_anns;
use zeta_reticula::ns_router_rs;
use zeta_reticula::prefill_kvquant_rs;
use zeta_reticula::salience_engine;

fn main() {
    println!("Zeta Reticula Workspace Example");
    println!("----------------------------");
    
    // Demonstrate that we can access the workspace crates
    println!("Available workspace components:");
    println!("- diffuse-llm-rs: {}", 
        std::any::type_name::<diffuse_llm_rs::DiffuseLLM>());
    println!("- fusion-anns: {}", 
        std::any::type_name::<fusion_anns::FusionANNS>());
    println!("- ns-router-rs: {}", 
        std::any::type_name::<ns_router_rs::NsRouter>());
    println!("- prefill-kvquant-rs: {}", 
        std::any::type_name::<prefill_kvquant_rs::PrefillKVQuant>());
    println!("- salience-engine: {}", 
        std::any::type_name::<salience_engine::SalienceEngine>());
    
    println!("\nWorkspace example completed successfully!");
}
