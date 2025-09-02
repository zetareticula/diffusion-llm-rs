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

//! Quantization module for LLM model optimization
//! 
//! This module provides functionality for quantizing LLM weights and activations
//! to lower precision formats to improve inference performance and reduce memory usage.

#![warn(missing_docs)]
#![allow(clippy::needless_doctest_main)]

pub mod calibrate;
pub mod error;
pub mod quantize;
pub mod types;

pub use calibrate::CalibrationData;
pub use error::QuantizationError;
pub use quantize::{
    QuantizationType, 
    Quantizer, 
    DefaultQuantizer, 
    utils as quant_utils
};
pub use types::{DType, QuantizationConfig, QuantizationParams, QuantizedTensor};

/// Result type for quantization operations
pub type Result<T> = std::result::Result<T, QuantizationError>;

/// Re-export common types for convenience
pub mod prelude {
    pub use super::{
        CalibrationData,
        DType,
        DefaultQuantizer,
        QuantizationConfig,
        QuantizationError,
        QuantizationParams,
        QuantizationType,
        QuantizedTensor,
        Quantizer,
        Result,
        quant_utils,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantization_roundtrip() -> Result<()> {
        let data = array![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].into_dyn();
        let qtype = QuantizationType::Int8;
        let quantizer = DefaultQuantizer::new(qtype.bits(), false, None);
        
        let quantized = quantizer.quantize(data.view(), qtype)?;
        let dequantized = quantizer.dequantize(&quantized)?;
        
        // Check shape preservation
        assert_eq!(dequantized.shape(), data.shape());
        
        // Check values are approximately equal
        for (a, b) in data.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < 0.1, "Mismatch: {} vs {}", a, b);
        }
        
        Ok(())
    }
}