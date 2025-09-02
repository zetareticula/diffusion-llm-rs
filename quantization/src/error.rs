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


use thiserror::Error;
use std::fmt;

/// Errors that can occur during quantization operations
#[derive(Error, Debug)]
pub enum QuantizationError {
    #[error("Invalid quantization parameters: {0}")]
    InvalidParams(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Calibration data required")]
    CalibrationRequired,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
}

impl serde::ser::StdError for QuantizationError {}

impl From<bincode::Error> for QuantizationError {
    fn from(err: bincode::Error) -> Self {
        QuantizationError::Serialization(err.to_string())
    }
}

impl From<serde_json::Error> for QuantizationError {
    fn from(err: serde_json::Error) -> Self {
        QuantizationError::Serialization(err.to_string())
    }
}
