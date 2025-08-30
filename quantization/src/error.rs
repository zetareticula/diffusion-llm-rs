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
