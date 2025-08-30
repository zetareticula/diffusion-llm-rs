use ndarray::{ArrayD, IxDyn};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Parameters for quantization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizationParams {
    pub bits: u8,
    pub scale: f32,
    pub zero_point: i32,
    pub symmetric: bool,
    pub axis: Option<usize>,
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            bits: 8,
            scale: 1.0,
            zero_point: 0,
            symmetric: true,
            axis: None,
        }
    }
}

/// A quantized tensor with its parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub params: QuantizationParams,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<u8>, shape: Vec<usize>, params: QuantizationParams) -> Self {
        Self { data, shape, params }
    }

    /// Get the total number of elements in the tensor
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the tensor's shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Dequantize the tensor back to f32
    pub fn dequantize(&self) -> ArrayD<f32> {
        let mut output = ArrayD::zeros(IxDyn(&self.shape));
        let scale = self.params.scale;
        let zero_point = self.params.zero_point as f32;
        
        for (i, &val) in self.data.iter().enumerate() {
            output[i] = (val as f32 - zero_point) * scale;
        }
        
        output
    }
}

/// Supported data types for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I8,
    U8,
    I4,
    U4,
    B1,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::I8 => write!(f, "i8"),
            DType::U8 => write!(f, "u8"),
            DType::I4 => write!(f, "i4"),
            DType::U4 => write!(f, "u4"),
            DType::B1 => write!(f, "b1"),
        }
    }
}

/// Configuration for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub bits: u8,
    pub group_size: usize,
    pub sym: bool,
    pub desc_act: bool,
    pub true_sequential: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quant_method: "gptq".to_string(),
            bits: 4,
            group_size: 128,
            sym: true,
            desc_act: true,
            true_sequential: true,
        }
    }
}
