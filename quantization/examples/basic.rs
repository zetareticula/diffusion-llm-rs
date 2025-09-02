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

//! Basic example of using the quantization module

use ndarray::array;
use num_traits::Float;
use quantization::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample tensor
    let data = array![
        [-1.5, -0.5, 0.5, 1.5],
        [2.0, 3.0, 4.0, 5.0],
    ]
    .into_dyn();

    println!("Original data:\n{:?}\n", data);

    // Quantize to 8-bit
    let quantized = quant_utils::quantize(data.view(), QuantizationType::Int8, true, None)?;
    println!("Quantized data (int8): {:?}", quantized.data);
    println!("Scale: {}, Zero point: {}\n", 
        quantized.params.scale, 
        quantized.params.zero_point);

    // Dequantize back to f32
    let dequantized = quant_utils::dequantize(&quantized)?;
    println!("Dequantized data:\n{:?}\n", dequantized);

    // Show error
    let error = &data - &dequantized;
    println!("Quantization error (max abs): {:.4}", 
        error.iter().fold(0.0f32, |max, &x| max.max(x.abs())));

    Ok(())
}