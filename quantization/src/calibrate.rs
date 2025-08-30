use ndarray::ArrayD;
use std::collections::HashMap;
use crate::{types::QuantizationParams, QuantizationError, Result};

/// Calibration data for quantization
pub struct CalibrationData {
    pub min: f32,
    pub max: f32,
    pub histogram: Vec<usize>,
    pub num_bins: usize,
    pub total_samples: usize,
    pub per_channel_stats: Option<HashMap<usize, (f32, f32)>>,
}

impl CalibrationData {
    /// Create a new calibration data collector
    pub fn new(num_bins: usize, per_channel: bool) -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            histogram: vec![0; num_bins],
            num_bins,
            total_samples: 0,
            per_channel_stats: if per_channel { Some(HashMap::new()) } else { None },
        }
    }
    
    /// Update the calibration data with a new tensor
    pub fn update(&mut self, data: &ArrayD<f32>, channel: Option<usize>) {
        let (min_val, max_val) = data.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
        
        self.min = self.min.min(min_val);
        self.max = self.max.max(max_val);
        self.total_samples += data.len();
        
        // Update per-channel statistics if enabled
        if let (Some(channel), Some(stats)) = (channel, &mut self.per_channel_stats) {
            let (ch_min, ch_max) = stats.entry(channel).or_insert((f32::MAX, f32::MIN));
            *ch_min = ch_min.min(min_val);
            *ch_max = ch_max.max(max_val);
        }
        
        // Update histogram
        if self.max > self.min {
            let bin_width = (self.max - self.min) / self.num_bins as f32;
            for &val in data.iter() {
                if val >= self.min && val <= self.max {
                    let bin = ((val - self.min) / bin_width).floor() as usize;
                    let bin = bin.min(self.num_bins - 1);
                    self.histogram[bin] += 1;
                }
            }
        }
    }
    
    /// Compute optimal quantization parameters
    pub fn compute_params(&self, bits: u8, symmetric: bool) -> Result<QuantizationParams> {
        if self.total_samples == 0 {
            return Err(QuantizationError::CalibrationRequired);
        }
        
        let num_levels = 2u32.pow(bits as u32) as f32;
        let range = self.max - self.min;
        
        if range <= f32::EPSILON {
            return Ok(QuantizationParams {
                bits,
                scale: 1.0,
                zero_point: 0,
                symmetric,
                axis: None,
            });
        }
        
        let scale = if symmetric {
            let max_abs = self.max.abs().max(self.min.abs());
            max_abs * 2.0 / (num_levels - 1.0)
        } else {
            range / (num_levels - 1.0)
        };
        
        let zero_point = if symmetric {
            (num_levels / 2.0 - 1.0) as i32
        } else {
            (-self.min / scale).round() as i32
        };
        
        Ok(QuantizationParams {
            bits,
            scale,
            zero_point,
            symmetric,
            axis: None,
        })
    }
    
    /// Get per-channel statistics if available
    pub fn get_per_channel_stats(&self) -> Option<&HashMap<usize, (f32, f32)>> {
        self.per_channel_stats.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_calibration() {
        let mut calib = CalibrationData::new(10, false);
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        calib.update(&data.into_dyn(), None);
        
        let params = calib.compute_params(8, false).unwrap();
        assert!((params.scale - 0.0235).abs() < 1e-3);
        assert_eq!(params.zero_point, -43);
    }
}
