//! Execution configuration options and canonical defaults for Engine and Decoder.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub attention: String,
    pub prefetch: bool,
    pub adaln_cache: bool,
    pub inference_kernels: bool,
    pub query_chunk: usize,
    pub ff_chunk: usize,
    pub head_chunk: usize,
    pub offload_refiner: bool,
    pub attention_cpu_outputs: bool,
    pub preload_host: bool,
    pub pin_host_weights: bool,
    pub adaln_disk_cache: bool,
    pub projection_chunk: usize,
    pub grouped_attention_outputs: bool,
    pub fp8_ff_recompute: bool,
    pub resident_blocks: usize,
    pub pin_host_gb: f64,
    pub cache_refined_text: bool,
    pub steps: usize,
    pub window_batch: usize,
    pub linear_compute: String,
    pub fp8_gemm: String,
    pub window_varlen: bool,
    pub varlen_smooth_k: bool,
    pub task: String,
    pub stream_weights: bool,
    pub head_parallelism: usize,
    pub residual_offload: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocator_config: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            attention: "dense".to_string(),
            prefetch: true,
            adaln_cache: false,
            inference_kernels: false,
            query_chunk: 0,
            ff_chunk: 0,
            head_chunk: 0,
            offload_refiner: false,
            attention_cpu_outputs: false,
            preload_host: false,
            pin_host_weights: false,
            adaln_disk_cache: false,
            projection_chunk: 0,
            grouped_attention_outputs: false,
            fp8_ff_recompute: false,
            resident_blocks: 0,
            pin_host_gb: 0.0,
            cache_refined_text: false,
            steps: 8,
            window_batch: 1,
            linear_compute: "native-fp8".to_string(),
            fp8_gemm: "auto".to_string(),
            window_varlen: false,
            varlen_smooth_k: true,
            task: "t2va".to_string(),
            stream_weights: false,
            head_parallelism: 1,
            residual_offload: false,
            allocator_config: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub offload: bool,
    pub prefetch: bool,
    pub tile_group: bool,
    pub preload: bool,
    pub pin_weights: bool,
    pub stream_output: bool,
    pub stream_weights: bool,
    pub resident_blocks: usize,
    pub linear_compute_cache: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            offload: false,
            prefetch: true,
            tile_group: false,
            preload: false,
            pin_weights: false,
            stream_output: false,
            stream_weights: false,
            resident_blocks: 0,
            linear_compute_cache: false,
        }
    }
}

pub fn engine_options(val: &serde_json::Value) -> Result<EngineConfig, serde_json::Error> {
    let mut default_val = serde_json::to_value(EngineConfig::default())?;
    if let serde_json::Value::Object(map) = val {
        if let serde_json::Value::Object(ref mut def_map) = default_val {
            for (k, v) in map {
                def_map.insert(k.clone(), v.clone());
            }
        }
    }
    serde_json::from_value(default_val)
}

pub fn decoder_options(val: &serde_json::Value) -> Result<DecoderConfig, serde_json::Error> {
    let mut default_val = serde_json::to_value(DecoderConfig::default())?;
    if let serde_json::Value::Object(map) = val {
        if let serde_json::Value::Object(ref mut def_map) = default_val {
            for (k, v) in map {
                def_map.insert(k.clone(), v.clone());
            }
        }
    }
    serde_json::from_value(default_val)
}
