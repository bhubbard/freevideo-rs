//! Capacity grid benchmarking and attention backend evaluation.

use crate::geometry::geometry;
use crate::hardware::{Hardware, GIB};
use crate::policy::choose;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTierResult {
    pub tier_gib: u64,
    pub architecture: String,
    pub admitted: bool,
    pub resident_blocks: usize,
    pub head_chunk: usize,
    pub prefetch: bool,
    pub residual_offload: bool,
    pub stream_weights: bool,
    pub pin_host_gb: f64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkGridReport {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub canvas_frames: u32,
    pub video_tokens: u32,
    pub tiers: Vec<BenchmarkTierResult>,
}

pub fn simulate_capacity_grid(
    width: u32,
    height: u32,
    frames: u32,
    architectures: Option<&[&str]>,
) -> BenchmarkGridReport {
    let canvas = geometry(width, height, Some(frames), None).unwrap();
    let arch_list = architectures.unwrap_or(&["ampere", "ada", "hopper", "blackwell-rtx"]);
    let vram_tiers = [8, 12, 16, 24, 32, 48];

    let mut results = Vec::new();

    for &arch in arch_list {
        for &tier in &vram_tiers {
            let hw = Hardware {
                gpu_name: format!("Simulated {} {}GiB", arch.to_uppercase(), tier),
                gpu_index: 0,
                gpu_uuid: format!("GPU-bench-{}-{}", arch, tier),
                capability: match arch {
                    "ampere" => (8, 0),
                    "ada" => (8, 9),
                    "hopper" => (9, 0),
                    _ => (12, 0),
                },
                architecture: arch.to_string(),
                vram_total: tier * GIB,
                vram_free: (tier * GIB).saturating_sub(GIB / 2),
                ram_total: 64 * GIB,
                ram_available: 56 * GIB,
                system: "Linux".to_string(),
                cgroup_ram_limit: None,
                driver_version: Some("560.35".to_string()),
            };

            match choose(
                &hw,
                Some(tier as f64),
                Some(64.0),
                Some("auto"),
                None,
                None,
                None,
                Some(&canvas),
                None,
                false,
                Some("generation"),
                0,
                0,
            ) {
                Ok(policy) => {
                    results.push(BenchmarkTierResult {
                        tier_gib: tier,
                        architecture: arch.to_string(),
                        admitted: true,
                        resident_blocks: policy.engine.resident_blocks,
                        head_chunk: policy.engine.head_chunk,
                        prefetch: policy.engine.prefetch,
                        residual_offload: policy.engine.residual_offload,
                        stream_weights: policy.engine.stream_weights,
                        pin_host_gb: policy.engine.pin_host_gb,
                        note: "Successfully admitted".to_string(),
                    });
                }
                Err(err) => {
                    results.push(BenchmarkTierResult {
                        tier_gib: tier,
                        architecture: arch.to_string(),
                        admitted: false,
                        resident_blocks: 0,
                        head_chunk: 0,
                        prefetch: false,
                        residual_offload: false,
                        stream_weights: false,
                        pin_host_gb: 0.0,
                        note: err.message,
                    });
                }
            }
        }
    }

    BenchmarkGridReport {
        canvas_width: canvas.width,
        canvas_height: canvas.height,
        canvas_frames: canvas.frames,
        video_tokens: canvas.video_tokens,
        tiers: results,
    }
}
