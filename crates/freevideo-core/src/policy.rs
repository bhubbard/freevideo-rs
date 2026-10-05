//! Geometry-aware resource placement, memory budgets, and adaptive execution planning.

use crate::execution_config::{DecoderConfig, EngineConfig};
use crate::geometry::Canvas;
use crate::hardware::{Hardware, GIB, GIB_F64};
use crate::system::{residual_host_headroom, weight_cache_headroom, HOST_WEIGHT_HEADROOM};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const BLOCK_BYTES: u64 = 432_500_000;
pub const DECODER_BLOCK_BYTES: u64 = 268_574_720;
pub const GPU_ATTENTION_OUTPUT_BYTES: u64 = (0.3 * GIB_F64) as u64;
pub const SMALL_NO_WEIGHTS_PEAK: u64 = (7.8 * GIB_F64) as u64;
pub const SMALL_HOST_OUTPUT_PEAK: u64 = (6.7 * GIB_F64) as u64;
pub const SMALL_ACTIVATION_RESERVE: u64 = (7.6 * GIB_F64) as u64;
pub const SMALL_ACTIVATION_RESERVES: [(usize, u64); 2] =
    [(8, (8.2 * GIB_F64) as u64), (4, (7.6 * GIB_F64) as u64)];
pub const WINDOWS_SMALL_HEAD8_TOKENS: u32 = 18144;
pub const WINDOWS_SMALL_HEAD8_FRAMES: u32 = 243;
pub const RESIDENT_TARGET: usize = 8;
pub const RESIDENT_VRAM_CEILING: usize = 44;
pub const PREFETCH_GPU_BUDGET: u64 = (13.5 * GIB_F64) as u64;
pub const ENCODER_HOST_BYTES: u64 = (3.6 * GIB_F64) as u64;
pub const RESIDUAL_ACTIVATION_RESERVE: u64 = (6.4 * GIB_F64) as u64;
pub const RESIDUAL_ADMISSION_FLOOR: u64 = (6.25 * GIB_F64) as u64;
pub const MIN_GPU_RESERVE_GIB: f64 = 0.2;

const ACTIVATION_BY_TOKENS_4: [(u32, f64); 3] = [(41472, 6.11), (72576, 7.69), (102816, 9.97)];
const ACTIVATION_BY_TOKENS_8: [(u32, f64); 3] = [(41472, 6.11), (72576, 8.03), (102816, 10.53)];
const ACTIVATION_BY_TOKENS_16: [(u32, f64); 3] = [(41472, 6.75), (72576, 10.05), (102816, 14.21)];

pub fn activation_bytes(head: usize, tokens: Option<u32>) -> u64 {
    let t = tokens.unwrap_or(72576) as f64;

    let interpolate = |anchors: &[(u32, f64); 3]| -> u64 {
        for i in 0..anchors.len() - 1 {
            let (low, at_low) = (anchors[i].0 as f64, anchors[i].1);
            let (high, at_high) = (anchors[i + 1].0 as f64, anchors[i + 1].1);
            if t <= high || i == anchors.len() - 2 {
                let slope = (at_high - at_low) / (high - low);
                let gib = at_low + slope * (t - low);
                return (gib * GIB_F64) as u64;
            }
        }
        (anchors.last().unwrap().1 * GIB_F64) as u64
    };

    let mut max_bytes = 0u64;
    if head >= 4 {
        max_bytes = max_bytes.max(interpolate(&ACTIVATION_BY_TOKENS_4));
    }
    if head >= 8 {
        max_bytes = max_bytes.max(interpolate(&ACTIVATION_BY_TOKENS_8));
    }
    if head >= 16 {
        max_bytes = max_bytes.max(interpolate(&ACTIVATION_BY_TOKENS_16));
    }
    max_bytes
}

pub fn prefetch_for_budget(
    gpu_budget: u64,
    ram_budget: Option<u64>,
    twelve: bool,
    ampere: bool,
    system: &str,
    architecture: &str,
) -> bool {
    let low_host = system == "Windows" && ram_budget.is_some_and(|b| b < 8 * GIB);
    gpu_budget >= 14 * GIB
        || (gpu_budget >= PREFETCH_GPU_BUDGET && low_host)
        || (gpu_budget >= PREFETCH_GPU_BUDGET
            && system == "Windows"
            && architecture == "blackwell-rtx")
        || (twelve && ampere)
}

pub fn windows_gpu_output_workspace(video_tokens: Option<u32>, prefetch: bool) -> u64 {
    let tokens = video_tokens.unwrap_or(72576) as f64;
    let base = (6.5 + 6.0 * (tokens / 72576.0)) * GIB_F64;
    (base as u64) + (if prefetch { BLOCK_BYTES } else { 0 })
}

pub fn decoder_workspace(canvas: Option<&Canvas>) -> u64 {
    let area = match canvas {
        Some(c) => (c.width as f64 * c.height as f64) / (1344.0 * 768.0),
        None => 1.0,
    };
    (3.25 * GIB_F64 * area.max(1.0)) as u64
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub gpu_capacity_bytes: u64,
    pub ram_capacity_bytes: u64,
    pub gpu_available_bytes: u64,
    pub ram_available_bytes: u64,
    pub gpu_system_reserve_bytes: u64,
    pub ram_system_reserve_bytes: u64,
    pub gpu_budget_bytes: u64,
    pub ram_budget_bytes: u64,
}

pub fn resource_budget(
    hardware: &Hardware,
    vram_gib: Option<f64>,
    ram_gib: Option<f64>,
    gpu_reserve_gib: Option<f64>,
    ram_reserve_gib: Option<f64>,
) -> Result<ResourceBudget, String> {
    if let Some(res) = gpu_reserve_gib {
        if !res.is_finite() || res < MIN_GPU_RESERVE_GIB {
            return Err("Keep at least 0.2 GiB GPU reserve for the system.".to_string());
        }
    }
    if let Some(r) = ram_reserve_gib {
        if !r.is_finite() || r < 0.0 {
            return Err("System RAM headroom must be finite and nonnegative.".to_string());
        }
    }

    let gpu_total = if let Some(v) = vram_gib {
        hardware.vram_total.min((v * GIB_F64) as u64)
    } else {
        hardware.vram_total
    };

    let ram_total = if let Some(r) = ram_gib {
        hardware.ram_total.min((r * GIB_F64) as u64)
    } else {
        hardware.ram_total
    };

    let desktop = hardware.system == "Windows";
    let free_gpu = gpu_total.min(hardware.vram_free);
    let mut free_ram = ram_total.min(hardware.ram_available);
    if let Some(limit) = hardware.cgroup_ram_limit {
        free_ram = free_ram.min(limit);
    }

    let reserve_gpu_f64 = match gpu_reserve_gib {
        Some(res) => res,
        None => {
            if desktop {
                0.25
            } else {
                ((free_gpu as f64 / GIB_F64) * 0.025).clamp(0.5, 1.0)
            }
        }
    };

    let reserve_ram_f64 = match ram_reserve_gib {
        Some(res) => res,
        None => {
            let pct = (free_ram as f64 / GIB_F64) * 0.05;
            if desktop {
                pct.clamp(2.0, 4.0)
            } else {
                pct.clamp(1.0, 4.0)
            }
        }
    };

    let gpu_reserve_bytes = (reserve_gpu_f64 * GIB_F64) as u64;
    let ram_reserve_bytes = (reserve_ram_f64 * GIB_F64) as u64;

    let gpu_budget = free_gpu.saturating_sub(gpu_reserve_bytes);
    let ram_budget = free_ram.saturating_sub(ram_reserve_bytes);

    Ok(ResourceBudget {
        gpu_capacity_bytes: gpu_total,
        ram_capacity_bytes: ram_total,
        gpu_available_bytes: free_gpu,
        ram_available_bytes: free_ram,
        gpu_system_reserve_bytes: gpu_reserve_bytes,
        ram_system_reserve_bytes: ram_reserve_bytes,
        gpu_budget_bytes: gpu_budget,
        ram_budget_bytes: ram_budget,
    })
}

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
#[error("FreeVideo resource planning could not admit this request: {message}")]
pub struct ResourceBudgetError {
    pub message: String,
    pub gpu: String,
    pub system: String,
    pub gpu_budget_bytes: u64,
    pub gpu_minimum_bytes: u64,
    pub ram_budget_bytes: u64,
    pub ram_minimum_bytes: u64,
    pub insufficient: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub schema_version: u32,
    pub hardware: serde_json::Value,
    pub gpu_budget_bytes: u64,
    pub ram_budget_bytes: u64,
    pub gpu_system_reserve_bytes: u64,
    pub ram_system_reserve_bytes: u64,
    pub allocator_config: String,
    pub engine: EngineConfig,
    pub decoder: DecoderConfig,
    pub evidence: String,
    pub notes: Vec<String>,
    pub capacity_trial: bool,
    pub lora_max_block_bytes: u64,
    pub lora_root_bytes: u64,
}

impl Policy {
    pub fn legacy_profile(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "Automatic VDN Engine policy",
            "gpu_budget_gb": self.gpu_budget_bytes as f64 / 1e9,
            "inference_ram_budget_gb": self.ram_budget_bytes as f64 / 1e9,
            "allocator_config": self.allocator_config,
            "engine": self.engine,
            "decoder": self.decoder,
            "policy": self
        })
    }
}

pub fn choose(
    hardware: &Hardware,
    vram_gib: Option<f64>,
    ram_gib: Option<f64>,
    attention: Option<&str>,
    gpu_reserve_gib: Option<f64>,
    ram_reserve_gib: Option<f64>,
    available_backends: Option<&[String]>,
    canvas: Option<&Canvas>,
    demonstrated_ram_bytes: Option<u64>,
    allow_capacity_trial: bool,
    stage: Option<&str>,
    lora_max_block_bytes: u64,
    lora_root_bytes: u64,
) -> Result<Policy, ResourceBudgetError> {
    let stage_str = stage.unwrap_or("generation");
    if stage_str != "encoding" && stage_str != "generation" {
        return Err(ResourceBudgetError {
            message: "Stage must be encoding or generation".to_string(),
            gpu: hardware.gpu_name.clone(),
            system: hardware.system.clone(),
            gpu_budget_bytes: 0,
            gpu_minimum_bytes: 0,
            ram_budget_bytes: 0,
            ram_minimum_bytes: 0,
            insufficient: vec![],
        });
    }

    let (lora_max, lora_root) = if stage_str == "encoding" {
        (0, 0)
    } else {
        (lora_max_block_bytes, lora_root_bytes)
    };

    let block_bytes = BLOCK_BYTES + lora_max;
    let lora_workspace = if lora_max > 0 || lora_root > 0 {
        128 * 1024 * 1024 + 2 * lora_max + lora_root
    } else {
        0
    };

    let reference_rows: u32 = canvas
        .map(|c| c.reference_video_tokens.unwrap_or(0) + c.reference_audio_tokens.unwrap_or(0))
        .unwrap_or(0);
    let effective_tokens = canvas.map(|c| c.video_tokens + reference_rows);

    let budget = resource_budget(
        hardware,
        vram_gib,
        ram_gib,
        gpu_reserve_gib,
        ram_reserve_gib,
    )
    .map_err(|e| ResourceBudgetError {
        message: e,
        gpu: hardware.gpu_name.clone(),
        system: hardware.system.clone(),
        gpu_budget_bytes: 0,
        gpu_minimum_bytes: 0,
        ram_budget_bytes: 0,
        ram_minimum_bytes: 0,
        insufficient: vec![],
    })?;

    let desktop = hardware.system == "Windows";
    let free_gpu = budget.gpu_available_bytes;
    let _free_ram = budget.ram_available_bytes;
    let mut reserve_gpu = budget.gpu_system_reserve_bytes as f64 / GIB_F64;
    let reserve_ram = budget.ram_system_reserve_bytes as f64 / GIB_F64;
    let original_reserve_gpu = reserve_gpu;

    let mut request_gpu_budget = budget.gpu_budget_bytes;
    let mut gpu_budget = request_gpu_budget.saturating_sub(lora_workspace);
    let ram_budget = budget.ram_budget_bytes;

    let mut retention_budget = ram_budget;
    if let Some(demo) = demonstrated_ram_bytes {
        let max_alloc = budget
            .ram_capacity_bytes
            .saturating_sub(budget.ram_system_reserve_bytes);
        retention_budget = ram_budget.max(demo.min(max_alloc));
    }

    let small_adjustment = if let Some(c) = canvas {
        let eff = effective_tokens.unwrap_or(c.video_tokens) as f64;
        let diff = eff / (72.0 * 1008.0) - 1.0;
        (4.6 * GIB_F64 * diff) as i64
    } else {
        0
    };

    let fixed_workspace = decoder_workspace(canvas);
    let residual_need =
        (fixed_workspace as i64).max(RESIDUAL_ACTIVATION_RESERVE as i64 + small_adjustment) as u64;
    let residual_floor = if desktop {
        residual_need
    } else {
        (fixed_workspace as i64).max(RESIDUAL_ADMISSION_FLOOR as i64 + small_adjustment) as u64
    };

    let gpu_minimum = fixed_workspace.max((5 * GIB).min(residual_floor));

    if !desktop && gpu_reserve_gib.is_none() && free_gpu < 10 * GIB && gpu_budget < residual_need {
        reserve_gpu = MIN_GPU_RESERVE_GIB;
        request_gpu_budget = free_gpu.saturating_sub((reserve_gpu * GIB_F64) as u64);
        gpu_budget = request_gpu_budget.saturating_sub(lora_workspace);
    }

    let ram_minimum = if stage_str == "encoding" {
        ENCODER_HOST_BYTES
    } else {
        4 * GIB
    };

    if gpu_budget < gpu_minimum || ram_budget < ram_minimum {
        let mut failed = Vec::new();
        if gpu_budget < gpu_minimum {
            failed.push("gpu".to_string());
        }
        if ram_budget < ram_minimum {
            failed.push("ram".to_string());
        }
        return Err(ResourceBudgetError {
            message: format!("Insufficient resources: {:?}", failed),
            gpu: hardware.gpu_name.clone(),
            system: hardware.system.clone(),
            gpu_budget_bytes: gpu_budget,
            gpu_minimum_bytes: gpu_minimum,
            ram_budget_bytes: ram_budget,
            ram_minimum_bytes: ram_minimum,
            insufficient: failed,
        });
    }

    if ram_budget < ENCODER_HOST_BYTES {
        return Err(ResourceBudgetError {
            message: "RAM budget below text encoder host requirement".to_string(),
            gpu: hardware.gpu_name.clone(),
            system: hardware.system.clone(),
            gpu_budget_bytes: gpu_budget,
            gpu_minimum_bytes: gpu_minimum,
            ram_budget_bytes: ram_budget,
            ram_minimum_bytes: ENCODER_HOST_BYTES,
            insufficient: vec!["ram".to_string()],
        });
    }

    let capacity_trial = allow_capacity_trial && gpu_budget < residual_floor;
    let default_backends = vec!["cudnn".to_string(), "torch-flash".to_string()];
    let available: Vec<String> = available_backends.unwrap_or(&default_backends).to_vec();

    let mut chosen_attn = attention.unwrap_or("auto").to_string();
    if chosen_attn == "auto" {
        let preference = ["sage2", "torch-flash", "cudnn", "fa2", "fa4"];
        chosen_attn = preference
            .iter()
            .find(|&&p| available.iter().any(|a| a == p))
            .unwrap_or(&"cudnn")
            .to_string();
    }

    let legs: Vec<String> = if chosen_attn.contains('/') {
        chosen_attn.split('/').map(|s| s.to_string()).collect()
    } else {
        vec![chosen_attn.clone(), chosen_attn.clone()]
    };

    let mut small = gpu_budget < 10 * GIB;
    if desktop
        && hardware.architecture != "ampere"
        && gpu_budget < 11 * GIB
        && gpu_budget < windows_gpu_output_workspace(effective_tokens, false)
    {
        small = true;
    }

    let twelve = !small && gpu_budget < 14 * GIB;
    let ampere = hardware.architecture == "ampere";
    let mut prefetch = prefetch_for_budget(
        gpu_budget,
        Some(ram_budget),
        twelve,
        ampere,
        &hardware.system,
        &hardware.architecture,
    );

    let mut head: usize = if desktop {
        let mut h = if twelve || gpu_budget < 20 * GIB {
            8
        } else {
            16
        };
        while h > 4 && activation_bytes(h, effective_tokens) > gpu_budget {
            h /= 2;
        }
        if !small && !ampere {
            while h > 8
                && (windows_gpu_output_workspace(effective_tokens, false)
                    + activation_bytes(h, effective_tokens)
                    - activation_bytes(8, effective_tokens))
                    > gpu_budget
            {
                h /= 2;
            }
        }
        h
    } else {
        let groups = [16, 8, 4];
        *groups
            .iter()
            .find(|&&g| gpu_budget >= activation_bytes(g, effective_tokens))
            .unwrap_or(&4)
    };

    let mut resident: usize;

    let mut residual_offload = false;
    let mut cpu_outputs = false;
    let mut no_weights = false;

    if small {
        let windows_small_head8 = desktop
            && !ampere
            && legs == ["sage2", "sage2"]
            && effective_tokens.is_some()
            && canvas.is_some_and(|c| c.frames <= WINDOWS_SMALL_HEAD8_FRAMES)
            && effective_tokens.unwrap_or(0) <= WINDOWS_SMALL_HEAD8_TOKENS;

        let mut selected: Option<(usize, u64)> = None;

        if !desktop && head >= 8 {
            selected = Some((
                head,
                activation_bytes(head, effective_tokens).saturating_sub(GPU_ATTENTION_OUTPUT_BYTES),
            ));
        } else {
            let mut available_candidates = Vec::new();
            for (heads, need) in SMALL_ACTIVATION_RESERVES {
                if heads <= 4
                    || !desktop
                    || (windows_small_head8
                        && gpu_budget as i64
                            >= need as i64 + small_adjustment + GPU_ATTENTION_OUTPUT_BYTES as i64)
                {
                    available_candidates.push((heads, need));
                }
            }
            for (heads, need) in available_candidates {
                let adjusted = (need as i64 + small_adjustment) as u64;
                if gpu_budget >= adjusted {
                    selected = Some((heads, adjusted));
                    break;
                }
            }
        }

        if selected.is_none()
            && !desktop
            && free_gpu as i64 >= SMALL_NO_WEIGHTS_PEAK as i64 + small_adjustment
        {
            selected = Some((
                4,
                (SMALL_ACTIVATION_RESERVE as i64 + small_adjustment) as u64,
            ));
            no_weights = true;
        }

        if selected.is_none()
            && !desktop
            && (reserve_gpu - original_reserve_gpu).abs() < 1e-6
            && free_gpu as i64 >= SMALL_HOST_OUTPUT_PEAK as i64 + small_adjustment
            && gpu_budget < SMALL_HOST_OUTPUT_PEAK + GPU_ATTENTION_OUTPUT_BYTES
        {
            selected = Some((4, (SMALL_HOST_OUTPUT_PEAK as i64 + small_adjustment) as u64));
        }

        if selected.is_none() && (gpu_budget >= residual_floor || capacity_trial) {
            selected = Some((4, residual_need));
            residual_offload = true;
        }

        if selected.is_none() {
            return Err(ResourceBudgetError {
                message: "Working set exceeds available capacity".to_string(),
                gpu: hardware.gpu_name.clone(),
                system: hardware.system.clone(),
                gpu_budget_bytes: gpu_budget,
                gpu_minimum_bytes: residual_floor,
                ram_budget_bytes: ram_budget,
                ram_minimum_bytes: ram_minimum,
                insufficient: vec!["gpu".to_string()],
            });
        }

        let (sel_head, mut reserve) = selected.unwrap();
        head = sel_head;

        cpu_outputs = !no_weights && gpu_budget < reserve + GPU_ATTENTION_OUTPUT_BYTES;
        if !cpu_outputs {
            reserve += GPU_ATTENTION_OUTPUT_BYTES;
        }
        residual_offload = residual_offload || cpu_outputs;

        if desktop && !residual_offload {
            reserve = reserve.max(activation_bytes(head, effective_tokens));
        }

        resident =
            RESIDENT_VRAM_CEILING.min((gpu_budget.saturating_sub(reserve) / block_bytes) as usize);
    } else {
        resident = RESIDENT_VRAM_CEILING.min(
            (gpu_budget.saturating_sub(activation_bytes(head, effective_tokens)) / block_bytes)
                as usize,
        );
    }

    let mut windows_workspace = None;
    let mut windows_activation_limited = false;

    if desktop && !ampere && !small {
        let mut ws = windows_gpu_output_workspace(effective_tokens, prefetch);
        ws += activation_bytes(head, effective_tokens)
            .saturating_sub(activation_bytes(8, effective_tokens));

        if prefetch && gpu_budget < 14 * GIB && hardware.architecture == "blackwell-rtx" {
            let with_slot = resident.min((gpu_budget.saturating_sub(ws) / block_bytes) as usize);
            let without_slot = resident.min(
                (gpu_budget.saturating_sub(ws.saturating_sub(BLOCK_BYTES)) / block_bytes) as usize,
            );
            let host_need = ((50 - with_slot) as u64) * block_bytes + (2.5 * GIB_F64) as u64;
            let host_cap = ((50 - without_slot) as u64) * block_bytes + (2.5 * GIB_F64) as u64;
            if host_need > ram_budget && ram_budget >= host_cap {
                prefetch = false;
                ws -= BLOCK_BYTES;
            }
        }

        if prefetch && ws > gpu_budget && gpu_budget >= ws.saturating_sub(BLOCK_BYTES) {
            prefetch = false;
            ws -= BLOCK_BYTES;
        }

        resident = resident.min((gpu_budget.saturating_sub(ws) / block_bytes) as usize);

        if ws > gpu_budget {
            if gpu_budget < residual_floor && !capacity_trial {
                return Err(ResourceBudgetError {
                    message: "Windows activation workspace exceeds GPU budget".to_string(),
                    gpu: hardware.gpu_name.clone(),
                    system: hardware.system.clone(),
                    gpu_budget_bytes: gpu_budget,
                    gpu_minimum_bytes: residual_floor,
                    ram_budget_bytes: ram_budget,
                    ram_minimum_bytes: ram_minimum,
                    insufficient: vec!["gpu".to_string()],
                });
            }
            head = 4;
            cpu_outputs = true;
            residual_offload = true;
            prefetch = false;
            resident = 0;
            ws = residual_need;
            windows_activation_limited = true;
        }
        windows_workspace = Some(ws);
    }

    if capacity_trial {
        head = 4;
        resident = 0;
        prefetch = false;
        cpu_outputs = true;
        residual_offload = true;
    }

    let mut pin =
        (retention_budget.saturating_sub(HOST_WEIGHT_HEADROOM) as f64).clamp(0.0, 22e9) / 1e9;
    pin = pin.min(((50 - resident) as u64 * block_bytes) as f64 / 1e9);

    let residual_host = if residual_offload {
        residual_host_headroom(canvas)
    } else {
        0
    };
    let host_activation =
        ((if small || cpu_outputs { 2.0 } else { 0.5 }) * GIB_F64) as u64 + residual_host;

    let host_room = retention_budget.saturating_sub(host_activation + 2 * GIB);
    let ram_wants = 50usize.saturating_sub((host_room / block_bytes) as usize);

    if !small {
        resident = resident.min(RESIDENT_TARGET.max(ram_wants));
    }

    let streamed_weights = (hardware.system == "Windows" && resident < 50)
        || ((50 - resident) as u64 * block_bytes + host_activation + 2 * GIB > retention_budget);

    let host_headroom = if streamed_weights {
        let h_room = if ram_budget >= 8 * GIB {
            weight_cache_headroom(Some(&hardware.system), true, cpu_outputs, canvas)
        } else {
            HOST_WEIGHT_HEADROOM
        } + residual_host;
        pin = (retention_budget.saturating_sub(h_room) as f64)
            .min(((50 - resident) as u64 * block_bytes) as f64)
            .max(0.0)
            / 1e9;
        h_room
    } else {
        HOST_WEIGHT_HEADROOM
    };

    if small && !streamed_weights && (pin * 1e9) < ((50 - resident) as u64 * block_bytes) as f64 {
        pin = 0.0;
    }

    let engine = EngineConfig {
        attention: legs.join("/"),
        prefetch,
        adaln_cache: true,
        inference_kernels: true,
        query_chunk: 0,
        ff_chunk: 2048,
        head_chunk: head,
        offload_refiner: true,
        attention_cpu_outputs: cpu_outputs,
        preload_host: false,
        pin_host_weights: false,
        adaln_disk_cache: true,
        projection_chunk: 1024,
        grouped_attention_outputs: cpu_outputs,
        fp8_ff_recompute: capacity_trial && hardware.capability.0 >= 10,
        resident_blocks: resident,
        pin_host_gb: (pin * 1000.0).round() / 1000.0,
        cache_refined_text: true,
        steps: 8,
        window_batch: if ampere { 1 } else { 4 },
        linear_compute: if ampere {
            "bf16-weight-only".to_string()
        } else {
            "native-fp8".to_string()
        },
        fp8_gemm: "auto".to_string(),
        window_varlen: false,
        varlen_smooth_k: true,
        task: "t2va".to_string(),
        stream_weights: streamed_weights,
        head_parallelism: 1,
        residual_offload,
        allocator_config: None,
    };

    let mut notes = vec![
        "Request resolution, frame count, steps and attention backend are preserved.".to_string(),
        "Budgets use current free VRAM and available physical/commit RAM, with growth headroom; OS usage is already excluded.".to_string(),
        "Automatic execution uses matching GPU kernel probes; read-only plans may use package discovery. Small probes do not prove full-video capacity.".to_string(),
        "Budgets guide placement; cgroup/MPS enforcement belongs to capacity benchmarks.".to_string(),
    ];

    if let Some(c) = canvas {
        let reference_tokens = (if small { 72 } else { 102 }) * 1008;
        let variable_gib = if small {
            4.6
        } else if head == 8 {
            6.0
        } else {
            7.5
        };
        let geom_adj = (variable_gib
            * GIB_F64
            * (effective_tokens.unwrap() as f64 / reference_tokens as f64 - 1.0))
            as i64;
        notes.push(format!(
            "Geometry: {}x{}, {} frames, {} video tokens; resident-weight allowance adjusted by {:.3} GiB for activation storage.",
            c.width, c.height, c.frames, c.video_tokens, geom_adj as f64 / GIB_F64
        ));
    }

    if desktop && small && head == 8 {
        notes.push(format!(
            "Use eight-head groups for this small Windows canvas ({} generated and reference rows).",
            effective_tokens.unwrap_or(0)
        ));
    }

    if capacity_trial {
        notes.push(format!(
            "The {:.2} GiB live budget is below the {:.2} GiB working-set estimate. Attempt the bounded host-output/residual path with no resident blocks.",
            gpu_budget as f64 / GIB_F64,
            residual_floor as f64 / GIB_F64
        ));
    }

    if windows_activation_limited {
        notes.push(format!(
            "The Windows GPU-output activation estimate exceeds the live budget even with zero resident blocks. Start this token count with four-head groups, host attention outputs and one host residual buffer; estimated workspace {:.2} GiB.",
            windows_workspace.unwrap_or(0) as f64 / GIB_F64
        ));
    }

    if residual_offload {
        notes.push(format!(
            "The original small path exceeds the live VRAM budget: stage residuals in one reusable CPU buffer, preserving kernel shapes, dtype and every requested row. Host weight caching leaves an additional {:.2} GiB for this buffer.",
            residual_host as f64 / GIB_F64
        ));
    }

    if streamed_weights {
        notes.push(format!(
            "Host weight cache keeps {:.2} GiB for this path's working memory, inside the separate OS growth reserve; the runtime rechecks available RAM.",
            host_headroom as f64 / GIB_F64
        ));
    }

    gpu_budget = request_gpu_budget;
    let decoder_offload = gpu_budget < 20 * GIB;

    let output_bytes = if let Some(c) = canvas {
        (c.width * c.height * c.frames * 3) as u64
    } else {
        1344 * 768 * 243 * 3
    };

    let stream_output = decoder_offload || ram_budget < 8 * GIB + 3 * output_bytes;
    let decoder_stream_weights = decoder_offload && ram_budget < 14 * GIB;

    let vae_workspace = decoder_workspace(canvas);
    let vae_resident = if decoder_offload {
        36.min((gpu_budget.saturating_sub(vae_workspace) / DECODER_BLOCK_BYTES) as usize)
    } else {
        0
    };

    let decoder_linear_cache = (hardware.architecture == "blackwell-rtx"
        && hardware.system == "Linux")
        || (["ada", "blackwell-rtx"].contains(&hardware.architecture.as_str())
            && hardware.system == "Windows");

    let decoder = DecoderConfig {
        offload: decoder_offload,
        prefetch: decoder_offload && !small,
        tile_group: decoder_offload,
        preload: false,
        pin_weights: decoder_offload && ram_budget >= 16 * GIB,
        stream_output,
        stream_weights: decoder_stream_weights,
        resident_blocks: vae_resident,
        linear_compute_cache: decoder_linear_cache,
    };

    Ok(Policy {
        schema_version: 2,
        hardware: hardware.to_json_value(),
        gpu_budget_bytes: gpu_budget,
        ram_budget_bytes: ram_budget,
        gpu_system_reserve_bytes: (reserve_gpu * GIB_F64) as u64,
        ram_system_reserve_bytes: (reserve_ram * GIB_F64) as u64,
        allocator_config: "backend:native,pinned_max_round_threshold_mb:1,expandable_segments:True"
            .to_string(),
        engine,
        decoder,
        evidence: "Architecture-specific candidate; validate this exact policy on the target GPU."
            .to_string(),
        notes,
        capacity_trial,
        lora_max_block_bytes: lora_max,
        lora_root_bytes: lora_root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::geometry;

    fn fixture_hw_ada_16g() -> Hardware {
        Hardware {
            gpu_name: "NVIDIA GeForce RTX 4080".to_string(),
            gpu_index: 0,
            gpu_uuid: "GPU-12345678-1234-1234-1234-123456789abc".to_string(),
            capability: (8, 9),
            architecture: "ada".to_string(),
            vram_total: 16 * GIB,
            vram_free: 15 * GIB,
            ram_total: 32 * GIB,
            ram_available: 28 * GIB,
            system: "Linux".to_string(),
            cgroup_ram_limit: None,
            driver_version: Some("560.35".to_string()),
        }
    }

    #[test]
    fn test_activation_bytes() {
        let b4 = activation_bytes(4, Some(72576));
        let b8 = activation_bytes(8, Some(72576));
        let b16 = activation_bytes(16, Some(72576));
        assert!(b4 <= b8);
        assert!(b8 <= b16);
    }

    #[test]
    fn test_choose_policy_ada_16g() {
        let hw = fixture_hw_ada_16g();
        let c = geometry(1344, 768, None, None).unwrap();
        let pol = choose(
            &hw,
            None,
            None,
            Some("auto"),
            None,
            None,
            None,
            Some(&c),
            None,
            false,
            Some("generation"),
            0,
            0,
        )
        .unwrap();

        assert_eq!(pol.engine.ff_chunk, 2048);
        assert_eq!(pol.engine.projection_chunk, 1024);
        assert!(pol.engine.resident_blocks > 0);
        assert_eq!(pol.engine.window_batch, 4);
    }
}
