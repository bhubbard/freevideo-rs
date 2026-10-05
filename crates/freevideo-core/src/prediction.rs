//! Local performance forecasting and memory headroom previews from verified request history.

use crate::geometry::Canvas;
use crate::history::AttemptRecord;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRange {
    pub lower: f64,
    pub estimate: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastReport {
    pub canvas: Canvas,
    pub observation_count: usize,
    pub stages: serde_json::Value,
    pub resources: serde_json::Value,
    pub note: String,
}

pub fn predict(observations: &[AttemptRecord], canvas: &Canvas) -> ForecastReport {
    let mut step_times: Vec<f64> = Vec::new();
    let mut gpu_peaks: Vec<u64> = Vec::new();
    let mut ram_peaks: Vec<u64> = Vec::new();

    for obs_record in observations {
        if let Some(ref obs) = obs_record.observation {
            if let Some(steps) = obs.get("step_seconds").and_then(|v| v.as_array()) {
                for s in steps {
                    if let Some(val) = s.as_f64() {
                        step_times.push(val);
                    }
                }
            }
            if let Some(p) = obs.get("whole_gpu_peak_bytes").and_then(|v| v.as_u64()) {
                gpu_peaks.push(p);
            }
            if let Some(r) = obs.get("ram_peak_bytes").and_then(|v| v.as_u64()) {
                ram_peaks.push(r);
            }
        }
    }

    let token_scale = canvas.video_tokens as f64 / 72576.0;

    let avg_step = if !step_times.is_empty() {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    } else {
        1.25 // Default nominal step duration in seconds for FP8 on modern GPU
    };

    let est_step = avg_step * token_scale;
    let total_steps = canvas.steps.unwrap_or(8) as f64;
    let sample_est = est_step * total_steps;

    let gpu_peak_est = if !gpu_peaks.is_empty() {
        gpu_peaks
            .iter()
            .copied()
            .max()
            .unwrap_or(12 * 1024 * 1024 * 1024)
    } else {
        12 * 1024 * 1024 * 1024
    };

    let ram_peak_est = if !ram_peaks.is_empty() {
        ram_peaks
            .iter()
            .copied()
            .max()
            .unwrap_or(8 * 1024 * 1024 * 1024)
    } else {
        8 * 1024 * 1024 * 1024
    };

    let stages = serde_json::json!({
        "step_seconds": {
            "lower": est_step * 0.9,
            "estimate": est_step,
            "upper": est_step * 1.15
        },
        "sample_seconds": {
            "lower": sample_est * 0.9,
            "estimate": sample_est,
            "upper": sample_est * 1.15
        },
        "load_seconds": {
            "lower": 2.0,
            "estimate": 3.5,
            "upper": 6.0
        },
        "video_decode_seconds": {
            "lower": 4.0 * token_scale,
            "estimate": 6.5 * token_scale,
            "upper": 10.0 * token_scale
        },
        "work_seconds": {
            "lower": (sample_est + 10.0) * 0.9,
            "estimate": sample_est + 10.0,
            "upper": (sample_est + 10.0) * 1.15
        }
    });

    let resources = serde_json::json!({
        "whole_gpu_peak_bytes": {
            "lower": (gpu_peak_est as f64 * 0.95) as u64,
            "estimate": gpu_peak_est,
            "upper": (gpu_peak_est as f64 * 1.05) as u64
        },
        "ram_peak_bytes": {
            "lower": (ram_peak_est as f64 * 0.95) as u64,
            "estimate": ram_peak_est,
            "upper": (ram_peak_est as f64 * 1.10) as u64
        }
    });

    ForecastReport {
        canvas: canvas.clone(),
        observation_count: observations.len(),
        stages,
        resources,
        note: "Local verified request history regression forecast; no GPU computation was started."
            .to_string(),
    }
}
