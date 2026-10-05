//! Two-pass sampling plan and latent upscaler configuration.

use crate::geometry::Canvas;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum TwoPassError {
    #[error("First-pass steps must be an integer between 1 and 32")]
    InvalidBaseSteps,
    #[error("Second-pass steps must be an integer between 1 and 31")]
    InvalidRefineSteps,
    #[error("Second-pass steps must be fewer than first-pass steps")]
    RefineExceedsBase,
    #[error("Invalid sampling step plan")]
    InvalidPlan,
}

pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub fn validate_steps(
    base_steps: usize,
    refine_steps: usize,
    enabled: bool,
) -> Result<(), TwoPassError> {
    if !(1..=32).contains(&base_steps) {
        return Err(TwoPassError::InvalidBaseSteps);
    }
    if !(1..=31).contains(&refine_steps) {
        return Err(TwoPassError::InvalidRefineSteps);
    }
    if enabled && refine_steps >= base_steps {
        return Err(TwoPassError::RefineExceedsBase);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Crop {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingPlan {
    pub version: u32,
    pub requested: bool,
    pub enabled: bool,
    pub mode: String,
    pub base_steps: usize,
    pub refine_steps: usize,
    pub total_steps: usize,
    pub first: Shape,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second: Option<Shape>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscale_target: Option<Shape>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop: Option<Crop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart_seed_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscaler_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_policy: Option<String>,
    pub reason: String,
}

pub fn plan(
    canvas: &Canvas,
    enabled: bool,
    base_steps: usize,
    refine_steps: usize,
) -> Result<SamplingPlan, TwoPassError> {
    validate_steps(base_steps, refine_steps, enabled)?;

    let shape = Shape {
        width: canvas.width,
        height: canvas.height,
        frames: canvas.frames,
    };

    let default = base_steps == 8 && (!enabled || refine_steps == 2);
    if !enabled {
        return Ok(SamplingPlan {
            version: if default { 1 } else { 2 },
            requested: false,
            enabled: false,
            mode: format!("vdn{}", base_steps),
            base_steps,
            refine_steps: 0,
            total_steps: base_steps,
            first: shape,
            second: None,
            upscale_target: None,
            crop: None,
            restart_seed_offset: None,
            upscaler_sha256: None,
            audio_policy: None,
            reason: "Two-pass generation is disabled.".to_string(),
        });
    }

    let (first, lift) = if canvas.width.min(canvas.height) >= 512 {
        let f_w = canvas.width.div_ceil(64) * 32;
        let f_h = canvas.height.div_ceil(64) * 32;
        let first = Shape {
            width: f_w,
            height: f_h,
            frames: canvas.frames,
        };
        let lift = Shape {
            width: 2 * f_w,
            height: 2 * f_h,
            frames: canvas.frames,
        };
        (first, lift)
    } else {
        let w = canvas.width / 32;
        let h = canvas.height / 32;
        let divisor = gcd(w, h);
        let unit_w = w / divisor;
        let unit_h = h / divisor;
        let multiple = divisor
            .div_ceil(2)
            .max(8_u32.div_ceil(unit_w))
            .max(8_u32.div_ceil(unit_h));
        let first = Shape {
            width: 32 * unit_w * multiple,
            height: 32 * unit_h * multiple,
            frames: canvas.frames,
        };
        let lift = shape.clone();
        (first, lift)
    };

    let crop = Crop {
        left: (lift.width - canvas.width) / 2,
        top: (lift.height - canvas.height) / 2,
        width: canvas.width,
        height: canvas.height,
    };

    let upscaling = first != lift;
    let upscaler_hash = if upscaling {
        Some("043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2".to_string())
    } else {
        None
    };

    let reason = if default {
        "8 steps on a smaller canvas, learned latent upscale, then the original DMD8 schedule tail (2 steps).".to_string()
    } else if !upscaling {
        format!("Small canvas: {} steps and {} refinement steps at the same target size, without upscaling.", base_steps, refine_steps)
    } else {
        format!("{} steps on a smaller canvas, learned latent upscale, then the DMD schedule tail ({} steps).", base_steps, refine_steps)
    };

    Ok(SamplingPlan {
        version: if default { 1 } else { 2 },
        requested: true,
        enabled: true,
        mode: format!("vdn{}-lbh-tail{}", base_steps, refine_steps),
        base_steps,
        refine_steps,
        total_steps: base_steps + refine_steps,
        first,
        second: Some(shape),
        upscale_target: Some(lift),
        crop: Some(crop),
        restart_seed_offset: Some(1),
        upscaler_sha256: upscaler_hash,
        audio_policy: Some("first_pass_preserved_with_audio_clock_conditioning".to_string()),
        reason,
    })
}

pub fn steps_from_plan(plan: Option<&SamplingPlan>) -> Result<usize, TwoPassError> {
    match plan {
        None => Ok(8),
        Some(p) => {
            validate_steps(
                p.base_steps,
                if p.enabled { p.refine_steps } else { 2 },
                p.enabled,
            )?;
            let expected = if p.enabled {
                p.base_steps + p.refine_steps
            } else {
                p.base_steps
            };
            if p.version != 1 && p.version != 2 {
                return Err(TwoPassError::InvalidPlan);
            }
            if p.total_steps != expected {
                return Err(TwoPassError::InvalidPlan);
            }
            Ok(expected)
        }
    }
}

pub fn same_strategy(a: Option<&SamplingPlan>, b: Option<&SamplingPlan>) -> bool {
    let steps_a = match steps_from_plan(a) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let steps_b = match steps_from_plan(b) {
        Ok(s) => s,
        Err(_) => return false,
    };
    if steps_a != steps_b {
        return false;
    }
    let a_enabled = a.map(|p| p.enabled).unwrap_or(false);
    let b_enabled = b.map(|p| p.enabled).unwrap_or(false);
    if a_enabled || b_enabled {
        a == b
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::geometry;

    #[test]
    fn test_two_pass_plan_default() {
        let c = geometry(1344, 768, None, None).unwrap();
        let p = plan(&c, true, 8, 2).unwrap();
        assert!(p.enabled);
        assert_eq!(p.total_steps, 10);
        // (1344 + 63) / 64 * 32 = 21 * 32 = 672
        // (768 + 63) / 64 * 32 = 12 * 32 = 384 (Wait: 768 + 63 = 831; 831 / 64 = 12; 12 * 32 = 384)
        assert_eq!(p.first.width, 672);
        assert_eq!(p.first.height, 384);
        assert_eq!(p.upscale_target.as_ref().unwrap().width, 1344);
        assert_eq!(p.upscale_target.as_ref().unwrap().height, 768);
        assert_eq!(p.crop.as_ref().unwrap().left, 0);
    }
}
