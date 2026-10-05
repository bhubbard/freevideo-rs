//! Operating system, memory headrooms, and platform adapters.

use crate::geometry::Canvas;
use crate::hardware::GIB;

pub const HOST_WEIGHT_HEADROOM: u64 = 8 * GIB;

pub fn residual_host_headroom(canvas: Option<&Canvas>) -> u64 {
    let rows = if let Some(c) = canvas {
        (c.video_tokens
            + c.reference_video_tokens.unwrap_or(0)
            + c.reference_audio_tokens.unwrap_or(0)) as f64
    } else {
        72576.0
    };
    (GIB as f64 * (rows / 72576.0).max(1.0)) as u64
}

pub fn weight_cache_headroom(
    system: Option<&str>,
    streamed: bool,
    cpu_outputs: bool,
    canvas: Option<&Canvas>,
) -> u64 {
    let sys = system.unwrap_or(if cfg!(target_os = "windows") {
        "Windows"
    } else {
        "Linux"
    });

    if sys != "Linux" && sys != "Windows" || !streamed {
        return HOST_WEIGHT_HEADROOM;
    }

    if !cpu_outputs {
        return (3.5 * GIB as f64) as u64;
    }

    let c = match canvas {
        Some(canvas) => canvas,
        None => return HOST_WEIGHT_HEADROOM,
    };

    let reference_tokens = (72 * 1008) as f64;
    let growth = ((c.video_tokens as f64 / reference_tokens) - 1.0).max(0.0);
    let base = if sys == "Windows" {
        HOST_WEIGHT_HEADROOM as f64 / GIB as f64
    } else {
        5.0
    };

    ((base + 2.5 * growth) * GIB as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::geometry;

    #[test]
    fn test_residual_host_headroom() {
        let c = geometry(1344, 768, None, None).unwrap();
        assert_eq!(residual_host_headroom(Some(&c)), GIB);
    }

    #[test]
    fn test_weight_cache_headroom() {
        let c = geometry(1344, 768, None, None).unwrap();
        // streamed, Linux, cpu_outputs = false -> 3.5 GiB
        assert_eq!(
            weight_cache_headroom(Some("Linux"), true, false, Some(&c)),
            (3.5 * GIB as f64) as u64
        );
        // streamed, Linux, cpu_outputs = true, default canvas -> 5.0 GiB
        assert_eq!(
            weight_cache_headroom(Some("Linux"), true, true, Some(&c)),
            5 * GIB
        );
    }
}
