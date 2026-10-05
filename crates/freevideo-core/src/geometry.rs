//! Request geometry without importing CUDA; follows the pinned H3 VAE grid.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum GeometryError {
    #[error("Width and height must be multiples of 32, at least 256 pixels.")]
    InvalidDimensions,
    #[error("Specify frames or seconds, not both.")]
    ConflictingDuration,
    #[error("Duration must be a positive finite number.")]
    InvalidSeconds,
    #[error("Frame count must be a positive integer.")]
    InvalidFrames,
    #[error("H3 hybrid window inference needs at least 39 aligned frames (1.625 seconds).")]
    TooFewFrames,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub requested_frames: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_seconds: Option<f64>,
    pub frames: u32,
    pub seconds: f64,
    pub latent_frames: u32,
    pub video_tokens: u32,
    pub alignment: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_plan: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_video_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
}

impl Canvas {
    pub fn new(
        width: u32,
        height: u32,
        frames: Option<u32>,
        seconds: Option<f64>,
    ) -> Result<Self, GeometryError> {
        if width < 256 || !width.is_multiple_of(32) || height < 256 || !height.is_multiple_of(32) {
            return Err(GeometryError::InvalidDimensions);
        }
        if frames.is_some() && seconds.is_some() {
            return Err(GeometryError::ConflictingDuration);
        }
        if let Some(sec) = seconds {
            if !sec.is_finite() || sec <= 0.0 {
                return Err(GeometryError::InvalidSeconds);
            }
        }

        let requested: u32 = if let Some(sec) = seconds {
            (sec * 24.0).ceil() as u32
        } else {
            frames.unwrap_or(243)
        };

        if requested < 1 {
            return Err(GeometryError::InvalidFrames);
        }

        let req_i64 = requested as i64;
        let rem = (5 - req_i64).rem_euclid(17);
        let aligned = (req_i64 + rem) as u32;

        let latent_frames = ((aligned as i64 - 5) / 17 * 5 + 2) as u32;
        if latent_frames < 12 {
            return Err(GeometryError::TooFewFrames);
        }

        let video_tokens = latent_frames * (height / 32) * (width / 32);

        Ok(Self {
            width,
            height,
            fps: 24,
            requested_frames: requested,
            requested_seconds: seconds,
            frames: aligned,
            seconds: aligned as f64 / 24.0,
            latent_frames,
            video_tokens,
            alignment: "Pinned H3 VAE: round up to 17*n+5 frames; no resampling or hidden resolution reduction.".to_string(),
            steps: None,
            task: None,
            sampling_plan: None,
            reference_video_tokens: None,
            reference_audio_tokens: None,
            text_tokens: None,
        })
    }

    pub fn total_tokens(&self) -> u32 {
        self.video_tokens
            + self.reference_video_tokens.unwrap_or(0)
            + self.reference_audio_tokens.unwrap_or(0)
    }
}

pub fn geometry(
    width: u32,
    height: u32,
    frames: Option<u32>,
    seconds: Option<f64>,
) -> Result<Canvas, GeometryError> {
    Canvas::new(width, height, frames, seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_geometry() {
        let c = geometry(1344, 768, None, None).unwrap();
        assert_eq!(c.width, 1344);
        assert_eq!(c.height, 768);
        assert_eq!(c.frames, 243);
        assert_eq!(c.latent_frames, 72);
        // (1344 / 32) * (768 / 32) = 42 * 24 = 1008
        // 72 * 1008 = 72576
        assert_eq!(c.video_tokens, 72576);
    }

    #[test]
    fn test_aligned_frame_rounding() {
        let c1 = geometry(512, 512, Some(90), None).unwrap();
        assert_eq!(c1.frames, 90); // 90 is 17 * 5 + 5
        assert_eq!(c1.latent_frames, 27); // (90 - 5) / 17 * 5 + 2 = 27

        let c2 = geometry(512, 512, Some(100), None).unwrap();
        assert_eq!(c2.frames, 107); // 100 + (5 - 100) % 17 = 107
        assert_eq!(c2.latent_frames, 32);
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(geometry(1000, 768, None, None).is_err()); // not div by 32
        assert!(geometry(128, 512, None, None).is_err()); // < 256
    }

    #[test]
    fn test_too_few_frames() {
        assert!(geometry(512, 512, Some(20), None).is_err()); // latent_frames < 12
    }
}
