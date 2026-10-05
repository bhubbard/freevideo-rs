//! System diagnostics and capability readiness checking.

use crate::hardware::{detect, installed_backends, Hardware};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathStatus {
    pub path: String,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub ready: bool,
    pub hardware: Hardware,
    pub installed_attention_packages: Vec<String>,
    pub paths: HashMap<String, PathStatus>,
    pub summary: String,
    pub validation: String,
}

pub fn default_paths() -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    let home = std::env::var("FREEVIDEO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    map.insert("vdn".to_string(), home.join("vdn"));
    map.insert("base".to_string(), home.join("models/h3-base"));
    map.insert("checkpoint".to_string(), home.join("models/linear_branch"));
    map.insert("comfy".to_string(), home.join("ComfyUI"));
    map
}

pub fn run_doctor(
    custom_paths: Option<HashMap<String, PathBuf>>,
    require_paths: bool,
) -> DoctorReport {
    let hw = detect();
    let backends = installed_backends();
    let paths_to_check = custom_paths.unwrap_or_else(default_paths);

    let mut path_statuses = HashMap::new();
    let mut all_paths_exist = true;

    for (k, p) in paths_to_check {
        let exists = p.exists();
        if !exists {
            all_paths_exist = false;
        }
        path_statuses.insert(
            k,
            PathStatus {
                path: p.to_string_lossy().to_string(),
                exists,
            },
        );
    }

    let ready = !require_paths || all_paths_exist;
    let summary = if ready {
        "All diagnostic checks passed. System ready for FreeVideo execution.".to_string()
    } else {
        "Some model directories or dependencies are not yet installed or mapped.".to_string()
    };

    DoctorReport {
        ready,
        hardware: hw,
        installed_attention_packages: backends,
        paths: path_statuses,
        summary,
        validation:
            "Package and path discovery only. Run benchmark to measure actual device throughput."
                .to_string(),
    }
}
