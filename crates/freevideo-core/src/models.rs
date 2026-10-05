//! Model manifest catalog and weight integrity validation.

use crate::weights::sha256_file;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileEntry {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_blob: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedVariant {
    pub cache_prefix: String,
    pub files: Vec<PreparedFileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedFileEntry {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedModelManifest {
    pub schema_version: u32,
    pub repo: String,
    pub revision: String,
    pub format: String,
    pub variants: HashMap<String, PreparedVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatus {
    pub total_files: usize,
    pub verified_files: usize,
    pub missing_files: Vec<String>,
    pub mismatched_files: Vec<String>,
    pub complete: bool,
}

pub fn parse_model_files(json_str: &str) -> Result<Vec<ModelFileEntry>, serde_json::Error> {
    serde_json::from_str(json_str)
}

pub fn parse_prepared_models(json_str: &str) -> Result<PreparedModelManifest, serde_json::Error> {
    serde_json::from_str(json_str)
}

pub fn verify_files<P: AsRef<Path>>(
    root: P,
    entries: &[ModelFileEntry],
    check_sha: bool,
) -> VerificationStatus {
    let base = root.as_ref();
    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    let mut verified = 0;

    for entry in entries {
        let p: PathBuf = base.join(&entry.file);
        if !p.is_file() {
            missing.push(entry.file.clone());
            continue;
        }

        if let Ok(meta) = p.metadata() {
            if meta.len() != entry.bytes {
                mismatched.push(format!(
                    "{}: size mismatch ({} != {})",
                    entry.file,
                    meta.len(),
                    entry.bytes
                ));
                continue;
            }
        } else {
            missing.push(entry.file.clone());
            continue;
        }

        if check_sha {
            if let Some(expected_sha) = &entry.sha256 {
                if let Ok(actual_sha) = sha256_file(&p) {
                    if &actual_sha != expected_sha {
                        mismatched.push(format!("{}: sha256 mismatch", entry.file));
                        continue;
                    }
                } else {
                    mismatched.push(format!("{}: could not read for hashing", entry.file));
                    continue;
                }
            }
        }

        verified += 1;
    }

    let complete = missing.is_empty() && mismatched.is_empty();
    VerificationStatus {
        total_files: entries.len(),
        verified_files: verified,
        missing_files: missing,
        mismatched_files: mismatched,
        complete,
    }
}
