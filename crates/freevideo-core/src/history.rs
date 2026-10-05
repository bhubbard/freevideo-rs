//! Local SQLite resource measurement history (schema version 6).
//! Tracks verified runs, placement decisions, and calibration metrics without importing CUDA.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use uuid::Uuid;

pub const SCHEMA_VERSION: i64 = 6;

#[derive(Error, Debug)]
pub enum ResourceHistoryError {
    #[error("Resource measurement history is unavailable: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub id: String,
    pub computation_key: String,
    pub evidence_key: String,
    pub state: String,
    pub outcome: String,
    pub started: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settled: Option<f64>,
    pub identity: serde_json::Value,
    pub config: serde_json::Value,
    pub geometry: serde_json::Value,
    pub owner: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<serde_json::Value>,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    pub details: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<serde_json::Value>,
}

pub fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut pairs = Vec::new();
            for k in keys {
                pairs.push(format!(
                    "{}:{}",
                    serde_json::to_string(k).unwrap(),
                    canonical_json(&map[k])
                ));
            }
            format!("{{{}}}", pairs.join(","))
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", items.join(","))
        }
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}

pub fn json_digest(value: &serde_json::Value) -> String {
    let canonical = canonical_json(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn default_history_path() -> PathBuf {
    if let Ok(val) = std::env::var("FREEVIDEO_HISTORY_PATH") {
        return PathBuf::from(val);
    }
    let home = std::env::var("FREEVIDEO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                std::env::var("LOCALAPPDATA")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join("FreeVideo")
            } else {
                std::env::var("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(".local/share/freevideo")
            }
        });
    home.join("resource-history.sqlite3")
}

pub struct ResourceHistory {
    pub path: PathBuf,
}

impl ResourceHistory {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, ResourceHistoryError> {
        let p = path.as_ref().to_path_buf();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ResourceHistoryError::Custom(e.to_string()))?;
        }

        let s = Self { path: p };
        s.init_db()?;
        Ok(s)
    }

    fn open(&self) -> Result<Connection, rusqlite::Error> {
        let conn = Connection::open(&self.path)?;
        conn.execute_batch(
            "PRAGMA secure_delete = ON;
             PRAGMA synchronous = FULL;
             PRAGMA journal_mode = DELETE;
             PRAGMA foreign_keys = ON;",
        )?;
        Ok(conn)
    }

    fn init_db(&self) -> Result<(), ResourceHistoryError> {
        let conn = self.open()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (version INTEGER NOT NULL)",
            [],
        )?;

        let mut stmt = conn.prepare("SELECT version FROM metadata")?;
        let versions: Vec<i64> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(Result::ok)
            .collect();

        if versions.is_empty() {
            conn.execute(
                "INSERT INTO metadata (version) VALUES (?)",
                params![SCHEMA_VERSION],
            )?;
        } else if versions[0] != SCHEMA_VERSION {
            conn.execute("UPDATE metadata SET version = ?", params![SCHEMA_VERSION])?;
        }

        conn.execute(
            "CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY,
                computation_key TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                state TEXT NOT NULL,
                started REAL NOT NULL,
                settled REAL,
                identity_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                geometry_json TEXT NOT NULL,
                owner_json TEXT NOT NULL,
                worker_json TEXT,
                purpose TEXT NOT NULL,
                phase TEXT,
                details_json TEXT NOT NULL,
                observation_json TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS attempts_computation ON attempts(computation_key, state)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS attempts_observations ON attempts(state, identity_json, started)",
            [],
        )?;

        Ok(())
    }

    pub fn begin(
        &self,
        identity: &serde_json::Value,
        config: &serde_json::Value,
        geometry: &serde_json::Value,
        purpose: &str,
        owner: Option<&serde_json::Value>,
        evidence: Option<&serde_json::Value>,
    ) -> Result<String, ResourceHistoryError> {
        let conn = self.open()?;
        let id = Uuid::new_v4().simple().to_string();

        let default_owner = serde_json::json!({
            "pid": std::process::id(),
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
        });
        let owner_val = owner.unwrap_or(&default_owner);

        let computation_key = json_digest(&serde_json::json!({
            "identity": identity,
            "config": config,
            "geometry": geometry,
            "purpose": purpose
        }));

        let evidence_key = json_digest(&serde_json::json!({
            "identity": identity,
            "config": config,
            "geometry": geometry
        }));

        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let details = serde_json::json!({
            "decision_evidence": evidence.unwrap_or(&serde_json::json!({}))
        });

        conn.execute(
            "INSERT INTO attempts (
                id, computation_key, evidence_key, state, started,
                identity_json, config_json, geometry_json, owner_json,
                purpose, details_json
            ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
            params![
                id,
                computation_key,
                evidence_key,
                started,
                serde_json::to_string(identity)?,
                serde_json::to_string(config)?,
                serde_json::to_string(geometry)?,
                serde_json::to_string(owner_val)?,
                purpose,
                serde_json::to_string(&details)?
            ],
        )?;

        Ok(id)
    }

    pub fn attach_worker(
        &self,
        attempt_id: &str,
        pid: u32,
        phase: Option<&str>,
    ) -> Result<(), ResourceHistoryError> {
        let conn = self.open()?;
        let worker_val = serde_json::json!({
            "pid": pid,
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
        });

        let changed = conn.execute(
            "UPDATE attempts SET worker_json = ?, phase = COALESCE(?, phase) WHERE id = ? AND state = 'pending'",
            params![serde_json::to_string(&worker_val)?, phase, attempt_id],
        )?;

        if changed != 1 {
            return Err(ResourceHistoryError::Custom(
                "Cannot attach a worker to an absent/settled attempt".to_string(),
            ));
        }
        Ok(())
    }

    pub fn set_phase(&self, attempt_id: &str, phase: &str) -> Result<(), ResourceHistoryError> {
        let conn = self.open()?;
        let changed = conn.execute(
            "UPDATE attempts SET phase = ? WHERE id = ? AND state = 'pending'",
            params![phase, attempt_id],
        )?;
        if changed != 1 {
            return Err(ResourceHistoryError::Custom(
                "Cannot update an absent/settled attempt".to_string(),
            ));
        }
        Ok(())
    }

    pub fn finish(
        &self,
        attempt_id: &str,
        outcome: &str,
        details: Option<&serde_json::Value>,
        observation: Option<&serde_json::Value>,
    ) -> Result<AttemptRecord, ResourceHistoryError> {
        let conn = self.open()?;
        let settled = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let details_json_str = details
            .map(|d| serde_json::to_string(d).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());
        let obs_json_str =
            observation.map(|o| serde_json::to_string(o).unwrap_or_else(|_| "null".to_string()));

        let changed = conn.execute(
            "UPDATE attempts SET state = ?, settled = ?, details_json = ?, observation_json = ? WHERE id = ? AND state = 'pending'",
            params![outcome, settled, details_json_str, obs_json_str, attempt_id],
        )?;

        if changed != 1 {
            return Err(ResourceHistoryError::Custom(
                "Cannot settle an absent/already settled attempt".to_string(),
            ));
        }

        self.attempt(attempt_id)?.ok_or_else(|| {
            ResourceHistoryError::Custom("Attempt record missing after update".to_string())
        })
    }

    pub fn recover_pending(&self) -> Result<Vec<String>, ResourceHistoryError> {
        let conn = self.open()?;
        let mut stmt = conn.prepare("SELECT id FROM attempts WHERE state = 'pending'")?;
        let pending_ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(Result::ok)
            .collect();

        for id in &pending_ids {
            conn.execute("DELETE FROM attempts WHERE id = ?", params![id])?;
        }

        Ok(pending_ids)
    }

    pub fn attempt(&self, identifier: &str) -> Result<Option<AttemptRecord>, ResourceHistoryError> {
        let conn = self.open()?;
        let mut stmt = conn.prepare(
            "SELECT id, computation_key, evidence_key, state, started, settled,
                    identity_json, config_json, geometry_json, owner_json, worker_json,
                    purpose, phase, details_json, observation_json
             FROM attempts WHERE id = ?",
        )?;

        let mut rows = stmt.query(params![identifier])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Self::parse_row(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn attempts(&self) -> Result<Vec<AttemptRecord>, ResourceHistoryError> {
        let conn = self.open()?;
        let mut stmt = conn.prepare(
            "SELECT id, computation_key, evidence_key, state, started, settled,
                    identity_json, config_json, geometry_json, owner_json, worker_json,
                    purpose, phase, details_json, observation_json
             FROM attempts ORDER BY started ASC",
        )?;

        let list = stmt
            .query_map([], Self::parse_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(list)
    }

    pub fn observations(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<AttemptRecord>, ResourceHistoryError> {
        let conn = self.open()?;
        let query = if let Some(lim) = limit {
            format!(
                "SELECT id, computation_key, evidence_key, state, started, settled,
                        identity_json, config_json, geometry_json, owner_json, worker_json,
                        purpose, phase, details_json, observation_json
                 FROM attempts WHERE state = 'success' AND observation_json IS NOT NULL
                 ORDER BY started DESC LIMIT {}",
                lim
            )
        } else {
            "SELECT id, computation_key, evidence_key, state, started, settled,
                    identity_json, config_json, geometry_json, owner_json, worker_json,
                    purpose, phase, details_json, observation_json
             FROM attempts WHERE state = 'success' AND observation_json IS NOT NULL
             ORDER BY started ASC"
                .to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let list = stmt
            .query_map([], Self::parse_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(list)
    }

    fn parse_row(row: &rusqlite::Row) -> Result<AttemptRecord, rusqlite::Error> {
        let id: String = row.get(0)?;
        let computation_key: String = row.get(1)?;
        let evidence_key: String = row.get(2)?;
        let state: String = row.get(3)?;
        let started: f64 = row.get(4)?;
        let settled: Option<f64> = row.get(5)?;
        let identity_str: String = row.get(6)?;
        let config_str: String = row.get(7)?;
        let geometry_str: String = row.get(8)?;
        let owner_str: String = row.get(9)?;
        let worker_str: Option<String> = row.get(10)?;
        let purpose: String = row.get(11)?;
        let phase: Option<String> = row.get(12)?;
        let details_str: String = row.get(13)?;
        let obs_str: Option<String> = row.get(14)?;

        Ok(AttemptRecord {
            id,
            computation_key,
            evidence_key,
            state: state.clone(),
            outcome: state,
            started,
            settled,
            identity: serde_json::from_str(&identity_str).unwrap_or(serde_json::Value::Null),
            config: serde_json::from_str(&config_str).unwrap_or(serde_json::Value::Null),
            geometry: serde_json::from_str(&geometry_str).unwrap_or(serde_json::Value::Null),
            owner: serde_json::from_str(&owner_str).unwrap_or(serde_json::Value::Null),
            worker: worker_str.and_then(|s| serde_json::from_str(&s).ok()),
            purpose,
            phase,
            details: serde_json::from_str(&details_str).unwrap_or(serde_json::Value::Null),
            observation: obs_str.and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_resource_history_lifecycle() {
        let tmp = NamedTempFile::new().unwrap();
        let rh = ResourceHistory::new(tmp.path()).unwrap();

        let identity = serde_json::json!({"gpu_uuid": "gpu-0", "system": "Linux"});
        let config = serde_json::json!({"engine": {"steps": 8}});
        let geometry = serde_json::json!({"width": 1344, "height": 768, "frames": 243});

        let id = rh
            .begin(&identity, &config, &geometry, "generation", None, None)
            .unwrap();
        assert!(!id.is_empty());

        rh.set_phase(&id, "sampling").unwrap();

        let obs = serde_json::json!({
            "full_request": true,
            "step_seconds": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "peak_reserved_bytes": 10000000000u64,
            "whole_gpu_peak_bytes": 11000000000u64,
            "ram_peak_bytes": 8000000000u64,
            "stage_seconds": {"sample_seconds": 8.0},
            "ram_metric": "resident_peak"
        });

        let finished = rh.finish(&id, "success", None, Some(&obs)).unwrap();
        assert_eq!(finished.state, "success");

        let all = rh.attempts().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);

        let obs_records = rh.observations(None).unwrap();
        assert_eq!(obs_records.len(), 1);
    }
}
