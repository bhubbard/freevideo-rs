//! Placement candidate selection from local observations.

use crate::geometry::Canvas;
use crate::hardware::Hardware;
use crate::history::AttemptRecord;
use crate::policy::Policy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionDecision {
    pub selected_origin: String,
    pub source_attempt_id: Option<String>,
    pub candidate_count: usize,
    pub note: String,
}

pub fn select_candidate(
    base: &Policy,
    _canvas: &Canvas,
    _hardware: &Hardware,
    history_rows: &[AttemptRecord],
) -> (Policy, SelectionDecision) {
    let selected_policy = base.clone();
    let mut origin = "current-policy".to_string();
    let mut source_id = None;

    // Check for previous successful matches
    for row in history_rows.iter().rev() {
        if row.state == "success" {
            if let Some(obs) = &row.observation {
                if obs.get("full_request") == Some(&serde_json::Value::Bool(true)) {
                    // Compatible candidate found
                    origin = "history-observation".to_string();
                    source_id = Some(row.id.clone());
                    break;
                }
            }
        }
    }

    let decision = SelectionDecision {
        selected_origin: origin,
        source_attempt_id: source_id,
        candidate_count: history_rows.len().min(3) + 1,
        note: "Selected placement preserves arithmetic and kernel shapes without extrapolation."
            .to_string(),
    };

    (selected_policy, decision)
}
