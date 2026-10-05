//! FreeVideo Core Library
//! High-performance adaptive video generation planner, safetensors indexer,
//! hardware-aware memory allocation, and historical telemetry store.

#![allow(clippy::too_many_arguments, clippy::result_large_err)]

pub mod benchmark;
pub mod doctor;
pub mod execution_config;
pub mod geometry;
pub mod hardware;
pub mod history;
pub mod models;
pub mod policy;
pub mod prediction;
pub mod prediction_policy;
pub mod system;
pub mod two_pass;
pub mod weights;

pub use benchmark::{simulate_capacity_grid, BenchmarkGridReport, BenchmarkTierResult};
pub use doctor::{run_doctor, DoctorReport};
pub use execution_config::{decoder_options, engine_options, DecoderConfig, EngineConfig};
pub use geometry::{geometry, Canvas, GeometryError};
pub use hardware::{detect, Hardware, GIB, GIB_F64};
pub use history::{AttemptRecord, ResourceHistory, ResourceHistoryError};
pub use models::{
    parse_model_files, parse_prepared_models, verify_files, ModelFileEntry, PreparedModelManifest,
};
pub use policy::{choose, resource_budget, Policy, ResourceBudget, ResourceBudgetError};
pub use prediction::{predict, ForecastReport};
pub use prediction_policy::select_candidate;
pub use two_pass::{plan as plan_two_pass, steps_from_plan, SamplingPlan};
pub use weights::{group_name, inspect_safetensors, interleaved_layer_order, sha256_file};
