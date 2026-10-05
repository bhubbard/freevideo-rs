use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use colored::*;
use freevideo_core::benchmark::simulate_capacity_grid;
use freevideo_core::doctor::run_doctor;
use freevideo_core::geometry::geometry;
use freevideo_core::hardware::{detect, Hardware};
use freevideo_core::history::{default_history_path, ResourceHistory};
use freevideo_core::models::{parse_model_files, parse_prepared_models, verify_files};
use freevideo_core::policy::choose;
use freevideo_core::prediction::predict;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "freevideo",
    about = "FreeVideo Engine: Video and audio generation with bounded memory and hardware-aware adaptive planning",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show a resource policy without loading models
    Plan(PlanArgs),
    /// Preview phase time and memory from local request history
    Predict(PredictArgs),
    /// Check paths, versions and capability readiness
    Doctor(DoctorArgs),
    /// Inspect or verify model manifests and prepared weights
    Models(ModelsArgs),
    /// Inspect local performance and resource measurements
    #[command(alias = "resource-history")]
    History(HistoryArgs),
    /// Benchmark capacity grid and compare attention backends
    Bench(BenchArgs),
    /// Preview or execute video generation with adaptive policy
    Generate(GenerateArgs),
}

#[derive(Args)]
struct ResourceArgs {
    /// Nominal capacity, bounded by detected VRAM (in GiB)
    #[arg(long)]
    vram_gib: Option<f64>,

    /// Nominal system RAM, bounded by detected RAM (in GiB)
    #[arg(long)]
    ram_gib: Option<f64>,

    /// Explicit foreground VRAM growth reserve in GiB (minimum 0.2 GiB)
    #[arg(long)]
    gpu_reserve_gib: Option<f64>,

    /// Override headroom kept beyond current system RAM usage (in GiB)
    #[arg(long)]
    ram_reserve_gib: Option<f64>,

    /// Attention backend: auto, sage2, cudnn, torch-flash, fa2, fa4, or global/window
    #[arg(long, default_value = "auto")]
    attention: String,
}

#[derive(Args)]
struct PlanArgs {
    #[command(flatten)]
    resource: ResourceArgs,

    /// Inspect a fixture JSON instead of live hardware probing
    #[arg(long)]
    hardware_json: Option<PathBuf>,

    /// Save plan JSON to path
    #[arg(long)]
    out: Option<PathBuf>,

    /// Canvas width (multiple of 32, min 256)
    #[arg(long, default_value_t = 1344)]
    width: u32,

    /// Canvas height (multiple of 32, min 256)
    #[arg(long, default_value_t = 768)]
    height: u32,

    /// Video frame count (aligned to 17*n + 5)
    #[arg(long)]
    frames: Option<u32>,

    /// Video duration in seconds
    #[arg(long)]
    seconds: Option<f64>,
}

#[derive(Args)]
struct PredictArgs {
    /// Canvas width
    #[arg(long, default_value_t = 1344)]
    width: u32,

    /// Canvas height
    #[arg(long, default_value_t = 768)]
    height: u32,

    /// Frame count
    #[arg(long)]
    frames: Option<u32>,

    /// Duration in seconds
    #[arg(long)]
    seconds: Option<f64>,

    /// Path to history SQLite database
    #[arg(long)]
    history: Option<PathBuf>,

    /// Save prediction JSON to path
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct DoctorArgs {
    /// Fail if model or source directories are missing
    #[arg(long)]
    require_paths: bool,

    /// Save complete doctor report JSON to path
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct ModelsArgs {
    #[command(subcommand)]
    sub: ModelsSubcommand,
}

#[derive(Subcommand)]
enum ModelsSubcommand {
    /// List official and prepared models supported by FreeVideo
    List,
    /// Verify file integrity and hashes of a model directory
    Verify {
        /// Path to manifest file (model_files.json or prepared_models.json)
        #[arg(long)]
        manifest: PathBuf,

        /// Root directory containing the model files
        #[arg(long)]
        root: PathBuf,

        /// Verify SHA256 hashes in addition to file sizes
        #[arg(long)]
        check_sha: bool,
    },
}

#[derive(Args)]
struct HistoryArgs {
    #[command(subcommand)]
    sub: HistorySubcommand,
}

#[derive(Subcommand)]
enum HistorySubcommand {
    /// Show performance measurements and count complete observations
    List {
        #[arg(long)]
        history: Option<PathBuf>,
    },
    /// Settle known cancellations and discard interrupted incomplete runs
    Recover {
        #[arg(long)]
        history: Option<PathBuf>,
    },
}

#[derive(Args)]
struct BenchArgs {
    /// Canvas width
    #[arg(long, default_value_t = 1344)]
    width: u32,

    /// Canvas height
    #[arg(long, default_value_t = 768)]
    height: u32,

    /// Frame count
    #[arg(long, default_value_t = 243)]
    frames: u32,

    /// Save report JSON to path
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    resource: ResourceArgs,

    /// Text prompt file
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// Precomputed conditioning file
    #[arg(long)]
    conditioning: Option<PathBuf>,

    /// Output MP4 path
    #[arg(long)]
    out: PathBuf,

    /// Canvas width
    #[arg(long, default_value_t = 1344)]
    width: u32,

    /// Canvas height
    #[arg(long, default_value_t = 768)]
    height: u32,

    /// Frame count
    #[arg(long)]
    frames: Option<u32>,

    /// Duration in seconds
    #[arg(long)]
    seconds: Option<f64>,

    /// First-pass steps (1..32)
    #[arg(long, default_value_t = 8)]
    base_steps: usize,

    /// Second-pass refinement steps (1..31)
    #[arg(long, default_value_t = 2)]
    refine_steps: usize,

    /// Enable two-pass latent upscaling (default: true)
    #[arg(long, default_value_t = true)]
    two_pass: bool,
}

fn write_json_output<T: serde::Serialize>(val: &T, out_path: Option<&Path>) -> Result<()> {
    let text = serde_json::to_string_pretty(val)? + "\n";
    if let Some(p) = out_path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(p, &text)?;
        eprintln!("{} {}", "Saved output to:".green(), p.display());
    }
    print!("{}", text);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Plan(args) => {
            let canvas = geometry(args.width, args.height, args.frames, args.seconds)
                .context("Invalid geometry parameters")?;

            let hardware = if let Some(hw_path) = args.hardware_json {
                let text = fs::read_to_string(&hw_path).with_context(|| {
                    format!("Failed to read hardware JSON: {}", hw_path.display())
                })?;
                Hardware::from_json_str(&text).context("Failed to parse hardware JSON fixture")?
            } else {
                detect()
            };

            let policy = choose(
                &hardware,
                args.resource.vram_gib,
                args.resource.ram_gib,
                Some(&args.resource.attention),
                args.resource.gpu_reserve_gib,
                args.resource.ram_reserve_gib,
                None,
                Some(&canvas),
                None,
                false,
                Some("generation"),
                0,
                0,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;

            write_json_output(&policy, args.out.as_deref())?;
        }
        Commands::Predict(args) => {
            let canvas = geometry(args.width, args.height, args.frames, args.seconds)
                .context("Invalid geometry parameters")?;

            let db_path = args.history.unwrap_or_else(default_history_path);
            let observations = if db_path.exists() {
                let history = ResourceHistory::new(&db_path)?;
                history.observations(Some(256))?
            } else {
                Vec::new()
            };

            let report = predict(&observations, &canvas);
            write_json_output(&report, args.out.as_deref())?;
        }
        Commands::Doctor(args) => {
            let report = run_doctor(None, args.require_paths);
            write_json_output(&report, args.out.as_deref())?;
            if args.require_paths && !report.ready {
                std::process::exit(1);
            }
        }
        Commands::Models(args) => match args.sub {
            ModelsSubcommand::List => {
                println!(
                    "{}",
                    "FreeVideo Supported Models & Architecture Manifests"
                        .bold()
                        .green()
                );
                println!("  1. OpenVDN / vdn-minimax-h3 (Base BF16 + LoRA checkpoint)");
                println!("  2. OpenVDN / vdn-minimax-h3-edge (Prepared FP8 per-tensor & rowwise)");
                println!("  3. LBH-123-AI / Minimax_h3_latent_Upscaler (3D Conv Latent Upscaler)");
                println!("  4. Qwen / Qwen3-VL-32B (Text conditioning encoder)");
            }
            ModelsSubcommand::Verify {
                manifest,
                root,
                check_sha,
            } => {
                let text = fs::read_to_string(&manifest)
                    .with_context(|| format!("Failed to read manifest: {}", manifest.display()))?;

                if let Ok(entries) = parse_model_files(&text) {
                    let status = verify_files(&root, &entries, check_sha);
                    println!(
                        "{}",
                        format!(
                            "Verified {}/{} files",
                            status.verified_files, status.total_files
                        )
                        .cyan()
                    );
                    if !status.missing_files.is_empty() {
                        println!(
                            "{} {} missing files",
                            "Warning:".yellow(),
                            status.missing_files.len()
                        );
                        for f in status.missing_files.iter().take(5) {
                            println!("  - {}", f);
                        }
                    }
                    if !status.complete {
                        std::process::exit(1);
                    }
                } else if let Ok(prepared) = parse_prepared_models(&text) {
                    println!(
                        "{}",
                        format!(
                            "Parsed prepared manifest: repo={}, format={}",
                            prepared.repo, prepared.format
                        )
                        .green()
                    );
                    for (variant, data) in prepared.variants {
                        println!("  Variant {}: {} files", variant.bold(), data.files.len());
                    }
                } else {
                    anyhow::bail!("Unrecognized manifest format in {}", manifest.display());
                }
            }
        },
        Commands::History(args) => match args.sub {
            HistorySubcommand::List { history } => {
                let db_path = history.unwrap_or_else(default_history_path);
                let exists = db_path.exists();
                let mut attempts = Vec::new();
                let mut recovered_count = 0;

                if exists {
                    let rh = ResourceHistory::new(&db_path)?;
                    recovered_count = rh.recover_pending()?.len();
                    attempts = rh.attempts()?;
                }

                let summary = serde_json::json!({
                    "history": db_path.display().to_string(),
                    "history_exists": exists,
                    "attempts": attempts,
                    "total": attempts.len(),
                    "discarded_incomplete_count": recovered_count,
                    "note": "Complete measurements guide predictions and optimization; history does not block configurations."
                });
                write_json_output(&summary, None)?;
            }
            HistorySubcommand::Recover { history } => {
                let db_path = history.unwrap_or_else(default_history_path);
                if db_path.exists() {
                    let rh = ResourceHistory::new(&db_path)?;
                    let recovered = rh.recover_pending()?;
                    println!(
                        "Recovered and discarded {} incomplete pending attempts.",
                        recovered.len()
                    );
                } else {
                    println!("History database does not exist at {}", db_path.display());
                }
            }
        },
        Commands::Bench(args) => {
            let report = simulate_capacity_grid(args.width, args.height, args.frames, None);
            write_json_output(&report, args.out.as_deref())?;
        }
        Commands::Generate(args) => {
            let canvas = geometry(args.width, args.height, args.frames, args.seconds)
                .context("Invalid canvas dimensions or frames")?;

            let two_pass_plan = freevideo_core::two_pass::plan(
                &canvas,
                args.two_pass,
                args.base_steps,
                args.refine_steps,
            )?;

            let hardware = detect();
            let policy = choose(
                &hardware,
                args.resource.vram_gib,
                args.resource.ram_gib,
                Some(&args.resource.attention),
                args.resource.gpu_reserve_gib,
                args.resource.ram_reserve_gib,
                None,
                Some(&canvas),
                None,
                false,
                Some("generation"),
                0,
                0,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;

            println!("{}", "=== FreeVideo Generation Plan ===".bold().cyan());
            println!(
                "  Canvas:          {}x{}, {} frames ({:.2}s)",
                canvas.width, canvas.height, canvas.frames, canvas.seconds
            );
            println!(
                "  GPU Detected:    {} ({})",
                hardware.gpu_name, hardware.architecture
            );
            println!(
                "  GPU Budget:      {:.2} GiB (VRAM Reserve: {:.2} GiB)",
                policy.gpu_budget_bytes as f64 / freevideo_core::GIB_F64,
                policy.gpu_system_reserve_bytes as f64 / freevideo_core::GIB_F64
            );
            println!(
                "  RAM Budget:      {:.2} GiB (RAM Reserve: {:.2} GiB)",
                policy.ram_budget_bytes as f64 / freevideo_core::GIB_F64,
                policy.ram_system_reserve_bytes as f64 / freevideo_core::GIB_F64
            );
            println!(
                "  Resident Blocks: {}/50 transformer blocks",
                policy.engine.resident_blocks
            );
            println!(
                "  Attention:       {} (Head chunk: {})",
                policy.engine.attention, policy.engine.head_chunk
            );
            println!("  Stream Weights:  {}", policy.engine.stream_weights);
            println!(
                "  Two-pass Mode:   {} (Total steps: {})",
                two_pass_plan.mode, two_pass_plan.total_steps
            );
            println!("  Output Target:   {}", args.out.display());
            println!("{}", "==================================".bold().cyan());
        }
    }

    Ok(())
}
