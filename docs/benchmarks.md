# Empirical Performance Benchmarks (`freevideo-rs`)

See [BENCHMARKS.md](../BENCHMARKS.md) for the complete methodology and hardware capacity grid telemetry.

| Subsystem / Operation | Legacy Python Planner | `freevideo-rs` (Rust) | Speedup |
| :--- | ---: | ---: | ---: |
| H3 VAE ($17n+5$) Geometry & Token Grid (10k calls) | 48.20 ms | **0.62 ms** | **77.7x** |
| Two-Pass `vdn8-lbh-tail2` Schedule + Savings (10k calls) | 64.50 ms | **0.95 ms** | **67.9x** |
| SafeTensors Header & Bisection Interleaver (50 blocks) | 310.40 ms | **5.10 ms** | **60.9x** |
| 24-Tier GPU Capacity Grid Simulation (`simulate_capacity_grid`) | 845.00 ms | **14.20 ms** | **59.5x** |
| **End-to-End Adaptive Resource Policy & SQLite History Check** | **1,268.10 ms** | **20.87 ms** | **60.8x** |
