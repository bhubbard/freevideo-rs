use std::process::Command;

#[test]
fn test_cli_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .arg("--help")
        .output()
        .expect("Failed to execute freevideo CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FreeVideo Engine"));
    assert!(stdout.contains("plan"));
    assert!(stdout.contains("predict"));
    assert!(stdout.contains("doctor"));
    assert!(stdout.contains("models"));
    assert!(stdout.contains("bench"));
    assert!(stdout.contains("generate"));
}

#[test]
fn test_cli_plan() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .args([
            "plan", "--width", "1344", "--height", "768", "--frames", "243",
        ])
        .output()
        .expect("Failed to execute plan");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("gpu_budget_bytes"));
    assert!(stdout.contains("engine"));
    assert!(stdout.contains("decoder"));
    assert!(stdout.contains("resident_blocks"));
}

#[test]
fn test_cli_doctor() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .arg("doctor")
        .output()
        .expect("Failed to execute doctor");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("hardware"));
    assert!(stdout.contains("installed_attention_packages"));
    assert!(stdout.contains("paths"));
}

#[test]
fn test_cli_bench() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .args([
            "bench", "--width", "1344", "--height", "768", "--frames", "243",
        ])
        .output()
        .expect("Failed to execute bench");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("tiers"));
    assert!(stdout.contains("tier_gib"));
    assert!(stdout.contains("architecture"));
}

#[test]
fn test_cli_models_list() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .args(["models", "list"])
        .output()
        .expect("Failed to execute models list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("OpenVDN"));
}

#[test]
fn test_cli_generate_preview() {
    let output = Command::new(env!("CARGO_BIN_EXE_freevideo"))
        .args([
            "generate", "--out", "test.mp4", "--width", "1344", "--height", "768", "--frames",
            "243",
        ])
        .output()
        .expect("Failed to execute generate");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FreeVideo Generation Plan"));
    assert!(stdout.contains("GPU Budget"));
    assert!(stdout.contains("Two-pass Mode"));
}
