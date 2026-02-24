use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");

    let build_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=GAM_BUILD_TIMESTAMP={build_ts}");

    if let Ok(tag) = std::env::var("GAM_RELEASE_TAG") {
        println!("cargo:rustc-env=GAM_RELEASE_TAG={tag}");
    }

    enforce_engine_domain_boundary();
}

fn enforce_engine_domain_boundary() {
    // Lightweight guard: keep engine free of app-specific branding and data parsing.
    let forbidden = [
        "GNOMON",
        "polars::",
        "CsvReader",
        "LazyFrame",
        "load_training_data(",
        "load_prediction_data(",
    ];

    let mut violations = Vec::new();
    let src_root = PathBuf::from("src");
    let mut stack = vec![src_root];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let rel = path.to_string_lossy().to_string();
            let src = match fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for (line_no, line) in src.lines().enumerate() {
                for token in &forbidden {
                    if line.contains(token) {
                        violations.push(format!(
                            "{}:{} contains forbidden token '{}'",
                            rel,
                            line_no + 1,
                            token
                        ));
                    }
                }
            }
        }
    }

    if !violations.is_empty() {
        eprintln!("\nerror: engine/domain boundary violations detected in gam core modules:");
        for v in violations {
            eprintln!("  - {v}");
        }
        panic!("domain leakage in gam engine modules");
    }

}
