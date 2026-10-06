//! A consumer built with `panic = "abort"` gets typed CUDA absence from the probe, not an abort.
//!
//! No in-tree test binary can observe the abort: libtest ignores a profile's `panic` setting, and
//! gam's own profiles unwind. This test builds the gam-gpu example
//! `cudarc_probe_under_panic_abort` as a `panic = "abort"` consumer builds it, every crate of its
//! graph with abort (`CARGO_PROFILE_DEV_PANIC=abort`), in a private target directory so the nested
//! cargo never waits on the outer build's lock, and runs it. Aborting the example crate alone
//! (`cargo rustc -- -C panic=abort`) does not link on macOS, where the Metal dependencies (`objc2`
//! and the crates around it) require the crates that use them to unwind as they do. The
//! nested resolve is `--locked` to the workspace lockfile but may fetch: a test shard that runs
//! from a nextest archive has the registry cache but no checkout of the workspace's git
//! dependencies, and `--offline` refused the resolve there. On a host where cudarc's loader opens
//! no libcuda candidate, the probe must answer `absent` and exit 0.
//! Before the preflight, the probe's first cudarc entry point panicked there and the process
//! aborted.

use std::path::PathBuf;
use std::process::Command;

const EXAMPLE: &str = "cudarc_probe_under_panic_abort";

fn build_example_with_panic_abort() -> PathBuf {
    let output = Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env(
            "CARGO_TARGET_DIR",
            concat!(env!("CARGO_TARGET_TMPDIR"), "/probe_under_panic_abort"),
        )
        .env("CARGO_PROFILE_DEV_PANIC", "abort")
        .args([
            "build",
            "--locked",
            "--package",
            "gam-gpu",
            "--example",
            EXAMPLE,
            "--message-format=json-render-diagnostics",
        ])
        .output()
        .expect("run cargo build for the panic=abort probe example");
    assert!(
        output.status.success(),
        "cargo build failed for the panic=abort probe example: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .unwrap_or_else(|error| panic!("cargo printed a non-JSON message ({error}): {line}"))
        })
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == EXAMPLE
        })
        .find_map(|message| message["executable"].as_str().map(PathBuf::from))
        .expect("cargo reported the example's executable")
}

/// Whether cudarc's own driver loader can open libcuda on this host: the candidates its `culib()`
/// walks, opened through the same call.
#[cfg(target_os = "linux")]
fn cudarc_loader_opens_libcuda() -> bool {
    ["cuda", "nvcuda"]
        .into_iter()
        .flat_map(cudarc::get_lib_name_candidates)
        // SAFETY: opens a CUDA driver candidate by cudarc's own names, exactly the load cudarc's
        // loader performs, and resolves no symbol from it.
        .any(|name| unsafe { libloading::Library::new(name) }.is_ok())
}

/// cudarc is not a dependency off Linux, and the probe answers an unsupported platform there.
#[cfg(not(target_os = "linux"))]
fn cudarc_loader_opens_libcuda() -> bool {
    false
}

#[test]
fn a_panic_abort_consumer_gets_typed_absence_where_cudarc_cannot_open_libcuda() {
    let executable = build_example_with_panic_abort();
    let output = Command::new(&executable)
        .output()
        .expect("run the panic=abort probe example");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the panic=abort probe did not exit cleanly, status {}: stdout={stdout} stderr={stderr}",
        output.status
    );
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some("panic=abort"),
        "the example was not built with panic=abort: {stdout}"
    );
    let outcome = lines.next().expect("the probe printed its outcome");
    let libcuda_opens = cudarc_loader_opens_libcuda();
    eprintln!("cudarc loader opens libcuda: {libcuda_opens}; probe outcome: {outcome}");
    if !libcuda_opens {
        assert!(
            outcome.starts_with("absent: "),
            "cudarc's loader opens no libcuda candidate here, so the probe must answer typed \
             absence: {outcome}"
        );
    }
}
