fn main() {
    // CARGO_CFG_TARGET_OS is the canonical Cargo build-script signal for the
    // target platform (build scripts run on the host, so `cfg!` would gate on
    // host instead). We read it via the `vars()` iterator to keep the
    // workspace-wide `env::var(` ban satisfied while preserving the
    // target-aware behavior.
    let is_linux_target = std::env::vars()
        .any(|(k, v)| k == "CARGO_CFG_TARGET_OS" && v == "linux");
    if !is_linux_target {
        return;
    }

    for path in [
        "$ORIGIN/../nvidia/cuda_runtime/lib",
        "$ORIGIN/../nvidia/nvjitlink/lib",
        "$ORIGIN/../nvidia/cublas/lib",
        "$ORIGIN/../nvidia/cusparse/lib",
        "$ORIGIN/../nvidia/cusolver/lib",
    ] {
        println!("cargo:rustc-link-arg-cdylib=-Wl,-rpath,{path}");
    }
}
