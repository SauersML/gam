// Build script that emits Linux-specific cdylib rpath link args. On non-
// Linux hosts the rpath syntax differs (and is unnecessary), so the entire
// emission is gated on the build-script HOST being Linux. cdylib pyffi
// wheels are built natively per platform, so host == target in practice.
#[cfg(target_os = "linux")]
fn main() {
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

#[cfg(not(target_os = "linux"))]
fn main() {}
