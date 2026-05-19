fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
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
