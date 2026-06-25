# gam-cli integration notes

- Remove the root `[[bin]]` target for `gam` from `/Cargo.toml`.
- Remove `default-run = "gam"` from the root package metadata in `/Cargo.toml`.
- Add `crates/gam-cli` to the root workspace members once the shared `/Cargo.toml` edit window is open.
- Remove the stale `pub mod report;` declaration from `/src/lib.rs` after reporting has moved into `gam-cli`.
- Update downstream `gam::report` imports outside this extraction scope, including `crates/gam-pyffi/src/ffi_prelude.rs`.
- Update `.github/workflows` jobs that build or run the `gam` binary so they target the new `crates/gam-cli` package. Current `cargo build --release --bin gam` sites are in `.github/workflows/benchmark.yml`, `.github/workflows/fuzz.yml`, `.github/workflows/large_scale.yml`, and `.github/workflows/test.yml`.
