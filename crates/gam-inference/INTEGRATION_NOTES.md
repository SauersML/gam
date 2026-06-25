# gam-inference Integration Notes

- Moved `src/inference/conformal.rs`, `src/inference/posterior_bands.rs`, and `src/inference/hmc.rs` into `crates/gam-inference/src/`.
- Added `gam-predict = { path = "../gam-predict" }` and rewrote moved predict references to `gam_predict::...` as requested. The `crates/gam-predict` directory is not present in this checkout despite the mission note that it exists.
- Added engine-side HMC back-edge definitions in `src/inference/hmc_io.rs` and repointed engine callers from `crate::inference::hmc::...` to `crate::inference::hmc_io::...`.
- `src/inference/mod.rs` is a shared integrator-owned file, so it was not edited. The integrator must add `pub(crate) mod hmc_io;` or equivalent wiring for the new engine-side module.
- Root `Cargo.toml`, `src/lib.rs`, `src/inference/mod.rs`, and `Cargo.lock` were intentionally left untouched.
