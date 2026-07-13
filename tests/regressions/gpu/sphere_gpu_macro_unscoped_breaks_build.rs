//! Failing-ticket regression: the `gam` crate does not compile at HEAD on a
//! Linux host, so nothing that links it — this whole `tests/` suite, the
//! `gamfit` Python wheel, any downstream consumer — can be built.
//!
//! Root cause (files/lines read):
//!
//!  * `gpu_err!` is declared with `#[macro_export]` inside the `gpu` module:
//!    `src/gpu/error.rs:71` (`macro_rules! gpu_err { ... }`). A
//!    `#[macro_export]` `macro_rules!` is reachable crate-wide only by its
//!    absolute path `crate::gpu_err!`, or unqualified after it is brought into
//!    textual scope (a `use crate::gpu_err;` import, or a `#[macro_use]` on the
//!    defining module). The `gpu` module is declared *without* `#[macro_use]`
//!    (`src/lib.rs:68` — `pub mod gpu;`; the only `#[macro_use]` is on
//!    `mod macros;` at `src/lib.rs:46-47`), and `terms` (`src/lib.rs:78`) is a
//!    sibling, so `gpu_err!` is **not** in scope there by default.
//!
//!  * `src/terms/sphere_gpu.rs` (added by commit 38ee68993, "Remove
//!    compatibility shims and unify GPU policy") calls the macro **bare**, with
//!    no `use crate::gpu_err;` and without qualifying it:
//!      - `src/terms/sphere_gpu.rs:592` `gpu_err!("sphere n={n} overflows i32")`
//!      - `src/terms/sphere_gpu.rs:593` `gpu_err!("sphere m={m} overflows i32")`
//!      - `src/terms/sphere_gpu.rs:705` `gpu_err!("sphere-hh n=...")`
//!      - `src/terms/sphere_gpu.rs:706` `gpu_err!("sphere-hh m=...")`
//!      - `src/terms/sphere_gpu.rs:985` `gpu_err!("solve_penalised_ls_device: n_aug=...")`
//!      - `src/terms/sphere_gpu.rs:987` `gpu_err!("solve_penalised_ls_device: p=...")`
//!      - `src/terms/sphere_gpu.rs:1010` `gpu_err!("solve_penalised_ls_device: negative lwork=...")`
//!    `cargo build --lib` (debug or release) therefore fails with seven copies
//!    of `error: cannot find macro 'gpu_err' in this scope`.
//!
//!  * Other off-`gpu`-module callers qualify the macro — e.g.
//!    `src/solver/reml/eval.rs:245` uses `crate::gpu_err!(...)`; the
//!    in-`gpu`-module callers that use it bare add
//!    `use crate::gpu_err;` (e.g. `src/gpu/driver.rs:25`,
//!    `src/gpu/cubic_bspline_moments.rs:77`). `sphere_gpu.rs` does neither.
//!
//! The one-line fix is to bring the macro into scope in `sphere_gpu.rs`
//! (`use crate::gpu_err;`) or to qualify the seven call sites as
//! `crate::gpu_err!`.
//!
//! This test is deliberately a plain end-to-end CPU fit of a `sphere(lat, lon)`
//! smooth — the public-API neighbour of the file that fails to compile. While
//! the bug is live the test (like the rest of the crate) does not build, so it
//! cannot pass; once the macro is in scope the crate compiles, this fit runs on
//! the CPU path (no CUDA required), and the assertions below pass unchanged.
//! The fit recovers a smooth dipole-plus-interaction field on S² to better than
//! R² = 0.7, which is comfortably below what the sphere smooth actually achieves
//! (~0.98 in-sample), so the assertion is meaningful rather than a tautology.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    smooth::build_term_collection_design,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Smooth field on the unit sphere, expressed through the embedded Cartesian
/// coordinates of the (lat, lon) point: a constant, a dipole along z, and a
/// degree-2 interaction. Entirely smooth, so a working sphere smooth recovers
/// it well.
fn truth(lat: f64, lon: f64) -> f64 {
    let x = lat.cos() * lon.cos();
    let y = lat.cos() * lon.sin();
    let z = lat.sin();
    1.0 + 2.0 * z + 1.0 * x * y
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    // Uniform on the sphere: lat = asin(U(-1,1)), lon = U(-pi, pi).
    let u_sin = Uniform::new(-1.0, 1.0).expect("uniform");
    let u_lon = Uniform::new(-std::f64::consts::PI, std::f64::consts::PI).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["y", "lat", "lon"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let s: f64 = u_sin.sample(&mut rng);
            let lat = s.asin();
            let lon: f64 = u_lon.sample(&mut rng);
            let y = truth(lat, lon) + noise.sample(&mut rng);
            StringRecord::from(vec![y.to_string(), lat.to_string(), lon.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn sphere_smooth_fits_and_recovers_a_smooth_field_on_s2() {
    init_parallelism();
    let data = build_dataset(600, 0.15, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ sphere(lat, lon, radians=true)", &data, &cfg)
        .unwrap_or_else(|e| panic!("sphere(lat, lon) failed to fit: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for sphere(lat, lon)");
    };

    // Predict on a fresh deterministic grid of sphere points and compare to the
    // smooth truth.
    let mut rng = StdRng::seed_from_u64(101);
    let u_sin = Uniform::new(-1.0, 1.0).expect("uniform");
    let u_lon = Uniform::new(-std::f64::consts::PI, std::f64::consts::PI).expect("uniform");
    let m = 400usize;
    let mut design_in = Array2::<f64>::zeros((m, 3));
    let mut truth_vals = Vec::with_capacity(m);
    for row in 0..m {
        let s: f64 = u_sin.sample(&mut rng);
        let lat = s.asin();
        let lon: f64 = u_lon.sample(&mut rng);
        design_in[[row, 0]] = 0.0; // y placeholder column
        design_in[[row, 1]] = lat;
        design_in[[row, 2]] = lon;
        truth_vals.push(truth(lat, lon));
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild sphere design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();

    assert!(
        pred.iter().all(|v| v.is_finite()),
        "sphere predictions must all be finite"
    );

    let mean_t = truth_vals.iter().sum::<f64>() / m as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(truth_vals.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = truth_vals.iter().map(|t| (t - mean_t).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(
        r2 > 0.7,
        "sphere smooth recovered the held-out field at only R²={r2:.4} (expected \
         ~0.98); a capable S² smooth must clear the generous 0.7 bar"
    );
}
