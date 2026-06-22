//! Bug hunt: a plain 1-D Gaussian measure-jet smooth `s(x, bs="mjs")` is
//! unfittable through the public formula API (`fit_from_formula`, which is the
//! exact path `gamfit.fit` / `fit_table` take) even though the `gam` CLI fits
//! the *identical* data without complaint.
//!
//! Reproduction (all on the same deterministic dataset written below):
//!
//! ```text
//! $ gam fit det.csv 'y ~ s(x, bs="mjs")' --out det.gam       # saved model: det.gam
//! >>> gamfit.fit(pd.read_csv("det.csv"), 'y ~ s(x, bs="mjs")')
//! IntegrationError: REML smoothing optimization failed to converge:
//!   spatial kappa optimization failed: Invalid input: anisotropic analytic
//!   optimization did not converge after 80 iterations
//!   (final_objective=-2.1e2, final_grad_norm=3.4e1)
//! ```
//!
//! Root cause (best read): the measure-jet representer length-scale is REML-learned
//! through the same anisotropic-dial joint κ optimizer as Matérn/Duchon
//! (`src/terms/smooth/spatial_optimization.rs`). The CLI builds its `FitOptions`
//! in `src/main/run_fit.rs` with the outer smoothing tolerance `tol = 1e-6`,
//! while the formula/FFI path builds them in
//! `src/solver/workflow/materialize.rs:2193` with `tol = 1e-10` (deliberately
//! tightened for the `w=c ⇔ c-fold replication` invariance, #893). At the
//! tighter outer tolerance the REML-seeded κ optimization hits its 80-iteration
//! cap with a large projected gradient and the non-convergence is escalated to a
//! fatal error at `spatial_optimization.rs:2859` instead of falling back to the
//! frozen baseline geometry (the path that succeeds for the CLI). The model is
//! demonstrably fittable — the CLI fits this exact data — so the formula API
//! aborting is the bug.
//!
//! This test fits the measure-jet smooth through `fit_from_formula` and asserts
//! the fit produces a usable result (finite coefficients, a finite and
//! genuinely-smoothed effective dof). It currently fails because
//! `fit_from_formula` returns `Err(IntegrationFailed)`. When the formula path is
//! made as robust as the CLI (align the tolerance, or fall back to the baseline
//! geometry on κ-optimizer non-convergence), it will pass without edits.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const N: usize = 250;

/// SplitMix64 finalizer mapped to [0, 1): deterministic, RNG-free per-index
/// pseudo-noise so the dataset is bit-reproducible across machines.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// `y = sin(2πx) + 0.1·noise` on a regular grid in [0, 1] — a smooth signal a
/// 1-D smoother recovers easily. The CLI fits `s(x, bs="mjs")` on exactly this
/// data.
fn measure_jet_1d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..N)
        .map(|i| {
            let x = i as f64 / (N as f64 - 1.0);
            let noise = 2.0 * hashed_unit(i as u64) - 1.0;
            let y = (std::f64::consts::TAU * x).sin() + 0.1 * noise;
            StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 1D measure-jet dataset")
}

#[test]
fn measure_jet_formula_fit_succeeds_like_the_cli() {
    init_parallelism();
    let data = measure_jet_1d_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ s(x, bs=\"mjs\")", &data, &config).expect(
        "a 1-D Gaussian measure-jet smooth must fit through the formula API \
         (the gam CLI fits this exact data); it currently aborts with a spatial \
         kappa optimization non-convergence",
    );

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };

    assert_eq!(
        fit.design.smooth.terms.len(),
        1,
        "expected exactly one measure-jet smooth term"
    );
    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must be finite"
    );

    let edf = fit
        .fit
        .edf_total()
        .expect("a fitted smooth must report a total effective dof");
    assert!(edf.is_finite(), "effective dof must be finite, got {edf}");
    // A genuine smooth of a full sine wave uses several degrees of freedom; an
    // EDF that collapsed to ~1 (a flat line) or blew past the sample size would
    // signal a degenerate fit rather than a recovered signal.
    assert!(
        edf > 2.0 && edf < N as f64,
        "effective dof {edf} is outside the sane range (2, {N}) for a recovered sine smooth"
    );
}
