//! Bug: a binding parametric coefficient inequality constraint
//! (`nonnegative(x)` / `linear(x, min=0)`) makes the
//! **whole fit abort** with a spurious KKT "stationarity" violation as soon as
//! the model contains at least one *other* free term.
//!
//! Reproduction (confirmed against the `gam` CLI, fully deterministic, no RNG,
//! no noise): a low-discrepancy scatter of `x_i = frac((i+1)·φ)` and
//! `x2_i = frac((i+1)·√2)` over `[0,1]`, with the noise-free surface
//!
//! ```text
//!   y = sin(2π·x) + 0.5·x2.
//! ```
//!
//! The ordinary-least-squares slope of `sin(2π·x)` against `x` on `[0,1]` is
//! negative, so the constraint `β_x ≥ 0` BINDS: the constrained MAP is
//! `β_x = 0`, with `x2` and the intercept free to absorb the rest. That optimum
//! is well defined, but the generic active-set path used to abort:
//!
//! ```text
//!   y ~ nonnegative(x) + x2    -> error: Parameter constraint violation:
//!                                  KKT residuals exceed tolerance:
//!                                  primal=0, dual=0, comp=0, stat=5.78e-5
//!                                  (tol=5.0e-6); active=1/1
//!   y ~ nonnegative(x)         -> fits fine (no extra free term)
//!   y ~ x + x2  (unconstrained)-> fits fine
//! ```
//!
//! Primal feasibility, dual feasibility and complementarity are all exactly
//! zero — only the *stationarity* channel fails, by ~one order of magnitude
//! (5.8e-5 vs the 5e-6 gate; the residual grows with the data, ~2e-4 on other
//! datasets), and the active face is full rank (`active=1/1`), so the
//! rank-deficient-face relaxation does not apply. The free-coordinate gradient
//! the diagnostic reports is non-zero even though the boundary MAP is feasible,
//! which points at a scaling mismatch
//! between the gradient/constraint coordinate systems the active-set QP solves
//! in and the KKT diagnostic measures in — not genuine non-stationarity.
//!
//! Best read of the cause (files/lines):
//!  * `src/solver/reml/runtime.rs` `enforce_constraint_kkt` (~line 5510) hard-
//!    errors when `kkt.stationarity > KKT_TOL_STAT` (`= 5e-6`, line 89). The
//!    degenerate-face relaxation (`ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL`)
//!    is gated on `working_set_rank_deficient`, which is false for a single
//!    full-rank active row, so the strict 5e-6 bound applies.
//!  * `src/solver/active_set.rs` `compute_constraint_kkt_diagnostics` (line 238)
//!    forms `stationarity = ‖gradient − Aᵀλ‖∞` with `λ` from
//!    `project_stationarity_residual_on_constraint_cone` (line 344). The
//!    active-set QP itself terminates on a tiny Newton step (`tol_step = 1e-12`,
//!    line 1294; short-circuit at line 1362) — i.e. it believes it converged —
//!    yet the post-hoc diagnostic measures a 5.8e-5 free-coordinate gradient.
//!    Same family as #791 (a wrong scale power in the parametric-constraint
//!    coordinate transform), here surfacing as a spurious KKT abort instead of a
//!    silently-wrong coefficient.
//!
//! The fit persists the boundary MAP separately from its reported coefficient:
//! `constrained_posterior.mode` is zero, while `fit.beta` and default prediction
//! are the strictly-interior truncated posterior mean mandated by SPEC. This
//! test checks both estimands rather than requiring the posterior mean to equal
//! the MAP.
//!
//! Related: see the sibling build-break ticket filed in the same run.

use gam::inference::model::FittedModel;
use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 400;

/// Deterministic low-discrepancy training data: `x_i = frac((i+1)·φ)`,
/// `x2_i = frac((i+1)·√2)`, `y = sin(2π·x) + 0.5·x2` (no noise). The scattered
/// (non-grid) `x` is what pushes the constrained QP into the regime where the
/// KKT stationarity diagnostic spuriously trips; a uniform grid does not.
fn write_training_csv(path: &Path) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let sqrt2 = 2.0_f64.sqrt();
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "x2", "y"]).expect("write header");
    for i in 0..N {
        let k = (i + 1) as f64;
        let x = (k * phi).fract();
        let x2 = (k * sqrt2).fract();
        let y = (2.0 * std::f64::consts::PI * x).sin() + 0.5 * x2;
        writer
            .write_record([format!("{x:.12}"), format!("{x2:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Four probe rows: a pair that isolates the `x` slope (x2 held at 0.4) and a
/// pair that isolates the `x2` slope (x held at 0.5).
fn write_probe_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create probe csv");
    writer.write_record(["x", "x2", "y"]).expect("write header");
    for &(x, x2) in &[(0.0_f64, 0.4_f64), (0.8, 0.4), (0.5, 0.0), (0.5, 1.0)] {
        writer
            .write_record([format!("{x:.12}"), format!("{x2:.12}"), "0.0".to_string()])
            .expect("write probe row");
    }
    writer.flush().expect("flush probe csv");
}

/// Read the explicit posterior-mean column from a standard prediction CSV.
fn read_means(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "posterior_mean")
        .expect("standard predict csv has a posterior_mean column");
    reader
        .records()
        .map(|rec| {
            rec.expect("predict csv row")[mean_idx]
                .parse::<f64>()
                .expect("numeric prediction")
        })
        .collect()
}

fn constrained_x_estimands(model_path: &Path) -> (f64, f64) {
    let model = FittedModel::load_from_path(model_path).expect("load constrained saved model");
    let fit = model.unified().expect("saved model carries unified fit");
    let posterior = fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .expect("nonnegative coefficient persists its constrained posterior");
    (posterior.mode[1], fit.beta[1])
}

/// Fit `y ~ <formula>` and return whether the fit succeeded plus its stderr.
fn try_fit(dir: &Path, label: &str, formula: &str) -> (bool, PathBuf, String) {
    let train = dir.join("train.csv");
    let model = dir.join(format!("model_{label}.json"));
    write_training_csv(&train);
    let out = Command::new(gam::gam_binary!())
        .arg("fit")
        .arg(&train)
        .arg(formula)
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model)
        .output()
        .expect("spawn gam fit");
    let ok = out.status.success() && model.is_file();
    (ok, model, String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Predict the four probe rows and return `(x_slope_per_unit, x2_slope)`.
fn probe_slopes(dir: &Path, label: &str, model: &Path) -> (f64, f64) {
    let probe = dir.join("probe.csv");
    let out_csv = dir.join(format!("pred_{label}.csv"));
    write_probe_csv(&probe);
    let status = Command::new(gam::gam_binary!())
        .arg("predict")
        .arg(model)
        .arg(&probe)
        .arg("--out")
        .arg(&out_csv)
        .status()
        .expect("spawn gam predict");
    assert!(status.success(), "gam predict `{label}` failed");
    let m = read_means(&out_csv);
    assert_eq!(m.len(), 4, "expected four probe predictions for `{label}`");
    let x_slope = (m[1] - m[0]) / 0.8; // Δx = 0.8 at x2 = 0.4
    let x2_slope = m[3] - m[2]; // Δx2 = 1.0 at x = 0.5
    (x_slope, x2_slope)
}

#[test]
fn nonnegative_constraint_with_free_term_fits_to_the_constrained_optimum() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let dir = tmp.path();

    // Sanity 1: the same constraint with NO other free term fits fine — proof
    // the active-set path and the data are otherwise healthy.
    let (alone_ok, _, alone_err) = try_fit(dir, "alone", "y ~ nonnegative(x)");
    assert!(
        alone_ok,
        "control failed: `y ~ nonnegative(x)` (no extra free term) should fit; stderr:\n{alone_err}"
    );

    // Sanity 2: the unconstrained model fits fine — the constraint is the only
    // new ingredient.
    let (unc_ok, _, unc_err) = try_fit(dir, "unc", "y ~ x + x2");
    assert!(
        unc_ok,
        "control failed: unconstrained `y ~ x + x2` should fit; stderr:\n{unc_err}"
    );

    // THE BUG: the active-set constraint path must fit this model. It used to
    // abort with `KKT residuals exceed tolerance: ... stat=5.78e-5 (tol=5e-6)`.
    let (nn_ok, nnmodel, nn_err) = try_fit(dir, "nonneg", "y ~ nonnegative(x) + x2");
    assert!(
        nn_ok,
        "BUG: `y ~ nonnegative(x) + x2` aborted instead of fitting the constrained \
         optimum. The binding constraint + one free term trips a spurious KKT stationarity \
         violation. gam fit stderr:\n{nn_err}"
    );

    let (mode_xslope, reported_xslope) = constrained_x_estimands(&nnmodel);
    assert!(
        mode_xslope.abs() < 1e-8,
        "nonnegative coefficient MAP must bind at 0, got {mode_xslope:.12}"
    );
    assert!(
        reported_xslope >= 0.0 && reported_xslope > mode_xslope,
        "reported coefficient {reported_xslope:.12} must be the strictly-interior \
         truncated posterior mean in [0, ∞), above its boundary MAP {mode_xslope:.12}"
    );

    // Default prediction must use that posterior mean, and the unconstrained
    // companion term must retain the signal the fixture assigned to it.
    let (nn_xslope, nn_x2slope) = probe_slopes(dir, "nonneg", &nnmodel);
    assert!(
        (nn_xslope - reported_xslope).abs() <= 1e-9 * reported_xslope.abs().max(1.0),
        "default prediction slope {nn_xslope:.12} must equal the persisted posterior \
         mean {reported_xslope:.12}"
    );
    assert!(
        (nn_x2slope - 0.5).abs() < 0.05,
        "the free x2 effect should remain near its generating slope 0.5, got {nn_x2slope:.6}"
    );
}
