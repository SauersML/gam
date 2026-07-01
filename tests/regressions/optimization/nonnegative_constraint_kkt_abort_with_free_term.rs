//! Bug: a binding parametric coefficient inequality constraint
//! (`nonnegative(x)` / `constrain(x, min=0)` / `linear(x, min=0)`) makes the
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
//! negative, so the constraint `β_x ≥ 0` BINDS: the constrained optimum is
//! `β_x = 0`, with `x2` and the intercept free to absorb the rest. That optimum
//! is well defined and *is* reachable — the exact-interval `bounded(x, …)`
//! path (a different solver) finds it: `β_x = 0`, `β_x2 ≈ 0.50`. But the
//! generic active-set path aborts:
//!
//! ```text
//!   y ~ nonnegative(x) + x2    -> error: Parameter constraint violation:
//!                                  KKT residuals exceed tolerance:
//!                                  primal=0, dual=0, comp=0, stat=5.78e-5
//!                                  (tol=5.0e-6); active=1/1
//!   y ~ nonnegative(x)         -> fits fine (no extra free term)
//!   y ~ x + x2  (unconstrained)-> fits fine
//!   y ~ bounded(x,min=0,max=5)+x2 -> fits fine, β_x = 0, β_x2 ≈ 0.50
//! ```
//!
//! Primal feasibility, dual feasibility and complementarity are all exactly
//! zero — only the *stationarity* channel fails, by ~one order of magnitude
//! (5.8e-5 vs the 5e-6 gate; the residual grows with the data, ~2e-4 on other
//! datasets), and the active face is full rank (`active=1/1`), so the
//! rank-deficient-face relaxation does not apply. The free-coordinate gradient
//! the diagnostic reports is non-zero even though `bounded()` demonstrates the
//! same constrained optimum is achievable, which points at a scaling mismatch
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
//! This test fits the constrained model and the `bounded()` anchor on identical
//! data and asserts the active-set path (a) succeeds and (b) lands on the same
//! constrained optimum the anchor reaches. While the bug is live the fit aborts,
//! so assertion (a) fails; once the active-set solver certifies the converged
//! constrained optimum, both pass unchanged.
//!
//! Related: see the sibling build-break ticket filed in the same run.

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

/// Read the `mean` column from a `gam predict --out` CSV.
fn read_means(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "mean")
        .or_else(|| headers.iter().position(|h| h == "eta"))
        .expect("predict csv has a mean / eta column");
    reader
        .records()
        .map(|rec| {
            rec.expect("predict csv row")[mean_idx]
                .parse::<f64>()
                .expect("numeric prediction")
        })
        .collect()
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

    // Anchor: the exact-interval `bounded()` path reaches the constrained
    // optimum on identical data — β_x clamped at the lower bound 0, β_x2 ≈ 0.50.
    // (If this ever stops binding the data stopped exercising the bug.)
    let (banchor_ok, bmodel, banchor_err) =
        try_fit(dir, "bounded", "y ~ bounded(x, min=0, max=5) + x2");
    assert!(
        banchor_ok,
        "anchor failed: `bounded(x,min=0,max=5) + x2` should fit; stderr:\n{banchor_err}"
    );
    let (b_xslope, b_x2slope) = probe_slopes(dir, "bounded", &bmodel);
    assert!(
        b_xslope.abs() < 1e-3,
        "anchor sanity: bounded x-slope should bind at 0, got {b_xslope:.6}"
    );
    assert!(
        (b_x2slope - 0.5).abs() < 0.05,
        "anchor sanity: bounded x2-slope should be ≈0.5, got {b_x2slope:.6}"
    );

    // THE BUG: the active-set constraint path must also fit this model. Today it
    // aborts with `KKT residuals exceed tolerance: ... stat=5.78e-5 (tol=5e-6)`
    // even though `bounded()` just proved the constrained optimum is reachable.
    let (nn_ok, nnmodel, nn_err) = try_fit(dir, "nonneg", "y ~ nonnegative(x) + x2");
    assert!(
        nn_ok,
        "BUG: `y ~ nonnegative(x) + x2` aborted instead of fitting the constrained \
         optimum that `bounded()` reaches (β_x=0, β_x2≈{b_x2slope:.3}). \
         The binding constraint + one free term trips a spurious KKT stationarity \
         violation. gam fit stderr:\n{nn_err}"
    );

    // Once it fits, it must land on the SAME constrained optimum as the anchor:
    // x clamped at 0, x2 effect recovered.
    let (nn_xslope, nn_x2slope) = probe_slopes(dir, "nonneg", &nnmodel);
    assert!(
        nn_xslope.abs() < 1e-3,
        "nonnegative x-slope should bind at 0 (constraint active), got {nn_xslope:.6}"
    );
    assert!(
        (nn_x2slope - b_x2slope).abs() < 0.02,
        "nonnegative x2-slope {nn_x2slope:.6} should match the bounded anchor {b_x2slope:.6} \
         (the two documented ways to constrain a coefficient must agree)"
    );
}
