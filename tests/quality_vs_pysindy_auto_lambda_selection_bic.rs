//! End-to-end quality: gam's automatic SINDy `λ` selection via BIC must RECOVER
//! THE TRUE SPARSE DYNAMICS of a system whose governing equations we know
//! exactly. This is a truth-recovery test, not a "reproduce pysindy" test:
//! matching another tool's fitted coefficients proves nothing about correctness,
//! so the pass/fail criterion is gam's distance to the *analytic ground-truth
//! coefficient matrix*, with **pysindy** (Brunton, Proctor & Kutz, PNAS 2016 —
//! the canonical SINDy implementation) demoted to a match-or-beat ACCURACY
//! baseline on that same truth metric.
//!
//! Data: an 800-row trajectory of the planar system
//!     dx/dt = 2x − y + 0.5·x²y,   dy/dt = −y + 0.1·x·y
//! integrated with fixed-step RK4 from a fixed seed; near-exact analytic
//! derivatives (noise std 1e-8). The library is the polynomial set
//! [1, x, y, x², xy, y², x²y, xy²] — a superset of the true terms so STLSQ has
//! genuine work to do pruning spurious columns. In the `(p, d)` coefficient
//! layout (one column per state derivative) the ground-truth matrix is therefore
//!     Ξ*[x, 0]=2, Ξ*[y, 0]=−1, Ξ*[x²y, 0]=0.5  (dx/dt column),
//!     Ξ*[y, 1]=−1, Ξ*[xy, 1]=0.1               (dy/dt column),
//! all other entries exactly zero. The SAME `Θ` and `Ẋ` matrices are handed to
//! both engines (gam reads them as ndarrays; pysindy gets them flattened over the
//! wire), so the baseline comparison is fair.
//!
//! Objective metric asserted (un-weakened):
//!   1. EXACT SUPPORT RECOVERY — gam's nonzero entries are precisely the five
//!      true-term locations and nothing else (no spurious columns survive, no
//!      true column is dropped).
//!   2. COEFFICIENT ACCURACY VS TRUTH — max-abs error of gam's nonzero
//!      coefficients against the analytic values is below a principled bar
//!      (the derivatives carry only ~1e-8 noise, so the true terms must be
//!      recovered to well under 1% of the smallest true coefficient, 0.1).
//!   3. MATCH-OR-BEAT — gam's relative-L2 error to truth is no worse than
//!      1.10× pysindy's relative-L2 error to the same truth.

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_auto_lam};
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
use ndarray::Array2;

/// Polynomial library width: [1, x, y, x², xy, y², x²y, xy²].
const P: usize = 8;
/// State dimension (x, y).
const D: usize = 2;
/// Hard threshold shared by both engines' STLSQ.
const TOL: f64 = 0.05;
/// STLSQ round cap shared by both engines.
const MAX_ROUNDS: usize = 20;

/// Evaluate the polynomial library row [1, x, y, x², xy, y², x²y, xy²].
fn library_row(x: f64, y: f64) -> [f64; P] {
    [1.0, x, y, x * x, x * y, y * y, x * x * y, x * y * y]
}

/// Analytic ground-truth coefficient matrix `Ξ* ∈ ℝ^{P × D}` for the planar
/// system, in the same row-major `(library_term, derivative)` layout gam emits.
/// Library order is [1, x, y, x², xy, y², x²y, xy²].
///   dx/dt = 2x − y + 0.5·x²y → col 0: index 1→2.0, 2→−1.0, 6→0.5
///   dy/dt = −y + 0.1·x·y     → col 1: index 2→−1.0, 4→0.1
fn truth_coefficients() -> Array2<f64> {
    let mut xi = Array2::<f64>::zeros((P, D));
    xi[(1, 0)] = 2.0; // x      in dx/dt
    xi[(2, 0)] = -1.0; // y      in dx/dt
    xi[(6, 0)] = 0.5; // x²y    in dx/dt
    xi[(2, 1)] = -1.0; // y      in dy/dt
    xi[(4, 1)] = 0.1; // xy     in dy/dt
    xi
}

/// Analytic vector field f(x, y) = (dx/dt, dy/dt).
fn vector_field(x: f64, y: f64) -> (f64, f64) {
    let dx = 2.0 * x - y + 0.5 * x * x * y;
    let dy = -y + 0.1 * x * y;
    (dx, dy)
}

#[test]
fn gam_sindy_auto_lambda_recovers_true_dynamics() {
    // ---- deterministic RK4 trajectory + library/derivative matrices --------
    let n = 800usize;
    let dt = 1.0e-3;
    // Tiny deterministic LCG noise (std ~1e-8) so derivatives are near-exact
    // but not bit-trivial; identical sequence feeds both Θ and Ẋ rows.
    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let mut noise = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5) * 2.0e-8
    };

    let mut theta = Array2::<f64>::zeros((n, P));
    let mut dz = Array2::<f64>::zeros((n, D));

    let (mut x, mut y) = (0.30_f64, -0.20_f64);
    for i in 0..n {
        // Library at the current state (with tiny measurement noise).
        let xs = x + noise();
        let ys = y + noise();
        let row = library_row(xs, ys);
        for (j, &v) in row.iter().enumerate() {
            theta[(i, j)] = v;
        }
        // Near-exact analytic derivative at the noisy state + tiny noise.
        let (dx, dy) = vector_field(xs, ys);
        dz[(i, 0)] = dx + noise();
        dz[(i, 1)] = dy + noise();

        // Advance the true (noise-free) state by one RK4 step.
        let (k1x, k1y) = vector_field(x, y);
        let (k2x, k2y) = vector_field(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
        let (k3x, k3y) = vector_field(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
        let (k4x, k4y) = vector_field(x + dt * k3x, y + dt * k3y);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    }

    // ---- gam: automatic λ selection via BIC over the geometric grid --------
    let (gam_lam, gam_res) = sindy_stlsq_auto_lam(
        theta.view(),
        dz.view(),
        TOL,
        MAX_ROUNDS,
        SindyPenaltyKind::Ridge,
        3.7,
    )
    .expect("gam sindy_stlsq_auto_lam");
    let gam_coef = &gam_res.coefficients;
    let truth = truth_coefficients();
    let truth_flat: Vec<f64> = truth.iter().copied().collect();

    // ---- pysindy baseline: its own STLSQ+BIC sweep on the same Θ/Ẋ ---------
    // Flatten Θ (row-major, n×P) and Ẋ (n×D) into wire columns; Python rebuilds
    // both matrices and runs pysindy.STLSQ at each grid point, picking its own
    // minimum-BIC model. This is the reference fit we judge AGAINST THE SAME
    // ground truth (it is not the pass criterion).
    let theta_flat_raw: Vec<f64> = theta.iter().copied().collect();
    let dz_flat_raw: Vec<f64> = dz.iter().copied().collect();
    let shape: Vec<f64> = vec![n as f64, P as f64, D as f64, TOL, MAX_ROUNDS as f64];

    // The CSV bridge demands every wire column share one row count. The three
    // payloads have different natural lengths (Θ is n·P, Ẋ is n·D, meta is 5),
    // so pad each to the common maximum with NaN; Python slices each back to its
    // true length (n·P, n·D, first 5) before use, so the padding is inert.
    let wire_len = theta_flat_raw.len().max(dz_flat_raw.len()).max(shape.len());
    let theta_flat = pad_to(&theta_flat_raw, wire_len);
    let dz_flat = pad_to(&dz_flat_raw, wire_len);
    let meta = pad_to(&shape, wire_len);

    let r = run_python(
        &[
            Column::new("theta_flat", &theta_flat),
            Column::new("dz_flat", &dz_flat),
            Column::new("meta", &meta),
        ],
        r#"
import numpy as np
from pysindy.optimizers import STLSQ

# Unpack metadata (first 5 entries of the padded `meta` column).
meta = np.asarray(df["meta"]).reshape(-1)
n = int(round(meta[0])); p = int(round(meta[1])); d = int(round(meta[2]))
tol = float(meta[3]); max_rounds = int(round(meta[4]))

Theta = np.asarray(df["theta_flat"]).reshape(-1)[: n * p].reshape(n, p)
Xdot = np.asarray(df["dz_flat"]).reshape(-1)[: n * d].reshape(n, d)

def bic(Theta, Xdot, Xi):
    # Xi has shape (p, d): one sparse coefficient column per state derivative.
    resid = Theta @ Xi - Xdot
    total = 0.0
    nf = float(n)
    for c in range(d):
        rss = max(float(np.sum(resid[:, c] ** 2)), 1.0e-300)
        k = int(np.count_nonzero(Xi[:, c]))
        total += nf * np.log(rss / nf) + k * np.log(nf)
    return total

# gam's geometric grid: tol.max(1e-6) * 10^(i-4), i = 0..8 (9 points).
base = max(tol, 1.0e-6)
grid = [base * (10.0 ** (i - 4)) for i in range(9)]

best_lam = None
best_bic = np.inf
best_xi = None
for lam in grid:
    # pysindy STLSQ: `threshold` is the hard cutoff (gam's tol), `alpha` is the
    # ridge weight (gam's lam). `fit(Theta, Xdot)` returns coef_ as (d, p).
    # unbias=False so pysindy KEEPS the ridge in its active-set refit, exactly
    # like gam's `ridge_diag_solve` (gam never does the OLS de-biasing pass that
    # pysindy applies by default); this makes the coefficient comparison a true
    # head-to-head of the same estimator, not ridge-vs-OLS.
    opt = STLSQ(threshold=tol, alpha=lam, max_iter=max_rounds,
                normalize_columns=False, fit_intercept=False, unbias=False)
    opt.fit(Theta, Xdot)
    Xi = np.asarray(opt.coef_).T  # -> (p, d)
    score = bic(Theta, Xdot, Xi)
    if score < best_bic:
        best_bic = score
        best_lam = lam
        best_xi = Xi

emit("lam", [best_lam])
emit("coef", best_xi.reshape(-1))  # row-major (p, d), matches gam ndarray order
"#,
    );

    let py_lam = r.scalar("lam");
    let py_coef = r.vector("coef");

    let gam_coef_flat: Vec<f64> = gam_coef.iter().copied().collect();
    assert_eq!(
        py_coef.len(),
        gam_coef_flat.len(),
        "pysindy coef length mismatch"
    );
    assert_eq!(
        gam_coef_flat.len(),
        truth_flat.len(),
        "gam coef length mismatch with truth"
    );

    // PRIMARY objective metric: distance of each engine's recovered coefficients
    // to the ANALYTIC ground truth (not to each other).
    let gam_err_vs_truth = relative_l2(&gam_coef_flat, &truth_flat);
    let py_err_vs_truth = relative_l2(py_coef, &truth_flat);
    let gam_max_abs_err = max_abs_diff(&gam_coef_flat, &truth_flat);

    // The five true-term locations, in the (j outer, c inner) order the scan
    // below produces them: (1,0)=x→dx, (2,0)=y→dx, (2,1)=y→dy, (4,1)=xy→dy,
    // (6,0)=x²y→dx.
    let true_support: Vec<(usize, usize)> = vec![(1, 0), (2, 0), (2, 1), (4, 1), (6, 0)];
    let gam_recovered_support: Vec<(usize, usize)> = (0..P)
        .flat_map(|j| (0..D).map(move |c| (j, c)))
        .filter(|&(j, c)| gam_coef[(j, c)] != 0.0)
        .collect();

    eprintln!(
        "sindy truth recovery: n={n} P={P} D={D} | gam_lam={gam_lam:.3e} py_lam={py_lam:.3e} \
         | gam_rel_l2_vs_truth={gam_err_vs_truth:.4e} py_rel_l2_vs_truth={py_err_vs_truth:.4e} \
         | gam_max_abs_err={gam_max_abs_err:.4e} | gam_support={gam_recovered_support:?}"
    );

    // (1) EXACT SUPPORT RECOVERY: gam's nonzero entries are precisely the five
    // true-term locations — no spurious columns survive, no true column drops.
    assert_eq!(
        gam_recovered_support, true_support,
        "gam did not recover the exact true support: got {gam_recovered_support:?}, \
         expected {true_support:?}"
    );

    // (2) COEFFICIENT ACCURACY VS TRUTH: with ~1e-8 derivative noise the true
    // coefficients must be recovered to far below 1% of the smallest true
    // coefficient (0.1), i.e. max-abs error well under 1e-3.
    assert!(
        gam_max_abs_err < 1.0e-3,
        "gam coefficients diverge from analytic truth: max_abs_err={gam_max_abs_err:.4e} \
         (true coefficients [2, -1, 0.5, -1, 0.1])"
    );

    // (3) MATCH-OR-BEAT pysindy on accuracy-to-truth: gam's relative-L2 error to
    // the ground truth is no worse than 1.10× the reference's error to the same
    // truth. This keeps pysindy as a baseline without making "same as pysindy"
    // the pass criterion — both are judged against the known answer.
    assert!(
        gam_err_vs_truth <= py_err_vs_truth * 1.10,
        "gam less accurate to truth than pysindy: gam_rel_l2={gam_err_vs_truth:.4e} \
         py_rel_l2={py_err_vs_truth:.4e} (allowed {:.4e})",
        py_err_vs_truth * 1.10
    );
}

/// Right-pad `v` with NaN up to `len` so every wire column shares one row count.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(v);
    out.resize(len, f64::NAN);
    out
}
