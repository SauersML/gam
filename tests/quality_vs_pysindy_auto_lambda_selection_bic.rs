//! End-to-end quality: gam's automatic SINDy `λ` selection via BIC must match
//! **pysindy** — the reference implementation of Sparse Identification of
//! Nonlinear Dynamics (Brunton, Proctor & Kutz, PNAS 2016) — when both run the
//! identical sequentially-thresholded least-squares (STLSQ) sweep over the same
//! geometric `λ` grid and pick the minimum-BIC model on the same library and
//! derivatives.
//!
//! Why pysindy is the right comparator: `sindy_stlsq_auto_lam` reimplements the
//! literature-standard SINDy auto-`λ` recipe (STLSQ per grid point, Schwarz-BIC
//! complexity rule). pysindy ships the canonical `STLSQ` optimizer; running it
//! on the *same* `Θ`/`Ẋ` and scoring with the *same* Gaussian profile-likelihood
//! BIC `n·log(RSS/n) + k·log(n)` is a head-to-head check that gam's grid
//! generation, thresholding, and BIC formula reproduce the reference choice.
//!
//! Data: an 800-row trajectory of the planar system
//!     dx/dt = 2x − y + 0.5·x²y,   dy/dt = −y + 0.1·x·y
//! integrated with fixed-step RK4 from a fixed seed; near-exact analytic
//! derivatives (noise std 1e-8). The library is the polynomial set
//! [1, x, y, x², xy, y², x²y, xy²] — a superset of the true terms so STLSQ has
//! genuine work to do pruning spurious columns. The SAME `Θ` and `Ẋ` matrices
//! are handed to both engines (gam reads them as ndarrays; pysindy gets them
//! flattened over the wire), so any disagreement is a real divergence in the
//! selection logic, not a data or library mismatch.
//!
//! Asserted (un-weakened) bounds:
//!   1. selected `λ` agrees within a factor of 2 (the spec's grid-step tolerance);
//!   2. coefficient matrices agree in relative L2 (both must recover the true
//!      sparse dynamics, so they coincide to STLSQ tolerance);
//!   3. the active support has identical cardinality.

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_auto_lam};
use gam::test_support::reference::{Column, relative_l2, run_python};
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

/// Analytic vector field f(x, y) = (dx/dt, dy/dt).
fn vector_field(x: f64, y: f64) -> (f64, f64) {
    let dx = 2.0 * x - y + 0.5 * x * x * y;
    let dy = -y + 0.1 * x * y;
    (dx, dy)
}

#[test]
fn gam_sindy_auto_lambda_matches_pysindy_bic() {
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
    let gam_support = gam_coef.iter().filter(|&&v| v != 0.0).count();

    // ---- pysindy: identical STLSQ sweep + identical BIC, same Θ/Ẋ ----------
    // Flatten Θ (row-major, n×P) and Ẋ (n×D) into wire columns; Python rebuilds
    // both matrices and runs pysindy.STLSQ at each grid point, scoring with the
    // same Gaussian profile-likelihood BIC gam uses.
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
emit("bic", [best_bic])
emit("support", [int(np.count_nonzero(best_xi))])
emit("coef", best_xi.reshape(-1))  # row-major (p, d), matches gam ndarray order
"#,
    );

    let py_lam = r.scalar("lam");
    let py_bic = r.scalar("bic");
    let py_support = r.scalar("support") as usize;
    let py_coef = r.vector("coef");

    // gam's own BIC at the selected model, computed identically, for reporting.
    let gam_bic = gaussian_profile_bic(&theta, &dz, gam_coef);
    let gam_coef_flat: Vec<f64> = gam_coef.iter().copied().collect();
    assert_eq!(py_coef.len(), gam_coef_flat.len(), "coef length mismatch");
    let coef_rel = relative_l2(&gam_coef_flat, py_coef);

    eprintln!(
        "sindy auto-λ BIC: n={n} P={P} D={D} | gam_lam={gam_lam:.3e} py_lam={py_lam:.3e} \
         | gam_bic={gam_bic:.3} py_bic={py_bic:.3} | gam_support={gam_support} \
         py_support={py_support} | coef_rel_l2={coef_rel:.4e}"
    );

    // (1) Selected λ within a factor of 2 (one grid step in a decade-spaced
    // grid is 10×, so factor-2 is well inside a single step): the two engines
    // must land on the same grid point.
    let lam_ratio = (gam_lam / py_lam).max(py_lam / gam_lam);
    assert!(
        lam_ratio <= 2.0,
        "selected λ disagree beyond factor 2: gam={gam_lam:.3e} py={py_lam:.3e} (ratio={lam_ratio:.3})"
    );

    // (2) Identical support cardinality: both must recover the same sparse
    // structure of the true dynamics.
    assert_eq!(
        gam_support, py_support,
        "active-support cardinality differs: gam={gam_support} py={py_support}"
    );

    // (3) Coefficient matrices coincide. With near-exact derivatives both engines
    // recover the same support and refit ridge-regularized LS on it, so the
    // fitted coefficients agree to well under 1% relative L2; 2e-2 leaves margin
    // for the differing ridge solvers (faer Cholesky vs sklearn) while still
    // catching any structural disagreement.
    assert!(
        coef_rel < 2.0e-2,
        "selected coefficient matrices diverge from pysindy: rel_l2={coef_rel:.4e}"
    );
}

/// Gaussian profile-likelihood BIC `Σ_c [ n·log(RSS_c/n) + k_c·log(n) ]` —
/// the same formula gam's `sindy_stlsq_auto_lam` minimizes, recomputed here on
/// the selected model for head-to-head reporting against pysindy's score.
fn gaussian_profile_bic(theta: &Array2<f64>, dz: &Array2<f64>, xi: &Array2<f64>) -> f64 {
    let n = theta.nrows();
    let d = dz.ncols();
    let resid = &theta.dot(xi) - dz;
    let n_f = n as f64;
    let mut bic = 0.0;
    for c in 0..d {
        let rss: f64 = resid
            .column(c)
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .max(1.0e-300);
        let k = xi.column(c).iter().filter(|&&v| v != 0.0).count() as f64;
        bic += n_f * (rss / n_f).ln() + k * n_f.ln();
    }
    bic
}

/// Right-pad `v` with NaN up to `len` so every wire column shares one row count.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(v);
    out.resize(len, f64::NAN);
    out
}
