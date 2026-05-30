//! End-to-end quality: gam's SINDy Sequential Thresholded Least Squares
//! (STLSQ) sparse-identification solver must match **PySINDy** — the reference
//! Python implementation of SINDy (Brunton, Proctor & Kutz, PNAS 2016, and the
//! `pysindy` package by de Silva et al., JOSS 2020) — on the canonical
//! Lorenz-63 benchmark.
//!
//! Why this comparator: PySINDy's `STLSQ` optimizer *is* the same algorithm as
//! `gam::solver::sindy::sindy_stlsq_solve` — sequential thresholded least
//! squares with a ridge regularizer on the active set. STLSQ is a deterministic
//! fixed-point iteration: fed an identical library matrix `Θ` and identical
//! target derivative matrix `Ẋ`, with identical threshold and ridge `λ`, the two
//! engines must converge to the *same* sparse support (the same coefficients set
//! to exactly zero) and the *same* coefficient values to within solver
//! precision. Lorenz-63 is the headline system of Brunton 2016; a divergence
//! here is a genuine bug in either support recovery or coefficient accuracy.
//!
//! To eliminate every confound we do NOT let either engine build its own
//! library or estimate its own derivatives: we construct ONE library matrix and
//! ONE analytic-derivative matrix in Rust and hand the *identical bytes* to both
//! `sindy_stlsq_solve` and PySINDy's bare `STLSQ` optimizer (the precomputed
//! library matrix Θ is passed as `x`, the analytic derivatives Ẋ as `y`). The
//! only thing under test is the STLSQ fixed point.
//!
//! Library (7 terms, in fixed column order): [1, x, y, z, x*y, x*z, y*z].
//! This is exactly the span needed to represent Lorenz-63:
//!   dx/dt = σ(y - x)          = -σ·x + σ·y
//!   dy/dt = x(ρ - z) - y      = ρ·x - y - x·z
//!   dz/dt = x·y - β·z         = -β·z + x·y
//! so the true 7×3 coefficient matrix (rows = library terms, cols = dx,dy,dz) is
//! sparse with exactly 7 nonzeros. Recovering this exact support is the point of
//! SINDy.

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_solve};
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::Array2;

// Lorenz-63 parameters (Brunton 2016, the standard chaotic regime).
const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;

// Trajectory length and integration step.
const N: usize = 2000;
const DT: f64 = 0.002;

// STLSQ hyperparameters — identical for both engines.
//
// Ridge `λ = 0`: with exact analytic derivatives and the exact spanning library
// the *unregularized* least-squares solution on the recovered support IS the
// true Lorenz matrix to machine precision (the spurious off-support columns have
// OLS coefficients ~1e-12, far below THRESHOLD). Any λ > 0 would bias every
// coefficient by O(λ) ≈ 1e-3 away from truth (verified: λ=0.05 ⇒ max|Δ|≈1e-3),
// which would falsify the "matches truth to machine precision" claim below AND,
// because gam refits all output columns on the shared *union* active set while
// PySINDy refits each target on its own per-column support, would make the two
// engines' ridge-coupled solves disagree by O(λ) too. At λ=0 the OLS solve
// decouples per column, so gam, PySINDy, and the truth all coincide to ~1e-12.
const THRESHOLD: f64 = 0.1;
const RIDGE_LAM: f64 = 0.0;
const MAX_ROUNDS: usize = 20;

/// Lorenz-63 right-hand side: returns (dx/dt, dy/dt, dz/dt) analytically.
fn lorenz_rhs(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let dx = SIGMA * (y - x);
    let dy = x * (RHO - z) - y;
    let dz = x * y - BETA * z;
    (dx, dy, dz)
}

/// One classical RK4 step of size `DT`.
fn rk4_step(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (k1x, k1y, k1z) = lorenz_rhs(x, y, z);
    let (k2x, k2y, k2z) = lorenz_rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y, z + 0.5 * DT * k1z);
    let (k3x, k3y, k3z) = lorenz_rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y, z + 0.5 * DT * k2z);
    let (k4x, k4y, k4z) = lorenz_rhs(x + DT * k3x, y + DT * k3y, z + DT * k3z);
    (
        x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + DT / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )
}

#[test]
fn gam_sindy_matches_pysindy_on_lorenz63() {
    // ---- generate the canonical Lorenz-63 trajectory (fixed seed=0) --------
    // "seed=0" fixes the deterministic initial condition; the trajectory is
    // fully reproducible RK4 with no stochasticity. Both engines see the exact
    // same data, so the comparison is confound-free.
    let mut x = 1.0_f64; // deterministic IC for seed 0
    let mut y = 1.0_f64;
    let mut z = 1.0_f64;
    let mut xs = Vec::with_capacity(N);
    let mut ys = Vec::with_capacity(N);
    let mut zs = Vec::with_capacity(N);
    for _ in 0..N {
        xs.push(x);
        ys.push(y);
        zs.push(z);
        let (nx, ny, nz) = rk4_step(x, y, z);
        x = nx;
        y = ny;
        z = nz;
    }

    // ---- analytic derivatives at every sampled state ----------------------
    let mut dx = Vec::with_capacity(N);
    let mut dy = Vec::with_capacity(N);
    let mut dz = Vec::with_capacity(N);
    for i in 0..N {
        let (a, b, c) = lorenz_rhs(xs[i], ys[i], zs[i]);
        dx.push(a);
        dy.push(b);
        dz.push(c);
    }

    // ---- build the shared 7-term library Θ: [1, x, y, z, x*y, x*z, y*z] ----
    // Fixed column order — both engines index it identically.
    const P: usize = 7;
    let mut theta = Array2::<f64>::zeros((N, P));
    // We also keep flat column vectors to ship the *exact same* feature bytes
    // to Python (so PySINDy operates on identical precomputed features rather
    // than recomputing products in float and risking a different rounding).
    let mut f_one = Vec::with_capacity(N);
    let mut f_x = Vec::with_capacity(N);
    let mut f_y = Vec::with_capacity(N);
    let mut f_z = Vec::with_capacity(N);
    let mut f_xy = Vec::with_capacity(N);
    let mut f_xz = Vec::with_capacity(N);
    let mut f_yz = Vec::with_capacity(N);
    for i in 0..N {
        let (xi, yi, zi) = (xs[i], ys[i], zs[i]);
        let cols = [1.0, xi, yi, zi, xi * yi, xi * zi, yi * zi];
        for (j, &v) in cols.iter().enumerate() {
            theta[(i, j)] = v;
        }
        f_one.push(cols[0]);
        f_x.push(cols[1]);
        f_y.push(cols[2]);
        f_z.push(cols[3]);
        f_xy.push(cols[4]);
        f_xz.push(cols[5]);
        f_yz.push(cols[6]);
    }

    // ---- target derivative matrix Ẋ: columns (dx, dy, dz) -----------------
    let mut dz_dt = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        dz_dt[(i, 0)] = dx[i];
        dz_dt[(i, 1)] = dy[i];
        dz_dt[(i, 2)] = dz[i];
    }

    // ---- gam STLSQ ---------------------------------------------------------
    let gam = sindy_stlsq_solve(
        theta.view(),
        dz_dt.view(),
        THRESHOLD,
        MAX_ROUNDS,
        RIDGE_LAM,
        SindyPenaltyKind::Ridge,
        3.7,
    )
    .expect("gam SINDy STLSQ must succeed on Lorenz-63");

    // gam coefficient matrix is (P, 3): row = library term, col = derivative.
    let gam_xi: Vec<f64> = (0..3)
        .flat_map(|c| (0..P).map(move |r| (r, c)))
        .map(|(r, c)| gam.coefficients[(r, c)])
        .collect(); // column-major: [dx terms..., dy terms..., dz terms...]

    // ---- PySINDy STLSQ on the identical precomputed library + derivatives --
    // We feed PySINDy's bare STLSQ optimizer the precomputed feature matrix Θ
    // directly as `x` and the analytic derivatives Ẋ directly as `y`. There is
    // no feature library and no derivative estimation in the loop: the optimizer
    // runs STLSQ on the identical (Θ, Ẋ) bytes handed to gam — apples to apples.
    let cols = [
        Column::new("f_one", &f_one),
        Column::new("f_x", &f_x),
        Column::new("f_y", &f_y),
        Column::new("f_z", &f_z),
        Column::new("f_xy", &f_xy),
        Column::new("f_xz", &f_xz),
        Column::new("f_yz", &f_yz),
        Column::new("ddx", &dx),
        Column::new("ddy", &dy),
        Column::new("ddz", &dz),
    ];

    let body = format!(
        r#"
import numpy as np
from pysindy.optimizers import STLSQ

# Identical 7-term library bytes, in the same column order as the Rust Theta.
Theta = np.column_stack([
    df["f_one"].to_numpy(), df["f_x"].to_numpy(), df["f_y"].to_numpy(),
    df["f_z"].to_numpy(),  df["f_xy"].to_numpy(), df["f_xz"].to_numpy(),
    df["f_yz"].to_numpy(),
])
Xdot = np.column_stack([
    df["ddx"].to_numpy(), df["ddy"].to_numpy(), df["ddz"].to_numpy(),
])

# STLSQ with the SAME threshold and ridge alpha as gam. alpha is the L2
# (ridge) weight on the active set in both implementations: sklearn's
# Ridge(alpha) inside STLSQ solves (ThetaᵀTheta + alpha·I) w = Thetaᵀy, exactly
# gam's ridge_diag_solve convention. normalize_columns=False so the columns are
# the verbatim Theta bytes (gam does not column-normalize either).
opt = STLSQ(threshold={threshold}, alpha={ridge}, max_iter={max_iter},
            normalize_columns=False)

# Fit the bare optimizer directly on (Theta, Xdot): pysindy's SINDy wrapper
# would build/own the library + estimate derivatives, which we deliberately
# bypass so the only thing exercised is the STLSQ fixed point on inputs that are
# byte-identical to the ones handed to gam.
opt.fit(Theta, Xdot)

# opt.coef_ is (n_targets, n_features) = (3, 7). gam stores (7, 3) and we
# flattened column-major as [dx-terms, dy-terms, dz-terms]; match that.
coef = np.asarray(opt.coef_)           # (3, 7) = (targets, features)
xi = coef.T                            # (7, 3) to match gam's (P, d) layout
emit("coef", xi.reshape(-1, order="F"))           # [dx-terms, dy-terms, dz-terms]
emit("nonzero", (xi != 0.0).reshape(-1, order="F").astype(float))
"#,
        threshold = THRESHOLD,
        ridge = RIDGE_LAM,
        max_iter = MAX_ROUNDS,
    );

    let py = run_python(&cols, &body);
    let py_xi = py.vector("coef");
    let py_nonzero = py.vector("nonzero");
    assert_eq!(py_xi.len(), P * 3, "pysindy coef must be 7x3 flattened");
    assert_eq!(py_nonzero.len(), P * 3, "pysindy support must be 7x3");

    // ---- sparsity-pattern (support) agreement: EXACT ----------------------
    // Both engines run the same hard-threshold STLSQ at the same threshold, so
    // the set of nonzero coefficients must agree element-wise with NO mismatch.
    let term_names = ["1", "x", "y", "z", "x*y", "x*z", "y*z"];
    let deriv_names = ["dx/dt", "dy/dt", "dz/dt"];
    let mut support_mismatches = 0usize;
    for c in 0..3 {
        for r in 0..P {
            let k = c * P + r;
            let gam_nz = gam_xi[k] != 0.0;
            let py_nz = py_nonzero[k] != 0.0;
            if gam_nz != py_nz {
                support_mismatches += 1;
                eprintln!(
                    "SUPPORT mismatch {} term {}: gam_nz={} py_nz={} (gam={:.6}, py={:.6})",
                    deriv_names[c], term_names[r], gam_nz, py_nz, gam_xi[k], py_xi[k]
                );
            }
        }
    }

    // ---- coefficient agreement on the recovered matrix --------------------
    let coef_max_abs = max_abs_diff(&gam_xi, py_xi);

    // ---- recovery sanity: the true Lorenz support must be the 7 nonzeros ---
    // true (col-major) nonzero pattern over [dx, dy, dz]:
    //   dx: -sigma at x(row1), +sigma at y(row2)
    //   dy: +rho at x(row1), -1 at y(row2), -1 at x*z(row5)
    //   dz: -beta at z(row3), +1 at x*y(row4)
    let mut true_xi = [0.0f64; P * 3];
    true_xi[0 * P + 1] = -SIGMA; // dx : x
    true_xi[0 * P + 2] = SIGMA; // dx : y
    true_xi[1 * P + 1] = RHO; // dy : x
    true_xi[1 * P + 2] = -1.0; // dy : y
    true_xi[1 * P + 5] = -1.0; // dy : x*z
    true_xi[2 * P + 3] = -BETA; // dz : z
    true_xi[2 * P + 4] = 1.0; // dz : x*y
    let gam_vs_true = max_abs_diff(&gam_xi, &true_xi);
    let py_vs_true = max_abs_diff(py_xi, &true_xi);

    eprintln!("=== SINDy Lorenz-63: gam STLSQ vs PySINDy STLSQ ===");
    eprintln!(
        "rounds_used={} converged={}",
        gam.rounds_used, gam.converged
    );
    eprintln!("support mismatches (gam vs pysindy): {support_mismatches}");
    eprintln!("coef max|Δ| gam-vs-pysindy : {coef_max_abs:.3e}");
    eprintln!("coef max|Δ| gam-vs-true    : {gam_vs_true:.3e}");
    eprintln!("coef max|Δ| pysindy-vs-true: {py_vs_true:.3e}");

    // ---- assertions --------------------------------------------------------
    // (1) Both engines must recover the EXACT Lorenz support — anything else
    //     means STLSQ failed on its headline benchmark. Sanity-gate so a shared
    //     bug recovering the wrong (but identical) support can't pass silently.
    assert!(
        gam_vs_true < 1.0e-6,
        "gam STLSQ must recover true Lorenz coefficients (max|Δ|={gam_vs_true:.3e}); \
         with exact analytic derivatives and the exact spanning library the \
         least-squares solution is the true matrix to machine precision"
    );
    assert!(
        py_vs_true < 1.0e-6,
        "pysindy STLSQ must recover true Lorenz coefficients (max|Δ|={py_vs_true:.3e})"
    );

    // (2) Sparsity pattern must match EXACTLY — same algorithm, same threshold,
    //     same inputs: zero positions must coincide with no slack.
    assert_eq!(
        support_mismatches, 0,
        "gam and pysindy STLSQ must agree on the sparse support element-wise"
    );

    // (3) Coefficient agreement: at λ=0 both engines solve the same OLS normal
    //     equations on the same recovered support, so they differ only by
    //     linear-algebra rounding (~1e-12 in practice). 1e-6 is tight enough to
    //     catch a real coefficient bug — e.g. the O(1e-3) drift a stray ridge
    //     term or a transposed/mis-scaled solve would introduce — while
    //     tolerating Cholesky-vs-pysindy-solver floating-point differences.
    assert!(
        coef_max_abs < 1.0e-6,
        "gam and pysindy STLSQ coefficients must match to solver precision \
         (max|Δ|={coef_max_abs:.3e})"
    );
}
