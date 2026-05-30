//! End-to-end quality: gam's SINDy STLSQ coefficient recovery must match
//! `pysindy` — the de-facto reference implementation of Sparse Identification of
//! Nonlinear Dynamics (Brunton, Proctor & Kutz, PNAS 2016) — on a known linear
//! ODE.
//!
//! We benchmark `gam::solver::sindy::sindy_stlsq_solve` against
//! `pysindy.optimizers.STLSQ` (the canonical SINDy optimizer). The system is the
//! damped harmonic oscillator
//!
//!     d(x)/dt = v
//!     d(v)/dt = -2 ζ ω₀ v - ω₀² x          (ζ = 0.1, ω₀ = 2 rad/s)
//!
//! integrated by fixed-step RK4 (dt = 0.01, 1500 rows, fixed initial condition),
//! with state derivatives recovered numerically by centered finite differences.
//! Both engines receive the IDENTICAL feature library Θ = [1, x, v] and the
//! IDENTICAL numeric-derivative target Ẋ = [d(x)/dt, d(v)/dt], and both run with
//! ridge weight λ = 0 (pure least squares) and threshold tol = 0.01. They
//! therefore solve the same sequential-thresholded least-squares problem and
//! must agree.
//!
//! A linear ODE has a single ground-truth sparse coefficient matrix, so this is
//! the cleanest possible SINDy test. We assert:
//!   1. exact agreement of the recovered sparsity pattern (support), and
//!   2. tight relative-L2 agreement of the coefficient matrices.
//! Because both engines solve the same thresholded normal equations, a real
//! divergence is a real bug in gam's STLSQ numerics — not a modeling choice.

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_solve};
use gam::test_support::reference::{Column, relative_l2, run_python};
use ndarray::Array2;

#[test]
fn gam_sindy_stlsq_matches_pysindy_on_damped_oscillator() {
    // ---- ground-truth physical constants ---------------------------------
    let zeta = 0.1_f64; // damping ratio
    let omega0 = 2.0_f64; // natural frequency [rad/s]
    let dt = 0.01_f64;
    let n_steps = 1500usize;

    // dz/dt = f(z): z = (x, v); damped harmonic oscillator.
    let rhs = |x: f64, v: f64| -> (f64, f64) {
        let dx = v;
        let dv = -2.0 * zeta * omega0 * v - omega0 * omega0 * x;
        (dx, dv)
    };

    // ---- fixed-seed (deterministic) RK4 trajectory -----------------------
    // Fixed initial condition; no randomness — the trajectory is reproducible
    // bit-for-bit and fed identically to both engines.
    let mut x = 1.0_f64;
    let mut v = 0.0_f64;
    let mut xs = Vec::<f64>::with_capacity(n_steps);
    let mut vs = Vec::<f64>::with_capacity(n_steps);
    for _ in 0..n_steps {
        xs.push(x);
        vs.push(v);
        // classic RK4 step on the 2-state system.
        let (k1x, k1v) = rhs(x, v);
        let (k2x, k2v) = rhs(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v);
        let (k3x, k3v) = rhs(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v);
        let (k4x, k4v) = rhs(x + dt * k3x, v + dt * k3v);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }

    // ---- numeric derivatives via centered finite differences --------------
    // Centered differences on the interior; the two endpoints are dropped so
    // that every retained row has a true centered estimate. This identical
    // (x, v, dx/dt, dv/dt) table is what both engines consume.
    let interior = n_steps - 2;
    let mut col_const = Vec::<f64>::with_capacity(interior);
    let mut col_x = Vec::<f64>::with_capacity(interior);
    let mut col_v = Vec::<f64>::with_capacity(interior);
    let mut dx_dt = Vec::<f64>::with_capacity(interior);
    let mut dv_dt = Vec::<f64>::with_capacity(interior);
    for i in 1..(n_steps - 1) {
        col_const.push(1.0);
        col_x.push(xs[i]);
        col_v.push(vs[i]);
        dx_dt.push((xs[i + 1] - xs[i - 1]) / (2.0 * dt));
        dv_dt.push((vs[i + 1] - vs[i - 1]) / (2.0 * dt));
    }

    // Library Θ = [1, x, v]  (n × 3); target Ẋ = [dx/dt, dv/dt]  (n × 2).
    let p = 3usize; // library columns: const, x, v
    let d = 2usize; // derivative outputs: d(x)/dt, d(v)/dt
    let mut theta = Array2::<f64>::zeros((interior, p));
    let mut dz_dt = Array2::<f64>::zeros((interior, d));
    for i in 0..interior {
        theta[(i, 0)] = col_const[i];
        theta[(i, 1)] = col_x[i];
        theta[(i, 2)] = col_v[i];
        dz_dt[(i, 0)] = dx_dt[i];
        dz_dt[(i, 1)] = dv_dt[i];
    }

    // ---- gam STLSQ: λ = 0 (pure LS), tol = 0.01, ridge family ------------
    let tol = 0.01_f64;
    let gam_res = sindy_stlsq_solve(
        theta.view(),
        dz_dt.view(),
        tol,
        20,
        0.0,
        SindyPenaltyKind::Ridge,
        3.7,
    )
    .expect("gam SINDy STLSQ must succeed on a well-conditioned linear system");
    assert!(
        gam_res.converged,
        "gam STLSQ did not converge on a linear system (rounds={})",
        gam_res.rounds_used
    );
    let gam_xi = gam_res.coefficients; // (p, d) = (3, 2)
    assert_eq!(gam_xi.dim(), (p, d));

    // Recovered coefficients should be physically sane before we even compare:
    // d(x)/dt = v  ->  [0, 0, 1];  d(v)/dt = -ω₀² x - 2ζω₀ v -> [0, -4, -0.4].
    eprintln!(
        "gam Xi: dx/dt=[{:.6}, {:.6}, {:.6}]  dv/dt=[{:.6}, {:.6}, {:.6}]",
        gam_xi[(0, 0)],
        gam_xi[(1, 0)],
        gam_xi[(2, 0)],
        gam_xi[(0, 1)],
        gam_xi[(1, 1)],
        gam_xi[(2, 1)],
    );

    // ---- pysindy STLSQ on the IDENTICAL Θ and Ẋ --------------------------
    // We hand pysindy the very same library matrix and derivative target via the
    // CSV bridge (columns const,x,v,dxdt,dvdt). pysindy's STLSQ optimizer with
    // threshold=tol and alpha=0 (no ridge) solves the same thresholded normal
    // equations gam does. We fit each derivative column independently — exactly
    // gam's per-output STLSQ — and emit the 6 coefficients in (p, d) order.
    let columns = [
        Column::new("const", &col_const),
        Column::new("xcol", &col_x),
        Column::new("vcol", &col_v),
        Column::new("dxdt", &dx_dt),
        Column::new("dvdt", &dv_dt),
    ];
    let body = r#"
import numpy as np
from pysindy.optimizers import STLSQ

Theta = np.column_stack([
    np.asarray(df["const"], dtype=float),
    np.asarray(df["xcol"], dtype=float),
    np.asarray(df["vcol"], dtype=float),
])
Xdot = np.column_stack([
    np.asarray(df["dxdt"], dtype=float),
    np.asarray(df["dvdt"], dtype=float),
])

# Same problem as gam: STLSQ with hard threshold = 0.01 and no ridge (alpha=0).
# fit_intercept=False because the constant term is an explicit library column,
# exactly as gam treats it. Solve both derivative outputs jointly.
opt = STLSQ(threshold=0.01, alpha=0.0, max_iter=20, fit_intercept=False)
opt.fit(Theta, Xdot)
coef = np.asarray(opt.coef_, dtype=float)  # shape (d, p) = (2, 3)
assert coef.shape == (2, 3), coef.shape

# Emit in gam's (p, d) layout, column-major over outputs:
# [Xi[0,0], Xi[1,0], Xi[2,0], Xi[0,1], Xi[1,1], Xi[2,1]].
flat = []
for c in range(2):
    for j in range(3):
        flat.append(coef[c, j])
emit("xi", flat)
"#;
    let py = run_python(&columns, body);
    let py_flat = py.vector("xi");
    assert_eq!(py_flat.len(), p * d, "pysindy must emit 6 coefficients");

    // pysindy emitted in (p, d) column-major order over outputs.
    let mut py_xi = Array2::<f64>::zeros((p, d));
    let mut k = 0usize;
    for c in 0..d {
        for j in 0..p {
            py_xi[(j, c)] = py_flat[k];
            k += 1;
        }
    }
    eprintln!(
        "pysindy Xi: dx/dt=[{:.6}, {:.6}, {:.6}]  dv/dt=[{:.6}, {:.6}, {:.6}]",
        py_xi[(0, 0)],
        py_xi[(1, 0)],
        py_xi[(2, 0)],
        py_xi[(0, 1)],
        py_xi[(1, 1)],
        py_xi[(2, 1)],
    );

    // ---- (1) support agreement: same nonzero pattern ---------------------
    // Both engines hard-threshold at tol, so the surviving support must be bit
    // identical. The true support is {(2,0)} and {(1,1),(2,1)}.
    let mut support_mismatch = 0usize;
    for c in 0..d {
        for j in 0..p {
            let gam_on = gam_xi[(j, c)] != 0.0;
            let py_on = py_xi[(j, c)].abs() >= tol; // pysindy zeros below threshold
            if gam_on != py_on {
                support_mismatch += 1;
                eprintln!(
                    "support mismatch at (j={j}, c={c}): gam={} (={:.3e}) pysindy={} (={:.3e})",
                    gam_on,
                    gam_xi[(j, c)],
                    py_on,
                    py_xi[(j, c)],
                );
            }
        }
    }
    assert_eq!(
        support_mismatch, 0,
        "gam and pysindy must recover the identical sparsity pattern (both threshold at {tol})",
    );

    // ---- (2) coefficient agreement: relative L2 over the 6-entry matrix ---
    let gam_vec: Vec<f64> = (0..d)
        .flat_map(|c| (0..p).map(move |j| (j, c)))
        .map(|(j, c)| gam_xi[(j, c)])
        .collect();
    let py_vec: Vec<f64> = (0..d)
        .flat_map(|c| (0..p).map(move |j| (j, c)))
        .map(|(j, c)| py_xi[(j, c)])
        .collect();
    let rel = relative_l2(&gam_vec, &py_vec);
    eprintln!("coefficient relative L2 (gam vs pysindy) = {rel:.3e}");

    // Both engines solve the same thresholded least-squares problem on the same
    // matrices; the only differences are inner linear-algebra paths (gam:
    // Cholesky normal equations; pysindy: lstsq). On this well-conditioned 3-
    // column system that is round-off, so 1e-3 relative L2 is a tight,
    // non-vacuous bound (the coefficients themselves are O(0.4-4)).
    assert!(
        rel < 1.0e-3,
        "gam SINDy coefficients diverge from pysindy: relative L2 = {rel:.3e} (bound 1e-3)",
    );
}
