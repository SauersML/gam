//! E4 (gam#2234) — zoo ground-truth validation of the on-manifold steering
//! primitive, no LLM, runs locally.
//!
//! A K=1 circle (periodic-harmonic) atom is fit on a PLANTED circle whose
//! per-row angle `θ_i = 2π i / n` is known analytically. Steering the fitted
//! atom's coordinate by the manifold group action `t ⊕ δ` (Circle phase add,
//! [`LatentManifold::retract`]) must move the decoded reconstruction to the
//! planted manifold point at `θ_i + Δ`:
//!
//! * `δ = 0`      ⇒ an EXACTLY zero ambient delta (retraction is idempotent on
//!                  the already-wrapped coordinates);
//! * `δ = period` ⇒ circle closure, `|Δ| ≈ 0` (return to start);
//! * `δ = period/4` ⇒ the steered decode matches the planted circle rotated by
//!                  `±π/2` with high R² (the sign is the fit's orientation
//!                  gauge — the periodic basis recovers `2π t = ±θ + φ`, so a
//!                  quarter-period `t`-step is a `±π/2` `θ`-rotation).
//!
//! This validates the primitive itself (the group action + basis refresh +
//! decode delta) before any model surgery, in the local Rust harness.

use super::tests::global_ev;
use super::tests_startup_validation_1782::{Topo, build_term};
use super::*;
use ndarray::{Array1, Array2, array};

/// Deterministic pseudo-noise in `[-0.5, 0.5)` — reproducible without an RNG.
fn det_noise(a: usize, b: usize) -> f64 {
    let s = ((a as f64 + 1.0) * 12.9898 + (b as f64 + 1.0) * 78.233).sin() * 43758.5453;
    s - s.floor() - 0.5
}

/// Plant a unit circle in a generic 2-plane of `ℝ^p` with known per-row angle
/// `θ_i = 2π i / n`. Returns `(z, u, v)` where the NOISE-FREE manifold point at
/// angle `φ` is `cos φ · u + sin φ · v` with `(u, v)` orthonormal — so the
/// ground-truth steered point at `θ_i + Δ` is analytic and needs no refit.
fn planted_circle_with_frame(n: usize, p: usize, sigma: f64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut u = Array1::<f64>::zeros(p);
    let mut v = Array1::<f64>::zeros(p);
    for j in 0..p {
        u[j] = ((j as f64 + 1.0) * 0.7).sin();
        v[j] = ((j as f64 + 1.0) * 0.7).cos();
    }
    // Gram-Schmidt orthonormalize (u, v) so the planted ring is a true circle.
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    let uv = u.dot(&v);
    for j in 0..p {
        v[j] -= uv * u[j];
    }
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);

    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (theta.cos(), theta.sin());
        for j in 0..p {
            z[[i, j]] = c * u[j] + s * v[j] + sigma * det_noise(i, j);
        }
    }
    (z, u, v)
}

/// The analytic (noise-free) planted point at `θ_i + delta_theta` for every row.
fn analytic_moved(u: &Array1<f64>, v: &Array1<f64>, n: usize, delta_theta: f64) -> Array2<f64> {
    let p = u.len();
    let mut out = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = std::f64::consts::TAU * i as f64 / n as f64 + delta_theta;
        let (c, s) = (theta.cos(), theta.sin());
        for j in 0..p {
            out[[i, j]] = c * u[j] + s * v[j];
        }
    }
    out
}

#[test]
fn zz_e4_circle_steer_matches_planted_moved() {
    let n = 240usize;
    let p = 6usize;
    let sigma = 0.01;
    let (z, u, v) = planted_circle_with_frame(n, p, sigma);

    // K=1 circle atom, softmax assignment (a single-atom softmax gate is exactly
    // 1.0, so the gate leaves the decode untouched — the cleanest E4 baseline).
    let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
    let mut rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]; 1],
    );
    term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
        .expect("E4 K=1 circle joint fit must run e2e");
    let fitted = term.try_fitted().expect("E4 fitted reconstruction");
    let native_ev = global_ev(z.view(), fitted.view());
    eprintln!("[E4] native K=1 circle fit EV = {native_ev:.4}");
    // Signal floor: the steering test is only meaningful if the atom actually
    // recovered the planted circle (a clean rank-2 ring; a healthy K=1 circle
    // fit clears this easily).
    assert!(
        native_ev > 0.8,
        "E4 fit must recover the planted circle before steering (EV={native_ev:.4})"
    );

    let rows: Vec<usize> = (0..n).collect();

    // (a) δ = 0 is an EXACTLY zero ambient delta (idempotent retraction on the
    //     already-wrapped fitted coordinates ⇒ bit-identical decode ⇒ zero).
    let zero = term
        .steer_rows(0, &rows, array![0.0].view())
        .expect("E4 steer δ=0");
    let max_zero = zero.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert_eq!(
        max_zero, 0.0,
        "steer by δ=0 must be exactly zero, got max|Δ|={max_zero:e}"
    );

    // (b) δ = full period ⇒ circle closure: the group action returns to the
    //     start, so the ambient delta is ≈ 0 up to floating-point wrap.
    let closed = term
        .steer_rows(0, &rows, array![1.0].view())
        .expect("E4 steer δ=period");
    let max_closed = closed.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(
        max_closed < 1.0e-6,
        "steer by a full period must close the circle (|Δ|≈0), got max|Δ|={max_closed:e}"
    );

    // (c) δ = period/4 ⇒ the steered decode matches the planted circle rotated
    //     by ±π/2 (orientation gauge). Compare the ABSOLUTE steered contribution
    //     (steer_decode) against the analytic moved points at θ ± 2π/4.
    let quarter = std::f64::consts::FRAC_PI_2; // 2π/4
    let steered = term
        .steer_decode(0, &rows, array![0.25].view())
        .expect("E4 steer δ=period/4");
    let moved_plus = analytic_moved(&u, &v, n, quarter);
    let moved_minus = analytic_moved(&u, &v, n, -quarter);
    let r2_plus = global_ev(moved_plus.view(), steered.view());
    let r2_minus = global_ev(moved_minus.view(), steered.view());
    let r2 = r2_plus.max(r2_minus);
    eprintln!(
        "[E4] steered-vs-planted R²: +π/2={r2_plus:.4}  −π/2={r2_minus:.4}  (orientation gauge)"
    );
    assert!(
        r2 > 0.9,
        "E4 steered decode must match the planted moved circle at θ±π/2 \
         (best R²={r2:.4}; +={r2_plus:.4}, −={r2_minus:.4})"
    );
}
