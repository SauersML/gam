//! #1404 / #1464 regression guard: the constant-curvature (`curv`) smooth's
//! profiled-REML criterion must IDENTIFY THE SIGN of the true curvature — it
//! must prefer a negative κ for hyperbolic-shaped data and a positive κ for
//! spherical-shaped data, instead of railing to the positive chart bound for
//! every dataset (the #1464 symptom: hyperbolic truth recovered as spherical).
//!
//! This drives the PUBLIC basis API (`build_constant_curvature_basis` at each
//! candidate κ + a self-contained profiled-Gaussian-REML λ sweep) rather than a
//! crate-private oracle, so it guards the user-visible path the #1404 cluster
//! reported on. Two things make the sign identifiable and are exercised here:
//!   * the fill-invariant effective length L(κ) baked into the basis builder
//!     (so changing κ moves the geodesic-distance SHAPE, not the kernel
//!     resolution), and
//!   * NO curvature-blind double-penalty ridge (#1464 default), which otherwise
//!     absorbs the fit independently of κ and lets the criterion rail.
//!
//! Reference-as-truth: every assertion is against the argmin of gam's own
//! profiled REML criterion over a κ grid — never another tool's output.

use gam::terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_kernel_matrix,
    realized_constant_curvature_length_scale,
};
use ndarray::{Array1, Array2};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Reproducible data on a disk of radius 0.45 (inside every κ chart in the grid).
fn disk_points(n: usize, seed: u64) -> Array2<f64> {
    let mut st = seed;
    let mut pts = Array2::<f64>::zeros((n, 2));
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        pts[[filled, 0]] = a * 0.45;
        pts[[filled, 1]] = b * 0.45;
        filled += 1;
    }
    pts
}

/// A curvature-shaped response: a smooth radial signal whose kernel "shape" is
/// generated at the TRUE κ (via the geodesic-exponential kernel), plus noise.
/// The shape — not the amplitude — carries the curvature sign.
fn curved_response(data: &Array2<f64>, kappa_true: f64, ell: f64, seed: u64) -> Array1<f64> {
    // A single radial bump centered at the origin under the true geometry.
    let center = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let k = constant_curvature_kernel_matrix(data.view(), center.view(), kappa_true, ell).unwrap();
    let mut st = seed ^ 0xD1B5_4A32;
    let mut y = Array1::<f64>::zeros(data.nrows());
    for i in 0..data.nrows() {
        y[i] = 2.0 * k[[i, 0]] - 1.0 + 0.02 * next_gauss(&mut st);
    }
    y
}

/// Profiled Gaussian REML deviance for design `b`, response `y`, single penalty
/// `s`, minimized over a log-λ grid. Mirrors the in-crate constant-curvature
/// oracle; self-contained so this test needs no crate-private helper.
fn profiled_reml(b: &Array2<f64>, y: &Array1<f64>, s: &Array2<f64>) -> f64 {
    use gam::linalg::faer_ndarray::FaerEigh;
    let n = b.nrows();
    let p = b.ncols();
    let btb = {
        let m = b.t().dot(b);
        (&m + &m.t()) * 0.5
    };
    let bty = b.t().dot(y);
    let s_sym = (s + &s.t()) * 0.5;
    let (s_evals, _sv) = FaerEigh::eigh(&s_sym, faer::Side::Lower).unwrap();
    let s_max = s_evals.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    let s_tol = s_max * 1e-9;
    let r = s_evals.iter().filter(|&&e| e > s_tol).count();
    let m_p = p - r;
    let dof = (n - m_p) as f64;
    let log_det_s_plus: f64 = s_evals
        .iter()
        .filter(|&&e| e > s_tol)
        .map(|&e| e.ln())
        .sum();
    let mut best = f64::INFINITY;
    for grid in -24i32..=24 {
        let lam = (0.5 * f64::from(grid)).exp();
        let h = {
            let m = &btb + &s_sym.mapv(|v| v * lam) + &(Array2::<f64>::eye(p) * (1e-10 * s_max.max(1.0)));
            (&m + &m.t()) * 0.5
        };
        let (hv, hq) = FaerEigh::eigh(&h, faer::Side::Lower).unwrap();
        let qty = hq.t().dot(&bty);
        let mut beta = Array1::<f64>::zeros(p);
        let mut log_det_h = 0.0_f64;
        for i in 0..p {
            let ev = hv[i].max(1e-300);
            log_det_h += ev.ln();
            let coef = qty[i] / ev;
            for j in 0..p {
                beta[j] += hq[[j, i]] * coef;
            }
        }
        let resid = y - &b.dot(&beta);
        let rss = resid.dot(&resid).max(1e-300);
        let log_det_lam_s = (r as f64) * lam.ln() + log_det_s_plus;
        let dev = dof * (rss / dof).ln() + log_det_h - log_det_lam_s;
        if dev < best {
            best = dev;
        }
    }
    best
}

/// argmin of the profiled criterion over a κ grid, for a given true curvature.
fn argmin_kappa(data: &Array2<f64>, ell_ref: f64, kappa_true: f64) -> f64 {
    let y = curved_response(data, kappa_true, ell_ref, 11);
    let grid: Vec<f64> = (-30..=30).map(|i| f64::from(i) * 0.1).collect();
    let mut best_k = f64::NAN;
    let mut best_v = f64::INFINITY;
    for &kappa in &grid {
        let spec = ConstantCurvatureBasisSpec {
            // A modest farthest-point center set keeps the per-κ eigensolves
            // cheap across the grid while still resolving the radial signal.
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
            kappa,
            length_scale: ell_ref,
            // No ridge: the curvature-blind double penalty defeats sign
            // identification (#1464). The RKHS Gram alone is full-rank PD.
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        };
        let built = build_constant_curvature_basis(data.view(), &spec).unwrap();
        let b = built.design.to_dense();
        assert_eq!(
            built.penalties.len(),
            1,
            "with double_penalty=false the curv smooth must expose exactly one (RKHS Gram) penalty"
        );
        let v = profiled_reml(&b, &y, &built.penalties[0]);
        if v < best_v {
            best_v = v;
            best_k = kappa;
        }
    }
    best_k
}

#[test]
fn curv_profiled_reml_identifies_curvature_sign_both_ways() {
    let data = disk_points(220, 0xC0FF_EE12);
    // κ=0 reference length (auto chart spacing) — the L(κ) target is pinned to it.
    let ell_ref = realized_constant_curvature_length_scale(data.view(), 0.0).unwrap();

    let k_hyp = argmin_kappa(&data, ell_ref, -2.0);
    let k_sph = argmin_kappa(&data, ell_ref, 2.0);
    eprintln!("[#1404] curvature-sign recovery: hyperbolic κ̂={k_hyp:.2}  spherical κ̂={k_sph:.2}");

    assert!(
        k_hyp < 0.0,
        "hyperbolic truth (κ⋆=−2) must recover NEGATIVE curvature; got κ̂={k_hyp} \
         (the #1464 bug rails this to the +chart bound)"
    );
    assert!(
        k_sph > 0.0,
        "spherical truth (κ⋆=+2) must recover POSITIVE curvature; got κ̂={k_sph}"
    );
    // The two signs must be genuinely DISTINGUISHED, not a coincidence of one bound.
    assert!(
        k_hyp < k_sph,
        "curvature criterion must separate hyperbolic from spherical truth: \
         hyperbolic κ̂={k_hyp} should be below spherical κ̂={k_sph}"
    );
}
