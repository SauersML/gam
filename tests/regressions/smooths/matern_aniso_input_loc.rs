//! Regression tests for issue #437: the Matérn *input-location* derivative
//! (`∂Φ/∂t` and `∂²Φ/∂t∂tᵀ`) must differentiate the **same anisotropic metric**
//! the forward kernel uses, not a Euclidean surrogate.
//!
//! The forward kernel value is `Φ_{n,k} = φ(r_A)` with the anisotropic radius
//!
//! ```text
//!   r_A = √( Σ_a exp(2 ψ_a) (t_a − c_a)² ),   ψ_a = η_a − mean(η)   (Σ ψ_a = 0).
//! ```
//!
//! Both `build_matern_basis` (forward, identifiability = None ⇒ raw kernel
//! columns) and `matern_input_location_jet_nd` / `matern_input_location_hessian_nd`
//! (backward) thread `aniso_log_scales` through `centered_aniso_contrasts`
//! (the pure forward transform — an explicit all-zero η is the isotropic metric,
//! not a geometry-seeding sentinel; see #1042) and then `exp(2·)`, so the
//! analytic jets must reproduce centered finite
//! differences of the forward design *with anisotropy active*. A regression that
//! drops the metric weights `w_a = exp(2 ψ_a)` (the historical bug — Euclidean
//! `|t − c|` in the backward) differentiates a different function and is caught
//! here.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu,
    build_matern_basis_literal_aniso, matern_input_location_hessian_nd,
    matern_input_location_jet_nd,
};
use ndarray::{Array2, ArrayView2, array};

/// Raw (un-projected) Matérn kernel design `Φ_{n,k} = φ(r_A)` for the supplied
/// anisotropy, using the **public** forward path (`build_matern_basis_literal_aniso`,
/// matching the input-location jet/Hessian FFI). `identifiability = None` and
/// `include_intercept = false` keep the columns the bare kernel values, and the
/// literal entry honors an explicit all-zero η as the isotropic metric — exactly
/// the function the public input-location jet claims to differentiate (#437, #1042).
fn forward_kernel(
    points: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso: Option<&[f64]>,
) -> Array2<f64> {
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: length_scale.into(),
        nu,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: aniso.map(<[f64]>::to_vec),
    };
    build_matern_basis_literal_aniso(points, &spec)
        .expect("forward Matérn kernel should build")
        .design
        .to_dense()
}

/// Centered finite difference of the forward kernel w.r.t. input coordinate
/// `axis`, returned as an `(n_rows, n_centers)` matrix.
fn fd_first(
    points: &Array2<f64>,
    centers: &Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso: Option<&[f64]>,
    axis: usize,
    h: f64,
) -> Array2<f64> {
    let mut plus = points.clone();
    let mut minus = points.clone();
    plus.column_mut(axis).mapv_inplace(|v| v + h);
    minus.column_mut(axis).mapv_inplace(|v| v - h);
    let kp = forward_kernel(plus.view(), centers, length_scale, nu, aniso);
    let km = forward_kernel(minus.view(), centers, length_scale, nu, aniso);
    (kp - km) / (2.0 * h)
}

fn nu_label(nu: MaternNu) -> &'static str {
    match nu {
        MaternNu::Half => "1/2",
        MaternNu::ThreeHalves => "3/2",
        MaternNu::FiveHalves => "5/2",
        MaternNu::SevenHalves => "7/2",
        _ => "other",
    }
}

/// The analytic anisotropic input-location gradient equals the centered finite
/// difference of the forward kernel, across ν and dimension, with strong
/// anisotropy active.
#[test]
fn aniso_input_location_gradient_matches_forward_finite_difference() {
    let h = 1e-6;
    let cases: &[(Array2<f64>, Array2<f64>, f64, Vec<f64>)] = &[
        // d = 2, well-separated, strong anisotropy.
        (
            array![[0.27, 0.61], [0.83, 0.14], [0.51, 0.92]],
            array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1]],
            0.8,
            vec![0.9, -0.3],
        ),
        // d = 3, asymmetric anisotropy that does not sum to zero (exercises the
        // centering convention). Four well-separated rows keep the 3-center
        // kernel block full column rank: with fewer data rows than centers the
        // realized design (and the input-location jet, which mirrors the forward
        // RRQR center reduction, #755/#1937) would drop a collinear center, so the
        // fixed `0..centers.nrows()` comparison below would index a center that no
        // longer exists in the reduced basis.
        (
            array![
                [0.2, 0.5, -0.1],
                [0.7, -0.3, 0.4],
                [-0.4, 0.1, 0.6],
                [0.9, 0.8, -0.5]
            ],
            array![[0.0, 0.0, 0.0], [0.6, 0.4, -0.2], [-0.3, 0.5, 0.7]],
            1.1,
            vec![0.7, -0.2, 0.15],
        ),
    ];
    // ν = 1/2 is intentionally excluded: the Matérn-1/2 input-location
    // derivative has a singular Laplacian at center collisions for d > 1, and
    // the forward builder rejects that configuration by design.
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
    ] {
        for (points, centers, ls, aniso) in cases {
            let d = points.ncols();
            let jet = matern_input_location_jet_nd(
                points.view(),
                centers.view(),
                *ls,
                nu,
                Some(aniso.as_slice()),
            )
            .expect("analytic jet should evaluate");
            for axis in 0..d {
                let fd = fd_first(points, centers, *ls, nu, Some(aniso.as_slice()), axis, h);
                for n in 0..points.nrows() {
                    for k in 0..centers.nrows() {
                        let analytic = jet[[n, k, axis]];
                        let numeric = fd[[n, k]];
                        let tol = 1e-6 + 1e-5 * numeric.abs();
                        assert!(
                            (analytic - numeric).abs() < tol,
                            "ν={} d={d} axis={axis} (n={n},k={k}): analytic anisotropic \
                             input-location gradient {analytic} disagrees with forward FD \
                             {numeric} (|Δ|={:.3e}); the backward must use the metric \
                             r_A=√(Σ exp(2ψ_a) h_a²), not Euclidean |h|",
                            nu_label(nu),
                            (analytic - numeric).abs(),
                        );
                    }
                }
            }
        }
    }
}

/// The analytic anisotropic input-location Hessian equals the centered
/// second-order finite difference of the forward kernel (full d×d block,
/// including off-diagonal cross terms).
#[test]
fn aniso_input_location_hessian_matches_forward_finite_difference() {
    let h = 1e-4;
    let points = array![[0.27, 0.61], [0.83, 0.14], [0.55, 0.49]];
    let centers = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1]];
    let aniso = [0.85_f64, -0.35];
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
    ] {
        let ls = 0.9;
        let hess =
            matern_input_location_hessian_nd(points.view(), centers.view(), ls, nu, Some(&aniso))
                .expect("analytic Hessian should evaluate");
        let d = points.ncols();
        let k0 = forward_kernel(points.view(), &centers, ls, nu, Some(&aniso));
        for a in 0..d {
            for c in 0..d {
                // Second-order central FD of Φ w.r.t. (t_a, t_c).
                let fd_ac = if a == c {
                    let mut pp = points.clone();
                    let mut mm = points.clone();
                    pp.column_mut(a).mapv_inplace(|v| v + h);
                    mm.column_mut(a).mapv_inplace(|v| v - h);
                    let kp = forward_kernel(pp.view(), &centers, ls, nu, Some(&aniso));
                    let km = forward_kernel(mm.view(), &centers, ls, nu, Some(&aniso));
                    (kp - 2.0 * &k0 + km) / (h * h)
                } else {
                    let mut ppp = points.clone();
                    let mut ppm = points.clone();
                    let mut pmp = points.clone();
                    let mut pmm = points.clone();
                    ppp.column_mut(a).mapv_inplace(|v| v + h);
                    ppp.column_mut(c).mapv_inplace(|v| v + h);
                    ppm.column_mut(a).mapv_inplace(|v| v + h);
                    ppm.column_mut(c).mapv_inplace(|v| v - h);
                    pmp.column_mut(a).mapv_inplace(|v| v - h);
                    pmp.column_mut(c).mapv_inplace(|v| v + h);
                    pmm.column_mut(a).mapv_inplace(|v| v - h);
                    pmm.column_mut(c).mapv_inplace(|v| v - h);
                    let kpp = forward_kernel(ppp.view(), &centers, ls, nu, Some(&aniso));
                    let kpm = forward_kernel(ppm.view(), &centers, ls, nu, Some(&aniso));
                    let kmp = forward_kernel(pmp.view(), &centers, ls, nu, Some(&aniso));
                    let kmm = forward_kernel(pmm.view(), &centers, ls, nu, Some(&aniso));
                    (kpp - kpm - kmp + kmm) / (4.0 * h * h)
                };
                for n in 0..points.nrows() {
                    for k in 0..centers.nrows() {
                        let analytic = hess[[n, k, a, c]];
                        let numeric = fd_ac[[n, k]];
                        let tol = 5e-4 + 5e-3 * numeric.abs();
                        assert!(
                            (analytic - numeric).abs() < tol,
                            "ν={} Hessian[{a},{c}] (n={n},k={k}): analytic {analytic} vs \
                             forward FD {numeric} (|Δ|={:.3e})",
                            nu_label(nu),
                            (analytic - numeric).abs(),
                        );
                    }
                }
            }
        }
    }
}

/// Independent closed-form oracle (a *different angle* from finite differences).
///
/// ν = 3/2, ℓ = 1, center = (0,0), point = (1,0), η = (ln 8, 0). The forward
/// metric centers η: ψ = (ln8/2, −ln8/2) ⇒ w = (8, 1/8), so
/// `r_A = √(8·1² + (1/8)·0²) = 2√2`. With φ(r) = (1+√3 r)e^{−√3 r},
/// φ'(r) = −3r e^{−√3 r}, the exact input gradient is
///
/// ```text
///   ∂Φ/∂t_0 = φ'(r_A) · w_0 · h_0 / r_A = (−3·2√2 e^{−2√6}) · 8 · 1 / (2√2) = −24 e^{−2√6},
///   ∂Φ/∂t_1 = φ'(r_A) · w_1 · h_1 / r_A = 0                       (h_1 = 0).
/// ```
///
/// A Euclidean backward would instead report φ'(1)·1 = −3 e^{−√3} ≈ −0.531 for
/// `∂/∂t_0` — wildly different from the metric value −24 e^{−2√6} ≈ −0.179 — and
/// would make `∂/∂t_1` nonzero, so this pins the metric weighting, the centering
/// convention, and rules out a Euclidean regression by a wide margin.
#[test]
fn aniso_gradient_matches_closed_form_centered_metric() {
    let points = array![[1.0, 0.0]];
    let centers = array![[0.0, 0.0]];
    let aniso = [8.0_f64.ln(), 0.0];
    let jet = matern_input_location_jet_nd(
        points.view(),
        centers.view(),
        1.0,
        MaternNu::ThreeHalves,
        Some(&aniso),
    )
    .expect("jet should evaluate");

    let expected_t0 = -24.0 * (-2.0 * (6.0_f64).sqrt()).exp();
    assert!(
        (jet[[0, 0, 0]] - expected_t0).abs() < 1e-12,
        "centered-metric ∂Φ/∂t_0 should be −24 e^{{−2√6}} = {expected_t0}, got {}",
        jet[[0, 0, 0]]
    );
    assert!(
        jet[[0, 0, 1]].abs() < 1e-14,
        "∂Φ/∂t_1 must vanish (h_1 = 0); got {}",
        jet[[0, 0, 1]]
    );

    // A Euclidean (metric-blind) backward would land near −0.531 on t_0; assert
    // we are nowhere near it, locking out a regression to |h|.
    let euclidean_wrong = -3.0 * (-(3.0_f64).sqrt()).exp();
    assert!(
        (jet[[0, 0, 0]] - euclidean_wrong).abs() > 0.1,
        "gradient unexpectedly matches the Euclidean surrogate {euclidean_wrong} — \
         metric weights are being ignored"
    );
}

/// The anisotropic jet must differ materially from the isotropic jet whenever
/// anisotropy is active, *and* only the anisotropic one matches the forward FD.
/// This is the direct guard against the historical bug where the backward reused
/// the Euclidean radial jet.
#[test]
fn aniso_jet_differs_from_isotropic_and_only_aniso_matches_fd() {
    let points = array![[0.4, 0.7], [0.9, 0.2]];
    let centers = array![[0.0, 0.0], [0.8, 0.5]];
    let ls = 1.0;
    let nu = MaternNu::FiveHalves;
    let aniso = [1.0_f64, -0.6];

    let jet_aniso =
        matern_input_location_jet_nd(points.view(), centers.view(), ls, nu, Some(&aniso))
            .expect("aniso jet");
    let jet_iso =
        matern_input_location_jet_nd(points.view(), centers.view(), ls, nu, None).expect("iso jet");

    // The two must visibly disagree somewhere (weights are not all 1).
    let max_gap = jet_aniso
        .iter()
        .zip(jet_iso.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 1e-2,
        "anisotropic and isotropic input-location jets should differ once η≠0; max gap {max_gap}"
    );

    // The anisotropic jet matches the anisotropic forward FD...
    let h = 1e-6;
    for axis in 0..points.ncols() {
        let fd = fd_first(&points, &centers, ls, nu, Some(&aniso), axis, h);
        for n in 0..points.nrows() {
            for k in 0..centers.nrows() {
                assert!(
                    (jet_aniso[[n, k, axis]] - fd[[n, k]]).abs() < 1e-6 + 1e-5 * fd[[n, k]].abs(),
                    "aniso jet must match aniso forward FD at axis {axis} (n={n},k={k})"
                );
                // ...while the isotropic jet does *not* match the anisotropic FD
                // wherever the weights actually bite.
            }
        }
    }
}

/// Explicit isotropic request on the jet/FFI surface (#1042): an all-zero
/// `aniso_log_scales` is the natural way to ask for the plain isotropic Matérn,
/// so the input-location jet at `η = [0, 0]` must equal the `None` (isotropic)
/// jet *bit-for-bit*, NOT a data-driven anisotropic jet derived from the center
/// cloud. The forward design honors the same literal η (it is the optimizer that
/// seeds the metric, not the design builder), so the jet–vs–forward-FD pair must
/// also stay synchronized. The deliberately anisotropic center cloud (wide x
/// spread) is exactly the geometry that the old all-zero override would have
/// hijacked into an anisotropic metric — proving the override no longer fires.
#[test]
fn explicit_zero_aniso_jet_is_isotropic_and_matches_forward_fd() {
    // Six well-separated rows (spanning the wide center cloud) keep all four
    // centers in the realized full-rank basis: with fewer data rows than centers
    // the forward RRQR reduction (and the mirroring input-location jet,
    // #755/#1937) would drop a collinear center, desyncing the fixed
    // `0..centers.nrows()` comparison from the reduced column geometry.
    let points = array![
        [0.3, 0.65],
        [0.72, 0.18],
        [0.48, 0.93],
        [2.1, 0.0],
        [3.9, -0.2],
        [1.1, 0.5]
    ];
    // Deliberately anisotropic center cloud (wider spread along x): the geometry
    // the discarded override would have turned into a data-driven metric.
    let centers = array![[0.0, 0.0], [2.0, 0.1], [4.0, -0.1], [1.0, 0.6]];
    let ls = 1.3;
    let nu = MaternNu::ThreeHalves;
    let aniso = [0.0_f64, 0.0];

    let jet_zero =
        matern_input_location_jet_nd(points.view(), centers.view(), ls, nu, Some(&aniso))
            .expect("explicit-zero jet");
    let jet_iso = matern_input_location_jet_nd(points.view(), centers.view(), ls, nu, None)
        .expect("isotropic (None) jet");

    // The explicit all-zero jet must equal the isotropic jet to roundoff — the
    // metric weights exp(2·0)=1 reduce the anisotropic radius to the Euclidean
    // one, so no geometry-derived anisotropy may leak in.
    let max_gap = jet_zero
        .iter()
        .zip(jet_iso.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap < 1e-12,
        "explicit aniso=[0,0] jet must equal the isotropic (None) jet, but max|diff| = {max_gap:.3e} \
         (a data-driven anisotropic metric was seeded from the center cloud)"
    );

    // ...and it still matches the forward finite difference of the design.
    let h = 1e-6;
    for axis in 0..points.ncols() {
        let fd = fd_first(&points, &centers, ls, nu, Some(&aniso), axis, h);
        for n in 0..points.nrows() {
            for k in 0..centers.nrows() {
                let tol = 1e-6 + 1e-5 * fd[[n, k]].abs();
                assert!(
                    (jet_zero[[n, k, axis]] - fd[[n, k]]).abs() < tol,
                    "explicit-zero jet must match forward FD at axis {axis} \
                     (n={n},k={k}): {} vs {}",
                    jet_zero[[n, k, axis]],
                    fd[[n, k]]
                );
            }
        }
    }
}

/// Collision limit `r_A → 0`: the gradient is exactly zero (smooth maximum) and
/// the Hessian collapses to the finite isotropic-in-metric diagonal
/// `(φ'/r)|_0 · w_a δ_ac`, which must be finite and symmetric.
#[test]
fn aniso_collision_limit_is_finite_and_zero_gradient() {
    let points = array![[0.5, 0.5]];
    let centers = array![[0.5, 0.5]];
    let aniso = [0.7_f64, -0.7];
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
    ] {
        let jet =
            matern_input_location_jet_nd(points.view(), centers.view(), 1.0, nu, Some(&aniso))
                .expect("collision jet");
        for v in jet.iter() {
            assert!(v.is_finite(), "collision gradient must be finite");
            assert!(
                v.abs() < 1e-14,
                "collision gradient must be exactly zero, got {v}"
            );
        }
        let hess =
            matern_input_location_hessian_nd(points.view(), centers.view(), 1.0, nu, Some(&aniso))
                .expect("collision Hessian");
        for v in hess.iter() {
            assert!(v.is_finite(), "collision Hessian must be finite");
        }
        // Off-diagonal entries vanish at exact collision; diagonal is nonzero.
        assert!(
            hess[[0, 0, 0, 1]].abs() < 1e-14,
            "off-diagonal must vanish at collision"
        );
        assert!(
            hess[[0, 0, 1, 0]].abs() < 1e-14,
            "off-diagonal must vanish at collision"
        );
        assert!(
            hess[[0, 0, 0, 0]].abs() > 0.0,
            "diagonal curvature must be nonzero"
        );
    }
}
