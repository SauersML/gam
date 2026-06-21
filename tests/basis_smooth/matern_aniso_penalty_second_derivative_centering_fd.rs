//! #1376 root-cause guard (second order): the analytic anisotropic Matérn
//! PENALTY second derivative w.r.t. the RAW per-axis log-scale `η_a` must match a
//! central finite difference of the realized penalty under a SINGLE-axis bump.
//!
//! The forward aniso metric is the centered contrast `ψ = C·η`,
//! `C = I − (1/d)·11ᵀ`, so the raw-η penalty Hessian is the two-sided projection
//! `H_η = Cᵀ H_ψ C` — `∂²S/∂η_a²` mixes ALL ψ-axis pairs, not just the diagonal.
//! The per-axis builders produce the ψ-Hessian (diagonal in `penalties_second_diag`,
//! off-diagonal via the cross provider); the dense path used to leave these in the
//! ψ gauge while only the FIRST derivatives were centered, so a single-axis raw-η
//! FD of the realized penalty disagreed with the analytic diagonal second. This
//! test pins the fix: `penalties_second_diag[a]` (now centered) equals the central
//! FD of the realized penalty under `η_a ± h`.
//!
//! Reference-as-truth: every assertion is against a central FD of gam's own
//! realized anisotropic penalty — never another tool's output.

use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternNu,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
};
use ndarray::Array2;

/// Realized penalty blocks at a given raw-η vector (centers/ℓ/ν/ident fixed).
fn realized_penalties_at(
    data: &Array2<f64>,
    spec: &MaternBasisSpec,
    eta: &[f64],
) -> Vec<Array2<f64>> {
    let mut trial = spec.clone();
    trial.aniso_log_scales = Some(eta.to_vec());
    build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
        .expect("realized aniso penalty build")
        .penalties
}

#[test]
fn aniso_penalty_raw_eta_second_derivative_matches_single_axis_fd() {
    let data = Array2::from_shape_vec(
        (8, 3),
        vec![
            -1.0, -0.6, 0.2, -0.7, 0.1, -0.4, -0.2, 0.9, 0.5, 0.3, -0.3, 0.8, 0.8, 0.5, -0.1, 1.1,
            -0.1, 0.4, 1.4, 0.8, -0.7, 0.55, -0.45, 0.15,
        ],
    )
    .unwrap();
    let eta0 = vec![0.35, -0.20, 0.10];
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: 0.9,
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: true,
        identifiability: Default::default(),
        aniso_log_scales: Some(eta0.clone()),
        nullspace_shrinkage_survived: None,
    };

    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    let dim = eta0.len();
    assert_eq!(deriv.penalties_second_diag.len(), dim);
    let num_blocks = deriv.penalties_second_diag[0].len();
    assert!(num_blocks >= 1, "expected at least the primary penalty block");

    let realized_at = |eta: &[f64]| realized_penalties_at(&data, &spec, eta);
    let s0 = realized_at(&eta0);
    assert_eq!(s0.len(), num_blocks, "block count must match realized penalty");

    let h = 1e-4;
    for a in 0..dim {
        let mut eta_p = eta0.clone();
        eta_p[a] += h;
        let mut eta_m = eta0.clone();
        eta_m[a] -= h;
        let sp = realized_at(&eta_p);
        let sm = realized_at(&eta_m);
        for blk in 0..num_blocks {
            // Central second difference of the realized penalty block.
            let fd = (&sp[blk] - &(&s0[blk] * 2.0) + &sm[blk]).mapv(|v| v / (h * h));
            let analytic = &deriv.penalties_second_diag[a][blk];
            assert_eq!(
                fd.raw_dim(),
                analytic.raw_dim(),
                "shape mismatch axis {a} block {blk}"
            );
            let scale = analytic.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            let max_err = (&fd - analytic).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            // Second-difference FD has O(h²) truncation + O(ε/h²) round-off; with
            // h=1e-4 the achievable tolerance is ~1e-3·scale.
            assert!(
                max_err < 2e-3 * scale.max(1.0),
                "aniso penalty raw-η second-derivative mismatch on axis {a} block {blk}: \
                 max_err={max_err:.3e} (scale={scale:.3e}) — the un-centered seconds \
                 (#1376) would fail this single-axis FD"
            );
        }
    }
}
