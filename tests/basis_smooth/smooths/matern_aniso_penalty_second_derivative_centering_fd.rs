//! #1376 root-cause guard (second order): the analytic anisotropic Matérn
//! PENALTY second derivative the κ-optimizer consumes must match a central finite
//! difference of the realized penalty taken in the OPTIMIZER's coordinate frame —
//! a single-axis bump of the raw coordinate `psi_a`, decoded by
//! `spatial_term_psi_to_length_scale_and_aniso` into BOTH the global length scale
//! `ℓ = exp(−mean(psi))` and the centered contrast `eta_a = psi_a − mean(psi)`.
//!
//! In the kernel argument `x² = r²/ℓ² = Σ_a exp(2·psi_a)·h_a²` the `mean(psi)`
//! cancels, so the effective per-axis exponent is the raw `psi_a` and the
//! criterion's per-axis derivative is the NATIVE per-axis ψ derivative — NOT the
//! fixed-ℓ centering projection `Cᵀ(·)C`. An earlier #1376 fix centered the
//! penalty derivatives (a fixed-ℓ frame), which disagrees with the optimizer's
//! `psi` frame by exactly the omitted `∂/∂ln ℓ` length-scale term (the two
//! cancel to the identity). This test pins the corrected native gauge:
//! `penalties_second_diag[a]` equals the central second difference of the
//! realized penalty under a single-axis `psi_a ± h`, decoded to (ℓ, η).
//!
//! Reference-as-truth: every assertion is against a central FD of gam's own
//! realized anisotropic penalty — never another tool's output.

use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternNu,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
};
use ndarray::Array2;

/// Decode the κ-optimizer coordinate `psi` into (length_scale, aniso_log_scales)
/// exactly as `spatial_term_psi_to_length_scale_and_aniso` does.
fn psi_to_length_scale_and_eta(psi: &[f64]) -> (f64, Vec<f64>) {
    let psi_bar = psi.iter().sum::<f64>() / psi.len() as f64;
    ((-psi_bar).exp(), psi.iter().map(|&v| v - psi_bar).collect())
}

/// Realized penalty blocks at a given OPTIMIZER coordinate `psi` (centers/ν/ident
/// fixed; ℓ and η decoded together from `psi`).
fn realized_penalties_at_psi(
    data: &Array2<f64>,
    spec: &MaternBasisSpec,
    psi: &[f64],
) -> Vec<Array2<f64>> {
    let (ls, eta) = psi_to_length_scale_and_eta(psi);
    let mut trial = spec.clone();
    trial.length_scale.set_resolved(ls);
    trial.aniso_log_scales = Some(eta);
    build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
        .expect("realized aniso penalty build")
        .penalties
}

#[test]
fn aniso_penalty_raw_psi_second_derivative_matches_single_axis_fd() {
    let data = Array2::from_shape_vec(
        (8, 3),
        vec![
            -1.0, -0.6, 0.2, -0.7, 0.1, -0.4, -0.2, 0.9, 0.5, 0.3, -0.3, 0.8, 0.8, 0.5, -0.1, 1.1,
            -0.1, 0.4, 1.4, 0.8, -0.7, 0.55, -0.45, 0.15,
        ],
    )
    .unwrap();
    let psi0 = vec![0.30, -0.15, 0.20];
    let (ls0, eta0) = psi_to_length_scale_and_eta(&psi0);
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: ls0.into(),
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: true,
        identifiability: Default::default(),
        aniso_log_scales: Some(eta0.clone()),
    };

    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    let dim = psi0.len();
    assert_eq!(deriv.penalties_second_diag.len(), dim);
    let num_blocks = deriv.penalties_second_diag[0].len();
    assert!(
        num_blocks >= 1,
        "expected at least the primary penalty block"
    );

    let realized_at = |psi: &[f64]| realized_penalties_at_psi(&data, &spec, psi);
    let s0 = realized_at(&psi0);
    assert_eq!(
        s0.len(),
        num_blocks,
        "block count must match realized penalty"
    );

    let h = 1e-4;
    for a in 0..dim {
        let mut psi_p = psi0.clone();
        psi_p[a] += h;
        let mut psi_m = psi0.clone();
        psi_m[a] -= h;
        let sp = realized_at(&psi_p);
        let sm = realized_at(&psi_m);
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
            let max_err = (&fd - analytic)
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()));
            // Second-difference FD has O(h²) truncation + O(ε/h²) round-off; with
            // h=1e-4 the achievable tolerance is ~1e-3·scale.
            assert!(
                max_err < 2e-3 * scale.max(1.0),
                "aniso penalty raw-psi second-derivative mismatch on axis {a} block {blk}: \
                 max_err={max_err:.3e} (scale={scale:.3e}) — the centered #1376 fixed-ℓ \
                 seconds would fail this single-axis psi FD"
            );
        }
    }
}
