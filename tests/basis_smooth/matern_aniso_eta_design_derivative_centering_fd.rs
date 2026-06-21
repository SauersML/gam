//! #1376 regression guard: the analytic anisotropic Matérn DESIGN first
//! derivative w.r.t. the RAW per-axis log-scale `η_a` must match a central
//! finite difference of the realized design — for a single-axis perturbation,
//! NOT just a contrast.
//!
//! The original #1376 defect (analytic outer-gradient ≠ FD, rel ≈ 0.85) was the
//! anisotropic design first-derivative being left UN-centered: the forward
//! anisotropic metric is the CENTERED contrast `w_a = exp(2·(η_a − mean(η)))`, so
//! a uniform shift of every η_a is a no-op and the raw-η Jacobian must be the
//! centering projection of the per-axis centered-ψ derivative,
//!
//!   ∂(design)/∂η_a = ∂(design)/∂ψ_a − (1/d) Σ_b ∂(design)/∂ψ_b.
//!
//! Centering was applied to the penalty derivatives but the DESIGN first
//! derivatives (which feed the deviance and the H-side of the outer REML
//! gradient) were un-centered, so the full outer gradient disagreed with a
//! central FD of the criterion by the un-centered common mode (fixed on main in
//! e0a7384e0 / 1f22db7c1).
//!
//! A penalty-contrast FD test already guards the contrast direction
//! (`anisotropic_penalty_contrast_derivative_matches_finite_difference`). This
//! test guards the DESIGN side AND, crucially, a SINGLE-axis raw-η perturbation,
//! which is exactly the common-mode the un-centered analytic got wrong: a one-axis
//! bump moves `mean(η)` too, so a correct analytic must subtract the cross-axis
//! mean.
//!
//! Reference-as-truth: every assertion is against a central finite difference of
//! gam's own realized anisotropic design — never another tool's output.

use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternNu,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
};
use ndarray::Array2;

/// Build the realized dense anisotropic Matérn design at a given raw-η vector,
/// holding everything else (centers, ℓ, ν, identifiability) fixed.
fn realized_design_at(data: &Array2<f64>, spec: &MaternBasisSpec, eta: &[f64]) -> Array2<f64> {
    let mut trial = spec.clone();
    trial.aniso_log_scales = Some(eta.to_vec());
    build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
        .expect("realized aniso design build")
        .design
        .to_dense()
}

#[test]
fn aniso_design_raw_eta_first_derivative_matches_single_axis_fd() {
    // A small 3-D anisotropic configuration: distinct per-axis scales so the
    // centering (mean over axes) is non-trivial, and centers == data
    // (UserProvided) so the basis geometry is byte-identical across the FD
    // perturbations.
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
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(eta0.clone()),
        nullspace_shrinkage_survived: None,
    };

    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    let dim = eta0.len();
    assert_eq!(
        deriv.design_first.len(),
        dim,
        "aniso design derivative builder must return one matrix per axis"
    );

    // Sum of the centered per-axis design derivatives must be ZERO: centering
    // projects out the all-ones common mode, so Σ_a ∂design/∂η_a = 0 (a uniform
    // η shift leaves the centered metric — hence the design — unchanged).
    let mut common_mode = deriv.design_first[0].clone();
    for a in 1..dim {
        common_mode = &common_mode + &deriv.design_first[a];
    }
    let common_mode_max = common_mode.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(
        common_mode_max < 1e-9,
        "centered aniso design derivatives must sum to zero (all-ones common mode \
         projected out); got max |Σ_a ∂design/∂η_a| = {common_mode_max:.3e}"
    );

    // Single-axis raw-η FD: bump ONLY η_a, rebuild the realized design, central
    // difference. This is the exact common-mode case the un-centered analytic
    // got wrong (one-axis bump also moves mean(η)). The columns of the realized
    // design are identifiability-fixed (UserProvided centers + default
    // residualization on the same data), so the per-entry difference is a clean
    // FD of the same parameterization.
    let h = 1e-6;
    for a in 0..dim {
        let mut eta_p = eta0.clone();
        eta_p[a] += h;
        let mut eta_m = eta0.clone();
        eta_m[a] -= h;
        let dplus = realized_design_at(&data, &spec, &eta_p);
        let dminus = realized_design_at(&data, &spec, &eta_m);
        assert_eq!(
            dplus.raw_dim(),
            deriv.design_first[a].raw_dim(),
            "realized design / analytic derivative shape mismatch on axis {a}"
        );
        let fd = (&dplus - &dminus).mapv(|v| v / (2.0 * h));
        let analytic = &deriv.design_first[a];
        let scale = analytic.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
        let max_err = (&fd - analytic)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            max_err < 1e-5 * scale,
            "aniso design raw-η derivative mismatch on axis {a}: max_err={max_err:.3e} \
             (scale={scale:.3e}) — the un-centered #1376 bug would fail this single-axis FD"
        );
    }
}
