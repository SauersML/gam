//! #1122 localizer: FD-verify the Matérn log-κ DESIGN derivative `∂X/∂ψ` in
//! the EXACT production frame the `matern(x1, x2)` outer-gradient merge gate
//! exercises — `nu = 5/2`, `Auto(FarthestPoint)` centers (so the cold build
//! rank-reduces them via `matern_rank_reduce_centers`), per-axis input
//! standardization, and the frozen identifiability transform + frozen
//! double-penalty nullspace-shrinkage decision pinned by
//! `freeze_geometry_from_metadata`.
//!
//! The trivial-frame FD test
//! (`matern_log_kappa_first_derivative_matches_finite_difference`) uses
//! `nu = 3/2`, `UserProvided` centers (no rank reduction), no standardization,
//! the default `CenterSumToZero` identifiability, and `double_penalty = false`.
//! It passes. The merge gate, however, used to DESYNC on the H-side
//! `∂(XᵀWX)/∂ψ` term — a β-independent quantity for the Gaussian-identity REML
//! criterion, so the gap can ONLY come from the design derivative `X_τ`.
//!
//! This test reproduces the gate's realized geometry by doing exactly what the
//! optimizer does: cold-build once to realize the centers / transform / scales,
//! freeze them into a `UserProvided` + `FrozenTransform` spec (mirroring
//! `freeze_geometry_from_metadata`), then central-difference the value design
//! across log-κ and compare against the analytic
//! `build_matern_basis_log_kappa_derivatives` in that frozen frame. The value
//! rebuilds and the analytic derivative therefore share one fixed geometry by
//! construction — exactly the production invariant — so any gap is a genuine
//! `X_τ` error.

use gam::smooth::input_standardization::{
    apply_input_standardization, compensate_length_scale_for_standardization,
};
use gam::terms::basis::{
    BasisMetadata, CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu,
    build_matern_basis, build_matern_basis_log_kappa_derivatives,
};
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

fn sample_cloud(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Deliberately anisotropic spread so the per-axis σ differ — this exercises
    // the standardization compensation chain that the merge gate's 2-D fit hits.
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let uy = Uniform::new(-2.0, 3.0).expect("uniform");
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = ux.sample(&mut rng);
        data[[i, 1]] = uy.sample(&mut rng);
    }
    data
}

/// Build the analytic design derivative and the central-difference design
/// derivative in a frame frozen exactly like production, and return the max
/// element-wise absolute error.
///
/// `freeze_rho` is the bootstrap κ₀ at which the geometry (centers + transform)
/// is frozen; `rho` is the (possibly different) κ at which the derivative is
/// evaluated. Production freezes ONCE at a bootstrap κ₀ and then the optimizer
/// moves κ away, so the value rebuilds AND the analytic derivative both run
/// against a STALE Z(κ₀). Passing `freeze_rho != rho` exercises exactly that.
fn frozen_frame_design_derivative_max_error(
    data: ArrayView2<'_, f64>,
    nu: MaternNu,
    double_penalty: bool,
    freeze_rho: f64,
    rho: f64,
) -> (f64, usize) {
    let kappa = rho.exp();
    let length_scale = 1.0 / kappa;

    // 1) Standardize the inputs exactly as the production isotropic-κ arm does
    //    (spatial_optimization.rs lines 70-73): apply per-column σ scaling and
    //    compensate the user length-scale by σ_geom.
    let scales: Vec<f64> = (0..data.ncols())
        .map(|j| {
            let col = data.column(j);
            let n = col.len() as f64;
            let mean = col.sum() / n;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt().max(1e-12)
        })
        .collect();
    let mut xs = data.to_owned();
    apply_input_standardization(&mut xs, &scales);
    let ls_eff = compensate_length_scale_for_standardization(length_scale, &scales);
    // The bootstrap κ₀ the geometry is frozen at (may differ from the eval κ).
    let ls_freeze_eff = compensate_length_scale_for_standardization((-freeze_rho).exp(), &scales);

    // 2) Cold build with the production seed: Auto(FarthestPoint) centers so the
    //    rank reduction fires, nu/double_penalty as requested. Cold-build at the
    //    BOOTSTRAP κ₀ so the rank-reduction + transform are frozen there.
    let cold = MaternBasisSpec {
        center_strategy: CenterStrategy::Auto(Box::new(CenterStrategy::FarthestPoint {
            num_centers: 40,
        })),
        periodic: None,
        length_scale: ls_freeze_eff,
        nu,
        include_intercept: false,
        double_penalty,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let cold_build = build_matern_basis(xs.view(), &cold).expect("cold build");
    let (centers, transform, shrinkage) = match &cold_build.metadata {
        BasisMetadata::Matern {
            centers,
            identifiability_transform,
            nullspace_shrinkage_survived,
            ..
        } => (
            centers.clone(),
            identifiability_transform.clone(),
            *nullspace_shrinkage_survived,
        ),
        other => panic!(
            "expected Matérn metadata, got {:?}",
            std::mem::discriminant(other)
        ),
    };

    // 3) Freeze EXACTLY as `freeze_geometry_from_metadata`: pin the reduced
    //    centers as `UserProvided` and the transform + shrinkage decision as a
    //    `FrozenTransform`. This is the spec the optimizer's per-trial value
    //    rebuild AND the analytic ψ-derivative arm both consume.
    let reduced_centers = centers.nrows();
    let mut frozen = cold.clone();
    frozen.center_strategy = CenterStrategy::UserProvided(centers);
    if let Some(z) = transform {
        frozen.identifiability = MaternIdentifiability::FrozenTransform {
            transform: z,
            nullspace_shrinkage_survived: Some(shrinkage),
        };
    }
    // The optimizer has moved κ to the EVAL point; the frozen spec carries the
    // eval-κ length-scale while keeping the κ₀-frozen centers/transform.
    frozen.length_scale = ls_eff;

    // 4) Analytic derivative in the frozen frame (evaluated at the eval κ).
    let analytic = build_matern_basis_log_kappa_derivatives(xs.view(), &frozen)
        .expect("analytic derivative")
        .first
        .design_derivative;

    // 5) Central-difference the value design across ψ = log κ, holding the
    //    standardized inputs and the frozen geometry fixed (only `length_scale`
    //    moves, mirroring the optimizer's `apply_psi`).
    let h = 1e-6;
    let value_at = |r: f64| -> Array2<f64> {
        let ls_r = (-r).exp();
        // The frozen frame compensates by σ_geom too — reproduce it so the FD
        // and the analytic derivative live in identical coordinates.
        let ls_r_eff = compensate_length_scale_for_standardization(ls_r, &scales);
        let mut s = frozen.clone();
        s.length_scale = ls_r_eff;
        build_matern_basis(xs.view(), &s)
            .expect("value rebuild")
            .design
            .to_dense()
    };
    let num = (value_at(rho + h) - value_at(rho - h)) / (2.0 * h);

    if analytic.shape() != num.shape() {
        panic!(
            "analytic shape {:?} != FD shape {:?} (nu double_penalty={double_penalty})",
            analytic.shape(),
            num.shape()
        );
    }
    let err = (&analytic - &num)
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |a: f64, &b| a.max(b));
    (err, reduced_centers)
}

#[test]
fn matern_gate_frame_log_kappa_design_derivative_matches_fd() {
    let data = sample_cloud(150, 1122);
    // The gate freezes the geometry at a BOOTSTRAP κ₀ and then the optimizer
    // moves κ away — so the derivative is evaluated against a STALE Z(κ₀). We
    // sweep `(freeze_rho, eval_rho)` pairs that include the stale case
    // (freeze ≠ eval), which the original single-κ test never exercised.
    let mut failures = Vec::new();
    for &dp in &[false, true] {
        for &nu in &[MaternNu::ThreeHalves, MaternNu::FiveHalves] {
            for &(freeze_rho, rho) in &[
                (0.0_f64, 0.0_f64), // frozen == eval (baseline)
                (0.0, 0.6),         // optimizer moved κ UP from κ₀
                (0.0, -0.6),        // optimizer moved κ DOWN from κ₀
                (-0.4, 0.8),        // larger stale gap
            ] {
                let (err, reduced) =
                    frozen_frame_design_derivative_max_error(data.view(), nu, dp, freeze_rho, rho);
                eprintln!(
                    "[gate-frame X_psi FD] nu={nu:?} double_penalty={dp} freeze_rho={freeze_rho:+.2} eval_rho={rho:+.2} reduced_centers={reduced} max_abs_err={err:.3e}"
                );
                // ∂X/∂ψ is the same regardless of penalty parameterization (the
                // design columns are penalty-independent), so a tight bound holds
                // in every frame. 1e-5 covers the FD truncation at h=1e-6.
                if !(err < 1e-5) {
                    failures.push(format!(
                        "nu={nu:?} double_penalty={dp} freeze_rho={freeze_rho:+.2} eval_rho={rho:+.2}: max_abs_err={err:.3e} ≥ 1e-5"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gate-frame Matérn log-κ design derivative diverges from finite difference:\n  - {}",
        failures.join("\n  - "),
    );
}
