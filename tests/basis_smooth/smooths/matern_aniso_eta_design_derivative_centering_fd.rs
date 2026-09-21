//! #1376 regression guard: the analytic anisotropic Matérn DESIGN per-axis
//! derivative the κ-optimizer consumes must match a central finite difference of
//! the realized design taken in the OPTIMIZER's coordinate frame — i.e. moving
//! the raw per-axis coordinate `psi_a`, which the coordinate map
//! `spatial_term_psi_to_length_scale_and_aniso` decodes into BOTH the global
//! length scale `ℓ = exp(−mean(psi))` AND the centered contrast
//! `eta_a = psi_a − mean(psi)` simultaneously.
//!
//! The original #1376 defect was a coordinate-frame error. An earlier fix
//! installed the cross-axis centering projection `P = I − 11ᵀ/d` on the per-axis
//! ψ derivatives, on the reasoning that a uniform η-shift is a no-op of the
//! centered metric. That is true ONLY at fixed ℓ — but the optimizer never holds
//! ℓ fixed: its coordinate `psi_a` drives ℓ and the contrast together. In the
//! kernel argument these recombine,
//!
//!   x² = r²/ℓ² = Σ_a exp(2·(psi_a − mean(psi)))·exp(2·mean(psi))·h_a²
//!              = Σ_a exp(2·psi_a)·h_a²,
//!
//! so the `mean(psi)` cancels exactly and the effective per-axis exponent is the
//! RAW `psi_a`. The criterion derivative w.r.t. `psi_a` is therefore the NATIVE,
//! un-centered per-axis derivative `∂φ/∂psi_a = q·s_a` (the centering and the
//! omitted `−(1/d)·∂/∂ln ℓ` length-scale term cancel to the identity). The
//! centering left the analytic outer gradient sum-zero / antisymmetric while the
//! FD of the full criterion is not — the rel≈0.85 gap. Removing it restores
//! agreement.
//!
//! This test pins the DESIGN side directly: it FDs the realized design under a
//! single-axis `psi_a` bump (decoded through the coordinate map so ℓ moves with
//! it) and asserts the analytic `design_first[a]` matches — the exact frame the
//! consumer (`spatial_log_kappa_hyper_dirs_frominfo_list`) and the outer FD
//! audit use. A regression that re-installed the fixed-ℓ centering would fail
//! this `psi`-frame FD.
//!
//! Reference-as-truth: every assertion is against a central finite difference of
//! gam's own realized anisotropic design — never another tool's output.

use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternNu,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
};
use ndarray::Array2;

/// Decode the κ-optimizer coordinate `psi` into the (length_scale, aniso_log_scales)
/// pair exactly as `spatial_term_psi_to_length_scale_and_aniso` does:
///   ℓ = exp(−mean(psi)),  eta_a = psi_a − mean(psi).
fn psi_to_length_scale_and_eta(psi: &[f64]) -> (f64, Vec<f64>) {
    let psi_bar = psi.iter().sum::<f64>() / psi.len() as f64;
    let ls = (-psi_bar).exp();
    let eta = psi.iter().map(|&v| v - psi_bar).collect();
    (ls, eta)
}

/// Build the realized dense anisotropic Matérn design at a given OPTIMIZER
/// coordinate `psi`, decoding it into (ℓ, η) the same way the κ-optimizer does,
/// holding everything else (centers, ν, identifiability) fixed.
fn realized_design_at_psi(data: &Array2<f64>, spec: &MaternBasisSpec, psi: &[f64]) -> Array2<f64> {
    let (ls, eta) = psi_to_length_scale_and_eta(psi);
    let mut trial = spec.clone();
    trial.length_scale.set_resolved(ls);
    trial.aniso_log_scales = Some(eta);
    build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
        .expect("realized aniso design build")
        .design
        .to_dense()
}

/// Floating-point evaluation error of one realized design entry, relative to
/// the largest design entry. An entry is a Matérn kernel value (`sqrt`, `exp`
/// and a short polynomial, about 16 roundings) contracted with the
/// identifiability transform over the 8 centers of these fixtures, so it
/// carries at most about (8 + 16)·ε of rounding. The bound allows for
/// cancellation in the contraction by rounding that up to 64·ε.
const EVAL_REL_ERR: f64 = 64.0 * f64::EPSILON;

/// Derived error bound for a central-difference stencil of order `order`
/// (1 = first derivative, 2 = second derivative) taken at step `h`, given the
/// same stencil at step `2h`, and `m` = the largest design entry.
///
/// * Truncation. Both stencils are O(h²), so D(h) − D* ≈ (D(2h) − D(h)) / 3
///   (Richardson). This bound uses that estimate as measured.
/// * Rounding. Each design entry carries `EVAL_REL_ERR·m` of evaluation
///   error. The stencil's weights sum in absolute value to 2/(2h) = 1/h for the
///   first derivative and to 4/h² for the second, so rounding contributes
///   `EVAL_REL_ERR·m/h` or `4·EVAL_REL_ERR·m/h²`.
///
/// The bound comes from the step and the fixture alone. It is not a fixed
/// tolerance: an analytic derivative that misses a term by more than the
/// finite difference can resolve fails.
fn central_difference_error_bound(
    d_h: &Array2<f64>,
    d_2h: &Array2<f64>,
    m: f64,
    h: f64,
    order: i32,
) -> f64 {
    let richardson = max_abs(&(d_2h - d_h)) / 3.0;
    let stencil_abs_weight = if order == 1 { 1.0 / h } else { 4.0 / (h * h) };
    richardson + EVAL_REL_ERR * m * stencil_abs_weight
}

fn max_abs(x: &Array2<f64>) -> f64 {
    x.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
}

#[test]
fn aniso_design_raw_psi_first_derivative_matches_single_axis_fd() {
    // A small 3-D anisotropic configuration: distinct per-axis scales, and
    // centers == data (UserProvided) so the basis geometry is byte-identical
    // across the FD perturbations.
    let data = Array2::from_shape_vec(
        (8, 3),
        vec![
            -1.0, -0.6, 0.2, -0.7, 0.1, -0.4, -0.2, 0.9, 0.5, 0.3, -0.3, 0.8, 0.8, 0.5, -0.1, 1.1,
            -0.1, 0.4, 1.4, 0.8, -0.7, 0.55, -0.45, 0.15,
        ],
    )
    .unwrap();
    // Optimizer coordinate psi0; decode it to the (ℓ, η) the spec is built at.
    let psi0 = vec![0.30, -0.15, 0.20];
    let (ls0, eta0) = psi_to_length_scale_and_eta(&psi0);
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: gam::terms::basis::MaternLengthScale::fixed(ls0),
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(eta0.clone()),
    };

    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    let dim = psi0.len();
    assert_eq!(
        deriv.design_first.len(),
        dim,
        "aniso design derivative builder must return one matrix per axis"
    );

    // Single-axis raw-psi FD: bump ONLY psi_a, decode to (ℓ, η) — which moves the
    // global length scale AND every centered contrast — rebuild the realized
    // design, central difference. This is the exact coordinate frame the outer
    // optimizer / FD audit perturb. The NATIVE (un-centered) analytic derivative
    // must match; the old fixed-ℓ centering would not.
    let h = 1e-6;
    let m = max_abs(&realized_design_at_psi(&data, &spec, &psi0));
    let first_difference = |a: usize, step: f64| {
        let mut psi_p = psi0.clone();
        psi_p[a] += step;
        let mut psi_m = psi0.clone();
        psi_m[a] -= step;
        let dplus = realized_design_at_psi(&data, &spec, &psi_p);
        let dminus = realized_design_at_psi(&data, &spec, &psi_m);
        (&dplus - &dminus).mapv(|v| v / (2.0 * step))
    };
    for a in 0..dim {
        let fd = first_difference(a, h);
        let fd_2h = first_difference(a, 2.0 * h);
        assert_eq!(
            fd.raw_dim(),
            deriv.design_first[a].raw_dim(),
            "realized design / analytic derivative shape mismatch on axis {a}"
        );
        let analytic = &deriv.design_first[a];
        let max_err = max_abs(&(&fd - analytic));
        let bound = central_difference_error_bound(&fd, &fd_2h, m, h, 1);
        eprintln!(
            "[aniso-fd-1] axis {a}: max_err={max_err:.3e} bound={bound:.3e} \
             max|dX|={:.3e} max|X|={m:.3e}",
            max_abs(analytic)
        );
        assert!(
            max_err <= bound,
            "aniso design raw-psi derivative mismatch on axis {a}: max_err={max_err:.3e} \
             exceeds the central-difference error bound {bound:.3e} — the centered #1376 \
             bug would fail this single-axis psi FD"
        );
    }
}

/// Companion guard: the analytic aniso DESIGN second-diagonal `∂²X/∂psi_a²` must
/// match a central second difference of the realized design under a single-axis
/// `psi_a` bump (decoded to (ℓ, η)). This pins that the SECOND-order design
/// channel is in the same native optimizer-coordinate gauge as the first.
#[test]
fn aniso_design_raw_psi_second_diagonal_matches_single_axis_fd() {
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
        length_scale: gam::terms::basis::MaternLengthScale::fixed(ls0),
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(eta0.clone()),
    };

    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    let dim = psi0.len();
    // The 8-row fixture is far below the operator-only threshold, so the build
    // must take the materialized path and return one dense second-diagonal block
    // per axis (the first-derivative test above requires the same of
    // `design_first`). An empty result is a regression to fail on, not a case
    // to skip.
    assert_eq!(
        deriv.design_second_diag.len(),
        dim,
        "aniso design derivative builder must return one second-diagonal matrix per axis \
         on this 8-row materialized fixture"
    );

    let s0 = realized_design_at_psi(&data, &spec, &psi0);
    let m = max_abs(&s0);
    let h = 1e-4;
    let second_difference = |a: usize, step: f64| {
        let mut psi_p = psi0.clone();
        psi_p[a] += step;
        let mut psi_m = psi0.clone();
        psi_m[a] -= step;
        let sp = realized_design_at_psi(&data, &spec, &psi_p);
        let sm = realized_design_at_psi(&data, &spec, &psi_m);
        (&sp - &(&s0 * 2.0) + &sm).mapv(|v| v / (step * step))
    };
    for a in 0..dim {
        let fd = second_difference(a, h);
        let fd_2h = second_difference(a, 2.0 * h);
        let analytic = &deriv.design_second_diag[a];
        assert_eq!(fd.raw_dim(), analytic.raw_dim(), "shape mismatch axis {a}");
        let max_err = max_abs(&(&fd - analytic));
        let bound = central_difference_error_bound(&fd, &fd_2h, m, h, 2);
        eprintln!(
            "[aniso-fd-2] axis {a}: max_err={max_err:.3e} bound={bound:.3e} \
             max|d2X|={:.3e} max|X|={m:.3e}",
            max_abs(analytic)
        );
        assert!(
            max_err <= bound,
            "aniso design raw-psi second-diagonal mismatch on axis {a}: max_err={max_err:.3e} \
             exceeds the central-difference error bound {bound:.3e}"
        );
    }
}
