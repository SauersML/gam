//! REGRESSION (#1375): the 2-D Duchon spatial smooth `duchon(x, z)` (mgcv
//! `s(..., bs="ds")`) must be INVARIANT to a pure covariate translation
//! `(x, z) → (x + b, z + b)`.
//!
//! The radial kernel block reads only coordinate DIFFERENCES `data − centers`,
//! so it is already translation-invariant. The polynomial NULL-SPACE block
//! `P = {1, x, z}` (appended as explicit unpenalized design columns) and the
//! side-condition `P(centers)ᵀα = 0` that defines the kernel reparameterization
//! `Z`, however, were assembled at ABSOLUTE coordinates. Under a large
//! translation `b`, the `{1, x}` columns become near-collinear, the design
//! ill-conditions, and REML λ-selection drifts into a different basin — moving
//! the fit even though `{1, x − x̄}` spans the SAME model space. This is the
//! exact defect #1269 fixed for the `bs="tp"` thin-plate path; the fix here
//! mirrors it by centering the polynomial / side-condition assembly on the
//! center-cloud per-axis mean (a fixed, frozen property replayed at predict).
//!
//! PENALTY-SENSITIVE PROBE. A pure column-SPAN projection is invariant to the
//! parameterization and would NOT see the bug — the defect lives in the PENALTY
//! geometry, the same `Zᵀ K_CC Z` / radial-reparam block REML drives. We probe
//! the realized PENALIZED smoother at a FIXED smoothing weight:
//!
//!   f̂(λ) = X (XᵀX + λ S)⁻¹ Xᵀ g ,   S = Σ_b S_b   (all emitted penalty blocks).
//!
//! `f̂(λ)` is exactly the map REML evaluates at each λ. If the basis is
//! translation-invariant the whole `(X, S)` pair lives in the same model space
//! (up to a reparameterization the smoother is invariant to), so `f̂(λ)` is
//! identical for `(x, z)` and `(x + b, z + b)`. Under the #1375 bug the uncentered
//! side-condition `Z` and the radial reparameterization derived from it drift,
//! the penalty `S` lands in a different geometry, and `f̂(λ)` moves — which is the
//! REML λ-selection drift the issue reports. The centers (deterministic
//! farthest-point) shift by the same `b`, so a correct basis is invariant.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basis,
};
use gam::faer_ndarray::FaerCholesky;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

/// Deterministic 2-D point cloud in `[-1, 1]²`.
fn synthetic_xz(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        for j in 0..2 {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

/// The default non-periodic 2-D Duchon spec: magic-cubic-like `(p=2, s=0.5)`,
/// linear null space `{1, x, z}` — exactly the `duchon(x, z)` of the issue.
fn duchon_xz_spec() -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// Dense Duchon design plus the SUM of all emitted penalty blocks (the design's
/// own coefficient space), materialized for `data` under the shared spec.
fn design_and_penalty(data: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let spec = duchon_xz_spec();
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis must succeed");
    let design: Array2<f64> = result
        .design
        .try_to_dense_arc("duchon translation-invariance test")
        .expect("design materializes")
        .as_ref()
        .clone();
    let p = design.ncols();
    let mut penalty = Array2::<f64>::zeros((p, p));
    for block in &result.penalties {
        assert_eq!(
            block.dim(),
            (p, p),
            "penalty block must act on the design's coefficient space"
        );
        penalty += block;
    }
    (design, penalty)
}

/// Penalized fitted values `f̂(λ) = X (XᵀX + λ S)⁻¹ Xᵀ g` — exactly the map REML
/// evaluates at smoothing weight `λ`. A tiny ridge `ε‖·‖` guards the unpenalized
/// (polynomial) null space so the system is SPD for the Cholesky solve.
fn penalized_fit(x: &Array2<f64>, s: &Array2<f64>, g: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let p = x.ncols();
    let xtx = x.t().dot(x);
    let max_diag = xtx.diag().iter().cloned().fold(1.0_f64, f64::max);
    let mut a = &xtx + &(s * lambda);
    let eps = 1e-10 * max_diag;
    for i in 0..p {
        a[[i, i]] += eps;
    }
    let xtg = x.t().dot(g);
    let chol = a
        .cholesky(faer::Side::Lower)
        .expect("Cholesky of XᵀX + λS + εI");
    let beta = chol.solvevec(&xtg);
    x.dot(&beta)
}

#[test]
fn duchon_xz_fit_invariant_to_diagonal_translation_1375() {
    assert!(file!().ends_with(".rs"));

    let n = 220usize;
    let base = synthetic_xz(n, 0x1375_2026);

    // A genuinely 2-D nonlinear target with a linear trend (exercises both the
    // polynomial null space and the radial bending block), sampled at the rows.
    let mut g = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x, z) = (base[[i, 0]], base[[i, 1]]);
        g[i] = 0.9 * x - 0.6 * z
            + 1.3 * (-(x * x + z * z) / (2.0 * 0.4 * 0.4)).exp()
            + 0.4 * (std::f64::consts::PI * x).sin() * (0.5 * std::f64::consts::PI * z).cos();
    }
    let signal_range = {
        let lo = g.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (hi - lo).max(1.0)
    };

    // Reference basis + penalty at the raw coordinates.
    let (design_base, penalty_base) = design_and_penalty(&base);

    // A LARGE diagonal translation: every covariate offset by the same `b`. The
    // farthest-point centers shift by `b` too, so the model space is identical.
    let b = 5000.0_f64;
    let mut shifted = base.clone();
    shifted.mapv_inplace(|v| v + b);

    let (design_shift, penalty_shift) = design_and_penalty(&shifted);
    assert_eq!(
        design_base.ncols(),
        design_shift.ncols(),
        "translation must not change the Duchon basis dimension"
    );

    // Compare the penalized smoother across a sweep of smoothing weights — REML
    // explores exactly this λ profile, so invariance here is invariance of the
    // λ-selection landscape. A single λ where the geometries disagree is enough
    // to drift the selected smooth.
    let mut worst_rel = 0.0_f64;
    for &lambda in &[1e-3_f64, 1e-1, 1.0, 1e1, 1e3] {
        let fit_base = penalized_fit(&design_base, &penalty_base, &g, lambda);
        let fit_shift = penalized_fit(&design_shift, &penalty_shift, &g, lambda);
        let max_abs_diff = fit_base
            .iter()
            .zip(fit_shift.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        worst_rel = worst_rel.max(max_abs_diff / signal_range);
    }

    assert!(
        worst_rel < 1e-6,
        "Duchon duchon(x, z) penalized fit drifted under a pure diagonal translation by \
         b={b}: worst max |Δf̂(λ)| = {worst_rel:.3e} of signal range {signal_range:.3e} \
         across the λ sweep; the polynomial null-space / side-condition block is not \
         assembled in a translation-invariant (centered) frame (#1375)"
    );
}
