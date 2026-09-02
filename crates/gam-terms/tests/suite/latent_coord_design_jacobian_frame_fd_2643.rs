//! FD gate for #2643: the latent-coordinate design Jacobian must be the
//! derivative of the design the fit actually builds.
//!
//! `LatentCoordDesignDerivative::{new_matern,new_duchon}` are handed the
//! metadata's `centers` (STANDARDIZED, `x / input_scale`) together with its
//! `length_scale` (ORIGINAL units), and evaluate both against the RAW latent
//! coordinates. Three frames meet in one kernel evaluation, and the result is
//! the analytic `∂X/∂t` that `LatentCoordDerivativeOp` uses to steer the joint
//! `[rho, latent]` REML directions.
//!
//! The ground truth here is not a hand-written formula — it is
//! `build_term_collection_design` itself, which is literally what the
//! latent-coordinate driver re-runs on every θ (`spatial_optimization.rs`
//! `ensure_theta` writes the raw latent values into `data` and rebuilds through
//! the frozen spec). Central-differencing that rebuild is therefore the
//! definition of the quantity the operator claims to supply.
//!
//! Why this test did not exist: nothing pinned `local_design_jacobian_row` at
//! all, and — this is the trap — the defect is INVISIBLE at `input_scale == 1`,
//! where the standardized and original frames coincide. Any fixture built on
//! unit-spread latents passes while the shipped code is wrong. The
//! `input_scale == 1` arm below is kept precisely so a future reader can see
//! that the sensitivity is to σ and to nothing else.

use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu};
use gam_terms::smooth::input_standardization::estimate_isotropic_scale;
use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, build_term_collection_design};
use ndarray::{Array1, Array2, s};

/// The user-facing kernel range, in ORIGINAL covariate units.
const USER_LENGTH_SCALE: f64 = 1.3;

/// Deterministic latent cloud whose realized `input_scale` is exactly
/// `target_sigma`.
///
/// The base point set is rescaled by `target_sigma / σ(base)` with `σ(base)`
/// MEASURED rather than written down, so the arms below name the frame ratio
/// they exercise instead of carrying a fitted normalizing literal.
fn latent_data(target_sigma: f64) -> Array2<f64> {
    let base: [[f64; 2]; 10] = [
        [-1.20, -0.40],
        [-0.65, 0.85],
        [-0.10, -0.95],
        [0.35, 0.30],
        [0.90, 1.15],
        [1.45, -0.20],
        [0.15, 1.50],
        [1.25, 0.55],
        [-0.85, 0.10],
        [0.60, -1.30],
    ];
    let mut data = Array2::<f64>::zeros((base.len(), 2));
    for (row, values) in base.iter().enumerate() {
        data[[row, 0]] = values[0];
        data[[row, 1]] = values[1];
    }
    let base_sigma = estimate_isotropic_scale(data.view())
        .expect("base isotropic scale")
        .get();
    data.mapv_inplace(|value| value * target_sigma / base_sigma);
    data
}

fn fresh_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "latent_matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::EqualMass { num_centers: 5 },
                    periodic: None,
                    length_scale: MaternLengthScale::fixed(USER_LENGTH_SCALE),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// The design row the fit actually uses at latent configuration `data`.
///
/// `spec` must already be FROZEN (carrying `input_scale: Some(σ)` and
/// `CenterStrategy::UserProvided`), so σ and the centers are held fixed and the
/// only thing the difference quotient moves is the latent coordinate.
fn design_row(data: &Array2<f64>, spec: &TermCollectionSpec, row: usize) -> Array1<f64> {
    let built = build_term_collection_design(data.view(), spec).expect("design rebuild");
    // The operator's Jacobian spans this TERM's columns, not the collection's:
    // the assembled design also carries the parametric block. Offset the term's
    // local range by the smooth block's start exactly as the latent-coordinate
    // driver does when it hands columns to `LatentCoordDerivativeOp`.
    let p_total = built.design.ncols();
    let smooth_start = p_total.saturating_sub(built.smooth.total_smooth_cols());
    let range = &built.smooth.terms[0].coeff_range;
    let dense = built.design.to_dense();
    dense
        .row(row)
        .slice(s![smooth_start + range.start..smooth_start + range.end])
        .to_owned()
}

/// `∂ X[row, :] / ∂ t[row, axis]` by central difference of the production
/// rebuild.
fn finite_difference_row(
    data: &Array2<f64>,
    spec: &TermCollectionSpec,
    row: usize,
    axis: usize,
    step: f64,
) -> Array1<f64> {
    let mut plus = data.clone();
    plus[[row, axis]] += step;
    let mut minus = data.clone();
    minus[[row, axis]] -= step;
    let forward = design_row(&plus, spec, row);
    let backward = design_row(&minus, spec, row);
    (forward - backward) / (2.0 * step)
}

