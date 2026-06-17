use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};
use crate::families::parameter_block::ParameterBlockInput;
use crate::matrix::{DenseDesignMatrix, DesignMatrix};
use crate::pirls::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Clone, Debug)]
pub struct WiggleBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

#[derive(Clone)]
pub(crate) struct SelectedWiggleBasis {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub block: ParameterBlockInput,
}

pub(crate) fn initializewiggle_knots_from_seed(
    seed: ArrayView1<'_, f64>,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    const MIN_WIGGLE_SEED_SPAN: f64 = 1e-8;
    const DEFAULT_WIGGLE_HALF_RANGE: f64 = 3.0;

    let mut seed_min = seed.iter().copied().fold(f64::INFINITY, f64::min);
    let mut seed_max = seed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !seed_min.is_finite() || !seed_max.is_finite() {
        return Err("non-finite seed for wiggle knot initialization".to_string());
    }
    if (seed_max - seed_min).abs() < MIN_WIGGLE_SEED_SPAN {
        let center = 0.5 * (seed_min + seed_max);
        seed_min = center - DEFAULT_WIGGLE_HALF_RANGE;
        seed_max = center + DEFAULT_WIGGLE_HALF_RANGE;
    }
    let (_, knots) = create_basis::<Dense>(
        seed,
        KnotSource::Generate {
            data_range: (seed_min, seed_max),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    Ok(knots)
}

#[inline]
pub(crate) fn monotone_wiggle_internal_degree(degree: usize) -> Result<usize, String> {
    // Public monotone-wiggle degree refers to the value basis. The low-level
    // I-spline builder integrates a degree-`internal_degree` specification
    // into a degree-`internal_degree + 1` value basis, so we subtract one here
    // to keep the public degree and the per-span value degree aligned.
    degree
        .checked_sub(1)
        .filter(|&internal_degree| internal_degree >= 1)
        .ok_or_else(|| "monotone wiggle degree must be >= 2".to_string())
}

pub fn buildwiggle_block_input_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    let design = monotone_wiggle_basis_from_knots(seed, knots, degree)?;
    let p = design.ncols();
    if p == 0 {
        return Err("wiggle basis has no free monotone columns".to_string());
    }
    let mut penalties: Vec<crate::model_types::PenaltySpec> = Vec::new();
    let mut nullspace_dims = Vec::new();
    if p == 1 {
        penalties.push(crate::model_types::PenaltySpec::Dense(Array2::<f64>::eye(
            1,
        )));
        nullspace_dims.push(0);
    } else {
        let effective_order = penalty_order.max(1).min(p - 1);
        let diff_penalty = create_difference_penalty_matrix(p, effective_order, None)
            .map_err(|e| e.to_string())?;
        penalties.push(crate::model_types::PenaltySpec::Dense(diff_penalty));
        nullspace_dims.push(effective_order);
    }
    if double_penalty {
        penalties.push(crate::model_types::PenaltySpec::Dense(Array2::<f64>::eye(
            p,
        )));
        nullspace_dims.push(0);
    }
    Ok(ParameterBlockInput {
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::zeros(seed.len()),
        penalties,
        nullspace_dims,
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    })
}

pub fn buildwiggle_block_input_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
) -> Result<(ParameterBlockInput, Array1<f64>), String> {
    let knots = initializewiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let block = buildwiggle_block_input_from_knots(
        seed,
        &knots,
        cfg.degree,
        cfg.penalty_order,
        cfg.double_penalty,
    )?;
    Ok((block, knots))
}

pub(crate) fn monotone_wiggle_basis_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    let (basis, _) = create_basis::<Dense>(
        seed,
        KnotSource::Provided(knots.view()),
        internal_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    Ok(basis.as_ref().clone())
}

pub fn monotone_wiggle_basis_with_derivative_order(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    if derivative_order == 0 {
        return monotone_wiggle_basis_from_knots(seed, knots, degree);
    }
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    create_ispline_derivative_dense(seed, knots, internal_degree, derivative_order)
        .map_err(|e| e.to_string())
}

pub(crate) fn monotone_wiggle_nonnegative_constraints(
    beta_dim: usize,
) -> Option<LinearInequalityConstraints> {
    if beta_dim == 0 {
        return None;
    }
    let mut a = Array2::<f64>::zeros((beta_dim, beta_dim));
    for i in 0..beta_dim {
        a[[i, i]] = 1.0;
    }
    Some(LinearInequalityConstraints {
        a,
        b: Array1::zeros(beta_dim),
    })
}

pub(crate) fn validate_monotone_wiggle_beta_nonnegative<'a>(
    beta: impl IntoIterator<Item = &'a f64>,
    context: &str,
) -> Result<(), String> {
    for (idx, &value) in beta.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{context} coefficient {idx} is non-finite"));
        }
        if value < -1e-12 {
            return Err(format!(
                "{context} coefficient {idx} is negative ({value:.3e}); monotone wiggle coefficients must be non-negative"
            ));
        }
    }
    Ok(())
}

/// Slack tolerance for the `beta >= 0` monotone-wiggle inequality constraints.
///
/// The constrained inner Newton/QP holds a binding coordinate at the boundary
/// only up to its own KKT tolerance, so an accepted step can leave the active
/// coordinate a few ULPs below zero (e.g. `-2e-9`). That is feasibility within
/// the solver tolerance, not a genuine sign violation, so the post-update hook
/// projects such coordinates back onto the non-negative cone (clamps them to
/// exactly `0`) rather than failing the fit. The band matches the constrained
/// blockwise solver's KKT tolerances (`1e-6 * scale + 1e-10`,
/// `1e-10 * (1 + scale)`); anything more negative survives the projection and
/// is rejected by [`validate_monotone_wiggle_beta_nonnegative`].
pub(crate) const MONOTONE_WIGGLE_ACTIVE_SET_TOL: f64 = 1e-6;

/// Project a monotone-wiggle coefficient vector onto the non-negative cone the
/// `beta >= 0` constraints define, clamping coordinates the constrained solve
/// left slightly negative (within [`MONOTONE_WIGGLE_ACTIVE_SET_TOL`]) to exactly
/// `0`. Coordinates more negative than the tolerance are left untouched so the
/// subsequent [`validate_monotone_wiggle_beta_nonnegative`] still rejects
/// genuine sign violations.
pub(crate) fn project_monotone_wiggle_beta_nonnegative(mut beta: Array1<f64>) -> Array1<f64> {
    for value in beta.iter_mut() {
        if *value < 0.0 && *value >= -MONOTONE_WIGGLE_ACTIVE_SET_TOL {
            *value = 0.0;
        }
    }
    beta
}

/// Resolve a requested wiggle penalty-order set into:
///
/// - the primary order used by the monotone I-spline coefficient penalty, and
/// - the remaining plain difference-penalty orders to append on the same basis.
///
/// The primary order is the smallest positive requested order. If no positive
/// order is requested, `fallback_primary` is used instead. Extra orders are
/// returned in the original order, deduplicated, and exclude the primary order.
pub fn split_wiggle_penalty_orders(
    fallback_primary: usize,
    penalty_orders: &[usize],
) -> (usize, Vec<usize>) {
    let primary_order = penalty_orders
        .iter()
        .copied()
        .filter(|&order| order >= 1)
        .min()
        .unwrap_or_else(|| fallback_primary.max(1));
    let mut extras = Vec::new();
    for &order in penalty_orders {
        if order == 0 || order == primary_order || extras.contains(&order) {
            continue;
        }
        extras.push(order);
    }
    (primary_order, extras)
}

/// Append raw difference penalties for the given orders to an existing block.
///
/// These are plain difference penalties `D_k^T D_k` on the monotone I-spline
/// coefficients, whose nullspace is the set of polynomial sequences of degree
/// ≤ k−1, giving `nullspace_dim = k`.
pub fn append_selected_wiggle_penalty_orders(
    block: &mut ParameterBlockInput,
    penalty_orders: &[usize],
) -> Result<(), String> {
    let p = block.design.ncols();
    if p == 0 {
        return Ok(());
    }
    for &order in penalty_orders {
        if order == 0 {
            continue;
        }
        if order >= p {
            // A k-th order difference operator applied to a length-p coefficient
            // vector produces a (p-k)-row matrix. When p <= k, that operator has
            // zero rows and `S = Dᵀ D` is the p×p zero matrix; equivalently,
            // every length-p sequence is a polynomial of degree < k restricted
            // to the integer grid, so the entire coefficient space is in the
            // penalty's null space. Append that degenerate-but-mathematically-
            // consistent penalty rather than silently dropping the user's
            // request — silently discarding requested penalty orders hides
            // misconfiguration and changes the model the caller asked for.
            let zero_penalty = ndarray::Array2::<f64>::zeros((p, p));
            block
                .penalties
                .push(crate::model_types::PenaltySpec::Dense(zero_penalty));
            block.nullspace_dims.push(p);
            continue;
        }
        let penalty =
            create_difference_penalty_matrix(p, order, None).map_err(|e| e.to_string())?;
        block
            .penalties
            .push(crate::model_types::PenaltySpec::Dense(penalty));
        block.nullspace_dims.push(order);
    }
    Ok(())
}

pub(crate) fn select_wiggle_basis_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
    penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let (primary_order, extra_orders) =
        split_wiggle_penalty_orders(cfg.penalty_order, penalty_orders);
    let effective_cfg = WiggleBlockConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_order: primary_order,
        double_penalty: cfg.double_penalty,
    };
    let (mut block, knots) = buildwiggle_block_input_from_seed(seed, &effective_cfg)?;
    append_selected_wiggle_penalty_orders(&mut block, &extra_orders)?;
    Ok(SelectedWiggleBasis {
        knots,
        degree: cfg.degree,
        block,
    })
}
