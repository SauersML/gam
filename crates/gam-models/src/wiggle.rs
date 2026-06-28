use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};
use crate::parameter_block::ParameterBlockInput;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_solve::pirls::LinearInequalityConstraints;
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

// #1521: relocated DOWN into `gam_terms::basis` (was a gamlss/wiggle helper).
// The knot-generation primitive carries no model-family type, so family modules
// and this module's own callers consume it from the basis layer via this
// re-export — keeping every `crate::wiggle::initializewiggle_knots_from_seed`
// call site (gamlss / bms / transformation-normal) resolving unchanged.
pub(crate) use gam_terms::basis::initializewiggle_knots_from_seed;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_types::PenaltySpec;
    use ndarray::Array1;

    fn dense_penalty(spec: &PenaltySpec) -> &Array2<f64> {
        match spec {
            PenaltySpec::Dense(m) => m,
            other => panic!("expected Dense penalty, got {other:?}"),
        }
    }

    fn is_symmetric(m: &Array2<f64>) -> bool {
        let n = m.nrows();
        if m.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                if (m[[i, j]] - m[[j, i]]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    // ---- monotone_wiggle_internal_degree ----

    #[test]
    fn internal_degree_rejects_degree_below_two() {
        // degree 0 and 1 yield internal_degree < 1 -> Err with the documented message.
        for d in [0usize, 1] {
            let err = monotone_wiggle_internal_degree(d).unwrap_err();
            assert_eq!(err, "monotone wiggle degree must be >= 2");
        }
    }

    #[test]
    fn internal_degree_is_degree_minus_one_for_valid_degrees() {
        // degree >= 2 -> Ok(degree - 1): the per-span value degree aligned to the
        // public value-basis degree.
        assert_eq!(monotone_wiggle_internal_degree(2).unwrap(), 1);
        assert_eq!(monotone_wiggle_internal_degree(3).unwrap(), 2);
        assert_eq!(monotone_wiggle_internal_degree(10).unwrap(), 9);
    }

    // ---- buildwiggle_block_input_from_knots (driven via seed for valid knots) ----

    fn build(double_penalty: bool, penalty_order: usize) -> (ParameterBlockInput, usize) {
        // A spread-out seed so knot generation yields several monotone columns.
        let seed = Array1::linspace(0.0, 1.0, 40);
        let cfg = WiggleBlockConfig {
            degree: 3,
            num_internal_knots: 5,
            penalty_order,
            double_penalty,
        };
        let knots = initializewiggle_knots_from_seed(seed.view(), cfg.degree, cfg.num_internal_knots)
            .expect("knot init");
        let block = buildwiggle_block_input_from_knots(
            seed.view(),
            &knots,
            cfg.degree,
            cfg.penalty_order,
            cfg.double_penalty,
        )
        .expect("build block");
        let p = block.design.ncols();
        (block, p)
    }

    #[test]
    fn single_penalty_block_shapes_and_invariants() {
        let (block, p) = build(false, 2);
        assert!(p >= 2, "expected multiple monotone columns, got p={p}");
        // Offset is zeros with length = seed length.
        assert_eq!(block.offset.len(), 40);
        assert!(block.offset.iter().all(|&v| v == 0.0));
        // initial_beta is Some(zeros(p)).
        let beta = block.initial_beta.as_ref().expect("initial_beta");
        assert_eq!(beta.len(), p);
        assert!(beta.iter().all(|&v| v == 0.0));
        // Without double penalty there is exactly one penalty.
        assert_eq!(block.penalties.len(), 1);
        assert_eq!(block.nullspace_dims.len(), 1);
        // The penalty is the p x p difference penalty; symmetric (S = Dᵀ D).
        let s = dense_penalty(&block.penalties[0]);
        assert_eq!(s.dim(), (p, p));
        assert!(is_symmetric(s));
        // effective_order = penalty_order.max(1).min(p-1); here 2 (<= p-1 since p>=3
        // for this seed). nullspace_dim equals the effective difference order.
        let effective = 2usize.max(1).min(p - 1);
        assert_eq!(block.nullspace_dims[0], effective);
    }

    #[test]
    fn double_penalty_appends_identity_ridge() {
        let (block, p) = build(true, 2);
        assert!(p >= 2);
        // double_penalty -> two penalties: difference penalty then p x p identity.
        assert_eq!(block.penalties.len(), 2);
        assert_eq!(block.nullspace_dims.len(), 2);
        let ridge = dense_penalty(&block.penalties[1]);
        assert_eq!(ridge.dim(), (p, p));
        // Identity: diagonal ones, off-diagonal zeros.
        for i in 0..p {
            for j in 0..p {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(ridge[[i, j]], expected);
            }
        }
        // The appended identity is full rank, so its nullspace dim is 0.
        assert_eq!(block.nullspace_dims[1], 0);
    }

    #[test]
    fn penalty_order_clamped_to_p_minus_one() {
        // Requesting an absurdly large penalty order clamps effective_order to p-1
        // (still a valid difference penalty), per `penalty_order.max(1).min(p-1)`.
        let (block, p) = build(false, 10_000);
        assert!(p >= 2);
        let s = dense_penalty(&block.penalties[0]);
        assert_eq!(s.dim(), (p, p));
        assert!(is_symmetric(s));
        assert_eq!(block.nullspace_dims[0], p - 1);
    }
}
