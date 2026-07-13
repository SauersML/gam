use super::*;

// ---------------------------------------------------------------------------
// Response-direction basis construction
// ---------------------------------------------------------------------------

/// Build the response-direction basis: an unconstrained location column plus
/// I-spline values `I_k(y)` with derivatives `M_k(y) = I'_k(y)`.
///
/// Returns (value_basis = `[1, I_k]`, derivative_basis = `[0, M_k]`,
/// penalties embedded with an unpenalized location row/column, regenerated
/// I-spline knots, identity coef_transform for the I-spline shape block).
pub(crate) fn build_response_basis(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array2<f64>,
    ),
    String,
> {
    let n = response.len();
    if n < 4 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("need at least 4 observations, got {n}"),
        }
        .into());
    }
    for (i, &v) in response.iter().enumerate() {
        if !v.is_finite() {
            return Err(TransformationNormalError::NonFinite {
                reason: format!("response[{i}] is not finite: {v}"),
            }
            .into());
        }
    }

    let response_degree = config.response_degree;
    if response_degree < 1 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "response_degree must be >= 1 for the I-spline basis, got {response_degree}"
            ),
        }
        .into());
    }
    let k_internal = config.response_num_internal_knots;
    let k_prime = k_internal.checked_sub(2).ok_or_else(|| {
        format!(
            "response_num_internal_knots = {k_internal}; I-spline contract \
             requires K' = K − 2 ≥ 0, so need K ≥ 2"
        )
    })?;

    // Regenerate clamped knots. The I-spline builder integrates a degree
    // `(response_degree + 1)` B-spline basis into a degree-`response_degree`
    // value basis, so the seed-time degree passed here is `response_degree + 1`.
    let mut knots =
        initializewiggle_knots_from_seed(response.view(), response_degree + 1, k_prime)?;
    let response_min = response.iter().copied().fold(f64::INFINITY, f64::min);
    let response_max = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let response_span = (response_max - response_min).abs().max(1.0);
    let support_guard = response_span * 1.0e-3;
    let boundary_repeats = response_degree + 2;
    if knots.len() >= 2 * boundary_repeats {
        for idx in 0..boundary_repeats {
            knots[idx] = response_min - support_guard;
            let right_idx = knots.len() - 1 - idx;
            knots[right_idx] = response_max + support_guard;
        }
    }

    // I-spline value basis I_k(y).
    let (i_val_basis, _) = create_basis::<Dense>(
        response.view(),
        KnotSource::Provided(knots.view()),
        response_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    let shape_val = i_val_basis.as_ref().clone();
    let p_shape = shape_val.ncols();

    // M-spline derivative basis M_k(y) = I'_k(y).
    let shape_deriv = create_ispline_derivative_dense(response.view(), &knots, response_degree, 1)
        .map_err(|e| e.to_string())?;
    if shape_deriv.ncols() != p_shape {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "I-spline derivative column count {} does not match value basis {p_shape}",
                shape_deriv.ncols()
            ),
        }
        .into());
    }
    if shape_deriv.nrows() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "I-spline derivative row count {} does not match n = {n}",
                shape_deriv.nrows()
            ),
        }
        .into());
    }

    let p_resp = p_shape + 1;
    let mut resp_val = Array2::<f64>::zeros((n, p_resp));
    resp_val.column_mut(0).fill(1.0);
    resp_val.slice_mut(s![.., 1..]).assign(&shape_val);
    let mut resp_deriv = Array2::<f64>::zeros((n, p_resp));
    resp_deriv.slice_mut(s![.., 1..]).assign(&shape_deriv);

    // SCOP-CTN coef-transform is identity: I-splines have no constant in
    // their span, and squaring γ removes the per-component sign null direction.
    let transform = Array2::<f64>::eye(p_shape);

    // SPEC-5: the response-direction penalty is the EXACT function-space
    // roughness of the represented I-spline value function, not a
    // coefficient-difference operator. For derivative order `m` the shape
    // block carries
    //
    //     S_{y,m} = Cᵀ (∫ B_q^{(m)}(y) B_q^{(m)}(y)ᵀ dy) C,
    //
    // assembled span by span by Gauss–Legendre with the I-spline cumulative
    // frame `C` (see `ispline_function_penalties`). This is a quadratic
    // functional of the represented function itself, so it is scale- and
    // knot-width-aware (a difference operator is not) while remaining exactly
    // quadratic in the shape coefficients, i.e. compatible with the fixed
    // Gaussian-quadratic REML normalizer.
    //
    // Scope note (#2306): SCOP finalizes the response coefficients as
    // α = γ², so this Gram penalizes the roughness of the shape factor γ
    // (the √α function) rather than the final transformation h(y,x). The true
    // final-function penalty ½·vec(A)ᵀ(S_{y,m}⊗G_x)vec(A) requires the
    // direct-α reparameterization (h linear in A); until that lands, the
    // exact function-space Gram on γ is the correct REML-compatible metric
    // and supplies the exact per-order S_{y,m} the tensor penalty will reuse.
    //
    // The represented value functions have per-span polynomial degree
    // `value_degree = response_degree + 1`; the `m`-th derivative of a
    // degree-`value_degree` piecewise polynomial vanishes identically for
    // `m > value_degree`, so such an order carries no function roughness and
    // is a hard configuration error rather than a silently skipped no-op.
    // Embed each Gram into the full response block with an unpenalized
    // location row/column.
    let value_degree = response_degree + 1;
    let mut resp_penalties = Vec::new();
    let add_penalty = |order: usize, penalties: &mut Vec<Array2<f64>>| -> Result<(), String> {
        if order == 0 {
            return Err(TransformationNormalError::InvalidInput {
                reason: "response penalty derivative order must be >= 1; order 0 is the value \
                         function, not a roughness penalty"
                    .to_string(),
            }
            .into());
        }
        if order > value_degree {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response penalty derivative order {order} exceeds the I-spline value degree \
                     {value_degree}; the {order}-th derivative of the response basis is \
                     identically zero, so this order carries no function-space roughness"
                ),
            }
            .into());
        }
        let function_penalty = ispline_function_penalties(knots.view(), response_degree, order, false)
            .map_err(|e| e.to_string())?;
        let shape_pen = function_penalty.roughness;
        if shape_pen.dim() != (p_shape, p_shape) {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "order-{order} I-spline function roughness is {}x{} but the response shape \
                     block has {p_shape} columns",
                    shape_pen.nrows(),
                    shape_pen.ncols(),
                ),
            }
            .into());
        }
        let mut full_pen = Array2::<f64>::zeros((p_resp, p_resp));
        full_pen.slice_mut(s![1.., 1..]).assign(&shape_pen);
        penalties.push(full_pen);
        Ok(())
    };
    add_penalty(config.response_penalty_order, &mut resp_penalties)?;
    for &order in &config.response_extra_penalty_orders {
        if order == config.response_penalty_order {
            continue;
        }
        add_penalty(order, &mut resp_penalties)?;
    }

    Ok((resp_val, resp_deriv, resp_penalties, knots, transform))
}

pub(crate) fn response_endpoint_value_bases(transform: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut lower = Array1::<f64>::zeros(transform.ncols() + 1);
    let mut upper = Array1::<f64>::zeros(transform.ncols() + 1);
    lower[0] = 1.0;
    upper[0] = 1.0;
    for col in 0..transform.ncols() {
        upper[col + 1] = transform.column(col).sum();
    }
    (lower, upper)
}

pub(crate) fn response_floor_offsets(
    response: &Array1<f64>,
    knots: &Array1<f64>,
    response_median: f64,
) -> (Array1<f64>, f64, f64) {
    let row_offsets = response.mapv(|y| TRANSFORMATION_MONOTONICITY_EPS * (y - response_median));
    let lower_y = knots
        .first()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::INFINITY, f64::min));
    let upper_y = knots
        .last()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    (
        row_offsets,
        TRANSFORMATION_MONOTONICITY_EPS * (lower_y - response_median),
        TRANSFORMATION_MONOTONICITY_EPS * (upper_y - response_median),
    )
}

/// Data-driven cap on the response-shape internal-knot budget keyed on how far
/// the marginal response distribution is from a location-scale Gaussian.
///
/// The CTN response-direction I-spline block exists solely to bend the
/// transformation `h(y)` away from the affine `(y − μ)/σ` map that already makes
/// a homoskedastic-Gaussian response standard normal. When the marginal
/// response is itself close to Gaussian (after centering/scaling), the
/// data carry essentially no information to identify those bend directions:
/// every shape×covariate tensor coordinate beyond "constant scale × location
/// shift" is weakly identified, so a degree-3, 10-internal-knot block (~13
/// response columns, ~100+ tensor coefficients) makes the custom-family
/// optimizer re-factorize a dense exact SCOP Hessian for directions that
/// contribute nothing to the likelihood — the #720 timeout.
///
/// The complexity score is `|skewness| + ½·|excess kurtosis|`, both classic
/// departures from normality that the response-shape basis is there to absorb.
/// For a clean location-scale Gaussian transformation the score is ≈ 0 and the
/// budget collapses to a handful of knots; for genuinely nonlinear / skewed /
/// heteroskedastic transformations (heavy-tailed survival times, censored or
/// log-normal responses, multimodal mixtures) the score is large and the budget
/// relaxes back to the configured count, preserving CTN's expressiveness on
/// real transformations. This adapts the *effective* basis size rather than
/// shrinking the default, so nonlinear accuracy is untouched.
pub(crate) fn transformation_complexity_knot_budget(
    response: ArrayView1<'_, f64>,
    min_internal: usize,
) -> usize {
    let n = response.len();
    if n < 8 {
        // Too few rows to estimate higher moments reliably; do not let a noisy
        // moment estimate gate the basis — fall back to the structural caps.
        return usize::MAX;
    }
    let n_f = n as f64;
    let mean = response.iter().copied().sum::<f64>() / n_f;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &y in response.iter() {
        let d = y - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n_f;
    m3 /= n_f;
    m4 /= n_f;
    if m2 <= 0.0 || !m2.is_finite() {
        // Degenerate (constant) response: no shape information at all.
        return min_internal;
    }
    let sd = m2.sqrt();
    let skewness = (m3 / (sd * sd * sd)).abs();
    // Excess kurtosis (Gaussian reference subtracts 3).
    let excess_kurtosis = (m4 / (m2 * m2) - 3.0).abs();
    let complexity = skewness + 0.5 * excess_kurtosis;
    // Each unit of non-normality unlocks a few extra interior knots. A clean
    // Gaussian (complexity ≈ 0) keeps just `min_internal`; moderate departures
    // (complexity ≳ 1) already unlock a rich block, and heavy departures
    // saturate the structural caps below. The slope is deliberately generous so
    // mild nonlinearity is not under-resolved.
    let extra = (complexity * 6.0).round() as usize;
    min_internal.saturating_add(extra)
}

pub fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
    response: ArrayView1<'_, f64>,
) -> usize {
    // I-spline contract requires K' = K − 2 ≥ 0, i.e. K ≥ 2 internal knots.
    let min_internal = 2usize;
    let sample_cap = (n_obs / 10).max(min_internal);
    let tensor_width_cap = (BASE_TRANSFORMATION_TENSOR_WIDTH + n_obs / 25)
        .min(LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH);
    let max_resp_cols_from_tensor =
        (tensor_width_cap / p_cov.max(1)).max(config.response_degree + 2);
    // One response column is the unconstrained location; the remaining columns
    // are the I-spline shape block controlled by response_num_internal_knots.
    let max_shape_cols_from_tensor = max_resp_cols_from_tensor.saturating_sub(1);
    let tensor_cap = max_shape_cols_from_tensor
        .saturating_sub(config.response_degree + 1)
        .max(min_internal);
    // Data-driven cap: a near-Gaussian transformation does not need (and cannot
    // identify) a heavy shape block. This trims the dense SCOP Hessian /
    // tensor-coefficient cost on easy signals while leaving genuinely nonlinear
    // transformations at the full structural budget.
    let complexity_cap = transformation_complexity_knot_budget(response, min_internal);
    config
        .response_num_internal_knots
        .min(sample_cap)
        .min(tensor_cap)
        .min(complexity_cap)
        .max(min_internal)
}

// ---------------------------------------------------------------------------
// Tensor product construction
// ---------------------------------------------------------------------------

pub(crate) fn assert_rowwise_kronecker_dimensions(
    n: usize,
    p_resp: usize,
    p_cov: usize,
    context: &str,
) -> Result<(), String> {
    if p_resp == 0 || p_cov == 0 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
            ),
        }
        .into());
    }
    Ok(())
}
