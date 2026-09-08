//! Deterministic posterior moments for logistic-normal softmax probabilities.
//!
//! A reference-coded multinomial model has `M = K - 1` active logits.  At one
//! prediction row the Laplace posterior induces
//!
//! ```text
//! eta ~ Normal(mu, V),
//! p(eta) = softmax(eta_0, ..., eta_{M-1}, 0).
//! ```
//!
//! This module computes `E[p]` and `Cov(p)` rather than the plug-in quantity
//! `softmax(E[eta])`.  The binary case is reduced to the controlled scalar
//! logistic-normal evaluator in `gam-solve`.  For `K > 2`, the covariance is
//! eigendecomposed and quadrature is performed only over its positive range.
//! Successive Smolyak levels built from odd-order Gauss-Hermite rules provide a
//! deterministic error check.  Failure to establish the requested tolerance is
//! an error; there is deliberately no Monte Carlo or plug-in fallback.

use crate::model_types::EstimationError;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::quadrature::gauss_hermite_rule as physicists_gauss_hermite_rule;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::BTreeMap;

/// Backward-error multiplier used when deciding whether a symmetric covariance
/// eigenvalue is negative beyond floating-point eigensolver roundoff.
///
/// This is not a variance jitter: the input matrix is never modified by adding
/// a diagonal ridge.  Eigenvalues below `-tol` are rejected, while values whose
/// magnitude is within the backward-error envelope are treated as numerical
/// zero.
const PSD_BACKWARD_ERROR_MULTIPLIER: f64 = 16.0;

/// Floating-point summation envelope for the signed Smolyak combination.
const SUMMATION_ROUNDOFF_MULTIPLIER: f64 = 16.0;

/// Explicit accuracy and work controls for multinomial posterior integration.
///
/// The production default is explicit through [`Default`] and is carried by
/// the prediction request into this kernel. `minimum_sparse_level >= 1`
/// guarantees at least one comparison against a preceding Smolyak level.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialPosteriorIntegrationControl {
    /// Per raw moment absolute tolerance.  Raw moments comprise every `E[p_c]`
    /// and `E[p_c p_d]` for `c <= d`.
    pub absolute_tolerance: f64,
    /// Per raw moment relative tolerance.
    pub relative_tolerance: f64,
    /// Earliest Smolyak refinement level that may certify convergence.
    pub minimum_sparse_level: usize,
    /// Last Smolyak refinement level attempted.
    pub maximum_sparse_level: usize,
    /// Maximum total integrand evaluations across all attempted levels.
    pub maximum_function_evaluations: usize,
}

impl Default for MultinomialPosteriorIntegrationControl {
    fn default() -> Self {
        // sqrt(machine epsilon) is the natural accuracy target for a nonlinear
        // transform of a covariance estimated in double precision: asking for
        // substantially more would certify quadrature noise below the input's
        // own numerical resolution. Three sparse levels are required before a
        // result may certify. The level ceiling is 12 (the 25-point
        // one-dimensional Gauss–Hermite rule): a converged integrand certifies
        // early and never visits the deeper levels, so the ceiling only
        // matters for WIDE posteriors — the #2344 equivariant class metric
        // honestly penalizes the class-mean logit direction at λ/K, ≈K× the
        // variance of the old ALR-anchored penalty, and at the former ceiling
        // of 8 such rows plateaued at a level difference ~1.6× the tolerance
        // with only ~0.1% of the evaluation budget spent (#2350). The
        // evaluation ceiling remains the cost guard; the streaming evaluator
        // never stores the node set.
        //
        // RAISED 12 -> 16 (#2612), for the same reason #2350 raised it 8 -> 12,
        // and the evidence is the same shape. `#2612` needed the multinomial df
        // floor fraction `f` raised to sharpen under-confident penguin
        // probabilities; a larger `f` caps rho lower, hence less shrinkage, hence
        // a WIDER posterior — and at `f = 0.90` a sibling train/test split stopped
        // PREDICTING with `did not converge through Smolyak level 12`, having
        // spent 9633 of its 2000000 evaluations. 0.5% of the cost guard: the LEVEL
        // bound was binding, not the budget. At 16 that split certifies and scores
        // held-out log-loss 0.17246 against nnet's 0.76930.
        //
        // This costs nothing where nothing is wrong. A converged integrand
        // certifies early and never visits the deeper levels, so the extra
        // headroom is only ever spent by the wide posteriors that need it — which
        // is why the ceiling, and not the evaluation budget, is the right thing to
        // move.
        let tolerance = f64::EPSILON.sqrt();
        Self {
            absolute_tolerance: tolerance,
            relative_tolerance: tolerance,
            minimum_sparse_level: 2,
            // The level ceiling STAYS, and here is the measurement that says so.
            //
            // I removed it (45e3f1876) on the argument that there were two cost
            // guards for one cost and only one was denominated in the cost:
            //   8 -> 12  (#2350): plateaued having spent ~0.1% of the budget
            //   12 -> 16 (#2612): stopped predicting at 9633/2000000  = 0.5%
            //   16 -> ?  (#2612): stops predicting at 28033/2000000 = 1.4%
            // Three raises, one argument, and each time the level bound really
            // was what bound. That reasoning is still right about the DEFECT.
            //
            // I THEN REVERTED IT ON A BAD COMPARISON, and this paragraph is the
            // correction. The revert cited "941 s and still running, against a
            // refusal in seconds before". The 941 s is real; the "refusal in
            // seconds" was a DIFFERENT TEST -- a residual-cascade fixture on
            // #2546 -- not this path. Re-measured AT the revert on an unloaded
            // node, the same penguins arm ran 931 s: indistinguishable from the
            // 941 s I had blamed on removing the ceiling. So the ceiling was not
            // the cause of the cost, and the evidence did not support the
            // conclusion I drew from it.
            //
            // What is actually known, so the next reader does not inherit my
            // error: this arm costs ~930 s in BOTH states, and no COMPLETED run
            // was obtained in either -- so whether removing the ceiling changes
            // the outcome, or the cost, is UNMEASURED.
            //
            // The original concern survives as a concern, not a measurement.
            // With the ceiling removed, the penguins real-data arm ran **941 s
            // and was still going**, against a refusal in seconds before. The
            // reason is the other half of the same observation: because the
            // level cap always bound first, the 2,000,000-evaluation budget was
            // NEVER calibrated to bind. It was sized on the assumption that
            // something else would stop first, so it is not a usable sole
            // guard — it is a backstop, and reaching it costs minutes per
            // prediction.
            //
            // So removing the binding guard without checking the remaining one
            // was sized for the job traded a fast honest refusal for a slow
            // grind. That is the same error as dropping a cost-relative
            // stationarity band and leaving only its resolution floor (#2613):
            // the two quantities are not interchangeable, and one of them was
            // never a budget.
            //
            // What a correct fix needs, and what I did not have: a budget
            // derived from the cost per level and the per-prediction time this
            // path is allowed to spend, so that ONE guard bounds the cost in
            // usable time. Until that derivation exists, a level cap that
            // refuses in seconds is better than a budget that grinds for
            // sixteen minutes — a refusal a caller can act on beats an answer
            // it cannot wait for.
            // AND IT IS NO LONGER THE THING THAT DECIDES A WIDE POSTERIOR.
            //
            // Every raise above was triggered by a posterior getting wider, and
            // the cause was the RULE, not the ceiling. Measured on a rank-two
            // covariance with posterior standard deviations (1, 5), against a
            // converged tensor oracle:
            //
            //   sparse level 10 (  4961 evals): error 6.61e-4
            //   sparse level 12 (  9633 evals): error 5.48e-4
            //   sparse level 14 ( 17025 evals): error 3.50e-4
            //   sparse level 16 ( 28033 evals): error 1.99e-4
            //   tensor   33/dim (  1089 evals): error 1.84e-4
            //   tensor   65/dim (  4225 evals): error 1.66e-6
            //   tensor  129/dim ( 16641 evals): error 3.82e-9
            //
            // The sparse ladder decays algebraically -- a factor 3.3 for 5.6x
            // the evaluations -- because a Smolyak grid gives one direction high
            // order only by giving every other direction a single node, while a
            // logistic-normal softmax needs order in EVERY wide direction at
            // once. At rank two the tensor product the grid is assembled FROM is
            // strictly better, reaching five more digits for fewer evaluations.
            // So this ceiling is no longer a wall: `integrate_general` tries
            // the sparse rule first and keeps it wherever it certifies -- which
            // costs a few hundred evaluations and is why it is tried first --
            // and reaching this ceiling now HANDS THE ROW to the tensor rule
            // instead of refusing. Raising it again would only make the sparse
            // rule spend longer before handing over (#2612).
            maximum_sparse_level: 16,
            maximum_function_evaluations: 2_000_000,
        }
    }
}

/// Integrated posterior means and marginal standard deviations for every row
/// of a multinomial prediction design.
#[derive(Clone, Debug)]
pub struct MultinomialPosteriorRowMoments {
    pub class_mean: Array2<f64>,
    pub class_standard_deviation: Array2<f64>,
}

/// Integrate the logistic-normal posterior induced by a coefficient mode and
/// its full joint covariance over every design row.
///
/// Coefficients have shape `(P, M)`, covariance has block-major shape
/// `(P*M, P*M)`, and `design` has shape `(N, P)`. For row `x`, this constructs
/// `mu_a = x' beta_a` and `V_ab = x' Sigma_ab x`, then delegates to the
/// controlled one-row integrator. Cross-class covariance blocks are retained.
pub fn integrate_multinomial_design_moments(
    coefficients: ArrayView2<'_, f64>,
    coefficient_covariance: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorRowMoments, EstimationError> {
    let (p, m) = coefficients.dim();
    if p == 0 || m == 0 {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction needs nonempty coefficients, got {p}x{m}"
        )));
    }
    if design.ncols() != p {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction design has {} columns, expected {p}",
            design.ncols()
        )));
    }
    let d = p.checked_mul(m).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior prediction coefficient dimension overflowed usize".to_string(),
        )
    })?;
    if coefficient_covariance.dim() != (d, d) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction covariance shape {:?} does not match (P*M, P*M) = ({d}, {d})",
            coefficient_covariance.dim()
        )));
    }

    let n = design.nrows();
    let k = m + 1;
    let mut class_mean = Array2::<f64>::zeros((n, k));
    let mut class_standard_deviation = Array2::<f64>::zeros((n, k));
    let mut active_mean = Array1::<f64>::zeros(m);
    let mut active_covariance = Array2::<f64>::zeros((m, m));
    // Every row follows the same deterministic order-doubling ladder.  A rule
    // depends only on its order, not on the row's mean or covariance, so build
    // each order once for this prediction call and reuse it across rows.
    let mut conditioned_three_class_rules = ConditionedThreeClassRuleLadder::default();
    for row in 0..n {
        let x = design.row(row);
        for a in 0..m {
            active_mean[a] = x.dot(&coefficients.column(a));
        }
        for a in 0..m {
            for b in 0..m {
                let mut value = 0.0_f64;
                let a_base = a * p;
                let b_base = b * p;
                for i in 0..p {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let mut row_product = 0.0_f64;
                    for j in 0..p {
                        row_product += coefficient_covariance[[a_base + i, b_base + j]] * x[j];
                    }
                    value += xi * row_product;
                }
                active_covariance[[a, b]] = value;
            }
        }
        let moments = integrate_logistic_normal_softmax_moments_with_rule_ladder(
            active_mean.view(),
            active_covariance.view(),
            control,
            &mut conditioned_three_class_rules,
        )?;
        class_mean.row_mut(row).assign(&moments.class_mean);
        class_standard_deviation
            .row_mut(row)
            .assign(&moments.class_standard_deviation);
    }
    Ok(MultinomialPosteriorRowMoments {
        class_mean,
        class_standard_deviation,
    })
}

impl MultinomialPosteriorIntegrationControl {
    fn validate(&self) -> Result<(), EstimationError> {
        if !(self.absolute_tolerance.is_finite() && self.absolute_tolerance >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration absolute_tolerance must be finite and >= 0, got {}",
                self.absolute_tolerance
            )));
        }
        if !(self.relative_tolerance.is_finite() && self.relative_tolerance >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration relative_tolerance must be finite and >= 0, got {}",
                self.relative_tolerance
            )));
        }
        if self.absolute_tolerance == 0.0 && self.relative_tolerance == 0.0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration requires a positive absolute or relative tolerance"
                    .to_string(),
            ));
        }
        if self.minimum_sparse_level == 0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration minimum_sparse_level must be >= 1 so a level difference exists"
                    .to_string(),
            ));
        }
        if self.maximum_sparse_level < self.minimum_sparse_level {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration maximum_sparse_level ({}) is below minimum_sparse_level ({})",
                self.maximum_sparse_level, self.minimum_sparse_level
            )));
        }
        if self.maximum_function_evaluations == 0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration maximum_function_evaluations must be positive"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Which deterministic rule produced a certified answer.
///
/// The rank of the retained posterior decides which rule can certify inside the
/// evaluation budget, and the two are not interchangeable: a sparse grid is a
/// saving only when the resolution a direction needs is much smaller than the
/// number of directions, and at rank two it is strictly worse than the tensor
/// product it is built from (#2612).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultinomialPosteriorRule {
    /// Exact reduction: the binary logistic-normal evaluator, or a covariance
    /// that is a point mass.
    Exact,
    /// Exact Gaussian conditioning reduces a three-class, rank-two posterior
    /// to one Gauss-Hermite direction plus the controlled scalar
    /// logistic-normal evaluator. Carries the outer rule's node count.
    ConditionedThreeClass(usize),
    /// Tensor-product Gauss-Hermite whose one-dimensional node count is chosen
    /// per retained posterior direction.  Carries those node counts.
    AnisotropicTensor(Vec<usize>),
    /// Isotropic Smolyak sparse grid, carrying the level that certified.
    IsotropicSparse(usize),
}

/// Integrated class-probability moments for one prediction row.
///
/// `class_covariance` includes the reference class and is singular in the
/// all-ones direction, as required by `sum_c p_c = 1`.  A value of this type is
/// only constructed after the requested level-difference certificate succeeds.
#[derive(Clone, Debug)]
pub struct MultinomialPosteriorMoments {
    /// `E[p_c]`, length `K`, including the reference class last.
    pub class_mean: Array1<f64>,
    /// `Cov(p_c, p_d)`, shape `(K, K)`.
    pub class_covariance: Array2<f64>,
    /// Marginal posterior standard deviations `sqrt(Var(p_c))`.
    pub class_standard_deviation: Array1<f64>,
    /// Positive numerical rank of the active-logit covariance.
    pub latent_rank: usize,
    /// The rule that certified convergence.
    pub rule: MultinomialPosteriorRule,
    /// Total softmax evaluations across all attempted sparse levels.
    pub function_evaluations: usize,
    /// Largest absolute difference among raw first/second moments between the
    /// certifying rule and the coarser rule it was compared against.  Zero on
    /// the exact binary and point-mass paths.
    pub max_raw_moment_level_difference: f64,
    /// Bound used for positive covariance eigenmodes discarded inside the
    /// eigensolver backward-error envelope.  Such modes are discarded only
    /// when this bound fits inside the requested absolute tolerance.
    pub covariance_range_projection_bound: f64,
}

/// Integrate reference-coded logistic-normal softmax moments for one row.
///
/// `active_mean` has length `M = K - 1`; `active_covariance` must be a finite,
/// symmetric positive-semidefinite `(M, M)` matrix in the same active-class
/// order.  The returned arrays include the implicit reference class as their
/// final entry.
pub fn integrate_logistic_normal_softmax_moments(
    active_mean: ArrayView1<'_, f64>,
    active_covariance: ArrayView2<'_, f64>,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let mut conditioned_three_class_rules = ConditionedThreeClassRuleLadder::default();
    integrate_logistic_normal_softmax_moments_with_rule_ladder(
        active_mean,
        active_covariance,
        control,
        &mut conditioned_three_class_rules,
    )
}

fn integrate_logistic_normal_softmax_moments_with_rule_ladder(
    active_mean: ArrayView1<'_, f64>,
    active_covariance: ArrayView2<'_, f64>,
    control: &MultinomialPosteriorIntegrationControl,
    conditioned_three_class_rules: &mut ConditionedThreeClassRuleLadder,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    control.validate()?;
    validate_inputs(active_mean, active_covariance)?;
    // Integrate the nearest symmetric matrix. (C + Cᵀ)/2 is exact in floating
    // point for the off-diagonal average and is what every downstream
    // eigenroutine assumes it was handed; propagating one arbitrary triangle
    // instead would make the result depend on which triangle happened to be
    // read.
    let symmetric_covariance = symmetrized_covariance(active_covariance);
    let active_covariance = symmetric_covariance.view();

    let mean = active_mean.to_vec();
    let m = mean.len();
    if m == 1 {
        return integrate_binary(mean[0], active_covariance[[0, 0]]);
    }

    let maximum_covariance_entry = active_covariance
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    if maximum_covariance_entry == 0.0 {
        return point_mass_moments(&mean);
    }

    let projected = project_active_covariance(active_covariance, control.absolute_tolerance)?;
    if projected.factor.ncols() == 0 {
        // This arm is reachable only when every positive eigenmode lies inside
        // the eigensolver backward-error envelope and its explicit probability
        // bound fits within the caller's tolerance.  It is therefore a
        // certified point-mass approximation, not a silent plug-in fallback.
        let mut out = point_mass_moments(&mean)?;
        out.covariance_range_projection_bound = projected.projection_bound;
        return Ok(out);
    }
    if m == 2 && projected.factor.ncols() == 2 {
        return integrate_three_class_conditionally(
            &mean,
            &projected,
            control,
            conditioned_three_class_rules,
        );
    }

    integrate_general(&mean, &projected, control)
}

fn validate_inputs(
    active_mean: ArrayView1<'_, f64>,
    active_covariance: ArrayView2<'_, f64>,
) -> Result<(), EstimationError> {
    let m = active_mean.len();
    if m == 0 {
        return Err(EstimationError::InvalidInput(
            "multinomial posterior integration needs at least one active logit (K >= 2)"
                .to_string(),
        ));
    }
    if active_covariance.dim() != (m, m) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance shape {:?} does not match active mean length {m}",
            active_covariance.dim()
        )));
    }
    if let Some((index, value)) = active_mean
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration active_mean[{index}] is non-finite: {value}"
        )));
    }
    if let Some(((row, column), value)) = active_covariance
        .indexed_iter()
        .map(|(index, &value)| (index, value))
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance[{row},{column}] is non-finite: {value}"
        )));
    }

    let scale = active_covariance
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    // STRUCTURAL asymmetry only. This covariance is symmetric by construction,
    // so whatever difference survives between the triangles is roundoff from
    // the chain that assembled it — and a `c·ε·m·scale` envelope silently
    // encodes an assumed chain length. It fired at 51·ε on a 2×2 penguins
    // posterior (asymmetry 5.218e-15 against a 3.260e-15 bound), refusing a
    // correct fit over noise carrying no information.
    //
    // A caller error that this check exists to catch — a transposed factor,
    // the wrong triangle — shows up at O(1) RELATIVE asymmetry, so gate there,
    // using the same √ε relative convention `outer_value_agreement_bound` uses
    // for two lanes that should agree up to roundoff. The matrix actually
    // integrated is the symmetrized one (see `symmetrized_covariance`), so
    // sub-threshold asymmetry is removed rather than propagated.
    let symmetry_tolerance = f64::EPSILON.sqrt() * scale.max(1.0);
    let mut maximum_asymmetry = 0.0_f64;
    for row in 0..m {
        for column in (row + 1)..m {
            maximum_asymmetry = maximum_asymmetry
                .max((active_covariance[[row, column]] - active_covariance[[column, row]]).abs());
        }
    }
    if maximum_asymmetry > symmetry_tolerance {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance is not symmetric: max asymmetry {maximum_asymmetry:.6e} exceeds structural tolerance {symmetry_tolerance:.6e} (scale {scale:.6e})"
        )));
    }
    Ok(())
}

/// Nearest symmetric matrix to `covariance` in the Frobenius norm.
///
/// The inputs to this module are symmetric by construction; this removes the
/// roundoff-level asymmetry their assembly chain leaves behind, so the
/// integration cannot depend on which triangle a downstream routine reads.
fn symmetrized_covariance(covariance: ArrayView2<'_, f64>) -> Array2<f64> {
    let m = covariance.nrows();
    let mut out = covariance.to_owned();
    for row in 0..m {
        for column in (row + 1)..m {
            let average = 0.5 * (covariance[[row, column]] + covariance[[column, row]]);
            out[[row, column]] = average;
            out[[column, row]] = average;
        }
    }
    out
}

fn covariance_roundoff_tolerance(scale: f64, dimension: usize) -> f64 {
    PSD_BACKWARD_ERROR_MULTIPLIER * f64::EPSILON * (dimension.max(1) as f64) * scale
}

fn integrate_binary(
    active_mean: f64,
    active_variance: f64,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    if active_variance < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "binary logistic-normal variance must be non-negative, got {active_variance:.6e}"
        )));
    }
    let sigma = active_variance.sqrt();
    let (probability_mean, mean_logistic_slope) =
        gam_solve::quadrature::logit_posterior_meanwith_deriv(active_mean, sigma)?;

    // sigmoid'(eta) = p(1-p) = p-p^2, hence
    // E[p^2] = E[p] - d/dmu E[p].  This supplies the binary probability
    // variance from the same controlled scalar integral without a second
    // numerical approximation.
    let probability_second_moment = probability_mean - mean_logistic_slope;
    let variance = (probability_second_moment - probability_mean * probability_mean).max(0.0);
    let reference_mean = 1.0 - probability_mean;

    let class_mean = Array1::from_vec(vec![probability_mean, reference_mean]);
    let class_covariance =
        Array2::from_shape_vec((2, 2), vec![variance, -variance, -variance, variance]).map_err(
            |error| {
                EstimationError::InvalidInput(format!(
                    "binary logistic-normal covariance construction failed: {error}"
                ))
            },
        )?;
    let standard_deviation = variance.sqrt();
    Ok(MultinomialPosteriorMoments {
        class_mean,
        class_covariance,
        class_standard_deviation: Array1::from_vec(vec![standard_deviation, standard_deviation]),
        latent_rank: if active_variance > 0.0 { 1 } else { 0 },
        rule: MultinomialPosteriorRule::Exact,
        function_evaluations: 0,
        max_raw_moment_level_difference: 0.0,
        covariance_range_projection_bound: 0.0,
    })
}

/// Three softmax classes admit an exact one-dimensional Rao-Blackwellization.
///
/// Condition active logit `X` on the other active logit `Y`.  With
///
/// ```text
/// L = sigmoid(X - softplus(Y)), q = sigmoid(Y),
/// ```
///
/// the class probabilities are `p_x=L`, `p_y=q(1-L)`, and
/// `p_ref=(1-q)(1-L)`.  The controlled scalar logistic-normal evaluator gives
/// `a=E[L|Y]` and `d=E[L(1-L)|Y]`, so all conditional first and second moments
/// are algebra:
///
/// ```text
/// E[L²|Y]       = a-d,
/// E[(1-L)²|Y]   = 1-a-d,
/// E[L(1-L)|Y]   = d.
/// ```
///
/// Only the Gaussian expectation over `Y` remains.  This changes the work for
/// the wide rank-two posterior in #2612 from the Cartesian pair
/// `1023² + 2047²` to two one-dimensional rules while preserving every raw
/// moment and the same caller-tolerance convergence check.
fn integrate_three_class_conditionally(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    control: &MultinomialPosteriorIntegrationControl,
    rules: &mut ConditionedThreeClassRuleLadder,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let integrand = ThreeClassConditionalIntegrand::new(active_mean, projected)?;
    let mut previous: Option<Vec<f64>> = None;
    let mut total_evaluations = 0usize;
    let mut rule_index = 1usize;
    let mut refinement_depth = 0usize;
    let mut last_difference = f64::INFINITY;
    let mut last_deciding: Option<DecidingMoment> = None;

    loop {
        let node_count = rule_index
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "three-class conditional Gauss-Hermite order overflowed usize".to_string(),
                )
            })?;
        let remaining = control
            .maximum_function_evaluations
            .saturating_sub(total_evaluations);
        if node_count > remaining {
            let deciding_report = match last_deciding {
                Some(moment) => format!(
                    "worst raw moment {} (normalized error {:.6e} = (difference {:.6e} + projection bound {:.6e}) / tolerance {:.6e})",
                    moment.index,
                    moment.normalized_error,
                    moment.difference,
                    projected.projection_bound,
                    moment.tolerance,
                ),
                None => "no two conditional rules fit, so no raw moment was compared".to_string(),
            };
            return Err(EstimationError::InvalidInput(format!(
                "multinomial logistic-normal quadrature did not converge: the next one-dimensional conditioned three-class rule needs {node_count} evaluations, against {remaining} remaining; final max raw-moment difference {last_difference:.6e}, evaluations {total_evaluations}/{}; {deciding_report}",
                control.maximum_function_evaluations
            )));
        }

        let rule = rules.rule(refinement_depth, rule_index)?;
        let current = integrand.raw_moments(
            rule,
            &mut total_evaluations,
            control.maximum_function_evaluations,
            control.absolute_tolerance,
        )?;
        if let Some(previous_moments) = previous.as_ref() {
            let mut certified = true;
            let mut maximum_difference = 0.0_f64;
            let mut deciding: Option<DecidingMoment> = None;
            for (index, (&new_value, &old_value)) in
                current.iter().zip(previous_moments.iter()).enumerate()
            {
                let difference = (new_value - old_value).abs();
                maximum_difference = maximum_difference.max(difference);
                let tolerance = control.absolute_tolerance
                    + control.relative_tolerance * new_value.abs().max(old_value.abs());
                let controlled_error = difference + projected.projection_bound;
                if controlled_error > tolerance {
                    certified = false;
                }
                let normalized_error = if tolerance > 0.0 {
                    controlled_error / tolerance
                } else if controlled_error > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                if deciding
                    .as_ref()
                    .map(|current| normalized_error > current.normalized_error)
                    .unwrap_or(true)
                {
                    deciding = Some(DecidingMoment {
                        index,
                        normalized_error,
                        difference,
                        tolerance,
                    });
                }
            }
            last_difference = maximum_difference;
            last_deciding = deciding;
            if certified {
                return moments_from_raw(
                    current,
                    3,
                    2,
                    MultinomialPosteriorRule::ConditionedThreeClass(node_count),
                    total_evaluations,
                    maximum_difference,
                    projected.projection_bound,
                );
            }
        }
        previous = Some(current);
        rule_index = rule_index.checked_mul(2).ok_or_else(|| {
            EstimationError::InvalidInput(
                "three-class conditional Gauss-Hermite refinement overflowed usize".to_string(),
            )
        })?;
        refinement_depth = refinement_depth.checked_add(1).ok_or_else(|| {
            EstimationError::InvalidInput(
                "three-class conditional Gauss-Hermite refinement depth overflowed usize"
                    .to_string(),
            )
        })?;
    }
}

/// Prediction-call-owned cache for the order-doubling rule ladder used by the
/// exact three-class conditional reduction.
///
/// Depth `d` always means rule index `2^d` and therefore `2^(d+1)-1` nodes.
/// Storing the ladder densely by depth gives direct indexing without either a
/// global, unbounded cache or a map lookup for an order already fixed by the
/// refinement schedule.
#[derive(Default)]
struct ConditionedThreeClassRuleLadder {
    rules: Vec<GaussHermiteRule>,
}

impl ConditionedThreeClassRuleLadder {
    fn rule(
        &mut self,
        refinement_depth: usize,
        rule_index: usize,
    ) -> Result<&GaussHermiteRule, EstimationError> {
        if self.rules.len() == refinement_depth {
            self.rules.push(gauss_hermite_rule(rule_index)?);
        }
        self.rules.get(refinement_depth).ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "conditioned three-class rule ladder is missing refinement depth {refinement_depth}"
            ))
        })
    }
}

/// Parameters of `X | Y` and the class mapping for the exact three-class
/// reduction above.
struct ThreeClassConditionalIntegrand<'a> {
    active_mean: &'a [f64],
    conditioned_class: usize,
    outer_class: usize,
    outer_standard_deviation: f64,
    conditional_regression: f64,
    conditional_standard_deviation: f64,
    upper_offsets: Vec<usize>,
}

impl<'a> ThreeClassConditionalIntegrand<'a> {
    fn new(active_mean: &'a [f64], projected: &ProjectedGaussian) -> Result<Self, EstimationError> {
        if active_mean.len() != 2 || projected.factor.dim() != (2, 2) {
            return Err(EstimationError::InvalidInput(format!(
                "conditioned three-class integration requires two active logits and a 2x2 retained factor, got {} logits and factor {:?}",
                active_mean.len(),
                projected.factor.dim()
            )));
        }
        let row_variance = |row: usize| {
            projected
                .factor
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        };
        let variances = [row_variance(0), row_variance(1)];
        // One-dimensional Gauss-Hermite resolution grows with the marginal
        // standard deviation of the remaining outer coordinate.  Conditioning
        // the other coordinate therefore minimizes the outer rule's derived
        // work requirement without a tuned routing threshold.
        let outer_class = if variances[0] <= variances[1] { 0 } else { 1 };
        let conditioned_class = 1 - outer_class;
        let outer_variance = variances[outer_class];
        if !(outer_variance.is_finite() && outer_variance > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "conditioned three-class outer variance must be finite and positive, got {outer_variance}"
            )));
        }
        let covariance = projected
            .factor
            .row(conditioned_class)
            .iter()
            .zip(projected.factor.row(outer_class).iter())
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let conditional_regression = covariance / outer_variance;
        // Form the conditional residual in factor space.  Its squared norm is
        // Var(X|Y), evaluated without the catastrophic cancellation in
        // Var(X)-Cov(X,Y)^2/Var(Y) near a rank-one covariance.
        let conditional_variance = projected
            .factor
            .row(conditioned_class)
            .iter()
            .zip(projected.factor.row(outer_class).iter())
            .map(|(conditioned, outer)| {
                let residual = conditioned - conditional_regression * outer;
                residual * residual
            })
            .sum::<f64>();
        if !(conditional_variance.is_finite() && conditional_variance >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "conditioned three-class residual variance is invalid: {conditional_variance}"
            )));
        }
        Ok(Self {
            active_mean,
            conditioned_class,
            outer_class,
            outer_standard_deviation: outer_variance.sqrt(),
            conditional_regression,
            conditional_standard_deviation: conditional_variance.sqrt(),
            upper_offsets: upper_triangle_offsets(3)?,
        })
    }

    fn raw_moments(
        &self,
        rule: &GaussHermiteRule,
        total_evaluations: &mut usize,
        maximum_function_evaluations: usize,
        absolute_tolerance: f64,
    ) -> Result<Vec<f64>, EstimationError> {
        let mut accumulator = QuadratureAccumulator::new(packed_moment_count(3)?)?;
        for (&standard_normal, &weight) in rule.nodes.iter().zip(rule.weights.iter()) {
            if *total_evaluations >= maximum_function_evaluations {
                return Err(EstimationError::InvalidInput(format!(
                    "multinomial conditioned three-class quadrature exhausted its function-evaluation budget ({maximum_function_evaluations}) before convergence"
                )));
            }
            *total_evaluations += 1;

            let outer_mean = self.active_mean[self.outer_class];
            let outer_eta = outer_mean + self.outer_standard_deviation * standard_normal;
            let conditioned_mean = self.active_mean[self.conditioned_class]
                + self.conditional_regression * (outer_eta - outer_mean);
            let scalar_location = conditioned_mean - gam_linalg::utils::stable_softplus(outer_eta);
            let (selected_mean, selected_slope) =
                gam_solve::quadrature::logit_posterior_meanwith_deriv(
                    scalar_location,
                    self.conditional_standard_deviation,
                )
                .map_err(|error| {
                    EstimationError::InvalidInput(format!(
                        "conditioned three-class scalar logistic-normal evaluation failed: {error}"
                    ))
                })?;
            let outer_share = (-gam_linalg::utils::stable_softplus(-outer_eta)).exp();
            let reference_share = 1.0 - outer_share;
            let selected_second = selected_mean - selected_slope;
            let remainder_second = 1.0 - selected_mean - selected_slope;

            let mut means = [0.0_f64; 3];
            means[self.conditioned_class] = selected_mean;
            means[self.outer_class] = outer_share * (1.0 - selected_mean);
            means[2] = reference_share * (1.0 - selected_mean);

            let mut seconds = [[0.0_f64; 3]; 3];
            seconds[self.conditioned_class][self.conditioned_class] = selected_second;
            seconds[self.conditioned_class][self.outer_class] = outer_share * selected_slope;
            seconds[self.outer_class][self.conditioned_class] =
                seconds[self.conditioned_class][self.outer_class];
            seconds[self.conditioned_class][2] = reference_share * selected_slope;
            seconds[2][self.conditioned_class] = seconds[self.conditioned_class][2];
            seconds[self.outer_class][self.outer_class] =
                outer_share * outer_share * remainder_second;
            seconds[self.outer_class][2] = outer_share * reference_share * remainder_second;
            seconds[2][self.outer_class] = seconds[self.outer_class][2];
            seconds[2][2] = reference_share * reference_share * remainder_second;

            accumulator.add_weight(weight);
            for (class, mean) in means.into_iter().enumerate() {
                accumulator.add_moment(class, weight * mean);
            }
            let second_offset = 3;
            for row in 0..3 {
                for column in row..3 {
                    let packed = second_offset + self.upper_offsets[row] + column - row;
                    accumulator.add_moment(packed, weight * seconds[row][column]);
                }
            }
        }
        let (mut raw_moments, mass, absolute_weight_sum) = accumulator.finish();
        normalize_by_mass(
            &mut raw_moments,
            mass,
            absolute_weight_sum,
            absolute_tolerance,
            &format!(
                "conditioned three-class rule with {} nodes",
                rule.nodes.len()
            ),
        )?;
        Ok(raw_moments)
    }
}

fn point_mass_moments(active_mean: &[f64]) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let class_mean = Array1::from_vec(softmax_with_reference(active_mean)?);
    let k = class_mean.len();
    Ok(MultinomialPosteriorMoments {
        class_mean,
        class_covariance: Array2::zeros((k, k)),
        class_standard_deviation: Array1::zeros(k),
        latent_rank: 0,
        rule: MultinomialPosteriorRule::Exact,
        function_evaluations: 1,
        max_raw_moment_level_difference: 0.0,
        covariance_range_projection_bound: 0.0,
    })
}

struct ProjectedGaussian {
    /// `factor factor^T` is the retained active-logit covariance.
    factor: Array2<f64>,
    projection_bound: f64,
    /// `sqrt(lambda)` per retained direction, in the column order of `factor`.
    /// The softmax argument is `mu + F z`, so this is the scale at which
    /// direction `d` moves the integrand and therefore the only thing that
    /// decides how much one-dimensional resolution that direction needs.
    standard_deviations: Vec<f64>,
}

fn project_active_covariance(
    covariance: ArrayView2<'_, f64>,
    absolute_tolerance: f64,
) -> Result<ProjectedGaussian, EstimationError> {
    let m = covariance.nrows();
    let symmetric = (&covariance.to_owned() + &covariance.t().to_owned()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior covariance eigendecomposition failed: {error}"
        ))
    })?;
    let eigenvalue_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    let tolerance = covariance_roundoff_tolerance(eigenvalue_scale, m);
    let minimum_eigenvalue = eigenvalues
        .iter()
        .fold(f64::INFINITY, |minimum, &value| minimum.min(value));
    if minimum_eigenvalue < -tolerance {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior active-logit covariance is not positive semidefinite: minimum eigenvalue {minimum_eigenvalue:.6e} is below -{tolerance:.6e} (scale {eigenvalue_scale:.6e})"
        )));
    }

    let small_positive_trace: f64 = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 0.0 && *value <= tolerance)
        .sum();
    // For every softmax raw moment used here, the Euclidean gradient norm is
    // at most one.  Coupling the retained Gaussian with the full Gaussian gives
    // |E f(full)-E f(retained)| <= E||delta|| <= sqrt(tr(V_discarded)).
    let candidate_projection_bound = small_positive_trace.sqrt();
    let discard_small_positive = candidate_projection_bound <= absolute_tolerance;

    let retained: Vec<(usize, f64)> = eigenvalues
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value > 0.0 && (!discard_small_positive || *value > tolerance))
        .collect();
    let projection_bound = if discard_small_positive {
        candidate_projection_bound
    } else {
        0.0
    };
    let mut factor = Array2::<f64>::zeros((m, retained.len()));
    let mut standard_deviations = Vec::new();
    standard_deviations
        .try_reserve_exact(retained.len())
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial posterior could not allocate retained standard deviations: {error}"
            ))
        })?;
    for (output_column, (eigenvector_column, eigenvalue)) in retained.into_iter().enumerate() {
        let scale = eigenvalue.sqrt();
        standard_deviations.push(scale);
        for row in 0..m {
            factor[[row, output_column]] = eigenvectors[[row, eigenvector_column]] * scale;
        }
    }
    Ok(ProjectedGaussian {
        factor,
        projection_bound,
        standard_deviations,
    })
}

/// Everything about one prediction row that every rule shares.
struct RowIntegrand<'a> {
    active_mean: &'a [f64],
    projected: &'a ProjectedGaussian,
    upper_offsets: Vec<usize>,
    moment_count: usize,
    k: usize,
}

impl<'a> RowIntegrand<'a> {
    fn new(
        active_mean: &'a [f64],
        projected: &'a ProjectedGaussian,
    ) -> Result<Self, EstimationError> {
        let k = active_mean.len() + 1;
        Ok(Self {
            active_mean,
            projected,
            upper_offsets: upper_triangle_offsets(k)?,
            moment_count: packed_moment_count(k)?,
            k,
        })
    }

    /// Raw moments under the tensor-product Gauss-Hermite rule whose
    /// one-dimensional rule index is `orders[d]` in direction `d`.  Rule index
    /// `i` carries `2i - 1` nodes, and index 1 is the single node `z = 0`, so a
    /// direction left at 1 is evaluated at the posterior mean exactly.
    ///
    /// The cache is keyed by rule index rather than filled densely up to the
    /// highest one used: this path visits indices that DOUBLE, so a dense cache
    /// would build every rule in between, and building a rule is an
    /// eigendecomposition of a `(2i - 1)`-square Jacobi matrix.  Filling 1..=512
    /// to reach index 512 costs more than every quadrature evaluation in this
    /// module put together.
    fn tensor_moments(
        &self,
        rules: &mut BTreeMap<usize, GaussHermiteRule>,
        orders: &[usize],
        total_evaluations: &mut usize,
        maximum_function_evaluations: usize,
        absolute_tolerance: f64,
    ) -> Result<Vec<f64>, EstimationError> {
        for &order in orders {
            if !rules.contains_key(&order) {
                rules.insert(order, gauss_hermite_rule(order)?);
            }
        }
        let axes: Vec<&GaussHermiteRule> = orders.iter().map(|order| &rules[order]).collect();
        let mut workspace = QuadratureWorkspace::new(
            self.active_mean,
            self.projected,
            &[],
            &self.upper_offsets,
            self.moment_count,
            total_evaluations,
            maximum_function_evaluations,
        )?;
        workspace.stream_axes(0, &axes, 1.0)?;
        let (mut raw_moments, mass, absolute_weight_sum) = workspace.accumulator.finish();
        let node_counts: Vec<usize> = orders.iter().map(|order| 2 * order - 1).collect();
        normalize_by_mass(
            &mut raw_moments,
            mass,
            absolute_weight_sum,
            absolute_tolerance,
            &format!("tensor rule with node counts {node_counts:?}"),
        )?;
        Ok(raw_moments)
    }

    /// One-dimensional rule index that resolves direction `direction` on its
    /// own, with every other direction held at the posterior mean.
    ///
    /// This chooses only the SHAPE of the tensor rule.  The certificate is taken
    /// on the full rule, so an optimistic reading here cannot certify anything
    /// -- it can only make the certified rule cheaper or dearer.  There is no
    /// chosen constant: the index doubles until the directional integral of
    /// every raw moment stops moving by more than the caller's own per-moment
    /// tolerance.
    fn directional_rule_index(
        &self,
        rules: &mut BTreeMap<usize, GaussHermiteRule>,
        direction: usize,
        control: &MultinomialPosteriorIntegrationControl,
        total_evaluations: &mut usize,
    ) -> Result<usize, EstimationError> {
        let rank = self.projected.factor.ncols();
        let mut orders = vec![1usize; rank];
        let mut previous: Option<Vec<f64>> = None;
        let mut index = 1usize;
        loop {
            orders[direction] = index;
            let current = self.tensor_moments(
                rules,
                &orders,
                total_evaluations,
                control.maximum_function_evaluations,
                control.absolute_tolerance,
            )?;
            if let Some(previous_moments) = previous.as_ref() {
                let resolved =
                    current
                        .iter()
                        .zip(previous_moments.iter())
                        .all(|(new_value, old_value)| {
                            let tolerance = control.absolute_tolerance
                                + control.relative_tolerance * new_value.abs().max(old_value.abs());
                            (new_value - old_value).abs() <= tolerance
                        });
                if resolved {
                    return Ok(index);
                }
            }
            previous = Some(current);
            let doubled = index.checked_mul(2).ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial posterior directional rule index overflowed usize".to_string(),
                )
            })?;
            // No direction can usefully be sized past the point where the
            // tensor rule carrying that order in EVERY direction would already
            // exceed the evaluation budget: such an order can never be part of a
            // rule this caller is allowed to evaluate.  That bound comes from
            // the budget the caller supplied, not from a chosen ceiling.
            if tensor_node_count(&vec![doubled; rank])
                .map(|count| count > control.maximum_function_evaluations)
                .unwrap_or(true)
            {
                return Ok(index);
            }
            index = doubled;
        }
    }
}

/// Number of nodes in the tensor rule with these one-dimensional rule indices.
fn tensor_node_count(orders: &[usize]) -> Result<usize, EstimationError> {
    let mut count = 1usize;
    for &order in orders {
        let nodes = order
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial posterior tensor rule order overflowed usize".to_string(),
                )
            })?;
        count = count.checked_mul(nodes).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior tensor node count overflowed usize".to_string(),
            )
        })?;
    }
    Ok(count)
}

/// Divide accumulated moments by the rule's own total weight, after checking
/// that the rule integrates the constant function to one.
fn normalize_by_mass(
    raw_moments: &mut [f64],
    mass: f64,
    absolute_weight_sum: f64,
    absolute_tolerance: f64,
    context: &str,
) -> Result<(), EstimationError> {
    if !(mass.is_finite() && mass > 0.0 && absolute_weight_sum.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior {context} produced invalid total weight {mass} (absolute sum {absolute_weight_sum})"
        )));
    }
    let mass_error = (mass - 1.0).abs();
    let summation_envelope =
        SUMMATION_ROUNDOFF_MULTIPLIER * f64::EPSILON * absolute_weight_sum.max(1.0);
    if mass_error > absolute_tolerance + summation_envelope {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior {context} failed constant-function exactness: total weight {mass:.17e}, error {mass_error:.6e}, allowed {:.6e}",
            absolute_tolerance + summation_envelope
        )));
    }
    for value in raw_moments.iter_mut() {
        *value /= mass;
    }
    Ok(())
}

/// The sparse rule first, and the tensor rule for what it refuses.
///
/// The retained posterior has `rank = number of positive covariance eigenvalues`
/// directions, at most `K - 1`. A Smolyak grid buys its saving by giving a
/// direction high order only when every other direction is held at a single
/// node, which is the right trade when `rank` is large compared with the order
/// each direction needs -- and where that holds the sparse grid certifies at a
/// low level for a few hundred evaluations, which nothing here should make more
/// expensive. So it is tried first and kept wherever it certifies: every row
/// that predicts today predicts on the same rule, at the same cost, with the
/// same numbers.
///
/// What changes is the row it REFUSES. A logistic-normal softmax needs
/// one-dimensional order growing with a direction's own standard deviation --
/// `softmax` has a transition of width O(1) in the logit while the Gaussian
/// along that direction has width `sqrt(lambda_d)` -- so a wide posterior needs
/// order in EVERY wide direction at once, and that is the one thing the sparse
/// construction will not supply. Measured on a rank-two posterior with standard
/// deviations (1, 5): the sparse ladder reaches `1.99e-4` in 28033 evaluations
/// while the tensor product it is assembled FROM reaches `3.82e-9` in 16641
/// (#2612). Raising the level ceiling has been the answer three times and each
/// time bought a factor of three; the rule is what was wrong.
fn integrate_general(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let sparse_refusal = match integrate_isotropic_sparse(active_mean, projected, control, 0) {
        Ok(moments) => return Ok(moments),
        Err(refusal) => refusal,
    };

    let integrand = RowIntegrand::new(active_mean, projected)?;
    let rank = projected.factor.ncols();
    let mut rules = BTreeMap::<usize, GaussHermiteRule>::new();
    let mut total_evaluations = 0usize;

    let mut orders = vec![1usize; rank];
    for direction in 0..rank {
        orders[direction] = integrand.directional_rule_index(
            &mut rules,
            direction,
            control,
            &mut total_evaluations,
        )?;
    }

    let refined: Vec<usize> = orders.iter().map(|order| order.saturating_mul(2)).collect();
    let certifying_pair_cost =
        tensor_node_count(&orders)?.saturating_add(tensor_node_count(&refined)?);
    let remaining = control
        .maximum_function_evaluations
        .saturating_sub(total_evaluations);
    if certifying_pair_cost > remaining {
        // Neither rule fits: the sparse refusal is the one the caller can act
        // on, and it is returned unaltered rather than restated as a tensor
        // refusal for a rule that was never evaluated.
        return Err(EstimationError::InvalidInput(format!(
            "{sparse_refusal}; the tensor rule sized for this posterior would need {certifying_pair_cost} evaluations to certify, against {remaining} remaining"
        )));
    }

    integrate_anisotropic_tensor(&integrand, &mut rules, orders, control, total_evaluations)
}

/// Tensor-product Gauss-Hermite, certified by comparing against the rule with
/// every one-dimensional order doubled.  Doubling preserves the anisotropic
/// profile that `directional_rule_index` measured, so refinement never undoes
/// the direction sizing; the returned answer is always the finer of the pair.
fn integrate_anisotropic_tensor(
    integrand: &RowIntegrand<'_>,
    rules: &mut BTreeMap<usize, GaussHermiteRule>,
    mut orders: Vec<usize>,
    control: &MultinomialPosteriorIntegrationControl,
    mut total_evaluations: usize,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let rank = orders.len();
    let projection_bound = integrand.projected.projection_bound;
    let mut coarse = integrand.tensor_moments(
        rules,
        &orders,
        &mut total_evaluations,
        control.maximum_function_evaluations,
        control.absolute_tolerance,
    )?;
    loop {
        let refined: Vec<usize> = orders.iter().map(|order| order * 2).collect();
        let fine = integrand.tensor_moments(
            rules,
            &refined,
            &mut total_evaluations,
            control.maximum_function_evaluations,
            control.absolute_tolerance,
        )?;

        let mut certified = true;
        let mut maximum_difference = 0.0_f64;
        let mut deciding: Option<DecidingMoment> = None;
        for (index, (&new_value, &old_value)) in fine.iter().zip(coarse.iter()).enumerate() {
            let difference = (new_value - old_value).abs();
            maximum_difference = maximum_difference.max(difference);
            let tolerance = control.absolute_tolerance
                + control.relative_tolerance * new_value.abs().max(old_value.abs());
            let controlled_error = difference + projection_bound;
            if controlled_error > tolerance {
                certified = false;
            }
            let normalized = if tolerance > 0.0 {
                controlled_error / tolerance
            } else if controlled_error > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            let supersedes = match deciding.as_ref() {
                Some(current_worst) => normalized > current_worst.normalized_error,
                None => true,
            };
            if supersedes {
                deciding = Some(DecidingMoment {
                    index,
                    normalized_error: normalized,
                    difference,
                    tolerance,
                });
            }
        }

        let node_counts: Vec<usize> = refined.iter().map(|order| 2 * order - 1).collect();
        if certified {
            return moments_from_raw(
                fine,
                integrand.k,
                rank,
                MultinomialPosteriorRule::AnisotropicTensor(node_counts),
                total_evaluations,
                maximum_difference,
                projection_bound,
            );
        }

        orders = refined;
        coarse = fine;
        let next: Vec<usize> = orders.iter().map(|order| order * 2).collect();
        let next_cost = tensor_node_count(&next)?;
        if next_cost
            > control
                .maximum_function_evaluations
                .saturating_sub(total_evaluations)
        {
            let standard_deviations: Vec<String> = integrand
                .projected
                .standard_deviations
                .iter()
                .map(|value| format!("{value:.6e}"))
                .collect();
            let deciding_report = match deciding.as_ref() {
                Some(moment) => format!(
                    "worst raw moment {} (normalized error {:.6e} = (difference {:.6e} + projection bound {:.6e}) / tolerance {:.6e})",
                    moment.index,
                    moment.normalized_error,
                    moment.difference,
                    projection_bound,
                    moment.tolerance,
                ),
                None => "no raw moment was compared".to_string(),
            };
            return Err(EstimationError::InvalidInput(format!(
                "multinomial logistic-normal quadrature did not converge: tensor rule with node counts {node_counts:?} over posterior standard deviations [{}] left a raw-moment difference {maximum_difference:.6e}, and doubling it again would need {next_cost} of the {} evaluations still allowed; evaluations {total_evaluations}/{}; {deciding_report}",
                standard_deviations.join(", "),
                control
                    .maximum_function_evaluations
                    .saturating_sub(total_evaluations),
                control.maximum_function_evaluations,
            )));
        }
    }
}

fn integrate_isotropic_sparse(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    control: &MultinomialPosteriorIntegrationControl,
    initial_evaluations: usize,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let rank = projected.factor.ncols();
    let k = active_mean.len() + 1;
    let mut rules = Vec::<GaussHermiteRule>::new();
    let mut previous: Option<Vec<f64>> = None;
    let mut total_evaluations = initial_evaluations;
    let mut last_max_difference = f64::INFINITY;
    let mut last_max_normalized_error = f64::INFINITY;
    let mut last_deciding: Option<DecidingMoment> = None;

    let mut last_level_attempted = 0usize;
    for level in 0..=control.maximum_sparse_level {
        last_level_attempted = level;
        let required_rule_count = level.checked_add(1).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior sparse level overflowed usize".to_string(),
            )
        })?;
        while rules.len() < required_rule_count {
            let rule_index = rules.len() + 1;
            rules.push(gauss_hermite_rule(rule_index)?);
        }

        let evaluation = evaluate_smolyak_level(
            active_mean,
            projected,
            &rules,
            level,
            k,
            &mut total_evaluations,
            control.maximum_function_evaluations,
            control.absolute_tolerance,
        )?;
        let current = evaluation.raw_moments;

        if let Some(previous_moments) = previous.as_ref() {
            let mut certified = level >= control.minimum_sparse_level;
            let mut maximum_difference = 0.0_f64;
            let mut maximum_normalized_error = 0.0_f64;
            // The coordinate whose normalized error is the maximum -- i.e. the
            // one that actually refused -- carried alongside the quantities that
            // decided it (#2612).
            //
            // `maximum_difference` and `maximum_normalized_error` are maxima over
            // the SAME loop but not over the same coordinate: one is scaled by a
            // per-moment tolerance and the other is not, so their argmaxes differ
            // whenever the moments differ in magnitude. Reported side by side they
            // read as a pair describing one moment, and a refusal citing
            // `level difference 1.199663e-2, max normalized error 4.130034e5` --
            // seven orders apart -- invites the reading that the normalization
            // divides by something near zero, when in fact the two numbers simply
            // describe different raw moments. Naming the deciding coordinate and
            // printing ITS difference and ITS tolerance removes the ambiguity
            // instead of inviting the next reader to re-derive it.
            let mut deciding: Option<DecidingMoment> = None;
            for (index, (&new_value, &old_value)) in
                current.iter().zip(previous_moments.iter()).enumerate()
            {
                let difference = (new_value - old_value).abs();
                maximum_difference = maximum_difference.max(difference);
                let tolerance = control.absolute_tolerance
                    + control.relative_tolerance * new_value.abs().max(old_value.abs());
                let controlled_error = difference + projected.projection_bound;
                if controlled_error > tolerance {
                    certified = false;
                }
                // A zero tolerance cannot normalize an error, and the previous
                // `if tolerance > 0.0` guard SKIPPED such a coordinate entirely --
                // so a moment that failed certification could contribute nothing
                // to the reported error, and the refusal could understate the very
                // quantity it refused on. Infinity is the honest normalized error
                // when a nonzero error is measured against a zero tolerance, and
                // it is reported rather than dropped.
                let normalized = if tolerance > 0.0 {
                    controlled_error / tolerance
                } else if controlled_error > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                let supersedes = match deciding.as_ref() {
                    Some(current_worst) => normalized > current_worst.normalized_error,
                    None => true,
                };
                if supersedes {
                    deciding = Some(DecidingMoment {
                        index,
                        normalized_error: normalized,
                        difference,
                        tolerance,
                    });
                }
                maximum_normalized_error = maximum_normalized_error.max(normalized);
            }
            last_max_difference = maximum_difference;
            last_max_normalized_error = maximum_normalized_error;
            last_deciding = deciding;

            if certified {
                return moments_from_raw(
                    current,
                    k,
                    rank,
                    MultinomialPosteriorRule::IsotropicSparse(level),
                    total_evaluations,
                    maximum_difference,
                    projected.projection_bound,
                );
            }
        }
        previous = Some(current);
    }

    // Report the level REACHED, not the configured ceiling.
    //
    // With the ceiling no longer a tuned number, the ceiling is not the fact a
    // reader needs; the depth actually attained before the budget ran out is.
    // The old message printed `control.maximum_sparse_level`, which said what
    // the cap was rather than what the integrand did -- and on a run that
    // stopped at 1.4% of its evaluation budget those are different stories.
    let deciding_report = match last_deciding.as_ref() {
        Some(moment) => format!(
            "worst raw moment {} (normalized error {:.6e} = (difference {:.6e} + projection bound {:.6e}) / tolerance {:.6e})",
            moment.index,
            moment.normalized_error,
            moment.difference,
            projected.projection_bound,
            moment.tolerance,
        ),
        // Reached only when no level after the first produced a comparison, so
        // there is no per-coordinate verdict to name. Said explicitly, because
        // an absent comparison and a passing one must not read alike (#2612).
        None => "no two levels were compared, so no coordinate refused".to_string(),
    };
    Err(EstimationError::InvalidInput(format!(
        "multinomial logistic-normal quadrature did not converge: reached Smolyak level {last_level_attempted} and exhausted the evaluation budget; final max raw-moment level difference {last_max_difference:.6e}, max normalized error {last_max_normalized_error:.6e}, projection bound {:.6e}, evaluations {total_evaluations}/{}; {deciding_report}",
        projected.projection_bound, control.maximum_function_evaluations
    )))
}

/// The raw moment whose normalized error is the maximum, with the quantities
/// that produced it (#2612).
///
/// Exists so the refusal names WHICH moment refused and against WHAT. The
/// aggregate `max normalized error` and `max level difference` are maxima over
/// different coordinates, so neither one alone identifies the failure and the
/// pair actively misleads.
#[derive(Clone, Copy, Debug)]
struct DecidingMoment {
    index: usize,
    normalized_error: f64,
    difference: f64,
    tolerance: f64,
}

struct SmolyakEvaluation {
    raw_moments: Vec<f64>,
}

fn evaluate_smolyak_level(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    rules: &[GaussHermiteRule],
    level: usize,
    k: usize,
    total_evaluations: &mut usize,
    maximum_function_evaluations: usize,
    absolute_tolerance: f64,
) -> Result<SmolyakEvaluation, EstimationError> {
    let rank = projected.factor.ncols();
    let q = rank.checked_add(level).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior Smolyak index overflowed usize".to_string(),
        )
    })?;
    let lower_total = q.saturating_sub(rank.saturating_sub(1)).max(rank);
    let moment_count = packed_moment_count(k)?;
    let upper_offsets = upper_triangle_offsets(k)?;
    let mut workspace = QuadratureWorkspace::new(
        active_mean,
        projected,
        rules,
        &upper_offsets,
        moment_count,
        total_evaluations,
        maximum_function_evaluations,
    )?;
    let mut indices = vec![1usize; rank];

    for total in lower_total..=q {
        let alternating_power = q - total;
        let mut coefficient = binomial_as_f64(rank - 1, alternating_power)?;
        if alternating_power % 2 == 1 {
            coefficient = -coefficient;
        }
        workspace.stream_compositions(0, total, &mut indices, coefficient)?;
    }

    let (mut raw_moments, mass, absolute_weight_sum) = workspace.accumulator.finish();
    normalize_by_mass(
        &mut raw_moments,
        mass,
        absolute_weight_sum,
        absolute_tolerance,
        &format!("Smolyak level {level}"),
    )?;
    Ok(SmolyakEvaluation { raw_moments })
}

fn packed_moment_count(k: usize) -> Result<usize, EstimationError> {
    let triangular = k
        .checked_add(1)
        .and_then(|next| k.checked_mul(next))
        .map(|product| product / 2)
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior moment dimension overflowed usize".to_string(),
            )
        })?;
    k.checked_add(triangular).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior packed moment count overflowed usize".to_string(),
        )
    })
}

fn upper_triangle_offsets(k: usize) -> Result<Vec<usize>, EstimationError> {
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(k).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate upper-triangle offsets: {error}"
        ))
    })?;
    let mut cursor = 0usize;
    for row in 0..k {
        offsets.push(cursor);
        cursor = cursor.checked_add(k - row).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior upper-triangle offset overflowed usize".to_string(),
            )
        })?;
    }
    Ok(offsets)
}

fn zeroed_vec(length: usize, label: &str) -> Result<Vec<f64>, EstimationError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate {label} (length {length}): {error}"
        ))
    })?;
    values.resize(length, 0.0);
    Ok(values)
}

struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let combined = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - combined) + value;
        } else {
            self.correction += (value - combined) + self.sum;
        }
        self.sum = combined;
    }

    fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

struct QuadratureAccumulator {
    sums: Vec<f64>,
    corrections: Vec<f64>,
    mass: CompensatedSum,
    absolute_weight_sum: f64,
}

impl QuadratureAccumulator {
    fn new(moment_count: usize) -> Result<Self, EstimationError> {
        Ok(Self {
            sums: zeroed_vec(moment_count, "quadrature sums")?,
            corrections: zeroed_vec(moment_count, "quadrature corrections")?,
            mass: CompensatedSum::new(),
            absolute_weight_sum: 0.0,
        })
    }

    fn add_moment(&mut self, index: usize, value: f64) {
        let combined = self.sums[index] + value;
        if self.sums[index].abs() >= value.abs() {
            self.corrections[index] += (self.sums[index] - combined) + value;
        } else {
            self.corrections[index] += (value - combined) + self.sums[index];
        }
        self.sums[index] = combined;
    }

    fn add_weight(&mut self, weight: f64) {
        self.mass.add(weight);
        self.absolute_weight_sum += weight.abs();
    }

    fn finish(mut self) -> (Vec<f64>, f64, f64) {
        for (sum, correction) in self.sums.iter_mut().zip(self.corrections.iter()) {
            *sum += *correction;
        }
        (self.sums, self.mass.value(), self.absolute_weight_sum)
    }
}

struct QuadratureWorkspace<'a, 'b> {
    active_mean: &'a [f64],
    projected: &'a ProjectedGaussian,
    rules: &'a [GaussHermiteRule],
    upper_offsets: &'a [usize],
    z: Vec<f64>,
    active_eta: Vec<f64>,
    probabilities: Vec<f64>,
    accumulator: QuadratureAccumulator,
    total_evaluations: &'b mut usize,
    maximum_function_evaluations: usize,
}

impl<'a, 'b> QuadratureWorkspace<'a, 'b> {
    fn new(
        active_mean: &'a [f64],
        projected: &'a ProjectedGaussian,
        rules: &'a [GaussHermiteRule],
        upper_offsets: &'a [usize],
        moment_count: usize,
        total_evaluations: &'b mut usize,
        maximum_function_evaluations: usize,
    ) -> Result<Self, EstimationError> {
        let rank = projected.factor.ncols();
        let m = active_mean.len();
        Ok(Self {
            active_mean,
            projected,
            rules,
            upper_offsets,
            z: zeroed_vec(rank, "standard-normal quadrature coordinate")?,
            active_eta: zeroed_vec(m, "active-logit quadrature buffer")?,
            probabilities: zeroed_vec(m + 1, "softmax quadrature buffer")?,
            accumulator: QuadratureAccumulator::new(moment_count)?,
            total_evaluations,
            maximum_function_evaluations,
        })
    }

    fn stream_compositions(
        &mut self,
        position: usize,
        remaining: usize,
        indices: &mut [usize],
        coefficient: f64,
    ) -> Result<(), EstimationError> {
        let dimensions_left = indices.len() - position;
        if dimensions_left == 1 {
            if remaining == 0 {
                return Ok(());
            }
            indices[position] = remaining;
            return self.stream_tensor(0, indices, coefficient);
        }
        let maximum_here = remaining.saturating_sub(dimensions_left - 1);
        for index in 1..=maximum_here {
            indices[position] = index;
            self.stream_compositions(position + 1, remaining - index, indices, coefficient)?;
        }
        Ok(())
    }

    fn stream_tensor(
        &mut self,
        axis: usize,
        indices: &[usize],
        weight: f64,
    ) -> Result<(), EstimationError> {
        if axis == indices.len() {
            return self.accumulate_node(weight);
        }
        let rule_index = indices[axis] - 1;
        let node_count = self.rules[rule_index].nodes.len();
        for node_index in 0..node_count {
            let node = self.rules[rule_index].nodes[node_index];
            let node_weight = self.rules[rule_index].weights[node_index];
            self.z[axis] = node;
            self.stream_tensor(axis + 1, indices, weight * node_weight)?;
        }
        Ok(())
    }

    /// Tensor product over one explicitly chosen rule per direction.
    ///
    /// `stream_tensor` above resolves each axis through `rules[index - 1]`,
    /// which forces the rule cache to be dense; the tensor path visits indices
    /// that double, so it resolves its axes once and hands them over directly.
    fn stream_axes(
        &mut self,
        axis: usize,
        axes: &[&GaussHermiteRule],
        weight: f64,
    ) -> Result<(), EstimationError> {
        if axis == axes.len() {
            return self.accumulate_node(weight);
        }
        let node_count = axes[axis].nodes.len();
        for node_index in 0..node_count {
            let node = axes[axis].nodes[node_index];
            let node_weight = axes[axis].weights[node_index];
            self.z[axis] = node;
            self.stream_axes(axis + 1, axes, weight * node_weight)?;
        }
        Ok(())
    }

    fn accumulate_node(&mut self, weight: f64) -> Result<(), EstimationError> {
        if *self.total_evaluations >= self.maximum_function_evaluations {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial logistic-normal quadrature exhausted its function-evaluation budget ({}) before convergence",
                self.maximum_function_evaluations
            )));
        }
        *self.total_evaluations += 1;

        for row in 0..self.active_mean.len() {
            let mut value = self.active_mean[row];
            for column in 0..self.z.len() {
                value += self.projected.factor[[row, column]] * self.z[column];
            }
            self.active_eta[row] = value;
        }
        softmax_with_reference_into(&self.active_eta, &mut self.probabilities)?;

        let k = self.probabilities.len();
        self.accumulator.add_weight(weight);
        for class in 0..k {
            self.accumulator
                .add_moment(class, weight * self.probabilities[class]);
        }
        let second_offset = k;
        for row in 0..k {
            for column in row..k {
                let packed = second_offset + self.upper_offsets[row] + column - row;
                self.accumulator.add_moment(
                    packed,
                    weight * self.probabilities[row] * self.probabilities[column],
                );
            }
        }
        Ok(())
    }
}

struct GaussHermiteRule {
    /// Nodes already transformed to standard-normal coordinates.
    nodes: Vec<f64>,
    /// Normalized standard-normal expectation weights (sum to one).
    weights: Vec<f64>,
}

fn gauss_hermite_rule(index: usize) -> Result<GaussHermiteRule, EstimationError> {
    let node_count = index
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior Gauss-Hermite order overflowed usize".to_string(),
            )
        })?;
    let physicists = physicists_gauss_hermite_rule(node_count).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior Gauss-Hermite rule {node_count} construction failed: {error}"
        ))
    })?;
    let nodes = physicists
        .nodes
        .into_iter()
        .map(|node| std::f64::consts::SQRT_2 * node)
        .collect::<Vec<_>>();
    let mut weights = physicists.weights;
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior Gauss-Hermite rule {node_count} has invalid weight sum {weight_sum}"
        )));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    Ok(GaussHermiteRule { nodes, weights })
}

fn binomial_as_f64(n: usize, k: usize) -> Result<f64, EstimationError> {
    if k > n {
        return Ok(0.0);
    }
    let k = k.min(n - k);
    let mut value = 1.0_f64;
    for step in 1..=k {
        value *= (n - k + step) as f64 / step as f64;
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior Smolyak binomial coefficient C({n},{k}) overflowed f64"
            )));
        }
    }
    Ok(value)
}

/// Reference-coded softmax of one row of ACTIVE logits, with the reference
/// class's `η = 0` appended last — the plug-in probability `softmax(η)` at a
/// single point, with no posterior integration.
///
/// Shared with `multinomial::predict_multinomial_formula_plugin` rather than
/// re-derived there: the max-shift, the implicit reference logit and the class
/// ordering are all conventions this module owns, and a second copy of them is
/// a second place for the reference class to move.
pub(crate) fn softmax_with_reference(active_eta: &[f64]) -> Result<Vec<f64>, EstimationError> {
    let mut probabilities = zeroed_vec(active_eta.len() + 1, "softmax result")?;
    softmax_with_reference_into(active_eta, &mut probabilities)?;
    Ok(probabilities)
}

fn softmax_with_reference_into(
    active_eta: &[f64],
    probabilities: &mut [f64],
) -> Result<(), EstimationError> {
    if probabilities.len() != active_eta.len() + 1 {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior softmax buffer length {} does not equal active-logit length {} + 1",
            probabilities.len(),
            active_eta.len()
        )));
    }
    let maximum = active_eta.iter().copied().fold(0.0_f64, f64::max);
    let reference = probabilities.len() - 1;
    let mut denominator = (-maximum).exp();
    probabilities[reference] = denominator;
    for (class, &eta) in active_eta.iter().enumerate() {
        let numerator = (eta - maximum).exp();
        probabilities[class] = numerator;
        denominator += numerator;
    }
    if !(denominator.is_finite() && denominator > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior softmax produced invalid denominator {denominator}"
        )));
    }
    for probability in probabilities {
        *probability /= denominator;
    }
    Ok(())
}

fn moments_from_raw(
    raw_moments: Vec<f64>,
    k: usize,
    latent_rank: usize,
    rule: MultinomialPosteriorRule,
    function_evaluations: usize,
    max_level_difference: f64,
    projection_bound: f64,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let upper_offsets = upper_triangle_offsets(k)?;
    let raw_error = max_level_difference + projection_bound;
    let covariance_error = 3.0 * raw_error + raw_error * raw_error;

    let mut means = raw_moments[..k].to_vec();
    for (class, mean) in means.iter_mut().enumerate() {
        if *mean < -raw_error || *mean > 1.0 + raw_error || !mean.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integrated mean for class {class} is outside its certified probability envelope: {mean} (raw error {raw_error:.6e})"
            )));
        }
        *mean = mean.clamp(0.0, 1.0);
    }
    let mean_sum: f64 = means.iter().sum();
    if !(mean_sum.is_finite() && mean_sum > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integrated class means have invalid sum {mean_sum}"
        )));
    }
    let simplex_error = (mean_sum - 1.0).abs();
    if simplex_error > (k as f64) * raw_error + covariance_roundoff_tolerance(1.0, k) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integrated class means violate the simplex: sum {mean_sum:.17e}, error {simplex_error:.6e}, raw moment error {raw_error:.6e}"
        )));
    }
    for mean in &mut means {
        *mean /= mean_sum;
    }

    let second_offset = k;
    let mut covariance = Array2::<f64>::zeros((k, k));
    for row in 0..k {
        for column in row..k {
            let packed = second_offset + upper_offsets[row] + column - row;
            let value = raw_moments[packed] - means[row] * means[column];
            covariance[[row, column]] = value;
            covariance[[column, row]] = value;
        }
    }
    covariance = project_covariance_to_simplex_tangent(&covariance);
    covariance = remove_covariance_roundoff(covariance, covariance_error)?;
    covariance = project_covariance_to_simplex_tangent(&covariance);

    let mut standard_deviation = Array1::<f64>::zeros(k);
    for class in 0..k {
        let variance = covariance[[class, class]];
        if variance < -covariance_error || !variance.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior variance for class {class} is invalid: {variance:.6e} (covariance error envelope {covariance_error:.6e})"
            )));
        }
        standard_deviation[class] = variance.max(0.0).sqrt();
    }

    Ok(MultinomialPosteriorMoments {
        class_mean: Array1::from_vec(means),
        class_covariance: covariance,
        class_standard_deviation: standard_deviation,
        latent_rank,
        rule,
        function_evaluations,
        max_raw_moment_level_difference: max_level_difference,
        covariance_range_projection_bound: projection_bound,
    })
}

fn project_covariance_to_simplex_tangent(covariance: &Array2<f64>) -> Array2<f64> {
    let k = covariance.nrows();
    let inverse_k = 1.0 / k as f64;
    let row_means: Vec<f64> = (0..k)
        .map(|row| covariance.row(row).sum() * inverse_k)
        .collect();
    let column_means: Vec<f64> = (0..k)
        .map(|column| covariance.column(column).sum() * inverse_k)
        .collect();
    let grand_mean = row_means.iter().sum::<f64>() * inverse_k;
    Array2::from_shape_fn((k, k), |(row, column)| {
        covariance[[row, column]] - row_means[row] - column_means[column] + grand_mean
    })
}

fn remove_covariance_roundoff(
    covariance: Array2<f64>,
    integration_error: f64,
) -> Result<Array2<f64>, EstimationError> {
    let symmetric = (&covariance + &covariance.t().to_owned()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial probability covariance eigendecomposition failed: {error}"
        ))
    })?;
    let scale = eigenvalues
        .iter()
        .fold(0.0_f64, |maximum, &value| maximum.max(value.abs()));
    let allowed_negative =
        integration_error + covariance_roundoff_tolerance(scale, covariance.nrows());
    let minimum = eigenvalues
        .iter()
        .fold(f64::INFINITY, |value, &candidate| value.min(candidate));
    if minimum < -allowed_negative {
        let negative_limit = -allowed_negative;
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior probability covariance is indefinite beyond the integration error: min eigenvalue {minimum:.6e}, allowed {negative_limit:.6e}"
        )));
    }
    let mut scaled_eigenvectors = eigenvectors.clone();
    for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
        let scale = eigenvalue.max(0.0);
        scaled_eigenvectors
            .column_mut(column)
            .mapv_inplace(|value| value * scale);
    }
    let reconstructed = scaled_eigenvectors.dot(&eigenvectors.t());
    Ok((&reconstructed + &reconstructed.t().to_owned()) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(absolute_tolerance: f64) -> MultinomialPosteriorIntegrationControl {
        MultinomialPosteriorIntegrationControl {
            absolute_tolerance,
            relative_tolerance: absolute_tolerance,
            minimum_sparse_level: 2,
            maximum_sparse_level: 8,
            maximum_function_evaluations: 2_000_000,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    /// A refusal must name WHICH raw moment refused and against WHAT (#2612).
    ///
    /// The aggregate pair the message used to carry -- `max raw-moment level
    /// difference` and `max normalized error` -- are maxima over DIFFERENT
    /// coordinates, because one is divided by a per-moment tolerance and the
    /// other is not. Printed side by side they read as one moment's story, and
    /// the observed `1.199663e-2` beside `4.130034e5` invited the conclusion
    /// that the normalization divides by something near zero. It does not; they
    /// were simply different moments.
    #[test]
    fn a_refusal_names_the_moment_that_decided_it_2612() {
        let active_mean = Array1::from_vec(vec![0.7, -1.3, 2.1]);
        let active_covariance =
            Array2::from_shape_vec((3, 3), vec![2.5, 0.4, -0.3, 0.4, 3.1, 0.6, -0.3, 0.6, 2.2])
                .expect("covariance shape");
        // Two levels and a tolerance no sparse rule can reach, so the loop
        // exhausts its levels and falls through to the refusal cheaply.
        let control = MultinomialPosteriorIntegrationControl {
            absolute_tolerance: 1.0e-300,
            relative_tolerance: 1.0e-300,
            minimum_sparse_level: 2,
            maximum_sparse_level: 2,
            maximum_function_evaluations: 2_000_000,
        };
        let error = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control,
        )
        .expect_err("a 1e-300 tolerance at level 2 cannot certify");
        let message = error.to_string();
        assert!(
            message.contains("worst raw moment"),
            "the refusal must identify the deciding coordinate: {message}"
        );
        assert!(
            message.contains("/ tolerance"),
            "the refusal must show the normalized error's denominator, so the ratio \
             can be checked rather than trusted: {message}"
        );
    }

    /// The normalized error must not silently drop the coordinate that refused.
    ///
    /// `tolerance = absolute_tolerance + relative_tolerance * max(|new|, |old|)`
    /// is ZERO whenever `absolute_tolerance` is zero and a raw moment is exactly
    /// zero at both levels. The old `if tolerance > 0.0` guard skipped exactly
    /// those coordinates, so a moment could set `certified = false` and then
    /// contribute nothing to the error the refusal reports. A zero absolute
    /// tolerance is admissible -- `validate` only requires that ONE of the two
    /// be positive -- so this is a reachable state, not a defensive branch.
    #[test]
    fn a_zero_absolute_tolerance_is_admissible_so_the_zero_denominator_is_reachable_2612() {
        let control = MultinomialPosteriorIntegrationControl {
            absolute_tolerance: 0.0,
            relative_tolerance: 1.0e-8,
            minimum_sparse_level: 2,
            maximum_sparse_level: 4,
            maximum_function_evaluations: 2_000_000,
        };
        control
            .validate()
            .expect("a zero absolute tolerance with a positive relative one is admissible");
    }

    #[test]
    fn binary_reduction_matches_controlled_logistic_normal_identity() {
        let active_mean = Array1::from_vec(vec![1.1]);
        let active_covariance = Array2::from_shape_vec((1, 1), vec![0.64]).unwrap();
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-10),
        )
        .expect("binary posterior moments");
        let (expected_mean, expected_slope) =
            gam_solve::quadrature::logit_posterior_meanwith_deriv(1.1, 0.8).unwrap();
        let expected_variance = expected_mean - expected_slope - expected_mean * expected_mean;

        assert_close(result.class_mean[0], expected_mean, 2.0e-14, "binary mean");
        assert_close(
            result.class_mean[1],
            1.0 - expected_mean,
            2.0e-14,
            "reference mean",
        );
        assert_close(
            result.class_covariance[[0, 0]],
            expected_variance,
            2.0e-14,
            "binary variance",
        );
        assert_close(
            result.class_covariance[[0, 1]],
            -expected_variance,
            2.0e-14,
            "binary covariance",
        );
        assert_eq!(result.latent_rank, 1);
        assert_eq!(result.rule, MultinomialPosteriorRule::Exact);
    }

    #[test]
    fn zero_covariance_is_exact_softmax_point_mass() {
        let active_mean = Array1::from_vec(vec![0.7, -0.4]);
        let active_covariance = Array2::<f64>::zeros((2, 2));
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-10),
        )
        .expect("point-mass posterior moments");
        let expected = softmax_with_reference(active_mean.as_slice().unwrap()).unwrap();
        for class in 0..3 {
            assert_close(
                result.class_mean[class],
                expected[class],
                1.0e-15,
                "point mean",
            );
            assert_eq!(result.class_standard_deviation[class], 0.0);
            for other in 0..3 {
                assert_eq!(result.class_covariance[[class, other]], 0.0);
            }
        }
        assert_eq!(result.latent_rank, 0);
        assert_eq!(result.rule, MultinomialPosteriorRule::Exact);
    }

    #[test]
    fn exchangeable_full_logits_require_cross_covariance_and_integrate_to_uniform() {
        // If full logits gamma_c are iid N(0,s^2), reference coding gives
        // eta_a=gamma_a-gamma_ref, hence diag(V)=2s^2 and offdiag(V)=s^2.
        // Exchangeability makes E[p_c]=1/3 exactly.  Dropping the off-diagonal
        // covariance destroys that identity for the reference class.
        let variance = 0.7;
        let active_mean = Array1::zeros(2);
        let active_covariance = Array2::from_shape_vec(
            (2, 2),
            vec![2.0 * variance, variance, variance, 2.0 * variance],
        )
        .unwrap();
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(2.0e-7),
        )
        .expect("exchangeable posterior moments");

        for class in 0..3 {
            assert_close(result.class_mean[class], 1.0 / 3.0, 8.0e-7, "uniform mean");
        }
        for class in 1..3 {
            assert_close(
                result.class_covariance[[class, class]],
                result.class_covariance[[0, 0]],
                2.0e-6,
                "exchangeable variance",
            );
        }
        for row in 0..3 {
            assert_close(
                result.class_covariance.row(row).sum(),
                0.0,
                2.0e-12,
                "simplex covariance row sum",
            );
        }
        assert_eq!(result.latent_rank, 2);
        assert_ne!(result.rule, MultinomialPosteriorRule::Exact);
    }

    #[test]
    fn rank_one_general_case_matches_independent_one_dimensional_gh_oracle() {
        let active_mean = Array1::from_vec(vec![0.45, -0.7]);
        let loading = [0.8_f64, -0.35_f64];
        let active_covariance =
            Array2::from_shape_fn((2, 2), |(row, column)| loading[row] * loading[column]);
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(5.0e-8),
        )
        .expect("rank-one posterior moments");
        assert_eq!(result.latent_rank, 1);

        // Independent high-order one-dimensional GH evaluation of the exact
        // rank-one representation eta=mu+loading*Z.
        let oracle_rule = gauss_hermite_rule(21).unwrap(); // 41 nodes
        let mut oracle_mean = [0.0_f64; 3];
        let mut oracle_second = [[0.0_f64; 3]; 3];
        for (&z, &weight) in oracle_rule.nodes.iter().zip(oracle_rule.weights.iter()) {
            let eta = [
                active_mean[0] + loading[0] * z,
                active_mean[1] + loading[1] * z,
            ];
            let probability = softmax_with_reference(&eta).unwrap();
            for row in 0..3 {
                oracle_mean[row] += weight * probability[row];
                for column in 0..3 {
                    oracle_second[row][column] += weight * probability[row] * probability[column];
                }
            }
        }
        for row in 0..3 {
            assert_close(
                result.class_mean[row],
                oracle_mean[row],
                3.0e-7,
                "rank-one mean",
            );
            for column in 0..3 {
                let oracle_covariance =
                    oracle_second[row][column] - oracle_mean[row] * oracle_mean[column];
                assert_close(
                    result.class_covariance[[row, column]],
                    oracle_covariance,
                    8.0e-7,
                    "rank-one covariance",
                );
            }
        }
        let allowed = 5.0e-8
            + 5.0e-8
                * result
                    .class_mean
                    .iter()
                    .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
        assert!(
            result.max_raw_moment_level_difference + result.covariance_range_projection_bound
                <= allowed * 1.01,
            "returned result must carry the level-difference certificate"
        );
    }

    #[test]
    fn insufficient_sparse_level_is_a_typed_error_not_a_plugin_result() {
        let active_mean = Array1::from_vec(vec![1.2, -0.8]);
        let active_covariance = Array2::from_shape_vec((2, 2), vec![2.0, 0.9, 0.9, 1.5]).unwrap();
        let strict_control = MultinomialPosteriorIntegrationControl {
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
            minimum_sparse_level: 1,
            maximum_sparse_level: 1,
            maximum_function_evaluations: 100_000,
        };
        // Aimed at the sparse rule ITSELF rather than at the entry point:
        // the entry point now dispatches this row to the tensor rule, which
        // does certify it, so asserting a refusal there would be asserting the
        // dispatch rather than the sparse rule's own level bound (#2612).
        let symmetric = symmetrized_covariance(active_covariance.view());
        let projected =
            project_active_covariance(symmetric.view(), strict_control.absolute_tolerance)
                .expect("project the active covariance");
        let error = integrate_isotropic_sparse(
            active_mean.as_slice().expect("contiguous active mean"),
            &projected,
            &strict_control,
            0,
        )
        .expect_err("one sparse refinement cannot certify this nonlinear integral");
        assert!(
            error.to_string().contains("did not converge"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn materially_indefinite_active_covariance_is_rejected() {
        let active_mean = Array1::from_vec(vec![0.0, 0.0]);
        let active_covariance = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let error = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-7),
        )
        .expect_err("indefinite covariance must fail");
        assert!(error.to_string().contains("not positive semidefinite"));
    }

    /// Rank-two active-logit covariance with eigenvalues `(1, 25)` in a basis
    /// rotated off the coordinate axes: posterior standard deviations `(1, 5)`.
    /// Built from its spectrum rather than from literals so the widths the test
    /// is about are visible in the source.
    fn wide_two_direction_covariance() -> Array2<f64> {
        let eigenvalues = [1.0_f64, 25.0_f64];
        let (sine, cosine) = 0.6_f64.sin_cos();
        let basis = [[cosine, -sine], [sine, cosine]];
        Array2::from_shape_fn((2, 2), |(row, column)| {
            (0..2)
                .map(|index| basis[row][index] * eigenvalues[index] * basis[column][index])
                .sum()
        })
    }

    /// A WIDE two-direction posterior: exact conditioning removes one Gaussian
    /// direction before quadrature, because the sparse grid cannot reach the
    /// simultaneous order the unreduced integrand needs (#2612).
    ///
    /// Measured at `origin/main` on this exact covariance, the isotropic sparse
    /// ladder's error against a converged tensor oracle falls only algebraically
    /// -- level 10 to level 16 buys a factor 3.3 for 5.6x the evaluations, from
    /// `6.61e-4` to `1.99e-4` -- while the tensor product reaches `3.82e-9` at
    /// 129 nodes per direction for 16641 evaluations, fewer than the 28033 the
    /// sparse ladder spends to reach `1.99e-4`.
    ///
    /// The retained posterior here has rank two. A Smolyak grid of level `L`
    /// over `r` directions admits only the tensor sub-rules whose
    /// one-dimensional indices sum to at most `r + L`, so the only way it gives
    /// one direction order `2L + 1` is by giving every other direction a single
    /// node. Its BALANCED sub-rule -- the one that resolves both directions at
    /// once -- carries roughly order `L` in each. The logistic-normal softmax
    /// needs one-dimensional order growing with a direction's own standard
    /// deviation, because the softmax transition has width O(1) in the logit
    /// while the Gaussian along that direction has width `sqrt(lambda_d)`, so a
    /// posterior this wide needs high order in BOTH directions simultaneously
    /// and the sparse grid is the one construction that refuses to supply it.
    ///
    /// The test asserts both halves, so it cannot pass by accident:
    ///   * the isotropic sparse rule REFUSES this row at its level ceiling, and
    ///   * the conditioned entry point certifies every raw moment against an
    ///     independent high-order tensor Gauss-Hermite oracle.
    #[test]
    fn a_wide_two_direction_posterior_is_reduced_before_quadrature_2612() {
        let active_mean = Array1::from_vec(vec![2.0, -1.0]);
        let active_covariance = wide_two_direction_covariance();
        let control = MultinomialPosteriorIntegrationControl::default();

        // Half one: the sparse rule, on its own, cannot certify this row.
        let symmetric = symmetrized_covariance(active_covariance.view());
        let projected = project_active_covariance(symmetric.view(), control.absolute_tolerance)
            .expect("project the active covariance");
        assert_eq!(projected.factor.ncols(), 2, "this row must retain rank two");
        let sparse_error = integrate_isotropic_sparse(
            active_mean.as_slice().expect("contiguous active mean"),
            &projected,
            &control,
            0,
        )
        .expect_err("the isotropic sparse rule must not certify this wide posterior");
        assert!(
            sparse_error.to_string().contains("did not converge"),
            "unexpected sparse-rule error: {sparse_error}"
        );

        // Half two: the shipped entry point certifies it after exact
        // Rao-Blackwellization to one dimension.
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control,
        )
        .expect("the wide posterior must be integrable");
        assert!(matches!(
            result.rule,
            MultinomialPosteriorRule::ConditionedThreeClass(_)
        ));

        // And the certified answer is the right one: an independent tensor
        // Gauss-Hermite evaluation at an order nothing above chose.
        // 301 nodes per direction: at this covariance the 241-node and 301-node
        // tensor rules agree to 3.02e-12, so the oracle is converged well below
        // what is being asserted.
        let oracle_rule = gauss_hermite_rule(151).expect("oracle rule");
        let mut oracle_mean = vec![0.0_f64; 3];
        let mut oracle_second = [[0.0_f64; 3]; 3];
        let mut mass = 0.0_f64;
        for (&first_node, &first_weight) in oracle_rule.nodes.iter().zip(oracle_rule.weights.iter())
        {
            for (&second_node, &second_weight) in
                oracle_rule.nodes.iter().zip(oracle_rule.weights.iter())
            {
                let weight = first_weight * second_weight;
                mass += weight;
                let eta = [
                    active_mean[0]
                        + projected.factor[[0, 0]] * first_node
                        + projected.factor[[0, 1]] * second_node,
                    active_mean[1]
                        + projected.factor[[1, 0]] * first_node
                        + projected.factor[[1, 1]] * second_node,
                ];
                let probability = softmax_with_reference(&eta).expect("oracle softmax");
                for class in 0..3 {
                    oracle_mean[class] += weight * probability[class];
                    for other in 0..3 {
                        oracle_second[class][other] +=
                            weight * probability[class] * probability[other];
                    }
                }
            }
        }
        for value in &mut oracle_mean {
            *value /= mass;
        }
        for row in &mut oracle_second {
            for value in row {
                *value /= mass;
            }
        }
        for class in 0..3 {
            assert_close(
                result.class_mean[class],
                oracle_mean[class],
                1.0e-7,
                "wide posterior class mean",
            );
            for other in 0..3 {
                let oracle_covariance =
                    oracle_second[class][other] - oracle_mean[class] * oracle_mean[other];
                assert_close(
                    result.class_covariance[[class, other]],
                    oracle_covariance,
                    3.0e-7,
                    "wide posterior class covariance",
                );
            }
        }
    }

    /// The Gauss-Hermite ladder is a property of the requested orders, not of
    /// a prediction row.  Reusing it across rows must avoid every repeated
    /// Golub-Welsch construction without changing a posterior bit.
    #[test]
    fn conditioned_three_class_rule_ladder_is_reused_across_rows_2612() {
        let active_mean = Array1::from_vec(vec![1.1, -0.6]);
        let active_covariance =
            Array2::from_shape_vec((2, 2), vec![3.0, 0.7, 0.7, 1.8]).expect("covariance");
        let control = control(2.0e-9);
        let symmetric = symmetrized_covariance(active_covariance.view());
        let projected = project_active_covariance(symmetric.view(), control.absolute_tolerance)
            .expect("project covariance");
        let mut rules = ConditionedThreeClassRuleLadder::default();

        let first = integrate_three_class_conditionally(
            active_mean.as_slice().expect("contiguous mean"),
            &projected,
            &control,
            &mut rules,
        )
        .expect("first prediction row");
        let constructions_after_first_row = rules.rules.len();
        assert!(
            constructions_after_first_row >= 2,
            "certification must compare at least two rules"
        );

        let second = integrate_three_class_conditionally(
            active_mean.as_slice().expect("contiguous mean"),
            &projected,
            &control,
            &mut rules,
        )
        .expect("identical second prediction row");
        assert_eq!(
            rules.rules.len(),
            constructions_after_first_row,
            "an identical row must reuse every constructed rule"
        );
        assert_eq!(first.rule, second.rule);
        assert_eq!(first.function_evaluations, second.function_evaluations);
        assert_eq!(
            first.max_raw_moment_level_difference.to_bits(),
            second.max_raw_moment_level_difference.to_bits()
        );
        for (&left, &right) in first.class_mean.iter().zip(second.class_mean.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        for (&left, &right) in first
            .class_covariance
            .iter()
            .zip(second.class_covariance.iter())
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    /// Re-reference and permute all three classes, including exchanging an
    /// active class with the reference class.  A structural conditional
    /// reduction must commute with this change of coordinates: choosing which
    /// logit to condition is a work decision, not a statistical one.
    #[test]
    fn conditioned_three_class_moments_are_invariant_to_class_permutation_2612() {
        let active_mean = Array1::from_vec(vec![1.4, -0.9]);
        let active_covariance =
            Array2::from_shape_vec((2, 2), vec![7.0, -1.3, -1.3, 2.5]).expect("covariance");
        let control = control(2.0e-9);
        let original = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control,
        )
        .expect("original conditioned moments");
        assert!(matches!(
            &original.rule,
            MultinomialPosteriorRule::ConditionedThreeClass(_)
        ));

        // New class j is old class permutation[j].  The new reference is old
        // active class zero, so this covers the reference-coding boundary
        // rather than merely swapping the two stored active columns.
        let permutation = [2usize, 1usize, 0usize];
        let old_full_mean = [active_mean[0], active_mean[1], 0.0];
        let new_active_mean = Array1::from_vec(vec![
            old_full_mean[permutation[0]] - old_full_mean[permutation[2]],
            old_full_mean[permutation[1]] - old_full_mean[permutation[2]],
        ]);
        let old_active_coefficient = |class: usize| match class {
            0 => [1.0, 0.0],
            1 => [0.0, 1.0],
            2 => [0.0, 0.0],
            _ => unreachable!("three classes"),
        };
        let reference_coefficient = old_active_coefficient(permutation[2]);
        let transformation = [
            {
                let class = old_active_coefficient(permutation[0]);
                [
                    class[0] - reference_coefficient[0],
                    class[1] - reference_coefficient[1],
                ]
            },
            {
                let class = old_active_coefficient(permutation[1]);
                [
                    class[0] - reference_coefficient[0],
                    class[1] - reference_coefficient[1],
                ]
            },
        ];
        let new_active_covariance = Array2::from_shape_fn((2, 2), |(row, column)| {
            let mut value = 0.0;
            for left in 0..2 {
                for right in 0..2 {
                    value += transformation[row][left]
                        * active_covariance[[left, right]]
                        * transformation[column][right];
                }
            }
            value
        });
        let permuted = integrate_logistic_normal_softmax_moments(
            new_active_mean.view(),
            new_active_covariance.view(),
            &control,
        )
        .expect("permuted conditioned moments");

        for new_class in 0..3 {
            let old_class = permutation[new_class];
            assert_close(
                permuted.class_mean[new_class],
                original.class_mean[old_class],
                2.0e-8,
                "permuted class mean",
            );
            for new_other in 0..3 {
                let old_other = permutation[new_other];
                assert_close(
                    permuted.class_covariance[[new_class, new_other]],
                    original.class_covariance[[old_class, old_other]],
                    5.0e-8,
                    "permuted class covariance",
                );
            }
        }
        for row in 0..3 {
            assert_close(
                permuted.class_covariance.row(row).sum(),
                0.0,
                3.0e-14,
                "conditioned simplex covariance row sum",
            );
        }
    }

    /// The work gap is structural, not a larger allowance.  Learn the outer
    /// resolution this posterior needs, then rerun with one fewer evaluation
    /// than a square grid at that same per-direction resolution.  The exact
    /// conditioned rule still fits because its work is linear in the node
    /// count; an unreduced Cartesian rule cannot.
    #[test]
    fn conditioned_three_class_rule_fits_below_the_corresponding_tensor_cost_2612() {
        let active_mean = Array1::from_vec(vec![2.0, -1.0]);
        let active_covariance = wide_two_direction_covariance();
        let baseline = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &MultinomialPosteriorIntegrationControl::default(),
        )
        .expect("baseline conditioned moments");
        let nodes = match baseline.rule {
            MultinomialPosteriorRule::ConditionedThreeClass(nodes) => nodes,
            other => panic!("expected conditioned rule, got {other:?}"),
        };
        assert!(nodes > 1, "wide posterior must require a refinement");
        let square_cost = nodes.checked_mul(nodes).expect("tensor cost");
        let tight_control = MultinomialPosteriorIntegrationControl {
            maximum_function_evaluations: square_cost - 1,
            ..MultinomialPosteriorIntegrationControl::default()
        };
        let conditioned = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &tight_control,
        )
        .expect("conditioned rule must fit below corresponding tensor cost");
        assert!(
            conditioned.function_evaluations < square_cost,
            "conditioned work {} must be below {nodes}²={square_cost}",
            conditioned.function_evaluations
        );
        for class in 0..3 {
            assert_close(
                conditioned.class_mean[class],
                baseline.class_mean[class],
                3.0e-8,
                "tight-budget conditioned mean",
            );
        }
    }
}
