#![cfg(test)]
//! Tests for the survival marginal-slope family (relocated verbatim).

// The parent declares this module as `#[cfg(test)] mod tests;`; declaring the
// test scope in-file makes that a claim the compiler enforces.
#![cfg(test)]

use super::*;
use crate::custom_family::{CustomFamily, ExactOuterDerivativeOrder};
use approx::assert_relative_eq;
use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::matrix::{DenseDesignMatrix, SymmetricMatrix};
// Trait import for the `.value()` calls in the linux-gated #932 pullback test —
// on other targets that test is compiled out and the import would be dead.
#[cfg(target_os = "linux")]
use gam_math::nested_dual::JetField;
use ndarray::array;

/// Local scalar closeness assertion used throughout this module's exactness
/// gates. Asserts `|lhs - rhs| <= tol`, reporting both operands on failure.
fn assert_close(lhs: f64, rhs: f64, tol: f64, label: &str) {
    assert!(
        (lhs - rhs).abs() <= tol,
        "{label} mismatch: lhs={lhs:.12e}, rhs={rhs:.12e}, tol={tol:.3e}"
    );
}

/// Pin the survival cross-block W metric to the survival row Hessian
/// formula `u2_eta1 = (1-d)·w·k2(-η, w·(1-d)) + d·w` rather than the
/// Bernoulli probit proxy `φ²/(Φ·(1-Φ))`. Three rows exercise the
/// censored-only, event-only, and weighted-censored branches; the
/// expected values are recomputed in-test from
/// `signed_probit_neglog_derivatives_up_to_fourth` to bind the metric
/// to its source-of-truth derivative routine.
#[test]
fn survival_pilot_irls_row_metric_matches_u2_eta1_formula() {
    let eta = ndarray::Array1::from(vec![0.3_f64, -0.7, 1.2]);
    let weights = ndarray::Array1::from(vec![1.0_f64, 2.5, 0.5]);
    let event = ndarray::Array1::from(vec![0.0_f64, 1.0, 0.0]);
    let computed = super::survival_pilot_irls_row_metric_at_eta(&eta, &weights, &event)
        .expect("survival pilot W metric");
    assert_eq!(computed.len(), 3);
    for i in 0..3 {
        let d = event[i];
        let w = weights[i];
        let e = eta[i];
        let (_, k2, _, _) = super::signed_probit_neglog_derivatives_up_to_fourth(-e, w * (1.0 - d))
            .expect("probit k2");
        let expected = k2 + w * d;
        approx::assert_relative_eq!(computed[i], expected, max_relative = 1e-12);
    }
    // Censored row (d=0): metric must be strictly positive at finite η —
    // the Mills-ratio second derivative of -log Φ(-η) is positive on ℝ.
    assert!(
        computed[0] > 0.0,
        "censored row metric must be > 0, got {}",
        computed[0],
    );
    // Event row (d=1): the formula collapses to w·d exactly; the censored
    // branch contributes zero because the k2 weight `w·(1-d)` is zero.
    approx::assert_relative_eq!(computed[1], weights[1], max_relative = 1e-12);
}

/// Length-mismatch contract: the metric helper must reject misaligned
/// inputs instead of producing a truncated W vector that silently
/// passes the cross-block routine but breaks the W-inner product.
#[test]
fn survival_pilot_irls_row_metric_rejects_length_mismatch() {
    let eta = ndarray::Array1::from(vec![0.0_f64, 1.0]);
    let weights = ndarray::Array1::from(vec![1.0_f64]); // wrong length
    let event = ndarray::Array1::from(vec![0.0_f64, 0.0]);
    let result = super::survival_pilot_irls_row_metric_at_eta(&eta, &weights, &event);
    match result {
        Ok(_) => panic!("expected length-mismatch error, got Ok"),
        Err(msg) => assert!(
            msg.contains("length mismatch"),
            "expected 'length mismatch' in error, got: {msg}",
        ),
    }
}

fn empty_termspec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
        level: Default::default(),
    }
}

fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(MarginalSlopeCovariance::diagonal(array![1.0]).unwrap())
}

fn no_spatial_joint_setup(rho_dim: usize) -> ExactJointHyperSetup {
    let no_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), Vec::new());
    // No block owns these coordinates, so the representable log-strength range
    // is their only domain.
    ExactJointHyperSetup::new(
        Array1::zeros(rho_dim),
        Array1::from_elem(rho_dim, gam_problem::LOG_STRENGTH_MIN),
        Array1::from_elem(rho_dim, gam_problem::LOG_STRENGTH_MAX),
        no_kappa.clone(),
        no_kappa.clone(),
        no_kappa,
    )
}

/// gam#2768: absence of a spatial outer certificate is the generic driver's
/// documented fast path, not failure of the ordinary smoothing optimizer.
/// The final rho must come from the certified inner fit that owned it.
#[test]
fn survival_no_spatial_outer_uses_certified_inner_rho_2768() {
    let setup = no_spatial_joint_setup(2);
    let fitted = array![-1.25, 0.75];
    let terminal = terminal_survival_hyper_theta(&setup, true, &fitted, None)
        .expect("the no-spatial fast path has a certified inner rho");
    assert_eq!(terminal, fitted);
}

/// The no-certificate route is not a fallback: a family/auxiliary coordinate
/// makes the spatial-joint outer solve mandatory and must still be refused.
#[test]
fn survival_missing_required_joint_outer_certificate_is_rejected_2768() {
    let setup = no_spatial_joint_setup(1).with_auxiliary(
        array![0.0],
        array![-2.0],
        array![2.0],
    );
    let error = terminal_survival_hyper_theta(&setup, true, &array![0.5], None)
        .expect_err("an auxiliary coordinate requires a joint outer certificate");
    assert!(
        error.contains("required an outer certificate"),
        "unexpected missing-certificate diagnostic: {error}",
    );
}

fn base_time_block() -> TimeBlockInput {
    TimeBlockInput {
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Array1::zeros(1),
        offset_exit: Array1::zeros(1),
        derivative_offset_exit: Array1::from_elem(
            1,
            DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        ),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(1)),
    }
}

/// gnomon#2336: the pilot baseline slope solves the fitted row objective, so a
/// row entering at the time origin contributes no entry factor there either.
/// Moving the entry offsets of landmarked rows must leave the pilot slope
/// unchanged, while the same move on delayed-entry rows must reach it.
#[test]
fn pooled_survival_baseline_ignores_origin_entry_offsets_2336() {
    let n = 12;
    let z = Array1::from_shape_fn(n, |i| (i as f64 - 5.5) / 4.0);
    let event = Array1::from_shape_fn(n, |i| if (i * 7) % 12 < 7 { 1.0 } else { 0.0 });
    let weights = Array1::from_elem(n, 1.0);
    let q1 = Array1::from_shape_fn(n, |i| -0.8 + 0.1 * i as f64);
    let qd1 = Array1::from_elem(n, 1.0);
    let entry_low = Array1::from_elem(n, -2.5);
    let entry_high = Array1::from_elem(n, -1.2);
    let unit_variance = Array1::from_elem(n, 1.0);
    let pilot = |entry_at_origin: &Array1<bool>, q0: &Array1<f64>| {
        pooled_survival_baseline(
            &event,
            &weights,
            entry_at_origin,
            &z,
            &unit_variance,
            q0,
            &q1,
            &qd1,
            1.0,
        )
        .expect("the pilot certifies a minimizer of the pooled objective")
    };
    let origin = Array1::from_elem(n, true);
    let delayed = Array1::from_elem(n, false);

    // A zero slope would mean the score already vanished at the origin, which
    // leaves the entry-offset comparison below without a solve to compare.
    let origin_slope = pilot(&origin, &entry_low);
    assert!(
        origin_slope.is_finite() && origin_slope != 0.0,
        "the landmarked pilot must solve for a slope; got {origin_slope:e}"
    );
    assert_eq!(
        origin_slope,
        pilot(&origin, &entry_high),
        "the pilot read the entry offsets of rows that enter at the origin"
    );
    assert_ne!(
        pilot(&delayed, &entry_low),
        pilot(&delayed, &entry_high),
        "the entry offsets must reach a delayed-entry pilot, or this gate is vacuous"
    );
}

/// gam#2952: the pilot baseline slope solves the fitted row objective, and that
/// objective's score variance is the conditional `Var(z | a)` the family's row
/// inputs read (gam#2766). At a non-unit variance the pilot must reach a lower
/// value of THAT objective, summed through the frame's own row value, than the
/// slope it reaches at unit variance, which is what a pilot that set
/// `1ᵀΣ1 = 1` returned.
#[test]
fn pooled_survival_baseline_solves_at_the_conditional_score_variance_2952() {
    let n = 12;
    let z = Array1::from_shape_fn(n, |i| (i as f64 - 5.5) / 4.0);
    let event = Array1::from_shape_fn(n, |i| if (i * 7) % 12 < 7 { 1.0 } else { 0.0 });
    let weights = Array1::from_elem(n, 1.0);
    let delayed = Array1::from_elem(n, false);
    let q0 = Array1::from_elem(n, -2.5);
    let q1 = Array1::from_shape_fn(n, |i| -0.8 + 0.1 * i as f64);
    let qd1 = Array1::from_elem(n, 1.0);
    let variance = 0.25;
    let pilot = |z_variance: f64| {
        pooled_survival_baseline(
            &event,
            &weights,
            &delayed,
            &z,
            &Array1::from_elem(n, z_variance),
            &q0,
            &q1,
            &qd1,
            1.0,
        )
        .expect("the pilot certifies a minimizer of the pooled objective")
    };
    let conditional_slope = pilot(variance);
    let unit_slope = pilot(1.0);
    // A zero slope would mean the score already vanished at the origin.
    assert!(
        conditional_slope.is_finite() && conditional_slope != 0.0,
        "the pilot must solve for a slope at Var(z | a) = {variance}; got {conditional_slope:e}"
    );
    let objective = |slope: f64| -> (f64, f64) {
        (0..n).fold((0.0, 0.0), |(total, magnitude), i| {
            let inputs = RigidRowInputs {
                row: i,
                wi: weights[i],
                wi_entry: weights[i],
                di: event[i],
                z_sum: z[i],
                covariance_ones: variance,
                probit_scale: 1.0,
                qd1_lower: 0.0,
                anchor: None,
            };
            let value = rigid_row_value::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                &[q0[i], q1[i], qd1[i], slope],
                &inputs,
            )
            .expect("an admissible row");
            (total + value, magnitude + value.abs())
        })
    };
    let (at_conditional, conditional_magnitude) = objective(conditional_slope);
    let (at_unit, unit_magnitude) = objective(unit_slope);
    // Each total is an n-term sum, so two of them can differ by rounding alone
    // up to γ_{n−1}·(Σ|termᵢ| + Σ|termⱼ|), with γ_k = k·u/(1 − k·u) and u = ε/2.
    let unit_roundoff = 0.5 * f64::EPSILON;
    let terms = (n - 1) as f64;
    let rounding = terms * unit_roundoff / (1.0 - terms * unit_roundoff)
        * (conditional_magnitude + unit_magnitude);
    assert!(
        at_conditional < at_unit - rounding,
        "the pilot ignored the conditional score variance: at Var(z | a) = {variance} its slope \
         {conditional_slope:.12e} scores {at_conditional:.15e}, not below the unit-variance slope \
         {unit_slope:.12e}'s {at_unit:.15e} by more than the rounding {rounding:.3e}"
    );
}

/// The twelve delayed-entry rows of the gnomon#2336 and gam#2952 gates, whose
/// pooled pilot objective has an interior minimizer.
struct PooledPilotFixture {
    event: Array1<f64>,
    weights: Array1<f64>,
    entry_at_origin: Array1<bool>,
    z: Array1<f64>,
    z_variance: Array1<f64>,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
}

impl PooledPilotFixture {
    fn new() -> Self {
        let n = 12;
        Self {
            event: Array1::from_shape_fn(n, |i| if (i * 7) % 12 < 7 { 1.0 } else { 0.0 }),
            weights: Array1::from_elem(n, 1.0),
            entry_at_origin: Array1::from_elem(n, false),
            z: Array1::from_shape_fn(n, |i| (i as f64 - 5.5) / 4.0),
            z_variance: Array1::from_elem(n, 1.0),
            q0: Array1::from_elem(n, -2.5),
            q1: Array1::from_shape_fn(n, |i| -0.8 + 0.1 * i as f64),
            qd1: Array1::from_elem(n, 1.0),
        }
    }

    fn pilot(&self) -> Result<f64, SurvivalMarginalSlopeError> {
        pooled_survival_baseline(
            &self.event,
            &self.weights,
            &self.entry_at_origin,
            &self.z,
            &self.z_variance,
            &self.q0,
            &self.q1,
            &self.qd1,
            1.0,
        )
    }

    /// The pooled score and curvature in the slope, summed in row order
    /// through the row program the pilot solves, each with the rounding band
    /// `γ_n·Σ|termᵢ|` of its sum: `(score, score_band, curvature, curvature_band)`.
    fn pooled_score(&self, slope: f64) -> (f64, f64, f64, f64) {
        let n = self.event.len();
        let mut sums = [0.0_f64; 4];
        for i in 0..n {
            let (_, grad, hess) = row_primary_closed_form(
                self.q0[i],
                self.q1[i],
                self.qd1[i],
                slope,
                self.z[i],
                self.z_variance[i],
                self.weights[i],
                if self.entry_at_origin[i] {
                    0.0
                } else {
                    self.weights[i]
                },
                self.event[i],
                0.0,
                1.0,
            )
            .expect("an admissible row");
            sums[0] += grad[3];
            sums[1] += grad[3].abs();
            sums[2] += hess[3][3];
            sums[3] += hess[3][3].abs();
        }
        (
            sums[0],
            gam_linalg::roundoff::accumulation_band(n, sums[1]),
            sums[2],
            gam_linalg::roundoff::accumulation_band(n, sums[3]),
        )
    }
}

/// gam#4250: the pilot slope becomes part of the model (the slope block's
/// offset, the frailty gate, the absorber columns and the saved baseline), so
/// it must be a certified root of the pooled score, not the lowest probe of a
/// capped search. At the returned slope the pooled score is zero to the
/// rounding of its own sum, or it changes sign across the adjacent floats, so
/// no representable slope sits closer to the root.
#[test]
fn pooled_survival_baseline_returns_a_certified_score_root_4250() {
    let fixture = PooledPilotFixture::new();
    let slope = fixture
        .pilot()
        .expect("the pilot certifies a minimizer of the pooled objective");
    assert!(
        slope.is_finite() && slope != 0.0,
        "the fixture's pooled score does not vanish at the origin, so its root is not 0; got \
         {slope:e}"
    );
    let (score, band, curvature, curvature_band) = fixture.pooled_score(slope);
    let (below, ..) = fixture.pooled_score(slope.next_down());
    let (above, ..) = fixture.pooled_score(slope.next_up());
    assert!(
        score.abs() <= band || (below <= 0.0 && above >= 0.0),
        "the pilot slope {slope:.17e} is not a certified root: pooled score {score:e} outside \
         its rounding band {band:e}, and no sign change across the adjacent floats (score \
         {below:e} below, {above:e} above)"
    );
    // The root is a minimizer, not a stationary point of another kind.
    assert!(
        curvature > curvature_band,
        "the pilot slope {slope:e} is not a minimizer of the pooled objective: curvature \
         {curvature:e} is not resolved positive (band {curvature_band:e})"
    );
}

/// gam#4250: a row that does not evaluate is an error naming that row, not a
/// silent `0.0` that the fit would read as "the pooled slope is zero".
#[test]
fn pooled_survival_baseline_refuses_a_row_that_does_not_evaluate_4250() {
    let mut fixture = PooledPilotFixture::new();
    // A decreasing baseline breaks the monotonicity every row is admitted under.
    fixture.qd1[3] = -1.0;
    match fixture.pilot() {
        Err(SurvivalMarginalSlopeError::NumericalFailure { reason }) => assert!(
            reason.contains("row 3"),
            "the refusal must name the row that failed to evaluate: {reason}"
        ),
        other => panic!("a row that does not evaluate must refuse the pilot; got {other:?}"),
    }
}

fn sparse_design(dense: &Array2<f64>) -> DesignMatrix {
    let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let value = dense[[i, j]];
            if value != 0.0 {
                triplets.push(Triplet::new(i, j, value));
            }
        }
    }
    let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
        .expect("assemble sparse design");
    DesignMatrix::Sparse(gam_linalg::matrix::SparseDesignMatrix::new(sparse))
}

/// Build an n-row closed-form survival family with empty time/marginal/
/// slope blocks (so q0/q1/qd1 come from offsets only). No flex
/// deviations are configured, so `log_likelihood_only` takes the
/// closed-form fast path.
fn make_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    // Pseudo-random rows uncorrelated with row parity, so an even-only
    // subsample is representative for the Horvitz-Thompson rescaling
    // check.
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    // qd1 must remain strictly above the derivative guard.
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        // Empty time/marginal/slope designs: `n_rows × 0` so the
        // closed-form q geometry is driven entirely by offsets.
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(Array2::zeros((n, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((n, 0)))).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

/// gnomon#2337: the rigid row kernel reads the marginal and time-constant slope
/// designs one row at a time. An operator-backed design is memoized once
/// through the governed chunked path (`no_densify_design` panics if anything
/// densifies it through `to_dense` instead), and every row read afterwards
/// returns what the streamed read returned.
#[test]
fn operator_backed_covariate_designs_are_memoized_and_read_the_same_rows_2337() {
    let n = 37;
    let marginal = Array2::from_shape_fn((n, 3), |(row, col)| {
        ((row * 7 + col * 3) % 11) as f64 * 0.25 - 1.0
    });
    let slope = Array2::from_shape_fn((n, 2), |(row, col)| {
        ((row * 5 + col) % 13) as f64 * 0.1 - 0.4
    });
    let mut family = make_closed_form_test_family(n);
    family.marginal_design = gam_linalg_test_support::no_densify_design(marginal.clone());
    family.slope_layout = gam_linalg_test_support::no_densify_design(slope.clone()).into();
    let beta = array![0.3, -1.2, 0.7];
    let streamed: Vec<f64> = (0..n)
        .map(|row| family.marginal_design.dot_row_view(row, beta.view()))
        .collect();
    assert!(family.marginal_design.as_dense_ref().is_none());

    family.memoize_operator_backed_designs();

    assert_eq!(family.marginal_design.as_dense_ref(), Some(&marginal));
    let slope_design = family
        .slope_layout
        .static_coefficient_design()
        .expect("a time-constant slope layout exposes its coefficient design");
    assert_eq!(slope_design.as_dense_ref(), Some(&slope));
    for (row, expected) in streamed.iter().enumerate() {
        assert_eq!(
            family.marginal_design.dot_row_view(row, beta.view()),
            *expected
        );
    }
}

/// Operator-backed design whose construction policy requires streamed storage.
struct StreamedOnlyDesignOperator {
    dense: Array2<f64>,
}

impl gam_linalg::matrix::LinearOperator for StreamedOnlyDesignOperator {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.dot(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.t().dot(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let weighted = &self.dense * &weights.view().insert_axis(Axis(1));
        Ok(self.dense.t().dot(&weighted))
    }
}

impl gam_linalg::matrix::DenseDesignOperator for StreamedOnlyDesignOperator {
    fn row_chunk_into(
        &self,
        rows: std::ops::Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), gam_runtime::resource::MatrixMaterializationError> {
        out.assign(&self.dense.slice(ndarray::s![rows, ..]));
        Ok(())
    }

    fn materialization_policy(&self) -> Option<gam_runtime::resource::MaterializationPolicy> {
        Some(gam_runtime::resource::MaterializationPolicy {
            max_single_dense_bytes: 0,
            max_cached_dense_bytes: 0,
            row_chunk_target_bytes: 1024,
            allow_operator_materialization: false,
            allow_diagnostic_materialization: false,
        })
    }

    fn to_dense(&self) -> Array2<f64> {
        // SAFETY: the construction policy above forbids densifying this
        // operator; a call here is the regression the test exists to catch.
        panic!("StreamedOnlyDesignOperator must stay streamed")
    }
}

/// gnomon#2337: a design whose construction policy requires streamed storage
/// is refused by the memo and keeps serving row reads from the operator.
#[test]
fn a_design_whose_construction_policy_requires_streaming_stays_streamed_2337() {
    let n = 11;
    let marginal = Array2::from_shape_fn((n, 2), |(row, col)| (row + 2 * col) as f64 * 0.5);
    let mut family = make_closed_form_test_family(n);
    family.marginal_design = DesignMatrix::from(DenseDesignMatrix::from(Arc::new(
        StreamedOnlyDesignOperator {
            dense: marginal.clone(),
        },
    )));

    family.memoize_operator_backed_designs();

    assert!(family.marginal_design.as_dense_ref().is_none());
    let beta = array![1.0, -1.0];
    assert_eq!(
        family.marginal_design.dot_row_view(4, beta.view()),
        marginal.row(4).dot(&beta)
    );
}

/// gnomon#2337 exactness gate: a memoized gauged design reproduces the lazy
/// operator in every per-row action the rigid row kernel takes.
///
/// The designs are built the way a Duchon term under a collection gauge is: a
/// `CoefficientTransformOperator` over a lazy `BlockDesignOperator[X, Q]` with
/// transform `[I; −R]`. The streamed path multiplies each row by the transform
/// through `fast_ab` over a one-row block; the memo multiplies whole row chunks
/// through the same `fast_ab`, which may select a different product kernel for
/// the larger block. Everything after the design row (`dot_row_view`,
/// `axpy_row_into`, their squared and cross forms) reduces in the same order on
/// both paths, so the only difference is the rounding of each design row:
/// `|streamed − memoized| ≤ 1e-14 · max(|streamed|, |memoized|, 1)`.
#[test]
fn memoized_gauged_designs_reproduce_the_operator_in_every_row_action_2337() {
    use crate::row_kernel::RowKernel;
    use gam_linalg::matrix::{BlockDesignOperator, CoefficientTransformOperator, DesignBlock};

    const PRIMARIES: usize = STATIC_SLOPE_PRIMARIES;

    fn noise(i: usize, j: usize, salt: f64) -> f64 {
        ((i as f64 * 12.9898 + j as f64 * 78.233 + salt).sin() * 43758.5453).fract() - 0.5
    }

    fn gauged(x: &Array2<f64>, q: &Array2<f64>, r: &Array2<f64>) -> DesignMatrix {
        let stacked = BlockDesignOperator::new(vec![
            DesignBlock::Dense(DenseDesignMatrix::from(x.clone())),
            DesignBlock::Dense(DenseDesignMatrix::from(q.clone())),
        ])
        .expect("stacked gauge block");
        let p = x.ncols();
        let mut transform = Array2::<f64>::zeros((p + q.ncols(), p));
        for column in 0..p {
            transform[[column, column]] = 1.0;
        }
        for range_column in 0..q.ncols() {
            for column in 0..p {
                transform[[p + range_column, column]] = -r[[range_column, column]];
            }
        }
        let operator = CoefficientTransformOperator::new(
            DenseDesignMatrix::from(Arc::new(stacked)),
            transform,
        )
        .expect("gauge coefficient transform");
        DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(operator)))
    }

    fn assert_rows_agree(streamed: &[f64], memoized: &[f64], label: &str, row: usize) {
        assert_eq!(streamed.len(), memoized.len());
        for (index, (&a, &b)) in streamed.iter().zip(memoized).enumerate() {
            let bound = 1e-14 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= bound,
                "{label} row {row} entry {index}: streamed={a:.17e} memoized={b:.17e}"
            );
        }
    }

    let n = 96;
    let (marginal_cols, slope_cols, gauge_rank) = (7, 4, 3);
    let q = Array2::from_shape_fn((n, gauge_rank), |(i, j)| noise(i, j, 1.0));
    let marginal_x = Array2::from_shape_fn((n, marginal_cols), |(i, j)| noise(i, j, 2.0));
    let marginal_r = Array2::from_shape_fn((gauge_rank, marginal_cols), |(i, j)| noise(i, j, 3.0));
    let slope_x = Array2::from_shape_fn((n, slope_cols), |(i, j)| noise(i, j, 4.0));
    let slope_r = Array2::from_shape_fn((gauge_rank, slope_cols), |(i, j)| noise(i, j, 5.0));
    // Two independent design objects: clones share one memo, so the streamed
    // family must be built from scratch rather than cloned.
    let build_family = || {
        let mut family = make_closed_form_test_family(n);
        family.marginal_design = gauged(&marginal_x, &q, &marginal_r);
        family.slope_layout = gauged(&slope_x, &q, &slope_r).into();
        family
    };
    let streamed_family = build_family();
    let memoized_family = build_family();
    memoized_family.memoize_operator_backed_designs();
    assert!(streamed_family.marginal_design.as_dense_ref().is_none());
    assert!(memoized_family.marginal_design.as_dense_ref().is_some());

    let states = || {
        vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(marginal_cols),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(slope_cols),
                eta: Array1::zeros(n),
            },
        ]
    };
    let streamed =
        SurvivalMarginalSlopeRowKernel::<PRIMARIES, StaticSlopeGeometry>::new(streamed_family, states());
    let memoized =
        SurvivalMarginalSlopeRowKernel::<PRIMARIES, StaticSlopeGeometry>::new(memoized_family, states());
    let total = marginal_cols + slope_cols;

    for row in 0..n {
        let direction: Vec<f64> = (0..total).map(|j| noise(row, j, 6.0)).collect();
        assert_rows_agree(
            &streamed.jacobian_action(row, &direction),
            &memoized.jacobian_action(row, &direction),
            "jacobian_action",
            row,
        );

        let primary: [f64; PRIMARIES] = std::array::from_fn(|a| noise(row, a, 7.0));
        let mut streamed_pullback = vec![0.0; total];
        let mut memoized_pullback = vec![0.0; total];
        streamed.jacobian_transpose_action(row, &primary, &mut streamed_pullback);
        memoized.jacobian_transpose_action(row, &primary, &mut memoized_pullback);
        assert_rows_agree(
            &streamed_pullback,
            &memoized_pullback,
            "jacobian_transpose_action",
            row,
        );

        let hessian: [[f64; PRIMARIES]; PRIMARIES] = std::array::from_fn(|a| {
            std::array::from_fn(|b| noise(row, a.min(b) * PRIMARIES + a.max(b), 8.0))
        });
        let mut streamed_diagonal = vec![0.0; total];
        let mut memoized_diagonal = vec![0.0; total];
        streamed.add_diagonal_quadratic(row, &hessian, &mut streamed_diagonal);
        memoized.add_diagonal_quadratic(row, &hessian, &mut memoized_diagonal);
        assert_rows_agree(
            &streamed_diagonal,
            &memoized_diagonal,
            "add_diagonal_quadratic",
            row,
        );
    }
}

#[test]
fn k1_shared_slope_uses_cached_arbitrary_variance_932() {
    let mut family = make_closed_form_test_family(3);
    family.score_covariance =
        ScoreCovarianceField::pooled(MarginalSlopeCovariance::diagonal(array![3.75]).unwrap());
    let cached = family
        .score_covariance
        .pooled_covariance()
        .ones_quadratic_form();
    assert!((cached - 3.75).abs() <= f64::EPSILON);
    for row in 0..family.n {
        assert_eq!(family.shared_slope_covariance_scale(row), cached);
    }
}

fn closed_form_block_states(
    family: &SurvivalMarginalSlopeFamily,
    g: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    vec![
        // Time block: empty beta; per-row eta entries unused (designs
        // are zero-column).
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        // Marginal block: empty beta.
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        // Slope block: empty beta with per-row eta = g (constant
        // slope across rows).
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_elem(n, g),
        },
    ]
}

#[test]
fn converged_identifiability_scalars_use_current_vector_geometry_932() {
    let n = 2;
    let mut family = make_closed_form_test_family(n);
    let design = array![[1.0], [2.0]];
    let offset = array![0.2, -0.1];
    family.slope_layout = SlopeTopology::shared()
        .materialize_identity(DesignMatrix::from(design.clone()), &offset)
        .unwrap();
    let beta_slope = array![0.5];
    let eta_slope = design.dot(&beta_slope) + &offset;
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_slope,
            eta: eta_slope.clone(),
        },
    ];

    let erased =
        <SurvivalMarginalSlopeFamily as CustomFamily>::current_identifiability_family_scalars(
            &family, &states,
        )
        .unwrap()
        .expect("survival must expose converged scalars");
    let scalars = erased
        .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
        .expect("survival scalar type");
    for row in 0..n {
        assert_eq!(scalars.q0_i[row], family.offset_entry[row]);
        assert_eq!(scalars.q1_i[row], family.offset_exit[row]);
        assert_eq!(scalars.qd1_i[row], family.derivative_offset_exit[row]);
        assert_eq!(
            scalars.c_i[row],
            (1.0 + eta_slope[row] * eta_slope[row]).sqrt(),
        );
    }
}

#[test]
fn survival_primary_g_fourth_cell_partials_are_zero() {
    let family = make_closed_form_test_family(1);
    let primary = flex_primary_slices(&family);
    let score_span = exact_kernel::LocalSpanCubic {
        left: -1.0,
        right: 1.0,
        c0: 0.2,
        c1: -0.1,
        c2: 0.05,
        c3: -0.02,
    };
    let link_span = exact_kernel::LocalSpanCubic {
        left: -0.5,
        right: 0.5,
        c0: 0.1,
        c1: 0.3,
        c2: -0.2,
        c3: 0.4,
    };
    let fixed = family
        .denested_cell_primary_fixed_partials(&primary, 0.2, 0.7, score_span, link_span, 0.0, 0.0)
        .expect("primary fixed partials");
    let (_, dc_daab, dc_dabb, dc_dbbb) = exact_kernel::denested_cell_third_partials(link_span);

    assert_eq!(fixed.coeff_aau[primary.g], dc_daab);
    assert_eq!(fixed.coeff_abu[primary.g], dc_dabb);
    assert_eq!(fixed.coeff_bbu[primary.g], dc_dbbb);
    assert!(dc_daab.iter().any(|value| *value != 0.0));
    assert!(dc_dabb.iter().any(|value| *value != 0.0));
    assert!(dc_dbbb.iter().any(|value| *value != 0.0));
    assert_eq!(fixed.coeff_aaau[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_aabu[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_abbu[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_bbbu[primary.g], [0.0; 4]);
}

#[test]
fn survival_log_likelihood_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let baseline = family
        .log_likelihood_only(&states)
        .expect("baseline ll (no subsample)");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full_mask = family
        .log_likelihood_only_with_options(&states, &opts_full)
        .expect("ll with mask=full");

    let rel = ((with_full_mask - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "subsample(mask=full) {} differs from baseline {} by rel {}",
        with_full_mask,
        baseline,
        rel
    );
}

#[test]
fn survival_log_likelihood_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .log_likelihood_only_with_options(&states, &opts_half)
        .expect("ll with mask=even");

    // Raw even-row sum: same mask but weight_scale = 1.0.
    let mut opts_even_unscaled = BlockwiseFitOptions::default();
    opts_even_unscaled.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::with_uniform_weight(even_mask, m, 0, 1.0),
    ));
    let raw_even_sum = family
        .log_likelihood_only_with_options(&states, &opts_even_unscaled)
        .expect("raw even-row ll sum");

    let expected_scaled = (n as f64 / m as f64) * raw_even_sum;
    let rel = ((scaled - expected_scaled) / expected_scaled.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "scaled {} != 2*even_sum {} (rel {})",
        scaled,
        expected_scaled,
        rel
    );

    // Horvitz-Thompson check: 2 * Σ_even ≈ full-data sum.
    let baseline = family.log_likelihood_only(&states).expect("baseline ll");
    let ht_rel = ((scaled - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        ht_rel < 0.05,
        "Horvitz-Thompson scaled {} not near baseline {} (rel {})",
        scaled,
        baseline,
        ht_rel
    );
}

fn dummy_blockspec(cols: usize) -> ParameterBlockSpec {
    // `validate_blockspecs` enforces unique block names so coefficient
    // labels stay unambiguous. A monotonic per-process counter keeps
    // each call's name distinct even when multiple specs are stacked
    // into one `Vec`.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let idx = SEQ.fetch_add(1, Ordering::Relaxed);
    ParameterBlockSpec {
        name: format!("dummy_{idx}"),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(cols)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn dummy_penalized_blockspec(cols: usize, penalties: usize) -> ParameterBlockSpec {
    let mut spec = dummy_blockspec(cols);
    spec.penalties = (0..penalties)
        .map(|_| PenaltyMatrix::Dense(Array2::eye(cols)))
        .collect();
    spec.nullspace_dims = vec![0; penalties];
    spec.initial_log_lambdas = Array1::zeros(penalties);
    spec
}

fn test_deviation_runtime() -> DeviationRuntime {
    build_score_warp_deviation_block_from_seed(
        &array![-1.0, 0.0, 1.0],
        &DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 1,
            penalty_order: 2,
            penalty_orders: vec![1, 2, 3],
            double_penalty: false,
            monotonicity_eps: 1e-4,
        },
    )
    .expect("build test deviation runtime")
    .runtime
}

fn max_abs_diff_vec(lhs: &Array1<f64>, rhs: &Array1<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_diff_mat(lhs: &Array2<f64>, rhs: &Array2<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_blockwise_matches_joint_principal_blocks(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
) {
    let eval = family
        .evaluate_blockwise_exact_newton(block_states)
        .expect("blockwise exact-newton evaluation");
    let (joint_ll, joint_gradient, joint_hessian) = family
        .evaluate_exact_newton_joint_dense(block_states)
        .expect("joint dense exact-newton evaluation");
    let slices = block_slices(family, block_states);
    let mut block_ranges = vec![
        slices.time.clone(),
        slices.marginal.clone(),
        slices.slope.clone(),
    ];
    if let Some(range) = slices.score_warp.clone() {
        block_ranges.push(range);
    }
    if let Some(range) = slices.link_dev.clone() {
        block_ranges.push(range);
    }

    assert!((eval.log_likelihood - joint_ll).abs() <= 1e-10);
    assert_eq!(eval.blockworking_sets.len(), block_ranges.len());
    for (work, range) in eval.blockworking_sets.iter().zip(block_ranges.iter()) {
        let BlockWorkingSet::ExactNewton { gradient, hessian } = work else {
            panic!("expected exact-newton block working set");
        };
        let expected_gradient = joint_gradient.slice(s![range.clone()]).to_owned();
        let expected_hessian = joint_hessian
            .slice(s![range.clone(), range.clone()])
            .to_owned();
        assert!(
            max_abs_diff_vec(gradient, &expected_gradient) <= 1e-10,
            "gradient block mismatch"
        );
        assert!(
            max_abs_diff_mat(&hessian.to_dense(), &expected_hessian) <= 1e-10,
            "hessian block mismatch"
        );
    }
}

fn test_family(
    score_warp: Option<DeviationRuntime>,
    link_dev: Option<DeviationRuntime>,
) -> SurvivalMarginalSlopeFamily {
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 2))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 3)))).into(),
        score_warp,
        link_dev,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

#[test]
fn validate_spec_rejects_coordinate_cone_without_guard_offset() {
    let spec = SurvivalMarginalSlopeTermSpec {
        age_entry: array![0.0, 0.0],
        age_exit: array![1.0, 1.0],
        event_target: array![0.0, 1.0],
        weights: array![1.0, 1.0],
        z: array![-1.0, 1.0].insert_axis(Axis(1)),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec: empty_termspec(),
        marginal_offset: Array1::zeros(2),
        frailty: FrailtySpec::None,
        slope_template: SurvivalCovariateTermBlockTemplate::Static,
        derivative_guard: 1e-4,
        baseline_hyper: SurvivalMarginalSlopeBaselineHyperSpec::Linear {
            config: crate::survival::construction::SurvivalBaselineConfig {
                target: crate::survival::construction::SurvivalBaselineTarget::Linear,
                scale: None,
                shape: None,
                rate: None,
                makeham: None,
            },
        },
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
            offset_entry: Array1::zeros(2),
            offset_exit: Array1::zeros(2),
            derivative_offset_exit: Array1::zeros(2),
            ..base_time_block()
        },
        timewiggle_block: None,
        slopespec: empty_termspec(),
        slopespecs: None,
        slope_offset: Array1::zeros(2),
        score_warp: None,
        link_dev: None,
        score_influence_jacobian: None,
        latent_z_policy: LatentZPolicy::default(),
        declared_latent_law: None,
    };

    let err = validate_spec(&spec).expect_err("coordinate cone without guard offset should fail");
    assert!(
        err.contains("coordinate-cone time block requires derivative offset >= guard"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_spec_accepts_learned_gaussian_shift_sigma() {
    // One value for the guard this spec declares AND the offset that must
    // clear it, so the two cannot drift apart again.
    const LEARNED_SHIFT_GUARD: f64 = 1e-4;
    let spec = SurvivalMarginalSlopeTermSpec {
        slope_template: SurvivalCovariateTermBlockTemplate::Static,
        age_entry: array![0.0, 0.0],
        age_exit: array![1.0, 1.0],
        event_target: array![0.0, 1.0],
        weights: array![1.0, 1.0],
        z: array![-1.0, 1.0].insert_axis(Axis(1)),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec: empty_termspec(),
        marginal_offset: Array1::zeros(2),
        frailty: FrailtySpec::GaussianShift {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
        },
        derivative_guard: LEARNED_SHIFT_GUARD,
        baseline_hyper: SurvivalMarginalSlopeBaselineHyperSpec::Linear {
            config: crate::survival::construction::SurvivalBaselineConfig {
                target: crate::survival::construction::SurvivalBaselineTarget::Linear,
                scale: None,
                shape: None,
                rate: None,
                makeham: None,
            },
        },
        // `base_time_block()` is a ONE-row block; this spec carries two data
        // rows, and `validate_spec` requires the time-block designs to have one
        // row per observation. Widen it so the frailty-scale claim under test is
        // what the validation actually reaches.
        //
        // The derivative offset must clear the guard THIS spec declares, not the
        // crate default. `validate_spec` forms A·beta >= derivative_guard -
        // derivative_offset_exit and checks `initial_beta` against it, so with
        // `base_time_block()`'s beta = 0 an offset below the declared guard is
        // infeasible by exactly their difference: carrying the crate default
        // (1e-6) under a declared guard of 1e-4 left slack = -9.900e-5.
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
            offset_entry: Array1::zeros(2),
            offset_exit: Array1::zeros(2),
            derivative_offset_exit: Array1::from_elem(2, LEARNED_SHIFT_GUARD),
            ..base_time_block()
        },
        timewiggle_block: None,
        slopespec: empty_termspec(),
        slopespecs: None,
        slope_offset: Array1::zeros(2),
        score_warp: None,
        link_dev: None,
        score_influence_jacobian: None,
        latent_z_policy: LatentZPolicy::default(),
        declared_latent_law: None,
    };

    let validation = validate_spec(&spec);
    assert!(
        validation.is_ok(),
        "learned GaussianShift scale must be an explicit family coordinate: {validation:?}"
    );
}

#[test]
fn block_slices_handles_link_only_survival_flex_layout() {
    let link_runtime = test_deviation_runtime();
    let family = test_family(None, Some(link_runtime.clone()));
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(3),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];

    let slices = block_slices(&family, &block_states);
    assert!(slices.score_warp.is_none());
    assert_eq!(
        slices.link_dev.as_ref().expect("link-only slice").len(),
        link_runtime.basis_dim()
    );
    assert_eq!(slices.total, 1 + 2 + 3 + link_runtime.basis_dim());
}

#[test]
fn exact_survival_callbacks_lock_the_family_owned_coefficient_chart() {
    use crate::custom_family::BlockEffectiveJacobian;

    let one = Arc::new(Array2::<f64>::ones((2, 1)));
    let time = TimeBlockJacobian::new(Arc::clone(&one), Arc::clone(&one), Arc::clone(&one));
    let marginal = MarginalBlockJacobian::new(Arc::clone(&one));
    let family = test_family(None, None);
    let slope = SlopeBlockJacobian::new(
        family.slope_layout.clone(),
        Arc::clone(&family.z),
        family.score_covariance.clone(),
    )
    .expect("test slope callback");

    assert!(time.locks_raw_width_reduction());
    assert!(marginal.locks_raw_width_reduction());
    assert!(slope.locks_raw_width_reduction());
}

// ── Single-source block-layout parity (#428) ─────────────────────────
//
// `HessBlock` + `BlockHessianAccumulator::block_view` are the one place
// the 5×5 block layout and its transpose relationships live. Every
// assembler (`to_dense`, `diagonal`, operator `mul_vec`, operator
// `bilinear`) is driven by them. These tests pin that machinery against a
// *separately hand-written* fifteen-block scatter so a layout or
// transpose regression in the shared helpers cannot hide behind itself.

/// Build contiguous block slices for the given per-block widths. A flex
/// block with width 0 is absent (`None`) and consumes no coordinates,
/// exactly mirroring how `block_slices` lays out optional deviation blocks.
fn parity_make_slices(
    pt: usize,
    pm: usize,
    pg: usize,
    ph: usize,
    pw: usize,
    pi: usize,
) -> BlockSlices {
    let mut cursor = 0usize;
    let mut take = |n: usize| {
        let r = cursor..cursor + n;
        cursor += n;
        r
    };
    let time = take(pt);
    let marginal = take(pm);
    let slope = take(pg);
    let score_warp = (ph > 0).then(|| take(ph));
    let link_dev = (pw > 0).then(|| take(pw));
    let influence = (pi > 0).then(|| take(pi));
    let total = cursor;
    BlockSlices {
        time,
        marginal,
        slope,
        score_warp,
        link_dev,
        influence,
        total,
    }
}

/// A genuinely-symmetric block (diagonal blocks of a Hessian are symmetric
/// by construction), with a per-block `tag` so cross-block contamination
/// is detectable.
fn parity_sym(n: usize, tag: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        tag + (i as f64 + j as f64) * 0.5 + (i as f64) * (j as f64) * 0.0625
    })
}

/// A general (non-symmetric) off-diagonal block, distinct per `tag` and
/// deliberately asymmetric in (i, j) so a missing/extra transpose shows up.
fn parity_gen(rows: usize, cols: usize, tag: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        tag + (i as f64) * 0.5 + (j as f64) * 0.125
    })
}

fn parity_filled_accumulator(
    pt: usize,
    pm: usize,
    pg: usize,
    ph: usize,
    pw: usize,
    pi: usize,
) -> BlockHessianAccumulator {
    BlockHessianAccumulator {
        h_tt: parity_sym(pt, 1.0),
        h_mm: parity_sym(pm, 2.0),
        h_gg: parity_sym(pg, 3.0),
        h_hh: parity_sym(ph, 4.0),
        h_ww: parity_sym(pw, 5.0),
        h_ii: parity_sym(pi, 6.0),
        h_tm: parity_gen(pt, pm, 10.0),
        h_tg: parity_gen(pt, pg, 11.0),
        h_th: parity_gen(pt, ph, 12.0),
        h_tw: parity_gen(pt, pw, 13.0),
        h_ti: parity_gen(pt, pi, 20.0),
        h_mg: parity_gen(pm, pg, 14.0),
        h_mh: parity_gen(pm, ph, 15.0),
        h_mw: parity_gen(pm, pw, 16.0),
        h_mi: parity_gen(pm, pi, 21.0),
        h_gh: parity_gen(pg, ph, 17.0),
        h_gw: parity_gen(pg, pw, 18.0),
        h_gi: parity_gen(pg, pi, 22.0),
        h_hw: parity_gen(ph, pw, 19.0),
        h_hi: parity_gen(ph, pi, 23.0),
        h_wi: parity_gen(pw, pi, 24.0),
    }
}

/// Independent dense oracle: scatter the fifteen stored blocks by hand,
/// placing each off-diagonal block in its upper position and its explicit
/// transpose in the mirror position. Deliberately avoids `block_view`,
/// `range_of`, and `for_each_offdiagonal_pair`.
fn parity_reference_dense(acc: &BlockHessianAccumulator, sl: &BlockSlices) -> Array2<f64> {
    let mut out = Array2::zeros((sl.total, sl.total));
    out.slice_mut(s![sl.time.clone(), sl.time.clone()])
        .assign(&acc.h_tt);
    out.slice_mut(s![sl.marginal.clone(), sl.marginal.clone()])
        .assign(&acc.h_mm);
    out.slice_mut(s![sl.slope.clone(), sl.slope.clone()])
        .assign(&acc.h_gg);
    if let Some(h) = &sl.score_warp {
        out.slice_mut(s![h.clone(), h.clone()]).assign(&acc.h_hh);
    }
    if let Some(w) = &sl.link_dev {
        out.slice_mut(s![w.clone(), w.clone()]).assign(&acc.h_ww);
    }
    if let Some(i) = &sl.influence {
        out.slice_mut(s![i.clone(), i.clone()]).assign(&acc.h_ii);
    }
    let mut place =
        |r: std::ops::Range<usize>, c: std::ops::Range<usize>, m: ArrayView2<'_, f64>| {
            out.slice_mut(s![r.clone(), c.clone()]).assign(&m);
            out.slice_mut(s![c, r]).assign(&m.t());
        };
    place(sl.time.clone(), sl.marginal.clone(), acc.h_tm.view());
    place(sl.time.clone(), sl.slope.clone(), acc.h_tg.view());
    place(sl.marginal.clone(), sl.slope.clone(), acc.h_mg.view());
    if let Some(h) = &sl.score_warp {
        place(sl.time.clone(), h.clone(), acc.h_th.view());
        place(sl.marginal.clone(), h.clone(), acc.h_mh.view());
        place(sl.slope.clone(), h.clone(), acc.h_gh.view());
    }
    if let Some(w) = &sl.link_dev {
        place(sl.time.clone(), w.clone(), acc.h_tw.view());
        place(sl.marginal.clone(), w.clone(), acc.h_mw.view());
        place(sl.slope.clone(), w.clone(), acc.h_gw.view());
    }
    if let (Some(h), Some(w)) = (&sl.score_warp, &sl.link_dev) {
        place(h.clone(), w.clone(), acc.h_hw.view());
    }
    if let Some(i) = &sl.influence {
        place(sl.time.clone(), i.clone(), acc.h_ti.view());
        place(sl.marginal.clone(), i.clone(), acc.h_mi.view());
        place(sl.slope.clone(), i.clone(), acc.h_gi.view());
        if let Some(h) = &sl.score_warp {
            place(h.clone(), i.clone(), acc.h_hi.view());
        }
        if let Some(w) = &sl.link_dev {
            place(w.clone(), i.clone(), acc.h_wi.view());
        }
    }
    out
}

const PARITY_LAYOUTS: [(usize, usize, usize, usize, usize, usize); 7] = [
    (2, 3, 2, 4, 3, 0), // full flex, no absorber
    (2, 3, 2, 0, 0, 0), // rigid: no flex blocks
    (2, 3, 2, 4, 0, 0), // score-warp only
    (2, 3, 2, 0, 3, 0), // link-deviation only
    (2, 3, 2, 0, 0, 5), // absorber only (no flex)
    (2, 3, 2, 4, 3, 5), // full flex + absorber (#461)
    (2, 3, 2, 4, 0, 5), // score-warp + absorber
];

#[test]
fn block_to_dense_matches_hand_scatter_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let got = acc.to_dense(&sl);
        let want = parity_reference_dense(&acc, &sl);
        assert_eq!(
            got,
            want,
            "to_dense diverged from hand scatter for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_diagonal_matches_dense_diagonal_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let got = acc.diagonal(&sl);
        let want = parity_reference_dense(&acc, &sl).diag().to_owned();
        assert_eq!(
            got,
            want,
            "diagonal diverged from dense diagonal for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_operator_matvec_matches_dense_gemv() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let dense = parity_reference_dense(&acc, &sl);
        let v = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.37).sin());
        let op = acc.into_operator(sl.clone());
        let got = op.mul_vec(&v);
        let want = dense.dot(&v);
        assert_relative_eq!(
            got.as_slice().unwrap(),
            want.as_slice().unwrap(),
            max_relative = 1e-12,
            epsilon = 1e-12
        );
    }
}

#[test]
fn block_operator_bilinear_matches_dense_quadratic_form() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let dense = parity_reference_dense(&acc, &sl);
        let v = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.37).sin());
        let u = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.53).cos());
        let want = v.dot(&dense.dot(&u));
        let op = acc.into_operator(sl.clone());
        let got = op.bilinear(&v, &u);
        assert_relative_eq!(got, want, max_relative = 1e-12, epsilon = 1e-12);
    }
}

#[test]
fn block_operator_dense_matches_accumulator_dense_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let direct = acc.to_dense(&sl);
        let op = acc.into_operator(sl.clone());
        let via_op = op.to_dense();
        assert_eq!(
            direct,
            via_op,
            "operator to_dense diverged from accumulator to_dense for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_view_is_transpose_symmetric_across_present_pairs() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        for a in HessBlock::ALL {
            if sl.range_of(a).is_none() {
                continue;
            }
            for b in HessBlock::ALL {
                if sl.range_of(b).is_none() {
                    continue;
                }
                let ab = acc.block_view(a, b).to_owned();
                let ba_t = acc.block_view(b, a).t().to_owned();
                assert_eq!(
                    ab, ba_t,
                    "block_view({a:?},{b:?}) != block_view({b:?},{a:?})^T"
                );
            }
        }
    }
}

#[test]
fn exact_flex_row_matches_rigid_closed_form_without_deviations() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.7]),
        z: Arc::new(array![0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![0.2]),
        offset_exit: Arc::new(array![0.4]),
        derivative_offset_exit: Arc::new(array![0.8]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.6],
        },
    ];
    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("row geometry");
    let primary = flex_primary_slices(&family);
    let (nll_exact, grad_exact, hess_exact) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("exact flex row");
    let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
        q_geom.q0,
        q_geom.q1,
        q_geom.qd1,
        block_states[2].eta[0],
        family.z[[0, 0]],
        family.shared_slope_covariance_scale(0),
        family.weights[0],
        family.entry_weight(0),
        family.event[0],
        family.derivative_guard,
        family.probit_frailty_scale(),
    )
    .expect("rigid row");

    assert!((nll_exact - nll_rigid).abs() < 1e-10);
    for idx in 0..N_PRIMARY {
        assert!((grad_exact[idx] - grad_rigid[idx]).abs() < 1e-8);
    }
    for i in 0..N_PRIMARY {
        for j in 0..N_PRIMARY {
            assert!((hess_exact[[i, j]] - hess_rigid[i][j]).abs() < 1e-7);
        }
    }
}

#[test]
fn row_primary_closed_form_rejects_negative_infinite_signed_margin() {
    let err = row_primary_closed_form(f64::INFINITY, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1e-6, 1.0)
        .expect_err("exact closed-form row should reject -inf signed margins");
    assert!(err.contains("non-finite signed margin"));
}

/// Mechanism-of-ρ=2 proof for the inner-PIRLS pathology on large-scale
/// saturated probit fits.
///
/// `add_pullback_primary_hessian` (line 9753) sums `h[0,0] + h[1,1]`
/// over rows into the marginal-block joint Hessian (β_marg enters both
/// `q0` and `q1` with the SAME `marginal_design` row, so the Jacobian
/// pullback adds the q0 and q1 second derivatives at the same slot).
///
/// Mathematical facts pinned here:
///   - Censored rows: `h[0,0] + h[1,1] = 0` to ULP at every η ≥ 0.
///     The entry term `+w·log Φ(−η₀)` (concave, curvature −w·c²) and
///     the exit term `−w·log Φ(−η₁)` (convex, curvature +w·c²) have
///     equal-and-opposite second derivatives when q₀ = q₁.
///   - Event rows: residual `+w·c²·(1/η² + O(1/η⁴))` from the Mills
///     asymptotic. The event-density term `+w·η₁²/2` contributes
///     exactly `+w·c²`; the entry-survival term contributes
///     `−w·c²·(1 − 1/η²)`. Sum = `+w·c²/η²`.
///
/// At large-scale saturation (η ~ 988), the marginal-block joint Hessian
/// collapses to `O(w/η²) = O(1e-6)` per event row from the likelihood
/// side; censored contributions are 0 to ULP. The Newton step in that
/// block is then dominated by the smoothing penalty `S_marg`. When
/// the saturating direction lies in the null space of `S_marg`
/// (typical for the duchon-smooth's polynomial null space), the
/// effective curvature drops to the f64 ridge floor, the inner
/// Newton step is set by ridge alone, and `actual = rhs·δ` while
/// `predicted = ½·rhs·δ` — yielding ρ ≡ 2 to floating-point precision
/// as observed.
#[test]
fn marginal_block_hessian_cancels_in_saturated_regime() {
    let probit_scale = 1.0_f64;
    let w = 1.0_f64;
    let derivative_guard = 1e-6;
    let qd1 = 1.0_f64;
    let g = 0.0_f64;
    let z = 0.0_f64;

    // Censored rows, q0 = q1 = η, at a wide range of saturations:
    // cancellation must be ULP-exact for every η.
    for &eta in &[0.5_f64, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0, 500.0, 988.0] {
        let (_nll, _grad, hess) =
            row_primary_closed_form(eta, eta, qd1, g, z, 1.0, w, w, 0.0, derivative_guard, probit_scale)
                .expect("rigid censored row");
        let sum = hess[0][0] + hess[1][1];
        assert!(
            sum.abs() <= 1e-12 * (hess[0][0].abs() + hess[1][1].abs()).max(1.0),
            "censored cancellation broke at η={eta}: h[0,0]={:.3e} h[1,1]={:.3e} sum={:.3e}",
            hess[0][0],
            hess[1][1],
            sum,
        );
    }

    // Event rows, q0 = q1 = η, deep saturation: residual scales as
    // 1/η² by Mills asymptotic M(−η) = η + 1/η + O(1/η³).
    for &eta in &[40.0_f64, 100.0, 500.0, 988.0] {
        let (_nll, _grad, hess) =
            row_primary_closed_form(eta, eta, qd1, g, z, 1.0, w, w, 1.0, derivative_guard, probit_scale)
                .expect("rigid event row");
        let sum = hess[0][0] + hess[1][1];
        let bound = 2.0 / (eta * eta);
        assert!(
            sum > 0.0 && sum <= bound,
            "event cancellation residual at η={eta}: sum={:.3e} expected (0, {:.3e}]",
            sum,
            bound,
        );
    }

    // Cross-check at η = 988 (the user's large-scale saturation):
    // both kinds of rows hit the predicted floor exactly.
    let (_, _, ev) =
        row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1e-6, 1.0).unwrap();
    let (_, _, ce) =
        row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1e-6, 1.0).unwrap();
    let ev_sum = ev[0][0] + ev[1][1];
    let ce_sum = ce[0][0] + ce[1][1];
    assert!(
        ev_sum > 0.0 && ev_sum < 2.0e-6,
        "event saturated h[0,0]+h[1,1] = {ev_sum:.3e}, expected ~1/988² ≈ 1e-6",
    );
    assert_eq!(
        ce_sum, 0.0,
        "censored saturated h[0,0]+h[1,1] must be EXACTLY 0, got {ce_sum:.3e}",
    );
}

#[test]
fn row_primary_closed_form_rejects_nan_signed_margin() {
    let err = row_primary_closed_form(f64::NAN, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1e-6, 1.0)
        .expect_err("exact closed-form row should reject NaN signed margins");
    assert!(err.contains("non-finite signed margin"));
}

#[test]
fn rigid_row_kernel_propagates_invalid_nonfinite_signed_margin_errors() {
    let mut family = test_family(None, None);
    family.offset_entry = Arc::new(array![f64::INFINITY]);
    family.offset_exit = Arc::new(array![0.0]);
    family.derivative_offset_exit = Arc::new(array![1.0]);
    family.event = Arc::new(array![1.0]);

    let kernel = SurvivalMarginalSlopeRowKernel::new(
        family,
        vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(3),
                eta: Array1::zeros(1),
            },
        ],
    );

    let err =
        <SurvivalMarginalSlopeRowKernel<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry> as crate::row_kernel::RowKernel<STATIC_SLOPE_PRIMARIES>>::row_kernel(
            &kernel, 0,
        )
            .expect_err("row kernel should propagate exact probit boundary failures");
    assert!(err.contains("non-finite signed margin"));
}

#[test]
fn rigid_row_kernel_propagates_nan_signed_margin_errors() {
    let mut family = test_family(None, None);
    family.offset_entry = Arc::new(array![f64::NAN]);
    family.offset_exit = Arc::new(array![0.0]);
    family.derivative_offset_exit = Arc::new(array![1.0]);
    family.event = Arc::new(array![1.0]);

    let kernel = SurvivalMarginalSlopeRowKernel::new(
        family,
        vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(3),
                eta: Array1::zeros(1),
            },
        ],
    );

    let err =
        <SurvivalMarginalSlopeRowKernel<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry> as crate::row_kernel::RowKernel<STATIC_SLOPE_PRIMARIES>>::row_kernel(
            &kernel, 0,
        )
            .expect_err("row kernel should propagate NaN probit boundary failures");
    assert!(err.contains("non-finite signed margin"));
}

/// The single-expression Taylor-jet tower (#932) of the rigid K=1
/// survival marginal-slope row NLL, written ONCE over generic jet
/// primaries `(q0, q1, qd1, g)`. It reuses the family's OWN hand-certified
/// `[f64; 5]` special-function derivative stacks (`unary_derivatives_sqrt`
/// / `_neglog_phi` / `_log_normal_pdf` / `_log`) through
/// `JetScalar::compose_unary`, so no probit/log primitive is re-derived here:
/// the tower mechanizes only the Leibniz / Faà di Bruno composition that
/// `row_primary_closed_form` previously coded by hand (where #736's
/// cross-block sign flip lived). The value channel of the returned tower
/// is the row NLL expression, so every derivative channel is exact by
/// construction.
struct SurvivalMarginalSlopeRigidNllProgram {
    primaries: Vec<[f64; 4]>,
    z: Vec<f64>,
    w: Vec<f64>,
    d: Vec<f64>,
    probit_scale: f64,
}

impl gam_math::jet_tower::RowProgram<4> for SurvivalMarginalSlopeRigidNllProgram {
    fn n_rows(&self) -> usize {
        self.primaries.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        self.primaries
            .get(row)
            .copied()
            .ok_or_else(|| format!("rigid nll program: row {row} out of range"))
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<4>>(
        &self,
        row: usize,
        p: &[S; 4],
    ) -> Result<S, String> {
        let z = *self
            .z
            .get(row)
            .ok_or_else(|| format!("rigid nll program: z row {row} out of range"))?;
        let w = self.w[row];
        let d = self.d[row];
        let s_f = self.probit_scale;
        let q0 = p[0];
        let q1 = p[1];
        let qd1 = p[2];
        let g = p[3];

        // c(g) = sqrt(1 + (s_f g)^2)  — K=1 covariance_ones = 1, exactly the
        // shared MultiDirJet `one_plus_b2 -> sqrt` composition.
        let observed_g = g.scale(s_f);
        let one_plus_b2 = observed_g.mul(&observed_g).add(&S::constant(1.0));
        let c = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.value()));

        let eta0 = q0.mul(&c).add(&observed_g.scale(z));
        let eta1 = q1.mul(&c).add(&observed_g.scale(z));
        let ad1 = qd1.mul(&c);

        // Entry survival: +w logΦ(-η0) = -1 * (-w logΦ(-η0)).
        let neg_eta0 = eta0.neg();
        let entry = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.value(), w))
            .scale(-1.0);
        // Exit survival: (1-d) * (-w logΦ(-η1)) carried with weight w(1-d).
        let neg_eta1 = eta1.neg();
        let exit = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.value(),
            w * (1.0 - d),
        ));
        // Event density: -w d logφ(η1).
        let event_density = if d > 0.0 {
            eta1.compose_unary(unary_derivatives_log_normal_pdf(eta1.value()))
                .scale(-w * d)
        } else {
            S::constant(0.0)
        };
        // Time derivative: -w d log(ad1).
        let time_deriv = if d > 0.0 {
            ad1.compose_unary(unary_derivatives_log(ad1.value()))
                .scale(-w * d)
        } else {
            S::constant(0.0)
        };

        Ok(exit.add(&entry).add(&event_density).add(&time_deriv))
    }
}

/// Build a rigid K=1 survival marginal-slope family whose entry/exit/
/// derivative/marginal/slope designs are all nontrivial dense blocks,
/// so every one of the four primaries `(q0, q1, qd1, g)` is exercised
/// when the kernel reads its designs. `n` rows, `event` per row.
fn oracle_rigid_family(
    n: usize,
    z: &[f64],
    weights: &[f64],
    event: &[f64],
    gaussian_frailty_sd: Option<f64>,
) -> SurvivalMarginalSlopeFamily {
    let z_col = Array2::from_shape_fn((n, 1), |(r, _)| z[r]);
    // Distinct entry/exit/derivative rows so q0 != q1 != qd1.
    let design_entry = Array2::from_shape_fn((n, 1), |(r, _)| {
        0.4 + 0.13 * (r as f64) - 0.05 * (r as f64).cos()
    });
    let design_exit = Array2::from_shape_fn((n, 1), |(r, _)| {
        0.9 + 0.07 * (r as f64) + 0.04 * (r as f64).sin()
    });
    // Strictly positive derivative-exit design so qd1 > 0 (monotone).
    let design_deriv = Array2::from_shape_fn((n, 1), |(r, _)| 1.2 + 0.21 * (r as f64).abs().sqrt());
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(Array1::from(event.to_vec())),
        weights: Arc::new(Array1::from(weights.to_vec())),
        z: Arc::new(z_col),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(design_entry),
        design_exit: DesignMatrix::from(design_exit),
        design_derivative_exit: DesignMatrix::from(design_deriv),
        offset_entry: Arc::new(Array1::from_shape_fn(n, |r| 0.05 * (r as f64) - 0.2)),
        offset_exit: Arc::new(Array1::from_shape_fn(n, |r| 0.15 - 0.03 * (r as f64))),
        derivative_offset_exit: Arc::new(Array1::from_elem(n, 0.0)),
        marginal_design: DesignMatrix::from(Array2::zeros((n, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((n, 0)))).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

/// #932 dedicated correctness oracle for the rigid family-direction
/// lowerings (`rigid_family_direction_terms` /
/// `rigid_family_direction_beta_drift` in `eval_family.rs`) — previously the
/// only jet-derived production lowerings of this family covered solely
/// transitively (through the ψ-terms integration tests).
///
/// The Dual2-over-`Order2` output must equal a central finite-difference
/// witness of the direct symbolic order-2 lowering (`rigid_row_order2`, a
/// different code path from the nested-dual instantiation) along the
/// quadratic primary path `p(t) = p + t·first + (t²/2)·second`:
/// the first channel is `d/dt|₀` and the second channel is `d²/dt²|₀` of
/// every (objective, gradient, Hessian) entry. The beta-drift lowering must
/// equal a central FD over realizable block-state shifts (marginal-η,
/// slope-η, and time-β directions) of the first-channel terms themselves,
/// pinning the `OneSeed::eps` wiring end-to-end through the production
/// helper.
#[test]
fn rigid_family_direction_terms_match_fd_witness_932() {
    use super::row_kernel::{rigid_row_inputs, rigid_row_kernel_primaries, rigid_row_order2};

    let n = 5;
    let z = [0.4, -1.1, 0.0, 0.7, -0.3];
    let weights = [1.0, 0.8, 1.3, 0.9, 1.1];
    let event = [1.0, 0.0, 0.0, 1.0, 1.0];
    let g_eta = array![0.2, -0.5, 0.35, -0.15, 0.6];
    let marginal_eta = array![0.1, -0.2, 0.05, 0.12, -0.08];

    let first = [0.3_f64, -0.6, 0.4, 0.7];
    let second = [-0.2_f64, 0.5, 0.1, -0.3];

    let close = |label: &str, got: f64, want: f64, band_scale: f64| {
        let band = band_scale * got.abs().max(want.abs()).max(1.0);
        assert!(
            got.is_finite() && want.is_finite() && (got - want).abs() <= band,
            "{label}: production={got:+.12e} fd={want:+.12e} band={band:.3e}"
        );
    };

    for frailty in [None, Some(0.6_f64)] {
        let family = oracle_rigid_family(n, &z, &weights, &event, frailty);
        let beta_time = array![0.85];
        let block_states = vec![
            ParameterBlockState {
                beta: beta_time.clone(),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: marginal_eta.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: g_eta.clone(),
            },
        ];

        for row in 0..n {
            let (terms_first, terms_second) = family
                .rigid_family_direction_terms::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                    row,
                    &block_states,
                    first,
                    second,
                )
                .expect("family-direction terms");

            let p = rigid_row_kernel_primaries::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                &family,
                &block_states,
                row,
            )
            .expect("primaries");
            let inputs = rigid_row_inputs(&family, &block_states, row, "family-direction oracle")
                .expect("row inputs");
            let at = |t: f64| -> (f64, [f64; 4], [[f64; 4]; 4]) {
                let shifted: [f64; 4] =
                    std::array::from_fn(|a| p[a] + t * first[a] + 0.5 * t * t * second[a]);
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(&shifted, &inputs)
                    .expect("shifted symbolic row")
            };

            // First channel: d/dt at 0 (central, h=1e-6).
            let h1 = 1.0e-6;
            let (plus_v, plus_g, plus_h) = at(h1);
            let (minus_v, minus_g, minus_h) = at(-h1);
            close(
                &format!("row {row} first objective"),
                terms_first.objective,
                (plus_v - minus_v) / (2.0 * h1),
                1.0e-7,
            );
            for a in 0..4 {
                close(
                    &format!("row {row} first gradient[{a}]"),
                    terms_first.gradient[a],
                    (plus_g[a] - minus_g[a]) / (2.0 * h1),
                    1.0e-7,
                );
                for b in 0..4 {
                    close(
                        &format!("row {row} first hessian[{a},{b}]"),
                        terms_first.hessian[[a, b]],
                        (plus_h[a][b] - minus_h[a][b]) / (2.0 * h1),
                        1.0e-7,
                    );
                }
            }

            // Second channel: d²/dt² at 0 (central, h=1e-4).
            let h2 = 1.0e-4;
            let (p2v, p2g, p2h) = at(h2);
            let (m2v, m2g, m2h) = at(-h2);
            let (zv, zg, zh) = at(0.0);
            close(
                &format!("row {row} second objective"),
                terms_second.objective,
                (p2v - 2.0 * zv + m2v) / (h2 * h2),
                5.0e-5,
            );
            for a in 0..4 {
                close(
                    &format!("row {row} second gradient[{a}]"),
                    terms_second.gradient[a],
                    (p2g[a] - 2.0 * zg[a] + m2g[a]) / (h2 * h2),
                    5.0e-5,
                );
                for b in 0..4 {
                    close(
                        &format!("row {row} second hessian[{a},{b}]"),
                        terms_second.hessian[[a, b]],
                        (p2h[a][b] - 2.0 * zh[a][b] + m2h[a][b]) / (h2 * h2),
                        5.0e-5,
                    );
                }
            }

            // Beta-drift channel: for each realizable block-state direction,
            // production `OneSeed::eps` must equal the central FD of the
            // first-channel terms across the shifted states.
            let probes: [(&str, usize, [f64; 4]); 3] = [
                ("marginal-eta", 1, [1.0, 1.0, 0.0, 0.0]),
                ("slope-eta", 2, [0.0, 0.0, 0.0, 1.0]),
                (
                    "time-beta",
                    0,
                    [
                        family.design_entry.dot_row(row, &array![1.0]),
                        family.design_exit.dot_row(row, &array![1.0]),
                        family.design_derivative_exit.dot_row(row, &array![1.0]),
                        0.0,
                    ],
                ),
            ];
            for (label, block, beta_dir) in probes {
                let drift = family
                    .rigid_family_direction_beta_drift::<
                        STATIC_SLOPE_PRIMARIES,
                        StaticSlopeGeometry,
                    >(row, &block_states, first, beta_dir)
                    .expect("family-direction beta drift");
                let hs = 1.0e-5;
                let shifted_terms = |s: f64| {
                    let mut states = block_states.clone();
                    if block == 0 {
                        states[0].beta[0] += s;
                    } else {
                        states[block].eta[row] += s;
                    }
                    family
                        .rigid_family_direction_terms::<
                            STATIC_SLOPE_PRIMARIES,
                            StaticSlopeGeometry,
                        >(row, &states, first, [0.0; STATIC_SLOPE_PRIMARIES])
                        .expect("shifted family-direction terms")
                        .0
                };
                let plus = shifted_terms(hs);
                let minus = shifted_terms(-hs);
                close(
                    &format!("row {row} {label} drift objective"),
                    drift.objective,
                    (plus.objective - minus.objective) / (2.0 * hs),
                    1.0e-6,
                );
                for a in 0..4 {
                    close(
                        &format!("row {row} {label} drift gradient[{a}]"),
                        drift.gradient[a],
                        (plus.gradient[a] - minus.gradient[a]) / (2.0 * hs),
                        1.0e-6,
                    );
                    for b in 0..4 {
                        close(
                            &format!("row {row} {label} drift hessian[{a},{b}]"),
                            drift.hessian[[a, b]],
                            (plus.hessian[[a, b]] - minus.hessian[[a, b]]) / (2.0 * hs),
                            1.0e-6,
                        );
                    }
                }
            }
        }
    }
}

/// #932 unified-feature oracle: the scalar/shared 5->4 order-two pullback must
/// agree with the generic `Order2<4>` feature-map evaluation on every channel,
/// and its admission wrapper must agree with the dependency-sliced witness
/// surface. The random grid covers both event branches, non-unit covariance,
/// and non-unit frailty scale from the sole `rigid_feature_program` declaration.
// The witness half of this fixture calls `rigid_row_admission_witnesses`, which
// is `cfg(target_os = "linux")` production code, so off-Linux the test target
// does not compile at all. Gate the fixture with the surface it exercises.
#[cfg(target_os = "linux")]
#[test]
fn rigid_feature_program_scalar_pullback_matches_generic_and_witnesses_932() {
    use gam_math::jet_scalar::{JetScalar, Order2};

    // Deterministic xorshift grid (no RNG dependency).
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut nx = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut max_rel = 0.0_f64;
    for _ in 0..4000 {
        let p = [nx() * 1.5, nx() * 1.5, 0.5 + nx().abs() * 2.0, nx() * 1.2];
        let inputs = {
            let wi = 0.5 + nx().abs();
            RigidRowInputs {
                row: 0,
                wi,
                wi_entry: wi,
                di: if nx() > 0.0 { 1.0 } else { 0.0 },
                z_sum: nx() * 1.2,
                covariance_ones: 0.7 + nx().abs(),
                probit_scale: 0.6 + nx().abs(),
                qd1_lower: -1.0,
                anchor: None,
            }
        };

        let dense_vars: [Order2<4>; 4] = std::array::from_fn(|a| Order2::variable(p[a], a));
        let dense =
            rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&dense_vars, &inputs)
                .expect("generic scalar feature map");
        let (value, gradient, hessian) =
            rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(&p, &inputs)
                .expect("scalar feature pullback");
        let observed_g = inputs.probit_scale * p[3];
        let (_, _, _, semantic_witnesses) = rigid_feature_frame_order2(
            &static_slope_feature_frame(
                p[0],
                p[1],
                p[2],
                observed_g * inputs.z_sum,
                (p[3] * p[3]) * inputs.covariance_ones,
                0.0,
            ),
            inputs.wi,
            inputs.wi_entry,
            inputs.di,
            inputs.probit_scale,
            follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(),
        );
        let sliced_witnesses = rigid_row_admission_witnesses::<
            STATIC_SLOPE_PRIMARIES,
            StaticSlopeGeometry,
        >(&p, &inputs);

        let mut check = |a: f64, b: f64| {
            let rel = (a - b).abs() / (1.0 + a.abs().max(b.abs()));
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                (a - b).abs() <= 1e-12 + 1e-12 * a.abs().max(b.abs()),
                "generated vs generic channel disagreement: {a:+.16e} vs {b:+.16e}"
            );
        };
        check(value, dense.value());
        for a in 0..4 {
            check(gradient[a], dense.g()[a]);
            for b in 0..4 {
                check(hessian[a][b], dense.h()[a][b]);
            }
        }
        for witness in 0..3 {
            check(semantic_witnesses[witness], sliced_witnesses[witness]);
        }
    }
    assert!(
        max_rel <= 1e-12,
        "generated vs generic max relative error {max_rel:.3e} exceeds 1e-12"
    );
}

#[test]
fn exact_flex_row_value_matches_rigid_with_zero_score_and_link_coefficients() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![0.9]),
        z: Arc::new(array![-0.35].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![-0.1]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.6]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.45],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("row geometry");
    let primary = flex_primary_slices(&family);
    let (nll_exact, grad_exact, hess_exact) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("exact flex row");
    let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
        q_geom.q0,
        q_geom.q1,
        q_geom.qd1,
        block_states[2].eta[0],
        family.z[[0, 0]],
        family.shared_slope_covariance_scale(0),
        family.weights[0],
        family.entry_weight(0),
        family.event[0],
        family.derivative_guard,
        family.probit_frailty_scale(),
    )
    .expect("rigid row");

    assert!((nll_exact - nll_rigid).abs() < 1e-10);
    assert!((grad_exact[primary.q0] - grad_rigid[0]).abs() < 1e-8);
    assert!((grad_exact[primary.q1] - grad_rigid[1]).abs() < 1e-8);
    assert!((grad_exact[primary.qd1] - grad_rigid[2]).abs() < 1e-8);
    assert!((grad_exact[primary.g] - grad_rigid[3]).abs() < 1e-8);
    assert!((hess_exact[[primary.q0, primary.q0]] - hess_rigid[0][0]).abs() < 1e-7);
    assert!((hess_exact[[primary.q0, primary.g]] - hess_rigid[0][3]).abs() < 1e-7);
    assert!((hess_exact[[primary.q1, primary.q1]] - hess_rigid[1][1]).abs() < 1e-7);
    assert!((hess_exact[[primary.q1, primary.g]] - hess_rigid[1][3]).abs() < 1e-7);
    assert!((hess_exact[[primary.qd1, primary.qd1]] - hess_rigid[2][2]).abs() < 1e-7);
    assert!((hess_exact[[primary.g, primary.g]] - hess_rigid[3][3]).abs() < 1e-7);
}

/// gam#932/#979: INDEPENDENT witness for the survival marginal-slope FLEX
/// higher-order tower (`row_flex_primary_{third,fourth}_contracted_exact`).
///
/// The rigid K=1 path is independently guarded by
/// `SurvivalMarginalSlopeRigidNllProgram` (a single-expression `Tower4` algebra
/// re-derivation, no shared jet code); the flex path was not.
///
/// This closes that gap on the `(q0, q1, qd1, g)` primary block: at ZERO
/// deviation coefficients the flex de-nested calibration / cell-moment / intercept
/// machinery is fully exercised (it does NOT short-circuit to the rigid kernel —
/// it runs the partition with zero-coefficient cubics), and its third/fourth
/// directional contractions over `(q0, q1, qd1, g)` MUST equal the independent
/// rigid `Tower4` truth. A planted cross-block sign flip must leave the band,
/// proving resolving power (the #736 genus the shared-input parity cannot catch).
#[test]
fn flex_contracted_tower_matches_independent_rigid_tower_and_catches_sign_flip() {
    use gam_math::jet_tower::{program_fourth_contracted, program_third_contracted};

    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    // Several fixture rows: events + censored, distinct q-geometry, frailty off
    // (probit scale = 1; the rigid program and the flex path share the closed
    // form there) and a non-zero slope g so the c(g) coupling is live.
    struct Fix {
        event: f64,
        weight: f64,
        z: f64,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
    }
    let fixtures = [
        Fix {
            event: 1.0,
            weight: 0.75,
            z: -0.2,
            q0: -0.4,
            q1: 0.6,
            qd1: 0.85,
            g: 0.32,
        },
        Fix {
            event: 0.0,
            weight: 1.35,
            z: -1.15,
            q0: -1.35,
            q1: -0.9,
            qd1: 0.42,
            g: -0.55,
        },
        Fix {
            event: 1.0,
            weight: 0.9,
            z: 0.7,
            q0: 0.15,
            q1: 1.05,
            qd1: 0.6,
            g: 0.45,
        },
    ];

    for fix in &fixtures {
        let family = SurvivalMarginalSlopeFamily {
            jeffreys_armed: true,
            latent_law: None,
            n: 1,
            entry_at_origin: Arc::new(Array1::from_elem(1, false)),
            event: Arc::new(array![fix.event]),
            weights: Arc::new(array![fix.weight]),
            z: Arc::new(array![fix.z].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![fix.q0]),
            offset_exit: Arc::new(array![fix.q1]),
            derivative_offset_exit: Arc::new(array![fix.qd1]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            flex_jet_arenas: new_flex_jet_arena_pool(),
        };
        // ZERO deviation coefficients: the flex calculus runs in full, but the
        // primary NLL reduces to the rigid closed form so the rigid Tower4 is the
        // exact independent truth on the (q0,q1,qd1,g) block.
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![fix.g],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];

        let primary = flex_primary_slices(&family);
        let p = primary.total;
        // The four time/marginal/slope primaries occupy the leading slots.
        let block_idx = [primary.q0, primary.q1, primary.qd1, primary.g];

        // Independent rigid Tower4 program at the SAME primaries.
        let program = SurvivalMarginalSlopeRigidNllProgram {
            primaries: vec![[fix.q0, fix.q1, fix.qd1, fix.g]],
            z: vec![fix.z],
            w: vec![fix.weight],
            d: vec![fix.event],
            probit_scale: family.probit_frailty_scale(),
        };

        // Distinct 4-vector directions confined to (q0,q1,qd1,g); embed each into
        // the full p-vector at the leading slots for the flex call.
        let dirs4: [[f64; 4]; 3] = [
            [0.7, -1.3, 0.5, 0.9],
            [-0.4, 0.6, -1.1, 0.3],
            [1.2, 0.2, -0.7, -0.5],
        ];
        let embed = |d4: &[f64; 4]| -> Array1<f64> {
            let mut full = Array1::zeros(p);
            for (k, &slot) in block_idx.iter().enumerate() {
                full[slot] = d4[k];
            }
            full
        };

        // ── Third-order: D_dir H over the (q0,q1,qd1,g) block ───────────────
        for d4 in &dirs4 {
            let flex_full = family
                .row_flex_primary_third_contracted_exact(0, &block_states, &embed(d4))
                .expect("flex third contracted at zero deviation");
            let rigid = program_third_contracted(&program, 0, d4).expect("rigid third");
            let scale = rigid
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (u, &bu) in block_idx.iter().enumerate() {
                for (v, &bv) in block_idx.iter().enumerate() {
                    let got = flex_full[[bu, bv]];
                    let want = rigid[u][v];
                    assert!(
                        (got - want).abs() <= 1e-7 * scale,
                        "third[{u},{v}] flex {got:+.9e} != independent rigid tower {want:+.9e} (z={}, event={})",
                        fix.z,
                        fix.event
                    );
                }
            }
        }

        // Planted sign-flip tripwire on a representative third-order cross block.
        {
            let d4 = &dirs4[0];
            let flex_full = family
                .row_flex_primary_third_contracted_exact(0, &block_states, &embed(d4))
                .expect("flex third contracted (tripwire)");
            let rigid = program_third_contracted(&program, 0, d4).expect("rigid third (tripwire)");
            // (q0, g) cross block — the marginal↔slope coupling.
            let want = rigid[0][3];
            let scale = want.abs().max(1.0);
            if want.abs() > 1e-6 {
                let flipped = -flex_full[[primary.q0, primary.g]];
                assert!(
                    (flipped - want).abs() > 1e-7 * scale,
                    "independent rigid tower failed to reject a planted (q0,g) sign flip: flipped {flipped:+.9e} vs truth {want:+.9e}"
                );
            }
        }

        // ── Fourth-order: D_u D_v H over the (q0,q1,qd1,g) block ─────────────
        let quad_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for &(iu, iv) in &quad_pairs {
            let du = &dirs4[iu];
            let dv = &dirs4[iv];
            let flex_full = family
                .row_flex_primary_fourth_contracted_exact(0, &block_states, &embed(du), &embed(dv))
                .expect("flex fourth contracted at zero deviation");
            let rigid = program_fourth_contracted(&program, 0, du, dv).expect("rigid fourth");
            let scale = rigid
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (u, &bu) in block_idx.iter().enumerate() {
                for (v, &bv) in block_idx.iter().enumerate() {
                    let got = flex_full[[bu, bv]];
                    let want = rigid[u][v];
                    assert!(
                        (got - want).abs() <= 1e-6 * scale,
                        "fourth[{u},{v}] flex {got:+.9e} != independent rigid tower {want:+.9e} (z={}, event={})",
                        fix.z,
                        fix.event
                    );
                }
            }
        }
    }
}

/// gam#932/#979: INDEPENDENT finite-difference witness for the survival
/// marginal-slope FLEX higher-order tower with NON-ZERO deviation coefficients —
/// the part Arm A (`flex_contracted_tower_matches_independent_rigid_tower_*`)
/// cannot reach, and the part most likely to harbor a shared-input bug.
///
/// The witness re-derives the scalar survival flex row NLL FROM SCRATCH over the
/// primary vector `p = [q0, q1, qd1, g, β_h..., β_w...]`:
///   * the deviation index `index(a, g, z) = a + g·warp(z) + linkdev(a + g·z)`
///     is reconstructed from the RAW basis matrices (`DeviationRuntime::design`),
///     dotted with the coefficients — no production jet / cell-moment code;
///   * the per-timepoint intercept `a(q)` is the calibration root
///     `∫ Φ(−index(a,g,z))·φ(z) dz = Φ(−q)`, solved by an independent secant on
///     a fine composite-Simpson quadrature of the latent normal, with the density
///     normalization `D = |∂F/∂a|` from the same quadrature;
///   * the NLL is assembled from the closed-form survival pieces
///     `w·[logΦ(−η0) − (1−d)logΦ(−η1) − d·logφ(η1) − d·log χ1 − d·logφ(q1)
///        + d·log D1 − d·log qd1]`.
///
/// To DE-RISK the witness (a witness-side re-derivation error would masquerade as
/// a production disagreement), the witness scalar NLL is first self-validated
/// against the production `row_neglog_flex_value` at the SAME non-zero β; only
/// then is it finite-differenced (Richardson O(h⁴)) and compared to the
/// production `row_flex_primary_{third,fourth}_contracted_exact`. A planted sign
/// flip on a deviation-touching cross block must leave the band.
#[test]
fn flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();

    // No frailty ⇒ probit scale = 1 ⇒ the calibration measure is the standard
    // normal latent and the index is unscaled — the regime the witness derives.
    let z_row = 0.3_f64;
    let q0v = -0.25_f64;
    let q1v = 0.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let weight = 0.85_f64;
    let event = 1.0_f64;

    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![event]),
        weights: Arc::new(array![weight]),
        z: Arc::new(array![z_row].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![q0v]),
        offset_exit: Arc::new(array![q1v]),
        derivative_offset_exit: Arc::new(array![qd1v]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let h_range = primary.h.clone().expect("score-warp primary range");
    let w_range = primary.w.clone().expect("link-dev primary range");

    // Small, distinct non-zero deviation coefficients so every basis column
    // carries signal into the derivative chain.
    let beta_h0: Vec<f64> = (0..h_dim)
        .map(|k| 0.04 * ((k as f64 + 1.3).sin()))
        .collect();
    let beta_w0: Vec<f64> = (0..w_dim)
        .map(|k| 0.035 * ((k as f64 + 0.7).cos()))
        .collect();

    // ── Independent basis evaluations (raw design rows · β) ──────────────────
    let warp_eval = |beta_h: &[f64], z: f64| -> f64 {
        let row = score_runtime
            .design(&array![z])
            .expect("score-warp basis row");
        row.row(0).iter().zip(beta_h).map(|(b, c)| b * c).sum()
    };
    let linkdev_eval = |beta_w: &[f64], u: f64| -> f64 {
        let row = link_runtime.design(&array![u]).expect("link-dev basis row");
        row.row(0).iter().zip(beta_w).map(|(b, c)| b * c).sum()
    };
    // Survival deviation index inside Φ: a + g·z + g·warp(z) + linkdev(a + g·z).
    // The rigid slope term `g·z` is explicit: the score-warp value basis is a
    // pure DEVIATION (identity excluded — deviation_runtime.rs documents "zero
    // coefficients mean the identity map"), and production adds the rigid `b·z`
    // separately in the denested cell coefficient. Omitting it here left an
    // odd-in-z residual that the φ(z)-weighted calibration mostly absorbed into
    // the intercept, surfacing as a ~8.4e-4 self-validation gap.
    let index = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64], z: f64| -> f64 {
        a + g * z + g * warp_eval(beta_h, z) + linkdev_eval(beta_w, a + g * z)
    };
    // Witness-exact standard-normal primitives (`libm::erfc`, no piecewise
    // rational approximation). The intercept density `d1 = |∂F/∂a|` is taken by
    // a central FD with ε = 1e-6, which divides any error in `F` by 2ε = 2e-6;
    // production's `normal_cdf` carries an ~5e-12 oscillating approximation
    // error, so `F(a±ε)` inherit independent ~5e-12 perturbations whose FD is
    // ~4e-6 — exactly the residual `d1` gap (and hence the `ln d1` term left a
    // ~1.3e-5 self-validation gap, #979). Routing the calibration and the
    // survival log-CDF through ulp-accurate `erfc` removes that amplified noise.
    fn wnorm_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
    }
    fn wnorm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
    // ── Like-for-like model representation: production's own denested cells ──
    // Production does NOT Simpson-integrate the symbolic composed `index` over a
    // uniform latent grid. It partitions z into cells at every score-warp knot and
    // every link-knot crossing `z=(τ−a)/g`, and on each cell represents the index
    // as the EXACT piecewise cubic `cell.eta(z)=c0+c1 z+c2 z²+c3 z³`. Because the
    // deviation bases are piecewise-cubic, `cell.eta(z)` equals the symbolic
    // `index(a,g,z)` pointwise — but a single uniform Simpson grid that straddles
    // those interior breakpoints integrates a piecewise-cubic integrand with nodes
    // off the breaks, leaving a ~4e-6 aliasing error in `d1` (the #979 value gap).
    // The witness here re-derives production's SAME moment integral by an
    // INDEPENDENT quadrature: composite Simpson applied PER production cell (so every
    // breakpoint is an exact node, and each subinterval integrand is a smooth product
    // of a single cubic with φ). Infinite tail cells are clamped to ±8 (φ decay).
    //
    // `index(a,g,z)` is retained above and used for the observed-node eta/χ, where a
    // single cell covers z_row and the symbolic form coincides with `cell.eta`.
    let denested_cells = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| {
        let beta_h_arr = Array1::from(beta_h.to_vec());
        let beta_w_arr = Array1::from(beta_w.to_vec());
        family
            .denested_partition_cells(a, g, Some(&beta_h_arr), Some(&beta_w_arr))
            .expect("production denested partition cells")
    };
    // Composite Simpson of `f` over a single (possibly infinite) cell, clamped to
    // [-8, 8]; `sub` even subintervals so every cell is integrated to round-off for
    // the smooth single-cubic·φ integrand.
    let cell_simpson = |left: f64, right: f64, sub: usize, f: &dyn Fn(f64) -> f64| -> f64 {
        let lo = left.max(-8.0_f64);
        let hi = right.min(8.0_f64);
        if !(hi > lo) {
            return 0.0;
        }
        let m = sub; // even
        let h = (hi - lo) / m as f64;
        let mut acc = 0.0_f64;
        for k in 0..=m {
            let z = lo + h * k as f64;
            let coef = if k == 0 || k == m {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            acc += coef * f(z);
        }
        acc * h / 3.0
    };
    // Calibration F(a) = ∑cells ∫_cell Φ(−eta_cell(z)) φ(z) dz − Φ(−q), Simpson per
    // production cell on its exact representation of the index.
    let calibration = |a: f64, q: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> f64 {
        let cells = denested_cells(a, g, beta_h, beta_w);
        let mut acc = 0.0_f64;
        for partition_cell in &cells {
            let cell = partition_cell.cell;
            acc += cell_simpson(cell.left, cell.right, 1024, &|z| {
                wnorm_cdf(-cell.eta(z)) * wnorm_pdf(z)
            });
        }
        acc - wnorm_cdf(-q)
    };
    // Intercept root + density normalization D = |∂F/∂a|.
    let solve_intercept = |q: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> (f64, f64) {
        let f = |a: f64| calibration(a, q, g, beta_h, beta_w);
        // Monotone in a; secant from two seeds around the rigid closed form.
        let c = (1.0 + g * g).sqrt();
        let mut a0 = q * c - 0.5;
        let mut a1 = q * c + 0.5;
        let mut f0 = f(a0);
        for _ in 0..200 {
            let f1 = f(a1);
            if (f1 - f0).abs() <= f64::MIN_POSITIVE {
                break;
            }
            let a2 = a1 - f1 * (a1 - a0) / (f1 - f0);
            a0 = a1;
            f0 = f1;
            a1 = a2;
            if (a1 - a0).abs() <= 1e-13 {
                break;
            }
        }
        let a = a1;
        // Density D = |∂F/∂a| = ∑cells ∫_cell φ(eta_cell(z))·(∂eta_cell/∂a)(z)·φ(z) dz,
        // Simpson per production cell. On each cell `∂eta/∂a` is the EXACT cubic in z
        // whose coefficients production derives from the same score/link spans via
        // `denested_cell_coefficient_partials` (this equals `1 + linkdev'(a+g·z)` on
        // the cell, but taken from production's own cell algebra so the witness
        // re-derives production's density with no symbolic-vs-cell aliasing — the
        // ~4e-6 `d1` value gap of #979). Integrating per cell puts every breakpoint on
        // a Simpson node, so the single-cubic·φ integrand resolves to round-off.
        let cells = denested_cells(a, g, beta_h, beta_w);
        let mut acc = 0.0_f64;
        for partition_cell in &cells {
            let cell = partition_cell.cell;
            let (dc_da, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                g,
            );
            acc += cell_simpson(cell.left, cell.right, 1024, &|z| {
                let deta_da = dc_da[0] + dc_da[1] * z + dc_da[2] * z * z + dc_da[3] * z * z * z;
                wnorm_pdf(cell.eta(z)) * deta_da * wnorm_pdf(z)
            });
        }
        let d = acc.abs();
        (a, d)
    };
    // linkdev'(u) from the link runtime's analytic first-derivative design —
    // the EXACT slope of the link deviation, no finite difference.
    let linkdev_prime_eval = |beta_w: &[f64], u: f64| -> f64 {
        let row = link_runtime
            .first_derivative_design(&array![u])
            .expect("link-dev first-derivative basis row");
        row.row(0).iter().zip(beta_w).map(|(b, c)| b * c).sum()
    };
    // χ1 = ∂η1/∂a at the observed node. The index is
    // a + g·z + g·warp(z) + linkdev(a + g·z), so ∂/∂a = 1 + linkdev'(a + g·z)
    // exactly. An earlier eps=1e-6 central FD here inherited the same amplified
    // ~5e-11 cancellation noise #979 removed from the intercept density: each
    // index value carries ~1e-16 absolute round-off, the FD divides the
    // difference by 2eps=2e-6, and the third-order witness stencils then
    // multiply that floor by ~1/h³≈5e7. Taking the slope analytically removes
    // that amplified noise so the witness third derivatives are limited only by
    // the (Richardson-extrapolated) truncation of the scalar NLL itself.
    let observed_eta_chi = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> (f64, f64) {
        let eta = index(a, g, beta_h, beta_w, z_row);
        let chi = 1.0 + linkdev_prime_eval(beta_w, a + g * z_row);
        (eta, chi)
    };
    // Independent scalar survival flex row NLL over the primary vector.
    let witness_nll = |pv: &[f64]| -> f64 {
        let q0 = pv[primary.q0];
        let q1 = pv[primary.q1];
        let qd1 = pv[primary.qd1];
        let g = pv[primary.g];
        let beta_h: Vec<f64> = h_range.clone().map(|i| pv[i]).collect();
        let beta_w: Vec<f64> = w_range.clone().map(|i| pv[i]).collect();
        let (a0, _) = solve_intercept(q0, g, &beta_h, &beta_w);
        let (a1, d1) = solve_intercept(q1, g, &beta_h, &beta_w);
        let (eta0, _) = observed_eta_chi(a0, g, &beta_h, &beta_w);
        let (eta1, chi1) = observed_eta_chi(a1, g, &beta_h, &beta_w);
        let log_surv0 = wnorm_cdf(-eta0).ln();
        let log_surv1 = wnorm_cdf(-eta1).ln();
        let tau_ln = std::f64::consts::TAU.ln();
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + tau_ln);
        let log_phi_q1 = -0.5 * (q1 * q1 + tau_ln);
        weight
            * (log_surv0
                - (1.0 - event) * log_surv1
                - event * log_phi_eta1
                - event * chi1.ln()
                - event * log_phi_q1
                + event * d1.ln()
                - event * qd1.ln())
    };

    // Base primary point with NON-ZERO deviation coefficients.
    let mut p0 = vec![0.0_f64; p];
    p0[primary.q0] = q0v;
    p0[primary.q1] = q1v;
    p0[primary.qd1] = qd1v;
    p0[primary.g] = gv;
    for (k, i) in h_range.clone().enumerate() {
        p0[i] = beta_h0[k];
    }
    for (k, i) in w_range.clone().enumerate() {
        p0[i] = beta_w0[k];
    }

    // ── De-risk: witness scalar NLL must match the production scalar value ───
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![gv],
        },
        ParameterBlockState {
            beta: Array1::from(beta_h0.clone()),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::from(beta_w0.clone()),
            eta: array![0.0],
        },
    ];
    let prod_value = family
        .row_neglog_flex_value(0, &block_states)
        .expect("production flex row value");
    let wit_value = witness_nll(&p0);
    assert!(
        (prod_value - wit_value).abs() <= 1e-7 * prod_value.abs().max(1.0),
        "witness re-derivation disagrees with production scalar NLL: witness {wit_value:+.10e} vs production {prod_value:+.10e} \
         (this is a witness-side error to fix BEFORE trusting the FD jets, NOT a production bug)"
    );

    // ── Richardson central differences of the witness scalar NLL ────────────
    let central = |axes: &[(usize, usize)], h: f64| -> f64 {
        fn stencil(order: usize) -> &'static [(i64, f64)] {
            match order {
                1 => &[(-1, -0.5), (1, 0.5)],
                2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
                3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
                4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
                _ => &[(0, 1.0)],
            }
        }
        fn walk(
            stencils: &[(usize, &'static [(i64, f64)])],
            h: f64,
            coeff_acc: f64,
            point: &mut Vec<f64>,
            f: &dyn Fn(&[f64]) -> f64,
        ) -> f64 {
            match stencils.split_first() {
                None => coeff_acc * f(point),
                Some((&(idx, st), rest)) => {
                    let mut acc = 0.0;
                    let saved = point[idx];
                    for &(off, c) in st {
                        point[idx] = saved + (off as f64) * h;
                        acc += walk(rest, h, coeff_acc * c, point, f);
                    }
                    point[idx] = saved;
                    acc
                }
            }
        }
        // Coalesce repeated axes before building stencils. A coordinate that
        // appears as two separate order-1 entries would be differenced by
        // COMPOSING two independent ±h shifts, i.e. 0.25·[f(x+2h) − 2f(x) +
        // f(x−2h)] — a second difference at the DOUBLED step 2h, whose leading
        // O(h²) truncation is ~16× that of the compact 3-point order-2 stencil
        // [(-1,1),(0,-2),(1,1)] at step h. Richardson extrapolation cancels the
        // leading term in both cases, but the residual O(h⁴) constant inherits
        // the same blow-up, so a cross-derivative that hits one axis twice
        // (e.g. the ∂³/∂g²∂β_w block, axes [(g,1),(β_w,1),(g,1)]) drifts far
        // enough to break the tight third-order gate while every all-distinct
        // block stays well inside it. Summing the orders of repeated indices
        // keeps each coordinate on its tightest single stencil at step h, so
        // the witness is a faithful ground truth for repeated-axis blocks too.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(axes.len());
        for &(idx, ord) in axes {
            if let Some(slot) = merged.iter_mut().find(|(i, _)| *i == idx) {
                slot.1 += ord;
            } else {
                merged.push((idx, ord));
            }
        }
        let mut total_order = 0usize;
        let stencils: Vec<(usize, &'static [(i64, f64)])> = merged
            .iter()
            .map(|&(idx, ord)| {
                total_order += ord;
                (idx, stencil(ord))
            })
            .collect();
        let mut point = p0.clone();
        let raw = walk(&stencils, h, 1.0, &mut point, &witness_nll);
        raw / h.powi(total_order as i32)
    };
    let central_rich = |axes: &[(usize, usize)], h: f64| -> f64 {
        let coarse = central(axes, h);
        let fine = central(axes, h * 0.5);
        (4.0 * fine - coarse) / 3.0
    };

    // The deviation axes the witness must witness (q, g, and a β_h / β_w coord).
    let q0i = primary.q0;
    let gi = primary.g;
    let hi0 = h_range.start;
    let wi0 = w_range.start;

    // Embed a unit direction along a single primary axis.
    let unit = |idx: usize| -> Array1<f64> {
        let mut d = Array1::zeros(p);
        d[idx] = 1.0;
        d
    };

    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("diagnostic q geometry");
    let (_, _, prod_hess) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("diagnostic production flex hessian");
    let witness_h_gw = central_rich(&[(gi, 1), (wi0, 1)], 6e-3);
    eprintln!(
        "#932 diagnostic base hess[g,w0]: production {:+.6e} witness {:+.6e}",
        prod_hess[[gi, wi0]],
        witness_h_gw
    );
    // gam#1454 regression: the BASE intercept Hessian must match an independent
    // gradient-FD witness at [g, w0] BEFORE the directional third is even asked
    // — the owner's decisive localizer (a base mismatch means the directional
    // third merely inherits it). The base f_aa/f_au moving-boundary a-axis flux
    // (first_full.rs `moving_density_boundary_flux_a`) closes this; if it
    // regresses, this fires pointing at the base Hessian rather than the
    // directional chain.
    {
        let want = witness_h_gw;
        let got = prod_hess[[gi, wi0]];
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 5e-3 * scale + 1e-6,
            "gam#1454 base hess[g,w0] production {got:+.6e} != gradient-FD witness {want:+.6e} \
             (base intercept moving-boundary flux missing/incorrect; the directional third \
             cannot be correct while the base Hessian it differentiates is off)"
        );
    }
    // ── Third order: production D_dir H[u,v] vs witness ∂³ along (u,v,dir) ───
    // Contract along the slope axis g; check cross blocks touching the
    // deviation coordinates (the channels Arm A cannot reach).
    let third = family
        .row_flex_primary_third_contracted_exact(0, &block_states, &unit(gi))
        .expect("production third contracted (nonzero deviation)");
    let third_checks = [(q0i, hi0), (gi, wi0), (hi0, wi0), (q0i, wi0)];
    for &(u, v) in &third_checks {
        let want = central_rich(&[(u, 1), (v, 1), (gi, 1)], 6e-3);
        let got = third[[u, v]];
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 5e-3 * scale + 1e-6,
            "third[{u},{v}] (contract g) production {got:+.6e} != independent FD witness {want:+.6e}"
        );
    }
    // Planted sign-flip tripwire on a deviation-touching cross block.
    {
        let want = central_rich(&[(q0i, 1), (wi0, 1), (gi, 1)], 6e-3);
        if want.abs() > 1e-5 {
            let flipped = -third[[q0i, wi0]];
            assert!(
                (flipped - want).abs() > 5e-3 * want.abs().max(1.0) + 1e-6,
                "independent FD witness failed to reject a planted (q0, β_w) sign flip: flipped {flipped:+.6e} vs witness {want:+.6e}"
            );
        }
    }

    // ── Fourth order: production D_u D_v H[p,q] vs witness ∂⁴ ────────────────
    let fourth = family
        .row_flex_primary_fourth_contracted_exact(0, &block_states, &unit(gi), &unit(wi0))
        .expect("production fourth contracted (nonzero deviation)");
    let fourth_checks = [(q0i, gi), (q0i, hi0), (gi, hi0)];
    for &(u, v) in &fourth_checks {
        let want = central_rich(&[(u, 1), (v, 1), (gi, 1), (wi0, 1)], 9e-3);
        let got = fourth[[u, v]];
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 2e-2 * scale + 1e-5,
            "fourth[{u},{v}] (contract g,β_w) production {got:+.6e} != independent FD witness {want:+.6e}"
        );
    }
}

#[test]
fn link_flex_family_supports_second_order_exact_outer_path() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(0),
        dummy_blockspec(score_runtime.basis_dim()),
        dummy_blockspec(link_runtime.basis_dim()),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

mod time_wiggle_and_psi_derivatives;
mod anchor_history_2983;
mod resolve_start_2926;

/// gam#2938: the survival likelihood reads a Gaussian-shift frailty only through the
/// observed slope `s(σ)·g`, `s(σ) = 1/√(1+σ²)`. Rescaling the slope by
/// `c = s(σ₀)/s(σ₁)` undoes a move `σ₀ ↦ σ₁` to rounding, while the same slope at the
/// new σ does not, so wherever the slope can rescale the likelihood does not identify σ.
#[test]
fn the_survival_likelihood_reads_a_frailty_only_through_the_observed_slope_2938() {
    let n = 6;
    let marginal_design =
        Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { -0.5 + 0.2 * i as f64 });
    let marginal_beta = array![0.35, -0.1];
    let slope = Array1::from_shape_fn(n, |i| 0.4 - 0.15 * i as f64);
    let family = |sigma: f64| SurvivalMarginalSlopeFamily {
        jeffreys_armed: false,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(Array1::from_shape_fn(n, |i| (i % 2) as f64)),
        weights: Arc::new(Array1::ones(n)),
        z: Arc::new(Array1::from_shape_fn(n, |i| -1.0 + 0.4 * i as f64).insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: Some(sigma),
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
        offset_entry: Arc::new(Array1::from_shape_fn(n, |i| -0.6 + 0.1 * i as f64)),
        offset_exit: Arc::new(Array1::from_shape_fn(n, |i| 0.1 + 0.1 * i as f64)),
        derivative_offset_exit: Arc::new(Array1::from_elem(n, 0.9)),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(Array2::ones((n, 1)))).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let states = |slope: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: slope.clone(),
            },
        ]
    };
    let scale = |sigma: f64| 1.0 / (1.0 + sigma * sigma).sqrt();
    let log_likelihood = |sigma: f64, slope: &Array1<f64>| {
        family(sigma)
            .evaluate(&states(slope))
            .expect("the survival row program evaluates")
            .log_likelihood
    };
    let base = log_likelihood(0.5, &slope);
    for sigma in [0.0, 1.5, 4.0] {
        let c = scale(0.5) / scale(sigma);
        let rescaled = log_likelihood(sigma, &slope.mapv(|g| c * g));
        assert!(
            (rescaled - base).abs() <= 1e-12 * base.abs(),
            "σ = {sigma}: the rescaled slope must reproduce the likelihood at σ = 0.5, \
             {rescaled:.17e} against {base:.17e}"
        );
        let unscaled = log_likelihood(sigma, &slope);
        assert!(
            (unscaled - base).abs() > 1e-6,
            "fixture invariant: σ = {sigma} must move the likelihood of an unscaled slope, \
             {unscaled:.17e} against {base:.17e}"
        );
    }
}

#[test]
fn sigma_exact_joint_psi_terms_returns_analytic_terms() {
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_beta = array![0.2];
    let sigma = 0.65;
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: Some(sigma),
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_beta.clone(),
        },
    ];
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(marginal_design.ncols()),
        dummy_blockspec(1),
    ];

    let terms = family
        .sigma_exact_joint_psi_terms(&block_states, &specs)
        .expect("sigma psi terms should evaluate analytically")
        .expect("sigma psi terms should be present");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(
        terms.score_psi.len(),
        block_slices(&family, &block_states).total
    );
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_operator.is_some());
}

#[test]
fn censored_rows_still_reject_invalid_time_derivative() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 1)),
        )),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 1)),
        )),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((1, 1)),
        )),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];

    let err = family
        .compute_row_primary_gradient_hessian_uncached(0, &block_states)
        .expect_err("censored rows must still enforce the time-derivative domain");
    assert!(
        err.contains("monotonicity violated at row 0"),
        "unexpected error: {err}"
    );
}

fn standard_test_time_wiggle() -> (Array1<f64>, usize, usize) {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let degree = 3usize;
    let ncols = time_wiggle_basis_ncols(&knots, degree).expect("timewiggle basis width");
    (knots, degree, ncols)
}

#[test]
fn exact_newton_evaluation_propagates_invalid_rows() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];

    let err = family
        .evaluate(&block_states)
        .expect_err("invalid rows must abort exact-newton evaluation");
    assert!(
        err.contains("monotonicity violated"),
        "unexpected error: {err}"
    );
}

#[test]
fn time_constraints_use_exact_derivative_guard_rows() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![0.0, 1.0]),
        weights: Arc::new(array![1.0, 1.0]),
        z: Arc::new(array![0.0, 0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 2)),
        )),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 2)),
        )),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 2.0], [3.0, 4.0]],
        )),
        offset_entry: Arc::new(Array1::zeros(2)),
        offset_exit: Arc::new(Array1::zeros(2)),
        derivative_offset_exit: Arc::new(array![0.25, 0.5]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 0)),
        )))
        .into(),
        time_linear_constraints: time_derivative_guard_constraints(
            &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0, 1.0],
                [1.0, -1.0]
            ])),
            &array![0.0, 0.25],
            1.0,
        )
        .expect("time derivative guard constraints"),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [0.0, 0.0],
            [0.0, 0.0]
        ])),
        offset: Array1::zeros(2),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // `block_linear_constraints` expresses its rows in the coefficient basis of
    // block `block_idx`, so it needs that block's state to check the width.
    let states = [ParameterBlockState {
        beta: Array1::zeros(2),
        eta: Array1::zeros(2),
    }];
    let constraints = match family
        .block_linear_constraints(&states, 0, &spec)
        .expect("constraint lookup")
        .expect("time constraints")
    {
        gam_problem::ConstraintSet::Dense(dense) => dense,
        other => panic!("time-block constraints must be dense rows, got {other:?}"),
    };
    let row_scale = 2.0_f64.sqrt();
    let expected_a = array![
        [1.0 / row_scale, 1.0 / row_scale],
        [1.0 / row_scale, -1.0 / row_scale]
    ];
    assert_eq!(constraints.a.dim(), expected_a.dim());
    for (got, want) in constraints.a.iter().zip(expected_a.iter()) {
        assert_relative_eq!(*got, *want, epsilon = 1e-12);
    }
    let expected_b = array![1.0 / row_scale, 0.75 / row_scale];
    assert_eq!(constraints.b.dim(), expected_b.dim());
    for (got, want) in constraints.b.iter().zip(expected_b.iter()) {
        assert_relative_eq!(*got, *want, epsilon = 1e-12);
    }
}

#[test]
fn time_block_constraints_synthesize_qd1_rows_when_stored_constraints_missing() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    // Same parallel-array precondition as above: block 0 must carry state.
    let states = [ParameterBlockState {
        beta: Array1::zeros(2),
        eta: Array1::zeros(1),
    }];
    let constraints = match family
        .block_linear_constraints(&states, 0, &spec)
        .expect("synthesized constraints")
        .expect("qd1 row")
    {
        gam_problem::ConstraintSet::Dense(dense) => dense,
        other => panic!("synthesized qd1 constraints must be dense rows, got {other:?}"),
    };
    assert_eq!(constraints.a, array![[1.0, 0.0]]);
    assert_eq!(constraints.b, array![0.0]);
}

#[test]
fn time_block_max_feasible_step_uses_synthesized_qd1_rows() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.4, 7.0],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, -10.0])
        .expect("synthesized qd1 step ceiling")
        .expect("binding synthesized qd1 row");

    // Synthesized guard row `[1, 0]` with offset and guard both `1e-6`, a unit
    // row: scaled slack `0.4`, scaled drift `-1`, exact fraction `0.4`, and the
    // clipped step stops one primal-feasibility tolerance short (gam#2695).
    assert_relative_eq!(alpha, 0.4, epsilon = 1e-12);
}

#[test]
fn coupled_qd1_guard_limits_time_step_before_post_update_projection() {
    let constraints = time_derivative_guard_constraints(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 1.0
        ]])),
        &array![0.0],
        1.0,
    )
    .expect("time derivative guard constraints")
    .expect("coupled row");
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1.0,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![0.0]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        time_linear_constraints: Some(constraints),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.6, 0.6],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
        .expect("coupled qd1 step limit")
        .expect("binding coupled row");

    // Row `[1, 1]` against guard `1.0`, so in the unit-normalized metric the
    // slack is `(1.2 - 1.0)/√2` and the drift of `[-1, 0]` is `-1/√2`: the exact
    // fraction to the boundary is `0.2`, and the clipped step lands ON the face
    // so the row can enter the working face (gam#2695, gam#2714).
    assert_relative_eq!(alpha, 0.2, epsilon = 1e-12);
}

#[test]
fn timewiggle_tail_constraints_are_part_of_time_block_feasibility() {
    let structural = time_derivative_guard_constraints(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0, 0.0
        ]])),
        &array![1e-6],
        1e-6,
    )
    .expect("time derivative guard constraints");
    let constraints = append_timewiggle_tail_nonnegative_constraints(structural, 3, 2)
        .expect("combined constraints")
        .expect("time constraints");

    assert_eq!(constraints.a, Array2::<f64>::eye(3));
    assert_eq!(constraints.b, Array1::<f64>::zeros(3));
}

#[test]
fn timewiggle_tail_step_is_clipped_before_it_can_flip_derivative() {
    let constraints = append_timewiggle_tail_nonnegative_constraints(None, 2, 1)
        .expect("tail constraints")
        .expect("time constraints");
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[0.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        time_linear_constraints: Some(constraints),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 1,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.0, 0.5],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![0.0, -1.0])
        .expect("timewiggle tail step ceiling")
        .expect("negative tail step should be bounded");
    // The binding row is the timewiggle tail's own `β₁ ≥ 0`, a unit row, so the
    // scaled slack is `0.5` and the scaled drift of `[0, -1]` is `-1`: the exact
    // fraction is `0.5`, and the clipped step lands ON the face (gam#2695).
    assert_relative_eq!(alpha, 0.5, epsilon = 1e-12);
}

#[test]
fn time_block_post_update_rejects_infeasible_beta_instead_of_projecting() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        time_linear_constraints: append_timewiggle_tail_nonnegative_constraints(
            time_derivative_guard_constraints(
                &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
                    1.0, 0.0
                ]])),
                &array![1e-6],
                1e-6,
            )
            .expect("time derivative guard constraints"),
            2,
            1,
        )
        .expect("combined time constraints"),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 1,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0
        ]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0, 0.0],
                eta: array![0.0],
            }],
            0,
            &spec,
            array![-0.3, -0.2],
        )
        .expect_err("post-update must not project an infeasible time beta");
    assert!(
        err.contains("violates monotonicity") && err.contains("proposed"),
        "unexpected error message: {err}"
    );
}

/// Regression guard for the phantom-multiplier failure mode: if a
/// hand-constructed family omits the derivative-guard rows from
/// `time_linear_constraints`, post-update must not repair the proposed
/// time beta by a hidden projection. The KKT system needs those rows; a
/// missing-row violation is an error, not a convergence mechanism.
#[test]
fn time_block_post_update_rejects_qd1_when_no_linear_constraints() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        // qd1 = 1.0 · β[0] + 0.0 · β[1] + offset
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        // offset = derivative_guard exactly (the production setup).
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let current = array![0.4, 7.0];
    // qd1 at current = 1.0·0.4 + 0.0·7.0 + 1e-6 ≈ 0.4 (feasible)
    // qd1 at proposed = 1.0·-0.6 + 0.0·-3.0 + 1e-6 ≈ -0.6 (infeasible)
    let proposed = array![-0.6, -3.0];
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: current.clone(),
                eta: array![0.0],
            }],
            0,
            &spec,
            proposed.clone(),
        )
        .expect_err("missing qd1 constraints must not be repaired by projection");
    assert!(
        err.contains("violates monotonicity")
            && err.contains("proposed")
            && err.contains("time_linear_constraints"),
        "unexpected error message: {err}"
    );
}

/// When `current` already violates the qd1 monotonicity, the projection
/// surfaces a structured error (with the row index and qd1 value)
/// instead of silently returning a still-infeasible β. This is the
/// invariant the score_warp / link_dev projection enforces; the time
/// block now matches.
#[test]
fn time_block_post_update_errors_when_current_violates_qd1() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[0.0]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // current qd1 = -1.0 + 1e-6 < guard → infeasible.
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![-1.0],
                eta: array![0.0],
            }],
            0,
            &spec,
            array![0.5],
        )
        .expect_err("infeasible current must surface an error");
    assert!(
        err.contains("violates monotonicity") && err.contains("row 0"),
        "unexpected error message: {err}"
    );
}

#[test]
fn time_block_feasible_step_stays_inside_derivative_guard() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![0.2]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        slope_layout: (DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )))
        .into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: time_derivative_guard_constraints(
            &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0
            ]])),
            &array![0.2],
            1e-4,
        )
        .expect("time derivative guard constraints"),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
        .expect("time step ceiling")
        .expect("time step should be bounded");
    // Starting at beta=[0,0] with derivative q' = design·beta + offset = 0.2,
    // far above the 1e-4 guard. Stepping along [-1, 0] drives q' toward the
    // guard; the largest feasible α satisfies -α + 0.2 = 1e-4, i.e. α ≈ 0.1999,
    // and the shared `feasible_step_fraction` lands the clipped step ON that
    // face (no retreat), so the row can enter the active-set solver's working
    // face on the next cycle (gam#2695, gam#2714).
    assert!(
        alpha > 0.0 && alpha < 1.0,
        "expected a clipped step, got {alpha}"
    );
    assert!(
        (alpha - 0.1999).abs() <= 1e-12,
        "the clipped fraction is the exact ratio to the face, got {alpha}"
    );
    let feasible = &states[0].beta + &(array![-1.0, 0.0] * alpha);
    let slack = family
        .time_linear_constraints
        .as_ref()
        .expect("constraints")
        .a
        .row(0)
        .dot(&feasible)
        - family
            .time_linear_constraints
            .as_ref()
            .expect("constraints")
            .b[0];
    assert!(
        slack.abs() <= 1e-12 && slack >= -gam_problem::PRIMAL_FEASIBILITY_TOL,
        "the clipped step lands on the face within round-off and inside the contract band; \
         slack={slack:e}"
    );
}

#[test]
fn mixed_blockwise_exact_newton_preserves_sparse_block_hessians() {
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.1, -0.2].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.6]])),
        design_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[0.9], [0.5]])),
        design_derivative_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [1.0]])),
        offset_entry: Arc::new(array![0.0, 0.0]),
        offset_exit: Arc::new(array![0.0, 0.0]),
        derivative_offset_exit: Arc::new(array![0.05, 0.05]),
        marginal_design: sparse_design(&array![[1.0, 0.0], [0.0, 1.0]]),
        slope_layout: (DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.5]]))).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.4],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.2, -0.1],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.3],
            eta: array![0.3, 0.3],
        },
    ];

    let eval = family
        .evaluate_blockwise_exact_newton(&block_states)
        .expect("mixed exact-newton evaluation");

    assert!(matches!(
        &eval.blockworking_sets[0],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_),
            ..
        }
    ));
    assert!(matches!(
        &eval.blockworking_sets[1],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_) | SymmetricMatrix::Sparse(_),
            ..
        }
    ));
    assert!(matches!(
        &eval.blockworking_sets[2],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_),
            ..
        }
    ));
}

/// Closed-form test family with `gaussian_frailty_sd` set so the sigma-aware
/// joint psi paths fire. Same pseudo-random row layout as
/// `make_closed_form_test_family` so half-row subsamples remain
/// representative.
fn make_sigma_aware_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let mut family = make_closed_form_test_family(n);
    family.gaussian_frailty_sd = Some(0.6);
    family
}

fn rel_diff_array1_survival(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut max = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs() / b[i].abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

fn rel_diff_array2_survival(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut max = 0.0f64;
    for ((i, j), &av) in a.indexed_iter() {
        let bv = b[[i, j]];
        let d = (av - bv).abs() / bv.abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

#[test]
fn survival_sigma_psi_terms_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

    let baseline = family
        .sigma_exact_joint_psi_terms(&states, &specs)
        .expect("baseline psi terms")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_terms_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_second_order_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let baseline = family
        .sigma_exact_joint_psisecond_order_terms(&states)
        .expect("baseline")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_second_order_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_sigma_psihessian_directional_derivative_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let slices = block_slices(&family, &states);
    let d_beta_flat = Array1::<f64>::zeros(slices.total);

    let baseline = family
        .sigma_exact_joint_psihessian_directional_derivative(&states, &d_beta_flat)
        .expect("baseline")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full")
        .expect("some");

    let rel = rel_diff_array2_survival(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_sigma_psihessian_directional_derivative_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let slices = block_slices(&family, &states);
    let d_beta_flat = Array1::<f64>::zeros(slices.total);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2_survival(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

/// Multi-row test family with non-empty marginal/slope designs but no
/// score_warp / link_dev / time_wiggle. Drives the rigid block path of
/// `psi_terms_inner` so we can subsample-check Horvitz-Thompson scaling.
fn make_block_psi_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    // Single-column marginal/slope designs with row-varying entries.
    let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
    });
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal_design),
        slope_layout: (DesignMatrix::from(slope_design)).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

fn block_psi_test_block_states(
    family: &SurvivalMarginalSlopeFamily,
    m_beta: f64,
    g_beta: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    let m_eta = m_design.dot(&array![m_beta]);
    let g_eta = g_design.dot(&array![g_beta]);
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: array![m_beta],
            eta: m_eta,
        },
        ParameterBlockState {
            beta: array![g_beta],
            eta: g_eta,
        },
    ]
}

fn make_rigid_baseline_psi_test_family(
    n: usize,
) -> (
    SurvivalMarginalSlopeFamily,
    crate::custom_family::CustomFamilyHyperLayout,
) {
    let mut family = make_block_psi_test_family(n);
    let age_entry = Array1::from_shape_fn(n, |row| 0.25 + 0.1 * row as f64);
    let age_exit = Array1::from_shape_fn(n, |row| age_entry[row] + 0.75 + 0.03 * row as f64);
    let baseline_config = crate::survival::construction::SurvivalBaselineConfig {
        target: crate::survival::construction::SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    };
    let geometry = Arc::new(
        crate::survival::construction::build_survival_marginal_slope_baseline_geometry(
            &age_entry,
            &age_exit,
            &baseline_config,
        )
        .expect("build rigid baseline geometry")
        .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );
    family.offset_entry = Arc::new(geometry.offset_entry.clone());
    family.offset_exit = Arc::new(geometry.offset_exit.clone());
    family.derivative_offset_exit = Arc::new(geometry.derivative_offset_exit.clone());
    family.family_hyper =
        SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
            .expect("install rigid baseline family coordinates");

    let family_axes = (0..geometry.theta.len()).collect::<Vec<_>>();
    let layout = crate::custom_family::CustomFamilyHyperLayout::new(
        vec![Vec::new(), Vec::new(), Vec::new()],
        family_axes,
        geometry.theta.clone(),
    )
    .expect("build typed baseline hyper layout");
    (family, layout)
}

fn make_flex_baseline_psi_test_fixture() -> (
    SurvivalMarginalSlopeFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    crate::custom_family::CustomFamilyHyperLayout,
) {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let (knots, degree, wiggle_width) = standard_test_time_wiggle();
    let time_width = 1 + wiggle_width;
    let age_entry = array![0.25];
    let age_exit = array![1.15];
    let baseline_config = crate::survival::construction::SurvivalBaselineConfig {
        target: crate::survival::construction::SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    };
    let geometry = Arc::new(
        crate::survival::construction::build_survival_marginal_slope_baseline_geometry(
            &age_entry,
            &age_exit,
            &baseline_config,
        )
        .expect("build FLEX baseline geometry")
        .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );
    let mut entry_design = Array2::zeros((1, time_width));
    let mut exit_design = Array2::zeros((1, time_width));
    let mut derivative_design = Array2::zeros((1, time_width));
    entry_design[[0, 0]] = 0.25;
    exit_design[[0, 0]] = 0.45;
    derivative_design[[0, 0]] = 0.15;
    let marginal_design = array![[0.30]];
    let slope_design = array![[0.40]];
    let family_hyper =
        SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
            .expect("install FLEX baseline family coordinates");
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![0.9]),
        z: Arc::new(array![0.2].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(entry_design),
        design_exit: DesignMatrix::from(exit_design.clone()),
        design_derivative_exit: DesignMatrix::from(derivative_design),
        offset_entry: Arc::new(geometry.offset_entry.clone()),
        offset_exit: Arc::new(geometry.offset_exit.clone()),
        derivative_offset_exit: Arc::new(geometry.derivative_offset_exit.clone()),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: DesignMatrix::from(slope_design.clone()).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(knots),
        time_wiggle_degree: Some(degree),
        time_wiggle_ncols: wiggle_width,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };

    let mut beta_time = Array1::zeros(time_width);
    beta_time[0] = 0.10;
    for local in 0..wiggle_width {
        beta_time[1 + local] = 0.006 + 0.002 * local as f64;
    }
    let beta_marginal = array![0.12];
    let beta_slope = array![0.20];
    let beta_score =
        Array1::from_iter((0..score_runtime.basis_dim()).map(|axis| 0.004 * (1.0 + axis as f64)));
    let beta_link =
        Array1::from_iter((0..link_runtime.basis_dim()).map(|axis| -0.003 + 0.001 * axis as f64));
    let states = vec![
        ParameterBlockState {
            eta: exit_design.dot(&beta_time) + geometry.offset_exit.clone(),
            beta: beta_time,
        },
        ParameterBlockState {
            eta: marginal_design.dot(&beta_marginal),
            beta: beta_marginal,
        },
        ParameterBlockState {
            eta: slope_design.dot(&beta_slope),
            beta: beta_slope,
        },
        ParameterBlockState {
            beta: beta_score,
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: beta_link,
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.17]],
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut values = vec![0.0];
    values.extend(geometry.theta.iter().copied());
    let layout = crate::custom_family::CustomFamilyHyperLayout::new(
        derivative_blocks,
        (0..geometry.theta.len()).collect(),
        Array1::from_vec(values),
    )
    .expect("build FLEX baseline/design hyper layout");
    let specs = states
        .iter()
        .map(|state| dummy_blockspec(state.beta.len()))
        .collect();
    (family, states, specs, layout)
}

fn assert_psi_first_terms_match(
    direct: &ExactNewtonJointPsiTerms,
    workspace: &ExactNewtonJointPsiTerms,
) {
    assert_close(
        direct.objective_psi,
        workspace.objective_psi,
        1e-13,
        "baseline first objective direct/workspace",
    );
    assert!(
        rel_diff_array1_survival(&direct.score_psi, &workspace.score_psi) < 1e-13,
        "baseline first score direct/workspace mismatch",
    );
    let direct_hessian = direct
        .hessian_psi_operator
        .as_ref()
        .expect("direct baseline Hessian operator")
        .to_dense();
    let workspace_hessian = workspace
        .hessian_psi_operator
        .as_ref()
        .expect("workspace baseline Hessian operator")
        .to_dense();
    assert!(
        rel_diff_array2_survival(&direct_hessian, &workspace_hessian) < 1e-13,
        "baseline first Hessian direct/workspace mismatch",
    );
}

fn assert_psi_second_terms_match(
    direct: &ExactNewtonJointPsiSecondOrderTerms,
    workspace: &ExactNewtonJointPsiSecondOrderTerms,
) {
    assert_close(
        direct.objective_psi_psi,
        workspace.objective_psi_psi,
        1e-13,
        "baseline pair objective direct/workspace",
    );
    assert!(
        rel_diff_array1_survival(&direct.score_psi_psi, &workspace.score_psi_psi) < 1e-13,
        "baseline pair score direct/workspace mismatch",
    );
    let direct_hessian = direct
        .hessian_psi_psi_operator
        .as_ref()
        .expect("direct baseline-pair Hessian operator")
        .to_dense();
    let workspace_hessian = workspace
        .hessian_psi_psi_operator
        .as_ref()
        .expect("workspace baseline-pair Hessian operator")
        .to_dense();
    assert!(
        rel_diff_array2_survival(&direct_hessian, &workspace_hessian) < 1e-13,
        "baseline pair Hessian direct/workspace mismatch",
    );
}

#[test]
fn rigid_baseline_dispatch_matches_direct_and_owned_workspace_without_fd() {
    let (family, hyper_layout) = make_rigid_baseline_psi_test_family(12);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(1), dummy_blockspec(1)];
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &BlockwiseFitOptions::default(),
        )
        .expect("construct rigid baseline workspace")
        .expect("rigid baseline workspace is available");

    for axis in 0..hyper_layout.len() {
        let direct = family
            .exact_newton_joint_psi_terms(&states, &specs, &hyper_layout, axis)
            .expect("direct rigid baseline first terms")
            .expect("direct rigid baseline first terms available");
        let owned = workspace
            .first_order_terms(axis)
            .expect("workspace rigid baseline first terms")
            .expect("workspace rigid baseline first terms available");
        assert_psi_first_terms_match(&direct, &owned);
    }

    let axis = 0;
    let other_axis = usize::from(hyper_layout.len() > 1);
    let direct_pair = family
        .exact_newton_joint_psisecond_order_terms(&states, &specs, &hyper_layout, axis, other_axis)
        .expect("direct rigid baseline pair terms")
        .expect("direct rigid baseline pair terms available");
    let owned_pair = workspace
        .second_order_terms(axis, other_axis)
        .expect("workspace rigid baseline pair terms")
        .expect("workspace rigid baseline pair terms available");
    assert_psi_second_terms_match(&direct_pair, &owned_pair);

    let direction = array![0.17, -0.09];
    let direct_drift = family
        .exact_newton_joint_psihessian_directional_derivative(
            &states,
            &specs,
            &hyper_layout,
            axis,
            &direction,
        )
        .expect("direct rigid baseline Hessian drift")
        .expect("direct rigid baseline Hessian drift available");
    let owned_drift = workspace
        .hessian_directional_derivative(axis, &direction)
        .expect("workspace rigid baseline Hessian drift")
        .expect("workspace rigid baseline Hessian drift available");
    let gam_problem::DriftDerivResult::Dense(owned_drift) = owned_drift else {
        panic!("rigid baseline workspace drift must preserve the dense exact carrier");
    };
    assert!(
        rel_diff_array2_survival(&direct_drift, &owned_drift) < 1e-13,
        "baseline Hessian drift direct/workspace mismatch",
    );
}

/// gam#2930: the one-pass contraction `⟨W, ∂_θ H²[e_a, e_b]⟩` along a baseline-chart coordinate
/// is the contraction with `W` of the workspace's column-by-column `{∂_θ H²[e_b, e_a]}`, to
/// roundoff, on every chart axis.
#[test]
fn rigid_baseline_contracted_trace_hessian_psi_matches_column_contraction_2930() {
    let (family, hyper_layout) = make_rigid_baseline_psi_test_family(12);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(1), dummy_blockspec(1)];
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &BlockwiseFitOptions::default(),
        )
        .expect("construct rigid baseline workspace")
        .expect("rigid baseline workspace is available");
    let total = 2;
    let weight = array![[0.7, -0.2], [-0.2, 0.4]];
    assert!(!hyper_layout.is_empty(), "the chart fixture must carry baseline coordinates");
    for axis in 0..hyper_layout.len() {
        assert!(
            workspace
                .contracted_trace_hessian_psi_axes()
                .expect("availability query")
                .contains(&axis),
            "chart axis {axis} must serve the one-pass contraction"
        );
        let contracted = workspace
            .contracted_trace_hessian_psi(axis, &weight)
            .expect("one-pass contraction")
            .expect("chart axis contraction is served");
        let mut reference = Array2::<f64>::zeros((total, total));
        for b in 0..total {
            let mut unit = Array1::<f64>::zeros(total);
            unit[b] = 1.0;
            let columns = workspace
                .hessian_second_directional_derivative_all_beta_axes(axis, &unit)
                .expect("column-by-column third information derivative")
                .expect("chart axis third information derivative is served");
            for a in 0..total {
                reference[[a, b]] = (&weight * &columns[a]).sum();
            }
        }
        assert!(
            reference.iter().any(|value| value.abs() > 1e-8),
            "chart axis {axis}: the reference contraction is identically zero, so it grades nothing"
        );
        assert!(
            rel_diff_array2_survival(&contracted, &reference) < 1e-12,
            "chart axis {axis}: one-pass {contracted:?} against column contraction {reference:?}"
        );
    }
}

#[test]
fn flex_timewiggle_baseline_public_workspace_owns_family_and_design_pairs_without_fd() {
    let (family, states, specs, hyper_layout) = make_flex_baseline_psi_test_fixture();
    assert!(family.flex_active());
    assert!(family.flex_timewiggle_active());
    assert!(family.score_warp.is_some());
    assert!(family.link_dev.is_some());
    let slices = block_slices(&family, &states);
    let dimension = slices.total;
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &BlockwiseFitOptions::default(),
        )
        .expect("construct FLEX baseline workspace")
        .expect("FLEX baseline workspace is available");
    let design_axis = 0;
    let baseline_axis = hyper_layout.design_axis_count();
    let other_baseline_axis = baseline_axis + 1;

    let first = workspace
        .first_order_terms(baseline_axis)
        .expect("FLEX baseline first callback")
        .expect("FLEX baseline first terms are present");
    assert_eq!(first.score_psi.len(), dimension);
    // gam#3061: the ζ composition serves this frame's chart terms with a dense θ Hessian.
    assert!(family.timewiggle_zeta_available());
    assert!(
        first.hessian_psi_operator.is_none(),
        "the ζ composition publishes a dense baseline θ Hessian"
    );
    let first_hessian = first.hessian_psi.clone();
    assert!(first_hessian.iter().any(|value| *value != 0.0));
    assert_eq!(first_hessian.dim(), (dimension, dimension));
    assert!(first.score_psi.iter().all(|value| value.is_finite()));
    assert!(first_hessian.iter().all(|value| value.is_finite()));

    let baseline_pair = workspace
        .second_order_terms(baseline_axis, other_baseline_axis)
        .expect("FLEX baseline-pair callback")
        .expect("FLEX baseline-pair terms are present");
    assert_eq!(baseline_pair.score_psi_psi.len(), dimension);
    // gam#3304: the ζ composition serves chart pairs with a dense θθ Hessian too.
    assert!(
        baseline_pair.hessian_psi_psi_operator.is_none(),
        "the ζ composition publishes a dense baseline-pair Hessian"
    );
    let baseline_pair_hessian = baseline_pair.hessian_psi_psi.clone();
    assert_eq!(baseline_pair_hessian.dim(), (dimension, dimension));
    assert!(
        baseline_pair
            .score_psi_psi
            .iter()
            .all(|value| value.is_finite())
    );

    let beta_direction =
        Array1::from_iter((0..dimension).map(|axis| -0.025 + 0.006 * (axis % 7) as f64));
    let drift = workspace
        .hessian_directional_derivative(baseline_axis, &beta_direction)
        .expect("FLEX baseline Hessian-drift callback")
        .expect("FLEX baseline Hessian drift is present");
    let gam_problem::DriftDerivResult::Dense(drift) = drift else {
        panic!("FLEX baseline workspace drift must preserve the dense exact carrier");
    };
    assert_eq!(drift.dim(), (dimension, dimension));
    assert!(drift.iter().all(|value| value.is_finite()));
    assert!(drift.iter().any(|value| *value != 0.0));

    // This is the large-scale family×design seam: the global design axis is a
    // real DesignPenalty manifest entry, not a coefficient direction.  The
    // workspace must route it through X_psi beta + X_psi Jet3 seeding and
    // return the complete fixed-beta pair.
    let mixed = workspace
        .second_order_terms(baseline_axis, design_axis)
        .expect("FLEX baseline-by-design callback")
        .expect("FLEX baseline-by-design terms are present");
    assert_eq!(mixed.score_psi_psi.len(), dimension);
    assert!(mixed.objective_psi_psi.is_finite());
    assert!(mixed.score_psi_psi.iter().all(|value| value.is_finite()));
    assert!(
        mixed.score_psi_psi.iter().any(|value| value.abs() > 1e-12),
        "active FLEX/timewiggle baseline-by-design score must not collapse to zero"
    );
    assert!(
        mixed.hessian_psi_psi_operator.is_none(),
        "the ζ composition publishes a dense baseline-by-design Hessian"
    );
    let mixed_hessian = mixed.hessian_psi_psi.clone();
    assert_eq!(mixed_hessian.dim(), (dimension, dimension));
    assert!(mixed_hessian.iter().all(|value| value.is_finite()));
    assert!(mixed_hessian.iter().any(|value| *value != 0.0));
}

/// Derivative blocks with a single ψ on the marginal block (block_idx=1).
/// `x_psi` has shape (n, 1) so the test family gets a per-row psi map.
fn block_psi_test_marginal_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ]
}

/// Derivative blocks with one ψ on marginal (block 1) and one on slope
/// (block 2), so second-order terms can mix.
fn block_psi_test_dual_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi_m = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    let x_psi_g = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 43 + 17) % n) as f64) / (n as f64)
    });
    vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_m,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_g,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
    ]
}

#[test]
fn survival_psi_terms_inner_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

    let baseline = family
        .psi_terms_inner(&states, &derivative_blocks, 0, None)
        .expect("baseline psi terms")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_psi_terms_inner_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_psi_terms_inner_batched_matches_per_axis() {
    // The batched first-order ψ row pass shares the per-row primary
    // gradient/Hessian across axes; this asserts it produces the same
    // ExactNewtonJointPsiTerms (objective, score, Hessian-operator action)
    // as K serial calls to `psi_terms_inner_with_options`.
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);
    let opts = BlockwiseFitOptions::default();

    let per_axis_0 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
        .expect("per-axis 0")
        .expect("some");
    let per_axis_1 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
        .expect("per-axis 1")
        .expect("some");

    let batched = family
        .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
        .expect("batched")
        .expect("batched simple-spatial path returned None unexpectedly");
    assert_eq!(batched.len(), 2, "batched should yield one term per axis");

    let per_axis = [&per_axis_0, &per_axis_1];
    for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
        let obj_rel =
            ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
        assert!(
            obj_rel < 1e-12,
            "axis {i} objective_psi rel {obj_rel} (per-axis={}, batched={})",
            lhs.objective_psi,
            rhs.objective_psi,
        );
        let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
        assert!(score_rel < 1e-12, "axis {i} score_psi rel {score_rel}");

        let op_a = lhs
            .hessian_psi_operator
            .as_ref()
            .expect("per-axis Hessian operator");
        let op_b = rhs
            .hessian_psi_operator
            .as_ref()
            .expect("batched Hessian operator");
        assert_eq!(op_a.dim(), op_b.dim(), "axis {i} operator dim mismatch");
        let dim = op_a.dim();
        let probe = Array1::from_shape_fn(dim, |j| {
            ((j as i64 * 37 + 11).rem_euclid(7)) as f64 * 0.1 - 0.3
        });
        let a = op_a.mul_vec(&probe);
        let b = op_b.mul_vec(&probe);
        let op_rel = rel_diff_array1_survival(&a, &b);
        assert!(op_rel < 1e-12, "axis {i} Hessian-action rel {op_rel}");
    }
}

#[test]
fn survival_psi_terms_inner_batched_subsample_matches_per_axis() {
    // Same equivalence under a half-row Horvitz-Thompson mask, exercising
    // the per-row weight branch of the batched fast path.
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let mut opts = BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::from_uniform_inclusion_mask(
        even_mask, n, 0xC0FFEE,
    )));

    let per_axis_0 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
        .expect("per-axis 0")
        .expect("some");
    let per_axis_1 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
        .expect("per-axis 1")
        .expect("some");

    let batched = family
        .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
        .expect("batched")
        .expect("batched simple-spatial path returned None under subsample");
    assert_eq!(batched.len(), 2);

    let per_axis = [&per_axis_0, &per_axis_1];
    for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
        let obj_rel =
            ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
        assert!(
            obj_rel < 1e-12,
            "axis {i} subsample objective_psi rel {obj_rel}"
        );
        let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
        assert!(
            score_rel < 1e-12,
            "axis {i} subsample score_psi rel {score_rel}"
        );

        let op_a = lhs.hessian_psi_operator.as_ref().unwrap();
        let op_b = rhs.hessian_psi_operator.as_ref().unwrap();
        let dim = op_a.dim();
        let probe = Array1::from_shape_fn(dim, |j| {
            ((j as i64 * 41 + 5).rem_euclid(11)) as f64 * 0.07 - 0.4
        });
        let a = op_a.mul_vec(&probe);
        let b = op_b.mul_vec(&probe);
        let op_rel = rel_diff_array1_survival(&a, &b);
        assert!(
            op_rel < 1e-12,
            "axis {i} subsample Hessian-action rel {op_rel}"
        );
    }
}

#[test]
fn survival_psi_second_order_terms_inner_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let baseline = family
        .psi_second_order_terms_inner(&states, &derivative_blocks, 0, 1, None)
        .expect("baseline psi second-order")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_full,
        )
        .expect("with full")
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_psi_second_order_terms_inner_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let baseline = family
        .psi_hessian_directional_derivative(&states, &derivative_blocks, 0, &d_beta_flat)
        .expect("baseline psi-Hessian directional derivative")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask")
        .expect("some");

    let rel = rel_diff_array2_survival(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2_survival(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_psi_workspace_hessian_directional_derivative_is_operator_and_matches_dense() {
    let n = 40usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(1), dummy_blockspec(1)];
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let dense = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("dense drift")
        .expect("dense drift available");
    let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
        derivative_blocks,
        Vec::new(),
        Array1::zeros(1),
    )
    .expect("build typed design-hyper layout");
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &BlockwiseFitOptions::default(),
        )
        .expect("workspace")
        .expect("workspace available");
    let result = workspace
        .hessian_directional_derivative(0, &d_beta_flat)
        .expect("workspace drift")
        .expect("workspace drift available");

    let gam_problem::DriftDerivResult::Operator(op) = result else {
        panic!("survival psi drift should use operator representation");
    };
    assert_eq!(op.dim(), dense.nrows());
    let operator_dense = op.to_dense();
    let rel = rel_diff_array2_survival(&operator_dense, &dense);
    assert!(rel < 1e-12, "operator/dense drift rel {rel}");
}

#[test]
fn survival_psi_hessian_directional_derivative_operator_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let baseline = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("baseline operator")
        .expect("some");
    let baseline_dense = baseline.to_dense();

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask")
        .expect("some");
    let with_full_dense = with_full.to_dense();

    let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_operator_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");
    let scaled_dense = scaled.to_dense();

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");
    let raw_dense = raw.to_dense();

    let factor = n as f64 / m as f64;
    let exp = &raw_dense * factor;
    let rel = rel_diff_array2_survival(&scaled_dense, &exp);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}

// ── Phase 7: joint-Hessian flex-no-wiggle directional-derivative
// operator subsample tests. The flex-no-wiggle helpers are the path
// taken by the joint-Hessian workspace's `directional_derivative_operator`
// when `effective_flex_active(states)` is true and timewiggle is off.
// We exercise the helpers directly through a flex-active fixture so the
// outer subsample threading is verified end-to-end.

fn make_flex_no_wiggle_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let score_runtime = test_deviation_runtime();
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
    });
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal_design),
        slope_layout: (DesignMatrix::from(slope_design)).into(),
        score_warp: Some(score_runtime),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

fn flex_no_wiggle_test_block_states(
    family: &SurvivalMarginalSlopeFamily,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    let m_beta = 0.15_f64;
    let g_beta = 0.25_f64;
    let m_eta = m_design.dot(&array![m_beta]);
    let g_eta = g_design.dot(&array![g_beta]);
    let score_dim = family
        .score_warp
        .as_ref()
        .map(|w| w.basis_dim())
        .unwrap_or(0);
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: array![m_beta],
            eta: m_eta,
        },
        ParameterBlockState {
            beta: array![g_beta],
            eta: g_eta,
        },
        ParameterBlockState {
            beta: Array1::zeros(score_dim),
            eta: Array1::zeros(n),
        },
    ]
}

#[test]
fn survival_jointhessian_flex_no_wiggle_operator_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 40usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let baseline = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("baseline operator");
    let baseline_dense = baseline.to_dense();

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask");
    let with_full_dense = with_full.to_dense();

    let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
    assert!(
        rel < 1e-10,
        "joint Hessian flex-no-wiggle dH operator drift rel {}",
        rel
    );
}

/// #979 flex hot path: the build-once all-axes directional-derivative sweep
/// (`..._flex_no_wiggle_all_axes`) must reproduce calling the single-direction
/// routine once per coordinate axis. The all-axes path hoists the
/// direction-independent per-row geometry (intercept solves, cached partitions,
/// base timepoints) out of the per-axis loop; this asserts the optimization
/// changes only the cost, never the result. The two paths feed the same per-row
/// assemblers but run independent rayon reductions, so equality is to a tight
/// relative tolerance (the same convention the other flex-no-wiggle operator
/// tests use), not bit-for-bit.
#[test]
fn survival_jointhessian_flex_no_wiggle_all_axes_matches_per_axis() {
    let n = 40usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let slices = block_slices(&family, &states);
    let p = slices.total;
    assert!(p >= 2, "test family must exercise multiple axes, got p={p}");

    let all_axes = family
        .exact_newton_joint_hessian_directional_derivative_flex_no_wiggle_all_axes(&states)
        .expect("all-axes sweep");
    assert_eq!(all_axes.len(), p);

    for axis_idx in 0..p {
        let mut axis = Array1::<f64>::zeros(p);
        axis[axis_idx] = 1.0;
        let per_axis = family
            .exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(&states, &axis)
            .expect("per-axis directional derivative");
        let batched = &all_axes[axis_idx];
        assert_eq!(batched.dim(), per_axis.dim());
        let rel = rel_diff_array2_survival(batched, &per_axis);
        assert!(
            rel < 1e-10,
            "axis {axis_idx}: build-once all-axes path diverged from per-axis sweep, rel {rel}"
        );
    }
}

/// #932. The flex no-wiggle arm pulls its jet-derived primary tower back into coefficient
/// space by hand (`accumulate_directional_joint_hessian_row` over the dynamic q geometry,
/// with the identity-block crosses). Its other tests compare the build-once sweep with the
/// per-axis route and the subsampled operator with the dense one, so the β-directional
/// derivatives themselves were never checked against the object they differentiate. This
/// applies the time-wiggle gate to this arm: `D_β H[v]` against a resolving central
/// difference of the dense joint Hessian, and `D²_β H[u, v]` against a resolving central
/// difference of `D_β H[v]` along `u`. Small nonzero score-warp coefficients make the warp
/// columns carry curvature.
#[test]
fn flex_no_wiggle_beta_hessian_directional_derivatives_match_finite_difference_932() {
    let family = make_flex_no_wiggle_test_family(40);
    let base_states = flex_no_wiggle_test_block_states(&family);
    let score_dim = base_states[3].beta.len();
    let beta = Array1::from_shape_fn(2 + score_dim, |i| match i {
        0 => base_states[1].beta[0],
        1 => base_states[2].beta[0],
        _ if i % 2 == 0 => 0.02,
        _ => -0.02,
    });
    let marginal_design = family.marginal_design.to_dense().to_owned();
    let slope_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    let states_at = |beta: &Array1<f64>| -> Vec<ParameterBlockState> {
        let marginal_beta = beta.slice(s![0..1]).to_owned();
        let slope_beta = beta.slice(s![1..2]).to_owned();
        vec![
            base_states[0].clone(),
            ParameterBlockState {
                eta: marginal_design.dot(&marginal_beta),
                beta: marginal_beta,
            },
            ParameterBlockState {
                eta: slope_design.dot(&slope_beta),
                beta: slope_beta,
            },
            ParameterBlockState {
                beta: beta.slice(s![2..]).to_owned(),
                eta: base_states[3].eta.clone(),
            },
        ]
    };
    let states = states_at(&beta);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let label = "flex no-wiggle";
    let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
    let h = 1e-3;
    let gate = |what: &str, analytic: &Array2<f64>, at: &dyn Fn(f64) -> Array2<f64>| {
        let coarse = (at(h) - at(-h)) / (2.0 * h);
        let fine = (at(0.5 * h) - at(-0.5 * h)) / h;
        let scale = analytic
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1e-12);
        assert!(
            scale > 1e-6,
            "{label}: {what} carries no curvature on this fixture ({scale:.3e}), so the \
             comparison would pass on zeros"
        );
        for ((index, &want), (&c, &f)) in analytic
            .indexed_iter()
            .zip(coarse.iter().zip(fine.iter()))
        {
            let value = (4.0 * f - c) / 3.0;
            let uncertainty = (f - c).abs() / 3.0;
            let denominator = scale.max(want.abs()).max(value.abs());
            assert!(
                uncertainty <= 0.05 * denominator,
                "{label}: {what}{index:?}: the difference oracle did not resolve \
                 (value={value:.6e}, uncertainty={uncertainty:.3e})"
            );
            assert!(
                (want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty,
                "{label}: {what}{index:?}: analytic={want:.9e} fd={value:.9e} \
                 uncertainty={uncertainty:.3e} scale={scale:.3e}"
            );
        }
    };
    let first = family
        .exact_newton_joint_hessian_directional_derivative(&states, &v)
        .expect("D_beta H[v]")
        .expect("the flex arm publishes D_beta H");
    gate("D_beta H[v]", &first, &|t| {
        family
            .exact_newton_joint_hessian(&states_at(&(&beta + &(&v * t))))
            .expect("joint Hessian")
            .expect("survival marginal-slope publishes an explicit joint Hessian")
    });
    let second = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &v)
        .expect("D2_beta H[u, v]")
        .expect("the flex arm publishes D2_beta H");
    gate("D2_beta H[u, v]", &second, &|t| {
        family
            .exact_newton_joint_hessian_directional_derivative(&states_at(&(&beta + &(&u * t))), &v)
            .expect("displaced D_beta H[v]")
            .expect("the flex arm publishes D_beta H")
    });
}

/// gam#2893: the build-once flex third contraction reads the absorbed-influence offset
/// `o_infl[row] = Z̃_infl[row,:]·γ` exactly as the single-direction route does. The base used to
/// drop it and contract every directional timepoint at `o_infl = 0`, so with an influence absorber
/// the build-once Jeffreys sweeps computed a different contraction from the per-axis one.
#[test]
fn survival_flex_third_contraction_from_base_reads_influence_offset_2893() {
    let n = 40usize;
    let mut family = make_flex_no_wiggle_test_family(n);
    family.influence_absorber = Some(Array2::from_shape_fn((n, 2), |(i, j)| {
        0.3 * (((i * (7 + 4 * j) + 3 * j + 1) % n) as f64 / n as f64 - 0.5)
    }));
    let mut states = flex_no_wiggle_test_block_states(&family);
    states.push(ParameterBlockState {
        beta: array![0.4, -0.3],
        eta: Array1::zeros(n),
    });
    assert!(family.effective_flex_active(&states).unwrap());
    let primary = flex_primary_slices(&family);
    let direction =
        Array1::from_shape_fn(primary.total, |i| ((i * 5 + 2) % 7) as f64 / 7.0 - 0.4);
    let mut max_offset = 0.0_f64;
    for row in [0usize, 7, 19, 33] {
        let offset = family
            .influence_index_offset(row, &states)
            .expect("influence offset");
        max_offset = max_offset.max(offset.abs());
        let exact = family
            .row_flex_primary_third_contracted_exact(row, &states, &direction)
            .expect("exact third contraction");
        let base = family
            .build_row_flex_third_base_with_states(row, &states, &primary)
            .expect("third-order base");
        let from_base = family
            .row_flex_third_contract_from_base(&base, &direction)
            .expect("build-once third contraction");
        let rel = rel_diff_array2_survival(&from_base, &exact);
        assert!(
            rel < 1e-12,
            "row {row} (o_infl={offset:.3e}): build-once third contraction diverged from the \
             exact route, rel {rel:e}"
        );
    }
    assert!(
        max_offset > 0.0,
        "the fixture must carry a nonzero influence offset"
    );
}

/// Block states of the flex no-wiggle fixture at a flat β (marginal, slope, score warp),
/// with every `η` rebuilt from β.
fn flex_no_wiggle_states_at_beta(
    family: &SurvivalMarginalSlopeFamily,
    beta: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let score_dim = family
        .score_warp
        .as_ref()
        .map_or(0, |runtime| runtime.basis_dim());
    assert_eq!(beta.len(), 2 + score_dim);
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    let m_beta = beta.slice(s![0..1]).to_owned();
    let g_beta = beta.slice(s![1..2]).to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            eta: m_design.dot(&m_beta),
            beta: m_beta,
        },
        ParameterBlockState {
            eta: g_design.dot(&g_beta),
            beta: g_beta,
        },
        ParameterBlockState {
            beta: beta.slice(s![2..]).to_owned(),
            eta: Array1::zeros(n),
        },
    ]
}

/// A flat β for the flex no-wiggle fixture with small alternating score-warp coefficients.
fn flex_no_wiggle_beta(family: &SurvivalMarginalSlopeFamily) -> Array1<f64> {
    let score_dim = family
        .score_warp
        .as_ref()
        .map_or(0, |runtime| runtime.basis_dim());
    Array1::from_shape_fn(2 + score_dim, |i| match i {
        0 => 0.15,
        1 => 0.25,
        _ if i % 2 == 0 => 0.03,
        _ => -0.02,
    })
}

/// Grade `analytic` against a Ridders-certified central difference of `at(t)` at `t = 0`.
fn assert_matches_ridders_2893(label: &str, analytic: &Array2<f64>, at: &dyn Fn(f64) -> Array2<f64>) {
    assert_all_match_ridders_2893(label, std::slice::from_ref(analytic), &|t| vec![at(t)]);
}

/// Grade every `analytic[k]` against a Ridders-certified central difference of `at(t)[k]` at
/// `t = 0`. `at` is evaluated once per ladder point, so a sweep of `p` matrices costs four
/// evaluations rather than four per matrix.
fn assert_all_match_ridders_2893(
    label: &str,
    analytic: &[Array2<f64>],
    at: &dyn Fn(f64) -> Vec<Array2<f64>>,
) {
    let h = 1e-3;
    let (plus, minus) = (at(h), at(-h));
    let (half_plus, half_minus) = (at(0.5 * h), at(-0.5 * h));
    for ladder in [&plus, &minus, &half_plus, &half_minus] {
        assert_eq!(
            ladder.len(),
            analytic.len(),
            "{label}: the difference oracle returned {} matrices for {}",
            ladder.len(),
            analytic.len()
        );
    }
    for (k, want_matrix) in analytic.iter().enumerate() {
        let coarse = (&plus[k] - &minus[k]) / (2.0 * h);
        let fine = (&half_plus[k] - &half_minus[k]) / h;
        let scale = want_matrix
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1e-12);
        for ((index, &want), (&c, &f)) in want_matrix
            .indexed_iter()
            .zip(coarse.iter().zip(fine.iter()))
        {
            let value = (4.0 * f - c) / 3.0;
            let uncertainty = (f - c).abs() / 3.0;
            let denominator = scale.max(want.abs()).max(value.abs());
            assert!(
                uncertainty <= 0.05 * denominator,
                "{label} [{k}] {index:?}: the difference oracle did not resolve (value={value:.6e}, uncertainty={uncertainty:.3e})"
            );
            assert!(
                (want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty,
                "{label} [{k}] {index:?}: analytic={want:.9e} fd={value:.9e} uncertainty={uncertainty:.3e} scale={scale:.3e}"
            );
        }
    }
}

/// gam#2893: the order-five flex contraction of one row, combined along the primary image of
/// a coefficient direction, matches a Ridders-certified central difference of the exact
/// fourth contraction along that direction. Without a time wiggle the primary map is linear,
/// so moving β along `d` moves the primaries along `J d`.
#[test]
fn survival_flex_fifth_contraction_matches_differenced_fourth_2893() {
    let family = make_flex_no_wiggle_test_family(40);
    let beta = flex_no_wiggle_beta(&family);
    let states = flex_no_wiggle_states_at_beta(&family, &beta);
    assert!(family.effective_flex_active(&states).unwrap());
    let primary = flex_primary_slices(&family);
    let slices = block_slices(&family, &states);
    let direction = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 2) % 7) as f64 / 7.0 - 0.4);
    let u = Array1::from_shape_fn(primary.total, |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    let v = Array1::from_shape_fn(primary.total, |i| ((i * 3 + 1) % 13) as f64 / 13.0 - 0.5);
    for row in [0usize, 7, 19] {
        let q_geom = family.row_dynamic_q_geometry(row, &states).expect("q geometry");
        let w = family
            .row_primary_direction_from_flat_dynamic_with_q_geometry(
                row, &states, &slices, &q_geom, &direction,
            )
            .expect("primary image of the direction");
        let base = family
            .build_row_flex_fifth_base_with_states(row, &states, &primary)
            .expect("fifth-order row base");
        let axes = family
            .row_flex_fifth_contract_all_primary_axes_from_base(&base, &u, &v)
            .expect("fifth contraction");
        let mut analytic = Array2::<f64>::zeros((primary.total, primary.total));
        for (axis, &weight) in w.iter().enumerate() {
            analytic.scaled_add(weight, &axes[axis]);
        }
        assert!(
            analytic.iter().any(|value| value.abs() > 1e-8),
            "row {row}: the fifth contraction must be nonzero on this fixture"
        );
        assert_matches_ridders_2893(&format!("row {row}"), &analytic, &|t| {
            family
                .row_flex_primary_fourth_contracted_exact(
                    row,
                    &flex_no_wiggle_states_at_beta(&family, &(&beta + &(&direction * t))),
                    &u,
                    &v,
                )
                .expect("fourth contraction")
        });
    }
}

/// #932 row 64: `flex_production_fourth_contraction_matches_scalar_fd_witness` checks the
/// production fourth contraction against a difference of the third only along g/h/w, because
/// q0, q1 and qd1 cannot be moved through block states one axis at a time. Without a time
/// wiggle the primary map is linear, so moving β along `d` moves the primaries along `J d`,
/// and `J d` carries q-axis components. This checks
/// `row_flex_primary_fourth_contracted_exact(u, J d)` against a Ridders-certified central
/// difference of `row_flex_primary_third_contracted_exact(u)` along `d`, at nonzero
/// score-warp coefficients, on three rows. It fails if the image of `d` leaves the q axes
/// still, so it cannot pass on the g/h/w components alone.
#[test]
fn survival_flex_fourth_contraction_matches_differenced_third_along_q_axes_932() {
    let family = make_flex_no_wiggle_test_family(40);
    let beta = flex_no_wiggle_beta(&family);
    let states = flex_no_wiggle_states_at_beta(&family, &beta);
    assert!(family.effective_flex_active(&states).unwrap());
    let primary = flex_primary_slices(&family);
    let slices = block_slices(&family, &states);
    let direction = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 2) % 7) as f64 / 7.0 - 0.4);
    let u = Array1::from_shape_fn(primary.total, |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    for row in [0usize, 7, 19] {
        let q_geom = family.row_dynamic_q_geometry(row, &states).expect("q geometry");
        let image = family
            .row_primary_direction_from_flat_dynamic_with_q_geometry(
                row, &states, &slices, &q_geom, &direction,
            )
            .expect("primary image of the direction");
        let q_mass = image[primary.q0].abs() + image[primary.q1].abs() + image[primary.qd1].abs();
        assert!(
            q_mass > 1e-6,
            "row {row}: the direction's primary image must move q0, q1 or qd1 (|image_q| = {q_mass:.3e})"
        );
        let analytic = family
            .row_flex_primary_fourth_contracted_exact(row, &states, &u, &image)
            .expect("fourth contraction");
        assert!(
            analytic.iter().any(|value| value.abs() > 1e-8),
            "row {row}: the fourth contraction must be nonzero on this fixture"
        );
        assert_matches_ridders_2893(&format!("row {row}"), &analytic, &|t| {
            family
                .row_flex_primary_third_contracted_exact(
                    row,
                    &flex_no_wiggle_states_at_beta(&family, &(&beta + &(&direction * t))),
                    &u,
                )
                .expect("third contraction")
        });
    }
}

/// gam#2893: the build-once flex no-wiggle sweep of `D²_β H[u, e_a]` reproduces the single-direction
/// second directional derivative on every coefficient axis.
#[test]
fn flex_no_wiggle_all_axes_second_directional_derivative_matches_single_axis_2893() {
    let family = make_flex_no_wiggle_test_family(40);
    let beta = flex_no_wiggle_beta(&family);
    let states = flex_no_wiggle_states_at_beta(&family, &beta);
    let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    let swept = family
        .exact_newton_joint_hessian_second_directional_derivative_flex_no_wiggle_all_axes(&states, &u)
        .expect("build-once all-axes second sweep");
    assert_eq!(swept.len(), beta.len());
    let single: Vec<Array2<f64>> = (0..beta.len())
        .map(|index| {
            let mut axis = Array1::<f64>::zeros(beta.len());
            axis[index] = 1.0;
            family
                .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &axis)
                .expect("single-axis D2_beta H")
                .expect("flex publishes D2_beta H")
        })
        .collect();
    let scale = single
        .iter()
        .flat_map(|matrix| matrix.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        scale > 1e-8,
        "D2_beta H[u, e_a] must be nonzero on this fixture"
    );
    for (index, (swept_axis, single_axis)) in swept.iter().zip(single.iter()).enumerate() {
        let gap = (swept_axis - single_axis)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            gap <= 1e-10 * scale,
            "axis {index}: build-once sweep vs single axis: gap {gap:e}, scale {scale:e}"
        );
    }
}

/// gam#2893: the flex no-wiggle joint third information derivative `D³H[u, v, e_a]`, served
/// by the Jeffreys hook, matches a Ridders-certified central difference of the build-once
/// `{D²H[v, e_a]}` sweep along `u` on every coefficient axis.
#[test]
fn survival_flex_joint_third_information_matches_differenced_second_directional_2893() {
    let family = make_flex_no_wiggle_test_family(40);
    let beta = flex_no_wiggle_beta(&family);
    let states = flex_no_wiggle_states_at_beta(&family, &beta);
    let score_dim = beta.len() - 2;
    let specs = vec![
        dummy_blockspec(0),
        dummy_blockspec(1),
        dummy_blockspec(1),
        dummy_blockspec(score_dim),
    ];
    assert!(family.jeffreys_third_information_derivative().is_some());
    let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
    let axes = family
        .jeffreys_third_information_derivative()
        .expect("the family exposes its third information derivative")
        .third_directional_all_axes(&states, &specs, &u, &v)
        .expect("third information derivative")
        .expect("flex without a time wiggle publishes the third information derivative");
    assert_eq!(axes.len(), beta.len());
    assert!(
        axes.iter().any(|matrix| matrix.iter().any(|value| value.abs() > 1e-8)),
        "the joint third information derivative must be nonzero on this fixture"
    );
    // Mixed partials commute: `D³H[u, v, e_a] = D_u D²H[v, e_a]`. One Ridders ladder along u of the
    // build-once `{D²H[v, e_a]}` sweep grades every axis in four displaced passes instead of four per
    // axis; flex_no_wiggle_all_axes_second_directional_derivative_matches_single_axis_2893 grades that
    // sweep against the single-direction routine.
    assert_all_match_ridders_2893("D3H[u, v, e_a]", &axes, &|t| {
        family
            .exact_newton_joint_hessian_second_directional_derivative_flex_no_wiggle_all_axes(
                &flex_no_wiggle_states_at_beta(&family, &(&beta + &(&u * t))),
                &v,
            )
            .expect("displaced D2_beta H[v, e_a] sweep")
    });
}

#[test]
fn survival_jointhessian_flex_no_wiggle_operator_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 40usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled");
    let scaled_dense = scaled.to_dense();

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw");
    let raw_dense = raw.to_dense();

    let factor = n as f64 / m as f64;
    let exp = &raw_dense * factor;
    let rel = rel_diff_array2_survival(&scaled_dense, &exp);
    assert!(
        rel < 1e-10,
        "joint Hessian flex-no-wiggle dH operator HT rel {}",
        rel
    );
}

// ────────────────────────────────────────────────────────────────────
// Independent fourth-contraction finite-difference fixtures.
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct FlexContractionFixture {
    label: &'static str,
    event: f64,
    weight: f64,
    z: f64,
    q0: f64,
    q1: f64,
    qd1: f64,
    score_eta: f64,
    h_scale: f64,
    w_scale: f64,
}

const FLEX_CONTRACTION_FIXTURES: &[FlexContractionFixture] = &[
    FlexContractionFixture {
        label: "event_nonzero_warps",
        event: 1.0,
        weight: 0.75,
        z: -0.2,
        q0: -0.4,
        q1: 0.6,
        qd1: 0.85,
        score_eta: 0.32,
        h_scale: 0.05,
        w_scale: 0.04,
    },
    FlexContractionFixture {
        label: "censored_left_tail",
        event: 0.0,
        weight: 1.35,
        z: -1.15,
        q0: -1.35,
        q1: -0.9,
        qd1: 0.42,
        score_eta: -0.55,
        h_scale: -0.035,
        w_scale: 0.025,
    },
    FlexContractionFixture {
        label: "near_boundary_derivative",
        event: 1.0,
        weight: 0.2,
        z: 0.95,
        q0: 0.15,
        q1: 1.05,
        qd1: 0.08,
        score_eta: 0.72,
        h_scale: 0.015,
        w_scale: -0.02,
    },
    FlexContractionFixture {
        label: "zero_warp_edge",
        event: 0.0,
        weight: 0.9,
        z: 0.0,
        q0: -0.05,
        q1: 0.25,
        qd1: 1.2,
        score_eta: 0.0,
        h_scale: 0.0,
        w_scale: 0.0,
    },
];

fn flex_contraction_fixture_family(
    fixture: FlexContractionFixture,
) -> (SurvivalMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![fixture.event]),
        weights: Arc::new(array![fixture.weight]),
        z: Arc::new(array![fixture.z].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![fixture.q0]),
        offset_exit: Arc::new(array![fixture.q1]),
        derivative_offset_exit: Arc::new(array![fixture.qd1]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let h_beta: Array1<f64> = (0..h_dim)
        .map(|k| fixture.h_scale * ((k as f64 + 1.0).sin()))
        .collect::<Vec<_>>()
        .into();
    let w_beta: Array1<f64> = (0..w_dim)
        .map(|k| fixture.w_scale * ((k as f64 + 1.0).cos()))
        .collect::<Vec<_>>()
        .into();
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![fixture.score_eta],
        },
        ParameterBlockState {
            beta: h_beta,
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: w_beta,
            eta: Array1::zeros(1),
        },
    ];
    (family, block_states)
}

fn flex_contraction_directions(p: usize) -> Vec<(&'static str, Array1<f64>)> {
    let mixed: Array1<f64> = (0..p)
        .map(|k| 0.1 + 0.07 * ((k as f64 + 1.7).sin()))
        .collect::<Vec<_>>()
        .into();
    let alternating: Array1<f64> = (0..p)
        .map(|k| if k % 2 == 0 { 0.16 } else { -0.11 })
        .collect::<Vec<_>>()
        .into();
    let mut qd_axis = Array1::zeros(p);
    if p > 2 {
        qd_axis[2] = 1.0;
    }
    let zero = Array1::zeros(p);
    vec![
        ("mixed", mixed),
        ("alternating", alternating),
        ("qd_axis", qd_axis),
        ("zero", zero),
    ]
}

/// Shift the g/h/w primary axes of a flex `block_states` by `step · dir` (the
/// q0/q1/qd1 offset axes are left fixed). `dir` is a full primary-space vector;
/// only its `g` (block_states[2].eta), `h` (block_states[3].beta), and `w`
/// (block_states[4].beta) components are applied — exactly the axes the
/// directional contraction differentiates through the score/link coefficient
/// chain (the #1454-sensitive cross channels).
fn perturb_flex_ghw_block_states(
    block_states: &[ParameterBlockState],
    primary: &FlexPrimarySlices,
    dir: &Array1<f64>,
    step: f64,
) -> Vec<ParameterBlockState> {
    let mut bs: Vec<ParameterBlockState> = block_states.to_vec();
    // g axis: marginal slope eta (block index 2).
    bs[2].eta[0] += step * dir[primary.g];
    // h axis: score-warp beta (block index 3).
    if let Some(h_range) = primary.h.as_ref() {
        for (local, idx) in h_range.clone().enumerate() {
            bs[3].beta[local] += step * dir[idx];
        }
    }
    // w axis: link-dev beta (block index 4).
    if let Some(w_range) = primary.w.as_ref() {
        for (local, idx) in w_range.clone().enumerate() {
            bs[4].beta[local] += step * dir[idx];
        }
    }
    bs
}

/// #932-2 / #1454: the production fourth contraction
/// `D_u D_v H = row_flex_primary_fourth_contracted_exact` is the single-source
/// jet path (`flex_jet` `flex_timepoint_inputs_generic` at `Jet4`). It is checked
/// here against an INDEPENDENT scalar finite difference of the production THIRD
/// contraction `D_u H = row_flex_primary_third_contracted_exact` along the second
/// direction `v` (g/h/w axes), re-solving the moving-boundary intercept exactly at
/// every perturbed point. The third contraction is
/// itself pinned to the same FD ground truth by
/// `flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation`, so a
/// central difference of it is a faithful fourth-order ground truth.
#[test]
fn flex_production_fourth_contraction_matches_scalar_fd_witness() {
    for &fixture in FLEX_CONTRACTION_FIXTURES {
        let (family, block_states) = flex_contraction_fixture_family(fixture);
        let primary = flex_primary_slices(&family);
        let p = primary.total;
        let dirs = flex_contraction_directions(p);
        let pairs = [(0usize, 0usize), (0, 1), (1, 0), (1, 2), (2, 3), (3, 0)];
        for &(u_idx, v_idx) in &pairs {
            let (u_label, dir_u) = &dirs[u_idx];
            let (v_label, dir_v_full) = &dirs[v_idx];
            // Restrict the FD (second) direction to the g/h/w block-state axes (the
            // q0/q1/qd1 offset axes are not block-state-perturbable here); the
            // production fourth is contracted against the SAME restricted direction so
            // both sides see the identical `v`.
            let mut dir_v = Array1::<f64>::zeros(p);
            dir_v[primary.g] = dir_v_full[primary.g];
            if let Some(h_range) = primary.h.as_ref() {
                for idx in h_range.clone() {
                    dir_v[idx] = dir_v_full[idx];
                }
            }
            if let Some(w_range) = primary.w.as_ref() {
                for idx in w_range.clone() {
                    dir_v[idx] = dir_v_full[idx];
                }
            }
            let production = family
                .row_flex_primary_fourth_contracted_exact(0, &block_states, dir_u, &dir_v)
                .unwrap_or_else(|err| {
                    panic!(
                        "{} / {u_label}->{v_label}: production fourth contraction failed: {err}",
                        fixture.label
                    )
                });
            assert_eq!(production.dim(), (p, p));
            assert!(production.iter().all(|value| value.is_finite()));
            if dir_v.iter().all(|x| *x == 0.0) || dir_u.iter().all(|x| *x == 0.0) {
                assert!(production.iter().all(|value| *value == 0.0));
                continue;
            }

            // Central difference (Richardson) of the production THIRD contraction
            // `D_u H` along the g/h/w direction `dir_v`.
            let third_at = |step: f64| -> ndarray::Array2<f64> {
                let bs = perturb_flex_ghw_block_states(&block_states, &primary, &dir_v, step);
                family
                    .row_flex_primary_third_contracted_exact(0, &bs, dir_u)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{} / {u_label}->{v_label}: perturbed third contraction failed: {err}",
                            fixture.label
                        )
                    })
            };
            let central =
                |h: f64| -> ndarray::Array2<f64> { (&third_at(h) - &third_at(-h)) / (2.0 * h) };
            let h0 = 4e-3;
            let coarse = central(h0);
            let fine = central(h0 * 0.5);
            // Richardson extrapolation (O(h⁴)) of the central difference.
            let fd_fourth = (&fine * 4.0 - &coarse) / 3.0;

            for u in 0..p {
                for v in 0..p {
                    let got = production[[u, v]];
                    let want = fd_fourth[[u, v]];
                    let scale = want.abs().max(1.0);
                    // A disagreement between stencil scales is a failed
                    // witness, not permission to omit a required channel.
                    assert!(
                        [got, want, fine[[u, v]], coarse[[u, v]]]
                            .iter()
                            .all(|value| value.is_finite())
                            && (fine[[u, v]] - coarse[[u, v]]).abs()
                                <= 2e-2 * scale + 1e-5,
                        "{} / {u_label}->{v_label} fourth[{u},{v}]: FD did not converge: \
                         coarse={:+.6e}, fine={:+.6e}, production={got:+.6e}",
                        fixture.label,
                        coarse[[u, v]],
                        fine[[u, v]],
                    );
                    assert!(
                        (got - want).abs() <= 2e-2 * scale + 1e-5,
                        "{} / {u_label}->{v_label} fourth[{u},{v}]: production {got:+.6e} != \
                         scalar-FD-of-third {want:+.6e}",
                        fixture.label
                    );
                }
            }
        }
    }
}

/// Build a one-row time-block family whose single derivative-design row has
/// the given coefficient and offset, so we can drive
/// `validate_time_qd1_feasible` with a controlled raw `qd1` and constraint
/// row scaling. Mirrors the field layout of the other in-module fixtures.
fn make_time_guard_family(deriv_coeff: f64, deriv_offset: f64) -> SurvivalMarginalSlopeFamily {
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(array![[deriv_coeff]]),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![deriv_offset]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 1))),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

/// Regression for #379: a full-rank PH marginal-slope fit aborted because
/// `validate_time_qd1_feasible` rejected a constrained time-block iterate
/// that the inequality-constrained active-set Newton solver considered
/// feasible. The solver only guarantees primal feasibility to
/// `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` in the *scaled* constraint-row units
/// produced by `time_derivative_guard_constraints` (row scale
/// `max(||design_row||, |guard - offset|, 1)`). On a design row with a
/// large norm, a scaled slack of ~1e-8 maps to a raw `qd1` shortfall of
/// ~1e-5 below the 1e-6 guard — exactly the overshoot in the issue. The
/// validator must accept that boundary iterate while still rejecting a
/// genuine monotonicity divergence.
#[test]
fn validate_time_qd1_accepts_scaled_boundary_overshoot_rejects_real_violation() {
    let guard = 1e-6_f64;

    // Boundary case: design row norm ~6000, so the solver's 1e-8 scaled
    // primal tolerance permits a raw qd1 shortfall of ~6e-5 below the guard.
    let deriv_coeff = 6000.0_f64;
    let family = make_time_guard_family(deriv_coeff, 0.0);
    // Choose beta so qd1 = guard - 6e-5 (raw overshoot ~6e-5, as observed).
    let target_qd1 = guard - 6e-5;
    let beta = array![target_qd1 / deriv_coeff];
    // Confirm the fixture actually reproduces the issue's regime: the raw
    // qd1 trips the historic 256·eps guard band (so the *old* validator
    // would have rejected this iterate and aborted the whole fit).
    assert!(
        super::survival_derivative_guard_violated(target_qd1, guard),
        "fixture must reproduce the sub-guard raw overshoot the old validator rejected",
    );
    // Scaled violation = 6e-5 / max(6000, |guard|, 1) = 1e-8, below the
    // solver-consistent feasibility band, so the iterate must be accepted.
    family
        .validate_time_qd1_feasible(&beta, "proposed")
        .expect("solver-feasible boundary iterate must be accepted");

    // Genuine violation: unit-scale row, qd1 driven far below the guard.
    let bad_family = make_time_guard_family(1.0, 0.0);
    let bad_beta = array![-0.5]; // qd1 = -0.5, scaled violation ~0.5 >> band
    let err = bad_family
        .validate_time_qd1_feasible(&bad_beta, "proposed")
        .expect_err("a true monotonicity divergence must still hard-error");
    assert!(
        err.contains("violates monotonicity"),
        "expected a monotonicity error, got: {err}",
    );
}

/// Fill every one of the 15 cross-block matrices of a
/// `BlockHessianAccumulator` with distinct, deterministic values keyed by
/// (block-pair, local-row, local-col) so that a transposed or mis-placed
/// block is impossible to miss in the dense/operator parity assertions.
fn fill_block_hessian_accumulator(
    p_t: usize,
    p_m: usize,
    p_g: usize,
    p_h: usize,
    p_w: usize,
) -> BlockHessianAccumulator {
    let mut acc = BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, 0);
    // Per-pair base offsets keep the diagonal blocks symmetric (required
    // for a valid Hessian) while making the off-diagonal blocks distinct
    // and asymmetric, so `to_dense` must place each transpose correctly.
    let sym = |m: &mut Array2<f64>, base: f64| {
        let (r, c) = m.dim();
        for i in 0..r {
            for j in 0..c {
                m[[i, j]] = base + (i.min(j) as f64) * 0.5 + (i.max(j) as f64) * 0.25;
            }
        }
    };
    sym(&mut acc.h_tt, 1.0);
    sym(&mut acc.h_mm, 2.0);
    sym(&mut acc.h_gg, 3.0);
    sym(&mut acc.h_hh, 4.0);
    sym(&mut acc.h_ww, 5.0);
    let rect = |m: &mut Array2<f64>, base: f64| {
        let (r, c) = m.dim();
        for i in 0..r {
            for j in 0..c {
                m[[i, j]] = base + (i as f64) * 1.0 + (j as f64) * 0.1;
            }
        }
    };
    rect(&mut acc.h_tm, 10.0);
    rect(&mut acc.h_tg, 20.0);
    rect(&mut acc.h_th, 30.0);
    rect(&mut acc.h_tw, 40.0);
    rect(&mut acc.h_mg, 50.0);
    rect(&mut acc.h_mh, 60.0);
    rect(&mut acc.h_mw, 70.0);
    rect(&mut acc.h_gh, 80.0);
    rect(&mut acc.h_gw, 90.0);
    rect(&mut acc.h_hw, 100.0);
    acc
}

fn full_block_slices(p_t: usize, p_m: usize, p_g: usize, p_h: usize, p_w: usize) -> BlockSlices {
    let time = 0..p_t;
    let marginal = time.end..time.end + p_m;
    let slope = marginal.end..marginal.end + p_g;
    let score_warp = slope.end..slope.end + p_h;
    let link_dev = score_warp.end..score_warp.end + p_w;
    let total = link_dev.end;
    BlockSlices {
        time,
        marginal,
        slope,
        score_warp: Some(score_warp),
        link_dev: Some(link_dev),
        influence: None,
        total,
    }
}

/// Parity guard for issue #428: the dense and operator block-Hessian
/// assembly are now one storage abstraction with a single scatter, so
/// `BlockHessianOperator` (the operator wrapping the accumulator) must
/// agree *exactly* with the dense scatter — to_dense, matvec, and the
/// bilinear form — across the full five-block layout (time, marginal,
/// slope, score_warp, link_dev) that the directional-derivative path
/// produces. Any future reintroduction of a second, divergent block
/// layout would break this test.
#[test]
fn block_hessian_dense_operator_parity_all_five_blocks() {
    let (p_t, p_m, p_g, p_h, p_w) = (3usize, 2, 2, 3, 2);
    let slices = full_block_slices(p_t, p_m, p_g, p_h, p_w);
    let acc = fill_block_hessian_accumulator(p_t, p_m, p_g, p_h, p_w);

    // The dense reference scatter (single source of truth).
    let dense = acc.to_dense(&slices);
    assert_eq!(dense.dim(), (slices.total, slices.total));

    // The full block Hessian must be symmetric: each off-diagonal block
    // appears with its transpose at the mirrored location.
    for i in 0..slices.total {
        for j in 0..slices.total {
            assert_relative_eq!(dense[[i, j]], dense[[j, i]], max_relative = 1e-12);
        }
    }

    // Operator densification must equal the dense scatter bit-for-bit:
    // both now route through `BlockHessianAccumulator::to_dense`.
    let op = acc.into_operator(slices.clone());
    let op_dense = op.to_dense();
    assert_eq!(op_dense.dim(), dense.dim());
    for i in 0..slices.total {
        for j in 0..slices.total {
            assert_eq!(
                op_dense[[i, j]],
                dense[[i, j]],
                "operator/dense mismatch at ({i}, {j})",
            );
        }
    }

    // Matvec parity: operator.mul_vec(v) == dense.dot(v) for arbitrary v,
    // including the directional-derivative use (the operator is exactly
    // what the operator-variant directional path returns).
    let v: Array1<f64> =
        Array1::from_iter((0..slices.total).map(|k| 0.7 + (k as f64) * 0.31 - 0.05 * k as f64));
    let u: Array1<f64> = Array1::from_iter((0..slices.total).map(|k| -1.3 + (k as f64) * 0.17));
    let mv = op.mul_vec(&v);
    let dense_mv = dense.dot(&v);
    for k in 0..slices.total {
        assert_relative_eq!(mv[k], dense_mv[k], max_relative = 1e-12);
    }
    // The view-based matvec must match the owned one exactly.
    let mv_view = op.mul_vec_view(v.view());
    for k in 0..slices.total {
        assert_eq!(mv_view[k], mv[k]);
    }

    // Bilinear parity: operator.bilinear(v, u) == vᵀ · dense · u.
    let bil = op.bilinear(&v, &u);
    let dense_bil = v.dot(&dense.dot(&u));
    assert_relative_eq!(bil, dense_bil, max_relative = 1e-12);
}

#[test]
fn zz_diag_failure1_flex_vs_rigid_vs_fdhess() {
    use gam_math::jet_tower::program_third_contracted;
    // FAILURE 1 fixture row.
    let event = 1.0_f64;
    let weight = 0.75_f64;
    let zr = -0.2_f64;
    let q0 = -0.4_f64;
    let q1 = 0.6_f64;
    let qd1 = 0.85_f64;
    let gv = 0.32_f64;

    let make = |q0: f64, q1: f64, qd1: f64, g: f64| {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let family = SurvivalMarginalSlopeFamily {
            jeffreys_armed: true,
            latent_law: None,
            n: 1,
            entry_at_origin: Arc::new(Array1::from_elem(1, false)),
            event: Arc::new(array![event]),
            weights: Arc::new(array![weight]),
            z: Arc::new(array![zr].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![q0]),
            offset_exit: Arc::new(array![q1]),
            derivative_offset_exit: Arc::new(array![qd1]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            flex_jet_arenas: new_flex_jet_arena_pool(),
        };
        let sd = score_runtime.basis_dim();
        let ld = link_runtime.basis_dim();
        let bs = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![g],
            },
            ParameterBlockState {
                beta: Array1::zeros(sd),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(ld),
                eta: Array1::zeros(1),
            },
        ];
        (family, bs)
    };

    let (family, bs) = make(q0, q1, qd1, gv);
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let bidx = [primary.q0, primary.q1, primary.qd1, primary.g];

    // Flex Hessian at a primary point (q0,q1,qd1,g).
    let flex_hess = |q0: f64, q1: f64, qd1: f64, g: f64| -> Array2<f64> {
        let (fam, bs) = make(q0, q1, qd1, g);
        let qg = fam.row_dynamic_q_geometry(0, &bs).unwrap();
        let pr = flex_primary_slices(&fam);
        let (_, _, h) = fam
            .compute_row_flex_primary_gradient_hessian_exact(0, &bs, &qg, &pr)
            .unwrap();
        h
    };

    let dir4 = [0.7f64, -1.3, 0.5, 0.9];
    let dirvec = {
        let mut v = Array1::zeros(p);
        for (k, &s) in bidx.iter().enumerate() {
            v[s] = dir4[k];
        }
        v
    };

    // Production flex third-contracted.
    let flex_third = family
        .row_flex_primary_third_contracted_exact(0, &bs, &dirvec)
        .unwrap();

    // Independent FD of flex Hessian along dir4 (central, Richardson).
    let fd_dir = |h: f64| -> Array2<f64> {
        let hp = flex_hess(
            q0 + h * dir4[0],
            q1 + h * dir4[1],
            qd1 + h * dir4[2],
            gv + h * dir4[3],
        );
        let hm = flex_hess(
            q0 - h * dir4[0],
            q1 - h * dir4[1],
            qd1 - h * dir4[2],
            gv - h * dir4[3],
        );
        (&hp - &hm) / (2.0 * h)
    };
    let fd_coarse = fd_dir(1e-3);
    let fd_fine = fd_dir(5e-4);
    let fd_rich = (&fd_fine * 4.0 - &fd_coarse) / 3.0;

    // Rigid tower.
    let program = SurvivalMarginalSlopeRigidNllProgram {
        primaries: vec![[q0, q1, qd1, gv]],
        z: vec![zr],
        w: vec![weight],
        d: vec![event],
        probit_scale: family.probit_frailty_scale(),
    };
    let rigid = program_third_contracted(&program, 0, &dir4).unwrap();

    const THIRD_CONTRACTED_TOL: f64 = 1e-5;
    for (u, &bu) in bidx.iter().enumerate() {
        for (v, &bv) in bidx.iter().enumerate() {
            let flex = flex_third[[bu, bv]];
            let fd = fd_rich[[bu, bv]];
            let rigid = rigid[u][v];

            assert_close(
                flex,
                fd,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted flex vs FD at primary block ({u}, {v})"),
            );
            assert_close(
                flex,
                rigid,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted flex vs rigid at primary block ({u}, {v})"),
            );
            assert_close(
                fd,
                rigid,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted FD vs rigid at primary block ({u}, {v})"),
            );
        }
    }
}

/// gnomon#2337: the survival dense Hessian closes its row-chunk Grams on parallel
/// workers and adds them in chunk order, so the assembled Hessian is the same bits
/// whatever the worker count. `n` spans three 8,192-row Gram chunks.
#[test]
fn survival_dense_hessian_is_bitwise_invariant_to_the_worker_count_2337() {
    use crate::row_kernel::{RowKernel, RowSet};

    let n = 20_000usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.37).sin() * 1.1).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.7 + 0.5 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();
    let marginal_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });
    let beta_marginal = Array1::from_vec(vec![0.18, -0.12]);
    let beta_slope = Array1::from_vec(vec![-0.2, 0.13]);
    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.65],
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal_design.dot(&beta_marginal),
        },
        ParameterBlockState {
            beta: beta_slope.clone(),
            eta: slope_design.dot(&beta_slope),
        },
    ];
    let kernel = SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
        family,
        block_states,
    );
    let cache = crate::row_kernel::build_row_kernel_cache(&kernel, &RowSet::All)
        .expect("rigid row-kernel cache");
    let assemble = |workers: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool")
            .install(|| crate::row_kernel::row_kernel_hessian_dense(&kernel, &cache, &RowSet::All))
            .expect("dense survival Hessian")
    };
    let one_worker = assemble(1);
    let four_workers = assemble(4);
    let p = RowKernel::n_coefficients(&kernel);
    assert_eq!(one_worker.dim(), (p, p));
    for (index, (a, b)) in one_worker.iter().zip(four_workers.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "dense Hessian entry {index}: 1 worker {a:e} vs 4 workers {b:e}"
        );
    }
}

/// gam#3035: the dense Hessian and all-axes overrides agree with the generic
/// per-row reductions on the full data and on a Horvitz–Thompson-weighted
/// subsample. `n = 700` puts the subsample's walk past one 256-row chunk.
#[test]
fn rigid_survival_dense_overrides_match_generic_on_every_row_set_3035() {
    let n = 700usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.37).sin() * 1.1).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.7 + 0.5 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();
    let marginal_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });
    let beta_marginal = Array1::from_vec(vec![0.18, -0.12]);
    let beta_slope = Array1::from_vec(vec![-0.2, 0.13]);
    for frailty in [None, Some(0.55_f64)] {
        let mut family = oracle_rigid_family(n, &z, &weights, &event, frailty);
        family.marginal_design = DesignMatrix::from(marginal_design.clone());
        family
            .slope_layout
            .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.65],
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: beta_marginal.clone(),
                eta: marginal_design.dot(&beta_marginal),
            },
            ParameterBlockState {
                beta: beta_slope.clone(),
                eta: slope_design.dot(&beta_slope),
            },
        ];
        let kernel = SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
            family,
            block_states,
        );
        crate::test_support::row_set_overrides::assert_dense_overrides_match_generic(
            &format!("rigid survival marginal-slope frailty={frailty:?}"),
            &kernel,
            &[0.4, -0.6, 0.3, 0.8, -0.2],
            &[0.5, 0.3, -0.7, 0.9, -0.4],
            1e-13,
        );
    }
}

/// gam#979 build-once equality contract for the rigid survival marginal-slope
/// kernel.
///
/// The inner-Newton Jeffreys/Firth term needs, each cycle, the directional
/// derivative of the joint Hessian for every canonical axis `e_a` (and the
/// outer-REML `H_Φ` drift needs the second-directional analogue). Before this
/// fix the rigid survival kernel implemented NEITHER
/// `directional_derivative_all_axes_dense_override` NOR its second-order
/// sibling, so `row_kernel_directional_derivative_all_axes` fell into the
/// generic per-axis fall-back — `p` independent full-data sweeps, each
/// rebuilding the per-row `Tower4<4>` for every row (`n·p` tower evaluations).
/// That redundant tower rebuild is the survival #979 hot path.
///
/// The new overrides build each row's tower ONCE and contract every axis off
/// that single build. A batched override is only a valid optimisation if it is
/// bit-for-bit what the per-axis sweep returns. This gate builds a
/// representative rigid fixture (non-trivial time / marginal / slope designs,
/// `n = 300` rows so the `ARROW_ROW_CHUNK = 256` chunked reduction spans more
/// than one tile, mixed event / censored, with and without Gaussian frailty so
/// the probit scale ≠ 1) and asserts, for EVERY coefficient axis, that the
/// build-once override equals the per-axis sweep to machine precision — the
/// exact contract a wrong override would silently violate while corrupting
/// survival derivatives.
#[test]
fn rigid_survival_all_axes_build_once_equals_per_axis_sweep_979() {
    use crate::row_kernel::{
        RowKernel, RowSet, row_kernel_directional_derivative,
        row_kernel_directional_derivative_all_axes, row_kernel_second_directional_derivative,
        row_kernel_second_directional_derivative_all_axes,
    };

    let n = 300usize;
    // Mixed event / censored rows, deterministic and finite-margin.
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.37).sin() * 1.1).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.7 + 0.5 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();

    // Non-trivial marginal (2-col) and slope (2-col) designs so the
    // all-axes sweep exercises the coupled marginal block (which feeds BOTH
    // the entry and exit primaries through `jacobian_action`) and the slope
    // block, not just the single time axis.
    let p_m = 2usize;
    let p_g = 2usize;
    let marginal_design = Array2::from_shape_fn((n, p_m), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, p_g), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });
    let beta_marginal = Array1::from_vec(vec![0.18, -0.12]);
    let beta_slope = Array1::from_vec(vec![-0.2, 0.13]);
    // Realized linear predictors carried on the block states (the marginal /
    // slope eta channels the rigid primaries read), exactly as a real fit
    // installs them: eta = design · beta.
    let marginal_eta = marginal_design.dot(&beta_marginal);
    let slope_eta = slope_design.dot(&beta_slope);

    for frailty in [None, Some(0.55_f64)] {
        let mut family = oracle_rigid_family(n, &z, &weights, &event, frailty);
        family.marginal_design = DesignMatrix::from(marginal_design.clone());
        family
            .slope_layout
            .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));

        let beta_time = array![0.65];
        let block_states = vec![
            ParameterBlockState {
                beta: beta_time.clone(),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: beta_marginal.clone(),
                eta: marginal_eta.clone(),
            },
            ParameterBlockState {
                beta: beta_slope.clone(),
                eta: slope_eta.clone(),
            },
        ];

        let kernel =
            SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
                family,
                block_states,
            );
        let p = RowKernel::n_coefficients(&kernel);
        assert_eq!(
            p,
            1 + p_m + p_g,
            "fixture coefficient width should be time(1)+marginal({p_m})+slope({p_g})"
        );

        // ---- Dense Hessian: BLAS-3 override vs canonical row scatter ---------
        // Exercise the production dispatcher (which selects the dense-design
        // override) against the generic per-row implementation at the same
        // cached primary Hessians.  The GEMM reduction may reassociate the row
        // sum, so equality is numerical rather than bitwise; the allowed error
        // is scaled to the largest entry instead of hiding a structural block
        // or sign error behind a fixed absolute tolerance.
        let cache = crate::row_kernel::build_row_kernel_cache(&kernel, &RowSet::All)
            .expect("rigid row-kernel cache");
        let blas3 = crate::row_kernel::row_kernel_hessian_dense(&kernel, &cache, &RowSet::All)
            .expect("dense-design BLAS-3 Hessian");
        let scalar = crate::row_kernel::row_kernel_hessian_dense_generic(
            &kernel,
            &RowSet::All,
            &cache.hessians,
        );
        let scale = scalar
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let max_hessian_gap = blas3
            .iter()
            .zip(scalar.iter())
            .map(|(fast, reference)| (fast - reference).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_hessian_gap <= 2.0e-12 * scale,
            "frailty {frailty:?}: dense-Hessian BLAS-3 override differs from row-scatter \
             oracle: max_gap={max_hessian_gap:e}, scale={scale:e}"
        );

        // ---- FIRST directional derivative: override vs per-axis sweep --------
        let batched = row_kernel_directional_derivative_all_axes(&kernel, &RowSet::All)
            .expect("build-once all-axes first directional derivative");
        assert_eq!(
            batched.len(),
            p,
            "the batched first-directional sweep returns one p×p matrix per axis"
        );
        for a in 0..p {
            let mut e_a = vec![0.0_f64; p];
            e_a[a] = 1.0;
            let per_axis = row_kernel_directional_derivative(&kernel, &RowSet::All, &e_a)
                .expect("per-axis first directional derivative");
            assert_eq!(batched[a].dim(), (p, p));
            assert_eq!(per_axis.dim(), (p, p));
            let mut max_gap = 0.0_f64;
            for r in 0..p {
                for c in 0..p {
                    max_gap = max_gap.max((batched[a][[r, c]] - per_axis[[r, c]]).abs());
                }
            }
            assert!(
                max_gap < 1e-9,
                "frailty {frailty:?} axis {a}: #979 build-once FIRST directional override \
                 diverged from the per-axis sweep by {max_gap:e}; the optimisation changed \
                 the math, not just the schedule"
            );
        }

        // ---- SECOND directional derivative: override vs per-axis sweep -------
        // Fix a non-trivial direction `u` touching every block.
        let d_u = vec![0.5, 0.3, -0.7, 0.9, -0.4];
        let batched2 =
            row_kernel_second_directional_derivative_all_axes(&kernel, &RowSet::All, &d_u)
                .expect("build-once all-axes second directional derivative");
        assert_eq!(
            batched2.len(),
            p,
            "the batched second-directional sweep returns one p×p matrix per axis"
        );
        for a in 0..p {
            let mut e_a = vec![0.0_f64; p];
            e_a[a] = 1.0;
            let per_axis =
                row_kernel_second_directional_derivative(&kernel, &RowSet::All, &d_u, &e_a)
                    .expect("per-axis second directional derivative");
            let mut max_gap = 0.0_f64;
            for r in 0..p {
                for c in 0..p {
                    max_gap = max_gap.max((batched2[a][[r, c]] - per_axis[[r, c]]).abs());
                }
            }
            assert!(
                max_gap < 1e-9,
                "frailty {frailty:?} axis {a}: #979 build-once SECOND directional override \
                 diverged from the per-axis sweep by {max_gap:e}"
            );
        }

        // The fifth-order information path contracts its two fixed directions
        // into a symmetric primary third tensor. Check the shared assembly
        // against independent row pullbacks with nonconstant, signed tensors.
        let tensors: Vec<[[[f64; 4]; 4]; 4]> = (0..n).map(|row| {
            std::array::from_fn(|a| std::array::from_fn(|b| std::array::from_fn(|c| {
                ((row + 3 * (a + b + c)) as f64 * 0.17).sin()
            })))
        }).collect();
        let assembled = kernel.all_axes_primary_tensor_pullback(&RowSet::All, &tensors).unwrap();
        for axis in 0..p {
            let mut direction = vec![0.0; p];
            direction[axis] = 1.0;
            let mut expected = Array2::<f64>::zeros((p, p));
            for row in 0..n {
                let projected = kernel.jacobian_action(row, &direction);
                let hessian = std::array::from_fn(|a| std::array::from_fn(|b| {
                    (0..4).map(|c| tensors[row][a][b][c] * projected[c]).sum()
                }));
                kernel.add_pullback_hessian(row, &hessian, &mut expected);
            }
            let scale = expected.iter().fold(1.0_f64, |s, x| s.max(x.abs()));
            assert!(assembled[axis].iter().zip(&expected)
                .all(|(actual, expected)| (actual - expected).abs() < 2e-12 * scale));
        }
    }
}

/// gnomon#2337: the tiled symmetric all-axes tensor pullback reassociates its row,
/// primary and group sums, so it is held to accuracy rather than bits. Against a
/// double-double reference of `D_{ajk} = Σ_i Σ_{αβγ} T_i[α][β][γ] J_i[α,a] J_i[β,j] J_i[γ,k]`,
/// its largest error scaled by the entry's term mass `Σ_i Σ_{αβγ} |T J J J|` must not
/// exceed that of the ten-Gram assembly it replaced, rebuilt here, and must sit inside
/// the `γ_m` bound of its longest sum. Its fixed group order makes it independent of
/// the pool width, pinned bitwise at 1, 4 and 12 workers. `n = 5000` rows span 79
/// tiles, so groups fold more than one tile.
#[test]
fn rigid_survival_all_axes_tensor_pullback_is_accurate_and_width_invariant_2337() {
    use crate::row_kernel::RowKernel;

    let n = 5_000usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.37).sin() * 1.1).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.7 + 0.5 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();
    let marginal_design = Array2::from_shape_fn((n, 6), |(r, j)| {
        0.2 + 0.05 * ((r * (j + 1)) as f64 * 0.011).cos() + 0.11 * (j as f64)
            - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 4), |(r, j)| {
        0.1 + 0.07 * ((r + 3 * j) as f64 * 0.017).sin() - 0.09 * (j as f64)
    });
    let beta_marginal = Array1::from_shape_fn(6, |j| 0.03 * (j as f64) - 0.08);
    let beta_slope = Array1::from_shape_fn(4, |j| 0.05 - 0.04 * (j as f64));
    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.65],
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal_design.dot(&beta_marginal),
        },
        ParameterBlockState {
            beta: beta_slope.clone(),
            eta: slope_design.dot(&beta_slope),
        },
    ];
    let kernel = SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
        family,
        block_states,
    );
    let p = RowKernel::n_coefficients(&kernel);
    assert_eq!(p, 11);
    // Signed, row-varying and symmetric in (a, b, c): built from a + b + c and the
    // elementary symmetric products.
    let tensors: Vec<[[[f64; 4]; 4]; 4]> = (0..n)
        .map(|row| {
            std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    std::array::from_fn(|c| {
                        let sum = (a + b + c) as f64;
                        let products = (a * b + b * c + a * c + a * b * c) as f64;
                        ((row as f64) * 0.013 + 0.7 * sum).sin() * (1.0 + 0.1 * products)
                    })
                })
            })
        })
        .collect();

    // jacobian[row][axis][primary]
    let jacobian: Vec<Vec<[f64; 4]>> = (0..n)
        .map(|row| {
            (0..p)
                .map(|axis| {
                    let mut direction = vec![0.0_f64; p];
                    direction[axis] = 1.0;
                    kernel.jacobian_action(row, &direction)
                })
                .collect()
        })
        .collect();
    let two_sum = |a: f64, b: f64| {
        let sum = a + b;
        let b_part = sum - a;
        (sum, (a - (sum - b_part)) + (b - b_part))
    };
    let two_product = |a: f64, b: f64| {
        let product = a * b;
        (product, a.mul_add(b, -product))
    };
    // Double-double reference and absolute term mass at sorted indices a ≤ j ≤ k.
    let mut reference = std::collections::HashMap::new();
    for a in 0..p {
        for j in a..p {
            for k in j..p {
                let (mut high, mut low, mut mass) = (0.0_f64, 0.0_f64, 0.0_f64);
                for row in 0..n {
                    let (ja, jj, jk) = (&jacobian[row][a], &jacobian[row][j], &jacobian[row][k]);
                    for alpha in 0..4 {
                        for beta in 0..4 {
                            for gamma in 0..4 {
                                let t = tensors[row][alpha][beta][gamma];
                                let (p1, e1) = two_product(t, ja[alpha]);
                                let (p2, e2) = two_product(p1, jj[beta]);
                                let e2 = e2 + e1 * jj[beta];
                                let (p3, e3) = two_product(p2, jk[gamma]);
                                let e3 = e3 + e2 * jk[gamma];
                                let (sum, error) = two_sum(high, p3);
                                let (renormalized, rest) = two_sum(sum, low + error + e3);
                                high = renormalized;
                                low = rest;
                                mass += (t * ja[alpha] * jj[beta] * jk[gamma]).abs();
                            }
                        }
                    }
                }
                reference.insert((a, j, k), (high + low, mass));
            }
        }
    }
    let worst_scaled_error = |axes: &[Array2<f64>]| {
        let mut worst = 0.0_f64;
        for a in 0..p {
            for j in 0..p {
                for k in 0..p {
                    let mut index = [a, j, k];
                    index.sort_unstable();
                    let (exact, mass) = reference[&(index[0], index[1], index[2])];
                    if mass > 0.0 {
                        worst = worst.max((axes[a][[j, k]] - exact).abs() / mass);
                    }
                }
            }
        }
        worst
    };

    // The ten-Gram assembly this pullback replaced, per axis a:
    // Σ_{α≤β} sym_{αβ}(J_αᵀ diag(Σ_γ T[α][β][γ] J_γ[:,a]) J_β).
    let identity = Array2::<f64>::eye(p);
    let packed = kernel
        .jacobian_action_matrix(identity.view())
        .expect("dense J·I projection");
    let blocks: [Array2<f64>; 4] =
        std::array::from_fn(|primary| packed.slice(s![.., primary * p..(primary + 1) * p]).to_owned());
    let former: Vec<Array2<f64>> = (0..p)
        .map(|axis| {
            let mut total = Array2::<f64>::zeros((p, p));
            for left in 0..4 {
                for right in left..4 {
                    let row_weights = Array1::from_shape_fn(n, |row| {
                        (0..4)
                            .map(|direction| tensors[row][left][right][direction] * blocks[direction][[row, axis]])
                            .sum::<f64>()
                    });
                    let gram = gam_linalg::faer_ndarray::fast_xt_diag_y(
                        &blocks[left],
                        &row_weights,
                        &blocks[right],
                    );
                    total.scaled_add(1.0, &gram);
                    if left != right {
                        total.scaled_add(1.0, &gram.t());
                    }
                }
            }
            total
        })
        .collect();

    let pullback = |workers: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool")
            .install(|| kernel.all_axes_primary_tensor_pullback(&crate::row_kernel::RowSet::All, &tensors))
            .expect("all-axes tensor pullback")
    };
    let one_worker = pullback(1);
    let tiled_error = worst_scaled_error(&one_worker);
    let former_error = worst_scaled_error(&former);
    // Longest sum: three products, three primary sums of four and the n rows.
    let terms = (3 + 3 * 4 + n) as f64;
    let gamma_bound = terms * f64::EPSILON / 2.0 / (1.0 - terms * f64::EPSILON / 2.0);
    eprintln!(
        "[2337 pullback] worst scaled error: tiled {tiled_error:e}, former ten-Gram {former_error:e}, gamma_m bound {gamma_bound:e}"
    );
    assert!(
        tiled_error <= former_error,
        "tiled pullback worst scaled error {tiled_error:e} exceeds the former assembly's {former_error:e}"
    );
    assert!(
        tiled_error <= gamma_bound,
        "tiled pullback worst scaled error {tiled_error:e} exceeds the gamma_m bound {gamma_bound:e}"
    );
    for axis in &one_worker {
        for j in 0..p {
            for k in 0..p {
                assert_eq!(axis[[j, k]].to_bits(), axis[[k, j]].to_bits(), "axis matrix not symmetric");
            }
        }
    }
    for workers in [4, 12] {
        let wide = pullback(workers);
        for (axis, (narrow, wide)) in one_worker.iter().zip(&wide).enumerate() {
            for (index, (x, y)) in narrow.iter().zip(wide.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "axis {axis} entry {index}: 1 worker {x:e} vs {workers} workers {y:e}"
                );
            }
        }
    }
}

/// #979: the batched all-axes second directional derivative builds the row
/// towers once for a whole batch of directions. Each direction's object must be
/// bit-identical to the single-direction build-once sweep, which rebuilds them.
#[test]
fn rigid_survival_second_all_axes_each_matches_single_direction_979() {
    use crate::row_kernel::{RowKernel, RowSet, row_kernel_second_directional_derivative_all_axes};

    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.29).cos() * 0.9).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.6 + 0.4 * ((r % 7) as f64) / 7.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 4 == 1) as u8) as f64).collect();
    let marginal_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.25 + 0.06 * (r as f64).sin() + 0.08 * (j as f64)
    });
    let slope_design = Array2::from_shape_fn((n, 2), |(r, j)| {
        0.12 - 0.05 * (r as f64).cos() + 0.07 * (j as f64)
    });
    let beta_marginal = Array1::from_vec(vec![0.14, -0.09]);
    let beta_slope = Array1::from_vec(vec![-0.17, 0.11]);
    let marginal_eta = marginal_design.dot(&beta_marginal);
    let slope_eta = slope_design.dot(&beta_slope);
    for frailty in [None, Some(0.55_f64)] {
        let mut family = oracle_rigid_family(n, &z, &weights, &event, frailty);
        family.marginal_design = DesignMatrix::from(marginal_design.clone());
        family
            .slope_layout
            .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.65],
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: beta_marginal.clone(),
                eta: marginal_eta.clone(),
            },
            ParameterBlockState {
                beta: beta_slope.clone(),
                eta: slope_eta.clone(),
            },
        ];
        let kernel =
            SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
                family,
                block_states,
            );
        let p = RowKernel::n_coefficients(&kernel);
        let directions: Vec<Vec<f64>> = (0..3)
            .map(|d| (0..p).map(|a| 0.3 * ((a + 2 * d) as f64 * 0.7).sin() - 0.05 * d as f64).collect())
            .collect();
        let slices: Vec<&[f64]> = directions.iter().map(|direction| direction.as_slice()).collect();
        let mut batched: Vec<Option<Vec<Array2<f64>>>> = vec![None; directions.len()];
        kernel
            .second_directional_derivative_all_axes_each(&slices, &mut |index, axes| {
                batched[index] = Some(axes);
                Ok(())
            })
            .expect("batched second directional derivative");
        for (index, direction) in directions.iter().enumerate() {
            let single = row_kernel_second_directional_derivative_all_axes(&kernel, &RowSet::All, direction)
                .expect("single-direction second directional derivative");
            let together = batched[index].as_ref().expect("every direction is consumed");
            assert_eq!(together.len(), single.len());
            assert!(
                single.iter().any(|axis| axis.iter().any(|value| *value != 0.0)),
                "frailty={frailty:?} direction={index}: the fixture must exercise a nonzero derivative"
            );
            for (axis, (left, right)) in together.iter().zip(&single).enumerate() {
                assert!(
                    left.iter().zip(right.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "frailty={frailty:?} direction={index} axis={axis}: batched differs from single"
                );
            }
        }
    }
}

/// gam#2894: the contracted trace Hessian's first and second directional derivatives,
/// which the outer gradient and Hessian of a criterion priced on the complete Jeffreys
/// curvature read, checked against central differences of the exact contracted trace
/// Hessian and of the first derivative, on the rigid fixture of the contracted-trace gate
/// below.
#[test]
fn survival_contracted_trace_hessian_directional_derivatives_match_fd_2894() {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.29).sin() * 0.9).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.6 + 0.4 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();

    let p_m = 2usize;
    let p_g = 2usize;
    let marginal_design = Array2::from_shape_fn((n, p_m), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, p_g), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });

    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));
    assert!(family.jeffreys_completion_outer_derivatives().is_some());

    let total = 1 + p_m + p_g;
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(p_m),
        dummy_blockspec(p_g),
    ];
    // beta_flat = [time(1), marginal(2), slope(2)].
    let states_at = |beta_flat: &Array1<f64>| -> Vec<ParameterBlockState> {
        let beta_time = beta_flat.slice(ndarray::s![0..1]).to_owned();
        let beta_marginal = beta_flat.slice(ndarray::s![1..1 + p_m]).to_owned();
        let beta_slope = beta_flat.slice(ndarray::s![1 + p_m..total]).to_owned();
        let marginal_eta = marginal_design.dot(&beta_marginal);
        let slope_eta = slope_design.dot(&beta_slope);
        vec![
            ParameterBlockState {
                beta: beta_time,
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: beta_marginal,
                eta: marginal_eta,
            },
            ParameterBlockState {
                beta: beta_slope,
                eta: slope_eta,
            },
        ]
    };

    let beta0 = array![0.6, 0.18, -0.12, -0.2, 0.13];
    let mut raw_weight = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in 0..total {
            raw_weight[[i, j]] = ((i * 7 + j * 11 + 2) % 13) as f64 * 0.1 - 0.6;
        }
    }
    let trace_weight = (&raw_weight + &raw_weight.t()).mapv(|value| value * 0.5);
    let direction_u = array![0.3, -0.2, 0.1, 0.25, -0.15];
    let direction_w = array![0.2, 0.1, -0.35, 0.15, 0.3];

    let contracted_at = |beta_flat: &Array1<f64>| {
        family
            .joint_jeffreys_information_contracted_trace_hessian_with_specs(
                &states_at(beta_flat),
                &specs,
                &trace_weight,
            )
            .expect("contracted trace Hessian call")
            .expect("the rigid path must supply the contracted trace Hessian")
    };
    let directional_at = |beta_flat: &Array1<f64>| {
        family
            .jeffreys_completion_outer_derivatives()
            .expect("the rigid static path exposes the completion outer derivatives")
            .contracted_trace_hessian_directional(
                &states_at(beta_flat),
                &specs,
                &trace_weight,
                &direction_u,
            )
            .expect("contracted trace Hessian directional call")
            .expect("the rigid static path must supply the directional contraction")
    };
    let second = family
        .jeffreys_completion_outer_derivatives()
        .expect("the rigid static path exposes the completion outer derivatives")
        .contracted_trace_hessian_second_directional(
            &states_at(&beta0),
            &specs,
            &trace_weight,
            &direction_u,
            &direction_w,
        )
        .expect("contracted trace Hessian second directional call")
        .expect("the rigid static path must supply the second directional contraction");
    let first = directional_at(&beta0);
    assert_eq!(first.dim(), (total, total));
    assert_eq!(second.dim(), (total, total));

    let step = 1.0e-5;
    let fd_first = (contracted_at(&(&beta0 + &(&direction_u * step)))
        - contracted_at(&(&beta0 - &(&direction_u * step))))
        / (2.0 * step);
    let fd_second = (directional_at(&(&beta0 + &(&direction_w * step)))
        - directional_at(&(&beta0 - &(&direction_w * step))))
        / (2.0 * step);
    for (label, analytic, fd) in [("first", &first, &fd_first), ("second", &second, &fd_second)] {
        let largest = analytic.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            largest > 1.0e-3,
            "{label}: the fixture must exercise a nonzero derivative, max={largest:e}"
        );
        for (actual, expected) in analytic.iter().zip(fd.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5 * (1.0 + expected.abs()),
                "{label}: analytic={actual:e} fd={expected:e}"
            );
        }
    }
}


/// gam#979 Jeffreys wide-p contracted-trace-Hessian FD verification (survival
/// twin of the BMS gate `bernoulli_jeffreys_contracted_trace_hessian_matches_fd_of_trace`).
///
/// Builds the same kind of non-trivial rigid fixture as the build-once gate
/// above (`oracle_rigid_family` + 2-column marginal/slope designs, so the
/// hook's `time` block genuinely shares 3 different design rows with the
/// `q0/q1/qd1` primaries and `marginal_design` genuinely couples `q0` and
/// `q1`), picks a fixed deterministic symmetric 5×5 trace weight `W`, and
/// checks `family.joint_jeffreys_information_contracted_trace_hessian_with_specs`
/// against central second differences of `tr(W · H(β))` (using the existing
/// `exact_newton_joint_hessian` to get `H` at perturbed β) over several
/// directions spanning all three blocks.
#[test]
fn survival_jeffreys_contracted_trace_hessian_matches_fd_of_trace() {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.29).sin() * 0.9).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.6 + 0.4 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();

    let p_m = 2usize;
    let p_g = 2usize;
    let marginal_design = Array2::from_shape_fn((n, p_m), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, p_g), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });

    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));

    let total = 1 + p_m + p_g;
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(p_m),
        dummy_blockspec(p_g),
    ];

    // beta_flat = [time(1), marginal(2), slope(2)].
    let states_at = |beta_flat: &Array1<f64>| -> Vec<ParameterBlockState> {
        let beta_time = beta_flat.slice(ndarray::s![0..1]).to_owned();
        let beta_marginal = beta_flat.slice(ndarray::s![1..1 + p_m]).to_owned();
        let beta_slope = beta_flat.slice(ndarray::s![1 + p_m..total]).to_owned();
        let marginal_eta = marginal_design.dot(&beta_marginal);
        let slope_eta = slope_design.dot(&beta_slope);
        vec![
            ParameterBlockState {
                beta: beta_time,
                // Unused by the closed-form likelihood (recomputed from
                // beta_time directly via the 3 time designs); zeros satisfy
                // the CustomFamily interface shape contract only.
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: beta_marginal,
                eta: marginal_eta,
            },
            ParameterBlockState {
                beta: beta_slope,
                eta: slope_eta,
            },
        ]
    };

    let beta0 = array![0.6, 0.18, -0.12, -0.2, 0.13];

    // Fixed asymmetric-then-symmetrized 5x5 trace weight `W` (deterministic
    // pseudo-noise pattern, no RNG dependency).
    let mut w_raw = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in 0..total {
            w_raw[[i, j]] = ((i * 7 + j * 11 + 2) % 13) as f64 * 0.1 - 0.6;
        }
    }
    let w_raw_t = w_raw.t().to_owned();
    let w = (&w_raw + &w_raw_t).mapv(|v| v * 0.5);

    let states0 = states_at(&beta0);
    let analytic = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(&states0, &specs, &w)
        .expect("contracted trace hessian call")
        .expect("rigid path must supply the contracted completion");
    assert_eq!(analytic.dim(), (total, total));

    let trace_of_hessian_at = |beta_flat: &Array1<f64>| -> f64 {
        let states = states_at(beta_flat);
        let h = family
            .exact_newton_joint_hessian(&states)
            .expect("exact_newton_joint_hessian")
            .expect("exact_newton_joint_hessian some");
        // tr(W H) = Σ_ij W_ij H_ij for symmetric W, H (H_ji = H_ij).
        (&w * &h).sum()
    };

    let directions = [
        array![1.0, 0.0, 0.0, 0.0, 0.0],
        array![0.0, 1.0, 0.0, 0.0, 0.0],
        array![0.0, 0.0, 1.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0, 1.0, 0.0],
        array![0.0, 0.0, 0.0, 0.0, 1.0],
        array![0.5, -0.4, 0.3, 0.6, -0.2],
        array![-0.3, 0.5, -0.6, 0.2, 0.4],
    ];

    // ── Truncation-free analytic cross-check: the AUTHORITATIVE assembly gate ──
    // The finite-difference loop further below is discretization-limited (see
    // its comment): `tr(W·H)` is already second-order in β, so its second
    // difference probes the FOURTH derivative and the Richardson residual floors
    // at `O(h⁴·f⁽⁸⁾)`, which the neglog-Φ Mills-ratio tails push to ~3e-4
    // relative on the high-magnitude time direction — a pure truncation artifact
    // that no achievable `h` removes. So the FD can only catch an O(1) formula
    // blunder; it cannot pin the assembly.
    //
    // This block pins the hook's FULL assembly (`primary_trace_weight`
    // W-projection + full-`t4` contraction + `add_pullback_hessian`) to machine
    // precision with NO finite-difference truncation, using the identity
    //     uᵀ · (∇²_β tr(W·H)) · u  ==  tr(W · ∂²H/∂β_u²).
    // The right-hand side is built by
    // `exact_newton_joint_hessiansecond_directional_derivative` — the rank-1
    // `fourth_contracted` second-directional path that the outer-REML Jeffreys
    // drift already relies on. It shares neither `primary_trace_weight` nor the
    // direct full-`t4` read with the hook: it assembles the whole `∂²H` matrix
    // first and contracts `W` in COEFFICIENT space, whereas the hook projects
    // `W` into PRIMARY space per row before contracting `t4`. Their agreement
    // therefore validates that the primary-space trace projection and the
    // Jacobian pullback commute with the coefficient-space trace. Both sides are
    // exact (no `h`), so the tolerance is tight (1e-9), and this — not the FD —
    // is what guarantees the #979 completion is correct.
    for (idx, dir) in directions.iter().enumerate() {
        let huu = family
            .exact_newton_joint_hessiansecond_directional_derivative(&states0, dir, dir)
            .expect("second directional derivative call")
            .expect("rigid path must supply the second directional derivative");
        let trace_w_huu = (&w * &huu).sum();
        let analytic_quad = dir.dot(&analytic.dot(dir));
        let rel = (trace_w_huu - analytic_quad).abs() / trace_w_huu.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "direction {idx}: contracted trace-Hessian assembly vs independent \
             tr(W·∂²H/∂β_u²) mismatch: analytic={analytic_quad:.12e} \
             independent={trace_w_huu:.12e} rel={rel:.3e}"
        );
    }

    // `tr(W·H(β))` is a second derivative of the row NLL, so its central
    // second difference probes the FOURTH β-derivative and carries an
    // `(h²/12)·(sixth-derivative-of-NLL)` truncation term. For the probit /
    // log-φ composite that sixth derivative summed over rows is large enough
    // that a single-`h` difference at `h=1e-3` sits right at the 1e-5 relative
    // tolerance on the `g`-spanning directions. Richardson-extrapolate two
    // central differences (`h`, `h/2`) to cancel the O(h²) term and validate
    // the analytic to O(h⁴). This is a strictly stronger check than the raw
    // difference: it cannot mask a genuine O(1) analytic error (both `D(h)`
    // and `D(h/2)` would then be wrong by ≈the same O(1) amount and the
    // combination stays wrong) — it only removes the discretization artifact.
    let eps = 1e-3;
    let center = trace_of_hessian_at(&beta0);
    for (idx, dir) in directions.iter().enumerate() {
        let central_second = |h: f64| -> f64 {
            let step: Array1<f64> = dir.mapv(|v| v * h);
            let plus = trace_of_hessian_at(&(&beta0 + &step));
            let minus = trace_of_hessian_at(&(&beta0 - &step));
            (plus - 2.0 * center + minus) / (h * h)
        };
        let d_h = central_second(eps);
        let d_h2 = central_second(eps * 0.5);
        let fd_second = (4.0 * d_h2 - d_h) / 3.0;
        let analytic_quad = dir.dot(&analytic.dot(dir));
        let rel = (fd_second - analytic_quad).abs() / fd_second.abs().max(1.0);
        // Discretization-limited cross-check, NOT the authoritative correctness
        // gate. The trace `tr(W·H)` is already second-order in β, so this second
        // difference probes the FOURTH derivative; the residual after Richardson
        // is `O(h⁴·f⁽⁸⁾)`, and survival's neglog-Φ Mills-ratio tails carry an
        // `f⁽⁸⁾` large enough (≳1e11 on the fixture's high-magnitude directions,
        // e.g. direction 0 at analytic≈1.06e3) that even the extrapolated FD
        // floors near ~3e-4 relative — a pure truncation artifact, not an
        // analytic error. The EXACT correctness of the contracted-trace tower is
        // pinned to machine precision by `survival_sparse_tower4_full_t4_matches_
        // dense_oracle_979` (dense-oracle agreement to 1e-9); this FD gate only
        // guards against an O(1) formula blunder, for which 1e-3 is ample.
        assert!(
            rel < 1e-3,
            "direction {idx}: survival contracted trace-Hessian FD mismatch: \
             analytic={analytic_quad:.10e} fd={fd_second:.10e} rel={rel:.3e}"
        );
    }
}

/// gam#979 perf datapoint + large-`p` correctness cross-check: the O(n·p²)
/// contracted-trace hook vs the `p(p+1)/2` pairwise second-directional
/// completion it replaces, at the issue's repro scale (`matern(...,
/// centers≈20)` ⇒ `p_marginal = p_slope = 20`, total `p = 41`).
///
/// This is the mechanism behind the #979 rescope: the exact Firth/Jeffreys
/// second-order completion is gated on
/// `joint_jeffreys_information_contracted_trace_hessian_available()`. Before the
/// hook that gate was `false` and the completion NEVER ran (the inner Newton
/// under-modelled curvature near a Firth-active mode ⇒ the near-separation
/// crawl / 2400 s large_scale timeouts). The completion could not simply be
/// switched on with the generic pairwise fallback because that costs
/// `p(p+1)/2` full-data row-streamed second-directional Hessian passes
/// (`O(n·p⁴)`) — "hundreds of passes" at production p, far too slow to run every
/// endgame cycle. The hook produces the identical completion in ONE `O(n·p²)`
/// family pass, cheap enough to run every cycle.
///
/// The wall-clock speedup is REPORTED (the `[979 perf]` line) but not asserted:
/// shared-node CI timing is non-deterministic, so the gate is the deterministic
/// CORRECTNESS cross-check — the cheap `O(n·p²)` hook must reproduce the
/// expensive `p(p+1)/2` pairwise assembly over the FULL `p×p` matrix
/// (truncation-free, via `uᵀ∇²tr(W·H)v = tr(W·∂²H/∂β_u∂β_v)`), a strictly
/// larger correctness surface than the 7-direction gate above. `n` is kept
/// modest so the `p(p+1)/2`-pass reference stays CI-fast (the `#[ignore]`
/// wall-clock-benchmark form is banned by the workspace hygiene scanner).
#[test]
fn survival_jeffreys_contracted_trace_hook_beats_pairwise_979() {
    use gam_math::paired_timing::{SpeedGate, paired_interleaved};

    let n = 800usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.29).sin() * 0.9).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.6 + 0.4 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();

    let p_m = 20usize;
    let p_g = 20usize;
    let marginal_design = Array2::from_shape_fn((n, p_m), |(r, j)| {
        0.2 + 0.05 * ((r + j) as f64).cos() + 0.011 * (j as f64) - 0.003 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, p_g), |(r, j)| {
        0.1 + 0.07 * ((r + 2 * j) as f64).sin() - 0.009 * (j as f64)
            + 0.004 * (r as f64) / (n as f64)
    });

    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));

    let total = 1 + p_m + p_g;
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(p_m),
        dummy_blockspec(p_g),
    ];

    let beta0 = Array1::from_shape_fn(total, |i| 0.1 + 0.03 * ((i as f64) * 1.7).sin());
    let beta_time = beta0.slice(ndarray::s![0..1]).to_owned();
    let beta_marginal = beta0.slice(ndarray::s![1..1 + p_m]).to_owned();
    let beta_slope = beta0.slice(ndarray::s![1 + p_m..total]).to_owned();
    let states0 = vec![
        ParameterBlockState {
            beta: beta_time,
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal_design.dot(&beta_marginal),
        },
        ParameterBlockState {
            beta: beta_slope.clone(),
            eta: slope_design.dot(&beta_slope),
        },
    ];

    let mut w_raw = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in 0..total {
            w_raw[[i, j]] = ((i * 7 + j * 11 + 2) % 13) as f64 * 0.1 - 0.6;
        }
    }
    let w = (&w_raw + &w_raw.t()).mapv(|v| v * 0.5);

    // ── Hook: ONE O(n·p²) family pass ────────────────────────────────────
    let hook = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(&states0, &specs, &w)
        .expect("contracted trace hessian call")
        .expect("rigid path must supply the contracted completion");
    assert_eq!(hook.dim(), (total, total));

    // ── Pairwise: p(p+1)/2 full second-directional passes (what the hook
    // replaces), reconstructing the SAME ∇²_β tr(W·H) via the truncation-free
    // identity uᵀ∇²tr(W·H)v = tr(W·∂²H/∂β_u∂β_v). ─────────────────────────
    let unit = |a: usize| -> Array1<f64> {
        let mut e = Array1::<f64>::zeros(total);
        e[a] = 1.0;
        e
    };
    let pairwise_assembly = |w: &Array2<f64>| -> Array2<f64> {
        let mut pairwise = Array2::<f64>::zeros((total, total));
        for a in 0..total {
            let ea = unit(a);
            for b in a..total {
                let eb = unit(b);
                let huv = family
                    .exact_newton_joint_hessiansecond_directional_derivative(&states0, &ea, &eb)
                    .expect("second directional derivative call")
                    .expect("rigid path must supply the second directional derivative");
                let val = (w * &huv).sum();
                pairwise[[a, b]] = val;
                pairwise[[b, a]] = val;
            }
        }
        pairwise
    };
    let pairwise = pairwise_assembly(&w);

    // Correctness: the cheap hook equals the expensive pairwise assembly over
    // the full p×p matrix, truncation-free.
    let mut max_rel = 0.0_f64;
    for a in 0..total {
        for b in 0..total {
            let rel = (hook[[a, b]] - pairwise[[a, b]]).abs() / pairwise[[a, b]].abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
    }
    let n_pairwise_passes = total * (total + 1) / 2;
    eprintln!(
        "[979] n={n} p={total} (p_m={p_m} p_g={p_g}) hook (1 pass) vs pairwise \
         ({n_pairwise_passes} passes): max_rel={max_rel:.3e}"
    );
    assert!(
        max_rel < 1e-9,
        "hook completion disagrees with pairwise assembly at large p: max_rel={max_rel:.3e}"
    );

    // The hook must be faster than the pairwise fallback it replaces (the
    // theoretical ratio is ~p(p+1)/2). Release profile only, paired and
    // interleaved: the previous form timed each side once, sequentially, and
    // an unpaired sequential wall-clock ratio cannot tell slow code from a
    // busy node. The nudge perturbs the trace weight so neither assembly is
    // loop-invariant across repetitions.
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("JEFFREYS-TRACE-HOOK-979");
    let timing = paired_interleaved(
        7,
        1,
        0x979_0_5EED,
        |nudge| {
            let mut w = w.clone();
            w[[0, 0]] += nudge;
            family
                .joint_jeffreys_information_contracted_trace_hessian_with_specs(
                    &states0, &specs, &w,
                )
                .expect("contracted trace hessian call")
                .expect("rigid path must supply the contracted completion")
                .sum()
        },
        |nudge| {
            let mut w = w.clone();
            w[[0, 0]] += nudge;
            pairwise_assembly(&w).sum()
        },
    );
    gate.faster(
        &format!("n={n} p={total} passes={n_pairwise_passes}"),
        &timing,
        "hook",
        "pairwise",
    );
    gate.finish();
}

/// gam#979 isolation gate: does `SurvivalMarginalSlopeRowKernel`'s
/// static-sparsity `SparseTower4<STATIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>` build the SAME full
/// `t4` (all 256 entries, not just the 3 `fourth_contracted(u,v)` direction
/// pairs `rigid_row_kernel_agrees_with_jet_tower_program_all_channels`
/// checks) as the dense `Tower4<4>` oracle, on the EXACT fixture/row the
/// contracted-trace-Hessian FD gate above exercises?
///
/// My `contracted_trace_hessian` hook is the FIRST production consumer to
/// read `t4[a][b][c][d]` directly for every `(a,b)` pair (a genuine
/// non-rank-1 bilinear contraction against a full 4×4 weight matrix,
/// `Σ_ab w_row[a,b]·t4[a][b][c][d]`) rather than through a single
/// `fourth_contracted(u, v)` rank-1 direction pair — every prior consumer
/// only ever exercised the latter. This test empirically checks whether that
/// previously-untested full-tensor read path agrees with the dense oracle,
/// independent of whether the contracted-trace-Hessian FD gate above passes
/// or fails.
#[test]
fn survival_sparse_tower4_full_t4_matches_dense_oracle_979() {
    use super::row_kernel::{
        SparseTower4, rigid_row_inputs, rigid_row_kernel_primaries, rigid_row_nll,
    };
    use super::slope_geometry::{RIGID_LINEAR_MASK, STATIC_SLOPE_PRIMARIES};
    use gam_math::jet_scalar::JetScalar;
    use gam_math::jet_tower::program_full_tower;

    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|r| ((r as f64) * 0.29).sin() * 0.9).collect();
    let weights: Vec<f64> = (0..n).map(|r| 0.6 + 0.4 * ((r % 5) as f64) / 5.0).collect();
    let event: Vec<f64> = (0..n).map(|r| ((r % 3 == 0) as u8) as f64).collect();

    let p_m = 2usize;
    let p_g = 2usize;
    let marginal_design = Array2::from_shape_fn((n, p_m), |(r, j)| {
        0.2 + 0.05 * (r as f64).cos() + 0.11 * (j as f64) - 0.013 * (r as f64) / (n as f64)
    });
    let slope_design = Array2::from_shape_fn((n, p_g), |(r, j)| {
        0.1 + 0.07 * (r as f64).sin() - 0.09 * (j as f64) + 0.004 * (r as f64) / (n as f64)
    });

    let mut family = oracle_rigid_family(n, &z, &weights, &event, None);
    family.marginal_design = DesignMatrix::from(marginal_design.clone());
    family
        .slope_layout
        .replace_coefficient_design(DesignMatrix::from(slope_design.clone()));

    let beta_time = array![0.6];
    let beta_marginal = array![0.18, -0.12];
    let beta_slope = array![-0.2, 0.13];
    let marginal_eta = marginal_design.dot(&beta_marginal);
    let slope_eta = slope_design.dot(&beta_slope);
    let block_states = vec![
        ParameterBlockState {
            beta: beta_time.clone(),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal_eta.clone(),
        },
        ParameterBlockState {
            beta: beta_slope.clone(),
            eta: slope_eta.clone(),
        },
    ];
    let probit_scale = family.probit_frailty_scale();

    let mut primaries = Vec::with_capacity(n);
    for row in 0..n {
        let q0 = family.design_entry.dot_row(row, &beta_time)
            + family.offset_entry[row]
            + marginal_eta[row];
        let q1 = family.design_exit.dot_row(row, &beta_time)
            + family.offset_exit[row]
            + marginal_eta[row];
        let qd1 = family.design_derivative_exit.dot_row(row, &beta_time)
            + family.derivative_offset_exit[row];
        let g = slope_eta[row];
        primaries.push([q0, q1, qd1, g]);
    }
    let program = SurvivalMarginalSlopeRigidNllProgram {
        primaries,
        z: z.clone(),
        w: weights.clone(),
        d: event.clone(),
        probit_scale,
    };

    let mut max_abs_gap = 0.0_f64;
    let mut max_rel_gap = 0.0_f64;
    for row in 0..n {
        let dense = program_full_tower(&program, row).expect("dense tower4 oracle");

        let p = rigid_row_kernel_primaries::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
            &family,
            &block_states,
            row,
        )
        .expect("primaries");
        let inputs = rigid_row_inputs(
            &family,
            &block_states,
            row,
            "sparse-vs-dense t4 isolation test",
        )
        .expect("rigid row inputs");
        let vars: [SparseTower4<STATIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>;
            STATIC_SLOPE_PRIMARIES] = std::array::from_fn(|a| SparseTower4::variable(p[a], a));
        let sparse =
            rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&vars, &inputs)
                .expect("sparse tower4");

        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let dv = dense.t4[a][b][c][d];
                        let sv = sparse.t4[a][b][c][d];
                        let abs_gap = (dv - sv).abs();
                        let rel_gap = abs_gap / dv.abs().max(1.0);
                        if abs_gap > max_abs_gap {
                            max_abs_gap = abs_gap;
                        }
                        if rel_gap > max_rel_gap {
                            max_rel_gap = rel_gap;
                        }
                        assert!(
                            rel_gap < 1e-9,
                            "row {row} t4[{a}][{b}][{c}][{d}]: sparse={sv:.12e} dense={dv:.12e} \
                             abs_gap={abs_gap:e} rel_gap={rel_gap:e}"
                        );
                    }
                }
            }
        }
    }
    println!(
        "[979 isolation] full t4 max_abs_gap={max_abs_gap:e} max_rel_gap={max_rel_gap:e} over {n} rows"
    );
}

// ── #2352 ports: slope block-jacobian production contract ────────────────
//
// Moved from `tests/survival/misc/frailty_scale_audit_plumbing.rs` and
// `tests/survival/survival/survival_marginal_slope_jacobian_hyperbolic_correction.rs`
// when `SlopeBlockJacobian` construction went crate-internal (layout +
// covariance record; probit scale read from the linearization state). The
// old root-tree contract "Err when family_scalars is None at nonzero β" was
// deliberately replaced by origin-linearization (q ≡ 0) for the pre-fit
// structural audit, so the ports assert the CURRENT contract.

fn slope_port_design(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, p));
    let mut state = seed;
    for i in 0..n {
        for j in 0..p {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out[[i, j]] = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
    }
    out
}

fn slope_port_z(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0xdeadbeef;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.3 + ((state >> 33) as f64) / (u32::MAX as f64) * 1.4
        })
        .collect()
}

fn slope_port_jacobian(
    design: &Array2<f64>,
    z: &[f64],
) -> super::block_jacobians::SlopeBlockJacobian {
    let n = design.nrows();
    let layout = SlopeLayout::shared(DesignMatrix::from(design.clone()), Array1::zeros(n));
    let z_mat = Array2::from_shape_fn((n, 1), |(i, _)| z[i]);
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0]).expect("unit covariance");
    super::block_jacobians::SlopeBlockJacobian::new(layout, Arc::new(z_mat), covariance.into())
        .expect("slope jacobian construction")
}

/// At β = 0 the slope Jacobian's η0/η1 channels are `s·z_i·G[i,j]` with `s`
/// read from `state.probit_frailty_scale` (NOT baked in at construction), and
/// the ad1 channel is zero; two states with different `s` scale exactly.
#[test]
fn slope_jacobian_reads_probit_scale_from_state_at_beta_zero() {
    use crate::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
    let n = 40;
    let p = 5;
    let design = slope_port_design(n, p, 42);
    let z = slope_port_z(n, 42);
    let s_f = 0.75_f64;
    let cb = slope_port_jacobian(&design, &z);

    let beta_zero = vec![0.0f64; p];
    let jac_at = |s: f64| {
        let state = FamilyLinearizationState {
            beta: &beta_zero,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: s,
        };
        cb.effective_jacobian_at(&state)
            .expect("beta=0 slope jacobian")
    };
    let jac_sf = jac_at(s_f);
    let jac_1 = jac_at(1.0);

    assert_eq!(jac_sf.nrows(), 3 * n, "jacobian must have 3*n rows");
    assert_eq!(jac_sf.ncols(), p, "jacobian must have p cols");
    for i in 0..n {
        for j in 0..p {
            let expected = s_f * z[i] * design[[i, j]];
            let got = jac_sf[[i, j]];
            let err = (got - expected).abs();
            assert!(
                err / expected.abs().max(1e-14) < 1e-10 || err < 1e-12,
                "eta0[{i},{j}]: got {got:.6e} expected {expected:.6e}"
            );
            let ad1 = jac_sf[[2 * n + i, j]];
            assert!(
                ad1.abs() < 1e-12,
                "ad1[{i},{j}] must be 0 at beta=0: {ad1:.3e}"
            );
            if got.abs() > 1e-14 {
                let ratio = jac_1[[i, j]] / got;
                assert!(
                    (ratio - 1.0 / s_f).abs() < 1e-10,
                    "state probit scale must set the jacobian scale: ratio {ratio:.12} != {:.12}",
                    1.0 / s_f
                );
            }
        }
    }
}

/// With populated `SurvivalMarginalSlopeFamilyScalars` at moderate β, the
/// production slope Jacobian carries the full hyperbolic correction
/// `(q·dc/dg + s·z)·G` — FD-verified against the frozen-q η stack. Without
/// scalars the same point linearizes q at the origin (pre-fit audit
/// convention) and returns the pure `s·z_i·G` rows.
#[test]
fn slope_jacobian_hyperbolic_correction_matches_fd_with_scalars() {
    use crate::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
    use std::any::Any;
    let n = 40;
    let p = 5;
    let design = slope_port_design(n, p, 31415);
    let z = slope_port_z(n, 31415);
    let s_f = 0.8_f64;
    let cb = slope_port_jacobian(&design, &z);
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0]).expect("unit covariance");

    // Moderate deterministic beta so g_i != 0 on every row scale.
    let beta: Vec<f64> = (0..p)
        .map(|j| 0.35 * ((j as f64 + 1.0) * 0.7).sin())
        .collect();
    let g: Vec<f64> = (0..n)
        .map(|i| {
            design
                .row(i)
                .iter()
                .zip(beta.iter())
                .map(|(&x, &b)| x * b)
                .sum()
        })
        .collect();
    assert!(
        g.iter().any(|&v| v.abs() > 0.05),
        "fixture must reach nonzero g"
    );

    // Frozen primary scalars (pilot q values held fixed for the FD functional).
    let q0: Vec<f64> = (0..n).map(|i| -0.5 + 0.3 * z[i]).collect();
    let q1: Vec<f64> = (0..n).map(|i| 0.2 + 0.4 * z[i]).collect();
    let qd1: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * (z[i] * z[i]).min(2.0)).collect();
    let slopes = Array2::from_shape_fn((n, 1), |(i, _)| g[i]);
    let scalars: Arc<dyn Any + Send + Sync> = Arc::new(
        SurvivalMarginalSlopeFamilyScalars::new(
            q0.clone(),
            q1.clone(),
            qd1.clone(),
            slopes,
            None,
            s_f,
            &covariance.clone().into(),
        )
        .expect("family scalars"),
    );
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("slope jacobian with scalars");
    assert_eq!(jac.nrows(), 3 * n);
    assert_eq!(jac.ncols(), p);

    // Central-difference reference of the frozen-q eta stack
    //   eta0 = q0·c(β) + s·g(β)·z, eta1 likewise, ad1 = qd1·c(β),
    //   c(β) = sqrt(1 + s²·g(β)²).
    let eta_stack = |b: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; 3 * n];
        for i in 0..n {
            let gi: f64 = design
                .row(i)
                .iter()
                .zip(b.iter())
                .map(|(&x, &bb)| x * bb)
                .sum();
            let c = (1.0 + s_f * s_f * gi * gi).sqrt();
            out[i] = q0[i] * c + s_f * gi * z[i];
            out[n + i] = q1[i] * c + s_f * gi * z[i];
            out[2 * n + i] = qd1[i] * c;
        }
        out
    };
    let h = 1e-6;
    let mut max_rel = 0.0_f64;
    for col in 0..p {
        let mut bp = beta.clone();
        let mut bm = beta.clone();
        bp[col] += h;
        bm[col] -= h;
        let ep = eta_stack(&bp);
        let em = eta_stack(&bm);
        for row in 0..3 * n {
            let fd = (ep[row] - em[row]) / (2.0 * h);
            let an = jac[[row, col]];
            // Scale by the derivative's own magnitude but floor the scale at 1,
            // not at 1e-8. A `1e-8` floor demands `|an - fd| < 1e-13` on any
            // near-zero entry of this Jacobian, which is two to three orders
            // BELOW the central difference's own noise: with `h = 1e-6` and eta
            // of order one, the roundoff term `eps*|eta|/h` is already ~1e-10
            // absolute, so those entries were being judged against a bound the
            // reference itself cannot meet, and the reported `max_rel` was
            // measuring FD noise divided by a fabricated denominator. Flooring
            // at 1 keeps the relative test verbatim wherever `|fd| >= 1` and
            // turns the small entries into a 1e-5 ABSOLUTE check — still five
            // orders above the noise floor, so a genuinely missing hyperbolic
            // term of any consequence still fails here.
            let denom = fd.abs().max(1.0);
            max_rel = max_rel.max((an - fd).abs() / denom);
        }
    }
    assert!(
        max_rel < 1e-5,
        "slope jacobian must carry the hyperbolic correction: max rel err vs FD = {max_rel:.3e}"
    );

    // Origin-linearized fallback: scalars=None at the SAME nonzero beta gives
    // the pure s·z·G rows (q ≡ 0) with a zero ad1 channel.
    let state_none = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac_none = cb
        .effective_jacobian_at(&state_none)
        .expect("origin-linearized slope jacobian");
    for i in 0..n {
        for j in 0..p {
            let expected = s_f * z[i] * design[[i, j]];
            let got = jac_none[[i, j]];
            assert!(
                (got - expected).abs() < 1e-10 * (1.0 + expected.abs()),
                "origin-linearized eta0[{i},{j}]: got {got:.6e} expected {expected:.6e}"
            );
            assert!(
                jac_none[[2 * n + i, j]].abs() < 1e-12,
                "origin-linearized ad1 channel must be zero"
            );
        }
    }
}

/// Clamped degree-3 knot vector with no internal knots: `len = 8` gives
/// `len - bs_degree - 1 - 1 = 3` monotone I-spline columns, the minimal valid
/// degree-3 wiggle.
fn timewiggle_test_knots() -> Array1<f64> {
    Array1::from_vec(vec![-3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, 3.0])
}

const TIMEWIGGLE_TEST_DEGREE: usize = 3;
const TIMEWIGGLE_TEST_NCOLS: usize = 3;

/// An n-row timewiggle-active family: `p_base` linear time columns followed by
/// `TIMEWIGGLE_TEST_NCOLS` wiggle amplitudes, `p_m` marginal columns, and a
/// `k`-dimensional slope carrying `covariance`.
///
/// The wiggle tail columns of the three time designs are zero. A wiggle
/// amplitude enters `q` through the B-spline composition `q = h + B(h)·β_w`,
/// not through a linear design column, which is why `time_wiggle_range` is a
/// trailing slice of the time block rather than a separate block.
fn make_timewiggle_test_family(
    n: usize,
    p_base: usize,
    p_m: usize,
    slope_layout: SlopeLayout,
    z: Array2<f64>,
    covariance: MarginalSlopeCovariance,
) -> SurvivalMarginalSlopeFamily {
    let p_time = p_base + TIMEWIGGLE_TEST_NCOLS;
    let padded = |base: f64, scale: f64| {
        let mut design = Array2::<f64>::zeros((n, p_time));
        for row in 0..n {
            for col in 0..p_base {
                design[[row, col]] = base + scale * (((row * 7 + col * 3) as f64) * 0.6).sin();
            }
        }
        DesignMatrix::from(design)
    };

    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let offset_entry: Array1<f64> = Array1::from_iter((0..n).map(|i| -0.30 + 0.11 * i as f64));
    let offset_exit: Array1<f64> = Array1::from_iter((0..n).map(|i| 0.15 + 0.09 * i as f64));
    // qd1 must stay well above the derivative guard at every perturbed beta.
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.8 + 0.05 * ((i * 23 + 1) % 3) as f64));
    let marginal_design =
        Array2::from_shape_fn((n, p_m), |(row, col)| 0.2 + 0.1 * (row + col) as f64 * 0.25);

    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z),
        score_covariance: covariance.into(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: padded(0.30, 0.10),
        design_exit: padded(0.55, 0.15),
        design_derivative_exit: padded(0.45, 0.05),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal_design),
        slope_layout,
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(timewiggle_test_knots()),
        time_wiggle_degree: Some(TIMEWIGGLE_TEST_DEGREE),
        time_wiggle_ncols: TIMEWIGGLE_TEST_NCOLS,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

/// The three blocks `current_identifiability_family_scalars` expects, in the
/// order it indexes them: time, marginal, slope.
///
/// The marginal block's `eta` is NOT a placeholder and must equal
/// `marginal_design · beta_marginal`. `row_dynamic_q_values` reads the marginal
/// contribution to `h0`/`h1` from `block_states[1].eta[row]`, whereas
/// `row_dynamic_q_gradient` differentiates through the marginal DESIGN. The two
/// agree only for states whose `eta` is in sync with `beta` — which production
/// states always are, and which a hand-built fixture must reproduce or the two
/// routines silently disagree. A zeros `eta` here makes `q` independent of
/// `beta_marginal`, so a finite difference in that direction is exactly `0`
/// while the published gradient row is not: a fixture artifact that looks like
/// a derivative bug.
fn timewiggle_states(
    family: &SurvivalMarginalSlopeFamily,
    beta_time: &Array1<f64>,
    beta_marginal: &Array1<f64>,
    beta_slope: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let eta_marginal =
        Array1::from_shape_fn(n, |row| family.marginal_design.dot_row(row, beta_marginal));
    vec![
        ParameterBlockState {
            beta: beta_time.clone(),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: eta_marginal,
        },
        ParameterBlockState {
            beta: beta_slope.clone(),
            eta: Array1::zeros(n),
        },
    ]
}

fn timewiggle_scalars(
    family: &SurvivalMarginalSlopeFamily,
    states: &[ParameterBlockState],
) -> Arc<dyn std::any::Any + Send + Sync> {
    <SurvivalMarginalSlopeFamily as CustomFamily>::current_identifiability_family_scalars(
        family, states,
    )
    .expect("current scalars")
    .expect("survival must expose current scalars")
}

/// gam#2473. At nonzero β the timewiggle block Jacobians no longer derive
/// `∂q/∂β` themselves: `981c3174d` moved that derivation into the family's
/// canonical `row_dynamic_q_gradient` and left the blocks scaling the rows the
/// family publishes on `family_scalars`. This gate pins the derivation at its
/// new home — the rows `current_identifiability_family_scalars` hands out must
/// equal central differences of the family's own forward `q` at the same
/// nonzero β, in both the time and marginal directions and on all three
/// channels (`q0`, `q1`, `qd1`).
///
/// It lives in-crate because the derivation does. The two integration gates in
/// `tests/basis_smooth/optimization/effective_jacobian_at_timewiggle.rs`
/// construct the callbacks without a family at all, so the only
/// `timewiggle_primary_rows` reachable from outside this crate are ones the
/// test derives itself — and checking those against the test's own `q` says
/// nothing about the derivation production runs.
#[test]
fn timewiggle_primary_rows_match_finite_differences_of_q_2473() {
    let n = 4usize;
    let p_base = 2usize;
    let p_time = p_base + TIMEWIGGLE_TEST_NCOLS;
    let p_m = 2usize;

    let slope_design = Array2::from_shape_fn((n, 1), |(row, _)| 0.4 + 0.2 * row as f64);
    let slope_offset: Array1<f64> = Array1::from_iter((0..n).map(|i| 0.10 + 0.03 * i as f64));
    let layout = SlopeTopology::shared()
        .materialize_identity(DesignMatrix::from(slope_design), &slope_offset)
        .unwrap();
    let z = Array2::from_shape_fn((n, 1), |(row, _)| -0.5 + 0.3 * row as f64);
    let family = make_timewiggle_test_family(
        n,
        p_base,
        p_m,
        layout,
        z,
        MarginalSlopeCovariance::diagonal(array![1.0]).unwrap(),
    );
    assert!(
        family.flex_timewiggle_active(),
        "the fixture must reach the timewiggle branch or this gate is vacuous",
    );

    let beta_time = Array1::from_iter((0..p_time).map(|j| 0.18 - 0.05 * j as f64));
    let beta_marginal = Array1::from_iter((0..p_m).map(|j| 0.09 + 0.04 * j as f64));
    let beta_slope = array![0.3];
    assert!(
        beta_time.iter().any(|value| *value != 0.0),
        "β must be nonzero: the refusal this gate covers fires only off β = 0",
    );

    let states = timewiggle_states(&family, &beta_time, &beta_marginal, &beta_slope);
    let erased = timewiggle_scalars(&family, &states);
    let scalars = erased
        .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
        .expect("survival scalar type");
    let (time_rows, marginal_rows) = scalars.timewiggle_primary_rows.as_ref().expect(
        "a timewiggle-active family must publish canonical q-gradient rows; without them \
         every nonzero-β block Jacobian refuses and the fit loses its analytic Jacobian",
    );
    assert_eq!(time_rows.dim(), (3 * n, p_time));
    assert_eq!(marginal_rows.dim(), (3 * n, p_m));

    let eps = 1e-6;
    let tol = 1e-6;
    for j in 0..p_time {
        let mut plus = beta_time.clone();
        let mut minus = beta_time.clone();
        plus[j] += eps;
        minus[j] -= eps;
        let states_plus = timewiggle_states(&family, &plus, &beta_marginal, &beta_slope);
        let states_minus = timewiggle_states(&family, &minus, &beta_marginal, &beta_slope);
        for row in 0..n {
            let hi = family
                .row_dynamic_q_values(row, &states_plus)
                .expect("forward q at +eps");
            let lo = family
                .row_dynamic_q_values(row, &states_minus)
                .expect("forward q at -eps");
            assert_close(
                (hi.q0 - lo.q0) / (2.0 * eps),
                time_rows[[row, j]],
                tol,
                &format!("dq0/dbeta_time[{j}] row {row}"),
            );
            assert_close(
                (hi.q1 - lo.q1) / (2.0 * eps),
                time_rows[[n + row, j]],
                tol,
                &format!("dq1/dbeta_time[{j}] row {row}"),
            );
            assert_close(
                (hi.qd1 - lo.qd1) / (2.0 * eps),
                time_rows[[2 * n + row, j]],
                tol,
                &format!("dqd1/dbeta_time[{j}] row {row}"),
            );
        }
    }

    for j in 0..p_m {
        let mut plus = beta_marginal.clone();
        let mut minus = beta_marginal.clone();
        plus[j] += eps;
        minus[j] -= eps;
        let states_plus = timewiggle_states(&family, &beta_time, &plus, &beta_slope);
        let states_minus = timewiggle_states(&family, &beta_time, &minus, &beta_slope);
        for row in 0..n {
            let hi = family
                .row_dynamic_q_values(row, &states_plus)
                .expect("forward q at +eps");
            let lo = family
                .row_dynamic_q_values(row, &states_minus)
                .expect("forward q at -eps");
            assert_close(
                (hi.q0 - lo.q0) / (2.0 * eps),
                marginal_rows[[row, j]],
                tol,
                &format!("dq0/dbeta_marginal[{j}] row {row}"),
            );
            assert_close(
                (hi.q1 - lo.q1) / (2.0 * eps),
                marginal_rows[[n + row, j]],
                tol,
                &format!("dq1/dbeta_marginal[{j}] row {row}"),
            );
            assert_close(
                (hi.qd1 - lo.qd1) / (2.0 * eps),
                marginal_rows[[2 * n + row, j]],
                tol,
                &format!("dqd1/dbeta_marginal[{j}] row {row}"),
            );
        }
    }
}

/// gam#2473, the `K > 1` arm the two integration gates cannot reach.
///
/// At `K = 1` the pre-`981c3174d` `c_i = sqrt(1 + (s·g)²)` and the current
/// `c_i = sqrt(1 + s²·gᵀΣg)` coincide for `Σ = [1]`, so a `P_G = 1` gate stays
/// green over a restored scalar derivation that mis-scales every `K > 1` fit.
/// This pins the vector form against a `Σ ≠ I` the scalar form cannot produce.
#[test]
fn current_scalars_use_the_vector_covariance_scale_at_k_two_2473() {
    let n = 3usize;
    let p_base = 2usize;
    let p_time = p_base + TIMEWIGGLE_TEST_NCOLS;
    let p_m = 2usize;

    let slope_design =
        Array2::from_shape_fn((n, 2), |(row, col)| 1.0 + 0.5 * row as f64 + col as f64);
    let z = Array2::from_shape_fn((n, 2), |(row, col)| {
        -0.4 + 0.2 * row as f64 + 0.1 * col as f64
    });
    let covariance = MarginalSlopeCovariance::diagonal(array![2.0, 0.5]).unwrap();
    let slope_offset: Array1<f64> = Array1::from_iter((0..n).map(|i| 0.10 + 0.03 * i as f64));
    let layout = SlopeTopology::per_score(vec![0..1, 1..2], 2)
        .unwrap()
        .materialize_identity(DesignMatrix::from(slope_design.clone()), &slope_offset)
        .unwrap();
    let family = make_timewiggle_test_family(n, p_base, p_m, layout, z, covariance);

    let beta_time = Array1::from_iter((0..p_time).map(|j| 0.12 - 0.04 * j as f64));
    let beta_marginal = Array1::from_iter((0..p_m).map(|j| 0.07 + 0.03 * j as f64));
    let beta_slope = array![0.25, -0.15];
    let states = timewiggle_states(&family, &beta_time, &beta_marginal, &beta_slope);
    let erased = timewiggle_scalars(&family, &states);
    let scalars = erased
        .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
        .expect("survival scalar type");

    let s = family.probit_frailty_scale();
    for row in 0..n {
        let g0 = slope_design[[row, 0]] * beta_slope[0] + slope_offset[row];
        let g1 = slope_design[[row, 1]] * beta_slope[1] + slope_offset[row];
        let quadratic = 2.0 * g0 * g0 + 0.5 * g1 * g1;
        let expected = (1.0 + s * s * quadratic).sqrt();
        assert_close(
            scalars.c_i[row],
            expected,
            1e-12,
            &format!("c_i at K=2 row {row}"),
        );
        let scalar_form = (1.0 + (s * g0).powi(2)).sqrt();
        assert!(
            (scalars.c_i[row] - scalar_form).abs() > 1e-6,
            "the fixture must separate the vector and scalar c_i forms, otherwise this \
             gate cannot see the drift it exists to catch (row {row})",
        );
    }
}

/// `rigid_row_primary_mixed_in_z` must equal a central finite difference of the
/// canonical row program's PRIMARY gradient in the latent score (gam#2768).
///
/// This is the only new row-level mathematics the generated-regressor covariance
/// correction needs: the block scatter that turns it into `∂(score_β)/∂ζ` reuses
/// the production gradient accumulator unchanged, so it is correct by
/// construction, and this gate is what stands behind the chain rule that
/// produces the vector it scatters.
///
/// The grid deliberately spans both censored and event rows, both signs of the
/// slope (`c(g)` is even in `g` while the `s(g)·z` channel is odd, so a sign
/// error in the `L`-channel term survives a positive-only grid), a non-unit
/// probit frailty scale, and a non-unit `covariance_ones` — the last of which is
/// what a hand derivation assuming `Var(z) = 1` would get wrong, because the
/// fitted score covariance is only approximately one.
#[test]
fn rigid_row_primary_mixed_in_z_matches_finite_difference() {
    let derivative_guard = 1e-8;
    let mut checked = 0usize;
    for &(q0, q1, qd1) in &[
        (-1.30, -0.40, 0.90),
        (-2.10, 0.65, 1.70),
        (0.25, 1.10, 0.35),
    ] {
        for &g in &[-0.85_f64, -0.20, 0.0, 0.30, 1.40] {
            for &z in &[-1.75_f64, -0.30, 0.55, 2.20] {
                for &d in &[0.0_f64, 1.0] {
                    for &(w, probit_scale, covariance_ones) in &[
                        (1.0_f64, 1.0_f64, 1.0_f64),
                        (0.7, 0.82, 1.13),
                        (2.4, 1.35, 0.76),
                    ] {
                        let inputs_at = |z_value: f64| RigidRowInputs {
                            row: 0,
                            wi: w,
                            wi_entry: w,
                            di: d,
                            z_sum: z_value,
                            covariance_ones,
                            probit_scale,
                            qd1_lower: derivative_guard,
                            anchor: None,
                        };
                        let primaries = [q0, q1, qd1, g];
                        let analytic = rigid_row_primary_mixed_in_z::<
                            STATIC_SLOPE_PRIMARIES,
                            StaticSlopeGeometry,
                        >(&primaries, &inputs_at(z))
                        .expect("analytic mixed derivative");
                        // Central difference of the primary gradient in z. The
                        // step is scaled to |z| so the relative truncation error
                        // stays ~h² across the grid.
                        let h = 1e-5 * (1.0 + z.abs());
                        let (_, grad_plus, _) =
                            rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                                &primaries,
                                &inputs_at(z + h),
                            )
                            .expect("gradient +h");
                        let (_, grad_minus, _) =
                            rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                                &primaries,
                                &inputs_at(z - h),
                            )
                            .expect("gradient -h");
                        for axis in 0..4 {
                            let fd = (grad_plus[axis] - grad_minus[axis]) / (2.0 * h);
                            let scale = analytic[axis].abs().max(fd.abs()).max(1.0);
                            assert_close(
                                analytic[axis],
                                fd,
                                2e-5 * scale,
                                &format!(
                                    "mixed d2(NLL)/d(primary {axis})dz at q=({q0},{q1},{qd1}) \
                                     g={g} z={z} d={d} w={w} s={probit_scale} V={covariance_ones}"
                                ),
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        checked, 360,
        "the grid must be exercised in full; a silently skipped cell is not a passing gate"
    );
}

/// A time block's exit design with `p_base` base columns and a zero placeholder
/// wiggle tail, the baseline predictor it carries, and the wiggle it declares.
fn placeholder_time_exit_with_wiggle(
    p_base: usize,
) -> (DesignMatrix, Array1<f64>, TimeWiggleBlockInput, Array2<f64>) {
    let n = 40;
    let offset_exit = Array1::from_shape_fn(n, |i| -1.5 + 3.0 * i as f64 / (n - 1) as f64);
    let degree = 3;
    let knots = gam_terms::basis::initializewiggle_knots_from_seed(offset_exit.view(), degree, 2)
        .expect("wiggle knots from the baseline predictor");
    let jacobian = crate::wiggle::monotone_wiggle_basis_from_knots(offset_exit.view(), &knots, degree)
        .expect("warp Jacobian at the baseline predictor");
    let ncols = jacobian.ncols();
    let mut placeholder = Array2::<f64>::zeros((n, p_base + ncols));
    for i in 0..n {
        for j in 0..p_base {
            placeholder[[i, j]] = offset_exit[i].powi(j as i32);
        }
    }
    (
        DesignMatrix::from(placeholder),
        offset_exit,
        TimeWiggleBlockInput { knots, degree, ncols },
        jacobian,
    )
}

fn embedded_penalty(p: usize, range: std::ops::Range<usize>, local: &Array2<f64>) -> Array2<f64> {
    let mut embedded = Array2::<f64>::zeros((p, p));
    embedded.slice_mut(s![range.clone(), range]).assign(local);
    embedded
}

/// gam#3061: a timewiggle-only time block (no base columns) is exactly the
/// production refusal when seeded against its placeholder design; seeded against
/// the acting design, its wiggle penalty reads the warp Jacobian's scale.
#[test]
fn timewiggle_penalties_are_seeded_against_the_warp_jacobian_3061() {
    let (placeholder, offset_exit, wiggle, jacobian) = placeholder_time_exit_with_wiggle(0);
    let wiggle_local = Array2::<f64>::eye(wiggle.ncols);
    let penalties = vec![embedded_penalty(wiggle.ncols, 0..wiggle.ncols, &wiggle_local)];

    let refused = block_log_lambda_seeds(&placeholder, penalties.iter())
        .expect_err("the control must refuse: the placeholder tail has an all-zero Gram");
    assert!(refused.contains("mean Gram diagonal is 0e0"), "{refused}");

    let acting = time_block_acting_exit_design(&placeholder, offset_exit.view(), Some(&wiggle))
        .expect("acting exit design");
    assert_eq!(acting.to_dense(), jacobian, "the wiggle tail must be the warp Jacobian");
    let seeds = time_block_log_lambda_seeds(&acting, &penalties, wiggle.ncols)
        .expect("a wiggle penalty seeds against the Jacobian");
    let expected = block_log_lambda_seeds(&DesignMatrix::from(jacobian.clone()), [&wiggle_local])
        .expect("seed against the Jacobian itself");
    assert_eq!(seeds, expected);

    // The ρ domain is read against the same design, and the placeholder's all-zero
    // Gram carried no resolvability information for it.
    let (lo, hi) =
        crate::fit_orchestration::drivers::penalized_block_rho_domain(&acting, penalties.iter());
    let (placeholder_lo, placeholder_hi) =
        crate::fit_orchestration::drivers::penalized_block_rho_domain(&placeholder, penalties.iter());
    assert!(lo[0].is_finite() && hi[0].is_finite() && lo[0] < hi[0], "[{}, {}]", lo[0], hi[0]);
    assert_ne!(
        (lo[0], hi[0]),
        (placeholder_lo[0], placeholder_hi[0]),
        "the acting design must resolve the wiggle penalty's own domain"
    );
}

/// gam#3061: with base columns beside the wiggle, each time penalty is seeded
/// against the part it acts on, and a penalty coupling the two parts is refused.
#[test]
fn time_penalties_are_seeded_against_the_part_they_act_on_3061() {
    let p_base = 2;
    let (placeholder, offset_exit, wiggle, jacobian) = placeholder_time_exit_with_wiggle(p_base);
    let p = p_base + wiggle.ncols;
    let base_local = array![[0.0, 0.0], [0.0, 1.0]];
    let wiggle_local = Array2::<f64>::eye(wiggle.ncols);
    let penalties = vec![
        embedded_penalty(p, 0..p_base, &base_local),
        embedded_penalty(p, p_base..p, &wiggle_local),
    ];
    let acting = time_block_acting_exit_design(&placeholder, offset_exit.view(), Some(&wiggle))
        .expect("acting exit design");
    let seeds = time_block_log_lambda_seeds(&acting, &penalties, wiggle.ncols)
        .expect("time seeds on the acting design");

    let base_design = DesignMatrix::from(placeholder.to_dense().slice(s![.., ..p_base]).to_owned());
    let base_seed = block_log_lambda_seeds(&base_design, [&base_local]).expect("base seed");
    let wiggle_seed = block_log_lambda_seeds(&DesignMatrix::from(jacobian), [&wiggle_local])
        .expect("wiggle seed");
    assert_relative_eq!(seeds[0], base_seed[0], max_relative = 1e-12);
    assert_relative_eq!(seeds[1], wiggle_seed[0], max_relative = 1e-12);

    let coupled = &penalties[0] + &penalties[1];
    let refused = time_block_log_lambda_seeds(&acting, &[coupled], wiggle.ncols)
        .expect_err("a penalty on both parts has no single scale");
    assert!(refused.contains("couples the base columns"), "{refused}");
}

/// #932 single-source pin, restored (#2818): the SPECIALIZED rigid-row
/// contractions must equal the INDEPENDENT dense `Tower4<4>`'s own contractions
/// on the exact inputs the release measurement is taken at, and — in release —
/// must be measurably faster than that dense tower.
///
/// ─── What was deleted, and what it cost ───
///
/// `c0a21b554` deleted this gate because it no longer compiled, and it no longer
/// compiled because `d484a091a` had deleted `Tower4::third_contracted` and
/// `Tower4::fourth_contracted` from `gam-math` — two `pub` Rust-library methods
/// whose only callers were tests, so no symbol for them appears in the CLI or
/// pyffi binary and the sweep's criterion held vacuously. `program_full_tower`,
/// `rigid_row_nll`, `RigidRowInputs` and `SurvivalMarginalSlopeRigidNllProgram`
/// were all untouched: only the two-line contraction of the dense tensor went.
///
/// The rebuild therefore performs that contraction here, from the `pub` `t3` /
/// `t4` fields, in the DEFINITIONAL full-nest form
/// `out[a][b] = Σ_c t3[a][b][c]·u[c]` / `Σ_{c,d} t4[a][b][c][d]·u[c]·v[d]`.
/// That is strictly better than what was lost: the oracle no longer routes
/// through a shared helper that the code under test could have been co-wrong
/// with, and the gate now owns nothing a symbol-table sweep can name.
#[test]
fn release_measure_rigid_contracted_towers_vs_generic_tower_932() {
    use super::row_kernel::{RigidRowInputs, rigid_row_nll};
    use gam_math::jet_scalar::{OneSeed, TwoSeed};
    use gam_math::jet_tower::program_full_tower;
    use gam_math::paired_timing::{SpeedGate, paired_interleaved};

    // The dense tower's contractions, written from its `pub` tensors. Closures,
    // not `fn`s: a reachability sweep computed from a stripped symbol table is
    // vacuously true of any test-only item, which is what orphaned this gate.
    let dense_third = |tower: &gam_math::jet_tower::Tower4<4>, dir: &[f64; 4]| {
        let mut out = [[0.0_f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                let mut acc = 0.0;
                for c in 0..4 {
                    acc += tower.t3[a][b][c] * dir[c];
                }
                out[a][b] = acc;
            }
        }
        out
    };
    let dense_fourth = |tower: &gam_math::jet_tower::Tower4<4>, u: &[f64; 4], v: &[f64; 4]| {
        let mut out = [[0.0_f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                let mut acc = 0.0;
                for c in 0..4 {
                    for d in 0..4 {
                        acc += tower.t4[a][b][c][d] * u[c] * v[d];
                    }
                }
                out[a][b] = acc;
            }
        }
        out
    };

    // One ordinary interior row per event branch (censored / event are
    // distinct live derivative stacks).
    let cases: [[f64; 8]; 2] = [
        [-0.7, 0.4, 0.8, -0.3, 0.6, 1.0, 0.0, 0.75],
        [0.2, -0.5, 1.4, 0.9, -1.1, 0.8, 1.0, 1.0],
    ];
    let dir_u = [0.7_f64, -1.3, 0.4, 0.6];
    let dir_v = [-0.4_f64, 0.6, 1.1, -0.2];

    // Non-vacuity: the parity assertions below must be judged against a tower
    // that is genuinely third- and fourth-order live. A fixture whose t3/t4
    // contractions were numerically zero would agree with anything.
    let mut max_dense_third = 0.0_f64;
    let mut max_dense_fourth = 0.0_f64;

    let mut gate = (!cfg!(debug_assertions)).then(|| SpeedGate::open("RIGID-CONTRACTED-932"));
    for &[q0, q1, qd1, g, z, w, d, probit_scale] in &cases {
        let inputs = RigidRowInputs {
            row: 0,
            wi: w,
            wi_entry: w,
            di: d,
            z_sum: z,
            covariance_ones: 1.0,
            probit_scale,
            qd1_lower: 1.0e-8,
            anchor: None,
        };
        let mut program = SurvivalMarginalSlopeRigidNllProgram {
            primaries: vec![[q0, q1, qd1, g]],
            z: vec![z],
            w: vec![w],
            d: vec![d],
            probit_scale,
        };

        // Parity pin on the exact benchmarked inputs: the specialized
        // contractions must equal the dense tower's contractions.
        let dense = program_full_tower(&program, 0).expect("dense tower");
        let third_vars: [OneSeed<4>; 4] =
            std::array::from_fn(|a| OneSeed::seed_direction([q0, q1, qd1, g][a], a, dir_u[a]));
        let third =
            rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&third_vars, &inputs)
                .expect("specialized third")
                .contracted_third();
        let fourth_vars: [TwoSeed<4>; 4] =
            std::array::from_fn(|a| TwoSeed::seed([q0, q1, qd1, g][a], a, dir_u[a], dir_v[a]));
        let fourth =
            rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&fourth_vars, &inputs)
                .expect("specialized fourth")
                .contracted_fourth();
        let dense_third_row = dense_third(&*dense, &dir_u);
        let dense_fourth_row = dense_fourth(&*dense, &dir_u, &dir_v);
        for a in 0..4 {
            for b in 0..4 {
                max_dense_third = max_dense_third.max(dense_third_row[a][b].abs());
                max_dense_fourth = max_dense_fourth.max(dense_fourth_row[a][b].abs());
                let band = 1e-11 * third[a][b].abs().max(dense_third_row[a][b].abs()).max(1.0);
                assert!(
                    (third[a][b] - dense_third_row[a][b]).abs() <= band,
                    "event={d:.0} third[{a}][{b}]: specialized {:+.15e} vs dense {:+.15e}",
                    third[a][b],
                    dense_third_row[a][b],
                );
                let band = 1e-11
                    * fourth[a][b]
                        .abs()
                        .max(dense_fourth_row[a][b].abs())
                        .max(1.0);
                assert!(
                    (fourth[a][b] - dense_fourth_row[a][b]).abs() <= band,
                    "event={d:.0} fourth[{a}][{b}]: specialized {:+.15e} vs dense {:+.15e}",
                    fourth[a][b],
                    dense_fourth_row[a][b],
                );
            }
        }

        let Some(gate) = gate.as_mut() else {
            continue;
        };
        // The nudge perturbs the slope primary, so no tower evaluation is
        // loop-invariant across calls.
        let third = paired_interleaved(
            15,
            20_000,
            0x9320_C0_03 ^ (d.to_bits() >> 60),
            |perturbation| {
                let perturbed_g = g + perturbation;
                let vars: [OneSeed<4>; 4] = std::array::from_fn(|a| {
                    OneSeed::seed_direction([q0, q1, qd1, perturbed_g][a], a, dir_u[a])
                });
                let t =
                    rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&vars, &inputs)
                        .expect("specialized third")
                        .contracted_third();
                t[0][0] + t[3][3]
            },
            |perturbation| {
                program.primaries[0][3] = g + perturbation;
                let tower = program_full_tower(&program, 0).expect("dense tower");
                let t = dense_third(&*tower, &dir_u);
                t[0][0] + t[3][3]
            },
        );
        gate.faster(
            &format!("order=3 event={d:.0}"),
            &third,
            "production",
            "generic_tower",
        );
        let fourth = paired_interleaved(
            15,
            20_000,
            0x9320_C0_04 ^ (d.to_bits() >> 60),
            |perturbation| {
                let perturbed_g = g + perturbation;
                let vars: [TwoSeed<4>; 4] = std::array::from_fn(|a| {
                    TwoSeed::seed([q0, q1, qd1, perturbed_g][a], a, dir_u[a], dir_v[a])
                });
                let t =
                    rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&vars, &inputs)
                        .expect("specialized fourth")
                        .contracted_fourth();
                t[0][0] + t[3][3]
            },
            |perturbation| {
                program.primaries[0][3] = g + perturbation;
                let tower = program_full_tower(&program, 0).expect("dense tower");
                let t = dense_fourth(&*tower, &dir_u, &dir_v);
                t[0][0] + t[3][3]
            },
        );
        gate.faster(
            &format!("order=4 event={d:.0}"),
            &fourth,
            "production",
            "generic_tower",
        );
    }
    assert!(
        max_dense_third > 1e-3 && max_dense_fourth > 1e-3,
        "the fixture's dense third/fourth contractions are numerically dead \
         (max|third|={max_dense_third:.3e}, max|fourth|={max_dense_fourth:.3e}), so the \
         parity assertions above would agree with anything"
    );
    if let Some(gate) = gate {
        gate.finish();
    }
}

/// The dense joint Hessian, and the default Jeffreys information built on it,
/// exist at every coefficient width. A retired `total >= 512` cutoff returned
/// `None` from `exact_newton_joint_hessian` while the row-kernel workspace kept
/// serving the same matrix, so above 512 columns the Jeffreys value path scored
/// `Φ = 0` against a step built from the workspace's Jeffreys term.
#[test]
fn joint_hessian_and_jeffreys_information_exist_above_512_columns() {
    let n = 6;
    let width = 520;
    let mut family = make_closed_form_test_family(n);
    let columns = Array2::from_shape_fn((n, width), |(row, column)| {
        ((row * 7 + column * 3) % 11) as f64 / 11.0 - 0.5
    });
    family.design_entry = DesignMatrix::from(columns.clone());
    family.design_exit = DesignMatrix::from(columns.clone());
    family.design_derivative_exit = DesignMatrix::from(columns);
    // Zero time coefficients leave q0, q1 and qd1 on their offsets, which the
    // closed-form fixture keeps admissible, while the time designs still carry
    // `width` columns of row curvature.
    let mut block_states = closed_form_block_states(&family, 0.3);
    block_states[0].beta = Array1::zeros(width);
    let specs = vec![dummy_blockspec(width), dummy_blockspec(0), dummy_blockspec(0)];

    let hessian = family
        .exact_newton_joint_hessian(&block_states)
        .expect("joint Hessian evaluation")
        .expect("a dense joint Hessian above 512 columns");
    assert_eq!(hessian.dim(), (width, width));
    assert!(
        hessian.iter().any(|value| *value != 0.0),
        "the time designs carry row curvature, so the joint Hessian cannot be zero"
    );
    let information = family
        .joint_jeffreys_information_with_specs(&block_states, &specs)
        .expect("Jeffreys information evaluation")
        .expect("the default Jeffreys information above 512 columns");
    assert_eq!(information, hessian);
}

/// gam#2971: a survival intercept is the root of its calibration identity, not
/// of its seed. A deep-tail exit index (`q₁ = −7.7`, the planted index at the
/// fixture's `t = 1e-3` floor, where the marginal failure probability is about
/// 7e-15) is solved cold, then from a warm slot seeded half a unit away. Each
/// root is certified independently: its log-tail residual lies within the
/// resolution the solve publishes, so each sits within that resolution over the
/// log slope of the true root, and two such roots differ by at most twice it.
/// The absolute probability residual this replaces met its `1e-12` at the seed
/// and returned the seed as the root.
#[test]
fn survival_intercept_root_does_not_follow_its_warm_seed_2971() {
    use super::family::{
        SurvivalInterceptSlotKind, SurvivalInterceptWarmStartCache, hash_intercept_warm_start_key,
        new_intercept_warm_start_cache,
    };
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let q0v = -8.5_f64;
    let q1v = -7.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let make_family = |cache: Option<Arc<SurvivalInterceptWarmStartCache>>| {
        SurvivalMarginalSlopeFamily {
            jeffreys_armed: true,
            latent_law: None,
            n: 1,
            entry_at_origin: Arc::new(Array1::from_elem(1, false)),
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.3].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![q0v]),
            offset_exit: Arc::new(array![q1v]),
            derivative_offset_exit: Arc::new(array![qd1v]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: cache,
            flex_jet_arenas: new_flex_jet_arena_pool(),
        }
    };
    let beta_h = Array1::from_iter((0..h_dim).map(|k| 0.04 * (k as f64 + 1.3).sin()));
    let beta_w = Array1::from_iter((0..w_dim).map(|k| 0.035 * (k as f64 + 0.7).cos()));

    let cold = make_family(None);
    let (a_cold, density_cold) = cold
        .solve_row_survival_intercept_with_slot(q1v, gv, Some(&beta_h), Some(&beta_w), None)
        .expect("cold survival intercept solve");

    // The slot is keyed on every input of the exit equation, exactly as the
    // production solve keys it; the entry equation at `q₀` shares the
    // coefficients and the row but must not share the key.
    let law = cold.flex_law_grid(Some(0)).expect("row law");
    let probit_scale = cold.probit_frailty_scale();
    let exit_key =
        hash_intercept_warm_start_key(q1v, gv, probit_scale, law, Some(&beta_h), Some(&beta_w));
    let entry_key =
        hash_intercept_warm_start_key(q0v, gv, probit_scale, law, Some(&beta_h), Some(&beta_w));
    assert_ne!(
        exit_key, entry_key,
        "equations with different targets must not share a warm-start key"
    );
    let cache = new_intercept_warm_start_cache(1);
    cache.store(0, SurvivalInterceptSlotKind::Exit, a_cold + 0.5, exit_key);
    let warm = make_family(Some(Arc::clone(&cache)));
    let (a_warm, density_warm) = warm
        .solve_row_survival_intercept_with_slot(
            q1v,
            gv,
            Some(&beta_h),
            Some(&beta_w),
            Some((0, SurvivalInterceptSlotKind::Exit)),
        )
        .expect("warm-seeded survival intercept solve");

    // The log slope at the root is `|T′|/T`, from the returned density and the
    // target tail `Φ(q₁)`; the resolution is the production certificate's.
    let log_target = crate::probability::normal_logcdf(q1v);
    let log_slope = density_cold / log_target.exp();
    let terms = score_runtime.breakpoints().len() + link_runtime.breakpoints().len() + 1;
    let rounding = crate::latent_anchor::anchor_residual_rounding(log_target, terms);
    let resolution = crate::latent_anchor::anchor_residual_resolution(a_cold, log_slope, rounding);
    let bound = 2.0 * resolution / log_slope;
    let gap = (a_warm - a_cold).abs();
    eprintln!(
        "survival intercept 2971: a_cold={a_cold:.15e} a_warm={a_warm:.15e} gap={gap:.3e} \
         bound={bound:.3e} density_cold={density_cold:.6e} density_warm={density_warm:.6e} \
         log_slope={log_slope:.6e}"
    );
    assert!(
        log_slope.is_finite() && log_slope > 0.0,
        "the cold root must carry a finite positive log slope: {log_slope:.3e}"
    );
    assert!(
        gap <= bound,
        "a warm seed half a unit from the root moved the certified intercept: a_cold={a_cold:.15e} \
         a_warm={a_warm:.15e} gap={gap:.3e} > bound={bound:.3e}"
    );
}

/// #2900 row 6.11 and gam#3024: the rigid survival row jet is weighed by its
/// own two executors, and the device returns the per-row CPU program. On a
/// host without a device nothing is admitted or raced, and the cache is the
/// per-row loop. On a CUDA host the first `auto` build of an untimed shape
/// races the per-row loop against the device pass and returns the per-row
/// result bit for bit; the shape is then decided from its timing without
/// another race, and the device pass is compared with `row_kernel(row)` on
/// every channel of every row at the `RowKernel::batched_value_grad_hess_all`
/// contract (≤ 1e-9). Two rows in seven are shifted 5 units into either
/// probability tail.
#[test]
fn rigid_row_jet_device_admission_and_parity_2900() {
    use crate::row_kernel::{RowKernel, RowSet, build_row_kernel_cache};
    let n = 2_048;
    let decide = || {
        rigid_row_jet_decision::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(n)
            .expect("survival row-jet admission must not fault")
    };
    let runtime = gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())
        .expect("CUDA runtime resolution must not fault");

    let mut family = make_closed_form_test_family(n);
    let into_tails = |values: &Array1<f64>| {
        Array1::from_iter(values.iter().enumerate().map(|(row, &value)| match row % 7 {
            3 => value + 5.0,
            5 => value - 5.0,
            _ => value,
        }))
    };
    family.offset_entry = Arc::new(into_tails(&family.offset_entry));
    family.offset_exit = Arc::new(into_tails(&family.offset_exit));
    let block_states = closed_form_block_states(&family, 0.4);
    let kernel = SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
        family,
        block_states,
    );
    let per_row: Vec<_> = (0..n)
        .map(|row| RowKernel::row_kernel(&kernel, row).expect("per-row CPU program"))
        .collect();
    let worst_gap =
        |nll: &[f64],
         gradients: &[[f64; STATIC_SLOPE_PRIMARIES]],
         hessians: &[[[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES]]| {
            let mut worst = (0.0_f64, 0);
            for (row, (value, grad, hess)) in per_row.iter().enumerate() {
                let channels = std::iter::once((nll[row], *value))
                    .chain(gradients[row].iter().copied().zip(grad.iter().copied()))
                    .chain(
                        hessians[row]
                            .iter()
                            .flatten()
                            .copied()
                            .zip(hess.iter().flatten().copied()),
                    );
                for (batched, cpu) in channels {
                    let gap = (batched - cpu).abs() / 1.0_f64.max(batched.abs()).max(cpu.abs());
                    if !(gap <= worst.0) {
                        worst = (gap, row);
                    }
                }
            }
            worst
        };

    let first = decide();
    if runtime.is_none() {
        assert_eq!(
            (first.use_gpu, first.race),
            (false, None),
            "a host without a device admits and races nothing: {}",
            first.reason
        );
    } else {
        assert!(
            first.race.is_some() || first.reason == "cpu-device-measured-slower" || first.use_gpu,
            "an untimed shape on a CUDA host is raced: {}",
            first.reason
        );
    }
    let raced = first.race.is_some();
    let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("rigid row-kernel cache");
    let (cache_gap, cache_row) = worst_gap(&cache.nll, &cache.gradients, &cache.hessians);
    if raced || !first.use_gpu {
        assert_eq!(
            cache_gap, 0.0,
            "a raced or CPU-decided build returns the per-row loop bit for bit; row {cache_row}"
        );
        if raced {
            assert!(decide().race.is_none(), "a raced shape is decided from its timing");
        }
    }
    #[cfg(target_os = "linux")]
    let device_gap = runtime.map(|_| {
        let (nll, gradients, hessians) = kernel
            .rigid_row_jet_on_device()
            .expect("the device row jet runs on a CUDA host");
        worst_gap(&nll, &gradients, &hessians)
    });
    #[cfg(not(target_os = "linux"))]
    let device_gap: Option<(f64, usize)> = None;
    eprintln!(
        "#2900/#3024 survival row jet: n={n} raced={raced} decision={} cache_gap={cache_gap:.3e} \
         device_gap={device_gap:?}",
        first.reason
    );
    assert!(
        cache_gap <= 1e-9,
        "survival row jet: the cache differs from the per-row program by {cache_gap:e} \
         (relative) at row {cache_row}"
    );
    if let Some((gap, row)) = device_gap {
        assert!(
            gap <= 1e-9,
            "survival row jet: the device pass differs from the per-row program by {gap:e} \
             (relative) at row {row}"
        );
    }
}

/// gam#3000 slice 2: the device row jet declares the four-primary Gaussian
/// frame only, and each frame's own declarations say what it asks for. A
/// follow-up-varying or anchored frame is outside the declaration at every row
/// count, so it is decided on the CPU with the missing capability named.
#[test]
fn rigid_row_jet_capability_follows_the_frame_declarations_3000() {
    use crate::gpu_kernels::survival_rowjet::SURVIVAL_ROWJET_CAPABILITY;

    let gaussian = rigid_row_jet_model::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>();
    assert_eq!(SURVIVAL_ROWJET_CAPABILITY.missing_for(&gaussian), None);
    let follow_up = rigid_row_jet_model::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>();
    assert_eq!(
        SURVIVAL_ROWJET_CAPABILITY.missing_for(&follow_up),
        Some("the follow-up-varying slope frame")
    );
    let anchored = rigid_row_jet_model::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>();
    assert_eq!(
        SURVIVAL_ROWJET_CAPABILITY.missing_for(&anchored),
        Some("the anchored lowering of a declared latent law")
    );
    for n in [0, 10_000_000] {
        for decision in [
            rigid_row_jet_decision::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(n),
            rigid_row_jet_decision::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(n),
        ] {
            let decision =
                decision.expect("auto and off refuse no model; required is never set here");
            assert!(!decision.use_gpu, "n={n}: {decision:?}");
            assert!(decision.missing_capability.is_some(), "n={n}: {decision:?}");
        }
    }
}

/// An equal-mass law at `m` standard-normal quantiles: the midpoint rule on the
/// probability scale, whose error on a smooth integrand is O(1/m²).
fn normal_quantile_law(m: usize) -> crate::bms::EmpiricalZGrid {
    let nodes: Vec<f64> = (0..m)
        .map(|k| {
            gam_math::probability::standard_normal_quantile((k as f64 + 0.5) / m as f64)
                .expect("standard-normal quantile")
        })
        .collect();
    crate::bms::EmpiricalZGrid::new(nodes, vec![1.0 / m as f64; m], "normal quantiles")
        .expect("equal-mass normal law")
}

/// gam#2926: the flex program's anchoring residual under a law that is its own
/// `N(0, 1)` vanishes, because the row's intercept anchors exactly that integral,
/// and without flex coefficients the program's residual is the rigid closed form's.
#[test]
fn flex_survival_anchoring_residual_matches_its_calibration_and_the_rigid_form_2926() {
    let family = make_flex_no_wiggle_test_family(8);
    let block_states = flex_no_wiggle_test_block_states(&family);
    let beta_h = family
        .flex_score_beta(&block_states)
        .expect("score-warp coefficients");
    assert!(
        beta_h.is_some(),
        "the fixture must carry score-warp coefficients, or this test measures the rigid form twice"
    );
    let law = normal_quantile_law(4001);
    for &(q, slope) in &[(-1.5, 0.4), (0.3, 0.8), (2.0, -0.6)] {
        let (flex_residual, _, _, _) = family
            .flex_survival_anchoring_residual(q, slope, beta_h, None, &law)
            .expect("flex anchoring residual");
        assert!(
            flex_residual.abs() < 1e-5,
            "under the program's own N(0, 1) the flex residual must vanish: q={q} slope={slope} \
             residual={flex_residual:e}"
        );
        let (through_flex, sd_through_flex, scale_through_flex, _) = family
            .flex_survival_anchoring_residual(q, slope, None, None, &law)
            .expect("rigid anchoring residual through the flex program");
        let (rigid, sd_rigid, scale_rigid, _) =
            crate::bms::estimated_latent_law::closed_form_survival_anchoring_residual(
                q,
                family.probit_frailty_scale() * slope,
                &law,
            );
        assert!(
            (through_flex - rigid).abs() <= 1e-10
                && (sd_through_flex - sd_rigid).abs() <= 1e-10
                && scale_through_flex == scale_rigid,
            "without flex coefficients the program's residual must be the rigid form's: q={q} \
             slope={slope} flex=({through_flex:e}, {sd_through_flex:e}) rigid=({rigid:e}, \
             {sd_rigid:e})"
        );
    }
}

/// gam#2926: the closed-form certificate scores each anchor's defining equation,
/// which is offset-free. With a CTN Stage-1 influence absorber installed and a
/// nonzero offset `o_infl` on every row, the certificate's anchors are exactly the
/// ones the same rows have with no absorber.
#[test]
fn closed_form_certificate_anchors_exclude_the_influence_offset_2926() {
    let n = 8;
    let family = make_flex_no_wiggle_test_family(n);
    let block_states = flex_no_wiggle_test_block_states(&family);
    let law = normal_quantile_law(401);

    let mut absorbed = make_flex_no_wiggle_test_family(n);
    absorbed.influence_absorber = Some(Array2::from_shape_fn((n, 1), |(row, _)| {
        0.3 + 0.1 * row as f64
    }));
    // The absorber is the trailing block, after the score-warp block.
    let mut absorbed_states = block_states.clone();
    let mut influence = block_states[3].clone();
    influence.beta = array![0.7];
    absorbed_states.push(influence);

    for row in 0..n {
        let offset = absorbed
            .influence_index_offset(row, &absorbed_states)
            .expect("absorber offset");
        assert!(
            offset.abs() > 0.1,
            "fixture invariant: row {row} must carry a nonzero absorber offset, got {offset}"
        );
        let plain = family
            .closed_form_certificate_anchors(row, &block_states, &law)
            .expect("certificate anchors without an absorber");
        let with_absorber = absorbed
            .closed_form_certificate_anchors(row, &absorbed_states, &law)
            .expect("certificate anchors with an absorber");
        assert_eq!(
            plain, with_absorber,
            "row {row}: the certificate must score the offset-free anchor, whatever o_infl = {offset}"
        );
    }
}

/// gam#2971: a survival event row's likelihood keeps its value and its slope as
/// the row's link-deviation argument `u = a₁ + g·z` crosses the support's right
/// end. The event density carries `ln χ₁`, `χ₁ = 1 + w′(u)`, and outside the
/// support the deviation is flat, so a link basis whose `w′` survives at the end
/// made the row likelihood jump by `ln(1 + w′(end))` there. The inner Newton
/// crept toward that cliff and never certified. The likelihood's gradient reads
/// `w″` through `∂χ₁/∂a`, so its slope jumps as well unless `w″` also vanishes.
///
/// The exit index `q₁*` putting `u` on the end is bisected, and the row
/// likelihood is read at `q₁* ± h, ± 2h, ± 3h`. Each side's three points give a
/// quadratic extrapolation of the value and the slope at `q₁*`. Its gap to the
/// linear extrapolation from the two nearer points is that side's truncation
/// estimate. The two sides must agree within the sum of their estimates.
#[test]
fn link_deviation_row_likelihood_is_c1_across_its_support_end_2971() {
    let score_runtime = test_deviation_runtime();
    let link_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let link_runtime = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed,
        &link_seed,
        &DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 3,
            penalty_order: 2,
            penalty_orders: vec![1, 2, 3],
            double_penalty: false,
            monotonicity_eps: 1e-4,
        },
    )
    .expect("build the production survival link deviation")
    .runtime;
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let q0v = -0.25_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.3].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![q0v]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![qd1v]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let beta_h = Array1::from_iter((0..h_dim).map(|k| 0.04 * (k as f64 + 1.3).sin()));
    let beta_w = Array1::from_iter((0..w_dim).map(|k| 0.035 * (k as f64 + 0.7).cos()));
    let z_obs = family.observed_score_projection(0);
    let breakpoints = link_runtime.breakpoints();
    let right_end = breakpoints[breakpoints.len() - 1];
    let last_interior = breakpoints[breakpoints.len() - 2];
    let argument = |q1: f64| -> f64 {
        let (a1, _) = family
            .solve_row_survival_intercept_with_slot(q1, gv, Some(&beta_h), Some(&beta_w), None)
            .expect("exit intercept solve");
        a1 + gv * z_obs
    };
    let neglog = |q1: f64| -> f64 {
        family
            .row_neglog_flex_value_from_parts(
                0,
                q0v,
                q1,
                qd1v,
                gv,
                Some(&beta_h),
                Some(&beta_w),
                0.0,
            )
            .expect("row neglog")
    };

    // The argument rises with q₁; bisect until the bracket stops shrinking.
    let (mut lo, mut hi) = (-4.0_f64, 4.0_f64);
    assert!(
        argument(lo) < right_end && argument(hi) > right_end,
        "the bracket must straddle the support end {right_end}: u(lo)={:.6} u(hi)={:.6}",
        argument(lo),
        argument(hi)
    );
    loop {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if argument(mid) <= right_end {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let q_star = 0.5 * (lo + hi);
    let h = 1e-3;
    let values: Vec<(f64, f64, f64)> = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        .iter()
        .map(|&k: &f64| {
            let q = q_star + k * h;
            (q, argument(q), neglog(q))
        })
        .collect();
    for (q, u, value) in &values {
        eprintln!(
            "support end 2971: q1={q:.12e} u={u:.12e} (end {right_end:.6}) neglog={value:.15e}"
        );
    }
    // Every left point lies in the last span and every right point in the flat
    // tail, so each side's stencil reads one polynomial piece.
    assert!(
        values[..3]
            .iter()
            .all(|&(_, u, _)| u > last_interior && u <= right_end)
            && values[3..].iter().all(|&(_, u, _)| u > right_end),
        "the stencil must sit in the last span on the left and in the tail on the right"
    );
    let side = |near: f64, mid: f64, far: f64, sign: f64| -> (f64, f64, f64, f64) {
        let value = 3.0 * near - 3.0 * mid + far;
        let value_linear = 2.0 * near - mid;
        let slope = sign * (2.5 * near - 4.0 * mid + 1.5 * far) / h;
        let slope_linear = sign * (near - mid) / h;
        (
            value,
            (value - value_linear).abs(),
            slope,
            (slope - slope_linear).abs(),
        )
    };
    let (left_value, left_value_est, left_slope, left_slope_est) =
        side(values[2].2, values[1].2, values[0].2, 1.0);
    let (right_value, right_value_est, right_slope, right_slope_est) =
        side(values[3].2, values[4].2, values[5].2, -1.0);
    let value_gap = (left_value - right_value).abs();
    let slope_gap = (left_slope - right_slope).abs();
    eprintln!(
        "support end 2971: q1*={q_star:.12e} value left={left_value:.15e} right={right_value:.15e} \
         gap={value_gap:.3e} bound={:.3e} | slope left={left_slope:.12e} right={right_slope:.12e} \
         gap={slope_gap:.3e} bound={:.3e}",
        left_value_est + right_value_est,
        left_slope_est + right_slope_est
    );
    assert!(
        value_gap <= left_value_est + right_value_est,
        "the row likelihood jumps at the link support end: gap={value_gap:.3e} > bound={:.3e}",
        left_value_est + right_value_est
    );
    assert!(
        slope_gap <= left_slope_est + right_slope_est,
        "the row likelihood's slope jumps at the link support end: gap={slope_gap:.3e} > \
         bound={:.3e}",
        left_slope_est + right_slope_est
    );
}
