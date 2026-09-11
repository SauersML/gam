//! The automatic latent-score measure gate for the survival marginal-slope
//! family (gam#2768).
//!
//! The Bernoulli marginal-slope family has run an automatic gate on its latent
//! score since #905: a Rao score test on `E[z|C]` and `Var(z|C)` over the
//! marginal-index span, escalating to the conditional location-scale correction
//! `ζ = (z − m(C))/√v(C)` when it fires, with rank inverse-normal and empirical
//! fallbacks below it. The survival marginal-slope family ran *none* of it. It
//! called `standardize_latent_z_with_policy` and nothing else, and under the
//! default policy — `Frozen { mean: 0, sd: 1 }` — that transform is the
//! identity: it checked, it warned, and it passed z through unchanged.
//!
//! That is not a cosmetic gap. The survival row index is
//! `η = q·c(g) + s(g)·z`, so a conditional shift `E[z|C] = m(C) ≠ 0` puts
//! `s(g(C))·m(C)` into the *influence* channel `q` — the same `b(C)·m(C)`
//! leakage the Bernoulli gate exists to remove, in a model whose whole point is
//! that `q` is the marginal index. The pooled marginal gate cannot see it (the
//! marginal law of z can be exactly N(0,1) while every conditional law is
//! shifted), and rank-INT provably cannot fix it (no transform depending only on
//! the marginal `F_Z` can enforce `E[T(Z)|C] ≡ const`).
//!
//! This module is the survival caller of the *shared* gate
//! ([`build_latent_measure_decision`]), not a second copy of it. The one thing
//! it declares that BMS does not is the family's kernel capability: the survival
//! row program is the closed-form standard-normal probit lowering and owns no
//! empirical-grid branch, so it asks for
//! [`EmpiricalLatentMeasureSupport::StandardNormalOnly`] and routes the gate's
//! residual verdict through the spec's own [`LatentZCheckMode`].

use super::*;

use crate::bms::{
    EmpiricalLatentMeasureSupport, LatentMeasureCalibration, LatentMeasureKind, LatentZCheckMode,
    LatentZConditionalCalibration, build_latent_measure_decision,
};

/// Everything the fit and its persistence need from the automatic gate: the
/// per-coordinate decisions, the score the gate saw *before* calibrating it, and
/// the conditioning block it conditioned on.
///
/// The raw score and the conditioning block are not diagnostics. The
/// Murphy-Topel generated-regressor correction needs both — it differentiates
/// the first stage, which regressed the RAW score on that block — and the
/// conditioning block is also what the persistence gate compares against the
/// finally-resolved marginal design to decide whether prediction can reproduce
/// the map at all.
pub(crate) struct SurvivalLatentScoreCalibration {
    /// One decision per latent-score column, in column order.
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    /// The (normalised, pre-calibration) latent scores the gate was handed.
    pub(crate) raw_scores: Array2<f64>,
    /// The conditioning span `a(C)` the conditional branch used, when it was
    /// built. `None` when the CTN Stage-1 absorber suppressed it.
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

impl SurvivalLatentScoreCalibration {
    /// The conditional location-scale calibration on the PRIMARY score, if the
    /// Rao gate escalated to one. This is the only branch with a generated
    /// regressor: rank-INT is a fixed monotone map of the marginal ECDF and the
    /// identity is not estimated at all.
    pub(crate) fn primary_conditional(&self) -> Option<&LatentZConditionalCalibration> {
        match self.per_score.first() {
            Some(LatentMeasureCalibration::ConditionalLocationScale(cal)) => Some(cal),
            _ => None,
        }
    }
}

/// Run the automatic latent-measure gate over every latent-score coordinate and
/// replace `spec.z` by the calibrated score in place.
///
/// Returns one [`LatentMeasureCalibration`] per z column, in column order, for
/// persistence: prediction MUST apply the identical map, so the fit's decision
/// travels with the model rather than being re-derived from the prediction
/// sample.
///
/// # Why per coordinate
///
/// With `K > 1` latent scores the row index is `η = q·c + Σ_k s(g_k)·z_k`, so
/// the leakage is `Σ_k s(g_k(C))·m_k(C)` — a sum of per-coordinate conditional
/// shifts. Under the current single global score covariance `Σ`, the
/// K-generalisation of the correction is therefore the per-coordinate
/// conditional standardisation on the same basis `a(C)`, after which `Σ` is
/// recomputed from the calibrated scores by the caller. (A conditional `Σ(C)` is
/// a different and larger question; it is gam#2766.)
pub(crate) fn resolve_survival_latent_score_calibration(
    spec: &mut SurvivalMarginalSlopeTermSpec,
    marginal_design: &TermCollectionDesign,
) -> Result<SurvivalLatentScoreCalibration, String> {
    // #461 seam, mirrored from BMS: when a CTN Stage-1 influence absorber is
    // active the conditional leakage is already absorbed by the absorber's own
    // orthogonalisation, and replacing z here would perturb the widened-marginal
    // predict seam. The conditional gate is then not engaged; the pooled gates
    // below it still are, because they are about the marginal law of z and the
    // absorber says nothing about that.
    let absorber_active = spec
        .score_influence_jacobian
        .as_ref()
        .is_some_and(|jacobian| jacobian.ncols() > 0);
    let resolved = resolve_latent_score_calibration_from_parts(
        &spec.z,
        &spec.weights,
        &spec.latent_z_policy,
        absorber_active,
        &marginal_design.design,
    )?;
    spec.z = resolved.calibrated_scores;
    Ok(SurvivalLatentScoreCalibration {
        per_score: resolved.per_score,
        raw_scores: resolved.raw_scores,
        conditioning: resolved.conditioning,
    })
}

/// The gate itself, over exactly the five things it reads.
///
/// Split out from the spec-shaped entry point above so it is directly
/// exercisable: the decision is about `z`, the weights, the policy, the
/// absorber flag and the marginal design, and nothing else about a survival term
/// spec bears on it.
pub(crate) struct ResolvedLatentScoreCalibration {
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    pub(crate) raw_scores: Array2<f64>,
    pub(crate) calibrated_scores: Array2<f64>,
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

pub(crate) fn resolve_latent_score_calibration_from_parts(
    scores: &Array2<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    absorber_active: bool,
    marginal_design: &DesignMatrix,
) -> Result<ResolvedLatentScoreCalibration, String> {
    let k = scores.ncols();
    let raw_scores = scores.clone();
    let conditioning = if absorber_active {
        None
    } else {
        Some(marginal_design.try_to_dense_arc("survival marginal-slope conditional latent-z gate")?)
    };

    let mut calibrations = Vec::with_capacity(k);
    let mut calibrated_scores = scores.clone();
    for col in 0..k {
        let raw = scores.column(col).to_owned();
        let decision = build_latent_measure_decision(
            &raw,
            weights,
            policy,
            conditioning.as_ref().map(|design| design.view()),
            EmpiricalLatentMeasureSupport::StandardNormalOnly,
            "survival-marginal-slope",
        )?;
        if !matches!(decision.kind, LatentMeasureKind::StandardNormal) {
            // Unreachable by construction — `StandardNormalOnly` never returns
            // another kind — but this family's whole kernel rests on it, so the
            // invariant is checked rather than assumed.
            return Err(
                "survival marginal-slope latent-measure gate returned a non-standard-normal \
                 measure for a standard-normal-only kernel"
                    .to_string(),
            );
        }
        if let Some(adequacy) = decision.unmodelled_residual.as_ref() {
            let message = format!(
                "survival-marginal-slope latent score column {col} still fails the \
                 standard-normal adequacy gate after the automatic {} calibration, and this \
                 family's row kernel has no empirical latent measure to carry the residual law: \
                 the closed-form standard-normal probit kernel is being applied to a sample the \
                 gate rejects. Point estimation still uses the calibrated axis (it is the closest \
                 available to the kernel's own assumption); what is unmodelled is the residual \
                 SHAPE. Adequacy ledger (x = statistic / bound, x<=1 passed): {}",
                calibration_label(&decision.calibration),
                adequacy.ledger(),
            );
            match policy.check_mode {
                LatentZCheckMode::Strict => return Err(message),
                LatentZCheckMode::WarnOnly => log::warn!("{message}"),
                LatentZCheckMode::Off => {}
            }
        }
        let calibrated = match &decision.calibration {
            LatentMeasureCalibration::None => raw,
            LatentMeasureCalibration::RankInverseNormal(cal) => cal.apply_to_training(&raw)?,
            LatentMeasureCalibration::ConditionalLocationScale(cal) => {
                // The conditional branch is only reachable when the gate had a
                // conditioning block to fire on, so it is present here.
                let a_block = conditioning.as_ref().ok_or_else(|| {
                    "survival marginal-slope conditional latent calibration requires the \
                     marginal conditioning block"
                        .to_string()
                })?;
                cal.apply(raw.view(), a_block.view())?
            }
        };
        if !matches!(decision.calibration, LatentMeasureCalibration::None) {
            log::info!(
                "[survival-marginal-slope latent-z] score column {col}: applied the {} \
                 calibration before any downstream consumer saw the score",
                calibration_label(&decision.calibration),
            );
        }
        calibrated_scores.column_mut(col).assign(&calibrated);
        calibrations.push(decision.calibration);
    }
    Ok(ResolvedLatentScoreCalibration {
        per_score: calibrations,
        raw_scores,
        calibrated_scores,
        conditioning,
    })
}

/// Decide the score-covariance field the fit will consume (gam#2766).
///
/// Split out from [`fit_survival_marginal_slope_terms_impl`] for the same reason
/// [`resolve_latent_score_calibration_from_parts`] is: the decision is about the
/// calibrated scores, the weights and the conditioning block, and nothing else
/// about a survival term spec bears on it — so it should be exercisable without
/// standing up a whole fit.
///
/// `scores` must be the axis the row kernel will see, i.e. AFTER
/// [`resolve_survival_latent_score_calibration`]. `Σ` is the covariance of the
/// score the likelihood integrates over, so running this on the raw axis would
/// model a different object than the one `c(a)` has to correct for.
///
/// Returns the pooled field when there is no conditioning block (the #461
/// absorber suppressed it — there is then no span to condition on and the
/// pooled matrix is the only defined answer), and when the pair-wise Rao gate
/// does not fire.
pub(crate) fn resolve_score_covariance_field(
    scores: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    conditioning: Option<ArrayView2<'_, f64>>,
) -> Result<ScoreCovarianceField, String> {
    let pooled = marginal_slope_covariance_from_scores(scores, &weights.to_owned())?;
    let Some(conditioning) = conditioning else {
        return Ok(ScoreCovarianceField::pooled(pooled));
    };
    match crate::bms::ConditionalScoreCovariance::fit(scores, weights, conditioning)? {
        Some(model) => {
            log::info!(
                "[survival-marginal-slope] conditional score covariance ENGAGED on K={} scores: \
                 pair Rao p-values {:?}",
                model.score_dim,
                model.pair_pvalues,
            );
            ScoreCovarianceField::conditional(pooled, model, conditioning)
        }
        None => Ok(ScoreCovarianceField::pooled(pooled)),
    }
}

fn calibration_label(calibration: &LatentMeasureCalibration) -> &'static str {
    match calibration {
        LatentMeasureCalibration::None => "identity",
        LatentMeasureCalibration::RankInverseNormal(_) => "rank inverse-normal",
        LatentMeasureCalibration::ConditionalLocationScale(_) => "conditional location-scale",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bms::{LatentZCheckMode, LatentZNormalizationMode};
    use gam_linalg::matrix::DenseDesignMatrix;

    /// Deterministic standard normals (Box–Muller over splitmix64).
    fn gaussians(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            // `unit` draws from the open interval (0, 1), so `ln u1` is finite.
            let u1 = unit();
            let u2 = unit();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(r * (std::f64::consts::TAU * u2).cos());
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
        out.truncate(n);
        out
    }

    fn standardized(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
        for value in v.iter_mut() {
            *value = (*value - mean) / sd;
        }
        v
    }

    /// `z = m·x + √(1−m²)·ζ`, exactly standard normal marginally and
    /// conditionally shifted on the marginal design's `x` column.
    fn shifted_fixture(n: usize, m: f64) -> (Array2<f64>, Array1<f64>, DesignMatrix, Vec<f64>) {
        let x = standardized(gaussians(n, 0x2768_A1));
        let zeta = standardized(gaussians(n, 0x2768_B2));
        let residual_sd = (1.0 - m * m).sqrt();
        let mut z = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            z[[row, 0]] = m * x[row] + residual_sd * zeta[row];
        }
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            design[[row, 1]] = x[row];
        }
        (
            z,
            Array1::<f64>::ones(n),
            DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            zeta,
        )
    }

    fn auto_policy(check_mode: LatentZCheckMode) -> LatentZPolicy {
        LatentZPolicy {
            check_mode,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            ..LatentZPolicy::frozen_transformation_normal()
        }
    }

    /// The gate must fire on a conditionally shifted score, and the score it
    /// hands the kernel must be the CLEAN one — not merely centred (gam#2768).
    ///
    /// Recovering `ζ` up to sign and a common scale is the whole claim: the fit's
    /// coefficients are then the ones the outcome was generated with, and the
    /// influence channel `q` no longer carries `b(C)·m(C)`.
    #[test]
    fn survival_gate_recovers_the_conditionally_standardized_score() {
        let n = 4000;
        let m = 0.6;
        let (z, weights, design, zeta_truth) = shifted_fixture(n, m);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");

        assert_eq!(resolved.per_score.len(), 1, "one decision per score column");
        assert!(
            matches!(
                resolved.per_score[0],
                LatentMeasureCalibration::ConditionalLocationScale(_)
            ),
            "the E[z|C] Rao gate must escalate to the conditional location-scale \
             correction at Corr(z, x) = {m} and n = {n}; got {}",
            calibration_label(&resolved.per_score[0])
        );

        // The calibrated column must be ζ itself. Correlation, because the
        // calibration is fit rather than known and carries estimation error in
        // m̂ and in the residual scale.
        let calibrated = resolved.calibrated_scores.column(0);
        let dot: f64 = calibrated
            .iter()
            .zip(zeta_truth.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_cal = calibrated.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_truth = zeta_truth.iter().map(|v| v * v).sum::<f64>().sqrt();
        let correlation = dot / (norm_cal * norm_truth);
        assert!(
            correlation > 0.9995,
            "the calibrated score must BE the clean score; correlation {correlation:.6}"
        );

        // Unit variance, which is what keeps `q` the marginal index.
        let mean = calibrated.iter().sum::<f64>() / n as f64;
        let sd = (calibrated.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(
            mean.abs() < 0.02 && (sd - 1.0).abs() < 0.05,
            "calibrated score must be standardised; mean={mean:.4} sd={sd:.4}"
        );

        // The raw score is kept for the generated-regressor correction, which
        // differentiates the FIRST stage and therefore needs the axis that stage
        // regressed — not the one the fit consumed.
        assert_eq!(resolved.raw_scores, z, "the raw score must survive the gate");
        assert!(
            resolved.conditioning.is_some(),
            "the conditional branch must retain the block it conditioned on"
        );
    }

    /// The gate must NOT fire on a score that is already conditionally standard
    /// normal. A trigger-happy gate would redefine the latent axis of every
    /// clean fit, which is a worse failure than the one it exists to prevent.
    #[test]
    fn survival_gate_stays_quiet_on_an_unshifted_score() {
        let n = 4000;
        let (_, weights, design, zeta) = shifted_fixture(n, 0.6);
        let mut clean = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            clean[[row, 0]] = zeta[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &clean,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");
        assert!(
            matches!(resolved.per_score[0], LatentMeasureCalibration::None),
            "got {}",
            calibration_label(&resolved.per_score[0])
        );
        assert_eq!(
            resolved.calibrated_scores, clean,
            "an unfired gate must leave the score untouched, byte for byte"
        );
    }

    /// `K = 2` scores whose conditional CORRELATION is `amplitude·x/√(1+x²)` and
    /// whose conditional marginals are exactly `N(0,1)`. At `amplitude = 0` the
    /// conditional covariance is constant. This is the gam#2766 shape: nothing
    /// for the per-coordinate gate above to correct, everything in the
    /// off-diagonal.
    fn varying_correlation_fixture(
        n: usize,
        amplitude: f64,
    ) -> (Array2<f64>, Array1<f64>, DesignMatrix) {
        let x = standardized(gaussians(n, 0x2766_11));
        let e0 = standardized(gaussians(n, 0x2766_22));
        let e1 = standardized(gaussians(n, 0x2766_33));
        let mut scores = Array2::<f64>::zeros((n, 2));
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            design[[row, 1]] = x[row];
            let phi = amplitude * x[row] / (1.0 + x[row] * x[row]).sqrt();
            scores[[row, 0]] = e0[row];
            scores[[row, 1]] = phi * e0[row] + (1.0 - phi * phi).max(0.0).sqrt() * e1[row];
        }
        (
            scores,
            Array1::<f64>::ones(n),
            DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        )
    }

    /// The fit installs a per-row `Σ(a_i)` when the pair gate fires, and the
    /// rows genuinely differ. Without this the model and the row program could
    /// both be right while the entry point never wires them together.
    #[test]
    fn the_fit_installs_a_conditional_field_when_the_pair_gate_fires() {
        let n = 20_000;
        let (scores, weights, design) = varying_correlation_fixture(n, 0.8);
        let conditioning = design
            .try_to_dense_arc("test conditioning")
            .expect("dense conditioning");
        let field = resolve_score_covariance_field(
            scores.view(),
            weights.view(),
            Some(conditioning.view()),
        )
        .expect("field");
        assert!(field.is_conditional(), "a varying Cov(z₀,z₁|a) must escalate");
        assert_eq!(field.materialised_rows(), Some(n));
        assert_eq!(field.dim(), 2);
        // The correlation must track the planted sign across the covariate.
        let correlation = |row: usize| {
            let sigma = field.at_row(row).to_dense();
            sigma[[0, 1]] / (sigma[[0, 0]] * sigma[[1, 1]]).sqrt()
        };
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for row in 0..n {
            let value = correlation(row);
            lowest = lowest.min(value);
            highest = highest.max(value);
        }
        assert!(
            lowest < -0.4 && highest > 0.4,
            "the installed field must span the planted correlation range; got [{lowest:.3}, {highest:.3}]"
        );
    }

    /// A constant conditional covariance leaves the pooled object in place, and
    /// so does a suppressed conditioning block: with no span to condition on the
    /// pooled matrix is the only defined answer, which is the #461 absorber
    /// seam one level up.
    #[test]
    fn the_fit_keeps_the_pooled_field_without_a_trigger_or_a_span() {
        let n = 8_000;
        let (scores, weights, design) = varying_correlation_fixture(n, 0.0);
        let conditioning = design
            .try_to_dense_arc("test conditioning")
            .expect("dense conditioning");
        let constant = resolve_score_covariance_field(
            scores.view(),
            weights.view(),
            Some(conditioning.view()),
        )
        .expect("field");
        assert!(
            !constant.is_conditional(),
            "a constant Cov(z₀,z₁|a) must leave the pooled Σ in place"
        );

        // Same scores, no conditioning block at all.
        let (varying, weights, _) = varying_correlation_fixture(n, 0.8);
        let suppressed =
            resolve_score_covariance_field(varying.view(), weights.view(), None).expect("field");
        assert!(
            !suppressed.is_conditional(),
            "with no conditioning span there is nothing to condition Σ on"
        );
        // And the pooled object it keeps is the one the fit would have built.
        let expected = marginal_slope_covariance_from_scores(varying.view(), &weights)
            .expect("pooled Σ")
            .to_dense();
        assert_eq!(suppressed.pooled_covariance().to_dense(), expected);
    }

    /// The #461 absorber seam: with a CTN Stage-1 influence absorber active the
    /// conditional leakage is already absorbed, and replacing z would perturb the
    /// widened-marginal predict seam. The conditional branch must then be
    /// unreachable even on a score that would otherwise fire it.
    #[test]
    fn survival_gate_does_not_condition_behind_an_active_influence_absorber() {
        let n = 4000;
        let (z, weights, design, _) = shifted_fixture(n, 0.6);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            true,
            &design,
        )
        .expect("gate");
        assert!(
            !matches!(
                resolved.per_score[0],
                LatentMeasureCalibration::ConditionalLocationScale(_)
            ),
            "the conditional branch must be suppressed behind an active absorber"
        );
        assert!(
            resolved.conditioning.is_none(),
            "no conditioning block may be built when the branch is suppressed"
        );
    }

    /// Every latent-score column is gated, not just the primary one: with `K > 1`
    /// the leakage is a SUM of per-coordinate conditional shifts, so a gate that
    /// only looked at column 0 would leave the rest of it in `q`.
    #[test]
    fn survival_gate_covers_every_score_column() {
        let n = 4000;
        let m = 0.6;
        let (first, weights, design, zeta) = shifted_fixture(n, m);
        let mut two = Array2::<f64>::zeros((n, 2));
        two.column_mut(0).assign(&first.column(0));
        // A second column shifted the OTHER way, so a gate that reused column
        // 0's calibration would leave a visible residual correlation.
        let residual_sd = (1.0 - m * m).sqrt();
        let x: Vec<f64> = (0..n)
            .map(|row| (first[[row, 0]] - residual_sd * zeta[row]) / m)
            .collect();
        let other = standardized(gaussians(n, 0x2768_C3));
        for row in 0..n {
            two[[row, 1]] = -m * x[row] + residual_sd * other[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &two,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");
        assert_eq!(resolved.per_score.len(), 2);
        for (column, calibration) in resolved.per_score.iter().enumerate() {
            assert!(
                matches!(
                    calibration,
                    LatentMeasureCalibration::ConditionalLocationScale(_)
                ),
                "column {column} must be gated on its own conditional moments; got {}",
                calibration_label(calibration)
            );
        }
        // Both calibrated columns must be conditionally uncorrelated with x.
        for column in 0..2 {
            let calibrated = resolved.calibrated_scores.column(column);
            let cov: f64 = calibrated
                .iter()
                .zip(x.iter())
                .map(|(z, x)| z * x)
                .sum::<f64>()
                / n as f64;
            assert!(
                cov.abs() < 0.03,
                "column {column} still carries a conditional shift: Cov(ζ, x) = {cov:.4}"
            );
        }
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::super::spec::split_persisted_latent_calibrations;
    use crate::bms::{
        LatentMeasureCalibration, LatentZConditionalCalibration, LatentZRankIntCalibration,
    };
    use ndarray::Array2;

    fn conditional() -> LatentMeasureCalibration {
        LatentMeasureCalibration::ConditionalLocationScale(LatentZConditionalCalibration {
            mean_coeffs: vec![0.0, 0.5],
            var_coeffs: Vec::new(),
            basis_ncols: 1,
            var_floor: 1e-8,
            homoskedastic_var: 0.75,
            post_mean: 0.0,
            post_sd: 1.0,
            theta1_cov: Array2::<f64>::zeros((2, 2)),
        })
    }

    fn rank_int() -> LatentMeasureCalibration {
        LatentMeasureCalibration::RankInverseNormal(LatentZRankIntCalibration {
            sorted_z: vec![-1.0, 0.0, 1.0],
            weighted_cdf: vec![0.25, 0.5, 0.75],
            post_mean: 0.0,
            post_sd: 1.0,
        })
    }

    #[test]
    fn an_unfired_gate_persists_nothing() {
        let (rank, cond) =
            split_persisted_latent_calibrations(&[LatentMeasureCalibration::None], true)
                .expect("no calibration is always persistable");
        assert!(rank.is_none() && cond.is_none());
    }

    #[test]
    fn the_two_branches_persist_into_their_own_field() {
        let (rank, cond) =
            split_persisted_latent_calibrations(&[rank_int()], true).expect("rank-INT persists");
        assert!(rank.is_some() && cond.is_none());
        let (rank, cond) = split_persisted_latent_calibrations(&[conditional()], true)
            .expect("conditional persists");
        assert!(rank.is_none() && cond.is_some());
    }

    /// A conditioning span the resolved marginal spec would NOT rebuild is a
    /// refusal, not a warning: the saved model would apply a different latent map
    /// than its coefficients were fitted under, and nothing downstream could tell.
    #[test]
    fn an_unreproducible_conditioning_span_refuses_to_persist() {
        let error = split_persisted_latent_calibrations(&[conditional()], false)
            .expect_err("an unreproducible span must refuse");
        assert!(
            error.contains("RESOLVED marginal spec"),
            "the refusal must name what prediction would rebuild instead; got {error}"
        );
        // A rank-INT calibration does not condition on anything, so the same
        // state must NOT refuse it.
        split_persisted_latent_calibrations(&[rank_int()], false)
            .expect("rank-INT conditions on nothing and is unaffected by the span");
    }

    /// The saved contract holds one score surface. A K>1 fit whose SECOND score
    /// was calibrated cannot be represented, and the loss is named at the point
    /// of loss rather than truncated.
    #[test]
    fn a_calibrated_secondary_score_refuses_to_persist() {
        let error = split_persisted_latent_calibrations(
            &[LatentMeasureCalibration::None, conditional()],
            true,
        )
        .expect_err("a calibrated second score must refuse");
        assert!(
            error.contains("column 1"),
            "the refusal must name the column that cannot be carried; got {error}"
        );
        // An uncalibrated second score carries nothing to lose.
        split_persisted_latent_calibrations(
            &[conditional(), LatentMeasureCalibration::None],
            true,
        )
        .expect("only the persisted surface was calibrated");
    }
}
