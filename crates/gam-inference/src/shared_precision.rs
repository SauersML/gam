//! Cross-fit shared precision of penalized terms (#3523).
//!
//! Several separately fitted models can carry "the same" penalized term, for
//! example one `group(g)` random effect per disease or one `s(age)` per
//! cohort. A shared precision group gives that term ONE precision `τ` across
//! the fits and returns its empirical-Bayes / Fellner–Schall update.
//!
//! # The model
//!
//! Fit `f` penalizes its block `b` with `λ_f · (β−m)ᵀS_b(β−m)`, where `S_b` is
//! the block's penalty in the fit's coefficient frame and `m` its centre (the
//! block's prior mean, or the gauge's affine shift, see
//! [`penalty_posterior_moments`]). The fit's likelihood is scaled by `1/c_f`,
//! where `c_f` is [`UnifiedFitResult::coefficient_covariance_scale`] (`σ̂²` for
//! the profiled Gaussian, `1` for every family whose working weights carry the
//! dispersion), so the Gaussian prior the penalty encodes has precision
//! `λ_f S_b / c_f` and the Laplace posterior is `N(β̂, Vb)` with
//! `Vb = c_f H⁻¹` ([`UnifiedFitResult::beta_covariance`]).
//!
//! The penalty is always on the function, never on the coefficients (SPEC:
//! "Penalties must always be on the final function itself, never on the model
//! coefficients"). Each fit stores `S_b` divided by the Frobenius
//! normalization `ν_f` it was built with (`ActivePenaltyInfo::normalization_scale`),
//! so `ν_f S_b` is the physical function-space penalty `J_b(f) = ν_f βᵀS_bβ`.
//! The shared precision is a precision of that functional, which is the same
//! object in every fit whatever basis or normalization each one used:
//!
//! ```text
//!     β_b,f | τ ~ N(m_f, c_f (τ ν_f S_b,f)⁺),    τ ~ Gamma(a, b).
//! ```
//!
//! # The update
//!
//! The E-step takes the expectation of the physical penalty under each fit's
//! posterior,
//!
//! ```text
//!     E_f = ν_f · ( (β̂−m)ᵀS_b(β̂−m) + tr(S_b Vb_bb) ) / c_f,
//! ```
//!
//! and the M-step maximizes `Σ_f [ ½ rank(S_b,f) log τ − ½ τ E_f ] +
//! (a−1) log τ − b τ`:
//!
//! ```text
//!     τ = ( Σ_f rank(S_b,f) + 2(a−1) ) / ( Σ_f E_f + 2b ).
//! ```
//!
//! `rank(S_b)` is the number of penalized directions, the degrees of freedom the
//! Gaussian prior has. It is not the block's coefficient count, which also
//! counts the unpenalized null space. `‖β‖² + tr(Σ)` is the special case
//! `S_b = I` with `c_f = ν_f = 1`. Fit `f`'s own smoothing parameter under the
//! shared precision is `λ_f = τ ν_f` (`implied_lambda`, on the same scale as
//! the fit's `fitted_lambda`). Iterating the update with each fit refitted at
//! `implied_lambda` is the EM/Fellner–Schall iteration for the shared precision.

use crate::model::FittedModel;
use gam_models::inference::saved_summary::saved_predictor_designs;
use gam_models::survival::predict::saved_fit_result;
use gam_solve::estimate::UnifiedFitResult;
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One shared precision group: the term each fit contributes, and the
/// `Gamma(shape, rate)` hyperprior on the shared precision.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedPrecisionGroup {
    pub name: String,
    pub shape: f64,
    pub rate: f64,
    /// The penalized term's name in each fit, parallel to the fits. A fit
    /// without that term does not contribute to the group.
    pub labels: Vec<String>,
    /// Which of the term's penalty blocks is shared, as its 0-based position
    /// among the term's blocks. A term with several blocks (a double-penalty
    /// smooth, a tensor product) must name one: each block is its own prior
    /// with its own precision.
    #[serde(default)]
    pub penalty: Option<usize>,
}

/// The posterior moments of one penalty block of a saved fit.
#[derive(Clone, Debug, Serialize)]
pub struct PenaltyPosteriorMoments {
    /// The term that owns the block.
    pub label: Option<String>,
    /// The block's predictor in a multi-predictor fit.
    pub predictor: Option<&'static str>,
    /// The block's index in the fit's flat smoothing-parameter layout.
    pub penalty_index: usize,
    pub penalty_source: String,
    /// The global coefficients `S_b` covers.
    pub coefficient_indices: Vec<usize>,
    /// `rank(S_b)`.
    pub rank: usize,
    /// `ν`: the stored `S_b` is the physical penalty divided by it.
    pub normalization_scale: f64,
    /// The fit's smoothing parameter for this block.
    pub fitted_lambda: f64,
    /// `c`, with `Vb = c H⁻¹`.
    pub coefficient_covariance_scale: f64,
    /// `(β̂−m)ᵀS_b(β̂−m)`.
    pub quadratic_form: f64,
    /// `tr(S_b Vb_bb)`.
    pub trace_penalty_covariance: f64,
}

impl PenaltyPosteriorMoments {
    /// `E = ν ((β̂−m)ᵀS_b(β̂−m) + tr(S_b Vb_bb)) / c`: the posterior expectation
    /// of the physical penalty in units of the fit's likelihood scale.
    pub fn physical_penalty_expectation(&self) -> f64 {
        self.normalization_scale * (self.quadratic_form + self.trace_penalty_covariance)
            / self.coefficient_covariance_scale
    }
}

/// `((β̂−m)ᵀS(β̂−m), tr(S Σ))` for one block.
fn penalty_moments(
    penalty: ArrayView2<'_, f64>,
    centre: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
) -> (f64, f64) {
    let quadratic = centre.dot(&penalty.dot(&centre));
    // tr(S Σ) = Σ_ij S_ij Σ_ji.
    let trace = (&penalty * &covariance.t()).sum();
    (quadratic, trace)
}

/// The centre `m` the fit's penalty on `columns` is measured from, returned as
/// `β̂ − m`.
///
/// A block is centred at its prior mean `μ`, and a coefficient gauge
/// `β = Tθ + a` centres its active penalty `θᵀS_θθ` at the shift `a` in the
/// saved frame ([`UnifiedFitResult::beta_from_gauge_shift`]). One of the two
/// is always zero on a block in the fits the engine builds. When both move the
/// same block the centre of the saved penalty is not determined by the saved
/// state, and the block is refused rather than measured from either.
fn centred_block(
    fit: &UnifiedFitResult,
    gauge_centred: &Array1<f64>,
    columns: std::ops::Range<usize>,
    prior_mean: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, String> {
    let raw = fit.beta.slice(s![columns.clone()]);
    let shifted = gauge_centred.slice(s![columns]);
    let gauge_moves_block = raw.iter().zip(shifted.iter()).any(|(r, g)| r != g);
    let prior_moves_block = prior_mean.iter().any(|&value| value != 0.0);
    match (gauge_moves_block, prior_moves_block) {
        (false, _) => Ok(&raw - prior_mean),
        (true, false) => Ok(shifted.to_owned()),
        (true, true) => Err(format!(
            "{context}: both the block's prior mean and the coefficient gauge's affine shift \
             are nonzero, so the saved fit does not determine the penalty's centre"
        )),
    }
}

/// The posterior moments of every penalty block of a saved fit, in the fit's
/// flat penalty order.
///
/// The penalty blocks, their ranks, normalizations and prior means come from
/// the frozen-basis replay the summary tables use
/// ([`saved_predictor_designs`]), checked against the fit's coefficient and
/// smoothing-parameter layout. `β̂`, `Vb` and `c` come from the saved fit.
pub fn penalty_posterior_moments(
    model: &FittedModel,
) -> Result<Vec<PenaltyPosteriorMoments>, String> {
    if let Some((feature_column, _)) = model.saved_spline_scan().map_err(|err| err.to_string())? {
        return Err(format!(
            "the model is an O(n) spline-scan fit of s({feature_column}), which keeps no \
             coefficient covariance, so its penalty posterior moments are not available"
        ));
    }
    let fit = saved_fit_result(model)?;
    let p = fit.beta.len();
    let covariance = fit.beta_covariance().ok_or_else(|| {
        "the saved fit carries no conditional coefficient covariance Vb; refit".to_string()
    })?;
    if covariance.dim() != (p, p) {
        return Err(format!(
            "the saved coefficient covariance is {}x{} but the fit has {p} coefficients",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    let scale = fit
        .coefficient_covariance_scale()
        .map_err(|err| format!("the fit's coefficient covariance scale is unavailable: {err}"))?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "the fit's coefficient covariance scale must be finite and positive, got {scale}"
        ));
    }
    let gauge_centred = fit.beta_from_gauge_shift()?;
    let mut moments = Vec::new();
    for predictor in saved_predictor_designs(model, fit, "penalty posterior moments")? {
        let design = &predictor.design;
        if design.penaltyinfo.len() != design.penalties.len() {
            return Err(format!(
                "the replayed design has {} penalty blocks but {} penalty records",
                design.penalties.len(),
                design.penaltyinfo.len()
            ));
        }
        for (position, (penalty, info)) in
            design.penalties.iter().zip(&design.penaltyinfo).enumerate()
        {
            let penalty_index = predictor.offset.penalties + position;
            let context = format!(
                "penalty block {penalty_index} ({:?} of term {:?})",
                info.penalty.source, info.termname
            );
            let fitted_lambda = *fit.lambdas.get(penalty_index).ok_or_else(|| {
                format!(
                    "{context}: the fit has only {} smoothing parameters",
                    fit.lambdas.len()
                )
            })?;
            let columns = (predictor.offset.coefficients + penalty.col_range.start)
                ..(predictor.offset.coefficients + penalty.col_range.end);
            let width = columns.len();
            if columns.end > p || penalty.local.dim() != (width, width) {
                return Err(format!(
                    "{context}: its {}x{} penalty on coefficients {columns:?} does not fit the \
                     fit's {p} coefficients",
                    penalty.local.nrows(),
                    penalty.local.ncols()
                ));
            }
            let prior_mean = penalty
                .prior_mean
                .evaluate(width, &context)
                .map_err(|err| err.to_string())?;
            let centre =
                centred_block(fit, &gauge_centred, columns.clone(), &prior_mean, &context)?;
            let (quadratic_form, trace_penalty_covariance) = penalty_moments(
                penalty.local.view(),
                centre.view(),
                covariance.slice(s![columns.clone(), columns.clone()]),
            );
            let normalization_scale = info.penalty.normalization_scale;
            if !(quadratic_form.is_finite()
                && trace_penalty_covariance.is_finite()
                && normalization_scale.is_finite()
                && normalization_scale > 0.0)
            {
                return Err(format!(
                    "{context}: non-finite moments (quadratic form {quadratic_form}, trace \
                     {trace_penalty_covariance}, normalization {normalization_scale})"
                ));
            }
            moments.push(PenaltyPosteriorMoments {
                label: info.termname.clone(),
                predictor: predictor.predictor,
                penalty_index,
                penalty_source: format!("{:?}", info.penalty.source),
                coefficient_indices: columns.collect(),
                rank: info.penalty.effective_rank,
                normalization_scale,
                fitted_lambda,
                coefficient_covariance_scale: scale,
                quadratic_form,
                trace_penalty_covariance,
            });
        }
    }
    Ok(moments)
}

/// One fit's contribution to a shared precision update.
#[derive(Clone, Debug, Serialize)]
pub struct SharedPrecisionContribution<K> {
    pub model: K,
    pub label: String,
    pub predictor: Option<&'static str>,
    pub penalty_index: usize,
    pub penalty_source: String,
    pub coefficient_indices: Vec<usize>,
    pub rank: usize,
    pub normalization_scale: f64,
    pub fitted_lambda: f64,
    /// `τ ν_f`: this fit's smoothing parameter under the shared precision.
    pub implied_lambda: f64,
    pub coefficient_covariance_scale: f64,
    pub quadratic_form: f64,
    pub trace_penalty_covariance: f64,
    /// `E_f`, see [`PenaltyPosteriorMoments::physical_penalty_expectation`].
    pub quadratic_contribution: f64,
}

/// A shared precision group's update.
#[derive(Clone, Debug, Serialize)]
pub struct SharedPrecisionUpdate<K> {
    /// `τ = numerator / denominator`.
    pub lambda: f64,
    pub log_lambda: f64,
    pub shape: f64,
    pub rate: f64,
    pub n_fits: usize,
    /// The common `rank(S_b)` of the matched blocks.
    pub dimension: usize,
    /// `Σ_f E_f`.
    pub quadratic_sum: f64,
    /// `n_fits · rank(S_b) + 2(a−1)`.
    pub numerator: f64,
    /// `Σ_f E_f + 2b`.
    pub denominator: f64,
    pub fits: Vec<SharedPrecisionContribution<K>>,
}

fn validate_groups(groups: &[SharedPrecisionGroup], n_fits: usize) -> Result<(), String> {
    if groups.is_empty() {
        return Err("at least one shared precision group is required".to_string());
    }
    let mut seen = BTreeSet::new();
    let duplicates = groups
        .iter()
        .filter(|group| !seen.insert(group.name.as_str()))
        .map(|group| group.name.as_str())
        .collect::<BTreeSet<_>>();
    if !duplicates.is_empty() {
        return Err(format!(
            "duplicate shared precision group name(s): {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    for group in groups {
        if group.labels.len() != n_fits {
            return Err(format!(
                "shared precision group {:?} has {} labels for {n_fits} model(s)",
                group.name,
                group.labels.len()
            ));
        }
        if !(group.shape.is_finite() && group.shape > 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite shape > 0",
                group.name
            ));
        }
        if !(group.rate.is_finite() && group.rate >= 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite rate >= 0",
                group.name
            ));
        }
    }
    Ok(())
}

/// The shared precision update of every group across `fits`, keyed by group
/// name. See the module docs for the model and the update.
pub fn shared_precision_updates<K: Clone + std::fmt::Display>(
    fits: &[(K, &FittedModel)],
    groups: &[SharedPrecisionGroup],
) -> Result<BTreeMap<String, SharedPrecisionUpdate<K>>, String> {
    if fits.is_empty() {
        return Err("at least one model is required".to_string());
    }
    validate_groups(groups, fits.len())?;
    let moments = fits
        .iter()
        .map(|(key, model)| {
            penalty_posterior_moments(model).map_err(|err| format!("model {key}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut updates = BTreeMap::new();
    for group in groups {
        let mut contributions = Vec::new();
        let mut ranks = BTreeSet::new();
        let mut quadratic_sum = 0.0;
        for (((key, _), fit_moments), label) in fits.iter().zip(&moments).zip(&group.labels) {
            let matched = fit_moments
                .iter()
                .filter(|block| block.label.as_deref() == Some(label.as_str()))
                .collect::<Vec<_>>();
            if matched.is_empty() {
                continue;
            }
            let listing = || {
                matched
                    .iter()
                    .enumerate()
                    .map(|(position, block)| format!("{position}: {}", block.penalty_source))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let block = match group.penalty {
                None if matched.len() == 1 => matched[0],
                None => {
                    return Err(format!(
                        "shared precision group {:?}: term {label:?} of model {key} has {} \
                         penalty blocks ({}), each its own prior; name the shared one with \
                         `penalty`",
                        group.name,
                        matched.len(),
                        listing()
                    ));
                }
                Some(position) => *matched.get(position).ok_or_else(|| {
                    format!(
                        "shared precision group {:?} names penalty {position} but term \
                         {label:?} of model {key} has {} penalty blocks ({})",
                        group.name,
                        matched.len(),
                        listing()
                    )
                })?,
            };
            let quadratic_contribution = block.physical_penalty_expectation();
            if !quadratic_contribution.is_finite() {
                return Err(format!(
                    "shared precision group {:?} has non-finite contribution in model {key}",
                    group.name
                ));
            }
            quadratic_sum += quadratic_contribution;
            ranks.insert(block.rank);
            contributions.push(SharedPrecisionContribution {
                model: key.clone(),
                label: label.clone(),
                predictor: block.predictor,
                penalty_index: block.penalty_index,
                penalty_source: block.penalty_source.clone(),
                coefficient_indices: block.coefficient_indices.clone(),
                rank: block.rank,
                normalization_scale: block.normalization_scale,
                fitted_lambda: block.fitted_lambda,
                implied_lambda: f64::NAN,
                coefficient_covariance_scale: block.coefficient_covariance_scale,
                quadratic_form: block.quadratic_form,
                trace_penalty_covariance: block.trace_penalty_covariance,
                quadratic_contribution,
            });
        }
        if contributions.is_empty() {
            return Err(format!(
                "shared precision group {:?} did not match any model coefficients",
                group.name
            ));
        }
        let dimension = match ranks.iter().copied().collect::<Vec<_>>().as_slice() {
            [rank] => *rank,
            ranks => {
                return Err(format!(
                    "shared precision group {:?} matched penalty blocks of inconsistent \
                     dimensions (penalty ranks): {ranks:?}",
                    group.name
                ));
            }
        };
        let numerator = contributions.len() as f64 * dimension as f64 + 2.0 * (group.shape - 1.0);
        let denominator = quadratic_sum + 2.0 * group.rate;
        if !(numerator > 0.0) {
            return Err(format!(
                "shared precision group {:?} has non-positive MAP numerator",
                group.name
            ));
        }
        if !(denominator > 0.0 && denominator.is_finite()) {
            return Err(format!(
                "shared precision group {:?} has non-positive/non-finite denominator",
                group.name
            ));
        }
        let lambda = numerator / denominator;
        for contribution in &mut contributions {
            contribution.implied_lambda = lambda * contribution.normalization_scale;
        }
        updates.insert(
            group.name.clone(),
            SharedPrecisionUpdate {
                lambda,
                log_lambda: lambda.ln(),
                shape: group.shape,
                rate: group.rate,
                n_fits: contributions.len(),
                dimension,
                quadratic_sum,
                numerator,
                denominator,
                fits: contributions,
            },
        );
    }
    Ok(updates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
    use gam_models::fit_orchestration::FitConfig;
    use gam_models::inference::model_payload_builders::fit_formula_to_payload;
    use ndarray::Array2;

    const LEVELS: [&str; 3] = ["a", "b", "c"];

    /// `x`, `z`, the factor `g` and `y = 1 + 0.05x + sin(2πz) + effect·scale +
    /// small deterministic noise`, with `reps` rows per level of `levels`.
    fn table(scale: f64, reps: usize, levels: &[&str]) -> EncodedDataset {
        let effects = [-0.6, 0.15, 0.7];
        let n = reps * levels.len();
        let mut x = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        for rep in 0..reps {
            for (pos, level) in levels.iter().enumerate() {
                let row = rep * levels.len() + pos;
                let xi = ((row * 7) % 11) as f64 - 5.0;
                let zi = row as f64 / (n - 1) as f64;
                let noise = 0.03 * (((row * 13) % 7) as f64 - 3.0);
                x.push(xi);
                z.push(zi);
                g.push(Some(*level));
                y.push(
                    1.0 + 0.05 * xi
                        + (2.0 * std::f64::consts::PI * zi).sin()
                        + effects[pos] * scale
                        + noise,
                );
            }
        }
        let (g_column, g_codes) =
            gam_data::encode_optional_categorical_column("g", &g).expect("encode g");
        let continuous = |name: &str| SchemaColumn {
            name: name.to_string(),
            kind: ColumnKindTag::Continuous,
            levels: vec![],
        };
        let columns = vec![continuous("x"), continuous("z"), g_column, continuous("y")];
        let values = Array2::from_shape_fn((n, 4), |(row, column)| match column {
            0 => x[row],
            1 => z[row],
            2 => g_codes[row],
            _ => y[row],
        });
        EncodedDataset {
            headers: columns.iter().map(|column| column.name.clone()).collect(),
            column_kinds: columns.iter().map(|column| column.kind).collect(),
            values,
            schema: DataSchema { columns },
        }
    }

    fn fit(formula: &str, data: &EncodedDataset) -> FittedModel {
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let payload = fit_formula_to_payload(formula.to_string(), data, &config)
            .unwrap_or_else(|err| panic!("fit {formula}: {err}"));
        FittedModel::from_payload(payload)
    }

    fn group(name: &str, shape: f64, rate: f64, n_fits: usize) -> SharedPrecisionGroup {
        SharedPrecisionGroup {
            name: name.to_string(),
            shape,
            rate,
            labels: vec![name.to_string(); n_fits],
            penalty: None,
        }
    }

    /// The update against a direct computation. `group(g)` is a ridge on the
    /// level indicators, `S_g = I` with `ν = 1` and `rank 3`, on the columns
    /// after the intercept and `x`, so directly
    /// `E_f = (‖β̂_g‖² + tr(Vb_gg)) / c_f` and
    /// `τ = (2·3 + 2(a−1)) / (E_1 + E_2 + 2b)`.
    #[test]
    fn shared_precision_update_matches_direct_computation_3523() {
        let left = fit("y ~ x + group(g)", &table(1.0, 12, &LEVELS));
        let right = fit("y ~ x + group(g)", &table(0.35, 12, &LEVELS));
        let (shape, rate) = (2.0, 0.5);
        let groups = [group("g", shape, rate, 2)];
        let updates =
            shared_precision_updates(&[("left", &left), ("right", &right)], &groups).unwrap();
        let update = &updates["g"];
        let g_columns = 2..5;

        let mut direct_sum = 0.0;
        for (model, contribution) in [&left, &right].into_iter().zip(&update.fits) {
            let fit = saved_fit_result(model).unwrap();
            // The direct formula has no centre: the block must be measured from 0.
            let gauge_centred = fit.beta_from_gauge_shift().unwrap();
            assert_eq!(
                gauge_centred.slice(s![g_columns.clone()]),
                fit.beta.slice(s![g_columns.clone()])
            );
            let c = fit.coefficient_covariance_scale().unwrap();
            // A profiled Gaussian: c is σ̂², which the unmodelled sin(2πz) holds
            // near its variance 1/2, so a formula that drops c misses by far
            // more than the tolerance below.
            assert!(
                (c - 1.0).abs() > 0.25,
                "σ̂² = {c} does not separate c from 1"
            );
            let vb = fit.beta_covariance().unwrap();
            let beta_sq: f64 = g_columns.clone().map(|j| fit.beta[j] * fit.beta[j]).sum();
            let trace: f64 = g_columns.clone().map(|j| vb[[j, j]]).sum();
            let direct = (beta_sq + trace) / c;
            direct_sum += direct;

            assert_eq!(
                contribution.coefficient_indices,
                g_columns.clone().collect::<Vec<_>>()
            );
            assert_eq!(contribution.rank, 3);
            assert_eq!(contribution.normalization_scale, 1.0);
            // Each positive sum below has at most 12 terms, so each side is within
            // γ_12 ≈ 12ε of the exact value, and the two agree to 32ε.
            let tolerance = 32.0 * f64::EPSILON;
            assert!(
                (contribution.quadratic_contribution - direct).abs() <= tolerance * direct,
                "E_f = {} but directly {direct}",
                contribution.quadratic_contribution
            );

            // The penalty the fit was actually solved with: on the g block
            // `H − X'WX = λ_g S_g = λ_g I`. `fitted_lambda` is read by the
            // penalty's position in the fit's layout, so this pins that lookup.
            // Off the diagonal both Hessians hold exact zeros (disjoint level
            // indicators). On it the Gram entry is a sum of n nonnegative terms
            // and H adds λ, so each is within nε of exact.
            let hessian = fit.saved_frame_penalized_hessian().unwrap().unwrap();
            let gram = fit.saved_frame_weighted_gram().unwrap().unwrap();
            let n = 36.0;
            for i in g_columns.clone() {
                for j in g_columns.clone() {
                    let penalty = hessian[[i, j]] - gram[[i, j]];
                    let expected = if i == j {
                        contribution.fitted_lambda
                    } else {
                        0.0
                    };
                    let bound = (2.0 * n + 3.0)
                        * f64::EPSILON
                        * (hessian[[i, j]].abs() + gram[[i, j]].abs());
                    assert!(
                        (penalty - expected).abs() <= bound,
                        "(H - X'WX)[{i},{j}] = {penalty}, expected {expected}"
                    );
                }
            }
        }
        let numerator = 2.0 * 3.0 + 2.0 * (shape - 1.0);
        let direct_lambda = numerator / (direct_sum + 2.0 * rate);
        assert_eq!(update.n_fits, 2);
        assert_eq!(update.dimension, 3);
        assert_eq!(update.numerator, numerator);
        let tolerance = 64.0 * f64::EPSILON;
        assert!(
            (update.lambda - direct_lambda).abs() <= tolerance * direct_lambda,
            "τ = {} but directly {direct_lambda}",
            update.lambda
        );
        for contribution in &update.fits {
            assert_eq!(contribution.implied_lambda, update.lambda);
        }
        assert_eq!(
            update.quadratic_sum,
            update
                .fits
                .iter()
                .map(|fit| fit.quadratic_contribution)
                .sum::<f64>()
        );
    }

    /// Absent terms are skipped, a factor with other levels has another rank,
    /// and a term's source name is not its label.
    #[test]
    fn shared_precision_groups_skip_absent_terms_and_refuse_rank_mismatch_3523() {
        let three = fit("y ~ x + group(g)", &table(1.0, 12, &LEVELS));
        let without = fit("y ~ x", &table(1.0, 12, &LEVELS));
        let updates =
            shared_precision_updates(&[(0, &three), (1, &without)], &[group("g", 1.5, 0.25, 2)])
                .unwrap();
        assert_eq!(updates["g"].n_fits, 1);
        assert_eq!(updates["g"].fits[0].model, 0);

        let two = fit("y ~ x + group(g)", &table(1.0, 12, &LEVELS[..2]));
        let err = shared_precision_updates(&[(0, &three), (1, &two)], &[group("g", 1.0, 0.0, 2)])
            .unwrap_err();
        assert!(err.contains("inconsistent dimensions"), "{err}");

        let err =
            shared_precision_updates(&[(0, &three)], &[group("group", 1.0, 0.0, 1)]).unwrap_err();
        assert!(
            err.contains("did not match any model coefficients"),
            "{err}"
        );
    }

    /// A term with several penalty blocks is several priors: the group must
    /// name one, and the one it names is the one used.
    #[test]
    fn shared_precision_group_names_one_block_of_a_multi_penalty_term_3523() {
        let model = fit(
            "y ~ x + s(z, double_penalty=true) + group(g)",
            &table(1.0, 20, &LEVELS),
        );
        let moments = penalty_posterior_moments(&model).unwrap();
        let smooth = moments
            .iter()
            .find(|block| block.penalty_source.contains("DoublePenaltyNullspace"))
            .and_then(|block| block.label.clone())
            .expect("the double-penalty smooth has a null-space block");
        let blocks = moments
            .iter()
            .filter(|block| block.label.as_deref() == Some(smooth.as_str()))
            .collect::<Vec<_>>();
        assert!(blocks.len() >= 2, "{smooth} has {} blocks", blocks.len());

        let mut shared = group(&smooth, 1.0, 0.0, 1);
        let err = shared_precision_updates(&[(0, &model)], &[shared.clone()]).unwrap_err();
        assert!(err.contains("name the shared one with `penalty`"), "{err}");

        shared.penalty = Some(1);
        let updates = shared_precision_updates(&[(0, &model)], &[shared]).unwrap();
        let update = &updates[&smooth];
        let chosen = &update.fits[0];
        assert_eq!(chosen.penalty_index, blocks[1].penalty_index);
        assert_eq!(update.dimension, blocks[1].rank);
        assert_eq!(
            chosen.quadratic_contribution,
            blocks[1].physical_penalty_expectation()
        );
        // Neither block penalizes every coefficient it covers: `rank(S_b)`,
        // not the block's width, is the degrees of freedom of its prior.
        for block in &blocks {
            assert!(
                block.rank < block.coefficient_indices.len(),
                "{} has rank {} on {} coefficients",
                block.penalty_source,
                block.rank,
                block.coefficient_indices.len()
            );
        }
    }

    #[test]
    fn shared_precision_groups_validate_their_request_3523() {
        let model = fit("y ~ x + group(g)", &table(1.0, 12, &LEVELS));
        let fits = [(0, &model)];
        let refused =
            |groups: &[SharedPrecisionGroup]| shared_precision_updates(&fits, groups).unwrap_err();
        assert!(refused(&[]).contains("at least one shared precision group"));
        assert!(
            refused(&[group("g", 1.0, 0.0, 1), group("g", 2.0, 0.0, 1)])
                .contains("duplicate shared precision group name")
        );
        assert!(refused(&[group("g", 0.0, 1.0, 1)]).contains("shape > 0"));
        assert!(refused(&[group("g", 1.0, -0.1, 1)]).contains("rate >= 0"));
        assert!(refused(&[group("g", 1.0, 0.0, 2)]).contains("2 labels for 1 model"));
        let no_fits: [(usize, &FittedModel); 0] = [];
        assert!(
            shared_precision_updates(&no_fits, &[group("g", 1.0, 0.0, 0)])
                .unwrap_err()
                .contains("at least one model")
        );
    }
}
