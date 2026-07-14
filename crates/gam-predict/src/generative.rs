//! Canonical observation-replicate capability for persisted models.
//!
//! Saved-model consumers must not rediscover a family allowlist at their API
//! boundary.  This module asks the persisted model for its actual prediction
//! class, reconstructs the fitted response distribution through the canonical
//! predictor/noise channels, and returns a typed unsupported error only for a
//! model class that genuinely has no scalar-row observation sampler.

use std::collections::HashMap;

use gam_models::inference::generative::{
    GenerativeSpec, NoiseModel, family_noise_parameter, generativespec_from_predict,
};
use gam_models::inference::model::{FittedEstimator, FittedFamily, FittedModel, PredictModelClass};
use gam_models::inference::predict_io::PredictResult;
use gam_models::survival::predict::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_competing_risks_survival, predict_latent_window_survival, predict_survival,
    resolve_saved_survival_time_columns,
};
use ndarray::{Array1, Array2, ArrayView2, s};

use crate::FittedModelPredictExt;
use crate::input::{build_predict_input_for_model, build_transformation_normal_quantile_grid};

/// Borrowed row inputs for saved-model observation generation.
pub struct SavedGenerativeInput<'a> {
    pub data: ArrayView2<'a, f64>,
    pub col_map: &'a HashMap<String, usize>,
    pub training_headers: Option<&'a Vec<String>>,
    pub offset: &'a Array1<f64>,
    pub offset_noise: &'a Array1<f64>,
    pub noise_offset_supplied: bool,
    /// `None` for an unweighted saved model. A weighted saved model requires
    /// the exact requested-row values from its persisted weight column.
    pub prior_weights: Option<&'a Array1<f64>>,
}

/// Typed failures from [`generative_spec_for_saved_model`].
#[derive(Debug)]
pub enum SavedGenerativeError {
    /// The persisted response family has no scalar-row observation sampler.
    UnsupportedSampler {
        model_class: PredictModelClass,
        family: String,
    },
    /// Persistence omitted state that the fitted generative law requires.
    MissingSavedState {
        model_class: PredictModelClass,
        reason: String,
    },
    /// New rows do not provide a value required by the fitted observation law.
    MissingRequiredInput { reason: String },
    /// Persisted state and the requested rows disagree structurally.
    InvalidState { reason: String },
    /// The canonical predictor or noise capability rejected the requested rows.
    Evaluation { reason: String },
}

impl std::fmt::Display for SavedGenerativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSampler {
                model_class,
                family,
            } => write!(
                f,
                "{} family {family:?} defines no observation-replicate sampler",
                model_class.name(),
            ),
            Self::MissingSavedState {
                model_class,
                reason,
            } => write!(
                f,
                "{} observation-replicate state is missing: {reason}",
                model_class.name(),
            ),
            Self::MissingRequiredInput { reason } => {
                write!(f, "observation-replicate input is missing: {reason}")
            }
            Self::InvalidState { reason } => {
                write!(f, "invalid saved observation-replicate state: {reason}")
            }
            Self::Evaluation { reason } => {
                write!(f, "observation-replicate evaluation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for SavedGenerativeError {}

fn validated_prior_weights<'a>(
    model: &FittedModel,
    n_rows: usize,
    prior_weights: Option<&'a Array1<f64>>,
) -> Result<Option<&'a Array1<f64>>, SavedGenerativeError> {
    match (model.payload().weight_column.as_deref(), prior_weights) {
        (Some(column), None) => Err(SavedGenerativeError::MissingRequiredInput {
            reason: format!(
                "saved weighted model requires row-weight column {column:?}; unit-weight \
                 substitution would change its fitted observation law"
            ),
        }),
        (None, Some(_)) => Err(SavedGenerativeError::InvalidState {
            reason: "row weights were supplied for a model whose saved fit is unweighted"
                .to_string(),
        }),
        (_, Some(weights)) if weights.len() != n_rows => Err(SavedGenerativeError::InvalidState {
            reason: format!(
                "row-weight length {} does not match prediction row count {n_rows}",
                weights.len(),
            ),
        }),
        (_, weights) => Ok(weights),
    }
}

fn spline_scan_generative_spec(
    model: &FittedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    prior_weights: Option<&Array1<f64>>,
) -> Result<Option<GenerativeSpec>, SavedGenerativeError> {
    let Some((feature_column, fit)) =
        model
            .saved_spline_scan()
            .map_err(|error| SavedGenerativeError::InvalidState {
                reason: error.to_string(),
            })?
    else {
        return Ok(None);
    };
    let feature_index =
        *col_map
            .get(feature_column)
            .ok_or_else(|| SavedGenerativeError::MissingRequiredInput {
                reason: format!("spline-scan feature column {feature_column:?}"),
            })?;
    if feature_index >= data.ncols() {
        return Err(SavedGenerativeError::InvalidState {
            reason: format!(
                "spline-scan feature column {feature_column:?} resolves to {feature_index}, \
                 outside the {}-column prediction table",
                data.ncols(),
            ),
        });
    }
    let weights = validated_prior_weights(model, data.nrows(), prior_weights)?;
    let mut mean = Array1::<f64>::zeros(data.nrows());
    for (row, &x) in data.column(feature_index).iter().enumerate() {
        mean[row] = fit
            .predict(x)
            .map_err(|reason| SavedGenerativeError::Evaluation {
                reason: format!("spline-scan predictor row {row}: {reason}"),
            })?
            .0;
    }
    let prediction = PredictResult {
        eta: mean.clone(),
        mean,
    };
    generativespec_from_predict(
        prediction,
        model.likelihood(),
        Some(fit.sigma2.sqrt()),
        weights,
    )
    .map(Some)
    .map_err(|error| SavedGenerativeError::InvalidState {
        reason: format!("spline-scan observation law: {error}"),
    })
}

fn plugin_survival_request<'a>(
    model: &'a FittedModel,
    data: ArrayView2<'a, f64>,
    col_map: &'a HashMap<String, usize>,
    training_headers: Option<&'a Vec<String>>,
    offset: &'a Array1<f64>,
    offset_noise: &'a Array1<f64>,
    time_grid: Option<&'a [f64]>,
) -> SurvivalPredictRequest<'a> {
    SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset: offset,
        noise_offset: offset_noise,
        time_grid,
        with_uncertainty: false,
        estimand: SurvivalPredictEstimand::Plugin,
    }
}

fn survival_entry_times(
    model: &FittedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Array1<f64>, SavedGenerativeError> {
    let columns = resolve_saved_survival_time_columns(model, col_map).map_err(|reason| {
        SavedGenerativeError::MissingRequiredInput {
            reason: format!("survival window columns: {reason}"),
        }
    })?;
    let mut entry = Array1::<f64>::zeros(data.nrows());
    for row in 0..data.nrows() {
        let value = columns.row_entry_time(data, row);
        if !(value.is_finite() && value >= 0.0) {
            return Err(SavedGenerativeError::MissingRequiredInput {
                reason: format!(
                    "survival entry time at row {row} must be finite and non-negative, got {value}"
                ),
            });
        }
        entry[row] = value;
    }
    Ok(entry)
}

fn entry_surface_chunk_rows(total_rows: usize, live_surfaces_per_cell: usize) -> usize {
    if total_rows == 0 {
        return 1;
    }
    let target_bytes =
        gam_runtime::resource::ResourcePolicy::default_library().row_chunk_target_bytes;
    let bytes_per_cell = std::mem::size_of::<f64>().saturating_mul(live_surfaces_per_cell.max(1));
    let max_cells = target_bytes / bytes_per_cell;
    max_cells.isqrt().max(1).min(total_rows)
}

fn conditional_window_survival(
    survival_entry: f64,
    survival_exit: f64,
    row: usize,
) -> Result<f64, SavedGenerativeError> {
    if !(survival_entry.is_finite()
        && survival_exit.is_finite()
        && survival_entry > 0.0
        && survival_exit >= 0.0)
    {
        return Err(SavedGenerativeError::Evaluation {
            reason: format!(
                "survival window row {row} produced invalid endpoint probabilities: entry={survival_entry}, exit={survival_exit}"
            ),
        });
    }
    let tolerance = 128.0 * f64::EPSILON * survival_entry.max(1.0);
    if survival_exit > survival_entry + tolerance {
        return Err(SavedGenerativeError::Evaluation {
            reason: format!(
                "survival window row {row} is non-monotone: S(exit)={survival_exit} exceeds S(entry)={survival_entry}"
            ),
        });
    }
    Ok((survival_exit / survival_entry).min(1.0))
}

fn single_cause_window_spec(
    model: &FittedModel,
    request: &SavedGenerativeInput<'_>,
) -> Result<GenerativeSpec, SavedGenerativeError> {
    let exit = predict_survival(
        plugin_survival_request(
            model,
            request.data.view(),
            request.col_map,
            request.training_headers,
            request.offset,
            request.offset_noise,
            None,
        ),
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .map_err(|error| SavedGenerativeError::Evaluation {
        reason: format!("saved survival exit probability: {error}"),
    })?;
    let n = request.data.nrows();
    if exit.survival.shape() != [n, 1] {
        return Err(SavedGenerativeError::InvalidState {
            reason: format!(
                "saved survival exit surface has shape {:?}, expected ({n}, 1)",
                exit.survival.shape()
            ),
        });
    }
    let entry_times = survival_entry_times(model, request.data.view(), request.col_map)?;
    let mut entry_survival = Array1::<f64>::zeros(n);
    // `predict_survival` owns hazard, survival, and cumulative-hazard squares
    // simultaneously. Budget the full live return, not only the one surface we
    // retain the diagonal from.
    let chunk_rows = entry_surface_chunk_rows(n, 3);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let grid = entry_times.slice(s![start..end]).to_vec();
        let offset = request.offset.slice(s![start..end]).to_owned();
        let offset_noise = request.offset_noise.slice(s![start..end]).to_owned();
        let chunk = predict_survival(
            plugin_survival_request(
                model,
                request.data.slice(s![start..end, ..]),
                request.col_map,
                request.training_headers,
                &offset,
                &offset_noise,
                Some(&grid),
            ),
            SurvivalPredictionCovarianceMode::Conditional,
        )
        .map_err(|error| SavedGenerativeError::Evaluation {
            reason: format!("saved survival entry probability rows {start}..{end}: {error}"),
        })?;
        if chunk.survival.shape() != [end - start, end - start] {
            return Err(SavedGenerativeError::InvalidState {
                reason: format!(
                    "saved survival entry surface rows {start}..{end} has shape {:?}, expected ({}, {})",
                    chunk.survival.shape(),
                    end - start,
                    end - start
                ),
            });
        }
        for local_row in 0..end - start {
            entry_survival[start + local_row] = chunk.survival[[local_row, local_row]];
        }
    }
    let mut event_probability = Array1::<f64>::zeros(n);
    for row in 0..n {
        event_probability[row] =
            1.0 - conditional_window_survival(entry_survival[row], exit.survival[[row, 0]], row)?;
    }
    Ok(GenerativeSpec {
        mean: event_probability,
        noise: NoiseModel::Bernoulli,
    })
}

fn competing_risks_window_spec(
    model: &FittedModel,
    request: &SavedGenerativeInput<'_>,
) -> Result<GenerativeSpec, SavedGenerativeError> {
    let exit = predict_competing_risks_survival(
        plugin_survival_request(
            model,
            request.data.view(),
            request.col_map,
            request.training_headers,
            request.offset,
            request.offset_noise,
            None,
        ),
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .map_err(|error| SavedGenerativeError::Evaluation {
        reason: format!("saved competing-risk exit probabilities: {error}"),
    })?;
    let n = request.data.nrows();
    let causes = exit.cif.len();
    if causes < 2 || exit.overall_survival.shape() != [n, 1] {
        return Err(SavedGenerativeError::InvalidState {
            reason: format!(
                "saved competing-risk exit law has causes={causes}, overall-survival shape {:?}",
                exit.overall_survival.shape()
            ),
        });
    }
    let entry_times = survival_entry_times(model, request.data.view(), request.col_map)?;
    let mut entry_survival = Array1::<f64>::zeros(n);
    let mut entry_cif = (0..causes)
        .map(|_| Array1::<f64>::zeros(n))
        .collect::<Vec<_>>();
    // Each cause returns hazard, survival, cumulative hazard, and CIF, plus one
    // overall-survival square. Account for all live matrices in the chunk
    // policy; counting a single f64 surface underestimates peak memory by
    // roughly 4K+1.
    let live_surfaces = causes.saturating_mul(4).saturating_add(1);
    let chunk_rows = entry_surface_chunk_rows(n, live_surfaces);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let grid = entry_times.slice(s![start..end]).to_vec();
        let offset = request.offset.slice(s![start..end]).to_owned();
        let offset_noise = request.offset_noise.slice(s![start..end]).to_owned();
        let chunk = predict_competing_risks_survival(
            plugin_survival_request(
                model,
                request.data.slice(s![start..end, ..]),
                request.col_map,
                request.training_headers,
                &offset,
                &offset_noise,
                Some(&grid),
            ),
            SurvivalPredictionCovarianceMode::Conditional,
        )
        .map_err(|error| SavedGenerativeError::Evaluation {
            reason: format!(
                "saved competing-risk entry probabilities rows {start}..{end}: {error}"
            ),
        })?;
        if chunk.cif.len() != causes || chunk.overall_survival.shape() != [end - start, end - start]
        {
            return Err(SavedGenerativeError::InvalidState {
                reason: format!(
                    "saved competing-risk entry law rows {start}..{end} has causes={}, overall-survival shape {:?}",
                    chunk.cif.len(),
                    chunk.overall_survival.shape()
                ),
            });
        }
        for local_row in 0..end - start {
            entry_survival[start + local_row] = chunk.overall_survival[[local_row, local_row]];
            for cause in 0..causes {
                entry_cif[cause][start + local_row] = chunk.cif[cause][[local_row, local_row]];
            }
        }
    }

    let labels = Array1::from_iter((0..=causes).map(|code| code as f64));
    let mut probabilities = Array2::<f64>::zeros((n, causes + 1));
    let mut mean = Array1::<f64>::zeros(n);
    for row in 0..n {
        let survival =
            conditional_window_survival(entry_survival[row], exit.overall_survival[[row, 0]], row)?;
        probabilities[[row, 0]] = survival;
        let event_probability = 1.0 - survival;
        let mut increments = vec![0.0_f64; causes];
        let mut increment_sum = 0.0_f64;
        for cause in 0..causes {
            let increment = exit.cif[cause][[row, 0]] - entry_cif[cause][row];
            let tolerance = 128.0 * f64::EPSILON;
            if !increment.is_finite() || increment < -tolerance {
                return Err(SavedGenerativeError::Evaluation {
                    reason: format!(
                        "competing-risk row {row}, cause {} has invalid CIF window increment {increment}",
                        cause + 1
                    ),
                });
            }
            increments[cause] = increment.max(0.0);
            increment_sum += increments[cause];
        }
        if event_probability > 0.0 && !(increment_sum.is_finite() && increment_sum > 0.0) {
            return Err(SavedGenerativeError::Evaluation {
                reason: format!(
                    "competing-risk row {row} has event probability {event_probability} but no positive cause incidence increment"
                ),
            });
        }
        for cause in 0..causes {
            let probability = if event_probability == 0.0 {
                0.0
            } else {
                event_probability * increments[cause] / increment_sum
            };
            probabilities[[row, cause + 1]] = probability;
            mean[row] += labels[cause + 1] * probability;
        }
    }
    Ok(GenerativeSpec {
        mean,
        noise: NoiseModel::Categorical {
            probabilities,
            labels,
        },
    })
}

fn survival_window_generative_spec(
    model: &FittedModel,
    request: &SavedGenerativeInput<'_>,
) -> Result<GenerativeSpec, SavedGenerativeError> {
    match &model.payload().family_state {
        FittedFamily::LatentSurvival { .. } | FittedFamily::LatentBinary { .. } => {
            let result = predict_latent_window_survival(plugin_survival_request(
                model,
                request.data.view(),
                request.col_map,
                request.training_headers,
                request.offset,
                request.offset_noise,
                None,
            ))
            .map_err(|error| SavedGenerativeError::Evaluation {
                reason: format!("saved latent window probability: {error}"),
            })?;
            let event_probability = result.window_survival.mapv(|survival| 1.0 - survival);
            Ok(GenerativeSpec {
                mean: event_probability,
                noise: NoiseModel::Bernoulli,
            })
        }
        FittedFamily::Survival { .. } if model.payload().survival_cause_count.unwrap_or(1) > 1 => {
            competing_risks_window_spec(model, request)
        }
        FittedFamily::Survival { .. } => single_cause_window_spec(model, request),
        family => Err(SavedGenerativeError::InvalidState {
            reason: format!("survival model class carries non-survival fitted family {family:?}"),
        }),
    }
}

/// Reconstruct the fitted observation distribution at new rows.
///
/// This is the sole saved-model capability dispatch for both the CLI generate
/// command and Python `Model.sample_replicates`.  It never refits, materializes
/// a scan model as a dense spline, substitutes unit weights, or keeps a second
/// response-family allowlist at an API boundary.
pub fn generative_spec_for_saved_model(
    model: &FittedModel,
    request: SavedGenerativeInput<'_>,
) -> Result<GenerativeSpec, SavedGenerativeError> {
    let SavedGenerativeInput {
        data,
        col_map,
        training_headers,
        offset,
        offset_noise,
        noise_offset_supplied,
        prior_weights,
    } = request;
    if let FittedEstimator::Expectile { .. } = model.estimator() {
        return Err(SavedGenerativeError::UnsupportedSampler {
            model_class: model.predict_model_class(),
            family: model.payload().family.clone(),
        });
    }
    if let Some(spec) = spline_scan_generative_spec(model, data.view(), col_map, prior_weights)? {
        return Ok(spec);
    }

    let model_class = model.predict_model_class();
    let weights = validated_prior_weights(model, data.nrows(), prior_weights)?;
    let predictor_response = || {
        let input = build_predict_input_for_model(
            model,
            data.view(),
            col_map,
            training_headers,
            offset,
            offset_noise,
            noise_offset_supplied,
        )
        .map_err(|reason| SavedGenerativeError::Evaluation {
            reason: format!("prediction input: {reason}"),
        })?;
        let predictor =
            model
                .predictor()
                .ok_or_else(|| SavedGenerativeError::MissingSavedState {
                    model_class,
                    reason: "canonical predictor could not be reconstructed".to_string(),
                })?;
        let prediction = predictor.predict_plugin_response(&input).map_err(|error| {
            SavedGenerativeError::Evaluation {
                reason: format!("plug-in response prediction: {error}"),
            }
        })?;
        Ok::<_, SavedGenerativeError>((input, predictor, prediction))
    };

    match model_class {
        PredictModelClass::GaussianLocationScale => {
            let (input, predictor, prediction) = predictor_response()?;
            let sigma = predictor
                .predict_noise_scale(&input)
                .map_err(|error| SavedGenerativeError::Evaluation {
                    reason: format!("Gaussian location-scale noise prediction: {error}"),
                })?
                .ok_or_else(|| SavedGenerativeError::MissingSavedState {
                    model_class,
                    reason: "canonical predictor defines no Gaussian sigma channel".to_string(),
                })?;
            if sigma.len() != prediction.mean.len() {
                return Err(SavedGenerativeError::InvalidState {
                    reason: format!(
                        "Gaussian location-scale sigma length {} does not match mean length {}",
                        sigma.len(),
                        prediction.mean.len(),
                    ),
                });
            }
            Ok(GenerativeSpec {
                mean: prediction.mean,
                noise: NoiseModel::Gaussian { sigma },
            })
        }
        PredictModelClass::DispersionLocationScale => {
            let (input, predictor, prediction) = predictor_response()?;
            let dispersion = predictor
                .predict_dispersion_scale(&input)
                .map_err(|error| SavedGenerativeError::Evaluation {
                    reason: format!("dispersion location-scale noise prediction: {error}"),
                })?
                .ok_or_else(|| SavedGenerativeError::MissingSavedState {
                    model_class,
                    reason: "canonical predictor defines no per-row dispersion channel".to_string(),
                })?;
            let noise = NoiseModel::from_likelihood_with_per_row_dispersion(
                &model.likelihood(),
                dispersion,
            )
            .map_err(|error| SavedGenerativeError::InvalidState {
                reason: format!("dispersion location-scale observation law: {error}"),
            })?;
            Ok(GenerativeSpec {
                mean: prediction.mean,
                noise,
            })
        }
        PredictModelClass::Standard => {
            let (_, _, prediction) = predictor_response()?;
            let fit = model
                .fit_result
                .as_ref()
                .or_else(|| model.unified())
                .ok_or_else(|| SavedGenerativeError::MissingSavedState {
                    model_class,
                    reason: "fitted dispersion metadata is absent".to_string(),
                })?;
            let likelihood = model.likelihood();
            let parameter =
                family_noise_parameter(fit.likelihood_scale, fit.standard_deviation, &likelihood)
                    .map_err(|error| SavedGenerativeError::InvalidState {
                    reason: format!("fitted family noise parameter: {error}"),
                })?;
            generativespec_from_predict(prediction, likelihood, parameter, weights).map_err(
                |error| SavedGenerativeError::InvalidState {
                    reason: format!("fitted observation law: {error}"),
                },
            )
        }
        PredictModelClass::BinomialLocationScale | PredictModelClass::BernoulliMarginalSlope => {
            let (_, _, prediction) = predictor_response()?;
            let noise =
                NoiseModel::from_likelihood(&model.likelihood(), prediction.mean.len(), None)
                    .map_err(|error| SavedGenerativeError::InvalidState {
                        reason: format!("fitted Bernoulli observation law: {error}"),
                    })?;
            Ok(GenerativeSpec {
                mean: prediction.mean,
                noise,
            })
        }
        PredictModelClass::TransformationNormal => {
            let grid = build_transformation_normal_quantile_grid(
                model,
                data,
                col_map,
                training_headers,
                offset,
            )
            .map_err(|reason| SavedGenerativeError::Evaluation {
                reason: format!("transformation-normal quantile grid: {reason}"),
            })?;
            Ok(GenerativeSpec {
                mean: grid.conditional_mean,
                noise: NoiseModel::TransformationNormalQuantile {
                    grid_y: grid.grid_y,
                    h_grid: grid.h_grid,
                },
            })
        }
        PredictModelClass::Survival => survival_window_generative_spec(
            model,
            &SavedGenerativeInput {
                data,
                col_map,
                training_headers,
                offset,
                offset_noise,
                noise_offset_supplied,
                prior_weights,
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use gam_models::inference::generative::sampleobservation_replicates;
    use gam_models::inference::model::{
        FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
    };
    use gam_models::inference::model_payload_builders::assemble_spline_scan_payload;
    use ndarray::{Array2, array};
    use rand::SeedableRng;

    #[test]
    fn event_window_probability_is_the_conditional_survival_ratio() {
        let conditional = conditional_window_survival(0.8, 0.2, 0).unwrap();
        assert_eq!(conditional.to_bits(), 0.25_f64.to_bits());
        let event_probability = 1.0 - conditional;
        assert_eq!(event_probability.to_bits(), 0.75_f64.to_bits());
    }

    #[test]
    fn event_window_probability_rejects_nonmonotone_or_depleted_entry_mass() {
        assert!(conditional_window_survival(0.4, 0.5, 3).is_err());
        assert!(conditional_window_survival(0.0, 0.0, 4).is_err());
    }

    #[test]
    fn entry_surface_chunk_budget_counts_every_live_matrix() {
        let surfaces = 9;
        let rows = entry_surface_chunk_rows(usize::MAX, surfaces);
        let live_bytes = rows * rows * surfaces * std::mem::size_of::<f64>();
        assert!(
            live_bytes
                <= gam_runtime::resource::ResourcePolicy::default_library().row_chunk_target_bytes
        );
    }

    #[test]
    fn expectile_surface_refuses_to_invent_a_gaussian_observation_law() {
        let mut payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            "y ~ 1".to_string(),
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: gam_problem::types::LikelihoodSpec::gaussian_identity(),
                link: Some(gam_problem::types::StandardLink::Identity),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            "expectile(0.9)".to_string(),
        );
        payload.estimator = FittedEstimator::Expectile { tau: 0.9 };
        let model = FittedModel::from_payload(payload);
        let data = Array2::<f64>::zeros((1, 0));
        let columns = HashMap::new();
        let zero = Array1::<f64>::zeros(1);
        let error = generative_spec_for_saved_model(
            &model,
            SavedGenerativeInput {
                data: data.view(),
                col_map: &columns,
                training_headers: None,
                offset: &zero,
                offset_noise: &zero,
                noise_offset_supplied: false,
                prior_weights: None,
            },
        )
        .expect_err("an expectile estimator alone does not define an observation sampler");
        assert!(matches!(
            error,
            SavedGenerativeError::UnsupportedSampler {
                family,
                model_class: PredictModelClass::Standard,
            } if family == "expectile(0.9)"
        ));
    }

    fn weighted_scan_model() -> (FittedModel, gam_solve::spline_scan::SplineScanFit) {
        let x: Vec<f64> = (0..80).map(|i| -2.0 + 4.0 * i as f64 / 79.0).collect();
        let weights: Vec<f64> = x
            .iter()
            .map(|&value| if value < 0.0 { 1.0 } else { 9.0 })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                0.4 + (1.3 * value).sin() + (i as f64 * 0.71).sin() / weights[i].sqrt()
            })
            .collect();
        let fit = gam_solve::spline_scan::fit_spline_scan(&x, &y, &weights, 2)
            .expect("weighted scan fit");
        let mut payload = assemble_spline_scan_payload(
            "y ~ s(x)".to_string(),
            "x".to_string(),
            &fit,
            DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "y".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "w".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            vec!["y".to_string(), "x".to_string(), "w".to_string()],
            vec![(0.0, 0.0), (-2.0, 2.0), (1.0, 9.0)],
        );
        payload.weight_column = Some("w".to_string());
        (FittedModel::from_payload(payload), fit)
    }

    fn spec_for_probe(
        model: &FittedModel,
        rows: ArrayView2<'_, f64>,
        weights: Option<&Array1<f64>>,
    ) -> Result<GenerativeSpec, SavedGenerativeError> {
        let columns = HashMap::from([
            ("y".to_string(), 0usize),
            ("x".to_string(), 1usize),
            ("w".to_string(), 2usize),
        ]);
        let offset = Array1::zeros(rows.nrows());
        generative_spec_for_saved_model(
            model,
            SavedGenerativeInput {
                data: rows,
                col_map: &columns,
                training_headers: model.payload().training_headers.as_ref(),
                offset: &offset,
                offset_noise: &offset,
                noise_offset_supplied: false,
                prior_weights: weights,
            },
        )
    }

    #[test]
    fn weighted_scan_spec_is_saved_bridge_mean_and_exact_sigma_over_sqrt_weight() {
        let (model, fit) = weighted_scan_model();
        let rows = array![[0.0, 0.15, 1.0], [0.0, 0.15, 9.0]];
        let weights = array![1.0, 9.0];
        let spec = spec_for_probe(&model, rows.view(), Some(&weights)).expect("scan spec");
        let expected_mean = fit.predict(0.15).expect("bridge prediction").0;
        assert_eq!(spec.mean[0].to_bits(), expected_mean.to_bits());
        assert_eq!(spec.mean[1].to_bits(), expected_mean.to_bits());
        let NoiseModel::Gaussian { sigma } = spec.noise else {
            panic!("weighted Gaussian scan must expose Gaussian observation noise");
        };
        assert_eq!(sigma[0].to_bits(), fit.sigma2.sqrt().to_bits());
        assert_eq!(sigma[1].to_bits(), (fit.sigma2.sqrt() / 3.0).to_bits());
    }

    #[test]
    fn weighted_scan_seed_and_saved_payload_round_trip_bit_exactly() {
        let (model, _) = weighted_scan_model();
        let bytes = serde_json::to_vec(&model).expect("serialize scan model");
        let restored: FittedModel = serde_json::from_slice(&bytes).expect("restore scan model");
        restored
            .validate_for_persistence()
            .expect("restored scan model validates");
        let rows: Array2<f64> = array![[0.0, -0.35, 0.75], [0.0, 0.65, 4.0]];
        let weights = array![0.75, 4.0];
        let original = spec_for_probe(&model, rows.view(), Some(&weights)).expect("original spec");
        let replay = spec_for_probe(&restored, rows.view(), Some(&weights)).expect("restored spec");
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(71);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(71);
        let draws_a = sampleobservation_replicates(&original, 128, &mut rng_a).expect("draw A");
        let draws_b = sampleobservation_replicates(&replay, 128, &mut rng_b).expect("draw B");
        assert_eq!(draws_a, draws_b);

        let error = spec_for_probe(&restored, rows.view(), None)
            .expect_err("weighted replay without row weights must fail");
        assert!(matches!(
            error,
            SavedGenerativeError::MissingRequiredInput { .. }
        ));
    }
}
