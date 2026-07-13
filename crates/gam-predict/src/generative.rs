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
use gam_models::inference::model::{FittedModel, PredictModelClass};
use gam_models::inference::predict_io::PredictResult;
use ndarray::{Array1, ArrayView2};

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
        PredictModelClass::Survival => Err(SavedGenerativeError::UnsupportedSampler {
            model_class,
            family: model.likelihood().pretty_name().to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use gam_models::inference::generative::sampleobservation_replicates;
    use gam_models::inference::model::FittedModel;
    use gam_models::inference::model_payload_builders::assemble_spline_scan_payload;
    use ndarray::{Array2, array};
    use rand::SeedableRng;

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
