//! The event-history prediction contract: everything a prediction reads, and
//! the saved artifact that carries it (gam#2966).
//!
//! A fit holds the training cohort — its node expansion, its per-mark designs
//! over those nodes, its subjects' histories. A prediction reads none of that.
//! It reads the frozen schema (the mark vocabulary and kinds, the covariate
//! encoders and levels, the bases), the fitted functions and the probability
//! law, the reference law the baselines are normalised against, and the
//! posterior representation a posterior-predictive average integrates over.
//! [`PredictionModel`] is exactly that list, borrowed, and every prediction
//! function in `super::forecast` and `super::posterior` reads it.
//!
//! Two owners lend one: an [`EventHistoryFit`], which has just been fitted and
//! still holds its cohort, and an [`EventHistoryPredictor`], which was loaded
//! from a file and never had one. Because both lend the same view, a forecast
//! made from a reloaded artifact runs the same code over the same numbers as
//! the forecast made in the session that fitted it, and the save → reload →
//! forecast identity is the ordinary consequence rather than a second
//! implementation kept in step by hand.
//!
//! ## What the artifact holds, and what it cannot
//!
//! The document is the shared saved-model envelope
//! ([`gam_model_api::saved_model`]) of kind `event-history`, so a payload of
//! another kind or another version is refused there, typed, and never
//! migrated. It holds the schema, the law, the reference evolution
//! ([`RiskSetCentring`], which carries the reference population's profiles and
//! grid and nothing of a training participant) and the coefficient posterior.
//! It does not hold the reference population's DESIGNS, because they are
//! derived: `reference_tables` builds them by crossing the profiles with the
//! grid times and handing those rows to the same `build_term_collection_design`
//! under the same frozen specification, so
//! [`ReferenceTables::for_prediction`](super::family::ReferenceTables) rebuilds
//! them bit for bit from what the document does hold. Loading checks that
//! rebuild against the block widths the document records, so a payload whose
//! schema and law disagree is refused rather than forecast from.
//!
//! One thing an artifact cannot do is forecast a TRAINING subject by index:
//! that names a row of a cohort it does not have. `forecast_history` is the
//! serving entry point and carries its own covariate rows, which is why
//! `forecast` keeps taking a fit and a cohort.

use super::chain::GaussHermite;
use super::cohort::{EventHistoryCohort, EventHistoryError, MarkKind};
use super::family::{
    EventHistoryFit, ReferenceLawShape, ReferenceTables, RiskSetCentring, atom_rates_of,
    reference_law, reference_normalisers,
};
use super::posterior::ParameterState;
use gam_model_api::saved_model::{
    SavedModelError, read_saved_model_file, read_saved_model_text, saved_model_text,
    write_saved_model,
};
use gam_terms::smooth::TermCollectionSpec;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Kind of the saved event-history model in the shared saved-model envelope.
const EVENT_HISTORY_MODEL_KIND: &str = "event-history";

/// Version of the saved event-history model. Version 1 is the first: it saves
/// the frozen schema, the probability law, the reference evolution and the
/// coefficient posterior a posterior-predictive average integrates over.
const EVENT_HISTORY_MODEL_VERSION: u64 = 1;

/// Everything a prediction reads, borrowed from whichever owner holds it.
///
/// A prediction never sees an [`EventHistoryFit`] or an
/// [`EventHistoryCohort`]: those hold training records, and nothing here is
/// one. The covariate names and levels are the encoder, not the data — a
/// history brings its own rows.
pub(crate) struct PredictionModel<'a> {
    pub marks: usize,
    /// The rank of the latent covariance: how many atoms the state carries.
    pub atoms: usize,
    /// The column of a node row that holds time; the covariates precede it.
    pub time_column: usize,
    pub mark_names: &'a [String],
    pub mark_kinds: &'a [MarkKind],
    pub covariate_names: &'a [String],
    pub covariate_levels: &'a [Vec<String>],
    pub frozen_specs: &'a [TermCollectionSpec],
    pub quadrature_order: usize,
    pub mesh_refinement: usize,
    pub time_scale: f64,
    pub gh: &'a GaussHermite,
    pub quadrature_tolerance: f64,
    /// Where each block starts in a flat coefficient vector, total width last.
    pub block_offsets: Vec<usize>,
    pub rate_band: (f64, f64),
    pub held_rates: &'a [Option<f64>],
    pub reference: Option<&'a ReferenceTables>,
    /// The coefficients the model was fitted at, in the block layout.
    pub fitted_coefficients: Vec<f64>,
    /// The global parameter state those coefficients name, with the reference
    /// evolution the fit returned for them.
    pub fitted: ParameterState,
    /// The posterior covariance of the coefficients, when the model publishes
    /// one. A posterior-predictive average integrates over it
    /// (`super::posterior`); a conditional prediction never reads it.
    pub posterior_covariance: Option<&'a Array2<f64>>,
}

impl PredictionModel<'_> {
    /// The total coefficient width, which is where the last block ends.
    pub(crate) fn total_width(&self) -> usize {
        self.block_offsets[self.block_offsets.len() - 1]
    }

    /// Every atom's dimensionless rate at a latent block, through the model's
    /// own rate chart.
    pub(crate) fn atom_rates(&self, latent: &Array1<f64>) -> Vec<f64> {
        atom_rates_of(
            self.rate_band,
            self.held_rates,
            self.marks * self.atoms,
            latent,
        )
    }

    /// The reference evolution at a coefficient vector, or `None` where the
    /// baselines are centred on the stationary prior.
    ///
    /// This is the fit's own `reference_at` without its typed-refusal relay:
    /// the relay exists so the custom-family engine, which sees an error only
    /// as text, can read a `ReferenceStep` refusal back typed, and a
    /// prediction has no engine to read it back through — it returns the
    /// refusal to its caller directly.
    pub(crate) fn reference_at_coefficients(
        &self,
        beta: &[f64],
    ) -> Result<Option<RiskSetCentring>, EventHistoryError> {
        let Some(tables) = self.reference else {
            return Ok(None);
        };
        let width = self.total_width();
        if beta.len() != width {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "the reference evolution needs {width} coefficients in the model's layout, got {}",
                    beta.len()
                ),
            });
        }
        let latent_offset = self.block_offsets[self.marks];
        let latent = Array1::from(beta[latent_offset..].to_vec());
        let rates = self.atom_rates(&latent);
        let shape = ReferenceLawShape {
            block_offsets: &self.block_offsets,
            marks: self.marks,
            atoms: self.atoms,
            time_scale: self.time_scale,
            gh: self.gh,
        };
        let loadings = &beta[latent_offset..latent_offset + self.marks * self.atoms];
        let values = reference_normalisers(tables, &shape, beta, loadings, &rates)?;
        Ok(Some(reference_law(tables, beta, values)))
    }
}

/// The frozen schema, the law, the reference evolution and the posterior, as
/// one document.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PredictorDocument {
    mark_names: Vec<String>,
    mark_kinds: Vec<MarkKind>,
    covariate_names: Vec<String>,
    covariate_levels: Vec<Vec<String>>,
    frozen_specs: Vec<TermCollectionSpec>,
    time_column: usize,
    /// Flat coefficients: the mark blocks in mark order, then the latent block.
    coefficients: Vec<f64>,
    /// The width of each block, in that order. The rebuilt designs are checked
    /// against these on load, which is what refuses a document whose schema
    /// and law were saved from different models.
    block_widths: Vec<usize>,
    atoms: usize,
    loadings: Vec<f64>,
    log_rates: Vec<f64>,
    rate_band: (f64, f64),
    held_rates: Vec<Option<f64>>,
    time_scale: f64,
    quadrature_order: usize,
    mesh_refinement: usize,
    gauss_hermite_order: usize,
    quadrature_tolerance: f64,
    /// The reference evolution at the fitted coefficients, carrying the
    /// reference population's grid and profiles.
    centring: Option<RiskSetCentring>,
    /// The coefficient posterior a posterior-predictive average integrates
    /// over, `width × width`.
    posterior_covariance: Option<Array2<f64>>,
}

/// A saved event-history model: everything a prediction needs and nothing of
/// the training participants (gam#2966).
pub struct EventHistoryPredictor {
    document: PredictorDocument,
    /// Derived on construction and on load, never saved.
    gh: Arc<GaussHermite>,
    reference: Option<ReferenceTables>,
    fitted: ParameterState,
    block_offsets: Vec<usize>,
}

/// A refusal that names a state the document does not hold, or holds
/// inconsistently.
fn inconsistent(reason: impl Into<String>) -> SavedModelError {
    SavedModelError::Inconsistent {
        reason: Box::new(EventHistoryError::InvalidInput {
            reason: reason.into(),
        }),
    }
}

impl EventHistoryPredictor {
    /// The predictor of a fit: the schema, the law, the reference evolution
    /// and the posterior, lifted out of the fit and its cohort.
    ///
    /// Nothing of the cohort's subjects travels. What is read from it is the
    /// encoder — the mark vocabulary and kinds, the covariate names and their
    /// level codes — which a served history is encoded against.
    pub fn of(
        fit: &EventHistoryFit,
        cohort: &EventHistoryCohort,
    ) -> Result<Self, EventHistoryError> {
        if cohort.mark_kinds != fit.mark_kinds {
            return Err(EventHistoryError::InvalidInput {
                reason: "the cohort's mark kinds differ from the fit's".to_string(),
            });
        }
        let offsets = fit.coefficient_block_offsets();
        let document = PredictorDocument {
            mark_names: cohort.mark_names.clone(),
            mark_kinds: fit.mark_kinds.clone(),
            covariate_names: cohort.covariate_names.clone(),
            covariate_levels: cohort.covariate_levels.clone(),
            frozen_specs: fit.frozen_specs.clone(),
            time_column: fit.nodes.time_column,
            coefficients: fit.fitted_coefficients(),
            block_widths: offsets.windows(2).map(|w| w[1] - w[0]).collect(),
            atoms: fit.rank(),
            loadings: ParameterState::fitted_loadings(fit),
            log_rates: fit.log_rates.clone(),
            rate_band: fit.family.rate_band(),
            held_rates: fit.family.held_rates().to_vec(),
            time_scale: fit.time_scale,
            quadrature_order: fit.quadrature_order,
            mesh_refinement: fit.mesh_refinement,
            gauss_hermite_order: fit.family.gauss_hermite().order,
            quadrature_tolerance: fit.family.quadrature_tolerance(),
            centring: fit.centring.clone(),
            posterior_covariance: fit.fit.beta_covariance_corrected().cloned(),
        };
        Self::derive(document).map_err(|error| EventHistoryError::Fit {
            reason: error.to_string(),
        })
    }

    /// Build the runtime state a document implies, refusing a document that
    /// does not hold a model.
    fn derive(document: PredictorDocument) -> Result<Self, SavedModelError> {
        let marks = document.mark_kinds.len();
        if marks == 0
            || document.mark_names.len() != marks
            || document.frozen_specs.len() != marks
        {
            return Err(inconsistent(format!(
                "a saved event-history model names {marks} mark kinds, {} mark names and {} frozen specifications",
                document.mark_names.len(),
                document.frozen_specs.len()
            )));
        }
        if document.covariate_names.len() != document.covariate_levels.len() {
            return Err(inconsistent(format!(
                "a saved event-history model names {} covariates and {} level lists",
                document.covariate_names.len(),
                document.covariate_levels.len()
            )));
        }
        // The latent block is present exactly when the model carries atoms.
        let blocks = marks + usize::from(document.atoms > 0);
        if document.block_widths.len() != blocks {
            return Err(inconsistent(format!(
                "a saved event-history model of {marks} marks and {} atoms needs {blocks} coefficient blocks, and holds {}",
                document.atoms,
                document.block_widths.len()
            )));
        }
        let mut block_offsets = Vec::with_capacity(blocks + 1);
        let mut acc = 0usize;
        for width in &document.block_widths {
            block_offsets.push(acc);
            acc += width;
        }
        block_offsets.push(acc);
        if document.coefficients.len() != acc {
            return Err(inconsistent(format!(
                "a saved event-history model's blocks are {acc} coefficients wide and it holds {}",
                document.coefficients.len()
            )));
        }
        if document.loadings.len() != marks * document.atoms
            || document.log_rates.len() != document.atoms
            || document.held_rates.len() != document.atoms
        {
            return Err(inconsistent(format!(
                "a saved event-history model of {marks} marks and {} atoms holds {} loadings, {} log-rates and {} rate holds",
                document.atoms,
                document.loadings.len(),
                document.log_rates.len(),
                document.held_rates.len()
            )));
        }
        if let Some(covariance) = document.posterior_covariance.as_ref()
            && (covariance.nrows() != acc || covariance.ncols() != acc)
        {
            return Err(inconsistent(format!(
                "a saved event-history model of {acc} coefficients holds a {}×{} posterior covariance",
                covariance.nrows(),
                covariance.ncols()
            )));
        }
        let gh = GaussHermite::new(document.gauss_hermite_order)
            .map_err(|error| inconsistent(error.to_string()))?;
        let reference = match document.centring.as_ref() {
            Some(centring) => {
                let tables = ReferenceTables::for_prediction(
                    centring.grid.clone(),
                    centring.profiles.clone(),
                    document.mark_kinds.clone(),
                    &document.frozen_specs,
                )
                .map_err(|error| inconsistent(error.to_string()))?;
                // The designs are rebuilt, so their widths are the check that
                // the saved schema and the saved law came from one model.
                for (d, design) in tables.designs.iter().enumerate() {
                    if design.ncols() != document.block_widths[d] {
                        return Err(inconsistent(format!(
                            "mark {d}'s frozen specification rebuilds a reference design of {} columns, and the saved law holds {} coefficients for it",
                            design.ncols(),
                            document.block_widths[d]
                        )));
                    }
                }
                Some(tables)
            }
            None => None,
        };
        let fitted = ParameterState {
            mark_betas: (0..marks)
                .map(|d| {
                    Array1::from(
                        document.coefficients[block_offsets[d]..block_offsets[d + 1]].to_vec(),
                    )
                })
                .collect(),
            loadings: document.loadings.clone(),
            log_rates: document.log_rates.clone(),
            centring: document.centring.clone(),
        };
        Ok(Self {
            document,
            gh: Arc::new(gh),
            reference,
            fitted,
            block_offsets,
        })
    }

    /// The view every prediction reads.
    pub(crate) fn model(&self) -> PredictionModel<'_> {
        PredictionModel {
            marks: self.document.mark_kinds.len(),
            atoms: self.document.atoms,
            time_column: self.document.time_column,
            mark_names: &self.document.mark_names,
            mark_kinds: &self.document.mark_kinds,
            covariate_names: &self.document.covariate_names,
            covariate_levels: &self.document.covariate_levels,
            frozen_specs: &self.document.frozen_specs,
            quadrature_order: self.document.quadrature_order,
            mesh_refinement: self.document.mesh_refinement,
            time_scale: self.document.time_scale,
            gh: &self.gh,
            quadrature_tolerance: self.document.quadrature_tolerance,
            block_offsets: self.block_offsets.clone(),
            rate_band: self.document.rate_band,
            held_rates: &self.document.held_rates,
            reference: self.reference.as_ref(),
            fitted_coefficients: self.document.coefficients.clone(),
            fitted: self.fitted.clone(),
            posterior_covariance: self.document.posterior_covariance.as_ref(),
        }
    }

    /// The mark vocabulary, in the order the model's marks are indexed.
    pub fn mark_names(&self) -> &[String] {
        &self.document.mark_names
    }

    /// The mark kinds, in the same order.
    pub fn mark_kinds(&self) -> &[MarkKind] {
        &self.document.mark_kinds
    }

    /// The covariate columns a served history's rows must be laid out in.
    pub fn covariate_names(&self) -> &[String] {
        &self.document.covariate_names
    }

    /// The level codes of each covariate; empty for a numeric one.
    pub fn covariate_levels(&self) -> &[Vec<String>] {
        &self.document.covariate_levels
    }

    /// The rank of the latent covariance the model carries.
    pub fn rank(&self) -> usize {
        self.document.atoms
    }

    /// Save the model to `path`, atomically.
    pub fn save(&self, path: &Path) -> Result<(), SavedModelError> {
        write_saved_model(path, self.saved_text()?.as_bytes())
    }

    /// Load a saved model, refusing another kind or version, or a document
    /// that does not hold a model.
    pub fn load(path: &Path) -> Result<Self, SavedModelError> {
        Self::from_saved_text(&read_saved_model_file(path)?)
    }

    /// The saved document's text.
    pub fn saved_text(&self) -> Result<String, SavedModelError> {
        saved_model_text(
            EVENT_HISTORY_MODEL_KIND,
            EVENT_HISTORY_MODEL_VERSION,
            &self.document,
        )
    }

    /// The model in a saved document's text, refusing another kind or
    /// version, or a document that does not hold a model.
    pub fn from_saved_text(text: &str) -> Result<Self, SavedModelError> {
        let document: PredictorDocument = read_saved_model_text(
            text,
            EVENT_HISTORY_MODEL_KIND,
            EVENT_HISTORY_MODEL_VERSION,
        )?;
        Self::derive(document)
    }
}
