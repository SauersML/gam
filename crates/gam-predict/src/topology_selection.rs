//! Select a topology by fitting every candidate and ranking their evidence (#2899 P10).
//!
//! Two front doors share this one owner:
//! - response selection: `response ~ s(<every other column>, type=AUTO)`, each candidate
//!   replacing the AUTO term ([`auto_topology_formula`]);
//! - latent re-topologizing: a caller formula whose latent block is refitted on each
//!   candidate manifold, with the AUTO term (if the formula has one) replaced as well.
//!
//! The candidate loop, the per-candidate formula and request, the fits, the evidence each
//! fitted model publishes and the ranking all happen here. A front door describes the
//! candidates ([`CandidateTopology`]) and the scalar fit request document and marshals the
//! result.

use gam_data::EncodedDataset;
use gam_models::fit_orchestration::WorkflowError;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::saved_summary::saved_model_summary;
use gam_solve::topology_formula::{
    CandidateTopology, assemble_candidate_formula, has_auto_smooth_term,
};
use gam_solve::{
    TopologyCandidateEvidence, TopologyCandidateFailure, TopologyCandidateFailureStage,
    TopologyCandidateOutcome, TopologyCandidateSelectionResult, TopologySelectionScoreKind,
    TopologySelectionScoreScale, select_topology_candidate_lifecycle,
};

/// One named topology candidate.
#[derive(Clone, Debug)]
pub struct TopologyCandidate {
    pub name: String,
    pub topology: CandidateTopology,
}

/// Where the candidate list came from, which decides what a dimension mismatch means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyCandidateSource {
    /// The caller named every candidate: one that cannot take the smooth's dimension is a
    /// failed candidate, reported with the rest.
    Explicit,
    /// A default portfolio: a topology that cannot take the smooth's dimension is not a
    /// candidate for this data.
    Defaults,
}

/// What a selection fits.
pub struct TopologySelectionRequest<'a> {
    /// The formula; its `s(..., type=AUTO)` term, if any, is each candidate's to fill.
    pub formula: &'a str,
    pub candidates: Vec<TopologyCandidate>,
    pub source: TopologyCandidateSource,
    /// The latent block each candidate retopologizes: its `manifold` becomes the
    /// candidate's name. `None` for response selection.
    pub latent: Option<&'a str>,
    pub score_kind: TopologySelectionScoreKind,
    pub score_scale: TopologySelectionScoreScale,
    /// The scalar fit request document every candidate is fitted with.
    pub config_json: Option<&'a str>,
}

/// A completed topology selection.
pub struct TopologySelection {
    pub result: TopologyCandidateSelectionResult,
    /// The fitted model of every candidate that ranks, by name.
    pub fits: Vec<(String, FittedModel)>,
    /// The typed error of every candidate whose fit failed, by name, so a front door can
    /// surface the resume evidence it carries.
    pub fit_errors: Vec<(String, WorkflowError)>,
}

/// `response ~ s(<every other column>, type=AUTO)` over `dataset`, and the number of
/// predictor columns.
pub fn auto_topology_formula(dataset: &EncodedDataset, response: &str) -> Result<(String, usize), String> {
    let response = response.trim();
    if response.contains('~') {
        return Err("select_topology response must be a response column name".to_string());
    }
    if !dataset.headers.iter().any(|name| name == response) {
        return Err(format!("response column {response:?} not found in data"));
    }
    let features: Vec<&str> = dataset
        .headers
        .iter()
        .map(String::as_str)
        .filter(|&name| name != response)
        .collect();
    if features.is_empty() {
        return Err("plain-response select_topology needs at least one feature column".to_string());
    }
    if dataset.values.nrows() == 0 {
        return Err("select_topology data cannot be empty".to_string());
    }
    Ok((format!("{response} ~ s({}, type=AUTO)", features.join(", ")), features.len()))
}

fn failure(
    name: &str,
    stage: TopologyCandidateFailureStage,
    error_type: &str,
    message: String,
    evidence_at_failure: Option<f64>,
) -> TopologyCandidateOutcome {
    TopologyCandidateOutcome::Failed(TopologyCandidateFailure {
        name: name.to_string(),
        stage,
        error_type: error_type.to_string(),
        message,
        evidence_at_failure,
    })
}

/// The evidence a candidate's fitted model publishes, read under the one field each is
/// published as; the error carries the criterion when the model published one.
fn candidate_evidence(
    name: &str,
    model: &FittedModel,
    n_obs: usize,
) -> Result<TopologyCandidateEvidence, (String, Option<f64>)> {
    let summary = saved_model_summary(model).map_err(|error| (error.to_string(), None))?;
    let raw_reml = summary.raw_reml_score.ok_or_else(|| {
        (
            format!(
                "the candidate publishes no criterion: {}",
                summary.reml_score_unavailable.unwrap_or("no reason recorded")
            ),
            None,
        )
    })?;
    let effective_dim = summary
        .edf_total
        .ok_or_else(|| ("the candidate summary publishes no edf_total".to_string(), Some(raw_reml)))?;
    Ok(TopologyCandidateEvidence {
        name: name.to_string(),
        raw_reml,
        laml: None,
        null_dim: summary.null_dim,
        null_space_logdet: summary.null_space_logdet,
        effective_dim,
        basis_size: summary.coefficients.len(),
        n_obs,
    })
}

/// The request document with `latent`'s manifold set to `manifold`.
fn retopologized_document(
    document: &serde_json::Map<String, serde_json::Value>,
    latent: &str,
    manifold: &str,
) -> Result<String, String> {
    let mut document = document.clone();
    let entry = document
        .get_mut("latent_coordinates")
        .and_then(|latents| latents.get_mut(latent))
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| format!("topology selection latent {latent:?} not found"))?;
    entry.insert("manifold".to_string(), serde_json::Value::String(manifold.to_string()));
    serde_json::to_string(&document).map_err(|err| err.to_string())
}

/// Fit every candidate and rank their evidence.
///
/// One complete converged fit per candidate: no screening pass and no survivor
/// truncation can change the winner. A candidate that cannot be assembled, whose fit
/// fails, or whose evidence is unusable is a failed outcome beside the others. Candidate
/// order is kept end to end; the lifecycle breaks exact ties by it.
pub fn select_topology(
    dataset: &EncodedDataset,
    request: TopologySelectionRequest<'_>,
) -> Result<TopologySelection, String> {
    let mut names = std::collections::BTreeSet::new();
    for candidate in &request.candidates {
        if !names.insert(candidate.name.as_str()) {
            return Err(format!("duplicate topology candidate name {:?}", candidate.name));
        }
    }
    let document: serde_json::Map<String, serde_json::Value> = match request.config_json {
        Some(raw) if !raw.trim().is_empty() => {
            serde_json::from_str(raw).map_err(|err| format!("invalid fit config object: {err}"))?
        }
        _ => serde_json::Map::new(),
    };
    let n_obs = dataset.values.nrows();
    if let Some(latent) = request.latent {
        let rows = document
            .get("latent_coordinates")
            .and_then(|latents| latents.get(latent))
            .and_then(|entry| entry.get("n"))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("topology selection latent {latent:?} not found"))?;
        if rows as usize != n_obs {
            return Err(format!(
                "topology selection latent {latent:?} has n={rows}, but data has {n_obs} rows"
            ));
        }
    }
    let auto = has_auto_smooth_term(request.formula)?;
    let assembled: Vec<(String, Result<String, String>)> = request
        .candidates
        .into_iter()
        .filter_map(|candidate| {
            if !auto {
                return Some((candidate.name, Ok(request.formula.to_string())));
            }
            match assemble_candidate_formula(
                request.formula,
                &candidate.topology,
                request.source == TopologyCandidateSource::Explicit,
            ) {
                Ok(Some(formula)) => Some((candidate.name, Ok(formula))),
                Ok(None) => None,
                Err(message) => Some((candidate.name, Err(message))),
            }
        })
        .collect();
    let minimum = if request.latent.is_some() { 1 } else { 2 };
    if assembled.len() < minimum {
        return Err(match (request.source, minimum) {
            (TopologyCandidateSource::Defaults, 2) => {
                "select_topology requires at least two default candidates for these predictors"
                    .to_string()
            }
            (_, 2) => "select_topology requires at least two candidates".to_string(),
            _ => "topology selection requires at least one candidate".to_string(),
        });
    }
    let mut outcomes = Vec::with_capacity(assembled.len());
    let mut fits = Vec::with_capacity(assembled.len());
    let mut fit_errors = Vec::new();
    for (name, formula) in assembled {
        let prepared = formula.and_then(|formula| {
            let config_json = match request.latent {
                Some(latent) => Some(retopologized_document(&document, latent, &name)?),
                None => request.config_json.map(str::to_string),
            };
            let config = gam_config::parse_fit_config_json(config_json.as_deref())?;
            Ok((formula, config))
        });
        let (formula, config) = match prepared {
            Ok(prepared) => prepared,
            Err(message) => {
                outcomes.push(failure(
                    &name,
                    TopologyCandidateFailureStage::Assembly,
                    "gam_solve::topology_formula::AssemblyError",
                    message,
                    None,
                ));
                continue;
            }
        };
        match fit_formula_to_payload(formula, dataset, &config) {
            Ok(payload) => {
                let model = FittedModel::from_payload(payload);
                match candidate_evidence(&name, &model, n_obs) {
                    Ok(evidence) => {
                        outcomes.push(TopologyCandidateOutcome::Fitted(evidence));
                        fits.push((name, model));
                    }
                    Err((message, evidence_at_failure)) => outcomes.push(failure(
                        &name,
                        TopologyCandidateFailureStage::Evidence,
                        "gam_predict::topology_selection::EvidenceError",
                        message,
                        evidence_at_failure,
                    )),
                }
            }
            Err(error) => {
                outcomes.push(failure(
                    &name,
                    TopologyCandidateFailureStage::Fit,
                    "gam_models::fit_orchestration::WorkflowError",
                    error.to_string(),
                    None,
                ));
                fit_errors.push((name, error));
            }
        }
    }
    let result = select_topology_candidate_lifecycle(
        outcomes,
        request.score_kind,
        request.score_scale,
    )?;
    Ok(TopologySelection {
        result,
        fits,
        fit_errors,
    })
}
