//! Auto-selection helpers for latent-coordinate topology candidates.
//!
//! This module deliberately returns a single selected topology rather than a
//! stacked predictive-distribution mixture. The selector is an evidence
//! comparator: its inputs are one scalar REML/LAML evidence summary per fitted
//! topology plus null-space normalizers. It does not receive a per-observation
//! held-out log predictive density table, nor does the saved-model prediction
//! path retain the alternative candidate fits required to evaluate a mixture at
//! new rows. A pseudo-BMA softmax over the scalar TK scores would therefore be
//! model-probability averaging of incomparable topology objects, not stacking of
//! predictive distributions. Until a real per-point LOO/ALO density consumer is
//! introduced alongside retained candidate predictors, the principled live path
//! is winner-take-all with deterministic score ordering.

use crate::solver::evidence::TopologyScoreScale;
use serde_json::Value as JsonValue;

const TK_LOG_2PI: f64 = 1.8378770664093453_f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoTopologyKind {
    Euclidean,
    Circle,
    Sphere,
    Torus,
    Cylinder,
}

impl AutoTopologyKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            AutoTopologyKind::Euclidean => "euclidean",
            AutoTopologyKind::Circle => "circle",
            AutoTopologyKind::Sphere => "sphere",
            AutoTopologyKind::Torus => "torus",
            AutoTopologyKind::Cylinder => "cylinder",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "euclidean" | "flat" | "euclidean_patch" | "euclideanpatch" => {
                Ok(AutoTopologyKind::Euclidean)
            }
            "circle" | "periodic" | "s1" => Ok(AutoTopologyKind::Circle),
            "sphere" | "s2" => Ok(AutoTopologyKind::Sphere),
            "torus" => Ok(AutoTopologyKind::Torus),
            "cylinder" => Ok(AutoTopologyKind::Cylinder),
            other => Err(format!(
                "topology candidate must be euclidean, circle, sphere, torus, or cylinder; got {other:?}"
            )),
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Circle,
            AutoTopologyKind::Sphere,
            AutoTopologyKind::Torus,
            AutoTopologyKind::Cylinder,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct TopologyAutoSelector {
    pub candidates: Vec<AutoTopologyKind>,
    pub score_scale: TopologyScoreScale,
    pub latent: Option<String>,
    pub screen_survivor_count: usize,
}

impl TopologyAutoSelector {
    pub fn new(candidates: Option<Vec<AutoTopologyKind>>) -> Self {
        Self {
            candidates: candidates.unwrap_or_else(AutoTopologyKind::all),
            score_scale: TopologyScoreScale::PerEffectiveDim,
            latent: None,
            screen_survivor_count: 2,
        }
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or_else(|| "topology_auto_selector must be an object".to_string())?;
        let candidates = match obj.get("candidates").filter(|value| !value.is_null()) {
            None => AutoTopologyKind::all(),
            Some(raw) => {
                let items = raw.as_array().ok_or_else(|| {
                    "topology_auto_selector.candidates must be a list".to_string()
                })?;
                if items.is_empty() {
                    return Err(
                        "topology_auto_selector.candidates must have at least one entry"
                            .to_string(),
                    );
                }
                let mut out = Vec::with_capacity(items.len());
                for (idx, item) in items.iter().enumerate() {
                    let name = item.as_str().ok_or_else(|| {
                        format!("topology_auto_selector.candidates[{idx}] must be a string")
                    })?;
                    let kind = AutoTopologyKind::parse(name)?;
                    if out.contains(&kind) {
                        return Err(format!(
                            "topology_auto_selector duplicate candidate {:?}",
                            kind.as_str()
                        ));
                    }
                    out.push(kind);
                }
                out
            }
        };
        let score_scale = match obj
            .get("score_scale")
            .and_then(JsonValue::as_str)
            .unwrap_or("per_effective_dim")
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "per_observation" => TopologyScoreScale::PerObservation,
            "per_effective_dim" => TopologyScoreScale::PerEffectiveDim,
            other => {
                return Err(format!(
                    "topology_auto_selector.score_scale must be per_effective_dim or per_observation; got {other:?}"
                ));
            }
        };
        let latent = obj
            .get("latent")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "topology_auto_selector.latent must be a string".to_string())
            })
            .transpose()?;
        let screen_survivor_count = match obj.get("screen_survivor_count") {
            None => 2,
            Some(value) => {
                let raw = value.as_u64().ok_or_else(|| {
                    "topology_auto_selector.screen_survivor_count must be a positive integer"
                        .to_string()
                })?;
                if raw == 0 {
                    return Err(
                        "topology_auto_selector.screen_survivor_count must be positive".to_string(),
                    );
                }
                usize::try_from(raw).map_err(|_| {
                    "topology_auto_selector.screen_survivor_count is too large".to_string()
                })?
            }
        };
        Ok(Self {
            candidates,
            score_scale,
            latent,
            screen_survivor_count,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TopologyAutoScreeningEvidence {
    pub topology_name: String,
    pub raw_reml: f64,
    pub null_dim: f64,
    pub null_space_logdet: Option<f64>,
    pub effective_dim: f64,
    pub n_obs: usize,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoFitEvidence<FitHandle> {
    pub topology_name: String,
    pub raw_reml: f64,
    pub null_dim: f64,
    pub null_space_logdet: Option<f64>,
    pub effective_dim: f64,
    pub n_obs: usize,
    pub fit_handle: FitHandle,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoRankedFit<FitHandle> {
    pub topology_name: String,
    pub tk_score: f64,
    pub raw_reml: f64,
    pub effective_dim: f64,
    pub n_obs: usize,
    pub fit_handle: FitHandle,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoRankedScreen {
    pub candidate_order: usize,
    pub kind: AutoTopologyKind,
    pub topology_name: String,
    pub tk_score: f64,
    pub raw_reml: f64,
    pub effective_dim: f64,
    pub n_obs: usize,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoScreeningResult {
    pub ranked: Vec<TopologyAutoRankedScreen>,
    pub survivor_kinds: Vec<AutoTopologyKind>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoSelectorResult<FitHandle> {
    pub ranked: Vec<TopologyAutoRankedFit<FitHandle>>,
    pub winner_index: usize,
}

impl<FitHandle> TopologyAutoSelectorResult<FitHandle> {
    pub fn winner(&self) -> Option<&TopologyAutoRankedFit<FitHandle>> {
        self.ranked.get(self.winner_index)
    }
}

pub fn screen_topology_candidates<ScreenErr>(
    selector: &TopologyAutoSelector,
    mut screen_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoScreeningEvidence, ScreenErr>,
) -> Result<TopologyAutoScreeningResult, String>
where
    ScreenErr: ToString,
{
    let mut ranked = Vec::with_capacity(selector.candidates.len());
    let mut errors = Vec::new();
    for (candidate_order, candidate) in selector.candidates.iter().enumerate() {
        match screen_one(*candidate) {
            Ok(evidence) => {
                let tk_score = tk_normalized_score(
                    evidence.raw_reml,
                    evidence.null_dim,
                    evidence.null_space_logdet,
                    evidence.effective_dim,
                    evidence.n_obs,
                    selector.score_scale,
                )?;
                ranked.push(TopologyAutoRankedScreen {
                    candidate_order,
                    kind: *candidate,
                    topology_name: evidence.topology_name,
                    tk_score,
                    raw_reml: evidence.raw_reml,
                    effective_dim: evidence.effective_dim,
                    n_obs: evidence.n_obs,
                });
            }
            Err(err) => errors.push(format!("{}: {}", candidate.as_str(), err.to_string())),
        }
    }
    if ranked.is_empty() {
        return Err(format!(
            "TopologyAutoSelector found no screenable topology candidates{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }
    ranked.sort_by(|lhs, rhs| {
        crate::solver::priority_search::compare_min_finite_scores(
            lhs.tk_score,
            rhs.tk_score,
            || lhs.candidate_order.cmp(&rhs.candidate_order),
        )
    });
    let survivor_count = selector.screen_survivor_count.min(ranked.len()).max(1);
    let survivor_kinds = ranked
        .iter()
        .take(survivor_count)
        .map(|candidate| candidate.kind)
        .collect();
    Ok(TopologyAutoScreeningResult {
        ranked,
        survivor_kinds,
        errors,
    })
}

fn rank_topology_fits<FitHandle>(
    selector: &TopologyAutoSelector,
    evidence_rows: Vec<TopologyAutoFitEvidence<FitHandle>>,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String> {
    let mut ranked_with_order = Vec::with_capacity(evidence_rows.len());
    for (candidate_order, evidence) in evidence_rows.into_iter().enumerate() {
        let tk_score = tk_normalized_score(
            evidence.raw_reml,
            evidence.null_dim,
            evidence.null_space_logdet,
            evidence.effective_dim,
            evidence.n_obs,
            selector.score_scale,
        )?;
        ranked_with_order.push((
            candidate_order,
            TopologyAutoRankedFit {
                topology_name: evidence.topology_name,
                tk_score,
                raw_reml: evidence.raw_reml,
                effective_dim: evidence.effective_dim,
                n_obs: evidence.n_obs,
                fit_handle: evidence.fit_handle,
            },
        ));
    }
    ranked_with_order.sort_by(|(lhs_order, lhs), (rhs_order, rhs)| {
        crate::solver::priority_search::compare_min_finite_scores(
            lhs.tk_score,
            rhs.tk_score,
            || lhs_order.cmp(&rhs_order),
        )
    });
    let ranked = ranked_with_order
        .into_iter()
        .map(|(_, ranked)| ranked)
        .collect();
    Ok(TopologyAutoSelectorResult {
        ranked,
        winner_index: 0,
    })
}

pub fn select_topology_with_screening<FitHandle, ScreenErr, FitErr>(
    selector: &TopologyAutoSelector,
    screen_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoScreeningEvidence, ScreenErr>,
    mut fit_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoFitEvidence<FitHandle>, FitErr>,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String>
where
    ScreenErr: ToString,
    FitErr: ToString,
{
    let screen = screen_topology_candidates(selector, screen_one)?;
    let mut evidence_rows = Vec::with_capacity(screen.survivor_kinds.len());
    let mut errors = screen.errors;
    for candidate in screen.survivor_kinds {
        match fit_one(candidate) {
            Ok(evidence) => evidence_rows.push(evidence),
            Err(err) => errors.push(format!("{}: {}", candidate.as_str(), err.to_string())),
        }
    }
    if evidence_rows.is_empty() {
        return Err(format!(
            "TopologyAutoSelector found no fittable topology candidates{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }
    rank_topology_fits(selector, evidence_rows)
}

pub fn select_topology_with_fit<FitHandle, FitErr>(
    selector: &TopologyAutoSelector,
    mut fit_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoFitEvidence<FitHandle>, FitErr>,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String>
where
    FitErr: ToString,
{
    let mut evidence_rows = Vec::with_capacity(selector.candidates.len());
    let mut errors = Vec::new();
    for candidate in &selector.candidates {
        match fit_one(*candidate) {
            Ok(evidence) => evidence_rows.push(evidence),
            Err(err) => errors.push(format!("{}: {}", candidate.as_str(), err.to_string())),
        }
    }
    if evidence_rows.is_empty() {
        return Err(format!(
            "TopologyAutoSelector found no fittable topology candidates{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }
    rank_topology_fits(selector, evidence_rows)
}

pub fn tk_normalized_score(
    raw_reml: f64,
    null_dim: f64,
    null_space_logdet: Option<f64>,
    effective_dim: f64,
    n_obs: usize,
    score_scale: TopologyScoreScale,
) -> Result<f64, String> {
    if !(raw_reml.is_finite() && null_dim.is_finite()) || null_dim < -1.0e-9 {
        return Err("TopologyAutoSelector received non-finite TK evidence inputs".to_string());
    }
    let normalizer = if null_dim.max(0.0) == 0.0 {
        0.0
    } else {
        let logdet = null_space_logdet.ok_or_else(|| {
            "TopologyAutoSelector TK normalizer requires null-space Hessian logdet".to_string()
        })?;
        if !logdet.is_finite() {
            return Err("TopologyAutoSelector null-space Hessian logdet is not finite".to_string());
        }
        -0.5 * null_dim.max(0.0) * TK_LOG_2PI + 0.5 * logdet
    };
    let tk = raw_reml + normalizer;
    match score_scale {
        TopologyScoreScale::PerObservation => {
            if n_obs == 0 {
                Err("TopologyAutoSelector requires n_obs > 0".to_string())
            } else {
                Ok(tk / n_obs as f64)
            }
        }
        TopologyScoreScale::PerEffectiveDim => {
            if !(effective_dim.is_finite() && effective_dim > 0.0) {
                Err("TopologyAutoSelector requires finite positive effective_dim".to_string())
            } else {
                Ok(tk / effective_dim)
            }
        }
    }
}
