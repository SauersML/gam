//! Auto-selection helpers for latent-coordinate topology candidates.

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
}

impl TopologyAutoSelector {
    pub fn new(candidates: Option<Vec<AutoTopologyKind>>) -> Self {
        Self {
            candidates: candidates.unwrap_or_else(AutoTopologyKind::all),
            score_scale: TopologyScoreScale::PerEffectiveDim,
            latent: None,
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
        Ok(Self {
            candidates,
            score_scale,
            latent,
        })
    }
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
pub struct TopologyAutoSelectorResult<FitHandle> {
    pub ranked: Vec<TopologyAutoRankedFit<FitHandle>>,
    pub winner_index: usize,
}

impl<FitHandle> TopologyAutoSelectorResult<FitHandle> {
    pub fn winner(&self) -> Option<&TopologyAutoRankedFit<FitHandle>> {
        self.ranked.get(self.winner_index)
    }
}

pub fn select_topology_with_fit<FitHandle, FitErr>(
    selector: &TopologyAutoSelector,
    mut fit_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoFitEvidence<FitHandle>, FitErr>,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String>
where
    FitErr: ToString,
{
    let mut ranked = Vec::with_capacity(selector.candidates.len());
    let mut errors = Vec::new();
    for candidate in &selector.candidates {
        match fit_one(*candidate) {
            Ok(evidence) => {
                let tk_score = tk_normalized_score(
                    evidence.raw_reml,
                    evidence.null_dim,
                    evidence.null_space_logdet,
                    evidence.effective_dim,
                    evidence.n_obs,
                    selector.score_scale,
                )?;
                ranked.push(TopologyAutoRankedFit {
                    topology_name: evidence.topology_name,
                    tk_score,
                    raw_reml: evidence.raw_reml,
                    effective_dim: evidence.effective_dim,
                    n_obs: evidence.n_obs,
                    fit_handle: evidence.fit_handle,
                });
            }
            Err(err) => errors.push(format!("{}: {}", candidate.as_str(), err.to_string())),
        }
    }
    if ranked.is_empty() {
        return Err(format!(
            "TopologyAutoSelector found no fittable topology candidates{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }
    ranked.sort_by(|lhs, rhs| {
        rhs.tk_score
            .partial_cmp(&lhs.tk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(TopologyAutoSelectorResult {
        ranked,
        winner_index: 0,
    })
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
            return Err(
                "TopologyAutoSelector null-space Hessian logdet is not finite".to_string(),
            );
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
