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
//!
//! ## Selection-time stacking is now wired (WP-C / Object 3a)
//!
//! The winner-take-all blocker above is specifically about the SAVED-MODEL
//! PREDICTION path: that path does not retain alternative candidate predictors,
//! so it cannot evaluate a mixture (or any losing candidate) at new rows. That
//! blocker still stands for out-of-sample prediction.
//!
//! It does NOT, however, block stacking *at selection time*. During the race
//! the candidate fits all exist, so we can build a per-observation held-out
//! predictive log-density table by cross-validation folds within the race and
//! feed it to [`crate::solver::evidence::solve_stacking_weights`]. This module
//! does exactly that when a race mixes model classes (smooth manifold vs the
//! discrete-mixture rung): the HEADLINE ranking statistic switches to held-out
//! predictive log-density / stacking weights, with the rank-aware Laplace
//! evidence retained as corroboration. Same-class races keep today's
//! winner-take-all evidence behavior. The class mix is auto-detected from the
//! candidate kinds — there is no flag.
//!
//! What is still future work: persisting a mixture predictor for OOS prediction
//! on new rows. The selection-time stacking table is computed from fits that
//! exist only during the race; it is not retained for the saved-model
//! prediction path. That OOS-retention package is out of scope here.

use crate::solver::evidence::{
    GaussianMixtureConfig, StackingConfig, StackingWeights, TopologyScoreScale,
    UNION_STRUCTURE_LADDER, UnionStructure, UnionStructureFit, fit_gaussian_mixture,
    fit_union_ladder, fit_union_structure, solve_stacking_weights, union_per_point_log_density,
};
use crate::solver::priority_selection::{PriorityCandidate, rank_priority_candidates};
use ndarray::{Array2, ArrayView2};
use serde_json::Value as JsonValue;

const TK_LOG_2PI: f64 = 1.8378770664093453_f64;

/// Fixed component ladder swept for the discrete-mixture rung. Deterministic;
/// each `k` is priced by its own free-parameter count via the rank-aware
/// Laplace evidence and ranked against the others in-class before the winning
/// mixture order competes cross-class.
pub const MIXTURE_K_LADDER: &[usize] = &[1, 2, 3, 5, 7, 9];

/// Number of cross-validation folds used to build the selection-time held-out
/// predictive log-density table for cross-class stacking. Fixed (no flag).
pub const STACKING_CV_FOLDS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoTopologyKind {
    Euclidean,
    Circle,
    Sphere,
    Torus,
    Cylinder,
    /// Discrete `k`-component Gaussian-mixture rung (Object 3a / WP-C). Not a
    /// smooth manifold: a finite-parameter clustered density. Its presence in a
    /// race triggers cross-class stacking adjudication.
    Mixture {
        k: usize,
    },
    /// Structured-union composite (#907): a small FIXED set of component
    /// structures joined by a hard responsibility split (e.g. circle+circle,
    /// circle+cluster, line+cluster). Like the mixture rung it is a discrete,
    /// non-smooth density class — its evidence is the SUMMED rank-aware Laplace
    /// evidence of its components, priced by total parameter count. Its presence
    /// in a race triggers cross-class stacking adjudication.
    Union {
        structure: UnionStructure,
    },
}

impl AutoTopologyKind {
    /// Stable display name. The mixture variant carries its order `k`, so its
    /// label is rendered with [`AutoTopologyKind::display_name`]; the borrowed
    /// `as_str` returns the class tag for the smooth variants and the bare
    /// `"mixture"` tag for the discrete rung.
    pub const fn as_str(self) -> &'static str {
        match self {
            AutoTopologyKind::Euclidean => "euclidean",
            AutoTopologyKind::Circle => "circle",
            AutoTopologyKind::Sphere => "sphere",
            AutoTopologyKind::Torus => "torus",
            AutoTopologyKind::Cylinder => "cylinder",
            AutoTopologyKind::Mixture { .. } => "mixture",
            AutoTopologyKind::Union { structure } => structure.as_str(),
        }
    }

    /// Owned display name including the mixture order, e.g. `"mixture_k7"`, or
    /// the union composite tag, e.g. `"union_circle+circle"`.
    pub fn display_name(self) -> String {
        match self {
            AutoTopologyKind::Mixture { k } => format!("mixture_k{k}"),
            other => other.as_str().to_string(),
        }
    }

    /// `true` iff this candidate is the discrete-mixture model class (as
    /// opposed to a smooth manifold / Euclidean latent topology).
    pub const fn is_discrete_mixture(self) -> bool {
        matches!(self, AutoTopologyKind::Mixture { .. })
    }

    /// `true` iff this candidate is the structured-union composite class (#907).
    pub const fn is_structured_union(self) -> bool {
        matches!(self, AutoTopologyKind::Union { .. })
    }

    /// `true` iff this candidate is a discrete (non-smooth) density class — the
    /// mixture rung or a structured union. Cross-class stacking adjudication is
    /// triggered when a race mixes a smooth/Euclidean candidate with any
    /// discrete one.
    pub const fn is_discrete_class(self) -> bool {
        self.is_discrete_mixture() || self.is_structured_union()
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        if let Some(structure) = parse_union_name(&normalized) {
            return Ok(AutoTopologyKind::Union { structure });
        }
        if let Some(rest) = normalized.strip_prefix("mixture") {
            // Accept "mixture", "mixture_k7", "mixture7", "mixture_7".
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                // Bare "mixture" expands to the full ladder; callers that want a
                // single order pass "mixture_k{n}". Here we default to the
                // richest ladder order so a singleton request is still valid.
                return Ok(AutoTopologyKind::Mixture {
                    k: *MIXTURE_K_LADDER.last().unwrap_or(&7),
                });
            }
            let k: usize = digits
                .parse()
                .map_err(|_| format!("mixture order must be a positive integer; got {value:?}"))?;
            if k == 0 {
                return Err("mixture order k must be >= 1".to_string());
            }
            return Ok(AutoTopologyKind::Mixture { k });
        }
        match normalized.as_str() {
            "euclidean" | "flat" | "euclidean_patch" | "euclideanpatch" => {
                Ok(AutoTopologyKind::Euclidean)
            }
            "circle" | "periodic" | "s1" => Ok(AutoTopologyKind::Circle),
            "sphere" | "s2" => Ok(AutoTopologyKind::Sphere),
            "torus" => Ok(AutoTopologyKind::Torus),
            "cylinder" => Ok(AutoTopologyKind::Cylinder),
            other => Err(format!(
                "topology candidate must be euclidean, circle, sphere, torus, cylinder, mixture[_k{{n}}], or a union (union_circle+circle, union_circle+cluster, union_line+cluster); got {other:?}"
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

    /// The full discrete-mixture rung: one candidate per `k` in
    /// [`MIXTURE_K_LADDER`].
    pub fn mixture_ladder() -> Vec<Self> {
        MIXTURE_K_LADDER
            .iter()
            .map(|&k| AutoTopologyKind::Mixture { k })
            .collect()
    }

    /// The full structured-union rung: one candidate per composite in
    /// [`UNION_STRUCTURE_LADDER`] (#907). Fixed and closed — no open-ended
    /// structure search (that stays owned by #976's move set).
    pub fn union_ladder() -> Vec<Self> {
        UNION_STRUCTURE_LADDER
            .iter()
            .map(|&structure| AutoTopologyKind::Union { structure })
            .collect()
    }
}

/// Parse a structured-union composite name (#907). Accepts the canonical display
/// tags (`union_circle+circle`, `union_circle+cluster`, `union_line+cluster`)
/// and tolerant `_`/`-` separators in place of `+`. Returns `None` for any name
/// that is not a union so [`AutoTopologyKind::parse`] can fall through to the
/// smooth/mixture variants. Input is assumed already lowercased with `-`→`_`.
pub fn parse_union_name(normalized: &str) -> Option<UnionStructure> {
    let Some(rest) = normalized.strip_prefix("union") else {
        return None;
    };
    // Canonicalize component separators: both `+` and the tolerant `_`/`-`
    // forms collapse to `+`, and any leading separator after the `union`
    // prefix is dropped (so `union_circle+circle` and `union__circle_circle`
    // both normalize to `circle+circle`).
    let body: String = rest
        .chars()
        .map(|c| if c == '_' || c == '-' { '+' } else { c })
        .collect();
    let body = body.trim_matches('+');
    match body {
        "circle+circle" => Some(UnionStructure::CircleCircle),
        "circle+cluster" | "circle+point+cluster" | "circle+pointcluster" => {
            Some(UnionStructure::CirclePointCluster)
        }
        "line+cluster" | "line+point+cluster" | "line+pointcluster" => {
            Some(UnionStructure::LineCluster)
        }
        _ => None,
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
    // Sign convention (issue #396, see `solver::evidence`): `tk_score` is a
    // minimised TK / REML cost, so LOWER is better. Route through the shared
    // priority selector so topology ranking, seed screening, and model
    // comparison share one deterministic ordering contract (#782).
    ranked = rank_priority_candidates(
        ranked
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.tk_score;
                PriorityCandidate::new(row, idx, score, 0)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect();
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

// ===========================================================================
// Discrete-mixture rung + cross-class adjudication (Object 3a / WP-C)
// ===========================================================================

/// One fitted entry of the discrete-mixture rung: the mixture order `k`, the
/// fitted Gaussian mixture, and its rank-aware Laplace **negative** log evidence
/// computed through the SAME [`crate::solver::evidence::laplace_evidence`]
/// entry point used by the smooth rungs. Lower negative-log-evidence is better.
#[derive(Debug, Clone)]
pub struct MixtureRungFit {
    pub k: usize,
    pub fit: crate::solver::evidence::GaussianMixtureFit,
    /// Free-parameter count `P` — the quantity that enters the rank-aware
    /// normalizer as `dim(H) − rank(S) = P − 0`.
    pub num_parameters: usize,
    /// Rank-aware Laplace negative log evidence on the smooth-rung scale.
    pub negative_log_evidence: f64,
}

/// Result of fitting the whole mixture ladder: every fitted order plus the index
/// of the in-class winner (lowest rank-aware Laplace negative-log-evidence).
#[derive(Debug, Clone)]
pub struct MixtureRungResult {
    pub fits: Vec<MixtureRungFit>,
    pub winner_index: usize,
}

impl MixtureRungResult {
    pub fn winner(&self) -> &MixtureRungFit {
        &self.fits[self.winner_index]
    }
}

/// Hard cap on the number of EXTRA orders the local refinement around the
/// coarse-ladder winner may probe. Refinement walks one neighbour at a time
/// and stops as soon as the running winner is bracketed (both immediate
/// neighbours fitted and worse), so this cap only binds on a pathological
/// evidence profile that keeps improving monotonically past the ladder — a
/// regime the rank-aware parameter pricing rules out for any real cluster
/// structure. It exists so the sweep stays a bounded pure function of the
/// data, never a runaway loop.
pub const MIXTURE_REFINEMENT_MAX_PROBES: usize = 16;

/// Fit the discrete-mixture rung over a fixed `k`-ladder, then **refine
/// locally around the winner**, and rank in-class by rank-aware Laplace
/// evidence. Each order is priced by its own free-parameter count entering the
/// `−½ (dim(H) − rank(S)) log(2π)` normalizer. Deterministic: the seeding is
/// the basis k-means farthest-point init, EM is a pure map, and the refinement
/// order is a pure function of the fitted scores.
///
/// The coarse ladder ([`MIXTURE_K_LADDER`]) keeps the sweep cheap but cannot
/// *name* every order (it skips 4, 6, 8, …). Refinement closes that hole: after
/// the sweep, the immediate missing neighbours `k*−1`, `k*+1` of the running
/// winner are fitted, repeating until the winner is **bracketed** — both
/// neighbours present and scoring worse — so a planted `k = 4` truth is
/// recovered as exactly 4, not as the nearest ladder rung. On an in-ladder
/// winner the bracketing typically costs two extra EM fits and terminates at
/// the same order the coarse sweep found.
pub fn fit_mixture_rung(
    data: ArrayView2<'_, f64>,
    ladder: &[usize],
    config: GaussianMixtureConfig,
) -> Result<MixtureRungResult, String> {
    let n = data.nrows();
    let mut fits: Vec<MixtureRungFit> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    // Every order ever attempted (fitted OR failed): refinement must not
    // re-propose a failed order forever.
    let mut attempted: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    let mut try_order = |k: usize,
                         fits: &mut Vec<MixtureRungFit>,
                         errors: &mut Vec<String>,
                         attempted: &mut std::collections::BTreeSet<usize>| {
        if k == 0 || k > n || !attempted.insert(k) {
            return;
        }
        match fit_gaussian_mixture(data, k, config) {
            Ok(fit) => match fit.laplace_negative_log_evidence(data) {
                Ok(nle) => {
                    let num_parameters = fit.num_free_parameters();
                    fits.push(MixtureRungFit {
                        k,
                        fit,
                        num_parameters,
                        negative_log_evidence: nle,
                    });
                }
                Err(e) => errors.push(format!("mixture k={k} evidence: {e}")),
            },
            Err(e) => errors.push(format!("mixture k={k} fit: {e}")),
        }
    };

    for &k in ladder {
        try_order(k, &mut fits, &mut errors, &mut attempted);
    }
    if fits.is_empty() {
        return Err(format!(
            "mixture rung produced no fittable orders{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }

    // Local refinement: bracket the running winner. The running winner uses
    // the same rule as the final ranking (lower negative-log-evidence, ties to
    // the smaller k), so refinement and ranking can never disagree about who
    // the winner is.
    let mut probes = 0usize;
    while probes < MIXTURE_REFINEMENT_MAX_PROBES {
        let best_k = fits
            .iter()
            .min_by(|a, b| {
                a.negative_log_evidence
                    .partial_cmp(&b.negative_log_evidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.k.cmp(&b.k))
            })
            .map(|f| f.k)
            .unwrap_or(1);
        let next = [best_k.saturating_sub(1), best_k + 1]
            .into_iter()
            .find(|&k| k >= 1 && k <= n && !attempted.contains(&k));
        let Some(k) = next else {
            break; // bracketed: both neighbours attempted (or out of range).
        };
        try_order(k, &mut fits, &mut errors, &mut attempted);
        probes += 1;
    }
    // In-class winner-take-all on the rank-aware evidence scale (lower wins).
    let ranked = rank_priority_candidates(
        fits.into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.negative_log_evidence;
                let tie = row.k; // simpler (smaller k) wins ties
                PriorityCandidate::new(row, idx, score, tie)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect::<Vec<_>>();
    Ok(MixtureRungResult {
        fits: ranked,
        winner_index: 0,
    })
}

// ===========================================================================
// Structured-union rung (#907)
// ===========================================================================

/// One fitted entry of the structured-union rung: the composite structure, its
/// summed rank-aware Laplace **negative** log evidence (the SUM `Σ_c V_c` of its
/// components, each scored through the identical [`crate::solver::evidence::laplace_evidence`]
/// entry point used by the smooth rungs and the mixture rung), and the TOTAL
/// free-parameter count across components (the complexity price). Lower
/// negative-log-evidence wins.
#[derive(Debug, Clone)]
pub struct UnionRungFit {
    pub structure: UnionStructure,
    pub fit: UnionStructureFit,
    /// `Σ_c P_c` — total free-parameter count across all components. This is the
    /// complexity quantity that the summed `+ ½ Σ_c P_c log(2π)` normalizer
    /// charges, so a union is strictly more expensive than either pure rung.
    pub total_parameters: usize,
    /// `Σ_c V_c` — summed rank-aware Laplace negative log evidence.
    pub negative_log_evidence: f64,
}

/// Result of fitting the whole fixed union ladder: every fitted composite plus
/// the index of the in-class winner (lowest summed rank-aware Laplace
/// negative-log-evidence).
#[derive(Debug, Clone)]
pub struct UnionRungResult {
    pub fits: Vec<UnionRungFit>,
    pub winner_index: usize,
}

impl UnionRungResult {
    pub fn winner(&self) -> &UnionRungFit {
        &self.fits[self.winner_index]
    }
}

/// Fit the structured-union rung over the FIXED ladder
/// [`crate::solver::evidence::UNION_STRUCTURE_LADDER`] and rank in-class by
/// summed rank-aware Laplace evidence. Each composite is hard-split into one
/// responsibility group per component (reusing the mixture rung's deterministic
/// seeding + EM), each component is REML/Laplace-fit on its group, and the
/// per-component evidences are SUMMED. Composites whose groups are too small to
/// identify their structure are skipped (they never enter the race rather than
/// scoring spuriously well). Deterministic: the split and the component fits are
/// pure functions of the data.
pub fn fit_union_rung(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
) -> Result<UnionRungResult, String> {
    // `fit_union_ladder` already fits the fixed ladder and ranks best-first by
    // summed rank-aware evidence (cheaper composite wins ties). Re-wrap each
    // fit with its complexity price for the rung view.
    let ladder = fit_union_ladder(data, config)?;
    let fits: Vec<UnionRungFit> = ladder
        .into_iter()
        .map(|fit| UnionRungFit {
            structure: fit.structure,
            total_parameters: fit.total_parameters,
            negative_log_evidence: fit.negative_log_evidence,
            fit,
        })
        .collect();
    if fits.is_empty() {
        return Err("union rung produced no fittable composites".to_string());
    }
    Ok(UnionRungResult {
        fits,
        winner_index: 0,
    })
}

/// Fit a SINGLE structured-union composite and return it as a rung fit. Thin
/// convenience over [`crate::solver::evidence::fit_union_structure`] for callers
/// (and the race) that already chose a specific composite.
pub fn fit_union_candidate(
    data: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<UnionRungFit, String> {
    let fit = fit_union_structure(data, structure, config)?;
    Ok(UnionRungFit {
        structure: fit.structure,
        total_parameters: fit.total_parameters,
        negative_log_evidence: fit.negative_log_evidence,
        fit,
    })
}

/// A selection-time predictive-density provider: given the row indices to TRAIN
/// on and the row indices to EVALUATE on, it returns the per-eval-row held-out
/// log predictive density `log p(y_eval | train)`. This is the decoupled seam
/// that lets the cross-class race build a stacking table without persisting any
/// predictor: the closure refits on each fold's training rows.
///
/// The mixture provider is constructed here ([`mixture_density_provider`]); a
/// smooth-manifold provider is supplied by the caller (it owns the smooth
/// fitting machinery). Both refit per fold so the table is genuinely held-out.
pub type HeldOutDensityProvider<'a> =
    Box<dyn Fn(&[usize], &[usize]) -> Result<Vec<f64>, String> + 'a>;

/// Build a mixture held-out-density provider for a fixed order `k`. It refits a
/// `k`-component mixture on the training rows and scores the eval rows.
pub fn mixture_density_provider<'a>(
    data: ArrayView2<'a, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let train_mat = gather_rows(owned.view(), train);
            let fit = fit_gaussian_mixture(train_mat.view(), k.min(train.len().max(1)), config)?;
            let eval_mat = gather_rows(owned.view(), eval);
            let dens = fit.per_point_log_density(eval_mat.view())?;
            Ok(dens.to_vec())
        },
    )
}

/// Build a structured-union held-out-density provider for a fixed composite. It
/// refits the union's component densities on the training rows and scores the
/// eval rows under the soft mixture `log Σ_c π_c p_c(y)` (the union analogue of
/// [`mixture_density_provider`]). Refits per fold, so the stacking table is
/// genuinely held out.
pub fn union_density_provider<'a>(
    data: ArrayView2<'a, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let train_mat = gather_rows(owned.view(), train);
            let eval_mat = gather_rows(owned.view(), eval);
            let dens =
                union_per_point_log_density(train_mat.view(), eval_mat.view(), structure, config)?;
            Ok(dens.to_vec())
        },
    )
}

fn gather_rows(data: ArrayView2<'_, f64>, idx: &[usize]) -> Array2<f64> {
    let d = data.ncols();
    let mut out = Array2::<f64>::zeros((idx.len(), d));
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..d {
            out[[r, c]] = data[[i, c]];
        }
    }
    out
}

/// Deterministic contiguous `folds`-way CV partition of `0..n` (no clock
/// randomness). Returns, for each fold, `(train_indices, eval_indices)`.
pub fn deterministic_cv_folds(n: usize, folds: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let folds = folds.clamp(2, n.max(2));
    let mut out = Vec::with_capacity(folds);
    for f in 0..folds {
        let mut train = Vec::new();
        let mut eval = Vec::new();
        for i in 0..n {
            // Strided assignment is deterministic and balances fold sizes.
            if i % folds == f {
                eval.push(i);
            } else {
                train.push(i);
            }
        }
        if !eval.is_empty() && !train.is_empty() {
            out.push((train, eval));
        }
    }
    out
}

/// Build the selection-time held-out predictive log-density table
/// `log_density[i, c] = log p_c(y_i | train_fold(i))`, with one column per
/// candidate provider. Each row `i` is scored by the candidate refit on the CV
/// fold whose eval set contains `i`, so every entry is genuinely held out. This
/// is exactly the table that feeds
/// [`crate::solver::evidence::solve_stacking_weights`].
pub fn build_cv_log_density_table(
    n: usize,
    folds: usize,
    providers: &[HeldOutDensityProvider<'_>],
) -> Result<Array2<f64>, String> {
    if providers.is_empty() {
        return Err("stacking table requires at least one candidate provider".to_string());
    }
    let partition = deterministic_cv_folds(n, folds);
    if partition.is_empty() {
        return Err("stacking CV partition is empty (n too small for folds)".to_string());
    }
    let mut table = Array2::<f64>::from_elem((n, providers.len()), f64::NEG_INFINITY);
    for (train, eval) in &partition {
        for (col, provider) in providers.iter().enumerate() {
            let dens = provider(train, eval)?;
            if dens.len() != eval.len() {
                return Err(format!(
                    "provider {col} returned {} densities for {} eval rows",
                    dens.len(),
                    eval.len()
                ));
            }
            for (slot, &row) in eval.iter().enumerate() {
                table[[row, col]] = dens[slot];
            }
        }
    }
    Ok(table)
}

/// Adjudicated outcome of a cross-class race. When the race mixes a smooth
/// manifold candidate with the discrete-mixture rung, `headline` is the stacking
/// verdict (held-out predictive log-density), and the rank-aware Laplace
/// evidence is retained per-candidate as corroboration. Same-class races report
/// `Headline::Evidence` (winner-take-all on rank-aware evidence).
#[derive(Debug, Clone)]
pub struct CrossClassRaceVerdict {
    /// Candidate display names, column-aligned with the stacking table / weights.
    pub candidate_names: Vec<String>,
    /// Whether the race actually mixed model classes (smooth vs discrete).
    pub is_cross_class: bool,
    /// Rank-aware Laplace negative-log-evidence per candidate (corroboration;
    /// lower is better).
    pub negative_log_evidence: Vec<f64>,
    /// Stacking weights over the candidates (present iff `is_cross_class`).
    pub stacking: Option<StackingWeights>,
    /// Index of the headline winner. For cross-class races this is the max
    /// stacking-weight candidate; for same-class it is the min-evidence one.
    pub winner_index: usize,
    /// Which statistic drove the headline.
    pub headline: Headline,
}

/// Which statistic adjudicated the headline ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Headline {
    /// Rank-aware Laplace evidence (same-class race, winner-take-all).
    Evidence,
    /// Held-out predictive log-density / stacking weights (cross-class race).
    Stacking,
}

/// One candidate entering the cross-class adjudicator: its kind, its rank-aware
/// Laplace negative-log-evidence (already computed on the common scale), and a
/// selection-time held-out-density provider that refits per CV fold.
pub struct CrossClassCandidate<'a> {
    pub kind: AutoTopologyKind,
    pub negative_log_evidence: f64,
    pub density_provider: HeldOutDensityProvider<'a>,
}

/// Adjudicate a race that may mix smooth-manifold and discrete candidates
/// (the discrete-mixture rung and/or a structured union, #907). Cross-class
/// mixing is auto-detected from the candidate kinds (a race is cross-class iff
/// it contains BOTH at least one smooth/Euclidean candidate AND at least one
/// discrete candidate — [`AutoTopologyKind::Mixture`] or
/// [`AutoTopologyKind::Union`]). When cross-class,
/// the headline switches to stacking over a selection-time CV held-out
/// log-density table; otherwise the headline is the rank-aware evidence winner.
pub fn adjudicate_cross_class_race(
    n: usize,
    candidates: Vec<CrossClassCandidate<'_>>,
    folds: usize,
    stacking_config: StackingConfig,
) -> Result<CrossClassRaceVerdict, String> {
    if candidates.is_empty() {
        return Err("cross-class race requires at least one candidate".to_string());
    }
    let names: Vec<String> = candidates.iter().map(|c| c.kind.display_name()).collect();
    let evidence: Vec<f64> = candidates.iter().map(|c| c.negative_log_evidence).collect();

    // Cross-class iff the race mixes at least one discrete (non-smooth) density
    // class — the mixture rung OR a structured union (#907) — with at least one
    // smooth/Euclidean manifold candidate. A union competing against a smooth
    // ring must therefore adjudicate by held-out predictive stacking, exactly
    // like the mixture rung does.
    let has_discrete = candidates.iter().any(|c| c.kind.is_discrete_class());
    let has_smooth = candidates.iter().any(|c| !c.kind.is_discrete_class());
    let is_cross_class = has_discrete && has_smooth;

    if !is_cross_class {
        // Same-class: winner-take-all on rank-aware evidence (lower wins).
        let mut winner_index = 0usize;
        let mut best = f64::INFINITY;
        for (idx, &nle) in evidence.iter().enumerate() {
            if nle.is_finite() && nle < best {
                best = nle;
                winner_index = idx;
            }
        }
        return Ok(CrossClassRaceVerdict {
            candidate_names: names,
            is_cross_class: false,
            negative_log_evidence: evidence,
            stacking: None,
            winner_index,
            headline: Headline::Evidence,
        });
    }

    // Cross-class: build the selection-time held-out density table and stack.
    let providers: Vec<HeldOutDensityProvider<'_>> =
        candidates.into_iter().map(|c| c.density_provider).collect();
    let table = build_cv_log_density_table(n, folds, &providers)?;
    let stacking = solve_stacking_weights(table.view(), stacking_config)?;
    // Headline winner = max stacking weight (most predictive mass).
    let mut winner_index = 0usize;
    let mut best_w = f64::NEG_INFINITY;
    for (idx, &w) in stacking.weights.iter().enumerate() {
        if w > best_w {
            best_w = w;
            winner_index = idx;
        }
    }
    Ok(CrossClassRaceVerdict {
        candidate_names: names,
        is_cross_class: true,
        negative_log_evidence: evidence,
        stacking: Some(stacking),
        winner_index,
        headline: Headline::Stacking,
    })
}
