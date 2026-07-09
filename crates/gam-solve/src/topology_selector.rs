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
//! feed it to [`crate::evidence::solve_stacking_weights`]. This module
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

use crate::row_sampling_measure::CoresetCertificate;
use crate::evidence::{
    GaussianMixtureConfig, StackingConfig, StackingWeights, TopologyScoreScale,
    UNION_STRUCTURE_LADDER, UnionStructure, UnionStructureFit, fit_gaussian_mixture,
    fit_union_ladder, fit_union_structure, solve_stacking_weights, union_per_point_log_density,
};
use crate::priority_selection::{PriorityCandidate, rank_priority_candidates};
use ndarray::{Array2, ArrayView2};
use serde_json::Value as JsonValue;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const TK_LOG_2PI: f64 = 1.8378770664093453_f64;

/// Fixed component ladder swept for the discrete-mixture rung. Deterministic;
/// each `k` is priced by its own free-parameter count via the rank-aware
/// Laplace evidence and ranked against the others in-class before the winning
/// mixture order competes cross-class.
pub const MIXTURE_K_LADDER: &[usize] = &[1, 2, 3, 5, 7, 9];

/// Number of cross-validation folds used to build the selection-time held-out
/// predictive log-density table for cross-class stacking. Fixed (no flag).
pub const STACKING_CV_FOLDS: usize = 5;

/// Default seed mixed into the deterministic CV fold assignment for cross-class
/// stacking. Matches the pyo3 `seed = 11` default of `adjudicate_atom_shape`, so
/// the FFI default and the in-tree default produce the identical (deterministic)
/// folding. Different seeds yield different — but still deterministic — foldings;
/// the same seed always reproduces the same folding (#1386).
pub const STACKING_CV_SEED: u64 = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoTopologyKind {
    Euclidean,
    Circle,
    Sphere,
    Torus,
    Cylinder,
    /// Möbius band (#2240): the unique non-orientable smooth `d = 2`
    /// candidate. Genuinely non-homotopic to the torus / cylinder / sphere /
    /// patch, so it races as its own discrete candidate — no curvature fusion
    /// applies to it.
    Mobius,
    /// Constant-curvature space form `M_κ` with the sectional curvature κ
    /// ESTIMATED (#944 stage 4). This single candidate replaces the family of
    /// fixed simply-connected constant-curvature geometries — `Euclidean`
    /// (κ = 0), `Sphere` (κ > 0), and the hyperbolic disk (κ < 0) — which are
    /// all the same `S^d ← ℝ^d → H^d` manifold at different κ. Rather than
    /// racing those as separate discrete candidates, the curvature is fitted as
    /// a continuous estimand (`curv(...)`), so the topology stack only has to
    /// adjudicate genuinely non-homotopic candidates (`Circle`/`Torus`/
    /// `Cylinder`/discrete rungs). "Within a candidate, curvature is estimated."
    ConstantCurvature,
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
            AutoTopologyKind::Mobius => "mobius",
            AutoTopologyKind::ConstantCurvature => "constant_curvature",
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
            "constant_curvature" | "curv" | "curvature" | "mkappa" | "m_kappa" => {
                Ok(AutoTopologyKind::ConstantCurvature)
            }
            other => Err(format!(
                "topology candidate must be euclidean, circle, sphere, torus, cylinder, constant_curvature, mixture[_k{{n}}], or a union (union_circle+circle, union_circle+cluster, union_line+cluster); got {other:?}"
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

    /// `true` iff this candidate is a FIXED member of the simply-connected
    /// constant-curvature space-form family `M_κ` — the geometries that differ
    /// only by their (fixed) sectional curvature κ and are therefore the *same*
    /// continuous `S^d ← ℝ^d → H^d` manifold at different κ:
    ///   * [`Euclidean`](AutoTopologyKind::Euclidean) — κ = 0,
    ///   * [`Sphere`](AutoTopologyKind::Sphere) — κ > 0.
    /// (`Circle`/`Torus`/`Cylinder` are NOT in this family: they are not
    /// simply connected — distinct topology, not distinct curvature — so they
    /// must keep racing as separate candidates.) The fitted-κ
    /// [`ConstantCurvature`](AutoTopologyKind::ConstantCurvature) candidate is
    /// itself the fusion target, not a member to be fused.
    pub const fn is_fixed_constant_curvature_form(self) -> bool {
        matches!(self, AutoTopologyKind::Euclidean | AutoTopologyKind::Sphere)
    }

    /// #944 stage 4 — "within a candidate, curvature is estimated." Collapse the
    /// fixed constant-curvature space forms in a candidate set into ONE
    /// [`ConstantCurvature`](AutoTopologyKind::ConstantCurvature) candidate whose
    /// κ is fitted, so the discrete topology stack only adjudicates genuinely
    /// non-homotopic candidates. Magic by default: the fusion fires exactly when
    /// the set contains **≥ 2** fixed constant-curvature forms (e.g. both
    /// `Euclidean` and `Sphere`) — a single such form is left untouched (there
    /// is nothing to estimate κ *across*), and a set already carrying an
    /// explicit `ConstantCurvature` candidate simply has its redundant fixed
    /// forms removed.
    ///
    /// Order is preserved and otherwise stable: the fused `ConstantCurvature`
    /// candidate takes the position of the first fixed form it replaces; all
    /// non-family candidates (`Circle`/`Torus`/`Cylinder`/`Mixture`/`Union`) and
    /// any duplicates keep their relative order. Idempotent.
    pub fn fuse_constant_curvature_family(candidates: &[Self]) -> Vec<Self> {
        let already_has_cc = candidates
            .iter()
            .any(|c| matches!(c, AutoTopologyKind::ConstantCurvature));
        let fixed_form_count = candidates
            .iter()
            .filter(|c| c.is_fixed_constant_curvature_form())
            .count();
        // Fuse when ≥2 fixed forms are present, OR when an explicit
        // ConstantCurvature candidate already subsumes ≥1 redundant fixed form.
        let should_fuse = fixed_form_count >= 2 || (already_has_cc && fixed_form_count >= 1);
        if !should_fuse {
            return candidates.to_vec();
        }
        let mut out = Vec::with_capacity(candidates.len());
        let mut emitted_cc = false;
        for &c in candidates {
            if c.is_fixed_constant_curvature_form() {
                // Replace the first fixed form by the fitted-κ candidate (unless
                // an explicit one already exists); drop the rest.
                if !already_has_cc && !emitted_cc {
                    out.push(AutoTopologyKind::ConstantCurvature);
                    emitted_cc = true;
                }
                continue;
            }
            if matches!(c, AutoTopologyKind::ConstantCurvature) {
                if emitted_cc {
                    continue; // collapse duplicate CC candidates
                }
                emitted_cc = true;
            }
            out.push(c);
        }
        out
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

/// Result for one candidate executed by [`run_topology_race_parallel`].
#[derive(Debug, Clone)]
pub struct TopologyRaceParallelCandidate<FitResult> {
    /// Original position in the input candidate vector.
    pub candidate_index: usize,
    /// The number of Rayon workers made available to this candidate's fit body.
    pub per_fit_threads: usize,
    /// Wall-clock time spent inside the candidate's local Rayon pool.
    pub wall_time: Duration,
    /// The fit closure's output. Use `FitResult = Result<T, E>` when individual
    /// candidate failures should be collected rather than short-circuiting.
    pub result: FitResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopologyRaceThreadPlan {
    coordinator_threads: usize,
    per_fit_threads: usize,
    concurrent_fits: usize,
}

impl TopologyRaceThreadPlan {
    fn for_budget(candidate_count: usize, max_total_threads: usize) -> Self {
        let max_total_threads = max_total_threads.max(1);
        if candidate_count <= 1 {
            return Self {
                coordinator_threads: 0,
                per_fit_threads: max_total_threads,
                concurrent_fits: candidate_count,
            };
        }

        let concurrent_fits = if max_total_threads >= 4 {
            candidate_count.min(max_total_threads / 2).max(1)
        } else {
            1
        };
        let coordinator_threads = concurrent_fits;
        let remaining = max_total_threads.saturating_sub(coordinator_threads);
        let per_fit_threads = if remaining == 0 {
            1
        } else {
            (remaining / concurrent_fits).max(1)
        };
        Self {
            coordinator_threads,
            per_fit_threads,
            concurrent_fits,
        }
    }
}

/// Run independent topology-race candidates concurrently with bounded nested
/// Rayon use.
///
/// Each candidate is executed inside its own local Rayon pool, so fit internals
/// that call `par_iter`, `rayon::join`, or faer-through-Rayon consume the
/// candidate's `per_fit_threads` budget rather than the global pool. For
/// multi-candidate races the runner batches candidates through a Rayon scope and
/// chooses `concurrent_fits`/`per_fit_threads` so the coordinator workers plus
/// per-fit workers do not exceed `std::thread::available_parallelism()` on hosts
/// with at least two cores. Single-core hosts run candidates sequentially.
///
/// The return vector is in input order and keeps each closure output intact; use
/// `FitResult = Result<T, E>` to collect per-candidate failures with wall times.
pub fn run_topology_race_parallel<Candidate, FitResult, FitOne>(
    candidates: Vec<Candidate>,
    fit_one: FitOne,
) -> Result<Vec<TopologyRaceParallelCandidate<FitResult>>, String>
where
    Candidate: Send,
    FitResult: Send,
    FitOne: Fn(Candidate) -> FitResult + Sync,
{
    let max_total_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    run_topology_race_parallel_with_budget(candidates, fit_one, max_total_threads)
}

fn run_topology_race_parallel_with_budget<Candidate, FitResult, FitOne>(
    candidates: Vec<Candidate>,
    fit_one: FitOne,
    max_total_threads: usize,
) -> Result<Vec<TopologyRaceParallelCandidate<FitResult>>, String>
where
    Candidate: Send,
    FitResult: Send,
    FitOne: Fn(Candidate) -> FitResult + Sync,
{
    let candidate_count = candidates.len();
    if candidate_count == 0 {
        return Ok(Vec::new());
    }

    let plan = TopologyRaceThreadPlan::for_budget(candidate_count, max_total_threads);
    let mut candidates: Vec<Option<Candidate>> = candidates.into_iter().map(Some).collect();
    let slots: Vec<Mutex<Option<TopologyRaceParallelCandidate<FitResult>>>> =
        (0..candidate_count).map(|_| Mutex::new(None)).collect();
    let pool_error: Mutex<Option<String>> = Mutex::new(None);

    if plan.concurrent_fits <= 1 {
        for idx in 0..candidate_count {
            let candidate = candidates[idx]
                .take()
                .expect("topology race candidate must be present");
            run_one_topology_race_candidate(
                idx,
                candidate,
                &fit_one,
                plan.per_fit_threads,
                &slots[idx],
                &pool_error,
            );
            if let Some(err) = pool_error.lock().expect("pool_error mutex poisoned").take() {
                return Err(err);
            }
        }
    } else {
        let coordinator_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(plan.coordinator_threads)
            .thread_name(|idx| format!("topology-race-coordinator-{idx}"))
            .build()
            .map_err(|err| format!("topology race coordinator Rayon pool: {err}"))?;
        let mut batch_start = 0usize;
        while batch_start < candidate_count {
            let batch_end = (batch_start + plan.concurrent_fits).min(candidate_count);
            coordinator_pool.scope(|scope| {
                for idx in batch_start..batch_end {
                    let candidate = candidates[idx]
                        .take()
                        .expect("topology race candidate must be present");
                    let slot = &slots[idx];
                    let pool_error = &pool_error;
                    let fit_one = &fit_one;
                    scope.spawn(move |_| {
                        run_one_topology_race_candidate(
                            idx,
                            candidate,
                            fit_one,
                            plan.per_fit_threads,
                            slot,
                            pool_error,
                        );
                    });
                }
            });
            if let Some(err) = pool_error.lock().expect("pool_error mutex poisoned").take() {
                return Err(err);
            }
            batch_start = batch_end;
        }
    }

    let mut out = Vec::with_capacity(candidate_count);
    for (idx, slot) in slots.into_iter().enumerate() {
        let row = slot
            .into_inner()
            .expect("topology race result mutex poisoned")
            .ok_or_else(|| format!("topology race candidate {idx} did not produce a result"))?;
        out.push(row);
    }
    Ok(out)
}

fn run_one_topology_race_candidate<Candidate, FitResult, FitOne>(
    candidate_index: usize,
    candidate: Candidate,
    fit_one: &FitOne,
    per_fit_threads: usize,
    slot: &Mutex<Option<TopologyRaceParallelCandidate<FitResult>>>,
    pool_error: &Mutex<Option<String>>,
) where
    Candidate: Send,
    FitResult: Send,
    FitOne: Fn(Candidate) -> FitResult + Sync,
{
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(per_fit_threads)
        .thread_name(move |idx| format!("topology-race-fit-{candidate_index}-{idx}"))
        .build()
    {
        Ok(pool) => pool,
        Err(err) => {
            *pool_error.lock().expect("pool_error mutex poisoned") =
                Some(format!("topology race candidate Rayon pool: {err}"));
            return;
        }
    };

    let started = Instant::now();
    // #2074 — each candidate fit runs inside its own nested Rayon pool. faer's
    // high-level solvers (arrow-Schur Cholesky/solve, SVD, QR) read the global
    // parallelism policy and, under the default `Par::rayon(0)`, dispatch through
    // faer's `spindle` barrier pool. From inside this already-nested Rayon worker
    // that barrier waits for pool slots the outer fan-out holds, deadlocking the
    // fit at 0% CPU. Pin faer to `Par::Seq` for the whole nested fit so it never
    // spawns a nested barrier pool; the per-candidate parallelism is the race
    // itself, and faer reductions are parallelism-invariant so the result is
    // bit-identical to the sequential path.
    let result = pool.install(|| {
        gam_linalg::faer_ndarray::with_faer_sequential(|| fit_one(candidate))
    });
    let wall_time = started.elapsed();
    *slot.lock().expect("topology race result mutex poisoned") =
        Some(TopologyRaceParallelCandidate {
            candidate_index,
            per_fit_threads,
            wall_time,
            result,
        });
}

pub fn select_topology_with_fit<FitHandle, FitErr>(
    selector: &TopologyAutoSelector,
    mut fit_one: impl FnMut(AutoTopologyKind) -> Result<TopologyAutoFitEvidence<FitHandle>, FitErr>,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String>
where
    FitErr: ToString,
{
    // #944 stage 4: collapse fixed simply-connected constant-curvature forms
    // (Euclidean/Sphere) into ONE estimated-κ ConstantCurvature candidate so the
    // discrete stack only adjudicates genuinely non-homotopic topologies.
    let fused = AutoTopologyKind::fuse_constant_curvature_family(&selector.candidates);
    let mut ranked = Vec::with_capacity(fused.len());
    let mut errors = Vec::new();
    for candidate in &fused {
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

/// Driver-level parallel sibling of [`select_topology_with_fit`] (#1017 Phase 0).
///
/// Topology candidates are INDEPENDENT fits — the sequential
/// [`select_topology_with_fit`] loop walks them one at a time, leaving 5–20× on
/// the table on a multi-core host. This variant fans the candidate fits across
/// the bounded-nested-Rayon [`run_topology_race_parallel`] driver (the same one
/// the closure-profile grid and the SAE-resident race already use), then ranks
/// the survivors through the IDENTICAL deterministic priority selector — results
/// come back in input order, so the winner is bit-identical to the sequential
/// path. The only contract difference is `fit_one: Fn + Sync` (each candidate
/// fit must be callable concurrently) instead of `FnMut`; callers whose fit
/// closure captures shared mutable state keep the sequential entry.
pub fn select_topology_with_fit_parallel<FitHandle, FitErr>(
    selector: &TopologyAutoSelector,
    fit_one: impl Fn(AutoTopologyKind) -> Result<TopologyAutoFitEvidence<FitHandle>, FitErr> + Sync,
) -> Result<TopologyAutoSelectorResult<FitHandle>, String>
where
    FitHandle: Send,
    FitErr: ToString + Send,
{
    // #944 stage 4: collapse fixed simply-connected constant-curvature forms
    // into ONE estimated-κ ConstantCurvature candidate before the parallel race.
    let candidates: Vec<AutoTopologyKind> =
        AutoTopologyKind::fuse_constant_curvature_family(&selector.candidates);
    let race = run_topology_race_parallel(candidates, |candidate| {
        // Carry the candidate kind alongside the fit so per-candidate failures
        // are reported with their topology name, exactly as the sequential path.
        (candidate, fit_one(candidate))
    })?;

    let mut ranked = Vec::with_capacity(race.len());
    let mut errors = Vec::new();
    for entry in race {
        let (candidate, fit_result) = entry.result;
        match fit_result {
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
    // Same deterministic priority ranking as the sequential path (#782): lower
    // tk_score is better; route through the shared selector so ordering is
    // identical regardless of which entry produced the candidates.
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

pub fn bic_score(deviance: f64, n_obs: usize, basis_size: usize) -> Result<f64, String> {
    if n_obs <= 1 {
        return Err("BIC scoring requires at least two observations".to_string());
    }
    if !deviance.is_finite() {
        return Err("BIC scoring requires finite deviance".to_string());
    }
    Ok(deviance + (n_obs as f64).ln() * basis_size as f64)
}

// ===========================================================================
// Discrete-mixture rung + cross-class adjudication (Object 3a / WP-C)
// ===========================================================================

/// One fitted entry of the discrete-mixture rung: the mixture order `k`, the
/// fitted Gaussian mixture, and its rank-aware Laplace **negative** log evidence
/// computed through the SAME [`crate::evidence::laplace_evidence`]
/// entry point used by the smooth rungs. Lower negative-log-evidence is better.
#[derive(Debug, Clone)]
pub struct MixtureRungFit {
    pub k: usize,
    pub fit: crate::evidence::GaussianMixtureFit,
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

    let try_order = |k: usize,
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
/// components, each scored through the identical [`crate::evidence::laplace_evidence`]
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
/// [`crate::evidence::UNION_STRUCTURE_LADDER`] and rank in-class by
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
/// convenience over [`crate::evidence::fit_union_structure`] for callers
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
///
/// Uses the default stacking seed ([`STACKING_CV_SEED`]); see
/// [`deterministic_cv_folds_seeded`] for the seed-reproducible variant (#1386).
pub fn deterministic_cv_folds(n: usize, folds: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    deterministic_cv_folds_seeded(n, folds, STACKING_CV_SEED)
}

/// SplitMix64 finalizer — a full-avalanche integer hash. Pure and deterministic:
/// it never touches the clock or any RNG state, so the folding it drives is
/// reproducible for a given `(seed, index)` and decorrelated across seeds.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic, seed-reproducible `folds`-way CV partition of `0..n` (no clock
/// randomness). The eval fold of sample `i` is `hash(seed, i) % folds`, so:
///   - the same `seed` always reproduces the identical folding (deterministic),
///   - different seeds give different (still-deterministic) foldings.
///
/// This is the seam that makes the `adjudicate_atom_shape` `seed=` kwarg
/// functional (#1386): it is mixed into the held-out CV partition rather than
/// being a silent no-op. Returns, for each fold, `(train_indices, eval_indices)`,
/// dropping any fold whose train or eval set is empty.
pub fn deterministic_cv_folds_seeded(
    n: usize,
    folds: usize,
    seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let folds = folds.clamp(2, n.max(2));
    // Precompute the seed-mixed fold of every sample once.
    let assign: Vec<usize> = (0..n)
        .map(|i| {
            // Mix the seed with the index through a full-avalanche hash so the
            // partition genuinely depends on both. The modulo keeps fold sizes
            // balanced in expectation while remaining deterministic.
            (splitmix64(seed ^ splitmix64(i as u64)) % folds as u64) as usize
        })
        .collect();
    let mut out = Vec::with_capacity(folds);
    for f in 0..folds {
        let mut train = Vec::new();
        let mut eval = Vec::new();
        for (i, &fold) in assign.iter().enumerate() {
            if fold == f {
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
/// [`crate::evidence::solve_stacking_weights`].
pub fn build_cv_log_density_table(
    n: usize,
    folds: usize,
    seed: u64,
    providers: &[HeldOutDensityProvider<'_>],
) -> Result<Array2<f64>, String> {
    if providers.is_empty() {
        return Err("stacking table requires at least one candidate provider".to_string());
    }
    let partition = deterministic_cv_folds_seeded(n, folds, seed);
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
    /// `Some(_)` when the same-class evidence winner's lead over the runner-up
    /// did NOT clear the decision margin required by approximate (enclosure /
    /// coreset) evidence — the verdict is provisional and the caller must
    /// refine or escalate to the exact path. `None` when the margin held (or no
    /// approximate evidence was involved, or the race adjudicated by stacking).
    pub insufficient_margin: Option<InsufficientRaceMargin>,
}

/// Which statistic adjudicated the headline ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Headline {
    /// Rank-aware Laplace evidence (same-class race, winner-take-all).
    Evidence,
    /// Held-out predictive log-density / stacking weights (cross-class race).
    Stacking,
}

/// How a candidate's `negative_log_evidence` was certified — the source of the
/// decision-margin the same-class race must respect before it can transfer a
/// verdict from approximate evidence to the full-corpus verdict.
///
/// * [`Exact`] — the evidence is a genuine point value (dense logdet, full
///   corpus); no margin floor.
/// * [`Enclosure`] — the log-determinant half came from a
///   [`block_preconditioned_logdet_enclosure`]; the race lead Δ must exceed the
///   enclosure gap (#1011 contract) before the winner is trustworthy.
/// * [`Coreset`] — the evidence was raced on a certified row coreset; the lead
///   must exceed the certificate's [`CoresetCertificate::race_transfer_margin`]
///   (#1012 contract — the SAME margin seam as the enclosure).
///
/// [`Exact`]: EvidenceCertification::Exact
/// [`Enclosure`]: EvidenceCertification::Enclosure
/// [`Coreset`]: EvidenceCertification::Coreset
/// [`block_preconditioned_logdet_enclosure`]: crate::logdet_bounds::block_preconditioned_logdet_enclosure
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvidenceCertification {
    Exact,
    Enclosure { gap: f64 },
    Coreset { certificate: CoresetCertificate },
}

impl EvidenceCertification {
    /// The smallest race lead Δ for which this candidate's evidence is
    /// trustworthy. Exact evidence transfers at any positive lead; an enclosure
    /// needs its gap; a coreset needs its certified transfer margin.
    pub fn required_margin(&self) -> f64 {
        match self {
            EvidenceCertification::Exact => 0.0,
            EvidenceCertification::Enclosure { gap } => *gap,
            EvidenceCertification::Coreset { certificate } => certificate.race_transfer_margin(),
        }
    }

    /// The unified `Verdict` (`gam_inference::certificates::Verdict`) for a race
    /// won by lead `race_lead` over the runner-up, on the shared certificate
    /// ladder (task #16). The race transfers its approximate-evidence verdict to
    /// the full corpus only when the lead strictly clears [`Self::required_margin`]
    /// — otherwise the consumer must refine or fall back to the exact dense path,
    /// so the verdict is `Insufficient`, never a silent pass. Exact evidence with
    /// any positive lead is `Certified`. A non-finite or non-positive lead leaves
    /// the race undecided (`Insufficient`); the per-family margin verdicts
    /// (`gam_inference::certificate_impls::coreset_race_verdict`,
    /// `gam_inference::certificate_impls::enclosure_margin_verdict`) supply
    /// the underlying mapping so there is one source of truth.
    pub fn race_verdict(&self, race_lead: f64) -> gam_problem::topology_certificates::Verdict {
        use gam_problem::topology_certificates::Verdict;
        if !(race_lead.is_finite() && race_lead > 0.0) {
            return Verdict::Insufficient;
        }
        match self {
            EvidenceCertification::Exact => Verdict::Certified,
            EvidenceCertification::Enclosure { gap } => {
                let enclosure = crate::logdet_bounds::LogdetEnclosure {
                    block_diag_logdet: 0.0,
                    lower: 0.0,
                    upper: *gap,
                    rho: 0.0,
                    p2: 0.0,
                    p3: None,
                };
                crate::inference::certificate_impls::enclosure_margin_verdict(&enclosure, race_lead)
            }
            EvidenceCertification::Coreset { certificate } => {
                crate::inference::certificate_impls::coreset_race_verdict(
                    certificate.certify_margin(race_lead),
                )
            }
        }
    }
}

/// One candidate entering the cross-class adjudicator: its kind, its rank-aware
/// Laplace negative-log-evidence (already computed on the common scale), how
/// that evidence was certified (for the margin contract), and a selection-time
/// held-out-density provider that refits per CV fold.
pub struct CrossClassCandidate<'a> {
    pub kind: AutoTopologyKind,
    pub negative_log_evidence: f64,
    /// Certification of `negative_log_evidence`. Defaults conceptually to
    /// [`EvidenceCertification::Exact`]; construct with [`Self::exact`] for the
    /// classic point-value path.
    pub certification: EvidenceCertification,
    pub density_provider: HeldOutDensityProvider<'a>,
}

impl<'a> CrossClassCandidate<'a> {
    /// Construct a candidate whose evidence is an exact point value (the
    /// classic full-corpus dense-logdet path — no margin floor).
    pub fn exact(
        kind: AutoTopologyKind,
        negative_log_evidence: f64,
        density_provider: HeldOutDensityProvider<'a>,
    ) -> Self {
        Self {
            kind,
            negative_log_evidence,
            certification: EvidenceCertification::Exact,
            density_provider,
        }
    }
}

/// Why a same-class race could not transfer its approximate-evidence verdict to
/// the full corpus: the winner's lead Δ over the runner-up did not clear the
/// required decision margin (the enclosure gap or the coreset transfer margin).
/// The consumer must refine (more moments / pair absorption / a larger coreset)
/// or re-run the top contenders on the exact dense path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InsufficientRaceMargin {
    /// Index of the provisional (below-margin) winner.
    pub provisional_winner: usize,
    /// Index of the runner-up whose evidence is within margin of the winner.
    pub contender: usize,
    /// The realized lead Δ = nle[contender] − nle[winner] (≥ 0).
    pub lead: f64,
    /// The margin the lead had to exceed (max of the two candidates'
    /// required margins).
    pub required_margin: f64,
}

/// Adjudicate a race that may mix smooth-manifold and discrete candidates
/// (the discrete-mixture rung and/or a structured union, #907). Cross-class
/// mixing is auto-detected from the candidate kinds (a race is cross-class iff
/// it contains BOTH at least one smooth/Euclidean candidate AND at least one
/// discrete candidate — [`AutoTopologyKind::Mixture`] or
/// [`AutoTopologyKind::Union`]). When cross-class,
/// the headline switches to stacking over a selection-time CV held-out
/// log-density table; otherwise the headline is the rank-aware evidence winner.
/// `seed` is mixed into the deterministic CV fold assignment (#1386): the same
/// seed reproduces the identical held-out folding, different seeds give
/// different — but still deterministic — foldings. It only affects the
/// cross-class (stacking) path; same-class races are winner-take-all on evidence
/// and ignore it. Pass [`STACKING_CV_SEED`] for the default folding.
pub fn adjudicate_cross_class_race(
    n: usize,
    candidates: Vec<CrossClassCandidate<'_>>,
    folds: usize,
    seed: u64,
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
        let certifications: Vec<EvidenceCertification> =
            candidates.iter().map(|c| c.certification).collect();
        let mut winner_index = 0usize;
        let mut best = f64::INFINITY;
        for (idx, &nle) in evidence.iter().enumerate() {
            if nle.is_finite() && nle < best {
                best = nle;
                winner_index = idx;
            }
        }
        // Decision-margin contract (#1011 enclosure / #1012 coreset, one seam):
        // the winner's lead over the closest contender must clear the larger of
        // the two candidates' required margins (an exact candidate floors at 0,
        // an enclosure at its gap, a coreset at its transfer margin). When the
        // lead is inside that margin the verdict is provisional — the
        // approximate evidence does not actually separate them — so we surface
        // an explicit escalation rather than silently anointing a winner the
        // bounds cannot distinguish.
        let mut insufficient_margin: Option<InsufficientRaceMargin> = None;
        for (idx, &nle) in evidence.iter().enumerate() {
            if idx == winner_index || !nle.is_finite() {
                continue;
            }
            let lead = nle - best;
            let required = certifications[winner_index]
                .required_margin()
                .max(certifications[idx].required_margin());
            if required > 0.0 && lead <= required {
                let tighter = insufficient_margin.map(|m| lead < m.lead).unwrap_or(true);
                if tighter {
                    insufficient_margin = Some(InsufficientRaceMargin {
                        provisional_winner: winner_index,
                        contender: idx,
                        lead,
                        required_margin: required,
                    });
                }
            }
        }
        return Ok(CrossClassRaceVerdict {
            candidate_names: names,
            is_cross_class: false,
            negative_log_evidence: evidence,
            stacking: None,
            winner_index,
            headline: Headline::Evidence,
            insufficient_margin,
        });
    }

    // Cross-class: build the selection-time held-out density table and stack.
    let providers: Vec<HeldOutDensityProvider<'_>> =
        candidates.into_iter().map(|c| c.density_provider).collect();
    let table = build_cv_log_density_table(n, folds, seed, &providers)?;
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
        // Cross-class headlines adjudicate by held-out predictive stacking on
        // the full corpus, not by the approximate-evidence scalar, so the
        // enclosure/coreset margin contract does not gate the verdict here.
        insufficient_margin: None,
    })
}

// ===========================================================================
// Closure-parameter smooth class (#1015): circle ⇄ interval as one estimand
// ===========================================================================

/// The deterministic closure-parameter grid the smooth-class profiler sweeps.
///
/// `γ = 1` is the closed circle, `γ = 0` the interval limit; the grid is dense
/// near both boundaries (where the Wilks crossing for the one-sided CI lives)
/// and is a fixed function of nothing but this constant, so the profile — and
/// thus the reported CI — is reproducible with no data-dependent node placement.
pub const CLOSURE_GAMMA_GRID: &[f64] = &[
    0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0,
];

/// One profiled point of the closure family: the closure value `γ`, the
/// profiled (θ and λ_smooth optimised) negative-log evidence at that γ, and the
/// fit handle the caller wants carried for the winner.
#[derive(Debug, Clone)]
pub struct ClosureProfilePoint<FitHandle> {
    pub gamma: f64,
    pub tk_score: f64,
    pub fit_handle: FitHandle,
}

/// The result of profiling the closure parameter inside the smooth class.
///
/// The headline is `gamma_hat` with its profile-likelihood CI (a regular Wilks
/// interval when the optimum is interior; one-sided when it touches a boundary).
/// `representative` is the γ̂ fit, scored on the SAME TK evidence surface the
/// discrete race uses, so this single smooth-class entry competes directly with
/// the non-homotopic candidates (mixture/union) the discrete race retains: the
/// closure family absorbs the within-smooth-class circle/line grid, it does not
/// replace the cross-class race.
#[derive(Debug, Clone)]
pub struct ClosureSelection<FitHandle> {
    pub ci: gam_geometry::ClosureProfileCi,
    pub representative: ClosureProfilePoint<FitHandle>,
    /// True when the profile pinned γ at the singular cluster boundary — the
    /// "not a regular smooth 1-D topology" signal that must be routed to the
    /// #907 mixture/union rung rather than reported as a regular closure.
    pub route_to_mixture_rung: bool,
}

/// Profile the closure parameter `γ`, returning the continuously refined
/// minimiser, its profile-likelihood CI, and the representative fit.
///
/// `fit_at_gamma` performs the inner fit at a fixed closure value and returns
/// `(tk_score, fit_handle)`, where `tk_score` is the profiled
/// negative-log evidence on the same scale [`tk_normalized_score`] produces (so
/// γ and λ_smooth must both be optimised inside the closure, per the issue's
/// confounding contract). Lower score is better. [`CLOSURE_GAMMA_GRID`] is
/// swept in parallel via [`run_topology_race_parallel`] to build the
/// reproducible profile curve for the Wilks CI; γ̂ itself is then refined by
/// golden-section minimization inside the bracketing interval, so the selected
/// closure is a continuous optimum, not a grid node.
pub fn profile_closure_within_smooth_class<FitHandle, FitAtGamma>(
    fit_at_gamma: FitAtGamma,
    level: f64,
) -> Result<ClosureSelection<FitHandle>, String>
where
    FitHandle: Send,
    FitAtGamma: Fn(f64) -> Result<(f64, FitHandle), String> + Sync,
{
    let gammas: Vec<f64> = CLOSURE_GAMMA_GRID.to_vec();
    let results = run_topology_race_parallel(gammas.clone(), |gamma| {
        fit_at_gamma(gamma).map(|(score, handle)| (gamma, score, handle))
    })?;

    let mut points: Vec<ClosureProfilePoint<FitHandle>> = Vec::with_capacity(results.len());
    for entry in results {
        let (gamma, tk_score, fit_handle) = entry.result?;
        if !tk_score.is_finite() {
            return Err(format!(
                "closure profile produced non-finite score at γ={gamma}"
            ));
        }
        points.push(ClosureProfilePoint {
            gamma,
            tk_score,
            fit_handle,
        });
    }
    if points.len() < 2 {
        return Err("closure profile needs at least two evaluable γ grid points".into());
    }
    points.sort_by(|a, b| a.gamma.partial_cmp(&b.gamma).expect("finite γ"));

    let grid: Vec<(f64, f64)> = points.iter().map(|p| (p.gamma, p.tk_score)).collect();
    let mut ci = gam_geometry::profile_ci_from_grid(&grid, level)?;

    // Pull the curve minimiser out of the points.
    let hat_index = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.tk_score.partial_cmp(&b.tk_score).expect("finite score"))
        .map(|(i, _)| i)
        .expect("non-empty points");
    let mut representative = points.swap_remove(hat_index);

    // Continuous γ̂ refinement (SPEC: grid search is never allowed). The fixed
    // grid supplies only the reproducible profile CURVE behind the Wilks CI;
    // the headline γ̂ is then refined by deterministic golden-section
    // minimization on the interval bracketing the curve minimum — the same
    // coarse-curve + golden-section pattern `fit_spline_scan` uses for log λ.
    // The best profiled evidence among the grid minimiser and the refinement
    // iterates wins, so refinement can only improve γ̂, and a boundary-pinned
    // minimiser (γ̂ ∈ {0, 1}) survives unless the interior genuinely beats it.
    const CLOSURE_GAMMA_TOL: f64 = 1e-3;
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut lo = grid[hat_index.saturating_sub(1)].0;
    let mut hi = grid[(hat_index + 1).min(grid.len() - 1)].0;
    if hi - lo > CLOSURE_GAMMA_TOL {
        let eval_at = |gamma: f64| -> Result<ClosureProfilePoint<FitHandle>, String> {
            let (tk_score, fit_handle) = fit_at_gamma(gamma)?;
            if !tk_score.is_finite() {
                return Err(format!(
                    "closure profile produced non-finite score at γ={gamma}"
                ));
            }
            Ok(ClosureProfilePoint {
                gamma,
                tk_score,
                fit_handle,
            })
        };
        let mut x1 = hi - INV_PHI * (hi - lo);
        let mut x2 = lo + INV_PHI * (hi - lo);
        let mut p1 = eval_at(x1)?;
        let mut p2 = eval_at(x2)?;
        while hi - lo > CLOSURE_GAMMA_TOL {
            if p1.tk_score > p2.tk_score {
                lo = x1;
                x1 = x2;
                p1 = p2;
                x2 = lo + INV_PHI * (hi - lo);
                p2 = eval_at(x2)?;
            } else {
                hi = x2;
                x2 = x1;
                p2 = p1;
                x1 = hi - INV_PHI * (hi - lo);
                p1 = eval_at(x1)?;
            }
        }
        for cand in [p1, p2] {
            if cand.tk_score < representative.tk_score {
                representative = cand;
            }
        }
        // Keep γ̂ and the representative fit coherent: the reported minimiser
        // is the refined one whenever refinement improved on the curve node.
        ci.gamma_hat = representative.gamma;
    }

    Ok(ClosureSelection {
        ci,
        representative,
        route_to_mixture_rung: ci.singular_boundary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    #[derive(Clone)]
    struct SyntheticRaceCandidate {
        seed: u64,
        len: usize,
    }

    fn synthetic_fit(candidate: SyntheticRaceCandidate) -> Vec<u64> {
        (0..candidate.len)
            .into_par_iter()
            .map(|i| {
                let x = candidate.seed ^ (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                x.rotate_left((i % 31) as u32)
                    .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            })
            .collect()
    }

    #[test]
    fn topology_race_parallel_matches_sequential_synthetic_candidates() {
        let candidates = vec![
            SyntheticRaceCandidate { seed: 11, len: 64 },
            SyntheticRaceCandidate { seed: 29, len: 64 },
            SyntheticRaceCandidate { seed: 47, len: 64 },
        ];
        let sequential = candidates
            .iter()
            .cloned()
            .map(synthetic_fit)
            .collect::<Vec<_>>();

        let parallel =
            run_topology_race_parallel_with_budget(candidates, synthetic_fit, 8).unwrap();
        assert_eq!(parallel.len(), 3);
        assert_eq!(
            parallel
                .iter()
                .map(|row| row.candidate_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(parallel.iter().all(|row| row.per_fit_threads == 1));
        let wall_times = parallel.iter().map(|row| row.wall_time).collect::<Vec<_>>();
        assert_eq!(wall_times.len(), 3);
        assert_eq!(
            parallel
                .into_iter()
                .map(|row| row.result)
                .collect::<Vec<_>>(),
            sequential
        );
    }

    fn trivial_provider<'a>() -> HeldOutDensityProvider<'a> {
        Box::new(|_train: &[usize], eval: &[usize]| Ok(vec![0.0; eval.len()]))
    }

    /// #1011/#1012 decision-margin contract on the same-class evidence race:
    /// when the winner's lead over the runner-up is inside the enclosure gap,
    /// the verdict is provisional (`insufficient_margin` set) so the caller must
    /// refine or escalate; a lead that clears the gap transfers cleanly.
    #[test]
    fn same_class_race_respects_enclosure_decision_margin() {
        // Two smooth candidates (same class) whose evidence came from a logdet
        // enclosure with gap 1.0. Lead of 0.5 < gap ⇒ provisional.
        let near = vec![
            CrossClassCandidate {
                kind: AutoTopologyKind::Circle,
                negative_log_evidence: 100.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
            CrossClassCandidate {
                kind: AutoTopologyKind::Euclidean,
                negative_log_evidence: 100.5,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
        ];
        let verdict = adjudicate_cross_class_race(
            8,
            near,
            STACKING_CV_FOLDS,
            STACKING_CV_SEED,
            StackingConfig::default(),
        )
        .expect("same-class race");
        assert!(!verdict.is_cross_class);
        assert_eq!(verdict.winner_index, 0);
        let escalation = verdict
            .insufficient_margin
            .expect("lead inside the enclosure gap must be flagged provisional");
        assert_eq!(escalation.provisional_winner, 0);
        assert_eq!(escalation.contender, 1);
        assert!((escalation.lead - 0.5).abs() < 1e-12);
        assert!((escalation.required_margin - 1.0).abs() < 1e-12);

        // A lead that clears the gap transfers the verdict cleanly.
        let far = vec![
            CrossClassCandidate {
                kind: AutoTopologyKind::Circle,
                negative_log_evidence: 100.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
            CrossClassCandidate {
                kind: AutoTopologyKind::Euclidean,
                negative_log_evidence: 105.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
        ];
        let verdict_far = adjudicate_cross_class_race(
            8,
            far,
            STACKING_CV_FOLDS,
            STACKING_CV_SEED,
            StackingConfig::default(),
        )
        .expect("same-class race");
        assert_eq!(verdict_far.winner_index, 0);
        assert!(
            verdict_far.insufficient_margin.is_none(),
            "a lead clearing the enclosure gap must transfer the verdict"
        );
    }

    /// The coreset transfer margin (#1012) flows through the SAME race seam: a
    /// lead inside `CoresetCertificate::race_transfer_margin` is provisional.
    #[test]
    fn same_class_race_respects_coreset_transfer_margin() {
        let cert = CoresetCertificate::new(0.05, 0.1, 32, 1000).expect("certificate");
        let required = cert.race_transfer_margin();
        // Lead strictly inside the certified transfer margin.
        let lead = 0.5 * required;
        let candidates = vec![
            CrossClassCandidate {
                kind: AutoTopologyKind::Circle,
                negative_log_evidence: 10.0,
                certification: EvidenceCertification::Coreset { certificate: cert },
                density_provider: trivial_provider(),
            },
            CrossClassCandidate {
                kind: AutoTopologyKind::Euclidean,
                negative_log_evidence: 10.0 + lead,
                certification: EvidenceCertification::Coreset { certificate: cert },
                density_provider: trivial_provider(),
            },
        ];
        let verdict = adjudicate_cross_class_race(
            8,
            candidates,
            STACKING_CV_FOLDS,
            STACKING_CV_SEED,
            StackingConfig::default(),
        )
        .expect("same-class race");
        let escalation = verdict
            .insufficient_margin
            .expect("lead inside the coreset transfer margin must be flagged");
        assert!((escalation.required_margin - required).abs() < 1e-9);
    }

    /// #1386: the `seed` mixed into the cross-class CV folding is functional, not
    /// a silent no-op. The same seed must reproduce the identical fold
    /// assignment, and two different seeds must produce a different assignment
    /// for at least one sample (while both remain deterministic). This test FAILS
    /// when the seed is ignored (a pure `i % folds` rule is seed-independent, so
    /// the two-seed inequality below can never hold) and PASSES once the seed is
    /// genuinely threaded into the fold rule.
    #[test]
    fn cv_folds_are_seed_reproducible_and_seed_varying() {
        const N: usize = 40;
        const FOLDS: usize = 5;

        // Flatten a partition into a per-sample fold-of-sample vector so two
        // foldings can be compared sample-by-sample regardless of fold order.
        fn fold_of_sample(n: usize, partition: &[(Vec<usize>, Vec<usize>)]) -> Vec<Option<usize>> {
            let mut assign = vec![None; n];
            for (fold, (_train, eval)) in partition.iter().enumerate() {
                for &i in eval {
                    assign[i] = Some(fold);
                }
            }
            assign
        }

        // (a) Reproducible: the same seed gives the identical fold assignment.
        let a1 = deterministic_cv_folds_seeded(N, FOLDS, 11);
        let a2 = deterministic_cv_folds_seeded(N, FOLDS, 11);
        assert_eq!(
            fold_of_sample(N, &a1),
            fold_of_sample(N, &a2),
            "same seed must reproduce the identical CV folding"
        );

        // (b) Seed-varying: two different seeds must differ for at least one
        // sample. If the seed were ignored this assertion could never pass.
        let b = deterministic_cv_folds_seeded(N, FOLDS, 12);
        assert_ne!(
            fold_of_sample(N, &a1),
            fold_of_sample(N, &b),
            "different seeds must produce different fold assignments (seed must \
             not be a no-op)"
        );

        // The default-seed convenience wrapper agrees with the explicit default
        // seed, so existing seed-less call sites keep their deterministic folding.
        assert_eq!(
            fold_of_sample(N, &deterministic_cv_folds(N, FOLDS)),
            fold_of_sample(
                N,
                &deterministic_cv_folds_seeded(N, FOLDS, STACKING_CV_SEED)
            ),
            "deterministic_cv_folds must equal the default-seeded folding"
        );
    }

    /// The unified certificate ladder (#16): `EvidenceCertification::race_verdict`
    /// maps the same margin contract onto `Verdict`. Exact transfers at any
    /// positive lead; an enclosure / coreset certifies only when the lead clears
    /// the required margin, else `Insufficient` (never a silent pass).
    #[test]
    fn race_verdict_maps_onto_unified_ladder() {
        use gam_problem::topology_certificates::Verdict;
        assert_eq!(
            EvidenceCertification::Exact.race_verdict(1e-6),
            Verdict::Certified
        );
        // Non-positive lead is undecided regardless of certification.
        assert_eq!(
            EvidenceCertification::Exact.race_verdict(0.0),
            Verdict::Insufficient
        );
        let enc = EvidenceCertification::Enclosure { gap: 0.2 };
        assert_eq!(enc.race_verdict(0.5), Verdict::Certified);
        assert_eq!(enc.race_verdict(0.1), Verdict::Insufficient);
        let cert = CoresetCertificate::new(0.05, 0.1, 32, 1000).expect("certificate");
        let required = cert.race_transfer_margin();
        let coreset = EvidenceCertification::Coreset { certificate: cert };
        assert_eq!(coreset.race_verdict(0.5 * required), Verdict::Insufficient);
        assert_eq!(
            coreset.race_verdict(2.0 * required + 1.0),
            Verdict::Certified
        );
    }

    #[test]
    fn closure_profiler_recovers_interior_minimum_and_ci() {
        // A planted parabolic profile in γ with minimum at 0.7: the closure
        // smooth class must recover γ̂ ≈ 0.7 with a CI that excludes both the
        // circle (γ=1) and the interval (γ=0) boundaries, and must NOT route to
        // the mixture rung (this is a regular interior optimum).
        let selection = profile_closure_within_smooth_class(
            |gamma| Ok::<_, String>((100.0 + 80.0 * (gamma - 0.7).powi(2), gamma)),
            0.95,
        )
        .expect("closure profile");
        assert!(
            (selection.ci.gamma_hat - 0.7).abs() < 0.06,
            "γ̂ {}",
            selection.ci.gamma_hat
        );
        assert!(!selection.ci.ci_includes_circle);
        assert!(!selection.ci.ci_includes_interval);
        assert!(!selection.route_to_mixture_rung);
        // The representative fit handle is the γ̂ point.
        assert!((selection.representative.gamma - selection.ci.gamma_hat).abs() < 1e-12);
    }

    #[test]
    fn closure_profiler_routes_collapse_to_mixture_rung() {
        // A profile that keeps improving toward γ=0 (support collapse) pins the
        // minimiser at the floor and must hand off to the mixture/union rung.
        let selection = profile_closure_within_smooth_class(
            |gamma| Ok::<_, String>((10.0 + 25.0 * gamma, gamma)),
            0.95,
        )
        .expect("closure profile");
        assert!(selection.ci.gamma_hat.abs() < 1e-9);
        assert!(selection.route_to_mixture_rung);
        assert!(selection.ci.ci_includes_interval);
    }

    #[test]
    fn topology_race_thread_plan_bounds_nested_rayon_threads() {
        let plan = TopologyRaceThreadPlan::for_budget(3, 8);
        assert_eq!(plan.concurrent_fits, 3);
        assert!(
            plan.coordinator_threads + plan.concurrent_fits * plan.per_fit_threads <= 8,
            "plan must bound coordinator plus per-fit Rayon workers"
        );

        let small = TopologyRaceThreadPlan::for_budget(3, 2);
        assert_eq!(small.concurrent_fits, 1);
        assert!(small.coordinator_threads + small.per_fit_threads <= 2);
    }

    // --- #944 stage-4 topology collapse tests --------------------------------

    /// Two fixed constant-curvature forms (Euclidean + Sphere) must be fused
    /// into a single estimated-κ ConstantCurvature candidate, at the position of
    /// the first fixed form. Non-CC candidates (Circle, Torus) keep their order.
    #[test]
    fn fuse_cc_family_collapses_euclidean_and_sphere() {
        let input = vec![
            AutoTopologyKind::Circle,
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Torus,
            AutoTopologyKind::Sphere,
        ];
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input);
        assert_eq!(
            fused,
            vec![
                AutoTopologyKind::Circle,
                AutoTopologyKind::ConstantCurvature, // replaced first fixed form
                AutoTopologyKind::Torus,
                // Sphere dropped — absorbed into ConstantCurvature
            ],
            "fused candidates: {fused:?}"
        );
    }

    /// A single fixed form alone must NOT be fused (nothing to estimate κ across).
    #[test]
    fn fuse_cc_family_leaves_single_form_intact() {
        let euclidean_only = vec![AutoTopologyKind::Euclidean, AutoTopologyKind::Circle];
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&euclidean_only);
        assert_eq!(fused, euclidean_only, "single fixed form must not be fused");

        let sphere_only = vec![AutoTopologyKind::Sphere];
        let fused2 = AutoTopologyKind::fuse_constant_curvature_family(&sphere_only);
        assert_eq!(fused2, sphere_only);
    }

    /// An explicit ConstantCurvature candidate alongside any fixed form fuses
    /// by dropping the fixed forms (the explicit CC already subsumes them).
    #[test]
    fn fuse_cc_family_explicit_cc_absorbs_fixed_forms() {
        let input = vec![
            AutoTopologyKind::ConstantCurvature,
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Circle,
        ];
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input);
        assert_eq!(
            fused,
            vec![
                AutoTopologyKind::ConstantCurvature,
                AutoTopologyKind::Circle
            ],
            "explicit CC must absorb the fixed Euclidean form"
        );
    }

    /// Idempotence: fusing an already-fused list leaves it unchanged.
    #[test]
    fn fuse_cc_family_is_idempotent() {
        let input = vec![
            AutoTopologyKind::Circle,
            AutoTopologyKind::ConstantCurvature,
            AutoTopologyKind::Torus,
        ];
        let once = AutoTopologyKind::fuse_constant_curvature_family(&input);
        let twice = AutoTopologyKind::fuse_constant_curvature_family(&once);
        assert_eq!(once, twice, "fuse must be idempotent");
        assert_eq!(once, input, "already-fused list must be unchanged");
    }

    /// Non-CC lists (no Euclidean/Sphere) are returned as-is.
    #[test]
    fn fuse_cc_family_noop_for_non_cc_list() {
        let input = vec![
            AutoTopologyKind::Circle,
            AutoTopologyKind::Torus,
            AutoTopologyKind::Cylinder,
        ];
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input);
        assert_eq!(fused, input);
    }
}
