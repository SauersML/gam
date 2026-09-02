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
//! discrete-mixture rung), or contains an adaptive order-selection class: the
//! HEADLINE ranking statistic switches to held-out predictive log-density /
//! stacking weights, with rank-aware Laplace evidence retained as
//! corroboration. Same-class races of fixed candidates keep winner-take-all
//! evidence behavior. The requirement is inferred from typed candidate kinds.
//!
//! What is still future work: persisting a mixture predictor for OOS prediction
//! on new rows. The selection-time stacking table is computed from fits that
//! exist only during the race; it is not retained for the saved-model
//! prediction path. That OOS-retention package is out of scope here.

use crate::evidence::{GaussianMixtureConfig, StackingConfig, StackingWeights, TopologyScoreScale, UnionStructure, UnionStructureFit, fit_gaussian_mixture, solve_stacking_weights};
use crate::priority_selection::{PriorityCandidate, rank_priority_candidates};
use crate::row_sampling_measure::CoresetCertificate;
use ndarray::{Array2, ArrayView2};
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
    /// Real projective plane represented by the even spherical harmonics on
    /// its antipodal `S2` cover.
    ProjectivePlane,
    /// Flat Klein bottle represented by the invariant Fourier modes on its
    /// two-torus cover.
    KleinBottle,
    /// Möbius band (#2240): the unique non-orientable smooth `d = 2`
    /// candidate. Genuinely non-homotopic to the torus / cylinder / sphere /
    /// patch, so it races as its own discrete candidate — no curvature fusion
    /// applies to it.
    Mobius,
    /// Flexible thin-plate (Duchon) 2-D sheet (#2240): topologically a plane,
    /// but with UNRESTRICTED embedding curvature — the chart for
    /// swiss-roll-class sheets a degree-2 flat patch cannot follow. It is not a
    /// fixed constant-curvature space form (its curvature is neither constant
    /// nor a single estimated κ), so the #944 Euclidean/Sphere fusion must not
    /// absorb it: it races as its own discrete candidate.
    DuchonSheet,
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
    /// `k` tight Gaussian clusters whose centers share one fitted circle
    /// (#2262). This is the density class for discrete cyclic concepts: it
    /// preserves clustered mass while pricing the common circular geometry.
    RingOfClusters {
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
    /// Coarse family tag for aggregate reports. This intentionally omits the
    /// fixed order of mixture/ring candidates; use [`Self::display_name`] for
    /// identity, serialization, diagnostics, and race columns.
    pub const fn family_tag(self) -> &'static str {
        match self {
            AutoTopologyKind::Euclidean => "euclidean",
            AutoTopologyKind::Circle => "circle",
            AutoTopologyKind::Sphere => "sphere",
            AutoTopologyKind::Torus => "torus",
            AutoTopologyKind::Cylinder => "cylinder",
            AutoTopologyKind::ProjectivePlane => "projective_plane",
            AutoTopologyKind::KleinBottle => "klein_bottle",
            AutoTopologyKind::Mobius => "mobius",
            AutoTopologyKind::DuchonSheet => "duchon_sheet",
            AutoTopologyKind::ConstantCurvature => "constant_curvature",
            AutoTopologyKind::Mixture { .. } => "mixture",
            AutoTopologyKind::RingOfClusters { .. } => "ring_clusters",
            AutoTopologyKind::Union { structure } => structure.as_str(),
        }
    }

    /// Owned display name including the mixture order, e.g. `"mixture_k7"`, or
    /// the union composite tag, e.g. `"union_circle+circle"`.
    pub fn display_name(self) -> String {
        match self {
            AutoTopologyKind::Mixture { k } => format!("mixture_k{k}"),
            AutoTopologyKind::RingOfClusters { k } => format!("ring_clusters_k{k}"),
            other => other.family_tag().to_string(),
        }
    }

    /// `true` iff this candidate is the discrete-mixture model class (as
    /// opposed to a smooth manifold / Euclidean latent topology).
    pub const fn is_discrete_mixture(self) -> bool {
        matches!(self, AutoTopologyKind::Mixture { .. })
    }

    pub const fn is_ring_of_clusters(self) -> bool {
        matches!(self, AutoTopologyKind::RingOfClusters { .. })
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
        self.is_discrete_mixture() || self.is_ring_of_clusters() || self.is_structured_union()
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        if let Some(rest) = value.strip_prefix("ring_clusters_k") {
            if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(format!(
                    "ring-of-clusters candidate must use ring_clusters_k{{n}}; got {value:?}"
                ));
            }
            let k: usize = rest
                .parse()
                .map_err(|_| format!("ring-of-clusters order is out of range; got {value:?}"))?;
            if k < 3 {
                return Err("ring-of-clusters order k must be >= 3".to_string());
            }
            if rest != k.to_string() {
                return Err(format!(
                    "ring-of-clusters candidate must use canonical ring_clusters_k{{n}} without leading zeroes; got {value:?}"
                ));
            }
            return Ok(AutoTopologyKind::RingOfClusters { k });
        }
        if value.starts_with("ring_clusters") {
            return Err(format!(
                "ring-of-clusters candidate must use ring_clusters_k{{n}}; got {value:?}"
            ));
        }
        if let Some(rest) = value.strip_prefix("mixture_k") {
            if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(format!(
                    "mixture candidate must use mixture_k{{n}}; got {value:?}"
                ));
            }
            let k: usize = rest
                .parse()
                .map_err(|_| format!("mixture order is out of range; got {value:?}"))?;
            if k == 0 {
                return Err("mixture order k must be >= 1".to_string());
            }
            if rest != k.to_string() {
                return Err(format!(
                    "mixture candidate must use canonical mixture_k{{n}} without leading zeroes; got {value:?}"
                ));
            }
            return Ok(AutoTopologyKind::Mixture { k });
        }
        if value.starts_with("mixture") {
            return Err(format!(
                "mixture candidate must use mixture_k{{n}}; got {value:?}"
            ));
        }
        if let Some(structure) = parse_union_name(value) {
            return Ok(AutoTopologyKind::Union { structure });
        }
        match value {
            "euclidean" => Ok(AutoTopologyKind::Euclidean),
            "circle" => Ok(AutoTopologyKind::Circle),
            "sphere" => Ok(AutoTopologyKind::Sphere),
            "torus" => Ok(AutoTopologyKind::Torus),
            "cylinder" => Ok(AutoTopologyKind::Cylinder),
            "projective_plane" => Ok(AutoTopologyKind::ProjectivePlane),
            "klein_bottle" => Ok(AutoTopologyKind::KleinBottle),
            "mobius" => Ok(AutoTopologyKind::Mobius),
            "duchon_sheet" => Ok(AutoTopologyKind::DuchonSheet),
            "constant_curvature" => Ok(AutoTopologyKind::ConstantCurvature),
            _ => Err(format!(
                "topology candidate must be an exact canonical name: euclidean, circle, sphere, torus, cylinder, projective_plane, klein_bottle, mobius, duchon_sheet, constant_curvature, mixture_k{{n}}, ring_clusters_k{{n}}, union_circle+circle, union_circle+cluster, or union_line+cluster; got {value:?}"
            )),
        }
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
    pub fn fuse_constant_curvature_family(
        candidates: &[Self],
        curvature_is_estimable: bool,
        subsumes: &[AutoTopologyKind],
    ) -> Vec<Self> {
        // The fusion asserts "within a candidate, curvature is estimated". If the
        // caller cannot estimate it, that assertion would be false, and the fixed
        // forms must race as themselves rather than be replaced by a name that
        // claims a fit nobody performed.
        if !curvature_is_estimable {
            return candidates.to_vec();
        }
        // Estimating κ is NECESSARY but not SUFFICIENT. This fusion DELETES the
        // fixed forms, so it is sound only for a form the caller's κ-fitting
        // candidate actually SPANS. `is_fixed_constant_curvature_form` answers
        // "is this a space form" — a statement about the GEOMETRY — and the two
        // questions come apart the moment a caller realizes two space forms in
        // different function spaces. gam-sae does exactly that: its `Euclidean`
        // is a monomial patch (which the κ-atom reproduces exactly at κ = 0) but
        // its `Sphere` is an AMBIENT HARMONIC basis, and a monomial patch in a
        // tangent chart does not span the `l = 2` harmonics at any κ. Fusing it
        // would shrink the race's span and report the loss as an estimated
        // curvature. So the caller declares which forms it is willing to have
        // deleted, and everything else races as itself.
        let is_fusible =
            |c: &AutoTopologyKind| c.is_fixed_constant_curvature_form() && subsumes.contains(c);
        let already_has_cc = candidates
            .iter()
            .any(|c| matches!(c, AutoTopologyKind::ConstantCurvature));
        let fixed_form_count = candidates.iter().filter(|c| is_fusible(c)).count();
        // Fuse when ≥2 fixed forms are present, OR when an explicit
        // ConstantCurvature candidate already subsumes ≥1 redundant fixed form.
        let should_fuse = fixed_form_count >= 2 || (already_has_cc && fixed_form_count >= 1);
        if !should_fuse {
            return candidates.to_vec();
        }
        let mut out = Vec::with_capacity(candidates.len());
        let mut emitted_cc = false;
        for &c in candidates {
            if is_fusible(&c) {
                // Replace the first SUBSUMED fixed form by the fitted-κ candidate
                // (unless an explicit one already exists); drop the rest. A fixed
                // form the caller did not declare subsumed falls through and races
                // as itself.
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
}

/// A predictive column in a cross-class race.
///
/// Fixed topology fits and adaptive model-selection procedures are different
/// statistical objects. In particular, an adaptive class chooses its order
/// independently inside every outer training fold; it must never alias a fixed
/// topology name or take the evidence-only same-class shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictiveCandidateKind {
    Fixed(AutoTopologyKind),
    MixtureClass,
    RingOfClustersClass,
}

impl PredictiveCandidateKind {
    /// Injective, stable identity for predictive columns and reports.
    pub fn display_name(self) -> String {
        match self {
            PredictiveCandidateKind::Fixed(kind) => kind.display_name(),
            PredictiveCandidateKind::MixtureClass => "mixture_class".to_string(),
            PredictiveCandidateKind::RingOfClustersClass => "ring_clusters_class".to_string(),
        }
    }

    /// Coarse density family used only where a report explicitly asks for a
    /// family label rather than a predictive-column identity.
    pub const fn family_tag(self) -> &'static str {
        match self {
            PredictiveCandidateKind::Fixed(kind) => kind.family_tag(),
            PredictiveCandidateKind::MixtureClass => "mixture",
            PredictiveCandidateKind::RingOfClustersClass => "ring_clusters",
        }
    }

    pub const fn is_discrete_class(self) -> bool {
        match self {
            PredictiveCandidateKind::Fixed(kind) => kind.is_discrete_class(),
            PredictiveCandidateKind::MixtureClass
            | PredictiveCandidateKind::RingOfClustersClass => true,
        }
    }

    /// Adaptive classes require honest outer-fold refitting even when every
    /// entry in the race is a discrete class.
    pub const fn requires_predictive_stacking(self) -> bool {
        matches!(
            self,
            PredictiveCandidateKind::MixtureClass | PredictiveCandidateKind::RingOfClustersClass
        )
    }

    pub const fn is_circular(self) -> bool {
        matches!(
            self,
            PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle)
                | PredictiveCandidateKind::Fixed(AutoTopologyKind::RingOfClusters { .. })
                | PredictiveCandidateKind::RingOfClustersClass
        )
    }
}

/// Parse one exact structured-union display name (#907).
pub fn parse_union_name(value: &str) -> Option<UnionStructure> {
    match value {
        "union_circle+circle" => Some(UnionStructure::CircleCircle),
        "union_circle+cluster" => Some(UnionStructure::CirclePointCluster),
        "union_line+cluster" => Some(UnionStructure::LineCluster),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct TopologyAutoSelector {
    pub candidates: Vec<AutoTopologyKind>,
    pub score_scale: TopologyScoreScale,
    /// Whether the CALLER can realize a candidate that actually fits the
    /// sectional curvature `κ`.
    ///
    /// The constant-curvature fusion replaces the fixed forms (`Euclidean` at
    /// `κ = 0`, `Sphere` at `κ > 0`) with one `ConstantCurvature` candidate on
    /// the promise that the consumer estimates `κ` inside it. That promise used
    /// to be assumed. A consumer that cannot keep it still received the fused
    /// candidate and had to answer for it — in practice by handing back a FIXED
    /// positively-curved fit under a name asserting curvature was estimated, so
    /// the race reported a κ it never fitted.
    ///
    /// Making it a declared capability means the lie is no longer expressible:
    /// a caller that cannot estimate `κ` races the fixed forms as themselves and
    /// reports `Euclidean` or `Sphere`, which is exactly what it measured.
    /// Default `false` — claiming the capability is opt-in.
    pub curvature_is_estimable: bool,
    /// The fixed space forms this caller's κ-fitting candidate actually SPANS,
    /// and is therefore willing to have DELETED by the fusion.
    ///
    /// Separate from [`Self::curvature_is_estimable`] because the two claims are
    /// different: estimability is about whether a κ is fitted, subsumption is
    /// about whether the fitted-κ basis reproduces the fixed form it replaces.
    /// A caller can honestly have the first and not the second — gam-sae fits κ
    /// on a monomial tangent patch, which reproduces its `Euclidean` candidate
    /// exactly at κ = 0 but never spans its ambient-harmonic `Sphere`.
    ///
    /// Defaults to both fixed forms, matching the pre-existing behaviour for
    /// callers that realize `Euclidean` and `Sphere` in one basis family.
    pub curvature_fusion_subsumes: &'static [AutoTopologyKind],
}

impl TopologyAutoSelector {
    /// A race is over an EXPLICIT candidate set.
    ///
    /// There used to be a `None` default backed by `AutoTopologyKind::all()`,
    /// which returned seven kinds — omitting `Mobius` and `DuchonSheet`, which
    /// the SAE race does realize, and `Mixture` / `RingOfClusters` / `Union`,
    /// which NO consumer realizes. It was therefore neither "all" nor "all
    /// raceable" but a hand-maintained subset with a name that claimed
    /// otherwise, and nothing ever called it: every real caller passes its own
    /// realizable set, because only the caller knows what it can realize.
    ///
    /// Requiring the set makes that explicit instead of offering a default that
    /// would be wrong for whoever first accepted it.
    pub fn new(candidates: Vec<AutoTopologyKind>) -> Self {
        Self {
            candidates,
            score_scale: TopologyScoreScale::PerEffectiveDim,
            curvature_is_estimable: false,
            curvature_fusion_subsumes: &[
                AutoTopologyKind::Euclidean,
                AutoTopologyKind::Sphere,
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TopologyAutoFitEvidence<FitHandle> {
    pub topology_name: String,
    pub raw_reml: f64,
    /// Forward-error bound on `raw_reml`, accumulated by whatever computed it
    /// (#2729). `None` states that the producer established no bound, and the
    /// selector then refuses to certify any margin against this candidate
    /// rather than treating the absence as an exact zero.
    pub raw_reml_roundoff: Option<f64>,
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
    /// The RESOLUTION of `tk_score`: the forward-error bound on `raw_reml`
    /// carried through exactly the normalization that produced `tk_score`
    /// (#2729). Two candidates whose scores differ by less than the sum of
    /// their resolutions are INDISTINGUISHABLE — the comparison between them is
    /// arithmetic debris, not evidence. `None` when the producer established no
    /// bound at all, which is a stronger statement than a large one: nothing is
    /// known about how many digits the comparison has.
    pub tk_score_resolution: Option<f64>,
    pub raw_reml: f64,
    pub effective_dim: f64,
    pub n_obs: usize,
    pub fit_handle: FitHandle,
}

#[derive(Debug, Clone)]
pub struct TopologyAutoSelectorResult<FitHandle> {
    pub ranked: Vec<TopologyAutoRankedFit<FitHandle>>,
    pub winner_index: usize,
    /// Every requested candidate that did not produce selectable, finite
    /// evidence. Failures remain part of the lifecycle result even when another
    /// candidate wins; callers must never reconstruct them from log strings.
    pub failed: Vec<TopologyAutoFailedCandidate>,
}

impl<FitHandle> TopologyAutoSelectorResult<FitHandle> {
    pub fn winner(&self) -> Option<&TopologyAutoRankedFit<FitHandle>> {
        self.ranked.get(self.winner_index)
    }

}

/// Stage at which a topology candidate became non-selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyCandidateFailureStage {
    Assembly,
    Fit,
    Evidence,
}

impl TopologyCandidateFailureStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assembly => "assembly",
            Self::Fit => "fit",
            Self::Evidence => "evidence",
        }
    }
}

/// Explicit record for a requested topology candidate that failed.
///
/// `evidence_at_failure` is populated when the fit converged but its evidence
/// metadata was invalid. A failed record is never ranked and is never silently
/// substituted by a cheaper or previously fitted candidate.
#[derive(Debug, Clone)]
pub struct TopologyAutoFailedCandidate {
    pub candidate: AutoTopologyKind,
    pub topology_name: String,
    pub stage: TopologyCandidateFailureStage,
    pub message: String,
    pub evidence_at_failure: Option<f64>,
}

/// Evidence family used to adjudicate a completed topology candidate fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologySelectionScoreKind {
    Reml,
    Laml,
    Bic,
    Tk,
}

impl TopologySelectionScoreKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reml => "reml",
            Self::Laml => "laml",
            Self::Bic => "bic",
            Self::Tk => "tk",
        }
    }
}

/// Scale applied after the candidate's raw evidence cost is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologySelectionScoreScale {
    Raw,
    PerObservation,
    PerEffectiveDim,
}

impl TopologySelectionScoreScale {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::PerObservation => "per_observation",
            Self::PerEffectiveDim => "per_effective_dim",
        }
    }
}

/// Typed metadata extracted from one completed candidate fit.
///
/// Optional fields are score-specific: LAML requires `laml`, BIC requires
/// `deviance`, and TK/LAML require `null_dim` (plus `null_space_logdet` when the
/// null dimension is non-zero). The lifecycle selector validates only the
/// requested headline score; unavailable secondary scores are omitted from the
/// disagreement diagnostic instead of changing candidate eligibility.
#[derive(Debug, Clone)]
pub struct TopologyCandidateEvidence {
    pub name: String,
    pub raw_reml: f64,
    pub laml: Option<f64>,
    pub deviance: Option<f64>,
    pub null_dim: Option<f64>,
    pub null_space_logdet: Option<f64>,
    pub effective_dim: f64,
    pub basis_size: usize,
    pub n_obs: usize,
}

/// One failed candidate supplied to, or produced by, the lifecycle selector.
#[derive(Debug, Clone)]
pub struct TopologyCandidateFailure {
    pub name: String,
    pub stage: TopologyCandidateFailureStage,
    pub error_type: String,
    pub message: String,
    pub evidence_at_failure: Option<f64>,
}

/// Exactly one terminal outcome for a requested topology candidate.
#[derive(Debug, Clone)]
pub enum TopologyCandidateOutcome {
    Fitted(TopologyCandidateEvidence),
    Failed(TopologyCandidateFailure),
}

impl TopologyCandidateOutcome {
    fn name(&self) -> &str {
        match self {
            Self::Fitted(evidence) => &evidence.name,
            Self::Failed(failure) => &failure.name,
        }
    }
}

/// A selectable candidate after score construction and validation.
#[derive(Debug, Clone)]
pub struct TopologyCandidateRanked {
    pub name: String,
    pub score: f64,
    pub raw_reml: f64,
    pub effective_dim: f64,
    pub basis_size: usize,
    pub n_obs: usize,
}

/// Complete candidate lifecycle: survivors, failures, winner, and diagnostics.
#[derive(Debug, Clone)]
pub struct TopologyCandidateSelectionResult {
    pub ranked: Vec<TopologyCandidateRanked>,
    pub winner_index: Option<usize>,
    pub failed: Vec<TopologyCandidateFailure>,
    pub warnings: Vec<String>,
}

fn failed_topology_summary(failed: &[TopologyAutoFailedCandidate]) -> String {
    failed
        .iter()
        .map(|failure| {
            format!(
                "{} [{}]: {}",
                failure.topology_name,
                failure.stage.as_str(),
                failure.message
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Result for one candidate executed by [`run_topology_race_parallel`].
#[derive(Debug, Clone)]
pub struct TopologyRaceParallelCandidate<FitResult> {
    /// Original position in the input candidate vector.
    pub candidate_index: usize,
    /// Diagnostic sizing metadata from `TopologyRaceThreadPlan::for_budget`
    /// (#2274 follow-up: no longer sizes a real per-candidate Rayon pool —
    /// each candidate runs on a plain OS thread and fans its own internal
    /// Rayon work into the shared global pool; see `run_one_topology_race_candidate`).
    pub per_fit_threads: usize,
    /// Wall-clock time spent running the candidate's fit body.
    pub wall_time: Duration,
    /// The fit closure's output. Use `FitResult = Result<T, E>` when individual
    /// candidate failures should be collected rather than short-circuiting.
    pub result: FitResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopologyRaceThreadPlan {
    per_fit_threads: usize,
    concurrent_fits: usize,
}

impl TopologyRaceThreadPlan {
    /// `per_fit_threads` is reporting metadata on the returned race rows, not
    /// a real Rayon pool size (#2274 follow-up — see
    /// `run_one_topology_race_candidate`): each candidate now runs on a plain
    /// OS thread and its own internal Rayon work fans into the shared global
    /// pool, so there is no per-candidate pool left to size. `concurrent_fits`
    /// remains the real knob — it still bounds how many candidate threads run
    /// simultaneously per batch. The one-slot-per-concurrent-candidate
    /// reservation this function used to attribute to a "coordinator" Rayon
    /// pool is folded directly into the `remaining` budget below (a scoped OS
    /// thread costs no Rayon pool slot, but keeping the reservation preserves
    /// the original conservative `per_fit_threads` sizing — halving it further
    /// would only be justified by evidence the old, more cautious number was
    /// ever a bottleneck, which it wasn't: the bottleneck was the nesting).
    fn for_budget(candidate_count: usize, max_total_threads: usize) -> Self {
        let max_total_threads = max_total_threads.max(1);
        if candidate_count <= 1 {
            return Self {
                per_fit_threads: max_total_threads,
                concurrent_fits: candidate_count,
            };
        }

        let concurrent_fits = if max_total_threads >= 4 {
            candidate_count.min(max_total_threads / 2).max(1)
        } else {
            1
        };
        let remaining = max_total_threads.saturating_sub(concurrent_fits);
        let per_fit_threads = if remaining == 0 {
            1
        } else {
            (remaining / concurrent_fits).max(1)
        };
        Self {
            per_fit_threads,
            concurrent_fits,
        }
    }
}

/// Run independent topology-race candidates concurrently on plain OS threads
/// (never nested Rayon pools).
///
/// Each candidate's fit runs on its own `std::thread`, NOT inside a per-candidate
/// `rayon::ThreadPool` (see the `#2274` follow-up note on
/// `run_one_topology_race_candidate` for why: a nested Rayon pool makes every
/// row-fan gate downstream — `SaeManifoldTerm::loss_scaled`,
/// `CpuBatchedBlockSolver::factor_blocks`, the reduced-Schur row loops — read
/// `rayon::current_thread_index().is_some()` and silently fall back to a fully
/// sequential per-row loop, discarding whatever thread budget the nested pool
/// was given). A plain OS thread is not a Rayon worker of anything, so those
/// gates see `None` and fan into the process's one shared global Rayon pool
/// exactly as they would for an ordinary top-level call; when multiple
/// candidates race concurrently they each submit to that same global pool,
/// which is precisely the many-producer-threads-one-pool pattern Rayon's
/// work-stealing scheduler is built to host (and it can rebalance dynamically
/// as candidates finish at different times, which a static per-candidate
/// thread slice could not). For multi-candidate races the runner still batches
/// candidates (via `TopologyRaceThreadPlan::for_budget`'s `concurrent_fits`) so
/// the number of simultaneously-live candidate threads stays bounded by
/// `std::thread::available_parallelism()`; `per_fit_threads` is retained purely
/// as reporting metadata on the returned rows (existing callers/tests read it)
/// and no longer sizes a real pool.
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
        // #2274 follow-up — batch concurrent candidates over plain OS threads
        // (`std::thread::scope`), NOT a `rayon::ThreadPool` "coordinator" pool.
        // The coordinator pool previously existed only to host `scope.spawn`
        // for each candidate; a scoped OS thread does the same job (bounded
        // concurrency within a batch, borrow-checked access to `candidates`/
        // `slots`/`pool_error`) without making the candidates' OWN internal
        // Rayon work nested under it. See `run_one_topology_race_candidate`
        // for why that nesting was the root cause of the row-fan collapse.
        let mut batch_start = 0usize;
        while batch_start < candidate_count {
            let batch_end = (batch_start + plan.concurrent_fits).min(candidate_count);
            std::thread::scope(|scope| {
                for idx in batch_start..batch_end {
                    let candidate = candidates[idx]
                        .take()
                        .expect("topology race candidate must be present");
                    let slot = &slots[idx];
                    let pool_error = &pool_error;
                    let fit_one = &fit_one;
                    scope.spawn(move || {
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

/// Extract a readable message from a caught `std::thread` panic payload,
/// mirroring `gam_pyffi::ffi::ffi_errors::panic_payload_message`'s handling of
/// the two payload shapes `std::panic!`/`unwrap`/`expect` actually produce.
fn topology_race_panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
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
    let started = Instant::now();
    // #2274 follow-up (root cause of the E1 SAE inner solve's 2/30-core
    // ceiling: `crates/gam-sae/src/manifold/stagewise.rs`'s birth race had the
    // SAME disease one layer up — a `seeds.par_iter()` fan running each
    // candidate's ENTIRE inner solve nested inside a Rayon worker, which
    // silently disabled every downstream CPU row-fan gate). This function had
    // the identical bug: `rayon::ThreadPoolBuilder::new().num_threads(per_fit_threads)`
    // + `pool.install(|| ...)` unconditionally makes the executing frame
    // report `rayon::current_thread_index() == Some(_)` (relative to that
    // fresh pool) for the ENTIRE nested call into `fit_one` — regardless of
    // how many threads the pool was built with. Every row-fan gate downstream
    // (`SaeManifoldTerm::loss_scaled`, `CpuBatchedBlockSolver::factor_blocks`,
    // the reduced-Schur row loops in reduced_solve.rs/system.rs/newton_step.rs)
    // reads exactly that condition to decide "am I nested, fall back to
    // sequential" — so `per_fit_threads` was NEVER actually available to those
    // gates; they always took the single-threaded fallback.
    //
    // Fix: run `fit_one` on a plain, non-Rayon OS thread instead. A
    // `std::thread` is not a worker of any Rayon pool, so
    // `current_thread_index()` reports `None` inside it, exactly like an
    // ordinary top-level (non-raced) call — every downstream row-fan gate
    // takes its real parallel branch, fanning into the process's one shared
    // global Rayon pool (the pool those gates were written assuming). When
    // several candidates race concurrently (the `concurrent_fits > 1` caller),
    // each gets its own OS thread and independently submits parallel work to
    // that SAME global pool: Rayon's work-stealing scheduler is explicitly
    // built to host many concurrent producer threads against one pool and
    // fairly interleaves their submissions, rebalancing dynamically as each
    // candidate's row-fan finishes at a different wall-clock time — something
    // a static per-candidate thread slice could never do. `per_fit_threads` is
    // kept as reporting/diagnostic metadata on the returned row (existing
    // callers/tests read it); it no longer sizes a real pool.
    //
    // Cannot deadlock: no nested Rayon pool is created here anymore, so the
    // #2074 class this function's old comment warned about (faer's `spindle`
    // barrier waiting on pool slots an outer fan-out already holds) cannot
    // occur — `with_faer_sequential` below still pins faer's own internal
    // solvers to `Par::Seq` for defense in depth, but there is no longer any
    // Rayon-in-Rayon nesting for that barrier to even be reached through. Nor
    // does this touch the OnceLock×Rayon deadlock class: a plain `std::thread`
    // does no lazy static init of its own, and the GPU-resident caller
    // (`run_resident_fits_multiplexed_with`) already documents that its
    // `OnceLock`s are warmed BEFORE any candidate here runs.
    let outcome = std::thread::scope(|scope| {
        scope
            .spawn(move || gam_linalg::faer_ndarray::with_faer_sequential(|| fit_one(candidate)))
            .join()
    });
    let wall_time = started.elapsed();
    match outcome {
        Ok(result) => {
            *slot.lock().expect("topology race result mutex poisoned") =
                Some(TopologyRaceParallelCandidate {
                    candidate_index,
                    per_fit_threads,
                    wall_time,
                    result,
                });
        }
        Err(payload) => {
            *pool_error.lock().expect("pool_error mutex poisoned") = Some(format!(
                "topology race candidate {candidate_index} panicked: {}",
                topology_race_panic_message(payload)
            ));
        }
    }
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
    let fused = AutoTopologyKind::fuse_constant_curvature_family(
        &selector.candidates,
        selector.curvature_is_estimable,
        selector.curvature_fusion_subsumes,
    );
    let mut ranked = Vec::with_capacity(fused.len());
    let mut failed = Vec::new();
    for candidate in &fused {
        match fit_one(*candidate) {
            Ok(evidence) => {
                let (tk_score, tk_score_resolution) = match tk_normalized_score_with_resolution(
                    evidence.raw_reml,
                    evidence.raw_reml_roundoff,
                    evidence.null_dim,
                    evidence.null_space_logdet,
                    evidence.effective_dim,
                    evidence.n_obs,
                    selector.score_scale,
                ) {
                    Ok(scored) => scored,
                    Err(message) => {
                        failed.push(TopologyAutoFailedCandidate {
                            candidate: *candidate,
                            topology_name: evidence.topology_name,
                            stage: TopologyCandidateFailureStage::Evidence,
                            message,
                            evidence_at_failure: evidence
                                .raw_reml
                                .is_finite()
                                .then_some(evidence.raw_reml),
                        });
                        continue;
                    }
                };
                ranked.push(TopologyAutoRankedFit {
                    topology_name: evidence.topology_name,
                    tk_score,
                    tk_score_resolution,
                    raw_reml: evidence.raw_reml,
                    effective_dim: evidence.effective_dim,
                    n_obs: evidence.n_obs,
                    fit_handle: evidence.fit_handle,
                });
            }
            Err(err) => failed.push(TopologyAutoFailedCandidate {
                candidate: *candidate,
                topology_name: candidate.display_name(),
                stage: TopologyCandidateFailureStage::Fit,
                message: err.to_string(),
                evidence_at_failure: None,
            }),
        }
    }
    if ranked.is_empty() {
        return Err(format!(
            "TopologyAutoSelector found no fittable topology candidates{}",
            if failed.is_empty() {
                String::new()
            } else {
                format!(" ({})", failed_topology_summary(&failed))
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
        failed,
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
    tk_normalized_score_with_resolution(
        raw_reml,
        None,
        null_dim,
        null_space_logdet,
        effective_dim,
        n_obs,
        score_scale,
    )
    .map(|(score, _)| score)
}

/// [`tk_normalized_score`] carrying the score's own numerical RESOLUTION
/// through the same normalization (#2729).
///
/// `raw_reml_roundoff` is the forward-error bound the evidence producer
/// accumulated for `raw_reml`. It is a bound on an ABSOLUTE error, so the TK
/// normalizer's own rounding is added to it in absolute terms and the common
/// divisor scales it exactly as it scales the score — no tolerance is chosen
/// anywhere. `None` in propagates to `None` out: an unmeasured resolution must
/// not become a zero one.
pub fn tk_normalized_score_with_resolution(
    raw_reml: f64,
    raw_reml_roundoff: Option<f64>,
    null_dim: f64,
    null_space_logdet: Option<f64>,
    effective_dim: f64,
    n_obs: usize,
    score_scale: TopologyScoreScale,
) -> Result<(f64, Option<f64>), String> {
    let normalizer = topology_tk_normalizer(Some(null_dim), null_space_logdet)?;
    let tk = raw_reml + normalizer;
    // The normalizer is itself a rounded sum of two products; the addition that
    // folds it into `raw_reml` rounds on the magnitude of the result.
    let unit_roundoff = 0.5 * f64::EPSILON;
    let tk_roundoff = raw_reml_roundoff
        .map(|bound| bound + unit_roundoff * (3.0 * normalizer.abs() + tk.abs()));
    let scale = match score_scale {
        TopologyScoreScale::PerObservation => {
            if n_obs == 0 {
                return Err("TopologyAutoSelector requires n_obs > 0".to_string());
            }
            n_obs as f64
        }
        TopologyScoreScale::PerEffectiveDim => {
            if !(effective_dim.is_finite() && effective_dim > 0.0) {
                return Err(
                    "TopologyAutoSelector requires finite positive effective_dim".to_string()
                );
            }
            effective_dim
        }
    };
    let score = tk / scale;
    let resolution = tk_roundoff.map(|bound| bound / scale + unit_roundoff * score.abs());
    Ok((score, resolution))
}

fn topology_tk_normalizer(
    null_dim: Option<f64>,
    null_space_logdet: Option<f64>,
) -> Result<f64, String> {
    let null_dim = null_dim.ok_or_else(|| {
        "topology evidence requires null-dimension metadata for TK normalization".to_string()
    })?;
    if !null_dim.is_finite() || null_dim < -1.0e-9 {
        return Err("topology evidence null dimension must be finite and non-negative".to_string());
    }
    if null_dim.max(0.0) == 0.0 {
        return Ok(0.0);
    }
    let logdet = null_space_logdet.ok_or_else(|| {
        "topology evidence TK normalizer requires null-space Hessian logdet".to_string()
    })?;
    if !logdet.is_finite() {
        return Err("topology evidence null-space Hessian logdet must be finite".to_string());
    }
    Ok(-0.5 * null_dim.max(0.0) * TK_LOG_2PI + 0.5 * logdet)
}

fn topology_candidate_raw_score(
    evidence: &TopologyCandidateEvidence,
    score_kind: TopologySelectionScoreKind,
) -> Result<f64, String> {
    if !evidence.effective_dim.is_finite() {
        return Err(format!(
            "candidate {:?} has non-finite effective_dim {:?}",
            evidence.name, evidence.effective_dim
        ));
    }
    if evidence.n_obs == 0 {
        return Err(format!("candidate {:?} requires n_obs > 0", evidence.name));
    }
    if !evidence.raw_reml.is_finite() {
        return Err(format!(
            "candidate {:?} has non-finite REML evidence {:?}",
            evidence.name, evidence.raw_reml
        ));
    }
    match score_kind {
        TopologySelectionScoreKind::Reml => Ok(evidence.raw_reml),
        TopologySelectionScoreKind::Tk => Ok(evidence.raw_reml
            + topology_tk_normalizer(evidence.null_dim, evidence.null_space_logdet)?),
        TopologySelectionScoreKind::Laml => {
            let laml = evidence.laml.ok_or_else(|| {
                format!(
                    "candidate {:?} is missing LAML evidence metadata",
                    evidence.name
                )
            })?;
            if !laml.is_finite() {
                return Err(format!(
                    "candidate {:?} has non-finite LAML evidence {laml:?}",
                    evidence.name
                ));
            }
            Ok(laml + topology_tk_normalizer(evidence.null_dim, evidence.null_space_logdet)?)
        }
        TopologySelectionScoreKind::Bic => {
            let deviance = evidence.deviance.ok_or_else(|| {
                format!(
                    "candidate {:?} is missing deviance metadata required for BIC",
                    evidence.name
                )
            })?;
            bic_score(deviance, evidence.n_obs, evidence.basis_size)
        }
    }
}

fn scale_topology_candidate_score(
    score: f64,
    scale: TopologySelectionScoreScale,
    evidence: &TopologyCandidateEvidence,
) -> Result<f64, String> {
    if !score.is_finite() {
        return Err(format!(
            "candidate {:?} has non-finite selected evidence {score:?}",
            evidence.name
        ));
    }
    match scale {
        TopologySelectionScoreScale::Raw => Ok(score),
        TopologySelectionScoreScale::PerObservation => {
            if evidence.n_obs == 0 {
                Err(format!(
                    "candidate {:?} requires n_obs > 0 for per-observation scoring",
                    evidence.name
                ))
            } else {
                Ok(score / evidence.n_obs as f64)
            }
        }
        TopologySelectionScoreScale::PerEffectiveDim => {
            if !(evidence.effective_dim.is_finite() && evidence.effective_dim > 0.0) {
                Err(format!(
                    "candidate {:?} requires finite positive effective_dim for per-effective-dimension scoring; got {:?}",
                    evidence.name, evidence.effective_dim
                ))
            } else {
                Ok(score / evidence.effective_dim)
            }
        }
    }
}

fn topology_candidate_score(
    evidence: &TopologyCandidateEvidence,
    score_kind: TopologySelectionScoreKind,
    score_scale: TopologySelectionScoreScale,
) -> Result<f64, String> {
    let raw = topology_candidate_raw_score(evidence, score_kind)?;
    scale_topology_candidate_score(raw, score_scale, evidence)
}

/// Resolve every declared candidate outcome through one evidence-validation,
/// failure-policy, deterministic-ranking, and winner-finalization entry.
///
/// Callers perform topology-specific assembly and invoke the model fitter, then
/// submit exactly one terminal outcome per declared name. A malformed lifecycle
/// (empty request or duplicate name) is rejected. Candidate-local evidence
/// errors become typed `Evidence` failures so one bad fit cannot erase the
/// other requested outcomes.
pub fn select_topology_candidate_lifecycle(
    outcomes: Vec<TopologyCandidateOutcome>,
    score_kind: TopologySelectionScoreKind,
    score_scale: TopologySelectionScoreScale,
) -> Result<TopologyCandidateSelectionResult, String> {
    if outcomes.is_empty() {
        return Err("topology selection requires at least one candidate outcome".to_string());
    }
    let mut names = std::collections::BTreeSet::new();
    for outcome in &outcomes {
        let name = outcome.name();
        if name.is_empty() {
            return Err("topology candidate names cannot be empty".to_string());
        }
        if !names.insert(name.to_string()) {
            return Err(format!("duplicate topology candidate {name:?}"));
        }
    }

    let mut evidence_survivors = Vec::new();
    let mut ranked = Vec::new();
    let mut failed = Vec::new();
    for (candidate_index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            TopologyCandidateOutcome::Failed(failure) => failed.push(failure),
            TopologyCandidateOutcome::Fitted(evidence) => {
                match topology_candidate_score(&evidence, score_kind, score_scale) {
                    Ok(score) => {
                        ranked.push(PriorityCandidate::new(
                            TopologyCandidateRanked {
                                name: evidence.name.clone(),
                                score,
                                raw_reml: evidence.raw_reml,
                                effective_dim: evidence.effective_dim,
                                basis_size: evidence.basis_size,
                                n_obs: evidence.n_obs,
                            },
                            candidate_index,
                            score,
                            0,
                        ));
                        evidence_survivors.push(evidence);
                    }
                    Err(message) => failed.push(TopologyCandidateFailure {
                        name: evidence.name,
                        stage: TopologyCandidateFailureStage::Evidence,
                        error_type: "gam_solve::topology_selector::EvidenceValidationError"
                            .to_string(),
                        message,
                        evidence_at_failure: evidence
                            .raw_reml
                            .is_finite()
                            .then_some(evidence.raw_reml),
                    }),
                }
            }
        }
    }
    let ranked: Vec<TopologyCandidateRanked> = rank_priority_candidates(ranked)
        .into_iter()
        .map(|candidate| candidate.item)
        .collect();
    let warnings = topology_score_disagreement_warnings(&evidence_survivors, score_scale);
    Ok(TopologyCandidateSelectionResult {
        winner_index: (!ranked.is_empty()).then_some(0),
        ranked,
        failed,
        warnings,
    })
}

fn topology_score_disagreement_warnings(
    evidence: &[TopologyCandidateEvidence],
    score_scale: TopologySelectionScoreScale,
) -> Vec<String> {
    let mut orders = Vec::new();
    for kind in [
        TopologySelectionScoreKind::Reml,
        TopologySelectionScoreKind::Laml,
        TopologySelectionScoreKind::Bic,
    ] {
        let scored: Result<Vec<_>, _> = evidence
            .iter()
            .enumerate()
            .map(|(index, row)| {
                topology_candidate_score(row, kind, score_scale)
                    .map(|score| PriorityCandidate::new(row.name.clone(), index, score, 0))
            })
            .collect();
        let Ok(scored) = scored else {
            continue;
        };
        let order: Vec<String> = rank_priority_candidates(scored)
            .into_iter()
            .map(|row| row.item)
            .collect();
        orders.push((kind, order));
    }
    if orders.len() < 2 || orders.windows(2).all(|pair| pair[0].1 == pair[1].1) {
        return Vec::new();
    }
    let detail = orders
        .iter()
        .map(|(kind, order)| format!("{}: {}", kind.as_str(), order.join(", ")))
        .collect::<Vec<_>>()
        .join("; ");
    if score_scale == TopologySelectionScoreScale::Raw {
        vec![format!(
            "Topology score rankings differ across score kinds ({detail}). BIC and REML can disagree when candidate basis sizes differ wildly."
        )]
    } else {
        vec![format!(
            "Scaled topology score rankings still differ across score kinds under score_scale={:?} ({detail}). Treat BIC as a secondary diagnostic; the Tierney-Kadane Laplace normalizer handles the known cross-basis evidence scale issue.",
            score_scale.as_str()
        )]
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
/// fitted Gaussian mixture, and its BIC-form negative log evidence. Lower is
/// better.
#[derive(Debug, Clone)]
pub struct MixtureRungFit {
    pub k: usize,
    pub fit: crate::evidence::GaussianMixtureFit,
    /// Free-parameter count `P` in the `P log(n) / 2` BIC price.
    pub num_parameters: usize,
    /// `-loglik + P log(n) / 2`, on the smooth parametric candidates' scale.
    pub bic: f64,
}

/// Result of fitting the whole mixture ladder: every fitted order plus the index
/// of the in-class winner (lowest BIC).
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

/// Which adaptive discrete class produced a rung error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveRungKind {
    GaussianMixture,
    RingOfClusters,
}

impl AdaptiveRungKind {
    const fn display_name(self) -> &'static str {
        match self {
            Self::GaussianMixture => "Gaussian-mixture",
            Self::RingOfClusters => "ring-of-clusters",
        }
    }
}

/// Stage at which one eligible adaptive-rung order failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveRungFailureStage {
    Fit,
    Evidence,
}

/// Failure of one specific, eligible order in an adaptive rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveRungOrderFailure {
    pub k: usize,
    pub stage: AdaptiveRungFailureStage,
    pub message: String,
}

/// A fail-closed adaptive-rung refusal.
///
/// Order selection is only meaningful when all evidence values being compared
/// came from certified fits. A failed order is therefore surfaced, even if a
/// different order fitted successfully; choosing among the survivors would
/// silently change the estimand. Likewise, hitting the local-refinement budget
/// is an explicit refusal rather than an unbracketed best-so-far result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveRungError {
    InvalidInput {
        kind: AdaptiveRungKind,
        message: String,
    },
    OrderFailures {
        kind: AdaptiveRungKind,
        failures: Vec<AdaptiveRungOrderFailure>,
    },
    RefinementBudgetExhausted {
        kind: AdaptiveRungKind,
        best_k: usize,
        completed_probes: usize,
    },
}

impl std::fmt::Display for AdaptiveRungError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { kind, message } => {
                write!(formatter, "invalid {} rung: {message}", kind.display_name())
            }
            Self::OrderFailures { kind, failures } => {
                write!(
                    formatter,
                    "{} rung refused because {} eligible order(s) failed",
                    kind.display_name(),
                    failures.len()
                )?;
                for failure in failures {
                    write!(
                        formatter,
                        "; k={} {:?}: {}",
                        failure.k, failure.stage, failure.message
                    )?;
                }
                Ok(())
            }
            Self::RefinementBudgetExhausted {
                kind,
                best_k,
                completed_probes,
            } => write!(
                formatter,
                "{} rung exhausted its {completed_probes}-probe refinement budget before bracketing k={best_k}",
                kind.display_name()
            ),
        }
    }
}

impl std::error::Error for AdaptiveRungError {}

fn eligible_adaptive_orders(
    kind: AdaptiveRungKind,
    ladder: &[usize],
    minimum_order: usize,
    n: usize,
) -> Result<Vec<usize>, AdaptiveRungError> {
    if ladder.is_empty() {
        return Err(AdaptiveRungError::InvalidInput {
            kind,
            message: "order ladder must not be empty".to_string(),
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut orders = Vec::with_capacity(ladder.len());
    for &k in ladder {
        if k < minimum_order || k > n {
            return Err(AdaptiveRungError::InvalidInput {
                kind,
                message: format!(
                    "requested order k={k} is outside this class on n={n} rows; require {minimum_order} <= k <= n"
                ),
            });
        }
        if !seen.insert(k) {
            return Err(AdaptiveRungError::InvalidInput {
                kind,
                message: format!("order ladder contains duplicate k={k}"),
            });
        }
        orders.push(k);
    }
    Ok(orders)
}

/// Hard cap on the number of EXTRA orders the local refinement around the
/// coarse-ladder winner may probe. Refinement walks one neighbour at a time
/// and stops as soon as the running winner is bracketed (both immediate
/// neighbours fitted and worse). If the cap binds, the rung returns
/// [`AdaptiveRungError::RefinementBudgetExhausted`]; it never treats an
/// unbracketed best-so-far order as a certified winner.
pub const MIXTURE_REFINEMENT_MAX_PROBES: usize = 16;

/// Fit the free-cluster adaptive class. Unlike the general Gaussian-mixture
/// rung, this class owns only orders `k >= 2`: the one-component full Gaussian
/// is the Euclidean candidate. The minimum is enforced inside refinement, so a
/// coarse `k = 2` winner can never be bracketed by an ineligible `k = 1` fit.
pub fn fit_free_cluster_rung(
    data: ArrayView2<'_, f64>,
    ladder: &[usize],
    config: GaussianMixtureConfig,
) -> Result<MixtureRungResult, AdaptiveRungError> {
    fit_mixture_rung_with_minimum_order(data, ladder, 2, config)
}

fn fit_mixture_rung_with_minimum_order(
    data: ArrayView2<'_, f64>,
    ladder: &[usize],
    minimum_order: usize,
    config: GaussianMixtureConfig,
) -> Result<MixtureRungResult, AdaptiveRungError> {
    let kind = AdaptiveRungKind::GaussianMixture;
    let n = data.nrows();
    let coarse_orders = eligible_adaptive_orders(kind, ladder, minimum_order, n)?;
    let mut fits: Vec<MixtureRungFit> = Vec::new();
    // Every order ever attempted (fitted OR failed): refinement must not
    // re-propose a failed order forever.
    let mut attempted: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    let try_order = |k: usize,
                     fits: &mut Vec<MixtureRungFit>,
                     attempted: &mut std::collections::BTreeSet<usize>|
     -> Option<AdaptiveRungOrderFailure> {
        assert!(k >= minimum_order && k <= n && !attempted.contains(&k));
        attempted.insert(k);
        match fit_gaussian_mixture(data, k, config) {
            Ok(fit) => {
                let num_parameters = fit.num_free_parameters();
                let bic = fit.bic();
                if bic.is_finite() {
                    fits.push(MixtureRungFit {
                        k,
                        fit,
                        num_parameters,
                        bic,
                    });
                    None
                } else {
                    Some(AdaptiveRungOrderFailure {
                        k,
                        stage: AdaptiveRungFailureStage::Evidence,
                        message: "BIC is not finite".to_string(),
                    })
                }
            }
            Err(error) => Some(AdaptiveRungOrderFailure {
                k,
                stage: AdaptiveRungFailureStage::Fit,
                message: error.to_string(),
            }),
        }
    };

    let mut failures = Vec::new();
    for k in coarse_orders {
        if let Some(failure) = try_order(k, &mut fits, &mut attempted) {
            failures.push(failure);
        }
    }
    if !failures.is_empty() {
        return Err(AdaptiveRungError::OrderFailures { kind, failures });
    }

    // Local refinement: bracket the running winner. The running winner uses
    // the same rule as the final ranking (lower BIC, ties to the smaller k), so
    // refinement and ranking can never disagree about who the winner is.
    let mut probes = 0usize;
    loop {
        let Some(best_k) = fits
            .iter()
            .min_by(|a, b| a.bic.total_cmp(&b.bic).then(a.k.cmp(&b.k)))
            .map(|f| f.k)
        else {
            return Err(AdaptiveRungError::InvalidInput {
                kind,
                message: "no eligible order produced a fit".to_string(),
            });
        };
        let next = [best_k.checked_sub(1), best_k.checked_add(1)]
            .into_iter()
            .flatten()
            .find(|&k| k >= minimum_order && k <= n && !attempted.contains(&k));
        let Some(k) = next else {
            break; // bracketed: both neighbours attempted (or out of range).
        };
        if probes == MIXTURE_REFINEMENT_MAX_PROBES {
            return Err(AdaptiveRungError::RefinementBudgetExhausted {
                kind,
                best_k,
                completed_probes: probes,
            });
        }
        if let Some(failure) = try_order(k, &mut fits, &mut attempted) {
            return Err(AdaptiveRungError::OrderFailures {
                kind,
                failures: vec![failure],
            });
        }
        probes += 1;
    }
    // In-class winner-take-all on the BIC scale (lower wins).
    let ranked = rank_priority_candidates(
        fits.into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.bic;
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

/// One order of the constrained ring-of-clusters rung (#2262).
#[derive(Debug, Clone)]
pub struct RingOfClustersRungFit {
    pub k: usize,
    pub fit: crate::evidence::RingGaussianMixtureFit,
    pub num_parameters: usize,
    pub bic: f64,
}

#[derive(Debug, Clone)]
pub struct RingOfClustersRungResult {
    pub fits: Vec<RingOfClustersRungFit>,
    pub winner_index: usize,
}

impl RingOfClustersRungResult {
    pub fn winner(&self) -> &RingOfClustersRungFit {
        &self.fits[self.winner_index]
    }
}

/// Fit and locally refine the constrained ring-of-clusters order ladder.
/// Orders below three are structurally incapable of identifying a circle and
/// are rejected as outside the class. Ranking uses the same BIC-form criterion
/// as the free Gaussian-mixture rung and the smooth parametric shape candidates.
pub fn fit_ring_of_clusters_rung(
    data: ArrayView2<'_, f64>,
    ladder: &[usize],
    config: GaussianMixtureConfig,
) -> Result<RingOfClustersRungResult, AdaptiveRungError> {
    let kind = AdaptiveRungKind::RingOfClusters;
    let n = data.nrows();
    let coarse_orders = eligible_adaptive_orders(kind, ladder, 3, n)?;
    let mut fits = Vec::<RingOfClustersRungFit>::new();
    let mut attempted = std::collections::BTreeSet::<usize>::new();
    let try_order = |k: usize,
                     fits: &mut Vec<RingOfClustersRungFit>,
                     attempted: &mut std::collections::BTreeSet<usize>|
     -> Option<AdaptiveRungOrderFailure> {
        assert!(k >= 3 && k <= n && !attempted.contains(&k));
        attempted.insert(k);
        match crate::evidence::fit_ring_gaussian_mixture(data, k, config) {
            Ok(fit) => {
                let bic = fit.bic();
                if bic.is_finite() {
                    fits.push(RingOfClustersRungFit {
                        k,
                        num_parameters: fit.num_free_parameters(),
                        bic,
                        fit,
                    });
                    None
                } else {
                    Some(AdaptiveRungOrderFailure {
                        k,
                        stage: AdaptiveRungFailureStage::Evidence,
                        message: "BIC is not finite".to_string(),
                    })
                }
            }
            Err(message) => Some(AdaptiveRungOrderFailure {
                k,
                stage: AdaptiveRungFailureStage::Fit,
                message,
            }),
        }
    };
    let mut failures = Vec::new();
    for k in coarse_orders {
        if let Some(failure) = try_order(k, &mut fits, &mut attempted) {
            failures.push(failure);
        }
    }
    if !failures.is_empty() {
        return Err(AdaptiveRungError::OrderFailures { kind, failures });
    }
    let mut probes = 0usize;
    loop {
        let Some(best_k) = fits
            .iter()
            .min_by(|left, right| left.bic.total_cmp(&right.bic).then(left.k.cmp(&right.k)))
            .map(|fit| fit.k)
        else {
            return Err(AdaptiveRungError::InvalidInput {
                kind,
                message: "no eligible order produced a fit".to_string(),
            });
        };
        let next = [best_k.checked_sub(1), best_k.checked_add(1)]
            .into_iter()
            .flatten()
            .find(|&k| k >= 3 && k <= n && !attempted.contains(&k));
        let Some(k) = next else {
            break;
        };
        if probes == MIXTURE_REFINEMENT_MAX_PROBES {
            return Err(AdaptiveRungError::RefinementBudgetExhausted {
                kind,
                best_k,
                completed_probes: probes,
            });
        }
        if let Some(failure) = try_order(k, &mut fits, &mut attempted) {
            return Err(AdaptiveRungError::OrderFailures {
                kind,
                failures: vec![failure],
            });
        }
        probes += 1;
    }
    let ranked = rank_priority_candidates(
        fits.into_iter()
            .enumerate()
            .map(|(index, fit)| {
                let score = fit.bic;
                let tie = fit.k;
                PriorityCandidate::new(fit, index, score, tie)
            })
            .collect(),
    )
    .into_iter()
    .map(|candidate| candidate.item)
    .collect();
    Ok(RingOfClustersRungResult {
        fits: ranked,
        winner_index: 0,
    })
}

// ===========================================================================
// Structured-union rung (#907)
// ===========================================================================

/// One fitted entry of the structured-union rung: the composite structure, its
/// normalized soft-mixture BIC/2, and the complete free-parameter count. Lower
/// is better.
#[derive(Debug, Clone)]
pub struct UnionRungFit {
    pub structure: UnionStructure,
    pub fit: UnionStructureFit,
    /// `Σ_c P_c + (m - 1)`, including free mixing weights.
    pub total_parameters: usize,
    /// BIC/2 of `Σ_c π_c p_c(y)` scored on all training rows.
    pub bic: f64,
}

/// Result of fitting the whole fixed union ladder: every fitted composite plus
/// the index of the in-class winner (lowest normalized soft-mixture BIC).
#[derive(Debug, Clone)]
pub struct UnionRungResult {
    pub fits: Vec<UnionRungFit>,
    pub winner_index: usize,
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

/// SplitMix64 finalizer — a full-avalanche integer hash. Pure and deterministic:
/// it never touches the clock or any RNG state, so the folding it drives is
/// reproducible for a given `(seed, index)` and decorrelated across seeds.
#[inline]
fn splitmix64(x: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(x)
}

/// Deterministic, seed-reproducible, exactly balanced `folds`-way CV partition
/// of `0..n` (no clock randomness). Rows are ordered by `hash(seed, i)` and
/// assigned round-robin, so:
///   - the same `seed` always reproduces the identical folding (deterministic),
///   - different seeds give different (still-deterministic) foldings.
///   - every requested fold is nonempty and fold sizes differ by at most one.
///
/// This is the seam that makes the `adjudicate_atom_shape` `seed=` kwarg
/// functional (#1386): it is mixed into the held-out CV partition rather than
/// being a silent no-op. Invalid requests (`n < 2`, `folds < 2`, or `folds > n`)
/// return an empty partition; selection-time callers reject them explicitly.
pub fn deterministic_cv_folds_seeded(
    n: usize,
    folds: usize,
    seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if n < 2 || folds < 2 || folds > n {
        return Vec::new();
    }
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&row| (splitmix64(seed ^ splitmix64(row as u64)), row));
    let mut assign = vec![0usize; n];
    for (rank, row) in order.into_iter().enumerate() {
        assign[row] = rank % folds;
    }
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
        out.push((train, eval));
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
    if n < 2 || folds < 2 || folds > n {
        return Err(format!(
            "stacking CV requires 2 <= folds <= n with n >= 2; got folds={folds}, n={n}"
        ));
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

/// Adjudicated outcome of a predictive race. A mixed smooth/discrete race and
/// any race containing an adaptive class use honest held-out stacking; only a
/// race of fixed candidates from one side of that boundary uses evidence alone.
#[derive(Debug, Clone)]
pub struct PredictiveRaceVerdict {
    /// Candidate display names, column-aligned with the stacking table / weights.
    pub candidate_names: Vec<String>,
    /// Whether the race actually mixed model classes (smooth vs discrete).
    pub is_cross_class: bool,
    /// Rank-aware Laplace negative-log-evidence per candidate (corroboration;
    /// lower is better).
    pub negative_log_evidence: Vec<f64>,
    /// Stacking weights over the candidates (present iff `headline` is
    /// [`Headline::Stacking`]).
    pub stacking: Option<StackingWeights>,
    /// Index of the headline winner. For stacking races this is the max-weight
    /// candidate; for evidence races it is the min-evidence one.
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
    /// Held-out predictive log-density / stacking weights (cross-class or
    /// adaptive-class race).
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

}

/// One candidate entering the predictive adjudicator: its kind, its rank-aware
/// Laplace negative-log-evidence (already computed on the common scale), how
/// that evidence was certified (for the margin contract), and a selection-time
/// held-out-density provider that refits per CV fold.
pub struct PredictiveRaceCandidate<'a> {
    pub kind: PredictiveCandidateKind,
    pub negative_log_evidence: f64,
    /// Certification of `negative_log_evidence`.
    pub certification: EvidenceCertification,
    pub density_provider: HeldOutDensityProvider<'a>,
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
    /// The realized lead Δ = nle\[contender\] − nle\[winner\] (≥ 0).
    pub lead: f64,
    /// The margin the lead had to exceed (max of the two candidates'
    /// required margins).
    pub required_margin: f64,
}

/// Adjudicate a race that may mix smooth-manifold and discrete candidates.
/// Cross-class mixing is auto-detected (at least one fixed smooth candidate and
/// at least one discrete candidate). Such races use stacking over a
/// selection-time CV held-out log-density table. An adaptive model-selection
/// class also forces stacking even in a discrete-only race, because its order
/// must be selected independently within every outer training fold. Evidence
/// alone adjudicates only same-class races made entirely of fixed candidates.
/// `seed` is mixed into the deterministic CV fold assignment (#1386): the same
/// seed reproduces the identical held-out folding, different seeds give
/// different — but still deterministic — foldings. It only affects the
/// stacking path; fixed same-class races are winner-take-all on evidence and
/// ignore it. Pass [`STACKING_CV_SEED`] for the default folding.
pub fn adjudicate_predictive_race(
    n: usize,
    candidates: Vec<PredictiveRaceCandidate<'_>>,
    folds: usize,
    seed: u64,
    stacking_config: StackingConfig,
) -> Result<PredictiveRaceVerdict, String> {
    if candidates.is_empty() {
        return Err("predictive race requires at least one candidate".to_string());
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.negative_log_evidence.is_finite() {
            return Err(format!(
                "predictive race candidate {index} ({}) has non-finite negative-log-evidence {:?}",
                candidate.kind.display_name(),
                candidate.negative_log_evidence
            ));
        }
        let required_margin = candidate.certification.required_margin();
        if !required_margin.is_finite() || required_margin < 0.0 {
            return Err(format!(
                "predictive race candidate {index} ({}) has invalid certification margin {required_margin:?}; margins must be finite and nonnegative",
                candidate.kind.display_name()
            ));
        }
    }
    for index in 0..candidates.len() {
        if let Some(first_index) = candidates[..index]
            .iter()
            .position(|candidate| candidate.kind == candidates[index].kind)
        {
            return Err(format!(
                "predictive race contains duplicate candidate {:?} at indices {first_index} and {index}",
                candidates[index].kind.display_name()
            ));
        }
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
    let has_adaptive_class = candidates
        .iter()
        .any(|candidate| candidate.kind.requires_predictive_stacking());
    let use_stacking = is_cross_class || has_adaptive_class;

    if !use_stacking {
        // Same-class: winner-take-all on rank-aware evidence (lower wins).
        let certifications: Vec<EvidenceCertification> =
            candidates.iter().map(|c| c.certification).collect();
        // Every value was validated above, so this reduction cannot silently
        // retain index zero merely because no comparable value existed.
        let winner_index = evidence
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "predictive race has no evidence values".to_string())?;
        let best = evidence[winner_index];
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
            if idx == winner_index {
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
        return Ok(PredictiveRaceVerdict {
            candidate_names: names,
            is_cross_class: false,
            negative_log_evidence: evidence,
            stacking: None,
            winner_index,
            headline: Headline::Evidence,
            insufficient_margin,
        });
    }

    // Predictive path: build the selection-time held-out density table and stack.
    let providers: Vec<HeldOutDensityProvider<'_>> =
        candidates.into_iter().map(|c| c.density_provider).collect();
    let table = build_cv_log_density_table(n, folds, seed, &providers)?;
    let stacking =
        solve_stacking_weights(table.view(), stacking_config).map_err(|error| error.to_string())?;
    // Headline winner = max stacking weight (most predictive mass).
    let mut winner_index = 0usize;
    let mut best_w = f64::NEG_INFINITY;
    for (idx, &w) in stacking.weights.iter().enumerate() {
        if w > best_w {
            best_w = w;
            winner_index = idx;
        }
    }
    Ok(PredictiveRaceVerdict {
        candidate_names: names,
        is_cross_class,
        negative_log_evidence: evidence,
        stacking: Some(stacking),
        winner_index,
        headline: Headline::Stacking,
        // Stacking headlines adjudicate by held-out predictive density on the
        // full corpus, not by the approximate-evidence scalar, so the
        // enclosure/coreset margin contract does not gate the verdict here.
        insufficient_margin: None,
    })
}

// ===========================================================================
// Closure-parameter smooth class (#1015): circle ⇄ interval as one estimand
// ===========================================================================

/// One profiled point of the closure family: the closure value `γ`, the
/// profiled (θ and λ_smooth optimised) negative-log evidence and its exact
/// first two profile derivatives at that γ, and the fit handle the caller wants
/// carried for the winner.
#[derive(Debug, Clone)]
pub struct ClosureProfilePoint<FitHandle> {
    pub gamma: f64,
    pub tk_score: f64,
    pub score_gradient: f64,
    pub score_curvature: f64,
    /// Explicit structural diagnostic from the converged inner fit. Merely
    /// attaining `gamma = 0` does not by itself prove support collapse.
    pub support_collapsed: bool,
    pub fit_handle: FitHandle,
}

/// Converged fixed-γ fit returned by the closure profile oracle.
#[derive(Debug, Clone)]
pub struct ClosureProfileFit<FitHandle> {
    pub tk_score: f64,
    pub score_gradient: f64,
    pub score_curvature: f64,
    pub support_collapsed: bool,
    pub fit_handle: FitHandle,
}

/// KKT location of the continuously selected closure optimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureOptimumKind {
    Interior,
    IntervalBoundary,
    CircleBoundary,
}

/// Optimality certificate for the selected closure parameter.
#[derive(Debug, Clone, Copy)]
pub enum ClosureOptimalityCertificate {
    /// Exact-real KKT certificate at an isolated stationary point or domain
    /// boundary.
    Kkt {
        kind: ClosureOptimumKind,
        /// Certified bracket for the stationary abscissa (a point at an exact
        /// endpoint optimum).
        bracket: gam_math::score_opt::ClosedInterval,
        /// Outward exact score, derivative, and curvature ranges on `bracket`.
        score_enclosure: gam_math::score_opt::DerivativeEnclosure,
    },
    /// Stationary structure is numerically unidentifiable, but the entire
    /// bracket is optimal at the score evaluator's representable resolution.
    ResolutionFlat {
        bracket: gam_math::score_opt::ClosedInterval,
        max_score_gap: f64,
        score_resolution: f64,
        score_enclosure: gam_math::score_opt::DerivativeEnclosure,
    },
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
    pub optimality: ClosureOptimalityCertificate,
    /// True when the profile pinned γ at the singular cluster boundary — the
    /// "not a regular smooth 1-D topology" signal that must be routed to the
    /// #907 mixture/union rung rather than reported as a regular closure.
    pub route_to_mixture_rung: bool,
}

#[cfg(test)]
mod tests {
    /// Both fixed space forms are subsumed — the historical fusion behaviour
    /// these cases were written against, kept explicit now that subsumption
    /// is a declared property rather than an assumption.
    const FUSE_BOTH: &[AutoTopologyKind] =
        &[AutoTopologyKind::Euclidean, AutoTopologyKind::Sphere];

    /// An UNSUBSUMED fixed form survives the fusion even when κ is estimable.
    ///
    /// This is the case gam-sae is in: it fits κ on a monomial tangent patch,
    /// which reproduces its `Euclidean` candidate exactly at κ = 0 but never
    /// spans its ambient-harmonic `Sphere`. Before subsumption was declared,
    /// turning on estimability deleted BOTH — silently shrinking the race's
    /// span and reporting the loss as an estimated curvature.
    #[test]
    fn fusion_deletes_only_the_forms_the_caller_declares_subsumed() {
        let input = vec![
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Sphere,
            AutoTopologyKind::ConstantCurvature,
            AutoTopologyKind::Torus,
        ];
        let euclid_only: &[AutoTopologyKind] = &[AutoTopologyKind::Euclidean];
        let fused =
            AutoTopologyKind::fuse_constant_curvature_family(&input, true, euclid_only);
        assert!(
            fused.contains(&AutoTopologyKind::Sphere),
            "an unsubsumed Sphere must race as itself, got {fused:?}"
        );
        assert!(
            !fused.contains(&AutoTopologyKind::Euclidean),
            "the subsumed Euclidean is redundant with the fitted-κ atom, got {fused:?}"
        );
        assert!(
            fused.contains(&AutoTopologyKind::ConstantCurvature),
            "the fitted-κ candidate must remain, got {fused:?}"
        );
        assert!(fused.contains(&AutoTopologyKind::Torus), "{fused:?}");

        // Declaring BOTH subsumed reproduces the historical collapse, so the
        // new parameter is the only thing that changed the outcome.
        let both = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
        assert!(!both.contains(&AutoTopologyKind::Sphere), "{both:?}");
        assert!(!both.contains(&AutoTopologyKind::Euclidean), "{both:?}");

        // Subsuming nothing leaves the set untouched: there is no redundancy to
        // remove, so estimability alone must not delete anything.
        let none = AutoTopologyKind::fuse_constant_curvature_family(&input, true, &[]);
        assert_eq!(none, input, "empty subsumption must be a no-op");
    }

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
    fn fixed_topology_parser_requires_canonical_ordered_density_names() {
        let canonical = [
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Circle,
            AutoTopologyKind::Sphere,
            AutoTopologyKind::Torus,
            AutoTopologyKind::Cylinder,
            AutoTopologyKind::ProjectivePlane,
            AutoTopologyKind::KleinBottle,
            AutoTopologyKind::Mobius,
            AutoTopologyKind::DuchonSheet,
            AutoTopologyKind::ConstantCurvature,
            AutoTopologyKind::Mixture { k: 7 },
            AutoTopologyKind::RingOfClusters { k: 5 },
            AutoTopologyKind::Union {
                structure: UnionStructure::CircleCircle,
            },
            AutoTopologyKind::Union {
                structure: UnionStructure::CirclePointCluster,
            },
            AutoTopologyKind::Union {
                structure: UnionStructure::LineCluster,
            },
        ];
        for kind in canonical {
            let displayed = kind.display_name();
            assert_eq!(
                AutoTopologyKind::parse(&displayed),
                Ok(kind),
                "display/parse must be a bijection for {displayed:?}"
            );
        }

        for malformed in [
            " circle",
            "circle ",
            "Circle",
            "MOBIUS",
            "flat",
            "euclideanpatch",
            "euclidean_patch",
            "periodic",
            "s1",
            "s2",
            "duchon",
            "duchonsheet",
            "duchon-sheet",
            "thin_plate",
            "thinplate",
            "curv",
            "curvature",
            "mkappa",
            "m_kappa",
            "constant-curvature",
            "mixture",
            "mixture7",
            "mixture_7",
            "mixture-k7",
            "mixture_k",
            "mixture_k7junk",
            "mixture_k07",
            "ring_clusters",
            "ring_clusters7",
            "ring_clusters-k7",
            "ring_clusters_k",
            "ring_clusters_k7junk",
            "ring_clusters_k07",
            "union-circle+circle",
            "union_circle_circle",
            "union__circle_circle",
            "union_circle+point+cluster",
            "union_circle+pointcluster",
            "union_line+point+cluster",
            "union_line+pointcluster",
        ] {
            assert!(
                AutoTopologyKind::parse(malformed).is_err(),
                "{malformed:?} must not alias a fixed candidate or adaptive class"
            );
        }
    }

    #[test]
    fn predictive_candidate_names_distinguish_fixed_fits_from_adaptive_classes() {
        assert_eq!(
            PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k: 7 }).display_name(),
            "mixture_k7"
        );
        assert_eq!(
            PredictiveCandidateKind::MixtureClass.display_name(),
            "mixture_class"
        );
        assert_eq!(
            PredictiveCandidateKind::RingOfClustersClass.display_name(),
            "ring_clusters_class"
        );
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
        Box::new(|_: &[usize], eval: &[usize]| Ok(vec![0.0; eval.len()]))
    }

    #[test]
    fn adaptive_rungs_reject_out_of_class_orders_before_fitting() {
        let data = Array2::<f64>::zeros((4, 2));
        let free_error =
            fit_free_cluster_rung(data.view(), &[1, 2], GaussianMixtureConfig::default())
                .expect_err("the free-cluster class must own k >= 2 inside the rung");
        assert!(matches!(
            &free_error,
            AdaptiveRungError::InvalidInput {
                kind: AdaptiveRungKind::GaussianMixture,
                ..
            }
        ));
        assert!(free_error.to_string().contains("require 2 <= k <= n"));

        let ring_error =
            fit_ring_of_clusters_rung(data.view(), &[2, 3], GaussianMixtureConfig::default())
                .expect_err("the ring-cluster class must own k >= 3 inside the rung");
        assert!(matches!(
            &ring_error,
            AdaptiveRungError::InvalidInput {
                kind: AdaptiveRungKind::RingOfClusters,
                ..
            }
        ));
        assert!(ring_error.to_string().contains("require 3 <= k <= n"));

        let oversized = fit_mixture_rung(data.view(), &[5], GaussianMixtureConfig::default())
            .expect_err("orders above n must not disappear from the requested estimand");
        assert!(oversized.to_string().contains("require 1 <= k <= n"));

        let duplicate = fit_mixture_rung(data.view(), &[1, 1], GaussianMixtureConfig::default())
            .expect_err("duplicate orders must not be silently deduplicated");
        assert!(duplicate.to_string().contains("duplicate k=1"));
    }

    #[test]
    fn adaptive_rungs_expose_every_eligible_coarse_fit_failure() {
        let data = Array2::<f64>::zeros((4, 2));
        let invalid_config = GaussianMixtureConfig {
            max_iter: 0,
            ..GaussianMixtureConfig::default()
        };
        let mixture_error = fit_mixture_rung(data.view(), &[1, 2], invalid_config)
            .expect_err("one surviving order must not hide another order's failed fit");
        match mixture_error {
            AdaptiveRungError::OrderFailures { kind, failures } => {
                assert_eq!(kind, AdaptiveRungKind::GaussianMixture);
                assert_eq!(
                    failures.iter().map(|failure| failure.k).collect::<Vec<_>>(),
                    vec![1, 2]
                );
                assert!(
                    failures
                        .iter()
                        .all(|failure| failure.stage == AdaptiveRungFailureStage::Fit)
                );
            }
            other => panic!("expected typed per-order failures, got {other:?}"),
        }

        let ring_error = fit_ring_of_clusters_rung(data.view(), &[3, 4], invalid_config)
            .expect_err("ring orders must also fail closed");
        match ring_error {
            AdaptiveRungError::OrderFailures { kind, failures } => {
                assert_eq!(kind, AdaptiveRungKind::RingOfClusters);
                assert_eq!(
                    failures.iter().map(|failure| failure.k).collect::<Vec<_>>(),
                    vec![3, 4]
                );
            }
            other => panic!("expected typed per-order failures, got {other:?}"),
        }
    }

    #[test]
    fn held_out_mixture_candidates_never_change_their_declared_order() {
        let data = Array2::<f64>::zeros((4, 2));
        let mixture = mixture_density_provider(data.view(), 3, GaussianMixtureConfig::default());
        let error = mixture(&[0, 1], &[2])
            .expect_err("a k=3 candidate must not silently become k=2 on a short fold");
        assert!(error.contains("fixed-order mixture k=3"), "{error}");

        let ring =
            ring_of_clusters_density_provider(data.view(), 3, GaussianMixtureConfig::default());
        let error = ring(&[0, 1], &[2])
            .expect_err("a k=3 ring candidate must not silently change order on a short fold");
        assert!(
            error.contains("fixed-order ring-of-clusters k=3"),
            "{error}"
        );

        let invalid_index =
            mixture_density_provider(data.view(), 1, GaussianMixtureConfig::default());
        let error = invalid_index(&[0, 9], &[1])
            .expect_err("public density providers must reject row indices instead of panicking");
        assert!(error.contains("out of bounds"), "{error}");
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
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                negative_log_evidence: 100.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
                negative_log_evidence: 100.5,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
        ];
        let verdict = adjudicate_predictive_race(
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
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                negative_log_evidence: 100.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
                negative_log_evidence: 105.0,
                certification: EvidenceCertification::Enclosure { gap: 1.0 },
                density_provider: trivial_provider(),
            },
        ];
        let verdict_far = adjudicate_predictive_race(
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
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                negative_log_evidence: 10.0,
                certification: EvidenceCertification::Coreset { certificate: cert },
                density_provider: trivial_provider(),
            },
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
                negative_log_evidence: 10.0 + lead,
                certification: EvidenceCertification::Coreset { certificate: cert },
                density_provider: trivial_provider(),
            },
        ];
        let verdict = adjudicate_predictive_race(
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

    #[test]
    fn adaptive_discrete_class_race_uses_honest_predictive_stacking() {
        let candidates = vec![
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::MixtureClass,
                // Deliberately make evidence prefer the other class: the test
                // must fail if this adaptive race takes the evidence shortcut.
                negative_log_evidence: 100.0,
                certification: EvidenceCertification::Exact,
                density_provider: Box::new(|_, eval| Ok(vec![0.0; eval.len()])),
            },
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::RingOfClustersClass,
                negative_log_evidence: 0.0,
                certification: EvidenceCertification::Exact,
                density_provider: Box::new(|_, eval| Ok(vec![-20.0; eval.len()])),
            },
        ];
        let verdict = adjudicate_predictive_race(
            10,
            candidates,
            5,
            STACKING_CV_SEED,
            StackingConfig::default(),
        )
        .expect("adaptive discrete-class race");

        assert!(!verdict.is_cross_class);
        assert_eq!(verdict.headline, Headline::Stacking);
        assert!(verdict.stacking.is_some());
        assert_eq!(verdict.winner_index, 0);
        assert_eq!(
            verdict.candidate_names,
            vec![
                "mixture_class".to_string(),
                "ring_clusters_class".to_string()
            ]
        );
    }

    #[test]
    fn predictive_race_rejects_duplicate_columns() {
        let duplicate = vec![
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                negative_log_evidence: 1.0,
                certification: EvidenceCertification::Exact,
                density_provider: trivial_provider(),
            },
            PredictiveRaceCandidate {
                kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                negative_log_evidence: 2.0,
                certification: EvidenceCertification::Exact,
                density_provider: trivial_provider(),
            },
        ];
        let error = adjudicate_predictive_race(
            8,
            duplicate,
            4,
            STACKING_CV_SEED,
            StackingConfig::default(),
        )
        .expect_err("duplicate predictive columns make stacking non-identifiable");
        assert!(error.contains("duplicate candidate \"circle\""), "{error}");
    }

    #[test]
    fn predictive_race_rejects_nonfinite_evidence_before_adjudication() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let candidates = vec![
                PredictiveRaceCandidate {
                    kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                    negative_log_evidence: 1.0,
                    certification: EvidenceCertification::Exact,
                    density_provider: trivial_provider(),
                },
                PredictiveRaceCandidate {
                    kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
                    negative_log_evidence: invalid,
                    certification: EvidenceCertification::Exact,
                    density_provider: trivial_provider(),
                },
            ];
            let error = adjudicate_predictive_race(
                8,
                candidates,
                4,
                STACKING_CV_SEED,
                StackingConfig::default(),
            )
            .expect_err("a non-finite candidate must not be skipped in favor of index zero");
            assert!(error.contains("candidate 1"), "{error}");
            assert!(
                error.contains("non-finite negative-log-evidence"),
                "{error}"
            );
        }
    }

    #[test]
    fn predictive_race_rejects_invalid_certification_margins() {
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            let candidates = vec![
                PredictiveRaceCandidate {
                    kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
                    negative_log_evidence: 1.0,
                    certification: EvidenceCertification::Enclosure { gap: invalid },
                    density_provider: trivial_provider(),
                },
                PredictiveRaceCandidate {
                    kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
                    negative_log_evidence: 2.0,
                    certification: EvidenceCertification::Exact,
                    density_provider: trivial_provider(),
                },
            ];
            let error = adjudicate_predictive_race(
                8,
                candidates,
                4,
                STACKING_CV_SEED,
                StackingConfig::default(),
            )
            .expect_err("invalid uncertainty cannot define a race margin");
            assert!(error.contains("candidate 0"), "{error}");
            assert!(error.contains("finite and nonnegative"), "{error}");
        }
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

        let balanced = deterministic_cv_folds_seeded(43, FOLDS, 17);
        assert_eq!(balanced.len(), FOLDS);
        let eval_sizes = balanced
            .iter()
            .map(|(_, eval)| eval.len())
            .collect::<Vec<_>>();
        assert_eq!(eval_sizes.iter().sum::<usize>(), 43);
        assert_eq!(eval_sizes.iter().copied().min(), Some(8));
        assert_eq!(eval_sizes.iter().copied().max(), Some(9));

        assert!(deterministic_cv_folds_seeded(5, 1, 11).is_empty());
        assert!(deterministic_cv_folds_seeded(5, 6, 11).is_empty());
        let providers = vec![trivial_provider()];
        let error = build_cv_log_density_table(5, 1, 11, &providers)
            .expect_err("selection must reject rather than silently clamp invalid fold counts");
        assert!(error.contains("2 <= folds <= n"), "{error}");
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
            |gamma| {
                Ok::<_, String>(ClosureProfileFit {
                    tk_score: 100.0 + 80.0 * (gamma - 0.7).powi(2),
                    score_gradient: 160.0 * (gamma - 0.7),
                    score_curvature: 160.0,
                    support_collapsed: false,
                    fit_handle: gamma,
                })
            },
            |lo, hi| {
                let center = gam_math::score_opt::ClosedInterval::point(0.7);
                let intercept = gam_math::score_opt::ClosedInterval::point(100.0);
                let endpoint_score = |gamma| {
                    let displacement =
                        gam_math::score_opt::ClosedInterval::point(gamma).sub(center);
                    intercept.add(displacement.mul(displacement).scale(80.0))
                };
                let endpoint_lo = endpoint_score(lo);
                let endpoint_hi = endpoint_score(hi);
                let score_lo = if lo <= center.lo && center.hi <= hi {
                    intercept.lo
                } else {
                    endpoint_lo.lo.min(endpoint_hi.lo)
                };
                Ok::<_, String>(gam_math::score_opt::DerivativeEnclosure {
                    score: gam_math::score_opt::ScoreValueEnclosure {
                        value: gam_math::score_opt::ClosedInterval::new(
                            score_lo,
                            endpoint_lo.hi.max(endpoint_hi.hi),
                        ),
                        evaluation_error: 1.0e-12,
                    },
                    derivative: gam_math::score_opt::ClosedInterval::new(lo, hi)
                        .sub(center)
                        .scale(160.0),
                    curvature: gam_math::score_opt::ClosedInterval::point(160.0),
                })
            },
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
        match selection.optimality {
            ClosureOptimalityCertificate::Kkt { kind, .. } => {
                assert_eq!(kind, ClosureOptimumKind::Interior);
            }
            ClosureOptimalityCertificate::ResolutionFlat { .. } => {
                panic!("a well-curved planted parabola must have an isolated KKT optimum")
            }
        }
        // The representative fit handle is the γ̂ point.
        assert!((selection.representative.gamma - selection.ci.gamma_hat).abs() < 1e-12);
    }

    #[test]
    fn closure_profiler_routes_collapse_to_mixture_rung() {
        // A profile that keeps improving toward γ=0 (support collapse) pins the
        // minimiser at the floor and must hand off to the mixture/union rung.
        let selection = profile_closure_within_smooth_class(
            |gamma| {
                Ok::<_, String>(ClosureProfileFit {
                    tk_score: 10.0 + 25.0 * gamma,
                    score_gradient: 25.0,
                    score_curvature: 0.0,
                    support_collapsed: gamma == 0.0,
                    fit_handle: gamma,
                })
            },
            |lo, hi| {
                Ok::<_, String>(gam_math::score_opt::DerivativeEnclosure {
                    score: gam_math::score_opt::ScoreValueEnclosure {
                        value: gam_math::score_opt::ClosedInterval::point(10.0)
                            .add(
                                gam_math::score_opt::ClosedInterval::new(lo, hi)
                                    .scale(25.0),
                            ),
                        evaluation_error: 1.0e-12,
                    },
                    derivative: gam_math::score_opt::ClosedInterval::point(25.0),
                    curvature: gam_math::score_opt::ClosedInterval::point(0.0),
                })
            },
            0.95,
        )
        .expect("closure profile");
        assert!(selection.ci.gamma_hat.abs() < 1e-9);
        assert!(selection.route_to_mixture_rung);
        assert!(selection.ci.ci_includes_interval);
        assert!(matches!(
            selection.optimality,
            ClosureOptimalityCertificate::Kkt {
                kind: ClosureOptimumKind::IntervalBoundary,
                ..
            }
        ));
    }

    #[test]
    fn closure_profiler_does_not_infer_collapse_from_gamma_zero() {
        let selection = profile_closure_within_smooth_class(
            |gamma| {
                Ok::<_, String>(ClosureProfileFit {
                    tk_score: 4.0 + gamma,
                    score_gradient: 1.0,
                    score_curvature: 0.0,
                    support_collapsed: false,
                    fit_handle: gamma,
                })
            },
            |lo, hi| {
                Ok::<_, String>(gam_math::score_opt::DerivativeEnclosure {
                    score: gam_math::score_opt::ScoreValueEnclosure {
                        value: gam_math::score_opt::ClosedInterval::point(4.0)
                            .add(gam_math::score_opt::ClosedInterval::new(lo, hi)),
                        evaluation_error: 1.0e-12,
                    },
                    derivative: gam_math::score_opt::ClosedInterval::point(1.0),
                    curvature: gam_math::score_opt::ClosedInterval::point(0.0),
                })
            },
            0.95,
        )
        .expect("regular interval-boundary profile");
        assert!(matches!(
            selection.optimality,
            ClosureOptimalityCertificate::Kkt {
                kind: ClosureOptimumKind::IntervalBoundary,
                ..
            }
        ));
        assert!(!selection.ci.singular_boundary);
        assert!(!selection.route_to_mixture_rung);
    }

    #[test]
    fn closure_profiler_selects_a_non_lattice_optimum_and_continuous_ci() {
        let planted = 0.713_271_828_f64;
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let selection = profile_closure_within_smooth_class(
            |gamma| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let displacement = gamma - planted;
                Ok::<_, String>(ClosureProfileFit {
                    tk_score: 7.0 + 32.0 * displacement * displacement,
                    score_gradient: 64.0 * displacement,
                    score_curvature: 64.0,
                    support_collapsed: false,
                    fit_handle: gamma,
                })
            },
            |lo, hi| {
                let center = gam_math::score_opt::ClosedInterval::point(planted);
                let intercept = gam_math::score_opt::ClosedInterval::point(7.0);
                let endpoint_score = |gamma| {
                    let displacement =
                        gam_math::score_opt::ClosedInterval::point(gamma).sub(center);
                    intercept.add(displacement.mul(displacement).scale(32.0))
                };
                let at_lo = endpoint_score(lo);
                let at_hi = endpoint_score(hi);
                let score_lo = if lo <= center.lo && center.hi <= hi {
                    intercept.lo
                } else {
                    at_lo.lo.min(at_hi.lo)
                };
                Ok::<_, String>(gam_math::score_opt::DerivativeEnclosure {
                    score: gam_math::score_opt::ScoreValueEnclosure {
                        value: gam_math::score_opt::ClosedInterval::new(
                            score_lo,
                            at_lo.hi.max(at_hi.hi),
                        ),
                        evaluation_error: 1.0e-12,
                    },
                    derivative: gam_math::score_opt::ClosedInterval::new(lo, hi)
                        .sub(center)
                        .scale(64.0),
                    curvature: gam_math::score_opt::ClosedInterval::point(64.0),
                })
            },
            0.95,
        )
        .expect("continuous closure profile");
        assert!((selection.representative.gamma - planted).abs() < 1.0e-7);
        assert!(selection.ci.ci_lo < planted && selection.ci.ci_hi > planted);
        // Adaptive optimization and two root walks are expected to revisit the
        // oracle; this assertion guards specifically against resurrection of
        // the old exactly-17 complete-fit lattice sweep.
        assert_ne!(calls.load(std::sync::atomic::Ordering::Relaxed), 17);
    }

    #[test]
    fn topology_race_thread_plan_bounds_nested_rayon_threads() {
        let plan = TopologyRaceThreadPlan::for_budget(3, 8);
        assert_eq!(plan.concurrent_fits, 3);
        assert!(
            plan.concurrent_fits + plan.concurrent_fits * plan.per_fit_threads <= 8,
            "plan must bound the one-slot-per-candidate reservation plus per-fit Rayon workers"
        );

        let small = TopologyRaceThreadPlan::for_budget(3, 2);
        assert_eq!(small.concurrent_fits, 1);
        assert!(small.concurrent_fits + small.per_fit_threads <= 2);
    }

    #[test]
    fn topology_selector_retains_failed_candidate_records() {
        let selector = TopologyAutoSelector::new(vec![
            AutoTopologyKind::Circle,
            AutoTopologyKind::Torus,
        ]);
        let result = select_topology_with_fit(&selector, |kind| match kind {
            AutoTopologyKind::Circle => Err("inner REML stationarity failed".to_string()),
            AutoTopologyKind::Torus => Ok(TopologyAutoFitEvidence {
                topology_name: "torus".to_string(),
                raw_reml: 3.0,
                raw_reml_roundoff: None,
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 40,
                fit_handle: (),
            }),
            _ => unreachable!(),
        })
        .expect("one converged candidate is selectable");
        assert_eq!(result.winner().unwrap().topology_name, "torus");
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.failed[0].topology_name, "circle");
        assert_eq!(result.failed[0].stage, TopologyCandidateFailureStage::Fit);
        assert!(result.failed[0].message.contains("stationarity"));
    }

    fn lifecycle_evidence(
        name: &str,
        raw_reml: f64,
        laml: Option<f64>,
        deviance: Option<f64>,
        effective_dim: f64,
    ) -> TopologyCandidateOutcome {
        TopologyCandidateOutcome::Fitted(TopologyCandidateEvidence {
            name: name.to_string(),
            raw_reml,
            laml,
            deviance,
            null_dim: Some(0.0),
            null_space_logdet: None,
            effective_dim,
            basis_size: 4,
            n_obs: 20,
        })
    }

    #[test]
    fn typed_lifecycle_owns_score_scaling_and_deterministic_winner() {
        let result = select_topology_candidate_lifecycle(
            vec![
                lifecycle_evidence("larger_raw", 5.0, Some(5.0), Some(6.0), 10.0),
                lifecycle_evidence("smaller_raw", 3.0, Some(3.0), Some(4.0), 2.0),
            ],
            TopologySelectionScoreKind::Reml,
            TopologySelectionScoreScale::PerEffectiveDim,
        )
        .expect("typed lifecycle");
        assert_eq!(result.winner_index, Some(0));
        assert_eq!(result.ranked[0].name, "larger_raw");
        assert!((result.ranked[0].score - 0.5).abs() < 1.0e-12);
        assert_eq!(result.ranked[1].name, "smaller_raw");
        assert!((result.ranked[1].score - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn typed_lifecycle_converts_bad_evidence_without_losing_other_failures() {
        let result = select_topology_candidate_lifecycle(
            vec![
                TopologyCandidateOutcome::Failed(TopologyCandidateFailure {
                    name: "assembly_bad".to_string(),
                    stage: TopologyCandidateFailureStage::Assembly,
                    error_type: "ValueError".to_string(),
                    message: "dimension mismatch".to_string(),
                    evidence_at_failure: None,
                }),
                lifecycle_evidence("evidence_bad", f64::NAN, None, None, 2.0),
                lifecycle_evidence("winner", 2.0, None, None, 2.0),
            ],
            TopologySelectionScoreKind::Reml,
            TopologySelectionScoreScale::Raw,
        )
        .expect("candidate-local evidence failure");
        assert_eq!(result.ranked.len(), 1);
        assert_eq!(result.ranked[0].name, "winner");
        assert_eq!(result.failed.len(), 2);
        assert_eq!(
            result.failed[0].stage,
            TopologyCandidateFailureStage::Assembly
        );
        assert_eq!(
            result.failed[1].stage,
            TopologyCandidateFailureStage::Evidence
        );
        assert!(result.failed[1].message.contains("non-finite REML"));
    }

    #[test]
    fn typed_lifecycle_rejects_duplicate_terminal_outcomes() {
        let error = select_topology_candidate_lifecycle(
            vec![
                lifecycle_evidence("circle", 1.0, None, None, 1.0),
                lifecycle_evidence("circle", 2.0, None, None, 1.0),
            ],
            TopologySelectionScoreKind::Reml,
            TopologySelectionScoreScale::Raw,
        )
        .expect_err("duplicate candidate must be structural error");
        assert!(error.contains("duplicate topology candidate"));
    }

    // --- #944 stage-4 topology collapse tests --------------------------------

    /// Two fixed constant-curvature forms (Euclidean + Sphere) must be fused
    /// into a single estimated-κ ConstantCurvature candidate, at the position of
    /// the first fixed form. Non-CC candidates (Circle, Torus) keep their order.
    /// A consumer that cannot fit κ must not be handed a candidate whose NAME
    /// asserts κ was fitted. Without the capability the fixed forms race as
    /// themselves, so whatever wins is reported as the thing that was actually
    /// measured — a flat patch or a unit sphere, never an "estimated" curvature
    /// nobody estimated.
    #[test]
    fn constant_curvature_fusion_requires_the_capability_to_estimate_it() {
        let input = vec![
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Sphere,
            AutoTopologyKind::Torus,
        ];
        let not_estimable = AutoTopologyKind::fuse_constant_curvature_family(&input, false, FUSE_BOTH);
        assert_eq!(
            not_estimable, input,
            "without the capability the fixed forms must survive unfused"
        );
        assert!(
            !not_estimable.contains(&AutoTopologyKind::ConstantCurvature),
            "a ConstantCurvature candidate must never appear for a consumer that cannot fit κ"
        );

        let estimable = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
        assert!(
            estimable.contains(&AutoTopologyKind::ConstantCurvature),
            "with the capability the fixed forms fuse as #944 intends"
        );
        assert!(!estimable.contains(&AutoTopologyKind::Euclidean));
        assert!(!estimable.contains(&AutoTopologyKind::Sphere));
    }

    #[test]
    fn fuse_cc_family_collapses_euclidean_and_sphere() {
        let input = vec![
            AutoTopologyKind::Circle,
            AutoTopologyKind::Euclidean,
            AutoTopologyKind::Torus,
            AutoTopologyKind::Sphere,
        ];
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
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
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&euclidean_only, true, FUSE_BOTH);
        assert_eq!(fused, euclidean_only, "single fixed form must not be fused");

        let sphere_only = vec![AutoTopologyKind::Sphere];
        let fused2 = AutoTopologyKind::fuse_constant_curvature_family(&sphere_only, true, FUSE_BOTH);
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
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
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
        let once = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
        let twice = AutoTopologyKind::fuse_constant_curvature_family(&once, true, FUSE_BOTH);
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
        let fused = AutoTopologyKind::fuse_constant_curvature_family(&input, true, FUSE_BOTH);
        assert_eq!(fused, input);
    }
}

/// #2729 — the topology race must refuse to decide when the margin between two
/// candidates is inside the resolution of the criterion that produced it.
///
/// The measured incident: two candidates that were the SAME MODEL (a constant
/// birth target lies in the null space of both penalties, so both collapse onto
/// the constant, `edf = 1.0` and a bit-identical `σ̂²`) were separated by
/// `2.4e-13` on scores of magnitude `63.6` — about 17 ulps — and the selector
/// reported a winner, which the atlas-agreement channel then republished as a
/// calibration datum.
///
/// The fixture below reproduces the structure without needing the SAE stack: two
/// designs that span the SAME column space (`X` and `X·Q` for an orthogonal `Q`)
/// fitted with the same closed-form REML. They are the same model, so their
/// evidence is equal in exact arithmetic and differs only by the roundoff of two
/// different orderings of the same arithmetic.
#[cfg(test)]
mod topology_race_tie_2729_tests {
    use super::*;
    use crate::gaussian_reml::gaussian_reml_multi_shared_dispersion_closed_form;
    use ndarray::Array2;

    /// Deterministic, well-conditioned, not reproduced exactly by the design, so
    /// the profiled dispersion stays resolvably positive.
    fn fixture_response(x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut y = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let t = row as f64 / n as f64;
            y[[row, 0]] = 1.0 + 2.0 * t + 0.25 * (7.0 * t).sin();
            y[[row, 1]] = 0.5 - 1.5 * t + 0.10 * (11.0 * t).cos();
        }
        y
    }

    fn design(n: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((n, 3));
        for row in 0..n {
            let t = row as f64 / n as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = t;
            x[[row, 2]] = t * t;
        }
        x
    }

    /// `X · Q` for a Givens rotation at an irrational angle: the same column
    /// space, reached by a different sequence of floating-point operations.
    fn rotated(x: &Array2<f64>) -> Array2<f64> {
        let theta = 0.7_f64;
        let (s, c) = theta.sin_cos();
        let mut rotated = x.clone();
        for row in 0..x.nrows() {
            let a = x[[row, 0]];
            let b = x[[row, 2]];
            rotated[[row, 0]] = c * a - s * b;
            rotated[[row, 2]] = s * a + c * b;
        }
        rotated
    }

    fn evidence(
        name: &str,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> TopologyAutoFitEvidence<&'static str> {
        let penalty = Array2::<f64>::zeros((x.ncols(), x.ncols()));
        let fit = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            None,
        )
        .expect("the fixture design is finite, full rank and not interpolating");
        TopologyAutoFitEvidence {
            topology_name: name.to_string(),
            raw_reml: fit.reml_score,
            raw_reml_roundoff: fit.reml_score_roundoff,
            null_dim: 0.0,
            null_space_logdet: None,
            effective_dim: fit.edf,
            n_obs: x.nrows(),
            fit_handle: if name == "euclidean" {
                "euclidean"
            } else {
                "circle"
            },
        }
    }

    fn race(y: &Array2<f64>, second: &Array2<f64>) -> TopologyAutoSelectorResult<&'static str> {
        let first = design(24);
        let selector = TopologyAutoSelector {
            candidates: vec![AutoTopologyKind::Euclidean, AutoTopologyKind::Circle],
            curvature_is_estimable: false,
            curvature_fusion_subsumes: &[],
            score_scale: TopologyScoreScale::PerObservation,
        };
        select_topology_with_fit(&selector, |kind| match kind {
            AutoTopologyKind::Euclidean => {
                Ok::<_, String>(evidence("euclidean", &first, y))
            }
            AutoTopologyKind::Circle => Ok::<_, String>(evidence("circle", second, y)),
            other => panic!("the fixture races exactly two candidates, not {other:?}"),
        })
        .expect("both fixture candidates are selectable")
    }

    #[test]
    fn two_bit_comparable_models_tie_instead_of_producing_a_winner() {
        let x = design(24);
        let y = fixture_response(&x);
        let result = race(&y, &rotated(&x));
        assert_eq!(result.ranked.len(), 2, "both candidates must be ranked");

        let winner = result.winner().expect("a ranked race has a first entry");
        let runner_up = &result.ranked[1];
        let gap = runner_up.tk_score - winner.tk_score;
        let scale = winner.tk_score.abs().max(runner_up.tk_score.abs());

        // NON-VACUITY: the two candidates really are the same model, i.e. the
        // scores agree to nearly the full mantissa. A tie verdict on scores that
        // were free to differ would prove nothing.
        assert!(
            gap.abs() <= 1.0e-12 * scale,
            "the fixture must present two numerically indistinguishable models: gap {gap:e} \
             on scores of magnitude {scale:e} ({:.17e} vs {:.17e})",
            winner.tk_score,
            runner_up.tk_score
        );

        // The resolutions must be REAL numbers: a `None` or an infinity would
        // make the tie verdict vacuous.
        let winner_resolution = winner
            .tk_score_resolution
            .expect("the closed-form evaluator must accumulate the score's roundoff");
        let runner_up_resolution = runner_up
            .tk_score_resolution
            .expect("the closed-form evaluator must accumulate the score's roundoff");
        assert!(
            winner_resolution.is_finite()
                && runner_up_resolution.is_finite()
                && winner_resolution > 0.0
                && runner_up_resolution > 0.0,
            "both resolutions must be finite and positive: {winner_resolution:e}, \
             {runner_up_resolution:e}"
        );

        // THE VERDICT: a tie, not a winner.
        assert!(
            !result.winner_is_resolved(),
            "the race must not certify a winner between two indistinguishable models: gap \
             {gap:e} against a combined resolution of {:e}",
            winner_resolution + runner_up_resolution
        );
        assert_eq!(
            result.tied_with_winner(),
            vec![1],
            "the runner-up must be reported as tied with the winner"
        );
        assert!(
            !topology_scores_are_resolvably_ordered(winner, runner_up),
            "the ordering predicate must refuse this pair directly, not only through the \
             aggregate"
        );

        // POSITIVE CONTROL, on the SAME measured resolutions: the band is not
        // so wide that nothing can ever be decided. Displace the runner-up by a
        // full TK unit — far outside any roundoff of a score of this magnitude —
        // and the same predicate must now resolve the ordering.
        let mut separated = runner_up.clone();
        separated.tk_score = winner.tk_score + 1.0;
        assert!(
            topology_scores_are_resolvably_ordered(winner, &separated),
            "a 1.0 TK margin must be resolvable against a combined resolution of {:e}",
            winner_resolution + runner_up_resolution
        );
    }

    #[test]
    fn an_unmeasured_resolution_is_never_treated_as_an_exact_zero() {
        let x = design(24);
        let y = fixture_response(&x);
        let result = race(&y, &rotated(&x));
        let mut winner = result.winner().expect("a ranked race has a winner").clone();
        let mut other = result.ranked[1].clone();
        // A producer that established no bound (e.g. a fit rebuilt from a wire
        // format that does not carry one) must not be silently promoted to
        // "resolved to the last bit".
        winner.tk_score_resolution = None;
        other.tk_score = winner.tk_score + 1.0e6;
        assert!(
            !topology_scores_are_resolvably_ordered(&winner, &other),
            "an unmeasured resolution must forbid certifying any margin, however large"
        );
    }
}
