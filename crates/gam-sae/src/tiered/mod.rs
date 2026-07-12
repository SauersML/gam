//! #2023 tiered SAE spine.
//!
//! **The "tiers" are an OPTIMIZER SCHEDULE, not a model structure.** There is ONE
//! model: an intercept `μ` (the fixed effect — [`Tier0Mean`]) plus a sparse
//! dictionary of complexity-priced 1-D atoms, where a *linear* atom is just the
//! rank-1 / `b₂=0` special case of the curved (trig) atom and the rank-charge
//! criterion (`Δloss > ½·d_eff·log n`, MP floor) selects each atom's complexity.
//! The dictionary width `K` is engineered CAPACITY; the number of *certified*
//! atoms is an evidence **output** of that criterion, never a hand-set width
//! target (no PCA-energy cutoff sets it). "Migration" between linear and curved
//! is therefore not a statistical border — it is per-atom rank re-selection under
//! the same criterion.
//!
//! This module owns the coordinate-descent machinery + fixed effect for fitting
//! that one model at large `K`:
//! - [`Tier0Mean`] — the shared intercept `μ` (schedule stage "Tier-0").
//! - [`Tier05SinkAtom`] — a flag-gated finite-anchor attention-sink atom
//!   (schedule stage "Tier-0.5") for fixed known supports such as position 0
//!   and delimiter classes.
//! - a linear sparse-dictionary bulk (schedule stage "Tier-1") whose atoms are
//!   the criterion-selected rank-1 special case.
//! - an evidence-selected curved refinement (schedule stage "Tier-2") fit on the
//!   RAW Tier-1 residual — higher per-atom complexity where the criterion pays
//!   for it. (No projector weight: curvature lives INSIDE the linear span, so a
//!   `Q⊥` whitener would blind the fit; anti-rechasing is the criterion's job.)
//!
//! This module owns the **spine-level** types every schedule stage hangs off:
//!   * [`Tier0Mean`] — the single shared mean μ. Moving the DC out of every atom
//!     into ONE Tier-0 mean is the structural kill of the co-collapse-to-mean
//!     class (issue #10 / #1893): on the de-meaned data the all-atoms-equal-to-
//!     mean state reconstructs zero, so it is EV-invisible and gets pruned rather
//!     than rewarded and PC-reseeded.
//!   * [`Tier05SinkAtom`] — a finite-set atom for known attention-sink support.
//!     It is typed structure, not a nuisance scalar: the atom keeps its
//!     fixed-support anchors and decoder rows, is charged as a finite set, and
//!     only the residual after peeling it is handed to semantic charting.
//!   * [`TieredConfig`] — the composed-fit knobs.
//!   * [`interference_subspace`] — Tier-1's active subspace `Q` (what the linear
//!     dictionary already explains) and its orthogonal complement `Q⊥`. Per the
//!     #2021 coupling the linear dictionary *is* the interference model for the
//!     curved fit: the Tier-2 GLS weight down-weights `Q` (penalizes `Q⊥`), so
//!     curved atoms chase only residual directions. Emitted so Tier-2 can install
//!     a HELD `behavioral_fisher` metric (`structured_whitening=False`) — the
//!     path that both realizes #2021 and avoids the structured-whitening fitter
//!     bug.
//!   * [`WhitenedResidualHandoff`] — the Mode-B (shared whitened residual)
//!     hand-off to Tier-2.
//!   * [`TieredSaeFit`] — the composed artifact. Generic over the Tier-2 artifact
//!     type `T2` so the `tier2-curved` owner defines that struct without a
//!     circular dependency on this module.
//!
//! Term-level composition (concatenating a Tier-1 linear term with a Tier-2
//! curved term into one solve) already lives in [`crate::manifold`]:
//! `SaeManifoldTerm::merge_tiers` + `manifold::stagewise::terminal_joint_assembly`
//! (exact additivity under independent ThresholdGate/ordered Beta--Bernoulli gates). The Mode-A per-block
//! scale-out (one K=1 curved chart per orthonormal Tier-1 block) consumes the
//! block frames on the block-sparse fit directly; see `sparse_dict::block`.

mod fit;
pub use fit::{TieredFitConfig, TieredFitReport, TieredSeedPolicy, fit_tiered};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::basis::{AnchorIndicatorEvaluator, SaeBasisEvaluator};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, finite_set_rank_charge};
use crate::sparse_dict::{SparseDictConfig, SparseDictFit};

/// Tier-0: the single shared mean μ (length `p`). The global DC lives here, not
/// duplicated across `K` per-atom intercepts.
#[derive(Clone, Debug)]
pub struct Tier0Mean {
    /// The shared mean, length `p`.
    pub mean: Array1<f64>,
}

impl Tier0Mean {
    /// Fit Tier-0 as the column mean of `z` (`N×P`). This is the train-split mean;
    /// hold it fixed and reuse it for out-of-sample de-meaning and for the EV
    /// baseline so held-out EV is measured against the same Tier-0 constant.
    pub fn fit(z: ArrayView2<'_, f64>) -> Result<Self, String> {
        if z.nrows() == 0 || z.ncols() == 0 {
            return Err("Tier0Mean::fit requires a non-empty (N, P) matrix".to_string());
        }
        let mean = z
            .mean_axis(Axis(0))
            .ok_or_else(|| "Tier0Mean::fit: mean_axis returned None".to_string())?;
        Ok(Self { mean })
    }

    /// De-mean: `R0 = z − μ` (row-broadcast). The Tier-1 bulk is fit on this.
    pub fn apply(&self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if z.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::apply: z has P={} but μ has length {}",
                z.ncols(),
                self.mean.len()
            ));
        }
        Ok(&z - &self.mean.view().insert_axis(Axis(0)))
    }

    /// Add μ back to a de-meaned reconstruction (`recon + μ`), row-broadcast.
    pub fn reconstruct(&self, recon: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if recon.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::reconstruct: recon has P={} but μ has length {}",
                recon.ncols(),
                self.mean.len()
            ));
        }
        Ok(&recon + &self.mean.view().insert_axis(Axis(0)))
    }
}

/// Tier-0 PER-CONTEXT mean: one mean vector per context/template group, with a
/// global fallback for groups unseen at fit time. On real residual streams a
/// per-prompt/per-template DC otherwise leaks into the fit (measured to drive
/// held-out EV negative), so per-template demean is the production Tier-0 for
/// grouped data; [`Tier0Mean`] is the single-group (global) special case. Same
/// structural DC-atom kill as `Tier0Mean` (#10), applied within each context.
#[derive(Clone, Debug)]
pub struct PerContextMean {
    /// Global fallback mean (used for groups unseen at fit time), length `p`.
    pub global: Array1<f64>,
    /// Per-group column means, keyed by context/template id.
    pub group_means: BTreeMap<i64, Array1<f64>>,
}

impl PerContextMean {
    /// Fit per-group column means from `z` (`N×P`) and `group_ids` (length `N`,
    /// one context id per row). Also stores the global mean as the fallback.
    pub fn fit(z: ArrayView2<'_, f64>, group_ids: &[i64]) -> Result<Self, String> {
        let n = z.nrows();
        let p = z.ncols();
        if n == 0 || p == 0 {
            return Err("PerContextMean::fit requires a non-empty (N, P) matrix".to_string());
        }
        if group_ids.len() != n {
            return Err(format!(
                "PerContextMean::fit: group_ids length {} != N {n}",
                group_ids.len()
            ));
        }
        let global = z
            .mean_axis(Axis(0))
            .ok_or_else(|| "PerContextMean::fit: global mean_axis returned None".to_string())?;
        let mut sums: BTreeMap<i64, (Array1<f64>, usize)> = BTreeMap::new();
        for (row, &g) in z.rows().into_iter().zip(group_ids.iter()) {
            let entry = sums
                .entry(g)
                .or_insert_with(|| (Array1::<f64>::zeros(p), 0usize));
            entry.0 += &row;
            entry.1 += 1;
        }
        let mut group_means = BTreeMap::new();
        for (g, (sum, count)) in sums {
            if count > 0 {
                group_means.insert(g, sum / count as f64);
            }
        }
        Ok(Self {
            global,
            group_means,
        })
    }

    /// The mean for a context: its own if seen at fit time, else the global fallback.
    pub fn row_mean(&self, group: i64) -> &Array1<f64> {
        self.group_means.get(&group).unwrap_or(&self.global)
    }

    /// De-mean each row by its context mean: `R0[i] = z[i] − μ_{group[i]}`.
    pub fn apply(&self, z: ArrayView2<'_, f64>, group_ids: &[i64]) -> Result<Array2<f64>, String> {
        if group_ids.len() != z.nrows() {
            return Err(format!(
                "PerContextMean::apply: group_ids length {} != N {}",
                group_ids.len(),
                z.nrows()
            ));
        }
        let mut out = z.to_owned();
        for (mut row, &g) in out.rows_mut().into_iter().zip(group_ids.iter()) {
            row -= self.row_mean(g);
        }
        Ok(out)
    }

    /// Add each row's context mean back to a de-meaned reconstruction.
    pub fn reconstruct(
        &self,
        recon: ArrayView2<'_, f64>,
        group_ids: &[i64],
    ) -> Result<Array2<f64>, String> {
        if group_ids.len() != recon.nrows() {
            return Err(format!(
                "PerContextMean::reconstruct: group_ids length {} != N {}",
                group_ids.len(),
                recon.nrows()
            ));
        }
        let mut out = recon.to_owned();
        for (mut row, &g) in out.rows_mut().into_iter().zip(group_ids.iter()) {
            row += self.row_mean(g);
        }
        Ok(out)
    }
}

/// Fixed delimiter classes that may be charted as Tier-0.5 sink anchors.
///
/// The class is supplied by the caller from tokenizer/template metadata. The
/// sink fitter never discovers delimiter support from activations, so the atom's
/// support is fixed before reconstruction is fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SinkDelimiterClass {
    Bos,
    Eos,
    Newline,
    ChatBoundary,
    Separator,
}

impl SinkDelimiterClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bos => "bos",
            Self::Eos => "eos",
            Self::Newline => "newline",
            Self::ChatBoundary => "chat_boundary",
            Self::Separator => "separator",
        }
    }
}

/// One finite support anchor in the Tier-0.5 attention-sink atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SinkAnchor {
    /// Reference/background category: rows that are neither position 0 nor one
    /// of the configured delimiter classes.
    Semantic,
    /// The known first-token attention sink.
    PositionZero,
    /// A configured delimiter class with fixed tokenizer/template support.
    Delimiter(SinkDelimiterClass),
}

impl SinkAnchor {
    pub fn label(self) -> &'static str {
        match self {
            Self::Semantic => "semantic_reference",
            Self::PositionZero => "position_0",
            Self::Delimiter(class) => class.label(),
        }
    }

    fn is_sink(self) -> bool {
        !matches!(self, Self::Semantic)
    }
}

/// Flag-gated Tier-0.5 sink-atom configuration.
#[derive(Clone, Debug)]
pub struct Tier05SinkAtomConfig {
    /// Disabled by default: callers must opt in after supplying row support.
    pub enabled: bool,
    /// Include the fixed position-0 support anchor.
    pub include_position_zero: bool,
    /// Include fixed delimiter-class support anchors.
    pub delimiter_classes: Vec<SinkDelimiterClass>,
}

impl Default for Tier05SinkAtomConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            include_position_zero: true,
            delimiter_classes: Vec::new(),
        }
    }
}

impl Tier05SinkAtomConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            include_position_zero: true,
            delimiter_classes: Vec::new(),
        }
    }

    pub fn position_zero() -> Self {
        Self {
            enabled: true,
            include_position_zero: true,
            delimiter_classes: Vec::new(),
        }
    }

    pub fn anchors(&self) -> Result<Vec<SinkAnchor>, String> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        if !self.include_position_zero && self.delimiter_classes.is_empty() {
            return Err(
                "Tier05SinkAtomConfig::anchors: enabled sink atom needs position-0 or delimiter support"
                    .to_string(),
            );
        }
        let mut anchors = vec![SinkAnchor::Semantic];
        if self.include_position_zero {
            anchors.push(SinkAnchor::PositionZero);
        }
        let unique: BTreeSet<SinkDelimiterClass> = self.delimiter_classes.iter().copied().collect();
        for class in unique {
            anchors.push(SinkAnchor::Delimiter(class));
        }
        Ok(anchors)
    }
}

/// Tier-0.5 finite-set attention-sink atom.
///
/// This is the typed counterpart to a nuisance regress-out: the basis kind is
/// [`SaeAtomBasisKind::FiniteSet`], the basis is a one-hot
/// [`AnchorIndicatorEvaluator`], and the row support is fixed from known
/// positions/delimiter metadata before the decoder is fit. The atom is additive:
/// downstream semantic charting sees `residual_after_sink`, while reconstruction
/// adds the sink contribution back.
#[derive(Clone, Debug)]
pub struct Tier05SinkAtom {
    pub atom: SaeManifoldAtom,
    pub anchors: Vec<SinkAnchor>,
    pub anchor_counts: Vec<usize>,
    pub rank_charge: usize,
    pub variance_absorbed: f64,
}

impl Tier05SinkAtom {
    /// Dense training reconstruction of the sink atom (`N×P`).
    pub fn reconstruction(&self) -> Array2<f64> {
        self.atom.basis_values.dot(&self.atom.decoder_coefficients)
    }

    /// Peel the sink from a same-row residual matrix before semantic charting.
    pub fn residual_after_sink(
        &self,
        residual: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let expected = (self.atom.basis_values.nrows(), self.atom.output_dim());
        if residual.dim() != expected {
            return Err(format!(
                "Tier05SinkAtom::residual_after_sink: residual shape {:?} incompatible with atom rows/output ({}, {})",
                residual.dim(),
                self.atom.basis_values.nrows(),
                self.atom.output_dim()
            ));
        }
        Ok(&residual - &self.reconstruction())
    }

    /// Add the sink contribution back to a semantic reconstruction.
    pub fn reconstruct_with_sink(
        &self,
        semantic_recon: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let expected = (self.atom.basis_values.nrows(), self.atom.output_dim());
        if semantic_recon.dim() != expected {
            return Err(format!(
                "Tier05SinkAtom::reconstruct_with_sink: reconstruction shape {:?} incompatible with atom rows/output ({}, {})",
                semantic_recon.dim(),
                self.atom.basis_values.nrows(),
                self.atom.output_dim()
            ));
        }
        Ok(&semantic_recon + &self.reconstruction())
    }
}

/// Fit the Tier-0.5 sink atom on a post-Tier-0 residual.
///
/// `positions` are within-sequence token positions. `delimiter_classes` is either
/// empty (no delimiter support supplied) or length `N`; entries not listed in
/// `config.delimiter_classes` remain in the semantic reference anchor. Position
/// 0 takes precedence over delimiter labels, because the measured confound is
/// the first-token sink itself.
pub fn fit_tier05_sink_atom(
    residual: ArrayView2<'_, f64>,
    positions: &[i64],
    delimiter_classes: &[Option<SinkDelimiterClass>],
    config: &Tier05SinkAtomConfig,
) -> Result<Option<Tier05SinkAtom>, String> {
    if !config.enabled {
        return Ok(None);
    }
    let n = residual.nrows();
    let p = residual.ncols();
    if n == 0 || p == 0 {
        return Err("fit_tier05_sink_atom: residual must be a non-empty N×P matrix".to_string());
    }
    if positions.len() != n {
        return Err(format!(
            "fit_tier05_sink_atom: positions length {} != N {n}",
            positions.len()
        ));
    }
    if !delimiter_classes.is_empty() && delimiter_classes.len() != n {
        return Err(format!(
            "fit_tier05_sink_atom: delimiter_classes length {} must be 0 or N {n}",
            delimiter_classes.len()
        ));
    }
    if !config.delimiter_classes.is_empty() && delimiter_classes.is_empty() {
        return Err(
            "fit_tier05_sink_atom: delimiter classes configured but no per-row delimiter labels supplied"
                .to_string(),
        );
    }

    let anchors = config.anchors()?;
    let delimiter_set: BTreeSet<SinkDelimiterClass> =
        config.delimiter_classes.iter().copied().collect();
    let mut anchor_lookup = BTreeMap::new();
    for (idx, anchor) in anchors.iter().copied().enumerate() {
        anchor_lookup.insert(anchor, idx);
    }

    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut counts = vec![0usize; anchors.len()];
    for row in 0..n {
        let delimiter = if delimiter_classes.is_empty() {
            None
        } else {
            delimiter_classes[row]
        };
        let anchor = if config.include_position_zero && positions[row] == 0 {
            SinkAnchor::PositionZero
        } else if let Some(class) = delimiter {
            if delimiter_set.contains(&class) {
                SinkAnchor::Delimiter(class)
            } else {
                SinkAnchor::Semantic
            }
        } else {
            SinkAnchor::Semantic
        };
        let idx = anchor_lookup.get(&anchor).copied().ok_or_else(|| {
            format!(
                "fit_tier05_sink_atom: support anchor {} was not configured",
                anchor.label()
            )
        })?;
        coords[[row, 0]] = idx as f64;
        counts[idx] += 1;
    }

    let sink_rows: usize = anchors
        .iter()
        .zip(counts.iter())
        .filter(|(anchor, _count)| anchor.is_sink())
        .map(|(_anchor, &count)| count)
        .sum();
    if sink_rows == 0 {
        return Err(
            "fit_tier05_sink_atom: enabled sink atom has no sink-supported rows".to_string(),
        );
    }

    let evaluator = Arc::new(AnchorIndicatorEvaluator::new(anchors.len())?);
    let (basis_values, basis_jacobian) = evaluator.evaluate(coords.view())?;
    let mut decoder = Array2::<f64>::zeros((anchors.len(), p));
    for row in 0..n {
        let anchor = coords[[row, 0]] as usize;
        for col in 0..p {
            decoder[[anchor, col]] += residual[[row, col]];
        }
    }
    for anchor in 0..anchors.len() {
        if counts[anchor] > 0 {
            let scale = 1.0 / counts[anchor] as f64;
            for col in 0..p {
                decoder[[anchor, col]] *= scale;
            }
        }
    }
    let smooth_penalty = Array2::<f64>::zeros((anchors.len(), anchors.len()));
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "tier0_5_attention_sink",
        SaeAtomBasisKind::FiniteSet,
        1,
        basis_values,
        basis_jacobian,
        decoder,
        smooth_penalty,
    )?
    .with_basis_second_jet(evaluator);

    let reconstruction = atom.basis_values.dot(&atom.decoder_coefficients);
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for row in 0..n {
        for col in 0..p {
            let r = residual[[row, col]] - reconstruction[[row, col]];
            rss += r * r;
            let v = residual[[row, col]];
            tss += v * v;
        }
    }
    let variance_absorbed = if tss <= 0.0 { 0.0 } else { 1.0 - rss / tss };

    Ok(Some(Tier05SinkAtom {
        atom,
        anchors,
        anchor_counts: counts,
        rank_charge: finite_set_rank_charge(anchor_lookup.len()),
        variance_absorbed,
    }))
}

/// Position-only convenience wrapper for the measured position-0 sink.
pub fn fit_position0_sink_atom(
    residual: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<Tier05SinkAtom, String> {
    let config = Tier05SinkAtomConfig::position_zero();
    fit_tier05_sink_atom(residual, positions, &[], &config)?.ok_or_else(|| {
        "fit_position0_sink_atom: position-0 sink config unexpectedly disabled".to_string()
    })
}

/// Pre-chart residual after Tier-0 and optional Tier-0.5 peeling.
#[derive(Clone, Debug)]
pub struct TieredPrechartResidual {
    pub tier0: Tier0Mean,
    pub tier05_sink: Option<Tier05SinkAtom>,
    /// Residual handed to semantic Tier-1/Tier-2 charting.
    pub residual: Array2<f64>,
}

/// Apply the additive pre-chart path: Tier-0 mean first, then the optional
/// Tier-0.5 finite-anchor sink atom, then hand only the residual to semantic
/// dictionary/chart fitting.
pub fn prechart_residual(
    z: ArrayView2<'_, f64>,
    positions: &[i64],
    delimiter_classes: &[Option<SinkDelimiterClass>],
    config: &TieredConfig,
) -> Result<TieredPrechartResidual, String> {
    let tier0 = Tier0Mean::fit(z)?;
    let after_tier0 = tier0.apply(z)?;
    let tier05_sink = fit_tier05_sink_atom(
        after_tier0.view(),
        positions,
        delimiter_classes,
        &config.tier05_sink,
    )?;
    let residual = match &tier05_sink {
        Some(sink) => sink.residual_after_sink(after_tier0.view())?,
        None => after_tier0,
    };
    Ok(TieredPrechartResidual {
        tier0,
        tier05_sink,
        residual,
    })
}

/// Knobs for a composed tiered fit.
#[derive(Clone, Debug)]
pub struct TieredConfig {
    /// Tier-1 collapsed-linear sparse dictionary configuration (carries `K`, the
    /// active budget `s`, epochs, and the GPU score-routing mode).
    pub tier1: SparseDictConfig,
    /// Rank `r` of the interference subspace `Q` handed to Tier-2 (`None` ⇒ pick
    /// by the 99% energy threshold in [`interference_subspace`]).
    pub lambda_seed_rank: Option<usize>,
    /// Whether to run the Tier-2 curved tier at all (`false` ⇒ Tier-0 + Tier-1
    /// only, the linear-bulk baseline).
    pub tier2_enabled: bool,
    /// Optional Tier-0.5 finite-anchor attention-sink atom, peeled after Tier-0
    /// and before semantic Tier-1/Tier-2 charting.
    pub tier05_sink: Tier05SinkAtomConfig,
}

impl TieredConfig {
    /// A Tier-0 + Tier-1 config at dictionary width `k_linear` (Tier-2 disabled).
    pub fn linear_bulk(k_linear: usize) -> Self {
        Self {
            tier1: SparseDictConfig::new(k_linear),
            lambda_seed_rank: None,
            tier2_enabled: false,
            tier05_sink: Tier05SinkAtomConfig::disabled(),
        }
    }
}

/// Tier-1's active subspace: the directions the linear dictionary already
/// explains (`q`), its orthogonal complement (`q_perp`), and the per-direction
/// energy scale (`scale`, the singular values of the usage-weighted decoder).
///
/// `q` is `P×r` with orthonormal columns; `q_perp` is `P×(P−r)` with orthonormal
/// columns; together they are a full orthonormal basis of `ℝ^P` (`q ⟂ q_perp`).
///
/// DIAGNOSTIC ONLY — span reporting. **Do NOT use `q_perp` as a GLS weight on the
/// Tier-2 residual.** A `G = q_perp q_perpᵀ = I − q qᵀ` weight is a proven design
/// error (audit 2026-07-03): a curve's chords span the curve's OWN plane, so the
/// post-linear curvature signal (chord-sag) lives INSIDE `span(q)` — the measured
/// counterexample put 95.4% of the residual energy inside `q` and a Q⊥ weight
/// crushed the in-plane signal to noise. Curvature is a constraint AMONG the
/// directions Tier-1 spans, not a direction it missed, so Tier-2 fits the RAW
/// residual; anti-rechasing is priced by the evidence criterion (the ledger),
/// never by a projector, and the only sanctioned reweighting is a soft `Σ̂`
/// estimated from the actual (anisotropic) residual. See the
/// `qperp_weight_is_blind_to_in_plane_curvature` regression test.
#[derive(Clone, Debug)]
pub struct InterferenceSubspace {
    /// Active subspace, `P×r`, orthonormal columns (what Tier-1 explains).
    pub q: Array2<f64>,
    /// Orthogonal complement, `P×(P−r)`, orthonormal columns (`Q⊥`).
    pub q_perp: Array2<f64>,
    /// Singular values of the usage-weighted decoder along `q`, length `r`.
    pub scale: Array1<f64>,
}

/// Compute Tier-1's [`InterferenceSubspace`] from a fitted sparse dictionary.
///
/// DIAGNOSTIC span report — how much of `ℝ^P` the linear dictionary already
/// spans. It must NOT weight or gate the Tier-2 fit (see [`InterferenceSubspace`]
/// for why `q_perp`-weighting is blind to curvature).
///
/// Forms the usage-weighted decoder Gram `G = Σ_k w_k d_k d_kᵀ` (`P×P`), where
/// `d_k` is atom `k`'s decoder row and `w_k = Σ_i codes[i,k]²` is its total fired
/// energy (so dead atoms contribute nothing and `Q` is genuinely the *active*
/// subspace). The eigenvectors of `G` split into the top-`r` (the active subspace
/// `Q`) and the trailing `P−r` (`Q⊥`); `scale = √eval` along `Q`.
///
/// `rank`: `Some(r)` pins `r = min(r, P)`; `None` keeps the smallest `r` whose
/// eigen-energy reaches 99% of the total (at least 1).
pub fn interference_subspace(
    fit: &SparseDictFit,
    rank: Option<usize>,
) -> Result<InterferenceSubspace, String> {
    let decoder = fit.decoder.view();
    let k = decoder.nrows();
    let p = decoder.ncols();
    if k == 0 || p == 0 {
        return Err("interference_subspace: empty decoder".to_string());
    }

    // Per-atom fired energy w_k = Σ_i codes[i,k]².
    let mut weight = vec![0.0f64; k];
    for (idx_row, code_row) in fit.indices.rows().into_iter().zip(fit.codes.rows()) {
        for (&atom_u32, &code) in idx_row.iter().zip(code_row.iter()) {
            let atom = atom_u32 as usize;
            if atom < k {
                weight[atom] += (code as f64) * (code as f64);
            }
        }
    }

    // Usage-weighted decoder `Dw` (K×P), Dw_k = √w_k · d_k, then G = Dwᵀ Dw (P×P)
    // via a single GEMM rather than K rank-1 updates.
    let mut dw = Array2::<f64>::zeros((k, p));
    for atom in 0..k {
        let sw = weight[atom].max(0.0).sqrt();
        if sw == 0.0 {
            continue;
        }
        let src = decoder.row(atom);
        let mut dst = dw.row_mut(atom);
        for c in 0..p {
            dst[c] = sw * (src[c] as f64);
        }
    }
    let gram = dw.t().dot(&dw);

    // Symmetric eigendecomposition: ascending eigenvalues, columns are the
    // orthonormal eigenvectors (leading direction is the LAST column).
    let (evals, evecs) = gram
        .eigh(faer::Side::Lower)
        .map_err(|err| format!("interference_subspace eigensolve failed: {err}"))?;
    let total: f64 = evals.iter().map(|&e| e.max(0.0)).sum();
    if total <= 0.0 {
        return Err(
            "interference_subspace: Tier-1 decoder carries no fired energy (all atoms dead)"
                .to_string(),
        );
    }

    // Choose r (columns are ascending, so the active subspace is the TAIL).
    let r = match rank {
        Some(r) => r.min(p).max(1),
        None => {
            // Smallest r whose top-r eigen-energy reaches 99% of the total.
            // 0.99 is a REPORTING tolerance for this span DIAGNOSTIC only (how
            // many directions to list); it gates/weights no fit and prices no
            // atom — decisions stay with the rank-charge evidence criterion.
            let mut acc = 0.0f64;
            let mut chosen = 1usize;
            for (taken, &e) in evals.iter().rev().enumerate() {
                acc += e.max(0.0);
                chosen = taken + 1;
                if acc >= 0.99 * total {
                    break;
                }
            }
            chosen.min(p).max(1)
        }
    };

    // q = last r columns (largest eigenvalues), scale = √eval along q.
    let mut q = Array2::<f64>::zeros((p, r));
    let mut scale = Array1::<f64>::zeros(r);
    for j in 0..r {
        let col = p - 1 - j; // descending: p-1 is the largest
        q.column_mut(j).assign(&evecs.column(col));
        scale[j] = evals[col].max(0.0).sqrt();
    }
    // q_perp = the leading (p − r) columns (smallest eigenvalues), the complement.
    let pr = p - r;
    let mut q_perp = Array2::<f64>::zeros((p, pr));
    for j in 0..pr {
        q_perp.column_mut(j).assign(&evecs.column(j));
    }

    Ok(InterferenceSubspace { q, q_perp, scale })
}

/// Mode-B hand-off: the RAW post-Tier-1 shared residual handed to the Tier-2
/// curved fit (`tier2-curved` / #17). Tier-2 fits its curved atoms on `residual`
/// DIRECTLY — no projector weight. Anti-rechasing (a curved atom duplicating
/// linear work) is priced by the evidence criterion, NOT enforced by a metric;
/// the only sanctioned reweighting is a soft `Σ̂` estimated from this actual
/// (anisotropic) residual. `interference` rides along as a span DIAGNOSTIC only —
/// it must NOT gate or weight the fit (`q_perp`-weighting is blind to curvature;
/// see [`InterferenceSubspace`]).
#[derive(Clone, Debug)]
pub struct WhitenedResidualHandoff {
    /// Post-Tier-1 residual, `N×P`, f64. Without Tier-0.5 this is
    /// `R = (z − μ) − T1.reconstruct()`; with a sink atom it is
    /// `R = (z − μ − sink) − T1.reconstruct()`. Tier-2 fits this RAW residual
    /// directly.
    pub residual: Array2<f64>,
    /// Tier-1's active subspace (`q`, `q_perp`, `scale`) — DIAGNOSTIC only.
    pub interference: InterferenceSubspace,
    /// The frozen Tier-1 decoder, `K×P` (for out-of-sample residual recompute).
    pub tier1_decoder: Array2<f32>,
    /// The Tier-0 shared mean μ, length `P`.
    pub mean: Array1<f64>,
    /// Optional Tier-0.5 finite-anchor attention-sink atom already peeled from
    /// `residual`; downstream reconstruction adds it back before Tier-0.
    pub tier05_sink: Option<Tier05SinkAtom>,
}

/// The composed tiered artifact. Generic over the Tier-2 artifact `T2` (defined
/// by the `tier2-curved` owner as `Tier2CurvedArtifact`) so this container has no
/// circular dependency on the curved-tier module. `tier2` is `None` when Tier-2
/// is disabled or every curved birth is rejected.
#[derive(Clone, Debug)]
pub struct TieredSaeFit<T2> {
    /// Tier-0 shared mean.
    pub tier0: Tier0Mean,
    /// Optional Tier-0.5 finite-anchor attention-sink atom.
    pub tier05_sink: Option<Tier05SinkAtom>,
    /// Tier-1 linear sparse-dictionary bulk.
    pub tier1: SparseDictFit,
    /// Tier-2 curved artifact (owner-defined), if present.
    pub tier2: Option<T2>,
    /// Combined held-in explained variance against the Tier-0 mean baseline.
    pub explained_variance: f64,
}

/// Explained variance `1 − RSS/TSS` of `recon` against `z`, with the total sum of
/// squares taken about the supplied Tier-0 `mean` (the honest tiered baseline: a
/// model must beat "predict the shared mean", not "predict zero").
pub fn explained_variance_vs_mean(
    z: ArrayView2<'_, f64>,
    recon: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
) -> f64 {
    let mut rss = 0.0f64;
    for (zr, rr) in z.rows().into_iter().zip(recon.rows()) {
        for c in 0..z.ncols() {
            let d = zr[c] - rr[c];
            rss += d * d;
        }
    }
    let baseline = &z - &mean.view().insert_axis(Axis(0));
    let tss: f64 = baseline.iter().map(|&v| v * v).sum();
    if tss <= 0.0 {
        return f64::NAN;
    }
    1.0 - rss / tss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_dict::DecoderSolveStats;
    use ndarray::array;

    #[test]
    fn tier0_mean_roundtrips() {
        let z = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let t0 = Tier0Mean::fit(z.view()).expect("fit");
        assert!((t0.mean[0] - 3.0).abs() < 1e-12);
        assert!((t0.mean[1] - 4.0).abs() < 1e-12);
        let demeaned = t0.apply(z.view()).expect("apply");
        // Column means of the de-meaned data are ~0.
        let cm = demeaned.mean_axis(Axis(0)).unwrap();
        assert!(cm[0].abs() < 1e-12 && cm[1].abs() < 1e-12);
        // reconstruct(apply(z)) == z.
        let back = t0.reconstruct(demeaned.view()).expect("reconstruct");
        for (a, b) in back.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn interference_subspace_q_and_qperp_are_orthonormal_complements() {
        // A 2-atom dictionary in p=3 spanning e0 and e1; e2 is unexplained.
        let decoder = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = array![[0u32, 1u32], [0u32, 1u32]];
        let codes = array![[2.0f32, 1.0f32], [2.0f32, 1.0f32]];
        let fit = SparseDictFit {
            decoder,
            indices,
            codes,
            explained_variance: 0.0,
            epochs: 0,
            convergence: crate::sparse_dict::SparseDictConvergence::trivially_converged(),
            active: 2,
            score_route_stats: Default::default(),
            decoder_solve_stats: DecoderSolveStats::default(),
        };
        let sub = interference_subspace(&fit, Some(2)).expect("subspace");
        assert_eq!(sub.q.dim(), (3, 2));
        assert_eq!(sub.q_perp.dim(), (3, 1));
        // q columns orthonormal.
        let gq = sub.q.t().dot(&sub.q);
        assert!((gq[[0, 0]] - 1.0).abs() < 1e-9 && (gq[[1, 1]] - 1.0).abs() < 1e-9);
        assert!(gq[[0, 1]].abs() < 1e-9);
        // q ⟂ q_perp.
        let cross = sub.q.t().dot(&sub.q_perp);
        assert!(cross.iter().all(|&v| v.abs() < 1e-9));
        // q_perp must be (±)e2, the unexplained direction.
        assert!(sub.q_perp[[2, 0]].abs() > 0.999);
        assert!(sub.q_perp[[0, 0]].abs() < 1e-6 && sub.q_perp[[1, 0]].abs() < 1e-6);
        // Atom 0 (code 2) carries more energy than atom 1 (code 1) ⇒ larger scale first.
        assert!(sub.scale[0] >= sub.scale[1]);
    }

    #[test]
    fn per_context_mean_zeros_each_group_and_falls_back() {
        // group 0 centered at (10,10), group 1 at (−5,−5).
        let z = array![[11.0, 9.0], [9.0, 11.0], [-4.0, -6.0], [-6.0, -4.0]];
        let groups = [0i64, 0, 1, 1];
        let pcm = PerContextMean::fit(z.view(), &groups).expect("fit");
        assert!((pcm.row_mean(0)[0] - 10.0).abs() < 1e-12);
        assert!((pcm.row_mean(1)[0] + 5.0).abs() < 1e-12);
        // Unseen context falls back to the global mean.
        assert!((pcm.row_mean(999)[0] - pcm.global[0]).abs() < 1e-12);
        // Per-context de-mean zeros each group (⇒ column sums ~0 overall).
        let demeaned = pcm.apply(z.view(), &groups).expect("apply");
        let col_sum = demeaned.sum_axis(Axis(0));
        assert!(col_sum[0].abs() < 1e-12 && col_sum[1].abs() < 1e-12);
        // Roundtrip.
        let back = pcm
            .reconstruct(demeaned.view(), &groups)
            .expect("reconstruct");
        for (a, b) in back.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn qperp_weight_is_blind_to_in_plane_curvature() {
        // The RETRACTED Q⊥ GLS weight (`G = I − QQᵀ`) is self-defeating: a curve's
        // chords span the curve's OWN plane, so the post-linear curvature signal
        // lives INSIDE `span(Q)` and `Q⊥`-weighting annihilates exactly what Tier-2
        // exists to model. This pins that so the idea can't be re-derived.
        //
        // Tier-1 linear atoms span the e0–e1 plane — the plane a circle would live
        // in; a fitted circle's chords span the same plane.
        let decoder = array![[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let indices = array![[0u32, 1u32], [0u32, 1u32], [0u32, 1u32]];
        let codes = array![[3.0f32, 2.0], [3.0, 2.0], [3.0, 2.0]];
        let fit = SparseDictFit {
            decoder,
            indices,
            codes,
            explained_variance: 0.0,
            epochs: 0,
            convergence: crate::sparse_dict::SparseDictConvergence::trivially_converged(),
            active: 2,
            score_route_stats: Default::default(),
            decoder_solve_stats: DecoderSolveStats::default(),
        };
        let sub = interference_subspace(&fit, Some(2)).expect("subspace");
        // The circle's plane (e0, e1) lives ENTIRELY inside Q: ‖Qᵀ e_j‖ ≈ 1.
        for j in 0..2 {
            let mut ej = Array1::<f64>::zeros(4);
            ej[j] = 1.0;
            let qte = sub.q.t().dot(&ej);
            let proj_norm = qte.dot(&qte).sqrt();
            assert!(
                (proj_norm - 1.0).abs() < 1e-9,
                "e{j} not fully in Q: {proj_norm}"
            );
        }
        // A residual that IS the in-plane curvature signal (chord-sag), unit norm.
        let curvature = array![0.6f64, -0.8, 0.0, 0.0];
        let sig_rms = curvature.dot(&curvature).sqrt();
        assert!(
            sig_rms > 0.99,
            "planted signal should be ~unit; got {sig_rms}"
        );
        // What the Q⊥ GLS weight would keep: `q_perpᵀ · residual`.
        let qperp_component = sub.q_perp.t().dot(&curvature);
        let qperp_rms = qperp_component.dot(&qperp_component).sqrt();
        assert!(
            qperp_rms < 1e-9,
            "Q⊥ weight crushes the in-plane curvature to noise ({qperp_rms}) — it is BLIND \
             to what Tier-2 must model; fit the raw residual instead"
        );
    }
}
