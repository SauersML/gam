//! #1026 — load-bearing curved-vs-linear hybrid split for the fitted SAE
//! dictionary.
//!
//! The selection machinery ([`gam_solve::evidence::select_hybrid_atom`],
//! [`gam_solve::evidence::select_hybrid_split`]) and the per-atom
//! integration helper
//! ([`crate::assignment::select_hybrid_atom_parameterization`]) are
//! correct and tested, but until now were called nowhere in the fitter: the
//! post-fit pass only *logged* each `d = 1` atom's fitted turning `Θ`. This
//! module makes the split LOAD-BEARING by building, per fitted `d = 1` atom, the
//! two already-realized candidates and adjudicating them by the common
//! evidence criterion.
//!
//! ## The common-evidence comparison on the data (#1202)
//!
//! Both candidates are scored against the SAME data: the portion of the
//! response matrix the atom is responsible for reconstructing, namely its
//! **leave-this-atom-out residual**
//!
//!     y_resp[i] = target[i] − ( Σ_j a[i,j]·γ_j(t_{ij}) − a[i,k]·γ_k(t_{ik}) )
//!               = target[i] − without_k[i],
//!
//! the response with every OTHER atom's contribution subtracted. Over the rows
//! assigned to atom `k` (assignment mass `a[i,k] = a_k`), the two candidates
//! predict that residual:
//!
//!   * the CURVED candidate is scored at its CONSTRAINED MINIMUM over the atom's
//!     decoder coefficients — re-fit on `y_resp` as
//!     `min_B ‖y_resp − a_k·(Φ(t)·B)‖²` ([`curved_refit_rss`]) — so its data-fit
//!     deviance is the smallest weighted RSS the curved family can attain on this
//!     residual at the realized codes, not the possibly-collapsed realized curve.
//!   * the LINEAR candidate predicts `a_k · (b₀ + (t − t̄)·b₁)`, the best
//!     weighted least-squares straight line fit to `y_resp` (design column
//!     scaled by the same assignment mass `a_k`), so its data-fit deviance is the
//!     weighted RSS of the best line against the SAME residual.
//!
//! ## A genuine NESTED min-vs-min comparison (#1051)
//!
//! Both arms are now at their constrained minimum on the SAME leave-one-atom-out
//! residual: the linear arm is the closed-form min-over-lines, and the curved arm
//! is the min-over-decoders refit ([`curved_refit_rss`]). The linear special case
//! `a_k·(b₀ + (t−t̄)·b₁)` is a MEMBER of the curved family whenever the straight
//! lane `[1, (t−t̄)]` lies in the column span of the curved basis `Φ` — exactly for
//! the interval / line-segment charts (whose basis carries the constant and linear
//! terms) and to the basis's expressiveness for the periodic charts. So after the
//! refit `curved_rss ≤ linear_rss` up to the least-squares solver tolerance: the
//! curved family CANNOT do worse than its own `Θ = 0` sub-model. That is the
//! nested-dominance property restored here — "curved match-or-beats linear" is now
//! a floor established by re-optimizing the curved arm, not merely asserted.
//!
//! The broken (#1051) euclidean / multi-atom OUTER continuation is deliberately
//! NOT re-entered: the direct per-atom `d = 1` decoder-only refit at the realized
//! codes is sufficient to score the curved arm at its constrained minimum. When
//! `Φ` is unavailable (no evaluator) or the refit solve is degenerate the arm
//! falls back to the already-realized curve's RSS — an honest degradation, never a
//! fabricated determinant. The argmin then trades the curved arm's (minimized)
//! data fit against its larger parameter / complexity price, so a genuinely
//! curved signal is preferred while a straight-line signal ties and collapses to
//! the cheaper linear lane.
//!
//! This replaces an earlier diagnostic in which both candidates targeted the
//! atom's already-fitted decoded image `γ_k(t)` (giving the curved arm a free
//! zero residual against itself, #1202) and its successor which scored the
//! already-REALIZED curved contribution (a post-fit heuristic that did not
//! establish nested dominance); the curved arm is now re-fit to its minimum.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::chart_canonicalization::d1_atom_fitted_turning;
use crate::manifold::{SaeManifoldAtom, solve_design_least_squares};
use gam_linalg::faer_ndarray::FaerSvd;
use gam_solve::evidence::{
    HybridAtomCandidate, HybridAtomChoice, HybridSplitSelection, select_hybrid_split,
};
use gam_terms::latent::LatentManifold;

/// The rank-aware Laplace negative-log-evidence of a reduced per-atom Gaussian
/// reconstruction sub-model: `residual_objective + ½ log|H|` with no smoothing
/// penalty logdet and a full-rank design (no null space), which is the form
/// [`gam_solve::evidence::laplace_evidence`] reduces to on this comparison.
/// Kept inline (rather than routed through `EvidenceLogDetSource`) because both
/// candidates' Hessian logdets are already the closed-form scalar moments of
/// their shared design — no factor cache or HVP callback to assemble.
///
/// SCALE CAVEAT: this is a FIXED-DISPERSION Laplace / penalized criterion, not a
/// profiled REML marginal likelihood. It assumes unit dispersion (`σ² = 1`)
/// after preprocessing — the residual objective is the bare `½ RSS` with no
/// `RSS/(2σ²)` rescaling and no profiled-over-σ² log-determinant correction. It
/// is therefore SENSITIVE to the response scale: rescaling `y → s·y` scales the
/// `residual_objective` (`RSS`) by `s²` but leaves the `½ log|H|` complexity term
/// unchanged, so the curved-vs-linear trade-off it expresses is not
/// scale-invariant. Callers must keep the targets on a consistent (preprocessed,
/// roughly unit-scale) footing for the comparison to mean what it says.
fn reduced_laplace_nle(residual_objective: f64, log_det_h: f64) -> f64 {
    residual_objective + 0.5 * log_det_h
}

/// The fitted straight sub-model `γ̃(t) = b₀ + (t − t̄)·b₁` of one `d = 1` atom:
/// the exact assignment-mass-weighted least-squares line fit to the atom's
/// leave-this-atom-out RESPONSE residual `y_resp` over its assigned rows (the
/// curved family's nested `Θ = 0` sub-model on common data, #1202). Carried on a
/// verdict that selects LINEAR so the collapsed reconstruction can replace the
/// curved decoded row with this straight image at any coordinate WITHOUT
/// re-entering the (broken, #1051) outer fit — the coefficients are already
/// realized inside the adjudication.
#[derive(Clone, Debug)]
pub struct AtomLinearImage {
    /// The atom's slot index in the dictionary (so the collapsed assembly knows
    /// which atom's decoded row to substitute).
    pub atom_idx: usize,
    /// The mass-weighted coordinate mean `t̄` the line is centered on.
    pub t_bar: f64,
    /// Per-output-channel centered intercept `b₀ = γ̄` at `t̄` (length `p`).
    pub b0: Array1<f64>,
    /// Per-output-channel slope `b₁` (length `p`).
    pub b1: Array1<f64>,
    /// #1777 collapse-rescue projection DIRECTION `v` (length `p`, unit norm), the
    /// top mass-weighted output direction of the atom's leave-this-atom-out
    /// residual. `Some` exactly when this is a collapse-rescued image; `None` for
    /// the ordinary straight-image path (which decodes at the atom's own
    /// coordinate). This is the only serializable rescue coordinate model: every
    /// train or OOS reconstruction must recompute a row's coordinate as
    /// `uᵢ = ⟨y_i − Σ_{j≠k} f_j(x_i), v⟩` — identical to the train code
    /// used to fit the image and valid for held-out rows by the same model (see
    /// [`Self::coordinate_from_residual`]). Length must equal `b0`/`b1`.
    pub v: Option<Array1<f64>>,
}

impl AtomLinearImage {
    /// Evaluate the straight sub-model `b₀ + (t − t̄)·b₁` into `out` (length `p`).
    pub fn fill_row(&self, t: f64, out: &mut [f64]) {
        let dt = t - self.t_bar;
        for (j, slot) in out.iter_mut().enumerate() {
            *slot = self.b0[j] + dt * self.b1[j];
        }
    }

    /// #1777 — the collapse-rescue coordinate of a row from ITS OWN
    /// leave-this-atom-out residual `resid = y_i − Σ_{j≠k} f_j(x_i)` (length `p`),
    /// namely `uᵢ = ⟨resid, v⟩`. `Some(uᵢ)` exactly when this is a collapse-rescued
    /// image (`v` is set); `None` for the ordinary straight-image path (which has
    /// no projection direction and decodes at the atom's own coordinate).
    ///
    /// This is the SAME math [`build_collapse_rescue_linear_image`] used to fit the
    /// train image, so train and OOS share one model. Returns `None` if `resid`'s
    /// length disagrees with `v`.
    pub fn coordinate_from_residual(&self, resid: &[f64]) -> Option<f64> {
        let v = self.v.as_ref()?;
        if resid.len() != v.len() {
            return None;
        }
        Some(v.iter().zip(resid).map(|(&vj, &rj)| vj * rj).sum())
    }

    /// Whether this image is a #1777 collapse-rescued image (carries a projection
    /// direction `v`) rather than an ordinary straight image evaluated at the
    /// atom's own coordinate.
    pub fn is_collapse_rescued(&self) -> bool {
        self.v.is_some()
    }
}

/// One fitted `d = 1` atom's hybrid-split verdict, surfaced in the model output.
#[derive(Clone, Debug)]
pub struct AtomHybridVerdict {
    /// The atom's name (slot identity in the dictionary).
    pub atom_name: String,
    /// The evidence-selected parameterization choice for this slot.
    pub choice: HybridAtomChoice,
    /// `true` iff the slot kept the CURVED parameterization (the fitted atom);
    /// `false` iff it yielded to the LINEAR special case (the straight tail).
    pub kept_curved: bool,
    /// The atom's fitted turning `Θ = ∫|κ| ds` (radians), the novel geometric
    /// quantity #1026 pairs against reconstruction EV: `Θ ≈ 0` is a linear-tail
    /// direction wearing a curved basis, `Θ ≈ 2π` is a full curved loop. `None`
    /// iff the evaluator has no analytic second jet or the curve is degenerate.
    /// Captured here (not just logged) so the EV-vs-Θ frontier is queryable
    /// structured data on the persisted report rather than a transient log line.
    pub fitted_turning: Option<f64>,
    /// The atom's training leave-one-atom-out explained-variance contribution
    /// `ΔEV_k = EV(full) − EV(full∖{k})` — how much reconstruction EV this single
    /// atom earns. Paired with [`Self::fitted_turning`] this is the `(Θ, ΔEV)`
    /// point the #1026 frontier reports: a `Θ ≈ 0` atom with large `ΔEV` is a
    /// genuine linear-tail direction; a high-`Θ` atom with large `ΔEV` is a
    /// genuine curved family. `None` iff the caller did not supply LOAO EV.
    pub train_loao_delta_ev: Option<f64>,
    /// Realized reconstruction explained variance of this atom's curved
    /// contribution `a_k·γ_k(t)` against the same leave-this-atom-out response
    /// residual `y_resp` used by the curved-vs-linear adjudication.
    pub curved_ev: Option<f64>,
    /// PCA/SVD envelope: explained variance of the top-`M` centered linear
    /// subspace on the same response residual, where `M` is this atom's decoder
    /// basis size. A single chart's image lies in a row-space of dimension at
    /// most `M`, so the realized curved EV cannot beat this bound.
    pub topm_linear_ev: Option<f64>,
    /// `curved_ev / topm_linear_ev`. Near 1 means the one-chart fit is at its
    /// information ceiling; well below 1 means solver/convergence headroom.
    pub curved_vs_envelope_ratio: Option<f64>,
    /// Chart efficiency `η = EV_curved / EV_lin(top-M)`. This is the same value
    /// as [`Self::curved_vs_envelope_ratio`] under the theorem-shaped name used
    /// by the K=1 ceiling contract.
    pub chart_efficiency_eta: Option<f64>,
    /// The fitted straight sub-model for this slot, present iff the verdict
    /// selected LINEAR (`kept_curved == false`). The collapsed reconstruction
    /// substitutes this for the atom's curved decoded image, making the verdict
    /// load-bearing on the reconstruction rather than a passive diagnostic.
    pub linear_image: Option<AtomLinearImage>,
}

/// The whole dictionary's hybrid-split report: one verdict per eligible `d = 1`
/// atom, plus the dictionary-level aggregates the EV-vs-Θ frontier reports
/// against.
#[derive(Clone, Debug)]
pub struct SaeHybridSplitReport {
    /// One adjudicated verdict per eligible `d = 1` atom, in slot order. Atoms
    /// that are not eligible (wrong dim, no evaluator, mid-homotopy) are absent
    /// — they carry no curved/linear adjudication.
    pub verdicts: Vec<AtomHybridVerdict>,
    /// The dictionary-level rolled-up selection (summed NLE, total parameters,
    /// curved/linear counts) over the eligible atoms.
    pub selection: HybridSplitSelection,
}

/// Below this many assigned rows a `d = 1` atom cannot support a two-parameter
/// straight-line fit with a residual estimate, so the linear candidate's
/// deviance is undefined. Such atoms are skipped (absent from the report),
/// never adjudicated on a fabricated deviance.
const MIN_ROWS_FOR_LINEAR_FIT: usize = 3;

/// #1610/#1026 — EV-PRESERVATION gate tolerance: a `d = 1` slot may collapse to
/// its linear tail only if doing so costs at most this fraction of the target's
/// total (centered) variance in full-reconstruction explained variance. The
/// evidence argmin trades data-fit against the curved arm's parameter price in
/// `NLE` units, but on a small / low-amplitude fixture that trade can prefer the
/// cheaper line even when the curve carries real reconstruction signal — the
/// collapse then DROPS EV (the observed 1.0 → 0.748 over-collapse). This gate is
/// a direct guard on the quantity that actually regressed: the per-atom collapse
/// EV impact equals `(linear_rss − curved_rss)/SST_full` exactly (collapsing
/// atom `k` raises the full reconstruction SSR by `linear_rss − curved_rss` and
/// the full target variance `SST_full` is fixed), so a collapse is vetoed when it
/// would lose more than this fraction. EV is a dimensionless quantity in `[0,1]`,
/// so an absolute EV-loss tolerance is itself scale-invariant — it does not
/// reintroduce the scale-incommensurability that sank the evidence-reformulation
/// attempt. A genuinely straight atom (the curved fit IS its own line) loses
/// `≈ 0` EV and still collapses losslessly; only a curve doing real
/// reconstruction work (loss `≫ 1e-3`) is kept. `1e-3` = 0.1% of total variance:
/// comfortably above f64 round-off on an exact-line collapse yet far below any
/// material EV loss, so it separates the lossless and load-bearing regimes
/// without tuning.
///
/// WHY A FIXED DIMENSIONLESS TOLERANCE AND NOT A NOISE/DISPERSION-DERIVED ONE
/// (#1610). This gate is a SAFETY BACKSTOP layered over the evidence/REML
/// selection ([`select_hybrid_split`]), which is itself the nested-model
/// statistical test: it already trades the curved arm's data-fit against its
/// Laplace parameter price in NLE units and picks the line when the curve is not
/// evidence-justified. The backstop exists only to catch the residual case where
/// that argmin prefers the cheaper line yet doing so DROPS reconstruction EV (the
/// observed 1.0 → 0.748). The correct instrument for a backstop over a
/// statistical test is a conservative negligibility tolerance on the quantity it
/// protects (full-reconstruction EV), NOT a second re-derived statistical
/// threshold. A noise/dispersion-derived per-atom tolerance — e.g.
/// `df_extra · σ̂² / SST_full`, the curve's expected spurious extra RSS under the
/// null that the image is straight, with `σ̂²` the curved fit's residual
/// dispersion — has, moreover, no SAFE calibration here: on an exact-line
/// collapse the fit's dispersion `σ̂² → 0`, so a pure noise threshold falls BELOW
/// the least-squares solver round-off of `linear_rss − curved_rss`
/// (`≈ κ·εmach·SST_full`) and would spuriously VETO the lossless collapses the
/// deterministic tests require; raising it to clear round-off, conversely, only
/// relaxes the backstop TOWARD the over-collapse boundary it was added to hold
/// (larger tolerance ⇒ fewer vetoes ⇒ more collapse). There is thus no safe
/// direction to make it noise-adaptive. The safe window is the wide, well-
/// separated band between solver round-off (`~1e-12` relative) and any material
/// EV loss (`~1e-2`); `1e-3` is the standard "0.1% of variance is negligible"
/// point inside it — dimensionless on an EV `∈ [0,1]` (hence scale-invariant, per
/// this issue's scale-invariance contract) with a single explicit meaning, not a
/// corpus-tuned magnitude. (A noise-adaptive backstop would also change
/// reconstruction on real activations in a way only the real-OLMo behavioral
/// battery can validate.)
///
/// PER-ATOM SCOPE: applied per slot, this tolerance bounds only ONE atom's
/// individual EV loss. It does NOT by itself bound the dictionary-level EV loss
/// when several atoms collapse at once: the true global RSS change is
/// `Σ_k Δ_k + 2 Σ_{j<k} <Δrecon_j, Δrecon_k>` — both the accumulation of many
/// individually-tolerable `Δ_k` AND the cross terms between co-active atoms are
/// invisible to the per-atom gate. [`build_hybrid_split_report`] adds an
/// aggregate global guard (interpreting this same fraction as a bound on
/// `Σ_k max(Δ_k, 0)`) on top of the per-atom gate; see there for what that guard
/// does and does not prove.
const SAE_HYBRID_COLLAPSE_MAX_EV_LOSS: f64 = 1.0e-3;

#[derive(Clone, Copy, Debug)]
struct CurvedEnvelopeMetrics {
    curved_ev: Option<f64>,
    topm_linear_ev: Option<f64>,
    curved_vs_envelope_ratio: Option<f64>,
}

fn reconstruction_ev(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> Option<f64> {
    if target.dim() != fitted.dim() {
        return None;
    }
    let (n, p) = target.dim();
    if n == 0 || p == 0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        for row in 0..n {
            means[col] += target[[row, col]];
        }
        means[col] /= n as f64;
    }
    let mut rss = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let residual = target[[row, col]] - fitted[[row, col]];
            rss += residual * residual;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    if rss.is_finite() && sst.is_finite() && sst > f64::MIN_POSITIVE {
        Some(1.0 - rss / sst)
    } else {
        None
    }
}

fn top_m_linear_ev(target: ArrayView2<'_, f64>, basis_size: usize) -> Option<f64> {
    let (n, p) = target.dim();
    if n == 0 || p == 0 || basis_size == 0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        for row in 0..n {
            means[col] += target[[row, col]];
        }
        means[col] /= n as f64;
    }
    let mut centered = Array2::<f64>::zeros((n, p));
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let v = target[[row, col]] - means[col];
            centered[[row, col]] = v;
            sst += v * v;
        }
    }
    if !(sst.is_finite() && sst > f64::MIN_POSITIVE) {
        return None;
    }
    let (_u, sigma, _vt) = centered.svd(false, false).ok()?;
    let keep = basis_size.min(sigma.len());
    let mut captured = 0.0_f64;
    for idx in 0..keep {
        captured += sigma[idx] * sigma[idx];
    }
    if captured.is_finite() {
        Some((captured / sst).min(1.0))
    } else {
        None
    }
}

fn curved_envelope_metrics(
    assign: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
    basis_size: usize,
) -> CurvedEnvelopeMetrics {
    let n = assign.len();
    let p = decoded.ncols();
    let mut curved_fit = Array2::<f64>::zeros((n, p));
    if decoded.nrows() == n && target_resid.dim() == (n, p) {
        for row in 0..n {
            let a = assign[row];
            for col in 0..p {
                curved_fit[[row, col]] = a * decoded[[row, col]];
            }
        }
    }
    let curved_ev = reconstruction_ev(target_resid, curved_fit.view());
    let topm_linear_ev = top_m_linear_ev(target_resid, basis_size);
    let curved_vs_envelope_ratio = match (curved_ev, topm_linear_ev) {
        (Some(curved), Some(topm)) if topm.is_finite() && topm > f64::MIN_POSITIVE => {
            Some(curved / topm)
        }
        _ => None,
    };
    CurvedEnvelopeMetrics {
        curved_ev,
        topm_linear_ev,
        curved_vs_envelope_ratio,
    }
}

/// The full-reconstruction SSR INCREASE from collapsing one `d = 1` atom to its
/// fitted straight sub-model: `linear_rss − curved_rss`, where both arms are
/// scored against the atom's leave-this-atom-out response residual `y_resp`
/// (`target_resid`) exactly as [`build_atom_candidates`] scores them. Because the
/// full reconstruction differs from the collapsed one ONLY in this atom's
/// contribution (`a_k·γ_k` → `a_k·line`) on its assigned rows, this scalar is the
/// exact amount the full reconstruction's SSR rises when the slot is collapsed;
/// dividing by the fixed total target variance gives the full-EV loss the
/// EV-preservation gate keys on. Positive ⇒ the curve out-fits its straight
/// projection (collapsing hurts); `≤ 0` ⇒ the line is at least as good (collapse
/// is lossless or improving, never gated). Mirrors the `curved_rss` / `linear_rss`
/// accumulation in [`build_atom_candidates`] bit-for-bit so the gate and the
/// evidence comparison see the same residuals.
fn collapse_ssr_increase(
    coords: ArrayView1<'_, f64>,
    assign: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
    t_bar: f64,
    b0: &Array1<f64>,
    b1: &Array1<f64>,
) -> f64 {
    let n = assign.len();
    let p = target_resid.ncols();
    let mut curved_rss = 0.0_f64;
    let mut linear_rss = 0.0_f64;
    for i in 0..n {
        let a = assign[i];
        let dt = coords[i] - t_bar;
        for j in 0..p {
            let y = target_resid[[i, j]];
            let r_curved = y - a * decoded[[i, j]];
            curved_rss += r_curved * r_curved;
            let r_linear = y - a * (b0[j] + dt * b1[j]);
            linear_rss += r_linear * r_linear;
        }
    }
    linear_rss - curved_rss
}

/// #1051/#1026 NESTED MIN — re-fit the curved atom's decoder on the SAME
/// leave-this-atom-out residual `y_resp` and return its **minimum** weighted
/// reconstruction RSS at the atom's realized codes.
///
/// This is the genuine constrained minimum of the curved family over its free
/// decoder coefficients `B`:
///
/// ```text
/// min_B Σᵢ ‖ y_resp[i] − a_k·( Φ(t_i)·B ) ‖²   (design = diag(a)·Φ, rhs = y_resp).
/// ```
///
/// It restores the nested-dominance floor `curved_rss ≤ linear_rss`. The linear
/// special case `a_k·(b₀ + (t−t̄)·b₁)` is itself a member of this family whenever
/// the straight lane `[1, (t−t̄)]` lies in the column span of the curved basis
/// `Φ` — exactly (interval / line-segment charts, whose basis carries the
/// constant and linear terms) or to the basis's expressiveness (the periodic
/// charts). So `min_B` over `Φ` cannot do WORSE than the best straight line: the
/// returned RSS is `≤` the linear arm's RSS up to the least-squares solver
/// tolerance. This is the direct per-atom `d = 1` refit the module owes; the
/// broken (#1051) euclidean / multi-atom OUTER continuation is deliberately NOT
/// re-entered — the decoder-only refit at the realized codes is sufficient to
/// score the curved arm at its constrained minimum.
///
/// `phi` is the curved design `Φ(t)` on the atom's rows (`n × M`); `assign` the
/// per-row mass `a_k` (NOT squared — the design weight is the mass itself, so the
/// residual is on the SAME footing the linear arm and the joint loss use).
/// Returns `None` when the solve is degenerate or non-finite; the caller then
/// falls back to the already-realized curve's RSS rather than fabricate a value.
///
/// #2023 DEMOTE surface: returns the fitted decoder `B` (`M × p`, the
/// constrained-minimum curved decoder) and its mass-weighted basis Gram
/// `G = ΦᵀWΦ = designᵀdesign` (`M × M`, `W = diag(a²)`) alongside the RSS — the
/// two inputs `realised_rank_charge_dof` needs to price the curved arm's realised
/// rank in the SAME currency the fit's REML criterion uses. `curved_refit_rss`
/// below is the historical RSS-only wrapper (unchanged `Option<f64>` contract).
fn curved_refit_decoder(
    phi: ArrayView2<'_, f64>,
    assign: ArrayView1<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
) -> Option<(f64, Array2<f64>, Array2<f64>)> {
    let n = phi.nrows();
    let m = phi.ncols();
    let p = target_resid.ncols();
    if m == 0 || n == 0 || assign.len() != n || target_resid.nrows() != n || p == 0 {
        return None;
    }
    // Weighted design `diag(a)·Φ` (n×M). The refit minimizes ‖diag(a)·Φ·B − y_resp‖²,
    // so the fitted prediction is diag(a)·Φ·B — the curved contribution `a_k·γ_k`
    // at its best decoder `B` on this residual.
    let mut design = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let a = assign[i];
        if !a.is_finite() {
            return None;
        }
        for c in 0..m {
            design[[i, c]] = a * phi[[i, c]];
        }
    }
    let b = solve_design_least_squares(design.view(), target_resid).ok()?;
    if b.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let pred = design.dot(&b);
    let mut rss = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let r = target_resid[[i, j]] - pred[[i, j]];
            rss += r * r;
        }
    }
    if !rss.is_finite() {
        return None;
    }
    // Mass-weighted basis Gram `G = designᵀdesign = ΦᵀWΦ` (W = diag(a²)) — the
    // rank-charge d_eff's `gram` input (its MP-count + basis_edf are read off G).
    let gram = design.t().dot(&design);
    Some((rss, b, gram))
}

/// Curved-arm refit RSS only — the scalar data-fit the hybrid selector scores.
/// Thin wrapper over [`curved_refit_decoder`]; identical to the historical
/// `Option<f64>` contract (the decoder + Gram are dropped here, consumed by the
/// rank-charge DEMOTE gate through the decoder-returning form).
fn curved_refit_rss(
    phi: ArrayView2<'_, f64>,
    assign: ArrayView1<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
) -> Option<f64> {
    curved_refit_decoder(phi, assign, target_resid).map(|(rss, _, _)| rss)
}

/// Build the curved + linear candidates for ONE fitted `d = 1` atom and return
/// them as `(linear, curved, (t̄, b₀, b₁))`, or `None` if the atom cannot present
/// an honest pair (too few rows, degenerate coordinate span, or non-finite
/// numbers). Both candidates are scored against the SAME data — the atom's
/// leave-this-atom-out response residual `y_resp` — at their CONSTRAINED MINIMUM:
/// the linear arm is the freshly-fit min-over-lines and the curved arm is the
/// min-over-decoders refit ([`curved_refit_rss`]), so the comparison is a genuine
/// nested min-vs-min one and `curved_rss ≤ linear_rss` holds up to solver
/// tolerance for a basis whose span contains the straight lane (#1051).
///
/// Inputs over the atom's assigned rows:
///   * `coords` — the fitted on-atom coordinate `t`.
///   * `assign` — the per-row assignment mass `a_k` (NOT squared; this routine
///     squares it where the design weight `a_k²` is needed).
///   * `decoded` — the atom's fitted decoded image `γ_k(t) = Φ(t) B_k` (`p` cols),
///     whose mass-scaled value `a_k·γ_k` is the curved candidate's PREDICTION.
///   * `target_resid` — the atom's leave-this-atom-out response residual `y_resp`
///     (`p` cols): the response with every OTHER atom's contribution removed.
///     This is the data both candidates fit.
///
/// The curved candidate's data-fit deviance is `½·min_B Σ ‖y_resp − a_k·(Φ·B)‖²`
/// (its constrained minimum over the decoder; the mass lives in the prediction);
/// the linear candidate fits the best mass-weighted straight line to `y_resp` and
/// pays `½ Σ ‖y_resp − a_k·(b₀ + (t − t̄)·b₁)‖²`. Because the linear prediction is
/// itself a curved-family member (the straight lane lies in `span(Φ)` for the
/// eligible charts), the curved arm's minimized RSS is `≤` the linear arm's up to
/// solver tolerance, so the argmin is a genuine nested min-vs-min dominance
/// comparison, not a post-fit compression heuristic.
/// Why [`build_atom_candidates`] declined to produce a linear/curved candidate
/// pair. The distinction is load-bearing (#2362): only a genuine coordinate
/// collapse is a #1777 rescue site; every other refusal means the slot cannot be
/// adjudicated at all and must be left curved, NOT decoded through a
/// target-dependent rescue image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomCandidateRefusal {
    /// The atom's own coordinate collapsed to a single point (`s_tt ≈ 0`, the
    /// rank-1 "chord-through-the-arc" fixed point). The caller recovers a fresh
    /// linear image from the residual's top direction (#1026/#1777).
    CoordinateCollapse,
    /// The candidate pair could not be built or priced for any other reason:
    /// too few rows / shape mismatch, a non-finite or negative assignment mass,
    /// zero or non-finite design weight, an absent or misshaped curved basis,
    /// an unusable dispersion / `n_obs`, a rank-charge factorization failure, or
    /// a non-finite evidence score. NONE of these is a collapse: there is no
    /// degenerate curve to rescue, so the slot stays curved (unadjudicated).
    Unadjudicable,
}

/// The closed-form mass-weighted straight-line fit of one `d = 1` atom's
/// coordinate against a `p`-channel response `y`, plus the derived quantities the
/// callers need. Shared by the fitted evidence adjudication
/// ([`build_atom_candidates`], where `y = y_resp` is the leave-this-atom-out
/// residual) and the fit-free no-target collapse verdict
/// ([`fit_free_linear_image`], where `y = a_k·γ_k` is the atom's own realized
/// contribution). Keeping ONE implementation of the normal equations guarantees
/// the fit-free straight image is bit-identical to the one a full fit would mint
/// for the same atom.
struct MassWeightedLine {
    /// Mass-weighted coordinate mean `t̄` the centered line is anchored on.
    t_bar: f64,
    /// Per-channel centered intercept `b₀` (length `p`).
    b0: Array1<f64>,
    /// Per-channel slope `b₁` (length `p`).
    b1: Array1<f64>,
    /// Effective mass `Σ a²` (the intercept Gram diagonal / `N_eff`).
    w_sum: f64,
    /// Mass-weighted coordinate spread `Σ a²·(t − t̄)²` (the slope Gram diagonal).
    s_tt: f64,
    /// The line's weighted residual sum of squares against `y`:
    /// `Σᵢ ‖y[i] − a_i·(b₀ + (t_i − t̄)·b₁)‖²`.
    linear_rss: f64,
}

/// Fit [`MassWeightedLine`] from `(coords, assign, y)` with design weight
/// `wᵢ = a_i²` and prediction `a_i·(b₀ + (t_i − t̄)·b₁)`. Returns the SAME
/// refusals [`build_atom_candidates`] surfaces: `CoordinateCollapse` when the
/// atom's coordinate has collapsed to a single point (`s_tt ≈ 0`), `Unadjudicable`
/// for a shape mismatch, a non-finite/negative mass, or a degenerate `w_sum`.
fn fit_mass_weighted_line(
    coords: ArrayView1<'_, f64>,
    assign: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Result<MassWeightedLine, AtomCandidateRefusal> {
    let n = coords.len();
    let p = y.ncols();
    if assign.len() != n || y.nrows() != n || p == 0 {
        return Err(AtomCandidateRefusal::Unadjudicable);
    }
    // The LINEAR candidate fits `a_k·(b₀ + (t − t̄)·b₁)` to `y`, so the natural
    // design column is `a_k·[1, (t − t̄)]` and the per-row Gram weight is
    // `wᵢ = a_k²`. Accumulate the mass-weighted coordinate mean `t̄` and spread
    // `s_tt` under that weight; a row that barely belongs to the atom (`a_k ≈ 0`)
    // contributes ≈ nothing, exactly as in the joint loss.
    let mut w_sum = 0.0_f64;
    let mut t_bar = 0.0_f64;
    for i in 0..n {
        let a = assign[i];
        if !(a.is_finite() && a >= 0.0) {
            return Err(AtomCandidateRefusal::Unadjudicable);
        }
        let w = a * a;
        w_sum += w;
        t_bar += w * coords[i];
    }
    if !(w_sum > 0.0) {
        return Err(AtomCandidateRefusal::Unadjudicable);
    }
    t_bar /= w_sum;

    // Weighted Σ wᵢ·(t − t̄)² with `wᵢ = a_k²` — the coordinate spread under the
    // line's design weight. A degenerate (single-point mass) coordinate has no
    // slope direction; refuse rather than divide by ~0.
    let mut s_tt = 0.0_f64;
    for i in 0..n {
        let dt = coords[i] - t_bar;
        s_tt += assign[i] * assign[i] * dt * dt;
    }
    // The atom's OWN coordinate has collapsed to a single point (the #1777
    // rank-1 "chord-through-the-arc" co-collapse). This is the SOLE refusal the
    // hybrid split rescues: the caller recovers a fresh linear image from the
    // residual's top direction. Every OTHER refusal is `Unadjudicable`.
    if !(s_tt > 1e-12 * (1.0 + t_bar * t_bar)) {
        return Err(AtomCandidateRefusal::CoordinateCollapse);
    }

    // Per-output-channel mass-weighted least squares. Minimizing
    // `Σᵢ ‖y[i] − a_k·(b₀ + (t − t̄)·b₁)‖²` in the centered basis has the diagonal
    // normal equations
    //   b₀[j] = (Σ a_k·y[i,j]) / w_sum,   (recall the design intercept is a_k)
    //   b₁[j] = (Σ a_k·(t − t̄)·y[i,j]) / s_tt.
    let mut b0 = Array1::<f64>::zeros(p);
    let mut b1 = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut s_1y = 0.0_f64;
        let mut s_ty = 0.0_f64;
        for i in 0..n {
            let a = assign[i];
            let dt = coords[i] - t_bar;
            let yij = y[[i, j]];
            s_1y += a * yij;
            s_ty += a * dt * yij;
        }
        b0[j] = s_1y / w_sum;
        b1[j] = s_ty / s_tt;
    }

    // The line's weighted RSS against `y`.
    let mut linear_rss = 0.0_f64;
    for i in 0..n {
        let a = assign[i];
        let dt = coords[i] - t_bar;
        for j in 0..p {
            let r_linear = y[[i, j]] - a * (b0[j] + dt * b1[j]);
            linear_rss += r_linear * r_linear;
        }
    }

    Ok(MassWeightedLine {
        t_bar,
        b0,
        b1,
        w_sum,
        s_tt,
        linear_rss,
    })
}

/// #1026 fit-free (no-target) collapse verdict for one `d = 1` atom. Given the
/// atom's own REALIZED contribution `realized[i] = a_i·γ_k(t_i)` over its assigned
/// rows, decide whether that contribution already lies in the straight lane
/// `span{1, (t − t̄)}` to numerical tolerance, and if so return the straight image
/// that reproduces it. This is the honest verdict the public no-target
/// reconstruction can reach WITHOUT a fitted `hybrid_split_report`: the fitted
/// adjudication needs a response to price curvature against, but a term with no
/// target can still observe that its realized decoded image is (to `ε`) a line in
/// `t` and collapse to it EV-neutrally.
///
/// The straight image is the exact [`MassWeightedLine`] the fitted path would
/// mint, and `v = None` (an ordinary straight image decoded at the atom's own
/// coordinate — never a target-dependent #1777 rescue). Returns `None` when the
/// atom's realized image is genuinely curved (relative line residual above
/// `rel_floor`), its coordinate has collapsed, its mass is degenerate, or its
/// realized image carries no energy (a zero atom — nothing to collapse).
///
/// `rel_floor` is the squared relative-residual ceiling: collapse iff
/// `linear_rss ≤ rel_floor · ‖realized‖²`. It is a numerical-linearity gate, NOT
/// a modeling knob — only an image that a straight line reproduces to `√rel_floor`
/// relative amplitude is admitted, so the substitution cannot degrade the
/// reconstruction beyond that floor (collapse-safety).
pub(crate) fn fit_free_linear_image(
    atom_idx: usize,
    coords: ArrayView1<'_, f64>,
    assign: ArrayView1<'_, f64>,
    realized: ArrayView2<'_, f64>,
    rel_floor: f64,
) -> Option<AtomLinearImage> {
    let n = coords.len();
    if n < MIN_ROWS_FOR_LINEAR_FIT {
        return None;
    }
    let line = fit_mass_weighted_line(coords, assign, realized).ok()?;
    // Total weighted energy of the realized image — the denominator that makes
    // the linearity test scale-invariant. A zero-energy atom (no realized mass)
    // has nothing to collapse.
    let mut energy = 0.0_f64;
    for i in 0..n {
        for j in 0..realized.ncols() {
            energy += realized[[i, j]] * realized[[i, j]];
        }
    }
    if !(energy > 0.0 && energy.is_finite() && line.linear_rss.is_finite()) {
        return None;
    }
    if line.linear_rss > rel_floor * energy {
        // Genuinely curved realized image: collapsing would degrade it. Leave curved.
        return None;
    }
    Some(AtomLinearImage {
        atom_idx,
        t_bar: line.t_bar,
        b0: line.b0,
        b1: line.b1,
        v: None,
    })
}

/// Squared relative-residual ceiling of the #1026 fit-free linearity gate: a
/// realized image is treated as a straight line iff its weighted line-fit RSS is
/// below this fraction of its total energy, i.e. the line reproduces it to `1e-6`
/// relative amplitude. Tight enough that only a numerically-linear realized image
/// collapses (a genuine arc sits orders of magnitude above it), loose enough to
/// admit the solver roundoff of an exactly-representable line.
pub(crate) const FIT_FREE_LINEAR_RELATIVE_FLOOR: f64 = 1e-12;

fn build_atom_candidates(
    coords: ArrayView1<'_, f64>,
    assign: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
    curved_num_params: usize,
    curved_phi: Option<ArrayView2<'_, f64>>,
    fitted_turning: Option<f64>,
    // #16 DEMOTE: price both arms in the joint fit's canonical currency —
    // ½·d_eff·log(n_obs) on the realised decoder rank. `n_obs` is the term's full
    // row count; `dispersion_r` is its reconstruction φ̂ (the MP-edge noise floor).
    n_obs: usize,
    dispersion_r: f64,
) -> Result<
    (
        HybridAtomCandidate,
        HybridAtomCandidate,
        (f64, Array1<f64>, Array1<f64>),
    ),
    AtomCandidateRefusal,
> {
    let n = coords.len();
    let p = decoded.ncols();
    if n < MIN_ROWS_FOR_LINEAR_FIT
        || decoded.nrows() != n
        || assign.len() != n
        || target_resid.nrows() != n
        || target_resid.ncols() != p
        || p == 0
    {
        return Err(AtomCandidateRefusal::Unadjudicable);
    }

    // The LINEAR candidate fits `a_k·(b₀ + (t − t̄)·b₁)` to the residual `y_resp`
    // via the shared mass-weighted normal equations (design column `a_k·[1,(t−t̄)]`,
    // per-row Gram weight `wᵢ = a_k²`), which also surfaces the coordinate mean
    // `t̄`, spread `s_tt`, effective mass `w_sum` and the line's `linear_rss`.
    // A single-point coordinate collapse becomes the #1777 rescue site.
    let MassWeightedLine {
        t_bar,
        b0,
        b1,
        w_sum,
        s_tt,
        linear_rss,
    } = fit_mass_weighted_line(coords, assign, target_resid)?;

    // #1051 NESTED MIN — the curved arm's data fit is `min_B ‖y_resp − diag(a)Φ B‖²`,
    // its constrained minimum over the decoder. Because the linear lane is a member
    // of the curved family (the straight columns lie in `span(Φ)` for the eligible
    // charts), this min-curved RSS is `≤ linear_rss` up to solver tolerance — the
    // "curved match-or-beats linear" floor. Without an evaluable basis and exact
    // decoder refit there is no comparable curved-model evidence, so refuse the
    // adjudication instead of substituting a different objective.
    let curved_phi = curved_phi
        .filter(|phi| phi.nrows() == n)
        .ok_or(AtomCandidateRefusal::Unadjudicable)?;
    let curved_rss = curved_refit_rss(curved_phi, assign, target_resid)
        .ok_or(AtomCandidateRefusal::Unadjudicable)?;

    // Gaussian-reconstruction deviance: the residual objective `½ RSS` the
    // Laplace normalizer is added to. The curved arm pays `½·curved_rss` (how
    // well its REALIZED curve explains the residual) plus its larger `M·p`
    // parameter price; the linear arm pays `½·linear_rss` plus a `2·p` price.
    // `curved_rss` is the realized (not re-optimized) curve's misfit, so it is NOT
    // guaranteed `≤ linear_rss`: when the realized curve underperforms its own best
    // straight projection the cheaper line simply wins. The argmin trades whatever
    // data-fit the realized curve buys against the curvature parameter price — a
    // post-fit compression decision, not a nested match-or-beat floor.
    let curved_residual_objective = 0.5 * curved_rss;
    let linear_residual_objective = 0.5 * linear_rss;

    // Linear candidate parameter price: intercept + slope per output channel.
    let linear_num_params = 2 * p;

    if !(w_sum > 0.0 && w_sum.is_finite() && s_tt.is_finite()) {
        return Err(AtomCandidateRefusal::Unadjudicable);
    }
    let (linear_nle, curved_nle) = {
        // #16 DEMOTE currency swap: charge ½·d_eff·log(n_obs) (realised decoder rank,
        // the SAME quantity the joint REML PROMOTE gate charges) in place of the
        // ½log|H| Laplace det (the #5-mispriced term + its column-symmetric ·p
        // over-count). d_eff = realised_rank_charge_dof(G, B, N_eff, p, R): a real
        // rank-2 circle → ~2×basis_edf, a vanishing decoder → 0. The migration gate
        // (curve earns Tier-2 iff Δloss > ½·Δd_eff·log n) then falls out of the SAME
        // select_hybrid_atom NLE comparison — one currency, no separate margin.
        if n_obs == 0 || !(dispersion_r.is_finite() && dispersion_r > 0.0) {
            return Err(AtomCandidateRefusal::Unadjudicable);
        }
        let n_obs_ln = (n_obs as f64).ln();
        let n_eff = w_sum; // effective sample size Σa² (MP-edge aspect)
        // Linear arm: decoder B=[b₀;b₁] (2×p), Gram G=diag(w_sum, s_tt) (2×2).
        let mut b_lin = Array2::<f64>::zeros((2, p));
        for j in 0..p {
            b_lin[[0, j]] = b0[j];
            b_lin[[1, j]] = b1[j];
        }
        let mut g_lin = Array2::<f64>::zeros((2, 2));
        g_lin[[0, 0]] = w_sum;
        g_lin[[1, 1]] = s_tt;
        let d_lin = crate::manifold::realised_rank_charge_dof(
            &g_lin,
            &b_lin,
            n_eff,
            p as f64,
            dispersion_r,
            0.0,
            None,
        )
        .map_err(|_| AtomCandidateRefusal::Unadjudicable)?;
        // Curved arm: refit decoder B + Gram G=ΦᵀWΦ on the same residual.
        let (_, b_curved, g_curved) = curved_refit_decoder(curved_phi, assign, target_resid)
            .ok_or(AtomCandidateRefusal::Unadjudicable)?;
        let d_curved = crate::manifold::realised_rank_charge_dof(
            &g_curved,
            &b_curved,
            n_eff,
            p as f64,
            dispersion_r,
            0.0,
            None,
        )
        .map_err(|_| AtomCandidateRefusal::Unadjudicable)?;
        // DEVIANCE, not raw SSE (#2124 units fix): the rank charge `½·d_eff·ln n`
        // is dimensionless, so trading it against the bare `½·RSS` makes the
        // linear↔curved decision depend on the response scale (exactly the
        // sensitivity `reduced_laplace_nle`'s SCALE CAVEAT documents). Divide the
        // residual objective by the term's reconstruction dispersion φ̂
        // (`dispersion_r`) so the boundary is `Δ(½RSS)/φ̂ vs ½·Δd_eff·ln n` —
        // scale-invariant, the BIC large-n limit of the Laplace evidence in
        // proper units.
        let inv_dispersion = dispersion_r.recip();
        (
            reduced_laplace_nle(linear_residual_objective * inv_dispersion, d_lin * n_obs_ln),
            reduced_laplace_nle(
                curved_residual_objective * inv_dispersion,
                d_curved * n_obs_ln,
            ),
        )
    };
    if !(linear_nle.is_finite() && curved_nle.is_finite()) {
        return Err(AtomCandidateRefusal::Unadjudicable);
    }

    let linear = HybridAtomCandidate::linear(linear_nle, linear_num_params);
    let curved = HybridAtomCandidate::curved(1, curved_nle, curved_num_params, fitted_turning);
    Ok((linear, curved, (t_bar, b0, b1)))
}

/// #1026 collapse rescue. When a `d = 1` atom's own coordinate has collapsed to a
/// single point (`build_atom_candidates` refuses because `s_tt ≈ 0`), the atom is
/// stuck in the degenerate "chord-through-the-arc" fixed point and its curved
/// decode is a constant — the rank-1 dictionary co-collapse (real-OLMo held-out EV
/// ≈ 0.13 vs the rank-K linear ceiling ≈ 0.74). The hybrid-split was DESIGNED to
/// let such a linear-tail atom decode as a straight line; the only reason it can't
/// here is that its own codes carry no spread to fit a slope against.
///
/// Recover FRESH per-row codes from the data instead: `uᵢ = yᵢ·v`, the projection
/// of the leave-this-atom-out residual onto its top mass-weighted output direction
/// `v` (the rank-1 of `Σᵢ wᵢ yᵢyᵢᵀ`, `wᵢ = a_k²` — the SAME design weight the line
/// fit uses). These codes span the residual's strongest linear axis by
/// construction, so the straight image `b₀ + (uᵢ − ū)·b₁` fit against them
/// reconstructs that axis at LINEAR quality — exactly the linear-tail reach the
/// split owes. Returns the forced-LINEAR candidate plus the image carrying `v`,
/// or `None` when the residual itself carries no usable direction (a genuine zero
/// atom the mass/decoder guards own). The per-row `uᵢ` values are fit-local and
/// are deliberately not stored; reconstruction must project its own target-aware
/// residual through `v`.
fn build_collapse_rescue_linear_image(
    atom_idx: usize,
    assign: ArrayView1<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
) -> Option<(HybridAtomCandidate, AtomLinearImage)> {
    let n = assign.len();
    let p = target_resid.ncols();
    if n < MIN_ROWS_FOR_LINEAR_FIT || target_resid.nrows() != n || p == 0 {
        return None;
    }
    let mut w_sum = 0.0_f64;
    for i in 0..n {
        let a = assign[i];
        if !(a.is_finite() && a >= 0.0) {
            return None;
        }
        w_sum += a * a;
    }
    if !(w_sum > 0.0) {
        return None;
    }
    // Top mass-weighted output direction `v` of the residual via power iteration on
    // `M = Σᵢ wᵢ yᵢyᵢᵀ` (p×p, never materialized): `v ← normalize(Σᵢ wᵢ yᵢ (yᵢ·v))`.
    // Seed from the per-channel weighted energy so a rank-1 residual converges in
    // one step and the seed is deterministic (no RNG).
    let mut v = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut e = 0.0_f64;
        for i in 0..n {
            let a = assign[i];
            let y = target_resid[[i, j]];
            e += a * a * y * y;
        }
        v[j] = e;
    }
    let mut vnorm = v.dot(&v).sqrt();
    if !(vnorm > 0.0) {
        return None;
    }
    v.mapv_inplace(|x| x / vnorm);
    for _ in 0..32 {
        let mut mv = Array1::<f64>::zeros(p);
        for i in 0..n {
            let a = assign[i];
            let w = a * a;
            let mut proj = 0.0_f64;
            for j in 0..p {
                proj += target_resid[[i, j]] * v[j];
            }
            let wp = w * proj;
            for j in 0..p {
                mv[j] += wp * target_resid[[i, j]];
            }
        }
        vnorm = mv.dot(&mv).sqrt();
        if !(vnorm > 0.0) {
            return None;
        }
        mv.mapv_inplace(|x| x / vnorm);
        let cos = mv.dot(&v).abs();
        v = mv;
        if cos > 1.0 - 1e-12 {
            break;
        }
    }
    // Fresh per-row codes `uᵢ = yᵢ·v` and the weighted line fit against them.
    let mut u = Array1::<f64>::zeros(n);
    let mut t_bar = 0.0_f64;
    for i in 0..n {
        let mut proj = 0.0_f64;
        for j in 0..p {
            proj += target_resid[[i, j]] * v[j];
        }
        u[i] = proj;
        t_bar += assign[i] * assign[i] * proj;
    }
    t_bar /= w_sum;
    let mut s_tt = 0.0_f64;
    for i in 0..n {
        let dt = u[i] - t_bar;
        s_tt += assign[i] * assign[i] * dt * dt;
    }
    if !(s_tt > 1e-12 * (1.0 + t_bar * t_bar)) {
        return None;
    }
    let mut b0 = Array1::<f64>::zeros(p);
    let mut b1 = Array1::<f64>::zeros(p);
    let mut linear_rss = 0.0_f64;
    for j in 0..p {
        let mut s_1y = 0.0_f64;
        let mut s_ty = 0.0_f64;
        for i in 0..n {
            let a = assign[i];
            let dt = u[i] - t_bar;
            let y = target_resid[[i, j]];
            s_1y += a * y;
            s_ty += a * dt * y;
        }
        b0[j] = s_1y / w_sum;
        b1[j] = s_ty / s_tt;
    }
    for i in 0..n {
        let a = assign[i];
        let dt = u[i] - t_bar;
        for j in 0..p {
            let r = target_resid[[i, j]] - a * (b0[j] + dt * b1[j]);
            linear_rss += r * r;
        }
    }
    let linear_log_det_h = (p as f64) * (w_sum.ln() + s_tt.ln());
    let linear_nle = reduced_laplace_nle(0.5 * linear_rss, linear_log_det_h);
    if !linear_nle.is_finite() {
        return None;
    }
    let linear = HybridAtomCandidate::linear(linear_nle, 2 * p);
    let image = AtomLinearImage {
        atom_idx,
        t_bar,
        b0,
        b1,
        // #1777 — persist the projection direction so every row's coordinate is
        // recomputed as ⟨residual, v⟩ rather than using a train-only cached code
        // or the atom's collapsed own coordinate.
        v: Some(v),
    };
    Some((linear, image))
}

/// #1026 item-2 — one collapsed slot's TRUE dictionary-level reconstruction
/// change `δ_k[i,j] = a_k·(γ_k(t_i) − line_k(t_i))` over ALL globally-aligned rows
/// (the caller presents `coords`/`assign`/`decoded`/`image` on the same `n` rows,
/// so these δ vectors ARE cross-atom aligned and their inner products are the
/// genuine cross terms). Ordinary images are evaluated at the atom coordinate;
/// collapse-rescued images project the same leave-this-atom-out `target_resid`
/// used to fit them. Collapsing atom `k` shifts the full reconstruction residual
/// by `+δ_k`.
fn slot_delta(
    coords: &Array1<f64>,
    assign: &Array1<f64>,
    decoded: &Array2<f64>,
    target_resid: &Array2<f64>,
    image: &AtomLinearImage,
) -> Result<Array2<f64>, String> {
    let n = decoded.nrows();
    let p = decoded.ncols();
    if coords.len() != n
        || assign.len() != n
        || target_resid.dim() != (n, p)
        || image.b0.len() != p
        || image.b1.len() != p
    {
        return Err(format!(
            "slot_delta: incompatible shapes coords={}, assign={}, decoded={:?}, target_resid={:?}, b0={}, b1={}",
            coords.len(),
            assign.len(),
            decoded.dim(),
            target_resid.dim(),
            image.b0.len(),
            image.b1.len()
        ));
    }
    let mut d = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let a = assign[i];
        let coord = if image.is_collapse_rescued() {
            let residual_row = target_resid.row(i);
            let residual = residual_row
                .as_slice()
                .expect("target_resid row must be contiguous");
            image.coordinate_from_residual(residual).ok_or_else(|| {
                format!(
                    "slot_delta: collapse-rescued image for atom {} cannot project a {p}-channel residual",
                    image.atom_idx
                )
            })?
        } else {
            coords[i]
        };
        let dt = coord - image.t_bar;
        for j in 0..p {
            let line = image.b0[j] + dt * image.b1[j];
            d[[i, j]] = a * (decoded[[i, j]] - line);
        }
    }
    Ok(d)
}

/// #1026 item-2 — the ALL-CURVED global reconstruction residual
/// `R0 = target − Σ_all a·γ` recovered from any single atom's leave-this-atom-out
/// residual: `R0 = y_resp_k − a_k·γ_k = target_resid − a_k·decoded`. Identical for
/// every atom (each `target_resid` adds back exactly that atom's own contribution),
/// so the caller computes it once from the first slot.
fn slot_r0(assign: &Array1<f64>, decoded: &Array2<f64>, target_resid: &Array2<f64>) -> Array2<f64> {
    let n = decoded.nrows();
    let p = decoded.ncols();
    let mut r0 = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let a = assign[i];
        for j in 0..p {
            r0[[i, j]] = target_resid[[i, j]] - a * decoded[[i, j]];
        }
    }
    r0
}

/// #1026 item-2 — GLOBAL cross-term collapse guard. Given the collapsed slots'
/// dictionary-level reconstruction-change vectors `δ_k` (globally row-aligned,
/// `n × p`) and the all-curved global residual `R0 = target − Σ_all a·γ`, decide
/// which collapses must revert to curved so the TRUE global reconstruction SSR
/// increase
///
/// ```text
/// ΔRSS(S) = ‖R0 + Σ_{k∈S} δ_k‖² − ‖R0‖²
///         = 2⟨R0, Σ_{k∈S} δ_k⟩ + ‖Σ_{k∈S} δ_k‖²
///         = Σ_{k∈S} Δ_k + 2 Σ_{j<k∈S} ⟨δ_j, δ_k⟩
/// ```
///
/// stays within `global_tol`. Unlike the per-atom / summed-loss guard this
/// INCLUDES the cross terms `2 Σ_{j<k} ⟨δ_j, δ_k⟩` between simultaneously-collapsed
/// atoms (correlated collapse errors), which the aggregate `Σ max(Δ_k, 0)` bound is
/// blind to. `forced` are the always-collapsed δ (euclidean / collapse-rescue slots
/// with no curved alternative): they stay in the reconstruction but cannot be
/// rolled back. `eligible` are `(slot, curved_evidence_margin, δ)` for
/// rollback-eligible collapses. When `ΔRSS` over ALL collapses exceeds tolerance,
/// the least-justified eligible collapses (largest margin — the most marginal
/// linear win) are reverted one at a time, RECOMPUTING the true global increase
/// (cross terms and all) after each revert, until within tolerance or none remain.
/// Returns the slot indices to revert to curved.
fn global_collapse_rollback(
    r0: ArrayView2<'_, f64>,
    eligible: &[(usize, f64, &Array2<f64>)],
    forced: &[&Array2<f64>],
    global_tol: f64,
) -> Vec<usize> {
    let (n, p) = r0.dim();
    // True global SSR increase for the active δ set: 2⟨R0, Σδ⟩ + ‖Σδ‖².
    let increase = |active: &[&Array2<f64>]| -> f64 {
        let mut cross = 0.0_f64;
        let mut self_sq = 0.0_f64;
        for i in 0..n {
            for j in 0..p {
                let mut s = 0.0_f64;
                for d in active {
                    s += d[[i, j]];
                }
                cross += r0[[i, j]] * s;
                self_sq += s * s;
            }
        }
        2.0 * cross + self_sq
    };
    let mut kept = vec![true; eligible.len()];
    let build_active = |kept: &[bool]| -> Vec<&Array2<f64>> {
        let mut v: Vec<&Array2<f64>> = forced.to_vec();
        for (idx, &(_, _, d)) in eligible.iter().enumerate() {
            if kept[idx] {
                v.push(d);
            }
        }
        v
    };
    if increase(&build_active(&kept)) <= global_tol {
        return Vec::new();
    }
    // Revert least-justified first: largest curved_evidence_margin (the most
    // marginal linear win is the cheapest to give back to curved).
    let mut order: Vec<usize> = (0..eligible.len()).collect();
    order.sort_by(|&a, &b| {
        eligible[b]
            .1
            .partial_cmp(&eligible[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut reverted = Vec::new();
    for idx in order {
        kept[idx] = false;
        reverted.push(eligible[idx].0);
        if increase(&build_active(&kept)) <= global_tol {
            break;
        }
    }
    reverted
}

/// Assemble the per-atom candidate slots for [`select_hybrid_split`] from the
/// fitted `d = 1` atoms, run the adjudication, and return the report.
///
/// `atoms` are the fitted dictionary atoms; `coords_for` yields the on-atom
/// coordinate column for a slot, `assign_for` the per-row assignment mass `a_k`,
/// `decoded_for` the fitted decoded image rows `γ_k`, and `target_resid_for` the
/// atom's leave-this-atom-out response residual `y_resp` (the data both
/// candidates are scored against, #1202). `manifold_for` yields the atom's chart
/// manifold (a flat / Euclidean chart can present only the linear candidate,
/// enforced inside the selector).
///
/// Returns `None` (no report) when no atom is eligible — there is nothing to
/// adjudicate.
pub fn build_hybrid_split_report<'a, C, W, D, R, M, E>(
    atoms: &'a [SaeManifoldAtom],
    eligible_d1: impl Iterator<Item = usize>,
    mut coords_for: C,
    mut assign_for: W,
    mut decoded_for: D,
    mut target_resid_for: R,
    mut manifold_for: M,
    mut delta_ev_for: E,
    // #1026 — the full target's total (column-centered) variance `SST_full`, the
    // fixed denominator of the EV-preservation gate. `≤ 0` / non-finite disables
    // the gate (a degenerate, varianceless target has no EV to preserve).
    total_centered_variance: f64,
    // #16 DEMOTE rank-charge currency. `n_obs` is the term's row count (the log-n
    // BIC scale, matching PROMOTE); `dispersion_r` is the reconstruction noise
    // floor φ̂ for the MP edge.
    n_obs: usize,
    dispersion_r: f64,
) -> Result<Option<SaeHybridSplitReport>, String>
where
    C: FnMut(usize) -> Array1<f64>,
    W: FnMut(usize) -> Array1<f64>,
    D: FnMut(usize) -> Array2<f64>,
    R: FnMut(usize) -> Array2<f64>,
    M: FnMut(usize) -> LatentManifold,
    // The atom's held-out LOAO `ΔEV_k`, keyed by atom index. `None` when LOAO EV
    // is unavailable (e.g. the caller has no target to measure against).
    E: FnMut(usize) -> Option<f64>,
{
    let mut slots: Vec<Vec<HybridAtomCandidate>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut manifolds: Vec<LatentManifold> = Vec::new();
    // Per-slot fitted straight sub-model `(atom_idx, t̄, b₀, b₁)`, surfaced onto
    // the verdict iff the slot selects LINEAR so the collapsed reconstruction can
    // substitute it for the curved decoded image.
    let mut linear_images: Vec<AtomLinearImage> = Vec::new();
    // Per-slot `(Θ, ΔEV)` — the #1026 frontier point — carried onto each verdict
    // so the geometry/EV pairing is structured report data, not a log line.
    let mut turnings: Vec<Option<f64>> = Vec::new();
    let mut delta_evs: Vec<Option<f64>> = Vec::new();
    let mut envelope_metrics: Vec<CurvedEnvelopeMetrics> = Vec::new();
    // #1026 item-2 — per-slot collapse loss `Δ_k = linear_rss − curved_rss` for the
    // GLOBAL EV-preservation guard below. `Some(Δ_k)` for a curveable slot that
    // retained a curved alternative (so a chosen collapse there can be rolled back);
    // `None` for euclidean slots (no curved option) and collapse-rescue slots (the
    // curve was already degenerate — collapsing recovers EV rather than losing it).
    let mut collapse_loss: Vec<Option<f64>> = Vec::new();
    // #1026 item-2 — per-slot dictionary-level reconstruction-change vector δ_k
    // (globally row-aligned n×p), and the all-curved global residual R0 (computed
    // once from the first slot). These feed the GLOBAL cross-term collapse guard,
    // which reconstructs the full dictionary with the selected collapses applied
    // and measures the TRUE global EV degradation (cross terms and all).
    let mut deltas: Vec<Array2<f64>> = Vec::new();
    let mut r0: Option<Array2<f64>> = None;

    for atom_idx in eligible_d1 {
        let atom = &atoms[atom_idx];
        let coords = coords_for(atom_idx);
        let assign = assign_for(atom_idx);
        let decoded = decoded_for(atom_idx);
        let target_resid = target_resid_for(atom_idx);
        // Curved parameter price = the decoder's `M · p` coefficients.
        let curved_num_params = atom.decoder_coefficients.len();
        let basis_size = atom.decoder_coefficients.nrows();
        let envelope = curved_envelope_metrics(
            assign.view(),
            decoded.view(),
            target_resid.view(),
            basis_size,
        );
        let fitted_turning = atom.basis_evaluator.as_ref().and_then(|evaluator| {
            d1_atom_fitted_turning(
                evaluator.as_ref(),
                atom.decoder_coefficients.view(),
                coords.view(),
            )
            .ok()
            .flatten()
        });
        // Evaluate the curved design `Φ(t)` on this atom's assigned rows so the
        // curved arm's Laplace complexity is the real weighted-design Gram
        // log-determinant rather than a parameter-count proxy (#1223). A `d = 1`
        // atom's coordinate column is presented as an `n × 1` design input. If
        // the evaluator is absent or refuses, `curved_phi` stays `None` and
        // `build_atom_candidates` falls back to the proxy.
        let coords_col = coords
            .view()
            .into_shape_with_order((coords.len(), 1))
            .ok()
            .map(|v| v.to_owned());
        let curved_phi = match (atom.basis_evaluator.as_ref(), coords_col.as_ref()) {
            (Some(evaluator), Some(col)) => {
                evaluator.evaluate(col.view()).ok().map(|(phi, _jet)| phi)
            }
            _ => None,
        };
        // A flat (Euclidean) chart cannot honestly present a curved candidate;
        // the selector drops it. Present both for curveable charts.
        let manifold = manifold_for(atom_idx);
        match build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            target_resid.view(),
            curved_num_params,
            curved_phi.as_ref().map(|phi| phi.view()),
            fitted_turning,
            n_obs,
            dispersion_r,
        ) {
            Ok((linear, curved, (t_bar, b0, b1))) => {
                // #1026 PER-ATOM EV-PRESERVATION gate. Collapsing this slot raises
                // the full reconstruction SSR by `linear_rss − curved_rss`; if that
                // is more than `SAE_HYBRID_COLLAPSE_MAX_EV_LOSS` of the fixed total
                // target variance the collapse would DROP this ONE atom's EV
                // materially, so veto it by presenting only the curved candidate (the
                // selector must keep curved). A lossless / improving collapse (`≤ 0`)
                // and a negligible one stay free to collapse — EV-neutral cases (the
                // top-k / birth-topology lines) are untouched. Only curveable charts
                // are gated; a euclidean chart never had a curved option. NOTE: this
                // gate is PER-ATOM only — the accumulation of many small collapses
                // and the dictionary-level cross terms are handled by the aggregate
                // guard after selection.
                let loss = collapse_ssr_increase(
                    coords.view(),
                    assign.view(),
                    decoded.view(),
                    target_resid.view(),
                    t_bar,
                    &b0,
                    &b1,
                );
                let collapse_loses_ev = total_centered_variance.is_finite()
                    && total_centered_variance > 0.0
                    && loss > SAE_HYBRID_COLLAPSE_MAX_EV_LOSS * total_centered_variance;
                let euclidean = manifold.is_euclidean();
                let slot = if euclidean {
                    vec![linear]
                } else if collapse_loses_ev {
                    vec![curved]
                } else {
                    vec![linear, curved]
                };
                // Build the straight image, then its globally-aligned δ_k for the
                // GLOBAL cross-term guard, before moving it into the report.
                let image = AtomLinearImage {
                    atom_idx,
                    t_bar,
                    b0,
                    b1,
                    // Ordinary straight image: decoded at the atom's own
                    // coordinate, so it carries no residual-projection direction.
                    v: None,
                };
                let delta = slot_delta(&coords, &assign, &decoded, &target_resid, &image)?;
                if r0.is_none() {
                    r0 = Some(slot_r0(&assign, &decoded, &target_resid));
                }
                slots.push(slot);
                // A euclidean slot never had a curved alternative, so its collapse
                // carries no recoverable EV loss for the global guard; record `None`.
                collapse_loss.push(if euclidean { None } else { Some(loss) });
                names.push(atom.name.clone());
                manifolds.push(manifold);
                turnings.push(fitted_turning);
                delta_evs.push(delta_ev_for(atom_idx));
                envelope_metrics.push(envelope);
                deltas.push(delta);
                linear_images.push(image);
            }
            // #1026 collapse rescue: `build_atom_candidates` refused because the
            // atom's own coordinate collapsed (`s_tt ≈ 0`) — the rank-1 co-collapse
            // fixed point. Recover a FRESH linear image from the residual's top
            // direction (fresh per-row codes) and force the LINEAR verdict (a
            // single-option slot the selector must take) so the slot reconstructs
            // its residual's best linear axis at linear quality instead of the
            // collapsed-curve constant. `None` only when the residual itself is
            // degenerate — then there is genuinely nothing to recover and we skip.
            //
            // #2362 — ONLY the typed coordinate-collapse refusal reaches this
            // rescue. Any other refusal (`Unadjudicable`) means the slot could not
            // be adjudicated at all — there is no degenerate curve to rescue, and
            // minting a target-dependent rescue image there is exactly the
            // misclassification #2362 removes. Such a slot is left curved (skip).
            Err(AtomCandidateRefusal::CoordinateCollapse) => match build_collapse_rescue_linear_image(
                atom_idx,
                assign.view(),
                target_resid.view(),
            ) {
                Some((linear, image)) => {
                    let delta = slot_delta(&coords, &assign, &decoded, &target_resid, &image)?;
                    if r0.is_none() {
                        r0 = Some(slot_r0(&assign, &decoded, &target_resid));
                    }
                    slots.push(vec![linear]);
                    // Forced-linear rescue: the curve was degenerate, so there is no
                    // curved alternative to roll back to and no recoverable EV loss.
                    collapse_loss.push(None);
                    names.push(atom.name.clone());
                    manifolds.push(manifold);
                    turnings.push(fitted_turning);
                    delta_evs.push(delta_ev_for(atom_idx));
                    envelope_metrics.push(envelope);
                    deltas.push(delta);
                    linear_images.push(image);
                }
                None => continue,
            },
            // #2362 — non-collapse refusal: the slot cannot be adjudicated, so it
            // keeps its curved decoder (no rescue image, no verdict entry).
            Err(AtomCandidateRefusal::Unadjudicable) => continue,
        }
    }

    if slots.is_empty() {
        return Ok(None);
    }

    let mut selection = select_hybrid_split(&slots)?;

    // #1026 item-2 — GLOBAL CROSS-TERM EV-preservation guard over the SELECTED
    // collapses.
    //
    // The per-atom gate above bounds each atom's individual EV loss, but the TRUE
    // dictionary-level RSS increase from collapsing a SET of atoms is
    //   ΔRSS = Σ_k Δ_k + 2 Σ_{j<k} ⟨δ_j, δ_k⟩,
    // so two effects escape any per-atom or summed-loss bound: (1) the accumulation
    // of many individually-tolerable `Δ_k`, and (2) the CROSS TERMS between
    // simultaneously-collapsed atoms. The old aggregate `Σ max(Δ_k, 0)` guard bounded
    // (1) but was blind to (2): correlated collapse errors whose per-atom losses each
    // sit under tolerance can still push the true global loss over it.
    //
    // Every slot now carries its exact dictionary-level reconstruction-change vector
    // δ_k (globally row-aligned — the caller presents all per-atom arrays on the same
    // n rows), so we reconstruct the full dictionary WITH the selected collapse set
    // applied and measure the real global increase `ΔRSS` DIRECTLY (cross terms
    // captured), reverting the least-justified collapses until the degradation is
    // within the same EV tolerance and re-adjudicating so `selection` stays consistent.
    if total_centered_variance.is_finite() && total_centered_variance > 0.0 {
        if let Some(r0) = r0.as_ref() {
            let global_tol = SAE_HYBRID_COLLAPSE_MAX_EV_LOSS * total_centered_variance;
            // Partition the SELECTED-collapsed slots: rollback-eligible ones (a curved
            // alternative still present and a finite per-atom loss) vs forced ones
            // (euclidean / collapse-rescue — no curved fallback, stay collapsed but
            // still enter the global reconstruction so their cross terms are counted).
            let mut eligible: Vec<(usize, f64, &Array2<f64>)> = Vec::new();
            let mut forced: Vec<&Array2<f64>> = Vec::new();
            for (i, choice) in selection.atoms.iter().enumerate() {
                if !choice.param.is_linear() {
                    continue;
                }
                let has_curved_alt = slots[i].iter().any(|c| !c.param.is_linear());
                let loss_finite = collapse_loss[i].map(|l| l.is_finite()).unwrap_or(false);
                if has_curved_alt && loss_finite {
                    eligible.push((i, choice.curved_evidence_margin, &deltas[i]));
                } else {
                    forced.push(&deltas[i]);
                }
            }
            let reverted = global_collapse_rollback(r0.view(), &eligible, &forced, global_tol);
            if !reverted.is_empty() {
                for slot in reverted {
                    if let Some(curved) = slots[slot].iter().find(|c| !c.param.is_linear()).copied()
                    {
                        slots[slot] = vec![curved];
                    }
                }
                selection = select_hybrid_split(&slots)?;
            }
        }
    }

    let mut verdicts: Vec<AtomHybridVerdict> = Vec::with_capacity(names.len());
    for slot in 0..names.len() {
        let choice = selection.atoms[slot];
        let kept_curved = !choice.param.is_linear();
        let envelope = envelope_metrics[slot];
        verdicts.push(AtomHybridVerdict {
            atom_name: names[slot].clone(),
            choice,
            kept_curved,
            fitted_turning: turnings[slot],
            train_loao_delta_ev: delta_evs[slot],
            curved_ev: envelope.curved_ev,
            topm_linear_ev: envelope.topm_linear_ev,
            curved_vs_envelope_ratio: envelope.curved_vs_envelope_ratio,
            chart_efficiency_eta: envelope.curved_vs_envelope_ratio,
            // Carry the straight sub-model only when the verdict collapses this
            // slot to linear — the curved slots keep their fitted image.
            linear_image: if kept_curved {
                None
            } else {
                Some(linear_images[slot].clone())
            },
        });
    }

    Ok(Some(SaeHybridSplitReport {
        verdicts,
        selection,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn envelope_ratio_is_one_at_topm_linear_ceiling() {
        let n = 4;
        let assign = Array1::<f64>::ones(n);
        let mut target = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let x = if row < 2 { -1.0 } else { 1.0 };
            target[[row, 0]] = x;
            target[[row, 1]] = 0.5 * x;
        }
        let metrics = curved_envelope_metrics(assign.view(), target.view(), target.view(), 1);
        let curved = metrics.curved_ev.expect("curved EV");
        let topm = metrics.topm_linear_ev.expect("top-M linear EV");
        let ratio = metrics
            .curved_vs_envelope_ratio
            .expect("curved/envelope ratio");
        assert!((curved - 1.0).abs() < 1.0e-12, "curved EV {curved}");
        assert!((topm - 1.0).abs() < 1.0e-12, "top-M EV {topm}");
        assert!((ratio - 1.0).abs() < 1.0e-12, "ratio {ratio}");
    }

    #[test]
    fn envelope_ratio_reports_headroom_below_topm_linear() {
        let n = 4;
        let assign = Array1::<f64>::ones(n);
        let mut target = Array2::<f64>::zeros((n, 2));
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let x = if row < 2 { -1.0 } else { 1.0 };
            let y = if row % 2 == 0 { -1.0 } else { 1.0 };
            target[[row, 0]] = x;
            target[[row, 1]] = y;
            decoded[[row, 0]] = 0.5 * x;
        }
        let metrics = curved_envelope_metrics(assign.view(), decoded.view(), target.view(), 1);
        let curved = metrics.curved_ev.expect("curved EV");
        let topm = metrics.topm_linear_ev.expect("top-M linear EV");
        let ratio = metrics
            .curved_vs_envelope_ratio
            .expect("curved/envelope ratio");
        assert!((curved - 0.375).abs() < 1.0e-12, "curved EV {curved}");
        assert!((topm - 0.5).abs() < 1.0e-12, "top-M EV {topm}");
        assert!((ratio - 0.75).abs() < 1.0e-12, "ratio {ratio}");
        assert!(ratio < 1.0, "headroom ratio must be below the envelope");
    }

    #[test]
    fn curved_ev_is_bounded_by_topm_linear_envelope() {
        let n = 8;
        let p = 5;
        let basis_size = 2;
        let assign = Array1::<f64>::ones(n);
        let mut target = Array2::<f64>::zeros((n, p));
        let mut decoded = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let x = if row < 4 { -1.0 } else { 1.0 };
            let y = if row % 4 < 2 { -1.0 } else { 1.0 };
            let z = if row % 2 == 0 { -1.0 } else { 1.0 };
            let target_row = [
                x + 0.25 * z,
                0.5 * x - 0.75 * y,
                y + 0.5 * z,
                0.25 * x + 0.5 * y - z,
                -0.5 * y + 0.25 * z,
            ];
            let decoded_row = [0.7 * x, 0.35 * x - 0.4 * y, 0.6 * y, 0.2 * x, -0.3 * y];
            for col in 0..p {
                target[[row, col]] = target_row[col];
                decoded[[row, col]] = decoded_row[col];
            }
        }
        let metrics =
            curved_envelope_metrics(assign.view(), decoded.view(), target.view(), basis_size);
        let curved = metrics.curved_ev.expect("curved EV");
        let topm = metrics.topm_linear_ev.expect("top-M linear EV");
        assert!(
            curved <= topm + 1.0e-12,
            "one curved atom EV must not exceed the top-M linear envelope: curved={curved}, topm={topm}"
        );
    }

    /// A straight RESPONSE residual (the atom's data is a line) is explained
    /// equally well by both candidates, so the cheaper linear special case wins.
    /// With `a_k = 1` the curved decoded image is straight too (Θ = 0), so both
    /// the dominance floor and the evidence argmin select linear. This is the
    /// common-data nested comparison (#1202): linear is the curved family's
    /// `Θ = 0` member, so it cannot lose when a line already explains the data.
    #[test]
    fn straight_residual_selects_linear() {
        let n = 40;
        let coords = Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::ones(n);
        // The data the atom must explain is a straight line in ℝ²; the curved
        // decoded image equals that same line (a Θ = 0 curved fit).
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = coords[i];
            data[[i, 1]] = 0.6 * coords[i];
            decoded[[i, 0]] = coords[i];
            decoded[[i, 1]] = 0.6 * coords[i];
        }
        let mut phi = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            phi[[i, 0]] = 1.0;
            phi[[i, 1]] = coords[i];
        }
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            data.view(),
            // a generous curved parameter price (M·p)
            10,
            Some(phi.view()),
            Some(0.0),
            coords.len(),
            0.0025,
        )
        .expect("straight residual yields a candidate pair");
        let choice =
            gam_solve::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert!(
            choice.param.is_linear(),
            "a straight response residual must keep the linear special case"
        );
    }

    /// A turning RESPONSE residual (the atom's data traces a full circle) is fit
    /// well by the curved decoded image (curved_rss ≈ 0) but poorly by any
    /// straight line (large linear_rss), so the curved candidate wins the common
    /// evidence comparison once its data-fit gain exceeds its extra parameter
    /// price (#1202).
    #[test]
    fn turning_residual_selects_curved_on_evidence() {
        let n = 60;
        let coords = Array1::from_iter((0..n).map(|i| (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::ones(n);
        // The data is a full circle; the curved decoded image is that same
        // circle (the curved atom reconstructs its assigned residual), so the
        // curved candidate has ≈ zero data-fit residual while a straight line
        // cannot follow the loop.
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = 2.0 * PI * coords[i];
            data[[i, 0]] = theta.cos();
            data[[i, 1]] = theta.sin();
            decoded[[i, 0]] = theta.cos();
            decoded[[i, 1]] = theta.sin();
        }
        let mut phi = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let theta = 2.0 * PI * coords[i];
            phi[[i, 0]] = 1.0;
            phi[[i, 1]] = theta.cos();
            phi[[i, 2]] = theta.sin();
        }
        // The curved atom has 5 parameters (just above the 4 = 2·p linear budget);
        // the full-circle linear residual exceeds the extra-parameter overhead, so
        // curved wins on evidence.
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            data.view(),
            5,
            Some(phi.view()),
            Some(2.0 * PI),
            coords.len(),
            0.0025,
        )
        .expect("turning residual yields a candidate pair");
        assert!(
            linear.negative_log_evidence > curved.negative_log_evidence,
            "the line must misfit the circular residual worse than the curve does \
             (linear NLE {} should exceed curved NLE {})",
            linear.negative_log_evidence,
            curved.negative_log_evidence
        );
        let choice =
            gam_solve::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert_eq!(
            choice.param,
            gam_solve::evidence::HybridAtomParam::Curved { latent_dim: 1 },
            "a full-circle response residual must keep the curved parameterization"
        );
        assert!(
            choice.curved_evidence_margin > 0.0,
            "curved must win a positive evidence margin over the linear secant"
        );
    }

    /// #16 DEMOTE: the two arms are priced in the joint fit's canonical
    /// ½·d_eff·log(n_obs) currency (d_eff read off the
    /// curved REFIT decoder + Gram via `realised_rank_charge_dof`), NOT the ½log|H|
    /// Laplace det. A full-circle residual still KEEPS curved (its realised rank-2
    /// d_eff earns the fit); a straight-line residual stays LINEAR (the periodic
    /// curve bends away from the line, so its extra realised DOF is not earned).
    /// A real `Φ = [1, cos, sin]` is passed so d_eff reads the exact refit decoder.
    #[test]
    fn rank_charge_demote_prices_realised_rank() {
        let n = 60;
        let coords = Array1::from_iter((0..n).map(|i| (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::ones(n);
        let mut phi = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let th = 2.0 * PI * coords[i];
            phi[[i, 0]] = 1.0;
            phi[[i, 1]] = th.cos();
            phi[[i, 2]] = th.sin();
        }
        // (a) full-circle residual → curved fits, linear misfits → CURVED kept.
        let mut circle = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let th = 2.0 * PI * coords[i];
            circle[[i, 0]] = th.cos();
            circle[[i, 1]] = th.sin();
        }
        let (lin, crv, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            circle.view(),
            circle.view(),
            6,
            Some(phi.view()),
            Some(2.0 * PI),
            n,
            0.0025,
        )
        .expect("circle candidate pair");
        let choice = gam_solve::evidence::select_hybrid_atom(&[lin, crv]).expect("non-empty slot");
        assert_eq!(
            choice.param,
            gam_solve::evidence::HybridAtomParam::Curved { latent_dim: 1 },
            "rank charge must KEEP curved on a full-circle residual"
        );
        assert!(
            choice.curved_evidence_margin > 0.0,
            "curved must clear a positive rank-charge margin on a circle"
        );

        // (b) straight-line residual → linear fits, periodic curve misfits → LINEAR.
        let mut line = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            line[[i, 0]] = coords[i];
            line[[i, 1]] = 0.5 * coords[i];
        }
        let (lin2, crv2, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            line.view(),
            line.view(),
            6,
            Some(phi.view()),
            Some(0.0),
            n,
            0.0025,
        )
        .expect("line candidate pair");
        let choice2 =
            gam_solve::evidence::select_hybrid_atom(&[lin2, crv2]).expect("non-empty slot");
        assert_eq!(
            choice2.param,
            gam_solve::evidence::HybridAtomParam::Linear,
            "rank charge must stay LINEAR on a straight-line residual"
        );
    }

    /// #1051 NESTED MIN — the curved arm re-fit on the residual match-or-beats the
    /// best straight line: `curved_refit_rss ≤ best_line_rss` up to solver tolerance
    /// on a basis whose span contains the straight lane (here `Φ = [1, t, t²]`). A
    /// genuinely curved (quadratic) signal is STRICTLY preferred by the curved arm,
    /// while an exactly-straight signal ties near zero (so the cheaper linear lane
    /// wins downstream). This is the property the realized-curve heuristic could not
    /// establish: comparing a possibly-collapsed realized curve against min-over-lines
    /// did NOT guarantee `curved ≤ linear`; re-fitting the curved decoder does.
    #[test]
    fn refit_curved_rss_matches_or_beats_best_line_nested() {
        let n = 40usize;
        let p = 2usize;
        let coords = Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::ones(n);
        // Φ = [1, t, t²]: its column span contains the straight lane [1, t], so the
        // decoder-only curved refit is a proper superset of the line fit.
        let mut phi = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            phi[[i, 0]] = 1.0;
            phi[[i, 1]] = coords[i];
            phi[[i, 2]] = coords[i] * coords[i];
        }
        // Best mass-weighted line RSS on `y` (assign = 1): fit design [1, (t − t̄)].
        let t_bar = coords.iter().sum::<f64>() / n as f64;
        let best_line_rss = |y: &Array2<f64>| -> f64 {
            let mut design = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                design[[i, 0]] = 1.0;
                design[[i, 1]] = coords[i] - t_bar;
            }
            let b = solve_design_least_squares(design.view(), y.view()).unwrap();
            let pred = design.dot(&b);
            let mut rss = 0.0_f64;
            for i in 0..n {
                for j in 0..p {
                    let r = y[[i, j]] - pred[[i, j]];
                    rss += r * r;
                }
            }
            rss
        };
        // (a) exactly straight, (b) quadratic curve, (c) noisy line.
        let mut y_line = Array2::<f64>::zeros((n, p));
        let mut y_curve = Array2::<f64>::zeros((n, p));
        let mut y_noisy = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = coords[i];
            y_line[[i, 0]] = 0.4 + 0.6 * t;
            y_line[[i, 1]] = -0.2 + 1.1 * t;
            y_curve[[i, 0]] = t * t;
            y_curve[[i, 1]] = 0.5 - t * t;
            y_noisy[[i, 0]] = 0.3 + 0.7 * t + 0.05 * (3.0 * t).sin();
            y_noisy[[i, 1]] = 0.9 * t;
        }
        for y in [&y_line, &y_curve, &y_noisy] {
            let curved = curved_refit_rss(phi.view(), assign.view(), y.view())
                .expect("non-degenerate refit");
            let line = best_line_rss(y);
            assert!(
                curved <= line + 1e-9 * (1.0 + line),
                "nested dominance: refit-curved RSS {curved} must be ≤ best-line RSS {line}"
            );
        }
        // A genuinely curved (quadratic) signal is STRICTLY preferred by the curve.
        let curved_c = curved_refit_rss(phi.view(), assign.view(), y_curve.view()).unwrap();
        let line_c = best_line_rss(&y_curve);
        assert!(
            curved_c < 0.5 * line_c,
            "a quadratic signal must be far better fit by the curve ({curved_c}) than \
             by the best line ({line_c})"
        );
        // A straight signal ties near zero — collapses to the cheaper linear lane.
        let curved_l = curved_refit_rss(phi.view(), assign.view(), y_line.view()).unwrap();
        assert!(
            curved_l < 1e-18 && best_line_rss(&y_line) < 1e-18,
            "an exactly-straight signal ties the two arms near zero (curved {curved_l})"
        );
    }

    /// #1026 item-2 GLOBAL CROSS-TERM guard: two collapses with CORRELATED
    /// (parallel) reconstruction-change errors whose per-atom losses each sit under
    /// tolerance — and whose SUM `Σ Δ_k` is also under tolerance (so the OLD
    /// aggregate `Σ max(Δ_k,0)` guard would ACCEPT) — but whose cross term
    /// `2⟨δ_1,δ_2⟩` pushes the TRUE global loss over tolerance. The global guard must
    /// roll back the least-justified collapse. An ORTHOGONAL control (disjoint
    /// support) with the same per-atom losses is accepted, isolating the cross term.
    #[test]
    fn global_guard_rejects_correlated_collapses_the_aggregate_would_accept() {
        let n = 4usize;
        let p = 1usize;
        // R0 = 0 ⇒ Δ_k = ‖δ_k‖² and ΔRSS(S) = ‖Σ_S δ_k‖² exactly.
        let r0 = Array2::<f64>::zeros((n, p));
        // Parallel δ_1 = δ_2 = 1 on all rows: ‖δ_k‖² = 4 each, Σ Δ_k = 8,
        // ΔRSS_global = ‖δ_1+δ_2‖² = 16.
        let mut d1 = Array2::<f64>::zeros((n, p));
        let mut d2 = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            d1[[i, 0]] = 1.0;
            d2[[i, 0]] = 1.0;
        }
        // tol = 10: each per-atom loss 4 ≤ 10, Σ Δ_k = 8 ≤ 10 (aggregate accepts),
        // but global 16 > 10 (cross term rejects).
        let global_tol = 10.0_f64;
        let eligible = vec![(0usize, 0.1_f64, &d1), (1usize, 0.2_f64, &d2)];
        let forced: Vec<&Array2<f64>> = Vec::new();
        let reverted = global_collapse_rollback(r0.view(), &eligible, &forced, global_tol);
        assert_eq!(
            reverted,
            vec![1usize],
            "the global cross-term guard must roll back the least-justified collapse \
             (largest margin = slot 1) that the summed-loss aggregate (8 ≤ 10) accepts"
        );

        // ORTHOGONAL control: same per-atom losses (δ on disjoint rows) ⇒ cross term
        // 0 ⇒ ΔRSS_global = 8 ≤ 10 ⇒ no rollback. Isolates the cross term as the cause.
        let mut o1 = Array2::<f64>::zeros((n, p));
        let mut o2 = Array2::<f64>::zeros((n, p));
        o1[[0, 0]] = 2.0; // ‖o1‖² = 4
        o2[[2, 0]] = 2.0; // ‖o2‖² = 4, disjoint support
        let eligible_o = vec![(0usize, 0.1_f64, &o1), (1usize, 0.2_f64, &o2)];
        let reverted_o = global_collapse_rollback(r0.view(), &eligible_o, &forced, global_tol);
        assert!(
            reverted_o.is_empty(),
            "uncorrelated collapses (cross term 0, global loss 8 ≤ 10) must be accepted"
        );
    }

    /// A degenerate (single-point-mass) coordinate has no slope direction and is
    /// refused rather than adjudicated on a fabricated deviance.
    #[test]
    fn degenerate_coordinate_is_refused() {
        let n = 5;
        let coords = Array1::<f64>::from_elem(n, 0.5); // no spread
        let assign = Array1::<f64>::ones(n);
        let decoded = Array2::<f64>::zeros((n, 2));
        let data = Array2::<f64>::zeros((n, 2));
        assert!(
            build_atom_candidates(
                coords.view(),
                assign.view(),
                decoded.view(),
                data.view(),
                6,
                None,
                Some(0.0),
                coords.len(),
                0.0,
            )
                        .map(|_| ())
            .unwrap_err()
                == AtomCandidateRefusal::CoordinateCollapse,
            "a degenerate coordinate span must be refused as a coordinate collapse (#2362)"
        );
    }
}
