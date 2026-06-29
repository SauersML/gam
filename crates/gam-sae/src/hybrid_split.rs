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
//!   * the CURVED candidate predicts `a_k · γ_k(t)` — the atom's actual
//!     already-fitted contribution; its data-fit deviance is the weighted RSS of
//!     that contribution against `y_resp`, no longer zero by construction.
//!   * the LINEAR candidate predicts `a_k · (b₀ + (t − t̄)·b₁)`, the best
//!     weighted least-squares straight line fit to `y_resp` (design column
//!     scaled by the same assignment mass `a_k`), so its data-fit deviance is the
//!     weighted RSS of the best line against the SAME residual.
//!
//! Because the curved family's `Θ = 0` member reproduces exactly the linear
//! prediction `a_k·(b₀ + (t − t̄)·b₁)` on this data, linear IS the nested `Θ = 0`
//! sub-model of the curved family on common data — so the per-slot evidence
//! argmin is a genuine "match-or-beat" comparison: the curved candidate is
//! preferred only when its extra curvature lowers the data-fit deviance by more
//! than its extra Laplace parameter price, and the linear special case wins
//! whenever a straight line already explains the residual.
//!
//! This replaces the earlier post-hoc curve-simplification diagnostic, in which
//! both candidates targeted the atom's already-fitted decoded image `γ_k(t)`
//! (giving the curved arm a free zero residual against itself) rather than the
//! response data. That version could not nest linear in curved on common data
//! and so carried no real dominance guarantee (#1202); it is removed. The
//! comparison here re-fits nothing in the (broken under #1051) euclidean /
//! multi-atom outer continuation — it scores the already-realized curved
//! contribution and the closed-form linear lane against the realized residual,
//! both on the data, with no joint Hessian or continuation spine.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use gam_linalg::faer_ndarray::FaerEigh;
use gam_solve::evidence::{
    HybridAtomCandidate, HybridAtomChoice, HybridSplitSelection, select_hybrid_split,
};
use gam_terms::latent::LatentManifold;
use crate::chart_canonicalization::d1_atom_fitted_turning;
use crate::manifold::SaeManifoldAtom;

/// The rank-aware Laplace negative-log-evidence of a reduced per-atom Gaussian
/// reconstruction sub-model: `residual_objective + ½ log|H|` with no smoothing
/// penalty logdet and a full-rank design (no null space), which is the form
/// [`gam_solve::evidence::laplace_evidence`] reduces to on this comparison.
/// Kept inline (rather than routed through `EvidenceLogDetSource`) because both
/// candidates' Hessian logdets are already the closed-form scalar moments of
/// their shared design — no factor cache or HVP callback to assemble.
fn reduced_laplace_nle(residual_objective: f64, log_det_h: f64) -> f64 {
    residual_objective + 0.5 * log_det_h
}

/// Rank-aware `log|ΦᵀWΦ|_+` of the curved atom's weighted design Gram over its
/// `M` decoder basis columns, with per-row weight `wᵢ = a_k²` (the same
/// assignment-mass design weight the linear arm uses), summed over the
/// eigenvalues above a relative spectral floor (#1223). This is the genuine
/// weighted-design determinant the linear arm already reports — `log|XᵀWX|` —
/// assembled for the curved basis so the two arms' Laplace complexity prices are
/// computed on the SAME footing instead of pricing the curved arm with a
/// parameter-count proxy `M·log(Σw)`.
///
/// Mirrors the linear arm exactly in what it does NOT include: no smoothing-
/// penalty `λS` normalizer (the linear arm's Gram is the bare data Gram
/// `diag(w_sum, s_tt)` too), so the comparison stays symmetric. The Gram is the
/// design's outer Gram over its basis columns; it is identical across the `p`
/// output channels (every channel shares the design `Φ`), so the per-channel
/// `log|G|_+` is multiplied by `p` — matching the linear arm's `p·(…)` form.
///
/// `phi` is the curved design `Φ(t)` evaluated on the atom's assigned rows
/// (`n × M`); `assign` is the per-row assignment mass `a_k` (NOT squared).
/// Returns `None` when `Φ` is missing rows, the Gram is non-finite, or it has no
/// positive eigenvalues (a fully rank-deficient design carries no determinant);
/// the caller then falls back to the parameter-count proxy rather than fabricate
/// a determinant.
fn curved_design_gram_logdet(
    phi: ArrayView2<'_, f64>,
    assign: ArrayView1<'_, f64>,
    p: usize,
) -> Option<f64> {
    let n = phi.nrows();
    let m = phi.ncols();
    if m == 0 || assign.len() != n || n == 0 {
        return None;
    }
    // G = Φᵀ diag(a²) Φ  (M×M, symmetric PSD).
    let mut gram = Array2::<f64>::zeros((m, m));
    for i in 0..n {
        let w = assign[i] * assign[i];
        if !(w.is_finite() && w >= 0.0) {
            return None;
        }
        if w == 0.0 {
            continue;
        }
        let row = phi.row(i);
        for a in 0..m {
            let wa = w * row[a];
            for b in a..m {
                gram[[a, b]] += wa * row[b];
            }
        }
    }
    // Symmetrize the lower triangle (we only filled the upper).
    for a in 0..m {
        for b in 0..a {
            gram[[a, b]] = gram[[b, a]];
        }
    }
    if gram.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let (vals, _vecs) = gram.eigh(faer::Side::Lower).ok()?;
    // Rank-aware log-determinant: sum log of eigenvalues above a relative floor
    // tied to the largest eigenvalue, dropping the numerically-null directions
    // (the curved design's null space, analogous to the linear arm's full-rank
    // 2-D Gram). A design with no positive eigenvalue carries no determinant.
    let lambda_max = vals.iter().cloned().fold(0.0_f64, f64::max);
    if !(lambda_max > 0.0 && lambda_max.is_finite()) {
        return None;
    }
    let floor = lambda_max * 1e-12;
    let mut log_det = 0.0_f64;
    let mut rank = 0usize;
    for &lambda in vals.iter() {
        if lambda > floor {
            log_det += lambda.ln();
            rank += 1;
        }
    }
    if rank == 0 || !log_det.is_finite() {
        return None;
    }
    // The design Gram is shared across the p output channels.
    Some((p as f64) * log_det)
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
    /// #1026 collapse-rescue per-row coordinates. `None` for the ordinary path:
    /// the line is evaluated at the atom's OWN realized coordinate `t`. `Some(u)`
    /// only when the atom's circle codes had collapsed to a single point
    /// (`s_tt ≈ 0`) so its own coordinate carries no spread — then the line is fit
    /// against, and reconstruct evaluates it at, these FRESH per-row codes `uᵢ`
    /// (the projection of the leave-this-atom-out residual onto its top
    /// mass-weighted output direction). This is what lets a circle atom that the
    /// joint fit drove into the degenerate "chord-through-the-arc" fixed point
    /// still reconstruct its residual's best linear direction — recovering the
    /// linear-tail reach the hybrid-split was designed to deliver — instead of a
    /// constant (its collapsed curve), which is the real-OLMo rank-1 co-collapse
    /// (held-out EV ≈ 0.13 vs the linear ceiling ≈ 0.74). Length `n` (one per
    /// reconstructed row); unassigned rows are gated to zero by `a_k` anyway.
    pub row_codes: Option<Array1<f64>>,
}

impl AtomLinearImage {
    /// Evaluate the straight sub-model `b₀ + (t − t̄)·b₁` into `out` (length `p`).
    pub fn fill_row(&self, t: f64, out: &mut [f64]) {
        let dt = t - self.t_bar;
        for (j, slot) in out.iter_mut().enumerate() {
            *slot = self.b0[j] + dt * self.b1[j];
        }
    }

    /// The coordinate at which row `row` should evaluate this image: the
    /// collapse-rescue fresh code `uᵢ` when present (#1026), else the atom's own
    /// realized coordinate `own_t` passed by the caller.
    pub fn coordinate_for_row(&self, row: usize, own_t: f64) -> f64 {
        match &self.row_codes {
            Some(u) if row < u.len() => u[row],
            _ => own_t,
        }
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
const SAE_HYBRID_COLLAPSE_MAX_EV_LOSS: f64 = 1.0e-3;

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

/// Build the curved + linear candidates for ONE fitted `d = 1` atom and return
/// them as `(linear, curved, (t̄, b₀, b₁))`, or `None` if the atom cannot present
/// an honest pair (too few rows, degenerate coordinate span, or non-finite
/// numbers). Both candidates are scored against the SAME data — the atom's
/// leave-this-atom-out response residual `y_resp` — so the comparison is a
/// genuine common-evidence one with linear nested as the curved family's `Θ = 0`
/// sub-model (#1202).
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
/// The curved candidate's data-fit deviance is `½ Σ ‖y_resp − a_k·γ_k‖²` (the
/// plain reconstruction SSE, matching the joint loss — the mass already lives in
/// the prediction `a_k·γ_k`); the linear candidate fits the best mass-weighted
/// straight line to `y_resp` and pays `½ Σ ‖y_resp − a_k·(b₀ + (t − t̄)·b₁)‖²`.
/// Because the curved family's `Θ = 0` member reproduces the linear prediction
/// exactly, linear is the nested sub-model and the argmin is the honest
/// match-or-beat criterion.
fn build_atom_candidates(
    coords: ArrayView1<'_, f64>,
    assign: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    target_resid: ArrayView2<'_, f64>,
    curved_num_params: usize,
    curved_phi: Option<ArrayView2<'_, f64>>,
    fitted_turning: Option<f64>,
) -> Option<(
    HybridAtomCandidate,
    HybridAtomCandidate,
    (f64, Array1<f64>, Array1<f64>),
)> {
    let n = coords.len();
    let p = decoded.ncols();
    if n < MIN_ROWS_FOR_LINEAR_FIT
        || decoded.nrows() != n
        || assign.len() != n
        || target_resid.nrows() != n
        || target_resid.ncols() != p
        || p == 0
    {
        return None;
    }

    // The LINEAR candidate fits `a_k·(b₀ + (t − t̄)·b₁)` to the residual `y_resp`,
    // so the natural design column is `a_k·[1, (t − t̄)]` and the per-row Gram
    // weight is `wᵢ = a_k²`. We accumulate the mass-weighted coordinate mean `t̄`
    // and spread `s_tt` under that weight; a row that barely belongs to the atom
    // (`a_k ≈ 0`) contributes ≈ nothing, exactly as in the joint loss.
    let mut w_sum = 0.0_f64;
    let mut t_bar = 0.0_f64;
    for i in 0..n {
        let a = assign[i];
        if !(a.is_finite() && a >= 0.0) {
            return None;
        }
        let w = a * a;
        w_sum += w;
        t_bar += w * coords[i];
    }
    if !(w_sum > 0.0) {
        return None;
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
    if !(s_tt > 1e-12 * (1.0 + t_bar * t_bar)) {
        return None;
    }

    // Per-output-channel mass-weighted least squares for the line fit to the
    // RESIDUAL `y_resp`. Minimizing `Σᵢ ‖y_resp[i] − a_k·(b₀ + (t − t̄)·b₁)‖²` in
    // the centered basis has the diagonal normal equations
    //   b₀[j] = (Σ a_k·y_resp[i,j]) / w_sum,   (recall the design intercept is a_k)
    //   b₁[j] = (Σ a_k·(t − t̄)·y_resp[i,j]) / s_tt.
    let mut b0 = Array1::<f64>::zeros(p);
    let mut b1 = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut s_1y = 0.0_f64;
        let mut s_ty = 0.0_f64;
        for i in 0..n {
            let a = assign[i];
            let dt = coords[i] - t_bar;
            let y = target_resid[[i, j]];
            s_1y += a * y;
            s_ty += a * dt * y;
        }
        b0[j] = s_1y / w_sum;
        b1[j] = s_ty / s_tt;
    }

    // Data-fit residual sums of squares of BOTH candidates against `y_resp`, the
    // common data. The curved candidate predicts the atom's actual mass-scaled
    // contribution `a_k·γ_k`; the linear candidate predicts the best line
    // `a_k·(b₀ + (t − t̄)·b₁)`. These are no longer trivially zero for the curved
    // arm — both are real misfits to the response residual, so the argmin is a
    // genuine common-evidence comparison (#1202).
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

    // Gaussian-reconstruction deviance: the residual objective `½ RSS` the
    // Laplace normalizer is added to. The curved arm pays `½·curved_rss` (how
    // well its fitted curve explains the residual) plus its larger `M·p`
    // parameter price; the linear arm pays `½·linear_rss` plus a `2·p` price.
    // Because the curved family's `Θ = 0` member equals the linear prediction,
    // `curved_rss ≤ linear_rss` whenever the fitted curve is at least as good a
    // residual fit as its own straight projection — the match-or-beat floor — and
    // the argmin trades that data-fit gain against the curvature parameter price.
    let curved_residual_objective = 0.5 * curved_rss;
    let linear_residual_objective = 0.5 * linear_rss;

    // Linear candidate parameter price: intercept + slope per output channel.
    let linear_num_params = 2 * p;

    // Laplace logdet of the (weighted) design Gram for the LINEAR candidate.
    //
    // For the centered weighted line fit `a_k·(b₀ + (t − t̄)·b₁)`, the per-output-
    // channel design column is `a_k·[1, (t − t̄)]`, whose Gram is DIAGONAL in the
    // centered basis: `diag(Σ a_k², Σ a_k²(t − t̄)²) = diag(w_sum, s_tt)`. Its log
    // determinant is `log(w_sum) + log(s_tt)` PER output channel, i.e.
    //
    //     log|H_linear| = p · ( log(w_sum) + log(s_tt) ).
    //
    // The `log(s_tt)` term is the slope direction's information: a line through a
    // wide, heavily-massed coordinate spread is better-determined than one through
    // a tiny spread, and the Laplace evidence must reflect that (#1203).
    //
    // The curved arm's Laplace determinant is now the genuine weighted-design
    // Gram log-determinant `p · log|ΦᵀWΦ|_+` (#1223): the SAME quantity the
    // linear arm reports (`p·(log w_sum + log s_tt) = p·log|XᵀWX|`), assembled
    // from the curved basis `Φ` on the atom's assigned rows under the same
    // assignment-mass design weight `wᵢ = a_k²`. Both arms omit the smoothing
    // `λS` normalizer, so the complexity price is computed on a symmetric
    // footing — no parameter-count proxy. Only when `Φ` is unavailable (the
    // caller could not evaluate the basis) or its Gram is fully rank-deficient do
    // we fall back to the historical `curved_num_params · log(w_sum)` proxy, so
    // the comparison degrades gracefully rather than fabricating a determinant.
    if !(w_sum > 0.0 && w_sum.is_finite() && s_tt.is_finite()) {
        return None;
    }
    let linear_log_det_h = (p as f64) * (w_sum.ln() + s_tt.ln());
    let curved_log_det_h = curved_phi
        .and_then(|phi| {
            if phi.nrows() == n {
                curved_design_gram_logdet(phi, assign, p)
            } else {
                None
            }
        })
        .unwrap_or_else(|| (curved_num_params as f64) * w_sum.ln());

    // Reduced Laplace NLE `residual_objective + ½ log|H|`. Both omit an explicit
    // smoothing-penalty logdet (the intrinsic smoothness penalty is
    // reparameterization-invariant and identical in expectation across the two
    // parameterizations of the same image).
    let linear_nle = reduced_laplace_nle(linear_residual_objective, linear_log_det_h);
    let curved_nle = reduced_laplace_nle(curved_residual_objective, curved_log_det_h);
    if !(linear_nle.is_finite() && curved_nle.is_finite()) {
        return None;
    }

    let linear = HybridAtomCandidate::linear(linear_nle, linear_num_params);
    let curved = HybridAtomCandidate::curved(1, curved_nle, curved_num_params, fitted_turning);
    Some((linear, curved, (t_bar, b0, b1)))
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
/// split owes. Returns the forced-LINEAR candidate plus the image carrying `uᵢ`,
/// or `None` when the residual itself carries no usable direction (a genuine zero
/// atom the mass/decoder guards own).
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
        row_codes: Some(u),
    };
    Some((linear, image))
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

    for atom_idx in eligible_d1 {
        let atom = &atoms[atom_idx];
        let coords = coords_for(atom_idx);
        let assign = assign_for(atom_idx);
        let decoded = decoded_for(atom_idx);
        let target_resid = target_resid_for(atom_idx);
        // Curved parameter price = the decoder's `M · p` coefficients.
        let curved_num_params = atom.decoder_coefficients.len();
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
        ) {
            Some((linear, curved, (t_bar, b0, b1))) => {
                // #1026 EV-PRESERVATION gate. Collapsing this slot raises the full
                // reconstruction SSR by `linear_rss − curved_rss`; if that is more
                // than `SAE_HYBRID_COLLAPSE_MAX_EV_LOSS` of the fixed total target
                // variance the collapse would DROP reconstruction EV materially, so
                // veto it by presenting only the curved candidate (the selector must
                // keep curved). A lossless / improving collapse (`≤ 0`) and a
                // negligible one stay free to collapse — EV-neutral cases (the
                // top-k / birth-topology lines) are untouched. Only curveable charts
                // are gated; a euclidean chart never had a curved option.
                let collapse_loses_ev = total_centered_variance.is_finite()
                    && total_centered_variance > 0.0
                    && collapse_ssr_increase(
                        coords.view(),
                        assign.view(),
                        decoded.view(),
                        target_resid.view(),
                        t_bar,
                        &b0,
                        &b1,
                    ) > SAE_HYBRID_COLLAPSE_MAX_EV_LOSS * total_centered_variance;
                let slot = if manifold.is_euclidean() {
                    vec![linear]
                } else if collapse_loses_ev {
                    vec![curved]
                } else {
                    vec![linear, curved]
                };
                slots.push(slot);
                names.push(atom.name.clone());
                manifolds.push(manifold);
                turnings.push(fitted_turning);
                delta_evs.push(delta_ev_for(atom_idx));
                linear_images.push(AtomLinearImage {
                    atom_idx,
                    t_bar,
                    b0,
                    b1,
                    row_codes: None,
                });
            }
            // #1026 collapse rescue: `build_atom_candidates` refused because the
            // atom's own coordinate collapsed (`s_tt ≈ 0`) — the rank-1 co-collapse
            // fixed point. Recover a FRESH linear image from the residual's top
            // direction (fresh per-row codes) and force the LINEAR verdict (a
            // single-option slot the selector must take) so the slot reconstructs
            // its residual's best linear axis at linear quality instead of the
            // collapsed-curve constant. `None` only when the residual itself is
            // degenerate — then there is genuinely nothing to recover and we skip.
            None => match build_collapse_rescue_linear_image(
                atom_idx,
                assign.view(),
                target_resid.view(),
            ) {
                Some((linear, image)) => {
                    slots.push(vec![linear]);
                    names.push(atom.name.clone());
                    manifolds.push(manifold);
                    turnings.push(fitted_turning);
                    delta_evs.push(delta_ev_for(atom_idx));
                    linear_images.push(image);
                }
                None => continue,
            },
        }
    }

    if slots.is_empty() {
        return Ok(None);
    }

    let selection = select_hybrid_split(&slots)?;
    let verdicts: Vec<AtomHybridVerdict> = names
        .into_iter()
        .zip(selection.atoms.iter().copied())
        .zip(linear_images.into_iter())
        .zip(turnings.into_iter())
        .zip(delta_evs.into_iter())
        .map(
            |((((atom_name, choice), linear_image), fitted_turning), train_loao_delta_ev)| {
                let kept_curved = !choice.param.is_linear();
                AtomHybridVerdict {
                    atom_name,
                    choice,
                    kept_curved,
                    fitted_turning,
                    train_loao_delta_ev,
                    // Carry the straight sub-model only when the verdict collapses
                    // this slot to linear — the curved slots keep their fitted image.
                    linear_image: if kept_curved {
                        None
                    } else {
                        Some(linear_image)
                    },
                }
            },
        )
        .collect();

    Ok(Some(SaeHybridSplitReport {
        verdicts,
        selection,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            data.view(),
            // a generous curved parameter price (M·p)
            10,
            None,
            Some(0.0),
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
        // The curved atom has 5 parameters (just above the 4 = 2·p linear budget);
        // the full-circle linear residual exceeds the extra-parameter overhead, so
        // curved wins on evidence.
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            data.view(),
            5,
            None,
            Some(2.0 * PI),
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

    /// The nested-dominance floor on common data (#1202): when the curved decoded
    /// image is a WORSE fit to the response residual than its own best straight
    /// projection, linear must win — the curved family cannot be charged extra
    /// parameters to fit the residual no better than its `Θ = 0` member. Here the
    /// data is a line but the curved image bends away from it, so curved_rss >
    /// linear_rss and the cheaper, better-fitting line is selected.
    #[test]
    fn linear_beats_curved_when_curve_misfits_residual() {
        let n = 50;
        let coords = Array1::from_iter((0..n).map(|i| (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::ones(n);
        // Data is a straight line; the curved decoded image is a parabola that
        // departs from it, so a straight line fits the data strictly better.
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = coords[i];
            data[[i, 0]] = t;
            data[[i, 1]] = 0.5 * t;
            decoded[[i, 0]] = t;
            decoded[[i, 1]] = t * t; // bends away from the linear data
        }
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            data.view(),
            // a real curved Θ above the floor so the dominance floor does not fire
            6,
            None,
            Some(1.0),
        )
        .expect("candidate pair");
        let choice =
            gam_solve::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert!(
            choice.param.is_linear(),
            "a curved image that fits the data worse than its own line must yield \
             to the linear special case on common-data evidence (#1202)"
        );
    }

    /// The LINEAR candidate's Laplace logdet is the genuine weighted-design Gram
    /// determinant `p·(log w_sum + log s_tt)` with `w_sum = Σ a_k²`, `s_tt =
    /// Σ a_k²(t − t̄)²` — it INCLUDES the coordinate-spread term `log(s_tt)`
    /// (#1203). Verify both contributions are present by reading the logdet off a
    /// candidate whose linear residual is exactly zero (response residual = the
    /// fitted line), so `NLE_linear = ½·logdet`. Doubling the coordinate spread
    /// (at fixed assignment mass) scales `s_tt` by 4 → logdet += `p·log(4)`;
    /// doubling all assignment masses scales BOTH `w_sum` and `s_tt` by 4 (they
    /// are quadratic in `a_k`) → logdet += `2p·log(4)`.
    #[test]
    fn linear_logdet_includes_weighted_coordinate_spread() {
        let n = 40;
        let p = 2usize;
        // Read the logdet back off a candidate with zero linear residual: the
        // response residual is exactly `a_k·(line)`, so the WLS line recovers it
        // with RSS == 0 and `NLE_linear = ½·logdet`.
        let logdet = |coords: &Array1<f64>, assign: &Array1<f64>| -> f64 {
            // A straight image; the response residual is the same line scaled by
            // the per-row assignment mass `a_k`, so the prediction `a_k·(b₀+dt·b₁)`
            // matches it exactly and linear_rss == 0.
            let line = |t: f64| -> [f64; 2] { [t, 0.6 * t] };
            let mut decoded = Array2::<f64>::zeros((n, p));
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let l = line(coords[i]);
                decoded[[i, 0]] = l[0];
                decoded[[i, 1]] = l[1];
                data[[i, 0]] = assign[i] * l[0];
                data[[i, 1]] = assign[i] * l[1];
            }
            let (linear, _curved, _) = build_atom_candidates(
                coords.view(),
                assign.view(),
                decoded.view(),
                data.view(),
                10,
                None,
                Some(0.0),
            )
            .expect("straight residual yields a pair");
            2.0 * linear.negative_log_evidence // = logdet (linear_rss == 0)
        };

        let base_coords =
            Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)));
        let ones = Array1::<f64>::ones(n);

        // Doubling the coordinate spread → s_tt ×4, w_sum fixed → logdet += p·log(4).
        let wide_coords = base_coords.mapv(|t| 2.0 * t);
        let d_spread = logdet(&wide_coords, &ones) - logdet(&base_coords, &ones);
        assert!(
            (d_spread - (p as f64) * 4.0_f64.ln()).abs() < 1e-9,
            "linear logdet must move by p·log(4) when coordinate spread doubles \
             (got {d_spread}); the spread term log(s_tt) must be present"
        );

        // Doubling all assignment masses → w_sum ×4 AND s_tt ×4 (quadratic in a_k)
        // → logdet += 2p·log(4).
        let twos = Array1::<f64>::from_elem(n, 2.0);
        let d_weight = logdet(&base_coords, &twos) - logdet(&base_coords, &ones);
        assert!(
            (d_weight - 2.0 * (p as f64) * 4.0_f64.ln()).abs() < 1e-9,
            "linear logdet must move by 2p·log(4) when all assignment masses double \
             (got {d_weight})"
        );
    }

    /// #1223 — the curved arm's Laplace complexity is the REAL weighted-design
    /// Gram log-determinant `p·log|ΦᵀWΦ|_+`, not a parameter-count proxy. Build a
    /// curved design whose columns are the constant and the centered coordinate
    /// (a 2-column basis), so `ΦᵀWΦ = diag(w_sum, s_tt)` exactly matches the
    /// linear arm's data Gram, and assert `curved_design_gram_logdet` returns
    /// `p·(log w_sum + log s_tt)` — the same determinant the linear arm reports
    /// on the same design weight. A proxy `M·log(w_sum)` would instead omit the
    /// `log(s_tt)` spread term, so this pins the genuine determinant.
    #[test]
    fn curved_gram_logdet_is_real_weighted_design_determinant() {
        let n = 40;
        let p = 3usize;
        let coords = Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)));
        let assign = Array1::<f64>::from_iter((0..n).map(|i| 0.5 + 0.01 * (i as f64)));

        // Mass-weighted coordinate mean and spread under wᵢ = a_k².
        let mut w_sum = 0.0;
        let mut t_bar = 0.0;
        for i in 0..n {
            let w = assign[i] * assign[i];
            w_sum += w;
            t_bar += w * coords[i];
        }
        t_bar /= w_sum;
        let mut s_tt = 0.0;
        for i in 0..n {
            let dt = coords[i] - t_bar;
            s_tt += assign[i] * assign[i] * dt * dt;
        }

        // Curved design columns: [1, (t − t̄)]. Its weighted Gram is exactly
        // diag(w_sum, s_tt) (the cross term Σ w·(t−t̄) vanishes by construction),
        // so log|ΦᵀWΦ| = log(w_sum) + log(s_tt).
        let mut phi = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            phi[[i, 0]] = 1.0;
            phi[[i, 1]] = coords[i] - t_bar;
        }
        let got = curved_design_gram_logdet(phi.view(), assign.view(), p)
            .expect("non-degenerate curved design has a determinant");
        let want = (p as f64) * (w_sum.ln() + s_tt.ln());
        assert!(
            (got - want).abs() < 1e-9,
            "curved Gram logdet must be the real p·log|ΦᵀWΦ| = {want}, got {got}"
        );

        // A rank-deficient design (a duplicated column) drops the null direction:
        // its determinant equals that of the single retained constant column,
        // p·log(w_sum), NOT a 2-column proxy.
        let mut phi_dup = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            phi_dup[[i, 0]] = 1.0;
            phi_dup[[i, 1]] = 1.0;
        }
        let got_dup = curved_design_gram_logdet(phi_dup.view(), assign.view(), p)
            .expect("rank-1 design still has a positive determinant");
        let want_dup = (p as f64) * (2.0 * w_sum).ln();
        assert!(
            (got_dup - want_dup).abs() < 1e-9,
            "rank-deficient curved Gram must report only its positive direction \
             (p·log(2·w_sum) = {want_dup}), got {got_dup}"
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
                Some(0.0)
            )
            .is_none(),
            "a degenerate coordinate span must be refused"
        );
    }
}
