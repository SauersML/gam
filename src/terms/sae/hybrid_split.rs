//! #1026 — load-bearing curved-vs-linear hybrid split for the fitted SAE
//! dictionary.
//!
//! The selection machinery ([`crate::solver::evidence::select_hybrid_atom`],
//! [`crate::solver::evidence::select_hybrid_split`]) and the per-atom
//! integration helper
//! ([`crate::terms::sae::assignment::select_hybrid_atom_parameterization`]) are
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

use crate::solver::evidence::{
    HybridAtomCandidate, HybridAtomChoice, HybridSplitSelection, select_hybrid_split,
};
use crate::terms::latent::LatentManifold;
use crate::terms::sae::chart_canonicalization::d1_atom_fitted_turning;
use crate::terms::sae::manifold::SaeManifoldAtom;

/// The rank-aware Laplace negative-log-evidence of a reduced per-atom Gaussian
/// reconstruction sub-model: `residual_objective + ½ log|H|` with no smoothing
/// penalty logdet and a full-rank design (no null space), which is the form
/// [`crate::solver::evidence::laplace_evidence`] reduces to on this comparison.
/// Kept inline (rather than routed through `EvidenceLogDetSource`) because both
/// candidates' Hessian logdets are already the closed-form scalar moments of
/// their shared design — no factor cache or HVP callback to assemble.
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
}

impl AtomLinearImage {
    /// Evaluate the straight sub-model `b₀ + (t − t̄)·b₁` into `out` (length `p`).
    pub fn fill_row(&self, t: f64, out: &mut [f64]) {
        let dt = t - self.t_bar;
        for (j, slot) in out.iter_mut().enumerate() {
            *slot = self.b0[j] + dt * self.b1[j];
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
    /// The atom's held-out leave-one-atom-out explained-variance contribution
    /// `ΔEV_k = EV(full) − EV(full∖{k})` — how much reconstruction EV this single
    /// atom earns. Paired with [`Self::fitted_turning`] this is the `(Θ, ΔEV)`
    /// point the #1026 frontier reports: a `Θ ≈ 0` atom with large `ΔEV` is a
    /// genuine linear-tail direction; a high-`Θ` atom with large `ΔEV` is a
    /// genuine curved family. `None` iff the caller did not supply LOAO EV.
    pub held_out_delta_ev: Option<f64>,
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
    // The curved arm's genuine Laplace determinant over its `M·p` decoder
    // coefficients is not assembled in this post-fit path (it would need the
    // atom's penalized smoothing Hessian); we price its complexity with the
    // parameter-count proxy `curved_num_params · log(w_sum)`. The data-fit terms,
    // which carry the actual evidence comparison, are now BOTH measured on the
    // common response residual — the asymmetry is confined to the complexity
    // (logdet) price, not the likelihood.
    if !(w_sum > 0.0 && w_sum.is_finite() && s_tt.is_finite()) {
        return None;
    }
    let linear_log_det_h = (p as f64) * (w_sum.ln() + s_tt.ln());
    let curved_log_det_h = (curved_num_params as f64) * w_sum.ln();

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
#[allow(clippy::too_many_arguments)]
pub fn build_hybrid_split_report<'a, C, W, D, R, M, E>(
    atoms: &'a [SaeManifoldAtom],
    eligible_d1: impl Iterator<Item = usize>,
    mut coords_for: C,
    mut assign_for: W,
    mut decoded_for: D,
    mut target_resid_for: R,
    mut manifold_for: M,
    mut delta_ev_for: E,
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
        let Some((linear, curved, (t_bar, b0, b1))) = build_atom_candidates(
            coords.view(),
            assign.view(),
            decoded.view(),
            target_resid.view(),
            curved_num_params,
            fitted_turning,
        ) else {
            continue;
        };
        // A flat (Euclidean) chart cannot honestly present a curved candidate;
        // the selector drops it. Present both for curveable charts.
        let manifold = manifold_for(atom_idx);
        let slot = if manifold.is_euclidean() {
            vec![linear]
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
        });
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
            |((((atom_name, choice), linear_image), fitted_turning), held_out_delta_ev)| {
                let kept_curved = !choice.param.is_linear();
                AtomHybridVerdict {
                    atom_name,
                    choice,
                    kept_curved,
                    fitted_turning,
                    held_out_delta_ev,
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
            Some(0.0),
        )
        .expect("straight residual yields a candidate pair");
        let choice =
            crate::solver::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
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
            crate::solver::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert_eq!(
            choice.param,
            crate::solver::evidence::HybridAtomParam::Curved { latent_dim: 1 },
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
            Some(1.0),
        )
        .expect("candidate pair");
        let choice =
            crate::solver::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
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
                Some(0.0)
            )
            .is_none(),
            "a degenerate coordinate span must be refused"
        );
    }
}
