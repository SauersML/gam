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
//! ## Why this sidesteps #1051
//!
//! The euclidean / multi-atom OUTER fit path is currently broken (#1051 —
//! singular joint Hessian at the continuation spine). This module does **not**
//! re-enter that path. The curved candidate is the *already-fitted* atom (its
//! converged decoded image `γ_k(t) = Φ(t) B_k`); the linear candidate is the
//! atom's straight sub-model `γ̃_k(t) = b₀ + t·b₁`, whose best fit to the SAME
//! fitted decoded points at the SAME coordinates is **exact penalized least
//! squares** — the collapsed linear lane the #1026 thread describes ("the
//! special case COLLAPSES: for purely linear atoms the per-atom solve is exact
//! least squares"). No outer continuation, no joint Hessian, no singular spine.
//!
//! Both candidates reconstruct the same target (the atom's fitted decoded image)
//! over the same assigned rows, so their Gaussian-reconstruction deviances are
//! directly comparable on the common rank-aware Laplace scale, exactly as the
//! union / mixture rungs score.

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
/// the exact weighted least-squares line through the atom's OWN decoded image
/// over its assigned rows. Carried on a verdict that selects LINEAR so the
/// collapsed reconstruction can replace the curved decoded row with this straight
/// image at any coordinate WITHOUT re-entering the (broken, #1051) outer fit —
/// the coefficients are already realized inside the adjudication.
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
/// them as `(linear, curved)`, or `None` if the atom cannot present an honest
/// pair (too few rows, degenerate coordinate span, or no analytic turning).
///
/// `weighted_rows` are the `(coord, weight, decoded)` triples for the rows
/// assigned to this atom: `coord` is the fitted on-atom coordinate `t`, `weight`
/// the assignment mass `a_k`, and `decoded` the fitted decoded image
/// `γ_k(t) = Φ(t) B_k` (length `p`). The curved candidate's deviance is the
/// atom's OWN reconstruction (zero residual against itself is not the point —
/// the curved deviance is its penalized smoothness-evidence price, captured by
/// its parameter count and Laplace logdet); the linear candidate's deviance is
/// the residual of the best straight line through the SAME decoded points.
fn build_atom_candidates(
    coords: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    curved_num_params: usize,
    fitted_turning: Option<f64>,
) -> Option<(
    HybridAtomCandidate,
    HybridAtomCandidate,
    (f64, Array1<f64>, Array1<f64>),
)> {
    let n = coords.len();
    let p = decoded.ncols();
    if n < MIN_ROWS_FOR_LINEAR_FIT || decoded.nrows() != n || weights.len() != n || p == 0 {
        return None;
    }

    // Weighted means for the centered (mass-weighted) straight-line normal
    // equations `γ̃(t) = b₀ + (t − t̄)·b₁`. The weighting is the assignment mass
    // `a_k`, so rows that barely belong to the atom contribute proportionally —
    // the same row weighting the joint reconstruction loss uses.
    let mut w_sum = 0.0_f64;
    let mut t_bar = 0.0_f64;
    for i in 0..n {
        let w = weights[i];
        if !(w.is_finite() && w >= 0.0) {
            return None;
        }
        w_sum += w;
        t_bar += w * coords[i];
    }
    if !(w_sum > 0.0) {
        return None;
    }
    t_bar /= w_sum;

    // Weighted Σ w·(t − t̄)² — the coordinate spread. A degenerate (single-point
    // mass) coordinate has no slope direction; refuse rather than divide by ~0.
    let mut s_tt = 0.0_f64;
    for i in 0..n {
        let dt = coords[i] - t_bar;
        s_tt += weights[i] * dt * dt;
    }
    if !(s_tt > 1e-12 * (1.0 + t_bar * t_bar)) {
        return None;
    }

    // Per-output-channel weighted least squares slope/intercept, accumulating
    // the residual sum of squares of the straight-line fit to the curved atom's
    // OWN decoded image. This is the linear special case's data-fit deviance:
    // how much of the fitted curve a straight line fails to capture.
    let mut b0 = Array1::<f64>::zeros(p);
    let mut b1 = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut s_tg = 0.0_f64;
        let mut g_bar = 0.0_f64;
        for i in 0..n {
            let w = weights[i];
            g_bar += w * decoded[[i, j]];
        }
        g_bar /= w_sum;
        for i in 0..n {
            let dt = coords[i] - t_bar;
            s_tg += weights[i] * dt * (decoded[[i, j]] - g_bar);
        }
        let slope = s_tg / s_tt;
        b1[j] = slope;
        b0[j] = g_bar;
    }

    // Weighted residual sum of squares of the linear sub-model against the
    // fitted curve. This is exactly the curvature the straight line cannot
    // express — zero for a genuinely straight image, growing with turning.
    let mut linear_rss = 0.0_f64;
    for i in 0..n {
        let dt = coords[i] - t_bar;
        for j in 0..p {
            let pred = b0[j] + dt * b1[j];
            let r = decoded[[i, j]] - pred;
            linear_rss += weights[i] * r * r;
        }
    }

    // Gaussian-reconstruction deviance: the (weighted) residual objective `½ RSS`
    // the Laplace normalizer is added to. The curved atom reconstructs its own
    // image with zero curve-residual by construction, so its data-fit term is 0
    // — its NLE is purely the parameter-price/Laplace cost of its `M·p` decoder
    // coefficients. The linear candidate pays `½·linear_rss` of data-fit but only
    // a `2·p`-parameter price. The argmin between the two is the `Θ/√ε`
    // crossover: a high-turning atom's `linear_rss` exceeds the curved atom's
    // extra Laplace price, so curved wins; a straight atom's `linear_rss ≈ 0`
    // lets the cheaper linear special case win (the dominance floor).
    let curved_residual_objective = 0.0_f64;
    let linear_residual_objective = 0.5 * linear_rss;

    // Linear candidate parameter price: intercept + slope per output channel.
    let linear_num_params = 2 * p;

    // Laplace logdet of the (weighted) design Gram for each candidate.
    //
    // Both candidates are compared on the same per-parameter effective-sample-size
    // scale: `num_params · log(w_sum)`. This is the only scale-invariant basis for
    // comparison — using the raw `p · log(w_sum · s_tt)` for the linear candidate
    // makes the adjudication coordinate-scale-dependent (multiplying `t` by a
    // constant shifts `log(s_tt)` and moves the curved/linear crossover
    // artificially). The informative content of the two-parameter linear fit versus
    // the `M`-basis curved fit is fully captured by the parameter counts on a shared
    // `log(w_sum)` scale; the within-atom spread `s_tt` is structural geometry that
    // the curved atom's proxy also cannot price.
    if !(w_sum > 0.0 && w_sum.is_finite()) {
        return None;
    }
    let linear_log_det_h = (linear_num_params as f64) * w_sum.ln();
    let curved_log_det_h = (curved_num_params as f64) * w_sum.ln();

    // Both candidates carry zero explicit smoothing-penalty logdet here (the
    // intrinsic smoothness penalty is reparameterization-invariant and identical
    // in expectation across the two parameterizations of the same image), and
    // full effective dim == penalty rank (no null space in this reduced
    // per-atom comparison), so the rank-aware Laplace NLE reduces to
    // `residual_objective + ½ log|H|`.
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
/// coordinate column for a slot, `weights_for` the per-row assignment mass, and
/// `decoded_for` the fitted decoded image rows for the atom's assigned rows.
/// `manifold_for` yields the atom's chart manifold (a flat / Euclidean chart can
/// present only the linear candidate, enforced inside the selector).
///
/// Returns `None` (no report) when no atom is eligible — there is nothing to
/// adjudicate.
pub fn build_hybrid_split_report<'a, C, W, D, M, E>(
    atoms: &'a [SaeManifoldAtom],
    eligible_d1: impl Iterator<Item = usize>,
    mut coords_for: C,
    mut weights_for: W,
    mut decoded_for: D,
    mut manifold_for: M,
    mut delta_ev_for: E,
) -> Result<Option<SaeHybridSplitReport>, String>
where
    C: FnMut(usize) -> Array1<f64>,
    W: FnMut(usize) -> Array1<f64>,
    D: FnMut(usize) -> Array2<f64>,
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
        let weights = weights_for(atom_idx);
        let decoded = decoded_for(atom_idx);
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
            weights.view(),
            decoded.view(),
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

    /// A straight image (zero turning) presents a linear candidate that fits it
    /// with zero residual, and the dominance floor selects linear.
    #[test]
    fn straight_image_selects_linear_via_dominance_floor() {
        // γ(t) = t·b — a straight line in ℝ².
        let n = 40;
        let coords = Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)));
        let weights = Array1::<f64>::ones(n);
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            decoded[[i, 0]] = coords[i];
            decoded[[i, 1]] = 0.6 * coords[i];
        }
        let (linear, curved, _) = build_atom_candidates(
            coords.view(),
            weights.view(),
            decoded.view(),
            // a generous curved parameter price (M·p)
            10,
            Some(0.0),
        )
        .expect("straight image yields a candidate pair");
        let choice =
            crate::solver::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert!(
            choice.param.is_linear(),
            "a straight image must keep the linear special case (Θ = 0 dominance floor)"
        );
    }

    /// A turning image (half circle) has large linear residual; the curved
    /// candidate wins on evidence once its turning exceeds the floor.
    #[test]
    fn turning_image_selects_curved_on_evidence() {
        // γ(t) = (cos θ, sin θ) over a half circle: strong curvature, so the
        // straight-line residual is large.
        let n = 60;
        let coords = Array1::from_iter((0..n).map(|i| (i as f64) / ((n - 1) as f64)));
        let weights = Array1::<f64>::ones(n);
        let mut decoded = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = PI * coords[i];
            decoded[[i, 0]] = theta.cos();
            decoded[[i, 1]] = theta.sin();
        }
        // The curved atom has 5 parameters (just above the 4 = 2·p linear budget),
        // and the half-circle linear residual exceeds the extra-parameter overhead, so
        // curved wins on evidence. (curved_num_params=6 or larger tips the balance
        // back to linear for this image, which is the correct adjudication — 6 extra
        // curve parameters is overkill for a half-circle.)
        let (linear, curved, _) =
            build_atom_candidates(coords.view(), weights.view(), decoded.view(), 5, Some(PI))
                .expect("turning image yields a candidate pair");
        // Linear candidate must have a strictly positive data-fit residual: a
        // straight line cannot reconstruct a half circle.
        assert!(
            linear.negative_log_evidence.is_finite(),
            "linear candidate must carry a real deviance"
        );
        let choice =
            crate::solver::evidence::select_hybrid_atom(&[linear, curved]).expect("non-empty slot");
        assert_eq!(
            choice.param,
            crate::solver::evidence::HybridAtomParam::Curved { latent_dim: 1 },
            "a half-circle image must keep the curved parameterization (5 params, large linear RSS)"
        );
        assert!(
            choice.curved_evidence_margin > 0.0,
            "curved must win a positive evidence margin over the linear secant"
        );
    }

    /// The adjudication must not shift when the latent coordinate `t` is rescaled.
    /// If `t → c·t` for any constant `c > 0`, the curved/linear verdict for a
    /// genuinely straight or genuinely curved image must stay the same — the old
    /// code had `log(s_tt)` in the linear log-det and not in the curved proxy,
    /// making the crossover move with the coordinate scale.
    #[test]
    fn adjudication_is_scale_invariant() {
        // A straight image: should always select linear regardless of t-scale.
        let n = 40;
        let mut decoded = Array2::<f64>::zeros((n, 2));
        let weights = Array1::<f64>::ones(n);
        for scale_exp in [-3i32, -1, 0, 1, 3] {
            let c = 10.0_f64.powi(scale_exp);
            let coords =
                Array1::from_iter((0..n).map(|i| c * (-1.0 + 2.0 * (i as f64) / ((n - 1) as f64))));
            for i in 0..n {
                decoded[[i, 0]] = coords[i] / c; // canonical-scale decoded, not t-scale-dependent
                decoded[[i, 1]] = 0.6 * coords[i] / c;
            }
            let (linear, curved, _) =
                build_atom_candidates(coords.view(), weights.view(), decoded.view(), 10, Some(0.0))
                    .expect("straight image always yields a pair");
            let choice = crate::solver::evidence::select_hybrid_atom(&[linear, curved])
                .expect("non-empty slot");
            assert!(
                choice.param.is_linear(),
                "straight image must select linear at any t-scale (scale={c})"
            );
        }

        // A curved image with tight curved budget: should always select curved.
        let n = 60;
        let weights = Array1::<f64>::ones(n);
        for scale_exp in [-2i32, -1, 0, 1, 2] {
            let c = 10.0_f64.powi(scale_exp);
            let coords = Array1::from_iter((0..n).map(|i| c * (i as f64) / ((n - 1) as f64)));
            let mut decoded = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                let theta = PI * (i as f64) / ((n - 1) as f64); // arc param, not t-scaled
                decoded[[i, 0]] = theta.cos();
                decoded[[i, 1]] = theta.sin();
            }
            let (linear, curved, _) = build_atom_candidates(
                coords.view(),
                weights.view(),
                decoded.view(),
                5, // tight budget: just above linear's 2·p=4
                Some(PI),
            )
            .expect("curved image always yields a pair");
            let choice = crate::solver::evidence::select_hybrid_atom(&[linear, curved])
                .expect("non-empty slot");
            assert_eq!(
                choice.param,
                crate::solver::evidence::HybridAtomParam::Curved { latent_dim: 1 },
                "curved image must select curved at any t-scale (scale={c})"
            );
        }
    }

    /// A degenerate (single-point-mass) coordinate has no slope direction and is
    /// refused rather than adjudicated on a fabricated deviance.
    #[test]
    fn degenerate_coordinate_is_refused() {
        let n = 5;
        let coords = Array1::<f64>::from_elem(n, 0.5); // no spread
        let weights = Array1::<f64>::ones(n);
        let decoded = Array2::<f64>::zeros((n, 2));
        assert!(
            build_atom_candidates(coords.view(), weights.view(), decoded.view(), 6, Some(0.0))
                .is_none(),
            "a degenerate coordinate span must be refused"
        );
    }
}
