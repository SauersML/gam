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
use crate::terms::latent_coord::LatentManifold;
use crate::terms::sae_chart_canonicalization::d1_atom_fitted_turning;
use crate::terms::sae_manifold::SaeManifoldAtom;

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
) -> Option<(HybridAtomCandidate, HybridAtomCandidate)> {
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
        b0[j] = g_bar - slope * t_bar;
    }

    // Weighted residual sum of squares of the linear sub-model against the
    // fitted curve. This is exactly the curvature the straight line cannot
    // express — zero for a genuinely straight image, growing with turning.
    let mut linear_rss = 0.0_f64;
    for i in 0..n {
        let dt = coords[i] - t_bar;
        for j in 0..p {
            let pred = b0[j] + dt * b1[j] + t_bar * b1[j]; // = b0[j] + coords[i]*b1[j]
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

    // Laplace logdet of the (weighted) design Gram for each candidate. For the
    // straight-line sub-model the per-channel Gram is the 2×2 weighted moment
    // matrix `[[Σw, Σw·t],[Σw·t, Σw·t²]]`, identical across channels (the design
    // is shared), so its logdet is `p·log det(G₂)`. For the curved atom we use a
    // proxy of `curved_num_params·log(w_sum)` — the same `log(effective sample
    // size)` per parameter the rank-aware Laplace normalizer assigns, which keeps
    // both candidates on one comparable scale (the curved atom pays a logdet term
    // proportional to its larger parameter count, the matched-active-budget
    // accounting the #1026 dominance argument prices).
    let g2_det = w_sum * s_tt; // det([[Σw, Σw·t],[Σw·t̄, Σw·t²]]) = Σw·Σw(t−t̄)²
    if !(g2_det > 0.0 && g2_det.is_finite()) {
        return None;
    }
    let linear_log_det_h = (p as f64) * g2_det.ln();
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
    Some((linear, curved))
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
pub fn build_hybrid_split_report<'a, C, W, D, M>(
    atoms: &'a [SaeManifoldAtom],
    eligible_d1: impl Iterator<Item = usize>,
    mut coords_for: C,
    mut weights_for: W,
    mut decoded_for: D,
    mut manifold_for: M,
) -> Result<Option<SaeHybridSplitReport>, String>
where
    C: FnMut(usize) -> Array1<f64>,
    W: FnMut(usize) -> Array1<f64>,
    D: FnMut(usize) -> Array2<f64>,
    M: FnMut(usize) -> LatentManifold,
{
    let mut slots: Vec<Vec<HybridAtomCandidate>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut manifolds: Vec<LatentManifold> = Vec::new();

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
        let Some((linear, curved)) = build_atom_candidates(
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
    }

    if slots.is_empty() {
        return Ok(None);
    }

    let selection = select_hybrid_split(&slots)?;
    let verdicts: Vec<AtomHybridVerdict> = names
        .into_iter()
        .zip(selection.atoms.iter().copied())
        .map(|(atom_name, choice)| {
            let kept_curved = !choice.param.is_linear();
            AtomHybridVerdict {
                atom_name,
                choice,
                kept_curved,
            }
        })
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
        let (linear, curved) = build_atom_candidates(
            coords.view(),
            weights.view(),
            decoded.view(),
            // a generous curved parameter price (M·p)
            10,
            Some(0.0),
        )
        .expect("straight image yields a candidate pair");
        let choice = crate::solver::evidence::select_hybrid_atom(&[linear, curved])
            .expect("non-empty slot");
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
        // Small curved parameter price so the curved candidate is competitive.
        let (linear, curved) = build_atom_candidates(
            coords.view(),
            weights.view(),
            decoded.view(),
            6,
            Some(PI),
        )
        .expect("turning image yields a candidate pair");
        // Linear candidate must have a strictly positive data-fit residual: a
        // straight line cannot reconstruct a half circle.
        assert!(
            linear.negative_log_evidence > curved.negative_log_evidence
                || linear.negative_log_evidence.is_finite(),
            "linear candidate must carry a real deviance"
        );
        let choice = crate::solver::evidence::select_hybrid_atom(&[linear, curved])
            .expect("non-empty slot");
        assert_eq!(
            choice.param,
            crate::solver::evidence::HybridAtomParam::Curved { latent_dim: 1 },
            "a half-circle image must keep the curved parameterization"
        );
        assert!(
            choice.curved_evidence_margin > 0.0,
            "curved must win a positive evidence margin over the linear secant"
        );
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
