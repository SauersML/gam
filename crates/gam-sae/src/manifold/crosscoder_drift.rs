//! Cross-layer drift statistic for a fitted manifold crosscoder (gam#2231 Inc E).
//!
//! A crosscoder shares ONE latent `t` and ONE routing across all layers: every
//! atom `k` carries a single set of basis coefficients evaluated through the SAME
//! chart `Φ(t)`, and its per-layer decoder `B_k^(ℓ)` is a column block of the one
//! augmented decoder (honest units after un-doing the `√λ_ℓ` target scaling —
//! [`SaeManifoldTerm::layer_decoder`]). The scientific question the crosscoder was
//! built to answer is: **how much does an atom's decoded feature direction rotate
//! as it moves through the layer stack?** This module measures that.
//!
//! For each atom `k` and each consecutive layer step `ℓ → ℓ+1` along the chain
//! `Anchor → Block(0) → … → Block(L−2)`, the drift statistic is
//!
//! ```text
//!   δ_k(ℓ) = ‖B_k^(ℓ+1) − B_k^(ℓ)‖_F / √(‖B_k^(ℓ)‖_F · ‖B_k^(ℓ+1)‖_F)
//! ```
//!
//! (the honest-units Frobenius drift shared with [`super::transport_law`]) together
//! with the principal angles between the two layer IMAGES — the row spaces of the
//! honest decoders in `ℝ^p`, whose largest angle is the worst-case rotation of the
//! decoded curve.
//!
//! **Chart-gauge invariance — what is and is not invariant.** Every layer of a
//! given atom is decoded through the SAME `Φ(t)`; a chart reparametrization
//! `t ↦ ψ(t)` changes `Φ` (hence every `B_k^(ℓ)`) by ONE common change of basis
//! `W` on the shared `M` basis rows. The principal angles between the row
//! spaces are invariant under ANY invertible common `W` (row spaces are
//! `W`-invariant) — they are true gauge invariants. The normalized Frobenius
//! drift is invariant only under an ORTHOGONAL common `W`: a general
//! non-orthogonal reparametrization changes `‖W(A−B)‖_F/√(‖WA‖·‖WB‖)`, so the
//! drift scalar (and the `atom_total_drift` ranking built on it) is
//! gauge-COVARIANT — meaningful within one fitted crosscoder's own chart, not
//! across re-gauged refits. Use the angles for cross-gauge comparisons.
//!
//! **`λ_ℓ` independence.** The decoders are read in honest units (`B_k^(ℓ)`
//! divides block `ℓ`'s columns by `√λ_ℓ`), so the per-block REML relevance weight
//! `λ_ℓ` — the outer-REML coordinate that scales the stacked target — cancels and
//! never enters the drift. Re-weighting a block changes what the fit optimizes, not
//! the geometry this statistic reports.

use super::*;
use crate::manifold::transport_law::{
    CrosscoderLayer, decoder_drift, honest_layer_decoder, principal_angles_between_images,
};

/// The drift of one atom across one consecutive layer step `source → target`.
#[derive(Clone, Debug)]
pub struct LayerStepDrift {
    /// The atom this step is measured for.
    pub atom: usize,
    /// The source layer of the step (`Anchor` or `Block(ℓ)`).
    pub source: CrosscoderLayer,
    /// The target layer of the step (the next layer in the chain).
    pub target: CrosscoderLayer,
    /// Honest-units Frobenius drift `δ_k(ℓ) = ‖B_tgt − B_src‖_F /
    /// √(‖B_src‖_F · ‖B_tgt‖_F)`. `NaN` when either decoder is numerically zero
    /// (a dead/empty atom at that layer).
    pub drift: f64,
    /// Principal angles (radians, ascending) between the two layer images (the row
    /// spaces of the honest decoders). Length `max(rank_src, rank_tgt)`; unmatched
    /// directions from a rank change are represented by `π/2`. Empty only when
    /// both images are numerically rank-0.
    pub principal_angles: Vec<f64>,
}

impl LayerStepDrift {
    /// The largest principal angle (radians) — the worst-case rotation of the
    /// decoded curve across this step. `0.0` only when both images are rank zero.
    pub fn max_principal_angle(&self) -> f64 {
        // `principal_angles` is ascending, so the last entry is the maximum.
        self.principal_angles.last().copied().unwrap_or(0.0)
    }
}

/// The whole-dictionary cross-layer drift report of a fitted crosscoder.
///
/// The `steps` are grouped by atom (all of atom `0`'s consecutive-step drifts, then
/// atom `1`'s, …), each group in chain order, so `steps[k * num_steps + s]` is
/// atom `k`'s step `s`. `num_steps = layer_chain.len() − 1`.
#[derive(Clone, Debug)]
pub struct CrosscoderDriftReport {
    /// Number of atoms `K` in the dictionary.
    pub num_atoms: usize,
    /// The ordered layer chain the drift walks: `Anchor` then every `Block(ℓ)` in
    /// order. Length `L` (one anchor + `L−1` output blocks).
    pub layer_chain: Vec<CrosscoderLayer>,
    /// Per-atom, per-step drift, grouped by atom then chain order (see the struct
    /// docs). Length `num_atoms · (layer_chain.len() − 1)`.
    pub steps: Vec<LayerStepDrift>,
}

impl CrosscoderDriftReport {
    /// Number of consecutive layer steps per atom (`L − 1`).
    pub fn num_steps(&self) -> usize {
        self.layer_chain.len().saturating_sub(1)
    }

    /// Atom `k`'s drift profile `[δ_k(0), …, δ_k(L−2)]` along the layer chain.
    ///
    /// # Panics
    /// If `k >= num_atoms`.
    pub fn atom_drift_profile(&self, k: usize) -> Vec<f64> {
        assert!(
            k < self.num_atoms,
            "atom_drift_profile: atom {k} out of range (K = {})",
            self.num_atoms
        );
        let ns = self.num_steps();
        self.steps[k * ns..(k + 1) * ns]
            .iter()
            .map(|s| s.drift)
            .collect()
    }

    /// Atom `k`'s total drift: the sum of its finite per-step drifts (a `NaN` step,
    /// a dead atom at some layer, contributes `0`). The dictionary-level ranking key
    /// for "how much does this feature move through the stack".
    ///
    /// # Panics
    /// If `k >= num_atoms`.
    pub fn atom_total_drift(&self, k: usize) -> f64 {
        self.atom_drift_profile(k)
            .into_iter()
            .filter(|d| d.is_finite())
            .sum()
    }

    /// Mean per-step drift over every atom and step whose drift is finite. `NaN`
    /// when there are no finite steps (e.g. a zero dictionary).
    pub fn mean_drift(&self) -> f64 {
        let (sum, count) = self
            .steps
            .iter()
            .map(|s| s.drift)
            .filter(|d| d.is_finite())
            .fold((0.0_f64, 0usize), |(sum, count), d| (sum + d, count + 1));
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }

    /// The atom with the largest total drift (the feature that rotates the most
    /// through the stack). `None` when there are no atoms or no finite drift.
    pub fn most_drifting_atom(&self) -> Option<usize> {
        self.extremal_atom(true)
    }

    /// The atom with the smallest total drift (the most layer-stable feature).
    /// `None` when there are no atoms or no finite drift.
    pub fn most_stable_atom(&self) -> Option<usize> {
        self.extremal_atom(false)
    }

    fn extremal_atom(&self, want_max: bool) -> Option<usize> {
        (0..self.num_atoms)
            .map(|k| (k, self.atom_total_drift(k)))
            .filter(|(_, d)| d.is_finite())
            .max_by(|(_, a), (_, b)| {
                let ord = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
                if want_max { ord } else { ord.reverse() }
            })
            .map(|(k, _)| k)
    }
}

/// Measure the cross-layer drift of every atom in a fitted crosscoder `term` under
/// its `layout` (design gam#2231 Inc E). See the module header for the definition
/// and the gauge/`λ_ℓ`-invariance argument.
///
/// The layer chain is `Anchor → Block(0) → … → Block(L−2)`; a step is measured
/// between consecutive layers. Because a crosscoder shares the residual-stream
/// dimension across layers, every layer in the chain must have the SAME ambient
/// width `p` — the principal-angle and drift geometry is only defined for images in
/// one ambient space. A layout with differing block widths is rejected up front.
///
/// Requires `layout.total_dim() == term.output_dim()` (the layout describes this
/// term's augmented columns) and at least one output block (`L ≥ 2`).
pub fn measure_crosscoder_drift(
    term: &SaeManifoldTerm,
    layout: &CrosscoderLayout,
) -> Result<CrosscoderDriftReport, String> {
    if layout.total_dim() != term.output_dim() {
        return Err(format!(
            "measure_crosscoder_drift: layout total width {} != term output_dim {} (the layout \
             must describe this term's augmented columns)",
            layout.total_dim(),
            term.output_dim()
        ));
    }
    if layout.num_blocks() == 0 {
        return Err(
            "measure_crosscoder_drift: need at least one output block (a plain SAE has no layer \
             chain to drift along)"
                .to_string(),
        );
    }

    // The chain must live in ONE ambient: the anchor width and every block width
    // must agree (a crosscoder shares the residual-stream dimension across layers).
    let p_x = layout.anchor_dim();
    for (l, &p_l) in layout.block_dims().iter().enumerate() {
        if p_l != p_x {
            return Err(format!(
                "measure_crosscoder_drift: layer widths differ (anchor p_x = {p_x}, block {l} \
                 '{}' p_ℓ = {p_l}) — cross-layer drift needs every layer image in one ambient \
                 space",
                layout.labels()[l]
            ));
        }
    }

    // Ordered layer chain: Anchor, then Block(0..L-1).
    let mut layer_chain = Vec::with_capacity(layout.num_blocks() + 1);
    layer_chain.push(CrosscoderLayer::Anchor);
    for l in 0..layout.num_blocks() {
        layer_chain.push(CrosscoderLayer::Block(l));
    }

    let num_atoms = term.atoms.len();
    let num_steps = layer_chain.len() - 1;
    let mut steps = Vec::with_capacity(num_atoms * num_steps);
    for k in 0..num_atoms {
        let decoder = term.atoms[k].full_width_decoder();
        // Honest-units decoder at each layer, once per atom (reused across steps).
        let mut honest: Vec<Array2<f64>> = Vec::with_capacity(layer_chain.len());
        for &layer in &layer_chain {
            honest.push(honest_layer_decoder(&decoder, layout, layer)?);
        }
        for s in 0..num_steps {
            let drift = decoder_drift(&honest[s], &honest[s + 1]);
            let principal_angles = principal_angles_between_images(&honest[s], &honest[s + 1])?;
            steps.push(LayerStepDrift {
                atom: k,
                source: layer_chain[s],
                target: layer_chain[s + 1],
                drift,
                principal_angles,
            });
        }
    }

    Ok(CrosscoderDriftReport {
        num_atoms,
        layer_chain,
        steps,
    })
}
