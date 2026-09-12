//! Crosscoder stacked-column layout on [`SaeManifoldTerm`]: installing a
//! [`CrosscoderLayout`] and reading each layer's honest-units decoder out of the
//! augmented width.

use super::*;

impl SaeManifoldTerm {
    /// Install the crosscoder stacked-column layout on the term, validating that
    /// its total width matches the atoms' output dimension
    /// (`p_x + Σ_ℓ p_ℓ == output_dim()`). Installing one enables
    /// [`Self::layer_decoder`] on a term whose decoders were fit at the augmented
    /// width.
    pub fn set_crosscoder_layout(&mut self, layout: CrosscoderLayout) -> Result<(), String> {
        if layout.total_dim() != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::set_crosscoder_layout: layout total width p_x + Σ p_ℓ = {} but \
                 the term's output_dim is {} (atoms must be built at the augmented width)",
                layout.total_dim(),
                self.output_dim()
            ));
        }
        self.crosscoder_layout = Some(layout);
        Ok(())
    }

    /// The installed crosscoder stacked-column layout, or `None` for a term with
    /// no multi-block layout recorded.
    pub fn crosscoder_layout(&self) -> Option<&CrosscoderLayout> {
        self.crosscoder_layout.as_ref()
    }

    /// The HONEST-units per-layer decoder `B_k^(ℓ)` of atom `k`, output block `ℓ`:
    /// the decoder columns of block `ℓ` divided by `√λ_ℓ` (un-doing the target
    /// scaling that `stack_augmented_target` applied), and — when a Tier-0
    /// column-equilibration scale is installed (#2015; see
    /// [`Self::set_tier0_scale`])
    /// — un-doing that per-column scale too, so the returned decoder is honest in
    /// the CALLER's raw target units regardless of the internal conditioning
    /// frame the inner solve actually ran on. Requires an installed
    /// [`CrosscoderLayout`] (via [`Self::set_crosscoder_layout`]).
    ///
    /// This is the first-class form of the by-hand slice+unscale
    /// (`decoder_coefficients[:, off_ℓ..off_ℓ+p_ℓ] / √λ_ℓ`, times the column's
    /// tier0 scale): identical values, but the offsets, `√λ_ℓ`, and tier0 scale
    /// are owned here so no caller recomputes them.
    pub fn layer_decoder(&self, k: usize, l: usize) -> Result<Array2<f64>, String> {
        let layout = self.crosscoder_layout.as_ref().ok_or_else(|| {
            "SaeManifoldTerm::layer_decoder: no crosscoder layout installed (call \
             set_crosscoder_layout first)"
                .to_string()
        })?;
        if k >= self.atoms.len() {
            return Err(format!(
                "SaeManifoldTerm::layer_decoder: atom index k={k} out of range (K = {})",
                self.atoms.len()
            ));
        }
        if l >= layout.num_blocks() {
            return Err(format!(
                "SaeManifoldTerm::layer_decoder: block index ℓ={l} out of range (L-1 = {})",
                layout.num_blocks()
            ));
        }
        let inv = 1.0 / layout.sqrt_lambda(l);
        // Materialize the full-width decoder (Tier-0 column scale already
        // undone) before crossing the layer boundary; the fit may use a
        // reduced Grassmann coordinate internally.
        let physical = self.tier0_unscaled_full_width_decoder(k);
        let scaled = physical.slice(s![.., layout.block_range(l)]);
        Ok(scaled.mapv(|value| inv * value))
    }

    /// The full-width (augmented) decoder of atom `k`, with the Tier-0
    /// column-equilibration scale (#2015; [`Self::set_tier0_scale`]) undone
    /// column-by-column
    /// when one is installed — a strict no-op on the historical (unequilibrated)
    /// path. Every consumer that carves an honest per-layer decoder out of the
    /// full augmented width ([`Self::layer_decoder`] above, the crosscoder
    /// drift/transport reports) starts from this, so an equilibration scale
    /// installed by the crosscoder fit entry is undone exactly once, in one
    /// place, rather than re-derived at each call site.
    pub(crate) fn tier0_unscaled_full_width_decoder(&self, k: usize) -> Array2<f64> {
        let mut decoder = self.atoms[k].full_width_decoder();
        if let Some(scale) = self.tier0_scale() {
            for (col, &s) in scale.iter().enumerate() {
                decoder.column_mut(col).mapv_inplace(|v| v * s);
            }
        }
        decoder
    }
}
