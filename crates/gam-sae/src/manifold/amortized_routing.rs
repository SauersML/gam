//! #1033 — the chart-geometry amortized-routing predictor (kept out of
//! `construction.rs`, which sits at the #780 line-count ceiling). A separate
//! `impl SaeManifoldTerm` block deriving the ρ-invariant frozen routing from the
//! current dictionary's encode-chart geometry; see
//! [`RoutingPredictor::ChartGeometry`].

use super::*;

impl SaeManifoldTerm {
    /// #1033 — CHART-GEOMETRY amortized routing predictor: derive a ρ-invariant
    /// per-(row, atom) routing LOGIT from the current dictionary's encode-chart
    /// geometry, with NO learned net. For each atom it runs the amortized encode
    /// of `targets` (the atlas distilled from the current decoder/basis) to a
    /// predicted coord `t̂_ik`, reconstructs the amplitude-1 image
    /// `γ_k(t̂_ik) = Bᵀ φ(t̂_ik)`, and maps the reconstruction ALIGNMENT
    /// `⟨x_i, γ̂_ik⟩` (with `γ̂` the unit-normalized image) to a logit via the
    /// fixed `gate_logit_scale`. Higher alignment ⇒ larger logit ⇒ the gate
    /// prefers atom `k` for row `i` — exactly the routing the dictionary's own
    /// geometry implies, recomputed from the CURRENT decoder so it tracks the
    /// dictionary as it evolves (unlike a static logit snapshot). The overall
    /// `gate_logit_scale` (a temperature) is the single calibratable knob; the
    /// cosine-aligned `⟨x, γ̂⟩` is otherwise on the natural `‖x‖` scale.
    ///
    /// This is the [`RoutingPredictor::ChartGeometry`] arm: install its output via
    /// `assignment.with_frozen_routing(Some(..))` to freeze the routing for an
    /// outer iterate. An atom whose predicted coord is uncertified falls back to
    /// the row's CURRENT logit (the encoder cannot improve a row it cannot
    /// certify, so the existing routing for that row is preserved).
    pub fn chart_geometry_routing_logits(
        &self,
        targets: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        gate_logit_scale: f64,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let p = self.output_dim();
        if targets.dim() != (n, p) {
            return Err(format!(
                "chart_geometry_routing_logits: targets {:?} != ({n}, {p})",
                targets.dim()
            ));
        }
        // Per-atom amortized encode (atlas distilled from the current dictionary)
        // → predicted coords t̂ + per-row certificates.
        let encoded = self.amortized_encode_fitted(targets, rho)?;
        // Start from the current logits so an uncertified row keeps its existing
        // routing (the predictor only OVERRIDES rows it can certify).
        let mut logits = self.assignment.logits.clone();
        for atom_idx in 0..k_atoms {
            let atom = &self.atoms[atom_idx];
            let evaluator = match atom.basis_evaluator.as_ref() {
                Some(e) => e.clone(),
                // No evaluator ⇒ cannot reconstruct at a new coord; keep current
                // logits for this atom.
                None => continue,
            };
            let result = &encoded[atom_idx];
            let m = atom.basis_size();
            for row in 0..n {
                if !result.certified[row] {
                    continue;
                }
                // Reconstruct the amplitude-1 image γ_k(t̂) = Bᵀ φ(t̂).
                let t_hat: Vec<f64> = result.coords.row(row).iter().copied().collect();
                let coord_2d = Array2::from_shape_vec((1, atom.latent_dim), t_hat)
                    .map_err(|e| format!("chart_geometry_routing_logits: coord reshape: {e}"))?;
                let (phi, _jet) = evaluator.evaluate(coord_2d.view())?;
                let mut gamma = vec![0.0_f64; p];
                for basis_col in 0..m {
                    let phi_v = phi[[0, basis_col]];
                    if phi_v == 0.0 {
                        continue;
                    }
                    for out in 0..p {
                        gamma[out] += phi_v * atom.decoder_coefficients[[basis_col, out]];
                    }
                }
                // Unit-normalize the reconstruction (routing is amplitude-free —
                // the gate cares about DIRECTION/alignment, not magnitude) and
                // align with the row.
                let norm = gamma.iter().map(|g| g * g).sum::<f64>().sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    continue;
                }
                let mut align = 0.0_f64;
                for out in 0..p {
                    align += targets[[row, out]] * (gamma[out] / norm);
                }
                if align.is_finite() {
                    logits[[row, atom_idx]] = gate_logit_scale * align;
                }
            }
        }
        Ok(logits)
    }
}
