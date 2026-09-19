// [#780 line-count gate] The decoder-smoothness effective dof and κ-penalty traces
// taken off a reduced-Schur probe bundle, split out of `construction.rs` (which sits
// against the 10k-line gate). Included via `include!` from `construction.rs` so it
// keeps the SAME module scope (`use super::*`), the same `impl SaeManifoldTerm`
// surface, and full private-field access.

impl SaeManifoldTerm {
    /// Per-atom decoder-smoothness effective dof `edof_k = tr((H⁻¹)_ββ·M_k)`,
    /// `M_k = (λ_k·½(S_k+S_kᵀ)) ⊗ I_{r_k}`, from the #2080 SHARED selected-inverse
    /// bundle instead of a dense `(H⁻¹)_ββ`: the surrogate lane's frozen probes
    /// `z_j` and their `S⁻¹ z_j` (t = 0) solves. Estimated through the tr(S⁻¹·M)
    /// umbrella `edof_k = (1/m)Σ_j (S⁻¹z_j)ᵀ(M_k z_j)` with `M_k` row-local (only
    /// atom `k`'s β-block is nonzero, so this is exactly `tr((S⁻¹)_{kk} M_k)`,
    /// matching the dense path's per-atom column trace). Reuses ONE
    /// `(probes, S⁻¹·probes)` pair across every gradient channel so the value and
    /// the ρ-gradient never desync — the matrix-free replacement for the dense
    /// `beta_inv` in
    /// [`SaeManifoldTerm::decoder_smoothness_effective_dof_with_solver_per_atom`]
    /// on the massive-`K` surrogate lane. The probe/solve vectors have length
    /// `border_dim` (the reduced-Schur dimension `cache.k`).
    pub(crate) fn decoder_smoothness_effective_dof_per_atom_from_probes(
        &self,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, String> {
        let (offsets, ranks) = self.decoder_border_blocks();
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = atom.smooth_penalty();
            let off = offsets[atom_idx];
            let r = ranks[atom_idx];
            let lambda = lambda_smooth[atom_idx];
            // M_k·v: block-diagonal `(λ_k·½(S_k+S_kᵀ)) ⊗ I_{r_k}` restricted to
            // atom `k`'s β-block, matching the exact path's `M[:,col]` column
            // construction row-for-row.
            let m_apply =
                |v: ArrayView1<'_, f64>| Self::decoder_penalty_block_apply(v, off, r, lambda, s);
            per_atom[atom_idx] =
                hutchinson_reduced_schur_inverse_trace(probes, sinv_probes, &m_apply).ok_or_else(
                    || {
                        format!(
                            "decoder_smoothness_effective_dof_per_atom_from_probes: non-finite \
                             Hutchinson trace for atom {atom_idx}"
                        )
                    },
                )?;
        }
        Ok(per_atom)
    }

    /// β-block offsets and channel counts of every atom in the border layout: the
    /// factored `M_k·r_k` blocks when frames are active, the full `M_k·p` otherwise.
    fn decoder_border_blocks(&self) -> (Vec<usize>, Vec<usize>) {
        if self.frames_active() {
            (
                self.factored_beta_offsets(),
                self.atoms.iter().map(|atom| atom.border_frame_rank()).collect(),
            )
        } else {
            (self.beta_offsets(), vec![self.output_dim(); self.atoms.len()])
        }
    }

    /// `(λ·½(P+Pᵀ)) ⊗ I_r` applied to the β-block of `v` at offset `off`: the
    /// penalty-curvature operator of a coordinate that scales (`P = S_k`) or
    /// reshapes (`P = ∂S_k/∂κ`) atom `k`'s penalty Gram.
    fn decoder_penalty_block_apply(
        v: ArrayView1<'_, f64>,
        off: usize,
        r: usize,
        lambda: f64,
        penalty: &Array2<f64>,
    ) -> Array1<f64> {
        let m = penalty.nrows();
        let mut out = Array1::<f64>::zeros(v.len());
        for nu in 0..m {
            for oc in 0..r {
                let mut acc = 0.0_f64;
                for mu in 0..m {
                    let p_nu_mu = 0.5 * (penalty[[nu, mu]] + penalty[[mu, nu]]);
                    acc += lambda * p_nu_mu * v[off + mu * r + oc];
                }
                out[off + nu * r + oc] = acc;
            }
        }
        out
    }

    /// #2935 — the atoms of `rho.kappa_atoms` whose penalty is
    /// curvature-parameterised, as `(flat index, atom index, ∂S/∂κ)`. An atom with
    /// no `∂S/∂κ` has no curvature to move and contributes to no channel.
    fn kappa_penalty_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<(usize, usize, &Array2<f64>)>, String> {
        let mut out = Vec::with_capacity(rho.kappa_atoms.len());
        for &atom_idx in &rho.kappa_atoms {
            let flat = rho.kappa_flat_index(atom_idx).ok_or_else(|| {
                format!("curvature atom {atom_idx} has no flat outer coordinate")
            })?;
            let atom = self.atoms.get(atom_idx).ok_or_else(|| {
                format!(
                    "curvature coordinate names atom {atom_idx}, outside term K={}",
                    self.atoms.len()
                )
            })?;
            if let Some(ds) = atom.smooth_penalty_kappa_derivative()? {
                out.push((flat, atom_idx, ds));
            }
        }
        Ok(out)
    }

    /// #2935 — `∂loss.smoothness/∂κ_k = ½·λ_k·<B_k, ∂S_k/∂κ B_k>` for every
    /// curvature coordinate, as `(flat index, derivative)`, before any
    /// `penalty_scale`. The energy is priced on the full decoder, so an active frame
    /// changes nothing here.
    pub(crate) fn decoder_smoothness_kappa_energy_derivatives(
        &self,
        rho: &SaeManifoldRho,
        lambda_smooth: &[f64],
    ) -> Result<Vec<(usize, f64)>, String> {
        let mut out = Vec::with_capacity(rho.kappa_atoms.len());
        for (flat, atom_idx, ds) in self.kappa_penalty_derivatives(rho)? {
            let decoder = self.atoms[atom_idx].decoder_coefficients();
            let energy = (decoder * &ds.dot(decoder)).sum();
            out.push((flat, 0.5 * lambda_smooth[atom_idx] * energy));
        }
        Ok(out)
    }

    /// #2935 — `tr((A⁻¹)_ββ·M_κ)`, `M_κ = (λ_k·½(∂S_k/∂κ + ∂S_k/∂κᵀ)) ⊗ I_{r_k}` on
    /// atom `k`'s β-block, for every curvature coordinate, off the same shared
    /// probe bundle as [`Self::decoder_smoothness_effective_dof_per_atom_from_probes`]
    /// so the κ log-determinant channel differentiates the value that bundle priced.
    pub(crate) fn decoder_kappa_penalty_trace_from_probes(
        &self,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        rho: &SaeManifoldRho,
        lambda_smooth: &[f64],
    ) -> Result<Vec<(usize, f64)>, String> {
        let (offsets, ranks) = self.decoder_border_blocks();
        let mut out = Vec::with_capacity(rho.kappa_atoms.len());
        for (flat, atom_idx, ds) in self.kappa_penalty_derivatives(rho)? {
            let off = offsets[atom_idx];
            let r = ranks[atom_idx];
            let lambda = lambda_smooth[atom_idx];
            let m_apply =
                |v: ArrayView1<'_, f64>| Self::decoder_penalty_block_apply(v, off, r, lambda, ds);
            let trace = hutchinson_reduced_schur_inverse_trace(probes, sinv_probes, &m_apply)
                .ok_or_else(|| {
                    format!(
                        "decoder_kappa_penalty_trace_from_probes: non-finite Hutchinson trace \
                         for curvature atom {atom_idx}"
                    )
                })?;
            out.push((flat, trace));
        }
        Ok(out)
    }
}
