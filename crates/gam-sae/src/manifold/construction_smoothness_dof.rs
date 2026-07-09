// [#780 line-count gate] Massive-K decoder-smoothness effective-dof estimator,
// split out of `construction.rs` (which sits against the 10k-line gate). This is
// the `SaeManifoldTerm` associated constants + the matrix-free Hutchinson
// stochastic-trace estimator that replaces the exact `Σ_k M_k·r_k`-solve per-atom
// effective dof `tr((H⁻¹)_ββ·M_k)` in the massive-dictionary (K up to 32k)
// regime. Included via `include!` from `construction.rs` so it keeps the SAME
// module scope (`use super::*`), the same `impl SaeManifoldTerm` surface, and
// full private-field access; the two gated exact/estimator entry points
// (`decoder_smoothness_effective_dof_per_atom` and the `_with_solver_` variant)
// stay in `construction.rs` and dispatch here at
// `K >= SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS`.

impl SaeManifoldTerm {
    /// Atom-count threshold above which the per-atom decoder-smoothness effective
    /// dof `tr((H⁻¹)_ββ·M_k)` switches from the exact column-by-column solve
    /// (`Σ_k M_k·r_k` back-substitutions — `O(K·M·p)` solves, the `O(K³·M·p)`
    /// massive-`K` wall) to the matrix-free Hutchinson stochastic-trace estimator
    /// [`Self::decoder_smoothness_effective_dof_per_atom_hutchinson`]. Chosen well
    /// above every exact-path test fixture so ordinary-`K` behaviour — and its
    /// bit-for-bit tests — is unchanged; the estimator engages only in the massive
    /// dictionary regime (`K` up to 32k) where the exact `K·M·p`-solve trace is
    /// computationally infeasible.
    pub(crate) const SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS: usize = 2048;
    /// Rademacher probe count for the Hutchinson smoothness-dof estimator. One
    /// `(H⁻¹)_ββ` solve per probe yields ALL per-atom traces at once, so this is
    /// the total solve count that replaces the exact `Σ_k M_k·r_k`.
    pub(crate) const SMOOTHNESS_DOF_HUTCHINSON_PROBES: usize = 64;
    /// Fixed base seed so the estimator is bit-reproducible across REML outer
    /// iterations (the outer loop must be deterministic; cf. the SLQ log-det seed).
    pub(crate) const SMOOTHNESS_DOF_HUTCHINSON_SEED: u64 = 0x5AED_0F5E_ED12_3456;

    /// Matrix-free Hutchinson estimator of the per-atom decoder-smoothness
    /// effective dof `edof_k = tr((H⁻¹)_ββ · M_k)`, `M_k = (λ_k·½(S_k+S_kᵀ)) ⊗
    /// I_{r_k}`. The massive-`K` replacement for the exact `Σ_k M_k·r_k` column
    /// solves in [`SaeManifoldTerm::decoder_smoothness_effective_dof_per_atom`] /
    /// [`SaeManifoldTerm::decoder_smoothness_effective_dof_with_solver_per_atom`].
    ///
    /// # Estimator (one solve → every atom's edf)
    ///
    /// `M = ⊕_k M_k` is block-diagonal over the β layout, so for a Rademacher
    /// probe `z` (`E[z zᵀ] = I`) and `u = (H⁻¹)_ββ (M z)`, the atom-`k` block dot
    /// `z_kᵀ u_k = Σ_{k'} z_kᵀ (H⁻¹)_{k k'} M_{k'} z_{k'}` has expectation exactly
    /// `tr((H⁻¹)_{kk} M_k) = edof_k` — the cross-atom `k'≠k` terms are mean-zero
    /// under `E[z_{k'} z_kᵀ] = 0`. So a SINGLE `(H⁻¹)_ββ` solve per probe gives an
    /// unbiased estimate of EVERY atom's edf simultaneously: `P` solves total
    /// instead of `Σ_k M_k·r_k`. Applying `M z` is a per-atom `O(M²·r)` GEMV (no
    /// solve). `P` solves × `O(k²)` (or the matrix-free apply cost) collapses the
    /// `O(K³·M·p)` wall to `O(P·K²)`.
    ///
    /// `solve_beta(rhs)` applies `(H⁻¹)_ββ` to a β-space rhs and returns the
    /// β-space solution — [`ArrowFactorCache::schur_inverse_apply`] on the dense
    /// factor path, or the matrix-free [`DeflatedArrowSolver`] β-solve. Probes run
    /// serially and accumulate in a fixed order, so for a fixed seed the estimate
    /// is bit-reproducible (the REML determinism contract).
    pub(crate) fn decoder_smoothness_effective_dof_per_atom_hutchinson(
        &self,
        border_dim: usize,
        offsets: &[usize],
        out_dim: &dyn Fn(usize) -> usize,
        lambda_smooth: &[f64],
        num_probes: usize,
        seed: u64,
        mut solve_beta: impl FnMut(ArrayView1<'_, f64>) -> Result<Array1<f64>, String>,
    ) -> Result<Vec<f64>, String> {
        let n_atoms = self.atoms.len();
        let probes = num_probes.max(1);
        let mut per_atom = vec![0.0_f64; n_atoms];
        let mut z = Array1::<f64>::zeros(border_dim);
        let mut mz = Array1::<f64>::zeros(border_dim);
        for probe in 0..probes {
            // Deterministic Rademacher probe (±1), seeded by `seed + probe`, so the
            // whole estimate is reproducible for a fixed `(seed, num_probes)`.
            let mut state = seed.wrapping_add(probe as u64);
            let mut bits = 0u64;
            let mut remaining = 0u32;
            for zi in z.iter_mut() {
                if remaining == 0 {
                    bits = gam_linalg::utils::splitmix64(&mut state);
                    remaining = 64;
                }
                *zi = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            // M z: block-diagonal `M = ⊕_k (λ_k·½(S_k+S_kᵀ)) ⊗ I_{r_k}`, matching
            // the exact path's `M[:,col]` column construction row-for-row.
            mz.fill(0.0);
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let s = &atom.smooth_penalty;
                let m = atom.basis_size();
                let off = offsets[atom_idx];
                let r = out_dim(atom_idx);
                let lambda = lambda_smooth[atom_idx];
                for nu in 0..m {
                    for oc in 0..r {
                        let mut acc = 0.0_f64;
                        for mu in 0..m {
                            let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                            acc += lambda * s_nu_mu * z[off + mu * r + oc];
                        }
                        mz[off + nu * r + oc] = acc;
                    }
                }
            }
            // One `(H⁻¹)_ββ` solve → every atom's block dot `z_kᵀ u_k`.
            let u = solve_beta(mz.view())?;
            if u.len() != border_dim {
                return Err(format!(
                    "decoder_smoothness_effective_dof_per_atom_hutchinson: solve returned \
                     length {} != border dim {border_dim}",
                    u.len()
                ));
            }
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let off = offsets[atom_idx];
                let blk = atom.basis_size() * out_dim(atom_idx);
                let mut dot = 0.0_f64;
                for i in off..off + blk {
                    dot += z[i] * u[i];
                }
                per_atom[atom_idx] += dot;
            }
        }
        let inv_p = 1.0 / (probes as f64);
        for v in per_atom.iter_mut() {
            *v *= inv_p;
        }
        Ok(per_atom)
    }

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
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            let lambda = lambda_smooth[atom_idx];
            // M_k·v: block-diagonal `(λ_k·½(S_k+S_kᵀ)) ⊗ I_{r_k}` restricted to
            // atom `k`'s β-block, matching the exact path's `M[:,col]` column
            // construction row-for-row.
            let m_apply = |v: ArrayView1<'_, f64>| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(v.len());
                for nu in 0..m {
                    for oc in 0..r {
                        let mut acc = 0.0_f64;
                        for mu in 0..m {
                            let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                            acc += lambda * s_nu_mu * v[off + mu * r + oc];
                        }
                        out[off + nu * r + oc] = acc;
                    }
                }
                out
            };
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
}
