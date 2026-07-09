//! Out-of-sample assignment-logit seeding for the frozen-decoder OOS solve.
//!
//! These two seeders place each held-out row in the correct atom basin BEFORE
//! the fixed-decoder Newton refinement runs, using only the frozen decoder's
//! per-atom decoded images `Φ_k(t)·B_k` at each row. They are the single source
//! of truth for OOS logit seeding: the CLI, a Rust library caller, and the
//! Python OOS entry all reach them through [`SaeManifoldTerm`]. Moved out of
//! `gam-pyffi` (which must contain only marshalling) per issue #2236 — the
//! math is orchestration, not a binding concern.

use ndarray::{Array2, ArrayView2};

use super::term::SaeManifoldTerm;

impl SaeManifoldTerm {
    /// Seed the softmax assignment logits from the per-atom reconstruction
    /// residual of each held-out row against the frozen decoder.
    ///
    /// Each row's logit for atom `k` is `-‖x − Φ_k(t)·B_k‖² / τ` evaluated at
    /// the row's currently-seeded coordinate (so a better-fitting atom scores
    /// higher). Logits are shifted per row by the last atom's value to fix the
    /// softmax gauge. Puts every row in the decisive basin before the
    /// fixed-decoder Newton refinement re-optimizes the coordinates and logits
    /// jointly.
    pub fn seed_oos_softmax_logits_from_projection_residuals(
        &mut self,
        target: ArrayView2<'_, f64>,
        tau: f64,
    ) {
        let (n_obs, p_out) = target.dim();
        let k_atoms = self.k_atoms();
        let mut seeded_logits = Array2::<f64>::zeros((n_obs, k_atoms));
        let mut decoded = vec![0.0_f64; p_out];
        for row in 0..n_obs {
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut decoded);
                let mut err = 0.0_f64;
                for out_col in 0..p_out {
                    let diff = target[[row, out_col]] - decoded[out_col];
                    err += diff * diff;
                }
                seeded_logits[[row, atom_idx]] = -err / tau;
            }
            let reference = seeded_logits[[row, k_atoms - 1]];
            for atom_idx in 0..k_atoms {
                seeded_logits[[row, atom_idx]] -= reference;
            }
        }
        self.assignment.logits.assign(&seeded_logits);
    }

    /// Seed the IBP-MAP assignment logits from a truncated stick-breaking
    /// box-constrained least-squares decode of each held-out row.
    ///
    /// For every row a coordinate-descent LSQ with the per-atom gates clamped to
    /// the truncated IBP stick-breaking prior mean `π_k = (α/(α+1))^{k+1}`
    /// (#614) recovers each atom's presence gate; the seeded logit is the
    /// inverse-sigmoid of the gate relative to its prior, scaled by `τ`, so the
    /// subsequent gate resolution starts at the prior-consistent occupancy.
    pub fn seed_oos_ibp_logits_from_projected_decoder_lsq(
        &mut self,
        target: ArrayView2<'_, f64>,
        tau: f64,
        alpha: f64,
    ) {
        let (n_obs, p_out) = target.dim();
        let k_atoms = self.k_atoms();
        // Consistent truncated IBP stick-breaking prior mean π_k = (α/(α+1))^(k+1)
        // (#614): the first atom is also shrunk by one Beta(α,1) stick mean, matching
        // the closed-form `ordered_geometric_shrinkage_prior` the fitter applies.
        let ratio = alpha / (alpha + 1.0);
        let mut prior = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            prior.push(ratio.powi(atom_idx as i32 + 1).max(f64::MIN_POSITIVE));
        }
        let mut decoded = vec![vec![0.0_f64; p_out]; k_atoms];
        let mut norm_sq = vec![0.0_f64; k_atoms];
        let mut gates = vec![0.0_f64; k_atoms];
        let mut fitted = vec![0.0_f64; p_out];
        let mut seeded_logits = Array2::<f64>::zeros((n_obs, k_atoms));
        const OOS_IBP_BOX_LSQ_SWEEPS: usize = 12;
        const OOS_IBP_GATE_EPS: f64 = 1.0e-6;
        for row in 0..n_obs {
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut decoded[atom_idx]);
                norm_sq[atom_idx] = decoded[atom_idx]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .max(1.0e-12);
                gates[atom_idx] = 0.0;
            }
            fitted.fill(0.0);
            for _ in 0..OOS_IBP_BOX_LSQ_SWEEPS {
                for atom_idx in 0..k_atoms {
                    let old_gate = gates[atom_idx];
                    let g_row = &decoded[atom_idx];
                    let mut numerator = 0.0_f64;
                    for out_col in 0..p_out {
                        let residual_without_atom =
                            target[[row, out_col]] - fitted[out_col] + old_gate * g_row[out_col];
                        numerator += g_row[out_col] * residual_without_atom;
                    }
                    let upper = prior[atom_idx] * (1.0 - OOS_IBP_GATE_EPS);
                    let new_gate = (numerator / norm_sq[atom_idx]).clamp(0.0, upper);
                    if new_gate != old_gate {
                        let delta = new_gate - old_gate;
                        for out_col in 0..p_out {
                            fitted[out_col] += delta * g_row[out_col];
                        }
                        gates[atom_idx] = new_gate;
                    }
                }
            }
            for atom_idx in 0..k_atoms {
                let q = (gates[atom_idx] / prior[atom_idx])
                    .clamp(OOS_IBP_GATE_EPS, 1.0 - OOS_IBP_GATE_EPS);
                seeded_logits[[row, atom_idx]] = tau * (q / (1.0 - q)).ln();
            }
        }
        self.assignment.logits.assign(&seeded_logits);
    }
}
