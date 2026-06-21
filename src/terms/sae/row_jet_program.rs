//! The SAE reconstruction row as a single Taylor-jet program (issue #932).
//!
//! # The row program
//!
//! The exact-LAML SAE engine needs, per row, the derivative tower of the
//! reconstruction
//!
//! ```text
//!   ẑ_row,c(p) = Σ_k ζ_k(ℓ) · decoded_{k,c}(t_k),   decoded_{k,c}(t) = Σ_b Φ_b(t)·B_{b,c}
//! ```
//!
//! — a **gate nonlinearity** `ζ(ℓ)` (softmax / IBP sigmoid) composed with a
//! **basis** `Φ(t)` composed with a **linear decoder** `B`, in the per-row
//! primary coordinates `p = (gate logits ℓ, latent coordinates t)`. Today the
//! arrow-Schur assembly (`SaeManifoldTerm::row_jets_for_logdet`) hand-packs the
//! `first`/`second` channels of this reconstruction from separate gate
//! derivative arrays (`gate_derivatives_for_row`) and basis jet tensors —
//! exactly the kind of hand-maintained cross-block tower whose sign flips are
//! the #736 / desync bug genus. The #1006 third-order logdet adjoint
//! `Γ_a = tr(H⁻¹ ∂H/∂θ_a)` is the consumer of those very channels.
//!
//! This module writes that reconstruction **once** over the
//! [`Tower4<K>`](crate::families::jet_tower::Tower4) scalar so the
//! value/gradient/Hessian/third channels of one row come from ONE jet
//! evaluation. [`SaeReconstructionRowProgram`] is generic over the gate kind
//! and the per-row basis jets; the gate, basis and decoder compose with plain
//! `Tower4` arithmetic, so there is no separate "channel" to forget.
//!
//! # The basis as a local jet
//!
//! The production assembly does NOT re-evaluate the manifold basis `Φ` as a
//! function of perturbed coordinates: it consumes the precomputed jet tensors
//! `(Φ, ∂Φ/∂t, ∂²Φ/∂t²)` evaluated at the current `t`. The reconstruction's
//! dependence on `t` is therefore *defined* by those tensors — the local
//! quadratic Taylor model of `Φ` about the current point. This program builds
//! each basis function as exactly that `Tower4` quadratic from the stored jets,
//! so the value/first/second channels it emits are the same object the hand
//! path packs — derived by independent arithmetic (tower Leibniz / Faà di
//! Bruno vs hand-summed cross terms). Agreement across both is a true
//! correctness proof of the hand kernel; disagreement names a dropped or
//! sign-flipped cross block loudly. That oracle is the riding test below.

use crate::families::jet_tower::Tower4;

/// Sentinel in [`SaeReconstructionRowProgram::coord_slot`] for an atom
/// coordinate that is fixed in this row's local chart (compact active-set rows
/// omit inactive atom coordinates, but softmax logit derivatives can still see
/// that atom's decoded value as a constant).
pub const SAE_FIXED_COORD_SLOT: usize = usize::MAX;

/// The gate nonlinearity `ζ(ℓ)` of the SAE assignment, as the row program sees
/// it. The production term carries the same two smooth branches (softmax over a
/// shared partition; per-atom IBP/JumpReLU sigmoid); the program reproduces the
/// branch the criterion evaluates so the value channel is the production gate.
#[derive(Debug, Clone, Copy)]
pub enum RowGate {
    /// Shared softmax over all atom logits with inverse temperature `inv_tau`.
    /// `ζ_k(ℓ) = softmax_k(ℓ · inv_tau)`.
    Softmax { inv_tau: f64 },
    /// Per-atom independent logistic gate `ζ_k(ℓ_k) = σ((ℓ_k − shift_k)·inv_tau)`
    /// — the IBP-MAP / JumpReLU smooth activation (the per-atom `shift_k`
    /// folds the IBP stick-breaking offset or the JumpReLU threshold). Each
    /// gate depends only on its own logit, so the gate Hessian is diagonal.
    PerAtomLogistic { inv_tau: f64 },
}

/// One atom's local basis jet at the current row: the stored
/// `(value, jacobian, second)` jet tensors of `Φ` plus the decoder block `B`.
/// Indexed `[basis_col]`, `[basis_col][axis]`, `[basis_col][axis_a][axis_b]`,
/// and `[basis_col][out_col]`.
#[derive(Debug, Clone)]
pub struct AtomRowBasisJet {
    /// `Φ_b` at the current coordinate (length `n_basis`).
    pub phi: Vec<f64>,
    /// `∂Φ_b/∂t_axis` (`[n_basis][latent_dim]`).
    pub d_phi: Vec<Vec<f64>>,
    /// `∂²Φ_b/∂t_a∂t_b` (`[n_basis][latent_dim][latent_dim]`).
    pub d2_phi: Vec<Vec<Vec<f64>>>,
    /// Decoder block `B_{b,c}` (`[n_basis][out_dim]`).
    pub decoder: Vec<Vec<f64>>,
    /// Latent dimension of this atom.
    pub latent_dim: usize,
}

impl AtomRowBasisJet {
    fn n_basis(&self) -> usize {
        self.phi.len()
    }

    fn out_dim(&self) -> usize {
        self.decoder.first().map_or(0, Vec::len)
    }

    /// `Φ_b(t)` as a `Tower4<K>` quadratic in the latent primaries occupying
    /// `coord_slots[axis]` (the seeded tower variable index for latent axis
    /// `axis` of this atom). A constant value plus first/second jet
    /// contributions — exactly the local Taylor model the production assembly
    /// consumes.
    fn basis_tower<const K: usize>(&self, basis_col: usize, coord_slots: &[usize]) -> Tower4<K> {
        // The latent coordinate increments enter as the seeded tower variables;
        // the basis value at the current point is the constant term.
        let mut acc = Tower4::<K>::constant(self.phi[basis_col]);
        for axis in 0..self.latent_dim {
            let slot = coord_slots[axis];
            let d1 = self.d_phi[basis_col][axis];
            if d1 != 0.0 {
                if slot != SAE_FIXED_COORD_SLOT {
                    acc = acc + Tower4::<K>::variable(0.0, slot).scale(d1);
                }
            }
        }
        // ½ Σ_ab d²Φ · δ_a δ_b, the quadratic term of the local Taylor model.
        for axis_a in 0..self.latent_dim {
            for axis_b in 0..self.latent_dim {
                let d2 = self.d2_phi[basis_col][axis_a][axis_b];
                if d2 == 0.0 {
                    continue;
                }
                if coord_slots[axis_a] == SAE_FIXED_COORD_SLOT
                    || coord_slots[axis_b] == SAE_FIXED_COORD_SLOT
                {
                    continue;
                }
                let va = Tower4::<K>::variable(0.0, coord_slots[axis_a]);
                let vb = Tower4::<K>::variable(0.0, coord_slots[axis_b]);
                acc = acc + va.mul(&vb).scale(0.5 * d2);
            }
        }
        acc
    }

    /// `decoded_{k,c}(t)` as a tower: `Σ_b Φ_b(t)·B_{b,c}`.
    fn decoded_tower<const K: usize>(&self, out_col: usize, coord_slots: &[usize]) -> Tower4<K> {
        let mut acc = Tower4::<K>::zero();
        for basis_col in 0..self.n_basis() {
            let b = self.decoder[basis_col][out_col];
            if b == 0.0 {
                continue;
            }
            acc = acc + self.basis_tower::<K>(basis_col, coord_slots).scale(b);
        }
        acc
    }
}

/// One row of the SAE reconstruction as a jet program: the per-atom basis jets,
/// the gate, the current gate-logit values, and the primary layout that maps
/// `(atom logit, atom latent axis)` to a seeded tower variable slot.
#[derive(Debug, Clone)]
pub struct SaeReconstructionRowProgram {
    /// Per-atom basis jets at the current row.
    pub atoms: Vec<AtomRowBasisJet>,
    /// Current gate activations `ζ_k` at the row (softmax/sigmoid values).
    pub gate_value: Vec<f64>,
    /// Current gate logits `ℓ_k` at the row.
    pub logits: Vec<f64>,
    /// Per-atom multiplicative scale for independent logistic gates. This is
    /// the IBP stick-breaking prior `π_k` for IBP-MAP, `1` for active JumpReLU,
    /// and `0` for JumpReLU rows at/below the hard threshold. Unused for
    /// softmax.
    pub gate_scale: Vec<f64>,
    /// Per-atom logistic shift (IBP offset / JumpReLU threshold); unused for
    /// softmax.
    pub gate_shift: Vec<f64>,
    /// The gate nonlinearity.
    pub gate: RowGate,
    /// Tower slot of atom `k`'s gate logit primary, or `None` if the gate logit
    /// is not a free primary for this atom (softmax `K==1`).
    pub logit_slot: Vec<Option<usize>>,
    /// Tower slot of atom `k`'s latent axis `j` primary (`coord_slot[k][j]`).
    pub coord_slot: Vec<Vec<usize>>,
    /// Total number of seeded primaries (= `K` of the tower).
    pub n_primaries: usize,
}

impl SaeReconstructionRowProgram {
    /// The gate activation `ζ_k(ℓ)` as a `Tower4<K>` in the gate-logit
    /// primaries. Softmax is the shared composition `exp(ℓ_k·inv_tau) /
    /// Σ_j exp(ℓ_j·inv_tau)`; the per-atom logistic is `σ((ℓ_k − shift_k)·
    /// inv_tau)` depending only on its own logit. Both carry every derivative
    /// channel automatically.
    fn gate_tower<const K: usize>(&self, atom: usize) -> Tower4<K> {
        match self.gate {
            RowGate::Softmax { inv_tau } => {
                // Build exp(ℓ_j·inv_tau − shift) for every atom that has a free
                // logit primary, as a tower; atoms without a free logit
                // contribute a constant exponential (their logit does not move).
                //
                // Stability: softmax is invariant to a common additive constant
                // in every exponent (`exp(a−s)/Σ exp(b−s) = exp(a)/Σ exp(b)`),
                // and the higher derivative channels are unchanged because the
                // shift is a numeric constant (a function of the base logit
                // *values* only, seeded as a `constant`, not of the tower
                // variables). We subtract the largest base exponent
                // `max_j ℓ_j·inv_tau` so the dominant `exp(·)` is `exp(0)=1` and
                // no term overflows. This mirrors the max-subtraction in the
                // production `softmax_row`.
                let shift = self
                    .logits
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    * inv_tau;
                let mut denom = Tower4::<K>::zero();
                let mut numer = Tower4::<K>::zero();
                for j in 0..self.gate_value.len() {
                    let lj = match self.logit_slot[j] {
                        Some(slot) => Tower4::<K>::variable(self.logits[j], slot),
                        None => Tower4::<K>::constant(self.logits[j]),
                    };
                    // (ℓ_j·inv_tau − shift): subtracting a constant shifts only
                    // the value channel, leaving every gradient/Hessian/t3/t4
                    // channel of the exponent (hence of exp via the chain rule)
                    // identical to the unshifted form.
                    let ej = (lj.scale(inv_tau) - shift).exp();
                    if j == atom {
                        numer = ej;
                    }
                    denom = denom + ej;
                }
                numer / denom
            }
            RowGate::PerAtomLogistic { inv_tau } => {
                let l = match self.logit_slot[atom] {
                    Some(slot) => Tower4::<K>::variable(self.logits[atom], slot),
                    None => Tower4::<K>::constant(self.logits[atom]),
                };
                // σ(x) = 1 / (1 + exp(−x)). Evaluated in a branch-stable form to
                // avoid overflow of the inner `exp`: the two algebraic identities
                //   x ≥ 0:  σ(x) = 1 / (1 + exp(−x))      (exp(−x) ∈ (0,1])
                //   x < 0:  σ(x) = exp(x) / (1 + exp(x))   (exp(x)  ∈ (0,1))
                // are equal everywhere, so they produce the identical tower in
                // every derivative channel. The branch is selected on the base
                // .v of x (a constant for a given row), so it is not a function
                // of the tower variables and the derivative stack is unchanged.
                let x = (l - self.gate_shift[atom]).scale(inv_tau);
                let one = Tower4::<K>::constant(1.0);
                let sigma = if x.v >= 0.0 {
                    one.clone() / (one + x.scale(-1.0).exp())
                } else {
                    let ex = x.exp();
                    ex.clone() / (one + ex)
                };
                sigma.scale(self.gate_scale[atom])
            }
        }
    }

    /// The reconstruction output column `c` as a single jet:
    /// `ẑ_c(p) = Σ_k ζ_k(ℓ) · decoded_{k,c}(t_k)`. Its `.v` is the production
    /// reconstruction value, `.g[a]` is `∂ẑ_c/∂p_a`, `.h[a][b]` is
    /// `∂²ẑ_c/∂p_a∂p_b`, and the `t3`/`t4` channels are the exact higher-order
    /// derivatives — all from this ONE evaluation.
    #[must_use]
    pub fn reconstruction_column<const K: usize>(&self, out_col: usize) -> Tower4<K> {
        assert_eq!(
            self.n_primaries, K,
            "SaeReconstructionRowProgram: tower arity K={K} must equal n_primaries={}",
            self.n_primaries
        );
        let mut acc = Tower4::<K>::zero();
        for (atom, atom_jet) in self.atoms.iter().enumerate() {
            let gate = self.gate_tower::<K>(atom);
            let decoded = atom_jet.decoded_tower::<K>(out_col, &self.coord_slot[atom]);
            acc = acc + gate.mul(&decoded);
        }
        acc
    }

    /// The number of reconstruction output columns.
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.atoms.first().map_or(0, AtomRowBasisJet::out_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replicate the production hand path (`row_jets_for_logdet`) arithmetic for
    /// the reconstruction `first`/`second` channels of ONE output column, from
    /// the same atom jets and softmax gate derivatives — independent code from
    /// the tower. The two must agree to machine precision; this is the #932
    /// universal oracle for the SAE row program (the analog of the survival
    /// `rigid_row_kernel_agrees_with_jet_tower_program` oracle).
    struct HandChannels {
        first: Vec<f64>,       // [primary]
        second: Vec<Vec<f64>>, // [primary][primary]
        value: f64,
    }

    /// Softmax gate first/second derivatives wrt logit primaries, term-for-term
    /// the production `gate_derivatives_for_row` softmax branch.
    fn softmax_gate_derivs(gate: &[f64], inv_tau: f64) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let k = gate.len();
        // dz[j][kk] = ∂ζ_kk/∂ℓ_j ; d2z[j][l][kk] = ∂²ζ_kk/∂ℓ_j∂ℓ_l.
        let mut dz = vec![vec![0.0_f64; k]; k];
        let mut d2z = vec![vec![vec![0.0_f64; k]; k]; k];
        for j in 0..k {
            for kk in 0..k {
                let ind = if kk == j { 1.0 } else { 0.0 };
                dz[j][kk] = gate[kk] * (ind - gate[j]) * inv_tau;
            }
        }
        for j in 0..k {
            for l in 0..k {
                for kk in 0..k {
                    let ikl = if kk == l { 1.0 } else { 0.0 };
                    let ikj = if kk == j { 1.0 } else { 0.0 };
                    let ijl = if j == l { 1.0 } else { 0.0 };
                    d2z[j][l][kk] = gate[kk]
                        * ((ikl - gate[l]) * (ikj - gate[j]) - gate[j] * (ijl - gate[l]))
                        * inv_tau
                        * inv_tau;
                }
            }
        }
        (dz, d2z)
    }

    /// Hand-pack the reconstruction column channels exactly as the production
    /// `row_jets_for_logdet` does for a softmax gate: gate-logit primaries first
    /// (one per atom), then each atom's latent coords.
    fn hand_softmax_column(
        prog: &SaeReconstructionRowProgram,
        out_col: usize,
        inv_tau: f64,
    ) -> HandChannels {
        let k = prog.atoms.len();
        let n = prog.n_primaries;
        // decoded[k] value, d1[k][axis], d2[k][a][b] for this out_col.
        let decoded: Vec<f64> = (0..k)
            .map(|kk| {
                (0..prog.atoms[kk].n_basis())
                    .map(|b| prog.atoms[kk].phi[b] * prog.atoms[kk].decoder[b][out_col])
                    .sum()
            })
            .collect();
        let d1: Vec<Vec<f64>> = (0..k)
            .map(|kk| {
                (0..prog.atoms[kk].latent_dim)
                    .map(|axis| {
                        (0..prog.atoms[kk].n_basis())
                            .map(|b| {
                                prog.atoms[kk].d_phi[b][axis] * prog.atoms[kk].decoder[b][out_col]
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let d2: Vec<Vec<Vec<f64>>> = (0..k)
            .map(|kk| {
                (0..prog.atoms[kk].latent_dim)
                    .map(|a| {
                        (0..prog.atoms[kk].latent_dim)
                            .map(|b| {
                                (0..prog.atoms[kk].n_basis())
                                    .map(|col| {
                                        prog.atoms[kk].d2_phi[col][a][b]
                                            * prog.atoms[kk].decoder[col][out_col]
                                    })
                                    .sum()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let (dz, d2z) = softmax_gate_derivs(&prog.gate_value, inv_tau);

        // Primary index of atom logit / coord, matching the program layout.
        let logit_idx = |kk: usize| prog.logit_slot[kk];
        let coord_idx = |kk: usize, axis: usize| prog.coord_slot[kk][axis];

        let value: f64 = (0..k).map(|kk| prog.gate_value[kk] * decoded[kk]).sum();

        let mut first = vec![0.0_f64; n];
        // Logit primaries: ∂ẑ/∂ℓ_j = Σ_kk dz[j][kk]·decoded[kk].
        for j in 0..k {
            if let Some(p) = logit_idx(j) {
                first[p] = (0..k).map(|kk| dz[j][kk] * decoded[kk]).sum();
            }
        }
        // Coord primaries: ∂ẑ/∂t_{kk,axis} = ζ_kk · d1[kk][axis].
        for kk in 0..k {
            for axis in 0..prog.atoms[kk].latent_dim {
                first[coord_idx(kk, axis)] = prog.gate_value[kk] * d1[kk][axis];
            }
        }

        let mut second = vec![vec![0.0_f64; n]; n];
        // Logit×Logit: Σ_kk d2z[j][l][kk]·decoded[kk].
        for j in 0..k {
            for l in 0..k {
                if let (Some(pj), Some(pl)) = (logit_idx(j), logit_idx(l)) {
                    second[pj][pl] = (0..k).map(|kk| d2z[j][l][kk] * decoded[kk]).sum();
                }
            }
        }
        // Logit×Coord (and symmetric): dz[j][kk]·d1[kk][axis].
        for j in 0..k {
            for kk in 0..k {
                for axis in 0..prog.atoms[kk].latent_dim {
                    if let Some(pj) = logit_idx(j) {
                        let pc = coord_idx(kk, axis);
                        let val = dz[j][kk] * d1[kk][axis];
                        second[pj][pc] = val;
                        second[pc][pj] = val;
                    }
                }
            }
        }
        // Coord×Coord same atom: ζ_kk · d2[kk][a][b].
        for kk in 0..k {
            for a in 0..prog.atoms[kk].latent_dim {
                for b in 0..prog.atoms[kk].latent_dim {
                    let pa = coord_idx(kk, a);
                    let pb = coord_idx(kk, b);
                    second[pa][pb] = prog.gate_value[kk] * d2[kk][a][b];
                }
            }
        }

        HandChannels {
            first,
            second,
            value,
        }
    }

    /// Build a two-atom softmax fixture with `latent_dim = 2` per atom and a
    /// dense decoder so every primary is exercised. Layout: logit slots
    /// 0,1; atom-0 coords 2,3; atom-1 coords 4,5 → K = 6 primaries.
    fn softmax_fixture(inv_tau: f64) -> (SaeReconstructionRowProgram, f64) {
        let n_basis = 3;
        let out_dim = 4;
        let mk_atom = |seed: f64| {
            let phi: Vec<f64> = (0..n_basis)
                .map(|b| 0.3 + 0.2 * (b as f64 + seed))
                .collect();
            let d_phi: Vec<Vec<f64>> = (0..n_basis)
                .map(|b| {
                    (0..2)
                        .map(|axis| 0.1 * (b as f64 + 1.0) - 0.05 * axis as f64 + 0.03 * seed)
                        .collect()
                })
                .collect();
            let d2_phi: Vec<Vec<Vec<f64>>> = (0..n_basis)
                .map(|b| {
                    (0..2)
                        .map(|a| {
                            (0..2)
                                .map(|bb| {
                                    // Symmetric in (a, bb).
                                    0.02 * (b as f64 + 1.0)
                                        + 0.01 * (a as f64)
                                        + 0.01 * (bb as f64)
                                        + 0.004 * seed
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let decoder: Vec<Vec<f64>> = (0..n_basis)
                .map(|b| {
                    (0..out_dim)
                        .map(|c| 0.5 - 0.1 * (b as f64) + 0.07 * (c as f64) + 0.02 * seed)
                        .collect()
                })
                .collect();
            AtomRowBasisJet {
                phi,
                d_phi,
                d2_phi,
                decoder,
                latent_dim: 2,
            }
        };
        let logits = vec![0.4_f64, -0.7];
        // Softmax gate values at these logits.
        let e: Vec<f64> = logits.iter().map(|&l| (l * inv_tau).exp()).collect();
        let s: f64 = e.iter().sum();
        let gate_value: Vec<f64> = e.iter().map(|&v| v / s).collect();
        let prog = SaeReconstructionRowProgram {
            atoms: vec![mk_atom(0.0), mk_atom(1.0)],
            gate_value,
            logits,
            gate_scale: vec![1.0, 1.0],
            gate_shift: vec![0.0, 0.0],
            gate: RowGate::Softmax { inv_tau },
            logit_slot: vec![Some(0), Some(1)],
            coord_slot: vec![vec![2, 3], vec![4, 5]],
            n_primaries: 6,
        };
        (prog, inv_tau)
    }

    /// INDEPENDENT scalar witness for the reconstruction column `ẑ_c(δ)` as a
    /// function of the primary-increment vector `δ` (the displacement of each
    /// tower primary from its seed value: a coord primary seeds at value 0, a
    /// logit primary at its current logit, so `δ` is the same offset the tower's
    /// seeded variables carry). This evaluator touches NONE of the `Tower4`
    /// arithmetic — no Leibniz product, no Faà di Bruno compose, no
    /// `basis_tower`/`decoded_tower`/`gate_tower` — it re-derives the closed-form
    /// reconstruction from the raw jet tensors and the softmax definition. It is
    /// the witness the t3/t4 FD oracle differences below.
    ///
    /// `ẑ_c(δ) = Σ_k softmax_k((ℓ + δ_logit)·inv_tau) · Σ_b Φ̃_{k,b}(δ_coord)·B_{k,b,c}`
    /// with the SAME local quadratic basis model the program consumes:
    /// `Φ̃_b(u) = phi[b] + Σ_a d_phi[b][a]·u_a + ½ Σ_{a,a'} d2_phi[b][a][a']·u_a·u_{a'}`.
    fn recon_scalar_softmax(
        prog: &SaeReconstructionRowProgram,
        out_col: usize,
        inv_tau: f64,
        delta: &[f64],
    ) -> f64 {
        let k = prog.atoms.len();
        // Softmax over (logit + δ_logit) for atoms with a free logit primary;
        // atoms without one keep their base logit (no δ).
        let exps: Vec<f64> = (0..k)
            .map(|kk| {
                let dl = match prog.logit_slot[kk] {
                    Some(slot) => delta[slot],
                    None => 0.0,
                };
                ((prog.logits[kk] + dl) * inv_tau).exp()
            })
            .collect();
        let denom: f64 = exps.iter().sum();
        let mut acc = 0.0;
        for kk in 0..k {
            let gate = exps[kk] / denom;
            let atom = &prog.atoms[kk];
            // decoded_{kk,c}(δ_coord) via the local quadratic basis model.
            let mut decoded = 0.0;
            for b in 0..atom.n_basis() {
                let mut phi = atom.phi[b];
                for a in 0..atom.latent_dim {
                    let ua = delta[prog.coord_slot[kk][a]];
                    phi += atom.d_phi[b][a] * ua;
                }
                for a in 0..atom.latent_dim {
                    let ua = delta[prog.coord_slot[kk][a]];
                    for a2 in 0..atom.latent_dim {
                        let ub = delta[prog.coord_slot[kk][a2]];
                        phi += 0.5 * atom.d2_phi[b][a][a2] * ua * ub;
                    }
                }
                decoded += phi * atom.decoder[b][out_col];
            }
            acc += gate * decoded;
        }
        acc
    }

    /// Fourth-order central FD of `recon_scalar_softmax` along axes (a,b,c,d) at
    /// the origin (δ = 0, the tower seed point). Uses the standard mixed
    /// fourth-difference stencil with sign vector ±h on each of the four axes
    /// (axes may coincide). 2⁴ = 16 evaluations.
    fn fd_fourth(
        prog: &SaeReconstructionRowProgram,
        out_col: usize,
        inv_tau: f64,
        axes: [usize; 4],
        h: f64,
    ) -> f64 {
        let n = prog.n_primaries;
        let mut acc = 0.0;
        for mask in 0..16u32 {
            let mut delta = vec![0.0_f64; n];
            let mut sign = 1.0;
            for (slot, &ax) in axes.iter().enumerate() {
                if (mask >> slot) & 1 == 1 {
                    delta[ax] += h;
                } else {
                    delta[ax] -= h;
                    sign = -sign;
                }
            }
            acc += sign * recon_scalar_softmax(prog, out_col, inv_tau, &delta);
        }
        acc / (16.0 * h * h * h * h)
    }

    /// Third-order central FD of `recon_scalar_softmax` along axes (a,b,c) at the
    /// origin: 2³ = 8 evaluations with the mixed third-difference stencil.
    fn fd_third(
        prog: &SaeReconstructionRowProgram,
        out_col: usize,
        inv_tau: f64,
        axes: [usize; 3],
        h: f64,
    ) -> f64 {
        let n = prog.n_primaries;
        let mut acc = 0.0;
        for mask in 0..8u32 {
            let mut delta = vec![0.0_f64; n];
            let mut sign = 1.0;
            for (slot, &ax) in axes.iter().enumerate() {
                if (mask >> slot) & 1 == 1 {
                    delta[ax] += h;
                } else {
                    delta[ax] -= h;
                    sign = -sign;
                }
            }
            acc += sign * recon_scalar_softmax(prog, out_col, inv_tau, &delta);
        }
        acc / (8.0 * h * h * h)
    }

    /// The #932 follow-up the issue flagged as missing: the SAE reconstruction
    /// program's THIRD- and FOURTH-order channels (`t3`/`t4`) validated against an
    /// INDEPENDENT witness (`recon_scalar_softmax`, finite-differenced), not just
    /// the value/first/second channels the hand-path oracle covers. Both the
    /// witness and the differencing are independent of the `Tower4` Leibniz /
    /// Faà-di-Bruno arithmetic that produces `t3`/`t4`, so agreement is a real
    /// cross-check of those higher-order channels — the analog of the survival
    /// kernel's `row_third_contracted` oracle, extended to fourth order.
    #[test]
    fn softmax_reconstruction_t3_t4_match_independent_fd_witness() {
        let (prog, inv_tau) = softmax_fixture(1.1);
        // Mixed fifth-derivative magnitude bounds the central-FD truncation; a
        // moderate step keeps both truncation and roundoff well under tol.
        let h3 = 2e-3;
        let h4 = 1e-2;
        for out_col in 0..prog.out_dim() {
            let tower = prog.reconstruction_column::<6>(out_col);

            let t3_floor = tower
                .t3
                .iter()
                .flatten()
                .flatten()
                .fold(0.0_f64, |m, x| m.max(x.abs()))
                .max(1e-9);
            let t4_floor = tower
                .t4
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .fold(0.0_f64, |m, x| m.max(x.abs()))
                .max(1e-9);

            for a in 0..6 {
                for b in 0..6 {
                    for c in 0..6 {
                        let fd = fd_third(&prog, out_col, inv_tau, [a, b, c], h3);
                        assert!(
                            (tower.t3[a][b][c] - fd).abs() <= 5e-5 * t3_floor,
                            "col {out_col} t3[{a}][{b}][{c}]: tower {:+.10e} vs fd {:+.10e}",
                            tower.t3[a][b][c],
                            fd
                        );
                        for d in 0..6 {
                            let fd4 = fd_fourth(&prog, out_col, inv_tau, [a, b, c, d], h4);
                            assert!(
                                (tower.t4[a][b][c][d] - fd4).abs() <= 5e-4 * t4_floor,
                                "col {out_col} t4[{a}][{b}][{c}][{d}]: tower {:+.10e} vs fd {:+.10e}",
                                tower.t4[a][b][c][d],
                                fd4
                            );
                        }
                    }
                }
            }
        }
    }

    /// A planted #736-style corruption in a t3 OR t4 channel is caught by the
    /// independent FD witness (loud at introduction). We perturb a copy of the
    /// tower's higher-order channel and assert the witness disagrees.
    #[test]
    fn planted_t3_t4_corruption_is_caught_by_fd_witness() {
        let (prog, inv_tau) = softmax_fixture(1.1);
        let out_col = 2;
        let tower = prog.reconstruction_column::<6>(out_col);
        // A real logit×coord×coord third block (atom-0 logit slot 0, atom-0
        // coords 2,3): the witness's third FD must match it...
        let axes3 = [0usize, 2, 3];
        let fd3 = fd_third(&prog, out_col, inv_tau, axes3, 2e-3);
        let t3_floor = tower
            .t3
            .iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
            .max(1e-9);
        assert!(
            (tower.t3[0][2][3] - fd3).abs() <= 5e-5 * t3_floor,
            "honest t3 must match witness"
        );
        // ...and a sign-flipped copy must NOT.
        let corrupt = -tower.t3[0][2][3];
        assert!(
            (corrupt - fd3).abs() > 5e-5 * t3_floor,
            "a sign-flipped t3 block must disagree with the FD witness"
        );

        let axes4 = [0usize, 0, 2, 3];
        let fd4 = fd_fourth(&prog, out_col, inv_tau, axes4, 1e-2);
        let t4_floor = tower
            .t4
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
            .max(1e-9);
        let corrupt4 = tower.t4[0][0][2][3] + 10.0 * t4_floor;
        assert!(
            (corrupt4 - fd4).abs() > 5e-4 * t4_floor,
            "a corrupted t4 block must disagree with the FD witness"
        );
    }

    #[test]
    fn softmax_reconstruction_tower_matches_hand_channels_all_columns() {
        let (prog, inv_tau) = softmax_fixture(1.3);
        for out_col in 0..prog.out_dim() {
            let tower = prog.reconstruction_column::<6>(out_col);
            let hand = hand_softmax_column(&prog, out_col, inv_tau);

            // Magnitude floors so structurally-zero entries don't demand
            // absolute equality (the verify_kernel_channels convention).
            let g_floor = tower.g.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let h_floor = tower
                .h
                .iter()
                .flatten()
                .fold(0.0_f64, |m, x| m.max(x.abs()));

            assert!(
                (tower.v - hand.value).abs() <= 1e-9 * hand.value.abs().max(1.0),
                "col {out_col} value: tower {} vs hand {}",
                tower.v,
                hand.value
            );
            for a in 0..6 {
                assert!(
                    (tower.g[a] - hand.first[a]).abs() <= 1e-9 * g_floor.max(1e-12),
                    "col {out_col} first[{a}]: tower {} vs hand {}",
                    tower.g[a],
                    hand.first[a]
                );
                for b in 0..6 {
                    assert!(
                        (tower.h[a][b] - hand.second[a][b]).abs() <= 1e-8 * h_floor.max(1e-12),
                        "col {out_col} second[{a}][{b}]: tower {} vs hand {}",
                        tower.h[a][b],
                        hand.second[a][b]
                    );
                }
            }
        }
    }

    /// A planted sign flip in the hand cross-block (logit×coord) is caught by the
    /// oracle — the same failure that #736 was, made loud at introduction.
    #[test]
    fn planted_cross_block_sign_flip_is_caught() {
        let (prog, inv_tau) = softmax_fixture(1.3);
        let out_col = 1;
        let tower = prog.reconstruction_column::<6>(out_col);
        let mut hand = hand_softmax_column(&prog, out_col, inv_tau);
        // Corrupt one logit×coord cross block (atom-0 logit slot 0, atom-1
        // coord slot 4): flip its sign, the #736 disease.
        hand.second[0][4] = -hand.second[0][4];
        hand.second[4][0] = -hand.second[4][0];
        let h_floor = tower
            .h
            .iter()
            .flatten()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let disagrees = (tower.h[0][4] - hand.second[0][4]).abs() > 1e-8 * h_floor.max(1e-12);
        assert!(
            disagrees,
            "a flipped cross block must disagree with the tower truth"
        );
    }

    /// The tower gate channels alone reproduce the softmax `gate_derivatives_for_row`
    /// arithmetic — isolating the gate nonlinearity from the basis/decoder so a
    /// regression in either is localizable.
    #[test]
    fn softmax_gate_tower_matches_hand_gate_derivatives() {
        let (prog, inv_tau) = softmax_fixture(0.9);
        let (dz, d2z) = softmax_gate_derivs(&prog.gate_value, inv_tau);
        for atom in 0..prog.atoms.len() {
            let gate = prog.gate_tower::<6>(atom);
            // ζ_atom value.
            assert!((gate.v - prog.gate_value[atom]).abs() < 1e-12);
            // ∂ζ_atom/∂ℓ_j == dz[j][atom].
            for j in 0..prog.atoms.len() {
                let slot = prog.logit_slot[j].unwrap();
                assert!(
                    (gate.g[slot] - dz[j][atom]).abs() < 1e-9,
                    "gate {atom} d/dlogit {j}: tower {} vs hand {}",
                    gate.g[slot],
                    dz[j][atom]
                );
            }
            // ∂²ζ_atom/∂ℓ_j∂ℓ_l == d2z[j][l][atom].
            for j in 0..prog.atoms.len() {
                for l in 0..prog.atoms.len() {
                    let sj = prog.logit_slot[j].unwrap();
                    let sl = prog.logit_slot[l].unwrap();
                    assert!(
                        (gate.h[sj][sl] - d2z[j][l][atom]).abs() < 1e-8,
                        "gate {atom} d2/dlogit {j}{l}: tower {} vs hand {}",
                        gate.h[sj][sl],
                        d2z[j][l][atom]
                    );
                }
            }
        }
    }

    /// The per-atom logistic gate (IBP/JumpReLU branch) is diagonal in the
    /// logits and reproduces `σ' = σ(1−σ)·inv_tau`, `σ'' = σ(1−σ)(1−2σ)·inv_tau²`.
    #[test]
    fn per_atom_logistic_gate_matches_closed_form() {
        let inv_tau = 1.4;
        let logit = 0.6;
        let shift = 0.2;
        let x: f64 = (logit - shift) * inv_tau;
        let sigma = 1.0 / (1.0 + (-x).exp());
        let prog = SaeReconstructionRowProgram {
            atoms: vec![AtomRowBasisJet {
                phi: vec![1.0],
                d_phi: vec![vec![0.0]],
                d2_phi: vec![vec![vec![0.0]]],
                decoder: vec![vec![1.0]],
                latent_dim: 1,
            }],
            gate_value: vec![sigma],
            logits: vec![logit],
            gate_scale: vec![1.0],
            gate_shift: vec![shift],
            gate: RowGate::PerAtomLogistic { inv_tau },
            logit_slot: vec![Some(0)],
            coord_slot: vec![vec![1]],
            n_primaries: 2,
        };
        let gate = prog.gate_tower::<2>(0);
        assert!((gate.v - sigma).abs() < 1e-12);
        let d1 = sigma * (1.0 - sigma) * inv_tau;
        let d2 = sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma) * inv_tau * inv_tau;
        assert!((gate.g[0] - d1).abs() < 1e-9, "σ': {} vs {}", gate.g[0], d1);
        assert!(
            (gate.h[0][0] - d2).abs() < 1e-9,
            "σ'': {} vs {}",
            gate.h[0][0],
            d2
        );
    }
}
