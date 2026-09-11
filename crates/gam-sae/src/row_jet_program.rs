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
//! — a **gate nonlinearity** `ζ(ℓ)` (softmax / ordered Beta--Bernoulli sigmoid)
//! composed with a **basis** `Φ(t)` composed with a **linear decoder** `B`, in
//! the per-row primary coordinates `p = (gate logits ℓ, latent coordinates t)`.
//! Production derives the complete arrow-Schur `first`/`second` and decoder-
//! border channels from this semantic program. Softmax rows use the borrowed
//! `SaeOrder2RowProgramSource` and the gate-specific structure-compiled schedules;
//! its bounded batch seam evaluates the same centered-moment identities on CPU
//! or CUDA. Other gate graphs use [`SaeReconstructionRowProgram`] over the
//! runtime jet algebra. The #1006 third-order logdet adjoint
//! `Γ_a = tr(H⁻¹ ∂H/∂θ_a)` consumes those shared channels.
//!
//! [`SaeReconstructionRowProgram`] is generic over the gate kind and per-row
//! basis jets, so gate, basis, and decoder are still written once. Dense
//! [`Tower4<K>`](gam_math::jet_tower::Tower4) evaluation remains an independent
//! test oracle for the production order-≤2 lowerings and the higher derivative
//! witnesses; it is not the softmax hot path.
//!
//! # The basis as a local jet
//!
//! The production assembly does NOT re-evaluate the manifold basis `Φ` as a
//! function of perturbed coordinates: it consumes the precomputed jet tensors
//! `(Φ, ∂Φ/∂t, ∂²Φ/∂t²)` evaluated at the current `t`. The reconstruction's
//! dependence on `t` is therefore *defined* by those tensors — the local
//! quadratic Taylor model of `Φ` about the current point. The generic runtime
//! jet and dense `Tower4` oracle build exactly that quadratic; the compiled
//! softmax schedule lowers it through centered moments. Tests compare both
//! lowerings with a historical explicit cross-term reference, so a dropped or
//! sign-flipped block is named independently rather than shared silently.

use gam_math::jet_scalar::{FixedRuntimeJet, Order1, Order2, RuntimeJetScalar};
use gam_math::jet_tower::Tower4;

/// `1/self` for any [`gam_math::jet_scalar::JetScalar`] via Faà di Bruno on `f(u) = 1/u`
/// (stack `[1/u, -1/u², 2/u³, -6/u⁴, 24/u⁵]`). Caller guarantees `self.value()`
/// is nonzero — softmax denominators are strictly positive sums of exponentials.
#[inline]
fn recip<'arena, S: RuntimeJetScalar<'arena>>(s: &S) -> S {
    s.recip()
}

/// Sentinel in [`SaeReconstructionRowProgram::coord_slot`] for an atom
/// coordinate that is fixed in this row's local chart (compact active-set rows
/// omit inactive atom coordinates, but softmax logit derivatives can still see
/// that atom's decoded value as a constant).
pub const SAE_FIXED_COORD_SLOT: usize = usize::MAX;

/// The gate nonlinearity `ζ(ℓ)` of the SAE assignment, as the row program sees
/// it. The production term carries the same two smooth branches (softmax over a
/// shared partition; per-atom independent sigmoid); the program reproduces the
/// branch the criterion evaluates so the value channel is the production gate.
#[derive(Debug, Clone, Copy)]
pub enum RowGate {
    /// Shared softmax over all atom logits with inverse temperature `inv_tau`.
    /// `ζ_k(ℓ) = softmax_k(ℓ · inv_tau)`.
    Softmax { inv_tau: f64 },
    /// Per-atom independent logistic gate `ζ_k(ℓ_k) = σ((ℓ_k − shift_k)·inv_tau)`
    /// — the ordered Beta--Bernoulli / threshold-gate activation (the per-atom
    /// `shift_k` carries the threshold-gate center). Each
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
    fn basis_tower<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        basis_col: usize,
        coord_slots: &[usize],
        dimension: usize,
        workspace: &'arena S::Workspace,
    ) -> S {
        // The latent coordinate increments enter as the seeded tower variables;
        // the basis value at the current point is the constant term.
        let mut acc = S::constant(self.phi[basis_col], dimension, workspace);
        for axis in 0..self.latent_dim {
            let slot = coord_slots[axis];
            let d1 = self.d_phi[basis_col][axis];
            if d1 != 0.0 {
                if slot != SAE_FIXED_COORD_SLOT {
                    acc = acc.add(&S::variable(0.0, slot, dimension, workspace).scale(d1));
                }
            }
        }
        // ½ Σ_ab d²Φ · δ_a δ_b, the quadratic term of the local Taylor model.
        // Hoist the axis_a fixed-slot skip and `va` build out of the inner loop.
        for axis_a in 0..self.latent_dim {
            let slot_a = coord_slots[axis_a];
            if slot_a == SAE_FIXED_COORD_SLOT {
                continue;
            }
            let va = S::variable(0.0, slot_a, dimension, workspace);
            for axis_b in 0..self.latent_dim {
                let d2 = self.d2_phi[basis_col][axis_a][axis_b];
                if d2 == 0.0 {
                    continue;
                }
                let slot_b = coord_slots[axis_b];
                if slot_b == SAE_FIXED_COORD_SLOT {
                    continue;
                }
                let vb = S::variable(0.0, slot_b, dimension, workspace);
                acc = acc.add(&va.mul(&vb).scale(0.5 * d2));
            }
        }
        acc
    }

    /// `decoded_{k,c}(t)` as a tower: `Σ_b Φ_b(t)·B_{b,c}`.
    fn decoded_tower<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        out_col: usize,
        coord_slots: &[usize],
        dimension: usize,
        workspace: &'arena S::Workspace,
    ) -> S {
        let mut acc = S::constant(0.0, dimension, workspace);
        for basis_col in 0..self.n_basis() {
            let b = self.decoder[basis_col][out_col];
            if b == 0.0 {
                continue;
            }
            acc = acc.add(
                &self
                    .basis_tower::<S>(basis_col, coord_slots, dimension, workspace)
                    .scale(b),
            );
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
    /// Per-atom logistic shift (zero for ordered Beta--Bernoulli, the smooth
    /// threshold center for threshold-gate); unused for
    /// softmax.
    pub gate_shift: Vec<f64>,
    /// The gate nonlinearity.
    pub gate: RowGate,
    /// Tower slot of atom `k`'s gate logit primary, or `None` if the gate logit
    /// is not a free primary for this atom (softmax `K==1`).
    pub logit_slot: Vec<Option<usize>>,
    /// Tower slot of atom `k`'s latent axis `j` primary (`coord_slot[k][j]`).
    pub coord_slot: Vec<Vec<usize>>,
    /// Per-atom FIXED-gate override (#1026/#1033). `Some(value)` pins atom `k`'s
    /// gate `ζ_k` to a CONSTANT equal to `value` — the active-routing gate the
    /// value assembly used — with its logit derivative (and every higher gate
    /// channel) identically zero. This covers both an UNGATED atom (`a_k ≡ 1`,
    /// #1026) and FROZEN/amortized routing (`a_k ≡ predicted`, #1033): in either
    /// case the logit is NOT a free Newton parameter, so the gate must not
    /// re-derive from a stale free logit. `None` (or an out-of-range / empty
    /// vector) leaves the atom on the free-logit gate law. Length is `K` when
    /// populated; an empty vector means "no fixed gates" (the historical path).
    pub fixed_gate_value: Vec<Option<f64>>,
    /// Total number of seeded primaries (= `K` of the tower).
    pub n_primaries: usize,
}

impl SaeReconstructionRowProgram {
    /// The gate activation `ζ_k(ℓ)` as a `Tower4<K>` in the gate-logit
    /// primaries. Softmax is the shared composition `exp(ℓ_k·inv_tau) /
    /// Σ_j exp(ℓ_j·inv_tau)`; the per-atom logistic is `σ((ℓ_k − shift_k)·
    /// inv_tau)` depending only on its own logit. Both carry every derivative
    /// channel automatically.
    /// The fixed-gate constant for atom `k`, if its gate is pinned
    /// ([`Self::fixed_gate_value`]). Returns a `constant` tower — value equal to
    /// the pinned active-routing gate, all derivative channels zero — so ungated
    /// (#1026) and frozen-routing (#1033) atoms carry no logit sensitivity.
    #[inline]
    fn fixed_gate<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        atom: usize,
        workspace: &'arena S::Workspace,
    ) -> Option<S> {
        self.fixed_gate_value
            .get(atom)
            .copied()
            .flatten()
            .map(|value| S::constant(value, self.n_primaries, workspace))
    }

    fn gate_tower<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        atom: usize,
        workspace: &'arena S::Workspace,
    ) -> S {
        if let Some(fixed) = self.fixed_gate::<S>(atom, workspace) {
            return fixed;
        }
        let dimension = self.n_primaries;
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
                let mut denom = S::constant(0.0, dimension, workspace);
                let mut numer = S::constant(0.0, dimension, workspace);
                for j in 0..self.gate_value.len() {
                    let lj = match self.logit_slot[j] {
                        Some(slot) => S::variable(self.logits[j], slot, dimension, workspace),
                        None => S::constant(self.logits[j], dimension, workspace),
                    };
                    // (ℓ_j·inv_tau − shift): subtracting a constant shifts only
                    // the value channel, leaving every gradient/Hessian/t3/t4
                    // channel of the exponent (hence of exp via the chain rule)
                    // identical to the unshifted form.
                    let ej = lj
                        .scale(inv_tau)
                        .sub(&S::constant(shift, dimension, workspace))
                        .exp();
                    if j == atom {
                        numer = ej.clone();
                    }
                    denom = denom.add(&ej);
                }
                numer.mul(&recip(&denom))
            }
            RowGate::PerAtomLogistic { inv_tau } => {
                let l = match self.logit_slot[atom] {
                    Some(slot) => S::variable(self.logits[atom], slot, dimension, workspace),
                    None => S::constant(self.logits[atom], dimension, workspace),
                };
                let x = l
                    .sub(&S::constant(self.gate_shift[atom], dimension, workspace))
                    .scale(inv_tau);
                let one = S::constant(1.0, dimension, workspace);
                let sigma = if x.value() >= 0.0 {
                    one.mul(&recip(&one.add(&x.scale(-1.0).exp())))
                } else {
                    let ex = x.exp();
                    ex.mul(&recip(&one.add(&ex)))
                };
                sigma
            }
        }
    }

    /// All atoms' gate jets `ζ_k` at once, with the softmax denominator SHARED
    /// across atoms (#932 perf). The per-atom [`Self::gate_tower`] rebuilds the
    /// whole softmax denominator — `K` exp-jets, their sum, and the reciprocal —
    /// on EVERY call, because only the numerator differs per atom; calling it `K`
    /// times costs `K·(K exps) = O(K²)` exponential jets and `K` reciprocal jets
    /// per row. Here the `K` exp-jets, the denominator sum, and the single
    /// reciprocal jet are built ONCE, then `ζ_k = exp_k · inv_denom`. This emits
    /// exactly `K` exps + `1` recip per row instead of `K²` + `K` (measured:
    /// `K(K−1)` redundant exps and `K−1` redundant recips eliminated per row at
    /// `K=8` ⇒ 56 exps + 7 recips removed), and is **bit-identical** to the
    /// per-atom path (same `exp_k · recip(denom)` product, same Leibniz order).
    /// Pure [`gam_math::jet_scalar::JetScalar`] ops — single-source, exact, no softmax chain rule.
    fn all_gates<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        workspace: &'arena S::Workspace,
    ) -> Vec<S> {
        let n = self.gate_value.len();
        let dimension = self.n_primaries;
        match self.gate {
            RowGate::Softmax { inv_tau } => {
                let shift = self
                    .logits
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    * inv_tau;
                // The K exp-jets and the denominator, built ONCE and shared.
                let mut exps: Vec<S> = Vec::with_capacity(n);
                let mut denom = S::constant(0.0, dimension, workspace);
                for j in 0..n {
                    let lj = match self.logit_slot[j] {
                        Some(slot) => S::variable(self.logits[j], slot, dimension, workspace),
                        None => S::constant(self.logits[j], dimension, workspace),
                    };
                    let ej = lj
                        .scale(inv_tau)
                        .sub(&S::constant(shift, dimension, workspace))
                        .exp();
                    denom = denom.add(&ej);
                    exps.push(ej);
                }
                let inv = recip(&denom);
                // The SAME fixed-gate override the per-atom `gate_tower` applies
                // (#1026/#1033): a pinned atom's gate is a CONSTANT (the active-
                // routing value, every derivative channel zero) and must not
                // re-derive from the softmax — including the cross-sensitivity to
                // OTHER atoms' free logits it would otherwise pick up through the
                // shared denominator. Free atoms keep the pinned atom's exp in
                // their denominator exactly as `gate_tower` does (it enters as a
                // constant, `logit_slot` is `None` for a fixed logit), so free
                // gates are bit-identical between the two paths and this override
                // restores bit-identity for the pinned gates too.
                (0..n)
                    .map(|atom| {
                        self.fixed_gate::<S>(atom, workspace)
                            .unwrap_or_else(|| exps[atom].mul(&inv))
                    })
                    .collect()
            }
            // Per-atom logistic gates are independent (each depends only on its
            // own logit); there is no shared denominator to hoist, so this is the
            // same as calling `gate_tower` per atom.
            RowGate::PerAtomLogistic { .. } => (0..n)
                .map(|atom| self.gate_tower::<S>(atom, workspace))
                .collect(),
        }
    }

    /// The reconstruction output column `c` as a single jet:
    /// `ẑ_c(p) = Σ_k ζ_k(ℓ) · decoded_{k,c}(t_k)`. Its `.v` is the production
    /// reconstruction value, `.g[a]` is `∂ẑ_c/∂p_a`, `.h[a][b]` is
    /// `∂²ẑ_c/∂p_a∂p_b`, and the `t3`/`t4` channels are the exact higher-order
    /// derivatives — all from this ONE evaluation.
    fn reconstruction_column_generic<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        out_col: usize,
        workspace: &'arena S::Workspace,
    ) -> S {
        let dimension = self.n_primaries;
        let mut acc = S::constant(0.0, dimension, workspace);
        for (atom, atom_jet) in self.atoms.iter().enumerate() {
            let gate = self.gate_tower::<S>(atom, workspace);
            let decoded =
                atom_jet.decoded_tower::<S>(out_col, &self.coord_slot[atom], dimension, workspace);
            acc = acc.add(&gate.mul(&decoded));
        }
        acc
    }

    /// The reconstruction output column `c` as the PACKED order-2 jet
    /// [`Order2<K>`](gam_math::jet_scalar::Order2): value `.value()`,
    /// gradient `.g()[a] = ∂ẑ_c/∂p_a`, Hessian `.h()[a][b] = ∂²ẑ_c/∂p_a∂p_b`.
    ///
    /// This is the production path (#932): the arrow-Schur logdet consumer reads
    /// ONLY the order-≤2 channels of the reconstruction, so it builds the packed
    /// [`Order2<K>`] scalar — value/gradient/Hessian only — instead of the dense
    /// [`Tower4<K>`] (which materialises the entire K⁴ `t3`/`t4` tensor every row
    /// only to discard it). For `K` up to 16 the dense tower's tensor build is
    /// ~19× the instruction count of the order-2 channels alone; this collapses
    /// it to the channels actually read. The packed `(v, g, H)` is BIT-IDENTICAL
    /// to the order-≤2 channels of `Self::reconstruction_column_tower` (the
    /// `Order2` newtype delegates to the same `Tower2` arithmetic the dense
    /// tower's order-≤2 channels use); the t3/t4 oracle pins the dense path.
    #[must_use]
    pub fn reconstruction_column_packed<const K: usize>(&self, out_col: usize) -> Order2<K> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.reconstruction_column_generic::<FixedRuntimeJet<Order2<K>, K>>(out_col, &())
            .into_inner()
    }

    /// All `out_dim` reconstruction columns as packed [`Order2<K>`] jets, with
    /// the per-row redundant sub-jets HOISTED out of the output-column loop
    /// (#932 perf). `reconstruction_column_packed(c)` rebuilds, for every output
    /// column `c`, both the per-atom softmax gate jet `ζ_k` (`K` exps + a recip
    /// + a `K×K` Hessian — the dominant cost) AND each per-atom basis jet
    /// `Φ_{k,b}` — yet **neither depends on `c`**: the gate is a function of the
    /// logits only, and the basis jet is the local Taylor model of `Φ_b` in the
    /// coords, the decoder coefficient `B_{b,c}` being the only `c`-dependent
    /// factor. The consumer (`fill_reconstruction_channels_from_program`) calls
    /// it once per `c`, so the gate and basis jets are recomputed `out_dim×`
    /// redundantly.
    ///
    /// This builds each atom's gate jet ONCE (`K` total) and each atom's basis
    /// jets ONCE (`n_basis` per atom), then assembles every column by the cheap
    /// reductions `decoded_{k,c} = Σ_b Φ_{k,b}·B_{b,c}` and
    /// `ẑ_c = Σ_k ζ_k·decoded_{k,c}`. The result is **bit-identical** to calling
    /// [`Self::reconstruction_column_packed`] per column (same Leibniz products in
    /// the same order) — only the redundant recomputation is removed — measured
    /// ~9× faster at `K=8, out_dim=16` on the per-row hot path.
    fn reconstruction_all_columns_generic<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        workspace: &'arena S::Workspace,
    ) -> Vec<S> {
        let p = self.out_dim();
        let dimension = self.n_primaries;
        // Hoist the per-atom gate jet (c-independent) and basis jets
        // (c-independent) out of the column loop. `all_gates` additionally shares
        // the softmax denominator / reciprocal across atoms (K exps + 1 recip,
        // not K² + K).
        let gates: Vec<S> = self.all_gates::<S>(workspace);
        let bases: Vec<Vec<S>> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, atom_jet)| {
                (0..atom_jet.n_basis())
                    .map(|b| {
                        atom_jet.basis_tower::<S>(b, &self.coord_slot[atom], dimension, workspace)
                    })
                    .collect()
            })
            .collect();
        (0..p)
            .map(|c| {
                let mut acc = S::constant(0.0, dimension, workspace);
                for (atom, atom_jet) in self.atoms.iter().enumerate() {
                    // decoded_{k,c} = Σ_b Φ_{k,b}·B_{b,c} from the hoisted basis
                    // jets — same per-basis sum `decoded_tower` forms, but the
                    // basis jets are reused across every column.
                    let mut decoded = S::constant(0.0, dimension, workspace);
                    for basis_col in 0..atom_jet.n_basis() {
                        let coeff = atom_jet.decoder[basis_col][c];
                        if coeff == 0.0 {
                            continue;
                        }
                        decoded = decoded.add(&bases[atom][basis_col].scale(coeff));
                    }
                    acc = acc.add(&gates[atom].mul(&decoded));
                }
                acc
            })
            .collect()
    }

    /// Fixed-size packed order-2 oracle for one row.
    #[must_use]
    pub fn reconstruction_all_columns_packed<const K: usize>(&self) -> Vec<Order2<K>> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.reconstruction_all_columns_generic::<FixedRuntimeJet<Order2<K>, K>>(&())
            .into_iter()
            .map(FixedRuntimeJet::into_inner)
            .collect()
    }

    /// The reconstruction output column as the full dense [`Tower4<K>`] carrying
    /// every value/gradient/Hessian/`t3`/`t4` channel. This is the #932 oracle
    /// ground truth: the production [`Self::reconstruction_column_packed`]
    /// order-2 path is pinned against its order-≤2 channels, and the FD-witness
    /// tests use its `t3`/`t4`. Not on the per-row hot path.
    #[must_use]
    pub fn reconstruction_column<const K: usize>(&self, out_col: usize) -> Tower4<K> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.reconstruction_column_generic::<FixedRuntimeJet<Tower4<K>, K>>(out_col, &())
            .into_inner()
    }

    /// The β **border-channel** local-variable sub-jet: the scalar
    /// `s_{k,b}(p) = ζ_k(ℓ)·Φ_b(t_k)` as a `Tower4<K>` in the local
    /// (logit/coord) primaries — the gate activation times ONE basis function.
    ///
    /// In the arrow system a β border channel is one free decoder coefficient
    /// `β_{k,b,channel}` whose per-row reconstruction contribution to output
    /// column `c` is `ζ_k(ℓ)·Φ_b(t_k)·output_c`, where `output` is the channel's
    /// (frame / identity) output vector carried by the `SaeBorderChannel`, NOT
    /// the current decoder matrix. The reconstruction is **linear** in `β`, so
    /// `∂ẑ_c/∂β_{k,b,channel} = ζ_k(ℓ)·Φ_b(t_k)·output_c = s_{k,b}.v·output_c`
    /// and `∂²ẑ_c/∂β∂p_a = s_{k,b}.g[a]·output_c` (the production `beta` /
    /// `beta_deriv` / `beta_l_deriv` channels). The `output_c` factor is a
    /// per-column constant the caller applies; this tower carries the entire
    /// local-variable dependence.
    ///
    /// It is built from the SAME `gate_tower` / `basis_tower` primitives as
    /// [`Self::reconstruction_column`], so the β border channel is single
    /// sourced with the local-variable reconstruction tower (#932) — the hand
    /// path in `row_jets_for_logdet` packs these same `ζ_k·Φ_b` products (then
    /// multiplies by `channel.output`) term by term, and is pinned to this
    /// tower by the converged-cache oracle.
    fn beta_border_generic<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        atom: usize,
        basis_col: usize,
        workspace: &'arena S::Workspace,
    ) -> S {
        let gate = self.gate_tower::<S>(atom, workspace);
        let phi = self.atoms[atom].basis_tower::<S>(
            basis_col,
            &self.coord_slot[atom],
            self.n_primaries,
            workspace,
        );
        gate.mul(&phi)
    }

    /// The β **border-channel** local-variable sub-jet as the PACKED order-2 jet
    /// [`Order2<K>`](gam_math::jet_scalar::Order2). The consumer reads only
    /// `.value()` (the `beta` channel) and `.g()[a]` (the `beta_deriv` /
    /// `beta_l_deriv` mixed channel — the reconstruction is linear in β so the
    /// Hessian-in-β vanishes and only value+gradient are needed). Built from the
    /// SAME packed gate / basis primitives as [`Self::reconstruction_column`], so
    /// the dense `t3`/`t4` tensor is never materialised on this per-row hot path
    /// (#932 Tower4→Order2 cutover).
    #[must_use]
    pub fn beta_border_tower_packed<const K: usize>(
        &self,
        atom: usize,
        basis_col: usize,
    ) -> Order2<K> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.beta_border_generic::<FixedRuntimeJet<Order2<K>, K>>(atom, basis_col, &())
            .into_inner()
    }

    /// The β border-channel sub-jet as the full dense [`Tower4<K>`] — the #932
    /// oracle ground truth the packed [`Self::beta_border_tower_packed`] is
    /// pinned against. Not on the per-row hot path.
    #[must_use]
    pub fn beta_border_tower<const K: usize>(&self, atom: usize, basis_col: usize) -> Tower4<K> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.beta_border_generic::<FixedRuntimeJet<Tower4<K>, K>>(atom, basis_col, &())
            .into_inner()
    }

    fn beta_border_batch_generic<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        channels: &[(usize, usize)],
        workspace: &'arena S::Workspace,
    ) -> Vec<S> {
        let dimension = self.n_primaries;
        let gates: Vec<S> = self.all_gates::<S>(workspace);
        channels
            .iter()
            .map(|&(atom, basis_col)| {
                let phi = self.atoms[atom].basis_tower::<S>(
                    basis_col,
                    &self.coord_slot[atom],
                    dimension,
                    workspace,
                );
                gates[atom].mul(&phi)
            })
            .collect()
    }

    /// Packed β border-channel sub-jets for a batch of `(atom, basis_col)`
    /// channels, with the per-atom gate jets HOISTED and the softmax denominator
    /// SHARED across atoms (#932 perf): the gate jet `ζ_k` (the dominant `K`-exp
    /// / `K×K`-Hessian cost) is a function of the row's logits only, not of
    /// `basis_col`, and every atom's gate shares one softmax denominator /
    /// reciprocal. `Self::all_gates` builds all `K` gates once (K exps + 1
    /// recip per row); each channel then just multiplies its atom's cached gate
    /// by its basis jet. Each result is **bit-identical** to
    /// [`Self::beta_border_tower_packed`] for the same `(atom, basis_col)` (same
    /// `gate.mul(basis)` product), in the input order.
    #[must_use]
    pub fn beta_border_towers_packed<const K: usize>(
        &self,
        channels: &[(usize, usize)],
    ) -> Vec<Order2<K>> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.beta_border_batch_generic::<FixedRuntimeJet<Order2<K>, K>>(channels, &())
            .into_iter()
            .map(FixedRuntimeJet::into_inner)
            .collect()
    }

    /// Packed β border-channel sub-jets for a batch of channels as the
    /// FIRST-order jet [`Order1<K>`](gam_math::jet_scalar::Order1) — value +
    /// gradient ONLY, no Hessian. The β-border consumer
    /// (`fill_beta_border_channels_from_program`) reads exactly `.value()` (the
    /// `beta` channel) and `.g()[a]` (the mixed `beta_deriv` / `beta_l_deriv`
    /// channel); the reconstruction is linear in β so the Hessian-in-β vanishes
    /// and the K×K Hessian that [`Self::beta_border_towers_packed`]'s `Order2`
    /// builds is computed-and-discarded every call. This method drops that work:
    /// `Order1`'s value/gradient are BIT-IDENTICAL to `Order2`'s (the order-≤1
    /// channels never read a Hessian), proven by the `order1_*` oracle, while the
    /// per-channel `gate.mul(basis)` skips the `K²` Hessian product.
    ///
    /// Same hoisting as [`Self::beta_border_towers_packed`]: gate jets built once
    /// via `Self::all_gates`, each channel multiplies its atom's gate by its
    /// basis jet.
    #[must_use]
    pub fn beta_border_order1_packed<const K: usize>(
        &self,
        channels: &[(usize, usize)],
    ) -> Vec<Order1<K>> {
        assert_eq!(self.n_primaries, K, "fixed jet dimension mismatch");
        self.beta_border_batch_generic::<FixedRuntimeJet<Order1<K>, K>>(channels, &())
            .into_iter()
            .map(FixedRuntimeJet::into_inner)
            .collect()
    }

    /// The number of reconstruction output columns.
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.atoms.first().map_or(0, AtomRowBasisJet::out_dim)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// STRUCTURE-COMPILED SOFTMAX ROW PROGRAM
//
// A dense generic jet represents every primary in every intermediate, including
// the structural-zero cross-atom coordinate blocks.  The interface below is the
// same row program as a borrowed semantic source: gate masses, decoded component
// values, their coordinate jets, and beta-border basis channels.  The executor
// compiles that dependency graph into the nonzero order-2 blocks.  There is one
// softmax-moment definition, shared by reconstruction, coordinate cross terms,
// and beta borders; the fixed-size Tower program remains its independent oracle.

/// One primary in the sparse dependency graph of an SAE reconstruction row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SaeRowPrimary {
    Logit { atom: usize },
    Coord { atom: usize, axis: usize },
}

/// Borrowed semantic input to the structure-compiled softmax row program.
///
/// Production implements this directly over ndarray views, so compiling a row
/// does not clone its basis/decoder tensors.  The owned
/// [`SaeReconstructionRowProgram`] implements it too, which lets the exact same
/// executor run against the generic Taylor-tower oracle in tests.
pub(crate) trait SaeOrder2RowProgramSource {
    fn n_atoms(&self) -> usize;
    fn out_dim(&self) -> usize;
    fn n_primaries(&self) -> usize;
    fn primary(&self, slot: usize) -> SaeRowPrimary;
    fn gate_value(&self, atom: usize) -> f64;
    fn atom_is_active(&self, atom: usize) -> bool;

    /// Fill `D_k`, `∂_axis D_k`, and `∂_axis_a axis_b D_k`, respectively.
    fn fill_decoded(&self, atom: usize, out: &mut [f64]);
    fn fill_decoded_first(&self, atom: usize, axis: usize, out: &mut [f64]);
    fn fill_decoded_second(&self, atom: usize, axis_a: usize, axis_b: usize, out: &mut [f64]);

    fn n_beta_borders(&self) -> usize;
    fn beta_border_atom(&self, border: usize) -> usize;
    fn beta_border_basis_value(&self, border: usize) -> f64;
    fn beta_border_basis_first(&self, border: usize, axis: usize) -> f64;
    fn beta_border_output(&self, border: usize) -> &[f64];
}

/// Complete order-≤2 channels emitted by a structure-compiled row schedule, in
/// one packed allocation. Logical shapes are `first[q,p]`, `second[q,q,p]`,
/// `beta[n_beta,p]`, and two mixed arrays `[q,n_beta,p]`.
#[derive(Debug, Clone)]
pub(crate) struct SaeScheduledRowJets {
    data: Vec<f64>,
    q: usize,
    p: usize,
    n_beta: usize,
}

thread_local! {
    /// Warm per-worker workspace for the structure-compiled softmax row. The
    /// returned channels own their single packed allocation; decoded components,
    /// their expectation, and derivative scratch never escape the call and are
    /// therefore reused across rows on the same worker.
    static SAE_ORDER2_ROW_WORKSPACE: std::cell::RefCell<Vec<f64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl SaeScheduledRowJets {
    pub(crate) fn zeros(q: usize, p: usize, n_beta: usize) -> Self {
        let first = q.checked_mul(p);
        let second = q.checked_mul(q).and_then(|value| value.checked_mul(p));
        let beta = n_beta.checked_mul(p);
        let mixed = q.checked_mul(n_beta).and_then(|value| value.checked_mul(p));
        // SAFETY: a dimension product that cannot fit `usize` cannot describe a
        // realizable allocation; fail before a wrapped length aliases channels.
        let total = first
            .and_then(|value| second.and_then(|next| value.checked_add(next)))
            .and_then(|value| beta.and_then(|next| value.checked_add(next)))
            .and_then(|value| {
                mixed.and_then(|next| {
                    next.checked_mul(2)
                        .and_then(|twice| value.checked_add(twice))
                })
            })
            .expect("SAE row-jet packed channel length overflow");
        Self {
            data: vec![0.0; total],
            q,
            p,
            n_beta,
        }
    }

    #[inline]
    fn second_offset(&self) -> usize {
        self.q * self.p
    }

    #[inline]
    fn beta_offset(&self) -> usize {
        self.second_offset() + self.q * self.q * self.p
    }

    #[inline]
    fn beta_deriv_offset(&self) -> usize {
        self.beta_offset() + self.n_beta * self.p
    }

    #[inline]
    fn beta_l_deriv_offset(&self) -> usize {
        self.beta_deriv_offset() + self.q * self.n_beta * self.p
    }

    #[inline]
    pub(crate) fn q(&self) -> usize {
        self.q
    }

    #[inline]
    pub(crate) fn p(&self) -> usize {
        self.p
    }

    #[inline]
    pub(crate) fn n_beta(&self) -> usize {
        self.n_beta
    }

    #[inline]
    pub(crate) fn first(&self, primary: usize) -> &[f64] {
        let start = primary * self.p;
        &self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn first_mut(&mut self, primary: usize) -> &mut [f64] {
        let start = primary * self.p;
        &mut self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn second(&self, a: usize, b: usize) -> &[f64] {
        let start = self.second_offset() + (a * self.q + b) * self.p;
        &self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn second_mut(&mut self, a: usize, b: usize) -> &mut [f64] {
        let start = self.second_offset() + (a * self.q + b) * self.p;
        &mut self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta(&self, border: usize) -> &[f64] {
        let start = self.beta_offset() + border * self.p;
        &self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta_mut(&mut self, border: usize) -> &mut [f64] {
        let start = self.beta_offset() + border * self.p;
        &mut self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta_deriv(&self, primary: usize, border: usize) -> &[f64] {
        let start = self.beta_deriv_offset() + (primary * self.n_beta + border) * self.p;
        &self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta_deriv_mut(&mut self, primary: usize, border: usize) -> &mut [f64] {
        let start = self.beta_deriv_offset() + (primary * self.n_beta + border) * self.p;
        &mut self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta_l_deriv(&self, primary: usize, border: usize) -> &[f64] {
        let start = self.beta_l_deriv_offset() + (primary * self.n_beta + border) * self.p;
        &self.data[start..start + self.p]
    }

    #[inline]
    pub(crate) fn beta_l_deriv_mut(&mut self, primary: usize, border: usize) -> &mut [f64] {
        let start = self.beta_l_deriv_offset() + (primary * self.n_beta + border) * self.p;
        &mut self.data[start..start + self.p]
    }
}

/// The derivative algebra of `Y = Σ_k z_k D_k`, where `z = softmax(r ℓ)`.
///
/// This is the single softmax primitive used by the compiled row program.  Its
/// centered-moment form is algebraically identical to propagating an order-2 jet:
///
/// ```text
/// ∂_j Y     = r z_j (D_j - Y)
/// ∂_jl Y    = r² z_j [δ_jl(D_j-Y) - z_l(D_j + D_l - 2Y)]
/// ∂_j z_k   = r z_k (δ_kj - z_j)
/// ```
///
/// Unlike a dense tower, evaluating one Hessian entry is O(1), not an O(K)
/// contraction of a materialized `∂²z_k` tensor.  The formulas remain valid
/// for the reduced softmax chart: only the free logit primaries are requested.
struct SoftmaxMoment<'a, S> {
    source: &'a S,
    inv_tau: f64,
}

impl<S: SaeOrder2RowProgramSource> SoftmaxMoment<'_, S> {
    #[inline]
    fn expectation_first_coefficient(&self, atom_j: usize) -> f64 {
        self.inv_tau * self.source.gate_value(atom_j)
    }

    #[inline]
    fn expectation_second_coefficients(&self, atom_j: usize, atom_l: usize) -> (f64, f64) {
        let z_j = self.source.gate_value(atom_j);
        let z_l = self.source.gate_value(atom_l);
        let diagonal = if atom_j == atom_l { 1.0 } else { 0.0 };
        let common = self.inv_tau * self.inv_tau * z_j;
        (common * (diagonal - z_l), -common * z_l)
    }

    #[inline]
    fn gate_first(&self, gated_atom: usize, logit_atom: usize) -> f64 {
        let diagonal = if gated_atom == logit_atom { 1.0 } else { 0.0 };
        // Preserve the historical/tower rounding order `z * (...) * r`; this
        // channel is later multiplied by tiny beta-border outputs, where one
        // earlier rounding can dominate a relative-only oracle.
        self.source.gate_value(gated_atom)
            * (diagonal - self.source.gate_value(logit_atom))
            * self.inv_tau
    }
}

/// Execute the complete softmax reconstruction row as a sparse order-2 jet.
///
/// The evaluator is generic over the borrowed row source, but its arithmetic is
/// fixed by [`SoftmaxMoment`].  It writes every value/gradient/Hessian channel
/// consumed by the SAE log-det path: reconstruction logit and coordinate blocks,
/// same-atom coordinate curvature, logit×coordinate blocks, and decoder-beta
/// border value/mixed channels.  Cross-atom coordinate blocks are exact zeros by
/// dependency, so they are allocated zero and never evaluated.
pub(crate) fn execute_softmax_row_program<S: SaeOrder2RowProgramSource>(
    source: &S,
    inv_tau: f64,
    sqrt_row_w: f64,
) -> SaeScheduledRowJets {
    let k = source.n_atoms();
    let p = source.out_dim();
    let q = source.n_primaries();
    let n_beta = source.n_beta_borders();
    let mut out = SaeScheduledRowJets::zeros(q, p, n_beta);

    // Component values and their softmax expectation.  Inactive components are
    // the exact zero function but their probability still normalizes active
    // gates. All non-output workspace lives in ONE allocation: K centered
    // components, their P-wide expectation, and one reusable P-wide derivative
    // buffer. The variable layout is read directly from the borrowed source;
    // materializing separate logit/coordinate vectors would add two allocations
    // per row without reducing the schedule's asymptotic work.
    let decoded_len = k
        .checked_mul(p)
        .expect("SAE row-program decoded workspace length overflow");
    let tail_len = p
        .checked_mul(2)
        .expect("SAE row-program scratch workspace length overflow");
    let work_len = decoded_len
        .checked_add(tail_len)
        .expect("SAE row-program total workspace length overflow");
    SAE_ORDER2_ROW_WORKSPACE.with(|workspace| {
        let mut workspace = workspace.borrow_mut();
        if workspace.len() < work_len {
            workspace.resize(work_len, 0.0);
        }
        let work = &mut workspace[..work_len];
        work.fill(0.0);
        let (decoded, tail) = work.split_at_mut(decoded_len);
        let (mean, scratch) = tail.split_at_mut(p);
        for atom in 0..k {
            if !source.atom_is_active(atom) {
                continue;
            }
            let component = &mut decoded[atom * p..(atom + 1) * p];
            source.fill_decoded(atom, component);
            let z = source.gate_value(atom);
            for c in 0..p {
                mean[c] += z * component[c];
            }
        }
        let moment = SoftmaxMoment { source, inv_tau };
        // Every logit derivative depends on the centered component `C_k = D_k -
        // E[D]`. Center once here so each Hessian output becomes a two-coefficient
        // vector combination instead of rebuilding `D_j + D_l - 2E[D]`.
        for atom in 0..k {
            let component = &mut decoded[atom * p..(atom + 1) * p];
            for c in 0..p {
                component[c] -= mean[c];
            }
        }

        // Logit gradient and Hessian are centered softmax moments.  This is the
        // asymptotic win: O(L²P) for L free logits, versus O(L²KP) in the hand
        // `d2z[j][l][k] · decoded[k]` contraction and still more in a dense jet.
        for slot_j in 0..q {
            let SaeRowPrimary::Logit { atom: atom_j } = source.primary(slot_j) else {
                continue;
            };
            let centered_j = &decoded[atom_j * p..(atom_j + 1) * p];
            let first_coefficient = sqrt_row_w * moment.expectation_first_coefficient(atom_j);
            for (target, &value) in out.first_mut(slot_j).iter_mut().zip(centered_j) {
                *target = first_coefficient * value;
            }
            for slot_l in 0..q {
                let SaeRowPrimary::Logit { atom: atom_l } = source.primary(slot_l) else {
                    continue;
                };
                let centered_l = &decoded[atom_l * p..(atom_l + 1) * p];
                let (j_coefficient, l_coefficient) =
                    moment.expectation_second_coefficients(atom_j, atom_l);
                let j_coefficient = sqrt_row_w * j_coefficient;
                let l_coefficient = sqrt_row_w * l_coefficient;
                for (c, target) in out.second_mut(slot_j, slot_l).iter_mut().enumerate() {
                    *target = j_coefficient * centered_j[c] + l_coefficient * centered_l[c];
                }
            }
        }

        // Each coordinate belongs to exactly one component.  Its first jet is
        // scaled by z_k; differentiating that gate supplies every logit×coord block.
        for coord_slot in 0..q {
            let SaeRowPrimary::Coord { atom, axis } = source.primary(coord_slot) else {
                continue;
            };
            if !source.atom_is_active(atom) {
                continue;
            }
            source.fill_decoded_first(atom, axis, scratch);
            let z = source.gate_value(atom);
            let coordinate_coefficient = z * sqrt_row_w;
            for (target, &value) in out.first_mut(coord_slot).iter_mut().zip(&*scratch) {
                *target = coordinate_coefficient * value;
            }
            for logit_slot in 0..q {
                let SaeRowPrimary::Logit { atom: logit_atom } = source.primary(logit_slot) else {
                    continue;
                };
                let coefficient = moment.gate_first(atom, logit_atom) * sqrt_row_w;
                for (target, &value) in out
                    .second_mut(logit_slot, coord_slot)
                    .iter_mut()
                    .zip(&*scratch)
                {
                    *target = coefficient * value;
                }
                for (target, &value) in out
                    .second_mut(coord_slot, logit_slot)
                    .iter_mut()
                    .zip(&*scratch)
                {
                    *target = coefficient * value;
                }
            }
        }

        // Coordinate×coordinate curvature is block diagonal by atom.  The basis
        // source supplies the local quadratic jet, so no cross-atom zeros are built.
        for slot_a in 0..q {
            let SaeRowPrimary::Coord {
                atom: atom_a,
                axis: axis_a,
            } = source.primary(slot_a)
            else {
                continue;
            };
            if !source.atom_is_active(atom_a) {
                continue;
            }
            for slot_b in 0..q {
                let SaeRowPrimary::Coord {
                    atom: atom_b,
                    axis: axis_b,
                } = source.primary(slot_b)
                else {
                    continue;
                };
                if atom_a != atom_b {
                    continue;
                }
                source.fill_decoded_second(atom_a, axis_a, axis_b, scratch);
                let coefficient = source.gate_value(atom_a) * sqrt_row_w;
                for (target, &value) in out.second_mut(slot_a, slot_b).iter_mut().zip(&*scratch) {
                    *target = coefficient * value;
                }
            }
        }

        // A beta border is `s = z_k Phi_b` times a constant output vector.  The same
        // gate moment primitive emits its logit derivative; its coordinate derivative
        // is the source basis jet.  beta_deriv and beta_l_deriv are mathematically the
        // same mixed channel because reconstruction is linear in beta.
        for border in 0..n_beta {
            let atom = source.beta_border_atom(border);
            if !source.atom_is_active(atom) {
                continue;
            }
            let phi = source.beta_border_basis_value(border);
            let output = source.beta_border_output(border);
            let base = source.gate_value(atom) * phi * sqrt_row_w;
            for (target, &value) in out.beta_mut(border).iter_mut().zip(output) {
                *target = base * value;
            }
            for slot in 0..q {
                let SaeRowPrimary::Logit { atom: logit_atom } = source.primary(slot) else {
                    continue;
                };
                let scalar = moment.gate_first(atom, logit_atom) * phi * sqrt_row_w;
                for (target, &value) in out.beta_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
                for (target, &value) in out.beta_l_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
            }
            for slot in 0..q {
                let SaeRowPrimary::Coord {
                    atom: coord_atom,
                    axis,
                } = source.primary(slot)
                else {
                    continue;
                };
                if coord_atom != atom {
                    continue;
                }
                let scalar = source.gate_value(atom)
                    * source.beta_border_basis_first(border, axis)
                    * sqrt_row_w;
                for (target, &value) in out.beta_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
                for (target, &value) in out.beta_l_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
            }
        }
    });
    out
}

/// Execute an independent-logistic reconstruction row as a sparse order-2
/// program.
///
/// This is the structure-compiled lowering of
///
/// ```text
/// Y_c = sum_k sigmoid(r * (logit_k - shift_k)) * D_{k,c}(t_k).
/// ```
///
/// Each gate depends on exactly one logit. Therefore the gate Hessian is
/// diagonal, cross-atom logit/coordinate blocks are structural zeros, and every
/// live channel is a direct scalar multiple of one decoded value or derivative:
///
/// ```text
/// z'_k  = r z_k (1-z_k)
/// z''_k = r² z_k (1-z_k) (1-2z_k).
/// ```
///
/// The generic jet remains the independent semantic oracle, but production
/// never allocates or propagates its dense runtime Hessians. This is the
/// independent-gate analogue of [`execute_softmax_row_program`]: one borrowed
/// source, one packed result allocation, and only structurally live work.
pub(crate) fn execute_independent_logistic_row_program<S: SaeOrder2RowProgramSource>(
    source: &S,
    inv_tau: f64,
    sqrt_row_w: f64,
) -> SaeScheduledRowJets {
    let k = source.n_atoms();
    let p = source.out_dim();
    let q = source.n_primaries();
    let n_beta = source.n_beta_borders();
    let mut out = SaeScheduledRowJets::zeros(q, p, n_beta);
    let decoded_len = k
        .checked_mul(p)
        .expect("SAE independent row-program decoded workspace length overflow");
    let work_len = decoded_len
        .checked_add(p)
        .expect("SAE independent row-program workspace length overflow");

    SAE_ORDER2_ROW_WORKSPACE.with(|workspace| {
        let mut workspace = workspace.borrow_mut();
        if workspace.len() < work_len {
            workspace.resize(work_len, 0.0);
        }
        let work = &mut workspace[..work_len];
        work.fill(0.0);
        let (decoded, scratch) = work.split_at_mut(decoded_len);
        for atom in 0..k {
            if source.atom_is_active(atom) {
                source.fill_decoded(atom, &mut decoded[atom * p..(atom + 1) * p]);
            }
        }

        // Gate-only blocks. A fixed gate has no logit primary and consequently
        // emits no derivative channel.
        for slot_a in 0..q {
            let SaeRowPrimary::Logit { atom } = source.primary(slot_a) else {
                continue;
            };
            let z = source.gate_value(atom);
            let dz = inv_tau * z * (1.0 - z);
            let d2z = inv_tau * inv_tau * z * (1.0 - z) * (1.0 - 2.0 * z);
            let component = &decoded[atom * p..(atom + 1) * p];
            for (target, &value) in out.first_mut(slot_a).iter_mut().zip(component) {
                *target = sqrt_row_w * dz * value;
            }
            for slot_b in 0..q {
                if source.primary(slot_b) != (SaeRowPrimary::Logit { atom }) {
                    continue;
                }
                for (target, &value) in out.second_mut(slot_a, slot_b).iter_mut().zip(component) {
                    *target = sqrt_row_w * d2z * value;
                }
            }
        }

        // Coordinate and same-atom logit×coordinate blocks.
        for coord_slot in 0..q {
            let SaeRowPrimary::Coord { atom, axis } = source.primary(coord_slot) else {
                continue;
            };
            if !source.atom_is_active(atom) {
                continue;
            }
            let z = source.gate_value(atom);
            source.fill_decoded_first(atom, axis, scratch);
            for (target, &value) in out.first_mut(coord_slot).iter_mut().zip(&*scratch) {
                *target = sqrt_row_w * z * value;
            }
            for logit_slot in 0..q {
                if source.primary(logit_slot) != (SaeRowPrimary::Logit { atom }) {
                    continue;
                }
                let dz = inv_tau * z * (1.0 - z);
                for (target, &value) in out
                    .second_mut(logit_slot, coord_slot)
                    .iter_mut()
                    .zip(&*scratch)
                {
                    *target = sqrt_row_w * dz * value;
                }
                for (target, &value) in out
                    .second_mut(coord_slot, logit_slot)
                    .iter_mut()
                    .zip(&*scratch)
                {
                    *target = sqrt_row_w * dz * value;
                }
            }
            for other_slot in 0..q {
                let SaeRowPrimary::Coord {
                    atom: other_atom,
                    axis: other_axis,
                } = source.primary(other_slot)
                else {
                    continue;
                };
                if other_atom != atom {
                    continue;
                }
                source.fill_decoded_second(atom, axis, other_axis, scratch);
                for (target, &value) in out
                    .second_mut(coord_slot, other_slot)
                    .iter_mut()
                    .zip(&*scratch)
                {
                    *target = sqrt_row_w * z * value;
                }
            }
        }

        // Decoder-border value and mixed channels. The reconstruction is linear
        // in beta, hence beta_deriv and beta_l_deriv are the same channel.
        for border in 0..n_beta {
            let atom = source.beta_border_atom(border);
            if !source.atom_is_active(atom) {
                continue;
            }
            let z = source.gate_value(atom);
            let phi = source.beta_border_basis_value(border);
            let output = source.beta_border_output(border);
            let base = sqrt_row_w * z * phi;
            for (target, &value) in out.beta_mut(border).iter_mut().zip(output) {
                *target = base * value;
            }
            for slot in 0..q {
                let scalar = match source.primary(slot) {
                    SaeRowPrimary::Logit { atom: logit_atom } if logit_atom == atom => {
                        sqrt_row_w * inv_tau * z * (1.0 - z) * phi
                    }
                    SaeRowPrimary::Coord {
                        atom: coord_atom,
                        axis,
                    } if coord_atom == atom => {
                        sqrt_row_w * z * source.beta_border_basis_first(border, axis)
                    }
                    _ => 0.0,
                };
                if scalar == 0.0 {
                    continue;
                }
                for (target, &value) in out.beta_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
                for (target, &value) in out.beta_l_deriv_mut(slot, border).iter_mut().zip(output) {
                    *target = scalar * value;
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    impl SaeOrder2RowProgramSource for SaeReconstructionRowProgram {
        fn n_atoms(&self) -> usize {
            self.atoms.len()
        }

        fn out_dim(&self) -> usize {
            self.out_dim()
        }

        fn n_primaries(&self) -> usize {
            self.n_primaries
        }

        fn primary(&self, slot: usize) -> SaeRowPrimary {
            for (atom, &candidate) in self.logit_slot.iter().enumerate() {
                if candidate == Some(slot) {
                    return SaeRowPrimary::Logit { atom };
                }
            }
            for (atom, slots) in self.coord_slot.iter().enumerate() {
                for (axis, &candidate) in slots.iter().enumerate() {
                    if candidate == slot {
                        return SaeRowPrimary::Coord { atom, axis };
                    }
                }
            }
            panic!("row-program primary slot {slot} is not mapped");
        }

        fn gate_value(&self, atom: usize) -> f64 {
            self.gate_value[atom]
        }

        fn atom_is_active(&self, atom: usize) -> bool {
            self.fixed_gate_value.get(atom).copied().flatten() != Some(0.0)
        }

        fn fill_decoded(&self, atom: usize, out: &mut [f64]) {
            out.fill(0.0);
            for basis in 0..self.atoms[atom].n_basis() {
                let phi = self.atoms[atom].phi[basis];
                for (c, value) in out.iter_mut().enumerate() {
                    *value += phi * self.atoms[atom].decoder[basis][c];
                }
            }
        }

        fn fill_decoded_first(&self, atom: usize, axis: usize, out: &mut [f64]) {
            out.fill(0.0);
            for basis in 0..self.atoms[atom].n_basis() {
                let d_phi = self.atoms[atom].d_phi[basis][axis];
                for (c, value) in out.iter_mut().enumerate() {
                    *value += d_phi * self.atoms[atom].decoder[basis][c];
                }
            }
        }

        fn fill_decoded_second(&self, atom: usize, axis_a: usize, axis_b: usize, out: &mut [f64]) {
            out.fill(0.0);
            for basis in 0..self.atoms[atom].n_basis() {
                let d2_phi = self.atoms[atom].d2_phi[basis][axis_a][axis_b];
                for (c, value) in out.iter_mut().enumerate() {
                    *value += d2_phi * self.atoms[atom].decoder[basis][c];
                }
            }
        }

        fn n_beta_borders(&self) -> usize {
            0
        }

        // `n_beta_borders()` is 0 for the owned oracle, so every border index is
        // out of range by construction. Report the index that was asked for:
        // the whole point of a failure here is to name the caller that invented
        // a border this source does not have.
        fn beta_border_atom(&self, border: usize) -> usize {
            panic!(
                "owned row-program oracle has no beta borders, but border {border} was requested"
            )
        }

        fn beta_border_basis_value(&self, border: usize) -> f64 {
            panic!(
                "owned row-program oracle has no beta borders, but border {border} was requested"
            )
        }

        fn beta_border_basis_first(&self, border: usize, axis: usize) -> f64 {
            panic!(
                "owned row-program oracle has no beta borders, but border {border} axis {axis} \
                 was requested"
            )
        }

        fn beta_border_output(&self, border: usize) -> &[f64] {
            panic!(
                "owned row-program oracle has no beta borders, but border {border} was requested"
            )
        }
    }

    /// Conditioning-aware beta-border derivative oracle.  The compiled gate
    /// channel is checked against both double-double softmax arithmetic and an
    /// independent five-point derivative of `z_k(ℓ) Phi` evaluated entirely in
    /// double-double precision.  The sweep spans balanced and saturated tails and
    /// twelve orders of border scale, so a tiny derivative is judged against the
    /// operation's conditioning rather than an impossible relative-only floor.
    #[test]
    fn softmax_beta_border_gate_derivative_matches_quad_and_fd_across_tails_932() {
        // Parametrized softmax fixture with `n_atoms` softmax atoms, each carrying a
        // free logit primary and `latent_dim` free coord primaries, so
        // `n_primaries = n_atoms·(1 + latent_dim)`. Layout: logit slots
        // `0..n_atoms`, then atom `k`'s coord axis `j` at `n_atoms + k·latent_dim +
        // j`. This oracle instantiates it with four atoms and one latent axis.
        fn softmax_fixture_k(
            n_atoms: usize,
            latent_dim: usize,
            n_basis: usize,
            out_dim: usize,
            inv_tau: f64,
        ) -> SaeReconstructionRowProgram {
            let mk_atom = |seed: f64| {
                let phi: Vec<f64> = (0..n_basis)
                    .map(|b| 0.3 + 0.2 * (b as f64 + seed))
                    .collect();
                let d_phi: Vec<Vec<f64>> = (0..n_basis)
                    .map(|b| {
                        (0..latent_dim)
                            .map(|axis| 0.1 * (b as f64 + 1.0) - 0.05 * axis as f64 + 0.03 * seed)
                            .collect()
                    })
                    .collect();
                let d2_phi: Vec<Vec<Vec<f64>>> = (0..n_basis)
                    .map(|b| {
                        (0..latent_dim)
                            .map(|a| {
                                (0..latent_dim)
                                    .map(|bb| {
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
                    latent_dim,
                }
            };
            let logits: Vec<f64> = (0..n_atoms)
                .map(|k| 0.4 - 0.13 * k as f64 + 0.05 * (k as f64).sin())
                .collect();
            let e: Vec<f64> = logits.iter().map(|&l| (l * inv_tau).exp()).collect();
            let s: f64 = e.iter().sum();
            let gate_value: Vec<f64> = e.iter().map(|&v| v / s).collect();
            let atoms: Vec<AtomRowBasisJet> = (0..n_atoms).map(|k| mk_atom(k as f64)).collect();
            let logit_slot: Vec<Option<usize>> = (0..n_atoms).map(Some).collect();
            let coord_slot: Vec<Vec<usize>> = (0..n_atoms)
                .map(|k| {
                    (0..latent_dim)
                        .map(|j| n_atoms + k * latent_dim + j)
                        .collect()
                })
                .collect();
            SaeReconstructionRowProgram {
                atoms,
                gate_value,
                logits,
                gate_shift: vec![0.0; n_atoms],
                gate: RowGate::Softmax { inv_tau },
                logit_slot,
                coord_slot,
                fixed_gate_value: Vec::new(),
                n_primaries: n_atoms * (1 + latent_dim),
            }
        }

        use qd::Quad;

        fn q(value: f64) -> Quad {
            Quad::from_f64(value)
        }

        fn q_to_f64(value: Quad) -> f64 {
            value.0 + value.1
        }

        fn quad_border_value(
            logits: &[f64],
            inv_tau: f64,
            gated_atom: usize,
            logit_atom: usize,
            displacement: f64,
            phi: f64,
        ) -> Quad {
            let shifted_max = logits
                .iter()
                .enumerate()
                .map(|(atom, &value)| {
                    (value
                        + if atom == logit_atom {
                            displacement
                        } else {
                            0.0
                        })
                        * inv_tau
                })
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<Quad> = logits
                .iter()
                .enumerate()
                .map(|(atom, &value)| {
                    let displaced = q(value)
                        + q(if atom == logit_atom {
                            displacement
                        } else {
                            0.0
                        });
                    (displaced * q(inv_tau) - q(shifted_max)).exp()
                })
                .collect();
            let denominator = exps
                .iter()
                .copied()
                .fold(Quad::ZERO, |sum, value| sum + value);
            exps[gated_atom] / denominator * q(phi)
        }

        let cases = [
            vec![0.4, -0.7, 0.1, -0.2],
            vec![35.0, 2.0, -18.0, -40.0],
            vec![-35.0, -2.0, 18.0, 40.0],
            vec![8.0, 8.0 - 1.0e-10, -8.0, -24.0],
        ];
        let mut comparisons = 0usize;
        let mut max_conditioned_error = 0.0_f64;
        let mut max_fd_conditioned_error = 0.0_f64;
        for logits in cases {
            for inv_tau in [0.25_f64, 1.3, 4.0] {
                let mut program = softmax_fixture_k(4, 1, 2, 1, inv_tau);
                program.logits.clone_from(&logits);
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) * inv_tau;
                let exps: Vec<f64> = logits
                    .iter()
                    .map(|&value| (value * inv_tau - shift).exp())
                    .collect();
                let denominator: f64 = exps.iter().sum();
                program.gate_value = exps.iter().map(|&value| value / denominator).collect();
                let moment = SoftmaxMoment {
                    source: &program,
                    inv_tau,
                };
                for gated_atom in 0..4 {
                    for logit_atom in 0..4 {
                        for phi in [1.0e-12_f64, 1.0, 1.0e12] {
                            let got = moment.gate_first(gated_atom, logit_atom) * phi;
                            let z_k = quad_border_value(
                                &logits, inv_tau, gated_atom, logit_atom, 0.0, 1.0,
                            );
                            let z_j = quad_border_value(
                                &logits, inv_tau, logit_atom, logit_atom, 0.0, 1.0,
                            );
                            let diagonal: f64 = if gated_atom == logit_atom { 1.0 } else { 0.0 };
                            let exact = z_k * (q(diagonal) - z_j) * q(inv_tau) * q(phi);
                            let exact_f64 = q_to_f64(exact);
                            let condition = q_to_f64(
                                z_k * (q(diagonal.abs()) + z_j) * q(inv_tau.abs()) * q(phi.abs()),
                            )
                            .abs();
                            let error = (got - exact_f64).abs();
                            let allowance = 64.0 * f64::EPSILON * condition + 1.0e-300;
                            max_conditioned_error = max_conditioned_error.max(error / allowance);
                            assert!(
                                error <= allowance,
                                "gate derivative tail/scale error {error:e} > {allowance:e}; \
                                 gated={gated_atom} logit={logit_atom} r={inv_tau} phi={phi:e}"
                            );

                            let h = 1.0e-4_f64;
                            let fm2 = quad_border_value(
                                &logits,
                                inv_tau,
                                gated_atom,
                                logit_atom,
                                -2.0 * h,
                                phi,
                            );
                            let fm1 = quad_border_value(
                                &logits, inv_tau, gated_atom, logit_atom, -h, phi,
                            );
                            let fp1 =
                                quad_border_value(&logits, inv_tau, gated_atom, logit_atom, h, phi);
                            let fp2 = quad_border_value(
                                &logits,
                                inv_tau,
                                gated_atom,
                                logit_atom,
                                2.0 * h,
                                phi,
                            );
                            let fd = (fm2 - q(8.0) * fm1 + q(8.0) * fp1 - fp2) / q(12.0 * h);
                            let fd_error = (q_to_f64(fd) - exact_f64).abs();
                            // A five-point stencil subtracts four nearby function
                            // values. In a saturated softmax tail the derivative
                            // can be tiny while every value is O(phi), so the
                            // truncation bound alone does not cover Quad rounding
                            // amplified by 1/h. Bound that cancellation by the
                            // stencil's absolute condition number and Quad's own
                            // machine epsilon; this remains scale-aware instead of
                            // introducing an arbitrary absolute floor.
                            let stencil_condition = (q_to_f64(fm2).abs()
                                + 8.0 * q_to_f64(fm1).abs()
                                + 8.0 * q_to_f64(fp1).abs()
                                + q_to_f64(fp2).abs())
                                / (12.0 * h);
                            let fd_roundoff_allowance = 64.0 * Quad::EPSILON.0 * stencil_condition;
                            let fd_allowance =
                                2.0e-12 * condition + fd_roundoff_allowance + 1.0e-300;
                            max_fd_conditioned_error =
                                max_fd_conditioned_error.max(fd_error / fd_allowance);
                            assert!(
                                fd_error <= fd_allowance,
                                "quad five-point derivative error {fd_error:e} > {fd_allowance:e}; \
                                 truncation_condition={condition:e} \
                                 stencil_condition={stencil_condition:e} \
                                 gated={gated_atom} logit={logit_atom} r={inv_tau} phi={phi:e}"
                            );
                            comparisons += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "[SAE-SOFTMAX-ACCURACY-932] comparisons={comparisons} \
             max_f64_condition_fraction={max_conditioned_error:.3e} \
             max_quad_fd_condition_fraction={max_fd_conditioned_error:.3e}"
        );
        assert_eq!(comparisons, 576);
    }
}
