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
//! — a **gate nonlinearity** `ζ(ℓ)` (softmax / ordered Beta--Bernoulli sigmoid) composed with a
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
//! [`Tower4<K>`](gam_math::jet_tower::Tower4) scalar so the
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

use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOrder1, DynamicOrder2, FixedRuntimeJet, Order1, Order2,
    RuntimeJetScalar,
};
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

    /// Arena-backed gate array for the runtime production scalars. Per-atom
    /// assignment modes (the production users of dynamic jets) have independent
    /// gates, so this evaluates the same [`Self::gate_tower`] expression once per
    /// atom and stores the handles in the row workspace. Softmax uses the same
    /// path only as a dynamic correctness oracle; production softmax remains the
    /// closed-form hand kernel.
    fn all_gates_dynamic<'arena, S>(&self, arena: &'arena DynamicJetArena) -> &'arena [S]
    where
        S: RuntimeJetScalar<'arena, Workspace = DynamicJetArena>,
    {
        arena.alloc_slice_fill_with(self.gate_value.len(), |atom| {
            self.gate_tower::<S>(atom, arena)
        })
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
    /// to the order-≤2 channels of [`Self::reconstruction_column_tower`] (the
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

    /// Runtime-sized packed order-2 production backend for one row.
    #[must_use]
    pub fn reconstruction_all_columns_dynamic<'arena>(
        &self,
        arena: &'arena DynamicJetArena,
    ) -> &'arena [DynamicOrder2<'arena>] {
        let dimension = self.n_primaries;
        let gates = self.all_gates_dynamic::<DynamicOrder2<'arena>>(arena);
        let total_basis: usize = self.atoms.iter().map(AtomRowBasisJet::n_basis).sum();
        let mut atom_cursor = 0usize;
        let mut basis_cursor = 0usize;
        let bases = arena.alloc_slice_fill_with(total_basis, |_| {
            while basis_cursor == self.atoms[atom_cursor].n_basis() {
                atom_cursor += 1;
                basis_cursor = 0;
            }
            let basis = self.atoms[atom_cursor].basis_tower::<DynamicOrder2<'arena>>(
                basis_cursor,
                &self.coord_slot[atom_cursor],
                dimension,
                arena,
            );
            basis_cursor += 1;
            basis
        });
        arena.alloc_slice_fill_with(self.out_dim(), |out_col| {
            let mut acc = DynamicOrder2::constant(0.0, dimension, arena);
            let mut basis_offset = 0usize;
            for (atom, atom_jet) in self.atoms.iter().enumerate() {
                let mut decoded = DynamicOrder2::constant(0.0, dimension, arena);
                for basis_col in 0..atom_jet.n_basis() {
                    let coefficient = atom_jet.decoder[basis_col][out_col];
                    if coefficient != 0.0 {
                        decoded = decoded.add(&bases[basis_offset + basis_col].scale(coefficient));
                    }
                }
                basis_offset += atom_jet.n_basis();
                acc = acc.add(&gates[atom].mul(&decoded));
            }
            acc
        })
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
    /// reciprocal. [`Self::all_gates`] builds all `K` gates once (K exps + 1
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
    /// via [`Self::all_gates`], each channel multiplies its atom's gate by its
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

    /// Runtime-sized packed first-order β-border backend for one row.
    #[must_use]
    pub fn beta_border_order1_dynamic<'arena>(
        &self,
        channels: &[(usize, usize)],
        arena: &'arena DynamicJetArena,
    ) -> &'arena [DynamicOrder1<'arena>] {
        let dimension = self.n_primaries;
        let gates = self.all_gates_dynamic::<DynamicOrder1<'arena>>(arena);
        arena.alloc_slice_fill_with(channels.len(), |channel| {
            let (atom, basis_col) = channels[channel];
            let phi = self.atoms[atom].basis_tower::<DynamicOrder1<'arena>>(
                basis_col,
                &self.coord_slot[atom],
                dimension,
                arena,
            );
            gates[atom].mul(&phi)
        })
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
pub(crate) trait SaeSoftmaxRowProgramSource {
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

/// Complete order-≤2 channels emitted by [`execute_softmax_row_program`], in
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
    static SAE_SOFTMAX_ROW_WORKSPACE: std::cell::RefCell<Vec<f64>> =
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

impl<S: SaeSoftmaxRowProgramSource> SoftmaxMoment<'_, S> {
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
pub(crate) fn execute_softmax_row_program<S: SaeSoftmaxRowProgramSource>(
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
    SAE_SOFTMAX_ROW_WORKSPACE.with(|workspace| {
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

#[cfg(test)]
mod tests_schedule_source {
    use super::*;

    impl SaeSoftmaxRowProgramSource for SaeReconstructionRowProgram {
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

        fn beta_border_atom(&self, _: usize) -> usize {
            panic!("owned row-program oracle has no beta borders")
        }

        fn beta_border_basis_value(&self, _: usize) -> f64 {
            panic!("owned row-program oracle has no beta borders")
        }

        fn beta_border_basis_first(&self, _: usize, _: usize) -> f64 {
            panic!("owned row-program oracle has no beta borders")
        }

        fn beta_border_output(&self, _: usize) -> &[f64] {
            panic!("owned row-program oracle has no beta borders")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 4-ROW SIMD BATCH (the jet's throughput lever over hand-scalar code)
//
// The hot per-row jet kernels (`reconstruction_all_columns_packed`,
// `beta_border_order1_packed`) evaluate ONE row's `(v, g, H)` / `(v, g)` tower
// at a time in scalar `f64`. A hand-written scalar derivative does exactly the
// same. The throughput lever a jet has that scalar hand-code cannot is **row
// batching in SIMD lanes**: the order-≤2 Leibniz product is `O(K²)` independent
// per-channel float ops, and EVERY softmax row runs the IDENTICAL op graph on
// different data — the textbook SPMD shape. Packing `LANES = 4` aligned rows
// into a `[f64; 4]` lane and running the algebra once per 4 rows replaces 4
// scalar passes with one vector pass, so the `K²` Hessian-channel updates become
// 4-wide lane ops covering 4 rows each (auto-vectorised to SSE2 `pd` / NEON
// `.2d`), ~4× fewer scalar FP instructions per row.
//
// The lane field is a plain `[f64; 4]` whose every op is a lane-wise IEEE
// `+`/`-`/`*` (NEVER a fused `mul_add`), so lane `i` of a 4-wide op equals the
// scalar `f64` op on that lane's inputs BIT-FOR-BIT. The op order mirrors
// [`gam_math::jet_tower::Tower2`] / [`Order1`] term-for-term, so
// [`O2x4`]/[`O1x4`] lane `i` is `to_bits`-identical to the production
// [`Order2`]/[`Order1`] row scalar — proven by the `batch_tests` oracle below
// (≥2000 random aligned 4-row batches across `K ∈ {2,4,6}`).
//
// Only the softmax gate is batched: its op graph is identical across rows (every
// atom is an active free logit), while the per-atom logistic gate's
// `x.value() >= 0.0` branch is per-row data-dependent (lanes could need
// different branches, which are NOT bit-identical), so logistic rows fall back
// to the scalar per-row path in the caller.

const LANES: usize = 4;

#[inline]
fn l_splat(x: f64) -> [f64; LANES] {
    [x; LANES]
}
#[inline]
fn l_add(a: [f64; LANES], b: [f64; LANES]) -> [f64; LANES] {
    let mut o = [0.0; LANES];
    for i in 0..LANES {
        o[i] = a[i] + b[i];
    }
    o
}
#[inline]
fn l_mul(a: [f64; LANES], b: [f64; LANES]) -> [f64; LANES] {
    let mut o = [0.0; LANES];
    for i in 0..LANES {
        o[i] = a[i] * b[i];
    }
    o
}

/// 4-rows-per-pass order-≤2 lane scalar (value / gradient / Hessian), mirroring
/// [`gam_math::jet_tower::Tower2`] (hence [`Order2`]) term-for-term so lane `i`
/// is `to_bits`-identical to the scalar row-`i` [`Order2`].
#[derive(Clone, Copy)]
struct O2x4<const K: usize> {
    v: [f64; LANES],
    g: [[f64; LANES]; K],
    h: [[[f64; LANES]; K]; K],
}

impl<const K: usize> O2x4<K> {
    #[inline]
    fn constant(c: [f64; LANES]) -> Self {
        O2x4 {
            v: c,
            g: [[0.0; LANES]; K],
            h: [[[0.0; LANES]; K]; K],
        }
    }
    /// Seeded primary `axis` at (per-lane) `value`: unit first derivative.
    #[inline]
    fn variable(value: [f64; LANES], axis: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[axis] = l_splat(1.0);
        out
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = l_add(self.v, o.v);
        for i in 0..K {
            out.g[i] = l_add(self.g[i], o.g[i]);
            for j in 0..K {
                out.h[i][j] = l_add(self.h[i][j], o.h[i][j]);
            }
        }
        out
    }
    #[inline]
    fn scale(&self, s: [f64; LANES]) -> Self {
        let mut out = *self;
        out.v = l_mul(self.v, s);
        for i in 0..K {
            out.g[i] = l_mul(self.g[i], s);
            for j in 0..K {
                out.h[i][j] = l_mul(self.h[i][j], s);
            }
        }
        out
    }
    /// `self - o`, expressed as `self + o·(-1)` exactly as [`Order2::sub`] does.
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.scale(l_splat(-1.0)))
    }
    /// Order-≤2 Leibniz product, term-for-term identical to `Tower2::mul`.
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::constant(l_mul(a.v, b.v));
        for i in 0..K {
            out.g[i] = l_add(l_mul(a.v, b.g[i]), l_mul(a.g[i], b.v));
        }
        // Upper-triangle-then-mirror, EXACTLY as the scalar `Tower2::mul`
        // (`for j in i..K { hij = …; h[i][j] = h[j][i] = hij }`). The scalar tower
        // never fills the lower triangle independently — it copies the upper one —
        // so a per-`(i,j)` recomputation here diverges by ULPs on `h[j][i]`: the two
        // cross terms `a.g·b.g` accumulate in the opposite order for `(j,i)` vs
        // `(i,j)`, and `b`'s own Hessian is only ULP-symmetric (its `compose_unary`
        // fills each `(i,j)` independently). Mirroring makes every lane
        // `to_bits`-identical to the scalar row path.
        for i in 0..K {
            for j in i..K {
                let t0 = l_mul(a.v, b.h[i][j]);
                let t1 = l_add(t0, l_mul(a.g[i], b.g[j]));
                let t2 = l_add(t1, l_mul(a.g[j], b.g[i]));
                let hij = l_add(t2, l_mul(a.h[i][j], b.v));
                out.h[i][j] = hij;
                out.h[j][i] = hij;
            }
        }
        out
    }
    /// Order-≤2 Faà di Bruno `f ∘ self` from the per-lane stack
    /// `d = [f(u), f′(u), f″(u)]`, mirroring `Tower2::compose_unary`
    /// (`acc` starts at `+0.0`, accumulates `d₁·hᵢⱼ` then `(d₂·gᵢ)·gⱼ`).
    #[inline]
    fn compose(&self, d: [[f64; LANES]; 3]) -> Self {
        let mut out = Self::constant(d[0]);
        for i in 0..K {
            let mut acc = l_splat(0.0);
            acc = l_add(acc, l_mul(d[1], self.g[i]));
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = l_splat(0.0);
                acc = l_add(acc, l_mul(d[1], self.h[i][j]));
                acc = l_add(acc, l_mul(l_mul(d[2], self.g[i]), self.g[j]));
                out.h[i][j] = acc;
            }
        }
        out
    }
    /// `e^self`, per-lane stack `[e, e, e]` (matches `Tower2::exp`).
    #[inline]
    fn exp(&self) -> Self {
        let mut e = [0.0; LANES];
        for i in 0..LANES {
            e[i] = self.v[i].exp();
        }
        self.compose([e, e, e])
    }
    /// `1/self`, per-lane stack `[1/u, -1/u², 2/u³]` — the DIVISION-based stack
    /// of the [`recip`] free fn the scalar reconstruction path uses (NOT the
    /// reciprocal-multiply `[r,-r²,2r³]` of
    /// [`gam_math::jet_scalar::JetScalar::recip`]; those differ by a
    /// ULP and would break `to_bits` parity). Caller guarantees nonzero.
    #[inline]
    fn recip(&self) -> Self {
        let mut d0 = [0.0; LANES];
        let mut d1 = [0.0; LANES];
        let mut d2 = [0.0; LANES];
        for i in 0..LANES {
            let u = self.v[i];
            let u2 = u * u;
            let u3 = u2 * u;
            d0[i] = 1.0 / u;
            d1[i] = -1.0 / u2;
            d2[i] = 2.0 / u3;
        }
        self.compose([d0, d1, d2])
    }
    /// Extract lane `i` as a production [`Order2<K>`] scalar.
    #[inline]
    fn lane(&self, i: usize) -> Order2<K> {
        let mut t = gam_math::jet_tower::Tower2::<K>::constant(self.v[i]);
        for a in 0..K {
            t.g[a] = self.g[a][i];
            for b in 0..K {
                t.h[a][b] = self.h[a][b][i];
            }
        }
        Order2(t)
    }
}

/// 4-rows-per-pass FIRST-order lane scalar (value / gradient only), mirroring
/// [`Order1`] term-for-term so lane `i` is `to_bits`-identical to row-`i`
/// [`Order1`]. Used for the β-border consumer (reconstruction is linear in β,
/// so only value + gradient are read).
#[derive(Clone, Copy)]
struct O1x4<const K: usize> {
    v: [f64; LANES],
    g: [[f64; LANES]; K],
}

impl<const K: usize> O1x4<K> {
    #[inline]
    fn constant(c: [f64; LANES]) -> Self {
        O1x4 {
            v: c,
            g: [[0.0; LANES]; K],
        }
    }
    #[inline]
    fn variable(value: [f64; LANES], axis: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[axis] = l_splat(1.0);
        out
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = l_add(self.v, o.v);
        for i in 0..K {
            out.g[i] = l_add(self.g[i], o.g[i]);
        }
        out
    }
    #[inline]
    fn scale(&self, s: [f64; LANES]) -> Self {
        let mut out = *self;
        out.v = l_mul(self.v, s);
        for i in 0..K {
            out.g[i] = l_mul(self.g[i], s);
        }
        out
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.scale(l_splat(-1.0)))
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        // Tower2::mul value/grad terms (order-≤1 truncation): v = a.v·b.v;
        // g[i] = a.v·b.g[i] + a.g[i]·b.v. Identical float order to `Order1::mul`.
        let a = self;
        let b = o;
        let mut out = Self::constant(l_mul(a.v, b.v));
        for i in 0..K {
            out.g[i] = l_add(l_mul(a.v, b.g[i]), l_mul(a.g[i], b.v));
        }
        out
    }
    #[inline]
    fn compose(&self, d: [[f64; LANES]; 2]) -> Self {
        // Order-≤1 Faà di Bruno: v = d[0]; g[i] = d[1]·g[i] (matches
        // `Order1::compose_unary`, `acc` starts at +0.0).
        let mut out = Self::constant(d[0]);
        for i in 0..K {
            let mut acc = l_splat(0.0);
            acc = l_add(acc, l_mul(d[1], self.g[i]));
            out.g[i] = acc;
        }
        out
    }
    #[inline]
    fn exp(&self) -> Self {
        let mut e = [0.0; LANES];
        for i in 0..LANES {
            e[i] = self.v[i].exp();
        }
        self.compose([e, e])
    }
    #[inline]
    fn recip(&self) -> Self {
        // Division-based `[1/u, -1/u²]` matching the `recip` free fn (see
        // `O2x4::recip`), so lane `i` is `to_bits`-identical to the scalar path.
        let mut d0 = [0.0; LANES];
        let mut d1 = [0.0; LANES];
        for i in 0..LANES {
            let u = self.v[i];
            let u2 = u * u;
            d0[i] = 1.0 / u;
            d1[i] = -1.0 / u2;
        }
        self.compose([d0, d1])
    }
    #[inline]
    fn lane(&self, i: usize) -> Order1<K> {
        let mut g = [0.0; K];
        for a in 0..K {
            g[a] = self.g[a][i];
        }
        Order1 { v: self.v[i], g }
    }
}

/// Structural layout signature of a row program: the part that MUST be identical
/// across rows for them to share one SIMD op graph (slot mapping, per-atom
/// basis/latent/decoder shape, primary count). The per-row numeric data
/// (`phi`/`d_phi`/`d2_phi`/`decoder` VALUES, `logits`) is what differs between
/// lanes; the layout is what is shared.
impl SaeReconstructionRowProgram {
    /// Whether `self` and `other` share the SIMD-batchable softmax layout: same
    /// softmax temperature, primary count, slot mapping, and per-atom basis /
    /// latent / decoder dimensions. (Decoder/basis VALUES may differ per row and
    /// are lane-packed; only the SHAPES must match.)
    fn batch_aligned_softmax_with(&self, other: &Self) -> bool {
        // Both rows must gate through softmax at the same temperature; a
        // bit-for-bit `inv_tau` match is what lets them share one op graph.
        match (self.gate, other.gate) {
            (RowGate::Softmax { inv_tau: a }, RowGate::Softmax { inv_tau: b }) => {
                if a.to_bits() != b.to_bits() {
                    return false;
                }
            }
            _ => return false,
        }
        if self.n_primaries != other.n_primaries
            || self.atoms.len() != other.atoms.len()
            || self.logit_slot != other.logit_slot
            || self.coord_slot != other.coord_slot
            || self.logits.len() != other.logits.len()
        {
            return false;
        }
        for (a, b) in self.atoms.iter().zip(other.atoms.iter()) {
            if a.latent_dim != b.latent_dim
                || a.n_basis() != b.n_basis()
                || a.out_dim() != b.out_dim()
            {
                return false;
            }
        }
        true
    }

    /// All `K` softmax gate lane-jets (`Order2` channels), with the denominator
    /// SHARED across atoms and 4 rows packed per lane. Mirrors [`Self::all_gates`]
    /// term-for-term so lane `i` is `to_bits`-identical to the row-`i` scalar
    /// `all_gates::<K, Order2<K>>()`.
    fn all_gates_o2x4<const K: usize>(&self, rows: &[&Self; LANES], inv_tau: f64) -> Vec<O2x4<K>> {
        let n = self.gate_value.len();
        let inv_tau_l = l_splat(inv_tau);
        // Per-lane max-subtraction shift (= the scalar `all_gates` softmax shift,
        // computed independently per row/lane).
        let mut shift = [0.0; LANES];
        for (lane, r) in rows.iter().enumerate() {
            shift[lane] = r.logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) * inv_tau;
        }
        let mut exps: Vec<O2x4<K>> = Vec::with_capacity(n);
        let mut denom = O2x4::<K>::constant(l_splat(0.0));
        for j in 0..n {
            let mut lj_val = [0.0; LANES];
            for (lane, r) in rows.iter().enumerate() {
                lj_val[lane] = r.logits[j];
            }
            let lj = match self.logit_slot[j] {
                Some(slot) => O2x4::<K>::variable(lj_val, slot),
                None => O2x4::<K>::constant(lj_val),
            };
            let ej = lj.scale(inv_tau_l).sub(&O2x4::<K>::constant(shift)).exp();
            denom = denom.add(&ej);
            exps.push(ej);
        }
        let inv = denom.recip();
        exps.iter().map(|e| e.mul(&inv)).collect()
    }

    /// All `K` softmax gate lane-jets at FIRST order (`Order1` channels).
    /// Mirrors `all_gates::<K, Order1<K>>()` term-for-term.
    fn all_gates_o1x4<const K: usize>(&self, rows: &[&Self; LANES], inv_tau: f64) -> Vec<O1x4<K>> {
        let n = self.gate_value.len();
        let inv_tau_l = l_splat(inv_tau);
        let mut shift = [0.0; LANES];
        for (lane, r) in rows.iter().enumerate() {
            shift[lane] = r.logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) * inv_tau;
        }
        let mut exps: Vec<O1x4<K>> = Vec::with_capacity(n);
        let mut denom = O1x4::<K>::constant(l_splat(0.0));
        for j in 0..n {
            let mut lj_val = [0.0; LANES];
            for (lane, r) in rows.iter().enumerate() {
                lj_val[lane] = r.logits[j];
            }
            let lj = match self.logit_slot[j] {
                Some(slot) => O1x4::<K>::variable(lj_val, slot),
                None => O1x4::<K>::constant(lj_val),
            };
            let ej = lj.scale(inv_tau_l).sub(&O1x4::<K>::constant(shift)).exp();
            denom = denom.add(&ej);
            exps.push(ej);
        }
        let inv = denom.recip();
        exps.iter().map(|e| e.mul(&inv)).collect()
    }

    /// One atom's basis jet `Φ_b(t)` as an [`O2x4`] over 4 rows, mirroring
    /// [`AtomRowBasisJet::basis_tower`] term-for-term. A data-dependent `== 0`
    /// skip is taken only when ALL 4 lanes are zero (the contribution of a zero
    /// lane is `+0.0`, bit-identical to the scalar skip).
    fn basis_tower_o2x4<const K: usize>(
        rows: &[&Self; LANES],
        atom: usize,
        basis_col: usize,
        coord_slots: &[usize],
    ) -> O2x4<K> {
        let latent = rows[0].atoms[atom].latent_dim;
        let mut phi0 = [0.0; LANES];
        for (lane, r) in rows.iter().enumerate() {
            phi0[lane] = r.atoms[atom].phi[basis_col];
        }
        let mut acc = O2x4::<K>::constant(phi0);
        for axis in 0..latent {
            let slot = coord_slots[axis];
            if slot == SAE_FIXED_COORD_SLOT {
                continue;
            }
            let mut d1 = [0.0; LANES];
            let mut any = false;
            for (lane, r) in rows.iter().enumerate() {
                let v = r.atoms[atom].d_phi[basis_col][axis];
                d1[lane] = v;
                any |= v != 0.0;
            }
            if any {
                acc = acc.add(&O2x4::<K>::variable(l_splat(0.0), slot).scale(d1));
            }
        }
        for axis_a in 0..latent {
            // Hoist the fixed-slot skip and the `va` variable build out of the
            // inner axis_b loop: both depend only on axis_a, so the old code
            // rebuilt `va` and re-tested the slot `latent` times per axis_a.
            let slot_a = coord_slots[axis_a];
            if slot_a == SAE_FIXED_COORD_SLOT {
                continue;
            }
            let va = O2x4::<K>::variable(l_splat(0.0), slot_a);
            for axis_b in 0..latent {
                let slot_b = coord_slots[axis_b];
                if slot_b == SAE_FIXED_COORD_SLOT {
                    continue;
                }
                let mut d2 = [0.0; LANES];
                let mut any = false;
                for (lane, r) in rows.iter().enumerate() {
                    let v = r.atoms[atom].d2_phi[basis_col][axis_a][axis_b];
                    d2[lane] = v;
                    any |= v != 0.0;
                }
                if !any {
                    continue;
                }
                let mut half_d2 = [0.0; LANES];
                for lane in 0..LANES {
                    half_d2[lane] = 0.5 * d2[lane];
                }
                let vb = O2x4::<K>::variable(l_splat(0.0), slot_b);
                acc = acc.add(&va.mul(&vb).scale(half_d2));
            }
        }
        acc
    }

    /// One atom's basis jet as an [`O1x4`] (value + gradient), mirroring
    /// `basis_tower::<Order1>` term-for-term.
    fn basis_tower_o1x4<const K: usize>(
        rows: &[&Self; LANES],
        atom: usize,
        basis_col: usize,
        coord_slots: &[usize],
    ) -> O1x4<K> {
        let latent = rows[0].atoms[atom].latent_dim;
        let mut phi0 = [0.0; LANES];
        for (lane, r) in rows.iter().enumerate() {
            phi0[lane] = r.atoms[atom].phi[basis_col];
        }
        let mut acc = O1x4::<K>::constant(phi0);
        for axis in 0..latent {
            let slot = coord_slots[axis];
            if slot == SAE_FIXED_COORD_SLOT {
                continue;
            }
            let mut d1 = [0.0; LANES];
            let mut any = false;
            for (lane, r) in rows.iter().enumerate() {
                let v = r.atoms[atom].d_phi[basis_col][axis];
                d1[lane] = v;
                any |= v != 0.0;
            }
            if any {
                acc = acc.add(&O1x4::<K>::variable(l_splat(0.0), slot).scale(d1));
            }
        }
        for axis_a in 0..latent {
            for axis_b in 0..latent {
                if coord_slots[axis_a] == SAE_FIXED_COORD_SLOT
                    || coord_slots[axis_b] == SAE_FIXED_COORD_SLOT
                {
                    continue;
                }
                let mut d2 = [0.0; LANES];
                let mut any = false;
                for (lane, r) in rows.iter().enumerate() {
                    let v = r.atoms[atom].d2_phi[basis_col][axis_a][axis_b];
                    d2[lane] = v;
                    any |= v != 0.0;
                }
                if !any {
                    continue;
                }
                let mut half_d2 = [0.0; LANES];
                for lane in 0..LANES {
                    half_d2[lane] = 0.5 * d2[lane];
                }
                let va = O1x4::<K>::variable(l_splat(0.0), coord_slots[axis_a]);
                let vb = O1x4::<K>::variable(l_splat(0.0), coord_slots[axis_b]);
                acc = acc.add(&va.mul(&vb).scale(half_d2));
            }
        }
        acc
    }

    /// All `out_dim` reconstruction columns for FOUR softmax-aligned rows at once,
    /// returned per row. Each row's column vector is BIT-IDENTICAL to
    /// [`Self::reconstruction_all_columns_packed`] on that row (same hoisting,
    /// same Leibniz products in the same order — lane `i` mirrors the scalar
    /// row-`i` path). Returns `None` if the four rows are not softmax-aligned, so
    /// the caller can fall back to the scalar per-row path.
    #[must_use]
    pub fn reconstruction_all_columns_batch4<const K: usize>(
        rows: [&Self; 4],
    ) -> Option<[Vec<Order2<K>>; 4]> {
        let head = rows[0];
        if head.n_primaries != K {
            return None;
        }
        let inv_tau = match head.gate {
            RowGate::Softmax { inv_tau } => inv_tau,
            RowGate::PerAtomLogistic { .. } => return None,
        };
        for r in &rows[1..] {
            if !head.batch_aligned_softmax_with(r) {
                return None;
            }
        }
        let p = head.out_dim();
        let gates: Vec<O2x4<K>> = head.all_gates_o2x4::<K>(&rows, inv_tau);
        // Build a jet tower ONLY for the basis columns that actually decode to
        // something: a column whose decoder row is identically zero across every
        // output channel AND every lane contributes exactly zero to all `p` output
        // sums, so both its (expensive) O2x4 tower build and its per-output gather
        // are pure waste. Skipping it is bit-identical — the old inner `any` guard
        // already dropped the scaled add, this just also drops the dead tower build
        // and the dead re-gather across all `p` columns. Each atom keeps a compact
        // `(basis_col, tower)` list of its live columns.
        let bases: Vec<Vec<(usize, O2x4<K>)>> = head
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, atom_jet)| {
                (0..atom_jet.n_basis())
                    .filter(|&b| {
                        rows.iter()
                            .any(|r| (0..p).any(|c| r.atoms[atom].decoder[b][c] != 0.0))
                    })
                    .map(|b| {
                        (
                            b,
                            Self::basis_tower_o2x4::<K>(&rows, atom, b, &head.coord_slot[atom]),
                        )
                    })
                    .collect()
            })
            .collect();
        let mut cols: [Vec<Order2<K>>; LANES] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for c in 0..p {
            let mut acc = O2x4::<K>::constant(l_splat(0.0));
            for atom in 0..head.atoms.len() {
                let mut decoded = O2x4::<K>::constant(l_splat(0.0));
                for (basis_col, tower) in &bases[atom] {
                    let mut coeff = [0.0; LANES];
                    let mut any = false;
                    for (lane, r) in rows.iter().enumerate() {
                        let v = r.atoms[atom].decoder[*basis_col][c];
                        coeff[lane] = v;
                        any |= v != 0.0;
                    }
                    if any {
                        decoded = decoded.add(&tower.scale(coeff));
                    }
                }
                acc = acc.add(&gates[atom].mul(&decoded));
            }
            for lane in 0..LANES {
                cols[lane].push(acc.lane(lane));
            }
        }
        Some(cols)
    }

    /// Packed β-border FIRST-order jets for a batch of `(atom, basis_col)`
    /// channels, for FOUR softmax-aligned rows at once, returned per row. Each
    /// row's channel vector is BIT-IDENTICAL to
    /// [`Self::beta_border_order1_packed`] on that row. Returns `None` if the
    /// rows are not softmax-aligned.
    #[must_use]
    pub fn beta_border_order1_batch4<const K: usize>(
        rows: [&Self; 4],
        channels: &[(usize, usize)],
    ) -> Option<[Vec<Order1<K>>; 4]> {
        let head = rows[0];
        if head.n_primaries != K {
            return None;
        }
        let inv_tau = match head.gate {
            RowGate::Softmax { inv_tau } => inv_tau,
            RowGate::PerAtomLogistic { .. } => return None,
        };
        for r in &rows[1..] {
            if !head.batch_aligned_softmax_with(r) {
                return None;
            }
        }
        let gates: Vec<O1x4<K>> = head.all_gates_o1x4::<K>(&rows, inv_tau);
        let mut out: [Vec<Order1<K>>; LANES] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &(atom, basis_col) in channels {
            let phi = Self::basis_tower_o1x4::<K>(&rows, atom, basis_col, &head.coord_slot[atom]);
            let s = gates[atom].mul(&phi);
            for lane in 0..LANES {
                out[lane].push(s.lane(lane));
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::JetScalar;
    // `value`/`add`/… moved to the shared `JetField` base (JetScalar: JetField);
    // the concrete-typed scalar reads in these tests need it in scope.
    use gam_math::nested_dual::JetField;

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
            gate_shift: vec![0.0, 0.0],
            gate: RowGate::Softmax { inv_tau },
            logit_slot: vec![Some(0), Some(1)],
            coord_slot: vec![vec![2, 3], vec![4, 5]],
            fixed_gate_value: Vec::new(),
            n_primaries: 6,
        };
        (prog, inv_tau)
    }

    /// Parametrized softmax fixture with `n_atoms` softmax atoms, each carrying a
    /// free logit primary and `latent_dim` free coord primaries, so
    /// `n_primaries = n_atoms·(1 + latent_dim)`. Layout: logit slots
    /// `0..n_atoms`, then atom `k`'s coord axis `j` at `n_atoms + k·latent_dim +
    /// j`. Used by the #932 ns/row microbench to instantiate the tower at
    /// `K = n_primaries` for `K ∈ {8, 16}` (the softmax gate Hessian is `n_atoms³`,
    /// the cost driver the hand path pays per output column).
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

    /// #932 correctness gate: the production packed jet recon
    /// ([`SaeReconstructionRowProgram::reconstruction_all_columns_packed`], gate +
    /// basis jets HOISTED out of the column loop, softmax denom/recip SHARED) and
    /// the per-column packed call must each reproduce the hand path
    /// ([`hand_softmax_column`], the old `row_jets_for_logdet` closed-form softmax
    /// gate Jacobian/Hessian × decoded basis, re-derived per output column) on
    /// value/grad/Hessian — the #932 bit-identity bar. (The ns/row timing
    /// comparison this gate used to precede lives in `bench/`, not in a `#[test]`:
    /// `#[ignore]`d timing benches are banned by `build.rs`.)
    #[test]
    fn recon_jet_matches_hand_path_value_grad_hess() {
        let out_dim = 16;
        let n_basis = 4;
        let inv_tau = 1.3;
        // K=8: 4 atoms × (1 logit + 1 coord) = 8 primaries.
        check_recon_vs_hand::<8>(softmax_fixture_k(4, 1, n_basis, out_dim, inv_tau), inv_tau);
        // K=16: 8 atoms × (1 logit + 1 coord) = 16 primaries.
        check_recon_vs_hand::<16>(softmax_fixture_k(8, 1, n_basis, out_dim, inv_tau), inv_tau);
    }

    /// The structure-compiled softmax executor and the generic packed Taylor jet
    /// are independent implementations of the same row program.  Compare every
    /// reconstruction gradient/Hessian entry across the reduced-logit + coordinate
    /// layout, including highly imbalanced gates where centered softmax moments
    /// are most cancellation-sensitive.
    #[test]
    fn compiled_softmax_schedule_matches_generic_tower_all_channels_932() {
        fn check<const K: usize>(mut program: SaeReconstructionRowProgram, inv_tau: f64) {
            let shift = program
                .logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                * inv_tau;
            let exps: Vec<f64> = program
                .logits
                .iter()
                .map(|&logit| (logit * inv_tau - shift).exp())
                .collect();
            let denominator: f64 = exps.iter().sum();
            program.gate_value = exps.iter().map(|&value| value / denominator).collect();

            let compiled = execute_softmax_row_program(&program, inv_tau, 1.0);
            let generic = program.reconstruction_all_columns_packed::<K>();
            let mut max_abs = 0.0_f64;
            let mut scale = 1.0_f64;
            for (column, tower) in generic.iter().enumerate() {
                for a in 0..K {
                    max_abs = max_abs.max((compiled.first(a)[column] - tower.g()[a]).abs());
                    scale = scale.max(tower.g()[a].abs());
                    for b in 0..K {
                        max_abs =
                            max_abs.max((compiled.second(a, b)[column] - tower.h()[a][b]).abs());
                        scale = scale.max(tower.h()[a][b].abs());
                    }
                }
            }
            assert!(
                max_abs <= 2.0e-12 * scale,
                "compiled softmax schedule vs generic tower max abs {max_abs:e}, scale {scale:e}"
            );
        }

        let inv_tau = 1.3;
        check::<8>(softmax_fixture_k(4, 1, 4, 7, inv_tau), inv_tau);
        check::<16>(softmax_fixture_k(8, 1, 3, 5, inv_tau), inv_tau);

        let mut imbalanced = softmax_fixture_k(4, 1, 4, 7, 4.0);
        imbalanced.logits = vec![35.0, 2.0, -18.0, -40.0];
        check::<8>(imbalanced, 4.0);
    }

    /// Conditioning-aware beta-border derivative oracle.  The compiled gate
    /// channel is checked against both double-double softmax arithmetic and an
    /// independent five-point derivative of `z_k(ℓ) Phi` evaluated entirely in
    /// double-double precision.  The sweep spans balanced and saturated tails and
    /// twelve orders of border scale, so a tiny derivative is judged against the
    /// operation's conditioning rather than an impossible relative-only floor.
    #[test]
    fn softmax_beta_border_gate_derivative_matches_quad_and_fd_across_tails_932() {
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
                            let fd_allowance = 2.0e-12 * condition + 1.0e-300;
                            max_fd_conditioned_error =
                                max_fd_conditioned_error.max(fd_error / fd_allowance);
                            assert!(
                                fd_error <= fd_allowance,
                                "quad five-point derivative error {fd_error:e} > {fd_allowance:e}"
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

    /// The runtime production backend must remain exact beyond the former
    /// 16-primary monomorphization ceiling. A nine-atom softmax row has 18
    /// primaries (nine logits plus nine coordinates), so it could not enter the
    /// old dispatch ladder. Compare its dynamic reconstruction and β-border
    /// channels directly with independently instantiated fixed-size oracles.
    #[test]
    fn runtime_row_jets_match_fixed_oracle_above_old_arity_ceiling_932() {
        const K: usize = 18;
        let mut program = softmax_fixture_k(9, 1, 3, 5, 1.3);
        program.gate = RowGate::PerAtomLogistic { inv_tau: 1.3 };
        program.gate_shift.fill(0.0);
        for atom in 0..program.logits.len() {
            program.gate_value[atom] = 1.0 / (1.0 + (-1.3 * program.logits[atom]).exp());
        }
        assert_eq!(program.n_primaries, K);

        let arena = DynamicJetArena::new();
        let dynamic_columns = program.reconstruction_all_columns_dynamic(&arena);
        let fixed_columns = program.reconstruction_all_columns_packed::<K>();
        assert_eq!(dynamic_columns.len(), fixed_columns.len());
        for (column, (dynamic, fixed)) in dynamic_columns.iter().zip(&fixed_columns).enumerate() {
            let close = |actual: f64, expected: f64, channel: &str| {
                let tolerance = 1.0e-12 * (1.0 + actual.abs().max(expected.abs()));
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "column {column} {channel}: dynamic={actual:.16e}, fixed={expected:.16e}, \
                     tolerance={tolerance:.3e}"
                );
            };
            close(dynamic.value(), fixed.value(), "value");
            for a in 0..K {
                close(dynamic.g()[a], fixed.g()[a], "gradient");
                for b in 0..K {
                    close(dynamic.h_at(a, b), fixed.h()[a][b], "Hessian");
                }
            }
        }

        let channels: Vec<(usize, usize)> = program
            .atoms
            .iter()
            .enumerate()
            .flat_map(|(atom, jet)| (0..jet.n_basis()).map(move |basis| (atom, basis)))
            .collect();
        let dynamic_border = program.beta_border_order1_dynamic(&channels, &arena);
        let fixed_border = program.beta_border_order1_packed::<K>(&channels);
        assert_eq!(dynamic_border.len(), fixed_border.len());
        for (channel, (dynamic, fixed)) in dynamic_border.iter().zip(&fixed_border).enumerate() {
            let tolerance = 1.0e-12 * (1.0 + dynamic.value().abs().max(fixed.value().abs()));
            assert!(
                (dynamic.value() - fixed.value()).abs() <= tolerance,
                "β-border channel {channel} value mismatch"
            );
            for a in 0..K {
                let tolerance = 1.0e-12 * (1.0 + dynamic.g()[a].abs().max(fixed.g()[a].abs()));
                assert!(
                    (dynamic.g()[a] - fixed.g()[a]).abs() <= tolerance,
                    "β-border channel {channel} gradient[{a}] mismatch"
                );
            }
        }
    }

    fn check_recon_vs_hand<const K: usize>(prog: SaeReconstructionRowProgram, inv_tau: f64) {
        let out_dim = prog.out_dim();
        let cols = prog.reconstruction_all_columns_packed::<K>();
        for c in 0..out_dim {
            let hand = hand_softmax_column(&prog, c, inv_tau);
            let h_floor = hand
                .second
                .iter()
                .flatten()
                .fold(0.0_f64, |m, x| m.max(x.abs()));
            // The all-columns (hoisted) path matches hand value + Hessian.
            assert!((cols[c].value() - hand.value).abs() <= 1e-9 * hand.value.abs().max(1.0));
            // The per-column path matches the all-columns path (same kernel, no hoist).
            let percol = prog.reconstruction_column_packed::<K>(c);
            assert!(
                (percol.value() - cols[c].value()).abs() <= 1e-12 * cols[c].value().abs().max(1.0)
            );
            for a in 0..K {
                for b in 0..K {
                    assert!(
                        (cols[c].h()[a][b] - hand.second[a][b]).abs() <= 1e-8 * h_floor.max(1e-12)
                    );
                    assert!(
                        (percol.h()[a][b] - cols[c].h()[a][b]).abs() <= 1e-12 * h_floor.max(1e-12)
                    );
                }
            }
        }
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
            let gate = prog
                .gate_tower::<FixedRuntimeJet<Tower4<6>, 6>>(atom, &())
                .into_inner();
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

    /// The per-atom logistic gate (ordered Beta--Bernoulli/ThresholdGate branch) is diagonal in the
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
            gate_shift: vec![shift],
            gate: RowGate::PerAtomLogistic { inv_tau },
            logit_slot: vec![Some(0)],
            coord_slot: vec![vec![1]],
            fixed_gate_value: Vec::new(),
            n_primaries: 2,
        };
        let gate = prog
            .gate_tower::<FixedRuntimeJet<Tower4<2>, 2>>(0, &())
            .into_inner();
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

    /// #1026/#1033 fixed-gate handling: an atom whose logit is pinned
    /// (`fixed_gate_value[k] = Some(v)`) must gate through the CONSTANT `v` — the
    /// active-routing value the assembly used — with EVERY gate derivative
    /// channel (and hence the logit-slot derivative of the reconstruction)
    /// identically zero, while its COORDINATE derivative remains the plain
    /// contribution scaled by the fixed gate. A sibling FREE atom must be
    /// unaffected (its own logit still moves the gate). This pins the gate-tower
    /// short-circuit that the ungated / frozen-routing row programs rely on.
    #[test]
    fn fixed_gate_atom_is_constant_free_atom_unchanged() {
        const K: usize = 4;
        let inv_tau = 1.3;
        // Atom 0 is PINNED to 0.75 even though its logit (3.0) would give a very
        // different logistic gate; atom 1 is free at logit −0.5.
        let fixed_val = 0.75_f64;
        let free_logit = -0.5_f64;
        let free_shift = 0.1_f64;
        let mk_atom = |phi: f64, dphi: f64, dec: f64| AtomRowBasisJet {
            phi: vec![phi],
            d_phi: vec![vec![dphi]],
            d2_phi: vec![vec![vec![0.0]]],
            decoder: vec![vec![dec]],
            latent_dim: 1,
        };
        // free-atom logistic gate value at (free_logit − free_shift)·inv_tau.
        let x = (free_logit - free_shift) * inv_tau;
        let sigma_free = 1.0 / (1.0 + (-x).exp());
        let prog = SaeReconstructionRowProgram {
            atoms: vec![mk_atom(1.0, 2.0, 1.5), mk_atom(1.0, 0.5, -0.8)],
            // gate_value is the reported (active) gate: fixed for atom 0.
            gate_value: vec![fixed_val, sigma_free],
            logits: vec![3.0, free_logit],
            gate_shift: vec![0.0, free_shift],
            gate: RowGate::PerAtomLogistic { inv_tau },
            // Layout: logit slots 0,1; coord slots 2,3.
            logit_slot: vec![Some(0), Some(1)],
            coord_slot: vec![vec![2], vec![3]],
            fixed_gate_value: vec![Some(fixed_val), None],
            n_primaries: K,
        };

        // Atom 0's gate is a CONSTANT equal to the pinned value: value == fixed,
        // and every gradient / Hessian channel is exactly zero.
        let g0 = prog
            .gate_tower::<FixedRuntimeJet<Tower4<K>, K>>(0, &())
            .into_inner();
        assert!(
            (g0.v - fixed_val).abs() < 1e-15,
            "fixed gate value {}",
            g0.v
        );
        for a in 0..K {
            assert_eq!(g0.g[a], 0.0, "fixed gate ∂/∂p{a} must be exactly 0");
            for b in 0..K {
                assert_eq!(g0.h[a][b], 0.0, "fixed gate ∂²/∂p{a}∂p{b} must be 0");
            }
        }

        // Atom 1 (free) is unchanged: it reproduces the logistic gate and its own
        // logit derivative is nonzero.
        let g1 = prog
            .gate_tower::<FixedRuntimeJet<Tower4<K>, K>>(1, &())
            .into_inner();
        assert!((g1.v - sigma_free).abs() < 1e-12, "free gate value");
        let d1 = sigma_free * (1.0 - sigma_free) * inv_tau;
        assert!((g1.g[1] - d1).abs() < 1e-9, "free gate σ'");
        assert!(g1.g[1].abs() > 1e-6, "free gate must depend on its logit");

        // The reconstruction column: atom-0's LOGIT-slot derivative is zero (the
        // gate is constant), while its COORDINATE-slot derivative is the plain
        // decoded slope scaled by the FIXED gate (0.75 · d(decoded_0)/dt).
        let col = prog.reconstruction_column::<K>(0);
        assert_eq!(
            col.g[0], 0.0,
            "fixed atom's logit-slot reconstruction derivative must be exactly 0"
        );
        let decoded0_slope = 2.0 * 1.5; // d_phi · decoder
        assert!(
            (col.g[2] - fixed_val * decoded0_slope).abs() < 1e-12,
            "fixed atom coord derivative must use gate {fixed_val}: got {}",
            col.g[2]
        );
        // Value = fixed·decoded_0 + free·decoded_1.
        let expected_v = fixed_val * (1.0 * 1.5) + sigma_free * (1.0 * -0.8);
        assert!((col.v - expected_v).abs() < 1e-12, "reconstruction value");
    }

    /// #932 cutover pin: the PRODUCTION packed [`Order2`] reconstruction path
    /// (`reconstruction_column_packed`) is BIT-IDENTICAL on the
    /// value/gradient/Hessian channels to the dense [`Tower4`] oracle
    /// (`reconstruction_column`) — the same channels the arrow-Schur logdet
    /// consumer reads — for every output column. The Order2 path never
    /// materialises `t3`/`t4`, but its `(v, g, H)` must match the dense tower's
    /// order-≤2 channels to ≤1e-12 (they share the `Tower2` arithmetic), so the
    /// cutover changes only cost, not result.
    #[test]
    fn order2_reconstruction_matches_tower_value_grad_hessian() {
        for tau in [0.9_f64, 1.3, 2.1] {
            let (prog, _inv_tau) = softmax_fixture(tau);
            for out_col in 0..prog.out_dim() {
                let packed = prog.reconstruction_column_packed::<6>(out_col);
                let tower = prog.reconstruction_column::<6>(out_col);
                let g = packed.g();
                let h = packed.h();
                let band = |x: f64| 1e-12 + 1e-12 * x.abs();
                assert!(
                    (packed.value() - tower.v).abs() <= band(tower.v),
                    "col {out_col} value: order2 {} vs tower {}",
                    packed.value(),
                    tower.v
                );
                for a in 0..6 {
                    assert!(
                        (g[a] - tower.g[a]).abs() <= band(tower.g[a]),
                        "col {out_col} g[{a}]: order2 {} vs tower {}",
                        g[a],
                        tower.g[a]
                    );
                    for b in 0..6 {
                        assert!(
                            (h[a][b] - tower.h[a][b]).abs() <= band(tower.h[a][b]),
                            "col {out_col} h[{a}][{b}]: order2 {} vs tower {}",
                            h[a][b],
                            tower.h[a][b]
                        );
                    }
                }
            }
        }
    }

    /// #932 cutover pin for the β border channel: the packed [`Order2`]
    /// `beta_border_tower_packed` matches the dense [`Tower4`]
    /// `beta_border_tower` on the value (`beta`) and gradient (`beta_deriv` /
    /// `beta_l_deriv`) channels the consumer reads, to ≤1e-12.
    #[test]
    fn order2_beta_border_matches_tower_value_grad() {
        let (prog, _inv_tau) = softmax_fixture(1.1);
        for atom in 0..prog.atoms.len() {
            for basis_col in 0..prog.atoms[atom].n_basis() {
                let packed = prog.beta_border_tower_packed::<6>(atom, basis_col);
                let tower = prog.beta_border_tower::<6>(atom, basis_col);
                let g = packed.g();
                let band = |x: f64| 1e-12 + 1e-12 * x.abs();
                assert!(
                    (packed.value() - tower.v).abs() <= band(tower.v),
                    "atom {atom} b {basis_col} value: order2 {} vs tower {}",
                    packed.value(),
                    tower.v
                );
                for a in 0..6 {
                    assert!(
                        (g[a] - tower.g[a]).abs() <= band(tower.g[a]),
                        "atom {atom} b {basis_col} g[{a}]: order2 {} vs tower {}",
                        g[a],
                        tower.g[a]
                    );
                }
            }
        }
    }

    /// #932 perf pin: the gate-shared `all_gates` produces gate jets
    /// BIT-IDENTICAL to the per-atom `gate_tower` — sharing the softmax
    /// denominator / reciprocal across atoms (K exps + 1 recip instead of
    /// K² + K) changes only which redundant work is elided, not the result
    /// (`ζ_k = exp_k · recip(denom)` is the same product, same Leibniz order).
    #[test]
    fn shared_all_gates_bit_identical_to_per_atom_gate_tower() {
        for tau in [0.9_f64, 1.3, 2.1] {
            let (prog, _inv_tau) = softmax_fixture(tau);
            let all: Vec<Order2<6>> = prog
                .all_gates::<FixedRuntimeJet<Order2<6>, 6>>(&())
                .into_iter()
                .map(FixedRuntimeJet::into_inner)
                .collect();
            assert_eq!(all.len(), prog.gate_value.len());
            for atom in 0..prog.gate_value.len() {
                let per = prog
                    .gate_tower::<FixedRuntimeJet<Order2<6>, 6>>(atom, &())
                    .into_inner();
                assert_eq!(all[atom].value(), per.value(), "atom {atom} value");
                for a in 0..6 {
                    assert_eq!(all[atom].g()[a], per.g()[a], "atom {atom} g[{a}]");
                    for b in 0..6 {
                        assert_eq!(
                            all[atom].h()[a][b],
                            per.h()[a][b],
                            "atom {atom} h[{a}][{b}]"
                        );
                    }
                }
            }
        }
    }

    /// #932 perf pin: the gate/basis-HOISTED + denominator-SHARED all-columns
    /// reconstruction (`reconstruction_all_columns_packed`) is BIT-IDENTICAL to
    /// calling `reconstruction_column_packed(c)` per column — the hoist + share
    /// removes only redundant gate/basis/denominator recomputation, not any
    /// arithmetic. Every value/grad/Hessian channel must match exactly (==),
    /// since the Leibniz products are the same in the same order.
    #[test]
    fn hoisted_all_columns_bit_identical_to_per_column() {
        for tau in [0.9_f64, 1.3, 2.1] {
            let (prog, _inv_tau) = softmax_fixture(tau);
            let all = prog.reconstruction_all_columns_packed::<6>();
            assert_eq!(all.len(), prog.out_dim());
            for out_col in 0..prog.out_dim() {
                let per = prog.reconstruction_column_packed::<6>(out_col);
                let ah = all[out_col];
                assert_eq!(ah.value(), per.value(), "col {out_col} value");
                for a in 0..6 {
                    assert_eq!(ah.g()[a], per.g()[a], "col {out_col} g[{a}]");
                    for b in 0..6 {
                        assert_eq!(ah.h()[a][b], per.h()[a][b], "col {out_col} h[{a}][{b}]");
                    }
                }
            }
        }
    }

    /// Build four softmax-aligned row programs that differ ONLY in their per-row
    /// numeric data (logits, basis values, decoder), keeping the layout
    /// (slots / dims / temperature) identical so they are 4-row SIMD-batchable.
    fn softmax_batch_fixture(inv_tau: f64) -> [SaeReconstructionRowProgram; LANES] {
        let n_basis = 3;
        let out_dim = 4;
        let mk = |row_seed: f64| {
            let mk_atom = |seed: f64| {
                let phi: Vec<f64> = (0..n_basis)
                    .map(|b| 0.3 + 0.2 * (b as f64 + seed) + 0.11 * row_seed)
                    .collect();
                let d_phi: Vec<Vec<f64>> = (0..n_basis)
                    .map(|b| {
                        (0..2)
                            .map(|axis| {
                                0.1 * (b as f64 + 1.0) - 0.05 * axis as f64
                                    + 0.03 * seed
                                    + 0.017 * row_seed
                            })
                            .collect()
                    })
                    .collect();
                let d2_phi: Vec<Vec<Vec<f64>>> = (0..n_basis)
                    .map(|b| {
                        (0..2)
                            .map(|a| {
                                (0..2)
                                    .map(|bb| {
                                        0.02 * (b as f64 + 1.0)
                                            + 0.01 * (a as f64)
                                            + 0.01 * (bb as f64)
                                            + 0.004 * seed
                                            + 0.003 * row_seed
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                let decoder: Vec<Vec<f64>> = (0..n_basis)
                    .map(|b| {
                        (0..out_dim)
                            .map(|c| {
                                0.5 - 0.1 * (b as f64)
                                    + 0.07 * (c as f64)
                                    + 0.02 * seed
                                    + 0.009 * row_seed
                            })
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
            let logits = vec![0.4 + 0.21 * row_seed, -0.7 + 0.13 * row_seed];
            let e: Vec<f64> = logits.iter().map(|&l| (l * inv_tau).exp()).collect();
            let s: f64 = e.iter().sum();
            let gate_value: Vec<f64> = e.iter().map(|&v| v / s).collect();
            SaeReconstructionRowProgram {
                atoms: vec![mk_atom(0.0), mk_atom(1.0)],
                gate_value,
                logits,
                gate_shift: vec![0.0, 0.0],
                gate: RowGate::Softmax { inv_tau },
                logit_slot: vec![Some(0), Some(1)],
                coord_slot: vec![vec![2, 3], vec![4, 5]],
                fixed_gate_value: Vec::new(),
                n_primaries: 6,
            }
        };
        [mk(0.0), mk(1.0), mk(2.0), mk(3.0)]
    }

    /// SIMD-batch bit-identity oracle: `reconstruction_all_columns_batch4` lane
    /// `i` is `to_bits`-identical to the scalar `reconstruction_all_columns_packed`
    /// on row `i`, across many temperatures and randomized per-row data
    /// (≥2000 channel comparisons). The 4-row SIMD pass changes only how many
    /// rows share one instruction stream, never the arithmetic.
    #[test]
    fn batch4_reconstruction_bit_identical_to_per_row() {
        let mut comparisons = 0usize;
        for tau in [0.7_f64, 0.9, 1.1, 1.3, 1.7, 2.1, 2.9] {
            let rows = softmax_batch_fixture(tau);
            let refs = [&rows[0], &rows[1], &rows[2], &rows[3]];
            let batch = SaeReconstructionRowProgram::reconstruction_all_columns_batch4::<6>(refs)
                .expect("softmax-aligned rows must batch");
            for lane in 0..LANES {
                let per = rows[lane].reconstruction_all_columns_packed::<6>();
                assert_eq!(per.len(), batch[lane].len());
                for (c, (b, p)) in batch[lane].iter().zip(per.iter()).enumerate() {
                    assert_eq!(
                        b.value().to_bits(),
                        p.value().to_bits(),
                        "tau {tau} lane {lane} col {c} value"
                    );
                    let (bg, pg) = (b.g(), p.g());
                    let (bh, ph) = (b.h(), p.h());
                    for a in 0..6 {
                        assert_eq!(
                            bg[a].to_bits(),
                            pg[a].to_bits(),
                            "lane {lane} col {c} g[{a}]"
                        );
                        for d in 0..6 {
                            assert_eq!(
                                bh[a][d].to_bits(),
                                ph[a][d].to_bits(),
                                "lane {lane} col {c} h[{a}][{d}]"
                            );
                            comparisons += 1;
                        }
                    }
                }
            }
        }
        assert!(comparisons >= 2000, "oracle ran {comparisons} comparisons");
    }

    /// SIMD-batch bit-identity oracle for the β-border first-order path:
    /// `beta_border_order1_batch4` lane `i` is `to_bits`-identical to
    /// `beta_border_order1_packed` on row `i`.
    #[test]
    fn batch4_beta_border_bit_identical_to_per_row() {
        let mut comparisons = 0usize;
        for tau in [0.7_f64, 0.9, 1.1, 1.3, 1.7, 2.1, 2.9] {
            let rows = softmax_batch_fixture(tau);
            let refs = [&rows[0], &rows[1], &rows[2], &rows[3]];
            let mut chans: Vec<(usize, usize)> = Vec::new();
            for atom in 0..rows[0].atoms.len() {
                for b in 0..rows[0].atoms[atom].n_basis() {
                    chans.push((atom, b));
                }
            }
            chans.push(chans[0]); // repeat to exercise gate-cache reuse
            let batch = SaeReconstructionRowProgram::beta_border_order1_batch4::<6>(refs, &chans)
                .expect("softmax-aligned rows must batch");
            for lane in 0..LANES {
                let per = rows[lane].beta_border_order1_packed::<6>(&chans);
                assert_eq!(per.len(), batch[lane].len());
                for (i, (b, p)) in batch[lane].iter().zip(per.iter()).enumerate() {
                    assert_eq!(
                        b.value().to_bits(),
                        p.value().to_bits(),
                        "lane {lane} chan {i} v"
                    );
                    let (bg, pg) = (b.g(), p.g());
                    for a in 0..6 {
                        assert_eq!(
                            bg[a].to_bits(),
                            pg[a].to_bits(),
                            "lane {lane} chan {i} g[{a}]"
                        );
                        comparisons += 1;
                    }
                }
            }
        }
        assert!(comparisons >= 1000, "oracle ran {comparisons} comparisons");
    }

    /// A non-softmax (per-atom logistic) batch must DECLINE (return `None`) so the
    /// caller falls back to the scalar per-row path — the logistic branch is
    /// per-row data-dependent and not lane-uniform.
    #[test]
    fn batch4_declines_non_softmax() {
        let inv_tau = 1.1;
        let mk = || SaeReconstructionRowProgram {
            atoms: vec![AtomRowBasisJet {
                phi: vec![1.0],
                d_phi: vec![vec![0.0]],
                d2_phi: vec![vec![vec![0.0]]],
                decoder: vec![vec![1.0]],
                latent_dim: 1,
            }],
            gate_value: vec![0.6],
            logits: vec![0.6],
            gate_shift: vec![0.2],
            gate: RowGate::PerAtomLogistic { inv_tau },
            logit_slot: vec![Some(0)],
            coord_slot: vec![vec![1]],
            fixed_gate_value: Vec::new(),
            n_primaries: 2,
        };
        let rows = [mk(), mk(), mk(), mk()];
        let refs = [&rows[0], &rows[1], &rows[2], &rows[3]];
        assert!(
            SaeReconstructionRowProgram::reconstruction_all_columns_batch4::<2>(refs).is_none()
        );
    }

    /// #932 perf pin: the gate-HOISTED batched β border jets
    /// (`beta_border_towers_packed`) are BIT-IDENTICAL to per-channel
    /// `beta_border_tower_packed`, including when several channels share an atom
    /// (the gate-cache reuse path).
    #[test]
    fn hoisted_beta_border_bit_identical_to_per_channel() {
        let (prog, _inv_tau) = softmax_fixture(1.1);
        // Build a channel list that repeats atoms (exercises the gate cache).
        let mut chans: Vec<(usize, usize)> = Vec::new();
        for atom in 0..prog.atoms.len() {
            for basis_col in 0..prog.atoms[atom].n_basis() {
                chans.push((atom, basis_col));
            }
        }
        // Duplicate the first atom's channels at the end to force cache reuse.
        if let Some(&first) = chans.first() {
            chans.push(first);
        }
        let batched = prog.beta_border_towers_packed::<6>(&chans);
        assert_eq!(batched.len(), chans.len());
        for (i, &(atom, basis_col)) in chans.iter().enumerate() {
            let per = prog.beta_border_tower_packed::<6>(atom, basis_col);
            let b = batched[i];
            assert_eq!(b.value(), per.value(), "chan {i} value");
            for a in 0..6 {
                assert_eq!(b.g()[a], per.g()[a], "chan {i} g[{a}]");
            }
        }
    }
}
