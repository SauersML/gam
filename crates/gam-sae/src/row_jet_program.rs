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

    fn out_dim(&self) -> usize {
        self.decoder.first().map_or(0, Vec::len)
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
mod tests_schedule_source {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    // The scalar algebra (`value`/`add`/…) these tests read on concrete scalars
    // lives on the shared `JetField` base now (JetScalar: JetField); JetScalar
    // itself is no longer named here, so only JetField is imported.
    use gam_math::nested_dual::JetField;

    /// Replicate the historical hand path formerly used by `row_jets_for_logdet`
    /// for the reconstruction `first`/`second` channels of one output column.
    /// It starts from the same atom jets and explicit softmax derivatives but is
    /// independent of both the generic tower and the production compiled
    /// schedule. Agreement makes it the #932 historical oracle for the SAE row
    /// program (the analog of the survival RowKernel oracle).
    struct HandChannels {
        first: Vec<f64>,       // [primary]
        second: Vec<Vec<f64>>, // [primary][primary]
        value: f64,
    }

    /// Softmax gate first/second derivatives wrt logit primaries, term-for-term
    /// the retired `gate_derivatives_for_row` softmax branch.
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

    /// #932 correctness gate: the generic packed jet reconstruction
    /// ([`SaeReconstructionRowProgram::reconstruction_all_columns_packed`], gate +
    /// basis jets HOISTED out of the column loop, softmax denom/recip SHARED) and
    /// and the per-column packed call must each reproduce the historical hand path
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

}
