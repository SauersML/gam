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

