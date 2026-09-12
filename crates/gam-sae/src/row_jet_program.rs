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
//! border channels from this semantic program. Rows use the borrowed
//! `SaeOrder2RowProgramSource` and the gate-specific structure-compiled schedules
//! (`execute_softmax_row_program`, `execute_independent_logistic_row_program`);
//! the bounded batch seam evaluates the same centered-moment identities on CPU
//! or CUDA. The #1006 third-order logdet adjoint `Γ_a = tr(H⁻¹ ∂H/∂θ_a)`
//! consumes those shared channels.
//!
//! # The basis as a local jet
//!
//! The production assembly does NOT re-evaluate the manifold basis `Φ` as a
//! function of perturbed coordinates: it consumes the precomputed jet tensors
//! `(Φ, ∂Φ/∂t, ∂²Φ/∂t²)` evaluated at the current `t`. The reconstruction's
//! dependence on `t` is therefore *defined* by those tensors — the local
//! quadratic Taylor model of `Φ` about the current point, which the compiled
//! schedules lower through centered moments.

// ─────────────────────────────────────────────────────────────────────────
// STRUCTURE-COMPILED SOFTMAX ROW PROGRAM
//
// A dense generic jet represents every primary in every intermediate, including
// the structural-zero cross-atom coordinate blocks.  The interface below is the
// same row program as a borrowed semantic source: gate masses, decoded component
// values, their coordinate jets, and beta-border basis channels.  The executor
// compiles that dependency graph into the nonzero order-2 blocks.  There is one
// softmax-moment definition, shared by reconstruction, coordinate cross terms,
// and beta borders.

/// One primary in the sparse dependency graph of an SAE reconstruction row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SaeRowPrimary {
    Logit { atom: usize },
    Coord { atom: usize, axis: usize },
}

/// Borrowed semantic input to the structure-compiled softmax row program.
///
/// Production implements this directly over ndarray views, so compiling a row
/// does not clone its basis/decoder tensors.
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

    /// Gate-value source for the softmax-moment oracle below. `gate_first` reads
    /// only the gate masses, so every other channel refuses by naming what was
    /// requested.
    struct GateValues(Vec<f64>);

    impl SaeOrder2RowProgramSource for GateValues {
        fn n_atoms(&self) -> usize {
            self.0.len()
        }

        fn out_dim(&self) -> usize {
            0
        }

        fn n_primaries(&self) -> usize {
            self.0.len()
        }

        fn primary(&self, slot: usize) -> SaeRowPrimary {
            SaeRowPrimary::Logit { atom: slot }
        }

        fn gate_value(&self, atom: usize) -> f64 {
            self.0[atom]
        }

        fn atom_is_active(&self, atom: usize) -> bool {
            atom < self.0.len()
        }

        fn fill_decoded(&self, atom: usize, out: &mut [f64]) {
            panic!(
                "gate-value oracle has no decoded components, but atom {atom} was requested into a \
                 {}-wide buffer",
                out.len()
            )
        }

        fn fill_decoded_first(&self, atom: usize, axis: usize, out: &mut [f64]) {
            panic!(
                "gate-value oracle has no decoded components, but atom {atom} axis {axis} was \
                 requested into a {}-wide buffer",
                out.len()
            )
        }

        fn fill_decoded_second(&self, atom: usize, axis_a: usize, axis_b: usize, out: &mut [f64]) {
            panic!(
                "gate-value oracle has no decoded components, but atom {atom} axes {axis_a},{axis_b} \
                 were requested into a {}-wide buffer",
                out.len()
            )
        }

        fn n_beta_borders(&self) -> usize {
            0
        }

        fn beta_border_atom(&self, border: usize) -> usize {
            panic!("gate-value oracle has no beta borders, but border {border} was requested")
        }

        fn beta_border_basis_value(&self, border: usize) -> f64 {
            panic!("gate-value oracle has no beta borders, but border {border} was requested")
        }

        fn beta_border_basis_first(&self, border: usize, axis: usize) -> f64 {
            panic!(
                "gate-value oracle has no beta borders, but border {border} axis {axis} was requested"
            )
        }

        fn beta_border_output(&self, border: usize) -> &[f64] {
            panic!("gate-value oracle has no beta borders, but border {border} was requested")
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
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) * inv_tau;
                let exps: Vec<f64> = logits
                    .iter()
                    .map(|&value| (value * inv_tau - shift).exp())
                    .collect();
                let denominator: f64 = exps.iter().sum();
                let program = GateValues(exps.iter().map(|&value| value / denominator).collect());
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
