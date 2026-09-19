//! Width-independent Hessian-trace third derivatives (gam#2998).
//!
//! The LAML ρ-gradient needs, per row, `g_c = Σ_ab G_ab ∂³f/∂x_a∂x_b∂x_c` for
//! a row gram `G`. Seeding one [`DynamicOneSeedBatch`](crate::jet_scalar::DynamicOneSeedBatch)
//! lane per output axis carries a full `r × r` Hessian in every lane, so the
//! row program costs `O(r³)` per node. D³f is symmetric in `(a, b)`, so only
//! `S = (G + Gᵀ)/2` matters, and with `S = Σ_k λ_k v_k v_kᵀ`
//!
//! ```text
//! g = Σ_k λ_k ∇_x (v_kᵀ H(x) v_k) = Σ_k λ_k D³f[v_k, v_k, ·].
//! ```
//!
//! [`DynamicTraceJet`] evaluates every `∇_x (v_kᵀ H v_k)` in one pass. It is
//! the truncated algebra `ℝ[t, x_1..x_r] / (t³, x_i x_j)`: one shared
//! value-plus-gradient base, and per direction lane the first and second
//! `t`-derivatives, each again value-plus-gradient. Seeding primary `i` as
//! `x_i + δx_i + t v_k[i]` makes lane `k`'s second derivative
//! `v_kᵀ H v_k + ∇(v_kᵀ H v_k)·δx`. A jet is `(1 + r)(1 + 2L)` floats, so a
//! node costs `O(L r)` rather than `O(L r²)`, and `L ≤ r`.
//!
//! The quotient ideal is homogeneous and the algebra's maximal ideal is
//! nilpotent of order four (any degree-four monomial holds `t³` or an
//! `x_i x_j`), so the filtered implicit lift is exact after three steps, as for
//! the one-seed jets.

use crate::jet_scalar::{DynamicJetArena, RuntimeJetScalar};

/// Reusable arena plus lane count for [`DynamicTraceJet`] evaluations.
///
/// A caller resets it once per row. The bump retains its largest chunk, so
/// warmed rows do not return to the global allocator.
#[derive(Debug)]
pub struct TraceJetWorkspace {
    arena: DynamicJetArena,
    lanes: usize,
}

impl TraceJetWorkspace {
    /// A workspace for `lanes` simultaneous directions.
    #[must_use]
    pub fn new(lanes: usize) -> Self {
        Self {
            arena: DynamicJetArena::new(),
            lanes,
        }
    }

    /// Reclaim all jet storage and select the next evaluation's lane count.
    pub fn reset(&mut self, lanes: usize) {
        self.arena.reset();
        self.lanes = lanes;
    }

    /// Bytes retained by the bump allocator after the largest evaluation.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }

    /// Allocate a primary array in the same arena as every jet channel.
    #[inline(always)]
    pub fn alloc_slice_fill_with<T>(&self, len: usize, fill: impl FnMut(usize) -> T) -> &mut [T] {
        self.arena.alloc_slice_fill_with(len, fill)
    }
}

/// `seed` value of a jet that is not a seeded primary.
const NOT_A_SEED: usize = usize::MAX;

/// Runtime jet for batched `∇_x (v_kᵀ H v_k)`; see the module docs.
///
/// Storage is one arena slice `[base | P_0 | Q_0 | P_1 | Q_1 | …]`. Every block
/// is `1 + dimension` wide: a value, then its gradient. `P_k` and `Q_k` are
/// lane `k`'s first and second `t`-derivatives.
#[derive(Clone, Copy, Debug)]
pub struct DynamicTraceJet<'arena> {
    data: &'arena [f64],
    dimension: usize,
    /// The axis when this jet is exactly a seeded primary: base gradient
    /// `e_axis`, every lane's first-derivative gradient zero and every lane's
    /// second derivative zero. Linear forms over seeds then read one value per
    /// lane instead of streaming whole blocks.
    seed: usize,
    arena: &'arena DynamicJetArena,
}

/// `out += scale · (a ⊗ b)` for value-plus-gradient blocks.
#[inline(always)]
fn block_mul_add(out: &mut [f64], a: &[f64], b: &[f64], scale: f64) {
    let (a0, b0) = (a[0] * scale, b[0] * scale);
    out[0] += a0 * b[0];
    for ((o, &ai), &bi) in out[1..].iter_mut().zip(&a[1..]).zip(&b[1..]) {
        *o += a0 * bi + b0 * ai;
    }
}

/// `out += scale · x` over whole slices.
#[inline(always)]
fn axpy(out: &mut [f64], scale: f64, x: &[f64]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += scale * v;
    }
}

impl<'arena> DynamicTraceJet<'arena> {
    /// Seed primary `axis` at `x`, with lane `k` moving along
    /// `direction_at(k)`.
    #[inline]
    #[must_use]
    pub fn seed_directions(
        x: f64,
        axis: usize,
        dimension: usize,
        workspace: &'arena TraceJetWorkspace,
        mut direction_at: impl FnMut(usize) -> f64,
    ) -> Self {
        assert!(axis < dimension, "trace jet seed axis {axis} >= dimension {dimension}");
        let width = dimension + 1;
        let lanes = workspace.lanes;
        let data = workspace
            .arena
            .alloc_slice_fill_with(width * (1 + 2 * lanes), |_| 0.0);
        data[0] = x;
        data[1 + axis] = 1.0;
        for lane in 0..lanes {
            data[width * (1 + 2 * lane)] = direction_at(lane);
        }
        Self {
            data,
            dimension,
            seed: axis,
            arena: &workspace.arena,
        }
    }

    /// Number of direction lanes.
    #[inline(always)]
    #[must_use]
    pub fn lanes(&self) -> usize {
        (self.data.len() / self.width() - 1) / 2
    }

    /// `v_kᵀ H v_k` for lane `k`.
    #[inline(always)]
    #[must_use]
    pub fn second_directional(&self, lane: usize) -> f64 {
        self.data[self.second_offset(lane)]
    }

    /// `∇_x (v_kᵀ H v_k) = D³f[v_k, v_k, ·]` for lane `k`.
    #[inline(always)]
    #[must_use]
    pub fn second_directional_gradient(&self, lane: usize) -> &[f64] {
        let start = self.second_offset(lane) + 1;
        &self.data[start..start + self.dimension]
    }

    #[inline(always)]
    fn width(&self) -> usize {
        self.dimension + 1
    }

    #[inline(always)]
    fn first_offset(&self, lane: usize) -> usize {
        self.width() * (1 + 2 * lane)
    }

    #[inline(always)]
    fn second_offset(&self, lane: usize) -> usize {
        self.width() * (2 + 2 * lane)
    }

    #[inline(always)]
    fn assert_compatible(&self, other: &Self) {
        assert_eq!(self.dimension, other.dimension, "trace jet dimension mismatch");
        assert_eq!(self.data.len(), other.data.len(), "trace jet lane mismatch");
    }

    #[inline(always)]
    fn zeros(&self) -> &'arena mut [f64] {
        self.arena.alloc_slice_fill_with(self.data.len(), |_| 0.0)
    }

    #[inline(always)]
    fn copied(&self) -> &'arena mut [f64] {
        let data = self.data;
        self.arena.alloc_slice_fill_with(data.len(), |index| data[index])
    }

    /// The zero jet and its storage, for kernels that accumulate in place.
    #[inline(always)]
    fn zero_jet(
        dimension: usize,
        workspace: &'arena TraceJetWorkspace,
    ) -> (Self, &'arena mut [f64]) {
        let data = workspace
            .arena
            .alloc_slice_fill_with((dimension + 1) * (1 + 2 * workspace.lanes), |_| 0.0);
        let shape = Self {
            data: &[],
            dimension,
            seed: NOT_A_SEED,
            arena: &workspace.arena,
        };
        (shape, data)
    }

    #[inline(always)]
    fn from_data(&self, data: &'arena [f64]) -> Self {
        Self {
            data,
            dimension: self.dimension,
            seed: NOT_A_SEED,
            arena: self.arena,
        }
    }

    /// `out += self ⊗ other`: base `X⊗Y`, first `X⊗S + P⊗Y`, second
    /// `X⊗T + 2P⊗S + Q⊗Y`.
    #[inline(always)]
    fn mul_add_into(&self, other: &Self, out: &mut [f64]) {
        let width = self.width();
        let (u, v) = (self.data, other.data);
        let (x, y) = (&u[..width], &v[..width]);
        block_mul_add(&mut out[..width], x, y, 1.0);
        for lane in 0..self.lanes() {
            let first = self.first_offset(lane);
            let second = first + width;
            let (p, q) = (&u[first..second], &u[second..second + width]);
            let (s, t) = (&v[first..second], &v[second..second + width]);
            let (out_p, out_q) = out[first..second + width].split_at_mut(width);
            block_mul_add(out_p, x, s, 1.0);
            block_mul_add(out_p, p, y, 1.0);
            block_mul_add(out_q, x, t, 1.0);
            block_mul_add(out_q, p, s, 2.0);
            block_mul_add(out_q, q, y, 1.0);
        }
    }

    /// `out += f(self)` from `f`'s derivative stack at `self.value()`.
    #[inline(always)]
    fn compose_add_into(&self, d: [f64; 5], out: &mut [f64]) {
        let width = self.width();
        let data = self.data;
        let xg = &data[1..width];
        out[0] += d[0];
        axpy(&mut out[1..width], d[1], xg);
        for lane in 0..self.lanes() {
            let first = self.first_offset(lane);
            let second = first + width;
            let (p, pg) = (data[first], &data[first + 1..second]);
            let (q, qg) = (data[second], &data[second + 1..second + width]);
            out[first] += d[1] * p;
            let out_pg = &mut out[first + 1..second];
            axpy(out_pg, d[1], pg);
            axpy(out_pg, d[2] * p, xg);
            out[second] += d[1] * q + d[2] * p * p;
            let out_qg = &mut out[second + 1..second + width];
            axpy(out_qg, d[1], qg);
            axpy(out_qg, 2.0 * d[2] * p, pg);
            axpy(out_qg, d[2] * q + d[3] * p * p, xg);
        }
    }

    /// `Σ_i weights[i] · seeds[i]` when every input is a seeded primary: a
    /// value and a gradient entry per input, and one first-derivative value
    /// per lane.
    #[inline(always)]
    fn seed_linear_combination_into(inputs: &[Self], weights: &[f64], out: &mut [f64]) {
        let first = inputs[0];
        for (input, &weight) in inputs.iter().zip(weights) {
            out[0] += weight * input.data[0];
            out[1 + input.seed] += weight;
            for lane in 0..first.lanes() {
                let offset = first.first_offset(lane);
                out[offset] += weight * input.data[offset];
            }
        }
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicTraceJet<'arena> {
    type Workspace = TraceJetWorkspace;

    #[inline]
    fn constant(c: f64, dimension: usize, workspace: &'arena TraceJetWorkspace) -> Self {
        let data = workspace
            .arena
            .alloc_slice_fill_with((dimension + 1) * (1 + 2 * workspace.lanes), |_| 0.0);
        data[0] = c;
        Self {
            data,
            dimension,
            seed: NOT_A_SEED,
            arena: &workspace.arena,
        }
    }

    #[inline]
    fn variable(
        x: f64,
        axis: usize,
        dimension: usize,
        workspace: &'arena TraceJetWorkspace,
    ) -> Self {
        Self::seed_directions(x, axis, dimension, workspace, |_| 0.0)
    }

    #[inline]
    fn constant_like(&self, c: f64) -> Self {
        let data = self.zeros();
        data[0] = c;
        self.from_data(data)
    }

    #[inline]
    fn with_value(&self, value: f64) -> Self {
        let data = self.copied();
        data[0] = value;
        // Only the value moved, so a seed stays a seed.
        Self {
            seed: self.seed,
            ..self.from_data(data)
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.data[0]
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let data = self.copied();
        axpy(data, 1.0, other.data);
        self.from_data(data)
    }

    #[inline]
    fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let data = self.copied();
        axpy(data, -1.0, other.data);
        self.from_data(data)
    }

    #[inline]
    fn mul(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let data = self.zeros();
        self.mul_add_into(other, data);
        self.from_data(data)
    }

    #[inline]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        self.assert_compatible(right);
        self.assert_compatible(addend);
        let data = addend.copied();
        self.mul_add_into(right, data);
        self.from_data(data)
    }

    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    #[inline]
    fn scale(&self, s: f64) -> Self {
        let source = self.data;
        let data = self
            .arena
            .alloc_slice_fill_with(source.len(), |index| s * source[index]);
        self.from_data(data)
    }

    #[inline]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let data = self.zeros();
        self.compose_add_into(d, data);
        self.from_data(data)
    }

    #[inline]
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        workspace: &'arena TraceJetWorkspace,
    ) -> Self {
        assert_eq!(inputs.len(), weights.len(), "linear-combination dimension mismatch");
        let (out, data) = Self::zero_jet(dimension, workspace);
        for input in inputs {
            assert!(
                input.dimension == dimension && input.data.len() == data.len(),
                "trace jet inputs must share lanes and dimension"
            );
        }
        if inputs.iter().all(|input| input.seed != NOT_A_SEED) && !inputs.is_empty() {
            Self::seed_linear_combination_into(inputs, weights, data);
        } else {
            for (input, &weight) in inputs.iter().zip(weights) {
                axpy(data, weight, input.data);
            }
        }
        out.from_data(data)
    }

    /// `addend + Σ_i L_i · f_i(R)` at one shared point `R`.
    ///
    /// Around `x = R.value()`, `f_i(R) = Σ_{m≤3} f_i^{(m)}(x) N^m/m!` exactly,
    /// with `N = R − x` (`N⁴ = 0`). So the sum is `Σ_m (N^m/m!) · C_m` with
    /// `C_m = Σ_i f_i^{(m)}(x) L_i`. When every `L_i` is a seeded primary, as
    /// the BMS link-deviation coefficients are, each `C_m` is a value, a sparse
    /// gradient and one first-derivative value per lane, and the four products
    /// collapse to the closed form below: `O(|L| + lanes · r)` instead of
    /// `|L|` full compositions and products.
    #[inline]
    fn weighted_compose_sum(
        lefts: &[Self],
        right: &Self,
        derivative_stacks: &[[f64; 5]],
        addend: &Self,
    ) -> Self {
        assert_eq!(
            lefts.len(),
            derivative_stacks.len(),
            "weighted compose sum needs one derivative stack per left factor"
        );
        addend.assert_compatible(right);
        for left in lefts {
            addend.assert_compatible(left);
        }
        if !lefts.iter().all(|left| left.seed != NOT_A_SEED) {
            let mut sum = *addend;
            for (left, stack) in lefts.iter().zip(derivative_stacks) {
                sum = left.multiply_add(&right.compose_unary(*stack), &sum);
            }
            return sum;
        }
        let width = right.width();
        let dimension = right.dimension;
        let lanes = right.lanes();
        // c[m] = Σ_i d_i[m] x_i, g[m] = Σ_i d_i[m] e_{axis_i} and
        // pi[m][k] = Σ_i d_i[m] (lane-k direction of primary i). `N³/6` has only
        // a second-derivative gradient, `p² ∇R`, so `C_3` enters through `c[3]`.
        let mut c = [0.0; 4];
        let g = right.arena.alloc_slice_fill_with(3 * dimension, |_| 0.0);
        let pi = right.arena.alloc_slice_fill_with(3 * lanes, |_| 0.0);
        for (left, stack) in lefts.iter().zip(derivative_stacks) {
            for m in 0..4 {
                c[m] += stack[m] * left.data[0];
            }
            for m in 0..3 {
                g[m * dimension + left.seed] += stack[m];
            }
            for lane in 0..lanes {
                let direction = left.data[left.first_offset(lane)];
                for m in 0..3 {
                    pi[m * lanes + lane] += stack[m] * direction;
                }
            }
        }
        let (g0, rest) = g.split_at(dimension);
        let (g1, rest) = rest.split_at(dimension);
        let g2 = &rest[..dimension];
        let data = addend.copied();
        let r = right.data;
        let xg = &r[1..width];
        data[0] += c[0];
        axpy(&mut data[1..width], 1.0, g0);
        axpy(&mut data[1..width], c[1], xg);
        for lane in 0..lanes {
            let first = right.first_offset(lane);
            let second = first + width;
            let (p, pg) = (r[first], &r[first + 1..second]);
            let (q, qg) = (r[second], &r[second + 1..second + width]);
            let (pi0, pi1, pi2) = (pi[lane], pi[lanes + lane], pi[2 * lanes + lane]);
            data[first] += pi0 + c[1] * p;
            let out_pg = &mut data[first + 1..second];
            axpy(out_pg, pi1 + c[2] * p, xg);
            axpy(out_pg, p, g1);
            axpy(out_pg, c[1], pg);
            data[second] += 2.0 * p * pi1 + c[1] * q + c[2] * p * p;
            let out_qg = &mut data[second + 1..second + width];
            axpy(out_qg, 2.0 * pi2 * p + c[2] * q + c[3] * p * p, xg);
            axpy(out_qg, 2.0 * (pi1 + c[2] * p), pg);
            axpy(out_qg, c[1], qg);
            axpy(out_qg, q, g1);
            axpy(out_qg, p * p, g2);
        }
        addend.from_data(data)
    }

    #[inline]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        workspace: &'arena TraceJetWorkspace,
    ) -> Self {
        assert_eq!(inputs.len(), input_scales.len());
        assert_eq!(inputs.len(), derivative_stacks.len());
        let (out, data) = Self::zero_jet(dimension, workspace);
        for ((input, &s), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks) {
            assert!(
                input.dimension == dimension && input.data.len() == data.len(),
                "trace jet inputs must share lanes and dimension"
            );
            // `f(s·J)` is `f` composed with `J` under the chain-rule-scaled stack.
            input.compose_add_into(
                [stack[0], stack[1] * s, stack[2] * s * s, stack[3] * s * s * s, 0.0],
                data,
            );
        }
        out.from_data(data)
    }
}
