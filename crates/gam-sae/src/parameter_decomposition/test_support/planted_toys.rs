#![cfg(test)]
//! The planted known-answer toys of #2951 (receipt
//! `experiments/issue-2951/receipts/opfirst_toys_planted.json`), shared by the tests of
//! every owner they exercise, with analytic parameter Jacobians whose rounding is carried
//! by ball arithmetic.
//!
//! Each toy's answer is written down by hand before any tool runs:
//! - **Paired copy** `σ(h) − σ(−h) = h`: no unit survives merging; its function-level fibre
//!   is `{W_in = [M; −M], W_out = [N, −N], NM = I, b_in = [β; −β], b_out = −Nβ}`, of
//!   dimension `d² + d`, while the declared GELU unit family is permutations only.
//! - **Planted modules** (read dimensions 3 and 5, 6 and 10 units, in a Hadamard frame),
//!   and the same with a linear cross edge `ε r₁ s₂ᵀ`: generic units, no continuous fibre.
//! - **Rotation block** with angles `0.3, 0.3, 1.1`: one identified plane, one repeated
//!   pair, a commutant of dimension `2·2² + 2·1² = 10`.
//! - **Routing toy**: three rotary heads; heads 1 and 2 share query and key, so they share
//!   one routing law and their value/output transports carry a cross-head `GL(4)`; head 3
//!   has query `2 Q₁`, the same subspaces but a different law. Its fibre is
//!   `12 + 16 + 4 = 32` against a declared `12 + 3·4 = 24`.
//! - **Data-subspace ambiguity**: data on a 4-dimensional span, a hidden direction off it.
//! - **Random null**: a dense random GELU block, one module, no fibre, no plane.
//!
//! Each constructor has its written-down structure beside it (`*_truth`, `rotation_toy`,
//! [`RoutingToy::truth`], `data_subspace_toy`), for any engine that must recover exactly the
//! planted structure.
//!
//! Every fixture is dyadic where a test compares tensors exactly, so a gauge move by a
//! unimodular integer matrix is carried without rounding.

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::probability::{NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_cdf_and_pdf, normal_pdf_bounded};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayView1, s};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use crate::parameter_decomposition::attention::{AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing};

/// The smallest positive subnormal: the absolute error of a product that underflows.
fn underflow() -> f64 {
    f64::from_bits(1)
}

/// A computed real and a radius that encloses the exact value: running error analysis as
/// box arithmetic. An operation adds the propagated radii of its operands and one rounding
/// of its result, `|fl(x) − x| ≤ u |x| ≤ 2u |fl(x)|` (plus the subnormal spacing for a
/// product). Radii are formed from nonnegative terms and each step is rounded up, so the
/// stored radius is never below the exact one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub value: f64,
    pub radius: f64,
}

impl Ball {
    pub fn exact(value: f64) -> Self {
        Self { value, radius: 0.0 }
    }

    pub fn new(value: f64, radius: f64) -> Self {
        Self { value, radius }
    }

    fn rounding(value: f64) -> f64 {
        (2.0 * UNIT_ROUNDOFF * value.abs()).next_up()
    }

    pub fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        Self {
            value,
            radius: ((self.radius + other.radius).next_up() + Self::rounding(value)).next_up(),
        }
    }

    pub fn sub(self, other: Self) -> Self {
        self.add(Self::new(-other.value, other.radius))
    }

    pub fn mul(self, other: Self) -> Self {
        let value = self.value * other.value;
        let propagated = ((self.value.abs() * other.radius).next_up() + (self.radius * other.value.abs()).next_up()).next_up()
            + (self.radius * other.radius).next_up();
        Self {
            value,
            radius: ((propagated.next_up() + Self::rounding(value)).next_up() + underflow()).next_up(),
        }
    }

    /// The sum of the balls `terms`, left to right.
    pub fn sum(terms: impl IntoIterator<Item = Self>) -> Self {
        terms.into_iter().fold(Self::exact(0.0), Self::add)
    }

    /// `Σ_k a_k b_k` over exact coefficients `a` and balls `b`.
    pub fn dot(coefficients: ArrayView1<'_, f64>, balls: &[Self]) -> Self {
        Self::sum(coefficients.iter().zip(balls).map(|(&a, &b)| Self::exact(a).mul(b)))
    }
}

/// `(Φ(t), φ(t))` as balls about `t̂` for every exact `t` within `t`'s radius. `Φ` errs by
/// the table's relative error and underflow floor and moves by at most `sup φ = 1/√(2π) <
/// 0.4` per unit of `t`; `φ` errs by `normal_pdf_bounded`'s bound and moves by at most
/// `sup |φ′| = φ(1) < 0.25`.
fn normal_balls(t: Ball) -> (Ball, Ball) {
    let (cdf, _) = normal_cdf_and_pdf(t.value);
    let (pdf, pdf_bound) = normal_pdf_bounded(t.value);
    let cdf_radius = ((NORMAL_CDF_RELATIVE_ERROR * cdf).next_up() + NORMAL_CDF_UNDERFLOW_FLOOR).next_up();
    (
        Ball::new(cdf, (cdf_radius + (0.4 * t.radius).next_up()).next_up()),
        Ball::new(pdf, (pdf_bound + (0.25 * t.radius).next_up()).next_up()),
    )
}

/// The exact GELU `σ(t) = t Φ(t)` and its slope `σ′(t) = Φ(t) + t φ(t)` as balls.
pub fn gelu_balls(t: Ball) -> (Ball, Ball) {
    let (cdf, pdf) = normal_balls(t);
    (t.mul(cdf), cdf.add(t.mul(pdf)))
}

/// A plain GELU block `F(h) = W_out σ(W_in h + b_in) + b_out + L h`.
#[derive(Clone, Debug)]
pub struct MlpToy {
    pub w_in: Array2<f64>,
    pub b_in: Array1<f64>,
    pub w_out: Array2<f64>,
    pub b_out: Array1<f64>,
    pub skip: Option<Array2<f64>>,
}

impl MlpToy {
    pub fn parameters(&self) -> usize {
        let (hidden, width) = self.w_in.dim();
        let output = self.w_out.nrows();
        hidden * width + hidden + output * hidden + output + self.skip.as_ref().map_or(0, |skip| skip.len())
    }

    /// The analytic Jacobian of `θ ↦ (F_θ(x_n))_n` over the rows of `inputs`, columns in
    /// the order `W_in` (row-major), `b_in`, `W_out` (row-major), `b_out`, then `L`
    /// (row-major) when present; rows `(n, i) ↦ n·d_out + i`. With it, a Frobenius bound on
    /// its distance from the exact Jacobian, from the ball radii of every entry.
    pub fn jacobian(&self, inputs: &Array2<f64>) -> (Array2<f64>, f64) {
        let (hidden, width) = self.w_in.dim();
        let output = self.w_out.nrows();
        let samples = inputs.nrows();
        let mut jacobian = Array2::<f64>::zeros((samples * output, self.parameters()));
        let mut squared = 0.0_f64;
        let (bias_in, weight_out) = (hidden * width, hidden * width + hidden);
        let bias_out = weight_out + output * hidden;
        let skip = bias_out + output;
        for n in 0..samples {
            let x = inputs.row(n);
            let inputs_exact: Vec<Ball> = x.iter().map(|&value| Ball::exact(value)).collect();
            for j in 0..hidden {
                let pre = Ball::dot(self.w_in.row(j), &inputs_exact).add(Ball::exact(self.b_in[j]));
                let (value, slope) = gelu_balls(pre);
                for i in 0..output {
                    let row = n * output + i;
                    let weighted = Ball::exact(self.w_out[[i, j]]).mul(slope);
                    for k in 0..width {
                        let entry = weighted.mul(Ball::exact(x[k]));
                        jacobian[[row, j * width + k]] = entry.value;
                        squared += entry.radius * entry.radius;
                    }
                    jacobian[[row, bias_in + j]] = weighted.value;
                    squared += weighted.radius * weighted.radius;
                    jacobian[[row, weight_out + i * hidden + j]] = value.value;
                    squared += value.radius * value.radius;
                }
            }
            for i in 0..output {
                jacobian[[n * output + i, bias_out + i]] = 1.0;
                if self.skip.is_some() {
                    for k in 0..width {
                        jacobian[[n * output + i, skip + i * width + k]] = x[k];
                    }
                }
            }
        }
        let formation = (squared * (1.0 + accumulation_growth(jacobian.len()))).sqrt().next_up();
        (jacobian, formation)
    }
}

/// Uniform dyadic rationals `k/denominator`, `|k| ≤ reach`.
pub fn dyadic(rng: &mut StdRng, rows: usize, cols: usize, reach: i32, denominator: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || f64::from(rng.random_range(-reach..=reach)) / denominator)
}

/// The paired copy `σ(h) − σ(−h) = h` on `ℝ^width`.
pub fn paired_copy(width: usize) -> MlpToy {
    let identity = Array2::<f64>::eye(width);
    let mut w_in = Array2::<f64>::zeros((2 * width, width));
    w_in.slice_mut(s![..width, ..]).assign(&identity);
    w_in.slice_mut(s![width.., ..]).assign(&(-&identity));
    MlpToy {
        w_in: w_in.clone(),
        b_in: Array1::zeros(2 * width),
        w_out: w_in.t().to_owned(),
        b_out: Array1::zeros(width),
        skip: None,
    }
}

/// The `8 × 8` Sylvester–Hadamard matrix, `H_ij = (−1)^{popcount(i & j)}`.
pub fn hadamard() -> Array2<f64> {
    Array2::from_shape_fn((8, 8), |(i, j)| if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 })
}

/// Two planted modules of a Hadamard frame on `ℝ⁸`: module 1 reads rows `0..3` of `H` and
/// writes rows `0..3` of `H` reversed, with 6 units; module 2 rows `3..8`, with 10 units.
/// Integer reads and writes, dyadic biases, the units shuffled. Returns the block and
/// every unit's module.
pub fn hadamard_modules(seed: u64) -> (MlpToy, Vec<usize>) {
    let width = 8;
    let reads = hadamard();
    let writes = reads.slice(s![.., ..;-1]).to_owned();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut units = Vec::new();
    for (module, (range, count)) in [(0..3, 6), (3..8, 10)].into_iter().enumerate() {
        for _ in 0..count {
            let mut read = Array1::<f64>::zeros(width);
            let mut write = Array1::<f64>::zeros(width);
            for row in range.clone() {
                read.scaled_add(f64::from(rng.random_range(-3_i32..=3)), &reads.row(row));
                write.scaled_add(f64::from(rng.random_range(-3_i32..=3)), &writes.row(row));
            }
            units.push((read, write, f64::from(rng.random_range(-8_i32..=8)) / 8.0, module));
        }
    }
    units.shuffle(&mut rng);
    let hidden = units.len();
    let mut block = MlpToy {
        w_in: Array2::zeros((hidden, width)),
        b_in: Array1::zeros(hidden),
        w_out: Array2::zeros((width, hidden)),
        b_out: Array1::zeros(width),
        skip: None,
    };
    let mut truth = Vec::with_capacity(hidden);
    for (unit, (read, write, bias, module)) in units.into_iter().enumerate() {
        block.w_in.row_mut(unit).assign(&read);
        block.w_out.column_mut(unit).assign(&write);
        block.b_in[unit] = bias;
        truth.push(module);
    }
    (block, truth)
}

/// The cross edge `ε r₁ s₂ᵀ` of the planted modules: module 1's first output direction
/// (row 0 of `H` reversed) reading module 2's first input direction (row 3 of `H`).
pub fn cross_edge(epsilon: f64) -> Array2<f64> {
    let reads = hadamard();
    let writes = reads.slice(s![.., ..;-1]).to_owned();
    Array2::from_shape_fn((8, 8), |(i, j)| epsilon * writes[[0, i]] * reads[[3, j]])
}

/// A random GELU block: reads and writes uniform with variance `1/fan_in`, biases uniform.
pub fn random_mlp(seed: u64, hidden: usize, width: usize) -> MlpToy {
    let mut rng = StdRng::seed_from_u64(seed);
    let read_reach = (3.0 / width as f64).sqrt();
    let write_reach = (3.0 / hidden as f64).sqrt();
    MlpToy {
        w_in: Array2::from_shape_simple_fn((hidden, width), || rng.random_range(-read_reach..read_reach)),
        b_in: Array1::from_shape_simple_fn(hidden, || rng.random_range(-0.5..0.5)),
        w_out: Array2::from_shape_simple_fn((width, hidden), || rng.random_range(-write_reach..write_reach)),
        b_out: Array1::from_shape_simple_fn(width, || rng.random_range(-0.1..0.1)),
        skip: None,
    }
}

/// Inputs uniform on `[−reach, reach]`.
pub fn uniform_inputs(seed: u64, samples: usize, width: usize, reach: f64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((samples, width), || rng.random_range(-reach..reach))
}

pub const ROUTING_WIDTH: usize = 16;
pub const ROUTING_HEAD_DIM: usize = 4;
pub const ROUTING_RANK: usize = 2;
pub const ROUTING_HEADS: usize = 3;
pub const ROUTING_TOKENS: usize = 6;
pub const ROUTING_SEQUENCES: usize = 24;

/// Three causal rotary heads on `ℝ¹⁶` (head dimension 4, planes at frequencies `1` and
/// `0.1`, rotate-half pairing, score scale `1/2`) with rank-2 value/output transports.
/// Heads 1 and 2 share query `Q₁` and key `K₁`; head 3 reads query `2 Q₁` and key `K₁`.
/// Every tensor, input and readout is dyadic.
#[derive(Clone, Debug)]
pub struct RoutingToy {
    /// Per head, `head_dim × width`.
    pub query: Vec<Array2<f64>>,
    pub key: Vec<Array2<f64>>,
    /// Per head, `rank × width`.
    pub value: Vec<Array2<f64>>,
    /// Per head, `width × rank`.
    pub output: Vec<Array2<f64>>,
    /// `tokens × width` each, at positions `0..tokens`.
    pub sequences: Vec<Array2<f64>>,
    /// A readout of the block's output, `1 × width`.
    pub readout: Array2<f64>,
}

/// `I + strictly_lower` and `I + strictly_upper` with entries in `{−1, 0, 1}`, their product
/// `M = L U` and its inverse `U⁻¹ L⁻¹`, all integer, so `M` moves dyadic tensors exactly.
pub fn unimodular_pair(order: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lower = Array2::<f64>::eye(order);
    let mut upper = Array2::<f64>::eye(order);
    for i in 0..order {
        for j in 0..order {
            let draw = f64::from(rng.random_range(-1_i32..=1));
            if i > j {
                lower[[i, j]] = draw;
            } else if i < j {
                upper[[i, j]] = draw;
            }
        }
    }
    // Unit-triangular inverses by substitution: integer arithmetic, exact in binary64.
    let mut lower_inverse = Array2::<f64>::eye(order);
    for i in 0..order {
        for j in 0..i {
            let value: f64 = (j..i).map(|k| lower[[i, k]] * lower_inverse[[k, j]]).sum();
            lower_inverse[[i, j]] = -value;
        }
    }
    let mut upper_inverse = Array2::<f64>::eye(order);
    for i in (0..order).rev() {
        for j in (i + 1)..order {
            let value: f64 = ((i + 1)..=j).map(|k| upper[[i, k]] * upper_inverse[[k, j]]).sum();
            upper_inverse[[i, j]] = -value;
        }
    }
    (lower.dot(&upper), upper_inverse.dot(&lower_inverse))
}

impl RoutingToy {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let (d, hd, r) = (ROUTING_WIDTH, ROUTING_HEAD_DIM, ROUTING_RANK);
        let query_one = dyadic(&mut rng, hd, d, 32, 64.0);
        let key_one = dyadic(&mut rng, hd, d, 32, 64.0);
        let value = (0..ROUTING_HEADS).map(|_| dyadic(&mut rng, r, d, 32, 64.0)).collect();
        let output = (0..ROUTING_HEADS).map(|_| dyadic(&mut rng, d, r, 64, 64.0)).collect();
        let sequences = (0..ROUTING_SEQUENCES).map(|_| dyadic(&mut rng, ROUTING_TOKENS, d, 32, 16.0)).collect();
        let readout = dyadic(&mut rng, 1, d, 16, 16.0);
        Self {
            query: vec![query_one.clone(), query_one.clone(), &query_one * 2.0],
            key: vec![key_one.clone(), key_one.clone(), key_one],
            value,
            output,
            sequences,
            readout,
        }
    }

    pub fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.1],
            attention_scaling: 1.0,
        }
    }

    pub fn positions() -> Vec<i64> {
        (0..ROUTING_TOKENS as i64).collect()
    }

    /// The block as the attention owner's native block: one key/value head per query head,
    /// each value block padded to the head dimension with zero rows and each output block
    /// with zero columns, which the function never reads.
    pub fn native(&self, value: &[Array2<f64>], output: &[Array2<f64>]) -> NativeAttention {
        let (d, hd, r, heads) = (ROUTING_WIDTH, ROUTING_HEAD_DIM, ROUTING_RANK, ROUTING_HEADS);
        let geometry = AttentionGeometry {
            model_dim: d,
            n_heads: heads,
            n_kv_heads: heads,
            head_dim: hd,
        };
        let stack = |blocks: &[Array2<f64>]| {
            let mut stacked = Array2::<f64>::zeros((heads * hd, d));
            for (head, block) in blocks.iter().enumerate() {
                stacked.slice_mut(s![head * hd..head * hd + block.nrows(), ..]).assign(block);
            }
            stacked
        };
        let mut output_weight = Array2::<f64>::zeros((d, heads * hd));
        for (head, block) in output.iter().enumerate() {
            output_weight.slice_mut(s![.., head * hd..head * hd + r]).assign(block);
        }
        let affine = |weight: Array2<f64>| AffineProjection {
            bias: Array1::zeros(weight.nrows()),
            weight,
        };
        NativeAttention::new(
            geometry,
            Self::rotary(),
            0.5,
            affine(stack(&self.query)),
            affine(stack(&self.key)),
            affine(stack(value)),
            affine(output_weight),
        )
        .expect("the routing toy matches its geometry")
    }

    /// `Σ_{h ∈ heads} O_h V_h`, `width × width`: exact for dyadic tensors.
    pub fn transport(value: &[Array2<f64>], output: &[Array2<f64>], heads: &[usize]) -> Array2<f64> {
        let mut transport = Array2::<f64>::zeros((ROUTING_WIDTH, ROUTING_WIDTH));
        for &head in heads {
            transport += &output[head].dot(&value[head]);
        }
        transport
    }

    /// The per-head value/output gauge `V_h ↦ S_h V_h`, `O_h ↦ O_h S_h⁻¹` with unimodular
    /// `S_h`: every head moves, no function does.
    pub fn per_head_gauge(&self, seed: u64) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut value = Vec::with_capacity(ROUTING_HEADS);
        let mut output = Vec::with_capacity(ROUTING_HEADS);
        for head in 0..ROUTING_HEADS {
            let (change, inverse) = unimodular_pair(ROUTING_RANK, seed + head as u64);
            value.push(change.dot(&self.value[head]));
            output.push(self.output[head].dot(&inverse));
        }
        (value, output)
    }

    /// The cross-head gauge of heads 1 and 2, which share one attention pattern:
    /// `[V₁; V₂] ↦ M [V₁; V₂]`, `[O₁, O₂] ↦ [O₁, O₂] M⁻¹` with a unimodular `M ∈ GL(4)`.
    pub fn cross_head_gauge(&self, seed: u64) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let r = ROUTING_RANK;
        let (change, inverse) = unimodular_pair(2 * r, seed);
        let mut stacked_value = Array2::<f64>::zeros((2 * r, ROUTING_WIDTH));
        stacked_value.slice_mut(s![..r, ..]).assign(&self.value[0]);
        stacked_value.slice_mut(s![r.., ..]).assign(&self.value[1]);
        let mut stacked_output = Array2::<f64>::zeros((ROUTING_WIDTH, 2 * r));
        stacked_output.slice_mut(s![.., ..r]).assign(&self.output[0]);
        stacked_output.slice_mut(s![.., r..]).assign(&self.output[1]);
        let moved_value = change.dot(&stacked_value);
        let moved_output = stacked_output.dot(&inverse);
        let mut value = self.value.clone();
        let mut output = self.output.clone();
        value[0] = moved_value.slice(s![..r, ..]).to_owned();
        value[1] = moved_value.slice(s![r.., ..]).to_owned();
        output[0] = moved_output.slice(s![.., ..r]).to_owned();
        output[1] = moved_output.slice(s![.., r..]).to_owned();
        (value, output)
    }

    pub fn parameters() -> usize {
        let (d, hd, r, heads) = (ROUTING_WIDTH, ROUTING_HEAD_DIM, ROUTING_RANK, ROUTING_HEADS);
        heads * (2 * hd * d + 2 * r * d)
    }

    /// Column offsets of `Q_h[i, j]`, `K_h[i, j]`, `V_h[k, j]` and `O_h[i, k]`.
    pub fn query_column(head: usize, i: usize, j: usize) -> usize {
        (head * ROUTING_HEAD_DIM + i) * ROUTING_WIDTH + j
    }

    pub fn key_column(head: usize, i: usize, j: usize) -> usize {
        ROUTING_HEADS * ROUTING_HEAD_DIM * ROUTING_WIDTH + Self::query_column(head, i, j)
    }

    pub fn value_column(head: usize, k: usize, j: usize) -> usize {
        2 * ROUTING_HEADS * ROUTING_HEAD_DIM * ROUTING_WIDTH + (head * ROUTING_RANK + k) * ROUTING_WIDTH + j
    }

    pub fn output_column(head: usize, i: usize, k: usize) -> usize {
        2 * ROUTING_HEADS * ROUTING_HEAD_DIM * ROUTING_WIDTH
            + ROUTING_HEADS * ROUTING_RANK * ROUTING_WIDTH
            + (head * ROUTING_WIDTH + i) * ROUTING_RANK
            + k
    }

    /// The analytic Jacobian of `θ = (Q, K, V, O) ↦` every output of every sequence, rows
    /// `(sequence, token, coordinate)`, columns by [`RoutingToy::query_column`] and its
    /// siblings, with a Frobenius bound on its distance from the exact Jacobian.
    ///
    /// The attention weights and their radii are the attention owner's executed ones
    /// ([`NativeAttention::execute`]), which enclose the exact softmax. Everything else is
    /// ball arithmetic on the stored tensors: `q_t = R_t Q x_t`, `k_s = R_s K x_s` with the
    /// rotation's `cos` and `sin` as balls of radius `γ₁|φ| + ε` (the angle's rounding and
    /// libm's ulp), `S_ts = ½ q_t·k_s`, and
    ///
    /// ```text
    /// ∂S_ts/∂Q_h[i,j] = ½ x_t[j] (R_t e_i)·k_s,   ∂S_ts/∂K_h[i,j] = ½ x_s[j] q_t·(R_s e_i),
    /// ∂A_ts = A_ts (∂S_ts − Σ_{s'≤t} A_ts' ∂S_ts'),   ∂y_t = Σ_{s≤t} ∂A_ts O_h V_h x_s,
    /// ∂y_t/∂V_h[k,j] = O_h[:,k] Σ_s A_ts x_s[j],   ∂y_t/∂O_h[i,k] = e_i Σ_s A_ts (V_h x_s)_k.
    /// ```
    pub fn jacobian(&self, governor: &MemoryGovernor) -> (Array2<f64>, f64) {
        let (d, hd, r, heads, tokens) = (ROUTING_WIDTH, ROUTING_HEAD_DIM, ROUTING_RANK, ROUTING_HEADS, ROUTING_TOKENS);
        let planes = hd / 2;
        let rotary = Self::rotary();
        let native = self.native(&self.value, &self.output);
        let positions = Self::positions();
        let rows = self.sequences.len() * tokens * d;
        let mut jacobian = Array2::<f64>::zeros((rows, Self::parameters()));
        let mut squared = 0.0_f64;
        // `(cos, sin)` of plane `p` at position `t`, as balls.
        let trig: Vec<Vec<(Ball, Ball)>> = (0..tokens)
            .map(|t| {
                (0..planes)
                    .map(|p| {
                        let angle = t as f64 * rotary.inverse_frequencies[p];
                        let (sin, cos) = angle.sin_cos();
                        let eta = ((accumulation_growth(1) * angle.abs()).next_up() + f64::EPSILON).next_up();
                        (Ball::new(cos, eta), Ball::new(sin, eta))
                    })
                    .collect()
            })
            .collect();
        let half = Ball::exact(0.5);
        let rotate = |raw: &[Ball], t: usize| -> Vec<Ball> {
            let mut rotated = raw.to_vec();
            for (p, &(cos, sin)) in trig[t].iter().enumerate() {
                let (a, b) = rotary.plane(p);
                rotated[a] = raw[a].mul(cos).sub(raw[b].mul(sin));
                rotated[b] = raw[b].mul(cos).add(raw[a].mul(sin));
            }
            rotated
        };
        // `(R_t e_i)·v` for a head vector `v`.
        let turned = |v: &[Ball], t: usize, i: usize| -> Ball {
            let p = i % planes;
            let (a, b) = rotary.plane(p);
            let (cos, sin) = trig[t][p];
            if i == a { cos.mul(v[a]).add(sin.mul(v[b])) } else { cos.mul(v[b]).sub(sin.mul(v[a])) }
        };
        let mut record = |jacobian: &mut Array2<f64>, row: usize, column: usize, entry: Ball| {
            jacobian[[row, column]] = entry.value;
            squared += entry.radius * entry.radius;
        };
        for (sequence, x) in self.sequences.iter().enumerate() {
            let executed = native.execute(governor, x.view(), &positions).expect("the routing toy executes");
            let exact: Vec<Vec<Ball>> = x.rows().into_iter().map(|row| row.iter().map(|&v| Ball::exact(v)).collect()).collect();
            let row_of = |t: usize, i: usize| (sequence * tokens + t) * d + i;
            for head in 0..heads {
                let weight = |t: usize, s: usize| Ball::new(executed.weights[[head, t, s]], executed.weight_radius[[head, t, s]]);
                let project = |matrix: &Array2<f64>, t: usize| -> Vec<Ball> {
                    (0..matrix.nrows()).map(|i| Ball::dot(matrix.row(i), &exact[t])).collect()
                };
                let queries: Vec<Vec<Ball>> = (0..tokens).map(|t| rotate(&project(&self.query[head], t), t)).collect();
                let keys: Vec<Vec<Ball>> = (0..tokens).map(|t| rotate(&project(&self.key[head], t), t)).collect();
                let reads: Vec<Vec<Ball>> = (0..tokens).map(|t| project(&self.value[head], t)).collect();
                let writes: Vec<Vec<Ball>> = reads
                    .iter()
                    .map(|read| (0..d).map(|i| Ball::dot(self.output[head].row(i), read)).collect())
                    .collect();
                // Value and output columns.
                for t in 0..tokens {
                    let mixed_read: Vec<Ball> = (0..r).map(|k| Ball::sum((0..=t).map(|s| weight(t, s).mul(reads[s][k])))).collect();
                    for j in 0..d {
                        let mixed_input = Ball::sum((0..=t).map(|s| weight(t, s).mul(exact[s][j])));
                        for k in 0..r {
                            for i in 0..d {
                                let entry = Ball::exact(self.output[head][[i, k]]).mul(mixed_input);
                                record(&mut jacobian, row_of(t, i), Self::value_column(head, k, j), entry);
                            }
                        }
                    }
                    for i in 0..d {
                        for k in 0..r {
                            record(&mut jacobian, row_of(t, i), Self::output_column(head, i, k), mixed_read[k]);
                        }
                    }
                }
                // Query and key columns through the softmax.
                for i in 0..hd {
                    for j in 0..d {
                        for (key_side, column) in [(false, Self::query_column(head, i, j)), (true, Self::key_column(head, i, j))] {
                            for t in 0..tokens {
                                let score_change: Vec<Ball> = (0..=t)
                                    .map(|s| {
                                        let raw = if key_side {
                                            exact[s][j].mul(turned(&queries[t], s, i))
                                        } else {
                                            exact[t][j].mul(turned(&keys[s], t, i))
                                        };
                                        half.mul(raw)
                                    })
                                    .collect();
                                let mean = Ball::sum((0..=t).map(|s| weight(t, s).mul(score_change[s])));
                                let weight_change: Vec<Ball> =
                                    (0..=t).map(|s| weight(t, s).mul(score_change[s].sub(mean))).collect();
                                for coordinate in 0..d {
                                    let entry = Ball::sum((0..=t).map(|s| weight_change[s].mul(writes[s][coordinate])));
                                    record(&mut jacobian, row_of(t, coordinate), column, entry);
                                }
                            }
                        }
                    }
                }
            }
        }
        (jacobian, (squared * (1.0 + accumulation_growth(rows * Self::parameters()))).sqrt().next_up())
    }

    /// The hand-derived tangents of the fibre, one per column (`parameters × 32`):
    /// - per head, `δK_h = X K_h`, `δQ_h = −Xᵀ Q_h` for `X` in the rotary commutant, spanned
    ///   on each plane by its identity and its quarter turn `J` (4 per head);
    /// - heads 1 and 2 together, `δ[V₁; V₂] = X [V₁; V₂]`, `δ[O₁, O₂] = −[O₁, O₂] X` for
    ///   `X ∈ gl(4)` (16);
    /// - head 3, `δV₃ = X V₃`, `δO₃ = −O₃ X` for `X ∈ gl(2)` (4).
    pub fn fibre_tangents(&self) -> Array2<f64> {
        let (d, hd, r, heads) = (ROUTING_WIDTH, ROUTING_HEAD_DIM, ROUTING_RANK, ROUTING_HEADS);
        let rotary = Self::rotary();
        let mut tangents: Vec<Array1<f64>> = Vec::new();
        for head in 0..heads {
            for p in 0..hd / 2 {
                let (a, b) = rotary.plane(p);
                for turn in [false, true] {
                    let mut generator = Array2::<f64>::zeros((hd, hd));
                    if turn {
                        generator[[a, b]] = -1.0;
                        generator[[b, a]] = 1.0;
                    } else {
                        generator[[a, a]] = 1.0;
                        generator[[b, b]] = 1.0;
                    }
                    let mut tangent = Array1::<f64>::zeros(Self::parameters());
                    let key = generator.dot(&self.key[head]);
                    let query = -generator.t().dot(&self.query[head]);
                    for i in 0..hd {
                        for j in 0..d {
                            tangent[Self::key_column(head, i, j)] = key[[i, j]];
                            tangent[Self::query_column(head, i, j)] = query[[i, j]];
                        }
                    }
                    tangents.push(tangent);
                }
            }
        }
        for group in [vec![0, 1], vec![2]] {
            let order = group.len() * r;
            for (row, col) in (0..order).flat_map(|row| (0..order).map(move |col| (row, col))) {
                let mut tangent = Array1::<f64>::zeros(Self::parameters());
                // `X = e_row e_colᵀ`: row `row` of the stacked values gains row `col`, and
                // column `col` of the stacked outputs loses column `row`.
                let (to_head, to_k) = (group[row / r], row % r);
                let (from_head, from_k) = (group[col / r], col % r);
                for j in 0..d {
                    tangent[Self::value_column(to_head, to_k, j)] += self.value[from_head][[from_k, j]];
                }
                for i in 0..d {
                    tangent[Self::output_column(from_head, i, from_k)] -= self.output[to_head][[i, to_k]];
                }
                tangents.push(tangent);
            }
        }
        let mut matrix = Array2::<f64>::zeros((Self::parameters(), tangents.len()));
        for (column, tangent) in tangents.iter().enumerate() {
            matrix.column_mut(column).assign(tangent);
        }
        matrix
    }
}

/// `γ_n ‖|J| |T|‖_F`: the rounding of the product of a computed Jacobian with tangents,
/// which adds to the Jacobian's formation bound.
pub fn product_band(jacobian: &Array2<f64>, tangents: &Array2<f64>) -> f64 {
    let magnitude = jacobian.mapv(f64::abs).dot(&tangents.mapv(f64::abs));
    accumulation_growth(jacobian.ncols()) * magnitude.iter().map(|value| value * value).sum::<f64>().sqrt()
}

// ------------------------------------------------------------------------------------------
// The written-down structure of every toy: what a recovery must return exactly.
// ------------------------------------------------------------------------------------------

/// The paired copy's planted structure on `ℝ^width`.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedCopyTruth {
    /// Units left after sign duplicates merge: none.
    pub merged_units: usize,
    /// The merged linear part: `I` exactly.
    pub linear: Array2<f64>,
    /// The function-level fibre, `d² + d` (`GL(d)` and a bias shift).
    pub fibre_dimension: usize,
    /// The declared GELU unit family's continuous orbit: none.
    pub declared_continuous: usize,
}

pub fn paired_copy_truth(width: usize) -> PairedCopyTruth {
    PairedCopyTruth {
        merged_units: 0,
        linear: Array2::eye(width),
        fibre_dimension: width * width + width,
        declared_continuous: 0,
    }
}

/// The planted modules' structure, for the units' module labels `hadamard_modules` returns.
#[derive(Clone, Debug, PartialEq)]
pub struct ModulesTruth {
    /// The unit partition, each part sorted, parts sorted.
    pub partition: Vec<Vec<usize>>,
    /// Each module's read subspace (rows of `H`) and write subspace (rows of `H` reversed).
    pub read_spaces: Vec<Array2<f64>>,
    pub write_spaces: Vec<Array2<f64>>,
    /// Generic GELU units: no continuous fibre, with or without the cross edge.
    pub fibre_dimension: usize,
    /// Rank of module 1's outputs pulled back through the block: its read dimension 3,
    /// and 4 once a cross edge of any size reads module 2's first direction.
    pub observable_rank: usize,
    pub observable_rank_with_cross_edge: usize,
}

pub fn hadamard_modules_truth(unit_modules: &[usize]) -> ModulesTruth {
    let reads = hadamard();
    let writes = reads.slice(s![.., ..;-1]).to_owned();
    let mut partition = vec![Vec::new(), Vec::new()];
    for (unit, &module) in unit_modules.iter().enumerate() {
        partition[module].push(unit);
    }
    partition.sort();
    let spaces = |matrix: &Array2<f64>| vec![matrix.slice(s![..3, ..]).to_owned(), matrix.slice(s![3.., ..]).to_owned()];
    ModulesTruth {
        partition,
        read_spaces: spaces(&reads),
        write_spaces: spaces(&writes),
        fibre_dimension: 0,
        observable_rank: 3,
        observable_rank_with_cross_edge: 4,
    }
}

/// The rotation toy: angles `0.3, 0.3, 1.1` in a hidden basis of `ℝ⁶`.
pub const ROTATION_ANGLES: [f64; 3] = [0.3, 0.3, 1.1];
pub const ROTATION_SEED: u64 = 0x2951_0004;

/// The rotation toy's planted matrix and its structure.
pub struct RotationToy {
    /// `R = Q B Qᵀ` and the float defects of its factors; plane `k` is basis columns
    /// `2k..2k + 2`.
    pub planted: super::Planted,
    /// The one identified plane: its angle and basis columns.
    pub identified_plane: (f64, std::ops::Range<usize>),
    /// The repeated pair: its angle, basis columns and plane count. Its individual planes
    /// are not determined by `R`.
    pub repeated: (f64, std::ops::Range<usize>, usize),
    /// `dim {X : X R = R X} = 2·2² + 2·1²`.
    pub commutant_dimension: usize,
    /// The Krylov closure of a generic readout under `R − I`: one pair per distinct
    /// eigenvalue pair, holding the identified plane and a 2-dimensional slice of the pair.
    pub krylov_dimension: usize,
}

pub fn rotation_toy() -> RotationToy {
    RotationToy {
        planted: super::plant(6, &ROTATION_ANGLES, 0, ROTATION_SEED),
        identified_plane: (1.1, 4..6),
        repeated: (0.3, 0..4, 2),
        commutant_dimension: 10,
        krylov_dimension: 4,
    }
}

/// The routing toy's structure.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingTruth {
    /// Heads grouped by equal query/key operators.
    pub laws: Vec<Vec<usize>>,
    /// Head 3's query/key operators are head 1's times this.
    pub third_head_scale: f64,
    /// Rank of each law's summed value/output transport.
    pub law_transport_ranks: Vec<usize>,
    /// The function-level fibre: rotary commutant `12`, `GL(4)` of the merged law `16`,
    /// `GL(2)` of head 3 `4`.
    pub fibre_dimension: usize,
    /// What the per-head declared families charge: `12 + 3·4`.
    pub declared_per_head: usize,
    /// Observable rank of one step with the readout: per law `3`, per head (over-counted) `4`.
    pub observable_rank_per_law: usize,
    pub observable_rank_per_head: usize,
}

impl RoutingToy {
    pub fn truth() -> RoutingTruth {
        RoutingTruth {
            laws: vec![vec![0, 1], vec![2]],
            third_head_scale: 2.0,
            law_transport_ranks: vec![4, 2],
            fibre_dimension: 32,
            declared_per_head: 24,
            observable_rank_per_law: 3,
            observable_rank_per_head: 4,
        }
    }
}

/// The data-subspace toy: data on the span of rows `0..4` of `H` (integer coefficients, so
/// every datum is exact) and a hidden direction `v` = row 4, orthogonal to the data. The
/// data Gramian has rank 4 and null space `S^⊥ ∋ v`; component reads `a + t v` and `a` agree
/// on every datum for every `t` and disagree off the data.
#[derive(Clone, Debug)]
pub struct DataSubspaceToy {
    /// `samples × 8`.
    pub data: Array2<f64>,
    /// `4 × 8`, the data span.
    pub span: Array2<f64>,
    /// The hidden direction.
    pub hidden: Array1<f64>,
    pub data_rank: usize,
}

pub fn data_subspace_toy(seed: u64, samples: usize) -> DataSubspaceToy {
    let basis = hadamard();
    let span = basis.slice(s![..4, ..]).to_owned();
    let mut rng = StdRng::seed_from_u64(seed);
    let coefficients = Array2::from_shape_simple_fn((samples, 4), || f64::from(rng.random_range(-4_i32..=4)));
    DataSubspaceToy {
        data: coefficients.dot(&span),
        span,
        hidden: basis.row(4).to_owned(),
        data_rank: 4,
    }
}

/// The random null (`random_mlp(7, 64, 16)`): one module, no continuous fibre, no plane.
pub const RANDOM_NULL_SEED: u64 = 7;
