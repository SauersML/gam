#![cfg(test)]
//! The planted known-answer toys of #2951, shared by the tests of every owner they exercise.
//!
//! Each toy's answer is written down by hand before any tool runs:
//! - **Routing toy**: three rotary heads; heads 1 and 2 share query and key, so they share one
//!   attention pattern and their value/output transports carry a cross-head `GL(4)`; head 3 has
//!   query `2 Q₁`, the same subspaces but a different pattern ([`RoutingToy::truth`]).
//!
//! Every fixture is dyadic where a test compares tensors exactly, so a gauge move by a
//! unimodular integer matrix ([`unimodular_pair`]) is carried without rounding.

use ndarray::{Array1, Array2, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::attention::{AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing};

/// Uniform dyadic rationals `k/denominator`, `|k| ≤ reach`.
pub fn dyadic(rng: &mut StdRng, rows: usize, cols: usize, reach: i32, denominator: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || f64::from(rng.random_range(-reach..=reach)) / denominator)
}

pub const ROUTING_WIDTH: usize = 16;
pub const ROUTING_HEAD_DIM: usize = 4;
pub const ROUTING_RANK: usize = 2;
pub const ROUTING_HEADS: usize = 3;

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
        let readout = dyadic(&mut rng, 1, d, 16, 16.0);
        Self {
            query: vec![query_one.clone(), query_one.clone(), &query_one * 2.0],
            key: vec![key_one.clone(), key_one.clone(), key_one],
            value,
            output,
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
}

// ------------------------------------------------------------------------------------------
// The written-down structure of every toy: what a recovery must return exactly.
// ------------------------------------------------------------------------------------------

/// The routing toy's structure.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingTruth {
    /// Heads grouped by equal query/key operators.
    pub laws: Vec<Vec<usize>>,
    /// Head 3's query/key operators are head 1's times this.
    pub third_head_scale: f64,
    /// Rank of each law's summed value/output transport.
    pub law_transport_ranks: Vec<usize>,
}

impl RoutingToy {
    pub fn truth() -> RoutingTruth {
        RoutingTruth {
            laws: vec![vec![0, 1], vec![2]],
            third_head_scale: 2.0,
            law_transport_ranks: vec![4, 2],
        }
    }
}

/// The seed of the random null fixtures.
pub const RANDOM_NULL_SEED: u64 = 7;
