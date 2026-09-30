#![cfg(test)]
//! Known-answer toys a decomposition engine must recover (#2951): an induction circuit,
//! modular addition by Fourier features, and the residual-MLP toys of Braun et al. (2025,
//! APD) with 1, 2 and 3 layers. The routing toy of [`super::planted_toys`] is the
//! attention-letters toy. Each network is built by hand, so its answer is written down
//! beside it, not learned:
//!
//! - **Induction** ([`induction_toy`]): a two-layer attention-only transformer on one-hot
//!   tokens. Layer 0 is one previous-token routing law split over two heads, each copying
//!   half of the vocabulary from the token subspace into the previous-token subspace.
//!   Layer 1 is one induction head: its query reads the current token, its key reads the
//!   previous-token subspace (K-composition with layer 0), and its value/output copies the
//!   attended token into the output subspace the unembedding reads.
//! - **Modular addition** ([`fourier_modadd`]): `logits(c) ≈ Σ_k w_k(a − b) cos(ω_k(a + b − c))`
//!   from a ReLU layer on the concatenated embeddings `[E(a); E(b)]`, with `E(a)` the
//!   planes `(cos ω_k a, sin ω_k a)` of the key frequencies only. Unit `(k, j)` reads
//!   `cos(ω_k a + θ_j) + cos(ω_k b + θ_j) = 2 cos(ω_k(a − b)/2) cos(ω_k(a + b)/2 + θ_j)`
//!   at phase `θ_j = 2πj/J` and writes `cos(2θ_j + ω_k c)`. The second harmonic of
//!   `ReLU(A cos x)` is `(2/3π)|A| cos 2x`, so the phase sum gives
//!   `(J/2)(2/3π)|A_k| cos(ω_k(a + b − c))`. For odd `J ≥ 5` the first and constant
//!   harmonics cancel and the only aliases are even harmonics `n ≡ ±2 (mod J)`, `n ≥ 2J − 2`,
//!   of relative size `3/(n² − 1)`. `|A_k| > 0` for odd `p`. The modules are the
//!   frequencies: each reads only the 2-dimensional sum `E_k(a) + E_k(b)`.
//! - **Residual MLP** ([`resid_mlp`]): features embedded by orthonormal rows `E_i` of a
//!   Hadamard frame, `r ← r + W_out relu(W_in r)` per layer, readout `E`. Feature `i` is
//!   computed in layer `i mod L` by one unit reading `E_i` and writing `E_i`, or by two
//!   duplicate units writing `E_i/2` each. The model computes `y = x + relu(x)` exactly for
//!   every input, and its decomposition is one rank-one component per feature in each of
//!   `W_in` and `W_out` of that feature's layer.
//!
//! The ResidMLP and induction tensors are dyadic, so their forward passes and their sums of
//! components are exact.

use ndarray::{Array1, Array2, ArrayView1, Axis, concatenate, s};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

use crate::attention::{
    AffineProjection, AttentionExecution, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
};
use gam_runtime::resource::MemoryGovernor;

/// Features of the residual-MLP toys.
pub const RESID_FEATURES: usize = 12;
/// Residual width of the residual-MLP toys: a `16 × 16` Hadamard frame holds 12 orthonormal
/// feature rows and 4 directions no feature uses.
pub const RESID_WIDTH: usize = 16;

/// One residual-MLP layer, `r ← r + W_out relu(W_in r)`, with no biases.
#[derive(Clone, Debug)]
pub struct ResidLayer {
    /// `units × width`.
    pub w_in: Array2<f64>,
    /// `width × units`.
    pub w_out: Array2<f64>,
}

impl ResidLayer {
    pub fn apply(&self, residual: &Array1<f64>) -> Array1<f64> {
        residual + &self.w_out.dot(&self.w_in.dot(residual).mapv(|t| t.max(0.0)))
    }
}

/// The residual MLP `y = E r_L`, `r_0 = Eᵀ x`, `r_{l+1} = r_l + W_out^l relu(W_in^l r_l)`.
#[derive(Clone, Debug)]
pub struct ResidMlp {
    /// `features × width`, orthonormal rows: the embedding and, transposed, the readout.
    pub embedding: Array2<f64>,
    pub layers: Vec<ResidLayer>,
}

impl ResidMlp {
    pub fn forward(&self, features: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut residual = self.embedding.t().dot(&features);
        for layer in &self.layers {
            residual = layer.apply(&residual);
        }
        self.embedding.dot(&residual)
    }
}

/// One feature's mechanism: its layer, its units there, and its pieces of that layer's
/// weights. Each piece is rank one; the pieces of all features sum to the weights exactly.
#[derive(Clone, Debug)]
pub struct ResidComponent {
    pub feature: usize,
    pub layer: usize,
    pub units: Vec<usize>,
    /// `units × width`: `Σ_{u ∈ units} e_u E_iᵀ`.
    pub w_in: Array2<f64>,
    /// `width × units`: `E_i Σ_{u ∈ units} e_uᵀ / |units|`.
    pub w_out: Array2<f64>,
}

/// The residual-MLP answer: one component per feature, in feature order.
#[derive(Clone, Debug)]
pub struct ResidMlpTruth {
    pub components: Vec<ResidComponent>,
}

/// Whether feature `i` is computed by two duplicate units of half write each.
pub fn resid_feature_duplicated(feature: usize) -> bool {
    feature % 4 == 3
}

/// The `layers`-layer residual MLP (1, 2 and 3 are the APD toys), units shuffled within
/// each layer by `seed`.
pub fn resid_mlp(layers: usize, seed: u64) -> (ResidMlp, ResidMlpTruth) {
    assert!(layers >= 1, "a residual MLP has a layer");
    let mut rng = StdRng::seed_from_u64(seed);
    let embedding = hadamard_frame(RESID_WIDTH).slice(s![..RESID_FEATURES, ..]).to_owned();
    let mut built = Vec::with_capacity(layers);
    let mut components = Vec::with_capacity(RESID_FEATURES);
    for layer in 0..layers {
        // (feature, write scale) per unit, in shuffled order.
        let mut units: Vec<(usize, f64)> = Vec::new();
        for feature in (layer..RESID_FEATURES).step_by(layers) {
            if resid_feature_duplicated(feature) {
                units.extend([(feature, 0.5), (feature, 0.5)]);
            } else {
                units.push((feature, 1.0));
            }
        }
        units.shuffle(&mut rng);
        let mut w_in = Array2::<f64>::zeros((units.len(), RESID_WIDTH));
        let mut w_out = Array2::<f64>::zeros((RESID_WIDTH, units.len()));
        for (unit, &(feature, scale)) in units.iter().enumerate() {
            w_in.row_mut(unit).assign(&embedding.row(feature));
            w_out.column_mut(unit).assign(&(&embedding.row(feature) * scale));
        }
        for feature in (layer..RESID_FEATURES).step_by(layers) {
            let owned: Vec<usize> = (0..units.len()).filter(|&unit| units[unit].0 == feature).collect();
            let mut piece_in = Array2::<f64>::zeros(w_in.dim());
            let mut piece_out = Array2::<f64>::zeros(w_out.dim());
            for &unit in &owned {
                piece_in.row_mut(unit).assign(&w_in.row(unit));
                piece_out.column_mut(unit).assign(&w_out.column(unit));
            }
            components.push(ResidComponent {
                feature,
                layer,
                units: owned,
                w_in: piece_in,
                w_out: piece_out,
            });
        }
        built.push(ResidLayer { w_in, w_out });
    }
    components.sort_by_key(|component| component.feature);
    (ResidMlp { embedding, layers: built }, ResidMlpTruth { components })
}

/// The orthonormal Sylvester–Hadamard frame of order `n` (a power of two), `H/√n`: dyadic
/// for `n` a power of four.
pub fn hadamard_frame(n: usize) -> Array2<f64> {
    assert!(n.is_power_of_two(), "a Sylvester–Hadamard order is a power of two");
    let scale = 1.0 / (n as f64).sqrt();
    Array2::from_shape_fn((n, n), |(i, j)| if (i & j).count_ones() % 2 == 0 { scale } else { -scale })
}

/// The modular-addition toy: `logits(a, b) = W_out relu(W_in [E(a); E(b)])`.
#[derive(Clone, Debug)]
pub struct FourierModAdd {
    pub modulus: usize,
    /// `p × 2|K|`: row `a` is `(cos ω_k a, sin ω_k a)` for each key frequency `k`.
    pub embedding: Array2<f64>,
    /// `units × 4|K|`, over `[E(a); E(b)]`.
    pub w_in: Array2<f64>,
    /// `p × units`.
    pub w_out: Array2<f64>,
}

impl FourierModAdd {
    pub fn input(&self, a: usize, b: usize) -> Array1<f64> {
        concatenate(Axis(0), &[self.embedding.row(a), self.embedding.row(b)]).expect("two embedding rows")
    }

    pub fn logits(&self, a: usize, b: usize) -> Array1<f64> {
        self.w_out.dot(&self.w_in.dot(&self.input(a, b)).mapv(|t| t.max(0.0)))
    }
}

/// The modular-addition answer.
#[derive(Clone, Debug)]
pub struct FourierModAddTruth {
    /// The key frequencies `K`, increasing.
    pub frequencies: Vec<usize>,
    /// Phases per frequency, `J`.
    pub phases: usize,
    /// Each unit's module: the index of its frequency in `frequencies`.
    pub unit_frequency: Vec<usize>,
    /// Each module's read space, `2 × 4|K|` orthonormal rows: the sum `E_k(a) + E_k(b)`.
    pub read_spaces: Vec<Array2<f64>>,
    /// A lower bound on the logit margin of `(a + b) mod p` over every other class and every
    /// pair, from the analysis in the module docs, before rounding.
    pub margin_floor: f64,
}

/// Modular addition mod odd `modulus` by `frequencies` at `phases` odd phases `≥ 5`, units
/// shuffled by `seed`.
pub fn fourier_modadd(modulus: usize, frequencies: &[usize], phases: usize, seed: u64) -> (FourierModAdd, FourierModAddTruth) {
    assert!(modulus % 2 == 1 && phases % 2 == 1 && phases >= 5, "odd modulus and odd phases ≥ 5");
    let width = 2 * frequencies.len();
    let angle = |k: usize, a: usize| 2.0 * PI * ((k * a) % modulus) as f64 / modulus as f64;
    let embedding = Array2::from_shape_fn((modulus, width), |(a, column)| {
        let theta = angle(frequencies[column / 2], a);
        if column % 2 == 0 { theta.cos() } else { theta.sin() }
    });
    let mut units: Vec<(usize, usize)> = (0..frequencies.len()).flat_map(|k| (0..phases).map(move |j| (k, j))).collect();
    units.shuffle(&mut StdRng::seed_from_u64(seed));
    let phase = |j: usize| 2.0 * PI * j as f64 / phases as f64;
    let mut w_in = Array2::<f64>::zeros((units.len(), 2 * width));
    let mut w_out = Array2::<f64>::zeros((modulus, units.len()));
    for (unit, &(k, j)) in units.iter().enumerate() {
        // cos(u + θ) = cos θ cos u − sin θ sin u, on both operands.
        let (sin, cos) = phase(j).sin_cos();
        for operand in 0..2 {
            w_in[[unit, operand * width + 2 * k]] = cos;
            w_in[[unit, operand * width + 2 * k + 1]] = -sin;
        }
        for c in 0..modulus {
            w_out[[c, unit]] = (2.0 * phase(j) + angle(frequencies[k], c)).cos();
        }
    }
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let read_spaces = (0..frequencies.len())
        .map(|k| {
            let mut rows = Array2::<f64>::zeros((2, 2 * width));
            for (row, column) in [(0, 2 * k), (1, 2 * k + 1)] {
                rows[[row, column]] = half;
                rows[[row, width + column]] = half;
            }
            rows
        })
        .collect();
    let margin_floor = fourier_margin_floor(modulus, frequencies, phases);
    let unit_frequency = units.iter().map(|&(k, _)| k).collect();
    (
        FourierModAdd { modulus, embedding, w_in, w_out },
        FourierModAddTruth {
            frequencies: frequencies.to_vec(),
            phases,
            unit_frequency,
            read_spaces,
            margin_floor,
        },
    )
}

/// The analytic margin: the main term `(J/2)(4/3π) Σ_k |cos(ω_k(a − b)/2)| (1 − cos ω_k δ)`
/// minimized over pairs and `δ ≠ 0`, less twice the aliases' total, each alias of harmonic
/// `n` bounded by `(J/2)(4/π)|cos(ω_k(a − b)/2)|/(n² − 1)` (with the `4/π`, not `4/3π`,
/// because its phase is free), summed over `n ≡ ±2 (mod J)` even, up to a tail
/// `Σ_{n > N} 4/(π(n² − 1)) < 4/(π(N − 1))`.
fn fourier_margin_floor(modulus: usize, frequencies: &[usize], phases: usize) -> f64 {
    let j = phases as f64;
    let omega = |k: usize| 2.0 * PI * k as f64 / modulus as f64;
    let cutoff = 64 * phases;
    let alias: f64 = (2..=cutoff)
        .filter(|n| n % 2 == 0 && *n > 2 && (n % phases == 2 || n % phases == phases - 2))
        .map(|n| 4.0 / (PI * ((n * n) as f64 - 1.0)))
        .sum::<f64>()
        + 4.0 / (PI * (cutoff as f64 - 1.0));
    let mut floor = f64::INFINITY;
    for difference in 0..modulus {
        let weights: Vec<f64> =
            frequencies.iter().map(|&k| (omega(k) * difference as f64 / 2.0).cos().abs()).collect();
        let total_weight: f64 = weights.iter().sum();
        for delta in 1..modulus {
            let main: f64 = frequencies
                .iter()
                .zip(&weights)
                .map(|(&k, &weight)| weight * (1.0 - (omega(k) * delta as f64).cos()))
                .sum::<f64>()
                * 4.0
                / (3.0 * PI);
            floor = floor.min(j / 2.0 * (main - 2.0 * alias * total_weight));
        }
    }
    floor
}

/// Vocabulary of the induction toy; token 0 is the beginning of sequence.
pub const INDUCTION_VOCAB: usize = 8;
/// Residual width: token, previous-token and output subspaces of `INDUCTION_VOCAB` each.
pub const INDUCTION_WIDTH: usize = 3 * INDUCTION_VOCAB;
/// Rotary planes of the previous-token law, at `ω_f = 2πf/16`, `f = 1..7`.
const PREVIOUS_PLANES: usize = 7;

/// A two-layer attention-only transformer: `h_0 = E[tokens]`, `h_{l+1} = h_l + attn_l(h_l)`,
/// `logits = h_2 Uᵀ`.
#[derive(Clone, Debug)]
pub struct InductionToy {
    /// `vocab × width`: token `t` is the unit vector of the token subspace.
    pub embedding: Array2<f64>,
    pub layers: [NativeAttention; 2],
    /// `vocab × width`: reads the output subspace.
    pub unembedding: Array2<f64>,
}

/// One executed forward pass: each layer's execution and the logits.
pub struct InductionRun {
    pub layers: Vec<AttentionExecution>,
    pub logits: Array2<f64>,
}

impl InductionToy {
    pub fn forward(&self, governor: &MemoryGovernor, tokens: &[usize]) -> InductionRun {
        let positions: Vec<i64> = (0..tokens.len() as i64).collect();
        let mut residual = Array2::<f64>::zeros((tokens.len(), INDUCTION_WIDTH));
        for (row, &token) in tokens.iter().enumerate() {
            residual.row_mut(row).assign(&self.embedding.row(token));
        }
        let mut layers = Vec::with_capacity(2);
        for layer in &self.layers {
            let executed = layer.execute(governor, residual.view(), &positions).expect("toy attention executes");
            residual += &executed.output;
            layers.push(executed);
        }
        InductionRun { logits: residual.dot(&self.unembedding.t()), layers }
    }
}

/// The induction answer.
#[derive(Clone, Debug)]
pub struct InductionTruth {
    /// Layer 0's routing laws: one previous-token law over both heads.
    pub previous_token_laws: Vec<Vec<usize>>,
    /// Layer 1's routing laws: the induction head alone.
    pub induction_laws: Vec<Vec<usize>>,
    /// Layer 0's law transport `Σ_h O_h V_h`: the token subspace copied into the
    /// previous-token subspace, `width × width`.
    pub previous_token_transport: Array2<f64>,
    /// Layer 1's transport: the token subspace copied into the output subspace.
    pub copy_transport: Array2<f64>,
    /// Layer 1's score form `W_Qᵀ W_K` on the residual: current token against the key
    /// position's previous token, scaled by the score gain.
    pub induction_score_form: Array2<f64>,
    /// A lower bound on layer 0's score at `s = t − 1` over its score at any other `s ≤ t`:
    /// the kernel's peak `7` over its off-peak maximum `0`, times the gain.
    pub previous_token_gap: f64,
    /// A lower bound on layer 1's score at the position after the current token's first
    /// occurrence over any other: the gain times `1 − 2 (tokens − 1) e^{−gap₀}`, since layer 0
    /// leaves at most that mass off the previous token.
    pub induction_gap: f64,
}

/// Coordinates of the token, previous-token and output subspaces.
pub fn token_coordinate(token: usize) -> usize {
    token
}

pub fn previous_coordinate(token: usize) -> usize {
    INDUCTION_VOCAB + token
}

pub fn output_coordinate(token: usize) -> usize {
    2 * INDUCTION_VOCAB + token
}

/// The previous-token law's score gain and the induction head's: both clear the causal
/// window's `ln(tokens)` by a wide margin.
const PREVIOUS_GAIN: f64 = 8.0;
const INDUCTION_GAIN: f64 = 64.0;

pub fn induction_toy() -> (InductionToy, InductionTruth) {
    let (vocab, width) = (INDUCTION_VOCAB, INDUCTION_WIDTH);
    let half = vocab / 2;
    let head_dim = 2 * PREVIOUS_PLANES;
    let embedding = Array2::from_shape_fn((vocab, width), |(token, column)| f64::from(u8::from(column == token_coordinate(token))));
    let unembedding = Array2::from_shape_fn((vocab, width), |(token, column)| f64::from(u8::from(column == output_coordinate(token))));

    // Layer 0: constant queries and keys, so the score is `Σ_f cos((s − t + 1) ω_f)` times
    // the gain, a Dirichlet kernel whose only peak on a window shorter than 16 is `s = t − 1`.
    let frequencies: Vec<f64> = (1..=PREVIOUS_PLANES).map(|f| 2.0 * PI * f as f64 / 16.0).collect();
    let rotary = RotaryEmbedding {
        pairing: RotaryPairing::HalfSplit,
        inverse_frequencies: frequencies.clone(),
        attention_scaling: 1.0,
    };
    let mut query_bias = Array1::<f64>::zeros(2 * head_dim);
    let mut key_bias = Array1::<f64>::zeros(2 * head_dim);
    for head in 0..2 {
        for (plane, &omega) in frequencies.iter().enumerate() {
            let (a, b) = rotary.plane(plane);
            query_bias[head * head_dim + a] = PREVIOUS_GAIN;
            key_bias[head * head_dim + a] = omega.cos();
            key_bias[head * head_dim + b] = omega.sin();
        }
    }
    let mut value = Array2::<f64>::zeros((2 * head_dim, width));
    let mut output = Array2::<f64>::zeros((width, 2 * head_dim));
    for head in 0..2 {
        for slot in 0..half {
            let token = head * half + slot;
            value[[head * head_dim + slot, token_coordinate(token)]] = 1.0;
            output[[previous_coordinate(token), head * head_dim + slot]] = 1.0;
        }
    }
    let previous = NativeAttention::new(
        AttentionGeometry { model_dim: width, n_heads: 2, n_kv_heads: 2, head_dim },
        rotary,
        1.0,
        AffineProjection { weight: Array2::zeros((2 * head_dim, width)), bias: query_bias },
        AffineProjection { weight: Array2::zeros((2 * head_dim, width)), bias: key_bias },
        AffineProjection { weight: value, bias: Array1::zeros(2 * head_dim) },
        AffineProjection { weight: output, bias: Array1::zeros(width) },
    )
    .expect("previous-token layer");

    // Layer 1: no rotary; query reads the token, key the previous token, value the token.
    let read = |coordinate: fn(usize) -> usize, gain: f64| {
        Array2::from_shape_fn((vocab, width), |(slot, column)| if column == coordinate(slot) { gain } else { 0.0 })
    };
    let copy_out = read(output_coordinate, 1.0).t().to_owned();
    let induction = NativeAttention::new(
        AttentionGeometry { model_dim: width, n_heads: 1, n_kv_heads: 1, head_dim: vocab },
        RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: Vec::new(), attention_scaling: 1.0 },
        1.0,
        AffineProjection { weight: read(token_coordinate, INDUCTION_GAIN), bias: Array1::zeros(vocab) },
        AffineProjection { weight: read(previous_coordinate, 1.0), bias: Array1::zeros(vocab) },
        AffineProjection { weight: read(token_coordinate, 1.0), bias: Array1::zeros(vocab) },
        AffineProjection { weight: copy_out, bias: Array1::zeros(width) },
    )
    .expect("induction layer");

    let map = |from: fn(usize) -> usize, to: fn(usize) -> usize, gain: f64| {
        let mut matrix = Array2::<f64>::zeros((width, width));
        for token in 0..vocab {
            matrix[[to(token), from(token)]] = gain;
        }
        matrix
    };
    let truth = InductionTruth {
        previous_token_laws: vec![vec![0, 1]],
        induction_laws: vec![vec![0]],
        previous_token_transport: map(token_coordinate, previous_coordinate, 1.0),
        copy_transport: map(token_coordinate, output_coordinate, 1.0),
        // Row: the query's residual coordinate; column: the key's.
        induction_score_form: map(previous_coordinate, token_coordinate, INDUCTION_GAIN),
        previous_token_gap: PREVIOUS_PLANES as f64 * PREVIOUS_GAIN,
        induction_gap: INDUCTION_GAIN
            * (1.0 - 2.0 * (2 * vocab - 2) as f64 * (-(PREVIOUS_PLANES as f64) * PREVIOUS_GAIN).exp()),
    };
    (InductionToy { embedding, layers: [previous, induction], unembedding }, truth)
}

/// `[BOS, x, x]` for a random arrangement `x` of the non-BOS tokens: the induction task.
/// Position `n + 1 + i` holds `x_i`, and for `i < n − 1` the answer there is `x_{i+1}`.
pub fn repeated_sequence(seed: u64) -> Vec<usize> {
    let mut tokens: Vec<usize> = (1..INDUCTION_VOCAB).collect();
    tokens.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut sequence = vec![0];
    sequence.extend(&tokens);
    sequence.extend(&tokens);
    sequence
}

/// Dyadic inputs `k/8`, `|k| ≤ 8`: every product with a `±1/4` frame entry is exact.
pub fn dyadic_features(seed: u64, samples: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((samples, RESID_FEATURES), || f64::from(rng.random_range(-8..=8)) / 8.0)
}
