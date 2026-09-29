//! Exact execution of a LlamaSimpleMLP decoder from its safetensors checkpoint (#2951).
//!
//! The source is `param_decomp/pretrain/models/llama_simple_mlp.py` (goodfire-ai/spd), the
//! architecture of the VPD paper's 4-layer Pile target: token embedding tied to the unembedding,
//! no positional embedding, and per layer
//!
//! ```text
//! x ← x + W_O attn(RoPE(W_Q n₁(x)), RoPE(W_K n₁(x)), W_V n₁(x))
//! x ← x + W_down g(W_fc n₂(x)),
//! ```
//!
//! then `logits = W_E n_f(x)`. Every norm is the RMS norm `w ⊙ x (mean x² + ε)^{-1/2}`, the
//! attention is causal multi-head attention with score multiplier `1/√head_dim` and the
//! rotate-half rotary embedding over the whole head, and `g` is the tanh GELU
//!
//! ```text
//! g(t) = ½ t (1 + tanh(k (t + c t³))),   k = fl(√(2/π)),  c = fl(0.044715).
//! ```
//!
//! No layer has a bias.
//!
//! # The program executed
//!
//! The program is the binary64 model: every stored tensor widened exactly
//! ([`super::safetensors`]), the declared constants `ε`, `k`, `c`, `1/√head_dim` as binary64
//! numbers, and the rotary inverse frequencies `1/fl(base^(i/P))`, `P = head_dim/2`. The source
//! trains and evaluates in lower precision: its norms cast to binary32, and it tabulates the
//! rotary angles in binary32. Those are roundings of the same real map, not part of it.
//!
//! # Radii
//!
//! Every stage carries a per-entry forward-error radius against the exact-arithmetic program at
//! the exact stored reals. It is built only from the owners the other native layers use:
//! [`head_rms_norm_with_radius`] for each norm (the whole row as one head),
//! [`linear_read`] for each projection and the unembedding,
//! [`RotaryCausalAttention::attend_projected`] for the rotary attention, and
//! [`log_softmax_with_error`] for the loss. Two owners are local:
//! * a residual addition `fl(x + y)` adds the rows' radii and its own rounding `u|fl(x + y)|`;
//! * the tanh GELU's band ([`gelu_tanh_with_radius`]).
//!
//! # Memory
//!
//! The binary64 weights are reserved on gam-runtime's memory governor when they are read. The
//! unembedding is executed in token tiles ([`LlamaSimpleMlp::next_token_loss`]), so a
//! `tokens × vocab` logit array exists for one tile at a time.

use std::f64::consts::PI;
use std::fmt;
use std::path::Path;

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::categorical::{CategoricalError, log_softmax_with_error};
use gam_runtime::resource::{Governed, MemoryGovernor};
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip, s};

use super::apply::ApplyError;
use super::attention::{
    AttentionGeometry, AttentionProgramError, ProjectedRows, RotaryCausalAttention, RotaryEmbedding, RotaryPairing,
    head_rms_norm_with_radius,
};
use super::block::{SUBNORMAL_SPACING, linear_read};
use super::gated_rewrite::GatedRewriteError;
use super::safetensors::{SafetensorsError, SafetensorsFile};

/// The source's `LlamaSimpleMLPConfig` fields the forward reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaSimpleMlpConfig {
    pub vocab: usize,
    pub model_dim: usize,
    pub layers: usize,
    pub heads: usize,
    /// `n_intermediate`, the MLP's hidden width.
    pub hidden: usize,
    pub rotary_base: f64,
    /// `rms_norm_eps`.
    pub epsilon: f64,
}

impl LlamaSimpleMlpConfig {
    pub fn head_dim(&self) -> usize {
        self.model_dim / self.heads
    }
}

#[derive(Debug)]
pub enum LlamaError {
    Config(String),
    Weights(SafetensorsError),
    Attention(AttentionProgramError),
    Read(ApplyError),
    Norm(GatedRewriteError),
    Loss(CategoricalError),
    /// A token id outside the vocabulary.
    Token { position: usize, token: u32 },
    /// A stage whose values or radius are not finite.
    NonFinite { stage: String },
}

impl fmt::Display for LlamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "config: {message}"),
            Self::Weights(error) => write!(f, "weights: {error}"),
            Self::Attention(error) => write!(f, "attention: {error}"),
            Self::Read(error) => write!(f, "linear read: {error}"),
            Self::Norm(error) => write!(f, "rms norm: {error}"),
            Self::Loss(error) => write!(f, "next-token loss: {error}"),
            Self::Token { position, token } => write!(f, "token {token} at position {position} is outside the vocabulary"),
            Self::NonFinite { stage } => write!(f, "{stage} is not finite"),
        }
    }
}

impl std::error::Error for LlamaError {}

impl From<SafetensorsError> for LlamaError {
    fn from(error: SafetensorsError) -> Self {
        Self::Weights(error)
    }
}

impl From<AttentionProgramError> for LlamaError {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(error)
    }
}

impl From<ApplyError> for LlamaError {
    fn from(error: ApplyError) -> Self {
        Self::Read(error)
    }
}

impl From<GatedRewriteError> for LlamaError {
    fn from(error: GatedRewriteError) -> Self {
        Self::Norm(error)
    }
}

/// One decoder layer's native tensors, each as the source stores it (`out × in`).
struct DecoderLayer {
    attention_gain: Array1<f64>,
    query: Governed<Array2<f64>>,
    key: Governed<Array2<f64>>,
    value: Governed<Array2<f64>>,
    output: Governed<Array2<f64>>,
    mlp_gain: Array1<f64>,
    up: Governed<Array2<f64>>,
    down: Governed<Array2<f64>>,
}

/// The source network on its native tensors.
pub struct LlamaSimpleMlp {
    config: LlamaSimpleMlpConfig,
    /// `wte.weight`, `vocab × model_dim`: the embedding table and, tied, the unembedding.
    embedding: Governed<Array2<f64>>,
    layers: Vec<DecoderLayer>,
    final_gain: Array1<f64>,
    attention: RotaryCausalAttention,
}

/// Rows with a per-entry radius against the exact program's rows.
#[derive(Clone, Debug)]
pub struct RadiusRows {
    pub values: Array2<f64>,
    pub radius: Array2<f64>,
}

impl RadiusRows {
    fn view(&self) -> ProjectedRows<'_> {
        ProjectedRows { values: self.values.view(), radius: self.radius.view() }
    }

    fn finite(self, stage: impl Fn() -> String) -> Result<Self, LlamaError> {
        if self.values.iter().chain(self.radius.iter()).all(|v| v.is_finite()) {
            Ok(self)
        } else {
            Err(LlamaError::NonFinite { stage: stage() })
        }
    }
}

/// One token's next-token cross-entropy `−log p(target)` and its radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenLoss {
    pub loss: f64,
    pub radius: f64,
    /// The largest logit's index, the greedy prediction.
    pub argmax: usize,
}

impl LlamaSimpleMlp {
    /// Read the source's state dict (`wte.weight`, `h.{i}.rms_1.weight`, `h.{i}.attn.{q,k,v,o}_proj.weight`,
    /// `h.{i}.rms_2.weight`, `h.{i}.mlp.{c_fc,down_proj}.weight`, `ln_f.weight`) from `path`.
    pub fn from_safetensors(
        governor: &MemoryGovernor,
        path: &Path,
        config: LlamaSimpleMlpConfig,
    ) -> Result<Self, LlamaError> {
        let (d, m, v) = (config.model_dim, config.hidden, config.vocab);
        if config.heads == 0 || !d.is_multiple_of(config.heads) || !config.head_dim().is_multiple_of(2) {
            return Err(LlamaError::Config(format!("{} heads do not split {d} into even head widths", config.heads)));
        }
        let file = SafetensorsFile::open(path)?;
        let layers = (0..config.layers)
            .map(|i| -> Result<DecoderLayer, LlamaError> {
                let name = |tail: &str| format!("h.{i}.{tail}");
                Ok(DecoderLayer {
                    attention_gain: file.vector(&name("rms_1.weight"), d)?,
                    query: file.matrix(governor, &name("attn.q_proj.weight"), d, d)?,
                    key: file.matrix(governor, &name("attn.k_proj.weight"), d, d)?,
                    value: file.matrix(governor, &name("attn.v_proj.weight"), d, d)?,
                    output: file.matrix(governor, &name("attn.o_proj.weight"), d, d)?,
                    mlp_gain: file.vector(&name("rms_2.weight"), d)?,
                    up: file.matrix(governor, &name("mlp.c_fc.weight"), m, d)?,
                    down: file.matrix(governor, &name("mlp.down_proj.weight"), d, m)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let head_dim = config.head_dim();
        let planes = head_dim / 2;
        let rotary = RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: (0..planes)
                .map(|i| 1.0 / config.rotary_base.powf(i as f64 / planes as f64))
                .collect(),
            attention_scaling: 1.0,
        };
        let geometry = AttentionGeometry { model_dim: d, n_heads: config.heads, n_kv_heads: config.heads, head_dim };
        let attention = RotaryCausalAttention::new(geometry, rotary, 1.0 / (head_dim as f64).sqrt())?;
        Ok(Self {
            config,
            embedding: file.matrix(governor, "wte.weight", v, d)?,
            layers,
            final_gain: file.vector("ln_f.weight", d)?,
            attention,
        })
    }

    pub fn config(&self) -> LlamaSimpleMlpConfig {
        self.config
    }

    /// The final-norm rows `n_f(x)` of one sequence at positions `0..tokens.len()`: the input
    /// the tied unembedding reads.
    pub fn final_rows(&self, governor: &MemoryGovernor, tokens: &[u32]) -> Result<RadiusRows, LlamaError> {
        let d = self.config.model_dim;
        let mut x = Array2::zeros((tokens.len(), d));
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= self.config.vocab {
                return Err(LlamaError::Token { position, token });
            }
            x.row_mut(position).assign(&self.embedding.row(token as usize));
        }
        // The embedding rows are stored reals: exact.
        let mut residual = RadiusRows { radius: Array2::zeros(x.dim()), values: x };
        let positions: Vec<i64> = (0..tokens.len() as i64).collect();
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = self.norm(&residual, &layer.attention_gain)?;
            let query = read(governor, &layer.query, &normed)?;
            let key = read(governor, &layer.key, &normed)?;
            let value = read(governor, &layer.value, &normed)?;
            let attended = self.attention.attend_projected(governor, query.view(), key.view(), value.view(), &positions)?;
            let mixed = ProjectedRows { values: attended.mixed.view(), radius: attended.mixed_radius.view() };
            let (output, output_radius) = linear_read(governor, layer.output.view(), mixed)?;
            residual = add_rows(&residual, output.view(), output_radius.view()).finite(|| format!("layer {i} attention residual"))?;
            let normed = self.norm(&residual, &layer.mlp_gain)?;
            let pre = read(governor, &layer.up, &normed)?;
            let activated = gelu_tanh_rows(&pre).finite(|| format!("layer {i} MLP activation"))?;
            let (output, output_radius) = linear_read(governor, layer.down.view(), activated.view())?;
            residual = add_rows(&residual, output.view(), output_radius.view()).finite(|| format!("layer {i} MLP residual"))?;
        }
        self.norm(&residual, &self.final_gain)?.finite(|| "final norm".into())
    }

    /// The tied unembedding `W_E r` of `rows` (`tokens × model_dim`) and its radius.
    pub fn logits(&self, governor: &MemoryGovernor, rows: ProjectedRows<'_>) -> Result<RadiusRows, LlamaError> {
        let (values, radius) = linear_read(governor, self.embedding.view(), rows)?;
        Ok(RadiusRows { values: values.to_owned(), radius })
    }

    /// The cross-entropy `−log p(targets[t])` of the logits at each position `t` of `tokens`
    /// (for next-token loss, `targets[t] = tokens[t + 1]` of the longer sequence), the
    /// unembedding executed `tile` positions at a time.
    ///
    /// With logits `ẑ` carrying radii `r`, [`log_softmax_with_error`]'s evaluation radius `e`
    /// covers the log weight at `ẑ`; moving every logit by at most `R = max r` moves both the
    /// target logit and the normalizer by at most `R`, so the loss radius is `e + 2R`.
    pub fn next_token_loss(
        &self,
        governor: &MemoryGovernor,
        tokens: &[u32],
        targets: &[u32],
        tile: usize,
    ) -> Result<Vec<TokenLoss>, LlamaError> {
        if targets.len() != tokens.len() {
            return Err(LlamaError::Config(format!("{} targets for {} tokens", targets.len(), tokens.len())));
        }
        if let Some((position, &token)) = targets.iter().enumerate().find(|(_, t)| **t as usize >= self.config.vocab) {
            return Err(LlamaError::Token { position, token });
        }
        let rows = self.final_rows(governor, tokens)?;
        let predicted = tokens.len();
        let mut losses = Vec::with_capacity(predicted);
        for start in (0..predicted).step_by(tile.max(1)) {
            let end = predicted.min(start + tile.max(1));
            let slice = ProjectedRows {
                values: rows.values.slice(s![start..end, ..]),
                radius: rows.radius.slice(s![start..end, ..]),
            };
            let logits = self.logits(governor, slice)?.finite(|| format!("logits at positions {start}..{end}"))?;
            for (offset, (row, radius)) in logits.values.outer_iter().zip(logits.radius.outer_iter()).enumerate() {
                let target = targets[start + offset] as usize;
                let row = row.to_vec();
                let (log_weights, evaluation) = log_softmax_with_error(&row).map_err(LlamaError::Loss)?;
                let widest = radius.iter().copied().fold(0.0, f64::max);
                let argmax = row
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |best, (j, &z)| if z > best.1 { (j, z) } else { best })
                    .0;
                losses.push(TokenLoss {
                    loss: -log_weights[target],
                    radius: ((evaluation[target] + 2.0 * widest) * (1.0 + 2.0 * UNIT_ROUNDOFF)).next_up(),
                    argmax,
                });
            }
        }
        Ok(losses)
    }

    /// The RMS norm of the whole row, `w ⊙ x (mean x² + ε)^{-1/2}`, through the per-head owner
    /// with one head as wide as the row.
    fn norm(&self, rows: &RadiusRows, gain: &Array1<f64>) -> Result<RadiusRows, LlamaError> {
        let (values, radius) = head_rms_norm_with_radius(rows.view(), self.config.model_dim, self.config.epsilon, gain.view())?;
        Ok(RadiusRows { values, radius })
    }
}

/// `x Wᵀ` of rows with a radius, through [`linear_read`].
fn read(governor: &MemoryGovernor, weight: &Array2<f64>, rows: &RadiusRows) -> Result<RadiusRows, LlamaError> {
    let (values, radius) = linear_read(governor, weight.view(), rows.view())?;
    Ok(RadiusRows { values: values.to_owned(), radius })
}

/// The residual addition `fl(x + y)`. The exact sum moves by at most `r_x + r_y` between the
/// computed and exact addends, and the addition rounds by at most `u |fl(x + y)|`. The bound's
/// own two additions and one product round by at most `γ_3 < 4u` relative, so it is inflated by
/// `1 + 4u`.
fn add_rows(rows: &RadiusRows, other: ArrayView2<'_, f64>, other_radius: ArrayView2<'_, f64>) -> RadiusRows {
    let values = &rows.values + &other;
    let mut radius = &rows.radius + &other_radius;
    Zip::from(&mut radius).and(&values).for_each(|r, &v| {
        *r = ((*r + UNIT_ROUNDOFF * v.abs()) * (1.0 + 4.0 * UNIT_ROUNDOFF)).next_up();
    });
    RadiusRows { values, radius }
}

/// `k = fl(√(2/π))`, the source's `math.sqrt(2.0 / math.pi)`.
fn gelu_tanh_scale() -> f64 {
    (2.0 / PI).sqrt()
}

/// `c = fl(0.044715)`.
const GELU_TANH_CUBIC: f64 = 0.044715;

/// The source's tanh GELU at `t`, in its own order of operations.
pub fn gelu_tanh(t: f64) -> f64 {
    0.5 * t * (1.0 + (gelu_tanh_scale() * (t + GELU_TANH_CUBIC * (t * t * t))).tanh())
}

/// An upper bound on `sech²` over `[lo, hi]`: one where the interval reaches zero, else
/// `1/cosh²` at the endpoint nearer zero, with `cosh` taken one ulp low (libm's accuracy) and
/// the quotient rounded up. A `cosh` that overflows leaves `SUBNORMAL_SPACING`, above the exact
/// `sech²` there.
fn sech_squared_bound(lo: f64, hi: f64) -> f64 {
    if lo <= 0.0 && hi >= 0.0 {
        return 1.0;
    }
    let nearest = lo.abs().min(hi.abs());
    let cosh = nearest.cosh() * (1.0 - 2.0 * UNIT_ROUNDOFF);
    (1.0 / (cosh * cosh) * (1.0 + 4.0 * UNIT_ROUNDOFF) + SUBNORMAL_SPACING).min(1.0)
}

/// [`gelu_tanh`] at `t` with a radius against the exact `g(x)` for every `|x − t| ≤ r`.
///
/// **Rounding at `t`.** `û = fl(k fl(t + fl(c fl(fl(t·t)·t))))`: `t` and `c t³` share a sign, so
/// the sum does not cancel, and the five operations give `|û − u| ≤ δ = γ_5 |û| / (1 − γ_5)`,
/// plus `4 · 2^-1074` for a cube that rounds into the subnormal range. `tanh` is 1-Lipschitz
/// with slope `sech²`, so `|tanh û − tanh u| ≤ S δ` with `S` the largest `sech²` on `[û − δ, û + δ]`,
/// and libm's `tanh` adds one ulp, at most `2u|τ̂|` (plus the subnormal spacing). The sum
/// `σ̂ = fl(1 + τ̂)` adds `u|σ̂|`, and `fl(fl(½t) σ̂)` adds `u|ĝ|` and a subnormal spacing. So
/// `|ĝ − g(t)| ≤ ½|t| (S δ + 2u|τ̂| + u|σ̂| + 2^-1074) + u|ĝ| + 2 · 2^-1074`.
///
/// **Propagation over `|x − t| ≤ r`.** `g(x) − g(t) = ½(x − t)(1 + tanh u(x)) + ½ t (tanh u(x) − tanh u(t))`,
/// and `|u(x) − u(t)| = k|x − t + c(x³ − t³)| ≤ Δ = k r (1 + c(3t² + 3|t|r + r²))`. On the box,
/// `u(x)` lies in `[û − δ − Δ, û + δ + Δ]`, where `sech²` is at most `S'` and `tanh` at most
/// `τ⁺ = tanh(û + δ + Δ)` plus one ulp. So `|g(x) − g(t)| ≤ ½ r (1 + τ⁺) + ½|t| S' Δ`.
///
/// Each bound is a sum and product of at most twelve nonnegative computed terms, whose own
/// rounding is below `γ_12 < 16u` relative; the radius is inflated by `1 + 16u` and one step up.
pub fn gelu_tanh_with_radius(t: f64, r: f64) -> (f64, f64) {
    let k = gelu_tanh_scale();
    let inner = k * (t + GELU_TANH_CUBIC * (t * t * t));
    let tau = inner.tanh();
    let sigma = 1.0 + tau;
    let value = 0.5 * t * sigma;
    let gamma = accumulation_growth(5);
    let delta = gamma * inner.abs() / (1.0 - gamma) + 4.0 * SUBNORMAL_SPACING;
    let slope = sech_squared_bound(inner - delta, inner + delta);
    let tanh_error = slope * delta + 2.0 * UNIT_ROUNDOFF * tau.abs() + SUBNORMAL_SPACING;
    let rounding = 0.5 * t.abs() * (tanh_error + UNIT_ROUNDOFF * sigma.abs()) + UNIT_ROUNDOFF * value.abs() + 2.0 * SUBNORMAL_SPACING;
    let propagation = if r > 0.0 {
        let spread = k * r * (1.0 + GELU_TANH_CUBIC * (3.0 * t * t + 3.0 * t.abs() * r + r * r));
        let (lo, hi) = (inner - delta - spread, inner + delta + spread);
        let upper = hi.tanh();
        let upper = (upper + 2.0 * UNIT_ROUNDOFF * upper.abs() + SUBNORMAL_SPACING).min(1.0);
        0.5 * r * (1.0 + upper) + 0.5 * t.abs() * sech_squared_bound(lo, hi) * spread
    } else {
        0.0
    };
    (value, ((rounding + propagation) * (1.0 + 16.0 * UNIT_ROUNDOFF)).next_up())
}

/// [`gelu_tanh_with_radius`] on every entry.
fn gelu_tanh_rows(rows: &RadiusRows) -> RadiusRows {
    let mut values = Array2::zeros(rows.values.dim());
    let mut radius = Array2::zeros(rows.values.dim());
    Zip::from(&mut values)
        .and(&mut radius)
        .and(&rows.values)
        .and(&rows.radius)
        .for_each(|value, bound, &t, &r| (*value, *bound) = gelu_tanh_with_radius(t, r));
    RadiusRows { values, radius }
}

/// The largest entry of each row of a nonnegative array, for reports.
pub fn row_max(radius: ArrayView2<'_, f64>) -> Vec<f64> {
    radius.map_axis(Axis(1), |row| row.iter().copied().fold(0.0, f64::max)).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qd::Quad;

    /// The tanh GELU in double-double: `tanh u = (e^{2u} − 1)/(e^{2u} + 1)`.
    fn gelu_tanh_quad(t: f64) -> Quad {
        let q = Quad::from_f64;
        let t = q(t);
        let u = q(gelu_tanh_scale()) * (t + q(GELU_TANH_CUBIC) * t * t * t);
        let e = (u * q(2.0)).exp();
        let tanh = if e.is_finite() { (e - q(1.0)) / (e + q(1.0)) } else { q(1.0) };
        q(0.5) * t * (q(1.0) + tanh)
    }

    /// `|computed − reference|` in its leading binary64 part.
    fn distance(computed: f64, reference: Quad) -> f64 {
        (Quad::from_f64(computed) - reference).0.abs()
    }

    #[test]
    fn gelu_tanh_band_holds_the_double_double_value() {
        let mut t = -12.0;
        while t <= 12.0 {
            let (value, radius) = gelu_tanh_with_radius(t, 0.0);
            assert_eq!(value.to_bits(), gelu_tanh(t).to_bits());
            let error = distance(value, gelu_tanh_quad(t));
            assert!(error <= radius, "t = {t}: error {error:e} above radius {radius:e}");
            assert!(radius <= 64.0 * f64::EPSILON * (1.0 + t.abs()), "t = {t}: radius {radius:e} is loose");
            t += 0.0137;
        }
    }

    #[test]
    fn gelu_tanh_radius_covers_every_point_of_the_box() {
        for &(t, r) in &[(-3.0, 0.5), (-0.75, 0.2), (0.0, 1e-3), (0.4, 0.05), (2.5, 1.0), (-9.0, 2.0), (6.0, 1e-9)] {
            let (value, radius) = gelu_tanh_with_radius(t, r);
            for step in 0..=200 {
                let x = t - r + 2.0 * r * step as f64 / 200.0;
                let error = distance(value, gelu_tanh_quad(x));
                assert!(error <= radius, "t = {t}, r = {r}, x = {x}: {error:e} above {radius:e}");
            }
        }
    }

    /// `sin x`, `cos x` in double-double by their Taylor series, for `|x| ≤ 8`.
    fn quad_sin_cos(x: Quad) -> (Quad, Quad) {
        let q = Quad::from_f64;
        let square = x * x;
        let (mut sin, mut cos) = (x, q(1.0));
        let (mut sin_term, mut cos_term) = (x, q(1.0));
        for n in 1..60 {
            let n = n as f64;
            sin_term = q(0.0) - sin_term * square / q((2.0 * n) * (2.0 * n + 1.0));
            cos_term = q(0.0) - cos_term * square / q((2.0 * n - 1.0) * (2.0 * n));
            sin = sin + sin_term;
            cos = cos + cos_term;
        }
        (sin, cos)
    }

    struct Tiny {
        config: LlamaSimpleMlpConfig,
        tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    }

    /// A two-layer model with small binary32 weights from a fixed linear congruential stream.
    fn tiny() -> Tiny {
        let config = LlamaSimpleMlpConfig { vocab: 13, model_dim: 8, layers: 2, heads: 2, hidden: 12, rotary_base: 10000.0, epsilon: 1e-6 };
        let mut state = 0x2951u64;
        let mut draw = |scale: f32| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
        };
        let (d, m, v) = (config.model_dim, config.hidden, config.vocab);
        let mut tensors = vec![("wte.weight".to_string(), vec![v, d], (0..v * d).map(|_| draw(2.0)).collect::<Vec<_>>())];
        for i in 0..config.layers {
            for (tail, shape, scale) in [
                ("rms_1.weight", vec![d], 0.0),
                ("attn.q_proj.weight", vec![d, d], 1.5),
                ("attn.k_proj.weight", vec![d, d], 1.5),
                ("attn.v_proj.weight", vec![d, d], 1.0),
                ("attn.o_proj.weight", vec![d, d], 1.0),
                ("rms_2.weight", vec![d], 0.0),
                ("mlp.c_fc.weight", vec![m, d], 1.5),
                ("mlp.down_proj.weight", vec![d, m], 1.0),
            ] {
                let len = shape.iter().product::<usize>();
                let values = (0..len).map(|_| if scale == 0.0 { 1.0 + draw(0.5) } else { draw(scale) }).collect();
                tensors.push((format!("h.{i}.{tail}"), shape, values));
            }
        }
        tensors.push(("ln_f.weight".to_string(), vec![d], (0..d).map(|_| 1.0 + draw(0.5)).collect()));
        Tiny { config, tensors }
    }

    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape, values) in tensors {
            let begin = data.len();
            values.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}));
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    /// The source's forward in double-double, straight from its definition.
    fn reference_logits(tiny: &Tiny, tokens: &[u32]) -> Vec<Vec<Quad>> {
        let q = Quad::from_f64;
        let c = tiny.config;
        let tensor = |name: &str| -> Vec<Quad> {
            tiny.tensors.iter().find(|(n, _, _)| n == name).unwrap().2.iter().map(|&v| q(f64::from(v))).collect()
        };
        let (d, m, hd) = (c.model_dim, c.hidden, c.head_dim());
        let matvec = |w: &[Quad], rows: usize, x: &[Quad]| -> Vec<Quad> {
            (0..rows).map(|r| (0..x.len()).fold(q(0.0), |acc, j| acc + w[r * x.len() + j] * x[j])).collect()
        };
        let norm = |x: &[Quad], gain: &[Quad]| -> Vec<Quad> {
            let mean = x.iter().fold(q(0.0), |acc, &v| acc + v * v) / q(d as f64) + q(c.epsilon);
            let scale = mean.sqrt().recip();
            x.iter().zip(gain).map(|(&v, &g)| g * (v * scale)).collect()
        };
        let planes = hd / 2;
        let rope = |x: &mut [Quad], position: usize| {
            for head in 0..c.heads {
                for p in 0..planes {
                    let inverse = 1.0 / c.rotary_base.powf(p as f64 / planes as f64);
                    // The double-double product of a small integer and a binary64 number is exact.
                    let (sin, cos) = quad_sin_cos(q(position as f64) * q(inverse));
                    let (a, b) = (head * hd + p, head * hd + p + planes);
                    let (xa, xb) = (x[a], x[b]);
                    x[a] = xa * cos - xb * sin;
                    x[b] = xb * cos + xa * sin;
                }
            }
        };
        let wte = tensor("wte.weight");
        let mut xs: Vec<Vec<Quad>> = tokens.iter().map(|&t| wte[t as usize * d..(t as usize + 1) * d].to_vec()).collect();
        for i in 0..c.layers {
            let name = |tail: &str| tensor(&format!("h.{i}.{tail}"));
            let (wq, wk, wv, wo) = (name("attn.q_proj.weight"), name("attn.k_proj.weight"), name("attn.v_proj.weight"), name("attn.o_proj.weight"));
            let normed: Vec<Vec<Quad>> = xs.iter().map(|x| norm(x, &name("rms_1.weight"))).collect();
            let mut qs: Vec<Vec<Quad>> = normed.iter().map(|h| matvec(&wq, d, h)).collect();
            let mut ks: Vec<Vec<Quad>> = normed.iter().map(|h| matvec(&wk, d, h)).collect();
            let vs: Vec<Vec<Quad>> = normed.iter().map(|h| matvec(&wv, d, h)).collect();
            (0..tokens.len()).for_each(|t| {
                rope(&mut qs[t], t);
                rope(&mut ks[t], t);
            });
            let scale = q(1.0 / (hd as f64).sqrt());
            for t in 0..tokens.len() {
                let mut mixed = vec![q(0.0); d];
                for head in 0..c.heads {
                    let range = head * hd..(head + 1) * hd;
                    let scores: Vec<Quad> = (0..=t)
                        .map(|s| range.clone().fold(q(0.0), |acc, j| acc + qs[t][j] * ks[s][j]) * scale)
                        .collect();
                    let top = scores.iter().map(|s| s.0).fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<Quad> = scores.iter().map(|&s| (s - q(top)).exp()).collect();
                    let total = weights.iter().fold(q(0.0), |acc, &w| acc + w);
                    for (s, &w) in weights.iter().enumerate() {
                        for j in range.clone() {
                            mixed[j] = mixed[j] + w / total * vs[s][j];
                        }
                    }
                }
                let out = matvec(&wo, d, &mixed);
                xs[t].iter_mut().zip(out).for_each(|(x, o)| *x = *x + o);
            }
            let (up, down) = (name("mlp.c_fc.weight"), name("mlp.down_proj.weight"));
            for x in xs.iter_mut() {
                let pre = matvec(&up, m, &norm(x, &name("rms_2.weight")));
                let act: Vec<Quad> = pre.iter().map(|&t| {
                    let u = q(gelu_tanh_scale()) * (t + q(GELU_TANH_CUBIC) * t * t * t);
                    let e = (u * q(2.0)).exp();
                    let tanh = if e.is_finite() { (e - q(1.0)) / (e + q(1.0)) } else { q(1.0) };
                    q(0.5) * t * (q(1.0) + tanh)
                }).collect();
                let out = matvec(&down, d, &act);
                x.iter_mut().zip(out).for_each(|(x, o)| *x = *x + o);
            }
        }
        xs.iter().map(|x| matvec(&wte, c.vocab, &norm(x, &tensor("ln_f.weight")))).collect()
    }

    #[test]
    fn forward_radius_holds_the_double_double_forward() {
        let tiny = tiny();
        let dir = std::env::temp_dir().join(format!("gam-llama-simple-mlp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        write_safetensors(&path, &tiny.tensors);
        let governor = crate::test_support::test_governor();
        let model = LlamaSimpleMlp::from_safetensors(governor, &path, tiny.config).unwrap();
        let tokens = [3u32, 11, 0, 7, 7, 12];
        let rows = model.final_rows(governor, &tokens).unwrap();
        let logits = model
            .logits(governor, ProjectedRows { values: rows.values.view(), radius: rows.radius.view() })
            .unwrap();
        let reference = reference_logits(&tiny, &tokens);
        for (t, row) in reference.iter().enumerate() {
            for (j, &exact) in row.iter().enumerate() {
                let (value, radius) = (logits.values[[t, j]], logits.radius[[t, j]]);
                assert!(distance(value, exact) <= radius, "logit ({t}, {j}): {value} vs {exact:?}, radius {radius:e}");
                assert!(radius < 1e-9, "logit ({t}, {j}) radius {radius:e} is loose");
            }
        }
        let losses = model.next_token_loss(governor, &tokens[..5], &tokens[1..], 4).unwrap();
        assert_eq!(losses.len(), tokens.len() - 1);
        for (t, loss) in losses.iter().enumerate() {
            let row = &reference[t];
            let top = row.iter().map(|z| z.0).fold(f64::NEG_INFINITY, f64::max);
            let total = row.iter().fold(Quad::from_f64(0.0), |acc, &z| acc + (z - Quad::from_f64(top)).exp());
            let exact = total.ln() + Quad::from_f64(top) - row[tokens[t + 1] as usize];
            assert!(distance(loss.loss, exact) <= loss.radius, "loss at {t}");
        }
        assert!(matches!(model.final_rows(governor, &[13]), Err(LlamaError::Token { position: 0, token: 13 })));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn residual_addition_band_covers_the_rounding() {
        let rows = RadiusRows { values: Array2::from_elem((1, 2), 1.0), radius: Array2::from_elem((1, 2), 1e-12) };
        let other = Array2::from_elem((1, 2), f64::EPSILON / 4.0);
        let sum = add_rows(&rows, other.view(), Array2::zeros((1, 2)).view());
        // `1 + ε/4` rounds to `1`, an error of `ε/4 = u/2`.
        assert_eq!(sum.values[[0, 0]], 1.0);
        assert!(sum.radius[[0, 0]] >= 1e-12 + f64::EPSILON / 4.0);
    }
}
