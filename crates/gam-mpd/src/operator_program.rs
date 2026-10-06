//! The operator program (#2951): an executable mechanism program whose components are
//! operators with exact interfaces, a law, and a parameter source in the original weights.
//!
//! # The object
//!
//! An [`OperatorProgram`] is a DAG of [`Node`]s evaluated in one batch over a declared finite
//! input family (every value is `N × width`, one row per input; positions are unrolled into
//! separate nodes). Every node's output is an [`Interface`]: a width partitioned into labelled
//! coordinate groups (a constant, a rotation plane `k`, a unit `n`, a token, a position). The
//! reals live in [`Operator`]s, shared by every node that references them, so a tied weight is
//! sent once. An operator is a block matrix over (row group × column group); each block is
//! present or absent, and the present blocks' reals sit on one dyadic lattice
//! ([`DeclaredPrecision`]). A component is therefore an operator with an input subspace (its
//! present column groups), a law (the node kind that applies it) and an output subspace (its
//! present row groups), and a [`Provenance`] naming the native tensors and the exact rewrite
//! chain that produced its reals. It is not an independently deletable piece of parameters.
//!
//! # Execution with forward-error bands
//!
//! [`OperatorProgram::execute`] returns every node's value and, when asked, a per-entry radius
//! `r` with `|computed − exact| ≤ r` against the exact-arithmetic program at the exact decoded
//! reals (every real is a lattice point, so the operators themselves carry no error). With
//! `u` the unit roundoff and `γ_k = ku/(1 − ku)`:
//!
//! * **Feature.** An indicator is exact. An indicator feature read only as affine terms is a
//!   *gathered* feature ([`OperatorProgram::gathered_tokens`]): its one-hot rows are never formed
//!   (its trace value and band hold no columns), and each affine term reading it adds the
//!   operator's column at the row's token, an exact one times each entry.
//! * **Affine** `y = Σ_t x_t A_tᵀ + b` over `K = Σ_t cols_t + 1` summands per entry: in any
//!   summation order `|fl(y) − y| ≤ γ_K (Σ_t |x_t||A_t|ᵀ + |b|)` (Higham, Lemma 3.1), and an
//!   input radius `r_t` moves the exact output by at most `r_t |A_t|ᵀ`. Both are one product
//!   `(γ_K |x_t| + r_t)|A_t|ᵀ`.
//! * **Bilinear** `s = c Σ_i l_i r_i` with a scale `c` that is exact by declaration: rounding
//!   `γ_{d+1}|c| Σ|l_i||r_i|`, propagation `|c| Σ (|l_i| ρ^r_i + ρ^l_i |r_i| + ρ^l_i ρ^r_i)`.
//! * **Softmax** over `J` scores with radii `ρ_j`, `R = max ρ`: at exact scores every weight moves
//!   by a factor in `[e^{−2R}, e^{2R}]`, so by at most `α_j (e^{2R} − 1)`. The computation
//!   `α̂_j = exp(fl(s_j − m)) / Σ` rounds the subtraction (`u|s_j − m|` in the exponent), the
//!   exponential (one ulp, `2u` relative), the sum (`γ_{J−1}`) and the quotient (`u`), a
//!   relative error `η ≤ exp(2u max|s − m|)(1 + 2u)²(1 + γ_{J−1})(1 + u) − 1`. So
//!   `|α̂_j − α̃_j| ≤ α̂_j (η + e^{2R} − 1)/(1 − η)`.
//! * **Mix** `y = Σ_j α_j p_j` and **Hadamard** (`J = 1`): rounding `γ_J Σ|α_j||p_j|`,
//!   propagation `Σ (|α_j| ρ^p + ρ^α |p_j| + ρ^α ρ^p)`.
//! * **Pointwise.** ReLU and the identity are 1-Lipschitz and exact, so the radius passes; the
//!   zero law has none.
//! * **Readout** `ℓ = y Φᵀ` with `Φ` the output basis at the classes (radius from the Feature
//!   rule): rounding `γ_K |y||Φ|ᵀ`, propagation `ρ^y (|Φ| + ρ^Φ)ᵀ + |y| (ρ^Φ)ᵀ`.
//!
//! Each radius is a sum of nonnegative computed terms; a computed nonnegative sum of `k` terms is
//! at least `(1 − γ_k)` times the exact one, so every radius is inflated by `1/(1 − γ_{k+2})`,
//! bounded above by `1 + 2γ_{k+2}` for `γ_{k+2} ≤ 1/2`.
//!
//! # The code
//!
//! [`OperatorProgram::encode`] writes one self-delimiting message through `codec`'s integer codes
//! and `precision`'s lattice codes, and the decoder reads it back given the [`Declarations`] (the
//! contract's input domains and slots, which the decoder already knows and which are never sent).
//! The message is:
//!
//! 1. the counts `#bases + 1`, `#operators + 1`, `#nodes` in the prefix code;
//! 2. each basis: its domain as a fixed index into the declared domains;
//! 3. each operator: its kind as a fixed index; its row and column interfaces as runs (count + 1,
//!    width, label kind as a fixed index, first label index + 1); for a dense operator, per row
//!    group the present column groups in the enumerative subset code, then the present reals as
//!    one lattice message: count + 1 and the fraction bits in the prefix codes, and each lattice
//!    index in the signed Elias δ code, which is subadditive, so splitting a real is never shorter.
//!    A dense operator whose rows' first present reals do not decrease down the rows may be sent as the ordered kind instead: each of those first reals after
//!    the first is sent as its increment over the previous one in the Elias δ code of `increment + 1`.
//!    Its rows are then a sorted list, whose order costs nothing, where the plain kind pays for an
//!    arbitrary order (about `log₂ n!` bits for `n` distinct rows). The encoder sends whichever kind
//!    is shorter, and the decoder recovers the positions from the present blocks. A diagonal
//!    operator (a norm gain) sends its one interface and its diagonal as one lattice message, and
//!    is held as that diagonal everywhere, never as a square matrix;
//! 4. each node: its kind as a fixed index and its references as fixed indices into the objects
//!    listed before it, a per-group law as a fixed index, a scale kind as a fixed index and its
//!    argument in the prefix code;
//! 5. the output node as a fixed index.
//!
//! [`Provenance`] is audit data about where the reals came from; the decoder needs none of it,
//! and it is not sent.

use super::codec::{
    BitReader, BitString, CodecError, decode_elias_delta, decode_fixed_index, decode_prefix_integer, decode_signed_delta,
    decode_signed_prefix_integer, decode_subset, elias_delta_len_bits, encode_elias_delta, encode_fixed_index,
    encode_prefix_integer, encode_signed_delta, encode_signed_prefix_integer, encode_subset, fixed_index_len_bits,
    prefix_integer_len_bits, signed_delta_len_bits,
    signed_prefix_integer_len_bits, skip_elias_delta, subset_code_len_bits,
};
use rayon::prelude::*;
use super::precision::{DecodableArtifact, DeclaredPrecision, LatticeCode};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::probability::{NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_cdf_and_pdf};
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::fmt;
use std::ops::Range;

/// What a coordinate group of an interface is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabelKind {
    /// A native coordinate block with no finer structure.
    Native,
    /// The constant function of a basis (the character `1`), or the constant input of a bias.
    Const,
    /// Rotation plane `k` of a character or recovered plane basis (two coordinates).
    Plane,
    /// One unit (a neuron, a head coordinate) of a pointwise law.
    Unit,
    /// One token of a finite domain.
    Token,
    /// One key position of a routing law.
    Position,
    /// One shared scalar control read by several units.
    Control,
    /// The product of group `index / G₂` of a left interface and group `index mod G₂` of a right
    /// interface of `G₂` groups.
    Pair,
    /// One coordinate of a fitted factor basis shared by the operators that read or write it.
    Factor,
}

const LABEL_KINDS: [LabelKind; 9] = [
    LabelKind::Native,
    LabelKind::Const,
    LabelKind::Plane,
    LabelKind::Unit,
    LabelKind::Token,
    LabelKind::Position,
    LabelKind::Control,
    LabelKind::Pair,
    LabelKind::Factor,
];

/// A labelled coordinate group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label {
    pub kind: LabelKind,
    pub index: u32,
}

impl Label {
    pub const fn new(kind: LabelKind, index: u32) -> Self {
        Self { kind, index }
    }
}

/// One coordinate group of an interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Group {
    pub width: usize,
    pub label: Label,
}

/// A width partitioned into labelled coordinate groups, in order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Interface {
    groups: Vec<Group>,
    offsets: Vec<usize>,
}

impl Interface {
    /// Refused for an empty interface or a group of width zero.
    pub fn new(groups: Vec<Group>) -> Result<Self, ProgramError> {
        if groups.is_empty() || groups.iter().any(|group| group.width == 0) {
            return Err(ProgramError::Interface("an interface needs groups of positive width".to_string()));
        }
        let mut offsets = Vec::with_capacity(groups.len() + 1);
        let mut offset = 0;
        offsets.push(0);
        for group in &groups {
            offset += group.width;
            offsets.push(offset);
        }
        Ok(Self { groups, offsets })
    }

    /// `count` groups of one width and label kind, labelled `first, first + 1, …`.
    pub fn uniform(count: usize, width: usize, kind: LabelKind, first: u32) -> Result<Self, ProgramError> {
        Self::new((0..count).map(|i| Group { width, label: Label::new(kind, first + i as u32) }).collect())
    }

    /// One native group of `width`.
    pub fn native(width: usize) -> Result<Self, ProgramError> {
        Self::uniform(1, width, LabelKind::Native, 0)
    }

    /// The one-coordinate constant interface a bias or a constant is read from.
    pub fn constant() -> Self {
        Self { groups: vec![Group { width: 1, label: Label::new(LabelKind::Const, 0) }], offsets: vec![0, 1] }
    }

    pub fn width(&self) -> usize {
        self.offsets[self.groups.len()]
    }

    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The coordinate range of group `g`.
    pub fn range(&self, group: usize) -> Range<usize> {
        self.offsets[group]..self.offsets[group + 1]
    }

    /// Maximal runs of groups with one width and kind and consecutive label indices.
    fn runs(&self) -> Vec<(usize, usize, LabelKind, u32)> {
        let mut runs: Vec<(usize, usize, LabelKind, u32)> = Vec::new();
        for group in &self.groups {
            if let Some(last) = runs.last_mut()
                && last.1 == group.width
                && last.2 == group.label.kind
                && last.3 + last.0 as u32 == group.label.index
            {
                last.0 += 1;
                continue;
            }
            runs.push((1, group.width, group.label.kind, group.label.index));
        }
        runs
    }
}

/// A finite input domain the contract declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub size: usize,
}

/// An input slot of the contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Slot {
    /// One token of a declared domain per input.
    Token { domain: usize },
    /// One real vector per input (a harvested residual, for instance).
    Raw { width: usize },
}

/// What the contract declares and the decoder therefore knows: domains and slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declarations {
    pub domains: Vec<Domain>,
    pub slots: Vec<Slot>,
    /// Declared scalar intervention variables (a control gain, an edit strength); a [`Node::Gain`]
    /// reads them. The native setting is every parameter at `1`.
    pub parameters: usize,
}

/// A basis of functions on a finite domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Basis {
    /// The indicator of each token: groups `Token(t)`, width 1 each.
    Indicator { domain: usize },
}

/// A matrix with a per-entry bound on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedMatrix {
    pub values: Array2<f64>,
    pub bands: Array2<f64>,
}
impl Basis {
    pub fn domain(&self) -> usize {
        match self {
            Self::Indicator { domain } => *domain,
        }
    }

    /// The basis interface.
    pub fn interface(&self, declarations: &Declarations) -> Result<Interface, ProgramError> {
        let size = declarations
            .domains
            .get(self.domain())
            .ok_or(ProgramError::Reference { what: "basis domain", index: self.domain() })?
            .size;
        match self {
            Self::Indicator { .. } => Interface::uniform(size, 1, LabelKind::Token, 0),
        }
    }

    /// `y Φᵀ`: rows of basis coordinates `y` read at every class of the domain (`Φ` is the basis at
    /// every class, `classes × width`). An indicator basis is the identity, so its read is `y`
    /// itself, exactly, and no `classes × classes` table is formed (a 50k-token vocabulary's would be
    /// 20 GB).
    pub fn read(&self, declarations: &Declarations, y: &Array2<f64>) -> Result<Array2<f64>, ProgramError> {
        let size = declarations.domains.get(self.domain())
            .ok_or(ProgramError::Reference { what: "basis domain", index: self.domain() })?.size;
        if y.ncols() != size {
            return Err(ProgramError::Input(format!("indicator read has {} coordinates, expected {size}", y.ncols())));
        }
        Ok(y.clone())
    }

    /// `g Φ`: the transpose of [`Basis::read`], from classes back to basis coordinates.
    pub fn read_transpose(&self, declarations: &Declarations, g: &Array2<f64>) -> Result<Array2<f64>, ProgramError> {
        self.read(declarations, g)
    }

    /// The banded [`Basis::read`]: an indicator's read is exact, so the band is `y`'s own `ry`.
    pub fn read_banded(&self, declarations: &Declarations, y: &Array2<f64>, ry: Option<&Array2<f64>>) -> Result<(Array2<f64>, Option<Array2<f64>>), ProgramError> {
        if ry.is_some_and(|r| r.dim() != y.dim()) {
            return Err(ProgramError::Input("indicator read's radius shape differs from its values".into()));
        }
        Ok((self.read(declarations, y)?, ry.cloned()))
    }

    /// The basis evaluated at `tokens`, one row per token, with the Feature radii of the module note.
    pub fn evaluate(&self, declarations: &Declarations, tokens: &[u32]) -> Result<BandedMatrix, ProgramError> {
        let interface = self.interface(declarations)?;
        let width = interface.width();
        let mut values = Array2::<f64>::zeros((tokens.len(), width));
        let bands = Array2::<f64>::zeros((tokens.len(), width));
        match self {
            Self::Indicator { .. } => {
                for (row, &token) in tokens.iter().enumerate() {
                    if token as usize >= width {
                        return Err(ProgramError::Input(format!("token {token} outside a domain of {width}")));
                    }
                    values[[row, token as usize]] = 1.0;
                }
            }
        }
        Ok(BandedMatrix { values, bands })
    }
}

/// The elementwise law of a pointwise node, per input group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Law {
    Relu,
    Identity,
    Zero,
    /// `t σ(t)`, computed as `t / (1 + exp(−t))`.
    Silu,
    /// The exact GELU `t Φ(t)`, with `Φ` from the probability owner's proven table.
    Gelu,
    /// The tanh GELU `½ t (1 + tanh(√(2/π) (t + 0.044715 t³)))`, with the constants as the
    /// architecture declares them.
    GeluTanh,
}

pub const LAWS: [Law; 6] = [Law::Relu, Law::Identity, Law::Zero, Law::Silu, Law::Gelu, Law::GeluTanh];

/// The largest slope of the tanh GELU is below this (it peaks at about 1.1289 near `t ≈ 1.5`).
const GELU_TANH_LIPSCHITZ: f64 = 1.13;

/// `½ t (1 + tanh(√(2/π)(t + 0.044715 t³)))` with `√(2/π)` as its `f64`.
fn gelu_tanh(t: f64) -> f64 {
    let inner = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * (t + 0.044715 * t * t * t);
    0.5 * t * (1.0 + inner.tanh())
}

/// The largest slope of the exact GELU, `max_t |Φ(t) + t φ(t)| = Φ(√2) + √2 φ(√2) < 1.129`.
const GELU_LIPSCHITZ: f64 = 1.129;

/// The largest slope of SiLU, `max_t |d/dt t σ(t)| < 1.0999` (attained near `t ≈ 2.40`).
const SILU_LIPSCHITZ: f64 = 1.0999;

/// `t / (1 + exp(−t))`: the exponential within one ulp (`2u` relative), the sum and the quotient
/// one rounding each, so the computed value is within `5u` of `t σ(t)` relative to itself.
fn silu(t: f64) -> f64 {
    t / (1.0 + (-t).exp())
}

impl Law {
    /// The law at `t`.
    pub fn apply(self, t: f64) -> f64 {
        match self {
            Self::Relu => t.max(0.0),
            Self::Identity => t,
            Self::Zero => 0.0,
            Self::Silu => silu(t),
            Self::Gelu => t * normal_cdf_and_pdf(t).0,
            Self::GeluTanh => gelu_tanh(t),
        }
    }

    /// The law's derivative at `t` (ReLU's is `0` at the kink, its left derivative).
    pub fn derivative(self, t: f64) -> f64 {
        match self {
            Self::Relu => {
                if t > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Identity => 1.0,
            Self::Zero => 0.0,
            Self::Silu => {
                let sigma = 1.0 / (1.0 + (-t).exp());
                sigma * (1.0 + t * (1.0 - sigma))
            }
            Self::Gelu => {
                let (cdf, pdf) = normal_cdf_and_pdf(t);
                cdf + t * pdf
            }
            Self::GeluTanh => {
                let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
                let inner = c * (t + 0.044715 * t * t * t);
                let th = inner.tanh();
                0.5 * (1.0 + th) + 0.5 * t * (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * t * t)
            }
        }
    }

    /// A bound on the law's slope: its Lipschitz constant on the reals.
    pub fn lipschitz(self) -> f64 {
        match self {
            Self::Relu | Self::Identity => 1.0,
            Self::Zero => 0.0,
            Self::Silu => SILU_LIPSCHITZ,
            Self::Gelu => GELU_LIPSCHITZ,
            Self::GeluTanh => GELU_TANH_LIPSCHITZ,
        }
    }

    /// The radius of the computed law `value` at a computed input `input` whose radius is `r`.
    pub(crate) fn radius(self, input: f64, value: f64, r: f64) -> f64 {
        match self {
            Self::Relu | Self::Identity => r,
            Self::Zero => 0.0,
            Self::Silu => (SILU_LIPSCHITZ * r + 5.0 * UNIT_ROUNDOFF * value.abs()).next_up(),
            // The inner argument `c(t + 0.044715 t³)` is computed within `δ = 8u·c(|t| + 0.044715|t|³)`
            // (five roundings and the constant's); tanh moves by at most `δ sech²(ξ)` for `ξ` between
            // the exact and computed arguments, and `sech²` decreases in `|ξ|`, so by
            // `δ sech²(max(0, |inner| − δ)) ≤ δ min(1, 4e^{−2(|inner|−δ)})`: a saturated unit's
            // rounding does not grow with `|t|³`. libm's tanh (one ulp, `2u`) and `1 + tanh` (`2u`)
            // add `4u` to `1 + tanh`, times `½|t|`; the final products add `3u|value|`.
            Self::GeluTanh => {
                let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
                let t = input.abs();
                let reach = (c * (t + 0.044715 * t * t * t)).next_up();
                let delta = (8.0 * UNIT_ROUNDOFF * reach).next_up();
                let x = (reach * (1.0 - 8.0 * UNIT_ROUNDOFF) - delta).max(0.0);
                let slope = (4.0 * (-2.0 * x).exp() * (1.0 + 4.0 * UNIT_ROUNDOFF)).min(1.0);
                (GELU_TANH_LIPSCHITZ * r + 0.5 * t * (delta * slope + 4.0 * UNIT_ROUNDOFF) + 3.0 * UNIT_ROUNDOFF * value.abs())
                    .next_up()
                    .next_up()
            }
            Self::Gelu => (GELU_LIPSCHITZ * r
                + 2.0 * NORMAL_CDF_RELATIVE_ERROR * value.abs()
                + input.abs() * NORMAL_CDF_UNDERFLOW_FLOOR
                + UNIT_ROUNDOFF * value.abs())
            .next_up(),
        }
    }
}

/// A bilinear score scale that is exact by declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scale {
    One,
    /// `fl(1/fl(√n))`, the rounded scale the network is declared to use.
    InverseSqrt(u32),
}

impl Scale {
    pub fn value(self) -> f64 {
        match self {
            Self::One => 1.0,
            Self::InverseSqrt(n) => 1.0 / f64::from(n).sqrt(),
        }
    }
}

/// Where an operator's reals come from: the native tensors and the exact rewrites applied.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Provenance {
    pub sources: Vec<String>,
    pub derivation: Vec<String>,
}

impl Provenance {
    pub fn native(tensor: &str) -> Self {
        Self { sources: vec![tensor.to_string()], derivation: Vec::new() }
    }

    /// The provenance of an operator derived from `parts` by `step`.
    pub fn derived(parts: &[&Provenance], step: String) -> Self {
        let mut sources: Vec<String> = parts.iter().flat_map(|p| p.sources.iter().cloned()).collect();
        sources.sort();
        sources.dedup();
        let mut derivation: Vec<String> = parts.iter().flat_map(|p| p.derivation.iter().cloned()).collect();
        derivation.push(step);
        Self { sources, derivation }
    }
}

/// The reals of an operator.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatorBody {
    /// The identity between equal interfaces; no reals.
    Identity,
    /// A block matrix: `values` (rows × cols) on the lattice of `precision`, zero off the present
    /// blocks; `present` is (row groups × column groups).
    Dense { values: DenseValues, present: Array2<bool>, precision: DeclaredPrecision },
    /// The product `left · right` of a `rows × r` and an `r × cols` factor, both on the lattice of
    /// `precision`. The executed operator is the exact product; its computed product is within
    /// `γ_r |left||right|` of it entrywise.
    LowRank { left: Array2<f64>, right: Array2<f64>, precision: DeclaredPrecision },
    /// `diag(values)` between equal interfaces, `values` on the lattice of `precision`: a norm
    /// gain. A product with it is a column scale.
    Diagonal { values: Array1<f64>, precision: DeclaredPrecision },
}

/// A dense operator's reals: in float64 on the host, or where a safetensors file stores them
/// ([`Stored`](crate::safetensors::Stored)). Stored reals are widened to float64 on the host only
/// when read there: [`Operator::matrix`] widens a copy, a read through `Deref` widens them once and
/// keeps them; a device takes them in its own storage ([`DenseValues::stored`]) without either.
#[derive(Clone)]
pub struct DenseValues {
    stored: Option<crate::safetensors::Stored>,
    host: std::sync::OnceLock<Array2<f64>>,
}

impl DenseValues {
    /// Reals where a file stores them.
    #[must_use]
    pub fn stored_in(stored: crate::safetensors::Stored) -> Self {
        Self { stored: Some(stored), host: std::sync::OnceLock::new() }
    }

    /// Where a file stores them, when it does and the host has not edited them.
    #[must_use]
    pub fn stored(&self) -> Option<&crate::safetensors::Stored> {
        self.stored.as_ref()
    }

    /// Their shape, rows × cols, without reading them.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        match (&self.stored, self.host.get()) {
            (_, Some(host)) => host.dim(),
            (Some(stored), None) => stored.dim(),
            (None, None) => (0, 0),
        }
    }

    /// The reals in float64 without keeping them: borrowed when the host holds them, widened from
    /// their file otherwise.
    #[must_use]
    pub fn matrix(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match (&self.stored, self.host.get()) {
            (_, Some(host)) => std::borrow::Cow::Borrowed(host),
            (Some(stored), None) => std::borrow::Cow::Owned(stored.matrix()),
            (None, None) => std::borrow::Cow::Owned(Array2::zeros((0, 0))),
        }
    }
}

impl From<Array2<f64>> for DenseValues {
    fn from(values: Array2<f64>) -> Self {
        Self { stored: None, host: std::sync::OnceLock::from(values) }
    }
}

impl std::ops::Deref for DenseValues {
    type Target = Array2<f64>;
    /// The reals in float64, widened from their file once and kept.
    fn deref(&self) -> &Array2<f64> {
        self.host.get_or_init(|| self.stored.as_ref().map_or_else(|| Array2::zeros((0, 0)), crate::safetensors::Stored::matrix))
    }
}

impl std::ops::DerefMut for DenseValues {
    /// The reals on the host to edit: an edit leaves the file's values behind.
    fn deref_mut(&mut self) -> &mut Array2<f64> {
        let held = std::mem::take(&mut self.host).into_inner();
        let values = held.unwrap_or_else(|| self.stored.as_ref().map_or_else(|| Array2::zeros((0, 0)), crate::safetensors::Stored::matrix));
        self.stored = None;
        self.host = std::sync::OnceLock::from(values);
        self.host.get_mut().expect("the host values were just set")
    }
}

impl PartialEq for DenseValues {
    /// The same reals, wherever they are held.
    fn eq(&self, other: &Self) -> bool {
        match (&self.stored, &other.stored) {
            (Some(a), Some(b)) if self.host.get().is_none() && other.host.get().is_none() => a == b,
            _ => *self.matrix() == *other.matrix(),
        }
    }
}

impl fmt::Debug for DenseValues {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.stored, self.host.get()) {
            (_, Some(host)) => write!(f, "{host:?}"),
            (Some(stored), None) => write!(f, "{stored:?}"),
            (None, None) => write!(f, "[]"),
        }
    }
}

/// Identity, dense, low-rank, dense with ordered rows, and diagonal (module note, "The code").
const OPERATOR_KINDS: usize = 5;

/// A shared operator between two interfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct Operator {
    pub name: String,
    pub rows: Interface,
    pub cols: Interface,
    pub body: OperatorBody,
    pub provenance: Provenance,
}

impl Operator {
    /// A dense operator with every block present, its reals rounded to `precision`'s lattice.
    pub fn dense(
        name: impl Into<String>,
        rows: Interface,
        cols: Interface,
        values: Array2<f64>,
        precision: DeclaredPrecision,
        provenance: Provenance,
    ) -> Result<Self, ProgramError> {
        let present = Array2::from_elem((rows.group_count(), cols.group_count()), true);
        Self::blocks(name, rows, cols, values, present, precision, provenance)
    }

    /// A dense operator with the declared present blocks; absent blocks are zeroed and present
    /// reals rounded to `precision`'s lattice, coarsened when the present reals' range needs it
    /// ([`DeclaredPrecision::within_range`]).
    pub fn blocks(
        name: impl Into<String>,
        rows: Interface,
        cols: Interface,
        mut values: Array2<f64>,
        present: Array2<bool>,
        precision: DeclaredPrecision,
        provenance: Provenance,
    ) -> Result<Self, ProgramError> {
        let name = name.into();
        if values.dim() != (rows.width(), cols.width()) || present.dim() != (rows.group_count(), cols.group_count()) {
            return Err(ProgramError::Shape(format!(
                "operator {name}: values {:?} and blocks {:?} against interfaces {}×{} with {}×{} groups",
                values.dim(),
                present.dim(),
                rows.width(),
                cols.width(),
                rows.group_count(),
                cols.group_count()
            )));
        }
        let largest = present
            .indexed_iter()
            .filter(|(_, keep)| **keep)
            .map(|((r, c), _)| values.slice(s![rows.range(r), cols.range(c)]).iter().fold(0.0_f64, |acc, v| acc.max(v.abs())))
            .fold(0.0_f64, f64::max);
        let precision = precision.within_range(largest);
        for ((r, c), &keep) in present.indexed_iter() {
            let mut block = values.slice_mut(s![rows.range(r), cols.range(c)]);
            if keep {
                for value in block.iter_mut() {
                    *value = round_to_lattice(*value, precision)?;
                }
            } else {
                block.fill(0.0);
            }
        }
        Ok(Self { name, rows, cols, body: OperatorBody::Dense { values: values.into(), present, precision }, provenance })
    }

    /// A dense operator with every block present whose reals stay where `stored` holds them, on
    /// the lattice of their exact precision (nothing to round).
    pub fn stored(name: impl Into<String>, rows: Interface, cols: Interface, stored: crate::safetensors::Stored, provenance: Provenance) -> Result<Self, ProgramError> {
        let name = name.into();
        if stored.dim() != (rows.width(), cols.width()) {
            return Err(ProgramError::Shape(format!("operator {name}: stored {:?} against interfaces {}×{}", stored.dim(), rows.width(), cols.width())));
        }
        let exact = exact_precision(stored.values())?;
        let precision = exact.within_range(stored.values().fold(0.0_f64, |acc, v| acc.max(v.abs())));
        if precision != exact {
            // Reals whose range coarsens their lattice are rounded on the host, as `blocks` does.
            return Self::dense(name, rows, cols, stored.matrix(), exact, provenance);
        }
        let present = Array2::from_elem((rows.group_count(), cols.group_count()), true);
        Ok(Self { name, rows, cols, body: OperatorBody::Dense { values: DenseValues::stored_in(stored), present, precision }, provenance })
    }

    /// The diagonal operator `diag(values)` on `interface`, its reals rounded to `precision`'s
    /// lattice, coarsened when their range needs it ([`DeclaredPrecision::within_range`]).
    pub fn diag(
        name: impl Into<String>,
        interface: Interface,
        values: Array1<f64>,
        precision: DeclaredPrecision,
        provenance: Provenance,
    ) -> Result<Self, ProgramError> {
        let name = name.into();
        if values.len() != interface.width() {
            return Err(ProgramError::Shape(format!("operator {name}: a diagonal of {} on an interface of {}", values.len(), interface.width())));
        }
        let precision = precision.within_range(values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())));
        let mut values = values;
        for value in values.iter_mut() {
            *value = round_to_lattice(*value, precision)?;
        }
        Ok(Self { name, rows: interface.clone(), cols: interface, body: OperatorBody::Diagonal { values, precision }, provenance })
    }

    /// A low-rank operator `left · right`, both factors rounded to `precision`'s lattice, coarsened
    /// when the factors' range needs it ([`DeclaredPrecision::within_range`]).
    pub fn low_rank(
        name: impl Into<String>,
        rows: Interface,
        cols: Interface,
        left: Array2<f64>,
        right: Array2<f64>,
        precision: DeclaredPrecision,
        provenance: Provenance,
    ) -> Result<Self, ProgramError> {
        let name = name.into();
        if left.nrows() != rows.width() || right.ncols() != cols.width() || left.ncols() != right.nrows() || left.ncols() == 0 {
            return Err(ProgramError::Shape(format!(
                "operator {name}: factors {:?} and {:?} against interfaces {}×{}",
                left.dim(),
                right.dim(),
                rows.width(),
                cols.width()
            )));
        }
        let largest = left.iter().chain(right.iter()).fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let precision = precision.within_range(largest);
        let round = |m: Array2<f64>| -> Result<Array2<f64>, ProgramError> {
            let mut m = m;
            for value in m.iter_mut() {
                *value = round_to_lattice(*value, precision)?;
            }
            Ok(m)
        };
        Ok(Self { name, rows, cols, body: OperatorBody::LowRank { left: round(left)?, right: round(right)?, precision }, provenance })
    }

    /// This operator's message length: (structure bits, real bits), the function the program's
    /// message uses.
    pub fn code_bits(&self) -> Result<(u64, u64), ProgramError> {
        operator_bits(self)
    }

    /// Structural wire fields only, without materializing or coding numerical values.
    /// This is exactly the first component of `code_bits`; validity of numerical
    /// payloads remains the encoder's obligation.
    pub fn structure_bits(&self) -> Result<u64, ProgramError> {
        let kind = u64::from(fixed_index_len_bits(OPERATOR_KINDS)?);
        let rows = interface_bits(&self.rows)?;
        let count = prefix_integer_len_bits(self.real_count() as u64 + 1)?;
        match &self.body {
            OperatorBody::Identity => Ok(kind + rows),
            OperatorBody::Diagonal { .. } => Ok(kind + rows + count),
            OperatorBody::LowRank { left, .. } => Ok(kind + rows + interface_bits(&self.cols)? + prefix_integer_len_bits(left.ncols() as u64)? + count),
            OperatorBody::Dense { present, .. } => {
                let mut total = kind + rows + interface_bits(&self.cols)? + count;
                for row in present.outer_iter() {
                    total += subset_code_len_bits(self.cols.group_count(), row.iter().filter(|keep| **keep).count())?;
                }
                Ok(total)
            }
        }
    }

    /// The identity on `interface`.
    pub fn identity(name: impl Into<String>, interface: Interface) -> Self {
        Self {
            name: name.into(),
            rows: interface.clone(),
            cols: interface,
            body: OperatorBody::Identity,
            provenance: Provenance::default(),
        }
    }

    /// [`Operator::matrix`] without copying a dense operator's reals: borrowed when dense, formed
    /// otherwise.
    pub fn matrix_cow(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match &self.body {
            OperatorBody::Dense { values, .. } => values.matrix(),
            _ => std::borrow::Cow::Owned(self.matrix()),
        }
    }

    /// The diagonal of an operator that is one: the identity, a diagonal operator, or a dense
    /// operator between equal interfaces of single-coordinate groups whose only present blocks are
    /// on the diagonal. A product with it is a column scale, not a matrix product.
    pub fn diagonal(&self) -> Option<Array1<f64>> {
        match &self.body {
            OperatorBody::Identity => Some(Array1::ones(self.rows.width())),
            OperatorBody::Diagonal { values, .. } => Some(values.clone()),
            OperatorBody::Dense { values, present, .. } => {
                let n = self.rows.width();
                let single = self.cols.width() == n && self.rows.group_count() == n && self.cols.group_count() == n;
                (single && present.indexed_iter().all(|((r, c), keep)| !*keep || r == c)).then(|| values.diag().to_owned())
            }
            OperatorBody::LowRank { .. } => None,
        }
    }

    /// The matrix (rows × cols) this operator applies.
    pub fn matrix(&self) -> Array2<f64> {
        match &self.body {
            OperatorBody::Identity => Array2::eye(self.rows.width()),
            OperatorBody::Dense { values, .. } => values.matrix().into_owned(),
            OperatorBody::LowRank { left, right, .. } => left.dot(right),
            OperatorBody::Diagonal { values, .. } => Array2::from_diag(values),
        }
    }

    /// `x Aᵀ`, the product an affine term applies, structural for the identity and a diagonal.
    pub fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        match &self.body {
            OperatorBody::Identity => x.clone(),
            OperatorBody::Diagonal { values, .. } => x * values,
            OperatorBody::Dense { values, .. } => fast_abt(x, values),
            OperatorBody::LowRank { left, right, .. } => fast_abt(&fast_abt(x, right), left),
        }
    }

    /// The number of reals the operator sends.
    pub fn real_count(&self) -> usize {
        match &self.body {
            OperatorBody::Identity => 0,
            OperatorBody::Diagonal { values, .. } => values.len(),
            OperatorBody::LowRank { left, right, .. } => left.len() + right.len(),
            OperatorBody::Dense { present, .. } => present
                .indexed_iter()
                .filter(|(_, keep)| **keep)
                .map(|((r, c), _)| self.rows.groups()[r].width * self.cols.groups()[c].width)
                .sum(),
        }
    }

    /// The present reals in message order: row groups, then column groups, row-major in a block.
    fn present_reals(&self) -> Vec<f64> {
        let mut reals = Vec::new();
        match &self.body {
            OperatorBody::Dense { values, present, .. } => {
                reals.reserve(self.real_count());
                if self.rows.groups().iter().all(|g| g.width == 1) && self.cols.groups().iter().all(|g| g.width == 1) {
                    // One-coordinate blocks: the message order is the row-major order of the present entries.
                    reals.extend(values.iter().zip(present.iter()).filter(|(_, keep)| **keep).map(|(v, _)| *v));
                } else {
                    for ((r, c), &keep) in present.indexed_iter() {
                        if keep {
                            for i in self.rows.range(r) {
                                for j in self.cols.range(c) {
                                    reals.push(values[[i, j]]);
                                }
                            }
                        }
                    }
                }
            }
            OperatorBody::LowRank { left, right, .. } => {
                reals.extend(left.iter().copied());
                reals.extend(right.iter().copied());
            }
            OperatorBody::Diagonal { values, .. } => reals.extend(values.iter().copied()),
            OperatorBody::Identity => {}
        }
        reals
    }
}

/// `round(x·2^p)·2^-p`, exact for an index within `2^53`.
pub fn round_to_lattice(value: f64, precision: DeclaredPrecision) -> Result<f64, ProgramError> {
    precision.round(value).map_err(ProgramError::Code)
}

/// A node of the program.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A basis evaluated at a token slot.
    Feature { slot: usize, basis: usize },
    /// A raw vector slot, one native group.
    Raw { slot: usize },
    /// An operator's single column (cols = the constant interface), the same on every input.
    Constant { operator: usize },
    /// `Σ_t x_t A_tᵀ + b`; every term's operator and the bias share one row interface.
    Affine { terms: Vec<(usize, usize)>, bias: Option<usize> },
    /// `c Σ_i l_i r_i` per row, one coordinate.
    Bilinear { left: usize, right: usize, scale: Scale },
    /// The softmax of one-coordinate scores, one column per score.
    Softmax { scores: Vec<usize> },
    /// `Σ_j α_{c_j} p_j` over payloads `(c_j, p_j)` of one interface, each read with weight column
    /// `c_j` of the weights node; the columns ascend strictly, and a column with no payload reads a
    /// zero payload.
    Mix { weights: usize, payloads: Vec<(usize, usize)> },
    /// A law per group of the input interface.
    Pointwise { input: usize, laws: Vec<Law> },
    /// The elementwise product of two nodes of one interface.
    Hadamard { left: usize, right: usize },
    /// Logits `y Φᵀ` over the classes of a basis's domain.
    Readout { input: usize, basis: usize },
    /// The row-wise outer product of two nodes: one group per pair of groups, each group's
    /// coordinates `l_i r_j` in row-major order.
    Outer { left: usize, right: usize },
    /// The columns of the parts side by side; the interface is the parts' groups in order.
    Concat { parts: Vec<usize> },
    /// Inside a rule body: the call's argument `index`.
    Param { index: usize },
    /// An application of rule `rule` to `arguments`, one per rule input; the value is the rule
    /// body's output on them. A body is stored once however often it is applied.
    Call { rule: usize, arguments: Vec<usize> },
    /// The input times a scalar polynomial in the declared parameters.
    Gain { input: usize, coefficient: Coefficient },
    /// Attention over the rows of a sequence: each row's query against the keys of the rows of its
    /// sequence (those at or before its position when `causal`), rotated by position when `rotary`,
    /// scored `c q·k`, softmax-weighted, reading the values.
    Attend { query: usize, key: usize, value: usize, scale: Scale, rotary: Option<Rotary>, causal: bool },
    /// `x / √(mean(x²) + ε)` per row, `ε` an exact real of the architecture.
    RmsNorm { input: usize, epsilon: f64 },
    /// `x A` for an operator `A` whose columns are this node's interface: an operator read in its
    /// transposed orientation (a tied unembedding), paid once.
    Transposed { input: usize, operator: usize },
}

const NODE_KINDS: usize = 18;

/// A rotary position embedding: plane `i` of a query or key at position `m` turned by
/// `m base^{-2i/dims}`. `half_split` pairs coordinate `i` with `i + dims/2` (rotate-half); otherwise
/// `2i` with `2i + 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotary {
    pub base: u32,
    pub dims: u32,
    pub half_split: bool,
}

impl Rotary {
    pub(crate) fn pairs(&self) -> Vec<(usize, usize)> {
        let half = self.dims as usize / 2;
        (0..half).map(|i| if self.half_split { (i, i + half) } else { (2 * i, 2 * i + 1) }).collect()
    }

    /// `(cos, sin)` of plane `i` at position `m`, with the angle as the architecture computes it.
    pub(crate) fn turn(&self, plane: usize, position: u32) -> (f64, f64) {
        let frequency = f64::from(self.base).powf(-2.0 * plane as f64 / f64::from(self.dims));
        let (sine, cosine) = (f64::from(position) * frequency).sin_cos();
        (cosine, sine)
    }

    /// `v` rotated to `position` in place, with the rotation's radius added to `radius`.
    pub(crate) fn rotate(&self, v: &mut [f64], radius: Option<&mut [f64]>, position: u32) {
        let mut radius = radius;
        for (plane, (a, b)) in self.pairs().into_iter().enumerate() {
            let (c, s) = self.turn(plane, position);
            let (x, y) = (v[a], v[b]);
            v[a] = c * x - s * y;
            v[b] = s * x + c * y;
            if let Some(r) = radius.as_deref_mut() {
                let (ra, rb) = (r[a], r[b]);
                let libm = 2.0 * UNIT_ROUNDOFF * (x.abs() + y.abs());
                r[a] = (c.abs() * ra + s.abs() * rb + libm + 2.0 * UNIT_ROUNDOFF * (c * x).abs().max((s * y).abs())).next_up();
                r[b] = (s.abs() * ra + c.abs() * rb + libm + 2.0 * UNIT_ROUNDOFF * (s * x).abs().max((c * y).abs())).next_up();
            }
        }
    }
}

/// A scalar polynomial in the declared parameters, with exact dyadic numbers.
#[derive(Clone, Debug, PartialEq)]
pub enum Coefficient {
    Parameter(usize),
    Number(f64),
    Sum(Vec<Coefficient>),
    Product(Vec<Coefficient>),
}

const COEFFICIENT_KINDS: usize = 4;

impl Coefficient {
    /// `(value, magnitude, operations)`: the computed value, the value of the same polynomial on
    /// the absolute values, and the rounded operations on its longest path; the computed value is
    /// within `γ_operations · magnitude` of the exact one.
    pub(crate) fn evaluate(&self, parameters: &[f64]) -> Result<(f64, f64, usize), ProgramError> {
        match self {
            Self::Parameter(index) => {
                let v = *parameters
                    .get(*index)
                    .ok_or(ProgramError::Reference { what: "parameter", index: *index })?;
                Ok((v, v.abs(), 0))
            }
            Self::Number(v) => Ok((*v, v.abs(), 0)),
            Self::Sum(terms) | Self::Product(terms) => {
                let product = matches!(self, Self::Product(_));
                let (mut value, mut magnitude, mut depth) = if product { (1.0, 1.0, 0) } else { (0.0, 0.0, 0) };
                for term in terms {
                    let (v, m, d) = term.evaluate(parameters)?;
                    if product {
                        value *= v;
                        magnitude *= m;
                    } else {
                        value += v;
                        magnitude += m;
                    }
                    depth = depth.max(d);
                }
                Ok((value, magnitude, depth + terms.len()))
            }
        }
    }
}

/// A named function of typed arguments: a body of nodes that reads its arguments through
/// [`Node::Param`], and the program's shared operators. A body may call only rules listed before
/// its own.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub name: String,
    pub inputs: Vec<Interface>,
    pub nodes: Vec<Node>,
    pub output: usize,
}

impl Node {
    fn kind_index(&self) -> usize {
        match self {
            Self::Feature { .. } => 0,
            Self::Raw { .. } => 1,
            Self::Constant { .. } => 2,
            Self::Affine { .. } => 3,
            Self::Bilinear { .. } => 4,
            Self::Softmax { .. } => 5,
            Self::Mix { .. } => 6,
            Self::Pointwise { .. } => 7,
            Self::Hadamard { .. } => 8,
            Self::Readout { .. } => 9,
            Self::Outer { .. } => 10,
            Self::Concat { .. } => 11,
            Self::Param { .. } => 12,
            Self::Call { .. } => 13,
            Self::Gain { .. } => 14,
            Self::Attend { .. } => 15,
            Self::RmsNorm { .. } => 16,
            Self::Transposed { .. } => 17,
        }
    }

    /// The nodes this node reads.
    pub fn arguments(&self) -> Vec<usize> {
        match self {
            Self::Feature { .. } | Self::Raw { .. } | Self::Constant { .. } | Self::Param { .. } => Vec::new(),
            Self::Call { arguments, .. } => arguments.clone(),
            Self::Gain { input, .. } | Self::RmsNorm { input, .. } | Self::Transposed { input, .. } => vec![*input],
            Self::Attend { query, key, value, .. } => vec![*query, *key, *value],
            Self::Affine { terms, .. } => terms.iter().map(|(node, _)| *node).collect(),
            Self::Bilinear { left, right, .. } | Self::Hadamard { left, right } | Self::Outer { left, right } => {
                vec![*left, *right]
            }
            Self::Softmax { scores } => scores.clone(),
            Self::Concat { parts } => parts.clone(),
            Self::Mix { weights, payloads } => {
                std::iter::once(*weights).chain(payloads.iter().map(|(_, node)| *node)).collect()
            }
            Self::Pointwise { input, .. } | Self::Readout { input, .. } => vec![*input],
        }
    }

    /// The operators this node reads.
    pub fn operators(&self) -> Vec<usize> {
        match self {
            Self::Constant { operator } | Self::Transposed { operator, .. } => vec![*operator],
            Self::Affine { terms, bias } => terms.iter().map(|(_, op)| *op).chain(bias.iter().copied()).collect(),
            _ => Vec::new(),
        }
    }
}

/// A refused program, input or message.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramError {
    Interface(String),
    Shape(String),
    Reference { what: &'static str, index: usize },
    ForwardReference { node: usize, argument: usize },
    Input(String),
    Code(String),
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interface(message) => write!(f, "operator program interface: {message}"),
            Self::Shape(message) => write!(f, "operator program shape: {message}"),
            Self::Reference { what, index } => write!(f, "operator program: unknown {what} {index}"),
            Self::ForwardReference { node, argument } => {
                write!(f, "operator program: node {node} reads node {argument}, which is not listed before it")
            }
            Self::Input(message) => write!(f, "operator program input: {message}"),
            Self::Code(message) => write!(f, "operator program code: {message}"),
        }
    }
}

impl std::error::Error for ProgramError {}

impl From<CodecError> for ProgramError {
    fn from(error: CodecError) -> Self {
        Self::Code(format!("{error:?}"))
    }
}

/// The program: bases, shared operators, nodes in topological order, and the output node.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorProgram {
    pub declarations: Declarations,
    pub bases: Vec<Basis>,
    /// Shared: a program's clone shares every operator it does not change (`Arc::make_mut` copies
    /// one on its first change), so a candidate costs only what it edits.
    pub operators: Vec<Arc<Operator>>,
    pub rules: Vec<Rule>,
    pub nodes: Vec<Node>,
    pub output: usize,
}

/// The inputs of a finite family: per slot, one token per input or one raw row per input.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotValues {
    Tokens(Vec<u32>),
    Raw(Array2<f64>),
}

/// A finite input family: `rows` inputs, each slot's values in input order.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyInputs {
    pub rows: usize,
    pub slots: Vec<SlotValues>,
    /// For a per-position family: each row's sequence and position. Attention reads the rows of the
    /// same sequence.
    pub layout: Option<SequenceLayout>,
}

/// Each row's sequence and position within it.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceLayout {
    pub sequence: Vec<u32>,
    pub position: Vec<u32>,
}

impl FamilyInputs {
    /// This family followed by `other`'s rows (the same slots, and a layout in both or neither).
    pub fn append(&self, other: &FamilyInputs) -> Result<FamilyInputs, ProgramError> {
        if self.slots.len() != other.slots.len() || self.layout.is_some() != other.layout.is_some() {
            return Err(ProgramError::Input("appended families differ in slots or layout".to_string()));
        }
        let slots = self
            .slots
            .iter()
            .zip(&other.slots)
            .map(|pair| match pair {
                (SlotValues::Tokens(a), SlotValues::Tokens(b)) => Ok(SlotValues::Tokens(a.iter().chain(b).copied().collect())),
                (SlotValues::Raw(a), SlotValues::Raw(b)) => ndarray::concatenate(Axis(0), &[a.view(), b.view()])
                    .map(SlotValues::Raw)
                    .map_err(|e| ProgramError::Input(e.to_string())),
                _ => Err(ProgramError::Input("appended slots differ in kind".to_string())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = match (&self.layout, &other.layout) {
            (Some(a), Some(b)) => {
                let offset = a.sequence.iter().copied().max().map_or(0, |m| m + 1);
                Some(SequenceLayout {
                    sequence: a.sequence.iter().copied().chain(b.sequence.iter().map(|s| s + offset)).collect(),
                    position: a.position.iter().chain(&b.position).copied().collect(),
                })
            }
            _ => None,
        };
        Ok(FamilyInputs { rows: self.rows + other.rows, slots, layout })
    }

    /// The inputs at `indices`, in that order.
    pub fn select(&self, indices: &[usize]) -> Self {
        Self {
            rows: indices.len(),
            slots: self
                .slots
                .iter()
                .map(|slot| match slot {
                    SlotValues::Tokens(tokens) => SlotValues::Tokens(indices.iter().map(|&i| tokens[i]).collect()),
                    SlotValues::Raw(rows) => SlotValues::Raw(rows.select(Axis(0), indices)),
                })
                .collect(),
            layout: self.layout.as_ref().map(|layout| SequenceLayout {
                sequence: indices.iter().map(|&i| layout.sequence[i]).collect(),
                position: indices.iter().map(|&i| layout.position[i]).collect(),
            }),
        }
    }
}

/// Whether `x` is at most one eighth nonzero: then [`sparse_abt`] reads fewer entries of `A` than
/// a dense product multiplies.
pub(crate) fn sparse_enough(x: &Array2<f64>) -> bool {
    let nonzero = x.iter().filter(|v| **v != 0.0).count();
    nonzero > 0 && 8 * nonzero <= x.len()
}

/// `x Aᵀ` over the nonzeros of `x` alone (a masked site's gated values, a few percent nonzero per
/// row): each output column is `A`'s row against every row's nonzeros, so `A` is read once.
/// Exact zeros add nothing; the sums run over the same products in another order.
pub(crate) fn sparse_abt(x: &Array2<f64>, a: &Array2<f64>) -> Array2<f64> {
    use rayon::prelude::*;
    let rows = x.nrows();
    let nonzeros: Vec<Vec<(usize, f64)>> = x
        .outer_iter()
        .map(|row| row.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(j, v)| (j, *v)).collect())
        .collect();
    let mut transposed = Array2::<f64>::zeros((a.nrows(), rows));
    transposed.axis_iter_mut(Axis(0)).into_par_iter().zip(a.axis_iter(Axis(0))).for_each(|(mut column, weights)| {
        for (r, entries) in nonzeros.iter().enumerate() {
            column[r] = entries.iter().map(|(j, v)| weights[*j] * v).sum();
        }
    });
    transposed.reversed_axes().as_standard_layout().into_owned()
}

/// Every node's value, and its error enclosure when bands were requested.
///
/// A node's computed value `x̂` encloses the exact value as `x̂ + e_box + e_ball` per row, with
/// `|e_box| ≤ bands` entrywise and `‖e_ball‖₂ ≤ balls` (one radius per row). The local rounding of
/// each node stays entrywise (a box); an error that an operator carries forward from its input is
/// propagated in `ℓ₂` through a proven spectral-norm bound of the operator (a ball), so it grows by
/// the operator's norm, not by its row `ℓ₁` norms layer after layer.
#[derive(Clone, Debug)]
pub struct Trace {
    pub values: Vec<Array2<f64>>,
    pub bands: Option<Vec<Array2<f64>>>,
    pub balls: Option<Vec<Array1<f64>>>,
}

/// Per-row ball radii split at `from`, as [`Layered`] splits values.
struct Balls<'a> {
    base: &'a [Array1<f64>],
    top: &'a [Array1<f64>],
    from: usize,
}

impl<'a> Balls<'a> {
    fn get(&self, node: usize) -> &'a Array1<f64> {
        if node < self.from { &self.base[node] } else { &self.top[node - self.from] }
    }
}

/// Per row, an upward bound on the `ℓ₂` norm of `r` (nonnegative entries).
fn row_norms(r: &Array2<f64>) -> Array1<f64> {
    r.outer_iter()
        .map(|row| inflate(row.iter().map(|v| v * v).sum::<f64>(), row.len()).sqrt().next_up())
        .collect()
}

/// A proven upper bound on `‖A‖₂` for an operator: `1` for the identity, the product of its
/// factors' bounds for a low-rank operator, and for a dense one [`matrix_spectral_bound`].
fn spectral_bound(op: &Operator) -> Result<f64, ProgramError> {
    match &op.body {
        OperatorBody::Identity => Ok(1.0),
        OperatorBody::Diagonal { values, .. } => Ok(values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))),
        OperatorBody::Dense { values, .. } => matrix_spectral_bound(values.view()),
        OperatorBody::LowRank { left, right, .. } => {
            Ok((matrix_spectral_bound(left.view())? * matrix_spectral_bound(right.view())?).next_up())
        }
    }
}

/// A content fingerprint of a matrix: its shape and two independent 64-bit hashes of its bits.
fn fingerprint(a: ArrayView2<'_, f64>) -> (usize, usize, u64, u64) {
    let (mut h1, mut h2) = (0xcbf2_9ce4_8422_2325_u64, 0x9e37_79b9_7f4a_7c15_u64);
    for v in a.iter() {
        let bits = v.to_bits();
        h1 = (h1 ^ bits).wrapping_mul(0x0000_0100_0000_01b3);
        h2 = (h2.rotate_left(23) ^ bits).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    (a.nrows(), a.ncols(), h1, h2)
}

/// A proven upper bound on `‖A‖₂`, the least of: the largest computed singular value plus the
/// decomposition's band (every singular value is within the band of an exact one,
/// `gam_linalg::decompose::svd`), the Frobenius norm, and for a matrix with at most one nonzero per row and per
/// column (a diagonal gain, a permutation) its largest entry, which is then exact. Bounds are kept
/// per matrix content, so an operator met again is not decomposed again.
pub(crate) fn matrix_spectral_bound(a: ArrayView2<'_, f64>) -> Result<f64, ProgramError> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(usize, usize, u64, u64), f64>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = fingerprint(a);
    if let Some(bound) = cache.lock().map_err(|_| ProgramError::Shape("a poisoned norm cache".to_string()))?.get(&key) {
        return Ok(*bound);
    }
    let largest = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let sparse = a.rows().into_iter().all(|row| row.iter().filter(|v| **v != 0.0).count() <= 1)
        && a.columns().into_iter().all(|column| column.iter().filter(|v| **v != 0.0).count() <= 1);
    let frobenius = inflate(a.iter().map(|v| v * v).sum::<f64>(), a.len()).sqrt().next_up();
    let bound = if largest == 0.0 {
        0.0
    } else if sparse {
        largest
    } else {
        let decomposition = gam_linalg::decompose::svd(a, false).map_err(|error| ProgramError::Shape(format!("spectral bound: {error:?}")))?;
        let top = decomposition.singular_values.iter().fold(0.0_f64, |m, v| m.max(*v));
        (top + decomposition.band).next_up().min(frobenius)
    };
    cache.lock().map_err(|_| ProgramError::Shape("a poisoned norm cache".to_string()))?.insert(key, bound);
    Ok(bound)
}

/// Node values split at `from`: earlier nodes from `base`, later ones from `top`.
fn value_of<'a>(values: &Layered<'a>, node: usize) -> &'a Array2<f64> {
    values.get(node)
}

/// The per-row scale `(mean(x²) + ε)^{-1/2}` an [`Node::RmsNorm`] multiplies its row by, as it computes it.
pub fn rms_scale(row: ndarray::ArrayView1<'_, f64>, epsilon: f64) -> f64 {
    let mean = row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64;
    1.0 / (mean + epsilon).sqrt()
}

/// `x / √(mean(x²) + ε)` per row with its radius: the input radius `r` moves the mean of squares by
/// at most `δ = (2/n) Σ|x_j| r_j + (1/n) Σ r_j²` and the scale `s = (m + ε)^{-1/2}` by at most
/// `s³ δ/2 ·(1 − δ s²)^{-3/2}` (refused to `+∞` when `δ s² ≥ 1/2`); the computation rounds the mean
/// (`γ_{n+1}`), the addition, the root and the division (`4u` on `s`, relative) and the product.
fn rms_norm(x: &Array2<f64>, bands: Option<&Array2<f64>>, epsilon: f64) -> (Array2<f64>, Option<Array2<f64>>) {
    let (rows, width) = x.dim();
    let mut out = Array2::<f64>::zeros((rows, width));
    let mut radius = bands.map(|_| Array2::<f64>::zeros((rows, width)));
    for row in 0..rows {
        let xr = x.row(row);
        let mean = xr.iter().map(|v| v * v).sum::<f64>() / width as f64;
        let scale = rms_scale(xr, epsilon);
        for c in 0..width {
            out[[row, c]] = xr[c] * scale;
        }
        if let (Some(radius), Some(bands)) = (radius.as_mut(), bands) {
            let rr = bands.row(row);
            let delta = (2.0 * xr.iter().zip(rr.iter()).map(|(v, r)| v.abs() * r).sum::<f64>()
                + rr.iter().map(|r| r * r).sum::<f64>())
                / width as f64;
            let rounding = accumulation_growth(width + 1) * mean + UNIT_ROUNDOFF * (mean + epsilon);
            let d = delta + rounding;
            let shrink = d * scale * scale;
            let scale_error = if shrink < 0.5 {
                scale.powi(3) * d / 2.0 * (1.0 - shrink).powf(-1.5) + 4.0 * UNIT_ROUNDOFF * scale
            } else {
                f64::INFINITY
            };
            for c in 0..width {
                radius[[row, c]] = inflate(
                    rr[c] * (scale + scale_error) + xr[c].abs() * scale_error + UNIT_ROUNDOFF * (xr[c] * scale).abs(),
                    4,
                );
            }
        }
    }
    (out, radius)
}

/// The arguments of the rule body being executed (none at the top level) and the parameter values.
struct Frame<'a> {
    args: &'a [(Array2<f64>, Option<Array2<f64>>)],
    parameters: &'a [f64],
    /// The node list being executed (the program's, or a rule body's) and its output node.
    nodes: &'a [Node],
    output: usize,
}

/// Whether node `index` of `nodes` (output `output`) is a gathered feature: an indicator basis's
/// one-hot rows, not the output, read at least once and only as affine terms (module note,
/// "Execution with forward-error bands").
fn gathered(nodes: &[Node], output: usize, bases: &[Basis], index: usize) -> bool {
    let Some(Node::Feature { basis, .. }) = nodes.get(index) else { return false };
    if index == output || !matches!(bases.get(*basis), Some(Basis::Indicator { .. })) {
        return false;
    }
    let mut read = false;
    for node in &nodes[index + 1..] {
        if node.arguments().contains(&index) {
            if !matches!(node, Node::Affine { .. }) {
                return false;
            }
            read = true;
        }
    }
    read
}

/// Add to each row of `out` column `tokens[row]` of `op` (an affine term on a gathered feature),
/// and to `radius` its rounding: `growth` times the column's magnitude, plus for a low-rank
/// operator the column's own product error `γ_r |left||right_t|`.
fn add_gathered(op: &Operator, tokens: &[u32], out: Option<&mut Array2<f64>>, radius: Option<&mut Array2<f64>>, growth: f64) {
    let (mut out, mut radius) = (out, radius);
    let diagonal = match &op.body {
        OperatorBody::Identity | OperatorBody::Diagonal { .. } => op.diagonal(),
        _ => None,
    };
    for (row, &token) in tokens.iter().enumerate() {
        let t = token as usize;
        match &op.body {
            OperatorBody::Dense { values, .. } => {
                if let Some(out) = out.as_deref_mut() {
                    out.row_mut(row).scaled_add(1.0, &values.column(t));
                }
                if let Some(radius) = radius.as_deref_mut() {
                    radius.row_mut(row).zip_mut_with(&values.column(t), |acc, v| *acc += growth * v.abs());
                }
            }
            OperatorBody::Identity | OperatorBody::Diagonal { .. } => {
                let v = diagonal.as_ref().map_or(1.0, |d| d[t]);
                if let Some(out) = out.as_deref_mut() {
                    out[[row, t]] += v;
                }
                if let Some(radius) = radius.as_deref_mut() {
                    radius[[row, t]] += growth * v.abs();
                }
            }
            OperatorBody::LowRank { left, right, .. } => {
                let column = left.dot(&right.column(t));
                if let Some(out) = out.as_deref_mut() {
                    out.row_mut(row).scaled_add(1.0, &column);
                }
                if let Some(radius) = radius.as_deref_mut() {
                    let product = left.mapv(f64::abs).dot(&right.column(t).mapv(f64::abs)) * accumulation_growth(left.ncols());
                    radius.row_mut(row).zip_mut_with(&column, |acc, v| *acc += growth * v.abs());
                    radius.row_mut(row).zip_mut_with(&product, |acc, p| *acc += p);
                }
            }
        }
    }
}

/// With `patch`, a node found there takes that value instead.
struct Layered<'a> {
    base: &'a [Array2<f64>],
    top: &'a [Array2<f64>],
    from: usize,
    patch: Option<&'a BTreeMap<usize, Array2<f64>>>,
}

impl<'a> Layered<'a> {
    fn get(&self, node: usize) -> &'a Array2<f64> {
        if let Some(value) = self.patch.and_then(|patch| patch.get(&node)) {
            return value;
        }
        if node < self.from { &self.base[node] } else { &self.top[node - self.from] }
    }

    /// Whether `node` takes a patched value.
    fn patched(&self, node: usize) -> bool {
        self.patch.is_some_and(|patch| patch.contains_key(&node))
    }
}

/// The column of the single `1` in each row of `x`, when every row is an exact indicator.
fn one_hot_columns(x: &Array2<f64>) -> Option<Vec<usize>> {
    let mut columns = Vec::with_capacity(x.nrows());
    for row in x.outer_iter() {
        let mut found = None;
        for (c, v) in row.iter().enumerate() {
            if *v == 1.0 && found.is_none() {
                found = Some(c);
            } else if *v != 0.0 {
                return None;
            }
        }
        columns.push(found?);
    }
    Some(columns)
}

/// `x / (1 − γ_{k+2})`, bounded by `x (1 + 2γ_{k+2})`: the outward inflation of a computed
/// nonnegative sum of `k` terms (module note).
fn inflate(value: f64, terms: usize) -> f64 {
    (value * (1.0 + 2.0 * accumulation_growth(terms + 2))).next_up()
}

/// The radius of a computed softmax weight `p` whose relative radius is `factor`: `p·factor`, but
/// never more than `max(p, 1 − p)`, since the exact weight lies in `[0, 1]`. The cap keeps the
/// radius finite when `factor` overflows (a score radius past ~355 makes `exp(2R) − 1` infinite,
/// and `0·∞` would be NaN).
fn softmax_radius(p: f64, factor: f64) -> f64 {
    let cap = p.max((1.0 - p).next_up());
    let bound = inflate(p * factor, 4);
    if bound >= 0.0 { bound.min(cap) } else { cap }
}

fn inflate_all(array: &mut Array2<f64>, terms: usize) {
    let factor = 1.0 + 2.0 * accumulation_growth(terms + 2);
    array.mapv_inplace(|value| (value * factor).next_up());
}

impl OperatorProgram {
    /// Validate references, order and interfaces, returning every node's interface.
    pub fn interfaces(&self) -> Result<Vec<Interface>, ProgramError> {
        for basis in &self.bases {
            basis.interface(&self.declarations)?;
        }
        for rule in 0..self.rules.len() {
            let body = rule_interfaces(&self.rules, rule, &self.operators, &self.bases, &self.declarations)?;
            if self.rules[rule].output >= body.len() {
                return Err(ProgramError::Reference { what: "rule output node", index: self.rules[rule].output });
            }
        }
        let mut out: Vec<Interface> = Vec::with_capacity(self.nodes.len());
        let scope = Scope { operators: &self.operators, bases: &self.bases, declarations: &self.declarations, rules: &self.rules, params: &[] };
        for (index, node) in self.nodes.iter().enumerate() {
            let interface = interface_of(index, node, &out, &scope)?;
            out.push(interface);
        }
        if self.output >= self.nodes.len() {
            return Err(ProgramError::Reference { what: "output node", index: self.output });
        }
        Ok(out)
    }

    /// The total number of reals the program sends.
    pub fn real_count(&self) -> usize {
        self.operators.iter().map(|op| op.real_count()).sum()
    }

    /// Execute on `inputs`, with forward-error bands when `bands`.
    pub fn execute(&self, inputs: &FamilyInputs, bands: bool) -> Result<Trace, ProgramError> {
        self.execute_at(inputs, bands, &vec![1.0; self.declarations.parameters])
    }

    fn execute_range(&self, inputs: &FamilyInputs, trace: &mut Trace, from: usize, parameters: &[f64]) -> Result<(), ProgramError> {
        self.check_inputs(inputs)?;
        trace.values.truncate(from);
        if let Some(bands) = trace.bands.as_mut() {
            bands.truncate(from);
        }
        if trace.bands.is_some() {
            let balls = trace.balls.get_or_insert_with(Vec::new);
            balls.truncate(from);
            balls.resize_with(from, || Array1::zeros(inputs.rows));
        }
        let interfaces = self.interfaces()?;
        let mut top: Vec<Array2<f64>> = Vec::new();
        let mut top_bands: Vec<Array2<f64>> = Vec::new();
        let mut top_balls: Vec<Array1<f64>> = Vec::new();
        for index in from..self.nodes.len() {
            let values = Layered { base: &trace.values, top: &top, from, patch: None };
            let frame = Frame { args: &[], parameters, nodes: &self.nodes, output: self.output };
            match (&trace.bands, &trace.balls) {
                (Some(bands), Some(balls)) => {
                    let bands = Layered { base: bands, top: &top_bands, from, patch: None };
                    let balls = Balls { base: balls, top: &top_balls, from };
                    let (value, band, ball) =
                        self.evaluate_enclosed(index, inputs, (&values, &bands, &balls), &interfaces, &frame)?;
                    top.push(value);
                    top_bands.push(band);
                    top_balls.push(ball);
                }
                _ => {
                    let (value, _) = self.evaluate_node(index, &self.nodes[index], inputs, &values, None, &interfaces, &frame)?;
                    top.push(value);
                }
            }
        }
        trace.values.extend(top);
        if let Some(bands) = trace.bands.as_mut() {
            bands.extend(top_bands);
        }
        if let Some(balls) = trace.balls.as_mut() {
            balls.extend(top_balls);
        }
        Ok(())
    }

    /// The unbanded trace of `inputs` with `edit` applied to each node's value as soon as it is
    /// computed, before any later node reads it: `edit(node, value, earlier)` sees the evaluated
    /// prefix (`earlier`, every node before `node`) and may change `value` in place (an
    /// intervention at a place, a state patched from another run, an explicit exception). The
    /// edited value must keep its shape.
    pub fn execute_edited<F>(&self, inputs: &FamilyInputs, mut edit: F) -> Result<Trace, ProgramError>
    where
        F: FnMut(usize, &mut Array2<f64>, &[Array2<f64>]) -> Result<(), String>,
    {
        self.check_inputs(inputs)?;
        let interfaces = self.interfaces()?;
        let ones = vec![1.0; self.declarations.parameters];
        let frame = Frame { args: &[], parameters: &ones, nodes: &self.nodes, output: self.output };
        let mut top: Vec<Array2<f64>> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let values = Layered { base: &[], top: &top, from: 0, patch: None };
            let mut value = self.evaluate_node(index, node, inputs, &values, None, &interfaces, &frame)?.0;
            let shape = value.dim();
            edit(index, &mut value, &top).map_err(ProgramError::Input)?;
            if value.dim() != shape {
                return Err(ProgramError::Input(format!("an edit of node {index} changed its shape {shape:?} to {:?}", value.dim())));
            }
            top.push(value);
        }
        Ok(Trace { values: top, bands: None, balls: None })
    }

    /// Causal (or full) attention over each row's sequence, with its radius: the score, softmax and
    /// read are the [`Node::Bilinear`], [`Node::Softmax`] and [`Node::Mix`] rules applied per row
    /// over its sequence's rows, after the rotation (orthogonal per plane, with libm's one ulp).
    ///
    /// With `balls` (per-row `ℓ₂` radii of the query, key and value errors besides their boxes), a
    /// score's error gains `c(‖q‖‖e_k‖ + ‖e_q‖‖k‖ + ‖e_q‖‖e_k‖)` over every mix of box and ball parts
    /// (a rotation keeps a ball's radius), and the read's ball is `Σ_j (α_j + Δα_j) ρ_{v,j}`: the
    /// value errors pass through the convex weights (and their perturbation) in `ℓ₂`.
    fn attend(
        &self,
        inputs: &FamilyInputs,
        (query, key, value): (&Array2<f64>, &Array2<f64>, &Array2<f64>),
        bands: Option<(&Array2<f64>, &Array2<f64>, &Array2<f64>)>,
        balls: Option<(&Array1<f64>, &Array1<f64>, &Array1<f64>)>,
        (scale, rotary, causal): (Scale, Option<Rotary>, bool),
    ) -> Result<(Array2<f64>, Option<Array2<f64>>, Option<Array1<f64>>), ProgramError> {
        if bands.is_none() && balls.is_none() {
            return Ok((super::tiled_attention::forward(inputs, (query, key, value), scale.value(), rotary, causal)?, None, None));
        }
        let layout = inputs
            .layout
            .as_ref()
            .ok_or_else(|| ProgramError::Input("an attend node needs a sequence layout".to_string()))?;
        let rows = inputs.rows;
        let mut by_sequence: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for row in 0..rows {
            by_sequence.entry(layout.sequence[row]).or_default().push(row);
        }
        let (mut q, mut k) = (query.clone(), key.clone());
        let (mut rq, mut rk) = match bands {
            Some((bq, bk, _)) => (Some(bq.clone()), Some(bk.clone())),
            None => (None, None),
        };
        if let Some(rotary) = rotary {
            for row in 0..rows {
                let position = layout.position[row];
                let mut qrow = q.row(row).to_vec();
                let mut krow = k.row(row).to_vec();
                let mut rqrow = rq.as_ref().map(|b| b.row(row).to_vec());
                let mut rkrow = rk.as_ref().map(|b| b.row(row).to_vec());
                rotary.rotate(&mut qrow, rqrow.as_deref_mut(), position);
                rotary.rotate(&mut krow, rkrow.as_deref_mut(), position);
                q.row_mut(row).assign(&ndarray::ArrayView1::from(&qrow));
                k.row_mut(row).assign(&ndarray::ArrayView1::from(&krow));
                if let (Some(b), Some(r)) = (rq.as_mut(), rqrow) {
                    b.row_mut(row).assign(&ndarray::ArrayView1::from(&r));
                }
                if let (Some(b), Some(r)) = (rk.as_mut(), rkrow) {
                    b.row_mut(row).assign(&ndarray::ArrayView1::from(&r));
                }
            }
        }
        let c = scale.value();
        let width = value.ncols();
        let dims = q.ncols();
        let mut out = Array2::<f64>::zeros((rows, width));
        let mut radius = bands.map(|_| Array2::<f64>::zeros((rows, width)));
        let mut ball = balls.map(|_| Array1::<f64>::zeros(rows));
        // Per row, upward `ℓ₂` norms of the rotated queries and keys and of their boxes.
        let norms = |m: &Array2<f64>| -> Array1<f64> { row_norms(&m.mapv(f64::abs)) };
        let (q_norm, k_norm) = (norms(&q), norms(&k));
        let (rq_norm, rk_norm) = (rq.as_ref().map(norms), rk.as_ref().map(norms));
        let u = UNIT_ROUNDOFF;
        for members in by_sequence.values() {
            for &row in members {
                let keys: Vec<usize> = members
                    .iter()
                    .copied()
                    .filter(|&other| !causal || layout.position[other] <= layout.position[row])
                    .collect();
                let scores: Vec<f64> = keys.iter().map(|&other| c * q.row(row).dot(&k.row(other))).collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let total: f64 = e.iter().sum();
                let alpha: Vec<f64> = e.iter().map(|v| v / total).collect();
                for (j, &other) in keys.iter().enumerate() {
                    out.row_mut(row).scaled_add(alpha[j], &value.row(other));
                }
                if let (Some(radius), Some((_, _, bv)), Some(rq), Some(rk)) = (radius.as_mut(), bands, rq.as_ref(), rk.as_ref()) {
                    let growth = accumulation_growth(dims + 1);
                    let score_radius: Vec<f64> = keys
                        .iter()
                        .map(|&other| {
                            let mut total = 0.0;
                            for i in 0..dims {
                                let (a, b) = (q[[row, i]].abs(), k[[other, i]].abs());
                                let (ra, rb) = (rq[[row, i]], rk[[other, i]]);
                                total += growth * a * b + a * rb + ra * b + ra * rb;
                            }
                            if let (Some((bq, bk, _)), Some(rqn), Some(rkn)) = (balls, rq_norm.as_ref(), rk_norm.as_ref()) {
                                let (pq, pk) = (bq[row], bk[other]);
                                total += q_norm[row] * pk + pq * k_norm[other] + pq * pk + rqn[row] * pk + pq * rkn[other];
                            }
                            inflate(c.abs() * total, 4 * dims)
                        })
                        .collect();
                    let spread = scores.iter().map(|v| (v - m).abs()).fold(0.0, f64::max);
                    let eta = ((2.0 * u * spread).exp()
                        * (1.0 + 2.0 * u).powi(2)
                        * (1.0 + accumulation_growth(keys.len().saturating_sub(1)))
                        * (1.0 + u)
                        - 1.0)
                        .next_up();
                    let big_r = score_radius.iter().copied().fold(0.0, f64::max);
                    let factor = ((eta + (2.0 * big_r).exp_m1()) / (1.0 - eta)).next_up();
                    let mix_growth = accumulation_growth(keys.len());
                    for (j, &other) in keys.iter().enumerate() {
                        let (av, ar) = (alpha[j].abs(), softmax_radius(alpha[j], factor));
                        for col in 0..width {
                            let (pv, pr) = (value[[other, col]].abs(), bv[[other, col]]);
                            radius[[row, col]] += mix_growth * av * pv + av * pr + ar * pv + ar * pr;
                        }
                        if let (Some(ball), Some((_, _, bv_ball))) = (ball.as_mut(), balls) {
                            ball[row] += (av + ar) * bv_ball[other];
                        }
                    }
                }
            }
        }
        if let Some(radius) = radius.as_mut() {
            inflate_all(radius, 4 * rows.max(1));
        }
        if let Some(ball) = ball.as_mut() {
            ball.mapv_inplace(|v| inflate(v, 4 * rows.max(1)));
        }
        Ok((out, radius, ball))
    }

    /// Rule `rule`'s output on `args`.
    fn execute_rule(
        &self,
        rule: usize,
        args: &[(Array2<f64>, Option<Array2<f64>>)],
        inputs: &FamilyInputs,
        banded: bool,
        parameters: &[f64],
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ProgramError> {
        let body = self.rules.get(rule).ok_or(ProgramError::Reference { what: "rule", index: rule })?;
        let interfaces = rule_interfaces(&self.rules, rule, &self.operators, &self.bases, &self.declarations)?;
        let frame = Frame { args, parameters, nodes: &body.nodes, output: body.output };
        let mut top: Vec<Array2<f64>> = Vec::with_capacity(body.nodes.len());
        let mut top_bands: Vec<Array2<f64>> = Vec::new();
        for (index, node) in body.nodes.iter().enumerate() {
            let values = Layered { base: &[], top: &top, from: 0, patch: None };
            let bands = banded.then(|| Layered { base: &[], top: &top_bands, from: 0, patch: None });
            let (value, band) = self.evaluate_node(index, node, inputs, &values, bands.as_ref(), &interfaces, &frame)?;
            top.push(value);
            if banded {
                top_bands.push(band.unwrap_or_else(|| Array2::zeros((0, 0))));
            }
        }
        let output = top.swap_remove(body.output);
        let band = banded.then(|| top_bands.swap_remove(body.output));
        Ok((output, band))
    }

    /// Every operator `node` reads, through the bodies of the rules it calls.
    pub fn node_operators(&self, node: &Node) -> Vec<usize> {
        let mut out = node.operators();
        if let Node::Call { rule, .. } = node
            && let Some(body) = self.rules.get(*rule)
        {
            for inner in &body.nodes {
                out.extend(self.node_operators(inner));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The token ids a gathered feature `node` of this program reads on `inputs` (module note,
    /// "Execution with forward-error bands"); `None` for any other node. A gathered feature's
    /// trace value holds no columns: its readers read these ids.
    pub fn gathered_tokens<'a>(&self, node: usize, inputs: &'a FamilyInputs) -> Option<&'a [u32]> {
        self.frame_tokens(&self.nodes, self.output, node, inputs)
    }

    fn frame_tokens<'a>(&self, nodes: &[Node], output: usize, node: usize, inputs: &'a FamilyInputs) -> Option<&'a [u32]> {
        if !gathered(nodes, output, &self.bases, node) {
            return None;
        }
        let Node::Feature { slot, .. } = &nodes[node] else { return None };
        match inputs.slots.get(*slot) {
            Some(SlotValues::Tokens(tokens)) => Some(tokens),
            _ => None,
        }
    }

    /// [`Self::execute`] at declared parameter values (`execute` is every parameter at `1`).
    pub fn execute_at(&self, inputs: &FamilyInputs, bands: bool, parameters: &[f64]) -> Result<Trace, ProgramError> {
        if parameters.len() != self.declarations.parameters {
            return Err(ProgramError::Input(format!(
                "{} parameter values for {} declared parameters",
                parameters.len(),
                self.declarations.parameters
            )));
        }
        self.interfaces()?;
        let mut trace = Trace { values: Vec::with_capacity(self.nodes.len()), bands: bands.then(Vec::new), balls: bands.then(Vec::new) };
        self.execute_range(inputs, &mut trace, 0, parameters)?;
        Ok(trace)
    }

    fn check_inputs(&self, inputs: &FamilyInputs) -> Result<(), ProgramError> {
        if inputs.slots.len() != self.declarations.slots.len() {
            return Err(ProgramError::Input("one value set per declared slot".to_string()));
        }
        Ok(())
    }

    /// One node's value and its enclosure (entrywise band, per-row ball): the affine, transposed,
    /// pointwise, norm, attention, indicator-readout and concatenation nodes carry their inputs'
    /// balls forward in `ℓ₂`; every other node first folds its arguments' balls into their boxes
    /// and is evaluated as a box ([`Trace`]).
    fn evaluate_enclosed(
        &self,
        index: usize,
        inputs: &FamilyInputs,
        (values, bands, balls): (&Layered<'_>, &Layered<'_>, &Balls<'_>),
        interfaces: &[Interface],
        frame: &Frame<'_>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), ProgramError> {
        let rows = inputs.rows;
        let node = &self.nodes[index];
        let value = |n: usize| values.get(n);
        let band = |n: usize| bands.get(n);
        let ball = |n: usize| balls.get(n);
        let carries = |n: usize| ball(n).iter().any(|v| *v != 0.0) || band(n).iter().any(|v| *v != 0.0);
        match node {
            Node::Affine { terms, bias } => {
                let (out, _) = self.evaluate_node(index, node, inputs, values, None, interfaces, frame)?;
                let tokens: Vec<Option<&[u32]>> =
                    terms.iter().map(|(argument, _)| self.frame_tokens(frame.nodes, frame.output, *argument, inputs).filter(|_| !values.patched(*argument))).collect();
                let gathers: Vec<Option<Vec<usize>>> = terms
                    .iter()
                    .zip(&tokens)
                    .map(|((argument, operator), tokens)| match (tokens, &self.operators[*operator].body) {
                        (Some(tokens), _) => Some(tokens.iter().map(|t| *t as usize).collect()),
                        (None, OperatorBody::Dense { .. }) => one_hot_columns(value(*argument)),
                        _ => None,
                    })
                    .collect();
                // A gather of an exact one-hot argument is one exact term, not `cols` of them.
                let summands: usize = terms
                    .iter()
                    .zip(&gathers)
                    .map(|((_, op), gather)| if gather.is_some() { 1 } else { self.operators[*op].cols.width() })
                    .sum::<usize>()
                    + 1;
                let growth = accumulation_growth(summands);
                let mut radius = Array2::<f64>::zeros(out.dim());
                let mut rho = Array1::<f64>::zeros(rows);
                for (((argument, operator), gather), tokens) in terms.iter().zip(&gathers).zip(&tokens) {
                    let op = &self.operators[*operator];
                    if let Some(tokens) = tokens {
                        // A gathered feature is exact: only the gather's own rounding.
                        add_gathered(op, tokens, None, Some(&mut radius), growth);
                        continue;
                    }
                    let x = value(*argument);
                    match &op.body {
                        OperatorBody::Identity => {
                            radius += band(*argument);
                            radius.zip_mut_with(x, |acc, xv| *acc += growth * xv.abs());
                            rho += ball(*argument);
                            continue;
                        }
                        OperatorBody::Diagonal { values: d, .. } => {
                            // A column scale carries the box entrywise and the ball by `max |d|`.
                            let magnitude = d.mapv(f64::abs);
                            let mut lifted = x.mapv(|xv| growth * xv.abs());
                            lifted += band(*argument);
                            radius += &(&lifted * &magnitude);
                            let largest = magnitude.iter().fold(0.0_f64, |acc, v| acc.max(*v));
                            rho.zip_mut_with(ball(*argument), |acc, b| *acc += (largest * b).next_up());
                            continue;
                        }
                        OperatorBody::Dense { values: a, .. } => match gather {
                            Some(columns) => {
                                for (row, &column) in columns.iter().enumerate() {
                                    radius.row_mut(row).zip_mut_with(&a.column(column), |acc, v| *acc += growth * v.abs());
                                }
                            }
                            None => radius += &x.mapv(|xv| growth * xv.abs()).dot(&a.mapv(f64::abs).t()),
                        },
                        OperatorBody::LowRank { left, right, .. } => {
                            let a = left.dot(right);
                            radius += &x.mapv(|xv| growth * xv.abs()).dot(&a.mapv(f64::abs).t());
                            let product = left.mapv(f64::abs).dot(&right.mapv(f64::abs)) * accumulation_growth(left.ncols());
                            radius += &x.mapv(f64::abs).dot(&product.t());
                        }
                    }
                    // The argument's error, box and ball, enters the output's ball through ‖A‖₂.
                    if carries(*argument) {
                        let sigma = spectral_bound(op)?;
                        let entering = row_norms(band(*argument)) + ball(*argument);
                        rho.zip_mut_with(&entering, |acc, e| *acc += (sigma * e).next_up());
                    }
                }
                if let Some(op) = bias {
                    let b = self.operators[*op].matrix().column(0).to_owned();
                    radius += &b.mapv(|bv| growth * bv.abs());
                }
                inflate_all(&mut radius, summands);
                rho.mapv_inplace(|v| inflate(v, terms.len()));
                Ok((out, radius, rho))
            }
            Node::Transposed { input, operator } => {
                let (out, _) = self.evaluate_node(index, node, inputs, values, None, interfaces, frame)?;
                let op = &self.operators[*operator];
                let a = op.matrix_cow();
                let x = value(*input);
                let mut radius = x.mapv(|v| accumulation_growth(a.nrows()) * v.abs()).dot(&a.mapv(f64::abs));
                inflate_all(&mut radius, a.nrows());
                let mut rho = Array1::<f64>::zeros(rows);
                if carries(*input) {
                    let sigma = spectral_bound(op)?;
                    rho = (row_norms(band(*input)) + ball(*input)).mapv(|e| inflate(sigma * e, 2));
                }
                Ok((out, radius, rho))
            }
            Node::Pointwise { laws, input } => {
                let (out, radius) = self.evaluate_node(index, node, inputs, values, Some(bands), interfaces, frame)?;
                let lipschitz = laws.iter().map(|law| law.lipschitz()).fold(0.0_f64, f64::max);
                let rho = ball(*input).mapv(|v| (lipschitz * v).next_up());
                Ok((out, radius.unwrap_or_else(|| Array2::zeros((rows, 0))), rho))
            }
            Node::RmsNorm { input, epsilon } => {
                let (out, radius) = self.evaluate_node(index, node, inputs, values, Some(bands), interfaces, frame)?;
                // `N(x) = x/s(x)` has `‖∂N‖₂ ≤ 1/s` at every point, so on the enclosure, whose norm is
                // at least `‖x̂‖ − ‖r‖₂ − ρ`, the ball's part moves the output by at most `ρ/s_min`.
                let x = value(*input);
                let n = x.ncols() as f64;
                let box_norms = row_norms(band(*input));
                let mut rho = Array1::<f64>::zeros(rows);
                for row in 0..rows {
                    let p = ball(*input)[row];
                    if p == 0.0 {
                        continue;
                    }
                    let squares = x.row(row).iter().map(|v| v * v).sum::<f64>() * (1.0 - accumulation_growth(x.ncols() + 2));
                    let low = (squares.max(0.0).sqrt() * (1.0 - 2.0 * UNIT_ROUNDOFF) - box_norms[row] - p).max(0.0);
                    let s_min = ((low * low / n) * (1.0 - 4.0 * UNIT_ROUNDOFF) + epsilon).sqrt() * (1.0 - 4.0 * UNIT_ROUNDOFF);
                    rho[row] = if s_min > 0.0 { (p / s_min).next_up().next_up() } else { f64::INFINITY };
                }
                Ok((out, radius.unwrap_or_else(|| Array2::zeros((rows, 0))), rho))
            }
            Node::Attend { query, key, value: payload, scale, rotary, causal } => {
                let (out, radius, rho) = self.attend(
                    inputs,
                    (value(*query), value(*key), value(*payload)),
                    Some((band(*query), band(*key), band(*payload))),
                    Some((ball(*query), ball(*key), ball(*payload))),
                    (*scale, *rotary, *causal),
                )?;
                Ok((out, radius.unwrap_or_else(|| Array2::zeros((rows, 0))), rho.unwrap_or_else(|| Array1::zeros(rows))))
            }
            Node::Readout { input, basis } if matches!(self.bases[*basis], Basis::Indicator { .. }) => {
                Ok((value(*input).clone(), band(*input).clone(), ball(*input).clone()))
            }
            Node::Concat { parts } => {
                let (out, radius) = self.evaluate_node(index, node, inputs, values, Some(bands), interfaces, frame)?;
                let mut rho = Array1::<f64>::zeros(rows);
                for part in parts {
                    rho.zip_mut_with(ball(*part), |acc, p| *acc += p * p);
                }
                rho.mapv_inplace(|v| inflate(v, parts.len()).sqrt().next_up());
                Ok((out, radius.unwrap_or_else(|| Array2::zeros((rows, 0))), rho))
            }
            _ => {
                // Fold every argument's ball into its box (`‖e‖₂ ≤ ρ` gives `|e_i| ≤ ρ`).
                let mut patch: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
                for argument in node.arguments() {
                    let rho = ball(argument);
                    if rho.iter().any(|v| *v != 0.0) {
                        let mut folded = band(argument).clone();
                        for (mut r, p) in folded.outer_iter_mut().zip(rho.iter()) {
                            r.mapv_inplace(|v| (v + p).next_up());
                        }
                        patch.insert(argument, folded);
                    }
                }
                let patched = Layered { base: bands.base, top: bands.top, from: bands.from, patch: Some(&patch) };
                let (out, radius) = self.evaluate_node(index, node, inputs, values, Some(&patched), interfaces, frame)?;
                let radius = radius.unwrap_or_else(|| Array2::zeros(out.dim()));
                Ok((out, radius, Array1::zeros(rows)))
            }
        }
    }

    fn evaluate_node(
        &self,
        index: usize,
        node: &Node,
        inputs: &FamilyInputs,
        values: &Layered<'_>,
        bands: Option<&Layered<'_>>,
        interfaces: &[Interface],
        frame: &Frame<'_>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ProgramError> {
        let rows = inputs.rows;
        let banded = bands.is_some();
        let value = |node: usize| values.get(node);
        let band = |node: usize| bands.map(|bands| bands.get(node));
        match node {
            Node::Param { index: param } => {
                let (v, b) = frame.args.get(*param).ok_or(ProgramError::Reference { what: "rule argument", index: *param })?;
                Ok((v.clone(), if banded { b.clone() } else { None }))
            }
            Node::Call { rule, arguments } => {
                let args: Vec<(Array2<f64>, Option<Array2<f64>>)> =
                    arguments.iter().map(|a| (value(*a).clone(), band(*a).cloned())).collect();
                self.execute_rule(*rule, &args, inputs, banded, frame.parameters)
            }
            Node::Attend { query, key, value, scale, rotary, causal } => {
                let (out, radius, _) = self.attend(
                    inputs,
                    (value_of(values, *query), value_of(values, *key), value_of(values, *value)),
                    bands.map(|b| (b.get(*query), b.get(*key), b.get(*value))),
                    None,
                    (*scale, *rotary, *causal),
                )?;
                Ok((out, radius))
            }
            Node::RmsNorm { input, epsilon } => Ok(rms_norm(value(*input), band(*input), *epsilon)),
            Node::Transposed { input, operator } => {
                let a = self.operators[*operator].matrix_cow();
                let x = value(*input);
                let out = fast_ab(x, a.as_ref());
                let radius = band(*input).map(|r| {
                    let growth = accumulation_growth(a.nrows());
                    let mut lifted = x.mapv(|v| growth * v.abs());
                    lifted += r;
                    let mut radius = fast_ab(&lifted, &a.mapv(f64::abs));
                    inflate_all(&mut radius, a.nrows());
                    radius
                });
                Ok((out, radius))
            }
            Node::Gain { input, coefficient } => {
                let (c, magnitude, operations) = coefficient.evaluate(frame.parameters)?;
                let x = value(*input);
                let out = x.mapv(|v| v * c);
                let radius = band(*input).map(|r| {
                    let c_error = accumulation_growth(operations) * magnitude;
                    let mut radius = Array2::<f64>::zeros(x.dim());
                    ndarray::Zip::from(&mut radius).and(x).and(r).for_each(|acc, &xv, &rv| {
                        *acc = inflate(c.abs() * rv + c_error * (xv.abs() + rv) + UNIT_ROUNDOFF * (xv * c).abs(), 4);
                    });
                    radius
                });
                Ok((out, radius))
            }
            Node::Feature { slot, basis } => {
                let SlotValues::Tokens(tokens) = &inputs.slots[*slot] else {
                    return Err(ProgramError::Input(format!("slot {slot} holds no tokens")));
                };
                if tokens.len() != rows {
                    return Err(ProgramError::Input(format!("slot {slot} has {} rows, not {rows}", tokens.len())));
                }
                if gathered(frame.nodes, frame.output, &self.bases, index) {
                    // A gathered feature: its readers read the token ids; the one-hot rows are
                    // never formed.
                    let width = self.bases[*basis].interface(&self.declarations)?.width();
                    if let Some(token) = tokens.iter().find(|t| **t as usize >= width) {
                        return Err(ProgramError::Input(format!("token {token} outside a domain of {width}")));
                    }
                    return Ok((Array2::zeros((rows, 0)), banded.then(|| Array2::zeros((rows, 0)))));
                }
                let features = self.bases[*basis].evaluate(&self.declarations, tokens)?;
                Ok((features.values, banded.then_some(features.bands)))
            }
            Node::Raw { slot } => {
                let SlotValues::Raw(values) = &inputs.slots[*slot] else {
                    return Err(ProgramError::Input(format!("slot {slot} holds no raw rows")));
                };
                if values.nrows() != rows {
                    return Err(ProgramError::Input(format!("slot {slot} has {} rows, not {rows}", values.nrows())));
                }
                Ok((values.clone(), banded.then(|| Array2::zeros(values.dim()))))
            }
            Node::Constant { operator } => {
                let column = self.operators[*operator].matrix().column(0).to_owned();
                let values = column.broadcast((rows, column.len())).map(|b| b.to_owned()).ok_or_else(|| {
                    ProgramError::Shape(format!("constant node {index} does not broadcast"))
                })?;
                Ok((values, banded.then(|| Array2::zeros((rows, column.len())))))
            }
            Node::Affine { terms, bias } => {
                let width = match terms.first() {
                    Some((_, op)) => self.operators[*op].rows.width(),
                    None => bias.map_or(0, |op| self.operators[op].rows.width()),
                };
                let summands: usize = terms.iter().map(|(_, op)| self.operators[*op].cols.width()).sum::<usize>() + 1;
                let growth = accumulation_growth(summands);
                let mut out = Array2::<f64>::zeros((rows, width));
                let mut radius = banded.then(|| Array2::<f64>::zeros((rows, width)));
                for (argument, operator) in terms {
                    let op = &self.operators[*operator];
                    // A gathered feature reads its tokens, unless the caller patched its value.
                    if let Some(tokens) = self.frame_tokens(frame.nodes, frame.output, *argument, inputs).filter(|_| !values.patched(*argument)) {
                        add_gathered(op, tokens, Some(&mut out), radius.as_mut(), growth);
                        continue;
                    }
                    let x = value(*argument);
                    match &op.body {
                        OperatorBody::Identity => {
                            out += x;
                            if let (Some(radius), Some(r)) = (radius.as_mut(), band(*argument)) {
                                radius.zip_mut_with(x, |acc, xv| *acc += growth * xv.abs());
                                *radius += r;
                            }
                        }
                        OperatorBody::Diagonal { values: d, .. } => {
                            // A column scale; one rounded product per entry is within the
                            // summation bound below.
                            out += &(x * d);
                            if let (Some(radius), Some(r)) = (radius.as_mut(), band(*argument)) {
                                let mut lifted = x.mapv(|xv| growth * xv.abs());
                                lifted += r;
                                *radius += &(&lifted * &d.mapv(f64::abs));
                            }
                        }
                        OperatorBody::Dense { values: a, .. } if one_hot_columns(x).is_some() => {
                            // Each row reads one column of `A` times an exact one: a gather, exact.
                            let columns = one_hot_columns(x).unwrap_or_default();
                            for (row, &column) in columns.iter().enumerate() {
                                out.row_mut(row).scaled_add(1.0, &a.column(column));
                            }
                            if let (Some(radius), Some(r)) = (radius.as_mut(), band(*argument)) {
                                if r.iter().any(|v| *v != 0.0) {
                                    *radius += &r.dot(&a.mapv(f64::abs).t());
                                }
                                for (row, &column) in columns.iter().enumerate() {
                                    radius.row_mut(row).zip_mut_with(&a.column(column), |acc, v| *acc += growth * v.abs());
                                }
                            }
                        }
                        OperatorBody::Dense { values: a, .. } => {
                            // A diagonal operator (a norm gain) is a column scale; one rounded
                            // product per entry is within the summation bound below.
                            let diagonal = op.diagonal();
                            match &diagonal {
                                Some(d) => out += &(x * d),
                                None if sparse_enough(x) => out += &sparse_abt(x, a),
                                None => {
                                    // A column of `x` zero on every row adds nothing, so only the
                                    // live columns enter the product (a masked site's gated
                                    // coordinates are mostly zero).
                                    let mut live = vec![false; x.ncols()];
                                    for row in x.outer_iter() {
                                        for (c, v) in row.iter().enumerate() {
                                            if *v != 0.0 {
                                                live[c] = true;
                                            }
                                        }
                                    }
                                    let columns: Vec<usize> = (0..x.ncols()).filter(|c| live[*c]).collect();
                                    if columns.len() == x.ncols() {
                                        out += &fast_abt(x, a);
                                    } else if !columns.is_empty() {
                                        out += &fast_abt(&x.select(Axis(1), &columns), &a.select(Axis(1), &columns));
                                    }
                                }
                            }
                            if let (Some(radius), Some(r)) = (radius.as_mut(), band(*argument)) {
                                let mut lifted = x.mapv(|xv| growth * xv.abs());
                                lifted += r;
                                match &diagonal {
                                    Some(d) => *radius += &(&lifted * &d.mapv(f64::abs)),
                                    None => *radius += &fast_abt(&lifted, &a.mapv(f64::abs)),
                                }
                            }
                        }
                        OperatorBody::LowRank { left, right, .. } => {
                            // Through the factors, never their product: `(x Rᵀ) Lᵀ`.
                            out += &fast_abt(&fast_abt(x, right), left);
                            if let (Some(radius), Some(r)) = (radius.as_mut(), band(*argument)) {
                                let a = left.dot(right);
                                let mut lifted = x.mapv(|xv| growth * xv.abs());
                                lifted += r;
                                *radius += &lifted.dot(&a.mapv(f64::abs).t());
                                let product = left.mapv(f64::abs).dot(&right.mapv(f64::abs)) * accumulation_growth(left.ncols());
                                *radius += &x.mapv(f64::abs).dot(&product.t());
                            }
                        }
                    }
                }
                if let Some(op) = bias {
                    let b = self.operators[*op].matrix().column(0).to_owned();
                    out += &b;
                    if let Some(radius) = radius.as_mut() {
                        *radius += &b.mapv(|bv| growth * bv.abs());
                    }
                }
                if let Some(radius) = radius.as_mut() {
                    inflate_all(radius, summands);
                }
                Ok((out, radius))
            }
            Node::Bilinear { left, right, scale } => {
                let (l, r) = (value(*left), value(*right));
                let c = scale.value();
                let mut out = Array2::<f64>::zeros((rows, 1));
                for (row, mut target) in out.outer_iter_mut().enumerate() {
                    target[0] = c * l.row(row).dot(&r.row(row));
                }
                let radius = match (band(*left), band(*right)) {
                    (Some(bl), Some(br)) => {
                        let growth = accumulation_growth(l.ncols() + 1);
                        let mut radius = Array2::<f64>::zeros((rows, 1));
                        for row in 0..rows {
                            let mut total = 0.0;
                            for i in 0..l.ncols() {
                                let (lv, rv) = (l[[row, i]].abs(), r[[row, i]].abs());
                                let (lr, rr) = (bl[[row, i]], br[[row, i]]);
                                total += growth * lv * rv + lv * rr + lr * rv + lr * rr;
                            }
                            radius[[row, 0]] = inflate(c.abs() * total, 4 * l.ncols());
                        }
                        Some(radius)
                    }
                    _ => None,
                };
                Ok((out, radius))
            }
            Node::Softmax { scores } => {
                let width = scores.len();
                let mut out = Array2::<f64>::zeros((rows, width));
                let mut radius = banded.then(|| Array2::<f64>::zeros((rows, width)));
                let u = UNIT_ROUNDOFF;
                for row in 0..rows {
                    let s: Vec<f64> = scores.iter().map(|&node| value(node)[[row, 0]]).collect();
                    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                    let total: f64 = e.iter().sum();
                    for (j, ej) in e.iter().enumerate() {
                        out[[row, j]] = ej / total;
                    }
                    if let Some(radius) = radius.as_mut() {
                        let spread = s.iter().map(|v| (v - m).abs()).fold(0.0, f64::max);
                        let eta = ((2.0 * u * spread).exp()
                            * (1.0 + 2.0 * u).powi(2)
                            * (1.0 + accumulation_growth(width.saturating_sub(1)))
                            * (1.0 + u)
                            - 1.0)
                            .next_up();
                        let big_r = scores
                            .iter()
                            .map(|&node| band(node).map_or(0.0, |b| b[[row, 0]]))
                            .fold(0.0, f64::max);
                        let factor = ((eta + (2.0 * big_r).exp_m1()) / (1.0 - eta)).next_up();
                        for j in 0..width {
                            radius[[row, j]] = softmax_radius(out[[row, j]], factor);
                        }
                    }
                }
                Ok((out, radius))
            }
            Node::Mix { weights, payloads } => {
                let alpha = value(*weights);
                let width = payloads
                    .first()
                    .map(|(_, node)| value(*node).ncols())
                    .ok_or_else(|| ProgramError::Shape(format!("mix node {index} has no payloads")))?;
                let mut out = Array2::<f64>::zeros((rows, width));
                for (j, payload) in payloads.iter() {
                    let (j, payload) = (*j, *payload);
                    let p = value(payload);
                    let a = alpha.column(j);
                    for row in 0..rows {
                        out.row_mut(row).scaled_add(a[row], &p.row(row));
                    }
                }
                let radius = match band(*weights) {
                    Some(ra) if banded => {
                        let growth = accumulation_growth(payloads.len());
                        let mut radius = Array2::<f64>::zeros((rows, width));
                        for &(j, payload) in payloads.iter() {
                            let (p, rp) = (value(payload), band(payload).ok_or_else(|| {
                                ProgramError::Shape("a banded execution lost a payload's band".to_string())
                            })?);
                            for row in 0..rows {
                                let (av, ar) = (alpha[[row, j]].abs(), ra[[row, j]]);
                                for col in 0..width {
                                    let (pv, pr) = (p[[row, col]].abs(), rp[[row, col]]);
                                    radius[[row, col]] += growth * av * pv + av * pr + ar * pv + ar * pr;
                                }
                            }
                        }
                        inflate_all(&mut radius, 4 * payloads.len());
                        Some(radius)
                    }
                    _ => None,
                };
                Ok((out, radius))
            }
            Node::Pointwise { input, laws } => {
                let x = value(*input);
                let interface = &interfaces[*input];
                let mut out = x.clone();
                let mut radius = band(*input).cloned();
                for (group, law) in laws.iter().enumerate() {
                    let range = interface.range(group);
                    let mut block = out.slice_mut(s![.., range.clone()]);
                    block.mapv_inplace(|v| law.apply(v));
                    if let Some(radius) = radius.as_mut() {
                        let mut r = radius.slice_mut(s![.., range.clone()]);
                        ndarray::Zip::from(&mut r)
                            .and(&block)
                            .and(&x.slice(s![.., range]))
                            .for_each(|r, v, t| *r = law.radius(*t, *v, *r));
                    }
                }
                Ok((out, radius))
            }
            Node::Hadamard { left, right } => {
                let (l, r) = (value(*left), value(*right));
                let out = l * r;
                let radius = match (band(*left), band(*right)) {
                    (Some(bl), Some(br)) => {
                        let mut radius = Array2::<f64>::zeros(out.dim());
                        ndarray::Zip::from(&mut radius).and(l).and(r).and(bl).and(br).for_each(
                            |acc, &lv, &rv, &lr, &rr| {
                                *acc = inflate(
                                    UNIT_ROUNDOFF * (lv * rv).abs() + lv.abs() * rr + lr * rv.abs() + lr * rr,
                                    4,
                                );
                            },
                        );
                        Some(radius)
                    }
                    _ => None,
                };
                Ok((out, radius))
            }
            Node::Concat { parts } => {
                let views: Vec<_> = parts.iter().map(|p| value(*p).view()).collect();
                let out = ndarray::concatenate(Axis(1), &views).map_err(|error| ProgramError::Shape(error.to_string()))?;
                let radius = match bands {
                    Some(_) => {
                        let band_views: Vec<_> = parts
                            .iter()
                            .map(|p| band(*p).map(|b| b.view()))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| ProgramError::Shape("a banded execution lost a part's band".to_string()))?;
                        Some(ndarray::concatenate(Axis(1), &band_views).map_err(|error| ProgramError::Shape(error.to_string()))?)
                    }
                    None => None,
                };
                Ok((out, radius))
            }
            Node::Outer { left, right } => {
                let (l, r) = (value(*left), value(*right));
                let (li, ri) = (&interfaces[*left], &interfaces[*right]);
                let width = l.ncols() * r.ncols();
                let mut out = Array2::<f64>::zeros((rows, width));
                let mut radius = banded.then(|| Array2::<f64>::zeros((rows, width)));
                let (bl, br) = (band(*left), band(*right));
                let mut offset = 0;
                for g1 in 0..li.group_count() {
                    for g2 in 0..ri.group_count() {
                        for i in li.range(g1) {
                            for j in ri.range(g2) {
                                for row in 0..rows {
                                    let (lv, rv) = (l[[row, i]], r[[row, j]]);
                                    out[[row, offset]] = lv * rv;
                                    if let (Some(radius), Some(bl), Some(br)) = (radius.as_mut(), bl, br) {
                                        let (lr, rr) = (bl[[row, i]], br[[row, j]]);
                                        radius[[row, offset]] = inflate(
                                            UNIT_ROUNDOFF * (lv * rv).abs() + lv.abs() * rr + lr * rv.abs() + lr * rr,
                                            4,
                                        );
                                    }
                                }
                                offset += 1;
                            }
                        }
                    }
                }
                Ok((out, radius))
            }
            Node::Readout { input, basis } => self.bases[*basis].read_banded(&self.declarations, value(*input), band(*input)),
        }
    }

    /// One node's interface (validates the whole program).
    pub fn node_interface(&self, node: usize) -> Result<Interface, ProgramError> {
        let mut interfaces = self.interfaces()?;
        if node >= interfaces.len() {
            return Err(ProgramError::Reference { what: "node", index: node });
        }
        Ok(interfaces.swap_remove(node))
    }

    /// Remove nodes the output does not read and operators and bases no node reads, keeping order.
    pub fn prune(&mut self) {
        let mut live = vec![false; self.nodes.len()];
        live[self.output] = true;
        for index in (0..self.nodes.len()).rev() {
            if live[index] {
                for argument in self.nodes[index].arguments() {
                    live[argument] = true;
                }
            }
        }
        let mut node_map = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if live[index] {
                node_map[index] = nodes.len();
                nodes.push(node.clone());
            }
        }
        let mut used_rules = vec![false; self.rules.len()];
        let mut pending: Vec<&Node> = nodes.iter().collect();
        while let Some(node) = pending.pop() {
            if let Node::Call { rule, .. } = node
                && !used_rules[*rule]
            {
                used_rules[*rule] = true;
                pending.extend(self.rules[*rule].nodes.iter());
            }
        }
        let mut used_ops = vec![false; self.operators.len()];
        let mut used_bases = vec![false; self.bases.len()];
        let rule_nodes = self.rules.iter().enumerate().filter(|(r, _)| used_rules[*r]).flat_map(|(_, rule)| rule.nodes.iter());
        for node in nodes.iter().chain(rule_nodes) {
            for op in node.operators() {
                used_ops[op] = true;
            }
            if let Node::Feature { basis, .. } | Node::Readout { basis, .. } = node {
                used_bases[*basis] = true;
            }
        }
        let op_map = compaction(&used_ops);
        let basis_map = compaction(&used_bases);
        let rule_map = compaction(&used_rules);
        let mut rules: Vec<Rule> = self.rules.iter().enumerate().filter(|(r, _)| used_rules[*r]).map(|(_, rule)| rule.clone()).collect();
        for rule in &mut rules {
            let identity: Vec<usize> = (0..rule.nodes.len()).collect();
            for node in &mut rule.nodes {
                remap_node(node, &identity, &op_map, &basis_map, &rule_map);
            }
        }
        let operators = self
            .operators
            .iter()
            .enumerate()
            .filter(|(i, _)| used_ops[*i])
            .map(|(_, op)| op.clone())
            .collect();
        let bases = self.bases.iter().enumerate().filter(|(i, _)| used_bases[*i]).map(|(_, b)| b.clone()).collect();
        for node in &mut nodes {
            remap_node(node, &node_map, &op_map, &basis_map, &rule_map);
        }
        self.output = node_map[self.output];
        self.nodes = nodes;
        self.operators = operators;
        self.bases = bases;
        self.rules = rules;
    }
}

/// What a node list may reference: the program's operators, bases and declarations, the rules it
/// may call, and (inside a rule body) the rule's argument interfaces.
struct Scope<'a> {
    operators: &'a [Arc<Operator>],
    bases: &'a [Basis],
    declarations: &'a Declarations,
    rules: &'a [Rule],
    params: &'a [Interface],
}

/// The interfaces of rule `rule`'s body nodes; the body may call only the rules before it.
fn rule_interfaces(
    rules: &[Rule],
    rule: usize,
    operators: &[Arc<Operator>],
    bases: &[Basis],
    declarations: &Declarations,
) -> Result<Vec<Interface>, ProgramError> {
    let body = rules.get(rule).ok_or(ProgramError::Reference { what: "rule", index: rule })?;
    let scope = Scope { operators, bases, declarations, rules: &rules[..rule], params: &body.inputs };
    let mut out = Vec::with_capacity(body.nodes.len());
    for (index, node) in body.nodes.iter().enumerate() {
        let interface = interface_of(index, node, &out, &scope)?;
        out.push(interface);
    }
    Ok(out)
}

/// The interface of node `index`, given the interfaces of the nodes before it.
fn interface_of(index: usize, node: &Node, out: &[Interface], scope: &Scope<'_>) -> Result<Interface, ProgramError> {
    let (operators, bases, declarations) = (scope.operators, scope.bases, scope.declarations);
    for argument in node.arguments() {
        if argument >= index {
            return Err(ProgramError::ForwardReference { node: index, argument });
        }
    }
    for operator in node.operators() {
        if operator >= operators.len() {
            return Err(ProgramError::Reference { what: "operator", index: operator });
        }
    }
    let interface = match node {
        Node::Feature { slot, basis } => {
            let basis_ref =
                bases.get(*basis).ok_or(ProgramError::Reference { what: "basis", index: *basis })?;
            match declarations.slots.get(*slot) {
                Some(Slot::Token { domain }) if *domain == basis_ref.domain() => {}
                _ => return Err(ProgramError::Reference { what: "token slot of the basis domain", index: *slot }),
            }
            basis_ref.interface(&declarations)?
        }
        Node::Raw { slot } => match declarations.slots.get(*slot) {
            Some(Slot::Raw { width }) => Interface::native(*width)?,
            _ => return Err(ProgramError::Reference { what: "raw slot", index: *slot }),
        },
        Node::Constant { operator } => {
            let op = &operators[*operator];
            if op.cols != Interface::constant() {
                return Err(ProgramError::Interface(format!("constant {} is not read from 1", op.name)));
            }
            op.rows.clone()
        }
        Node::Affine { terms, bias } => {
            let rows = terms
                .first()
                .map(|(_, op)| operators[*op].rows.clone())
                .or_else(|| bias.map(|op| operators[op].rows.clone()))
                .ok_or_else(|| ProgramError::Interface(format!("affine node {index} has no terms")))?;
            for (argument, operator) in terms {
                let op = &operators[*operator];
                if op.rows != rows || op.cols != out[*argument] {
                    return Err(ProgramError::Interface(format!(
                        "affine node {index}: operator {} does not map node {argument}'s interface to the node's",
                        op.name
                    )));
                }
            }
            if let Some(op) = bias {
                let op = &operators[*op];
                if op.rows != rows || op.cols != Interface::constant() {
                    return Err(ProgramError::Interface(format!("affine node {index}: bias {}", op.name)));
                }
            }
            rows
        }
        Node::Bilinear { left, right, .. } => {
            if out[*left].width() != out[*right].width() {
                return Err(ProgramError::Interface(format!("bilinear node {index}: widths differ")));
            }
            Interface::native(1)?
        }
        Node::Softmax { scores } => {
            if scores.is_empty() || scores.iter().any(|score| out[*score].width() != 1) {
                return Err(ProgramError::Interface(format!("softmax node {index} needs one-coordinate scores")));
            }
            Interface::uniform(scores.len(), 1, LabelKind::Position, 0)?
        }
        Node::Mix { weights, payloads } => {
            let (_, first) = payloads
                .first()
                .ok_or_else(|| ProgramError::Interface(format!("mix node {index} has no payloads")))?;
            let ascending = payloads.windows(2).all(|pair| pair[0].0 < pair[1].0);
            let inside = payloads.iter().all(|(column, _)| *column < out[*weights].width());
            if !ascending || !inside || payloads.iter().any(|(_, p)| out[*p] != out[*first]) {
                return Err(ProgramError::Interface(format!("mix node {index}: weights and payloads disagree")));
            }
            out[*first].clone()
        }
        Node::Pointwise { input, laws } => {
            if laws.len() != out[*input].group_count() {
                return Err(ProgramError::Interface(format!("pointwise node {index}: one law per group")));
            }
            out[*input].clone()
        }
        Node::Hadamard { left, right } => {
            if out[*left].width() != out[*right].width() {
                return Err(ProgramError::Interface(format!("hadamard node {index}: widths differ")));
            }
            out[*left].clone()
        }
        Node::Concat { parts } => {
            if parts.is_empty() {
                return Err(ProgramError::Interface(format!("concat node {index} has no parts")));
            }
            Interface::new(parts.iter().flat_map(|p| out[*p].groups().iter().copied()).collect())?
        }
        Node::Param { index: param } => {
            scope.params.get(*param).cloned().ok_or(ProgramError::Reference { what: "rule argument", index: *param })?
        }
        Node::Call { rule, arguments } => {
            let body = scope.rules.get(*rule).ok_or(ProgramError::Reference { what: "callable rule", index: *rule })?;
            if arguments.len() != body.inputs.len()
                || arguments.iter().zip(&body.inputs).any(|(argument, input)| out[*argument] != *input)
            {
                return Err(ProgramError::Interface(format!("call node {index}: arguments do not match rule {}", body.name)));
            }
            let body_interfaces = rule_interfaces(scope.rules, *rule, operators, bases, declarations)?;
            body_interfaces
                .get(body.output)
                .cloned()
                .ok_or(ProgramError::Reference { what: "rule output node", index: body.output })?
        }
        Node::Gain { input, coefficient } => {
            check_coefficient(coefficient, declarations.parameters)?;
            out[*input].clone()
        }
        Node::Attend { query, key, value, rotary, .. } => {
            let width = out[*query].width();
            if out[*key].width() != width || rotary.is_some_and(|r| r.dims as usize > width || r.dims % 2 == 1) {
                return Err(ProgramError::Interface(format!("attend node {index}: query, key and rotary widths disagree")));
            }
            out[*value].clone()
        }
        Node::RmsNorm { input, epsilon } => {
            if !(epsilon.is_finite() && *epsilon >= 0.0) {
                return Err(ProgramError::Interface(format!("rms norm node {index}: epsilon {epsilon}")));
            }
            out[*input].clone()
        }
        Node::Transposed { input, operator } => {
            let op = &operators[*operator];
            if op.rows != out[*input] {
                return Err(ProgramError::Interface(format!("transposed node {index}: operator {} rows are not its input", op.name)));
            }
            op.cols.clone()
        }
        Node::Outer { left, right } => {
            let (l, r) = (&out[*left], &out[*right]);
            let groups = l
                .groups()
                .iter()
                .enumerate()
                .flat_map(|(g1, a)| {
                    r.groups().iter().enumerate().map(move |(g2, b)| Group {
                        width: a.width * b.width,
                        label: Label::new(LabelKind::Pair, (g1 * r.group_count() + g2) as u32),
                    })
                })
                .collect();
            Interface::new(groups)?
        }
        Node::Readout { input, basis } => {
            let basis_ref =
                bases.get(*basis).ok_or(ProgramError::Reference { what: "basis", index: *basis })?;
            if basis_ref.interface(&declarations)? != out[*input] {
                return Err(ProgramError::Interface(format!("readout node {index}: input is not in the basis")));
            }
            Interface::uniform(declarations.domains[basis_ref.domain()].size, 1, LabelKind::Token, 0)?
        }
    };
    Ok(interface)
}

fn check_coefficient(coefficient: &Coefficient, parameters: usize) -> Result<(), ProgramError> {
    match coefficient {
        Coefficient::Parameter(index) if *index >= parameters => Err(ProgramError::Reference { what: "parameter", index: *index }),
        Coefficient::Number(v) if !v.is_finite() => Err(ProgramError::Interface(format!("a non-finite number {v}"))),
        Coefficient::Sum(terms) | Coefficient::Product(terms) => {
            if terms.is_empty() {
                return Err(ProgramError::Interface("an empty coefficient sum or product".to_string()));
            }
            terms.iter().try_for_each(|t| check_coefficient(t, parameters))
        }
        _ => Ok(()),
    }
}

fn compaction(used: &[bool]) -> Vec<usize> {
    let mut next = 0;
    used.iter()
        .map(|&keep| {
            if keep {
                next += 1;
                next - 1
            } else {
                usize::MAX
            }
        })
        .collect()
}

/// Rewrite a node's node, operator, basis and rule references through the maps.
pub fn remap_node(node: &mut Node, nodes: &[usize], operators: &[usize], bases: &[usize], rules: &[usize]) {
    match node {
        Node::Param { .. } => {}
        Node::Call { rule, arguments } => {
            *rule = rules[*rule];
            for argument in arguments.iter_mut() {
                *argument = nodes[*argument];
            }
        }
        Node::Gain { input, .. } | Node::RmsNorm { input, .. } => *input = nodes[*input],
        Node::Transposed { input, operator } => {
            *input = nodes[*input];
            *operator = operators[*operator];
        }
        Node::Attend { query, key, value, .. } => {
            *query = nodes[*query];
            *key = nodes[*key];
            *value = nodes[*value];
        }
        Node::Feature { basis, .. } => *basis = bases[*basis],
        Node::Raw { .. } => {}
        Node::Constant { operator } => *operator = operators[*operator],
        Node::Affine { terms, bias } => {
            for (argument, operator) in terms.iter_mut() {
                *argument = nodes[*argument];
                *operator = operators[*operator];
            }
            if let Some(op) = bias.as_mut() {
                *op = operators[*op];
            }
        }
        Node::Bilinear { left, right, .. } | Node::Hadamard { left, right } | Node::Outer { left, right } => {
            *left = nodes[*left];
            *right = nodes[*right];
        }
        Node::Softmax { scores } => {
            for score in scores.iter_mut() {
                *score = nodes[*score];
            }
        }
        Node::Concat { parts } => {
            for part in parts.iter_mut() {
                *part = nodes[*part];
            }
        }
        Node::Mix { weights, payloads } => {
            *weights = nodes[*weights];
            for (_, payload) in payloads.iter_mut() {
                *payload = nodes[*payload];
            }
        }
        Node::Pointwise { input, .. } => *input = nodes[*input],
        Node::Readout { input, basis } => {
            *input = nodes[*input];
            *basis = bases[*basis];
        }
    }
}

// ------------------------------------------------------------------------------------------------ code

fn interface_bits(interface: &Interface) -> Result<u64, ProgramError> {
    let runs = interface.runs();
    let mut bits = prefix_integer_len_bits(runs.len() as u64)?;
    for (count, width, _, first) in runs {
        bits += prefix_integer_len_bits(count as u64 + 1)?
            + prefix_integer_len_bits(width as u64)?
            + u64::from(fixed_index_len_bits(LABEL_KINDS.len())?)
            + prefix_integer_len_bits(u64::from(first) + 1)?;
    }
    Ok(bits)
}

fn write_interface(out: &mut BitString, interface: &Interface) -> Result<(), ProgramError> {
    let runs = interface.runs();
    encode_prefix_integer(out, runs.len() as u64)?;
    for (count, width, kind, first) in runs {
        encode_prefix_integer(out, count as u64 + 1)?;
        encode_prefix_integer(out, width as u64)?;
        let kind_index = LABEL_KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        encode_fixed_index(out, kind_index, LABEL_KINDS.len())?;
        encode_prefix_integer(out, u64::from(first) + 1)?;
    }
    Ok(())
}

fn read_interface(reader: &mut BitReader<'_>) -> Result<Interface, ProgramError> {
    let runs = decode_prefix_integer(reader)?;
    let mut groups = Vec::new();
    for _ in 0..runs {
        let count = decode_prefix_integer(reader)? - 1;
        let width = decode_prefix_integer(reader)? as usize;
        let kind = LABEL_KINDS[decode_fixed_index(reader, LABEL_KINDS.len())?];
        let first = (decode_prefix_integer(reader)? - 1) as u32;
        if count > u64::from(u32::MAX) {
            return Err(ProgramError::Code(format!("an interface run of {count} groups")));
        }
        groups.extend((0..count as u32).map(|i| Group { width, label: Label::new(kind, first + i) }));
    }
    Interface::new(groups)
}

/// The length of [`write_lattice`]'s message.
fn lattice_bits(reals: &[f64], precision: DeclaredPrecision) -> Result<u64, ProgramError> {
    ordered_lattice_bits(&lattice_indices(reals, precision)?, precision, &[])
}

/// Reals per lattice chunk. A lattice of more reals is indexed, sized and encoded chunk by chunk in
/// parallel, and decoded so when its chunk boundaries are known; its message is the same.
const LATTICE_CHUNK: usize = 1 << 18;

/// The lattice indices of `reals` ([`LatticeCode::encode`]), large lattices in parallel chunks.
fn lattice_indices(reals: &[f64], precision: DeclaredPrecision) -> Result<Vec<i64>, ProgramError> {
    if reals.len() <= LATTICE_CHUNK {
        return Ok(LatticeCode::encode(reals, precision).map_err(ProgramError::Code)?.into_indices());
    }
    let chunks = reals
        .par_chunks(LATTICE_CHUNK)
        .enumerate()
        .map(|(c, chunk)| {
            LatticeCode::encode(chunk, precision)
                .map(LatticeCode::into_indices)
                .map_err(|error| ProgramError::Code(format!("lattice chunk at real {}: {error}", c * LATTICE_CHUNK)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(chunks.concat())
}

/// The length of a lattice message whose reals at `leads` after the first are sent as increments.
fn ordered_lattice_bits(indices: &[i64], precision: DeclaredPrecision, leads: &[usize]) -> Result<u64, ProgramError> {
    let mut bits = prefix_integer_len_bits(indices.len() as u64 + 1)?
        + signed_prefix_integer_len_bits(i64::from(precision.fraction_bits()))?;
    bits += indices
        .par_chunks(LATTICE_CHUNK)
        .map(|chunk| chunk.iter().try_fold(0u64, |sum, &index| Ok::<_, CodecError>(sum + signed_delta_len_bits(index)?)))
        .try_reduce(|| 0, |a, b| Ok(a + b))?;
    for pair in leads.windows(2) {
        bits -= signed_delta_len_bits(indices[pair[1]])?;
        bits += elias_delta_len_bits(increment(indices[pair[0]], indices[pair[1]])? + 1)?;
    }
    Ok(bits)
}

fn increment(from: i64, to: i64) -> Result<u64, ProgramError> {
    u64::try_from(i128::from(to) - i128::from(from))
        .map_err(|_| ProgramError::Code(format!("ordered rows decrease from {from} to {to}")))
}

/// The position in [`Operator::present_reals`] of each nonempty row's first real, in row order
/// (blocks row group by row group, row-major within a block); `None` for fewer than two.
fn row_leads(rows: &Interface, cols: &Interface, present: &Array2<bool>) -> Option<Vec<usize>> {
    let mut leads = Vec::new();
    let mut at = 0;
    for (g, group) in rows.groups().iter().enumerate() {
        let mut first = true;
        for (c, col) in cols.groups().iter().enumerate() {
            if !present[[g, c]] {
                continue;
            }
            if first {
                leads.extend((0..group.width).map(|k| at + k * col.width));
                first = false;
            }
            at += group.width * col.width;
        }
    }
    (leads.len() >= 2).then_some(leads)
}

/// The leads of a dense operator sent as the ordered kind: its rows' first reals do not decrease,
/// and the ordered message is strictly shorter than the plain one.
fn ordered_leads(operator: &Operator) -> Result<Option<Vec<usize>>, ProgramError> {
    let OperatorBody::Dense { precision, .. } = &operator.body else { return Ok(None) };
    ordered_leads_of(operator, &lattice_indices(&operator.present_reals(), *precision)?)
}

/// [`ordered_leads`] given the operator's lattice indices.
fn ordered_leads_of(operator: &Operator, indices: &[i64]) -> Result<Option<Vec<usize>>, ProgramError> {
    let OperatorBody::Dense { present, precision, .. } = &operator.body else { return Ok(None) };
    let Some(leads) = row_leads(&operator.rows, &operator.cols, present) else { return Ok(None) };
    if leads.windows(2).any(|pair| indices[pair[1]] < indices[pair[0]]) {
        return Ok(None);
    }
    let ordered = ordered_lattice_bits(indices, *precision, &leads)?;
    Ok((ordered < ordered_lattice_bits(indices, *precision, &[])?).then_some(leads))
}

/// Reals on one lattice as a message: `count + 1` in the prefix code, the fraction bits in the
/// signed prefix code, then each lattice index in the signed Elias δ code, whose subadditivity
/// (`codec::elias_delta_len_bits`) keeps a split of a real from ever being shorter than the real.
fn write_lattice(out: &mut BitString, reals: &[f64], precision: DeclaredPrecision, leads: &[usize]) -> Result<(), ProgramError> {
    write_lattice_indices(out, &lattice_indices(reals, precision)?, precision, leads)
}

/// [`write_lattice`] given the lattice indices. Chunks of a large lattice are encoded in parallel
/// and appended in order, so the message is bit-identical to the sequential one.
fn write_lattice_indices(out: &mut BitString, indices: &[i64], precision: DeclaredPrecision, leads: &[usize]) -> Result<(), ProgramError> {
    encode_prefix_integer(out, indices.len() as u64 + 1)?;
    encode_signed_prefix_integer(out, i64::from(precision.fraction_bits()))?;
    if indices.len() <= LATTICE_CHUNK {
        return write_index_range(out, indices, leads, 0..indices.len());
    }
    let chunks = (0..indices.len().div_ceil(LATTICE_CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut chunk = BitString::new();
            write_index_range(&mut chunk, indices, leads, c * LATTICE_CHUNK..((c + 1) * LATTICE_CHUNK).min(indices.len()))?;
            Ok(chunk)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    for chunk in &chunks {
        out.append(chunk);
    }
    Ok(())
}

/// The codewords of `indices[range]`: increments at leads after the first, signed δ otherwise.
fn write_index_range(out: &mut BitString, indices: &[i64], leads: &[usize], range: Range<usize>) -> Result<(), ProgramError> {
    // Leads are increasing; those before the range were passed by the sequential writer.
    let mut lead = leads.partition_point(|l| *l < range.start);
    for k in range {
        let index = indices[k];
        if lead < leads.len() && leads[lead] == k {
            lead += 1;
            if lead > 1 {
                encode_elias_delta(out, increment(indices[leads[lead - 2]], index)? + 1)?;
                continue;
            }
        }
        encode_signed_delta(out, index)?;
    }
    Ok(())
}

/// Pass over a [`write_lattice`] message, returning the bit position of every [`LATTICE_CHUNK`]th
/// index codeword (the first after the precision). The same codewords are refused as on reading.
fn skip_lattice(reader: &mut BitReader<'_>) -> Result<Vec<u64>, ProgramError> {
    let count = decode_prefix_integer(reader)? - 1;
    if count > reader.remaining_bits() {
        return Err(ProgramError::Code(format!("a lattice of {count} reals beyond the message")));
    }
    decode_signed_prefix_integer(reader)?;
    let mut starts = Vec::with_capacity((count as usize).div_ceil(LATTICE_CHUNK).max(1));
    for k in 0..count as usize {
        if k % LATTICE_CHUNK == 0 {
            starts.push(reader.position());
        }
        skip_elias_delta(reader)?;
    }
    Ok(starts)
}

/// [`read_lattice`] with known chunk starts ([`skip_lattice`]): chunks decode in parallel, each
/// required to end exactly where the next begins, and the lattice to end the reader. Every
/// codeword is read exactly as [`read_lattice`] reads it, so the result is the same.
fn read_lattice_chunked(reader: &mut BitReader<'_>, leads: &[usize], starts: &[u64]) -> Result<(DeclaredPrecision, Vec<f64>), ProgramError> {
    let count = decode_prefix_integer(reader)? - 1;
    if count > reader.remaining_bits() {
        return Err(ProgramError::Code(format!("a lattice of {count} reals beyond the message")));
    }
    let fraction_bits = i32::try_from(decode_signed_prefix_integer(reader)?)
        .map_err(|error| ProgramError::Code(format!("fraction bits: {error}")))?;
    let precision = DeclaredPrecision::new(fraction_bits).map_err(ProgramError::Code)?;
    let count = count as usize;
    if starts.len() != count.div_ceil(LATTICE_CHUNK) || starts.first().is_some_and(|s| *s != reader.position()) {
        return Err(ProgramError::Code("lattice chunk starts do not match the message".into()));
    }
    let end = reader.position() + reader.remaining_bits();
    let span = gam_gpu::trace::host_span("codec.lattice.chunks");
    let raw = starts
        .par_iter()
        .enumerate()
        .map(|(c, &start)| {
            let stop = starts.get(c + 1).copied().unwrap_or(end);
            let mut part = reader.window(start, stop)?;
            let n = LATTICE_CHUNK.min(count - c * LATTICE_CHUNK);
            let values = (0..n).map(|_| decode_elias_delta(&mut part)).collect::<Result<Vec<u64>, _>>()?;
            part.finish()?;
            Ok(values)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?
        .concat();
    drop(span);
    let span = gam_gpu::trace::host_span("codec.lattice.values");
    reader.skip_to(end)?;
    let mut indices: Vec<i64> = Vec::with_capacity(count);
    let mut lead = 0;
    for (k, value) in raw.into_iter().enumerate() {
        if lead < leads.len() && leads[lead] == k {
            lead += 1;
            if lead > 1 {
                let step = value - 1;
                let previous = indices[leads[lead - 2]];
                let index = i64::try_from(i128::from(previous) + i128::from(step))
                    .map_err(|_| ProgramError::Code(format!("an ordered increment {step} overflows")))?;
                indices.push(index);
                continue;
            }
        }
        indices.push(crate::codec::signed_delta_of_codeword(value));
    }
    if lead != leads.len() {
        return Err(ProgramError::Code(format!("{} ordered rows for {count} reals", leads.len())));
    }
    let code = LatticeCode::from_indices(precision, indices).map_err(ProgramError::Code)?;
    let values = code.decode().map_err(ProgramError::Code)?;
    drop(span);
    Ok((precision, values))
}

/// Read a [`write_lattice`] message with the given leads: the precision and the decoded reals.
fn read_lattice(reader: &mut BitReader<'_>, leads: &[usize]) -> Result<(DeclaredPrecision, Vec<f64>), ProgramError> {
    let count = decode_prefix_integer(reader)? - 1;
    if count > reader.remaining_bits() {
        return Err(ProgramError::Code(format!("a lattice of {count} reals beyond the message")));
    }
    let fraction_bits = i32::try_from(decode_signed_prefix_integer(reader)?)
        .map_err(|error| ProgramError::Code(format!("fraction bits: {error}")))?;
    let precision = DeclaredPrecision::new(fraction_bits).map_err(ProgramError::Code)?;
    let mut indices: Vec<i64> = Vec::with_capacity(count as usize);
    let mut lead = 0;
    for k in 0..count as usize {
        if lead < leads.len() && leads[lead] == k {
            lead += 1;
            if lead > 1 {
                let step = decode_elias_delta(reader)? - 1;
                let previous = indices[leads[lead - 2]];
                let index = i64::try_from(i128::from(previous) + i128::from(step))
                    .map_err(|_| ProgramError::Code(format!("an ordered increment {step} overflows")))?;
                indices.push(index);
                continue;
            }
        }
        indices.push(decode_signed_delta(reader)?);
    }
    if lead != leads.len() {
        return Err(ProgramError::Code(format!("{} ordered rows for {count} reals", leads.len())));
    }
    let code = LatticeCode::from_indices(precision, indices).map_err(ProgramError::Code)?;
    Ok((precision, code.decode().map_err(ProgramError::Code)?))
}

/// An operator's message length as (structure bits, precision bits): the lattice message's count
/// field is structure (the present blocks fix it), its fraction bits and indices are precision.
fn operator_bits(operator: &Operator) -> Result<(u64, u64), ProgramError> {
    let kind = u64::from(fixed_index_len_bits(OPERATOR_KINDS)?);
    let split = |reals: &[f64], precision: DeclaredPrecision| -> Result<(u64, u64), ProgramError> {
        let count = prefix_integer_len_bits(reals.len() as u64 + 1)?;
        Ok((count, lattice_bits(reals, precision)? - count))
    };
    match &operator.body {
        OperatorBody::Identity => Ok((kind + interface_bits(&operator.rows)?, 0)),
        OperatorBody::Diagonal { precision, .. } => {
            let (count, reals) = split(&operator.present_reals(), *precision)?;
            Ok((kind + interface_bits(&operator.rows)? + count, reals))
        }
        OperatorBody::LowRank { left, precision, .. } => {
            let (count, reals) = split(&operator.present_reals(), *precision)?;
            Ok((
                kind + interface_bits(&operator.rows)?
                    + interface_bits(&operator.cols)?
                    + prefix_integer_len_bits(left.ncols() as u64)?
                    + count,
                reals,
            ))
        }
        OperatorBody::Dense { present, precision, .. } => {
            let mut structure = kind + interface_bits(&operator.rows)? + interface_bits(&operator.cols)?;
            let columns = operator.cols.group_count();
            for row in present.outer_iter() {
                structure += subset_code_len_bits(columns, row.iter().filter(|keep| **keep).count())?;
            }
            let reals = operator.present_reals();
            let total = match ordered_leads(operator)? {
                Some(leads) => {
                    let code = LatticeCode::encode(&reals, *precision).map_err(ProgramError::Code)?;
                    ordered_lattice_bits(code.indices(), *precision, &leads)?
                }
                None => lattice_bits(&reals, *precision)?,
            };
            let count = prefix_integer_len_bits(reals.len() as u64 + 1)?;
            Ok((structure + count, total - count))
        }
    }
}

/// What a node's code references: the alphabets its fixed indices are drawn from.
struct NodeCode<'a> {
    operators: usize,
    bases: usize,
    slots: usize,
    parameters: usize,
    /// The input count of each callable rule.
    rule_inputs: &'a [usize],
    /// The argument count of the enclosing rule (0 at the top level).
    params: usize,
}

fn encode_coefficient(out: &mut BitString, coefficient: &Coefficient, parameters: usize) -> Result<(), ProgramError> {
    match coefficient {
        Coefficient::Parameter(index) => {
            encode_fixed_index(out, 0, COEFFICIENT_KINDS)?;
            encode_fixed_index(out, *index, parameters.max(1))?;
        }
        Coefficient::Number(value) => {
            encode_fixed_index(out, 1, COEFFICIENT_KINDS)?;
            write_lattice(out, &[*value], exact_precision([*value])?, &[])?;
        }
        Coefficient::Sum(terms) | Coefficient::Product(terms) => {
            encode_fixed_index(out, if matches!(coefficient, Coefficient::Sum(_)) { 2 } else { 3 }, COEFFICIENT_KINDS)?;
            encode_prefix_integer(out, terms.len() as u64)?;
            for term in terms {
                encode_coefficient(out, term, parameters)?;
            }
        }
    }
    Ok(())
}

fn decode_coefficient(reader: &mut BitReader<'_>, parameters: usize, depth: usize) -> Result<Coefficient, ProgramError> {
    if depth > 64 {
        return Err(ProgramError::Code("a coefficient nested beyond 64 levels".to_string()));
    }
    Ok(match decode_fixed_index(reader, COEFFICIENT_KINDS)? {
        0 => Coefficient::Parameter(decode_fixed_index(reader, parameters.max(1))?),
        1 => {
            let (_, reals) = read_lattice(reader, &[])?;
            Coefficient::Number(*reals.first().ok_or_else(|| ProgramError::Code("an empty coefficient number".to_string()))?)
        }
        kind => {
            let count = decode_prefix_integer(reader)?;
            if count > reader.remaining_bits() {
                return Err(ProgramError::Code("coefficient terms beyond the message".to_string()));
            }
            let terms = (0..count).map(|_| decode_coefficient(reader, parameters, depth + 1)).collect::<Result<_, _>>()?;
            if kind == 2 { Coefficient::Sum(terms) } else { Coefficient::Product(terms) }
        }
    })
}

/// Write node `index` of a node list whose earlier nodes have `interfaces`.
fn encode_node(out: &mut BitString, node: &Node, index: usize, code: &NodeCode<'_>, interfaces: &[Interface]) -> Result<(), ProgramError> {
    let refs = index.max(1);
    let (ops, bases, slots) = (code.operators.max(1), code.bases.max(1), code.slots.max(1));
    encode_fixed_index(out, node.kind_index(), NODE_KINDS)?;
    match node {
        Node::Feature { slot, basis } => {
            encode_fixed_index(out, *slot, slots)?;
            encode_fixed_index(out, *basis, bases)?;
        }
        Node::Raw { slot } => encode_fixed_index(out, *slot, slots)?,
        Node::Constant { operator } => encode_fixed_index(out, *operator, ops)?,
        Node::Affine { terms, bias } => {
            encode_prefix_integer(out, terms.len() as u64 + 1)?;
            for (argument, operator) in terms {
                encode_fixed_index(out, *argument, refs)?;
                encode_fixed_index(out, *operator, ops)?;
            }
            out.push_bit(bias.is_some());
            encode_fixed_index(out, bias.unwrap_or(0), ops)?;
        }
        Node::Bilinear { left, right, scale } => {
            encode_fixed_index(out, *left, refs)?;
            encode_fixed_index(out, *right, refs)?;
            match scale {
                Scale::One => out.push_bit(false),
                Scale::InverseSqrt(n) => {
                    out.push_bit(true);
                    encode_prefix_integer(out, u64::from(*n))?;
                }
            }
        }
        Node::Softmax { scores: list } | Node::Concat { parts: list } => {
            encode_prefix_integer(out, list.len() as u64)?;
            for item in list {
                encode_fixed_index(out, *item, refs)?;
            }
        }
        Node::Mix { weights, payloads } => {
            encode_fixed_index(out, *weights, refs)?;
            let columns: Vec<usize> = payloads.iter().map(|(column, _)| *column).collect();
            encode_subset(out, interfaces[*weights].width(), &columns)?;
            for (_, payload) in payloads {
                encode_fixed_index(out, *payload, refs)?;
            }
        }
        Node::Pointwise { input, laws } => {
            encode_fixed_index(out, *input, refs)?;
            for law in laws {
                let at = LAWS.iter().position(|l| l == law).unwrap_or(0);
                encode_fixed_index(out, at, LAWS.len())?;
            }
        }
        Node::Hadamard { left, right } | Node::Outer { left, right } => {
            encode_fixed_index(out, *left, refs)?;
            encode_fixed_index(out, *right, refs)?;
        }
        Node::Readout { input, basis } => {
            encode_fixed_index(out, *input, refs)?;
            encode_fixed_index(out, *basis, bases)?;
        }
        Node::Param { index: param } => encode_fixed_index(out, *param, code.params.max(1))?,
        Node::Call { rule, arguments } => {
            encode_fixed_index(out, *rule, code.rule_inputs.len().max(1))?;
            for argument in arguments {
                encode_fixed_index(out, *argument, refs)?;
            }
        }
        Node::Gain { input, coefficient } => {
            encode_fixed_index(out, *input, refs)?;
            encode_coefficient(out, coefficient, code.parameters)?;
        }
        Node::Attend { query, key, value, scale, rotary, causal } => {
            encode_fixed_index(out, *query, refs)?;
            encode_fixed_index(out, *key, refs)?;
            encode_fixed_index(out, *value, refs)?;
            match scale {
                Scale::One => out.push_bit(false),
                Scale::InverseSqrt(n) => {
                    out.push_bit(true);
                    encode_prefix_integer(out, u64::from(*n))?;
                }
            }
            out.push_bit(rotary.is_some());
            if let Some(rotary) = rotary {
                encode_prefix_integer(out, u64::from(rotary.base))?;
                encode_prefix_integer(out, u64::from(rotary.dims))?;
                out.push_bit(rotary.half_split);
            }
            out.push_bit(*causal);
        }
        Node::RmsNorm { input, epsilon } => {
            encode_fixed_index(out, *input, refs)?;
            write_lattice(out, &[*epsilon], exact_precision([*epsilon])?, &[])?;
        }
        Node::Transposed { input, operator } => {
            encode_fixed_index(out, *input, refs)?;
            encode_fixed_index(out, *operator, ops)?;
        }
    }
    Ok(())
}

/// Read node `index` of a node list whose earlier nodes have `interfaces`.
fn decode_node(reader: &mut BitReader<'_>, index: usize, code: &NodeCode<'_>, interfaces: &[Interface]) -> Result<Node, ProgramError> {
    let refs = index.max(1);
    let (ops, bases, slots) = (code.operators.max(1), code.bases.max(1), code.slots.max(1));
    let interface = |node: usize| interfaces.get(node).ok_or(ProgramError::Reference { what: "node", index: node });
    Ok(match decode_fixed_index(reader, NODE_KINDS)? {
        0 => Node::Feature { slot: decode_fixed_index(reader, slots)?, basis: decode_fixed_index(reader, bases)? },
        1 => Node::Raw { slot: decode_fixed_index(reader, slots)? },
        2 => Node::Constant { operator: decode_fixed_index(reader, ops)? },
        3 => {
            let count = decode_prefix_integer(reader)? - 1;
            if count > reader.remaining_bits() {
                return Err(ProgramError::Code("affine term count beyond the message".to_string()));
            }
            let mut terms = Vec::new();
            for _ in 0..count {
                terms.push((decode_fixed_index(reader, refs)?, decode_fixed_index(reader, ops)?));
            }
            let has_bias = reader.read_bit()?;
            let bias = decode_fixed_index(reader, ops)?;
            Node::Affine { terms, bias: has_bias.then_some(bias) }
        }
        4 => {
            let left = decode_fixed_index(reader, refs)?;
            let right = decode_fixed_index(reader, refs)?;
            let scale = if reader.read_bit()? {
                Scale::InverseSqrt(
                    u32::try_from(decode_prefix_integer(reader)?)
                        .map_err(|error| ProgramError::Code(format!("scale argument: {error}")))?,
                )
            } else {
                Scale::One
            };
            Node::Bilinear { left, right, scale }
        }
        kind @ (5 | 11) => {
            let count = decode_prefix_integer(reader)?;
            if count > reader.remaining_bits() {
                return Err(ProgramError::Code("list length beyond the message".to_string()));
            }
            let list: Vec<usize> = (0..count).map(|_| decode_fixed_index(reader, refs)).collect::<Result<_, _>>()?;
            if kind == 5 { Node::Softmax { scores: list } } else { Node::Concat { parts: list } }
        }
        6 => {
            let weights = decode_fixed_index(reader, refs)?;
            let width = interface(weights)?.width();
            let columns = decode_subset(reader, width)?;
            let payloads = columns
                .into_iter()
                .map(|column| decode_fixed_index(reader, refs).map(|node| (column, node)))
                .collect::<Result<_, _>>()?;
            Node::Mix { weights, payloads }
        }
        7 => {
            let input = decode_fixed_index(reader, refs)?;
            let groups = interface(input)?.group_count();
            let laws = (0..groups).map(|_| decode_fixed_index(reader, LAWS.len()).map(|at| LAWS[at])).collect::<Result<_, _>>()?;
            Node::Pointwise { input, laws }
        }
        8 => Node::Hadamard { left: decode_fixed_index(reader, refs)?, right: decode_fixed_index(reader, refs)? },
        9 => Node::Readout { input: decode_fixed_index(reader, refs)?, basis: decode_fixed_index(reader, bases)? },
        10 => Node::Outer { left: decode_fixed_index(reader, refs)?, right: decode_fixed_index(reader, refs)? },
        12 => Node::Param { index: decode_fixed_index(reader, code.params.max(1))? },
        13 => {
            let rule = decode_fixed_index(reader, code.rule_inputs.len().max(1))?;
            let count = *code.rule_inputs.get(rule).ok_or(ProgramError::Reference { what: "callable rule", index: rule })?;
            let arguments = (0..count).map(|_| decode_fixed_index(reader, refs)).collect::<Result<_, _>>()?;
            Node::Call { rule, arguments }
        }
        14 => {
            let input = decode_fixed_index(reader, refs)?;
            Node::Gain { input, coefficient: decode_coefficient(reader, code.parameters, 0)? }
        }
        15 => {
            let query = decode_fixed_index(reader, refs)?;
            let key = decode_fixed_index(reader, refs)?;
            let value = decode_fixed_index(reader, refs)?;
            let scale = if reader.read_bit()? {
                Scale::InverseSqrt(
                    u32::try_from(decode_prefix_integer(reader)?)
                        .map_err(|error| ProgramError::Code(format!("scale argument: {error}")))?,
                )
            } else {
                Scale::One
            };
            let rotary = if reader.read_bit()? {
                let base = u32::try_from(decode_prefix_integer(reader)?).map_err(|e| ProgramError::Code(e.to_string()))?;
                let dims = u32::try_from(decode_prefix_integer(reader)?).map_err(|e| ProgramError::Code(e.to_string()))?;
                Some(Rotary { base, dims, half_split: reader.read_bit()? })
            } else {
                None
            };
            Node::Attend { query, key, value, scale, rotary, causal: reader.read_bit()? }
        }
        16 => {
            let input = decode_fixed_index(reader, refs)?;
            let (_, reals) = read_lattice(reader, &[])?;
            Node::RmsNorm { input, epsilon: *reals.first().ok_or_else(|| ProgramError::Code("an rms norm without epsilon".to_string()))? }
        }
        _ => Node::Transposed { input: decode_fixed_index(reader, refs)?, operator: decode_fixed_index(reader, ops)? },
    })
}

/// Write the rules section: `#rules + 1`, then per rule its input count + 1 and interfaces, its node
/// count + 1, its nodes (calling only earlier rules) and its output node.
fn encode_rules(out: &mut BitString, program: &OperatorProgram) -> Result<(), ProgramError> {
    encode_prefix_integer(out, program.rules.len() as u64 + 1)?;
    let rule_inputs: Vec<usize> = program.rules.iter().map(|r| r.inputs.len()).collect();
    for (r, rule) in program.rules.iter().enumerate() {
        encode_prefix_integer(out, rule.inputs.len() as u64 + 1)?;
        for input in &rule.inputs {
            write_interface(out, input)?;
        }
        encode_prefix_integer(out, rule.nodes.len() as u64 + 1)?;
        let interfaces = rule_interfaces(&program.rules, r, &program.operators, &program.bases, &program.declarations)?;
        let code = NodeCode {
            operators: program.operators.len(),
            bases: program.bases.len(),
            slots: program.declarations.slots.len(),
            parameters: program.declarations.parameters,
            rule_inputs: &rule_inputs[..r],
            params: rule.inputs.len(),
        };
        for (index, node) in rule.nodes.iter().enumerate() {
            encode_node(out, node, index, &code, &interfaces)?;
        }
        encode_fixed_index(out, rule.output, rule.nodes.len().max(1))?;
    }
    Ok(())
}

fn decode_rules(
    reader: &mut BitReader<'_>,
    operators: &[Arc<Operator>],
    bases: &[Basis],
    declarations: &Declarations,
) -> Result<Vec<Rule>, ProgramError> {
    let count = decode_prefix_integer(reader)? - 1;
    if count > reader.remaining_bits() {
        return Err(ProgramError::Code("rule count beyond the message".to_string()));
    }
    let mut rules: Vec<Rule> = Vec::new();
    for r in 0..count as usize {
        let inputs_count = decode_prefix_integer(reader)? - 1;
        if inputs_count > reader.remaining_bits() {
            return Err(ProgramError::Code("rule input count beyond the message".to_string()));
        }
        let inputs = (0..inputs_count).map(|_| read_interface(reader)).collect::<Result<Vec<_>, _>>()?;
        let node_count = decode_prefix_integer(reader)? - 1;
        if node_count > reader.remaining_bits() {
            return Err(ProgramError::Code("rule node count beyond the message".to_string()));
        }
        let rule_inputs: Vec<usize> = rules.iter().map(|rule| rule.inputs.len()).collect();
        let code = NodeCode {
            operators: operators.len(),
            bases: bases.len(),
            slots: declarations.slots.len(),
            parameters: declarations.parameters,
            rule_inputs: &rule_inputs,
            params: inputs.len(),
        };
        let mut nodes = Vec::new();
        let mut interfaces: Vec<Interface> = Vec::new();
        for index in 0..node_count as usize {
            let node = decode_node(reader, index, &code, &interfaces)?;
            let scope = Scope { operators, bases, declarations, rules: &rules[..r], params: &inputs };
            interfaces.push(interface_of(index, &node, &interfaces, &scope)?);
            nodes.push(node);
        }
        let output = decode_fixed_index(reader, (node_count as usize).max(1))?;
        rules.push(Rule { name: format!("rule{r}"), inputs, nodes, output });
    }
    Ok(rules)
}

/// Fixed native operator codewords and independently decoded witnesses.
/// Only this immutable source is cached; candidate bodies never enter the cache.
/// The byte budget covers packed buffer capacities plus logical decoded numeric
/// buffers. Metadata/allocator overhead, caller-owned source buffers and one
/// constructor's transient operator decode are outside that explicit scope.
pub struct NativeOperatorCodec {
    declarations: Declarations,
    entries: Vec<NativeOperatorCodeword>,
    source_indices: BTreeMap<usize, usize>,
    /// Entries by (prefix length, hash of their first `prefix length` bits), the prefix being
    /// the whole codeword or its first [`CODEWORD_PREFIX_BITS`]; and the distinct lengths.
    prefixes: std::collections::HashMap<(u64, u64), Vec<usize>>,
    prefix_lengths: Vec<u64>,
    /// Entries by [`content_key`], for operators equal to a witnessed one but not its Arc.
    contents: std::collections::HashMap<u64, Vec<usize>>,
    stats: NativeOperatorCodecStats,
    encoded_hits: std::sync::atomic::AtomicU64,
    decoded_hits: std::sync::atomic::AtomicU64,
}
struct NativeOperatorCodeword {
    source: std::sync::Weak<Operator>,
    message: BitString,
    decoded: Arc<Operator>,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct NativeOperatorCodecStats {
    pub operators: usize,
    pub encoded_capacity_bytes: usize,
    pub decoded_numeric_bytes: usize,
    pub budget_bytes: usize,
    pub source_arcs_reused: usize,
    pub largest_transient_decoded_numeric_bytes: usize,
    /// Encode/decode/witness construction once; excluded from warm-cache timing.
    pub initialization_seconds: f64,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct NativeOperatorCodecUsage {
    pub encoded_native_operator_hits: u64,
    pub decoded_native_operator_hits: u64,
}
/// Bits of a codeword that key its content lookup when its operator index moved.
const CODEWORD_PREFIX_BITS: u64 = 256;

/// A hash of the next `bits` bits (at most what remains); nothing is consumed.
fn prefix_hash(reader: &BitReader<'_>, bits: u64) -> Result<u64, CodecError> {
    use std::hash::Hasher;
    let mut peek = reader.clone();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut left = bits;
    while left != 0 {
        let width = left.min(64) as u32;
        hasher.write_u64(peek.read_bits(width)?);
        left -= u64::from(width);
    }
    Ok(hasher.finish())
}

fn operator_numeric_bytes(operator: &Operator) -> Result<usize, ProgramError> {
    let (reals, mask) = match &operator.body {
        OperatorBody::Identity => (0, 0),
        OperatorBody::Diagonal { values, .. } => (values.len(), 0),
        OperatorBody::Dense {
            values, present, ..
        } => (values.len(), present.len()),
        OperatorBody::LowRank { left, right, .. } => (
            left.len()
                .checked_add(right.len())
                .ok_or_else(|| ProgramError::Code("native codec buffer size overflow".into()))?,
            0,
        ),
    };
    reals
        .checked_mul(8)
        .and_then(|n| n.checked_add(mask))
        .ok_or_else(|| ProgramError::Code("native codec buffer size overflow".into()))
}
fn witnessed_operator_equal(source: &Operator, decoded: &Operator) -> bool {
    // Full metadata as well as bits: signed zero must not permit substitution.
    source.name == decoded.name
        && source.provenance == decoded.provenance
        && same_codeword_content(source, decoded)
}
/// A lookup key of an operator's codeword content (interfaces, kind, precision, shapes and a few
/// sampled reals). Only a lookup: [`same_codeword_content`] licenses every reuse.
fn content_key(operator: &Operator) -> u64 {
    use std::hash::{Hash, Hasher};
    type State = std::collections::hash_map::DefaultHasher;
    // Eight evenly spaced reals in logical order, by index (no scan of the tensor).
    fn sample(len: usize, value: impl Fn(usize) -> f64, state: &mut State) {
        len.hash(state);
        for k in 0..len.min(8) {
            value(k * len / len.min(8)).to_bits().hash(state);
        }
    }
    fn matrix(a: &Array2<f64>, state: &mut State) {
        a.dim().hash(state);
        sample(a.len(), |i| a[[i / a.ncols(), i % a.ncols()]], state);
    }
    let mut state = State::new();
    operator.rows.hash(&mut state);
    operator.cols.hash(&mut state);
    match &operator.body {
        OperatorBody::Identity => 0_u8.hash(&mut state),
        OperatorBody::Diagonal { values, precision } => {
            1_u8.hash(&mut state);
            precision.hash(&mut state);
            sample(values.len(), |i| values[i], &mut state);
        }
        OperatorBody::Dense { values, present, precision } => {
            2_u8.hash(&mut state);
            precision.hash(&mut state);
            present.dim().hash(&mut state);
            matrix(values, &mut state);
        }
        OperatorBody::LowRank { left, right, precision } => {
            3_u8.hash(&mut state);
            precision.hash(&mut state);
            matrix(left, &mut state);
            matrix(right, &mut state);
        }
    }
    state.finish()
}
/// Whether two operators have the same codeword: interfaces and every body bit, signed zero
/// included. Names and provenance are not on the wire.
fn same_codeword_content(source: &Operator, decoded: &Operator) -> bool {
    let matrix = |a: &Array2<f64>, b: &Array2<f64>| {
        a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    source.rows == decoded.rows
        && source.cols == decoded.cols
        && match (&source.body, &decoded.body) {
            (OperatorBody::Identity, OperatorBody::Identity) => true,
            (
                OperatorBody::Diagonal {
                    values: a,
                    precision: ap,
                },
                OperatorBody::Diagonal {
                    values: b,
                    precision: bp,
                },
            ) => {
                ap == bp
                    && a.len() == b.len()
                    && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (
                OperatorBody::Dense {
                    values: a,
                    present: am,
                    precision: ap,
                },
                OperatorBody::Dense {
                    values: b,
                    present: bm,
                    precision: bp,
                },
            ) => ap == bp && am == bm && matrix(a, b),
            (
                OperatorBody::LowRank {
                    left: a,
                    right: ar,
                    precision: ap,
                },
                OperatorBody::LowRank {
                    left: b,
                    right: br,
                    precision: bp,
                },
            ) => ap == bp && matrix(a, b) && matrix(ar, br),
            _ => false,
        }
}
impl NativeOperatorCodec {
    pub fn new(source: &OperatorProgram, budget_bytes: usize) -> Result<Self, ProgramError> {
        let started = std::time::Instant::now();
        source.interfaces()?;
        let mut entries = Vec::new();
        let mut source_indices = BTreeMap::new();
        let mut prefixes: std::collections::HashMap<(u64, u64), Vec<usize>> = std::collections::HashMap::new();
        let mut contents: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
        let mut stats = NativeOperatorCodecStats {
            operators: source.operators.len(),
            encoded_capacity_bytes: 0,
            decoded_numeric_bytes: 0,
            budget_bytes,
            source_arcs_reused: 0,
            largest_transient_decoded_numeric_bytes: 0,
            initialization_seconds: 0.0,
        };
        for (index, operator) in source.operators.iter().enumerate() {
            let mut message = BitString::new();
            encode_operator(&mut message, operator)?;
            let capacity = message.shrink_storage();
            let mut reader = message.reader();
            let decoded = decode_operator(&mut reader, index as u64)?;
            reader.finish()?;
            let numeric = operator_numeric_bytes(&decoded)?;
            stats.encoded_capacity_bytes = stats
                .encoded_capacity_bytes
                .checked_add(capacity)
                .ok_or_else(|| ProgramError::Code("native codec buffer size overflow".into()))?;
            stats.decoded_numeric_bytes = stats
                .decoded_numeric_bytes
                .checked_add(numeric)
                .ok_or_else(|| ProgramError::Code("native codec buffer size overflow".into()))?;
            if stats
                .encoded_capacity_bytes
                .checked_add(stats.decoded_numeric_bytes)
                .is_none_or(|bytes| bytes > budget_bytes)
            {
                return Err(ProgramError::Code(format!(
                    "native codec exceeds declared {budget_bytes}-byte packed/numeric buffer budget"
                )));
            }
            stats.largest_transient_decoded_numeric_bytes =
                stats.largest_transient_decoded_numeric_bytes.max(numeric);
            let decoded = if witnessed_operator_equal(operator, &decoded) {
                stats.source_arcs_reused += 1;
                Arc::clone(operator)
            } else {
                decoded
            };
            source_indices.insert(Arc::as_ptr(operator) as usize, index);
            source_indices.insert(Arc::as_ptr(&decoded) as usize, index);
            let length = message.len_bits().min(CODEWORD_PREFIX_BITS);
            prefixes.entry((length, prefix_hash(&message.reader(), length)?)).or_default().push(index);
            contents.entry(content_key(&decoded)).or_default().push(index);
            entries.push(NativeOperatorCodeword {
                source: Arc::downgrade(operator),
                message,
                decoded,
            });
        }
        stats.initialization_seconds = started.elapsed().as_secs_f64();
        let mut prefix_lengths: Vec<u64> = prefixes.keys().map(|(length, _)| *length).collect();
        prefix_lengths.sort_unstable();
        prefix_lengths.dedup();
        Ok(Self {
            declarations: source.declarations.clone(),
            entries,
            source_indices,
            prefixes,
            prefix_lengths,
            contents,
            stats,
            encoded_hits: std::sync::atomic::AtomicU64::new(0),
            decoded_hits: std::sync::atomic::AtomicU64::new(0),
        })
    }
    pub fn stats(&self) -> NativeOperatorCodecStats {
        self.stats
    }
    pub fn usage(&self) -> NativeOperatorCodecUsage {
        NativeOperatorCodecUsage {
            encoded_native_operator_hits: self
                .encoded_hits
                .load(std::sync::atomic::Ordering::Relaxed),
            decoded_native_operator_hits: self
                .decoded_hits
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
    fn check_declarations(&self, declarations: &Declarations) -> Result<(), ProgramError> {
        if &self.declarations != declarations {
            return Err(ProgramError::Code(
                "native codec declarations differ from this message".into(),
            ));
        }
        Ok(())
    }
    /// The witnessed codeword of `operator`: by its Arc (the source's or the decoded one), else
    /// by exact codeword content (a renamed copy, say).
    fn codeword(&self, operator: &Arc<Operator>) -> Option<&BitString> {
        let by_arc = self
            .source_indices
            .get(&(Arc::as_ptr(operator) as usize))
            .map(|&index| &self.entries[index])
            .filter(|entry| {
                Arc::ptr_eq(&entry.decoded, operator)
                    || entry.source.upgrade().is_some_and(|source| Arc::ptr_eq(&source, operator))
            });
        let entry = by_arc.or_else(|| {
            self.contents
                .get(&content_key(operator))?
                .iter()
                .map(|&index| &self.entries[index])
                .find(|entry| same_codeword_content(&entry.decoded, operator))
        })?;
        self.encoded_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(&entry.message)
    }
    /// The operator whose witnessed codeword is next, consuming it; exactly what the ordinary
    /// decoder reads there. The entry at the same index is tried first and shared. Otherwise any
    /// entry whose prefix hash matches is tried: operator codewords are prefix-free, so a complete
    /// match is the codeword here, and its body is copied under this index's decoder label
    /// (a memory copy, not a decode; [`Self::codeword`] recognizes the copy by content).
    fn decode_at(&self, reader: &mut BitReader<'_>, index: u64) -> Option<Arc<Operator>> {
        // An index beyond usize has no entry.
        let at = usize::try_from(index).unwrap_or(usize::MAX);
        if let Some(entry) = self.entries.get(at).filter(|entry| reader.consume_exact_prefix(&entry.message)) {
            self.decoded_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(Arc::clone(&entry.decoded));
        }
        let remaining = reader.remaining_bits();
        let moved = self.prefix_lengths.iter().take_while(|length| **length <= remaining).find_map(|&length| {
            let key = (length, prefix_hash(reader, length).ok()?);
            self.prefixes.get(&key)?.iter().filter(|e| **e != at).find_map(|&e| {
                let entry = &self.entries[e];
                reader.consume_exact_prefix(&entry.message).then_some(entry)
            })
        })?;
        self.decoded_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(Arc::new(Operator {
            name: format!("decoded{index}"),
            provenance: Provenance::default(),
            ..(*moved.decoded).clone()
        }))
    }
}

fn encode_operator(out: &mut BitString, operator: &Operator) -> Result<(), ProgramError> {
    match &operator.body {
        OperatorBody::Identity => {
            encode_fixed_index(out, 0, OPERATOR_KINDS)?;
            write_interface(out, &operator.rows)?;
        }
        OperatorBody::Diagonal { precision, .. } => {
            encode_fixed_index(out, 4, OPERATOR_KINDS)?;
            write_interface(out, &operator.rows)?;
            write_lattice(out, &operator.present_reals(), *precision, &[])?;
        }
        OperatorBody::LowRank {
            left, precision, ..
        } => {
            encode_fixed_index(out, 2, OPERATOR_KINDS)?;
            write_interface(out, &operator.rows)?;
            write_interface(out, &operator.cols)?;
            encode_prefix_integer(out, left.ncols() as u64)?;
            write_lattice(out, &operator.present_reals(), *precision, &[])?;
        }
        OperatorBody::Dense {
            present, precision, ..
        } => {
            let indices = lattice_indices(&operator.present_reals(), *precision)?;
            let leads = ordered_leads_of(operator, &indices)?;
            encode_fixed_index(out, if leads.is_some() { 3 } else { 1 }, OPERATOR_KINDS)?;
            write_interface(out, &operator.rows)?;
            write_interface(out, &operator.cols)?;
            for row in present.outer_iter() {
                let kept: Vec<usize> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| **k)
                    .map(|(c, _)| c)
                    .collect();
                encode_subset(out, operator.cols.group_count(), &kept)?;
            }
            write_lattice_indices(out, &indices, *precision, leads.as_deref().unwrap_or(&[]))?;
        }
    }
    Ok(())
}

fn decode_operator(reader: &mut BitReader<'_>, index: u64) -> Result<Arc<Operator>, ProgramError> {
    decode_operator_with(reader, index, None)
}

/// An operator codeword's structure: everything before its lattice.
struct OperatorHeader {
    kind: usize,
    rows: Interface,
    cols: Option<Interface>,
    rank: usize,
    present: Option<Array2<bool>>,
    leads: Vec<usize>,
}

fn read_operator_header(reader: &mut BitReader<'_>, index: u64) -> Result<OperatorHeader, ProgramError> {
    let kind = decode_fixed_index(reader, OPERATOR_KINDS)?;
    let rows = read_interface(reader)?;
    let mut header = OperatorHeader { kind, rows, cols: None, rank: 0, present: None, leads: Vec::new() };
    if kind == 0 || kind == 4 {
        return Ok(header);
    }
    let cols = read_interface(reader)?;
    if kind == 2 {
        header.rank = decode_prefix_integer(reader)? as usize;
        header.cols = Some(cols);
        return Ok(header);
    }
    let mut present = Array2::from_elem((header.rows.group_count(), cols.group_count()), false);
    for r in 0..header.rows.group_count() {
        for c in decode_subset(reader, cols.group_count())? {
            present[[r, c]] = true;
        }
    }
    if kind == 3 {
        header.leads = row_leads(&header.rows, &cols, &present).ok_or_else(|| {
            ProgramError::Code(format!(
                "operator {index}: ordered rows without two nonempty rows"
            ))
        })?;
    }
    header.cols = Some(cols);
    header.present = Some(present);
    Ok(header)
}

/// Pass over one operator codeword, returning its lattice chunk starts ([`skip_lattice`]).
fn skip_operator(reader: &mut BitReader<'_>, index: u64) -> Result<Vec<u64>, ProgramError> {
    if read_operator_header(reader, index)?.kind == 0 {
        return Ok(Vec::new());
    }
    skip_lattice(reader)
}

/// [`decode_operator`]; with the lattice chunk starts of a reader bounded to exactly this
/// codeword, a large lattice decodes in parallel ([`read_lattice_chunked`]) to the same operator.
fn decode_operator_with(reader: &mut BitReader<'_>, index: u64, starts: Option<&[u64]>) -> Result<Arc<Operator>, ProgramError> {
    let header = read_operator_header(reader, index)?;
    let name = format!("decoded{index}");
    let OperatorHeader { kind, rows, cols, rank, present, leads } = header;
    if kind == 0 {
        return Ok(Arc::new(Operator::identity(name, rows)));
    }
    let (precision, reals) = match starts {
        Some(starts) if starts.len() > 1 => read_lattice_chunked(reader, &leads, starts)?,
        _ => read_lattice(reader, &leads)?,
    };
    if kind == 4 {
        if reals.len() != rows.width() {
            return Err(ProgramError::Code(format!(
                "operator {index}: a diagonal of {} on an interface of {}",
                reals.len(),
                rows.width()
            )));
        }
        return Ok(Arc::new(Operator {
            name,
            cols: rows.clone(),
            rows,
            body: OperatorBody::Diagonal {
                values: Array1::from(reals),
                precision,
            },
            provenance: Provenance::default(),
        }));
    }
    let cols = cols.ok_or_else(|| ProgramError::Code(format!("operator {index}: no column interface")))?;
    if kind == 2 {
        let split = rows.width() * rank;
        if reals.len() != split + rank * cols.width() {
            return Err(ProgramError::Code(format!(
                "operator {index}: {} reals for rank {rank}",
                reals.len()
            )));
        }
        let left = Array2::from_shape_vec((rows.width(), rank), reals[..split].to_vec())
            .map_err(|error| ProgramError::Code(error.to_string()))?;
        let right = Array2::from_shape_vec((rank, cols.width()), reals[split..].to_vec())
            .map_err(|error| ProgramError::Code(error.to_string()))?;
        return Ok(Arc::new(Operator {
            name,
            rows,
            cols,
            body: OperatorBody::LowRank {
                left,
                right,
                precision,
            },
            provenance: Provenance::default(),
        }));
    }
    let present = present.ok_or_else(|| ProgramError::Code(format!("operator {index}: no present blocks")))?;
    let span = gam_gpu::trace::host_span("codec.operator.assemble");
    let mut values = Array2::<f64>::zeros((rows.width(), cols.width()));
    let mut next = reals.iter();
    for ((r, c), &keep) in present.indexed_iter() {
        if keep {
            for value in values
                .slice_mut(s![rows.range(r), cols.range(c)])
                .iter_mut()
            {
                *value = *next.next().ok_or_else(|| {
                    ProgramError::Code(format!("operator {index} has too few reals"))
                })?;
            }
        }
    }
    if next.next().is_some() {
        return Err(ProgramError::Code(format!(
            "operator {index} has too many reals"
        )));
    }
    drop(span);
    Ok(Arc::new(Operator {
        name,
        rows,
        cols,
        body: OperatorBody::Dense {
            values: values.into(),
            present,
            precision,
        },
        provenance: Provenance::default(),
    }))
}
impl OperatorProgram {
    fn top_code<'a>(&self, rule_inputs: &'a [usize]) -> NodeCode<'a> {
        NodeCode {
            operators: self.operators.len(),
            bases: self.bases.len(),
            slots: self.declarations.slots.len(),
            parameters: self.declarations.parameters,
            rule_inputs,
            params: 0,
        }
    }

    /// The program's message (module note, "The code").
    pub fn encode(&self) -> Result<BitString, ProgramError> { self.encode_using(None) }

    /// The ordinary standalone message, reusing fixed witnessed source chunks.
    pub fn encode_with_native_codec(&self, codec: &NativeOperatorCodec) -> Result<BitString, ProgramError> {
        codec.check_declarations(&self.declarations)?;
        self.encode_using(Some(codec))
    }
    fn encode_using(&self, codec: Option<&NativeOperatorCodec>) -> Result<BitString, ProgramError> {
        let interfaces = self.interfaces()?;
        let mut out = BitString::new();
        encode_prefix_integer(&mut out, self.bases.len() as u64 + 1)?;
        encode_prefix_integer(&mut out, self.operators.len() as u64 + 1)?;
        encode_prefix_integer(&mut out, self.nodes.len() as u64)?;
        let domains = self.declarations.domains.len();
        for basis in &self.bases {
            let Basis::Indicator { domain } = basis;
            // Reserved kind 1 is the unsupported legacy Characters basis.
            encode_fixed_index(&mut out, 0, 2)?;
            encode_fixed_index(&mut out, *domain, domains)?;
        }
        // Each operator's codeword depends on it alone: code them in parallel, append in order.
        let span = gam_gpu::trace::host_span("codec.encode.operators");
        let parts = self
            .operators
            .par_iter()
            .map(|operator| match codec.and_then(|cache| cache.codeword(operator)) {
                Some(message) => Ok(std::borrow::Cow::Borrowed(message)),
                None => {
                    let mut part = BitString::new();
                    encode_operator(&mut part, operator)?;
                    Ok(std::borrow::Cow::Owned(part))
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        drop(span);
        gam_gpu::trace::within_host("codec.encode.append", || {
            for part in &parts {
                out.append(part);
            }
        });
        drop(parts);
        encode_rules(&mut out, self)?;
        let rule_inputs: Vec<usize> = self.rules.iter().map(|r| r.inputs.len()).collect();
        let code = self.top_code(&rule_inputs);
        for (index, node) in self.nodes.iter().enumerate() {
            encode_node(&mut out, node, index, &code, &interfaces)?;
        }
        encode_fixed_index(&mut out, self.output, self.nodes.len())?;
        Ok(out)
    }

    /// Decode one bounded borrowed program message; trailing bits remain an error.
    pub(crate) fn decode_reader(reader: &mut BitReader<'_>, declarations: &Declarations, codec: Option<&NativeOperatorCodec>) -> Result<Self, ProgramError> {
        if let Some(codec) = codec { codec.check_declarations(declarations)?; }
        let basis_count = decode_prefix_integer(reader)? - 1;
        let operator_count = decode_prefix_integer(reader)? - 1;
        let node_count = decode_prefix_integer(reader)? as usize;
        let bound = reader.remaining_bits();
        if basis_count > bound || operator_count > bound || node_count as u64 > bound {
            return Err(ProgramError::Code("counts beyond the message".to_string()));
        }
        let domains = declarations.domains.len();
        let mut bases = Vec::new();
        for _ in 0..basis_count {
            if decode_fixed_index(reader, 2)? != 0 {
                return Err(ProgramError::Code("unsupported legacy Characters basis (kind 1)".into()));
            }
            let domain = decode_fixed_index(reader, domains)?;
            bases.push(Basis::Indicator { domain });
        }
        // Witnessed codewords are matched in order; every other codeword's extent is found by a
        // pass that refuses exactly what decoding refuses, then those operators decode in
        // parallel, each required to consume exactly its extent: the sequential result.
        let mut slots = Vec::new();
        let mut pending = Vec::new();
        for index in 0..operator_count {
            if let Some(operator) = codec.and_then(|cache| gam_gpu::trace::within_host("codec.decode.witnessed", || cache.decode_at(reader, index))) {
                slots.push(Some(operator));
                continue;
            }
            let start = reader.position();
            let starts = gam_gpu::trace::within_host("codec.decode.skip", || skip_operator(reader, index))?;
            pending.push((slots.len(), start, reader.position(), starts));
            slots.push(None);
        }
        let message: &BitReader<'_> = reader;
        let span = gam_gpu::trace::host_span("codec.decode.operators");
        let decoded = pending
            .par_iter()
            .map(|(index, start, end, starts)| {
                let mut part = message.window(*start, *end)?;
                let operator = decode_operator_with(&mut part, *index as u64, Some(starts))?;
                part.finish()?;
                Ok((*index, operator))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        drop(span);
        for (index, operator) in decoded {
            slots[index] = Some(operator);
        }
        let operators = slots
            .into_iter()
            .map(|operator| operator.ok_or_else(|| ProgramError::Code("an undecoded operator".into())))
            .collect::<Result<Vec<_>, _>>()?;
        let rules = decode_rules(reader, &operators, &bases, declarations)?;
        let rule_inputs: Vec<usize> = rules.iter().map(|r| r.inputs.len()).collect();
        let code = NodeCode {
            operators: operators.len(),
            bases: bases.len(),
            slots: declarations.slots.len(),
            parameters: declarations.parameters,
            rule_inputs: &rule_inputs,
            params: 0,
        };
        let mut nodes = Vec::new();
        let mut interfaces: Vec<Interface> = Vec::new();
        for index in 0..node_count {
            let node = decode_node(reader, index, &code, &interfaces)?;
            interfaces.push(interface_of(index, &node, &interfaces, &Scope {
                operators: &operators,
                bases: &bases,
                declarations,
                rules: &rules,
                params: &[],
            })?);
            nodes.push(node);
        }
        let output = decode_fixed_index(reader, node_count)?;
        if reader.remaining_bits() != 0 {
            return Err(ProgramError::Code(format!("{} bits left after the program", reader.remaining_bits())));
        }
        let program = Self { declarations: declarations.clone(), bases, operators, rules, nodes, output };
        program.interfaces()?;
        Ok(program)
    }
}

/// The finest lattice on which every value of `values` is exact with an index within `2^53`,
/// or, when no such lattice exists, the finest whose largest index stays within `2^53`.
pub fn exact_precision(values: impl IntoIterator<Item = f64>) -> Result<DeclaredPrecision, ProgramError> {
    let mut finest = i32::MIN;
    let mut largest = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return Err(ProgramError::Code(format!("a non-finite real {value}")));
        }
        if value == 0.0 {
            continue;
        }
        largest = largest.max(value.abs());
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = bits & ((1u64 << 52) - 1);
        let trailing = if mantissa == 0 { 52 } else { mantissa.trailing_zeros() as i32 };
        let unbiased = if exponent == 0 { -1074 } else { exponent - 1075 };
        finest = finest.max(-(unbiased + trailing.min(52)));
    }
    if finest == i32::MIN {
        return DeclaredPrecision::new(0).map_err(ProgramError::Code);
    }
    // A subnormal's lattice is finer than any declarable one: the finest declarable holds it to
    // within half a step.
    Ok(DeclaredPrecision::new(finest.min(-(f64::MIN_EXP - 1))).map_err(ProgramError::Code)?.within_range(largest))
}

#[cfg(test)]
#[path = "operator_program_codec_tests.rs"]
mod codec_tests;
