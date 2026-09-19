//! Implementation gauges of native tensors (#2951, Doc B §10.4).
//!
//! An implementation gauge is a group `G` acting on a block's native tensors,
//! `θ ↦ g·θ`, with `f_{g·θ}(x) = f_θ(x)` at every input `x`. Moving along an
//! orbit changes no intervention, so a gauge coordinate is not a parameter-family
//! label, a computational-state coordinate or a structured-edit coordinate. A gauge
//! change here acts only through its own family's `apply`, and no structured edit
//! is read as one.
//!
//! # Families
//!
//! * **Linear pass-through, `GL(r)`.** A read `A ∈ ℝ^{r×d_in}` followed by a write
//!   `B ∈ ℝ^{d_out×r}` with no primitive between them executes `B A`. Examples are
//!   an attention head's value and output projections, or a low-rank factor
//!   `P = U Vᵀ`. `(A, B) ↦ (S A, B S⁻¹)` leaves `B A` unchanged for every
//!   `S ∈ GL(r)`. The orbit's tangent is `X ↦ (X A, −B X)`, whose kernel is
//!   `null(B) ⊗ leftnull(A)`, so the orbit has dimension
//!   `r² − (r − rank A)(r − rank B)`. At full rank the orbit is the whole fibre of
//!   `(A, B) ↦ B A`.
//! * **Hidden units across an elementwise primitive,** `y = W₂ σ(W₁ x + b₁)`. A
//!   joint permutation of the units is always a gauge. A per-unit scale with
//!   `σ(s t) = s σ(t)` for every `t` acts as
//!   `(row_i, b_i, col_i) ↦ (s_i row_i, s_i b_i, col_i / s_i)`. Every `s > 0`
//!   qualifies for ReLU; for the exact GELU and SiLU, which are not homogeneous,
//!   only `s = 1` does.
//! * **SwiGLU hidden units,** `y = W_d [s(W_g x) ⊙ W_u x]`. The block is bilinear
//!   in the up read, so `(up_i, col_i) ↦ (s_i up_i, col_i / s_i)` is a gauge for
//!   every `s_i ≠ 0`. The gate rows only permute.
//! * **Norm gains.** A norm output `w ⊙ n(h) + β` read only by linear reads `W_k`
//!   is unchanged under `(w, β, W_k) ↦ (D w, D β, W_k D⁻¹)` for every invertible
//!   diagonal `D`, because `W_k (diag(w) n + β) = W_k D⁻¹ (diag(D w) n + D β)`. The
//!   normalizer `n(h)` is formed before the gain and never sees `D`.
//! * **Rotary query/key.** A score `σ α² qᵀ R_Δ k`, with `q = W_Q x`, `k = W_K x` and
//!   the source's rotations `R_Δ = R_1^Δ`, is unchanged under
//!   `(W_K,g, W_Q,h) ↦ (S W_K,g, S⁻ᵀ W_Q,h)` for every query head `h` of key/value head
//!   `g` iff `S` commutes with `R_1`.
//!   - For frequencies in `(0, π)`, the eigenvalues `e^{±iθ}` of distinct frequencies
//!     and the pass-through eigenvalue `1` are pairwise distinct. So `S` is block
//!     diagonal over the planes of one frequency and the pass-through coordinates.
//!   - On `m` planes of one frequency, `R_1 = cos θ I + sin θ J` with `sin θ ≠ 0`, so `S`
//!     commutes with `J`. That is `GL(m, ℂ)` on the complex coordinates `x_a + i x_b`,
//!     of real dimension `2m²`.
//!   - The pass-through coordinates carry `GL(n_pass)`.
//!
//!   A per-head norm between the projection and the rotation (Qwen3's
//!   `q_norm`/`k_norm`) does not execute this program. `RotaryQueryKey::new` refuses such
//!   a block, and its family is the next one.
//! * **Rotary query/key behind a per-head RMS norm** ([`NormedRotaryQueryKey`]).
//!   - **The block.** Each head reads `q̂ = D_w ν(u) u`, with `u = W_Q x + b_Q` and
//!     `ν(u) = (‖u‖²/d + ε)^{−1/2}`. Likewise each key head reads `k̂ = D_v ν(u_k) u_k`.
//!     Every query head shares the gains `w`, and every key head shares `v`.
//!   - **The candidate change.** For each key/value head `g`: `u ↦ A_g u` on each of its
//!     query heads and `u_k ↦ B_g u_k` on its key, with new gains `w′`, `v′`.
//!
//!   The derivation:
//!   - **The head maps are orthogonal.** With `ε > 0`, `ν` sees `‖u‖`. When a head's
//!     weight rows have full row rank, every `u` occurs along its whole ray. So the
//!     scores survive only if `A_g` and `B_g` are orthogonal. Then `q̂ ↦ T_g q̂` with
//!     `T_g = D_{w′} A_g D_w⁻¹`, and `k̂ ↦ U_g k̂`.
//!   - **`T_g` is in the rotary commutant.** As for the family above, `Δ = 0` forces
//!     `U_g = T_g⁻ᵀ`, and the span of the `R_Δ` forces `T_g` into the rotary commutant.
//!   - **Each plane.** With distinct frequencies and no pass-through, `T_g = ρ R_θ` on
//!     each plane. `A_g = D_{w′}⁻¹ T_g D_w` is orthogonal iff
//!     `R_θ D_w² R_θᵀ = ρ⁻² D_{w′}²`. The left side is diagonal iff `|w_a| = |w_b|` or
//!     `θ ∈ (π/2)ℤ`, and then `|w′| = ρ|w|`, with the two entries swapped when `θ/(π/2)`
//!     is odd. The keys satisfy the same condition with `v` and `1/ρ`.
//!   - **What the heads share.** The gains are shared, so every head has the same `ρ`.
//!     Unless the plane is coincident, every head also has the same parity of `θ`.
//!
//!   So on each plane the group has:
//!   - a shared positive scale `ρ`, with the query gains multiplied by `ρ` and the key
//!     gains divided by it;
//!   - per key/value head, a head map `ρ R^t`, with one parity of `t` for every head;
//!     the odd parity swaps both gain pairs;
//!   - the signs of the four gain entries, each carried by its rows in every head;
//!   - on a coincident plane (`|w_a| = |w_b|` and `|v_a| = |v_b|`, compared exactly), a
//!     rotation `R_θ` per key/value head, which absorbs the parity.
//!
//!   The family is refused, typed, wherever the derivation does not hold:
//!   - a zero gain;
//!   - `ε ≤ 0`;
//!   - a head that does not resolve full row rank;
//!   - pass-through coordinates;
//!   - two planes that share a frequency.
//!
//!   [`QueryKeyGauge::new`] reads whichever of the two families a native block carries.
//! * **Residual-stream basis.** `h ↦ Q h` acts on every write (embedding columns,
//!   block writes and their biases) and `W ↦ W Q⁻¹` on every read.
//!   - A stream read only linearly admits `GL(d)`.
//!   - An RMSNorm read needs `ν(Q h) = ν(h)` for every `h`, which holds iff `Q` is
//!     orthogonal. Its gain is carried by the read: `W_k ↦ W_k diag(w) Qᵀ diag(w)⁻¹`.
//!   - A LayerNorm read also needs `Q` to commute with the centring `I − 1 1ᵀ/d`,
//!     so `Q 1 = ±1`, giving the group `O(1) × O(d − 1)`.
//!
//!   The stream's group is the smallest one its reads allow.
//!
//! # Exact null coordinates
//!
//! Some coordinates move the executed function by exactly nothing, and not through
//! a group:
//! - a unit whose write column is zero reads with any row and bias;
//! - a unit whose read row is zero, and whose activation is zero on a neighbourhood
//!   of its bias, writes with any column.
//!
//! These are [`GaugeFamily::null_coordinates`]. A unit or coordinate counted there
//! is not also counted in the orbit, since its null space already contains the
//! orbit's tangent. Zero tests are on the registered bits, so they carry no
//! tolerance.
//!
//! # Quotients
//!
//! * **Codes.** A code for a block needs at most
//!   [`GaugeFamily::real_coordinates_at_most`] real coordinates: the parameter
//!   count minus the proven orbit dimension and the exact null coordinates.
//!   - The ranks behind an orbit dimension are resolved against the SVD's
//!     backward-error band ([`factor_singular_band`]). By Weyl, a computed singular
//!     value above the band proves a nonzero true one, so the resolved rank is a
//!     lower bound, and the orbit dimension is increasing in both ranks.
//!   - The orbit dimension is reported as an interval, and only its resolved side
//!     is charged.
//!   - The count bounds what a code must carry. It never claims that no further
//!     invariance exists.
//! * **Intervention sets.** Two settings of a linear pass-through execute the same
//!   map iff `B′A′ = B A`, and at full rank that holds iff they lie on one `GL(r)`
//!   orbit. [`LinearPassthrough::operator_difference`] reports
//!   `sup_x ‖(B′A′ − B A) x‖/‖x‖` with its numerical error.
//!   [`quotient_intervention_set`] merges the settings it does not prove distinct,
//!   and reports the certified gap of each class.
//!
//! # Masks (P1)
//!
//! A detected family is declared to P1's owner as the group with its commutant
//! ([`GaugeFamily::declared_gauge`], [`RotaryQueryKey::declared_gauge`]).
//! `operators::classify_under` then says whether a fixed internal mask is an intrinsic
//! intervention under that implementation gauge. Every witness it returns lies in the
//! declared group, and this module's `apply` admits it.
//! * `GL(r)` and `O(r)` act irreducibly, so their commutant is the scalars. A
//!   pass-through and an RMSNorm stream are declared as `Blocks([r])`.
//! * Scales with unit relabellings have commutant `cI`, declared as `ScaledPermutations`.
//!   This covers ReLU's positive scales and the SwiGLU up/down nonzero scales; the
//!   witnesses (scale 2, a transposition) lie in both groups.
//! * Coordinate scales alone (norm gains) have the diagonal commutant: `Blocks([1; d])`.
//! * Relabellings alone (GELU, SiLU) have commutant `span{I, 1 1ᵀ}`, declared as
//!   `UnitPermutations`.
//! * The LayerNorm stream group `O(1) × O(d − 1)` is also declared as `UnitPermutations`.
//!   It preserves `span{1}` and its complement and acts irreducibly on the complement,
//!   so its commutant is `span{1 1ᵀ/d, I − 1 1ᵀ/d} = span{I, 1 1ᵀ}`. Every permutation
//!   witness is orthogonal and fixes `1`, so it lies in the group.
//! * The rotary commutant is `RotaryCommutant`, built from the plane groups of equal
//!   frequency.
//! * The normed family declares the same `RotaryCommutant`. On each plane its head maps
//!   commute with exactly `αI + βJ`, which is the rotary commutant's commutant, and the
//!   owner's one-plane witnesses (scale 2 on a plane, `J` on a plane) are its changes.
//!
//! Every operator acts through its factors or matrix-free. Nothing here forms a
//! `d_out × d_in` product or a `d × d` stream basis change.

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd, fast_ab, fast_abt, fast_av};
use gam_linalg::roundoff::{accumulation_growth, factor_singular_band};
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};

use super::attention::{AttentionGeometry, NativeAttention, RotaryEmbedding};
use super::operators::{DeclaredGauge, GaugeBlocks, OperatorRefusal, RotaryGroups};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

/// Why a gauge detection, gauge change or comparison was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum GaugeRefusal {
    /// A dimension disagrees with the object it is combined with.
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// An input carries a non-finite entry.
    NonFinite { what: &'static str },
    /// A tensor has no rows or no columns.
    EmptyTensor { what: &'static str },
    /// A gauge change names an element outside the family's group. Examples are a
    /// nonpositive ReLU scale, a GELU or SiLU scale other than one, and a zero
    /// scale.
    NotInGroup {
        what: &'static str,
        index: usize,
        value: f64,
    },
    /// A unit relabelling is not a bijection of `0..length`.
    NotAPermutation { length: usize },
    /// A gauge change has fewer singular values above its SVD's rounding band than
    /// its order.
    SingularGaugeChange { resolved: usize, order: usize },
    /// A residual-stream carry of a normed read divides by a gain entry that is
    /// zero.
    ZeroGain { coordinate: usize },
    /// The linear-algebra backend failed to decompose a matrix.
    Decomposition { what: &'static str, detail: String },
    /// A reported number was refused by its evidence constructor.
    Evidence(EvidenceStatusError),
    /// A rotary frequency outside `(0, π)`. There two frequencies, or a plane and the
    /// pass-through coordinates, may share a rotation eigenvalue, so the commutant is
    /// not the block structure detected here.
    FrequencyOutsideHalfTurn { plane: usize, frequency: f64 },
    /// A zero or non-finite `attention_scaling` scales every score by zero or by
    /// nothing finite, so it determines no query/key gauge.
    ZeroRotaryScaling,
    /// [`RotaryQueryKey::new`] on a native attention block with a per-head query/key norm
    /// between its projections and its rotation. That block's gauge is
    /// [`NormedRotaryQueryKey`]'s, a proper subgroup of the rotary commutant.
    QueryKeyNorm,
    /// [`NormedRotaryQueryKey::new`] on a block without a query/key norm. Its gauge is
    /// [`RotaryQueryKey`]'s.
    NoQueryKeyNorm,
    /// A query/key norm whose `ε` is not positive and finite. At `ε = 0` the norm forgets
    /// every scale of its input, so a projection scale is a gauge that the normed
    /// derivation excludes.
    QueryKeyNormEpsilon { epsilon: f64 },
    /// A zero query or key norm gain. Its coordinate is zero after the norm whatever the
    /// projection reads, so its rows are free, which the normed derivation excludes.
    QueryKeyNormZeroGain { side: QueryKeySide, coordinate: usize },
    /// A projection head whose weight rows do not resolve full row rank. The normed
    /// derivation needs every pre-norm vector to occur along every ray, which is what
    /// forces an orthogonal head map.
    QueryKeyNormHeadRank {
        side: QueryKeySide,
        head: usize,
        resolved: usize,
        order: usize,
    },
    /// Head coordinates past the rotary dimension behind a query/key norm. Their group
    /// permutes and rotates classes of equal `|w_i v_i|`, which is not derived here.
    QueryKeyNormPassThrough { coordinates: usize },
    /// Two planes of one frequency behind a query/key norm. Their `GL(m, ℂ)` meets the
    /// norm through pairwise gain coincidences across planes, which are not derived here.
    QueryKeyNormSharedFrequency { first: usize, second: usize },
    /// The P1 owner refused a declared gauge.
    Operator(OperatorRefusal),
}

impl From<EvidenceStatusError> for GaugeRefusal {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl From<OperatorRefusal> for GaugeRefusal {
    fn from(error: OperatorRefusal) -> Self {
        Self::Operator(error)
    }
}

/// The continuous part of a gauge group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuousGauge {
    /// `GL(order)`.
    GeneralLinear { order: usize },
    /// `(ℝ_{>0})^order`: positive per-unit scales.
    PositiveDiagonal { order: usize },
    /// `(ℝ^×)^order`: nonzero per-unit scales.
    NonzeroDiagonal { order: usize },
    /// `O(order)`.
    Orthogonal { order: usize },
    /// `O(1) × O(order − 1)`: the orthogonal maps with `Q 1 = ±1`.
    OrthogonalFixingOnes { order: usize },
    /// `copies` independent copies, one per key/value head, of a rotary commutant of
    /// real dimension `per_copy`.
    RotaryCommutant { copies: usize, per_copy: usize },
    /// Behind a per-head query/key norm: one positive gain scale per rotated plane, shared
    /// by every head, and `plane_rotations` rotations, one per key/value head and
    /// coincident plane.
    NormedRotary { plane_scales: usize, plane_rotations: usize },
    /// No continuous gauge.
    Trivial,
}

impl ContinuousGauge {
    /// The group's dimension as a manifold.
    pub fn dimension(&self) -> usize {
        match *self {
            Self::GeneralLinear { order } => order * order,
            Self::PositiveDiagonal { order } | Self::NonzeroDiagonal { order } => order,
            Self::Orthogonal { order } => order * order.saturating_sub(1) / 2,
            Self::OrthogonalFixingOnes { order } => {
                let rest = order.saturating_sub(1);
                rest * rest.saturating_sub(1) / 2
            }
            Self::RotaryCommutant { copies, per_copy } => copies * per_copy,
            Self::NormedRotary {
                plane_scales,
                plane_rotations,
            } => plane_scales + plane_rotations,
            Self::Trivial => 0,
        }
    }
}

/// The discrete part of a gauge group that is not already inside its continuous
/// part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscreteGauge {
    /// Nothing beyond the continuous part.
    None,
    /// Joint relabellings of `units` hidden units.
    UnitPermutations { units: usize },
    /// Behind a per-head query/key norm, the components over each plane.
    /// - Every plane: the signs of its four gain entries, `2⁴`.
    /// - A plane whose gains are not coincident, in addition: the parity of the shared
    ///   quarter turn, and one sign per key/value head, `2^{n_kv + 1}`.
    ///
    /// The order is `2^{(n_kv + 5)·generic_planes + 4·coincident_planes}`.
    NormedRotary {
        generic_planes: usize,
        coincident_planes: usize,
        key_value_heads: usize,
    },
}

/// Which native structure a detected gauge family acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaugeFamilyKind {
    LinearPassthrough,
    HiddenUnits(GaussianActivation),
    SwigluUnits,
    NormGain,
    RotaryQueryKey,
    NormedRotaryQueryKey,
}

/// A dimension resolved from below at the registered tensors, with the largest
/// value the tensors' shapes allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimensionInterval {
    pub resolved: usize,
    pub at_most: usize,
}

impl DimensionInterval {
    pub fn is_exact(&self) -> bool {
        self.resolved == self.at_most
    }
}

/// A gauge family detected on one block's native tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaugeFamily {
    pub kind: GaugeFamilyKind,
    pub continuous: ContinuousGauge,
    pub discrete: DiscreteGauge,
    /// The real coordinates of the tensors the family acts on.
    pub parameter_coordinates: usize,
    /// The dimension of the continuous orbit through the registered tensors, over
    /// the units or coordinates not counted in `null_coordinates`.
    pub orbit_dimension: DimensionInterval,
    /// Coordinates that move the executed function by exactly nothing.
    pub null_coordinates: usize,
}

impl GaugeFamily {
    /// The code quotient: at most this many real coordinates carry the executed
    /// function of the block.
    pub fn real_coordinates_at_most(&self) -> usize {
        self.parameter_coordinates - self.orbit_dimension.resolved - self.null_coordinates
    }

    /// This family's gauge as declared to `operators::classify_under`, or `None` when no
    /// declared group has its commutant (see the module's Masks section).
    pub fn declared_gauge(&self) -> Result<Option<DeclaredGauge>, GaugeRefusal> {
        declared_gauge(self.continuous, self.discrete)
    }
}

/// The declared P1 group with this gauge's commutant.
///
/// Positive coordinate scales with no relabelling have the diagonal commutant, but their
/// classifier witness (the reflection) is not a positive scale. No detector here produces
/// that group, so it is left undeclared rather than declared with a witness outside it.
/// The rotary commutant needs its plane groups and is declared by
/// [`RotaryQueryKey::declared_gauge`], and the normed family by
/// [`NormedRotaryQueryKey::declared_gauge`].
fn declared_gauge(continuous: ContinuousGauge, discrete: DiscreteGauge) -> Result<Option<DeclaredGauge>, GaugeRefusal> {
    Ok(match (continuous, discrete) {
        (ContinuousGauge::NormedRotary { .. }, _) | (_, DiscreteGauge::NormedRotary { .. }) => None,
        (ContinuousGauge::GeneralLinear { order }, _) | (ContinuousGauge::Orthogonal { order }, _) => {
            Some(DeclaredGauge::Blocks(GaugeBlocks::new(&[order])?))
        }
        (ContinuousGauge::PositiveDiagonal { order }, DiscreteGauge::UnitPermutations { .. })
        | (ContinuousGauge::NonzeroDiagonal { order }, DiscreteGauge::UnitPermutations { .. }) => {
            Some(DeclaredGauge::ScaledPermutations { units: order })
        }
        (ContinuousGauge::NonzeroDiagonal { order }, DiscreteGauge::None) => {
            Some(DeclaredGauge::Blocks(GaugeBlocks::new(&vec![1; order])?))
        }
        (ContinuousGauge::Trivial, DiscreteGauge::UnitPermutations { units }) => Some(DeclaredGauge::UnitPermutations { units }),
        (ContinuousGauge::OrthogonalFixingOnes { order }, _) => Some(DeclaredGauge::UnitPermutations { units: order }),
        (ContinuousGauge::PositiveDiagonal { .. }, DiscreteGauge::None)
        | (ContinuousGauge::RotaryCommutant { .. }, _)
        | (ContinuousGauge::Trivial, DiscreteGauge::None) => None,
    })
}

/// The region of every nonzero input of a `width`-dimensional map. Over it an
/// operator difference `Δ` is reported as `sup_x ‖Δ x‖/‖x‖`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllInputs {
    pub width: usize,
}

/// A read `A` (`r × d_in`) followed by a write `B` (`d_out × r`) with no primitive
/// between them, executing `x ↦ B A x`.
#[derive(Clone, Debug)]
pub struct LinearPassthrough {
    read: Array2<f64>,
    write: Array2<f64>,
}

impl LinearPassthrough {
    pub fn new(read: Array2<f64>, write: Array2<f64>) -> Result<Self, GaugeRefusal> {
        require_nonempty("pass-through read", read.view())?;
        require_nonempty("pass-through write", write.view())?;
        check_len("pass-through write columns", read.nrows(), write.ncols())?;
        require_finite_matrix("pass-through read", read.view())?;
        require_finite_matrix("pass-through write", write.view())?;
        Ok(Self { read, write })
    }

    /// The internal dimension `r`.
    pub fn order(&self) -> usize {
        self.read.nrows()
    }

    pub fn read(&self) -> ArrayView2<'_, f64> {
        self.read.view()
    }

    pub fn write(&self) -> ArrayView2<'_, f64> {
        self.write.view()
    }

    /// `B (A x)`, through the `r` internal coordinates.
    pub fn apply_to(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, GaugeRefusal> {
        check_len("pass-through input", self.read.ncols(), x.len())?;
        Ok(fast_av(&self.write, &fast_av(&self.read, &x)))
    }

    /// The `GL(r)` family through these tensors. Its orbit dimension is
    /// `r² − (r − rank A)(r − rank B)`, resolved from the ranks above each factor's
    /// SVD band; its upper side takes the largest ranks the shapes allow.
    pub fn family(&self) -> Result<GaugeFamily, GaugeRefusal> {
        let r = self.order();
        let (d_in, d_out) = (self.read.ncols(), self.write.nrows());
        let read_rank = resolved_rank("pass-through read", self.read.view())?;
        let write_rank = resolved_rank("pass-through write", self.write.view())?;
        let orbit = |a_rank: usize, b_rank: usize| r * r - (r - a_rank) * (r - b_rank);
        Ok(GaugeFamily {
            kind: GaugeFamilyKind::LinearPassthrough,
            continuous: ContinuousGauge::GeneralLinear { order: r },
            discrete: DiscreteGauge::None,
            parameter_coordinates: r * (d_in + d_out),
            orbit_dimension: DimensionInterval {
                resolved: orbit(read_rank, write_rank),
                at_most: orbit(r.min(d_in), r.min(d_out)),
            },
            null_coordinates: 0,
        })
    }

    /// `(A, B) ↦ (S A, B S⁻¹)`. It refuses an `S` whose SVD resolves fewer than `r`
    /// singular values.
    pub fn apply(&self, gauge_change: ArrayView2<'_, f64>) -> Result<Self, GaugeRefusal> {
        let r = self.order();
        check_len("gauge change rows", r, gauge_change.nrows())?;
        check_len("gauge change columns", r, gauge_change.ncols())?;
        require_finite_matrix("gauge change", gauge_change)?;
        let inverse = invert_gauge_change(gauge_change)?;
        Ok(Self {
            read: fast_ab(&gauge_change, &self.read),
            write: fast_ab(&self.write, &inverse),
        })
    }

    /// `sup_x ‖(B′A′ − B A) x‖/‖x‖ = ‖B′A′ − B A‖₂` for `other = (A′, B′)`, as an
    /// exact algebraic extremum with its numerical error and a witness input.
    ///
    /// The difference is `L Rᵀ` with `L = [B′ | −B]` and `R = [A′ᵀ | Aᵀ]`, each with
    /// `k = r′ + r` columns. A factor with at least `k` rows is replaced by its thin
    /// QR `Q T`. Householder QR is backward stable: `T` is the exact triangle of
    /// `L + δL` for an exactly orthonormal `Q`, with `‖δL‖₂` inside
    /// `factor_singular_band(rows, k, ‖L‖_F)`. So `‖T_L T_Rᵀ‖₂` is the norm of the
    /// perturbed difference. A factor with fewer rows than `k` is used as it is,
    /// with no error.
    ///
    /// The numerical error, to first order, is the sum of:
    /// - `‖δL‖ ‖R‖_F + ‖L‖_F ‖δR‖`;
    /// - the rounding of the `k`-term core product, `γ_k ‖T_L‖_F ‖T_R‖_F`;
    /// - the core SVD's own band.
    ///
    /// The witness is the input along the top right singular vector.
    pub fn operator_difference(&self, other: &Self) -> Result<EvidenceStatus<Array1<f64>, AllInputs>, GaugeRefusal> {
        check_len("compared input width", self.read.ncols(), other.read.ncols())?;
        check_len("compared output width", self.write.nrows(), other.write.nrows())?;
        let negated = self.write.mapv(|entry| -entry);
        let left = concatenate(Axis(1), &[other.write.view(), negated.view()]).map_err(shape_failed)?;
        let right = concatenate(Axis(1), &[other.read.t(), self.read.t()]).map_err(shape_failed)?;
        let left_range = RangeFactor::new("difference write range", left.view())?;
        let right_range = RangeFactor::new("difference read range", right.view())?;
        let core = fast_abt(&left_range.core, &right_range.core);
        let decomposition = core
            .svd(false, true)
            .map_err(|failure| decomposition_failed("difference core", &failure))?;
        let sigma = decomposition.1;
        let Some(right_singular) = decomposition.2 else {
            return Err(GaugeRefusal::Decomposition {
                what: "difference core",
                detail: "right singular vectors were requested but not returned".to_string(),
            });
        };
        let top = (0..sigma.len()).fold(0, |best, index| if sigma[index] > sigma[best] { index } else { best });
        let direction = right_singular.row(top).to_owned();
        let witness = match &right_range.basis {
            Some(basis) => fast_av(basis, &direction),
            None => direction,
        };
        let left_norm = frobenius(left.view());
        let right_norm = frobenius(right.view());
        let numerical_error = left_range.backward_error * right_norm
            + left_norm * right_range.backward_error
            + accumulation_growth(left.ncols()) * frobenius(left_range.core.view()) * frobenius(right_range.core.view())
            + factor_singular_band(core.nrows(), core.ncols(), frobenius(core.view()));
        Ok(EvidenceStatus::exact(
            sigma[top],
            numerical_error,
            ExactBasis::Algebraic,
            Some(witness),
            AllInputs {
                width: self.read.ncols(),
            },
        )?)
    }
}

/// A factor `M` (`rows × k`) reduced to a core with the same singular values: its
/// thin QR triangle when `rows ≥ k`, otherwise `M` itself.
struct RangeFactor {
    core: Array2<f64>,
    basis: Option<Array2<f64>>,
    /// The QR backward-error band on `‖δM‖₂`, zero when no QR was taken.
    backward_error: f64,
}

impl RangeFactor {
    fn new(what: &'static str, factor: ArrayView2<'_, f64>) -> Result<Self, GaugeRefusal> {
        let (rows, k) = factor.dim();
        if rows < k {
            return Ok(Self {
                core: factor.to_owned(),
                basis: None,
                backward_error: 0.0,
            });
        }
        let (q, t) = factor.qr().map_err(|failure| decomposition_failed(what, &failure))?;
        Ok(Self {
            core: t.slice(s![..k, ..k]).to_owned(),
            basis: Some(q.slice(s![.., ..k]).to_owned()),
            backward_error: factor_singular_band(rows, k, frobenius(factor)),
        })
    }
}

/// One class of an intervention set of linear pass-through settings.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionClass {
    /// The setting the members were compared against.
    pub representative: usize,
    /// Every setting of the class, the representative first.
    pub members: Vec<usize>,
    /// The largest certified `sup_x ‖(f_member − f_representative) x‖/‖x‖` over the
    /// members: every member executes the representative's map to within it.
    pub largest_gap: f64,
}

/// Quotient an intervention set of pass-through settings by the executed map.
///
/// Each setting joins the first class whose representative it is not proven to
/// differ from, and otherwise opens a class. A setting is proven distinct when
/// `‖B′A′ − B A‖₂` minus its numerical error is positive. Two settings on one
/// `GL(r)` orbit execute one map, so they fall into one class, and each class
/// reports the certified gap its members are within.
pub fn quotient_intervention_set(settings: &[LinearPassthrough]) -> Result<Vec<InterventionClass>, GaugeRefusal> {
    let mut classes: Vec<InterventionClass> = Vec::new();
    for (index, setting) in settings.iter().enumerate() {
        let mut joined = false;
        for class in classes.iter_mut() {
            let difference = settings[class.representative].operator_difference(setting)?;
            if difference.lower_bound().is_some_and(|lower| lower > 0.0) {
                continue;
            }
            let gap = difference.upper_bound().unwrap_or(f64::INFINITY);
            class.members.push(index);
            class.largest_gap = class.largest_gap.max(gap);
            joined = true;
            break;
        }
        if !joined {
            classes.push(InterventionClass {
                representative: index,
                members: vec![index],
                largest_gap: 0.0,
            });
        }
    }
    Ok(classes)
}

/// A joint relabelling and rescaling of hidden units: new unit `i` is old unit
/// `permutation[i]`, scaled by `scales[i]`.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitGaugeChange {
    pub permutation: Vec<usize>,
    pub scales: Array1<f64>,
}

impl UnitGaugeChange {
    fn check(&self, units: usize) -> Result<(), GaugeRefusal> {
        check_len("unit permutation", units, self.permutation.len())?;
        check_len("unit scales", units, self.scales.len())?;
        let mut seen = vec![false; units];
        for &old in &self.permutation {
            if old >= units || seen[old] {
                return Err(GaugeRefusal::NotAPermutation { length: units });
            }
            seen[old] = true;
        }
        require_finite_vector("unit scales", self.scales.view())
    }
}

/// The hidden units of `y = W₂ σ(W₁ x + b₁)`: `W₁` (`H × d`), `b₁` (`H`) and `W₂`
/// (`d_out × H`), with the block's elementwise activation.
#[derive(Clone, Debug)]
pub struct HiddenUnits {
    read_in: Array2<f64>,
    bias_in: Array1<f64>,
    write_out: Array2<f64>,
    activation: GaussianActivation,
}

impl HiddenUnits {
    pub fn new(
        read_in: Array2<f64>,
        bias_in: Array1<f64>,
        write_out: Array2<f64>,
        activation: GaussianActivation,
    ) -> Result<Self, GaugeRefusal> {
        require_nonempty("hidden read", read_in.view())?;
        require_nonempty("hidden write", write_out.view())?;
        check_len("hidden bias length", read_in.nrows(), bias_in.len())?;
        check_len("hidden write columns", read_in.nrows(), write_out.ncols())?;
        require_finite_matrix("hidden read", read_in.view())?;
        require_finite_vector("hidden bias", bias_in.view())?;
        require_finite_matrix("hidden write", write_out.view())?;
        Ok(Self {
            read_in,
            bias_in,
            write_out,
            activation,
        })
    }

    pub fn units(&self) -> usize {
        self.read_in.nrows()
    }

    pub fn read_in(&self) -> ArrayView2<'_, f64> {
        self.read_in.view()
    }

    pub fn bias_in(&self) -> ArrayView1<'_, f64> {
        self.bias_in.view()
    }

    pub fn write_out(&self) -> ArrayView2<'_, f64> {
        self.write_out.view()
    }

    pub fn activation(&self) -> GaussianActivation {
        self.activation
    }

    /// The unit gauge family through these tensors, with exact zero tests.
    ///
    /// A unit is null, and not counted in the orbit, when one of these holds:
    /// - **unread:** its write column is zero, so its row and bias are free
    ///   (`d + 1` coordinates);
    /// - **silent:** its read row is zero and its activation is zero on a
    ///   neighbourhood of its bias, so its column is free (`d_out`). For ReLU
    ///   with `b < 0` the bias is free as well (`d_out + 1`). At `b = 0` every
    ///   activation here vanishes, `σ(0) = 0`, but a positive bias would move it,
    ///   so only the column counts.
    ///
    /// Every other unit carries one orbit coordinate under ReLU's positive scales,
    /// and none under GELU or SiLU.
    pub fn family(&self) -> GaugeFamily {
        let (units, width) = self.read_in.dim();
        let out = self.write_out.nrows();
        let relu = matches!(self.activation, GaussianActivation::Relu);
        let (mut orbit, mut null) = (0, 0);
        for i in 0..units {
            let unread = self.write_out.column(i).iter().all(|&entry| entry == 0.0);
            let row_zero = self.read_in.row(i).iter().all(|&entry| entry == 0.0);
            let bias = self.bias_in[i];
            if unread {
                null += width + 1;
            } else if row_zero && relu && bias < 0.0 {
                null += out + 1;
            } else if row_zero && bias == 0.0 {
                null += out;
            } else if relu {
                orbit += 1;
            }
        }
        GaugeFamily {
            kind: GaugeFamilyKind::HiddenUnits(self.activation),
            continuous: if relu {
                ContinuousGauge::PositiveDiagonal { order: units }
            } else {
                ContinuousGauge::Trivial
            },
            discrete: DiscreteGauge::UnitPermutations { units },
            parameter_coordinates: units * (width + 1) + out * units,
            orbit_dimension: DimensionInterval {
                resolved: orbit,
                at_most: orbit,
            },
            null_coordinates: null,
        }
    }

    /// `(row_i, b_i, col_i) ↦ (s_i row_{π(i)}, s_i b_{π(i)}, col_{π(i)} / s_i)`. It
    /// refuses a scale outside the activation's group: `s > 0` for ReLU and `s = 1`
    /// for GELU and SiLU.
    pub fn apply(&self, change: &UnitGaugeChange) -> Result<Self, GaugeRefusal> {
        let units = self.units();
        change.check(units)?;
        for (index, &scale) in change.scales.iter().enumerate() {
            let admitted = match self.activation {
                GaussianActivation::Relu => scale > 0.0,
                GaussianActivation::ExactGelu | GaussianActivation::Silu => scale == 1.0,
            };
            if !admitted {
                return Err(GaugeRefusal::NotInGroup {
                    what: "hidden unit scale",
                    index,
                    value: scale,
                });
            }
        }
        let mut read_in = Array2::<f64>::zeros(self.read_in.raw_dim());
        let mut bias_in = Array1::<f64>::zeros(units);
        let mut write_out = Array2::<f64>::zeros(self.write_out.raw_dim());
        for (new, (&old, &scale)) in change.permutation.iter().zip(change.scales.iter()).enumerate() {
            read_in.row_mut(new).assign(&self.read_in.row(old).mapv(|entry| entry * scale));
            bias_in[new] = self.bias_in[old] * scale;
            write_out.column_mut(new).assign(&self.write_out.column(old).mapv(|entry| entry / scale));
        }
        Ok(Self {
            read_in,
            bias_in,
            write_out,
            activation: self.activation,
        })
    }
}

/// The hidden units of a SwiGLU block `y = W_d [s(W_g x) ⊙ W_u x]`: gate `W_g` and
/// up `W_u` (`H × d`), down `W_d` (`d_out × H`).
#[derive(Clone, Debug)]
pub struct SwigluUnits {
    gate: Array2<f64>,
    up: Array2<f64>,
    down: Array2<f64>,
}

impl SwigluUnits {
    pub fn new(gate: Array2<f64>, up: Array2<f64>, down: Array2<f64>) -> Result<Self, GaugeRefusal> {
        require_nonempty("SwiGLU gate", gate.view())?;
        require_nonempty("SwiGLU down", down.view())?;
        check_len("SwiGLU up rows", gate.nrows(), up.nrows())?;
        check_len("SwiGLU up columns", gate.ncols(), up.ncols())?;
        check_len("SwiGLU down columns", gate.nrows(), down.ncols())?;
        require_finite_matrix("SwiGLU gate", gate.view())?;
        require_finite_matrix("SwiGLU up", up.view())?;
        require_finite_matrix("SwiGLU down", down.view())?;
        Ok(Self { gate, up, down })
    }

    pub fn units(&self) -> usize {
        self.gate.nrows()
    }

    pub fn gate(&self) -> ArrayView2<'_, f64> {
        self.gate.view()
    }

    pub fn up(&self) -> ArrayView2<'_, f64> {
        self.up.view()
    }

    pub fn down(&self) -> ArrayView2<'_, f64> {
        self.down.view()
    }

    /// The unit gauge family through these tensors, with exact zero tests.
    ///
    /// A unit writes nothing when its down column is zero, or when its up or gate
    /// row is zero, since `s(0) = 0`:
    /// - a zero column leaves both rows free (`2d` coordinates);
    /// - a zero up or gate row leaves the other row and the column free
    ///   (`d + d_out`).
    ///
    /// When several hold, the free sets are a union of subspaces, and the larger
    /// dimension counts. Every other unit carries one orbit coordinate.
    pub fn family(&self) -> GaugeFamily {
        let (units, width) = self.gate.dim();
        let out = self.down.nrows();
        let (mut orbit, mut null) = (0, 0);
        for i in 0..units {
            let column_zero = self.down.column(i).iter().all(|&entry| entry == 0.0);
            let row_zero = self.gate.row(i).iter().all(|&entry| entry == 0.0)
                || self.up.row(i).iter().all(|&entry| entry == 0.0);
            let mut free = 0;
            if column_zero {
                free = free.max(2 * width);
            }
            if row_zero {
                free = free.max(width + out);
            }
            if free > 0 {
                null += free;
            } else {
                orbit += 1;
            }
        }
        GaugeFamily {
            kind: GaugeFamilyKind::SwigluUnits,
            continuous: ContinuousGauge::NonzeroDiagonal { order: units },
            discrete: DiscreteGauge::UnitPermutations { units },
            parameter_coordinates: units * (2 * width + out),
            orbit_dimension: DimensionInterval {
                resolved: orbit,
                at_most: orbit,
            },
            null_coordinates: null,
        }
    }

    /// `(gate_i, up_i, col_i) ↦ (gate_{π(i)}, s_i up_{π(i)}, col_{π(i)} / s_i)`. It
    /// refuses a zero scale.
    pub fn apply(&self, change: &UnitGaugeChange) -> Result<Self, GaugeRefusal> {
        let units = self.units();
        change.check(units)?;
        if let Some(index) = change.scales.iter().position(|&scale| scale == 0.0) {
            return Err(GaugeRefusal::NotInGroup {
                what: "SwiGLU up scale",
                index,
                value: change.scales[index],
            });
        }
        let mut gate = Array2::<f64>::zeros(self.gate.raw_dim());
        let mut up = Array2::<f64>::zeros(self.up.raw_dim());
        let mut down = Array2::<f64>::zeros(self.down.raw_dim());
        for (new, (&old, &scale)) in change.permutation.iter().zip(change.scales.iter()).enumerate() {
            gate.row_mut(new).assign(&self.gate.row(old));
            up.row_mut(new).assign(&self.up.row(old).mapv(|entry| entry * scale));
            down.column_mut(new).assign(&self.down.column(old).mapv(|entry| entry / scale));
        }
        Ok(Self { gate, up, down })
    }
}

/// A norm's gain `w` (`d`), its bias `β` when the norm has one (LayerNorm), and the
/// linear reads `W_k` (`r_k × d`) that are the only consumers of its output.
#[derive(Clone, Debug)]
pub struct NormGain {
    gain: Array1<f64>,
    bias: Option<Array1<f64>>,
    reads: Vec<Array2<f64>>,
}

impl NormGain {
    pub fn new(gain: Array1<f64>, bias: Option<Array1<f64>>, reads: Vec<Array2<f64>>) -> Result<Self, GaugeRefusal> {
        let width = gain.len();
        if width == 0 {
            return Err(GaugeRefusal::EmptyTensor { what: "norm gain" });
        }
        if reads.is_empty() {
            return Err(GaugeRefusal::EmptyTensor { what: "norm reads" });
        }
        require_finite_vector("norm gain", gain.view())?;
        if let Some(bias) = &bias {
            check_len("norm bias length", width, bias.len())?;
            require_finite_vector("norm bias", bias.view())?;
        }
        for read in &reads {
            require_nonempty("norm read", read.view())?;
            check_len("norm read columns", width, read.ncols())?;
            require_finite_matrix("norm read", read.view())?;
        }
        Ok(Self { gain, bias, reads })
    }

    pub fn gain(&self) -> ArrayView1<'_, f64> {
        self.gain.view()
    }

    pub fn bias(&self) -> Option<ArrayView1<'_, f64>> {
        self.bias.as_ref().map(|bias| bias.view())
    }

    pub fn reads(&self) -> &[Array2<f64>] {
        &self.reads
    }

    /// The coordinate-scale family through these tensors, with exact zero tests.
    ///
    /// A coordinate is null when one of these holds:
    /// - **unread:** every read's column is zero, so its gain and bias are free;
    /// - **silent:** its gain and bias are zero, so its output is zero at every
    ///   input and every read's column is free (`Σ_k r_k` coordinates).
    ///
    /// Every other coordinate carries one orbit coordinate.
    pub fn family(&self) -> GaugeFamily {
        let width = self.gain.len();
        let read_rows: usize = self.reads.iter().map(|read| read.nrows()).sum();
        let per_coordinate = 1 + usize::from(self.bias.is_some());
        let (mut orbit, mut null) = (0, 0);
        for i in 0..width {
            let unread = self
                .reads
                .iter()
                .all(|read| read.column(i).iter().all(|&entry| entry == 0.0));
            let silent = self.gain[i] == 0.0 && self.bias.as_ref().is_none_or(|bias| bias[i] == 0.0);
            if unread {
                null += per_coordinate;
            } else if silent {
                null += read_rows;
            } else {
                orbit += 1;
            }
        }
        GaugeFamily {
            kind: GaugeFamilyKind::NormGain,
            continuous: ContinuousGauge::NonzeroDiagonal { order: width },
            discrete: DiscreteGauge::None,
            parameter_coordinates: width * per_coordinate + read_rows * width,
            orbit_dimension: DimensionInterval {
                resolved: orbit,
                at_most: orbit,
            },
            null_coordinates: null,
        }
    }

    /// `(w, β, W_k) ↦ (D w, D β, W_k D⁻¹)` for `D = diag(scales)`. It refuses a zero
    /// scale.
    pub fn apply(&self, scales: ArrayView1<'_, f64>) -> Result<Self, GaugeRefusal> {
        check_len("coordinate scales", self.gain.len(), scales.len())?;
        require_finite_vector("coordinate scales", scales)?;
        if let Some(index) = scales.iter().position(|&scale| scale == 0.0) {
            return Err(GaugeRefusal::NotInGroup {
                what: "norm coordinate scale",
                index,
                value: scales[index],
            });
        }
        let gain = &self.gain * &scales;
        let bias = self.bias.as_ref().map(|bias| bias * &scales);
        let reads = self.reads.iter().map(|read| read / &scales).collect();
        Ok(Self { gain, bias, reads })
    }
}

/// One block of the commutant of a rotary embedding's rotations within one head.
#[derive(Clone, Debug, PartialEq)]
pub enum RotaryBlock {
    /// The planes rotated at one frequency: `GL(m, ℂ)` on `x_a + i x_b`.
    Planes { frequency: f64, planes: Vec<usize> },
    /// Head coordinates `start..end` past the rotary dimension: `GL(end − start)`.
    PassThrough { start: usize, end: usize },
}

impl RotaryBlock {
    /// The block's real dimension: `2m²` for `m` planes, `n²` for `n` pass-through
    /// coordinates.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Planes { planes, .. } => 2 * planes.len() * planes.len(),
            Self::PassThrough { start, end } => (end - start) * (end - start),
        }
    }
}

/// The commutant blocks of `rotary`'s rotations within a head of `head_dim`
/// coordinates. Planes are grouped by bitwise-equal frequency, in first-occurrence
/// order. It refuses a frequency outside `(0, π)` and a zero or non-finite scaling.
pub fn rotary_commutant(rotary: &RotaryEmbedding, head_dim: usize) -> Result<Vec<RotaryBlock>, GaugeRefusal> {
    if rotary.rotary_dim() > head_dim {
        return Err(GaugeRefusal::DimensionMismatch {
            what: "rotary dimension within one head",
            expected: head_dim,
            found: rotary.rotary_dim(),
        });
    }
    if !(rotary.attention_scaling.is_finite() && rotary.attention_scaling != 0.0) {
        return Err(GaugeRefusal::ZeroRotaryScaling);
    }
    let mut blocks: Vec<RotaryBlock> = Vec::new();
    for (plane, &frequency) in rotary.inverse_frequencies.iter().enumerate() {
        if !(frequency > 0.0 && frequency < std::f64::consts::PI) {
            return Err(GaugeRefusal::FrequencyOutsideHalfTurn { plane, frequency });
        }
        let existing = blocks.iter().position(|block| {
            matches!(block, RotaryBlock::Planes { frequency: shared, .. } if shared.to_bits() == frequency.to_bits())
        });
        match existing {
            Some(index) => {
                if let RotaryBlock::Planes { planes, .. } = &mut blocks[index] {
                    planes.push(plane);
                }
            }
            None => blocks.push(RotaryBlock::Planes {
                frequency,
                planes: vec![plane],
            }),
        }
    }
    if rotary.rotary_dim() < head_dim {
        blocks.push(RotaryBlock::PassThrough {
            start: rotary.rotary_dim(),
            end: head_dim,
        });
    }
    Ok(blocks)
}

/// Whether `S` (`head_dim × head_dim`) lies in the commutant. Every nonzero entry must
/// couple coordinates of one block, and on a plane block `S[a_i, a_j] = S[b_i, b_j]`
/// and `S[b_i, a_j] = −S[a_i, b_j]`. The check is exact on the declared entries, as
/// for any declared group element.
fn check_rotary_commutant(
    gauge_change: ArrayView2<'_, f64>,
    rotary: &RotaryEmbedding,
    blocks: &[RotaryBlock],
) -> Result<(), GaugeRefusal> {
    let head_dim = gauge_change.nrows();
    let mut block_of = vec![0usize; head_dim];
    for (index, block) in blocks.iter().enumerate() {
        match block {
            RotaryBlock::Planes { planes, .. } => {
                for &plane in planes {
                    let (a, b) = rotary.plane(plane);
                    block_of[a] = index;
                    block_of[b] = index;
                }
            }
            RotaryBlock::PassThrough { start, end } => {
                for slot in block_of.iter_mut().take(*end).skip(*start) {
                    *slot = index;
                }
            }
        }
    }
    for ((i, j), &entry) in gauge_change.indexed_iter() {
        if entry != 0.0 && block_of[i] != block_of[j] {
            return Err(GaugeRefusal::NotInGroup {
                what: "rotary gauge coupling across commutant blocks",
                index: i * head_dim + j,
                value: entry,
            });
        }
    }
    for block in blocks {
        if let RotaryBlock::Planes { planes, .. } = block {
            for &first in planes {
                for &second in planes {
                    let (ai, bi) = rotary.plane(first);
                    let (aj, bj) = rotary.plane(second);
                    if gauge_change[[ai, aj]] != gauge_change[[bi, bj]] || gauge_change[[bi, aj]] != -gauge_change[[ai, bj]] {
                        return Err(GaugeRefusal::NotInGroup {
                            what: "rotary gauge not complex-linear on one frequency",
                            index: ai * head_dim + aj,
                            value: gauge_change[[ai, aj]],
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// The query and key projections `W x + b` of a rotary attention block, with the
/// source's geometry and rotary embedding. The weights are `n_heads·head_dim × d` and
/// `n_kv_heads·head_dim × d`, and each has its bias.
#[derive(Clone, Debug)]
pub struct RotaryQueryKey {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    query_weight: Array2<f64>,
    query_bias: Array1<f64>,
    key_weight: Array2<f64>,
    key_bias: Array1<f64>,
}

impl RotaryQueryKey {
    /// The query/key gauge of a native attention block, read through the attention owner's
    /// accessors. The owner's constructor has already checked the head grouping and every
    /// projection shape. A bias-free source carries zero biases.
    ///
    /// It refuses a block with a per-head query/key norm (Qwen3's `q_norm`/`k_norm`)
    /// between the projections and the rotation. That norm commutes only with orthogonal
    /// maps that respect its gain, so charging the full commutant would overclaim the
    /// orbit.
    pub fn new(native: &NativeAttention) -> Result<Self, GaugeRefusal> {
        if native.has_query_key_norm() {
            return Err(GaugeRefusal::QueryKeyNorm);
        }
        let geometry = native.geometry();
        let rotary = native.rotary().clone();
        let (query_weight, query_bias) = (native.query().weight.clone(), native.query().bias.clone());
        let (key_weight, key_bias) = (native.key().weight.clone(), native.key().bias.clone());
        if geometry.head_dim == 0 {
            return Err(GaugeRefusal::DimensionMismatch {
                what: "a nonempty attention head",
                expected: 1,
                found: 0,
            });
        }
        require_finite_matrix("query weight", query_weight.view())?;
        require_finite_vector("query bias", query_bias.view())?;
        require_finite_matrix("key weight", key_weight.view())?;
        require_finite_vector("key bias", key_bias.view())?;
        rotary_commutant(&rotary, geometry.head_dim)?;
        Ok(Self {
            geometry,
            rotary,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
        })
    }

    pub fn query_weight(&self) -> ArrayView2<'_, f64> {
        self.query_weight.view()
    }

    pub fn query_bias(&self) -> ArrayView1<'_, f64> {
        self.query_bias.view()
    }

    pub fn key_weight(&self) -> ArrayView2<'_, f64> {
        self.key_weight.view()
    }

    pub fn key_bias(&self) -> ArrayView1<'_, f64> {
        self.key_bias.view()
    }

    fn head_rows(&self, head: usize) -> std::ops::Range<usize> {
        head * self.geometry.head_dim..(head + 1) * self.geometry.head_dim
    }

    /// The homogeneous rows `[W | b]` of one head, `head_dim × (d + 1)`: the read of
    /// `(x, 1)`.
    fn homogeneous_rows(weight: &Array2<f64>, bias: &Array1<f64>, rows: std::ops::Range<usize>) -> Result<Array2<f64>, GaugeRefusal> {
        concatenate(
            Axis(1),
            &[weight.slice(s![rows.clone(), ..]), bias.slice(s![rows]).insert_axis(Axis(1))],
        )
        .map_err(shape_failed)
    }

    /// The commutant family, with `c = Σ 2m² + n_pass²` per key/value head.
    ///
    /// A head's orbit is charged `c` when its key's homogeneous rows `[W_K | b_K]`, or
    /// those of one of its query heads, resolve full row rank. That forces the tangent
    /// condition `X [W_K | b_K] = 0` or `Xᵀ [W_Q | b_Q] = 0` to give `X = 0`. Otherwise
    /// nothing is charged for that head.
    pub fn family(&self) -> Result<GaugeFamily, GaugeRefusal> {
        let g = self.geometry;
        let per_copy: usize = rotary_commutant(&self.rotary, g.head_dim)?
            .iter()
            .map(RotaryBlock::dimension)
            .sum();
        let mut resolved = 0;
        for kv in 0..g.n_kv_heads {
            let key_rows = Self::homogeneous_rows(&self.key_weight, &self.key_bias, self.head_rows(kv))?;
            let mut full = resolved_rank("key head", key_rows.view())? == g.head_dim;
            for head in 0..g.n_heads {
                if full {
                    break;
                }
                if g.key_value_head(head) == kv {
                    let query_rows = Self::homogeneous_rows(&self.query_weight, &self.query_bias, self.head_rows(head))?;
                    full = resolved_rank("query head", query_rows.view())? == g.head_dim;
                }
            }
            if full {
                resolved += per_copy;
            }
        }
        Ok(GaugeFamily {
            kind: GaugeFamilyKind::RotaryQueryKey,
            continuous: ContinuousGauge::RotaryCommutant {
                copies: g.n_kv_heads,
                per_copy,
            },
            discrete: DiscreteGauge::None,
            parameter_coordinates: (g.n_heads + g.n_kv_heads) * g.head_dim * (g.model_dim + 1),
            orbit_dimension: DimensionInterval {
                resolved,
                at_most: g.n_kv_heads * per_copy,
            },
            null_coordinates: 0,
        })
    }

    /// `([W_K | b_K]_g, [W_Q | b_Q]_h) ↦ (S_g [W_K | b_K]_g, S_g⁻ᵀ [W_Q | b_Q]_h)`, one
    /// `S_g` per key/value head. It refuses an `S_g` outside the commutant, or one whose
    /// SVD resolves fewer than `head_dim` singular values.
    pub fn apply(&self, gauge_changes: &[Array2<f64>]) -> Result<Self, GaugeRefusal> {
        let g = self.geometry;
        check_len("rotary gauge changes", g.n_kv_heads, gauge_changes.len())?;
        let blocks = rotary_commutant(&self.rotary, g.head_dim)?;
        let mut carried = self.clone();
        for (kv, gauge_change) in gauge_changes.iter().enumerate() {
            check_len("rotary gauge change rows", g.head_dim, gauge_change.nrows())?;
            check_len("rotary gauge change columns", g.head_dim, gauge_change.ncols())?;
            require_finite_matrix("rotary gauge change", gauge_change.view())?;
            check_rotary_commutant(gauge_change.view(), &self.rotary, &blocks)?;
            let inverse = invert_gauge_change(gauge_change.view())?;
            let rows = self.head_rows(kv);
            carried
                .key_weight
                .slice_mut(s![rows.clone(), ..])
                .assign(&fast_ab(gauge_change, &self.key_weight.slice(s![rows.clone(), ..])));
            carried
                .key_bias
                .slice_mut(s![rows.clone()])
                .assign(&fast_av(gauge_change, &self.key_bias.slice(s![rows])));
            for head in 0..g.n_heads {
                if g.key_value_head(head) == kv {
                    let rows = self.head_rows(head);
                    carried
                        .query_weight
                        .slice_mut(s![rows.clone(), ..])
                        .assign(&fast_ab(&inverse.t(), &self.query_weight.slice(s![rows.clone(), ..])));
                    carried
                        .query_bias
                        .slice_mut(s![rows.clone()])
                        .assign(&fast_av(&inverse.t(), &self.query_bias.slice(s![rows])));
                }
            }
        }
        Ok(carried)
    }

    /// The rotary commutant of one key/value head's coordinates, declared to
    /// `operators::classify_under`. Each group lists the coordinate pairs of the planes
    /// sharing one frequency, and the pass-through range follows the rotary dimension.
    pub fn declared_gauge(&self) -> Result<DeclaredGauge, GaugeRefusal> {
        rotary_declaration(&self.rotary, self.geometry.head_dim)
    }
}

fn rotary_declaration(rotary: &RotaryEmbedding, head_dim: usize) -> Result<DeclaredGauge, GaugeRefusal> {
    let mut plane_groups = Vec::new();
    for block in rotary_commutant(rotary, head_dim)? {
        if let RotaryBlock::Planes { planes, .. } = block {
            plane_groups.push(planes.iter().map(|&plane| rotary.plane(plane)).collect());
        }
    }
    let pass_through = rotary.rotary_dim()..head_dim;
    Ok(DeclaredGauge::RotaryCommutant(RotaryGroups::new(plane_groups, pass_through)?))
}

/// Which side of a query/key pair a norm gain or a projection head is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryKeySide {
    Query,
    Key,
}

/// A rotary query/key block behind a per-head query/key RMS norm (Qwen3 `q_norm`,
/// `k_norm`): the projections, the norm's `ε`, and the query and key gains that every
/// head shares. Its group is derived in the module docs.
#[derive(Clone, Debug)]
pub struct NormedRotaryQueryKey {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    query_weight: Array2<f64>,
    query_bias: Array1<f64>,
    key_weight: Array2<f64>,
    key_bias: Array1<f64>,
    epsilon: f64,
    query_gain: Array1<f64>,
    key_gain: Array1<f64>,
}

/// An element of [`NormedRotaryQueryKey`]'s group, in the group's own coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct NormedRotaryChange {
    /// `ρ_p > 0` for each rotated plane: its query gains scale by `ρ_p` and its key gains
    /// by `1/ρ_p`.
    pub plane_scales: Array1<f64>,
    /// `n_kv_heads × planes`. Key/value head `g`'s map on plane `p` is `ρ_p R^t`, where
    /// `R` is the quarter turn `(x_a, x_b) ↦ (−x_b, x_a)` and `t` is in `0..4`.
    pub quarter_turns: Array2<u8>,
    /// The query gain coordinates whose sign flips. Each flip is carried by its row in
    /// every query head.
    pub query_gain_flips: Vec<bool>,
    /// The key gain coordinates whose sign flips. Each flip is carried by its row in
    /// every key head.
    pub key_gain_flips: Vec<bool>,
}

impl NormedRotaryQueryKey {
    /// The normed query/key gauge of a native attention block, read through the attention
    /// owner's accessors.
    ///
    /// It refuses, typed, each block the derivation does not cover: no norm, `ε` not
    /// positive, a zero gain, pass-through coordinates, two planes of one frequency, and a
    /// projection head whose weight rows do not resolve full row rank.
    pub fn new(native: &NativeAttention) -> Result<Self, GaugeRefusal> {
        let norm = native.query_key_norm().ok_or(GaugeRefusal::NoQueryKeyNorm)?;
        let epsilon = norm.epsilon();
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(GaugeRefusal::QueryKeyNormEpsilon { epsilon });
        }
        let geometry = native.geometry();
        let rotary = native.rotary().clone();
        for block in rotary_commutant(&rotary, geometry.head_dim)? {
            match block {
                RotaryBlock::PassThrough { start, end } => {
                    return Err(GaugeRefusal::QueryKeyNormPassThrough { coordinates: end - start });
                }
                RotaryBlock::Planes { planes, .. } if planes.len() > 1 => {
                    return Err(GaugeRefusal::QueryKeyNormSharedFrequency {
                        first: planes[0],
                        second: planes[1],
                    });
                }
                RotaryBlock::Planes { .. } => {}
            }
        }
        let (query_gain, key_gain) = (norm.query_gain().to_owned(), norm.key_gain().to_owned());
        for (side, gain) in [(QueryKeySide::Query, &query_gain), (QueryKeySide::Key, &key_gain)] {
            require_finite_vector("query/key norm gain", gain.view())?;
            if let Some(coordinate) = gain.iter().position(|&entry| entry == 0.0) {
                return Err(GaugeRefusal::QueryKeyNormZeroGain { side, coordinate });
            }
        }
        let block = Self {
            geometry,
            rotary,
            query_weight: native.query().weight.clone(),
            query_bias: native.query().bias.clone(),
            key_weight: native.key().weight.clone(),
            key_bias: native.key().bias.clone(),
            epsilon,
            query_gain,
            key_gain,
        };
        require_finite_matrix("query weight", block.query_weight.view())?;
        require_finite_vector("query bias", block.query_bias.view())?;
        require_finite_matrix("key weight", block.key_weight.view())?;
        require_finite_vector("key bias", block.key_bias.view())?;
        let hd = geometry.head_dim;
        for (side, weight, heads) in [
            (QueryKeySide::Query, &block.query_weight, geometry.n_heads),
            (QueryKeySide::Key, &block.key_weight, geometry.n_kv_heads),
        ] {
            for head in 0..heads {
                let resolved = resolved_rank("normed projection head", weight.slice(s![head * hd..(head + 1) * hd, ..]))?;
                if resolved < hd {
                    return Err(GaugeRefusal::QueryKeyNormHeadRank {
                        side,
                        head,
                        resolved,
                        order: hd,
                    });
                }
            }
        }
        Ok(block)
    }

    pub fn query_weight(&self) -> ArrayView2<'_, f64> {
        self.query_weight.view()
    }

    pub fn query_bias(&self) -> ArrayView1<'_, f64> {
        self.query_bias.view()
    }

    pub fn key_weight(&self) -> ArrayView2<'_, f64> {
        self.key_weight.view()
    }

    pub fn key_bias(&self) -> ArrayView1<'_, f64> {
        self.key_bias.view()
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn query_gain(&self) -> ArrayView1<'_, f64> {
        self.query_gain.view()
    }

    pub fn key_gain(&self) -> ArrayView1<'_, f64> {
        self.key_gain.view()
    }

    /// Whether plane `plane` carries a rotation per key/value head. That holds when its
    /// two query gains are equal in magnitude and so are its two key gains, compared
    /// exactly.
    pub fn coincident(&self, plane: usize) -> bool {
        let (a, b) = self.rotary.plane(plane);
        self.query_gain[a].abs() == self.query_gain[b].abs() && self.key_gain[a].abs() == self.key_gain[b].abs()
    }

    /// The family through the registered tensors. Its orbit dimension is exact. Every gain
    /// is nonzero, so the plane scales move the gains freely. Every head resolves full row
    /// rank, so each rotation moves its head's rows, and different rotations move
    /// different rows.
    pub fn family(&self) -> GaugeFamily {
        let g = self.geometry;
        let planes = self.rotary.inverse_frequencies.len();
        let coincident = (0..planes).filter(|&plane| self.coincident(plane)).count();
        let rotations = g.n_kv_heads * coincident;
        GaugeFamily {
            kind: GaugeFamilyKind::NormedRotaryQueryKey,
            continuous: ContinuousGauge::NormedRotary {
                plane_scales: planes,
                plane_rotations: rotations,
            },
            discrete: DiscreteGauge::NormedRotary {
                generic_planes: planes - coincident,
                coincident_planes: coincident,
                key_value_heads: g.n_kv_heads,
            },
            parameter_coordinates: (g.n_heads + g.n_kv_heads) * g.head_dim * (g.model_dim + 1) + 2 * g.head_dim,
            orbit_dimension: DimensionInterval {
                resolved: planes + rotations,
                at_most: planes + rotations,
            },
            null_coordinates: 0,
        }
    }

    /// The block under `change`. On plane `p`, key/value head `g` maps the pre-norm rows
    /// of its key, and of each of its query heads, by the signed permutation
    /// `A = D_{w′}⁻¹ ρ_p R^t D_w`. The gains become `w′ = ρ_p τ(w)` and `v′ = τ(v)/ρ_p`,
    /// where `τ` swaps the plane's two gain entries when its gains are not coincident and
    /// its turns are odd. Gain flips follow. Every carried row is exact, and the gains are
    /// exact when every `ρ_p` is a power of two.
    ///
    /// It refuses a scale that is not positive, a turn outside `0..4`, and turns of both
    /// parities on a plane whose gains are not coincident, where the heads that do not turn
    /// cannot follow the shared gain swap. It realises the elements whose head maps are
    /// quarter turns. On a coincident plane those are the quarter turns among its rotations.
    pub fn apply(&self, change: &NormedRotaryChange) -> Result<Self, GaugeRefusal> {
        let g = self.geometry;
        let hd = g.head_dim;
        let planes = self.rotary.inverse_frequencies.len();
        check_len("plane scales", planes, change.plane_scales.len())?;
        check_len("quarter-turn key/value heads", g.n_kv_heads, change.quarter_turns.nrows())?;
        check_len("quarter-turn planes", planes, change.quarter_turns.ncols())?;
        check_len("query gain flips", hd, change.query_gain_flips.len())?;
        check_len("key gain flips", hd, change.key_gain_flips.len())?;
        for (plane, &scale) in change.plane_scales.iter().enumerate() {
            if !(scale.is_finite() && scale > 0.0) {
                return Err(GaugeRefusal::NotInGroup {
                    what: "plane gain scale",
                    index: plane,
                    value: scale,
                });
            }
        }
        for ((kv, plane), &turn) in change.quarter_turns.indexed_iter() {
            if turn > 3 {
                return Err(GaugeRefusal::NotInGroup {
                    what: "quarter-turn count",
                    index: kv * planes + plane,
                    value: f64::from(turn),
                });
            }
        }
        let mut carried = self.clone();
        for plane in 0..planes {
            let (a, b) = self.rotary.plane(plane);
            let coincident = self.coincident(plane);
            let turns = change.quarter_turns.column(plane);
            let odd = turns.iter().next().is_some_and(|&turn| turn % 2 == 1);
            if !coincident {
                if let Some(kv) = turns.iter().position(|&turn| (turn % 2 == 1) != odd) {
                    return Err(GaugeRefusal::NotInGroup {
                        what: "quarter turns of both parities on a plane whose gains are not coincident",
                        index: kv * planes + plane,
                        value: f64::from(turns[kv]),
                    });
                }
            }
            let swap = !coincident && odd;
            let scale = change.plane_scales[plane];
            for i in [a, b] {
                let source = if swap { a + b - i } else { i };
                carried.query_gain[i] = scale * self.query_gain[source];
                carried.key_gain[i] = self.key_gain[source] / scale;
            }
            for (kv, &turn) in turns.iter().enumerate() {
                let plane_turn = PlaneTurn { a, b, turn, swap };
                plane_turn.carry(&mut carried.key_weight, &mut carried.key_bias, &self.key_weight, &self.key_bias, kv * hd, &self.key_gain);
                for head in (0..g.n_heads).filter(|&head| g.key_value_head(head) == kv) {
                    plane_turn.carry(
                        &mut carried.query_weight,
                        &mut carried.query_bias,
                        &self.query_weight,
                        &self.query_bias,
                        head * hd,
                        &self.query_gain,
                    );
                }
            }
        }
        flip_gain_rows(&mut carried.query_gain, &mut carried.query_weight, &mut carried.query_bias, g.n_heads, hd, &change.query_gain_flips);
        flip_gain_rows(&mut carried.key_gain, &mut carried.key_weight, &mut carried.key_bias, g.n_kv_heads, hd, &change.key_gain_flips);
        Ok(carried)
    }

    /// The rotary commutant, declared to `operators::classify_under` as for
    /// [`RotaryQueryKey`]. On one head's normed coordinates this family's head maps are
    /// `ρ R^t`, and `ρ R_θ` on a coincident plane. Those commute with exactly `αI + βJ` on
    /// each plane, and with nothing across planes, because each plane has its own scale.
    /// That is the rotary commutant's commutant. The owner's witnesses for one-plane
    /// groups are scale 2 on a plane and `J` on a plane, and both are
    /// [`NormedRotaryChange`]s.
    pub fn declared_gauge(&self) -> Result<DeclaredGauge, GaugeRefusal> {
        rotary_declaration(&self.rotary, self.geometry.head_dim)
    }
}

/// One plane's quarter turn `R^turn` on the pre-norm rows `a`, `b` of one projection head.
struct PlaneTurn {
    a: usize,
    b: usize,
    turn: u8,
    swap: bool,
}

impl PlaneTurn {
    /// Rows `a`, `b` of the head at row `offset`, under `A = D_{g′}⁻¹ ρ R^turn D_g` with
    /// `g′ = ρ τ(g)`. `R^turn` reads row `σ(i)` into row `i` with a sign, so
    /// `A_{iσ(i)} = R^turn_{iσ(i)} g_{σ(i)} / g_{τ(i)}`. Where `σ(i) = τ(i)` that is the
    /// turn's sign. Where they differ the plane is coincident, `|g_a| = |g_b|`, so it is
    /// the turn's sign times the gains' signs. No entry rounds.
    fn carry(
        &self,
        weight: &mut Array2<f64>,
        bias: &mut Array1<f64>,
        source_weight: &Array2<f64>,
        source_bias: &Array1<f64>,
        offset: usize,
        gain: &Array1<f64>,
    ) {
        let (a, b) = (self.a, self.b);
        let (odd, sign_a, sign_b) = match self.turn {
            0 => (false, 1.0, 1.0),
            1 => (true, -1.0, 1.0),
            2 => (false, -1.0, -1.0),
            _ => (true, 1.0, -1.0),
        };
        for (row, turn_sign) in [(a, sign_a), (b, sign_b)] {
            let read = if odd { a + b - row } else { row };
            let kept = if self.swap { a + b - row } else { row };
            let sign = if read == kept {
                turn_sign
            } else {
                turn_sign * gain[read].signum() * gain[kept].signum()
            };
            weight
                .row_mut(offset + row)
                .assign(&source_weight.row(offset + read).mapv(|entry| sign * entry));
            bias[offset + row] = sign * source_bias[offset + read];
        }
    }
}

/// Flips each flagged gain coordinate, together with its row, weight and bias, in every head.
fn flip_gain_rows(gain: &mut Array1<f64>, weight: &mut Array2<f64>, bias: &mut Array1<f64>, heads: usize, head_dim: usize, flips: &[bool]) {
    for (coordinate, &flip) in flips.iter().enumerate() {
        if flip {
            gain[coordinate] = -gain[coordinate];
            for head in 0..heads {
                let row = head * head_dim + coordinate;
                weight.row_mut(row).mapv_inplace(|entry| -entry);
                bias[row] = -bias[row];
            }
        }
    }
}

/// The query/key gauge of a native attention block. It is the rotary commutant when the
/// block has no query/key norm, and the normed family when it has one. Only norm variants
/// outside the normed derivation are refused.
#[derive(Clone, Debug)]
pub enum QueryKeyGauge {
    Rotary(RotaryQueryKey),
    NormedRotary(NormedRotaryQueryKey),
}

impl QueryKeyGauge {
    pub fn new(native: &NativeAttention) -> Result<Self, GaugeRefusal> {
        if native.has_query_key_norm() {
            NormedRotaryQueryKey::new(native).map(Self::NormedRotary)
        } else {
            RotaryQueryKey::new(native).map(Self::Rotary)
        }
    }

    pub fn family(&self) -> Result<GaugeFamily, GaugeRefusal> {
        match self {
            Self::Rotary(block) => block.family(),
            Self::NormedRotary(block) => Ok(block.family()),
        }
    }

    pub fn declared_gauge(&self) -> Result<DeclaredGauge, GaugeRefusal> {
        match self {
            Self::Rotary(block) => block.declared_gauge(),
            Self::NormedRotary(block) => block.declared_gauge(),
        }
    }
}

/// How one read site reads the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualRead {
    /// A linear read of the stream itself.
    Linear,
    /// A read through an RMSNorm, `w ⊙ h (mean(h²) + ε)^{-1/2}`.
    RmsNorm,
    /// A read through a LayerNorm, which centres `h` before normalizing it.
    LayerNorm,
}

/// The basis gauge of a residual stream of width `d`, as its reads allow.
///
/// Ties are not seen here. A storage tensor read at one site and written at
/// another, such as a tied embedding and unembedding, constrains the group
/// further, and its owner refuses a carry that would untie it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualStreamGauge {
    width: usize,
    continuous: ContinuousGauge,
}

impl ResidualStreamGauge {
    /// `GL(d)` when every read is linear. `O(d)` when some read passes an RMSNorm
    /// and none a LayerNorm. `O(1) × O(d − 1)` when some read passes a LayerNorm.
    /// It refuses a stream with no reads.
    pub fn new(width: usize, reads: &[ResidualRead]) -> Result<Self, GaugeRefusal> {
        if width == 0 {
            return Err(GaugeRefusal::EmptyTensor {
                what: "residual stream",
            });
        }
        if reads.is_empty() {
            return Err(GaugeRefusal::EmptyTensor {
                what: "residual reads",
            });
        }
        let continuous = if reads.contains(&ResidualRead::LayerNorm) {
            ContinuousGauge::OrthogonalFixingOnes { order: width }
        } else if reads.contains(&ResidualRead::RmsNorm) {
            ContinuousGauge::Orthogonal { order: width }
        } else {
            ContinuousGauge::GeneralLinear { order: width }
        };
        Ok(Self { width, continuous })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn continuous(&self) -> ContinuousGauge {
        self.continuous
    }

    /// The stream's gauge as declared to `operators::classify_under`.
    pub fn declared_gauge(&self) -> Result<Option<DeclaredGauge>, GaugeRefusal> {
        declared_gauge(self.continuous, DiscreteGauge::None)
    }
}

/// A residual basis change `Q = H_k ⋯ H_1`, a product of Householder reflections
/// `H_j = I − 2 v_j v_jᵀ / ‖v_j‖²` (columns `v_j` of a `d × k` array), applied
/// matrix-free in `O(d k)` per vector.
///
/// Every element of `O(d)` is a product of at most `d` reflections
/// (Cartan–Dieudonné). Every element of `O(1) × O(d − 1)` is such a product of
/// reflections with `v ⊥ 1`, which fix `1`, and at most one with `v ∥ 1`, which
/// negates it. `GL(d)` elements outside `O(d)` are not carried here.
#[derive(Clone, Debug)]
pub struct ResidualReflections {
    vectors: Array2<f64>,
    squared_norms: Array1<f64>,
}

impl ResidualReflections {
    /// It refuses a vector whose computed squared norm is zero.
    ///
    /// Under `O(1) × O(d − 1)` it also refuses a vector that is neither constant
    /// nor centred. A vector counts as centred when `1ᵀ v` is not resolved from
    /// zero by its own evaluation: `|Σ v_i| ≤ γ_{d−1} Σ |v_i|`.
    pub fn new(vectors: Array2<f64>, gauge: &ResidualStreamGauge) -> Result<Self, GaugeRefusal> {
        require_nonempty("reflection vectors", vectors.view())?;
        check_len("reflection vector length", gauge.width, vectors.nrows())?;
        require_finite_matrix("reflection vectors", vectors.view())?;
        let fixing_ones = matches!(gauge.continuous, ContinuousGauge::OrthogonalFixingOnes { .. });
        let mut squared_norms = Array1::<f64>::zeros(vectors.ncols());
        for (j, vector) in vectors.columns().into_iter().enumerate() {
            let squared = vector.dot(&vector);
            if squared == 0.0 {
                return Err(GaugeRefusal::NotInGroup {
                    what: "reflection vector norm",
                    index: j,
                    value: squared,
                });
            }
            if fixing_ones {
                let sum: f64 = vector.sum();
                let absolute: f64 = vector.iter().map(|entry| entry.abs()).sum();
                let centred = sum.abs() <= accumulation_growth(vector.len() - 1) * absolute;
                let constant = vector.iter().all(|&entry| entry == vector[0]);
                if !centred && !constant {
                    return Err(GaugeRefusal::NotInGroup {
                        what: "reflection vector sum under a LayerNorm read",
                        index: j,
                        value: sum,
                    });
                }
            }
            squared_norms[j] = squared;
        }
        Ok(Self { vectors, squared_norms })
    }

    /// `Q x`.
    pub fn apply(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, GaugeRefusal> {
        Ok(self.carry_write(x.insert_axis(Axis(1)))?.remove_axis(Axis(1)))
    }

    /// `Q W` for a write `W` (`d × r`): every column carried.
    pub fn carry_write(&self, write: ArrayView2<'_, f64>) -> Result<Array2<f64>, GaugeRefusal> {
        check_len("carried write rows", self.vectors.nrows(), write.nrows())?;
        let mut carried = write.to_owned();
        for (vector, &squared) in self.vectors.columns().into_iter().zip(self.squared_norms.iter()) {
            let coefficients = carried.t().dot(&vector) * (2.0 / squared);
            for (mut column, &coefficient) in carried.columns_mut().into_iter().zip(coefficients.iter()) {
                column.scaled_add(-coefficient, &vector);
            }
        }
        Ok(carried)
    }

    /// `W Qᵀ` for a linear read `W` (`r × d`): every row carried.
    pub fn carry_read(&self, read: ArrayView2<'_, f64>) -> Result<Array2<f64>, GaugeRefusal> {
        Ok(self.carry_write(read.t())?.t().to_owned())
    }

    /// `W diag(w) Qᵀ diag(w)⁻¹` for a read `W` (`r × d`) behind a norm with gain `w`,
    /// which keeps the gain unchanged. It refuses a zero gain entry.
    pub fn carry_normed_read(&self, gain: ArrayView1<'_, f64>, read: ArrayView2<'_, f64>) -> Result<Array2<f64>, GaugeRefusal> {
        require_nonzero_gain(gain, self.vectors.nrows())?;
        check_len("normed read columns", gain.len(), read.ncols())?;
        let scaled = &read * &gain;
        Ok(self.carry_read(scaled.view())? / &gain)
    }

    /// `diag(w) Q diag(w)⁻¹ β` for a LayerNorm bias `β` with gain `w`, so the carried
    /// normed reads see `W diag(w) Qᵀ diag(w)⁻¹ (diag(w) Q P h ν + β′) = W (diag(w) P h ν + β)`.
    /// It refuses a zero gain entry.
    pub fn carry_norm_bias(&self, gain: ArrayView1<'_, f64>, bias: ArrayView1<'_, f64>) -> Result<Array1<f64>, GaugeRefusal> {
        require_nonzero_gain(gain, self.vectors.nrows())?;
        check_len("norm bias length", gain.len(), bias.len())?;
        let unscaled = &bias / &gain;
        Ok(self.apply(unscaled.view())? * &gain)
    }
}

fn require_nonzero_gain(gain: ArrayView1<'_, f64>, width: usize) -> Result<(), GaugeRefusal> {
    check_len("norm gain length", width, gain.len())?;
    require_finite_vector("norm gain", gain)?;
    match gain.iter().position(|&entry| entry == 0.0) {
        Some(coordinate) => Err(GaugeRefusal::ZeroGain { coordinate }),
        None => Ok(()),
    }
}

fn check_len(what: &'static str, expected: usize, found: usize) -> Result<(), GaugeRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(GaugeRefusal::DimensionMismatch { what, expected, found })
    }
}

fn require_nonempty(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<(), GaugeRefusal> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        Err(GaugeRefusal::EmptyTensor { what })
    } else {
        Ok(())
    }
}

fn require_finite_matrix(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<(), GaugeRefusal> {
    if matrix.iter().all(|entry| entry.is_finite()) {
        Ok(())
    } else {
        Err(GaugeRefusal::NonFinite { what })
    }
}

fn require_finite_vector(what: &'static str, vector: ArrayView1<'_, f64>) -> Result<(), GaugeRefusal> {
    if vector.iter().all(|entry| entry.is_finite()) {
        Ok(())
    } else {
        Err(GaugeRefusal::NonFinite { what })
    }
}

fn decomposition_failed(what: &'static str, failure: &FaerLinalgError) -> GaugeRefusal {
    GaugeRefusal::Decomposition {
        what,
        detail: format!("{failure:?}"),
    }
}

fn shape_failed(failure: ndarray::ShapeError) -> GaugeRefusal {
    GaugeRefusal::Decomposition {
        what: "factor concatenation",
        detail: failure.to_string(),
    }
}

fn frobenius(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|entry| entry * entry).sum::<f64>().sqrt()
}

/// The number of singular values of `matrix` above its SVD's backward-error band
/// `max(rows, cols)·ε·σ_max`: a lower bound on the rank of the registered tensor.
fn resolved_rank(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<usize, GaugeRefusal> {
    let sigma = matrix
        .svd(false, false)
        .map_err(|failure| decomposition_failed(what, &failure))?
        .1;
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(matrix.nrows(), matrix.ncols(), sigma_max);
    Ok(sigma.iter().filter(|&&value| value > band).count())
}

/// `S⁻¹ = V Σ⁻¹ Uᵀ` from the SVD `S = U Σ Vᵀ`. It refuses a singular value inside
/// the SVD's rounding band, where the inverse carries no correct digit.
fn invert_gauge_change(gauge_change: ArrayView2<'_, f64>) -> Result<Array2<f64>, GaugeRefusal> {
    let what = "gauge change";
    let order = gauge_change.nrows();
    let (left, sigma, right_t) = gauge_change
        .svd(true, true)
        .map_err(|failure| decomposition_failed(what, &failure))?;
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(order, order, sigma_max);
    let resolved = sigma.iter().filter(|&&value| value > band).count();
    if resolved < order {
        return Err(GaugeRefusal::SingularGaugeChange { resolved, order });
    }
    let (Some(left), Some(right_t)) = (left, right_t) else {
        return Err(GaugeRefusal::Decomposition {
            what,
            detail: "singular vectors were requested but not returned".to_string(),
        });
    };
    let mut right_scaled = right_t.t().to_owned();
    for (j, &value) in sigma.iter().enumerate() {
        right_scaled.column_mut(j).mapv_inplace(|entry| entry / value);
    }
    Ok(fast_abt(&right_scaled, &left))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::attention::{AffineProjection, AttentionExecution, RotaryPairing};
    use crate::parameter_decomposition::operators::{MaskGaugeVerdict, classify_under};
    use crate::parameter_decomposition::rewrite::NativeMlp;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows, cols));
        for entry in out.iter_mut() {
            *entry = rng.random_range(-1.0..1.0);
        }
        out
    }

    fn uniform_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(len);
        for entry in out.iter_mut() {
            *entry = rng.random_range(-1.0..1.0);
        }
        out
    }

    /// Scales of magnitude in `[½, 2]`, negated at random when `signed`.
    fn random_scales(rng: &mut StdRng, len: usize, signed: bool) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(len);
        for entry in out.iter_mut() {
            let magnitude = rng.random_range(0.5..2.0);
            *entry = if signed && rng.random_range(0.0..1.0) < 0.5 {
                -magnitude
            } else {
                magnitude
            };
        }
        out
    }

    fn random_permutation(rng: &mut StdRng, len: usize) -> Vec<usize> {
        let mut permutation: Vec<usize> = (0..len).collect();
        permutation.shuffle(rng);
        permutation
    }

    fn norm(v: &Array1<f64>) -> f64 {
        v.dot(v).sqrt()
    }

    fn exact_parts(status: &EvidenceStatus<Array1<f64>, AllInputs>) -> (f64, f64, Array1<f64>) {
        match status {
            EvidenceStatus::Exact {
                value,
                numerical_error,
                witness: Some(witness),
                ..
            } => (*value, *numerical_error, witness.clone()),
            other => panic!("an operator difference is an exact extremum with a witness, got {other:?}"),
        }
    }

    /// First-order bound on `‖B Ŝ⁻¹ fl(S A) − B A‖₂` for the regauged setting.
    ///
    /// The computed inverse is the exact inverse of `S + E` with `‖E‖₂ ≤ r·ε·‖S‖₂`
    /// (the SVD backward error, [`factor_singular_band`]'s convention). It is formed
    /// from products of at most `2r` rounded terms, whose absolute magnitudes
    /// `κ_F = ‖S‖_F ‖Ŝ⁻¹‖_F` majorizes. So `‖Ŝ⁻¹ S − I‖ ≤ η = κ_F·r·(ε + γ_{2r})`.
    /// `fl(S A)` and `fl(B Ŝ⁻¹)` each round `r` terms per entry, on magnitudes
    /// majorized by `‖S‖_F ‖A‖_F` and `‖B‖_F ‖Ŝ⁻¹‖_F`, and each meets the other
    /// factor once. Together they give `‖A‖_F ‖B‖_F (η + 2 γ_r κ_F)`.
    fn regauge_band(teacher: &LinearPassthrough, gauge_change: &Array2<f64>) -> f64 {
        let r = teacher.order();
        let inverse = invert_gauge_change(gauge_change.view()).expect("the fixture's gauge change is invertible");
        let kappa = frobenius(gauge_change.view()) * frobenius(inverse.view());
        let eta = kappa * r as f64 * (f64::EPSILON + accumulation_growth(2 * r));
        frobenius(teacher.read()) * frobenius(teacher.write()) * (eta + 2.0 * accumulation_growth(r) * kappa)
    }

    /// A2's `GL(r)` half through the executed map: the regauged setting executes the
    /// teacher's map within its construction band. The one-sided change `(S A, B)`
    /// is the positive control, and it must be proven distinct: its lower bound
    /// exceeds the rounding of `fl(S A)`, `γ_r ‖S‖_F ‖A‖_F ‖B‖_F`, so the exact
    /// `B S A ≠ B A`. Its witness input must move the executed output by more than
    /// both executions' rounding.
    #[test]
    fn a_general_linear_change_keeps_the_executed_map_and_a_one_sided_change_moves_it() {
        let mut rng = StdRng::seed_from_u64(295_401);
        let (d_in, d_out, r) = (7, 6, 4);
        let teacher = LinearPassthrough::new(uniform_matrix(&mut rng, r, d_in), uniform_matrix(&mut rng, d_out, r))
            .expect("finite factors");
        let gauge_change = uniform_matrix(&mut rng, r, r);
        let regauged = teacher.apply(gauge_change.view()).expect("a random change is invertible");
        let band = regauge_band(&teacher, &gauge_change);
        let (value, numerical_error, witness) =
            exact_parts(&teacher.operator_difference(&regauged).expect("comparable settings"));
        assert!(
            value <= band + numerical_error,
            "regauged map moved by {value:.3e}, construction band {band:.3e} + evaluation error {numerical_error:.3e}"
        );
        assert_eq!(witness.len(), d_in);

        let one_sided = LinearPassthrough::new(fast_ab(&gauge_change, &teacher.read), teacher.write.clone())
            .expect("finite factors");
        let one_sided_band = accumulation_growth(r)
            * frobenius(gauge_change.view())
            * frobenius(teacher.read())
            * frobenius(teacher.write());
        let difference = teacher.operator_difference(&one_sided).expect("comparable settings");
        let lower = difference.lower_bound().expect("an exact extremum bounds itself");
        assert!(
            lower > one_sided_band,
            "a one-sided change must be proven distinct: lower bound {lower:.3e}, rounding {one_sided_band:.3e}"
        );
        let (value, numerical_error, witness) = exact_parts(&difference);
        let executed = norm(
            &(one_sided.apply_to(witness.view()).expect("apply") - &teacher.apply_to(witness.view()).expect("apply")),
        );
        let widest = d_in.max(d_out).max(r);
        let execution_band = accumulation_growth(2 * widest)
            * frobenius(teacher.write())
            * (frobenius(teacher.read()) + frobenius(one_sided.read()))
            * norm(&witness)
            + one_sided_band * norm(&witness);
        assert!(
            executed > execution_band,
            "the witness moved the executed output by {executed:.3e}, band {execution_band:.3e} (sup {value:.3e} ± {numerical_error:.3e})"
        );
    }

    /// The quotient merges settings on one `GL(r)` orbit and separates settings on
    /// different orbits: the teacher and its regauging form one class, and the
    /// one-sided change and its own regauging form another. Each class's certified
    /// gap sits inside its construction band plus the evaluation error.
    #[test]
    fn the_intervention_quotient_merges_gauge_orbits_and_separates_distinct_maps() {
        let mut rng = StdRng::seed_from_u64(295_402);
        let (d_in, d_out, r) = (6, 5, 3);
        let teacher = LinearPassthrough::new(uniform_matrix(&mut rng, r, d_in), uniform_matrix(&mut rng, d_out, r))
            .expect("finite factors");
        let first_change = uniform_matrix(&mut rng, r, r);
        let second_change = uniform_matrix(&mut rng, r, r);
        let one_sided = LinearPassthrough::new(fast_ab(&first_change, &teacher.read), teacher.write.clone())
            .expect("finite factors");
        let settings = vec![
            teacher.clone(),
            teacher.apply(first_change.view()).expect("invertible"),
            one_sided.clone(),
            one_sided.apply(second_change.view()).expect("invertible"),
        ];
        let classes = quotient_intervention_set(&settings).expect("comparable settings");
        let members: Vec<Vec<usize>> = classes.iter().map(|class| class.members.clone()).collect();
        assert_eq!(members, vec![vec![0, 1], vec![2, 3]]);
        let evaluation_bars: Vec<f64> = [(0, 1), (2, 3)]
            .iter()
            .map(|&(representative, member)| {
                exact_parts(&settings[representative].operator_difference(&settings[member]).expect("comparable")).1
            })
            .collect();
        let construction_bars = [regauge_band(&teacher, &first_change), regauge_band(&one_sided, &second_change)];
        for ((class, construction), evaluation) in classes.iter().zip(construction_bars).zip(evaluation_bars) {
            assert!(
                class.largest_gap <= construction + 2.0 * evaluation,
                "class {:?} gap {:.3e} exceeds its construction band {construction:.3e} and evaluation error {evaluation:.3e}",
                class.members,
                class.largest_gap
            );
        }
    }

    /// The orbit dimension `r² − (r − rank A)(r − rank B)` is charged from resolved
    /// ranks only. The positive control removes one rank from each factor with
    /// exact zeros, and the resolved side must drop by one while the shape bound
    /// stays.
    #[test]
    fn the_orbit_dimension_is_charged_from_resolved_ranks() {
        let mut rng = StdRng::seed_from_u64(295_403);
        let (d_in, d_out, r) = (6, 5, 3);
        let full = LinearPassthrough::new(uniform_matrix(&mut rng, r, d_in), uniform_matrix(&mut rng, d_out, r))
            .expect("finite factors");
        let family = full.family().expect("svd");
        assert_eq!(family.continuous.dimension(), r * r);
        assert_eq!(family.orbit_dimension, DimensionInterval { resolved: r * r, at_most: r * r });
        assert_eq!(family.real_coordinates_at_most(), r * (d_in + d_out) - r * r);

        let mut read = full.read.clone();
        read.row_mut(r - 1).fill(0.0);
        let mut write = full.write.clone();
        write.column_mut(0).fill(0.0);
        let deficient = LinearPassthrough::new(read, write).expect("finite factors");
        let family = deficient.family().expect("svd");
        assert_eq!(family.orbit_dimension, DimensionInterval { resolved: r * r - 1, at_most: r * r });
        assert!(!family.orbit_dimension.is_exact());
        assert_eq!(family.real_coordinates_at_most(), r * (d_in + d_out) - (r * r - 1));
    }

    fn native_mlp(units: &HiddenUnits, bias_out: &Array1<f64>) -> NativeMlp {
        NativeMlp::new(
            units.read_in.clone(),
            units.bias_in.clone(),
            units.write_out.clone(),
            bias_out.clone(),
            units.activation,
        )
        .expect("the fixture's shapes compose")
    }

    /// Entrywise first-order band of comparing the residual block
    /// `x + W₂ σ(W₁ x + b₁) + b₂` before and after a ReLU unit gauge change.
    ///
    /// Both executions compute one exact value. Each computed output is within
    /// `γ_{d + H + 6}` of it, relative to the absolute majorant
    /// `|W₂| (|W₁| |x| + |b₁|) + |b₂| + |x|`, because `|σ(z)| ≤ |z|` and ReLU is
    /// 1-Lipschitz. The operations counted per output entry are:
    /// - the scale of a row entry and of the bias;
    /// - a `d`-term inner product and the bias add;
    /// - the column's division;
    /// - one product inside an `H`-term sum;
    /// - the output bias and residual adds.
    ///
    /// The scale `s_i` multiplies and `1/s_i` divides the same term, so the majorant
    /// does not depend on the change.
    fn relu_gauge_band(units: &HiddenUnits, bias_out: &Array1<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        let (hidden, width) = units.read_in.dim();
        let pre = inputs.mapv(f64::abs).dot(&units.read_in.mapv(f64::abs).t()) + &units.bias_in.mapv(f64::abs);
        let majorant = pre.dot(&units.write_out.mapv(f64::abs).t()) + &bias_out.mapv(f64::abs) + &inputs.mapv(f64::abs);
        majorant * (2.0 * accumulation_growth(width + hidden + 6))
    }

    /// The ReLU unit gauge (positive scales and a relabelling) leaves the executed
    /// block within its band. The positive control is the same change under the
    /// exact GELU. `apply` refuses it, and executing the forced tensors moves the
    /// output past the same band.
    #[test]
    fn relu_unit_scales_keep_the_block_while_gelu_refuses_them_and_moves() {
        let mut rng = StdRng::seed_from_u64(295_404);
        let (width, hidden, rows) = (5, 7, 6);
        let teacher = HiddenUnits::new(
            uniform_matrix(&mut rng, hidden, width),
            uniform_vector(&mut rng, hidden),
            uniform_matrix(&mut rng, width, hidden),
            GaussianActivation::Relu,
        )
        .expect("finite tensors");
        let bias_out = uniform_vector(&mut rng, width);
        let inputs = uniform_matrix(&mut rng, rows, width);
        let change = UnitGaugeChange {
            permutation: random_permutation(&mut rng, hidden),
            scales: random_scales(&mut rng, hidden, false),
        };
        let regauged = teacher.apply(&change).expect("positive scales are in ReLU's group");
        let band = relu_gauge_band(&teacher, &bias_out, &inputs);
        let before = native_mlp(&teacher, &bias_out).execute(inputs.view()).expect("execute");
        let after = native_mlp(&regauged, &bias_out).execute(inputs.view()).expect("execute");
        for ((&b, &a), &limit) in before.iter().zip(after.iter()).zip(band.iter()) {
            assert!((a - b).abs() <= limit, "ReLU gauge moved an output by {:.3e}, band {limit:.3e}", (a - b).abs());
        }

        let gelu = HiddenUnits {
            activation: GaussianActivation::ExactGelu,
            ..teacher.clone()
        };
        assert!(matches!(gelu.apply(&change), Err(GaugeRefusal::NotInGroup { .. })));
        let forced = HiddenUnits {
            activation: GaussianActivation::ExactGelu,
            ..regauged.clone()
        };
        let gelu_before = native_mlp(&gelu, &bias_out).execute(inputs.view()).expect("execute");
        let gelu_after = native_mlp(&forced, &bias_out).execute(inputs.view()).expect("execute");
        let moved = gelu_before
            .iter()
            .zip(gelu_after.iter())
            .zip(band.iter())
            .filter(|((b, a), limit)| (**a - **b).abs() > **limit)
            .count();
        assert!(moved > 0, "rescaled GELU units must move the block past the band");

        let duplicate = UnitGaugeChange {
            permutation: vec![0; hidden],
            scales: Array1::ones(hidden),
        };
        assert_eq!(
            teacher.apply(&duplicate).err(),
            Some(GaugeRefusal::NotAPermutation { length: hidden })
        );
    }

    /// Exactly null units are charged their free coordinates and leave the orbit:
    /// - an unread unit frees `d + 1`;
    /// - a silent ReLU unit with `b < 0` frees `d_out + 1`;
    /// - a silent unit at `b = 0` frees `d_out`.
    ///
    /// Under GELU the `b < 0` unit is not silent, since `σ(b) ≠ 0`. The positive
    /// control is the random block, which has no null unit and a full orbit.
    #[test]
    fn null_units_leave_the_orbit_and_charge_their_free_coordinates() {
        let mut rng = StdRng::seed_from_u64(295_405);
        let (width, hidden) = (4, 6);
        let read_in = uniform_matrix(&mut rng, hidden, width);
        let bias_in = uniform_vector(&mut rng, hidden);
        let write_out = uniform_matrix(&mut rng, width, hidden);
        let parameters = hidden * (width + 1) + width * hidden;
        let random = HiddenUnits::new(read_in.clone(), bias_in.clone(), write_out.clone(), GaussianActivation::Relu)
            .expect("finite tensors")
            .family();
        assert_eq!(random.null_coordinates, 0);
        assert_eq!(random.orbit_dimension.resolved, hidden);
        assert_eq!(random.real_coordinates_at_most(), parameters - hidden);

        let (mut read_in, mut bias_in, mut write_out) = (read_in, bias_in, write_out);
        write_out.column_mut(0).fill(0.0);
        read_in.row_mut(1).fill(0.0);
        bias_in[1] = -0.5;
        read_in.row_mut(2).fill(0.0);
        bias_in[2] = 0.0;
        let relu = HiddenUnits::new(read_in.clone(), bias_in.clone(), write_out.clone(), GaussianActivation::Relu)
            .expect("finite tensors")
            .family();
        assert_eq!(relu.null_coordinates, (width + 1) + (width + 1) + width);
        assert_eq!(relu.orbit_dimension, DimensionInterval { resolved: hidden - 3, at_most: hidden - 3 });
        assert_eq!(relu.parameter_coordinates, parameters);
        let gelu = HiddenUnits::new(read_in, bias_in, write_out, GaussianActivation::ExactGelu)
            .expect("finite tensors")
            .family();
        assert_eq!(gelu.null_coordinates, (width + 1) + width);
        assert_eq!(gelu.orbit_dimension.resolved, 0);
        assert_eq!(gelu.continuous, ContinuousGauge::Trivial);
    }

    fn silu(t: f64) -> f64 {
        t / (1.0 + (-t).exp())
    }

    fn swiglu(units: &SwigluUnits, inputs: &Array2<f64>) -> Array2<f64> {
        let gated = inputs.dot(&units.gate.t()).mapv(silu) * &inputs.dot(&units.up.t());
        gated.dot(&units.down.t())
    }

    /// Entrywise first-order band of comparing `W_d [s(W_g x) ⊙ W_u x]` before and
    /// after an up/down unit gauge change.
    ///
    /// Each computed output is within `γ_{2d + H + 10}` of the common exact value,
    /// relative to `|W_d| [(|s(z)| + (5/4) |W_g| |x|) ⊙ |W_u| |x|]`. The operations
    /// counted are:
    /// - the gate and up inner products;
    /// - SiLU's exponential, add and divide, at most one ulp each;
    /// - the up scale, the Hadamard product and the down division;
    /// - one product inside an `H`-term sum.
    ///
    /// The gate term covers rounding of the gate inner product through SiLU's
    /// Lipschitz constant. `s′(t) = σ(t) + t σ(t)(1 − σ(t))`, and
    /// `|t| σ(t)(1 − σ(t)) = |t| / (4 cosh²(t/2)) ≤ |t| / (4 + t²) ≤ 1/4`, since
    /// `cosh² ≥ 1 + (t/2)²` and `4 + t² ≥ 4|t|`. So `|s′| ≤ 5/4`.
    fn swiglu_gauge_band(units: &SwigluUnits, inputs: &Array2<f64>) -> Array2<f64> {
        let (hidden, width) = units.gate.dim();
        let abs_inputs = inputs.mapv(f64::abs);
        let gate_mass = abs_inputs.dot(&units.gate.mapv(f64::abs).t());
        let up_mass = abs_inputs.dot(&units.up.mapv(f64::abs).t());
        let gated = inputs.dot(&units.gate.t()).mapv(|z| silu(z).abs()) + &(gate_mass * 1.25);
        let majorant = (gated * &up_mass).dot(&units.down.mapv(f64::abs).t());
        majorant * (2.0 * accumulation_growth(2 * width + hidden + 10))
    }

    /// The SwiGLU up/down gauge (signed scales and a relabelling) keeps the block
    /// within its band. The positive control scales the gate rows instead,
    /// compensated on down; SiLU is not homogeneous, so the output moves past the
    /// same band. A zero scale is refused.
    #[test]
    fn swiglu_up_and_down_scales_are_a_gauge_and_gate_scales_are_not() {
        let mut rng = StdRng::seed_from_u64(295_406);
        let (width, hidden, out, rows) = (5, 6, 4, 5);
        let teacher = SwigluUnits::new(
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, out, hidden),
        )
        .expect("finite tensors");
        let inputs = uniform_matrix(&mut rng, rows, width);
        let change = UnitGaugeChange {
            permutation: random_permutation(&mut rng, hidden),
            scales: random_scales(&mut rng, hidden, true),
        };
        let regauged = teacher.apply(&change).expect("nonzero scales are in the up read's group");
        let band = swiglu_gauge_band(&teacher, &inputs);
        let before = swiglu(&teacher, &inputs);
        let after = swiglu(&regauged, &inputs);
        for ((&b, &a), &limit) in before.iter().zip(after.iter()).zip(band.iter()) {
            assert!((a - b).abs() <= limit, "SwiGLU gauge moved an output by {:.3e}, band {limit:.3e}", (a - b).abs());
        }

        let mut gate = teacher.gate.clone();
        let mut down = teacher.down.clone();
        for (i, &scale) in change.scales.iter().enumerate() {
            gate.row_mut(i).mapv_inplace(|entry| entry * scale);
            down.column_mut(i).mapv_inplace(|entry| entry / scale);
        }
        let gate_scaled = SwigluUnits::new(gate, teacher.up.clone(), down).expect("finite tensors");
        let moved = swiglu(&gate_scaled, &inputs)
            .iter()
            .zip(before.iter())
            .zip(band.iter())
            .filter(|((a, b), limit)| (**a - **b).abs() > **limit)
            .count();
        assert!(moved > 0, "rescaled gate rows must move the block past the band");

        let mut zero = change.clone();
        zero.scales[2] = 0.0;
        assert!(matches!(teacher.apply(&zero), Err(GaugeRefusal::NotInGroup { index: 2, .. })));
    }

    /// A SwiGLU unit whose up row, gate row or down column is zero writes nothing:
    /// - a zero up row frees `d + d_out`;
    /// - a zero down column frees `2d`;
    /// - a unit with both a zero gate row and a zero column frees the larger of the
    ///   two.
    ///
    /// The positive control is the random block, with no null unit and a full
    /// orbit.
    #[test]
    fn swiglu_units_that_write_nothing_are_null() {
        let mut rng = StdRng::seed_from_u64(295_407);
        let (width, hidden, out) = (5, 6, 4);
        let (mut gate, mut up, mut down) = (
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, out, hidden),
        );
        let parameters = hidden * (2 * width + out);
        let random = SwigluUnits::new(gate.clone(), up.clone(), down.clone())
            .expect("finite tensors")
            .family();
        assert_eq!(random.null_coordinates, 0);
        assert_eq!(random.orbit_dimension.resolved, hidden);
        assert_eq!(random.parameter_coordinates, parameters);

        up.row_mut(0).fill(0.0);
        down.column_mut(1).fill(0.0);
        gate.row_mut(2).fill(0.0);
        down.column_mut(2).fill(0.0);
        let family = SwigluUnits::new(gate, up, down).expect("finite tensors").family();
        assert_eq!(family.null_coordinates, (width + out) + 2 * width + (2 * width).max(width + out));
        assert_eq!(family.orbit_dimension, DimensionInterval { resolved: hidden - 3, at_most: hidden - 3 });
        assert_eq!(family.continuous, ContinuousGauge::NonzeroDiagonal { order: hidden });
        assert_eq!(family.discrete, DiscreteGauge::UnitPermutations { units: hidden });
    }

    fn gain_reads(norm: &NormGain, normalized: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut outputs = normalized * &norm.gain;
        if let Some(bias) = &norm.bias {
            outputs += bias;
        }
        norm.reads.iter().map(|read| outputs.dot(&read.t())).collect()
    }

    /// Entrywise first-order band of the reads of `w ⊙ n + β` before and after a
    /// coordinate-scale change. The normalizer `n` is formed before the gain, so
    /// both sides read one fixed set of normalized rows.
    ///
    /// Each computed read is within `γ_{d + 5}` of the common exact value, relative
    /// to `|W_k| (|w| |n| + |β|)`. The operations counted are the gain and bias
    /// scales, the gain product, the bias add, the read's division and a `d`-term
    /// inner product.
    fn gain_gauge_band(norm: &NormGain, normalized: &Array2<f64>) -> Vec<Array2<f64>> {
        let width = norm.gain.len();
        let mut majorant = normalized.mapv(f64::abs) * &norm.gain.mapv(f64::abs);
        if let Some(bias) = &norm.bias {
            majorant += &bias.mapv(f64::abs);
        }
        norm.reads
            .iter()
            .map(|read| majorant.dot(&read.mapv(f64::abs).t()) * (2.0 * accumulation_growth(width + 5)))
            .collect()
    }

    /// A norm's coordinate scales fold into its reads within the band. The positive
    /// control scales the gain and bias without compensating the reads, and it
    /// moves past the same band.
    ///
    /// The exact null coordinates follow the family's rules:
    /// - a coordinate no read uses frees its gain and bias;
    /// - a coordinate with zero gain and bias frees every read's column.
    ///
    /// The random RMSNorm-shaped gain, with no bias, has none.
    #[test]
    fn norm_gain_scales_fold_into_the_reads_and_an_uncompensated_gain_moves() {
        let mut rng = StdRng::seed_from_u64(295_408);
        let (width, rows) = (6, 5);
        let norm = NormGain::new(
            random_scales(&mut rng, width, true),
            Some(uniform_vector(&mut rng, width)),
            vec![uniform_matrix(&mut rng, 3, width), uniform_matrix(&mut rng, 2, width)],
        )
        .expect("finite tensors");
        let normalized = uniform_matrix(&mut rng, rows, width);
        let scales = random_scales(&mut rng, width, true);
        let regauged = norm.apply(scales.view()).expect("nonzero scales are in the gain's group");
        let band = gain_gauge_band(&norm, &normalized);
        let before = gain_reads(&norm, &normalized);
        let after = gain_reads(&regauged, &normalized);
        for ((b, a), limit) in before.iter().zip(after.iter()).zip(band.iter()) {
            for ((&b, &a), &limit) in b.iter().zip(a.iter()).zip(limit.iter()) {
                assert!((a - b).abs() <= limit, "gain gauge moved a read by {:.3e}, band {limit:.3e}", (a - b).abs());
            }
        }

        let uncompensated = NormGain::new(
            &norm.gain * &scales,
            norm.bias.as_ref().map(|bias| bias * &scales),
            norm.reads.clone(),
        )
        .expect("finite tensors");
        let moved: usize = gain_reads(&uncompensated, &normalized)
            .iter()
            .zip(before.iter())
            .zip(band.iter())
            .map(|((a, b), limit)| {
                a.iter()
                    .zip(b.iter())
                    .zip(limit.iter())
                    .filter(|((a, b), limit)| (**a - **b).abs() > **limit)
                    .count()
            })
            .sum();
        assert!(moved > 0, "an uncompensated gain must move the reads past the band");
        let mut zero = scales.clone();
        zero[4] = 0.0;
        assert!(matches!(norm.apply(zero.view()), Err(GaugeRefusal::NotInGroup { index: 4, .. })));

        let read_rows = 5;
        let (mut gain, mut bias, mut reads) = (norm.gain.clone(), norm.bias.clone().expect("a bias"), norm.reads.clone());
        for read in reads.iter_mut() {
            read.column_mut(0).fill(0.0);
        }
        gain[1] = 0.0;
        bias[1] = 0.0;
        let family = NormGain::new(gain, Some(bias), reads).expect("finite tensors").family();
        assert_eq!(family.null_coordinates, 2 + read_rows);
        assert_eq!(family.orbit_dimension.resolved, width - 2);
        assert_eq!(family.parameter_coordinates, 2 * width + read_rows * width);
        let rms = NormGain::new(norm.gain.clone(), None, norm.reads.clone()).expect("finite tensors").family();
        assert_eq!(rms.null_coordinates, 0);
        assert_eq!(rms.orbit_dimension.resolved, width);
        assert_eq!(rms.real_coordinates_at_most(), width + read_rows * width - width);
    }

    /// The stream's group is the smallest its reads allow. Under a LayerNorm read,
    /// a reflection must fix or negate `1`: an exactly centred vector and a
    /// constant vector are accepted, and a generic vector is refused. The same
    /// generic vector is accepted under an RMSNorm read, which is the positive
    /// control of the refusal.
    #[test]
    fn residual_stream_groups_follow_the_reads() {
        let mut rng = StdRng::seed_from_u64(295_410);
        let width = 6;
        let linear = ResidualStreamGauge::new(width, &[ResidualRead::Linear, ResidualRead::Linear]).expect("reads");
        assert_eq!(linear.continuous(), ContinuousGauge::GeneralLinear { order: width });
        assert_eq!(linear.continuous().dimension(), width * width);
        let rms = ResidualStreamGauge::new(width, &[ResidualRead::Linear, ResidualRead::RmsNorm]).expect("reads");
        assert_eq!(rms.continuous(), ContinuousGauge::Orthogonal { order: width });
        assert_eq!(rms.continuous().dimension(), width * (width - 1) / 2);
        let layer = ResidualStreamGauge::new(width, &[ResidualRead::RmsNorm, ResidualRead::LayerNorm]).expect("reads");
        assert_eq!(layer.continuous(), ContinuousGauge::OrthogonalFixingOnes { order: width });
        assert_eq!(layer.continuous().dimension(), (width - 1) * (width - 2) / 2);
        assert_eq!(
            ResidualStreamGauge::new(width, &[]).err(),
            Some(GaugeRefusal::EmptyTensor {
                what: "residual reads"
            })
        );

        let centred = Array2::from_shape_vec((width, 1), vec![0.3, -0.3, 0.7, -0.7, 1.1, -1.1]).expect("shape");
        assert!(ResidualReflections::new(centred, &layer).is_ok());
        assert!(ResidualReflections::new(Array2::from_elem((width, 1), 0.4), &layer).is_ok());
        let generic = uniform_matrix(&mut rng, width, 1);
        assert!(matches!(
            ResidualReflections::new(generic.clone(), &layer),
            Err(GaugeRefusal::NotInGroup { index: 0, .. })
        ));
        assert!(ResidualReflections::new(generic, &rms).is_ok());
    }

    /// `w ⊙ P h ν(P h) + β` with `ν(g) = (mean(g²) + ε)^{-1/2}`, where `P` is the
    /// centring when `centred` (LayerNorm) and the identity otherwise (RMSNorm).
    fn stream_norm(h: ArrayView1<'_, f64>, gain: &Array1<f64>, bias: Option<&Array1<f64>>, epsilon: f64, centred: bool) -> Array1<f64> {
        let g = if centred {
            &h - h.mean().expect("nonempty")
        } else {
            h.to_owned()
        };
        let nu = 1.0 / (g.mapv(|entry| entry * entry).mean().expect("nonempty") + epsilon).sqrt();
        let mut out = g * nu * gain;
        if let Some(bias) = bias {
            out += bias;
        }
        out
    }

    /// First-order band of one logit `W_k (w ⊙ P h ν + β)` before and after a
    /// residual reflection carry, with `k` reflections of a width-`d` stream.
    ///
    /// Terms:
    /// - **Reflections.** One reflection of `y` rounds its coefficient
    ///   `2 vᵀy/‖v‖²` within `γ_{2d+3} · 2‖y‖/‖v‖` (Cauchy–Schwarz on `Σ|v_j y_j|`)
    ///   and its update within `γ_2 (‖y‖ + |c| ‖v‖)`, so within `5 γ_{2d+5} ‖y‖`.
    ///   Orthogonal steps keep norms, so `Q` rounds within `δ_Q = 5 k γ_{2d+5}`.
    /// - **Normalizer.** Centring rounds within `2 γ_{d+2} ‖h‖`. The mean square,
    ///   `ε`, the root, the scale, the gain and the bias add round within `γ_{2d+8}`,
    ///   relative to `M = |w|_∞ ν ‖h‖ + ‖β‖`, which majorizes the norm output. By
    ///   `‖∂(g ν(g))/∂g‖₂ ≤ ν`, a stream perturbation `Δ` moves the output by at most
    ///   `|w|_∞ ν ‖Δ‖`.
    /// - **Carried read and bias.** `diag(w)⁻¹ Q diag(w)` has norm at most
    ///   `κ_w = max |w_i| / min |w_i|`, and its two scalings add `2 γ_1`. So the carried read
    ///   and bias round within `κ_w (δ_Q + 2γ_1)` of their exact values, relative to
    ///   `‖W_k‖` and `‖β‖`.
    /// - **Read.** A `d`-term inner product rounds within `γ_d ‖W′_k‖ ‖y′‖`.
    ///
    /// Summing, with `‖W′_k‖ ≤ κ_w ‖W_k‖` and `‖y′‖ ≤ κ_w M`:
    /// `‖W_k‖ M [κ_w (2 κ_w (δ_Q + 2γ_1) + δ_Q + 3γ_{2d+8} + γ_d) + 3γ_{2d+8} + γ_d]`.
    fn stream_band(
        read: &Array2<f64>,
        gain: &Array1<f64>,
        bias: Option<&Array1<f64>>,
        token: ArrayView1<'_, f64>,
        epsilon: f64,
        centred: bool,
        reflections: usize,
    ) -> Array1<f64> {
        let width = gain.len();
        let delta_q = 5.0 * reflections as f64 * accumulation_growth(2 * width + 5);
        let gamma_one = accumulation_growth(1);
        let normalizer = accumulation_growth(2 * width + 8);
        let inner = accumulation_growth(width);
        let largest = gain.iter().fold(0.0_f64, |acc, entry| acc.max(entry.abs()));
        let smallest = gain.iter().fold(f64::INFINITY, |acc, entry| acc.min(entry.abs()));
        let kappa = largest / smallest;
        let g = if centred {
            &token - token.mean().expect("nonempty")
        } else {
            token.to_owned()
        };
        let nu = 1.0 / (g.mapv(|entry| entry * entry).mean().expect("nonempty") + epsilon).sqrt();
        let token_norm = token.dot(&token).sqrt();
        let bias_norm = bias.map_or(0.0, |bias| norm(bias));
        let majorant = largest * nu * token_norm + bias_norm;
        let factor = kappa * (2.0 * kappa * (delta_q + 2.0 * gamma_one) + delta_q + 3.0 * normalizer + inner)
            + 3.0 * normalizer
            + inner;
        read.rows().into_iter().map(|row| row.dot(&row).sqrt() * majorant * factor).collect()
    }

    /// A2's residual-stream half, through executed norms and logits. Reflections
    /// carried onto the embedding, the normed unembedding and the LayerNorm bias keep
    /// every logit within its band, under RMSNorm and under LayerNorm with
    /// reflections that fix `1`. The positive control is the shear
    /// `S = I + 1.5 e₀ e₁ᵀ`, carried the same way (`S E`, `W diag(w) S⁻¹ diag(w)⁻¹`,
    /// `diag(w) S diag(w)⁻¹ β`). It is invertible but not orthogonal, so `ν(S h) ≠ ν(h)`
    /// and the logits move past the same band. `ε = 10⁻⁶` is Qwen3's declared
    /// `rms_norm_eps`.
    #[test]
    fn a_reflected_residual_basis_keeps_normed_logits_and_a_shear_moves_them() {
        let mut rng = StdRng::seed_from_u64(295_409);
        let (width, vocab, reflections) = (6, 4, 3);
        let epsilon = 1e-6;
        for (centred, read) in [(false, ResidualRead::RmsNorm), (true, ResidualRead::LayerNorm)] {
            let gauge = ResidualStreamGauge::new(width, &[read]).expect("reads");
            let mut vectors = uniform_matrix(&mut rng, width, reflections);
            for mut column in vectors.columns_mut() {
                for pair in 0..width / 2 {
                    column[2 * pair + 1] = -column[2 * pair];
                }
            }
            let carry = ResidualReflections::new(vectors, &gauge).expect("exactly centred reflections");
            let embedding = uniform_matrix(&mut rng, width, vocab);
            let gain = random_scales(&mut rng, width, true);
            let bias = centred.then(|| uniform_vector(&mut rng, width));
            let unembed = uniform_matrix(&mut rng, vocab, width);
            let carried_embedding = carry.carry_write(embedding.view()).expect("carry write");
            let carried_unembed = carry.carry_normed_read(gain.view(), unembed.view()).expect("carry read");
            let carried_bias = bias
                .as_ref()
                .map(|bias| carry.carry_norm_bias(gain.view(), bias.view()).expect("carry bias"));

            let shear = 1.5;
            let mut sheared_embedding = embedding.clone();
            let row_one = embedding.row(1).to_owned();
            sheared_embedding.row_mut(0).scaled_add(shear, &row_one);
            let mut scaled_unembed = &unembed * &gain;
            let column_zero = scaled_unembed.column(0).to_owned();
            scaled_unembed.column_mut(1).scaled_add(-shear, &column_zero);
            let sheared_unembed = scaled_unembed / &gain;
            let sheared_bias = bias.as_ref().map(|bias| {
                let mut unscaled = bias / &gain;
                unscaled[0] += shear * unscaled[1];
                unscaled * &gain
            });

            let mut moved = 0;
            for token in 0..vocab {
                let before = unembed.dot(&stream_norm(embedding.column(token), &gain, bias.as_ref(), epsilon, centred));
                let after = carried_unembed.dot(&stream_norm(
                    carried_embedding.column(token),
                    &gain,
                    carried_bias.as_ref(),
                    epsilon,
                    centred,
                ));
                let band = stream_band(&unembed, &gain, bias.as_ref(), embedding.column(token), epsilon, centred, reflections);
                for ((&b, &a), &limit) in before.iter().zip(after.iter()).zip(band.iter()) {
                    assert!(
                        (a - b).abs() <= limit,
                        "centred={centred}: a reflected basis moved a logit by {:.3e}, band {limit:.3e}",
                        (a - b).abs()
                    );
                }
                let sheared = sheared_unembed.dot(&stream_norm(
                    sheared_embedding.column(token),
                    &gain,
                    sheared_bias.as_ref(),
                    epsilon,
                    centred,
                ));
                moved += before
                    .iter()
                    .zip(sheared.iter())
                    .zip(band.iter())
                    .filter(|((b, a), limit)| (**a - **b).abs() > **limit)
                    .count();
            }
            assert!(moved > 0, "centred={centred}: a sheared basis must move the logits past the band");
        }
    }

    fn half_split_rotary(frequencies: &[f64]) -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: frequencies.to_vec(),
            attention_scaling: 1.0,
        }
    }

    /// A native attention block with the given query and key projections, plus random
    /// value and output projections.
    fn native_block(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        query_weight: Array2<f64>,
        query_bias: Array1<f64>,
        key_weight: Array2<f64>,
        key_bias: Array1<f64>,
    ) -> NativeAttention {
        let mut rng = StdRng::seed_from_u64(295_416);
        NativeAttention::new(
            geometry,
            rotary,
            1.0 / (geometry.head_dim as f64).sqrt(),
            AffineProjection {
                weight: query_weight,
                bias: query_bias,
            },
            AffineProjection {
                weight: key_weight,
                bias: key_bias,
            },
            AffineProjection {
                weight: uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
                bias: uniform_vector(&mut rng, geometry.key_value_dim()),
            },
            AffineProjection {
                weight: uniform_matrix(&mut rng, geometry.model_dim, geometry.query_dim()),
                bias: uniform_vector(&mut rng, geometry.model_dim),
            },
        )
        .expect("the fixture's shapes compose")
    }

    /// The rotary family of a native block built from these projections.
    fn rotary_block(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        query_weight: Array2<f64>,
        query_bias: Array1<f64>,
        key_weight: Array2<f64>,
        key_bias: Array1<f64>,
    ) -> Result<RotaryQueryKey, GaugeRefusal> {
        RotaryQueryKey::new(&native_block(geometry, rotary, query_weight, query_bias, key_weight, key_bias))
    }

    /// `A + iB` on the complex coordinates `x_a + i x_b` of every rotated plane (rows
    /// `a`: `A` on `a`, `−B` on `b`; rows `b`: `B` on `a`, `A` on `b`), and a random
    /// `GL(n_pass)` block on the pass-through coordinates. Every entry is a copy or a
    /// negation of a draw, so the complex-linear relations hold bitwise.
    fn complex_plane_change(rng: &mut StdRng, rotary: &RotaryEmbedding, head_dim: usize) -> Array2<f64> {
        let planes = rotary.inverse_frequencies.len();
        let real = uniform_matrix(rng, planes, planes);
        let imaginary = uniform_matrix(rng, planes, planes);
        let mut change = Array2::<f64>::zeros((head_dim, head_dim));
        for i in 0..planes {
            for j in 0..planes {
                let (ai, bi) = rotary.plane(i);
                let (aj, bj) = rotary.plane(j);
                change[[ai, aj]] = real[[i, j]];
                change[[bi, bj]] = real[[i, j]];
                change[[bi, aj]] = imaginary[[i, j]];
                change[[ai, bj]] = -imaginary[[i, j]];
            }
        }
        let rotary_dim = rotary.rotary_dim();
        let pass = uniform_matrix(rng, head_dim - rotary_dim, head_dim - rotary_dim);
        change.slice_mut(s![rotary_dim.., rotary_dim..]).assign(&pass);
        change
    }

    /// The carry `S [W_K | b_K]`, `S⁻ᵀ [W_Q | b_Q]` without the membership check, for
    /// the forced positive control.
    fn forced_query_key_carry(teacher: &RotaryQueryKey, changes: &[Array2<f64>]) -> RotaryQueryKey {
        let g = teacher.geometry;
        let mut carried = teacher.clone();
        for (kv, change) in changes.iter().enumerate() {
            let inverse = invert_gauge_change(change.view()).expect("invertible");
            let rows = teacher.head_rows(kv);
            carried
                .key_weight
                .slice_mut(s![rows.clone(), ..])
                .assign(&fast_ab(change, &teacher.key_weight.slice(s![rows.clone(), ..])));
            carried
                .key_bias
                .slice_mut(s![rows.clone()])
                .assign(&fast_av(change, &teacher.key_bias.slice(s![rows])));
            for head in 0..g.n_heads {
                if g.key_value_head(head) == kv {
                    let rows = teacher.head_rows(head);
                    carried
                        .query_weight
                        .slice_mut(s![rows.clone(), ..])
                        .assign(&fast_ab(&inverse.t(), &teacher.query_weight.slice(s![rows.clone(), ..])));
                    carried
                        .query_bias
                        .slice_mut(s![rows.clone()])
                        .assign(&fast_av(&inverse.t(), &teacher.query_bias.slice(s![rows])));
                }
            }
        }
        carried
    }

    fn native_attention(block: &RotaryQueryKey, score_scale: f64, value: &AffineProjection, output: &AffineProjection) -> NativeAttention {
        NativeAttention::new(
            block.geometry,
            block.rotary.clone(),
            score_scale,
            AffineProjection {
                weight: block.query_weight.clone(),
                bias: block.query_bias.clone(),
            },
            AffineProjection {
                weight: block.key_weight.clone(),
                bias: block.key_bias.clone(),
            },
            value.clone(),
            output.clone(),
        )
        .expect("the fixture's shapes compose")
    }

    /// A complex-linear change on the planes of one frequency, with any invertible
    /// pass-through block, keeps every executed score within a bar.
    ///
    /// The bar is both executions' radii (attention.rs's own forward-error bars) plus
    /// the construction band
    /// `|σ| α² ‖(x_t, 1)‖ ‖(x_s, 1)‖ ‖[W_Q | b_Q]_h‖_F ‖[W_K | b_K]_g‖_F (η + 2 γ_hd κ_F)`.
    /// It is derived as for the pass-through, on the homogeneous reads, since
    /// `Ŝ⁻¹ R S − R = (Ŝ⁻¹ − S⁻¹) R S` meets `η` once. The biases are random, so a carry
    /// that dropped them would move the scores.
    ///
    /// The positive control is the same construction with distinct frequencies. Its
    /// cross-plane entries couple two commutant blocks, so `apply` refuses it, and the
    /// forced carry moves some score with `s < t` past the same bar. The diagonal
    /// `s = t` has `Δ = 0` and is invariant under every `S`.
    #[test]
    fn a_complex_linear_change_on_one_frequency_keeps_every_score_and_is_refused_across_frequencies() {
        let mut rng = StdRng::seed_from_u64(295_411);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let positions: Vec<i64> = vec![0, 1, 3, 4, 7];
        let score_scale = 1.0 / (geometry.head_dim as f64).sqrt();
        let query_weight = uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim);
        let query_bias = uniform_vector(&mut rng, geometry.query_dim());
        let key_weight = uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim);
        let key_bias = uniform_vector(&mut rng, geometry.key_value_dim());
        let value = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
            bias: uniform_vector(&mut rng, geometry.key_value_dim()),
        };
        let output = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.model_dim, geometry.query_dim()),
            bias: uniform_vector(&mut rng, geometry.model_dim),
        };
        let x = uniform_matrix(&mut rng, positions.len(), geometry.model_dim);
        let hd = geometry.head_dim;
        for (frequencies, shared) in [([0.7, 0.7], true), ([1.0, 0.3], false)] {
            let rotary = half_split_rotary(&frequencies);
            let teacher = rotary_block(
                geometry,
                rotary.clone(),
                query_weight.clone(),
                query_bias.clone(),
                key_weight.clone(),
                key_bias.clone(),
            )
            .expect("shapes");
            let mut changes = Vec::new();
            while changes.len() < geometry.n_kv_heads {
                changes.push(complex_plane_change(&mut rng, &rotary, hd));
            }
            let carried = if shared {
                teacher.apply(&changes).expect("complex-linear on one frequency")
            } else {
                assert!(matches!(teacher.apply(&changes), Err(GaugeRefusal::NotInGroup { .. })));
                forced_query_key_carry(&teacher, &changes)
            };
            let before = native_attention(&teacher, score_scale, &value, &output)
                .execute(x.view(), &positions)
                .expect("execute");
            let after = native_attention(&carried, score_scale, &value, &output)
                .execute(x.view(), &positions)
                .expect("execute");
            let mut moved = 0;
            for head in 0..geometry.n_heads {
                let kv = geometry.key_value_head(head);
                let inverse = invert_gauge_change(changes[kv].view()).expect("invertible");
                let kappa = frobenius(changes[kv].view()) * frobenius(inverse.view());
                let eta = kappa * hd as f64 * (f64::EPSILON + accumulation_growth(2 * hd));
                let query_rows = RotaryQueryKey::homogeneous_rows(&query_weight, &query_bias, teacher.head_rows(head)).expect("rows");
                let key_rows = RotaryQueryKey::homogeneous_rows(&key_weight, &key_bias, teacher.head_rows(kv)).expect("rows");
                let factors = frobenius(query_rows.view()) * frobenius(key_rows.view());
                for t in 0..positions.len() {
                    for s_index in 0..=t {
                        let (xt, xs) = (x.row(t), x.row(s_index));
                        let construction = score_scale.abs()
                            * (xt.dot(&xt) + 1.0).sqrt()
                            * (xs.dot(&xs) + 1.0).sqrt()
                            * factors
                            * (eta + 2.0 * accumulation_growth(hd) * kappa);
                        let bar =
                            before.score_radius[[head, t, s_index]] + after.score_radius[[head, t, s_index]] + construction;
                        let gap = (after.scores[[head, t, s_index]] - before.scores[[head, t, s_index]]).abs();
                        if shared {
                            assert!(gap <= bar, "head {head} ({t}, {s_index}): score moved {gap:.3e}, bar {bar:.3e}");
                        } else if s_index < t && gap > bar {
                            moved += 1;
                        }
                    }
                }
            }
            assert!(shared || moved > 0, "a cross-frequency coupling must move an off-diagonal score past the bar");
        }
    }

    /// The commutant dimension is `2m²` per equal-frequency group plus `n_pass²`: 12
    /// for two planes at one frequency with two pass-through coordinates, and 8 for two
    /// distinct frequencies.
    ///
    /// The orbit is charged per key/value head only when a key or query head's
    /// homogeneous rows resolve full row rank. The positive control zeros one key/value
    /// head and its query heads, weights and biases, and the resolved side must drop to
    /// one copy. Frequencies outside `(0, π)` are refused.
    #[test]
    fn the_rotary_commutant_counts_equal_frequency_groups_and_resolved_heads() {
        let mut rng = StdRng::seed_from_u64(295_412);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let mut query_weight = uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim);
        let mut query_bias = uniform_vector(&mut rng, geometry.query_dim());
        let mut key_weight = uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim);
        let mut key_bias = uniform_vector(&mut rng, geometry.key_value_dim());
        let block = |rotary: RotaryEmbedding, qw: &Array2<f64>, qb: &Array1<f64>, kw: &Array2<f64>, kb: &Array1<f64>| {
            rotary_block(geometry, rotary, qw.clone(), qb.clone(), kw.clone(), kb.clone())
        };
        for (frequencies, per_copy) in [([0.7, 0.7], 12), ([1.0, 0.3], 8)] {
            let family = block(half_split_rotary(&frequencies), &query_weight, &query_bias, &key_weight, &key_bias)
                .expect("shapes")
                .family()
                .expect("svd");
            assert_eq!(family.continuous.dimension(), 2 * per_copy);
            assert_eq!(family.orbit_dimension, DimensionInterval { resolved: 2 * per_copy, at_most: 2 * per_copy });
            assert_eq!(family.parameter_coordinates, 6 * 6 * 9);
        }
        key_weight.slice_mut(s![6..12, ..]).fill(0.0);
        key_bias.slice_mut(s![6..12]).fill(0.0);
        query_weight.slice_mut(s![12..24, ..]).fill(0.0);
        query_bias.slice_mut(s![12..24]).fill(0.0);
        let family = block(half_split_rotary(&[0.7, 0.7]), &query_weight, &query_bias, &key_weight, &key_bias)
            .expect("shapes")
            .family()
            .expect("svd");
        assert_eq!(family.orbit_dimension, DimensionInterval { resolved: 12, at_most: 24 });
        for frequency in [std::f64::consts::PI, 0.0] {
            assert!(matches!(
                block(half_split_rotary(&[0.7, frequency]), &query_weight, &query_bias, &key_weight, &key_bias),
                Err(GaugeRefusal::FrequencyOutsideHalfTurn { plane: 1, .. })
            ));
        }
    }

    /// A native block without a query/key norm yields the rotary family. The positive
    /// control is the same block after `with_query_key_norm`, which is refused with
    /// `QueryKeyNorm`, because behind that norm the commutant overclaims the gauge.
    /// `ε = 10⁻⁶` is Qwen3's declared `rms_norm_eps`.
    #[test]
    fn a_native_block_with_a_query_key_norm_is_refused() {
        let mut rng = StdRng::seed_from_u64(295_417);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let native = native_block(
            geometry,
            half_split_rotary(&[0.7, 0.7]),
            uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim),
            uniform_vector(&mut rng, geometry.query_dim()),
            uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
            uniform_vector(&mut rng, geometry.key_value_dim()),
        );
        assert!(RotaryQueryKey::new(&native).is_ok());
        let normed = native
            .with_query_key_norm(1e-6, Array1::ones(geometry.head_dim), Array1::ones(geometry.head_dim))
            .expect("gains of head width");
        assert_eq!(RotaryQueryKey::new(&normed).err(), Some(GaugeRefusal::QueryKeyNorm));
    }

    /// A normed block with its value and output projections, inputs and positions.
    struct NormedFixture {
        block: NormedRotaryQueryKey,
        value: AffineProjection,
        output: AffineProjection,
        score_scale: f64,
        inputs: Array2<f64>,
        positions: Vec<i64>,
    }

    impl NormedFixture {
        /// The source block with `block`'s projections and norm.
        fn execute(&self, block: &NormedRotaryQueryKey) -> AttentionExecution {
            NativeAttention::new(
                block.geometry,
                block.rotary.clone(),
                self.score_scale,
                AffineProjection {
                    weight: block.query_weight.clone(),
                    bias: block.query_bias.clone(),
                },
                AffineProjection {
                    weight: block.key_weight.clone(),
                    bias: block.key_bias.clone(),
                },
                self.value.clone(),
                self.output.clone(),
            )
            .expect("the fixture's shapes compose")
            .with_query_key_norm(block.epsilon, block.query_gain.clone(), block.key_gain.clone())
            .expect("gains of head width")
            .execute(self.inputs.view(), &self.positions)
            .expect("execute")
        }
    }

    /// Three planes at distinct frequencies filling a 6-coordinate head, random projections
    /// and random signed gains of magnitude in `[½, 2]`, behind Qwen3's `ε = 10⁻⁶`. On
    /// plane `coincident`, if given, `w_b = −w_a` and `v_b = v_a`.
    fn normed_fixture(seed: u64, coincident: Option<usize>) -> NormedFixture {
        let mut rng = StdRng::seed_from_u64(seed);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let rotary = half_split_rotary(&[0.9, 0.5, 0.2]);
        let mut query_gain = random_scales(&mut rng, geometry.head_dim, true);
        let mut key_gain = random_scales(&mut rng, geometry.head_dim, true);
        if let Some(plane) = coincident {
            let (a, b) = rotary.plane(plane);
            query_gain[b] = -query_gain[a];
            key_gain[b] = key_gain[a];
        }
        let score_scale = 1.0 / (geometry.head_dim as f64).sqrt();
        let query = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim),
            bias: uniform_vector(&mut rng, geometry.query_dim()),
        };
        let key = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
            bias: uniform_vector(&mut rng, geometry.key_value_dim()),
        };
        let value = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
            bias: uniform_vector(&mut rng, geometry.key_value_dim()),
        };
        let output = AffineProjection {
            weight: uniform_matrix(&mut rng, geometry.model_dim, geometry.query_dim()),
            bias: uniform_vector(&mut rng, geometry.model_dim),
        };
        let native = NativeAttention::new(geometry, rotary, score_scale, query, key, value.clone(), output.clone())
            .expect("the fixture's shapes compose")
            .with_query_key_norm(1e-6, query_gain, key_gain)
            .expect("gains of head width");
        NormedFixture {
            block: NormedRotaryQueryKey::new(&native).expect("a covered normed block"),
            value,
            output,
            score_scale,
            inputs: uniform_matrix(&mut rng, 5, geometry.model_dim),
            positions: vec![0, 1, 3, 4, 7],
        }
    }

    fn same_bits<'a>(left: impl IntoIterator<Item = &'a f64>, right: impl IntoIterator<Item = &'a f64>) -> bool {
        left.into_iter().zip(right).all(|(p, q)| p.to_bits() == q.to_bits())
    }

    /// The causal scores of `after` farther from `before`'s than both executions' radii.
    /// Each radius is attention.rs's forward-error bar against its own block's exact
    /// scores. So when the two blocks have the same exact scores, no score lies past the
    /// sum of the two radii.
    fn scores_past_radii(before: &AttentionExecution, after: &AttentionExecution) -> usize {
        before
            .scores
            .iter()
            .zip(after.scores.iter())
            .zip(before.score_radius.iter().zip(after.score_radius.iter()))
            .filter(|((p, q), (r, s))| p.is_finite() && (**p - **q).abs() > **r + **s)
            .count()
    }

    fn no_flips(width: usize) -> (Vec<bool>, Vec<bool>) {
        (vec![false; width], vec![false; width])
    }

    /// The positive control: plane scales that are powers of two, per-head sign turns
    /// (`t = 2`) and gain sign flips are in the normed family, and each carried tensor is
    /// exact. A power of two scales without rounding, a sign is exact, and `ν` sees only
    /// `‖u‖²`, which a sign keeps bit for bit. So the carried block executes the same
    /// scores, weights and output bit for bit. The control on the bitwise comparison is
    /// the teacher with one query gain moved by one ulp, which changes some score's bits.
    #[test]
    fn a_normed_block_reproduces_bit_for_bit_under_power_of_two_scales_and_signs() {
        let fixture = normed_fixture(295_418, None);
        let change = NormedRotaryChange {
            plane_scales: Array1::from(vec![2.0, 0.5, 4.0]),
            quarter_turns: Array2::from_shape_vec((2, 3), vec![0, 2, 0, 2, 2, 0]).expect("shape"),
            query_gain_flips: vec![true, false, false, true, false, false],
            key_gain_flips: vec![false, true, false, false, false, true],
        };
        let carried = fixture.block.apply(&change).expect("in the normed family");
        let before = fixture.execute(&fixture.block);
        let after = fixture.execute(&carried);
        assert!(same_bits(&before.scores, &after.scores), "the scores must reproduce bit for bit");
        assert!(same_bits(&before.weights, &after.weights), "the weights must reproduce bit for bit");
        assert!(same_bits(&before.output, &after.output), "the output must reproduce bit for bit");
        let mut nudged = fixture.block.clone();
        nudged.query_gain[0] = nudged.query_gain[0].next_up();
        assert!(!same_bits(&before.scores, &fixture.execute(&nudged).scores), "a one-ulp gain change must show in the bits");
    }

    /// Odd quarter turns on every head of a plane whose gains are not coincident, with the
    /// gains swapped, and a quarter turn on one key/value head of the coincident plane are
    /// in the normed family. The carried tensors are exact: signed row permutations and
    /// power-of-two gain scales. So both blocks have the same exact scores, and every
    /// computed score agrees within the sum of both radii. The family counts one scale per
    /// plane and one rotation per key/value head on the coincident plane.
    #[test]
    fn quarter_turns_and_a_coincident_plane_turn_keep_every_score_within_both_radii() {
        let fixture = normed_fixture(295_419, Some(1));
        let family = fixture.block.family();
        assert_eq!(
            family.continuous,
            ContinuousGauge::NormedRotary {
                plane_scales: 3,
                plane_rotations: 2
            }
        );
        assert_eq!(
            family.discrete,
            DiscreteGauge::NormedRotary {
                generic_planes: 2,
                coincident_planes: 1,
                key_value_heads: 2
            }
        );
        assert_eq!(family.orbit_dimension, DimensionInterval { resolved: 5, at_most: 5 });
        assert_eq!(family.parameter_coordinates, 6 * 6 * 9 + 12);
        assert_eq!(
            (0..3).map(|plane| fixture.block.coincident(plane)).collect::<Vec<_>>(),
            vec![false, true, false]
        );
        let (query_gain_flips, key_gain_flips) = no_flips(6);
        let change = NormedRotaryChange {
            plane_scales: Array1::from(vec![1.0, 2.0, 0.5]),
            quarter_turns: Array2::from_shape_vec((2, 3), vec![1, 1, 3, 3, 0, 1]).expect("shape"),
            query_gain_flips,
            key_gain_flips,
        };
        let carried = fixture.block.apply(&change).expect("in the normed family");
        let (a, b) = fixture.block.rotary.plane(0);
        assert_eq!(
            (carried.query_gain[a], carried.query_gain[b]),
            (fixture.block.query_gain[b], fixture.block.query_gain[a]),
            "odd turns on a plane that is not coincident swap its query gains"
        );
        let past = scores_past_radii(&fixture.execute(&fixture.block), &fixture.execute(&carried));
        assert_eq!(past, 0, "{past} scores moved past both radii under a change in the family");
    }

    /// The negative controls: outside the normed family the scores move.
    /// - A quarter turn on one key/value head of a plane whose gains are not coincident,
    ///   with the gains kept, is the rotary commutant's own change. `RotaryQueryKey::apply`
    ///   admits it on the same projections without the norm. The normed `apply` refuses it,
    ///   and carried by hand (`u ↦ R u` on that head's pre-norm rows) it moves some score
    ///   past both radii.
    /// - So does a gain scale that differs between a plane's two coordinates.
    ///
    /// Both carries are exact, so the radii are the whole bar. The same one-head quarter
    /// turn on the coincident plane, in the previous test, is the matching positive control.
    #[test]
    fn a_rotary_commutant_turn_outside_the_normed_family_moves_a_score() {
        let fixture = normed_fixture(295_419, Some(1));
        let block = &fixture.block;
        let before = fixture.execute(block);
        let (query_gain_flips, key_gain_flips) = no_flips(6);
        let one_head = NormedRotaryChange {
            plane_scales: Array1::ones(3),
            quarter_turns: Array2::from_shape_vec((2, 3), vec![1, 0, 0, 0, 0, 0]).expect("shape"),
            query_gain_flips,
            key_gain_flips,
        };
        assert!(matches!(block.apply(&one_head), Err(GaugeRefusal::NotInGroup { .. })));
        let (a, b) = block.rotary.plane(0);
        let mut quarter = Array2::<f64>::eye(6);
        quarter[[a, a]] = 0.0;
        quarter[[b, b]] = 0.0;
        quarter[[a, b]] = -1.0;
        quarter[[b, a]] = 1.0;
        let unnormed = RotaryQueryKey {
            geometry: block.geometry,
            rotary: block.rotary.clone(),
            query_weight: block.query_weight.clone(),
            query_bias: block.query_bias.clone(),
            key_weight: block.key_weight.clone(),
            key_bias: block.key_bias.clone(),
        };
        assert!(unnormed.apply(&[quarter, Array2::eye(6)]).is_ok(), "the quarter turn lies in the rotary commutant");
        let turn_rows = |weight: &mut Array2<f64>, bias: &mut Array1<f64>, head: usize| {
            let (ra, rb) = (head * 6 + a, head * 6 + b);
            let (row_a, row_b) = (weight.row(ra).to_owned(), weight.row(rb).to_owned());
            weight.row_mut(ra).assign(&row_b.mapv(|entry| -entry));
            weight.row_mut(rb).assign(&row_a);
            let (bias_a, bias_b) = (bias[ra], bias[rb]);
            bias[ra] = -bias_b;
            bias[rb] = bias_a;
        };
        let mut forced = block.clone();
        turn_rows(&mut forced.key_weight, &mut forced.key_bias, 0);
        for head in (0..block.geometry.n_heads).filter(|&head| block.geometry.key_value_head(head) == 0) {
            turn_rows(&mut forced.query_weight, &mut forced.query_bias, head);
        }
        let moved = scores_past_radii(&before, &fixture.execute(&forced));
        assert!(moved > 0, "the rotary commutant's quarter turn must move a score behind the norm");
        let (c, _) = block.rotary.plane(2);
        let mut unequal = block.clone();
        unequal.query_gain[c] *= 2.0;
        unequal.key_gain[c] /= 2.0;
        let moved = scores_past_radii(&before, &fixture.execute(&unequal));
        assert!(moved > 0, "a gain scale that differs within a plane must move a score");
    }

    /// Only the norm variants outside the derivation are refused, each by name.
    /// [`QueryKeyGauge::new`] reads a covered normed block as the normed family, and a
    /// block without the norm as the rotary commutant.
    #[test]
    fn the_normed_family_refuses_by_name_each_norm_variant_its_derivation_does_not_cover() {
        let mut rng = StdRng::seed_from_u64(295_420);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let mut native = |geometry: AttentionGeometry, frequencies: &[f64]| {
            native_block(
                geometry,
                half_split_rotary(frequencies),
                uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.query_dim()),
                uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.key_value_dim()),
            )
        };
        let gains = Array1::from(vec![1.5, -0.7, 0.9, 1.2, -1.9, 0.6]);
        let normed = |plain: NativeAttention, epsilon: f64, key_gain: Array1<f64>| {
            plain.with_query_key_norm(epsilon, gains.clone(), key_gain).expect("gains of head width")
        };
        let full = native(geometry, &[0.9, 0.5, 0.2]);
        assert!(matches!(QueryKeyGauge::new(&full), Ok(QueryKeyGauge::Rotary(_))));
        assert_eq!(NormedRotaryQueryKey::new(&full).err(), Some(GaugeRefusal::NoQueryKeyNorm));
        let covered = normed(full.clone(), 1e-6, gains.clone());
        assert!(matches!(QueryKeyGauge::new(&covered), Ok(QueryKeyGauge::NormedRotary(_))));
        assert_eq!(RotaryQueryKey::new(&covered).err(), Some(GaugeRefusal::QueryKeyNorm));
        assert_eq!(
            NormedRotaryQueryKey::new(&normed(full.clone(), 0.0, gains.clone())).err(),
            Some(GaugeRefusal::QueryKeyNormEpsilon { epsilon: 0.0 })
        );
        let mut zero = gains.clone();
        zero[4] = 0.0;
        assert_eq!(
            NormedRotaryQueryKey::new(&normed(full, 1e-6, zero)).err(),
            Some(GaugeRefusal::QueryKeyNormZeroGain {
                side: QueryKeySide::Key,
                coordinate: 4
            })
        );
        assert_eq!(
            QueryKeyGauge::new(&normed(native(geometry, &[0.9, 0.4]), 1e-6, gains.clone())).err(),
            Some(GaugeRefusal::QueryKeyNormPassThrough { coordinates: 2 })
        );
        assert_eq!(
            QueryKeyGauge::new(&normed(native(geometry, &[0.7, 0.7, 0.3]), 1e-6, gains.clone())).err(),
            Some(GaugeRefusal::QueryKeyNormSharedFrequency { first: 0, second: 1 })
        );
        let narrow = AttentionGeometry { model_dim: 4, ..geometry };
        assert_eq!(
            QueryKeyGauge::new(&normed(native(narrow, &[0.9, 0.5, 0.2]), 1e-6, gains.clone())).err(),
            Some(GaugeRefusal::QueryKeyNormHeadRank {
                side: QueryKeySide::Query,
                head: 0,
                resolved: 4,
                order: 6
            })
        );
    }

    /// The owner's witnesses for one-plane groups are normed changes. Scale 2 on a plane is
    /// `ρ_p = 2`, and `J` on a plane is one quarter turn on every key/value head. The normed
    /// family declares the rotary commutant, and `apply` admits both witnesses.
    #[test]
    fn every_rotary_witness_is_a_normed_family_change() {
        let fixture = normed_fixture(295_421, None);
        let declared = fixture.block.declared_gauge().expect("a valid rotary declaration");
        assert_eq!(
            declared,
            DeclaredGauge::RotaryCommutant(RotaryGroups::new(vec![vec![(0, 3)], vec![(1, 4)], vec![(2, 5)]], 6..6).expect("groups"))
        );
        let identity = Array2::<f64>::eye(6);
        let mut coupling = identity.clone();
        coupling[[0, 1]] = 0.3;
        let mut not_complex = identity.clone();
        not_complex[[3, 0]] = 0.4;
        not_complex[[0, 3]] = 0.4;
        for mask in [&coupling, &not_complex] {
            let witness = basis_dependent_witness(mask, &declared);
            let change = normed_change_from_head_map(&witness, &fixture.block.rotary, 2);
            assert!(fixture.block.apply(&change).is_ok(), "the owner's witness must be a normed change");
        }
    }

    /// A head map that is a signed scale or a signed quarter turn on each plane, as the
    /// same change on every key/value head.
    fn normed_change_from_head_map(map: &Array2<f64>, rotary: &RotaryEmbedding, key_value_heads: usize) -> NormedRotaryChange {
        let planes = rotary.inverse_frequencies.len();
        let mut plane_scales = Array1::<f64>::ones(planes);
        let mut quarter_turns = Array2::<u8>::zeros((key_value_heads, planes));
        for plane in 0..planes {
            let (a, b) = rotary.plane(plane);
            let (alpha, beta) = (map[[a, a]], map[[b, a]]);
            assert!(
                alpha == map[[b, b]] && beta == -map[[a, b]] && alpha * beta == 0.0,
                "plane {plane}: the witness is not a signed scale or quarter turn"
            );
            let (scale, turn) = if beta == 0.0 {
                (alpha.abs(), if alpha > 0.0 { 0 } else { 2 })
            } else {
                (beta.abs(), if beta > 0.0 { 1 } else { 3 })
            };
            plane_scales[plane] = scale;
            quarter_turns.column_mut(plane).fill(turn);
        }
        NormedRotaryChange {
            plane_scales,
            quarter_turns,
            query_gain_flips: vec![false; map.nrows()],
            key_gain_flips: vec![false; map.nrows()],
        }
    }

    /// `W₂ M σ(W₁ x + b₁)` for each row: a fixed internal mask `M` on the hidden units.
    fn masked_hidden_write(units: &HiddenUnits, mask: &Array2<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        let native = native_mlp(units, &Array1::zeros(units.write_out.nrows()));
        let summed = native.summed_input(inputs.view()).expect("shape");
        let hidden = native.activate(summed.view()).expect("activation");
        hidden.dot(&mask.t()).dot(&units.write_out.t())
    }

    /// Entrywise band of comparing `W₂ M σ(W₁ x + b₁)` before and after a ReLU unit change
    /// whose scale ratio is `kappa = max s / min s`.
    ///
    /// Each execution rounds:
    /// - the `d`-term summed input and its bias add;
    /// - the `H`-term mask product and the `H`-term write;
    /// - the change's three scalings.
    ///
    /// So each is within `γ_{d + 2H + 4}` of its exact value, relative to its absolute
    /// majorant. The teacher's majorant is `A = |W₂| |M| (|W₁| |x| + |b₁|)`, using
    /// `|σ(z)| ≤ |z|`. The changed units meet `M` through `s_i` and `1/s_j`, so their
    /// majorant is at most `kappa · A`. The band is `γ_{d + 2H + 4} (1 + kappa) A`.
    fn masked_band(units: &HiddenUnits, mask: &Array2<f64>, inputs: &Array2<f64>, kappa: f64) -> Array2<f64> {
        let (hidden, width) = units.read_in.dim();
        let pre = inputs.mapv(f64::abs).dot(&units.read_in.mapv(f64::abs).t()) + &units.bias_in.mapv(f64::abs);
        let majorant = pre.dot(&mask.mapv(f64::abs).t()).dot(&units.write_out.mapv(f64::abs).t());
        majorant * (accumulation_growth(width + 2 * hidden + 4) * (1.0 + kappa))
    }

    fn scale_ratio(scales: &Array1<f64>) -> f64 {
        scales.fold(0.0_f64, |largest, &scale| largest.max(scale)) / scales.fold(f64::INFINITY, |smallest, &scale| smallest.min(scale))
    }

    /// A witness in a scaled-permutation group as a unit change.
    ///
    /// P1's gauge change is `U → U S`, `V → V S⁻ᵀ` with the mask held fixed. For hidden
    /// units that means `W₂ → W₂ S` and `h → S⁻¹ h`, so the relabelled write is
    /// `W₂ S M S⁻¹ h`. A scaled permutation with column `j` nonzero at row `τ(j)`, value
    /// `σ_j`, has `S⁻¹` sending `e_{τ(j)}` to `e_j / σ_j`. The new unit `j` is therefore old
    /// unit `τ(j)` scaled by `1/σ_j`, which is [`HiddenUnits::apply`]'s convention.
    fn unit_change_from_witness(witness: &Array2<f64>) -> UnitGaugeChange {
        let units = witness.nrows();
        let mut permutation = Vec::with_capacity(units);
        let mut scales = Array1::<f64>::zeros(units);
        for column in 0..units {
            let old = (0..units)
                .position(|row| witness[[row, column]] != 0.0)
                .expect("a scaled permutation has one nonzero per column");
            permutation.push(old);
            scales[column] = 1.0 / witness[[old, column]];
        }
        UnitGaugeChange { permutation, scales }
    }

    fn basis_dependent_witness(mask: &Array2<f64>, gauge: &DeclaredGauge) -> Array2<f64> {
        match classify_under(mask.view(), gauge).expect("finite mask") {
            MaskGaugeVerdict::BasisDependent { gauge_change, .. } => gauge_change,
            other => panic!("the fixture mask is basis dependent, got {other:?}"),
        }
    }

    /// Each family declares the P1 group with its commutant:
    /// - ReLU and SwiGLU scaled relabellings: `ScaledPermutations`;
    /// - GELU relabellings: `UnitPermutations`;
    /// - norm gains: `Blocks([1; d])`;
    /// - pass-through, and the linear or RMSNorm stream: `Blocks([r])`;
    /// - the LayerNorm stream: `UnitPermutations`;
    /// - rotary: the plane groups of equal frequency with the pass-through range.
    #[test]
    fn families_declare_the_group_with_their_commutant() {
        let mut rng = StdRng::seed_from_u64(295_413);
        let (width, hidden) = (4, 5);
        let blocks = |sizes: &[usize]| Some(DeclaredGauge::Blocks(GaugeBlocks::new(sizes).expect("nonempty blocks")));
        let relu = HiddenUnits::new(
            uniform_matrix(&mut rng, hidden, width),
            uniform_vector(&mut rng, hidden),
            uniform_matrix(&mut rng, width, hidden),
            GaussianActivation::Relu,
        )
        .expect("finite tensors");
        assert_eq!(relu.family().declared_gauge(), Ok(Some(DeclaredGauge::ScaledPermutations { units: hidden })));
        let gelu = HiddenUnits {
            activation: GaussianActivation::ExactGelu,
            ..relu.clone()
        };
        assert_eq!(gelu.family().declared_gauge(), Ok(Some(DeclaredGauge::UnitPermutations { units: hidden })));
        let swiglu = SwigluUnits::new(
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, hidden, width),
            uniform_matrix(&mut rng, width, hidden),
        )
        .expect("finite tensors");
        assert_eq!(swiglu.family().declared_gauge(), Ok(Some(DeclaredGauge::ScaledPermutations { units: hidden })));
        let norm = NormGain::new(random_scales(&mut rng, width, true), None, vec![uniform_matrix(&mut rng, 3, width)])
            .expect("finite tensors");
        assert_eq!(norm.family().declared_gauge(), Ok(blocks(&vec![1; width])));
        let passthrough = LinearPassthrough::new(uniform_matrix(&mut rng, 3, width), uniform_matrix(&mut rng, width, 3))
            .expect("finite factors");
        assert_eq!(passthrough.family().expect("svd").declared_gauge(), Ok(blocks(&[3])));
        for (read, expected) in [
            (ResidualRead::Linear, blocks(&[width])),
            (ResidualRead::RmsNorm, blocks(&[width])),
            (ResidualRead::LayerNorm, Some(DeclaredGauge::UnitPermutations { units: width })),
        ] {
            assert_eq!(ResidualStreamGauge::new(width, &[read]).expect("reads").declared_gauge(), Ok(expected));
        }
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        for (frequencies, groups) in [([0.7, 0.7], vec![vec![(0, 2), (1, 3)]]), ([1.0, 0.3], vec![vec![(0, 2)], vec![(1, 3)]])] {
            let block = rotary_block(
                geometry,
                half_split_rotary(&frequencies),
                uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.query_dim()),
                uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.key_value_dim()),
            )
            .expect("shapes");
            assert_eq!(
                block.declared_gauge(),
                Ok(DeclaredGauge::RotaryCommutant(RotaryGroups::new(groups, 4..6).expect("groups")))
            );
        }
    }

    /// The executed transfer under ReLU's `ScaledPermutations`.
    ///
    /// The declared scalar mask `cI` is `Intrinsic` with controls `[c]`, and its write stays
    /// within the band under a random positive relabelling. The owner's witness for a
    /// diagonal mask with one unequal entry (a transposition) and for an off-diagonal mask
    /// (scale 2) are both admitted by `HiddenUnits::apply`, so they lie in ReLU's group.
    /// Each moves the fixed mask's write past the band, which is the positive control for
    /// the invariance bar.
    ///
    /// A witness moves `W₂ (S M S⁻¹ − M) σ(W₁ x + b₁)`, which reads only the units in the
    /// columns it changes. A unit that is off on every row would hide that change, and the
    /// positive control would pass or fail on the draw. So unit `j` also reads its own
    /// direction on one row, `x_j = c_j w_j` with `c_j = (1 + |b_j|)/‖w_j‖²`, where
    /// `w_j · x_j + b_j = 1 + |b_j| + b_j ≥ 1`. The random rows still cross the kink.
    #[test]
    fn declared_witnesses_are_admitted_by_the_family_and_move_a_fixed_mask() {
        let mut rng = StdRng::seed_from_u64(295_414);
        let (width, hidden, rows) = (4, 5, 4);
        let relu = HiddenUnits::new(
            uniform_matrix(&mut rng, hidden, width),
            uniform_vector(&mut rng, hidden),
            uniform_matrix(&mut rng, width, hidden),
            GaussianActivation::Relu,
        )
        .expect("finite tensors");
        let declared = relu
            .family()
            .declared_gauge()
            .expect("declaration")
            .expect("ReLU units are declared");
        let firing = Array2::from_shape_fn((hidden, width), |(unit, column)| {
            let read = relu.read_in.row(unit);
            read[column] * (1.0 + relu.bias_in[unit].abs()) / read.dot(&read)
        });
        let inputs = concatenate(Axis(0), &[uniform_matrix(&mut rng, rows, width).view(), firing.view()])
            .expect("equal widths");
        let summed = inputs.dot(&relu.read_in.t()) + &relu.bias_in;
        for unit in 0..hidden {
            assert!(summed.column(unit).iter().any(|&z| z > 0.0), "unit {unit} must fire on its own row");
        }

        let scalar = Array2::<f64>::eye(hidden) * 0.6;
        assert_eq!(
            classify_under(scalar.view(), &declared),
            Ok(MaskGaugeVerdict::Intrinsic { controls: vec![0.6] })
        );
        let change = UnitGaugeChange {
            permutation: random_permutation(&mut rng, hidden),
            scales: random_scales(&mut rng, hidden, false),
        };
        let relabelled = relu.apply(&change).expect("positive scales are in ReLU's group");
        let band = masked_band(&relu, &scalar, &inputs, scale_ratio(&change.scales));
        let before = masked_hidden_write(&relu, &scalar, &inputs);
        let after = masked_hidden_write(&relabelled, &scalar, &inputs);
        for ((&b, &a), &limit) in before.iter().zip(after.iter()).zip(band.iter()) {
            assert!((a - b).abs() <= limit, "a relabelling moved a cI-masked write by {:.3e}, band {limit:.3e}", (a - b).abs());
        }

        let mut diagonal = Array2::<f64>::eye(hidden);
        diagonal[[1, 1]] = 0.25;
        let mut off_diagonal = Array2::<f64>::eye(hidden);
        off_diagonal[[0, 1]] = 0.3;
        for (kind, mask) in [("diagonal", diagonal), ("off-diagonal", off_diagonal)] {
            let witness_change = unit_change_from_witness(&basis_dependent_witness(&mask, &declared));
            let moved_units = relu
                .apply(&witness_change)
                .expect("the owner's witness lies in ReLU's group");
            let band = masked_band(&relu, &mask, &inputs, scale_ratio(&witness_change.scales));
            let moved = masked_hidden_write(&relu, &mask, &inputs)
                .iter()
                .zip(masked_hidden_write(&moved_units, &mask, &inputs).iter())
                .zip(band.iter())
                .filter(|((b, a), limit)| (**a - **b).abs() > **limit)
                .count();
            assert!(moved > 0, "the owner's witness must move the fixed {kind} mask's write past the band");
        }
    }

    /// Group membership agrees across the two modules. For masks that are:
    /// - not complex-linear on a plane,
    /// - coupling two planes,
    /// - or off-diagonal on the pass-through,
    ///
    /// the owner returns a `RotaryCommutant` witness, and `RotaryQueryKey::apply` admits
    /// it, at equal and at distinct frequencies. The positive control is the plain
    /// `Blocks([head_dim])` witness for the not-complex-linear mask, a reflection that
    /// breaks the plane's complex structure. `apply` refuses it.
    #[test]
    fn every_rotary_witness_is_admitted_by_the_family_and_a_foreign_witness_is_refused() {
        let mut rng = StdRng::seed_from_u64(295_415);
        let geometry = AttentionGeometry {
            model_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
        };
        let identity = Array2::<f64>::eye(geometry.head_dim);
        let foreign_gauge = DeclaredGauge::Blocks(GaugeBlocks::new(&[geometry.head_dim]).expect("one block"));
        for frequencies in [[0.7, 0.7], [1.0, 0.3]] {
            let teacher = rotary_block(
                geometry,
                half_split_rotary(&frequencies),
                uniform_matrix(&mut rng, geometry.query_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.query_dim()),
                uniform_matrix(&mut rng, geometry.key_value_dim(), geometry.model_dim),
                uniform_vector(&mut rng, geometry.key_value_dim()),
            )
            .expect("shapes");
            let declared = teacher.declared_gauge().expect("a valid rotary declaration");
            let (a0, b0) = teacher.rotary.plane(0);
            let (a1, b1) = teacher.rotary.plane(1);
            let mut not_complex = identity.clone();
            not_complex[[b0, a0]] = 0.4;
            not_complex[[a0, b0]] = 0.4;
            let mut coupling = identity.clone();
            coupling[[a0, a1]] = 0.3;
            coupling[[b0, b1]] = 0.3;
            let mut pass_through = identity.clone();
            pass_through[[4, 5]] = 0.2;
            for mask in [&not_complex, &coupling, &pass_through] {
                let witness = basis_dependent_witness(mask, &declared);
                assert!(
                    teacher.apply(&[witness, identity.clone()]).is_ok(),
                    "frequencies {frequencies:?}: the owner's witness must lie in the rotary commutant"
                );
            }
            let foreign = basis_dependent_witness(&not_complex, &foreign_gauge);
            assert!(matches!(
                teacher.apply(&[foreign, identity.clone()]),
                Err(GaugeRefusal::NotInGroup { .. })
            ));
        }
    }
}

