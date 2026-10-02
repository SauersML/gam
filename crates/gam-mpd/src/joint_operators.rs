//! Factored, gauge-invariant joint operators of an attention block (#2951).
//!
//! The implementation gauges of [`super::gauge`] move a block's tensors but not the products
//! its function reads. Those products are the block's joint operators, and this module builds
//! them as factors `M = L Rᵀ` (`width × rank` each), never as `width × width` matrices.
//!
//! # Query/key
//!
//! Query head `h` of key/value head `g` reads `u = W_Q,h x + b_Q,h` and its key reads
//! `u_k = W_K,g x + b_K,g`. Behind Qwen3's per-head norm the rows are `q̂ = D_w ν(u) u` and
//! `k̂ = D_v ν(u_k) u_k`, `ν(u) = (mean u² + ε)^{-1/2}`; without it `ν = 1` and `D = I`. The
//! source rotates plane `j = (a, b)` by `α Rot(p ω_j)`, with `Rot(φ) = cos φ I + sin φ J` and
//! `J (x_a, x_b) = (−x_b, x_a)`, so `(α Rot(p_t ω) q)ᵀ (α Rot(p_s ω) k) = α² qᵀ Rot((p_s − p_t) ω) k`.
//! With `Δ = p_s − p_t`, the score of query token `t` on key token `s` is
//!
//! ```text
//! S_ts = ν_q(x_t) ν_k(x_s) [ σα² Σ_j (cos(ω_j Δ) x_tᵀ A_hj x_s + sin(ω_j Δ) x_tᵀ B_hj x_s) + σ x_tᵀ Π_h x_s ],
//! A_hj = W̃_Q,hᵀ P_j W̃_K,g,   B_hj = W̃_Q,hᵀ J_j W̃_K,g,   Π_h = W̃_Q,hᵀ P_pass W̃_K,g,
//! ```
//!
//! with `W̃ = D W diag(γ)`: the query/key gains folded into the rows and, when declared, the
//! preceding residual norm's gain `γ` folded into the columns. Each `A_hj`, `B_hj` has rank at
//! most 2 and `Π_h` at most the pass-through count; the pass-through coordinates are neither
//! rotated nor scaled by `α`. A block with a projection bias reads the homogeneous input
//! `(x, 1)`, and its factors are one row wider.
//!
//! **Excluded scalars.** The per-head normalizers `ν_q(x_t) ν_k(x_s)` depend on the input and
//! scale every plane of the head together, and the residual norm's own normalizer scales
//! every head of a token together; neither is a parameter product, so the operators exclude
//! them. `σα²` and `σ` are constants reported beside the operators, not folded in.
//!
//! **Invariance.** Under the normed family's element (plane scales `ρ`, quarter turns, gain
//! flips) `D_w W_Q` and `D_v W_K` move by `R^t` and `R^{-t}` on each plane, which commute with
//! `P_j` and `J_j` and are orthogonal, so `A_hj` and `B_hj` are unchanged. A folded residual
//! gain moves `diag(γ)` from the norm into the columns, which the operators already carry.
//!
//! # Value/output
//!
//! Head `h`'s write of its weighted value read is `O_h (V_g x + b_V,g)`, so its operator is
//! `C_h = O_h V_g diag(γ)` (homogeneous when `b_V ≠ 0`). The key/value group's summed
//! operator `Σ_{h∈g} C_h` is what the group writes when its heads attend alike; its factors
//! are the stacked `[O_h]_h` and the repeated `V_g`, so it is formed with no rounding. The
//! output bias is a constant write, not part of either.
//!
//! # Grams and comparisons
//!
//! `⟨M_i, M_j⟩_F = tr(R_i L_iᵀ L_j R_jᵀ) = tr[(L_iᵀ L_j)(R_jᵀ R_i)]`, so a family's Gram needs only
//! the factor Grams ([`family_gram`]). [`compare_operators`] tests operator equality and
//! proportionality at full precision through the factors' QR triangles.
//!
//! **Subspace overlap is not the same law.** Two operators whose ranges coincide (a captured
//! energy or a top principal cosine of one) can still differ, e.g. by a factor 2, and then they
//! route differently. Sharing claims need [`compare_operators`]: its difference certifies two
//! operators distinct, and its proportionality residual says whether they differ only by
//! scale.
//!
//! # Routing laws
//!
//! [`routing_laws`] groups query heads whose whole score operators are equal within band
//! ([`compare_heads`]), and reports how the laws relate (proportional, with the common
//! scale, or not). [`attention_letters`] turns the laws into an attention step's
//! observability letters, one summed transport per law (with a pre-norm input gain and a
//! post-norm output gain folded in), the only letters `state::ObservabilityLetter::RoutingLaws`
//! accepts for attention.

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, fast_abt, fast_atb, fast_atv};
use gam_linalg::roundoff::{accumulation_growth, householder_qr_backward_band};
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Axis, Dimension, concatenate, s};

use super::attention::{AttentionGeometry, AttentionProgramError, NativeAttention, RotaryEmbedding};
use super::block::{SUBNORMAL_SPACING, up};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

/// Why a joint operator, Gram or comparison was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum JointRefusal {
    /// A factor, input or gain disagrees with the block.
    Shape {
        what: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A non-finite entry.
    NonFinite { what: &'static str },
    /// The attention owner refused the block.
    Attention(AttentionProgramError),
    /// The process memory governor refused a Gram's footprint.
    Memory(MemoryReservationError),
    /// A reported number was refused by its evidence constructor.
    Evidence(EvidenceStatusError),
    /// The linear-algebra backend failed.
    Decomposition { what: &'static str, detail: String },
}

impl From<EvidenceStatusError> for JointRefusal {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl From<AttentionProgramError> for JointRefusal {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(error)
    }
}

fn expect_shape(what: &'static str, expected: (usize, usize), found: (usize, usize)) -> Result<(), JointRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(JointRefusal::Shape { what, expected, found })
    }
}

fn decomposition(what: &'static str, failure: &FaerLinalgError) -> JointRefusal {
    JointRefusal::Decomposition {
        what,
        detail: format!("{failure:?}"),
    }
}

/// An upper bound on `‖v‖₂` of a computed vector (`‖M‖_F` of a matrix): its sum of squares
/// within `γ_n`, rounded up.
fn upper_norm<'a, V: AsArray<'a, f64, D>, D: Dimension>(values: V) -> f64 {
    let values = values.into();
    let squares = values.iter().fold(0.0, |sum, &entry| up(sum + up(entry * entry)));
    up(up(squares * up(1.0 + up(2.0 * accumulation_growth(values.len().max(1))))).sqrt())
}

/// An operator `M = L Rᵀ` held by its factors, `L` and `R` both `width × rank`. The operator is
/// the exact product of the stored factors.
#[derive(Clone, Debug, PartialEq)]
pub struct FactoredOperator {
    left: Array2<f64>,
    right: Array2<f64>,
}

impl FactoredOperator {
    pub fn new(left: Array2<f64>, right: Array2<f64>) -> Result<Self, JointRefusal> {
        expect_shape("operator right factor", left.dim(), right.dim())?;
        if left.iter().chain(right.iter()).any(|entry| !entry.is_finite()) {
            return Err(JointRefusal::NonFinite { what: "operator factor" });
        }
        Ok(Self { left, right })
    }

    pub fn left(&self) -> ArrayView2<'_, f64> {
        self.left.view()
    }

    pub fn right(&self) -> ArrayView2<'_, f64> {
        self.right.view()
    }

    pub fn width(&self) -> usize {
        self.left.nrows()
    }

    pub fn rank(&self) -> usize {
        self.left.ncols()
    }

    /// `xᵀ M y = (Lᵀ x) · (Rᵀ y)` and a bound on its rounding. Each of `a = Lᵀ x` and
    /// `b = Rᵀ y` is a `width`-term inner product within `e = γ_w |L|ᵀ |x|` (resp. `|R|ᵀ |y|`);
    /// the final `rank`-term sum adds `γ_r Σ |â_k b̂_k|`, and the box products
    /// `|â| e_b + e_a |b̂| + e_a e_b` carry the inputs' errors.
    pub fn bilinear(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> Result<(f64, f64), JointRefusal> {
        expect_shape("bilinear left input", (self.width(), 1), (x.len(), 1))?;
        expect_shape("bilinear right input", (self.width(), 1), (y.len(), 1))?;
        let (a, b) = (fast_atv(&self.left, &x), fast_atv(&self.right, &y));
        let growth = accumulation_growth(self.width());
        let (ea, eb) = (
            fast_atv(&self.left.mapv(f64::abs), &x.mapv(f64::abs)),
            fast_atv(&self.right.mapv(f64::abs), &y.mapv(f64::abs)),
        );
        let inflate = up(1.0 + up(2.0 * growth));
        let mut value = 0.0;
        let mut band = 0.0;
        let mut absolute = 0.0;
        for k in 0..self.rank() {
            let (ra, rb) = (up(up(growth * ea[k]) * inflate), up(up(growth * eb[k]) * inflate));
            value += a[k] * b[k];
            absolute = up(absolute + (a[k] * b[k]).abs());
            band = up(band + up(up(a[k].abs() * rb) + up(ra * b[k].abs()) + up(ra * rb)));
        }
        let rounding = up(accumulation_growth(self.rank().max(1)) * absolute);
        Ok((value, up(up(band + rounding) + up(2.0 * self.rank() as f64 * SUBNORMAL_SPACING))))
    }

    /// An upper bound on `‖δM‖_F` when the stored factors lie within `ρ_L |L|` and `ρ_R |R|`
    /// entrywise of represented ones: `‖δL‖_F ‖R‖_F + ‖L‖_F ‖δR‖_F + ‖δL‖_F ‖δR‖_F`.
    pub fn relative_defect_reach(&self, left_relative: f64, right_relative: f64) -> f64 {
        let (left, right) = (upper_norm(self.left.view()), upper_norm(self.right.view()));
        entrywise_defect_reach(left, right, up(left_relative * left), up(right_relative * right))
    }
}

/// `‖δL‖_F ‖R‖_F + ‖L‖_F ‖δR‖_F + ‖δL‖_F ‖δR‖_F`, an upper bound on `‖(L + δL)(R + δR)ᵀ − L Rᵀ‖_F`
/// from the Frobenius norms of the factors and their defects.
pub fn entrywise_defect_reach(left: f64, right: f64, left_defect: f64, right_defect: f64) -> f64 {
    up(up(up(left_defect * right) + up(left * right_defect)) + up(left_defect * right_defect))
}

/// The query/key joint operators of one attention block.
#[derive(Clone, Debug)]
pub struct QueryKeyOperators {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
    /// `(A_hj, B_hj)` at `head · planes + plane`.
    planes: Vec<(FactoredOperator, FactoredOperator)>,
    /// `Π_h` per query head, when the head has pass-through coordinates.
    pass_through: Vec<Option<FactoredOperator>>,
    formation_defect: f64,
}

/// The query or key rows `W̃ = D W diag(γ)` of one head, with the bias column `D b` when the
/// block is homogeneous.
fn folded_rows(
    weight: ArrayView2<'_, f64>,
    bias: ArrayView1<'_, f64>,
    head_gain: Option<ArrayView1<'_, f64>>,
    input_gain: Option<ArrayView1<'_, f64>>,
    homogeneous: bool,
) -> Array2<f64> {
    let (rows, width) = weight.dim();
    let columns = width + usize::from(homogeneous);
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        let scale = head_gain.map_or(1.0, |gain| gain[row]);
        if column == width {
            scale * bias[row]
        } else {
            scale * weight[[row, column]] * input_gain.map_or(1.0, |gain| gain[column])
        }
    })
}

/// The query/key operators of `native`, with `input_gain` (the preceding residual norm's gain)
/// folded into the columns when it is declared.
pub fn query_key_operators(native: &NativeAttention, input_gain: Option<ArrayView1<'_, f64>>) -> Result<QueryKeyOperators, JointRefusal> {
    let g = native.geometry();
    let rotary = native.rotary().clone();
    if let Some(gain) = input_gain {
        expect_shape("input gain", (g.model_dim, 1), (gain.len(), 1))?;
    }
    let homogeneous = native.query().bias.iter().chain(native.key().bias.iter()).any(|&entry| entry != 0.0);
    let norm = native.query_key_norm();
    let (query_gain, key_gain) = (norm.map(|norm| norm.query_gain()), norm.map(|norm| norm.key_gain()));
    let hd = g.head_dim;
    let head_rows = |weight: &Array2<f64>, bias: &Array1<f64>, head: usize, gain: Option<ArrayView1<'_, f64>>| {
        folded_rows(
            weight.slice(s![head * hd..(head + 1) * hd, ..]),
            bias.slice(s![head * hd..(head + 1) * hd]),
            gain,
            input_gain,
            homogeneous,
        )
    };
    let planes = rotary.inverse_frequencies.len();
    let mut operators = Vec::with_capacity(g.n_heads * planes);
    let mut pass_through = Vec::with_capacity(g.n_heads);
    for head in 0..g.n_heads {
        let query = head_rows(&native.query().weight, &native.query().bias, head, query_gain);
        let key = head_rows(&native.key().weight, &native.key().bias, g.key_value_head(head), key_gain);
        for plane in 0..planes {
            let (a, b) = rotary.plane(plane);
            let left = concatenate(Axis(1), &[query.row(a).insert_axis(Axis(1)), query.row(b).insert_axis(Axis(1))]).map_err(shape_failed)?;
            let right = concatenate(Axis(1), &[key.row(a).insert_axis(Axis(1)), key.row(b).insert_axis(Axis(1))]).map_err(shape_failed)?;
            let turned_b = key.row(a).to_owned();
            let turned_a = key.row(b).mapv(|entry| -entry);
            let turned = concatenate(Axis(1), &[turned_a.view().insert_axis(Axis(1)), turned_b.view().insert_axis(Axis(1))]).map_err(shape_failed)?;
            operators.push((FactoredOperator::new(left.clone(), right)?, FactoredOperator::new(left, turned)?));
        }
        let pass = rotary.rotary_dim()..hd;
        pass_through.push(if pass.is_empty() {
            None
        } else {
            Some(FactoredOperator::new(
                query.slice(s![pass.clone(), ..]).t().to_owned(),
                key.slice(s![pass, ..]).t().to_owned(),
            )?)
        });
    }
    let roundings = usize::from(norm.is_some()) + usize::from(input_gain.is_some());
    Ok(QueryKeyOperators {
        geometry: g,
        rotary,
        score_scale: native.score_scale(),
        planes: operators,
        pass_through,
        formation_defect: accumulation_growth(roundings),
    })
}

fn shape_failed(failure: ndarray::ShapeError) -> JointRefusal {
    JointRefusal::Decomposition {
        what: "factor assembly",
        detail: failure.to_string(),
    }
}

impl QueryKeyOperators {
    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn planes(&self) -> usize {
        self.rotary.inverse_frequencies.len()
    }

    /// `A_hj`.
    pub fn cosine(&self, head: usize, plane: usize) -> &FactoredOperator {
        &self.planes[head * self.planes() + plane].0
    }

    /// `B_hj`.
    pub fn sine(&self, head: usize, plane: usize) -> &FactoredOperator {
        &self.planes[head * self.planes() + plane].1
    }

    /// `Π_h`, when the head has pass-through coordinates.
    pub fn pass_through(&self, head: usize) -> Option<&FactoredOperator> {
        self.pass_through[head].as_ref()
    }

    /// `σα²`, the constant on every rotated plane.
    pub fn rotary_multiplier(&self) -> f64 {
        self.score_scale * self.rotary.attention_scaling * self.rotary.attention_scaling
    }

    /// `σ`, the constant on the pass-through coordinates.
    pub fn pass_through_multiplier(&self) -> f64 {
        self.score_scale
    }

    /// The relative defect of every factor entry against the product of the stored tensors it
    /// folds: one rounding per folded gain.
    pub fn formation_defect(&self) -> f64 {
        self.formation_defect
    }

    /// Head `head`'s score at inputs `x_t`, `x_s` (homogeneous when the block has a bias) and
    /// `Δ = p_s − p_t`, without the normalizers `ν_q ν_k`, and a bound on its rounding
    /// against the stored factors. Each plane's `cos(ω_j Δ)`, `sin(ω_j Δ)` rounds its angle by
    /// `γ_1 |φ|` and libm by one ulp; those errors, each operator's [`FactoredOperator::bilinear`]
    /// band, the products and the `2P + 1`-term sum make up the band.
    pub fn score(&self, head: usize, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>, delta: i64) -> Result<(f64, f64), JointRefusal> {
        let (mut rotated, mut rotated_band, mut absolute) = (0.0, 0.0, 0.0);
        for (plane, &frequency) in self.rotary.inverse_frequencies.iter().enumerate() {
            let angle = delta as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            let trig = up(up(accumulation_growth(1) * angle.abs()) + f64::EPSILON);
            let (a, band_a) = self.cosine(head, plane).bilinear(x, y)?;
            let (b, band_b) = self.sine(head, plane).bilinear(x, y)?;
            rotated += cos * a + sin * b;
            absolute = up(absolute + up((cos * a).abs() + (sin * b).abs()));
            let box_a = up(up(cos.abs() * band_a) + up(trig * up(a.abs() + band_a)));
            let box_b = up(up(sin.abs() * band_b) + up(trig * up(b.abs() + band_b)));
            rotated_band = up(rotated_band + up(box_a + box_b));
        }
        let terms = 2 * self.planes() + 2;
        let multiplier = self.rotary_multiplier();
        let (pass, pass_band) = match self.pass_through(head) {
            Some(operator) => operator.bilinear(x, y)?,
            None => (0.0, 0.0),
        };
        let value = multiplier * rotated + self.score_scale * pass;
        let rounding = up(accumulation_growth(terms + 2) * up(up(multiplier.abs() * absolute) + up((self.score_scale * pass).abs())));
        let band = up(up(up(multiplier.abs() * rotated_band) + up(self.score_scale.abs() * pass_band)) + rounding);
        Ok((value, band))
    }

}

/// `Σ_a ‖l_a‖ ‖r_a‖` over the factor columns, rounded up.
fn column_reach(operator: &FactoredOperator) -> f64 {
    (0..operator.rank()).fold(0.0, |sum, column| {
        up(sum + up(upper_norm(operator.left.column(column)) * upper_norm(operator.right.column(column))))
    })
}

/// The rounding band of one Gram entry `⟨M_i, M_j⟩ = Σ_{a∈i, b∈j} (L_iᵀL_j)_ab (R_iᵀR_j)_ab`.
/// Every factor Gram entry is a `w`-term inner product within `γ_w ‖l_a‖ ‖l_b‖`, so each product
/// of two is within `(1 + γ_w)² − 1` of `P_ab = ‖l_a‖‖l_b‖‖r_a‖‖r_b‖`, and the `r_i r_j`-term sum
/// adds `γ_{r_i r_j}` of the computed absolute sum. `Σ_ab P_ab = n_i n_j` with
/// `n = Σ_a ‖l_a‖‖r_a‖`, and `(1 + γ_w)²(1 + γ_k) ≤ 1 + γ_{2w + k}`.
fn entry_band(width: usize, first_rank: usize, second_rank: usize, first_reach: f64, second_reach: f64) -> f64 {
    let growth = accumulation_growth(2 * width + first_rank * second_rank);
    up(up(growth * up(first_reach * second_reach)) + up((first_rank * second_rank) as f64 * SUBNORMAL_SPACING))
}

/// A family's Frobenius Gram with an entrywise rounding band, holding its memory reservation.
#[derive(Debug)]
pub struct FamilyGram {
    pub gram: Array2<f64>,
    pub band: Array2<f64>,
    footprint: MemoryReservation,
}

impl FamilyGram {
    pub fn reserved_bytes(&self) -> usize {
        self.footprint.bytes()
    }
}

/// `G_ij = ⟨M_i, M_j⟩_F = tr[(L_iᵀL_j)(R_jᵀR_i)]` over a family of operators of one width,
/// from the stacked factor Grams, never forming a `width × width` operator. The stacked factors,
/// both factor Grams, the Gram and its band are reserved on `governor` before any is allocated.
pub fn family_gram(governor: &MemoryGovernor, family: &[&FactoredOperator]) -> Result<FamilyGram, JointRefusal> {
    let width = family.first().map_or(0, |operator| operator.width());
    for operator in family {
        expect_shape("family operator", (width, operator.rank()), operator.left.dim())?;
    }
    let total: usize = family.iter().map(|operator| operator.rank()).sum();
    let count = family.len();
    let cells = (2usize.saturating_mul(width).saturating_mul(total))
        .saturating_add(2usize.saturating_mul(total).saturating_mul(total))
        .saturating_add(2usize.saturating_mul(count).saturating_mul(count));
    let footprint = governor
        .try_reserve(cells.saturating_mul(std::mem::size_of::<f64>()), "joint operator family Gram")
        .map_err(JointRefusal::Memory)?;
    let lefts: Vec<_> = family.iter().map(|operator| operator.left.view()).collect();
    let rights: Vec<_> = family.iter().map(|operator| operator.right.view()).collect();
    let (left, right) = if count == 0 {
        (Array2::zeros((width, 0)), Array2::zeros((width, 0)))
    } else {
        (
            concatenate(Axis(1), &lefts).map_err(shape_failed)?,
            concatenate(Axis(1), &rights).map_err(shape_failed)?,
        )
    };
    let (left_gram, right_gram) = (fast_atb(&left, &left), fast_atb(&right, &right));
    let offsets: Vec<usize> = family
        .iter()
        .scan(0, |offset, operator| {
            let start = *offset;
            *offset += operator.rank();
            Some(start)
        })
        .collect();
    let reaches: Vec<f64> = family.iter().map(|operator| column_reach(operator)).collect();
    let mut gram = Array2::zeros((count, count));
    let mut band = Array2::zeros((count, count));
    for i in 0..count {
        let rows = offsets[i]..offsets[i] + family[i].rank();
        for j in i..count {
            let columns = offsets[j]..offsets[j] + family[j].rank();
            let value = (&left_gram.slice(s![rows.clone(), columns.clone()]) * &right_gram.slice(s![rows.clone(), columns])).sum();
            let entry = entry_band(width, family[i].rank(), family[j].rank(), reaches[i], reaches[j]);
            gram[[i, j]] = value;
            gram[[j, i]] = value;
            band[[i, j]] = entry;
            band[[j, i]] = entry;
        }
    }
    Ok(FamilyGram { gram, band, footprint })
}

/// The value/output joint operators of one attention block.
#[derive(Clone, Debug)]
pub struct ValueOutputOperators {
    heads: Vec<FactoredOperator>,
    groups: Vec<FactoredOperator>,
    formation_defect: f64,
}

impl ValueOutputOperators {
    /// `C_h = O_h V_g diag(γ)`.
    pub fn head(&self, head: usize) -> &FactoredOperator {
        &self.heads[head]
    }

    /// `Σ_{h∈g} C_h`, factored as `[O_h]_h` against the repeated value rows.
    pub fn group(&self, group: usize) -> &FactoredOperator {
        &self.groups[group]
    }

    /// The relative defect of every right factor entry against the stored `V diag(γ)`: one
    /// rounding when a gain is folded. The left factors are the stored output columns.
    pub fn formation_defect(&self) -> f64 {
        self.formation_defect
    }
}

/// The value/output operators of `native`, with `input_gain` folded into the value columns
/// when it is declared.
pub fn value_output_operators(native: &NativeAttention, input_gain: Option<ArrayView1<'_, f64>>) -> Result<ValueOutputOperators, JointRefusal> {
    let g = native.geometry();
    if let Some(gain) = input_gain {
        expect_shape("input gain", (g.model_dim, 1), (gain.len(), 1))?;
    }
    let hd = g.head_dim;
    let homogeneous = native.value().bias.iter().any(|&entry| entry != 0.0);
    let values: Vec<Array2<f64>> = (0..g.n_kv_heads)
        .map(|group| {
            folded_rows(
                native.value().weight.slice(s![group * hd..(group + 1) * hd, ..]),
                native.value().bias.slice(s![group * hd..(group + 1) * hd]),
                None,
                input_gain,
                homogeneous,
            )
            .t()
            .to_owned()
        })
        .collect();
    // A homogeneous operator's left factor carries a zero row for the constant input.
    let columns = |head: usize| {
        let block = native.output().weight.slice(s![.., head * hd..(head + 1) * hd]);
        let rows = block.nrows() + usize::from(homogeneous);
        Array2::from_shape_fn((rows, hd), |(row, column)| if row < block.nrows() { block[[row, column]] } else { 0.0 })
    };
    let mut heads = Vec::with_capacity(g.n_heads);
    for head in 0..g.n_heads {
        heads.push(FactoredOperator::new(columns(head), values[g.key_value_head(head)].clone())?);
    }
    let per_group = g.n_heads / g.n_kv_heads;
    let mut groups = Vec::with_capacity(g.n_kv_heads);
    for (group, value) in values.iter().enumerate() {
        let members: Vec<Array2<f64>> = (group * per_group..(group + 1) * per_group).map(columns).collect();
        let left = concatenate(Axis(1), &members.iter().map(|member| member.view()).collect::<Vec<_>>()).map_err(shape_failed)?;
        let right = concatenate(Axis(1), &vec![value.view(); per_group]).map_err(shape_failed)?;
        groups.push(FactoredOperator::new(left, right)?);
    }
    Ok(ValueOutputOperators {
        heads,
        groups,
        formation_defect: accumulation_growth(usize::from(input_gain.is_some())),
    })
}

/// Which pair of operators a comparison is about: operators on `width`-dimensional inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorPair {
    pub width: usize,
}

/// Equality and proportionality of two operators.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorComparison {
    /// `‖M₁ − M₂‖_F`, exact with its numerical error.
    pub difference: EvidenceStatus<(), OperatorPair>,
    pub first_norm: EvidenceStatus<(), OperatorPair>,
    pub second_norm: EvidenceStatus<(), OperatorPair>,
    /// `c = ⟨M₁, M₂⟩/‖M₂‖²`, as computed; `0` when `M₂` is zero.
    pub scale: f64,
    /// `‖M₁ − c M₂‖_F` at the stored `c`, exact with its numerical error. It bounds
    /// `min_c ‖M₁ − c M₂‖_F` from above.
    pub proportionality_residual: EvidenceStatus<(), OperatorPair>,
}

impl OperatorComparison {
    /// Whether the difference is certified nonzero: `M₁ ≠ M₂` as operators.
    pub fn proven_distinct(&self) -> bool {
        self.difference.lower_bound().is_some_and(|lower| lower > 0.0)
    }
}

/// `‖L Rᵀ‖_F` through the factors' triangles, and a bound on its error.
///
/// With `k` columns, a factor with at least `k` rows is replaced by its thin QR `Q T`.
/// Householder QR is backward stable: `T` is the exact triangle of `L + δL` for an orthonormal
/// `Q`, with `‖δL‖_F ≤` `householder_qr_backward_band`. So `‖T_L T_Rᵀ‖_F` is the norm of the
/// perturbed product, within `‖δL‖_F ‖R‖_F + ‖L‖_F ‖δR‖_F + ‖δL‖_F ‖δR‖_F` of the exact one. The
/// `k`-term core product adds `γ_k ‖|T_L| |T_R|ᵀ‖_F`, and the norm's own sum of squares at most
/// `γ_{k²}` of the value. A factor with fewer rows than `k` is used as it is.
fn factored_norm(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Result<(f64, f64), JointRefusal> {
    let reduce = |factor: ArrayView2<'_, f64>, what: &'static str| -> Result<(Array2<f64>, f64), JointRefusal> {
        let (rows, k) = factor.dim();
        if rows < k || k == 0 {
            return Ok((factor.to_owned(), 0.0));
        }
        let triangle = factor.qr().map_err(|failure| decomposition(what, &failure))?.1;
        let t = triangle.slice(s![..k, ..k]).to_owned();
        let band = householder_qr_backward_band(rows, k, upper_norm(t.view()));
        Ok((t, band))
    };
    let (left_core, left_band) = reduce(left, "left factor")?;
    let (right_core, right_band) = reduce(right, "right factor")?;
    let core = fast_abt(&left_core, &right_core);
    let value = core.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
    let (left_norm, right_norm) = (upper_norm(left), upper_norm(right));
    let k = left.ncols().max(1);
    let product = up(accumulation_growth(k) * upper_norm(fast_abt(&left_core.mapv(f64::abs), &right_core.mapv(f64::abs)).view()));
    let error = up(up(entrywise_defect_reach(left_norm, right_norm, left_band, right_band) + product)
        + up(accumulation_growth(core.len().max(1) + 2) * value));
    Ok((value, error))
}

fn exact_norm(value: f64, error: f64, width: usize) -> Result<EvidenceStatus<(), OperatorPair>, JointRefusal> {
    Ok(EvidenceStatus::exact(value, error, ExactBasis::Algebraic, None, OperatorPair { width })?)
}

/// Compare two operators of one width: `‖M₁ − M₂‖_F`, both norms, the least-squares scale
/// `c` and `‖M₁ − c M₂‖_F`, each as the norm of the product of the stacked factors'
/// thin-QR triangles, with its backward-error band, on the stacked factors
/// `[L₁ | −c L₂]`, `[R₁ | R₂]`. `c` comes from the pair's [`family_gram`], which `governor`
/// reserves.
pub fn compare_operators(
    governor: &MemoryGovernor,
    first: &FactoredOperator,
    second: &FactoredOperator,
) -> Result<OperatorComparison, JointRefusal> {
    let width = first.width();
    expect_shape("compared operator", (width, second.rank()), second.left.dim())?;
    let stacked = |scale: f64| scaled_difference(first, second, scale);
    let gram = family_gram(governor, &[first, second])?;
    let scale = if gram.gram[[1, 1]] > 0.0 { gram.gram[[0, 1]] / gram.gram[[1, 1]] } else { 0.0 };
    let (difference, difference_error) = stacked(1.0)?;
    let (first_norm, first_error) = factored_norm(first.left.view(), first.right.view())?;
    let (second_norm, second_error) = factored_norm(second.left.view(), second.right.view())?;
    let (residual, residual_error) = stacked(scale)?;
    Ok(OperatorComparison {
        difference: exact_norm(difference, difference_error, width)?,
        first_norm: exact_norm(first_norm, first_error, width)?,
        second_norm: exact_norm(second_norm, second_error, width)?,
        scale,
        proportionality_residual: exact_norm(residual, residual_error, width)?,
    })
}

/// `‖M₁ − c M₂‖_F` at the given `c`, through [`factored_norm`] on the stacked factors
/// `[L₁ | −c L₂]`, `[R₁ | R₂]`. `−c L₂` rounds each entry once, a relative defect `γ_1` on
/// those columns, which is added to the error.
fn scaled_difference(first: &FactoredOperator, second: &FactoredOperator, scale: f64) -> Result<(f64, f64), JointRefusal> {
    let negated = second.left.mapv(|entry| -scale * entry);
    let left = concatenate(Axis(1), &[first.left.view(), negated.view()]).map_err(shape_failed)?;
    let right = concatenate(Axis(1), &[first.right.view(), second.right.view()]).map_err(shape_failed)?;
    let (value, error) = factored_norm(left.view(), right.view())?;
    let scaled = up(accumulation_growth(1) * up(scale.abs() * upper_norm(second.left.view())));
    Ok((value, up(error + up(scaled * upper_norm(second.right.view())))))
}

// ---------------------------------------------------------------------------------------
// Routing laws
// ---------------------------------------------------------------------------------------
//
// Two query heads attend alike at every input exactly when their scores agree at every
// token pair and offset. Without a query/key norm the score of head `h` is
// `σα² Σ_j (cos(ω_j Δ) xᵀ A_hj y + sin(ω_j Δ) xᵀ B_hj y) + σ xᵀ Π_h y`, and the functions
// `cos(ω_j Δ)`, `sin(ω_j Δ)` of distinct frequencies are independent over the offsets, so
// equal scores need equal operators: the head's score operator is the direct sum
// `S_h = ⊕_j σα² (A_hj ⊕ B_hj) ⊕ σ Π_h`, and two heads share a routing law iff
// `S_h = S_g`. `‖S_h − S_g‖²_F = Σ_j σ²α⁴ (‖A_hj − A_gj‖² + ‖B_hj − B_gj‖²) + σ² ‖Π_h − Π_g‖²`,
// read part by part through [`compare_operators`]' factored norms, never forming `S`.
// Behind a query/key norm the score also carries the normalizers `ν_q(u) ν_k(u_k)`, which
// the operators exclude. Equal raw query rows (and raw key rows) give equal normalizers, so
// there two heads also need their homogeneous raw rows `[W | b]` not proven distinct.
//
// Heads that share a law have one attention pattern, so their value/output transports act
// as one: the law's letter is `Σ_{h∈law} C_h`, and a cross-head `GL` of the law's stacked
// value rows moves each `C_h` but not the sum. A per-head letter is therefore not gauge
// invariant, and an attention step's observability letters are the laws' transports.

/// One part of a head's score operator and its weight in `S_h`.
fn head_parts(operators: &QueryKeyOperators, head: usize) -> Vec<(&FactoredOperator, f64)> {
    let rotary = operators.rotary_multiplier();
    let mut parts = Vec::with_capacity(2 * operators.planes() + 1);
    for plane in 0..operators.planes() {
        parts.push((operators.cosine(head, plane), rotary));
        parts.push((operators.sine(head, plane), rotary));
    }
    if let Some(pass) = operators.pass_through(head) {
        parts.push((pass, operators.pass_through_multiplier()));
    }
    parts
}

/// `√(Σ_k (w_k v_k)²)` from per-part values `v_k` within `e_k`, and a bound on its error: the
/// root of the per-part lower ends and of the per-part upper ends, each weighted with its own
/// rounding, bracket the exact value.
fn combined_norm(parts: &[(f64, f64, f64)]) -> (f64, f64) {
    let growth = accumulation_growth(2 * parts.len() + 2);
    let (mut value, mut low, mut high) = (0.0, 0.0, 0.0);
    for &(weight, part, error) in parts {
        let weight = weight.abs();
        let reach = up(up(weight * error) + up(accumulation_growth(1) * up(weight * part)));
        let centre = weight * part;
        value += centre * centre;
        let lower = (centre - reach).max(0.0);
        low += lower * lower;
        let upper = up(centre + reach);
        high = up(high + up(upper * upper));
    }
    let value = value.sqrt();
    let low = (low * (1.0 - growth)).max(0.0).sqrt();
    let high = up(up(high * up(1.0 + growth)).sqrt());
    (value, up((value - low).max(high - value).max(0.0)))
}

/// Query heads `first` and `second` compared as the score operators `S_h` (module docs,
/// *Routing laws*): `‖S₁ − S₂‖_F`, both norms, the least-squares common scale
/// `c = Σ_k w_k² ⟨M₁ₖ, M₂ₖ⟩ / Σ_k w_k² ‖M₂ₖ‖²` from each part's [`family_gram`], and
/// `‖S₁ − c S₂‖_F`. The width reported is the operators' input width.
pub fn compare_heads(
    governor: &MemoryGovernor,
    operators: &QueryKeyOperators,
    first: usize,
    second: usize,
) -> Result<OperatorComparison, JointRefusal> {
    let heads = operators.geometry().n_heads;
    for head in [first, second] {
        if head >= heads {
            return Err(JointRefusal::Shape {
                what: "compared head",
                expected: (heads, 1),
                found: (head, 1),
            });
        }
    }
    let (left, right) = (head_parts(operators, first), head_parts(operators, second));
    let width = left[0].0.width();
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (&(a, weight), &(b, _)) in left.iter().zip(&right) {
        let gram = family_gram(governor, &[a, b])?;
        numerator += weight * weight * gram.gram[[0, 1]];
        denominator += weight * weight * gram.gram[[1, 1]];
    }
    let scale = if denominator > 0.0 { numerator / denominator } else { 0.0 };
    let (mut difference, mut residual, mut first_norm, mut second_norm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (&(a, weight), &(b, _)) in left.iter().zip(&right) {
        let (value, error) = scaled_difference(a, b, 1.0)?;
        difference.push((weight, value, error));
        let (value, error) = scaled_difference(a, b, scale)?;
        residual.push((weight, value, error));
        let (value, error) = factored_norm(a.left.view(), a.right.view())?;
        first_norm.push((weight, value, error));
        let (value, error) = factored_norm(b.left.view(), b.right.view())?;
        second_norm.push((weight, value, error));
    }
    let status = |parts: &[(f64, f64, f64)]| {
        let (value, error) = combined_norm(parts);
        exact_norm(value, error, width)
    };
    Ok(OperatorComparison {
        difference: status(&difference)?,
        first_norm: status(&first_norm)?,
        second_norm: status(&second_norm)?,
        scale,
        proportionality_residual: status(&residual)?,
    })
}

/// The homogeneous raw rows `[W | b]` of head block `head` of a projection, transposed
/// (`width × head_dim`, one row wider when `homogeneous`), against the embedded identity:
/// an operator whose Frobenius norm is that of the rows.
fn raw_rows(weight: &Array2<f64>, bias: &Array1<f64>, head: usize, head_dim: usize, homogeneous: bool) -> Result<FactoredOperator, JointRefusal> {
    let rows = folded_rows(
        weight.slice(s![head * head_dim..(head + 1) * head_dim, ..]),
        bias.slice(s![head * head_dim..(head + 1) * head_dim]),
        None,
        None,
        homogeneous,
    );
    let width = rows.ncols();
    if width < head_dim {
        return Err(JointRefusal::Shape {
            what: "raw rows no narrower than the head",
            expected: (head_dim, 1),
            found: (width, 1),
        });
    }
    FactoredOperator::new(rows.t().to_owned(), Array2::from_shape_fn((width, head_dim), |(i, j)| if i == j { 1.0 } else { 0.0 }))
}

/// How a law's first head relates to an earlier law's first head.
#[derive(Clone, Debug, PartialEq)]
pub struct LawRelation {
    pub law: usize,
    pub earlier: usize,
    /// [`compare_heads`] of the two first heads: the difference is certified nonzero, or the
    /// normalizers differ behind a query/key norm.
    pub comparison: OperatorComparison,
}

impl LawRelation {
    /// The two laws' score operators differ only by the stored scale `c` (the residual is not
    /// resolved from zero): one pattern at two temperatures, `S_law = c S_earlier`.
    pub fn proportional(&self) -> bool {
        self.comparison.proportionality_residual.lower_bound().is_some_and(|lower| lower <= 0.0)
    }
}

/// The query heads of a block grouped into routing laws.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingLaws {
    /// Each law's heads, increasing, laws in order of their first head. A head joins the
    /// first law none of whose heads it is proven distinct from ([`compare_heads`], and the
    /// raw rows behind a query/key norm), otherwise it opens a new law. Two heads in
    /// different laws are certified to route differently; two heads in one law are equal
    /// within the arithmetic's bands.
    pub laws: Vec<Vec<usize>>,
    /// Every later law's first head against every earlier law's first head.
    pub relations: Vec<LawRelation>,
}

impl RoutingLaws {
    /// The law of query head `head`.
    pub fn law_of(&self, head: usize) -> Option<usize> {
        self.laws.iter().position(|law| law.contains(&head))
    }
}

/// The routing laws of `native`'s query heads, from its query/key operators `operators`
/// (built with the same `input_gain`).
pub fn routing_laws(governor: &MemoryGovernor, native: &NativeAttention, operators: &QueryKeyOperators) -> Result<RoutingLaws, JointRefusal> {
    let g = native.geometry();
    expect_shape("routing heads", (g.n_heads, g.head_dim), (operators.geometry().n_heads, operators.geometry().head_dim))?;
    let normed = native.has_query_key_norm();
    let homogeneous = native.query().bias.iter().chain(native.key().bias.iter()).any(|&entry| entry != 0.0);
    let raw = |head: usize| -> Result<(FactoredOperator, FactoredOperator), JointRefusal> {
        Ok((
            raw_rows(&native.query().weight, &native.query().bias, head, g.head_dim, homogeneous)?,
            raw_rows(&native.key().weight, &native.key().bias, g.key_value_head(head), g.head_dim, homogeneous)?,
        ))
    };
    let distinct = |first: usize, second: usize| -> Result<bool, JointRefusal> {
        if compare_heads(governor, operators, first, second)?.proven_distinct() {
            return Ok(true);
        }
        if normed {
            let ((query_a, key_a), (query_b, key_b)) = (raw(first)?, raw(second)?);
            if compare_operators(governor, &query_a, &query_b)?.proven_distinct() {
                return Ok(true);
            }
            if g.key_value_head(first) != g.key_value_head(second) && compare_operators(governor, &key_a, &key_b)?.proven_distinct() {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let mut laws: Vec<Vec<usize>> = Vec::new();
    for head in 0..g.n_heads {
        let mut joined = None;
        for (index, law) in laws.iter().enumerate() {
            let mut shared = true;
            for &member in law {
                if distinct(member, head)? {
                    shared = false;
                    break;
                }
            }
            if shared {
                joined = Some(index);
                break;
            }
        }
        match joined {
            Some(index) => laws[index].push(head),
            None => laws.push(vec![head]),
        }
    }
    let mut relations = Vec::new();
    for law in 1..laws.len() {
        for earlier in 0..law {
            relations.push(LawRelation {
                law,
                earlier,
                comparison: compare_heads(governor, operators, laws[law][0], laws[earlier][0])?,
            });
        }
    }
    Ok(RoutingLaws { laws, relations })
}

/// An attention step's observability letters: one per routing law, the law's summed
/// value/output transport `T = diag(ω) Σ_{h∈law} O_h V_g(h) diag(γ)` on the block's input,
/// `width × width`, with `γ` the input norm's gain (pre-norm) and `ω` the attention-output
/// norm's gain (post-norm) when declared. The only constructor is [`attention_letters`], so an
/// attention step's letters are always the laws' and never one per head. A value bias makes
/// the operators homogeneous; its column is a constant write that no linear pull-back reads,
/// and it is dropped. The transports hold their memory reservation.
#[derive(Debug)]
pub struct RoutingLawLetters {
    pub laws: RoutingLaws,
    transports: Vec<Array2<f64>>,
    bands: Vec<f64>,
    footprint: MemoryReservation,
}

impl RoutingLawLetters {
    /// The laws' transports, in law order.
    pub fn transports(&self) -> &[Array2<f64>] {
        &self.transports
    }

    /// A Frobenius bound on each transport's distance from the exact product of the stored
    /// tensors: the `k`-term accumulation `γ_k |L| |R|ᵀ` over the law's stacked factors, plus
    /// the value factors' formation defect.
    pub fn bands(&self) -> &[f64] {
        &self.bands
    }

    /// Bytes the transports hold on the process memory governor.
    pub fn reserved_bytes(&self) -> usize {
        self.footprint.bytes()
    }
}

/// The routing laws of `native` and their observability letters. `input_gain` (the preceding
/// residual norm's gain, pre-norm) is folded into both operator families; `output_gain` (the
/// gain of a norm applied to the attention output before the residual add, post-norm) scales
/// the transports' rows. The post-norm normalizer itself is one positive scalar per token
/// shared by every head, so it neither enters the laws nor the letters.
///
/// Each transport is `k`-term inner products of the stacked factors (`k` the law's total
/// rank), within `γ_k |L| |R|ᵀ`; the value factors carry their formation defect, and folding
/// `ω` into the left factor rounds each entry once more.
pub fn attention_letters(
    governor: &MemoryGovernor,
    native: &NativeAttention,
    input_gain: Option<ArrayView1<'_, f64>>,
    output_gain: Option<ArrayView1<'_, f64>>,
) -> Result<RoutingLawLetters, JointRefusal> {
    let width = native.geometry().model_dim;
    if let Some(gain) = output_gain {
        expect_shape("output gain", (width, 1), (gain.len(), 1))?;
        if gain.iter().any(|entry| !entry.is_finite()) {
            return Err(JointRefusal::NonFinite { what: "output gain" });
        }
    }
    let operators = query_key_operators(native, input_gain)?;
    let laws = routing_laws(governor, native, &operators)?;
    drop(operators);
    let value_output = value_output_operators(native, input_gain)?;
    // The transports, and one product and its magnitude while a law is formed.
    let cells = width.saturating_mul(width).saturating_mul(laws.laws.len().saturating_add(2));
    let footprint = governor
        .try_reserve(cells.saturating_mul(std::mem::size_of::<f64>()), "routing-law letters")
        .map_err(JointRefusal::Memory)?;
    let folding = if output_gain.is_some() { accumulation_growth(1) } else { 0.0 };
    let (mut transports, mut bands) = (Vec::with_capacity(laws.laws.len()), Vec::with_capacity(laws.laws.len()));
    for law in &laws.laws {
        let lefts: Vec<_> = law.iter().map(|&head| value_output.head(head).left()).collect();
        let rights: Vec<_> = law.iter().map(|&head| value_output.head(head).right()).collect();
        let mut left = concatenate(Axis(1), &lefts).map_err(shape_failed)?;
        let right = concatenate(Axis(1), &rights).map_err(shape_failed)?;
        if let Some(gain) = output_gain {
            for (row, &scale) in left.rows_mut().into_iter().zip(gain.iter()) {
                row.into_iter().for_each(|entry| *entry *= scale);
            }
        }
        let product = fast_abt(&left, &right);
        let magnitude = fast_abt(&left.mapv(f64::abs), &right.mapv(f64::abs));
        let relative = up(up(accumulation_growth(left.ncols().max(1)) + value_output.formation_defect()) + folding);
        let transport = product.slice(s![..width, ..width]).to_owned();
        let band = up(up(relative * upper_norm(magnitude.slice(s![..width, ..width]))) * up(1.0 + folding));
        transports.push(transport);
        bands.push(up(band + up(left.ncols() as f64 * SUBNORMAL_SPACING)));
    }
    Ok(RoutingLawLetters { laws, transports, bands, footprint })
}

#[cfg(test)]
#[path = "joint_operators_tests.rs"]
mod tests;
