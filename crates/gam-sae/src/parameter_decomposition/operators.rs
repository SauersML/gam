//! Gauge and operator structure of parameter edits (#2951 P1, P2, P4).
//!
//! Three facts decide what an intervention on a factored or rotational edit
//! means, and each has its own type here.
//!
//! * **Mask gauge (P1).** A rank-`r` operator `P = U Vᵀ` with full-column-rank
//!   factors is the same operator after `U → U S`, `V → V S⁻ᵀ` for every
//!   `S ∈ GL(r)`. An internal mask `M` held fixed while the basis changes gives
//!   `U S M S⁻¹ Vᵀ`, which equals `U M Vᵀ` for every `S ∈ GL(r)` iff `M = cI`:
//!   `S M = M S` for all `S`, diagonal `S` forces `M` diagonal and permutations
//!   force equal entries. Over a declared block gauge `GL(r₁) ⊕ … ⊕ GL(r_K)` the
//!   same argument leaves exactly the block scalars `⊕ c_k I_{r_k}`, one tied
//!   control per block. Any other mask names an intervention only together with
//!   the basis it was written in, and is carried as an operator, `M → S⁻¹ M S`.
//!   Independent diagonal masks on an arbitrary basis of one block are not
//!   intrinsic interventions. [`mask_gauge_evidence`] reports the verdict as an
//!   [`EvidenceStatus`]: exact at 0, or a counterexample with its witness.
//! * **Curved versus straight paths (P2).** For a rotation
//!   `W = I + U (R(α) − I) Uᵀ` in orthonormal planes, the angle path
//!   `W(t) = I + U (R(tα) − I) Uᵀ` is orthogonal for every `t` and moves the
//!   in-plane angle, `R_{tα} D(φ) = D(φ + tα)` with `D(φ) = (cos φ, sin φ)`. The
//!   chord path `L(t) = (1 − t) I + t W`, which is what the residual anchor's mask
//!   does to `Δ = W − I`, satisfies `L(t)ᵀ L(t) = [1 − 2t(1 − t)(1 − cos α)] I` on
//!   each plane, so its midpoint radius is `|cos(α/2)|`. The two paths answer
//!   different questions: removing a contribution versus varying an angle.
//! * **Composition (P4).** `(I + Δ_A)(I + Δ_B) = I + Δ_A + Δ_B + Δ_A Δ_B`. The
//!   cross term is induced by the composition and is not a third mechanism;
//!   dropping it turns Compose into Sum. The order-swap gap of two edits is
//!   exactly the commutator `[Δ_A, Δ_B]`, and for rotation generators the edit
//!   loop is `e^{sA} e^{tB} e^{−sA} e^{−tB} = I + st [A, B] + O(s²t + st²)`.
//!
//! Every operator is applied to a vector: nothing here forms a `d × d` matrix.
//! The rotation blocks come from the `SO(2)` owner in `gam-geometry`.

use std::ops::Range;

use gam_geometry::manifolds::lie_so::rho_so2;
use gam_linalg::faer_ndarray::{FaerSvd, fast_ab, fast_abt, fast_atb, fast_atv, fast_av};
use gam_linalg::roundoff::{accumulation_growth, factor_singular_band};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

/// Why an operator construction or application was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatorRefusal {
    /// A dimension disagrees with the object it is applied to.
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// An input carries a non-finite entry.
    NonFinite { what: &'static str },
    /// No gauge block was declared, or a declared block has no coordinates.
    EmptyGaugeBlock,
    /// A factor of `P = U Vᵀ` has fewer singular values above its SVD's rounding
    /// band than the declared rank. The operator then has a gauge larger than the
    /// declared one, and a basis-dependent verdict would not be a counterexample.
    RankDeficientFactor {
        factor: &'static str,
        resolved: usize,
        declared: usize,
    },
    /// A gauge change couples coordinates of two declared blocks, so it is not an
    /// element of the declared group.
    GaugeOutsideDeclaredGroup { row: usize, col: usize },
    /// A gauge change has fewer singular values above its SVD's rounding band
    /// than its order.
    SingularGaugeChange { resolved: usize, order: usize },
    /// A plane basis has an odd column count, so its columns do not pair into
    /// planes.
    OddPlaneBasis { columns: usize },
    /// A plane basis is further from orthonormal than any stably orthogonalized
    /// basis of its shape: `‖UᵀU − I‖_F` exceeds the floor derived at
    /// [`PlaneRotation::new`].
    NonOrthonormalPlaneBasis { defect: f64, floor: f64 },
    /// A rotary declaration lists a coordinate outside `0..dim`, or lists one twice.
    InvalidRotaryDeclaration { coordinate: usize },
    /// The linear-algebra backend failed to decompose a matrix.
    DecompositionFailed { what: &'static str, detail: String },
    /// The evidence constructor refused the status, e.g. a counterexample whose shift overflowed
    /// to a non-finite value.
    Evidence(EvidenceStatusError),
}

/// A declared implementation gauge of an `r`-coordinate internal space: the group
/// `GL(r₁) ⊕ … ⊕ GL(r_K)` acting on consecutive coordinate blocks.
///
/// One block of size `r` is the full `GL(r)` of an unstructured rank-`r` factor.
/// `r` blocks of size one is the per-component scale gauge of `r` rank-one
/// components `u_c v_cᵀ`, under which independent diagonal masks are intrinsic.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeBlocks {
    /// Block `k` holds coordinates `offsets[k]..offsets[k + 1]`.
    offsets: Vec<usize>,
}

impl GaugeBlocks {
    pub fn new(sizes: &[usize]) -> Result<Self, OperatorRefusal> {
        if sizes.is_empty() || sizes.contains(&0) {
            return Err(OperatorRefusal::EmptyGaugeBlock);
        }
        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0);
        for &size in sizes {
            offsets.push(offsets[offsets.len() - 1] + size);
        }
        Ok(Self { offsets })
    }

    /// The internal dimension `r = Σ r_k`.
    pub fn rank(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    pub fn block_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The coordinates of block `k`.
    pub fn block(&self, k: usize) -> Range<usize> {
        self.offsets[k]..self.offsets[k + 1]
    }

    /// The block holding coordinate `i`: the offsets start at 0 and increase
    /// strictly, so it is the last block whose first coordinate is at most `i`.
    fn block_of(&self, i: usize) -> usize {
        self.offsets.partition_point(|&offset| offset <= i) - 1
    }
}

/// An internal mask on a factored operator, typed by how it transforms under the
/// operator's declared gauge.
#[derive(Clone, Debug, PartialEq)]
pub enum InternalMask {
    /// `⊕ c_k I_{r_k}`, one tied control per declared gauge block. It commutes with
    /// every element of the declared group, so `U M Vᵀ` does not depend on the
    /// basis the factors were written in.
    BlockScalar(Vec<f64>),
    /// An `r × r` operator mask written in the internal basis of the factors it is
    /// carried with. A gauge change `U → U S`, `V → V S⁻ᵀ` carries it to
    /// `S⁻¹ M S`, so the intervention is covariant; the matrix alone names no
    /// intervention.
    Operator(Array2<f64>),
}

/// Whether a fixed internal mask is an intrinsic intervention under a declared
/// gauge.
#[derive(Clone, Debug, PartialEq)]
pub enum MaskGaugeVerdict {
    /// The mask is exactly `⊕ c_k I_{r_k}`: it commutes with the whole declared
    /// group.
    Intrinsic { controls: Vec<f64> },
    /// A gauge change `S` in the declared group with `S M S⁻¹ ≠ M`, the internal
    /// entry it moves, and by how much. With full-column-rank factors the fixed
    /// mask gives a different intervention after `U → U S`, `V → V S⁻ᵀ`: a
    /// counterexample, not an estimate.
    BasisDependent {
        gauge_change: Array2<f64>,
        moved: (usize, usize),
        shift: f64,
    },
}

/// Classify a fixed `r × r` internal mask under a declared gauge (P1).
///
/// `M` commutes with every element of `GL(r₁) ⊕ … ⊕ GL(r_K)` iff it is a block
/// scalar. The check is algebraic on the declared entries and carries no
/// tolerance: an entry that differs by any amount moves the intervention by that
/// amount along the witness. The witness is the reflection `S = I − 2 e_i e_iᵀ`,
/// which negates an off-diagonal entry `M_ij`, or the transposition of two
/// coordinates of one block, which swaps two unequal diagonal entries. Both lie
/// in the declared group.
pub fn classify_internal_mask(
    mask: ArrayView2<'_, f64>,
    gauge: &GaugeBlocks,
) -> Result<MaskGaugeVerdict, OperatorRefusal> {
    let r = gauge.rank();
    check_square("internal mask order", r, mask)?;
    if !mask.iter().all(|entry| entry.is_finite()) {
        return Err(OperatorRefusal::NonFinite {
            what: "internal mask",
        });
    }
    for ((i, j), &entry) in mask.indexed_iter() {
        if i != j && entry != 0.0 {
            let mut gauge_change = Array2::<f64>::eye(r);
            gauge_change[[i, i]] = -1.0;
            return Ok(MaskGaugeVerdict::BasisDependent {
                gauge_change,
                moved: (i, j),
                shift: -2.0 * entry,
            });
        }
    }
    let mut controls = Vec::with_capacity(gauge.block_count());
    for k in 0..gauge.block_count() {
        let block = gauge.block(k);
        let lead = block.start;
        let control = mask[[lead, lead]];
        for i in block.start + 1..block.end {
            if mask[[i, i]] != control {
                let mut gauge_change = Array2::<f64>::eye(r);
                gauge_change[[lead, lead]] = 0.0;
                gauge_change[[i, i]] = 0.0;
                gauge_change[[lead, i]] = 1.0;
                gauge_change[[i, lead]] = 1.0;
                return Ok(MaskGaugeVerdict::BasisDependent {
                    gauge_change,
                    moved: (lead, lead),
                    shift: mask[[i, i]] - control,
                });
            }
        }
        controls.push(control);
    }
    Ok(MaskGaugeVerdict::Intrinsic { controls })
}

/// A declared implementation gauge of an `r`-coordinate internal space, named by its
/// group.
///
/// A fixed internal mask is an intrinsic intervention iff it lies in the group's
/// commutant. [`classify_under`] decides that on the declared entries; otherwise it
/// returns a witness that lies in the group and moves the fixed mask's intervention.
#[derive(Clone, Debug, PartialEq)]
pub enum DeclaredGauge {
    /// `GL(r₁) ⊕ … ⊕ GL(r_K)`, with commutant `⊕ c_k I_{r_k}`.
    Blocks(GaugeBlocks),
    /// The permutations `S_r` of `units` hidden units, acting as `M → P M Pᵀ`. This is
    /// the gauge of an activation that commutes with permutations and nothing larger
    /// (GELU, SiLU). Its commutant is `span{I, 11ᵀ}`.
    UnitPermutations { units: usize },
    /// Unit permutations with per-unit scalings: `(ℝ_{>0})^r ⋊ S_r` for ReLU, and
    /// `(ℝ^×)^r ⋊ S_r` for a SwiGLU up/down pair or a norm gain. Both groups have
    /// commutant `cI`, and every witness [`classify_under`] returns lies in both, so the
    /// declaration does not name the sign.
    ScaledPermutations { units: usize },
    /// The commutant of a rotary map `R_1` on head coordinates,
    /// `⊕_k GL(m_k, ℂ) ⊕ GL(n_pass)`, whose own commutant is
    /// `⊕_k (α_k I + β_k J_k) ⊕ γ I`.
    RotaryCommutant(RotaryGroups),
}

impl DeclaredGauge {
    /// The internal dimension `r`.
    pub fn rank(&self) -> usize {
        match self {
            DeclaredGauge::Blocks(blocks) => blocks.rank(),
            DeclaredGauge::UnitPermutations { units } | DeclaredGauge::ScaledPermutations { units } => *units,
            DeclaredGauge::RotaryCommutant(groups) => groups.dim(),
        }
    }
}

/// The frequency groups of a rotary map on head coordinates.
///
/// Group `k` lists the `(a, b)` coordinate pairs of the planes rotating at one frequency
/// in `(0, π)`. For each plane, `x_a + i x_b` is the complex coordinate and
/// `J_k (x_a, x_b) = (−x_b, x_a)`. `pass_through` holds the coordinates the map fixes.
///
/// Detecting the frequencies is the caller's job. Since `R_Δ = R_1^Δ`, and frequencies in
/// `(0, π)` give eigenvalues `e^{±iθ}` that differ between groups and differ from the
/// pass-through eigenvalue 1, an `S` commuting with every `R_Δ` is block diagonal and
/// commutes with each `J_k`.
#[derive(Clone, Debug, PartialEq)]
pub struct RotaryGroups {
    plane_groups: Vec<Vec<(usize, usize)>>,
    pass_through: Range<usize>,
    dim: usize,
}

impl RotaryGroups {
    /// `dim = 2 Σ m_k + |pass_through|`.
    ///
    /// Refuses an empty group, and any coordinate listed outside `0..dim` or listed twice.
    /// Exactly `dim` coordinates are listed, so with no duplicate and none out of range they
    /// cover `0..dim` exactly once.
    pub fn new(plane_groups: Vec<Vec<(usize, usize)>>, pass_through: Range<usize>) -> Result<Self, OperatorRefusal> {
        if plane_groups.iter().any(|group| group.is_empty()) {
            return Err(OperatorRefusal::EmptyGaugeBlock);
        }
        let dim = 2 * plane_groups.iter().map(Vec::len).sum::<usize>() + pass_through.len();
        let mut covered = vec![false; dim];
        let listed = plane_groups
            .iter()
            .flatten()
            .flat_map(|&(a, b)| [a, b])
            .chain(pass_through.clone());
        for coordinate in listed {
            if coordinate >= dim || covered[coordinate] {
                return Err(OperatorRefusal::InvalidRotaryDeclaration { coordinate });
            }
            covered[coordinate] = true;
        }
        Ok(Self {
            plane_groups,
            pass_through,
            dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn plane_groups(&self) -> &[Vec<(usize, usize)>] {
        &self.plane_groups
    }

    pub fn pass_through(&self) -> Range<usize> {
        self.pass_through.clone()
    }

    /// The block of each coordinate: `k` for group `k`, and `plane_groups.len()` for the
    /// pass-through.
    fn block_ids(&self) -> Vec<usize> {
        let mut ids = vec![self.plane_groups.len(); self.dim];
        for (k, group) in self.plane_groups.iter().enumerate() {
            for &(a, b) in group {
                ids[a] = k;
                ids[b] = k;
            }
        }
        ids
    }
}

/// Classify a fixed internal mask under a declared gauge (P1).
///
/// The checks compare declared entries with `==` and carry no tolerance: an entry that
/// differs by any amount moves the intervention by that amount along the witness.
/// Every witness lies in the declared group. Witnesses are built from permutations,
/// reflections, `J_k` and scalings by 2, so `S M S⁻¹` only permutes, negates or doubles
/// entries.
///
/// Witnesses by variant:
/// - `Blocks`: as [`classify_internal_mask`].
/// - `UnitPermutations`:
///   - unequal diagonal entries: the transposition `(0 i)`;
///   - an off-diagonal entry unequal to `M_01`: the permutation sending `(i, j)` to `(0, 1)`.
///     It exists because `S_r` is 2-transitive on ordered pairs.
/// - `ScaledPermutations`:
///   - an off-diagonal `M_ij ≠ 0`: scale 2 on coordinate `i`, which doubles the entry and
///     lies in both scale groups;
///   - unequal diagonal entries: the transposition.
/// - `RotaryCommutant`, checked in this order:
///   - a cross-block entry: scale 2 on the row's block;
///   - a block that is not complex-linear, `M J_k ≠ J_k M`: `J_k` on group `k`;
///   - a complex off-diagonal entry: scale 2 on plane `p`;
///   - unequal complex diagonal entries: the complex transposition of planes `0` and `p`,
///     moving both coordinates together;
///   - the pass-through: the GL witnesses.
///
/// `Intrinsic { controls }` by variant:
/// - `Blocks`: one per block.
/// - `UnitPermutations`: `[diagonal, off_diagonal]`, or `[diagonal]` for one unit.
/// - `ScaledPermutations`: `[c]`.
/// - `RotaryCommutant`: `[α_1, β_1, …, α_K, β_K]`, followed by `γ` when the pass-through is
///   non-empty. `α_k = M[a, a]` and `β_k = M[b, a]` on the group's first plane.
pub fn classify_under(mask: ArrayView2<'_, f64>, gauge: &DeclaredGauge) -> Result<MaskGaugeVerdict, OperatorRefusal> {
    match gauge {
        DeclaredGauge::Blocks(blocks) => classify_internal_mask(mask, blocks),
        DeclaredGauge::UnitPermutations { units } => {
            check_declared_mask(mask, *units)?;
            Ok(classify_unit_permutations(mask))
        }
        DeclaredGauge::ScaledPermutations { units } => {
            check_declared_mask(mask, *units)?;
            Ok(classify_scaled_permutations(mask))
        }
        DeclaredGauge::RotaryCommutant(groups) => {
            check_declared_mask(mask, groups.dim())?;
            Ok(classify_rotary(mask, groups))
        }
    }
}

/// A point of the domain of [`mask_gauge_evidence`]: a gauge change in the declared group and
/// the internal entry whose movement it measures.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskWitness {
    /// `S`, an element of the declared group.
    pub gauge_change: Array2<f64>,
    /// `(i, j)`, the entry of `S M S⁻¹ − M` the value measures.
    pub moved: (usize, usize),
}

/// The evidence status of P1's claim for one fixed mask under one declared gauge.
pub type MaskGaugeEvidence = EvidenceStatus<MaskWitness, DeclaredGauge>;

/// P1 as evidence. The claim is that a fixed internal mask names an intervention that does
/// not depend on the basis: `sup Q <= 0` with `Q(S, (i, j)) = |(S M S⁻¹ − M)_ij|` over the
/// gauge changes `S` of the declared group and the internal entries `(i, j)`. With
/// full-column-rank factors ([`FactoredOperator::new`] refuses others), a moved entry moves
/// `U M Vᵀ`.
///
/// - [`MaskGaugeVerdict::Intrinsic`] is `Exact` with value 0, basis `Algebraic` and the
///   declared gauge as its domain. A mask in the group's commutant commutes with every
///   element, and nothing is evaluated.
/// - [`MaskGaugeVerdict::BasisDependent`] is a `Counterexample` at threshold 0, with the
///   witness and value `|shift|`. [`classify_under`]'s witnesses only permute, negate or double
///   entries, so each shift is exact (`−2 M_ij` or `M_ij`) or one rounded subtraction of two
///   declared entries, `fl(a − b) = (a − b)(1 + δ)` with `|δ| <= u`. Then
///   `|fl(a − b) − (a − b)| <= γ₁ |fl(a − b)| <= ε |fl(a − b)|`. `ε |shift|` is a power-of-two
///   scaling, exact unless it underflows, and rounding it up once covers the underflow. A
///   subnormal difference of two floats is exact, because both are multiples of the smallest
///   subnormal, so a subnormal shift carries no error. `a ≠ b` gives `fl(a − b) ≠ 0`, so every
///   finite shift exceeds its error.
/// - A shift that overflows is refused. That happens when `|M_ij| > f64::MAX / 2` under a
///   reflection, or when two opposite-sign entries differ by more than `f64::MAX`.
///   [`EvidenceStatus::counterexample`] refuses the non-finite value, and the refusal returns as
///   [`OperatorRefusal::Evidence`] although the mask is basis-dependent.
pub fn mask_gauge_evidence(mask: ArrayView2<'_, f64>, gauge: &DeclaredGauge) -> Result<MaskGaugeEvidence, OperatorRefusal> {
    let status = match classify_under(mask, gauge)? {
        MaskGaugeVerdict::Intrinsic { .. } => EvidenceStatus::exact(0.0, 0.0, ExactBasis::Algebraic, None, gauge.clone()),
        MaskGaugeVerdict::BasisDependent {
            gauge_change,
            moved,
            shift,
        } => {
            let value = shift.abs();
            let numerical_error = if value < f64::MIN_POSITIVE {
                0.0
            } else {
                (f64::EPSILON * value).next_up()
            };
            EvidenceStatus::counterexample(value, numerical_error, 0.0, MaskWitness { gauge_change, moved })
        }
    };
    status.map_err(OperatorRefusal::Evidence)
}

fn check_declared_mask(mask: ArrayView2<'_, f64>, order: usize) -> Result<(), OperatorRefusal> {
    if order == 0 {
        return Err(OperatorRefusal::EmptyGaugeBlock);
    }
    check_square("internal mask order", order, mask)?;
    if !mask.iter().all(|entry| entry.is_finite()) {
        return Err(OperatorRefusal::NonFinite {
            what: "internal mask",
        });
    }
    Ok(())
}

fn basis_dependent(gauge_change: Array2<f64>, moved: (usize, usize), shift: f64) -> MaskGaugeVerdict {
    MaskGaugeVerdict::BasisDependent {
        gauge_change,
        moved,
        shift,
    }
}

/// `I − 2 e_i e_iᵀ`.
fn reflection(r: usize, i: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::eye(r);
    s[[i, i]] = -1.0;
    s
}

/// The permutation matrix with `P e_k = e_{sigma[k]}`, so `(P M Pᵀ)[σ(k), σ(l)] = M[k, l]`.
fn permutation_matrix(sigma: &[usize]) -> Array2<f64> {
    let mut p = Array2::<f64>::zeros((sigma.len(), sigma.len()));
    for (k, &image) in sigma.iter().enumerate() {
        p[[image, k]] = 1.0;
    }
    p
}

/// The transpositions of the listed coordinate pairs, applied together.
fn swap_coordinates(r: usize, pairs: &[(usize, usize)]) -> Array2<f64> {
    let mut sigma: Vec<usize> = (0..r).collect();
    for &(i, j) in pairs {
        sigma.swap(i, j);
    }
    permutation_matrix(&sigma)
}

/// Scale 2 on the listed coordinates and 1 elsewhere. The factor 2 is exact in binary,
/// so `(S M S⁻¹)_ij = 2 M_ij` for a row `i` in the list and a column `j` outside it.
fn scale_coordinates(r: usize, coordinates: &[usize]) -> Array2<f64> {
    let mut s = Array2::<f64>::eye(r);
    for &i in coordinates {
        s[[i, i]] = 2.0;
    }
    s
}

fn classify_unit_permutations(mask: ArrayView2<'_, f64>) -> MaskGaugeVerdict {
    let r = mask.nrows();
    let diagonal = mask[[0, 0]];
    for i in 1..r {
        if mask[[i, i]] != diagonal {
            return basis_dependent(swap_coordinates(r, &[(0, i)]), (0, 0), mask[[i, i]] - diagonal);
        }
    }
    if r == 1 {
        return MaskGaugeVerdict::Intrinsic {
            controls: vec![diagonal],
        };
    }
    let off_diagonal = mask[[0, 1]];
    for ((i, j), &entry) in mask.indexed_iter() {
        if i != j && entry != off_diagonal {
            let mut sigma = vec![0; r];
            sigma[j] = 1;
            let mut next = 2;
            for (k, image) in sigma.iter_mut().enumerate() {
                if k != i && k != j {
                    *image = next;
                    next += 1;
                }
            }
            return basis_dependent(permutation_matrix(&sigma), (0, 1), entry - off_diagonal);
        }
    }
    MaskGaugeVerdict::Intrinsic {
        controls: vec![diagonal, off_diagonal],
    }
}

fn classify_scaled_permutations(mask: ArrayView2<'_, f64>) -> MaskGaugeVerdict {
    let r = mask.nrows();
    for ((i, j), &entry) in mask.indexed_iter() {
        if i != j && entry != 0.0 {
            return basis_dependent(scale_coordinates(r, &[i]), (i, j), entry);
        }
    }
    let control = mask[[0, 0]];
    for i in 1..r {
        if mask[[i, i]] != control {
            return basis_dependent(swap_coordinates(r, &[(0, i)]), (0, 0), mask[[i, i]] - control);
        }
    }
    MaskGaugeVerdict::Intrinsic {
        controls: vec![control],
    }
}

fn classify_rotary(mask: ArrayView2<'_, f64>, groups: &RotaryGroups) -> MaskGaugeVerdict {
    let r = groups.dim();
    let ids = groups.block_ids();
    for ((u, v), &entry) in mask.indexed_iter() {
        if ids[u] != ids[v] && entry != 0.0 {
            let row_block: Vec<usize> = (0..r).filter(|&w| ids[w] == ids[u]).collect();
            return basis_dependent(scale_coordinates(r, &row_block), (u, v), entry);
        }
    }
    let mut controls = Vec::with_capacity(2 * groups.plane_groups.len() + 1);
    for group in &groups.plane_groups {
        for &(a_p, b_p) in group {
            for &(a_q, b_q) in group {
                let (upper_left, upper_right) = (mask[[a_p, a_q]], mask[[a_p, b_q]]);
                let (lower_left, lower_right) = (mask[[b_p, a_q]], mask[[b_p, b_q]]);
                if lower_right != upper_left || upper_right != -lower_left {
                    let mut j = Array2::<f64>::eye(r);
                    for &(a, b) in group {
                        j[[a, a]] = 0.0;
                        j[[b, b]] = 0.0;
                        j[[b, a]] = 1.0;
                        j[[a, b]] = -1.0;
                    }
                    // Conjugating by J sends [[A, B], [C, D]] to [[D, −C], [−B, A]].
                    return if lower_right != upper_left {
                        basis_dependent(j, (a_p, a_q), lower_right - upper_left)
                    } else {
                        basis_dependent(j, (a_p, b_q), -lower_left - upper_right)
                    };
                }
            }
        }
        let (a_0, b_0) = group[0];
        for &(a_p, b_p) in group {
            for &(a_q, ..) in group {
                if a_p != a_q && (mask[[a_p, a_q]] != 0.0 || mask[[b_p, a_q]] != 0.0) {
                    let s = scale_coordinates(r, &[a_p, b_p]);
                    return if mask[[a_p, a_q]] != 0.0 {
                        basis_dependent(s, (a_p, a_q), mask[[a_p, a_q]])
                    } else {
                        basis_dependent(s, (b_p, a_q), mask[[b_p, a_q]])
                    };
                }
            }
            if mask[[a_p, a_p]] != mask[[a_0, a_0]] || mask[[b_p, a_p]] != mask[[b_0, a_0]] {
                let s = swap_coordinates(r, &[(a_0, a_p), (b_0, b_p)]);
                return if mask[[a_p, a_p]] != mask[[a_0, a_0]] {
                    basis_dependent(s, (a_0, a_0), mask[[a_p, a_p]] - mask[[a_0, a_0]])
                } else {
                    basis_dependent(s, (b_0, a_0), mask[[b_p, a_p]] - mask[[b_0, a_0]])
                };
            }
        }
        controls.push(mask[[a_0, a_0]]);
        controls.push(mask[[b_0, a_0]]);
    }
    let pass: Vec<usize> = groups.pass_through().collect();
    if let Some(&lead) = pass.first() {
        for &u in &pass {
            for &v in &pass {
                if u != v && mask[[u, v]] != 0.0 {
                    return basis_dependent(reflection(r, u), (u, v), -2.0 * mask[[u, v]]);
                }
            }
            if mask[[u, u]] != mask[[lead, lead]] {
                return basis_dependent(swap_coordinates(r, &[(lead, u)]), (lead, lead), mask[[u, u]] - mask[[lead, lead]]);
            }
        }
        controls.push(mask[[lead, lead]]);
    }
    MaskGaugeVerdict::Intrinsic { controls }
}

/// A rank-`r` operator `P = U Vᵀ` with full-column-rank factors and a declared
/// gauge, stored as its `d_out × r` and `d_in × r` factors and applied to vectors
/// only.
#[derive(Clone, Debug)]
pub struct FactoredOperator {
    u: Array2<f64>,
    v: Array2<f64>,
    gauge: GaugeBlocks,
}

impl FactoredOperator {
    /// Refuses factors whose column count is not the gauge's rank, non-finite
    /// entries, and a factor with a singular value inside its SVD's rounding band
    /// [`factor_singular_band`]: P1's characterization needs full column rank.
    pub fn new(u: Array2<f64>, v: Array2<f64>, gauge: GaugeBlocks) -> Result<Self, OperatorRefusal> {
        let r = gauge.rank();
        for (what, factor) in [("left factor", &u), ("right factor", &v)] {
            check_len(what, r, factor.ncols())?;
            if !factor.iter().all(|entry| entry.is_finite()) {
                return Err(OperatorRefusal::NonFinite { what });
            }
            let sigma = factor
                .svd(false, false)
                .map_err(|failure| decomposition_failed(what, &failure))?
                .1;
            let resolved = resolved_singular_count(&sigma, factor.nrows(), factor.ncols());
            if resolved < r {
                return Err(OperatorRefusal::RankDeficientFactor {
                    factor: what,
                    resolved,
                    declared: r,
                });
            }
        }
        Ok(Self { u, v, gauge })
    }

    pub fn gauge(&self) -> &GaugeBlocks {
        &self.gauge
    }

    /// `y = U M Vᵀ x`, through the `r` internal coordinates only.
    pub fn apply(&self, mask: &InternalMask, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("operator input", self.v.nrows(), x.len())?;
        let internal = fast_atv(&self.v, &x);
        let masked = match mask {
            InternalMask::BlockScalar(controls) => {
                check_len("block-scalar controls", self.gauge.block_count(), controls.len())?;
                let mut scaled = internal;
                for (k, &control) in controls.iter().enumerate() {
                    for i in self.gauge.block(k) {
                        scaled[i] *= control;
                    }
                }
                scaled
            }
            InternalMask::Operator(matrix) => {
                check_square("operator mask order", self.gauge.rank(), matrix.view())?;
                fast_av(matrix, &internal)
            }
        };
        Ok(fast_av(&self.u, &masked))
    }

    /// Reparametrize by a gauge change `S` in the declared group, `U → U S` and
    /// `V → V S⁻ᵀ`, carrying the mask covariantly. A block-scalar mask is unchanged
    /// because it commutes with `S`; an operator mask becomes `S⁻¹ M S`. The
    /// operator and every masked intervention are unchanged up to roundoff.
    pub fn regauge(
        &self,
        gauge_change: ArrayView2<'_, f64>,
        mask: &InternalMask,
    ) -> Result<(Self, InternalMask), OperatorRefusal> {
        let r = self.gauge.rank();
        check_square("gauge change order", r, gauge_change)?;
        if !gauge_change.iter().all(|entry| entry.is_finite()) {
            return Err(OperatorRefusal::NonFinite {
                what: "gauge change",
            });
        }
        for ((i, j), &entry) in gauge_change.indexed_iter() {
            if entry != 0.0 && self.gauge.block_of(i) != self.gauge.block_of(j) {
                return Err(OperatorRefusal::GaugeOutsideDeclaredGroup { row: i, col: j });
            }
        }
        let inverse = invert_gauge_change(gauge_change)?;
        let carried = match mask {
            InternalMask::BlockScalar(controls) => {
                check_len("block-scalar controls", self.gauge.block_count(), controls.len())?;
                InternalMask::BlockScalar(controls.clone())
            }
            InternalMask::Operator(matrix) => {
                check_square("operator mask order", r, matrix.view())?;
                InternalMask::Operator(fast_ab(&fast_ab(&inverse, matrix), &gauge_change))
            }
        };
        let regauged = Self {
            u: fast_ab(&self.u, &gauge_change),
            v: fast_abt(&self.v, &inverse),
            gauge: self.gauge.clone(),
        };
        Ok((regauged, carried))
    }

    /// This operator under `mask` read as the residual edit `Δ = U M Vᵀ` of a
    /// square map; refuses a rectangular operator.
    pub fn residual_edit<'a>(&'a self, mask: &'a InternalMask) -> Result<MaskedOperator<'a>, OperatorRefusal> {
        check_len("residual edit output dimension", self.v.nrows(), self.u.nrows())?;
        Ok(MaskedOperator { operator: self, mask })
    }
}

fn decomposition_failed(what: &'static str, failure: &gam_linalg::faer_ndarray::FaerLinalgError) -> OperatorRefusal {
    OperatorRefusal::DecompositionFailed {
        what,
        detail: format!("{failure:?}"),
    }
}

/// The number of singular values of a `rows × cols` matrix above the SVD's
/// rounding band `max(rows, cols)·ε·σ_max` ([`factor_singular_band`]).
fn resolved_singular_count(sigma: &Array1<f64>, rows: usize, cols: usize) -> usize {
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(rows, cols, sigma_max);
    sigma.iter().filter(|&&value| value > band).count()
}

/// `S⁻¹ = V Σ⁻¹ Uᵀ` from the SVD `S = U Σ Vᵀ`, refusing a singular value inside
/// the SVD's rounding band.
fn invert_gauge_change(gauge_change: ArrayView2<'_, f64>) -> Result<Array2<f64>, OperatorRefusal> {
    let what = "gauge change";
    let order = gauge_change.nrows();
    let (left, sigma, right_t) = gauge_change
        .svd(true, true)
        .map_err(|failure| decomposition_failed(what, &failure))?;
    let resolved = resolved_singular_count(&sigma, order, order);
    if resolved < order {
        return Err(OperatorRefusal::SingularGaugeChange { resolved, order });
    }
    let (Some(left), Some(right_t)) = (left, right_t) else {
        return Err(OperatorRefusal::DecompositionFailed {
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

/// A parameter edit written as a residual `W = I + Δ` on a `dim`-dimensional
/// space and applied matrix-free.
pub trait ResidualEdit {
    /// The dimension of the space `W` acts on.
    fn dim(&self) -> usize;

    /// `Δ x`.
    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal>;
}

/// A square factored operator under an internal mask, as the residual edit
/// `Δ = U M Vᵀ`. Built by [`FactoredOperator::residual_edit`].
#[derive(Clone, Copy, Debug)]
pub struct MaskedOperator<'a> {
    operator: &'a FactoredOperator,
    mask: &'a InternalMask,
}

impl ResidualEdit for MaskedOperator<'_> {
    fn dim(&self) -> usize {
        self.operator.v.nrows()
    }

    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        self.operator.apply(self.mask, x)
    }
}

/// Which path a rotation edit is read along (P2). The two are different
/// experiments, so a path is never inferred from a bare scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotationPath {
    /// `W(t) = I + U (R(tα) − I) Uᵀ`: a structured-edit coordinate that varies the
    /// angles. It is orthogonal up to the basis defect `δ = ‖UᵀU − I‖_F`, which
    /// [`PlaneRotation::new`] bounds by its refusal floor.
    ///
    /// With `UᵀU = I + F`, the identity terms of `WᵀW` cancel because
    /// `(Rᵀ − I)(R − I) = 2I − R − Rᵀ`. That leaves `U (Rᵀ − I) F (R − I) Uᵀ`, so
    /// `‖W(t)ᵀ W(t) − I‖₂ ≤ ‖U‖₂² ‖R − I‖₂² δ ≤ 4δ(1 + δ)`.
    Angle(f64),
    /// `L(t) = I + t U (R(α) − I) Uᵀ`: the residual anchor's mask on `Δ = W − I`.
    /// It removes the contribution and contracts plane `j` by
    /// [`PlaneRotation::chord_radii`].
    Chord(f64),
}

/// A rotation in `k` mutually orthogonal planes, `W = I + U (R(α) − I) Uᵀ`, where
/// columns `2j` and `2j + 1` of the `d × 2k` basis `U` span plane `j` and
/// `R(α) = ⊕_j R_{α_j}` with `R_θ = [[cos θ, −sin θ], [sin θ, cos θ]]`.
#[derive(Clone, Debug)]
pub struct PlaneRotation {
    basis: Array2<f64>,
    angles: Array1<f64>,
    orthonormality_defect: f64,
    orthonormality_floor: f64,
}

impl PlaneRotation {
    /// Refuses a basis whose measured defect `δ = ‖UᵀU − I‖_F` exceeds the floor
    /// `2k·max(d, 2k)·ε + γ_d ‖U‖_F²`. The floor has two parts, and only the second is
    /// derived.
    /// - **Orthogonalization, a convention.** `2k·max(d, 2k)·ε` is the `m·n·ε` band of
    ///   [`factor_singular_band`] at `σ = 1`. It is not derived for this computation.
    ///   Householder QR's bound `‖Q̂ − Q‖_F ≤ √n·γ̃_{mn}` (Higham, Thm 19.4) carries an
    ///   unstated constant, so no constant-free floor exists for every stable
    ///   orthogonalization. The test `householder_qr_bases_stay_under_the_plane_basis_floor`
    ///   checks FaerQr bases at `d ∈ {8, 64, 512}` and `2k ∈ {2, 8, 32}` from inputs graded
    ///   down to `1e-12`, and places the refusal at the floor itself.
    /// - **Forming `UᵀU`.** Each entry is an inner product of `d` terms. By
    ///   Cauchy–Schwarz the rounding is at most `γ_d ‖U‖_F²` in Frobenius norm.
    ///
    /// An accepted basis keeps its measured defect `δ ≤ floor`. Every identity of the
    /// module holds for this edit up to `δ` (see [`RotationPath::Angle`]).
    pub fn new(basis: Array2<f64>, angles: Array1<f64>) -> Result<Self, OperatorRefusal> {
        if basis.ncols() % 2 != 0 {
            return Err(OperatorRefusal::OddPlaneBasis {
                columns: basis.ncols(),
            });
        }
        check_len("plane angles", basis.ncols() / 2, angles.len())?;
        if !basis.iter().all(|entry| entry.is_finite()) {
            return Err(OperatorRefusal::NonFinite { what: "plane basis" });
        }
        if !angles.iter().all(|angle| angle.is_finite()) {
            return Err(OperatorRefusal::NonFinite { what: "plane angles" });
        }
        let gram = fast_atb(&basis, &basis);
        let defect_sq: f64 = gram
            .indexed_iter()
            .map(|((i, j), &entry)| {
                let deviation = if i == j { entry - 1.0 } else { entry };
                deviation * deviation
            })
            .sum();
        let defect = defect_sq.sqrt();
        let (dim, internal) = (basis.nrows(), basis.ncols());
        let basis_sq: f64 = basis.iter().map(|entry| entry * entry).sum();
        let floor = internal as f64 * dim.max(internal) as f64 * f64::EPSILON + accumulation_growth(dim) * basis_sq;
        if defect > floor {
            return Err(OperatorRefusal::NonOrthonormalPlaneBasis { defect, floor });
        }
        Ok(Self {
            basis,
            angles,
            orthonormality_defect: defect,
            orthonormality_floor: floor,
        })
    }

    pub fn dim(&self) -> usize {
        self.basis.nrows()
    }

    pub fn plane_count(&self) -> usize {
        self.angles.len()
    }

    pub fn angles(&self) -> ArrayView1<'_, f64> {
        self.angles.view()
    }

    /// `‖UᵀU − I‖_F`, measured at construction.
    pub fn orthonormality_defect(&self) -> f64 {
        self.orthonormality_defect
    }

    /// The derived floor that [`PlaneRotation::new`] requires the defect not to exceed.
    pub fn orthonormality_floor(&self) -> f64 {
        self.orthonormality_floor
    }

    /// `Δ x` along `path`.
    pub fn apply_delta(&self, path: RotationPath, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        match path {
            RotationPath::Angle(t) => self.plane_delta(&self.angles.mapv(|alpha| t * alpha), x),
            RotationPath::Chord(t) => Ok(self.plane_delta(&self.angles, x)? * t),
        }
    }

    /// `x + Δ x` along `path`.
    pub fn apply(&self, path: RotationPath, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        Ok(self.apply_delta(path, x)? + &x)
    }

    /// This rotation read along `path` as a [`ResidualEdit`].
    pub fn edit(&self, path: RotationPath) -> RotationEdit<'_> {
        RotationEdit { rotation: self, path }
    }

    /// The singular value of the chord path `L(t)` on each plane,
    /// `√(1 − 2t(1 − t)(1 − cos α_j))`, which is `|cos(α_j/2)|` at `t = ½`.
    ///
    /// The radicand is evaluated as a sum of non-negative terms, so nothing
    /// cancels: `(1 − 2t)² + 4t(1 − t) cos²(α/2)` where `t(1 − t) ≥ 0`, and
    /// `1 + 4t(t − 1) sin²(α/2)` elsewhere.
    pub fn chord_radii(&self, t: f64) -> Array1<f64> {
        self.angles.mapv(|alpha| {
            let half = 0.5 * alpha;
            let spread = t * (1.0 - t);
            let squared = if spread >= 0.0 {
                let lever = 1.0 - 2.0 * t;
                lever * lever + 4.0 * spread * half.cos().powi(2)
            } else {
                1.0 - 4.0 * spread * half.sin().powi(2)
            };
            squared.sqrt()
        })
    }

    /// `A x = U Ω Uᵀ x` with `Ω = ⊕_j α_j J`, `J = [[0, −1], [1, 0]]`: the tangent of
    /// the angle path at `t = 0`, and `W(t) = exp(tA)` for an orthonormal basis.
    ///
    /// `A` is skew for every basis, since `(U Ω Uᵀ)ᵀ = U Ωᵀ Uᵀ = −A`. The edit loop's
    /// third-order bound (P4) holds only for skew generators, whose exponentials are
    /// orthogonal, and this is the only generator the module exposes.
    pub fn apply_generator(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("rotation input", self.dim(), x.len())?;
        let internal = fast_atv(&self.basis, &x);
        let mut turned = Array1::<f64>::zeros(internal.len());
        for (j, &alpha) in self.angles.iter().enumerate() {
            turned[2 * j] = -alpha * internal[2 * j + 1];
            turned[2 * j + 1] = alpha * internal[2 * j];
        }
        Ok(fast_av(&self.basis, &turned))
    }

    /// `U (R(θ) − I) Uᵀ x` for per-plane angles `θ`.
    fn plane_delta(&self, theta: &Array1<f64>, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("rotation input", self.dim(), x.len())?;
        let internal = fast_atv(&self.basis, &x);
        let rotations = rho_so2(theta.view());
        let mut moved = Array1::<f64>::zeros(internal.len());
        for j in 0..theta.len() {
            let (first, second) = (internal[2 * j], internal[2 * j + 1]);
            moved[2 * j] = (rotations[[j, 0, 0]] - 1.0) * first + rotations[[j, 0, 1]] * second;
            moved[2 * j + 1] = rotations[[j, 1, 0]] * first + (rotations[[j, 1, 1]] - 1.0) * second;
        }
        Ok(fast_av(&self.basis, &moved))
    }
}

/// A plane rotation read along one path, as a [`ResidualEdit`]. Built by
/// [`PlaneRotation::edit`].
#[derive(Clone, Copy, Debug)]
pub struct RotationEdit<'a> {
    rotation: &'a PlaneRotation,
    path: RotationPath,
}

impl ResidualEdit for RotationEdit<'_> {
    fn dim(&self) -> usize {
        self.rotation.dim()
    }

    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        self.rotation.apply_delta(self.path, x)
    }
}

/// The terms of the composition `(I + Δ_A)(I + Δ_B) x`, with `B` applied first
/// (P4).
#[derive(Clone, Debug, PartialEq)]
pub struct ComposeTerms {
    /// `Δ_B x`.
    pub first: Array1<f64>,
    /// `Δ_A x`.
    pub second: Array1<f64>,
    /// `Δ_A Δ_B x`: induced by the composition, with no control and no code of its
    /// own.
    pub cross: Array1<f64>,
}

impl ComposeTerms {
    /// `x + Δ_A x + Δ_B x`: the Sum of the two edits, which omits the cross term.
    pub fn sum(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        &self.first + &self.second + &x
    }

    /// `x + Δ_A x + Δ_B x + Δ_A Δ_B x`: the Compose.
    pub fn compose(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        self.sum(x) + &self.cross
    }
}

/// Split `(I + Δ_second)(I + Δ_first) x` into its two edit terms and the induced
/// cross term.
pub fn compose_terms(
    second: &dyn ResidualEdit,
    first: &dyn ResidualEdit,
    x: ArrayView1<'_, f64>,
) -> Result<ComposeTerms, OperatorRefusal> {
    check_len("composed edit dimensions", first.dim(), second.dim())?;
    check_len("composition input", first.dim(), x.len())?;
    let first_delta = first.apply_delta(x)?;
    let second_delta = second.apply_delta(x)?;
    let cross = second.apply_delta(first_delta.view())?;
    Ok(ComposeTerms {
        first: first_delta,
        second: second_delta,
        cross,
    })
}

/// `[Δ_A, Δ_B] x = Δ_A Δ_B x − Δ_B Δ_A x`: exactly the order-swap gap
/// `(I + Δ_A)(I + Δ_B) x − (I + Δ_B)(I + Δ_A) x`, since `I` commutes with both.
pub fn commutator_apply(
    a: &dyn ResidualEdit,
    b: &dyn ResidualEdit,
    x: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, OperatorRefusal> {
    check_len("commuted edit dimensions", a.dim(), b.dim())?;
    check_len("commutator input", a.dim(), x.len())?;
    let ab = a.apply_delta(b.apply_delta(x)?.view())?;
    let ba = b.apply_delta(a.apply_delta(x)?.view())?;
    Ok(ab - &ba)
}

fn check_len(what: &'static str, expected: usize, found: usize) -> Result<(), OperatorRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(OperatorRefusal::DimensionMismatch { what, expected, found })
    }
}

fn check_square(what: &'static str, order: usize, matrix: ArrayView2<'_, f64>) -> Result<(), OperatorRefusal> {
    check_len(what, order, matrix.nrows())?;
    check_len(what, order, matrix.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use gam_linalg::faer_ndarray::FaerQr;
    use ndarray::s;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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

    fn norm(v: &Array1<f64>) -> f64 {
        v.dot(v).sqrt()
    }

    fn frobenius(m: &Array2<f64>) -> f64 {
        m.iter().map(|entry| entry * entry).sum::<f64>().sqrt()
    }

    /// One modified Gram–Schmidt sweep over the columns.
    fn orthogonalize_columns(q: &mut Array2<f64>) {
        for j in 0..q.ncols() {
            for i in 0..j {
                let earlier = q.column(i).to_owned();
                let projection = earlier.dot(&q.column(j));
                q.column_mut(j).scaled_add(-projection, &earlier);
            }
            let length = q.column(j).dot(&q.column(j)).sqrt();
            q.column_mut(j).mapv_inplace(|entry| entry / length);
        }
    }

    /// `columns` orthonormal columns in `ℝ^dim`, from the Householder QR owner that
    /// spectral.rs's tests use. Taking the first `columns` columns gives the same
    /// basis whether `qr` returns a thin or a full `Q`.
    fn orthonormal_columns(rng: &mut StdRng, dim: usize, columns: usize) -> Array2<f64> {
        let q = uniform_matrix(rng, dim, columns).qr().expect("Householder QR of a uniform draw").0;
        q.slice(s![.., 0..columns]).to_owned()
    }

    /// `U_j D(φ)`: the unit vector at angle `φ` in plane `j` of `basis`.
    fn plane_vector(basis: &Array2<f64>, plane: usize, angle: f64) -> Array1<f64> {
        &basis.column(2 * plane) * angle.cos() + &basis.column(2 * plane + 1) * angle.sin()
    }

    /// Rounding band of `plane_vector` at `angle`: the angle's own sum, `cos`, `sin`,
    /// two products and an add each round once, on magnitudes at most
    /// `2 + |angle|`.
    fn plane_vector_band(angle: f64) -> f64 {
        accumulation_growth(6) * (2.0 + angle.abs())
    }

    fn mask_norm(mask: &InternalMask, gauge: &GaugeBlocks) -> f64 {
        match mask {
            InternalMask::BlockScalar(controls) => controls
                .iter()
                .enumerate()
                .map(|(k, control)| control * control * gauge.block(k).len() as f64)
                .sum::<f64>()
                .sqrt(),
            InternalMask::Operator(matrix) => frobenius(matrix),
        }
    }

    /// First-order rounding band of comparing `U M Vᵀ x` with the regauged
    /// `(U S) M' (V Ŝ⁻ᵀ)ᵀ x`.
    ///
    /// The computed inverse is the exact inverse of `S + E` with `‖E‖₂ ≤ r·ε·‖S‖₂`
    /// (the SVD backward error, [`factor_singular_band`]'s convention), formed from
    /// products of at most `2r` rounded terms whose absolute magnitudes
    /// `√r·‖Σ⁻¹‖_F·√r` majorizes. So `‖S Ŝ⁻¹ − I‖` is at most
    /// `η = κ_F·r·(ε + γ_{2r})` with `κ_F = ‖S‖_F ‖Ŝ⁻¹‖_F ≥ κ₂(S)`. A block-scalar
    /// mask meets `S Ŝ⁻¹` once and a carried operator mask twice, so `2η` covers
    /// both. The six products of the two evaluations (`Vᵀx`, `U S`, `V Ŝ⁻ᵀ`,
    /// `Ŝ⁻¹ M S`, `M z`, `U z`) each round at most `max(d_in, d_out, r)` terms per
    /// entry, on absolute magnitudes majorized by `κ_F² ‖U‖_F ‖M‖_F ‖V‖_F ‖x‖`.
    fn regauge_band(before: &FactoredOperator, gauge_change: &Array2<f64>, mask: &InternalMask, x: &Array1<f64>) -> f64 {
        let r = before.gauge.rank();
        let inverse = invert_gauge_change(gauge_change.view()).expect("the fixture's gauge change is invertible");
        let kappa = frobenius(gauge_change) * frobenius(&inverse);
        let eta = kappa * r as f64 * (f64::EPSILON + accumulation_growth(2 * r));
        let widest = before.u.nrows().max(before.v.nrows()).max(r);
        let stages = accumulation_growth(6 * widest) * kappa * kappa;
        (2.0 * eta + stages) * frobenius(&before.u) * mask_norm(mask, &before.gauge) * frobenius(&before.v) * norm(x)
    }

    /// First-order rounding band of one executed rotation edit
    /// `x + t U (R − I) Uᵀ x`, with `lever = max(1, |t|)`.
    ///
    /// `Uᵀx` rounds `d` terms per coordinate, the plane map at most three (the
    /// chord's scale included), `U z` `2k`, and the final add one, so each stage
    /// sits inside `γ_{d + 2k + 4}` times its absolute magnitude. Frobenius norms
    /// majorize those magnitudes. The five contributions, carried to the output:
    /// - `Uᵀx`, on `‖Uᵀx‖ ≤ ‖U‖_F ‖x‖`, through `|R − I| ≤ 2` and `U`: `2‖U‖_F²‖x‖`;
    /// - the plane map, `‖|R − I|‖_F ≤ 4` on `‖z‖ ≤ ‖U‖_F ‖x‖`, through `U`: `4‖U‖_F²‖x‖`;
    /// - `U z'` on `‖z'‖ ≤ 2‖U‖_F ‖x‖`: `2‖U‖_F²‖x‖`;
    /// - the final add, on `‖x‖ + ‖U z'‖`: `(1 + 2‖U‖_F²)‖x‖`;
    /// - the rounding of `cos` and `sin`, within `u` per entry, through `U` on both
    ///   sides: `‖U‖_F²‖x‖`.
    ///
    /// Summed: `γ·(1 + (2 + 4 + 2 + 2 + 1)‖U‖_F²)‖x‖ = γ·(1 + 11‖U‖_F²)‖x‖`. On the chord,
    /// every stage after `Uᵀx` scales by `|t|`, so `lever = max(1, |t|)` multiplies the 11.
    /// This is rounding only. The basis defect enters through [`plane_vector_defect`]
    /// or [`loop_bar`], wherever an orthonormal-basis map is the reference.
    fn rotation_apply_band(rotation: &PlaneRotation, lever: f64, x_norm: f64) -> f64 {
        let basis_sq = frobenius(&rotation.basis).powi(2);
        let stage = accumulation_growth(rotation.dim() + 2 * rotation.plane_count() + 4);
        stage * (1.0 + 11.0 * lever * basis_sq) * x_norm
    }

    /// How far the basis defect `δ = ‖UᵀU − I‖_F` moves an executed edit of a plane
    /// vector `x = U d`, `‖d‖ = 1`, from the edit on an orthonormal basis.
    ///
    /// With `G = UᵀU`, the executed `x + t U (R − I) Uᵀ x` is
    /// `U M d + t U (R − I)(G − I) d`. Here `M` is the map on an orthonormal basis:
    /// `I + t(R − I)` on the chord, `R(tα)` on the angle path. The second term is at
    /// most `‖U‖₂ · 2 lever · δ`, with `‖R − I‖₂ ≤ 2` and `‖U‖₂² = ‖G‖₂ ≤ 1 + δ`.
    /// Norms read through `U` move as well: `‖U w‖² − ‖w‖² = wᵀ(G − I)w`, so
    /// `|‖U w‖ − ‖w‖| ≤ δ‖w‖`, and a norm comparison at radius `r` pays that twice,
    /// `2rδ`, at the call site.
    fn plane_vector_defect(rotation: &PlaneRotation, lever: f64) -> f64 {
        let delta = rotation.orthonormality_defect();
        2.0 * lever * (1.0 + delta).sqrt() * delta
    }

    /// A2 (P1) over one `GL(4)` block: after `U → U S`, `V → V S⁻ᵀ` a block-scalar
    /// mask and a carried operator mask give the same intervention, while a
    /// diagonal mask held fixed moves it. The fixed diagonal mask is the positive
    /// control the invariance bar must fail on.
    #[test]
    fn scalar_and_carried_masks_survive_a_gauge_change_and_a_fixed_diagonal_mask_moves() {
        let mut rng = StdRng::seed_from_u64(295_101);
        let (d_out, d_in, r) = (7, 6, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, d_out, r),
            uniform_matrix(&mut rng, d_in, r),
            gauge,
        )
        .expect("random factors have full column rank");
        let gauge_change = uniform_matrix(&mut rng, r, r);
        let x = uniform_vector(&mut rng, d_in);

        let scalar = InternalMask::BlockScalar(vec![0.37]);
        let carried_operator = InternalMask::Operator(uniform_matrix(&mut rng, r, r));
        for mask in [&scalar, &carried_operator] {
            let before = operator.apply(mask, x.view()).expect("apply");
            let (regauged, carried) = operator.regauge(gauge_change.view(), mask).expect("regauge");
            let after = regauged.apply(&carried, x.view()).expect("apply regauged");
            let moved = norm(&(&after - &before));
            let band = regauge_band(&operator, &gauge_change, mask, &x);
            assert!(
                moved <= band,
                "a covariant mask moved the intervention by {moved:e} > band {band:e}: {mask:?}"
            );
        }

        let diagonal = InternalMask::Operator(Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5])));
        let before = operator.apply(&diagonal, x.view()).expect("apply");
        let (regauged, carried) = operator.regauge(gauge_change.view(), &diagonal).expect("regauge");
        let band = regauge_band(&operator, &gauge_change, &diagonal, &x);
        let carried_moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
        let fixed_moved = norm(&(&regauged.apply(&diagonal, x.view()).expect("apply") - &before));
        assert!(
            carried_moved <= band,
            "the carried diagonal mask moved the intervention by {carried_moved:e} > band {band:e}"
        );
        assert!(
            fixed_moved > band,
            "positive control: a diagonal mask held fixed under S must move the intervention, moved {fixed_moved:e} <= band {band:e}"
        );
    }

    /// P1's counterexamples are exact: the witness of a non-scalar mask is a
    /// reflection or a transposition, so `S M Sᵀ` only negates or permutes entries,
    /// and the fixed mask moves a full-rank intervention past its rounding band.
    #[test]
    fn a_non_scalar_mask_has_an_exact_gauge_counterexample() {
        let mut rng = StdRng::seed_from_u64(295_102);
        let (dim, r) = (6, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        assert_eq!(
            classify_internal_mask((Array2::<f64>::eye(r) * 0.6).view(), &gauge).expect("classify"),
            MaskGaugeVerdict::Intrinsic { controls: vec![0.6] }
        );

        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, r),
            uniform_matrix(&mut rng, dim, r),
            gauge.clone(),
        )
        .expect("random factors have full column rank");
        let x = uniform_vector(&mut rng, dim);
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let mut off_diagonal = Array2::<f64>::eye(r);
        off_diagonal[[0, 2]] = 0.3;
        for (mask, expected_moved, expected_shift) in [(&diagonal, (0, 0), -1.0), (&off_diagonal, (0, 2), -0.6)] {
            let verdict = classify_internal_mask(mask.view(), &gauge).expect("classify");
            assert!(
                matches!(verdict, MaskGaugeVerdict::BasisDependent { .. }),
                "a non-scalar mask on one GL(4) block was classified intrinsic: {verdict:?}"
            );
            if let MaskGaugeVerdict::BasisDependent {
                gauge_change,
                moved,
                shift,
            } = verdict
            {
                assert_eq!(moved, expected_moved);
                assert_eq!(shift, expected_shift);
                let conjugated = gauge_change.dot(mask).dot(&gauge_change.t());
                assert_eq!(conjugated[moved] - mask[moved], shift);

                let fixed = InternalMask::Operator(mask.clone());
                let before = operator.apply(&fixed, x.view()).expect("apply");
                let (regauged, carried) = operator.regauge(gauge_change.view(), &fixed).expect("regauge");
                let band = regauge_band(&operator, &gauge_change, &fixed, &x);
                let carried_moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
                let fixed_moved = norm(&(&regauged.apply(&fixed, x.view()).expect("apply") - &before));
                assert!(carried_moved <= band, "carried mask moved {carried_moved:e} > band {band:e}");
                assert!(
                    fixed_moved > band,
                    "the witness gauge change must move the fixed mask's intervention, moved {fixed_moved:e} <= band {band:e}"
                );
            }
        }
    }

    /// Declaring the four components as separate scale gauges makes the same
    /// diagonal mask intrinsic, a diagonal gauge change leaves its intervention in
    /// place, and a gauge change that mixes the components is refused as outside
    /// the declared group.
    #[test]
    fn diagonal_masks_are_intrinsic_under_the_per_component_scale_gauge() {
        let mut rng = StdRng::seed_from_u64(295_103);
        let (dim, r) = (6, 4);
        let per_component = GaugeBlocks::new(&[1, 1, 1, 1]).expect("four scale blocks");
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let controls = vec![1.0, 0.0, 1.0, 0.5];
        assert_eq!(
            classify_internal_mask(diagonal.view(), &per_component).expect("classify"),
            MaskGaugeVerdict::Intrinsic {
                controls: controls.clone()
            }
        );
        assert!(matches!(
            classify_internal_mask(diagonal.view(), &GaugeBlocks::new(&[r]).expect("one block")),
            Ok(MaskGaugeVerdict::BasisDependent { .. })
        ));

        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, r),
            uniform_matrix(&mut rng, dim, r),
            per_component,
        )
        .expect("random factors have full column rank");
        let x = uniform_vector(&mut rng, dim);
        let mask = InternalMask::BlockScalar(controls);
        let scale_change = Array2::from_diag(&uniform_vector(&mut rng, r));
        let before = operator.apply(&mask, x.view()).expect("apply");
        let (regauged, carried) = operator.regauge(scale_change.view(), &mask).expect("regauge");
        let moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
        let band = regauge_band(&operator, &scale_change, &mask, &x);
        assert!(
            moved <= band,
            "a per-component scale change moved the intervention by {moved:e} > band {band:e}"
        );

        let mixing_change = uniform_matrix(&mut rng, r, r);
        assert!(matches!(
            operator.regauge(mixing_change.view(), &mask),
            Err(OperatorRefusal::GaugeOutsideDeclaredGroup { .. })
        ));
    }

    /// A factor with an exactly dependent column is refused, and the same factor
    /// with the column restored is accepted.
    #[test]
    fn a_rank_deficient_factor_is_refused() {
        let mut rng = StdRng::seed_from_u64(295_104);
        let (dim, r) = (7, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        let independent = uniform_matrix(&mut rng, dim, r);
        let v = uniform_matrix(&mut rng, dim, r);
        let mut dependent = independent.clone();
        let doubled = &independent.column(0) * 2.0;
        dependent.column_mut(3).assign(&doubled);
        assert!(matches!(
            FactoredOperator::new(dependent, v.clone(), gauge.clone()),
            Err(OperatorRefusal::RankDeficientFactor {
                factor: "left factor",
                declared: 4,
                ..
            })
        ));
        assert!(FactoredOperator::new(independent, v, gauge).is_ok());
    }

    /// P2: along the angle path a plane vector turns by `tα` at unit norm; along the
    /// chord its midpoint is `cos(α/2)` times the bisector, `‖L(t) x‖` matches
    /// `chord_radii(t)` for `t` inside and outside `[0, 1]`, and at `t = ½` the
    /// radius is `|cos(α/2)|`. The chord's norm loss is the positive control the
    /// angle path's norm bar must fail on.
    #[test]
    fn the_chord_midpoint_contracts_by_cos_half_angle_while_the_angle_path_turns() {
        let mut rng = StdRng::seed_from_u64(295_105);
        let dim = 8;
        let basis = orthonormal_columns(&mut rng, dim, 4);
        let angles = Array1::from(vec![1.1, 2.6]);
        let rotation = PlaneRotation::new(basis.clone(), angles.clone()).expect("rotation");
        let phi = 0.4;
        let midpoint_radii = rotation.chord_radii(0.5);
        for plane in 0..rotation.plane_count() {
            let alpha = angles[plane];
            let x = plane_vector(&basis, plane, phi);
            let x_norm = norm(&x);
            let executed_band =
                rotation_apply_band(&rotation, 1.0, x_norm) + plane_vector_band(phi) + plane_vector_defect(&rotation, 1.0);

            for t in [0.25, 0.5, 1.0] {
                let turned = rotation.apply(RotationPath::Angle(t), x.view()).expect("angle path");
                let target = plane_vector(&basis, plane, phi + t * alpha);
                let band = executed_band + plane_vector_band(phi + t * alpha);
                let gap = norm(&(&turned - &target));
                assert!(gap <= band, "angle path at t={t} missed D(φ + tα) by {gap:e} > {band:e}");
            }

            let chord = rotation.apply(RotationPath::Chord(0.5), x.view()).expect("chord path");
            let bisector = plane_vector(&basis, plane, phi + 0.5 * alpha) * (0.5 * alpha).cos();
            let band = executed_band + plane_vector_band(phi + 0.5 * alpha) + accumulation_growth(2);
            let gap = norm(&(&chord - &bisector));
            assert!(gap <= band, "chord midpoint missed cos(α/2)·D(φ + α/2) by {gap:e} > {band:e}");

            let half_cos = (0.5 * alpha).cos().abs();
            assert!(
                (midpoint_radii[plane] - half_cos).abs() <= accumulation_growth(3) * half_cos,
                "chord radius at t=½ is {} but |cos(α/2)| = {half_cos}",
                midpoint_radii[plane]
            );

            let turned = rotation.apply(RotationPath::Angle(0.5), x.view()).expect("angle path");
            // The angle path's radius is 1, so reading the two norms through `U` costs `2δ`.
            let norm_band = executed_band + accumulation_growth(dim + 1) * x_norm + 2.0 * rotation.orthonormality_defect();
            let angle_norm_gap = (norm(&turned) - x_norm).abs();
            let chord_norm_gap = (norm(&chord) - x_norm).abs();
            assert!(
                angle_norm_gap <= norm_band,
                "the angle path changed the norm by {angle_norm_gap:e} > {norm_band:e}"
            );
            assert!(
                chord_norm_gap > norm_band,
                "positive control: the chord midpoint must contract the norm, gap {chord_norm_gap:e} <= {norm_band:e}"
            );

            for t in [0.3_f64, 1.7, -0.4] {
                let executed = norm(&rotation.apply(RotationPath::Chord(t), x.view()).expect("chord path"));
                let predicted = rotation.chord_radii(t)[plane] * x_norm;
                let lever = 1.0_f64.max(t.abs());
                let radius = rotation.chord_radii(t)[plane];
                let band = rotation_apply_band(&rotation, lever, x_norm)
                    + plane_vector_band(phi)
                    + plane_vector_defect(&rotation, lever)
                    + 2.0 * radius * rotation.orthonormality_defect()
                    + accumulation_growth(2 * dim + 8) * (predicted + executed);
                assert!(
                    (executed - predicted).abs() <= band,
                    "chord radius at t={t}: executed {executed} vs predicted {predicted}"
                );
            }
        }
    }

    /// A4 (P4), the cross term: Compose equals the executed composition and Sum
    /// misses it by the induced cross term (the positive control), for two rotation
    /// edits read along different paths and for a factored edit composed onto a
    /// rotation. The order-swap gap is the commutator, which is resolved for
    /// overlapping planes and inside the rounding band for disjoint ones.
    #[test]
    fn compose_carries_the_induced_cross_term_and_the_order_swap_gap_is_the_commutator() {
        let mut rng = StdRng::seed_from_u64(295_106);
        let dim = 6;
        let q = orthonormal_columns(&mut rng, dim, 4);
        let a = PlaneRotation::new(q.slice(s![.., 0..2]).to_owned(), Array1::from(vec![0.9])).expect("a");
        let b = PlaneRotation::new(q.slice(s![.., 1..3]).to_owned(), Array1::from(vec![1.3])).expect("b");
        let disjoint = PlaneRotation::new(q.slice(s![.., 2..4]).to_owned(), Array1::from(vec![-0.7])).expect("c");
        let x = uniform_vector(&mut rng, dim);
        let x_norm = norm(&x);

        let edit_a = a.edit(RotationPath::Angle(1.0));
        let edit_b = b.edit(RotationPath::Chord(0.6));
        let rotation_band = rotation_apply_band(&a, 1.0, x_norm).max(rotation_apply_band(&b, 1.0, x_norm));
        // Two executed edits against three term evaluations and two sums: every
        // stage stays inside a per-edit band on inputs of norm at most 2‖x‖, so six
        // per-edit bands majorize the comparison.
        let band = 6.0 * rotation_band;

        let executed_b = b.apply(RotationPath::Chord(0.6), x.view()).expect("b");
        let executed = a.apply(RotationPath::Angle(1.0), executed_b.view()).expect("a");
        let terms = compose_terms(&edit_a, &edit_b, x.view()).expect("compose terms");
        let compose_gap = norm(&(&terms.compose(x.view()) - &executed));
        let sum_gap = norm(&(&terms.sum(x.view()) - &executed));
        assert!(
            compose_gap <= band,
            "Compose missed the executed composition by {compose_gap:e} > {band:e}"
        );
        assert!(
            sum_gap > band,
            "positive control: Sum must miss the executed composition by the cross term, gap {sum_gap:e} <= {band:e}"
        );
        assert!((sum_gap - norm(&terms.cross)).abs() <= band);

        let executed_a = a.apply(RotationPath::Angle(1.0), x.view()).expect("a");
        let swapped = b.apply(RotationPath::Chord(0.6), executed_a.view()).expect("b");
        let commutator = commutator_apply(&edit_a, &edit_b, x.view()).expect("commutator");
        let swap_gap = norm(&(&(&executed - &swapped) - &commutator));
        assert!(
            swap_gap <= 2.0 * band,
            "the order-swap gap missed the commutator by {swap_gap:e} > {:e}",
            2.0 * band
        );
        assert!(
            norm(&commutator) > 2.0 * band,
            "positive control: edits in overlapping planes must not commute"
        );

        let edit_disjoint = disjoint.edit(RotationPath::Angle(1.0));
        let commuting = commutator_apply(&edit_a, &edit_disjoint, x.view()).expect("commutator");
        // `Δ_A Δ_C = U_A (R_A − I) U_Aᵀ U_C (R_C − I) U_Cᵀ`, so the commutator is at most
        // `2 ‖U_A‖₂ · 2 · ‖U_Aᵀ U_C‖₂ · 2 · ‖U_C‖₂ ‖x‖`. The measured cross block of the
        // two bases decides how far disjoint planes commute; neither basis's own
        // defect does.
        let cross_block = frobenius(&fast_atb(&a.basis, &disjoint.basis));
        let disjoint_band =
            2.0 * band + 8.0 * frobenius(&a.basis) * frobenius(&disjoint.basis) * cross_block * x_norm;
        assert!(
            norm(&commuting) <= disjoint_band,
            "edits in disjoint planes must commute, commutator {:e} > {disjoint_band:e}",
            norm(&commuting)
        );

        let gauge = GaugeBlocks::new(&[2]).expect("one block");
        let factored = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, 2),
            uniform_matrix(&mut rng, dim, 2),
            gauge,
        )
        .expect("random factors have full column rank");
        let factored_mask = InternalMask::Operator(uniform_matrix(&mut rng, 2, 2));
        let factored_edit = factored.residual_edit(&factored_mask).expect("square operator");
        let factored_executed = &executed_b + &factored.apply(&factored_mask, executed_b.view()).expect("factored");
        let factored_terms = compose_terms(&factored_edit, &edit_b, x.view()).expect("terms");
        let factored_norm = frobenius(&factored.u) * mask_norm(&factored_mask, factored.gauge()) * frobenius(&factored.v);
        // The factored edit's stages round at most `dim` terms on magnitudes the
        // product of its Frobenius norms majorizes, on inputs of norm at most 2‖x‖.
        let factored_band = band * (1.0 + factored_norm);
        let factored_gap = norm(&(&factored_terms.compose(x.view()) - &factored_executed));
        assert!(
            factored_gap <= factored_band,
            "Compose with a factored edit missed by {factored_gap:e} > {factored_band:e}"
        );
        assert!(
            norm(&factored_terms.cross) > factored_band,
            "positive control: the factored edit's cross term must be resolved"
        );
    }

    /// `e^{sA} e^{tB} e^{−sA} e^{−tB} x`, executed along angle paths.
    fn edit_loop(a: &PlaneRotation, b: &PlaneRotation, s: f64, t: f64, x: &Array1<f64>) -> Array1<f64> {
        let step1 = b.apply(RotationPath::Angle(-t), x.view()).expect("e^{-tB}");
        let step2 = a.apply(RotationPath::Angle(-s), step1.view()).expect("e^{-sA}");
        let step3 = b.apply(RotationPath::Angle(t), step2.view()).expect("e^{tB}");
        a.apply(RotationPath::Angle(s), step3.view()).expect("e^{sA}")
    }

    /// The loop bar `3(s²|t| ‖A‖²‖B‖ + |s|t² ‖A‖‖B‖²)‖x‖`, valid ONLY for skew `A` and
    /// `B`, whose exponentials are orthogonal.
    ///
    /// With `P = e^{sA}` and `Q = e^{tB}`, `F − I = [P, Q] P⁻¹ Q⁻¹`, so
    /// `F − I − st[A, B] = [P, Q](P⁻¹Q⁻¹ − I) + [P − I − sA, Q − I] + [sA, Q − I − tB]`.
    /// For a skew generator `‖e^{sA} − I‖ ≤ |s|‖A‖` and `‖e^{sA} − I − sA‖ ≤ s²‖A‖²/2`.
    /// So `‖[P, Q]‖ = ‖[P − I, Q − I]‖ ≤ 2|st|‖A‖‖B‖` and `‖P⁻¹Q⁻¹ − I‖ ≤ |s|‖A‖ + |t|‖B‖`,
    /// and the last two commutators are at most `s²|t|‖A‖²‖B‖` and `|s|t²‖A‖‖B‖²`: three
    /// of each monomial in total (mpd-verify's route, #2951). A generator that is not
    /// skew breaks the first two inequalities; see
    /// `the_loop_bar_fails_for_non_skew_generators`.
    fn skew_loop_bar(s: f64, t: f64, norm_a: f64, norm_b: f64, x_norm: f64) -> f64 {
        3.0 * (s * s * t.abs() * norm_a * norm_a * norm_b + s.abs() * t * t * norm_a * norm_b * norm_b) * x_norm
    }

    /// The derived bar of the executed loop against `x + st[A, B] x`, where
    /// `A = U_A Ω_A U_Aᵀ` and each factor executes as `I + U (R(sα) − I) Uᵀ`.
    ///
    /// Let `δ = ‖UᵀU − I‖_F`, `U = Ũ H` with `H = (UᵀU)^{1/2}`, so `‖H − I‖₂ ≤ δ`. The
    /// bar has three terms.
    /// - **Skew bar.** [`skew_loop_bar`] holds for the skew `A`, using
    ///   `‖A‖₂ ≤ ‖U‖₂² ‖Ω‖ ≤ (1 + δ) max_j |α_j|`.
    /// - **Executed factors versus exponentials.** An executed factor is within
    ///   `‖H (R − I) H − (R − I)‖ ≤ ‖R − I‖ δ(2 + δ) ≤ |s| ‖Ω‖ δ(2 + δ)` of the orthogonal
    ///   `I + Ũ (R(sα) − I) Ũᵀ = e^{s Ũ Ω Ũᵀ}`.
    ///   - That is within `|s| ‖Ω − HΩH‖ ≤ |s| ‖Ω‖ δ(2 + δ)` of `e^{sA}`, because both
    ///     generators are skew.
    ///   - So each factor is within `ε = 2|s| ‖Ω‖ δ(2 + δ)` of its exponential.
    ///   - Telescoping the four factors against orthogonal exponentials gives
    ///     `2(ε_A + ε_B)(1 + max ε)³ ‖x‖`. The metric defect enters through `ε`, not with
    ///     coefficient 1 (mpd-verify's question, #2951).
    /// - **Rounding.** Six per-edit bands majorize the four executed edits, the
    ///   commutator's four generator applications scaled by `st ≤ 1`, and the differences.
    fn loop_bar(a: &PlaneRotation, b: &PlaneRotation, s: f64, t: f64, x_norm: f64) -> f64 {
        let omega_a = a.angles.iter().fold(0.0_f64, |largest, angle| largest.max(angle.abs()));
        let omega_b = b.angles.iter().fold(0.0_f64, |largest, angle| largest.max(angle.abs()));
        let (delta_a, delta_b) = (a.orthonormality_defect(), b.orthonormality_defect());
        let eps_a = 2.0 * s.abs() * omega_a * delta_a * (2.0 + delta_a);
        let eps_b = 2.0 * t.abs() * omega_b * delta_b * (2.0 + delta_b);
        let executed = 2.0 * (eps_a + eps_b) * (1.0 + eps_a.max(eps_b)).powi(3) * x_norm;
        let rounding = 6.0 * rotation_apply_band(a, 1.0, x_norm).max(rotation_apply_band(b, 1.0, x_norm));
        skew_loop_bar(s, t, omega_a * (1.0 + delta_a), omega_b * (1.0 + delta_b), x_norm) + executed + rounding
    }

    /// A4 (P4), the loop: for the generators of two plane rotations,
    /// `e^{sA} e^{tB} e^{−sA} e^{−tB} x − x − st[A, B] x` is inside [`loop_bar`].
    /// `apply_generator` builds `A = U Ω Uᵀ` with `Ω` skew, so `A` is skew for every basis
    /// and [`skew_loop_bar`] applies; the basis defect enters through `loop_bar`'s
    /// executed-factor term. Dropping the commutator term is the positive control: at the smallest
    /// step the second-order term of a vector the commutator moves is outside the
    /// third-order bar.
    #[test]
    fn the_edit_loop_remainder_is_inside_its_third_order_bar() {
        let mut rng = StdRng::seed_from_u64(295_107);
        let dim = 5;
        let q = orthonormal_columns(&mut rng, dim, 3);
        let a = PlaneRotation::new(q.slice(s![.., 0..2]).to_owned(), Array1::from(vec![1.0])).expect("a");
        let b = PlaneRotation::new(q.slice(s![.., 1..3]).to_owned(), Array1::from(vec![-0.8])).expect("b");
        let steps = [0.2, 0.1, 0.05, 0.025, 0.0125];
        let random_x = uniform_vector(&mut rng, dim);
        // [A, B] = αβ (u₀u₂ᵀ − u₂u₀ᵀ), so u₂ is moved by the full |αβ|.
        let aligned_x = q.column(2).to_owned();

        for x in [&random_x, &aligned_x] {
            let x_norm = norm(x);
            let bracket = &a.apply_generator(b.apply_generator(x.view()).expect("Bx").view()).expect("ABx")
                - &b.apply_generator(a.apply_generator(x.view()).expect("Ax").view()).expect("BAx");
            for h in steps {
                let (s, t) = (h, 0.6 * h);
                let looped = edit_loop(&a, &b, s, t, x);
                let remainder = norm(&(&(&looped - x) - &(&bracket * (s * t))));
                let bar = loop_bar(&a, &b, s, t, x_norm);
                assert!(
                    remainder <= bar,
                    "loop remainder at h={h}: {remainder:e} > third-order bar {bar:e}"
                );
            }
        }

        let x_norm = norm(&aligned_x);
        let (s, t) = (steps[steps.len() - 1], 0.6 * steps[steps.len() - 1]);
        let bar = loop_bar(&a, &b, s, t, x_norm);
        let without_commutator = norm(&(&edit_loop(&a, &b, s, t, &aligned_x) - &aligned_x));
        assert!(
            without_commutator > bar,
            "positive control: without the commutator term the loop must leave the third-order bar, {without_commutator:e} <= {bar:e}"
        );
    }

    /// The skew premise of [`skew_loop_bar`] is load-bearing (mpd-verify's
    /// counterexample, #2951). Take `A = diag(λ, 0)` and `B = e₁e₂ᵀ`.
    /// - `e^{sA} = diag(e^{sλ}, 1)`, and `e^{tB} = I + tB` because `B² = 0`.
    /// - So `e^{sA} B e^{−sA} = e^{sλ} B`, and the loop is `I + t(e^{sλ} − 1) B`.
    /// - `[A, B] = λB`, so the remainder is `t(e^{sλ} − 1 − sλ) B`.
    ///
    /// At `sλ = 5` the remainder is about `142 t`. That is outside both the constant-4
    /// bar that gate 1175070 tested and [`skew_loop_bar`]: a bound derived for
    /// orthogonal factors does not hold for these generators. The exponentials are the
    /// closed forms of a diagonal and a nilpotent matrix, written out in the fixture.
    #[test]
    fn the_loop_bar_fails_for_non_skew_generators() {
        let (lambda, s, t) = (1.0_f64, 5.0_f64, 1e-3_f64);
        let (norm_a, norm_b) = (lambda.abs(), 1.0_f64);
        let mut a = Array2::<f64>::zeros((2, 2));
        a[[0, 0]] = lambda;
        let mut b = Array2::<f64>::zeros((2, 2));
        b[[0, 1]] = 1.0;
        let exp_a = |scale: f64| Array2::from_diag(&Array1::from(vec![(scale * lambda).exp(), 1.0]));
        let exp_b = |scale: f64| Array2::<f64>::eye(2) + &b * scale;
        let looped = exp_a(s).dot(&exp_b(t)).dot(&exp_a(-s)).dot(&exp_b(-t));
        let bracket = a.dot(&b) - b.dot(&a);
        let x = Array1::from(vec![0.0, 1.0]);
        let x_norm = norm(&x);
        let remainder = norm(&(&(&looped.dot(&x) - &x) - &(bracket.dot(&x) * (s * t))));
        let predicted = t * ((s * lambda).exp() - 1.0 - s * lambda);
        // Rounding: seven 2×2 products and three differences, each at most two
        // rounded terms per entry. The factors' absolute magnitudes are majorized by
        // `e^{|sλ|}(1 + t)²`.
        let rounding = accumulation_growth(32) * (s * lambda).abs().exp() * (1.0 + t).powi(2) * x_norm;
        assert!(
            (remainder - predicted).abs() <= rounding + accumulation_growth(4) * predicted,
            "executed non-skew remainder {remainder:e} vs closed form t(e^(sλ) − 1 − sλ) = {predicted:e}"
        );
        let constant_four_bar = 4.0 * (s * s * t * norm_a * norm_a * norm_b + s * t * t * norm_a * norm_b * norm_b) * x_norm;
        let skew_bar = skew_loop_bar(s, t, norm_a, norm_b, x_norm);
        assert!(
            remainder > constant_four_bar + rounding,
            "positive control: non-skew generators must leave the constant-4 bar, {remainder:e} <= {constant_four_bar:e}"
        );
        assert!(
            remainder > skew_bar + rounding,
            "positive control: non-skew generators must leave the skew bar, {remainder:e} <= {skew_bar:e}"
        );
    }

    /// Rounding band, in Frobenius norm, of `gam_geometry::manifold::matrix_exp(a)` for an
    /// exactly skew `n × n` input `a`, following the owner's algorithm.
    ///
    /// The owner halves `s` times until `θ = ‖a‖_F / 2^s ≤ 1/4`, sums the degree-12 Taylor
    /// series and squares `s` times. The band is assembled phase by phase.
    /// - **Taylor phase.** `term_k = term_{k−1} · a_s / k` rounds `n` products and one
    ///   division per entry, and the sum adds one more rounding. With
    ///   `‖|X||Y|‖_F ≤ ‖X‖_F ‖Y‖_F`, the error `τ_k` of `term_k` obeys
    ///   `τ_k ≤ θ(τ_{k−1} + γ_{n+1}(‖term_{k−1}‖_F + τ_{k−1}))/k`, with
    ///   `‖term_k‖_F ≤ √n θ^k/k!`. Each addition adds `γ_1(√n e^θ + ‖term_k‖_F)`.
    /// - **Truncation.** The owner's tail bound `θ^{13}/13!/(1 − θ)` per unit norm, times `√n`.
    /// - **Squaring phase.** The exact factors are orthogonal because `a` is skew, so
    ///   `‖X_j‖₂ = 1`. A computed square `(X + E)(X + E)` therefore carries error at most
    ///   `2e + e² + γ_n(√n + e)²` from an input error `e`.
    ///
    /// Every step is evaluated as its recurrence, not linearized.
    fn matrix_exp_band(a: &Array2<f64>) -> f64 {
        let n = a.nrows();
        let root_n = (n as f64).sqrt();
        let frob = frobenius(a);
        let squarings = if frob > 0.25 { (frob / 0.25).log2().ceil() as i32 } else { 0 };
        let theta = frob / 2.0_f64.powi(squarings);
        let mut term_norm = root_n;
        let mut term_error = 0.0_f64;
        let mut sum_error = 0.0_f64;
        let mut factorial = 1.0_f64;
        for k in 1..=12 {
            let k_f = k as f64;
            term_error = theta * (term_error + accumulation_growth(n + 1) * (term_norm + term_error)) / k_f;
            term_norm = term_norm * theta / k_f;
            sum_error += term_error + accumulation_growth(1) * (root_n * theta.exp() + term_norm);
            factorial *= k_f;
        }
        let tail = root_n * theta.powi(13) / (factorial * 13.0) / (1.0 - theta);
        let mut error = sum_error + tail;
        for _ in 0..squarings {
            error = 2.0 * error + error * error + accumulation_growth(n) * (root_n + error).powi(2);
        }
        error
    }

    /// `U Ω Uᵀ` formed densely for a test, with `Ω = ⊕_j α_j J`, then made exactly skew
    /// by `½(A − Aᵀ)`: the two entries of each pair are exact negatives of each other.
    fn dense_generator(rotation: &PlaneRotation) -> Array2<f64> {
        let internal = rotation.basis.ncols();
        let mut omega = Array2::<f64>::zeros((internal, internal));
        for (j, &alpha) in rotation.angles.iter().enumerate() {
            omega[[2 * j + 1, 2 * j]] = alpha;
            omega[[2 * j, 2 * j + 1]] = -alpha;
        }
        let formed = rotation.basis.dot(&omega).dot(&rotation.basis.t());
        (&formed - &formed.t()) * 0.5
    }

    /// The executed angle-path factor of a plane rotation is the matrix exponential of
    /// its generator: `x + U(R(sα) − I)Uᵀx` equals `matrix_exp(s U Ω Uᵀ) x`.
    ///
    /// The band combines:
    /// - the executed edit's rounding, [`rotation_apply_band`];
    /// - the basis defect `ε = 2|s|‖Ω‖₂ δ(2 + δ)` (derived at [`loop_bar`]);
    /// - the dense generator's formation error `|s|‖δA‖_F` with
    ///   `‖δA‖_F ≤ γ_{4k+4} ‖U‖_F² ‖Ω‖_F`, since both generators are skew and
    ///   `‖e^X − e^Y‖ ≤ ‖X − Y‖`;
    /// - [`matrix_exp_band`];
    /// - the dense matvec `γ_{d+1}(√d + e)‖x‖`.
    ///
    /// Positive control: `matrix_exp(−sA) x` misses the executed factor by
    /// `2|sin(sα_j)|` on each plane.
    #[test]
    fn the_executed_plane_rotation_factor_is_the_matrix_exponential_of_its_generator() {
        let mut rng = StdRng::seed_from_u64(295_108);
        let dim = 6;
        let basis = orthonormal_columns(&mut rng, dim, 4);
        let rotation = PlaneRotation::new(basis, Array1::from(vec![0.9, -2.3])).expect("rotation");
        let generator = dense_generator(&rotation);
        let x = uniform_vector(&mut rng, dim);
        let x_norm = norm(&x);
        let omega_spectral = rotation.angles.iter().fold(0.0_f64, |largest, angle| largest.max(angle.abs()));
        let omega_frobenius = 2.0_f64.sqrt() * norm(&rotation.angles.to_owned());
        let delta = rotation.orthonormality_defect();
        let formation = accumulation_growth(4 * rotation.basis.ncols() + 4) * frobenius(&rotation.basis).powi(2) * omega_frobenius;
        for s in [0.4_f64, 1.0, 1.7] {
            let executed = rotation.apply(RotationPath::Angle(s), x.view()).expect("angle path");
            let scaled = &generator * s;
            let exponential = gam_geometry::manifold::matrix_exp(&scaled).expect("matrix_exp");
            let exp_error = matrix_exp_band(&scaled);
            let band = rotation_apply_band(&rotation, 1.0, x_norm)
                + (2.0 * s.abs() * omega_spectral * delta * (2.0 + delta) + s.abs() * formation + exp_error) * x_norm
                + accumulation_growth(dim + 1) * ((dim as f64).sqrt() + exp_error) * x_norm;
            let gap = norm(&(&executed - &exponential.dot(&x)));
            assert!(gap <= band, "executed factor vs matrix_exp(sA) at s={s}: {gap:e} > {band:e}");
            let reversed = gam_geometry::manifold::matrix_exp(&(&generator * -s)).expect("matrix_exp");
            let reversed_gap = norm(&(&executed - &reversed.dot(&x)));
            assert!(
                reversed_gap > band + matrix_exp_band(&(&generator * -s)) * x_norm,
                "positive control: matrix_exp(−sA) must miss the executed factor at s={s}, {reversed_gap:e}"
            );
        }
    }

    /// A random exactly skew `n × n` matrix.
    fn random_skew(rng: &mut StdRng, n: usize) -> Array2<f64> {
        let raw = uniform_matrix(rng, n, n);
        (&raw - &raw.t()) * 0.5
    }

    /// An upper bound on `‖a‖₂`: the computed largest singular value plus its SVD
    /// rounding band ([`factor_singular_band`]).
    fn spectral_norm_bound(a: &Array2<f64>) -> f64 {
        let sigma = a.svd(false, false).expect("svd").1;
        let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
        sigma_max + factor_singular_band(a.nrows(), a.ncols(), sigma_max)
    }

    /// A4 (P4) beyond plane generators: for random dense skew `A`, `B` executed through
    /// `matrix_exp`, the loop remainder stays inside [`skew_loop_bar`] plus the four
    /// factors' exponential bands, telescoped against orthogonal exponentials as
    /// `(Σ e_i)(1 + max e)³‖x‖`. Rounding covers the four dense matvecs,
    /// `4γ_{n+1}(√n + max e)‖x‖`, and the commutator's formation,
    /// `st · 2γ_{2n+1} ‖A‖_F ‖B‖_F ‖x‖`.
    ///
    /// Positive control: `x` is the column that `[A, B]` moves the most. Dropping `st[A, B]`
    /// at the smallest step leaves the bar.
    #[test]
    fn the_skew_loop_bar_holds_for_dense_skew_generators() {
        let mut rng = StdRng::seed_from_u64(295_109);
        let n = 5;
        let a = random_skew(&mut rng, n);
        let b = random_skew(&mut rng, n);
        let (norm_a, norm_b) = (spectral_norm_bound(&a), spectral_norm_bound(&b));
        let bracket = a.dot(&b) - b.dot(&a);
        let column = (0..n)
            .map(|j| (j, norm(&bracket.column(j).to_owned())))
            .fold((0, 0.0_f64), |best, candidate| if candidate.1 > best.1 { candidate } else { best })
            .0;
        let mut aligned_x = Array1::<f64>::zeros(n);
        aligned_x[column] = 1.0;
        let random_x = uniform_vector(&mut rng, n);
        let steps = [0.2, 0.1, 0.05, 0.025, 0.0125];
        let loop_through_exp = |s: f64, t: f64, x: &Array1<f64>| -> (Array1<f64>, f64) {
            let factors = [&b * -t, &a * -s, &b * t, &a * s];
            let mut value = x.clone();
            let mut errors = Vec::with_capacity(4);
            for generator in &factors {
                value = gam_geometry::manifold::matrix_exp(generator).expect("matrix_exp").dot(&value);
                errors.push(matrix_exp_band(generator));
            }
            let largest = errors.iter().fold(0.0_f64, |largest, &e| largest.max(e));
            let total: f64 = errors.iter().sum();
            let executed = total * (1.0 + largest).powi(3) + 4.0 * accumulation_growth(n + 1) * ((n as f64).sqrt() + largest);
            (value, executed)
        };
        for x in [&random_x, &aligned_x] {
            let x_norm = norm(x);
            let bracket_x = bracket.dot(x);
            for h in steps {
                let (s, t) = (h, 0.6 * h);
                let (looped, executed) = loop_through_exp(s, t, x);
                let remainder = norm(&(&(&looped - x) - &(&bracket_x * (s * t))));
                let bar = skew_loop_bar(s, t, norm_a, norm_b, x_norm)
                    + executed * x_norm
                    + s * t * 2.0 * accumulation_growth(2 * n + 1) * frobenius(&a) * frobenius(&b) * x_norm;
                assert!(remainder <= bar, "dense skew loop remainder at h={h}: {remainder:e} > {bar:e}");
            }
        }
        let (s, t) = (steps[steps.len() - 1], 0.6 * steps[steps.len() - 1]);
        let (looped, executed) = loop_through_exp(s, t, &aligned_x);
        let bar = skew_loop_bar(s, t, norm_a, norm_b, 1.0)
            + executed
            + s * t * 2.0 * accumulation_growth(2 * n + 1) * frobenius(&a) * frobenius(&b);
        let without_commutator = norm(&(&looped - &aligned_x));
        assert!(
            without_commutator > bar,
            "positive control: without the commutator term the dense loop must leave the bar, {without_commutator:e} <= {bar:e}"
        );
    }

    /// mpd-spec NOTE 9: a basis further from orthonormal than a stable
    /// orthogonalization leaves one is refused, so a non-orthogonal map is never typed
    /// as an angle path. Positive control: the Householder-QR basis the perturbation
    /// starts from is accepted, with its defect under the floor production derives.
    #[test]
    fn a_non_orthonormal_plane_basis_is_refused() {
        let mut rng = StdRng::seed_from_u64(295_111);
        let basis = orthonormal_columns(&mut rng, 8, 4);
        let angles = Array1::from(vec![0.4, 1.2]);
        let accepted = PlaneRotation::new(basis.clone(), angles.clone()).expect("a Householder-QR basis is accepted");
        assert!(
            accepted.orthonormality_defect() <= accepted.orthonormality_floor(),
            "accepted defect {:e} above its floor {:e}",
            accepted.orthonormality_defect(),
            accepted.orthonormality_floor()
        );
        // A second backward-stable orthogonalization, Gram–Schmidt applied twice, is also
        // accepted: the floor bounds any stable orthogonalization, not one algorithm's output.
        let mut reorthogonalized = uniform_matrix(&mut rng, 8, 4);
        orthogonalize_columns(&mut reorthogonalized);
        orthogonalize_columns(&mut reorthogonalized);
        let gram_schmidt =
            PlaneRotation::new(reorthogonalized, angles.clone()).expect("a twice-orthogonalized basis is accepted");
        assert!(
            gram_schmidt.orthonormality_defect() <= gram_schmidt.orthonormality_floor(),
            "Gram–Schmidt defect {:e} above its floor {:e}",
            gram_schmidt.orthonormality_defect(),
            gram_schmidt.orthonormality_floor()
        );
        let mut stretched = basis;
        stretched.column_mut(1).mapv_inplace(|entry| entry * (1.0 + 1e-6));
        assert!(matches!(
            PlaneRotation::new(stretched, angles),
            Err(OperatorRefusal::NonOrthonormalPlaneBasis { .. })
        ));
    }

    /// The exact inverse of a witness, exact in binary: `Sᵀ` for an orthogonal witness
    /// (permutations, reflections, `J_k`), and the reciprocal diagonal for a scaling by 2.
    fn exact_inverse(s: &Array2<f64>) -> Array2<f64> {
        if s.t().dot(s) == Array2::<f64>::eye(s.nrows()) {
            s.t().to_owned()
        } else {
            s.mapv(|entry| if entry == 0.0 { 0.0 } else { 1.0 / entry })
        }
    }

    /// A witness lies in the declared group (`member`). With the exact inverse, `S M S⁻¹`
    /// moves the reported entry by exactly the reported shift, and that shift is nonzero.
    fn assert_witness(mask: &Array2<f64>, verdict: &MaskGaugeVerdict, member: &dyn Fn(&Array2<f64>) -> bool) {
        assert!(
            matches!(verdict, MaskGaugeVerdict::BasisDependent { .. }),
            "expected a witness, got {verdict:?}"
        );
        if let MaskGaugeVerdict::BasisDependent {
            gauge_change,
            moved,
            shift,
        } = verdict
        {
            assert!(member(gauge_change), "the witness is outside the declared group: {gauge_change:?}");
            let conjugated = gauge_change.dot(mask).dot(&exact_inverse(gauge_change));
            assert_eq!(conjugated[*moved] - mask[*moved], *shift);
            assert!(*shift != 0.0, "a witness must move its entry");
        }
    }

    fn is_permutation(s: &Array2<f64>) -> bool {
        s.iter().all(|&entry| entry == 0.0 || entry == 1.0)
            && s.rows().into_iter().all(|row| row.iter().filter(|&&entry| entry == 1.0).count() == 1)
            && s.columns().into_iter().all(|column| column.iter().filter(|&&entry| entry == 1.0).count() == 1)
    }

    fn is_positive_monomial(s: &Array2<f64>) -> bool {
        s.iter().all(|&entry| entry >= 0.0)
            && s.rows().into_iter().all(|row| row.iter().filter(|&&entry| entry > 0.0).count() == 1)
            && s.columns().into_iter().all(|column| column.iter().filter(|&&entry| entry > 0.0).count() == 1)
    }

    /// Block diagonal over the rotary blocks, and commuting with `J = ⊕_k J_k ⊕ 0`. `J` is
    /// zero on the pass-through, so for a block-diagonal `S`, `S J = J S` holds iff each
    /// group's block commutes with its `J_k`.
    fn is_rotary_commutant(groups: &RotaryGroups, s: &Array2<f64>) -> bool {
        let ids = groups.block_ids();
        let block_diagonal = s.indexed_iter().all(|((u, v), &entry)| ids[u] == ids[v] || entry == 0.0);
        let mut j = Array2::<f64>::zeros((groups.dim(), groups.dim()));
        for group in groups.plane_groups() {
            for &(a, b) in group {
                j[[b, a]] = 1.0;
                j[[a, b]] = -1.0;
            }
        }
        block_diagonal && s.dot(&j) == j.dot(s)
    }

    #[test]
    fn unit_permutation_masks_are_intrinsic_iff_they_span_identity_and_ones() {
        let r = 4;
        let gauge = DeclaredGauge::UnitPermutations { units: r };
        let mut intrinsic = Array2::<f64>::from_elem((r, r), 0.2);
        for i in 0..r {
            intrinsic[[i, i]] = 0.5;
        }
        assert_eq!(
            classify_under(intrinsic.view(), &gauge).expect("classify"),
            MaskGaugeVerdict::Intrinsic {
                controls: vec![0.5, 0.2]
            }
        );
        // Positive control: once per-unit scalings join the group, the same mask is not intrinsic.
        let scaled = classify_under(intrinsic.view(), &DeclaredGauge::ScaledPermutations { units: r }).expect("classify");
        assert!(matches!(scaled, MaskGaugeVerdict::BasisDependent { .. }));
        assert_witness(&intrinsic, &scaled, &is_positive_monomial);

        let mut off_diagonal = intrinsic.clone();
        off_diagonal[[2, 3]] = 0.7;
        let verdict = classify_under(off_diagonal.view(), &gauge).expect("classify");
        assert!(matches!(verdict, MaskGaugeVerdict::BasisDependent { moved: (0, 1), .. }));
        assert_witness(&off_diagonal, &verdict, &is_permutation);

        let mut diagonal = intrinsic.clone();
        diagonal[[1, 1]] = -0.1;
        let verdict = classify_under(diagonal.view(), &gauge).expect("classify");
        assert!(matches!(verdict, MaskGaugeVerdict::BasisDependent { moved: (0, 0), .. }));
        assert_witness(&diagonal, &verdict, &is_permutation);

        assert_eq!(
            classify_under(Array2::from_elem((1, 1), 0.4).view(), &DeclaredGauge::UnitPermutations { units: 1 })
                .expect("classify"),
            MaskGaugeVerdict::Intrinsic { controls: vec![0.4] }
        );
        assert!(matches!(
            classify_under(Array2::<f64>::zeros((0, 0)).view(), &DeclaredGauge::UnitPermutations { units: 0 }),
            Err(OperatorRefusal::EmptyGaugeBlock)
        ));
    }

    #[test]
    fn scaled_permutation_masks_are_intrinsic_only_as_scalars() {
        let r = 4;
        let gauge = DeclaredGauge::ScaledPermutations { units: r };
        assert_eq!(
            classify_under((Array2::<f64>::eye(r) * 0.6).view(), &gauge).expect("classify"),
            MaskGaugeVerdict::Intrinsic { controls: vec![0.6] }
        );
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let verdict = classify_under(diagonal.view(), &gauge).expect("classify");
        assert!(matches!(verdict, MaskGaugeVerdict::BasisDependent { moved: (0, 0), .. }));
        assert_witness(&diagonal, &verdict, &is_positive_monomial);

        let mut off_diagonal = Array2::<f64>::eye(r);
        off_diagonal[[3, 1]] = -0.25;
        let verdict = classify_under(off_diagonal.view(), &gauge).expect("classify");
        assert!(matches!(verdict, MaskGaugeVerdict::BasisDependent { moved: (3, 1), .. }));
        assert_witness(&off_diagonal, &verdict, &is_positive_monomial);

        // Positive control: under the per-component scale gauge alone, with no permutations,
        // the diagonal mask is intrinsic, through `classify_under` and `classify_internal_mask`.
        let per_component = GaugeBlocks::new(&[1, 1, 1, 1]).expect("four scale blocks");
        let expected = MaskGaugeVerdict::Intrinsic {
            controls: vec![1.0, 0.0, 1.0, 0.5],
        };
        assert_eq!(
            classify_under(diagonal.view(), &DeclaredGauge::Blocks(per_component.clone())).expect("classify"),
            expected
        );
        assert_eq!(classify_internal_mask(diagonal.view(), &per_component).expect("classify"), expected);
    }

    /// Groups `[(0, 1), (2, 3)]` and `[(4, 5)]`, pass-through `6..8`, and the intrinsic mask
    /// `(0.7 I − 0.3 J_0) ⊕ (0.2 I + 0.9 J_1) ⊕ 0.4 I`.
    fn rotary_fixture() -> (RotaryGroups, Array2<f64>) {
        let groups = RotaryGroups::new(vec![vec![(0, 1), (2, 3)], vec![(4, 5)]], 6..8).expect("rotary groups");
        let mut mask = Array2::<f64>::zeros((8, 8));
        for (group, (alpha, beta)) in groups.plane_groups().iter().zip([(0.7, -0.3), (0.2, 0.9)]) {
            for &(a, b) in group {
                mask[[a, a]] = alpha;
                mask[[b, b]] = alpha;
                mask[[b, a]] = beta;
                mask[[a, b]] = -beta;
            }
        }
        mask[[6, 6]] = 0.4;
        mask[[7, 7]] = 0.4;
        (groups, mask)
    }

    #[test]
    fn rotary_masks_are_intrinsic_iff_complex_scalars_per_frequency_group() {
        let (groups, intrinsic) = rotary_fixture();
        let gauge = DeclaredGauge::RotaryCommutant(groups.clone());
        assert_eq!(
            classify_under(intrinsic.view(), &gauge).expect("classify"),
            MaskGaugeVerdict::Intrinsic {
                controls: vec![0.7, -0.3, 0.2, 0.9, 0.4]
            }
        );
        let member = |s: &Array2<f64>| is_rotary_commutant(&groups, s);
        // One perturbation per witness path, in the classifier's order:
        // - a cross-block entry;
        // - a block that is not complex-linear;
        // - a complex off-diagonal entry, set on both coordinates so complex-linearity holds;
        // - an unequal complex diagonal on plane (2, 3);
        // - a pass-through off-diagonal entry;
        // - an unequal pass-through diagonal.
        let perturbations: Vec<(Vec<((usize, usize), f64)>, (usize, usize))> = vec![
            (vec![((0, 4), 0.1)], (0, 4)),
            (vec![((1, 1), 0.5)], (0, 0)),
            (vec![((0, 2), 0.3), ((1, 3), 0.3)], (0, 2)),
            (vec![((2, 2), -0.2), ((3, 3), -0.2)], (0, 0)),
            (vec![((6, 7), 0.25)], (6, 7)),
            (vec![((7, 7), -0.4)], (6, 6)),
        ];
        for (entries, expected_moved) in perturbations {
            let mut mask = intrinsic.clone();
            for (index, value) in entries {
                mask[index] = value;
            }
            let verdict = classify_under(mask.view(), &gauge).expect("classify");
            assert!(
                matches!(verdict, MaskGaugeVerdict::BasisDependent { moved, .. } if moved == expected_moved),
                "expected the witness to move {expected_moved:?}, got {verdict:?}"
            );
            assert_witness(&mask, &verdict, &member);
        }
    }

    /// A declared-group witness moves a full-rank intervention. Every witness is
    /// invertible, so it lies in GL(8), which one gauge block declares for `regauge`.
    /// The fixed mask moves the intervention past the rounding band; the carried mask
    /// stays inside it.
    #[test]
    fn a_declared_group_witness_moves_the_fixed_mask_intervention() {
        let mut rng = StdRng::seed_from_u64(295_110);
        let (groups, intrinsic) = rotary_fixture();
        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, 9, 8),
            uniform_matrix(&mut rng, 10, 8),
            GaugeBlocks::new(&[8]).expect("one block"),
        )
        .expect("random factors have full column rank");
        let x = uniform_vector(&mut rng, 10);
        let mut not_complex_linear = intrinsic.clone();
        not_complex_linear[[1, 1]] = 0.5;
        let verdict = classify_under(not_complex_linear.view(), &DeclaredGauge::RotaryCommutant(groups)).expect("classify");
        assert!(matches!(verdict, MaskGaugeVerdict::BasisDependent { .. }));
        if let MaskGaugeVerdict::BasisDependent { gauge_change, .. } = verdict {
            let fixed = InternalMask::Operator(not_complex_linear);
            let before = operator.apply(&fixed, x.view()).expect("apply");
            let (regauged, carried) = operator.regauge(gauge_change.view(), &fixed).expect("regauge");
            let band = regauge_band(&operator, &gauge_change, &fixed, &x);
            let carried_moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
            let fixed_moved = norm(&(&regauged.apply(&fixed, x.view()).expect("apply") - &before));
            assert!(carried_moved <= band, "carried mask moved {carried_moved:e} > band {band:e}");
            assert!(
                fixed_moved > band,
                "the J witness must move the fixed mask's intervention, moved {fixed_moved:e} <= band {band:e}"
            );
        }
    }

    #[test]
    fn a_rotary_declaration_must_cover_every_coordinate_once() {
        assert!(matches!(
            RotaryGroups::new(vec![vec![(0, 1), (1, 2)]], 4..4),
            Err(OperatorRefusal::InvalidRotaryDeclaration { coordinate: 1 })
        ));
        assert!(matches!(
            RotaryGroups::new(vec![vec![(0, 5)]], 2..3),
            Err(OperatorRefusal::InvalidRotaryDeclaration { coordinate: 5 })
        ));
        assert!(matches!(
            RotaryGroups::new(vec![vec![]], 0..2),
            Err(OperatorRefusal::EmptyGaugeBlock)
        ));
        // Positive control: a declaration covering 0..3 exactly once is accepted.
        let valid = RotaryGroups::new(vec![vec![(0, 1)]], 2..3).expect("valid declaration");
        assert_eq!(valid.dim(), 3);
    }

    /// P1 as evidence: a commutant mask is `Exact` at 0, and any other mask is a
    /// `Counterexample` whose witness moves its entry by exactly the reported value.
    ///
    /// Positive controls:
    /// - the diagonal mask that GL(4) refutes is exact under the per-component scale gauge;
    /// - the numerical error covers a shift that rounds. Under unit permutations the shift of
    ///   `diag(1, x)` with `x = 1e-17` is `fl(x − 1) = −1`, which drops `x`, so a zero error
    ///   would claim an exactness the subtraction does not have;
    /// - a subnormal shift is an exact counterexample, which the rounded-up bound alone would refuse;
    /// - an overflowing shift is refused as non-finite (mpd-verify's review note).
    #[test]
    fn mask_gauge_evidence_is_exact_or_a_counterexample() {
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let full = DeclaredGauge::Blocks(GaugeBlocks::new(&[4]).expect("one block"));
        let per_component = DeclaredGauge::Blocks(GaugeBlocks::new(&[1, 1, 1, 1]).expect("four scale blocks"));
        for (mask, gauge) in [(Array2::<f64>::eye(4) * 0.6, &full), (diagonal.clone(), &per_component)] {
            let status = mask_gauge_evidence(mask.view(), gauge).expect("evidence");
            assert!(
                matches!(
                    &status,
                    EvidenceStatus::Exact {
                        value,
                        numerical_error,
                        basis: ExactBasis::Algebraic,
                        witness: None,
                        domain,
                        ..
                    } if *value == 0.0 && *numerical_error == 0.0 && domain == gauge
                ),
                "expected exact evidence at 0 under {gauge:?}, got {status:?}"
            );
            assert!(status.certifies_at_most(0.0));
        }

        let (groups, rotary) = rotary_fixture();
        let mut cross_block = rotary;
        cross_block[[0, 4]] = 0.1;
        let mut off_diagonal = Array2::<f64>::from_elem((4, 4), 0.2);
        off_diagonal[[2, 3]] = 0.7;
        let cases = [
            (diagonal, full),
            (off_diagonal.clone(), DeclaredGauge::UnitPermutations { units: 4 }),
            (off_diagonal, DeclaredGauge::ScaledPermutations { units: 4 }),
            (cross_block, DeclaredGauge::RotaryCommutant(groups)),
        ];
        for (mask, gauge) in &cases {
            let status = mask_gauge_evidence(mask.view(), gauge).expect("evidence");
            assert!(status.refutes_at_most(0.0) && !status.certifies_at_most(0.0));
            let EvidenceStatus::Counterexample {
                value,
                threshold,
                witness,
                ..
            } = &status
            else {
                panic!("expected a counterexample under {gauge:?}, got {status:?}");
            };
            assert_eq!(*threshold, 0.0);
            let MaskGaugeVerdict::BasisDependent {
                gauge_change,
                moved,
                shift,
            } = classify_under(mask.view(), gauge).expect("classify")
            else {
                panic!("the classifier and the evidence disagree under {gauge:?}");
            };
            assert_eq!((&witness.gauge_change, witness.moved, *value), (&gauge_change, moved, shift.abs()));
            let conjugated = witness.gauge_change.dot(mask).dot(&exact_inverse(&witness.gauge_change));
            assert_eq!((conjugated[witness.moved] - mask[witness.moved]).abs(), *value);
        }

        let x = 1e-17;
        let rounding = Array2::from_diag(&Array1::from(vec![1.0, x]));
        let status = mask_gauge_evidence(rounding.view(), &DeclaredGauge::UnitPermutations { units: 2 }).expect("evidence");
        let EvidenceStatus::Counterexample {
            value,
            numerical_error,
            ..
        } = &status
        else {
            panic!("expected a counterexample, got {status:?}");
        };
        assert_eq!(*value, 1.0);
        assert_eq!((x - 1.0) + 1.0, 0.0, "the control needs a subtraction that drops x");
        assert!(*numerical_error >= x, "the error bound {numerical_error:e} misses the dropped {x:e}");

        // A subnormal shift is exact and carries no error. The rounded-up power-of-two bound
        // alone would leave it inside its own error, and the constructor would refuse it.
        let tiny = f64::from_bits(1);
        let subnormal = Array2::from_diag(&Array1::from(vec![0.0, tiny]));
        let status = mask_gauge_evidence(subnormal.view(), &DeclaredGauge::UnitPermutations { units: 2 }).expect("evidence");
        assert!(
            matches!(
                &status,
                EvidenceStatus::Counterexample {
                    value,
                    numerical_error,
                    ..
                } if *value == tiny && *numerical_error == 0.0
            ),
            "a subnormal shift is an exact counterexample, got {status:?}"
        );
        assert!(EvidenceStatus::<(), ()>::counterexample(tiny, (f64::EPSILON * tiny).next_up(), 0.0, ()).is_err());

        // An overflowing shift is a typed refusal, never a status.
        let mut overflow = Array2::<f64>::zeros((2, 2));
        overflow[[0, 1]] = f64::MAX;
        assert!(matches!(
            mask_gauge_evidence(overflow.view(), &DeclaredGauge::Blocks(GaugeBlocks::new(&[2]).expect("one block"))),
            Err(OperatorRefusal::Evidence(EvidenceStatusError::NonFinite { .. }))
        ));
    }

    /// mpd-verify's review note: the floor's `m·n·ε` term is factor_singular_band's convention,
    /// not a bound derived for this computation. This is evidence at the swept dimensions.
    ///
    /// - Sweep: Householder-QR bases from uniform draws with columns graded from 1 down to `10^e`,
    ///   `e ∈ {0, −6, −12}`, at `d ∈ {8, 64, 512}` and `2k ∈ {2, 8, 32}` with `2k <= d`. Every
    ///   one is accepted, and the largest `defect / floor` ratio is printed.
    /// - Boundary: the refusal sits at the floor. At 512×32, column 0 is stretched by `s`. With
    ///   `UᵀU = I + F` and `D = diag(s, 1, …)`, the stretched Gram deviation is
    ///   `(D² − I) + D F D`, whose true norm lies within `s²‖F‖_F` of `s² − 1`. `‖F‖_F` is at most
    ///   the measured defect plus `γ_d‖U‖_F²`, and the measured stretched defect is within
    ///   `γ_d‖U_s‖_F²` of its true norm.
    ///   - Below, `s² − 1 = floor₀/8`: the derived upper side is under production's floor, and it accepts.
    ///   - Above, `s² − 1 = 2·floor₀`: the derived lower side is over production's floor, and it refuses.
    #[test]
    fn householder_qr_bases_stay_under_the_plane_basis_floor() {
        let mut rng = StdRng::seed_from_u64(295_112);
        let graded_qr_basis = |rng: &mut StdRng, dim: usize, internal: usize, exponent: f64| -> Array2<f64> {
            let mut input = uniform_matrix(rng, dim, internal);
            for (j, mut column) in input.columns_mut().into_iter().enumerate() {
                let scale = 10f64.powf(exponent * j as f64 / (internal - 1) as f64);
                column.mapv_inplace(|entry| entry * scale);
            }
            let q = input.qr().expect("Householder QR of a graded draw").0;
            q.slice(s![.., 0..internal]).to_owned()
        };
        let mut largest_ratio = 0.0_f64;
        for dim in [8, 64, 512] {
            for internal in [2, 8, 32].into_iter().filter(|&internal| internal <= dim) {
                for exponent in [0.0, -6.0, -12.0] {
                    let basis = graded_qr_basis(&mut rng, dim, internal, exponent);
                    let angles = Array1::from(vec![0.9; internal / 2]);
                    match PlaneRotation::new(basis, angles) {
                        Ok(rotation) => {
                            largest_ratio = largest_ratio.max(rotation.orthonormality_defect() / rotation.orthonormality_floor());
                        }
                        Err(refusal) => {
                            panic!("a Householder-QR basis {dim}×{internal} graded to 1e{exponent} was refused: {refusal:?}")
                        }
                    }
                }
            }
        }
        eprintln!("QR_FLOOR_RATIO largest defect/floor over the sweep: {largest_ratio:e}");
        assert!(largest_ratio <= 1.0);

        let (dim, internal) = (512, 32);
        let basis = graded_qr_basis(&mut rng, dim, internal, 0.0);
        let angles = Array1::from(vec![0.9; internal / 2]);
        let unstretched = PlaneRotation::new(basis.clone(), angles.clone()).expect("the QR basis is accepted");
        let (defect_0, floor_0) = (unstretched.orthonormality_defect(), unstretched.orthonormality_floor());
        let gamma = accumulation_growth(dim);
        let formation_0 = gamma * frobenius(&basis).powi(2);
        // s = √(1 + excess); (s − 1)(s + 1) measures s² − 1 including the rounding of s.
        let stretch = |excess: f64| {
            let s = (1.0 + excess).sqrt();
            let mut stretched = basis.clone();
            stretched.column_mut(0).mapv_inplace(|entry| entry * s);
            let s_sq_minus_1 = (s - 1.0) * (s + 1.0);
            let deviation = s * s * (defect_0 + formation_0) + gamma * frobenius(&stretched).powi(2);
            (PlaneRotation::new(stretched, angles.clone()), s_sq_minus_1, deviation)
        };
        let floor_of = |outcome: &Result<PlaneRotation, OperatorRefusal>| match outcome {
            Ok(rotation) => rotation.orthonormality_floor(),
            Err(OperatorRefusal::NonOrthonormalPlaneBasis { floor, .. }) => *floor,
            Err(other) => panic!("unexpected refusal {other:?}"),
        };

        let (below, excess_below, deviation_below) = stretch(floor_0 / 8.0);
        let floor_below = floor_of(&below);
        let upper_side = excess_below + deviation_below;
        assert!(upper_side <= floor_below, "fixture: upper side {upper_side:e} not under the floor {floor_below:e}");
        assert!(below.is_ok(), "a basis derived under the floor was refused: {:?}", below.as_ref().err());

        let (above, excess_above, deviation_above) = stretch(2.0 * floor_0);
        let floor_above = floor_of(&above);
        let lower_side = excess_above - deviation_above;
        assert!(lower_side > floor_above, "fixture: lower side {lower_side:e} not over the floor {floor_above:e}");
        assert!(
            matches!(above, Err(OperatorRefusal::NonOrthonormalPlaneBasis { .. })),
            "a basis derived over the floor was accepted"
        );
    }
}
