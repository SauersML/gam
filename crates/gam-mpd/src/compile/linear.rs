//! Linear-site feasibility: a fixed edit `ΔW` of one stored matrix that makes every declared
//! use of it change as requested, or a witness that none exists.
//!
//! # The problem
//!
//! A stored matrix `W` (`rows × cols`) is read by its use sites. A requirement sets a use's
//! output on declared inputs to declared targets (a set-type atom: it names the value after
//! the edit, never an increment), and the demanded change is the target minus `θ`'s own
//! response, known to the declared radius plus its formation band. A setting whose every
//! demanded change lies within that radius is the native one and compiles to `ρ = θ`.
//! The demanded changes give:
//!
//! * an identity use (`y = W x`) gives **right constraints** `ΔW X_r = Y_r`;
//! * a transposed use (`y = Wᵀ x`, e.g. a tied output head reading an embedding) gives
//!   **left constraints** `X_lᵀ ΔW = Y_lᵀ`;
//! * a stored-row read (an embedding lookup of row `t`) gives the left constraint
//!   `e_tᵀ ΔW = y_tᵀ`.
//!
//! Every use of the storage reads the one edited tensor, so all requirements are solved
//! together: a tie is never untied. Uses the requirements do not name are reached too, and
//! are reported as [`LinearSiteReport::undeclared_uses`].
//!
//! # Feasibility
//!
//! With `A = X_lᵀ`, `C = Y_lᵀ`, `B = X_r`, `D = Y_r`, the system `A ΔW = C`, `ΔW B = D` has a
//! solution iff (Ben-Israel & Greville, Thm 2.13)
//!
//! 1. `ker X_r ⊆ ker Y_r` (`D B⁺ B = D`);
//! 2. `ker X_l ⊆ ker Y_l` (`A A⁺ C = C`);
//! 3. `X_lᵀ Y_r = Y_lᵀ X_r`: both sides fix the bilinear values `x_lᵀ ΔW x_r` and must agree.
//!
//! A failure of 1 or 2 is witnessed by a unit `v` over the constraint columns with
//! `X v ≈ 0` and `Y v ≠ 0`: for every edit, `‖(ΔW X − Y) v‖ ≥ ‖Y v‖ − ‖ΔW‖₂ ‖X v‖`, so no
//! edit of metric norm below `‖Ỹ v‖ / ‖X̃ v‖` (scaled coordinates, below) realizes it, and
//! within the SVD's backward error `X` has `v` in its kernel exactly. A failure of 3 is
//! witnessed by the pair of columns `(i, j)` whose two demands on `x_{l,i}ᵀ ΔW x_{r,j}`
//! differ. Each witness is an [`EvidenceStatus::Counterexample`]: the violation exceeds the
//! evaluation band of its own value, plus the declared target radius.
//!
//! # The minimum-norm edit
//!
//! The metric is `‖D_r ΔW D_c‖_F` with declared positive diagonal scales ([`EditMetric`]);
//! `D = I` is the Frobenius norm of the tensor as given. Frobenius "in canonical gauge" is
//! this metric with the canonical gauge's diagonal elements as scales (a folded norm gain
//! `γ` moves `W ↦ W diag(γ)`, so its Frobenius norm there is `col_scale = γ`), or the
//! Frobenius norm of tensors passed already in canonical gauge. With `E = D_r ΔW D_c`, `X̃_r = D_c⁻¹ X_r`,
//! `Ỹ_r = D_r Y_r`, `X̃_l = D_r⁻¹ X_l`, `Ỹ_l = D_c Y_l`, the constraints keep their form and
//! the minimum-Frobenius `E` is
//!
//! ```text
//! E = A⁺C + (I − A⁺A) D B⁺ = U_l S_l⁻¹ V_lᵀ Ỹ_lᵀ + (I − U_l U_lᵀ) Ỹ_r V_r S_r⁻¹ U_rᵀ,
//! ```
//!
//! with the thin SVDs `X̃_l = U_l S_l V_lᵀ`, `X̃_r = U_r S_r V_rᵀ` over their resolved singular
//! values (above `factor_singular_band`). The general solution adds
//! `(I − A⁺A) Z (I − B B⁺)`, which is Frobenius-orthogonal to both terms, so `E` is the
//! minimum. It is stored as factors `[U_l | (I − U_lU_lᵀ)Ỹ_r V_r S_r⁻¹]` and
//! `[Ỹ_l V_l S_l⁻¹ | U_r]`, unscaled by `D_r⁻¹` and `D_c⁻¹`, and never formed.
//!
//! # Residual and allowance
//!
//! The stored edit's residuals `ΔW X_r − Y_r` and `X_lᵀ ΔW − Y_lᵀ` are evaluated directly
//! from the factors in the original coordinates, with a per-entry band (`γ_n` inner
//! products). The construction is exact in exact arithmetic; its arithmetic makes the
//! stored edit the exact solution of data within backward error of the declared data. The
//! **allowance** bounds that: `‖E‖_F β_X + β_Y`, with `β_X` the SVD's backward band on `X̃`
//! plus the orthonormality defect `ω` of the computed singular bases (`(2 + ω) ω ‖X̃‖₂`, the
//! distance of `V̂V̂ᵀ` from an exact projector), and `β_Y` the formation rounding of the
//! factors against `‖Ỹ‖_F` plus the declared target radius. A residual resolved above
//! `band + allowance` is reported as a failed construction, never as realized.
//!
//! # Coverage
//!
//! A requirement declares its response class ([`ResponseClass`]): only its sampled inputs,
//! every input, or the span of a declared basis. The edit is linear, so it is fixed on
//! `span(X_r)` (resp. `span(X_l)`); a class inside that span is covered and the result holds
//! on the whole class. A class that reaches outside is reported with its uncovered dimension
//! and a unit direction of the class the samples do not reach, and the result is only
//! empirical beyond the sampled inputs.

use gam_linalg::faer_ndarray::{fast_ab, fast_atb};
use gam_linalg::roundoff::{accumulation_growth, basis_orthonormality_defect};
use gam_linalg::utils::frobenius_norm;
use gam_math::roundoff::{UNIT_ROUNDOFF, inflated};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};

use super::super::apply::FactoredEdit;
use super::super::dense::svd;
use super::super::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::ties::{TieConstraint, TieViolation, check_ties};
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, DescriptiveReason, NativeEditPlan,
    require_finite, require_shape,
};
use gam_runtime::resource::MemoryGovernor;

/// Which inputs a requirement's claim is about.
#[derive(Clone, Debug)]
pub enum ResponseClass<'a> {
    /// Only the sampled inputs: the result is empirical beyond them by declaration.
    Sample,
    /// Every input vector of the read width.
    AllInputs,
    /// The span of the rows of a declared basis (`k × read width`).
    Span(ArrayView2<'a, f64>),
}

/// One declared requirement on the edited storage.
#[derive(Clone, Debug)]
pub enum Requirement<'a> {
    /// A linear use: its inputs (`n × read width`, rows are observations) and what it must
    /// write on them after the edit (`n × written width`), a set-type value.
    Linear {
        site: UseSiteId,
        inputs: ArrayView2<'a, f64>,
        targets: ArrayView2<'a, f64>,
        /// Entrywise radius within which the declared targets are known (`0` for exact).
        target_radius: f64,
        class: ResponseClass<'a>,
    },
    /// Stored rows read as values (an embedding lookup), set to `targets`:
    /// `(W + ΔW)[rows[i], :] = targets[i, :]`.
    StoredRows {
        rows: Vec<usize>,
        targets: ArrayView2<'a, f64>,
        target_radius: f64,
    },
}

/// The declared edit metric `‖diag(row_scale) ΔW diag(col_scale)‖_F`; absent scales are 1.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditMetric {
    row_scale: Option<Vec<f64>>,
    col_scale: Option<Vec<f64>>,
}

impl EditMetric {
    /// The Frobenius norm of the tensor as given.
    pub fn frobenius() -> Self {
        Self::default()
    }

    /// Positive finite diagonal scales on the rows and columns.
    pub fn weighted(row_scale: Option<Vec<f64>>, col_scale: Option<Vec<f64>>) -> Result<Self, CompileError> {
        for (what, scale) in [("row scale", &row_scale), ("column scale", &col_scale)] {
            if let Some(values) = scale
                && let Some(bad) = values.iter().find(|value| !(value.is_finite() && **value > 0.0))
            {
                return Err(CompileError::InvalidDeclaration {
                    what,
                    reason: format!("every scale must be positive and finite; found {bad}"),
                });
            }
        }
        Ok(Self { row_scale, col_scale })
    }

    pub fn row_scale(&self) -> Option<&[f64]> {
        self.row_scale.as_deref()
    }

    pub fn col_scale(&self) -> Option<&[f64]> {
        self.col_scale.as_deref()
    }
}

/// A linear-site problem on one storage tensor.
#[derive(Clone, Debug)]
pub struct LinearSiteProblem<'a> {
    pub registry: &'a TensorRegistry,
    pub storage: TensorId,
    /// The stored values `θ` of the edited tensor, against which set-type targets become
    /// changes.
    pub native: ArrayView2<'a, f64>,
    pub requirements: Vec<Requirement<'a>>,
    pub metric: EditMetric,
    /// Declared ties between separately stored blocks; the plan is checked against them.
    pub ties: Vec<TieConstraint>,
    /// Off-target inputs of linear uses, on which the edit's damage is reported.
    pub off_target: Vec<OffTargetInputs<'a>>,
}

/// Inputs of a linear use the edit is not meant to change.
#[derive(Clone, Debug)]
pub struct OffTargetInputs<'a> {
    pub site: UseSiteId,
    /// `n × read width`, rows are observations.
    pub inputs: ArrayView2<'a, f64>,
}

/// Where the largest off-target damage occurs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffTargetWitness {
    /// Index into [`LinearSiteProblem::off_target`] and the observation within it.
    pub set: usize,
    pub observation: usize,
}

/// The family an off-target damage is stated over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffTargetDomain {
    pub observations: usize,
}

/// Which constraint family a column belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintSide {
    /// `ΔW x = y`.
    Right,
    /// `ΔWᵀ x = y`.
    Left,
}

/// Where one constraint column came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnOrigin {
    pub requirement: usize,
    pub observation: usize,
}

/// The finite family a linear-site residual is stated over.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSiteDomain {
    pub right_columns: usize,
    pub left_columns: usize,
    /// Every requirement's declared class lies in the span its side's inputs fix.
    pub classes_covered: bool,
}

/// A witness of a linear-site finding.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearWitness {
    /// A unit `v` over one side's constraint columns with `X v` in the numerical kernel and
    /// `Y v` resolved from zero.
    Kernel {
        side: ConstraintSide,
        direction: Vec<f64>,
        /// An upper bound on `‖X v‖₂` (original coordinates).
        input_norm_upper: f64,
        /// A lower bound on `‖Ỹ v‖₂ / ‖X̃ v‖₂`, which bounds the metric norm of every exact
        /// solution from below; `+∞` when `X v = 0` is certified.
        edit_norm_lower_bound: f64,
    },
    /// Left column `left` and right column `right` demand different `x_lᵀ ΔW x_r`.
    Compatibility { left: usize, right: usize },
    /// The entry of the largest residual.
    Residual { side: ConstraintSide, column: usize, entry: usize },
    /// The plan would change one side of a declared tie and not the other.
    Tie(TieViolation),
}

/// Whether a requirement's class is fixed by the sampled inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum Coverage {
    /// The class is the sample.
    SampleOnly,
    /// Stored rows: the requirement names exactly the entries it fixes.
    StoredRows,
    /// The class lies in the span the side's inputs fix.
    Covered,
    /// `dimension` directions of the class are not fixed; `direction` is one of them, a unit
    /// vector of the read width (original coordinates).
    Uncovered { dimension: usize, direction: Vec<f64> },
}

/// The status type of a linear-site control.
pub type LinearRealization = ControlRealization<LinearWitness, LinearSiteDomain>;

/// What [`compile_linear_site`] found.
#[derive(Clone, Debug)]
pub struct LinearSiteReport {
    pub compiled: CompiledControl<LinearWitness, LinearSiteDomain>,
    /// `‖D_r ΔW D_c‖_F` of the stored edit and its band, when an edit was built.
    pub metric_norm: Option<(f64, f64)>,
    /// Resolved ranks of `X̃_r` and `X̃_l`.
    pub right_rank: usize,
    pub left_rank: usize,
    pub right_origins: Vec<ColumnOrigin>,
    pub left_origins: Vec<ColumnOrigin>,
    /// Per requirement, in order.
    pub coverage: Vec<Coverage>,
    /// Use sites of the storage that no requirement names; the global edit reaches them.
    pub undeclared_uses: Vec<UseSiteId>,
    /// The largest residual entry the construction's arithmetic can leave, per side.
    pub allowance: [f64; 2],
    /// The edit's damage on the declared off-target inputs, when a plan was built and
    /// off-target inputs were declared.
    pub off_target_damage: Option<EvidenceStatus<OffTargetWitness, OffTargetDomain>>,
}

/// Assembled constraint columns of one side, in original coordinates.
struct Side {
    /// `width × n`: inputs as columns.
    x: Array2<f64>,
    /// `other width × n`: requested changes as columns.
    y: Array2<f64>,
    /// Entrywise radius of each column of `y`.
    radius: Vec<f64>,
    origins: Vec<ColumnOrigin>,
}

impl Side {
    fn empty(x_width: usize, y_width: usize) -> Self {
        Self {
            x: Array2::zeros((x_width, 0)),
            y: Array2::zeros((y_width, 0)),
            radius: Vec::new(),
            origins: Vec::new(),
        }
    }

    fn push(
        &mut self,
        requirement: usize,
        inputs: ArrayView2<'_, f64>,
        changes: ArrayView2<'_, f64>,
        radii: &[f64],
    ) -> Result<(), CompileError> {
        let x = concatenate(Axis(1), &[self.x.view(), inputs.t()]).map_err(|error| CompileError::InvalidDeclaration {
            what: "requirement inputs",
            reason: error.to_string(),
        })?;
        let y = concatenate(Axis(1), &[self.y.view(), changes.t()]).map_err(|error| CompileError::InvalidDeclaration {
            what: "requirement changes",
            reason: error.to_string(),
        })?;
        self.x = x;
        self.y = y;
        for (observation, &radius) in radii.iter().enumerate() {
            self.radius.push(radius);
            self.origins.push(ColumnOrigin { requirement, observation });
        }
        Ok(())
    }

    fn columns(&self) -> usize {
        self.x.ncols()
    }
}

/// A resolved thin SVD `M = U S Vᵀ` over the singular values above its backward band.
struct Resolved {
    u: Array2<f64>,
    s: Vec<f64>,
    v: Array2<f64>,
    sigma_max: f64,
    band: f64,
    /// Bound on the orthonormality defect of both computed bases.
    omega: f64,
}

impl Resolved {
    fn of(matrix: ArrayView2<'_, f64>) -> Result<Self, CompileError> {
        let (rows, cols) = matrix.dim();
        if rows == 0 || cols == 0 {
            return Ok(Self {
                u: Array2::zeros((rows, 0)),
                s: Vec::new(),
                v: Array2::zeros((cols, 0)),
                sigma_max: 0.0,
                band: 0.0,
                omega: 0.0,
            });
        }
        let decomposed = svd(matrix, false)?;
        let rank = decomposed.singular_values.iter().filter(|&&value| value > decomposed.band).count();
        let u = decomposed.u.slice(s![.., ..rank]).to_owned();
        let v = decomposed.vt.slice(s![..rank, ..]).t().to_owned();
        let omega = basis_orthonormality_defect(u.view()).max(basis_orthonormality_defect(v.view()));
        Ok(Self {
            u,
            s: decomposed.singular_values.iter().take(rank).copied().collect(),
            v,
            sigma_max: decomposed.singular_values.first().copied().unwrap_or(0.0),
            band: decomposed.band,
            omega,
        })
    }

    fn rank(&self) -> usize {
        self.s.len()
    }

    /// The backward band on the matrix the resolved factors exactly decompose: the SVD's
    /// band, the dropped singular values (each at most the band), and the distance of the
    /// computed bases from orthonormal ones.
    fn backward(&self) -> f64 {
        let projector = (2.0 + self.omega) * self.omega * self.sigma_max;
        inflated(2.0 * self.band + 2.0 * projector, 4)
    }
}

/// `‖fl(M v)‖₂` and a bound on its distance from `‖(M + δM) v‖₂` for every `|δM_ij| ≤
/// radius_j`: `‖γ_n |M||v| + |δM||v|‖₂` from the products and the declared radius, and
/// `γ_{m+1}` of the value for the norm's own sum and square root.
fn product_norm(matrix: ArrayView2<'_, f64>, vector: ArrayView1<'_, f64>, radius: &[f64]) -> (f64, f64) {
    let (rows, cols) = matrix.dim();
    let product = matrix.dot(&vector);
    let absolute = matrix.mapv(f64::abs).dot(&vector.mapv(f64::abs));
    let declared: f64 = radius.iter().zip(vector.iter()).map(|(r, v)| r * v.abs()).sum();
    let growth = accumulation_growth(cols);
    let error = absolute
        .iter()
        .map(|entry| {
            let bound = growth * entry + declared;
            bound * bound
        })
        .sum::<f64>()
        .sqrt();
    let value = frobenius_norm(product.view());
    (value, inflated(error + accumulation_growth(rows + 1) * value, 2))
}

fn scaled_rows(matrix: ArrayView2<'_, f64>, scale: Option<&[f64]>, invert: bool) -> Array2<f64> {
    let mut out = matrix.to_owned();
    if let Some(scale) = scale {
        for (mut row, &factor) in out.rows_mut().into_iter().zip(scale) {
            if invert {
                row.mapv_inplace(|value| value / factor);
            } else {
                row.mapv_inplace(|value| value * factor);
            }
        }
    }
    out
}

/// The kernel finding of one side: a direction over its columns in the numerical kernel
/// of `X̃` along which `Y` is resolved from zero.
fn kernel_violation(
    side: ConstraintSide,
    constraints: &Side,
    x_scaled: ArrayView2<'_, f64>,
    y_scaled: ArrayView2<'_, f64>,
    resolved: &Resolved,
) -> Result<Option<EvidenceStatus<LinearWitness, LinearSiteDomain>>, CompileError> {
    let columns = constraints.columns();
    if columns == 0 || resolved.rank() == columns {
        return Ok(None);
    }
    // Ỹ (I − V Vᵀ): the part of the demand on directions the inputs do not resolve.
    let projected = fast_ab(&fast_ab(&y_scaled, &resolved.v), &resolved.v.t());
    let perpendicular = &y_scaled - &projected;
    let top = svd(perpendicular.view(), false)?;
    if top.singular_values.is_empty() || top.singular_values[0] == 0.0 {
        return Ok(None);
    }
    let direction = top.vt.row(0).to_owned();
    let (change, change_band) = product_norm(constraints.y.view(), direction.view(), &constraints.radius);
    if !(change - change_band > 0.0) {
        return Ok(None);
    }
    let zeros = vec![0.0; columns];
    let (input, input_band) = product_norm(x_scaled, direction.view(), &zeros);
    let input_upper = inflated(input + input_band, 1);
    // `v` must lie in the numerical kernel: `‖X̃ v‖` not resolved above the backward band.
    if input - input_band > resolved.backward() {
        return Ok(None);
    }
    let (scaled_change, scaled_band) = product_norm(y_scaled, direction.view(), &constraints.radius);
    let lower_change = (scaled_change - scaled_band).max(0.0);
    let bound = if input_upper == 0.0 {
        f64::INFINITY
    } else {
        (lower_change / input_upper).next_down().max(0.0)
    };
    let (original_input, original_band) = product_norm(constraints.x.view(), direction.view(), &zeros);
    Ok(Some(EvidenceStatus::counterexample(
        change,
        change_band,
        0.0,
        LinearWitness::Kernel {
            side,
            direction: direction.to_vec(),
            input_norm_upper: inflated(original_input + original_band, 1),
            edit_norm_lower_bound: bound,
        },
    )?))
}

/// `X_lᵀ Y_r − Y_lᵀ X_r`, the entry most resolved from zero, as a counterexample.
fn compatibility_violation(
    left: &Side,
    right: &Side,
) -> Result<(Option<EvidenceStatus<LinearWitness, LinearSiteDomain>>, f64), CompileError> {
    if left.columns() == 0 || right.columns() == 0 {
        return Ok((None, 0.0));
    }
    let first = fast_atb(&left.x, &right.y);
    let second = fast_atb(&left.y, &right.x);
    let first_abs = fast_atb(&left.x.mapv(f64::abs), &right.y.mapv(f64::abs));
    let second_abs = fast_atb(&left.y.mapv(f64::abs), &right.x.mapv(f64::abs));
    let left_mass: Vec<f64> = left.x.columns().into_iter().map(|c| c.iter().map(|v| v.abs()).sum()).collect();
    let right_mass: Vec<f64> = right.x.columns().into_iter().map(|c| c.iter().map(|v| v.abs()).sum()).collect();
    let (rows, cols) = (left.x.nrows(), right.x.nrows());
    let mut best: Option<(f64, f64, usize, usize)> = None;
    let mut squares = 0.0;
    for i in 0..left.columns() {
        for j in 0..right.columns() {
            let difference = first[[i, j]] - second[[i, j]];
            let band = inflated(
                accumulation_growth(rows) * first_abs[[i, j]]
                    + accumulation_growth(cols) * second_abs[[i, j]]
                    + left_mass[i] * right.radius[j]
                    + left.radius[i] * right_mass[j]
                    + UNIT_ROUNDOFF * difference.abs(),
                2,
            );
            let upper = difference.abs() + band;
            squares += upper * upper;
            let excess = difference.abs() - band;
            if excess > 0.0 && best.is_none_or(|(value, err, ..)| excess > value - err) {
                best = Some((difference.abs(), band, i, j));
            }
        }
    }
    let upper = inflated(squares.sqrt(), left.columns() * right.columns());
    let witness = match best {
        Some((value, band, i, j)) => Some(EvidenceStatus::counterexample(
            value,
            band,
            0.0,
            LinearWitness::Compatibility { left: i, right: j },
        )?),
        None => None,
    };
    Ok((witness, upper))
}

/// Coverage of one requirement's class by the resolved range `U` of its side (scaled
/// coordinates `D⁻¹`).
fn coverage(
    class: &ResponseClass<'_>,
    width: usize,
    scale: Option<&[f64]>,
    resolved: &Resolved,
) -> Result<Coverage, CompileError> {
    let basis = match class {
        ResponseClass::Sample => return Ok(Coverage::SampleOnly),
        ResponseClass::AllInputs => {
            if resolved.rank() == width {
                return Ok(Coverage::Covered);
            }
            // Every input: the complement of the resolved range has dimension
            // `width − rank`. One direction: the coordinate axis the range reaches least,
            // made orthogonal to the range.
            let reach: Vec<f64> = resolved.u.rows().into_iter().map(|row| frobenius_norm(row)).collect();
            let axis = (0..width).fold(0, |best, index| if reach[index] < reach[best] { index } else { best });
            let mut direction = Array1::<f64>::zeros(width);
            direction[axis] = 1.0;
            let along = resolved.u.t().dot(&direction);
            direction = direction - resolved.u.dot(&along);
            let unscaled = unscale_direction(direction.view(), scale, true);
            return Ok(Coverage::Uncovered {
                dimension: width - resolved.rank(),
                direction: unscaled,
            });
        }
        ResponseClass::Span(basis) => basis,
    };
    require_shape("response class basis", (basis.nrows(), width), basis.dim())?;
    require_finite("response class basis", basis.iter().copied())?;
    let class = scaled_rows(basis.t(), scale, true);
    let along = fast_ab(&resolved.u, &fast_atb(&resolved.u, &class));
    let outside = &class - &along;
    let decomposed = svd(outside.view(), false)?;
    let threshold = inflated(
        decomposed.band
            + (2.0 + resolved.omega) * resolved.omega * frobenius_norm(class.view())
            + accumulation_growth(2 * resolved.rank() + 1) * frobenius_norm(class.view()),
        2,
    );
    let dimension = decomposed.singular_values.iter().filter(|&&value| value > threshold).count();
    if dimension == 0 {
        return Ok(Coverage::Covered);
    }
    Ok(Coverage::Uncovered {
        dimension,
        direction: unscale_direction(decomposed.u.column(0), scale, true),
    })
}

/// A scaled-coordinate direction `d̃ = D⁻¹ d` back to a unit vector of original coordinates.
fn unscale_direction(direction: ArrayView1<'_, f64>, scale: Option<&[f64]>, was_inverted: bool) -> Vec<f64> {
    let mut out = direction.to_owned();
    if let Some(scale) = scale {
        for (value, &factor) in out.iter_mut().zip(scale) {
            *value = if was_inverted { *value * factor } else { *value / factor };
        }
    }
    let length = frobenius_norm(out.view());
    if length > 0.0 {
        out.mapv_inplace(|value| value / length);
    }
    out.to_vec()
}

/// The residual `left (rightᵀ X) − Y` of one side (`X` `width × n` read by `right`), its
/// largest entry and the entry's band, and that entry's position.
fn side_residual(
    reader: ArrayView2<'_, f64>,
    writer: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> (f64, f64, usize, usize) {
    let terms = reader.ncols();
    if x.ncols() == 0 {
        return (0.0, 0.0, 0, 0);
    }
    let inner = fast_atb(&reader, &x);
    let inner_error = fast_atb(&reader.mapv(f64::abs), &x.mapv(f64::abs)) * accumulation_growth(reader.nrows());
    let product = fast_ab(&writer, &inner);
    let writer_abs = writer.mapv(f64::abs);
    let product_error =
        fast_ab(&writer_abs, &inner.mapv(f64::abs)) * accumulation_growth(terms) + fast_ab(&writer_abs, &inner_error);
    let residual = &product - &y;
    let mut best = (0.0, 0.0, 0, 0);
    let mut best_excess = f64::NEG_INFINITY;
    for ((entry, column), value) in residual.indexed_iter().map(|(index, value)| (index, *value)) {
        let band = inflated(product_error[[entry, column]] + UNIT_ROUNDOFF * value.abs(), 1);
        if value.abs() - band > best_excess || (best_excess == f64::NEG_INFINITY) {
            best_excess = value.abs() - band;
            best = (value.abs(), band, column, entry);
        }
    }
    best
}

/// A set-type target turned into the change the edit must make, `targets − inputs · map`
/// (`map` is `Wᵀ` for an identity use, `W` for a transposed use or a row selection), and per
/// observation the entrywise radius that change is known to: the declared radius plus the
/// formation band `γ_{k+1} (|inputs||map| + |targets|)` of its largest entry.
fn set_to_change(
    targets: ArrayView2<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    map: ArrayView2<'_, f64>,
    declared: f64,
) -> (Array2<f64>, Vec<f64>) {
    let native = inputs.dot(&map);
    let changes = &targets - &native;
    let magnitude = inputs.mapv(f64::abs).dot(&map.mapv(f64::abs)) + targets.mapv(f64::abs);
    let growth = accumulation_growth(inputs.ncols() + 1);
    let radii = magnitude
        .rows()
        .into_iter()
        .map(|row| inflated(declared + growth * row.iter().fold(0.0_f64, |m, v| m.max(*v)), 1))
        .collect();
    (changes, radii)
}

/// `sup` over the declared off-target observations of `‖ΔW x‖₂` (identity use) or
/// `‖ΔWᵀ x‖₂` (transposed use), exhaustive over them, with its evaluation band: the damage
/// of the approximate transformation restricted to off-target inputs.
pub fn off_target_damage(
    plan: &NativeEditPlan,
    registry: &TensorRegistry,
    storage: &TensorId,
    off_target: &[OffTargetInputs<'_>],
) -> Result<Option<EvidenceStatus<OffTargetWitness, OffTargetDomain>>, CompileError> {
    let observations: usize = off_target.iter().map(|set| set.inputs.nrows()).sum();
    if observations == 0 {
        return Ok(None);
    }
    let edit = plan.edits().iter().find(|edit| &edit.storage == storage);
    let mut best: Option<(f64, f64, OffTargetWitness)> = None;
    for (set_index, set) in off_target.iter().enumerate() {
        require_finite("off-target inputs", set.inputs.iter().copied())?;
        let read = registry.resolve_use_site(&set.site)?;
        if &read.storage != storage {
            return Err(CompileError::UseSite {
                site: set.site.0.clone(),
                reason: format!("reads {:?}, not the edited {:?}", read.storage.0, storage.0),
            });
        }
        let (reader, writer) = match (read.map, edit) {
            (UseMap::Stored, _) => {
                return Err(CompileError::UseSite {
                    site: set.site.0.clone(),
                    reason: "off-target damage is measured on linear uses".to_string(),
                });
            }
            (UseMap::Linear(_), None) => {
                best = best.or(Some((0.0, 0.0, OffTargetWitness { set: set_index, observation: 0 })));
                continue;
            }
            (UseMap::Linear(TieOrientation::Identity), Some(edit)) => (edit.delta.right(), edit.delta.left()),
            (UseMap::Linear(TieOrientation::Transpose), Some(edit)) => (edit.delta.left(), edit.delta.right()),
        };
        require_shape("off-target inputs", (set.inputs.nrows(), reader.nrows()), set.inputs.dim())?;
        let zeros = vec![0.0; writer.ncols()];
        let reader_abs = reader.mapv(f64::abs);
        let writer_abs = writer.mapv(f64::abs);
        for (observation, x) in set.inputs.rows().into_iter().enumerate() {
            let inner = reader.t().dot(&x);
            let inner_error = reader_abs.t().dot(&x.mapv(f64::abs)) * accumulation_growth(reader.nrows());
            let (value, band) = product_norm(writer, inner.view(), &zeros);
            let band = inflated(band + frobenius_norm(writer_abs.dot(&inner_error).view()), 1);
            if best.is_none_or(|(v, b, _)| value + band > v + b) {
                best = Some((value, band, OffTargetWitness { set: set_index, observation }));
            }
        }
    }
    let Some((value, band, witness)) = best else {
        return Ok(None);
    };
    Ok(Some(EvidenceStatus::exact(
        value,
        band,
        ExactBasis::Exhaustive {
            cardinality: observations as u64,
        },
        Some(witness),
        OffTargetDomain { observations },
    )?))
}

/// Compiles the requirements into one global edit of `problem.storage`, or a witness.
pub fn compile_linear_site(
    problem: &LinearSiteProblem<'_>,
    control: &str,
    governor: &MemoryGovernor,
) -> Result<LinearSiteReport, CompileError> {
    let registry = problem.registry;
    let stored = registry
        .storage(&problem.storage)
        .ok_or_else(|| CompileError::NotStorage(problem.storage.0.clone()))?;
    if stored.shape.len() != 2 {
        return Err(CompileError::InvalidDeclaration {
            what: "storage",
            reason: format!("{:?} has shape {:?}, not a matrix", problem.storage.0, stored.shape),
        });
    }
    let (rows, cols) = (stored.shape[0], stored.shape[1]);
    require_shape("native values", (rows, cols), problem.native.dim())?;
    require_finite("native values", problem.native.iter().copied())?;
    if let Some(scale) = problem.metric.row_scale() {
        require_shape("metric row scale", (rows, 1), (scale.len(), 1))?;
    }
    if let Some(scale) = problem.metric.col_scale() {
        require_shape("metric column scale", (cols, 1), (scale.len(), 1))?;
    }
    let mut right = Side::empty(cols, rows);
    let mut left = Side::empty(rows, cols);
    let mut named = Vec::new();
    for (index, requirement) in problem.requirements.iter().enumerate() {
        match requirement {
            Requirement::Linear {
                site,
                inputs,
                targets,
                target_radius,
                ..
            } => {
                require_radius(*target_radius)?;
                require_finite("requirement inputs", inputs.iter().copied())?;
                require_finite("requirement targets", targets.iter().copied())?;
                let read = registry.resolve_use_site(site)?;
                if read.storage != problem.storage {
                    return Err(CompileError::UseSite {
                        site: site.0.clone(),
                        reason: format!("reads {:?}, not the edited {:?}", read.storage.0, problem.storage.0),
                    });
                }
                named.push(site.clone());
                match read.map {
                    UseMap::Linear(TieOrientation::Identity) => {
                        require_shape("identity-use inputs", (inputs.nrows(), cols), inputs.dim())?;
                        require_shape("identity-use targets", (inputs.nrows(), rows), targets.dim())?;
                        let (changes, radii) = set_to_change(*targets, *inputs, problem.native.t(), *target_radius);
                        right.push(index, *inputs, changes.view(), &radii)?;
                    }
                    UseMap::Linear(TieOrientation::Transpose) => {
                        require_shape("transposed-use inputs", (inputs.nrows(), rows), inputs.dim())?;
                        require_shape("transposed-use targets", (inputs.nrows(), cols), targets.dim())?;
                        let (changes, radii) = set_to_change(*targets, *inputs, problem.native, *target_radius);
                        left.push(index, *inputs, changes.view(), &radii)?;
                    }
                    UseMap::Stored => {
                        return Err(CompileError::UseSite {
                            site: site.0.clone(),
                            reason: "a stored read is constrained through Requirement::StoredRows".to_string(),
                        });
                    }
                }
            }
            Requirement::StoredRows {
                rows: selected,
                targets,
                target_radius,
            } => {
                require_radius(*target_radius)?;
                require_finite("stored-row targets", targets.iter().copied())?;
                require_shape("stored-row targets", (selected.len(), cols), targets.dim())?;
                let mut selector = Array2::<f64>::zeros((selected.len(), rows));
                for (position, &row) in selected.iter().enumerate() {
                    if row >= rows || selected[..position].contains(&row) {
                        return Err(CompileError::InvalidDeclaration {
                            what: "stored rows",
                            reason: format!("row {row} is out of range or repeated for {rows} rows"),
                        });
                    }
                    selector[[position, row]] = 1.0;
                }
                let (changes, radii) = set_to_change(*targets, selector.view(), problem.native, *target_radius);
                left.push(index, selector.view(), changes.view(), &radii)?;
            }
        }
    }
    let undeclared_uses = registry
        .use_sites_of(&problem.storage)
        .into_iter()
        .filter(|site| !named.contains(site))
        .cloned()
        .collect();
    let row_scale = problem.metric.row_scale();
    let col_scale = problem.metric.col_scale();
    let right_x = scaled_rows(right.x.view(), col_scale, true);
    let right_y = scaled_rows(right.y.view(), row_scale, false);
    let left_x = scaled_rows(left.x.view(), row_scale, true);
    let left_y = scaled_rows(left.y.view(), col_scale, false);
    let right_resolved = Resolved::of(right_x.view())?;
    let left_resolved = Resolved::of(left_x.view())?;
    let domain_classes = |covered: bool| LinearSiteDomain {
        right_columns: right.columns(),
        left_columns: left.columns(),
        classes_covered: covered,
    };
    let coverage = problem
        .requirements
        .iter()
        .map(|requirement| match requirement {
            Requirement::Linear { site, class, .. } => {
                let read = registry.resolve_use_site(site)?;
                match read.map {
                    UseMap::Linear(TieOrientation::Transpose) => coverage(class, rows, row_scale, &left_resolved),
                    UseMap::Linear(TieOrientation::Identity) | UseMap::Stored => {
                        coverage(class, cols, col_scale, &right_resolved)
                    }
                }
            }
            Requirement::StoredRows { .. } => Ok(Coverage::StoredRows),
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let covered = coverage
        .iter()
        .all(|entry| matches!(entry, Coverage::Covered | Coverage::StoredRows));
    let report = |compiled: CompiledControl<LinearWitness, LinearSiteDomain>, metric_norm, allowance| {
        let off_target_damage = match &compiled.plan {
            Some(plan) => off_target_damage(plan, registry, &problem.storage, &problem.off_target)?,
            None => None,
        };
        Ok::<_, CompileError>(LinearSiteReport {
            compiled,
            metric_norm,
            right_rank: right_resolved.rank(),
            left_rank: left_resolved.rank(),
            right_origins: right.origins.clone(),
            left_origins: left.origins.clone(),
            coverage: coverage.clone(),
            undeclared_uses: Vec::clone(&undeclared_uses),
            allowance,
            off_target_damage,
        })
    };
    let descriptive = |reason, witness: EvidenceStatus<LinearWitness, LinearSiteDomain>| {
        CompiledControl::new(
            control.to_string(),
            None,
            ControlRealization::descriptive(reason, Some(witness))?,
        )
    };
    let cardinality = (right.columns() + left.columns()) as u64;
    if cardinality == 0 {
        return Err(CompileError::InvalidDeclaration {
            what: "requirements",
            reason: "a linear-site problem needs at least one constraint column".to_string(),
        });
    }
    // The native setting: every demanded change lies within the radius it is known to, so
    // the targets are θ's own response and ρ = θ runs the original tensors.
    let unresolved = |side: &Side| {
        side.y
            .columns()
            .into_iter()
            .zip(&side.radius)
            .all(|(column, radius)| column.iter().all(|value| value.abs() <= *radius))
    };
    if unresolved(&right) && unresolved(&left) {
        let largest = |side: &Side| side.y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let widest = |side: &Side| side.radius.iter().fold(0.0_f64, |m, v| m.max(*v));
        let status = EvidenceStatus::exact(
            largest(&right).max(largest(&left)),
            widest(&right).max(widest(&left)),
            if covered { ExactBasis::Algebraic } else { ExactBasis::Exhaustive { cardinality } },
            None,
            domain_classes(covered),
        )?;
        let realization = if covered {
            ControlRealization::exactly_realized(status)?
        } else {
            ControlRealization::empirically_validated(status)?
        };
        return report(
            CompiledControl::new(control.to_string(), Some(NativeEditPlan::native()), realization)?,
            Some((0.0, 0.0)),
            [0.0, 0.0],
        );
    }
    for (side, constraints, x, y, resolved) in [
        (ConstraintSide::Right, &right, &right_x, &right_y, &right_resolved),
        (ConstraintSide::Left, &left, &left_x, &left_y, &left_resolved),
    ] {
        if let Some(witness) = kernel_violation(side, constraints, x.view(), y.view(), resolved)? {
            return report(descriptive(DescriptiveReason::KernelViolation, witness)?, None, [0.0, 0.0]);
        }
    }
    let (compatibility, compatibility_upper) = compatibility_violation(&left, &right)?;
    if let Some(witness) = compatibility {
        return report(descriptive(DescriptiveReason::IncompatibleSides, witness)?, None, [0.0, 0.0]);
    }

    // E = U_l S_l⁻¹ V_lᵀ Ỹ_lᵀ + (I − U_l U_lᵀ) Ỹ_r V_r S_r⁻¹ U_rᵀ, as factors.
    let inverse = |values: &[f64]| Array2::from_diag(&Array1::from_iter(values.iter().map(|value| 1.0 / value)));
    let left_first = left_resolved.u.clone();
    let right_first = fast_ab(&fast_ab(&left_y, &left_resolved.v), &inverse(&left_resolved.s));
    let pulled = fast_ab(&fast_ab(&right_y, &right_resolved.v), &inverse(&right_resolved.s));
    let left_second = &pulled - &fast_ab(&left_resolved.u, &fast_atb(&left_resolved.u, &pulled));
    let right_second = right_resolved.u.clone();
    let scaled_left = concatenate(Axis(1), &[left_first.view(), left_second.view()]).map_err(|error| {
        CompileError::InvalidDeclaration {
            what: "edit factors",
            reason: error.to_string(),
        }
    })?;
    let scaled_right = concatenate(Axis(1), &[right_first.view(), right_second.view()]).map_err(|error| {
        CompileError::InvalidDeclaration {
            what: "edit factors",
            reason: error.to_string(),
        }
    })?;
    let terms = scaled_left.ncols();
    let gram_product = fast_atb(&scaled_left, &scaled_left) * fast_atb(&scaled_right, &scaled_right);
    let squared: f64 = gram_product.sum();
    let squared_band = accumulation_growth(rows.max(cols) + terms * terms)
        * (fast_atb(&scaled_left.mapv(f64::abs), &scaled_left.mapv(f64::abs))
            * fast_atb(&scaled_right.mapv(f64::abs), &scaled_right.mapv(f64::abs)))
        .sum();
    let metric_norm = squared.max(0.0).sqrt();
    let metric_band = if metric_norm > 0.0 {
        inflated(squared_band / (2.0 * metric_norm) + UNIT_ROUNDOFF * metric_norm, 1)
    } else {
        squared_band.sqrt()
    };
    let edit_upper = inflated(metric_norm + metric_band, 1);

    let delta_left = scaled_rows(scaled_left.view(), row_scale, true);
    let delta_right = scaled_rows(scaled_right.view(), col_scale, true);
    let inv_max = |scale: Option<&[f64]>| scale.map_or(1.0, |values| values.iter().fold(0.0_f64, |m, v| m.max(1.0 / v)));
    let target_mass = |side: &Side, y: &Array2<f64>, scale: Option<&[f64]>| {
        let declared = side.radius.iter().fold(0.0_f64, |m, r| m.max(*r)) * scale.map_or(1.0, |v| v.iter().fold(0.0, |m: f64, s| m.max(*s)));
        accumulation_growth(side.columns() + 2 * terms + 2) * frobenius_norm(y.view()) + declared * (side.columns() as f64).sqrt()
    };
    let right_allowance = inflated(
        inv_max(row_scale)
            * (edit_upper * right_resolved.backward()
                + target_mass(&right, &right_y, row_scale)
                + if left_resolved.rank() > 0 {
                    compatibility_upper / (left_resolved.s[left_resolved.rank() - 1] - left_resolved.band).max(f64::MIN_POSITIVE)
                } else {
                    0.0
                }),
        4,
    );
    let left_allowance = inflated(
        inv_max(col_scale)
            * (edit_upper * (left_resolved.backward() + (2.0 + left_resolved.omega) * left_resolved.omega * left_resolved.sigma_max)
                + target_mass(&left, &left_y, col_scale)),
        4,
    );
    let allowance = [right_allowance, left_allowance];

    let (right_value, right_band, right_column, right_entry) =
        side_residual(delta_right.view(), delta_left.view(), right.x.view(), right.y.view());
    let (left_value, left_band, left_column, left_entry) =
        side_residual(delta_left.view(), delta_right.view(), left.x.view(), left.y.view());
    let (value, band, side, column, entry, side_allowance) = if right_value - right_band - right_allowance
        >= left_value - left_band - left_allowance
    {
        (right_value, right_band, ConstraintSide::Right, right_column, right_entry, right_allowance)
    } else {
        (left_value, left_band, ConstraintSide::Left, left_column, left_entry, left_allowance)
    };
    let witness = LinearWitness::Residual { side, column, entry };
    if (value - band).next_down() > side_allowance {
        let refuted = EvidenceStatus::counterexample(value, band, side_allowance, witness)?;
        return report(
            descriptive(DescriptiveReason::ResidualResolved, refuted)?,
            Some((metric_norm, metric_band)),
            allowance,
        );
    }

    let plan = if terms == 0 {
        NativeEditPlan::native()
    } else {
        NativeEditPlan::new(
            registry,
            vec![CompiledParameterEdit {
                storage: problem.storage.clone(),
                delta: FactoredEdit::new(delta_left, delta_right)?,
            }],
        )?
    };
    if let Some(violation) = check_ties(&plan, &problem.ties, governor)?.into_iter().next() {
        let refuted = EvidenceStatus::counterexample(
            violation.difference,
            violation.band,
            0.0,
            LinearWitness::Tie(violation),
        )?;
        return report(
            descriptive(DescriptiveReason::TieBroken, refuted)?,
            Some((metric_norm, metric_band)),
            allowance,
        );
    }
    let realization = if covered {
        ControlRealization::exactly_realized(EvidenceStatus::exact(
            value,
            band,
            ExactBasis::Algebraic,
            Some(witness),
            domain_classes(true),
        )?)?
    } else {
        ControlRealization::empirically_validated(EvidenceStatus::exact(
            value,
            band,
            ExactBasis::Exhaustive { cardinality },
            Some(witness),
            domain_classes(false),
        )?)?
    };
    report(
        CompiledControl::new(control.to_string(), Some(plan), realization)?,
        Some((metric_norm, metric_band)),
        allowance,
    )
}

fn require_radius(radius: f64) -> Result<(), CompileError> {
    if radius.is_finite() && radius >= 0.0 {
        Ok(())
    } else {
        Err(CompileError::InvalidDeclaration {
            what: "target radius",
            reason: format!("must be finite and nonnegative; got {radius}"),
        })
    }
}

