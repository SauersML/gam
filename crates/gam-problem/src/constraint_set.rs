//! Typed structured constraint carriers for large factored coefficient blocks.
//!
//! The dense [`LinearInequalityConstraints`] system stores every row
//! explicitly, which is exact and fine for the small monotone blocks (a
//! `p × p` identity cone). A Khatri-Rao tensor block is different: the
//! monotonicity cone of a conditional transformation `h(y|x) = Σ_k α_k(x)
//! v_k(y)` is `α_k(x_i) ≥ 0` for every observation row `i` and every shape
//! column `k` — `n · p_shape` rows over `p_resp · p_cov` coefficients whose
//! dense materialization is gigabytes (gam#2306), while every operation an
//! active-set method actually performs factors through the covariate design
//! `Ψ` (`n × p_cov`):
//!
//! * constraint values are the columns of `Γ = Ψ Aᵀ` (one `n × p_cov` GEMM
//!   per shape column),
//! * a single row is `(e_k ⊗ ψ_i)ᵀ` — gathered densely only for the (small)
//!   active set,
//! * row norms are `‖ψ_i‖`, shared by every shape column.
//!
//! [`ConstraintSet`] is the closed union the solver plumbing carries: the
//! dense system verbatim, or the factored cone. Semantics are IDENTICAL to
//! canonicalizing the equivalent dense system: every slack / violation is
//! measured on unit-normalized rows, so tolerances stay geometric.

use crate::linear_constraints::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

/// Primal-feasibility tolerance of the inequality-constrained active-set Newton
/// solver, measured in the unit-normalized row metric this module defines:
/// a point `β` is feasible for a [`ConstraintSet`] exactly when
///
/// ```text
/// max_r  (b_r − a_r·β) / ‖a_r‖  ≤  PRIMAL_FEASIBILITY_TOL
/// ```
///
/// over the non-vacuous rows — the quantity [`ConstraintSet::max_scaled_violation`]
/// returns. This is the ONE definition of "feasible" in the codebase; the solver
/// certifies its returned iterate against it, every entry gate admits against it,
/// and [`ConstraintSet::max_contract_feasible_step`] sizes steps against it.
///
/// It lives beside the metric rather than in the solver because the two are the
/// same statement: the metric says what is measured, this says at what resolution.
/// `gam_solve::active_set` re-exports it as `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`.
///
/// Any consumer that re-derives a RAW (un-scaled) feasibility tolerance from a
/// returned iterate must scale this value by the per-row normalization the
/// constraint builder applied; demanding tighter feasibility than this is
/// inconsistent with the solver contract and will spuriously reject valid
/// boundary solutions (gam#2719: a step rule that demanded exact feasibility
/// refused 314 steps that violated nothing at this tolerance).
pub const PRIMAL_FEASIBILITY_TOL: f64 = 1e-8;

/// Can this row's feasibility be DECIDED by comparison at all?
///
/// EVERY feasibility rule in this module decides with an ordering predicate on
/// per-row quantities — `slack < −tol`, `drift ≥ 0`, `t < step`,
/// `violation > worst` — and EVERY one of those is `false` for `NaN`. A row
/// carrying a `NaN` therefore contributes NOTHING to any of those minima and
/// maxima, and the rule answers with its neutral element: "take the whole
/// step", "nothing is violated". That is the exact opposite of the truth, and
/// it is gam#2721: a step with a `NaN` component was certified at `α = 1.0`,
/// and its caller rejects only `!α.is_finite() || α ≤ 0.0`, neither of which
/// `1.0` is.
///
/// The quantities are therefore tested BEFORE they are compared, and a row that
/// cannot be decided is refused by name rather than skipped. The predicate is
/// exported — rather than re-written at each site — because the defect WAS the
/// rule existing in several copies and being repaired in one of them: the two
/// fraction-to-boundary rules here, the violation sweep here, the saddle-escape
/// chord truncation in `gam-custom-family`, and the Bernoulli marginal-slope
/// segment cap in `gam-models` all decide with the same comparisons.
///
/// `NaN` is the value that cannot be compared, but it is not the only value
/// that must be refused. An infinite drift passes `drift ≥ 0` and an infinite
/// iterate value drives the violation to `−∞`, both of which read as "this row
/// does not object" for an argument that is not a point. And the carrier's own
/// constructor already holds the same line —
/// `LinearInequalityConstraints::new` rejects a non-finite `A` or `b` with the
/// identical reason — so requiring finiteness here keeps the row descriptors
/// and the per-iterate quantities under ONE rule rather than two.
///
/// A `NaN` `row_norm` additionally defeats the `norm <= 0.0` vacuity test that
/// would otherwise be the branch to catch it, which is why the norm is checked
/// here and not left to that branch.
///
/// This does NOT collide with the legitimately-vacuous row: `‖a‖ = 0` with a
/// bound at or below zero is finite, passes here, and keeps its own
/// disposition in each rule.
pub fn feasibility_quantities_are_finite(quantities: &[f64]) -> bool {
    quantities.iter().all(|q| q.is_finite())
}

/// The contract-feasible ratio test itself, over already-evaluated constraint
/// values, so every carrier — the dense system, the factored cone, the
/// block-diagonal composition — runs the SAME arithmetic without any of them
/// having to be materialized as another.
///
/// `values[r]` is `a_r·β`, `directional[r]` is `a_r·δ` (the constraint
/// functional is linear, so its value at `δ` IS the directional derivative).
/// The rule is documented on [`ConstraintSet::max_contract_feasible_step`].
pub(crate) fn contract_feasible_step_over_rows<B, N>(
    values: &Array1<f64>,
    directional: &Array1<f64>,
    bound: B,
    row_norm: N,
) -> Result<ContractFeasibleStep, ContractFeasibleStepError>
where
    B: Fn(usize) -> Result<f64, String>,
    N: Fn(usize) -> Result<f64, String>,
{
    let tol = PRIMAL_FEASIBILITY_TOL;
    let mut limit = ContractFeasibleStep::UNLIMITED;
    for row in 0..values.len() {
        let norm = row_norm(row).map_err(ContractFeasibleStepError::Carrier)?;
        let bound = bound(row).map_err(ContractFeasibleStepError::Carrier)?;
        // A ROW THAT CANNOT BE COMPARED IS NOT A FEASIBLE ROW (gam#2721) — see
        // [`feasibility_quantities_are_finite`] for why skipping it certifies
        // the whole step. Refuse instead, and name the condition: "this row is
        // not a number" is a different condition from "the current iterate
        // violates this row", so it gets its own variant rather than being
        // reported through `InfeasibleIterate`.
        if !feasibility_quantities_are_finite(&[norm, bound, values[row], directional[row]]) {
            return Err(ContractFeasibleStepError::NonFinite {
                row,
                scaled_slack: (values[row] - bound) / norm,
                scaled_drift: directional[row] / norm,
            });
        }
        if !(norm.is_finite() && norm > 0.0) {
            // A vacuous row constrains nothing unless its bound is positive,
            // in which case the feasible set is empty and no step fraction
            // exists. Same disposition as the solver's own violation scan.
            if bound > 0.0 {
                return Err(ContractFeasibleStepError::InfeasibleIterate {
                    row,
                    scaled_slack: f64::NEG_INFINITY,
                });
            }
            continue;
        }
        let slack = (values[row] - bound) / norm;
        let drift = directional[row] / norm;
        if slack < -tol {
            return Err(ContractFeasibleStepError::InfeasibleIterate {
                row,
                scaled_slack: slack,
            });
        }
        if drift >= 0.0 {
            continue;
        }
        if slack + drift >= -tol {
            // The endpoint of the FULL step is feasible on this row to the
            // contract. Nothing to limit.
            continue;
        }
        // `slack ≥ −tol` and `slack + drift < −tol` give `slack < −drift`, so
        // this ratio is strictly below 1 and non-negative.
        let fraction = (slack.max(0.0) / -drift).clamp(0.0, 1.0);
        if fraction < limit.fraction {
            limit = ContractFeasibleStep {
                fraction,
                blocking_row: Some(row),
                blocking_scaled_slack: slack,
                blocking_scaled_drift: drift,
            };
        }
    }
    Ok(limit)
}

/// Result of the contract-feasible ratio test
/// ([`ConstraintSet::max_contract_feasible_step`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContractFeasibleStep {
    /// Largest fraction in `[0, 1]` such that `β + fraction·δ` is feasible at
    /// [`PRIMAL_FEASIBILITY_TOL`]. `1.0` means no row limits the step.
    ///
    /// `0.0` is a legitimate, non-exceptional answer: it says a row is active
    /// at `β` and `δ` points strictly out of it by more than round-off, so no
    /// positive multiple of `δ` is admissible. The remedy is a projection onto
    /// the active face, not a smaller `δ` — the ratio test is invariant under
    /// `δ ↦ cδ` once the numerator is zero, so shrinking a trust radius against
    /// it cannot converge (gam#2719).
    pub fraction: f64,
    /// Row that limited `fraction`, if any.
    pub blocking_row: Option<usize>,
    /// Scaled slack `(a·β − b)/‖a‖` of `blocking_row` at `β`.
    pub blocking_scaled_slack: f64,
    /// Scaled drift `(a·δ)/‖a‖` of `blocking_row` (strictly negative when a
    /// row blocks).
    pub blocking_scaled_drift: f64,
}

impl ContractFeasibleStep {
    /// The unlimited answer: the whole direction is admissible.
    pub const UNLIMITED: Self = Self {
        fraction: 1.0,
        blocking_row: None,
        blocking_scaled_slack: f64::INFINITY,
        blocking_scaled_drift: 0.0,
    };

}

/// Why the contract-feasible ratio test could not answer.
///
/// Every variant is a violated PRECONDITION of the ratio test, never a small
/// step: "no admissible step exists" is reported as
/// [`ContractFeasibleStep::fraction`] `== 0.0`, not as an error.
#[derive(Clone, Debug, PartialEq)]
pub enum ContractFeasibleStepError {
    /// `beta` / `direction` widths disagree with the constraint carrier.
    Dimension {
        beta: usize,
        direction: usize,
        expected: usize,
    },
    /// The CURRENT iterate violates a row by more than
    /// [`PRIMAL_FEASIBILITY_TOL`], so the ratio test has no feasible origin to
    /// step from. This is the genuine "infeasible iterate" condition and stays
    /// loud.
    InfeasibleIterate { row: usize, scaled_slack: f64 },
    /// A row cannot be DECIDED by comparison: a non-finite row norm, bound,
    /// `a·β` or `a·δ`. Reported rather than skipped: every comparison in the
    /// rule is false for NaN, so a skipped row would silently certify a step
    /// that is not a number as fully feasible (gam#2721). The reported slack
    /// and drift are the scaled quantities as computed, so the offending one is
    /// visible.
    NonFinite {
        row: usize,
        scaled_slack: f64,
        scaled_drift: f64,
    },
    /// The carrier could not evaluate `Aβ` / `Aδ` or a row descriptor.
    Carrier(String),
}

impl std::fmt::Display for ContractFeasibleStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractFeasibleStepError::Dimension {
                beta,
                direction,
                expected,
            } => write!(
                f,
                "constraint step dimension mismatch: beta={beta}, direction={direction}, constraints={expected}"
            ),
            ContractFeasibleStepError::InfeasibleIterate { row, scaled_slack } => write!(
                f,
                "current iterate violates constraint row {row}: scaled slack={scaled_slack:.3e} \
                 below the primal-feasibility contract {PRIMAL_FEASIBILITY_TOL:.3e}"
            ),
            ContractFeasibleStepError::NonFinite {
                row,
                scaled_slack,
                scaled_drift,
            } => write!(
                f,
                "constraint row {row} has a non-finite ratio test: scaled slack={scaled_slack:.3e}, \
                 scaled drift={scaled_drift:.3e}"
            ),
            ContractFeasibleStepError::Carrier(reason) => write!(f, "{reason}"),
        }
    }
}

/// Nonnegativity cone `(e_k ⊗ ψ_i)ᵀ β ≥ 0` for a row-major Khatri-Rao block.
///
/// The coefficient block is `β = vec(A)` with `A` reshaped row-major as
/// `p_left × p_cov` (coefficient `A[k, j] = β[k · p_cov + j]`). The cone
/// constrains the factored linear functionals `α_k(x_i) = ψ_iᵀ A_{k,:}` to be
/// non-negative for every observation row `i` of `factor` and every
/// `k ∈ coupled_rows`.
///
/// Row identifiers are stable and dense: row `r = s · n + i` where `s` indexes
/// into `coupled_rows` and `i` is the observation row. Active-set warm starts
/// therefore survive across iterations exactly as with the dense system.
#[derive(Clone, Debug)]
pub struct KhatriRaoConeConstraints {
    /// Covariate factor `Ψ` (`n × p_cov`).
    factor: Arc<Array2<f64>>,
    /// Euclidean norm of each `Ψ` row (unit-normalization denominators).
    factor_row_norms: Array1<f64>,
    /// Coefficient rows of `A` (indices into `0..p_left`) that carry the cone.
    coupled_rows: Vec<usize>,
    /// Total number of coefficient rows in the block reshape.
    p_left: usize,
    /// Per-row right-hand sides. The homogeneous cone has `b ≡ 0`; a
    /// delta-coordinate solve (`β = β₀ + δ`) shifts them to `−(rowᵀβ₀)`.
    /// Bounds are `O(nrows)` — cheap even when the matrix is not.
    bounds: Option<Array1<f64>>,
}

impl KhatriRaoConeConstraints {
    pub fn new(
        factor: Arc<Array2<f64>>,
        coupled_rows: Vec<usize>,
        p_left: usize,
    ) -> Result<Self, String> {
        if factor.nrows() == 0 || factor.ncols() == 0 {
            return Err("KhatriRaoConeConstraints: factor must be non-empty".to_string());
        }
        if factor.iter().any(|v| !v.is_finite()) {
            return Err("KhatriRaoConeConstraints: factor must be finite".to_string());
        }
        if coupled_rows.is_empty() {
            return Err(
                "KhatriRaoConeConstraints: at least one coupled coefficient row is required"
                    .to_string(),
            );
        }
        let mut seen = vec![false; p_left];
        for &k in &coupled_rows {
            if k >= p_left {
                return Err(format!(
                    "KhatriRaoConeConstraints: coupled row {k} out of range (p_left = {p_left})"
                ));
            }
            if seen[k] {
                return Err(format!(
                    "KhatriRaoConeConstraints: coupled row {k} is duplicated"
                ));
            }
            seen[k] = true;
        }
        let factor_row_norms =
            Array1::from_iter(factor.rows().into_iter().map(|row| row.dot(&row).sqrt()));
        Ok(Self {
            factor,
            factor_row_norms,
            coupled_rows,
            p_left,
            bounds: None,
        })
    }

    pub fn factor(&self) -> &Array2<f64> {
        self.factor.as_ref()
    }

    pub fn coupled_rows(&self) -> &[usize] {
        &self.coupled_rows
    }

    pub fn p_left(&self) -> usize {
        self.p_left
    }

    /// One coupled response-row slot as a standalone cone over a single
    /// `p_cov` coefficient block. The covariate factor remains shared by
    /// [`Arc`]; only the small row-norm vector and this slot's optional bounds
    /// are copied. This is the exact block decomposition of an identity-Hessian
    /// projection, not a reduced-data approximation.
    pub fn single_coupled_slot(&self, slot: usize) -> Result<Self, String> {
        if slot >= self.coupled_rows.len() {
            return Err(format!(
                "KhatriRaoConeConstraints: coupled slot {slot} out of range ({} slots)",
                self.coupled_rows.len()
            ));
        }
        let n = self.factor.nrows();
        let bounds = self
            .bounds
            .as_ref()
            .map(|all| all.slice(ndarray::s![slot * n..(slot + 1) * n]).to_owned());
        Ok(Self {
            factor: Arc::clone(&self.factor),
            factor_row_norms: self.factor_row_norms.clone(),
            coupled_rows: vec![0],
            p_left: 1,
            bounds,
        })
    }

    pub fn nrows(&self) -> usize {
        self.coupled_rows.len() * self.factor.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.p_left * self.factor.ncols()
    }

    /// Decompose a row id into `(coupled-row slot, observation row)`.
    #[inline]
    fn split_row_id(&self, row: usize) -> Result<(usize, usize), String> {
        let n = self.factor.nrows();
        let slot = row / n;
        if slot >= self.coupled_rows.len() {
            return Err(format!(
                "KhatriRaoConeConstraints: row id {row} out of range ({} rows)",
                self.nrows()
            ));
        }
        Ok((slot, row % n))
    }

    /// Raw (un-normalized) constraint values `A β` for the full row set,
    /// laid out slot-major (`r = s·n + i`).
    ///
    /// Cost: one `n × p_cov · p_cov` product per coupled row — never the
    /// `nrows × ncols` dense system.
    pub fn values(&self, beta: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let p_cov = self.factor.ncols();
        if beta.len() != self.ncols() {
            return Err(format!(
                "KhatriRaoConeConstraints: beta length {} != {}",
                beta.len(),
                self.ncols()
            ));
        }
        let n = self.factor.nrows();
        let mut out = Array1::<f64>::zeros(self.nrows());
        for (slot, &k) in self.coupled_rows.iter().enumerate() {
            let block = beta.slice(ndarray::s![k * p_cov..(k + 1) * p_cov]);
            let alpha = self.factor.dot(&block);
            out.slice_mut(ndarray::s![slot * n..(slot + 1) * n])
                .assign(&alpha);
        }
        Ok(out)
    }

    /// Unit-normalization denominator of one row (`‖ψ_i‖`, shared across
    /// coupled slots). Zero rows are vacuous (`0ᵀβ ≥ 0` always holds) exactly
    /// like the canonicalized dense system keeps them inert.
    pub fn row_norm(&self, row: usize) -> Result<f64, String> {
        let (_, i) = self.split_row_id(row)?;
        Ok(self.factor_row_norms[i])
    }

    /// The coefficient columns row `row` acts on, ascending.
    ///
    /// Row `(slot, i)` has normal `e_k ⊗ ψ_i` with `k = coupled_rows[slot]`, and
    /// [`Self::values`] reads exactly the block `β[k·p_cov .. (k+1)·p_cov]`, so
    /// the support is `k·p_cov + j` over the columns `j` where `ψ_{i,j} ≠ 0`.
    /// Every other coefficient has a structurally zero coefficient in this row.
    pub fn row_column_support(&self, row: usize) -> Result<Vec<usize>, String> {
        let (slot, i) = self.split_row_id(row)?;
        let p_cov = self.factor.ncols();
        let base = self.coupled_rows[slot] * p_cov;
        Ok((0..p_cov)
            .filter(|&j| self.factor[[i, j]] != 0.0)
            .map(|j| base + j)
            .collect())
    }

    /// Per-row right-hand side (`0` for the homogeneous cone, shifted values
    /// after [`ConstraintSet::shifted_to_delta`]).
    pub fn bound(&self, row: usize) -> Result<f64, String> {
        self.split_row_id(row)?;
        Ok(self.bounds.as_ref().map_or(0.0, |bounds| bounds[row]))
    }

    /// Materialize the requested rows as a dense system (active-set KKT use;
    /// the id order of `rows` is preserved). Rows come out RAW (un-normalized),
    /// matching the raw dense construction path; callers that need geometric
    /// tolerances canonicalize the gathered system.
    pub fn gather_rows(&self, rows: &[usize]) -> Result<LinearInequalityConstraints, String> {
        let p_cov = self.factor.ncols();
        let mut a = Array2::<f64>::zeros((rows.len(), self.ncols()));
        let mut b = Array1::<f64>::zeros(rows.len());
        for (out_row, &row) in rows.iter().enumerate() {
            let (slot, i) = self.split_row_id(row)?;
            let k = self.coupled_rows[slot];
            a.row_mut(out_row)
                .slice_mut(ndarray::s![k * p_cov..(k + 1) * p_cov])
                .assign(&self.factor.row(i));
            b[out_row] = self.bound(row)?;
        }
        LinearInequalityConstraints::new(a, b)
    }

    /// Exact dense equivalent of the ENTIRE cone. Test/oracle use only — this
    /// is the materialization the carrier exists to avoid.
    pub fn to_dense(&self) -> Result<LinearInequalityConstraints, String> {
        let all: Vec<usize> = (0..self.nrows()).collect();
        self.gather_rows(&all)
    }
}

/// A row index in a [`ConstraintSet`]'s OWN constraint-row space — the space
/// addressed by [`ConstraintSet::values`], [`ConstraintSet::bound`] and
/// [`ConstraintSet::row_norm`], i.e. `0..nrows()`.
///
/// This is NOT a coefficient (β) index. The two spaces have different sizes
/// (`nrows()` vs `ncols()`) and different meanings, and they coincide only in
/// the special case of a square carrier whose row `r` is exactly the box
/// `β_r ≥ 0`. A block-diagonal composition breaks that coincidence: its row ids
/// are the CONCATENATION of the member row counts while its columns are the
/// concatenation of the member column ranges, so as soon as one member has
/// `nrows() < ncols()` (a monotone sub-basis alongside unconstrained intercept /
/// covariate columns) row id `r` of a later block names a β coordinate owned by
/// an EARLIER block. The newtype exists so that mistake cannot be made silently;
/// to go from a row to the coefficients it acts on, call
/// [`ConstraintSet::row_column_support`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintRowId(pub usize);

impl ConstraintRowId {
    /// The raw index, for addressing a `values()` / `bound()` / `row_norm()`
    /// result. Deliberately explicit: reach for this only when indexing
    /// something that really is in constraint-row space.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// One block of a [`ConstraintSet::BlockDiagonal`] composition: an inner set
/// acting on the coefficient columns `[col_start, col_start + set.ncols())` of
/// the joint vector.
#[derive(Clone, Debug)]
pub struct PlacedConstraintBlock {
    pub col_start: usize,
    pub set: ConstraintSet,
}

/// Closed union of the constraint carriers the blockwise solvers accept.
#[derive(Clone, Debug)]
pub enum ConstraintSet {
    /// Explicit rows, exactly as today.
    Dense(LinearInequalityConstraints),
    /// Factored Khatri-Rao nonnegativity cone.
    KhatriRaoCone(KhatriRaoConeConstraints),
    /// Block-diagonal composition over disjoint column ranges of a joint
    /// coefficient vector (the multi-block joint-Newton assembly). Row ids
    /// are the concatenation of the member row ids in order.
    BlockDiagonal {
        blocks: Vec<PlacedConstraintBlock>,
        total_cols: usize,
    },
}

impl ConstraintSet {
    /// Validated block-diagonal composition: member column ranges must lie
    /// inside the joint width and must not overlap.
    pub fn block_diagonal(
        blocks: Vec<PlacedConstraintBlock>,
        total_cols: usize,
    ) -> Result<Self, String> {
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(blocks.len());
        for block in &blocks {
            let end = block.col_start + block.set.ncols();
            if end > total_cols {
                return Err(format!(
                    "ConstraintSet::block_diagonal: block columns {}..{} exceed joint width {}",
                    block.col_start, end, total_cols
                ));
            }
            ranges.push((block.col_start, end));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(format!(
                    "ConstraintSet::block_diagonal: overlapping column ranges {:?} and {:?}",
                    pair[0], pair[1]
                ));
            }
        }
        Ok(ConstraintSet::BlockDiagonal { blocks, total_cols })
    }

    /// Locate the member block owning a joint row id.
    fn block_for_row<'a>(
        blocks: &'a [PlacedConstraintBlock],
        row: usize,
    ) -> Result<(&'a PlacedConstraintBlock, usize), String> {
        let mut offset = 0usize;
        for block in blocks {
            let rows = block.set.nrows();
            if row < offset + rows {
                return Ok((block, row - offset));
            }
            offset += rows;
        }
        Err(format!(
            "ConstraintSet: row {row} out of range ({offset} rows)"
        ))
    }

    pub fn nrows(&self) -> usize {
        match self {
            ConstraintSet::Dense(dense) => dense.a.nrows(),
            ConstraintSet::KhatriRaoCone(cone) => cone.nrows(),
            ConstraintSet::BlockDiagonal { blocks, .. } => {
                blocks.iter().map(|block| block.set.nrows()).sum()
            }
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            ConstraintSet::Dense(dense) => dense.a.ncols(),
            ConstraintSet::KhatriRaoCone(cone) => cone.ncols(),
            ConstraintSet::BlockDiagonal { total_cols, .. } => *total_cols,
        }
    }

    /// Raw constraint values `Aβ` (dense) / factored functional values (cone).
    pub fn values(&self, beta: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        match self {
            ConstraintSet::Dense(dense) => {
                if beta.len() != dense.a.ncols() {
                    return Err(format!(
                        "ConstraintSet: beta length {} != {}",
                        beta.len(),
                        dense.a.ncols()
                    ));
                }
                Ok(dense.a.dot(&beta))
            }
            ConstraintSet::KhatriRaoCone(cone) => cone.values(beta),
            ConstraintSet::BlockDiagonal { blocks, total_cols } => {
                if beta.len() != *total_cols {
                    return Err(format!(
                        "ConstraintSet: beta length {} != {}",
                        beta.len(),
                        total_cols
                    ));
                }
                let mut out = Array1::<f64>::zeros(self.nrows());
                let mut offset = 0usize;
                for block in blocks {
                    let width = block.set.ncols();
                    let local = beta.slice(ndarray::s![block.col_start..block.col_start + width]);
                    let values = block.set.values(local)?;
                    let rows = values.len();
                    out.slice_mut(ndarray::s![offset..offset + rows])
                        .assign(&values);
                    offset += rows;
                }
                Ok(out)
            }
        }
    }

    /// Right-hand sides (`b` dense; cone bounds are zero unless delta-shifted).
    pub fn bound(&self, row: usize) -> Result<f64, String> {
        match self {
            ConstraintSet::Dense(dense) => dense.b.get(row).copied().ok_or_else(|| {
                format!(
                    "ConstraintSet: row {row} out of range ({} rows)",
                    dense.b.len()
                )
            }),
            ConstraintSet::KhatriRaoCone(cone) => cone.bound(row),
            ConstraintSet::BlockDiagonal { blocks, .. } => {
                let (block, local) = Self::block_for_row(blocks, row)?;
                block.set.bound(local)
            }
        }
    }

    pub fn row_norm(&self, row: usize) -> Result<f64, String> {
        match self {
            ConstraintSet::Dense(dense) => {
                if row >= dense.a.nrows() {
                    return Err(format!(
                        "ConstraintSet: row {row} out of range ({} rows)",
                        dense.a.nrows()
                    ));
                }
                let r = dense.a.row(row);
                Ok(r.dot(&r).sqrt())
            }
            ConstraintSet::KhatriRaoCone(cone) => cone.row_norm(row),
            ConstraintSet::BlockDiagonal { blocks, .. } => {
                let (block, local) = Self::block_for_row(blocks, row)?;
                block.set.row_norm(local)
            }
        }
    }

    /// The same constraint system expressed in delta coordinates around
    /// `beta`: `A(β + δ) ≥ b  ⇔  Aδ ≥ b − Aβ`. The matrix carrier is shared;
    /// only the `O(nrows)` bounds change.
    pub fn shifted_to_delta(&self, beta: ArrayView1<'_, f64>) -> Result<Self, String> {
        let values = self.values(beta)?;
        match self {
            ConstraintSet::Dense(dense) => Ok(ConstraintSet::Dense(
                LinearInequalityConstraints::new(dense.a.clone(), &dense.b - &values)?,
            )),
            ConstraintSet::KhatriRaoCone(cone) => {
                let mut shifted = cone.clone();
                let base = shifted
                    .bounds
                    .take()
                    .unwrap_or_else(|| Array1::zeros(values.len()));
                shifted.bounds = Some(&base - &values);
                Ok(ConstraintSet::KhatriRaoCone(shifted))
            }
            ConstraintSet::BlockDiagonal { blocks, total_cols } => {
                let mut shifted_blocks = Vec::with_capacity(blocks.len());
                for block in blocks {
                    let width = block.set.ncols();
                    let local = beta.slice(ndarray::s![block.col_start..block.col_start + width]);
                    shifted_blocks.push(PlacedConstraintBlock {
                        col_start: block.col_start,
                        set: block.set.shifted_to_delta(local)?,
                    });
                }
                Ok(ConstraintSet::BlockDiagonal {
                    blocks: shifted_blocks,
                    total_cols: *total_cols,
                })
            }
        }
    }

    /// Scaled violation sweep: `max_r (b_r − (Aβ)_r) / ‖a_r‖` restricted to
    /// non-vacuous rows, plus the arg-max row. Matches the canonicalized dense
    /// geometry (unit rows) without materializing it.
    ///
    /// This is THE feasibility metric: `β` is feasible exactly when the value
    /// returned here is at or below [`PRIMAL_FEASIBILITY_TOL`].
    ///
    /// A vacuous row (`‖a‖ = 0`) with a bound at or below zero is `0 ≥ b`, true
    /// for every `β`, and contributes nothing. A vacuous row with a POSITIVE
    /// bound is `0 ≥ b > 0`: no `β` satisfies it, so its violation is infinite
    /// and the feasible set is empty. Reporting that as `+∞` — rather than
    /// skipping the row — is what makes this metric agree with
    /// `ConstraintSetOps::scaled_slack`, which already answers `−∞` for exactly
    /// this row, and keeps a gate built on this metric from silently admitting
    /// an unsatisfiable system.
    ///
    /// A row that cannot be decided by comparison — a non-finite row norm,
    /// bound or `a·β` — is refused rather than skipped (gam#2721): feasibility
    /// of an iterate that is not a number is undefined, and `violation > worst`
    /// being false for `NaN` would report the neutral `0.0` — "nothing is
    /// violated" — for exactly the iterate this metric exists to catch.
    pub fn max_scaled_violation(
        &self,
        beta: ArrayView1<'_, f64>,
    ) -> Result<(f64, Option<usize>), String> {
        let values = self.values(beta)?;
        let mut worst = 0.0_f64;
        let mut worst_row = None;
        for (row, &value) in values.iter().enumerate() {
            let norm = self.row_norm(row)?;
            let bound = self.bound(row)?;
            // Decidability before comparison (gam#2721): `violation > worst` is
            // FALSE for `NaN`, so an undecidable row would leave `worst` at
            // `0.0` and this metric — THE feasibility verdict — would call an
            // iterate that is not a number feasible. `norm <= 0.0` is false for
            // a `NaN` norm too, so the vacuous-row branch below cannot be the
            // one that catches it. Refuse, naming the row and the quantities.
            if !feasibility_quantities_are_finite(&[norm, bound, value]) {
                return Err(format!(
                    "ConstraintSet::max_scaled_violation: row {row} cannot be decided \
                     (row norm {norm:.3e}, bound {bound:.3e}, value {value:.3e}); \
                     feasibility of a non-finite iterate is undefined and every \
                     comparison in the sweep is false for NaN, so the row cannot \
                     be skipped (gam#2721)"
                ));
            }
            if norm <= 0.0 {
                if bound > 0.0 {
                    return Ok((f64::INFINITY, Some(row)));
                }
                continue;
            }
            let violation = (bound - value) / norm;
            if violation > worst {
                worst = violation;
                worst_row = Some(row);
            }
        }
        Ok((worst, worst_row))
    }

    /// Largest `t ∈ [0, 1]` with `β + t·δ` feasible for every row, together
    /// with the first blocking row (the EXACT ratio test of a primal
    /// active-set method — zero tolerance, raw slacks). Rows already violated
    /// at `β` are reported as blocking at `t = 0`.
    ///
    /// This is the *pivot* rule: it answers "where does this chord cross a
    /// hyperplane in exact arithmetic", and its consumers (the feasible-chord
    /// clipper) want exactly that. It is NOT the rule for sizing a Newton step
    /// — a globalization that demands exact feasibility rejects steps this
    /// carrier's own contract calls feasible. Use
    /// [`ConstraintSet::max_contract_feasible_step`] for that.
    ///
    /// Like the contract rule, this one is TOTAL (gam#2721): a row that cannot
    /// be decided by comparison — a non-finite row norm, bound, `a·β` or `a·δ`
    /// — and that was not explicitly skipped is refused, because every
    /// comparison it would otherwise feed is false for `NaN` and the answer
    /// would be an unlimited `t = 1`.
    pub fn max_feasible_step(
        &self,
        beta: ArrayView1<'_, f64>,
        delta: ArrayView1<'_, f64>,
        skip_rows: &[usize],
    ) -> Result<(f64, Option<usize>), String> {
        let values = self.values(beta)?;
        let directional = self.values(delta)?;
        let mut skip = vec![false; values.len()];
        for &row in skip_rows {
            if row < skip.len() {
                skip[row] = true;
            }
        }
        let mut step = 1.0_f64;
        let mut blocking = None;
        for row in 0..values.len() {
            if skip[row] {
                continue;
            }
            let norm = self.row_norm(row)?;
            let bound = self.bound(row)?;
            let value = values[row];
            let rate = directional[row];
            // Same decidability requirement as the contract rule (gam#2721): a
            // `NaN` fails `rate >= 0.0` AND `t < step`, so the row would be
            // skipped twice over and this exact ratio test would answer
            // `step = 1.0` — "the whole chord is feasible" — for a chord that
            // is not a point. The clipper built on it would then accept the
            // endpoint. Refuse before comparing.
            if !feasibility_quantities_are_finite(&[norm, bound, value, rate]) {
                return Err(format!(
                    "ConstraintSet::max_feasible_step: row {row} cannot be decided \
                     (row norm {norm:.3e}, bound {bound:.3e}, value {value:.3e}, \
                     drift {rate:.3e}); every comparison in the ratio test is false \
                     for NaN, so skipping the row would report the whole step \
                     feasible (gam#2721)"
                ));
            }
            if norm <= 0.0 {
                continue;
            }
            if rate >= 0.0 {
                continue;
            }
            let t = (value - bound) / (-rate);
            if t < step {
                step = t.max(0.0);
                blocking = Some(row);
            }
        }
        Ok((step, blocking))
    }

    /// Materialize the requested rows densely (KKT systems on the active set).
    pub fn gather_rows(&self, rows: &[usize]) -> Result<LinearInequalityConstraints, String> {
        match self {
            ConstraintSet::Dense(dense) => {
                let mut a = Array2::<f64>::zeros((rows.len(), dense.a.ncols()));
                let mut b = Array1::<f64>::zeros(rows.len());
                for (out_row, &row) in rows.iter().enumerate() {
                    if row >= dense.a.nrows() {
                        return Err(format!(
                            "ConstraintSet: row {row} out of range ({} rows)",
                            dense.a.nrows()
                        ));
                    }
                    a.row_mut(out_row).assign(&dense.a.row(row));
                    b[out_row] = dense.b[row];
                }
                LinearInequalityConstraints::new(a, b)
            }
            ConstraintSet::KhatriRaoCone(cone) => cone.gather_rows(rows),
            ConstraintSet::BlockDiagonal { blocks, total_cols } => {
                let mut a = Array2::<f64>::zeros((rows.len(), *total_cols));
                let mut b = Array1::<f64>::zeros(rows.len());
                for (out_row, &row) in rows.iter().enumerate() {
                    let (block, local) = Self::block_for_row(blocks, row)?;
                    let gathered = block.set.gather_rows(&[local])?;
                    a.row_mut(out_row)
                        .slice_mut(ndarray::s![
                            block.col_start..block.col_start + block.set.ncols()
                        ])
                        .assign(&gathered.a.row(0));
                    b[out_row] = gathered.b[0];
                }
                LinearInequalityConstraints::new(a, b)
            }
        }
    }

    /// Exact dense equivalent of the whole set (tests / small systems only).
    pub fn to_dense(&self) -> Result<LinearInequalityConstraints, String> {
        match self {
            ConstraintSet::Dense(dense) => Ok(dense.clone()),
            _ => {
                let all: Vec<usize> = (0..self.nrows()).collect();
                self.gather_rows(&all)
            }
        }
    }
}

impl From<LinearInequalityConstraints> for ConstraintSet {
    fn from(dense: LinearInequalityConstraints) -> Self {
        ConstraintSet::Dense(dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn cone_fixture() -> KhatriRaoConeConstraints {
        // Ψ: 3 observations × 2 covariate columns; A is 3 coefficient rows
        // (row 0 = location, rows 1..2 = shape) × 2 columns.
        let psi = array![[1.0_f64, 0.5], [2.0, -1.0], [0.0, 3.0]];
        KhatriRaoConeConstraints::new(Arc::new(psi), vec![1, 2], 3).expect("cone fixture")
    }

    fn beta_fixture() -> Array1<f64> {
        // vec(A) row-major, A = [[9, -4], [1, 2], [0.5, -0.25]]
        array![9.0_f64, -4.0, 1.0, 2.0, 0.5, -0.25]
    }

    #[test]
    fn cone_values_match_dense_system() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = ConstraintSet::Dense(cone.to_dense().expect("dense"));
        let beta = beta_fixture();
        let via_cone = set.values(beta.view()).expect("cone values");
        let via_dense = dense.values(beta.view()).expect("dense values");
        assert_eq!(via_cone.len(), 6);
        for (a, b) in via_cone.iter().zip(via_dense.iter()) {
            assert!((a - b).abs() < 1e-14, "cone/dense mismatch: {a} vs {b}");
        }
        // Spot-check one functional exactly: slot 0 (A row 1), observation 1:
        // ψ = (2, −1), A_{1,:} = (1, 2) → 2·1 − 1·2 = 0.
        assert!((via_cone[1] - 0.0).abs() < 1e-15);
    }

    /// `row_column_support` is the sanctioned row → β conversion, so it must
    /// agree with the explicit dense system row by row: the columns it names are
    /// exactly the structurally nonzero entries of that row of `A`.
    #[test]
    fn cone_row_column_support_matches_the_dense_row_nonzeros() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = ConstraintSet::Dense(cone.to_dense().expect("dense"));
        for row in 0..set.nrows() {
            let via_cone = set
                .row_column_support(ConstraintRowId(row))
                .expect("cone support");
            let via_dense = dense
                .row_column_support(ConstraintRowId(row))
                .expect("dense support");
            assert_eq!(via_cone, via_dense, "row {row} support mismatch");
        }
        // Slot 0 carries coefficient row k = 1, so its columns are 1·p_cov + j.
        // Observation 2 has ψ = (0, 3): the zero factor entry drops column 2.
        assert_eq!(
            set.row_column_support(ConstraintRowId(0)).expect("r0"),
            vec![2, 3]
        );
        assert_eq!(
            set.row_column_support(ConstraintRowId(2)).expect("r2"),
            vec![3]
        );
        // Slot 1 carries coefficient row k = 2 → columns 4, 5.
        assert_eq!(
            set.row_column_support(ConstraintRowId(3)).expect("r3"),
            vec![4, 5]
        );
    }

    /// The block-diagonal arm offsets support by `col_start` while it decodes
    /// the row by the running `nrows()`. When a member has `nrows() < ncols()`
    /// the two run at different rates, and only the conversion tracks columns
    /// correctly: joint row 1 belongs to the block starting at column 3.
    #[test]
    fn block_diagonal_row_column_support_uses_col_start_not_the_row_offset() {
        let narrow = PlacedConstraintBlock {
            col_start: 0,
            set: ConstraintSet::Dense(
                LinearInequalityConstraints::new(
                    array![[1.0_f64, 0.0, 0.0]],
                    Array1::<f64>::zeros(1),
                )
                .expect("narrow"),
            ),
        };
        let square = PlacedConstraintBlock {
            col_start: 3,
            set: ConstraintSet::Dense(
                LinearInequalityConstraints::new(
                    array![[1.0_f64, 0.0], [0.0, 1.0]],
                    Array1::<f64>::zeros(2),
                )
                .expect("square"),
            ),
        };
        let set = ConstraintSet::block_diagonal(vec![narrow, square], 5).expect("joint");
        assert_eq!(set.nrows(), 3);
        assert_eq!(set.ncols(), 5);
        assert_eq!(
            set.row_column_support(ConstraintRowId(0)).expect("r0"),
            vec![0]
        );
        // Row 1 is the second block's first row: column 3, NOT column 1.
        assert_eq!(
            set.row_column_support(ConstraintRowId(1)).expect("r1"),
            vec![3]
        );
        assert_eq!(
            set.row_column_support(ConstraintRowId(2)).expect("r2"),
            vec![4]
        );
        assert!(set.row_column_support(ConstraintRowId(3)).is_err());
    }

    #[test]
    fn cone_row_norms_are_factor_row_norms_for_every_slot() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone);
        let expected = [(1.0_f64 + 0.25).sqrt(), (4.0_f64 + 1.0).sqrt(), 3.0_f64];
        for slot in 0..2 {
            for i in 0..3 {
                let norm = set.row_norm(slot * 3 + i).expect("norm");
                assert!((norm - expected[i]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn max_scaled_violation_agrees_with_canonicalized_dense() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let beta = beta_fixture();
        let (violation, row) = set.max_scaled_violation(beta.view()).expect("violation");
        // Dense oracle: canonicalize, then measure b − Aβ on unit rows.
        let dense = cone
            .to_dense()
            .expect("dense")
            .canonicalized()
            .expect("canon");
        let values = dense.a.dot(&beta);
        let mut worst = 0.0_f64;
        let mut worst_row = None;
        for r in 0..values.len() {
            let v = dense.b[r] - values[r];
            if v > worst {
                worst = v;
                worst_row = Some(r);
            }
        }
        assert!((violation - worst).abs() < 1e-14);
        assert_eq!(row, worst_row);
        assert!(violation > 0.0, "fixture must have a violated row");
    }

    #[test]
    fn max_feasible_step_matches_scalar_ratio_test() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone);
        // Feasible start: shape rows of A strictly positive functionals.
        // A = [[0, 0], [1, 0.1], [1, 0.1]] → α values Ψ·(1, 0.1):
        // (1.05, 1.9, 0.3) — all positive for both slots.
        let beta = array![0.0_f64, 0.0, 1.0, 0.1, 1.0, 0.1];
        // Direction pushing slot 0 observation 2 down: δA_{1,:} = (0, −1) →
        // rate = ψ_2 · (0, −1) = −3; slack = 0.3 → t = 0.1. All other rows
        // untouched (rate 0 for slot 1, rates −0.5/1 for slot 0 rows 0/1:
        // row 0 rate = ψ_0·(0,−1) = −0.5, slack 1.05 → t = 2.1).
        let delta = array![0.0_f64, 0.0, 0.0, -1.0, 0.0, 0.0];
        let (step, blocking) = set
            .max_feasible_step(beta.view(), delta.view(), &[])
            .expect("step");
        assert!((step - 0.1).abs() < 1e-14, "expected 0.1, got {step}");
        assert_eq!(blocking, Some(2));
        // Skipping the blocking row exposes the next ratio (row 0, t = 2.1 → clamped to 1).
        let (step_skipped, blocking_skipped) = set
            .max_feasible_step(beta.view(), delta.view(), &[2])
            .expect("step skipped");
        assert!((step_skipped - 1.0).abs() < 1e-14);
        assert_eq!(blocking_skipped, None);
    }

    #[test]
    fn gather_rows_places_factor_rows_in_the_coupled_slot() {
        let cone = cone_fixture();
        // Row id 4 = slot 1 (A row 2), observation 1 → ψ = (2, −1) in cols 4..6.
        let gathered = cone.gather_rows(&[4]).expect("gather");
        assert_eq!(gathered.a.nrows(), 1);
        assert_eq!(gathered.a.ncols(), 6);
        let expected = [0.0, 0.0, 0.0, 0.0, 2.0, -1.0];
        for (j, &e) in expected.iter().enumerate() {
            assert_eq!(gathered.a[[0, j]], e);
        }
        assert_eq!(gathered.b[0], 0.0);
    }

    #[test]
    fn constructor_rejects_bad_coupled_rows() {
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(KhatriRaoConeConstraints::new(Arc::new(psi.clone()), vec![3], 3).is_err());
        assert!(KhatriRaoConeConstraints::new(Arc::new(psi.clone()), vec![1, 1], 3).is_err());
        assert!(KhatriRaoConeConstraints::new(Arc::new(psi), vec![], 3).is_err());
    }

    #[test]
    fn shifted_to_delta_matches_dense_shift() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone);
        let beta = beta_fixture();
        let shifted = set.shifted_to_delta(beta.view()).expect("shift");
        // Oracle: dense shift b' = b − Aβ.
        let dense = set.to_dense().expect("dense");
        let expected_b = &dense.b - &dense.a.dot(&beta);
        for row in 0..set.nrows() {
            assert!(
                (shifted.bound(row).expect("bound") - expected_b[row]).abs() < 1e-14,
                "shifted bound mismatch at row {row}"
            );
        }
        // The delta system at δ = 0 has slack equal to the original at β.
        let zero = Array1::<f64>::zeros(set.ncols());
        let (viol_delta, row_delta) = shifted
            .max_scaled_violation(zero.view())
            .expect("delta violation");
        let (viol_orig, row_orig) = set.max_scaled_violation(beta.view()).expect("violation");
        assert!((viol_delta - viol_orig).abs() < 1e-14);
        assert_eq!(row_delta, row_orig);
    }

    #[test]
    fn block_diagonal_composes_ids_bounds_and_values() {
        // Block 0: dense 2-row system on columns 0..2; block 1: cone on 2..8.
        let dense = LinearInequalityConstraints::new(
            array![[1.0_f64, 0.0], [0.0, -2.0]],
            array![0.5_f64, -1.0],
        )
        .expect("dense block");
        let cone = cone_fixture();
        let joint = ConstraintSet::block_diagonal(
            vec![
                PlacedConstraintBlock {
                    col_start: 0,
                    set: ConstraintSet::Dense(dense.clone()),
                },
                PlacedConstraintBlock {
                    col_start: 2,
                    set: ConstraintSet::KhatriRaoCone(cone.clone()),
                },
            ],
            8,
        )
        .expect("joint");
        assert_eq!(joint.nrows(), 2 + 6);
        assert_eq!(joint.ncols(), 8);
        let mut beta = Array1::<f64>::zeros(8);
        beta[0] = 2.0;
        beta[1] = 1.0;
        beta.slice_mut(ndarray::s![2..8]).assign(&beta_fixture());
        let values = joint.values(beta.view()).expect("values");
        assert!((values[0] - 2.0).abs() < 1e-15);
        assert!((values[1] + 2.0).abs() < 1e-15);
        let cone_values = cone.values(beta_fixture().view()).expect("cone values");
        for (idx, &cv) in cone_values.iter().enumerate() {
            assert!((values[2 + idx] - cv).abs() < 1e-15);
        }
        assert_eq!(joint.bound(0).expect("b0"), 0.5);
        assert_eq!(joint.bound(2).expect("b2"), 0.0);
        // Gathered joint row 3 (= cone row 1) occupies columns 2 + [2..4).
        let gathered = joint.gather_rows(&[3]).expect("gather");
        assert_eq!(gathered.a.ncols(), 8);
        assert_eq!(gathered.a[[0, 4]], 2.0);
        assert_eq!(gathered.a[[0, 5]], -1.0);
        // Overlapping ranges are rejected.
        assert!(
            ConstraintSet::block_diagonal(
                vec![
                    PlacedConstraintBlock {
                        col_start: 0,
                        set: ConstraintSet::Dense(dense.clone()),
                    },
                    PlacedConstraintBlock {
                        col_start: 1,
                        set: ConstraintSet::Dense(dense),
                    },
                ],
                8,
            )
            .is_err()
        );
    }

    #[test]
    fn zero_factor_rows_are_vacuous_not_violations() {
        // Ψ with an all-zero observation row: 0ᵀβ ≥ 0 is vacuous and must be
        // skipped by violation and ratio sweeps (norm 0), matching the dense
        // canonicalization contract for zero rows with b ≤ 0.
        let psi = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let cone = KhatriRaoConeConstraints::new(Arc::new(psi), vec![1], 2).expect("cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let beta = array![0.0_f64, 0.0, -5.0, 4.0];
        // Slot 0: values (0, −1). Row 0 vacuous; row 1 violated by 1/√2.
        let (violation, row) = set.max_scaled_violation(beta.view()).expect("violation");
        assert_eq!(row, Some(1));
        assert!((violation - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
    }

    /// `β ≥ 0` on two coordinates, expressed with a deliberately non-unit row
    /// so the scaled/raw distinction is observable.
    fn scaled_box() -> ConstraintSet {
        // Row 0: 1e-3·β₀ ≥ 0 (‖a‖ = 1e-3). Row 1: β₁ ≥ 0 (‖a‖ = 1).
        ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0e-3_f64, 0.0], [0.0, 1.0]],
                Array1::<f64>::zeros(2),
            )
            .expect("scaled box"),
        )
    }

    /// gam#2719, the headline: at a coordinate sitting EXACTLY on its bound, a
    /// drift far below the feasibility contract must not crush the step. The
    /// exact ratio test answers 0 (its numerator is the exact slack); the
    /// contract ratio test answers 1, because the endpoint of the full step is
    /// a point this very carrier calls feasible.
    #[test]
    fn a_sub_tolerance_drift_off_an_active_row_does_not_limit_the_step() {
        let set = scaled_box();
        let beta = array![0.0_f64, 0.0];
        let direction = array![1.0_f64, -1.0e-15];

        let (exact, blocking) = set
            .max_feasible_step(beta.view(), direction.view(), &[])
            .expect("exact ratio test");
        assert_eq!(exact, 0.0, "the exact rule crushes the step");
        assert_eq!(blocking, Some(1));

        let contract = set
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("contract ratio test");
        assert_eq!(contract.fraction, 1.0);
        assert_eq!(contract.blocking_row, None);

        // And the claim the relief rests on: the endpoint really is feasible.
        let endpoint = &beta + &direction;
        let (violation, _) = set
            .max_scaled_violation(endpoint.view())
            .expect("endpoint violation");
        assert!(violation <= PRIMAL_FEASIBILITY_TOL);
    }

    /// The relief is bounded by the contract, not open-ended: a drift one
    /// order ABOVE the tolerance still blocks, and blocks at the true
    /// boundary (fraction 0 here, since the slack is exactly 0).
    #[test]
    fn a_drift_above_the_contract_still_blocks_at_the_true_boundary() {
        let set = scaled_box();
        let beta = array![0.0_f64, 0.0];
        let contract = set
            .max_contract_feasible_step(beta.view(), array![0.0_f64, -1.0e-7].view())
            .expect("contract ratio test");
        assert_eq!(contract.fraction, 0.0);
        assert_eq!(contract.blocking_row, Some(1));
        assert!(contract.is_blocked_by_active_face());
    }

    /// A healthy interior step is limited exactly where the exact rule limits
    /// it: the contract clause only ever fires on sub-tolerance excursions, so
    /// ordinary fraction-to-boundary behaviour is untouched.
    #[test]
    fn an_interior_iterate_gets_the_ordinary_fraction_to_boundary() {
        let set = scaled_box();
        let beta = array![1.0_f64, 0.25];
        let direction = array![0.0_f64, -1.0];
        let contract = set
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("contract ratio test");
        assert_eq!(contract.blocking_row, Some(1));
        assert!((contract.fraction - 0.25).abs() < 1e-15);
        let (exact, _) = set
            .max_feasible_step(beta.view(), direction.view(), &[])
            .expect("exact ratio test");
        assert!((contract.fraction - exact).abs() < 1e-15);
    }

    /// gam#2721, asked of all three rules at once: hand each of them the
    /// MAXIMALLY bad argument — every component `NaN` — and require a refusal.
    ///
    /// Each arm carries a positive control on the same fixture first, so a
    /// green cannot come from the fixture never reaching the rule: the finite
    /// direction must be evaluated and clipped, and the finite violating
    /// iterate must be measured as violating.
    #[test]
    fn every_feasibility_rule_refuses_an_argument_that_is_not_a_number() {
        let set = scaled_box();
        let interior = array![1.0_f64, 1.0];
        let nan = array![f64::NAN, f64::NAN];

        // --- the contract ratio test ---
        let clipped = set
            .max_contract_feasible_step(interior.view(), array![0.0_f64, -2.0].view())
            .expect("a finite binding direction must be evaluated");
        assert!(
            clipped.fraction > 0.0 && clipped.fraction < 1.0,
            "positive control: a binding finite direction must clip, got {}",
            clipped.fraction
        );
        match set.max_contract_feasible_step(interior.view(), nan.view()) {
            Err(ContractFeasibleStepError::NonFinite { row, .. }) => {
                assert_eq!(row, 0, "the refusal must name the offending row");
            }
            other => panic!("a NaN step must be refused, got {other:?}"),
        }

        // --- the exact ratio test (the feasible-chord clipper's rule) ---
        let (exact, blocking) = set
            .max_feasible_step(interior.view(), array![0.0_f64, -2.0].view(), &[])
            .expect("a finite binding direction must be evaluated");
        assert!(
            exact > 0.0 && exact < 1.0,
            "positive control: a binding finite direction must clip, got {exact}"
        );
        assert_eq!(blocking, Some(1));
        let refusal = set
            .max_feasible_step(interior.view(), nan.view(), &[])
            .expect_err("a NaN chord must be refused, not reported as fully feasible");
        assert!(
            refusal.contains("max_feasible_step") && refusal.contains("cannot be decided"),
            "the refusal must name the rule and the condition, got: {refusal}"
        );

        // --- the violation sweep (THE feasibility verdict on an iterate) ---
        let (violation, row) = set
            .max_scaled_violation(array![-1.0_f64, -0.5].view())
            .expect("a finite iterate must be measured");
        assert!(
            violation > 0.0 && row.is_some(),
            "positive control: a violating finite iterate must be seen as violating"
        );
        let refusal = set
            .max_scaled_violation(nan.view())
            .expect_err("a NaN iterate must be refused, not reported as feasible");
        assert!(
            refusal.contains("max_scaled_violation") && refusal.contains("cannot be decided"),
            "the refusal must name the rule and the condition, got: {refusal}"
        );
    }

    /// The disposition the refusal must NOT swallow: a VACUOUS row (`‖a‖ = 0`
    /// with a non-positive bound) is `0 ≥ b`, true for every `β`. It is
    /// perfectly decidable and each rule already has its own answer for it —
    /// skip. A refusal placed at the wrong point, or written over the wrong
    /// quantity, turns that row into a hard error, so the `NaN` arms above are
    /// paired with this one.
    #[test]
    fn a_vacuous_row_keeps_its_own_disposition_and_is_not_refused() {
        // Row 0: 0ᵀβ ≥ 0 (vacuous). Row 1: β₁ ≥ 0.
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[0.0_f64, 0.0], [0.0, 1.0]],
                Array1::<f64>::zeros(2),
            )
            .expect("vacuous-row system"),
        );
        let beta = array![-1.0e9_f64, 1.0];

        let (violation, row) = set
            .max_scaled_violation(beta.view())
            .expect("a vacuous row must be decided, not refused");
        assert_eq!(violation, 0.0, "a vacuous row cannot be violated");
        assert_eq!(row, None);

        // The vacuous row must not limit the step either; the real one still
        // does, at its exact ratio.
        let direction = array![-1.0e12_f64, -2.0];
        let (exact, blocking) = set
            .max_feasible_step(beta.view(), direction.view(), &[])
            .expect("a vacuous row must be decided, not refused");
        assert!((exact - 0.5).abs() < 1e-15, "expected 0.5, got {exact}");
        assert_eq!(blocking, Some(1));

        let contract = set
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("a vacuous row must be decided, not refused");
        assert!((contract.fraction - 0.5).abs() < 1e-15);
        assert_eq!(contract.blocking_row, Some(1));
    }

    /// The tolerance is a SCALED one. Row 0 has ‖a‖ = 1e-3, so a raw drift of
    /// −1e-8 on β₀ is a scaled drift of −1e-8 · 1e-3 / 1e-3 = −1e-8 · … — the
    /// point being that the rule must divide by ‖a‖ on BOTH slack and drift,
    /// or the same geometric step gets different verdicts on differently
    /// normalized rows. Two carriers that differ only by a positive per-row
    /// rescaling must give the identical fraction.
    #[test]
    fn the_fraction_is_invariant_to_per_row_rescaling() {
        let unit = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, 1.0]],
                Array1::<f64>::zeros(2),
            )
            .expect("unit box"),
        );
        let beta = array![0.5_f64, 3.0];
        let direction = array![-1.0_f64, 0.25];
        let scaled = scaled_box();
        let a = unit
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("unit");
        let b = scaled
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("scaled");
        assert_eq!(a.blocking_row, b.blocking_row);
        assert!((a.fraction - b.fraction).abs() < 1e-15);
        assert!((a.fraction - 0.5).abs() < 1e-15);
    }

    /// A round-off-negative slack inside the band is AT the boundary, not a
    /// violation: the iterate is accepted and the step is limited at the true
    /// boundary (fraction 0), never reported as an infeasible iterate.
    #[test]
    fn an_in_band_negative_slack_is_a_boundary_not_an_infeasible_iterate() {
        let set = scaled_box();
        let beta = array![0.0_f64, -1.0e-9];
        let contract = set
            .max_contract_feasible_step(beta.view(), array![0.0_f64, -1.0].view())
            .expect("in-band slack is feasible");
        assert_eq!(contract.fraction, 0.0);
        assert_eq!(contract.blocking_row, Some(1));

        // One order out of the band IS an infeasible iterate, and stays loud.
        let outside = array![0.0_f64, -1.0e-7];
        match set.max_contract_feasible_step(outside.view(), array![0.0_f64, 1.0].view()) {
            Err(ContractFeasibleStepError::InfeasibleIterate { row, scaled_slack }) => {
                assert_eq!(row, 1);
                assert!((scaled_slack + 1.0e-7).abs() < 1e-20);
            }
            other => panic!("expected an infeasible-iterate report, got {other:?}"),
        }
    }

    /// Repeated full steps whose excursion is individually sub-tolerance
    /// cannot walk the iterate past the contract: the admitted violation is
    /// bounded by the tolerance for any number of steps.
    #[test]
    fn sub_tolerance_relief_cannot_accumulate_past_the_contract() {
        let set = scaled_box();
        let direction = array![0.0_f64, -2.0e-9];
        let mut beta = array![0.0_f64, 0.0];
        for step in 0..64 {
            let contract = set
                .max_contract_feasible_step(beta.view(), direction.view())
                .unwrap_or_else(|e| panic!("step {step} must keep a feasible origin: {e}"));
            beta = &beta + &(&direction * contract.fraction);
            let (violation, _) = set
                .max_scaled_violation(beta.view())
                .expect("violation sweep");
            assert!(
                violation <= PRIMAL_FEASIBILITY_TOL,
                "step {step} left scaled violation {violation:.3e} outside the contract"
            );
        }
    }

    /// The factored cone answers identically to its dense materialization —
    /// the ratio test must not be a dense-only rule.
    #[test]
    fn cone_and_dense_agree_on_the_contract_fraction() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = ConstraintSet::Dense(cone.to_dense().expect("dense"));
        // A_{1,:} = A_{2,:} = (1, 1) makes every cone functional positive, so
        // the ratio test has a feasible origin; the direction pulls A_{1,:}
        // toward the ψ₁ = (2, −1) face, which binds first at t = 1/2.
        let beta = array![9.0_f64, -4.0, 1.0, 1.0, 1.0, 1.0];
        let direction = array![0.0_f64, 0.0, -1.0, 0.0, 0.0, 0.0];
        let via_cone = set
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("cone");
        let via_dense = dense
            .max_contract_feasible_step(beta.view(), direction.view())
            .expect("dense");
        assert_eq!(via_cone.blocking_row, via_dense.blocking_row);
        assert!((via_cone.fraction - via_dense.fraction).abs() < 1e-14);
        assert_eq!(via_cone.blocking_row, Some(1));
        assert!((via_cone.fraction - 0.5).abs() < 1e-14);
    }

    /// A vacuous row with a positive bound is an empty feasible set; no step
    /// fraction exists and the ratio test says so instead of returning 1.
    #[test]
    fn a_vacuous_row_with_a_positive_bound_has_no_feasible_origin() {
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[0.0_f64, 0.0]], array![1.0_f64])
                .expect("vacuous row"),
        );
        match set.max_contract_feasible_step(array![1.0_f64, 1.0].view(), array![1.0_f64, 0.0].view())
        {
            Err(ContractFeasibleStepError::InfeasibleIterate { row, scaled_slack }) => {
                assert_eq!(row, 0);
                assert_eq!(scaled_slack, f64::NEG_INFINITY);
            }
            other => panic!("expected an empty-feasible-set report, got {other:?}"),
        }
    }

    #[test]
    fn width_mismatches_are_reported_before_any_arithmetic() {
        let set = scaled_box();
        match set.max_contract_feasible_step(array![0.0_f64].view(), array![0.0_f64, 0.0].view()) {
            Err(ContractFeasibleStepError::Dimension {
                beta,
                direction,
                expected,
            }) => {
                assert_eq!((beta, direction, expected), (1, 2, 2));
            }
            other => panic!("expected a dimension report, got {other:?}"),
        }
    }
}
