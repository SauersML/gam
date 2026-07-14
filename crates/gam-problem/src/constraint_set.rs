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
    factor: Array2<f64>,
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
        factor: Array2<f64>,
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
        let factor_row_norms = Array1::from_iter(
            factor
                .rows()
                .into_iter()
                .map(|row| row.dot(&row).sqrt()),
        );
        Ok(Self {
            factor,
            factor_row_norms,
            coupled_rows,
            p_left,
            bounds: None,
        })
    }

    pub fn factor(&self) -> &Array2<f64> {
        &self.factor
    }

    pub fn coupled_rows(&self) -> &[usize] {
        &self.coupled_rows
    }

    pub fn p_left(&self) -> usize {
        self.p_left
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
                    let local = beta.slice(ndarray::s![
                        block.col_start..block.col_start + width
                    ]);
                    let values = block.set.values(local)?;
                    let rows = values.len();
                    out.slice_mut(ndarray::s![offset..offset + rows]).assign(&values);
                    offset += rows;
                }
                Ok(out)
            }
        }
    }

    /// Right-hand sides (`b` dense; cone bounds are zero unless delta-shifted).
    pub fn bound(&self, row: usize) -> Result<f64, String> {
        match self {
            ConstraintSet::Dense(dense) => {
                dense.b.get(row).copied().ok_or_else(|| {
                    format!(
                        "ConstraintSet: row {row} out of range ({} rows)",
                        dense.b.len()
                    )
                })
            }
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
                    let local = beta.slice(ndarray::s![
                        block.col_start..block.col_start + width
                    ]);
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

    /// Scaled violation sweep: `max_r (b_r − (Aβ)_r) / max(‖a_r‖, 1)` restricted
    /// to non-vacuous rows, plus the arg-max row. Matches the canonicalized
    /// dense geometry (unit rows) without materializing it.
    pub fn max_scaled_violation(
        &self,
        beta: ArrayView1<'_, f64>,
    ) -> Result<(f64, Option<usize>), String> {
        let values = self.values(beta)?;
        let mut worst = 0.0_f64;
        let mut worst_row = None;
        for (row, &value) in values.iter().enumerate() {
            let norm = self.row_norm(row)?;
            if norm <= 0.0 {
                continue;
            }
            let violation = (self.bound(row)? - value) / norm;
            if violation > worst {
                worst = violation;
                worst_row = Some(row);
            }
        }
        Ok((worst, worst_row))
    }

    /// Largest `t ∈ [0, 1]` with `β + t·δ` feasible for every row, together
    /// with the first blocking row (the exact ratio test of a primal
    /// active-set method). Rows already violated at `β` (beyond `tol` in
    /// scaled units) are reported as blocking at `t = 0`.
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
            if norm <= 0.0 {
                continue;
            }
            let slack = values[row] - self.bound(row)?;
            let rate = directional[row];
            if rate >= 0.0 {
                continue;
            }
            let t = slack / (-rate);
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
        KhatriRaoConeConstraints::new(psi, vec![1, 2], 3).expect("cone fixture")
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

    #[test]
    fn cone_row_norms_are_factor_row_norms_for_every_slot() {
        let cone = cone_fixture();
        let set = ConstraintSet::KhatriRaoCone(cone);
        let expected = [
            (1.0_f64 + 0.25).sqrt(),
            (4.0_f64 + 1.0).sqrt(),
            3.0_f64,
        ];
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
        let dense = cone.to_dense().expect("dense").canonicalized().expect("canon");
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
        assert!(KhatriRaoConeConstraints::new(psi.clone(), vec![3], 3).is_err());
        assert!(KhatriRaoConeConstraints::new(psi.clone(), vec![1, 1], 3).is_err());
        assert!(KhatriRaoConeConstraints::new(psi, vec![], 3).is_err());
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
        let cone = KhatriRaoConeConstraints::new(psi, vec![1], 2).expect("cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let beta = array![0.0_f64, 0.0, -5.0, 4.0];
        // Slot 0: values (0, −1). Row 0 vacuous; row 1 violated by 1/√2.
        let (violation, row) = set.max_scaled_violation(beta.view()).expect("violation");
        assert_eq!(row, Some(1));
        assert!((violation - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
    }
}
