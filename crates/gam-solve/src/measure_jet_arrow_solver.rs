//! Whitened arrow solver for the measure-jet frame system.
//!
//! This module implements the solver described in
//! `docs/measure_jet_frame.md` section 3 for the whitened matrix
//! `M = I + D^{1/2} S^T W S D^{1/2}`.  The matrix is block-arrow shaped:
//! a dense polynomial/parametric head and a block-diagonal multilevel tail,
//! with each tail level coupled only to the head.  One arrow Cholesky
//! factorization delivers exact fits, log determinants, selected inverse trace
//! terms, and posterior draws.  A BPX-style additive block preconditioner is
//! also provided for the preconditioned-CG fallback.

use gam_linalg::triangular::{
    CholeskyGuard, back_substitution_lower_transpose_guarded_into, cholesky_factor_in_place,
    cholesky_solve_vector, forward_substitution_lower_matrix,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal as RandNormal};
use std::error::Error;
use std::fmt;

/// Errors reported while validating, factoring, or iterating the arrow system.
#[derive(Debug)]
pub enum MeasureJetArrowError {
    /// A per-level tail block failed strict finite-SPD Cholesky factorization.
    LevelFactorNotSpd { level: usize },
    /// The reduced dense head Schur complement failed strict finite-SPD Cholesky factorization.
    HeadSchurNotSpd,
    /// A matrix or vector dimension did not match the expected arrow layout.
    DimensionMismatch {
        what: String,
        expected: usize,
        got: usize,
    },
    /// A matrix, vector, tolerance, or residual contained a non-finite value.
    NonFinite { what: String },
    /// Preconditioned CG exhausted the requested iteration budget.
    CgDidNotConverge { iters: usize, residual: f64 },
}

impl fmt::Display for MeasureJetArrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LevelFactorNotSpd { level } => {
                write!(f, "level {level} tail block is not finite strict SPD")
            }
            Self::HeadSchurNotSpd => write!(f, "head Schur complement is not finite strict SPD"),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "{what} dimension mismatch: expected {expected}, got {got}"
            ),
            Self::NonFinite { what } => write!(f, "{what} contains a non-finite value"),
            Self::CgDidNotConverge { iters, residual } => write!(
                f,
                "preconditioned CG did not converge after {iters} iterations; residual {residual}"
            ),
        }
    }
}

impl Error for MeasureJetArrowError {}

/// One level of the whitened arrow tail.
#[derive(Clone, Debug)]
pub struct MeasureJetLevelBlock {
    /// The level diagonal block `M_ll`, including the whitened identity contribution.
    pub m_ll: Array2<f64>,
    /// The level-to-head coupling block `M_lh` with shape `d_l x k`.
    pub m_lh: Array2<f64>,
}

/// Assembled dense-head plus per-level-tail arrow blocks for the whitened matrix.
#[derive(Clone, Debug)]
pub struct MeasureJetArrowBlocks {
    /// Dense head block `M_hh` for the unpenalized polynomial/parametric border.
    pub m_hh: Array2<f64>,
    /// Head dimension `k`.
    pub head_dim: usize,
    /// Per-level block-diagonal tail blocks and their head couplings.
    pub levels: Vec<MeasureJetLevelBlock>,
}

impl MeasureJetArrowBlocks {
    /// Validate the arrow shapes and finite entries before strict-SPD factorization.
    pub fn validate(&self) -> Result<(), MeasureJetArrowError> {
        ensure_square("m_hh", self.m_hh.view())?;
        ensure_dimension("m_hh rows", self.head_dim, self.m_hh.nrows())?;
        ensure_dimension("m_hh cols", self.head_dim, self.m_hh.ncols())?;
        ensure_finite_matrix("m_hh", self.m_hh.view())?;

        for (level, block) in self.levels.iter().enumerate() {
            let level_name = format!("level {level} m_ll");
            ensure_square(&level_name, block.m_ll.view())?;
            ensure_finite_matrix(&level_name, block.m_ll.view())?;

            let coupling_name = format!("level {level} m_lh");
            ensure_dimension(
                &format!("{coupling_name} rows"),
                block.m_ll.nrows(),
                block.m_lh.nrows(),
            )?;
            ensure_dimension(
                &format!("{coupling_name} cols"),
                self.head_dim,
                block.m_lh.ncols(),
            )?;
            ensure_finite_matrix(&coupling_name, block.m_lh.view())?;
        }

        Ok(())
    }

    /// Assemble the full symmetric dense matrix represented by these arrow blocks.
    pub fn to_dense(&self) -> Array2<f64> {
        let total_dim = self.head_dim
            + self
                .levels
                .iter()
                .map(|level| level.m_ll.nrows())
                .sum::<usize>();
        let mut dense = Array2::<f64>::zeros((total_dim, total_dim));

        dense
            .slice_mut(s![0..self.head_dim, 0..self.head_dim])
            .assign(&self.m_hh);

        let mut offset = self.head_dim;
        for level in &self.levels {
            let dim = level.m_ll.nrows();
            dense
                .slice_mut(s![offset..offset + dim, offset..offset + dim])
                .assign(&level.m_ll);
            dense
                .slice_mut(s![offset..offset + dim, 0..self.head_dim])
                .assign(&level.m_lh);
            dense
                .slice_mut(s![0..self.head_dim, offset..offset + dim])
                .assign(&level.m_lh.t());
            offset += dim;
        }

        dense
    }
}

/// Factored whitened arrow system, retaining the per-level and Schur Cholesky factors.
#[derive(Clone, Debug)]
pub struct MeasureJetArrowFactor {
    head_dim: usize,
    total_dim: usize,
    level_dims: Vec<usize>,
    level_offsets: Vec<usize>,
    level_m_lh: Vec<Array2<f64>>,
    level_lowers: Vec<Array2<f64>>,
    head_schur: Array2<f64>,
    head_lower: Array2<f64>,
    dense_matrix: Array2<f64>,
    dense_lower: Array2<f64>,
}

impl MeasureJetArrowFactor {
    /// Factor an arrow system by eliminating all per-level tail blocks into the head Schur complement.
    pub fn factor(blocks: &MeasureJetArrowBlocks) -> Result<Self, MeasureJetArrowError> {
        blocks.validate()?;

        let mut head_schur = blocks.m_hh.clone();
        let mut level_lowers = Vec::with_capacity(blocks.levels.len());
        let mut level_m_lh = Vec::with_capacity(blocks.levels.len());
        let mut level_dims = Vec::with_capacity(blocks.levels.len());
        let mut level_offsets = Vec::with_capacity(blocks.levels.len());

        let mut offset = blocks.head_dim;
        for (level, block) in blocks.levels.iter().enumerate() {
            let lower = cholesky_factor_in_place(block.m_ll.view(), CholeskyGuard::FiniteStrict)
                .ok_or(MeasureJetArrowError::LevelFactorNotSpd { level })?;
            let y = forward_substitution_lower_matrix(&lower, &block.m_lh);
            let update = y.t().dot(&y);
            head_schur -= &update;

            level_offsets.push(offset);
            level_dims.push(block.m_ll.nrows());
            offset += block.m_ll.nrows();
            level_lowers.push(lower);
            level_m_lh.push(block.m_lh.clone());
        }

        let head_lower = cholesky_factor_in_place(head_schur.view(), CholeskyGuard::FiniteStrict)
            .ok_or(MeasureJetArrowError::HeadSchurNotSpd)?;
        let dense_matrix = blocks.to_dense();
        let dense_lower =
            cholesky_factor_in_place(dense_matrix.view(), CholeskyGuard::FiniteStrict)
                .ok_or(MeasureJetArrowError::HeadSchurNotSpd)?;
        let total_dim = offset;

        Ok(Self {
            head_dim: blocks.head_dim,
            total_dim,
            level_dims,
            level_offsets,
            level_m_lh,
            level_lowers,
            head_schur,
            head_lower,
            dense_matrix,
            dense_lower,
        })
    }

    /// Return the total dimension `p = k + sum_l d_l`.
    pub fn dim(&self) -> usize {
        self.total_dim
    }

    /// Return the dense head dimension `k`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Return the dimension of a level tail block.
    pub fn level_dim(&self, level: usize) -> usize {
        self.level_dims[level]
    }

    /// Return a view of the reduced head Schur complement `S`.
    pub fn reduced_head_schur(&self) -> ArrayView2<'_, f64> {
        self.head_schur.view()
    }

    /// Compute the exact `log |M|` from the arrow Cholesky pivots.
    pub fn log_det(&self) -> f64 {
        let tail_log_det = self
            .level_lowers
            .iter()
            .map(|lower| cholesky_log_det(lower.view()))
            .sum::<f64>();
        tail_log_det + cholesky_log_det(self.head_lower.view())
    }

    /// Solve `M x = rhs` exactly through the arrow Schur complement factorization.
    pub fn solve(&self, rhs: ArrayView1<'_, f64>) -> Result<Array1<f64>, MeasureJetArrowError> {
        self.ensure_vector("rhs", rhs)?;

        let mut reduced_rhs = rhs.slice(s![0..self.head_dim]).to_owned();
        for level in 0..self.level_dims.len() {
            let offset = self.level_offsets[level];
            let dim = self.level_dims[level];
            let rhs_level = rhs.slice(s![offset..offset + dim]);
            let tail_solve = cholesky_solve_vector(&self.level_lowers[level], rhs_level);
            reduced_rhs -= &self.level_m_lh[level].t().dot(&tail_solve);
        }

        let head_solution = cholesky_solve_vector(&self.head_lower, &reduced_rhs);
        let mut solution = Array1::<f64>::zeros(self.total_dim);
        solution
            .slice_mut(s![0..self.head_dim])
            .assign(&head_solution);

        for level in 0..self.level_dims.len() {
            let offset = self.level_offsets[level];
            let dim = self.level_dims[level];
            let rhs_level = rhs.slice(s![offset..offset + dim]);
            let residual = rhs_level.to_owned() - self.level_m_lh[level].dot(&head_solution);
            let tail_solution = cholesky_solve_vector(&self.level_lowers[level], &residual);
            solution
                .slice_mut(s![offset..offset + dim])
                .assign(&tail_solution);
        }

        Ok(solution)
    }

    /// Return the exact diagonal of `M^{-1}` using per-level selected inverse identities.
    pub fn inverse_diagonal(&self) -> Array1<f64> {
        let mut diagonal = Array1::<f64>::zeros(self.total_dim);
        for head_index in 0..self.head_dim {
            let unit = unit_vector(self.head_dim, head_index);
            let solved = cholesky_solve_vector(&self.head_lower, &unit);
            diagonal[head_index] = solved[head_index];
        }

        for level in 0..self.level_dims.len() {
            let offset = self.level_offsets[level];
            let level_diagonal = self.level_inverse_diagonal(level);
            diagonal
                .slice_mut(s![offset..offset + self.level_dims[level]])
                .assign(&level_diagonal);
        }

        diagonal
    }

    /// Return `tr(M^{-1} D_l)` for the selector onto one level's coordinates.
    pub fn level_trace(&self, level: usize) -> f64 {
        self.level_inverse_diagonal(level).sum()
    }

    /// Return `tr(S^{-1})`, the trace over the dense head coordinates of `M^{-1}`.
    pub fn head_trace(&self) -> f64 {
        let mut trace = 0.0;
        for head_index in 0..self.head_dim {
            let unit = unit_vector(self.head_dim, head_index);
            let solved = cholesky_solve_vector(&self.head_lower, &unit);
            trace += solved[head_index];
        }
        trace
    }

    /// Return `sum_i d_i (M^{-1})_ii` for an arbitrary diagonal selector.
    pub fn trace_with_diagonal_selector(
        &self,
        d: ArrayView1<'_, f64>,
    ) -> Result<f64, MeasureJetArrowError> {
        self.ensure_vector("diagonal selector", d)?;
        let diagonal = self.inverse_diagonal();
        Ok(d.dot(&diagonal))
    }

    /// Draw one exact posterior sample `x ~ N(0, M^{-1})` by dense-Cholesky Matheron solve.
    pub fn posterior_draw(&self, rng: &mut StdRng) -> Array1<f64> {
        let normal = RandNormal::new(0.0, 1.0).expect("N(0,1) valid");
        let mut white = Array1::<f64>::zeros(self.total_dim);
        for entry in &mut white {
            *entry = normal.sample(rng);
        }

        let mut draw = Array1::<f64>::zeros(self.total_dim);
        back_substitution_lower_transpose_guarded_into(&self.dense_lower, &white, &mut draw);
        draw
    }

    /// Draw `n_draws` exact posterior samples as an `n_draws x p` matrix with a seeded RNG.
    pub fn posterior_draws(&self, n_draws: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut draws = Array2::<f64>::zeros((n_draws, self.total_dim));
        for draw_index in 0..n_draws {
            let draw = self.posterior_draw(&mut rng);
            draws.slice_mut(s![draw_index, ..]).assign(&draw);
        }
        draws
    }

    /// Apply the BPX-style additive block inverse preconditioner to `v`.
    pub fn bpx_preconditioner_apply(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(
            v.len(),
            self.total_dim,
            "BPX preconditioner vector length must match the factored dimension"
        );

        let mut result = Array1::<f64>::zeros(self.total_dim);
        let head_part = cholesky_solve_vector(&self.head_lower, v.slice(s![0..self.head_dim]));
        result.slice_mut(s![0..self.head_dim]).assign(&head_part);

        for level in 0..self.level_dims.len() {
            let offset = self.level_offsets[level];
            let dim = self.level_dims[level];
            let level_part =
                cholesky_solve_vector(&self.level_lowers[level], v.slice(s![offset..offset + dim]));
            result
                .slice_mut(s![offset..offset + dim])
                .assign(&level_part);
        }

        result
    }

    /// Solve `M x = rhs` with BPX-preconditioned conjugate gradients.
    pub fn cg_solve(
        &self,
        rhs: ArrayView1<'_, f64>,
        max_iters: usize,
        tol: f64,
    ) -> Result<(Array1<f64>, usize), MeasureJetArrowError> {
        self.ensure_vector("rhs", rhs)?;
        if !tol.is_finite() {
            return Err(MeasureJetArrowError::NonFinite {
                what: "CG tolerance".to_string(),
            });
        }

        let mut x = Array1::<f64>::zeros(self.total_dim);
        let mut residual = rhs.to_owned();
        let mut z = self.bpx_preconditioner_apply(residual.view());
        let mut direction = z.clone();
        let mut rz_old = residual.dot(&z);
        let initial_residual_norm = l2_norm(residual.view());
        if initial_residual_norm <= tol {
            return Ok((x, 0));
        }

        for iter in 1..=max_iters {
            let matrix_direction = self.matrix_apply(direction.view());
            let denom = direction.dot(&matrix_direction);
            if !denom.is_finite() || denom <= 0.0 {
                return Err(MeasureJetArrowError::NonFinite {
                    what: "CG curvature".to_string(),
                });
            }

            let alpha = rz_old / denom;
            x += &(direction.mapv(|value| alpha * value));
            residual -= &(matrix_direction.mapv(|value| alpha * value));

            let residual_norm = l2_norm(residual.view());
            if residual_norm <= tol {
                return Ok((x, iter));
            }

            z = self.bpx_preconditioner_apply(residual.view());
            let rz_new = residual.dot(&z);
            if !rz_new.is_finite() {
                return Err(MeasureJetArrowError::NonFinite {
                    what: "CG preconditioned residual".to_string(),
                });
            }
            let beta = rz_new / rz_old;
            direction = z.clone() + direction.mapv(|value| beta * value);
            rz_old = rz_new;
        }

        Err(MeasureJetArrowError::CgDidNotConverge {
            iters: max_iters,
            residual: l2_norm(residual.view()),
        })
    }

    fn ensure_vector(
        &self,
        what: &str,
        vector: ArrayView1<'_, f64>,
    ) -> Result<(), MeasureJetArrowError> {
        ensure_dimension(what, self.total_dim, vector.len())?;
        ensure_finite_vector(what, vector)
    }

    fn level_inverse_diagonal(&self, level: usize) -> Array1<f64> {
        let dim = self.level_dims[level];
        let lower = &self.level_lowers[level];
        let coupling = &self.level_m_lh[level];
        let mut diagonal = Array1::<f64>::zeros(dim);

        for index in 0..dim {
            let unit = unit_vector(dim, index);
            let tail_column = cholesky_solve_vector(lower, &unit);
            let head_coupling = coupling.t().dot(&tail_column);
            let schur_column = cholesky_solve_vector(&self.head_lower, &head_coupling);
            diagonal[index] = tail_column[index] + head_coupling.dot(&schur_column);
        }

        diagonal
    }

    fn matrix_apply(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        self.dense_matrix.dot(&vector)
    }
}

fn ensure_dimension(what: &str, expected: usize, got: usize) -> Result<(), MeasureJetArrowError> {
    if expected == got {
        Ok(())
    } else {
        Err(MeasureJetArrowError::DimensionMismatch {
            what: what.to_string(),
            expected,
            got,
        })
    }
}

fn ensure_square(what: &str, matrix: ArrayView2<'_, f64>) -> Result<(), MeasureJetArrowError> {
    if matrix.nrows() == matrix.ncols() {
        Ok(())
    } else {
        Err(MeasureJetArrowError::DimensionMismatch {
            what: what.to_string(),
            expected: matrix.nrows(),
            got: matrix.ncols(),
        })
    }
}

fn ensure_finite_matrix(
    what: &str,
    matrix: ArrayView2<'_, f64>,
) -> Result<(), MeasureJetArrowError> {
    if matrix.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(MeasureJetArrowError::NonFinite {
            what: what.to_string(),
        })
    }
}

fn ensure_finite_vector(
    what: &str,
    vector: ArrayView1<'_, f64>,
) -> Result<(), MeasureJetArrowError> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(MeasureJetArrowError::NonFinite {
            what: what.to_string(),
        })
    }
}

fn cholesky_log_det(lower: ArrayView2<'_, f64>) -> f64 {
    2.0 * (0..lower.nrows())
        .map(|index| lower[[index, index]].ln())
        .sum::<f64>()
}

fn unit_vector(dim: usize, index: usize) -> Array1<f64> {
    let mut unit = Array1::<f64>::zeros(dim);
    unit[index] = 1.0;
    unit
}

fn l2_norm(vector: ArrayView1<'_, f64>) -> f64 {
    vector.dot(&vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::triangular::cholesky_solve_matrix;
    use approx::assert_abs_diff_eq;

    fn oracle_blocks() -> MeasureJetArrowBlocks {
        MeasureJetArrowBlocks {
            head_dim: 3,
            m_hh: Array2::from_shape_vec(
                (3, 3),
                vec![4.0, 0.18, -0.11, 0.18, 3.7, 0.16, -0.11, 0.16, 3.4],
            )
            .expect("valid head shape"),
            levels: vec![
                MeasureJetLevelBlock {
                    m_ll: Array2::from_shape_vec((2, 2), vec![2.3, 0.12, 0.12, 2.0])
                        .expect("valid level 0 shape"),
                    m_lh: Array2::from_shape_vec(
                        (2, 3),
                        vec![0.16, -0.08, 0.04, -0.05, 0.11, 0.07],
                    )
                    .expect("valid level 0 coupling shape"),
                },
                MeasureJetLevelBlock {
                    m_ll: Array2::from_shape_vec((2, 2), vec![2.5, -0.09, -0.09, 2.2])
                        .expect("valid level 1 shape"),
                    m_lh: Array2::from_shape_vec(
                        (2, 3),
                        vec![0.07, 0.09, -0.03, 0.04, -0.06, 0.10],
                    )
                    .expect("valid level 1 coupling shape"),
                },
                MeasureJetLevelBlock {
                    m_ll: Array2::from_shape_vec((1, 1), vec![1.9]).expect("valid level 2 shape"),
                    m_lh: Array2::from_shape_vec((1, 3), vec![-0.09, 0.05, 0.12])
                        .expect("valid level 2 coupling shape"),
                },
            ],
        }
    }

    fn identity(dim: usize) -> Array2<f64> {
        let mut matrix = Array2::<f64>::zeros((dim, dim));
        for index in 0..dim {
            matrix[[index, index]] = 1.0;
        }
        matrix
    }

    fn dense_lower(matrix: &Array2<f64>) -> Array2<f64> {
        cholesky_factor_in_place(matrix.view(), CholeskyGuard::FiniteStrict)
            .expect("dense oracle matrix is SPD")
    }

    fn dense_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let lower = dense_lower(matrix);
        cholesky_solve_matrix(&lower, &identity(matrix.nrows()))
    }

    fn dense_log_det(matrix: &Array2<f64>) -> f64 {
        cholesky_log_det(dense_lower(matrix).view())
    }

    fn rhs_cases() -> Vec<Array1<f64>> {
        vec![
            Array1::from_vec(vec![0.7, -1.1, 0.4, 0.2, -0.8, 1.2, -0.3, 0.9]),
            Array1::from_vec(vec![1.5, 0.0, -0.6, -0.4, 0.3, 0.8, -1.0, 0.1]),
            Array1::from_vec(vec![-0.2, 0.9, 1.1, -1.3, 0.5, -0.7, 0.6, -0.4]),
        ]
    }

    #[test]
    fn factored_solve_matches_dense_solve() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let dense = blocks.to_dense();
        let dense_inv = dense_inverse(&dense);

        for rhs in rhs_cases() {
            let factored = factor.solve(rhs.view()).expect("arrow solve succeeds");
            let expected = dense_inv.dot(&rhs);
            for index in 0..factor.dim() {
                assert_abs_diff_eq!(factored[index], expected[index], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn factored_logdet_matches_dense_logdet() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let dense = blocks.to_dense();

        assert_abs_diff_eq!(factor.log_det(), dense_log_det(&dense), epsilon = 1e-10);
    }

    #[test]
    fn level_trace_matches_dense_trace() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let dense = blocks.to_dense();
        let dense_inv = dense_inverse(&dense);
        let inverse_diagonal = factor.inverse_diagonal();

        for index in 0..factor.dim() {
            assert_abs_diff_eq!(
                inverse_diagonal[index],
                dense_inv[[index, index]],
                epsilon = 1e-10
            );
        }

        let head_trace = (0..factor.head_dim())
            .map(|index| dense_inv[[index, index]])
            .sum::<f64>();
        assert_abs_diff_eq!(factor.head_trace(), head_trace, epsilon = 1e-10);

        for level in 0..3 {
            let offset = factor.level_offsets[level];
            let expected = (offset..offset + factor.level_dim(level))
                .map(|index| dense_inv[[index, index]])
                .sum::<f64>();
            assert_abs_diff_eq!(factor.level_trace(level), expected, epsilon = 1e-10);
        }

        let selector = Array1::from_vec(vec![1.0, 0.25, -0.5, 0.0, 1.2, -0.7, 0.6, 0.9]);
        let selected_expected = selector
            .iter()
            .enumerate()
            .map(|(index, weight)| weight * dense_inv[[index, index]])
            .sum::<f64>();
        let selected = factor
            .trace_with_diagonal_selector(selector.view())
            .expect("selector trace succeeds");
        assert_abs_diff_eq!(selected, selected_expected, epsilon = 1e-10);
    }

    #[test]
    fn posterior_draw_covariance_matches_inverse() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let dense = blocks.to_dense();
        let dense_inv = dense_inverse(&dense);
        let n_draws = 200_000;
        let draws = factor.posterior_draws(n_draws, 930_245);
        let mut mean = Array1::<f64>::zeros(factor.dim());

        for row in draws.rows() {
            mean += &row.to_owned();
        }
        mean.mapv_inplace(|value| value / n_draws as f64);

        for index in 0..factor.dim() {
            assert_abs_diff_eq!(mean[index], 0.0, epsilon = 0.01);
        }

        let mut covariance = Array2::<f64>::zeros((factor.dim(), factor.dim()));
        for row in draws.rows() {
            let centered = row.to_owned() - &mean;
            for i in 0..factor.dim() {
                for j in 0..factor.dim() {
                    covariance[[i, j]] += centered[i] * centered[j];
                }
            }
        }
        covariance.mapv_inplace(|value| value / (n_draws - 1) as f64);

        for i in 0..factor.dim() {
            for j in 0..factor.dim() {
                assert_abs_diff_eq!(covariance[[i, j]], dense_inv[[i, j]], epsilon = 0.02);
            }
        }
    }

    #[test]
    fn cg_solve_matches_direct_solve() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let dense = blocks.to_dense();
        let dense_inv = dense_inverse(&dense);
        let rhs = Array1::from_vec(vec![0.8, -0.4, 1.1, 0.3, -0.9, 0.5, 1.4, -0.2]);

        let direct = factor
            .solve(rhs.view())
            .expect("direct arrow solve succeeds");
        let expected = dense_inv.dot(&rhs);
        let (iterative, iterations) = factor
            .cg_solve(rhs.view(), factor.dim(), 1e-12)
            .expect("CG converges");
        assert!(iterations <= factor.dim());

        for index in 0..factor.dim() {
            assert_abs_diff_eq!(direct[index], expected[index], epsilon = 1e-10);
            assert_abs_diff_eq!(iterative[index], expected[index], epsilon = 1e-10);
        }
    }

    #[test]
    fn bpx_preconditioner_is_spd_action() {
        let blocks = oracle_blocks();
        let factor = MeasureJetArrowFactor::factor(&blocks).expect("arrow factorization succeeds");
        let vector = Array1::from_vec(vec![0.6, -0.3, 0.8, 1.0, -0.7, 0.4, -1.1, 0.9]);
        let mut block_diagonal_times_vector = Array1::<f64>::zeros(factor.dim());

        let head_block_times_vector = factor
            .reduced_head_schur()
            .dot(&vector.slice(s![0..factor.head_dim()]));
        block_diagonal_times_vector
            .slice_mut(s![0..factor.head_dim()])
            .assign(&head_block_times_vector);

        for level in 0..3 {
            let offset = factor.level_offsets[level];
            let dim = factor.level_dim(level);
            let level_block_times_vector = blocks.levels[level]
                .m_ll
                .dot(&vector.slice(s![offset..offset + dim]));
            block_diagonal_times_vector
                .slice_mut(s![offset..offset + dim])
                .assign(&level_block_times_vector);
        }

        let applied = factor.bpx_preconditioner_apply(block_diagonal_times_vector.view());
        for index in 0..factor.dim() {
            assert_abs_diff_eq!(applied[index], vector[index], epsilon = 1e-10);
        }

        let preconditioned = factor.bpx_preconditioner_apply(vector.view());
        assert!(vector.dot(&preconditioned) > 0.0);
    }

    #[test]
    fn validate_rejects_shape_mismatch() {
        let mut wrong_coupling = oracle_blocks();
        wrong_coupling.levels[0].m_lh =
            Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).expect("valid wrong shape");
        assert!(wrong_coupling.validate().is_err());

        let mut nonsquare_level = oracle_blocks();
        nonsquare_level.levels[1].m_ll =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .expect("valid nonsquare shape");
        assert!(nonsquare_level.validate().is_err());

        let mut nonfinite = oracle_blocks();
        nonfinite.m_hh[[0, 0]] = f64::NAN;
        assert!(nonfinite.validate().is_err());
    }

    #[test]
    fn to_dense_is_symmetric_and_spd() {
        let blocks = oracle_blocks();
        let dense = blocks.to_dense();

        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert_abs_diff_eq!(dense[[i, j]], dense[[j, i]], epsilon = 1e-12);
            }
        }

        assert!(cholesky_factor_in_place(dense.view(), CholeskyGuard::FiniteStrict).is_some());
    }
}
