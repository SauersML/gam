// One Gauge object (#933).
//
// Every identifiability mechanism in the engine performs the same
// mathematical act: quotient the coefficient space by directions in
// ker(J) ∩ ker(S), pick a section, fit in the reduced coordinates θ,
// and lift estimates / covariance / geometry back to the raw
// coordinates β. This module owns that act once.
//
// A `Gauge` is the affine section itself: the lift matrix
// `T : reduced → raw` (`β_raw = T · θ`) together with the per-block
// partitions of both coordinate systems. Block-diagonal `T`
// (independent per-block reductions, the canonical-audit case) and
// block-upper-triangular `T` (cross-block residualisation, the
// survival V+M-exact compile) are the same object — the partitions
// record where each block's rows/columns live.
//
// Lift conventions (the whole point — there is exactly one):
//   - point estimate:   β_raw = T · θ
//   - covariance / any symmetric bilinear form: Σ_raw = T · Σ_θ · Tᵀ
//   - η is invariant:   X_raw · T · θ = X_reduced · θ
//
// Raw directions outside the section (zero rows of `T`) receive exactly
// zero estimate, zero variance, and zero covariance with every other
// coordinate: a coordinate the reduced fit cannot move carries no
// posterior uncertainty in raw space.

use ndarray::{Array1, Array2};

use crate::linalg::faer_ndarray::{fast_ab, fast_abt};

/// The lift `T : reduced → raw` plus the per-block partitions of both
/// coordinate systems. See the module docs for the lift conventions.
#[derive(Debug, Clone)]
pub struct Gauge {
    /// Global lift matrix, shape `(Σ p_b_raw) × (Σ r_b_reduced)`.
    pub t_full: Array2<f64>,
    /// Raw-coordinate block partition: `block_starts_raw[b]..block_starts_raw[b+1]`
    /// is block `b`'s raw row range in `t_full`. Length `n_blocks + 1`, starts at 0.
    pub block_starts_raw: Vec<usize>,
    /// Reduced-coordinate block partition (columns of `t_full`), same layout.
    pub block_starts_reduced: Vec<usize>,
}

fn starts_from_widths(widths: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(widths.len() + 1);
    starts.push(0);
    for w in widths {
        starts.push(starts.last().copied().unwrap() + w);
    }
    starts
}

/// Assemble a block-upper-triangular lift `T` from per-block diagonal
/// `V_b` matrices and strictly-upper residualisation blocks `R_{a→b}`.
///
/// `r_per_term[b]` (when `Some`) packs ALL strictly-upper off-diagonal
/// columns for block `b` stacked row-wise across all earlier-priority
/// blocks `a < b`: `nrows = Σ_{a<b} v_per_term[a].nrows()`,
/// `ncols = v_per_term[b].ncols()`. The assembled `T` carries `V_b` on
/// the diagonal and `−R_{a→b}` at `(a, b)`. `r_per_term[0]` must be
/// `None` (no earlier block to residualise against).
pub fn assemble_block_triangular_t(
    v_per_term: &[Array2<f64>],
    r_per_term: &[Option<Array2<f64>>],
) -> Array2<f64> {
    assert_eq!(
        v_per_term.len(),
        r_per_term.len(),
        "assemble_block_triangular_t: v_per_term len {} != r_per_term len {}",
        v_per_term.len(),
        r_per_term.len(),
    );
    let raw_widths: Vec<usize> = v_per_term.iter().map(|v| v.nrows()).collect();
    let kept_widths: Vec<usize> = v_per_term.iter().map(|v| v.ncols()).collect();
    let row_offsets = starts_from_widths(&raw_widths);
    let col_offsets = starts_from_widths(&kept_widths);
    let total_rows = row_offsets.last().copied().unwrap_or(0);
    let total_cols = col_offsets.last().copied().unwrap_or(0);
    let mut t = Array2::<f64>::zeros((total_rows, total_cols));
    // Diagonal: place V_b at (b, b).
    for (b, v) in v_per_term.iter().enumerate() {
        let r = v.nrows();
        let c = v.ncols();
        if r > 0 && c > 0 {
            t.slice_mut(ndarray::s![
                row_offsets[b]..row_offsets[b] + r,
                col_offsets[b]..col_offsets[b] + c
            ])
            .assign(v);
        }
    }
    // Strict upper triangle: for each b ≥ 1, place −R_{a→b} at (a, b),
    // a < b, slicing the row-stacked `r_per_term[b]` in earlier-block order.
    for b in 1..v_per_term.len() {
        let Some(r_stack) = r_per_term[b].as_ref() else {
            continue;
        };
        let kept_b = kept_widths[b];
        assert_eq!(
            r_stack.ncols(),
            kept_b,
            "assemble_block_triangular_t: r_per_term[{b}] has {} cols, expected {}",
            r_stack.ncols(),
            kept_b,
        );
        let expected_rows: usize = raw_widths.iter().take(b).sum();
        assert_eq!(
            r_stack.nrows(),
            expected_rows,
            "assemble_block_triangular_t: r_per_term[{b}] has {} rows, expected {} \
             (sum of raw_widths[0..{}])",
            r_stack.nrows(),
            expected_rows,
            b,
        );
        let mut local_row = 0usize;
        for a in 0..b {
            let r_a = raw_widths[a];
            if r_a == 0 || kept_b == 0 {
                local_row += r_a;
                continue;
            }
            let block = r_stack.slice(ndarray::s![local_row..local_row + r_a, ..]);
            let mut dst = t.slice_mut(ndarray::s![
                row_offsets[a]..row_offsets[a] + r_a,
                col_offsets[b]..col_offsets[b] + kept_b
            ]);
            for i in 0..r_a {
                for j in 0..kept_b {
                    dst[[i, j]] = -block[[i, j]];
                }
            }
            local_row += r_a;
        }
    }
    t
}

impl Gauge {
    /// The trivial section: raw == reduced for every block.
    pub fn identity(raw_widths: &[usize]) -> Self {
        let transforms: Vec<Array2<f64>> =
            raw_widths.iter().map(|&w| Array2::<f64>::eye(w)).collect();
        Self::from_block_transforms(&transforms)
    }

    /// Block-diagonal section from independent per-block lifts
    /// `T_b : reduced_b → raw_b` (selection matrices from the canonical
    /// audit, orthogonalisation `V_b`s, or their compositions).
    pub fn from_block_transforms(transforms: &[Array2<f64>]) -> Self {
        let r_none: Vec<Option<Array2<f64>>> = transforms.iter().map(|_| None).collect();
        Self::from_v_and_r(transforms, &r_none)
    }

    /// Block-upper-triangular section from per-block `V_b` plus
    /// cross-block residualisation stacks `R_{a→b}` — see
    /// [`assemble_block_triangular_t`] for the packing convention.
    pub fn from_v_and_r(v_per_term: &[Array2<f64>], r_per_term: &[Option<Array2<f64>>]) -> Self {
        let raw_widths: Vec<usize> = v_per_term.iter().map(|v| v.nrows()).collect();
        let reduced_widths: Vec<usize> = v_per_term.iter().map(|v| v.ncols()).collect();
        Self {
            t_full: assemble_block_triangular_t(v_per_term, r_per_term),
            block_starts_raw: starts_from_widths(&raw_widths),
            block_starts_reduced: starts_from_widths(&reduced_widths),
        }
    }

    /// The sum-to-zero (centering) section as a first-class single-block
    /// gauge. `z` is the `(k × (k−1))` reparametrisation matrix returned by
    /// [`crate::terms::basis::duchon_thinplate::apply_sum_to_zero_constraint`]
    /// (an orthonormal basis for `null(cᵀ)`, `c = Bᵀw` the weighted column
    /// sums): the constrained design is `B_c = B · z`, so on the model
    /// `η = B · β_raw = B_c · θ = B · z · θ` the raw coefficients lift back
    /// from the reduced (centred) coefficients by exactly `β_raw = z · θ`.
    ///
    /// That is the one Gauge convention with `T = z` over a single block, so
    /// the centring constraint stops being a special-cased outside-the-object
    /// transform and becomes a `Gauge` section like every other reduction:
    /// the covariance / penalised-Hessian of the centred fit pushes forward to
    /// the raw basis through the SAME `z` via [`Gauge::lift_covariance`].
    ///
    /// `z` is taken as the section itself (rather than recomputed from a basis)
    /// because the constraint matrix is the only gauge-relevant artifact — the
    /// basis the column sums were taken over is irrelevant to the lift. The
    /// only requirement is the structural one of a centring section:
    /// `z.ncols() < z.nrows()` (at least one direction is removed); an identity
    /// `z` would be `Gauge::identity` and is rejected so callers do not silently
    /// treat an unconstrained block as centred.
    pub fn sum_to_zero(z: Array2<f64>) -> Self {
        let (k, r) = z.dim();
        assert!(
            k > 0 && r < k,
            "Gauge::sum_to_zero: z must be a tall reparametrisation ({k}×{r}); \
             a centring section removes at least one direction (r < k)",
        );
        Self::from_block_transforms(&[z])
    }

    /// Wrap an already-assembled global `T` given the per-block raw and
    /// reduced width partitions.
    pub fn from_t(t_full: Array2<f64>, raw_widths: &[usize], reduced_widths: &[usize]) -> Self {
        assert_eq!(
            raw_widths.len(),
            reduced_widths.len(),
            "Gauge::from_t: raw_widths len {} != reduced_widths len {}",
            raw_widths.len(),
            reduced_widths.len(),
        );
        let total_raw: usize = raw_widths.iter().sum();
        let total_reduced: usize = reduced_widths.iter().sum();
        assert_eq!(
            t_full.dim(),
            (total_raw, total_reduced),
            "Gauge::from_t: T has shape {:?}, expected ({total_raw}, {total_reduced})",
            t_full.dim(),
        );
        Self {
            t_full,
            block_starts_raw: starts_from_widths(raw_widths),
            block_starts_reduced: starts_from_widths(reduced_widths),
        }
    }

    /// Build from a [`CompiledMap`] emitted by
    /// [`crate::identifiability::families::compiler::compile_from_raw_grams`]:
    /// `map.raw_from_compiled` IS the global triangular `T`, and the
    /// block ranges give both partitions. `ordering` is accepted purely
    /// as a length sanity check.
    ///
    /// [`CompiledMap`]: crate::identifiability::families::compiler::CompiledMap
    pub fn from_compiled_map(
        map: &crate::identifiability::families::compiler::CompiledMap,
        ordering: &[crate::identifiability::families::compiler::BlockOrder],
    ) -> Self {
        assert_eq!(
            map.raw_block_ranges.len(),
            map.compiled_block_ranges.len(),
            "Gauge::from_compiled_map: CompiledMap raw_block_ranges len {} != \
             compiled_block_ranges len {}",
            map.raw_block_ranges.len(),
            map.compiled_block_ranges.len(),
        );
        assert_eq!(
            map.raw_block_ranges.len(),
            ordering.len(),
            "Gauge::from_compiled_map: ordering len {} != block count {}",
            ordering.len(),
            map.raw_block_ranges.len(),
        );
        let mut block_starts_raw = Vec::with_capacity(map.raw_block_ranges.len() + 1);
        block_starts_raw.push(0);
        for r in &map.raw_block_ranges {
            block_starts_raw.push(r.end);
        }
        let mut block_starts_reduced = Vec::with_capacity(map.compiled_block_ranges.len() + 1);
        block_starts_reduced.push(0);
        for r in &map.compiled_block_ranges {
            block_starts_reduced.push(r.end);
        }
        Self {
            t_full: map.raw_from_compiled.clone(),
            block_starts_raw,
            block_starts_reduced,
        }
    }

    /// Number of blocks in the partition.
    pub fn n_blocks(&self) -> usize {
        self.block_starts_raw.len().saturating_sub(1)
    }

    /// Total raw width `Σ p_b`.
    pub fn raw_total(&self) -> usize {
        self.block_starts_raw.last().copied().unwrap_or(0)
    }

    /// Total reduced width `Σ r_b`.
    pub fn reduced_total(&self) -> usize {
        self.block_starts_reduced.last().copied().unwrap_or(0)
    }

    /// Per-block raw widths.
    pub fn raw_widths(&self) -> Vec<usize> {
        self.block_starts_raw
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }

    /// Per-block reduced widths.
    pub fn reduced_widths(&self) -> Vec<usize> {
        self.block_starts_reduced
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }

    /// The diagonal slab `T_b = T[raw_b, reduced_b]` of block `b`.
    /// For a block-diagonal gauge this is the whole story for the
    /// block; for a triangular gauge it omits the cross-block `−R`.
    pub fn block_transform(&self, b: usize) -> Array2<f64> {
        assert!(
            b < self.n_blocks(),
            "Gauge::block_transform: block {b} out of range {}",
            self.n_blocks(),
        );
        self.t_full
            .slice(ndarray::s![
                self.block_starts_raw[b]..self.block_starts_raw[b + 1],
                self.block_starts_reduced[b]..self.block_starts_reduced[b + 1]
            ])
            .to_owned()
    }

    /// Append blocks that were never reduced (raw == reduced, identity
    /// lift). Used to lift joint objects that span both gauged blocks
    /// and untouched ones (e.g. the survival flex blocks alongside the
    /// compiled parametric blocks).
    pub fn extend_with_identity(&self, extra_raw_widths: &[usize]) -> Self {
        let extra_total: usize = extra_raw_widths.iter().sum();
        let raw_total = self.raw_total();
        let reduced_total = self.reduced_total();
        let mut t = Array2::<f64>::zeros((raw_total + extra_total, reduced_total + extra_total));
        t.slice_mut(ndarray::s![0..raw_total, 0..reduced_total])
            .assign(&self.t_full);
        for k in 0..extra_total {
            t[[raw_total + k, reduced_total + k]] = 1.0;
        }
        let mut block_starts_raw = self.block_starts_raw.clone();
        let mut block_starts_reduced = self.block_starts_reduced.clone();
        for &w in extra_raw_widths {
            block_starts_raw.push(block_starts_raw.last().copied().unwrap() + w);
            block_starts_reduced.push(block_starts_reduced.last().copied().unwrap() + w);
        }
        Self {
            t_full: t,
            block_starts_raw,
            block_starts_reduced,
        }
    }

    /// Lift per-block reduced coefficients to per-block raw
    /// coefficients: concatenate into θ, apply `β = T · θ`, split at
    /// the raw partition.
    pub fn lift_block_betas(&self, reduced_block_betas: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let n_blocks = self.n_blocks();
        assert_eq!(
            reduced_block_betas.len(),
            n_blocks,
            "Gauge::lift_block_betas: got {} reduced block betas, expected {}",
            reduced_block_betas.len(),
            n_blocks,
        );
        for (b, beta) in reduced_block_betas.iter().enumerate() {
            let expected = self.block_starts_reduced[b + 1] - self.block_starts_reduced[b];
            assert_eq!(
                beta.len(),
                expected,
                "Gauge::lift_block_betas: block {b} has β of len {}, expected reduced width {}",
                beta.len(),
                expected,
            );
        }
        let mut theta_full = Array1::<f64>::zeros(self.reduced_total());
        for (b, beta) in reduced_block_betas.iter().enumerate() {
            let c0 = self.block_starts_reduced[b];
            let c1 = self.block_starts_reduced[b + 1];
            theta_full.slice_mut(ndarray::s![c0..c1]).assign(beta);
        }
        let beta_full = self.t_full.dot(&theta_full);
        let mut out = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let r0 = self.block_starts_raw[b];
            let r1 = self.block_starts_raw[b + 1];
            out.push(beta_full.slice(ndarray::s![r0..r1]).to_owned());
        }
        out
    }

    /// Push a reduced-coordinate symmetric matrix (posterior covariance,
    /// penalized Hessian — any symmetric bilinear form on θ) forward to
    /// raw coordinates via the exact sandwich `M_raw = T · M_θ · Tᵀ`.
    ///
    /// The result is explicitly symmetrised: `T · M · Tᵀ` is symmetric
    /// for symmetric `M`, but the two matmuls accumulate independent
    /// rounding, so the transpose pair is averaged to land an exactly
    /// symmetric matrix for downstream Cholesky / eigensolves.
    pub fn lift_covariance(&self, m_reduced: &Array2<f64>) -> Array2<f64> {
        let total_reduced = self.reduced_total();
        assert_eq!(
            m_reduced.dim(),
            (total_reduced, total_reduced),
            "Gauge::lift_covariance: matrix has shape {:?}, expected ({total_reduced}, {total_reduced})",
            m_reduced.dim(),
        );
        let t_m = fast_ab(&self.t_full, m_reduced);
        let mut raw = fast_abt(&t_m, &self.t_full);
        let n = raw.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (raw[[i, j]] + raw[[j, i]]);
                raw[[i, j]] = avg;
                raw[[j, i]] = avg;
            }
        }
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_gauge_round_trips_betas_and_covariance() {
        let gauge = Gauge::identity(&[2, 3]);
        assert_eq!(gauge.n_blocks(), 2);
        assert_eq!(gauge.raw_total(), 5);
        assert_eq!(gauge.reduced_total(), 5);
        let theta = vec![
            Array1::from(vec![0.5, -0.25]),
            Array1::from(vec![1.0, 2.0, -3.0]),
        ];
        let raw = gauge.lift_block_betas(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[0.5, -0.25]);
        assert_eq!(raw[1].as_slice().unwrap(), &[1.0, 2.0, -3.0]);

        let mut cov = Array2::<f64>::eye(5);
        cov[[0, 3]] = 0.4;
        cov[[3, 0]] = 0.4;
        let lifted = gauge.lift_covariance(&cov);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (lifted[[i, j]] - cov[[i, j]]).abs() < 1e-14,
                    "identity gauge must be a covariance no-op at ({i},{j})",
                );
            }
        }
    }

    #[test]
    fn block_diagonal_gauge_matches_per_block_lift() {
        // Block 0: selection keeping raw cols {0, 2} of width 3.
        let mut t0 = Array2::<f64>::zeros((3, 2));
        t0[[0, 0]] = 1.0;
        t0[[2, 1]] = 1.0;
        // Block 1: full identity of width 2.
        let t1 = Array2::<f64>::eye(2);
        let gauge = Gauge::from_block_transforms(&[t0.clone(), t1.clone()]);
        assert_eq!(gauge.raw_widths(), vec![3, 2]);
        assert_eq!(gauge.reduced_widths(), vec![2, 2]);

        let theta = vec![Array1::from(vec![1.5, -2.5]), Array1::from(vec![0.5, 4.0])];
        let raw = gauge.lift_block_betas(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[1.5, 0.0, -2.5]);
        assert_eq!(raw[1].as_slice().unwrap(), &[0.5, 4.0]);

        // block_transform recovers the diagonal slabs exactly.
        assert_eq!(gauge.block_transform(0), t0);
        assert_eq!(gauge.block_transform(1), t1);
    }

    #[test]
    fn triangular_gauge_applies_negative_r_off_diagonal() {
        // Two blocks, raw widths 2 and 2; block 1 keeps 1 column and is
        // residualised against block 0 by R (2×1).
        let v_a = Array2::<f64>::eye(2);
        let mut v_b = Array2::<f64>::zeros((2, 1));
        v_b[[0, 0]] = 1.0;
        let mut r_ab = Array2::<f64>::zeros((2, 1));
        r_ab[[0, 0]] = 0.5;
        r_ab[[1, 0]] = -0.25;
        let gauge = Gauge::from_v_and_r(&[v_a, v_b], &[None, Some(r_ab)]);

        let theta = vec![Array1::from(vec![1.0, 2.0]), Array1::from(vec![4.0])];
        let raw = gauge.lift_block_betas(&theta);
        // β_a = V_a·θ_a − R_{a→b}·θ_b = [1 − 0.5·4, 2 + 0.25·4] = [−1, 3].
        assert!((raw[0][0] - (-1.0)).abs() < 1e-14);
        assert!((raw[0][1] - 3.0).abs() < 1e-14);
        // β_b = V_b·θ_b = [4, 0].
        assert!((raw[1][0] - 4.0).abs() < 1e-14);
        assert!((raw[1][1] - 0.0).abs() < 1e-14);
    }

    /// The covariance lift must be the exact pushforward of the SAME
    /// `T` the β lift applies: for a rank-1 `Σ_θ = θθᵀ`, the lifted
    /// covariance must equal `(Tθ)(Tθ)ᵀ` built from the lifted β.
    #[test]
    fn covariance_lift_is_rank1_consistent_with_beta_lift() {
        let v_a = Array2::<f64>::eye(2);
        let mut v_b = Array2::<f64>::zeros((2, 1));
        v_b[[0, 0]] = 1.0;
        let mut r_ab = Array2::<f64>::zeros((2, 1));
        r_ab[[0, 0]] = 0.3;
        r_ab[[1, 0]] = 0.7;
        let gauge = Gauge::from_v_and_r(&[v_a, v_b], &[None, Some(r_ab)]);

        let theta = vec![Array1::from(vec![0.8, -1.2]), Array1::from(vec![2.0])];
        let raw = gauge.lift_block_betas(&theta);
        let beta_full: Vec<f64> = raw.iter().flat_map(|b| b.iter().copied()).collect();

        let theta_full = Array1::from(vec![0.8, -1.2, 2.0]);
        let cov_rank1 = {
            let n = theta_full.len();
            Array2::from_shape_fn((n, n), |(i, j)| theta_full[i] * theta_full[j])
        };
        let lifted = gauge.lift_covariance(&cov_rank1);
        assert_eq!(lifted.dim(), (4, 4));
        for i in 0..4 {
            for j in 0..4 {
                let expected = beta_full[i] * beta_full[j];
                assert!(
                    (lifted[[i, j]] - expected).abs() < 1e-12,
                    "rank-1 covariance lift must equal (Tθ)(Tθ)ᵀ at ({i},{j}): \
                     got {} expected {expected}",
                    lifted[[i, j]],
                );
            }
        }
    }

    /// `Gauge::sum_to_zero(z)` must lift exactly as `β_raw = z · θ`, and the
    /// lift must preserve the linear predictor: for any centred design
    /// `B_c = B · z` and any reduced coefficient `θ`, the raw prediction
    /// `B · (z · θ)` equals the reduced prediction `B_c · θ`. This is the
    /// invariant that makes `z` the correct section — a wrong gauge would
    /// preserve coefficients but break η.
    #[test]
    fn sum_to_zero_gauge_lifts_via_z_and_preserves_eta() {
        // A concrete orthonormal centring section: null space of c = [1,1,1]ᵀ
        // (the unweighted sum-to-zero constraint on a width-3 block), built as
        // two orthonormal columns each summing to zero.
        let s = 1.0 / 2.0_f64.sqrt();
        let s6 = 1.0 / 6.0_f64.sqrt();
        let mut z = Array2::<f64>::zeros((3, 2));
        z[[0, 0]] = s;
        z[[1, 0]] = -s;
        z[[2, 0]] = 0.0;
        z[[0, 1]] = s6;
        z[[1, 1]] = s6;
        z[[2, 1]] = -2.0 * s6;
        // The columns are orthonormal and sum to zero (cᵀz = 0).
        for j in 0..2 {
            assert!((z.column(j).sum()).abs() < 1e-14, "column {j} must sum to 0");
            assert!(
                (z.column(j).dot(&z.column(j)) - 1.0).abs() < 1e-14,
                "column {j} must be unit norm"
            );
        }

        let gauge = Gauge::sum_to_zero(z.clone());
        assert_eq!(gauge.n_blocks(), 1);
        assert_eq!(gauge.raw_widths(), vec![3]);
        assert_eq!(gauge.reduced_widths(), vec![2]);
        assert_eq!(gauge.block_transform(0), z);

        // Lift β_raw = z · θ exactly.
        let theta = Array1::from(vec![1.3, -0.7]);
        let raw = gauge.lift_block_betas(&[theta.clone()]);
        let expected_raw = z.dot(&theta);
        for i in 0..3 {
            assert!((raw[0][i] - expected_raw[i]).abs() < 1e-14);
        }
        // Centring is satisfied: the raw coefficients sum to zero.
        assert!(raw[0].sum().abs() < 1e-14, "lifted β must be centred");

        // η preservation: B · (z · θ) == (B · z) · θ for an arbitrary B.
        let b = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, -1.0, 0.5, -0.5, 3.0, 2.0, 1.0, 1.0, -1.0, 0.0, 4.0,
            ],
        )
        .unwrap();
        let b_c = fast_ab(&b, &z); // the constrained design B_c
        let eta_reduced = b_c.dot(&theta);
        let eta_raw = b.dot(&expected_raw);
        for i in 0..4 {
            assert!(
                (eta_reduced[i] - eta_raw[i]).abs() < 1e-13,
                "η must be invariant under the centring lift at row {i}",
            );
        }

        // Covariance pushforward through the SAME z (rank-1 consistency).
        let cov_rank1 = Array2::from_shape_fn((2, 2), |(i, j)| theta[i] * theta[j]);
        let lifted = gauge.lift_covariance(&cov_rank1);
        assert_eq!(lifted.dim(), (3, 3));
        for i in 0..3 {
            for j in 0..3 {
                let expect = expected_raw[i] * expected_raw[j];
                assert!(
                    (lifted[[i, j]] - expect).abs() < 1e-13,
                    "centring covariance lift must equal (zθ)(zθ)ᵀ at ({i},{j})",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "removes at least one direction")]
    fn sum_to_zero_rejects_identity_section() {
        // A square z removes no direction — that is not a centring section.
        let _ = Gauge::sum_to_zero(Array2::<f64>::eye(3));
    }

    #[test]
    fn extend_with_identity_passes_extra_blocks_through() {
        let mut t0 = Array2::<f64>::zeros((2, 1));
        t0[[0, 0]] = 1.0;
        let gauge = Gauge::from_block_transforms(&[t0]).extend_with_identity(&[2]);
        assert_eq!(gauge.n_blocks(), 2);
        assert_eq!(gauge.raw_total(), 4);
        assert_eq!(gauge.reduced_total(), 3);

        let theta = vec![Array1::from(vec![3.0]), Array1::from(vec![1.0, -1.0])];
        let raw = gauge.lift_block_betas(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[3.0, 0.0]);
        assert_eq!(raw[1].as_slice().unwrap(), &[1.0, -1.0]);

        // Covariance: the extra (untouched) block's diagonal sub-matrix
        // survives the lift bit-for-bit; the reduced block zero-pads.
        let mut cov = Array2::<f64>::eye(3);
        cov[[1, 2]] = 0.25;
        cov[[2, 1]] = 0.25;
        let lifted = gauge.lift_covariance(&cov);
        assert_eq!(lifted.dim(), (4, 4));
        assert!((lifted[[0, 0]] - 1.0).abs() < 1e-14);
        assert!(
            (lifted[[1, 1]] - 0.0).abs() < 1e-14,
            "dropped raw row has zero variance"
        );
        assert!((lifted[[2, 2]] - 1.0).abs() < 1e-14);
        assert!((lifted[[3, 3]] - 1.0).abs() < 1e-14);
        assert!((lifted[[2, 3]] - 0.25).abs() < 1e-14);
    }
}
