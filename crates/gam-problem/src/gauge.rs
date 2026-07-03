// One Gauge object (#933).
//
// Every identifiability mechanism in the engine performs the same
// mathematical act: quotient the coefficient space by directions in
// ker(J) ∩ ker(S), pick a section, fit in the reduced coordinates θ,
// and lift estimates / covariance / geometry back to the raw
// coordinates β. This module owns that act once.
//
// A `Gauge` is the affine section itself: the lift matrix
// `T : reduced → raw` plus an affine shift `a`
// (`β_raw = T · θ + a`) together with the per-block partitions
// of both coordinate systems. Block-diagonal `T`
// (independent per-block reductions, the canonical-audit case) and
// block-upper-triangular `T` (cross-block residualisation, the
// survival V+M-exact compile) are the same object — the partitions
// record where each block's rows/columns live.
//
// Lift conventions (the whole point — there is exactly one):
//   - point estimate:   β_raw = T · θ + a
//   - covariance / any symmetric bilinear form: Σ_raw = T · Σ_θ · Tᵀ
//   - η is invariant:   X_raw · (T · θ + a) = X_reduced · θ + offset_reduced
//
// Raw directions outside the section (zero rows of `T`) receive exactly
// zero estimate, zero variance, and zero covariance with every other
// coordinate: a coordinate the reduced fit cannot move carries no
// posterior uncertainty in raw space.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};

/// Neutral view of a compiled identifiability reparametrisation that
/// [`Gauge::from_compiled_map`] consumes. The concrete `CompiledMap`
/// emitted by the identifiability compiler lives ABOVE this crate, so
/// `Gauge` names only this trait (inverted dependency #1521); the
/// compiler crate provides the `impl`.
///
/// `raw_from_compiled` IS the global triangular lift `T`; the two block
/// range slices give the raw-width and compiled-width column partitions.
pub trait CompiledBlockMap {
    /// The `(p_raw × p_compiled)` raw-from-compiled reparam matrix `T`.
    fn raw_from_compiled(&self) -> &Array2<f64>;
    /// Per-block raw-width column ranges.
    fn raw_block_ranges(&self) -> &[std::ops::Range<usize>];
    /// Per-block compiled-width column ranges, parallel to
    /// [`Self::raw_block_ranges`].
    fn compiled_block_ranges(&self) -> &[std::ops::Range<usize>];
}

/// The lift `T : reduced → raw` plus the per-block partitions of both
/// coordinate systems. See the module docs for the lift conventions.
#[derive(Debug, Clone)]
pub struct Gauge {
    /// Global lift matrix, shape `(Σ p_b_raw) × (Σ r_b_reduced)`.
    pub t_full: Array2<f64>,
    /// Global affine shift in raw coordinates, length `Σ p_b_raw`.
    pub affine_shift: Array1<f64>,
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
        let raw_total: usize = transforms.iter().map(|t| t.nrows()).sum();
        Self::from_block_transforms_with_shift(transforms, Array1::zeros(raw_total))
    }

    /// Block-diagonal affine section from independent per-block lifts
    /// plus one concatenated raw-coordinate shift.
    pub fn from_block_transforms_with_shift(
        transforms: &[Array2<f64>],
        affine_shift: Array1<f64>,
    ) -> Self {
        let r_none: Vec<Option<Array2<f64>>> = transforms.iter().map(|_| None).collect();
        let mut gauge = Self::from_v_and_r(transforms, &r_none);
        assert_eq!(
            affine_shift.len(),
            gauge.raw_total(),
            "Gauge::from_block_transforms_with_shift: affine shift len {} != raw width {}",
            affine_shift.len(),
            gauge.raw_total(),
        );
        gauge.affine_shift = affine_shift;
        gauge
    }

    /// Single-block affine section.
    pub fn from_block_transform_with_shift(
        transform: Array2<f64>,
        affine_shift: Array1<f64>,
    ) -> Self {
        Self::from_block_transforms_with_shift(&[transform], affine_shift)
    }

    /// Block-upper-triangular section from per-block `V_b` plus
    /// cross-block residualisation stacks `R_{a→b}` — see
    /// [`assemble_block_triangular_t`] for the packing convention.
    pub fn from_v_and_r(v_per_term: &[Array2<f64>], r_per_term: &[Option<Array2<f64>>]) -> Self {
        let raw_widths: Vec<usize> = v_per_term.iter().map(|v| v.nrows()).collect();
        let reduced_widths: Vec<usize> = v_per_term.iter().map(|v| v.ncols()).collect();
        Self {
            t_full: assemble_block_triangular_t(v_per_term, r_per_term),
            affine_shift: Array1::zeros(raw_widths.iter().sum::<usize>()),
            block_starts_raw: starts_from_widths(&raw_widths),
            block_starts_reduced: starts_from_widths(&reduced_widths),
        }
    }

    /// The sum-to-zero (centering) section as a first-class single-block
    /// gauge. `z` is the `(k × (k−1))` reparametrisation matrix returned by
    /// `terms::basis::duchon_thinplate::apply_sum_to_zero_constraint`
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
        let total_raw: usize = raw_widths.iter().sum();
        Self::from_t_with_shift(t_full, raw_widths, reduced_widths, Array1::zeros(total_raw))
    }

    /// Wrap an already-assembled global affine section `β = Tθ + a` given the
    /// per-block raw and reduced width partitions.
    pub fn from_t_with_shift(
        t_full: Array2<f64>,
        raw_widths: &[usize],
        reduced_widths: &[usize],
        affine_shift: Array1<f64>,
    ) -> Self {
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
        assert_eq!(
            affine_shift.len(),
            total_raw,
            "Gauge::from_t_with_shift: affine shift len {} != raw width {total_raw}",
            affine_shift.len(),
        );
        Self {
            t_full,
            affine_shift,
            block_starts_raw: starts_from_widths(raw_widths),
            block_starts_reduced: starts_from_widths(reduced_widths),
        }
    }

    /// Build from a compiled identifiability reparametrisation
    /// (see [`CompiledBlockMap`], implemented for the `CompiledMap` emitted by
    /// the identifiability compiler): `map.raw_from_compiled()` IS the global
    /// triangular `T`, and the block ranges give both partitions. `ordering`
    /// is accepted purely as a length sanity check.
    pub fn from_compiled_map<M: CompiledBlockMap, O>(map: &M, ordering: &[O]) -> Self {
        assert_eq!(
            map.raw_block_ranges().len(),
            map.compiled_block_ranges().len(),
            "Gauge::from_compiled_map: CompiledMap raw_block_ranges len {} != \
             compiled_block_ranges len {}",
            map.raw_block_ranges().len(),
            map.compiled_block_ranges().len(),
        );
        assert_eq!(
            map.raw_block_ranges().len(),
            ordering.len(),
            "Gauge::from_compiled_map: ordering len {} != block count {}",
            ordering.len(),
            map.raw_block_ranges().len(),
        );
        let mut block_starts_raw = Vec::with_capacity(map.raw_block_ranges().len() + 1);
        block_starts_raw.push(0);
        for r in map.raw_block_ranges() {
            block_starts_raw.push(r.end);
        }
        let mut block_starts_reduced = Vec::with_capacity(map.compiled_block_ranges().len() + 1);
        block_starts_reduced.push(0);
        for r in map.compiled_block_ranges() {
            block_starts_reduced.push(r.end);
        }
        let total_raw = block_starts_raw.last().copied().unwrap_or(0);
        Self {
            t_full: map.raw_from_compiled().clone(),
            affine_shift: Array1::zeros(total_raw),
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

    /// Compose a raw design with the section: `X_reduced = X_raw · T`.
    pub fn restrict_design<S: Data<Elem = f64>>(
        &self,
        raw_design: &ArrayBase<S, Ix2>,
    ) -> Array2<f64> {
        let raw_total = self.raw_total();
        assert_eq!(
            raw_design.ncols(),
            raw_total,
            "Gauge::restrict_design: design has {} columns, expected raw width {raw_total}",
            raw_design.ncols(),
        );
        // A trivial section (`T = I`) leaves the design untouched: `X·I = X`
        // bit-for-bit (every off-diagonal `T` entry is an exact zero, the
        // diagonal an exact one, so the reduction is the identity map). The
        // unconstrained Wahba sphere chart hits this on every build, and the
        // skipped GEMM is an `(n × w)·(w × w)` product — ~0.8 s of host
        // matrixmultiply at production shapes (n ≳ 1e5, w ~ 200). Detecting
        // identity costs O(w²), negligible beside the O(n·w²) it elides.
        if self.t_full_is_identity() {
            return raw_design.to_owned();
        }
        fast_ab(raw_design, &self.t_full)
    }

    /// Whether the lift `T` is the exact identity (square with unit diagonal
    /// and zero off-diagonal). When true, `restrict_design`/`restrict_penalty`
    /// are no-ops and skip their GEMMs. The comparison is exact equality, not
    /// a tolerance — only a literal identity short-circuits, so the fast path
    /// is always bit-identical to the full product.
    fn t_full_is_identity(&self) -> bool {
        let (r, c) = self.t_full.dim();
        if r != c {
            return false;
        }
        self.t_full
            .indexed_iter()
            .all(|((i, j), &v)| v == if i == j { 1.0 } else { 0.0 })
    }

    /// Compose a raw design and offset with the affine section:
    /// `X_raw · (Tθ + a) + o_raw = (X_raw · T)θ + (o_raw + X_raw · a)`.
    pub fn restrict_design_and_offset<S: Data<Elem = f64>>(
        &self,
        raw_design: &ArrayBase<S, Ix2>,
        raw_offset: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        assert_eq!(
            raw_design.nrows(),
            raw_offset.len(),
            "Gauge::restrict_design_and_offset: design rows {} != offset len {}",
            raw_design.nrows(),
            raw_offset.len(),
        );
        let reduced_design = self.restrict_design(raw_design);
        let reduced_offset = raw_offset + &raw_design.dot(&self.affine_shift);
        (reduced_design, reduced_offset)
    }

    /// Pull a raw-coordinate quadratic form back to reduced coordinates:
    /// `S_reduced = Tᵀ · S_raw · T`.
    pub fn restrict_penalty<S: Data<Elem = f64>>(
        &self,
        raw_penalty: &ArrayBase<S, Ix2>,
    ) -> Array2<f64> {
        let raw_total = self.raw_total();
        assert_eq!(
            raw_penalty.dim(),
            (raw_total, raw_total),
            "Gauge::restrict_penalty: matrix has shape {:?}, expected ({raw_total}, {raw_total})",
            raw_penalty.dim(),
        );
        // `Tᵀ S T = S` exactly when `T = I` (see `restrict_design`). Skip the
        // two `(w × w)·(w × w)` products on the unconstrained chart.
        if self.t_full_is_identity() {
            return raw_penalty.to_owned();
        }
        let t_s = fast_atb(&self.t_full, raw_penalty);
        fast_ab(&t_s, &self.t_full)
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
        let mut affine_shift = Array1::<f64>::zeros(raw_total + extra_total);
        affine_shift
            .slice_mut(ndarray::s![0..raw_total])
            .assign(&self.affine_shift);
        Self {
            t_full: t,
            affine_shift,
            block_starts_raw,
            block_starts_reduced,
        }
    }

    /// Lift per-block reduced coefficients to per-block raw
    /// coefficients: concatenate into θ, apply `β = T · θ + a`, split at
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
        let beta_full = self.t_full.dot(&theta_full) + &self.affine_shift;
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
    fn identity_section_short_circuits_restrict_bit_exactly() {
        // A trivial section must restrict design/penalty to the *exact* input,
        // matching the full GEMM bit-for-bit while skipping it.
        let gauge = Gauge::identity(&[4]);
        assert!(gauge.t_full_is_identity());

        // An irregular design with values that would perturb under a real GEMM
        // if any rounding crept in.
        let raw_design = Array2::<f64>::from_shape_fn((7, 4), |(i, j)| {
            ((i as f64) * 0.3 - (j as f64) * 1.7).sin() * 1.000000001
        });
        let restricted = gauge.restrict_design(&raw_design);
        // Bit-exact equality with the input (the identity map).
        assert_eq!(restricted, raw_design);
        // And bit-exact with the full product it elides.
        let via_gemm = fast_ab(&raw_design, &gauge.t_full);
        assert_eq!(restricted, via_gemm);

        let raw_penalty = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| {
            (i as f64 + 1.0) * (j as f64 + 2.0) * 0.111
        });
        let restricted_pen = gauge.restrict_penalty(&raw_penalty);
        assert_eq!(restricted_pen, raw_penalty);
        let pen_via_gemm = fast_ab(&fast_atb(&gauge.t_full, &raw_penalty), &gauge.t_full);
        assert_eq!(restricted_pen, pen_via_gemm);
    }

    #[test]
    fn non_identity_section_is_not_short_circuited() {
        // A real reparametrisation must NOT take the identity fast path.
        let mut t = Array2::<f64>::eye(3);
        t[[0, 1]] = 0.5;
        let gauge = Gauge::from_t(t.clone(), &[3], &[3]);
        assert!(!gauge.t_full_is_identity());
        let raw = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| i as f64 + j as f64 * 0.25);
        let restricted = gauge.restrict_design(&raw);
        assert_eq!(restricted, fast_ab(&raw, &t));
    }

    #[test]
    fn rectangular_section_is_not_identity() {
        // A tall centring section is square-free and must never be mistaken
        // for the identity (it removes a direction).
        let z =
            Array2::<f64>::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0]).unwrap();
        let gauge = Gauge::sum_to_zero(z);
        assert!(!gauge.t_full_is_identity());
    }

    #[test]
    fn affine_gauge_lifts_betas_and_restricts_offsets() {
        let t = Array2::from_shape_vec((3, 1), vec![2.0, -1.0, 0.5]).unwrap();
        let shift = Array1::from(vec![0.25, 1.5, -0.75]);
        let gauge = Gauge::from_block_transform_with_shift(t.clone(), shift.clone());
        let theta = Array1::from(vec![4.0]);

        let raw = gauge.lift_block_betas(&[theta.clone()]);
        let expected_raw = t.dot(&theta) + &shift;
        assert_eq!(raw[0], expected_raw);

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, -1.0, 3.0, 0.5]).unwrap();
        let offset = Array1::from(vec![0.1, -0.2]);
        let (x_reduced, offset_reduced) = gauge.restrict_design_and_offset(&x, &offset);
        assert_eq!(x_reduced, x.dot(&t));
        assert_eq!(offset_reduced, &offset + &x.dot(&shift));

        let eta_raw = x.dot(&expected_raw) + &offset;
        let eta_reduced = x_reduced.dot(&theta) + &offset_reduced;
        for i in 0..eta_raw.len() {
            assert!((eta_raw[i] - eta_reduced[i]).abs() < 1e-14);
        }

        let cov_reduced = Array2::from_elem((1, 1), 3.0);
        let lifted_cov = gauge.lift_covariance(&cov_reduced);
        let expected_cov = t.dot(&cov_reduced).dot(&t.t());
        assert_eq!(lifted_cov, expected_cov);
    }

    /// The covariance pushforward of an affine section `β = T·θ + a` must be
    /// EXACTLY independent of the affine shift `a` — `Cov(T·θ + a) = T·Cov(θ)·Tᵀ`
    /// for any constant `a`, because a deterministic offset adds no variance. The
    /// b≡1 unit-log-t pin (#892) folds the warp into `a`; this is the property
    /// that guarantees reporting the pinned coefficients carries the same
    /// posterior uncertainty as the unpinned linear section. We assert it two
    /// ways: (1) the analytic lift is bit-identical across a sweep of shift
    /// magnitudes spanning the zero-shift linear case up to 1e7; and (2) an
    /// empirical check — the sample covariance of `T·θ_k + a` over reduced draws
    /// `θ_k` is unchanged when `a` is replaced by a 1e6-scale offset (the offset
    /// cancels under centering).
    #[test]
    fn affine_shift_leaves_lifted_covariance_invariant() {
        // A non-trivial 4-raw × 2-reduced section (so T mixes coordinates).
        let t =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.5, -1.0, 2.0, 0.3, -0.4, 1.5]).unwrap();
        let raw_widths = [4usize];
        let reduced_widths = [2usize];

        // A non-diagonal reduced covariance.
        let cov_reduced = Array2::from_shape_vec((2, 2), vec![2.0, -0.7, -0.7, 1.3]).unwrap();

        // The reference lift is the zero-shift (purely linear) section.
        let base =
            Gauge::from_t_with_shift(t.clone(), &raw_widths, &reduced_widths, Array1::zeros(4));
        let reference = base.lift_covariance(&cov_reduced);

        // (1) Bit-identical across a wide sweep of shift magnitudes.
        for &mag in &[0.0, 1e-7, 1.0, 1e3, 1e7] {
            let shift = Array1::from(vec![mag, -mag, 0.5 * mag, -2.0 * mag]);
            let gauge = Gauge::from_t_with_shift(t.clone(), &raw_widths, &reduced_widths, shift);
            let lifted = gauge.lift_covariance(&cov_reduced);
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(
                        lifted[[i, j]],
                        reference[[i, j]],
                        "affine shift magnitude {mag} must not perturb the lifted covariance \
                         at ({i},{j}) — covariance is offset-invariant",
                    );
                }
            }
        }

        // (2) Empirical check: draw reduced samples, push them through
        // β = T·θ + a for two very different shifts, and confirm the sample
        // covariance is the same for both shifts. Draws use a fixed Cholesky
        // colouring of cov_reduced so the test is deterministic (no RNG).
        let chol = {
            let l00 = cov_reduced[[0, 0]].sqrt();
            let l10 = cov_reduced[[1, 0]] / l00;
            let l11 = (cov_reduced[[1, 1]] - l10 * l10).sqrt();
            Array2::from_shape_vec((2, 2), vec![l00, 0.0, l10, l11]).unwrap()
        };
        let z_raw = [
            [1.2, -0.4],
            [-0.8, 0.9],
            [0.3, 1.7],
            [-1.5, -0.6],
            [0.6, -1.1],
            [-0.2, 0.3],
            [1.9, 0.2],
            [-1.4, -0.9],
        ];
        let sample_cov_for_shift = |shift: &Array1<f64>| -> Array2<f64> {
            let n = z_raw.len();
            let betas: Vec<Array1<f64>> = z_raw
                .iter()
                .map(|z| {
                    let theta = chol.dot(&Array1::from(vec![z[0], z[1]]));
                    t.dot(&theta) + shift
                })
                .collect();
            let mut mean = Array1::<f64>::zeros(4);
            for b in &betas {
                mean = &mean + b;
            }
            mean /= n as f64;
            let mut cov = Array2::<f64>::zeros((4, 4));
            for b in &betas {
                let c = b - &mean;
                for i in 0..4 {
                    for j in 0..4 {
                        cov[[i, j]] += c[i] * c[j] / n as f64;
                    }
                }
            }
            cov
        };
        let cov_small = sample_cov_for_shift(&Array1::zeros(4));
        let cov_big = sample_cov_for_shift(&Array1::from(vec![1e6, -1e6, 5e5, -2e6]));
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cov_small[[i, j]] - cov_big[[i, j]]).abs() < 1e-6,
                    "empirical sample covariance must be offset-invariant at ({i},{j}): \
                     small-shift {} vs big-shift {}",
                    cov_small[[i, j]],
                    cov_big[[i, j]],
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

    /// For a zero-shift gauge, covariance lift must be the exact pushforward of
    /// the SAME `T` the β lift applies: for a rank-1 `Σ_θ = θθᵀ`, the lifted
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
            assert!(
                (z.column(j).sum()).abs() < 1e-14,
                "column {j} must sum to 0"
            );
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
        assert_eq!(gauge.restrict_design(&b), b_c);
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

        let raw_penalty = Array2::from_shape_vec(
            (3, 3),
            vec![2.0, 0.5, 0.0, 0.5, 3.0, -0.25, 0.0, -0.25, 4.0],
        )
        .unwrap();
        let reduced_penalty = gauge.restrict_penalty(&raw_penalty);
        let expected_reduced_penalty = fast_ab(&fast_atb(&z, &raw_penalty), &z);
        assert_eq!(reduced_penalty, expected_reduced_penalty);
    }

    #[test]
    #[should_panic(expected = "removes at least one direction")]
    fn sum_to_zero_rejects_identity_section() {
        // A square z removes no direction — that is not a centring section.
        drop(Gauge::sum_to_zero(Array2::<f64>::eye(3)));
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
