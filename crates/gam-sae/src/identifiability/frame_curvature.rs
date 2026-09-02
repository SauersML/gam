//! The residual-gauge curvature operator, held in the structure the
//! decoder-frame parameterization actually gives it (#2757).
//!
//! # The theorem this module is built on
//!
//! The certificate's free-parameter vector is the concatenation of the fitted
//! atoms' flattened frames, `vec(frame_0) ⊕ vec(frame_1) ⊕ …`, each
//! `frame_k ∈ ℝ^{p × d_k}` stored row-major. So a parameter index decomposes
//! exactly as
//!
//! ```text
//! c = offset_k + i·d_k + a        atom k, OUTPUT coordinate i, frame axis a
//! ```
//!
//! and the per-row pinning Jacobian is, by construction,
//!
//! ```text
//! J_n[i, (k, i', a)] = δ_{i,i'} · a_{nk} · ∂g_k/∂t_a(n)[i]
//! ```
//!
//! — a frame perturbation of output coordinate `i'` moves the reconstruction
//! only on output coordinate `i'`. `J_n` is therefore **output-coordinate
//! diagonal**: it has `p·D` nonzeros (`D = Σ_k d_k`), not `p · param_dim`.
//!
//! The data curvature `H = Σ_n J_nᵀ M_n J_n` inherits from that exactly the
//! output-coordinate coupling the *metric* has and nothing else:
//!
//! ```text
//! H[(k,i,a), (k',i',a')] = Σ_n M_n[i,i'] · g_n[i,(k,a)] · g_n[i',(k',a')]
//! ```
//!
//! So when every per-row metric is diagonal in the output coordinates — in
//! particular when it is the identity, which is what
//! `SaeManifoldTerm::diagnostic_metric` installs whenever no output-Fisher
//! harvest ran — `H` is **exactly block diagonal**, `p` blocks of size
//! `D × D`, and the off-block entries are not small but structurally never
//! written.
//!
//! Holding that object as a dense `param_dim × param_dim = (p·D)²` matrix and
//! taking its dense symmetric eigendecomposition costs `(p·D)²` memory and
//! `(p·D)³` flops for a quantity that is `p·D²` numbers and `p·D³` flops — a
//! factor of `p` in memory and `p²` in time. At the `p = 4096` shape #2757 was
//! filed on that is 45.1 GiB and ~1.7·10⁷× the necessary work.
//!
//! # The one part the grouping does not diagonalize
//!
//! An installed isometry pin adds one curvature-root row per
//! `(atom, frame axis)`, and each carries that atom's frame column across
//! *every* output coordinate. Those rows cannot be folded into the blocks:
//! eliminating one against block `i` scatters that block's row into every other
//! block, so the QR of `[⊕R_i ; L]` fills in completely and the sum's spectrum
//! is genuinely global.
//!
//! It is exact and cheap regardless. `H = ⊕_i B_i + VVᵀ` is block diagonal plus
//! a symmetric update of rank `k = Σ_k d_k`, and Sylvester's law of inertia
//! gives its eigenvalue count above **any** shift from a `k × k` determinant
//! ([`BlockPlusRowsSpectrum`]) — which is all the certificate ever asks: the
//! pinning rank is the count above `τ²`, and `λ_max` is the shift at which the
//! count reaches zero.
//!
//! # What this module provides
//!
//! * [`FrameColumnLayout`] — the `c ↔ (i, l)` bijection above, the single place
//!   the frame-column index arithmetic is written down.
//! * [`OutputBlockRootAccumulator`] — the streaming builder, which keeps each
//!   block's triangular **root** by Givens rotations rather than its Gram, so
//!   the cheap representation is also the accurate one (a Gram squares the
//!   condition number under a rank tolerance chosen to sit just above an SVD's
//!   backward error).
//! * [`TriangularRootAccumulator`] — the same streaming discipline for the
//!   branch where the metric DOES couple output coordinates and no block
//!   structure survives: the root's rows are folded into one
//!   `param_dim`-square upper-triangular factor `T` with `TᵀT = RᵀR`.
//! * [`ResidualGaugeCurvature`] — the curvature as the builder is able to
//!   produce it: output-coordinate block roots plus the pin's dense rows when
//!   the metric does not couple output coordinates, and otherwise the root —
//!   whole when it has fewer rows than columns, folded into `T` when it has
//!   more. **Every production representation now carries a ROOT**, so every
//!   rank decision is taken on `σ`; [`ResidualGaugeCurvature::DenseGram`]
//!   survives only for callers that hand-build a Gram, and is the one
//!   representation forced onto `λ = σ²`.
//! * [`BlockPlusRowsSpectrum`] — the inertia machinery above.
//!
//! # What is not solved here, and cannot be
//!
//! A metric that couples output coordinates leaves `H` with no exploitable
//! structure at all: it is a sum of `n · metric_rank` rank-one terms in
//! `param_dim` dimensions, and its exact spectrum costs
//! `min(rows, param_dim)²` memory whichever side it is taken from. The fold
//! above removes the factor of two (`eigh` allocates eigenvectors the reduction
//! discards; a values-only SVD allocates none), removes the squared rank
//! tolerance, and removes the second representation — but the surviving
//! `param_dim²` is the price of asking for an exact full spectrum, not an
//! artifact of how it is asked for. At production width that branch needs the
//! certificate to stop asking, which is a change to what
//! `super::residual_gauge_inner` reads rather than to how this module stores.

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use gam_linalg::lanczos::{SymmetricExtremeLanczosOptions, symmetric_extreme_lanczos_eigenpairs};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayViewMut2, s};

/// One atom's slot in the joint frame-column layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameAtomSlot {
    /// `offset_k` — the atom's first column in the joint parameter vector.
    offset: usize,
    /// `d_k` — the atom's frame axis count (its latent dimension).
    axes: usize,
    /// `dstart_k = Σ_{k' < k} d_{k'}` — the atom's first local axis index.
    axis_start: usize,
}

/// A local axis index `l ∈ [0, D)` resolved to the atom slot it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalAxis {
    /// `offset_k` of the owning atom.
    offset: usize,
    /// `d_k` of the owning atom.
    axes: usize,
    /// `a = l − dstart_k`, the axis within that atom's frame.
    axis: usize,
}

/// The joint parameter vector's frame-column layout: the bijection between a
/// parameter index `c` and the pair `(i, l)` of OUTPUT coordinate
/// `i ∈ [0, p)` and LOCAL frame axis `l ∈ [0, D)`, `D = Σ_k d_k`.
///
/// This is the whole content of "the pinning Jacobian is output-coordinate
/// diagonal": every consumer that wants to group parameters by the output
/// coordinate they drive reads the grouping from here rather than
/// re-deriving `offset_k + i·d_k + a` for itself.
///
/// A layout exists only when every atom's frame has the same height `p` —
/// which the fitted model always satisfies, since each frame is built as
/// `(p, d_k)` against the term's own output dimension. A model whose frames
/// disagree in height has no common output coordinate to group by, and
/// [`FrameColumnLayout::for_frames`] reports `None` rather than inventing one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameColumnLayout {
    p: usize,
    d_total: usize,
    param_dim: usize,
    atoms: Vec<FrameAtomSlot>,
    locals: Vec<LocalAxis>,
}

impl FrameColumnLayout {
    /// The layout for atoms whose frames are `(p, axes_per_atom[k])`.
    ///
    /// `p == 0` or an empty atom list still yields a valid (degenerate) layout:
    /// `param_dim == 0`, which every consumer already treats as the empty
    /// certificate.
    pub fn new(p: usize, axes_per_atom: &[usize]) -> Self {
        let mut atoms = Vec::with_capacity(axes_per_atom.len());
        let mut locals = Vec::new();
        let mut offset = 0usize;
        let mut axis_start = 0usize;
        for &axes in axes_per_atom {
            atoms.push(FrameAtomSlot {
                offset,
                axes,
                axis_start,
            });
            for axis in 0..axes {
                locals.push(LocalAxis { offset, axes, axis });
            }
            offset += p * axes;
            axis_start += axes;
        }
        Self {
            p,
            d_total: axis_start,
            param_dim: offset,
            atoms,
            locals,
        }
    }

    /// The layout of a fitted model's frames, or `None` when the frames do not
    /// share one output dimension (no common coordinate to group by).
    pub fn for_frames<'a>(frames: impl IntoIterator<Item = &'a Array2<f64>>) -> Option<Self> {
        let dims: Vec<(usize, usize)> = frames.into_iter().map(|f| f.dim()).collect();
        // An atom with no axes contributes no columns, so its (vacuous) frame
        // height must not veto the layout the remaining atoms agree on.
        let mut p: Option<usize> = None;
        for &(rows, cols) in &dims {
            if cols == 0 {
                continue;
            }
            match p {
                None => p = Some(rows),
                Some(known) if known == rows => {}
                Some(_) => return None,
            }
        }
        let p = p.unwrap_or(0);
        let axes: Vec<usize> = dims.iter().map(|&(_, cols)| cols).collect();
        Some(Self::new(p, &axes))
    }

    /// Output dimension `p`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.p
    }

    /// `D = Σ_k d_k` — the block size, i.e. the number of frame axes that
    /// share each output coordinate.
    #[inline]
    pub fn block_dim(&self) -> usize {
        self.d_total
    }

    /// `param_dim = p · D`.
    #[inline]
    pub fn param_dim(&self) -> usize {
        self.param_dim
    }

    /// Number of atoms.
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Atom `k`'s first local axis index `dstart_k`, so its axis `a` is local
    /// axis `local_axis_base(k) + a`.
    #[inline]
    pub fn local_axis_base(&self, atom: usize) -> usize {
        self.atoms[atom].axis_start
    }

    /// The parameter index `c` of `(output coordinate i, local axis l)`.
    #[inline]
    pub fn column(&self, output: usize, local: usize) -> usize {
        let slot = self.locals[local];
        slot.offset + output * slot.axes + slot.axis
    }

    /// Gather the `D` entries of `v` that live on output coordinate `output`
    /// into `out` (length `D`), in local-axis order.
    #[inline]
    pub fn gather_output(&self, v: ArrayView1<'_, f64>, output: usize, out: &mut [f64]) {
        for (local, slot) in self.locals.iter().enumerate() {
            out[local] = v[slot.offset + output * slot.axes + slot.axis];
        }
    }
}

/// Fold one row `v` into the upper-triangular factor `r` (`D × D`) by Givens
/// rotations, so that afterwards `rᵀr` equals the old `rᵀr + vvᵀ`.
///
/// This is the whole reason the block accumulator keeps a *root* rather than a
/// Gram. The certificate's rank decision is
/// `σ_i(R) > α·ε·max(m, param_dim)·max(σ_max, 1)` — a statement about singular
/// values of `R`. Accumulating `RᵀR` and recovering `σ = √λ` squares the
/// condition number: eigenvalues below `ε·λ_max` are indistinguishable from the
/// decomposition's own roundoff, so every singular value below `√ε·σ_max ≈
/// 1.5e-8·σ_max` is unresolvable while the tolerance asks about ones near
/// `1e-12·σ_max`. Rotating rows in instead keeps `σ` to full precision at the
/// same `O(D²)` per row and the same `p·D²` storage — the cheap representation
/// is also the accurate one.
///
/// `v` is consumed (left as the annihilated residual) so the caller can reuse
/// one buffer.
fn fold_row_into_triangular_factor(r: &mut ArrayViewMut2<'_, f64>, v: &mut [f64]) {
    let d = v.len();
    for j in 0..d {
        let vj = v[j];
        if vj == 0.0 {
            continue;
        }
        let rjj = r[[j, j]];
        let norm = rjj.hypot(vj);
        if norm == 0.0 {
            continue;
        }
        let (c, s) = (rjj / norm, vj / norm);
        r[[j, j]] = norm;
        v[j] = 0.0;
        for k in (j + 1)..d {
            let rjk = r[[j, k]];
            let vk = v[k];
            r[[j, k]] = c * rjk + s * vk;
            v[k] = c * vk - s * rjk;
        }
    }
}

/// Streaming accumulator for the output-coordinate block roots.
///
/// One `D × D` upper-triangular factor per output coordinate, grown by
/// `fold_row_into_triangular_factor` as observations arrive. Memory `p·D²`
/// and cost `O(rows · p · D²)` — the same as accumulating the blocks' Grams,
/// without the squaring.
pub struct OutputBlockRootAccumulator {
    roots: Array3<f64>,
    layout: FrameColumnLayout,
    scratch: Vec<f64>,
}

impl OutputBlockRootAccumulator {
    pub fn new(layout: FrameColumnLayout) -> Self {
        let d = layout.block_dim();
        Self {
            roots: Array3::<f64>::zeros((layout.output_dim(), d, d)),
            layout,
            scratch: vec![0.0_f64; d],
        }
    }

    /// Fold one observation's frame Jacobian `g` (`p × D`, `g[i, l]`) in: it
    /// contributes the rank-one term `g(i)g(i)ᵀ` to output coordinate `i`'s
    /// block and nothing anywhere else.
    pub fn push_row_jacobian(&mut self, g: &Array2<f64>) {
        let d = self.layout.block_dim();
        for i in 0..self.layout.output_dim() {
            let mut any = false;
            for l in 0..d {
                let v = g[[i, l]];
                self.scratch[l] = v;
                any |= v != 0.0;
            }
            if !any {
                continue;
            }
            let mut block = self.roots.slice_mut(s![i, .., ..]);
            fold_row_into_triangular_factor(&mut block, &mut self.scratch);
        }
    }

    /// Close the accumulation. `dense_rows` is the part of the curvature root
    /// that is NOT output-coordinate diagonal — the isometry pin's `Σ_k d_k`
    /// rows, each spread across every output coordinate — as a
    /// `(rows, param_dim)` matrix. Empty when no pin is installed.
    pub fn finish_with_rows(
        self,
        dense_rows: Array2<f64>,
        root_rows: usize,
    ) -> Result<ResidualGaugeCurvature, String> {
        if dense_rows.nrows() > 0 && dense_rows.ncols() != self.layout.param_dim() {
            return Err(format!(
                "residual gauge curvature: dense rows have {} columns but param_dim = {}",
                dense_rows.ncols(),
                self.layout.param_dim()
            ));
        }
        Ok(ResidualGaugeCurvature::OutputBlockRoots {
            roots: self.roots,
            dense_rows,
            layout: self.layout,
            root_rows,
        })
    }

    pub fn finish(self, root_rows: usize) -> ResidualGaugeCurvature {
        let param_dim = self.layout.param_dim();
        self.finish_with_rows(Array2::<f64>::zeros((0, param_dim)), root_rows)
            .expect("an empty dense-row block is conformable with any layout")
    }
}

/// Streaming accumulator for a curvature root with NO output-coordinate
/// structure to exploit — the branch where the metric couples output
/// coordinates, so `H`'s off-block entries are genuinely nonzero.
///
/// # What it replaces and why
///
/// That branch used to fork on whether the root had more rows than columns.
/// With fewer, it kept the root whole. With more — which at any production row
/// count is always, since the root has `n · metric_rank` rows — it assembled the
/// dense `param_dim × param_dim` **Gram** instead, and the certificate then read
/// its spectrum through a symmetric eigendecomposition. That is #2757's own
/// defect, verbatim, surviving on the half of the fork the block-structured
/// curvature never reached.
///
/// The Gram is the wrong object on both counts the certificate cares about:
///
/// * **Resolution.** `gram_spectral_rank` has to
///   take the rank decision on `λ = σ²` against a threshold `τ²` that lands
///   below a symmetric eigensolver's own backward error, so it is floored at
///   `ε·param_dim·λ_max` and cannot resolve below it — at `param_dim = 65 536`
///   that is `1.5e-11·λ_max`, where the root-side decision resolves
///   `(α·ε·N)² ≈ 1e-16·λ_max`. Folding rows in keeps `σ` to full precision
///   instead of squaring the condition number, exactly as
///   [`OutputBlockRootAccumulator`] already argues for the block case.
/// * **Peak memory.** The Gram is `param_dim²`, and `eigh` allocates a second
///   `param_dim²` for eigenvectors the reduction discards. A values-only SVD of
///   the triangular factor allocates neither.
///
/// # What it does NOT fix, stated plainly
///
/// This does not make a coupling metric affordable at production width. `H` for
/// such a metric is a sum of `n · metric_rank` rank-one terms in `param_dim`
/// dimensions with no exploitable structure — the per-row Jacobian is
/// output-coordinate diagonal, but `M_n` couples those coordinates, so the
/// product is dense — and its exact spectrum costs `min(rows, param_dim)²`
/// memory whichever side it is taken from. What this removes is the factor of
/// two, the squared rank tolerance, and the second representation; the
/// remaining `param_dim²` is a property of asking for an exact full spectrum at
/// all, not of how it is asked for.
pub struct TriangularRootAccumulator {
    factor: Array2<f64>,
}

impl TriangularRootAccumulator {
    pub fn new(param_dim: usize) -> Self {
        Self {
            factor: Array2::<f64>::zeros((param_dim, param_dim)),
        }
    }

    /// Fold one root row into the factor. `row` is consumed (left as the
    /// annihilated residual) so the caller may reuse one buffer.
    ///
    /// A row of the wrong width is a caller that built its root against a
    /// different parameterization, which the fold would silently absorb into
    /// the leading columns; it is refused rather than folded.
    pub fn push_root_row(&mut self, row: &mut [f64]) -> Result<(), String> {
        if row.len() != self.factor.ncols() {
            return Err(format!(
                "residual gauge curvature: root row has {} entries but the factor is over {} \
                 parameters",
                row.len(),
                self.factor.ncols()
            ));
        }
        let mut view = self.factor.view_mut();
        fold_row_into_triangular_factor(&mut view, row);
        Ok(())
    }

    /// Close the accumulation. `root_rows` is the number of rows that were
    /// folded in — the true row count of `R`, which is what sets the rank
    /// tolerance's scale, not the `param_dim` rows the factor happens to have.
    pub fn finish(self, root_rows: usize) -> ResidualGaugeCurvature {
        ResidualGaugeCurvature::DualRoot {
            root: self.factor,
            root_rows,
        }
    }

    /// The accumulated upper-triangular factor itself, `T` with `TᵀT = RᵀR`.
    ///
    /// Used where the folded object is not a curvature over the model's own
    /// parameters but over a small basis of them — the `G × G` factor of `R Ξ`
    /// that [`StreamedFrameCurvature::project_root`] returns — so wrapping it in
    /// a [`ResidualGaugeCurvature`] would misdescribe what it is a curvature of.
    pub fn into_factor(self) -> Array2<f64> {
        self.factor
    }

    /// Fold another accumulator of the same width into this one, so that
    /// afterwards `TᵀT` is the sum of the two factors' Grams.
    ///
    /// This is what makes the fold associative and therefore splittable: a
    /// stream can be cut into disjoint pieces, each folded independently, and
    /// merged — with the same arithmetic, since a merge is just folding the
    /// other factor's rows in one at a time. It is the reason a parallel pass
    /// over observations produces the same operator as a serial one (not the
    /// same bits — Givens rotations do not commute — but the same `TᵀT` to
    /// rounding, which is the object every consumer reads).
    pub fn merge(&mut self, other: Self) -> Result<(), String> {
        if other.factor.ncols() != self.factor.ncols() {
            return Err(format!(
                "residual gauge curvature: cannot merge a factor over {} parameters into one \
                 over {}",
                other.factor.ncols(),
                self.factor.ncols()
            ));
        }
        let mut row = vec![0.0_f64; self.factor.ncols()];
        for r in 0..other.factor.nrows() {
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = other.factor[[r, c]];
            }
            self.push_root_row(&mut row)?;
        }
        Ok(())
    }
}

/// The exact spectrum machinery for `H = ⊕_i B_i + VVᵀ` — a block-diagonal
/// operator plus a symmetric update of rank `k = V.ncols()`.
///
/// # Why this exists
///
/// The isometry pin's curvature root has one row per `(atom, frame axis)` and
/// each of those rows is spread across *every* output coordinate, so it is
/// exactly the part of `R` that the output-coordinate grouping does not
/// diagonalize. It cannot be folded into the blocks: a Givens elimination of
/// such a row against block `i` scatters that block's row into every other
/// block, so the QR of `[⊕R_i ; L]` fills in completely. The rank and `σ_max`
/// of the sum genuinely are global quantities.
///
/// They are nevertheless **exactly computable in `O(p·D·k²)`**, without forming
/// `H`, from Sylvester's law of inertia. Border the shifted operator:
///
/// ```text
/// M = [ B − sI   V ]      Schur on (1,1):  In(M) = In(B−sI) + In(−I − Vᵀ(B−sI)⁻¹V)
///     [   Vᵀ    −I ]      Schur on (2,2):  In(M) = In(−I_k) + In(B − sI + VVᵀ)
/// ```
///
/// so, since `In(−I_k) = (0, 0, k)`,
///
/// ```text
/// n₊(H − sI) = n₊(B − sI) + n₊(−I_k − Vᵀ(B − sI)⁻¹V)
/// ```
///
/// The first term is a comparison against the block eigenvalues, which are
/// already known. The second is the positive-eigenvalue count of a `k × k`
/// matrix, and `Vᵀ(B−sI)⁻¹V` is assembled blockwise from each block's own
/// eigenbasis. Counting eigenvalues above an arbitrary shift is therefore
/// cheap and exact — which gives BOTH consumers:
///
/// * the pinning rank is `n₊(H − τ²I)` at the rank tolerance `τ`;
/// * `λ_max(H)` is `inf{ s : n₊(H − sI) = 0 }`, found by bisection on the same
///   counter, with `λ_max(B) ≤ λ_max(H) ≤ λ_max(B) + ‖V‖_F²` as the bracket.
///
/// Both are the *same* decisions the dense `param_dim × param_dim`
/// eigendecomposition would make, at `p·D·k²` instead of `(p·D)³`.
pub struct BlockPlusRowsSpectrum {
    /// `(p, D)` — the eigenvalues of each block `B_i = R_iᵀR_i`, obtained as
    /// the squared singular values of `R_i` so they are non-negative by
    /// construction rather than by rounding.
    block_eigenvalues: Array2<f64>,
    /// `(p, k, D)` — `Q_iᵀ v_j(i)`, each update column expressed in each
    /// block's eigenbasis. Independent of the shift, so it is built once.
    projected: Array3<f64>,
    update_rank: usize,
    block_lambda_max: f64,
    update_norm_sq: f64,
}

impl BlockPlusRowsSpectrum {
    /// Decompose every block root and project the update columns into the
    /// blocks' eigenbases. `O(p·D³ + p·k·D²)`.
    pub fn new(
        roots: &Array3<f64>,
        dense_rows: &Array2<f64>,
        layout: &FrameColumnLayout,
    ) -> Result<Self, String> {
        let p = layout.output_dim();
        let d = layout.block_dim();
        let k = dense_rows.nrows();
        let mut block_eigenvalues = Array2::<f64>::zeros((p, d));
        let mut projected = Array3::<f64>::zeros((p, k, d));
        let mut gathered = vec![0.0_f64; d];
        let mut block_lambda_max = 0.0_f64;
        for i in 0..p {
            let block = roots.slice(s![i, .., ..]);
            // `B_i = R_iᵀR_i = V Σ² Vᵀ` for `R_i = U Σ Vᵀ`, so the block's
            // eigenvalues are the squared singular values (non-negative by
            // construction) and its eigenvectors are the right singular vectors.
            //
            // `None` means "the identity is already an eigenbasis": a zero
            // block, whose eigenvalues are all zero so ANY orthonormal basis
            // diagonalizes it, and a 1×1 block, whose eigenvector is ±1 and
            // whose sign cancels because only products of two projections are
            // ever read. A zero block must still be projected — its output
            // coordinate lies entirely in `null(B)`, which is exactly where the
            // update's mass decides the rank, so skipping it would silently
            // drop that mass (measured: `λ_max` short by 13% on a fixture with
            // one empty output coordinate).
            let basis_t = if block.iter().all(|v| *v == 0.0) || d == 1 {
                if d == 1 {
                    block_eigenvalues[[i, 0]] = block[[0, 0]] * block[[0, 0]];
                    block_lambda_max = block_lambda_max.max(block_eigenvalues[[i, 0]]);
                }
                None
            } else {
                let (_u, sv, vt) = block.to_owned().svd(false, true).map_err(|e| {
                    format!("residual gauge curvature: SVD of block {i} failed: {e}")
                })?;
                let vt = vt.ok_or_else(|| {
                    format!("residual gauge curvature: block {i} SVD returned no right factor")
                })?;
                for (t, sigma) in sv.iter().enumerate() {
                    block_eigenvalues[[i, t]] = sigma * sigma;
                    block_lambda_max = block_lambda_max.max(block_eigenvalues[[i, t]]);
                }
                Some(vt)
            };
            for j in 0..k {
                layout.gather_output(dense_rows.row(j), i, &mut gathered);
                match &basis_t {
                    // Row `t` of `Vᵀ` is eigenvector `t`, so `(Qᵀw)[t] = Vᵀ w`.
                    Some(vt) => {
                        for t in 0..d {
                            let mut acc = 0.0_f64;
                            for a in 0..d {
                                acc += vt[[t, a]] * gathered[a];
                            }
                            projected[[i, j, t]] = acc;
                        }
                    }
                    None => {
                        for t in 0..d {
                            projected[[i, j, t]] = gathered[t];
                        }
                    }
                }
            }
        }
        let update_norm_sq = dense_rows.iter().map(|v| v * v).sum::<f64>();
        Ok(Self {
            block_eigenvalues,
            projected,
            update_rank: k,
            block_lambda_max,
            update_norm_sq,
        })
    }

    /// `n₊(H − sI)` — how many eigenvalues of `H` exceed `shift`.
    ///
    /// `shift` must not sit on a block eigenvalue, where `(B − sI)⁻¹` does not
    /// exist. A shift within rounding of one is nudged just past it.
    ///
    /// The collision test is **relative to the eigenvalue**, not to the
    /// operator's overall scale, and that distinction is load-bearing. The rank
    /// shift is `τ² ≈ (α·ε·N)²·λ_max`, which at `N = 10⁵` is `~10⁻²⁵·λ_max` —
    /// so an absolute guard of even `2⁻⁴⁰·λ_max` would declare every
    /// structurally zero block "in collision" with it and push the shift up
    /// thirteen orders of magnitude, silently dropping every genuine singular
    /// value in between from the reported rank. A relative guard fires only
    /// where `λ − s` has actually lost its significant digits, which is the only
    /// place the identity is in trouble and is already a tie the certificate
    /// breaks arbitrarily.
    pub fn count_above(&self, shift: f64) -> Result<usize, String> {
        let (p, d) = self.block_eigenvalues.dim();
        let mut shift = shift;
        // Each pass clears the highest colliding eigenvalue; a handful suffices
        // because the shift only ever moves upward past distinct values.
        for _ in 0..8 {
            let mut collided: Option<f64> = None;
            for lambda in self.block_eigenvalues.iter() {
                let tol = 8.0 * f64::EPSILON * lambda.abs().max(shift.abs());
                if (lambda - shift).abs() <= tol {
                    collided = Some(collided.map_or(*lambda, |worst: f64| worst.max(*lambda)));
                }
            }
            let Some(lambda) = collided else { break };
            let step = (16.0 * f64::EPSILON * lambda.abs().max(shift.abs()))
                .max(f64::MIN_POSITIVE * 16.0);
            shift = lambda + step;
        }
        let mut count = 0usize;
        for i in 0..p {
            for t in 0..d {
                if self.block_eigenvalues[[i, t]] > shift {
                    count += 1;
                }
            }
        }
        let k = self.update_rank;
        if k == 0 {
            return Ok(count);
        }
        // `reduced = −I_k − Vᵀ(B − sI)⁻¹V`, assembled blockwise in each block's
        // eigenbasis where the inverse is diagonal.
        let mut reduced = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            reduced[[j, j]] = -1.0;
        }
        for i in 0..p {
            for t in 0..d {
                let denom = self.block_eigenvalues[[i, t]] - shift;
                if denom == 0.0 {
                    return Err(
                        "residual gauge curvature: inertia shift collides with a block eigenvalue"
                            .to_string(),
                    );
                }
                let inv = 1.0 / denom;
                for j in 0..k {
                    let pj = self.projected[[i, j, t]];
                    if pj == 0.0 {
                        continue;
                    }
                    for l in 0..=j {
                        reduced[[j, l]] -= pj * self.projected[[i, l, t]] * inv;
                    }
                }
            }
        }
        for j in 0..k {
            for l in 0..j {
                reduced[[l, j]] = reduced[[j, l]];
            }
        }
        let (evals, _) = reduced.eigh(faer::Side::Lower).map_err(|e| {
            format!("residual gauge curvature: inertia of the {k}x{k} reduced matrix failed: {e}")
        })?;
        count += evals.iter().filter(|v| **v > 0.0).count();
        Ok(count)
    }

    /// `λ_max(H)`, by bisection on [`Self::count_above`].
    ///
    /// `λ_max(B) ≤ λ_max(H) ≤ λ_max(B) + ‖V‖_F²` because `VVᵀ ⪰ 0` and
    /// `λ_max(VVᵀ) ≤ tr(VVᵀ) = ‖V‖_F²`, so the bracket is valid without any
    /// spectral information about the update.
    pub fn lambda_max(&self) -> Result<f64, String> {
        if self.update_rank == 0 || self.update_norm_sq == 0.0 {
            return Ok(self.block_lambda_max);
        }
        let mut lo = self.block_lambda_max;
        let mut hi = self.block_lambda_max + self.update_norm_sq;
        if self.count_above(lo)? == 0 {
            return Ok(lo);
        }
        // 100 halvings drives the bracket below any representable relative
        // width; the loop exits on the width test long before that, and the
        // bound only exists so a pathological counter cannot spin.
        for _ in 0..100 {
            if hi - lo <= f64::EPSILON * hi.abs().max(1.0) {
                break;
            }
            let mid = lo + 0.5 * (hi - lo);
            if mid <= lo || mid >= hi {
                break;
            }
            if self.count_above(mid)? > 0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(hi)
    }

}

/// The residual-gauge curvature `H = RᵀR` as the streaming builder produced it.
///
/// Every variant carries `root_rows` — the row count `R` would have had — which
/// is the scale the rank tolerance is calibrated against, and is therefore part
/// of the object rather than a parameter threaded beside it.
///
/// Two of the three variants hold a *root* rather than a Gram. That is not an
/// implementation preference: the certificate's rank decision is about singular
/// values of `R`, and forming `RᵀR` squares the condition number before that
/// decision is taken. [`Self::DenseGram`] is the only representation that
/// cannot avoid it, and it is also the only one that costs `param_dim³`.
pub enum ResidualGaugeCurvature {
    /// `p` independent `D × D` upper-triangular roots, one per output
    /// coordinate, in `layout`'s `(i, l)` coordinates:
    /// `roots[i]ᵀ roots[i]` is the data curvature's block at output coordinate
    /// `i`, and the data curvature has no entries between two output
    /// coordinates at all.
    ///
    /// Produced exactly when the per-row metric does not couple output
    /// coordinates.
    OutputBlockRoots {
        roots: Array3<f64>,
        /// The rows of `R` that are NOT output-coordinate diagonal: the
        /// isometry pin contributes one per `(atom, frame axis)`, each carrying
        /// that atom's frame column across every output coordinate. Empty
        /// (`0 × param_dim`) when no pin is installed, which is exactly the
        /// certificate's `diffeomorphism-unpinned` escalation condition.
        ///
        /// `H = ⊕_i R_iᵀR_i + dense_rowsᵀ·dense_rows` — block diagonal plus a
        /// symmetric update of rank at most `Σ_k d_k`, which is a shape whose
        /// spectrum is exactly computable without ever forming `H`
        /// ([`BlockPlusRowsSpectrum`]).
        dense_rows: Array2<f64>,
        layout: FrameColumnLayout,
        root_rows: usize,
    },
    /// The stacked root `R ∈ ℝ^{m × param_dim}` with `m ≤ param_dim`.
    ///
    /// `H = RᵀR` has `param_dim − m` structural zero eigenvalues and `R`'s `m`
    /// singular values, so the rank decision and `σ_max` come from an `m`-sized
    /// decomposition instead of a `param_dim`-sized one.
    DualRoot { root: Array2<f64>, root_rows: usize },
    /// The dense Gram, for a curvature with neither structure.
    ///
    /// The only variant whose rank decision must be taken on `λ = σ²`, and
    /// therefore the only one that cannot resolve a singular value below
    /// `√ε·σ_max`. It is reached only when the root is the *larger* object,
    /// which is also when there is nothing cheaper to decompose.
    DenseGram { gram: Array2<f64>, root_rows: usize },
}

impl ResidualGaugeCurvature {
    /// The row count of the root `R` this curvature came from.
    pub fn root_rows(&self) -> usize {
        match self {
            Self::OutputBlockRoots { root_rows, .. }
            | Self::DualRoot { root_rows, .. }
            | Self::DenseGram { root_rows, .. } => *root_rows,
        }
    }

    /// Whether every stored entry is finite.
    ///
    /// The certificate's refusal contract: a non-finite curvature must produce
    /// a typed error, not a spectrum. Checked on the stored representation
    /// rather than on the fit that produced it, so it holds for every builder.
    pub fn is_finite(&self) -> bool {
        match self {
            Self::OutputBlockRoots {
                roots, dense_rows, ..
            } => roots.iter().all(|v| v.is_finite()) && dense_rows.iter().all(|v| v.is_finite()),
            Self::DualRoot { root, .. } => root.iter().all(|v| v.is_finite()),
            Self::DenseGram { gram, .. } => gram.iter().all(|v| v.is_finite()),
        }
    }

    /// The parameter dimension this curvature is defined over.
    pub fn param_dim(&self) -> usize {
        match self {
            Self::OutputBlockRoots { layout, .. } => layout.param_dim(),
            Self::DualRoot { root, .. } => root.ncols(),
            Self::DenseGram { gram, .. } => gram.ncols(),
        }
    }

}

// ============================================================================
// #2757 — the curvature as an OPERATOR, for the branch where no materialized
// representation of `H` is smaller than `param_dim²`.
// ============================================================================

/// The residual-gauge curvature `H = RᵀR`, exposed as an operator over a root
/// that is **re-streamed on demand** instead of stored.
///
/// # Why a fourth representation, and why it is not a storage format
///
/// The other three are storage: a shape `H` happens to have, held in the fewest
/// scalars that shape allows. This one is the statement that `H` has no such
/// shape. With a per-row metric that couples output coordinates,
/// `H = Σ_n J_nᵀ M_n J_n` is a sum of `n · metric_rank` rank-one terms in
/// `param_dim` dimensions whose only exploitable structure — the output-coordinate
/// diagonality of `J_n` — is destroyed by `M_n`. Every materialized form of it
/// costs `min(root_rows, param_dim)²` scalars, and an exact full spectrum costs
/// the cube from whichever side it is taken.
///
/// So the fix is not another storage format. It is that **the certificate stops
/// asking for a full spectrum**. Of the three things it reads off `H`
///
/// * `ξᵀHξ` along each enumerated generator — the numerator of every verdict,
/// * `λ_max(H)` — the denominator of every verdict,
/// * the pinning rank — reported, and read by no verdict,
///
/// the first two are *streamable*: the first exactly, in one pass
/// ([`Self::project_root`]); the second to a certified relative residual by a
/// matrix-free Krylov method ([`streamed_lambda_max`]) whose only interaction
/// with `H` is [`Self::apply`]. The third is not, and the certificate says so
/// rather than reporting a number it did not measure — see
/// [`PinningRankSupport`](super::PinningRankSupport).
///
/// # The contract
///
/// `H` must be positive semi-definite and identical across calls: [`Self::apply`]
/// and [`Self::project_root`] are two readings of ONE operator, and
/// [`streamed_lambda_max`] cross-checks them against
/// [`Self::diagonal`]'s trace, which is a rigorous upper bound on `λ_max` for a
/// PSD operator. An implementation whose three readings disagree is refused
/// rather than certified.
pub trait StreamedFrameCurvature: Sync {
    /// The parameter dimension `H` acts on.
    fn param_dim(&self) -> usize;

    /// The number of rows the root `R` would have had — the scale the rank
    /// tolerance is calibrated against, exactly as for the stored variants.
    fn root_rows(&self) -> usize;

    /// `y ← H x`. `y` has length `param_dim` and is fully overwritten.
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), String>;

    /// `diag(H)`, in one pass.
    ///
    /// For a PSD operator `tr(H) = Σ diag(H)` brackets the top of the spectrum
    /// two-sidedly — `tr(H)/param_dim ≤ λ_max ≤ tr(H)` — which is what makes the
    /// Krylov solve's breakdown threshold and its acceptance check
    /// scale-relative rather than absolute, and what lets an identically-zero
    /// curvature be recognised exactly instead of iterated on.
    fn diagonal(&self) -> Result<Array1<f64>, String>;

    /// The `G × G` upper-triangular factor `T` of `W = R Ξ`, where `Ξ` has the
    /// supplied directions as its columns: `TᵀT = WᵀW = ΞᵀHΞ`, in ONE pass.
    ///
    /// Returning the ROOT `T` rather than the Gram `ΞᵀHΞ` is the same decision
    /// [`OutputBlockRootAccumulator`] and [`TriangularRootAccumulator`] take, for
    /// the same reason: the rank question the certificate asks is about singular
    /// values, and forming the Gram squares the condition number before it is
    /// asked. The energies are the factor's own column norms,
    /// `ξ_jᵀHξ_j = Σ_a T[a, j]²`, so one object answers both.
    fn project_root(&self, directions: &[ArrayView1<'_, f64>]) -> Result<Array2<f64>, String>;
}

/// What a matrix-free Krylov read of a streamed curvature resolved.
#[derive(Debug, Clone, Copy)]
pub struct StreamedLambdaMax {
    /// `λ_max(H)`, as the largest Ritz value of the certified Krylov solve. A
    /// Rayleigh quotient of an explicit vector, hence a LOWER bound on the true
    /// `λ_max` up to the solve's own rounding.
    pub lambda_max: f64,
    /// `β_k·|e_kᵀy|` for the returned pair, relative to `max(λ_max, 1)` — the
    /// sharp Ritz residual the solve certified against.
    pub relative_residual: f64,
    /// `tr(H)`, the rigorous PSD upper bound `λ_max` was checked against.
    pub trace: f64,
    /// How many passes over the root the solve took: one for the diagonal, then
    /// one per Krylov step. This is the certificate's whole cost on this route
    /// besides the single generator-projection pass, so it is reported rather
    /// than left to a profiler — a solve that needed hundreds of passes would be
    /// a spectrum this instrument is wrong for, and the number says so.
    pub passes: usize,
}

/// The relative accuracy the certificate requires of a streamed `λ_max`.
///
/// `λ_max` is the DENOMINATOR of every generator verdict, which compares
/// `ξᵀHξ/λ_max` against a tolerance of
/// [`GENERATOR_FLAT_ENERGY_TOL`](super::GENERATOR_FLAT_ENERGY_TOL) `= 1e-3`, so a
/// relative error `δ` in `λ_max` moves every fraction by `δ`. The requirement is
/// therefore bracketed from both sides rather than chosen:
///
/// * it cannot be tighter than `≈ ε`, the attainable accuracy of a Ritz value of
///   an operator of norm `λ_max` (`|θ̃ − θ| ≲ ε‖H‖`);
/// * it must be far looser than nothing and far tighter than the `1e-3` the
///   verdict resolves.
///
/// `√ε ≈ 1.49e-8` is the standard residual-based stopping point that sits five
/// orders below the consumer tolerance and eight above the instrument floor —
/// the same one [`gam_terms`'s Duchon spectral chart] asks of the same solver.
/// A solve that cannot reach it is a REFUSAL, never a looser certificate.
fn streamed_lambda_max_relative_tol() -> f64 {
    f64::EPSILON.sqrt()
}

/// The deterministic Krylov start vector.
///
/// Reproducibility is a property the certificate must have: two runs of the same
/// fit have to certify the same group, and a start vector selects a
/// finite-precision Krylov chart. This is the same tiny LCG
/// `gam_terms::basis::duchon_thinplate` uses on the same solver, so the two
/// callers do not each invent a source of randomness.
fn streamed_lanczos_start(dim: usize) -> Vec<f64> {
    let mut state = 1_u64;
    let mut start = vec![0.0_f64; dim];
    for value in &mut start {
        state = (state * 106 + 1283) % 6075;
        *value = state as f64 / 6075.0 - 0.5;
    }
    start
}

/// `λ_max(H)` for a streamed curvature, matrix-free and certified.
///
/// # Method
///
/// One pass computes `diag(H)`. For a PSD operator its sum is `tr(H)`, and
/// `tr(H) = 0 ⟺ H = 0` — so an identically-flat curvature is recognised exactly,
/// in one pass, with no iteration and no tolerance. Otherwise the trace is
/// * the SCALE for the Krylov breakdown threshold (an absolute threshold on a
///   residual norm is meaningless for an operator whose norm is unknown), and
/// * a rigorous UPPER bound on `λ_max`, so a returned Ritz value above it is a
///   contradiction between [`StreamedFrameCurvature::apply`] and
///   [`StreamedFrameCurvature::diagonal`] and is refused rather than reported.
///
/// The solve itself is [`symmetric_extreme_lanczos_eigenpairs`] with full
/// reorthogonalization, which stops at the first step whose sharp Ritz residual
/// `β_k|e_kᵀy|` certifies the pair and errors if it never does. Its step budget
/// is `min(param_dim, root_rows)`: the Krylov space of `H = RᵀR` cannot exceed
/// `rank(H) ≤ min(param_dim, root_rows)` dimensions, so that bound is the exact
/// one and not a guess — and a solve that ran to it would have retained the
/// `param_dim²` basis this route exists to avoid, which is why the honest outcome
/// there is the solver's own refusal.
pub fn streamed_lambda_max(
    operator: &dyn StreamedFrameCurvature,
) -> Result<StreamedLambdaMax, String> {
    let dim = operator.param_dim();
    if dim == 0 || operator.root_rows() == 0 {
        return Ok(StreamedLambdaMax {
            lambda_max: 0.0,
            relative_residual: 0.0,
            trace: 0.0,
            passes: 0,
        });
    }
    let diagonal = operator.diagonal()?;
    if diagonal.len() != dim {
        return Err(format!(
            "streamed curvature: diagonal has {} entries but param_dim = {dim}",
            diagonal.len()
        ));
    }
    if let Some(bad) = diagonal.iter().find(|v| !v.is_finite() || **v < 0.0) {
        return Err(format!(
            "streamed curvature: diag(H) must be finite and non-negative for a PSD \
             curvature; found {bad:.6e}"
        ));
    }
    let trace = diagonal.iter().sum::<f64>();
    if trace == 0.0 {
        // A PSD operator with zero trace is the zero operator: every diagonal
        // entry is a squared norm. No iteration can add to that.
        return Ok(StreamedLambdaMax {
            lambda_max: 0.0,
            relative_residual: 0.0,
            trace: 0.0,
            passes: 1,
        });
    }
    let steps = dim.min(operator.root_rows()).max(1);
    let check_every = 10usize.min((dim / 10).max(1));
    let start = streamed_lanczos_start(dim);
    let mut matvecs = 0usize;
    let pairs = symmetric_extreme_lanczos_eigenpairs(
        dim,
        &start,
        SymmetricExtremeLanczosOptions {
            target_rank: 1,
            max_steps: steps,
            check_every,
            relative_residual_tol: streamed_lambda_max_relative_tol(),
            breakdown_tol: f64::EPSILON * trace,
        },
        |q, image| {
            matvecs += 1;
            operator.apply(q, image)
        },
    )
    .map_err(|e| format!("streamed curvature: λ_max solve did not certify: {e}"))?;
    let lambda_max = pairs.eigenvalues[0];
    let residual = pairs.residual_bounds[0];
    if !lambda_max.is_finite() {
        return Err("streamed curvature: λ_max solve returned a non-finite Ritz value".to_string());
    }
    // `H` is a Gram, so its spectrum is non-negative and bounded by its trace.
    // Both ends are checked: the operator and the diagonal are two independent
    // readings of the same object, and this is the one place they can be
    // compared without materializing anything.
    let slack = f64::EPSILON * (dim as f64) * trace;
    if lambda_max < -slack || lambda_max > trace + slack {
        return Err(format!(
            "streamed curvature: λ_max = {lambda_max:.6e} is outside the PSD bracket \
             [0, tr(H) = {trace:.6e}] its own diagonal gives; the operator's matvec and \
             its diagonal disagree"
        ));
    }
    let lambda_max = lambda_max.clamp(0.0, trace);
    Ok(StreamedLambdaMax {
        lambda_max,
        relative_residual: residual / lambda_max.max(1.0),
        trace,
        passes: matvecs + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_column_and_output_are_inverse() {
        let layout = FrameColumnLayout::new(5, &[2, 1, 3]);
        assert_eq!(layout.param_dim(), 5 * 6);
        assert_eq!(layout.block_dim(), 6);
        for i in 0..5 {
            for l in 0..6 {
                let c = layout.column(i, l);
                assert_eq!(layout.output_of(c), Some(i), "column {c} = (i={i}, l={l})");
            }
        }
        // The map is a bijection onto [0, param_dim).
        let mut seen = vec![false; layout.param_dim()];
        for i in 0..5 {
            for l in 0..6 {
                let c = layout.column(i, l);
                assert!(!seen[c], "column {c} produced twice");
                seen[c] = true;
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn layout_matches_the_certificate_index_arithmetic() {
        // `offset_k + i·d_k + a`, written out independently.
        let axes = [2usize, 1, 3];
        let p = 4;
        let layout = FrameColumnLayout::new(p, &axes);
        let mut offset = 0usize;
        for (k, &d) in axes.iter().enumerate() {
            for i in 0..p {
                for a in 0..d {
                    let expected = offset + i * d + a;
                    let local = layout.local_axis_base(k) + a;
                    assert_eq!(layout.column(i, local), expected);
                }
            }
            offset += p * d;
        }
    }

    #[test]
    fn layout_for_frames_rejects_disagreeing_frame_heights() {
        let a = Array2::<f64>::zeros((4, 2));
        let b = Array2::<f64>::zeros((3, 1));
        assert!(FrameColumnLayout::for_frames([&a, &b]).is_none());
        let c = Array2::<f64>::zeros((4, 1));
        let layout = FrameColumnLayout::for_frames([&a, &c]).expect("agreeing heights");
        assert_eq!(layout.output_dim(), 4);
        assert_eq!(layout.block_dim(), 3);
        assert_eq!(layout.param_dim(), 12);
    }

    #[test]
    fn layout_for_frames_ignores_an_axisless_atom_height() {
        let empty = Array2::<f64>::zeros((0, 0));
        let real = Array2::<f64>::zeros((6, 2));
        let layout = FrameColumnLayout::for_frames([&empty, &real]).expect("layout");
        assert_eq!(layout.output_dim(), 6);
        assert_eq!(layout.block_dim(), 2);
        assert_eq!(layout.param_dim(), 12);
        assert_eq!(layout.local_axis_base(1), 0);
    }

    #[test]
    fn folding_rows_into_a_triangular_factor_reproduces_their_gram() {
        // The Givens accumulator's whole contract: after folding rows
        // `v_1 … v_m`, `rᵀr = Σ_j v_j v_jᵀ` — exactly the Gram it replaces,
        // without ever forming it.
        let d = 5usize;
        let mut seed = 0x2757_ACC0_0000_0001u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let rows: Vec<Vec<f64>> = (0..17).map(|_| (0..d).map(|_| next()).collect()).collect();
        let mut expected = Array2::<f64>::zeros((d, d));
        for row in &rows {
            for a in 0..d {
                for b in 0..d {
                    expected[[a, b]] += row[a] * row[b];
                }
            }
        }
        let mut factor = Array2::<f64>::zeros((d, d));
        for row in &rows {
            let mut v = row.clone();
            fold_row_into_triangular_factor(&mut factor.view_mut(), &mut v);
        }
        // Upper triangular by construction.
        for a in 0..d {
            for b in 0..a {
                assert_eq!(factor[[a, b]], 0.0, "({a},{b}) below the diagonal");
            }
        }
        let recovered = factor.t().dot(&factor);
        let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = recovered
            .iter()
            .zip(expected.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= 1.0e-13 * scale,
            "rᵀr must equal the accumulated Gram: worst |Δ| {worst:.3e} against {scale:.3e}"
        );
    }

    #[test]
    fn folding_survives_magnitudes_that_overflow_a_gram() {
        // `hypot` is not a convenience here: a Gram accumulation squares its
        // entries, so a decoder tangent near `1e200` overflows to infinity
        // before any decomposition sees it. The root accumulation never
        // squares, so the same rows stay finite and the singular value is
        // recovered exactly.
        let mut factor = Array2::<f64>::zeros((2, 2));
        for scale in [1.0e200_f64, 2.0e200] {
            let mut v = vec![scale, 0.0];
            fold_row_into_triangular_factor(&mut factor.view_mut(), &mut v);
        }
        assert!(
            factor.iter().all(|v| v.is_finite()),
            "the triangular factor must stay finite where the Gram would overflow"
        );
        let expected = (1.0e200_f64).hypot(2.0e200);
        assert!(
            ((factor[[0, 0]] - expected) / expected).abs() <= 1.0e-14,
            "σ = {} against {expected}",
            factor[[0, 0]]
        );
        assert!(
            !(1.0e200_f64 * 1.0e200).is_finite(),
            "the Gram entry does overflow"
        );
    }

    #[test]
    fn the_inertia_shift_guard_is_relative_to_the_eigenvalue_not_the_operator() {
        // Four 1x1 blocks with eigenvalues {1, 0, 0, 1e-14} and a negligible
        // rank-one update, counted above 1e-20. The answer is 2, analytically:
        // the update's squared norm is 4e-60, far below every gap.
        //
        // This is the regression for an ABSOLUTE collision guard. One of
        // `2^-40 * lambda_max` would see the zero blocks as colliding with a
        // shift of 1e-20 and push the shift to ~1.8e-12 -- past the genuine
        // 1e-14 eigenvalue, which then vanishes from the reported rank. That is
        // not hypothetical at production scale: the rank shift is
        // `tau^2 ~ (alpha*eps*N)^2 * lambda_max`, which at `N = 1e5` is
        // `1e-25 * lambda_max`, thirteen orders below such a guard.
        let layout = FrameColumnLayout::new(4, &[1]);
        let mut roots = Array3::<f64>::zeros((4, 1, 1));
        roots[[0, 0, 0]] = 1.0;
        roots[[3, 0, 0]] = 1.0e-7;
        let dense_rows = Array2::<f64>::from_elem((1, layout.param_dim()), 1.0e-30);
        let spectrum =
            BlockPlusRowsSpectrum::new(&roots, &dense_rows, &layout).expect("inertia machinery");
        assert_eq!(
            spectrum.count_above(1.0e-20).expect("count"),
            2,
            "both the unit eigenvalue and the 1e-14 one are above 1e-20"
        );
        // And the shift still lands correctly on either side of each.
        assert_eq!(spectrum.count_above(1.0e-16).expect("count"), 2);
        assert_eq!(spectrum.count_above(1.0e-12).expect("count"), 1);
        assert_eq!(spectrum.count_above(2.0).expect("count"), 0);
    }

    #[test]
    fn a_degenerate_layout_yields_an_empty_curvature_rather_than_an_error() {
        for axes in [vec![], vec![0usize], vec![1usize, 2]] {
            for p in [0usize, 3] {
                let layout = FrameColumnLayout::new(p, &axes);
                let curvature = OutputBlockRootAccumulator::new(layout.clone()).finish(0);
                assert_eq!(curvature.param_dim(), layout.param_dim());
                assert!(curvature.is_finite());
                let gram = curvature.to_dense_gram();
                assert_eq!(gram.dim(), (layout.param_dim(), layout.param_dim()));
                assert!(gram.iter().all(|v| *v == 0.0));
            }
        }
    }

    #[test]
    fn is_finite_refuses_a_nan_in_any_representation() {
        let layout = FrameColumnLayout::new(2, &[1]);
        let mut roots = Array3::<f64>::zeros((2, 1, 1));
        roots[[0, 0, 0]] = 1.0;
        let clean = ResidualGaugeCurvature::OutputBlockRoots {
            roots: roots.clone(),
            dense_rows: Array2::<f64>::zeros((0, layout.param_dim())),
            layout: layout.clone(),
            root_rows: 3,
        };
        assert!(clean.is_finite());
        roots[[1, 0, 0]] = f64::NAN;
        let dirty = ResidualGaugeCurvature::OutputBlockRoots {
            roots,
            dense_rows: Array2::<f64>::zeros((0, layout.param_dim())),
            layout,
            root_rows: 3,
        };
        assert!(!dirty.is_finite());
        assert!(
            !ResidualGaugeCurvature::DualRoot {
                root: Array2::<f64>::from_elem((1, 2), f64::INFINITY),
                root_rows: 1,
            }
            .is_finite()
        );
        assert!(
            !ResidualGaugeCurvature::DenseGram {
                gram: Array2::<f64>::from_elem((2, 2), f64::NAN),
                root_rows: 1,
            }
            .is_finite()
        );
    }

    #[test]
    fn output_blocks_densify_to_a_block_diagonal_gram() {
        let layout = FrameColumnLayout::new(3, &[1, 2]);
        let mut roots = Array3::<f64>::zeros((3, 3, 3));
        for i in 0..3 {
            for a in 0..3 {
                for b in a..3 {
                    roots[[i, a, b]] = (i + 1) as f64 * ((a + 1) + (b + 1)) as f64;
                }
            }
        }
        let curvature = ResidualGaugeCurvature::OutputBlockRoots {
            roots,
            dense_rows: Array2::<f64>::zeros((0, layout.param_dim())),
            layout: layout.clone(),
            root_rows: 7,
        };
        let dense = curvature.to_dense_gram();
        assert_eq!(dense.dim(), (9, 9));
        for a in 0..9 {
            for b in 0..9 {
                let (ia, ib) = (
                    layout.output_of(a).expect("in range"),
                    layout.output_of(b).expect("in range"),
                );
                if ia != ib {
                    assert_eq!(dense[[a, b]], 0.0, "off-block ({a},{b}) must be zero");
                } else {
                    assert!(dense[[a, b]] != 0.0, "in-block ({a},{b}) must be populated");
                }
            }
        }
    }
}
