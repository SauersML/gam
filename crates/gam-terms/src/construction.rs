use crate::EstimationError;
use crate::basis::analyze_penalty_block;
use crate::smooth::PenaltyStructureHint;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side};
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd};
use gam_linalg::matrix::symmetrize_in_place;
use gam_math::sparse_grid::CompensatedSum;
use ndarray::{ArcArray2, Array1, Array2, ArrayView2, ArrayViewMut2, s};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

#[derive(Clone)]
pub enum PenaltyRepresentation {
    Dense(Array2<f64>),
    Banded {
        bands: Vec<Array1<f64>>,
        offsets: Vec<i32>,
    },
    Kronecker {
        /// Full penalty-block Kronecker product `left ⊗ right`: each entry of
        /// `left` scales an entire copy of `right` in the dense expansion.
        ///
        /// This is distinct from chunked kernel design assembly, where center
        /// rows are kernel-evaluation arguments rather than matrix factors.
        left: Array2<f64>,
        right: Array2<f64>,
    },
}

impl PenaltyRepresentation {}

#[derive(Clone)]
pub struct PenaltyMatrix {
    pub col_range: Range<usize>,
    pub representation: PenaltyRepresentation,
}

impl PenaltyMatrix {
    fn accumulate_into(&self, mut dest: ArrayViewMut2<'_, f64>, weight: f64) {
        if weight == 0.0 {
            return;
        }
        match &self.representation {
            PenaltyRepresentation::Dense(block) => {
                dest.scaled_add(weight, block);
            }
            PenaltyRepresentation::Banded { bands, offsets } => {
                let positive_offsets: HashSet<usize> = offsets
                    .iter()
                    .filter_map(|&off| (off >= 0).then_some(off as usize))
                    .collect();
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    let off = offset.unsigned_abs() as usize;
                    if offset < 0 && positive_offsets.contains(&off) {
                        continue;
                    }
                    for (idx, &value) in band.iter().enumerate() {
                        let (i, j) = if offset >= 0 {
                            (idx, idx + off)
                        } else {
                            (idx + off, idx)
                        };
                        let Some(entry_ij) = dest.get_mut((i, j)) else {
                            continue;
                        };
                        *entry_ij += weight * value;
                        if i != j
                            && let Some(entry_ji) = dest.get_mut((j, i))
                        {
                            *entry_ji += weight * value;
                        }
                    }
                }
            }
            PenaltyRepresentation::Kronecker { left, right } => {
                let (lrows, l_cols) = left.dim();
                let (rrows, r_cols) = right.dim();
                for i in 0..lrows {
                    for j in 0..l_cols {
                        let scale = left[(i, j)] * weight;
                        if scale == 0.0 {
                            continue;
                        }
                        let mut block = dest.slice_mut(s![
                            i * rrows..(i + 1) * rrows,
                            j * r_cols..(j + 1) * r_cols
                        ]);
                        block.scaled_add(scale, right);
                    }
                }
            }
        }
    }

    pub fn to_dense(&self, total_dim: usize) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((total_dim, total_dim));
        self.accumulate_into(
            dense.slice_mut(s![self.col_range.clone(), self.col_range.clone()]),
            1.0,
        );
        dense
    }
}

pub(crate) fn array_to_faer<S: ndarray::Data<Elem = f64>>(
    array: &ndarray::ArrayBase<S, ndarray::Ix2>,
) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[[i, j]])
}

pub(crate) fn mat_to_array(mat: &Mat<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn mat_max_abs_element(matrix: MatRef<'_, f64>) -> f64 {
    let (rows, cols) = matrix.shape();
    let mut maxval = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            if val.is_finite() {
                maxval = maxval.max(val.abs());
            }
        }
    }
    maxval
}

fn sanitize_symmetric_faer(matrix: &Mat<f64>) -> Mat<f64> {
    let (rows, cols) = matrix.as_ref().shape();
    assert_eq!(rows, cols, "Matrix must be square for sanitization");

    let mut sanitized = matrix.clone();

    for i in 0..rows {
        let diag = sanitized[(i, i)];
        if !diag.is_finite() {
            sanitized[(i, i)] = 0.0;
        }
        for j in (i + 1)..cols {
            let mut upper = sanitized[(i, j)];
            let mut lower = sanitized[(j, i)];
            if !upper.is_finite() {
                upper = 0.0;
            }
            if !lower.is_finite() {
                lower = 0.0;
            }
            let avg = 0.5 * (upper + lower);
            sanitized[(i, j)] = avg;
            sanitized[(j, i)] = avg;
        }
    }

    // Finite entries are kept as computed, however small. Before an
    // eigendecomposition, `classify_eigenvalues_strict` already snaps roundoff at
    // the resolved-eigenvalue band `p·ε·λ_max + assembly`, and every entry-level
    // roundoff is bounded by it because `|M_ij| ≤ ‖M‖₂`.
    for i in 0..rows {
        for j in 0..cols {
            if !sanitized[(i, j)].is_finite() {
                sanitized[(i, j)] = 0.0;
            }
        }
    }

    sanitized
}

/// `tr(U diag(1/(d+δ)) Uᵀ S)` for `S = RᵀR` restricted to its leading
/// `block_dim` coordinates, contracted through the root: each diagonal term is
/// `u_lᵀ S u_l = ‖R[:, ..block_dim] u_l‖²`, so `S` itself is never formed.
fn trace_root_penalty_in_orthogonal_basis(
    root: &Mat<f64>,
    block_dim: usize,
    orthogonal: &Mat<f64>,
    rotated_eigenvalues: &[f64],
    delta: f64,
) -> f64 {
    let root_block = root.as_ref().submatrix(0, 0, root.nrows(), block_dim);
    let cols = orthogonal.ncols();
    assert!(rotated_eigenvalues.len() >= cols);
    let mut projected = Mat::<f64>::zeros(root.nrows(), cols);
    matmul(
        projected.as_mut(),
        Accum::Replace,
        root_block,
        orthogonal.as_ref(),
        1.0,
        Par::Seq,
    );
    let mut trace = CompensatedSum::default();
    for l in 0..cols {
        let mut diag_ll = CompensatedSum::default();
        for i in 0..root.nrows() {
            let v = projected[(i, l)];
            diag_ll.add(v * v);
        }
        trace.add(diag_ll.value() / (rotated_eigenvalues[l] + delta));
    }
    trace.value()
}

pub fn trace_reduced_penalty_covariance(
    reduced_penalty: &Array2<f64>,
    covariance_basis: &Array2<f64>,
) -> f64 {
    assert_eq!(
        reduced_penalty.dim(),
        covariance_basis.dim(),
        "trace_reduced_penalty_covariance dimension mismatch"
    );
    let r = covariance_basis.nrows();
    let mut trace = CompensatedSum::default();
    for i in 0..r {
        for j in 0..r {
            trace.add(covariance_basis[[i, j]] * reduced_penalty[[j, i]]);
        }
    }
    trace.value()
}

pub fn trace_penalty_covariance_in_orthogonal_basis(
    matrix: &Array2<f64>,
    orthogonal: &Array2<f64>,
    covariance_basis: &Array2<f64>,
) -> f64 {
    let reduced = gam_linalg::faer_ndarray::fast_ab(
        &gam_linalg::faer_ndarray::fast_atb(orthogonal, matrix),
        orthogonal,
    );
    trace_reduced_penalty_covariance(&reduced, covariance_basis)
}

/// Strict spectral classifier used as a final guard on penalty eigendecompositions.
///
/// Penalty matrices fed to the GAM solver are required to be PSD by construction.
/// This routine snaps eigenvalues the decomposition has not resolved from zero to
/// exact zero, accepts resolved positive eigenvalues, and rejects resolved
/// negative (materially indefinite) or non-finite spectra with a hard error
/// rather than silently rewriting them.
///
/// The band is [`gam_linalg::roundoff::resolved_eigenvalue_band`]: the
/// eigensolver's backward error `p·ε·max|λ|` plus `assembly_band`, the caller's
/// bound on the error the matrix's own formation left in it. By Weyl every
/// computed eigenvalue is within that band of the exact one, so an eigenvalue
/// inside it is not resolved from zero and its sign is not a measurement, and an
/// eigenvalue below `−band` is a genuinely negative eigenvalue of the operator
/// the caller meant to form. This is the SAME predicate the structural rank
/// counts with (`resolved_eigenvalue_count(eigs, assembly_band)`), so the snap
/// can never pre-empt the rank decision: every eigenvalue it zeroes is one the
/// rank rule already scores as null (#4057). Extreme-λ assembly roundoff (#1619)
/// is covered by the caller's `assembly_band`, which grows with the λ-weighted
/// row norms that produced it, not by a fixed relative floor.
fn classify_eigenvalues_strict(
    eigenvalues: &mut [f64],
    assembly_band: f64,
    context: &str,
) -> Result<(), EstimationError> {
    let mut scale = 0.0_f64;
    for (idx, &val) in eigenvalues.iter().enumerate() {
        if !val.is_finite() {
            return Err(EstimationError::PenaltySpectrumNonFinite {
                context: context.to_string(),
                index: idx,
                value: val,
            });
        }
        scale = scale.max(val.abs());
    }

    let tolerance = gam_linalg::roundoff::resolved_eigenvalue_band(eigenvalues, assembly_band);

    for (idx, val) in eigenvalues.iter_mut().enumerate() {
        if val.abs() <= tolerance {
            *val = 0.0;
        } else if *val < 0.0 {
            return Err(EstimationError::PenaltySpectrumIndefinite {
                context: context.to_string(),
                index: idx,
                value: *val,
                tolerance,
                scale,
            });
        }
    }
    Ok(())
}

fn robust_eighwith_policy<M, V, E, Validate, Sanitize, EigCall, MapErr>(
    matrix: &M,
    assembly_band: f64,
    context: &str,
    validate_input: Validate,
    sanitize: Sanitize,
    mut eig_call: EigCall,
    map_error: MapErr,
) -> Result<(Vec<f64>, V), EstimationError>
where
    Validate: Fn(&M, &str) -> Result<(), EstimationError>,
    Sanitize: Fn(&M) -> M,
    EigCall: FnMut(&M) -> Result<(Vec<f64>, V), E>,
    MapErr: Fn(E, &str) -> EstimationError,
{
    validate_input(matrix, context)?;

    // The sanitize step only enforces exact symmetry by averaging M and M^T; it
    // neither zeroes small entries nor adds a diagonal ridge. Either would change
    // the matrix being decomposed, which silently changes the optimisation
    // objective downstream. If eigh genuinely fails on a finite symmetric input,
    // surface the error instead of mutating the spectrum.
    let candidate = sanitize(matrix);
    match eig_call(&candidate) {
        Ok((mut eigenvalues, eigenvectors)) => {
            classify_eigenvalues_strict(&mut eigenvalues, assembly_band, context)?;
            Ok((eigenvalues, eigenvectors))
        }
        Err(err) => Err(map_error(err, context)),
    }
}

/// Symmetric eigendecomposition of a penalty operator with the strict PSD
/// classification of [`classify_eigenvalues_strict`] at `assembly_band`, the
/// caller's bound on the formation error of `matrix` (zero for exact input).
pub(crate) fn robust_eigh_faer(
    matrix: &Mat<f64>,
    side: Side,
    assembly_band: f64,
    context: &str,
) -> Result<(Vec<f64>, Mat<f64>), EstimationError> {
    robust_eighwith_policy(
        matrix,
        assembly_band,
        context,
        |mat, ctx| {
            let (rows, cols) = mat.as_ref().shape();
            for i in 0..rows {
                for j in 0..cols {
                    let val = mat[(i, j)];
                    if !val.is_finite() {
                        let max_abs = mat_max_abs_element(mat.as_ref());
                        crate::bail_invalid_estim!(
                            "{} contains non-finite entries (max finite magnitude {:.3e})",
                            ctx,
                            max_abs
                        );
                    }
                }
            }
            Ok(())
        },
        sanitize_symmetric_faer,
        |candidate| {
            let (values, eigenvectors) =
                gam_linalg::faer_ndarray::self_adjoint_evd(candidate.as_ref(), side)?;
            let eigenvalues = (0..values.dim()).map(|idx| values[idx]).collect();
            Ok((eigenvalues, eigenvectors))
        },
        |err, _| EstimationError::EigendecompositionFailed(FaerLinalgError::SelfAdjointEigen(err)),
    )
}

#[derive(Debug, Clone, Copy)]
struct SubspaceLeakageMetrics {
    max_abs_sq: f64,
    max_rel_sq: f64,
    worst_penalty: usize,
    max_cross_gram_abs: f64,
}

fn assess_subspace_leakage(
    qs: &Mat<f64>,
    rs_transformed: &[Mat<f64>],
    structural_rank: usize,
    p: usize,
) -> SubspaceLeakageMetrics {
    let mut max_abs_sq = 0.0_f64;
    let mut max_rel_sq = 0.0_f64;
    let mut worst_penalty = 0usize;

    for (k, rs) in rs_transformed.iter().enumerate() {
        let rows = rs.nrows();
        let cols = rs.ncols().min(p);
        let null_start = structural_rank.min(cols);
        let mut abs_sq = 0.0_f64;
        let mut total_sq = 0.0_f64;
        for i in 0..rows {
            for j in 0..cols {
                let v = rs[(i, j)];
                let vv = v * v;
                total_sq += vv;
                if j >= null_start {
                    abs_sq += vv;
                }
            }
        }
        let rel_sq = if total_sq > 0.0 {
            abs_sq / total_sq
        } else {
            0.0
        };
        if rel_sq > max_rel_sq {
            max_rel_sq = rel_sq;
            worst_penalty = k;
        }
        max_abs_sq = max_abs_sq.max(abs_sq);
    }

    let mut max_cross_gram_abs = 0.0_f64;
    let null_count = p.saturating_sub(structural_rank);
    if structural_rank > 0 && null_count > 0 {
        for i in 0..structural_rank {
            for j in 0..null_count {
                let qn_col = structural_rank + j;
                let mut dot = 0.0_f64;
                for r in 0..p {
                    dot += qs[(r, i)] * qs[(r, qn_col)];
                }
                max_cross_gram_abs = max_cross_gram_abs.max(dot.abs());
            }
        }
    }

    SubspaceLeakageMetrics {
        max_abs_sq,
        max_rel_sq,
        worst_penalty,
        max_cross_gram_abs,
    }
}

/// True when the penalized/null subspace split is numerically self-consistent.
///
/// The split has two independent invariants:
///
/// 1. **Orthogonality** — `Qs = [Q_p | Q_n]` must be orthonormal, so the range
///    and null blocks share no direction (`max |Qp'Qn| ≤ orth_tol`). This is the
///    structural correctness guarantee and is checked at machine precision.
///
/// 2. **Bounded root leakage** — the transformed penalty roots must keep no
///    more relative energy on the null columns than `null_leakage_tolerance`,
///    the invariant's own bound ([`balanced_null_leakage_tolerance`]) on what
///    roundoff of the split can leave there. The null columns are exactly the
///    balanced operator's eigenvalues NOT resolved from zero, so their energy is
///    bounded by that operator's resolution; it no longer tracks a fixed
///    relative rank floor. Under the old `1e-8`-relative cut a manifold / Duchon
///    / sphere penalty whose spectrum decays through the cut with no clean gap
///    (#1802) had a resolved eigenvalue just below it declared null, and its
///    `~1e-8` energy there had to be admitted by a matching `p·1e-8` leakage
///    floor. The cut is now the resolved-eigenvalue predicate, so such a
///    direction is penalized and the leakage the split can carry is roundoff
///    of the operator — anything above the bound means the roots and the
///    operator the split was read from disagree.
fn subspace_split_is_consistent(
    leakage: &SubspaceLeakageMetrics,
    null_leakage_tolerance: f64,
) -> bool {
    let orth_tol = 1e-10;
    let root_leaks = leakage.max_rel_sq > null_leakage_tolerance;
    let split_nonorthogonal = leakage.max_cross_gram_abs > orth_tol;
    !(root_leaks || split_nonorthogonal)
}

fn compose_qs_from_split(q_pen: &Mat<f64>, q_null: &Mat<f64>, p: usize) -> Mat<f64> {
    let rank = q_pen.ncols();
    let null_count = q_null.ncols();
    let mut qs = Mat::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..rank {
            qs[(i, j)] = q_pen[(i, j)];
        }
        for j in 0..null_count {
            qs[(i, rank + j)] = q_null[(i, j)];
        }
    }
    qs
}

/// Result of the stable reparameterization algorithm from Wood (2011) Appendix B
#[derive(Clone)]
pub struct ReparamResult {
    /// Penalty matrix in TRANSFORMED coefficient coordinates.
    ///
    /// This must be compatible with `beta_transformed` and `X_transformed = X * Qs`.
    pub s_transformed: Array2<f64>,
    /// Log-determinant of the penalty matrix (stable computation)
    pub log_det: f64,
    /// First derivatives of log-determinant w.r.t. log-smoothing parameters
    pub det1: Array1<f64>,
    /// Orthogonal transformation matrix Qs
    pub qs: Array2<f64>,
    /// The canonical penalty roots rotated into the TRANSFORMED frame, before the
    /// split's projection. They describe each canonical `S_k`, not the `S̃` the
    /// engine put in `H`: a rotated root keeps a relative leakage onto the null
    /// coordinates, so a consumer whose Hessian carries `s_transformed` reads
    /// [`ReparamResult::applied_penalties`] instead.
    pub canonical_transformed: Vec<CanonicalPenalty>,
    /// Lambda-dependent penalty square root in TRANSFORMED coordinates (rank x p matrix).
    /// This is used for applying the actual penalty in the least squares solve.
    pub e_transformed: Array2<f64>,
    /// Truncated eigenvectors (p × m where m = p - structural_rank).
    ///
    /// Coordinate frame note:
    /// - This matrix is stored in the TRANSFORMED coefficient frame (post-`Qs`),
    ///   i.e. it is compatible with `canonical_transformed`, `beta_transformed`,
    ///   and transformed Hessians without additional coordinate mapping.
    ///
    /// These vectors span the structural null space used by positive-part
    /// log-determinant conventions.
    pub u_truncated: Array2<f64>,
}

/// The coefficient frame a set of penalty roots is expressed in.
///
/// The reparameterization's declared-null basis is stored in the TRANSFORMED
/// (post-`Qs`) frame, so projecting an ORIGINAL-frame penalty against it
/// requires rotating the basis by `Qs` first. Naming the frame at the call site
/// is what keeps that rotation from being a coin flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PenaltyFrame {
    Original,
    Transformed,
}

/// The λ-invariant declared-null subspace of a penalty reparameterization,
/// materialized in both coefficient frames (#2454).
///
/// # The invariant this type exists to hold
///
/// `gam.reparam` keeps only balanced-penalty eigendirections above a relative
/// rank tolerance and rebuilds the penalty the model applies as
/// `S̃(λ) = E(λ)ᵀE(λ)` on that subspace alone. **Everything the criterion is
/// made of is a function of `S̃`**: the inner solve minimizes `−ℓ + ½βᵀS̃β`,
/// `H = −∇²ℓ + S̃`, and the reported penalty energy is `‖E β̂‖²`.
///
/// A per-block `S_k` whose own root rank exceeds the split's penalized rank
/// therefore describes a DIFFERENT penalty. Two things go wrong if one is used
/// anywhere the criterion is differentiated or normalized:
///
///  1. `½ λ_k β̂ᵀS_kβ̂` charges β̂'s energy in directions `S̃` never penalized —
///     and `β̂` is free there, so that energy is `O(1)` and `∂/∂ρ_k` multiplies
///     it by `λ_k`. That is an additive `c·λ` in the outer gradient, invisible
///     at `‖ρ‖ ≤ 1` and sign-flipping it a dozen e-folds up.
///  2. `−½log|Σ_k λ_k S_k|₊` charges MORE directions than `½log|H|` can ever
///     inflate, because `H`'s penalty part is `S̃`. The two halves of the LAML
///     ratio then saturate at different rates and the criterion acquires an
///     asymptotic slope of `½(rank(S̃) − rank(S))` per unit ρ — unbounded
///     below, with no interior optimum and no certifiable λ=∞ rail.
///
/// Projecting restores the identity every outer derivative is built on:
/// `Σ_k λ_k (Π S_k Π) = Π (Σ_k λ_k S_k) Π = S̃(λ)`, and because `Π` is
/// λ-invariant by construction, `∂S̃/∂ρ_k = λ_k · Π S_k Π` exactly. One penalty
/// object then serves the value, the quadratic, the per-block scores, the
/// `tr(H⁻¹Ḣ_k)` drift, the `log|S|₊` and its rank, and the outer Hessian.
///
/// A `None` basis means "nothing declared null" and every `project` is the
/// identity, which is the case for any penalty whose numerical rank agrees
/// with the split's.
#[derive(Clone, Debug)]
pub struct PenaltyNullSplit {
    transformed: Option<Array2<f64>>,
    original: Option<Array2<f64>>,
}

impl PenaltyNullSplit {
    /// The identity split: nothing is declared null, every projection is a
    /// no-op. This is what a penalty system with no dense reparameterization
    /// (Kronecker marginal grids, sparse-native identity frames) carries.
    pub fn identity() -> Self {
        Self {
            transformed: None,
            original: None,
        }
    }

    /// The number of declared-null directions, or 0 for the identity split.
    pub fn declared_null_dim(&self) -> usize {
        self.transformed.as_ref().map_or(0, |n| n.ncols())
    }

    /// The declared-null basis in `frame`, if this split declares anything.
    pub fn basis(&self, frame: PenaltyFrame) -> Option<&Array2<f64>> {
        match frame {
            PenaltyFrame::Original => self.original.as_ref(),
            PenaltyFrame::Transformed => self.transformed.as_ref(),
        }
    }

    /// `Π S_k Π` for a canonical penalty expressed in `frame`.
    ///
    /// The penalty is returned unchanged when this split declares nothing null,
    /// or when the basis does not match the penalty's dimension — a shape
    /// disagreement the projection must not paper over by guessing.
    pub fn project_canonical(
        &self,
        penalty: &CanonicalPenalty,
        frame: PenaltyFrame,
    ) -> Result<CanonicalPenalty, EstimationError> {
        Ok(self
            .projected_canonical(penalty, frame)?
            .unwrap_or_else(|| penalty.clone()))
    }

    /// [`Self::project_canonical`], returning `None` where the projection
    /// leaves `penalty` as it is.
    pub fn projected_canonical(
        &self,
        penalty: &CanonicalPenalty,
        frame: PenaltyFrame,
    ) -> Result<Option<CanonicalPenalty>, EstimationError> {
        match self.basis(frame) {
            Some(n) if n.nrows() == penalty.total_dim => {
                penalty.projected_out_of_null_directions(n.view())
            }
            _ => Ok(None),
        }
    }

    /// `Π S_k Π` for a solver-side penalty coordinate expressed in `frame`.
    pub fn project_coordinate(
        &self,
        coord: &gam_problem::PenaltyCoordinate,
        frame: PenaltyFrame,
    ) -> gam_problem::PenaltyCoordinate {
        match self.basis(frame) {
            Some(n) if n.nrows() == coord.dim() => coord.project_out_null_directions(n.view()),
            _ => coord.clone(),
        }
    }
}

impl ReparamResult {
    /// The λ-invariant subspace split this reparameterization declared.
    ///
    /// `u_truncated` is `p × m` in the transformed frame; the original-frame
    /// basis is `Qs · u_truncated` (`Qs` orthogonal, and `u_truncated` is
    /// itself defined as `Qsᵀ Q_null`). A degenerate or absent basis yields the
    /// identity split rather than a guessed rotation.
    pub fn null_split(&self) -> PenaltyNullSplit {
        let u = &self.u_truncated;
        if u.ncols() == 0 || u.nrows() == 0 {
            return PenaltyNullSplit::identity();
        }
        let qs = &self.qs;
        let original = if qs.nrows() == u.nrows() && qs.ncols() == u.nrows() {
            qs.dot(u)
        } else {
            // No usable `Qs`: the frames coincide (sparse-native / identity
            // reparameterization) or the shapes are inconsistent, in which case
            // `project` declines on the dimension check anyway.
            u.clone()
        };
        PenaltyNullSplit {
            transformed: Some(u.clone()),
            original: Some(original),
        }
    }

    /// The penalties the criterion applies, one per smoothing coordinate, in the
    /// transformed frame: `S̃_k = Π S_k Π`, the rotated roots projected onto the
    /// λ-invariant penalized block this reparameterization declared (#2454).
    /// `Σ_k λ_k S̃_k` is the `S̃ = EᵀE` in `s_transformed`, and because `Π` does not
    /// move with ρ, `∂S̃/∂ρ_k = λ_k S̃_k` exactly.
    ///
    /// Every consumer whose Hessian carries `s_transformed` reads its per-block
    /// penalties here: traces `λ_k tr(H⁻¹S̃_k)`, the Gram `H − S̃`, the mode
    /// response `H⁻¹λ_k S̃_k β̂`, `log|S̃|₊` and its derivatives. On
    /// `y ~ s(x) + s(x, g, bs='fs')` at λ = 1.98e12, the raw roots' leakage onto
    /// the null coordinates reached 8.9e4 against a data curvature of 121, and
    /// the raw fs block's trace was 6.09e4 against 20.01 here (#2901).
    pub fn applied_penalties(&self) -> Result<Vec<CanonicalPenalty>, EstimationError> {
        let split = self.null_split();
        self.canonical_transformed
            .iter()
            .map(|penalty| split.project_canonical(penalty, PenaltyFrame::Transformed))
            .collect()
    }

    /// Bytes this reparameterization owns on the heap. Every rotated penalty in
    /// `canonical_transformed` is dense in the transformed frame, so with `K`
    /// smoothing coordinates this is about `(K + 3) p²` doubles: forty
    /// coordinates at `p = 221` hold 16 MB here, far more than `S̃`, `Qs` and `E`.
    pub fn resident_bytes(&self) -> usize {
        let owned = self.s_transformed.len()
            + self.det1.len()
            + self.qs.len()
            + self.e_transformed.len()
            + self.u_truncated.len();
        owned * std::mem::size_of::<f64>()
            + self
                .canonical_transformed
                .iter()
                .map(CanonicalPenalty::resident_bytes)
                .sum::<usize>()
    }
}

// ---------------------------------------------------------------------------
// Kronecker factor decomposition primitives
// ---------------------------------------------------------------------------

/// Per-factor decomposition result for Kronecker penalties.
struct KroneckerFactorDecomp {
    root: Array2<f64>,              // rank_j × q_j
    positive_eigenvalues: Vec<f64>, // length = rank_j
    rank: usize,
    dim: usize,
}

/// Eigendecompose each Kronecker factor separately at O(Σ q_j³).
/// Returns per-factor decompositions, or `None` if any factor is zero.
fn decompose_kronecker_factors(
    factors: &[Array2<f64>],
    context: &str,
) -> Result<Option<Vec<KroneckerFactorDecomp>>, EstimationError> {
    let mut decomps = Vec::with_capacity(factors.len());
    for (j, factor) in factors.iter().enumerate() {
        let q_j = factor.nrows();
        if q_j != factor.ncols() {
            crate::bail_invalid_estim!(
                "{context}: Kronecker factor {j} must be square, got {}x{}",
                factor.nrows(),
                factor.ncols()
            );
        }
        let is_identity = {
            let mut is_id = true;
            'outer: for r in 0..q_j {
                for c in 0..q_j {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    if factor[[r, c]] != expected {
                        is_id = false;
                        break 'outer;
                    }
                }
            }
            is_id
        };
        if is_identity {
            decomps.push(KroneckerFactorDecomp {
                root: Array2::eye(q_j),
                positive_eigenvalues: vec![1.0; q_j],
                rank: q_j,
                dim: q_j,
            });
            continue;
        }
        let analysis = analyze_penalty_block(factor).map_err(|err| {
            EstimationError::InvalidInput(format!(
                "{context}: Kronecker factor {j} eigendecomp failed: {err}"
            ))
        })?;
        if analysis.rank == 0 {
            return Ok(None);
        }
        // Build the factor root from ONLY the range (positive-curvature)
        // directions via the canonical classifier — never the null or
        // negative-curvature directions (#1425).
        let factor_classes = crate::basis::SpectralClassification::new(
            &analysis.eigenvalues,
            analysis.rank_tol,
            analysis.noise_tol,
        );
        let mut root_j = Array2::zeros((analysis.rank, q_j));
        let mut pos_eigs = Vec::with_capacity(analysis.rank);
        for (row_idx, &i) in factor_classes.range_idx.iter().enumerate() {
            let eigenval = analysis.eigenvalues[i];
            let sqrt_ev = eigenval.sqrt();
            let evec = analysis.eigenvectors.column(i);
            for (col, &v) in evec.iter().enumerate() {
                root_j[[row_idx, col]] = sqrt_ev * v;
            }
            pos_eigs.push(eigenval);
        }
        decomps.push(KroneckerFactorDecomp {
            root: root_j,
            positive_eigenvalues: pos_eigs,
            rank: analysis.rank,
            dim: q_j,
        });
    }
    Ok(Some(decomps))
}

/// Build the block-local Kronecker root from pre-computed factor decompositions.
fn assemble_kronecker_root_local(decomps: &[KroneckerFactorDecomp]) -> Array2<f64> {
    let mut kron_root = decomps[0].root.clone();
    for fr in &decomps[1..] {
        let (r1, c1) = kron_root.dim();
        let (r2, c2) = (fr.rank, fr.dim);
        let mut new_root = Array2::zeros((r1 * r2, c1 * c2));
        for i1 in 0..r1 {
            for i2 in 0..r2 {
                for j1 in 0..c1 {
                    for j2 in 0..c2 {
                        new_root[[i1 * r2 + i2, j1 * c2 + j2]] =
                            kron_root[[i1, j1]] * fr.root[[i2, j2]];
                    }
                }
            }
        }
        kron_root = new_root;
    }
    kron_root
}

/// `RᵀR` of the Kronecker root [`assemble_kronecker_root_local`] builds, as
/// `⊗_j R_jᵀR_j` (`(A⊗B)ᵀ(A⊗B) = AᵀA ⊗ BᵀB`), at `O(Σ_j q_j³ + block_dim²)`
/// instead of the `O(block_dim³)` of forming it from the assembled root.
fn assemble_kronecker_gram_local(decomps: &[KroneckerFactorDecomp]) -> Array2<f64> {
    let factor_gram = |decomp: &KroneckerFactorDecomp| decomp.root.t().dot(&decomp.root);
    let mut gram = factor_gram(&decomps[0]);
    for decomp in &decomps[1..] {
        let right = factor_gram(decomp);
        let (n1, n2) = (gram.nrows(), right.nrows());
        let mut product = Array2::zeros((n1 * n2, n1 * n2));
        for i1 in 0..n1 {
            for j1 in 0..n1 {
                let left = gram[[i1, j1]];
                if left == 0.0 {
                    continue;
                }
                for i2 in 0..n2 {
                    for j2 in 0..n2 {
                        product[[i1 * n2 + i2, j1 * n2 + j2]] = left * right[[i2, j2]];
                    }
                }
            }
        }
        gram = product;
    }
    gram
}

/// Compute eigenvalues of the Kronecker product from per-factor eigenvalues.
fn kronecker_eigenvalues(decomps: &[KroneckerFactorDecomp], block_dim: usize) -> (Vec<f64>, usize) {
    let mut kron_eigs = decomps[0].positive_eigenvalues.clone();
    for fd in &decomps[1..] {
        let mut new_eigs = Vec::with_capacity(kron_eigs.len() * fd.positive_eigenvalues.len());
        for &a in &kron_eigs {
            for &b in &fd.positive_eigenvalues {
                new_eigs.push(a * b);
            }
        }
        kron_eigs = new_eigs;
    }
    // Every factor eigenvalue was classified positive against its own factor's
    // spectrum, and a product of positives is positive, resolved to the sum of its
    // factors' relative resolutions. Each product is one row of the assembled
    // Kronecker root, so every one is kept: a cut against the joint maximum would
    // drop eigenvalues whose root rows the penalty still carries.
    let nullity = block_dim - kron_eigs.len();
    (kron_eigs, nullity)
}

// ---------------------------------------------------------------------------
// CanonicalPenalty — block-local processed penalty for the solver
// ---------------------------------------------------------------------------

/// A canonicalized penalty with block-local root, ready for the solver.
///
/// Instead of storing a full `p x p` penalty matrix, this stores only the
/// `rank x block_dim` root and the column range, enabling O(p_k^2) operations
/// instead of O(p^2).
#[derive(Clone)]
pub struct CanonicalPenalty {
    /// Square root matrix: S_k = root^T * root.
    /// Shape: `rank x block_dim` for block-local, `rank x p` for dense.
    ///
    /// Shared (copy-on-write) storage: every outer evaluation's
    /// reparameterization carries the canonical penalties forward, and a
    /// random-effect block's root is `levels × levels`, so an owned root would
    /// be deep-copied on each of those clones.
    pub root: ArcArray2<f64>,
    /// Column range in the global coefficient vector [start..end).
    /// For dense penalties this is `0..p`.
    pub col_range: std::ops::Range<usize>,
    /// Full parameter dimension p.
    pub total_dim: usize,
    /// Structural nullity of the local penalty.
    pub nullity: usize,
    /// The block-local penalty matrix `root^T * root` (block_dim × block_dim).
    /// Cached at construction time to avoid recomputing it in hot paths
    /// (penalty assembly, trace products). It must be that reconstruction, not
    /// the raw input: the reparameterization reads its penalized subspace from
    /// the `local`s and measures its consistency on the `root`s, at a
    /// tolerance derived from the rounding of forming one from the other.
    /// Shared storage for the same reason as `root`.
    pub local: ArcArray2<f64>,
    /// Positive eigenvalues of the local penalty matrix (length = rank).
    /// Cached at construction time for REML logdet block-factored paths.
    pub positive_eigenvalues: Vec<f64>,
    /// Optional operator-form handle bit-equivalent to `local`. Propagated
    /// from `PenaltySpec::Block.op`. Downstream PIRLS and REML exact operator
    /// algebra route through this for dense-Gram-free matvec when present.
    pub op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
}

impl std::fmt::Debug for CanonicalPenalty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalPenalty")
            .field(
                "root",
                &format_args!("{}×{}", self.root.nrows(), self.root.ncols()),
            )
            .field("col_range", &self.col_range)
            .field("total_dim", &self.total_dim)
            .field("nullity", &self.nullity)
            .field(
                "local",
                &format_args!("{}×{}", self.local.nrows(), self.local.ncols()),
            )
            .field("positive_eigenvalues", &self.positive_eigenvalues)
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}

impl CanonicalPenalty {
    /// Construct a dense (full-width) canonical penalty from a `rank x p` root.
    /// Used to wrap reparam-transformed roots for consumers that expect
    /// `&[CanonicalPenalty]`.
    pub fn from_dense_root(root: Array2<f64>, p: usize) -> Self {
        let local = root.t().dot(&root);
        let positive_eigenvalues = Vec::new(); // not needed for TK paths
        Self {
            root: root.into_shared(),
            col_range: 0..p,
            total_dim: p,
            nullity: 0,
            local: local.into_shared(),
            positive_eigenvalues,
            op: None,
        }
    }

    /// Bytes this penalty owns on the heap: its root, its cached `local` Gram,
    /// and its eigenvalues. A shared `op` handle is not owned here.
    /// A dense (reparam-rotated) penalty's `local` is a full `p × p` matrix.
    pub fn resident_bytes(&self) -> usize {
        (self.root.len() + self.local.len() + self.positive_eigenvalues.len())
            * std::mem::size_of::<f64>()
    }

    /// Embed the block-local root into a full-width `rank × total_dim` matrix.
    /// For dense penalties (col_range = 0..p), returns the root unchanged.
    pub fn full_width_root(&self) -> Array2<f64> {
        if self.col_range.start == 0 && self.col_range.end == self.total_dim {
            return self.root.to_owned();
        }
        let rank = self.root.nrows();
        let mut full = Array2::<f64>::zeros((rank, self.total_dim));
        full.slice_mut(ndarray::s![.., self.col_range.clone()])
            .assign(&self.root);
        full
    }

    /// Numerical rank of this penalty.
    pub fn rank(&self) -> usize {
        self.root.nrows()
    }

    /// Block dimension (number of columns this penalty covers).
    pub fn block_dim(&self) -> usize {
        self.col_range.len()
    }

    /// Whether this penalty is block-local (col_range != 0..total_dim).
    pub const fn is_block_local(&self) -> bool {
        self.col_range.start != 0 || self.col_range.end != self.total_dim
    }

    /// Return a reference to the cached local penalty matrix.
    /// Shape: `block_dim x block_dim`.
    pub fn local_ref(&self) -> &ArcArray2<f64> {
        &self.local
    }

    /// Return an owned copy of the local penalty matrix.
    /// Prefer `local_ref()` when a reference suffices.
    pub fn local_penalty(&self) -> Array2<f64> {
        self.local.to_owned()
    }

    /// Accumulate lambda * S_k into a pre-allocated `p x p` target matrix.
    /// Only touches the block [col_range × col_range].
    pub fn accumulate_weighted(&self, target: &mut Array2<f64>, lambda: f64) {
        if lambda == 0.0 || self.rank() == 0 {
            return;
        }
        let r = &self.col_range;
        target
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(lambda, &self.local);
    }

    /// Compute `scale * v^T S_k v` (quadratic form).
    /// Only reads `v[start..end]` — O(rank × block_dim) not O(rank × p).
    pub fn quadratic(&self, v: &Array1<f64>, scale: f64) -> f64 {
        if self.rank() == 0 || scale == 0.0 {
            return 0.0;
        }
        let v_block = v.slice(s![self.col_range.start..self.col_range.end]);
        let rv = self.root.dot(&v_block);
        scale * rv.dot(&rv)
    }

    /// `Π S_k Π` for the declared-null basis `N` (orthonormal, `total_dim × m`)
    /// in THIS penalty's coefficient frame, with `Π = I − N Nᵀ` (#2454).
    ///
    /// See [`PenaltyNullSplit`] for why every penalty the criterion touches has
    /// to be projected onto the reparameterization's penalized subspace. This
    /// is the `CanonicalPenalty` face of [`gam_problem::PenaltyCoordinate::
    /// project_out_null_directions`]; both go through the same root primitive.
    ///
    /// Returns `self` unchanged — `op` handle, cached spectrum and all — when
    /// the projection is below the root's own representation noise
    /// (`‖R N‖_max ≤ ‖R‖_max · ε · p`). That is not a shortcut but a statement
    /// of exactness: a null direction the root does not resolve carries no
    /// energy the criterion's `log|S|₊` can see either, so the projected and
    /// unprojected penalties agree to every digit either object represents.
    /// It is also the ordinary case — for any penalty whose structural null
    /// space is exact, `N` spans `∩_k null(S_k)` and `Π S_k Π = S_k` in exact
    /// arithmetic.
    ///
    /// When the projection does bite, block locality is preserved if the null
    /// basis is itself block-local, `local` is rebuilt as `RᵀR` from the
    /// projected root, and `positive_eigenvalues` becomes the projected root's
    /// squared singular values — so `Σ positive_eigenvalues = tr(Π S_k Π)`
    /// stays exact for the consumers that read it as a trace. The `op` handle
    /// is dropped: it is declared bit-equivalent to `local`, and the projected
    /// `local` is a different operator.
    pub fn project_out_null_directions(
        &self,
        null_basis: ndarray::ArrayView2<'_, f64>,
    ) -> Result<Self, EstimationError> {
        Ok(self
            .projected_out_of_null_directions(null_basis)?
            .unwrap_or_else(|| self.clone()))
    }

    /// [`Self::project_out_null_directions`], returning `None` where that
    /// returns the penalty itself, so a caller that keeps the original need
    /// not copy its root to learn that nothing moved.
    pub fn projected_out_of_null_directions(
        &self,
        null_basis: ndarray::ArrayView2<'_, f64>,
    ) -> Result<Option<Self>, EstimationError> {
        use gam_problem::ProjectedBlockRoot;
        if null_basis.ncols() == 0 || self.rank() == 0 {
            return Ok(None);
        }
        if null_basis.nrows() != self.total_dim {
            return Err(EstimationError::LayoutError(format!(
                "CanonicalPenalty::project_out_null_directions: null-basis row count {} does \
                 not match the penalty's total dimension {}",
                null_basis.nrows(),
                self.total_dim
            )));
        }
        let projected = gam_problem::project_block_root_out_of_null_directions(
            self.root.view(),
            self.col_range.start,
            self.col_range.end,
            self.total_dim,
            null_basis,
        );
        // How much the projection actually moved, against the root's own
        // representation noise.
        let root_scale = self.root.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if projected.moved() <= root_scale * f64::EPSILON * (self.total_dim as f64) {
            return Ok(None);
        }

        let (root, col_range) = match projected {
            ProjectedBlockRoot::Unchanged => return Ok(None),
            ProjectedBlockRoot::BlockLocal { block, .. } => (block, self.col_range.clone()),
            ProjectedBlockRoot::FullWidth { root, .. } => (root, 0..self.total_dim),
        };
        // `RᵀR` eigenvalues are the squared singular values of `R`; taking them
        // from the root rather than from an eigendecomposition of the assembled
        // `local` keeps the small ones (the whole point of a root-scale
        // representation) and costs O(rank² · block_dim).
        let positive_eigenvalues = match root.svd(false, false) {
            Ok((_, singular_values, _)) => {
                let mut eigs = vec![0.0_f64; root.nrows()];
                for (slot, &sigma) in eigs.iter_mut().zip(singular_values.iter()) {
                    *slot = sigma * sigma;
                }
                eigs
            }
            Err(error) => {
                return Err(EstimationError::LayoutError(format!(
                    "CanonicalPenalty::project_out_null_directions: SVD of the projected root \
                     failed: {error:?}"
                )));
            }
        };
        let local = root.t().dot(&root);
        Ok(Some(Self {
            root: root.into_shared(),
            col_range,
            total_dim: self.total_dim,
            // The projection can only remove curvature, so the block's nullity
            // can only grow; the honest floor is what it already declared.
            nullity: self.nullity,
            local: local.into_shared(),
            positive_eigenvalues,
            op: None,
        }))
    }

    /// Convert to a PenaltyCoordinate for the unified REML evaluator.
    pub fn to_penalty_coordinate(&self) -> gam_problem::PenaltyCoordinate {
        use gam_problem::PenaltyCoordinate;
        if self.is_block_local() {
            PenaltyCoordinate::from_block_root(
                self.root.to_owned(),
                self.col_range.start,
                self.col_range.end,
                self.total_dim,
            )
        } else {
            PenaltyCoordinate::from_dense_root(self.root.to_owned())
        }
    }
}

/// Measure and report how close each pair of penalties in the canonical bundle
/// comes to being proportional.
///
/// Two penalties `S_i`, `S_j` with the same `col_range` are compared by the
/// **relative defect of the proportionality claim itself**,
///
/// ```text
///     delta(S_i, S_j) = min_c ||S_j - c S_i||_F / ||S_i||_F,
///     attained at      c = <S_i, S_j>_F / ||S_i||_F^2,
/// ```
///
/// which is what `sum_i w_i S_i = 0` actually asks about. The cosine is the
/// same information (`delta^2 = 1 - cos^2` for unit-normalized operators), but
/// it is the WRONG COORDINATE to report it in and this issue is the cost of
/// having done so: `delta` enters the cosine SQUARED, so a pair that is
/// `1.9e-5` apart — five orders above any arithmetic — prints as
/// `cos = 1.000000` and reads as an exact identity. #2676 ran on that reading
/// for its whole life. Measured on `geo_disease_matern` (centers=10, n=1500,
/// n_pcs=16, the #2676 penalty-map probe):
///
/// ```text
///     length_scale   delta       certified nullity
///       2.05e-2      2.079e-15          1     <- exact, to the residual's own round-off
///       1.64e-1      1.874e-5           0     <- prints cos = 1.000000
///       1.27e0       3.396e-1           0     <- what the fit realizes
/// ```
///
/// so the "structural identity" that issue is named for is the small-length-
/// scale LIMIT of two genuinely different operators, and it is not present at
/// the geometry any of those fits settle on.
/// Because `local` is symmetric, `<A, B>_F = sum_{r,c} A[r,c] * B[r,c] =
/// tr(A·B)`. Pairs with different `col_range` cannot be proportional by
/// construction and are skipped.
///
/// # ⚠ This detector is a SCREEN, not the authority (#2676)
///
/// It sees proportional PAIRS. The criterion's actual invariance is the null
/// space of the penalty map's Gram, `{w : sum_i w_i S_i = 0}`, which contains
/// every linear redundancy — including three-term ones no pairwise measure can
/// find. `gam_solve::penalty_invariance::PenaltyMapInvariance` computes that
/// subspace and is what the curvature certificate and the smoothing correction
/// consult. Use this function for the human-facing report; do not use it to
/// decide anything numeric.
///
/// # ⚠ And an EXACT redundancy is not a saddle
///
/// This warning used to say the redundancy makes the LAML cost carry "a
/// Z₂-symmetric saddle". It does not. Where the redundancy is EXACT the
/// criterion is exactly constant along it in `lambda` — a flat direction, not
/// a descending one — and every apparent negative curvature there is the
/// chain-rule term of
/// `H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)`, whose sign is
/// the sign of a rounding residual. An optimizer converging there has not
/// converged to a saddle; it has converged to a point on a manifold of points
/// that all give the SAME fit, because they all assemble the same penalty.
/// That is a non-identifiability of `lambda`, worth reporting, and not a defect
/// in the answer.
///
/// Where the redundancy is only NEAR-exact none of that holds: the criterion is
/// not constant along the direction, it carries genuine curvature of order
/// `delta^2`, and calling it "structurally identical" asserts a flatness the
/// evidence does not support. Which of the two a pair is is decided below, at
/// the residual's own arithmetic floor, and it is said in the message.
///
/// # The exactness bar, and why it is derived rather than chosen
///
/// `delta` is computed as a residual norm over `m = block_dim^2` products of
/// `O(1)` entries, so its own round-off is `sqrt(m) * EPSILON` — the error of
/// the norm, not of the operators. A pair at or under that is proportional to
/// everything this arithmetic can see; a pair above it is measurably not.
/// Measured, the two populations sit ten orders apart (`2.079e-15` against
/// `1.874e-5` on the sweep above, at `sqrt(m) * EPSILON = 2.2e-15` for
/// `block_dim = 10`), so nothing rides on where in that gap the bar falls.
///
/// Logging policy, `delta`-denominated throughout:
/// - `delta <= sqrt(m) * EPSILON` → `log::debug!` with `[PENALTY-REDUNDANCY]`.
///   Only their combination `lambda_i + c*lambda_j` is identified.
/// - `sqrt(m) * EPSILON < delta <= 1e-1` → `log::debug!` with
///   `[PENALTY-SIMILARITY]`, carrying `delta`. At large scale (`k > 64`) only
///   the three smallest-`delta` such pairs are logged to bound log volume.
///
/// Returns `Vec<(i, j, delta)>` for the **exactly proportional** pairs,
/// primarily to make this function unit-testable without a log capture.
///
/// Performance: this is O(k² · block_dim²); intended to be called exactly
/// once per fit (e.g. from `RemlState::newwith_offset_shared`).
pub fn report_penalty_pair_redundancy(canonical: &[CanonicalPenalty]) -> Vec<(usize, usize, f64)> {
    // A pair `delta` above this is measurably NOT proportional, and calling it
    // "similar" past a tenth of the operator's own size says nothing. Purely a
    // log-volume bound: no verdict is taken on it.
    const SIMILARITY_REPORTING_DEFECT: f64 = 1e-1;
    const LARGE_SCALE_K_THRESHOLD: usize = 64;
    const TOP_SIMILARITY_PAIRS: usize = 3;

    let k = canonical.len();
    let mut redundant: Vec<(usize, usize, f64)> = Vec::new();
    let mut similar: Vec<(usize, usize, f64, f64)> = Vec::new();

    // Pre-compute tr(S_i^2) = sum of squares of S_i entries (Frobenius norm
    // squared). `local` is symmetric, so this equals tr(S_i^T S_i) = tr(S_i^2).
    let trace_sq: Vec<f64> = canonical
        .iter()
        .map(|p| p.local.iter().map(|&v| v * v).sum::<f64>())
        .collect();

    for i in 0..k {
        if trace_sq[i] == 0.0 {
            continue;
        }
        for j in (i + 1)..k {
            if trace_sq[j] == 0.0 {
                continue;
            }
            // Different col_range → cannot be proportional by construction (the
            // block-local matrices live in disjoint or mismatched parameter
            // subspaces).
            if canonical[i].col_range != canonical[j].col_range {
                continue;
            }
            // Shapes must match — they do when col_range matches because
            // `local` is `block_dim × block_dim` and `block_dim = col_range.len()`.
            assert_eq!(canonical[i].local.dim(), canonical[j].local.dim());

            let inner: f64 = canonical[i]
                .local
                .iter()
                .zip(canonical[j].local.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            // The least-squares proportionality constant and the residual it
            // leaves, formed DIRECTLY rather than through `1 - cos`: the cosine
            // route squares `delta` and then subtracts from one, which loses
            // every digit of a defect under `sqrt(EPSILON)` — the loss this
            // whole issue is made of.
            let scale = inner / trace_sq[i];
            let residual_sq: f64 = canonical[i]
                .local
                .iter()
                .zip(canonical[j].local.iter())
                .map(|(&a, &b)| {
                    let residual = b - scale * a;
                    residual * residual
                })
                .sum();
            let defect = (residual_sq / trace_sq[i]).sqrt();
            let entries = canonical[i].local.len() as f64;
            let arithmetic_floor = entries.sqrt() * f64::EPSILON;

            if defect <= arithmetic_floor {
                redundant.push((i, j, defect));
            } else if defect <= SIMILARITY_REPORTING_DEFECT {
                similar.push((i, j, defect, scale));
            }
        }
    }

    // Always emit every exact redundancy — these are structural model errors.
    for &(i, j, defect) in &redundant {
        log::debug!(
            "[PENALTY-REDUNDANCY] penalties i={i} j={j} are proportional to the arithmetic \
             that formed them (relative defect min_c ||S_{j} - c S_{i}||_F / ||S_{i}||_F = \
             {defect:.6e}) — only their COMBINATION is identified, so the criterion is exactly \
             constant along their antisymmetric direction and lambda_{i} / lambda_{j} are not \
             separately estimable. The fit itself is unaffected (every point of that manifold \
             assembles the same penalty), and the curvature certificate deflates the direction \
             rather than judging a chain-rule term there (#2676). Consider re-specifying (e.g. \
             anisotropic→isotropic for spatial smoothers with weak axis signal) if you need the \
             individual smoothing parameters to mean something."
        );
    }

    // Cap similarity log volume at large scale.
    if k > LARGE_SCALE_K_THRESHOLD && similar.len() > TOP_SIMILARITY_PAIRS {
        similar.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        similar.truncate(TOP_SIMILARITY_PAIRS);
    }
    for (i, j, defect, scale) in similar {
        log::debug!(
            "[PENALTY-SIMILARITY] penalties i={i} j={j} are close but MEASURABLY distinct \
             (relative defect {defect:.6e} at the best scale c={scale:.6e}) — the outer Hessian \
             may be ill-conditioned along their antisymmetric direction, and the criterion \
             carries genuine curvature of order defect^2 there. This is NOT an invariance: \
             nothing is exactly constant along it and no direction is deflated for it (#2676)."
        );
    }

    redundant
}

/// Canonicalize a single `PenaltySpec` into a `CanonicalPenalty` by computing
/// the block-local eigendecomposition and extracting the root.
///
/// This is O(block_dim^3) instead of O(p^3) for block-local penalties.
/// Returns `None` if the penalty has rank zero (should be dropped).
pub fn canonicalize_penalty_spec(
    spec: &crate::PenaltySpec,
    p: usize,
    idx: usize,
    context: &str,
) -> Result<Option<CanonicalPenalty>, EstimationError> {
    use crate::PenaltySpec;

    crate::validate_penalty_spec_shape(idx, spec, p, context)?;

    let (local_matrix, col_range, hint, op) = match spec {
        PenaltySpec::Block {
            local,
            col_range,
            structure_hint,
            op,
        } => (
            local.view(),
            col_range.clone(),
            structure_hint.as_ref(),
            op.clone(),
        ),
        PenaltySpec::Dense(m) => (m.view(), 0..p, None, None),
    };

    let block_dim = col_range.len();

    // ── Ridge fast path: closed-form, no eigendecomposition ──
    if let Some(PenaltyStructureHint::Ridge(scale)) = hint {
        if *scale <= 0.0 {
            return Ok(None);
        }
        let sqrt_scale = scale.sqrt();
        let mut root = Array2::zeros((block_dim, block_dim));
        for i in 0..block_dim {
            root[[i, i]] = sqrt_scale;
        }
        // Ridge penalties are diagonal by construction, but still route through
        // the crate-wide ndarray symmetrizer so every construction variant uses
        // the same "average the transpose" cleanup instead of a local copy.
        let mut local_sym = local_matrix.to_owned();
        symmetrize_in_place(&mut local_sym);
        return Ok(Some(CanonicalPenalty {
            root: root.into_shared(),
            col_range,
            total_dim: p,
            nullity: 0,
            local: local_sym.into_shared(),
            positive_eigenvalues: vec![*scale; block_dim],
            op,
        }));
    }

    // ── Kronecker fast path: single per-factor eigendecomposition ──
    if let Some(PenaltyStructureHint::Kronecker(factors)) = hint {
        let decomps =
            match decompose_kronecker_factors(factors, &format!("{context} penalty {idx}"))? {
                None => return Ok(None),
                Some(d) => d,
            };
        let (positive_eigenvalues, nullity) = kronecker_eigenvalues(&decomps, block_dim);
        if positive_eigenvalues.is_empty() {
            return Ok(None);
        }
        let root = assemble_kronecker_root_local(&decomps);
        // Store the PSD reconstruction RᵀR, as the generic path does, rather
        // than the raw symmetrised input: each factor root keeps only its
        // factor's resolved range, so the raw product can carry directions
        // (the dropped factor eigenvalues) the root has no energy on, and the
        // balanced operator the split is read from would then count them.
        let local = assemble_kronecker_gram_local(&decomps);
        return Ok(Some(CanonicalPenalty {
            root: root.into_shared(),
            col_range,
            total_dim: p,
            nullity,
            local: local.into_shared(),
            positive_eigenvalues,
            op,
        }));
    }

    // ── Generic block-local path: eigendecompose at O(block_dim³) ──
    let local_owned = local_matrix.to_owned();
    let analysis = analyze_penalty_block(&local_owned).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "{context}: penalty canonicalization failed at index {idx}: {err}"
        ))
    })?;

    if analysis.rank == 0 {
        log::trace!(
            "Dropped inactive penalty block idx={idx} reason={}",
            if analysis.iszero {
                "ZeroMatrix"
            } else {
                "NumericalRankZero"
            }
        );
        return Ok(None);
    }

    // Reuse the eigendecomposition from analyze_penalty_block and route the
    // range / null / negative-curvature split through the one canonical
    // classifier, so this root construction cannot disagree with the block's
    // own `rank` / `nullity` / `negative_dim` about which directions are
    // penalized, unpenalized, or non-PSD (#1425).
    let tolerance = analysis.rank_tol;
    let classes = crate::basis::SpectralClassification::new(
        &analysis.eigenvalues,
        tolerance,
        analysis.noise_tol,
    );
    let rank_k = classes.rank();
    assert_eq!(
        rank_k, analysis.rank,
        "penalty-root rank disagreement: SpectralClassification rank={rank_k} vs analyze_penalty_block rank={} (#1425 canonical-classifier invariant)",
        analysis.rank
    );

    // Build the penalty root R from ONLY the range directions (positive
    // curvature): R has one row per range eigenpair, scaled by sqrt(ev), so
    // RᵀR reconstructs S on range(S). Null directions contribute nothing
    // (their eigenvalue is zero); negative-curvature directions are NEVER
    // square-rooted into R (their sqrt is imaginary) and are NOT null — they
    // are simply dropped from R, exactly as the closed-form Duchon kernels at
    // high d require to preserve the q_pen / q_null invariant downstream.
    let mut root = Array2::zeros((rank_k, block_dim));
    let mut positive_eigenvalues = Vec::with_capacity(rank_k);
    for (row_idx, &i) in classes.range_idx.iter().enumerate() {
        let eigenval = analysis.eigenvalues[i];
        let eigenvec = analysis.eigenvectors.column(i);
        root.row_mut(row_idx).assign(&(&eigenvec * eigenval.sqrt()));
        positive_eigenvalues.push(eigenval);
    }

    // Surface any genuine negative curvature honestly: it is neither range
    // (dropped from R) nor null (excluded from `nullity`), so it would
    // otherwise vanish without a trace. A non-PSD penalty reaching this path
    // is a real geometric fact (e.g. high-d Duchon kernels) the operator
    // should be able to see.
    if classes.is_indefinite() {
        log::trace!(
            "{context}: penalty block idx={idx} carries {} negative-curvature \
             eigendirection(s) below -tol={tolerance:e}; dropped from the canonical \
             root and NOT counted as null space (rank={rank_k}, nullity={})",
            classes.negative_dim(),
            classes.nullity()
        );
    }

    // Store the PSD reconstruction RᵀR rather than the raw symmetrised input so
    // the cached `local` matches the rank truncation embedded in `root`
    // (negative-curvature directions are excluded from both, as above).
    let local = root.t().dot(&root);
    Ok(Some(CanonicalPenalty {
        root: root.into_shared(),
        col_range,
        total_dim: p,
        nullity: classes.nullity(),
        local: local.into_shared(),
        positive_eigenvalues,
        op,
    }))
}

/// Canonicalize a batch of penalty specs, dropping zero-rank penalties.
/// Returns (active_penalties, active_nullspace_dims).
pub fn canonicalize_penalty_specs(
    specs: &[crate::PenaltySpec],
    nullspace_dims: &[usize],
    p: usize,
    context: &str,
) -> Result<(Vec<CanonicalPenalty>, Vec<usize>), EstimationError> {
    if specs.len() != nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "{context}: nullspace_dims length mismatch: penalties={}, nullspace_dims={}",
            specs.len(),
            nullspace_dims.len()
        );
    }

    let mut active = Vec::with_capacity(specs.len());
    let mut active_nullspace = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        if let Some(canonical) = canonicalize_penalty_spec(spec, p, idx, context)? {
            active_nullspace.push(nullspace_dims[idx]);
            active.push(canonical);
        }
    }
    Ok((active, active_nullspace))
}

/// Structural rank of each penalty block, for a fit that re-realizes its
/// penalties while ψ moves.
///
/// A generic block counts its resolved eigenvalues through the one Gram-side
/// predicate, [`gam_linalg::roundoff::resolved_eigenvalue_count`], which counts
/// those above the eigensolver's backward-error band `p·ε·‖S‖₂`. A
/// canonicalization spec carries no assembly bound, so no assembly term applies.
/// A hinted block (ridge, Kronecker) keeps the rank of its closed-form root. A
/// joint ρ+ψ fit computes these once at the build ψ, and every later realization
/// is canonicalized at them through [`canonicalize_penalty_specs_at_frozen_ranks`],
/// which reads the same predicate.
pub fn penalty_structural_ranks_at_rounding_band(
    specs: &[crate::PenaltySpec],
    p: usize,
    context: &str,
) -> Result<Vec<usize>, EstimationError> {
    let mut ranks = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        if penalty_spec_has_structure_hint(spec) {
            ranks.push(
                canonicalize_penalty_spec(spec, p, idx, context)?
                    .map_or(0, |canonical| canonical.root.nrows()),
            );
            continue;
        }
        crate::validate_penalty_spec_shape(idx, spec, p, context)?;
        let analysis = analyze_penalty_block(&penalty_spec_local_matrix(spec)).map_err(|err| {
            EstimationError::InvalidInput(format!(
                "{context}: structural rank analysis failed at penalty {idx}: {err}"
            ))
        })?;
        ranks.push(gam_linalg::roundoff::resolved_eigenvalue_count(
            &analysis.eigenvalues.to_vec(),
            0.0,
        ));
    }
    Ok(ranks)
}

/// Canonicalize a batch of penalty specs at structural ranks frozen for the
/// fit, dropping blocks frozen at rank zero.
///
/// [`canonicalize_penalty_spec`] roots a block from the eigenpairs above the
/// spectral rank cutoff, so a moving ψ can push one of them across the cutoff
/// and the priced penalty drops or regains a rank-one piece between two nearby
/// trials. On the periodic Matérn bug-hunt fixture (MSI job 602008, and again
/// at 575e25e4d in job 608882) the REML cost split at psi = 1.098662, -147.4001
/// against -148.0976. Exactly one eigenvalue of `S_λ` moved (4.163e-3 →
/// 3.203e-3), and its drop over λ₄ = 2.06e5 is the cutoff of a normalized
/// 89-column block. Every line search that bracketed that psi failed.
///
/// Here each generic block is rooted from its `frozen_ranks[idx]` largest
/// eigenpairs at every ψ, so the priced penalty is a continuous function of the
/// realized block. If the frozen rank exceeds the block's resolved eigenvalue
/// count at this trial ([`gam_linalg::roundoff::resolved_eigenvalue_count`], the
/// predicate the freeze counted with), the block cannot be rooted at that rank.
/// Two states reach that one count, and the refusal names which: the kept
/// eigenvalue is inside the rounding band, so the rank is UNRESOLVED; or it is
/// resolved below `−band`, so the block is INDEFINITE at this trial and is not
/// the PSD Gram a penalty must be. That trial
/// is refused (`TrialPointRefused`), never re-ranked. Hinted blocks keep their
/// closed-form roots and are refused when that root's rank differs from the
/// frozen rank.
pub fn canonicalize_penalty_specs_at_frozen_ranks(
    specs: &[crate::PenaltySpec],
    nullspace_dims: &[usize],
    frozen_ranks: &[usize],
    p: usize,
    context: &str,
) -> Result<(Vec<CanonicalPenalty>, Vec<usize>), EstimationError> {
    if specs.len() != nullspace_dims.len() || specs.len() != frozen_ranks.len() {
        crate::bail_invalid_estim!(
            "{context}: penalty topology mismatch: penalties={}, nullspace_dims={}, frozen ranks={}",
            specs.len(),
            nullspace_dims.len(),
            frozen_ranks.len()
        );
    }
    let mut active = Vec::with_capacity(specs.len());
    let mut active_nullspace = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let frozen_rank = frozen_ranks[idx];
        let canonical = if penalty_spec_has_structure_hint(spec) {
            let canonical = canonicalize_penalty_spec(spec, p, idx, context)?;
            let realized_rank = canonical.as_ref().map_or(0, |c| c.root.nrows());
            if realized_rank != frozen_rank {
                return Err(EstimationError::TrialPointRefused {
                    reason: format!(
                        "{context}: hinted penalty block idx={idx} has root rank {realized_rank} \
                         at this trial but the fit froze it at {frozen_rank}"
                    ),
                });
            }
            canonical
        } else {
            canonicalize_penalty_spec_at_frozen_rank(spec, p, idx, frozen_rank, context)?
        };
        if let Some(canonical) = canonical {
            active_nullspace.push(nullspace_dims[idx]);
            active.push(canonical);
        }
    }
    Ok((active, active_nullspace))
}

fn penalty_spec_has_structure_hint(spec: &crate::PenaltySpec) -> bool {
    matches!(
        spec,
        crate::PenaltySpec::Block {
            structure_hint: Some(_),
            ..
        }
    )
}

fn penalty_spec_local_matrix(spec: &crate::PenaltySpec) -> Array2<f64> {
    match spec {
        crate::PenaltySpec::Block { local, .. } => local.to_owned(),
        crate::PenaltySpec::Dense(matrix) => matrix.to_owned(),
    }
}

fn canonicalize_penalty_spec_at_frozen_rank(
    spec: &crate::PenaltySpec,
    p: usize,
    idx: usize,
    frozen_rank: usize,
    context: &str,
) -> Result<Option<CanonicalPenalty>, EstimationError> {
    crate::validate_penalty_spec_shape(idx, spec, p, context)?;
    if frozen_rank == 0 {
        return Ok(None);
    }
    let (col_range, op) = match spec {
        crate::PenaltySpec::Block { col_range, op, .. } => (col_range.clone(), op.clone()),
        crate::PenaltySpec::Dense(_) => (0..p, None),
    };
    let block_dim = col_range.len();
    if frozen_rank > block_dim {
        crate::bail_invalid_estim!(
            "{context}: penalty {idx} frozen at rank {frozen_rank} exceeds its block dimension {block_dim}"
        );
    }
    let analysis = analyze_penalty_block(&penalty_spec_local_matrix(spec)).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "{context}: penalty canonicalization failed at index {idx}: {err}"
        ))
    })?;
    let mut descending: Vec<usize> = (0..analysis.eigenvalues.len()).collect();
    descending.sort_by(|&a, &b| analysis.eigenvalues[b].total_cmp(&analysis.eigenvalues[a]));
    let kept = &descending[..frozen_rank];
    let smallest_kept = analysis.eigenvalues[kept[frozen_rank - 1]];
    // The predicate the freeze counted with
    // (`penalty_structural_ranks_at_rounding_band`): a direction is penalized iff
    // its eigenvalue is resolved ABOVE the band, so `resolved_eigenvalue_count`
    // counts only `λ > band` and a resolved NEGATIVE eigenvalue is not a
    // penalized direction either. The threshold is read from the one predicate
    // (`resolved_eigenvalue_band`) rather than recomputed, so the number the
    // refusal prints is the number the count compared against.
    //
    // TWO MECHANISMS, NOT ONE. The eigenvalues are sorted descending, so the
    // `frozen_rank`-th largest failing `> band` means either
    //   (a) it lies inside ±band: the block's rank at this trial is UNRESOLVED.
    //       Pricing the direction would put `ln(roundoff)` into `log|S|₊`, and
    //       its sign carries no information; or
    //   (b) it lies below −band: the block is INDEFINITE here. That eigenvalue
    //       is resolved, and its sign is a measurement — the block is not the
    //       PSD Gram a penalty must be, and `analyze_penalty_block` reports such
    //       a direction as negative curvature rather than refusing it.
    // Both refuse this trial, because neither gives a rank-`frozen_rank` PSD
    // root, but they are different findings and a refusal that names (a) for a
    // block in state (b) sends the reader to the rounding band for a defect that
    // is nowhere near it (gam#2959).
    let eigenvalues = analysis.eigenvalues.to_vec();
    let band = gam_linalg::roundoff::resolved_eigenvalue_band(&eigenvalues, 0.0);
    if gam_linalg::roundoff::resolved_eigenvalue_count(&eigenvalues, 0.0) < frozen_rank {
        let head = format!(
            "{context}: penalty block idx={idx} was frozen at structural rank {frozen_rank}, \
             but at this trial its {frozen_rank}-th largest eigenvalue {smallest_kept:e}"
        );
        let reason = if smallest_kept < -band {
            format!(
                "{head} is negative and resolved below the Gram's rounding band {band:e}: \
                 the block is indefinite at this trial, not unresolved, and it carries \
                 {} resolved penalized direction(s)",
                gam_linalg::roundoff::resolved_eigenvalue_count(&eigenvalues, 0.0)
            )
        } else {
            format!("{head} is unresolved within the Gram's rounding band {band:e}")
        };
        return Err(EstimationError::TrialPointRefused { reason });
    }
    let mut root = Array2::zeros((frozen_rank, block_dim));
    let mut positive_eigenvalues = Vec::with_capacity(frozen_rank);
    for (row_idx, &i) in kept.iter().enumerate() {
        let eigenval = analysis.eigenvalues[i];
        root.row_mut(row_idx)
            .assign(&(&analysis.eigenvectors.column(i) * eigenval.sqrt()));
        positive_eigenvalues.push(eigenval);
    }
    let local = root.t().dot(&root);
    Ok(Some(CanonicalPenalty {
        root: root.into_shared(),
        col_range,
        total_dim: p,
        nullity: block_dim - frozen_rank,
        local: local.into_shared(),
        positive_eigenvalues,
        op,
    }))
}

/// Hard cap on the dimension `p` allowed to fall back to a dense p × p
/// eigendecomposition of the overlapping balanced penalty.
///
/// Beyond this cap the overlapping-penalty path errors out instead of
/// allocating an O(p²) workspace whose eigendecomposition would dominate the
/// solve. ResourcePolicy threading is the long-term home for this cap (the
/// resource_serialize agent is widening ResourcePolicy coverage); until that
/// lands, the overlapping branch of the reparameterization reads it from here.
pub(crate) const OVERLAPPING_PENALTY_DENSE_FALLBACK_MAX_P: usize = 4096;

/// The λ-invariant penalty operator the structural rank is read from:
/// `B = Σ_k S_k / ‖S_k‖_F`, each component embedded on its own column range,
/// together with the bound on the rounding its formation left in it.
///
/// # One rule, two readers
///
/// The reparameterization splits the coefficient space into the penalized
/// subspace `H + S̃` carries and the λ-invariant null space from THIS operator's
/// spectrum. The outer criterion's `−½ log|S(λ)|₊` must range over exactly that
/// penalized subspace — its asymptotic slope in `ρ_k` is `½(rank(S̃) −
/// rank|S|₊)` per unit `ρ_k`, so any disagreement in rank is a criterion with no
/// interior optimum along that coordinate. The criterion used to take its
/// structural rank from the UNWEIGHTED sum `Σ_k S_k` at a `100·p·ε·max` cut,
/// which is a different function of the same penalties: a component whose
/// Frobenius norm is small against its neighbours (a double-penalty null-space
/// term beside a Matérn range penalty) can sit above one cut and below the
/// other. Measured on the iso-κ Matérn double-penalty ladder (gam#2454):
/// `logdet_rank = 10` against `penalized_rank = 9`, and the gate
/// `penalty_logdet_ranks_the_same_subspace_the_hessian_carries_2454` red on
/// main. Normalizing every component first makes the rank a property of the
/// penalties' supports, not of their scales, which is what "structural" means.
///
/// # The rank predicate
///
/// A direction is penalized iff its eigenvalue of `B` is resolved from zero:
/// `gam_linalg::roundoff::resolved_eigenvalue_count(eigs, assembly_band)`, the
/// eigensolver's backward error `d·ε·max|λ|` plus [`Self::assembly_band`]. Both
/// readers — [`balanced_penalty_structural_rank`] and the reparameterization's
/// split in [`precompute_reparam_invariant_from_canonical`] — call that one
/// predicate on the same blocks (#4057), so no independent relative constant can
/// make them disagree.
pub struct BalancedPenalty {
    /// `B`, over the block's own columns.
    pub matrix: Array2<f64>,
    /// Spectral-norm bound, in the units of `B`, on `B̂ − B` where `B` is the
    /// balanced sum of the exact Grams the components represent.
    ///
    /// Each component is a PSD Gram `S_k = R_kᵀR_k` of at most `d_k` root rows
    /// (the [`CanonicalPenalty`] contract: `local` is the reconstruction of the
    /// stored root), so its computed entries carry `γ_{d_k}·(|R_k|ᵀ|R_k|)`;
    /// scaling by `1/‖S_k‖_F` rounds once more and the `K` embedded components
    /// of a block are added with `K − 1` additions. The entrywise majorant is
    /// `Σ_k γ_{d_k+K+1}·|R_k|ᵀ|R_k| / ‖S_k‖_F`, PSD, so by Perron–Frobenius its
    /// spectral norm — and the error's — is at most its trace
    /// `Σ_k γ_{d_k+K+1}·tr(S_k)/‖S_k‖_F`, with `tr(S_k) = Σ_i |S_k,ii|`
    /// (`gam_linalg::roundoff::weighted_gram_assembly_band`). The rounding of
    /// the scale `1/‖S_k‖_F` itself only rescales a component by `1 + O(ε)`,
    /// which moves no eigenvalue across zero.
    pub assembly_band: f64,
}

/// Assemble [`BalancedPenalty`] on `p_total` columns. Components with zero
/// Frobenius norm carry no structure and are skipped.
pub fn balanced_penalty_sum<'a, I>(components: I, p_total: usize) -> BalancedPenalty
where
    I: IntoIterator<Item = (ArrayView2<'a, f64>, std::ops::Range<usize>)>,
{
    let kept: Vec<(ArrayView2<'a, f64>, std::ops::Range<usize>, f64)> = components
        .into_iter()
        .filter_map(|(local, range)| {
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            (frob_norm > 0.0).then_some((local, range, frob_norm))
        })
        .collect();
    let additions = kept.len().saturating_sub(1);
    let mut matrix = Array2::<f64>::zeros((p_total, p_total));
    let mut assembly_band = 0.0_f64;
    for (local, range, frob_norm) in &kept {
        let scale = 1.0 / frob_norm;
        for i in 0..local.nrows() {
            for j in 0..local.ncols() {
                matrix[[range.start + i, range.start + j]] += scale * local[[i, j]];
            }
        }
        let trace: f64 = local.diag().iter().map(|value| value.abs()).sum();
        // Gram depth `d_k` (inner products of at most `d_k` root rows, `k =
        // d_k − 1 + 1`), one scale rounding, and the block's additions.
        assembly_band += gam_linalg::roundoff::weighted_gram_assembly_band(
            local.nrows(),
            2 + additions,
            trace * scale,
        );
    }
    BalancedPenalty {
        matrix,
        assembly_band,
    }
}

/// One block of the balanced penalty operator's partition: the components
/// whose column ranges are exactly `col_range` (or, when two distinct ranges
/// overlap, every component on one global block), assembled over the block.
struct BalancedPenaltyBlock {
    col_range: Range<usize>,
    /// Indices of the member components, in input order.
    members: Vec<usize>,
    balanced: BalancedPenalty,
    /// Every member is diagonal, so `B` is and its spectrum is its diagonal.
    diagonal: bool,
}

/// The column blocks the balanced operator is block-diagonal over, as
/// `(col_range, member component indices)`, and whether two distinct ranges
/// overlap. Components with no nonzero entry carry no structure and are
/// dropped. When ranges overlap the operator does not decompose, and the single
/// group is the whole `0..p_total`.
fn balanced_penalty_groups(
    components: &[(ArrayView2<'_, f64>, Range<usize>)],
    p_total: usize,
) -> (Vec<(Range<usize>, Vec<usize>)>, bool) {
    let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (index, (local, range)) in components.iter().enumerate() {
        if local.iter().any(|&value| value != 0.0) {
            groups.entry((range.start, range.end)).or_default().push(index);
        }
    }
    // Sorted by start: a range overlaps an earlier one iff it starts before the
    // furthest end seen so far (a running maximum, since a short range nested in
    // a long one does not bound the ranges after it).
    let mut furthest_end = 0usize;
    let mut overlapping = false;
    for &(start, end) in groups.keys() {
        if start < furthest_end {
            overlapping = true;
            break;
        }
        furthest_end = furthest_end.max(end);
    }
    if overlapping {
        let mut members: Vec<usize> = groups.into_values().flatten().collect();
        members.sort_unstable();
        return (vec![(0..p_total, members)], true);
    }
    (
        groups
            .into_iter()
            .map(|((start, end), members)| (start..end, members))
            .collect(),
        false,
    )
}

/// Assemble one group of [`balanced_penalty_groups`] over its own columns.
fn balanced_penalty_block(
    components: &[(ArrayView2<'_, f64>, Range<usize>)],
    col_range: Range<usize>,
    members: Vec<usize>,
) -> BalancedPenaltyBlock {
    let offset = col_range.start;
    let balanced = balanced_penalty_sum(
        members.iter().map(|&index| {
            let (local, member_range) = &components[index];
            (
                local.view(),
                (member_range.start - offset)..(member_range.end - offset),
            )
        }),
        col_range.len(),
    );
    let diagonal = members
        .iter()
        .all(|&index| is_diagonal(components[index].0.view()));
    BalancedPenaltyBlock {
        col_range,
        members,
        balanced,
        diagonal,
    }
}

/// Bound on the relative root energy `‖R_k Q̂_n‖_F² / ‖R_k‖_F²` a penalty of a
/// balanced block keeps on the `null_count` computed null directions `Q̂_n` of
/// that block, when the block's eigenvalues were read against `resolved_band`.
///
/// A computed null eigenpair `(λ̂, q̂)` of `B̂` has `|λ̂| ≤ resolved_band` (it
/// was not resolved from zero) and residual `‖B̂q̂ − λ̂q̂‖ ≤ d·ε·‖B̂‖ ≤
/// resolved_band` (backward stability), and `‖B − B̂‖₂ ≤ assembly_band ≤
/// resolved_band`, so `q̂ᵀBq̂ ≤ 2·resolved_band·‖q̂‖²`. Each member satisfies
/// `S_k/‖S_k‖_F ≼ B`, so `tr(Q̂_nᵀS_kQ̂_n) ≤ 2·n·resolved_band·‖S_k‖_F·‖q̂‖²`,
/// and `‖S_k‖_F ≤ tr(S_k) = ‖R_k‖_F²` for a PSD Gram: the exact product's
/// relative null energy is at most `2·n·resolved_band·‖q̂‖²`. Forming `R_k Q̂`
/// in floating point adds at most `γ_d·|R_k||Q̂_n|` entrywise, whose Frobenius
/// norm is at most `γ_d·√n·‖R_k‖_F`, and `‖q̂‖² ≤ 1 + γ_d`. Hence the bound
/// `(1 + γ_d)·(√(2·n·resolved_band) + γ_d·√n)²`. Energy above it is not
/// roundoff of the split: the roots and the operator the split was read from
/// disagree.
fn balanced_null_leakage_tolerance(block_dim: usize, null_count: usize, resolved_band: f64) -> f64 {
    if null_count == 0 {
        return 0.0;
    }
    let gamma = gam_linalg::roundoff::accumulation_growth(block_dim);
    let n = null_count as f64;
    let amplitude = (2.0 * n * resolved_band).sqrt() + gamma * n.sqrt();
    (1.0 + gamma) * amplitude * amplitude
}

/// Spectrum of a balanced block: a diagonal block's is its diagonal (read in
/// `O(d)`, eigenvectors the coordinate vectors, returned as `None`); any other
/// block is symmetrized by averaging with its transpose (the same cleanup
/// [`robust_eigh_faer`] applies) and eigendecomposed. Both rank readers call
/// this one routine, so they count the same computed eigenvalues.
fn balanced_block_eigh(
    block: &BalancedPenaltyBlock,
) -> Result<(Vec<f64>, Option<Mat<f64>>), EstimationError> {
    if block.balanced.matrix.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!(
            "balanced penalty block {:?} contains non-finite entries",
            block.col_range
        );
    }
    if block.diagonal {
        return Ok((block.balanced.matrix.diag().to_vec(), None));
    }
    let symmetric = sanitize_symmetric_faer(&array_to_faer(&block.balanced.matrix));
    let (values, vectors) =
        gam_linalg::faer_ndarray::self_adjoint_evd(symmetric.as_ref(), Side::Lower).map_err(
            |err| EstimationError::EigendecompositionFailed(FaerLinalgError::SelfAdjointEigen(err)),
        )?;
    Ok(((0..values.dim()).map(|idx| values[idx]).collect(), Some(vectors)))
}

/// Structural rank of a set of penalty components: the number of resolved
/// eigenvalues of each [`balanced_penalty_groups`] block
/// (`resolved_eigenvalue_count` at the block's [`BalancedPenalty::assembly_band`]),
/// summed. This is the rank the reparameterization's penalized subspace has —
/// [`precompute_reparam_invariant_from_canonical`] partitions and counts with the
/// same two calls — and the rank the criterion's `log|S(λ)|₊` must range over.
///
/// Only the count is read here, so no PSD classification is applied: a
/// resolved negative eigenvalue is simply not counted, and the reparameterization
/// is where an indefinite penalty is refused.
pub fn balanced_penalty_structural_rank<'a, I>(
    components: I,
    p_total: usize,
) -> Result<usize, EstimationError>
where
    I: IntoIterator<Item = (ArrayView2<'a, f64>, std::ops::Range<usize>)>,
{
    if p_total == 0 {
        return Ok(0);
    }
    let components: Vec<(ArrayView2<'a, f64>, Range<usize>)> = components.into_iter().collect();
    let (groups, _) = balanced_penalty_groups(&components, p_total);
    let mut rank = 0usize;
    for (col_range, members) in groups {
        let block = balanced_penalty_block(&components, col_range, members);
        let (eigenvalues, _) = balanced_block_eigh(&block)?;
        rank += gam_linalg::roundoff::resolved_eigenvalue_count(
            &eigenvalues,
            block.balanced.assembly_band,
        );
    }
    Ok(rank)
}

#[derive(Clone)]
struct SubspaceSplit {
    q_pen: Array2<f64>,
    q_null: Array2<f64>,
}

impl SubspaceSplit {
    fn identity(p: usize) -> Self {
        Self {
            q_pen: Array2::zeros((p, 0)),
            q_null: Array2::eye(p),
        }
    }

    fn from_ordered_qs(
        qs: &Mat<f64>,
        penalized_rank: usize,
        p: usize,
    ) -> Result<Self, EstimationError> {
        if qs.nrows() != p || qs.ncols() != p {
            return Err(EstimationError::LayoutError(format!(
                "Invalid Q basis dimensions: expected {p}x{p}, got {}x{}",
                qs.nrows(),
                qs.ncols()
            )));
        }
        if penalized_rank > p {
            return Err(EstimationError::LayoutError(format!(
                "Invalid penalized rank {penalized_rank} for p={p}"
            )));
        }

        let null_count = p - penalized_rank;
        let mut q_pen = Array2::<f64>::zeros((p, penalized_rank));
        let mut q_null = Array2::<f64>::zeros((p, null_count));
        for i in 0..p {
            for j in 0..penalized_rank {
                q_pen[(i, j)] = qs[(i, j)];
            }
            for j in 0..null_count {
                q_null[(i, j)] = qs[(i, penalized_rank + j)];
            }
        }

        Ok(Self { q_pen, q_null })
    }

    fn rank(&self) -> usize {
        self.q_pen.ncols()
    }

    fn p(&self) -> usize {
        self.q_pen.nrows()
    }

    fn compose_qs(&self) -> Array2<f64> {
        let p = self.p();
        let rank = self.rank();
        let null_count = self.q_null.ncols();
        let mut qs = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..rank {
                qs[(i, j)] = self.q_pen[(i, j)];
            }
            for j in 0..null_count {
                qs[(i, rank + j)] = self.q_null[(i, j)];
            }
        }
        qs
    }
}

/// Lambda-independent reparameterization invariants derived from penalty structure.
#[derive(Clone)]
pub struct ReparamInvariant {
    split: SubspaceSplit,
    /// The balanced eigenvector matrix Q (p x p). Block-local roots are
    /// transformed on-the-fly as `R_block @ Q[start..end, :]` instead of
    /// storing pre-multiplied full-width roots.
    qs_base: Array2<f64>,
    has_nonzero: bool,
    /// The disjoint column blocks the penalties partition into, when they do.
    /// `None` when two distinct penalty column ranges overlap, in which case
    /// the split came from one global eigendecomposition and has no block
    /// structure to exploit.
    blocks: Option<Vec<InvariantBlock>>,
    /// Largest relative root energy `‖R_k Q_n‖_F² / ‖R_k‖_F²` any penalty may
    /// keep on the null columns as roundoff of this split
    /// ([`balanced_null_leakage_tolerance`] of each block); zero when nothing
    /// is penalized.
    null_leakage_tolerance: f64,
}

/// One disjoint penalty block of a [`ReparamInvariant`]: every penalty whose
/// column range is exactly `col_range`, and that block's share of the split.
#[derive(Clone)]
struct InvariantBlock {
    col_range: Range<usize>,
    penalty_indices: Vec<usize>,
    /// `block_dim × pen_rank` penalized directions of this block.
    q_pen_local: Array2<f64>,
    /// `block_dim × null_rank` unpenalized directions of this block.
    q_null_local: Array2<f64>,
    /// Set when every member penalty's local Gram is diagonal (a ridge or a
    /// random-effect variance). The penalized directions are then the
    /// coordinate vectors of these block-local columns, in `q_pen_local`
    /// order, and every λ-dependent quantity is per-coordinate arithmetic.
    diagonal_pen_cols: Option<Vec<usize>>,
}

/// Whether every off-diagonal entry of `matrix` is exactly zero.
pub fn is_diagonal(matrix: ArrayView2<'_, f64>) -> bool {
    matrix
        .indexed_iter()
        .all(|((i, j), &value)| i == j || value == 0.0)
}

/// Precompute the lambda-invariant reparameterization structure from canonical penalties.
///
/// Uses block-local roots directly instead of requiring rank x p global roots.
/// Each `CanonicalPenalty` carries its own block-local root and column range,
/// so the balanced sum can be assembled without ever materializing full-size
/// penalty matrices.
///
/// The partition into blocks, the balanced operator of each block, its
/// spectrum and the penalized/null cut are exactly those
/// [`balanced_penalty_structural_rank`] reads (`balanced_penalty_groups`,
/// `balanced_penalty_block`, `balanced_block_eigh`, `resolved_eigenvalue_count`),
/// so the penalized subspace here has the structural rank the criterion's
/// `log|S(λ)|₊` ranges over by construction (#4057).
pub fn precompute_reparam_invariant_from_canonical(
    penalties: &[CanonicalPenalty],
    p_total: usize,
) -> Result<ReparamInvariant, EstimationError> {
    use std::cmp::Ordering;

    // A penalty with an empty root carries no penalized direction.
    let active: Vec<usize> = (0..penalties.len())
        .filter(|&index| penalties[index].rank() > 0)
        .collect();
    let components: Vec<(ArrayView2<'_, f64>, Range<usize>)> = active
        .iter()
        .map(|&index| {
            (
                penalties[index].local_ref().view(),
                penalties[index].col_range.clone(),
            )
        })
        .collect();
    let (groups, overlapping) = balanced_penalty_groups(&components, p_total);

    if groups.is_empty() {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p_total),
            qs_base: Array2::eye(p_total),
            has_nonzero: false,
            blocks: None,
            null_leakage_tolerance: 0.0,
        });
    }

    if overlapping && p_total > OVERLAPPING_PENALTY_DENSE_FALLBACK_MAX_P {
        // Without this guard, large-scale models with overlapping penalties allocated a full
        // p_total × p_total workspace and ran an O(p³) eigendecomposition
        // before any solver code saw the problem size.
        return Err(EstimationError::LayoutError(format!(
            "overlapping penalty reparameterization would require dense {}x{} eigendecomposition; \
             large-model dense fallback is disabled. Keep penalties structured or \
             extend the overlapping-penalty solver path",
            p_total, p_total
        )));
    }

    struct BlockSplit {
        col_range: Range<usize>,
        /// Indices into `penalties`.
        penalty_indices: Vec<usize>,
        q_pen_local: Array2<f64>,  // block_dim × pen_rank
        q_null_local: Array2<f64>, // block_dim × null_rank
        diagonal_pen_cols: Option<Vec<usize>>,
        null_leakage_tolerance: f64,
    }

    let context = if overlapping {
        "balanced penalty matrix"
    } else {
        "balanced penalty block"
    };
    let split_block = |(col_range, members): (Range<usize>, Vec<usize>)| -> Result<
        BlockSplit,
        EstimationError,
    > {
        let block = balanced_penalty_block(&components, col_range, members);
        let block_dim = block.col_range.len();
        let assembly_band = block.balanced.assembly_band;
        // A diagonal block (every member a ridge or random-effect variance)
        // has the coordinate vectors as eigenvectors, and its spectrum is read
        // off the diagonal in O(block_dim) instead of an O(block_dim³) solve.
        let (mut eigenvalues, eigenvectors) = balanced_block_eigh(&block)?;
        let resolved_band =
            gam_linalg::roundoff::resolved_eigenvalue_band(&eigenvalues, assembly_band);
        let penalized_rank =
            gam_linalg::roundoff::resolved_eigenvalue_count(&eigenvalues, assembly_band);
        // Same band: every eigenvalue this snaps to zero is one the count
        // above left out, and a resolved negative one is refused.
        classify_eigenvalues_strict(&mut eigenvalues, assembly_band, context)?;

        let mut order: Vec<usize> = (0..block_dim).collect();
        order.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(Ordering::Equal)
                .then(i.cmp(&j))
        });
        let null_count = block_dim - penalized_rank;

        let mut q_pen_local = Array2::zeros((block_dim, penalized_rank));
        let mut q_null_local = Array2::zeros((block_dim, null_count));
        for (col_idx, &idx) in order.iter().enumerate() {
            let mut target = if col_idx < penalized_rank {
                q_pen_local.column_mut(col_idx)
            } else {
                q_null_local.column_mut(col_idx - penalized_rank)
            };
            match eigenvectors.as_ref() {
                Some(vectors) => {
                    for row in 0..block_dim {
                        target[row] = vectors[(row, idx)];
                    }
                }
                None => target[idx] = 1.0,
            }
        }
        let diagonal_pen_cols = eigenvectors
            .is_none()
            .then(|| order[..penalized_rank].to_vec());

        Ok(BlockSplit {
            col_range: block.col_range,
            penalty_indices: block.members.iter().map(|&member| active[member]).collect(),
            q_pen_local,
            q_null_local,
            diagonal_pen_cols,
            null_leakage_tolerance: balanced_null_leakage_tolerance(
                block_dim,
                null_count,
                resolved_band,
            ),
        })
    };

    // Groups come out of `balanced_penalty_groups` in column order; collecting
    // the indexed parallel iterator keeps that order while each independent
    // block is eigendecomposed concurrently.
    let block_splits: Vec<BlockSplit> = groups
        .into_par_iter()
        .map(split_block)
        .collect::<Result<_, _>>()?;
    let null_leakage_tolerance = block_splits
        .iter()
        .map(|block| block.null_leakage_tolerance)
        .fold(0.0_f64, f64::max);

    if overlapping {
        // One global block over all `p_total` columns.
        let block = &block_splits[0];
        let penalized_rank = block.q_pen_local.ncols();
        let mut qs = Mat::<f64>::zeros(p_total, p_total);
        for row in 0..p_total {
            for col in 0..penalized_rank {
                qs[(row, col)] = block.q_pen_local[[row, col]];
            }
            for col in 0..block.q_null_local.ncols() {
                qs[(row, penalized_rank + col)] = block.q_null_local[[row, col]];
            }
        }
        let split = SubspaceSplit::from_ordered_qs(&qs, penalized_rank, p_total)?;
        return Ok(ReparamInvariant {
            split,
            qs_base: mat_to_array(&qs),
            has_nonzero: true,
            blocks: None,
            null_leakage_tolerance,
        });
    }

    // -----------------------------------------------------------------------
    // Non-overlapping: block-diagonal eigendecomposition at O(Σ p_k³).
    // -----------------------------------------------------------------------
    // The balanced sum is block-diagonal ⟹ its eigenvectors are block-local.
    // Q_pen and Q_null are assembled by embedding block-local eigenvectors;
    // columns no block covers are unpenalized coordinate directions.
    let mut covered = vec![false; p_total];
    for block in &block_splits {
        for j in block.col_range.clone() {
            covered[j] = true;
        }
    }
    let uncovered_cols: Vec<usize> = (0..p_total).filter(|&j| !covered[j]).collect();

    let total_pen_rank: usize = block_splits
        .iter()
        .map(|block| block.q_pen_local.ncols())
        .sum();
    let block_null: usize = block_splits
        .iter()
        .map(|block| block.q_null_local.ncols())
        .sum();
    let mut q_pen = Array2::zeros((p_total, total_pen_rank));
    let mut q_null = Array2::zeros((p_total, block_null + uncovered_cols.len()));
    let mut pen_offset = 0usize;
    let mut null_offset = 0usize;
    for block in &block_splits {
        let rows = block.col_range.clone();
        let pen_rank = block.q_pen_local.ncols();
        let null_rank = block.q_null_local.ncols();
        q_pen
            .slice_mut(s![rows.clone(), pen_offset..(pen_offset + pen_rank)])
            .assign(&block.q_pen_local);
        q_null
            .slice_mut(s![rows, null_offset..(null_offset + null_rank)])
            .assign(&block.q_null_local);
        pen_offset += pen_rank;
        null_offset += null_rank;
    }
    for &j in &uncovered_cols {
        q_null[[j, null_offset]] = 1.0;
        null_offset += 1;
    }

    let split = SubspaceSplit { q_pen, q_null };
    let blocks = block_splits
        .into_iter()
        .map(|block| InvariantBlock {
            col_range: block.col_range,
            penalty_indices: block.penalty_indices,
            q_pen_local: block.q_pen_local,
            q_null_local: block.q_null_local,
            diagonal_pen_cols: block.diagonal_pen_cols,
        })
        .collect();

    // Store the global Q_s = [Q_pen | Q_null] from the split.
    // Block-local roots are transformed on-the-fly as R_block @ Q[start..end, :]
    // inside the reparam engine, avoiding O(k * rank * p) storage.
    let qs_global = split.compose_qs();

    Ok(ReparamInvariant {
        split,
        qs_base: qs_global,
        has_nonzero: true,
        blocks: Some(blocks),
        null_leakage_tolerance,
    })
}

/// Apply stable reparameterization using precomputed lambda-invariant structures.
///
/// Write the penalized-block spectrum and rotation from a right-singular
/// factorization of the stacked scaled roots `E = [√λₖ Rₖ]ₖ`.
///
/// Both admissible routes to that factorization — the SVD of `E` and the SVD of
/// its Householder QR factor `R` — produce the identical pair, because
/// `EᵀE = RᵀR`. Absorbing them through one function is what makes "the same two
/// outputs by a different computation" a fact of the code rather than a claim
/// about it.
fn absorb_right_singular_factorization(
    singular_values: &Array1<f64>,
    vt: &Array2<f64>,
    penalized_rank: usize,
    range_eigenvalues_sorted: &mut Vec<f64>,
    range_rotation: &mut Mat<f64>,
) {
    // `singular_values` descending → eigenvalues `d_i = σ_i²` descending;
    // `vt` is `penalized_rank × penalized_rank` with row i = vᵢᵀ.
    let n_sv = singular_values.len().min(penalized_rank).min(vt.nrows());
    *range_eigenvalues_sorted = (0..penalized_rank)
        .map(|i| {
            if i < n_sv {
                let s = singular_values[i];
                s * s
            } else {
                0.0
            }
        })
        .collect();
    for col_idx in 0..n_sv {
        for row in 0..penalized_rank {
            range_rotation[(row, col_idx)] = vt[[col_idx, row]];
        }
    }
}

/// The facts a refusing stacked-root SVD was decided against (#2465): the shape
/// it was handed, whether that input was even finite, its magnitude, and the λ
/// dynamic range that set it. A refusal naming only "no convergence" cannot
/// distinguish a genuinely unusable pencil from an iteration that gave up on a
/// well-formed one, and those two want opposite responses.
fn describe_stacked_roots(e_stacked: &Array2<f64>, lambdas: &[f64]) -> String {
    let mut max_abs = 0.0_f64;
    let mut nonfinite = 0usize;
    for &value in e_stacked.iter() {
        if value.is_finite() {
            max_abs = max_abs.max(value.abs());
        } else {
            nonfinite += 1;
        }
    }
    let finite_lambdas: Vec<f64> = lambdas.iter().copied().filter(|l| l.is_finite()).collect();
    let lambda_min = finite_lambdas.iter().copied().fold(f64::INFINITY, f64::min);
    let lambda_max = finite_lambdas
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    format!(
        "rows={} cols={} nonfinite_entries={} max_abs={:.6e} n_lambda={} lambda_min={:.6e} \
         lambda_max={:.6e}",
        e_stacked.nrows(),
        e_stacked.ncols(),
        nonfinite,
        max_abs,
        lambdas.len(),
        lambda_min,
        lambda_max,
    )
}

/// Penalized-block spectrum `Σ_k λ_k S_k = U diag(d) Uᵀ`, with the resolution
/// of the route that computed it.
struct PenalizedBlockSpectrum {
    /// Descending eigenvalues `d`.
    eigenvalues: Vec<f64>,
    /// The matching orthonormal rotation `U`.
    rotation: Mat<f64>,
    /// The eigenvalue magnitude at or below which the route has not separated
    /// an eigenvalue from zero, in the units of `d`: `(max(m, n)·ε·σ_max)²` for
    /// the two factor routes (the square of the backward-error band on the
    /// singular values of `E`), `n·ε·d_max` plus the Gram's own formation band
    /// for the Gram route. Both are
    /// proportional to the λ-weighted spectrum itself, so scaling every λ by
    /// `c` scales the resolution by `c` and every floored eigenvalue with it.
    resolution: f64,
}

impl PenalizedBlockSpectrum {
    /// The floor every λ-weighted penalized eigenvalue is held at: the route's
    /// resolution, below which the computed value is roundoff of the largest one.
    ///
    /// A spectrum that vanishes altogether (every λ of the block zero) has no
    /// resolution to floor at: `log|S|₊` is `−∞` there, so it is refused rather
    /// than replaced by an arbitrary ridge.
    fn eigenvalue_floor(&self, lambdas: &[f64]) -> Result<f64, EstimationError> {
        if self.resolution.is_finite() && self.resolution > 0.0 {
            return Ok(self.resolution);
        }
        Err(EstimationError::LayoutError(format!(
            "the lambda-weighted penalty vanishes on its structurally penalized subspace or \
             is non-finite (largest eigenvalue {:.3e}, lambdas {lambdas:?}): log|S|+ is not \
             finite",
            self.eigenvalues.first().copied().unwrap_or(0.0)
        )))
    }
}

/// Penalized-block spectrum `Σ_k λ_k S_k = U diag(d) Uᵀ` of the stacked
/// transformed roots `rs_transformed` (each `rank_k × penalized_rank`), as
/// descending eigenvalues, the matching orthonormal rotation `U`, and the
/// resolution of the route that computed them.
///
/// The route ladder and its accuracy argument are documented inline; both the
/// global engine and the blockwise original-frame engine call this one routine
/// so the two can never compute the spectrum differently.
fn penalized_block_spectrum(
    rs_transformed: &[Mat<f64>],
    lambdas: &[f64],
    penalized_rank: usize,
) -> Result<PenalizedBlockSpectrum, EstimationError> {
    let mut range_eigenvalues_sorted: Vec<f64> = Vec::new();
    let mut range_rotation = Mat::<f64>::zeros(penalized_rank, penalized_rank);
    let total_root_rows: usize = rs_transformed.iter().map(Mat::nrows).sum();
    // Thin SVD yields a COMPLETE orthonormal `V` (all `penalized_rank`
    // directions, including exactly-zero σ) only when `E` is tall or square
    // (`total_root_rows ≥ penalized_rank`).  That always holds structurally —
    // the union of the penalty root ranges spans the penalized subspace — but a
    // pathological degenerate layout is handled by falling back to the Gram
    // eigendecomposition so `range_rotation` is never left rank-deficient.
    //
    // ROUTE LADDER (#2581).  The shape test gates only the ATTEMPT, not the
    // choice.  Both routes below compute the SAME two outputs
    // (`range_eigenvalues_sorted`, `range_rotation`), so a stacked-root SVD
    // that fails to CONVERGE is exactly the pathological case the Gram route
    // was written for: non-convergence is a property of the bidiagonal
    // iteration, not evidence that the pencil is unusable.  Selecting on the
    // shape alone turned that into a fatal `LayoutError` raised one branch
    // away from a trusted routine for the same quantity, aborting a whole
    // converging fit (measured: nottem `cyclic(month, k=12)`, one 75% partition,
    // λ ≈ 5, an 11×11 `E` — the outer BFGS had already certified a nearby ρ
    // before the refinement pass hit it).
    //
    // The ladder has three rungs, in accuracy order: the direct SVD of `E`;
    // the R-SVD of its Householder QR factor, which is EXACTLY as accurate
    // and merely a different computation; and only then the Gram route,
    // whose real cost is the squared condition number.  Reaching either
    // lower rung is REPORTED, so the route that produced the answer is never
    // silently substituted.
    let mut have_rotation = false;
    let mut rescued_by_r_svd = false;
    let mut svd_refusal: Option<String> = None;
    // The Gram route's formation band, set only when that route ran.
    let mut gram_assembly_band: Option<f64> = None;
    if total_root_rows >= penalized_rank {
        let mut e_stacked = Array2::<f64>::zeros((total_root_rows, penalized_rank));
        let mut row_off = 0usize;
        for (lambda, root) in lambdas.iter().zip(rs_transformed.iter()) {
            let sqrt_lambda = lambda.max(0.0).sqrt();
            let rk = root.nrows();
            for r in 0..rk {
                for c in 0..penalized_rank {
                    e_stacked[[row_off + r, c]] = sqrt_lambda * root[(r, c)];
                }
            }
            row_off += rk;
        }
        match e_stacked.svd(false, true) {
            Ok((_, singular_values, Some(vt))) => {
                absorb_right_singular_factorization(
                    &singular_values,
                    &vt,
                    penalized_rank,
                    &mut range_eigenvalues_sorted,
                    &mut range_rotation,
                );
                have_rotation = true;
            }
            direct => {
                let facts = describe_stacked_roots(&e_stacked, lambdas);
                svd_refusal = Some(match direct {
                    Err(err) => format!("failed: {err:?}; {facts}"),
                    _ => format!("returned no right singular vectors; {facts}"),
                });
                // RUNG 2, and the reason the Gram route is a LAST resort
                // rather than the only alternative.  `E = QR` by Householder
                // reflections is direct — it has no convergence criterion to
                // miss — and `EᵀE = RᵀR`, so `R`'s right singular vectors
                // and singular values ARE `E`'s.  It therefore delivers the
                // same two outputs at the same `O(ε²·d_max)` resolution the
                // direct SVD promises, on a `penalized_rank`-square problem
                // instead of a `total_root_rows`-tall one.
                //
                // Measured on the refusing input (nottem `cyclic(month, k=12)`,
                // split 1, an 11×11 `E`, one λ = 6.068, no non-finite
                // entries): a power-of-two rescale of `E` refuses
                // identically — so the failure is not a scaling artefact —
                // while this route and `svd(Eᵀ)` both converge and agree on
                // `σ₀ = 1.653863e0`.
                if let Ok((_, r_factor)) = e_stacked.qr()
                    && let Ok((_, singular_values, Some(vt))) = r_factor.svd(false, true)
                {
                    absorb_right_singular_factorization(
                        &singular_values,
                        &vt,
                        penalized_rank,
                        &mut range_eigenvalues_sorted,
                        &mut range_rotation,
                    );
                    have_rotation = true;
                    rescued_by_r_svd = true;
                }
            }
        }
    }
    if let Some(reason) = svd_refusal.as_deref() {
        if rescued_by_r_svd {
            log::debug!(
                "penalized-block rotation: stacked-root SVD {reason}. Recovered the SAME \
                 right-singular basis from the Householder QR of `E` followed by the SVD \
                 of its triangular factor `R`: `EᵀE = RᵀR`, so no accuracy is given up."
            );
        } else {
            // The accuracy downgrade is observable rather than silent: the
            // Gram route resolves a recessive penalized eigenvalue only down
            // to `O(ε·d_max)`, where the SVD of `E` reaches `O(ε²·d_max)`.
            log::debug!(
                "penalized-block rotation: stacked-root SVD {reason}, and so did the R-SVD \
                 of its Householder QR factor. Recomputing it from the Gram `Σₖ λₖ Sₖ`, \
                 which squares the condition number: recessive eigenvalues are resolved to \
                 O(ε·d_max) rather than O(ε²·d_max)."
            );
        }
    }
    if !have_rotation {
        // Assemble the Gram and eigendecompose it. This route serves a layout
        // whose penalty roots cannot span the penalized subspace
        // (`total_root_rows < penalized_rank`) AND a stacked-root SVD that
        // did not converge.
        let mut range_block = Mat::<f64>::zeros(penalized_rank, penalized_rank);
        let mut weighted_row_norm_sum = 0.0_f64;
        for (lambda, root) in lambdas.iter().zip(rs_transformed.iter()) {
            let root_pen = root.as_ref().submatrix(0, 0, root.nrows(), penalized_rank);
            matmul(
                range_block.as_mut(),
                Accum::Add,
                root_pen.transpose(),
                root_pen,
                *lambda,
                Par::Seq,
            );
            let mut row_norm_sq = 0.0_f64;
            for r in 0..root.nrows() {
                for c in 0..penalized_rank {
                    row_norm_sq += root[(r, c)] * root[(r, c)];
                }
            }
            weighted_row_norm_sum += lambda.abs() * row_norm_sq;
        }
        // `Σ_k λ_k R_kᵀR_k` is the inner products of the `total_root_rows`
        // stacked root rows, each term rounding twice (the product and its λ
        // weight): its formation error is at most
        // `γ_{rows+1}·Σ_k |λ_k|·‖R_k‖_F²` in spectral norm.
        let assembly_band = gam_linalg::roundoff::weighted_gram_assembly_band(
            total_root_rows,
            2,
            weighted_row_norm_sum,
        );
        let (range_eigenvalues, range_eigenvectors) = robust_eigh_faer(
            &range_block,
            Side::Lower,
            assembly_band,
            "range penalty block",
        )?;
        gram_assembly_band = Some(assembly_band);
        let mut range_order: Vec<usize> = (0..penalized_rank).collect();
        range_order.sort_by(|&i, &j| {
            range_eigenvalues[j]
                .partial_cmp(&range_eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        });
        range_eigenvalues_sorted = range_order
            .iter()
            .map(|&idx| range_eigenvalues[idx])
            .collect();
        for (col_idx, &idx) in range_order.iter().enumerate() {
            for row in 0..penalized_rank {
                range_rotation[(row, col_idx)] = range_eigenvectors[(row, idx)];
            }
        }
    }
    let resolution = match gram_assembly_band {
        None => {
            let sigma_max = range_eigenvalues_sorted.first().map_or(0.0, |d| d.sqrt());
            let band = gam_linalg::roundoff::factor_singular_band(
                total_root_rows,
                penalized_rank,
                sigma_max,
            );
            band * band
        }
        Some(assembly_band) => gam_linalg::roundoff::resolved_eigenvalue_band(
            &range_eigenvalues_sorted,
            assembly_band,
        ),
    };
    Ok(PenalizedBlockSpectrum {
        eigenvalues: range_eigenvalues_sorted,
        rotation: range_rotation,
        resolution,
    })
}

pub fn stable_reparameterizationwith_invariant(
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
    invariant: &ReparamInvariant,
) -> Result<ReparamResult, EstimationError> {
    let m = penalties.len();

    if lambdas.len() != m {
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "Lambda count mismatch: expected {} lambdas for {} penalties, got {}",
            m,
            m,
            lambdas.len()
        )));
    }

    // No separate length check needed — penalties are matched against lambdas above,
    // and the invariant's qs_base is p x p (dimension-checked by the split).

    // #1074: the gam#1379 finite-ceiling on λ_k = exp(ρ_k) (clamp to 1e300 to
    // avoid `∞·0 = NaN` when the outer optimizer drives a redundant penalty
    // direction's log-λ past ~709) was DELETED. It masked the real defect: the
    // optimizer drives a redundant/unidentified penalty direction off to ∞
    // instead of that direction being detected and dropped from the model.
    // The root fix (detect+drop the redundant penalty direction at construction)
    // is tracked separately; λ now passes through raw.

    if m == 0 {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs: Array2::eye(p),
            canonical_transformed: vec![],
            e_transformed: Array2::zeros((0, p)),
            // All modes truncated when no penalties; already in transformed frame.
            u_truncated: Array2::eye(p),
        });
    }

    if !invariant.has_nonzero {
        let qs = invariant.split.compose_qs();
        let u_truncated = qs.t().dot(&invariant.split.q_null);
        // All penalties are zero — canonical_transformed = originals (no rotation needed).
        let canonical_transformed: Vec<CanonicalPenalty> = penalties.to_vec();
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(m),
            qs,
            canonical_transformed,
            e_transformed: Array2::zeros((0, p)),
            u_truncated,
        });
    }

    let q_pen = array_to_faer(&invariant.split.q_pen);
    let q_null = array_to_faer(&invariant.split.q_null);
    let qs_base = array_to_faer(&invariant.qs_base);
    // Each penalty root transform is independent: R_k_block @ Q[start..end, :].
    // Run those per-penalty products in parallel, then collect in slice order so
    // all downstream accumulation stays deterministic and bit-for-bit stable with
    // respect to penalty ordering. Every later S_k contraction goes through these
    // roots, so the p×p `S_k = R_kᵀR_k` is never formed.
    let rs_transformed: Vec<Mat<f64>> = penalties
        .par_iter()
        .map(|cp| {
            let r = &cp.col_range;
            let root_faer = array_to_faer(&cp.root);
            let q_block = qs_base.submatrix(r.start, 0, cp.block_dim(), p);
            let mut product = Mat::<f64>::zeros(cp.rank(), p);
            matmul(
                product.as_mut(),
                Accum::Replace,
                root_faer.as_ref(),
                q_block,
                1.0,
                Par::Seq,
            );
            product
        })
        .collect();

    let penalized_rank = invariant.split.rank();

    let spectrum = if penalized_rank > 0 {
        // Penalized-block spectrum `Σ_k λ_k S_k = U diag(d) Uᵀ` (restricted to the
        // λ-invariant penalized subspace).  Compute it from the SVD of the STACKED
        // SCALED ROOTS `E = [√λ_k R_k]_k` (root rows stacked), NOT from an
        // eigendecomposition of the assembled Gram `range_block = EᵀE`.
        //
        // Why (#2123): `S_k = R_kᵀ R_k`, so `Σ_k λ_k S_k = EᵀE` with
        // `E = vstack_k(√λ_k R_k)`.  Assembling the Gram and eigendecomposing it
        // SQUARES the condition number (`κ(EᵀE) = κ(E)²`).  When the outer optimizer
        // drives one margin toward its null space the λ dynamic range is enormous
        // (here `te(x,z)` with the near-linear z axis reaches `λ_ratio ≳ 1e8`), so a
        // recessive-penalty eigenvalue `d_min ≈ λ_min·σ²` is swamped by the
        // eigensolver's `O(ε·d_max) = O(ε·λ_max)` absolute floor — its eigenVECTOR
        // rotates into numerical noise, the genuinely-penalized direction is lost
        // from `e_transformed`, and the inner P-IRLS solve then fits that direction
        // to the data (a WIGGLY β̂ with `βᵀSβ̂ ≈ 0` despite `λ → ∞`).  That silent
        // loss-of-penalty is a discontinuous function of ρ (it flips as the noise
        // floor crosses `d_min`), so it injects spurious cliffs into the REML/LAML
        // objective and a false low-cost basin in the high-λ corner — which the
        // outer optimizer then lands in for some training-row orders but not others
        // (the row-order-dependent EDF/SE of #2123).
        //
        // The SVD operates on `E` directly, so `σ_min(E) = √d_min` is resolved
        // whenever `√λ_min·σ ≳ ε·√λ_max·σ` — a λ dynamic range up to ~1e32 instead
        // of ~1e16 — keeping the reparameterized penalty faithful across the whole ρ
        // box and the outer objective smooth and permutation-invariant.  `EᵀE`
        // equals the previous `range_block` bit-for-bit, so well-conditioned fits
        // are unchanged; only the ill-conditioned corner is corrected.
        //
        // As before, the right singular vectors `V` (= the penalized-block rotation)
        // are used ONLY to build `E`/`S⁺`/traces below; they are NOT applied to
        // `q_pen` or `rs_transformed`, so `Q_s` stays λ-independent and the
        // quasi-Newton coordinate system does not drift at eigenvalue crossings.
        penalized_block_spectrum(&rs_transformed, lambdas, penalized_rank)?
    } else {
        PenalizedBlockSpectrum {
            eigenvalues: Vec::new(),
            rotation: Mat::<f64>::zeros(0, 0),
            resolution: 0.0,
        }
    };

    // Subspace-invariant penalty spectral calculus:
    // - Penalized and null spaces are fixed by the lambda-invariant basis `qs_base`.
    // - Runtime lambda dependence only appears in the penalized block eigenvalues.
    // This avoids basis mixing inside the degenerate zero-eigenspace.
    let structural_rank = penalized_rank;
    // The floor is the spectrum's own resolution, so it carries the units of
    // `Σ λ_k S_k`. An absolute floor (it was `1e-12·max(balanced, 1)`) is not
    // equivariant under the λ rescaling that a change of response units
    // induces: for an inverse-Gaussian fit in small units the optimal λ·s sits
    // near 1e-13, every penalized eigenvalue was raised to the same absolute
    // ridge, and the fit was crushed to edf ≈ 1.4 where the mode has edf ≈ 13.
    let eigenvalue_floor = if penalized_rank > 0 {
        spectrum.eigenvalue_floor(lambdas)?
    } else {
        0.0
    };
    let PenalizedBlockSpectrum {
        eigenvalues: range_eigs_sorted,
        rotation: range_rotation,
        ..
    } = spectrum;
    let qs = compose_qs_from_split(&q_pen, &q_null, p);

    // Guard against any accidental penalized/null mixing. The transformed penalty
    // roots must have negligible support on null columns by construction.
    let leakage = assess_subspace_leakage(&qs, &rs_transformed, structural_rank, p);
    if !subspace_split_is_consistent(&leakage, invariant.null_leakage_tolerance) {
        return Err(EstimationError::LayoutError(format!(
            "Reparameterization subspace split is inconsistent: max null leakage {:.3e} (rel {:.3e}, worst penalty {}), max |Qp'Qn| {:.3e}",
            leakage.max_abs_sq.sqrt(),
            leakage.max_rel_sq.sqrt(),
            leakage.worst_penalty,
            leakage.max_cross_gram_abs,
        )));
    }

    // Truncated basis in transformed coordinates:
    //   U_⊥^(t) = Qs^T U_⊥^(orig) = Qs^T Q_n.
    let mut u_truncated_mat = Mat::<f64>::zeros(p, q_null.ncols());
    matmul(
        u_truncated_mat.as_mut(),
        Accum::Replace,
        qs.transpose(),
        q_null.as_ref(),
        1.0,
        Par::Seq,
    );

    // E is represented in TRANSFORMED coordinates (beta_t).  Because the
    // penalized subspace is NOT rotated by the lambda-dependent eigenvectors
    // (to keep Q_s stable across BFGS iterations), E is no longer diagonal.
    // Instead E = diag(√d) · U' embedded in structural_rank × p, so that
    // E'E = U diag(d) U' = Σ λ_k S_k in the invariant penalized basis.
    let mut e_transformed_mat = Mat::<f64>::zeros(structural_rank, p);
    for row_idx in 0..structural_rank {
        let safe_eigenval = range_eigs_sorted[row_idx].max(eigenvalue_floor);
        let sqrt_eigenval = safe_eigenval.sqrt();
        // E[row, j] = sqrt(d_row) * U'[row, j] = sqrt(d_row) * U[j, row]
        for j in 0..penalized_rank {
            e_transformed_mat[(row_idx, j)] = sqrt_eigenval * range_rotation[(j, row_idx)];
        }
    }

    // Pseudo-logdet on the structural penalized block.  The null block is split
    // out above, so there is no nullspace normalization here.  Eigenvalues the
    // spectrum route does not resolve from zero are floored to its resolution
    // `eigenvalue_floor` to keep the log-det finite and consistent with the
    // floored values used to construct `e_transformed_mat` above.  This avoids spurious P-IRLS failures when the
    // lambda dynamic range is wide (e.g. during BFGS line search probing extreme
    // rho candidates).  Materially negative or non-finite spectra are already
    // rejected by the strict classifier upstream; this loop re-checks the
    // *post-shrinkage* range eigenvalues against the same floor.
    //
    // The same floored spectrum is used in the trace formula tr(S⁺ S_k) below,
    // matching the rank structure embedded in `e_transformed_mat` and avoiding
    // a 1/0 in the trace contraction when an eigenvalue was floored to 0.
    let mut floored_eigs: Vec<f64> = Vec::with_capacity(range_eigs_sorted.len());
    let mut log_det_sum = CompensatedSum::default();
    for (idx, &ev) in range_eigs_sorted.iter().enumerate() {
        if !ev.is_finite() || ev < -eigenvalue_floor {
            return Err(EstimationError::LayoutError(format!(
                "Penalty pseudo-logdet has a non-finite or large-negative structural eigenvalue at index {idx}: {ev:.3e}"
            )));
        }
        let safe_ev = ev.max(eigenvalue_floor);
        floored_eigs.push(safe_ev);
        if idx < penalized_rank {
            log_det_sum.add(safe_ev.ln());
        }
    }
    let log_det = log_det_sum.value();
    let delta = 0.0;

    // The det1 contractions are independent once the eigensystem is fixed.  Use
    // indexed parallel collection so the output vector preserves lambda order.
    let det1vec: Vec<f64> = (0..lambdas.len())
        .into_par_iter()
        .map(|k| {
            // Compute tr((S+δI)⁻¹ S_k) in the range eigenbasis without ever
            // materializing (S+δI)⁻¹ or S_k.
            let trace = trace_root_penalty_in_orthogonal_basis(
                &rs_transformed[k],
                penalized_rank,
                &range_rotation,
                &floored_eigs,
                delta,
            );
            lambdas[k] * trace
        })
        .collect();

    // Rebuild s_transformed from e_transformed to ensure rank consistency.
    //
    // The sum of λ*S_k may contain numerical noise modes (eigenvalues ~1e-15) that
    // become significant when λ is large (e.g., 10^12). These modes would appear in H
    // but are truncated from log|S|_+, creating a "phantom penalty" in the objective.
    //
    // By reconstructing s_transformed = E^T * E, we force the penalty matrix used
    // in H to have the EXACT same rank structure as the one used for log|S|_+.
    // Any mode truncated from the prior is now strictly zero in the Hessian
    // calculation, ensuring mathematical consistency of the gradients.
    let mut s_truncated = Mat::<f64>::zeros(p, p);
    matmul(
        s_truncated.as_mut(),
        Accum::Replace,
        e_transformed_mat.transpose(),
        e_transformed_mat.as_ref(),
        1.0,
        Par::Seq,
    );

    {
        // Structural check: transformed S must not leak into declared null coordinates.
        let mut max_null_diag = 0.0_f64;
        let mut max_null_offdiag = 0.0_f64;
        for i in structural_rank..p {
            max_null_diag = max_null_diag.max(s_truncated[(i, i)].abs());
            for j in 0..p {
                if i != j {
                    max_null_offdiag = max_null_offdiag.max(s_truncated[(i, j)].abs());
                }
            }
        }
        assert!(
            max_null_diag <= 1e-10 && max_null_offdiag <= 1e-10,
            "null-space leakage in transformed penalty: max_null_diag={max_null_diag:.3e}, max_null_offdiag={max_null_offdiag:.3e}"
        );
    }

    let qs_array = mat_to_array(&qs);
    let canonical_transformed: Vec<CanonicalPenalty> = rs_transformed
        .par_iter()
        .map(|r| CanonicalPenalty::from_dense_root(mat_to_array(r), p))
        .collect();
    Ok(ReparamResult {
        s_transformed: mat_to_array(&s_truncated),
        log_det,
        det1: Array1::from(det1vec),
        qs: qs_array,
        canonical_transformed,
        e_transformed: mat_to_array(&e_transformed_mat),
        u_truncated: mat_to_array(&u_truncated_mat),
    })
}

/// Minimal engine layout descriptor that avoids domain-specific layout coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineDims {
    pub p: usize,
    pub k: usize,
}

impl EngineDims {
    pub fn new(p: usize, k: usize) -> Self {
        Self { p, k }
    }
}

/// Engine-facing stable reparameterization API using only `(p, k)`.
///
/// When `cached_invariant` is `Some`, reuses the precomputed eigendecomposition
/// (the hot path inside the REML loop). When `None`, computes the invariant on
/// the fly (the post-REML refit path).
/// Stable reparameterization from block-local canonical penalties.
pub fn stable_reparameterization_engine_canonical(
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
    dims: EngineDims,
    cached_invariant: Option<&ReparamInvariant>,
) -> Result<ReparamResult, EstimationError> {
    let owned;
    let invariant = match cached_invariant {
        Some(inv) => inv,
        None => {
            owned = precompute_reparam_invariant_from_canonical(penalties, dims.p)?;
            &owned
        }
    };
    stable_reparameterizationwith_invariant(penalties, lambdas, dims.p, invariant)
}

/// Stable reparameterization expressed in the ORIGINAL coefficient frame
/// (`qs = I`), the frame the sparse-native inner solve works in.
///
/// The result carries `S = Qs·S̃·Qsᵀ`, its root `E = Ẽ·Qsᵀ`, the unpenalized
/// basis `Qs·Ũ⊥ = Q_null`, and the `log|S|₊` / `det1` of
/// [`stable_reparameterizationwith_invariant`], where `S̃`, `Ẽ`, `Ũ⊥` are that
/// engine's transformed-frame outputs.
///
/// When the penalties partition into disjoint column blocks, `Q_pen` is block
/// diagonal and so is the λ-dependent spectrum `Σ_k λ_k S_k` restricted to it.
/// The spectrum is then computed block by block with the same stacked-root
/// route, so the eigenvalues, the log-determinant, the traces `tr(S⁺S_k)` and
/// the Gram `EᵀE` are those of the global engine; `E` differs from `Ẽ·Qsᵀ`
/// only by an orthogonal mixing of its rows, which no consumer of `E` can
/// observe. Each block floors its eigenvalues at the resolution of its own
/// factorization, which is finer than the global one: the two engines differ
/// only on an eigenvalue the global factorization does not resolve from zero. The cost is `O(Σ_b q_b³)` over
/// the block dimensions `q_b` — `O(q_b)` for a diagonal block such as a
/// random effect — instead of the `O(p³)` of a global factorization followed
/// by two `p × p` frame changes.
pub fn stable_reparameterization_original_frame(
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
    dims: EngineDims,
    cached_invariant: Option<&ReparamInvariant>,
) -> Result<ReparamResult, EstimationError> {
    let owned;
    let invariant = match cached_invariant {
        Some(inv) => inv,
        None => {
            owned = precompute_reparam_invariant_from_canonical(penalties, dims.p)?;
            &owned
        }
    };
    let p = dims.p;
    let blocks = match invariant.blocks.as_ref() {
        Some(blocks) if invariant.has_nonzero && lambdas.len() == penalties.len() => blocks,
        _ => {
            let base = stable_reparameterizationwith_invariant(penalties, lambdas, p, invariant)?;
            return Ok(reparam_in_original_frame(base, penalties, p));
        }
    };

    let spectra: Vec<BlockOriginalFrame> = blocks
        .par_iter()
        .map(|block| block_original_frame(block, penalties, lambdas))
        .collect::<Result<_, _>>()?;

    let mut leakage = SubspaceLeakageMetrics {
        max_abs_sq: 0.0,
        max_rel_sq: 0.0,
        worst_penalty: 0,
        max_cross_gram_abs: 0.0,
    };
    let mut det1 = Array1::<f64>::zeros(penalties.len());
    let mut log_det = CompensatedSum::default();
    let penalized_rank = invariant.split.rank();
    let mut s_original = Array2::<f64>::zeros((p, p));
    let mut e_original = Array2::<f64>::zeros((penalized_rank, p));
    let mut row_offset = 0usize;
    for (block, spectrum) in blocks.iter().zip(spectra) {
        for (k, abs_sq, rel_sq) in spectrum.null_leakage {
            if rel_sq > leakage.max_rel_sq {
                leakage.max_rel_sq = rel_sq;
                leakage.worst_penalty = k;
            }
            leakage.max_abs_sq = leakage.max_abs_sq.max(abs_sq);
        }
        leakage.max_cross_gram_abs = leakage.max_cross_gram_abs.max(spectrum.cross_gram_abs);
        for (k, value) in spectrum.det1 {
            det1[k] = value;
        }
        log_det.add(spectrum.log_det);
        let start = block.col_range.start;
        let end = block.col_range.end;
        match spectrum.root {
            BlockOriginalRoot::Diagonal { eigenvalues } => {
                let cols = block
                    .diagonal_pen_cols
                    .as_ref()
                    .expect("a diagonal block root comes from a diagonal block");
                for (j, (&col, &eigenvalue)) in cols.iter().zip(eigenvalues.iter()).enumerate() {
                    e_original[[row_offset + j, start + col]] = eigenvalue.sqrt();
                    s_original[[start + col, start + col]] = eigenvalue;
                }
                row_offset += cols.len();
            }
            BlockOriginalRoot::Dense(e_local) => {
                let rows = e_local.nrows();
                s_original
                    .slice_mut(s![start..end, start..end])
                    .assign(&e_local.t().dot(&e_local));
                e_original
                    .slice_mut(s![row_offset..row_offset + rows, start..end])
                    .assign(&e_local);
                row_offset += rows;
            }
        }
    }
    if !subspace_split_is_consistent(&leakage, invariant.null_leakage_tolerance) {
        return Err(EstimationError::LayoutError(format!(
            "Reparameterization subspace split is inconsistent: max null leakage {:.3e} (rel {:.3e}, worst penalty {}), max |Qp'Qn| {:.3e}",
            leakage.max_abs_sq.sqrt(),
            leakage.max_rel_sq.sqrt(),
            leakage.worst_penalty,
            leakage.max_cross_gram_abs,
        )));
    }

    Ok(ReparamResult {
        s_transformed: s_original,
        log_det: log_det.value(),
        det1,
        qs: Array2::eye(p),
        canonical_transformed: penalties.to_vec(),
        e_transformed: e_original,
        u_truncated: invariant.split.q_null.clone(),
    })
}

/// Map a transformed-frame reparameterization back to original coordinates:
/// `S = Qs·S̃·Qsᵀ`, `E = Ẽ·Qsᵀ`, `U⊥ = Qs·Ũ⊥`, with `qs = I`.
fn reparam_in_original_frame(
    base: ReparamResult,
    penalties: &[CanonicalPenalty],
    p: usize,
) -> ReparamResult {
    use gam_linalg::faer_ndarray::fast_ab;
    let qs = &base.qs;
    let s_original = fast_ab(&fast_ab(qs, &base.s_transformed), &qs.t().to_owned());
    let e_original = fast_ab(&base.e_transformed, &qs.t().to_owned());
    let u_original = fast_ab(qs, &base.u_truncated);
    ReparamResult {
        s_transformed: s_original,
        log_det: base.log_det,
        det1: base.det1,
        qs: Array2::eye(p),
        canonical_transformed: penalties.to_vec(),
        e_transformed: e_original,
        u_truncated: u_original,
    }
}

/// The original-frame root of one penalty block's floored spectrum.
enum BlockOriginalRoot {
    /// Floored eigenvalues on the block's diagonal penalized columns.
    Diagonal { eigenvalues: Vec<f64> },
    /// `pen_rank × block_dim` root `diag(√d̃)·Uᵀ·Q_penᵀ`.
    Dense(Array2<f64>),
}

struct BlockOriginalFrame {
    root: BlockOriginalRoot,
    log_det: f64,
    /// `(penalty index, λ_k tr(S⁺S_k))` for the block's member penalties.
    det1: Vec<(usize, f64)>,
    /// `(penalty index, null energy, relative null energy)` of each member root.
    null_leakage: Vec<(usize, f64, f64)>,
    /// `max |Q_penᵀ Q_null|` within the block.
    cross_gram_abs: f64,
}

/// Floor a λ-weighted penalized eigenvalue at its spectrum's resolution,
/// refusing a non-finite or materially negative one.
fn floor_penalized_eigenvalue(
    value: f64,
    index: usize,
    eigenvalue_floor: f64,
) -> Result<f64, EstimationError> {
    if !value.is_finite() || value < -eigenvalue_floor {
        return Err(EstimationError::LayoutError(format!(
            "Penalty pseudo-logdet has a non-finite or large-negative structural eigenvalue at index {index}: {value:.3e}"
        )));
    }
    Ok(value.max(eigenvalue_floor))
}

fn block_original_frame(
    block: &InvariantBlock,
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
) -> Result<BlockOriginalFrame, EstimationError> {
    let members = &block.penalty_indices;
    if let Some(cols) = block.diagonal_pen_cols.as_ref() {
        let mut null_cols = vec![true; block.col_range.len()];
        for &col in cols {
            null_cols[col] = false;
        }
        let mut eigenvalues = vec![0.0_f64; cols.len()];
        for &k in members {
            let local = penalties[k].local_ref();
            let weight = lambdas[k].max(0.0);
            for (eigenvalue, &col) in eigenvalues.iter_mut().zip(cols) {
                *eigenvalue += weight * local[[col, col]];
            }
        }
        // Each eigenvalue is a sum of non-negative products, exact to its own
        // relative rounding: there is no unresolved one to floor, and one that
        // is not strictly positive has a `−∞` log.
        let mut log_det = CompensatedSum::default();
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            if !(eigenvalue.is_finite() && eigenvalue > 0.0) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty pseudo-logdet has a non-finite or non-positive diagonal \
                     structural eigenvalue at index {index}: {eigenvalue:.3e}"
                )));
            }
            log_det.add(eigenvalue.ln());
        }
        let mut det1 = Vec::with_capacity(members.len());
        let mut null_leakage = Vec::with_capacity(members.len());
        for &k in members {
            let local = penalties[k].local_ref();
            let mut trace = CompensatedSum::default();
            for (&eigenvalue, &col) in eigenvalues.iter().zip(cols) {
                trace.add(local[[col, col]] / eigenvalue);
            }
            det1.push((k, lambdas[k] * trace.value()));
            let mut total_sq = 0.0_f64;
            let mut null_sq = 0.0_f64;
            for (col, &is_null) in null_cols.iter().enumerate() {
                total_sq += local[[col, col]];
                if is_null {
                    null_sq += local[[col, col]];
                }
            }
            let rel_sq = if total_sq > 0.0 {
                null_sq / total_sq
            } else {
                0.0
            };
            null_leakage.push((k, null_sq, rel_sq));
        }
        return Ok(BlockOriginalFrame {
            root: BlockOriginalRoot::Diagonal { eigenvalues },
            log_det: log_det.value(),
            det1,
            null_leakage,
            cross_gram_abs: 0.0,
        });
    }

    let block_dim = block.col_range.len();
    let pen_rank = block.q_pen_local.ncols();
    let q_pen = array_to_faer(&block.q_pen_local);
    let q_null = array_to_faer(&block.q_null_local);
    let mut rs_local = Vec::with_capacity(members.len());
    let mut null_leakage = Vec::with_capacity(members.len());
    for &k in members {
        let root = array_to_faer(&penalties[k].root);
        let mut product = Mat::<f64>::zeros(root.nrows(), pen_rank);
        matmul(
            product.as_mut(),
            Accum::Replace,
            root.as_ref(),
            q_pen.as_ref(),
            1.0,
            Par::Seq,
        );
        let mut null_part = Mat::<f64>::zeros(root.nrows(), q_null.ncols());
        matmul(
            null_part.as_mut(),
            Accum::Replace,
            root.as_ref(),
            q_null.as_ref(),
            1.0,
            Par::Seq,
        );
        let null_sq = null_part.squared_norm_l2();
        let total_sq = null_sq + product.squared_norm_l2();
        let rel_sq = if total_sq > 0.0 {
            null_sq / total_sq
        } else {
            0.0
        };
        null_leakage.push((k, null_sq, rel_sq));
        rs_local.push(product);
    }
    let mut cross_gram = Mat::<f64>::zeros(pen_rank, q_null.ncols());
    matmul(
        cross_gram.as_mut(),
        Accum::Replace,
        q_pen.transpose(),
        q_null.as_ref(),
        1.0,
        Par::Seq,
    );
    let cross_gram_abs = mat_max_abs_element(cross_gram.as_ref());

    let member_lambdas: Vec<f64> = members.iter().map(|&k| lambdas[k]).collect();
    let (eigenvalues, rotation, eigenvalue_floor) = if pen_rank > 0 {
        let spectrum = penalized_block_spectrum(&rs_local, &member_lambdas, pen_rank)?;
        let floor = spectrum.eigenvalue_floor(&member_lambdas)?;
        (spectrum.eigenvalues, spectrum.rotation, floor)
    } else {
        (Vec::new(), Mat::<f64>::zeros(0, 0), 0.0)
    };
    let mut floored = Vec::with_capacity(pen_rank);
    let mut log_det = CompensatedSum::default();
    for (index, &value) in eigenvalues.iter().enumerate() {
        let value = floor_penalized_eigenvalue(value, index, eigenvalue_floor)?;
        log_det.add(value.ln());
        floored.push(value);
    }
    let det1 = members
        .iter()
        .zip(rs_local.iter())
        .map(|(&k, root)| {
            let trace =
                trace_root_penalty_in_orthogonal_basis(root, pen_rank, &rotation, &floored, 0.0);
            (k, lambdas[k] * trace)
        })
        .collect();
    // E_b = diag(√d̃) · Uᵀ · Q_penᵀ  (pen_rank × block_dim).
    let mut e_local = Mat::<f64>::zeros(pen_rank, block_dim);
    matmul(
        e_local.as_mut(),
        Accum::Replace,
        rotation.transpose(),
        q_pen.transpose(),
        1.0,
        Par::Seq,
    );
    for (row, &value) in floored.iter().enumerate() {
        let scale = value.sqrt();
        for col in 0..block_dim {
            e_local[(row, col)] *= scale;
        }
    }
    Ok(BlockOriginalFrame {
        root: BlockOriginalRoot::Dense(mat_to_array(&e_local)),
        log_det: log_det.value(),
        det1,
        null_leakage,
        cross_gram_abs,
    })
}

#[cfg(test)]
mod tests {
    /// #2469: the freeze and the trial read one band. A kept eigenvalue inside
    /// the Gram's rounding band is an unresolved rank, refused whatever its sign.
    /// The sign test it replaces accepted positive roundoff and priced
    /// `ln(roundoff)` into `log|S|₊`. A tail above the band is still priced.
    #[test]
    fn frozen_rank_trial_refuses_a_kept_eigenvalue_inside_the_rounding_band_2469() {
        use ndarray::array;
        let angle = 0.3_f64;
        let rotation = array![
            [angle.cos(), -angle.sin(), 0.0],
            [angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0]
        ];
        let spec_with_tail = |tail: f64| {
            let diagonal = array![[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, tail]];
            crate::PenaltySpec::Block {
                local: rotation.dot(&diagonal).dot(&rotation.t()),
                col_range: 0..3,
                structure_hint: None,
                op: None,
            }
        };
        for tail in [3.0e-17, -3.0e-17] {
            let outcome = super::canonicalize_penalty_specs_at_frozen_ranks(
                &[spec_with_tail(tail)],
                &[0],
                &[3],
                3,
                "band test",
            );
            assert!(
                matches!(
                    &outcome,
                    Err(super::EstimationError::TrialPointRefused { reason })
                        if reason.contains("unresolved within the Gram's rounding band")
                ),
                "tail {tail:e} inside the rounding band must be an unresolved-rank refusal, got {:?}",
                outcome.as_ref().map(|(active, _)| active.len())
            );
        }
        let (active, _) = super::canonicalize_penalty_specs_at_frozen_ranks(
            &[spec_with_tail(1.0e-6)],
            &[0],
            &[3],
            3,
            "band test",
        )
        .expect("a resolved tail is priced");
        assert_eq!(active[0].root.nrows(), 3);
    }

    /// gam#2454: the reparameterization's penalized rank and the shared
    /// balanced structural rank are one number. Three unit directions penalized
    /// at Frobenius norms `1e6`, `1e-9` and `1` (overlapping full-width
    /// components, the dense fallback path): the unweighted sum would drop the
    /// `1e-9` direction at a `100·p·ε·max` cut; the balanced rule keeps all three
    /// and the split's `q_pen` spans all three.
    #[test]
    fn reparam_split_rank_is_the_balanced_structural_rank_2454() {
        use ndarray::array;
        let p = 3usize;
        let roots = [
            array![[1.0e3, 0.0, 0.0]],
            array![[0.0, 1.0e-9_f64.sqrt(), 0.0]],
            array![[0.0, 0.0, 1.0]],
        ];
        let penalties: Vec<super::CanonicalPenalty> = roots
            .iter()
            .map(|root| super::CanonicalPenalty::from_dense_root(root.clone(), p))
            .collect();
        let balanced = super::balanced_penalty_structural_rank(
            penalties
                .iter()
                .map(|cp| (cp.local_ref().view(), cp.col_range.clone())),
            p,
        )
        .expect("balanced rank");
        assert_eq!(balanced, 3);
        let invariant = super::precompute_reparam_invariant_from_canonical(&penalties, p)
            .expect("reparam invariant");
        assert_eq!(
            invariant.split.q_pen.ncols(),
            balanced,
            "the split's penalized subspace has the balanced structural rank"
        );
        assert_eq!(invariant.split.q_null.ncols(), 0);
    }

    /// pyGAM audit speed F1: the PIRLS cache budgets its entries by
    /// `resident_bytes`, so it must count the rotated penalties. Each of `K`
    /// block-local penalties becomes dense in the transformed frame and carries
    /// its own `p × p` Gram, so the reparameterization holds at least `K p²`
    /// doubles beyond `S̃`, `Qs` and `E`, not the few `p²` those three alone hold.
    #[test]
    fn reparam_resident_bytes_counts_every_rotated_penalty_gram() {
        let blocks = 6usize;
        let width = 4usize;
        let p = blocks * width;
        let penalties: Vec<super::CanonicalPenalty> = (0..blocks)
            .map(|block| {
                let mut root = Array2::<f64>::zeros((width - 1, p));
                for row in 0..width - 1 {
                    root[[row, block * width + row]] = 1.0;
                    root[[row, block * width + row + 1]] = -1.0;
                }
                super::CanonicalPenalty::from_dense_root(root, p)
            })
            .collect();
        let lambdas = vec![1.0; blocks];
        let invariant =
            precompute_reparam_invariant_from_canonical(&penalties, p).expect("reparam invariant");
        let reparam = stable_reparameterizationwith_invariant(&penalties, &lambdas, p, &invariant)
            .expect("reparameterization");
        assert_eq!(reparam.canonical_transformed.len(), blocks);
        let f64_bytes = std::mem::size_of::<f64>();
        let rotated_grams = blocks * p * p * f64_bytes;
        let frame_matrices =
            (reparam.s_transformed.len() + reparam.qs.len() + reparam.e_transformed.len())
                * f64_bytes;
        assert!(
            reparam.resident_bytes() >= frame_matrices + rotated_grams,
            "resident {} must cover the frame matrices {frame_matrices} and the {blocks} rotated \
             p x p Grams {rotated_grams}",
            reparam.resident_bytes()
        );
    }

    use super::{
        CanonicalPenalty, EngineDims, SubspaceLeakageMetrics,
        assess_subspace_leakage, classify_eigenvalues_strict,
        precompute_reparam_invariant_from_canonical, report_penalty_pair_redundancy,
        stable_reparameterization_original_frame, stable_reparameterizationwith_invariant,
        subspace_split_is_consistent,
    };
    use crate::EstimationError;
    use faer::Mat;
    use gam_linalg::utils::inf_norm;
    use ndarray::{Array2, array};

    /// Build CanonicalPenalty values from full-width roots for tests.
    fn canonical_from_roots(rs_list: &[Array2<f64>], p: usize) -> Vec<CanonicalPenalty> {
        rs_list
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                CanonicalPenalty {
                    root: r.clone().into_shared(),
                    col_range: 0..p,
                    total_dim: p,
                    nullity: 0,
                    local: local.into_shared(),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect()
    }

    fn metrics_for(
        qs: &Mat<f64>,
        rs: &[Mat<f64>],
        structural_rank: usize,
        p: usize,
    ) -> SubspaceLeakageMetrics {
        assess_subspace_leakage(qs, rs, structural_rank, p)
    }

    #[test]
    fn subspace_leakage_iszero_for_clean_split() {
        let p = 4usize;
        let structural_rank = 2usize;
        let qs = Mat::<f64>::identity(p, p);
        let mut r0 = Mat::<f64>::zeros(2, p);
        r0[(0, 0)] = 1.0;
        r0[(1, 1)] = 2.0;

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_abs_sq <= 1e-16);
        assert!(m.max_rel_sq <= 1e-16);
        assert!(m.max_cross_gram_abs <= 1e-16);
    }

    #[test]
    fn subspace_leakage_detects_null_column_energy() {
        let p = 4usize;
        let structural_rank = 2usize;
        let qs = Mat::<f64>::identity(p, p);
        let mut r0 = Mat::<f64>::zeros(1, p);
        r0[(0, 2)] = 3.0;

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_abs_sq > 0.0);
        assert!(m.max_rel_sq > 0.99);
    }

    #[test]
    fn subspace_leakage_detects_qp_qn_nonorthogonality() {
        let p = 3usize;
        let structural_rank = 1usize;
        let mut qs = Mat::<f64>::identity(p, p);
        qs[(0, 1)] = 0.2;
        let r0 = Mat::<f64>::zeros(1, p);

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_cross_gram_abs > 1e-3);
    }

    /// `root = diag(√σ)·Hᵀ` for a dense Householder reflection `H`, so the
    /// penalty `S = RᵀR` has spectrum `σ` (plus one exact null direction) in a
    /// rotated, non-coordinate basis.
    fn rotated_spectrum_root(sigma: &[f64]) -> Array2<f64> {
        let p = sigma.len() + 1;
        let v: Vec<f64> = (0..p).map(|i| 1.0 + i as f64).collect();
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        let householder =
            Array2::from_shape_fn((p, p), |(i, j)| {
                let identity = if i == j { 1.0 } else { 0.0 };
                identity - 2.0 * v[i] * v[j] / v_norm_sq
            });
        Array2::from_shape_fn((sigma.len(), p), |(row, col)| {
            sigma[row].sqrt() * householder[[col, row]]
        })
    }

    #[test]
    fn a_spectrum_decaying_through_the_old_rank_floor_is_penalized_and_splits_cleanly_1802() {
        // #1802: on sphere / Duchon / spline-on-sphere bases the REML outer
        // startup rejected every seed with "Reparameterization subspace split
        // is inconsistent: max null leakage 1.174e-4 (rel 1.031e-4)". The
        // manifold spectrum decays through the old `1e-8`-relative rank cut
        // with no gap, so resolved eigenvalues just below the cut were
        // declared null and carried `~1e-8` relative root energy onto the null
        // columns. The cut is now the resolved-eigenvalue predicate: every
        // eigenvalue here is far above the roundoff band `~p·ε`, so all of them
        // are penalized, the split leaves only roundoff on the null column, and
        // the derived leakage bound is roundoff-level instead of `p·1e-8`.
        let sigma = [1.0, 1.0e-2, 1.0e-4, 1.0e-6, 5.0e-9, 1.0e-10, 1.0e-11];
        let p = sigma.len() + 1;
        let penalty = CanonicalPenalty::from_dense_root(rotated_spectrum_root(&sigma), p);
        let structural = super::balanced_penalty_structural_rank(
            [(penalty.local_ref().view(), penalty.col_range.clone())],
            p,
        )
        .expect("structural rank");
        assert_eq!(structural, sigma.len(), "every resolved eigenvalue is penalized");
        let penalties = vec![penalty];
        let invariant =
            precompute_reparam_invariant_from_canonical(&penalties, p).expect("invariant");
        assert_eq!(invariant.split.q_pen.ncols(), structural);
        assert_eq!(invariant.split.q_null.ncols(), 1);
        // `(1+γ)(√(2·band) + γ)²` with `band ≈ (p + γ-depth)·ε`: of order
        // 1e-14, where the old admission floor was `p·1e-8 = 8e-8`.
        assert!(
            invariant.null_leakage_tolerance > 0.0 && invariant.null_leakage_tolerance < 1.0e-12,
            "derived leakage bound must be roundoff-level, got {:.3e}",
            invariant.null_leakage_tolerance
        );
        let reparam = stable_reparameterizationwith_invariant(&penalties, &[1.0], p, &invariant)
            .expect("the split of a resolved decaying spectrum is consistent");
        assert!(reparam.log_det.is_finite());
    }

    #[test]
    fn subspace_split_rejects_leakage_above_the_derived_bound_1802() {
        // The derived bound admits only roundoff of the split. The #1802
        // signature — `~1.06e-8` relative root energy on a null column — is no
        // longer roundoff: it can only arise when the roots and the operator
        // the split was read from disagree, and it is rejected, as is a whole
        // penalized mode dumped on the null block.
        let p = 40usize;
        let structural_rank = p - 1;
        let qs = Mat::<f64>::identity(p, p);
        let tolerance = super::balanced_null_leakage_tolerance(
            p,
            1,
            gam_linalg::roundoff::resolved_eigenvalue_band(&vec![1.0; p], 0.0),
        );
        for null_energy in [1.06e-8_f64, 1.0] {
            let mut rs = Mat::<f64>::zeros(1, p);
            rs[(0, 0)] = (1.0 - null_energy).sqrt();
            rs[(0, p - 1)] = null_energy.sqrt();
            let leakage = metrics_for(&qs, &[rs], structural_rank, p);
            assert!((leakage.max_rel_sq - null_energy).abs() <= 1e-12 * null_energy);
            assert!(
                !subspace_split_is_consistent(&leakage, tolerance),
                "relative null energy {null_energy:e} above the bound {tolerance:.3e} is an \
                 inconsistent split"
            );
        }
        // Exactly clean roots on an orthonormal split are admitted.
        let mut clean_root = Mat::<f64>::zeros(1, p);
        clean_root[(0, 0)] = 1.0;
        let clean = metrics_for(&qs, &[clean_root], structural_rank, p);
        assert!(subspace_split_is_consistent(&clean, tolerance));

        // A non-orthogonal split is rejected regardless of root leakage.
        let mut qs_bad = Mat::<f64>::identity(3, 3);
        qs_bad[(0, 1)] = 0.2;
        let zero_root = Mat::<f64>::zeros(1, 3);
        let leakage2 = metrics_for(&qs_bad, &[zero_root], 1, 3);
        assert!(leakage2.max_cross_gram_abs > 1e-3);
        assert!(
            !subspace_split_is_consistent(&leakage2, tolerance),
            "a non-orthogonal Qp/Qn split must be rejected"
        );
    }

    /// #4057: the rank the criterion's `log|S|₊` ranges over
    /// (`balanced_penalty_structural_rank`) and the rank of the
    /// reparameterization's penalized subspace are one number, for diagonal
    /// and dense blocks alike, including relative eigenvalues between `1e-12`
    /// and `1e-8` that the old fixed `1e-8` cut dropped from the split only.
    #[test]
    fn structural_rank_and_reparam_split_rank_agree_below_the_old_floor_4057() {
        let sigma = [1.0, 3.0e-9, 2.0e-10, 4.0e-12];
        let dense_dim = sigma.len() + 1;
        let diagonal_dim = sigma.len();
        let p = dense_dim + diagonal_dim;
        let dense = block_penalty(rotated_spectrum_root(&sigma), 0, p);
        let mut diagonal_root = Array2::<f64>::zeros((diagonal_dim, diagonal_dim));
        for (i, &value) in sigma.iter().enumerate() {
            diagonal_root[[i, i]] = value.sqrt();
        }
        let diagonal = block_penalty(diagonal_root, dense_dim, p);
        let penalties = vec![dense, diagonal];
        let structural = super::balanced_penalty_structural_rank(
            penalties
                .iter()
                .map(|cp| (cp.local_ref().view(), cp.col_range.clone())),
            p,
        )
        .expect("structural rank");
        assert_eq!(structural, 2 * sigma.len());
        let invariant =
            precompute_reparam_invariant_from_canonical(&penalties, p).expect("invariant");
        assert_eq!(invariant.split.q_pen.ncols(), structural);
        assert_eq!(invariant.split.q_null.ncols(), p - structural);
        let blocks = invariant.blocks.as_ref().expect("non-overlapping blocks");
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].diagonal_pen_cols.is_none());
        assert_eq!(blocks[1].diagonal_pen_cols.as_ref().map(Vec::len), Some(sigma.len()));
        stable_reparameterizationwith_invariant(&penalties, &[1.0, 1.0], p, &invariant)
            .expect("dense-frame reparameterization");
        stable_reparameterization_original_frame(
            &penalties,
            &[1.0, 1.0],
            EngineDims::new(p, penalties.len()),
            Some(&invariant),
        )
        .expect("original-frame reparameterization");
    }

    #[test]
    fn u_truncated_is_transformed_frame_in_nonzero_case() {
        let p = 3usize;
        let rs_list = vec![array![[1.0, 0.0, 0.0]]];
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![2.0];
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv)
            .expect("stable reparam");

        let expected = rep.qs.t().dot(&inv.split.q_null);
        let diff = &rep.u_truncated - &expected;
        let max_abs = inf_norm(diff.iter().copied());
        assert!(
            max_abs <= 1e-10,
            "u_truncated frame mismatch: max_abs={max_abs}"
        );
    }

    #[test]
    fn infinite_lambda_keeps_range_penalty_block_finite_1379() {
        // gam#1379 / gam#1074: a genuinely infinite λ = exp(ρ) is NOT silently
        // clamped to a finite ceiling. The original #1379 fix added a 1e300
        // ceiling so `∞ · 0` could not poison the range block Σ_k λ_k S_k, but
        // #1074 DELETED that clamp on purpose (see the comment at the top of
        // `stable_reparameterizationwith_invariant`): masking ∞ hid the real
        // defect — the outer optimizer driving a redundant/unidentified penalty
        // direction off to ∞ instead of that direction being detected and
        // dropped. With the clamp gone, a literal `f64::INFINITY` λ surfaces as
        // a clean, detectable error (the eigensolver rejects the NaN-poisoned
        // block) rather than a silent finite success. Pin that contract: ∞ must
        // ERROR, not be quietly clamped.
        //
        // Fixture: two penalties on a 3-wide block. The first penalizes only
        // coordinate 0 (so its block S_k has structural zeros everywhere except
        // [0,0]); give it λ = +∞. The second penalizes coordinate 1 at a normal
        // λ.
        let p = 3usize;
        let rs_list = vec![array![[1.0, 0.0, 0.0]], array![[0.0, 1.0, 0.0]]];
        let canonical = canonical_from_roots(&rs_list, p);
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");

        let lambdas_inf = vec![f64::INFINITY, 3.0];
        let inf_result = stable_reparameterizationwith_invariant(&canonical, &lambdas_inf, p, &inv);
        assert!(
            inf_result.is_err(),
            "an infinite lambda must surface as an error, not be silently clamped (#1074)"
        );

        // A finite (even very large) λ must still produce an all-finite reparam:
        // the function is robust to large-but-finite penalties; only the
        // non-finite input is rejected.
        let lambdas_big = vec![1e300_f64, 3.0];
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas_big, p, &inv)
            .expect("stable reparam at large-but-finite lambda");
        assert!(
            rep.s_transformed.iter().all(|v| v.is_finite()),
            "transformed penalty must be finite at large-but-finite lambda"
        );
        assert!(
            rep.qs.iter().all(|v| v.is_finite()),
            "reparam rotation must be finite at large-but-finite lambda"
        );
        assert!(
            rep.log_det.is_finite(),
            "penalty log-det must be finite at large-but-finite lambda"
        );
        assert!(
            rep.det1.iter().all(|v| v.is_finite()),
            "penalty log-det derivatives must be finite at large-but-finite lambda"
        );
    }

    /// A penalty whose `rank × (end - start)` root acts on columns `start..end`.
    fn block_penalty(root: Array2<f64>, start: usize, p: usize) -> CanonicalPenalty {
        let block_dim = root.ncols();
        let local = root.t().dot(&root);
        CanonicalPenalty {
            root: root.into_shared(),
            col_range: start..start + block_dim,
            total_dim: p,
            nullity: 0,
            local: local.into_shared(),
            positive_eigenvalues: Vec::new(),
            op: None,
        }
    }

    fn difference_root(block_dim: usize, stencil: &[f64]) -> Array2<f64> {
        let rows = block_dim + 1 - stencil.len();
        let mut root = Array2::<f64>::zeros((rows, block_dim));
        for row in 0..rows {
            for (offset, &weight) in stencil.iter().enumerate() {
                root[[row, row + offset]] = weight;
            }
        }
        root
    }

    #[test]
    fn reparameterization_is_equivariant_under_a_common_lambda_scale() {
        // A change of response units rescales every optimal λ by one factor
        // (inverse-Gaussian: λ ∝ c⁴ under y → c·y), and the penalized spectrum
        // `Σ λ_k S_k` with it. Both engines must then return `c·S`, `c·EᵀE`,
        // `log|S|₊ + r·ln c` and unchanged `det1`, whatever `c`. An absolute
        // eigenvalue floor (1e-12) broke this once `λ·s` fell below it: every
        // penalized eigenvalue became the same ridge.
        let p = 11usize;
        let penalties = vec![
            // Two overlapping difference penalties on one dense block…
            block_penalty(difference_root(6, &[1.0, -2.0, 1.0]), 0, p),
            block_penalty(difference_root(6, &[-1.0, 1.0]), 0, p),
            // …a disjoint diagonal (random-effect) block…
            block_penalty(Array2::eye(4), 6, p),
            // …and one unpenalized column.
        ];
        let invariant =
            precompute_reparam_invariant_from_canonical(&penalties, p).expect("invariant");
        let lambdas = [2.0, 0.5, 3.0];
        let rank = 5 + 4;
        for scale in [1e-15, 1e-9, 1e9] {
            let scaled: Vec<f64> = lambdas.iter().map(|l| scale * l).collect();
            for original_frame in [false, true] {
                let run = |lambdas: &[f64]| {
                    if original_frame {
                        stable_reparameterization_original_frame(
                            &penalties,
                            lambdas,
                            EngineDims::new(p, penalties.len()),
                            Some(&invariant),
                        )
                    } else {
                        stable_reparameterizationwith_invariant(&penalties, lambdas, p, &invariant)
                    }
                    .expect("reparameterization")
                };
                let base = run(&lambdas);
                let rep = run(&scaled);
                assert_eq!(rep.e_transformed.nrows(), rank);
                let expected_log_det = base.log_det + rank as f64 * scale.ln();
                assert!(
                    (rep.log_det - expected_log_det).abs() <= 1e-9 * expected_log_det.abs(),
                    "scale {scale:e} (original frame {original_frame}): log|S|+ {} vs {}",
                    rep.log_det,
                    expected_log_det
                );
                for (a, b) in rep.det1.iter().zip(base.det1.iter()) {
                    assert!(
                        (a - b).abs() <= 1e-9 * b.abs(),
                        "scale {scale:e}: det1 {a} vs {b}"
                    );
                }
                let s_norm = inf_norm(base.s_transformed.iter().copied());
                let gram_base = base.e_transformed.t().dot(&base.e_transformed);
                let gram = rep.e_transformed.t().dot(&rep.e_transformed);
                for ((s, s0), (g, g0)) in rep
                    .s_transformed
                    .iter()
                    .zip(base.s_transformed.iter())
                    .zip(gram.iter().zip(gram_base.iter()))
                {
                    assert!(
                        (s / scale - s0).abs() <= 1e-9 * s_norm,
                        "scale {scale:e}: S entry {s} vs {}",
                        scale * s0
                    );
                    assert!(
                        (g / scale - g0).abs() <= 1e-9 * s_norm,
                        "scale {scale:e}: EᵀE entry {g} vs {}",
                        scale * g0
                    );
                }
            }
        }
    }

    #[test]
    fn u_truncated_is_identitywhen_no_penalties() {
        let p = 4usize;
        let canonical: Vec<CanonicalPenalty> = Vec::new();
        let lambdas: Vec<f64> = Vec::new();
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv)
            .expect("stable reparam");
        assert_eq!(rep.u_truncated, Array2::<f64>::eye(p));
    }

    #[test]
    fn transformed_penalty_is_diagonal_in_transformed_frame() {
        let p = 3usize;
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        // Penalize a rotated direction in original space so Qs is non-trivial.
        let rs_list = vec![array![[inv_sqrt2, inv_sqrt2, 0.0]]];
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![4.0];
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv)
            .expect("stable reparam");

        assert_eq!(rep.e_transformed.nrows(), 1);
        assert!(rep.e_transformed[[0, 0]].abs() > 0.0);
        assert!(rep.e_transformed[[0, 1]].abs() <= 1e-12);
        assert!(rep.e_transformed[[0, 2]].abs() <= 1e-12);
        // Exact pseudo-logdet on the structural penalized block has no
        // delta-dependent nullspace normalization.
        let expected_det1 = 1.0_f64;
        assert!((rep.det1[0] - expected_det1).abs() <= 1e-12);

        let s = rep.s_transformed;
        let mut max_offdiag = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    max_offdiag = max_offdiag.max(s[[i, j]].abs());
                }
            }
        }
        assert!(
            max_offdiag <= 1e-10,
            "transformed penalty should be diagonal, max offdiag={max_offdiag}"
        );
        assert!(s[[1, 1]].abs() <= 1e-10);
        assert!(s[[2, 2]].abs() <= 1e-10);
    }

    /// det1 is contracted through the transformed roots (`u_lᵀS_k u_l =
    /// ‖R_k u_l‖²`) rather than a formed `S_k`. It must still be the exact
    /// `∂ log|S|₊ / ∂ρ_k` for overlapping rank-deficient penalties with a shared
    /// null direction, and sum to the penalized rank.
    #[test]
    fn root_contracted_det1_is_the_log_det_gradient_for_overlapping_penalties() {
        let p = 6usize;
        let mut state = 0x5EED_DE71_u64;
        let mut unit = || {
            let bits = gam_linalg::utils::splitmix64(&mut state) >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        // Three rank-2 roots that never touch the last coefficient, so the
        // penalized rank is p - 1 and the null direction is shared.
        let rs_list: Vec<Array2<f64>> = (0..3)
            .map(|_| {
                let mut root = Array2::<f64>::zeros((2, p));
                for i in 0..2 {
                    for j in 0..p - 1 {
                        root[[i, j]] = unit();
                    }
                }
                root
            })
            .collect();
        let canonical = canonical_from_roots(&rs_list, p);
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rho = [0.7_f64, -1.3, 2.1];
        let reparam_at = |rho: &[f64]| {
            let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
            stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv)
                .expect("stable reparam")
        };
        let rep = reparam_at(&rho);
        let det1_sum: f64 = rep.det1.iter().sum();
        assert!(
            (det1_sum - (p - 1) as f64).abs() <= 1e-10,
            "Σ_k λ_k tr(S⁺S_k) = rank(S) = {}, got {det1_sum}",
            p - 1
        );
        let step = 1e-5_f64;
        for k in 0..rho.len() {
            let mut plus = rho;
            let mut minus = rho;
            plus[k] += step;
            minus[k] -= step;
            let central = (reparam_at(&plus).log_det - reparam_at(&minus).log_det) / (2.0 * step);
            assert!(
                (rep.det1[k] - central).abs() <= 1e-7,
                "det1[{k}] = {} but central difference of log|S|₊ is {central}",
                rep.det1[k]
            );
        }
    }

    #[test]
    fn det1_matches_rank_for_single_full_rank_penalty() {
        let p = 2usize;
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        // Q^T for a 45-degree rotation.
        let q_t = [[inv_sqrt2, inv_sqrt2], [-inv_sqrt2, inv_sqrt2]];
        // R = diag(3, 1) * Q^T gives S = Q * diag(9, 1) * Q^T.
        let rs = array![
            [3.0 * q_t[0][0], 3.0 * q_t[0][1]],
            [1.0 * q_t[1][0], 1.0 * q_t[1][1]]
        ];
        let rs_list = vec![rs];
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![5.0];

        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv)
            .expect("stable reparam");

        assert_eq!(rep.e_transformed.nrows(), p);
        let det1 = rep.det1[0];
        // Exact pseudo-logdet on the structural penalized block:
        //   det1 = lambda * sum_l d_l / (lambda*d_l)
        // where d_l are eigenvalues of S_k.
        let s_k_eigs = [9.0_f64, 1.0_f64];
        let lambda = 5.0_f64;
        let expected_det1: f64 = s_k_eigs.iter().map(|&d| lambda * d / (lambda * d)).sum();
        assert!(
            (det1 - expected_det1).abs() <= 1e-12,
            "expected det1={expected_det1}, got {det1}",
        );

        let s = rep.s_transformed;
        assert!(s[[0, 1]].abs() <= 1e-10);
        assert!(s[[1, 0]].abs() <= 1e-10);
        assert!(s[[0, 0]] > 0.0);
        assert!(s[[1, 1]] > 0.0);
    }

    #[test]
    fn classify_strict_rejects_nan_eigenvalue() {
        let mut eigs = [1.0, f64::NAN, 0.5];
        match classify_eigenvalues_strict(&mut eigs, 0.0, "test_nan") {
            Err(EstimationError::PenaltySpectrumNonFinite {
                context,
                index,
                value,
            }) => {
                assert_eq!(context, "test_nan");
                assert_eq!(index, 1);
                assert!(value.is_nan());
            }
            other => panic!("expected PenaltySpectrumNonFinite, got {:?}", other),
        }
    }

    #[test]
    fn classify_strict_rejects_inf_eigenvalue() {
        let mut eigs = [1.0, 0.5, f64::INFINITY];
        match classify_eigenvalues_strict(&mut eigs, 0.0, "test_inf") {
            Err(EstimationError::PenaltySpectrumNonFinite { index, value, .. }) => {
                assert_eq!(index, 2);
                assert!(value.is_infinite());
            }
            other => panic!("expected PenaltySpectrumNonFinite, got {:?}", other),
        }
    }

    #[test]
    fn classify_strict_rejects_materially_indefinite() {
        // -1e-2 with scale ~1.0 is well above any reasonable roundoff tolerance.
        let mut eigs = [1.0, -1e-2, 0.5];
        match classify_eigenvalues_strict(&mut eigs, 0.0, "test_indef") {
            Err(EstimationError::PenaltySpectrumIndefinite {
                context,
                index,
                value,
                ..
            }) => {
                assert_eq!(context, "test_indef");
                assert_eq!(index, 1);
                assert!((value + 1e-2).abs() <= 1e-15);
            }
            other => panic!("expected PenaltySpectrumIndefinite, got {:?}", other),
        }
    }

    #[test]
    fn classify_strict_accepts_roundoff_negative() {
        // Exact input (zero assembly band): -1e-16·scale is inside the
        // eigensolver's own band 4·ε·scale ≈ 8.9e-16·scale.
        let scale = 1.0_f64;
        let roundoff = -1e-16 * scale;
        let mut eigs = [scale, 0.5 * scale, roundoff, 0.25 * scale];
        classify_eigenvalues_strict(&mut eigs, 0.0, "test_roundoff")
            .expect("roundoff must classify");
        // The roundoff eigenvalue is snapped to exact zero.
        assert_eq!(eigs[2], 0.0);
        // Strictly positive entries must be preserved.
        assert!(eigs[0] > 0.0 && eigs[1] > 0.0 && eigs[3] > 0.0);
    }

    /// #1619: range blocks at extreme λ used to fail as "indefinite" on
    /// roundoff, and a fixed `1e-8`-relative floor was added to admit it. The
    /// admissible roundoff is instead the block's own formation band. Here the
    /// block is what `penalized_block_spectrum` forms, `Σ_i λ_i d_iᵀd_i` for a
    /// second-difference root with λ spanning `1e3..1e12`, so its exact null
    /// space is `{1, x}` and every other eigenvalue is at least
    /// `λ_min·σ_min(D)² ≈ 24`. At the derived band
    /// `weighted_gram_assembly_band(rows, 2, Σ λ_i‖d_i‖²) ≈ 8e-3` (plus the
    /// eigensolver's `p·ε·‖G‖`) the two null eigenvalues — computed at
    /// far inside it, of either sign — snap to zero, nothing else does, and a
    /// genuinely negative direction ten bands deep is refused.
    #[test]
    fn classify_strict_admits_extreme_lambda_gram_roundoff_at_its_formation_band_1619() {
        let block_dim = 12usize;
        let root = difference_root(block_dim, &[1.0, -2.0, 1.0]);
        let rows = root.nrows();
        let lambdas: Vec<f64> = (0..rows)
            .map(|i| 10f64.powf(3.0 + 9.0 * i as f64 / (rows - 1) as f64))
            .collect();
        let mut gram = Mat::<f64>::zeros(block_dim, block_dim);
        let mut weighted_row_norm_sum = 0.0_f64;
        for (i, &lambda) in lambdas.iter().enumerate() {
            for a in 0..block_dim {
                weighted_row_norm_sum += lambda * root[[i, a]] * root[[i, a]];
                for b in 0..block_dim {
                    gram[(a, b)] += lambda * (root[[i, a]] * root[[i, b]]);
                }
            }
        }
        let assembly_band =
            gam_linalg::roundoff::weighted_gram_assembly_band(rows, 2, weighted_row_norm_sum);
        let (eigenvalues, _) =
            super::robust_eigh_faer(&gram, faer::Side::Lower, assembly_band, "range penalty block")
                .expect("formation roundoff of a PSD Gram is admitted at its own band");
        assert_eq!(
            eigenvalues.iter().filter(|&&value| value == 0.0).count(),
            2,
            "exactly the exact null space {{1, x}} snaps to zero: {eigenvalues:?}"
        );
        assert!(eigenvalues.iter().all(|&value| value >= 0.0));

        // Negative control: a genuinely negative direction ten bands deep.
        let band = gam_linalg::roundoff::resolved_eigenvalue_band(&eigenvalues, assembly_band);
        let depth = 10.0 * band;
        let unit = 1.0 / (block_dim as f64).sqrt();
        let mut indefinite = gram.clone();
        for a in 0..block_dim {
            for b in 0..block_dim {
                indefinite[(a, b)] -= depth * unit * unit;
            }
        }
        match super::robust_eigh_faer(
            &indefinite,
            faer::Side::Lower,
            assembly_band,
            "range penalty block",
        ) {
            Err(EstimationError::PenaltySpectrumIndefinite { value, .. }) => {
                assert!(value < -band, "refused eigenvalue {value:e} lies below -{band:e}");
            }
            other => panic!("expected PenaltySpectrumIndefinite, got {:?}", other.map(|r| r.0)),
        }
    }

    #[test]
    fn classify_strict_tolerance_is_the_resolved_eigenvalue_band_2469() {
        // The reported tolerance is exactly
        // `resolved_eigenvalue_band(eigs, assembly_band)` — the predicate the
        // structural rank counts with — not a local floor or slack multiplier.
        let scale = 3.0_f64;
        let assembly_band = 1.0e-9 * scale;
        let offending = -2.0 * assembly_band;
        let mut eigs = [scale, 0.5 * scale, offending];
        let expected = gam_linalg::roundoff::resolved_eigenvalue_band(&eigs, assembly_band);
        match classify_eigenvalues_strict(&mut eigs, assembly_band, "test_band") {
            Err(EstimationError::PenaltySpectrumIndefinite { tolerance, .. }) => {
                assert_eq!(tolerance, expected);
            }
            other => panic!("expected PenaltySpectrumIndefinite, got {:?}", other),
        }
        // Negative control: just inside the same tolerance the eigenvalue is
        // roundoff and snaps to zero.
        let mut eigs = [scale, 0.5 * scale, -0.5 * assembly_band];
        classify_eigenvalues_strict(&mut eigs, assembly_band, "test_band")
            .expect("inside the band snaps");
        assert_eq!(eigs[2], 0.0);
    }

    #[test]
    fn classify_strict_snaps_subtol_positive_to_zero() {
        // Positive eigenvalues below the tolerance are also snapped to exact 0
        // so downstream rank counts and pseudo-logdets are deterministic. With
        // exact input the band is the eigensolver's `2·ε·scale ≈ 4.4e-16·scale`.
        let scale = 10.0_f64;
        let subtol = 1e-16 * scale;
        let mut eigs = [scale, subtol];
        classify_eigenvalues_strict(&mut eigs, 0.0, "test_sub_pos").expect("sub-tol positive ok");
        assert_eq!(eigs[1], 0.0);
    }

    /// Build a `CanonicalPenalty` directly from a symmetric `local` matrix.
    /// Bypasses root extraction — the redundancy diagnostic only reads `local`
    /// and `col_range`, so the rest is filler.
    fn canonical_from_local(
        local: Array2<f64>,
        col_range: std::ops::Range<usize>,
        total_dim: usize,
    ) -> CanonicalPenalty {
        let block_dim = local.nrows();
        // A trivially valid root: zero rank. The diagnostic doesn't read root.
        let root = Array2::<f64>::zeros((0, block_dim));
        CanonicalPenalty {
            root: root.into_shared(),
            col_range,
            total_dim,
            nullity: 0,
            local: local.into_shared(),
            positive_eigenvalues: Vec::new(),
            op: None,
        }
    }

    #[test]
    fn report_penalty_pair_redundancy_detects_identical_pair() {
        // Penalty 0: a "generic" SPD matrix on cols 0..3.
        let s0 = ndarray::array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.5],];
        // Penalties 1 and 2: identical block-local penalty on the SAME col_range.
        // This is the Z₂-symmetric saddle scenario.
        let s_shared = ndarray::array![[1.0, -0.5, 0.0], [-0.5, 2.0, -0.5], [0.0, -0.5, 1.0],];

        let bundle = vec![
            canonical_from_local(s0, 0..3, 3),
            canonical_from_local(s_shared.clone(), 0..3, 3),
            canonical_from_local(s_shared, 0..3, 3),
        ];

        let redundant = report_penalty_pair_redundancy(&bundle);

        // Exactly one redundant pair: (1, 2). Pairs (0, 1) and (0, 2) involve
        // distinct matrices and must NOT be flagged.
        assert_eq!(
            redundant.len(),
            1,
            "expected exactly one redundant pair, got {:?}",
            redundant
        );
        let (i, j, defect) = redundant[0];
        assert_eq!((i, j), (1, 2));
        assert_eq!(
            defect, 0.0,
            "bit-identical penalties leave an EXACTLY zero residual, not a small one"
        );
    }

    /// The regression this whole screen was rewritten for (#2676): a pair whose
    /// relative defect is `1.9e-5` — the measured `geo_disease_matern` figure at
    /// its cold geometry — must NOT be reported as proportional, even though its
    /// cosine rounds to `1.000000` at six decimals.
    ///
    /// The two assertions are a pair. The first is the fix; the second is the
    /// evidence that the OLD screen would have failed it, so the test cannot go
    /// green for the boring reason that the fixture is not marginal.
    #[test]
    fn a_pair_1p9e_minus_5_apart_is_not_proportional_2676() {
        const DEFECT: f64 = 1.874_020e-5;
        let base = ndarray::array![[1.0, -0.5, 0.0], [-0.5, 2.0, -0.5], [0.0, -0.5, 1.0]];
        let base_norm = base.iter().map(|v| v * v).sum::<f64>().sqrt();
        // A perturbation ORTHOGONAL to `base` in the Frobenius inner product,
        // so the defect is exactly the perturbation's relative size and no part
        // of it is absorbed into the best scale `c`.
        let mut direction = ndarray::array![[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let overlap = direction
            .iter()
            .zip(base.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (base_norm * base_norm);
        direction = &direction - &(overlap * &base);
        let direction_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        let perturbed = &base + &(direction * (DEFECT * base_norm / direction_norm));

        let bundle = vec![
            canonical_from_local(base.clone(), 0..3, 3),
            canonical_from_local(perturbed.clone(), 0..3, 3),
        ];
        assert!(
            report_penalty_pair_redundancy(&bundle).is_empty(),
            "a pair {DEFECT:.3e} apart is measurably distinct and must not be called proportional"
        );

        // Negative control: the cosine the old screen thresholded on cannot see
        // it. `1 - cos = defect^2/2 = 1.8e-10`, six orders under the `1e-8` bar
        // that used to declare the pair "structurally identical".
        let inner = base
            .iter()
            .zip(perturbed.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let cosine = inner / (base_norm * perturbed.iter().map(|v| v * v).sum::<f64>().sqrt());
        assert!(
            cosine > 1.0 - 1e-8,
            "the fixture must be one the OLD cos > 1 - 1e-8 screen accepted; got cos = {cosine}"
        );
        assert_eq!(
            format!("{cosine:.6}"),
            "1.000000",
            "and one whose six-decimal print is indistinguishable from an exact identity"
        );
    }

    #[test]
    fn report_penalty_pair_redundancy_skips_different_col_ranges() {
        // Two identical local matrices but on disjoint col_ranges. The
        // function must NOT flag them — they live in different parameter
        // subspaces by construction.
        let s = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let bundle = vec![
            canonical_from_local(s.clone(), 0..2, 4),
            canonical_from_local(s, 2..4, 4),
        ];
        let redundant = report_penalty_pair_redundancy(&bundle);
        assert!(
            redundant.is_empty(),
            "different col_ranges must not be flagged"
        );
    }
}
