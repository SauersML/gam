//! Numeric kernels for identifiability-theorem diagnostics.
//!
//! The kernels return scalar facts for iVAE auxiliary richness, decoder
//! Jacobian sparsity, and manifold-SAE anchor coverage. Rust, Python, and CLI
//! layers turn those facts into user-facing reports.

use ndarray::{Array2, ArrayView2, Axis};

/// Maximum sweeps for the cyclic-by-largest-pivot Jacobi eigensolver.
///
/// Jacobi converges quadratically once off-diagonals are small, and the
/// matrices here are tiny (< 64×64 identifiability normal-equation blocks),
/// so a converged solve needs only a handful of sweeps. 200 is a generous
/// safety cap that the `JACOBI_OFFDIAG_TOL` break almost always reaches
/// first; it only bounds pathological non-converging inputs.
const JACOBI_MAX_SWEEPS: usize = 200;

/// Off-diagonal magnitude below which the Jacobi sweep is considered
/// converged. `1e-14` is two orders above f64 unit roundoff, tight enough
/// that residual off-diagonal mass cannot perturb the rank/pseudo-inverse
/// decisions these diagnostics make.
const JACOBI_OFFDIAG_TOL: f64 = 1.0e-14;

/// Maximum distinct values per aux column for it to count as "discrete".
///
/// An integer-valued column with at most this many levels is treated as a
/// categorical/discrete covariate (the regime the iVAE auxiliary-richness
/// theorem is stated for); above it the column is treated as continuous.
const AUX_DISCRETE_MAX_LEVELS: usize = 64;

/// Absolute gap below which two aux values count as the same distinct level.
/// Integer-valued aux data dedups exactly; this only guards float dust from
/// the `round()` check above.
const AUX_LEVEL_DEDUP_TOL: f64 = 1.0e-12;

/// Scalar facts about the auxiliary covariate / latent pair feeding an iVAE.
#[derive(Debug, Clone)]
pub struct AuxRichnessMetrics {
    /// `true` iff every entry of the aux matrix is finite.
    pub aux_observed: bool,
    /// Number of non-finite entries in the aux matrix.
    pub n_nonfinite_aux: usize,
    /// Aux dimension (column count).
    pub aux_dim: usize,
    /// Latent dimension (column count of `latents`).
    pub latent_dim: usize,
    /// Row count `N`.
    pub n_rows: usize,
    /// Column indices (sorted, ascending) that are constant across rows.
    pub constant_columns: Vec<usize>,
    /// `true` iff aux is integer-valued and every column has <= 64 unique values.
    pub aux_is_discrete: bool,
    /// Joint distinct-row count of aux (only computed when `aux_is_discrete`).
    pub n_distinct_levels: usize,
    /// Empirical rank of the least-squares Jacobian `B = (Aᵀ A)^{-1} Aᵀ Z`.
    /// `usize::MAX` sentinel if the rank could not be estimated (e.g. too few rows).
    pub jacobian_rank: usize,
    /// True iff we had enough rows + finite data to estimate the Jacobian rank.
    pub jacobian_rank_estimated: bool,
}

/// Compute the iVAE auxiliary-richness numeric facts.
///
/// `aux` is `(N, aux_dim)`; `latents` is `(N, latent_dim)`. The empirical
/// Jacobian is the linear-regression slope ``B`` of ``Z ~ A`` (centred). For
/// a non-linear iVAE encoder this is a first-order surrogate; a deficient
/// rank here forecloses identifiability regardless of nonlinear postproc.
pub fn aux_richness_metrics(aux: ArrayView2<f64>, latents: ArrayView2<f64>) -> AuxRichnessMetrics {
    let (n, aux_dim) = aux.dim();
    let (n_z, latent_dim) = latents.dim();
    assert_eq!(n, n_z, "aux and latents must share row count");

    // 1. Finiteness.
    let mut n_nonfinite_aux: usize = 0;
    for &v in aux.iter() {
        if !v.is_finite() {
            n_nonfinite_aux += 1;
        }
    }
    let aux_observed = n_nonfinite_aux == 0;

    // 2. Constant columns. Skip non-finite columns entirely (they will be
    //    flagged by `aux_observed=false`).
    let mut constant_columns: Vec<usize> = Vec::new();
    if aux_observed && n >= 1 {
        for j in 0..aux_dim {
            let col = aux.column(j);
            // sample std (population formula — exact zero iff constant).
            let mean: f64 = col.sum() / n as f64;
            let mut var = 0.0_f64;
            for &v in col.iter() {
                let d = v - mean;
                var += d * d;
            }
            var /= n as f64;
            if var <= 1.0e-24 {
                constant_columns.push(j);
            }
        }
    }

    // 3. Discreteness + distinct level count.
    let (aux_is_discrete, n_distinct_levels) = if aux_observed && n >= 1 {
        let mut discrete = true;
        for &v in aux.iter() {
            if (v - v.round()).abs() > 0.0 {
                discrete = false;
                break;
            }
        }
        if discrete {
            for j in 0..aux_dim {
                let col = aux.column(j);
                let mut sorted: Vec<f64> = col.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted.dedup_by(|a, b| (*a - *b).abs() < AUX_LEVEL_DEDUP_TOL);
                if sorted.len() > AUX_DISCRETE_MAX_LEVELS {
                    discrete = false;
                    break;
                }
            }
        }
        if discrete {
            // Joint distinct rows.
            let mut keys: Vec<Vec<i64>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(aux_dim);
                for j in 0..aux_dim {
                    row.push(aux[[i, j]].round() as i64);
                }
                keys.push(row);
            }
            keys.sort();
            keys.dedup();
            (true, keys.len())
        } else {
            (false, 0)
        }
    } else {
        (false, 0)
    };

    // 4. Empirical Jacobian rank.
    let need_rows = aux_dim.max(latent_dim) + 1;
    let mut jacobian_rank_estimated = false;
    let mut jacobian_rank: usize = usize::MAX;
    let z_finite = latents.iter().all(|v| v.is_finite());
    if aux_observed && z_finite && n >= need_rows && aux_dim >= 1 && latent_dim >= 1 {
        // Centre A and Z.
        let mut a_c = aux.to_owned();
        let mut z_c = latents.to_owned();
        let a_mean = a_c.mean_axis(Axis(0)).unwrap();
        let z_mean = z_c.mean_axis(Axis(0)).unwrap();
        for mut row in a_c.rows_mut() {
            row -= &a_mean;
        }
        for mut row in z_c.rows_mut() {
            row -= &z_mean;
        }
        // Solve B = (Aᵀ A)^{+} Aᵀ Z via SVD on (Aᵀ A) — small (aux_dim x aux_dim).
        let ata = a_c.t().dot(&a_c);
        let atz = a_c.t().dot(&z_c);
        let b_hat = pinv_solve(ata.view(), atz.view());
        jacobian_rank = matrix_rank(b_hat.view(), 1.0e-8);
        jacobian_rank_estimated = true;
    }

    AuxRichnessMetrics {
        aux_observed,
        n_nonfinite_aux,
        aux_dim,
        latent_dim,
        n_rows: n,
        constant_columns,
        aux_is_discrete,
        n_distinct_levels,
        jacobian_rank,
        jacobian_rank_estimated,
    }
}

/// Moore-Penrose pseudo-inverse times rhs via SVD. Stable for the small
/// `(aux_dim x aux_dim)` normal-equation matrices encountered here. Tolerance
/// is `1e-12 * max_singular_value`.
fn pinv_solve(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    assert_eq!(m, n, "pinv_solve expects a square normal-equation matrix");
    // Symmetric eigen-decomposition via Jacobi (matrices are small, < 64x64
    // in any realistic identifiability check — Jacobi is robust and avoids
    // pulling in a heavier dependency for this code path).
    let (eigvals, eigvecs) = jacobi_symmetric_eigen(a);
    let max_abs = eigvals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = 1.0e-12 * max_abs.max(1.0);
    // Build A^+ = V diag(1/λ_i if |λ_i|>tol else 0) Vᵀ.
    let k = eigvals.len();
    let mut inv_diag = vec![0.0_f64; k];
    for i in 0..k {
        if eigvals[i].abs() > tol {
            inv_diag[i] = 1.0 / eigvals[i];
        }
    }
    // A^+ b  =  V D Vᵀ b  where D = diag(inv_diag).
    let vtb = eigvecs.t().dot(&b);
    let mut dvtb = vtb.clone();
    for i in 0..k {
        let scale = inv_diag[i];
        for j in 0..dvtb.ncols() {
            dvtb[[i, j]] *= scale;
        }
    }
    eigvecs.dot(&dvtb)
}

/// Jacobi rotation eigen-decomposition for small symmetric matrices.
/// Returns `(eigenvalues, eigenvectors)` with `A = V diag(λ) Vᵀ`.
fn jacobi_symmetric_eigen(a: ArrayView2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    let mut m = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    for _ in 0..JACOBI_MAX_SWEEPS {
        // Find largest off-diagonal.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let av = m[[i, j]].abs();
                if av > max_off {
                    max_off = av;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < JACOBI_OFFDIAG_TOL {
            break;
        }
        let app = m[[p, p]];
        let aqq = m[[q, q]];
        let apq = m[[p, q]];
        let theta = 0.5 * (aqq - app) / apq;
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Update M.
        let new_pp = app - t * apq;
        let new_qq = aqq + t * apq;
        m[[p, p]] = new_pp;
        m[[q, q]] = new_qq;
        m[[p, q]] = 0.0;
        m[[q, p]] = 0.0;
        for i in 0..n {
            if i != p && i != q {
                let aip = m[[i, p]];
                let aiq = m[[i, q]];
                m[[i, p]] = c * aip - s * aiq;
                m[[p, i]] = m[[i, p]];
                m[[i, q]] = s * aip + c * aiq;
                m[[q, i]] = m[[i, q]];
            }
        }
        // Update V.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }
    let eigvals: Vec<f64> = (0..n).map(|i| m[[i, i]]).collect();
    (eigvals, v)
}

/// Numeric rank of `m` via its singular values (computed as
/// `sqrt(eig(MᵀM))`). `tol` is absolute; entries with singular value
/// `<= tol` are considered zero.
fn matrix_rank(m: ArrayView2<f64>, tol: f64) -> usize {
    let gram = m.t().dot(&m);
    let (eigvals, _) = jacobi_symmetric_eigen(gram.view());
    let mut rank = 0usize;
    for &lam in eigvals.iter() {
        if lam.max(0.0).sqrt() > tol {
            rank += 1;
        }
    }
    rank
}

/// Scalar facts about decoder Jacobian sparsity.
#[derive(Debug, Clone)]
pub struct JacobianSparsityMetrics {
    /// `(N_samples, P, latent_dim)` shape elements.
    pub n_samples: usize,
    pub p_features: usize,
    pub latent_dim: usize,
    /// Fraction of entries with `|J| < zero_threshold * max|J|`, averaged
    /// across samples.
    pub mean_sparsity: f64,
    /// Maximum absolute entry of the Jacobian stack.
    pub max_abs: f64,
    /// Per-sample numeric column rank (each entry in `[0, latent_dim]`).
    pub ranks: Vec<usize>,
}

/// Compute mean sparsity and per-sample rank of a stack of Jacobians.
///
/// `jacobians` is `(N_samples, P, latent_dim)`, flattened to a `(N*P, latent_dim)`
/// row-major view. `n_samples` is the leading axis size.
pub fn jacobian_sparsity_metrics(
    jacobians_flat: ArrayView2<f64>,
    n_samples: usize,
    zero_threshold: f64,
) -> JacobianSparsityMetrics {
    let (np_rows, latent_dim) = jacobians_flat.dim();
    assert!(np_rows % n_samples == 0, "rows not divisible by n_samples");
    let p_features = np_rows / n_samples;

    // Max abs.
    let mut max_abs = 0.0_f64;
    for &v in jacobians_flat.iter() {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let cutoff = zero_threshold * max_abs;

    let mut total_near_zero: usize = 0;
    let total_entries = np_rows * latent_dim;
    if max_abs > 0.0 {
        for &v in jacobians_flat.iter() {
            if v.abs() < cutoff {
                total_near_zero += 1;
            }
        }
    } else {
        // All zero Jacobian: maximally "sparse" but degenerate; caller flags this.
        total_near_zero = total_entries;
    }
    let mean_sparsity = if total_entries > 0 {
        total_near_zero as f64 / total_entries as f64
    } else {
        0.0
    };

    // Per-sample rank.
    let mut ranks = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let start = s * p_features;
        let end = start + p_features;
        let view = jacobians_flat.slice(ndarray::s![start..end, ..]);
        // Use `cutoff` (absolute) as the rank tolerance: an entry below it is
        // considered zero, which matches the sparsity decision.
        ranks.push(matrix_rank(view, cutoff.max(1.0e-300)));
    }

    JacobianSparsityMetrics {
        n_samples,
        p_features,
        latent_dim,
        mean_sparsity,
        max_abs,
        ranks,
    }
}

/// Scalar facts about the per-atom anchor structure of an assignment matrix.
#[derive(Debug, Clone)]
pub struct AnchorConsistencyMetrics {
    /// `N` (row count of the assignment matrix).
    pub n_rows: usize,
    /// `K` (column count = atom count).
    pub n_atoms: usize,
    /// Total number of anchor rows
    /// (rows with `max|A|/sum|A| >= anchor_dominance`).
    pub n_anchors: usize,
    /// Per-atom anchor count `(length K)`: for each anchor row, the
    /// dominant atom is tallied.
    pub anchors_per_atom: Vec<usize>,
}

/// Compute anchor counts from an assignment matrix.
///
/// `assignments` is `(N, K)`. A row is an anchor when its maximum-magnitude
/// entry contributes at least `anchor_dominance ∈ (0, 1]` of the row's L1
/// mass. Zero-mass rows are *not* anchors.
fn anchor_consistency_metrics(
    assignments: ArrayView2<f64>,
    anchor_dominance: f64,
) -> AnchorConsistencyMetrics {
    let (n, k) = assignments.dim();
    let mut anchors_per_atom = vec![0_usize; k];
    let mut n_anchors = 0_usize;
    for i in 0..n {
        let row = assignments.row(i);
        let mut mass = 0.0_f64;
        let mut max_val = 0.0_f64;
        let mut max_j = 0_usize;
        for j in 0..k {
            let a = row[j].abs();
            mass += a;
            if a > max_val {
                max_val = a;
                max_j = j;
            }
        }
        if mass > 0.0 && max_val / mass >= anchor_dominance {
            n_anchors += 1;
            anchors_per_atom[max_j] += 1;
        }
    }
    AnchorConsistencyMetrics {
        n_rows: n,
        n_atoms: k,
        n_anchors,
        anchors_per_atom,
    }
}

/// Typed pass/fail verdict for the anchor-consistency identifiability check.
///
/// A manifold-SAE with `K` atoms is identified up to permutation of atoms only
/// when the assignment matrix contains enough *anchor* rows (rows where one
/// atom carries at least `anchor_dominance` of the row's L1 mass). The
/// thresholds here are derived from that separability argument, not tuned:
///
/// * `enough_anchors_total`: permutation identifiability needs at least one
///   anchor per atom, hence `n_anchors >= K` is the weakest necessary count.
/// * `anchors_cover_all_atoms`: an atom with zero anchors has no row that
///   pins it individually, so it is only identified up to a linear mix with
///   its neighbours.
///
/// `K == 1` passes vacuously: a single atom has no permutation ambiguity.
///
/// The verdict lives in the core so the CLI, Rust library, and Python wrapper
/// all report identical diagnostics; presentation layers only format it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorConsistencyPreconditions {
    /// At least one anchor row exists per atom in aggregate.
    pub enough_anchors_total: bool,
    /// Every atom is the dominant atom of at least one anchor row.
    pub anchors_cover_all_atoms: bool,
}

#[derive(Debug, Clone)]
pub struct AnchorConsistencyReport {
    /// The underlying anchor counts.
    pub metrics: AnchorConsistencyMetrics,
    /// The dominance threshold the counts were computed with.
    pub anchor_dominance: f64,
    /// Fraction of rows that are anchors (`n_anchors / max(n_rows, 1)`).
    pub anchor_fraction: f64,
    /// Preconditions derived from the number of fitted atoms.
    pub preconditions: AnchorConsistencyPreconditions,
    /// One human-readable statement per failed precondition.
    pub violations: Vec<String>,
    /// One concrete remediation per violation (`len == violations.len()`).
    pub recommendations: Vec<String>,
    /// Atoms with zero anchor rows (empty when coverage holds).
    pub uncovered_atoms: Vec<usize>,
}

impl AnchorConsistencyReport {
    /// `true` iff every precondition holds.
    pub fn passes(&self) -> bool {
        self.preconditions.enough_anchors_total
            && self.preconditions.anchors_cover_all_atoms
    }
}

/// Default anchor-dominance threshold.
///
/// The separability argument only *requires* the dominant atom to outweigh
/// all others combined (`> 1/2`); this convention adds a robustness margin so
/// that near-boundary rows produced by soft assignments do not count as
/// anchors. It is the project-wide atom-anchor convention shared by the
/// kernel tests and the Python diagnostic default.
pub const ANCHOR_DOMINANCE_DEFAULT: f64 = 0.95;

/// Run the full anchor-consistency identifiability check and return the typed
/// verdict. `assignments` is `(N, K)`; `anchor_dominance` defaults to
/// [`ANCHOR_DOMINANCE_DEFAULT`] when `None`.
pub fn anchor_consistency_report(
    assignments: ArrayView2<f64>,
    anchor_dominance: Option<f64>,
) -> Result<AnchorConsistencyReport, String> {
    let anchor_dominance = anchor_dominance.unwrap_or(ANCHOR_DOMINANCE_DEFAULT);
    if !(anchor_dominance > 0.5 && anchor_dominance <= 1.0) {
        return Err(format!(
            "anchor_dominance must be in (0.5, 1]; got {anchor_dominance}"
        ));
    }
    let (_, k) = assignments.dim();
    if k < 1 {
        return Err("assignments must have at least one atom column".to_string());
    }
    let metrics = anchor_consistency_metrics(assignments, anchor_dominance);
    let anchor_fraction = metrics.n_anchors as f64 / metrics.n_rows.max(1) as f64;

    let mut violations = Vec::new();
    let mut recommendations = Vec::new();
    let mut uncovered_atoms = Vec::new();

    let preconditions = if k == 1 {
        AnchorConsistencyPreconditions {
            enough_anchors_total: true,
            anchors_cover_all_atoms: true,
        }
    } else {
        let enough_anchors = metrics.n_anchors >= k;
        if !enough_anchors {
            violations.push(format!(
                "Only {} anchor row(s) (dominance >= {:.2}) found in a K={}-atom \
                 model; need at least {}. The recovered atoms are identified only \
                 up to a linear transformation in atom space.",
                metrics.n_anchors, anchor_dominance, k, k
            ));
            recommendations.push(format!(
                "Reduce K to <= {}, sharpen the assignment prior (e.g. lower \
                 temperature / stronger IBP concentration), or collect more \
                 anchor-like rows where a single atom dominates.",
                metrics.n_anchors.max(1)
            ));
        }
        uncovered_atoms = metrics
            .anchors_per_atom
            .iter()
            .enumerate()
            .filter_map(|(j, &count)| (count == 0).then_some(j))
            .collect();
        let cover_ok = uncovered_atoms.is_empty();
        if !cover_ok {
            violations.push(format!(
                "Atom(s) {:?} have zero anchor rows; they are not individually \
                 identifiable and may be redundant or merged with neighbours.",
                uncovered_atoms
            ));
            recommendations.push(format!(
                "Prune the {} uncovered atom(s) (refit with K={}) or strengthen \
                 the per-atom sparsity prior so that each atom acquires a \
                 dominant region.",
                uncovered_atoms.len(),
                (k - uncovered_atoms.len()).max(1)
            ));
        }
        AnchorConsistencyPreconditions {
            enough_anchors_total: enough_anchors,
            anchors_cover_all_atoms: cover_ok,
        }
    };

    Ok(AnchorConsistencyReport {
        metrics,
        anchor_dominance,
        anchor_fraction,
        preconditions,
        violations,
        recommendations,
        uncovered_atoms,
    })
}

/// Stack a list of per-atom decoder blocks (each shape `(basis_size_k, P)`)
/// column-wise into a single Jacobian of shape `(P, sum_k basis_size_k)`.
/// Used by the Python diagnostics dispatcher to feed
/// [`jacobian_sparsity_metrics`] from a `ManifoldSAE.decoder_blocks` payload
/// without doing the concatenation in Python.
pub fn concat_decoder_blocks(blocks: &[ArrayView2<f64>]) -> Result<Array2<f64>, String> {
    if blocks.is_empty() {
        return Err("concat_decoder_blocks: empty block list".into());
    }
    let p = blocks[0].ncols();
    for (i, b) in blocks.iter().enumerate() {
        if b.ncols() != p {
            return Err(format!(
                "concat_decoder_blocks: block {} has {} cols, expected {}",
                i,
                b.ncols(),
                p
            ));
        }
    }
    let total_k: usize = blocks.iter().map(|b| b.nrows()).sum();
    let mut out = Array2::<f64>::zeros((p, total_k));
    let mut col = 0_usize;
    for b in blocks {
        // Block has shape (basis_size, P); transpose into columns of out.
        for k in 0..b.nrows() {
            for row in 0..p {
                out[[row, col]] = b[[k, row]];
            }
            col += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn aux_richness_passes_on_rich_2d_aux() {
        let aux = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ];
        let lat = array![
            [0.10, 0.05],
            [0.02, 1.01],
            [1.05, 0.04],
            [1.01, 1.02],
            [2.03, 0.07],
            [2.04, 1.01],
            [0.05, 2.02],
            [1.02, 2.01],
            [2.01, 2.05],
        ];
        let m = aux_richness_metrics(aux.view(), lat.view());
        assert!(m.aux_observed);
        assert_eq!(m.aux_dim, 2);
        assert_eq!(m.latent_dim, 2);
        assert!(m.constant_columns.is_empty());
        assert!(m.aux_is_discrete);
        assert!(m.n_distinct_levels >= 3);
        assert!(m.jacobian_rank_estimated);
        assert_eq!(m.jacobian_rank, 2);
    }

    #[test]
    fn aux_richness_flags_constant_aux() {
        let aux = Array2::<f64>::zeros((20, 1));
        let mut lat = Array2::<f64>::zeros((20, 2));
        for i in 0..20 {
            lat[[i, 0]] = i as f64;
            lat[[i, 1]] = (i as f64).cos();
        }
        let m = aux_richness_metrics(aux.view(), lat.view());
        assert_eq!(m.aux_dim, 1);
        assert_eq!(m.latent_dim, 2);
        assert_eq!(m.constant_columns, vec![0_usize]);
    }

    #[test]
    fn aux_richness_flags_nonfinite_aux() {
        let mut aux = Array2::<f64>::zeros((10, 1));
        aux[[3, 0]] = f64::NAN;
        let lat = Array2::<f64>::zeros((10, 1));
        let m = aux_richness_metrics(aux.view(), lat.view());
        assert!(!m.aux_observed);
        assert_eq!(m.n_nonfinite_aux, 1);
    }

    #[test]
    fn jacobian_sparsity_passes_on_diagonal() {
        // P=4, K=3, n_samples=1; mostly zero.
        let j = array![
            [1.0_f64, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ];
        let m = jacobian_sparsity_metrics(j.view(), 1, 1.0e-3);
        assert_eq!(m.p_features, 4);
        assert_eq!(m.latent_dim, 3);
        assert!(m.mean_sparsity > 0.5);
        assert_eq!(m.ranks, vec![3_usize]);
    }

    #[test]
    fn jacobian_sparsity_dense_has_low_sparsity() {
        let mut j = Array2::<f64>::zeros((4, 3));
        for i in 0..4 {
            for k in 0..3 {
                j[[i, k]] = 1.0 + 0.1 * (i + k) as f64;
            }
        }
        let m = jacobian_sparsity_metrics(j.view(), 1, 1.0e-3);
        assert!(m.mean_sparsity < 0.1);
    }

    #[test]
    fn anchor_consistency_three_clusters() {
        let mut a = Array2::<f64>::from_elem((9, 3), 0.01);
        for i in 0..3 {
            a[[i, 0]] = 1.0;
        }
        for i in 3..6 {
            a[[i, 1]] = 1.0;
        }
        for i in 6..9 {
            a[[i, 2]] = 1.0;
        }
        let m = anchor_consistency_metrics(a.view(), 0.95);
        assert_eq!(m.n_atoms, 3);
        assert_eq!(m.n_anchors, 9);
        assert_eq!(m.anchors_per_atom, vec![3, 3, 3]);
    }

    #[test]
    fn anchor_consistency_uniform_has_zero_anchors() {
        let a = Array2::<f64>::from_elem((10, 4), 0.25);
        let m = anchor_consistency_metrics(a.view(), 0.95);
        assert_eq!(m.n_anchors, 0);
        assert_eq!(m.anchors_per_atom, vec![0, 0, 0, 0]);
    }

    #[test]
    fn anchor_consistency_report_owns_the_pass_fail_verdict() {
        let a = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let report = anchor_consistency_report(a.view(), None).unwrap();
        assert_eq!(report.anchor_dominance, ANCHOR_DOMINANCE_DEFAULT);
        assert!(report.passes());
        assert_eq!(
            report.preconditions,
            AnchorConsistencyPreconditions {
                enough_anchors_total: true,
                anchors_cover_all_atoms: true,
            }
        );
        assert!(report.uncovered_atoms.is_empty());
    }

    #[test]
    fn anchor_consistency_report_derives_thresholds_from_atom_count() {
        let a = Array2::<f64>::from_elem((7, 4), 0.25);
        let report = anchor_consistency_report(a.view(), Some(0.95)).unwrap();
        assert!(!report.passes());
        assert_eq!(
            report.preconditions,
            AnchorConsistencyPreconditions {
                enough_anchors_total: false,
                anchors_cover_all_atoms: false,
            }
        );
        assert_eq!(report.uncovered_atoms, vec![0, 1, 2, 3]);
        assert_eq!(report.violations.len(), 2);
        assert_eq!(report.recommendations.len(), report.violations.len());
        assert!(report.violations[0].contains("need at least 4"));
    }

    #[test]
    fn anchor_consistency_report_rejects_invalid_dominance() {
        let a = Array2::<f64>::ones((2, 2));
        let error = anchor_consistency_report(a.view(), Some(0.0)).unwrap_err();
        assert!(error.contains("anchor_dominance must be in (0.5, 1]"));
    }

    #[test]
    fn anchor_consistency_report_distinguishes_count_from_atom_coverage() {
        let a = array![
            [1.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let report = anchor_consistency_report(a.view(), None).unwrap();
        assert!(report.preconditions.enough_anchors_total);
        assert!(!report.preconditions.anchors_cover_all_atoms);
        assert_eq!(report.uncovered_atoms, vec![2]);
        assert!(!report.passes());
        assert_eq!(report.violations.len(), 1);
    }
}
