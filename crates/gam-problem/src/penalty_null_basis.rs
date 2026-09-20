//! The null space a penalty's declared nullity names, and what survives of it
//! under a coefficient map (gam#3023).
//!
//! A block declares `nullspace_dims[k]`, the structural nullity of penalty `S`.
//! That is a fact about the exact matrix. The stored one only approximates it:
//! its null eigenvalues sit at `O(u·‖S‖)`, not at zero. The identifiability
//! canonicaliser reparameterises a block as `β = Tθ`, where `T` has
//! orthonormal columns. `T` is a column selection `I[:, K]` or an orthonormal
//! `V_b`, and the reduced penalty is `TᵀST`. For PSD `S`,
//! `uᵀTᵀSTu = ‖S^{1/2}Tu‖²`, so `TᵀSTu = 0` exactly when `Tu ∈ ker S`. `T` is
//! injective, so with an orthonormal null basis `N` (`p × m`) of `S`:
//!
//! ```text
//! nullity(TᵀST) = dim(ker S ∩ range T) = m − rank((I − TTᵀ) N).
//! ```
//!
//! A null vector `Nc` lies in `range T` exactly when `(I − TTᵀ)Nc = 0`. The
//! dimension `m` alone does not determine this. A basis does.
//!
//! # The basis, from the declaration
//!
//! Given `m`, the null basis is the eigenvectors of the stored matrix's `m`
//! smallest eigenvalues `λ̂₀ ≤ … ≤ λ̂_{m−1}`. How close those eigenvectors are
//! to the exact null space is a Davis–Kahan question. Let
//! `E = p·ε·‖S‖₂ + formation_band` be the eigensolver's Weyl band plus the
//! producer's bound on the stored matrix's formation error. Then:
//!
//! * the residual `ρ = max_{i<m} |λ̂ᵢ| + E` bounds `‖S N̂‖₂` for the computed
//!   `N̂`;
//! * `δ = λ̂_m − E − ρ` bounds from below the separation between `N̂ᵀSN̂`'s
//!   spectrum (inside `[−ρ, ρ]`) and `S`'s remaining eigenvalues (each at least
//!   `λ̂_m − E`, by Weyl).
//!
//! When `δ > 0`, `‖sin Θ(N̂, ker S)‖₂ ≤ ρ/δ` (Davis and Kahan 1970, the sin Θ
//! theorem in residual form), and `min_Q ‖N̂ − N₀Q‖₂ ≤ √2·sin Θ` for an
//! orthonormal basis `N₀` of `ker S`. The eigensolver's loss of orthogonality,
//! `p·ε`, is added to that. When `δ ≤ 0`, the declared null space is not
//! separated from the rest of the spectrum, and no basis is taken. That loses
//! nothing: the spectrum then resolves fewer than `p − m` eigenvalues, so the
//! declared cap `min(resolved, p − m)` would not bind anyway.
//!
//! `ρ` includes the stored null eigenvalues themselves. They are the part of
//! the formation error that shows up in the null space, so a producer that
//! passes `formation_band = 0` still has its visible rounding counted. If the
//! true error is larger than the bound, a null direction that lies in `range T`
//! reads as resolved. That understates the reduced nullity, which is the safe
//! direction.
//!
//! # What an error in the basis can cost
//!
//! A declared nullity only caps the rank the spectral rule would keep, so
//! overstating it is the dangerous direction. Suppose the singular values of
//! the formed `R = N̂ − T(TᵀN̂)` sit within `b` of those of `(I − TTᵀ)N₀`. A
//! column counted null then has computed singular value at most `b`, so its
//! exact component outside `range T` is at most `2b`. Its null vector `n`
//! therefore pulls back to a reduced direction `u = Tᵀn` with
//! `‖u‖² ≥ 1 − 4b²`. Since `Tu = n − (I − TTᵀ)n`, its reduced curvature is
//! `(Tu)ᵀS(Tu) ≤ 4b²·‖S‖₂`. By Courant–Fischer, the cap removes only reduced
//! eigenvalues at or below `4b²‖S‖₂/(1 − 4b²)`.
//!
//! The reduced matrix inherits the stored penalty's absolute rounding, so an
//! eigenvalue below `S`'s own resolution `p·ε·‖S‖₂ + formation_band` is not a
//! measurement. The pulled-back declaration is therefore used only when
//! `4b²‖S‖₂ ≤ resolution·(1 − 4b²)`. Otherwise it is 0, and the spectral rule
//! decides alone.
//!
//! # Block-local penalties
//!
//! A [`PenaltyMatrix::Blockwise`] penalty declares the nullity of its `local`
//! matrix. That is the convention the block solver roots it by. At block
//! width, its null space is `local`'s null space embedded at `col_range`, plus
//! every coefficient axis outside `col_range`. Those axes are exact.

use crate::PenaltyMatrix;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use gam_linalg::roundoff::{
    accumulation_growth, factor_singular_band, symmetric_spectrum_rounding_band,
};
use ndarray::Array2;

/// Why a declared null basis could not be formed or pulled back.
#[derive(Clone, Debug, PartialEq)]
pub enum PenaltyNullBasisError {
    NotSquare { nrows: usize, ncols: usize },
    NullityExceedsDimension { dim: usize, declared: usize },
    MalformedBlockwise { local: usize, col_range: std::ops::Range<usize>, total_dim: usize },
    TransformShape { dim: usize, rows: usize, cols: usize },
    EigendecompositionFailed { reason: String },
    SingularValuesFailed { reason: String },
}

impl std::fmt::Display for PenaltyNullBasisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "penalty matrix is not square: {nrows}x{ncols}")
            }
            Self::NullityExceedsDimension { dim, declared } => {
                write!(f, "penalty declares nullity {declared} above its dimension {dim}")
            }
            Self::MalformedBlockwise {
                local,
                col_range,
                total_dim,
            } => write!(
                f,
                "block-local penalty of width {local} at columns {col_range:?} does not fit \
                 total dimension {total_dim}"
            ),
            Self::TransformShape { dim, rows, cols } => write!(
                f,
                "coefficient transform is {rows}x{cols}; a pullback of a penalty of \
                 dimension {dim} needs {dim} rows and at most {dim} columns"
            ),
            Self::EigendecompositionFailed { reason } => {
                write!(f, "penalty eigendecomposition failed: {reason}")
            }
            Self::SingularValuesFailed { reason } => {
                write!(f, "null-residual singular values failed: {reason}")
            }
        }
    }
}

impl std::error::Error for PenaltyNullBasisError {}

/// An orthonormal basis of a subspace of a penalty's exact null space, with a
/// bound on its distance from the exact one (see the module documentation).
#[derive(Clone, Debug)]
pub struct PenaltyNullBasis {
    /// `p × m`, orthonormal columns.
    basis: Array2<f64>,
    /// Bound on `min_Q ‖basis − N₀Q‖₂` for an orthonormal `N₀` of the exact
    /// null subspace the columns approximate.
    subspace_error: f64,
    /// `‖S‖₂` of the stored penalty.
    penalty_norm: f64,
    /// `p·ε·‖S‖₂ + formation_band`: the curvature below which the stored
    /// penalty does not resolve an eigenvalue from zero.
    resolution: f64,
}

impl PenaltyNullBasis {
    /// The null basis that `declared_nullity` names for `penalty`, taken from
    /// its stored spectrum. `formation_band` is the producer's bound, in
    /// eigenvalue units, on the error its construction left in the stored
    /// matrix. Callers without such a bound pass 0, as they do for
    /// [`crate::structural_penalty_root`].
    ///
    /// When the declared null space is not separated from the rest of the
    /// spectrum, or the basis is too uncertain to certify, the dense part
    /// contributes no columns. Fewer columns can only understate a pulled-back
    /// nullity.
    pub fn from_declared_nullity(
        penalty: &PenaltyMatrix,
        declared_nullity: usize,
        formation_band: f64,
    ) -> Result<Self, PenaltyNullBasisError> {
        match penalty {
            PenaltyMatrix::Labeled { inner, .. } | PenaltyMatrix::Fixed { inner, .. } => {
                Self::from_declared_nullity(inner, declared_nullity, formation_band)
            }
            PenaltyMatrix::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                if col_range.end > *total_dim || col_range.len() != local.nrows() {
                    return Err(PenaltyNullBasisError::MalformedBlockwise {
                        local: local.nrows(),
                        col_range: col_range.clone(),
                        total_dim: *total_dim,
                    });
                }
                let local_basis = Self::from_dense(local, declared_nullity, formation_band)?;
                Ok(local_basis.embedded(col_range.clone(), *total_dim))
            }
            PenaltyMatrix::Dense(_)
            | PenaltyMatrix::Diagonal(_)
            | PenaltyMatrix::KroneckerFactored { .. } => {
                Self::from_dense(&penalty.as_dense_cow(), declared_nullity, formation_band)
            }
        }
    }

    /// Dimension of the null subspace the basis spans.
    pub fn nullity(&self) -> usize {
        self.basis.ncols()
    }

    /// The orthonormal basis, `p × nullity`.
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// Bound on the basis's distance from the exact null subspace.
    pub fn subspace_error(&self) -> f64 {
        self.subspace_error
    }

    /// The certified nullity of `TᵀST` for a `transform` `T` (`p × r`, `r ≤ p`)
    /// with orthonormal columns: `m − rank((I − TTᵀ)N)`, where the rank counts
    /// the singular values above their error band. It is 0 when that band is
    /// too wide to certify (see the module documentation).
    pub fn pulled_back_nullity(
        &self,
        transform: &Array2<f64>,
    ) -> Result<usize, PenaltyNullBasisError> {
        let p = self.basis.nrows();
        let (rows, r) = transform.dim();
        if rows != p || r > p {
            return Err(PenaltyNullBasisError::TransformShape {
                dim: p,
                rows,
                cols: r,
            });
        }
        let m = self.nullity();
        if m == 0 || r == 0 {
            return Ok(0);
        }
        // `T` is square with orthonormal columns, so it is invertible and
        // `TᵀST` is a congruence of `S`: nullity is preserved exactly.
        if r == p {
            return Ok(m);
        }
        // The residual `R = N − T(TᵀN)`. Entrywise, its formation error is at
        // most `γ_{p+r+1}(|N| + |T||T|ᵀ|N|)` (an inner product of depth `p`,
        // one of depth `r`, one subtraction). `‖|N|‖_F = √m`, and
        // `‖|T||T|ᵀ|N|‖₂ ≤ ‖T‖_F²·‖N‖_F = r·√m`.
        let projected = transform.t().dot(&self.basis);
        let residual = &self.basis - &transform.dot(&projected);
        let formation = accumulation_growth(p + r + 1) * (r as f64 + 1.0) * (m as f64).sqrt();
        // `T`'s measured distance from orthonormality. `TTᵀ` differs from the
        // orthogonal projector onto `range T` by at most `d(1 + d)`, where
        // `d ≥ ‖TᵀT − I‖₂`. That bound includes the Gram's own formation error,
        // `γ_p·‖T‖_F² = γ_p·r`.
        let mut gram = transform.t().dot(transform);
        for i in 0..r {
            gram[[i, i]] -= 1.0;
        }
        let defect = gram.iter().map(|v| v * v).sum::<f64>().sqrt()
            + accumulation_growth(p) * r as f64;
        let (_, singular, _) = residual
            .svd(false, false)
            .map_err(|e| PenaltyNullBasisError::SingularValuesFailed {
                reason: e.to_string(),
            })?;
        let s_max = singular.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        let band = self.subspace_error
            + formation
            + defect * (1.0 + defect)
            + factor_singular_band(p, m, s_max);
        if !over_cut_is_unresolved(band, self.penalty_norm, self.resolution) {
            return Ok(0);
        }
        let rank = singular.iter().filter(|&&s| s > band).count();
        Ok(m - rank)
    }

    fn from_dense(
        matrix: &Array2<f64>,
        declared_nullity: usize,
        formation_band: f64,
    ) -> Result<Self, PenaltyNullBasisError> {
        let p = matrix.nrows();
        if matrix.ncols() != p {
            return Err(PenaltyNullBasisError::NotSquare {
                nrows: p,
                ncols: matrix.ncols(),
            });
        }
        let m = declared_nullity;
        if m > p {
            return Err(PenaltyNullBasisError::NullityExceedsDimension { dim: p, declared: m });
        }
        let empty = |penalty_norm: f64, resolution: f64| Self {
            basis: Array2::zeros((p, 0)),
            subspace_error: 0.0,
            penalty_norm,
            resolution,
        };
        if m == 0 {
            return Ok(empty(0.0, 0.0));
        }
        // The declaration says the exact penalty is zero, so every axis is null.
        // The reduced curvature of any direction is then 0, which is within any
        // resolution.
        if m == p {
            return Ok(Self {
                basis: Array2::eye(p),
                subspace_error: 0.0,
                penalty_norm: 0.0,
                resolution: 0.0,
            });
        }
        let (eigenvalues, eigenvectors) = FaerEigh::eigh(matrix, faer::Side::Lower).map_err(
            |e| PenaltyNullBasisError::EigendecompositionFailed {
                reason: e.to_string(),
            },
        )?;
        let values = eigenvalues.to_vec();
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let penalty_norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let resolution = symmetric_spectrum_rounding_band(&values) + formation_band;
        let null_residual = order[..m]
            .iter()
            .fold(0.0_f64, |acc, &i| acc.max(values[i].abs()))
            + resolution;
        let separation = values[order[m]] - resolution - null_residual;
        if separation.is_nan() || separation <= 0.0 {
            return Ok(empty(penalty_norm, resolution));
        }
        let sin_theta = (null_residual / separation).min(1.0);
        let subspace_error = std::f64::consts::SQRT_2 * sin_theta + p as f64 * f64::EPSILON;
        if !over_cut_is_unresolved(subspace_error, penalty_norm, resolution) {
            return Ok(empty(penalty_norm, resolution));
        }
        let mut basis = Array2::<f64>::zeros((p, m));
        for (column, &index) in order[..m].iter().enumerate() {
            basis.column_mut(column).assign(&eigenvectors.column(index));
        }
        Ok(Self {
            basis,
            subspace_error,
            penalty_norm,
            resolution,
        })
    }

    /// The block-width basis of a block-local penalty: every axis outside
    /// `col_range` (exact), then the local basis embedded at `col_range`.
    fn embedded(self, col_range: std::ops::Range<usize>, total_dim: usize) -> Self {
        let outside: Vec<usize> = (0..total_dim).filter(|i| !col_range.contains(i)).collect();
        let local_nullity = self.basis.ncols();
        let mut basis = Array2::<f64>::zeros((total_dim, outside.len() + local_nullity));
        for (column, &axis) in outside.iter().enumerate() {
            basis[[axis, column]] = 1.0;
        }
        basis
            .slice_mut(ndarray::s![col_range, outside.len()..])
            .assign(&self.basis);
        Self {
            basis,
            subspace_error: self.subspace_error,
            penalty_norm: self.penalty_norm,
            resolution: self.resolution,
        }
    }
}

/// Whether null columns counted within `band` can cost only curvature that
/// the stored penalty does not resolve: `4b²‖S‖₂ ≤ resolution·(1 − 4b²)`.
fn over_cut_is_unresolved(band: f64, penalty_norm: f64, resolution: f64) -> bool {
    let q = 4.0 * band * band;
    q < 1.0 && q * penalty_norm <= resolution * (1.0 - q)
}

/// The declared nullity of `penalty` carried to the reduced penalty `TᵀST`,
/// at the reduced penalty's full width.
///
/// A square `T` is a congruence, so it carries the block-width declaration
/// unchanged. For a block-local penalty, that is `declared + (p − w)`. A
/// narrowing `T` is answered by [`PenaltyNullBasis::pulled_back_nullity`].
pub fn pulled_back_declared_nullity(
    penalty: &PenaltyMatrix,
    declared_nullity: usize,
    transform: &Array2<f64>,
    formation_band: f64,
) -> Result<usize, PenaltyNullBasisError> {
    let p = penalty.dim();
    let (rows, r) = transform.dim();
    if rows != p || r > p {
        return Err(PenaltyNullBasisError::TransformShape {
            dim: p,
            rows,
            cols: r,
        });
    }
    if r == p {
        return block_width_declared_nullity(penalty, declared_nullity);
    }
    PenaltyNullBasis::from_declared_nullity(penalty, declared_nullity, formation_band)?
        .pulled_back_nullity(transform)
}

/// The declared nullity of `penalty` at block width. A block-local penalty
/// declares its local matrix's nullity, and the axes outside its columns
/// are null too.
fn block_width_declared_nullity(
    penalty: &PenaltyMatrix,
    declared_nullity: usize,
) -> Result<usize, PenaltyNullBasisError> {
    match penalty {
        PenaltyMatrix::Labeled { inner, .. } | PenaltyMatrix::Fixed { inner, .. } => {
            block_width_declared_nullity(inner, declared_nullity)
        }
        PenaltyMatrix::Blockwise {
            local,
            col_range,
            total_dim,
        } => {
            if col_range.end > *total_dim || col_range.len() != local.nrows() {
                return Err(PenaltyNullBasisError::MalformedBlockwise {
                    local: local.nrows(),
                    col_range: col_range.clone(),
                    total_dim: *total_dim,
                });
            }
            if declared_nullity > local.nrows() {
                return Err(PenaltyNullBasisError::NullityExceedsDimension {
                    dim: local.nrows(),
                    declared: declared_nullity,
                });
            }
            Ok(declared_nullity + (*total_dim - local.nrows()))
        }
        PenaltyMatrix::Dense(_)
        | PenaltyMatrix::Diagonal(_)
        | PenaltyMatrix::KroneckerFactored { .. } => {
            let p = penalty.dim();
            if declared_nullity > p {
                return Err(PenaltyNullBasisError::NullityExceedsDimension {
                    dim: p,
                    declared: declared_nullity,
                });
            }
            Ok(declared_nullity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::resolved_eigenvalue_count;

    /// `DᵀD` for the order-`order` difference operator on `p` coefficients.
    /// Its null space is the polynomials of degree below `order`.
    fn difference_penalty(p: usize, order: usize) -> Array2<f64> {
        let mut d = Array2::<f64>::eye(p);
        for _ in 0..order {
            let rows = d.nrows() - 1;
            let mut next = Array2::<f64>::zeros((rows, p));
            for i in 0..rows {
                for j in 0..p {
                    next[[i, j]] = d[[i + 1, j]] - d[[i, j]];
                }
            }
            d = next;
        }
        d.t().dot(&d)
    }

    fn selection(p: usize, kept: &[usize]) -> Array2<f64> {
        let mut t = Array2::<f64>::zeros((p, kept.len()));
        for (column, &row) in kept.iter().enumerate() {
            t[[row, column]] = 1.0;
        }
        t
    }

    /// `r − resolved` of `TᵀST`: what the spectrum alone says.
    fn spectral_nullity(s: &Array2<f64>, t: &Array2<f64>) -> usize {
        let reduced = t.t().dot(&s.dot(t));
        let (values, _) = FaerEigh::eigh(&reduced, faer::Side::Lower).unwrap();
        reduced.nrows() - resolved_eigenvalue_count(&values.to_vec(), 0.0)
    }

    #[test]
    fn selection_keeps_the_null_directions_that_vanish_on_the_dropped_columns() {
        // ker(D₂ᵀD₂) = span{1, t}. Dropping column 0 keeps `t`, which is 0
        // there. Dropping columns 0 and 7 keeps nothing, since `a + b·t` has
        // two distinct roots.
        let p = 8;
        let s = PenaltyMatrix::Dense(difference_penalty(p, 2));
        let one_dropped = selection(p, &(1..p).collect::<Vec<_>>());
        let two_dropped = selection(p, &(1..p - 1).collect::<Vec<_>>());
        assert_eq!(pulled_back_declared_nullity(&s, 2, &one_dropped, 0.0).unwrap(), 1);
        assert_eq!(pulled_back_declared_nullity(&s, 2, &two_dropped, 0.0).unwrap(), 0);
        let dense = s.to_dense();
        assert_eq!(spectral_nullity(&dense, &one_dropped), 1);
        assert_eq!(spectral_nullity(&dense, &two_dropped), 0);
    }

    #[test]
    fn orthonormal_narrowing_keeps_the_null_directions_inside_its_range() {
        // `V` spans the complement of the constant vector. Of span{1, t}, only
        // `t − t̄·1` survives.
        let p = 9;
        let s = difference_penalty(p, 2);
        let centering = Array2::<f64>::eye(p) - Array2::<f64>::from_elem((p, p), 1.0 / p as f64);
        let (values, vectors) = FaerEigh::eigh(&centering, faer::Side::Lower).unwrap();
        let kept: Vec<usize> = (0..p).filter(|&i| values[i] > 0.5).collect();
        assert_eq!(kept.len(), p - 1);
        let mut v = Array2::<f64>::zeros((p, p - 1));
        for (column, &index) in kept.iter().enumerate() {
            v.column_mut(column).assign(&vectors.column(index));
        }
        let penalty = PenaltyMatrix::Dense(s);
        assert_eq!(pulled_back_declared_nullity(&penalty, 2, &v, 0.0).unwrap(), 1);
    }

    #[test]
    fn square_transform_carries_the_declaration_at_block_width() {
        let p = 8;
        let local = difference_penalty(4, 2);
        let blockwise = PenaltyMatrix::Blockwise {
            local,
            col_range: 2..6,
            total_dim: p,
        }
        .with_precision_label("wiggle");
        let identity = Array2::<f64>::eye(p);
        // Two local null directions plus the four axes outside columns 2..6.
        assert_eq!(pulled_back_declared_nullity(&blockwise, 2, &identity, 0.0).unwrap(), 6);
        // Dropping axis 0 (outside the local range) removes one exact null axis.
        let t = selection(p, &(1..p).collect::<Vec<_>>());
        assert_eq!(pulled_back_declared_nullity(&blockwise, 2, &t, 0.0).unwrap(), 5);
        assert_eq!(spectral_nullity(&blockwise.to_dense(), &t), 5);
        // Dropping local column 2 leaves the local `t` direction that is 0 there.
        let t = selection(p, &(0..p).filter(|&i| i != 2).collect::<Vec<_>>());
        assert_eq!(pulled_back_declared_nullity(&blockwise, 2, &t, 0.0).unwrap(), 5);
        assert_eq!(spectral_nullity(&blockwise.to_dense(), &t), 5);
    }

    #[test]
    fn declaration_holds_through_formation_error_the_producer_bounds() {
        // A stored penalty that carries symmetric formation error of
        // spectral norm at most `noise_norm`, with the bound passed as the
        // formation band.
        let p = 10;
        let mut s = difference_penalty(p, 2);
        let mut noise = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..=i {
                let value = 1e-12 * (((i * 7 + j * 3) % 5) as f64 - 2.0);
                noise[[i, j]] = value;
                noise[[j, i]] = value;
            }
        }
        let noise_norm = noise.iter().map(|v| v * v).sum::<f64>().sqrt();
        s += &noise;
        let penalty = PenaltyMatrix::Dense(s);
        let t = selection(p, &(1..p).collect::<Vec<_>>());
        assert_eq!(pulled_back_declared_nullity(&penalty, 2, &t, noise_norm).unwrap(), 1);
    }

    #[test]
    fn an_overstated_declaration_is_not_certified_through_a_narrowing() {
        // ker(D₁ᵀD₁) = span{1}. Declaring 2 treats a curved eigenvector as
        // null, and that is not separated from the rest of the spectrum.
        let p = 8;
        let penalty = PenaltyMatrix::Dense(difference_penalty(p, 1));
        let basis = PenaltyNullBasis::from_declared_nullity(&penalty, 2, 0.0).unwrap();
        assert_eq!(basis.nullity(), 0);
        let t = selection(p, &(1..p).collect::<Vec<_>>());
        assert_eq!(pulled_back_declared_nullity(&penalty, 2, &t, 0.0).unwrap(), 0);
    }

    #[test]
    fn declared_basis_spans_the_exact_null_space() {
        let p = 12;
        let penalty = PenaltyMatrix::Dense(difference_penalty(p, 3));
        let basis = PenaltyNullBasis::from_declared_nullity(&penalty, 3, 0.0).unwrap();
        assert_eq!(basis.nullity(), 3);
        // Each quadratic `t²` lies in the span: its residual after projection is
        // within the certified error.
        let t2 = ndarray::Array1::from_shape_fn(p, |i| (i * i) as f64);
        let t2 = &t2 / t2.dot(&t2).sqrt();
        let n = basis.basis();
        let residual = &t2 - &n.dot(&n.t().dot(&t2));
        assert!(residual.dot(&residual).sqrt() <= basis.subspace_error());
    }

    #[test]
    fn refuses_a_declaration_above_the_dimension_and_a_widening_transform() {
        let penalty = PenaltyMatrix::Dense(difference_penalty(4, 1));
        assert!(matches!(
            pulled_back_declared_nullity(&penalty, 5, &Array2::eye(4), 0.0),
            Err(PenaltyNullBasisError::NullityExceedsDimension { dim: 4, declared: 5 })
        ));
        assert!(matches!(
            pulled_back_declared_nullity(&penalty, 1, &Array2::zeros((4, 5)), 0.0),
            Err(PenaltyNullBasisError::TransformShape { .. })
        ));
    }
}
