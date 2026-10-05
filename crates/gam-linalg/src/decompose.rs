//! Dense float64 decompositions on faer with deterministic signs and rounding bands.
//!
//! Every decomposition is faer's at the parallelism [`decomposition_parallelism`] names, never
//! LAPACK's, and two runs of one matrix agree bit for bit. A caller that serializes a frame, reads
//! a determinant off one, or compares two fits needs the same vectors every time, not the signs one
//! solver run happened to return: [`canonical_sign`] is the one rule for that, here and in every
//! crate that fixes a sign itself.
//!
//! # Sign conventions
//!
//! A decomposition fixes each vector only up to sign. Here:
//! - each eigenvector (column) has its largest-magnitude entry positive (the first
//!   such entry on a tie);
//! - each left singular vector (column of `U`) likewise, and the matching right
//!   singular vector (row of `Vᵀ`) flips with it, so `U diag(σ) Vᵀ` is unchanged; a
//!   right singular vector with no left partner (a full `Vᵀ` past `min(m, n)`) takes
//!   the rule itself;
//! - QR has `diag(R) ≥ 0`: row `i` of `R` and column `i` of `Q` flip together.
//!
//! Within a repeated eigenvalue or singular value the basis of the eigenspace is
//! whatever faer returns; only signs are canonical.
//!
//! # Symmetry
//!
//! A symmetric eigensolver reads one triangle. [`eigh`] first checks the other one
//! against the band the caller's [`SymmetricAssembly`] declares
//! ([`strict_symmetric_eigh`]), so an asymmetric matrix is refused rather than
//! silently decomposed from its lower triangle.
//!
//! # Bands
//!
//! Singular values carry [`factor_singular_band`] (`max(m, n) ε σ₁`) and eigenvalues
//! [`symmetric_spectrum_rounding_band`] (`n (ε ρ + η)`): a value within its band of
//! zero is not resolved from zero by the decomposition that produced it.

use std::fmt;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{self as faer_svd, ComputeSvdVectors};
use faer::{Mat, MatRef, Side};
use crate::faer_ndarray::{
    FaerArrayView, FaerLinalgError, FaerLu, FaerQr, FaerSvd, decomposition_parallelism, fast_abt, strict_symmetric_eigh,
};
use crate::matrix::symmetrize_in_place;
use crate::roundoff::{SymmetricAssembly, factor_singular_band, symmetric_spectrum_rounding_band};
use ndarray::{Array1, Array2, ArrayView2, Axis, concatenate, s};

/// Why a dense decomposition was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseError {
    /// An entry is NaN or infinite.
    NonFinite { what: &'static str },
    /// The operation needs a square matrix.
    NotSquare { rows: usize, cols: usize },
    /// Two shapes that must agree do not.
    Shape { what: &'static str, expected: usize, found: usize },
    /// `LU` met a zero or non-finite pivot at this column.
    Singular { column: usize },
    /// A requested eigenvalue index range is empty or outside `0..n`.
    InvalidRange { start: usize, end: usize, order: usize },
    /// The two triangles of a symmetric input disagree beyond the declared assembly's band.
    Asymmetric { detail: String },
    /// The decomposition did not converge.
    Decomposition { detail: String },
    /// A PSD resolution floor is negative or non-finite.
    InvalidFloor { value: f64 },
    /// An eigenvalue of a matrix declared positive semidefinite lies below `−band`, the larger of
    /// the caller's floor and the decomposition's rounding band: indefiniteness neither resolves.
    Indefinite { index: usize, value: f64, band: f64 },
}

impl fmt::Display for DenseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what } => write!(formatter, "dense {what}: non-finite entry"),
            Self::NotSquare { rows, cols } => write!(formatter, "dense: a {rows} x {cols} matrix is not square"),
            Self::Shape { what, expected, found } => {
                write!(formatter, "dense {what}: expected {expected}, found {found}")
            }
            Self::Singular { column } => write!(formatter, "dense solve: singular at column {column}"),
            Self::InvalidRange { start, end, order } => write!(
                formatter,
                "dense eigh: eigenvalue indices {start}..{end} are empty or outside 0..{order}"
            ),
            Self::Asymmetric { detail } => write!(formatter, "dense eigh: {detail}"),
            Self::Decomposition { detail } => write!(formatter, "dense: decomposition failed: {detail}"),
            Self::InvalidFloor { value } => write!(formatter,"dense PSD map: floor must be finite and nonnegative, got {value:?}"),
            Self::Indefinite { index, value, band } => {
                write!(formatter, "indefinite at eigenvalue {index}: {value:.3e} < -{band:.3e}")
            }
        }
    }
}

impl std::error::Error for DenseError {}

fn require_finite(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<(), DenseError> {
    if matrix.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(DenseError::NonFinite { what })
    }
}

fn require_square(matrix: ArrayView2<'_, f64>) -> Result<usize, DenseError> {
    let (rows, cols) = matrix.dim();
    if rows == cols {
        Ok(rows)
    } else {
        Err(DenseError::NotSquare { rows, cols })
    }
}

fn decomposition(error: impl fmt::Debug) -> DenseError {
    DenseError::Decomposition {
        detail: format!("{error:?}"),
    }
}

fn to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(row, col)| mat[(row, col)])
}

/// `-1` when the first largest-magnitude entry of `vector` is negative, else `1` (also for a zero
/// vector): the sign that makes a vector defined only up to sign canonical.
pub fn canonical_sign(vector: impl IntoIterator<Item = f64>) -> f64 {
    let mut best = 0.0_f64;
    let mut sign = 1.0;
    for value in vector {
        if value.abs() > best {
            best = value.abs();
            sign = if value < 0.0 { -1.0 } else { 1.0 };
        }
    }
    sign
}

/// Flips every column of `vectors` to its [`canonical_sign`], and returns the signs.
pub fn canonical_column_signs(vectors: &mut Array2<f64>) -> Vec<f64> {
    let mut signs = Vec::with_capacity(vectors.ncols());
    for mut column in vectors.columns_mut() {
        let sign = canonical_sign(column.iter().copied());
        if sign < 0.0 {
            column.mapv_inplace(|value| -value);
        }
        signs.push(sign);
    }
    signs
}

/// A symmetric eigendecomposition `A = V diag(λ) Vᵀ`.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigh {
    /// Eigenvalues in increasing order.
    pub values: Array1<f64>,
    /// The matching eigenvectors as columns, `n × k`.
    pub vectors: Array2<f64>,
    /// Every eigenvalue of the whole spectrum is within this of an exact one.
    pub band: f64,
}

impl Eigh {
    /// The spectral function `V diag(f(λ)) Vᵀ` over the eigenpairs held, mirrored to exact
    /// symmetry; an eigenpair with `f(λ) = 0` (one below a caller's floor, say) adds nothing and
    /// costs nothing.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Array2<f64> {
        let weights: Vec<(usize, f64)> = self.values.iter().map(|&value| f(value)).enumerate().filter(|(_, weight)| *weight != 0.0).collect();
        let kept: Vec<usize> = weights.iter().map(|(index, _)| *index).collect();
        let vectors = self.vectors.select(Axis(1), &kept);
        let mut scaled = vectors.clone();
        for (mut column, (_, weight)) in scaled.columns_mut().into_iter().zip(&weights) {
            column.mapv_inplace(|value| value * weight);
        }
        let mut out = fast_abt(&scaled, &vectors);
        symmetrize_in_place(&mut out);
        out
    }

    /// [`Eigh::map`] of a matrix declared positive semidefinite: `f` at the eigenvalues above
    /// `floor`, the rest dropped. A negative eigenvalue within the larger of `floor` and the
    /// decomposition's band is unresolved from zero (the resolution that drops a positive one that
    /// small cannot read its sign) and is dropped with it; one below that is refused
    /// ([`DenseError::Indefinite`]), never repaired.
    pub fn psd_map(&self, floor: f64, f: impl Fn(f64) -> f64) -> Result<Array2<f64>, DenseError> {
        if !floor.is_finite() || floor<0.0 {return Err(DenseError::InvalidFloor {value:floor});}
        let band = self.band.max(floor);
        if let Some((index, &value)) = self.values.iter().enumerate().find(|(_, value)| **value < -band) {
            return Err(DenseError::Indefinite { index, value, band });
        }
        Ok(self.map(|value| if value > floor { f(value) } else { 0.0 }))
    }
}

/// The eigendecomposition of the symmetric matrix `a`; with `indices`, only the
/// eigenpairs `indices` in increasing eigenvalue order.
///
/// `assembly` declares how `a` was built, which fixes how far its two triangles may
/// disagree ([`SymmetricAssembly::Mirrored`]: not at all). A matrix outside that band
/// is refused ([`DenseError::Asymmetric`]).
pub fn eigh(
    a: ArrayView2<'_, f64>,
    assembly: SymmetricAssembly,
    indices: Option<(usize, usize)>,
) -> Result<Eigh, DenseError> {
    let order = require_square(a)?;
    require_finite("eigh", a)?;
    let (start, end) = indices.unwrap_or((0, order));
    if start >= end || end > order {
        return Err(DenseError::InvalidRange { start, end, order });
    }
    let (values, vectors) = strict_symmetric_eigh(&a, assembly, Side::Lower).map_err(|error| match error {
        FaerLinalgError::StrictSelfAdjointEigenInvalidInput { reason } => DenseError::Asymmetric { detail: reason },
        other => decomposition(other),
    })?;
    let band = symmetric_spectrum_rounding_band(values.as_slice().unwrap_or(&values.to_vec()));
    let mut ranked: Vec<usize> = (0..order).collect();
    ranked.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let chosen = &ranked[start..end];
    let values = Array1::from_iter(chosen.iter().map(|&index| values[index]));
    let mut vectors = vectors.select(Axis(1), chosen);
    canonical_column_signs(&mut vectors);
    Ok(Eigh { values, vectors, band })
}

/// A singular value decomposition `A = U diag(σ) Vᵀ`.
#[derive(Clone, Debug, PartialEq)]
pub struct Svd {
    /// `m × k`, `k = min(m, n)` (thin) or `m` (full).
    pub u: Array2<f64>,
    /// `σ₁ ≥ σ₂ ≥ … ≥ 0`, `min(m, n)` of them.
    pub singular_values: Array1<f64>,
    /// `k × n`, `k = min(m, n)` (thin) or `n` (full).
    pub vt: Array2<f64>,
    /// Every singular value is within this of an exact one.
    pub band: f64,
}

fn singular_band(a: ArrayView2<'_, f64>, singular_values: &Array1<f64>) -> f64 {
    let sigma_max = singular_values.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    factor_singular_band(a.nrows(), a.ncols(), sigma_max)
}

/// The thin (`full = false`) or full singular value decomposition of `a`.
pub fn svd(a: ArrayView2<'_, f64>, full: bool) -> Result<Svd, DenseError> {
    require_finite("svd", a)?;
    let (rows, cols) = a.dim();
    let (u, sigma, vt) = if full {
        full_svd(a)?
    } else {
        let (u, sigma, vt) = a.svd(true, true).map_err(decomposition)?;
        let missing = || DenseError::Decomposition {
            detail: "faer returned no singular vectors".to_string(),
        };
        (u.ok_or_else(missing)?, sigma, vt.ok_or_else(missing)?)
    };
    // Decreasing order, carrying the vectors.
    let rank = rows.min(cols);
    let mut order: Vec<usize> = (0..rank).collect();
    order.sort_by(|&left, &right| sigma[right].total_cmp(&sigma[left]));
    let leading_u = u.select(Axis(1), &order);
    let leading_vt = vt.select(Axis(0), &order);
    let mut u = concatenate![Axis(1), leading_u, u.slice(s![.., rank..])];
    let mut vt = concatenate![Axis(0), leading_vt, vt.slice(s![rank.., ..])];
    let singular_values = Array1::from_iter(order.iter().map(|&index| sigma[index]));
    let signs = canonical_column_signs(&mut u);
    for (index, mut row) in vt.rows_mut().into_iter().enumerate() {
        let sign = if index < rank { signs[index] } else { canonical_sign(row.iter().copied()) };
        if sign < 0.0 {
            row.mapv_inplace(|value| -value);
        }
    }
    let band = singular_band(a, &singular_values);
    Ok(Svd {
        u,
        singular_values,
        vt,
        band,
    })
}

impl Svd {
    /// The indices of the singular values beyond the band: the directions the decomposition
    /// resolves from zero.
    fn resolved(&self) -> Vec<usize> {
        (0..self.singular_values.len()).filter(|&index| self.singular_values[index] > self.band).collect()
    }
}

/// The Moore–Penrose pseudo-inverse `A⁺ = V Σ⁺ Uᵀ` of `a` (`n × m` for an `m × n` `a`), with
/// `σᵢ⁺ = 1/σᵢ` for a singular value beyond the decomposition's band and `0` within it: a value
/// within the band is not resolved from zero, so its direction is dropped rather than amplified.
pub fn pseudo_inverse(a: ArrayView2<'_, f64>) -> Result<Array2<f64>, DenseError> {
    let d = svd(a, false)?;
    let kept = d.resolved();
    let inverse = Array1::from_iter(kept.iter().map(|&index| 1.0 / d.singular_values[index]));
    let scaled = &d.u.select(Axis(1), &kept).t() * &inverse.insert_axis(Axis(1));
    Ok(d.vt.select(Axis(0), &kept).t().dot(&scaled))
}

/// `A⁺ B` ([`pseudo_inverse`]'s truncation) without forming `A⁺`: the minimum-norm least-squares
/// solution `X` of `A X = B` over the directions `A`'s decomposition resolves.
pub fn pseudo_inverse_solve(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>, DenseError> {
    require_finite("pseudo-inverse right-hand side", b)?;
    if b.nrows() != a.nrows() {
        return Err(DenseError::Shape { what: "pseudo-inverse right-hand side rows", expected: a.nrows(), found: b.nrows() });
    }
    let d = svd(a, false)?;
    let kept = d.resolved();
    let mut projected = d.u.select(Axis(1), &kept).t().dot(&b);
    for (mut row, &index) in projected.rows_mut().into_iter().zip(&kept) {
        row /= d.singular_values[index];
    }
    Ok(d.vt.select(Axis(0), &kept).t().dot(&projected))
}

fn full_svd(a: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), DenseError> {
    let view = FaerArrayView::new(&a);
    let matrix = view.as_ref();
    let (rows, cols) = matrix.shape();
    let mut singular = faer::diag::Diag::<f64>::zeros(rows.min(cols));
    let mut u = Mat::<f64>::zeros(rows, rows);
    let mut v = Mat::<f64>::zeros(cols, cols);
    let par = decomposition_parallelism();
    let mut memory = MemBuffer::new(faer_svd::svd_scratch::<f64>(
        rows,
        cols,
        ComputeSvdVectors::Full,
        ComputeSvdVectors::Full,
        par,
        Default::default(),
    ));
    faer_svd::svd(
        matrix,
        singular.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut memory),
        Default::default(),
    )
    .map_err(decomposition)?;
    let sigma = Array1::from_iter((0..rows.min(cols)).map(|index| singular[index]));
    Ok((to_array(u.as_ref()), sigma, to_array(v.as_ref()).reversed_axes().as_standard_layout().into_owned()))
}

/// What a QR decomposition returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrMode {
    /// `Q` `m × k`, `R` `k × n`, `k = min(m, n)`.
    Economic,
    /// `Q` `m × m`, `R` `m × n`.
    Full,
    /// `R` alone, `k × n`.
    R,
}

/// `A = Q R` with `diag(R) ≥ 0`; `q` is absent in [`QrMode::R`].
#[derive(Clone, Debug, PartialEq)]
pub struct Qr {
    pub q: Option<Array2<f64>>,
    pub r: Array2<f64>,
}

/// The Householder QR of `a` (faer).
///
/// The full `Q` is the QR of `[A | I_m]`: Householder QR reduces columns in order, so
/// its first `n` columns are reduced exactly as `A`'s are, its `R` begins with `A`'s
/// `R` (zero below row `min(m, n)`), and its `Q` is `m × m`.
pub fn qr(a: ArrayView2<'_, f64>, mode: QrMode) -> Result<Qr, DenseError> {
    require_finite("qr", a)?;
    let (rows, cols) = a.dim();
    let (mut q, mut r) = match mode {
        QrMode::Full if rows > cols => {
            let augmented = concatenate![Axis(1), a, Array2::<f64>::eye(rows)];
            let (q, r) = augmented.qr().map_err(decomposition)?;
            (q, r.slice(s![.., ..cols]).to_owned())
        }
        _ => a.qr().map_err(decomposition)?,
    };
    for index in 0..r.nrows().min(r.ncols()) {
        if r[[index, index]] < 0.0 {
            r.row_mut(index).mapv_inplace(|value| -value);
            q.column_mut(index).mapv_inplace(|value| -value);
        }
    }
    Ok(Qr {
        q: (mode != QrMode::R).then_some(q),
        r,
    })
}

/// `A⁻¹ B` for a square `A` by partial-pivot LU; refused at a zero pivot.
pub fn solve(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>, DenseError> {
    let order = require_square(a)?;
    require_finite("solve matrix", a)?;
    require_finite("solve right-hand side", b)?;
    if b.nrows() != order {
        return Err(DenseError::Shape {
            what: "solve right-hand side rows",
            expected: order,
            found: b.nrows(),
        });
    }
    let matrix = FaerArrayView::new(&a);
    let lu = FaerLu::new(matrix.as_ref()).map_err(|column| DenseError::Singular { column })?;
    let rhs = FaerArrayView::new(&b);
    Ok(to_array(lu.solve(rhs.as_ref()).as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn close(left: &Array2<f64>, right: &Array2<f64>, tolerance: f64) -> bool {
        left.dim() == right.dim() && left.iter().zip(right).all(|(a, b)| (a - b).abs() <= tolerance)
    }

    #[test]
    fn pseudo_inverse_meets_the_penrose_conditions_and_drops_unresolved_directions() {
        // Rank 2 of a 4 × 3 matrix: the third column is the sum of the first two.
        let a = array![[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 1.0, 2.0], [3.0, -1.0, 2.0]];
        let p = pseudo_inverse(a.view()).expect("pseudo-inverse");
        assert_eq!(p.dim(), (3, 4));
        let tolerance = 1e-12;
        assert!(close(&a.dot(&p).dot(&a), &a, tolerance));
        assert!(close(&p.dot(&a).dot(&p), &p, tolerance));
        let (ap, pa) = (a.dot(&p), p.dot(&a));
        assert!(close(&ap, &ap.t().to_owned(), tolerance));
        assert!(close(&pa, &pa.t().to_owned(), tolerance));
        let b = array![[1.0, -2.0], [0.5, 0.0], [2.0, 1.0], [-1.0, 3.0]];
        assert!(close(&pseudo_inverse_solve(a.view(), b.view()).expect("solve"), &p.dot(&b), tolerance));
        assert!(matches!(pseudo_inverse_solve(a.view(), b.slice(s![..3, ..])), Err(DenseError::Shape { .. })));
    }

    #[test]
    fn psd_map_rejects_invalid_floor_before_mapping() {
        let matrix=array![[1.0,0.0],[0.0,2.0]];
        let d=eigh(matrix.view(),SymmetricAssembly::Mirrored,None).expect("finite PSD matrix");
        for floor in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY,-1.0] {
            let called=std::cell::Cell::new(false);
            assert!(matches!(d.psd_map(floor,|x| {called.set(true);x}),Err(DenseError::InvalidFloor {..})));
            assert!(!called.get());
        }
        assert!(d.psd_map(-0.0,|x|x).is_ok());
    }

    #[test]
    fn eigh_of_a_known_matrix_has_canonical_signs() {
        // Eigenvalues 1 and 3 of [[2, 1], [1, 2]], eigenvectors (1, −1)/√2 and (1, 1)/√2.
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let decomposed = eigh(a.view(), SymmetricAssembly::Mirrored, None).expect("eigh");
        assert!((decomposed.values[0] - 1.0).abs() < 1e-14 && (decomposed.values[1] - 3.0).abs() < 1e-14);
        let root = std::f64::consts::FRAC_1_SQRT_2;
        // Largest-magnitude entries tie; the first one is made positive.
        assert!(close(&decomposed.vectors, &array![[root, root], [-root, root]], 1e-14));
        let top = eigh(a.view(), SymmetricAssembly::Mirrored, Some((1, 2))).expect("subset");
        assert_eq!(top.values.len(), 1);
        assert!((top.values[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn an_asymmetric_matrix_beyond_its_band_is_refused() {
        // The upper triangle disagrees with the lower: refused, not read from one side.
        let lopsided = array![[2.0, 99.0], [1.0, 2.0]];
        assert!(matches!(eigh(lopsided.view(), SymmetricAssembly::Mirrored, None), Err(DenseError::Asymmetric { .. })));
        assert!(matches!(eigh(lopsided.view(), SymmetricAssembly::PsdAccumulation { depth: 64 }, None), Err(DenseError::Asymmetric { .. })));
        // A one-ulp disagreement is refused for a mirrored matrix and accepted inside
        // an accumulation's band.
        let off = 1.0_f64;
        let nudged = array![[2.0, off], [f64::from_bits(off.to_bits() + 1), 2.0]];
        assert!(matches!(eigh(nudged.view(), SymmetricAssembly::Mirrored, None), Err(DenseError::Asymmetric { .. })));
        let accepted = eigh(nudged.view(), SymmetricAssembly::PsdAccumulation { depth: 4 }, None).expect("inside the band").values;
        assert!((accepted[0] - 1.0).abs() < 1e-14 && (accepted[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn a_repeated_eigenvalue_keeps_an_orthonormal_eigenspace() {
        let a = Array2::<f64>::eye(3) * 2.0;
        let decomposed = eigh(a.view(), SymmetricAssembly::Mirrored, None).expect("eigh");
        assert!(decomposed.values.iter().all(|&value| (value - 2.0).abs() < 1e-14));
        let gram = decomposed.vectors.t().dot(&decomposed.vectors);
        assert!(close(&gram, &Array2::eye(3), 1e-14));
        for column in decomposed.vectors.columns() {
            assert!(canonical_sign(column.iter().copied()) > 0.0);
        }
    }

    #[test]
    fn a_spectral_function_rebuilds_the_matrix_and_its_pseudo_inverse_root() {
        // Rank two: eigenvalues 0, 1, 4 of a rotated diagonal.
        let q = qr(array![[1.0, 2.0, 0.5], [0.0, 1.0, -1.0], [3.0, 0.0, 1.0]].view(), QrMode::Economic).expect("qr").q.expect("q");
        let a = q.dot(&Array2::from_diag(&array![0.0, 1.0, 4.0])).dot(&q.t());
        let a = (&a + &a.t()) * 0.5;
        let decomposed = eigh(a.view(), SymmetricAssembly::Mirrored, None).expect("eigh");
        let rebuilt = decomposed.map(|value| value);
        assert!(close(&rebuilt, &a, 1e-13));
        assert_eq!(rebuilt, rebuilt.t());
        // The pseudo-inverse root over the eigenvalues above the band: R A R is A's range projector.
        let root = decomposed.map(|value| if value > decomposed.band { 1.0 / value.sqrt() } else { 0.0 });
        let projector = root.dot(&a).dot(&root);
        let range = q.select(Axis(1), &[1, 2]);
        assert!(close(&projector, &range.dot(&range.t()), 1e-13));
        // Nothing kept is the zero matrix.
        assert_eq!(decomposed.map(|_| 0.0), Array2::<f64>::zeros((3, 3)));
        // As a PSD function with a floor at the band it is the same root; a negative eigenvalue past
        // the floor is refused, one inside it dropped.
        let checked = decomposed.psd_map(decomposed.band, |value| 1.0 / value.sqrt()).expect("positive semidefinite");
        assert_eq!(checked, root);
        let shifted = eigh((&a - &Array2::<f64>::eye(3) * 1e-3).view(), SymmetricAssembly::Mirrored, None).expect("eigh");
        assert!(matches!(shifted.psd_map(0.0, |value| value), Err(DenseError::Indefinite { index: 0, .. })));
        assert!(shifted.psd_map(1e-2, |value| value).is_ok());
    }

    #[test]
    fn svd_reconstructs_and_is_sign_canonical_including_rank_deficiency() {
        // Rank one: σ = (5√2, 0).
        let a = array![[3.0, 4.0], [3.0, 4.0], [0.0, 0.0]];
        for full in [false, true] {
            let decomposed = svd(a.view(), full).expect("svd");
            assert!((decomposed.singular_values[0] - 50.0_f64.sqrt()).abs() < 1e-13);
            assert!(decomposed.singular_values[1].abs() <= decomposed.band);
            let k = decomposed.singular_values.len();
            let rebuilt = decomposed.u.slice(s![.., ..k]).dot(&Array2::from_diag(&decomposed.singular_values)).dot(&decomposed.vt.slice(s![..k, ..]));
            assert!(close(&rebuilt, &a, 1e-13));
            assert_eq!(decomposed.u.dim(), if full { (3, 3) } else { (3, 2) });
            assert_eq!(decomposed.vt.dim(), (2, 2));
            for column in decomposed.u.columns() {
                assert!(canonical_sign(column.iter().copied()) > 0.0);
            }
            let orthogonal = decomposed.u.t().dot(&decomposed.u);
            assert!(close(&orthogonal, &Array2::eye(decomposed.u.ncols()), 1e-13));
        }
    }

    #[test]
    fn qr_has_a_nonnegative_diagonal_in_every_mode() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]];
        for mode in [QrMode::Economic, QrMode::Full, QrMode::R] {
            let decomposed = qr(a.view(), mode).expect("qr");
            let r = &decomposed.r;
            for index in 0..r.nrows().min(r.ncols()) {
                assert!(r[[index, index]] >= 0.0);
            }
            match (&decomposed.q, mode) {
                (None, QrMode::R) => assert_eq!(r.dim(), (2, 2)),
                (Some(q), QrMode::Economic) => {
                    assert_eq!(q.dim(), (3, 2));
                    assert!(close(&q.dot(r), &a, 1e-13));
                }
                (Some(q), QrMode::Full) => {
                    assert_eq!((q.dim(), r.dim()), ((3, 3), (3, 2)));
                    assert!(close(&q.dot(r), &a, 1e-13));
                    assert!(close(&q.t().dot(q), &Array2::eye(3), 1e-13));
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        // R alone is the economic R.
        assert_eq!(qr(a.view(), QrMode::R).expect("r").r, qr(a.view(), QrMode::Economic).expect("e").r);
    }

    #[test]
    fn solve_matches_a_known_answer_and_refuses_a_singular_system() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![[3.0], [5.0]];
        let x = solve(a.view(), b.view()).expect("solve");
        assert!(close(&x, &array![[0.8], [1.4]], 1e-14));
        assert_eq!(
            solve(array![[1.0, 2.0], [2.0, 4.0]].view(), b.view()),
            Err(DenseError::Singular { column: 1 })
        );
        assert!(matches!(eigh(array![[1.0, 2.0]].view(), SymmetricAssembly::Mirrored, None), Err(DenseError::NotSquare { .. })));
        assert!(matches!(svd(array![[f64::NAN]].view(), false), Err(DenseError::NonFinite { .. })));
        assert!(matches!(eigh(a.view(), SymmetricAssembly::Mirrored, Some((1, 1))), Err(DenseError::InvalidRange { .. })));
    }
}
