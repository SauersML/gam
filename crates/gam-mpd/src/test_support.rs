//! Shared test fixtures for `gam_mpd` (#2951).
//!
//! Every tolerance a fixture hands to a test is derived from the fixture's own float
//! defects, so a test can assert against a planted truth without a constant.
#![cfg(test)]

use gam_linalg::faer_ndarray::{FaerEigh, FaerQr};
use gam_linalg::roundoff::{accumulation_growth, symmetric_spectrum_rounding_band};
use faer::Side;
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// The planted known-answer toys and their ball-arithmetic Jacobians.
pub mod planted_toys;

// Hand-built networks with a known decomposition: induction, modular addition, residual MLPs.
pub mod known_answer_toys;

pub fn frobenius_norm(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// A random orthonormal basis: Householder QR of a seeded uniform draw.
pub fn hidden_basis(dimension: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let draw =
        Array2::<f64>::from_shape_fn((dimension, dimension), |_| rng.random_range(-1.0..1.0));
    let (q, _) = draw.qr().expect("Householder QR of a uniform draw");
    q
}

/// Block-diagonal normal form: `[[c, -s], [s, c]]` per angle, then `negative_axes`
/// entries `-1`, then `+1`.
pub fn normal_form(dimension: usize, angles: &[f64], negative_axes: usize) -> Array2<f64> {
    let mut form = Array2::<f64>::eye(dimension);
    for (plane, &angle) in angles.iter().enumerate() {
        let (sine, cosine) = angle.sin_cos();
        let offset = 2 * plane;
        form[[offset, offset]] = cosine;
        form[[offset, offset + 1]] = -sine;
        form[[offset + 1, offset]] = sine;
        form[[offset + 1, offset + 1]] = cosine;
    }
    for axis in 0..negative_axes {
        let index = 2 * angles.len() + axis;
        form[[index, index]] = -1.0;
    }
    form
}

/// `||X - X_o||_2 = max |sigma_i - 1| <= ||X^T X - I||_F`, plus the rounding of the
/// Gram: `gamma_n` times each entry's absolute term sum.
pub fn orthogonality_bound(factor: &Array2<f64>) -> f64 {
    let columns = factor.ncols();
    let gram = factor.t().dot(factor) - Array2::<f64>::eye(columns);
    let absolute = factor.mapv(f64::abs);
    frobenius_norm(gram.view())
        + accumulation_growth(columns) * frobenius_norm(absolute.t().dot(&absolute).view())
}

/// The planted matrix `W = Q B Q^T` and bounds on how far its float factors are from
/// their exactly orthogonal polar factors `Q_o`, `B_o`. Plane `k` is
/// `basis.slice(s![.., 2k..2k + 2])`, then the `-1` axes, then the fixed space.
pub struct Planted {
    pub basis: Array2<f64>,
    pub matrix: Array2<f64>,
    /// Bound on `||Q - Q_o||_2`.
    pub basis_defect: f64,
    /// Bound on `||B - B_o||_2`.
    pub form_defect: f64,
    /// Bound on `||W - Q_o B_o Q_o^T||_2`: the error the planted matrix declares.
    pub matrix_defect: f64,
}

pub fn plant(dimension: usize, angles: &[f64], negative_axes: usize, seed: u64) -> Planted {
    let basis = hidden_basis(dimension, seed);
    let form = normal_form(dimension, angles, negative_axes);
    let matrix = basis.dot(&form).dot(&basis.t());
    let basis_defect = orthogonality_bound(&basis);
    let form_defect = orthogonality_bound(&form);
    let absolute_basis = basis.mapv(f64::abs);
    let formation = accumulation_growth(2 * dimension)
        * frobenius_norm(
            absolute_basis
                .dot(&form.mapv(f64::abs))
                .dot(&absolute_basis.t())
                .view(),
        );
    // ||Q B Q^T - Q_o B_o Q_o^T|| <= ||Q - Q_o|| ||B|| ||Q||
    //   + ||Q_o|| ||B - B_o|| ||Q|| + ||Q_o|| ||B_o|| ||Q - Q_o||.
    let matrix_defect = basis_defect * (1.0 + form_defect) * (1.0 + basis_defect)
        + form_defect * (1.0 + basis_defect)
        + basis_defect
        + formation;
    Planted {
        basis,
        matrix,
        basis_defect,
        form_defect,
        matrix_defect,
    }
}

/// `||V_a V_a^T - V_b V_b^T||_2` and the rounding band of that measurement.
pub fn projector_distance(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> (f64, f64) {
    let dimension = left.nrows();
    let left_projector = left.dot(&left.t());
    let right_projector = right.dot(&right.t());
    let mut difference = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..dimension {
        for col in 0..dimension {
            difference[[row, col]] = 0.5
                * ((left_projector[[row, col]] - right_projector[[row, col]])
                    + (left_projector[[col, row]] - right_projector[[col, row]]));
        }
    }
    let (values, _) = difference
        .eigh(Side::Lower)
        .expect("projector difference eigendecomposition");
    let distance = values.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let absolute_left = left.mapv(f64::abs);
    let absolute_right = right.mapv(f64::abs);
    let band = accumulation_growth(left.ncols())
        * frobenius_norm(absolute_left.dot(&absolute_left.t()).view())
        + accumulation_growth(right.ncols())
            * frobenius_norm(absolute_right.dot(&absolute_right.t()).view())
        + accumulation_growth(3)
            * (frobenius_norm(left_projector.view()) + frobenius_norm(right_projector.view()))
        + symmetric_spectrum_rounding_band(&values.to_vec());
    (distance, band)
}

/// The ledger a test's kernels reserve on when the test does not assert on
/// reservations: a private governor, so no test draws on or reads the process-wide
/// ledger. Its budget, `2^34` bytes, is far above any fixture's footprint and far
/// below what a planted refusal requests; a test that asserts on its reservations
/// builds its own `MemoryGovernor::with_budget_bytes` instead.
pub fn test_governor() -> &'static gam_runtime::resource::MemoryGovernor {
    static GOVERNOR: std::sync::OnceLock<gam_runtime::resource::MemoryGovernor> = std::sync::OnceLock::new();
    GOVERNOR.get_or_init(|| gam_runtime::resource::MemoryGovernor::with_budget_bytes(1 << 34))
}
