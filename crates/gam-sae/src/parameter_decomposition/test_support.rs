//! Shared test fixtures for `parameter_decomposition` (#2951).
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

/// A random residual MLP block with overcomplete reads on both weights, its
/// inputs and its mask kinds, shared by the tests of `rewrite` and
/// `rewrite_program`, so both run one seeded fixture.
pub mod component_mlp {
    use crate::parameter_decomposition::rewrite::{ComponentMlp, ComponentRead, NativeMlp};
    use gam_math::gaussian_activation::GaussianActivation;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    pub const WIDTH: usize = 4;
    pub const HIDDEN: usize = 6;
    pub const READ_IN_COMPONENTS: usize = 7;
    pub const WRITE_OUT_COMPONENTS: usize = 8;
    pub const ROWS: usize = 5;

    pub fn uniform(rng: &mut StdRng, rows: usize, cols: usize, half_width: f64) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-half_width..half_width))
    }

    pub fn uniform_vector(rng: &mut StdRng, len: usize, half_width: f64) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-half_width..half_width))
    }

    /// A random block with overcomplete reads on both weights, and its inputs.
    pub fn random_block(activation: GaussianActivation, seed: u64) -> (ComponentMlp, Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let native = NativeMlp::new(
            uniform(&mut rng, HIDDEN, WIDTH, 1.0),
            uniform_vector(&mut rng, HIDDEN, 1.0),
            uniform(&mut rng, WIDTH, HIDDEN, 1.0),
            uniform_vector(&mut rng, WIDTH, 1.0),
            activation,
        )
        .expect("the random block's shapes compose");
        let read_in = uniform(&mut rng, READ_IN_COMPONENTS, WIDTH, 1.0);
        let read_in_candidate = uniform(&mut rng, HIDDEN, READ_IN_COMPONENTS, 1.0);
        let write_out = uniform(&mut rng, WRITE_OUT_COMPONENTS, HIDDEN, 1.0);
        let write_out_candidate = uniform(&mut rng, WIDTH, WRITE_OUT_COMPONENTS, 1.0);
        let block = ComponentMlp::new(
            native,
            ComponentRead {
                read: read_in.view(),
                candidate_write: read_in_candidate.view(),
            },
            ComponentRead {
                read: write_out.view(),
                candidate_write: write_out_candidate.view(),
            },
        )
        .expect("a random overcomplete read is resolved");
        let inputs = uniform(&mut rng, ROWS, WIDTH, 2.0);
        (block, inputs)
    }

    /// Continuous, binary, signed and algebraically all-on masks.
    pub fn mask_family(rng: &mut StdRng, len: usize) -> [(&'static str, Array1<f64>); 4] {
        let continuous = Array1::from_shape_simple_fn(len, || rng.random_range(0.0..1.0));
        let binary = Array1::from_shape_fn(len, |component| (component % 2) as f64);
        let mut signed = Array1::from_shape_simple_fn(len, || rng.random_range(-2.0..2.0));
        signed[0] = -1.5;
        [
            ("continuous", continuous),
            ("binary", binary),
            ("signed", signed),
            ("ones", Array1::ones(len)),
        ]
    }

    pub fn bitwise_equal(left: &Array2<f64>, right: &Array2<f64>) -> bool {
        left.dim() == right.dim()
            && left
                .iter()
                .zip(right.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
