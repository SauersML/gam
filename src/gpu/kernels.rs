use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// CPU reference implementation of the row scaling kernel used by the CUDA
/// path. Keeping this here makes validation deterministic and gives the GPU
/// module a single semantic definition of weighted design chunks.
#[must_use]
pub fn scale_rows_reference<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Array2<f64> {
    let (n, p) = x.dim();
    debug_assert_eq!(n, w.len());
    Array2::from_shape_fn((n, p), |(i, j)| x[[i, j]] * w[i])
}

#[must_use]
pub fn weighted_response_reference<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    w: &ArrayBase<S1, Ix1>,
    y: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    debug_assert_eq!(w.len(), y.len());
    Array1::from_iter(w.iter().zip(y.iter()).map(|(wi, yi)| wi * yi))
}
