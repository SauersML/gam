//! The radial-shell chart shared by the in-frame curved lane and the block
//! chart lane: every whitened row is projected onto the shell of the training
//! rows' mean radius, and its held-out fit is scored per row against the linear
//! reconstruction with the same squared error.

use ndarray::Array2;

/// Radial-shell chart: project each whitened row to the train mean radius shell.
pub(crate) fn radial_predict(train: &Array2<f64>, eval: &Array2<f64>) -> Array2<f64> {
    let d = train.ncols();
    let mut radius = 0.0;
    for i in 0..train.nrows() {
        let mut ss = 0.0;
        for j in 0..d {
            ss += train[[i, j]] * train[[i, j]];
        }
        radius += ss.sqrt();
    }
    radius /= train.nrows().max(1) as f64;
    let mut out = Array2::<f64>::zeros(eval.dim());
    for i in 0..eval.nrows() {
        let mut norm = 0.0;
        for j in 0..d {
            norm += eval[[i, j]] * eval[[i, j]];
        }
        let norm = norm.sqrt();
        // A zero row has no direction to project onto the shell; it stays zero.
        if norm > 0.0 {
            for j in 0..d {
                out[[i, j]] = radius * eval[[i, j]] / norm;
            }
        }
    }
    out
}

/// Squared error between row `row` of `a` and of `b`.
pub(crate) fn row_sse(a: &Array2<f64>, b: &Array2<f64>, row: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        let d = a[[row, j]] - b[[row, j]];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every nonzero row lands on the shell, however small; a zero row has no
    /// direction and stays zero. A norm floor used to leave the small row far
    /// inside the shell.
    #[test]
    fn radial_predict_projects_small_rows_onto_the_shell() {
        let train = ndarray::array![[3.0, 4.0]];
        let tiny = 2.0_f64.powi(-60);
        let eval = ndarray::array![[3.0 * tiny, 4.0 * tiny], [0.0, 0.0]];
        let out = radial_predict(&train, &eval);
        assert_eq!(out[[0, 0]], 3.0);
        assert_eq!(out[[0, 1]], 4.0);
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[1, 1]], 0.0);
    }
}
