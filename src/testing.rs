//! Generic finite-difference testing utilities.

#[cfg(test)]
pub mod testing {
    use ndarray::Array2;

    /// Assert that a central difference of an array-producing function matches the analytical derivative.
    #[macro_export]
    macro_rules! assert_central_difference_array {
        ($x:expr, $h:expr, |$var:ident| $eval:expr, $analytical:expr, $tol:expr) => {
            let f_plus = {
                let $var = $x + $h;
                $eval
            };
            let f_minus = {
                let $var = $x - $h;
                $eval
            };
            assert_eq!(f_plus.len(), $analytical.len());
            for j in 0..$analytical.len() {
                let fd = (f_plus[j] - f_minus[j]) / (2.0 * $h);
                approx::assert_abs_diff_eq!(fd, $analytical[j], epsilon = $tol);
            }
        };
    }

    /// Asserts that a finite difference dense matrix closely matches an analytically computed 
    /// directional derivative matrix, both in tolerance and in component-wise sign.
    pub fn assert_matrix_derivative_fd(
        fd: &Array2<f64>,
        analytic: &Array2<f64>,
        tol: f64,
        label: &str,
    ) {
        assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
        for i in 0..analytic.nrows() {
            for j in 0..analytic.ncols() {
                assert_eq!(
                    analytic[[i, j]].signum(),
                    fd[[i, j]].signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic[[i, j]],
                    fd[[i, j]]
                );
                assert!(
                    (analytic[[i, j]] - fd[[i, j]]).abs() < tol,
                    "{} value mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic[[i, j]],
                    fd[[i, j]]
                );
            }
        }
    }
}
