//! Sufficient-statistic caches for #1033 mechanism (a), the measure-jet
//! fixed-design case.
//!
//! This module is for single-scale-mode measure jets where `dX/dpsi == 0`: the
//! design matrix `X` is theta-invariant across the lambda/rho outer loop, while
//! the penalty and, for GLM PIRLS, the scalar working-weight diagonal `W` may
//! change. It is distinct from `GaussianFixedCache`, which covers only the
//! Gaussian+identity lane with constant `W`, and from `PsiGramTensor`, which
//! covers design-moving psi via Chebyshev expansions, #1033 mechanism (b).
//!
//! Invariant: n-row work from the measure-jet basis builder happens once per fit
//! at construction. Gaussian constant-`W` accessors are O(p^3) or cheaper and do
//! not re-touch the n design rows. The GLM changing-`W` lane keeps the fixed
//! rows cached and performs only the irreducible weighted contractions needed
//! when PIRLS weights move.

use gam_linalg::faer_ndarray::{fast_xt_diag_x, fast_xt_diag_y};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Gaussian / constant-`W` sufficient statistics for a fixed design.
///
/// This stores `X'WX`, `X'W(y - offset)`, and `(y - offset)'W(y - offset)` so
/// per-lambda assembly and RSS/evidence terms are n-free. It generalizes the
/// constant-design idea beyond the existing Gaussian+identity-only cache while
/// keeping the same fixed-`W` requirement for this lane.
pub struct FixedDesignGramCache {
    xtwx: Array2<f64>,
    xtwy: Array1<f64>,
    ywy: f64,
    n: usize,
    p: usize,
}

impl FixedDesignGramCache {
    /// Build fixed-design Gaussian sufficient statistics.
    ///
    /// The right-hand side is routed through `fast_xt_diag_y`, the same weighted
    /// contraction primitive used by the runtime recompute path.
    pub fn build(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        offset: Option<ArrayView1<'_, f64>>,
        weights: Option<ArrayView1<'_, f64>>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n {
            return Err(format!(
                "y length {} must match design row count {}",
                y.len(),
                n
            ));
        }
        if let Some(offset_values) = offset {
            if offset_values.len() != n {
                return Err(format!(
                    "offset length {} must match design row count {}",
                    offset_values.len(),
                    n
                ));
            }
        }
        if let Some(weight_values) = weights {
            if weight_values.len() != n {
                return Err(format!(
                    "weights length {} must match design row count {}",
                    weight_values.len(),
                    n
                ));
            }
            validate_nonnegative_finite_weights(weight_values)?;
        }
        validate_finite_vector("y", y)?;
        if let Some(offset_values) = offset {
            validate_finite_vector("offset", offset_values)?;
        }
        validate_finite_matrix("x", x)?;

        let r = match offset {
            Some(offset_values) => &y.to_owned() - &offset_values.to_owned(),
            None => y.to_owned(),
        };
        let w = match weights {
            Some(weight_values) => weight_values.to_owned(),
            None => Array1::ones(n),
        };
        let x_owned = x.to_owned();
        let xtwx = fast_xt_diag_x(&x_owned, &w);
        let r2 = r.view().insert_axis(ndarray::Axis(1));
        let xtwy_mat = fast_xt_diag_y(&x_owned, &w, &r2);
        let xtwy = xtwy_mat.column(0).to_owned();
        let ywy = weighted_sum_squares(w.view(), r.view());

        Ok(Self {
            xtwx,
            xtwy,
            ywy,
            n,
            p,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn p(&self) -> usize {
        self.p
    }

    pub fn xtwx(&self) -> ArrayView2<'_, f64> {
        self.xtwx.view()
    }

    pub fn xtwy(&self) -> ArrayView1<'_, f64> {
        self.xtwy.view()
    }

    pub fn ywy(&self) -> f64 {
        self.ywy
    }

    /// Assemble `X'WX + S` for the inner solver without revisiting design rows.
    pub fn penalized_normal_matrix(
        &self,
        penalty: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if penalty.nrows() != self.p || penalty.ncols() != self.p {
            return Err(format!(
                "penalty shape {}x{} must match {}x{}",
                penalty.nrows(),
                penalty.ncols(),
                self.p,
                self.p
            ));
        }
        let mut normal = self.xtwx.clone();
        normal += &penalty;
        Ok(normal)
    }

    /// Compute penalty-free weighted RSS from sufficient statistics.
    pub fn penalized_rss(&self, beta: ArrayView1<'_, f64>) -> Result<f64, String> {
        if beta.len() != self.p {
            return Err(format!(
                "beta length {} must match design column count {}",
                beta.len(),
                self.p
            ));
        }
        // Expanding (r - Xb)'W(r - Xb) gives ywy - 2 b'X'Wr + b'X'WXb.
        let gram_beta = self.xtwx.dot(&beta);
        let linear = beta.dot(&self.xtwy);
        let quadratic = beta.dot(&gram_beta);
        Ok(self.ywy - 2.0 * linear + quadratic)
    }
}

/// Cached fixed design rows for GLM / changing-`W` PIRLS trials.
///
/// This cache owns the theta-invariant `X` rows once. Each trial recomputes
/// `X'WX` and `X'Wz` because the scalar working weights and working response
/// genuinely move during PIRLS. The Gaussian constant-Gram trick does not apply
/// when `W` changes; the saved work is the expensive measure-jet basis/design
/// construction, not the unavoidable weighted contraction over fixed rows.
pub struct FixedDesignRowCache {
    x: Array2<f64>,
    n: usize,
    p: usize,
}

impl FixedDesignRowCache {
    /// Cache a finite, non-empty fixed design.
    pub fn build(x: ArrayView2<'_, f64>) -> Result<Self, String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(format!(
                "design must be non-empty, got shape {}x{}",
                x.nrows(),
                x.ncols()
            ));
        }
        validate_finite_matrix("x", x)?;
        let n = x.nrows();
        let p = x.ncols();
        Ok(Self {
            x: x.to_owned(),
            n,
            p,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn p(&self) -> usize {
        self.p
    }

    pub fn design(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    /// Recompute `X' diag(weights) X` over cached rows.
    ///
    /// This remains O(n p^2), the irreducible weighted contraction when `W`
    /// changes. It avoids rebuilding the n-row measure-jet design.
    pub fn xtwx(&self, weights: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.validate_changing_weights(weights)?;
        Ok(fast_xt_diag_x(&self.x, &weights))
    }

    /// Recompute `X' diag(weights) z` over cached rows for a PIRLS response.
    pub fn xtwz(
        &self,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        self.validate_changing_weights(weights)?;
        if z.len() != self.n {
            return Err(format!(
                "z length {} must match design row count {}",
                z.len(),
                self.n
            ));
        }
        validate_finite_vector("z", z)?;
        let z2 = z.insert_axis(ndarray::Axis(1));
        let xtwz_mat = fast_xt_diag_y(&self.x, &weights, &z2);
        Ok(xtwz_mat.column(0).to_owned())
    }

    fn validate_changing_weights(&self, weights: ArrayView1<'_, f64>) -> Result<(), String> {
        if weights.len() != self.n {
            return Err(format!(
                "weights length {} must match design row count {}",
                weights.len(),
                self.n
            ));
        }
        validate_finite_vector("weights", weights)
    }
}

fn validate_finite_matrix(name: &str, matrix: ArrayView2<'_, f64>) -> Result<(), String> {
    for ((row, col), value) in matrix.indexed_iter() {
        if !(*value).is_finite() {
            return Err(format!("{name}[{row},{col}] must be finite"));
        }
    }
    Ok(())
}

fn validate_finite_vector(name: &str, vector: ArrayView1<'_, f64>) -> Result<(), String> {
    for (index, value) in vector.iter().enumerate() {
        if !(*value).is_finite() {
            return Err(format!("{name}[{index}] must be finite"));
        }
    }
    Ok(())
}

fn validate_nonnegative_finite_weights(weights: ArrayView1<'_, f64>) -> Result<(), String> {
    for (index, weight) in weights.iter().enumerate() {
        if !(*weight).is_finite() {
            return Err(format!("weights[{index}] must be finite"));
        }
        if *weight < 0.0 {
            return Err(format!("weights[{index}] must be non-negative"));
        }
    }
    Ok(())
}

fn weighted_sum_squares(weights: ArrayView1<'_, f64>, values: ArrayView1<'_, f64>) -> f64 {
    weights
        .iter()
        .zip(values.iter())
        .map(|(weight, value)| *weight * *value * *value)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{FixedDesignGramCache, FixedDesignRowCache};
    use approx::assert_abs_diff_eq;
    use gam_linalg::faer_ndarray::fast_xt_diag_x;
    use ndarray::{Array1, Array2};

    fn deterministic_design(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            let row = i as f64 + 1.0;
            let col = j as f64 + 1.0;
            ((row * 0.17 + col * 0.31).sin()) + row * col * 0.002
        })
    }

    fn deterministic_response(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            let row = i as f64 + 1.0;
            (row * 0.23).cos() + row * 0.015
        })
    }

    fn deterministic_offset(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            let row = i as f64 + 1.0;
            0.2 * (row * 0.11).sin() - 0.01 * row
        })
    }

    fn deterministic_weights(n: usize, scale: f64) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            let row = i as f64 + 1.0;
            0.4 + scale * (1.0 + (row * 0.19).sin())
        })
    }

    fn naive_xtx(x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut out = Array2::zeros((p, p));
        for row in 0..n {
            for a in 0..p {
                for b in 0..p {
                    out[[a, b]] += x[[row, a]] * x[[row, b]];
                }
            }
        }
        out
    }

    fn naive_xtwy(x: &Array2<f64>, weights: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut out = Array1::zeros(p);
        for row in 0..n {
            for col in 0..p {
                out[col] += x[[row, col]] * weights[row] * r[row];
            }
        }
        out
    }

    fn naive_xtwz(x: &Array2<f64>, weights: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
        naive_xtwy(x, weights, z)
    }

    fn naive_ywy(weights: &Array1<f64>, r: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for row in 0..weights.len() {
            sum += weights[row] * r[row] * r[row];
        }
        sum
    }

    fn assert_matrix_close(actual: ndarray::ArrayView2<'_, f64>, expected: &Array2<f64>, eps: f64) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert_abs_diff_eq!(actual[[row, col]], expected[[row, col]], epsilon = eps);
            }
        }
    }

    fn assert_vector_close(actual: ndarray::ArrayView1<'_, f64>, expected: &Array1<f64>, eps: f64) {
        assert_eq!(actual.len(), expected.len());
        for index in 0..expected.len() {
            assert_abs_diff_eq!(actual[index], expected[index], epsilon = eps);
        }
    }

    #[test]
    fn gaussian_xtwx_matches_naive() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let y = deterministic_response(n);
        let cache = FixedDesignGramCache::build(x.view(), y.view(), None, None).unwrap();
        let naive = naive_xtx(&x);
        assert_matrix_close(cache.xtwx(), &naive, 1.0e-9);
    }

    #[test]
    fn gaussian_xtwy_and_ywy_match_naive() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let y = deterministic_response(n);
        let offset = deterministic_offset(n);
        let weights = deterministic_weights(n, 0.35);
        let r = &y - &offset;
        let cache = FixedDesignGramCache::build(
            x.view(),
            y.view(),
            Some(offset.view()),
            Some(weights.view()),
        )
        .unwrap();
        let expected_xtwy = naive_xtwy(&x, &weights, &r);
        let expected_ywy = naive_ywy(&weights, &r);
        assert_vector_close(cache.xtwy(), &expected_xtwy, 1.0e-9);
        assert_abs_diff_eq!(cache.ywy(), expected_ywy, epsilon = 1.0e-9);
    }

    #[test]
    fn penalized_rss_matches_direct_residual() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let y = deterministic_response(n);
        let offset = deterministic_offset(n);
        let weights = deterministic_weights(n, 0.21);
        let beta = Array1::from_vec(vec![0.4, -0.2, 0.15, 0.05]);
        let r = &y - &offset;
        let cache = FixedDesignGramCache::build(
            x.view(),
            y.view(),
            Some(offset.view()),
            Some(weights.view()),
        )
        .unwrap();
        let mut direct = 0.0;
        for row in 0..n {
            let mut fit = 0.0;
            for col in 0..p {
                fit += x[[row, col]] * beta[col];
            }
            let residual = r[row] - fit;
            direct += weights[row] * residual * residual;
        }
        let cached = cache.penalized_rss(beta.view()).unwrap();
        assert_abs_diff_eq!(cached, direct, epsilon = 1.0e-8);
    }

    #[test]
    fn penalized_normal_matrix_adds_penalty() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let y = deterministic_response(n);
        let cache = FixedDesignGramCache::build(x.view(), y.view(), None, None).unwrap();
        let penalty = Array2::from_shape_fn((p, p), |(row, col)| {
            if row == col {
                0.5 + row as f64 * 0.1
            } else {
                0.02 * (row + col) as f64
            }
        });
        let normal = cache.penalized_normal_matrix(penalty.view()).unwrap();
        for row in 0..p {
            for col in 0..p {
                let expected = cache.xtwx()[[row, col]] + penalty[[row, col]];
                assert_abs_diff_eq!(normal[[row, col]], expected, epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn row_cache_xtwx_matches_fresh_build_across_weights() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let cache = FixedDesignRowCache::build(x.view()).unwrap();
        let weight_sets = [
            deterministic_weights(n, 0.12),
            deterministic_weights(n, 0.27),
            deterministic_weights(n, 0.41),
        ];
        for weights in weight_sets.iter() {
            let cached = cache.xtwx(weights.view()).unwrap();
            let fresh = fast_xt_diag_x(&x, weights);
            assert_matrix_close(cached.view(), &fresh, 1.0e-12);
        }
    }

    #[test]
    fn row_cache_xtwz_matches_naive() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let weights = deterministic_weights(n, 0.33);
        let z = Array1::from_shape_fn(n, |i| {
            let row = i as f64 + 1.0;
            (row * 0.07).sin() + 0.03 * row
        });
        let cache = FixedDesignRowCache::build(x.view()).unwrap();
        let cached = cache.xtwz(weights.view(), z.view()).unwrap();
        let expected = naive_xtwz(&x, &weights, &z);
        assert_vector_close(cached.view(), &expected, 1.0e-9);
    }

    #[test]
    fn build_rejects_shape_mismatch() {
        let n = 40;
        let p = 4;
        let x = deterministic_design(n, p);
        let mismatched_y = deterministic_response(n - 1);
        assert!(FixedDesignGramCache::build(x.view(), mismatched_y.view(), None, None).is_err());

        let y = deterministic_response(n);
        let mut weights = deterministic_weights(n, 0.2);
        weights[3] = f64::NAN;
        assert!(
            FixedDesignGramCache::build(x.view(), y.view(), None, Some(weights.view())).is_err()
        );
    }
}
