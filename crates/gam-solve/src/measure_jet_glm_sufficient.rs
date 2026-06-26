//! GLM fixed-design sufficient-statistic reuse for #1033 mechanism (a),
//! extending the n-free lever from the Gaussian lane to non-Gaussian families.
//!
//! Scope: single-scale-mode measure jets where `dX/dpsi == 0`, i.e. the design matrix
//! `X` is theta-invariant across the lambda/rho outer loop AND across the inner
//! IRLS/PIRLS iterations. For non-Gaussian families the scalar working-weight
//! diagonal `W` (and the working response `z`) genuinely change every IRLS step,
//! so the Gaussian constant-Gram cache in `FixedDesignGramCache` does not apply.
//! What is reusable is the design itself.
//!
//! This module owns the fixed `X` rows once and exposes the two weighted
//! contractions the GLM normal equations need each iteration:
//!
//!   * `weighted_gram(w)`     = X' diag(w) X        (the IRLS Hessian block)
//!   * `weighted_xty(w, z)`   = X' diag(w) z        (the IRLS working RHS)
//!
//! These match the PIRLS semantics formed in `solver::reml::assembly`
//! (`xt_diag_x_dense_into`) and `linalg::faer_ndarray` (`fast_xt_diag_x`,
//! `fast_xt_diag_y`), routing through the very same weighted-contraction
//! primitives so values agree bit-for-bit with the runtime recompute path.
//!
//! What is SAVED across trials: the O(n p) construction of the n-row measure-jet
//! design. The n-row basis kernel is evaluated exactly once, at construction,
//! and never again as the outer lambda loop sweeps or as IRLS reweights. The
//! stored rows are immutable; `assert_design_unchanged` lets callers (and the
//! oracle tests) prove the cache never touches the n-row design on a query.
//!
//! What is NOT saved (and cannot be, when `W` moves): the O(n p^2) weighted
//! contraction `X' diag(w) X` and the O(n p) `X' diag(w) z`. Those are the
//! irreducible IRLS work and run every iteration over the cached rows.
//!
//! This is distinct from `measure_jet_gram_cache::FixedDesignRowCache`, which
//! exposes recompute accessors without the byte-stability invariant; this module
//! makes the n-free-across-trials guarantee a first-class, testable property by
//! fingerprinting the stored design and proving it is never mutated by a query.

use gam_linalg::faer_ndarray::{fast_xt_diag_x, fast_xt_diag_y};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// GLM fixed-design sufficient-statistic provider.
///
/// Holds the theta-invariant design `X` (n x p) once. Every IRLS iteration and
/// every outer lambda trial reuses these stored rows for the weighted
/// contractions, so the n-row measure-jet basis is built exactly once per fit.
pub struct GlmFixedDesignSufficient {
    x: Array2<f64>,
    n: usize,
    p: usize,
    /// Order-insensitive fingerprint of the stored design bytes, captured at
    /// construction. A query that respects the n-free-across-trials invariant
    /// must leave this fingerprint unchanged.
    design_fingerprint: u64,
}

impl GlmFixedDesignSufficient {
    /// Cache a finite, non-empty fixed design.
    ///
    /// The n-row work happens here, once. Subsequent `weighted_gram` /
    /// `weighted_xty` calls never rebuild the design.
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
        let x_owned = x.to_owned();
        let design_fingerprint = fingerprint_matrix(x_owned.view());
        Ok(Self {
            x: x_owned,
            n,
            p,
            design_fingerprint,
        })
    }

    /// Row count `n` of the fixed design.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Column count `p` of the fixed design.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Borrow the stored, immutable design rows.
    pub fn design(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    /// Weighted Gram `X' diag(w) X` over the stored rows.
    ///
    /// This is the GLM IRLS Hessian block. It is recomputed each iteration
    /// because `w` moves, but it reuses the cached design: the O(n p^2)
    /// contraction runs, while the n-row design construction does not repeat.
    /// Routed through `fast_xt_diag_x` to match the runtime PIRLS recompute path
    /// exactly.
    pub fn weighted_gram(&self, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.validate_weights(w)?;
        Ok(fast_xt_diag_x(&self.x, &w))
    }

    /// Weighted cross product `X' diag(w) z` over the stored rows.
    ///
    /// This is the GLM IRLS working right-hand side for working response `z`.
    /// Recomputed each iteration (`w` and `z` move) but over the cached design.
    /// Routed through `fast_xt_diag_y` to match the runtime PIRLS recompute path.
    pub fn weighted_xty(
        &self,
        w: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        self.validate_weights(w)?;
        if z.len() != self.n {
            return Err(format!(
                "z length {} must match design row count {}",
                z.len(),
                self.n
            ));
        }
        validate_finite_vector("z", z)?;
        let z2 = z.insert_axis(ndarray::Axis(1));
        let xtwz_mat = fast_xt_diag_y(&self.x, &w, &z2);
        Ok(xtwz_mat.column(0).to_owned())
    }

    /// Confirm the stored design bytes are unchanged since construction.
    ///
    /// This is the n-free-across-trials invariant: a `weighted_gram` /
    /// `weighted_xty` query must never touch the n-row design. Returns an error
    /// if the recomputed fingerprint differs from the one captured at build.
    pub fn assert_design_unchanged(&self) -> Result<(), String> {
        let current = fingerprint_matrix(self.x.view());
        if current != self.design_fingerprint {
            return Err(format!(
                "stored design fingerprint changed: built {} now {}",
                self.design_fingerprint, current
            ));
        }
        Ok(())
    }

    fn validate_weights(&self, weights: ArrayView1<'_, f64>) -> Result<(), String> {
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

/// Order-sensitive bit fingerprint of a dense matrix.
///
/// Mixes each entry's raw IEEE-754 bits with its position so a permutation or
/// any single-bit change of the stored design is detected. Used to prove the
/// cache never mutates the n-row design on a query.
fn fingerprint_matrix(matrix: ArrayView2<'_, f64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let (nrows, ncols) = matrix.dim();
    hash = mix(hash, nrows as u64);
    hash = mix(hash, ncols as u64);
    for ((row, col), value) in matrix.indexed_iter() {
        hash = mix(hash, row as u64);
        hash = mix(hash, col as u64);
        hash = mix(hash, value.to_bits());
    }
    hash
}

/// FNV-1a style 64-bit mixing step.
fn mix(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    hash
}

#[cfg(test)]
mod tests {
    use super::GlmFixedDesignSufficient;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn deterministic_design(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            let row = i as f64 + 1.0;
            let col = j as f64 + 1.0;
            ((row * 0.17 + col * 0.31).sin()) + row * col * 0.002
        })
    }

    /// Seeded, strictly positive working weights (GLM IRLS weights are positive).
    fn seeded_weights(n: usize, rng: &mut StdRng) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| 0.05 + rng.random::<f64>())
    }

    fn seeded_working_response(n: usize, rng: &mut StdRng) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| rng.random::<f64>() * 2.0 - 1.0)
    }

    fn naive_weighted_gram(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut out = Array2::zeros((p, p));
        for row in 0..n {
            let wi = w[row];
            for a in 0..p {
                for b in 0..p {
                    out[[a, b]] += x[[row, a]] * wi * x[[row, b]];
                }
            }
        }
        out
    }

    fn naive_weighted_xty(x: &Array2<f64>, w: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut out = Array1::zeros(p);
        for row in 0..n {
            let scaled = w[row] * z[row];
            for col in 0..p {
                out[col] += x[[row, col]] * scaled;
            }
        }
        out
    }

    fn assert_matrix_close(actual: ndarray::ArrayView2<'_, f64>, expected: &Array2<f64>, eps: f64) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                let diff = (actual[[row, col]] - expected[[row, col]]).abs();
                assert!(
                    diff <= eps,
                    "matrix mismatch at [{row},{col}]: {} vs {} (diff {diff})",
                    actual[[row, col]],
                    expected[[row, col]]
                );
            }
        }
    }

    fn assert_vector_close(actual: ndarray::ArrayView1<'_, f64>, expected: &Array1<f64>, eps: f64) {
        assert_eq!(actual.len(), expected.len());
        for index in 0..expected.len() {
            let diff = (actual[index] - expected[index]).abs();
            assert!(
                diff <= eps,
                "vector mismatch at [{index}]: {} vs {} (diff {diff})",
                actual[index],
                expected[index]
            );
        }
    }

    /// `weighted_gram` matches a from-scratch X' diag(w) X to 1e-12 across
    /// several seeded working-weight vectors.
    #[test]
    fn weighted_gram_matches_naive_across_seeded_weights() {
        let n = 48;
        let p = 5;
        let x = deterministic_design(n, p);
        let cache = GlmFixedDesignSufficient::build(x.view()).unwrap();
        let mut rng = StdRng::seed_from_u64(0x1033_0001);
        for _ in 0..6 {
            let w = seeded_weights(n, &mut rng);
            let cached = cache.weighted_gram(w.view()).unwrap();
            let naive = naive_weighted_gram(&x, &w);
            assert_matrix_close(cached.view(), &naive, 1.0e-12);
        }
    }

    /// `weighted_xty` matches a from-scratch X' diag(w) z to 1e-12 across
    /// several seeded weight/response pairs.
    #[test]
    fn weighted_xty_matches_naive_across_seeded_weights() {
        let n = 48;
        let p = 5;
        let x = deterministic_design(n, p);
        let cache = GlmFixedDesignSufficient::build(x.view()).unwrap();
        let mut rng = StdRng::seed_from_u64(0x1033_0002);
        for _ in 0..6 {
            let w = seeded_weights(n, &mut rng);
            let z = seeded_working_response(n, &mut rng);
            let cached = cache.weighted_xty(w.view(), z.view()).unwrap();
            let naive = naive_weighted_xty(&x, &w, &z);
            assert_vector_close(cached.view(), &naive, 1.0e-12);
        }
    }

    /// The n-free-across-trials invariant: re-querying with a DIFFERENT `w`
    /// reuses the same stored design. The stored design bytes are unchanged, so
    /// the captured fingerprint still matches and the borrowed rows are
    /// element-for-element identical to the original input across all queries.
    #[test]
    fn stored_design_unchanged_across_different_weights() {
        let n = 48;
        let p = 5;
        let x = deterministic_design(n, p);
        let cache = GlmFixedDesignSufficient::build(x.view()).unwrap();

        // Snapshot of the stored design before any query.
        let baseline = cache.design().to_owned();
        cache.assert_design_unchanged().unwrap();

        let mut rng = StdRng::seed_from_u64(0x1033_0003);
        let mut last_gram: Option<Array2<f64>> = None;
        for _ in 0..5 {
            let w = seeded_weights(n, &mut rng);
            let z = seeded_working_response(n, &mut rng);

            // Different W each pass exercises the changing-weight lane. The RHS
            // query is exercised too (its result is checked for length here so
            // the call is not a discarded computation).
            let gram = cache.weighted_gram(w.view()).unwrap();
            let rhs = cache.weighted_xty(w.view(), z.view()).unwrap();
            assert_eq!(rhs.len(), p);

            // The stored design must not have been mutated by the query.
            cache.assert_design_unchanged().unwrap();
            let current = cache.design();
            assert_eq!(current.dim(), baseline.dim());
            for row in 0..n {
                for col in 0..p {
                    assert_eq!(current[[row, col]], baseline[[row, col]]);
                }
            }

            // A genuinely different W yields a genuinely different Gram, proving
            // the reuse is of the design, not of a stale cached result.
            if let Some(prev) = last_gram.as_ref() {
                let mut any_diff = false;
                for a in 0..p {
                    for b in 0..p {
                        if (gram[[a, b]] - prev[[a, b]]).abs() > 1.0e-9 {
                            any_diff = true;
                        }
                    }
                }
                assert!(any_diff, "distinct weights must yield distinct Gram");
            }
            last_gram = Some(gram);
        }

        // After the full sweep the design is still byte-identical to the input.
        let final_design = cache.design();
        for row in 0..n {
            for col in 0..p {
                assert_eq!(final_design[[row, col]], x[[row, col]]);
            }
        }
    }

    #[test]
    fn build_rejects_empty_and_nonfinite() {
        let empty = Array2::<f64>::zeros((0, 3));
        assert!(GlmFixedDesignSufficient::build(empty.view()).is_err());

        let mut x = deterministic_design(10, 3);
        x[[4, 1]] = f64::NAN;
        assert!(GlmFixedDesignSufficient::build(x.view()).is_err());
    }

    #[test]
    fn query_rejects_shape_and_nonfinite_inputs() {
        let n = 12;
        let p = 3;
        let x = deterministic_design(n, p);
        let cache = GlmFixedDesignSufficient::build(x.view()).unwrap();

        let short_w = Array1::<f64>::ones(n - 1);
        assert!(cache.weighted_gram(short_w.view()).is_err());

        let mut bad_w = Array1::<f64>::ones(n);
        bad_w[2] = f64::INFINITY;
        assert!(cache.weighted_gram(bad_w.view()).is_err());

        let good_w = Array1::<f64>::ones(n);
        let short_z = Array1::<f64>::ones(n - 1);
        assert!(cache.weighted_xty(good_w.view(), short_z.view()).is_err());
    }
}
