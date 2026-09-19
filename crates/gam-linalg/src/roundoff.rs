//! Derived floating-point roundoff bands.
//!
//! Numerical code is full of the question "is this quantity indistinguishable
//! from zero?" — a negative eigenvalue that should be a zero one, a residual
//! that should be a converged one, a directional derivative that should be a
//! flat one. The answer is not a taste parameter: it is the backward-error band
//! of the arithmetic that produced the quantity, and that band is a function of
//! how many operations were accumulated and how large the accumulated terms
//! were. Both are known at the call site.
//!
//! The bound is Wilkinson's, in the form given by Higham (*Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Lemma 3.1 and §3.1):
//!
//! ```text
//! |fl(s) − s|  ≤  γ_k · Σ|terms|,      γ_k = k·u / (1 − k·u),
//! ```
//!
//! with `u` the unit roundoff and `k` the number of rounded operations on an
//! accumulation path. Writing a bare multiple of `EPSILON` instead substitutes
//! a constant for `γ_k`, which is wrong in both directions as the problem size
//! moves — too tight for long accumulations, so exact-arithmetic-zero
//! quantities get rejected as materially nonzero, and needlessly loose for
//! short ones.
//!
//! # `k` is not always `n`
//!
//! The two accumulations that look alike carry different depths, and conflating
//! them is the easiest way to get a bound that is wrong by one factor of `u`:
//!
//! * An **inner product** of length `n` forms `n` products and then sums them
//!   with `n − 1` additions, so `k = n` — this is [`accumulation_band`].
//! * A **sum of terms that are already formed** commits only the `n − 1`
//!   additions, so `k = n − 1`: a caller holding `Σ|xᵢ|` and the term count
//!   passes `n - 1` to [`accumulation_growth`] directly rather than using
//!   [`accumulation_band`].
//! * A **pairwise (tree) reduction** has depth `⌈log₂ n⌉` rather than `n − 1`;
//!   that is a genuinely tighter bound and the right `k` whenever the
//!   reduction is pairwise.
//! * A **compensated (Kahan/Neumaier) sum** has a bound with no `n` in it at
//!   all — [`compensated_band`].

// Owned by gam-math, the lowest crate, so the math crate's bounds read the same definitions.
pub use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth};

/// Backward-error band of an inner product of length `terms` whose summands
/// have absolute sum `absolute_sum`: `γ_terms · absolute_sum`.
///
/// This is the magnitude below which the computed result is indistinguishable
/// from zero: an inner product whose exact value is zero can be computed as
/// anything within `±accumulation_band(n, Σ|xᵢyᵢ|)`, and nothing outside it.
///
/// The depth is `terms`, not `terms − 1`, because forming each product rounds
/// once before any addition happens. For a sum of pre-formed terms see the
/// module documentation.
pub fn accumulation_band(terms: usize, absolute_sum: f64) -> f64 {
    accumulation_growth(terms) * absolute_sum
}

/// Rounding band `p·ε·‖H‖₂` of a symmetric `p×p` matrix's computed spectrum,
/// read off its eigenvalues.
///
/// A backward-stable symmetric eigensolver returns the exact spectrum of `H + E`
/// with `‖E‖₂ = O(p·ε·‖H‖₂)`, so by Weyl an eigenvalue whose magnitude is at or
/// below this band is not resolved from zero by the decomposition that produced
/// it, and a sign inside it is not a measurement.
pub fn symmetric_spectrum_rounding_band(eigenvalues: &[f64]) -> f64 {
    let spectral_radius = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    eigenvalues.len() as f64 * f64::EPSILON * spectral_radius
}

/// The rank of a symmetric Gram, read off its eigenvalues: those above
/// [`symmetric_spectrum_rounding_band`] plus `assembly_band`.
///
/// `assembly_band` is the caller's bound, in eigenvalue units, on the error the
/// Gram's formation left in the matrix (zero for exact input). A frozen rank and
/// every later trial of the same block read this one predicate. An eigenvalue
/// inside the band is not resolved from zero by the decomposition that produced
/// it, and its sign carries no information.
pub fn resolved_eigenvalue_count(eigenvalues: &[f64], assembly_band: f64) -> usize {
    let band = symmetric_spectrum_rounding_band(eigenvalues) + assembly_band;
    eigenvalues.iter().filter(|&&value| value > band).count()
}

/// Forward-error band of a **compensated** summation (Kahan–Babuška–Neumaier)
/// whose summands have absolute sum `absolute_sum`, where building each summand
/// cost `formation_roundings` floating-point operations.
///
/// Compensated summation satisfies `|fl(S) − S| ≤ 2u·Σ|xᵢ|` **with no
/// dependence on the term count** (Higham, *ASNA* 2nd ed., §4.3) — shedding the
/// `n` is the entire reason to pay for the compensation. Scaling a compensated
/// sum's band by `n` anyway applies the naive-summation model to an algorithm
/// chosen to defeat it, and inflates the band by a factor of `n`.
///
/// `formation_roundings` covers the arithmetic that produces each summand
/// before any addition happens: an inner-product term formed as `(a/p)*(b/q)`
/// costs three. Count the operations at the call site, where they are visible.
pub fn compensated_band(formation_roundings: usize, absolute_sum: f64) -> f64 {
    (2.0 + formation_roundings as f64) * UNIT_ROUNDOFF * absolute_sum
}

/// Backward-error band on the singular values of an `m × n` factor with largest
/// singular value `sigma_max`.
///
/// A backward-stable SVD returns the exact singular values of `A + δA` with
/// `‖δA‖₂ ≤ max(m, n)·ε·‖A‖₂`, and by Weyl each computed `σᵢ` is within that of
/// the true one. A singular value above the band is resolved. At or below it the
/// decomposition has not separated it from zero.
pub fn factor_singular_band(rows: usize, cols: usize, sigma_max: f64) -> f64 {
    rows.max(cols) as f64 * f64::EPSILON * sigma_max
}

/// Rank partition of a quadratic `S = AᵀA`, read off its energy factor `A`.
///
/// `λᵢ(S) = σᵢ(A)²`, so `A` carries `S`'s spectrum at `A`'s own conditioning, and
/// the rank counts the singular values above [`factor_singular_band`]. Forming
/// `AᵀA` first squares the conditioning: a mode with `σᵢ/σ₁ = 1e-10` is resolved
/// here, but its eigenvalue lies below the Gram's rounding band, where it comes
/// back as roundoff of either sign.
pub struct FactorRankPartition {
    /// Singular values of `A`, descending.
    pub singular_values: Vec<f64>,
    /// The right singular vectors as rows, in the same order (`n × n`).
    pub right_vectors: ndarray::Array2<f64>,
    /// Number of singular values above the backward-error band.
    pub rank: usize,
}

impl FactorRankPartition {
    /// The root rows `σᵢ·vᵢᵀ` for the leading `rank` modes, so `RᵀR` equals `S`
    /// on the resolved range.
    pub fn root_rows(&self, rank: usize) -> ndarray::Array2<f64> {
        let mut root = self.right_vectors.slice(ndarray::s![..rank, ..]).to_owned();
        for (mut row, &sigma) in root.rows_mut().into_iter().zip(self.singular_values.iter()) {
            row.mapv_inplace(|value| value * sigma);
        }
        root
    }
}

/// Partition `S = AᵀA` by the singular values of `A` (see [`FactorRankPartition`]).
pub fn factor_rank_partition(
    factor: &ndarray::Array2<f64>,
) -> Result<FactorRankPartition, crate::faer_ndarray::FaerLinalgError> {
    use crate::faer_ndarray::FaerSvd;
    let (rows, cols) = factor.dim();
    // A thin SVD of an `m × n` factor with `m < n` omits part of the null space
    // of `AᵀA`. Zero rows add no energy and complete the right singular basis.
    let completed;
    let square = if rows < cols {
        let mut padded = ndarray::Array2::<f64>::zeros((cols, cols));
        padded.slice_mut(ndarray::s![..rows, ..]).assign(factor);
        completed = padded;
        &completed
    } else {
        factor
    };
    let (_, sigma, vt) = square.svd(false, true)?;
    let vt = vt.ok_or(crate::faer_ndarray::FaerLinalgError::SvdNoConvergence {
        context: "factor_rank_partition: right singular vectors",
    })?;
    let mut order: Vec<usize> = (0..sigma.len()).collect();
    order.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    let singular_values: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let mut right_vectors = ndarray::Array2::<f64>::zeros((order.len(), cols));
    for (row, &i) in order.iter().enumerate() {
        right_vectors.row_mut(row).assign(&vt.row(i));
    }
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let band = factor_singular_band(rows, cols, sigma_max);
    let rank = singular_values.iter().filter(|&&value| value > band).count();
    Ok(FactorRankPartition {
        singular_values,
        right_vectors,
        rank,
    })
}

/// Forward-error band of a penalty trace `t = λ·Σ_c r_cᵀ H⁻¹ r_c` for a symmetric
/// `H`, computed as `t̂ = λ·Σ_c r_cᵀ x̂_c` from solved columns `x̂_c ≈ H⁻¹ r_c`
/// (#2901).
///
/// With the true residual `ρ_c = H·x̂_c − r_c`, exactly `x_c = x̂_c − H⁻¹ρ_c`, so
///
/// ```text
///   t − t̂ = −λ·Σ_c r_cᵀ H⁻¹ ρ_c = −λ·Σ_c x_cᵀ ρ_c,
///   |t − t̂| ≤ λ·Σ_c ‖ρ_c‖₂·(‖x̂_c‖₂ + ‖H⁻¹‖₂·‖ρ_c‖₂).
/// ```
///
/// - **Residual.** The computed residual `residual` differs from `ρ_c` by its own
///   formation. Each entry is an inner product of `rows + 1` terms bounded by
///   `rows·‖H‖_max·‖x̂_c‖_max + |r_ic|`, so `‖ρ_c‖₂` is charged the computed norm
///   plus that band.
/// - **Inverse norm.** For symmetric `H`, `‖H⁻¹‖₂ ≤ ‖H⁻¹‖₁`. That norm is
///   `inverse_one_norm_estimate`, the Hager–Higham lower-bound estimate
///   ([`crate::condition::estimate_inverse_one_norm`]), so the second-order term
///   is an estimate, not a certificate.
/// - **Formation of `t̂`.** Forming `t̂` is an inner product of `rows·columns`
///   terms, which adds `accumulation_band(rows·columns, λ·Σ|r_ic·x̂_ic|)`.
///
/// The band measures the solve that produced the trace. A trace published
/// outside `[0, rank]` by more than this band is not rounding.
pub fn solved_penalty_trace_band(
    lambda: f64,
    rhs: ndarray::ArrayView2<'_, f64>,
    solution: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView2<'_, f64>,
    matrix_max_abs: f64,
    inverse_one_norm_estimate: f64,
) -> Result<f64, String> {
    let (rows, columns) = rhs.dim();
    if solution.dim() != (rows, columns) || residual.dim() != (rows, columns) {
        return Err(format!(
            "solved_penalty_trace_band: a {rows}x{columns} right-hand side against a {}x{} solution \
             and a {}x{} residual",
            solution.nrows(),
            solution.ncols(),
            residual.nrows(),
            residual.ncols()
        ));
    }
    let residual_growth = accumulation_growth(rows + 1);
    let mut solve_band = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for column in 0..columns {
        let solution_max = solution
            .column(column)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let product_bound = rows as f64 * matrix_max_abs * solution_max;
        let mut solution_norm_sq = 0.0_f64;
        let mut residual_norm_sq = 0.0_f64;
        let mut formation_norm_sq = 0.0_f64;
        for row in 0..rows {
            let x = solution[[row, column]];
            let r = rhs[[row, column]];
            solution_norm_sq += x * x;
            residual_norm_sq += residual[[row, column]] * residual[[row, column]];
            let formation = residual_growth * (product_bound + r.abs());
            formation_norm_sq += formation * formation;
            absolute_sum += (r * x).abs();
        }
        let charged_residual = residual_norm_sq.sqrt() + formation_norm_sq.sqrt();
        solve_band +=
            charged_residual * (solution_norm_sq.sqrt() + inverse_one_norm_estimate * charged_residual);
    }
    Ok(lambda.abs() * (solve_band + accumulation_growth(rows * columns) * absolute_sum))
}

/// The trace `t̂ = λ·Σ_c r_cᵀ x̂_c` of solved penalty-root columns, with its
/// [`solved_penalty_trace_band`] (#2901).
///
/// `rhs` holds the columns `r_c` a route solved against `operator`, and `solution`
/// the computed `x̂_c`. For a pseudoinverse solve, `rhs` is the root projected onto
/// the operator's range, the right-hand side that solve reproduces. The true
/// residual `operator·x̂_c − r_c` is formed here, so every route prices its trace
/// against the operator its solve represents.
pub fn solved_penalty_trace(
    lambda: f64,
    rhs: ndarray::ArrayView2<'_, f64>,
    solution: ndarray::ArrayView2<'_, f64>,
    operator: ndarray::ArrayView2<'_, f64>,
    inverse_one_norm_estimate: f64,
) -> Result<(f64, f64), String> {
    if operator.nrows() != rhs.nrows() || operator.ncols() != solution.nrows() {
        return Err(format!(
            "solved_penalty_trace: a {}x{} operator against a {}x{} right-hand side and a {}x{} \
             solution",
            operator.nrows(),
            operator.ncols(),
            rhs.nrows(),
            rhs.ncols(),
            solution.nrows(),
            solution.ncols()
        ));
    }
    let residual = operator.dot(&solution) - &rhs;
    let matrix_max_abs = operator
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let band = solved_penalty_trace_band(
        lambda,
        rhs,
        solution,
        residual.view(),
        matrix_max_abs,
        inverse_one_norm_estimate,
    )?;
    let trace: f64 = rhs.iter().zip(solution.iter()).map(|(r, x)| r * x).sum();
    Ok((lambda * trace, band))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composed trace is `λ·Σ r·x̂`, and its band covers the error of a
    /// displaced solution through the residual it forms against the operator.
    #[test]
    fn a_solved_penalty_trace_prices_its_solution_against_the_operator_2901() {
        let operator = ndarray::array![[2.0, 0.0], [0.0, 4.0]];
        let rhs = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let exact = ndarray::array![[0.5, 0.0], [0.0, 0.25]];
        let (trace, band) =
            solved_penalty_trace(3.0, rhs.view(), exact.view(), operator.view(), 0.5)
                .expect("exact solve");
        assert_eq!(trace, 2.25);
        let displaced = ndarray::array![[0.501, 0.0], [0.0, 0.25]];
        let (displaced_trace, displaced_band) =
            solved_penalty_trace(3.0, rhs.view(), displaced.view(), operator.view(), 0.5)
                .expect("displaced solve");
        let error = (displaced_trace - trace).abs();
        assert!(
            error <= displaced_band,
            "displaced trace error {error:e} escapes its band {displaced_band:e}"
        );
        assert!(
            error > band,
            "the exact solve's band {band:e} must not cover a displacement of {error:e}"
        );
    }

    /// #2901: a solve left inexact by a known perturbation moves the trace by
    /// `−λ·Σ x_cᵀρ_c`, and the band carries it. `H = diag(4, 0.5)`, `r = (1, 1)`,
    /// `λ = 3`: the exact trace is `3·(1/4 + 2) = 6.75`. A solution off by
    /// `δ = (1e-3, −2e-3)` has residual `Hδ` and moves the trace by `3·(1e-3 − 2e-3)`.
    /// The control passes a zero residual and shows the residual term is what
    /// covers the error.
    #[test]
    fn a_solved_trace_band_covers_the_error_its_residual_carries_2901() {
        let lambda = 3.0;
        let rhs = ndarray::array![[1.0], [1.0]];
        let exact = ndarray::array![[0.25], [2.0]];
        let offset = ndarray::array![[1.0e-3], [-2.0e-3]];
        let solution = &exact + &offset;
        let residual = ndarray::array![[4.0 * 1.0e-3], [0.5 * -2.0e-3]];
        let exact_trace = lambda * (0.25 + 2.0);
        let computed_trace = lambda * (solution[[0, 0]] + solution[[1, 0]]);
        let inverse_one_norm = 2.0;
        let band = solved_penalty_trace_band(
            lambda,
            rhs.view(),
            solution.view(),
            residual.view(),
            4.0,
            inverse_one_norm,
        )
        .unwrap();
        assert!(
            (computed_trace - exact_trace).abs() <= band,
            "the trace error {:.4e} escaped its band {band:.4e}",
            (computed_trace - exact_trace).abs()
        );
        let unmeasured = solved_penalty_trace_band(
            lambda,
            rhs.view(),
            solution.view(),
            ndarray::Array2::<f64>::zeros((2, 1)).view(),
            4.0,
            inverse_one_norm,
        )
        .unwrap();
        assert!(
            (computed_trace - exact_trace).abs() > unmeasured,
            "without the residual the band {unmeasured:.4e} must not cover {:.4e}",
            (computed_trace - exact_trace).abs()
        );
        assert!(
            solved_penalty_trace_band(
                lambda,
                rhs.view(),
                solution.view(),
                ndarray::Array2::<f64>::zeros((3, 1)).view(),
                4.0,
                inverse_one_norm
            )
            .is_err()
        );
    }

    /// One Gram-side predicate: eigenvalues above the eigensolver band plus the
    /// caller's assembly band are resolved, and roundoff of either sign is not.
    #[test]
    fn resolved_eigenvalue_count_reads_the_band_not_the_sign() {
        let spectrum = [1.0, 1.0e-8, 1.0e-17, -1.0e-17];
        assert_eq!(resolved_eigenvalue_count(&spectrum, 0.0), 2);
        assert_eq!(resolved_eigenvalue_count(&spectrum, 1.0e-7), 1);
        let band = symmetric_spectrum_rounding_band(&spectrum);
        assert!(1.0e-17 <= band, "the roundoff pair must sit inside the eigensolver band");
    }

    /// A graded factor whose small singular value is resolved from `A` though its
    /// eigenvalue lies below the Gram's rounding band. A wide factor returns a
    /// complete right singular basis, including the directions of `AᵀA`'s null
    /// space that a thin SVD omits.
    #[test]
    fn factor_rank_partition_resolves_below_the_gram_band_and_completes_wide_factors() {
        let graded = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0e-10, 0.0], [0.0, 0.0, 0.0]];
        let partition = factor_rank_partition(&graded).expect("graded factor");
        assert_eq!(partition.rank, 2);
        let gram_band = symmetric_spectrum_rounding_band(&[1.0, 1.0e-20, 0.0]);
        assert!(
            1.0e-20 <= gram_band,
            "the second mode's eigenvalue must lie inside the Gram's band"
        );
        let root = partition.root_rows(partition.rank);
        let rebuilt = root.t().dot(&root);
        assert!((rebuilt[[0, 0]] - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!((rebuilt[[1, 1]] - 1.0e-20).abs() <= 1.0e-30);

        let wide = ndarray::array![[1.0, 2.0, 0.0, 0.0]];
        let wide_partition = factor_rank_partition(&wide).expect("wide factor");
        assert_eq!(wide_partition.rank, 1);
        assert_eq!(wide_partition.right_vectors.dim(), (4, 4));
        let gram = wide_partition
            .right_vectors
            .dot(&wide_partition.right_vectors.t());
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((gram[[i, j]] - expected).abs() <= 8.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn unit_roundoff_is_half_an_epsilon_gap() {
        assert_eq!(UNIT_ROUNDOFF * 2.0, f64::EPSILON);
        // 1.0 + u rounds to 1.0 (ties-to-even), 1.0 + 2u does not.
        assert_eq!(1.0_f64 + UNIT_ROUNDOFF, 1.0);
        assert!(1.0_f64 + 2.0 * UNIT_ROUNDOFF > 1.0);
    }

    #[test]
    fn growth_is_zero_for_exact_arithmetic_and_grows_linearly() {
        assert_eq!(accumulation_growth(0), 0.0);
        let one = accumulation_growth(1);
        assert!((one - UNIT_ROUNDOFF).abs() <= UNIT_ROUNDOFF * UNIT_ROUNDOFF * 4.0);
        // gamma_n / (n u) -> 1 from above, and is monotone in n.
        let mut previous = 0.0_f64;
        for n in [1_usize, 8, 64, 1024, 1 << 20] {
            let gamma = accumulation_growth(n);
            assert!(gamma > previous, "gamma must be monotone in n");
            assert!(gamma >= n as f64 * UNIT_ROUNDOFF);
            assert!(gamma <= n as f64 * UNIT_ROUNDOFF * 1.000_001);
            previous = gamma;
        }
    }

    #[test]
    fn compensated_band_is_the_naive_one_divided_by_the_term_count() {
        // The compensated band takes no term count at all -- independence from
        // `n` is a property of the signature, so the thing worth asserting is
        // the GAP against the naive bound, which is what a caller pays for by
        // scaling a compensated sum by `n` anyway.
        assert_eq!(compensated_band(0, 1.0), 2.0 * UNIT_ROUNDOFF);
        assert_eq!(compensated_band(3, 1.0), 5.0 * UNIT_ROUNDOFF);
        // Linear in the magnitude sum, as a forward-error bound must be.
        assert_eq!(compensated_band(3, 4.0), 4.0 * compensated_band(3, 1.0));
        // The gap grows without bound in n: that IS the defect it exists to
        // fix, so pin its size rather than merely asserting an inequality.
        let mut previous = 0.0_f64;
        for terms in [100_usize, 2_500, 10_000] {
            let ratio = accumulation_band(terms, 1.0) / compensated_band(3, 1.0);
            assert!(ratio > previous, "the gap must widen with n");
            // gamma_n ~ n*u, so the ratio approaches n/5.
            let expected = terms as f64 / 5.0;
            assert!(
                (ratio / expected - 1.0).abs() < 1.0e-6,
                "n={terms}: ratio {ratio} is not ~n/5 ({expected})"
            );
            previous = ratio;
        }
    }

    #[test]
    fn growth_saturates_rather_than_going_negative() {
        // n u >= 1 makes the textbook quotient negative; the band must not be.
        let vacuous = (1.0 / UNIT_ROUNDOFF).ceil() as usize;
        assert_eq!(accumulation_growth(vacuous), f64::INFINITY);
        assert_eq!(accumulation_growth(usize::MAX), f64::INFINITY);
    }

    #[test]
    fn band_bounds_a_cancelling_sum_that_is_exactly_zero() {
        // Every magnitude appears once positive and once negative, so the sum
        // is exactly zero in real arithmetic. The two signs are kept apart so
        // the cancellation is not absorbed by adjacent-pair exactness, which
        // would make the computed sum trivially 0.0.
        let magnitudes: Vec<f64> = (1..=256)
            .map(|k| (k as f64) * 0.1_f64.powi(k % 7))
            .collect();
        let terms: Vec<f64> = magnitudes
            .iter()
            .copied()
            .chain(magnitudes.iter().map(|value| -value))
            .collect();
        let absolute_sum: f64 = terms.iter().map(|value| value.abs()).sum();
        let computed: f64 = terms.iter().sum();
        let band = accumulation_band(terms.len(), absolute_sum);
        assert!(band > 0.0);
        assert!(
            computed.abs() <= band,
            "computed {computed:e} escaped its band {band:e}"
        );
    }
}
