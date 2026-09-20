//! Per-term diagnostics of a standard GAM's additive predictor, evaluated on a
//! design built at caller-supplied rows.

use ndarray::{ArrayView1, ArrayView2};
use std::ops::Range;

/// Partial dependence of one term block: `f_t(x) = X_t(x) β_t` and the matching
/// delta-method standard error `sqrt(diag(X_t V_t X_tᵀ))`, where `V_t` is the
/// term block of the coefficient covariance.
///
/// A projected variance `x_tᵀ V_t x_t` below minus the roundoff band of its own
/// accumulation means `V_t` is not positive semidefinite along that row; that
/// is a defect in the covariance, reported as an error rather than clamped to a
/// zero standard error. Only a negative inside the band — a zero variance the
/// arithmetic could not resolve — reads as zero.
pub fn term_partial_dependence(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
    block: Range<usize>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let p = beta.len();
    if block.start > block.end
        || block.end > p
        || block.end > design.ncols()
        || block.end > covariance.nrows()
        || block.end > covariance.ncols()
    {
        return Err(format!(
            "term partial dependence block {block:?} must lie inside the {p} coefficients, the \
             {}-column design and the {:?} covariance",
            design.ncols(),
            covariance.dim()
        ));
    }
    let n = design.nrows();
    let mut predicted = vec![0.0_f64; n];
    let mut se = vec![0.0_f64; n];
    for i in 0..n {
        let xi = design.row(i);
        let mut f = 0.0_f64;
        for c in block.clone() {
            f += xi[c] * beta[c];
        }
        predicted[i] = f;
        let mut var = 0.0_f64;
        let mut absolute_sum = 0.0_f64;
        for a in block.clone() {
            let xa = xi[a];
            for b in block.clone() {
                let term = xa * covariance[[a, b]] * xi[b];
                var += term;
                absolute_sum += term.abs();
            }
        }
        // Each summand rounds twice (two products) before `k² − 1` additions,
        // so the accumulation depth is `k² + 1` for a `k`-column block.
        let width = block.len();
        let band = gam_linalg::roundoff::accumulation_band(width * width + 1, absolute_sum);
        if !var.is_finite() || var < -band {
            return Err(format!(
                "term partial dependence: row {i} has projected variance {var:e} below the \
                 roundoff band -{band:e}; the covariance of block {block:?} is not positive \
                 semidefinite along this design row"
            ));
        }
        se[i] = var.max(0.0).sqrt();
    }
    Ok((predicted, se))
}

/// Per-term variance share `cov(X_t β_t, X β) / var(X β)` for each named block.
///
/// This is a genuine variance decomposition: `var(η) = Σ_t cov(f_t, η)`, so the
/// shares over every term sum to exactly 1 (an intercept contributes a constant
/// with zero covariance). Each term's cross-covariance with every other term is
/// split symmetrically — half to each side — which is the Shapley allocation
/// for a sum of terms. The naive `var(f_t) / var(η)` drops all cross terms: for
/// `f_1 = x`, `f_2 = -0.9x` it reports shares 100 and 81 against a total of
/// 0.01·var(x). A share can exceed 1 or be negative only when terms genuinely
/// anticorrelate, which is honest rather than a bug.
pub fn term_variance_shares(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    blocks: &[(String, Range<usize>)],
) -> Result<Vec<(String, f64)>, String> {
    let p = beta.len();
    if design.ncols() < p {
        return Err(format!(
            "term variance shares need a design with at least the {p} coefficient columns, got {}",
            design.ncols()
        ));
    }
    if let Some((name, block)) = blocks
        .iter()
        .find(|(_, block)| block.start > block.end || block.end > p)
    {
        return Err(format!(
            "term variance shares: block {block:?} of term {name:?} lies outside the {p} coefficients"
        ));
    }
    let n = design.nrows();
    let mut eta = vec![0.0_f64; n];
    for i in 0..n {
        let xi = design.row(i);
        let mut s = 0.0_f64;
        for c in 0..p {
            s += xi[c] * beta[c];
        }
        eta[i] = s;
    }
    let total_var = population_variance(&eta);
    // A share is `cov(f_t, η) / var(η)`; with a constant predictor over the
    // supplied rows every share is 0/0, and reporting zeros would contradict
    // the decomposition's sum-to-one identity.
    if !(total_var.is_finite() && total_var > 0.0) {
        return Err(format!(
            "term variance shares are undefined: the additive predictor has variance \
             {total_var:e} over the {n} supplied rows; supply rows on which the predictor varies"
        ));
    }
    let mut out: Vec<(String, f64)> = Vec::with_capacity(blocks.len());
    for (name, block) in blocks {
        let mut contrib = vec![0.0_f64; n];
        for i in 0..n {
            let xi = design.row(i);
            let mut s = 0.0_f64;
            for c in block.clone() {
                s += xi[c] * beta[c];
            }
            contrib[i] = s;
        }
        let share = population_covariance(&contrib, &eta) / total_var;
        out.push((name.clone(), share));
    }
    Ok(out)
}

/// Population variance (divide by `n`, matching numpy `np.var`'s default).
fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64
}

/// Population covariance of two equal-length slices (divide by `n`, matching
/// `population_variance`).
fn population_covariance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&va, &vb)| (va - mean_a) * (vb - mean_b))
        .sum::<f64>()
        / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn partial_dependence_rejects_a_covariance_indefinite_along_a_row() {
        let design = array![[1.0, 0.0], [1.0, 1.0]];
        let beta = array![0.5, 2.0];
        // PSD on the first row (variance 1) but x = (1, 1) sees 1 + 0 + 0 - 2 = -1.
        let covariance = array![[1.0, 0.0], [0.0, -2.0]];
        let err = term_partial_dependence(design.view(), beta.view(), covariance.view(), 0..2)
            .expect_err("an indefinite projected variance must not be clamped to a zero SE");
        assert!(err.contains("row 1"), "{err}");
    }

    #[test]
    fn partial_dependence_standard_error_is_the_projected_quadratic_form() {
        let design = array![[1.0, 2.0], [0.0, 1.0]];
        let beta = array![1.0, -1.0];
        let covariance = array![[2.0, 0.5], [0.5, 1.0]];
        let (predicted, se) =
            term_partial_dependence(design.view(), beta.view(), covariance.view(), 0..2)
                .expect("a PSD covariance yields finite standard errors");
        assert_eq!(predicted, vec![-1.0, -1.0]);
        // Row 0: 2 + 2·(0.5·2) + 4·1 = 8; row 1: 1.
        assert!((se[0] - 8.0_f64.sqrt()).abs() <= 4.0 * f64::EPSILON * 8.0_f64.sqrt());
        assert_eq!(se[1], 1.0);
    }

    #[test]
    fn variance_shares_reject_a_constant_predictor() {
        let design = array![[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]];
        let beta = array![1.0, 2.0];
        let blocks = vec![("x".to_string(), 1..2)];
        let err = term_variance_shares(design.view(), beta.view(), &blocks)
            .expect_err("0/0 shares must not be reported as zero");
        assert!(err.contains("undefined"), "{err}");
    }
}
