//! One walk of the fit's penalty layout, shared by every summary surface.
//!
//! Presenting a fitted model's smooth terms means walking two structures at
//! once: the design's term list, and the fit's FLAT penalty layout
//! (`lambdas` / `penalty_block_trace` / `edf_by_block`). The walk is not
//! obvious — the flat layout opens with one `LinearTermRidge` block per
//! double-penalized linear term, an unpenalized or empty random-effect block
//! owns NO entry at all, and `SmoothTerm::coeff_range` is block-local while the
//! covariance it indexes is global. Every one of those has been a filed defect:
//! #1219 and #1277 (per-term EDF as the influence trace, not the block sum),
//! #1360 (the block-local → global shift), #1368 (advance the cursor by the
//! blocks a random-effect term ACTUALLY owns) and #1372 (the `LinearTermRidge`
//! prologue).
//!
//! Each of those five had to be landed TWICE, because the walk was written out
//! twice: once for the in-process CLI/report summary and once for the
//! persisted-model summary the Python `summary()` reads. The copies agree today
//! — that is worth stating plainly, since it means this module is not repairing
//! a live disagreement. What it removes is the standing obligation to land the
//! next one twice, and the window in between: the persisted copy's own comment
//! records #1368 as "fixed on the in-process `model_summary.rs` path but never
//! propagated here", where until it was propagated it collapsed
//! `ref_df`/`chi_sq`/`p_value` to `0`/`None` for every smooth following a `by=`
//! factor. Five fixes, five chances to miss one.
//!
//! So the walk lives here, once (issue #2470). What genuinely differs between
//! the surfaces stays a parameter: the in-process path has the real training
//! design and can hand over the exact weighted Gram, while the persisted path
//! replays frozen basis geometry and reconstructs an unweighted one. That is a
//! difference in the *evidence available*, not in the accounting, and it is the
//! only thing a caller is asked for.
//!
//! Reference-distribution inputs are read off the fit, not off the caller:
//! `wald_residual_degrees_of_freedom` for the denominator and
//! `LikelihoodScaleMetadata::wald_scale_is_estimated` for the `χ²`-vs-`F`
//! choice. Those two WERE a live divergence — the persisted path keyed the
//! scale predicate on the family NAME, which cannot distinguish a Gamma whose
//! shape was estimated from one whose shape the user pinned — and both are now
//! single-sourced (`fd998d957`).
//!
//! One asymmetry survives on purpose: `continuous_order` and `basis_note` are
//! computed here for every caller, but the persisted-model payload has no field
//! for them, so Python drops them. Surfacing them there is now a field mapping
//! rather than a second implementation.

use crate::estimate::summary::{
    SmoothPValueUnavailable, SmoothTermSummary, compute_continuous_smoothness_order,
};
use crate::model_types::result_types::UnifiedFitResult;
use gam_terms::basis::{BasisMetadata, PenaltySource};
use gam_terms::inference::smooth_test::{
    SmoothTestInput, SmoothTestScale, wood_smooth_test,
};
use gam_terms::smooth::{ShapeSpec, TermCollectionDesign, TermCollectionSpec};
use ndarray::Array2;

/// Build the smooth/random-effect rows of a model summary.
///
/// `design` and `spec` describe the term structure being presented — the real
/// training design on the in-process path, the frozen-basis replay on the
/// persisted one. `fit` owns every fitted quantity, including both inputs to
/// the Wald reference distribution. `whitening_gram` is the Wood (2013)
/// design-whitening metric `G = X'WX` in the fit's coefficient layout (the
/// exact weighted Gram when the inference block survived, else a reconstructed
/// unweighted `X'X`); `None` falls back to truncating the raw coefficient
/// covariance, which is the documented behaviour for a persisted model whose
/// Gram was not serialized.
///
/// Random-effect rows carry EDF only: they are boundary variance-component
/// tests, and a naive coefficient Wald `χ²` on them is anti-conservative.
pub fn smooth_term_summary_rows(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    whitening_gram: Option<&Array2<f64>>,
) -> Vec<SmoothTermSummary> {
    // The Wald smooth test uses the CONDITIONAL Bayesian covariance
    // `Vb = H⁻¹·φ̂` (mgcv's `Vp`, the covariance mgcv's `testStat` whitens by
    // default), NOT the smoothing-parameter-corrected `Vc`. `Vc` adds the λ̂
    // uncertainty `(∂β/∂ρ)·Cov(ρ)·(∂β/∂ρ)ᵀ`, whose variance concentrates in the
    // wiggle directions (those are the ones λ controls). For a heavily-smoothed,
    // near-linear term that inflation can exceed the linear direction's variance
    // and flip the whitened eigenvalue ordering, so the rank-`round(edf)`
    // truncation keeps a wiggle mode where β̂≈0 and reports the term
    // non-significant even though its linear effect is real (#2142). `Vc` is for
    // prediction/credible bands and is NEVER a substitute here: silently
    // swapping it in changes the Wald p-values (#2296). When the conditional
    // matrix is absent the smooth test is simply not reported.
    let cov_forwald = fit.beta_covariance();
    // Both reference-distribution inputs are fit-owned so they cannot drift
    // between presentation surfaces. The denominator is `n − edf` on the real
    // training row count; a representative/replayed design is basis geometry,
    // never a sample-size source.
    let residual_df = fit.wald_residual_degrees_of_freedom();
    let scale = if fit.likelihood_scale.wald_scale_is_estimated() {
        SmoothTestScale::Estimated
    } else {
        SmoothTestScale::Known
    };

    let gauge_free_gram = whitening_gram.map(|gram| {
        intercept_projected_gram(gram, design.intercept_range.clone())
            .unwrap_or_else(|| gram.clone())
    });
    let whitening_gram = gauge_free_gram.as_ref();

    let mut rows = Vec::<SmoothTermSummary>::new();

    // The fit's GLOBAL penalty layout (and thus `penalty_block_trace`) opens with
    // ONE `LinearTermRidge` block PER linear term carrying `double_penalty=true`
    // — not one shared block (`smooth/term_design.rs:289-311`; every non-intercept
    // effect owns its own REML coordinate so an unsupported slope can be shrunk
    // independently). Random-effect and smooth penalty blocks follow them.
    // Seeding `penalty_cursor` at 0 ignored those leading blocks, sliding every
    // per-term trace window off by the number of penalized linear terms and
    // masking the bug only on small dense fits (where `per_term_edf` reads the
    // influence matrix instead, #1372). Start the cursor PAST them by COUNTING
    // them in the recorded global ordering rather than re-deriving it — which is
    // what the `.count()` below does, and why it must not be replaced by a
    // boolean.
    let mut penalty_cursor = design
        .penaltyinfo
        .iter()
        .filter(|info| {
            matches!(&info.penalty.source, PenaltySource::Other(s) if s == "LinearTermRidge")
        })
        .count();

    for (re_idx, (name, range)) in design.random_effect_ranges.iter().enumerate() {
        // The design's RE-penalty loop skips a block that does not
        // `owns_penalty_block` (unpenalised, empty, or a single-level level
        // carrier), so such a term owns NO entry in the flat
        // `lambdas`/`penalty_block_trace`/`edf_by_block` layout. Advancing the
        // cursor by a fixed 1 (the #1368 defect) slides it one block past every
        // RE/smooth term that follows, so the trailing smooth's `cursor..+k`
        // window runs off the end of `penalty_block_trace`, `per_term_edf`
        // returns 0, the Wood test is skipped, and ref_df/chi_sq/p_value
        // collapse to 0/None. Ask the same predicate the design asks.
        let owns_penalty = spec
            .random_effect_terms
            .get(re_idx)
            .is_none_or(|term| term.owns_penalty_block(range.len()));
        let k_pen = usize::from(owns_penalty && !range.is_empty());
        // Per-term EDF as the influence-matrix trace over the term's coefficient
        // block (#1219, #1277) — never the legacy per-block-EDF sum, which
        // double-counts shared coefficients and can exceed the model total.
        let edf = fit.per_term_edf(range.clone(), penalty_cursor, k_pen);
        let edf_rank_bound = edf_rank_bound_label(fit, penalty_cursor, k_pen);
        penalty_cursor += k_pen;
        // Random-effect smooths are variance-component tests on the boundary; a
        // naive coefficient Wald χ² p-value is anti-conservative, so only EDF is
        // reported.
        rows.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df: edf.max(0.0),
            chi_sq: None,
            pvalue: None,
            continuous_order: None,
            basis_note: None,
            edf_rank_bound,
            pvalue_unavailable: None,
        });
    }

    // `SmoothTerm::coeff_range` is block-local (0-based within the smooth block);
    // the global coefficient layout is [intercept | linear | random | smooth], so
    // every term's block must be shifted by `smooth_start` before indexing the
    // global `fit.beta` / covariance / influence matrix. Omitting this offset
    // (the #1360 defect) slid each smooth's window one-per-preceding-column off,
    // folding the intercept and a neighbouring term's coefficients into the test.
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());

    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        let term_penalty_start = penalty_cursor;
        // Per-term EDF as the influence-matrix trace over the term's coefficient
        // block, NOT the legacy `Σ_kk edf_by_block` per-penalty sum. For a tensor
        // product `te`/`ti` (and anisotropic / adaptive smooths) several penalty
        // blocks span the SAME shared coefficient range, so the block-sum
        // double-counts and reports a per-term EDF exceeding the model total and
        // the design column count (#1219 / #1277).
        let global_range =
            (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let edf = fit.per_term_edf(global_range.clone(), penalty_cursor, k);
        let edf_rank_bound = edf_rank_bound_label(fit, penalty_cursor, k);
        penalty_cursor += k;
        let pvalue_unavailable = smooth_pvalue_unavailable(&term.shape);
        let smooth_test = if pvalue_unavailable.is_none() {
            cov_forwald.and_then(|cov| {
                wood_smooth_test(SmoothTestInput {
                    beta: fit.beta.view(),
                    covariance: cov,
                    influence_matrix: fit.coefficient_influence(),
                    // Wood (2013) design-whitening Gram in the original
                    // coefficient basis (#2142). Without it the rank-r
                    // truncation keeps the wrong eigen-subspace and a dominant
                    // wiggly smooth reads as non-significant.
                    whitening_gram,
                    coeff_range: global_range.clone(),
                    edf,
                    nullspace_dim: term.wald_unpenalized_dim(),
                    residual_df,
                    scale,
                })
            })
        } else {
            None
        };
        rows.push(SmoothTermSummary {
            name: term.name.clone(),
            edf,
            ref_df: smooth_test
                .as_ref()
                .map(|test| test.ref_df)
                .unwrap_or(edf.max(0.0)),
            chi_sq: smooth_test.as_ref().map(|test| test.statistic),
            pvalue: smooth_test.as_ref().map(|test| test.p_value),
            continuous_order: continuous_order_for_term(design, fit, term_penalty_start, k),
            basis_note: match &term.metadata {
                BasisMetadata::BSpline1D {
                    auto_shrink_note, ..
                } => auto_shrink_note.clone(),
                _ => None,
            },
            edf_rank_bound,
            pvalue_unavailable,
        });
    }

    rows
}

/// The whitening Gram with the intercept direction projected out,
/// `G − G[:, 0]·G[0, :] / G[0, 0]`, whose term block is the Gram of the term's
/// columns residualized on the intercept in the same metric.
///
/// A centered smooth is identified only up to a constant: its columns are
/// `B·Z` with `Z` spanning the null space of a sum-to-zero constraint, and the
/// constant it gives up lives in the intercept. Which constraint picks `Z` is a
/// gauge choice that leaves the fitted function unchanged, and moves every block
/// Gram `Z'B'WBZ` by a rank-one constant component. The rank-truncated statistic
/// follows the Gram, so it depended on the gauge. The concrete case: the
/// constraint sums over training rows without their prior weights, so a fit with
/// integer frequency weights and the fit on the duplicated rows, which are the
/// same model with the same fitted function, get different centerings and
/// different p-values. Residualizing on the intercept removes the constant
/// component from every term block, so the statistic is invariant to the
/// centering. It is the Gram the Wood (2013) statistic whitens by whenever the
/// constraint already holds in the Gram's metric, as it does for the unweighted
/// Gaussian training Gram, where `G[0, j] = 1'X_j = 0` and nothing changes.
///
/// `None` (use `gram` as is) when the design has no single intercept column or
/// the intercept's Gram entry is not positive.
fn intercept_projected_gram(
    gram: &Array2<f64>,
    intercept: std::ops::Range<usize>,
) -> Option<Array2<f64>> {
    if intercept.len() != 1 || intercept.start >= gram.nrows() || gram.nrows() != gram.ncols() {
        return None;
    }
    let col = intercept.start;
    let pivot = gram[[col, col]];
    if !(pivot.is_finite() && pivot > 0.0) {
        return None;
    }
    let cross = gram.column(col).to_owned();
    let mut projected = gram.clone();
    for i in 0..gram.nrows() {
        for j in 0..gram.ncols() {
            projected[[i, j]] -= cross[i] * cross[j] / pivot;
        }
    }
    Some(projected)
}

/// The reason a smooth of this shape has no valid significance reference, if
/// any. One predicate for every p-value surface (the Wald summary table and the
/// likelihood-ratio `smooth_significance`), so they cannot disagree about which
/// terms are testable.
pub fn smooth_pvalue_unavailable(shape: &ShapeSpec) -> Option<SmoothPValueUnavailable> {
    // Any held shape (an atom, a conjunction, or a per-margin tensor request)
    // restricts the coefficients to a cone, so the unconstrained reference
    // distribution does not apply.
    if shape.is_none() {
        None
    } else {
        Some(SmoothPValueUnavailable::ShapeConstrained)
    }
}

/// The label a term's EDF carries when a penalty block among its `count` blocks from
/// `start` is not rank-bound certified (#2901): "rank bound not assessed" when the
/// governor refused a certificate, else "rank bound not certified". Such a block's
/// trace is published raw, so the term's EDF is not clamped to its dimension.
fn edf_rank_bound_label(fit: &UnifiedFitResult, start: usize, count: usize) -> Option<String> {
    let bounds = fit.edf_rank_bound().get(start..start + count)?;
    if bounds
        .iter()
        .any(|bound| matches!(bound, crate::estimate::EdfRankBound::NotAssessed { .. }))
    {
        Some("rank bound not assessed".to_string())
    } else if bounds.iter().any(|bound| !bound.is_certified()) {
        Some("rank bound not certified".to_string())
    } else {
        None
    }
}

/// Invert the three-λ Matérn identity for a continuous-order smooth, in
/// PHYSICAL λ.
///
/// Unscaling identity: `S̃_k = S_k / c_k`, so `λ̃_k·S̃_k = (λ̃_k/c_k)·S_k` and
/// the physical λ the diagnostic needs is `λ_k = λ̃_k / c_k`. Returns `None`
/// unless the term owns exactly the three penalty blocks the identity is
/// written over and every one of them reports a usable normalization scale.
fn continuous_order_for_term(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    term_penalty_start: usize,
    k: usize,
) -> Option<crate::estimate::summary::ContinuousSmoothnessOrder> {
    if k != 3
        || term_penalty_start + 2 >= fit.lambdas.len()
        || term_penalty_start + 2 >= design.penaltyinfo.len()
    {
        return None;
    }
    let normalized_scale = |idx: usize| {
        let c = design.penaltyinfo[idx].penalty.normalization_scale;
        (c.is_finite() && c > 0.0).then_some(c)
    };
    let lambda_tilde = [
        fit.lambdas[term_penalty_start],
        fit.lambdas[term_penalty_start + 1],
        fit.lambdas[term_penalty_start + 2],
    ];
    let scales = [
        normalized_scale(term_penalty_start)?,
        normalized_scale(term_penalty_start + 1)?,
        normalized_scale(term_penalty_start + 2)?,
    ];
    Some(compute_continuous_smoothness_order(lambda_tilde, scales))
}

#[cfg(test)]
mod tests {
    use super::intercept_projected_gram;
    use gam_terms::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
    use ndarray::{Array1, Array2};

    fn weighted_gram(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
        let wx = x * &w.view().insert_axis(ndarray::Axis(1));
        x.t().dot(&wx)
    }

    /// Gauss-Jordan inverse with partial pivoting, for the small SPD test matrix.
    fn inverse(m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut a = m.clone();
        let mut inv = Array2::<f64>::eye(n);
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| a[[i, col]].abs().total_cmp(&a[[j, col]].abs()))
                .expect("a non-empty column");
            for k in 0..n {
                a.swap([col, k], [pivot, k]);
                inv.swap([col, k], [pivot, k]);
            }
            let d = a[[col, col]];
            for k in 0..n {
                a[[col, k]] /= d;
                inv[[col, k]] /= d;
            }
            for row in (0..n).filter(|&row| row != col) {
                let f = a[[row, col]];
                for k in 0..n {
                    a[[row, k]] -= f * a[[col, k]];
                    inv[[row, k]] -= f * inv[[col, k]];
                }
            }
        }
        inv
    }

    fn statistic(beta: &Array1<f64>, cov: &Array2<f64>, gram: &Array2<f64>) -> f64 {
        wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: cov,
            influence_matrix: None,
            whitening_gram: Some(gram),
            coeff_range: 1..5,
            edf: 2.6,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("the smooth test is defined")
        .statistic
    }

    /// Two centerings of one smooth give the same model: the term columns
    /// `X₂ = X₁·A + 1·cᵀ` span the same space as `X₁` once the intercept is
    /// included, and the coefficients map as `β₁ = P·β₂` with
    /// `P = [[1, cᵀ], [0, A]]`. This is the weighted-rows versus duplicated-rows
    /// case, where the sum-to-zero constraint is taken over different row
    /// multisets. The statistic whitened by the intercept-residualized Gram is the
    /// same in both gauges; whitened by the raw block Gram it is not.
    #[test]
    fn the_whitened_statistic_does_not_depend_on_the_smooth_centering() {
        let n = 40;
        let x = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let w = Array1::from_shape_fn(n, |i| 1.0 + (i % 3) as f64);
        let raw = |i: usize, j: usize| {
            let t = x[i];
            match j {
                0 => t,
                1 => t * t,
                2 => (3.0 * t).sin(),
                _ => (5.0 * t).cos(),
            }
        };
        // Gauge 1: term columns centered over the rows, ignoring the weights.
        let mut x1 = Array2::<f64>::ones((n, 5));
        for j in 0..4 {
            let mean = (0..n).map(|i| raw(i, j)).sum::<f64>() / n as f64;
            for i in 0..n {
                x1[[i, j + 1]] = raw(i, j) - mean;
            }
        }
        let a = Array2::from_shape_fn((4, 4), |(i, j)| {
            if i == j { 1.5 + 0.1 * i as f64 } else { 0.2 / (1.0 + (i + 2 * j) as f64) }
        });
        let c = Array1::from(vec![0.3, -0.2, 0.15, 0.05]);
        let mut p = Array2::<f64>::eye(5);
        for j in 0..4 {
            p[[0, j + 1]] = c[j];
            for i in 0..4 {
                p[[i + 1, j + 1]] = a[[i, j]];
            }
        }
        // X₂ = X₁·P keeps the fitted values: X₂·β₂ = X₁·P·β₂ = X₁·β₁.
        let x2 = x1.dot(&p);
        let beta2 = Array1::from(vec![0.4, 0.8, -0.5, 0.3, 0.1]);
        let beta1 = p.dot(&beta2);
        let g2 = weighted_gram(&x2, &w);
        let mut h2 = g2.clone();
        for i in 1..5 {
            h2[[i, i]] += 5.0 * (i * i) as f64;
        }
        let v2 = inverse(&h2);
        let v1 = p.dot(&v2).dot(&p.t());
        let g1 = weighted_gram(&x1, &w);

        let raw_1 = statistic(&beta1, &v1, &g1);
        let raw_2 = statistic(&beta2, &v2, &g2);
        assert!(
            (raw_1 - raw_2).abs() > 1e-3 * raw_1.abs(),
            "the raw block Gram must be gauge-dependent for this check to mean anything: {raw_1} vs {raw_2}"
        );
        let projected_1 = statistic(&beta1, &v1, &intercept_projected_gram(&g1, 0..1).unwrap());
        let projected_2 = statistic(&beta2, &v2, &intercept_projected_gram(&g2, 0..1).unwrap());
        assert!(
            (projected_1 - projected_2).abs() <= 1e-9 * projected_1.abs(),
            "{projected_1} vs {projected_2}"
        );
    }

    #[test]
    fn a_design_without_a_single_intercept_column_keeps_its_gram() {
        let gram = Array2::<f64>::eye(3);
        assert!(intercept_projected_gram(&gram, 0..0).is_none());
        assert!(intercept_projected_gram(&gram, 0..2).is_none());
        let mut singular = gram.clone();
        singular[[0, 0]] = 0.0;
        assert!(intercept_projected_gram(&singular, 0..1).is_none());
    }
}
