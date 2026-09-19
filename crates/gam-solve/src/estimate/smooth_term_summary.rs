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
//! So the walk lives here, once (issue #2470). Every input it reads is
//! fit-owned or part of the term layout, so the in-process and persisted
//! surfaces hand over the same three objects and cannot differ in evidence.
//! That includes both reference-distribution inputs:
//! `wald_residual_degrees_of_freedom` for the denominator and
//! `LikelihoodScaleMetadata::wald_scale_is_estimated` for the known-vs-estimated
//! scale choice (`fd998d957`).
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
use gam_terms::inference::selection_replay::{
    SmoothLrSelectionDecline, SmoothWaldSelectionTest, lr_tested_block,
    smooth_wald_selection_test, symmetrized,
};
use gam_terms::smooth::{ShapeSpec, TermCollectionDesign, TermCollectionSpec};
use ndarray::{Array2, ArrayView1, s};

/// Build the smooth/random-effect rows of a model summary.
///
/// `design` and `spec` describe the term structure being presented — the real
/// training design on the in-process path, the frozen-basis replay on the
/// persisted one. `fit` owns every fitted quantity, including every input to
/// the smooth Wald test ([`smooth_wald_test`]).
///
/// Random-effect rows carry EDF only: they are boundary variance-component
/// tests, and a naive coefficient Wald `χ²` on them is anti-conservative.
pub fn smooth_term_summary_rows(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
) -> Vec<SmoothTermSummary> {
    // Both reference-distribution inputs are fit-owned so they cannot drift
    // between presentation surfaces. The denominator is `n − edf` on the real
    // training row count; a representative/replayed design is basis geometry,
    // never a sample-size source. An estimated scale without a residual count
    // has no reference, so its terms publish no p-value.
    let residual_df = if fit.likelihood_scale.wald_scale_is_estimated() {
        fit.wald_residual_degrees_of_freedom().map(Some)
    } else {
        Some(None)
    };

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
        // The design's RE-penalty loop skips a block when EITHER it is
        // unpenalised OR its coefficient range is empty
        // (`design_construction.rs` `range.is_empty() || !penalized` →
        // `continue`), so such a term owns NO entry in the flat
        // `lambdas`/`penalty_block_trace`/`edf_by_block` layout. A factor `by=`
        // smooth injects exactly such an UNPENALISED treatment-coded factor
        // main-effect block, and a penalised RE term with zero kept groups is
        // the empty-range case. Advancing the cursor by a fixed 1 (the #1368
        // defect) slides it one block past every RE/smooth term that follows, so
        // the trailing smooth's `cursor..+k` window runs off the end of
        // `penalty_block_trace`, `per_term_edf` returns 0, the Wood test is
        // skipped, and ref_df/chi_sq/p_value collapse to 0/None. Mirror BOTH
        // design conditions.
        let penalized = spec
            .random_effect_terms
            .get(re_idx)
            .map(|term| term.penalized)
            .unwrap_or(true);
        let k_pen = usize::from(penalized && !range.is_empty());
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
        let mut pvalue_unavailable = smooth_pvalue_unavailable(&term.shape);
        let smooth_test = match (pvalue_unavailable, residual_df) {
            (None, Some(residual_df)) => smooth_wald_test(
                design,
                fit,
                global_range.clone(),
                term_penalty_start..term_penalty_start + k,
                residual_df,
            ),
            _ => None,
        };
        let smooth_test = match smooth_test {
            Some(Ok(test)) => Some(test),
            Some(Err(_)) => {
                pvalue_unavailable = Some(SmoothPValueUnavailable::SelectionRefused);
                None
            }
            None => None,
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

/// The summary smooth Wald test of one term, from the fit alone.
///
/// The tested block's conditional Bayesian covariance `Vb = H⁻¹·φ̂` (mgcv's
/// `Vp`), NOT the smoothing-parameter-corrected `Vc`: the selection of `λ̂` is
/// priced by the replay, and folding its uncertainty into the covariance as
/// well would count it twice (#2142, #2296). Its `H⁻¹` block `B`, the term's own
/// penalty components at their fitted `λ̂` and the estimate `β̂` are handed to
/// [`term_wald_test`]. When the covariance or its scale is absent the test is
/// not reported (`None`); a layout or geometry refusal is `Some(Err)`, and the
/// term then publishes no p-value, never a conditional one.
fn smooth_wald_test(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    coeff_range: std::ops::Range<usize>,
    penalty_blocks: std::ops::Range<usize>,
    residual_df: Option<f64>,
) -> Option<Result<SmoothWaldSelectionTest, SmoothLrSelectionDecline>> {
    let covariance = fit.beta_covariance()?;
    let covariance_scale = fit
        .coefficient_covariance_scale()
        .ok()
        .filter(|scale| scale.is_finite() && *scale > 0.0)?;
    if coeff_range.end > covariance.nrows()
        || coeff_range.end > covariance.ncols()
        || coeff_range.end > fit.beta.len()
    {
        return Some(Err(SmoothLrSelectionDecline::GeometryRefused));
    }
    let hessian_inverse = covariance
        .slice(s![coeff_range.clone(), coeff_range.clone()])
        .mapv(|value| value / covariance_scale);
    let dimension = coeff_range.len();
    let mut penalties = Vec::<(Array2<f64>, f64)>::new();
    for index in penalty_blocks {
        let (Some(block), Some(&lambda)) = (design.penalties.get(index), fit.lambdas.get(index))
        else {
            return Some(Err(SmoothLrSelectionDecline::GeometryRefused));
        };
        let range = &block.col_range;
        if range.start < coeff_range.start
            || range.end > coeff_range.end
            || block.local.dim() != (range.len(), range.len())
        {
            return Some(Err(SmoothLrSelectionDecline::GeometryRefused));
        }
        let mut local = Array2::<f64>::zeros((dimension, dimension));
        let offset = range.start - coeff_range.start;
        local
            .slice_mut(s![offset..offset + range.len(), offset..offset + range.len()])
            .assign(&block.local);
        penalties.push((local, lambda));
    }
    Some(term_wald_test(
        &hessian_inverse,
        fit.beta.slice(s![coeff_range]),
        &penalties,
        covariance_scale,
        residual_df,
    ))
}

/// [`smooth_wald_selection_test`] on one tested block.
///
/// `hessian_inverse` is the block `B` of `H⁻¹`, `beta` the block's estimate,
/// `penalties` the term's unit penalty components on the block with their
/// fitted `λ̂`, and `covariance_scale` the `φ̂` with `Vb = H⁻¹·φ̂`. The observed
/// whitened score is `u = Dᵀβ̂/√φ̂` for the dual `D` of [`lr_tested_block`],
/// which is `N(0, I)` under the null.
///
/// Each `λ_i`'s log-scale window is the solver's own resolvability rule
/// ([`resolvability_domain_from_gram_blocks`](crate::estimate::rho_domain::resolvability_domain_from_gram_blocks))
/// read in the whitened coordinates the replay moves in: data curvature `I`
/// against `WᵀS_iW`. That is the block's Schur-profiled information `Ĩ` against
/// `S_i`, carried by the congruence `W` rather than formed as `B⁻¹ − S_λ`,
/// whose difference cancels exactly where a term is shrunk. The information is
/// the fit's own, so the window is a property of the fitted model: it needs no
/// design and moves with the coefficients under any reparameterization of the
/// block, including a change of the smooth's centering constraint.
fn term_wald_test(
    hessian_inverse: &Array2<f64>,
    beta: ArrayView1<'_, f64>,
    penalties: &[(Array2<f64>, f64)],
    covariance_scale: f64,
    residual_df: Option<f64>,
) -> Result<SmoothWaldSelectionTest, SmoothLrSelectionDecline> {
    let dimension = beta.len();
    let mut s_lambda = Array2::<f64>::zeros((dimension, dimension));
    let mut unit_penalties = Vec::<Array2<f64>>::new();
    let mut log_lambda = Vec::<f64>::new();
    for (penalty, lambda) in penalties {
        if !(lambda.is_finite() && *lambda >= 0.0) || penalty.dim() != (dimension, dimension) {
            return Err(SmoothLrSelectionDecline::GeometryRefused);
        }
        // A component at `λ = 0` penalizes nothing and has no scale to select.
        if *lambda == 0.0 {
            continue;
        }
        s_lambda.scaled_add(*lambda, penalty);
        unit_penalties.push(penalty.clone());
        log_lambda.push(lambda.ln());
    }
    let block = lr_tested_block(Some(hessian_inverse), Some(&s_lambda), &(0..dimension))
        .ok_or(SmoothLrSelectionDecline::GeometryRefused)?;
    let dual = block
        .dual
        .as_ref()
        .ok_or(SmoothLrSelectionDecline::GeometryRefused)?;
    let root_scale = covariance_scale.sqrt();
    let observed: Vec<f64> = dual.t().dot(&beta).iter().map(|value| value / root_scale).collect();
    let identified = block.whitener.ncols();
    let whitened: Vec<Array2<f64>> = unit_penalties
        .iter()
        .map(|penalty| symmetrized(block.whitener.t().dot(penalty).dot(&block.whitener)))
        .collect();
    let (lower, upper) = crate::estimate::rho_domain::resolvability_domain_from_gram_blocks(
        &Array2::<f64>::eye(identified),
        whitened.iter().map(|penalty| (0..identified, penalty)),
        whitened.len(),
    );
    let windows: Vec<(f64, f64)> = lower
        .iter()
        .zip(upper.iter())
        .zip(log_lambda.iter())
        .map(|((&low, &high), &rho)| (low - rho, high - rho))
        .collect();
    smooth_wald_selection_test(
        &block.whitener,
        &observed,
        &unit_penalties,
        &log_lambda,
        &windows,
        residual_df,
    )
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
    use super::term_wald_test;
    use gam_terms::inference::selection_replay::SmoothLrSelectionDecline;
    use ndarray::{Array1, Array2};

    /// Gauss-Jordan inverse with partial pivoting, for the small test matrices.
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

    /// A second-difference penalty on `d` coefficients: rank `d − 2`, so the
    /// term keeps an unpenalized null space the way a smooth does.
    fn second_difference_penalty(d: usize) -> Array2<f64> {
        let mut difference = Array2::<f64>::zeros((d - 2, d));
        for row in 0..d - 2 {
            difference[[row, row]] = 1.0;
            difference[[row, row + 1]] = -2.0;
            difference[[row, row + 2]] = 1.0;
        }
        difference.t().dot(&difference)
    }

    /// One tested block in the fit's own gauge: data curvature `G`, penalty `S`
    /// at `λ`, so `B = (G + λS)⁻¹`.
    fn block(d: usize, lambda: f64) -> (Array2<f64>, Array2<f64>) {
        let gram = Array2::from_shape_fn((d, d), |(i, j)| {
            let (x, y) = (i as f64 / d as f64, j as f64 / d as f64);
            4.0 * (-(x - y).powi(2) * 6.0).exp() + if i == j { 0.5 } else { 0.0 }
        });
        let penalty = second_difference_penalty(d);
        let hessian = &gram + &(lambda * &penalty);
        (inverse(&hessian), penalty)
    }

    /// Two parameterizations of one smooth give the same model: coefficients
    /// `β₁ = A·β₂`, covariance `B₁ = A·B₂·Aᵀ` and penalty `S₁ = A⁻ᵀ·S₂·A⁻¹`
    /// (so `β₁ᵀS₁β₁ = β₂ᵀS₂β₂`). This is the weighted-rows versus duplicated-rows
    /// case, where the centering constraint is taken over different row
    /// multisets. The statistic, its reference and the p-value are properties of
    /// the model, so they must agree in both gauges — which the design-Gram
    /// whitening this test replaces did not.
    #[test]
    fn the_wald_test_does_not_depend_on_the_block_parameterization() {
        let d = 6;
        let lambda = 3.0;
        let (b2, s2) = block(d, lambda);
        let beta2 = Array1::from(vec![0.4, -0.3, 0.25, 0.1, -0.2, 0.15]);
        let a = Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j { 1.5 + 0.1 * i as f64 } else { 0.2 / (1.0 + (i + 2 * j) as f64) }
        });
        let a_inverse = inverse(&a);
        let beta1 = a.dot(&beta2);
        let b1 = a.dot(&b2).dot(&a.t());
        let s1 = a_inverse.t().dot(&s2).dot(&a_inverse);
        for (scale, residual_df) in [(1.0, None), (0.7, Some(40.0))] {
            let one = term_wald_test(&b1, beta1.view(), &[(s1.clone(), lambda)], scale, residual_df)
                .expect("the test is defined");
            let two = term_wald_test(&b2, beta2.view(), &[(s2.clone(), lambda)], scale, residual_df)
                .expect("the test is defined");
            let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
            assert!(close(one.statistic, two.statistic), "{} vs {}", one.statistic, two.statistic);
            assert!(close(one.ref_df, two.ref_df), "{} vs {}", one.ref_df, two.ref_df);
            assert!(close(one.p_value, two.p_value), "{} vs {}", one.p_value, two.p_value);
        }
    }

    /// A smoothing parameter the fit cannot have produced is a geometry refusal:
    /// the term then publishes no p-value rather than a conditional one.
    #[test]
    fn a_non_finite_lambda_is_refused() {
        let (b, s) = block(5, 2.0);
        let beta = Array1::from(vec![0.1; 5]);
        assert_eq!(
            term_wald_test(&b, beta.view(), &[(s, f64::NAN)], 1.0, None).err(),
            Some(SmoothLrSelectionDecline::GeometryRefused)
        );
    }
}
