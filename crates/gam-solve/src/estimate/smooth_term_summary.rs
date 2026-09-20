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
//! So the walk lives here, once (issue #2470). The surfaces differ only in the
//! term structure they present — the real training design in process, the
//! frozen-basis replay for a persisted model — and every fitted quantity the
//! significance test reads comes from the fit itself.
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
use gam_terms::inference::random_effect_test::RandomEffectTestOutcome;
use gam_terms::inference::smooth_score_test::{
    SmoothScoreTestInput, SmoothScoreTestRefusal, smooth_score_test,
};
use gam_terms::inference::smooth_test::{SmoothTestResult, SmoothTestScale};
use gam_terms::smooth::{
    BOUNDED_SHRINKAGE_PENALTY_SOURCE, ShapeSpec, SmoothTerm, TermCollectionDesign,
};
use ndarray::Array2;

/// Where the presented design's predictor block sits in the fit's flat
/// coefficient and penalty layouts.
///
/// A single-predictor fit is the whole layout, [`SummaryBlockOffset::default`].
/// A multi-predictor fit (the Bernoulli marginal-slope marginal and slope
/// surfaces, #2997) presents each predictor from its own frozen design, and
/// that design's block-local coefficient columns and penalty blocks start after
/// the coefficients and λ of every block the fit orders before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SummaryBlockOffset {
    /// Coefficients of the fit's blocks before this one.
    pub coefficients: usize,
    /// Penalty blocks (λ) of the fit's blocks before this one.
    pub penalties: usize,
}

/// Build the smooth/random-effect rows of a model summary.
///
/// `design` describes the term structure being presented — the real
/// training design on the in-process path, the frozen-basis replay on the
/// persisted one. `fit` owns every fitted quantity: the smooth test is the
/// variance-component score test of
/// [`gam_terms::inference::smooth_score_test`], read off the fit's exact
/// penalized Hessian and weighted Gram, and a term the test cannot be computed
/// for carries the typed [`SmoothPValueUnavailable`] reason instead. `offset`
/// places `design` inside the fit's layouts: every index into `fit` (β,
/// penalized Hessian, weighted Gram, covariance, influence matrix, λ, traces)
/// is global, every index into `design` is block-local.
///
/// Random-effect rows do not use the Wald test: their null `σ²_b = 0` is on the
/// boundary, where a coefficient Wald `χ²` has no valid reference. They carry
/// the variance-component score test the fit recorded in
/// `FitArtifacts::random_effect_tests` (exact spectral reference; see
/// `gam_terms::inference::random_effect_test`), or its typed absence.
pub fn smooth_term_summary_rows(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    offset: SummaryBlockOffset,
) -> Vec<SmoothTermSummary> {
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
    // The score test's inputs are fit-level and shared by every smooth: `H =
    // X'WX + S(λ)` and `X'WX` from the one inference block, stored in the
    // coefficient gauge's active frame and pushed forward to the saved frame
    // of `beta` and the terms' coefficient ranges, so they belong to the same
    // fit and the same layout (gam#3346). A Gram rebuilt without the fitted
    // weights is not a substitute.
    let score_fit = ScoreTestFit::of(fit, residual_df, scale);

    let shift = |range: &std::ops::Range<usize>| {
        (offset.coefficients + range.start)..(offset.coefficients + range.end)
    };

    let mut rows = Vec::<SmoothTermSummary>::new();

    // The fit's GLOBAL penalty layout (and thus `penalty_block_trace`) opens with
    // ONE `LinearTermRidge` block PER linear term carrying `double_penalty=true`
    // (or a `BoundedShrinkage` block per shrinkage-prior `bounded()` term)
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
    let mut penalty_cursor = offset.penalties
        + design
            .penaltyinfo
            .iter()
            .filter(|info| {
                matches!(
                    &info.penalty.source,
                    PenaltySource::Other(s)
                        if s == "LinearTermRidge" || s == BOUNDED_SHRINKAGE_PENALTY_SOURCE
                )
            })
            .count();

    for (name, local_range) in design.random_effect_ranges.iter() {
        let range = &shift(local_range);
        // Every random-effect block owns exactly one ridge in the flat
        // `lambdas`/`penalty_block_trace`/`edf_by_block` layout, placed after
        // the linear ridges and before the smooths in the design's order.
        let k_pen = 1;
        // Per-term EDF as the influence-matrix trace over the term's coefficient
        // block (#1219, #1277) — never the legacy per-block-EDF sum, which
        // double-counts shared coefficients and can exceed the model total.
        let edf = fit.per_term_edf(range.clone(), penalty_cursor, k_pen);
        let edf_rank_bound = edf_rank_bound_label(fit, penalty_cursor, k_pen);
        penalty_cursor += k_pen;
        // The variance component's null is on the boundary, so the row reports
        // the score test the fit recorded for this exact term — matched by name
        // AND coefficient range, so a replayed design whose layout drifted from
        // the training one finds no record rather than a neighbour's.
        let recorded = fit
            .artifacts
            .random_effect_tests
            .iter()
            .find(|record| record.term == *name && record.coefficient_range == *range)
            .map(|record| &record.outcome);
        let (ref_df, chi_sq, pvalue, pvalue_unavailable) = match recorded {
            Some(RandomEffectTestOutcome::Tested(test)) => {
                (test.reference_df, Some(test.statistic), Some(test.p_value), None)
            }
            Some(RandomEffectTestOutcome::Unavailable { reason }) => (
                edf.max(0.0),
                None,
                None,
                Some(SmoothPValueUnavailable::RandomEffect(*reason)),
            ),
            None => (
                edf.max(0.0),
                None,
                None,
                Some(SmoothPValueUnavailable::RandomEffectTestNotRecorded),
            ),
        };
        rows.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df,
            chi_sq,
            pvalue,
            continuous_order: None,
            basis_note: None,
            edf_rank_bound,
            pvalue_unavailable,
        });
    }

    // `SmoothTerm::coeff_range` is block-local (0-based within the smooth block);
    // the global coefficient layout is [intercept | linear | random | smooth], so
    // every term's block must be shifted by `smooth_start` before indexing the
    // global `fit.beta` / covariance / influence matrix. Omitting this offset
    // (the #1360 defect) slid each smooth's window one-per-preceding-column off,
    // folding the intercept and a neighbouring term's coefficients into the test.
    let smooth_start = offset.coefficients
        + design
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
        let smooth_test = match (smooth_pvalue_unavailable(&term.shape), &score_fit) {
            (Some(reason), _) => Err(reason),
            (None, Err(reason)) => Err(*reason),
            (None, Ok(score_fit)) => score_fit.test_term(term, global_range.clone()),
        };
        let (smooth_test, pvalue_unavailable) = match smooth_test {
            Ok(test) => (Some(test), None),
            Err(reason) => (None, Some(reason)),
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
            continuous_order: continuous_order_for_term(
                design,
                fit,
                term_penalty_start,
                term_penalty_start - offset.penalties,
                k,
            ),
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

/// The fit-level inputs of the smooth score test, shared by every term.
struct ScoreTestFit<'a> {
    beta: ndarray::ArrayView1<'a, f64>,
    penalized_hessian: std::borrow::Cow<'a, Array2<f64>>,
    weighted_gram: std::borrow::Cow<'a, Array2<f64>>,
    covariance_scale: f64,
    residual_df: Option<f64>,
    scale: SmoothTestScale,
}

impl<'a> ScoreTestFit<'a> {
    fn of(
        fit: &'a UnifiedFitResult,
        residual_df: Option<f64>,
        scale: SmoothTestScale,
    ) -> Result<Self, SmoothPValueUnavailable> {
        // Both curvatures are read in the saved frame, the frame of `beta` and
        // of every term's coefficient range; a fit whose gauge leaves them no
        // unique saved-frame form has no score test (gam#3346).
        if fit.inference.is_none() {
            return Err(SmoothPValueUnavailable::FitCurvatureUnavailable);
        }
        let saved = |form: Result<Option<std::borrow::Cow<'a, Array2<f64>>>, String>| {
            form.ok()
                .flatten()
                .ok_or(SmoothPValueUnavailable::FitCurvatureUnavailable)
        };
        let penalized_hessian = saved(fit.saved_frame_penalized_hessian())?;
        let weighted_gram = saved(fit.saved_frame_weighted_gram())?;
        let covariance_scale = fit
            .coefficient_covariance_scale()
            .map_err(|_| SmoothPValueUnavailable::DispersionUnavailable)?;
        Ok(Self {
            beta: fit.beta.view(),
            penalized_hessian,
            weighted_gram,
            covariance_scale,
            residual_df,
            scale,
        })
    }

    /// The score test of one smooth term against its active structural
    /// penalties, one variance component per penalty.
    ///
    /// The penalties are passed at the scale their bases were built at; the
    /// test puts each on its own null scale, so it is invariant to rescaling
    /// any one of them. A penalty that is neither a dense `m × m` block nor an
    /// operator of that dimension cannot be read, and the term is refused
    /// rather than tested against part of its penalty.
    fn test_term(
        &self,
        term: &SmoothTerm,
        coeff_range: std::ops::Range<usize>,
    ) -> Result<SmoothTestResult, SmoothPValueUnavailable> {
        let m = coeff_range.len();
        if term.active_penalties.is_empty() {
            return Err(SmoothPValueUnavailable::UnpenalizedDirection);
        }
        let mut structural_penalties = Vec::with_capacity(term.active_penalties.len());
        for penalty in &term.active_penalties {
            let matrix = if penalty.matrix.dim() == (m, m) {
                penalty.matrix.clone()
            } else {
                match &penalty.op {
                    Some(op) if op.dim() == m => op.as_dense(),
                    _ => return Err(SmoothPValueUnavailable::FitCurvatureUnavailable),
                }
            };
            if !matrix.iter().any(|v| *v != 0.0) || matrix.iter().any(|v| !v.is_finite()) {
                return Err(SmoothPValueUnavailable::FitCurvatureUnavailable);
            }
            structural_penalties.push(matrix);
        }
        smooth_score_test(SmoothScoreTestInput {
            beta: self.beta,
            penalized_hessian: &self.penalized_hessian,
            weighted_gram: &self.weighted_gram,
            coeff_range,
            structural_penalties: &structural_penalties,
            covariance_scale: self.covariance_scale,
            residual_df: self.residual_df,
            scale: self.scale,
        })
        .map_err(|refusal| match refusal {
            SmoothScoreTestRefusal::InconsistentFit => {
                SmoothPValueUnavailable::FitCurvatureUnavailable
            }
            SmoothScoreTestRefusal::UnpenalizedDirection => {
                SmoothPValueUnavailable::UnpenalizedDirection
            }
            SmoothScoreTestRefusal::NotIdentified => SmoothPValueUnavailable::NotIdentified,
            SmoothScoreTestRefusal::IndefiniteCurvature => {
                SmoothPValueUnavailable::IndefiniteCurvature
            }
            SmoothScoreTestRefusal::ResidualDfUnavailable => {
                SmoothPValueUnavailable::ResidualDfUnavailable
            }
        })
    }
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
/// `term_penalty_start` indexes the fit's λ, `local_penalty_start` the same
/// blocks in `design`'s own penalty list.
fn continuous_order_for_term(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    term_penalty_start: usize,
    local_penalty_start: usize,
    k: usize,
) -> Option<crate::estimate::summary::ContinuousSmoothnessOrder> {
    if k != 3
        || term_penalty_start + 2 >= fit.lambdas.len()
        || local_penalty_start + 2 >= design.penaltyinfo.len()
    {
        return None;
    }
    let normalized_scale = |idx: usize| {
        let c = design.penaltyinfo[local_penalty_start + idx].penalty.normalization_scale;
        (c.is_finite() && c > 0.0).then_some(c)
    };
    let lambda_tilde = [
        fit.lambdas[term_penalty_start],
        fit.lambdas[term_penalty_start + 1],
        fit.lambdas[term_penalty_start + 2],
    ];
    let scales = [normalized_scale(0)?, normalized_scale(1)?, normalized_scale(2)?];
    Some(compute_continuous_smoothness_order(lambda_tilde, scales))
}
