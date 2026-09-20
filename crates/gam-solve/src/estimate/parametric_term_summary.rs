//! The parametric rows of a model summary, shared by every surface.
//!
//! The in-process CLI summary and the persisted-model summary the Python
//! `summary()` reads present the same tables from the same fit, so they are
//! built once, here, beside [`super::smooth_term_summary_rows`]. The Wald
//! reference distribution is read off the fit — `N(0, 1)` / `χ²_q` when the
//! scale is known, Student-t / `F(q, n − edf)` on
//! `wald_residual_degrees_of_freedom` when it is estimated — so no caller
//! chooses it.
//!
//! A linear term carries its own REML ridge (its function-mass penalty `S_J`),
//! and its posterior covariance charges that prior's variance to the estimate:
//! `β̂_J/√(V_p)_JJ` is too small under the null and its test is conservative.
//! With `A = X'WX + S_O` (the smooths' priors kept) and `H = A + S_J`, the rows
//! `J` of `H⁻¹` are `(I + (A⁻¹)_JJ S_J)⁻¹` times those of `A⁻¹`, so `β̂_J` and
//! its covariance `φ[H⁻¹AH⁻¹]_JJ` share that factor and the Wald statistic
//! `β̂_J' (φ[H⁻¹AH⁻¹]_JJ)⁻¹ β̂_J` equals the one of the fit with `J` left
//! unpenalized, whatever `λ_J` is. Since the statistic does not depend on
//! `λ_J`, `λ_J`'s uncertainty is not charged to it: the covariance is the
//! conditional `φ[H⁻¹AH⁻¹]_JJ = (V_c)_JJ − φ(H⁻¹S_JH⁻¹)_JJ`, not the
//! smoothing-corrected one, whose first-order `λ_J` term made the test
//! conservative again. An unpenalized term has `S_J = 0` and keeps the display
//! covariance.
//!
//! A fixed factor (`+ g`, `factor(g)`) is a full one-hot block of its `L`
//! levels under its own REML ridge `λI`, beside the intercept, which a model
//! keeps whenever a fixed factor spans the constant. The block's constant
//! direction `1` is aliased with the intercept: with `M` the
//! Schur complement of `A` on the block, `M1 = 0`, so the data say nothing about
//! the mean of the level effects and the ridge sets it to zero. The factor's
//! null — all levels equal — is a hypothesis on the `L − 1` contrasts `Q'β_J`,
//! `Q` an orthonormal basis of `1⊥`. Because `1` is an eigenvector of `M`, `Q`
//! spans an invariant subspace of `M` and of `(M + λI)⁻¹`, so the contrasts'
//! Wald statistic under the covariance above is again the unpenalized one,
//! whatever `λ` is, on `L − 1` degrees of freedom. The level deviations
//! themselves are a variance-component block, reported with the smooth terms;
//! only the factor's joint test is parametric.

use crate::estimate::smooth_term_summary::SummaryBlockOffset;
use crate::estimate::summary::{
    ParametricPValueUnavailable, ParametricTermSummary, ParametricTermTest,
};
use crate::model_types::result_types::UnifiedFitResult;
use gam_linalg::triangular::{
    CholeskyGuard, cholesky_factor_in_place, cholesky_solve_matrix,
    forward_substitution_lower_vector,
};
use gam_math::probability::{
    chi_square_sf, fisher_snedecor_sf, normal_two_sided_probability,
    student_t_two_sided_probability,
};
use gam_terms::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, TermCollectionDesign,
    TermCollectionSpec,
};
use gam_spec::LikelihoodScaleMetadata;
use ndarray::{Array2, Axis, s};
use std::ops::Range;

/// The parametric tables of a model summary: one row per coefficient, and one
/// joint test per term.
#[derive(Clone, Debug)]
pub struct ParametricTermTables {
    /// Intercept and linear-term coefficients.
    pub coefficients: Vec<ParametricTermSummary>,
    /// One Wald test per parametric term: a factor with `L` levels is tested
    /// once on `L − 1` degrees of freedom, any other term on its own columns.
    pub term_tests: Vec<ParametricTermTest>,
}

/// One parametric term: its coefficient block and the label of each column.
struct ParametricTerm {
    name: String,
    /// The term's columns in the fit's coefficient layout.
    range: Range<usize>,
    column_labels: Vec<String>,
    /// A coefficient on a bound or inside a bounded geometry has no Wald
    /// sampling distribution at the boundary.
    bounded: bool,
    /// What the term's joint test is on.
    tested: TestedDirections,
}

/// The coefficient directions a parametric term's joint Wald test is on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TestedDirections {
    /// Every coefficient of the term, one row each in the coefficient table.
    Coefficients,
    /// The `L − 1` contrasts of a ridged fixed factor, whose level mean is
    /// aliased with the level carrier and set by the ridge (module doc). Its
    /// level deviations have no coefficient rows here.
    Contrasts,
}

/// The reference distribution of one parametric term's Wald statistics.
#[derive(Clone, Copy)]
enum WaldReference {
    /// Known scale: `N(0, 1)` for one coefficient, `χ²_q` for a block.
    Known,
    /// Estimated scale: Student-t / `F(q, ·)` on `residual_df`, with the
    /// statistic's covariance multiplied by `scale_ratio`, the term's scale
    /// over the fit's (1 unless the scale is the residual-sum-of-squares one;
    /// see [`ResidualScale`]). `residual_df` is `None` when the fit leaves
    /// none; `unavailable` names a reference that could not be formed at all.
    Estimated {
        residual_df: Option<f64>,
        scale_ratio: f64,
        unavailable: Option<ParametricPValueUnavailable>,
    },
}

/// How the fit's scale enters each parametric term's reference.
enum FitReference {
    Known,
    /// A scale estimated other than from the residual sum of squares (a Gamma
    /// shape, a Pearson or deviance dispersion): the reference is on the fit's
    /// Wald residual degrees of freedom `n − edf`.
    Estimated(Option<f64>),
    /// The profiled Gaussian scale, whose per-term residual degrees of freedom
    /// charge the smoothing parameters' uncertainty ([`ResidualScale`]).
    ResidualScale(Result<ResidualScale, ParametricPValueUnavailable>),
}

impl WaldReference {
    /// The Student-t / F denominator degrees of freedom, or why the estimated
    /// scale leaves none. A known scale has no residual degrees of freedom and
    /// never asks.
    fn residual_df(self) -> Result<f64, ParametricPValueUnavailable> {
        match self {
            Self::Known => Err(ParametricPValueUnavailable::NoResidualDegreesOfFreedom),
            Self::Estimated {
                unavailable: Some(reason),
                ..
            } => Err(reason),
            Self::Estimated { residual_df, .. } => {
                residual_df.ok_or(ParametricPValueUnavailable::NoResidualDegreesOfFreedom)
            }
        }
    }

    /// The residual degrees of freedom the payload publishes: those of an
    /// estimated-scale reference, absent on a known scale or when none exist.
    fn published_residual_df(self) -> Option<f64> {
        match self {
            Self::Known => None,
            Self::Estimated { residual_df, .. } => residual_df,
        }
    }
}

impl FitReference {
    fn of(design: &TermCollectionDesign, fit: &UnifiedFitResult, offset: SummaryBlockOffset) -> Self {
        if !fit.likelihood_scale.wald_scale_is_estimated() {
            Self::Known
        } else if matches!(fit.likelihood_scale, LikelihoodScaleMetadata::ProfiledGaussian) {
            Self::ResidualScale(ResidualScale::of(design, fit, offset))
        } else {
            Self::Estimated(fit.wald_residual_degrees_of_freedom())
        }
    }

    fn for_term(&self, range: &Range<usize>) -> WaldReference {
        match self {
            Self::Known => WaldReference::Known,
            Self::Estimated(residual_df) => WaldReference::Estimated {
                residual_df: *residual_df,
                scale_ratio: 1.0,
                unavailable: None,
            },
            Self::ResidualScale(Ok(scale)) => scale.for_term(range),
            Self::ResidualScale(Err(reason)) => WaldReference::Estimated {
                residual_df: None,
                scale_ratio: 1.0,
                unavailable: Some(*reason),
            },
        }
    }
}

/// The profiled Gaussian scale `σ̂² = RSS/(n − edf)` and the residual degrees
/// of freedom each parametric term's statistic is referred to.
///
/// `n − edf` is the residual degrees of freedom at fixed smoothing parameters,
/// but `ρ̂ = log λ̂` is itself fitted to the same residuals, and a fit that
/// spends its smoothing freedom on them leaves a smaller RSS than `n − edf`
/// accounts for: `σ̂²` is biased low and every estimated-scale reference built
/// on it is anti-conservative. The first-order (Wood–Pya–Säfken) charge for
/// that freedom is the corrected EDF `τ = edf + tr(X'WX · J V_ρ J')/s`, with
/// `J = dβ̂/dρ`, `J[:,k] = −H⁻¹λ_kS_k(β̂ − a)`, `V_ρ` the fit's
/// smoothing-parameter covariance and `s` the coefficient-covariance scale:
/// `RSS/(n − τ)` is unbiased at the null.
///
/// A term that carries its own ridge `λ_J` is tested by the statistic of the
/// fit with `J` unpenalized (module doc), which does not depend on `λ_J`, so
/// its residual degrees of freedom charge only the other smoothing parameters,
/// at their uncertainty given `ρ_J`: `τ_{−J} = edf + tr(G V_{o|J})` with
/// `G = J'X'WXJ/s` and `V_{o|J}` the covariance `V_ρ` conditioned on `ρ_J`
/// (which zeroes the rows and columns of `J`'s own coordinates). Charging `ρ_J`
/// as well would inflate the scale exactly when the term looks non-null — `λ̂_J`
/// is least certain there — and make its test conservative. A term without a
/// penalty of its own conditions on nothing: `τ_{−J} = τ`. The term is
/// referred to `n − τ_{−J}`, on the scale `σ̂²(n − edf)/(n − τ_{−J})`.
struct ResidualScale {
    n: f64,
    edf: f64,
    /// `G = J'X'WXJ/s`, one row and column per smoothing parameter.
    gram: Array2<f64>,
    /// `V_ρ`, aligned with `gram`.
    rho_covariance: Array2<f64>,
    /// The coefficient columns each smoothing parameter's penalty acts on.
    penalty_columns: Vec<Range<usize>>,
}

impl ResidualScale {
    fn of(
        design: &TermCollectionDesign,
        fit: &UnifiedFitResult,
        offset: SummaryBlockOffset,
    ) -> Result<Self, ParametricPValueUnavailable> {
        use ParametricPValueUnavailable::{
            NoResidualDegreesOfFreedom, SmoothingParameterUncertaintyUnavailable,
        };
        let edf = fit.edf_total().filter(|edf| edf.is_finite()).ok_or(NoResidualDegreesOfFreedom)?;
        let n = fit.training_sample_size() as f64;
        let m = fit.lambdas.len();
        // Every smoothing parameter's penalty must be read off `design`, the
        // one layout this summary knows, in `fit.lambdas` order.
        if offset.coefficients != 0 || offset.penalties != 0 || design.penalties.len() != m {
            return Err(SmoothingParameterUncertaintyUnavailable);
        }
        let penalty_columns: Vec<Range<usize>> =
            design.penalties.iter().map(|block| block.col_range.clone()).collect();
        if m == 0 {
            return Ok(Self {
                n,
                edf,
                gram: Array2::zeros((0, 0)),
                rho_covariance: Array2::zeros((0, 0)),
                penalty_columns,
            });
        }
        let rho_covariance = fit
            .artifacts
            .rho_covariance
            .as_ref()
            .filter(|cov| cov.dim() == (m, m) && cov.iter().all(|v| v.is_finite()))
            .ok_or(SmoothingParameterUncertaintyUnavailable)?
            .clone();
        let hessian = fit
            .saved_frame_penalized_hessian()
            .ok()
            .flatten()
            .ok_or(SmoothingParameterUncertaintyUnavailable)?;
        let weighted_gram = fit
            .saved_frame_weighted_gram()
            .ok()
            .flatten()
            .ok_or(SmoothingParameterUncertaintyUnavailable)?;
        let beta = fit
            .beta_from_gauge_shift()
            .map_err(|_| SmoothingParameterUncertaintyUnavailable)?;
        let covariance_scale = fit
            .coefficient_covariance_scale()
            .ok()
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or(SmoothingParameterUncertaintyUnavailable)?;
        let p = beta.len();
        if hessian.dim() != (p, p) || weighted_gram.dim() != (p, p) {
            return Err(SmoothingParameterUncertaintyUnavailable);
        }
        // `R[:,k] = λ_k S_k (β̂ − a)`, so `J = −H⁻¹R`.
        let mut penalty_gradients = Array2::<f64>::zeros((p, m));
        for (k, (block, lambda)) in design.penalties.iter().zip(&fit.lambdas).enumerate() {
            let cols = block.col_range.clone();
            let width = cols.len();
            if cols.end > p {
                return Err(SmoothingParameterUncertaintyUnavailable);
            }
            let local = if block.local.dim() == (width, width) {
                std::borrow::Cow::Borrowed(&block.local)
            } else {
                match &block.op {
                    Some(op) if op.dim() == width => std::borrow::Cow::Owned(op.as_dense()),
                    _ => return Err(SmoothingParameterUncertaintyUnavailable),
                }
            };
            let gradient = local.dot(&beta.slice(s![cols.clone()])) * *lambda;
            penalty_gradients.slice_mut(s![cols, k]).assign(&gradient);
        }
        let factor = cholesky_factor_in_place(hessian.view(), CholeskyGuard::FiniteStrict)
            .ok_or(SmoothingParameterUncertaintyUnavailable)?;
        let jacobian = cholesky_solve_matrix(&factor, &penalty_gradients);
        let gram = jacobian.t().dot(&weighted_gram.dot(&jacobian)) / covariance_scale;
        if gram.iter().any(|v| !v.is_finite()) {
            return Err(SmoothingParameterUncertaintyUnavailable);
        }
        Ok(Self {
            n,
            edf,
            gram,
            rho_covariance,
            penalty_columns,
        })
    }

    /// The reference of the term on `range`: `n − τ_{−J}` residual degrees of
    /// freedom and the scale ratio `(n − edf)/(n − τ_{−J})`.
    fn for_term(&self, range: &Range<usize>) -> WaldReference {
        // Condition `V_ρ` on the term's own coordinates one at a time. A
        // coordinate with no variance (a railed or unidentified `ρ_k`, which
        // the certified pseudoinverse leaves at exactly zero, with its row)
        // carries nothing to condition on.
        let mut conditional = self.rho_covariance.clone();
        for (k, cols) in self.penalty_columns.iter().enumerate() {
            if cols.is_empty() || cols.start < range.start || cols.end > range.end {
                continue;
            }
            let pivot = conditional[[k, k]];
            if pivot > 0.0 {
                let column = conditional.column(k).to_owned();
                let outer = column
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&column.view().insert_axis(Axis(0)));
                conditional.scaled_add(-1.0 / pivot, &outer);
            }
        }
        let charge = (&self.gram * &conditional).sum();
        let tau = self.edf + charge;
        let residual_df = self.n - tau;
        let scale_ratio = (self.n - self.edf) / residual_df;
        let valid = residual_df.is_finite() && residual_df > 0.0 && scale_ratio.is_finite();
        WaldReference::Estimated {
            residual_df: valid.then_some(residual_df),
            scale_ratio: if valid { scale_ratio } else { 1.0 },
            unavailable: None,
        }
    }
}

/// Build the parametric coefficient rows and term tests of a model summary.
///
/// Standard errors and the joint covariance start from the fit's display pair
/// (`UnifiedFitResult::display_coefficient_uncertainty`), so they carry one
/// recorded covariance definition (#2296). A term carrying its own ridge has
/// that prior's variance removed (module doc); its rows report the resulting
/// null sampling SD with `penalized` set. A multi-column linear term yields one
/// row per column, suffixed `[i]`; a constrained or bounded coefficient names
/// its geometry in the row label. A fixed factor contributes one joint test on
/// its `L − 1` contrasts and no coefficient rows. `offset` places `design`
/// inside the fit's coefficient and penalty layout, as for
/// [`super::smooth_term_summary_rows`]: every index into `fit` is global, every
/// index into `design` block-local.
pub fn parametric_term_summary_rows(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    offset: SummaryBlockOffset,
) -> ParametricTermTables {
    let terms = parametric_terms(design, spec, offset.coefficients);
    let uncertainty = fit.display_coefficient_uncertainty();
    let se = uncertainty.as_ref().map(|view| &view.standard_errors);
    let covariance = uncertainty.as_ref().and_then(|view| view.covariance);
    let fit_reference = FitReference::of(design, fit, offset);

    let mut coefficients = Vec::new();
    let mut term_tests = Vec::new();
    for term in &terms {
        let own_penalty = own_penalty(design, fit, offset, &term.range);
        let penalized = own_penalty.is_some();
        let reference = fit_reference.for_term(&term.range);
        let scale_ratio = match reference {
            WaldReference::Known => 1.0,
            WaldReference::Estimated { scale_ratio, .. } => scale_ratio,
        };
        let null_covariance = null_sampling_covariance(fit, covariance, &term.range, own_penalty)
            .map(|block| block * scale_ratio);
        let first = coefficients.len();
        let columns = match term.tested {
            TestedDirections::Coefficients => term.range.len(),
            TestedDirections::Contrasts => 0,
        };
        for (local, (idx, label)) in term.range.clone().zip(&term.column_labels).take(columns).enumerate() {
            let estimate = fit.beta.get(idx).copied().unwrap_or(f64::NAN);
            let std_error = if penalized {
                null_covariance
                    .as_ref()
                    .map(|block| block[[local, local]])
                    .filter(|variance| *variance >= 0.0)
                    .map(f64::sqrt)
            } else {
                se.and_then(|s| s.get(idx).copied())
                    .map(|s| s * scale_ratio.sqrt())
            };
            coefficients.push(coefficient_row(
                label.clone(),
                estimate,
                std_error,
                penalized,
                term.bounded,
                reference,
            ));
        }
        if term.name == "Intercept" {
            continue;
        }
        term_tests.push(term_test(
            term,
            &coefficients[first..],
            fit,
            null_covariance.as_ref(),
            reference,
        ));
    }
    ParametricTermTables {
        coefficients,
        term_tests,
    }
}

/// The intercept, the linear terms, and the fixed factors, in coefficient
/// order, each placed at `first_coefficient` in the fit's layout.
fn parametric_terms(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    first_coefficient: usize,
) -> Vec<ParametricTerm> {
    let global = |range: &Range<usize>| first_coefficient + range.start..first_coefficient + range.end;
    let mut terms = Vec::new();
    if !design.intercept_range.is_empty() {
        terms.push(ParametricTerm {
            name: "Intercept".to_string(),
            range: global(&design.intercept_range),
            column_labels: vec!["Intercept".to_string(); design.intercept_range.len()],
            bounded: false,
            tested: TestedDirections::Coefficients,
        });
    }
    for (name, range) in &design.linear_ranges {
        let meta = spec.linear_terms.iter().find(|term| term.name == *name);
        let label = linear_term_label(name, meta);
        let column_labels = (0..range.len())
            .map(|i| {
                if range.len() > 1 {
                    format!("{label}[{i}]")
                } else {
                    label.clone()
                }
            })
            .collect();
        let bounded = meta.is_some_and(|meta| {
            meta.coefficient_min.is_some()
                || meta.coefficient_max.is_some()
                || !matches!(
                    meta.coefficient_geometry,
                    LinearCoefficientGeometry::Unconstrained
                )
        });
        terms.push(ParametricTerm {
            name: name.clone(),
            range: global(range),
            column_labels,
            bounded,
            tested: TestedDirections::Coefficients,
        });
    }
    // A genuine random effect (`group(g)`, `re(g)`) is a variance component,
    // tested (or not) with the smooth terms. A fixed factor is parametric:
    // tested on its contrasts, since the intercept carries its level.
    for (re_idx, (name, range)) in design.random_effect_ranges.iter().enumerate() {
        let fixed_factor = spec
            .random_effect_terms
            .get(re_idx)
            .is_some_and(|meta| !meta.lenient_unseen);
        if !fixed_factor || range.is_empty() {
            continue;
        }
        terms.push(ParametricTerm {
            name: name.clone(),
            range: global(range),
            column_labels: Vec::new(),
            bounded: false,
            tested: TestedDirections::Contrasts,
        });
    }
    terms
}

/// The term's own penalty `Σ_k λ_k S_k` over the blocks that lie inside its
/// coefficient range, or `None` when it owns no penalty. From
/// `offset.penalties` on, `fit.lambdas` is laid out one entry per
/// `design.penalties` block, in order.
fn own_penalty(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    offset: SummaryBlockOffset,
    range: &Range<usize>,
) -> Option<Array2<f64>> {
    let q = range.len();
    let mut penalty = Array2::<f64>::zeros((q, q));
    let mut owned = false;
    let lambdas = fit.lambdas.iter().skip(offset.penalties);
    for (block, lambda) in design.penalties.iter().zip(lambdas) {
        let cols = offset.coefficients + block.col_range.start
            ..offset.coefficients + block.col_range.end;
        if cols.is_empty() || cols.start < range.start || cols.end > range.end {
            continue;
        }
        let at = cols.start - range.start;
        penalty
            .slice_mut(s![at..at + cols.len(), at..at + cols.len()])
            .scaled_add(*lambda, &block.local);
        owned = true;
    }
    owned.then_some(penalty)
}

/// The covariance a term's Wald statistic is scaled by: the display block for an
/// unpenalized term, and for a term carrying its own penalty `S_J` the
/// conditional block less its own prior,
/// `φ[H⁻¹(H − S_J)H⁻¹]_JJ = (V_c)_JJ − (V_c)_JJ S_J (V_c)_JJ / φ` with
/// `V_c = φH⁻¹` (see the module docs). `None` when an input is absent.
fn null_sampling_covariance(
    fit: &UnifiedFitResult,
    covariance: Option<&Array2<f64>>,
    range: &Range<usize>,
    own_penalty: Option<Array2<f64>>,
) -> Option<Array2<f64>> {
    let Some(own_penalty) = own_penalty else {
        return Some(covariance?.slice(s![range.clone(), range.clone()]).to_owned());
    };
    let conditional = fit
        .beta_covariance()?
        .slice(s![range.clone(), range.clone()])
        .to_owned();
    let phi = fit.dispersion_phi().ok().filter(|phi| *phi > 0.0)?;
    let own_prior = conditional.dot(&own_penalty).dot(&conditional) / phi;
    Some(conditional - own_prior)
}

/// One coefficient's Wald row.
fn coefficient_row(
    name: String,
    estimate: f64,
    std_error: Option<f64>,
    penalized: bool,
    bounded: bool,
    reference: WaldReference,
) -> ParametricTermSummary {
    let statistic = std_error
        .filter(|s| *s > 0.0)
        .map(|s| estimate / s)
        .filter(|z| z.is_finite());
    // Both tails come from the function that computes them, never from
    // `1 - CDF`, which saturates to p = 0 above |z| ≈ 8.3 (#2562).
    let residual_df = reference.residual_df();
    let pvalue = match (bounded, statistic, reference, residual_df) {
        (true, ..) | (_, None, ..) => None,
        (false, Some(z), WaldReference::Known, _) => Some(normal_two_sided_probability(z)),
        (false, Some(z), WaldReference::Estimated { .. }, Ok(df)) => {
            Some(student_t_two_sided_probability(z, df))
        }
        (false, Some(_), WaldReference::Estimated { .. }, Err(_)) => None,
    }
    .filter(|p| p.is_finite());
    let pvalue_unavailable = pvalue.is_none().then(|| {
        if bounded {
            ParametricPValueUnavailable::BoundedCoefficient
        } else if std_error.is_none() {
            ParametricPValueUnavailable::NoCovariance
        } else if statistic.is_none() {
            ParametricPValueUnavailable::SingularCovariance
        } else {
            match (reference, residual_df) {
                (WaldReference::Estimated { .. }, Err(reason)) => reason,
                _ => ParametricPValueUnavailable::SingularCovariance,
            }
        }
    });
    ParametricTermSummary {
        name,
        estimate,
        std_error,
        penalized,
        statistic,
        pvalue,
        pvalue_unavailable,
        residual_df: reference.published_residual_df(),
    }
}

/// The joint Wald test of one term's coefficients being all zero.
///
/// A one-column term's test is its coefficient row squared, with the same
/// p-value bit for bit (`t² ~ F(1, ν)`, `z² ~ χ²_1`), so the two tables never
/// disagree about the same hypothesis.
fn term_test(
    term: &ParametricTerm,
    rows: &[ParametricTermSummary],
    fit: &UnifiedFitResult,
    null_covariance: Option<&Array2<f64>>,
    reference: WaldReference,
) -> ParametricTermTest {
    let df = match term.tested {
        TestedDirections::Coefficients => term.range.len(),
        TestedDirections::Contrasts => term.range.len().saturating_sub(1),
    };
    let residual_df = reference.published_residual_df();
    if let [row] = rows {
        return ParametricTermTest {
            name: term.name.clone(),
            df,
            residual_df,
            statistic: row.statistic.map(|z| z * z),
            pvalue: row.pvalue,
            pvalue_unavailable: row.pvalue_unavailable,
        };
    }
    let unavailable = |reason| ParametricTermTest {
        name: term.name.clone(),
        df,
        residual_df,
        statistic: None,
        pvalue: None,
        pvalue_unavailable: Some(reason),
    };
    if term.bounded {
        return unavailable(ParametricPValueUnavailable::BoundedCoefficient);
    }
    // A one-level factor has no contrast to test.
    if df == 0 {
        return unavailable(ParametricPValueUnavailable::SingularCovariance);
    }
    let Some(block) = null_covariance else {
        return unavailable(ParametricPValueUnavailable::NoCovariance);
    };
    let beta = fit.beta.slice(s![term.range.clone()]).to_owned();
    let (beta, block) = match term.tested {
        TestedDirections::Coefficients => (beta, block.clone()),
        TestedDirections::Contrasts => {
            let basis = helmert_contrasts(term.range.len());
            (basis.t().dot(&beta), basis.t().dot(block).dot(&basis))
        }
    };
    let Some(factor) = cholesky_factor_in_place(block.view(), CholeskyGuard::FiniteStrict) else {
        return unavailable(ParametricPValueUnavailable::SingularCovariance);
    };
    let whitened = forward_substitution_lower_vector(&factor, beta.view());
    let wald = whitened.dot(&whitened);
    if !wald.is_finite() {
        return unavailable(ParametricPValueUnavailable::SingularCovariance);
    }
    let q = df as f64;
    let (statistic, pvalue) = match (reference, reference.residual_df()) {
        (WaldReference::Known, _) => (wald, chi_square_sf(wald, q)),
        (WaldReference::Estimated { .. }, Err(reason)) => {
            return ParametricTermTest {
                statistic: Some(wald / q),
                ..unavailable(reason)
            };
        }
        (WaldReference::Estimated { .. }, Ok(residual_df)) => {
            let f = wald / q;
            (f, fisher_snedecor_sf(f, q, residual_df))
        }
    };
    ParametricTermTest {
        name: term.name.clone(),
        df,
        residual_df,
        statistic: Some(statistic),
        pvalue: pvalue.is_finite().then_some(pvalue),
        pvalue_unavailable: (!pvalue.is_finite())
            .then_some(ParametricPValueUnavailable::SingularCovariance),
    }
}

/// An orthonormal basis of the contrasts of `levels` level effects, the
/// complement of the constant direction: the normalized Helmert columns, whose
/// `k`-th column compares level `k + 1` with the mean of the levels before it.
fn helmert_contrasts(levels: usize) -> Array2<f64> {
    let mut basis = Array2::<f64>::zeros((levels, levels.saturating_sub(1)));
    for k in 1..levels {
        let norm = ((k * (k + 1)) as f64).sqrt();
        basis.slice_mut(s![..k, k - 1]).fill(1.0 / norm);
        basis[[k, k - 1]] = -(k as f64) / norm;
    }
    basis
}

/// A linear term's row label, naming any coefficient bound or bounded geometry.
fn linear_term_label(name: &str, meta: Option<&LinearTermSpec>) -> String {
    let Some(meta) = meta else {
        return name.to_string();
    };
    let bounds = match (meta.coefficient_min, meta.coefficient_max) {
        (Some(lb), Some(ub)) => Some(format!("coef in [{lb:.3}, {ub:.3}]")),
        (Some(lb), None) => Some(format!("coef >= {lb:.3}")),
        (None, Some(ub)) => Some(format!("coef <= {ub:.3}")),
        (None, None) => None,
    };
    match &meta.coefficient_geometry {
        LinearCoefficientGeometry::Unconstrained => match bounds {
            Some(bounds) => format!("{name} [{bounds}]"),
            None => name.to_string(),
        },
        LinearCoefficientGeometry::Bounded { min, max, prior } => {
            let prior = match prior {
                BoundedCoefficientPriorSpec::None => "no-prior".to_string(),
                BoundedCoefficientPriorSpec::Shrinkage => "shrinkage(REML)".to_string(),
                BoundedCoefficientPriorSpec::Uniform => "Uniform(log-Jacobian)".to_string(),
                BoundedCoefficientPriorSpec::Beta { a, b } => format!("Beta({a:.3},{b:.3})"),
            };
            let bounds = bounds.map(|b| format!(", {b}")).unwrap_or_default();
            format!("{name} [bounded {min:.3}..{max:.3}, {prior}{bounds}]")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::helmert_contrasts;
    use ndarray::Array2;

    #[test]
    fn helmert_contrasts_are_an_orthonormal_basis_of_the_non_constant_directions() {
        for levels in 1..7 {
            let basis = helmert_contrasts(levels);
            assert_eq!(basis.dim(), (levels, levels - 1));
            let gram = basis.t().dot(&basis);
            let identity = Array2::<f64>::eye(levels - 1);
            assert!((&gram - &identity).iter().all(|e| e.abs() < 1e-14), "{gram:?}");
            let column_sums = basis.sum_axis(ndarray::Axis(0));
            assert!(column_sums.iter().all(|e| e.abs() < 1e-14), "{column_sums:?}");
        }
    }
}
