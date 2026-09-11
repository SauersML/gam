//! Penalized multi-output Bernoulli/binomial fitter at fixed λ.
//!
//! This is the row-diagonal sibling of [`crate::multinomial`]: the
//! same shared design `X ∈ ℝ^{N×P}` and shared penalty `S ∈ ℝ^{P×P}` are
//! reused across `K` independent bounded-link response columns. Per-column
//! smoothing parameters `λ_a` (length `K`) scale `S` independently for each
//! response. Because the Fisher information has no cross-column coupling
//! (`H_{n,a,b} = δ_{ab} · w_n · I_{n,a}`), the joint penalized
//! Hessian is block-diagonal in the `K` `P × P` per-response systems; the
//! shared [`crate::penalized_vector_glm`] engine factors that
//! block-diagonal Hessian in a single coupled damped-Newton loop, which is
//! mathematically identical to `K` independent per-column solves.
//!
//! # Fit problem
//!
//! Minimise the penalized negative log-likelihood
//!
//! ```text
//!   F(β) = − Σ_(n,a)∈C w_{n,a} [ y_{n,a} log μ_{n,a} + (1 − y_{n,a}) log(1 − μ_{n,a}) ]
//!           + ½ Σ_a λ_a · β_aᵀ S β_a
//! ```
//!
//! with `μ_{n,a} = g⁻¹(η_{n,a})`, `η_{n,a} = (X β_a)_n`. The per-column Fisher
//! step solves
//!
//! ```text
//!   (Xᵀ diag(1_C w_{n,a} I_{n,a}) X + λ_a S) δ_a
//!     = Xᵀ score_η − λ_a S β_a
//! ```
//!
//! where `I = (dμ/dη)² / (μ(1−μ))`. Logit, probit, complementary-log-log,
//! log-log, cauchit, and the parameterized bounded inverse links all use the
//! same cancellation-free natural-coordinate Bernoulli kernel as scalar GAMs.
//! The update is followed by a backtracking line search on `F` (full step first, halve up
//! to 8 times) so monotone descent is enforced even when the quadratic
//! model overshoots near saturation. This is precisely the shared
//! [`crate::penalized_vector_glm`] scaffold; this module supplies
//! only the row-diagonal binomial Fisher block, residual, and log-likelihood
//! via `BinomialMultiLikelihood`.
//!
//! # Relation to the multi-class softmax driver
//!
//! [`crate::multinomial::fit_penalized_multinomial`] handles the
//! coupled softmax Fisher block `H_{n,a,b} = w_n μ_{n,a} (δ_{ab} − μ_{n,b})`
//! and is the right entry when the user wants a single normalized
//! probability vector per row. This driver is the right entry when the
//! user has `K` independent binary marginals sharing a smooth basis (e.g.
//! multi-label classification, multi-trait penalised logistic regression
//! on a Duchon latent design). Both families are thin Fisher-block adapters
//! over the same `penalized_vector_glm` engine: the only difference is that
//! the softmax block is dense across outputs while these binomial columns are
//! row-diagonal.
//!
//! The function-boundary contract mirrors `fit_penalized_multinomial` so
//! the two are interchangeable at the FFI layer: same input arity, same
//! convergence semantics, same `(N, K)` fitted-probability output.

use crate::model_types::EstimationError;
use crate::penalized_vector_glm::{
    PenalizedVectorGlmInputs, VectorGlmSolve, fit_penalized_vector_glm,
};
use crate::vector_response::{VectorLikelihood, validate_vector_likelihood_inputs};
use gam_model_kernels::bernoulli_link::{
    bernoulli_natural_jet, bernoulli_natural_observation,
};
use gam_problem::{FixedLambdaSolverStage, SeparableCellMeasure};
use gam_spec::InverseLink;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

/// Inputs for [`fit_penalized_binomial_multi`].
#[derive(Debug, Clone)]
pub struct BinomialMultiFitInputs<'a> {
    /// Design matrix `X ∈ ℝ^{N×P}` (one row per observation, shared across
    /// all response columns).
    pub design: ArrayView2<'a, f64>,
    /// Multi-column binomial response `Y ∈ ℝ^{N×K}`. Each column is treated
    /// as an independent binomial response, so every entry must be a
    /// binomial proportion in `[0, 1]` (hard `{0, 1}` Bernoulli labels and soft
    /// proportions / probabilities alike). Entries outside `[0, 1]` are
    /// rejected because the per-entry log-likelihood is then unbounded in `η`.
    pub y: ArrayView2<'a, f64>,
    /// Bounded inverse link shared by all response columns. The likelihood is
    /// separable over output cells, but its link is part of the family rather
    /// than an implicit solver default.
    pub link: InverseLink,
    /// Optional known additive predictor offset, shape `(N, K)`. The fitted
    /// predictor is `η = Xβ + offset`; `None` means an exact zero offset.
    pub offset: Option<ArrayView2<'a, f64>>,
    /// Shared smoothing penalty `S ∈ ℝ^{P×P}` (symmetric, PSD).
    pub penalty: ArrayView2<'a, f64>,
    /// Per-response smoothing parameter `λ_a` (length `K`).
    pub lambdas: ArrayView1<'a, f64>,
    /// Structural cell activity and numerical likelihood weights. These are
    /// deliberately distinct: a not-at-risk/missing cell is absent, whereas a
    /// present cell may legitimately carry numerical weight zero.
    pub measure: SeparableCellMeasure<'a>,
    /// Optional per-row Fisher-block override, shape `(N, K, K)`. The `K`
    /// binomial columns are fit independently, so only the per-column diagonal
    /// `[n, a, a]` is consumed as the curvature `w_n I_{n,a}`;
    /// off-diagonals must be zero (enforced at the FFI boundary) since a
    /// non-zero cross term cannot be represented by the separable per-column
    /// solve. The gradient/residual path stays analytic — this is a
    /// curvature-only override (issue #349). Diagonal entries must be finite
    /// and non-negative.
    pub fisher_w_override: Option<ArrayView3<'a, f64>>,
    /// Maximum Newton iterations per response column; recommend 50.
    pub max_iter: usize,
    /// Relative-step convergence tolerance; recommend 1e-7.
    pub tol: f64,
}

/// Outputs of [`fit_penalized_binomial_multi`].
#[derive(Debug, Clone)]
pub struct BinomialMultiFitOutputs {
    /// Coefficient matrix, shape `(P, K)` (column `a` is `β_a`).
    pub coefficients: Array2<f64>,
    /// Fitted probabilities `μ_{n,a} = g⁻¹((X β_a)_n + offset_{n,a})`,
    /// shape `(N, K)`. Values are returned for every requested prediction cell;
    /// structural activity controls fitting, not whether a predictor is defined.
    pub fitted_probabilities: Array2<f64>,
    /// Number of joint Newton iterations executed (including the final step
    /// that satisfied the tolerance). The `K` columns share the design and
    /// are fitted by a single coupled damped-Newton loop over the
    /// block-diagonal penalized Hessian, so there is one iteration count for
    /// the whole solve.
    pub iterations: usize,
    /// Penalized negative log-likelihood at the returned `β̂`:
    /// `−log L(β̂) + ½ Σ_a λ_a · β̂_aᵀ S β̂_a`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)` for diagnostic reporting.
    pub deviance: f64,
}

/// Row-diagonal multi-output binomial likelihood adapter for the shared
/// [`crate::penalized_vector_glm`] engine.
///
/// The `K` response columns are mutually independent binomial marginals
/// sharing the design `X`, so the per-row Fisher block is **diagonal across
/// outputs**: `H_{n,a,b} = δ_{ab} · w_n · I_{n,a}`. The engine works in
/// `η = X β` space with `μ = g⁻¹(η)`; this adapter supplies the log-likelihood,
/// natural-coordinate score, and row-diagonal Fisher block.
struct BinomialMultiLikelihood<'a> {
    measure: SeparableCellMeasure<'a>,
    link: InverseLink,
}

impl BinomialMultiLikelihood<'_> {
    #[inline]
    fn active_weight(&self, row: usize, output: usize) -> Option<f64> {
        self.measure.active_weight(row, output)
    }
}

impl VectorLikelihood for BinomialMultiLikelihood<'_> {
    /// `Σ_(n,a)∈C w_{n,a} [ y_{n,a} log μ_{n,a} + (1 − y_{n,a}) log(1 − μ_{n,a}) ]`,
    /// evaluated through cancellation-free log-probability towers. No fitted
    /// probability is clamped, so value, score, and Fisher curvature remain
    /// derivatives of one likelihood even in representable link tails.
    fn log_lik(
        &self,
        eta: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<f64, EstimationError> {
        validate_vector_likelihood_inputs("BinomialMultiLikelihood::log_lik", eta, y, None)?;
        let (n, k) = eta.dim();
        let mut acc = 0.0_f64;
        for row in 0..n {
            for a in 0..k {
                let Some(w) = self.active_weight(row, a) else {
                    continue;
                };
                if w == 0.0 {
                    continue;
                }
                let observation =
                    bernoulli_natural_observation(row, y[[row, a]], eta[[row, a]], &self.link)?;
                acc += w * observation.log_likelihood;
            }
        }
        Ok(acc)
    }

    /// `∂ log L / ∂η`, evaluated from the link's log-probability derivatives.
    fn grad_eta(
        &self,
        eta: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        validate_vector_likelihood_inputs("BinomialMultiLikelihood::grad_eta", eta, y, None)?;
        let (n, k) = eta.dim();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            for a in 0..k {
                let Some(w) = self.active_weight(row, a) else {
                    continue;
                };
                if w == 0.0 {
                    continue;
                }
                let observation =
                    bernoulli_natural_observation(row, y[[row, a]], eta[[row, a]], &self.link)?;
                out[[row, a]] = w * observation.score;
            }
        }
        Ok(out)
    }

    /// Per-output Fisher curvature `1_C(n,a) w_{n,a} (dμ/dη)²/[μ(1−μ)]`.
    fn hess_diag(
        &self,
        eta: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        validate_vector_likelihood_inputs("BinomialMultiLikelihood::hess_diag", eta, y, None)?;
        let (n, k) = eta.dim();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            for a in 0..k {
                let Some(w) = self.active_weight(row, a) else {
                    continue;
                };
                if w == 0.0 {
                    continue;
                }
                let observation =
                    bernoulli_natural_observation(row, y[[row, a]], eta[[row, a]], &self.link)?;
                out[[row, a]] = (w.ln() + observation.log_fisher).exp();
            }
        }
        Ok(out)
    }

    /// Row-diagonal Fisher block `H_{n,a,b} = δ_{ab} · w_n I_{n,a}`.
    /// The independent columns have no cross-output coupling, so the off-diagonal
    /// entries are identically zero; lifting [`Self::hess_diag`] onto the per-row
    /// diagonal (the [`VectorLikelihood`] default) is exact here.
    fn hess_block(
        &self,
        eta: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        let diag = self.hess_diag(eta, y)?;
        let (n, k) = diag.dim();
        let mut out = Array3::<f64>::zeros((n, k, k));
        for row in 0..n {
            for a in 0..k {
                out[[row, a, a]] = diag[[row, a]];
            }
        }
        Ok(out)
    }
}

/// Fit `K` independent penalized binomial GLMs sharing the design `X`
/// and penalty `S`. See the module docs for the optimization problem.
pub fn fit_penalized_binomial_multi(
    inputs: BinomialMultiFitInputs<'_>,
) -> Result<BinomialMultiFitOutputs, EstimationError> {
    let BinomialMultiFitInputs {
        design,
        y,
        link,
        offset,
        penalty,
        lambdas,
        measure,
        fisher_w_override,
        max_iter,
        tol,
    } = inputs;

    // ──────────────────────── family-specific validation ───────────────────
    // The engine re-validates the shared geometry (nonempty design, penalty
    // shape, λ finiteness/non-negativity, override `(N, M, M)` shape, finite
    // design), but the binomial family owns three preconditions the generic
    // scaffold cannot know: the response must be a `[0, 1]` proportion, the
    // optional row weights must be finite and non-negative, and the optional
    // curvature override must be **row-diagonal** (independent columns carry no
    // cross-output coupling, so a non-zero off-diagonal cannot be represented).
    let n_obs = design.nrows();
    let (y_rows, k) = y.dim();
    if y_rows != n_obs {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: y rows {y_rows} ≠ design rows {n_obs}"
        );
    }
    if k == 0 {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: y must have at least one column (got K=0)"
        );
    }
    if lambdas.len() != k {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: lambdas length {} ≠ K = {k}",
            lambdas.len()
        );
    }
    measure.validate(n_obs, k).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "fit_penalized_binomial_multi: invalid separable response measure: {error}"
        ))
    })?;
    if let Some(fw) = fisher_w_override.as_ref() {
        if fw.dim() != (n_obs, k, k) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: fisher_w_override shape {:?} ≠ (N, K, K) = ({n_obs}, {k}, {k})",
                fw.dim()
            );
        }
        // Independent binomial columns have a strictly row-diagonal Fisher
        // block; a non-zero cross term `[n, a, b]` (a ≠ b) cannot be the
        // curvature of a separable per-column objective, so reject it rather
        // than silently couple the columns through the shared dense solve.
        for ((n_idx, a, b), &v) in fw.indexed_iter() {
            if !v.is_finite() || (a == b && v < 0.0) {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: fisher_w_override[{n_idx},{a},{b}] must be finite and its diagonal non-negative (got {v})"
                );
            }
            if a != b && v != 0.0 {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: fisher_w_override[{n_idx},{a},{b}] must be zero \
                     (independent columns have a row-diagonal Fisher block); got {v}"
                );
            }
            if a == b && measure.active_weight(n_idx, a).is_none() && v != 0.0 {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: fisher_w_override[{n_idx},{a},{a}] must be zero because the response cell is structurally inactive (got {v})"
                );
            }
        }
    }
    for ((i, j), &v) in y.indexed_iter() {
        if measure.active_weight(i, j).is_none() {
            continue;
        }
        // The per-entry objective y log μ + (1 − y) log(1 − μ) is the binomial
        // (Bernoulli / proportion) log-likelihood only when 0 ≤ y ≤ 1. Outside
        // that range it is unbounded above in η (e.g. y = 2 gives
        // 2η − log(1 + e^η) → ∞), so a finite-but-invalid entry would make the
        // stated likelihood not a binomial likelihood at all. Reject it here.
        if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: y[{i},{j}] must be a binomial proportion in [0,1] (got {v})"
            );
        }
    }

    // ─────────────────── shared penalized vector-GLM solve ─────────────────
    // Validate the family before entering the optimizer. The kernel repeats
    // the same refusal at evaluation boundaries, so no caller can bypass the
    // bounded-link contract with a direct likelihood invocation.
        bernoulli_natural_jet(0, 0.0, &link)?;
    let likelihood = BinomialMultiLikelihood {
        measure,
        link: link.clone(),
    };
    let solve = fit_penalized_vector_glm(
        PenalizedVectorGlmInputs {
            design,
            y,
            offset,
            penalty,
            lambdas,
            fisher_w_override,
            max_iter,
            tol,
            // Independent-binomial columns ARE genuinely independent outputs, so
            // the per-output Diagonal penalty is correct here (the #1587 Centered
            // metric is softmax-specific — there is no shared reference class).
            class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Diagonal,
            resume_from: None,
        },
        &likelihood,
        "fit_penalized_binomial_multi",
    )?;

    let fit = match solve {
        VectorGlmSolve::Converged(fit) => fit,
        VectorGlmSolve::Stalled(stall) => {
            // SPEC: a fit object must only ever come from a converged
            // optimization. Exhausting `max_iter` is a typed error carrying
            // evidence from the checkpoint, never an `Ok` with a flag.
            return Err(stall.into_nonconvergence_error(
                FixedLambdaSolverStage::BinomialMultiNewton,
                "fit_penalized_binomial_multi (fixed-λ vector-GLM damped Newton)",
            )?);
        }
    };

    let mut fitted = Array2::<f64>::zeros(fit.eta.dim());
    for ((row, output), value) in fitted.indexed_iter_mut() {
        *value = bernoulli_natural_jet(row, fit.eta[[row, output]], &link)?.mu;
    }

    Ok(BinomialMultiFitOutputs {
        coefficients: fit.coefficients,
        fitted_probabilities: fitted,
        iterations: fit.iterations,
        penalized_neg_log_likelihood: -fit.log_likelihood + fit.penalty_term,
        deviance: -2.0 * fit.log_likelihood,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::{IndexedCellSet, LikelihoodWeights, StructuralCells};
    use gam_spec::StandardLink;
    use ndarray::{Array1, Array3};

    fn logit_link() -> InverseLink {
        InverseLink::Standard(StandardLink::Logit)
    }

    fn logit_mu(eta: f64) -> f64 {
        bernoulli_natural_jet(0, eta, &logit_link())
            .expect("finite logit mean")
            .mu
    }

    fn toy_inputs() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 12;
        let p = 2;
        let k = 2;
        let design =
            Array2::<f64>::from_shape_fn(
                (n, p),
                |(i, j)| {
                    if j == 0 { 1.0 } else { ((i + 1) as f64).sin() }
                },
            );
        let y =
            Array2::<f64>::from_shape_fn((n, k), |(i, a)| if (i + a) % 2 == 0 { 1.0 } else { 0.0 });
        let penalty = Array2::<f64>::eye(p);
        let lambdas = Array1::<f64>::from_elem(k, 0.5);
        (design, y, penalty, lambdas)
    }

    #[test]
    fn fisher_override_none_reproduces_analytic_bit_for_bit() {
        // Issue #349: a None override must give exactly the analytic result.
        let (design, y, penalty, lambdas) = toy_inputs();
        let base = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        // Explicit None again — identical result.
        let again = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        for (a, b) in base.coefficients.iter().zip(again.coefficients.iter()) {
            assert_eq!(a, b, "None override must be deterministic");
        }
    }

    #[test]
    fn per_cell_offset_is_part_of_the_linear_predictor() {
        // A zero design makes the known offset the entire predictor. This
        // isolates the offset channel from coefficient estimation and proves
        // that different outputs on the same row may carry different known
        // exposures without being encoded as likelihood weights.
        let design = Array2::<f64>::zeros((4, 1));
        let y = ndarray::array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
        let offset = ndarray::array![[-2.0, 1.0], [-1.0, 0.5], [0.0, -0.5], [2.0, -1.0]];
        let penalty = Array2::<f64>::eye(1);
        let lambdas = ndarray::array![1.0, 1.0];
        let fit = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: Some(offset.view()),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 10,
            tol: 1.0e-12,
        })
        .expect("known-offset fit must converge");

        assert_eq!(fit.coefficients, Array2::<f64>::zeros((1, 2)));
        for ((row, output), &value) in fit.fitted_probabilities.indexed_iter() {
            assert_eq!(value, logit_mu(offset[[row, output]]));
        }
    }

    #[test]
    fn cloglog_offsets_return_exact_piecewise_hazard_probabilities() {
        let design = Array2::<f64>::zeros((4, 1));
        let y = ndarray::array![[0.0], [1.0], [0.0], [1.0]];
        let offset = ndarray::array![[-3.0], [-1.0], [0.0], [1.0]];
        let penalty = Array2::<f64>::eye(1);
        let lambdas = ndarray::array![1.0];
        let fit = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: InverseLink::Standard(StandardLink::CLogLog),
            offset: Some(offset.view()),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 10,
            tol: 1.0e-12,
        })
        .expect("cloglog known-offset fit");

        assert_eq!(fit.coefficients, Array2::<f64>::zeros((1, 1)));
        for row in 0..offset.nrows() {
            let expected = -(-offset[[row, 0]].exp()).exp_m1();
            assert!(
                (fit.fitted_probabilities[[row, 0]] - expected).abs() <= 4.0 * f64::EPSILON,
                "row {row}: cloglog mean must equal 1-exp(-exp(eta))"
            );
        }
    }

    #[test]
    fn active_zero_weight_cell_never_evaluates_an_impossible_tail() {
        let weights = ndarray::array![[0.0]];
        let measure = SeparableCellMeasure::new(
            StructuralCells::All,
            LikelihoodWeights::ByCell(weights.view()),
        );
        let link = InverseLink::Standard(StandardLink::CLogLog);
        let likelihood = BinomialMultiLikelihood {
            measure,
            link,
        };
        let eta = ndarray::array![[1_000.0]];
        let y = ndarray::array![[0.0]];
        assert_eq!(
            likelihood
                .log_lik(eta.view(), y.view())
                .expect("zero-weight value"),
            0.0
        );
        assert_eq!(
            likelihood
                .grad_eta(eta.view(), y.view())
                .expect("zero-weight score")[[0, 0]],
            0.0
        );
        assert_eq!(
            likelihood
                .hess_diag(eta.view(), y.view())
                .expect("zero-weight Fisher")[[0, 0]],
            0.0
        );
    }

    #[test]
    fn malformed_offset_is_rejected_at_the_shared_solver_boundary() {
        let (design, y, penalty, lambdas) = toy_inputs();
        let wrong_columns = Array2::<f64>::zeros((design.nrows(), y.ncols() + 1));
        let error = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: Some(wrong_columns.view()),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("an offset outside the active response geometry must fail");
        assert!(format!("{error}").contains("offset shape"));
    }

    #[test]
    fn structurally_inactive_cells_have_no_likelihood_score_or_curvature() {
        let excluded =
            IndexedCellSet::from_cells(2, 2, vec![(0, 1)]).expect("valid sparse exclusion set");
        let cell_weights = ndarray::array![[2.0, 7.0], [3.0, 5.0]];
        let measure = SeparableCellMeasure::new(
            StructuralCells::AllExcept(&excluded),
            LikelihoodWeights::ByCell(cell_weights.view()),
        );
        measure.validate(2, 2).expect("valid indexed measure");
        let link = logit_link();
        let likelihood = BinomialMultiLikelihood {
            measure,
            link,
        };
        let eta = ndarray::array![[0.2, -0.8], [0.5, 1.1]];
        let y = ndarray::array![[1.0, 1.0], [0.0, 1.0]];

        let gradient = likelihood
            .grad_eta(eta.view(), y.view())
            .expect("indexed score");
        let curvature = likelihood
            .hess_diag(eta.view(), y.view())
            .expect("indexed curvature");
        assert_eq!(gradient[[0, 1]], 0.0);
        assert_eq!(curvature[[0, 1]], 0.0);

        // Active cells carry the Bernoulli score and Fisher weight. Production
        // evaluates them through the stable natural-parameter observation (the
        // curvature in the log domain), so they agree with the textbook
        // `w·(y − μ)` and `w·μ(1 − μ)` to roundoff, not bit for bit.
        let agrees = |value: f64, reference: f64| {
            (value - reference).abs() <= 8.0 * f64::EPSILON * reference.abs().max(1.0)
        };
        for &(row, output) in &[(0usize, 0usize), (1, 0), (1, 1)] {
            let mu = logit_mu(eta[[row, output]]);
            let weight = cell_weights[[row, output]];
            let score = weight * (y[[row, output]] - mu);
            let fisher = weight * mu * (1.0 - mu);
            assert!(
                agrees(gradient[[row, output]], score),
                "cell ({row},{output}) score {} vs w(y-mu) {score}",
                gradient[[row, output]]
            );
            assert!(
                agrees(curvature[[row, output]], fisher),
                "cell ({row},{output}) curvature {} vs w mu(1-mu) {fisher}",
                curvature[[row, output]]
            );
        }
    }

    #[test]
    fn curvature_override_cannot_reintroduce_an_inactive_cell() {
        let (design, y, penalty, lambdas) = toy_inputs();
        let excluded =
            IndexedCellSet::from_cells(design.nrows(), y.ncols(), vec![(0usize, 1usize)])
                .expect("valid sparse exclusion set");
        let measure = SeparableCellMeasure::new(
            StructuralCells::AllExcept(&excluded),
            LikelihoodWeights::Uniform,
        );
        let mut override_blocks = Array3::<f64>::zeros((design.nrows(), y.ncols(), y.ncols()));
        for row in 0..design.nrows() {
            for output in 0..y.ncols() {
                override_blocks[[row, output, output]] = 0.25;
            }
        }
        let error = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure,
            fisher_w_override: Some(override_blocks.view()),
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("curvature on an absent response cell must fail");
        assert!(format!("{error}").contains("structurally inactive"));
    }

    #[test]
    fn exhausted_fixed_lambda_budget_is_typed_error_not_fit() {
        let (design, y, penalty, lambdas) = toy_inputs();
        let error = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 0,
            tol: 1.0e-9,
        })
        .expect_err("a zero-budget Newton solve must not mint a binomial fit");
        assert!(matches!(
            error,
            EstimationError::FixedLambdaNewtonDidNotConverge {
                objective_value,
                checkpoint,
                ..
            } if objective_value.is_finite()
                && checkpoint.stage() == FixedLambdaSolverStage::BinomialMultiNewton
                && checkpoint.completed_iterations() == 0
        ));
    }

    #[test]
    fn out_of_range_response_is_rejected() {
        // Issue #452: a finite but invalid entry (y = 2) makes the per-entry
        // binomial log-likelihood unbounded in η, so it must be rejected rather
        // than silently fit. The same guard covers negative entries.
        let (design, y, penalty, lambdas) = toy_inputs();
        let mut bad = y.clone();
        bad[[0, 0]] = 2.0;
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: bad.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("out-of-range response must error");
        assert!(format!("{err}").contains("binomial proportion in [0,1]"));

        let mut neg = y.clone();
        neg[[1, 1]] = -0.5;
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: neg.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("negative response must error");
        assert!(format!("{err}").contains("binomial proportion in [0,1]"));
    }

    #[test]
    fn fisher_override_shape_mismatch_is_rejected() {
        let (design, y, penalty, lambdas) = toy_inputs();
        let n = design.nrows();
        let k = y.ncols();
        let bad = Array3::<f64>::zeros((n, k + 1, k + 1));
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            link: logit_link(),
            offset: None,
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            measure: SeparableCellMeasure::uniform(),
            fisher_w_override: Some(bad.view()),
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("mismatched override shape must error");
        assert!(format!("{err}").contains("fisher_w_override shape"));
    }

    #[test]
    fn fisher_override_replaces_curvature_diagonal() {
        // A scaled curvature override changes the Newton step from β = 0:
        // with curvature scaled by α the first step is 1/α of the analytic
        // step (gradient unchanged), so the fitted β must differ from analytic.
        let (design, y, penalty, lambdas) = toy_inputs();
        let n = design.nrows();
        let k = y.ncols();
        // Analytic diagonal at β = 0 is μ(1−μ) = 0.25 for every column.
        let mut over = Array3::<f64>::zeros((n, k, k));
        for row in 0..n {
            for a in 0..k {
                over[[row, a, a]] = 0.25 * 4.0; // 4× the analytic curvature
            }
        }
        let link = logit_link();
        let likelihood = BinomialMultiLikelihood {
            measure: SeparableCellMeasure::uniform(),
            link,
        };
        let scaled = fit_penalized_vector_glm(
            PenalizedVectorGlmInputs {
                design: design.view(),
                y: y.view(),
                offset: None,
                penalty: penalty.view(),
                lambdas: lambdas.view(),
                fisher_w_override: Some(over.view()),
                max_iter: 1,
                tol: 1.0e-9,
                class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Diagonal,
                resume_from: None,
            },
            &likelihood,
            "binomial scaled-curvature first-step test",
        )
        .expect("scaled-curvature engine step must be finite");
        let analytic = fit_penalized_vector_glm(
            PenalizedVectorGlmInputs {
                design: design.view(),
                y: y.view(),
                offset: None,
                penalty: penalty.view(),
                lambdas: lambdas.view(),
                fisher_w_override: None,
                max_iter: 1,
                tol: 1.0e-9,
                class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Diagonal,
                resume_from: None,
            },
            &likelihood,
            "binomial analytic-curvature first-step test",
        )
        .expect("analytic-curvature engine step must be finite");
        let checkpoint_coefficients = |solve| match solve {
            VectorGlmSolve::Converged(fit) => fit.coefficients,
            VectorGlmSolve::Stalled(stall) => stall.coefficients,
        };
        let scaled = checkpoint_coefficients(scaled);
        let analytic = checkpoint_coefficients(analytic);
        let differs = scaled
            .iter()
            .zip(analytic.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-6);
        assert!(differs, "scaled curvature override must change the step");
    }
}
