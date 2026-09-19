//! Penalized score (Rao) lack-of-fit test for basis adequacy.
//!
//! # The question this answers
//!
//! A converged, certified GAM fit says nothing about whether the basis it was
//! given is rich enough to represent the function it was asked to model. The
//! two failure modes look identical from the optimizer's side: a smooth whose
//! basis spans the truth and a smooth whose basis cannot reach it both reach a
//! stationary REML point, both certify, and both report a per-term EDF that is
//! some fraction of the term's column count. What separates them is whether the
//! **residuals still carry structure in the term's own covariates**.
//!
//! This module tests exactly that. Given an *enrichment* design `Z` — a set of
//! higher-resolution directions over the term's covariates that the fitted
//! design `X` does not already span — it computes the penalized score statistic
//! for `H₀: γ = 0` in the augmented model `η = Xβ + Zγ`, evaluated at the fit's
//! own `β̂`. A significant statistic is a positive statement: *there is signal
//! in this smooth's covariates that its realized basis cannot represent.*
//!
//! # Why a score test and not an EDF-saturation rule
//!
//! The engine's `basis_is_saturated` predicate asks whether the term's
//! *penalized* EDF has reached its algebraic ceiling `realized_width −
//! nullspace_dim`. That fires only when λ has been driven to its floor and the
//! basis is exhausted. It cannot see a basis that is far too small while λ is
//! still binding, because basis size and λ both control smoothness and REML
//! trades them off against each other. On the #2774 fixture — a 16-D Duchon
//! smooth with `centers=24`, whose 17-column linear null space leaves a
//! penalized capacity of ~6 — the fit sits at penalized EDF 3.91, i.e. 65% of
//! capacity, so `basis_is_saturated` reports "certified" while the residual
//! confounding is large enough to produce a `6.2e-5` false association.
//!
//! Nor is local residual differencing (mgcv's `k.index`) a substitute. Measured
//! on that same fixture: the 16-D nearest-neighbour index reads `0.928` with a
//! randomization `p = 0.43`, and even an *oracle* ordering — sorting rows by the
//! true simulated confounder — only reaches `0.976`. Differencing throws away
//! the signal it is looking for whenever the missing component is a small
//! fraction of a Bernoulli residual variance. The score statistic below reads
//! `p = 9.5e-16` on the same fit, because its non-centrality grows like
//! `n × (explained variance fraction)` instead of being buried in local noise.
//!
//! # The statistic
//!
//! Write `s` for the working score, so that `∂ℓ/∂γ|_{γ=0} = Zᵀs`, and
//! `Var(s) = φ·W_F` (`W_F` the Fisher/score-side IRLS weights). Let
//! `G = XᵀW_H X` be the design's weighted Gram and
//!
//! ```text
//!     Z̃ = Z − X G⁻ XᵀW_H Z
//! ```
//!
//! the enrichment with the fitted design projected out **in the `W_H` metric**.
//! Then `Z̃ᵀW_H X = 0` exactly, and since the fit's error propagates into the
//! score only through `β̂ − β`,
//!
//! ```text
//!     U = Z̃ᵀ s(β̂) = Z̃ᵀ s(β) − Z̃ᵀW_H X(β̂ − β) = Z̃ᵀ s(β),
//!     Var(U) = φ · Z̃ᵀ W_F Z̃ =: φ · V,     T = Uᵀ V⁻ U / φ.
//! ```
//!
//! `T` is referred to `χ²_r` when the dispersion is known, the same
//! `Known`/`Estimated` split as [`crate::inference::smooth_test`]. When the
//! dispersion is estimated from the fit's `ν` residual d.f., the residual sum
//! `ν·φ̂` already contains the tested directions' share `T·φ̂`. So the
//! reference is the added-variable `F`:
//!
//! ```text
//!     F = (T / r) / ((ν − T) / (ν − r))   on (r, ν − r) d.f.,
//! ```
//!
//! whose denominator is the part of the residual sum the numerator does not
//! use. It is exact for an unpenalized Gaussian fit; `T/r` against `F(r, ν)`
//! would divide by a scale containing its own numerator and read conservative.
//!
//! # Why the UNPENALIZED Gram, and not `H⁻¹`
//!
//! The obvious construction projects with the fit's own penalized Hessian
//! `H⁻¹ = (XᵀW_H X + S_λ)⁻¹`, which is what the first-order expansion of a
//! penalized score test hands you. It is wrong here, for a reason that is about
//! the QUESTION and not about the algebra.
//!
//! A penalized fit is biased: `E[β̂] − β ≈ −H⁻¹S_λβ`, so the residuals carry a
//! systematic component `W_H X H⁻¹S_λβ` that lives inside `span(X)`. Under the
//! `H⁻¹` projection `Z̃ᵀW_H X = ZᵀW_H X H⁻¹S_λ ≠ 0`, and that component leaks
//! straight into `E[U]`, so the statistic is non-central under `H₀` by an amount set by
//! how hard λ is shrinking — it reports "λ is doing work", which is true of
//! every GAM ever fitted and grows with `n`. Projecting in the `W_H` metric
//! annihilates it exactly, because the entire shrinkage bias lies in `span(X)`.
//!
//! The semantic statement is the same one: with `G⁻`, `T` tests only directions
//! the realized design **cannot represent at all**. Shrinking a direction the
//! basis HAS is a smoothing-parameter question, not a basis-size one, and this
//! statistic deliberately declines to answer it. The invariance
//! `Z → Z + X·A ⟹ T unchanged` (pinned in
//! `statistic_is_invariant_to_shifting_the_enrichment_by_design_columns`) is the
//! executable form of that contract; the `H⁻¹` projection does not satisfy it.
//!
//! `V` is accumulated as the Gram `Z̃ᵀW_F Z̃` of the residualized enrichment
//! rather than as the algebraically equal Schur complement
//! `ZᵀW_F Z − ZᵀW_F X G⁻ XᵀW_F Z`. The Gram form is PSD to machine precision and
//! its small eigenvalues are genuinely small instead of being the residue of a
//! subtraction — which matters exactly when part of the enrichment is nearly
//! inside `span(X)`, i.e. always.
//!
//! # Which directions count, and why the question is a ratio
//!
//! `r` is the number of enrichment directions the fitted design cannot
//! represent, and deciding that is not a question about how LARGE a residual
//! eigenvalue is. The enrichment is a radial kernel design, so its residual
//! spectrum is a Karhunen–Loève tail: it decays geometrically, with no gap
//! anywhere for an absolute threshold to sit in, and it decays FASTER the more
//! centers the alternative has. Every absolute threshold therefore truncates
//! harder as the alternative gets wider — the opposite of what a wider
//! alternative is for.
//!
//! What is scale-free is the fraction of a direction's own energy the
//! projection left behind. Those fractions are the generalized eigenvalues of
//! `(V, E)` with `E = ZᵀW_F Z` — `sin²` of the principal angles when the two
//! row metrics agree, and still the relevant retained-energy ratios when they do
//! not. A direction the design cannot reach keeps a finite ratio no matter how
//! little absolute energy it carries; one the design spans exactly has only
//! projection roundoff left.
//!
//! The computation is deliberately whitened from `V`, not from `E`. Whitening
//! `E` first must decide the numerical rank of the RAW radial-kernel Gram against
//! its largest, low-frequency direction. That silently discards the fine tail
//! before asking whether the fitted design represents it — the same shared-scale
//! error as #2788/#2789 in another coordinate system. Whitening `V` first uses
//! the numerical rank of the covariance the score statistic actually inverts;
//! the raw Gram then serves only to reject projection dust through a reciprocal
//! generalized-energy test. Both tolerances are derived from matrix dimension
//! and `f64::EPSILON`, so widening the alternative cannot move an unrelated
//! hard-coded floor through its spectrum.
//!
//!
//!
//! # It is exact on a row subset, and that is a cost lever
//!
//! Everything the construction rests on — `Z̃ᵀW_H X_S = 0`, `Var(s_S) = φ·W_F,S`
//! — is a property of the SELECTED sub-design, not of the whole sample. So a
//! caller may compute the report on any fixed subset of rows and the reference
//! law is unchanged; the only thing a subset costs is non-centrality, which
//! grows linearly in the row count. That matters because this check runs on
//! every fit: it lets a caller bound its cost by `m` instead of `n` without
//! weakening or approximating anything, and
//! `a_row_subset_gives_the_same_answer_as_the_subset_design` is the executable
//! statement of it.
//! # It does not assume the fit solved an unmodified score equation
//!
//! `U = Z̃ᵀ s(β̂) = Z̃ᵀ s(β) − Z̃ᵀ W_H X (β̂ − β)`, and the second term vanishes
//! because `Z̃ᵀW_H X = 0` — for ANY `β̂`, not just the one a plain penalized
//! IRLS produces. A Firth/Jeffreys-adjusted fit, which solves
//! `Xᵀs − S_λβ̂ + ∂ log|I|/∂β = 0` rather than `Xᵀs = S_λβ̂`, is therefore
//! handled with no special case: the adjustment moves `β̂`, and the projection
//! removes whatever `β̂` does. The `H⁻¹`-projected variant has no such property,
//! since its correction term is a specific function of the score equation it
//! assumed.
//!
//! # The conditional reference for canonical families
//!
//! For a Gaussian response the score is exactly normal and the `χ²`/`F`
//! references above are exact. For a binomial or Poisson response they are
//! only first-order, and the first-order error is not small and not
//! symmetric: at `n = 200` with a 24-direction enrichment the product
//! statistic read KS `p = 0.0045` (binomial) and `1.4e-7` (Poisson) against
//! uniformity, with sizes `0.030 / 0.0027` at nominal `0.05 / 0.01` — a
//! conservative test, which is as much a miscalibration as an
//! anti-conservative one. The cause is that the score is evaluated at an
//! estimated `β`: its conditional mean is not zero, its covariance is not `I`,
//! and its fourth cumulant is not the Gaussian one, each at `O(1/n)` per
//! direction and so `O(r/n)` on the statistic — the same order as its
//! deviation from `r`.
//!
//! A canonical link removes the nuisance exactly. `Xᵀ(w∘y)` is sufficient for
//! `β` under `H₀`, so the law of the data GIVEN it does not depend on `β` at
//! all; and the unpenalized null MLE `μ̃` is a function of it, so every weight
//! built from `μ̃` — `W = wV`, `W₁ = wκ₃`, `W₂ = wκ₄`, the projection, the
//! whitening, `L = Z̃K` — is a constant under that conditioning. The test
//! refers the score `u = Lᵀ w(y − μ̃)` to its conditional law, expanded
//! (the double-saddlepoint cumulant expansion of the score given the
//! sufficient statistic) to the order the unconditional reference misses.
//! With `H = X G⁻ Xᵀ`, `h = diag(H)`, `A = LLᵀ`:
//!
//! ```text
//!     E[u | ·]   = δ = −½ Lᵀ(W₁∘h)
//!     Cov[u | ·] = Σ = I − ½(D₁ − D₂ − D₃)
//!                  D₁ = Lᵀ diag(W₂∘h) L,  D₂ = Lᵀ diag(W₁∘H(W₁∘h)) L,
//!                  D₃ = (W₁∘L)ᵀ (H∘H) (W₁∘L)
//!     K₄ = Σ W₂ A_nn² − 2 Σ W₁ₙW₁ₙ′ A_nn′² H_nn′ − (W₁∘diag A)ᵀ H (W₁∘diag A)
//! ```
//!
//! `T_c = (u − δ)ᵀΣ⁻¹(u − δ)` then has conditional mean `r` and variance
//! `2r + K₄` to this order, and `c·χ²_{r/c}` with `c = 1 + K₄/(2r)` is the
//! scaled `χ²` with those two moments — so the tail is matched in both
//! directions rather than bounded in one. Measured at `n = 200`, 10 000
//! replicates: binomial KS `p = 0.14`, sizes `0.097 / 0.049 / 0.0094` at
//! `0.1 / 0.05 / 0.01`; Poisson KS `p = 0.82`. The Rust tests
//! `conditional_test_is_uniform_under_the_null_*` pin both tails.
//!
//! The expansion is at the UNPENALIZED null MLE on the test's rows, not at the
//! penalized fit: the penalized `β̂` is not the conditional law's natural
//! centre, and the projection annihilates everything in `span(X)` anyway, so
//! the penalty's only effect on the score would be the shrinkage bias this
//! module exists to exclude. Separation needs no special case: the null MLE
//! then sits on a face, the separated rows' weights vanish, and their
//! responses — fixed by the conditioning — contribute nothing.
//!
//! The expansion's small parameter is, row by row, the leverage `W h` times
//! the standardized cumulants `κ₃/V^{3/2}`, `κ₄/V²` — and those grow like
//! `1/V` where `μ̃` is extreme. A row that is both high-leverage and near a
//! boundary (the edge of a polynomial-like design with `μ̃` near 0 or 1, or a
//! low Poisson count) can therefore push `Σ` off positive definiteness or
//! `c` below zero. That is the expansion saying it does not describe the
//! conditional law there, and the test reports "not measured" rather than a
//! number: 20 of 4000 null replicates in the binomial unit-test harness. The
//! same mechanism is the open accuracy limit of this reference: with Poisson
//! means `0.14–0.82` at `n = 200` and 24 null-plus-enrichment directions, 19%
//! of null replicates are refused and the measured ones read KS `p ≈ 0.01`
//! over 8000 — resolving that regime needs the next order of the expansion,
//! not a tolerance.
//!
//! # What it does not claim
//!
//! `λ̂` is held at its fitted value and the enrichment is a fixed alternative,
//! so `T` is conditional on both, exactly as the summary table's Wald statistic
//! is conditional on `λ̂`. A rejection says there is signal in this smooth's
//! covariates outside its realized column span; it does not say how much of the
//! *estimand* that signal moves. The caller pairs it with the term's
//! EDF-vs-capacity evidence and reports both.

use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use gam_math::probability::{chi_square_sf, fisher_snedecor_sf};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub use crate::inference::smooth_test::SmoothTestScale;

/// Inputs to [`basis_adequacy_score_test`].
///
/// Every matrix is in the fit's own coefficient/row layout. `design`,
/// `hessian_weights`, `score_weights` and `score` share the fit's row order;
/// `enrichment` must be evaluated at those same rows.
pub struct BasisAdequacyInput<'a> {
    /// `Z` — the enrichment design (`m × q`): higher-resolution directions over
    /// the tested term's covariates. Columns already inside `span(X)` are
    /// harmless; they leave the estimable rank rather than biasing it.
    pub enrichment: ArrayView2<'a, f64>,
    /// `X_S` — the fitted design on the SAME `m` rows, in the same coefficient
    /// frame as `design_gram`.
    ///
    /// Every array here is on the rows the test is computed on, which need not
    /// be all of them. The statistic is EXACT on any fixed subset: the
    /// identities it rests on — `Z̃ᵀW_H X_S = 0` and `Var(s_S) = φ·W_F,S` — are
    /// properties of the selected sub-design, not of the whole sample. Choosing
    /// the subset is the caller's job because only the caller knows what the
    /// test is FOR and what it may cost.
    pub design: ArrayView2<'a, f64>,
    /// `W_H` — the diagonal curvature weights the fit's penalized Hessian was
    /// assembled from (observed information where the fit tracked it), on the
    /// same `m` rows. Used only to build the projection.
    pub hessian_weights: ArrayView1<'a, f64>,
    /// `W_F` — the Fisher/score-side IRLS weights, i.e. `Var(s) = φ·W_F`.
    /// Equal to `hessian_weights` for a canonical link.
    pub score_weights: ArrayView1<'a, f64>,
    /// `s` — the per-row working score, `sᵢ = wᵢ(yᵢ − μ̂ᵢ)(dμ/dη)ᵢ / V(μ̂ᵢ)`, so
    /// that `U = Zᵀ s` is the score for the enrichment coefficients.
    pub score: ArrayView1<'a, f64>,
    /// Factored `G = XᵀW_H X` — the design's weighted Gram, **without** the
    /// penalty and **without** dispersion scaling. Factored once by the caller
    /// and reused across the model's smooth terms; see [`DesignGramFactor`].
    /// The module header explains why this is the unpenalized Gram and not the
    /// penalized Hessian.
    pub design_gram: &'a DesignGramFactor,
    /// `φ̂` — the fitted dispersion. `1.0` for families that carry their
    /// dispersion inside the IRLS weight.
    pub dispersion: f64,
    /// `ν` — the residual d.f. `φ̂` was estimated on. The `Estimated`-scale
    /// reference is `F(r, ν − r)`, since `r` of those d.f. carry the tested
    /// directions. Ignored on the `Known` branch.
    pub residual_df: Option<f64>,
    pub scale: SmoothTestScale,
}

/// Outcome of the penalized score lack-of-fit test.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisAdequacyResult {
    /// `T = Uᵀ V⁻ U / φ̂`.
    pub statistic: f64,
    /// Number of estimable enrichment directions actually summed — the
    /// reference d.f. This is the enrichment width MINUS whatever part of it the
    /// fitted design already spanned, so it reports how much genuinely new
    /// resolution the alternative carried.
    pub rank: usize,
    /// `P(χ²_rank > T)`, or, when the scale is estimated, the added-variable
    /// `F(rank, ν − rank)` tail at `(T/rank)·(ν − rank)/(ν − T)`.
    pub p_value: f64,
}

/// Penalized score (Rao) test of `H₀: γ = 0` in `η = Xβ + Zγ` at the fit's `β̂`.
///
/// Returns `None` — never a stand-in value — when the inputs cannot support the
/// test: mismatched shapes, a non-finite entry anywhere in the assembled
/// quadratic form, a non-positive dispersion, no estimable enrichment direction
/// left after projection, or an `Estimated` scale with no usable residual d.f.
/// (none, `ν ≤ rank`, or `T ≥ ν`).
/// An absent verdict is a caller-visible "not measured", which is the only
/// honest report when the geometry is missing.
pub fn basis_adequacy_score_test(input: BasisAdequacyInput<'_>) -> Option<BasisAdequacyResult> {
    let m = input.design.nrows();
    let p = input.design.ncols();
    let q = input.enrichment.ncols();
    if m == 0
        || p == 0
        || q == 0
        || input.enrichment.nrows() != m
        || input.hessian_weights.len() != m
        || input.score_weights.len() != m
        || input.score.len() != m
        || input.design_gram.dimension() != p
        || !(input.dispersion.is_finite() && input.dispersion > 0.0)
    {
        return None;
    }

    let geometry = enrichment_geometry(
        input.enrichment,
        input.design,
        input.hessian_weights,
        input.score_weights,
        input.score,
        input.design_gram,
    )?;
    let rank = geometry.basis.ncols();
    let statistic = geometry.projected.iter().map(|value| value * value).sum::<f64>();
    let statistic = statistic / input.dispersion;
    if !statistic.is_finite() || statistic < 0.0 {
        return None;
    }

    let reference_df = rank as f64;
    let p_value = match input.scale {
        SmoothTestScale::Known => chi_square_sf(statistic, reference_df),
        SmoothTestScale::Estimated => {
            let residual_df = input
                .residual_df
                .filter(|value| value.is_finite() && *value > 0.0)?;
            // `φ̂` was estimated from the fit's residual sum `ν·φ̂`, and the
            // residual's projection onto the tested directions, `T·φ̂` on `r`
            // d.f., is part of that sum. `T/r` against `F(r, ν)` therefore divides
            // by a scale that contains its own numerator: the ratio is
            // `(ν/r)·Beta(r/2, (ν − r)/2)`, bounded and conservative at every
            // level. The scale that is independent of the numerator is the rest
            // of the sum, `(ν − T)·φ̂` on `ν − r` d.f., which gives the classical
            // added-variable `F` (exact for an unpenalized Gaussian fit). With
            // `ν ≤ r` or `T ≥ ν`, nothing is left to estimate that scale from.
            let independent_df = residual_df - reference_df;
            let independent_sum = residual_df - statistic;
            if !(independent_df > 0.0 && independent_sum > 0.0) {
                return None;
            }
            let f_statistic = (statistic / reference_df) / (independent_sum / independent_df);
            fisher_snedecor_sf(f_statistic, reference_df, independent_df)
        }
    };
    if !p_value.is_finite() {
        return None;
    }
    Some(BasisAdequacyResult {
        statistic,
        rank,
        p_value,
    })
}

/// A canonical-link exponential family whose sufficient statistic `Xᵀ(w∘y)`
/// the conditional test in [`conditional_basis_adequacy_test`] conditions on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalExponentialFamily {
    /// `y` a proportion in `[0, 1]`, prior weight the trial count;
    /// `b(η) = log(1 + eᶯ)`.
    BinomialLogit,
    /// `y` a count, `b(η) = eᶯ`.
    PoissonLog,
}

/// The unit cumulant function and its first four derivatives at `η`.
struct UnitCumulants {
    /// `b(η)`.
    log_partition: f64,
    /// `μ = b′(η)`.
    mean: f64,
    /// `V = b″(η)`.
    variance: f64,
    /// `κ₃ = b‴(η)`.
    third: f64,
    /// `κ₄ = b⁗(η)`.
    fourth: f64,
}

impl CanonicalExponentialFamily {
    fn cumulants(self, eta: f64) -> UnitCumulants {
        match self {
            Self::BinomialLogit => {
                // Every quantity from `e^{−|η|}`, so none of them cancels:
                // `μ(1 − μ)` formed from a rounded `μ ≈ 1` loses all its digits
                // exactly where a well-separated row puts it.
                let decay = (-eta.abs()).exp();
                let log_partition = eta.max(0.0) + decay.ln_1p();
                let mean = if eta >= 0.0 {
                    1.0 / (1.0 + decay)
                } else {
                    decay / (1.0 + decay)
                };
                let variance = decay / ((1.0 + decay) * (1.0 + decay));
                // `1 − 2μ = −tanh(η/2)`.
                let skew = -(0.5 * eta).tanh();
                UnitCumulants {
                    log_partition,
                    mean,
                    variance,
                    third: variance * skew,
                    fourth: variance * (1.0 - 6.0 * variance),
                }
            }
            Self::PoissonLog => {
                let mean = eta.exp();
                UnitCumulants {
                    log_partition: mean,
                    mean,
                    variance: mean,
                    third: mean,
                    fourth: mean,
                }
            }
        }
    }

    fn admits_response(self, response: f64) -> bool {
        match self {
            Self::BinomialLogit => (0.0..=1.0).contains(&response),
            Self::PoissonLog => response.is_finite() && response >= 0.0,
        }
    }
}

/// The unpenalized maximum-likelihood fit of the null model `η = Xβ + offset`
/// on the test's rows, and the cumulant weights the conditional reference law
/// is built from. Produced by [`canonical_null_fit`].
pub struct CanonicalNullFit {
    /// `W = w·V(μ̃)`: Fisher and Hessian weight at once (canonical link).
    weights: Array1<f64>,
    /// `W₁ = w·κ₃(μ̃)`.
    third: Array1<f64>,
    /// `W₂ = w·κ₄(μ̃)`.
    fourth: Array1<f64>,
    /// `s = w∘(y − μ̃)`, the score of the null model's log-likelihood in `η`.
    score: Array1<f64>,
    /// `G = XᵀWX` factored at `μ̃`.
    gram: DesignGramFactor,
    /// `X̄` with `X̄X̄ᵀ = X G⁻ Xᵀ`, from the same factor.
    whitened: Array2<f64>,
}

/// The unpenalized MLE of the null model on the given rows, by damped Newton in
/// the linear predictor.
///
/// The iteration lives in `η`, not `β`: starting from any `η₀` in the model's
/// affine span (the penalized fit's own linear predictor, offset included, is
/// the natural one), each step `η ← η + t·X G⁻ Xᵀ w(y − μ)` stays in that span,
/// so no coefficient frame and no offset has to be carried. A step is taken
/// only if it STRICTLY lowers `f(η) = Σ w(b(η) − yη)`; otherwise `t` halves.
///
/// Converged is a certificate, not a count. The Newton decrement
/// `λ² = gᵀG⁻g` is twice the predicted remaining decrease, and once that is
/// inside the rounding band of `f` itself — `γ_{m+p²}·Σ|w b(η)| + |w y η|`,
/// the same band P-IRLS certifies against — no further step can be told apart
/// from roundoff. There is no iteration cap: every accepted step strictly
/// decreases a floating-point value that is bounded below, so the loop ends,
/// and it ends either certified or with a step so small that `η + t·Δη`
/// rounds back to `η`, which is reported as `None`.
///
/// An MLE that does not exist in the interior (separation) is not special-cased.
/// The iterates walk the separated rows to the boundary, where their weight and
/// score vanish; what remains is the MLE of the face model, and the conditional
/// law on the separated rows is degenerate (their responses are fixed by the
/// conditioning), which is exactly what zero weight encodes.
pub fn canonical_null_fit(
    design: ArrayView2<'_, f64>,
    response: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    start_eta: ArrayView1<'_, f64>,
    family: CanonicalExponentialFamily,
) -> Option<CanonicalNullFit> {
    let m = design.nrows();
    let p = design.ncols();
    if m == 0
        || p == 0
        || response.len() != m
        || prior_weights.len() != m
        || start_eta.len() != m
        || response.iter().any(|&y| !family.admits_response(y))
        || prior_weights.iter().any(|&w| !(w.is_finite() && w >= 0.0))
        || start_eta.iter().any(|eta| !eta.is_finite())
        || design.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let band_growth = gam_linalg::roundoff::accumulation_growth(m + p * p);
    // `(f(η), Σ|terms|)`, or `None` when a term is not finite (an overflowing
    // Poisson mean), which the line search treats as "not a decrease".
    let objective = |eta: &Array1<f64>| -> Option<(f64, f64)> {
        let mut value = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for row in 0..m {
            let weight = prior_weights[row];
            if weight == 0.0 {
                continue;
            }
            let partition = weight * family.cumulants(eta[row]).log_partition;
            let linear = weight * response[row] * eta[row];
            value += partition - linear;
            magnitude += partition.abs() + linear.abs();
        }
        (value.is_finite() && magnitude.is_finite()).then_some((value, magnitude))
    };
    let mut eta = start_eta.to_owned();
    let (mut value, mut magnitude) = objective(&eta)?;
    loop {
        let mut weights = Array1::<f64>::zeros(m);
        let mut third = Array1::<f64>::zeros(m);
        let mut fourth = Array1::<f64>::zeros(m);
        let mut score = Array1::<f64>::zeros(m);
        for row in 0..m {
            let weight = prior_weights[row];
            let unit = family.cumulants(eta[row]);
            weights[row] = weight * unit.variance;
            third[row] = weight * unit.third;
            fourth[row] = weight * unit.fourth;
            score[row] = weight * (response[row] - unit.mean);
        }
        let gram = DesignGramFactor::new(weighted_gram(design, weights.view())?.view())?;
        let gradient = design.t().dot(&score);
        let direction = gram.solve(&gradient.clone().insert_axis(ndarray::Axis(1)))?;
        let direction = direction.column(0).to_owned();
        let decrement = gradient.dot(&direction);
        if !decrement.is_finite() {
            return None;
        }
        if decrement <= 2.0 * band_growth * magnitude {
            let whitened = gram.whiten(design)?;
            return Some(CanonicalNullFit {
                weights,
                third,
                fourth,
                score,
                gram,
                whitened,
            });
        }
        let step = design.dot(&direction);
        let mut length = 1.0_f64;
        loop {
            let candidate = &eta + &(length * &step);
            if candidate == eta {
                return None;
            }
            if let Some((next_value, next_magnitude)) = objective(&candidate)
                && next_value < value
            {
                eta = candidate;
                value = next_value;
                magnitude = next_magnitude;
                break;
            }
            length *= 0.5;
        }
    }
}

/// Conditional score test of `H₀: γ = 0` in `η = Xβ + Zγ` for a
/// canonical-link binomial or Poisson model, given the null model's sufficient
/// statistic `Xᵀ(w∘y)`.
///
/// Conditioning on `Xᵀ(w∘y)` removes the nuisance `β` exactly: under `H₀` the
/// conditional law of the data does not depend on it. The statistic is the
/// score `u = LᵀS`, `L = Z̃K` the whitened residualized enrichment at the
/// unpenalized null MLE `μ̃` and `S = w∘(y − μ̃)`, and its conditional
/// law is expanded to the order at which the unconditional `χ²_r` reference is
/// wrong (see the module header, "The conditional reference for canonical
/// families"):
///
/// ```text
///     E[u | Xᵀy] ≈ δ,   Cov[u | Xᵀy] ≈ Σ,   T_c = (u − δ)ᵀ Σ⁻¹ (u − δ),
///     p = P(χ²_{r/c} > T_c / c),   c = 1 + K₄ / (2r),
/// ```
///
/// the scaled `χ²` matching the conditional mean and variance of `T_c`.
///
/// `None` when the geometry supports no test (as in
/// [`basis_adequacy_score_test`]), when `Σ` is not positive definite, or when
/// `c ≤ 0` — the expansion is then outside the range where it describes the
/// conditional law, and "not measured" is the honest report.
pub fn conditional_basis_adequacy_test(
    enrichment: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    null_fit: &CanonicalNullFit,
) -> Option<BasisAdequacyResult> {
    use gam_linalg::faer_ndarray::FaerCholesky;
    const ROW_BLOCK: usize = 4096;
    let m = design.nrows();
    let p = design.ncols();
    let q = enrichment.ncols();
    if m == 0
        || p == 0
        || q == 0
        || enrichment.nrows() != m
        || null_fit.weights.len() != m
        || null_fit.whitened.nrows() != m
        || null_fit.gram.dimension() != p
    {
        return None;
    }
    let geometry = enrichment_geometry(
        enrichment,
        design,
        null_fit.weights.view(),
        null_fit.weights.view(),
        null_fit.score.view(),
        &null_fit.gram,
    )?;
    let r = geometry.basis.ncols();

    // `L = Z̃K`, `m × r`: no wider than the enrichment the caller already holds.
    let mut basis_rows = Array2::<f64>::zeros((m, r));
    let mut start = 0usize;
    while start < m {
        let stop = (start + ROW_BLOCK).min(m);
        let mut residualized = enrichment.slice(ndarray::s![start..stop, ..]).to_owned();
        residualized -= &design
            .slice(ndarray::s![start..stop, ..])
            .dot(&geometry.coefficient_shift);
        basis_rows
            .slice_mut(ndarray::s![start..stop, ..])
            .assign(&residualized.dot(&geometry.basis));
        start = stop;
    }
    let whitened = &null_fit.whitened;
    let third = &null_fit.third;
    let fourth = &null_fit.fourth;
    // `h = diag(H)` and `A_d = diag(LLᵀ)`, `H = X̄X̄ᵀ = X G⁻ Xᵀ`.
    let leverage: Array1<f64> = whitened.rows().into_iter().map(|row| row.dot(&row)).collect();
    let enrichment_leverage: Array1<f64> = basis_rows
        .rows()
        .into_iter()
        .map(|row| row.dot(&row))
        .collect();
    let skewed_leverage = third * &leverage;
    // `H(W₁∘h)`.
    let smoothed_skew = whitened.dot(&whitened.t().dot(&skewed_leverage));

    // Conditional mean offset `δ = −½ Lᵀ(W₁∘h)`.
    let offset = -0.5 * basis_rows.t().dot(&skewed_leverage);

    // `D₁ − D₂ = Lᵀ diag(W₂∘h − W₁∘H(W₁∘h)) L`.
    let diagonal_weight = &(fourth * &leverage) - &(third * &smoothed_skew);
    let mut weighted_rows = basis_rows.clone();
    for (mut row, &weight) in weighted_rows.rows_mut().into_iter().zip(diagonal_weight.iter()) {
        row.iter_mut().for_each(|value| *value *= weight);
    }
    let mut curvature = basis_rows.t().dot(&weighted_rows);

    // `D₃ = Σ_{j,k} M_jkᵀ M_jk`, `M_jk = Σ_n W₁ₙ x̄ₙⱼ x̄ₙₖ Lₙ`, and the kurtosis
    // cross term `k₂ = 2 Σ_j ‖Lᵀ diag(W₁∘x̄_j) L‖²_F`, together in one sweep
    // over the whitened columns: `O(m·r·p²/2 + m·r²·p)`, with only `m × r` and
    // `p × r` scratch alive at once.
    let mut coupling = Array2::<f64>::zeros((r, r));
    let mut kurtosis_cross = 0.0_f64;
    let columns = whitened.ncols();
    for j in 0..columns {
        let mut scaled = basis_rows.clone();
        for (mut row, (&skew, &value)) in scaled
            .rows_mut()
            .into_iter()
            .zip(third.iter().zip(whitened.column(j).iter()))
        {
            let factor = skew * value;
            row.iter_mut().for_each(|entry| *entry *= factor);
        }
        let skew_gram = basis_rows.t().dot(&scaled);
        kurtosis_cross += 2.0 * skew_gram.iter().map(|value| value * value).sum::<f64>();
        // Rows `k ≥ j` of `M_j = X̄[:, j..]ᵀ (diag(W₁∘x̄_j) L)`; the `k > j`
        // blocks stand for both `(j, k)` and `(k, j)`.
        let pair = whitened.slice(ndarray::s![.., j..]).t().dot(&scaled);
        let diagonal = pair.row(0).to_owned();
        coupling += &(2.0 * pair.t().dot(&pair));
        for a in 0..r {
            for b in 0..r {
                coupling[(a, b)] -= diagonal[a] * diagonal[b];
            }
        }
    }
    curvature -= &coupling;

    // `Σ = I − ½(D₁ − D₂ − D₃)`.
    let mut covariance = -0.5 * &curvature;
    for a in 0..r {
        covariance[(a, a)] += 1.0;
    }
    let covariance = 0.5 * (&covariance + &covariance.t());
    if covariance.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let factor = covariance.cholesky(Side::Lower).ok()?;
    let centered = &geometry.projected - &offset;
    let statistic = centered.dot(&factor.solvevec(&centered));

    // `K₄ = Σ W₂ A_d² − k₂ − ‖X̄ᵀ(W₁∘A_d)‖²`.
    let skewed_enrichment = third * &enrichment_leverage;
    let smoothed = whitened.t().dot(&skewed_enrichment);
    let fourth_cumulant = (fourth * &enrichment_leverage.mapv(|value| value * value)).sum()
        - kurtosis_cross
        - smoothed.dot(&smoothed);
    let reference_df = r as f64;
    let scale = 1.0 + fourth_cumulant / (2.0 * reference_df);
    if !(statistic.is_finite() && statistic >= 0.0 && scale.is_finite() && scale > 0.0) {
        return None;
    }
    let p_value = chi_square_sf(statistic / scale, reference_df / scale);
    p_value.is_finite().then_some(BasisAdequacyResult {
        statistic,
        rank: r,
        p_value,
    })
}

/// The whitened geometry of an enrichment against a fitted design: everything
/// both tests in this module share before they choose a reference law.
struct EnrichmentGeometry {
    /// `C = G⁻(XᵀW_H Z)` (`p × q`), so `Z̃ = Z − X·C`.
    coefficient_shift: Array2<f64>,
    /// `K` (`q × r`): the estimable directions, whitened so that
    /// `Kᵀ Z̃ᵀ W_F Z̃ K = I`. `L = Z̃K` is the orthonormal-in-`W_F` score basis.
    basis: Array2<f64>,
    /// `Kᵀ Z̃ᵀ s` — the score in that basis, `Var = φ·I` under the null.
    projected: Array1<f64>,
}

/// Project the fitted design out of the enrichment in the `W_H` metric,
/// whiten the residual from its own `W_F` covariance, and keep the directions
/// the design cannot represent. The module header derives each step.
///
/// Shapes are the caller's contract: `enrichment` is `m × q`, `design` `m × p`
/// with `design_gram.dimension() == p`, and the three vectors have length `m`.
fn enrichment_geometry(
    enrichment: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    hessian_weights: ArrayView1<'_, f64>,
    score_weights: ArrayView1<'_, f64>,
    score: ArrayView1<'_, f64>,
    design_gram: &DesignGramFactor,
) -> Option<EnrichmentGeometry> {
    let m = design.nrows();
    let p = design.ncols();
    let q = enrichment.ncols();
    // Row-blocked first pass: `X_SᵀW_H Z` (the projection's right-hand side) and
    // `E = ZᵀW_F Z`, the enrichment's own UNPROJECTED Gram. Blocked for the same
    // reason the second pass is — a second `m × q` array is 37 MB at
    // `m = 50_000, q = 92`, and a diagnostic may not be the peak-memory term of
    // the fit it is diagnosing.
    //
    // `E` is the denominator of the geometric estimability test: a direction is
    // kept when it retains a numerically resolvable fraction of ITS OWN energy,
    // not when it clears some bar shared with the rest of the enrichment.
    // Accumulating the whole Gram rather than its diagonal costs one more
    // `O(m·q²)` product on top of the two the second pass already runs — a third
    // more of the report's dominant term, which the caller's row cap bounds
    // independently of `n`.
    const ROW_BLOCK: usize = 4096;
    let mut cross = Array2::<f64>::zeros((p, q));
    let mut raw_information = Array2::<f64>::zeros((q, q));
    let mut start = 0usize;
    while start < m {
        let stop = (start + ROW_BLOCK).min(m);
        let block = enrichment.slice(ndarray::s![start..stop, ..]);
        let mut hessian_weighted = block.to_owned();
        let mut fisher_weighted = block.to_owned();
        for local in 0..(stop - start) {
            let curvature = hessian_weights[start + local];
            let fisher = score_weights[start + local];
            if !curvature.is_finite() || !(fisher.is_finite() && fisher >= 0.0) {
                return None;
            }
            hessian_weighted
                .row_mut(local)
                .iter_mut()
                .for_each(|value| *value *= curvature);
            fisher_weighted
                .row_mut(local)
                .iter_mut()
                .for_each(|value| *value *= fisher);
        }
        raw_information += &block.t().dot(&fisher_weighted);
        cross += &design
            .slice(ndarray::s![start..stop, ..])
            .t()
            .dot(&hessian_weighted);
        start = stop;
    }
    // A cheap refusal before the second pass: an enrichment with no weighted
    // energy anywhere carries nothing to test, whatever the projection does to
    // it.
    let energy_scale = (0..q).fold(0.0_f64, |widest, column| {
        widest.max(raw_information[(column, column)])
    });
    if !(energy_scale > 0.0)
        || raw_information.iter().any(|value| !value.is_finite())
        || cross.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    // C = G⁻ (X_SᵀW_H Z): the `W_H`-orthogonal projection of the enrichment onto
    // the fitted column span over these rows. `Z̃ = Z − X_S·C` is the part of the
    // enrichment the realized design cannot represent, and it satisfies
    // `Z̃ᵀW_H X_S = 0`.
    let coefficient_shift = design_gram.solve(&cross)?;
    if coefficient_shift.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // `U = Z̃ᵀ s` and `V = Z̃ᵀ W_F Z̃`, accumulated together in row blocks so the
    // residualized enrichment never has to exist as a second `m × q` array.
    //
    // The score MUST be contracted against `Z̃`, not `Z`. `Zᵀs = Z̃ᵀs + (X·C)ᵀs`
    // and the fit solves the PENALIZED score equation `Xᵀs = S_λβ̂`, so the
    // second term is `Cᵀ S_λ β̂` — precisely the shrinkage this construction
    // exists to remove, re-entering through the numerator after the projection
    // took it out of the denominator. It also breaks the `Z → Z + X·A`
    // invariance, since `C → C + A`. Both failures are pinned as tests.
    let mut information = Array2::<f64>::zeros((q, q));
    let mut u = Array1::<f64>::zeros(q);
    let mut start = 0usize;
    while start < m {
        let stop = (start + ROW_BLOCK).min(m);
        let rows = stop - start;
        let mut residualized = enrichment.slice(ndarray::s![start..stop, ..]).to_owned();
        residualized -= &design
            .slice(ndarray::s![start..stop, ..])
            .dot(&coefficient_shift);
        u += &residualized
            .t()
            .dot(&score.slice(ndarray::s![start..stop]));
        let mut weighted = residualized.clone();
        for local in 0..rows {
            let weight = score_weights[start + local];
            if !(weight.is_finite() && weight >= 0.0) {
                return None;
            }
            weighted
                .row_mut(local)
                .iter_mut()
                .for_each(|value| *value *= weight);
        }
        information += &residualized.t().dot(&weighted);
        start = stop;
    }
    if information.iter().any(|value| !value.is_finite())
        || u.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    // Symmetrize the accumulated Gram: the block sum is symmetric in exact
    // arithmetic, and `strict_symmetric_eigh` refuses anything that is not
    // symmetric on the nose rather than silently repairing it.
    let symmetric = 0.5 * (&information + &information.t());

    // `V` is the covariance the score statistic actually pseudo-inverts. Its
    // numerical rank must therefore be decided against ITS OWN leading
    // eigenvalue: the ordinary backward-error bound for a `q × q` symmetric
    // eigenproblem. This ordering is load-bearing. Whitening `E` first, as the
    // initial #2788/#2789 fix did, made the numerical rank of the raw
    // radial-kernel Gram a prerequisite. Its low-frequency spectrum is much
    // larger than the fine tail, so that version still plateaued at about 32
    // d.f. while the alternative grew from 60 to 156 columns.
    let (information_values, information_vectors) =
        strict_symmetric_eigh(&symmetric, Side::Lower).ok()?;
    let information_max = information_values.iter().cloned().fold(0.0_f64, f64::max);
    if !(information_max > 0.0) {
        return None;
    }
    let information_floor = information_max * (q as f64) * f64::EPSILON;
    let realized: Vec<usize> = (0..q)
        .filter(|&column| information_values[column] > information_floor)
        .collect();
    if realized.is_empty() {
        return None;
    }
    let mut whitening = Array2::<f64>::zeros((q, realized.len()));
    for (slot, &column) in realized.iter().enumerate() {
        let scale = 1.0 / information_values[column].sqrt();
        if !scale.is_finite() {
            return None;
        }
        for row in 0..q {
            whitening[(row, slot)] = information_vectors[(row, column)] * scale;
        }
    }

    // Whitening from `V` gives `BᵀVB = I`. In that metric the eigenvalues `τ`
    // of `BᵀEB` are the RECIPROCALS of the retained-energy fractions: large
    // `τ` is a direction whose residual is only projection dust, finite `τ` is
    // genuinely new resolution. This generalized test is still necessary:
    // using only `V`'s relative rank would promote roundoff when the enrichment
    // lies entirely in `span(X)`, because roundoff would then be `V`'s largest
    // direction as well.
    let symmetric_energy = 0.5 * (&raw_information + &raw_information.t());
    let retained = whitening.t().dot(&symmetric_energy).dot(&whitening);
    let retained = 0.5 * (&retained + &retained.t());
    if retained.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (raw_energy_per_residual, rotation) = strict_symmetric_eigh(&retained, Side::Lower).ok()?;
    let raw_energy_scale = raw_energy_per_residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let psd_roundoff = raw_energy_scale * (realized.len() as f64) * f64::EPSILON;
    if raw_energy_per_residual
        .iter()
        .any(|value| *value < -psd_roundoff)
    {
        return None;
    }
    // A symmetric generalized eigenproblem of this size cannot resolve an
    // energy fraction below `dimension · ε`; deriving the boundary from the
    // arithmetic removes the production `1e-9` knob that caused these issues.
    // Since `τ = raw/residual`, retain exactly `τ · floor < 1`.
    let geometry_floor = (p.max(q) as f64) * f64::EPSILON;
    let kept: Vec<usize> = raw_energy_per_residual
        .iter()
        .enumerate()
        .filter(|(_, raw_energy)| raw_energy.max(0.0) * geometry_floor < 1.0)
        .map(|(index, _)| index)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let rotated = whitening.dot(&rotation);
    let mut basis = Array2::<f64>::zeros((q, kept.len()));
    for (slot, &index) in kept.iter().enumerate() {
        basis.column_mut(slot).assign(&rotated.column(index));
    }
    let projected = basis.t().dot(&u);
    if basis.iter().any(|value| !value.is_finite())
        || projected.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(EnrichmentGeometry {
        coefficient_shift,
        basis,
        projected,
    })
}

/// Gather the rows `rows` selects out of a design into a dense `m × p` array.
///
/// Streams the design in row blocks and keeps only the selected rows inside
/// each. Two properties are load-bearing:
///
/// * it serves EVERY backing. `DesignMatrix::as_dense_ref` is `Some` only for
///   `Dense(Materialized)`, and a reparameterized smooth ships `Dense(Lazy(op))`
///   — `X·Qs` held as an operator — which is what every radial term the fit path
///   reparameterizes becomes, the #2774 fixture included. A report that required
///   a dense view went dark on exactly the fits it exists to diagnose.
/// * it reads the design ONCE. A lazy backing can recompute per chunk, so a
///   caller that streamed it again for each pass and each term would pay that
///   recompute several times over. The gathered array is `m × p`, and `m` is the
///   caller's cap, so this is the one place the design's size enters at all.
pub fn gather_design_rows(
    design: &gam_linalg::matrix::DesignMatrix,
    rows: &[usize],
) -> Option<Array2<f64>> {
    const ROW_BLOCK: usize = 4096;
    let n_total = design.nrows();
    let p = design.ncols();
    if p == 0
        || rows.is_empty()
        || rows.last().is_some_and(|last| *last >= n_total)
        || rows.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let mut gathered = Array2::<f64>::zeros((rows.len(), p));
    let mut selected = 0usize;
    let mut start = 0usize;
    while start < n_total && selected < rows.len() {
        let stop = (start + ROW_BLOCK).min(n_total);
        let first = selected;
        while selected < rows.len() && rows[selected] < stop {
            selected += 1;
        }
        if selected > first {
            let block = design.try_row_chunk(start..stop).ok()?;
            for (offset, &row) in rows[first..selected].iter().enumerate() {
                gathered
                    .row_mut(first + offset)
                    .assign(&block.row(row - start));
            }
        }
        start = stop;
    }
    gathered
        .iter()
        .all(|value| value.is_finite())
        .then_some(gathered)
}

/// `Xᵀ diag(w) X` for an already-gathered design.
///
/// The projection in [`basis_adequacy_score_test`] must be orthogonal in the
/// `W_H` metric ON THE ROWS THE TEST USES — `Z̃ᵀW_H X_S = 0` is what annihilates
/// the penalized fit's shrinkage bias, and it is a property of the selected
/// sub-design. A Gram formed over all `n` rows does not give it.
pub fn weighted_gram(
    design: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Option<Array2<f64>> {
    let m = design.nrows();
    if m == 0 || design.ncols() == 0 || weights.len() != m {
        return None;
    }
    let mut weighted = design.to_owned();
    for row in 0..m {
        let weight = weights[row];
        if !weight.is_finite() {
            return None;
        }
        weighted.row_mut(row).iter_mut().for_each(|v| *v *= weight);
    }
    let gram = design.t().dot(&weighted);
    gram.iter().all(|value| value.is_finite()).then_some(gram)
}

/// A once-per-fit factorization of the weighted design Gram `G = XᵀW_H X`.
///
/// The projection `C = G⁻(XᵀW_H Z)` is applied once per SMOOTH TERM, but `G`
/// depends only on the design and the weights. Factoring it inside the test
/// would pay `O(p³)` per term — on a model with ten smooths that is ten extra
/// IRLS-iteration-equivalents on a fit that runs a few dozen, which is a
/// diagnostic charging a third of the fit. Building the factor is therefore the
/// caller's job and it is a type, not a convention: the input struct cannot be
/// constructed with a raw matrix that someone forgot to reuse.
pub struct DesignGramFactor {
    kind: DesignGramFactorKind,
    dimension: usize,
}

enum DesignGramFactorKind {
    /// The ordinary route. `O(p³)` once, then `O(p²q)` per solve.
    Cholesky(gam_linalg::faer_ndarray::FaerCholeskyFactor),
    /// Rank-deficient fallback: the spectral pseudo-inverse, held as its square
    /// root `R = U diag(1/√λ)` over the directions above the rank floor, so that
    /// `G⁺ = R Rᵀ`. It projects onto `range(G)`, which is the right answer for a
    /// design that is rank-deficient in the fit's own frame — directions the
    /// design cannot span in the `W_H` metric are not directions to project
    /// out. A dense symmetric eigendecomposition is the expensive route (it is
    /// the #2757 cost complaint at `p = 4096`), so it is the exception rather
    /// than the default.
    SpectralPseudoInverse(Array2<f64>),
}

impl DesignGramFactor {
    /// Factor `G`. `None` when the matrix is empty, non-square, non-finite, or
    /// has no positive spectrum at all.
    pub fn new(gram: ArrayView2<'_, f64>) -> Option<Self> {
        use gam_linalg::faer_ndarray::FaerCholesky;
        let dimension = gram.nrows();
        if dimension == 0
            || gram.ncols() != dimension
            || gram.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let owned = gram.to_owned();
        if let Ok(factor) = owned.cholesky(Side::Lower) {
            return Some(Self {
                kind: DesignGramFactorKind::Cholesky(factor),
                dimension,
            });
        }
        let symmetric = 0.5 * (&owned + &owned.t());
        let (eigenvalues, eigenvectors) = strict_symmetric_eigh(&symmetric, Side::Lower).ok()?;
        let largest = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        if !(largest > 0.0) {
            return None;
        }
        let floor = largest * (dimension as f64) * f64::EPSILON;
        let kept: Vec<usize> = (0..dimension)
            .filter(|&index| eigenvalues[index] > floor)
            .collect();
        let mut root = Array2::<f64>::zeros((dimension, kept.len()));
        for (slot, &index) in kept.iter().enumerate() {
            let scale = 1.0 / eigenvalues[index].sqrt();
            for row in 0..dimension {
                root[(row, slot)] = eigenvectors[(row, index)] * scale;
            }
        }
        root.iter()
            .all(|value| value.is_finite())
            .then_some(Self {
                kind: DesignGramFactorKind::SpectralPseudoInverse(root),
                dimension,
            })
    }

    /// Side length of the factored Gram, i.e. the design's column count.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn solve(&self, rhs: &Array2<f64>) -> Option<Array2<f64>> {
        let solved = match &self.kind {
            DesignGramFactorKind::Cholesky(factor) => factor.solve_mat(rhs),
            DesignGramFactorKind::SpectralPseudoInverse(root) => root.dot(&root.t().dot(rhs)),
        };
        solved
            .iter()
            .all(|value| value.is_finite())
            .then_some(solved)
    }

    /// The design in `G`-whitened coordinates, `X̄ = X R` with `R Rᵀ = G⁻`.
    ///
    /// Its rows give the hat matrix directly, `X G⁻ Xᵀ = X̄ X̄ᵀ`, through the
    /// SAME factor every projection in this module uses: the Cholesky route
    /// takes `R = L⁻ᵀ`, the pseudo-inverse route its stored square root. So the
    /// leverages and the projection agree on which directions `range(G)` has,
    /// which two independently thresholded factorizations would not guarantee.
    fn whiten(&self, design: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if design.ncols() != self.dimension {
            return None;
        }
        let whitened = match &self.kind {
            DesignGramFactorKind::Cholesky(factor) => {
                // `L⁻¹` by one triangular solve against the identity, then a
                // single GEMM for all rows: `O(p³ + m·p²)` in blocked kernels,
                // where a per-row triangular solve would be `m` scalar sweeps.
                let lower = factor.lower_triangular();
                let inverse = gam_linalg::triangular::forward_substitution_lower_matrix(
                    lower.view(),
                    Array2::<f64>::eye(self.dimension).view(),
                );
                design.dot(&inverse.t())
            }
            DesignGramFactorKind::SpectralPseudoInverse(root) => design.dot(root),
        };
        whitened
            .iter()
            .all(|value| value.is_finite())
            .then_some(whitened)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// Deterministic linear-congruential normal draws, so the size/power checks
    /// below are reproducible without pulling a sampler dependency into this
    /// crate's test surface.
    struct Lcg(u64);

    impl Lcg {
        fn next_uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn next_normal(&mut self) -> f64 {
            // Box-Muller. `next_uniform` lies in [0, 1); redrawing only an exact
            // 0 gives the uniform law on (0, 1), so its log is finite without
            // truncating the tail, and every other draw is used as it comes.
            let u1 = loop {
                let u = self.next_uniform();
                if u > 0.0 {
                    break u;
                }
            };
            let u2 = self.next_uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Gaussian-identity harness: `W = 1`, `β̂` is the RIDGE-penalized least
    /// squares fit (`H = XᵀX + ridge·I`), `s = y − Xβ̂`, and the Gram handed to
    /// the test is the unpenalized `XᵀX`. The ridge is a knob so a test can vary
    /// how hard the fit is shrunk without touching anything else.
    struct GaussianHarness {
        design: Array2<f64>,
        enrichment: Array2<f64>,
        weights: Array1<f64>,
        score: Array1<f64>,
        design_gram: DesignGramFactor,
    }

    impl GaussianHarness {
        fn new(design: Array2<f64>, enrichment: Array2<f64>, y: Array1<f64>, ridge: f64) -> Self {
            let n = design.nrows();
            let p = design.ncols();
            let gram = design.t().dot(&design);
            let mut hessian = gram.clone();
            for index in 0..p {
                hessian[(index, index)] += ridge;
            }
            let beta = invert_symmetric(&hessian).dot(&design.t().dot(&y));
            let score = &y - &design.dot(&beta);
            Self {
                design,
                enrichment,
                weights: Array1::ones(n),
                score,
                design_gram: DesignGramFactor::new(gram.view())
                    .expect("test harness Gram is factorable"),
            }
        }

        fn input(&self) -> BasisAdequacyInput<'_> {
            BasisAdequacyInput {
                enrichment: self.enrichment.view(),
                design: self.design.view(),
                hessian_weights: self.weights.view(),
                score_weights: self.weights.view(),
                score: self.score.view(),
                design_gram: &self.design_gram,
                dispersion: 1.0,
                residual_df: None,
                scale: SmoothTestScale::Known,
            }
        }
    }

    fn invert_symmetric(matrix: &Array2<f64>) -> Array2<f64> {
        let (values, vectors) = strict_symmetric_eigh(matrix, Side::Lower)
            .expect("test harness matrix is symmetric positive definite");
        let mut inverse = Array2::<f64>::zeros(matrix.raw_dim());
        for (index, &value) in values.iter().enumerate() {
            let column = vectors.column(index);
            let scale = 1.0 / value;
            for row in 0..matrix.nrows() {
                for col in 0..matrix.ncols() {
                    inverse[(row, col)] += scale * column[row] * column[col];
                }
            }
        }
        inverse
    }

    /// An enrichment entirely inside `span(X)` leaves no estimable direction:
    /// the projection annihilates it and the test refuses rather than reporting
    /// a degenerate statistic against a zero-variance direction.
    #[test]
    fn enrichment_inside_the_fitted_span_has_no_estimable_direction() {
        let design = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        // Exact linear combinations of the two design columns.
        let enrichment = design.dot(&array![[2.0, -1.0], [0.5, 3.0]]);
        let y = array![0.3, -0.2, 0.7, 0.1, -0.5, 0.4];
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        assert_eq!(basis_adequacy_score_test(harness.input()), None);
    }

    /// The rank reports the genuinely NEW resolution: a `q = 3` enrichment whose
    /// first column duplicates a design column is rank 2.
    #[test]
    fn rank_counts_only_directions_outside_the_fitted_span() {
        let mut rng = Lcg(20_260_823);
        let n = 200;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = row as f64 / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x; // already in span(X)
            enrichment[(row, 1)] = x * x;
            enrichment[(row, 2)] = x * x * x;
            y[row] = 0.5 + 2.0 * x + 0.1 * rng.next_normal();
        }
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let out = basis_adequacy_score_test(harness.input())
            .expect("two enrichment directions remain estimable");
        assert_eq!(out.rank, 2);
    }

    /// A correctly specified fit produces uniform p-values.
    ///
    /// This is the null-behaviour anchor. `y` is linear in `x` and the design
    /// spans that exactly, so the quadratic/cubic enrichment tests a true `H₀`.
    #[test]
    fn null_statistic_is_uniform_under_its_reference() {
        let n = 400;
        let replicates = 2000;
        let mut rng = Lcg(1_234_567);
        let mut p_values = Vec::with_capacity(replicates);
        for _ in 0..replicates {
            let mut design = Array2::<f64>::zeros((n, 2));
            let mut enrichment = Array2::<f64>::zeros((n, 3));
            let mut y = Array1::<f64>::zeros(n);
            for row in 0..n {
                let x = (row as f64 + 0.5) / n as f64;
                design[(row, 0)] = 1.0;
                design[(row, 1)] = x;
                enrichment[(row, 0)] = x * x;
                enrichment[(row, 1)] = x * x * x;
                enrichment[(row, 2)] = (6.0 * x).sin();
                y[row] = 0.5 + 2.0 * x + rng.next_normal();
            }
            let harness = GaussianHarness::new(design, enrichment, y, 0.0);
            let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
            assert_eq!(out.rank, 3);
            p_values.push(out.p_value);
        }
        // An unpenalized Gaussian fit with known scale makes `T` exactly `χ²₃`,
        // so both tails are held to Monte Carlo error: a conservative
        // reference fails this exactly as an inflated one does.
        assert_uniform(&p_values, "gaussian known-scale score test");
    }

    /// A basis that cannot reach the truth is detected: the same design/enrichment
    /// pair as the null check, but with a quadratic mean the linear design cannot
    /// represent, rejects overwhelmingly.
    #[test]
    fn missing_curvature_is_detected() {
        let n = 400;
        let mut rng = Lcg(7_654_321);
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = x * x * x;
            enrichment[(row, 2)] = (6.0 * x).sin();
            y[row] = 0.5 + 2.0 * x + 3.0 * x * x + rng.next_normal();
        }
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
        assert!(
            out.p_value < 1e-3,
            "quadratic lack of fit should be detected, got p={}",
            out.p_value
        );
    }

    /// **The defining contract.** Shifting the enrichment by ANY multiple of the
    /// design columns (`Z → Z + X·A`) leaves the statistic bit-comparably
    /// unchanged, because `Z̃` is the `W_H`-orthogonal complement of `span(X)`
    /// and `X·A` lies entirely inside it.
    ///
    /// This is what separates the shipped construction from the `H⁻¹`-projected
    /// penalized score test, which does NOT satisfy it: there `Z̃ = ... G H⁻¹S_λ`
    /// picks up whatever part of `X·A` the penalty shrinks, so the "same"
    /// alternative reparameterized differently gives a different answer, and the
    /// difference is the fit's shrinkage bias rather than any lack of fit.
    #[test]
    fn statistic_is_invariant_to_shifting_the_enrichment_by_design_columns() {
        let n = 300;
        let mut rng = Lcg(4_242);
        let mut design = Array2::<f64>::zeros((n, 3));
        let mut enrichment = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            design[(row, 2)] = (3.0 * x).cos();
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = (5.0 * x).sin();
            y[row] = 1.0 + x + 0.8 * x * x + 0.4 * rng.next_normal();
        }
        // A heavy ridge, so the fit is visibly shrunk and any leak of the
        // shrinkage bias into the statistic would be large.
        let base = GaussianHarness::new(design.clone(), enrichment.clone(), y.clone(), 40.0);
        let shift = array![[7.0, -2.0], [0.5, 3.0], [-1.5, 4.0]];
        let shifted_enrichment = &enrichment + &design.dot(&shift);
        let shifted = GaussianHarness::new(design, shifted_enrichment, y, 40.0);
        let a = basis_adequacy_score_test(base.input()).expect("base result");
        let b = basis_adequacy_score_test(shifted.input()).expect("shifted result");
        assert_eq!(a.rank, b.rank);
        assert!(
            (a.statistic - b.statistic).abs() <= 1e-8 * a.statistic.max(1.0),
            "statistic must not move under Z -> Z + X·A; got {} vs {}",
            a.statistic,
            b.statistic
        );
    }

    /// The shrinkage bias of the penalized fit does not enter the statistic:
    /// varying the ridge over four orders of magnitude, with the DATA held
    /// fixed, leaves the null statistic in the same neighbourhood instead of
    /// growing with how hard the fit is shrunk.
    #[test]
    fn statistic_does_not_track_the_penalty_strength_under_the_null() {
        let n = 500;
        let mut rng = Lcg(31_337);
        let mut design = Array2::<f64>::zeros((n, 3));
        let mut enrichment = Array2::<f64>::zeros((n, 4));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            design[(row, 2)] = x * x;
            enrichment[(row, 0)] = x * x * x;
            enrichment[(row, 1)] = (7.0 * x).sin();
            enrichment[(row, 2)] = (7.0 * x).cos();
            enrichment[(row, 3)] = (11.0 * x).sin();
            // Truth is exactly in span(X): H₀ holds however hard the ridge bites.
            y[row] = 1.0 + 3.0 * x - 2.0 * x * x + rng.next_normal();
        }
        let mut statistics = Vec::new();
        for ridge in [0.0, 1.0, 1.0e2, 1.0e4] {
            let harness =
                GaussianHarness::new(design.clone(), enrichment.clone(), y.clone(), ridge);
            let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
            statistics.push(out.statistic);
        }
        let span = statistics.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - statistics.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            span < 4.0,
            "the ridge must not drive the null statistic; got {statistics:?}"
        );
    }

    /// Gathering serves an operator-backed design, which is what a
    /// reparameterized smooth ships and what a dense-view requirement went dark
    /// on. Round-trips the exact rows requested, in the order requested.
    #[test]
    fn gathering_selects_the_requested_rows_from_any_backing() {
        let mut source = Array2::<f64>::zeros((10, 3));
        for row in 0..10 {
            for column in 0..3 {
                source[(row, column)] = (row * 3 + column) as f64;
            }
        }
        let design = gam_linalg::matrix::DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(source.clone()),
        );
        let rows = [0usize, 4, 5, 9];
        let gathered = gather_design_rows(&design, &rows).expect("rows are in range and sorted");
        assert_eq!(gathered.dim(), (4, 3));
        for (local, &row) in rows.iter().enumerate() {
            assert_eq!(gathered.row(local), source.row(row));
        }
        // Out of range, unsorted and empty selections refuse rather than
        // silently returning a shorter or misaligned block.
        assert!(gather_design_rows(&design, &[0, 10]).is_none());
        assert!(gather_design_rows(&design, &[4, 4]).is_none());
        assert!(gather_design_rows(&design, &[4, 1]).is_none());
        assert!(gather_design_rows(&design, &[]).is_none());
    }

    /// **The subsetting contract.** Running the test on a subset of rows gives
    /// exactly the same answer as building the subset design directly — the
    /// statistic has no dependence on rows it was not handed.
    ///
    /// This is what makes a caller's row cap a POWER decision and nothing else.
    /// If the identities held only on the full sample, a capped report would be
    /// measuring a different hypothesis than the uncapped one and the two could
    /// not be compared.
    #[test]
    fn a_row_subset_gives_the_same_answer_as_the_subset_design() {
        let n = 600;
        let mut rng = Lcg(5_150);
        let mut design = Array2::<f64>::zeros((n, 3));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            design[(row, 2)] = (2.0 * x).cos();
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = (9.0 * x).sin();
            enrichment[(row, 2)] = x * x * x;
            y[row] = 0.4 + 1.3 * x + 0.9 * x * x + 0.3 * rng.next_normal();
        }
        // Fit ONCE on all rows, then evaluate the report on every third row.
        let full = GaussianHarness::new(design.clone(), enrichment.clone(), y, 2.0);
        let rows: Vec<usize> = (0..n).step_by(3).collect();
        let sub_design = select(&design, &rows);
        let sub_enrichment = select(&enrichment, &rows);
        let sub_weights = Array1::<f64>::ones(rows.len());
        let sub_score = Array1::from_iter(rows.iter().map(|&row| full.score[row]));
        let sub_gram = weighted_gram(sub_design.view(), sub_weights.view())
            .and_then(|gram| DesignGramFactor::new(gram.view()))
            .expect("the subset Gram factors");
        let subset = basis_adequacy_score_test(BasisAdequacyInput {
            enrichment: sub_enrichment.view(),
            design: sub_design.view(),
            hessian_weights: sub_weights.view(),
            score_weights: sub_weights.view(),
            score: sub_score.view(),
            design_gram: &sub_gram,
            dispersion: 1.0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("the subset carries estimable directions");
        // Gathering the same rows out of a `DesignMatrix` must reproduce it bit
        // for bit — the two routes into the statistic cannot disagree.
        let backing = gam_linalg::matrix::DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(design),
        );
        let gathered = gather_design_rows(&backing, &rows).expect("gather");
        let gathered_gram = weighted_gram(gathered.view(), sub_weights.view())
            .and_then(|gram| DesignGramFactor::new(gram.view()))
            .expect("the gathered Gram factors");
        let via_gather = basis_adequacy_score_test(BasisAdequacyInput {
            enrichment: sub_enrichment.view(),
            design: gathered.view(),
            hessian_weights: sub_weights.view(),
            score_weights: sub_weights.view(),
            score: sub_score.view(),
            design_gram: &gathered_gram,
            dispersion: 1.0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("the gathered subset carries estimable directions");
        assert_eq!(subset, via_gather);
        // And a subset genuinely tests less than the whole: its reference d.f.
        // is the same but its statistic is not the full-sample one, so a caller
        // reading a capped report is reading a real, weaker measurement rather
        // than a rescaled copy of the full one.
        let full_result = basis_adequacy_score_test(full.input()).expect("full result");
        assert_eq!(full_result.rank, subset.rank);
        assert!(full_result.statistic > subset.statistic);
    }

    /// An enrichment column the fitted design already spans does not get to
    /// decide whether the OTHER columns are estimable, however much energy it
    /// carries.
    ///
    /// This is #2788/#2789 in miniature. The floor this replaced was
    /// `ESTIMABLE_DIRECTION_FLOOR × max_j (ZᵀW_F Z)_jj` — one bar for the whole
    /// enrichment — so an absorbed direction with a large weight set a level the
    /// genuinely-new directions could not clear, and the report went dark
    /// (`rank = 0`, `None`) on a fit whose residuals plainly carry the quadratic
    /// structure the design omits. Judging each direction against its own
    /// unprojected energy is scale-free and cannot be moved by a column that
    /// contributes no estimable direction at all.
    #[test]
    fn an_absorbed_column_does_not_decide_the_other_directions() {
        let n = 200;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut lean = Array2::<f64>::zeros((n, 2));
        let mut padded = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        let mut rng = Lcg(2_788_2_789);
        for row in 0..n {
            let x = row as f64 / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            lean[(row, 0)] = x * x;
            lean[(row, 1)] = x * x * x;
            // The same two directions, behind a third that is exactly
            // `1e4·x`. It is a design column, so the projection annihilates it,
            // and its energy is eight orders above the residual energy of the
            // other two — far enough that a floor shared across the enrichment
            // sits ABOVE them (`1e-9 · max_j (ZᵀW_F Z)_jj = 6.7` against
            // residual eigenvalues of 1.1 and less) and takes both out.
            padded[(row, 0)] = 1.0e4 * x;
            padded[(row, 1)] = x * x;
            padded[(row, 2)] = x * x * x;
            y[row] = 0.5 + 2.0 * x + 3.0 * x * x + 0.1 * rng.next_normal();
        }
        let lean_out = basis_adequacy_score_test(
            GaussianHarness::new(design.clone(), lean, y.clone(), 0.0).input(),
        )
        .expect("the quadratic and cubic directions are estimable");
        let padded_out =
            basis_adequacy_score_test(GaussianHarness::new(design, padded, y, 0.0).input())
                .expect("padding with an absorbed column may not blind the test");
        assert_eq!(lean_out.rank, 2);
        assert_eq!(
            padded_out.rank, 2,
            "an exact design column is not new resolution, and it is not a floor either"
        );
        let relative =
            (padded_out.statistic - lean_out.statistic).abs() / lean_out.statistic.max(1.0);
        assert!(
            relative < 1e-6,
            "the statistic moved with a column carrying no estimable direction: \
             {} vs {}",
            padded_out.statistic,
            lean_out.statistic
        );
    }

    /// A wider alternative supplies MORE reference d.f., not less.
    ///
    /// The enrichment is `width` orthogonal cosines with geometrically decaying
    /// weights — the shape of a smooth kernel's residual spectrum, a
    /// Karhunen–Loève tail with no gap — and the design is exactly its first
    /// `DESIGN_WIDTH` columns. The answer is then arithmetic rather than
    /// statistical: those columns are absorbed exactly, every other one is
    /// untouched, so the estimable rank is `width − DESIGN_WIDTH` at every
    /// width.
    ///
    /// Against a floor shared across the enrichment this stops being true as the
    /// width grows, because the shared scale is set by the widest column and
    /// does not decay with the tail: the old rule returned 6, 12 and 12 for
    /// these three widths — the count stopped moving while the alternative kept
    /// growing, which is the shape #2789 was filed on.
    #[test]
    fn reference_df_grows_with_the_width_of_the_alternative() {
        const N: usize = 400;
        const DESIGN_WIDTH: usize = 6;
        const DECAY: f64 = 0.55;
        let column = |row: usize, index: usize| {
            let x = row as f64 / N as f64;
            DECAY.powi(index as i32) * (std::f64::consts::PI * (index + 1) as f64 * x).cos()
        };
        for width in [12usize, 18, 24] {
            let mut design = Array2::<f64>::zeros((N, DESIGN_WIDTH));
            let mut enrichment = Array2::<f64>::zeros((N, width));
            let mut y = Array1::<f64>::zeros(N);
            let mut rng = Lcg(2_789_2_788);
            for row in 0..N {
                for index in 0..DESIGN_WIDTH {
                    design[(row, index)] = column(row, index);
                }
                for index in 0..width {
                    enrichment[(row, index)] = column(row, index);
                }
                y[row] = design[(row, 0)] + 0.1 * rng.next_normal();
            }
            let out =
                basis_adequacy_score_test(GaussianHarness::new(design, enrichment, y, 0.0).input())
                    .unwrap_or_else(|| panic!("width {width}: no verdict at all"));
            assert_eq!(
                out.rank,
                width - DESIGN_WIDTH,
                "width {width}: the design spans exactly {DESIGN_WIDTH} of the \
                 alternative's directions, so {} must stay estimable",
                width - DESIGN_WIDTH
            );
        }
    }

    /// Fine residual directions are ranked in the covariance the statistic
    /// inverts, not against the absorbed low-frequency head of the raw kernel.
    ///
    /// This is the part the first principal-angle fix for #2788/#2789 still
    /// missed. Its `E`-first whitening discarded every tail column below
    /// `lambda_max(E) * n * EPSILON` before it formed the generalized problem.
    /// Here all eight tail columns are mutually orthogonal and wholly outside
    /// the design, but their common scale puts them below that old global floor.
    /// Their residual Gram is perfectly conditioned, so all eight are real
    /// score directions and must survive together.
    #[test]
    fn fine_residual_subspace_is_not_ranked_against_the_absorbed_raw_head() {
        const N: usize = 256;
        const DESIGN_WIDTH: usize = 4;
        const TAIL_WIDTH: usize = 8;
        const TAIL_SCALE: f64 = 1.0e-8;
        let mode = |row: usize, index: usize| {
            let angle = std::f64::consts::PI * (row as f64 + 0.5) * index as f64 / N as f64;
            angle.cos()
        };

        let mut design = Array2::<f64>::zeros((N, DESIGN_WIDTH));
        let mut enrichment = Array2::<f64>::zeros((N, DESIGN_WIDTH + TAIL_WIDTH));
        let mut y = Array1::<f64>::zeros(N);
        let mut rng = Lcg(2_788_2_789_2_788);
        for row in 0..N {
            for index in 0..DESIGN_WIDTH {
                let value = mode(row, index);
                design[(row, index)] = value;
                enrichment[(row, index)] = value;
            }
            for tail in 0..TAIL_WIDTH {
                enrichment[(row, DESIGN_WIDTH + tail)] =
                    TAIL_SCALE * mode(row, DESIGN_WIDTH + tail);
            }
            y[row] = 2.0 * enrichment[(row, DESIGN_WIDTH)] + 0.1 * rng.next_normal();
        }

        let raw = enrichment.t().dot(&enrichment);
        let head = raw[(0, 0)];
        let tail = raw[(DESIGN_WIDTH, DESIGN_WIDTH)];
        assert!(
            tail < head * (N as f64) * f64::EPSILON,
            "fixture must sit below the obsolete raw-Gram floor: tail={tail:e}, head={head:e}"
        );
        let out =
            basis_adequacy_score_test(GaussianHarness::new(design, enrichment, y, 0.0).input())
                .expect("the well-conditioned residual tail supports a verdict");
        assert_eq!(
            out.rank, TAIL_WIDTH,
            "every orthogonal tail mode is new resolution, irrespective of its scale"
        );
    }

    /// Row-selected `XᵀWX` matches the direct product, including the weights.
    #[test]
    fn weighted_gram_matches_the_direct_product() {
        let design = array![[1.0, 0.5], [1.0, -2.0], [1.0, 3.0], [1.0, 0.0]];
        let weights = array![0.25, 2.0, 1.5, 0.0];
        let gram = weighted_gram(design.view(), weights.view()).expect("finite inputs");
        let mut expected = Array2::<f64>::zeros((2, 2));
        for row in 0..4 {
            for i in 0..2 {
                for j in 0..2 {
                    expected[(i, j)] += weights[row] * design[(row, i)] * design[(row, j)];
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((gram[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            }
        }
        assert!(weighted_gram(design.view(), array![1.0, 2.0].view()).is_none());
    }

    fn select(matrix: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows.len(), matrix.ncols()));
        for (local, &row) in rows.iter().enumerate() {
            out.row_mut(local).assign(&matrix.row(row));
        }
        out
    }

    /// With the scale estimated, the p-value is the classical added-variable
    /// `F` test. Take an unpenalized least-squares fit, `φ̂ = RSS₀/ν` with
    /// `ν = n − p`, and an enrichment of width `r` outside `span(X)`. The exact
    /// reference is `F = ((RSS₀ − RSS₁)/r) / (RSS₁/(ν − r))` on `(r, ν − r)`
    /// d.f., where `RSS₁` is the residual sum of squares after also fitting
    /// `Z`.
    ///
    /// Referring `T/r` to `F(r, ν)` instead divides by a `φ̂` whose residual
    /// sum contains the numerator. The ratio is then `(ν/r)·Beta`, bounded and
    /// under-dispersed, and it reads conservative at every level.
    #[test]
    fn estimated_scale_p_value_is_the_exact_added_variable_f_test() {
        let n = 60;
        let mut rng = Lcg(20_260_919);
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = x * x * x;
            enrichment[(row, 2)] = (6.0 * x).sin();
            y[row] = 0.5 + 2.0 * x + 0.4 * x * x + 0.3 * rng.next_normal();
        }
        let residual_sum = |columns: &Array2<f64>| {
            let beta = invert_symmetric(&columns.t().dot(columns)).dot(&columns.t().dot(&y));
            let residual = &y - &columns.dot(&beta);
            residual.dot(&residual)
        };
        let rss_null = residual_sum(&design);
        let rss_alternative =
            residual_sum(&ndarray::concatenate![ndarray::Axis(1), design, enrichment]);
        let (p, r) = (design.ncols() as f64, enrichment.ncols() as f64);
        let residual_df = n as f64 - p;
        let exact_f = ((rss_null - rss_alternative) / r) / (rss_alternative / (residual_df - r));
        let exact = fisher_snedecor_sf(exact_f, r, residual_df - r);

        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let mut input = harness.input();
        input.dispersion = rss_null / residual_df;
        input.residual_df = Some(residual_df);
        input.scale = SmoothTestScale::Estimated;
        let out = basis_adequacy_score_test(input).expect("three estimable directions");
        assert_eq!(out.rank, 3);
        // The score statistic itself is `(RSS₀ − RSS₁)/φ̂`: in the Gaussian
        // identity case the residual's projection onto `span(Z̃)` IS the drop in
        // residual sum from fitting `Z`.
        let expected_statistic = (rss_null - rss_alternative) / (rss_null / residual_df);
        assert!(
            (out.statistic - expected_statistic).abs() <= 1e-9 * expected_statistic,
            "statistic {} against {expected_statistic}",
            out.statistic
        );
        assert!(
            (out.p_value - exact).abs() <= 1e-9 * exact.max(1e-12),
            "p-value {} against the exact added-variable F tail {exact}",
            out.p_value
        );
    }

    /// The estimated-scale branch refuses when the fit's residual d.f. cannot
    /// hold the tested directions: `ν ≤ r` leaves no degrees of freedom for a
    /// scale estimate independent of the numerator.
    #[test]
    fn estimated_scale_refuses_when_the_residual_df_cannot_hold_the_rank() {
        let n = 40;
        let mut rng = Lcg(7);
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = x * x * x;
            enrichment[(row, 2)] = (6.0 * x).sin();
            y[row] = 0.5 + 2.0 * x + rng.next_normal();
        }
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let mut input = harness.input();
        input.scale = SmoothTestScale::Estimated;
        input.residual_df = Some(3.0);
        assert_eq!(basis_adequacy_score_test(input), None);
    }

    /// Shape and finiteness guards refuse rather than returning a stand-in.
    #[test]
    fn degenerate_inputs_refuse() {
        let design = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let enrichment = array![[0.0], [1.0], [4.0]];
        let y = array![0.1, 0.2, 0.3];
        let harness = GaussianHarness::new(design, enrichment, y, 1.0);

        let mut bad_dispersion = harness.input();
        bad_dispersion.dispersion = 0.0;
        assert_eq!(basis_adequacy_score_test(bad_dispersion), None);

        let mismatched = Array2::<f64>::zeros((2, 1));
        let mut bad_rows = harness.input();
        bad_rows.enrichment = mismatched.view();
        assert_eq!(basis_adequacy_score_test(bad_rows), None);

        let mut estimated_without_df = harness.input();
        estimated_without_df.scale = SmoothTestScale::Estimated;
        estimated_without_df.residual_df = None;
        assert_eq!(basis_adequacy_score_test(estimated_without_df), None);
    }

    /// Kolmogorov–Smirnov `p` of `p_values` against `U(0, 1)`: the asymptotic
    /// Kolmogorov tail at Stephens' finite-sample-corrected statistic.
    fn kolmogorov_smirnov_uniform_p_value(p_values: &[f64]) -> f64 {
        let mut sorted = p_values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len() as f64;
        let distance = sorted
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let above = (index as f64 + 1.0) / count - value;
                let below = value - index as f64 / count;
                above.max(below)
            })
            .fold(0.0_f64, f64::max);
        let root = count.sqrt();
        let scaled = distance * (root + 0.12 + 0.11 / root);
        // `Q(t) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²t²}`, summed until a term no longer
        // moves the partial sum.
        let mut tail = 0.0_f64;
        let mut k = 1.0_f64;
        loop {
            let term = 2.0 * (-2.0 * k * k * scaled * scaled).exp();
            let signed = if (k as u64) % 2 == 1 { term } else { -term };
            if tail + signed == tail {
                break;
            }
            tail += signed;
            k += 1.0;
        }
        tail.clamp(0.0, 1.0)
    }

    /// Hold null p-values to `U(0, 1)` in BOTH directions: a two-sided
    /// Kolmogorov–Smirnov test over the whole range, and the empirical size at
    /// each conventional level within 3.5 Monte Carlo standard errors of that
    /// level. A conservative reference — sizes below nominal, mass piled near
    /// `p = 1` — fails here exactly as an anti-conservative one does.
    fn assert_uniform(p_values: &[f64], label: &str) {
        assert!(
            p_values.iter().all(|value| (0.0..=1.0).contains(value)),
            "{label}: p-values outside [0, 1]"
        );
        let count = p_values.len() as f64;
        let ks = kolmogorov_smirnov_uniform_p_value(p_values);
        assert!(ks > 1e-3, "{label}: KS p = {ks:.3e} against U(0,1) over {count} replicates");
        for level in [0.01, 0.05, 0.10] {
            let size = p_values.iter().filter(|&&value| value <= level).count() as f64 / count;
            let standard_error = (level * (1.0 - level) / count).sqrt();
            assert!(
                (size - level).abs() <= 3.5 * standard_error,
                "{label}: size {size:.4} at level {level} is {:+.1} Monte Carlo SE from nominal",
                (size - level) / standard_error
            );
        }
    }

    /// Fixed-design canonical GLM harness: `x` on a regular grid, `X` the
    /// Legendre polynomials of degree `< p` in `2x − 1`, `Z` the cubic radial
    /// functions `|x − c|³` at `q` equally spaced centres, and `η₀ = Xβ₀`
    /// inside `span(X)`, so the enrichment tests a true `H₀`.
    struct CanonicalHarness {
        design: Array2<f64>,
        enrichment: Array2<f64>,
        eta: Array1<f64>,
        family: CanonicalExponentialFamily,
    }

    impl CanonicalHarness {
        fn new(
            n: usize,
            p: usize,
            q: usize,
            coefficients: &[f64],
            family: CanonicalExponentialFamily,
        ) -> Self {
            let mut design = Array2::<f64>::zeros((n, p));
            let mut enrichment = Array2::<f64>::zeros((n, q));
            for row in 0..n {
                let x = (row as f64 + 0.5) / n as f64;
                let t = 2.0 * x - 1.0;
                let (mut previous, mut current) = (1.0, t);
                design[(row, 0)] = 1.0;
                if p > 1 {
                    design[(row, 1)] = t;
                }
                for degree in 1..p.saturating_sub(1) {
                    let k = degree as f64;
                    let next = ((2.0 * k + 1.0) * t * current - k * previous) / (k + 1.0);
                    previous = current;
                    current = next;
                    design[(row, degree + 1)] = next;
                }
                for column in 0..q {
                    let centre = (column as f64 + 0.5) / q as f64;
                    enrichment[(row, column)] = (x - centre).abs().powi(3);
                }
            }
            let mut beta = Array1::<f64>::zeros(p);
            for (slot, &value) in beta.iter_mut().zip(coefficients) {
                *slot = value;
            }
            let eta = design.dot(&beta);
            Self {
                design,
                enrichment,
                eta,
                family,
            }
        }

        fn draw(&self, extra_eta: &Array1<f64>, rng: &mut Lcg) -> Array1<f64> {
            self.eta
                .iter()
                .zip(extra_eta)
                .map(|(&base, &extra)| {
                    let eta = base + extra;
                    match self.family {
                        CanonicalExponentialFamily::BinomialLogit => {
                            let mean = 1.0 / (1.0 + (-eta).exp());
                            if rng.next_uniform() < mean { 1.0 } else { 0.0 }
                        }
                        CanonicalExponentialFamily::PoissonLog => {
                            // Knuth's multiplicative sampler: exact, and cheap at
                            // the small means this harness uses.
                            let floor = (-eta.exp()).exp();
                            let mut product = rng.next_uniform();
                            let mut count = 0.0;
                            while product > floor {
                                product *= rng.next_uniform();
                                count += 1.0;
                            }
                            count
                        }
                    }
                })
                .collect()
        }

        fn p_value(&self, response: &Array1<f64>) -> Option<BasisAdequacyResult> {
            let n = response.len();
            let null_fit = canonical_null_fit(
                self.design.view(),
                response.view(),
                Array1::<f64>::ones(n).view(),
                Array1::<f64>::zeros(n).view(),
                self.family,
            )?;
            conditional_basis_adequacy_test(self.enrichment.view(), self.design.view(), &null_fit)
        }

        /// Null p-values over `replicates` draws, and how many draws the
        /// expansion refused (`Σ` not positive definite or `c ≤ 0`).
        fn null_p_values(&self, replicates: usize, seed: u64) -> (Vec<f64>, usize) {
            let mut rng = Lcg(seed);
            let none = Array1::<f64>::zeros(self.eta.len());
            let mut p_values = Vec::with_capacity(replicates);
            for _ in 0..replicates {
                let response = self.draw(&none, &mut rng);
                if let Some(out) = self.p_value(&response) {
                    p_values.push(out.p_value);
                }
            }
            let refused = replicates - p_values.len();
            (p_values, refused)
        }
    }

    /// A refusal is "not measured", never a p-value, so it cannot bias the
    /// reported ones in either direction — but it must stay rarer than the
    /// Monte Carlo resolution at the smallest level `assert_uniform` checks, or
    /// the measured replicates would no longer stand for the null draws.
    fn assert_refusals_rare(refused: usize, replicates: usize, label: &str) {
        let level = 0.01_f64;
        let count = replicates as f64;
        let tolerance = 3.5 * (level * (1.0 - level) / count).sqrt();
        assert!(
            (refused as f64) / count <= tolerance,
            "{label}: {refused} of {replicates} replicates refused"
        );
    }

    /// The unconditional `χ²_r` reference is wrong at `O(r/n)` for a binomial
    /// response — in this harness its p-values fail uniformity at KS
    /// `p ≈ 1e-9` over 4000 replicates. The conditional reference holds both
    /// tails.
    #[test]
    fn conditional_test_is_uniform_under_the_null_binomial() {
        let harness = CanonicalHarness::new(
            200,
            6,
            12,
            &[0.3, 0.8, -0.5],
            CanonicalExponentialFamily::BinomialLogit,
        );
        let (p_values, refused) = harness.null_p_values(4000, 20_260_919);
        assert_refusals_rare(refused, 4000, "binomial conditional score test");
        assert_uniform(&p_values, "binomial conditional score test");
    }

    #[test]
    fn conditional_test_is_uniform_under_the_null_poisson() {
        let harness = CanonicalHarness::new(
            200,
            6,
            12,
            &[1.0, 0.5, -0.3],
            CanonicalExponentialFamily::PoissonLog,
        );
        let (p_values, refused) = harness.null_p_values(4000, 20_260_920);
        assert_refusals_rare(refused, 4000, "poisson conditional score test");
        assert_uniform(&p_values, "poisson conditional score test");
    }

    /// A truth with structure the degree-5 design cannot represent is rejected
    /// decisively, for both families.
    #[test]
    fn conditional_test_detects_a_basis_too_small_for_the_truth() {
        for (family, coefficients) in [
            (CanonicalExponentialFamily::BinomialLogit, [0.3, 0.8, -0.5]),
            (CanonicalExponentialFamily::PoissonLog, [1.0, 0.5, -0.3]),
        ] {
            let harness = CanonicalHarness::new(2000, 6, 12, &coefficients, family);
            let missing: Array1<f64> = (0..2000)
                .map(|row| (20.0 * (row as f64 + 0.5) / 2000.0).sin())
                .collect();
            let mut rng = Lcg(20_260_921);
            let response = harness.draw(&missing, &mut rng);
            let out = harness.p_value(&response).expect("measured");
            assert!(out.p_value < 1e-8, "{family:?}: p = {:.3e} under a missing sin(20x)", out.p_value);
        }
    }

    /// The null fit is the unpenalized MLE: its score is orthogonal to the
    /// design to within the certification band, from any start in the span.
    #[test]
    fn canonical_null_fit_solves_the_score_equation() {
        let harness = CanonicalHarness::new(
            300,
            4,
            6,
            &[0.2, 0.9],
            CanonicalExponentialFamily::BinomialLogit,
        );
        let mut rng = Lcg(99);
        let response = harness.draw(&Array1::zeros(300), &mut rng);
        let start = harness.design.column(1).mapv(|value| 3.0 * value);
        let fit = canonical_null_fit(
            harness.design.view(),
            response.view(),
            Array1::<f64>::ones(300).view(),
            start.view(),
            harness.family,
        )
        .expect("interior MLE");
        let gradient = harness.design.t().dot(&fit.score);
        let scale = harness.design.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        assert!(
            gradient.iter().all(|value| value.abs() < 1e-6 * scale * 300.0),
            "score equation residual {gradient:?}"
        );
    }

    #[test]
    fn canonical_null_fit_refuses_inadmissible_responses() {
        let harness = CanonicalHarness::new(
            20,
            2,
            3,
            &[0.0],
            CanonicalExponentialFamily::PoissonLog,
        );
        let mut response = Array1::<f64>::ones(20);
        response[3] = -1.0;
        assert!(
            canonical_null_fit(
                harness.design.view(),
                response.view(),
                Array1::<f64>::ones(20).view(),
                Array1::<f64>::zeros(20).view(),
                harness.family,
            )
            .is_none()
        );
    }
}
