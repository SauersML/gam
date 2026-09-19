// The #1063 per-term smooth significance test: a genuine likelihood-ratio
// statistic from a constrained refit, its Lawley Bartlett correction, and the
// reference distribution it is scored against (#2672).
//
// Split out of `spatial_optimization.rs` under the #780 line-count gate. It is
// `include!`d into `drivers/mod.rs` alongside the driver it came from, so it
// keeps the same flat namespace and the same import surface — nothing here
// changed except which file it lives in.

// The λ̂-selection replay the reference is read against lives in `gam-terms`, so
// the summary Wald test below the fit orchestration can read against it too.
pub use gam_terms::inference::selection_replay::{
    SmoothLrSelection, SmoothLrSelectionDecline, SmoothLrSelectionReplay,
};
use gam_terms::inference::selection_replay::lr_tested_block;

/// Provenance tag for the smooth-term significance correction (#1063): which
/// statistic the reported p-value is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrCorrection {
    /// A per-term LR statistic corrected by the full estimated-λ Lawley factor,
    /// including the ρ̂-sampling-variation contribution from the regularized
    /// inverse REML/LAML outer Hessian.
    LawleyLrEstimatedLambda,
    /// A per-term likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)` that has
    /// been Bartlett-corrected with the fixed-λ Lawley factor `c = E[W|λ]/d`
    /// (`W* = W/c`, referenced against `χ²_d`). This is used only when the
    /// estimated-λ handoff is unavailable.
    LawleyLrFixedLambda,
    /// No second-order correction was applied — either the family has no
    /// closed-form Lawley cumulant jets or the null refit did not converge — so
    /// the uncorrected `χ²_d` of the raw LR statistic stands.
    None,
}

impl SmoothLrCorrection {
    /// The serialized provenance label surfaced in the summary table.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrCorrection::LawleyLrEstimatedLambda => "lawley_lr_estimated_lambda",
            SmoothLrCorrection::LawleyLrFixedLambda => "lawley_lr_fixed_lambda",
            SmoothLrCorrection::None => "none",
        }
    }
}

/// Which lane supplied a [`SmoothLrReferenceDf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrReferenceSource {
    /// The statistic's own null spectrum `w`, in full, scored by inverting its
    /// moment generating function along a saddle-point contour. This is the
    /// exact lane: the reference IS the null law, not a distribution fitted to
    /// some of its moments.
    ///
    /// The spectrum is assembled from `[H⁻¹]_jj` and the term's own λ-weighted
    /// penalty block through the symmetric similarity
    /// `w_j = 1 − eig(B^{1/2} S_jj B^{1/2})²` — see
    /// `lr_tested_block` for why that is the same spectrum as
    /// `eig(2·F_jj − F_jj²)` and why it is the better-conditioned way to reach
    /// it.
    NullSpectrum,
    /// `[H⁻¹]_jj` or the penalty block was unavailable, but the
    /// coefficient-influence matrix was, so only the first two *moments* of the
    /// spectrum are recoverable (`tr A` and `tr A²` for `A = 2F_jj − F_jj²`,
    /// both traces of powers of one block). The reference is then the
    /// two-moment match `g·χ²_ν`.
    ///
    /// It is EXACT at both ends of the shrinkage range and wrong in between,
    /// which is worth stating precisely because the ends are where the intuition
    /// goes. An unpenalized term has `w ≡ 1` and the match is the textbook
    /// `χ²_q`; a term REML has shrunk to its null space has one weight of order
    /// one over a tail of dust — measured on a null-true `k = 12` fit, `w =
    /// (0.322, 5.9e-7, 7.1e-8, …)` — and a single distinct weight is a scaled
    /// chi-square exactly. The gap opens at moderate shrinkage, where several
    /// weights are comparable and unequal: on `f_j = 1/(1 + λγ_j)` for a
    /// second-difference penalty at `λ = 0.01`, `k = 20`, the size delivered at
    /// a nominal `α` is `1.02×` at `0.05`, `1.11×` at `0.01`, `1.31×` at `10⁻³`
    /// and `1.61×` at `10⁻⁴` — one-signed, anti-conservative, and worse the
    /// deeper the tail.
    ///
    /// It is a surrogate for the lane above, not a different claim about the
    /// statistic.
    SpectralMomentMatch,
    /// Neither the spectrum nor its moments were recoverable, so the reference
    /// falls back to the classical unit-weight shape
    /// `χ²_{max(edf, null_dim, 1)}` — every retained direction counted as if it
    /// were unpenalized. It is the only reference recoverable from a scalar
    /// EDF, and it is conservative for the same reason the whole pre-#2672
    /// assembly was: unit weights over-state the statistic's spread.
    UnitWeightFallback,
}

/// The reference distribution [`SmoothTermLrInference`] scores its statistic
/// against, reported as the two spectral moments it is built from (#2672).
///
/// # What the statistic's null law actually is
///
/// Expand the log-likelihood quadratically about the unpenalized MLE `β̃` and
/// write `I = X'WX`, `S` for the penalty, `H = I + S`, `j` for the tested block
/// and `n` for the retained one. The penalized fit is `β̂ = Fβ̃` with
/// `F = H⁻¹I`, and the null fit is the retained block's own projection, so
///
/// ```text
/// W = β̃_j' (Ĩ_jj − N) β̃_j ,   β̃_j ~ N(0, Ĩ_jj⁻¹)
/// ```
///
/// with `Ĩ_jj = I_jj − I_jn I_nn⁻¹ I_nj` the Schur complement and
/// `N = [S H⁻¹ I H⁻¹ S]_jj`. Setting `H̃ = Ĩ_jj + S_jj` and `P = H̃⁻¹S_jj`, that
/// collapses to `(Ĩ_jj − N)Ĩ_jj⁻¹ = H̃(I − P²)H̃⁻¹`, and — because the block of
/// the GLOBAL influence matrix equals the Schur-complement influence,
/// `F_jj = H̃⁻¹Ĩ_jj = I − P` — the eigenvalues are exactly `2F_jj − F_jj²`.
/// So
///
/// ```text
/// W = Σ_j w_j χ²_1 ,   w = eig(2·F_jj − F_jj²) ∈ (0, 1]^q.
/// ```
///
/// # Consequences, and what this replaced
///
/// `Σ w_j = 2 tr(F_jj) − tr(F_jj²)` is Wood's `edf1`. So `edf1` is not a
/// citation here, it is the statistic's first-order null MEAN, derived. What it
/// is not is a chi-square degrees of freedom: `Var(W) = 2 Σ w_j²` against the
/// mean-matched `χ²_{Σw}`'s `2 Σ w_j`, and `w_j ≤ 1`, so a mean-matched chi-square
/// is over-dispersed for every penalized term and the test is conservative by
/// construction.
///
/// # Why the reference is the spectrum and not two of its moments
///
/// Matching the second moment as well — `W ≈ g·χ²_ν` with `ν = (Σw)²/Σw²`,
/// `g = Σw²/Σw` — fixes the *shape* with no free constant, and is EXACT whenever
/// the weights are equal (which includes the classical unpenalized case
/// `w ≡ 1 ⇒ ν = q, g = 1 ⇒ χ²_q`). It is not exact otherwise, and the error is
/// one-signed and grows with the depth of the tail, which is the half of the
/// p-value range that decides anything. Measured against the exact law on the
/// shrinkage spectrum `f_j = 1/(1 + λγ_j)` of a second-difference penalty, over
/// six decades of `λ` and `k ∈ {6, 12, 20}`, the size a two-moment reference
/// actually delivers at a nominal `α` is
///
/// ```text
/// α = 0.05   0.99 – 1.02 ×      α = 1e-3   1.01 – 1.31 ×
/// α = 0.01   1.00 – 1.11 ×      α = 1e-4   1.14 – 1.61 ×
/// ```
///
/// i.e. it is fine where the test is least discriminating and up to 61%
/// anti-conservative where it is most.
///
/// Where the gap lives matters as much as its size, and it is not where the
/// intuition puts it. The surrogate is exact at BOTH ends of the shrinkage
/// range — `w ≡ 1` unpenalized, and a single distinct weight once REML has
/// shrunk a term to its null space (measured on a null-true `k = 12` fit:
/// `w = (0.322, 5.9e-7, 7.1e-8, …)`, where the two references agree to eight
/// figures). It opens in the middle, at moderate shrinkage, which is exactly
/// where a smooth term carrying real signal sits. Nothing about the statistic requires that
/// trade: the weights are the parameters of an exactly invertible
/// characteristic function, and
/// `gam_math::probability::signed_weighted_chi_square_sf`
/// inverts it with a *returned* error bound, relative to the tail itself, so
/// the smallest tail any of the numbers above resolves is resolved to near
/// full precision. So the
/// reference is `P(Σ_j w_j χ²_1 > W)` itself, and the `(ν, g)` pair survives only
/// as a two-number summary of the spectrum's shape, published for continuity and
/// no longer consulted when the spectrum is known.
///
/// The one-moment reference this replaced sits at the far end of the same axis,
/// with the same sign: on a spectrum shaped like a shrunk smooth
/// (`0.08, 0.02, 0.005, 0.001, 2e-4`), at `x = 8·Σw` the exact tail is `1.4e-3`
/// and the mean-matched chi-square reports `3.6e-2` — 26× conservative.
///
/// # What went away with the mean-only reference
///
/// Three things, and none of them needed a replacement:
///
/// * `+ tr(X'WX · J Var(ρ̂) Jᵀ)/φ`, the Wood–Pya–Säfken smoothing-parameter
///   inflation added under #1872. That is a *coefficient-covariance* correction
///   for AIC; it is not a term in this statistic's null law, and it is largest
///   exactly where the outer criterion is flattest — i.e. where the term has the
///   LEAST effective d.f. Measured on the #2672 fixture: a replicate with
///   `edf = 0.070` was handed `rho_uncertainty = 1.79`, twenty-five times the
///   term's own effective d.f. It was holding the size up by an unrelated
///   mechanism. λ̂'s sampling variation enters `E[W]` through the estimated-λ
///   Lawley shift already applied as the Bartlett factor, at the `O(n⁻¹)` order
///   it belongs to.
/// * `.max(edf)` and `.max(null_dim)`. Both are automatic: `w_j = 1` exactly on
///   an unpenalized direction, so `Σ w_j ≥ null_dim` by construction, and `Σ w_j`
///   dominates `tr(F_jj) = edf` because `w_j = 2f_j − f_j² ≥ f_j` for `f_j ∈ [0,1]`.
/// * `.max(1.0)`, the #1766 degeneracy floor. It existed because `χ²_d` with
///   `d → 0` reports any positive `W` as maximally significant. The scaled
///   reference cannot degenerate that way: as REML shrinks a term the weights and
///   the statistic collapse *together*, `W/g` stays `O(1)`, and `ν → q`. The floor
///   was a patch on the wrong shape, not on a missing quantity.
///
/// # And what all of it is the reference FOR
///
/// Everything above describes the law of `Q`, the statistic a KNOWN-scale
/// likelihood ratio is. A profiled Gaussian's `W` is not `Q` — it is
/// `n·ln(1 + Q/V) + B`, with `V` the residual sum of squares the same fit
/// estimated its `σ̂` from. Scoring `W` against `Q`'s law is anti-conservative
/// at `O(1/ν)`: measured `size@.05 = 0.0792` pooled over 480 replicates at
/// `n ∈ {30, 50}` against a nominal `0.05`, with the Lawley factor inert
/// throughout, so the whole miss belonged here. [`Self::profiled_scale`] is
/// that channel and [`SmoothLrProfiledScale`] is the derivation; it is `None`
/// on every family whose dispersion is already inside the IRLS weight.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothLrReferenceDf {
    /// The null spectrum itself, `w_j ∈ [0, 1]`, sorted descending — the whole
    /// reference on the [`SmoothLrReferenceSource::NullSpectrum`] lane. Empty on
    /// the two lanes that could not reach it, which is exactly the condition
    /// under which `Self::tail_probability_with_bound` falls back to the `(ν, g)` pair.
    pub weights: Vec<f64>,
    /// First spectral moment `Σ_j w_j = 2·tr(F_jj) − tr(F_jj²)` — Wood's `edf1`,
    /// and exactly the statistic's first-order null mean `E[W|λ]`. This is the
    /// `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated in.
    pub mean: f64,
    /// Second spectral moment `Σ_j w_j² = tr((2F_jj − F_jj²)²)`, i.e. `Var(W)/2`.
    pub second_moment: f64,
    /// Shape of the two-moment SUMMARY `ν = mean²/second_moment`. It is what the
    /// reference used to be, and it is still what the reference is on the
    /// [`SmoothLrReferenceSource::SpectralMomentMatch`] and
    /// [`SmoothLrReferenceSource::UnitWeightFallback`] lanes; on the exact lane
    /// it is a published descriptor of the spectrum's shape and is not consulted.
    pub chi_square_df: f64,
    /// Scale of that summary, `g = second_moment/mean`. Same status as
    /// [`Self::chi_square_df`].
    pub scale: f64,
    /// The agreement between the two independently-assembled routes to this
    /// spectrum, when the fit supplied the inputs for both: the larger of the
    /// two relative residuals between `(Σw, Σw²)` read off `[H⁻¹]_jj S_jj` and
    /// `(tr A, tr A²)` read off the influence block, `A = 2F_jj − F_jj²`.
    ///
    /// The two are the same object by an algebraic identity that depends on the
    /// penalty being block-diagonal by term AND on `Vb`, `F` and `S` being
    /// published in one coefficient basis. Neither is checkable by inspection,
    /// and both have been wrong here before (`#2672`'s similarity-map drop, its
    /// internal-basis first-order correction, and its block-local
    /// `coeff_range`). So the driver measures the identity on every fit that can
    /// support it and publishes the number rather than assuming it.
    ///
    /// `None` when only one route was available — which is a statement about the
    /// fit, not a failure.
    pub moment_residual: Option<f64>,
    /// The term's conditional effective degrees of freedom `tr(F_jj)`
    /// (`per_term_edf`), reported for continuity with the summary table and used
    /// as the fallback base when neither the spectrum nor its moments are
    /// available.
    pub edf: f64,
    /// The term's joint unpenalized null-space dimension `dim(∩_k null(S_k))`,
    /// reported because it is the analytic lower bound on `mean` and therefore
    /// the cheapest check that the spectrum was assembled on the right block.
    pub null_dim: usize,
    /// Which lane supplied the reference.
    pub source: SmoothLrReferenceSource,
    /// The λ̂-selection replay, or the NAMED reason there is none (#2672).
    ///
    /// A decline means the conditional law IS the selection law here — nothing
    /// was selected — and the tail is read from [`Self::weights`] alone. It is
    /// an enum rather than an `Option` because a missing replay is a statement
    /// about the fit, and a reader who does not have to look at which statement
    /// will not.
    pub selection: SmoothLrSelection,
    /// The ESTIMATED-SCALE channel, present exactly when the fit profiled its
    /// own Gaussian dispersion out of a residual sum of squares (#2672).
    ///
    /// `None` is a statement about the family, not a missing measurement: every
    /// other likelihood on this path carries its dispersion in the IRLS weight
    /// and its `W` is not a function of a second, independently-estimated
    /// scalar. See [`SmoothLrProfiledScale`].
    pub profiled_scale: Option<SmoothLrProfiledScale>,
}

/// What the reference needs in order to score a statistic whose SCALE was
/// estimated from the same residuals (#2672).
///
/// # The statistic is a ratio, exactly
///
/// gam's profiled Gaussian log-likelihood is `ℓ = −½[n·ln 2π + n·ln(D/ν) −
/// Σ ln w_i + ν]` with `D` the weighted residual sum of squares and `ν` the
/// residual degrees of freedom it divides by — so the whole-term LR statistic
/// is, with no expansion anywhere,
///
/// ```text
///   W = 2(ℓ_full − ℓ_null) = n·ln(D_0/D_f) + n·ln(ν_f/ν_0) + (ν_0 − ν_f)
///     = n·ln(1 + Q/V) + B,
///   Q = (D_0 − D_f)/σ²,   V = D_f/σ²,   B = n·ln(ν_f/ν_0) + (ν_0 − ν_f).
/// ```
///
/// `Q` is what [`SmoothLrReferenceDf::weights`] is the null spectrum OF, and
/// the known-scale reference scores it directly. The shipped reference
/// therefore answers the question "how extreme is `Q`" when the question asked
/// was "how extreme is `n·ln(1 + Q/V) + B`" — and `V` is a random variable of
/// the same data, with mean `ν` and spread `√(2ν)`. Both the mean shift
/// `ν/(ν−2)` and the extra spread push the test anti-conservative, and both are
/// `O(1/ν)`: invisible at `n = 1000`, worth `0.03` in size at `n = 30`. It is
/// the same reason mgcv's smooth-term p-values take an `F` reference when the
/// scale is estimated and a `χ²` when it is known.
///
/// # Inverting it costs nothing, because the map is monotone
///
/// `W > w ⟺ Q/V > exp((w − B)/n) − 1`, so
///
/// ```text
///   P(W > w) = P( Q − c(w)·V > 0 ),   c(w) = expm1((w − B)/n),
/// ```
///
/// a linear combination of independent chi-squares with a NEGATIVE weight,
/// evaluated at zero — which is exactly
/// [`gam_math::probability::signed_weighted_chi_square_sf`]. There
/// is no expansion, no `κ`-convention to pick, and no separate `F`-family
/// approximation: `n` and `ν` appear where the log-likelihood actually put
/// them.
///
/// # Where the residual law comes from
///
/// The same spectral object the numerator uses, taken over the whole model
/// instead of over the tested block. The profiled Gaussian's hat matrix
/// `A = X H⁻¹X'W` is symmetric in the whitened coordinates the weighted RSS is
/// a sum of squares in (`X̃ = W^{1/2}X`, where `ε̃ = W^{1/2}ε` has covariance
/// `σ²I` because `Var(y_i) = σ²/w_i`), with eigenvalues `f_i = 1 − p_i` — the
/// `p_i` being the penalty shares `lr_tested_block` returns, which are
/// unchanged by the whitening since `H⁻¹S` is — and `n − p` further zeros. The
/// true mean is annihilated because it lies in the penalty's null space. So
///
/// ```text
///   V = ε̃'(I − Ã)²ε̃ ~ Σ_i p_i²·χ²_1  +  χ²_{n−p},
/// ```
///
/// exact at fixed `λ`, with `n` the POSITIVE-WEIGHT row count in both places.
/// The `n − p` unit directions are folded into ONE term with `n − p` degrees of
/// freedom, which is what keeps an `n`-sized reference the same cost as a
/// `p`-sized one.
///
/// # What is approximated, stated plainly
///
/// Three things, all inherited rather than introduced, and none of them the
/// `O(1/ν)` term this channel exists to remove.
///
/// * `Q`'s spectrum is the reference's own claim, unchanged.
/// * `Q` and `V` are taken INDEPENDENT — exact for the unpenalized linear model
///   by Cochran, approximate under penalization, and the same independence
///   every `F` reference for a penalized smooth rests on.
/// * `V` is taken CENTRAL. It is exactly central when the mean lies in the
///   penalty's null space, which is what the tested term being null gives on
///   its own block; a DIFFERENT term in the model carrying real signal that the
///   penalty shrinks adds a non-centrality. That inflates `V`, which inflates
///   the p-value — so the residual runs conservative, in the direction a test
///   is allowed to be wrong.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothLrProfiledScale {
    /// `n` — the multiplier the profiled `ln σ̂²` carries in the log-likelihood.
    pub observations: f64,
    /// `B = n·ln(ν_f/ν_0) + (ν_0 − ν_f)`, the part of `W` that is a function of
    /// the two fits' residual degrees of freedom and of nothing random.
    ///
    /// It is NOT negligible and it is not the same sign as the rest: on the
    /// `n = 30` Gaussian cells it runs `−0.13` to `−0.61`.
    pub deterministic_offset: f64,
    /// `p_i²` over the whole model's penalty shares — the non-trivial half of
    /// the residual quadratic form's spectrum.
    pub residual_weights: Vec<f64>,
    /// `n − p`, the residual directions no design column reaches. Each carries
    /// weight exactly one, so they are one term with this many degrees of
    /// freedom rather than this many terms.
    pub residual_unit_dimension: f64,
}

impl SmoothLrReferenceDf {
    /// `P(W > statistic)` under this reference, with an absolute bound on its
    /// error.
    ///
    /// On the exact lane this is `P(Σ_j w_j χ²_1 > W)` by inversion of its moment
    /// generating function; on the two surrogate lanes it is the two-moment
    /// `P(χ²_ν > W/g)`. Both are scale-equivariant in the same way, which is what
    /// lets the Bartlett correction be applied as `W/c` on either.
    ///
    /// A non-finite statistic propagates as `NaN` rather than being scored: the
    /// LR statistic is `NaN` exactly when the null refit did not produce a finite
    /// log-likelihood, and there is no p-value for a test that was not run.
    ///
    /// The bound is the inversion's own, derived from the arithmetic that
    /// produced the value (see `gam_math::probability::signed_weighted_chi_square_sf`),
    /// plus twice the selection replay's Monte-Carlo standard error when a
    /// replay corrects it.
    pub fn tail_probability_with_bound(&self, statistic: f64) -> (f64, f64) {
        let replay = match &self.selection {
            SmoothLrSelection::Replayed(replay) => replay,
            SmoothLrSelection::Declined(reason) if reason.is_refusal() => {
                return (f64::NAN, f64::NAN);
            }
            SmoothLrSelection::Declined(_) => {
                return self.conditional_tail_with_bound(statistic);
            }
        };
        let (conditional, bound) = self.conditional_tail_with_bound(statistic);
        if !conditional.is_finite() {
            return (conditional, bound);
        }
        let (shift, standard_error) = replay.tail_shift(self.selection_threshold(statistic));
        (
            (conditional + shift).clamp(0.0, 1.0),
            bound + 2.0 * standard_error,
        )
    }

    /// The threshold the λ̂-selection replay has to be asked about, which is not
    /// the statistic itself once the scale is profiled.
    ///
    /// The replay samples the statistic's KNOWN-SCALE law `Q` under two ways of
    /// choosing `λ`, and its shift is the difference of the two tails at a
    /// `Q`-threshold. With an estimated scale the event `W > w` is
    /// `Q > c(w)·V`, so the `Q`-threshold is random; the selection correction is
    /// `E_V[Δ(c(w)·V)]`, and evaluating `Δ` at `E[V]` is the first-order term of
    /// that expectation. `E[V] = Σ_i v_i + (n − p)` is the residual law's own
    /// mean, already carried.
    ///
    /// As `ν → ∞` this returns the statistic: `B → 0`, `E[V]/n → 1`, and
    /// `expm1(w/n)·n → w`. So the correction composes with the known-scale
    /// behaviour rather than replacing it.
    fn selection_threshold(&self, statistic: f64) -> f64 {
        let Some(scale) = self.profiled_scale.as_ref() else {
            return statistic;
        };
        let residual_mean: f64 = scale.residual_weights.iter().sum::<f64>()
            + scale.residual_unit_dimension;
        let ratio = ((statistic - scale.deterministic_offset) / scale.observations).exp_m1();
        ratio.max(0.0) * residual_mean
    }

    /// The statistic's OWN null law, as a list of `λ_j·χ²_{h_j}` terms: the
    /// exact spectrum where the fit supplied one, and the two-moment summary
    /// `(g, ν)` where it did not.
    ///
    /// The two lanes are one object here rather than two branches because the
    /// summary is not an approximation of a different shape — it is the same
    /// linear combination with one term. Reading it this way is what lets the
    /// profiled-scale route below apply on every lane instead of only on the
    /// lane that reached the spectrum.
    fn null_law_terms(&self) -> Vec<gam_math::probability::WeightedChiSquareTerm> {
        use gam_math::probability::WeightedChiSquareTerm;
        if self.weights.is_empty() {
            return vec![WeightedChiSquareTerm {
                weight: self.scale,
                degrees_of_freedom: self.chi_square_df,
            }];
        }
        self.weights
            .iter()
            .map(|&weight| WeightedChiSquareTerm {
                weight,
                degrees_of_freedom: 1.0,
            })
            .collect()
    }

    /// The CONDITIONAL tail — the fixed-`λ` law alone, with the λ̂-selection
    /// replay held out — and its certified truncation bound. This is the tail
    /// [`Self::tail_probability_with_bound`] reports when nothing was selected,
    /// and it is the law the replay corrects.
    ///
    /// It is a building block of the reference, not a p-value: read against a
    /// fitted `λ̂` it prices the smoothing parameter as given and is
    /// anti-conservative, which is why [`SmoothTermLrInference`] does not
    /// publish it.
    pub fn conditional_tail_with_bound(&self, statistic: f64) -> (f64, f64) {
        if !statistic.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let summary =
            |w: f64| gam_math::probability::chi_square_sf(w / self.scale, self.chi_square_df);
        if self.weights.is_empty() && self.profiled_scale.is_none() {
            // The summary IS the reference on the two degraded lanes, and it is
            // a closed form: no truncation, so no bound to report.
            return (summary(statistic), 0.0);
        }
        let mut terms = self.null_law_terms();
        let Some(scale) = self.profiled_scale.as_ref() else {
            return tail_with_bound(&terms, statistic);
        };
        // `W > w  ⟺  Q/V > expm1((w − B)/n)`, so the tail is the SIGNED
        // combination `Q − c·V` at zero. See [`SmoothLrProfiledScale`].
        let ratio = ((statistic - scale.deterministic_offset) / scale.observations).exp_m1();
        if !ratio.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        if ratio <= 0.0 {
            // `W` is at or under the value the two fits' degrees of freedom
            // alone produce. `Q ≥ 0` and `V > 0`, so `Q − cV ≥ 0` with
            // certainty and the statistic is not evidence of anything.
            return (1.0, 0.0);
        }
        terms.extend(scale.residual_weights.iter().map(|&weight| {
            gam_math::probability::WeightedChiSquareTerm {
                weight: -ratio * weight,
                degrees_of_freedom: 1.0,
            }
        }));
        if scale.residual_unit_dimension > 0.0 {
            terms.push(gam_math::probability::WeightedChiSquareTerm {
                weight: -ratio,
                degrees_of_freedom: scale.residual_unit_dimension,
            });
        }
        tail_with_bound(&terms, 0.0)
    }
}

/// The weighted chi-square tail with its bound stated absolutely, the form the
/// replay's shift is added to.
fn tail_with_bound(terms: &[gam_math::probability::WeightedChiSquareTerm], statistic: f64) -> (f64, f64) {
    let tail = gam_math::probability::signed_weighted_chi_square_sf(terms, statistic);
    (tail.probability, tail.absolute_error())
}

/// A smooth term the per-term LR test does not report, with the typed reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmoothTermLrUnavailable {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// Why no p-value exists for this term.
    pub reason: gam_solve::estimate::SmoothPValueUnavailable,
}

/// The smooth terms of `resolvedspec` that [`smooth_term_lr_inference_forspec`]
/// cannot test, each with its typed reason. A shape-constrained term's null
/// `f = 0` is the apex of its constraint cone, so the LR statistic has no
/// calibrated reference law (see [`gam_solve::estimate::SmoothPValueUnavailable`]).
pub fn smooth_term_lr_unavailable_forspec(
    resolvedspec: &TermCollectionSpec,
) -> Vec<SmoothTermLrUnavailable> {
    resolvedspec
        .smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(term_idx, term)| {
            gam_solve::estimate::smooth_pvalue_unavailable(&term.shape).map(|reason| {
                SmoothTermLrUnavailable {
                    name: term.name.clone(),
                    term_idx,
                    reason,
                }
            })
        })
        .collect()
}

/// The Bartlett-corrected per-term significance report for one penalized smooth
/// term (#1063). Unlike the summary table's Wood rank-truncated **Wald**
/// statistic, this is a genuine **likelihood-ratio** statistic from a
/// constrained refit (the smooth dropped), so the exact Lawley LR Bartlett
/// factor corrects the right quantity.
#[derive(Clone, Debug)]
pub struct SmoothTermLrInference {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// The uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)`,
    /// floored at zero (a non-negative LR by construction).
    pub statistic_lr: f64,
    /// The statistic's first-order null mean `d = E[W|λ] = Σ_j w_j`, which is
    /// Wood's `edf1 = 2·tr(F_bb) − tr(F_bb²)` exactly (see
    /// [`SmoothLrReferenceDf`] for why that is a derivation and not a citation).
    /// This is the `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated
    /// in. It is **not** a chi-square degrees of freedom — the reference the
    /// p-values are read from is [`Self::ref_df_provenance`]'s
    /// `chi_square_df`/`scale` pair, which coincides with `ref_df` only when the
    /// tested block is unpenalized.
    pub ref_df: f64,
    /// The reference distribution itself: both spectral moments of the null law,
    /// the `(ν, g)` pair resolved from them, and which lane supplied it (#2672).
    pub ref_df_provenance: SmoothLrReferenceDf,
    /// Lawley LR Bartlett factor `c = E[W]/d = 1 + Δε/d` when computable, else
    /// `1.0` (no correction).
    pub bartlett_factor: f64,
    /// Fixed-λ conditional factor `c_cond = 1 + Δε(ρ̂)/d` when the estimated-λ
    /// correction was applied. `None` means the applied factor was either the
    /// fixed-λ factor itself or no Lawley correction was available.
    pub bartlett_factor_conditional: Option<f64>,
    /// Increment in Lawley's LR mean shift due solely to ρ̂ sampling variation,
    /// `0.5 * tr(H_Δε Cov(ρ̂))`, when estimated-λ correction was applied.
    pub rho_variation_shift: Option<f64>,
    /// Bartlett-corrected statistic `W* = W / c`.
    pub statistic_corrected: f64,
    /// The term's p-value: `P(W > W*)` under [`Self::ref_df_provenance`] — the
    /// statistic's own null law with the λ̂-selection replay applied
    /// ([`SmoothLrReferenceDf::tail_probability_with_bound`]), read at the
    /// Bartlett-corrected statistic. Dividing the statistic by `c` and scaling
    /// every spectral weight by `c` are the same operation on this reference, so
    /// the Bartlett correction composes without a second convention.
    ///
    /// This is the only p-value the report publishes. The fixed-`λ` conditional
    /// tail and the uncorrected tail it replaces are anti-conservative under the
    /// null (they price `λ̂` as given and drop the `O(1/n)` mean shift), so they
    /// are not offered as alternatives; the reference is published whole, so a
    /// consumer who wants to see how the correction moved the answer can still
    /// evaluate it at `statistic_lr`.
    ///
    /// NaN when there is no answer to publish: the statistic itself is NaN (a
    /// null refit refused), or `λ̂` was chosen but its selection replay refused
    /// ([`SmoothLrSelectionDecline::is_refusal`]; the reason is in
    /// `ref_df_provenance.selection`). A fixed-`λ` tail is never substituted.
    pub p_value: f64,
    /// Whether the second-order correction is **material** (#939 deliverable 4):
    /// the per-test diagnostic "is `n` too small for first-order inference
    /// *here*?". `true` when a correction was applied and it moves the result by
    /// more than [`SMOOTH_LR_MATERIAL_THRESHOLD`] — measured as the larger of the
    /// relative Bartlett-factor distance from one `|c − 1|` and the relative
    /// p-value change `|p* − p| / max(p, p*, ε)`. `false` when `correction` is
    /// [`SmoothLrCorrection::None`] (no correction was applied).
    pub material: bool,
    /// Which statistic the p-value is built from.
    pub correction: SmoothLrCorrection,
}

/// The materiality threshold for [`SmoothTermLrInference::material`] (#939
/// deliverable 4): a correction is flagged material when it changes the result
/// by more than 10%.
pub const SMOOTH_LR_MATERIAL_THRESHOLD: f64 = 0.10;

/// Build `S_b = lambda_b * S_b^unit` as global `p_total x p_total` matrices in
/// exactly the fitted rho/lambda ordering. This is the narrow handoff the
/// estimated-lambda Lawley correction needs: the same `design.penalties` order
/// already paired with `fit.lambdas`, without changing #740's outer-Hessian
/// algebra or the production penalty assembly.
fn fitted_rho_penalty_components(
    penalties: &[BlockwisePenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Result<Vec<gam_terms::inference::lawley::RhoPenaltyComponent>, EstimationError> {
    if penalties.len() != lambdas.len() {
        return Err(EstimationError::InvalidInput(format!(
            "smooth_term_lr_inference: penalty/lambda count mismatch ({} penalties, {} lambdas)",
            penalties.len(),
            lambdas.len()
        )));
    }
    let mut components = Vec::with_capacity(penalties.len());
    for (idx, (penalty, &lambda)) in penalties.iter().zip(lambdas.iter()).enumerate() {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: lambda[{idx}] is invalid: {lambda}"
            )));
        }
        let r = &penalty.col_range;
        if r.end > p_total {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: penalty[{idx}] range {:?} exceeds coefficient dimension {p_total}",
                r
            )));
        }
        let mut s_component = Array2::<f64>::zeros((p_total, p_total));
        s_component
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(lambda, &penalty.local);
        components.push(gam_terms::inference::lawley::RhoPenaltyComponent { s_component });
    }
    Ok(components)
}

/// The end-to-end per-term likelihood-ratio significance report for every
/// penalized (shape-unconstrained) smooth term in a fitted model, magically
/// Bartlett-corrected when the family carries closed-form Lawley cumulant jets
/// (#1063, follow-up to #939).
///
/// # Why an LR statistic (not the summary Wald)
///
/// The summary table's `wood_smooth_test` is Wood's rank-truncated **Wald**
/// statistic `T = β̂'Σ̂⁻β̂`. Lawley's ε corrects the **likelihood-ratio**
/// statistic, and under penalization the Wald form is already a weighted χ²
/// whose second-order mean is *not* `d + Δε` — dividing `T` by the LR factor
/// would correct the wrong statistic. The principled route (#1063 Option 1) is
/// to compute a real per-term LR statistic by a constrained refit and correct
/// *that*:
///
/// ```text
/// W = 2(ℓ_full − ℓ_null),   W* = W / c,   c = 1 + Δε/d,   p = P(χ²_d > W*).
/// ```
///
/// # Method
///
/// 1. Fit the full model and read `ℓ_full` and the per-term coefficient ranges /
///    EDF / influence block. The full design's column layout fixes the tested
///    block for the Lawley factor.
/// 2. For each penalized smooth term, refit a null model with that term dropped
///    from the spec; `W = max(2(ℓ_full − ℓ_null), 0)`.
/// 3. The reference d.f. `d` is the Wood truncation `tr(F)²/tr(F²)` on the
///    term's influence block (the same `ref_df` the summary Wald row reports),
///    floored at `max(edf, null_dim, 1)`: this LR test drops the whole term, so
///    `d` is at least the dimension the term spans when present (its null-space
///    dimension, never below 1). The non-symmetric `tr(F²)` can collapse toward
///    0 at a shrunk-to-null fit and violate that bound — see the inline note at
///    the `ref_df` binding.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`gam_terms::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets, or a null refit that did not converge) the
///    uncorrected `χ²_d` stands with provenance `none` — never weakened.
///
/// Random-effect smooths are skipped (their tests are not a central-χ² LR).
/// Shape-constrained smooths are skipped too; they have no calibrated LR
/// reference, and [`smooth_term_lr_unavailable_forspec`] names them with the
/// typed reason, matching the summary table's policy.
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrInference>, EstimationError> {
    use gam_terms::inference::lawley::{
        LAWLEY_PAIR_MATRIX_MAX_ROWS, known_scale_expected_jets_with_dispersion,
        lawley_lr_bartlett_factor, lawley_lr_mean_shift_with_rho_variation,
    };

    let n = data.nrows();
    // Full fit: ℓ_full, the per-term coefficient ranges/EDF/influence, and the
    // full design whose column layout fixes each tested block for Lawley.
    let full = fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        family.clone(),
        options,
    )?;
    let ll_full = full.fit.log_likelihood;
    let p_total = full.design.design.ncols();
    let lambdas = full.fit.lambdas.as_slice().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth_term_lr_inference: non-contiguous lambda vector".to_string(),
        )
    })?;
    let s_lambda = weighted_blockwise_penalty_sum(&full.design.penalties, lambdas, p_total);
    let rho_penalty_components =
        fitted_rho_penalty_components(&full.design.penalties, lambdas, p_total)?;
    // One `ρ` per penalty component is an invariant of the fit, so a covariance
    // of any other shape is a defect upstream — not a reason to quietly price
    // the Bartlett factor as if `λ̂` were fixed.
    let rho_covariance = full.fit.artifacts.rho_covariance.as_ref();
    if let Some(cov) = rho_covariance
        && (cov.nrows() != rho_penalty_components.len()
            || cov.ncols() != rho_penalty_components.len())
    {
        return Err(EstimationError::InvalidInput(format!(
            "smooth_term_lr_inference: rho covariance is {}x{} but the fit has {} penalty \
             components",
            cov.nrows(),
            cov.ncols(),
            rho_penalty_components.len()
        )));
    }
    // Full design as a dense n×p array for the Lawley pair-matrix reduction.
    let full_design_dense = full.design.design.to_dense();
    let influence = full.fit.coefficient_influence();
    // `H⁻¹`, unscaled: `beta_covariance()` publishes `Vb = H⁻¹·scale`, and the
    // scale is the family's own documented coefficient-covariance multiplier
    // (`σ̂²` for the profiled Gaussian, `1` for every family whose IRLS weight
    // already carries the dispersion). The null spectrum is `1 − eig(H⁻¹_jj
    // S_jj)²`, a product of two matrices in reciprocal units, so the multiplier
    // has to come off exactly here or every weight is wrong by that factor.
    // A family with no scalar multiplier (custom/GAMLSS) yields `None` and the
    // reference drops to the two-moment rung, which needs no scale at all.
    let hessian_inverse = full
        .fit
        .coefficient_covariance_scale()
        .ok()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .zip(full.fit.beta_covariance())
        .map(|(scale, covariance)| covariance.mapv(|value| value / scale));
    // `SmoothTerm::coeff_range` is BLOCK-LOCAL — 0-based within the smooth block
    // — while the global coefficient layout is `[intercept | linear | random |
    // smooth]`. Every consumer that indexes a global object with it has to shift
    // by `smooth_start` first (`smooth_term_summary.rs`, the constraint audit and
    // the anisotropic provider all do). This driver did not, and it indexes FOUR
    // global objects with it: the influence matrix `F` (both the per-term EDF
    // trace and Wood's `edf1`), the weighted Gram and correction inside the WPS
    // trace, and — worst — the `tested` column set handed to Lawley, which
    // decides WHICH HYPOTHESIS the mean shift is computed for.
    //
    // This is the #1360 defect in a fourth place: the window slides one column
    // per preceding parametric column, folding the intercept and the linear
    // terms into the smooth's block and dropping as many real smooth columns off
    // the end. It is never zero — the intercept alone makes `smooth_start ≥ 1`.
    //
    // It was invisible because three of the four consumers were degraded to
    // index-free fallbacks: `coefficient_influence` was `None` on every model
    // with a conditioned parametric column (fixed alongside this, #2672), so
    // `per_term_edf` fell through to the penalty-block-trace channel — which is
    // indexed by PENALTY block, not by coefficient, and is therefore correct —
    // and `wood_reference_df` returned `None` outright. Restoring `F` is what
    // made the offset observable: on this issue's `y ~ x + s(z)` fixture the
    // per-term EDF of a null smooth jumped from `0.054` (penalty-trace channel,
    // correct) to `2.040` (influence trace over columns `0..9`), which is the
    // unpenalized intercept's `1` plus the parametric `x`'s `1` plus the smooth's
    // own `0.04` — the offset read off the arithmetic.
    let smooth_start = p_total.saturating_sub(full.design.smooth.total_smooth_cols());
    let fitted_likelihood = resolved_likelihood_for_fit(&full.fit)?;
    let family_disp = lawley_dispersion_for_family(&fitted_likelihood, &full.fit)?;
    // The estimated-scale channel (#2672), assembled once because every part of
    // it except the deterministic offset is a property of the FULL fit and not
    // of which term is being tested.
    //
    // The observation count is the POSITIVE-WEIGHT row count, not the raw one.
    // That is the count the optimizer's own `φ̂ = weighted_rss/(n − edf)` uses
    // (#584: a zero-weight row is exactly an absent row, and counting it in the
    // denominator while the numerator excludes it biases `φ̂` low), and it is
    // also the count that multiplies `ln σ̂²` in the log-likelihood, since a
    // zero-weight row's `−½(… − ln w_i …)` term is not summable at all. The two
    // have to be the same number or `W = n·ln(D_0/D_f) + B` is not an identity.
    let profiled_observations = weights.iter().filter(|weight| **weight > 0.0).count();
    let profiled_residual_shares = profiled_scale_residual_shares(
        &fitted_likelihood,
        hessian_inverse.as_ref(),
        &s_lambda,
        p_total,
    )?;
    let full_residual_df = profiled_residual_shares
        .as_ref()
        .and(profiled_residual_degrees_of_freedom(
            &full.fit,
            profiled_observations,
        ));
    // `(v, h)`: the non-trivial residual weights and the degrees of freedom of
    // the weight-one block. On the exact rung that block is the `n − p`
    // directions no column reaches; on the summary rung the whole residual law
    // is folded into it at the fit's own `ν`.
    let profiled_residual = profiled_residual_shares.zip(full_residual_df).map(
        |(shares, residual_df)| match shares {
            Some(spectrum) => (spectrum, profiled_observations.saturating_sub(p_total) as f64),
            None => (Vec::new(), residual_df),
        },
    );

    let mut out = Vec::<SmoothTermLrInference>::new();
    for (term_idx, design_term) in full.design.smooth.terms.iter().enumerate() {
        let penalty_range = full
            .design
            .smooth_term_penalty_range(term_idx)
            .map_err(EstimationError::InvalidInput)?;
        let (block_start, k) = penalty_range
            .map(|range| (range.start, range.len()))
            .unwrap_or((0, 0));
        // Shape-constrained smooths have no calibrated LR reference; they are
        // reported by `smooth_term_lr_unavailable_forspec` instead.
        if gam_solve::estimate::smooth_pvalue_unavailable(&design_term.shape).is_some() {
            continue;
        }
        // Shifted into the GLOBAL coefficient layout — see `smooth_start` above.
        let coeff_range = (smooth_start + design_term.coeff_range.start)
            ..(smooth_start + design_term.coeff_range.end);
        if coeff_range.start >= coeff_range.end || coeff_range.end > p_total {
            continue;
        }
        // Per-term EDF for the χ² reference df FALLBACK (used only when the
        // influence matrix `F` is unavailable). Route through `per_term_edf`,
        // which uses the ADDITIVE per-block trace channel
        // (`|coeff_range| − Σ_{kk∈term} tr_kk`) and caps at the model total,
        // rather than the raw `edf_by_block` block-sum `Σ_{kk}(rank_kk − tr_kk)`.
        // For a multi-penalty term (te/ti/double-penalty) the penalties share one
        // coefficient range, so the rank-based block-sum OVER-COUNTS the term EDF
        // (Σ rank_kk > |coeff_range|) and would inflate the LR reference df,
        // biasing the smooth-term test conservative on large/sparse fits where `F`
        // is not materialised. (Same per-block over-count class as the multinomial
        // `edf_per_class` fix.)
        let edf = full.fit.per_term_edf(coeff_range.clone(), block_start, k);
        // The term's **joint** unpenalized null-space dimension: the coefficient
        // directions penalized by *no* active penalty — the polynomial part a
        // penalized smooth always carries when present, which no penalty can
        // shrink. This is `dim(∩_k null(S_k)) = p_local − rank(Σ_k S_k)`, the
        // INTERSECTION of the per-penalty null spaces, computed by
        // `wald_unpenalized_dim()` — the very same scalar the summary Wald test
        // (`wood_smooth_test`) floors its reference d.f. at, so the LR and Wald
        // tests reference a consistent d.f.
        //
        // It must NOT be `nullspace_dims.iter().sum()`: that *unions* the null
        // spaces (the #1360 defect — see `joint_unpenalized_dim`'s docs). A
        // double-penalty smooth carries a bending penalty (null space = its
        // polynomial part) plus a complementary null-space ridge (which penalizes
        // exactly that polynomial part), so the two null spaces are disjoint and
        // the joint null space is EMPTY (dim 0) — yet the per-penalty dims sum to
        // ~`p_local`. Flooring `ref_df` at that sum pins it to the full basis
        // dimension for every fit (e.g. 11 for a k=12 s(x)), making the LR test
        // badly conservative for genuine moderate signals while only accidentally
        // masking the collapse.
        let null_dim = design_term.wald_unpenalized_dim();
        // The reference the whole-term LR statistic is scored against: the first
        // two moments of its OWN null law, not a chi-square fitted to its mean.
        // See `lr_null_reference` for the derivation and for what this replaced.
        // The term's own penalties restricted to the tested coefficient block,
        // carried λ-FREE with their `ρ̂_i = ln λ̂_i` alongside. The separation is
        // not bookkeeping: the replay's criterion needs `log|Σ_i λ_i S_i|₊`, and
        // that quantity is only computable from the components and their scales
        // — an assembled sum has already lost it (#2644). See
        // [`SelectionGeometry`].
        let mut term_penalties = Vec::<Array2<f64>>::new();
        let mut term_log_lambda = Vec::<f64>::new();
        for (blockwise, &lambda) in full
            .design
            .penalties
            .get(block_start..block_start + k)
            .into_iter()
            .flatten()
            .zip(lambdas[block_start..(block_start + k).min(lambdas.len())].iter())
        {
            let range = &blockwise.col_range;
            if range.start < coeff_range.start || range.end > coeff_range.end {
                continue;
            }
            if !(lambda.is_finite() && lambda > 0.0) {
                continue;
            }
            let mut local = Array2::<f64>::zeros((coeff_range.len(), coeff_range.len()));
            let offset = range.start - coeff_range.start;
            let width = range.end - range.start;
            for row in 0..width {
                for column in 0..width {
                    local[[offset + row, offset + column]] = blockwise.local[[row, column]];
                }
            }
            term_penalties.push(local);
            term_log_lambda.push(lambda.ln());
        }
        // ONE window per scale. The outer search moved each `ρ_i` independently
        // inside its box, so scale `i` could reach `ln t_i ∈ [−B − ρ̂_i, B − ρ̂_i]`
        // and no further. The single COMMON-shift window this used to compute is
        // the intersection of those intervals: correct for a slice that moves
        // every scale together (which is what `generate_common_scale` still
        // derives from these), and wrong for a grid that moves them separately.
        // On a null-true double-penalty smooth at `ρ̂ = (18, −24)` each axis has
        // ~50 of room and the intersection leaves 18; when one `λ̂` rails the
        // intersection is EMPTY and the replay was declined outright.
        // #2812: each term's log-scale window is its own resolvability
        // interval, `[ln(ε γ_min), ln(γ_max / ε)]` over the design-relative
        // penalty spectrum on the tested block, expressed relative to `ρ̂`.
        // Below it the term is unpenalized to working precision, above it the
        // term sits on its null space; the replay has nothing to explore past
        // either edge. A term whose geometry cannot be projected keeps the
        // precision box around unit strength.
        let block_gram = {
            let dense = full.design.design.to_dense();
            let block = dense.slice(ndarray::s![.., coeff_range.start..coeff_range.end]);
            block.t().dot(&block)
        };
        let log_scale_windows: Vec<(f64, f64)> = term_penalties
            .iter()
            .zip(term_log_lambda.iter())
            .map(|(local, &rho)| {
                let interval =
                    gam_solve::estimate::rho_domain::penalty_range_gammas_from_gram(
                        &block_gram,
                        local,
                    )
                    .as_deref()
                    .and_then(gam_solve::estimate::rho_domain::resolvability_interval);
                let (lo, hi) = gam_solve::estimate::rho_domain::coordinate_domain(interval, None);
                (lo - rho, hi - rho)
            })
            .collect();
        let reference = lr_null_reference(
            influence,
            hessian_inverse.as_ref(),
            Some(&s_lambda),
            &coeff_range,
            edf,
            null_dim,
            &log_scale_windows,
            &term_penalties,
            &term_log_lambda,
        );
        let mut reference = reference;
        let ref_df = reference.mean;
        if !(ref_df.is_finite()
            && ref_df > 0.0
            && reference.chi_square_df.is_finite()
            && reference.chi_square_df > 0.0
            && reference.scale.is_finite()
            && reference.scale > 0.0)
        {
            continue;
        }

        // Null model: drop this smooth term from the spec and refit. The term's
        // name pins which spec entry to remove (design and spec share names).
        let mut null_spec = resolvedspec.clone();
        let Some(spec_pos) = null_spec
            .smooth_terms
            .iter()
            .position(|t| t.name == design_term.name)
        else {
            continue;
        };
        null_spec.smooth_terms.remove(spec_pos);
        let null_fit = fit_term_collection_forspec(
            data,
            y,
            weights,
            offset,
            &null_spec,
            family.clone(),
            options,
        );
        let (statistic_lr, eta_null, null_residual_df) = match null_fit {
            Ok(null) if null.fit.log_likelihood.is_finite() => {
                let w = (2.0 * (ll_full - null.fit.log_likelihood)).max(0.0);
                // η at the null fit: X_null β_null + affine_offset + offset
                // (per-row linear predictor; design-layout independent — Lawley
                // reads it on the full design rows). `compose_offset` folds the
                // design's fixed affine channel (non-zero endpoint anchor,
                // #2297) into the user offset.
                let null_offset = null
                    .design
                    .compose_offset(offset, "smooth likelihood-ratio null model")
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                let mut eta = null.design.design.dot(&null.fit.beta);
                eta += &null_offset;
                let residual_df =
                    profiled_residual_degrees_of_freedom(&null.fit, profiled_observations);
                (w, Some(eta), residual_df)
            }
            _ => (f64::NAN, None, None),
        };

        // The estimated-scale channel needs BOTH fits' residual degrees of
        // freedom, so it is completed here rather than where the rest of the
        // reference was built: `B = n·ln(ν_f/ν_0) + (ν_0 − ν_f)` is the only
        // part of `W` that is a function of the null refit and of nothing
        // random (#2672).
        if let Some((residual_weights, residual_unit_dimension)) = profiled_residual.as_ref()
            && let (Some(full_df), Some(null_df)) = (full_residual_df, null_residual_df)
            && full_df > 0.0
            && null_df > 0.0
        {
            let observations = profiled_observations as f64;
            reference.profiled_scale = Some(SmoothLrProfiledScale {
                observations,
                deterministic_offset: observations * (full_df / null_df).ln()
                    + (null_df - full_df),
                residual_weights: residual_weights.clone(),
                residual_unit_dimension: *residual_unit_dimension,
            });
        }
        let ref_df_provenance = reference.clone();

        // The uncorrected tail is read only to judge how far the correction
        // moved the answer (`material`); it is not published as a p-value.
        let (p_uncorrected, _) = reference.tail_probability_with_bound(statistic_lr);

        // Magic Bartlett correction: only when the LR statistic is finite, the
        // family has closed-form jets, n is in the resolvable regime, and the
        // factor is computable. Otherwise the uncorrected χ² stands.
        let mut bartlett_factor = 1.0;
        let mut bartlett_factor_conditional = None;
        let mut rho_variation_shift = None;
        let mut statistic_corrected = statistic_lr;
        let mut p_corrected = p_uncorrected;
        let mut correction = SmoothLrCorrection::None;
        if let (Some(eta), true, true) = (
            eta_null.as_ref(),
            statistic_lr.is_finite(),
            n <= LAWLEY_PAIR_MATRIX_MAX_ROWS,
        ) {
            let kappas: Option<Vec<_>> = (0..n)
                .map(|i| {
                    known_scale_expected_jets_with_dispersion(
                        &fitted_likelihood.spec,
                        eta[i],
                        family_disp,
                    )
                    .and_then(|jets| jets.kappas().ok())
                })
                .collect();
            if let Some(kappas) = kappas {
                let fixed_factor = lawley_lr_bartlett_factor(
                    full_design_dense.view(),
                    &kappas,
                    Some(s_lambda.view()),
                    coeff_range.clone(),
                    ref_df,
                );
                if let Ok(c_cond) = fixed_factor
                    && c_cond.is_finite()
                    && c_cond > 0.0
                {
                    let mut c_applied = c_cond;
                    correction = SmoothLrCorrection::LawleyLrFixedLambda;
                    if let Some(cov) = rho_covariance
                        && let Ok(total_shift) = lawley_lr_mean_shift_with_rho_variation(
                            full_design_dense.view(),
                            &kappas,
                            s_lambda.view(),
                            coeff_range.clone(),
                            &rho_penalty_components,
                            cov.view(),
                        )
                    {
                        let mean_w = ref_df + total_shift;
                        if let Some(c_est) =
                            gam_terms::inference::higher_order::bartlett_factor_from_mean(
                                mean_w, ref_df,
                            )
                            && c_est.is_finite()
                            && c_est > 0.0
                        {
                            let conditional_shift = (c_cond - 1.0) * ref_df;
                            c_applied = c_est;
                            bartlett_factor_conditional = Some(c_cond);
                            rho_variation_shift = Some(total_shift - conditional_shift);
                            correction = SmoothLrCorrection::LawleyLrEstimatedLambda;
                        }
                    }
                    bartlett_factor = c_applied;
                    statistic_corrected = statistic_lr / c_applied;
                    // `W* = W/c` and "rescale every spectral weight by `c`" are
                    // the same operation on this reference — the law is exactly
                    // scale-equivariant — so the correction composes with the
                    // scaled reference without a second convention.
                    p_corrected = reference.tail_probability_with_bound(statistic_corrected).0;
                }
            }
        }

        // Materiality (#939 deliverable 4): only when a correction was actually
        // applied, flagged when it moves the result by more than the 10%
        // threshold — by the Bartlett factor's distance from one OR the relative
        // p-value shift, whichever is larger (a factor near one can still flip a
        // p-value sitting on the α boundary, and vice versa).
        let material = match correction {
            SmoothLrCorrection::LawleyLrEstimatedLambda
            | SmoothLrCorrection::LawleyLrFixedLambda => {
                let factor_move = (bartlett_factor - 1.0).abs();
                let p_denom = p_uncorrected.max(p_corrected);
                // Two zero p-values have not moved.
                let p_move = if p_uncorrected.is_finite() && p_corrected.is_finite() && p_denom > 0.0
                {
                    (p_corrected - p_uncorrected).abs() / p_denom
                } else {
                    0.0
                };
                factor_move > SMOOTH_LR_MATERIAL_THRESHOLD || p_move > SMOOTH_LR_MATERIAL_THRESHOLD
            }
            SmoothLrCorrection::None => false,
        };

        out.push(SmoothTermLrInference {
            name: design_term.name.clone(),
            term_idx,
            statistic_lr,
            ref_df,
            ref_df_provenance,
            bartlett_factor,
            bartlett_factor_conditional,
            rho_variation_shift,
            statistic_corrected,
            p_value: p_corrected,
            material,
            correction,
        });
    }
    Ok(out)
}

/// The residual degrees of freedom the fit's profiled `σ̂` ACTUALLY divided by,
/// read back as `D/σ̂²` from two published fields.
///
/// Not recomputed as `n − edf_total`. That is what the optimizer uses on one of
/// its two branches — the other, taken when inference is off, divides by `n` —
/// and a reference that assumed the branch would be silently wrong on the other
/// one. `D` is the weighted residual sum of squares (the Gaussian deviance) and
/// `σ̂² = D/ν` by construction, so `ν = D/σ̂²` inverts the fit's own convention
/// whatever it was.
///
/// The inversion is exact rather than approximate, and the reason is a
/// type-level one: `ProfiledGaussian` implies the IDENTITY link, because
/// `gam_spec`'s `legal_cell_kind` admits no other Gaussian cell — a
/// `(Gaussian, log)` model is not constructible rather than merely unusual. The
/// optimizer's weighted-RSS channel is wired for the identity link, so on every
/// fit that reaches this function `D` IS `Σ w_i(y_i − μ_i)²` and `σ̂² = D/ν`
/// holds by construction.
///
/// `None` when the two fields cannot produce a residual degrees of freedom that
/// is finite, positive, and no larger than the sample size — a statement that
/// this fit's `σ̂` is not the profiled residual one the identity above assumes,
/// which is exactly when the estimated-scale channel must stay switched off.
fn profiled_residual_degrees_of_freedom(fit: &UnifiedFitResult, n: usize) -> Option<f64> {
    let variance = fit.standard_deviation * fit.standard_deviation;
    if !(variance.is_finite() && variance > 0.0 && fit.deviance.is_finite() && fit.deviance > 0.0) {
        return None;
    }
    let residual_df = fit.deviance / variance;
    // The `n` ceiling carries a tolerance because `ν = D/σ̂²` is a ratio of two
    // separately-rounded published numbers, and the branch that divides by `n`
    // exactly lands on the ceiling.
    let ceiling = n as f64 * (1.0 + 8.0 * f64::EPSILON);
    (residual_df.is_finite() && residual_df > 0.0 && residual_df <= ceiling).then_some(residual_df)
}

/// The residual quadratic form's spectrum, `v_i = p_i²` over the WHOLE model's
/// penalty shares — or `None` when the family does not profile a Gaussian scale
/// out of a residual sum of squares.
///
/// This is `lr_tested_block` over the full coefficient range rather than over
/// a term's: the same self-adjoint decomposition and the same `[0, 1]` shares,
/// deliberately not a second route to the same object. See
/// [`SmoothLrProfiledScale`] for why `V ~ Σ_i p_i²·χ²_1 + χ²_{n−p}`.
///
/// `Some(None)` — the family profiles a scale but the shares are unreachable —
/// is a rung, not a refusal, on the same ladder the numerator's reference
/// already has: the fit that cannot publish `H⁻¹` is the fit whose numerator is
/// also a two-moment summary. What the caller does with it is fold the whole
/// residual law into `V ~ χ²_{ν}` at the fit's own `ν`, which is mgcv's `F`
/// reference exactly. Two things are then inexact rather than exact, both
/// bounded by `Σ_i f_i(1 − f_i)` — the count of PARTIALLY shrunk directions,
/// zero at both ends of the shrinkage range: `E[V] = tr((I−A)²) = ν −
/// Σ f(1−f)` rather than `ν`, and `Var(V)/2 = Σv² ≤ Σv` so `χ²_ν`
/// over-disperses. That is strictly better than the known-scale reference,
/// which is what the alternative — dropping the channel — would silently
/// restore.
fn profiled_scale_residual_shares(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    hessian_inverse: Option<&Array2<f64>>,
    penalty: &Array2<f64>,
    p_total: usize,
) -> Result<Option<Option<Vec<f64>>>, EstimationError> {
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if !matches!(resolved, gam_spec::ResolvedLikelihoodScale::ProfiledGaussian) {
        return Ok(None);
    }
    Ok(Some(
        lr_tested_block(hessian_inverse, Some(penalty), &(0..p_total))
            .map(|block| block.shares.iter().map(|share| share * share).collect()),
    ))
}

fn resolved_likelihood_for_fit(
    fit: &UnifiedFitResult,
) -> Result<gam_spec::GlmLikelihoodSpec, EstimationError> {
    let spec = fit.likelihood_family.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth-term LR inference requires an engine-level GLM likelihood".to_string(),
        )
    })?;
    gam_spec::GlmLikelihoodSpec::try_new(spec.clone(), fit.likelihood_scale.clone())
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))
}

/// The response dispersion `phi` Lawley needs for cumulant scaling. This is
/// deliberately distinct from the coefficient-covariance multiplier used by
/// the WPS trace below: Gamma Lawley uses `1 / shape`, while its PIRLS Hessian
/// already carries `shape` and therefore has covariance multiplier one.
fn lawley_dispersion_for_family(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    fit: &UnifiedFitResult,
) -> Result<f64, EstimationError> {
    let profiled_standard_deviation = matches!(
        likelihood
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        gam_spec::ResolvedLikelihoodScale::ProfiledGaussian
    )
    .then_some(fit.standard_deviation);
    gam_solve::estimate::dispersion_from_likelihood(likelihood, profiled_standard_deviation)
        .map(|dispersion| dispersion.phi())
}

/// The reference distribution for the whole-term LR statistic: its own null
/// spectrum `w` when that is recoverable, and the two-moment summary of it when
/// only the moments are.
///
/// The derivation, and why the spectrum rather than two of its moments, is on
/// [`SmoothLrReferenceDf`]. What is worth stating at the code is the ladder, and
/// that each rung is a strictly weaker instrument on the SAME quantity rather
/// than a different claim:
///
/// 1. **The spectrum** (`lr_tested_block`) — needs `[H⁻¹]_jj`
///    and the term's λ-weighted penalty block. Exact.
/// 2. **Its first two moments** ([`lr_null_spectral_moments`]) — needs only the
///    coefficient-influence block, because with `A = 2F − F²`
///
///    ```text
///    Σ w   = tr A  = 2·tr F − tr F²
///    Σ w²  = tr A² = 4·tr F² − 4·tr F³ + tr F⁴
///    ```
///
///    are traces of powers of one `q × q` block. Reading the weights THEMSELVES
///    off `F_jj` is what rung 1 avoids: `F_jj = H̃⁻¹Ĩ_jj` is not symmetric, so it
///    would need a general eigensolver, while rung 1 reaches the same spectrum
///    through a self-adjoint one.
/// 3. **A scalar EDF** — `χ²_{max(edf, null_dim, 1)}`, the unit-weight shape.
///
/// The lane taken is tagged in the returned provenance, so a consumer can tell
/// an exact reference from a summary of one instead of inferring it from the
/// numbers.
///
/// `term_penalties` are the term's λ-FREE penalty components on the tested
/// block and `term_log_lambda` their fitted `ρ̂_i`, carried separately rather
/// than pre-multiplied. That is not bookkeeping: the selection replay's
/// criterion needs `log|Σ_i λ_i S_i|₊`, and an assembled sum has already lost
/// it whenever the `λ_i` separate (#2644 — see [`SelectionGeometry`]).
/// `log_scale_windows` carries ONE window per component, because the outer
/// search moved each `ρ_i` independently inside its own box.
fn lr_null_reference(
    influence: Option<&Array2<f64>>,
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
    edf: f64,
    null_dim: usize,
    log_scale_windows: &[(f64, f64)],
    term_penalties: &[Array2<f64>],
    term_log_lambda: &[f64],
) -> SmoothLrReferenceDf {
    let from_moments = |mean: f64, second_moment: f64, source| SmoothLrReferenceDf {
        weights: Vec::new(),
        mean,
        second_moment,
        chi_square_df: mean * mean / second_moment,
        scale: second_moment / mean,
        moment_residual: None,
        edf,
        null_dim,
        source,
        // The degraded lanes do not have the spectrum, so they cannot have the
        // geometry the replay is built from either. That is only harmless when
        // nothing was selected; a penalized term that lands here had its `λ̂`
        // chosen and no way to price the choice, which is a refusal.
        selection: SmoothLrSelection::Declined(if term_penalties.is_empty() {
            SmoothLrSelectionDecline::NoPenaltyComponents
        } else {
            SmoothLrSelectionDecline::GeometryRefused
        }),
        // Completed by the caller once the null refit has produced the second
        // residual degrees of freedom `B` needs (#2672).
        profiled_scale: None,
    };
    let unit_weight = || {
        let df = edf.max(null_dim as f64).max(1.0);
        from_moments(df, df, SmoothLrReferenceSource::UnitWeightFallback)
    };
    let influence_moments = lr_null_spectral_moments(influence, coeff_range);

    // Rung 1 — the spectrum itself.
    if let Some(block) = lr_tested_block(hessian_inverse, penalty, coeff_range) {
        let mut weights: Vec<f64> = block.shares.iter().map(|&p| 1.0 - p * p).collect();
        weights.sort_by(|a, b| b.partial_cmp(a).expect("finite weights"));
        let mean: f64 = weights.iter().sum();
        let second_moment: f64 = weights.iter().map(|w| w * w).sum();
        if mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0 {
            // The identity check, measured rather than assumed. Denominated
            // relatively and floored at one so a term shrunk to nothing does not
            // report a huge residual for a difference of `1e-16`.
            let moment_residual = influence_moments.map(|[trace_mean, trace_second]| {
                let first = (mean - trace_mean).abs() / mean.abs().max(1.0);
                let second = (second_moment - trace_second).abs() / second_moment.abs().max(1.0);
                first.max(second)
            });
            return SmoothLrReferenceDf {
                weights,
                mean,
                second_moment,
                chi_square_df: mean * mean / second_moment,
                scale: second_moment / mean,
                moment_residual,
                edf,
                null_dim,
                source: SmoothLrReferenceSource::NullSpectrum,
                // One entry point for both lanes. `generate` whitens the term's
                // λ-free components by the Schur-complemented information,
                // factors them into roots, and dispatches on how many scales the
                // term actually selects. The generalized spectrum it reports is
                // read off that geometry rather than reconstructed as
                // `p_k/(1 − p_k)` from the penalty shares — a share is a number
                // in `[0, 1]`, so a structural zero and a `1e-17` of roundoff are
                // one machine epsilon apart there, and the criterion's
                // log-determinant is the one place that difference is worth
                // `log(1 + 1e17)`.
                selection: SmoothLrSelectionReplay::generate(
                    &block.whitener,
                    term_penalties,
                    term_log_lambda,
                    log_scale_windows,
                ),
                profiled_scale: None,
            };
        }
    }

    // Rung 2 — two moments of it, off the influence block.
    let Some([mean, second_moment]) = influence_moments else {
        return unit_weight();
    };
    if !(mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0) {
        return unit_weight();
    }
    from_moments(
        mean,
        second_moment,
        SmoothLrReferenceSource::SpectralMomentMatch,
    )
}

/// `[tr A, tr A²]` for `A = 2·F_jj − F_jj²` on the tested coefficient block.
///
/// Returns `None` when the influence matrix is absent, the block is outside it,
/// or either trace is non-finite — the caller then falls back to the unit-weight
/// shape rather than scoring against a spectrum it could not compute.
fn lr_null_spectral_moments(
    influence: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
) -> Option<[f64; 2]> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]).to_owned();
    let squared = block.dot(&block);
    let cubed = squared.dot(&block);
    let quartic = squared.dot(&squared);
    let trace = |m: &Array2<f64>| (0..m.nrows()).map(|i| m[[i, i]]).sum::<f64>();
    let (t1, t2, t3, t4) = (
        trace(&block),
        trace(&squared),
        trace(&cubed),
        trace(&quartic),
    );
    let mean = 2.0 * t1 - t2;
    let second_moment = 4.0 * t2 - 4.0 * t3 + t4;
    (mean.is_finite() && second_moment.is_finite()).then_some([mean, second_moment])
}

#[cfg(test)]
mod lr_null_reference_tests {
    use super::{
        SmoothLrReferenceSource, lr_null_reference, lr_null_spectral_moments,
        lr_tested_block,
    };
    use ndarray::Array2;

    /// No selection window: these unit tests are about the CONDITIONAL law, so
    /// they hold `λ` fixed and the replay is inert. The replay's own behaviour
    /// is pinned separately.
    const WINDOW: &[(f64, f64)] = &[];

    /// `M⁻¹` for a symmetric PD `M`, through the same self-adjoint entry point
    /// the production path uses. The tests need an inverse only to BUILD the two
    /// inputs (`H⁻¹` and `F = H⁻¹(H − S)`) from one `H`; nothing under test reads
    /// it.
    fn symmetric_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let (values, vectors) =
            gam_linalg::faer_ndarray::strict_symmetric_eigh(matrix, faer::Side::Lower)
                .expect("symmetric PD inverse");
        let mut scaled = vectors.clone();
        for (mut column, &value) in scaled.columns_mut().into_iter().zip(values.iter()) {
            column.mapv_inplace(|entry| entry / value);
        }
        scaled.dot(&vectors.t())
    }

    /// A diagonal influence block has `F_jj` eigenvalues on the diagonal, so the
    /// spectrum is `2f − f²` term by term and both moments are hand-computable.
    /// This is the identity the whole reference rests on; it is checked against
    /// the definition rather than against another implementation of itself.
    #[test]
    fn the_spectral_moments_are_the_weights_of_the_null_law() {
        let f_diag = [0.9_f64, 0.5, 0.2, 0.05];
        let mut influence = Array2::<f64>::zeros((6, 6));
        // Deliberately offset: the block is columns 2..6, and rows/columns
        // outside it carry values that must not leak into either trace.
        influence[[0, 0]] = 7.0;
        influence[[1, 1]] = -3.0;
        influence[[0, 3]] = 11.0;
        influence[[5, 1]] = -2.0;
        for (i, &f) in f_diag.iter().enumerate() {
            influence[[2 + i, 2 + i]] = f;
        }
        let [mean, second] =
            lr_null_spectral_moments(Some(&influence), &(2..6)).expect("moments available");
        let weights: Vec<f64> = f_diag.iter().map(|f| 2.0 * f - f * f).collect();
        let want_mean: f64 = weights.iter().sum();
        let want_second: f64 = weights.iter().map(|w| w * w).sum();
        assert!(
            (mean - want_mean).abs() < 1e-12 && (second - want_second).abs() < 1e-12,
            "moments ({mean}, {second}) vs weights {weights:?} -> ({want_mean}, {want_second})"
        );
    }

    /// THE IDENTITY THE EXACT LANE RESTS ON, on a design where every block is
    /// coupled to every other: the spectrum read off `[H⁻¹]_jj` and `S_jj` has
    /// the same two moments as the spectrum read off the influence block, which
    /// are computed by completely different arithmetic (two self-adjoint
    /// decompositions versus four traces of powers of a non-symmetric matrix).
    ///
    /// `(I − F)_jj = [H⁻¹]_jj S_jj` is only true because the penalty is
    /// block-diagonal by term, so the fixture puts a SEPARATE penalty on the
    /// retained block as well: the identity must survive other terms being
    /// penalized (that is the difference between the Schur complement of the
    /// penalized retained block and of the unpenalized one), and it must fail if
    /// anyone ever lets a penalty couple two terms.
    #[test]
    fn the_penalty_spectrum_and_the_influence_moments_are_the_same_object() {
        let (retained, tested) = (3usize, 5usize);
        let p = retained + tested;
        // A dense SPD Gram with real cross-block coupling.
        let mut gram = Array2::<f64>::zeros((p, p));
        for row in 0..p {
            for col in 0..p {
                gram[[row, col]] = 1.0 / (1.0 + (row as f64 - col as f64).abs())
                    + if row == col { 0.75 } else { 0.0 };
            }
        }
        for lambda in [0.0_f64, 1e-3, 1.0, 25.0, 1e4, 1e7] {
            // Block-diagonal penalty: a second-difference block on the tested
            // term and an unrelated ridge on the retained one.
            let mut penalty = Array2::<f64>::zeros((p, p));
            for row in 0..retained {
                penalty[[row, row]] = 0.3;
            }
            for row in 0..tested.saturating_sub(2) {
                for (offset_a, coefficient_a) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                    for (offset_b, coefficient_b) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                        penalty[[retained + row + offset_a, retained + row + offset_b]] +=
                            lambda * coefficient_a * coefficient_b;
                    }
                }
            }
            let hessian = &gram + &penalty;
            let hessian_inverse = symmetric_inverse(&hessian);
            let influence = hessian_inverse.dot(&gram);

            let weights =
                lr_tested_block(Some(&hessian_inverse), Some(&penalty), &(retained..p))
                    .map(|block| block.shares)
                    .map(|shares| shares.iter().map(|&q| 1.0 - q * q).collect::<Vec<f64>>())
                    .expect("spectrum available");
            let [mean, second] = lr_null_spectral_moments(Some(&influence), &(retained..p))
                .expect("moments available");
            let spectrum_mean: f64 = weights.iter().sum();
            let spectrum_second: f64 = weights.iter().map(|w| w * w).sum();
            // `1e-7` relative, not roundoff: at `λ = 1e7` the INFLUENCE route
            // is what loses the digits — `2·trF − trF²` differences two nearly
            // equal quantities while `trF → 0` — and it comes in at `2.5e-9`
            // relative there against `<1e-15` at every smaller `λ`. That
            // asymmetry is one of the reasons the penalty route is the primary
            // one; the bar is set where the WEAKER of the two routes lives.
            assert!(
                (spectrum_mean - mean).abs() < 1e-7 * mean.abs().max(1.0)
                    && (spectrum_second - second).abs() < 1e-7 * second.abs().max(1.0),
                "lambda={lambda}: spectrum moments ({spectrum_mean}, {spectrum_second}) \
                 disagree with influence-trace moments ({mean}, {second})"
            );
            assert!(
                weights.iter().all(|w| (0.0..=1.0).contains(w)),
                "lambda={lambda}: weights escaped [0,1]: {weights:?}"
            );
            assert!(
                weights.windows(2).all(|pair| pair[0] >= pair[1]),
                "lambda={lambda}: weights are not sorted descending: {weights:?}"
            );
        }
    }

    /// Each rung of the ladder degrades to the next and SAYS so. A consumer that
    /// cannot tell an exact reference from a summary of one, or a summary from a
    /// scalar-EDF fallback, cannot reason about the number it was handed.
    #[test]
    fn each_missing_input_degrades_exactly_one_rung_and_visibly() {
        let q = 4;
        let influence = Array2::<f64>::eye(q) * 0.5;
        let hessian_inverse = Array2::<f64>::eye(q);
        let penalty = Array2::<f64>::eye(q) * 0.5;

        // Everything present: the exact lane, carrying weights.
        let exact = lr_null_reference(
            Some(&influence),
            Some(&hessian_inverse),
            Some(&penalty),
            &(0..q),
            2.0,
            1,
            WINDOW,
            &[],
            &[],
        );
        assert_eq!(exact.source, SmoothLrReferenceSource::NullSpectrum);
        assert_eq!(exact.weights.len(), q);

        // No `H⁻¹` (or no penalty): the moments off `F`, and NO weights — which
        // is exactly the condition `tail_probability_with_bound` switches on.
        for degraded in [
            lr_null_reference(Some(&influence), None, Some(&penalty), &(0..q), 2.0, 1, WINDOW, &[], &[]),
            lr_null_reference(
                Some(&influence),
                Some(&hessian_inverse),
                None,
                &(0..q),
                2.0,
                1,
                WINDOW,
                &[],
                &[],
            ),
        ] {
            assert_eq!(degraded.source, SmoothLrReferenceSource::SpectralMomentMatch);
            assert!(degraded.weights.is_empty());
            assert!((degraded.mean - exact.mean).abs() < 1e-12);
        }

        // Nothing at all: the unit-weight shape with its `max(edf, null_dim, 1)`.
        let fallback = lr_null_reference(None, None, None, &(0..q), 2.5, 1, WINDOW, &[], &[]);
        assert_eq!(fallback.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert!(fallback.weights.is_empty());
        assert_eq!(fallback.chi_square_df, 2.5);
        assert_eq!(fallback.scale, 1.0);
        // The `max(edf, null_dim, 1)` shape is retained only on this lane.
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 3, WINDOW, &[], &[]).chi_square_df,
            3.0
        );
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 0, WINDOW, &[], &[]).chi_square_df,
            1.0
        );
    }
}

/// The ESTIMATED-SCALE channel (#2672): the reference a profiled Gaussian's
/// `W = n·ln(1 + Q/V) + B` actually has.
#[cfg(test)]
mod profiled_scale_reference_tests {
    use super::{
        SmoothLrProfiledScale, SmoothLrReferenceDf, SmoothLrReferenceSource, SmoothLrSelection,
        SmoothLrSelectionDecline,
    };

    /// A reference carrying an explicit spectrum, no selection replay, and the
    /// strictest tail accuracy the clamp allows.
    fn reference(weights: Vec<f64>, profiled_scale: Option<SmoothLrProfiledScale>) -> SmoothLrReferenceDf {
        let mean: f64 = weights.iter().sum();
        let second_moment: f64 = weights.iter().map(|w| w * w).sum();
        SmoothLrReferenceDf {
            weights,
            mean,
            second_moment,
            chi_square_df: mean * mean / second_moment,
            scale: second_moment / mean,
            moment_residual: None,
            edf: mean,
            null_dim: 0,
            source: SmoothLrReferenceSource::NullSpectrum,
            selection: SmoothLrSelection::Declined(SmoothLrSelectionDecline::NoPenaltyComponents),
            profiled_scale,
        }
    }

    /// When the tested block's spectrum is FLAT the ratio law is a Fisher-
    /// Snedecor tail in closed form, so the whole channel — the `expm1`
    /// inversion, the residual spectrum, the multiplicity fold — is checkable
    /// against `fisher_snedecor_sf` with nothing shared but the arithmetic.
    ///
    /// `Q = g·χ²_q` and `V = χ²_ν`, so
    /// `P(W > w) = P(g·χ²_q > c·χ²_ν) = P(F_{q,ν} > c·ν/(g·q))`.
    #[test]
    fn a_flat_spectrum_makes_the_profiled_reference_an_f_tail() {
        for &(q, scale) in &[(1usize, 1.0_f64), (4, 1.0), (4, 0.37), (9, 2.5)] {
            for &nu in &[8.0_f64, 26.0, 191.0] {
                let observations = nu + q as f64;
                let subject = reference(
                    vec![scale; q],
                    Some(SmoothLrProfiledScale {
                        observations,
                        deterministic_offset: 0.0,
                        // `V ~ χ²_ν` exactly: no partially-shrunk direction, so
                        // the whole residual law is the unit-multiplicity term.
                        residual_weights: Vec::new(),
                        residual_unit_dimension: nu,
                    }),
                );
                for &statistic in &[0.05_f64, 0.8, 3.0, 12.0, 30.0] {
                    let (got, bound) = subject.tail_probability_with_bound(statistic);
                    let ratio = (statistic / observations).exp_m1();
                    let want = gam_math::probability::fisher_snedecor_sf(
                        ratio * nu / (scale * q as f64),
                        q as f64,
                        nu,
                    );
                    assert!(
                        (got - want).abs() <= bound + 1e-9,
                        "q={q} g={scale} ν={nu} W={statistic}: {got} vs F-tail {want} \
                         (certified {bound:.3e})"
                    );
                }
            }
        }
    }

    /// The deterministic offset is part of the statistic, not a nuisance: a `W`
    /// at or below it corresponds to `Q/V ≤ 0`, which cannot happen, so the
    /// p-value is exactly one rather than something the quadrature invents.
    #[test]
    fn a_statistic_under_the_deterministic_offset_is_not_evidence() {
        let subject = reference(
            vec![1.0_f64, 0.5],
            Some(SmoothLrProfiledScale {
                observations: 30.0,
                deterministic_offset: -0.61,
                residual_weights: vec![0.25],
                residual_unit_dimension: 24.0,
            }),
        );
        for &statistic in &[-5.0_f64, -0.61, -0.7] {
            let (got, bound) = subject.tail_probability_with_bound(statistic);
            assert_eq!(got, 1.0, "W={statistic}");
            assert_eq!(bound, 0.0);
        }
        // And it is a strict boundary: just above the offset the tail is still
        // essentially one, but it is no longer the exact branch.
        let (just_above, _) = subject.tail_probability_with_bound(-0.6099);
        assert!(just_above < 1.0 && just_above > 0.999, "{just_above}");
    }

    /// A `λ̂` that WAS chosen but whose selection replay refused has no p-value.
    ///
    /// The tail used to fall through to the conditional (fixed-`λ`) law on ANY
    /// decline. On a Bernoulli null term the penalty had absorbed, the
    /// multiscale replay refused (`selection_unresolved`) and that fall-through
    /// published `p = 0.0005`. A decline that says nothing was selectable keeps
    /// the conditional tail, which is then the whole law.
    #[test]
    fn a_refused_selection_replay_publishes_no_p_value() {
        let statistic = 3.0;
        let declines = [
            SmoothLrSelectionDecline::NoPenaltyComponents,
            SmoothLrSelectionDecline::NoInformation,
            SmoothLrSelectionDecline::GeometryRefused,
            SmoothLrSelectionDecline::WindowClosed,
            SmoothLrSelectionDecline::GridRefused,
            SmoothLrSelectionDecline::SelectionUnresolved,
        ];
        for decline in declines {
            let mut subject = reference(vec![1.0_f64, 0.6, 0.2], None);
            subject.selection = SmoothLrSelection::Declined(decline);
            let (conditional, _) = subject.conditional_tail_with_bound(statistic);
            assert!(conditional.is_finite() && conditional > 0.0 && conditional < 1.0);
            let (published, bound) = subject.tail_probability_with_bound(statistic);
            if decline.is_refusal() {
                assert!(
                    published.is_nan() && bound.is_nan(),
                    "{}: a refused replay must publish NaN, got p={published}",
                    decline.label()
                );
            } else {
                assert_eq!(
                    published,
                    conditional,
                    "{}: with nothing selectable the conditional law is the law",
                    decline.label()
                );
            }
        }
        assert!(SmoothLrSelectionDecline::SelectionUnresolved.is_refusal());
        assert!(!SmoothLrSelectionDecline::WindowClosed.is_refusal());
    }
}

#[cfg(test)]
mod lr_null_spectrum_moment_tests {
    use super::*;

    /// No penalty components and therefore no selection window: these tests are
    /// about the CONDITIONAL law, so the replay is inert.
    const WINDOW: &[(f64, f64)] = &[];

    // The whole-term LR reference (#1766, #1872, #2672). The first spectral
    // moment `tr(2F − F²)` IS Wood's `edf1`; what changed under #2672 is that it
    // is now the MEAN of a reference whose SHAPE comes from the second moment,
    // instead of being handed to a chi-square as a degrees of freedom.

    #[test]
    fn the_first_moment_is_wood_edf1() {
        // A symmetric smoother block with eigenvalues {0.9, 0.4}: a partially
        // shrunk penalized term. edf = tr = 1.3; tr(F²) = 0.81 + 0.16 = 0.97;
        // edf1 = 2·1.3 − 0.97 = 1.63. (Diagonal ⇒ block F² trace = Σ λ².)
        let f = ndarray::array![[0.9_f64, 0.0], [0.0, 0.4]];
        let [mean, second] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(
            (mean - 1.63).abs() < 1e-12,
            "the first spectral moment is Wood's edf1 = 2*tr - tr(F^2) = 1.63, got {mean}"
        );
        // And it dominates the raw edf, analytically: w_j = 2f_j − f_j² ≥ f_j on
        // [0, 1], so no `.max(edf)` guard is needed to make it hold.
        assert!(mean >= 1.3 - 1e-12, "edf1 {mean} must be >= edf 1.3");
        // Second moment: w = {0.99, 0.64} ⇒ Σw² = 0.9801 + 0.4096.
        assert!((second - 1.3897).abs() < 1e-12, "second moment {second}");
    }

    #[test]
    fn a_corrupted_block_degrades_to_the_fallback_rather_than_being_floored() {
        // A real influence block has eigenvalues in [0, 1], so `tr(F²)` cannot
        // run away. Numerical corruption can still produce one that does, and
        // the pre-#2672 code floored `edf1` back at `tr` — silently returning a
        // reference derived from a block it had just decided was unusable. The
        // spectral reference does not paper over it: the first moment goes
        // negative and the assembly degrades to the unit-weight lane, VISIBLY.
        let f = ndarray::array![[0.5_f64, 40.0], [40.0, 0.5]];
        let [mean, _] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(mean < 0.0, "the corrupted block's first moment is {mean}");
        let reference = lr_null_reference(Some(&f), None, None, &(0..2), 1.0, 1, WINDOW, &[], &[]);
        assert_eq!(reference.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert_eq!(reference.chi_square_df, 1.0);
        assert_eq!(reference.scale, 1.0);
    }

    #[test]
    fn returns_none_on_a_missing_or_out_of_range_block() {
        // No influence matrix at all → None (caller falls back to the
        // unit-weight `max(edf, null_dim, 1)` shape).
        assert!(lr_null_spectral_moments(None, &(0..2)).is_none());
        // An out-of-bounds range → None, never a panic.
        let f = ndarray::array![[0.5_f64, 0.0], [0.0, 0.5]];
        assert!(lr_null_spectral_moments(Some(&f), &(0..5)).is_none());
        // A fully-shrunk block has ZERO moments, which is not a usable
        // reference either — the caller must see the fallback, not a divide by
        // zero.
        let zero = ndarray::array![[0.0_f64, 0.0], [0.0, 0.0]];
        assert_eq!(
            lr_null_spectral_moments(Some(&zero), &(0..2)).unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(
            lr_null_reference(Some(&zero), None, None, &(0..2), 0.0, 0, WINDOW, &[], &[]).source,
            SmoothLrReferenceSource::UnitWeightFallback
        );
    }
}
