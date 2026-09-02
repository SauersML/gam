//! Self-certifying numerical differentiation.
//!
//! A finite difference is an estimator, not a fact. Its error is
//! `ν/h + h²·f‴/6`, and neither `ν` (the evaluator's absolute noise floor) nor
//! `f‴` is known a priori — so a fixed-step difference returns a number whose
//! accuracy is unknown, and any disagreement between it and an analytic
//! derivative is unattributable: it may be the analytic side, or it may be the
//! difference. gam#2461 is one instance measured end to end: a criterion whose
//! ψ-profile had characteristic scale `1.66e-3` was differenced at `h = 3e-4`,
//! and the resulting `(h/s)²/6 = 5.4e-3` sinc defect was filed as an analytic
//! gradient error, stable to four digits because the fixture saturated.
//!
//! This module differentiates by Ridders' method — central differences on a
//! shrinking geometric ladder, Neville-extrapolated — and returns the
//! extrapolant TOGETHER with an estimate of its own error, plus a three-way
//! verdict (`Agree` / `Disagree` / `Unresolved`) so a caller can decline to
//! judge a component its oracle cannot measure.
//!
//! It lives in the production tree, not under `test_support`, because the
//! outer-gradient FD audit that the κ/geometry gates consume is production
//! code: it runs inside the outer optimizer and differences the real
//! criterion. One implementation serves both it and the test-level checkers.

/// Configuration for the self-certifying [`ridders_derivative`] oracle.
///
/// The defaults are a geometric ladder `h₀ · r⁻ⁱ` with `h₀ = 1e-2`, `r = 2` and
/// 12 rungs, i.e. steps from `1e-2` down to `4.9e-6`. That span is what makes
/// the oracle scale-free: it brackets both a slowly-varying objective (whose
/// truncation only becomes negligible at the coarse end, where evaluator noise
/// is smallest) and a sharply-varying one (which needs the fine end before the
/// `O(h²)` law even starts to hold).
#[derive(Clone, Copy, Debug)]
pub struct RiddersConfig {
    /// Largest step in the ladder. Must be strictly positive.
    pub initial_step: f64,
    /// Ladder ratio: step `i` is `initial_step / shrink^i`. Must exceed 1.
    pub shrink: f64,
    /// Number of ladder rungs; the oracle costs `2 · rungs` evaluations.
    pub rungs: usize,
}

impl Default for RiddersConfig {
    fn default() -> Self {
        Self {
            initial_step: 1.0e-2,
            shrink: 2.0,
            rungs: 12,
        }
    }
}

/// A directional derivative measured together with a bound on the
/// measurement's OWN error.
///
/// A finite difference is an estimator, not a fact: its error is
/// `ν/h + h²·f‴/6`, and neither `ν` (the evaluator's absolute noise) nor `f‴`
/// is known a priori. A fixed-step oracle therefore reports a number whose
/// accuracy is unknown, and any disagreement with an analytic derivative is
/// unattributable — it may be the analytic side, or it may be the oracle. This
/// type makes the oracle's accuracy part of the answer, so a comparison can be
/// gated on `|analytic − value| > tol + uncertainty` and an oracle that cannot
/// resolve a component says so instead of manufacturing a violation.
#[derive(Clone, Debug)]
pub struct FdDerivative {
    /// Best estimate of the directional derivative.
    pub value: f64,
    /// Estimate of `|value − true derivative|`, taken from the disagreement
    /// between the two lower-order tableau entries that produced `value`.
    /// `f64::INFINITY` when no rung produced a usable entry.
    ///
    /// This is Ridders' estimate, not a proof: on an objective whose noise is
    /// *coherent* across neighbouring steps the tableau can agree better than
    /// it deserves. What it does reliably is stay LARGE when the ladder is
    /// incoherent, which is what [`FdDerivative::resolved`] keys on — so the
    /// safe reading is "small uncertainty ⇒ the ladder converged", and the
    /// unsafe one is "uncertainty is a certified error bar".
    pub uncertainty: f64,
    /// The ladder step whose column produced `value`.
    pub step: f64,
    /// Truncation order of the accepted extrapolant: `2` is a raw central
    /// difference, `4` one Richardson stage, `6` two, and so on.
    pub order: usize,
    /// The raw central differences `(h, D(h))`, coarsest first — kept so a
    /// diagnostic can print the law the gap follows without re-running.
    pub ladder: Vec<(f64, f64)>,
}

/// What a self-certifying oracle is entitled to conclude about one analytic
/// derivative component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdVerdict {
    /// The oracle resolved the component and the analytic value sits inside the
    /// tolerance band widened by the oracle's own uncertainty.
    Agree,
    /// The oracle resolved the component and the analytic value is outside that
    /// band. This is the only verdict that indicts the analytic derivative.
    Disagree,
    /// The oracle's own uncertainty is wider than the tolerance band, so it
    /// cannot tell agreement from disagreement. A statement about the
    /// OBJECTIVE at this point — too noisy, or too sharply curved to difference
    /// — not about the analytic gradient. Reporting it as a gradient violation
    /// is a category error.
    Unresolved,
}

impl FdDerivative {
    /// How sharply the oracle knows its OWN answer:
    /// `rel_tol · max(|value|, abs_floor)`.
    ///
    /// Deliberately independent of any analytic value. Folding the analytic
    /// value in here would let a large analytic number widen the bar the
    /// measurement is held to, so a ladder that returned `-3.2e2 ± 4.9e1`
    /// (15% of itself — pure noise) would count as "resolved" merely because
    /// the analytic side claimed `1e7`. That is how an unmeasurable component
    /// becomes a confident `Disagree`.
    pub fn self_band(&self, rel_tol: f64, abs_floor: f64) -> f64 {
        rel_tol * self.value.abs().max(abs_floor)
    }

    /// The tolerance band the COMPARISON is judged at:
    /// `rel_tol · max(|value|, |analytic|, abs_floor)`. This one does include
    /// the analytic value, because a relative tolerance on a disagreement must
    /// be relative to the larger of the two things being compared.
    pub fn band(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> f64 {
        rel_tol * self.value.abs().max(analytic.abs()).max(abs_floor)
    }

    /// Whether the oracle measured this component sharply enough to be worth
    /// judging against: `uncertainty ≤ self_band`.
    ///
    /// An oracle that knows a `1.1e-3` derivative to `±4.8e-4` has measured
    /// something, but not to a precision that can decide anything, and
    /// pretending otherwise is how a conditioning limit becomes a "gradient
    /// defect".
    pub fn resolved(&self, rel_tol: f64, abs_floor: f64) -> bool {
        self.uncertainty.is_finite() && self.uncertainty <= self.self_band(rel_tol, abs_floor)
    }

}

/// Self-certifying numerical derivative of `f` at `t = 0` (Ridders' method).
///
/// `f` must evaluate the objective along the probe line, i.e. `f(t)` is the
/// objective at `base + t · direction`; the returned value estimates `f′(0)`.
///
/// The method evaluates central differences on a shrinking geometric ladder and
/// runs a Neville extrapolation across it, so column `j` of the tableau has
/// truncation order `2(j+1)`. It then accepts the tableau entry whose two
/// parents agree most closely, and reports that agreement as the uncertainty.
/// This is the standard cure for the fact that no single step is right for
/// every objective: shrinking `h` trades truncation (`h²·f‴/6`) for noise
/// (`ν/h`), and the crossover sits wherever the objective's third derivative
/// and the evaluator's noise floor happen to put it — which, for a criterion
/// evaluated through an inner solve, moves by many orders across a probe grid.
///
/// Cost is `2 · config.rungs` evaluations of `f`; the ladder is run to the end
/// rather than exited early, because a criterion that is not yet in its
/// asymptotic `O(h²)` regime at the coarse rungs produces a non-monotone error
/// sequence there, and an early exit on the first non-improvement would accept
/// a pre-asymptotic entry.
pub fn ridders_derivative<F>(mut f: F, config: RiddersConfig) -> FdDerivative
where
    F: FnMut(f64) -> f64,
{
    ridders_from_stencil(|h| (f(h) - f(-h)) / (2.0 * h), config)
}

/// [`ridders_derivative`] over an arbitrary `O(h²)` derivative stencil.
///
/// `stencil(h)` must return an estimate of the same derivative whose leading
/// error term is `c·h²` — the central difference `(f(h) − f(−h))/2h` is the
/// usual one, but the one-sided three-point rules
/// `(−3f₀ + 4f₁ − f₂)/2h` and `(3f₀ − 4f₁ + f₂)/2h` qualify too, which is what
/// lets a coordinate pinned against a box face be measured with the same
/// self-certification as an interior one instead of being differenced once at a
/// guessed step and reported as fact.
///
/// The Neville recurrence below cancels `h²`, then `h⁴`, and so on, so it is
/// valid for any stencil whose error expansion is in even powers of `h`. A
/// stencil with an `O(h)` term would need ratio `r` rather than `r²` per stage;
/// passing one here silently under-cancels, which the uncertainty estimate then
/// reports as poor agreement rather than hiding.
pub fn ridders_from_stencil<F>(mut stencil: F, config: RiddersConfig) -> FdDerivative
where
    F: FnMut(f64) -> f64,
{
    assert!(
        config.initial_step > 0.0 && config.initial_step.is_finite(),
        "ridders_derivative: initial_step must be finite and positive"
    );
    assert!(
        config.shrink > 1.0 && config.shrink.is_finite(),
        "ridders_derivative: shrink must exceed 1"
    );
    // Four, not two: an extrapolant is only accepted once TWO same-order
    // extrapolants one and two rungs coarser exist to compare it against (see
    // below), and the first of those appears at `i = 3`.
    assert!(
        config.rungs >= 4,
        "ridders_derivative: need at least 4 rungs"
    );

    // `tableau[i][j]` is the order-`2(j+1)` extrapolant built from rungs
    // `i-j ..= i`. Column 0 is the raw stencil value at step `h_i`.
    let mut tableau: Vec<Vec<f64>> = Vec::with_capacity(config.rungs);
    let mut ladder: Vec<(f64, f64)> = Vec::with_capacity(config.rungs);
    let mut best = FdDerivative {
        value: f64::NAN,
        uncertainty: f64::INFINITY,
        step: f64::NAN,
        order: 0,
        ladder: Vec::new(),
    };
    let ratio_sq = config.shrink * config.shrink;

    let mut h = config.initial_step;
    for i in 0..config.rungs {
        let d = stencil(h);
        ladder.push((h, d));
        let mut row = vec![d];
        if i > 0 {
            // Neville across the ladder: each stage removes the leading
            // even power of `h` still present in its two parents.
            let mut factor = ratio_sq;
            for j in 1..=i {
                let left = row[j - 1];
                let up = tableau[i - 1][j - 1];
                let extrapolant = (factor * left - up) / (factor - 1.0);
                row.push(extrapolant);
                // Ridders' error estimate is the agreement of the two entries
                // an extrapolant cancels — but that is an INTERNAL consistency
                // check, and on a noise-dominated ladder two neighbouring
                // garbage values can cancel into agreement by luck. Measured on
                // the #2425 ladder at `rho0@27`, where the raw stencil sweeps
                // `+2.3e-5, +8.9e-5, −6.9e-6, −8.9e-5, +1.7e-4, …` (sign
                // flipping, magnitude growing as `h` shrinks — the `ν/h` law),
                // the pairwise rule accepted an entry to `±8.3e-6` out of
                // `−9.7e-3` and turned a component nothing could measure into a
                // confident gradient violation.
                //
                // So an entry must also be STABLE as the ladder shrinks, over
                // a PLATEAU rather than a single step: it is eligible only once
                // the same-order extrapolants one AND two rungs coarser exist,
                // and both disagreements count. One predecessor is not enough —
                // on the ladder above, rungs 8–10 happen to look locally like
                // `h²` convergence and an order-10 entry agrees with its single
                // predecessor to `1.2e-5`, still minting the false verdict.
                // With two, the best eligible entry is `-1.16e-4 ± 2.3e-4`,
                // which is `Unresolved` — the honest reading of a stencil that
                // sweeps three orders and flips sign four times.
                //
                // Measured cost on converged ladders: none. Across a sinc with
                // scale `1.66e-3`, `e^(2+t)·ln(1+0.3t)`, `sin(1.3+t)·e^(0.4+t)`
                // and a cubic, going from one predecessor to two changes the
                // realized relative error by at most `4e-13` and never by more
                // than the entry's own uncertainty. What it does cost is a
                // SHORT plateau: a component whose ladder is coherent for only
                // three rungs (`rho1@27` on the #2425 ladder, coherent to 0.2%
                // over `h = 1e-2 … 2.5e-3` and then excursion) comes back
                // `Unresolved` instead of `Disagree`. That is the safe
                // direction — a false `Disagree` is precisely the disease
                // #2461 is a case of — and the ladder length is a knob for a
                // caller who wants the plateau extended.
                let plateau = if j + 2 <= i {
                    Some((tableau[i - 1][j], tableau[i - 2][j]))
                } else {
                    None
                };
                if let Some((previous, before_that)) = plateau {
                    let error = (extrapolant - left)
                        .abs()
                        .max((extrapolant - up).abs())
                        .max((extrapolant - previous).abs())
                        .max((extrapolant - before_that).abs());
                    if extrapolant.is_finite() && error < best.uncertainty {
                        best.value = extrapolant;
                        best.uncertainty = error;
                        best.step = h;
                        best.order = 2 * (j + 1);
                    }
                }
                factor *= ratio_sq;
            }
        }
        tableau.push(row);
        h /= config.shrink;
    }
    best.ladder = ladder;
    best
}

