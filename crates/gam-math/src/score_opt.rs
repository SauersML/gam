//! Global maximization of a smooth 1-D score on a bounded domain from joint
//! `(value, derivative)` evaluations — the shared smoothing-parameter (log-λ)
//! selector behind the REML/profile searches (SPEC rule 18: grid search is
//! never allowed).
//!
//! The search isolates every stationary interval by ADAPTIVE subdivision
//! driven by derivative sign changes and a cubic-Hermite model test — not a
//! fixed lattice. An interval whose endpoint derivatives share a sign is
//! split at its midpoint until the midpoint sample agrees with the Hermite
//! model built from the endpoint `(value, derivative)` pairs AND that model
//! is derivative-sign-constant, at which point the interval is certified to
//! contain no interior stationary point and dropped. Every derivative sign
//! change is refined to a stationary point by a safeguarded secant/bisection
//! root find on the derivative. Subdivision is best-first: intervals are
//! processed in order of an optimistic value ceiling, so the evaluation
//! budget concentrates where the score can still beat the incumbent.
//!
//! The returned optimum is the best value over every sample taken: all
//! bracketed interior stationary points compared against the exactly
//! evaluated domain endpoints, so boundary optima (the null-recovery /
//! no-penalty ends of a log-λ domain) always compete on equal terms.

/// One evaluation of the score: abscissa, value, and analytic derivative.
#[derive(Clone, Copy, Debug)]
pub struct ScoreSample {
    pub x: f64,
    pub value: f64,
    pub derivative: f64,
}

/// Relative tolerance of the Hermite-model agreement test: a midpoint sample
/// within this fraction of the interval's own value-variation scale confirms
/// the model; a larger disagreement forces subdivision.
const MODEL_RTOL: f64 = 1e-2;
/// Depth cap on recursive subdivision (width shrinks 2^24-fold below the
/// domain — far past any physically meaningful log-λ feature). A safety
/// valve against evaluation noise masquerading as structure, not a lattice.
const MAX_DEPTH: usize = 24;
/// Evaluation-count safety valve (NOT a wall-clock budget): past it the
/// best-first queue is drained and the incumbent returned. Generous against
/// the ≈75 evaluations the lattice+golden scheme this replaces used to burn.
const MAX_EVALS: usize = 200;
/// Iteration cap of the safeguarded derivative root find inside one bracket;
/// forced bisection guarantees the bracket halves at least every other step,
/// so 80 iterations resolve any bracket to ~2^-40 of its width.
const BRACKET_MAX_ITERS: usize = 80;

struct Search<'a, E> {
    eval: &'a mut dyn FnMut(f64) -> Result<(f64, f64), E>,
    evals: usize,
    best: Option<ScoreSample>,
}

impl<E: From<String>> Search<'_, E> {
    fn sample(&mut self, x: f64) -> Result<ScoreSample, E> {
        let (value, derivative) = (self.eval)(x)?;
        if !(value.is_finite() && derivative.is_finite()) {
            return Err(E::from(format!(
                "score search: non-finite score at x = {x} (value {value}, derivative {derivative})"
            )));
        }
        self.evals += 1;
        let s = ScoreSample {
            x,
            value,
            derivative,
        };
        if self.best.is_none_or(|b| s.value > b.value) {
            self.best = Some(s);
        }
        Ok(s)
    }

    /// Safeguarded secant/bisection root find on the derivative inside a
    /// sign-change bracket; every iterate is a candidate via `sample`.
    fn refine_bracket(
        &mut self,
        mut a: ScoreSample,
        mut b: ScoreSample,
        x_tol: f64,
    ) -> Result<(), E> {
        let mut force_bisect = false;
        for _ in 0..BRACKET_MAX_ITERS {
            let width = b.x - a.x;
            if width <= x_tol || self.evals >= MAX_EVALS {
                break;
            }
            let mid = 0.5 * (a.x + b.x);
            let x = if force_bisect {
                mid
            } else {
                // Secant step on the derivative, safeguarded into the
                // bracket's interior (margin keeps progress when the secant
                // hugs an endpoint).
                let denom = b.derivative - a.derivative;
                let secant = (a.x * b.derivative - b.x * a.derivative) / denom;
                let margin = 0.01 * width;
                if secant.is_finite() && secant > a.x + margin && secant < b.x - margin {
                    secant
                } else {
                    mid
                }
            };
            let s = self.sample(x)?;
            if s.derivative == 0.0 {
                break;
            }
            if (s.derivative > 0.0) == (a.derivative > 0.0) {
                a = s;
            } else {
                b = s;
            }
            // If the accepted step failed to halve the bracket, force a
            // bisection next round (Dekker-style guarantee).
            force_bisect = (b.x - a.x) > 0.5 * width;
        }
        Ok(())
    }
}

/// Cubic-Hermite midpoint value prediction on `[a, b]`.
fn hermite_mid_value(a: &ScoreSample, b: &ScoreSample) -> f64 {
    let h = b.x - a.x;
    0.5 * (a.value + b.value) + 0.125 * h * (a.derivative - b.derivative)
}

/// Cubic-Hermite midpoint derivative prediction on `[a, b]`.
fn hermite_mid_deriv(a: &ScoreSample, b: &ScoreSample) -> f64 {
    let h = b.x - a.x;
    1.5 * (b.value - a.value) / h - 0.25 * (a.derivative + b.derivative)
}

/// Whether the cubic-Hermite interpolant on `[a, b]` has a stationary point
/// strictly inside the interval: its derivative is the quadratic (in the
/// normalized coordinate `t ∈ [0, 1]`)
/// `p′(t) = (3d₀ + 3d₁ − 6Δ)·t² + (6Δ − 4d₀ − 2d₁)·t + d₀`, `Δ = (v₁−v₀)/h`.
fn hermite_interior_stationary(a: &ScoreSample, b: &ScoreSample) -> bool {
    let h = b.x - a.x;
    let d0 = a.derivative;
    let d1 = b.derivative;
    let slope = (b.value - a.value) / h;
    let qa = 3.0 * (d0 + d1) - 6.0 * slope;
    let qb = 6.0 * slope - 4.0 * d0 - 2.0 * d1;
    let qc = d0;
    let root_inside = |t: f64| t.is_finite() && t > 0.0 && t < 1.0;
    if qa == 0.0 {
        return qb != 0.0 && root_inside(-qc / qb);
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    // Numerically stable quadratic roots.
    let q = -0.5 * (qb + if qb >= 0.0 { sq } else { -sq });
    let r1 = q / qa;
    let r2 = if q != 0.0 { qc / q } else { f64::NAN };
    root_inside(r1) || root_inside(r2)
}

/// Globally maximize a smooth score on `[lo, hi]` from joint
/// `(value, derivative)` evaluations. Returns the best sample found: the
/// winner among all certified interior stationary points and the two domain
/// endpoints. `x_tol` is the abscissa resolution demanded of each interior
/// stationary point.
pub fn maximize_score_1d<E: From<String>>(
    lo: f64,
    hi: f64,
    x_tol: f64,
    eval: &mut dyn FnMut(f64) -> Result<(f64, f64), E>,
) -> Result<ScoreSample, E> {
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(E::from(format!(
            "score search: invalid domain [{lo}, {hi}]"
        )));
    }
    if !(x_tol > 0.0 && x_tol.is_finite()) {
        return Err(E::from(format!(
            "score search: invalid abscissa tolerance {x_tol}"
        )));
    }
    let mut search = Search {
        eval,
        evals: 0,
        best: None,
    };
    // Boundary (null-recovery) candidates first — evaluated exactly, they
    // compete with every interior stationary point on value.
    let a0 = search.sample(lo)?;
    if hi == lo {
        return Ok(a0);
    }
    let b0 = search.sample(hi)?;
    let mut intervals: Vec<(ScoreSample, ScoreSample, usize)> = vec![(a0, b0, 0)];
    while !intervals.is_empty() && search.evals < MAX_EVALS {
        // Best-first: pop the interval with the largest optimistic value
        // ceiling (endpoint value plus a one-sided derivative extrapolation to
        // the midpoint). Ordering only — nothing is pruned by the ceiling.
        let mut pick = 0usize;
        let mut pick_ub = f64::NEG_INFINITY;
        for (i, (a, b, _)) in intervals.iter().enumerate() {
            let ub = a.value.max(b.value)
                + 0.5 * (b.x - a.x) * a.derivative.abs().max(b.derivative.abs());
            if ub > pick_ub {
                pick_ub = ub;
                pick = i;
            }
        }
        let (a, b, depth) = intervals.swap_remove(pick);
        if b.x - a.x <= x_tol {
            continue;
        }
        if a.derivative * b.derivative < 0.0 {
            search.refine_bracket(a, b, x_tol)?;
            continue;
        }
        if depth >= MAX_DEPTH {
            continue;
        }
        let m = search.sample(0.5 * (a.x + b.x))?;
        // Hermite-model agreement at the midpoint: value AND derivative must
        // match within the interval's own variation scale for its halves to
        // be eligible for monotonicity certification.
        let h = b.x - a.x;
        let value_scale = (b.value - a.value).abs()
            + 0.25 * h * (a.derivative.abs() + m.derivative.abs() + b.derivative.abs())
            + f64::EPSILON * (a.value.abs() + b.value.abs());
        let deriv_scale = a.derivative.abs() + m.derivative.abs() + b.derivative.abs()
            + (b.value - a.value).abs() / h
            + f64::EPSILON;
        let model_ok = (m.value - hermite_mid_value(&a, &b)).abs() <= MODEL_RTOL * value_scale
            && (m.derivative - hermite_mid_deriv(&a, &b)).abs() <= MODEL_RTOL * deriv_scale;
        for (u, v) in [(a, m), (m, b)] {
            if u.derivative * v.derivative < 0.0 {
                // Sign-change bracket: queue for root refinement.
                intervals.push((u, v, depth + 1));
            } else if !model_ok
                || u.derivative * v.derivative == 0.0
                || hermite_interior_stationary(&u, &v)
            {
                intervals.push((u, v, depth + 1));
            }
            // Otherwise: certified derivative-sign-constant — no interior
            // stationary point; endpoints already compete as candidates.
        }
    }
    search
        .best
        .ok_or_else(|| E::from("score search: no sample taken".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        lo: f64,
        hi: f64,
        f: impl Fn(f64) -> (f64, f64),
    ) -> ScoreSample {
        let mut eval = |x: f64| -> Result<(f64, f64), String> { Ok(f(x)) };
        maximize_score_1d(lo, hi, 1e-9, &mut eval).unwrap()
    }

    /// A narrow high peak next to a broad low one: a 25-point lattice over
    /// [−18, 18] (step 1.5) samples the narrow basin at best 0.5σ off-center
    /// and hands golden-section the WRONG basin; the adaptive search must
    /// find the true global maximum.
    #[test]
    fn narrow_high_peak_beats_broad_basin() {
        let f = |x: f64| {
            let n = 1.5 * (-((x - 2.0) / 0.35_f64).powi(2) / 2.0).exp();
            let b = (-((x + 6.0) / 3.0_f64).powi(2) / 2.0).exp();
            let dn = n * (-(x - 2.0) / (0.35_f64 * 0.35));
            let db = b * (-(x + 6.0) / 9.0);
            (n + b, dn + db)
        };
        let best = run(-18.0, 18.0, f);
        assert!(
            (best.x - 2.0).abs() < 1e-6,
            "expected the narrow global basin at x = 2, got x = {} (value {})",
            best.x,
            best.value
        );
        assert!((best.value - 1.5).abs() < 1e-9);
    }

    /// Monotone score: the boundary (null-recovery analogue) must win.
    #[test]
    fn boundary_optimum_wins() {
        let best = run(-4.0, 9.0, |x| (0.3 * x, 0.3));
        assert_eq!(best.x, 9.0);
    }

    /// Oscillatory score with several interior maxima: the global one wins.
    #[test]
    fn oscillatory_global_maximum() {
        let f = |x: f64| (x.sin() + 0.1 * x, x.cos() + 0.1);
        let best = run(0.0, 20.0, f);
        // Maxima at cos x = −0.1: x = acos(−0.1) + 2πk; the tilt makes the
        // largest in-domain one (k = 2) global.
        let expect = (-0.1_f64).acos() + 4.0 * std::f64::consts::PI;
        assert!(
            (best.x - expect).abs() < 1e-6,
            "expected x = {expect}, got {}",
            best.x
        );
    }

    /// Exact quadratic: the stationary point is resolved to tolerance.
    #[test]
    fn quadratic_interior_maximum() {
        let best = run(-10.0, 10.0, |x| (-(x - 1.25) * (x - 1.25), -2.0 * (x - 1.25)));
        assert!((best.x - 1.25).abs() < 1e-7);
    }

    /// Evaluation errors propagate.
    #[test]
    fn eval_error_propagates() {
        let mut eval =
            |_: f64| -> Result<(f64, f64), String> { Err("boom".to_string()) };
        assert!(maximize_score_1d(0.0, 1.0, 1e-9, &mut eval).is_err());
    }
}
