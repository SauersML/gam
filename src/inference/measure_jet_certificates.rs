//! V∞ §7 per-fit certificate battery for the measure-jet stack.
//!
//! This module computes the small family of *algebraic* certificates that
//! guard a single fit: every numerical shortcut taken inside the measure-jet
//! machinery (frame change-of-basis, far-field kernel truncation, band-depth
//! refinement) must produce an error that is negligible against the statistical
//! noise of the reported quantities. "Negligible" is made precise by the
//! **do-no-harm budget**: every algebraic tolerance must sit at or below
//! `δ · min_T se(T)`, i.e. a fraction `δ` of the smallest reported standard
//! error in the fit. A tolerance below that budget cannot move any reported
//! number by a statistically meaningful amount, so the shortcut "does no harm".
//!
//! Each individual certificate returns a [`CertVerdict`] (`Pass` / `Refine`
//! / `Fail`) together with the signed numeric **margin** by which the bound
//! cleared (or missed) its threshold. The certificates are pure functions of
//! their numeric inputs — no global state, no environment, fully deterministic
//! — so they are cheap to evaluate once per fit and trivial to oracle-test.
//!
//! The four certificates:
//!
//! * **Frame ratio `B/A`** — the conditioning of the analysis/synthesis frame
//!   used to express the measure jet in a working basis. A ratio of `1` is a
//!   tight (Parseval) frame; large ratios mean an ill-conditioned change of
//!   basis. Consumed from the frame module if one is present, else the ratio is
//!   supplied directly as `(a_bound, b_bound)`.
//! * **FGT / far-field truncation** — truncating the fast-Gauss / kernel
//!   far-field at radius `ρ` (in units of the kernel scale) drops a tail
//!   bounded by `e^{-ρ²/2}`. The certificate reports that closed-form bound and
//!   checks it against the budget.
//! * **Refine-one-level δ** — given the band energy at depth `L` and at depth
//!   `L+1`, the relative change `|E_{L+1} − E_L| / |E_L|` measures whether one
//!   more level of refinement still moves the answer. Small ⇒ converged.
//! * **Do-no-harm budget** — the umbrella check: the maximum algebraic
//!   tolerance across the battery must be `≤ δ · min_T se(T)`.
//!
//! All four feed a [`MeasureJetCertBattery`] whose [`std::fmt::Display`] (and the
//! explicit [`MeasureJetCertBattery::report_lines`]) renders aligned text report
//! lines in the style of `src/report.rs` (scientific `{:.6e}` floats, an em-dash
//! `—` for absent quantities, an explicit verdict marker per line).

use std::fmt;

/// Default do-no-harm fraction: an algebraic tolerance is "harmless" when it is
/// at most this fraction of the smallest reported standard error in the fit.
///
/// Budgeting at `1e-3 · min se` means the worst-case algebraic error is three
/// orders of magnitude below the tightest statistical error bar, so it cannot
/// perturb any reported figure by a statistically meaningful amount.
pub const DEFAULT_DO_NO_HARM_DELTA: f64 = 1.0e-3;

/// Em-dash rendered for an absent / non-finite quantity (matches `report.rs`).
const EM_DASH: &str = "\u{2014}";

/// Three-state outcome of a single certificate.
///
/// The ladder is deliberately ordered `Fail < Refine < Pass` so a battery's
/// overall verdict is simply the minimum over its members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CertVerdict {
    /// The bound is violated outright; the shortcut is *not* safe and the
    /// result must not be trusted as-is.
    Fail,
    /// The bound is not cleared with margin to spare; one more level of
    /// refinement (finer band depth / larger truncation radius) is advised.
    Refine,
    /// The bound clears its threshold with margin; the shortcut does no harm.
    Pass,
}

impl CertVerdict {
    /// Short uppercase tag used in rendered report lines.
    pub fn tag(self) -> &'static str {
        match self {
            CertVerdict::Pass => "PASS",
            CertVerdict::Refine => "REFINE",
            CertVerdict::Fail => "FAIL",
        }
    }

    /// `true` only for [`CertVerdict::Pass`].
    pub fn is_pass(self) -> bool {
        matches!(self, CertVerdict::Pass)
    }
}

impl fmt::Display for CertVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// Result of one certificate: a verdict plus the signed margin by which the
/// bound cleared (`> 0`) or missed (`< 0`) its threshold.
///
/// `value` is the certificate's primary computed quantity (the bound, the
/// ratio, the relative change…) and `threshold` is the budget it was compared
/// against. `margin = threshold − value` for "smaller is better" certificates,
/// so a positive margin always means "inside budget".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertResult {
    /// The verdict for this certificate.
    pub verdict: CertVerdict,
    /// The certificate's primary computed quantity.
    pub value: f64,
    /// The budget / threshold the value was compared against (`NaN` if the
    /// certificate has no scalar threshold, e.g. a pure diagnostic).
    pub threshold: f64,
    /// Signed clearance: `threshold − value` for smaller-is-better bounds.
    /// Positive ⇒ inside budget; negative ⇒ over budget.
    pub margin: f64,
}

impl CertResult {
    /// Build a "smaller-is-better" result: `value` must clear `threshold`.
    ///
    /// `value ≤ threshold` ⇒ inside budget. To leave room for `Refine`, the
    /// `pass` band requires the value to clear the threshold by the relative
    /// `slack` fraction (`value ≤ (1 − slack)·threshold`); between that and the
    /// threshold is `Refine`; above the threshold is `Fail`.
    fn smaller_is_better(value: f64, threshold: f64, slack: f64) -> CertResult {
        let margin = threshold - value;
        let verdict = if !value.is_finite() || !threshold.is_finite() {
            CertVerdict::Fail
        } else if value <= threshold * (1.0 - slack) {
            CertVerdict::Pass
        } else if value <= threshold {
            CertVerdict::Refine
        } else {
            CertVerdict::Fail
        };
        CertResult { verdict, value, threshold, margin }
    }
}

/// Frame-ratio certificate input: the analysis/synthesis frame bounds `A ≤ B`.
///
/// When a frame module supplies the bounds, pass them through here; otherwise
/// the caller may construct this directly from a precomputed ratio via
/// [`FrameBounds::from_ratio`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameBounds {
    /// Lower frame bound `A` (`> 0`).
    pub a_bound: f64,
    /// Upper frame bound `B` (`≥ A`).
    pub b_bound: f64,
}

impl FrameBounds {
    /// Construct from explicit bounds.
    pub fn new(a_bound: f64, b_bound: f64) -> FrameBounds {
        FrameBounds { a_bound, b_bound }
    }

    /// Construct from a precomputed ratio `r = B/A` (sets `A = 1`, `B = r`),
    /// for the placeholder path where only the ratio is known.
    pub fn from_ratio(ratio: f64) -> FrameBounds {
        FrameBounds { a_bound: 1.0, b_bound: ratio }
    }

    /// The frame condition number `B/A` (`1` ⇒ tight/Parseval frame).
    pub fn ratio(&self) -> f64 {
        if self.a_bound > 0.0 {
            self.b_bound / self.a_bound
        } else {
            f64::INFINITY
        }
    }
}

/// Frame-ratio certificate: how far the working frame is from tight (`B/A = 1`).
///
/// The relevant *algebraic tolerance* contributed by an ill-conditioned frame
/// scales with `(B/A − 1)`: a tight frame contributes nothing, while a ratio of
/// `1 + ε` injects a relative round-trip error of order `ε`. The certificate
/// passes when that excess is within the supplied tolerance budget `tol`.
pub fn frame_ratio_cert(bounds: FrameBounds, tol: f64) -> CertResult {
    let excess = (bounds.ratio() - 1.0).abs();
    CertResult::smaller_is_better(excess, tol, 0.5)
}

/// FGT / kernel far-field truncation certificate.
///
/// Truncating a Gaussian far-field expansion at radius `rho` (in units of the
/// kernel scale) drops a tail bounded *exactly* by `e^{-ρ²/2}`. This is the
/// closed-form bound used throughout the fast-Gauss / kernel machinery. The
/// certificate reports that bound and checks it against the tolerance `tol`.
pub fn truncation_bound(rho: f64) -> f64 {
    (-0.5 * rho * rho).exp()
}

/// FGT / far-field truncation certificate: the dropped tail `e^{-ρ²/2}` must be
/// within the tolerance budget `tol`.
pub fn truncation_cert(rho: f64, tol: f64) -> CertResult {
    let bound = truncation_bound(rho);
    CertResult::smaller_is_better(bound, tol, 0.5)
}

/// Relative energy change between band depth `L` and `L+1`.
///
/// `|E_{L+1} − E_L| / |E_L|`. Monotone non-decreasing in the energy gap
/// `|E_{L+1} − E_L|` for a fixed `E_L`. Returns `+∞` when `E_L` is zero so a
/// non-trivial change off a zero base is never spuriously "converged".
pub fn refine_delta(energy_l: f64, energy_l_plus_1: f64) -> f64 {
    let gap = (energy_l_plus_1 - energy_l).abs();
    let base = energy_l.abs();
    if base > 0.0 {
        gap / base
    } else if gap == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Refine-one-level certificate: the relative change from adding one more band
/// level must be within the tolerance budget `tol` (i.e. one more level of
/// refinement would not move the answer meaningfully).
pub fn refine_cert(energy_l: f64, energy_l_plus_1: f64, tol: f64) -> CertResult {
    let delta = refine_delta(energy_l, energy_l_plus_1);
    CertResult::smaller_is_better(delta, tol, 0.5)
}

/// The do-no-harm budget: `δ · min_T se(T)`.
///
/// `delta` is the harmless-fraction (see [`DEFAULT_DO_NO_HARM_DELTA`]) and
/// `min_se` is the smallest reported standard error across the fit. Returns
/// `0.0` (an unsatisfiable budget, forcing `Fail`) when the inputs are not
/// finite or non-positive, so a degenerate `se` can never silently certify.
pub fn do_no_harm_budget(delta: f64, min_se: f64) -> f64 {
    if delta.is_finite() && min_se.is_finite() && delta > 0.0 && min_se > 0.0 {
        delta * min_se
    } else {
        0.0
    }
}

/// Do-no-harm certificate: the maximum algebraic tolerance taken anywhere in the
/// battery (`max_tol`) must be `≤ δ · min_T se(T)`.
///
/// Passes iff `max_tol ≤ budget`; the boundary `max_tol == budget` is a pass
/// (the tolerance is exactly at — not over — the harm threshold).
pub fn do_no_harm_cert(max_tol: f64, delta: f64, min_se: f64) -> CertResult {
    let budget = do_no_harm_budget(delta, min_se);
    let margin = budget - max_tol;
    let verdict = if !max_tol.is_finite() || budget <= 0.0 {
        CertVerdict::Fail
    } else if max_tol <= budget {
        CertVerdict::Pass
    } else {
        CertVerdict::Fail
    };
    CertResult { verdict, value: max_tol, threshold: budget, margin }
}

/// Inputs for assembling a full per-fit certificate battery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureJetCertInputs {
    /// Analysis/synthesis frame bounds (`B/A` conditioning).
    pub frame: FrameBounds,
    /// Far-field truncation radius `ρ` in kernel-scale units.
    pub truncation_rho: f64,
    /// Band energy at the current refinement depth `L`.
    pub energy_l: f64,
    /// Band energy at depth `L+1`.
    pub energy_l_plus_1: f64,
    /// Harmless fraction `δ` for the do-no-harm budget.
    pub delta: f64,
    /// Smallest reported standard error `min_T se(T)` across the fit.
    pub min_se: f64,
}

/// The assembled per-fit certificate battery (V∞ §7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureJetCertBattery {
    /// Frame-ratio certificate.
    pub frame: CertResult,
    /// Far-field truncation certificate.
    pub truncation: CertResult,
    /// Refine-one-level certificate.
    pub refine: CertResult,
    /// Do-no-harm umbrella certificate.
    pub do_no_harm: CertResult,
    /// The do-no-harm budget `δ · min_T se(T)` (the tolerance every other
    /// certificate is budgeted against).
    pub budget: f64,
}

impl MeasureJetCertBattery {
    /// Assemble the battery from raw inputs.
    ///
    /// The do-no-harm budget `δ · min_se` is computed first and used as the
    /// per-certificate tolerance for the frame, truncation and refine checks
    /// (so every algebraic shortcut is held to the same harm threshold). The
    /// do-no-harm certificate then verifies that the *largest* of those
    /// algebraic quantities is itself within budget.
    pub fn assemble(inputs: MeasureJetCertInputs) -> MeasureJetCertBattery {
        let budget = do_no_harm_budget(inputs.delta, inputs.min_se);

        let frame = frame_ratio_cert(inputs.frame, budget);
        let truncation = truncation_cert(inputs.truncation_rho, budget);
        let refine = refine_cert(inputs.energy_l, inputs.energy_l_plus_1, budget);

        // The umbrella check: the worst (largest) algebraic quantity vs budget.
        let max_tol = frame
            .value
            .max(truncation.value)
            .max(refine.value);
        let do_no_harm = do_no_harm_cert(max_tol, inputs.delta, inputs.min_se);

        MeasureJetCertBattery { frame, truncation, refine, do_no_harm, budget }
    }

    /// Overall verdict: the minimum (most severe) verdict across the battery.
    pub fn verdict(&self) -> CertVerdict {
        self.frame
            .verdict
            .min(self.truncation.verdict)
            .min(self.refine.verdict)
            .min(self.do_no_harm.verdict)
    }

    /// `true` iff every certificate in the battery passes.
    pub fn all_pass(&self) -> bool {
        self.verdict().is_pass()
    }

    /// Render the battery as aligned text report lines (style of `report.rs`:
    /// scientific `{:.6e}` floats, em-dash for absent quantities, an explicit
    /// verdict tag per line).
    pub fn report_lines(&self) -> Vec<String> {
        let row = |label: &str, r: &CertResult| -> String {
            format!(
                "  {label:<22} value={} threshold={} margin={} [{}]",
                fmt_sci(r.value),
                fmt_sci(r.threshold),
                fmt_sci(r.margin),
                r.verdict.tag(),
            )
        };
        vec![
            "measure-jet certificate battery (V∞ §7)".to_string(),
            format!("  do-no-harm budget δ·min se = {}", fmt_sci(self.budget)),
            row("frame ratio B/A", &self.frame),
            row("far-field truncation", &self.truncation),
            row("refine-one-level", &self.refine),
            row("do-no-harm", &self.do_no_harm),
            format!("  overall: [{}]", self.verdict().tag()),
        ]
    }
}

impl fmt::Display for MeasureJetCertBattery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in self.report_lines() {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

/// Format a float in `report.rs` style: scientific `{:.6e}`, em-dash if absent.
fn fmt_sci(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6e}")
    } else {
        EM_DASH.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The truncation bound matches the closed form `e^{-ρ²/2}` exactly.
    #[test]
    fn truncation_bound_matches_closed_form() {
        for &rho in &[0.0_f64, 0.5, 1.0, 2.0, 3.0, 4.5] {
            let expected = (-0.5 * rho * rho).exp();
            let got = truncation_bound(rho);
            assert_eq!(got, expected, "rho={rho}");
        }
        // ρ = 0 ⇒ no truncation ⇒ full tail bound of 1.
        assert_eq!(truncation_bound(0.0), 1.0);
    }

    /// Do-no-harm passes iff `max_tol ≤ δ·min_se`; boundary is a pass.
    #[test]
    fn do_no_harm_boundary() {
        let delta = 1.0e-3;
        let min_se = 2.0;
        let budget = do_no_harm_budget(delta, min_se);
        assert_eq!(budget, 2.0e-3);

        // Strictly inside budget ⇒ Pass.
        let inside = do_no_harm_cert(budget * 0.5, delta, min_se);
        assert_eq!(inside.verdict, CertVerdict::Pass);
        assert!(inside.margin > 0.0);

        // Exactly at the boundary ⇒ Pass (≤ is inclusive).
        let at = do_no_harm_cert(budget, delta, min_se);
        assert_eq!(at.verdict, CertVerdict::Pass);
        assert_eq!(at.margin, 0.0);

        // Just over the boundary ⇒ Fail.
        let over = do_no_harm_cert(budget * (1.0 + 1.0e-9), delta, min_se);
        assert_eq!(over.verdict, CertVerdict::Fail);
        assert!(over.margin < 0.0);
    }

    /// A degenerate (non-positive / non-finite) se cannot certify.
    #[test]
    fn do_no_harm_degenerate_se_fails() {
        let d = DEFAULT_DO_NO_HARM_DELTA;
        assert_eq!(do_no_harm_budget(d, 0.0), 0.0);
        assert_eq!(do_no_harm_budget(d, -1.0), 0.0);
        assert_eq!(do_no_harm_budget(d, f64::NAN), 0.0);
        // Even a tiny tolerance fails against a zero budget.
        let r = do_no_harm_cert(1.0e-30, d, 0.0);
        assert_eq!(r.verdict, CertVerdict::Fail);
    }

    /// refine-delta is monotone non-decreasing in the energy gap.
    #[test]
    fn refine_delta_monotone_in_gap() {
        let base = 4.0_f64;
        let gaps = [0.0_f64, 0.1, 0.5, 1.0, 2.0, 10.0];
        let mut prev = f64::NEG_INFINITY;
        for &g in &gaps {
            let d = refine_delta(base, base + g);
            assert!(d >= prev, "non-monotone at gap={g}: {d} < {prev}");
            prev = d;
        }
        // Closed-form value at a known point.
        assert_eq!(refine_delta(4.0, 5.0), 0.25);
        // Symmetric in the sign of the gap.
        assert_eq!(refine_delta(4.0, 3.0), refine_delta(4.0, 5.0));
    }

    /// refine-delta off a zero base: zero gap converged, non-zero gap diverges.
    #[test]
    fn refine_delta_zero_base() {
        assert_eq!(refine_delta(0.0, 0.0), 0.0);
        assert_eq!(refine_delta(0.0, 1.0), f64::INFINITY);
    }

    /// Frame ratio: a tight frame (B/A = 1) contributes zero excess.
    #[test]
    fn frame_ratio_tight_is_zero_excess() {
        let tight = FrameBounds::new(1.0, 1.0);
        assert_eq!(tight.ratio(), 1.0);
        let cert = frame_ratio_cert(tight, 1.0e-3);
        assert_eq!(cert.value, 0.0);
        assert_eq!(cert.verdict, CertVerdict::Pass);

        // from_ratio round-trips the ratio.
        assert_eq!(FrameBounds::from_ratio(3.0).ratio(), 3.0);
        // Degenerate A ⇒ infinite ratio.
        assert!(FrameBounds::new(0.0, 1.0).ratio().is_infinite());
    }

    /// Verdict ordering: Fail < Refine < Pass, so battery min is most severe.
    #[test]
    fn verdict_ordering() {
        assert!(CertVerdict::Fail < CertVerdict::Refine);
        assert!(CertVerdict::Refine < CertVerdict::Pass);
        assert_eq!(
            CertVerdict::Pass.min(CertVerdict::Fail),
            CertVerdict::Fail
        );
    }

    /// A fully-passing battery: all four certificates pass and render cleanly.
    #[test]
    fn battery_all_pass_and_renders() {
        // min_se large so the budget comfortably covers small algebraic terms.
        let inputs = MeasureJetCertInputs {
            frame: FrameBounds::new(1.0, 1.0),     // tight ⇒ excess 0
            truncation_rho: 6.0,                    // e^{-18} ≈ 1.5e-8
            energy_l: 1.0,
            energy_l_plus_1: 1.0,                   // refine delta 0
            delta: 1.0,                             // budget = min_se
            min_se: 1.0e-3,                         // budget 1e-3 ≥ all terms
        };
        let battery = MeasureJetCertBattery::assemble(inputs);
        assert!(battery.all_pass(), "battery: {battery}");
        assert_eq!(battery.verdict(), CertVerdict::Pass);

        let lines = battery.report_lines();
        assert!(lines[0].contains("V∞ §7"));
        assert!(lines.iter().any(|l| l.contains("frame ratio B/A")));
        assert!(lines.iter().any(|l| l.contains("PASS")));
        // Display matches report_lines.
        let shown = format!("{battery}");
        assert!(shown.contains("do-no-harm"));
    }

    /// A failing battery: an over-budget truncation forces overall Fail.
    #[test]
    fn battery_fails_on_over_budget_truncation() {
        let inputs = MeasureJetCertInputs {
            frame: FrameBounds::new(1.0, 1.0),
            truncation_rho: 0.5,    // e^{-0.125} ≈ 0.88, far over a tiny budget
            energy_l: 1.0,
            energy_l_plus_1: 1.0,
            delta: 1.0e-3,
            min_se: 1.0e-3,         // budget 1e-6, tiny
        };
        let battery = MeasureJetCertBattery::assemble(inputs);
        assert!(!battery.all_pass());
        assert_eq!(battery.verdict(), CertVerdict::Fail);
        assert_eq!(battery.truncation.verdict, CertVerdict::Fail);
    }
}
