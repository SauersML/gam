//! Subsample-converge-then-full-pass ρ schedule with importance weights (#973).
//!
//! # The schedule
//!
//! Every outer ρ (smoothing-parameter) step is, formally, a **full corpus
//! pass**: the REML/EFS gradient is a sum over all corpus rows. On a
//! multi-million-row out-of-core corpus, running every early ρ step over the
//! full corpus is wasteful — the early steps are far from the optimum and only
//! need a *coarse* gradient to point downhill. So the cascade runs early ρ
//! steps on a deterministic **subsample** of rows and the late steps on the
//! full corpus, sharpening as it converges:
//!
//! ```text
//!   step 0 .. k-1 : subsample of fraction f_t  (coarse, cheap)
//!   step k ..     : full corpus pass           (exact, for the final descent)
//! ```
//!
//! # Subsample-honesty contract
//!
//! A subsample step must remain an **unbiased estimator of the full-corpus
//! pass** — otherwise the cascade would converge to the wrong ρ. The honesty
//! contract: each *included* row carries an **importance weight** equal to the
//! reciprocal of its inclusion probability, so the weighted subsample sum has
//! the same expectation as the full-corpus sum. With a uniform-fraction `f_t`
//! subsample every included row's weight is `1 / f_t`; the full pass has weight
//! `1.0` for every row. A row's contribution to the gradient is multiplied by
//! its weight, so the subsample "represents" the whole corpus rather than a
//! shrunken copy of it.
//!
//! # Deterministic selection
//!
//! Membership is decided by hashing the row's stable `row_id` (from
//! [`super::shard_reader::CorpusRowSource`]) through the canonical
//! `splitmix64` finalizer and comparing the top bits against a per-step
//! threshold. This is:
//!
//! * **deterministic** — no RNG state, no clock; the same `row_id` makes the
//!   same in/out decision on every run and platform (the determinism contract
//!   the rest of #973 depends on),
//! * **stable across batches and shards** — membership depends only on
//!   `row_id`, never on which batch a row arrives in, and
//! * **monotone-free of bias** — `splitmix64` makes the hashed ids uniform, so
//!   the threshold cut realizes the intended fraction in expectation.

use crate::linalg::utils::splitmix64_hash;

/// Fraction of the full `u64` hash space; comparing a hashed id against
/// `(fraction * 2^64)` realizes a Bernoulli(`fraction`) inclusion.
const HASH_SPACE: f64 = u64::MAX as f64 + 1.0;

/// One step of the ρ cascade: the subsample fraction to use and the
/// corresponding importance weight every included row carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhoStepPlan {
    /// 0-based index of this outer ρ step.
    pub step: usize,
    /// Fraction of corpus rows this step visits (`1.0` == full pass).
    pub fraction: f64,
    /// Importance weight an included row contributes (`1.0 / fraction`).
    pub importance_weight: f64,
    /// True once this step is a full corpus pass (the honest final descent).
    pub is_full_pass: bool,
}

impl RhoStepPlan {
    /// Decide whether `row_id` is in this step's subsample, and if so return its
    /// importance weight. A full pass includes every row at weight `1.0`.
    #[inline]
    pub fn includes(&self, row_id: u64) -> Option<f64> {
        if self.is_full_pass {
            return Some(1.0);
        }
        if row_in_fraction(row_id, self.fraction) {
            Some(self.importance_weight)
        } else {
            None
        }
    }
}

/// Deterministic Bernoulli inclusion of `row_id` at the given `fraction`.
///
/// Hashes `row_id` with the canonical `splitmix64` finalizer and includes the
/// row iff the hash falls in the leading `fraction` of the `u64` space.
/// `fraction <= 0` excludes everything; `fraction >= 1` includes everything.
#[inline]
pub fn row_in_fraction(row_id: u64, fraction: f64) -> bool {
    if fraction >= 1.0 {
        return true;
    }
    if fraction <= 0.0 {
        return false;
    }
    let threshold = (fraction * HASH_SPACE) as u64;
    splitmix64_hash(row_id) < threshold
}

/// The full subsample → full-pass ρ schedule.
///
/// Constructed from the corpus size and the number of outer ρ steps; auto-
/// derives the per-step fraction (geometric ramp from a small floor up to the
/// full corpus) with no CLI knobs. The last `full_pass_steps` steps are always
/// honest full passes so the final ρ is computed on the exact corpus.
#[derive(Debug, Clone)]
pub struct RhoCascadeSchedule {
    steps: Vec<RhoStepPlan>,
    total_rows: u64,
}

/// Smallest subsample fraction the ramp starts from. Auto-derived floor: even
/// the coarsest early step sees at least this fraction so the gradient sign is
/// reliable. Not a CLI knob.
const MIN_FRACTION: f64 = 1.0 / 64.0;

/// Number of trailing steps forced to full corpus passes. The honest final
/// descent must be exact, so the last two ρ steps never subsample.
const FULL_PASS_TAIL_STEPS: usize = 2;

impl RhoCascadeSchedule {
    /// Build a schedule for `total_rows` rows over `n_steps` outer ρ steps.
    ///
    /// The fraction ramps geometrically from [`MIN_FRACTION`] (or a higher
    /// floor if the corpus is small enough that the floor would visit fewer
    /// than [`MIN_SUBSAMPLE_ROWS`] rows) up to `1.0`, and the trailing
    /// [`FULL_PASS_TAIL_STEPS`] steps are pinned to full passes. A corpus small
    /// enough that subsampling saves nothing degenerates to all-full-passes.
    pub fn new(total_rows: u64, n_steps: usize) -> Self {
        let n_steps = n_steps.max(1);
        // Floor the starting fraction up if MIN_FRACTION would select too few
        // rows to be a stable gradient estimate.
        let min_rows = MIN_SUBSAMPLE_ROWS.min(total_rows.max(1));
        let floor_fraction = if total_rows == 0 {
            1.0
        } else {
            (min_rows as f64 / total_rows as f64)
                .max(MIN_FRACTION)
                .min(1.0)
        };

        let full_from = n_steps.saturating_sub(FULL_PASS_TAIL_STEPS);
        let mut steps = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let (fraction, is_full_pass) = if step >= full_from {
                (1.0, true)
            } else {
                // Geometric ramp from floor_fraction to 1.0 across the
                // subsample steps. With `full_from` subsample steps, step s
                // gets floor_fraction^(1 - s/(full_from)) — monotone up to 1.0.
                let subsample_steps = full_from.max(1);
                let t = step as f64 / subsample_steps as f64;
                // exp-interpolate in log space between floor and 1.0.
                let log_floor = floor_fraction.ln();
                let frac = (log_floor * (1.0 - t)).exp();
                let frac = frac.clamp(floor_fraction, 1.0);
                (frac, frac >= 1.0)
            };
            let importance_weight = if fraction > 0.0 { 1.0 / fraction } else { 1.0 };
            steps.push(RhoStepPlan {
                step,
                fraction,
                importance_weight,
                is_full_pass,
            });
        }
        Self { steps, total_rows }
    }

    /// The planned steps, in ρ order.
    pub fn steps(&self) -> &[RhoStepPlan] {
        &self.steps
    }

    /// Plan for a specific outer step (clamped to the last step if `step`
    /// runs past the schedule — extra ρ iterations are honest full passes).
    pub fn step_plan(&self, step: usize) -> RhoStepPlan {
        if let Some(plan) = self.steps.get(step) {
            *plan
        } else {
            // Past the planned horizon: full pass at the final step index.
            RhoStepPlan {
                step,
                fraction: 1.0,
                importance_weight: 1.0,
                is_full_pass: true,
            }
        }
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Expected number of rows visited at a given step (for accumulator sizing
    /// / progress reporting). Deterministic for full passes; expectation for
    /// subsample steps.
    pub fn expected_rows(&self, step: usize) -> u64 {
        let plan = self.step_plan(step);
        if plan.is_full_pass {
            self.total_rows
        } else {
            (plan.fraction * self.total_rows as f64).round() as u64
        }
    }
}

/// Below this many rows, subsampling is not worth the bias risk: a step's
/// subsample should never be smaller than this (and a corpus smaller than this
/// is always a full pass). Auto-derived floor.
const MIN_SUBSAMPLE_ROWS: u64 = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_fraction_includes_every_row() {
        for id in 0..1000u64 {
            assert!(row_in_fraction(id, 1.0));
            assert!(row_in_fraction(id, 2.0));
        }
    }

    #[test]
    fn zero_fraction_excludes_every_row() {
        for id in 0..1000u64 {
            assert!(!row_in_fraction(id, 0.0));
            assert!(!row_in_fraction(id, -0.5));
        }
    }

    #[test]
    fn subsample_is_deterministic() {
        // Same id => same decision, every call.
        for id in 0..10_000u64 {
            let a = row_in_fraction(id, 0.25);
            let b = row_in_fraction(id, 0.25);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn subsample_fraction_is_approximately_realized() {
        let n = 200_000u64;
        let frac = 0.1;
        let included = (0..n).filter(|&id| row_in_fraction(id, frac)).count();
        let realized = included as f64 / n as f64;
        // splitmix64 is a good mixer; expect within a few percent of target.
        assert!(
            (realized - frac).abs() < 0.01,
            "realized fraction {realized} too far from {frac}"
        );
    }

    #[test]
    fn importance_weight_unbiases_subsample() {
        // The honesty contract: sum over a uniform subsample, each weighted by
        // 1/fraction, estimates the full-corpus sum. Check on a constant field.
        let n = 100_000u64;
        let frac = 0.2;
        let weight = 1.0 / frac;
        let full_sum = n as f64; // each row contributes 1.0
        let sub_sum: f64 = (0..n)
            .filter(|&id| row_in_fraction(id, frac))
            .map(|_| weight)
            .sum();
        let rel_err = (sub_sum - full_sum).abs() / full_sum;
        assert!(
            rel_err < 0.02,
            "weighted subsample {sub_sum} vs full {full_sum}"
        );
    }

    #[test]
    fn schedule_ends_in_full_passes() {
        let sched = RhoCascadeSchedule::new(10_000_000, 8);
        let steps = sched.steps();
        let last = steps.last().expect("nonempty");
        assert!(last.is_full_pass);
        assert_eq!(last.importance_weight, 1.0);
        // The tail steps are full passes.
        assert!(steps[steps.len() - 1].is_full_pass);
        assert!(steps[steps.len() - 2].is_full_pass);
    }

    #[test]
    fn schedule_fraction_is_monotone_nondecreasing() {
        let sched = RhoCascadeSchedule::new(10_000_000, 8);
        let fracs: Vec<f64> = sched.steps().iter().map(|s| s.fraction).collect();
        for w in fracs.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "fractions not monotone: {fracs:?}");
        }
        // Early step subsamples, weight > 1.
        assert!(sched.steps()[0].fraction < 1.0);
        assert!(sched.steps()[0].importance_weight > 1.0);
    }

    #[test]
    fn step_plan_includes_consistent_with_fraction() {
        let sched = RhoCascadeSchedule::new(10_000_000, 8);
        let plan = sched.step_plan(0);
        for id in 0..1000u64 {
            match plan.includes(id) {
                Some(w) => {
                    assert!((w - plan.importance_weight).abs() < 1e-12);
                    assert!(row_in_fraction(id, plan.fraction) || plan.is_full_pass);
                }
                None => assert!(!row_in_fraction(id, plan.fraction)),
            }
        }
    }

    #[test]
    fn tiny_corpus_is_all_full_passes() {
        let sched = RhoCascadeSchedule::new(100, 5);
        for s in sched.steps() {
            assert!(s.is_full_pass);
            assert_eq!(s.fraction, 1.0);
        }
    }
}
