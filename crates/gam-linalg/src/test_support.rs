//! Low-level numerical test fixtures shared across the workspace.
//!
//! The operator fixtures exercise `gam-linalg`'s design machinery, so they live
//! here — the crate that owns [`LinearOperator`], [`DesignMatrix`], and friends
//! — rather than in a downstream crate. Following the workspace convention for
//! `test_support` modules (and matching the root `gam` crate), this is a plain
//! always-compiled `pub mod`: feature gates and `#[cfg(test)]` module gates are
//! banned here, and a `cfg(test)` module would be invisible to downstream
//! crates' test builds anyway. The contents are `pub`, so they are reachable
//! (no dead-code lint) yet only ever called from `#[cfg(test)]` code.
//!
//! [`fd_checker`] carries the same argument one level down: a central-difference
//! derivative check is `ndarray` in, `ndarray` out and owns no model-layer type,
//! so it belongs to the leaf that owns the dense-array seam. Keeping it here
//! means a crate cross-checking an analytic derivative pulls one leaf dependency
//! it already has instead of the whole model layer.
//!
//! [`paired_holdout_partition`] is likewise model-free deterministic numerical
//! test design. Keeping its row-ranking primitive here lets downstream quality
//! tests share one paired partition without making its invariant test compile
//! the reference bridge and the entire model stack.
//!
//! [`PairedFoldComparison`] closes that design: the partition primitive hands
//! both implementations the SAME folds, and this is the estimator that spends
//! the pairing instead of discarding it. It is `f64` slices in, statistics out,
//! so it belongs beside the partition it consumes rather than in the reference
//! bridge.

pub mod fd_checker;

/// #2738 — serialize every test that MUTATES or OBSERVES faer's process-global
/// parallelism, across every module in the binary.
///
/// `faer::set_global_parallelism` and [`crate::faer_ndarray::FaerSequentialScope`]
/// write one process-wide cell. Two `#[test]`s touching it run on different
/// threads of the same binary, so one test's `Par::rayon(4)` lands inside
/// another's sequential scope and the reader sees a state neither test created
/// — a verdict that depends on which other tests share the process. Holding this
/// lock for the whole body makes the observation belong to the test that made
/// it.
///
/// It lives in `test_support` rather than beside either test module because the
/// participants are in different modules and both must take the SAME lock; a
/// per-module `static` would serialize each module against itself only, which is
/// indistinguishable from no lock at all in exactly the failing case.
pub fn with_global_parallelism_serialized<T>(body: impl FnOnce() -> T) -> T {
    static GLOBAL_PARALLELISM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let serialized = GLOBAL_PARALLELISM_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = body();
    drop(serialized);
    out
}

/// Assert that a central difference of an array-producing function matches the
/// analytical derivative.
///
/// Expands `approx::assert_abs_diff_eq!` at the call site, so the caller needs
/// `approx` in scope; it deliberately does not force that dependency on this
/// crate's non-test build.
///
/// `#[macro_export]` puts this at the crate root regardless of the module it is
/// written in, so the call path stays `gam_linalg::assert_central_difference_array!`.
/// It lives beside [`fd_checker`] in `test_support` rather than in `lib.rs`
/// because it is a TEST assertion — its only callers are `#[cfg(test)]` code —
/// and the #1440 production-FD ban rightly reads a central-difference macro
/// sitting in `lib.rs` as production finite differencing.
#[macro_export]
macro_rules! assert_central_difference_array {
    ($x:expr, $h:expr, |$var:ident| $eval:expr, $analytical:expr, $tol:expr) => {
        let f_plus = {
            let $var = $x + $h;
            $eval
        };
        let f_minus = {
            let $var = $x - $h;
            $eval
        };
        assert_eq!(f_plus.len(), $analytical.len());
        for j in 0..$analytical.len() {
            let fd = (f_plus[j] - f_minus[j]) / (2.0 * $h);
            approx::assert_abs_diff_eq!(fd, $analytical[j], epsilon = $tol);
        }
    };
}

use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use gam_runtime::resource::MatrixMaterializationError;
use ndarray::{Array1, Array2, Axis, s};
use std::ops::Range;
use std::sync::Arc;

/// One exact-cardinality train/test partition for a paired quality comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedHoldout {
    /// Source-row indices used to fit both implementations.
    pub train: Vec<usize>,
    /// Source-row indices scored for both implementations.
    pub test: Vec<usize>,
    /// Full-width external-tool mask: `1.0` for test, `0.0` for train.
    pub mask: Vec<f64>,
}

/// Build a reproducible, exact-cardinality paired holdout partition.
///
/// Rows are ranked by a SplitMix64 score keyed by `split_key`; the lowest
/// `round(n * holdout_fraction)` scores form the test set. Ranking instead of
/// thresholding a pseudo-random value keeps every split the same size. That
/// matters when per-split metrics are averaged: every term then represents the
/// same amount of held-out evidence, and small fixtures cannot accidentally
/// produce a degenerate fold.
///
/// The returned row indices retain source order. A quality test must use the
/// same returned partition for both the implementation under test and its
/// reference tool.
pub fn paired_holdout_partition(n: usize, holdout_fraction: f64, split_key: u64) -> PairedHoldout {
    assert!(n >= 2, "paired holdout needs at least two rows, got {n}");
    assert!(
        holdout_fraction.is_finite() && 0.0 < holdout_fraction && holdout_fraction < 1.0,
        "paired holdout fraction must be finite and strictly between zero and one, got {holdout_fraction}"
    );

    let test_len = (n as f64 * holdout_fraction).round() as usize;
    assert!(
        0 < test_len && test_len < n,
        "paired holdout fraction {holdout_fraction} yields {test_len} test rows for n={n}"
    );

    const GOLDEN_RATIO: u64 = 0x9E3779B97F4A7C15;
    let score = |row: usize| {
        let mut z = (row as u64)
            .wrapping_add(split_key.wrapping_mul(GOLDEN_RATIO))
            .wrapping_add(GOLDEN_RATIO);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };

    let mut ranked: Vec<(u64, usize)> = (0..n).map(|row| (score(row), row)).collect();
    ranked.sort_unstable();

    let mut held_out = vec![false; n];
    for &(_, row) in &ranked[..test_len] {
        held_out[row] = true;
    }

    let train = (0..n).filter(|&row| !held_out[row]).collect();
    let test = (0..n).filter(|&row| held_out[row]).collect();
    let mask = held_out
        .into_iter()
        .map(|is_test| if is_test { 1.0 } else { 0.0 })
        .collect();
    PairedHoldout { train, test, mask }
}

/// One-sided tail probability at which a paired quality panel calls a deficit
/// RESOLVED — the only free parameter of the decision rule below.
///
/// This is the per-panel false-alarm rate, so the suite-wide cost of the rule is
/// `(number of gated panels) * RESOLUTION_TAIL` expected noise-driven failures
/// per full run. #2395's census of the reference-quality dashboard counted ~140
/// `QUALITY_PAIR` comparisons, so demanding fewer than one noise-driven red per
/// whole-suite run bounds the tail by `1/140 ~= 0.0071`. This is the largest
/// conventional tail under that bound: it budgets `140 * 0.005 = 0.7` expected
/// false reds per run.
///
/// It is not a slack knob. It buys resolution: at `K = 10` folds the rule fires
/// on any paired deficit wider than `t_{9,0.995} ~= 3.25` paired standard
/// errors, which on these fixtures is a small-single-digit percentage — far
/// inside the flat `* 1.10` ceiling that was previously the only bar.
pub const RESOLUTION_TAIL: f64 = 0.005;

/// Exact Student-t CDF for integer degrees of freedom.
///
/// Integrating the t density by parts gives a CLOSED FORM for integer `df` — a
/// finite trigonometric sum, no incomplete beta and no new dependency. With
/// `theta = atan(t / sqrt(df))`:
///
/// * even `df`: `P(|T| <= t) = sin(theta) * sum_{j=1..df/2} d_j cos^{2j-2}(theta)`,
///   `d_1 = 1`, `d_j = d_{j-1} * (2j-3) / (2j-2)`
/// * odd `df`: `P(|T| <= t) = (2/pi) * [theta + sin(theta) * sum_{j=1..(df-1)/2}
///   c_j cos^{2j-1}(theta)]`, `c_1 = 1`, `c_j = c_{j-1} * (2j-2) / (2j-1)`
///
/// and `P(T <= t) = (1 + sign(t) * P(|T| <= |t|)) / 2`.
pub fn student_t_cdf(t: f64, df: usize) -> f64 {
    assert!(df >= 1, "Student-t needs at least one degree of freedom");
    assert!(
        t.is_finite(),
        "Student-t CDF needs a finite quantile, got {t}"
    );

    let theta = (t.abs() / (df as f64).sqrt()).atan();
    let (sin_theta, cos_theta) = theta.sin_cos();

    let central = if df.is_multiple_of(2) {
        let mut term = 1.0;
        let mut sum = 1.0;
        for j in 2..=(df / 2) {
            term *= cos_theta * cos_theta * (2 * j - 3) as f64 / (2 * j - 2) as f64;
            sum += term;
        }
        sin_theta * sum
    } else {
        let mut term = cos_theta;
        let mut sum = if df >= 3 { cos_theta } else { 0.0 };
        for j in 2..=((df - 1) / 2) {
            term *= cos_theta * cos_theta * (2 * j - 2) as f64 / (2 * j - 1) as f64;
            sum += term;
        }
        std::f64::consts::FRAC_2_PI * (theta + sin_theta * sum)
    };

    let central = central.clamp(0.0, 1.0);
    if t >= 0.0 {
        0.5 * (1.0 + central)
    } else {
        0.5 * (1.0 - central)
    }
}

/// The `t` with `P(T > t) = tail` for integer `df`, by bisecting [`student_t_cdf`].
///
/// The CDF is strictly increasing and exact, so bisection converges to the f64
/// neighbourhood of the true quantile; the bracket starts wide enough to cover
/// the heavy `df = 1` tail.
pub fn student_t_upper_quantile(df: usize, tail: f64) -> f64 {
    assert!(
        tail.is_finite() && 0.0 < tail && tail < 0.5,
        "one-sided tail must be finite and in (0, 0.5), got {tail}"
    );
    let target = 1.0 - tail;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while student_t_cdf(hi, df) < target {
        hi *= 2.0;
        assert!(
            hi.is_finite(),
            "no finite Student-t quantile for df={df} tail={tail}"
        );
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if student_t_cdf(mid, df) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A gam-vs-reference comparison scored on the SAME folds, spending the pairing.
///
/// K-fold averaging alone does not fix a near-tie gate. Averaging the two arms
/// SEPARATELY and comparing the two means throws away the fact that fold `k` was
/// the same fold for both tools: the fold-to-fold swing (which split of a small
/// fixture you drew) is common to both arms and cancels in the per-fold
/// difference. #2395's near-ties are exactly the regime where that common swing
/// dominates the between-tool gap, which is why single-split verdicts flipped
/// sign. The paired difference removes it, and its own spread — not a flat
/// percentage — is the correct yardstick for the remaining gap.
///
/// The per-fold effect is sign-normalized so that **negative means gam is
/// better**, matching `bench/aggregate_quality_gate_1561.py`:
///
/// * `lower_is_better`: `e_k = ln(gam_k / reference_k)`
/// * higher-is-better: `e_k = ln(reference_k / gam_k)`
///
/// Both arms must be strictly positive on every fold; a signed metric (R^2,
/// ELPD) has to be compared on a positive error scale instead, so that the
/// suite keeps ONE definition of "effect".
#[derive(Clone, Debug)]
pub struct PairedFoldComparison {
    /// Number of folds/seeds, identical for both arms by construction.
    pub folds: usize,
    /// Whether a smaller metric value is better.
    pub lower_is_better: bool,
    /// Per-fold sign-normalized log-ratio; negative means gam won that fold.
    pub effect: Vec<f64>,
    /// Arithmetic mean of gam's per-fold metric (the value the gate reports).
    pub gam_mean: f64,
    /// Arithmetic mean of the reference's per-fold metric.
    pub reference_mean: f64,
    /// Fold-to-fold sample SD of gam's metric, in the metric's own units.
    pub gam_fold_sd: f64,
    /// Fold-to-fold sample SD of the reference's metric, in the metric's units.
    pub reference_fold_sd: f64,
    /// Mean paired effect. Negative means gam is ahead on average.
    pub effect_mean: f64,
    /// Sample SD of the paired effects — the fold-to-fold spread that survives
    /// pairing, and the yardstick the decision rule measures the gap against.
    pub effect_sd: f64,
    /// Standard error of [`Self::effect_mean`]: `effect_sd / sqrt(folds)`.
    pub effect_sem: f64,
    /// What the standard error WOULD have been comparing two independently
    /// averaged arms: `sqrt((var(ln gam) + var(ln reference)) / folds)`. The
    /// ratio to [`Self::effect_sem`] is the power the pairing recovers.
    pub unpaired_sem: f64,
    /// Folds on which gam strictly beat the reference.
    pub gam_wins: usize,
}

impl PairedFoldComparison {
    /// Build the comparison from two per-fold score vectors measured on the same
    /// folds, in the same fold order.
    pub fn new(gam: &[f64], reference: &[f64], lower_is_better: bool) -> Self {
        assert_eq!(
            gam.len(),
            reference.len(),
            "paired fold comparison needs one reference score per gam score \
             (gam has {}, reference has {}); the two arms must score the SAME folds",
            gam.len(),
            reference.len()
        );
        let folds = gam.len();
        assert!(
            folds >= 2,
            "a paired fold comparison needs at least two folds to have a spread, got {folds}"
        );
        for (k, (&g, &r)) in gam.iter().zip(reference).enumerate() {
            assert!(
                g.is_finite() && g > 0.0 && r.is_finite() && r > 0.0,
                "fold {k} scores must be finite and strictly positive to form a \
                 log-ratio effect, got gam={g} reference={r}"
            );
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let sample_sd = |v: &[f64], m: f64| {
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };

        let log_gam: Vec<f64> = gam.iter().map(|g| g.ln()).collect();
        let log_ref: Vec<f64> = reference.iter().map(|r| r.ln()).collect();
        let sign = if lower_is_better { 1.0 } else { -1.0 };
        let effect: Vec<f64> = log_gam
            .iter()
            .zip(&log_ref)
            .map(|(g, r)| sign * (g - r))
            .collect();

        let gam_mean = mean(gam);
        let reference_mean = mean(reference);
        let effect_mean = mean(&effect);
        let effect_sd = sample_sd(&effect, effect_mean);
        let root_k = (folds as f64).sqrt();
        let log_gam_sd = sample_sd(&log_gam, mean(&log_gam));
        let log_ref_sd = sample_sd(&log_ref, mean(&log_ref));
        let gam_wins = effect.iter().filter(|&&e| e < 0.0).count();

        Self {
            folds,
            lower_is_better,
            effect,
            gam_mean,
            reference_mean,
            gam_fold_sd: sample_sd(gam, gam_mean),
            reference_fold_sd: sample_sd(reference, reference_mean),
            effect_mean,
            effect_sd,
            effect_sem: effect_sd / root_k,
            unpaired_sem: (log_gam_sd * log_gam_sd + log_ref_sd * log_ref_sd).sqrt() / root_k,
            gam_wins,
        }
    }

    /// One-sided Student-t critical value at [`RESOLUTION_TAIL`] with
    /// `folds - 1` degrees of freedom — the multiplier on the paired SEM that
    /// separates "resolved" from "could be fold noise".
    pub fn critical_t(&self) -> f64 {
        student_t_upper_quantile(self.folds - 1, RESOLUTION_TAIL)
    }

    /// Standardized paired effect size `d_z = effect_mean / effect_sd`: how many
    /// paired fold-SDs separate the two tools. Zero when the two arms are
    /// bit-identical on every fold.
    pub fn effect_size(&self) -> f64 {
        if self.effect_sd > 0.0 {
            self.effect_mean / self.effect_sd
        } else {
            0.0
        }
    }

    /// Standard errors saved by pairing: `unpaired_sem / effect_sem`. Values
    /// above one quantify the common fold-draw swing that pairing cancels.
    pub fn pairing_gain(&self) -> f64 {
        if self.effect_sem > 0.0 {
            self.unpaired_sem / self.effect_sem
        } else {
            1.0
        }
    }

    /// One-sided lower confidence bound on the true mean effect. Positive means
    /// gam is worse by an amount fold noise cannot explain.
    pub fn deficit_lower_bound(&self) -> f64 {
        self.effect_mean - self.critical_t() * self.effect_sem
    }

    /// One-sided upper confidence bound on the true mean effect. Negative means
    /// gam is better by an amount fold noise cannot explain.
    pub fn advantage_upper_bound(&self) -> f64 {
        self.effect_mean + self.critical_t() * self.effect_sem
    }

    /// gam is behind by more than the paired fold noise can explain.
    pub fn gam_resolved_worse(&self) -> bool {
        self.deficit_lower_bound() > 0.0
    }

    /// gam is ahead by more than the paired fold noise can explain.
    pub fn gam_resolved_better(&self) -> bool {
        self.advantage_upper_bound() < 0.0
    }

    /// One-word verdict for logs and the aggregator.
    pub fn verdict(&self) -> &'static str {
        if self.gam_resolved_worse() {
            "gam_resolved_worse"
        } else if self.gam_resolved_better() {
            "gam_resolved_better"
        } else {
            "unresolved_tie"
        }
    }

    /// One human-readable diagnostic line carrying the effect size and both
    /// spreads, so a run log shows WHY a near-tie is or is not decidable.
    pub fn report(&self, label: &str) -> String {
        format!(
            "[PAIRED_FOLDS] {label} folds={} verdict={} gam_mean={:.6e} reference_mean={:.6e} \
             gam_fold_sd={:.3e} reference_fold_sd={:.3e} effect_mean={:+.5} effect_sd={:.5} \
             effect_sem={:.5} effect_size={:+.3} gam_wins={}/{} deficit_lower_bound={:+.5} \
             advantage_upper_bound={:+.5} critical_t={:.4} pairing_gain={:.2}x",
            self.folds,
            self.verdict(),
            self.gam_mean,
            self.reference_mean,
            self.gam_fold_sd,
            self.reference_fold_sd,
            self.effect_mean,
            self.effect_sd,
            self.effect_sem,
            self.effect_size(),
            self.gam_wins,
            self.folds,
            self.deficit_lower_bound(),
            self.advantage_upper_bound(),
            self.critical_t(),
            self.pairing_gain(),
        )
    }
}

/// The shared match-or-beat decision rule for every paired quality panel.
///
/// It is the CONJUNCTION of two clauses, so it is strictly stronger than the
/// flat ceiling it replaces and can never be more permissive:
///
/// 1. **Resolved-deficit clause (new).** gam must not be behind by more than the
///    paired fold noise can explain: [`PairedFoldComparison::gam_resolved_worse`]
///    must be false. This is what gives the gate power — it fires on gaps orders
///    of magnitude below `ceiling_ratio` once they are consistent across folds,
///    and it cannot fire on the sign-flipping single-split noise of #2395.
/// 2. **Hard ceiling (unchanged).** The averaged metric must still sit inside
///    `ceiling_ratio`, whatever the spread. No amount of fold noise buys a pass
///    on a gap this wide.
///
/// `ceiling_ratio > 1.0` is the pre-existing per-panel tolerance and is never
/// relaxed here; the caller passes exactly the ratio its old assertion used.
pub fn assert_paired_match_or_beat(label: &str, cmp: &PairedFoldComparison, ceiling_ratio: f64) {
    assert!(
        ceiling_ratio > 1.0,
        "{label}: match-or-beat ceiling must exceed one, got {ceiling_ratio}"
    );

    assert!(
        !cmp.gam_resolved_worse(),
        "{label}: gam is RESOLVED worse than the reference across folds — the paired \
         deficit's one-sided lower bound is {:+.5} > 0 (mean effect {:+.5} = {:+.3}%, \
         paired SEM {:.5}, effect size {:+.3}, gam won {}/{} folds). This is not split \
         noise: it survives the fold-to-fold spread.\n{}",
        cmp.deficit_lower_bound(),
        cmp.effect_mean,
        100.0 * cmp.effect_mean.exp_m1(),
        cmp.effect_sem,
        cmp.effect_size(),
        cmp.gam_wins,
        cmp.folds,
        cmp.report(label),
    );

    let within_ceiling = if cmp.lower_is_better {
        cmp.gam_mean <= cmp.reference_mean * ceiling_ratio
    } else {
        cmp.gam_mean * ceiling_ratio >= cmp.reference_mean
    };
    assert!(
        within_ceiling,
        "{label}: gam's fold-averaged metric {:.6e} misses the reference {:.6e} by more \
         than the {:.0}% ceiling.\n{}",
        cmp.gam_mean,
        cmp.reference_mean,
        100.0 * (ceiling_ratio - 1.0),
        cmp.report(label),
    );
}

/// A dense-backed [`LinearOperator`] that refuses to materialize itself.
///
/// It services every operator-aware code path (`apply`, `apply_transpose`,
/// `row_chunk_into`, `diag_xtw_x`) but panics from [`to_dense`](DenseDesignOperator::to_dense).
/// Wrapping a design in this fixture turns "a code path densified when it should
/// have stayed lazy" — the regression we guard against — into a hard test
/// failure instead of a silent, slow correctness-preserving fallback.
#[derive(Clone)]
struct NoDensifyOperator {
    dense: Array2<f64>,
}

impl LinearOperator for NoDensifyOperator {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.dot(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.t().dot(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "NoDensifyOperator weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows()
            ));
        }
        let weighted = &self.dense * &weights.view().insert_axis(Axis(1));
        Ok(self.dense.t().dot(&weighted))
    }
}

impl DenseDesignOperator for NoDensifyOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        out.assign(&self.dense.slice(s![rows, ..]));
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        // `NoDensifyOperator` is a test fixture asserting that
        // operator-aware code paths never densify.
        // SAFETY: a call here means a code path under test bypassed
        // `row_chunk_into` and tried to materialize — the regression
        // this fixture is designed to catch.
        panic!("NoDensifyOperator must stay lazy")
    }
}

/// Build an operator-backed [`DesignMatrix`] from a dense array that will panic
/// if any consumer tries to densify it. See `NoDensifyOperator`.
pub fn no_densify_design(dense: Array2<f64>) -> DesignMatrix {
    DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyOperator {
        dense,
    })))
}

#[cfg(test)]
mod tests {
    use super::{PairedFoldComparison, RESOLUTION_TAIL, assert_paired_match_or_beat, paired_holdout_partition, student_t_cdf, student_t_upper_quantile};

    #[test]
    fn paired_holdout_is_exact_reproducible_and_partitioned() {
        let first = paired_holdout_partition(221, 0.20, 17);
        let replay = paired_holdout_partition(221, 0.20, 17);
        let other = paired_holdout_partition(221, 0.20, 18);

        assert_eq!(first, replay);
        assert_ne!(first.test, other.test);
        assert_eq!(first.test.len(), 44);
        assert_eq!(first.train.len(), 177);
        assert_eq!(first.mask.len(), 221);

        let mut memberships = vec![0usize; 221];
        for &row in &first.train {
            memberships[row] += 1;
            assert_eq!(first.mask[row], 0.0);
        }
        for &row in &first.test {
            memberships[row] += 1;
            assert_eq!(first.mask[row], 1.0);
        }
        assert!(memberships.into_iter().all(|count| count == 1));
    }

    /// A quality verdict must be a function of the DATA, never of the path the
    /// run took to reach it. `paired_holdout_partition` therefore has to be a
    /// pure function of `(n, holdout_fraction, split_key)`: no process-global
    /// counter, no RNG state threaded between calls, no dependence on how many
    /// partitions were drawn first. Interleaving unrelated draws of every shape
    /// — different `n`, different fraction, different key — must not perturb a
    /// repeat of the original, so a failure reproduces from its three arguments
    /// alone and cannot move with test-execution order.
    #[test]
    fn paired_holdout_ignores_call_order_and_carries_no_hidden_state() {
        let reference = paired_holdout_partition(221, 0.20, 17);
        for (n, fraction, key) in [
            (240_usize, 0.25_f64, 3_u64),
            (1000, 0.20, 17),
            (221, 0.35, 17),
            (221, 0.20, 18),
            (17, 0.5, 221),
        ] {
            let intervening = paired_holdout_partition(n, fraction, key);
            assert_eq!(intervening.mask.len(), n);
            assert_eq!(
                paired_holdout_partition(221, 0.20, 17),
                reference,
                "an intervening ({n}, {fraction}, {key}) draw perturbed a repeat \
                 of (221, 0.20, 17); the partition is carrying hidden state"
            );
        }
    }

    /// The closed-form integer-df Student-t CDF must reproduce published
    /// quantiles. `df = 9` is what every K=10 quality panel uses; `df = 4` is
    /// kept both as the even-branch counterpart and because it prices what a
    /// smaller panel would cost (`4.604` vs `3.250` on the same spread). The
    /// Cauchy and `df = 2` cases have independent elementary closed forms, so
    /// they check the series against something other than a table.
    #[test]
    fn student_t_closed_form_matches_published_quantiles() {
        // Cauchy: F(t) = 1/2 + atan(t)/pi.
        for &t in &[-3.0_f64, -0.5, 0.0, 0.75, 4.0] {
            let closed = 0.5 + t.atan() / std::f64::consts::PI;
            assert!(
                (student_t_cdf(t, 1) - closed).abs() < 1e-14,
                "df=1 at t={t}: {} vs {closed}",
                student_t_cdf(t, 1)
            );
        }
        // df=2: F(t) = 1/2 + t / (2 sqrt(t^2 + 2)).
        for &t in &[-2.5_f64, -0.25, 0.0, 1.5, 6.0] {
            let closed = 0.5 + t / (2.0 * (t * t + 2.0).sqrt());
            assert!(
                (student_t_cdf(t, 2) - closed).abs() < 1e-14,
                "df=2 at t={t}: {} vs {closed}",
                student_t_cdf(t, 2)
            );
        }
        // Symmetry and monotonicity on both parities.
        for df in 1..=12 {
            assert!(
                (student_t_cdf(0.0, df) - 0.5).abs() < 1e-15,
                "median df={df}"
            );
            assert!(
                (student_t_cdf(1.7, df) + student_t_cdf(-1.7, df) - 1.0).abs() < 1e-14,
                "symmetry df={df}"
            );
            assert!(
                student_t_cdf(0.9, df) < student_t_cdf(1.1, df),
                "monotone df={df}"
            );
        }
        // Published one-sided quantiles (Student-t tables).
        for &(df, tail, want) in &[
            (4_usize, 0.05_f64, 2.131_847_f64),
            (9, 0.05, 1.833_113),
            (4, 0.005, 4.604_095),
            (9, 0.005, 3.249_836),
            (19, 0.025, 2.093_024),
        ] {
            let got = student_t_upper_quantile(df, tail);
            assert!(
                (got - want).abs() < 1e-5,
                "t_{{{df},{}}} = {got} but the table says {want}",
                1.0 - tail
            );
        }
    }

    /// The estimator must spend the pairing: a fold-draw swing common to both
    /// arms has to cancel, so a small but perfectly consistent gap is RESOLVED
    /// while a large but sign-flipping gap is not. This is the #2395 failure
    /// mode reduced to arithmetic.
    #[test]
    fn paired_comparison_resolves_consistent_gaps_and_refuses_split_noise() {
        // Common fold-draw swing of +-40%, plus a consistent 3% gam deficit.
        let swing = [1.0, 1.4, 0.7, 1.2, 0.85, 1.3, 0.75, 1.1, 0.95, 1.25];
        let reference: Vec<f64> = swing.iter().map(|s| 0.5 * s).collect();
        let gam: Vec<f64> = reference.iter().map(|r| r * 1.03).collect();
        let consistent = PairedFoldComparison::new(&gam, &reference, true);

        assert_eq!(consistent.folds, 10);
        assert_eq!(consistent.gam_wins, 0);
        assert!(
            (consistent.effect_mean - 1.03_f64.ln()).abs() < 1e-12,
            "a pure ratio offset must give exactly its log-ratio, got {}",
            consistent.effect_mean
        );
        assert!(
            consistent.effect_sd < 1e-12,
            "the common swing must cancel in the paired difference, got sd={}",
            consistent.effect_sd
        );
        assert!(
            consistent.effect_sem * 1e6 < consistent.unpaired_sem,
            "pairing must recover the whole common swing: paired sem={} unpaired sem={}",
            consistent.effect_sem,
            consistent.unpaired_sem
        );
        assert!(
            consistent.gam_resolved_worse(),
            "{}",
            consistent.report("consistent")
        );
        assert!(!consistent.gam_resolved_better());
        assert_eq!(consistent.verdict(), "gam_resolved_worse");
        // The unpaired view of the SAME data cannot see it: comparing two
        // separately-averaged arms, the 3% gap sits well inside the
        // two-independent-means confidence bound and stays UNRESOLVED.
        assert!(
            consistent.effect_mean.abs() < consistent.critical_t() * consistent.unpaired_sem,
            "the two-independent-means view should be blind here: gap={} bound={}",
            consistent.effect_mean,
            consistent.critical_t() * consistent.unpaired_sem
        );

        // Same average gap, but delivered as sign-flipping split noise.
        let noisy_gam: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(k, r)| r * if k % 2 == 0 { 1.30 } else { 0.80 })
            .collect();
        let noisy = PairedFoldComparison::new(&noisy_gam, &reference, true);
        assert_eq!(noisy.gam_wins, 5);
        assert!(
            !noisy.gam_resolved_worse() && !noisy.gam_resolved_better(),
            "sign-flipping folds must stay undecided: {}",
            noisy.report("noisy")
        );
        assert_eq!(noisy.verdict(), "unresolved_tie");

        // Sign normalization: on a higher-is-better metric the same numbers
        // must flip which tool is ahead.
        let higher = PairedFoldComparison::new(&gam, &reference, false);
        assert!((higher.effect_mean + consistent.effect_mean).abs() < 1e-15);
        assert!(higher.gam_resolved_better(), "{}", higher.report("higher"));
        assert_eq!(higher.gam_wins, 10);
    }

    /// The per-panel false-alarm budget must keep the whole-suite expectation
    /// under one noise-driven red, and a genuinely-ahead panel must pass both
    /// clauses of the shared rule.
    #[test]
    fn paired_match_or_beat_passes_a_panel_gam_actually_wins() {
        assert!(
            RESOLUTION_TAIL < 1.0 / 140.0,
            "the per-panel tail must budget under one false red across the suite's \
             ~140 QUALITY_PAIR panels, got {RESOLUTION_TAIL}"
        );

        let reference = ceiling_fixture_reference();
        let better: Vec<f64> = reference.iter().map(|r| r * 0.95).collect();
        let cmp = PairedFoldComparison::new(&better, &reference, true);
        assert_paired_match_or_beat("ahead", &cmp, 1.10);
        assert!(cmp.gam_resolved_better(), "{}", cmp.report("ahead"));
        assert_eq!(cmp.gam_wins, 10);
    }

    /// The sharp clause: a 3% deficit sits comfortably inside the 10% ceiling
    /// the old bar allowed, but it is consistent across folds, so the paired
    /// rule resolves it. This is the power the gate gains.
    #[test]
    #[should_panic(expected = "RESOLVED worse")]
    fn paired_rule_catches_a_consistent_deficit_the_flat_ceiling_allows() {
        let reference = ceiling_fixture_reference();
        let gam: Vec<f64> = reference.iter().map(|r| r * 1.03).collect();
        let cmp = PairedFoldComparison::new(&gam, &reference, true);
        assert!(
            cmp.gam_mean <= cmp.reference_mean * 1.10,
            "the flat ceiling this replaces must pass this panel"
        );
        assert_paired_match_or_beat("consistent-3pct", &cmp, 1.10);
    }

    /// The blunt clause is untouched: a gap too noisy to resolve still has to
    /// respect the pre-existing ceiling, so the rule is never more permissive.
    #[test]
    #[should_panic(expected = "ceiling")]
    fn paired_rule_keeps_the_hard_ceiling_for_unresolvable_gaps() {
        let reference = ceiling_fixture_reference();
        let wild: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(k, r)| r * if k % 2 == 0 { 3.0 } else { 0.5 })
            .collect();
        let cmp = PairedFoldComparison::new(&wild, &reference, true);
        assert!(
            !cmp.gam_resolved_worse(),
            "this fixture must be unresolvable so the ceiling is what fires: {}",
            cmp.report("wild")
        );
        assert_paired_match_or_beat("wild", &cmp, 1.10);
    }

    /// Ten irregular reference fold scores shared by the decision-rule tests.
    fn ceiling_fixture_reference() -> Vec<f64> {
        vec![0.5, 0.7, 0.35, 0.6, 0.42, 0.65, 0.38, 0.55, 0.48, 0.62]
    }
}
