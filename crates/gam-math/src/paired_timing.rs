//! One way to measure "does A beat B", for every speed gate in this workspace.
//!
//! Fifteen separate timing harnesses across ten files were doing this in three
//! different ways, and the differences decide whether a gate can tell a
//! regression from a busy machine (issue #932, and #2470 for the duplication).
//! Two of the fifteen interleaved the arms; thirteen did not.
//!
//! # Why the majority pattern cannot resolve what it asserts
//!
//! The `best_ns` family — thirteen copies — times each arm in a **separate
//! call**: A is measured to completion (five rounds, minimum taken), and only
//! then is B. Two things go wrong at once.
//!
//! * **The arms occupy different wall-clock windows.** Anything that drifts
//!   between them — a neighbour job starting, a frequency ramp, cache or
//!   branch-predictor state warmed by the first arm, first-touch page faults
//!   amortised by the first arm — lands entirely in the ratio. Taking a minimum
//!   rejects a transient *spike*; it does nothing about a systematic offset.
//! * **A minimum is the order statistic most exposed to exactly that.** It is
//!   the single most favourable draw for each arm, so a small constant advantage
//!   to whichever arm ran second survives the minimum intact rather than
//!   averaging out. And pairing two independently-minimised blocks throws away
//!   the pairing that would have cancelled the drift in the first place.
//!
//! The consequence is not hypothetical. On the same tree, one such gate PASSED
//! on a quiet node (whole suite 5.1 s) and FAILED at 1.62x against a 1.5x bar on
//! a loaded one (same suite 219.1 s); another picked a **different loser on
//! consecutive runs** of one tree, 9% then 3%. Both were then guarded off in
//! debug builds, which hid the symptom without touching the cause: the harness
//! cannot resolve a margin of a few percent, and most of these gates assert
//! margins of a few percent.
//!
//! # What this does instead
//!
//! * **Interleave per repetition, not per block.** Each repetition times A and B
//!   adjacent in time, so drift slower than one repetition is common to both
//!   sides of that repetition's ratio and divides out.
//! * **Randomise the order within each repetition.** If a first-versus-second
//!   advantage exists at all, randomisation makes it cancel in expectation
//!   instead of accruing to a fixed arm — and [`PairedTiming::first_position_bias`]
//!   reports the residual so it is measured rather than assumed away.
//! * **Report the distribution of PAIRED ratios**, not a ratio of aggregate
//!   extrema. The per-repetition ratio is the quantity the claim is about; a
//!   median of paired ratios is robust, and the spread of that distribution is
//!   the gate's own resolution — which is the number you need in order to know
//!   whether a 3% claim is assertable at all.
//! * **Feed each iteration from the previous result.** A dependence chain
//!   through `checksum` is what stops the optimizer hoisting or vectorising
//!   across iterations; `black_box` alone permits both, and one of the replaced
//!   harnesses relied on `black_box` with no data dependence.
//!
//! # `wins_fraction` is evidence, not a gate
//!
//! Lead a report with it: `wins = 1.00` over fifteen repetitions is a
//! distribution-free sign test at `2^-15`, it does not depend on the resolution
//! estimate, and that matters most exactly when the resolution estimate is the
//! thing under suspicion. It settled a multinomial cell whose margin was only
//! **1.6x** its own resolution — a comparison of ratio against resolution
//! declined to certify that cell.
//!
//! **Do not put it in a bar.** It is a within-run confidence statement *at that
//! host's noise level*, so it degrades in the opposite direction from the
//! quantity it would be guarding. Measured on one 1.6% effect: `wins = 0.00` on
//! a quiet node, `0.27 / 0.40 / 0.27` on a node ~30x noisier, and three runs of
//! **identical code** giving `0.67 / 0.87 / 1.00` — while `median_ratio` stayed
//! inside a 0.8% band. ANDed into a gate it can only manufacture failures on a
//! busy runner. **Gate on `median_ratio`; report `wins` and `ratio_resolution`.**
//!
//! # A derived margin is sometimes zero
//!
//! "Derive the margin from the resolution" **cannot mean "add a margin."**
//! Sometimes the resolution says none is warranted and the derived answer is the
//! bar already there. A zero-margin bar that looked like a coin flip was in fact
//! guarding a real regression at many times its own resolution, and the obvious
//! 5% tolerance would have passed that regression silently.
//!
//! The converse case is just as sharp. One gate's cell sits ~1.5% below its
//! opponent across six measurements on three nodes, with two candidate
//! mechanisms measured and refuted; the estimator it replaced called that cell a
//! comfortable pass at `1.043911`. **Fixing the estimator did not make the bar
//! assertable — it made clear the old numbers were not measuring the quantity at
//! all.** Whether such a cell keeps a strict bar is a contract decision, and it
//! must be taken explicitly with a stated reason, never by widening a bar in the
//! commit that measured it.
//!
//! # The arm must be large relative to one closure call
//!
//! This harness costs a closure call plus a `black_box` per iteration, and it
//! calls the arm through a `&mut F`. That cost lands in **both** arms, so it
//! cannot manufacture a winner on its own — with an equal per-call overhead
//! `c`, a true ratio `b / a` is measured as `(b + c) / (a + c)`, which is
//! monotone toward 1 and **never crosses it**.
//!
//! What it can do is let a *difference* in that overhead decide a small margin.
//! The two arms are distinct closures wrapping distinct callees, so they need
//! not inline identically, and the residual asymmetry is a fixed number of
//! nanoseconds rather than a fraction of the arm.
//!
//! Measured: an SLS value/gradient/Hessian gate whose arm was **one row**
//! (~43 ns) read `42.77 / 44.17 ns` under the old min-of-N harness and
//! `90.25 / 87.50 ns` here — both arms roughly doubled, and the verdict changed
//! sign. Solving `(44.17 + c) / (42.77 + c) = 0.9668` needs `c = -85 ns`, so a
//! symmetric overhead cannot explain it; a ~4 ns asymmetry between the two
//! closures can, because the quantity under test was only 1.4 ns.
//!
//! Batching that same gate to 64 rows per call (arm ~2685 ns, overhead under
//! 2%) settles it in the opposite direction and unanimously:
//! `median_ratio = 1.045250, wins = 1.00, resolution = 0.0092` — generated is
//! 4.5% faster, a margin 4.9x its own resolution. The one-row reading was
//! measuring the harness.
//!
//! **So make one arm call do a batch.** Every other gate migrated to this
//! harness already did without anyone choosing it — a 512-row pass, a full
//! Fisher sweep, a bundle — which is why they were unaffected. A single-row
//! arm is the case that needs an explicit inner loop, sized so the per-call
//! cost is under ~1% of the arm.
//!
//! [`PairedTiming::summary`] prints the per-arm `ns/iter` precisely so this is
//! checkable: if those numbers are of the same order as a function call, the
//! ratio is not measuring what it claims to.
//!
//! # Why not measure the arms separately and normalise afterwards
//!
//! Because it does not work, and it fails in **both** directions. `iperf2`
//! measured this directly on `gam-solve::inner_fit_core_scaling`, a gate whose
//! two arms genuinely cannot be interleaved — one fans out over the whole Rayon
//! pool and the other is held serial by a guard, so external load hurts them
//! unequally. They divided the ratio by the parallel headroom the machine was
//! delivering at that moment, measured on an embarrassingly parallel kernel:
//!
//! * **Normaliser sampled once.** Headroom on four saturated cores bounced
//!   `2.36 / 1.28 / 2.81` across three consecutive repetitions. At the `1.28`
//!   sample a genuinely serial solve scores `1.0 / 1.28 = 0.78` and **passes** a
//!   `0.5` bar — a false green in which the gate certifies the exact defect it
//!   exists to catch.
//! * **Normaliser as max over five repetitions** (the right estimator for a
//!   capability, since interference only pushes an observed speedup down). Fixes
//!   the false green, and then loaded runs score `0.44` and `0.52` against the
//!   same `0.5` bar — red on working code.
//!
//! The underlying reason generalises past that one gate:
//!
//! > **A ratio whose two arms are measured at different times, on a machine
//! > whose load moves on that timescale, cannot be normalised after the fact.**
//! > Interleaving per repetition is not tidiness — it is what makes the arms
//! > share machine state instead of sampling it twice.
//!
//! The same lane's control is the cleanest demonstration that the *measurement*
//! rather than the *code* is what breaks: on one node, same four cores, back to
//! back, the identical solve scored `2.91` / `3.43` idle and `1.94` under four
//! spinners — straddling its own bar with nothing about the solver changed. A
//! width sweep at 2/4/8/16/32 cores on dedicated allocations tracked the pool
//! width at every width.
//!
//! # When the arms cannot be interleaved at all
//!
//! Some comparisons are between configurations that *want different machines* —
//! different core counts, different memory pressure — and no amount of
//! interleaving makes them share state. For those, take the confound away from
//! the measurement instead of modelling it: `.config/nextest.toml` supports
//!
//! ```toml
//! [[profile.default.overrides]]
//! threads-required = 'num-test-threads'
//! ```
//!
//! which reserves every runner slot so the test runs alone. It is already in use
//! in this repository for exactly this reason. The general rule, which is worth
//! more than the mechanism: **before building something to cancel an
//! environmental confound, check whether the runner can remove the confound
//! instead.**
//!
//! # Using it as a gate
//!
//! Open a [`SpeedGate`] (release profile only — the test decides), record one
//! paired cell per contract with [`SpeedGate::faster`] or
//! [`SpeedGate::not_slower`], and [`SpeedGate::finish`]. The gate prints
//! [`PairedTiming::summary`] for every cell whatever the outcome and asserts on
//! [`PairedTiming::median_ratio`] alone; `wins_fraction` and
//! `ratio_resolution` travel on the same line as evidence (see above for why
//! `wins` must not be a bar). Arms of a few tens of nanoseconds go through
//! [`batched`], so the harness's own per-call cost is not what is measured.
//!
//! **Lead a report with `wins_fraction`, not the ratio.** It is the statistic that
//! survives someone disbelieving the rest of the output. `wins == 1.0` over `n`
//! repetitions is a sign test at `2⁻ⁿ` — 15 repetitions is `≈3e-5` — and it is
//! **distribution-free**: it does not depend on [`PairedTiming::ratio_resolution`]
//! being correctly characterised, which is the one number a skeptic can
//! reasonably question. The ratio says *how much*; `wins` says *whether*. When
//! the first real migration onto this harness reported `median_ratio=0.938934`
//! with `wins=0.00`, it was the `wins` that settled a question two lanes had
//! been arguing from opposite directions.
//!
//! # The design lesson, for the next gate
//!
//! [`PairedTiming::first_position_bias`] is here because of a specific failure:
//! a fixed-order harness cannot separate a real 6% margin from a 6%
//! first-versus-second offset, since **both** produce a stable ratio with noisy
//! absolutes. The pre-existing answer was to run the whole gate a second time
//! with the arms swapped and see whether the verdict flipped — which works, but
//! only ever yields yes/no, costs a full second measurement, and has to be
//! redone by hand every time anyone doubts it.
//!
//! Randomising the order and reporting the residual **apportions** the confound
//! instead: on the measurement above, ordering contributed 0.0001 and the code
//! contributed 0.061. Generalising:
//!
//! > **Report a confound as a measured field rather than eliminating it by
//! > argument.** An argument that a confound was controlled has to be re-made,
//! > and re-believed, by every later reader. A field in the output is checked
//! > once and then simply read.
//!
//! That is the property to copy when building the next gate here, more than any
//! particular statistic in this module.

use std::hint::black_box;
use std::time::Instant;

/// Paired per-repetition timings for two implementations of one computation.
///
/// `a_ns[i]` and `b_ns[i]` were measured adjacent in time within repetition `i`,
/// in an order chosen by the repetition's coin flip. `ratios[i]` is
/// `b_ns[i] / a_ns[i]`, so a value **above 1 means A is faster** — the same
/// orientation as the `hand_over_production` token these gates already print.
#[derive(Clone, Debug)]
pub struct PairedTiming {
    /// Nanoseconds per iteration for arm A, one entry per repetition.
    pub a_ns: Vec<f64>,
    /// Nanoseconds per iteration for arm B, one entry per repetition.
    pub b_ns: Vec<f64>,
    /// `b_ns[i] / a_ns[i]`, one entry per repetition. Above 1 ⇒ A faster.
    pub ratios: Vec<f64>,
    /// `true` when arm A was timed first in that repetition.
    pub a_went_first: Vec<bool>,
}

impl PairedTiming {

}

/// Whether a wall-clock ratio measured in this build is about the SHIPPED
/// codegen.
///
/// It is not the optimisation level: `[profile.test]` already carries
/// `opt-level = 2`. It is codegen LAYOUT. `[profile.test.package.gam-models]`
/// sets `codegen-units = 16` and the test profile carries no LTO, while
/// `[profile.release]` is `codegen-units = 1` plus thin-LTO, and the whole
/// margin of a compiled-vs-hand row kernel can be cross-CGU inlining. A ratio
/// taken in the test profile therefore measures a different program than the
/// one that ships, and a debug build measures fixed per-call overhead and
/// nothing else. Every speed gate in this workspace opens only there.
///
/// That decision is made by the TEST that opens the gate, never by this
/// module: test code may query its own build configuration, library code may
/// not (`build.rs` bans `cfg!(debug_assertions)` outside test modules, because
/// a library branch that only runs in one build configuration silently means
/// something else in the other). A gate opened in the dev lane would assert
/// about the wrong program, so the test returns before opening it:
///
/// ```text
/// if cfg!(debug_assertions) {
///     return; // dev lane: the codegen is not the shipped one
/// }
/// let mut gate = SpeedGate::open("RIGID-BERNOULLI-VGH-932");
/// ```
/// One speed gate: a named set of paired cells, each printed as it is
/// measured and all asserted together at the end.
///
/// This is the ONE shape a wall-clock contract takes in this workspace, and
/// its call site is the marker the release lane derives the gate population
/// from: `scripts/speed_gates.py` walks the crates for every `#[test]` whose
/// body calls [`SpeedGate::open`], resolves each to an exact test path in the
/// compiled release binary, runs exactly that set, and refuses a run in which
/// any derived gate did not execute. A gate therefore cannot be forgotten by
/// a name-prefix filter, cannot print `ok` having asserted nothing, and
/// cannot assert in a lane whose codegen is not the shipped one.
///
/// # Shape of a gate
///
/// ```text
/// // parity pins run in EVERY build, before the gate opens
/// if cfg!(debug_assertions) {
///     return; // dev lane: skip the measurement, its verdict is about the wrong program
/// }
/// let mut gate = SpeedGate::open("RIGID-BERNOULLI-VGH-932");
/// let timing = paired_interleaved(15, 300_000, seed, production_arm, hand_arm);
/// gate.faster("y=1", &timing, "production", "hand");
/// gate.finish();
/// ```
///
/// The profile check is the test's, not the gate's (see the module
/// documentation above): a gate that is opened always asserts, and the dev
/// lane does not pay for millions of timed iterations whose result it could
/// not use because the test never opens one there.
///
/// # Two contracts, no third
///
/// * [`SpeedGate::faster`] — the #932 contract: A (the compiled lowering) must
///   be strictly faster than B (the strongest hand path or the generic tower
///   it specialises). Loss when `median_ratio() <= 1`.
/// * [`SpeedGate::not_slower`] — for a cell whose two arms do the same work by
///   construction and where no speed claim is made: A must not be measurably
///   slower than B, where "measurably" is the measurement's OWN resolution,
///   [`PairedTiming::ratio_resolution`]. Loss when
///   `median_ratio() + ratio_resolution() < 1`. There is no chosen tolerance
///   here: the instrument reports its noise floor, and that is the only
///   denominator a parity bar can honestly be stated in.
///
/// A gate that is opened and dropped without [`SpeedGate::finish`] panics, and
/// a gate finished with no cells panics: both are gates that verified nothing.
pub struct SpeedGate {
    token: &'static str,
    cells: usize,
    losses: Vec<String>,
    finished: bool,
}

impl SpeedGate {

}

impl Drop for SpeedGate {
}

/// SplitMix64 — a deterministic order sequence, so the interleave is randomised
/// but a run is reproducible. Deliberately not a dependency: the harness must
/// not be able to perturb the timing through an allocation or a dynamic call.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two arms doing identical work must land near ratio 1, and the harness
    /// must say so through `wins_fraction` too: identical arms should win about
    /// half the time, which is the signature a gate uses to recognise "no
    /// measurable difference" rather than reading a point estimate of 1.02 as a
    /// 2% win.
    #[test]
    fn identical_arms_report_no_winner() {
        let work = |x: f64| {
            let mut acc = x;
            for i in 0..64 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let timing = paired_interleaved(21, 2_000, 0xA5A5_1234, work, work);
        let median = timing.median_ratio();
        assert!(
            (median - 1.0).abs() < 0.35,
            "identical arms should sit near ratio 1: {}",
            timing.summary("a", "b")
        );
        let wins = timing.wins_fraction();
        assert!(
            (0.15..=0.85).contains(&wins),
            "identical arms should win about half the repetitions: {}",
            timing.summary("a", "b")
        );
    }

    /// A genuinely faster arm must be detected with every repetition agreeing —
    /// the property that separates a real margin from one inside the noise.
    #[test]
    fn a_large_real_difference_is_detected_unanimously() {
        let fast = |x: f64| {
            let mut acc = x;
            for i in 0..16 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let slow = |x: f64| {
            let mut acc = x;
            for i in 0..256 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let timing = paired_interleaved(15, 2_000, 0x5EED, fast, slow);
        assert!(
            timing.median_ratio() > 2.0,
            "a 16x work difference must show as a large ratio: {}",
            timing.summary("fast", "slow")
        );
        assert_eq!(
            timing.wins_fraction(),
            1.0,
            "a large real difference must win EVERY repetition: {}",
            timing.summary("fast", "slow")
        );
    }

    /// The order must actually vary. A harness that believes it randomises but
    /// does not is indistinguishable from the ones being replaced, and the
    /// position-bias diagnostic would silently become `NaN`.
    #[test]
    fn both_orders_occur_and_position_bias_is_reportable() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(20, 500, 7, work, work);
        let a_first = timing.a_went_first.iter().filter(|f| **f).count();
        assert!(
            a_first > 0 && a_first < timing.a_went_first.len(),
            "both arm orders must occur across repetitions, got {a_first} of {}",
            timing.a_went_first.len()
        );
        assert!(
            timing.first_position_bias().is_finite(),
            "position bias must be reportable once both orders occur"
        );
    }

    /// `ratio_resolution` is what tells a caller whether its bar is assertable.
    /// It must be finite and positive on a real measurement, or the gate has no
    /// way to know it is asserting inside its own noise.
    #[test]
    fn resolution_is_reported_and_positive() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(15, 500, 99, work, work);
        let resolution = timing.ratio_resolution();
        assert!(
            resolution.is_finite() && resolution > 0.0,
            "resolution must be a usable number: {}",
            timing.summary("a", "b")
        );
    }

    /// The summary must carry the numbers that could overturn the verdict, not
    /// just the verdict. A gate that prints only the ratio is how a 3% claim
    /// with 6% resolution gets read as established.
    #[test]
    fn summary_carries_the_overturning_numbers() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(9, 500, 3, work, work);
        let line = timing.summary("production", "hand");
        for field in [
            "production=",
            "hand=",
            "median_ratio=",
            "wins=",
            "resolution=",
            "position_bias=",
            "reps=",
        ] {
            assert!(line.contains(field), "summary is missing {field}: {line}");
        }
    }

    #[test]
    #[should_panic(expected = "at least one repetition")]
    fn zero_repetitions_is_refused_not_silently_empty() {
        let work = |x: f64| x;
        // Called as a bare statement. A discarding binding is banned in this
        // workspace, and it would be the wrong shape regardless: this call is
        // expected to panic, so there is no result to discard.
        paired_interleaved(0, 10, 1, work, work);
    }
}
