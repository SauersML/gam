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
    /// Median of the paired ratios — the headline estimate.
    ///
    /// The median rather than the mean because a single descheduled repetition
    /// produces an arbitrarily large outlier in one direction only, and rather
    /// than a minimum because a minimum is the statistic that preserves a
    /// systematic offset (see the module docs).
    pub fn median_ratio(&self) -> f64 {
        median(&self.ratios)
    }

    /// Fraction of repetitions in which A was faster than B.
    ///
    /// This is the gate's honesty check. A median ratio of 1.06 with a wins
    /// fraction of 1.0 is a real effect; the same 1.06 with 0.55 means the
    /// repetitions disagree with each other and the point estimate is riding on
    /// a few draws.
    pub fn wins_fraction(&self) -> f64 {
        if self.ratios.is_empty() {
            return f64::NAN;
        }
        let wins = self
            .a_ns
            .iter()
            .zip(self.b_ns.iter())
            .filter(|(a, b)| a < b)
            .count();
        wins as f64 / self.a_ns.len() as f64
    }

    /// Half the central 90% span of the paired ratios, as a fraction of the
    /// median — the gate's own **resolution**.
    ///
    /// A claimed margin smaller than this is not measurable by this harness at
    /// this repetition count, and asserting it is asserting noise. Report it
    /// beside the margin so the comparison is visible.
    pub fn ratio_resolution(&self) -> f64 {
        if self.ratios.len() < 2 {
            return f64::NAN;
        }
        let mut sorted = self.ratios.clone();
        sorted.sort_by(f64::total_cmp);
        let lo = quantile_sorted(&sorted, 0.05);
        let hi = quantile_sorted(&sorted, 0.95);
        let med = median(&self.ratios);
        if !med.is_finite() || med == 0.0 {
            return f64::NAN;
        }
        ((hi - lo) / 2.0 / med).abs()
    }

    /// Median ratio among repetitions where A ran first, minus the median among
    /// those where B ran first.
    ///
    /// This is the diagnostic the old harnesses could not produce, because they
    /// never varied the order. A value near zero says position does not matter
    /// on this host; a large value says the measurement is dominated by
    /// whichever arm goes first, and **no ordering of a non-randomised harness
    /// would have been trustworthy**. Returns `NaN` if either group is empty.
    pub fn first_position_bias(&self) -> f64 {
        let a_first: Vec<f64> = self
            .ratios
            .iter()
            .zip(self.a_went_first.iter())
            .filter(|(_, first)| **first)
            .map(|(r, _)| *r)
            .collect();
        let b_first: Vec<f64> = self
            .ratios
            .iter()
            .zip(self.a_went_first.iter())
            .filter(|(_, first)| !**first)
            .map(|(r, _)| *r)
            .collect();
        if a_first.is_empty() || b_first.is_empty() {
            return f64::NAN;
        }
        median(&a_first) - median(&b_first)
    }

    /// One line carrying everything needed to audit the verdict, including the
    /// numbers that would reveal the verdict as unsupported.
    pub fn summary(&self, a_label: &str, b_label: &str) -> String {
        format!(
            "{a_label}={:.2} ns/iter {b_label}={:.2} ns/iter \
             median_ratio={:.6} wins={:.2} resolution={:.4} position_bias={:.4} reps={}",
            median(&self.a_ns),
            median(&self.b_ns),
            self.median_ratio(),
            self.wins_fraction(),
            self.ratio_resolution(),
            self.first_position_bias(),
            self.ratios.len(),
        )
    }
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
    /// Open a gate. It always asserts; the test decides whether this build is
    /// one whose verdict is meaningful before calling (see the type docs).
    ///
    /// `token` is the stable, grep-able prefix every cell line of this gate
    /// is printed under (for example `RIGID-BERNOULLI-VGH-932`).
    #[must_use]
    pub fn open(token: &'static str) -> Self {
        Self {
            token,
            cells: 0,
            losses: Vec::new(),
            finished: false,
        }
    }

    fn record(&mut self, verdict: &str, cell: &str, timing: &PairedTiming, a: &str, b: &str) {
        self.cells += 1;
        eprintln!(
            "{} {cell} {} verdict={verdict}",
            self.token,
            timing.summary(a, b),
        );
        if verdict != "pass" {
            self.losses
                .push(format!("{cell}: {} ({verdict})", timing.summary(a, b)));
        }
    }

    /// Record a cell whose contract is "A is strictly faster than B".
    pub fn faster(&mut self, cell: &str, timing: &PairedTiming, a: &str, b: &str) {
        let verdict = if timing.median_ratio() > 1.0 {
            "pass"
        } else {
            "FAIL: A must be faster than B"
        };
        self.record(verdict, cell, timing, a, b);
    }

    /// Assert that every recorded cell met its contract, naming all that did
    /// not. Consumes the gate.
    pub fn finish(mut self) {
        self.finished = true;
        assert!(
            self.cells > 0,
            "{}: a speed gate finished with no measured cell verifies nothing",
            self.token
        );
        assert!(
            self.losses.is_empty(),
            "{}: {} of {} cell(s) failed their speed contract:\n{}",
            self.token,
            self.losses.len(),
            self.cells,
            self.losses.join("\n"),
        );
    }
}

impl Drop for SpeedGate {
    fn drop(&mut self) {
        // SAFETY: a gate dropped without `finish()` verified nothing, and a
        // test that reached this point would otherwise print `ok`; failing
        // loudly is the whole contract, and the `panicking()` guard keeps an
        // unwinding test from double-panicking.
        if !self.finished && !std::thread::panicking() {
            panic!(
                "{}: a speed gate was opened and dropped without `finish()`; it asserted nothing",
                self.token
            );
        }
    }
}

/// Make one arm call evaluate `rows` rows, so the arm is large relative to the
/// harness's per-call cost (see the module documentation: a single-row arm of a
/// few tens of nanoseconds lets a few nanoseconds of closure-inlining asymmetry
/// decide a small margin, and a 43 ns arm changed sign under batching).
///
/// The rows are independent of one another, as production's rows are: a data
/// loop hands each row its own inputs and folds the results, and the processor
/// overlaps consecutive rows, so what a batch measures is throughput. The
/// first version of this adapter chained the rows instead -- row `i + 1` was
/// perturbed by the fold of rows `0..=i` -- and that measured the latency of
/// one input's path through the arm, not the arm's work: the tower-3 prune of
/// the binomial coefficients, which does strictly less arithmetic than the
/// full tower, went from a 1.56x win under one call per iteration to a 0.99
/// loss under the chain, and two order-4 formulas within one instruction of
/// each other in every class were ranked by where the nudged coefficient
/// enters the expression.
///
/// Each row is perturbed by a distinct multiple of a negligible step, so no
/// two rows of a batch share an input: a pure outlined arm's calls cannot be
/// merged, and an inlined arm cannot be evaluated once for the batch. What
/// this adapter cannot prevent is the hoisting of an inlined arm's
/// fixture-invariant work out of the batch; an arm that is timed must be
/// outlined (`#[inline(never)]`) or take its fixture through an opaque
/// boundary, and that is the arm's responsibility, not the adapter's. The
/// returned closure has the `FnMut(f64) -> f64` shape the harness times.
pub fn batched<F: FnMut(f64) -> f64>(rows: usize, mut arm: F) -> impl FnMut(f64) -> f64 {
    assert!(rows > 0, "a batched arm needs at least one row");
    move |nudge| {
        let mut fold = 0.0_f64;
        for row in 0..rows {
            fold += arm(nudge + row as f64 * 1e-18);
        }
        fold
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

/// Linear-interpolated quantile of an already-sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

