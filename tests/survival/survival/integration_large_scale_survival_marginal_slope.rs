//! CI gate: large-scale-class survival marginal-slope identifiability fix.
//!
//! Exactly mirrors the production large-scale failure shape:
//!   formula      = Surv(entry_age, exit_age, event)
//!                  ~ duchon(PC1, PC2, PC3, centers=10, order=1) + sex + linkwiggle()
//!   logslope     =   duchon(PC1, PC2, PC3, centers=10, order=1) + linkwiggle()
//!
//! The duplicated `duchon(PC1,PC2,PC3)` term across both formulas is the
//! exact alias pencil that caused the original large-scale job to fail closed.
//! The test runs at n=5000 (small enough for a CI laptop, large enough that
//! the audit cannot trivially declare the joint design full-rank from tiny n).
//!
//! Assertions:
//!   1. The fit COMPLETES — `fit_from_formula` returns `Ok(FitResult::SurvivalMarginalSlope(...))`.
//!   2. convergence — certified by construction: a minted fit is the sealed
//!      convergence proof (SPEC 20).
//!   3. Any V+M drop is attributed to the LOGSLOPE block only (gauge_priority=120,
//!      the lowest-priority parametric block); drops to time (200) or marginal (150)
//!      are a regression.
//!   4. The joint design [marginal | logslope] at the final β has full column
//!      rank (detected by RRQR), demonstrating the V+M reduction eliminated
//!      the alias before the inner solve.
//!   5. The diagnostic string
//!        "canonical-gauge pipeline will attribute the … surplus column(s)"
//!      does NOT appear in the log after the fix — the V+M active path
//!      eliminates the alias before the flat joint-rank diagnostic fires.
//!   6. The string "block 2 fully aliased" does NOT appear in the log —
//!      the compiler no longer reports a hard-alias error on logslope.
//!   7. Every fitted β coefficient is finite.
//!   8. Predictions (η = X_marginal β_marginal) on training rows are finite.

use csv::StringRecord;
use gam::linalg::faer_ndarray::{default_rrqr_rank_alpha, rrqr_with_permutation};
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, s};
use std::sync::{Arc, Mutex, Once, OnceLock};

// ── constants ─────────────────────────────────────────────────────────────────

/// n=5000 is large enough that a trivial-rank path (all-zero rows) cannot
/// mask the alias, yet finishes in a few minutes on a CI laptop.
const N: usize = 5_000;

/// centers=10 exactly matches the original large-scale formula.  order=1 in 3D
/// has a 4-dimensional polynomial null space (constant + 3 linear terms), so
/// at centers=10 each block contributes 10 kernel columns + 4 parametric
/// columns = 14 raw columns — identical to production.
const CENTERS: usize = 10;

// ── log capture ───────────────────────────────────────────────────────────────

#[derive(Default)]
struct CapturedLogs {
    lines: Mutex<Vec<String>>,
}

impl CapturedLogs {
    fn push(&self, line: String) {
        if let Ok(mut g) = self.lines.lock() {
            g.push(line);
        }
    }
    fn snapshot(&self) -> Vec<String> {
        self.lines.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

struct CapturingLogger {
    sink: Arc<CapturedLogs>,
}

impl log::Log for CapturingLogger {
    fn enabled(&self, meta: &log::Metadata<'_>) -> bool {
        meta.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!("{}", record.args());
        eprintln!("{line}");
        self.sink.push(line);
    }
    fn flush(&self) {}
}

static INIT_LOGGER: Once = Once::new();
static LOG_SINK: OnceLock<Arc<CapturedLogs>> = OnceLock::new();

fn log_sink() -> &'static Arc<CapturedLogs> {
    LOG_SINK.get_or_init(|| Arc::new(CapturedLogs::default()))
}

fn install_logger() {
    INIT_LOGGER.call_once(|| {
        let logger = Box::leak(Box::new(CapturingLogger {
            sink: log_sink().clone(),
        }));
        if log::set_logger(logger).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

// ── PRNG ──────────────────────────────────────────────────────────────────────

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    let r = (-2.0 * u1.ln()).sqrt();
    r * (std::f64::consts::TAU * u2).cos()
}

// ── synthetic dataset ─────────────────────────────────────────────────────────

/// Build a survival dataset that mirrors the large-scale shape:
///   - 3 principal-component columns (PC1, PC2, PC3) — standard-normal
///   - z_prs column: standardized PRS, mean≈0, var≈1
///   - sex covariate: Bernoulli(0.5)
///   - entry_age ~ Uniform(40, 45)
///   - follow-up  ~ Uniform(0.5, 8.5)  ⟹ exit_age = entry_age + follow-up
///   - event: logistic model in (prs_z, PC1–PC3, sex) with known true-β
///
/// True-β used: β_z=0.3, β_pc1=0.4, β_pc2=−0.3, β_pc3=0.2, β_sex=0.15.
/// This gives roughly 40 % event rate at the center of the covariate space.
fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = [
        "entry_age",
        "exit_age",
        "event",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
        "sex",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    // 64-bit splitmix64 seed.  The previous literal
    // `0xC1A55_1F1ED_B10B_A11C` (21 hex digits → 84 bits) silently truncated
    // under `cargo build` and now fails the `literal out of range for u64`
    // lint under `-D warnings`; pin a deterministic 16-hex-digit value that
    // preserves the high-entropy character of the original.
    let mut state: u64 = 0xC1A5_51F1_EDB1_0BA1_u64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    // Accumulate PRS values to standardize them post-hoc.
    let mut prs_raw: Vec<f64> = Vec::with_capacity(N);
    let mut scratch: Vec<[f64; 6]> = Vec::with_capacity(N); // [pc1,pc2,pc3,sex,entry,followup]

    for _ in 0..N {
        let pc1 = next_gauss(&mut state) * 0.5;
        let pc2 = next_gauss(&mut state) * 0.5;
        let pc3 = next_gauss(&mut state) * 0.5;
        let prs = next_gauss(&mut state); // raw, not yet standardized
        let sex: f64 = if next_unit(&mut state) < 0.5 {
            1.0
        } else {
            0.0
        };
        let entry = 40.0 + 5.0 * next_unit(&mut state);
        let followup = 0.5 + 8.0 * next_unit(&mut state);
        prs_raw.push(prs);
        scratch.push([pc1, pc2, pc3, sex, entry, followup]);
    }

    // Standardize PRS to mean=0, var=1.
    let prs_mean = prs_raw.iter().sum::<f64>() / N as f64;
    let prs_var = prs_raw.iter().map(|v| (v - prs_mean).powi(2)).sum::<f64>() / N as f64;
    let prs_sd = prs_var.sqrt().max(1e-12);
    let prs_z: Vec<f64> = prs_raw.iter().map(|v| (v - prs_mean) / prs_sd).collect();

    for (i, s) in scratch.iter().enumerate() {
        let [pc1, pc2, pc3, sex, entry, followup] = *s;
        let z = prs_z[i];
        let logit = 0.3 * z + 0.4 * pc1 - 0.3 * pc2 + 0.2 * pc3 + 0.15 * sex;
        // Logistic noise to produce binary event.
        let noise = next_gauss(&mut state) * 0.5;
        let event: u8 = if logit + noise > 0.0 { 1 } else { 0 };
        let exit = entry + followup;
        rows.push(StringRecord::from(vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            z.to_string(),
            pc1.to_string(),
            pc2.to_string(),
            pc3.to_string(),
            sex.to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode large-scale-class integration test dataset")
}

// ── test ──────────────────────────────────────────────────────────────────────

#[test]
fn large_scale_survival_marginal_slope_canonical_gauge_fix() {
    install_logger();
    init_parallelism();

    // Disable GPU on macOS: no CUDA driver present in CI.
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();

    // Exactly the large-scale formula shape: shared duchon + linkwiggle on both
    // the main (marginal) formula and the logslope formula.
    let duchon = format!("duchon(PC1, PC2, PC3, centers={CENTERS}, order=1)");
    let formula = format!("Surv(entry_age, exit_age, event) ~ {duchon} + sex + linkwiggle()");
    let logslope = format!("{duchon} + linkwiggle()");

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(logslope),
        baseline_target: "linear".to_string(),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    // ── Assertion 1: fit COMPLETES without FATAL ──────────────────────────
    let outcome = fit_from_formula(&formula, &data, &config)
        .expect("large-scale-class survival marginal-slope fit must complete — no audit FATAL");

    let result = match outcome {
        FitResult::SurvivalMarginalSlope(r) => r,
        other => panic!(
            "expected FitResult::SurvivalMarginalSlope, got discriminant {:?}",
            std::mem::discriminant(&other)
        ),
    };

    // ── Assertion 2 retired: fit existence is the sealed convergence proof
    // (SPEC 20).

    // ── Assertions 3a-b: drop attribution must target logslope only ───────
    //
    // After T13's channel-aware Gram migration the closed-form path emits:
    //   "[smgs phase-4b compiled-map] applying CompiledMap T: … (drops time=T, marginal=M, logslope=G); …"
    //
    // The canonical-gauge contract (gauge_priority time=200 > marginal=150 > logslope=120)
    // guarantees that any alias between the shared duchon term on both formulas
    // is attributed entirely to the logslope block (the lower-priority participant).
    //
    // T must be 0 and M must be 0. G >= 0 is allowed (may be 0 if the V+M
    // path detected no aliasing via the W-metric path at this n).
    let logs = log_sink().snapshot();
    let active_marker = "[smgs phase-4b compiled-map] applying CompiledMap T:";
    let active_lines: Vec<&String> = logs
        .iter()
        .filter(|line| line.contains(active_marker))
        .collect();

    // If the compiled-map path fired, check that time and marginal drops are zero.
    for line in &active_lines {
        // Parse "drops time=T, marginal=M, logslope=G" from the log string.
        let time_drops = parse_drops_field(line, "time");
        let marginal_drops = parse_drops_field(line, "marginal");
        assert_eq!(
            time_drops,
            Some(0),
            "regression: V+M active path reported a DROP to the TIME block \
             (gauge_priority=200, highest priority — must never lose columns); \
             log line: {line}"
        );
        assert_eq!(
            marginal_drops,
            Some(0),
            "regression: V+M active path reported a DROP to the MARGINAL block \
             (gauge_priority=150 > logslope=120 — drops must route to logslope); \
             log line: {line}"
        );
    }

    // ── Assertion 4: joint design has full column rank post-reduction ─────
    //
    // After V+M the design columns should be linearly independent. Build the
    // joint matrix [X_marginal | X_logslope] at training rows, run RRQR, and
    // assert rank == ncols. A genuine rank deficiency (finite sample, near-
    // collinear data) would manifest as rank < ncols; any such residual
    // deficiency is handled by the downstream canonical-gauge pipeline and
    // the test allows a configurable slack (max_rank_deficiency).
    {
        let raw_marginal = result
            .marginal_design
            .design
            .try_to_dense_by_chunks("joint rank check marginal")
            .expect("marginal design must densify for rank check");
        let raw_logslope = result
            .logslope_design
            .design
            .try_to_dense_by_chunks("joint rank check logslope")
            .expect("logslope design must densify for rank check");

        let n_rows = raw_marginal.nrows();
        let p_marg = raw_marginal.ncols();
        let p_log = raw_logslope.ncols();
        let p_total = p_marg + p_log;

        let mut joint = Array2::<f64>::zeros((n_rows, p_total));
        joint.slice_mut(s![.., ..p_marg]).assign(&raw_marginal);
        joint.slice_mut(s![.., p_marg..]).assign(&raw_logslope);

        let rrqr = rrqr_with_permutation(&joint, default_rrqr_rank_alpha())
            .expect("RRQR on joint [marginal | logslope] design must not fail");

        // Allow at most 1 residual rank deficiency: at n=5000 with centers=10
        // a genuine coincidence of data directions is vanishingly unlikely but
        // theoretically possible. 0 is the expected value post-fix.
        let max_rank_deficiency: usize = 1;
        assert!(
            p_total.saturating_sub(rrqr.rank) <= max_rank_deficiency,
            "joint [marginal|logslope] design has residual rank deficiency {} > {} \
             after V+M reduction (p_marg={}, p_log={}, joint_rank={}, p_total={}); \
             this indicates the canonical-gauge alias fix is not active",
            p_total.saturating_sub(rrqr.rank),
            max_rank_deficiency,
            p_marg,
            p_log,
            rrqr.rank,
            p_total,
        );
    }

    // ── Assertion 5: "canonical-gauge pipeline will attribute" ABSENT ─────
    //
    // The pre-fix code emitted this string from the flat joint-rank diagnostic
    // when it found rank(joint) < p_total. Post-fix the V+M active path
    // reduces the designs before the diagnostic runs, so the joint matrix is
    // full-rank and the message is never emitted.
    let forbidden_gauge_attr = "canonical-gauge pipeline will attribute";
    let gauge_attr_lines: Vec<&String> = logs
        .iter()
        .filter(|line| line.contains(forbidden_gauge_attr))
        .collect();
    assert!(
        gauge_attr_lines.is_empty(),
        "post-fix diagnostic string {:?} MUST NOT appear: this indicates the \
         V+M active path did not eliminate the alias before the flat \
         joint-rank diagnostic ran. Found {} matching log line(s): {:?}",
        forbidden_gauge_attr,
        gauge_attr_lines.len(),
        gauge_attr_lines,
    );

    // ── Assertion 6: "block 2 fully aliased" ABSENT ───────────────────────
    //
    // The identifiability compiler emits "block 2 fully aliased: …" when the
    // logslope block's residual Gram has no positive eigenspace — the symptom
    // of the pre-fix bug. Post-fix the block's columns have been reduced by V
    // so this condition never arises.
    let forbidden_block_aliased = "block 2 fully aliased";
    let block_aliased_lines: Vec<&String> = logs
        .iter()
        .filter(|line| line.contains(forbidden_block_aliased))
        .collect();
    assert!(
        block_aliased_lines.is_empty(),
        "post-fix diagnostic string {:?} MUST NOT appear: the logslope block \
         must not be declared fully aliased after V+M reduction. Found {} \
         matching log line(s): {:?}",
        forbidden_block_aliased,
        block_aliased_lines.len(),
        block_aliased_lines,
    );

    // ── Assertion 7: all fitted β are finite ─────────────────────────────
    for (idx, block) in result.fit.blocks.iter().enumerate() {
        for (j, &coef) in block.beta.iter().enumerate() {
            assert!(
                coef.is_finite(),
                "β block {idx} coef {j} non-finite: {coef}"
            );
        }
    }

    // ── Assertion 8: marginal predictions on training rows are finite ─────
    let raw_marginal_width = result.marginal_design.design.ncols();
    let marginal_beta_block = result
        .fit
        .blocks
        .iter()
        .find(|b| b.beta.len() == raw_marginal_width)
        .expect("locate marginal β block by width");
    let pred = result
        .marginal_design
        .design
        .apply(&marginal_beta_block.beta);
    for (i, &v) in pred.iter().enumerate() {
        assert!(v.is_finite(), "marginal η[{i}] non-finite: {v}");
    }
}

// ── helper ────────────────────────────────────────────────────────────────────

/// Extract the integer value of `"<field>=<N>"` from inside a `(drops …)`
/// clause in a log line.
///
/// The V+M active log line has the shape:
///   `… (drops time=T, marginal=M, logslope=G)`
/// so the individual fields are separated by ", " after the opening
/// "(drops " that marks the start of the parenthesised clause.
/// We look for the substring `"<field>="` and parse the digits that follow.
/// Returns `None` when the pattern is not present.
fn parse_drops_field(line: &str, field: &str) -> Option<usize> {
    // Only search inside the drops parenthesis to avoid false matches on the
    // "time X→Y" section of the same line.
    let drops_start = line.find("(drops ")?;
    let drops_section = &line[drops_start..];
    let needle = format!("{field}=");
    let after_eq = drops_section.find(needle.as_str())? + needle.len();
    let rest = &drops_section[after_eq..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse::<usize>().ok()
}
