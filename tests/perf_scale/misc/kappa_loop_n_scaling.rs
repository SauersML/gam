//! Measurement: is the spatial length-scale (κ) outer loop n-independent?
//!
//! The #1033b Chebyshev-in-ψ Gram tensor (`solver/psi_gram_tensor.rs`) makes
//! every κ-trial inside the spatial length-scale optimizer cost O(D²k²) — free
//! of the sample size n — by pre-expanding the conditioned design's Gram into a
//! certified polynomial in ψ = log κ. Only the *one-time* tensor build and the
//! *final* PIRLS assembly remain O(n). So as n grows, the wall-clock spent
//! inside the κ outer loop (beyond the single final fit) should stay roughly
//! flat rather than scaling with n.
//!
//! This harness isolates per-callback κ-trial cost through the optimizer's
//! structured `KAPPA-PHASE-SUMMARY` counters. Those counters start after the
//! one-time tensor setup / cold realization pass and report cost/eval/EFS trial
//! callback totals plus call counts.
//!
//! **#1868 deterministic gate.** The hard assertions no longer read a noisy
//! wall-clock per-callback *ratio* (which needed a ~320× n lever to rise above
//! shared-node timing jitter, and whose 320k rung took ~2.2 h to run — the O(n)
//! loop being itself what made the large-n rungs slow). Instead they read the
//! exact-integer `nfree_skip_row_touches` counter: the number of length-`n`
//! row-element touches the Gaussian inner synthesis performed on the #1033
//! n-free κ-trial *skip* path. The architectural invariant (#1033: "an in-window
//! hyperparameter trial touches only k×k objects") is literally `touches == 0`,
//! at every n. Paired with the `slow_path_resets` soundness gate (which pins the
//! *design re-realization* O(n) lane off), this deterministically certifies BOTH
//! O(n) lanes of a κ trial are zero — verifiable in milliseconds at small n,
//! because an integer that must be 0 cannot hide an O(n) term behind a small
//! constant. Wall-clock is retained only as a report-only secondary signal.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        SpatialLengthScaleOptimizationTiming, TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// 1-D isotropic Gaussian-identity spatial fixture — exactly the tensor-eligible
/// path (`coord_dim == 1`, Gaussian + identity link). Deterministic truth keeps
/// this a geometry/timing check, not a stochastic power test.
fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        // gentle smooth signal — a well-conditioned target for the κ optimizer
        y[i] = (t).sin();
    }
    (x, y)
}

fn spec_1d(aniso: bool) -> TermCollectionSpec {
    // 1-D Gaussian HYBRID DUCHON (length_scale=Some) — the basis on which the
    // #1033 n-free κ lane is COMPLETE: the value-lane PsiGramTensor attaches AND
    // `supports_nfree_penalty_rekey()` is true (Duchon metadata re-keys S(ψ)
    // exactly n-free from the frozen centers/collocation points), so the design-
    // realization skip fires and the BFGS-routing arm (#1033 b437d9ff2) engages.
    //
    // NOT Matérn: the realized Matérn design carries the operator-triplet penalty
    // (mass/tension/stiffness). The n-free re-key CAN in fact reproduce that triplet
    // byte-exactly across ψ (the #1274 gate
    // `matern_2d_nfree_penalty_rekey_is_byte_exact_but_design_skip_is_not_admitted`
    // pins this to <1e-10), so the historical "the re-key cannot reproduce the
    // operator triplet" rationale was wrong. The real reason Matérn is excluded from
    // `supports_nfree_penalty_rekey` (the #1033 6a5a2e1 re-admission was reverted by
    // feb0eb50b, #1274) is twofold: (1) the #1264 `reduced_basis_equal` design-skip
    // gate refuses Matérn's rotating collocation geometry, so the O(n) design re-
    // realization still fires per trial even with the penalty re-keyed — no speed
    // win; and (2) re-admitting Matérn perturbs the selected fit enough to miss the
    // truth-recovery bar (the `matern_nu_sweep_uniform_quality_on_sin1` probe goes
    // slower AND fails when Matérn is admitted). So Matérn stays on the exact slow
    // re-key path. Measuring n-independence on Matérn would test a basis the n-free
    // architecture intentionally does NOT cover. This is exactly the config the
    // passing bit-identity gate `psi_gram_tensor_fast_path_skips_n_row_lane_and_
    // matches_streamed` uses to exercise the armed skip.
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    // None routes the isotropic κ optimizer (the n-free-arming
                    // case); Some(_) routes the per-axis (anisotropic) optimizer
                    // even for a single axis — the discriminator under test.
                    aniso_log_scales: if aniso { Some(vec![0.0]) } else { None },
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                // PRODUCTION geometry: None lets the 1-D axis auto-standardize
                // (#1214/#1215), the real default-fit path. An input_scale:[1.0]
                // pin would be a gamed gate masking the open geometry gap.
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-6,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

/// Outcome of one fit attempt: whole-fit wall-clock plus the internal κ-trial
/// timing when κ optimization actually ran. The κ timing excludes the one-time
/// tensor/cold setup pass by construction; it is the object #1033 accepts on.
#[derive(Clone, Copy, Debug)]
struct FitTiming {
    wall_s: f64,
    kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
}

impl FitTiming {
}

/// Outcome of one fit attempt: either timings (converged) or the failure reason
/// string (so the diagnostic can tabulate instead of aborting).
fn run_fit(
    n: usize,
    kappa_enabled: bool,
    aniso: bool,
    bounds: (f64, f64),
) -> Result<FitTiming, String> {
    let (x, y) = simulate_1d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: kappa_enabled,
        max_outer_iter: if kappa_enabled { 15 } else { 1 },
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: bounds.0,
        max_length_scale: bounds.1,
        pilot_subsample_threshold: 0,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(weights),
        offset: std::sync::Arc::new(offset),
        spec: spec_1d(aniso),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        estimate_tweedie_p: false,
    }))
    .map_err(|e| format!("{e:?}"))?;
    let dt = t0.elapsed().as_secs_f64();

    match result {
        FitResult::Standard(s) => {
            if !s.fit.beta.iter().all(|v: &f64| v.is_finite()) {
                return Err("non-finite coefficients".to_string());
            }
            // Non-degeneracy guard (#1033): the n-free κ path must not pass the
            // timing gate by collapsing to a trivially-cheap degenerate fit (an
            // all-zero / fully-flattened smooth is fast but wrong). The target
            // `y = sin(t)` has unit-order amplitude, so a smooth that actually
            // tracks it has a non-trivial coefficient norm. A collapsed fit would
            // shrink β to ≈0. This is a coarse tripwire against a "fast because
            // wrong" optimum; the principled κ-optimum/fit-quality oracle is the
            // separate mgcv/truth-recovery quality suite (gam_duchon_1d_matches_
            // mgcv_ds, gam_matern_smooth_recovers_truth).
            let beta_norm = s.fit.beta.iter().map(|v| v * v).sum::<f64>().sqrt();
            if kappa_enabled && beta_norm < 1e-3 {
                return Err(format!(
                    "κ fit collapsed to a near-zero smooth (‖β‖={beta_norm:.3e}); the \
                     n-free outer loop must recover the sin(t) signal, not a fast \
                     degenerate optimum"
                ));
            }
            Ok(FitTiming {
                wall_s: dt,
                kappa_timing: s.kappa_timing,
            })
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

fn run_kappa_trial_seconds(n: usize, aniso: bool, bounds: (f64, f64)) -> Result<FitTiming, String> {
    let timing = run_fit(n, true, aniso, bounds)?;
    if timing.kappa_timing.is_none() {
        return Err("κ optimizer did not report internal trial timing".to_string());
    }
    Ok(timing)
}

/// Diagnostic: which 1-D Gaussian κ configuration actually converges? Isolates
/// the optimizer path (isotropic-analytic vs per-axis) and the length-scale
/// bounds (tight vs wide), so a non-convergence can be attributed to a real
/// gradient/optimizer defect rather than a boundary solution or a bad fixture.
#[test]
fn kappa_iso_1d_convergence_diagnostic() {
    let n = 600usize;
    let tight = (1e-2, 1e2);
    let wide = (1e-4, 1e4);
    let configs: [(&str, bool, (f64, f64)); 4] = [
        ("iso  / tight", false, tight),
        ("iso  / wide ", false, wide),
        ("aniso/ tight", true, tight),
        ("aniso/ wide ", true, wide),
    ];
    eprintln!("[kappa-diag] n={n}  (1-D hybrid Duchon, Gaussian-identity, single penalty)");
    let mut outcomes = Vec::new();
    for (label, aniso, bounds) in configs {
        let r = run_fit(n, true, aniso, bounds);
        match &r {
            Ok(timing) => eprintln!("[kappa-diag] {label}: CONVERGED in {:.3}s", timing.wall_s),
            Err(reason) => eprintln!("[kappa-diag] {label}: FAILED — {reason}"),
        }
        outcomes.push((label, r));
    }
    // GATED as of gam#2760. This was report-only with the note "the follow-up
    // measurement/fix lands once the converging path is known"; it is known.
    //
    // Why this is the right gate for the #2760 root causes, from an angle the
    // n-ladder does not cover. At `n = 600` the scalar-ρ incumbent already puts
    // 4 of 5 coordinates below `−JOINT_RHO_BOUND`, so the pre-#2760 box pasted
    // its lower wall onto them; and the ψ-Gram surrogate is armed on `tight`
    // bounds and not on `wide`. `iso / tight` — both defects at once — was the
    // one red cell here, refusing at `|Pg| = 7.011e-2` against `4.168e-2` after
    // a `StepSizeTooSmall` line search at 14 iterations. All four cells converge
    // now, and the four differ in exactly the two axes the repairs touch
    // (which optimizer routes the ψ block; whether the surrogate arms), so a
    // regression of either root cause turns a cell red here in seconds — long
    // before the 16k ladder rung would notice.
    for (label, outcome) in &outcomes {
        assert!(
            outcome.is_ok(),
            "[kappa-diag] {label}: the 1-D Gaussian κ fit must converge at n={n} in every \
             (optimizer route × length-scale window) configuration; it refused with {:?}. \
             This cell was red before gam#2760 whenever the joint ρ box pinned the graded \
             incumbent on its own wall, or the certified n-free ψ-Gram surrogate was still \
             the measure at certification time.",
            outcome.as_ref().err(),
        );
    }
    eprintln!(
        "[kappa-diag] all {} configurations converged",
        outcomes.len()
    );
}

/// Pin the sample-size threshold at which the isotropic-analytic κ optimizer
/// tips from converging to non-converging on the *same* well-conditioned 1-D
/// Duchon Gaussian fixture (gentle `y=sin(t)`, 12 centers, single penalty, tight
/// bounds). n=600 converges; earlier runs showed n=1000 failing with a stuck
/// `grad_norm≈1.9e3`. This sweep brackets the transition so the defect report
/// carries an exact reproducer.
///
/// GATED as of gam#2760, for the same reason as
/// `kappa_iso_1d_convergence_diagnostic`: the transition it was written to
/// bracket is gone, so "the printed sweep is the deliverable" no longer buys
/// anything a pass/fail would not. There IS no threshold now — 600, 800, 1000
/// and 1200 all converge — and that sentence is the claim worth defending. It
/// is a cheap and sharp one: this sweep runs four fits in seconds, and each of
/// them crosses the regime where the scalar-ρ incumbent falls outside the joint
/// ±12 prior and where the ψ-Gram surrogate is armed, so a regression of either
/// #2760 root cause reappears here as a rung going red rather than as a wall-
/// clock mystery at 16k.
#[test]
fn kappa_iso_1d_n_threshold_sweep() {
    let bounds = (1e-2, 1e2);
    eprintln!("[kappa-nthresh] iso-1D hybrid Duchon, Gaussian, single penalty, bounds={bounds:?}");
    let mut refusals = Vec::new();
    for &n in &[600usize, 800, 1000, 1200] {
        match run_fit(n, true, false, bounds) {
            Ok(timing) => eprintln!(
                "[kappa-nthresh] n={n:>5}: CONVERGED in {:.1}s",
                timing.wall_s
            ),
            Err(reason) => {
                eprintln!("[kappa-nthresh] n={n:>5}: FAILED — {reason}");
                refusals.push((n, reason));
            }
        }
    }
    assert!(
        refusals.is_empty(),
        "[kappa-nthresh] the iso-1D κ fit must converge at every rung of this sweep — there is \
         no sample-size threshold left to bracket. Refused at: {:?}",
        refusals.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
    );
}

/// #1033 MICRO read (2 points, n=1k vs 2k): the smallest discriminant of
/// n-independence. Per-callback cost flat ⇒ n-free; tracking the 2× n ⇒ O(n).
/// Finishes in seconds — a development-loop probe, NOT the close gate (the full
/// 1k→16k/320k ladders are). No bar tightening here vs the headline ≤8× / flat
/// reset contract; this just surfaces the ratio fast.
/// Minimal stderr logger so the #1033 `[NFREE-RESET ...]` info diagnostics
/// emitted by the solver's reset lanes surface in the micro probe's output. Only
/// forwards records whose message starts with `[NFREE-RESET` to keep the trace
/// readable; everything else is dropped. Installed once per test process (nextest
/// isolates test binaries per process, so `set_logger` cannot race).
struct NfreeResetLogger {
    file: std::sync::Mutex<std::fs::File>,
}
impl log::Log for NfreeResetLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        use std::io::Write;
        let msg = format!("{}", record.args());
        if msg.starts_with("[NFREE-RESET")
            || msg.starts_with("[KAPPA-PHASE-SUMMARY")
            || msg.starts_with("[KAPPA-PHASE-PRIME")
            || msg.starts_with("[KAPPA-PHASE-FLOOR")
            || msg.starts_with("[KAPPA-PHASE-CEIL")
        {
            if let Ok(mut f) = self.file.lock() {
                // A logger cannot log its own write failure without recursing,
                // so the trace file losing a record is reported on stderr.
                if let Err(error) = writeln!(f, "{msg}").and_then(|()| f.flush()) {
                    eprintln!("[nfree-trace] dropped a record: {error}");
                }
            }
        }
    }
    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            use std::io::Write;
            if let Err(error) = f.flush() {
                eprintln!("[nfree-trace] flush failed: {error}");
            }
        }
    }
}

static NFREE_RESET_LOGGER: std::sync::OnceLock<NfreeResetLogger> = std::sync::OnceLock::new();

fn install_nfree_reset_logger() {
    // Route the solver's `[NFREE-RESET ...]` info diagnostics to a file
    // (`/tmp/nfree_trace.log`), bypassing nextest's stdout/stderr buffering
    // which silently drops the early reset-time records. Idempotent: the
    // `OnceLock` + `set_logger` error-swallow make repeated calls safe.
    let logger = NFREE_RESET_LOGGER.get_or_init(|| {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("/tmp/nfree_trace.log")
            .expect("open nfree trace log");
        NfreeResetLogger {
            file: std::sync::Mutex::new(file),
        }
    });
    // Losing the race to an already-installed logger is the documented
    // idempotent path and carries nothing to report.
    drop(log::set_logger(logger));
    log::set_max_level(log::LevelFilter::Info);
}

#[test]
fn zzz_diag_n16000_reset_reasons() {
    install_nfree_reset_logger();
    let (aniso, bounds) = (false, (1e-2, 1e2));
    // Warm-up only — a failure here must not fail the diagnostic, but it
    // explains any oddity in the 16k timings below, so say so.
    if let Err(error) = run_fit(1000, true, aniso, bounds) {
        eprintln!("[diag-16k] warm-up fit failed: {error}");
    }
    let r = run_kappa_trial_seconds(16_000, aniso, bounds).unwrap_or_else(|reason| {
        // The refusal text carries |Pg|, its bound, the rung, the rail tests and a
        // rho checkpoint. A bare `.unwrap()` buries all of it in a panic payload,
        // where a capture filter can drop it and leave a panic with no reason --
        // indistinguishable from an unexplained crash. The warm-up above already
        // uses this form; match it.
        panic!("[diag-16k] n=16000 kappa trial failed: {reason}")
    });
    let t = r.kappa_timing.unwrap();
    eprintln!(
        "[diag-16k] resets={} miss(shape/value/grad/pen/rev)={}/{}/{}/{}/{}",
        t.slow_path_resets,
        t.nfree_miss_shape,
        t.nfree_miss_value,
        t.nfree_miss_gradient,
        t.nfree_miss_penalty,
        t.nfree_miss_revision,
    );
}

// #1033 acceptance gate, split PER-`n` so each rung has its own nextest
// slow-timeout budget (the former single monolithic `#[test]` ran all three
// fits serially and the 16k rung could push the whole function past the 600s
// SIGKILL, making the deterministic `touches == 0` assertion structurally
// unreachable in the standard shard). The `touches == 0` integer gate already
// pins n-independence EXACTLY at each individual n — a value that must be 0
// cannot hide a linear term behind a constant — so the multi-point ladder was
// never load-bearing for the hard assertion; it fed only a report-only
// wall-clock trend (preserved in `_fast_ladder`). Two points (1k, 16k) already
// pin the exact n-independence; the 4k rung is kept as an interior witness.

// ───────────────────────── GLM (non-Gaussian) κ-loop ─────────────────────────
//
// The Gaussian-identity κ loop is FULLY n-free: it takes the design-revision
// skip (`skip_design_realization`) so the design is never re-realized, and the
// inner solve reads `XᵀWX(ψ)`/`XᵀWz(ψ)` straight from the n-free `PsiGramTensor`.
//
// The GLM (Poisson/Binomial/Gamma/NB) κ loop is a DIFFERENT, weaker lane and is
// NOT n-independent by construction. `skip_design_realization` is gated on the
// Gaussian `PsiGramTensor` covering ψ; a GLM fit installs no such tensor, so
// every κ trial still runs `ensure_theta` → `apply_log_kappa`, re-realizing the
// O(n·k) design. What the GLM lane DOES save is the per-trial Gram RE-STREAM:
// the certified frozen-weight (`FrozenWeightGramTensor`) serves the first-Fisher
// `XᵀWX(ψ)` n-free (O(D·k²) instead of O(n·k²)) whenever the warm-β Fisher
// weights are within drift tolerance. So the GLM per-trial cost drops from
// O(n·k²) to O(n·k) — Gram-reduced, but still O(n) through the design realize.
//
// Therefore a ≤8× n-INDEPENDENCE gate would be ARCHITECTURALLY FALSE on GLM. The
// honest GLM measurement is this REPORT-ONLY ladder: it documents the residual
// O(n·k) design-realization floor and exercises the frozen-W Gram lane (and the
// #1033 per-trial Fisher-weight memo). The only hard assertion is a SUPER-LINEAR
// catastrophe tripwire — if the frozen-W Gram lane stops firing, the per-trial
// cost reverts to O(n·k²) and the per-callback ratio would track ~n_ratio² (way
// past the design-realize O(n) floor); the tripwire catches exactly that
// regression without pretending the GLM lane is n-free.

