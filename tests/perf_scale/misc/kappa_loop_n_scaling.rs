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

/// #1264 basis-rotation fallback allowance (n-INDEPENDENT).
///
/// On the near-singular production Duchon Gram (κ(G) ≈ 9.5e14) the conditioned
/// reduced basis ROTATES with ψ. The #1264 `reduced_basis_equal` soundness gate
/// then CORRECTLY forces a bounded number of κ trials off the n-free skip and
/// onto the exact O(n) `reset_surface` path — β̂-soundness over n-independence
/// (the skip's Chebyshev-interpolated Gram moves β̂ by ~1.7e-5, 17× the 1e-6 bar,
/// across a basis rotation; #1033 is frontier-blocked on rotating Duchon
/// geometry). The COUNT of such fallbacks is bounded by how many times the
/// optimizer's ψ trajectory crosses a basis-rotation boundary — an
/// n-INDEPENDENT quantity (observed 0–4 across n = 1k–16k), NOT the
/// O(n)-per-callback synthesis #1868 is about (which the deterministic
/// `nfree_skip_row_touches == 0` gate pins to zero at every n).
///
/// This replaces the old `resets_last <= resets_first + 1` gate, which wrongly
/// assumed ZERO rotation fallbacks — false for the production rotating-Duchon
/// geometry the fixture exercises. A genuinely DISARMED skip would instead
/// re-enter the exact lane on ≈EVERY κ trial (≥ 36 at n ≥ 16k here — the trial
/// count, not the crossing count), far above this cap; so the gate still catches
/// a broken skip while tolerating the bounded, correctness-mandated rotation
/// fallback. Paired with the value-only miss-attribution assertion (all resets
/// must be #1264 `nfree_miss_value` rotations, never penalty/revision/gradient
/// misses that would signal a genuine skip-logic break), this is a strictly more
/// honest n-independence gate than the flat-reset assumption it replaces.
///
/// NOTE: commit 4defbc478 (fix #1868) further drops the redundant `covers_skip`
/// clause from the *cost-probe* gate (a value-only probe returns only the
/// β̂-stationary REML scalar, so it needs no basis-rotation soundness witness),
/// which serves those probes n-free ACROSS rotations and currently drives the
/// observed reset count to 0 at every n on this fixture. The cap is kept nonzero
/// as bounded headroom for the #1264 rotation fallback on the still-gated
/// gradient eval and for parallel-reduction trajectory jitter.
const NFREE_ROTATION_FALLBACK_CAP: u64 = 8;

/// Sum of `slow_path_reset` misses that indicate a GENUINE skip-logic break —
/// every miss category EXCEPT the #1264 `nfree_miss_value` basis-rotation
/// fallback, which is the one correctness-mandated, n-independent-bounded
/// exception (see [`NFREE_ROTATION_FALLBACK_CAP`]). A nonzero value here means
/// the n-free skip fell through for a reason OTHER than a basis rotation
/// (wrong shape, uncertified gradient window, stale penalty re-key, unpinned
/// revision, an unexpected second-order/Hessian trial) — a real regression that
/// the bounded-count cap alone would not catch.
fn non_rotation_resets(t: &SpatialLengthScaleOptimizationTiming) -> u64 {
    t.nfree_miss_shape
        + t.nfree_miss_gradient
        + t.nfree_miss_penalty
        + t.nfree_miss_revision
        + t.nfree_miss_second_order
        + t.nfree_miss_other
}

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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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
    fn kappa_callback_avg_s(self) -> Option<f64> {
        self.kappa_timing.map(|timing| {
            let calls = timing.cost_calls + timing.eval_calls + timing.efs_calls;
            if calls == 0 {
                timing.trial_total_s()
            } else {
                timing.trial_total_s() / calls as f64
            }
        })
    }
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
        outcomes.push((label, r.is_ok()));
    }
    // Report-only: this diagnostic exists to attribute the failure, not to gate
    // CI on it. The follow-up measurement/fix lands once the converging path is
    // known. (No assertion here — the printed matrix is the deliverable.)
    let any_ok = outcomes.iter().any(|(_, ok)| *ok);
    eprintln!("[kappa-diag] any-converged={any_ok}");
}

/// Pin the sample-size threshold at which the isotropic-analytic κ optimizer
/// tips from converging to non-converging on the *same* well-conditioned 1-D
/// Duchon Gaussian fixture (gentle `y=sin(t)`, 12 centers, single penalty, tight
/// bounds). n=600 converges; earlier runs showed n=1000 failing with a stuck
/// `grad_norm≈1.9e3`. This sweep brackets the transition so the defect report
/// carries an exact reproducer. Report-only (the printed sweep is the
/// deliverable); the companion measurement stays gated until it is fixed.
#[test]
fn kappa_iso_1d_n_threshold_sweep() {
    let bounds = (1e-2, 1e2);
    eprintln!("[kappa-nthresh] iso-1D hybrid Duchon, Gaussian, single penalty, bounds={bounds:?}");
    for &n in &[600usize, 800, 1000, 1200] {
        match run_fit(n, true, false, bounds) {
            Ok(timing) => eprintln!(
                "[kappa-nthresh] n={n:>5}: CONVERGED in {:.1}s",
                timing.wall_s
            ),
            Err(reason) => eprintln!("[kappa-nthresh] n={n:>5}: FAILED — {reason}"),
        }
    }
}

/// #1033 FAST-READ companion to `kappa_outer_loop_is_n_independent`: the same
/// per-callback κ-trial measurement on a SMALL n-ladder (1k → 16k) that
/// completes quickly, so the n-free skip's flat-vs-linear behaviour can be read
/// inside an iteration loop without waiting on the 320k sweep.
///
/// The discriminant is unambiguous at this scale: across a 16× n increase an
/// O(n) per-trial regression tracks ~16×, while a truly n-free outer loop holds
/// the average callback cost ~flat (drifting only with the fixed O(D²k²) trial
/// cost and shared-node timing jitter). The same ≤8× bar as the headline applies — at
/// 16× n it is ~n^0.72, decisively sub-linear but safely above timing noise. A
/// green here is the fast close-signal; the full 320k sweep is the final stamp.
#[test]
fn kappa_outer_loop_is_n_independent_fast_ladder() {
    let (aniso, bounds) = (false, (1e-2, 1e2));
    let warm = run_fit(1000, true, aniso, bounds).unwrap_or_else(|reason| {
        panic!(
            "[kappa-fast-ladder] warm iso-κ fit failed ({reason}) — iso-1D κ \
             convergence is fixed (#1053/#1066/#1069); a failure here is a real \
             regression in the tensor-eligible isotropic length-scale optimizer"
        )
    });
    eprintln!(
        "[kappa-fast-ladder] warm-up fit primed caches in {:.4}s",
        warm.wall_s
    );

    // Small ladder: 1k → 16k (16×). Enough to read the slope; ~2–3 min total.
    let ns = [1_000usize, 4_000, 16_000];
    let mut kappa_phase = Vec::with_capacity(ns.len());
    let mut kappa_calls = Vec::with_capacity(ns.len());
    let mut kappa_resets = Vec::with_capacity(ns.len());
    let mut kappa_skip_touches = Vec::with_capacity(ns.len());
    let mut kappa_timings = Vec::with_capacity(ns.len());
    eprintln!(
        "[kappa-fast-ladder] {:>9}  {:>10}  {:>12}  {:>12}  {:>9}  {:>9}  {:>6}  {:>6}  {:>6}  {:>9}  {:>9}  {:>9}",
        "n",
        "t_kappa_s",
        "kappa_sum_s",
        "callback_s",
        "resets",
        "revs",
        "cost",
        "eval",
        "efs",
        "cost_s",
        "eval_s",
        "efs_s"
    );
    for &n in &ns {
        let kappa = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let timing = kappa.kappa_timing.unwrap();
        let phase = timing.trial_total_s().max(0.0);
        let callback_avg = kappa.kappa_callback_avg_s().unwrap_or(0.0).max(0.0);
        let calls = timing.cost_calls + timing.eval_calls + timing.efs_calls;
        kappa_phase.push(phase);
        kappa_calls.push(calls);
        kappa_resets.push(timing.slow_path_resets);
        kappa_skip_touches.push(timing.nfree_skip_row_touches);
        kappa_timings.push(timing);
        eprintln!(
            "[kappa-fast-ladder-touch] n={n} nfree_skip_row_touches={} (÷n={:.3}) calls={} per_callback_touches={:.3}",
            timing.nfree_skip_row_touches,
            timing.nfree_skip_row_touches as f64 / n as f64,
            calls,
            if calls == 0 {
                0.0
            } else {
                timing.nfree_skip_row_touches as f64 / calls as f64
            },
        );
        eprintln!(
            "[kappa-fast-ladder] {n:>9}  {:>10.4}  {:>12.4}  {:>12.4}  {:>9}  {:>9}  {:>6}  {:>6}  {:>6}  {:>9.4}  {:>9.4}  {:>9.4}",
            kappa.wall_s,
            timing.trial_total_s(),
            callback_avg,
            timing.slow_path_resets,
            timing.design_revision_delta,
            timing.cost_calls,
            timing.eval_calls,
            timing.efs_calls,
            timing.cost_total_s,
            timing.eval_total_s,
            timing.efs_total_s
        );
        eprintln!(
            "[kappa-fast-ladder-miss] n={n} shape={} value={} gradient={} penalty={} revision={} second_order={} other={}",
            timing.nfree_miss_shape,
            timing.nfree_miss_value,
            timing.nfree_miss_gradient,
            timing.nfree_miss_penalty,
            timing.nfree_miss_revision,
            timing.nfree_miss_second_order,
            timing.nfree_miss_other,
        );
    }

    let callback_avg: Vec<f64> = kappa_phase
        .iter()
        .zip(&kappa_calls)
        .map(|(&total, &calls)| {
            if calls == 0 {
                0.0
            } else {
                total / calls as f64
            }
        })
        .collect();
    let first_cb = callback_avg.first().copied().unwrap_or(0.0).max(1e-6);
    let last_cb = callback_avg.last().copied().unwrap_or(0.0).max(1e-6);
    let first_sum = kappa_phase.first().copied().unwrap_or(0.0).max(1e-3);
    let last_sum = kappa_phase.last().copied().unwrap_or(0.0).max(1e-3);
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 16
    let cb_ratio = last_cb / first_cb;
    eprintln!(
        "[kappa-fast-ladder] n grew {n_ratio:.0}× ; PER-CALLBACK avg grew {cb_ratio:.2}× \
         (n-independent ⇒ ~1×, n-linear ⇒ ~{n_ratio:.0}×) ; summed-total grew {:.2}× \
         (context only) — fast #1033 close-signal",
        last_sum / first_sum
    );
    // Reset soundness (n-independence of the exact-lane fallback COUNT): every
    // fallback must be the bounded, correctness-mandated #1264 basis-rotation
    // kind, and the count must stay under the n-independent cap. See
    // `NFREE_ROTATION_FALLBACK_CAP` for why this replaces the old
    // `resets_last <= resets_first + 1` flat-reset assumption.
    for (&n, timing) in ns.iter().zip(&kappa_timings) {
        assert_eq!(
            non_rotation_resets(timing),
            0,
            "[kappa-fast-ladder] n={n}: {} non-rotation skip miss(es) — the n-free skip fell \
             through for a reason other than a #1264 basis rotation",
            non_rotation_resets(timing),
        );
        assert!(
            timing.slow_path_resets <= NFREE_ROTATION_FALLBACK_CAP,
            "[kappa-fast-ladder] n={n}: slow_path_resets={} exceeds the bounded #1264 \
             basis-rotation cap ({NFREE_ROTATION_FALLBACK_CAP}) — the n-free skip is \
             re-entering the exact O(n) lane far more than the bounded rotation fallback allows",
            timing.slow_path_resets,
        );
    }
    // #1868 DETERMINISTIC n-independence gate (replaces the noisy wall-clock
    // `cb_ratio <= 8×` tripwire above, which needed a 320× n sweep to rise above
    // shared-node timing jitter and took ~2.2 h to run to failure). The
    // `nfree_skip_row_touches` counter is the exact integer count of length-`n`
    // row-element touches the Gaussian inner synthesis performed on the #1033
    // n-free κ-trial skip path. The architectural invariant (#1033: an in-window
    // trial touches only k×k objects) is precisely `touches == 0`, at every n:
    // the stale-row placeholders (η≡μ≡offset, z≡y, w) are shared O(1) from the
    // once-built frozen bundle, never re-materialised per callback. This is a
    // strictly STRONGER statement than the timing ratio (exact, not thresholded)
    // AND millisecond-cheap to verify — no large-n lever needed, because an
    // integer that must be 0 cannot hide an O(n) term behind a small constant.
    for (&n, &touches) in ns.iter().zip(&kappa_skip_touches) {
        assert_eq!(
            touches,
            0,
            "[kappa-fast-ladder] n={n}: nfree_skip_row_touches={touches} (≠0) — the #1033 \
             n-free κ-trial skip path is STILL materialising length-n row arrays per \
             callback (≈{:.1}× n), the #1868 O(n)-per-callback regression. Each in-window \
             trial must share the frozen row bundle O(1) and touch only k×k objects.",
            touches as f64 / n as f64,
        );
    }
    // Report-only wall-clock context (the OLD gate) — kept as a secondary signal
    // that the k-space trial cost is genuinely flat, but no longer asserted.
    eprintln!(
        "[kappa-fast-ladder] (report-only) PER-CALLBACK wall-clock grew {cb_ratio:.2}× \
         across {n_ratio:.0}× n"
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
                writeln!(f, "{msg}").ok();
                f.flush().ok();
            }
        }
    }
    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            use std::io::Write;
            f.flush().ok();
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
    log::set_logger(logger).ok();
    log::set_max_level(log::LevelFilter::Info);
}

#[test]
fn zzz_diag_n16000_reset_reasons() {
    install_nfree_reset_logger();
    let (aniso, bounds) = (false, (1e-2, 1e2));
    run_fit(1000, true, aniso, bounds).ok();
    let r = run_kappa_trial_seconds(16_000, aniso, bounds).unwrap();
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

#[test]
fn kappa_micro_2point_n_independence() {
    install_nfree_reset_logger();
    let (aniso, bounds) = (false, (1e-2, 1e2));
    let warm = run_fit(1000, true, aniso, bounds)
        .unwrap_or_else(|reason| panic!("[kappa-micro] warm iso-κ fit failed ({reason})"));
    eprintln!("[kappa-micro] warm-up primed caches in {:.4}s", warm.wall_s);

    let ns = [1_000usize, 2_000];
    let mut cb = Vec::new();
    let mut resets = Vec::new();
    let mut skip_touches = Vec::new();
    let mut timings = Vec::new();
    for &n in &ns {
        let kappa = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let timing = kappa.kappa_timing.unwrap();
        let calls = (timing.cost_calls + timing.eval_calls + timing.efs_calls).max(1);
        let per_cb = timing.trial_total_s().max(0.0) / calls as f64;
        cb.push(per_cb.max(1e-6));
        resets.push(timing.slow_path_resets);
        skip_touches.push(timing.nfree_skip_row_touches);
        timings.push(timing);
        eprintln!(
            "[kappa-micro] n={n:>5}  per_callback_s={per_cb:.5}  skip_row_touches={}  resets={}  \
             miss(shape/value/grad/pen/rev/2nd/oth)={}/{}/{}/{}/{}/{}/{}",
            timing.nfree_skip_row_touches,
            timing.slow_path_resets,
            timing.nfree_miss_shape,
            timing.nfree_miss_value,
            timing.nfree_miss_gradient,
            timing.nfree_miss_penalty,
            timing.nfree_miss_revision,
            timing.nfree_miss_second_order,
            timing.nfree_miss_other,
        );
    }
    let ratio = cb[1] / cb[0];
    eprintln!(
        "[kappa-micro] n grew 2× ; PER-CALLBACK wall grew {ratio:.2}× (report-only) ; \
         skip_row_touches {}→{} ; resets {}→{}",
        skip_touches[0], skip_touches[1], resets[0], resets[1]
    );
    // Reset soundness: the exact-lane fallbacks must be the bounded,
    // n-independent #1264 basis-rotation kind, not a genuine skip break.
    for (&n, timing) in ns.iter().zip(&timings) {
        assert_eq!(
            non_rotation_resets(timing),
            0,
            "[kappa-micro] n={n}: {} non-rotation skip miss(es) (shape/grad/pen/rev/2nd/other) — \
             the n-free skip fell through for a reason other than a #1264 basis rotation",
            non_rotation_resets(timing),
        );
        assert!(
            timing.slow_path_resets <= NFREE_ROTATION_FALLBACK_CAP,
            "[kappa-micro] n={n}: slow_path_resets={} exceeds the bounded #1264 \
             basis-rotation cap ({NFREE_ROTATION_FALLBACK_CAP}) — the n-free skip is \
             re-entering the exact O(n) lane far more than the bounded rotation fallback \
             allows (a disarmed skip resets on ≈every trial)",
            timing.slow_path_resets,
        );
    }
    // #1868 deterministic gate: zero length-n row touches per κ trial at BOTH n
    // (the wall-clock ratio above is now report-only — noisy at this micro
    // scale). A non-zero, n-scaling touch count is the O(n)-per-callback bug.
    for (&n, &touches) in ns.iter().zip(&skip_touches) {
        assert_eq!(
            touches,
            0,
            "[kappa-micro] n={n}: nfree_skip_row_touches={touches} (≠0, ≈{:.1}×n) — the \
             #1033 outer loop is still doing O(n) row work per trial (#1868)",
            touches as f64 / n as f64,
        );
    }
}

/// One rung of the #1033 n-independence acceptance gate at a fixed `n`.
///
/// This is the per-`n` body factored out of the former monolithic
/// `kappa_outer_loop_is_n_independent`, so each rung can be its own `#[test]`
/// case with its OWN nextest slow-timeout budget. Structurally the old
/// single-`#[test]` ran a warm fit plus THREE full iso-κ fits (n=1k/4k/16k)
/// serially; on a shared node the 16k rung alone could push the whole function
/// past nextest's 600s SIGKILL, so the DETERMINISTIC `touches == 0` assertion
/// was structurally unreachable in the standard shard. Splitting per-`n` does
/// NOT weaken anything: every hard assertion below was already applied inside a
/// PER-`n` loop (`non_rotation_resets == 0`, `slow_path_resets <= CAP`,
/// `touches == 0`, `calls > 0`) — there was NO asserted cross-`n` quantity. The
/// old cross-`n` per-callback wall-clock ratio was report-only `eprintln!`, and
/// is preserved (still report-only) in
/// `kappa_outer_loop_is_n_independent_fast_ladder`.
fn assert_kappa_n_free_at(n: usize) {
    // ISOTROPIC path (`aniso=false`, `aniso_log_scales=None`): the single
    // design-moving coordinate ψ=log κ on a Gaussian-identity fit — exactly the
    // `PsiGramTensor` eligibility (`coord_dim==1` ∧ Gaussian + identity). The
    // tensor auto-installs over the optimizer's ψ window, so every in-window
    // trial's `XᵀWX(ψ)`/`XᵀWz(ψ)` and ψ-gradient come from the k-space Chebyshev
    // representation rather than an O(n) design re-stream. The aniso path routes
    // the per-axis optimizer that bypasses the tensor entirely, so it would NOT
    // measure this lane. Now that the iso-1D κ outer converges across n
    // (#1053/#1066/#1069 fixed), this is the measurement path.
    let (aniso, bounds) = (false, (1e-2, 1e2));
    let warm = run_fit(1000, true, aniso, bounds).unwrap_or_else(|reason| {
        panic!(
            "[kappa-n-scaling] warm iso-κ fit failed ({reason}) — iso-1D κ \
             convergence is fixed (#1053/#1066/#1069); a failure here is a real \
             regression in the tensor-eligible isotropic length-scale optimizer"
        )
    });
    eprintln!(
        "[kappa-n-scaling] n={n} warm-up fit primed caches in {:.4}s",
        warm.wall_s
    );

    let kappa = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
    let timing = kappa.kappa_timing.unwrap();
    let callback_avg = kappa.kappa_callback_avg_s().unwrap_or(0.0).max(0.0);
    let calls = timing.cost_calls + timing.eval_calls + timing.efs_calls;
    eprintln!(
        "[kappa-n-scaling-touch] n={n} nfree_skip_row_touches={} (÷n={:.3}) calls={} per_callback_touches={:.3}",
        timing.nfree_skip_row_touches,
        timing.nfree_skip_row_touches as f64 / n as f64,
        calls,
        if calls == 0 {
            0.0
        } else {
            timing.nfree_skip_row_touches as f64 / calls as f64
        },
    );
    eprintln!(
        "[kappa-n-scaling] n={n:>9}  t_kappa={:>10.4}  kappa_sum={:>12.4}  callback={:>12.6}  \
         resets={:>4}  revs={:>4}  cost={:>4}  eval={:>4}  efs={:>4}",
        kappa.wall_s,
        timing.trial_total_s(),
        callback_avg,
        timing.slow_path_resets,
        timing.design_revision_delta,
        timing.cost_calls,
        timing.eval_calls,
        timing.efs_calls,
    );
    eprintln!(
        "[kappa-n-scaling-miss] n={n} shape={} value={} gradient={} penalty={} revision={} second_order={} other={}",
        timing.nfree_miss_shape,
        timing.nfree_miss_value,
        timing.nfree_miss_gradient,
        timing.nfree_miss_penalty,
        timing.nfree_miss_revision,
        timing.nfree_miss_second_order,
        timing.nfree_miss_other,
    );

    // SOUNDNESS GATE (n-independence of the exact-lane fallback COUNT): the
    // n-free skip must actually fire for essentially every trial. The only
    // permitted exact-lane fallbacks are the bounded, correctness-mandated #1264
    // basis-rotation crossings (`nfree_miss_value`) — an n-INDEPENDENT count
    // (observed 0–4 across n=1k–16k), NOT the per-callback O(n) synthesis (pinned
    // to zero by the `nfree_skip_row_touches` gate below). A fallback of any
    // OTHER kind, or a count above the n-independent cap, means the skip is not
    // firing.
    assert_eq!(
        non_rotation_resets(&timing),
        0,
        "n={n}: {} non-rotation skip miss(es) (shape/grad/pen/rev/2nd/other) — the #1033 \
         n-free skip fell through for a reason other than a #1264 basis rotation; the skip \
         logic is broken, not merely paying the bounded rotation fallback",
        non_rotation_resets(&timing),
    );
    assert!(
        timing.slow_path_resets <= NFREE_ROTATION_FALLBACK_CAP,
        "n={n}: slow_path_resets={} exceeds the bounded #1264 basis-rotation cap \
         ({NFREE_ROTATION_FALLBACK_CAP}) — every moved-ψ trial is re-entering the O(n) \
         reset_surface lane (a disarmed skip resets on ≈every κ trial), so the per-callback \
         cost cannot be n-independent",
        timing.slow_path_resets,
    );
    // #1868 DETERMINISTIC n-independence bar. `nfree_skip_row_touches` is the
    // exact integer count of length-`n` row-element touches the Gaussian inner
    // synthesis performed on the #1033 n-free κ-trial skip path. The
    // architectural invariant — "each in-window hyperparameter trial touches only
    // k×k objects" — is literally `touches == 0`, at every n. Together with the
    // slow-path-reset soundness gate above (which certifies the *design*
    // re-realization O(n) lane also stayed off), this pins BOTH O(n) lanes of the
    // κ trial to zero deterministically. A non-zero, n-scaling touch count is
    // exactly the #1868 O(n)-per-callback regression. This is exact (not
    // thresholded) and verifiable in milliseconds — no large-n lever required.
    assert!(
        calls > 0,
        "n={n}: no κ-trial callbacks were recorded — the n-independence gate would be \
         vacuous (the optimizer never entered the measured trial phase)"
    );
    assert_eq!(
        timing.nfree_skip_row_touches,
        0,
        "kappa outer-loop performed {} length-n row touches (≈{:.1}×n) on the \
         #1033 n-free κ-trial skip path at n={n} — the PsiGramTensor \
         sufficient-statistic lane is STILL doing O(n) work inside each trial callback \
         (the architectural invariant requires each hyperparameter trial to touch only \
         k×k objects; the stale-row placeholders must be shared O(1) from the frozen \
         row bundle, not re-materialised per callback). This is the #1868 regression.",
        timing.nfree_skip_row_touches,
        timing.nfree_skip_row_touches as f64 / n as f64,
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
#[test]
fn kappa_outer_loop_is_n_independent_n1000() {
    assert_kappa_n_free_at(1_000);
}

#[test]
fn kappa_outer_loop_is_n_independent_n4000() {
    assert_kappa_n_free_at(4_000);
}

#[test]
fn kappa_outer_loop_is_n_independent_n16000() {
    assert_kappa_n_free_at(16_000);
}

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

/// Poisson-log 1-D spatial fixture, mirroring `simulate_1d_gaussian` but with a
/// count response `y ~ Poisson(exp(0.6·sin(t)))`. Deterministic (a fixed LCG of
/// the row index, mapped to a Poisson draw by inverse-CDF) so the ladder is a
/// geometry/timing check, not a stochastic power test.
fn simulate_1d_poisson(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        let mu = (0.6 * t.sin()).exp();
        // Deterministic Poisson(mu) via inverse-CDF on a fixed per-row uniform.
        let u = (((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11) as f64)
            / ((1u64 << 53) as f64);
        let mut k = 0u32;
        let mut cdf = (-mu).exp();
        let mut p = cdf;
        while u > cdf && k < 100 {
            k += 1;
            p *= mu / k as f64;
            cdf += p;
        }
        y[i] = k as f64;
    }
    (x, y)
}

/// One Poisson-log κ fit, returning the internal κ-trial timing. Mirrors
/// `run_fit` but with the Poisson family + log link (the frozen-W GLM lane).
fn run_fit_poisson(n: usize, bounds: (f64, f64)) -> Result<FitTiming, String> {
    let (x, y) = simulate_1d_poisson(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 15,
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
        spec: spec_1d(false),
        family: LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
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
            Ok(FitTiming {
                wall_s: dt,
                kappa_timing: s.kappa_timing,
            })
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

/// #1033 GLM lane n-scaling: REPORT-ONLY measurement of the Poisson-log κ outer
/// loop across a 16× n sweep, with a SUPER-LINEAR catastrophe tripwire.
///
/// Reading the printed table: the GLM per-callback average is expected to grow
/// roughly with n (the residual O(n·k) design-realization floor — the GLM lane
/// is NOT n-free; see the module note above). What it must NOT do is grow with
/// n² — that would mean the frozen-W Gram lane stopped firing and every trial is
/// re-streaming the O(n·k²) Gram. The hard gate is therefore a generous
/// super-linear tripwire (≤ n_ratio^1.5), not an n-independence (≤8×) bar.
#[test]
fn kappa_glm_poisson_loop_n_scaling_report() {
    let bounds = (1e-2, 1e2);
    let warm = run_fit_poisson(1000, bounds);
    match &warm {
        Ok(t) => eprintln!(
            "[kappa-glm] warm Poisson κ fit primed caches in {:.4}s",
            t.wall_s
        ),
        Err(reason) => {
            // The GLM frozen-W lane is a best-effort accelerator; if this fixture
            // does not converge in the CI budget, report and return rather than
            // fail (the Gaussian ladders are the close gate, not this one).
            eprintln!("[kappa-glm] warm Poisson κ fit did not converge ({reason}); skipping");
            return;
        }
    }

    let ns = [1_000usize, 4_000, 16_000];
    let mut callback_avg = Vec::with_capacity(ns.len());
    eprintln!(
        "[kappa-glm] {:>9}  {:>10}  {:>12}  {:>9}  {:>9}",
        "n", "t_kappa_s", "callback_s", "resets", "calls"
    );
    for &n in &ns {
        let kappa = match run_fit_poisson(n, bounds) {
            Ok(k) if k.kappa_timing.is_some() => k,
            Ok(_) => {
                eprintln!("[kappa-glm] n={n}: κ optimizer reported no internal timing; skipping");
                return;
            }
            Err(reason) => {
                eprintln!("[kappa-glm] n={n}: fit failed ({reason}); skipping ladder");
                return;
            }
        };
        let timing = kappa.kappa_timing.unwrap();
        let calls = (timing.cost_calls + timing.eval_calls + timing.efs_calls).max(1);
        let per_cb = (timing.trial_total_s().max(0.0)) / calls as f64;
        callback_avg.push(per_cb.max(1e-6));
        eprintln!(
            "[kappa-glm] {n:>9}  {:>10.4}  {:>12.5}  {:>9}  {:>9}",
            kappa.wall_s, per_cb, timing.slow_path_resets, calls
        );
    }

    let first = callback_avg.first().copied().unwrap_or(0.0).max(1e-6);
    let last = callback_avg.last().copied().unwrap_or(0.0).max(1e-6);
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 16
    let cb_ratio = last / first;
    // Super-linear tripwire: the GLM lane is O(n·k) per trial (design realize), so
    // the per-callback grows ~n_ratio; a frozen-W Gram-lane miss reverts it to
    // O(n·k²) ⇒ ~n_ratio² growth. Gate at n_ratio^1.5 — comfortably above the
    // honest O(n) floor (and shared-node jitter) yet below the n² catastrophe.
    let tripwire = n_ratio.powf(1.5);
    eprintln!(
        "[kappa-glm] n grew {n_ratio:.0}× ; PER-CALLBACK grew {cb_ratio:.2}× \
         (GLM lane is O(n·k) by design ⇒ ~{n_ratio:.0}×; a frozen-W Gram-lane miss ⇒ \
         ~{:.0}× = O(n·k²)) ; super-linear tripwire = {tripwire:.1}×",
        n_ratio * n_ratio
    );
    assert!(
        cb_ratio <= tripwire,
        "[kappa-glm] Poisson κ per-callback grew {cb_ratio:.2}× across a {n_ratio:.0}× n \
         increase — past the {tripwire:.1}× super-linear tripwire. The frozen-W GLM Gram \
         lane is no longer firing (every trial re-streams the O(n·k²) Gram instead of \
         serving it n-free from the certified FrozenWeightGramTensor)"
    );
}
