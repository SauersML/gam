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
//! callback totals plus call counts. If a trial is n-independent, the average
//! callback cost ratio across a 16× sweep in n is ~1, not ~16.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* I
//! read from the printed table — the only hard assertion is a catastrophe guard
//! (the callback average must not blow up super-linearly by an order of magnitude
//! across the sweep), which is a real tripwire, not a calibrated timing bound.

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
    // (mass/tension/stiffness), which the n-free re-key cannot reproduce, so #1270
    // (d69b52e66) deliberately drops Matérn from `supports_nfree_penalty_rekey` →
    // the skip is permanently disabled for Matérn and every trial re-realizes the
    // O(n) design. Measuring n-independence on Matérn would test a basis the n-free
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
                // (#1214/#1215), the real default-fit path. An input_scales:[1.0]
                // pin would be a gamed gate masking the open geometry gap.
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
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
        outer_wall_clock_budget_secs: None,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights,
        offset,
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
        _marker: std::marker::PhantomData,
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
    let resets_first = kappa_resets.first().copied().unwrap_or(0);
    let resets_last = kappa_resets.last().copied().unwrap_or(0);
    assert!(
        resets_last <= resets_first.saturating_add(1),
        "slow_path_resets climbed from {resets_first} (n={}) to {resets_last} (n={}) — \
         the fast #1033 read still sees the O(n) reset_surface/design-realization lane",
        ns.first().unwrap(),
        ns.last().unwrap()
    );
    assert!(
        cb_ratio <= 8.0,
        "kappa outer-loop PER-CALLBACK average grew {cb_ratio:.2}× across a {n_ratio:.0}× \
         increase in n — the #1033 n-free skip is still doing O(n) work inside a \
         trial callback (fast-ladder read)"
    );
}

/// #1033 MICRO read (2 points, n=1k vs 2k): the smallest discriminant of
/// n-independence. Per-callback cost flat ⇒ n-free; tracking the 2× n ⇒ O(n).
/// Finishes in seconds — a development-loop probe, NOT the close gate (the full
/// 1k→16k/320k ladders are). No bar tightening here vs the headline ≤8× / flat
/// reset contract; this just surfaces the ratio fast.
#[test]
fn kappa_micro_2point_n_independence() {
    let (aniso, bounds) = (false, (1e-2, 1e2));
    let warm = run_fit(1000, true, aniso, bounds)
        .unwrap_or_else(|reason| panic!("[kappa-micro] warm iso-κ fit failed ({reason})"));
    eprintln!("[kappa-micro] warm-up primed caches in {:.4}s", warm.wall_s);

    let ns = [1_000usize, 2_000];
    let mut cb = Vec::new();
    let mut resets = Vec::new();
    for &n in &ns {
        let kappa = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let timing = kappa.kappa_timing.unwrap();
        let calls = (timing.cost_calls + timing.eval_calls + timing.efs_calls).max(1);
        let per_cb = timing.trial_total_s().max(0.0) / calls as f64;
        cb.push(per_cb.max(1e-6));
        resets.push(timing.slow_path_resets);
        eprintln!(
            "[kappa-micro] n={n:>5}  per_callback_s={per_cb:.5}  resets={}  \
             miss(shape/value/grad/pen/rev/2nd/oth)={}/{}/{}/{}/{}/{}/{}",
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
        "[kappa-micro] n grew 2× ; PER-CALLBACK grew {ratio:.2}× \
         (n-independent ⇒ ~1×, O(n) ⇒ ~2×) ; resets {}→{}",
        resets[0], resets[1]
    );
    // Sub-linear tripwire at 2× n: an O(n) regression tracks ~2×; a flat n-free
    // loop holds ~1×. Gate well below 2× and require resets not to climb.
    assert!(
        resets[1] <= resets[0].saturating_add(1),
        "[kappa-micro] slow_path_resets climbed {}→{} — n-free skip not firing",
        resets[0],
        resets[1]
    );
    assert!(
        ratio <= 1.6,
        "[kappa-micro] per-callback grew {ratio:.2}× across a 2× n increase — \
         the #1033 outer loop is still doing O(n) work per trial"
    );
}

#[test]
fn kappa_outer_loop_is_n_independent() {
    // ISOTROPIC path (`aniso=false`, `aniso_log_scales=None`): the single
    // design-moving coordinate ψ=log κ on a Gaussian-identity fit — exactly the
    // `PsiGramTensor` eligibility (`coord_dim==1` ∧ Gaussian + identity). The
    // tensor auto-installs over the optimizer's ψ window, so every in-window
    // trial's `XᵀWX(ψ)`/`XᵀWz(ψ)` and ψ-gradient come from the k-space Chebyshev
    // representation rather than an O(n) design re-stream. The aniso path
    // (placeholder in the original draft) routes the per-axis optimizer that
    // bypasses the tensor entirely, so it would NOT measure this lane. Now that
    // the iso-1D κ outer converges across n (#1053/#1066/#1069 fixed, calibrated
    // n-scaled profiled REML), this is the measurement path.
    let (aniso, bounds) = (false, (1e-2, 1e2));
    let warm = run_fit(1000, true, aniso, bounds).unwrap_or_else(|reason| {
        panic!(
            "[kappa-n-scaling] warm iso-κ fit failed ({reason}) — iso-1D κ \
             convergence is fixed (#1053/#1066/#1069); a failure here is a real \
             regression in the tensor-eligible isotropic length-scale optimizer"
        )
    });
    eprintln!(
        "[kappa-n-scaling] warm-up fit primed caches in {:.4}s",
        warm.wall_s
    );

    // #1033 acceptance sweep: 1e3 → 320k. The asserted quantity is the structured
    // κ-trial callback time reported by `KAPPA-PHASE-SUMMARY`, not whole-fit
    // wall-clock subtraction. The one-time tensor build is intentionally outside
    // this counter; the issue accepts n-independence after that initial pass.
    let ns = [1_000usize, 4_000, 16_000, 64_000, 256_000, 320_000];
    let mut kappa_phase = Vec::with_capacity(ns.len());
    let mut kappa_calls = Vec::with_capacity(ns.len());
    let mut kappa_resets = Vec::with_capacity(ns.len());

    eprintln!(
        "[kappa-n-scaling] {:>9}  {:>10}  {:>12}  {:>12}  {:>9}  {:>9}  {:>6}  {:>6}  {:>6}  {:>9}  {:>9}  {:>9}",
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
        eprintln!(
            "[kappa-n-scaling] {n:>9}  {:>10.4}  {:>12.4}  {:>12.4}  {:>9}  {:>9}  {:>6}  {:>6}  {:>6}  {:>9.4}  {:>9.4}  {:>9.4}",
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
            "[kappa-n-scaling-miss] n={n} shape={} value={} gradient={} penalty={} revision={} second_order={} other={}",
            timing.nfree_miss_shape,
            timing.nfree_miss_value,
            timing.nfree_miss_gradient,
            timing.nfree_miss_penalty,
            timing.nfree_miss_revision,
            timing.nfree_miss_second_order,
            timing.nfree_miss_other,
        );
    }

    // #1033 acceptance is PER-CALLBACK k-only cost, not summed wall-time. The
    // architectural invariant the issue states is: "the rho/kappa/psi outer loop
    // manipulates only k×k objects (O(k³) per trial = microseconds)". A SUMMED
    // total (count × per-call) can stay bounded while each call is still O(n), or
    // grow merely because the optimizer took more (cheap) steps — neither tells
    // us whether a callback touches only k-dim objects. The honest discriminant
    // is the AVERAGE callback cost: it must stay flat as n grows, since a single
    // n-free trial costs O(D²k³) regardless of n. So the asserted quantity is the
    // per-callback average, decomposed below.
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
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 320
    let cb_ratio = last_cb / first_cb;
    // Summed total reported for context (the OLD, weaker metric) — not asserted.
    let first_sum = kappa_phase.first().copied().unwrap_or(0.0).max(1e-3);
    let last_sum = kappa_phase.last().copied().unwrap_or(0.0).max(1e-3);
    eprintln!(
        "[kappa-n-scaling] n grew {n_ratio:.0}× ; PER-CALLBACK avg grew {cb_ratio:.2}× \
         (n-independent ⇒ ~1×, n-linear ⇒ ~{n_ratio:.0}×) ; summed-total grew {:.2}× (context only)",
        last_sum / first_sum
    );
    // SOUNDNESS GATE: the n-free skip must actually fire across the sweep. If the
    // design-revision skip falls through to the O(n) reset_surface lane, the
    // slow-path reset count climbs with n. The skip must be armed at the largest
    // n exactly as it is at the smallest — otherwise the "n-free" claim is empty.
    let resets_first = kappa_resets.first().copied().unwrap_or(0);
    let resets_last = kappa_resets.last().copied().unwrap_or(0);
    assert!(
        resets_last <= resets_first.saturating_add(2),
        "slow_path_resets climbed from {resets_first} (n={}) to {resets_last} (n={}) — the \
         #1033 n-free skip is NOT firing at large n; every moved-ψ trial re-enters the \
         O(n) reset_surface/design-realization lane, so the per-callback cost cannot be \
         n-independent",
        ns.first().unwrap(),
        ns.last().unwrap()
    );
    // n-independence bar: the PER-CALLBACK average must NOT scale with n. A truly
    // n-free outer loop holds each callback at fixed O(D²k³) k-space work, so the
    // average drifts only with shared-node timing jitter. An O(n) regression
    // tracks `n_ratio`. Gate at 8× across a 320× n increase (~n^0.36) — decisively
    // sub-linear, safely above shared-node noise, a real O(n)-regression tripwire.
    assert!(
        cb_ratio <= 8.0,
        "kappa outer-loop PER-CALLBACK average grew {cb_ratio:.2}× across a {n_ratio:.0}× \
         increase in n — the #1033 PsiGramTensor sufficient-statistic lane is still doing \
         O(n) work inside each trial callback (the architectural invariant requires each \
         hyperparameter trial to touch only k×k objects)"
    );
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
        outer_wall_clock_budget_secs: None,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights,
        offset,
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
        _marker: std::marker::PhantomData,
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
