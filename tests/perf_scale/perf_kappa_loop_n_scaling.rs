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
//! This harness isolates that κ-phase through the optimizer's structured
//! `KAPPA-PHASE-SUMMARY` counters. Those counters start after the one-time tensor
//! setup / cold realization pass and sum only cost/eval/EFS trial callbacks. If
//! that trial cost is n-independent, the ratio across a 16× sweep in n is ~1, not
//! ~16.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* I
//! read from the printed table — the only hard assertion is a catastrophe guard
//! (the κ-phase must not blow up super-linearly by an order of magnitude across
//! the sweep), which is a real tripwire, not a calibrated timing bound.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
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
    kappa_trial_s: Option<f64>,
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
                kappa_trial_s: s.kappa_timing.map(|timing| timing.trial_total_s()),
            })
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

fn run_kappa_trial_seconds(n: usize, aniso: bool, bounds: (f64, f64)) -> Result<FitTiming, String> {
    let timing = run_fit(n, true, aniso, bounds)?;
    if timing.kappa_trial_s.is_none() {
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
/// marginal κ-phase measurement on a SMALL n-ladder (1k → 16k) that completes in
/// ~2–3 min, so the n-free skip's flat-vs-linear behaviour can be read inside an
/// iteration loop without waiting on the 320k sweep (which walls the 1:30 slot).
///
/// The discriminant is unambiguous at this scale: across a 16× n increase an
/// O(n) per-trial regression tracks ~16×, while a truly n-free outer loop holds
/// the marginal κ-phase ~flat (drifting only with the fixed O(D²k²) trial cost
/// and shared-node timing jitter). The same ≤8× bar as the headline applies — at
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
    eprintln!(
        "[kappa-fast-ladder] {:>9}  {:>10}  {:>10}  {:>12}",
        "n", "t_kappa_s", "t_single_s", "kappa_phase_s"
    );
    for &n in &ns {
        let kappa_a = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let kappa_b = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let kappa = if kappa_a.kappa_trial_s.unwrap() <= kappa_b.kappa_trial_s.unwrap() {
            kappa_a
        } else {
            kappa_b
        };
        let t_single = run_fit(n, false, aniso, bounds)
            .unwrap()
            .wall_s
            .min(run_fit(n, false, aniso, bounds).unwrap().wall_s);
        let phase = kappa.kappa_trial_s.unwrap().max(0.0);
        kappa_phase.push(phase);
        eprintln!(
            "[kappa-fast-ladder] {n:>9}  {:>10.4}  {t_single:>10.4}  {phase:>12.4}",
            kappa.wall_s
        );
    }

    let first = kappa_phase.first().copied().unwrap_or(0.0).max(1e-3);
    let last = kappa_phase.last().copied().unwrap_or(0.0).max(1e-3);
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 16
    let phase_ratio = last / first;
    eprintln!(
        "[kappa-fast-ladder] n grew {n_ratio:.0}× ; kappa-phase grew {phase_ratio:.2}× \
         (n-independent ⇒ ~1×, n-linear ⇒ ~{n_ratio:.0}×) — fast #1033 close-signal"
    );
    assert!(
        phase_ratio <= 8.0,
        "kappa outer-loop phase grew {phase_ratio:.2}× across a {n_ratio:.0}× \
         increase in n — the #1033 n-free skip is still falling to an O(n) \
         per-trial pass across the reduced-basis rotation (fast-ladder read)"
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

    eprintln!(
        "[kappa-n-scaling] {:>9}  {:>10}  {:>10}  {:>12}",
        "n", "t_kappa_s", "t_single_s", "kappa_phase_s"
    );
    for &n in &ns {
        // Best-of-two to suppress shared-cluster wall-clock noise.
        let kappa_a = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let kappa_b = run_kappa_trial_seconds(n, aniso, bounds).unwrap();
        let kappa = if kappa_a.kappa_trial_s.unwrap() <= kappa_b.kappa_trial_s.unwrap() {
            kappa_a
        } else {
            kappa_b
        };
        let t_single = run_fit(n, false, aniso, bounds)
            .unwrap()
            .wall_s
            .min(run_fit(n, false, aniso, bounds).unwrap().wall_s);
        let phase = kappa.kappa_trial_s.unwrap().max(0.0);
        kappa_phase.push(phase);
        eprintln!(
            "[kappa-n-scaling] {n:>9}  {:>10.4}  {t_single:>10.4}  {phase:>12.4}",
            kappa.wall_s
        );
    }

    let first = kappa_phase.first().copied().unwrap_or(0.0).max(1e-3);
    let last = kappa_phase.last().copied().unwrap_or(0.0).max(1e-3);
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 256
    let phase_ratio = last / first;
    eprintln!(
        "[kappa-n-scaling] n grew {n_ratio:.0}× ; kappa-phase grew {phase_ratio:.2}× \
         (n-independent ⇒ ~1×, n-linear ⇒ ~{n_ratio:.0}×)"
    );
    // n-independence bar: the marginal κ-phase must NOT scale with n. A truly
    // n-free outer loop holds the marginal ~flat (ratio ~1, drifting only with
    // the fixed O(D²k²) trial cost and timing noise); an O(n) regression would
    // track `n_ratio`. Gate at a generous absolute ceiling well below linear —
    // 8× across a 256× n increase is ~n^0.37, still decisively sub-linear and
    // safely above shared-node timing jitter, so this is a real O(n)-regression
    // tripwire rather than a calibrated timing bound.
    assert!(
        phase_ratio <= 8.0,
        "kappa outer-loop phase grew {phase_ratio:.2}× across a {n_ratio:.0}× \
         increase in n — the #1033 PsiGramTensor sufficient-statistic lane \
         regressed to an O(n) per-trial pass"
    );
}
