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
//! This harness isolates that κ-phase without any internal A/B switch (the
//! tensor auto-installs; there is no off-flag, by design). For each n it times:
//!   * `t_kappa`  — a fit with the κ outer loop ENABLED (several outer iters), and
//!   * `t_single` — the same fit with the loop DISABLED (one length-scale, one fit).
//! The difference `t_kappa - t_single` is the marginal cost of the κ search on
//! top of one ordinary fit, i.e. the κ-phase. If that marginal cost is
//! n-independent, the ratio across a 16× sweep in n is ~1, not ~16.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* I
//! read from the printed table — the only hard assertion is a catastrophe guard
//! (the κ-phase must not blow up super-linearly by an order of magnitude across
//! the sweep), which is a real tripwire, not a calibrated timing bound.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu},
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
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_1d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: 1.0,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    // None routes the isotropic analytic κ optimizer; Some(_)
                    // routes the per-axis (anisotropic) optimizer even for a
                    // single axis — the discriminator under test.
                    aniso_log_scales: if aniso { Some(vec![0.0]) } else { None },
                    nullspace_shrinkage_survived: None,
                },
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

/// Outcome of one fit attempt: either the wall-clock seconds (converged) or the
/// failure reason string (so the diagnostic can tabulate instead of aborting).
fn run_fit(n: usize, kappa_enabled: bool, aniso: bool, bounds: (f64, f64)) -> Result<f64, String> {
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
            Ok(dt)
        }
        _ => Err("expected Standard fit result".to_string()),
    }
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
    eprintln!("[kappa-diag] n={n}  (1-D Matérn ν=5/2, Gaussian-identity, single penalty)");
    let mut outcomes = Vec::new();
    for (label, aniso, bounds) in configs {
        let r = run_fit(n, true, aniso, bounds);
        match &r {
            Ok(dt) => eprintln!("[kappa-diag] {label}: CONVERGED in {dt:.3}s"),
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
/// Matérn Gaussian fixture (gentle `y=sin(t)`, 12 centers, single penalty, tight
/// bounds). n=600 converges; earlier runs showed n=1000 failing with a stuck
/// `grad_norm≈1.9e3`. This sweep brackets the transition so the defect report
/// carries an exact reproducer. Report-only (the printed sweep is the
/// deliverable); the companion measurement stays gated until it is fixed.
#[test]
fn kappa_iso_1d_n_threshold_sweep() {
    let bounds = (1e-2, 1e2);
    eprintln!("[kappa-nthresh] iso-1D Matérn ν=5/2, Gaussian, single penalty, bounds={bounds:?}");
    for &n in &[600usize, 800, 1000, 1200] {
        match run_fit(n, true, false, bounds) {
            Ok(dt) => eprintln!("[kappa-nthresh] n={n:>5}: CONVERGED in {dt:.1}s"),
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
    eprintln!("[kappa-fast-ladder] warm-up fit primed caches in {warm:.4}s");

    // Small ladder: 1k → 16k (16×). Enough to read the slope; ~2–3 min total.
    let ns = [1_000usize, 4_000, 16_000];
    let mut kappa_phase = Vec::with_capacity(ns.len());
    eprintln!(
        "[kappa-fast-ladder] {:>9}  {:>10}  {:>10}  {:>12}",
        "n", "t_kappa_s", "t_single_s", "kappa_phase_s"
    );
    for &n in &ns {
        let t_kappa = run_fit(n, true, aniso, bounds)
            .unwrap()
            .min(run_fit(n, true, aniso, bounds).unwrap());
        let t_single = run_fit(n, false, aniso, bounds)
            .unwrap()
            .min(run_fit(n, false, aniso, bounds).unwrap());
        let phase = (t_kappa - t_single).max(0.0);
        kappa_phase.push(phase);
        eprintln!("[kappa-fast-ladder] {n:>9}  {t_kappa:>10.4}  {t_single:>10.4}  {phase:>12.4}");
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
    eprintln!("[kappa-n-scaling] warm-up fit primed caches in {warm:.4}s");

    // #1033 acceptance sweep: 1e3 → 2.56e5 (256×). `t_single` (κ loop OFF, one
    // length-scale, one fit) absorbs the one-time O(n) tensor build + the final
    // O(n) PIRLS assembly; `t_single` therefore scales with n. The MARGINAL
    // `t_kappa - t_single` isolates the cost of the OUTER κ-trial loop on top of
    // one ordinary fit — the object the issue requires to be n-INDEPENDENT. If
    // the per-trial design realization were still O(n) it would grow ~256× here;
    // the PsiGramTensor sufficient-statistic lane keeps it flat (the only n-work
    // left is the single build, which lands in `t_single`, not the marginal).
    let ns = [1_000usize, 4_000, 16_000, 64_000, 256_000, 320_000];
    let mut kappa_phase = Vec::with_capacity(ns.len());

    eprintln!(
        "[kappa-n-scaling] {:>9}  {:>10}  {:>10}  {:>12}",
        "n", "t_kappa_s", "t_single_s", "kappa_phase_s"
    );
    for &n in &ns {
        // Best-of-two to suppress shared-cluster wall-clock noise.
        let t_kappa = run_fit(n, true, aniso, bounds)
            .unwrap()
            .min(run_fit(n, true, aniso, bounds).unwrap());
        let t_single = run_fit(n, false, aniso, bounds)
            .unwrap()
            .min(run_fit(n, false, aniso, bounds).unwrap());
        let phase = (t_kappa - t_single).max(0.0);
        kappa_phase.push(phase);
        eprintln!("[kappa-n-scaling] {n:>9}  {t_kappa:>10.4}  {t_single:>10.4}  {phase:>12.4}");
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
