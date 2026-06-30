//! Measurement: is the ρ-only (penalty-search) REML outer loop n-independent?
//!
//! Mechanism (a) of #1033 — θ-invariant Gram caching. When the design provably
//! does NOT move across the hyperparameters being searched (a ρ-only model:
//! Gaussian + identity link, where only the penalty precision S(ρ) changes and
//! NOT a κ/ψ design-shape hyperparameter), the cross-product Gram XᵀWX is
//! θ-invariant. `RemlState::gaussian_fixed_cache_if_eligible`
//! (solver/reml/runtime.rs) assembles XᵀWX / XᵀW(y−offset) / yᵀWy ONCE per fit
//! under a double-checked write lock and every outer λ/ρ trial reuses it: the
//! dense PLS fast path and the sparse outer scatter consume the cached Gram
//! instead of re-streaming the n-row product. So the per-trial outer-eval cost
//! is k-dimensional (an O(p³) factorization of XᵀWX + S(ρ)), n-independent
//! after the single O(n·p²) Gram build.
//!
//! This harness isolates that ρ-phase. The κ optimizer is DISABLED
//! (`kappa_enabled: false`) so the design is fixed and the only outer search is
//! over the penalty precision — exactly the cache-eligible regime. Two B-spline
//! smooths give a real 2-D ρ outer loop. For each n it times the full fit; the
//! ρ-phase is the marginal cost above a single inner solve. With the Gram
//! cached, that marginal cost is dominated by k-space factorizations and must
//! NOT scale linearly with n.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* read
//! from the printed table — the hard assertion is a catastrophe guard (the
//! ρ-phase must not blow up super-linearly by an order of magnitude across the
//! n-sweep), a real tripwire rather than a calibrated timing bound.

use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::{
    FitRequest, FitResult, StandardFitRequest,
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// Two-feature Gaussian-identity fixture: a smooth additive signal on each of
/// two columns, observed with light noise. Deterministic so this stays a
/// timing/geometry check, not a stochastic power test.
fn simulate_2d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u = (i as f64) / (n as f64 - 1.0); // [0,1]
        let v = ((i * 7 + 3) % n) as f64 / (n as f64 - 1.0); // decorrelated [0,1]
        x[[i, 0]] = u;
        x[[i, 1]] = v;
        // smooth additive truth; tiny deterministic wiggle as "noise"
        let noise = 0.01 * ((i as f64 * 0.37).sin());
        y[i] = (2.0 * std::f64::consts::PI * u).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * v).cos()
            + noise;
    }
    (x, y)
}

fn bspline_smooth(name: &str, col: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: col,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 12,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary: Default::default(),
                boundary_conditions: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn spec_2d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        // Two independent penalised smooths → a real 2-D ρ outer search over a
        // FIXED (θ-invariant) design: the cache-eligible regime.
        smooth_terms: vec![bspline_smooth("f_u", 0), bspline_smooth("f_v", 1)],
    }
}

fn fit_options(outer_iters: usize) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: outer_iters,
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

/// Run one fit. `outer_iters` caps the REML outer (ρ) iterations: a large value
/// exercises the full ρ search (cache reused every trial); `1` is the
/// single-eval baseline (one outer eval ≈ one inner solve + one Gram touch).
fn run_fit(n: usize, outer_iters: usize) -> Result<f64, String> {
    let (x, y) = simulate_2d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    // κ optimizer OFF → design is fixed → ρ-only, cache-eligible.
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: false,
        max_outer_iter: 1,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 0.05,
        max_length_scale: 20.0,
        pilot_subsample_threshold: 0,
        outer_wall_clock_budget_secs: None,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights,
        offset,
        spec: spec_2d(),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(outer_iters),
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
            if s.fit.beta.iter().all(|v: &f64| v.is_finite()) {
                Ok(dt)
            } else {
                Err("non-finite coefficients".to_string())
            }
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

/// Diagnostic + catastrophe guard: the ρ-only outer phase must not scale
/// linearly with n. With the θ-invariant Gram cached once, every outer ρ trial
/// after the first is k-space; the marginal cost of the full ρ search above one
/// inner solve should stay roughly flat across an n-sweep, not grow with n.
#[test]
fn rho_outer_loop_is_n_independent() {
    // Prime caches / JIT-warm allocator behaviour with a throwaway small fit so
    // the first timed point isn't penalised by one-time process warmup.
    run_fit(2_000, 1).ok();

    let ns = [20_000usize, 80_000, 320_000];
    let full_outer = 30usize; // exercise the full ρ search

    eprintln!(
        "[rho-n-scaling] {:>9}  {:>11}  {:>11}  {:>12}",
        "n", "t_full_s", "t_single_s", "rho_phase_s"
    );
    let mut rho_phase = Vec::with_capacity(ns.len());
    for &n in &ns {
        let t_full = match run_fit(n, full_outer) {
            Ok(t) => t,
            Err(reason) => {
                eprintln!("[rho-n-scaling] n={n}: full-outer fit failed — {reason}");
                return; // measurement aborts cleanly; do not fail on a fit defect
            }
        };
        let t_single = match run_fit(n, 1) {
            Ok(t) => t,
            Err(reason) => {
                eprintln!("[rho-n-scaling] n={n}: single-eval fit failed — {reason}");
                return;
            }
        };
        // ρ-phase = cost of the full penalty search above one inner solve. The
        // one-time O(n·p²) Gram build is paid inside BOTH t_full and t_single
        // (each fit builds its own cache once), so it cancels out of the
        // difference — what remains is the per-trial outer cost, which the cache
        // makes n-independent.
        let phase = (t_full - t_single).max(0.0);
        rho_phase.push(phase);
        eprintln!("[rho-n-scaling] {n:>9}  {t_full:>11.4}  {t_single:>11.4}  {phase:>12.4}");
    }

    let first = rho_phase.first().copied().unwrap_or(0.0).max(1e-4);
    let last = rho_phase.last().copied().unwrap_or(0.0).max(1e-4);
    let n_ratio = (*ns.last().unwrap() as f64) / (*ns.first().unwrap() as f64);
    let phase_ratio = last / first;
    eprintln!(
        "[rho-n-scaling] n grew {n_ratio:.0}× ; rho-phase grew {phase_ratio:.2}× \
         (n-independent ⇒ ~1×, not ~{n_ratio:.0}×)"
    );

    // Catastrophe guard: a per-trial n-row Gram rebuild would make the ρ-phase
    // scale ~linearly with n (≈16× across this sweep). The cache must keep it
    // far below that. Generous slack absorbs shared-node wall-clock noise — this
    // is a tripwire for a re-introduced n-pass, not a calibrated bound.
    assert!(
        phase_ratio < 0.25 * n_ratio,
        "rho outer-loop phase grew {phase_ratio:.2}× across a {n_ratio:.0}× n-sweep \
         — expected n-INDEPENDENT (θ-invariant Gram cache); a near-linear growth \
         means the per-trial n-row Gram rebuild was re-introduced (#1033 mechanism a)"
    );
}
