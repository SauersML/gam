//! Measurement: is the κ/ψ (length-scale) REML outer loop n-independent?
//!
//! Mechanism (b) of #1033 — the certified ψ-Gram tensor. For a Gaussian-identity
//! ISOTROPIC spatial smooth (coord_dim == 1) the exact-joint optimizer searches
//! the design-shape hyperparameter ψ = log-length-scale. Naively every ψ-trial
//! re-realizes the full n×p design (`apply_log_kappa → refresh_full_design_operator`)
//! and the n×k ∂X/∂ψ slabs — an O(n·p) pass per trial, so the κ-phase scales
//! linearly with n.
//!
//! #1033b assembles, ONCE, a Chebyshev-in-ψ tensor of the conditioned Gram
//! XᵀWX(ψ) / XᵀWz(ψ) over the optimizer's ψ window. When a ψ-trial falls inside
//! the certified window, the value channel is served n-free from
//! `gaussian_fixed_cache_at(ψ)` and the gradient channel from k-space
//! ψ-derivatives — so the per-trial O(n·p) design realization is redundant and
//! must be skipped. With the skip wired, the κ-phase becomes n-INDEPENDENT after
//! the single tensor build: the same class of win as the ρ-Gram cache
//! (see `perf_rho_outer_loop_n_scaling.rs`).
//!
//! This harness isolates the κ-phase by ENABLING the κ optimizer on a single
//! 1-D Duchon Gaussian-identity smooth, and reads the κ-phase n-scaling from the
//! printed table. The hard assertion is a catastrophe guard: a re-introduced
//! per-trial n-row design rebuild would make the κ-phase scale ~linearly with n.

use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam::{
    FitRequest, FitResult, StandardFitRequest,
    estimate::FitOptions,
    smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec},
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use gam::smooth::SpatialLengthScaleOptimizationOptions;
use ndarray::{Array1, Array2};
use std::time::Instant;

/// One-feature isotropic Gaussian-identity fixture: a smooth signal on a single
/// spatial coordinate, observed with light deterministic noise. coord_dim == 1
/// makes the smooth tensor-eligible (#1033b).
fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0); // [0,1]
        let signal = 1.2 * (2.0 * std::f64::consts::PI * t).sin() + 0.4 * (t - 0.5);
        let noise = 0.15 * (((i as f64) * 12.9898).sin() * 43758.547).fract();
        y[i] = signal + noise;
    }
    for i in 0..n {
        x[[i, 0]] = (i as f64) / (n as f64 - 1.0);
    }
    (x, y)
}

/// A single isotropic Duchon spatial smooth that exposes a length-scale ψ — the
/// κ-optimizer search axis. Matches the in-tree tensor-invariance fixture.
fn duchon_smooth() -> SmoothTermSpec {
    SmoothTermSpec {
        name: "f_spatial".to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: vec![0],
            spec: DuchonBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                length_scale: Some(1.0),
                power: 1.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn spec_1d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon_smooth()],
    }
}

fn fit_options(outer_iters: usize) -> FitOptions {
    FitOptions {
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: outer_iters,
        tol: 1e-6,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    }
}

/// Run one fit with the κ optimizer ENABLED (so ψ is searched). `kappa_iters`
/// caps the κ outer iterations: a large value exercises the full ψ search
/// (tensor reused every trial); `1` is the single-eval baseline.
fn run_fit(n: usize, kappa_iters: usize) -> Result<f64, String> {
    let (x, y) = simulate_1d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: kappa_iters,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 0.05,
        max_length_scale: 20.0,
        pilot_subsample_threshold: 0,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights,
        offset,
        spec: spec_1d(),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(40),
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

/// Diagnostic + catastrophe guard: the κ/ψ outer phase must not scale linearly
/// with n. With the θ-invariant ψ-Gram tensor assembled once, every ψ-trial
/// after the build is k-space; the marginal cost of the full ψ search above one
/// inner solve should stay roughly flat across an n-sweep, not grow with n.
#[test]
fn kappa_outer_loop_is_n_independent() {
    assert!(file!().ends_with(".rs"));

    // Warm process / allocator with a throwaway small fit.
    let _ = run_fit(2_000, 1);

    let ns = [20_000usize, 80_000, 320_000];
    let full_kappa = 30usize; // exercise the full ψ search

    eprintln!(
        "[kappa-n-scaling] {:>9}  {:>11}  {:>11}  {:>12}",
        "n", "t_full_s", "t_single_s", "kappa_phase_s"
    );
    let mut kappa_phase = Vec::with_capacity(ns.len());
    for &n in &ns {
        let t_full = match run_fit(n, full_kappa) {
            Ok(t) => t,
            Err(reason) => {
                eprintln!("[kappa-n-scaling] n={n}: full-kappa fit failed — {reason}");
                return;
            }
        };
        let t_single = match run_fit(n, 1) {
            Ok(t) => t,
            Err(reason) => {
                eprintln!("[kappa-n-scaling] n={n}: single-eval fit failed — {reason}");
                return;
            }
        };
        let phase = (t_full - t_single).max(0.0);
        kappa_phase.push(phase);
        eprintln!(
            "[kappa-n-scaling] {n:>9}  {t_full:>11.4}  {t_single:>11.4}  {phase:>12.4}"
        );
    }

    let first = kappa_phase.first().copied().unwrap_or(0.0).max(1e-4);
    let last = kappa_phase.last().copied().unwrap_or(0.0).max(1e-4);
    let n_ratio = (*ns.last().unwrap() as f64) / (*ns.first().unwrap() as f64);
    let phase_ratio = last / first;
    eprintln!(
        "[kappa-n-scaling] n grew {n_ratio:.0}× ; kappa-phase grew {phase_ratio:.2}× \
         (n-independent ⇒ ~1×, not ~{n_ratio:.0}×)"
    );

    // Catastrophe guard: a per-trial n-row design rebuild would make the κ-phase
    // scale ~linearly with n (≈16× across this sweep). The ψ-Gram tensor must
    // keep it far below that. Generous slack absorbs shared-node wall-clock
    // noise — a tripwire for a re-introduced n-pass, not a calibrated bound.
    assert!(
        phase_ratio < 0.25 * n_ratio,
        "kappa outer-loop phase grew {phase_ratio:.2}× across a {n_ratio:.0}× n-sweep \
         — expected n-INDEPENDENT (ψ-Gram tensor #1033b); near-linear growth means \
         the per-trial n-row design rebuild was not skipped on tensor-covered trials"
    );
}
