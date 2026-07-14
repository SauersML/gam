//! #932 INDEPENDENT adversarial verifier for the BMS **flex** row derivative
//! tower — authored by `bms-flex-verify`, NOT by the implementer.
//!
//! This module is the separate witness for the last hand-coded BMS derivative
//! chain: the per-row gradient / Hessian assembled by
//! [`super::row_primary_hessian::…::lower_bms_flex_row_order2_from_parts`]
//! (the `coeff_au` / `coeff_bu` / `g_au_fixed` / `g_bu_fixed` + implicit-
//! function-theorem `a_u`/`a_uv` chains).
//!
//! The discipline here is deliberately blunt and external: drive the production
//! compiled lowering of the canonical BMS FLEX program to get its analytic
//! value/gradient/Hessian, then central-difference that lowering's OWN returned
//! value `ℓ(θ) = −w·logΦ(s_y·η(z; a(θ), θ))` w.r.t.
//! every primary `θ = [q, logslope, β…]`, re-solving the calibrated intercept
//! `a(θ)` at every stencil point (the value is intercept-Jacobian-independent, so
//! the re-solve shares no derivative-chain code with the analytic path). A
//! 4th-order Richardson stencil pins grad to ≤1e-7 and Hessian to ≤1e-5 with no
//! weakening. Runs on BOTH the score-warp and link-deviation blocks with a death
//! (`y = 1`).
//!
//! Three further gates:
//!   * an INDEPENDENT bracketed intercept solve cross-checks the production root,
//!   * the moving-edge **Leibniz** sliver (where dropped boundary-flux terms
//!     hide) is cross-checked by coupling a cell edge to a knot crossing
//!     `zE = (τ − a)/b` and central-differencing a moving-domain quadrature, and
//!   * a planted-corruption tripwire drops a fold term and asserts the witness
//!     FAILS — proving the oracle has teeth.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod flex_verify_932_tests;` in
//! `bms/mod.rs` (the `*_tests` allowed name), reaching the production path as a
//! private child of the common ancestor `bms`.

use super::family::*;
use super::hessian_paths::*;
use super::*;
use crate::probability::normal_cdf;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

// ------------------------------------------------------------------
// Independent fixture (replicated, NOT shared with the implementer's
// private `cell_moment_assembly` test fixture).
// ------------------------------------------------------------------

struct VFixture {
    family: BernoulliMarginalSlopeFamily,
    primary: PrimarySlices,
    runtime: DeviationRuntime,
    is_score_warp: bool,
    grid: EmpiricalZGrid,
    beta_dev: Array1<f64>,
}

fn vgrid() -> EmpiricalZGrid {
    let nodes = vec![-1.4_f64, -0.6, 0.1, 0.8, 1.5];
    let raw = [0.14_f64, 0.24, 0.28, 0.20, 0.14];
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    EmpiricalZGrid::new(nodes, weights, "flex_verify_932 grid").expect("valid grid")
}

fn vruntime() -> DeviationRuntime {
    // 11 uniform knots over [-2.45, 2.55]; order-3 smoothness penalty (the
    // only valid order for a cubic I-spline value basis — its span carries
    // quadratics in the droppable null space). Half-span offset keeps the
    // FD stencils off the spline knots so production differentiates a single
    // local cubic branch.
    let n_knots = 11usize;
    let knots = Array1::from_iter(
        (0..n_knots).map(|i| -2.45_f64 + 5.0_f64 * (i as f64) / ((n_knots - 1) as f64)),
    );
    DeviationRuntime::try_new(knots, 0.0, 3).expect("deviation runtime")
}

fn vfixture(is_score_warp: bool) -> VFixture {
    let grid = vgrid();
    let runtime = vruntime();
    let basis_dim = runtime.basis_dim();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let dummy = || {
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            ndarray::Array2::zeros((1, 1)),
        ))
    };
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::from_vec(vec![1.0])),
        weights: Arc::new(Array1::from_vec(vec![1.0])),
        z: Arc::new(Array1::from_vec(vec![0.45])),
        latent_measure: LatentMeasureKind::GlobalEmpirical { grid: grid.clone() },
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: dummy(),
        logslope_design: dummy(),
        score_warp: if is_score_warp {
            Some(runtime.clone())
        } else {
            None
        },
        link_dev: if is_score_warp {
            None
        } else {
            Some(runtime.clone())
        },
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let primary = PrimarySlices {
        q: 0,
        logslope: 1,
        h: if is_score_warp {
            Some(2..2 + basis_dim)
        } else {
            None
        },
        w: if is_score_warp {
            None
        } else {
            Some(2..2 + basis_dim)
        },
        total: 2 + basis_dim,
    };
    let beta_dev = Array1::from_shape_fn(basis_dim, |i| {
        let center = 0.5 * (basis_dim.saturating_sub(1) as f64);
        let radius = center.max(1.0);
        0.06 * ((i as f64) - center) / radius
    });
    VFixture {
        family,
        primary,
        runtime,
        is_score_warp,
        grid,
        beta_dev,
    }
}

// ------------------------------------------------------------------
// Independent scalar model (re-derived; shares no jet / IFT code).
// ------------------------------------------------------------------

/// Observed index η(a; z) = scale·(a + b·z + warp). score-warp:
/// warp = b·Σβⱼ·Φⱼ(z); link-dev: warp = Σβⱼ·Φⱼ(u), u = a + b·z. Basis values
/// come from the SEPARATE `DeviationRuntime::design` API.
fn veta(fx: &VFixture, a: f64, b: f64, beta: &Array1<f64>, z: f64, scale: f64) -> f64 {
    let u = a + b * z;
    let mut inside = u;
    if fx.is_score_warp {
        let row = fx
            .runtime
            .design(&Array1::from_vec(vec![z]))
            .expect("score-warp basis");
        let warp: f64 = row.row(0).iter().zip(beta.iter()).map(|(v, c)| v * c).sum();
        inside += b * warp;
    } else {
        let row = fx
            .runtime
            .design(&Array1::from_vec(vec![u]))
            .expect("link-dev basis");
        let dev: f64 = row.row(0).iter().zip(beta.iter()).map(|(v, c)| v * c).sum();
        inside += dev;
    }
    scale * inside
}

/// INDEPENDENT bracketed intercept solver for the flex calibration root
/// Σ_k π_k Φ(η(a; x_k)) = μ. Pure bisection driven to f64 resolution; no
/// shared IFT / jet-Newton code.
fn vintercept(fx: &VFixture, mu: f64, b: f64, beta: &Array1<f64>, scale: f64) -> f64 {
    let calib = |a: f64| -> f64 {
        let mut acc = -mu;
        for (node, weight) in fx.grid.pairs() {
            acc += weight * normal_cdf(veta(fx, a, b, beta, node, scale));
        }
        acc
    };
    let mut lo = -1.0_f64;
    let mut hi = 1.0_f64;
    let mut flo = calib(lo);
    let mut fhi = calib(hi);
    for _ in 0..100 {
        if flo <= 0.0 && fhi >= 0.0 {
            break;
        }
        if flo > 0.0 {
            hi = lo;
            fhi = flo;
            lo *= 2.0;
            flo = calib(lo);
        } else {
            lo = hi;
            flo = fhi;
            hi *= 2.0;
            fhi = calib(hi);
        }
    }
    assert!(
        flo <= 0.0 && fhi >= 0.0,
        "failed to bracket flex calibration root F({lo})={flo} F({hi})={fhi}"
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let fmid = calib(mid);
        if fmid == 0.0 || (hi - lo).abs() <= 1e-16 * mid.abs().max(1.0) {
            return mid;
        }
        if fmid < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Independent scalar row NLL ℓ(θ) = −w·logΦ(s_y·η(z_obs; a(θ), θ)), with
/// a(θ) re-solved by the independent bracketed solver.
fn vnll(fx: &VFixture, p: &[f64]) -> f64 {
    let q = p[fx.primary.q];
    let b = p[fx.primary.logslope];
    let dev = if fx.is_score_warp {
        fx.primary.h.clone().unwrap()
    } else {
        fx.primary.w.clone().unwrap()
    };
    let beta = Array1::from_iter(dev.map(|i| p[i]));
    let scale = fx.family.probit_frailty_scale();
    let marginal = bernoulli_marginal_link_map(&InverseLink::Standard(StandardLink::Probit), q)
        .expect("link map");
    let a = vintercept(fx, marginal.mu, b, &beta, scale);
    let z = fx.family.z[0];
    let eta = veta(fx, a, b, &beta, z, scale);
    let s_y = 2.0 * fx.family.y[0] - 1.0;
    let logcdf = normal_cdf(s_y * eta).max(1e-300).ln();
    -fx.family.weights[0] * logcdf
}

// ------------------------------------------------------------------
// Production canonical-lowering drivers.
// ------------------------------------------------------------------

fn beta_vec(fx: &VFixture, p: &[f64]) -> Array1<f64> {
    let dev = if fx.is_score_warp {
        fx.primary.h.clone().unwrap()
    } else {
        fx.primary.w.clone().unwrap()
    };
    Array1::from_iter(dev.map(|i| p[i]))
}

/// Hand-path returned value ℓ(θ). The calibrated intercept `a(θ)` is
/// re-solved at EVERY stencil point by the INDEPENDENT bisection
/// ([`vintercept`]) driven to f64-ulp resolution — far tighter than the
/// production solver's ~1e-12 acceptance band, so the second-difference
/// Hessian roundoff stays ~1e-9 (a 1e-12 root residual would otherwise leak
/// ~1e-5 into the Richardson Hessian). The returned NLL reads only `a` (not
/// the calibration Jacobian `m_a`), so feeding a placeholder `m_a` here is
/// exact for the value channel — the derivative chain under test is reached
/// only through the separate analytic call in `production_grad_hess`.
fn production_value(fx: &VFixture, p: &[f64]) -> f64 {
    let q = p[fx.primary.q];
    let b = p[fx.primary.logslope];
    let beta = beta_vec(fx, p);
    let (beta_h, beta_w) = if fx.is_score_warp {
        (Some(&beta), None)
    } else {
        (None, Some(&beta))
    };
    let scale = fx.family.probit_frailty_scale();
    let marginal = bernoulli_marginal_link_map(&InverseLink::Standard(StandardLink::Probit), q)
        .expect("link map");
    let intercept = vintercept(fx, marginal.mu, b, &beta, scale);
    let row_ctx = BernoulliMarginalSlopeRowExactContext {
        intercept,
        m_a: 1.0,
        intercept_fast_path: false,
        degree9_cells: None,
    };
    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(fx.primary.total);
    fx.family
        .lower_bms_flex_row_order2_from_parts(
            0,
            &fx.primary,
            q,
            b,
            beta_h,
            beta_w,
            &row_ctx,
            None,
            None,
            false,
            &mut scratch,
        )
        .expect("production canonical-lowering value")
}

/// Production compiled-lowering analytic (value, gradient, Hessian) at θ.
fn production_grad_hess(fx: &VFixture, p: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let r = fx.primary.total;
    let q = p[fx.primary.q];
    let b = p[fx.primary.logslope];
    let beta = beta_vec(fx, p);
    let (beta_h, beta_w) = if fx.is_score_warp {
        (Some(&beta), None)
    } else {
        (None, Some(&beta))
    };
    let (intercept, m_a, _) = fx
        .family
        .solve_row_intercept_base(0, q, b, beta_h, beta_w, None)
        .expect("intercept solve");
    let row_ctx = BernoulliMarginalSlopeRowExactContext {
        intercept,
        m_a,
        intercept_fast_path: false,
        degree9_cells: None,
    };
    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);
    let v = fx
        .family
        .lower_bms_flex_row_order2_from_parts(
            0,
            &fx.primary,
            q,
            b,
            beta_h,
            beta_w,
            &row_ctx,
            None,
            None,
            true,
            &mut scratch,
        )
        .expect("production canonical-lowering grad/hess");
    let grad = scratch.grad.iter().copied().collect::<Vec<_>>();
    let mut hess = vec![0.0; r * r];
    for u in 0..r {
        for w in 0..r {
            hess[u * r + w] = scratch.hess[[u, w]];
        }
    }
    (v, grad, hess)
}

// ------------------------------------------------------------------
// 4th-order Richardson central-difference of the hand-path VALUE.
//
// These deliberately do NOT route through
// `gam_test_support::fd_checker`: that helper is the plain two-point
// central difference (O(h²)), whereas these Richardson-extrapolate two
// central differences at `h` and `h/2` into an O(h⁴) estimate, and the
// diagonal second derivative here uses the `h`-spaced three-point stencil
// `(f(x+h) − 2·f(x) + f(x−h))/h²` rather than the `2h`-spaced four-point
// stencil the canonical Hessian helper (and `bms::test_support`) use. The
// tight grad/Hessian gates below (`1e-7` / `1e-5`) are calibrated to this
// higher-order scheme, so folding it into the plain canonical helper would
// change the FD numbers. It is an honest higher-order carve-out, not a
// duplicated copy of the canonical two-point math.
// ------------------------------------------------------------------

/// 4th-order Richardson first derivative along axis `i`.
fn fd_grad(fx: &VFixture, p0: &[f64], i: usize, h: f64) -> f64 {
    let central = |step: f64| {
        let mut pp = p0.to_vec();
        let mut pm = p0.to_vec();
        pp[i] += step;
        pm[i] -= step;
        (production_value(fx, &pp) - production_value(fx, &pm)) / (2.0 * step)
    };
    let g_h = central(h);
    let g_h2 = central(0.5 * h);
    // Richardson: (4·D(h/2) − D(h))/3  ⇒ O(h⁴).
    (4.0 * g_h2 - g_h) / 3.0
}

/// 4th-order Richardson second derivative (diagonal or mixed) along (i, j).
fn fd_hess(fx: &VFixture, p0: &[f64], i: usize, j: usize, h: f64) -> f64 {
    let cross = |step: f64| {
        if i == j {
            let mut pp = p0.to_vec();
            let mut pm = p0.to_vec();
            pp[i] += step;
            pm[i] -= step;
            let f0 = production_value(fx, p0);
            (production_value(fx, &pp) - 2.0 * f0 + production_value(fx, &pm)) / (step * step)
        } else {
            let mut tpp = p0.to_vec();
            let mut tpm = p0.to_vec();
            let mut tmp = p0.to_vec();
            let mut tmm = p0.to_vec();
            tpp[i] += step;
            tpp[j] += step;
            tpm[i] += step;
            tpm[j] -= step;
            tmp[i] -= step;
            tmp[j] += step;
            tmm[i] -= step;
            tmm[j] -= step;
            (production_value(fx, &tpp) - production_value(fx, &tpm) - production_value(fx, &tmp)
                + production_value(fx, &tmm))
                / (4.0 * step * step)
        }
    };
    let d_h = cross(h);
    let d_h2 = cross(0.5 * h);
    (4.0 * d_h2 - d_h) / 3.0
}

// ==================================================================
// GATE 1: production compiled-lowering grad/Hessian vs independent FD of its
// value, both deviation branches, death (y = 1).
// ==================================================================
fn run_production_gate(is_score_warp: bool) {
    let fx = vfixture(is_score_warp);
    let r = fx.primary.total;
    let label = if is_score_warp {
        "score-warp"
    } else {
        "link-dev"
    };

    let q0 = 0.2_f64;
    let b0 = 0.35_f64;
    let mut p0 = vec![0.0; r];
    p0[fx.primary.q] = q0;
    p0[fx.primary.logslope] = b0;
    let dev = if is_score_warp {
        fx.primary.h.clone().unwrap()
    } else {
        fx.primary.w.clone().unwrap()
    };
    for (k, i) in dev.clone().enumerate() {
        p0[i] = fx.beta_dev[k];
    }

    // (a) independent value cross-check: production lowering == scalar NLL.
    let v_production = production_value(&fx, &p0);
    let v_ind = vnll(&fx, &p0);
    assert!(
        (v_production - v_ind).abs() <= 1e-9 * v_ind.abs().max(1.0),
        "{label} production value {v_production:+.12e} != independent scalar {v_ind:+.12e}"
    );

    // (b) independent intercept cross-check vs production root.
    let marginal = bernoulli_marginal_link_map(&InverseLink::Standard(StandardLink::Probit), q0)
        .expect("link map");
    let scale = fx.family.probit_frailty_scale();
    let beta = Array1::from_iter(dev.clone().map(|i| p0[i]));
    let (beta_h, beta_w) = if is_score_warp {
        (Some(&beta), None)
    } else {
        (None, Some(&beta))
    };
    let a_ind = vintercept(&fx, marginal.mu, b0, &beta, scale);
    let (a_prod, _, _) = fx
        .family
        .solve_row_intercept_base(0, q0, b0, beta_h, beta_w, None)
        .expect("prod intercept");
    assert!(
        (a_ind - a_prod).abs() <= 1e-9 * a_prod.abs().max(1.0),
        "{label} intercept independent {a_ind:+.12e} != production {a_prod:+.12e}"
    );

    // (c) analytic grad/Hessian vs 4th-order Richardson FD of the value.
    let (v_gh, grad, hess) = production_grad_hess(&fx, &p0);
    // The analytic path (production root) and the FD path (ulp-tight
    // independent root) must agree on the value at the base point.
    assert!(
        (v_gh - v_production).abs() <= 1e-9 * v_production.abs().max(1.0),
        "{label} analytic-call value {v_gh:+.12e} != value-call {v_production:+.12e}"
    );
    let h = 1.0e-3_f64;
    let mut max_g = 0.0_f64;
    let mut max_hd = 0.0_f64;
    for i in 0..r {
        let fdg = fd_grad(&fx, &p0, i, h);
        let e = (grad[i] - fdg).abs();
        max_g = max_g.max(e);
        assert!(
            e <= 1e-7 * fdg.abs().max(1.0) + 1e-9,
            "{label} grad[{i}] analytic {:+.12e} != fd {fdg:+.12e} (err {e:.2e})",
            grad[i]
        );
        for j in i..r {
            let fdh = fd_hess(&fx, &p0, i, j, h);
            let e = (hess[i * r + j] - fdh).abs();
            max_hd = max_hd.max(e);
            assert!(
                e <= 1e-5 * fdh.abs().max(1.0) + 1e-7,
                "{label} hess[{i},{j}] analytic {:+.12e} != fd {fdh:+.12e} (err {e:.2e})",
                hess[i * r + j]
            );
            // symmetry
            assert!((hess[i * r + j] - hess[j * r + i]).abs() <= 1e-12);
        }
    }
    eprintln!("#932 verify {label}: r={r}  max|grad−fd|={max_g:.2e}  max|hess−fd|={max_hd:.2e}");
}

#[test]
fn production_flex_grad_hess_matches_independent_fd_score_warp_932() {
    run_production_gate(true);
}

#[test]
fn production_flex_grad_hess_matches_independent_fd_link_dev_932() {
    run_production_gate(false);
}

// ==================================================================
// GATE 2: one REAL StandardNormal FLEX row, with score-warp and link-
// deviation blocks active, must carry one coherent production
// derivative ladder V -> G -> H -> t3 -> t4.
//
// This deliberately contains no second likelihood, spline, calibration,
// implicit-root, or tensor formula. Each finite difference consumes the
// immediately lower channel from the production canonical row lowerings:
//
//   d(V)[d]       = G . d
//   d(G)[d]       = H d
//   d(H)[d]       = t3[d]
//   d(t3[d])[d]   = t4[d,d]
//
// Every perturbed state rebuilds the exact cache, re-solves the row root,
// and regenerates its StandardNormal cell moments. A duplicated synthetic
// algebra could agree with itself; this derivative ladder cannot.
// ==================================================================

fn standard_normal_flex_fixture() -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let score_seed = Array1::linspace(-2.0, 2.0, 8);
    let link_seed = Array1::linspace(-1.8, 1.8, 8);
    let config = DeviationBlockConfig {
        num_internal_knots: 3,
        ..DeviationBlockConfig::default()
    };
    let score = build_score_warp_deviation_block_from_seed(&score_seed, &config)
        .expect("build StandardNormal score-warp block");
    let link = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &link_seed, &config,
    )
    .expect("build StandardNormal link-deviation block");

    // A one-row fitted state keeps this fourth-order lock focused while still
    // traversing the genuine StandardNormal cell partition and moment ladder.
    let marginal_x = Array2::ones((1, 1));
    let logslope_x = Array2::ones((1, 1));
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::from_vec(vec![1.0])),
        weights: Arc::new(Array1::from_vec(vec![0.9])),
        z: Arc::new(Array1::from_vec(vec![0.35])),
        latent_measure: LatentMeasureKind::StandardNormal,
        gaussian_frailty_sd: Some(0.15),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
        logslope_design: DesignMatrix::Dense(DenseDesignMatrix::from(logslope_x.clone())),
        score_warp: Some(score.runtime.clone()),
        link_dev: Some(link.runtime.clone()),
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };

    let marginal_beta = Array1::from_vec(vec![0.18]);
    let logslope_beta = Array1::from_vec(vec![0.32]);
    let score_beta = Array1::from_shape_fn(score.runtime.basis_dim(), |index| {
        0.0015 * (index as f64 + 1.0)
    });
    let link_beta = Array1::from_shape_fn(link.runtime.basis_dim(), |index| {
        -0.001 * (index as f64 + 1.0)
    });
    let states = vec![
        ParameterBlockState {
            eta: marginal_x.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: logslope_x.dot(&logslope_beta),
            beta: logslope_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(1),
            beta: score_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(1),
            beta: link_beta,
        },
    ];
    (family, states)
}

struct StandardNormalFlexChannels {
    value: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
    third: Array2<f64>,
    fourth: Option<Array2<f64>>,
}

fn standard_normal_flex_channels(
    family: &BernoulliMarginalSlopeFamily,
    states: &[ParameterBlockState],
    cache: &super::exact_eval_cache::BernoulliMarginalSlopeExactEvalCache,
    row: usize,
    direction: &Array1<f64>,
    need_fourth: bool,
) -> StandardNormalFlexChannels {
    assert!(
        matches!(family.latent_measure, LatentMeasureKind::StandardNormal),
        "canonical derivative ladder must stay on the StandardNormal branch"
    );
    let primary = &cache.primary;
    assert_eq!(direction.len(), primary.total);
    let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(cache, row);
    let row_moments = cache
        .row_cell_moments
        .as_ref()
        .and_then(|bundle| bundle.row(row, 9))
        .expect("real StandardNormal FLEX row must materialize degree-9 production moments");
    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
    let value = family
        .lower_bms_flex_row_order2_with_moments(
            row,
            states,
            primary,
            row_ctx,
            Some(row_moments),
            cache.cell_family_forest.as_ref(),
            true,
            &mut scratch,
        )
        .expect("canonical StandardNormal V/G/H lowering");
    let third = family
        .row_primary_third_contracted_with_moments(row, states, cache, row_ctx, direction)
        .expect("canonical StandardNormal t3 lowering");
    let fourth = need_fourth.then(|| {
        family
            .row_primary_fourth_contracted_ordered(
                row, states, cache, row_ctx, direction, direction,
            )
            .expect("canonical StandardNormal t4 lowering")
    });
    StandardNormalFlexChannels {
        value,
        gradient: scratch.grad,
        hessian: scratch.hess,
        third,
        fourth,
    }
}

fn perturb_standard_normal_flex_states(
    states: &[ParameterBlockState],
    primary: &PrimarySlices,
    row: usize,
    direction: &Array1<f64>,
    step: f64,
) -> Vec<ParameterBlockState> {
    let mut perturbed = states.to_vec();
    let marginal_delta = step * direction[primary.q];
    perturbed[0].eta[row] += marginal_delta;
    perturbed[0].beta[0] += marginal_delta;
    let logslope_delta = step * direction[primary.logslope];
    perturbed[1].eta[row] += logslope_delta;
    perturbed[1].beta[0] += logslope_delta;
    if let Some(range) = primary.h.as_ref() {
        for (local, index) in range.clone().enumerate() {
            perturbed[2].beta[local] += step * direction[index];
        }
    }
    if let Some(range) = primary.w.as_ref() {
        for (local, index) in range.clone().enumerate() {
            perturbed[3].beta[local] += step * direction[index];
        }
    }
    perturbed
}

fn derivative_ladder_relative_error(analytic: f64, finite_difference: f64) -> f64 {
    (analytic - finite_difference).abs() / (1.0 + analytic.abs().max(finite_difference.abs()))
}

#[test]
fn standard_normal_flex_canonical_derivative_ladder_matches_vgh_t3_t4_932() {
    let row = 0usize;
    let (family, states) = standard_normal_flex_fixture();
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("base StandardNormal FLEX exact cache");
    let primary = cache.primary.clone();
    let h_range = primary.h.as_ref().expect("active score-warp range");
    let w_range = primary.w.as_ref().expect("active link-deviation range");
    assert!(!h_range.is_empty() && !w_range.is_empty());

    // One mixed direction forces q, logslope, score-warp, and link-deviation
    // cross terms through every derivative order without materializing a dense
    // t3/t4 tensor.
    let mut direction = Array1::<f64>::zeros(primary.total);
    direction[primary.q] = 0.55;
    direction[primary.logslope] = -0.35;
    direction[h_range.start] = 0.45;
    direction[w_range.start] = -0.40;

    let base = standard_normal_flex_channels(&family, &states, &cache, row, &direction, true);
    let step = 2.0e-4_f64;
    let plus_states = perturb_standard_normal_flex_states(&states, &primary, row, &direction, step);
    let minus_states =
        perturb_standard_normal_flex_states(&states, &primary, row, &direction, -step);
    let plus_cache = family
        .build_exact_eval_cache(&plus_states)
        .expect("positive-direction StandardNormal FLEX exact cache");
    let minus_cache = family
        .build_exact_eval_cache(&minus_states)
        .expect("negative-direction StandardNormal FLEX exact cache");
    let plus =
        standard_normal_flex_channels(&family, &plus_states, &plus_cache, row, &direction, false);
    let minus =
        standard_normal_flex_channels(&family, &minus_states, &minus_cache, row, &direction, false);

    let max_vg = derivative_ladder_relative_error(
        base.gradient.dot(&direction),
        (plus.value - minus.value) / (2.0 * step),
    );
    let mut max_gh = 0.0_f64;
    let mut max_h3 = 0.0_f64;
    let mut max_34 = 0.0_f64;
    let hessian_direction = base.hessian.dot(&direction);
    let fourth = base.fourth.as_ref().expect("requested t4 channel");
    for u in 0..primary.total {
        let gradient_fd = (plus.gradient[u] - minus.gradient[u]) / (2.0 * step);
        max_gh = max_gh.max(derivative_ladder_relative_error(
            hessian_direction[u],
            gradient_fd,
        ));
        for v in 0..primary.total {
            let third_fd = (plus.hessian[[u, v]] - minus.hessian[[u, v]]) / (2.0 * step);
            max_h3 = max_h3.max(derivative_ladder_relative_error(
                base.third[[u, v]],
                third_fd,
            ));
            let fourth_fd = (plus.third[[u, v]] - minus.third[[u, v]]) / (2.0 * step);
            max_34 = max_34.max(derivative_ladder_relative_error(fourth[[u, v]], fourth_fd));
            assert_eq!(
                base.hessian[[u, v]].to_bits(),
                base.hessian[[v, u]].to_bits(),
                "canonical StandardNormal H lost exact symmetry at [{u},{v}]"
            );
            assert_eq!(
                base.third[[u, v]].to_bits(),
                base.third[[v, u]].to_bits(),
                "canonical StandardNormal t3[d] lost exact symmetry at [{u},{v}]"
            );
            assert_eq!(
                fourth[[u, v]].to_bits(),
                fourth[[v, u]].to_bits(),
                "canonical StandardNormal t4[d,d] lost exact symmetry at [{u},{v}]"
            );
        }
    }

    assert!(base.value.is_finite());
    assert!(base.gradient.iter().all(|value| value.is_finite()));
    assert!(base.hessian.iter().all(|value| value.is_finite()));
    assert!(base.third.iter().all(|value| value.is_finite()));
    assert!(fourth.iter().all(|value| value.is_finite()));
    assert!(
        base.third.iter().any(|value| value.abs() > 1e-10)
            && fourth.iter().any(|value| value.abs() > 1e-10),
        "StandardNormal t3/t4 parity lock must carry nonzero signal"
    );

    assert!(
        max_vg <= 2e-7,
        "V->G directional relative error {max_vg:.3e}"
    );
    assert!(
        max_gh <= 2e-6,
        "G->H directional relative error {max_gh:.3e}"
    );
    assert!(
        max_h3 <= 2e-5,
        "H->t3 directional relative error {max_h3:.3e}"
    );
    assert!(
        max_34 <= 2e-4,
        "t3->t4 directional relative error {max_34:.3e}"
    );
    eprintln!(
        "#932 StandardNormal FLEX canonical ladder: V->G={max_vg:.3e} G->H={max_gh:.3e} H->t3={max_h3:.3e} t3->t4={max_34:.3e}"
    );
}

// ==================================================================
// GATE 3: moving-edge Leibniz cross-check. Couple a cell edge to a
// knot crossing zE = (τ − a)/b and central-difference a moving-domain
// quadrature; the test_support sliver jet must track the boundary flux
// as b sweeps the crossing across the cell.
// ==================================================================

fn veta_cell(c: [f64; 4], z: f64) -> f64 {
    c[0] + c[1] * z + c[2] * z * z + c[3] * z * z * z
}
fn vq_cell(c: [f64; 4], z: f64) -> f64 {
    let e = veta_cell(c, z);
    0.5 * (z * z + e * e)
}
/// High-accuracy composite-Simpson moment ∫_{zl}^{zr} zⁿ·e^{−q(z;c)} dz.
fn vmoment(c: [f64; 4], zl: f64, zr: f64, n: usize, panels: usize) -> f64 {
    let m = panels * 2;
    let hstep = (zr - zl) / (m as f64);
    let f = |z: f64| z.powi(n as i32) * (-vq_cell(c, z)).exp();
    let mut acc = f(zl) + f(zr);
    for k in 1..m {
        let z = zl + (k as f64) * hstep;
        acc += if k % 2 == 1 { 4.0 } else { 2.0 } * f(z);
    }
    acc * hstep / 3.0
}

#[test]
fn moving_edge_leibniz_tracks_boundary_flux_932() {
    use super::test_support::{Jet2, RuntimeJet, cell_base_moment_jets_moving};

    // Primaries θ = (a, b). The right edge is a knot crossing zE=(τ−a)/b;
    // the left edge is a fixed (−∞-side) base point. As b moves, zE sweeps.
    let a0 = 0.30_f64;
    let b0 = 0.80_f64;
    let tau = 1.10_f64; // knot in u-space; crossing z = (τ − a)/b.
    let zl0 = -0.90_f64;
    let zr0 = (tau - a0) / b0; // ≈ 1.0 — inside the support, moves with b.
    let base_c = [0.10_f64, 0.45, -0.18, 0.08];
    let max_n = 4usize;
    let scalar_deg = max_n + 12;
    let panels = 6000usize;

    let p = 2usize; // (a, b)
    let a_jet = Jet2::primary(a0, 0, p);
    let b_jet = Jet2::primary(b0, 1, p);
    // zE(a,b) = (τ − a)/b  as an order-2 jet via the substrate algebra.
    let tau_c = Jet2::constant(tau, p);
    let inv_b = {
        // 1/b through compose_unary: f=1/b, f'=−1/b², f''=2/b³.
        let v = b0;
        b_jet.compose_unary([1.0 / v, -1.0 / (v * v), 2.0 / (v * v * v), 0.0, 0.0])
    };
    let zr_jet = tau_c.sub(&a_jet).mul(&inv_b);
    let zl_jet = Jet2::constant(zl0, p);
    // c is fixed here (edge motion only) — constant coefficient jets.
    let c_jets: [Jet2; 4] = std::array::from_fn(|k| Jet2::constant(base_c[k], p));

    // Base scalar moments over the FIXED base domain.
    let scalar_moments: Vec<f64> = (0..=scalar_deg)
        .map(|n| vmoment(base_c, zl0, zr0, n, panels))
        .collect();

    let m_jets = cell_base_moment_jets_moving(
        &c_jets,
        base_c,
        &scalar_moments,
        max_n,
        &zl_jet,
        zl0,
        &zr_jet,
        zr0,
    );

    // Independent FD of the moving-domain moment as a function of (a,b),
    // with zE(a,b) = (τ−a)/b and zl fixed.
    let m_of = |a: f64, b: f64, n: usize| -> f64 {
        let zr = (tau - a) / b;
        vmoment(base_c, zl0, zr, n, panels)
    };
    let h = 1e-4_f64;
    // confirm the crossing actually sweeps a meaningful distance under h.
    let sweep = ((tau - a0) / (b0 + h) - (tau - a0) / (b0 - h)).abs();
    assert!(sweep > 1e-4, "knot crossing did not sweep ({sweep:.2e})");

    let mut max_g = 0.0_f64;
    let mut max_h = 0.0_f64;
    for n in 0..=max_n {
        assert!(
            (m_jets[n].v - scalar_moments[n]).abs() <= 1e-9 * scalar_moments[n].abs().max(1.0),
            "moving M[{n}] value mismatch"
        );
        // grad over (a,b)
        let ga = (m_of(a0 + h, b0, n) - m_of(a0 - h, b0, n)) / (2.0 * h);
        let gb = (m_of(a0, b0 + h, n) - m_of(a0, b0 - h, n)) / (2.0 * h);
        max_g = max_g
            .max((m_jets[n].g[0] - ga).abs())
            .max((m_jets[n].g[1] - gb).abs());
        assert!(
            (m_jets[n].g[0] - ga).abs() <= 1e-6 * ga.abs().max(1.0) + 1e-8,
            "moving dM[{n}]/da jet {:+.12e} != fd {ga:+.12e}",
            m_jets[n].g[0]
        );
        assert!(
            (m_jets[n].g[1] - gb).abs() <= 1e-6 * gb.abs().max(1.0) + 1e-8,
            "moving dM[{n}]/db jet {:+.12e} != fd {gb:+.12e}",
            m_jets[n].g[1]
        );
        // Hessian bb (where the sliver flux + its z-derivative live).
        let hbb = (m_of(a0, b0 + h, n) - 2.0 * m_of(a0, b0, n) + m_of(a0, b0 - h, n)) / (h * h);
        let hab = (m_of(a0 + h, b0 + h, n) - m_of(a0 + h, b0 - h, n) - m_of(a0 - h, b0 + h, n)
            + m_of(a0 - h, b0 - h, n))
            / (4.0 * h * h);
        max_h = max_h
            .max((m_jets[n].h[p + 1] - hbb).abs())
            .max((m_jets[n].h[1] - hab).abs());
        assert!(
            (m_jets[n].h[p + 1] - hbb).abs() <= 1e-3 * hbb.abs().max(1.0) + 1e-5,
            "moving d2M[{n}]/db2 jet {:+.12e} != fd {hbb:+.12e}",
            m_jets[n].h[p + 1]
        );
        assert!(
            (m_jets[n].h[1] - hab).abs() <= 1e-3 * hab.abs().max(1.0) + 1e-5,
            "moving d2M[{n}]/dadb jet {:+.12e} != fd {hab:+.12e}",
            m_jets[n].h[1]
        );
    }
    eprintln!(
        "#932 verify leibniz: sweep={sweep:.3e}  max|grad−fd|={max_g:.2e}  max|hess−fd|={max_h:.2e}"
    );
}

// ==================================================================
// GATE 4: planted-corruption tripwire. Re-derive the Leibniz sliver
// with one fold term DROPPED (the ½·g_z·δ² boundary-curvature term) and
// confirm the moving-domain FD REJECTS it — proving the oracle's teeth.
// ==================================================================

/// A deliberately CORRUPT moving base-moment: interior + a sliver that
/// drops the ½·g_z·δ² second-order boundary term. If the witness had no
/// teeth this would still pass; it must NOT.
/// The left edge is held constant in this fixture, so only the right
/// (moving) edge contributes a sliver — no `zl0` parameter is needed.
fn corrupt_moving_moment(
    c0: [f64; 4],
    scalar_moments: &[f64],
    n: usize,
    zr_jet: &super::test_support::Jet2,
    zr0: f64,
) -> super::test_support::Jet2 {
    use super::test_support::{Jet2, RuntimeJet};
    let p = zr_jet.p();
    let cst = |x: f64| Jet2::constant(x, p);
    // interior fixed-domain moment (c fixed ⇒ just the base scalar moment).
    let interior = cst(scalar_moments[n]);
    // sliver with the δ² curvature term DROPPED: only g·δ (g constant here).
    let eta0 = veta_cell(c0, zr0);
    let q0 = 0.5 * (zr0 * zr0 + eta0 * eta0);
    let g0 = zr0.powi(n as i32) * (-q0).exp();
    let delta = zr_jet.sub(&cst(zr0));
    let sliver_r = delta.scale(g0); // CORRUPT: missing ½·g_z·δ²
    interior.add(&sliver_r)
}

#[test]
fn planted_corruption_tripwire_fails_932() {
    use super::test_support::{Jet2, RuntimeJet};
    let a0 = 0.30_f64;
    let b0 = 0.80_f64;
    let tau = 1.10_f64;
    let zl0 = -0.90_f64;
    let zr0 = (tau - a0) / b0;
    let base_c = [0.10_f64, 0.45, -0.18, 0.08];
    let n = 2usize;
    let panels = 4000usize;
    let scalar_deg = n + 12;

    let p = 2usize;
    let a_jet = Jet2::primary(a0, 0, p);
    let b_jet = Jet2::primary(b0, 1, p);
    let inv_b = b_jet.compose_unary([1.0 / b0, -1.0 / (b0 * b0), 2.0 / (b0 * b0 * b0), 0.0, 0.0]);
    let zr_jet = Jet2::constant(tau, p).sub(&a_jet).mul(&inv_b);

    let scalar_moments: Vec<f64> = (0..=scalar_deg)
        .map(|k| vmoment(base_c, zl0, zr0, k, panels))
        .collect();
    let corrupt = corrupt_moving_moment(base_c, &scalar_moments, n, &zr_jet, zr0);

    // FD of the true moving moment.
    let m_of = |a: f64, b: f64| -> f64 {
        let zr = (tau - a) / b;
        vmoment(base_c, zl0, zr, n, panels)
    };
    let h = 5e-4_f64;
    let hbb = (m_of(a0, b0 + h) - 2.0 * m_of(a0, b0) + m_of(a0, b0 - h)) / (h * h);

    // The dropped ½·g_z·δ² term is exactly the second-order boundary
    // curvature, so the corrupt Hessian-bb must DIVERGE from FD. Assert the
    // witness catches it (i.e. the corrupt value FAILS the GATE-2 bound).
    let err = (corrupt.h[p + 1] - hbb).abs();
    let bound = 1e-3 * hbb.abs().max(1.0) + 1e-5;
    assert!(
        err > bound,
        "TRIPWIRE TOOTHLESS: corrupt sliver Hessian-bb err {err:.3e} <= bound {bound:.3e} \
             (the dropped ½·g_z·δ² term went undetected — the moving-edge oracle has no teeth)"
    );
    eprintln!(
        "#932 verify tripwire: corrupt err={err:.3e} > bound={bound:.3e}  (oracle has teeth)"
    );
}

#[test]
fn selected_gpu_consumers_cannot_retry_on_cpu_932() {
    let axis_source = include_str!("axis_direction_search.rs");
    let workspace_source = include_str!("custom_family_impl.rs");
    let cache_source = include_str!("exact_eval_cache.rs");
    let dense_source = include_str!("row_primary_hessian.rs");
    let device_source = include_str!("gpu/row.rs");

    assert_eq!(
        axis_source.matches("require_selected_gpu_result(").count(),
        9,
        "seven HVP/diagonal dispatches plus the joint-gradient and dense cache-boundary adapters must share the fail-closed contract"
    );
    assert_eq!(
        workspace_source
            .matches("require_selected_gpu_result(")
            .count(),
        0,
        "workspace must delegate to cache-boundary adapters instead of owning CUDA launch policy"
    );
    assert!(!axis_source.contains("falling back to CPU"));
    assert!(!workspace_source.contains("falling back to CPU"));
    assert!(!workspace_source.contains("p_total <= crate::bms::gpu::row::DENSE_BLOCK_MAX_P"));
    assert!(axis_source.contains("launch_bms_flex_row_dense(device_state)"));
    assert!(axis_source.contains("launch_bms_flex_row_joint_gradient(device_state)"));
    assert!(workspace_source.contains("selected_device_joint_gradient_from_cache"));
    assert!(workspace_source.contains("device_joint_gradient:"));
    // The workspace struct definition (with the once-cached device joint
    // gradient field's full type) moved to row_kernel.rs; the usage seams
    // above stay in custom_family_impl.rs.
    assert!(
        include_str!("row_kernel.rs")
            .contains("OnceLock<Result<Arc<ExactNewtonJointGradientEvaluation>, String>>")
    );
    assert!(cache_source.contains("reject_device_cpu_recompute"));
    assert!(dense_source.contains("selected_device_joint_gradient_from_cache"));
    assert!(dense_source.contains("selected_device_dense_hessian_from_cache"));
    assert!(device_source.contains("bms_flex_row_joint_gradient_partial"));
    assert!(device_source.contains("bms_flex_row_joint_gradient_reduce"));
    assert!(!device_source.contains("drop(d_neglog)"));
    assert!(!device_source.contains("drop(d_grad)"));
}
