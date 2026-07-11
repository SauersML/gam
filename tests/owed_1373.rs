//! Owed-work regression gate for issue #1373 — quality-suite CORRECTNESS for
//! the multinomial-softmax and Poisson-tensor recovery capabilities.
//!
//! #1373 split off five quality-suite failures from #1082. This file pins the
//! parts that are RESOLVED and OBJECTIVELY VERIFIABLE WITHOUT R, so a regression
//! of the landed fixes fails CI even on a runner that has no R / mgcv / VGAM.
//! The full `quality_vs_*` fixtures additionally gate on a `run_r`
//! match-or-beat-the-reference arm; those arms (and their disposition) are
//! documented per-test below.
//!
//! ## What is covered here (R-free, gam-side objective recovery)
//!
//! 1. **Heterogeneous multinomial smoothness** — FIXED by `e2afbf80d`
//!    ("size hetero multinomial x1 basis to its true df (k=12)"). The DGP draws a
//!    `1.6·sin(3.3π·x1)+0.8·cos(2x1)` term (true df ≈ 8); the earlier `k = 6`
//!    basis could span only ~5 spline df, so the unpenalized ORACLE on that basis
//!    was ~0.186 — above the 0.10 truth bar — making the test fail on basis
//!    CAPACITY, not on adaptive-smoothing quality. Sizing the wiggly x1 term to
//!    `k = 12` (~11 spline df) lets the basis express the truth, and gam's
//!    per-(class, term) REML recovers the heterogeneous simplex under the 0.10
//!    bar. We pin gam's OWN RMSE-to-truth here; the strict beat-VGAM-fixed-df arm
//!    in the parent fixture needs R.
//!
//! 2. **Poisson(log) × tensor-product te() truth recovery** — gam recovers the
//!    known mean surface `mu=exp(0.8+0.3·sin(x)+0.2·z²)` under the absolute
//!    truth-recovery bar (`RMSE(gam,truth) ≤ 0.18·range(mu)`) with effective
//!    degrees of freedom in a sane non-degenerate range. te() correctly defaults
//!    to `double_penalty=false` (mgcv `select=FALSE`), so it does not carry the
//!    1-D null-space shrinkage that would over-smooth a tensor (term_builder.rs
//!    ~3035). This pins the gam-side recovery so a further regression (e.g. a
//!    tensor-penalty or PIRLS-row break that collapses or smears the surface)
//!    fails CI.
//!
//! ## What is NOT asserted here (open, needs R + a supervised REML calibration)
//!
//!   - `quality_vs_mgcv_poisson_tensor`'s SECONDARY `gam_err ≤ 1.10·mgcv_err`
//!     match-or-beat arm: gam recovers the surface (passes the absolute bar) but
//!     selects fewer effective df than mgcv (edf ≈ 6.9 vs 10.8) and loses the
//!     head-to-head by ~2×. This is a genuine REML λ-precision gap in the
//!     GLM-family path (a diagnostic probe lives at
//!     `examples/probe_poisson_tensor_oversmooth.rs`), with no isolated one-line
//!     defect and broad collateral risk to the sibling tensor tests — it needs a
//!     focused, human-supervised λ-calibration pass, not a blind autonomous
//!     patch, and an R-enabled runner to measure.
//!   - `quality_vs_vgam_multinomial_softmax::recovers_true_simplex`'s
//!     `gam_err ≤ 1.05·mgcv_err` arm: gam lands ~4% over the 0.06 absolute bar
//!     (RMSE 0.0624) at a genuine REML stationary point — the same multinomial
//!     penalty-scale calibration, same broad-risk + needs-R caveat.
//!   - `quality_vs_vgam_multinomial_smooth_by_factor::recovers_truth`: a
//!     gam-side INNER-SOLVE crash on separable multinomial geometry — the
//!     known-open #1040 identifiable-subspace defect, out of REML/basis scope.
//!   - `quality_vs_nnet_multinom_penguins_species` synthetic arm: borderline at
//!     the 0.90 accuracy bar and R-gated; the real-data penguin arm passes.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const K: usize = 3;
const N_HETERO: usize = 200;

/// Stable softmax of a K-vector of logits (reference class last, η ≡ 0).
fn softmax(eta: &[f64; K]) -> [f64; K] {
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0_f64; K];
    let mut sum = 0.0;
    for k in 0..K {
        exps[k] = (eta[k] - m).exp();
        sum += exps[k];
    }
    let mut out = [0.0_f64; K];
    for k in 0..K {
        out[k] = exps[k] / sum;
    }
    out
}

/// Heterogeneous true log-odds: a wiggly x1 term (true df ≈ 8) plus a near-linear
/// x2 term (true df ≈ 2); reference class K-1 pinned to η ≡ 0. Mirrors the
/// `true_eta_hetero` DGP of `quality_vs_vgam_multinomial_softmax.rs`.
fn true_eta_hetero(x1: f64, x2: f64, x3: f64) -> [f64; K] {
    let wiggly = 1.6 * (3.3 * PI * x1).sin() + 0.8 * (2.0 * x1).cos();
    let nearly_linear = 1.4 * x2;
    let eta0 = 0.4 + wiggly + 0.3 * nearly_linear + 1.2 * x3;
    let eta1 = -0.3 - 0.6 * wiggly + nearly_linear - 0.7 * x3;
    [eta0, eta1, 0.0]
}

/// #1373 (heterogeneous arm): with the wiggly x1 term sized to its true df
/// (`k = 12`, the #e2afbf80d fix), gam's per-(class, term) REML must recover the
/// heterogeneous true class-probability surface under the SAME absolute 0.10
/// truth bar the parent fixture uses. A regression to `k = 6` (or a fused
/// single-λ-per-class driver) would push the unpenalized oracle above 0.10 and
/// fail this. This is the R-free, gam-side half of
/// `gam_multinomial_softmax_heterogeneous_smoothness_beats_fixed_df`.
#[test]
fn hetero_multinomial_recovers_true_simplex_at_true_df_basis_1373() {
    init_parallelism();

    // Same seed and uniforms as the parent fixture so the DGP is identical.
    let mut rng = StdRng::seed_from_u64(0x5EED_C0DE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform x1");
    let u01 = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x2");
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform x3");
    let udraw = Uniform::new(0.0_f64, 1.0_f64).expect("uniform draw");

    let mut x1 = Vec::with_capacity(N_HETERO);
    let mut x2 = Vec::with_capacity(N_HETERO);
    let mut x3 = Vec::with_capacity(N_HETERO);
    let mut cls_code = Vec::with_capacity(N_HETERO);
    let mut true_prob_by_code: Vec<[f64; K]> = Vec::with_capacity(N_HETERO);
    for _ in 0..N_HETERO {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let p = softmax(&true_eta_hetero(a, b, c));
        let u = udraw.sample(&mut rng);
        let mut acc = 0.0;
        let mut chosen = K - 1;
        for (k, &pk) in p.iter().enumerate() {
            acc += pk;
            if u <= acc {
                chosen = k;
                break;
            }
        }
        x1.push(a);
        x2.push(b);
        x3.push(c);
        cls_code.push(chosen);
        true_prob_by_code.push(p);
    }

    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<StringRecord> = (0..N_HETERO)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                x3[i].to_string(),
                format!("c{}", cls_code[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode hetero multinomial");

    let cfg = FitConfig::default();
    // The #1373 fix: x1 at k=12 (~11 spline df > the true df≈8) so the basis can
    // EXPRESS the wiggle; x2 stays modest at k=6 (true df≈2). Adaptive per-term
    // REML must then recover the surface, not be capped by basis capacity.
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, "y ~ s(x1, k=12) + s(x2, k=6) + x3", &cfg)
    })
    .expect("gam hetero multinomial fit");
    assert_eq!(
        model.class_levels.len(),
        K,
        "gam should recover K=3 classes"
    );

    let gam_probs = predict_multinomial_formula(&model, &ds).expect("gam predict probabilities");
    assert_eq!(
        gam_probs.dim(),
        (N_HETERO, K),
        "gam probability matrix shape"
    );

    // Probabilities must lie on the simplex (closure + range), independent of R.
    let mut worst_row_sum_err = 0.0_f64;
    let mut min_entry = f64::INFINITY;
    let mut max_entry = f64::NEG_INFINITY;
    for i in 0..N_HETERO {
        let mut s = 0.0;
        for k in 0..K {
            let v = gam_probs[[i, k]];
            s += v;
            min_entry = min_entry.min(v);
            max_entry = max_entry.max(v);
        }
        worst_row_sum_err = worst_row_sum_err.max((s - 1.0).abs());
    }
    assert!(
        worst_row_sum_err < 1e-6,
        "hetero fitted probabilities are not on the simplex: worst row-sum error={worst_row_sum_err:.2e}"
    );
    assert!(
        min_entry >= -1e-9 && max_entry <= 1.0 + 1e-9,
        "hetero fitted probabilities escape [0,1]: min={min_entry:.4} max={max_entry:.4}"
    );

    // RMSE of gam's fitted simplex against the TRUE simplex, in gam's class order.
    let gam_levels: Vec<String> = model.class_levels.clone();
    let col_code: Vec<usize> = gam_levels
        .iter()
        .map(|lvl| {
            lvl.trim_start_matches('c')
                .parse::<usize>()
                .expect("gam level label is c<code>")
        })
        .collect();
    let mut sse = 0.0_f64;
    for (k, &code) in col_code.iter().enumerate() {
        for i in 0..N_HETERO {
            let d = gam_probs[[i, k]] - true_prob_by_code[i][code];
            sse += d * d;
        }
    }
    let gam_err = (sse / (N_HETERO * K) as f64).sqrt();

    eprintln!(
        "#1373 hetero multinomial (k=12 on x1): gam_RMSE_vs_truth={gam_err:.5} \
         iters={}",
        model.iterations
    );

    // The absolute truth bar (0.10) is now REACHABLE because the basis can
    // express the df≈8 wiggle. A regression to k=6 makes the oracle ~0.186 and
    // this fails — exactly the basis-capacity confound #e2afbf80d removed. This
    // is the SAME 0.10 bar the parent fixture asserts; it is NOT loosened.
    assert!(
        gam_err <= 0.10,
        "gam does not recover the heterogeneous true simplex at its true-df basis \
         (k=12 on x1): RMSE(P_gam, P_true)={gam_err:.5} > 0.10. A regression of the \
         #e2afbf80d basis-capacity fix (x1 back to k=6, ~5 spline df < the true df≈8) \
         pushes even the unpenalized oracle to ~0.186."
    );
}

/// Deterministic 15×20 Poisson surface (n=300), identical to the
/// `quality_vs_mgcv_poisson_tensor` fixture DGP: log-mean
/// `eta = 0.8 + 0.3·sin(x) + 0.2·z²`, `y ~ Poisson(exp(eta))`, only the response
/// carries noise. Returns `(x, z, y, mu_true)`.
fn make_poisson_tensor_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nx = 15usize;
    let nz = 20usize;
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut y = Vec::with_capacity(nx * nz);
    let mut mu_true = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        let xi = (ix as f64) / ((nx - 1) as f64) * (2.0 * PI);
        for iz in 0..nz {
            let zi = -1.0 + 2.0 * (iz as f64) / ((nz - 1) as f64);
            let eta = 0.8 + 0.3 * xi.sin() + 0.2 * zi * zi;
            let lambda = eta.exp();
            let pois = Poisson::new(lambda).expect("poisson lambda > 0");
            let yi: f64 = pois.sample(&mut rng);
            x.push(xi);
            z.push(zi);
            y.push(yi);
            mu_true.push(lambda);
        }
    }
    (x, y, z, mu_true)
}

/// #1373 (poisson-tensor arm): gam's Poisson(log) × tensor-product `te(x,z)` fit
/// must recover the known mean surface under the ABSOLUTE truth-recovery bar
/// (`RMSE(gam,truth) ≤ 0.18·range(mu)`) with effective df in a sane range. This
/// is the R-free, gam-side half of
/// `gam_poisson_tensor_recovers_true_mean_surface`; the SECONDARY
/// `gam_err ≤ 1.10·mgcv_err` match-or-beat arm (the open REML λ-precision gap) is
/// intentionally NOT asserted here — see the file header. Pinning the absolute
/// recovery catches a tensor-penalty / PIRLS-row break that would smear or bias
/// the surface (collapse to a near-flat plane or blow up the df).
#[test]
fn poisson_tensor_recovers_true_mean_surface_gam_side_1373() {
    init_parallelism();

    let (x, y, z, mu_true) = make_poisson_tensor_data(345);
    let n = x.len();
    assert_eq!(n, 300, "grid 15x20 => n=300");

    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for poisson(log) + te()");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild tensor design at training points");
    let gam_eta = design.design.apply(&fit.fit.beta);
    let gam_mean: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    let mut sse = 0.0_f64;
    for i in 0..n {
        let d = gam_mean[i] - mu_true[i];
        sse += d * d;
    }
    let gam_err = (sse / n as f64).sqrt();

    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = mu_max - mu_min;

    eprintln!(
        "#1373 poisson te(x,z) gam-side recovery: n={n} mu_range=[{mu_min:.3},{mu_max:.3}] \
         gam_rmse_to_truth={gam_err:.4} gam_edf={gam_edf:.3}"
    );

    // PRIMARY (R-free, tool-independent): gam recovers the true mean surface
    // under the same absolute bar the parent fixture uses (0.18·range). This is
    // the truth-recovery claim, not the match-or-beat-mgcv claim.
    let abs_bar = 0.18 * mu_range;
    assert!(
        gam_err <= abs_bar,
        "Poisson+te() failed to recover the true mean surface: \
         RMSE(gam, truth)={gam_err:.4} > {abs_bar:.4} (= 0.18 * range {mu_range:.4})"
    );

    // EDF must be non-degenerate (more than a flat plane) and far below the full
    // 6*6-1 = 35-dim tensor basis. This catches a fit that collapses to a
    // constant (edf→1, would also smear the surface) or saturates the basis.
    assert!(
        gam_edf > 1.0 && gam_edf < 35.0,
        "Poisson+te() effective degrees of freedom outside the sane range (1, 35): \
         gam_edf={gam_edf:.3}"
    );
}
