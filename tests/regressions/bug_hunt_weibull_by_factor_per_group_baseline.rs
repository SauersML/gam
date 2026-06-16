//! Regression for #900: a Weibull-AFT fit with a **factor by-smooth**
//! covariate effect `s(x, by=group)` must carry each group's own baseline
//! LEVEL, not drop or shrink it.
//!
//! ## Root cause (fixed in `src/terms/smooth.rs`)
//!
//! `s(x, by=group)` for a categorical `group` expands into one
//! `ByVariable { kind: Level }` smooth block per level, each the inner basis
//! multiplied by that level's row indicator, plus an auto-added unpenalized
//! treatment-coded factor main effect (a random-effect term) that is supposed
//! to carry the per-group baseline level. Every per-level gated smooth block's
//! column span contains the per-level CONSTANT — a vector that is `1` on that
//! level's rows and `0` elsewhere — which is exactly the column the factor main
//! effect carries. The global-identifiability step centered each level's gated
//! design against the GLOBAL intercept, removing only its global mean and
//! leaving the within-level constant to collide with the factor main effect: a
//! rank-1 collinearity. The penalty/ridge then split the per-group baseline
//! level between the two collinear blocks and under-recovered it, so the
//! per-group log-cumulative-hazard offset (~0.69 in this DGP) leaked out and the
//! whole survival surface was miscalibrated (RMSE(S vs truth) ≈ 0.69).
//!
//! The fix centers each factor-by-level smooth against its GATED level
//! indicator, removing the within-level constant cleanly and leaving the level
//! entirely to the factor main effect (mgcv's by-factor convention), while the
//! per-level slope/curvature deviation stays in the smooth.
//!
//! ## Objective metric
//!
//! The data are simulated from an explicit AFT model with known per-group
//! Weibull baselines and known per-group acceleration slopes, so the true
//! survival surface `S_true(t | x, g)` is computable in closed form. The
//! assertions are:
//!   1. `RMSE( S_gam, S_true )` over a (group × x × t) grid is well under the
//!      0.69 the bug produced — held to the same tight bar the other survival
//!      truth-recovery quality tests use (`<= 0.15`).
//!   2. The two groups' baseline survival is correctly SEPARATED and ORDERED:
//!      at the reference covariate `x = 0`, group B (larger Weibull scale ⇒
//!      longer survival) must have strictly higher survival than group A across
//!      the grid, by a margin near the true inter-group gap — a broken fit that
//!      collapses the per-group level would predict the two groups on top of
//!      each other.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::families::survival::construction::evaluate_survival_baseline;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal, Weibull};

const N_PER_GROUP: usize = 100;
const SEED: u64 = 20260609;

// rand_distr::Weibull::new(scale, shape): CDF 1 - exp(-(t/scale)^shape). A
// COMMON shape, distinct scales (the per-group LEVEL the factor main effect
// must carry) and opposite-sign AFT acceleration slopes (the per-group signal
// the by-factor smooth must recover).
const SCALE_A: f64 = 0.8;
const SHAPE_A: f64 = 1.1;
const SCALE_B: f64 = 1.5;
const SHAPE_B: f64 = 1.1;
const BETA_A: f64 = 0.35;
const BETA_B: f64 = -0.25;

/// Closed-form ground-truth survival under the data-generating AFT model:
/// `T = T0 · exp(beta_g · x)`, `T0 ~ Weibull(scale_g, shape_g)`, hence
/// `S(t | x, g) = exp( -( t·exp(-beta_g·x) / scale_g )^shape_g )`.
fn true_survival(t: f64, x: f64, group_a: bool) -> f64 {
    let (scale, shape, beta) = if group_a {
        (SCALE_A, SHAPE_A, BETA_A)
    } else {
        (SCALE_B, SHAPE_B, BETA_B)
    };
    let s = t * (-beta * x).exp();
    (-((s / scale).powf(shape))).exp()
}

#[test]
fn gam_weibull_by_factor_recovers_per_group_baseline_level() {
    init_parallelism();

    // ---- synthesize data: group A rows first, then group B -----------------
    let mut rng = StdRng::seed_from_u64(SEED);
    let weib_a = Weibull::new(SCALE_A, SHAPE_A).expect("weibull A");
    let weib_b = Weibull::new(SCALE_B, SHAPE_B).expect("weibull B");
    let xdist = Normal::new(0.0, 1.0).expect("normal x");
    let cens_a = Exp::new(0.25_f64).expect("exp censor A");
    let cens_b = Exp::new(0.20_f64).expect("exp censor B");

    let n = 2 * N_PER_GROUP;
    let mut time = Vec::<f64>::with_capacity(n);
    let mut event = Vec::<f64>::with_capacity(n);
    let mut x = Vec::<f64>::with_capacity(n);
    let mut is_a = Vec::<bool>::with_capacity(n);

    for group_a in [true, false] {
        let (weib, beta, cens) = if group_a {
            (&weib_a, BETA_A, &cens_a)
        } else {
            (&weib_b, BETA_B, &cens_b)
        };
        for _ in 0..N_PER_GROUP {
            let xi = xdist.sample(&mut rng);
            let t0 = weib.sample(&mut rng);
            let ti = t0 * (beta * xi).exp();
            let ci = cens.sample(&mut rng) + 1e-3;
            let observed = ti.min(ci);
            let ev = if ti <= ci { 1.0 } else { 0.0 };
            time.push(observed);
            event.push(ev);
            x.push(xi);
            is_a.push(group_a);
        }
    }

    // ---- fit gam: Weibull AFT + by-FACTOR smooth on x ----------------------
    // `group` MUST be a categorical label ("A"/"B") so `s(x, by=group)` is the
    // per-level by-FACTOR expansion (one smooth per level + a treatment-coded
    // factor main effect) this test exercises, not a numeric varying coefficient.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    let group_label = |group_a: bool| if group_a { "A" } else { "B" };
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                time[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                group_label(is_a[i]).to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["group"];

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, event) ~ x + s(x, by=group) + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &ds,
        &cfg,
    )
    .expect("gam Weibull-AFT by-factor fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };

    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );

    // ---- predicted survival surface via gam's own forward map --------------
    // For each (group, x) build the covariate design from the frozen spec, dot
    // with the covariate coefficient slice for `cov_eta`, then
    // `S = exp(-exp(log H0(t) + cov_eta))` with the shared Weibull baseline.
    let x_eval = [-1.0_f64, 0.0, 1.0];
    let t_grid: Vec<f64> = (1..=10).map(|k| 0.25 * k as f64).collect();
    let pred_groups = [0.0_f64, 1.0];
    let n_pred_rows = pred_groups.len() * x_eval.len();

    let mut grid = Array2::<f64>::zeros((n_pred_rows, ds.headers.len()));
    let mut row = 0usize;
    for &gc in &pred_groups {
        for &xv in &x_eval {
            grid[[row, x_idx]] = xv;
            grid[[row, g_idx]] = gc;
            row += 1;
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at prediction rows");
    let dense = design.design.to_dense();
    assert_eq!(
        dense.ncols(),
        beta.len() - cov_start,
        "covariate design width must match the covariate coefficient slice"
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    let nt = t_grid.len();
    let mut gam_surv: Vec<f64> = Vec::with_capacity(n_pred_rows * nt);
    let mut true_surv: Vec<f64> = Vec::with_capacity(n_pred_rows * nt);
    let mut idx = 0usize;
    // Capture the reference-covariate (x = 0) survival curve per group so we can
    // assert the per-group baseline LEVEL is separated and ordered.
    let mut surv_x0_group_a: Vec<f64> = Vec::with_capacity(nt);
    let mut surv_x0_group_b: Vec<f64> = Vec::with_capacity(nt);
    for &gc in &pred_groups {
        let group_a = gc == 0.0;
        for &xv in &x_eval {
            let cov_eta: f64 = dense.row(idx).dot(&cov_beta);
            for &t in &t_grid {
                let (log_h0, _) = evaluate_survival_baseline(t, &fit.baseline_cfg)
                    .expect("evaluate gam baseline log-cumulative-hazard");
                let s = (-(log_h0 + cov_eta).exp()).exp();
                gam_surv.push(s);
                true_surv.push(true_survival(t, xv, group_a));
                if xv == 0.0 {
                    if group_a {
                        surv_x0_group_a.push(s);
                    } else {
                        surv_x0_group_b.push(s);
                    }
                }
            }
            idx += 1;
        }
    }

    // ---- (1) truth recovery: RMSE well under the 0.69 the bug produced -----
    let rmse = {
        let sumsq: f64 = gam_surv
            .iter()
            .zip(&true_surv)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (sumsq / gam_surv.len() as f64).sqrt()
    };
    eprintln!(
        "weibull-AFT by-factor #900: n={n} RMSE(S vs truth)={rmse:.4} \
         baseline scale={:?} shape={:?}",
        fit.baseline_cfg.scale, fit.baseline_cfg.shape
    );
    assert!(
        rmse <= 0.15,
        "gam fails to recover the true per-group Weibull-AFT survival surface: \
         RMSE(S vs truth)={rmse:.4} > 0.15 (the #900 collapsed-baseline bug produced ~0.69)"
    );

    // ---- (2) per-group baseline LEVEL is separated and correctly ordered ---
    // Group B has the larger Weibull scale (1.5 vs 0.8) ⇒ longer survival ⇒
    // strictly higher S(t | x=0) than group A. The true inter-group gap at
    // x = 0 averages ~0.16 over this grid; a fit that dropped the per-group
    // level would predict the groups on top of each other (gap ≈ 0).
    assert_eq!(surv_x0_group_a.len(), nt);
    assert_eq!(surv_x0_group_b.len(), nt);
    let mut sum_gap = 0.0_f64;
    let mut min_gap = f64::INFINITY;
    for k in 0..nt {
        let gap = surv_x0_group_b[k] - surv_x0_group_a[k];
        sum_gap += gap;
        min_gap = min_gap.min(gap);
    }
    let mean_gap = sum_gap / nt as f64;
    let true_mean_gap = {
        let mut s = 0.0;
        for &t in &t_grid {
            s += true_survival(t, 0.0, false) - true_survival(t, 0.0, true);
        }
        s / nt as f64
    };
    eprintln!(
        "  per-group x=0 survival gap: gam mean={mean_gap:.4} min={min_gap:.4} \
         (true mean={true_mean_gap:.4})"
    );
    assert!(
        min_gap > 0.0,
        "group B (larger scale, longer survival) must have higher S than group A at every t; \
         min gap={min_gap:.4} <= 0 means the per-group baseline level collapsed"
    );
    assert!(
        mean_gap >= 0.5 * true_mean_gap,
        "gam under-recovers the per-group baseline separation: mean gap={mean_gap:.4}, \
         true mean gap={true_mean_gap:.4} (need >= half the true gap)"
    );
}
