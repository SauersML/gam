//! End-to-end quality: gam's factor-smooth term `s(x, group, bs="fs")` must
//! match `mgcv` — the mature, standard GAM implementation and the canonical
//! reference for the `fs` ("factor smooth") penalty structure — on identical
//! data.
//!
//! mgcv's `bs="fs"` builds, for every level of a factor `group`, the *same*
//! marginal smooth basis in `x` with its own coefficients, penalized by a
//! *shared* smoothing parameter through a marginal second-derivative (wiggliness)
//! penalty AND a penalty on the marginal null space (so the per-level
//! intercept/linear trend are themselves shrunk — the random-effect flavour of a
//! smooth). It is the de-facto reference for "one smooth shape per group that
//! borrows strength across groups". gam implements this as
//! `FactorSmoothFlavour::Fs` (a B-spline marginal with `double_penalty=true`
//! plus a null-space penalty of order `m=2`), routed from the formula
//! `s(x, group, bs="fs")`.
//!
//! Both engines fit by REML, so they target the same penalized log-likelihood
//! and the per-group fitted curves should essentially coincide. We:
//!   1. fit `y ~ s(x, group, bs="fs")` with gam (REML, gaussian), and the SAME
//!      model with `mgcv::gam(..., method="REML")`;
//!   2. rebuild gam's design on a shared (x-grid x group) lattice and apply
//!      `beta` to get gam's fitted per-group curves; obtain mgcv's fitted curves
//!      at the IDENTICAL lattice via `predict`;
//!   3. assert the per-group fitted curves agree (relative L2 over the full
//!      lattice), that the curve SHAPE correlates (Pearson over the lattice), and
//!      that the single shared smooth EDF agrees in magnitude.
//!
//! A genuine divergence here is a real bug in gam's factor-smooth penalty (e.g.
//! a missing null-space penalty or a per-level rather than shared smoothing
//! parameter), and this test is meant to catch exactly that.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 400;
const N_GROUPS: usize = 4;
const SEED: u64 = 33;
const SIGMA: f64 = 0.15;
/// Amplitude of the shared sinusoidal shape; per-group offsets sit on top.
const AMP: f64 = 1.0;
/// Distinct per-group mean offsets μ_g — these live in the marginal null space
/// that the fs penalty shrinks, so they exercise the null-space penalty path.
const MU: [f64; N_GROUPS] = [-0.8, -0.2, 0.4, 1.0];

fn build_data() -> (gam::data::EncodedDataset, Vec<f64>, Vec<String>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");
    let four_pi = 4.0 * std::f64::consts::PI;

    let mut x = Vec::with_capacity(N);
    let mut grp = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let xi = ux.sample(&mut rng);
        let g = i % N_GROUPS; // round-robin keeps all groups well-populated
        let yi = MU[g] + AMP * (four_pi * xi).sin() + noise.sample(&mut rng);
        x.push(xi);
        grp.push(g);
        y.push(yi);
    }

    // Group labels "g0".."g3"; since rows 0..3 introduce g0,g1,g2,g3 in order,
    // gam's categorical encoder assigns level codes 0,1,2,3 in that order, which
    // we mirror on the prediction grid below.
    let headers = vec!["x".to_string(), "group".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                format!("g{}", grp[i]),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode factor-smooth data");

    // String columns handed to R (so mgcv sees the IDENTICAL factor).
    let group_labels: Vec<String> = grp.iter().map(|&g| format!("g{g}")).collect();
    (ds, y, group_labels)
}

#[test]
fn gam_factor_smooth_fs_matches_mgcv() {
    init_parallelism();

    let (ds, y, group_labels) = build_data();
    let colmap = ds.column_map();
    let x_idx = colmap["x"];
    let group_idx = colmap["group"];
    let x_vals: Vec<f64> = ds.values.column(x_idx).to_vec();

    // ---- fit with gam: y ~ s(x, group, bs="fs"), REML, gaussian -----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, group, bs=\"fs\")", &ds, &cfg).expect("gam factor-smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian factor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- shared evaluation lattice: x in {0,0.1,...,1.0} x each group ------
    let x_grid: Vec<f64> = (0..=10).map(|k| k as f64 / 10.0).collect();
    let n_eval = x_grid.len() * N_GROUPS;
    // gam side: build a design at the lattice. The group column carries the
    // categorical level CODE (0.0..3.0 as f64), matching the encoder above.
    let mut grid = Array2::<f64>::zeros((n_eval, ds.headers.len()));
    // R side: feed mgcv the IDENTICAL lattice as (x, group-label) pairs.
    let mut grid_x = Vec::with_capacity(n_eval);
    let mut grid_group = Vec::with_capacity(n_eval);
    let mut row = 0;
    for g in 0..N_GROUPS {
        for &xv in &x_grid {
            grid[[row, x_idx]] = xv;
            grid[[row, group_idx]] = g as f64;
            grid_x.push(xv);
            grid_group.push(format!("g{g}"));
            row += 1;
        }
    }
    assert_eq!(row, n_eval);

    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild factor-smooth design at lattice");
    let gam_curves: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_curves.len(), n_eval, "gam lattice prediction length");

    // ---- fit the SAME model with mgcv (the mature fs reference) -----------
    // `group` must be a factor; bs="fs" with method="REML" reproduces the shared
    // smoothing parameter + null-space penalty. We predict the smooth on the
    // identical lattice. mgcv emits the lattice columns row-major in the order
    // we send them, so the returned `fitted` aligns elementwise with gam_curves.
    let r = run_r(
        &[
            Column::new("x", &x_vals),
            // group passed as the integer code; R reconstructs the same factor.
            Column::new(
                "group_code",
                &group_labels
                    .iter()
                    .map(|s| s[1..].parse::<f64>().expect("group code"))
                    .collect::<Vec<f64>>(),
            ),
            Column::new("y", &y),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            df$group <- factor(paste0("g", df$group_code), levels = c({levels}))
            m <- gam(y ~ s(x, group, bs = "fs"), data = df, method = "REML")
            gx <- c({gx})
            gg <- factor(c({gg}), levels = c({levels}))
            nd <- data.frame(x = gx, group = gg)
            emit("fitted", as.numeric(predict(m, newdata = nd)))
            emit("edf", sum(m$edf))
            "#,
            levels = (0..N_GROUPS)
                .map(|g| format!("\"g{g}\""))
                .collect::<Vec<_>>()
                .join(", "),
            gx = grid_x
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(", "),
            gg = grid_group
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    );
    let mgcv_curves = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_curves.len(),
        n_eval,
        "mgcv lattice prediction length mismatch"
    );

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_curves, mgcv_curves);
    let corr = pearson(&gam_curves, mgcv_curves);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    // Per-group diagnostics so a divergence localizes to a level.
    for g in 0..N_GROUPS {
        let lo = g * x_grid.len();
        let hi = lo + x_grid.len();
        let rg = relative_l2(&gam_curves[lo..hi], &mgcv_curves[lo..hi]);
        eprintln!("[fs] group g{g}: rel_l2={rg:.4}");
    }
    eprintln!(
        "[fs] s(x,group,bs=fs): n={N} groups={N_GROUPS} gam_edf={gam_edf:.3} \
         mgcv_edf={mgcv_edf:.3} (rel={edf_rel:.3}) lattice_rel_l2={rel:.4} pearson={corr:.5}"
    );

    // Both engines REML-fit the identical fs penalty, so the per-group fitted
    // curves must nearly coincide. rel_l2 < 0.05 is the principled bound from the
    // spec: it is far tighter than any basis/parameterization quirk could excuse
    // (curves span ~[-1.8, 2.0]) yet leaves margin for REML λ-selection differing
    // in the last few significant digits.
    assert!(
        rel < 0.05,
        "factor-smooth fitted curves diverge from mgcv: rel_l2={rel:.4}"
    );
    // Shape correlation across the whole lattice: the fs penalty's defining
    // property is one COMMON shape per group, so the concatenated curves must be
    // essentially collinear with mgcv's.
    assert!(
        corr > 0.95,
        "factor-smooth curve shapes uncorrelated with mgcv: pearson={corr:.5}"
    );
    // Total smooth complexity (shared EDF across the fs block) is basis- and
    // null-space-convention sensitive (gam's B-spline marginal vs mgcv's default
    // tp marginal), so we assert same-ballpark complexity rather than equality:
    // within 30% relative, which still rejects a wrong penalty structure that
    // would mis-count degrees of freedom by 2x or collapse to a single curve.
    assert!(
        edf_rel < 0.30,
        "factor-smooth EDF disagrees with mgcv: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
