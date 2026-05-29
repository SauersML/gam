//! End-to-end quality: gam's *multinomial* GAM with a smooth-by-factor
//! interaction must match VGAM — the mature, standard multinomial-GLM/GAM
//! implementation — on real data, not merely run without panicking.
//!
//! Reference tool: **VGAM** (`VGAM::vgam` with `family = multinomial()`).
//! VGAM is the canonical R package for vector-response GLM/GAM families;
//! `multinomial()` is its reference-coded softmax (last factor level is the
//! baseline with η ≡ 0), which is *exactly* gam's softmax gauge
//! (`MultinomialLogitLikelihood::softmax_with_baseline`, reference class =
//! last level). `vgam` fits a per-predictor penalized smoother through `s(x)`,
//! and a smooth-by-group interaction is expressed by adding a per-group smooth
//! term — the same structure gam builds for `s(x) + s(x, by=group)` (a global
//! smooth, per-level smooths, and a treatment-coded factor main effect, see
//! `terms::term_builder`). This is the loaded combination the spec targets:
//! a smooth term crossed with class *and* crossed with a grouping factor in a
//! genuinely multi-class (K = 3) response.
//!
//! Data: the real `wine.csv` vintage table. To obtain an honest 3-class
//! structure with continuous covariates we use the complete-case rows
//! (1952–1989, where `price` is observed) and derive, RNG-free and purely by
//! data-driven quantiles:
//!   * smooth covariate `x`  = harvest temperature `h_temp` (real continuous),
//!   * grouping factor `group` = tertiles of summer temperature `s_temp`,
//!   * 3-class response `y`   = tertiles of `price` (a real economic outcome).
//! The identical numeric table is handed to gam and to VGAM, so any divergence
//! is a real modelling difference, not a data-prep artifact.
//!
//! We compare *fitted class probabilities* on a dense grid of `x` over its
//! observed range, holding `group` at each of its three levels (so the grid
//! exercises the full smooth-by-factor surface). Probabilities are the
//! quantity that matters for a classifier: they are gauge-invariant (unlike the
//! reference-coded coefficients) and directly comparable element-wise between
//! the two engines. Both engines fit a penalized softmax additive model with
//! the same reference class, so close agreement is the correct expectation and
//! a real divergence is a real bug.

use gam::data::{EncodedDataset, UnseenCategoryPolicy, encode_recordswith_schema};
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use csv::StringRecord;
use ndarray::Array2;

/// Complete-case rows of the real `wine.csv` (years 1952–1989, where `price`
/// is observed). Columns: (h_temp, s_temp, price). `parker` is dropped (NA for
/// these vintages) and `year` is an index, not a covariate.
const WINE_ROWS: &[(f64, f64, f64)] = &[
    (14.3, 17.1, 37.0),
    (17.3, 16.7, 63.0),
    (16.8, 15.4, 12.0),
    (16.8, 17.1, 45.0),
    (17.2, 15.6, 15.0),
    (16.2, 16.1, 22.0),
    (19.1, 16.4, 18.0),
    (18.7, 17.5, 66.0),
    (15.8, 16.4, 14.0),
    (20.4, 17.3, 100.0),
    (17.2, 16.3, 33.0),
    (16.2, 15.7, 17.0),
    (18.8, 17.3, 31.0),
    (14.8, 15.4, 11.0),
    (18.4, 16.5, 47.0),
    (16.5, 16.2, 19.0),
    (16.4, 16.2, 11.0),
    (16.6, 16.5, 12.0),
    (18.0, 16.7, 40.0),
    (16.9, 16.8, 27.0),
    (14.6, 15.0, 10.0),
    (17.9, 17.1, 16.0),
    (16.2, 16.3, 11.0),
    (17.2, 16.9, 30.0),
    (16.1, 17.6, 25.0),
    (16.8, 15.6, 11.0),
    (17.4, 15.8, 27.0),
    (17.3, 16.2, 21.0),
    (18.4, 16.0, 14.0),
    (18.0, 17.0, 17.0),
    (18.5, 17.4, 37.0),
    (17.9, 17.4, 17.0),
    (16.0, 16.5, 11.0),
    (18.9, 16.8, 18.0),
    (17.5, 16.3, 21.0),
    (18.9, 17.0, 15.0),
    (16.8, 17.1, 17.0),
    (18.4, 18.6, 23.0),
];

/// Deterministic tertile-cut of `vals` into 3 integer labels {0,1,2} using the
/// empirical 1/3 and 2/3 sample quantiles (linear interpolation, type-7 — the
/// default both R's `quantile` and this routine use). RNG-free and identical
/// across engines.
fn tertile_labels(vals: &[f64]) -> Vec<usize> {
    let q = |p: f64| -> f64 {
        let mut s: Vec<f64> = vals.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = (s.len() as f64 - 1.0) * p;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        s[lo] + (h - lo as f64) * (s[hi] - s[lo])
    };
    let q1 = q(1.0 / 3.0);
    let q2 = q(2.0 / 3.0);
    vals.iter()
        .map(|&v| {
            if v <= q1 {
                0
            } else if v <= q2 {
                1
            } else {
                2
            }
        })
        .collect()
}

/// Assemble the shared modelling table for gam, plus the raw numeric columns
/// that VGAM will receive. `y`/`group` carry string labels so the inferred
/// schema marks them categorical (gam) and they become factors (R).
fn build_dataset() -> (EncodedDataset, Vec<f64>, Vec<f64>, Vec<f64>) {
    let h_temp_raw: Vec<f64> = WINE_ROWS.iter().map(|r| r.0).collect();
    let s_temp_raw: Vec<f64> = WINE_ROWS.iter().map(|r| r.1).collect();
    let price_raw: Vec<f64> = WINE_ROWS.iter().map(|r| r.2).collect();

    let group_lab_raw = tertile_labels(&s_temp_raw);
    let y_lab_raw = tertile_labels(&price_raw);

    // Reorder rows by the response label (stable) so the first appearance of
    // the categorical response is c0, then c1, then c2. The inferred schema
    // codes categorical levels in first-appearance order, so this pins gam's
    // `class_levels` to ["c0","c1","c2"] with c2 as the softmax reference (last
    // level) — matching VGAM's `multinomial(refLevel = 3)`. Row order is
    // irrelevant to a GAM fit (observations are exchangeable), so this is a
    // pure presentation choice that aligns the two engines' class columns.
    let mut order: Vec<usize> = (0..WINE_ROWS.len()).collect();
    order.sort_by_key(|&i| y_lab_raw[i]);

    let h_temp: Vec<f64> = order.iter().map(|&i| h_temp_raw[i]).collect();
    let group_lab: Vec<usize> = order.iter().map(|&i| group_lab_raw[i]).collect();
    let y_lab: Vec<usize> = order.iter().map(|&i| y_lab_raw[i]).collect();

    // gam dataset: x = h_temp (continuous), group/y categorical via string
    // labels ("g0".. / "c0"..). The exact same numeric partition (tertiles) is
    // handed to VGAM below, so the two engines see identical data.
    let headers: Vec<String> = ["x", "group", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..h_temp.len())
        .map(|i| {
            StringRecord::from(vec![
                format!("{}", h_temp[i]),
                format!("g{}", group_lab[i]),
                format!("c{}", y_lab[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode wine dataset");

    // Numeric copies for VGAM: y/group as 0/1/2 integer codes (identical
    // partition and row order as gam), x as the same h_temp values.
    let y_code: Vec<f64> = y_lab.iter().map(|&c| c as f64).collect();
    let group_code: Vec<f64> = group_lab.iter().map(|&c| c as f64).collect();
    (ds, h_temp, group_code, y_code)
}

#[test]
fn gam_multinomial_smooth_by_factor_matches_vgam_on_wine() {
    init_parallelism();

    let (ds, x, group_code, y_code) = build_dataset();
    let n = x.len();
    assert_eq!(n, WINE_ROWS.len());

    // ---- fit with gam: y ~ s(x) + s(x, by=group), multinomial softmax ------
    // `s(x)` is the shared global smooth; `s(x, by=group)` adds the per-group
    // smooth-by-factor interaction (and a treatment-coded `group` main effect).
    let cfg = FitConfig {
        family: Some("multinomial".to_string()),
        ..FitConfig::default()
    };
    let model = fit_penalized_multinomial_formula(
        &ds,
        "y ~ s(x, bs='tp') + s(x, by=group, bs='tp')",
        &cfg,
        1.0,   // init_lambda warm-start; outer REML selects per-class λ
        100,   // inner Newton cycles
        1e-8,  // inner tolerance
    )
    .expect("gam multinomial fit");
    assert_eq!(model.class_levels.len(), 3, "expected K=3 classes");
    assert_eq!(model.n_active_classes, 2, "K-1 = 2 active class blocks");

    // ---- evaluation grid: x over its observed range × each group level -----
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let grid_per_group = 40usize;
    let groups = [0.0_f64, 1.0, 2.0];
    let mut grid_x: Vec<f64> = Vec::with_capacity(grid_per_group * groups.len());
    let mut grid_group: Vec<f64> = Vec::with_capacity(grid_per_group * groups.len());
    for &g in &groups {
        for i in 0..grid_per_group {
            let t = i as f64 / (grid_per_group as f64 - 1.0);
            grid_x.push(x_min + t * (x_max - x_min));
            grid_group.push(g);
        }
    }
    let n_grid = grid_x.len();

    // gam predict: build the grid dataset under the *training* schema so the
    // categorical `group` levels encode to the identical numeric codes the
    // by=group smooth was frozen against (ByVariable matches on the encoded
    // value's bit pattern, see terms::smooth). A freshly-inferred schema could
    // assign different level codes by first-appearance order and silently break
    // the level match — encoding with `ds.schema` is the robust path.
    let grid_headers: Vec<String> = ["x", "group", "y"].into_iter().map(String::from).collect();
    // `y` is unused for prediction (the design only references x and group) but
    // every column must carry a schema-valid value; use a known class label.
    let grid_rows: Vec<StringRecord> = (0..n_grid)
        .map(|i| {
            StringRecord::from(vec![
                format!("{}", grid_x[i]),
                format!("g{}", grid_group[i] as usize),
                "c0".to_string(),
            ])
        })
        .collect();
    let grid_ds = encode_recordswith_schema(
        grid_headers,
        grid_rows,
        &ds.schema,
        UnseenCategoryPolicy::Error,
    )
    .expect("encode grid dataset under training schema");
    let gam_probs: Array2<f64> =
        predict_multinomial_formula(&model, &grid_ds).expect("gam grid prediction");
    assert_eq!(gam_probs.dim(), (n_grid, 3));

    // ---- fit the SAME model with VGAM (the mature reference) ---------------
    // VGAM's vglm multinomial is the standard multi-class softmax GLM. The
    // smooth-by-group interaction `s(x) + s(x, by=group)` is expressed as a
    // group main effect plus a per-group spline of x: a single global natural
    // cubic spline basis `ns(x, df)` crossed with the group factor gives each
    // group its own smooth η-curve for every class — exactly gam's per-level
    // smooth structure. `multinomial(refLevel = 3)` makes the last level
    // (class c2) the baseline with η ≡ 0, matching gam's softmax gauge
    // (reference = last class_levels entry), so the emitted probability columns
    // align with gam's class order c0, c1, c2. df = 4 is chosen to sit near
    // gam's per-class effective complexity for this n = 38 / K = 3 problem.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("grp", &group_code),
            Column::new("yc", &y_code),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(VGAM))
            suppressPackageStartupMessages(library(splines))
            df$yf  <- factor(df$yc,  levels = c(0,1,2))
            df$grp <- factor(df$grp, levels = c(0,1,2))
            # Freeze the spline basis on the training x so train and grid share
            # identical knots; cross it with the group factor for per-group
            # curves. grp main effect carries the per-group level shift.
            xb <- ns(df$x, df = 4)
            df$xb <- xb
            m <- vglm(
                yf ~ grp + grp:xb,
                family = multinomial(refLevel = 3),
                data = df
            )
            gx  <- c({grid_x})
            gg  <- factor(c({grid_g}), levels = c(0,1,2))
            xbg <- predict(xb, newx = gx)   # evaluate frozen basis at grid x
            nd <- data.frame(grp = gg)
            nd$xb <- xbg
            pr <- predict(m, newdata = nd, type = "response")
            # columns are class 0,1,2 in that order; emit column-major.
            emit("p0", as.numeric(pr[, 1]))
            emit("p1", as.numeric(pr[, 2]))
            emit("p2", as.numeric(pr[, 3]))
            "#,
            grid_x = grid_x
                .iter()
                .map(|v| format!("{v:.10}"))
                .collect::<Vec<_>>()
                .join(","),
            grid_g = grid_group
                .iter()
                .map(|v| format!("{}", *v as usize))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    let p0 = r.vector("p0");
    let p1 = r.vector("p1");
    let p2 = r.vector("p2");
    assert_eq!(p0.len(), n_grid, "VGAM grid length mismatch");

    // ---- compare fitted class probabilities, grid-aligned, per class -------
    let gam_c0: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 0]]).collect();
    let gam_c1: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 1]]).collect();
    let gam_c2: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 2]]).collect();

    let mut flat_gam: Vec<f64> = Vec::with_capacity(3 * n_grid);
    flat_gam.extend_from_slice(&gam_c0);
    flat_gam.extend_from_slice(&gam_c1);
    flat_gam.extend_from_slice(&gam_c2);
    let mut flat_ref: Vec<f64> = Vec::with_capacity(3 * n_grid);
    flat_ref.extend_from_slice(p0);
    flat_ref.extend_from_slice(p1);
    flat_ref.extend_from_slice(p2);

    let rel = relative_l2(&flat_gam, &flat_ref);
    let corr0 = pearson(&gam_c0, p0);
    let corr1 = pearson(&gam_c1, p1);
    let corr2 = pearson(&gam_c2, p2);

    eprintln!(
        "wine multinomial s(x)+s(x,by=group): n={n} K=3 grid={n_grid} \
         rel_l2={rel:.4} pearson(c0)={corr0:.5} pearson(c1)={corr1:.5} pearson(c2)={corr2:.5} \
         lambdas={:?}",
        model.lambdas
    );

    // Both engines fit a reference-coded penalized softmax additive model with
    // the same baseline class and the same smooth-by-group structure, so the
    // fitted class-probability *surfaces* must essentially coincide on the
    // grid. The Frobenius relative-L2 over the stacked (3 × n_grid) probability
    // matrix measures the whole surface; <0.015 leaves only the small slack
    // from differing smoother backends (gam's REML-penalized thin-plate vs
    // VGAM's natural-cubic-spline df=4 basis) while still catching any real
    // divergence. Each
    // class's grid trace must track the reference with pearson > 0.997 — a
    // genuinely matching shape, not merely a loose correlation.
    assert!(
        rel < 0.015,
        "fitted multinomial probability surfaces diverge from VGAM: rel_l2={rel:.4}"
    );
    assert!(
        corr0 > 0.997 && corr1 > 0.997 && corr2 > 0.997,
        "per-class probability traces should track VGAM: \
         pearson(c0)={corr0:.5} pearson(c1)={corr1:.5} pearson(c2)={corr2:.5}"
    );
}
