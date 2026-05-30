//! End-to-end quality: gam's *multinomial* GAM with a smooth-by-factor
//! interaction (`s(x) + s(x, by=group)`) must RECOVER THE TRUE class-probability
//! surface that generated the data — an objective accuracy claim, not "matches a
//! reference tool's fitted output".
//!
//! OBJECTIVE METRIC (the pass criterion): TRUTH RECOVERY. The labels are drawn
//! from a known 3-class softmax whose log-odds `true_eta(x, g)` are a closed-form
//! group-specific smooth function of `x`. The true class-probability surface on
//! the evaluation grid is therefore known exactly. We assert that gam's fitted
//! probability surface has small RMSE against that TRUTH:
//!   * `rmse(gam_probs, true_probs) <= PROB_RMSE_BAR` (absolute accuracy bar on
//!     the simplex), and
//!   * `rmse(gam) <= rmse(VGAM) * 1.10` (gam matches-or-beats the mature tool on
//!     the SAME truth-recovery error — accuracy, not mutual agreement).
//! The primary claim is "gam recovers the data-generating probabilities". VGAM is
//! demoted to a BASELINE TO MATCH-OR-BEAT on that same objective error; it is no
//! longer the thing gam is asserted to reproduce. (We still compute VGAM's fit and
//! print gam↔VGAM rel_l2 for context only — it is never a pass/fail criterion.)
//!
//! Reference baseline tool: **VGAM** (`VGAM::vglm` with `family = multinomial()`
//! and a group-crossed natural-cubic-spline basis `ns(x, df)`). VGAM is the
//! canonical R package for multi-class softmax GLM/GAM. `vglm` is used over
//! `vgam` because VGAM's `s()` smoother takes **no `by=` argument** (a per-group
//! smooth-by-factor is an mgcv feature, not a VGAM one), so the standard VGAM
//! idiom for "a separate smooth curve of x per group" is to freeze one spline
//! basis on the training x and cross it with the group factor —
//! `vglm(y ~ grp + grp:ns(x, df))`. VGAM's `multinomial(refLevel = K)` makes the
//! last factor level the η ≡ 0 baseline, which is *exactly* gam's softmax gauge
//! (`MultinomialLogitLikelihood::softmax_with_baseline`, reference = last
//! `class_levels` entry); we pin VGAM's factor levels to gam's reported
//! `class_levels` order so both engines share the identical reference class and
//! every emitted probability column aligns class-for-class — to TRUTH and to each
//! other.
//!
//! gam's `s(x, by=group)` over a categorical `group` builds (see
//! `terms::term_builder`): a treatment-coded `group` main effect, one penalized
//! smooth of x *per group level*, plus the shared global `s(x)`. Both engines
//! therefore endow each group with its own smooth η-curve for every active class.
//!
//! Data: a synthetic, fixed-seed (RNG-reproducible) 3-class softmax draw whose
//! true log-odds carry a genuinely *group-specific* smooth shape of x (the loaded
//! combination the spec targets: a smooth crossed with class AND with a grouping
//! factor in a real K = 3 response). The identical numeric table is handed to gam
//! and to VGAM. We evaluate both fits — and the closed-form truth — on a dense
//! grid of x over its observed range at each of the three group levels, exercising
//! the full smooth-by-factor surface.

use csv::StringRecord;
use gam::data::{EncodedDataset, UnseenCategoryPolicy, encode_recordswith_schema};
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N_PER_GROUP: usize = 160;
const N_GROUPS: usize = 3;
const N: usize = N_PER_GROUP * N_GROUPS;
const K: usize = 3;

/// True K-class log-odds at covariate `x` in group `g`, with the reference
/// class K-1 pinned to η = 0. Active classes 0 and 1 each carry a smooth shape
/// of `x` whose curvature and phase *depend on the group* — that is the
/// smooth-by-factor signal both engines must recover. Identical math drives the
/// labels handed to gam and VGAM, so the comparison is honest.
fn true_eta(x: f64, g: usize) -> [f64; K] {
    // Per-group smooth shapes (genuinely nonlinear, distinct per group), plus a
    // per-group level shift (the `group` main effect). Class 0 follows a
    // group-shifted sine; class 1 follows a group-scaled cubic. The shapes are
    // smooth and strong so a light-penalty fit tracks them closely.
    let phase = g as f64 * 0.9;
    let amp = 1.0 + 0.6 * g as f64;
    let shift0 = 0.4 - 0.5 * g as f64;
    let shift1 = -0.3 + 0.4 * g as f64;
    let eta0 = shift0 + amp * (1.8 * x + phase).sin();
    let eta1 = shift1 + (0.9 + 0.5 * g as f64) * (x.powi(3) - 0.6 * x);
    [eta0, eta1, 0.0]
}

fn softmax(eta: &[f64; K]) -> [f64; K] {
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut ex = [0.0; K];
    let mut s = 0.0;
    for k in 0..K {
        ex[k] = (eta[k] - m).exp();
        s += ex[k];
    }
    for k in 0..K {
        ex[k] /= s;
    }
    ex
}

#[test]
fn gam_multinomial_smooth_by_factor_recovers_truth() {
    init_parallelism();

    // ---- synthesize the shared dataset (fixed seed, fed to BOTH engines) ----
    let mut rng = StdRng::seed_from_u64(0x5A7B_1234_u64);
    let ux = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform x");
    let udraw = Uniform::new(0.0_f64, 1.0_f64).expect("uniform draw");

    let mut x: Vec<f64> = Vec::with_capacity(N);
    let mut group_code: Vec<usize> = Vec::with_capacity(N);
    let mut cls_code: Vec<usize> = Vec::with_capacity(N);
    for g in 0..N_GROUPS {
        for _ in 0..N_PER_GROUP {
            let a = ux.sample(&mut rng);
            let p = softmax(&true_eta(a, g));
            // inverse-CDF class sample on a single uniform draw
            let u = udraw.sample(&mut rng);
            let mut acc = 0.0;
            let mut chosen = K - 1;
            for k in 0..K {
                acc += p[k];
                if u <= acc {
                    chosen = k;
                    break;
                }
            }
            x.push(a);
            group_code.push(g);
            cls_code.push(chosen);
        }
    }

    // ---- gam dataset: x continuous, group/y categorical via string labels ----
    // String labels make the inferred schema mark `group`/`y` categorical (gam)
    // and become factors (R). Level codes "g0".. / "c0".. encode in
    // first-appearance order; we build VGAM's factors from the SAME integer
    // codes (and pin gam's class order below), so both engines see identical
    // data and the same reference class (last level, η ≡ 0).
    let headers: Vec<String> = ["x", "group", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                format!("{}", x[i]),
                format!("g{}", group_code[i]),
                format!("c{}", cls_code[i]),
            ])
        })
        .collect();
    let ds: EncodedDataset =
        encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");

    // ---- fit with gam: y ~ s(x) + s(x, by=group), multinomial softmax --------
    // `s(x)` is the shared global smooth; `s(x, by=group)` adds the per-group
    // smooth-by-factor interaction (one penalized smooth per group level) plus a
    // treatment-coded `group` main effect.
    let cfg = FitConfig {
        family: Some("multinomial".to_string()),
        ..FitConfig::default()
    };
    let model = fit_penalized_multinomial_formula(
        &ds,
        "y ~ s(x, bs='tp') + s(x, by=group, bs='tp')",
        &cfg,
        1.0,  // init_lambda warm-start; outer REML selects per-class λ
        100,  // inner Newton cycles
        1e-8, // inner tolerance
    )
    .expect("gam multinomial fit");
    assert_eq!(model.class_levels.len(), K, "expected K=3 classes");
    assert_eq!(model.n_active_classes, K - 1, "K-1 = 2 active class blocks");

    // gam's level order is order-of-first-appearance; the reference class is the
    // LAST level (η = 0). Map each "c{code}" label back to its integer code so
    // VGAM can pin the identical factor-level order (and thus reference class).
    let gam_level_codes: Vec<usize> = model
        .class_levels
        .iter()
        .map(|lab| {
            lab.trim_start_matches('c')
                .parse::<usize>()
                .expect("gam class level label is c<code>")
        })
        .collect();

    // ---- evaluation grid: x over its observed range × each group level -------
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let grid_per_group = 40usize;
    let groups = [0usize, 1, 2];
    let mut grid_x: Vec<f64> = Vec::with_capacity(grid_per_group * groups.len());
    let mut grid_group: Vec<usize> = Vec::with_capacity(grid_per_group * groups.len());
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
                format!("g{}", grid_group[i]),
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
    assert_eq!(gam_probs.dim(), (n_grid, K));

    // ---- fit the SAME model with VGAM (the mature reference) -----------------
    // vglm + ns(x, df) crossed with the group factor: each group gets its own
    // spline coefficients (its own smooth η-curve per class), and the frozen
    // basis is evaluated at the grid x so train and grid share identical knots.
    // `multinomial(refLevel = K)` makes the last level the η ≡ 0 baseline; we
    // pin the factor levels to gam's class order so column k aligns class-for-
    // class. df = 5 sits near gam's effective per-group complexity for this
    // strong-signal n = N / K problem.
    let group_f64: Vec<f64> = group_code.iter().map(|&g| g as f64).collect();
    let cls_f64: Vec<f64> = cls_code.iter().map(|&c| c as f64).collect();
    let gam_level_codes_f64: Vec<f64> = (0..N)
        .map(|i| gam_level_codes[i % K] as f64) // tile K codes to length N (harness rejects ragged columns)
        .collect();

    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("grp", &group_f64),
            Column::new("yc", &cls_f64),
            Column::new("levorder", &gam_level_codes_f64),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(VGAM))
            suppressPackageStartupMessages(library(splines))
            # Recover gam's class-level order (first K tiled codes) and pin the
            # response factor to it: the LAST level is the multinomial reference
            # for both engines.
            lev_codes  <- unique(round(df$levorder))
            lev_labels <- paste0("c", lev_codes)
            df$yf  <- factor(paste0("c", round(df$yc)), levels = lev_labels)
            df$grp <- factor(round(df$grp), levels = c(0, 1, 2))
            # Freeze the spline basis on the training x so train and grid share
            # identical knots; cross it with the group factor for per-group
            # curves. The grp main effect carries the per-group level shift.
            xb <- ns(df$x, df = 5)
            df$xb <- xb
            m <- vglm(
                yf ~ grp + grp:xb,
                family = multinomial(refLevel = {ref_level}),
                data = df
            )
            gx  <- c({grid_x})
            gg  <- factor(c({grid_g}), levels = c(0, 1, 2))
            xbg <- predict(xb, newx = gx)   # evaluate the frozen basis at grid x
            nd  <- data.frame(grp = gg)
            nd$xb <- xbg
            pr  <- predict(m, newdata = nd, type = "response")
            # Columns are in factor-level order == gam's class order; emit by
            # gam's class index so the Rust side aligns class-for-class.
            emit("ncol", ncol(pr))
            emit("p0", as.numeric(pr[, 1]))
            emit("p1", as.numeric(pr[, 2]))
            emit("p2", as.numeric(pr[, 3]))
            "#,
            ref_level = K, // last level is the reference (1-based in R)
            grid_x = grid_x
                .iter()
                .map(|v| format!("{v:.10}"))
                .collect::<Vec<_>>()
                .join(","),
            grid_g = grid_group
                .iter()
                .map(|g| format!("{g}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    assert_eq!(
        r.scalar("ncol") as usize,
        K,
        "VGAM returned K probability columns"
    );
    let p0 = r.vector("p0");
    let p1 = r.vector("p1");
    let p2 = r.vector("p2");
    assert_eq!(p0.len(), n_grid, "VGAM grid length mismatch");

    // ---- TRUTH: closed-form data-generating probabilities on the grid -------
    // The labels were drawn from softmax(true_eta(x, g)); evaluating that exact
    // function on the SAME grid (grid_x, grid_group) gives the true class-
    // probability surface that the fit is trying to recover. This is the
    // objective target — independent of any reference tool.
    let mut true_c0: Vec<f64> = Vec::with_capacity(n_grid);
    let mut true_c1: Vec<f64> = Vec::with_capacity(n_grid);
    let mut true_c2: Vec<f64> = Vec::with_capacity(n_grid);
    for i in 0..n_grid {
        let p = softmax(&true_eta(grid_x[i], grid_group[i]));
        true_c0.push(p[0]);
        true_c1.push(p[1]);
        true_c2.push(p[2]);
    }
    let mut flat_true: Vec<f64> = Vec::with_capacity(K * n_grid);
    flat_true.extend_from_slice(&true_c0);
    flat_true.extend_from_slice(&true_c1);
    flat_true.extend_from_slice(&true_c2);

    // gam_probs columns follow model.class_levels == VGAM's pinned factor order
    // == true_eta's class index, so column k corresponds to true class k and to
    // VGAM's pr[, k+1] (= emit pk).
    let gam_c0: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 0]]).collect();
    let gam_c1: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 1]]).collect();
    let gam_c2: Vec<f64> = (0..n_grid).map(|i| gam_probs[[i, 2]]).collect();

    let mut flat_gam: Vec<f64> = Vec::with_capacity(K * n_grid);
    flat_gam.extend_from_slice(&gam_c0);
    flat_gam.extend_from_slice(&gam_c1);
    flat_gam.extend_from_slice(&gam_c2);
    let mut flat_ref: Vec<f64> = Vec::with_capacity(K * n_grid);
    flat_ref.extend_from_slice(p0);
    flat_ref.extend_from_slice(p1);
    flat_ref.extend_from_slice(p2);

    // ---- OBJECTIVE accuracy: RMSE of each fit against the TRUTH -------------
    let gam_truth_rmse = rmse(&flat_gam, &flat_true);
    let ref_truth_rmse = rmse(&flat_ref, &flat_true);

    // Context only (never a pass/fail criterion): how close the two FITS are to
    // each other, and per-class shape correlation against truth.
    let rel_gam_ref = relative_l2(&flat_gam, &flat_ref);
    let corr0 = pearson(&gam_c0, &true_c0);
    let corr1 = pearson(&gam_c1, &true_c1);
    let corr2 = pearson(&gam_c2, &true_c2);

    eprintln!(
        "multinomial s(x)+s(x,by=group): N={N} K={K} grid={n_grid} converged={} \
         gam_truth_rmse={gam_truth_rmse:.5} vgam_truth_rmse={ref_truth_rmse:.5} \
         rel_l2(gam,vgam)={rel_gam_ref:.4} \
         pearson_vs_truth(c0)={corr0:.5} (c1)={corr1:.5} (c2)={corr2:.5} \
         lambdas={:?}",
        model.converged, model.lambdas
    );

    // PRIMARY claim — gam recovers the data-generating probability surface.
    // Bar: 0.05 RMSE on the [0,1] simplex. The probabilities span essentially
    // the full simplex range (each class crosses from near-0 to near-1 across
    // the group-specific smooth signal), so a 0.05 RMSE is a small fraction of
    // the signal range — a genuine accuracy bar, not a loose one. It is the
    // honest irreducible floor here: with finite per-group N the fitted surface
    // cannot reach the noise-free truth exactly, but a faithful penalized
    // smooth-by-factor softmax must land within a few percent of it.
    const PROB_RMSE_BAR: f64 = 0.05;
    assert!(
        gam_truth_rmse <= PROB_RMSE_BAR,
        "gam did not recover the true multinomial probability surface: \
         rmse(gam, truth)={gam_truth_rmse:.5} > {PROB_RMSE_BAR}"
    );

    // SECONDARY claim — gam matches or beats the mature baseline (VGAM) on the
    // SAME objective truth-recovery error (accuracy, not mutual agreement). A
    // 10% slack absorbs the legitimate basis/penalty difference (gam's penalized
    // thin-plate vs VGAM's fixed-df natural cubic spline) without letting gam be
    // meaningfully less accurate than the trusted reference.
    assert!(
        gam_truth_rmse <= ref_truth_rmse * 1.10,
        "gam is less accurate at recovering the truth than the VGAM baseline: \
         rmse(gam, truth)={gam_truth_rmse:.5} > 1.10 * rmse(vgam, truth)={ref_truth_rmse:.5}"
    );
}
