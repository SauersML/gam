//! End-to-end quality: gam's *multinomial* GAM with a smooth-by-factor
//! interaction (`s(x) + s(x, by=group)`) must RECOVER THE TRUE class-probability
//! surface that generated the data — an objective accuracy claim, not "matches a
//! reference tool's fitted output".
//!
//! OBJECTIVE METRIC (the pass criterion): TRUTH RECOVERY, scored relative to the
//! best a mature tool can achieve on this exact data. The labels are drawn from a
//! known 3-class softmax whose log-odds `true_eta(x, g)` are a closed-form
//! group-specific smooth function of `x`, so the true class-probability surface on
//! the evaluation grid is known exactly. We assert that gam's fitted probability
//! surface recovers that TRUTH. The group-specific signal is STRONG and
//! genuinely recoverable: the η's are high-amplitude monotone `tanh` ramps
//! (distinct direction/steepness per group), so the softmax surface sweeps the
//! simplex CORNERS — every class probability crosses from near 0 to near 1 across
//! x within each group. The truth is therefore FAR from the uniform 1/K centroid
//! (`rmse(uniform, truth) ≈ 0.36`), and a faithful per-group fit beats uniform by
//! a wide margin (a crude 12-bin per-group estimator already recovers it to
//! RMSE ≈ 0.09 at this N). We assert recovery on two objective axes:
//!   * `rmse(gam) <= rmse(VGAM) * 1.10` (gam matches-or-beats the mature tool on
//!     the SAME truth-recovery error — accuracy, not mutual agreement), AND
//!   * `rmse(gam) <= 0.85 * rmse(uniform)` (gam recovers genuine group/class
//!     structure, comfortably beating the trivial no-signal 1/K predictor — this
//!     guards against an over-smoothing collapse to the uniform surface, the
//!     failure mode a degenerate by-factor fit would exhibit).
//! Per the reference-as-truth paradigm, the mature tool is the match-or-beat
//! accuracy baseline, while the below-uniform guard certifies real structure
//! recovery. (We still compute VGAM's fit and print gam↔VGAM rel_l2 for context
//! only — it is never a pass/fail criterion.)
//!
//! A NOTE on calibration history: an earlier revision used a low-amplitude
//! *oscillating sine* signal at N = 480, which was information-pathological — the
//! truth sat so close to uniform that the no-signal 1/K predictor (RMSE ≈ 0.21)
//! BEAT both gam (0.305) and VGAM (0.314), and both tools fit the class-0
//! oscillation backwards. No estimator could recover that fixture. This revision
//! re-derives the truth with strong, monotone, corner-sweeping ramps so the
//! per-group structure is recoverable and both bars above are satisfiable by a
//! correct fit.
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
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::test_support::reference::{Column, QualityPair, pearson, relative_l2, rmse, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N_PER_GROUP: usize = 90;
const N_GROUPS: usize = 3;
const N: usize = N_PER_GROUP * N_GROUPS;
const K: usize = 3;

/// True K-class log-odds at covariate `x ∈ [-1.5, 1.5]` in group `g`, with the
/// reference class K-1 pinned to η = 0. Active classes 0 and 1 each carry a
/// strong, smooth, MONOTONE-RAMP shape of `x` whose direction and steepness
/// *depend on the group* — the smooth-by-factor signal both engines must
/// recover. Identical math drives the labels handed to gam and VGAM, so the
/// comparison is honest.
///
/// The shapes are deliberately high-amplitude `tanh` ramps (not the previous
/// low-amplitude oscillating sine): the η's sweep a wide range so the softmax
/// surface visits the simplex CORNERS (probabilities near 0 and near 1) rather
/// than hovering near the uniform centroid. That makes the per-group signal
/// genuinely recoverable from finite samples — the no-signal uniform predictor
/// is then far from the truth, and a faithful fit beats it by a wide margin.
/// `tanh` is monotone and gentle, so a penalized thin-plate / fixed-df spline
/// tracks it accurately (no high-frequency content to alias at this N).
fn true_eta(x: f64, g: usize) -> [f64; K] {
    let gf = g as f64;
    // Class 0: a group-specific tanh ramp in x. Group 0 ramps UP with x, group 2
    // ramps DOWN, group 1 is centered/steep — distinct directions and steepness
    // per group (the by=group interaction), each with large amplitude (±3.5).
    let slope0 = 2.2 - 1.6 * gf; // g0:+2.2  g1:+0.6 → steep center  g2:-1.0 (down)
    let center0 = -0.6 + 0.6 * gf; // shifts the ramp's inflection per group
    let eta0 = 3.5 * (slope0 * (x - center0)).tanh() + (1.0 - gf);
    // Class 1: a group-specific tanh ramp of opposite orientation, so classes 0
    // and 1 trade dominance across x differently in each group. Large amplitude
    // again, with a per-group level offset (the `group` main effect).
    let slope1 = -1.4 - 0.7 * gf; // all ramp DOWN, steepening with group
    let center1 = 0.4 - 0.5 * gf;
    let eta1 = 3.2 * (slope1 * (x - center1)).tanh() + (-0.8 + 0.9 * gf);
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
    // The smooth-by-factor model carries one λ for the global `s(x)` plus one
    // per group level, per active class. The native multinomial driver routes
    // this large smoothing-parameter vector through the scalable exact-gradient
    // outer path; convergence, not elapsed wall time, is the correctness gate.
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &ds,
        formula: "y ~ s(x, bs='tp', k=5) + s(x, by=group, bs='tp', k=5)",
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 60,
        tol: 1e-8,
    })
    .expect("gam multinomial fit");
    assert_eq!(model.class_levels.len(), K, "expected K=3 classes");
    assert_eq!(model.n_active_classes, K - 1, "K-1 = 2 active class blocks");
    // The per-class smoothing-parameter vector must NOT be fused (#561): with one
    // global smooth + one smooth per group level, each active class carries
    // n_terms = 4 independent λ. A fused single-λ-per-class driver would report
    // only K-1 = 2 total λ; the per-term driver reports (K-1)·n_terms.
    assert!(
        model.lambdas.len() >= (K - 1) * 2,
        "smooth-by-factor must select an independent λ per (class, term), not one \
         fused λ per class: got {} total λ for {} active classes",
        model.lambdas.len(),
        K - 1,
    );

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
    let grid_per_group = 25usize;
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
            xb <- ns(df$x, df = 4)
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
        "multinomial s(x)+s(x,by=group): N={N} K={K} grid={n_grid} \
         gam_truth_rmse={gam_truth_rmse:.5} vgam_truth_rmse={ref_truth_rmse:.5} \
         rel_l2(gam,vgam)={rel_gam_ref:.4} \
         pearson_vs_truth(c0)={corr0:.5} (c1)={corr1:.5} (c2)={corr2:.5} \
         lambdas={:?}",
        model.lambdas
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_vgam_multinomial_smooth_by_factor",
            "simplex_rmse_to_truth",
            gam_truth_rmse,
            "vgam",
            ref_truth_rmse,
        )
        .line()
    );

    // The truth-recovery target is scored two ways: reference-relative
    // (match-or-beat the mature tool) and against the no-signal baseline (the
    // strong, corner-sweeping signal makes uniform a poor predictor, so a
    // faithful fit must crush it). The strong monotone-ramp η's (see `true_eta`)
    // put real, recoverable per-group structure into the data, so both bars are
    // satisfiable by a correct fit — unlike the earlier oscillating-sine fixture
    // where the truth was so near uniform that no estimator could beat it.

    // PRIMARY claim — gam recovers the truth AT LEAST as accurately as the
    // mature reference (VGAM) on the SAME objective error. A 10% slack absorbs
    // the legitimate basis/penalty difference (gam's REML-penalized thin-plate
    // vs VGAM's fixed-df natural-cubic spline) without letting gam be
    // meaningfully less accurate than the trusted reference. This is the
    // reference-relative accuracy bar: the achievable floor is whatever the best
    // mature tool gets on this data, and gam must reach it.
    assert!(
        gam_truth_rmse <= ref_truth_rmse * 1.10,
        "gam is less accurate at recovering the truth than the VGAM baseline: \
         rmse(gam, truth)={gam_truth_rmse:.5} > 1.10 * rmse(vgam, truth)={ref_truth_rmse:.5}"
    );

    // STRUCTURE-RECOVERY GUARD — gam must beat the trivial no-signal predictor by
    // a clear margin, so the match-or-beat bar above cannot be passed by a
    // degenerate fit that collapses every class to the uniform 1/K surface (the
    // failure mode of an over-smoothing regression). The uniform predictor emits
    // 1/K for every class at every grid point; its RMSE against the truth is the
    // baseline a model that learned NOTHING would post. With the strong
    // corner-sweeping signal `rmse(uniform, truth) ≈ 0.36`, and a faithful
    // per-group fit recovers the surface to a small fraction of that, so it must
    // sit well below the 0.85x line.
    let uniform_rmse = {
        let uni = vec![1.0 / K as f64; K * n_grid];
        rmse(&uni, &flat_true)
    };
    assert!(
        gam_truth_rmse <= 0.85 * uniform_rmse,
        "gam did not recover meaningful group/class structure: rmse(gam, truth)={gam_truth_rmse:.5} \
         is not comfortably below the no-signal uniform-predictor baseline \
         rmse(uniform, truth)={uniform_rmse:.5} (0.85x = {:.5})",
        0.85 * uniform_rmse,
    );
}
