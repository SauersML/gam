//! End-to-end OBJECTIVE quality: gam's penalized multinomial-logit (softmax)
//! GAM with smooth terms must RECOVER the true class-probability surface from
//! which the categorical labels were drawn.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not "same as a reference tool"):
//! the data are sampled from a *known* softmax surface
//! `P_true(x) = softmax(true_eta(x1,x2,x3))` (cubic shape in x1, sigmoid in x2,
//! per-class linear slopes on x3, reference class η ≡ 0). After fitting
//! `y ~ s(x1) + s(x2) + x3` we evaluate gam's fitted probability matrix at the
//! training rows and assert
//!     RMSE(P_gam, P_true)  <=  PROB_RMSE_BAR
//! over the whole N×K simplex — i.e. gam's fitted probabilities are close to the
//! TRUTH in an absolute, tool-independent sense. We additionally assert the
//! simplex STRUCTURE directly (every row sums to 1, every entry in [0,1]).
//!
//! VGAM as a BASELINE TO MATCH-OR-BEAT (not as the pass criterion): we also fit
//! the identical model with `VGAM::vgam(..., family = multinomial())` on the
//! same data and same reference-class gauge, evaluate ITS error against the SAME
//! true surface, and assert
//!     RMSE(P_gam, P_true)  <=  RMSE(P_vgam, P_true) * 1.10
//! so gam recovers the truth at least as accurately (within 10%) as the mature
//! reference. The primary claim is truth recovery; VGAM is only a yardstick on
//! that objective accuracy. Closeness of P_gam to P_vgam is printed for context
//! via eprintln! but is NOT a pass criterion — matching another smoother's noisy
//! fit (gam REML λ vs VGAM fixed df=4 backfit) proves nothing about quality.
//!
//! Reference tool: `VGAM::vgam(..., family = multinomial())`, the canonical R
//! package for multinomial GAMs with smooth predictors. We pin VGAM's factor
//! levels to gam's `class_levels` order so both share the reference class (last
//! level, η = 0) and their probability columns line up with the truth columns.

use csv::StringRecord;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N: usize = 300;
const K: usize = 3;

/// Absolute probability-RMSE bar against the TRUE simplex. A consistent
/// penalized softmax GAM on N=300 draws recovers each probability to a few
/// percent; 0.06 sits well above that yet a real softmax/penalty/gauge bug
/// (wrong reference class, mis-assembled per-class penalty, broken smooth)
/// drives the error far past it.
const PROB_RMSE_BAR: f64 = 0.06;

/// One stable softmax surface: builds the K class log-odds (reference class 2
/// is pinned to η = 0). Identical math feeds the label draw AND the truth
/// matrix the fits are scored against.
fn true_eta(x1: f64, x2: f64, x3: f64) -> [f64; K] {
    let cubic = 2.0 * x1.powi(3) - 1.0 * x1; // s(x1): cubic shape
    let sigmoid = 3.0 / (1.0 + (-6.0 * (x2 - 0.5)).exp()) - 1.5; // s(x2): sigmoid shape
    // Active classes 0 and 1; reference class 2 has eta = 0.
    let eta0 = 0.6 + cubic + 0.5 * sigmoid + 1.5 * x3;
    let eta1 = -0.4 - 0.5 * cubic + sigmoid - 0.8 * x3;
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
fn gam_multinomial_softmax_recovers_true_simplex() {
    init_parallelism();

    // ---- synthesize the shared dataset (fixed seed) -----------------------
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform x1");
    let u01 = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x2");
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform x3");
    let udraw = Uniform::new(0.0_f64, 1.0_f64).expect("uniform draw");

    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut x3 = Vec::with_capacity(N);
    let mut cls_code = Vec::with_capacity(N); // realized class index 0..K
    // True per-row class probabilities keyed by code (0,1,2). We re-order these
    // into gam's reported class_levels order once the model is fit.
    let mut true_prob_by_code: Vec<[f64; K]> = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let p = softmax(&true_eta(a, b, c));
        let u = udraw.sample(&mut rng);
        // inverse-CDF class sample
        let mut acc = 0.0;
        let mut chosen = K - 1;
        for k in 0..K {
            acc += p[k];
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

    // Class labels gam will treat as categorical (non-numeric strings).
    let label = |code: usize| format!("c{code}");

    // ---- fit with gam: y ~ s(x1) + s(x2) + x3, multinomial driver ----------
    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                x3[i].to_string(),
                label(cls_code[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");

    let cfg = FitConfig::default();
    let model =
        fit_penalized_multinomial_formula(&ds, "y ~ s(x1) + s(x2) + x3", &cfg, 1.0, 50, 1e-8)
            .expect("gam multinomial fit");
    assert_eq!(
        model.class_levels.len(),
        K,
        "gam should recover K=3 classes"
    );

    // gam fitted probabilities at the training rows. Columns follow
    // `model.class_levels` (order of first appearance).
    let gam_probs = predict_multinomial_formula(&model, &ds).expect("gam predict probabilities");
    assert_eq!(gam_probs.dim(), (N, K), "gam probability matrix shape");

    // ---- STRUCTURE: simplex closure (rows sum to 1, entries in [0,1]) ------
    let mut worst_row_sum_err = 0.0_f64;
    let mut min_entry = f64::INFINITY;
    let mut max_entry = f64::NEG_INFINITY;
    for i in 0..N {
        let mut s = 0.0;
        for k in 0..K {
            let p = gam_probs[[i, k]];
            min_entry = min_entry.min(p);
            max_entry = max_entry.max(p);
            s += p;
        }
        worst_row_sum_err = worst_row_sum_err.max((s - 1.0).abs());
    }

    // Build the TRUTH matrix in gam's class_levels column order. gam levels are
    // strings "c{code}"; map each column k to its integer code.
    let gam_levels: Vec<String> = model.class_levels.clone();
    let col_code: Vec<usize> = gam_levels
        .iter()
        .map(|lvl| {
            lvl.trim_start_matches('c')
                .parse::<usize>()
                .expect("gam level label is c<code>")
        })
        .collect();

    // Flattened (column-major) probability vectors aligned across gam, truth.
    let mut gam_flat = Vec::with_capacity(N * K);
    let mut truth_flat = Vec::with_capacity(N * K);
    for k in 0..K {
        let code = col_code[k];
        for i in 0..N {
            gam_flat.push(gam_probs[[i, k]]);
            truth_flat.push(true_prob_by_code[i][code]);
        }
    }

    // ---- fit the SAME model with VGAM (baseline yardstick on accuracy) -----
    // Reconstruct the factor in R with levels in gam's order so VGAM's fitted
    // columns line up with the same truth columns. levorder tiles the K codes
    // cyclically to N rows (harness rejects ragged columns); R recovers the
    // first-seen order via unique().
    let level_codes: Vec<f64> = (0..N).map(|i| col_code[i % K] as f64).collect();
    let cls_f64: Vec<f64> = cls_code.iter().map(|&c| c as f64).collect();
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("x3", &x3),
            Column::new("cls", &cls_f64),
            Column::new("levorder", &level_codes),
        ],
        r#"
        suppressPackageStartupMessages(library(VGAM))
        lev_codes <- unique(round(df$levorder))
        lev_labels <- paste0("c", lev_codes)
        yfac <- factor(paste0("c", round(df$cls)), levels = lev_labels)
        dat <- data.frame(x1 = df$x1, x2 = df$x2, x3 = df$x3, y = yfac)
        m <- vgam(y ~ s(x1) + s(x2) + x3, family = multinomial(), data = dat)
        pr <- predict(m, type = "response")
        emit("nrow", nrow(pr))
        emit("ncol", ncol(pr))
        emit("probs", as.numeric(as.vector(pr)))
        "#,
    );

    let vg_nrow = r.scalar("nrow") as usize;
    let vg_ncol = r.scalar("ncol") as usize;
    assert_eq!(vg_nrow, N, "VGAM fitted-prob rows");
    assert_eq!(vg_ncol, K, "VGAM fitted-prob cols");
    let vg_flat = r.vector("probs"); // column-major, columns in gam's level order
    assert_eq!(vg_flat.len(), N * K, "VGAM flattened prob length");

    // ---- OBJECTIVE accuracy: error of each fit against the TRUE surface ----
    let gam_err = rmse(&gam_flat, &truth_flat);
    let vg_err = rmse(vg_flat, &truth_flat);

    // Context only (NOT a pass criterion): how close the two fits are to EACH
    // OTHER. Different smoothers (gam REML λ vs VGAM fixed df=4 backfit) land on
    // materially different surfaces; matching VGAM is not a quality claim.
    let frob_rel_gam_vs_vgam = relative_l2(&gam_flat, vg_flat);

    eprintln!(
        "multinomial s(x1)+s(x2)+x3: N={N} K={K} converged={} iters={} \
         gam_RMSE_vs_truth={gam_err:.5} vgam_RMSE_vs_truth={vg_err:.5} \
         row_sum_err={worst_row_sum_err:.2e} min_p={min_entry:.4} max_p={max_entry:.4} \
         frob_rel_gam_vs_vgam(context)={frob_rel_gam_vs_vgam:.4} lambdas={:?}",
        model.converged, model.iterations, model.lambdas
    );

    // ---- assertions: STRUCTURE then TRUTH RECOVERY then MATCH-OR-BEAT ------
    assert!(
        worst_row_sum_err < 1e-6,
        "fitted probabilities are not on the simplex: worst row-sum error={worst_row_sum_err:.2e}"
    );
    assert!(
        min_entry >= -1e-9 && max_entry <= 1.0 + 1e-9,
        "fitted probabilities escape [0,1]: min={min_entry:.4} max={max_entry:.4}"
    );
    assert!(
        gam_err <= PROB_RMSE_BAR,
        "gam does not recover the true class-probability surface: \
         RMSE(P_gam, P_true)={gam_err:.5} > bar={PROB_RMSE_BAR}"
    );
    assert!(
        gam_err <= vg_err * 1.10,
        "gam is less accurate than VGAM against the truth: \
         gam_RMSE={gam_err:.5} vgam_RMSE={vg_err:.5} (allowed gam <= 1.10*vgam)"
    );
}
