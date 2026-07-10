//! Objective quality of gam's penalized multinomial-logit (softmax) GAM, judged
//! by **held-out predictive accuracy against a known generating rule**, not by
//! agreement with any reference tool's fitted output.
//!
//! The data is generated from a fully deterministic categorical rule: each row's
//! class is the `argmax` of the softmax logits `[1.5·sin(x1), −0.8·cos(x1)·x2, 0]`
//! over the rectangle `[0, 2π] × [-3, 3]`. Because the labels are the hard argmax
//! of a smooth logit field, the Bayes-optimal classifier here has *zero* error —
//! the only thing standing between a fitted model and perfect held-out accuracy
//! is whether its smooths recover the true decision boundary. That makes
//! held-out accuracy a genuine, tool-independent quality metric: a model that
//! recovers the truth scores ~1.0; a model that under/over-smooths the boundary
//! loses accuracy.
//!
//! OBJECTIVE METRIC (the pass/fail claim):
//!   * Train gam on a deterministic 70% slice, predict the held-out 30%, and
//!     assert **held-out classification accuracy ≥ 0.90** (truth recovery: the
//!     boundary is learnable to near-perfection, so 0.90 is a principled,
//!     un-weakened bar that still fails a mis-fit boundary).
//!   * Assert the held-out **multinomial log-loss ≤ 0.45 nats/row** — a proper
//!     scoring rule that additionally penalizes over-confident wrong calls and
//!     under-confident right ones (a hard-accuracy-only model can pass accuracy
//!     while being badly calibrated; log-loss closes that gap).
//!
//! BASELINE TO MATCH-OR-BEAT (the reference is demoted, never the pass gate):
//!   mgcv's `multinom(K=2)` GAM is fit on the *identical* train rows and scored
//!   on the *identical* test rows. We additionally require gam to
//!   **match-or-beat** it: gam accuracy ≥ mgcv accuracy − 0.02 AND gam log-loss
//!   ≤ mgcv log-loss × 1.10. mgcv is a baseline on the objective metric, not the
//!   definition of correctness — if mgcv itself mis-fit, gam still has to clear
//!   the absolute bars above.
//!
//! STRUCTURAL SANITY (cheap correctness invariant, not a reference comparison):
//!   gam's predicted probability rows lie on the simplex (each row sums to 1, all
//!   entries in [0,1]); the converged stored unpenalized deviance equals an
//!   independent softmax recompute `-2·Σ log p̂` on the training rows. These are
//!   internal-consistency checks (no peer tool involved), retained because a
//!   broken simplex or a leaking-penalty deviance would silently corrupt every
//!   downstream AIC/LRT consumer.
//!
//! Combination under test (bugs hide in combinations): a single multinomial fit
//! that simultaneously loads a cyclic 1-D smooth `s(x1, bs='cc')`, a thin-plate
//! 1-D smooth `s(x2, bs='tp')`, AND a tensor-product interaction
//! `te(x1, x2, bs=c('cc','tp'))` — three penalty blocks per active class,
//! replicated across `K-1 = 2` softmax linear predictors.

use gam::data::EncodedDataset;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::test_support::reference::{Column, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

use csv::StringRecord;
use ndarray::Array2;
use std::f64::consts::PI;

/// One generated observation: covariates plus the deterministic hard class.
struct Obs {
    x1: f64,
    x2: f64,
    label: String,
}

/// Synthetic, fully deterministic (RNG-free) categorical dataset.
///
/// `(x1, x2)` sweep the rectangle `[0, 2π] × [-3, 3]` on a deterministic
/// space-filling lattice; the class is the `argmax` of the softmax logits
/// `[1.5·sin(x1), −0.8·cos(x1)·x2, 0]`, encoded as the string labels
/// `"A"/"B"/"C"`.
fn make_observations(n: usize) -> Vec<Obs> {
    // Two coprime irrational strides give a deterministic, well-spread
    // additive-recurrence (Weyl) sequence over the unit square — no RNG, no
    // duplicate rows, and good coverage of every corner of the rectangle.
    let stride1 = (2.0_f64).sqrt().fract(); // ≈ 0.41421356
    let stride2 = (3.0_f64).sqrt().fract(); // ≈ 0.73205081
    let mut u1 = 0.12_f64;
    let mut u2 = 0.37_f64;

    let mut obs = Vec::with_capacity(n);
    for _ in 0..n {
        u1 = (u1 + stride1).fract();
        u2 = (u2 + stride2).fract();
        // Map the unit square onto [0, 2π] × [-3, 3].
        let a = 2.0 * PI * u1;
        let b = -3.0 + 6.0 * u2;

        // Softmax logits with the reference class (index 2) pinned at 0.
        let l0 = 1.5 * a.sin();
        let l1 = -0.8 * a.cos() * b;
        let l2 = 0.0;
        // Deterministic hard class = argmax of the logits.
        let label = if l0 >= l1 && l0 >= l2 {
            "A"
        } else if l1 >= l0 && l1 >= l2 {
            "B"
        } else {
            "C"
        };

        obs.push(Obs {
            x1: a,
            x2: b,
            label: label.to_string(),
        });
    }
    obs
}

/// Encode a slice of observations into gam's `EncodedDataset` (categorical `y`).
fn encode(obs: &[Obs]) -> EncodedDataset {
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = obs
        .iter()
        .map(|o| {
            StringRecord::from(vec![
                format!("{:.17e}", o.x1),
                format!("{:.17e}", o.x2),
                o.label.clone(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset")
}

/// Multinomial log-loss (cross-entropy, nats/row) of predicted simplex rows
/// `probs` against the realized class indices, with a tiny clamp so a single
/// zero probability cannot send the score to +∞ and mask everything else.
fn log_loss(probs: &Array2<f64>, class_idx: &[usize]) -> f64 {
    let n = class_idx.len();
    assert_eq!(probs.nrows(), n, "log_loss row mismatch");
    let mut total = 0.0_f64;
    for (i, &c) in class_idx.iter().enumerate() {
        let p = probs[[i, c]].clamp(1e-12, 1.0);
        total -= p.ln();
    }
    total / n.max(1) as f64
}

/// Top-1 classification accuracy of `probs` against realized class indices.
fn accuracy(probs: &Array2<f64>, class_idx: &[usize]) -> f64 {
    let n = class_idx.len();
    assert_eq!(probs.nrows(), n, "accuracy row mismatch");
    let mut correct = 0usize;
    for (i, &c) in class_idx.iter().enumerate() {
        let mut best = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for k in 0..probs.ncols() {
            if probs[[i, k]] > best_p {
                best_p = probs[[i, k]];
                best = k;
            }
        }
        if best == c {
            correct += 1;
        }
    }
    correct as f64 / n.max(1) as f64
}

#[test]
fn multinomial_recovers_decision_boundary_on_held_out_split() {
    init_parallelism();

    let n = 400;
    let obs = make_observations(n);

    // Deterministic 70/30 split by index parity-free stride: every 10th row in a
    // rotating window lands in test, giving a reproducible, well-mixed 30% hold
    // out that still covers all three classes and the full covariate ranges.
    let mut train: Vec<Obs> = Vec::new();
    let mut test: Vec<Obs> = Vec::new();
    for (i, o) in obs.into_iter().enumerate() {
        if i % 10 < 3 {
            test.push(o);
        } else {
            train.push(o);
        }
    }
    assert!(!train.is_empty() && !test.is_empty(), "non-empty split");

    let ds_train = encode(&train);
    let ds_test = encode(&test);

    // ---- fit gam's multinomial-logit GAM with the loaded combination -------
    // Cyclic 1-D smooth on x1 (the angular covariate), thin-plate 1-D smooth on
    // x2, and a tensor-product interaction across both. Three penalty blocks
    // per active class, replicated over K-1 = 2 softmax predictors.
    let formula = "y ~ s(x1, bs='cc', k=8) + s(x2, bs='tp', k=5) + te(x1, x2, bs=c('cc','tp'))";
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &ds_train,
        formula,
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 50,
        tol: 1e-7,
    })
    .expect("multinomial formula fit");

    assert_eq!(
        model.class_levels.len(),
        3,
        "expected K=3 classes, got levels {:?}",
        model.class_levels
    );
    assert_eq!(
        model.n_active_classes, 2,
        "K-1 active classes expected for K=3"
    );

    // Map a label to gam's own class-column index (reference class = last level
    // in first-appearance order) so every score indexes probabilities in gam's
    // gauge regardless of which split a label first appears in.
    let class_index = |label: &str| -> usize {
        model
            .class_levels
            .iter()
            .position(|lvl| lvl == label)
            .unwrap_or_else(|| {
                panic!(
                    "label {label:?} not among class levels {:?}",
                    model.class_levels
                )
            })
    };

    // ---- gam held-out predictions -----------------------------------------
    let probs_test =
        predict_multinomial_formula(&model, &ds_test).expect("multinomial predict (test)");
    assert_eq!(
        probs_test.dim(),
        (test.len(), model.class_levels.len()),
        "test probs shape"
    );

    // STRUCTURAL SANITY: predicted rows must lie on the probability simplex.
    for i in 0..probs_test.nrows() {
        let mut row_sum = 0.0_f64;
        for k in 0..probs_test.ncols() {
            let p = probs_test[[i, k]];
            assert!(
                p.is_finite() && (-1e-12..=1.0 + 1e-9).contains(&p),
                "row {i} class {k}: probability {p} off the simplex"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {i}: predicted probabilities sum to {row_sum}, not 1"
        );
    }

    let test_idx: Vec<usize> = test.iter().map(|o| class_index(&o.label)).collect();
    let gam_acc = accuracy(&probs_test, &test_idx);
    let gam_ll = log_loss(&probs_test, &test_idx);

    // ---- mature baseline: mgcv multinom on the SAME train/test rows ---------
    // Score mgcv on the identical held-out rows. mgcv returns, for each row, the
    // probability of class index 1..K-1 (reference = first level); we rebuild
    // the full simplex and read off the realized class in mgcv's gauge.
    let train_x1: Vec<f64> = train.iter().map(|o| o.x1).collect();
    let train_x2: Vec<f64> = train.iter().map(|o| o.x2).collect();
    // Numeric class code in *first-appearance* order matching mgcv's factor
    // levels: build a stable level list from the training labels.
    let mut levels: Vec<String> = Vec::new();
    for o in &train {
        if !levels.contains(&o.label) {
            levels.push(o.label.clone());
        }
    }
    let code = |label: &str| -> f64 {
        levels
            .iter()
            .position(|lvl| lvl == label)
            .expect("test label present among training levels") as f64
    };
    let train_y: Vec<f64> = train.iter().map(|o| code(&o.label)).collect();
    let test_x1: Vec<f64> = test.iter().map(|o| o.x1).collect();
    let test_x2: Vec<f64> = test.iter().map(|o| o.x2).collect();
    let test_y_code: Vec<f64> = test.iter().map(|o| code(&o.label)).collect();

    // mgcv's multinom needs one formula per active class; we mirror gam's term
    // combination on each. We split data into train/test by row position: the
    // first `n_train` rows of the emitted columns are training, the rest test.
    let n_train = train.len();
    let mut col_x1 = train_x1.clone();
    col_x1.extend_from_slice(&test_x1);
    let mut col_x2 = train_x2.clone();
    col_x2.extend_from_slice(&test_x2);
    let mut col_y = train_y.clone();
    col_y.extend_from_slice(&test_y_code);
    let col_train = {
        let mut v = vec![1.0_f64; n_train];
        v.extend(std::iter::repeat_n(0.0_f64, test.len()));
        v
    };

    let columns = [
        Column::new("x1", &col_x1),
        Column::new("x2", &col_x2),
        Column::new("yc", &col_y),
        Column::new("is_train", &col_train),
    ];
    let r_body = r#"
suppressMessages(library(mgcv))
tr <- df[df$is_train > 0.5, ]
te <- df[df$is_train < 0.5, ]
tr$yc <- as.integer(round(tr$yc))
# multinom(K) models classes 0..K against reference 0; gam uses K-1=2 active.
fit <- gam(
  list(
    yc ~ s(x1, bs = "cc", k = 8) + s(x2, bs = "tp", k = 5) + te(x1, x2, bs = c("cc","tp")),
        ~ s(x1, bs = "cc", k = 8) + s(x2, bs = "tp", k = 5) + te(x1, x2, bs = c("cc","tp"))
  ),
  family = multinom(K = 2), data = tr
)
# predict type="response" gives P(class=1..K) per row as a (n x K) matrix.
pr <- predict(fit, newdata = te, type = "response")
pr <- as.matrix(pr)
if (ncol(pr) == 2) {           # some mgcv builds return only the K active cols
  pr <- cbind(1 - rowSums(pr), pr)
}
# Clamp + renormalize for a clean simplex.
pr[pr < 1e-12] <- 1e-12
pr <- pr / rowSums(pr)
ytrue <- as.integer(round(te$yc))            # 0-based class codes
pred  <- max.col(pr) - 1L                     # 0-based argmax
acc <- mean(pred == ytrue)
ll  <- -mean(log(pr[cbind(seq_len(nrow(pr)), ytrue + 1L)]))
emit("mgcv_acc", acc)
emit("mgcv_logloss", ll)
"#;
    let reference = run_r(&columns, r_body);
    let mgcv_acc = reference.scalar("mgcv_acc");
    let mgcv_ll = reference.scalar("mgcv_logloss");

    // ---- structural identity on TRAIN rows (internal consistency only) ------
    // The stored unpenalized deviance must equal an independent softmax
    // recompute `-2·Σ log p̂` over the training rows — no penalty leakage, no
    // permuted/dropped reference class. This is a bookkeeping invariant, not a
    // peer-tool comparison.
    let probs_train =
        predict_multinomial_formula(&model, &ds_train).expect("multinomial predict (train)");
    let mut loglik_train = 0.0_f64;
    for (i, o) in train.iter().enumerate() {
        let c = class_index(&o.label);
        let p = probs_train[[i, c]];
        assert!(
            p.is_finite() && p > 0.0,
            "train row {i}: realized-class probability {p} non-positive/non-finite"
        );
        loglik_train += p.ln();
    }
    let deviance_recompute = -2.0 * loglik_train;
    let dev_abs = (model.deviance - deviance_recompute).abs();
    let dev_rel = dev_abs / deviance_recompute.abs().max(1.0);

    eprintln!(
        "[multinomial-quality] n_train={} n_test={} K={}\n  \
         gam:  acc={gam_acc:.4} logloss={gam_ll:.4}\n  \
         mgcv: acc={mgcv_acc:.4} logloss={mgcv_ll:.4}\n  \
         stored-deviance identity: abs={dev_abs:.3e} rel={dev_rel:.3e}",
        n_train,
        test.len(),
        model.class_levels.len(),
    );

    // ── OBJECTIVE PASS/FAIL ────────────────────────────────────────────────
    // 1. Absolute truth-recovery bars on the held-out split.
    assert!(
        gam_acc >= 0.90,
        "held-out accuracy {gam_acc:.4} below the 0.90 truth-recovery bar \
         (the Bayes-optimal boundary is learnable to ~1.0)"
    );
    assert!(
        gam_ll <= 0.45,
        "held-out multinomial log-loss {gam_ll:.4} nats/row above the 0.45 bar \
         (model is mis-calibrated even if argmax accuracy is acceptable)"
    );

    // 2. Match-or-beat the mature baseline on the SAME objective metric.
    assert!(
        gam_acc >= mgcv_acc - 0.02,
        "gam held-out accuracy {gam_acc:.4} trails mgcv {mgcv_acc:.4} by more than 0.02"
    );
    assert!(
        gam_ll <= mgcv_ll * 1.10,
        "gam held-out log-loss {gam_ll:.4} exceeds mgcv {mgcv_ll:.4} × 1.10"
    );

    // 3. Structural bookkeeping invariant (internal consistency).
    assert!(
        dev_abs < 1e-8 && dev_rel < 1e-10,
        "stored deviance disagrees with independent softmax recompute: \
         abs={dev_abs:.3e} rel={dev_rel:.3e} (penalty leak / permuted reference class?)"
    );
}
