//! Objective quality of gam's penalized multinomial-logit (softmax) GAM, judged
//! by **held-out predictive accuracy and log-loss against a known generating
//! rule**, not by agreement with any reference tool's fitted output.
//!
//! The data is generated from a known categorical rule: each row's class is
//! SAMPLED from the softmax of the logits `[1.5·sin(x1), −0.8·cos(x1)·x2, 0]` over
//! the rectangle `[0, 2π] × [-3, 3]`, with a deterministic LCG so the labels are
//! byte-identical run to run. The labels used to be the hard `argmax` of those
//! logits. That separates the classes completely: the unpenalized MLE diverges,
//! and mgcv's `gam.fit5` stops with "non finite values in Hessian" (pool job
//! 646273). Sampled labels overlap across classes, the same fixture choice
//! `quality_vs_statsmodels_multinomial` makes (#1082).
//!
//! OBJECTIVE METRIC (the pass/fail claim): train gam on a deterministic 70%
//! slice, predict the held-out 30%, and score multiclass accuracy and multinomial
//! log-loss (a proper scoring rule, which also penalizes over- and
//! under-confident calls that accuracy cannot see). Because the generating field
//! `p*` is known, the best any classifier can do on these rows is computed
//! exactly:
//!   * the Bayes classifier's expected accuracy `A* = mean_i max_c p*_c(x_i)`,
//!     whose per-row sampling variance is `m_i (1 − m_i)` with `m_i = max_c p*_c`;
//!   * the true field's expected log-loss `H* = mean_i H(p*(x_i))`, whose per-row
//!     sampling variance is `Σ_c p*_c (ln p*_c)² − H(p*(x_i))²`.
//! gam must reach accuracy `A* − z·σ_A` and log-loss at most `H* + z·σ_L`, where
//! `σ` is the exact standard deviation of the realized mean over the held-out
//! rows and `z` the one-sided normal quantile at `GATE_FALSE_ALARM_RATE`. The true
//! field itself, scored on the same realized labels, must clear both bars: that
//! positive control shows the band describes this draw.
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
//! that simultaneously loads a cyclic 1-D smooth `s(x1, bs='cyclic')`, a thin-plate
//! 1-D smooth `s(x2, bs='tps')`, AND a tensor-product interaction
//! `te(x1, x2, bs=c('cyclic','tps'))` — three penalty blocks per active class,
//! replicated across `K-1 = 2` softmax linear predictors.

use gam::data::EncodedDataset;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::test_support::calibration::{COVERAGE_FALSE_POSITIVE_RATE, standard_normal_quantile};
use gam::test_support::reference::{Column, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

use csv::StringRecord;
use ndarray::Array2;
use std::f64::consts::PI;

/// The gate's error rate: the probability that a fit exactly as good as the
/// generating field fails the Bayes bars below by label sampling alone. It is the
/// same 1% budget the calibration gates share.
const GATE_FALSE_ALARM_RATE: f64 = COVERAGE_FALSE_POSITIVE_RATE;

/// Seed of the label draw.
const LABEL_SEED: u64 = 0x1082_0085;

/// The class labels, in the order of the generating logits.
const LABELS: [&str; 3] = ["A", "B", "C"];

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// One generated observation: covariates, the sampled class, and the generating
/// field's class probabilities there (in `LABELS` order).
struct Obs {
    x1: f64,
    x2: f64,
    label: String,
    true_probs: [f64; 3],
}

/// Synthetic categorical dataset with a known generating field.
///
/// `(x1, x2)` sweep the rectangle `[0, 2π] × [-3, 3]` on a deterministic
/// space-filling lattice; the class is sampled from the softmax of the logits
/// `[1.5·sin(x1), −0.8·cos(x1)·x2, 0]` with a seeded LCG, encoded as the string
/// labels `"A"/"B"/"C"`.
fn make_observations(n: usize) -> Vec<Obs> {
    // Two coprime irrational strides give a deterministic, well-spread
    // additive-recurrence (Weyl) sequence over the unit square — no RNG, no
    // duplicate rows, and good coverage of every corner of the rectangle.
    let stride1 = (2.0_f64).sqrt().fract(); // ≈ 0.41421356
    let stride2 = (3.0_f64).sqrt().fract(); // ≈ 0.73205081
    let mut u1 = 0.12_f64;
    let mut u2 = 0.37_f64;
    let mut draw = Lcg(LABEL_SEED);

    let mut obs = Vec::with_capacity(n);
    for _ in 0..n {
        u1 = (u1 + stride1).fract();
        u2 = (u2 + stride2).fract();
        // Map the unit square onto [0, 2π] × [-3, 3].
        let a = 2.0 * PI * u1;
        let b = -3.0 + 6.0 * u2;

        // Softmax logits with the reference class (index 2) pinned at 0.
        let logits = [1.5 * a.sin(), -0.8 * a.cos() * b, 0.0];
        let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = logits.map(|logit| (logit - shift).exp());
        let total: f64 = weights.iter().sum();
        let true_probs = weights.map(|weight| weight / total);
        // Inverse-CDF draw of the class from the generating softmax.
        let u = draw.unit();
        let class = if u < true_probs[0] {
            0
        } else if u < true_probs[0] + true_probs[1] {
            1
        } else {
            2
        };

        obs.push(Obs {
            x1: a,
            x2: b,
            label: LABELS[class].to_string(),
            true_probs,
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
    let formula = "y ~ s(x1, bs='cyclic', k=8) + s(x2, bs='tps', k=5) + te(x1, x2, bs=c('cyclic','tps'))";
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

    // ---- the generating field's exact Bayes scores on the held-out rows -------
    // `A*` and `H*` are the expectations, over the label draw, of the Bayes
    // classifier's accuracy and the true field's log-loss on these covariates; the
    // variances are each row's own sampling variance, summed exactly. The field is
    // also scored on the realized labels, as the positive control.
    let n_test = test.len() as f64;
    let mut bayes_accuracy_sum = 0.0_f64;
    let mut bayes_accuracy_variance = 0.0_f64;
    let mut bayes_log_loss_sum = 0.0_f64;
    let mut bayes_log_loss_variance = 0.0_f64;
    let mut field_correct = 0usize;
    let mut field_log_loss_sum = 0.0_f64;
    for o in &test {
        let p = o.true_probs;
        let mut best = 0usize;
        for class in 1..LABELS.len() {
            if p[class] > p[best] {
                best = class;
            }
        }
        let largest = p[best];
        bayes_accuracy_sum += largest;
        bayes_accuracy_variance += largest * (1.0 - largest);
        let entropy: f64 = p.iter().map(|&value| -value * value.ln()).sum();
        let second_moment: f64 = p.iter().map(|&value| value * value.ln().powi(2)).sum();
        bayes_log_loss_sum += entropy;
        bayes_log_loss_variance += second_moment - entropy * entropy;
        let realized = LABELS
            .iter()
            .position(|label| *label == o.label)
            .expect("every label is drawn from LABELS");
        if best == realized {
            field_correct += 1;
        }
        field_log_loss_sum -= p[realized].ln();
    }
    let bayes_accuracy = bayes_accuracy_sum / n_test;
    let bayes_log_loss = bayes_log_loss_sum / n_test;
    let z = standard_normal_quantile(1.0 - GATE_FALSE_ALARM_RATE);
    let accuracy_bar = bayes_accuracy - z * bayes_accuracy_variance.sqrt() / n_test;
    let log_loss_bar = bayes_log_loss + z * bayes_log_loss_variance.sqrt() / n_test;
    let field_accuracy = field_correct as f64 / n_test;
    let field_log_loss = field_log_loss_sum / n_test;

    // ---- structural identity on TRAIN rows (internal consistency only) ------
    // The stored unpenalized deviance must equal an independent softmax
    // recompute `-2·Σ log p̂` over the training rows — no penalty leakage, no
    // permuted/dropped reference class. This is a bookkeeping invariant, not a
    // peer-tool comparison.
    //
    // `p̂` is the MODE's own `softmax(η̂)`, the probability `−2 · log L(β̂)` is
    // defined against, rebuilt from the payload's training design and active
    // coefficients with the reference class's `η = 0` last.
    // `predict_multinomial_formula` publishes the posterior mean `E[softmax(η)]`
    // (gam#2612), a different estimand, so its log-likelihood is not the stored
    // deviance's to within any roundoff bar.
    let probs_train = {
        let design = model.training_design().expect("training design");
        let beta = model.coefficients_active().expect("active coefficients");
        let eta = design.dot(&beta);
        let active = eta.ncols();
        let mut probs = Array2::<f64>::zeros((eta.nrows(), active + 1));
        for (row, mut out) in eta.rows().into_iter().zip(probs.rows_mut()) {
            let shift = row.iter().copied().fold(0.0_f64, f64::max);
            let partition =
                (-shift).exp() + row.iter().map(|&value| (value - shift).exp()).sum::<f64>();
            for (class, &value) in row.iter().enumerate() {
                out[class] = (value - shift).exp() / partition;
            }
            out[active] = (-shift).exp() / partition;
        }
        probs
    };
    assert_eq!(probs_train.nrows(), train.len(), "saved training design rows");
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

    // gam's own held-out quality and bookkeeping are printed and judged BEFORE
    // anything that needs another tool. A comparator that is missing, or cannot
    // fit this fixture at all, costs the COMPARISON, not the tool-independent
    // metrics: on these completely separated hard-argmax labels mgcv's
    // `gam.fit5` stops with "non finite values in Hessian" (pool job 646273),
    // and that used to take gam's numbers down with it (#1082).
    eprintln!(
        "[multinomial-quality] n_train={} n_test={} K={}\n  \
         gam:   acc={gam_acc:.4} logloss={gam_ll:.4}\n  \
         field: acc={field_accuracy:.4} logloss={field_log_loss:.4} (positive control)\n  \
         Bayes: A*={bayes_accuracy:.4} H*={bayes_log_loss:.4} z={z:.4} \
         bars: acc>={accuracy_bar:.4} logloss<={log_loss_bar:.4}\n  \
         stored-deviance identity: abs={dev_abs:.3e} rel={dev_rel:.3e}",
        train.len(),
        test.len(),
        model.class_levels.len(),
    );
    // The selected smoothing parameters and each penalty's EDF, block-major. Every λ
    // on the upper rail with EDF near zero is an intercept-only collapse, which a
    // held-out accuracy near the class shares cannot tell apart from a weak fit.
    let scientific = |values: &[f64]| -> String {
        values
            .iter()
            .map(|value| format!("{value:.3e}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!(
        "[multinomial-quality] lambda [{}] edf_per_penalty [{}] jeffreys_armed={}",
        scientific(&model.lambdas),
        model
            .edf_per_penalty
            .as_deref()
            .map_or_else(|| "not reported".to_string(), scientific),
        model.separation_evidence.is_some(),
    );

    // ── OBJECTIVE PASS/FAIL ────────────────────────────────────────────────
    // 0. Positive control: the generating field itself, on the same realized
    //    labels, clears both bars. A field that fails them means the band does not
    //    describe this draw, so gam's result against it would say nothing.
    assert!(
        field_accuracy >= accuracy_bar,
        "positive control: the generating field's held-out accuracy {field_accuracy:.4} is \
         below its own Bayes bar {accuracy_bar:.4} (A* = {bayes_accuracy:.4})"
    );
    assert!(
        field_log_loss <= log_loss_bar,
        "positive control: the generating field's held-out log-loss {field_log_loss:.4} is \
         above its own Bayes bar {log_loss_bar:.4} (H* = {bayes_log_loss:.4})"
    );

    // 1. gam against the exact Bayes bars on the held-out split.
    assert!(
        gam_acc >= accuracy_bar,
        "held-out accuracy {gam_acc:.4} below the Bayes bar {accuracy_bar:.4} \
         (A* = {bayes_accuracy:.4}, one-sided false-alarm rate {GATE_FALSE_ALARM_RATE})"
    );
    assert!(
        gam_ll <= log_loss_bar,
        "held-out multinomial log-loss {gam_ll:.4} nats/row above the Bayes bar \
         {log_loss_bar:.4} (H* = {bayes_log_loss:.4}, one-sided false-alarm rate \
         {GATE_FALSE_ALARM_RATE})"
    );

    // 2. Structural bookkeeping invariant (internal consistency).
    assert!(
        dev_abs < 1e-8 && dev_rel < 1e-10,
        "stored deviance disagrees with independent softmax recompute: \
         abs={dev_abs:.3e} rel={dev_rel:.3e} (penalty leak / permuted reference class?)"
    );

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
te_rows <- df[df$is_train < 0.5, ]
tr$yc <- as.integer(round(tr$yc))
# multinom(K) models classes 0..K against reference 0; gam uses K-1=2 active.
#
# INTERACTION SPELLING: in mgcv a `te(x1, x2)` tensor product CONTAINS the x1
# and x2 main effects, so `s(x1) + s(x2) + te(x1, x2)` duplicates that
# main-effect space and is rank-deficient by construction (see `?ti`, which
# documents `ti` as the term to use precisely when the main effects are also
# in the model). The duplicated space is what made the outer
# smoothing-parameter iteration go NaN: first as
# `Error in if (sum(uconv.ind) == 0) ... : missing value where TRUE/FALSE
# needed` under the default "newton" optimizer, and then -- after switching to
# "bfgs", which only moved where the NaN surfaced -- as
# `Error in qr.default(...) : NA/NaN/Inf in foreign function call (arg 1)`,
# raised from `kappa()`, i.e. mgcv taking the condition number of a matrix
# that had already gone non-finite. `ti()` is mgcv's identifiable spelling of
# the SAME function space (marginal main effects plus the pure interaction),
# so the baseline is not weakened, merely written the way mgcv requires. The
# penalty DECOMPOSITION differs, so the resulting log-loss is a NEW baseline
# number, not a resumption of the old one.
# With an identifiable model the stock outer optimizer is fine, so the "bfgs"
# workaround is dropped; `method = "REML"` is stated explicitly to match the
# criterion gam minimises (general families use REML regardless).
#
# NOTE: the held-out frame is `te_rows`, not `te` -- a data frame named `te`
# SHADOWS mgcv's `te()` term constructor in the formula environment.
fit <- gam(
  list(
    yc ~ s(x1, bs = "cc", k = 8) + s(x2, bs = "tp", k = 5) + ti(x1, x2, bs = c("cc","tp")),
        ~ s(x1, bs = "cc", k = 8) + s(x2, bs = "tp", k = 5) + ti(x1, x2, bs = c("cc","tp"))
  ),
  family = multinom(K = 2), data = tr, method = "REML"
)
# predict type="response" gives P(class=1..K) per row as a (n x K) matrix.
pr <- predict(fit, newdata = te_rows, type = "response")
pr <- as.matrix(pr)
if (ncol(pr) == 2) {           # some mgcv builds return only the K active cols
  pr <- cbind(1 - rowSums(pr), pr)
}
# Clamp + renormalize for a clean simplex.
pr[pr < 1e-12] <- 1e-12
pr <- pr / rowSums(pr)
ytrue <- as.integer(round(te_rows$yc))       # 0-based class codes
pred  <- max.col(pr) - 1L                     # 0-based argmax
acc <- mean(pred == ytrue)
ll  <- -mean(log(pr[cbind(seq_len(nrow(pr)), ytrue + 1L)]))
emit("mgcv_acc", acc)
emit("mgcv_logloss", ll)
"#;
    let reference = run_r(&columns, r_body);
    let mgcv_acc = reference.scalar("mgcv_acc");
    let mgcv_ll = reference.scalar("mgcv_logloss");
    eprintln!(
        "[multinomial-quality] n_train={n_train} mgcv: acc={mgcv_acc:.4} logloss={mgcv_ll:.4}"
    );

    // 3. Match-or-beat the mature baseline on the SAME objective metric.
    assert!(
        gam_acc >= mgcv_acc - 0.02,
        "gam held-out accuracy {gam_acc:.4} trails mgcv {mgcv_acc:.4} by more than 0.02"
    );
    assert!(
        gam_ll <= mgcv_ll * 1.10,
        "gam held-out log-loss {gam_ll:.4} exceeds mgcv {mgcv_ll:.4} × 1.10"
    );
}
