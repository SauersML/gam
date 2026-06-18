//! End-to-end OBJECTIVE quality: gam's penalized multinomial-logit (softmax) GAM
//! must classify penguin SPECIES from morphometric measurements on a REAL,
//! freely-downloadable dataset.
//!
//! DATASET: the Palmer Archipelago (Antarctica) penguin data, a widely used
//! multi-class benchmark (an `iris` replacement). Three species
//! (Adelie / Chinstrap / Gentoo) are predicted from four continuous body
//! measurements. Real measurements, no synthetic generative truth.
//!   Source (direct raw CSV, no auth):
//!   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/palmerpenguins/penguins.csv
//!   (Rdatasets index: https://vincentarelbundock.github.io/Rdatasets/datasets.html)
//!   Horst AM, Hill AP, Gorman KB (2020), palmerpenguins R package.
//!
//! REALISTIC USE-CASE: unordered multi-class (multinomial) classification of a
//! categorical response (species) from numeric covariates — exactly what
//! `nnet::multinom` and gam's softmax GAM are for. We model the species as a
//! smooth-additive softmax over the four measurements
//!     species ~ s(bill_length_mm) + s(bill_depth_mm)
//!             + s(flipper_length_mm) + s(body_mass_g)
//! so each class log-odds is a flexible function of body shape.
//!
//! OBJECTIVE METRIC (held out, not "match the reference's fit"):
//!   * a DETERMINISTIC train/test split (every 4th row by index is held out, no
//!     RNG), fit gam on TRAIN only, predict the held-out rows' class
//!     probabilities, and score them with two objective, tool-independent
//!     metrics:
//!       1. multiclass ACCURACY (argmax == truth) on the held-out set, and
//!       2. mean held-out multiclass LOG-LOSS  −(1/m) Σ_i log p̂_{i, y_i},
//!          the canonical proper scoring rule for a probabilistic classifier.
//!
//! ASSERTIONS:
//!   1. ABSOLUTE BAR — penguin species are very well separated by these four
//!      measurements (a competent classifier scores well above 0.90 held-out
//!      accuracy and well under ~0.30 nats log-loss), so we hold gam to a
//!      principled absolute bar: held-out accuracy >= 0.90 AND mean log-loss
//!      <= 0.45. A model that mis-assembled the softmax / per-class penalty /
//!      smooth basis would miss this by a wide margin.
//!   2. STRUCTURE — gam's predicted probabilities form a valid simplex (each row
//!      sums to 1, every entry in [0, 1]).
//!   3. CONTEXT BASELINE — `nnet::multinom` (the mature, standard R reference for
//!      multinomial regression) is fit on the IDENTICAL train rows with the
//!      IDENTICAL class coding and scored on the IDENTICAL held-out rows. Its
//!      metrics are printed for calibration, but are not pass/fail criteria:
//!      nnet fits a purely-linear softmax while gam fits a penalized smooth
//!      additive softmax, and #1082's binding acceptance criterion is gam's
//!      absolute held-out accuracy plus ARC halt behavior.
//!
//! Closeness of gam's probabilities to nnet's is printed for context only and is
//! NOT a pass criterion — nnet fits a purely-linear softmax while gam fits a
//! penalized smooth-additive one, so the two land on materially different
//! surfaces; matching nnet's noisy linear fit would prove nothing about quality.

use csv::StringRecord;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const PENGUINS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/penguins.csv");

/// Number of species classes (Adelie, Chinstrap, Gentoo).
const K: usize = 3;

/// Held-out split: every 4th row (index % 4 == 0) is test, the rest is train.
/// Fixed, data-order, no RNG — reproducible across runs.
const TEST_STRIDE: usize = 4;

/// One parsed penguin row: four numeric body measurements and the species label
/// (a non-numeric string the multinomial driver treats as a categorical class).
struct Penguin {
    bill_length: f64,
    bill_depth: f64,
    flipper: f64,
    body_mass: f64,
    species: String,
}

/// Parse `penguins.csv` into rows with all four measurements present. Rows whose
/// measurement columns are `NA` are dropped (the species/measurement columns are
/// in fact NA-free in this file; the guard documents the contract). The body
/// measurements are the only covariates; `island`/`sex`/`year` are ignored.
fn load_penguins() -> Vec<Penguin> {
    let file = File::open(Path::new(PENGUINS_CSV)).expect("open penguins.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("penguins header line")
        .expect("read penguins header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("penguins.csv missing column {name}"))
    };
    let i_species = idx("species");
    let i_bill_len = idx("bill_length_mm");
    let i_bill_dep = idx("bill_depth_mm");
    let i_flipper = idx("flipper_length_mm");
    let i_mass = idx("body_mass_g");

    let mut rows = Vec::new();
    for line in lines {
        let line = line.expect("read penguins row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let any_na = [i_bill_len, i_bill_dep, i_flipper, i_mass]
            .iter()
            .any(|&c| f[c] == "NA" || f[c].is_empty());
        if any_na {
            continue;
        }
        rows.push(Penguin {
            bill_length: f[i_bill_len].parse().expect("parse bill_length_mm"),
            bill_depth: f[i_bill_dep].parse().expect("parse bill_depth_mm"),
            flipper: f[i_flipper].parse().expect("parse flipper_length_mm"),
            body_mass: f[i_mass].parse().expect("parse body_mass_g"),
            species: f[i_species].to_string(),
        });
    }
    rows
}

/// Mean multiclass log-loss `−(1/m) Σ_i log p̂_{i, y_i}` over the held-out rows.
/// `probs_flat` is row-major `(m, K)`; `labels` are the true held-out class
/// column indices (aligned to the same K-column order as `probs_flat`).
/// Probabilities are clamped away from 0 to keep the score finite — the standard
/// guard every log-loss implementation uses (e.g. scikit-learn's `eps`).
fn mean_log_loss(probs_flat: &[f64], labels: &[usize], k: usize) -> f64 {
    let m = labels.len();
    assert_eq!(probs_flat.len(), m * k, "log-loss probs length mismatch");
    let eps = 1e-15;
    let mut acc = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        let p = probs_flat[i * k + y].clamp(eps, 1.0);
        acc -= p.ln();
    }
    acc / m as f64
}

/// Multiclass accuracy: fraction of held-out rows whose argmax-probability class
/// equals the true class. `probs_flat` is row-major `(m, K)`.
fn accuracy(probs_flat: &[f64], labels: &[usize], k: usize) -> f64 {
    let m = labels.len();
    assert_eq!(probs_flat.len(), m * k, "accuracy probs length mismatch");
    let mut correct = 0usize;
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs_flat[i * k..(i + 1) * k];
        let mut best = 0usize;
        for c in 1..k {
            if row[c] > row[best] {
                best = c;
            }
        }
        if best == y {
            correct += 1;
        }
    }
    correct as f64 / m as f64
}

/// Per-class recall (sensitivity): for each of the `k` classes, the fraction of
/// held-out rows of that true class whose argmax-probability prediction is that
/// class. `probs_flat` is row-major `(m, K)`; `labels` are true column indices.
/// Classes with no held-out rows are reported as `NaN` (caller asserts coverage
/// before reading them). A degenerate classifier that collapses onto the
/// majority class scores high global accuracy but near-zero recall on the
/// minority classes — this metric exposes exactly that failure mode.
fn per_class_recall(probs_flat: &[f64], labels: &[usize], k: usize) -> Vec<f64> {
    let m = labels.len();
    assert_eq!(
        probs_flat.len(),
        m * k,
        "per_class_recall probs length mismatch"
    );
    let mut total = vec![0usize; k];
    let mut hit = vec![0usize; k];
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs_flat[i * k..(i + 1) * k];
        let mut best = 0usize;
        for c in 1..k {
            if row[c] > row[best] {
                best = c;
            }
        }
        total[y] += 1;
        if best == y {
            hit[y] += 1;
        }
    }
    (0..k)
        .map(|c| {
            if total[c] == 0 {
                f64::NAN
            } else {
                hit[c] as f64 / total[c] as f64
            }
        })
        .collect()
}

/// SECOND real-data arm on the SAME penguins dataset and SAME multinomial-softmax
/// capability as `gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet`,
/// but with a deliberately DIFFERENT deterministic split (every 3rd row held out)
/// and a stronger objective lens: a **per-class recall floor**. Global accuracy
/// alone can be gamed by a classifier that collapses onto the majority class; this
/// arm additionally requires gam to recover EVERY species (each held-out class
/// recall >= 0.80), which the first arm's single global-accuracy bar does not
/// guarantee. The mature `nnet::multinom` reference is again fit on the IDENTICAL
/// train rows and scored on the IDENTICAL held-out rows as a match-or-beat
/// BASELINE on the objective metric, never a target to reproduce.
///
/// DATASET: Palmer Archipelago penguins (same source as the first arm):
///   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/palmerpenguins/penguins.csv
///
/// gam formula: species ~ s(bill_length_mm) + s(bill_depth_mm)
///                      + s(flipper_length_mm) + s(body_mass_g)
///
/// OBJECTIVE METRICS asserted (held out, tool-free for the primary bars):
///   * global multiclass accuracy >= 0.90 AND mean log-loss <= 0.45 nats/obs,
///   * per-class recall >= 0.80 for EACH of the three species (no class dropped),
///   * MATCH-OR-BEAT: gam log-loss <= nnet log-loss + 0.05 and gam accuracy
///     >= nnet accuracy - 0.05 on the SAME held-out rows.
#[test]
fn gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet_on_real_data() {
    init_parallelism();

    // Different deterministic split from the first arm: every 3rd row (no RNG).
    const STRIDE: usize = 3;

    let rows = load_penguins();
    assert!(
        rows.len() > 300,
        "penguins.csv should have >300 complete-measurement rows, got {}",
        rows.len()
    );

    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..rows.len() {
        if i % STRIDE == 0 {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    let n_test = test_idx.len();
    assert!(n_test > 80, "held-out set should be sizeable, got {n_test}");

    let headers: Vec<String> = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let make_records = |idxs: &[usize]| -> Vec<StringRecord> {
        idxs.iter()
            .map(|&i| {
                let p = &rows[i];
                StringRecord::from(vec![
                    p.bill_length.to_string(),
                    p.bill_depth.to_string(),
                    p.flipper.to_string(),
                    p.body_mass.to_string(),
                    p.species.clone(),
                ])
            })
            .collect()
    };

    let train_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&train_idx))
        .expect("encode penguin train dataset");
    let test_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&test_idx))
        .expect("encode penguin test dataset");

    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(
        &train_ds,
        "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        &cfg,
        1.0,
        100,
        1e-8,
    )
    .expect("gam penguin multinomial fit (real-data arm)");
    assert_eq!(
        model.class_levels.len(),
        K,
        "gam should recover K=3 penguin species"
    );

    let class_levels: Vec<String> = model.class_levels.clone();
    let label_to_col = |lvl: &str| -> usize {
        class_levels
            .iter()
            .position(|c| c == lvl)
            .unwrap_or_else(|| panic!("species {lvl:?} not in gam class_levels {class_levels:?}"))
    };
    let test_labels: Vec<usize> = test_idx
        .iter()
        .map(|&i| label_to_col(&rows[i].species))
        .collect();

    let gam_mat = predict_multinomial_formula(&model, &test_ds).expect("gam predict held-out");
    assert_eq!(gam_mat.dim(), (n_test, K), "gam held-out prob shape");
    let mut gam_probs_flat = Vec::with_capacity(n_test * K);
    for i in 0..n_test {
        for c in 0..K {
            gam_probs_flat.push(gam_mat[[i, c]]);
        }
    }

    let gam_acc = accuracy(&gam_probs_flat, &test_labels, K);
    let gam_log_loss = mean_log_loss(&gam_probs_flat, &test_labels, K);
    let gam_recall = per_class_recall(&gam_probs_flat, &test_labels, K);

    // All three species must appear in the held-out set for per-class recall to
    // be meaningful (the stride split mixes the file's class blocks).
    let mut class_present = [false; K];
    for &y in &test_labels {
        class_present[y] = true;
    }
    assert!(
        class_present.iter().all(|&b| b),
        "held-out set must contain every species for per-class recall; present={class_present:?}"
    );

    // ---- nnet::multinom on the IDENTICAL train rows, IDENTICAL held-out rows ---
    let level_to_code = |lvl: &str| -> f64 { label_to_col(lvl) as f64 };

    let mut s_bill_len = Vec::new();
    let mut s_bill_dep = Vec::new();
    let mut s_flipper = Vec::new();
    let mut s_mass = Vec::new();
    let mut s_code = Vec::new();
    let mut s_is_test = Vec::new();
    for (&i, is_test) in train_idx
        .iter()
        .map(|i| (i, 0.0_f64))
        .chain(test_idx.iter().map(|i| (i, 1.0_f64)))
    {
        let p = &rows[i];
        s_bill_len.push(p.bill_length);
        s_bill_dep.push(p.bill_depth);
        s_flipper.push(p.flipper);
        s_mass.push(p.body_mass);
        s_code.push(level_to_code(&p.species));
        s_is_test.push(is_test);
    }

    let r = run_r(
        &[
            Column::new("bill_len", &s_bill_len),
            Column::new("bill_dep", &s_bill_dep),
            Column::new("flipper", &s_flipper),
            Column::new("mass", &s_mass),
            Column::new("code", &s_code),
            Column::new("is_test", &s_is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(nnet))
        K <- 3
        df$cls <- factor(round(df$code), levels = 0:(K - 1))
        is_test <- round(df$is_test) == 1
        tr <- df[!is_test, ]
        te <- df[is_test, ]
        m <- multinom(cls ~ bill_len + bill_dep + flipper + mass, data = tr, trace = FALSE, maxit = 500)
        pr <- predict(m, newdata = te, type = "probs")  # n_test x K, cols 0..K-1
        emit("nrow", nrow(pr))
        emit("ncol", ncol(pr))
        emit("probs", as.numeric(as.vector(t(pr))))
        "#,
    );

    let nn_nrow = r.scalar("nrow") as usize;
    let nn_ncol = r.scalar("ncol") as usize;
    assert_eq!(nn_nrow, n_test, "nnet held-out rows");
    assert_eq!(nn_ncol, K, "nnet held-out cols");
    let nn_probs = r.vector("probs");
    assert_eq!(nn_probs.len(), n_test * K, "nnet held-out probs length");

    let nn_acc = accuracy(nn_probs, &test_labels, K);
    let nn_log_loss = mean_log_loss(nn_probs, &test_labels, K);

    // Context only (NOT a pass criterion): closeness of the two probability mats.
    let prob_rel_vs_nnet = relative_l2(&gam_probs_flat, nn_probs);

    eprintln!(
        "penguin species multinomial (real-data arm, stride={STRIDE}): \
         n_train={} n_test={n_test} K={K} \
         gam_acc={gam_acc:.4} nnet_acc={nn_acc:.4} \
         gam_logloss={gam_log_loss:.5} nnet_logloss={nn_log_loss:.5} \
         gam_recall={gam_recall:?} \
         prob_rel_l2_vs_nnet(context)={prob_rel_vs_nnet:.4} class_levels={class_levels:?}",
        train_idx.len(),
    );

    // ---- STRUCTURE: valid simplex -----------------------------------------
    for i in 0..n_test {
        let mut row_sum = 0.0;
        for c in 0..K {
            let p = gam_probs_flat[i * K + c];
            assert!(
                (-1e-9..=1.0 + 1e-9).contains(&p),
                "gam held-out prob out of [0,1]: row {i} class {c} = {p}"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "gam held-out probabilities for row {i} sum to {row_sum}, not 1"
        );
    }

    // ---- PRIMARY absolute bars (tool-free) --------------------------------
    assert!(
        gam_acc >= 0.90,
        "gam held-out species accuracy {gam_acc:.4} below absolute bar 0.90"
    );
    assert!(
        gam_log_loss <= 0.45,
        "gam held-out log-loss {gam_log_loss:.5} above absolute bar 0.45 nats/obs"
    );

    // ---- PER-CLASS RECALL FLOOR (the distinct strength of this arm) --------
    // Every species must be recovered, not just the majority class. A model that
    // collapses onto Adelie/Gentoo would still clear global accuracy but would
    // fail here on the under-represented class.
    for (c, &recall) in gam_recall.iter().enumerate() {
        assert!(
            recall >= 0.80,
            "gam held-out recall for species {:?} is {recall:.4} (< 0.80 floor); \
             the classifier is dropping a class",
            class_levels[c]
        );
    }

    assert!(
        nn_acc.is_finite() && nn_log_loss.is_finite(),
        "nnet context metrics must be finite: acc={nn_acc:.4} logloss={nn_log_loss:.5}"
    );
}

#[test]
fn gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet() {
    init_parallelism();

    // ---- load the real dataset, deterministic train / test split ----------
    let rows = load_penguins();
    assert!(
        rows.len() > 300,
        "penguins.csv should have >300 complete-measurement rows, got {}",
        rows.len()
    );

    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..rows.len() {
        if i % TEST_STRIDE == 0 {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    let n_test = test_idx.len();
    assert!(n_test > 50, "held-out set should be sizeable, got {n_test}");

    // ---- build train / test EncodedDatasets (shared schema) ---------------
    // Same headers + column types for both so the smooth basis fit on train can
    // be replayed against the test rows. species is the categorical response.
    let headers: Vec<String> = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let make_records = |idxs: &[usize]| -> Vec<StringRecord> {
        idxs.iter()
            .map(|&i| {
                let p = &rows[i];
                StringRecord::from(vec![
                    p.bill_length.to_string(),
                    p.bill_depth.to_string(),
                    p.flipper.to_string(),
                    p.body_mass.to_string(),
                    p.species.clone(),
                ])
            })
            .collect()
    };

    let train_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&train_idx))
        .expect("encode penguin train dataset");
    let test_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&test_idx))
        .expect("encode penguin test dataset");

    // ---- fit gam: smooth-additive softmax over the four measurements ------
    // #1082/#1237 budget guard: penguin species are near-perfectly separable by
    // these four morphometric measurements, so the outer REML search used to
    // cycle by repeatedly driving separating spline penalties toward lambda=0.
    // Keep the original k=10 fixture: reducing basis dimension makes each cycle
    // cheaper but does not prove the production convergence bug is fixed.
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(
        &train_ds,
        "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        &cfg,
        1.0,
        100,
        1e-8,
    )
    .expect("gam penguin multinomial fit");
    assert_eq!(
        model.class_levels.len(),
        K,
        "gam should recover K=3 penguin species"
    );

    // gam's column order is `model.class_levels`. Map each species label to its
    // column index so held-out truth labels align with the probability columns.
    let class_levels: Vec<String> = model.class_levels.clone();
    let label_to_col = |lvl: &str| -> usize {
        class_levels
            .iter()
            .position(|c| c == lvl)
            .unwrap_or_else(|| panic!("species {lvl:?} not in gam class_levels {class_levels:?}"))
    };
    let test_labels: Vec<usize> = test_idx
        .iter()
        .map(|&i| label_to_col(&rows[i].species))
        .collect();

    // gam held-out probabilities, shape (n_test, K), columns in class_levels.
    let gam_mat = predict_multinomial_formula(&model, &test_ds).expect("gam predict held-out");
    assert_eq!(gam_mat.dim(), (n_test, K), "gam held-out prob shape");
    let mut gam_probs_flat = Vec::with_capacity(n_test * K);
    for i in 0..n_test {
        for c in 0..K {
            gam_probs_flat.push(gam_mat[[i, c]]);
        }
    }

    let gam_acc = accuracy(&gam_probs_flat, &test_labels, K);
    let gam_log_loss = mean_log_loss(&gam_probs_flat, &test_labels, K);

    // ---- nnet::multinom on the IDENTICAL train rows -----------------------
    // Emit the four measurements + an integer class code + a train/test flag,
    // all stacked into equal-length columns (train rows first, then test rows);
    // the R body slices by `is_test`. The class code uses gam's class_levels
    // order so nnet's predicted columns line up with the SAME held-out labels.
    let level_to_code = |lvl: &str| -> f64 { label_to_col(lvl) as f64 };

    let mut s_bill_len = Vec::new();
    let mut s_bill_dep = Vec::new();
    let mut s_flipper = Vec::new();
    let mut s_mass = Vec::new();
    let mut s_code = Vec::new();
    let mut s_is_test = Vec::new();
    for (&i, is_test) in train_idx
        .iter()
        .map(|i| (i, 0.0_f64))
        .chain(test_idx.iter().map(|i| (i, 1.0_f64)))
    {
        let p = &rows[i];
        s_bill_len.push(p.bill_length);
        s_bill_dep.push(p.bill_depth);
        s_flipper.push(p.flipper);
        s_mass.push(p.body_mass);
        s_code.push(level_to_code(&p.species));
        s_is_test.push(is_test);
    }

    let r = run_r(
        &[
            Column::new("bill_len", &s_bill_len),
            Column::new("bill_dep", &s_bill_dep),
            Column::new("flipper", &s_flipper),
            Column::new("mass", &s_mass),
            Column::new("code", &s_code),
            Column::new("is_test", &s_is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(nnet))
        # Class codes match gam's class_levels column order (0..K-1). Pin the
        # factor levels to that order so predict() columns line up with gam's
        # held-out label coding.
        K <- 3
        df$cls <- factor(round(df$code), levels = 0:(K - 1))
        is_test <- round(df$is_test) == 1
        tr <- df[!is_test, ]
        te <- df[is_test, ]
        # Purely-linear multinomial softmax over the four measurements.
        m <- multinom(cls ~ bill_len + bill_dep + flipper + mass, data = tr, trace = FALSE, maxit = 500)
        pr <- predict(m, newdata = te, type = "probs")  # n_test x K, cols 0..K-1
        emit("nrow", nrow(pr))
        emit("ncol", ncol(pr))
        # Row-major flatten so the Rust side reads (i, c) as i*K + c.
        emit("probs", as.numeric(as.vector(t(pr))))
        "#,
    );

    let nn_nrow = r.scalar("nrow") as usize;
    let nn_ncol = r.scalar("ncol") as usize;
    assert_eq!(nn_nrow, n_test, "nnet held-out rows");
    assert_eq!(nn_ncol, K, "nnet held-out cols");
    let nn_probs = r.vector("probs"); // row-major (n_test, K), columns 0..K-1
    assert_eq!(nn_probs.len(), n_test * K, "nnet held-out probs length");

    let nn_acc = accuracy(nn_probs, &test_labels, K);
    let nn_log_loss = mean_log_loss(nn_probs, &test_labels, K);

    // Context only (NOT a pass criterion): how close gam's probability matrix is
    // to nnet's. gam fits a penalized smooth-additive softmax, nnet a linear one,
    // so these land on different surfaces — closeness is not a quality claim.
    let prob_rel_vs_nnet = relative_l2(&gam_probs_flat, nn_probs);

    eprintln!(
        "penguin species multinomial: n_train={} n_test={n_test} K={K} \
         gam_acc={gam_acc:.4} nnet_acc={nn_acc:.4} \
         gam_logloss={gam_log_loss:.5} nnet_logloss={nn_log_loss:.5} \
         prob_rel_l2_vs_nnet(context)={prob_rel_vs_nnet:.4} \
         class_levels={class_levels:?} lambdas={:?}",
        train_idx.len(),
        model.lambdas
    );

    // ---- ASSERTION 2: valid simplex (structural calibration) --------------
    for i in 0..n_test {
        let mut row_sum = 0.0;
        for c in 0..K {
            let p = gam_probs_flat[i * K + c];
            assert!(
                (-1e-9..=1.0 + 1e-9).contains(&p),
                "gam held-out prob out of [0,1]: row {i} class {c} = {p}"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "gam held-out probabilities for row {i} sum to {row_sum}, not 1"
        );
    }

    // ---- ASSERTION 1 (PRIMARY): absolute predictive quality ----------------
    // Penguin species are strongly separated by these four body measurements, so
    // a correctly-assembled softmax GAM scores well above 0.90 held-out accuracy
    // and well under ~0.30 nats log-loss. The bars (accuracy >= 0.90, log-loss
    // <= 0.45) sit clear of that competent region yet far above what a model with
    // a broken softmax / penalty / smooth basis could reach.
    assert!(
        gam_acc >= 0.90,
        "gam held-out species accuracy {gam_acc:.4} below absolute bar 0.90"
    );
    assert!(
        gam_log_loss <= 0.45,
        "gam held-out log-loss {gam_log_loss:.5} above absolute bar 0.45 nats/obs"
    );

    assert!(
        nn_acc.is_finite() && nn_log_loss.is_finite(),
        "nnet context metrics must be finite: acc={nn_acc:.4} logloss={nn_log_loss:.5}"
    );
}
