//! Internal-consistency identity for gam's penalized multinomial-logit
//! (softmax) GAM: the **stored deviance must equal an independent softmax
//! recompute** of `-2 · Σ_i log p̂(y_i | x_i)` from the fitted probabilities.
//!
//! There is no external multinomial GAM that exposes the *exact same* penalty
//! gauge and basis as gam, so the mature, unimpeachable reference here is the
//! *definition itself*: for a categorical likelihood the unpenalized deviance
//! is `-2 · Σ_i log softmax(η_i)[y_i]`. gam stores this value in
//! `MultinomialSavedModel::deviance` (read straight off the converged
//! unpenalized log-likelihood inside the REML driver, issue #348). A
//! completely independent re-derivation — rebuild the design from the frozen
//! termspec, evaluate `softmax(X·β)` via the public predict path, then sum the
//! per-row log-probability of the realized class — MUST reproduce that stored
//! number to floating-point precision. If the two diverge, every downstream
//! consumer of the stored deviance (AIC/BIC model selection, deviance-residual
//! diagnostics, likelihood-ratio tests) is silently wrong.
//!
//! Combination under test (bugs hide in combinations): a single multinomial
//! fit that simultaneously loads a cyclic 1-D smooth `s(x1, bs='cc')`, a
//! thin-plate 1-D smooth `s(x2, bs='tp')`, AND a tensor-product interaction
//! `te(x1, x2, bs=c('cc','tp'))` — three penalty blocks per active class,
//! replicated across `K-1 = 2` softmax linear predictors. This exercises the
//! full parse → termspec → multi-block design → one-hot → REML/Newton →
//! stored-deviance pipeline, then checks the bookkeeping identity end-to-end.
//!
//! Bound: `|deviance_stored − deviance_recompute|` must be below 1e-8 absolute
//! and 1e-10 relative. Both numbers are `-2·Σ log p` over the SAME β̂ and the
//! SAME rows; the only differences are (a) gam sums the log-likelihood inside
//! the family `evaluate` while we sum `log(predict_probabilities)` here, and
//! (b) the predict path recomputes `softmax(X·β)` from the stored coefficients
//! rather than reusing the converged `η`. Both are the identical arithmetic in
//! f64, so the residual is pure summation-order / reassociation rounding —
//! O(n · ε_machine) ≈ 300 · 2.2e-16 ≈ 7e-14 in the worst case. 1e-8 absolute is
//! a deliberately generous ceiling that still fails hard on any *structural*
//! mismatch (wrong reference class, dropped row, penalty leaking into the
//! "unpenalized" deviance), while 1e-10 relative guards the same on scale.

use gam::data::EncodedDataset;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

use csv::StringRecord;
use std::f64::consts::PI;

/// Synthetic, fully deterministic (RNG-free) categorical dataset.
///
/// `(x1, x2)` sweep the rectangle `[0, 2π] × [-3, 3]` on a deterministic
/// space-filling lattice; the class is the `argmax` of the softmax logits
/// `[1.5·sin(x1), −0.8·cos(x1)·x2, 0]`, encoded as the string labels
/// `"A"/"B"/"C"`. Returns the encoded dataset plus the raw `(x1, x2, label)`
/// triples so the recompute can re-derive the realized class index against
/// gam's own level ordering.
fn make_multinomial_dataset(n: usize) -> (EncodedDataset, Vec<f64>, Vec<f64>, Vec<String>) {
    // Two coprime irrational strides give a deterministic, well-spread
    // additive-recurrence (Weyl) sequence over the unit square — no RNG, no
    // duplicate rows, and good coverage of every corner of the rectangle.
    let stride1 = (2.0_f64).sqrt().fract(); // ≈ 0.41421356
    let stride2 = (3.0_f64).sqrt().fract(); // ≈ 0.73205081
    let mut u1 = 0.12_f64;
    let mut u2 = 0.37_f64;

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

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
        let class = if l0 >= l1 && l0 >= l2 {
            "A"
        } else if l1 >= l0 && l1 >= l2 {
            "B"
        } else {
            "C"
        };

        x1.push(a);
        x2.push(b);
        labels.push(class.to_string());
    }

    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                labels[i].clone(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");
    (ds, x1, x2, labels)
}

#[test]
fn multinomial_stored_deviance_matches_independent_softmax_recompute() {
    init_parallelism();

    let n = 300;
    let (ds, _x1, _x2, labels) = make_multinomial_dataset(n);

    // ---- fit gam's multinomial-logit GAM with the loaded combination -------
    // Cyclic 1-D smooth on x1 (the angular covariate), thin-plate 1-D smooth on
    // x2, and a tensor-product interaction across both. Three penalty blocks
    // per active class, replicated over K-1 = 2 softmax predictors.
    let formula = "y ~ s(x1, bs='cc', k=8) + s(x2, bs='tp', k=5) + te(x1, x2, bs=c('cc','tp'))";
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&ds, formula, &cfg, 1.0, 50, 1e-7)
        .expect("multinomial formula fit");

    // The synthetic data has all three classes present; the reference class is
    // the last level in first-appearance order.
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

    // ---- independent softmax recompute of the deviance ---------------------
    // Replay the SAME data through the public predict path: this rebuilds the
    // design from the frozen termspec and evaluates softmax(X·β̂) afresh from
    // the stored coefficients — no shared state with the value gam stored.
    let probs = predict_multinomial_formula(&model, &ds).expect("multinomial predict probabilities");
    assert_eq!(probs.dim(), (n, model.class_levels.len()), "probs shape");

    // Map each row's realized label to gam's own class column index, so the
    // recompute indexes the predicted probabilities in the exact gauge gam
    // used (reference class = last level). Re-deriving the index from gam's
    // `class_levels` (rather than assuming A/B/C order) makes the recompute
    // robust to the schema's first-appearance level ordering.
    let class_index = |label: &str| -> usize {
        model
            .class_levels
            .iter()
            .position(|lvl| lvl == label)
            .unwrap_or_else(|| panic!("label {label:?} not among class levels {:?}", model.class_levels))
    };

    // deviance = -2 · Σ_i log p̂(y_i | x_i), summed exactly the way the
    // definition prescribes. Rows must each have a finite, strictly-positive
    // realized-class probability or the recompute is undefined.
    let mut neg2_loglik = 0.0_f64;
    for (i, label) in labels.iter().enumerate() {
        let c = class_index(label);
        let p = probs[[i, c]];
        assert!(
            p.is_finite() && p > 0.0,
            "row {i}: realized-class probability {p} is non-positive/non-finite"
        );
        neg2_loglik += p.ln();
    }
    let deviance_recompute = -2.0 * neg2_loglik;

    let deviance_stored = model.deviance;
    let abs_diff = (deviance_stored - deviance_recompute).abs();
    let rel_diff = abs_diff / deviance_recompute.abs().max(1.0);

    eprintln!(
        "[multinomial-deviance-identity] n={n} K={} converged={} \
         deviance_stored={deviance_stored:.12} deviance_recompute={deviance_recompute:.12} \
         abs_diff={abs_diff:.3e} rel_diff={rel_diff:.3e}",
        model.class_levels.len(),
        model.converged
    );

    // The stored deviance is the unpenalized -2·log-lik; it must equal the
    // independent softmax recompute to f64 reassociation precision. A larger
    // gap means a structural bookkeeping bug (penalty leaking into "deviance",
    // a permuted/dropped reference class, or a stale coefficient block) — not
    // rounding. See the module header for the 1e-8 / 1e-10 derivation.
    assert!(
        abs_diff < 1e-8,
        "stored deviance disagrees with independent softmax recompute: \
         stored={deviance_stored:.12} recompute={deviance_recompute:.12} abs_diff={abs_diff:.3e}"
    );
    assert!(
        rel_diff < 1e-10,
        "stored deviance disagrees with independent softmax recompute (relative): \
         stored={deviance_stored:.12} recompute={deviance_recompute:.12} rel_diff={rel_diff:.3e}"
    );
}
