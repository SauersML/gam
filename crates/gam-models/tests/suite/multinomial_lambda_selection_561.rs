//! Regression guard for #561, from a different angle than the R-backed
//! `quality_vs_vgam_multinomial_softmax` quality arm: this test needs no R and
//! pins the STRUCTURAL invariants the fused-λ / dead-selection bugs violate.
//!
//! The #561 family of defects all left the multinomial outer REML/LAML loop
//! unable to actually SELECT smoothing parameters — either fusing every smooth
//! term into one shared λ, or (after the #1587 joint-penalty refactor) leaving
//! the per-term λ pinned at their seed because the custom-family outer path
//! rejected every ρ-evaluation (dimension contract), reused a stale operator
//! (cache fingerprint), dropped `½log|H|` from the cost (projected logdet), or
//! folded a phantom KKT-residual correction (inner residual omitted the joint
//! penalty). In every case the fit came back essentially UNPENALIZED (EDF near
//! the coefficient count) with `λ ≡ init`, and the recovered probability surface
//! sat far past the truth.
//!
//! This test synthesizes the same known softmax surface as the quality arm
//! (cubic in x1, sigmoid in x2, per-class linear x3), fits the penalized
//! multinomial GAM, and asserts three things that hold ONLY when per-term REML
//! selection genuinely runs:
//!   1. truth recovery — RMSE(P_gam, P_true) is small;
//!   2. penalization is active — per-class EDF is well below the coefficient
//!      count (a near-unpenalized overfit is the dead-selection signature);
//!   3. per-term λ are genuinely selected and differ — a fused/dead selector
//!      returns near-equal λ (or all equal to the seed), so the within-class
//!      spread collapses.
//! It also fits from two very different seeds and asserts the selected λ do not
//! merely echo `init_lambda` (the smoking gun that selection never moved).

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N: usize = 300;
const K: usize = 3;

fn true_eta(x1: f64, x2: f64, x3: f64) -> [f64; K] {
    let cubic = 2.0 * x1.powi(3) - 1.0 * x1;
    let sigmoid = 3.0 / (1.0 + (-6.0 * (x2 - 0.5)).exp()) - 1.5;
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

fn synth() -> (gam_data::EncodedDataset, Vec<[f64; K]>) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).unwrap();
    let u01 = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).unwrap();
    let udraw = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let mut rows = Vec::with_capacity(N);
    let mut truth = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let p = softmax(&true_eta(a, b, c));
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
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            format!("c{chosen}"),
        ]));
        truth.push(p);
    }
    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        truth,
    )
}

fn rmse_vs_truth(
    model: &gam_models::multinomial::MultinomialSavedModel,
    ds: &gam_data::EncodedDataset,
    truth: &[[f64; K]],
) -> f64 {
    let probs = predict_multinomial_formula(model, ds).expect("predict");
    let col_code: Vec<usize> = model
        .class_levels
        .iter()
        .map(|l| l.trim_start_matches('c').parse::<usize>().unwrap())
        .collect();
    let mut se = 0.0;
    for k in 0..K {
        for i in 0..N {
            let d = probs[[i, k]] - truth[i][col_code[k]];
            se += d * d;
        }
    }
    (se / (N * K) as f64).sqrt()
}

#[test]
fn multinomial_outer_reml_selects_per_term_lambda_and_recovers_truth() {
    let (ds, truth) = synth();
    let cfg = FitConfig::default();
    let formula = "y ~ s(x1, k=6) + s(x2, k=6) + x3";
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, formula, &cfg)
    })
    .expect("gam multinomial fit");
    // (4)'s second fit is run HERE, before any assertion, because it is the one
    // measurement that separates the two things this test can fail for.
    let model_hi = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 50.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, formula, &cfg)
    })
    .expect("gam multinomial fit (init=50)");

    // ── Every quantity this test asserts on, MEASURED AND PRINTED FIRST ──
    //
    // (#4004) The four assertions below are ordered truth-recovery first, and
    // truth recovery is the one of the four that does NOT distinguish a dead
    // smoothing selection from a live selection that landed in a worse basin.
    // When (1) fired, the run ended there and (2), (3) and (4) — the three that
    // do distinguish them — were never evaluated, so the receipt said only
    // "RMSE is 0.0688" and the attribution had to be guessed. A fixture's
    // diagnostics must not be conditional on its assertions, so every number is
    // taken and reported before the first `assert!`.
    let rmse = rmse_vs_truth(&model, &ds, &truth);
    let p_per_class = model.p_per_class as f64;
    let edf = model
        .edf_per_class
        .as_ref()
        .expect("REML fit must report per-class EDF");
    let per_block = &model.lambdas_per_block;
    let n0 = per_block.first().copied().unwrap_or(0);
    let class0 = &model.lambdas[..n0.min(model.lambdas.len())];
    let lam_max = class0.iter().cloned().fold(f64::MIN, f64::max);
    let lam_min = class0.iter().cloned().fold(f64::MAX, f64::min);
    let echoes_seed = model_hi.lambdas.iter().all(|&l| (l - 50.0).abs() < 1e-6);
    eprintln!(
        "multinomial per-term lambda (#561/#4004): rmse={rmse:.5} (bar 0.065; the pinned-lambda \
         driver measured >= 0.07 and the fused-lambda driver >= 0.13 on this DGP) \
         p_per_class={p_per_class} edf_per_class={edf:?} lambdas_per_block={per_block:?} \
         class0_lambda=[{lam_min:.6}, {lam_max:.6}] span={:.4} \
         lambdas(init=1)={:?} lambdas(init=50)={:?} echoes_seed={echoes_seed}",
        if lam_min > 0.0 { lam_max / lam_min } else { f64::INFINITY },
        model.lambdas,
        model_hi.lambdas,
    );

    // The shape of what was measured. These are preconditions for reading the
    // four bars at all, so they stay immediate.
    assert_eq!(edf.len(), K - 1, "one EDF entry per active class");
    assert!(!per_block.is_empty(), "must report per-class λ block sizes");
    assert!(
        n0 >= 2,
        "class 0 must carry ≥2 penalty components, got {n0}"
    );

    // (#4004) The four bars are COLLECTED and raised together, diagnosing ones
    // first. Truth recovery is the only one of the four that does not tell a
    // dead smoothing selection from a live selection that landed in a worse
    // basin, and it used to be asserted first, so a red run stopped there and
    // reported "RMSE is 0.0688" while (2), (3) and (4) -- the three that name
    // WHICH of the two it is -- were never asserted. A fixture that stops at
    // its least informative bar makes every run cost one more run.
    let mut failures: Vec<String> = Vec::new();

    // (2) Penalization is active. Per-class EDF must sit well below the
    // per-class coefficient count; a near-unpenalized fit (EDF ≈ p) is the
    // dead-selection signature (λ pinned at the seed).
    for (a, &e) in edf.iter().enumerate() {
        if !(e.is_finite() && e > 0.0 && e < 0.75 * p_per_class) {
            failures.push(format!(
                "(2) class {a} EDF={e:.3} is not in (0, 0.75·p={:.2}): the fit is \
                 near-unpenalized, so REML never selected a smoothing parameter",
                0.75 * p_per_class
            ));
        }
    }

    // (3) Per-term λ are genuinely selected and DIFFER. With #561 fixed the
    // rough cubic term and the smoother sigmoid/null-space terms take very
    // different λ; a fused or dead selector returns near-equal λ (or all == the
    // seed 1.0). Check the within-class span.
    if !(lam_max / lam_min > 5.0) {
        failures.push(format!(
            "(3) within-class per-term λ barely differ (max={lam_max:.4} min={lam_min:.4}); \
             REML is not selecting independent per-term smoothing (fused-λ regression)"
        ));
    }

    // (4) Selection is not a passthrough of the seed. The init=50 fit above is
    // the control: its recovered λ must NOT all be ≈ init (the exact
    // dead-selection fingerprint: λ ≡ init for every component).
    if echoes_seed {
        failures.push(format!(
            "(4) every selected λ equals the init seed 50.0 — the outer smoothing search \
             never moved (dead selection): {:?}",
            model_hi.lambdas
        ));
    }

    // (1) Truth recovery. The fused-λ driver measured ≥ 0.13 and the pinned-λ
    // (dead-selection) driver ≥ 0.07 on this DGP; a working per-term REML fit
    // recovers to a few percent.
    //
    // NOTE (#4004): 0.065 against a pinned-λ driver at 0.07 leaves a 7% window
    // between "working" and the failure mode this bar names, so a value inside
    // it does not tell a small quality regression from dead selection. That is
    // an argument for deriving the bar, not for moving it, and it is why (2),
    // (3) and (4) are the assertions that carry the diagnosis. The bar is
    // unchanged here: the two reference numbers it sits between were measured
    // on drivers this test cannot construct -- `MultinomialFitRequest` carries
    // no way to hold λ at its seed -- so denominating it against a control
    // needs that control to exist first.
    if !(rmse < 0.065) {
        failures.push(format!(
            "(1) multinomial fit did not recover the true simplex: RMSE={rmse:.5} (>= 0.065 \
             indicates fused-λ or a stalled outer smoothing selection)"
        ));
    }

    assert!(
        failures.is_empty(),
        "multinomial per-term λ selection (#561/#4004), every failing bar:\n{failures:#?}"
    );
}
