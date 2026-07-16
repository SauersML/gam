//! Round-trip regression for the #2273 exact-separation binomial rescue.
//!
//! #2273's fix let a perfectly-separated binomial `y ~ x` fit mint a model
//! instead of hard-failing (a `StalledAtValidMinimum` inner state whose
//! outer criterion certificate certifies, or — after #2316's PIRLS
//! final-state gate — a terminal state promoted straight to `Converged`).
//! That fix landed at the in-process `fit_from_formula` layer
//! (`crates/gam-models/src/fit_orchestration/perfect_binomial_separation_2273_tests.rs`),
//! which proves the fit itself mints. It does not exercise what a CLI user
//! actually does next: save the minted model to disk with `--out`, then load
//! it back in a *separate* process for `gam predict`.
//!
//! That gap mattered in practice: `FitConvergenceEvidence` has two
//! independent gates guarding the same invariant — `try_from_parts` (used
//! when the fit is first assembled) and `from_serialized`/`Deserialize`
//! (used every time a saved model is loaded back, e.g. by `gam predict`,
//! `gam report`, `gam diagnose`). Between 1fd52f05b (which added a
//! `StalledAtValidMinimum`-with-certificate exception to `try_from_parts`
//! only) and cea1a6ed5 (`fix(#2316): certify stalled final PIRLS states`,
//! which replaced that exception with an upstream status promotion), a
//! model minted by `gam fit --out` for this exact scenario could still be
//! rejected by `gam predict`/`gam report` with "inner optimizer status
//! StalledAtValidMinimum is not converged" — a model that fits and saves
//! successfully but can never be loaded again. Two gates enforcing "the same"
//! contract can silently drift apart; only a real save-then-reload-in-a-new-
//! process test catches that.
//!
//! This test pins the full CLI round trip: fit the exact-separation fixture
//! (n=40, same construction as the #2273 issue's n-sweep table) to `--out`,
//! then `gam predict` the saved model on a small grid spanning both classes,
//! and assert the predicted probabilities actually separate (near 0 for the
//! class-0 region, near 1 for the class-1 region) rather than landing on a
//! degenerate near-flat ~0.5 fit that happens to satisfy the file-format
//! checks without capturing the signal at all.

use std::process::Command;

fn tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn exact_separation_binomial_fit_saves_and_predicts_after_reload() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/exact_separation_binomial_n40_2273.csv"
    );
    let model = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp model path");

    let fit = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(fixture)
        .arg("y ~ x")
        .arg("--family")
        .arg("binomial-logit")
        .arg("--out")
        .arg(model.path())
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "gam fit on the exact-separation fixture must mint a model (#2273), \
         not hard-fail (exit {:?}).\nstderr tail: {}",
        fit.status.code(),
        tail(&fit.stderr)
    );
    let saved = std::fs::metadata(model.path()).expect("model file must exist after fit");
    assert!(saved.len() > 0, "saved model file must be non-empty");

    // A grid spanning both the class-0 support ([1, 3]) and the class-1
    // support ([10, 12]), plus the midpoint gap.
    let grid = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp grid path");
    std::fs::write(grid.path(), "x\n2\n6\n11\n").expect("write grid csv");

    let preds = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp predictions path");
    let predict = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("predict")
        .arg(model.path())
        .arg(grid.path())
        .arg("--out")
        .arg(preds.path())
        .output()
        .expect("spawn gam predict");
    assert!(
        predict.status.success(),
        "gam predict must load the model gam fit just saved for this exact-\
         separation scenario (#2273 round trip) (exit {:?}).\nstderr tail: {}",
        predict.status.code(),
        tail(&predict.stderr)
    );

    let report = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("report")
        .arg(model.path())
        .arg(fixture)
        .arg(
            tempfile::Builder::new()
                .suffix(".html")
                .tempfile()
                .expect("temp report path")
                .path(),
        )
        .output()
        .expect("spawn gam report");
    assert!(
        report.status.success(),
        "gam report must also load the saved exact-separation model (exit {:?}).\n\
         stderr tail: {}",
        report.status.code(),
        tail(&report.stderr)
    );

    let csv = std::fs::read_to_string(preds.path()).expect("read predictions csv");
    let mut rows = csv.lines();
    let header = rows.next().expect("header row");
    let mean_col = header
        .split(',')
        .position(|c| c == "mean")
        .expect("predictions csv must have a mean column");
    let means: Vec<f64> = rows
        .map(|line| {
            line.split(',')
                .nth(mean_col)
                .expect("mean value")
                .parse::<f64>()
                .expect("mean value parses as f64")
        })
        .collect();
    assert_eq!(means.len(), 3, "expected one prediction per grid row");
    let (p_class0, p_mid, p_class1) = (means[0], means[1], means[2]);

    // The fit must actually separate the classes, not collapse to a
    // near-constant ~0.5 prediction that would still pass every structural
    // check above while being useless.
    assert!(
        p_class0 < 0.3,
        "predicted P(y=1) in the class-0 support region must be small, got {p_class0} \
         (means={means:?})"
    );
    assert!(
        p_class1 > 0.7,
        "predicted P(y=1) in the class-1 support region must be large, got {p_class1} \
         (means={means:?})"
    );
    assert!(
        p_mid > p_class0 && p_mid < p_class1,
        "predicted P(y=1) midway through the gap must sit strictly between the two \
         class regions' predictions, got class0={p_class0} mid={p_mid} class1={p_class1}"
    );
}
