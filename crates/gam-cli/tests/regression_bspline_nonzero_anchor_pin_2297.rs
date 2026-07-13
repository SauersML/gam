//! Regression (#2297): a non-zero B-spline endpoint anchor pins the fitted
//! smooth to the requested boundary value end to end (parse → fit → save →
//! load → predict), not just at basis-construction time.
//!
//! An `Anchored { value }` left endpoint decomposes the smooth as
//!
//!     f(x) = B_raw(x)·β_p  +  (B_raw(x)·Z)·γ,
//!
//! where `β_p` is the fixed particular solution reproducing `f(left)=value`
//! with `f'(left)=0`, and the constrained columns `B_raw·Z` vanish at the
//! anchored endpoint (they span the *homogeneous* value+slope null space). The
//! fixed `B_raw·β_p` channel is the design's affine offset, which the fit and
//! predict paths must fold into the linear predictor. At the left boundary the
//! prediction therefore collapses to exactly the anchor value, independently of
//! the data level there and of the fitted `γ`.
//!
//! This exercises the full CLI stack — the earlier coverage was basis-level
//! only (`bspline_nonzero_anchor_has_fixed_affine_lift_and_homogeneous_chart`)
//! and never fit or predicted a model. The training fixture sits at level ≈5
//! near x=0 and rises to ≈8 near x=1, so an *unanchored* smooth reports ≈5 at
//! x=0; the anchored fit must instead report exactly 1.0 there while still
//! tracking the data at the free right endpoint.

use std::process::Command;

fn parse_named_column(csv: &str, name: &str) -> Vec<f64> {
    let mut lines = csv.lines();
    let header = lines.next().expect("prediction CSV has a header row");
    let idx = header
        .split(',')
        .position(|h| h.trim() == name)
        .unwrap_or_else(|| panic!("prediction CSV has a `{name}` column; header was: {header}"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .nth(idx)
                .unwrap_or_else(|| panic!("row has a `{name}` cell"))
                .trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("`{name}` cell parses as f64"))
        })
        .collect()
}

fn stderr_tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Fit `formula` on the training fixture, predict on the newdata grid, and
/// return the `mean` column of the predictions.
fn fit_and_predict_mean(formula: &str) -> Vec<f64> {
    let train = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bspline_nonzero_anchor_train.csv"
    );
    let newdata = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bspline_nonzero_anchor_newdata.csv"
    );
    let model = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp model path");
    let out = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp output path");

    let fit = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(train)
        .arg(formula)
        .arg("--family")
        .arg("gaussian")
        .arg("--out")
        .arg(model.path())
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "gam fit `{formula}` failed (exit {:?}).\nstderr tail:\n{}",
        fit.status.code(),
        stderr_tail(&fit.stderr)
    );

    let predict = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("predict")
        .arg(model.path())
        .arg(newdata)
        .arg("--out")
        .arg(out.path())
        .output()
        .expect("spawn gam predict");
    assert!(
        predict.status.success(),
        "gam predict for `{formula}` failed (exit {:?}).\nstderr tail:\n{}",
        predict.status.code(),
        stderr_tail(&predict.stderr)
    );

    let csv = std::fs::read_to_string(out.path()).expect("read predictions");
    parse_named_column(&csv, "mean")
}

#[test]
fn nonzero_left_anchor_pins_prediction_to_value_end_to_end() {
    // The newdata grid starts at the anchored left endpoint x=0 and ends at the
    // free right endpoint x=1 (see the fixture).
    let anchored = fit_and_predict_mean("y ~ s(x, bc_left=anchored, anchor_left=1.0)");
    let free = fit_and_predict_mean("y ~ s(x)");

    assert_eq!(anchored.len(), 7, "one prediction per newdata row");
    assert_eq!(free.len(), 7);
    assert!(
        anchored.iter().chain(free.iter()).all(|v| v.is_finite()),
        "all predictions must be finite"
    );

    // (1) Exact pin: the anchored smooth equals the requested value at the
    // anchored endpoint to floating-point precision. The constrained design
    // columns vanish there, so the whole prediction is the affine offset =
    // anchor value. A loose tolerance would let a dropped-offset regression
    // (which returns ≈0 or the sum-to-zero data level) slip through.
    assert!(
        (anchored[0] - 1.0).abs() < 1e-4,
        "anchored prediction at the left endpoint must be the anchor value 1.0, got {}",
        anchored[0]
    );

    // (2) The pin is a genuine constraint, not the data: the SAME data fit
    // without an anchor reports the local data level (≈5) at x=0. If the anchor
    // offset were silently dropped, the anchored and free fits would agree here.
    assert!(
        free[0] > 3.0,
        "unanchored fit should report the data level (~5) at x=0, got {}",
        free[0]
    );
    assert!(
        (anchored[0] - free[0]).abs() > 2.0,
        "the anchor must move the left-endpoint prediction well away from the \
         unanchored data level (anchored={}, free={})",
        anchored[0],
        free[0]
    );

    // (3) The anchor is a LOCAL boundary pin, not a global level shift: away
    // from the anchored endpoint the anchored smooth still follows the data, so
    // at the free right endpoint it tracks the unanchored fit closely.
    let last = anchored.len() - 1;
    assert!(
        (anchored[last] - free[last]).abs() < 1.0,
        "at the free right endpoint the anchored and unanchored fits should agree \
         (anchored={}, free={})",
        anchored[last],
        free[last]
    );
    // And the anchored curve must actually rise from the pin toward the data,
    // i.e. it is not clamped flat at the anchor value everywhere.
    assert!(
        anchored[last] - anchored[0] > 2.0,
        "the anchored smooth must rise from the boundary pin toward the interior \
         data (pin={}, right={})",
        anchored[0],
        anchored[last]
    );
}
