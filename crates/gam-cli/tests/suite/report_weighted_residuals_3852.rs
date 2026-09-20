//! #3852: `gam report` residual diagnostics must use the fit's prior weights.
//!
//! A weighted Gaussian fit models `Var(y_i) = σ²/w_i`, so its studentized
//! residual is `√w_i (y_i − μ̂_i) / (σ̂ √(1 − h_ii))` with the weighted
//! `σ̂² = Σ w (y − μ̂)² / (n₊ − edf)` over the `n₊` positive-weight rows. The
//! report used to drop `w` from both, so the residuals of a correct fit were a
//! scale mixture (the heavy rows collapsed toward zero, the light rows
//! inflated), and zero-weight rows — rows the fit never saw (#584) — were
//! plotted and counted in `n − edf`.
//!
//! The fixture is a correct model: `y = sin(2πx) + ε/√w` with `ε ~ N(0, 1)`
//! and weights alternating 1 and 100, plus zero-weight rows carrying a wild
//! response. Under the fix every weight group's residuals have unit variance
//! and the zero-weight rows are absent from the residual plot.

use std::fmt::Write as _;
use std::process::Command;

fn tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n")
}

/// SplitMix64: a fixed, dependency-free source for the fixture's noise.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_open(state: &mut u64) -> f64 {
    // 53 random mantissa bits, shifted off zero so `ln` is finite.
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Box–Muller standard-normal draw.
fn standard_normal(state: &mut u64) -> f64 {
    let u1 = unit_open(state);
    let u2 = unit_open(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[test]
fn weighted_gaussian_report_residuals_are_unit_scale_per_weight_group() {
    const OBSERVED: usize = 400;
    const ABSENT: usize = 8;
    let mut rng = 0x3852_u64;
    let mut csv = String::from("x,y,w\n");
    let mut weights = Vec::with_capacity(OBSERVED);
    for i in 0..OBSERVED {
        let x = (i as f64 + 0.5) / OBSERVED as f64;
        let w = if i % 2 == 0 { 1.0 } else { 100.0 };
        let y = (2.0 * std::f64::consts::PI * x).sin() + standard_normal(&mut rng) / f64::sqrt(w);
        writeln!(csv, "{x},{y},{w}").unwrap();
        weights.push(w);
    }
    for j in 0..ABSENT {
        let x = (j as f64 + 0.5) / ABSENT as f64;
        // A zero-weight row is exactly an absent row; its response is never
        // seen by the fit, so an outlandish value must leave no trace.
        writeln!(csv, "{x},1000,0").unwrap();
    }
    let data = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp data path");
    std::fs::write(data.path(), &csv).expect("write fixture");
    let model = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp model path");
    let html = tempfile::Builder::new()
        .suffix(".html")
        .tempfile()
        .expect("temp report path");

    let fit = Command::new(gam_test_support::gam_binary!())
        .arg("fit")
        .arg(data.path())
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("gaussian")
        .arg("--weights-column")
        .arg("w")
        .arg("--out")
        .arg(model.path())
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "weighted Gaussian fit failed (exit {:?}).\nstderr tail: {}",
        fit.status.code(),
        tail(&fit.stderr)
    );

    let report = Command::new(gam_test_support::gam_binary!())
        .arg("report")
        .arg(model.path())
        .arg(data.path())
        .arg(html.path())
        .output()
        .expect("spawn gam report");
    assert!(
        report.status.success(),
        "gam report failed (exit {:?}).\nstderr tail: {}",
        report.status.code(),
        tail(&report.stderr)
    );

    let page = std::fs::read_to_string(html.path()).expect("read report html");
    let marker = "Plotly.newPlot('resid_hist',[{x:";
    let start = page
        .find(marker)
        .expect("report has no residual histogram; the residuals were omitted")
        + marker.len();
    let end = start + page[start..].find(']').expect("residual array is closed") + 1;
    let residuals: Vec<f64> =
        serde_json::from_str(&page[start..end]).expect("residual array is JSON numbers");

    assert_eq!(
        residuals.len(),
        OBSERVED,
        "the {ABSENT} zero-weight rows are absent from the fit and must be absent from the \
         residual diagnostics"
    );

    // Internally studentized residuals of a linear smoother at fixed λ have
    // E[r_i²] = 1 exactly, and r_i² is approximately χ²₁, so the mean of the
    // m = 200 squared residuals in a weight group has standard deviation
    // √(2/m) = 0.1. The tolerance is five of those. Dropping the weights
    // instead puts the groups near 2 (w = 1) and 0.02 (w = 100).
    let tolerance = 5.0 * (2.0 / (OBSERVED / 2) as f64).sqrt();
    for group in [1.0, 100.0] {
        let squares: Vec<f64> = residuals
            .iter()
            .zip(&weights)
            .filter(|&(_, &w)| w == group)
            .map(|(&r, _)| r * r)
            .collect();
        let mean_square = squares.iter().sum::<f64>() / squares.len() as f64;
        assert!(
            (mean_square - 1.0).abs() <= tolerance,
            "weight-{group} rows have mean squared studentized residual {mean_square}, not 1 \
             within {tolerance}: the report residuals ignore the prior weights"
        );
    }
}
