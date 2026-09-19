// The Wood (2013) smooth test of a fit that publishes no influence matrix.
//
// The kernel's reference df is `max(tr(F_JJ)²/tr(F_JJ²), r)` over the term's
// block of the influence matrix `F = H⁻¹X'WX`, and it whitens by the term's
// block of `X'WX`. The standard lane publishes both. Every custom-family and
// survival lane (Gaussian and survival location-scale, Royston-Parmar
// survival) published neither, so its smooth table fell back to the
// truncation rank for the reference df and to the caller's unweighted `X'X`
// for the whitening. For a location-scale fit the mean block's curvature is
// `X' diag(1/σᵢ²) X`, not `X'X`, and the mean-smooth test of a covariate that
// moves only the scale rejected above its level
// (bench/pvalue_calibration/pv-multi-predictor).
//
// Both blocks are the term's own: `F_JJ = I − (V_JJ/c)·S_JJ` and
// `G_JJ = H_JJ − S_JJ`, because no other term's penalty touches `J`. The
// summary now derives them when the fit does not publish them. The exact check
// is on a standard fit: stripped of `F` and `X'WX`, it must report the same
// test as with them.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_solve::estimate::{SmoothTermSummary, smooth_term_summary_rows};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `y = sin(2πz) + ε`, with `ε ~ N(0, σ(x)²)` when `heteroscedastic`, else
/// `N(0, 0.5²)`; `x` never moves the mean.
fn dataset(seed: u64, n: usize, response_units: f64, heteroscedastic: bool) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 1.0).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let sigma = if heteroscedastic {
                (-1.0 + 1.2 * x).exp()
            } else {
                0.5
            };
            let e: f64 = noise.sample(&mut rng);
            let y = response_units * ((2.0 * std::f64::consts::PI * z).sin() + sigma * e);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

fn row<'a>(rows: &'a [SmoothTermSummary], needle: &str) -> &'a SmoothTermSummary {
    rows.iter()
        .find(|row| row.name.contains(needle))
        .unwrap_or_else(|| panic!("no smooth term named like {needle:?}"))
}

fn is_integer(value: f64) -> bool {
    (value - value.round()).abs() < 1e-9
}

#[test]
fn stripped_standard_fit_reports_the_published_smooth_test() {
    init_parallelism();
    let data = dataset(20260919, 400, 1.0, false);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let fit = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = fit else {
        panic!("expected a standard Gaussian fit");
    };
    assert!(
        fit.fit.coefficient_influence().is_some() && fit.fit.weighted_gram().is_some(),
        "the standard lane publishes F and X'WX; without them this test compares nothing"
    );
    let published = smooth_term_summary_rows(
        &fit.design,
        &fit.resolvedspec,
        &fit.fit,
        fit.fit.weighted_gram(),
    )
    .expect("published rows");

    let mut stripped = fit.fit.clone();
    let inference = stripped.inference.as_mut().expect("inference block");
    inference.coefficient_influence = None;
    inference.weighted_gram = None;
    let derived = smooth_term_summary_rows(&fit.design, &fit.resolvedspec, &stripped, None)
        .expect("derived rows");

    // Non-vacuity: Wood's reference df is not the truncation rank here.
    assert!(
        published.iter().any(|row| !is_integer(row.ref_df)),
        "every published ref_df is an integer, so the rank-only fallback would agree: {published:?}"
    );
    for (published, derived) in published.iter().zip(&derived) {
        assert_eq!(published.name, derived.name);
        let close = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1e-300),
            _ => false,
        };
        assert!(
            close(Some(published.ref_df), Some(derived.ref_df))
                && close(published.chi_sq, derived.chi_sq)
                && close(published.pvalue, derived.pvalue),
            "{}: published (ref_df {}, chi_sq {:?}, p {:?}) but derived (ref_df {}, chi_sq {:?}, p {:?})",
            published.name,
            published.ref_df,
            published.chi_sq,
            published.pvalue,
            derived.ref_df,
            derived.chi_sq,
            derived.pvalue,
        );
    }
}

#[test]
fn location_scale_mean_smooth_uses_the_wood_reference_df_in_any_response_units() {
    init_parallelism();
    let mut tables = Vec::new();
    for units in [1.0, 1000.0] {
        let data = dataset(20260920, 400, units, true);
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("s(x)".to_string()),
            ..FitConfig::default()
        };
        let fit = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("fit");
        let FitResult::GaussianLocationScale(fit) = fit else {
            panic!("expected a Gaussian location-scale fit");
        };
        let fit = fit.fit;
        assert!(
            fit.fit.coefficient_influence().is_none(),
            "the location-scale lane now publishes F; this test no longer exercises the derivation"
        );
        let rows = smooth_term_summary_rows(&fit.mean_design, &fit.meanspec_resolved, &fit.fit, None)
            .expect("mean rows");
        let z = row(&rows, "z");
        // The fitted s(z) is a wiggly sine: its EDF is fractional, so Wood's
        // tr(F)²/tr(F²) is too. The rank-only fallback is an integer.
        assert!(
            !is_integer(z.ref_df) && z.ref_df > z.edf,
            "s(z) ref_df {} (edf {}) is the truncation rank, not Wood's reference df",
            z.ref_df,
            z.edf
        );
        tables.push(rows);
    }
    // The fit reports β in the response's units; the test must not see them.
    for (unit, thousand) in tables[0].iter().zip(&tables[1]) {
        assert!(
            (unit.ref_df - thousand.ref_df).abs() <= 1e-4 * unit.ref_df,
            "{}: ref_df {} in units, {} in thousands",
            unit.name,
            unit.ref_df,
            thousand.ref_df
        );
    }
}
