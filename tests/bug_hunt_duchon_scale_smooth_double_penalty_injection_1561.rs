//! Permanent guard for the #1561 follow-up: the null-space double-penalty
//! default-OFF applied to secondary (scale / distributional) smooths must NOT
//! be injected into a Duchon smooth.
//!
//! #1561 defaults the Marra & Wood null-space "double" penalty OFF for a
//! secondary-predictor smooth (it otherwise over-shrinks the scale surface
//! toward homoscedasticity). The default flip is implemented by injecting
//! `double_penalty=false` into the smooth's option map when the user did not set
//! it. Duchon smooths, however, carry no such penalty (they ship their own
//! reproducing-norm penalty plus a null-space ridge) and their builder REJECTS
//! any `double_penalty=` key outright with an `incompatible_config` error. So a
//! scale/distributional formula using `duchon(...)` would abort at
//! materialization ("Duchon smooth ... does not support double_penalty") once
//! the parsimony pass injected the key — an otherwise-valid fit turned into a
//! hard error. The fix excludes `duchon` from the injection; this test pins that
//! a Gaussian location-scale fit whose SCALE block uses a Duchon smooth
//! completes instead of aborting.

use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Heteroscedastic Gaussian data: mean and log-sigma both vary smoothly in `x`,
/// so the scale block has a genuine surface to model (not the homoscedastic
/// null space) and the Duchon scale smooth is actually exercised.
fn locscale_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state = 20_260_701_u64;
    let mut x: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let mut y = Vec::with_capacity(n);
    for &xi in &x {
        // Box-Muller standard normal.
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
        let mean = (two_pi * xi).sin();
        let sigma = 0.1 + 0.2 * (1.0 + (two_pi * xi).sin());
        y.push(mean + sigma * z);
    }
    (x, y)
}

/// Fit a Gaussian location-scale model whose scale (noise) block is
/// `scale_formula`. Returns the fit result or the error message text.
fn fit_scale_smooth(scale_formula: &str) -> Result<FitResult, String> {
    let (x, y) = locscale_data(240);
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode toy locscale data");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(scale_formula.to_string()),
        ..FitConfig::default()
    };
    fit_from_formula("y ~ s(x)", &ds, &cfg).map_err(|e| e.to_string())
}

#[test]
fn duchon_scale_smooth_does_not_abort_on_1561_double_penalty_injection() {
    // Before the fix: `apply_secondary_predictor_basis_parsimony` injected
    // `double_penalty=false` into this Duchon scale smooth, and the Duchon
    // builder rejected the key, so the fit returned
    // `Err("Duchon smooth 'x' does not support double_penalty ...")`.
    match fit_scale_smooth("duchon(x)") {
        Ok(FitResult::GaussianLocationScale(_)) => {}
        Ok(_) => panic!(
            "expected a Gaussian location-scale fit for a scale-block Duchon smooth, \
             got a different FitResult variant"
        ),
        Err(e) => {
            assert!(
                !e.contains("double_penalty"),
                "#1561 regression: scale-block Duchon fit aborted on an injected \
                 double_penalty key: {e}"
            );
            panic!("scale-block Duchon fit failed for an unrelated reason: {e}");
        }
    }
}

#[test]
fn explicit_double_penalty_on_duchon_scale_smooth_is_still_rejected() {
    // Sibling guard: the #1561 fix only stops the secondary-smooth parsimony pass
    // from INJECTING `double_penalty=false` into a Duchon smooth — it does not
    // loosen the Duchon builder itself. An EXPLICIT user `double_penalty=` on a
    // Duchon smooth must still be rejected, so the exclusion is narrow (drop the
    // silent injection) rather than a blanket "Duchon now accepts double_penalty".
    match fit_scale_smooth("duchon(x, double_penalty=true)") {
        Err(e) => assert!(
            e.contains("double_penalty"),
            "expected the Duchon `double_penalty` rejection, got a different error: {e}"
        ),
        Ok(_) => {
            panic!("a Duchon smooth must still reject an explicit `double_penalty=` key")
        }
    }
}
