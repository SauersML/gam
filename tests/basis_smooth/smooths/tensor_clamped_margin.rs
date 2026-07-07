//! Tensor-margin `bc=`/`boundary=` semantics, end-to-end through `fit_from_formula`.
//!
//! In the tensor DSL a *non-periodic* margin is spelled `clamped` (in the
//! B-spline sense of a **clamped knot vector**: the ordinary open spline that is
//! free at its two ends and does not wrap) — the direct analog of mgcv
//! `te(bs=c("cc","ps"))`. `clamped`/`open`/`natural`/`free` are therefore inert
//! non-periodic markers and must be ACCEPTED (the cylinder / torus mixed-boundary
//! tensors the manifold quality suite builds depend on it; fix 47efdebe6, unit-
//! locked by `tensor_boundary_tokens_accept_clamped_open_reject_anchored`).
//!
//! A genuine endpoint boundary condition such as `anchored` (a zero-value
//! endpoint reparameterization) has no ordinary-margin meaning on a tensor, so it
//! must NOT be silently accepted as a no-op — it is surfaced as a clean
//! unsupported-feature error rather than dropped.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

fn make_2d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y", "z"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(400);
    for i in 0..20 {
        let x = (i as f64) / 19.0;
        for j in 0..20 {
            let y = (j as f64) / 19.0;
            let z = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            rows.push(StringRecord::from(vec![
                x.to_string(),
                y.to_string(),
                z.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// The inert non-periodic markers (`clamped`/`natural`/`open`/`free`) name an
/// ordinary open tensor margin and must fit, not hard-error.
#[test]
fn tensor_with_clamped_open_margin_fits() {
    init_parallelism();
    let data = make_2d_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formulas = [
        "z ~ te(x, y, bc=['clamped', 'natural'], k=5)",
        "z ~ te(x, y, bc=['natural', 'clamped'], k=5)",
        "z ~ te(x, y, bc=['clamped', 'open'], k=5)",
    ];
    for f in &formulas {
        fit_from_formula(f, &data, &cfg).unwrap_or_else(|e| {
            panic!(
                "`{f}`: clamped/open/natural are inert non-periodic tensor-margin \
                 markers and must fit (the cylinder/torus tensor analog), got: {e}"
            )
        });
        eprintln!("[te-clamped] `{f}`: fit ok (inert non-periodic margin)");
    }
}

/// A genuine endpoint constraint such as `anchored` has no ordinary-margin
/// meaning on a tensor and must be rejected cleanly, not silently dropped.
#[test]
fn tensor_with_endpoint_bc_margin_rejected_cleanly() {
    init_parallelism();
    let data = make_2d_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formulas = [
        "z ~ te(x, y, bc=['anchored', 'natural'], k=5)",
        "z ~ te(x, y, bc=['clamped', 'anchored'], k=5)",
    ];
    for f in &formulas {
        let err = match fit_from_formula(f, &data, &cfg) {
            Ok(_) => panic!("`{f}` silently accepted unsupported tensor endpoint bc"),
            Err(err) => err,
        };
        let lower = err.to_string().to_lowercase();
        assert!(
            lower.contains("tensor") && lower.contains("bc") && lower.contains("not supported"),
            "`{f}`: rejection should explain unsupported tensor bc, got: {err}",
        );
        eprintln!("[te-clamped] `{f}`: clean error: {err}");
    }
}
