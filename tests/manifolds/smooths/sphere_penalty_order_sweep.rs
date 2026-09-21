//! Sphere fit with each supported `penalty_order` (m=1..4). Each sweep case is
//! scored against the known truth by the across-the-function coverage
//! statistic of its own posterior, at a Bonferroni share of a stated
//! family-wise size.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::calibration::{AcrossFunctionCoverage, audit_across_function_coverage};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Family-wise upper-tail size of each sweep's coverage gates.
const FAMILY_ALPHA: f64 = 0.01;

struct SweepFit {
    rmse: f64,
    min: f64,
    max: f64,
    coverage: AcrossFunctionCoverage,
}

fn run(formula: &str, alpha: f64) -> Result<SweepFit, String> {
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -175.0 + 350.0 * (j as f64) / 14.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err("non-finite: range".to_string());
    }
    let truth: Vec<f64> = pts
        .iter()
        .map(|(lat, lon)| {
            0.5 + 0.6 * lat.to_radians().sin()
                + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
        })
        .collect();
    let error = Array1::from_shape_fn(pred.len(), |i| pred[i] - truth[i]);
    let cov = fit
        .fit
        .beta_covariance()
        .ok_or_else(|| "no coefficient covariance".to_string())?;
    let coverage = audit_across_function_coverage(
        error.view(),
        design.design.to_dense().view(),
        cov.view(),
        alpha,
    );
    let rmse = (error.dot(&error) / error.len() as f64).sqrt();
    let min = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[m-sweep] `{formula}` rmse={rmse:.4} range=[{min:.3}, {max:.3}] Q={:.3} \
         (E=1, bound={:.3}, h={:.1})",
        coverage.q, coverage.bound, coverage.dof,
    );
    Ok(SweepFit {
        rmse,
        min,
        max,
        coverage,
    })
}

fn push_coverage_failure(failures: &mut Vec<String>, m: usize, fit: &SweepFit) {
    if !fit.coverage.passes() {
        failures.push(format!(
            "m={m}: Q={:.3} > bound {:.3} (α={:.4}); rmse={:.4}",
            fit.coverage.q, fit.coverage.bound, fit.coverage.alpha, fit.rmse,
        ));
    }
}

#[test]
fn sphere_wahba_penalty_order_sweep_low_orders() {
    init_parallelism();
    // Wahba m=1, 2, 3 use closed-form kernels that fit a smooth truth
    // to RMSE ≲ 0.02 with σ=0.05 noise. m=4 (rarely used in practice)
    // has a numerical conditioning issue in the current implementation
    // where REML chooses an extremely large λ and the smooth contribution
    // collapses to zero — see the dedicated documented test below.
    //
    // m=1 carries an explicit `lmax=`. The untruncated Sobolev `K_1` is
    // log-singular at coincidence, so it has no Gram diagonal and the basis
    // builder refuses the family (#2475); `lmax=` is the shipped way to state
    // the spectral resolution a finite m=1 diagonal implies.
    let mut failures = Vec::new();
    let orders = [1usize, 2, 3];
    for m in orders {
        let formula = if m == 1 {
            "y ~ sphere(lat, lon, k=30, penalty_order=1, lmax=200)".to_string()
        } else {
            format!("y ~ sphere(lat, lon, k=30, penalty_order={m})")
        };
        match run(&formula, FAMILY_ALPHA / orders.len() as f64) {
            Ok(fit) => push_coverage_failure(&mut failures, m, &fit),
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "wahba m sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn sphere_wahba_m4_must_fit_smooth_truth() {
    // BUG TICKET: Wahba m=4 collapses the fit to a near-constant on a
    // smooth low-degree truth. The closed-form q4 polynomial in
    // `wahba_sphere_kernel_from_cos` agrees with its SIMD sibling and
    // the Gram is PSD, so the issue is not a SIMD/scalar mismatch — it
    // looks like a constant-offset / normalization error in the m=4
    // kernel form that pushes REML to a degenerate λ.
    //
    // Observed at HEAD: rmse=0.43, predictions collapse to [0.502, 0.502]
    // (the response mean) — i.e. the smooth contribution is ~0 while the
    // truth peak-to-peak is ~1.4.
    //
    // This test asserts the FIXED quality target. It will fail until
    // someone derives the correct m=4 kernel constants. Don't silence
    // it — that's the whole point of failing here.
    init_parallelism();
    let SweepFit {
        rmse,
        min: mn,
        max: mx,
        ..
    } = run("y ~ sphere(lat, lon, k=30, penalty_order=4)", FAMILY_ALPHA)
        .expect("wahba m=4 fit must succeed");
    // The other Wahba orders (m=1, 2, 3) all hit rmse ≤ 0.018 on the
    // same data. Require m=4 to be in the same ballpark — generous 5×
    // budget = 0.10.
    assert!(
        rmse <= 0.10,
        "Wahba m=4 collapsed: rmse={rmse:.4} (budget 0.10), range=[{mn:.3}, {mx:.3}]. \
         m=1,2,3 all fit at rmse ≤ 0.018 — m=4 must reach the same quality. \
         Likely cause: constant-offset / normalization in the closed-form q4 \
         polynomial in wahba_sphere_kernel_from_cos (basis.rs:13685).",
    );
}

#[test]
fn sphere_harmonic_penalty_order_sweep() {
    init_parallelism();
    let mut failures = Vec::new();
    let orders = [1usize, 2, 3, 4];
    for m in orders {
        let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree=4, penalty_order={m})");
        match run(&formula, FAMILY_ALPHA / orders.len() as f64) {
            Ok(fit) => push_coverage_failure(&mut failures, m, &fit),
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "harmonic m sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn sphere_invalid_penalty_order_rejected_cleanly() {
    init_parallelism();
    let data = make_dataset(100);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for bad_m in [0usize, 5, 10, 99] {
        let formula = format!("y ~ sphere(lat, lon, k=20, penalty_order={bad_m})");
        let r = fit_from_formula(&formula, &data, &cfg);
        match r {
            Ok(_) => panic!("m={bad_m} must be rejected (valid range is 1..=4)"),
            Err(e) => {
                // The refusal must name the option and the offending value:
                // "... penalty_order must be one of 1, 2, 3, 4; got {bad_m}".
                // Any weaker disjunct (e.g. a bare letter) is satisfied by
                // an unrelated error and cannot fail.
                let msg = e.to_string();
                assert!(
                    msg.contains("penalty_order") && msg.contains(&format!("got {bad_m}")),
                    "m={bad_m} reject must name penalty_order and the value {bad_m}; got: {e}",
                );
                eprintln!("[m-sweep] m={bad_m}: clean error: {e}");
            }
        }
    }
}
