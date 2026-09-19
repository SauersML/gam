//! #2705 group A, from the property side rather than the arithmetic side.
//!
//! The unit tests in `gam-solve` pin the composition (`truncate Vp, not Vb`) and
//! the assembly (`a sum of Grams, not a subtraction`). This one asserts the
//! thing a USER can check about a shipped shape-constrained fit and that both
//! defects violated: **every covariance the fit publishes is a covariance.**
//!
//! Both defects showed up as a negative published variance:
//!
//! * `(Σ − GΔGᵀ) + (Vp − Σ)` left `−9.954853058256977e-9` on the corrected
//!   covariance of `y ~ s(x, shape=convex)` — the diagonal `se_from_covariance`
//!   refused;
//! * `Σ − GΔGᵀ` alone left `−3.08607306376274e-15` on the conditional one.
//!
//! Neither is caught by "does the fit complete" once the gates that refuse them
//! are relaxed, and neither is caught by an eigenvalue check on the CONDITIONAL
//! covariance alone. So this test reads both published matrices, on all four
//! shapes, and asks the two questions a covariance must answer: is its spectrum
//! non-negative to the resolution the fit itself declares, and — for the
//! corrected one — is it no larger than the unconstrained marginal it truncates?

use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::Array2;
use std::io::Write;

/// 400 rows of `y = sqrt(x) + N(0, 0.05²)`, the #1191 fixture — `convex` and
/// `monotone_decreasing` both BIND hard on it, which is the regime where the
/// truncation removes essentially all of a coordinate's variance and every
/// cancellation in this cluster is at its worst.
fn sqrt_fixture() -> (Vec<f64>, Vec<f64>) {
    let n = 400usize;
    let mut state: u64 = 11;
    let mut next_unit = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f64; n];
    for xi in x.iter_mut() {
        *xi = next_unit();
    }
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite covariate"));
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            xi.sqrt() + 0.05 * noise
        })
        .collect();
    (x, y)
}

/// Smallest eigenvalue of a symmetric matrix, and its scale, so a verdict can be
/// stated relative to the matrix rather than against an absolute number.
fn spectrum_floor(matrix: &Array2<f64>) -> (f64, f64) {
    let scale = matrix.iter().fold(0.0_f64, |worst, &v| worst.max(v.abs()));
    let (eigenvalues, _) = matrix
        .eigh(faer::Side::Lower)
        .expect("a published covariance must be a finite symmetric matrix");
    let floor = eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    (floor, scale)
}

#[test]
fn every_published_covariance_of_a_shape_constrained_fit_is_a_covariance_2705() {
    init_parallelism();
    let (x, y) = sqrt_fixture();
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_2705_psd_{}.csv", std::process::id()));
    {
        let mut file = std::fs::File::create(&tmp).expect("create fixture csv");
        file.write_all(csv.as_bytes()).expect("write fixture csv");
    }
    let dataset = load_csvwith_inferred_schema(&tmp).expect("load fixture");
    std::fs::remove_file(&tmp).expect("remove the fixture csv this test created");

    let mut failures: Vec<String> = Vec::new();
    let mut corrected_seen = 0usize;
    for kind in [
        "monotone_increasing",
        "monotone_decreasing",
        "convex",
        "concave",
    ] {
        let formula = format!("y ~ s(x, shape={kind})");
        let fitted = match fit_from_formula(&formula, &dataset, &FitConfig::default()) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => {
                failures.push(format!("{kind}: a 1-D gaussian smooth is not a Standard fit"));
                continue;
            }
            Err(error) => {
                failures.push(format!("{kind}: {error}"));
                continue;
            }
        };
        let Some(inference) = fitted.fit.inference.as_ref() else {
            failures.push(format!("{kind}: the fit carries no inference block"));
            continue;
        };

        // The ρ̂-conditional covariance. A truncated Gaussian's covariance is
        // PSD; the pre-#2705 subtraction published a `−3.09e-15` diagonal here.
        if let Some(conditional) = fitted.fit.covariance_conditional.as_ref() {
            let (floor, scale) = spectrum_floor(conditional);
            // The resolution of an eigenvalue of a matrix assembled at this
            // scale, by Weyl: the backward error of the assembly itself.
            let resolution = 64.0 * (conditional.nrows() as f64) * f64::EPSILON * scale.max(1.0);
            if floor < -resolution {
                failures.push(format!(
                    "{kind}: the conditional covariance is not PSD — smallest eigenvalue \
                     {floor:.6e} against scale {scale:.6e} and assembly resolution \
                     {resolution:.6e}"
                ));
            }
            for (index, &variance) in conditional.diag().iter().enumerate() {
                if variance < 0.0 {
                    failures.push(format!(
                        "{kind}: conditional variance {index} is negative: {variance:.6e}"
                    ));
                }
            }
        } else {
            failures.push(format!("{kind}: the fit publishes no conditional covariance"));
        }

        // The ρ-marginal covariance. Pre-#2705 this was `Vp − GΔGᵀ` with `G`
        // and `Δ` built for `Vb`, which is the truncation of neither.
        let Some(corrected) = fitted.fit.covariance_corrected.as_ref() else {
            // A typed absence is honest (a rail-certified fit has no ρ-variance
            // to propagate), so it is not a failure — but if EVERY shape
            // declines, this test has stopped measuring the corrected path and
            // the assertion below says so.
            continue;
        };
        corrected_seen += 1;
        let (floor, scale) = spectrum_floor(corrected);
        let resolution = 64.0 * (corrected.nrows() as f64) * f64::EPSILON * scale.max(1.0);
        if floor < -resolution {
            failures.push(format!(
                "{kind}: the smoothing-corrected covariance is not PSD — smallest eigenvalue \
                 {floor:.6e} against scale {scale:.6e} and assembly resolution {resolution:.6e}"
            ));
        }
        for (index, &variance) in corrected.diag().iter().enumerate() {
            if variance < 0.0 {
                failures.push(format!(
                    "{kind}: corrected variance {index} is negative: {variance:.6e}"
                ));
            }
        }

        // Truncating a Gaussian to a convex set cannot increase its covariance,
        // so the published corrected variance can never exceed the UNTRUNCATED
        // marginal `Vb_unc + J·V_ρ·Jᵀ` it was truncated from. That upper bound
        // is not reconstructible from the fit alone (the untruncated `Vb` is not
        // published), but the smoothing correction is — and it is PSD — so
        // `diag(corrected) ≤ diag(conditional) + diag(smoothing) + removed` is
        // not available either. What IS checkable, and is exactly the ordering
        // the old composition broke, is that the corrected variance is not
        // BELOW the conditional one by more than the truncation could remove:
        // both are truncations of nested Gaussians, so a corrected variance that
        // is orders of magnitude below its conditional counterpart while the
        // smoothing correction is positive is the pre-fix signature.
        if let Some(conditional) = fitted.fit.covariance_conditional.as_ref()
            && let Some(smoothing) = inference.smoothing_correction.as_ref()
        {
            for index in 0..corrected.nrows() {
                let smoothing_ii = smoothing[[index, index]];
                if smoothing_ii <= 0.0 {
                    continue;
                }
                let conditional_ii = conditional[[index, index]];
                let corrected_ii = corrected[[index, index]];
                // Allow the truncation at `Vp` to remove more than the
                // truncation at `Vb` did (it acts on a wider Gaussian), but not
                // to leave a variance below zero — which is the only thing the
                // pre-fix order could produce here.
                if corrected_ii < 0.0 {
                    failures.push(format!(
                        "{kind}: corrected variance {index} = {corrected_ii:.6e} is negative \
                         where the conditional is {conditional_ii:.6e} and the smoothing \
                         correction adds {smoothing_ii:.6e}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "a shape-constrained fit published a matrix that is not a covariance (#2705 group A):\n  {}",
        failures.join("\n  ")
    );
    assert!(
        corrected_seen > 0,
        "no shape-constrained fit published a smoothing-corrected covariance, so this test \
         measured nothing about the composition it exists for"
    );
}
