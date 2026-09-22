//! Does the certified log-lambda search's COST grow with `n` through its
//! evaluation count, or only through the `O(n)` cost of one evaluation?
//!
//! `spline_scan_million_row_fit_scales_linearly_and_recovers_truth` did not
//! finish 100,000 rows inside 900 s on main (#2627). The scan's whole premise is
//! that a smoothing parameter costs `O(n)` per trial, so a fit is
//! `evaluations x O(n)` and only the left factor can be pathological. Nothing in
//! the search ties that factor to `n`:
//!
//! * one evaluation is [`certified_concentrated_criterion_jet`] over every
//!   pooled node, which is `O(n)` by construction;
//! * the traversal is bounded by `gam_math::score_opt::subdivision_budget`,
//!   `8*depth^2` with `depth = ceil(log2(width / resolution))`;
//! * [`scan_log_lambda_domain`] sets that width from a Gershgorin bound whose
//!   dominant term is `(2m-1)*ln(1/dx)` and a trace whose dominant term is
//!   `ln n`, so for unit weights on a unit span the width is about `2m*ln n`.
//!
//! So the budget is nearly flat in `n`: between `n = 1_000` and `n = 16_000` the
//! width rises by `ln(16000)/ln(1000) = 1.40`, which is `log2(1.40) = 0.49` of a
//! level on a depth of about thirty, under 2%. A search that certifies the same
//! criterion shape at both sizes must therefore spend about the same number of
//! evaluations at both, and the whole growth in cost must be the `O(n)` inside
//! each one.
//!
//! This probe measures that directly, through the same four functions
//! `fit_spline_scan` calls and the same `sqrt(eps)` resolution it passes, so a
//! count measured here is the count the production fit pays. It runs at sizes
//! whose product with the count stays small; it deliberately does not run at
//! `n = 100_000`, because the question is the SHAPE of the count in `n` and a
//! probe that reproduces the timeout answers nothing the timeout did not.

use super::*;

/// The integration fixture's law, at a requested size: a smooth truth on a
/// mildly irregular grid with deterministic golden-ratio noise, unit weights.
/// Same generator as
/// `tests/basis_smooth/smooths/spline_scan_workflow_equivalence.rs`, so the
/// criterion shape this probe reads is the one that test fits.
fn scaling_xyw(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let truth = |x: f64| (6.0 * x).sin() + 0.5 * x * x;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        let xi = u + 0.35 * (std::f64::consts::PI * u).sin() / (n as f64);
        let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 0.3;
        x.push(xi);
        y.push(truth(xi) + noise);
    }
    let w = vec![1.0; n];
    (x, y, w)
}

/// `(evaluations, domain width, elapsed seconds)` of the certified search at
/// one size, driven exactly as `fit_spline_scan` drives it.
fn search_evaluations(n: usize, order: usize) -> (u64, f64, f64) {
    let (x, y, w) = scaling_xyw(n);
    let (nodes, ssr_within, n_obs, _origin) = pool_nodes(&x, &y, &w, order).expect("pool nodes");
    let (lo, hi) = scan_log_lambda_domain(&nodes, order).expect("derived log-lambda domain");
    let n_nodes = nodes.len();
    let evaluations = std::cell::Cell::new(0u64);
    let endpoint_certificates =
        RefCell::new(std::collections::HashMap::<u64, CertifiedCriterionJet>::new());
    let started = std::time::Instant::now();
    let outcome = gam_math::score_opt::maximize_score_1d(
        lo,
        hi,
        f64::EPSILON.sqrt(),
        |log_lambda| {
            evaluations.set(evaluations.get() + 1);
            let certificate =
                certified_concentrated_criterion_jet(&nodes, ssr_within, n_obs, log_lambda, order)?;
            endpoint_certificates
                .borrow_mut()
                .insert(log_lambda.to_bits(), certificate);
            Ok(certificate.jet)
        },
        |left, right| {
            let certificates = endpoint_certificates.borrow();
            let left_certificate = certificates
                .get(&left.x.to_bits())
                .copied()
                .ok_or(SplineScoreProofError::MissingEndpointCertificate { log_lambda: left.x })?;
            let right_certificate = certificates.get(&right.x.to_bits()).copied().ok_or(
                SplineScoreProofError::MissingEndpointCertificate {
                    log_lambda: right.x,
                },
            )?;
            concentrated_criterion_enclosure(
                n_nodes,
                n_obs,
                left,
                right,
                left_certificate,
                right_certificate,
                order,
            )
        },
    );
    let elapsed = started.elapsed().as_secs_f64();
    assert!(
        outcome.is_ok(),
        "n={n} order={order}: the certified search refused: {}",
        outcome.as_ref().err().map(ToString::to_string).unwrap_or_default()
    );
    (evaluations.get(), hi - lo, elapsed)
}

/// The criterion-evaluation count of the certified log-lambda search must not
/// grow with `n` (#2627).
///
/// The bar is the search's own budget law. `subdivision_budget` depends on the
/// domain width and the resolution alone, and
/// [`scan_log_lambda_domain`]'s width grows like `ln n`, so over the 16x size
/// range below the admissible depth rises by `log2(ln(16000)/ln(1000)) = 0.49`
/// of one level out of about thirty. The count may therefore rise by a few
/// percent, and the factor of two asserted here is a hundred times that
/// allowance: it passes any search whose cell count tracks the domain and fails
/// one whose cell count tracks the data.
///
/// Order 2 is the cubic smoothing spline the integration fixture fits
/// (`y ~ s(x, double_penalty=false)`: degree 3, penalty order 2).
#[test]
fn certified_search_evaluation_count_does_not_grow_with_n_2627() {
    const ORDER: usize = 2;
    const SIZES: [usize; 3] = [1_000, 4_000, 16_000];

    let mut measured = Vec::with_capacity(SIZES.len());
    for size in SIZES {
        let (evaluations, width, elapsed) = search_evaluations(size, ORDER);
        let depth = (width / f64::EPSILON.sqrt()).log2().ceil();
        eprintln!(
            "[scan-eval-scaling] n={size} order={ORDER} evaluations={evaluations} \
             domain_width={width:.3} depth={depth:.0} budget={:.0} elapsed={elapsed:.3}s \
             per_evaluation={:.3e}s",
            8.0 * depth * depth,
            elapsed / evaluations.max(1) as f64,
        );
        measured.push((size, evaluations));
    }

    let (small_n, small) = measured[0];
    let (large_n, large) = measured[measured.len() - 1];
    assert!(
        large <= 2 * small,
        "the certified search spent {large} evaluations at n={large_n} against {small} at \
         n={small_n}, more than the domain's own growth admits: the count tracks the data, not \
         the criterion, so the fit's cost is super-linear in n and not the O(n) the scan claims"
    );
}
