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
//! So the BUDGET is nearly flat in `n`: between `n = 1_000` and `n = 16_000` the
//! width rises by `ln(16000)/ln(1000) = 1.40`, which is `log2(1.40) = 0.49` of a
//! level on a depth of about thirty, under 2%.
//!
//! # The count is not the budget, and it is not flat (measured 2026-09-21)
//!
//! This probe's first run refuted the paragraph that used to end here, which
//! argued the COUNT must be nearly flat because the BUDGET is. Read at the sizes
//! the probe first ran, `[1_000, 4_000, 16_000]`:
//!
//! ```text
//! n =  1_000  evaluations =  591  width = 65.063  depth = 33  budget = 8712
//! n =  4_000  evaluations =  772  width = 70.607
//! n = 16_000  evaluations = 1129  width = 76.151
//! ```
//!
//! Two separate readings come out of that table, and only one of them is a
//! defect.
//!
//! The count is nowhere near the budget, so the search stops on its own
//! certificates and the budget never binds. A budget the search never reaches
//! cannot bound a count it does not spend, so the flatness of the budget says
//! nothing at all about the growth of the count. That is the reasoning error.
//!
//! And the count does grow, as `sqrt(n)`: the increments over equal 4x steps in
//! `n` are 181 and 357, a ratio of 1.972 against the 2.0 that `sqrt(n)` predicts
//! and the 1.0 that `ln n` predicts. Fitting `a + b*sqrt(n)` to the outer two
//! points gives `b = 5.67`, `a = 411.7`, which reproduces all three counts to
//! 0.2% and whose slope agrees to 1.4% across the two disjoint segments.
//!
//! Per evaluation the cost is clean `O(n)`: 7.394e-2, 2.986e-1 and 1.164 seconds
//! at the three sizes, an exponent of 0.994 over the 16x range. The fit's total
//! cost is therefore `count x per_evaluation`, about `n^1.5`, not the `O(n)` the
//! scan claims. That is #2627's cost, located.
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

/// One size's reading: what the certified search spent, and which terminal proof
/// retired the cells it spent it on. Every leaf of the subdivision tree ends on
/// one of the three, so which one grows with `n` names the mechanism.
struct SearchCost {
    evaluations: u64,
    domain_width: f64,
    elapsed: f64,
    flat_regions: usize,
    dominated_regions: usize,
    stationary_points: usize,
    /// The widest resolution any resolution-flat retirement was tested at, and
    /// the widest gap it had to clear. A cell retires flat when
    /// `min(diameter, max|D|*width) <= 2*evaluation_error`
    /// (`gam_math::score_opt::ResolutionFlatRegion`), so if the count grows
    /// because retiring got harder, the gap grows against the resolution.
    widest_flat_resolution: f64,
    widest_flat_gap: f64,
}

/// The certified search's cost at one size, driven exactly as `fit_spline_scan`
/// drives it.
fn search_evaluations(n: usize, order: usize) -> SearchCost {
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
    // The assertion above IS the refusal report (199202e195), and `build.rs`
    // bans `panic!(` and `unreachable!(`, so the result is read through the
    // `Option` the assertion has already decided. The `map_or` defaults are on a
    // branch the assertion cannot leave open.
    let searched = outcome.as_ref().ok();
    let flat = |pick: fn(&gam_math::score_opt::ResolutionFlatRegion) -> f64| -> f64 {
        searched.map_or(0.0, |result| {
            result
                .resolution_flat_regions
                .iter()
                .map(pick)
                .fold(0.0_f64, f64::max)
        })
    };
    SearchCost {
        evaluations: evaluations.get(),
        domain_width: hi - lo,
        elapsed,
        flat_regions: searched.map_or(0, |result| result.resolution_flat_regions.len()),
        dominated_regions: searched.map_or(0, |result| result.dominated_regions.len()),
        stationary_points: searched.map_or(0, |result| result.stationary_points.len()),
        widest_flat_resolution: flat(|region| region.score_resolution),
        widest_flat_gap: flat(|region| region.max_score_gap),
    }
}

/// The certified log-lambda search must keep its evaluation count inside the
/// budget that licenses it, and its terminal cells inside the tree that budget
/// can build (#2627).
///
/// # Why the old bar is gone
///
/// This test used to assert that the count does not grow with `n`, at a factor
/// of two over a 16x size range. The header records the run that refuted the
/// claim: the count grows as `a + b*sqrt(n)`, and at those sizes it grew by
/// 1.91x, which the 2x allowance admitted by an accident of where the sizes
/// fell. So the assertion passed while measuring a count that does exactly what
/// its own name says it must not do, and the next size up would have turned it
/// red for a reason that is not a regression but the law the probe itself
/// measured. A test can only fail informatively if its failure means something
/// is wrong.
///
/// # Why the new bar is not a growth law either
///
/// The `sqrt(n)` law is empirical. Nothing in `subdivision_budget`,
/// [`scan_log_lambda_domain`] or `ResolutionFlatRegion` derives it, and pinning
/// two fitted constants would assert a curve with no argument behind it. The
/// attribution printed below is what will identify the mechanism; until it does,
/// this asserts the one bound that IS derived and IS load-bearing.
///
/// That bound is the search's own contract. `maximize_score_1d` refuses with
/// `ScoreSearchError::SubdivisionBudget` once `subdivisions > 8*depth^2`, so a
/// count that reaches the budget is not a slow fit, it is a REFUSED one. Every
/// subdivision evaluates its own midpoint, so the counter here is at least the
/// subdivision count, and `evaluations < budget` therefore implies
/// `subdivisions < budget` — a conservative reading of the only derived bound in
/// sight. The dominated-region audit states the same tree from the other side: a
/// binary tree with at most `B` subdivisions has at most `B + 1` leaves, so the
/// three terminal proofs must sum to no more than that.
///
/// # What this predicts, and what would overturn it
///
/// Extrapolating the measured law against the budget, `411.7 + 5.67*sqrt(n)`
/// reaches `8712` at `n` near `2.1e6`. So the certified search is predicted to
/// stop fitting at all somewhere above two million rows, with the million-row
/// integration fixture sitting at about 70% of its budget. That is a three-point
/// fit extrapolated two decades past its range, and it is written down so it can
/// be falsified: one run at `n = 4e6` either refuses with `SubdivisionBudget` or
/// does not.
///
/// Order 2 is the cubic smoothing spline the integration fixture fits
/// (`y ~ s(x, double_penalty=false)`: degree 3, penalty order 2). The sizes are
/// a quarter of the header's, because cost is `count x O(n)` and the largest arm
/// dominates it: at the header's sizes the three arms cost about 1590 s, and at
/// these they cost about 285 s.
#[test]
fn certified_search_evaluation_count_stays_inside_its_subdivision_budget_2627() {
    const ORDER: usize = 2;
    const SIZES: [usize; 3] = [250, 1_000, 4_000];

    for size in SIZES {
        let cost = search_evaluations(size, ORDER);
        let depth = (cost.domain_width / f64::EPSILON.sqrt()).log2().ceil();
        let budget = 8.0 * depth * depth;
        let terminal = cost.flat_regions + cost.dominated_regions + cost.stationary_points;
        eprintln!(
            "[scan-eval-scaling] n={size} order={ORDER} evaluations={} \
             domain_width={:.3} depth={depth:.0} budget={budget:.0} elapsed={:.3}s \
             per_evaluation={:.3e}s | terminal cells: flat={} dominated={} stationary={} \
             total={terminal} | widest flat gap={:.6e} against resolution={:.6e}",
            cost.evaluations,
            cost.domain_width,
            cost.elapsed,
            cost.elapsed / cost.evaluations.max(1) as f64,
            cost.flat_regions,
            cost.dominated_regions,
            cost.stationary_points,
            cost.widest_flat_gap,
            cost.widest_flat_resolution,
        );

        assert!(
            (cost.evaluations as f64) < budget,
            "n={size}: the certified search spent {} evaluations against a subdivision budget \
             of {budget:.0}. At the budget the search does not return a slow fit, it refuses \
             with SubdivisionBudget, so this is the size at which the scan stops fitting at all",
            cost.evaluations
        );
        assert!(
            (terminal as f64) <= budget + 1.0,
            "n={size}: {terminal} terminal cells (flat={}, dominated={}, stationary={}) exceed \
             the {budget:.0}+1 leaves a binary tree of that many subdivisions can have, so the \
             audit and the traversal disagree about the same tree",
            cost.flat_regions,
            cost.dominated_regions,
            cost.stationary_points
        );
    }
}
