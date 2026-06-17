//! Regression test for #859: CTN cross-fit must pin a fold-invariant response
//! knot count.
//!
//! The old failure was not a CTN solver failure. The cross-fit caller resolved a
//! response knot count once, but each fold refit recomputed it from that fold's
//! response subsample. The data-driven skew/kurtosis cap then rounded to
//! different values per fold, so `p_resp` and `p1 = p_resp * p_cov` drifted.
//!
//! This test stays on that contract boundary: it builds a response vector whose
//! unpinned per-fold complements resolve to multiple knot counts, then asserts
//! the pinned config shape used by cross-fit collapses every fold to one width.

use gam::transformation_normal::{
    TransformationNormalConfig, effective_response_num_internal_knots,
};
use ndarray::Array1;
use std::collections::BTreeSet;

/// Right-skewed, heavy-tailed Stage-1 score whose fold complements straddle the
/// complexity-cap rounding boundary that caused #859.
fn skewed_score(n: usize) -> Array1<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut score = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = 2.0 * unif() - 1.0;
        let e1 = -(unif().max(1e-12)).ln();
        let e2 = -(unif().max(1e-12)).ln();
        score[i] = (e1 + e2) - 2.0 + 0.3 * xi;
    }
    score
}

fn fold_complement(response: &Array1<f64>, fold: usize, k: usize) -> Array1<f64> {
    let kept: Vec<f64> = response
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if i % k == fold { None } else { Some(v) })
        .collect();
    Array1::from_vec(kept)
}

fn fit_effective_knots(config: &TransformationNormalConfig, n: usize, p_cov: usize, response: &Array1<f64>) -> usize {
    if config.response_num_internal_knots_pinned {
        config.response_num_internal_knots
    } else {
        effective_response_num_internal_knots(config, n, p_cov, response.view())
    }
}

#[test]
fn ctn_crossfit_pins_response_knots_across_folds_859() {
    let response = skewed_score(1000);
    let config = TransformationNormalConfig::default();
    let p_cov = 10usize;
    let k = 5usize;

    let unpinned: Vec<usize> = (0..k)
        .map(|fold| {
            let complement = fold_complement(&response, fold, k);
            fit_effective_knots(&config, complement.len(), p_cov, &complement)
        })
        .collect();
    let distinct: BTreeSet<_> = unpinned.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "fixture must reproduce the old #859 drift: unpinned per-fold knot counts were {unpinned:?}"
    );

    let min_complement = (0..k)
        .map(|fold| fold_complement(&response, fold, k).len())
        .min()
        .unwrap();
    let pinned_count =
        effective_response_num_internal_knots(&config, min_complement, p_cov, response.view());
    let mut pinned_config = config.clone();
    pinned_config.response_num_internal_knots = pinned_count;
    pinned_config.response_num_internal_knots_pinned = true;

    let pinned: Vec<usize> = (0..k)
        .map(|fold| {
            let complement = fold_complement(&response, fold, k);
            fit_effective_knots(&pinned_config, complement.len(), p_cov, &complement)
        })
        .collect();
    assert!(
        pinned.iter().all(|&count| count == pinned_count),
        "#859 regression: pinned CTN fold config must use one response knot count; \
         pinned_count={pinned_count}, per-fold={pinned:?}, unpinned={unpinned:?}"
    );
}
