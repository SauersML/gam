//! gam#2929 item 2: a SAVED local-empirical latent law rebuilds its training
//! rows' mixtures from their conditioning covariates.
//!
//! `LatentMeasureKind::LocalEmpirical::train_row_mixtures` is `#[serde(skip)]`,
//! and that is not an omission: what a model file carries is the centres, the
//! context grids, the bandwidth and the mixture rule, and a row's weights are a
//! function of those and of the row's own covariates. Every replay of the
//! fitted rows — ALO for the Bernoulli family, and since this issue for the
//! survival marginal-slope family — therefore rebuilds them instead of reading
//! persisted values.
//!
//! The bar is equality with the law the fit minted, not closeness to it: the
//! rebuild composes each row's weights with the same
//! `local_empirical_mixture_for_point` the fit used, so the rebuilt law must be
//! the fitted law field for field, and every row's combined grid identical.
//! Both refusals are graded too, because a rebuild that quietly accepted the
//! wrong conditioning would replay a different law under the fitted name.
#![cfg(test)]

use super::estimated_latent_law::local_empirical_mixture_for_point;
use super::{EmpiricalZGrid, LatentMeasureKind, LocalLawMixture};
use ndarray::Array2;
use std::sync::Arc;

const TOP_K: usize = 2;
const BANDWIDTH: f64 = 0.75;
const MIXTURE: LocalLawMixture = LocalLawMixture::VanishingAtTruncation { floor: 0.05 };

/// A three-node law centred on `shift`. Distinct per context, so a row that
/// mixes two contexts differently from another has a different grid.
fn context_grid(shift: f64) -> EmpiricalZGrid {
    EmpiricalZGrid::new(
        vec![shift - 1.0, shift, shift + 1.0],
        vec![0.25, 0.5, 0.25],
        "gam#2929 fixture context law",
    )
    .expect("the fixture law has ascending nodes and positive weights summing to one")
}

fn centers() -> Vec<Vec<f64>> {
    vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]
}

/// The training rows' scaled conditioning covariates: one row on a centre, two
/// between centres, and one far from every centre so the truncation term and
/// the pooled floor both carry weight.
fn conditioning() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.9, 0.1, -0.7, 1.2, 2.4, -1.8])
        .expect("the fixture conditioning block is 4x2")
}

/// The law as the fit mints it (`LocalLawParts::into_kind`): centres, the
/// context grids with the pooled law last, and every training row's mixture
/// from the shared composer.
fn fitted_local_law() -> LatentMeasureKind {
    let centers = centers();
    let grids = vec![
        context_grid(-0.4),
        context_grid(0.0),
        context_grid(0.6),
        context_grid(0.1),
    ];
    let train_row_mixtures = conditioning()
        .rows()
        .into_iter()
        .map(|row| {
            local_empirical_mixture_for_point(&row.to_vec(), &centers, TOP_K, BANDWIDTH, MIXTURE)
                .expect("a fixture row mixes its nearest centres and the pooled law")
        })
        .collect::<Vec<_>>();
    let kind = LatentMeasureKind::LocalEmpirical {
        feature_cols: vec![2, 5],
        input_scales: Some(vec![1.0, 1.0]),
        centers,
        grids,
        top_k: TOP_K,
        bandwidth: BANDWIDTH,
        mixture: MIXTURE,
        train_row_mixtures: Arc::new(train_row_mixtures),
    };
    kind.validate("gam#2929 fixture")
        .expect("the fixture is a valid local-empirical law");
    kind
}

/// The law as a model file carries it: the same law with its training mixtures
/// dropped, which is the state every replay of a saved fit starts from.
fn saved_local_law() -> LatentMeasureKind {
    let json = serde_json::to_string(&fitted_local_law()).expect("the fitted law serializes");
    serde_json::from_str(&json).expect("the saved law deserializes")
}

/// The per-row mixtures of a local law; `None` for a law that has none, so the
/// caller states which it expected.
fn training_mixtures(kind: &LatentMeasureKind) -> Option<Arc<Vec<Vec<(usize, f64)>>>> {
    match kind {
        LatentMeasureKind::LocalEmpirical {
            train_row_mixtures, ..
        } => Some(Arc::clone(train_row_mixtures)),
        LatentMeasureKind::StandardNormal | LatentMeasureKind::GlobalEmpirical { .. } => None,
    }
}

#[test]
fn saving_a_local_law_drops_exactly_the_training_mixtures_2929() {
    let fitted = fitted_local_law();
    let saved = saved_local_law();
    assert!(
        training_mixtures(&saved)
            .expect("the saved fixture law is local-empirical")
            .is_empty(),
        "a saved local law carries no training mixtures; they are rebuilt"
    );
    assert_eq!(
        training_mixtures(&fitted)
            .expect("the fitted fixture law is local-empirical")
            .len(),
        conditioning().nrows(),
        "the fitted law carries one mixture per training row"
    );
    // Nothing else is lost: the two laws differ in that one field alone, which
    // is what makes the rebuild a reconstruction rather than a substitution.
    let restored = saved
        .with_rebuilt_training_mixtures(conditioning().view())
        .expect("the saved law rebuilds its training mixtures");
    assert_eq!(
        restored, fitted,
        "the rebuilt law is the law the fit minted, field for field"
    );
    // The saved law itself still names no per-row grid: the rebuild is what
    // supplies it, so a replay that skipped it would be refused, not silently
    // served the pooled law.
    assert!(
        saved.empirical_grid_for_training_row(0).is_err(),
        "a saved local law has no grid for a training row until its mixtures are rebuilt"
    );
}

#[test]
fn every_rebuilt_row_reads_the_grid_the_fit_read_2929() {
    let fitted = fitted_local_law();
    let rebuilt = saved_local_law()
        .with_rebuilt_training_mixtures(conditioning().view())
        .expect("the saved law rebuilds its training mixtures");
    let fitted_mixtures =
        training_mixtures(&fitted).expect("the fitted fixture law is local-empirical");
    let rebuilt_mixtures =
        training_mixtures(&rebuilt).expect("the rebuilt fixture law is local-empirical");
    assert_eq!(
        rebuilt_mixtures.len(),
        conditioning().nrows(),
        "the rebuild produces one mixture per conditioning row"
    );
    assert_eq!(rebuilt_mixtures.len(), fitted_mixtures.len());
    for row in 0..rebuilt_mixtures.len() {
        // One composer, one set of inputs: the weights are identical, not close.
        assert_eq!(
            rebuilt_mixtures[row], fitted_mixtures[row],
            "row {row} mixes the contexts the fit mixed, at the weights the fit used"
        );
        let fitted_grid = fitted
            .empirical_grid_for_training_row(row)
            .expect("the fitted law names this row's grid")
            .expect("a local law has a per-row grid");
        let rebuilt_grid = rebuilt
            .empirical_grid_for_training_row(row)
            .expect("the rebuilt law names this row's grid")
            .expect("a local law has a per-row grid");
        assert_eq!(
            rebuilt_grid.nodes, fitted_grid.nodes,
            "row {row}'s combined nodes"
        );
        assert_eq!(
            rebuilt_grid.weights, fitted_grid.weights,
            "row {row}'s combined weights"
        );
    }
    // Positive control on the fixture itself: the rows do not all read one law,
    // so the equality above is a claim the fixture is free to violate.
    let first = fitted
        .empirical_grid_for_training_row(0)
        .expect("row 0 has a grid")
        .expect("a local law has a per-row grid")
        .weights
        .clone();
    let last = fitted
        .empirical_grid_for_training_row(3)
        .expect("row 3 has a grid")
        .expect("a local law has a per-row grid")
        .weights
        .clone();
    assert_ne!(
        first, last,
        "the fixture's rows read different laws, so agreeing on them is evidence"
    );
}

#[test]
fn a_rebuild_refuses_conditioning_the_centres_cannot_be_read_against_2929() {
    let saved = saved_local_law();
    let error = saved
        .with_rebuilt_training_mixtures(Array2::<f64>::zeros((4, 3)).view())
        .expect_err("conditioning of the wrong width names no distance to a centre");
    assert!(
        error.contains("3 columns wide") && error.contains("2-dimensional"),
        "the refusal states both widths: {error}"
    );
}

#[test]
fn a_law_with_no_per_row_mixtures_is_returned_unchanged_2929() {
    // The rebuild is total over the kinds so a caller does not have to know
    // which one it holds; the two laws that have no per-row mixtures come back
    // untouched rather than reinterpreted against the conditioning.
    let global = LatentMeasureKind::GlobalEmpirical {
        grid: context_grid(0.0),
    };
    assert_eq!(
        global
            .with_rebuilt_training_mixtures(conditioning().view())
            .expect("a global law rebuilds nothing"),
        global
    );
    let standard = LatentMeasureKind::StandardNormal;
    assert_eq!(
        standard
            .with_rebuilt_training_mixtures(conditioning().view())
            .expect("the closed form rebuilds nothing"),
        standard
    );
}
