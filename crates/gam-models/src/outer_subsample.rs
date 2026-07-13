//! Outer-score row subsampling and the model-layer row-measure bridge.
//!
//! The canonical definitions of [`OuterScoreSubsample`], [`RowSet`], and
//! [`WeightedOuterRow`] are neutral low-layer primitives that live in the
//! `gam-problem` crate so the `CustomFamily` trait layer (and the model
//! families above it) can depend on them downward without duplication. This
//! module is the single model-layer authority that translates those primitives
//! into custom-family fit options.

pub use gam_problem::outer_subsample::*;

/// Derive the exact family-facing outer options from the spatial optimizer's
/// authoritative row measure.
///
/// This is the sole bridge from [`RowSet`] into custom-family options.  In
/// particular, `All` must clear both an inherited pilot and automatic sampling;
/// otherwise a pilot mask can survive the optimizer's full-data transition and
/// make the inner mode, objective, gradient, and Hessian describe different
/// measures.
pub(crate) fn exact_outer_options_for_row_set(
    options: &crate::custom_family::BlockwiseFitOptions,
    row_set: &RowSet,
) -> crate::custom_family::BlockwiseFitOptions {
    let mut effective = options.clone();
    effective.auto_outer_subsample = false;
    effective.outer_score_subsample = match row_set {
        RowSet::All => None,
        RowSet::Subsample { rows, n_full } => Some(std::sync::Arc::new(
            OuterScoreSubsample::from_weighted_rows(rows.as_ref().clone(), *n_full, 0),
        )),
    };
    effective
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_outer_options_bind_analytic_and_efs_lanes_to_one_row_measure() {
        let mut options = crate::custom_family::BlockwiseFitOptions::default();
        options.auto_outer_subsample = true;
        options.outer_score_subsample = Some(std::sync::Arc::new(
            OuterScoreSubsample::from_uniform_inclusion_mask(vec![9], 10, 17),
        ));
        let rows = std::sync::Arc::new(vec![
            WeightedOuterRow {
                index: 1,
                weight: 2.5,
                stratum: 3,
            },
            WeightedOuterRow {
                index: 7,
                weight: 4.0,
                stratum: 8,
            },
        ]);

        let sampled = exact_outer_options_for_row_set(
            &options,
            &RowSet::Subsample { rows, n_full: 11 },
        );
        assert!(!sampled.auto_outer_subsample);
        let installed = sampled
            .outer_score_subsample
            .as_ref()
            .expect("authoritative exact-outer row measure");
        assert_eq!(installed.n_full, 11);
        assert_eq!(installed.seed, 0);
        assert_eq!(installed.rows.len(), 2);
        assert_eq!(installed.rows[0].index, 1);
        assert_eq!(installed.rows[0].weight.to_bits(), 2.5_f64.to_bits());
        assert_eq!(installed.rows[0].stratum, 3);
        assert_eq!(installed.rows[1].index, 7);
        assert_eq!(installed.rows[1].weight.to_bits(), 4.0_f64.to_bits());
        assert_eq!(installed.rows[1].stratum, 8);

        let full = exact_outer_options_for_row_set(&options, &RowSet::All);
        assert!(!full.auto_outer_subsample);
        assert!(
            full.outer_score_subsample.is_none(),
            "full-data replay must clear every stale pilot mask"
        );
        assert!(
            options.outer_score_subsample.is_some(),
            "deriving the effective measure must not mutate caller options"
        );
    }
}
