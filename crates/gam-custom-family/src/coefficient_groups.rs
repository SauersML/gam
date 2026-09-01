//! Coefficient-group realization for the custom-family blockwise carrier:
//! resolve declared `(block, column)` group selectors into penalty pieces +
//! one tied Gamma-precision rho coordinate per group, with hierarchical
//! parent/child concatenation.
//!
//! The realizer emits each physical penalty together with its nullity, initial
//! precision, label, and optimizer-coordinate prior, then materializes every
//! public side vector from that single ordered sequence. The public realizer is
//! re-exported by the parent module.

use super::{
    CoefficientBlockSelector, CoefficientGroupSpec, CustomFamilyError, ParameterBlockSpec,
    PenaltyMatrix, RealizedCoefficientGroup, RealizedCoefficientGroupSpecs,
    penalty_label_layout_with_joint, resolved_physical_penalty_label, validate_blockspecs,
};
use ndarray::{Array1, Array2};
use std::collections::{BTreeMap, BTreeSet};

/// One physical penalty emission in its final block-local order. Keeping the
/// matrix, nullity mode, initial precision, label, and prior in one record makes
/// it impossible for a group penalty to be inserted into a block while its
/// metadata is appended to a differently ordered side vector.
#[derive(Clone)]
struct RealizedPenaltyEmission {
    penalty: PenaltyMatrix,
    nullspace_dim: Option<usize>,
    initial_log_lambda: f64,
    label: String,
    /// Prior for this penalty's optimizer coordinate. Fixed penalties have no
    /// optimizer coordinate; tied physical pieces carry the same prior.
    prior: Option<gam_problem::RhoPrior>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_spec::CoefficientGroupPrior;

    fn one_penalty_block(
        name: &str,
        initial_log_lambda: f64,
        nullspace_dims: Vec<usize>,
    ) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: crate::DesignMatrix::from(Array2::<f64>::eye(2)),
            offset: Array1::zeros(2),
            penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2))],
            nullspace_dims,
            initial_log_lambdas: Array1::from_vec(vec![initial_log_lambda]),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    #[test]
    fn multi_block_group_priors_follow_realized_penalty_order_2315() {
        let specs = vec![
            // Empty means infer every nullity in this block. Appending the group
            // must keep it empty rather than manufacturing a partial vector.
            one_penalty_block("early", -1.0, Vec::new()),
            one_penalty_block("late", -2.0, vec![0]),
        ];
        let mut group =
            CoefficientGroupSpec::new("early_group", vec![crate::coefficient_label("early", 0)])
                .with_prior(CoefficientGroupPrior::NormalLogPrecision {
                    mean: 30.0,
                    sd: 3.0,
                });
        group.initial_log_precision = Some(7.0);
        let base_prior = gam_problem::RhoPrior::Independent(vec![
            gam_problem::RhoPrior::Normal {
                mean: 10.0,
                sd: 1.0,
            },
            gam_problem::RhoPrior::Normal {
                mean: 20.0,
                sd: 2.0,
            },
        ]);

        let realized = realize_coefficient_groups_for_custom_family(
            &specs,
            std::slice::from_ref(&group),
            base_prior,
        )
        .expect("two-block coefficient-group layout must realize");

        let expected_labels = vec![
            "__block_0_penalty_0".to_string(),
            "early_group".to_string(),
            "__block_1_penalty_0".to_string(),
        ];
        assert_eq!(realized.penalty_labels, expected_labels);
        assert_eq!(realized.outer_labels, expected_labels);
        assert_eq!(realized.specs[0].penalties.len(), 2);
        assert_eq!(realized.specs[0].nullspace_dims, Vec::<usize>::new());
        assert_eq!(
            realized.specs[0].initial_log_lambdas.as_slice(),
            Some(&[-1.0, 7.0][..])
        );
        assert_eq!(realized.specs[1].penalties.len(), 1);
        assert_eq!(realized.specs[1].nullspace_dims, vec![0]);

        let gam_problem::RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("realized coefficient-group prior must be coordinate-wise")
        };
        assert_eq!(
            priors,
            &vec![
                gam_problem::RhoPrior::Normal {
                    mean: 10.0,
                    sd: 1.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 30.0,
                    sd: 3.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 20.0,
                    sd: 2.0,
                },
            ]
        );

        let layout =
            crate::penalty_label_layout_with_joint(&realized.specs, vec![2, 1], Vec::new())
                .expect("realized specs must reproduce the exported outer layout");
        assert_eq!(layout.initial_rho.as_slice(), Some(&[-1.0, 7.0, -2.0][..]));
        assert_eq!(layout.physical_to_outer, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn tied_and_fixed_base_penalties_use_optimizer_coordinate_priors_2315() {
        let mut early = one_penalty_block("early", -1.0, vec![0]);
        early.penalties = vec![
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("tied_base"),
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("tied_base"),
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_fixed_log_lambda(4.0),
        ];
        early.nullspace_dims = vec![0, 0, 0];
        early.initial_log_lambdas = Array1::from_vec(vec![-1.0, -1.0, 3.0]);
        let late = one_penalty_block("late", -2.0, vec![0]);

        let mut group =
            CoefficientGroupSpec::new("group", vec![crate::coefficient_label("early", 0)])
                .with_prior(CoefficientGroupPrior::NormalLogPrecision {
                    mean: 30.0,
                    sd: 3.0,
                });
        group.initial_log_precision = Some(7.0);
        let base_prior = gam_problem::RhoPrior::Independent(vec![
            gam_problem::RhoPrior::Normal {
                mean: 10.0,
                sd: 1.0,
            },
            gam_problem::RhoPrior::Normal {
                mean: 20.0,
                sd: 2.0,
            },
        ]);

        let realized =
            realize_coefficient_groups_for_custom_family(&[early, late], &[group], base_prior)
                .expect("base priors must follow pre-group optimizer coordinates");

        assert_eq!(
            realized.penalty_labels,
            vec![
                "tied_base".to_string(),
                "tied_base".to_string(),
                "__block_0_penalty_2".to_string(),
                "group".to_string(),
                "__block_1_penalty_0".to_string(),
            ]
        );
        assert_eq!(
            realized.outer_labels,
            vec![
                "tied_base".to_string(),
                "group".to_string(),
                "__block_1_penalty_0".to_string(),
            ]
        );
        let gam_problem::RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("realized coefficient-group prior must be coordinate-wise")
        };
        assert_eq!(
            priors,
            &vec![
                gam_problem::RhoPrior::Normal {
                    mean: 10.0,
                    sd: 1.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 30.0,
                    sd: 3.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 20.0,
                    sd: 2.0,
                },
            ]
        );

        let layout =
            crate::penalty_label_layout_with_joint(&realized.specs, vec![4, 1], Vec::new())
                .expect("realized specs must preserve tied and fixed base topology");
        assert_eq!(layout.initial_rho.as_slice(), Some(&[-1.0, 7.0, -2.0][..]));
        assert_eq!(
            layout.physical_to_outer,
            vec![Some(0), Some(0), None, Some(1), Some(2)]
        );
    }

    #[test]
    fn coefficient_group_labels_cannot_reclassify_base_penalties_2315() {
        let cases = vec![
            (
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("declared_base"),
                "declared_base",
            ),
            (
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)),
                "__block_0_penalty_0",
            ),
        ];

        for (penalty, colliding_label) in cases {
            let mut spec = one_penalty_block("base", -1.0, vec![0]);
            spec.penalties[0] = penalty;
            let group = CoefficientGroupSpec::new(
                colliding_label,
                vec![crate::coefficient_label("base", 0)],
            );
            let error = realize_coefficient_groups_for_custom_family(
                &[spec],
                &[group],
                gam_problem::RhoPrior::Flat,
            )
            .expect_err("group and base penalty labels must have distinct owners");
            assert!(
                error.to_string().contains("collides with an existing base penalty label"),
                "unexpected collision error: {error}"
            );
        }
    }
}
