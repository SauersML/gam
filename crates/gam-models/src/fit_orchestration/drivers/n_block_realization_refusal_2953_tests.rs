// #2953: the n-block exact-joint design cache keeps a realizer's trial refusal
// typed. README.md:201's CTN fit died at an ARC trial, ψ = 21.09, whose Matérn
// rebuild dropped its odd-order blocks. The realizer refused the trial, but
// `ExactJointDesignCache::ensure_theta` turned the error into a string and the
// n-block closures raised it again as InvalidInput, which aborted the fit instead
// of the trial (MSI job 1245518). Self-contained `#[cfg(test)] mod`.
#[cfg(test)]
mod n_block_realization_refusal_2953_tests {
    use super::*;
    use super::design_assembly_constraint_tests::two_block_exact_joint_hyper_setup;
    use gam_terms::basis::{MaternBasisSpec, MaternNu};

    #[test]
    fn a_trial_the_realizer_refuses_stays_a_refusal_through_the_n_block_cache_2953() {
        let n = 20usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (0.19 * i as f64).sin();
        }
        // The two-block cache fixture's anisotropic ν = 5/2 Matérn: each block
        // carries per-axis log-κ coordinates the cache realizes.
        let spec = |name: &str| TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: name.to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.7),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: Some(vec![0.0, 0.0]),
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let (meanspec, noisespec) = (spec("mean"), spec("noise"));
        let joint_setup = two_block_exact_joint_hyper_setup(data.view(), &meanspec, &noisespec);
        let block = |spec: &TermCollectionSpec| {
            let design = build_term_collection_design(data.view(), spec).expect("block design");
            let frozen = freeze_term_collection_from_design(spec, &design).expect("frozen block");
            let terms = spatial_length_scale_term_indices(&frozen);
            (frozen, design, terms)
        };
        let mut cache = ExactJointDesignCache::new(
            data.view(),
            vec![block(&meanspec), block(&noisespec)],
            joint_setup.rho_dim(),
            joint_setup.log_kappa_dims_per_term(),
        )
        .expect("n-block cache");
        let theta0 = joint_setup.theta0();
        cache.ensure_theta(&theta0).expect("the seed realizes");

        // ψ = 21 on every axis of the mean term: ℓ = e^{−21}, far below the
        // center spacing, where the kernel underflows between centers and the
        // odd-order collocation Grams are exactly zero, so the rebuild drops the
        // cached tension block and the realizer refuses the trial.
        let mut trial = theta0.clone();
        for axis in 0..joint_setup.log_kappa_dims_per_term()[0] {
            trial[joint_setup.rho_dim() + axis] = 21.0;
        }
        let error = cache.ensure_theta(&trial).expect_err("the trial must not realize");
        assert!(
            matches!(
                &error,
                EstimationError::TrialPointRefused { reason } if reason.contains("was dropped as")
            ),
            "the realizer's dropped-block refusal must reach the n-block caller typed, got {error:?}"
        );
    }
}
