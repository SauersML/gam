// gam#2760 — the outer θ layout and the joint route's hyper-direction count are
// two readings of ONE decision: how many ψ coordinates a spatial term enrolls.
// The layout is sized from the resolved spec and the directions are built from
// the realizer's replay spec, so any predicate that reads a field the two specs
// disagree about splits the decision in half. `joint_null_rotation` is exactly
// such a field — the collection clears it once it composes `Q·T0` into the
// gauge, and the replay spec carries it so the local build reproduces the
// collection's block — and reading it for enrollment refused every anisotropic
// Duchon fit before its first outer evaluation:
//
//     joint hyper-gradient derivative count mismatch: psi_dim=6, hyper_dirs=1
//
// `Q` belongs in the per-axis producer, which now applies it exactly as the
// isotropic arm always has.
#[cfg(test)]
mod spatial_aniso_psi_layout_2760_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OneDimensionalBoundary,
        SpatialIdentifiability, duchon_max_active_operator_derivative_order, resolve_duchon_orders,
    };
    use gam_terms::smooth::spatial_term_uses_per_axis_psi;
    use ndarray::{Array1, Array2};

    const AXES: usize = 6;

    fn lcg_normal(state: &mut u64) -> f64 {
        let mut unit = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let u1 = unit().max(1e-300);
        let u2 = unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn aniso_duchon_spec() -> TermCollectionSpec {
        let operator_penalties = DuchonOperatorPenaltySpec::default();
        let (nullspace_order, power) = resolve_duchon_orders(
            AXES,
            DuchonNullspaceOrder::Linear,
            duchon_max_active_operator_derivative_order(&operator_penalties),
            Some(1.0),
        );
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "duchon_aniso_2760".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: (0..AXES).collect(),
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                        periodic: None,
                        length_scale: Some(1.0),
                        power: power as f64,
                        nullspace_order,
                        identifiability: SpatialIdentifiability::default(),
                        // The `--scale-dimensions` request: one log-scale per axis.
                        aniso_log_scales: Some(vec![0.0; AXES]),
                        operator_penalties,
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    #[test]
    fn the_psi_layout_and_the_hyper_directions_agree_across_the_replay_2760() {
        let n = 180usize;
        let mut state = 0x2760_A115_0001_0001u64;
        let mut x = Array2::<f64>::zeros((n, AXES));
        for i in 0..n {
            for j in 0..AXES {
                x[[i, j]] = lcg_normal(&mut state);
            }
        }
        let spec = aniso_duchon_spec();
        // Non-vacuity 1: the fixture is the enrolled per-axis case. Without
        // this the whole test passes on a term that never asks for `d` axes.
        assert!(
            spatial_term_uses_per_axis_psi(&spec, 0),
            "the fixture must enroll per-axis ψ, or it pins nothing"
        );

        let design = build_term_collection_design(x.view(), &spec).expect("collection design");
        let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");
        let spatial_terms = spatial_length_scale_term_indices(&frozen);
        let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
        assert_eq!(
            dims_per_term,
            vec![AXES],
            "the resolved spec sizes θ at one ψ coordinate per axis"
        );
        let rho_dim = design.penalties.len();
        let psi_dim: usize = dims_per_term.iter().sum();

        let mut cache = SingleBlockExactJointDesignCache::new(
            x.view(),
            frozen.clone(),
            design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("single-block cache");
        // Realize away from the fit's ψ and back, so the replay spec and design
        // are the realizer's own, not the initial collection's.
        let mut theta_away = Array1::<f64>::zeros(rho_dim + psi_dim);
        for axis in 0..psi_dim {
            theta_away[rho_dim + axis] = 0.05 * (1.0 + axis as f64);
        }
        cache.ensure_theta(&theta_away).expect("realize away");

        // Non-vacuity 2: this fixture really has a joint-null rotation, and it
        // lives in the two places the collection keeps it. Without a rotation
        // the two layouts could not disagree and the pin would be empty.
        //
        // The REPLAY SPEC carries it so the term-local build reproduces the
        // collection's block — that is the field the enrollment predicate used
        // to read, and the whole reason the two readers disagreed. The
        // COLLECTION GAUGE carries it because that is where a collection keeps
        // `Q`: `place_term_in_collection_gauge` folds `Q` into the term's chart
        // metadata and the spliced term then reports `joint_null_rotation:
        // None`, "exactly as a collection-built term reports it". So on this
        // path the per-axis producer receives `Q` already composed into the
        // frozen chart it is handed, which is why removing the predicate's
        // rotation clause is sound here; the producer's own rotation covers the
        // standalone realization, where the term does carry it separately.
        assert!(
            cache.spec().smooth_terms[0].joint_null_rotation.is_some(),
            "the replay spec must carry the collection's joint-null rotation (gam#2760)"
        );
        let gauge = cache.design().smooth.terms[0]
            .collection_gauge
            .as_ref()
            .expect("a spliced collection term keeps its gauge");
        let gauge_rotation = gauge
            .joint_null_rotation
            .as_ref()
            .expect("the fixture must have a joint-null rotation, or this pin is empty (gam#2760)");
        assert_eq!(
            gauge_rotation.rotation.nrows(),
            gauge_rotation.rotation.ncols(),
            "the absorption rotation is square in the term's local coefficients"
        );
        assert!(
            gauge_rotation.joint_nullity > 0,
            "a recorded rotation always absorbs a non-trivial joint null space"
        );
        assert!(
            cache.design().smooth.terms[0]
                .joint_null_rotation
                .is_none(),
            "a spliced collection term reports no separate rotation: the gauge folded it into \
             the chart"
        );

        // The decision itself: the same on the spec that sized θ and on the
        // spec that builds the directions.
        assert_eq!(
            spatial_dims_per_term(cache.spec(), &spatial_terms),
            dims_per_term,
            "the ψ layout must not depend on which realization path built the spec"
        );

        // And the count the joint route validates against θ.
        let info_list = try_build_spatial_log_kappa_derivativeinfo_list(
            x.view(),
            cache.spec(),
            cache.design(),
            &spatial_terms,
        )
        .expect("the per-axis ψ derivative list builds on a rotated replay")
        .expect("the joint κ route is available for an enrolled per-axis term");
        assert_eq!(
            info_list.len(),
            psi_dim,
            "one hyper direction per ψ coordinate, or the joint route refuses on the shape"
        );
    }
}
