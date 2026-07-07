// #1270 / #1274 — the single-spatial-term n-free penalty re-key (the #1033
// κ-optimizer fast path) must reproduce the Matérn design's ACTUAL penalty
// topology at every trial length-scale ψ.
//
// Root cause #1270 pinned: the realized Matérn design always overrides the
// kernel double-penalty with the canonical `{mass, tension, stiffness}`
// operator triplet (`build_inner_smooth_basis` →
// `matern_operator_penalty_triplet_from_metadata`). For ν=5/2, d=2 that is 3
// penalty blocks. The old re-key rebuilt `S(ψ)` through the kernel
// double-penalty path instead, which yields a SINGLE block, so every skip-path
// eval staged a 1-block surface against a 3-block frozen design, tripped
// "penalty topology is not ψ-stable", cleared the stage and then hard-errored
// "no exact S(psi) was staged" — aborting `matern(x1, x2)` before any optimizer
// iteration ran. The fix routes BOTH the design builder and the re-key through
// one shared length-scale-parameterized triplet builder
// (`matern_operator_penalty_triplet_at_length_scale`).
//
// These tests pin both halves of the #1274 contract: the n-free penalty re-key
// machinery is byte/topology-exact for Matérn across ψ, and a single frozen
// Matérn term is admitted by `supports_nfree_penalty_rekey` so the κ-loop may
// use the n-free re-key instead of O(n) design re-realization.

#[cfg(test)]
mod matern_nfree_rekey_topology_tests {
    use super::*;
    use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use gam_terms::smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionDesign, TermCollectionSpec,
        matern_operator_penalty_triplet_from_metadata, set_spatial_length_scale,
    };
    use ndarray::Array2;

    fn matern_2d_dataset(n: usize) -> Array2<f64> {
        // Deterministic low-discrepancy-ish 2-D cloud (no RNG dependency): a
        // golden-ratio lattice over [0, 1]^2. Standardization (σ_geom ≠ 1) is
        // exercised because the per-axis spreads differ slightly.
        let mut data = Array2::<f64>::zeros((n, 2));
        let phi = 0.618_033_988_749_894_9_f64;
        for i in 0..n {
            let a = (i as f64 + 0.5) / n as f64;
            let b = ((i as f64 + 1.0) * phi).fract();
            data[[i, 0]] = a;
            data[[i, 1]] = 0.85 * b + 0.05 * a;
        }
        data
    }

    fn matern_2d_spec(length_scale: f64) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::EqualMass { num_centers: 24 },
                        length_scale,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        // The user default for `matern(x1, x2)` — and the exact
                        // configuration #1270 aborts on.
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                        nullspace_shrinkage_survived: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    /// Build the frozen (resolved-spec, design) pair the κ-optimizer's
    /// incremental realizer consumes, for `matern(x1, x2)` at `length_scale`.
    fn frozen_matern_2d(
        data: ndarray::ArrayView2<'_, f64>,
        length_scale: f64,
    ) -> (TermCollectionSpec, TermCollectionDesign) {
        let spec = matern_2d_spec(length_scale);
        let base = build_term_collection_design(data, &spec).expect("base matern 2d design");
        let resolved =
            freeze_term_collection_from_design(&spec, &base).expect("freeze matern 2d spec");
        let design =
            build_term_collection_design(data, &resolved).expect("frozen matern 2d design");
        (resolved, design)
    }

    #[test]
    fn matern_2d_nfree_rekey_preserves_penalty_topology_across_psi() {
        let data = matern_2d_dataset(360);
        let length_scale = 0.30;
        let (resolved, design) = frozen_matern_2d(data.view(), length_scale);

        // ν=5/2, d=2 ⇒ Sobolev order m = ν + d/2 = 3.5 > 2, so all three operator
        // dials (mass, tension, stiffness) are active: the design carries 3 blocks.
        let frozen_blocks = design.penalties.len();
        assert!(
            frozen_blocks >= 3,
            "expected the ν=5/2 d=2 Matérn design to ship the 3-block operator triplet; got {frozen_blocks}"
        );

        let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
            data.view(),
            resolved,
            design.clone(),
        )
        .expect("build incremental realizer");
        let spatial_terms = vec![0usize];

        // The single ψ coordinate is log-κ = -log(length_scale). Sweep a wide band
        // around the seed: the re-key must return the SAME number of blocks as the
        // frozen design at EVERY ψ, never the 1-block kernel surface (#1270).
        let psi0 = -length_scale.ln();
        for step in [-2.0_f64, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0] {
            let psi = psi0 + step;
            let (penalties, nullspace_dims) = realizer
                .canonical_penalties_at_psi(&spatial_terms, &[psi])
                .unwrap_or_else(|e| panic!("re-key must succeed at psi={psi} (step {step}): {e}"));
            assert_eq!(
                penalties.len(),
                frozen_blocks,
                "n-free re-key produced {} blocks at psi={psi} but the frozen design carries {} \
                 — penalty topology must be ψ-stable (#1270)",
                penalties.len(),
                frozen_blocks
            );
            assert_eq!(
                nullspace_dims.len(),
                frozen_blocks,
                "nullspace-dim vector length must track the block count at psi={psi}"
            );
        }
    }

    #[test]
    fn matern_2d_nfree_penalty_rekey_is_byte_exact_and_design_skip_is_admitted_1274() {
        // #1274 contract: the n-free operator-triplet penalty re-key for Matérn
        // is byte/topology-IDENTICAL to the slow-path realized design across
        // the ψ window, and the support predicate admits that exact re-key so
        // the κ-loop is not forced through an O(n) design realization merely to
        // obtain the same penalty surface.
        let data = matern_2d_dataset(360);
        let seed_length_scale = 0.30;
        let (resolved, design) = frozen_matern_2d(data.view(), seed_length_scale);

        // ν=5/2, d=2 ⇒ m = ν + d/2 = 3.5 > 2: all three operator dials active.
        let frozen_blocks = design.penalties.len();
        assert!(
            frozen_blocks >= 3,
            "expected the ν=5/2 d=2 Matérn design to ship the 3-block operator triplet; got {frozen_blocks}"
        );

        let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
            data.view(),
            resolved.clone(),
            design.clone(),
        )
        .expect("build incremental realizer");
        let spatial_terms = vec![0usize];

        assert!(
            realizer.supports_nfree_penalty_rekey(&spatial_terms),
            "a single frozen Matérn term must be admitted to the n-free penalty re-key once \
             the re-key rebuilds the realized operator-triplet topology exactly (#1274)"
        );

        // The penalty re-key itself is byte/topology-exact across ψ:
        // same block count, same nullspace dims, entrywise-equal to the
        // slow-path realized operator triplet at every trial length scale.
        let psi0 = -seed_length_scale.ln();
        let mut worst_diff = 0.0_f64;
        for step in [-1.0_f64, -0.4, 0.0, 0.4, 1.0] {
            let psi = psi0 + step;
            let trial_length_scale = (-psi).exp();

            // Slow path: rebuild the frozen design with only the length scale
            // moved, then derive its operator triplet exactly as a cold κ trial
            // would.
            let mut trial_spec = resolved.clone();
            set_spatial_length_scale(&mut trial_spec, 0, trial_length_scale)
                .expect("set trial length scale");
            let trial_design = build_term_collection_design(data.view(), &trial_spec)
                .expect("slow-path trial design");
            let truth_metadata = &trial_design.smooth.terms[0].metadata;
            let (truth_locals, truth_nulldims, _info) =
                matern_operator_penalty_triplet_from_metadata(truth_metadata)
                    .expect("slow-path operator triplet");

            // Fast path: the n-free re-key at the same ψ.
            let (rekey, rekey_nulldims) = realizer
                .canonical_penalties_at_psi(&spatial_terms, &[psi])
                .unwrap_or_else(|e| panic!("re-key must succeed at psi={psi} (step {step}): {e}"));

            assert_eq!(
                rekey.len(),
                truth_locals.len(),
                "psi={psi}: re-key block count {} != slow-path {} — topology must be identical",
                rekey.len(),
                truth_locals.len()
            );
            assert_eq!(
                rekey.len(),
                frozen_blocks,
                "psi={psi}: re-key block count {} != frozen design {frozen_blocks} (ψ-stable triplet)",
                rekey.len()
            );
            assert_eq!(
                rekey_nulldims, truth_nulldims,
                "psi={psi}: nullspace dims must match the slow path"
            );

            for (k, (rk, truth)) in rekey.iter().zip(truth_locals.iter()).enumerate() {
                assert_eq!(
                    rk.local.dim(),
                    truth.dim(),
                    "psi={psi} block {k}: re-key local shape {:?} != slow-path {:?}",
                    rk.local.dim(),
                    truth.dim()
                );
                let max_abs_diff = rk
                    .local
                    .iter()
                    .zip(truth.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                worst_diff = worst_diff.max(max_abs_diff);
                assert!(
                    max_abs_diff < 1e-10,
                    "psi={psi} block {k}: n-free re-key deviates from the slow-path realized \
                     Matérn penalty by {max_abs_diff:.3e} — must be byte/topology-identical (#1274)"
                );
            }
        }
        eprintln!("[#1274] max entrywise re-key vs slow-path diff across ψ window = {worst_diff:.3e}");
    }

    #[test]
    fn matern_2d_nfree_rekey_matches_frozen_design_at_seed() {
        // The strongest exactness contract: at the SEED ψ the design was built
        // at, the n-free re-key must reconstruct the frozen design's own
        // canonical penalty surface byte-for-byte — otherwise the κ fast path
        // would silently pair the inner solve with a different S than the slow
        // path uses.
        let data = matern_2d_dataset(360);
        let seed_length_scale = 0.30;
        let (resolved, design) = frozen_matern_2d(data.view(), seed_length_scale);

        // The frozen design's canonical penalty surface, reconstructed through
        // the SAME `canonicalize_penalty_specs` pipeline the re-key uses, so a
        // value mismatch reflects a real numeric divergence and not a
        // representational one. The single spatial term owns the whole penalty
        // list.
        let p_total = design.design.ncols();
        let truth_specs: Vec<gam_solve::estimate::PenaltySpec> = design
            .penalties
            .iter()
            .map(|b| gam_solve::estimate::PenaltySpec::Block {
                local: b.local.clone(),
                col_range: b.col_range.clone(),
                prior_mean: b.prior_mean.clone(),
                structure_hint: b.structure_hint.clone(),
                op: b.op.clone(),
            })
            .collect();
        let truth_nulldims: Vec<usize> = design.nullspace_dims.clone();
        let (truth, _truth_nd) = gam_terms::construction::canonicalize_penalty_specs(
            &truth_specs,
            &truth_nulldims,
            p_total,
            "frozen-design-truth",
        )
        .expect("canonicalize frozen design penalties");

        let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
            data.view(),
            resolved,
            design.clone(),
        )
        .expect("build incremental realizer");
        let spatial_terms = vec![0usize];

        let psi_seed = -seed_length_scale.ln();
        let (rekey, _rekey_nd) = realizer
            .canonical_penalties_at_psi(&spatial_terms, &[psi_seed])
            .expect("re-key at seed psi");

        assert_eq!(
            rekey.len(),
            truth.len(),
            "re-key block count {} disagrees with the frozen design {} at the seed length scale",
            rekey.len(),
            truth.len()
        );
        for (k, (rk, tr)) in rekey.iter().zip(truth.iter()).enumerate() {
            assert_eq!(
                rk.local.dim(),
                tr.local.dim(),
                "block {k}: re-key local shape {:?} != frozen design {:?}",
                rk.local.dim(),
                tr.local.dim()
            );
            let max_abs_diff = rk
                .local
                .iter()
                .zip(tr.local.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs_diff < 1e-10,
                "block {k}: n-free re-key local penalty deviates from the frozen design by \
                 {max_abs_diff:.3e} at the seed ψ (must be byte-identical so the κ fast path and \
                 slow path agree)"
            );
        }
    }

    #[test]
    fn matern_2d_nfree_rekey_matches_slow_rebuild_at_trial_psi() {
        // Re-key at a DIFFERENT length scale than the seed must equal a full
        // slow-path rebuild at that scale (centers + identifiability transform
        // stay frozen; only ℓ moves). This pins the σ_geom length-scale
        // compensation (#706): the ψ-decoded original-coord scale and the
        // metadata-stored scale must compensate to the same effective scale.
        let data = matern_2d_dataset(360);
        let seed_length_scale = 0.30;
        let (resolved, design) = frozen_matern_2d(data.view(), seed_length_scale);
        let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
            data.view(),
            resolved.clone(),
            design,
        )
        .expect("build incremental realizer");
        let spatial_terms = vec![0usize];

        // Ground truth at a different length scale: rebuild the frozen design
        // with only the length scale moved. Its `{mass, tension, stiffness}`
        // triplet is exactly what a slow-path κ trial would pair with the inner
        // solve.
        let trial_length_scale = seed_length_scale * 0.7;
        let mut trial_spec = resolved.clone();
        set_spatial_length_scale(&mut trial_spec, 0, trial_length_scale)
            .expect("set trial length scale");
        let trial_design = build_term_collection_design(data.view(), &trial_spec)
            .expect("slow-path trial design");
        let truth_metadata = &trial_design.smooth.terms[0].metadata;
        let (truth_locals, truth_nulldims, _info) =
            matern_operator_penalty_triplet_from_metadata(truth_metadata)
                .expect("slow-path operator triplet");

        let psi_trial = -trial_length_scale.ln();
        let (rekey, rekey_nulldims) = realizer
            .canonical_penalties_at_psi(&spatial_terms, &[psi_trial])
            .expect("re-key at trial psi");

        assert_eq!(
            rekey.len(),
            truth_locals.len(),
            "re-key block count {} disagrees with the slow-path rebuild {} at the trial length scale",
            rekey.len(),
            truth_locals.len()
        );
        assert_eq!(
            rekey_nulldims, truth_nulldims,
            "nullspace dims must match the slow path"
        );

        for (k, (rk, truth)) in rekey.iter().zip(truth_locals.iter()).enumerate() {
            assert_eq!(
                rk.local.dim(),
                truth.dim(),
                "block {k}: re-key local shape {:?} != slow-path {:?}",
                rk.local.dim(),
                truth.dim()
            );
            let max_abs_diff = rk
                .local
                .iter()
                .zip(truth.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs_diff < 1e-10,
                "block {k}: n-free re-key local penalty deviates from the slow-path rebuild by \
                 {max_abs_diff:.3e} (must be byte-identical so the κ fast path and slow path agree)"
            );
        }
    }
}
