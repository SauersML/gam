//! Production robustness: the structured-residual alternation must DEGRADE
//! gracefully to the pass-0 iid fit when the dictionary already explains the
//! target to numerical precision.
//!
//! Root cause (diagnosed on the #2023 tier0 primary red): the structured-residual
//! pass runs magic-by-default on every SAE fit. On a target the dictionary fits
//! near-exactly (e.g. a clean circle fit by a periodic atom), the post-dictionary
//! residual is pure convergence noise. `StructuredResidualModel::fit` had no
//! absolute floor (its idiosyncratic diagonal `D` is floored only at
//! `f64::MIN_POSITIVE`), so it built a degenerate model whose whitening `1/D` is
//! near-singular; the whitened-residual REML the outer ρ-optimizer then descends
//! is ill-conditioned with no interior stationary point, and the outer correctly
//! REFUSED to certify — a fit that should succeed instead failed with
//! "all declared solver plans exhausted".
//!
//! Fix: `sae_structured_residual_model` returns `None` (→ the alternation breaks
//! and the already-certified pass-0 iid fit is returned) when the relative
//! residual energy is below [`STRUCTURED_RESIDUAL_MIN_REL_ENERGY`]. These tests
//! pin both halves: a near-exact fit certifies with the structured pass SKIPPED,
//! and a genuinely-residual fit still RUNS the structured pass (no regression).

#[cfg(test)]
mod tests {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitReport, SaeFitRequest, SaeFitSeedReport,
        SaeFitSeedRequest, SaeMinimalSeedReport, SaeMinimalSeedRequest, SaeOuterVerdict,
        build_sae_fit_seed, build_sae_minimal_seed, run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::Array2;

    /// Eight points on the unit circle plus a global DC offset. A single periodic
    /// atom represents `[cos θ, sin θ]` (near-)exactly, so the post-dictionary
    /// residual collapses to convergence noise — the degenerate regime.
    fn circle_target(offset: f64) -> Array2<f64> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let base = [
            [1.0, 0.0],
            [s, s],
            [0.0, 1.0],
            [-s, s],
            [-1.0, 0.0],
            [-s, -s],
            [0.0, -1.0],
            [s, -s],
        ];
        Array2::from_shape_fn((8, 2), |(i, j)| base[i][j] + offset)
    }

    /// Deterministic per-cell perturbation (a small LCG hash of the index) so the
    /// dictionary can NOT explain the target exactly — the residual then carries
    /// real, above-floor covariance the structured pass must model.
    fn with_noise(mut target: Array2<f64>, sigma: f64) -> Array2<f64> {
        let (n, p) = target.dim();
        for i in 0..n {
            for j in 0..p {
                let mut s =
                    (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (j as u64).wrapping_add(1);
                s ^= s >> 33;
                s = s.wrapping_mul(0xFF51AFD7ED558CCD);
                s ^= s >> 33;
                let u = (s >> 11) as f64 / ((1u64 << 53) as f64); // [0,1)
                target[[i, j]] += sigma * (2.0 * u - 1.0);
            }
        }
        target
    }

    /// Drive the full typed primary pipeline on `target` (mirrors
    /// `examples/sae_fit.rs` / the tier0 primary test with a single periodic atom).
    /// The structured-residual alternation runs UNCONDITIONALLY inside this entry
    /// (it is not gated by `run_outer_rho_search`/`run_structure_search`), so this
    /// exercises the degeneracy guard directly.
    fn run_primary(target: Array2<f64>) -> SaeFitReport {
        let assignment_kind = SaeFitAssignmentKind::Softmax;
        let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: target.view(),
            atom_basis: vec!["periodic".to_string()],
            atom_dim: vec![1],
            assignment_kind,
            alpha: 1.0,
            tau: 1.0,
            threshold: 0.0,
            top_k: None,
            ibp_alpha_override: None,
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("minimal seed");
        let SaeMinimalSeedReport {
            atom_basis,
            effective_atom_dim,
            atom_centers,
            basis_values,
            basis_jacobian,
            basis_sizes,
            decoder_coefficients,
            smooth_penalties,
            initial_logits,
            initial_coords,
            refine_routing,
        } = minimal;

        let registry = AnalyticPenaltyRegistry::new();
        let seed = build_sae_fit_seed(SaeFitSeedRequest {
            target: target.view(),
            atom_basis: &atom_basis,
            atom_dim: &effective_atom_dim,
            atom_centers: &atom_centers,
            basis_values: basis_values.view(),
            basis_jacobian: basis_jacobian.view(),
            basis_sizes: &basis_sizes,
            decoder_coefficients: decoder_coefficients.view(),
            smooth_penalties: smooth_penalties.view(),
            initial_logits: initial_logits.view(),
            initial_coords: initial_coords.view(),
            alpha: 1.0,
            tau: 1.0,
            learnable_alpha: false,
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 4,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            native_ard_enabled: true,
            seed_refine_routing: refine_routing,
            seed_refine_random_state: 0,
            data_row_reseed: false,
            fit_config: SaeFitConfig::default(),
            temperature_schedule: None,
            fisher_metric: None,
            row_loss_weights: None,
            registry: &registry,
        })
        .expect("fit seed");
        let SaeFitSeedReport {
            base_term,
            initial_rho,
            isometry_pin_active,
            metric_provenance,
        } = seed;

        run_sae_manifold_fit(SaeFitRequest {
            base_term,
            target,
            registry,
            initial_rho,
            max_iter: 4,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            assignment_kind,
            alpha: 1.0,
            top_k: None,
            isometry_pin_active,
            metric_provenance,
            promote_from_residual: false,
            run_structure_search: false,
            run_outer_rho_search: false,
            structured_residual_passes: None,
            cancel: None,
        })
        .expect("primary fit certifies (structured pass must degrade gracefully)")
    }

    /// A near-exactly-explained target: the primary fit must CERTIFY (not refuse),
    /// and it must do so by SKIPPING the structured-residual pass (no diagnostics)
    /// — degrading to the already-certified pass-0 iid fit.
    #[test]
    fn near_exact_fit_skips_structured_pass_and_certifies() {
        let target = circle_target(7.0);
        let report = run_primary(target);
        // Reaching here means run_sae_manifold_fit returned Ok — before the floor
        // guard this panicked with the StructuredResidual outer non-certification.
        assert!(
            report.structured_residual_diagnostics.is_empty(),
            "near-exact fit must SKIP the structured-residual pass (nothing to \
             whiten); got {} pass diagnostic(s)",
            report.structured_residual_diagnostics.len()
        );
    }

    /// A target with genuine residual structure (added noise the single periodic
    /// atom cannot absorb): the structured-residual pass MUST still run — the
    /// guard must not over-trigger and suppress a real whitened refit.
    #[test]
    fn residual_bearing_fit_still_runs_structured_pass() {
        let target = with_noise(circle_target(7.0), 0.1);
        let report = run_primary(target);
        assert!(
            !report.structured_residual_diagnostics.is_empty(),
            "a fit that leaves real residual energy must RUN the structured-residual \
             pass (the degeneracy guard must not over-trigger)"
        );
        assert!(
            matches!(report.outer_termination.verdict, SaeOuterVerdict::FixedRho),
            "run_outer_rho_search=false must remain fixed-rho through structured passes"
        );
    }
}
