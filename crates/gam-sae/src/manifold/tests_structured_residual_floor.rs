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
//! and the already-certified iid fit is returned) when the certified fit's
//! reconstruction loss is within the objective resolution the certified state can
//! resolve, `√(√(n·P)·ε)·|f|`. These tests pin both halves: a near-exact fit
//! certifies with the structured pass SKIPPED, and a genuinely-residual fit still
//! RUNS the structured pass (no regression).

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

    /// The guard's own measurement, in the frame the fit runs in. The Tier-0 peel
    /// centers every output column and divides it by its centered RMS `σ_c`, so each
    /// standardized column carries energy `n` and the guard's fraction is
    /// `Σ_c RSS_c / σ_c² / (n·p)`. No fixture column is anywhere near the peel's
    /// empty-column gate, so every column is standardized.
    fn guard_frame_residual_fraction(target: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
        let (n, p) = target.dim();
        let mut residual_energy = 0.0_f64;
        for c in 0..p {
            let column = target.column(c);
            let mean = column.sum() / n as f64;
            let sigma_sq = column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
            let rss: f64 = column
                .iter()
                .zip(fitted.column(c).iter())
                .map(|(t, f)| (t - f) * (t - f))
                .sum();
            residual_energy += rss / sigma_sq;
        }
        residual_energy / (n * p) as f64
    }

    /// The inner iteration budget the production front door runs (`gamfit`'s
    /// `sae_manifold_fit`, `n_iter = 50`). The fixture used to run 4. Guarded job
    /// 581184 at bde47070e split the pass-0 premise's 2.107e-9 into radial 8.29e-10
    /// and tangential 1.28e-9, so the residual was mostly unconverged coordinates.
    const PRODUCTION_INNER_ITERATIONS: usize = 50;

    /// Drive the full typed primary pipeline on `target` (mirrors
    /// `examples/sae_fit.rs` / the tier0 primary test with a single periodic atom).
    /// The structured-residual alternation runs UNCONDITIONALLY inside this entry
    /// (it is not gated by `run_outer_rho_search`/`run_structure_search`), so this
    /// exercises the degeneracy guard directly. `smoothness` is the seed's
    /// dimensionless smoothing strength, held fixed because the outer ρ search is off.
    fn run_primary(
        target: Array2<f64>,
        smoothness: f64,
        structured_residual_passes: usize,
    ) -> SaeFitReport {
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
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("minimal seed");
        let SaeMinimalSeedReport {
            geometry_plans,
            basis_values,
            basis_jacobian,
            decoder_coefficients,
            smooth_penalties,
            initial_logits,
            initial_coords,
            refine_routing,
        } = minimal;

        let registry = AnalyticPenaltyRegistry::new();
        let seed = build_sae_fit_seed(SaeFitSeedRequest {
            target: target.view(),
            geometry_plans: &geometry_plans,
            basis_values: basis_values.view(),
            basis_jacobian: basis_jacobian.view(),
            decoder_coefficients: decoder_coefficients.view(),
            smooth_penalties: smooth_penalties.view(),
            initial_logits: initial_logits.view(),
            initial_coords: initial_coords.view(),
            alpha: 1.0,
            tau: 1.0,
            learnable_alpha: false,
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness,
            max_iter: PRODUCTION_INNER_ITERATIONS,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            seed_refine_routing: refine_routing,
            seed_refine_random_state: 0,
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
            reconstruction_optimism_folds: None,
            base_term,
            target,
            registry,
            initial_rho,
            max_iter: PRODUCTION_INNER_ITERATIONS,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            alpha: 1.0,
            isometry_pin_active,
            metric_provenance,
            promote_from_residual: false,
            run_structure_search: false,
            run_outer_rho_search: false,
            structured_residual_passes,
            cancel: None,
        })
        .expect("primary fit certifies (structured pass must degrade gracefully)")
        .manifold_or_error()
        .expect("planted circle must retain a manifold atom")
    }

    /// A near-exactly-explained target: the primary fit must CERTIFY (not refuse),
    /// and it must do so by SKIPPING the structured-residual pass (no diagnostics)
    /// — degrading to the already-certified pass-0 iid fit.
    #[test]
    fn near_exact_fit_skips_structured_pass_and_certifies() {
        // #2822: an exactly representable target leaves a profiled residual of zero, which
        // Gaussian REML refuses to score by design (#2723), so the target carries a small
        // perturbation that keeps the residual resolvable.
        //
        // The guard skips when the certified fit's reconstruction loss is within the
        // objective resolution the certified state can resolve, `√(√(n·P)·ε)·|f|`. Guarded
        // job 543027 at 96b42e9e8 read a standardized residual fraction of 2.02e-9 at
        // smoothness 1 and σ = 3e-5, and job 558087 at 3aab85774 read 2.040e-9 at smoothness
        // 1e-3 and σ = 5e-6: 1000× less smoothing and 36× less perturbation energy left the
        // residual where the solve's convergence put it. The premise message still reports
        // that fraction and its radial/tangential split.
        const SMOOTHNESS: f64 = 1.0e-3;
        let target = with_noise(circle_target(7.0), 5.0e-6);
        let (n_rows, n_columns) = target.dim();
        let relative_resolution = (((n_rows * n_columns) as f64).sqrt() * f64::EPSILON).sqrt();
        // Premise, measured rather than assumed: the regime this test exists for is a
        // pass-0 fit whose reconstruction loss is already within its certified resolution.
        let pass0 = run_primary(target.clone(), SMOOTHNESS, 0);
        let pass0_resolution = relative_resolution * pass0.loss.total().abs();
        let pass0_fraction = guard_frame_residual_fraction(&target, &pass0.fitted);
        let (n, p) = target.dim();
        let mut radial_energy = 0.0_f64;
        let mut total_energy = 0.0_f64;
        let mut worst_cell = 0.0_f64;
        for i in 0..n {
            let mut along = 0.0_f64;
            let mut radius_sq = 0.0_f64;
            for c in 0..p {
                let column = target.column(c);
                let mean = column.sum() / n as f64;
                let sigma = (column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                    / n as f64)
                    .sqrt();
                let standardized = (target[[i, c]] - mean) / sigma;
                let residual = (target[[i, c]] - pass0.fitted[[i, c]]) / sigma;
                along += residual * standardized;
                radius_sq += standardized * standardized;
                total_energy += residual * residual;
                worst_cell = worst_cell.max(residual.abs());
            }
            radial_energy += along * along / radius_sq;
        }
        let cells = (n * p) as f64;
        assert!(
            pass0.loss.data_fit <= pass0_resolution,
            "premise: the pass-0 fit leaves reconstruction loss {:e} above its certified \
             objective resolution {:e} (relative resolution {:e}, penalized loss {:e}), so this \
             fixture is not in the near-exact regime the skip guard exists for, and the skip \
             assertion below would measure nothing. Standardized residual fraction {:e}; split \
             radial {:e}, tangential {:e}, worst standardized cell {:e}; pass-0 \
             log_lambda_sparse {}, log_lambda_smooth {:?}, R² {}",
            pass0.loss.data_fit,
            pass0_resolution,
            relative_resolution,
            pass0.loss.total(),
            pass0_fraction,
            radial_energy / cells,
            (total_energy - radial_energy) / cells,
            worst_cell,
            pass0.rho.log_lambda_sparse,
            pass0.rho.log_lambda_smooth,
            pass0.reconstruction_r2
        );
        let report = run_primary(target.clone(), SMOOTHNESS, 2);
        // Reaching here means run_sae_manifold_fit returned Ok — before the floor
        // guard this panicked with the StructuredResidual outer non-certification.
        assert!(
            report.structured_residual_diagnostics.is_empty(),
            "near-exact fit must SKIP the structured-residual pass (nothing to \
             whiten); got {} pass diagnostic(s) {:?}; returned reconstruction loss {:e} \
             against its certified resolution {:e}, standardized residual fraction {:e}",
            report.structured_residual_diagnostics.len(),
            report.structured_residual_diagnostics,
            report.loss.data_fit,
            relative_resolution * report.loss.total().abs(),
            guard_frame_residual_fraction(&target, &report.fitted),
        );
    }

    /// A target with genuine residual structure (added noise the single periodic
    /// atom cannot absorb): the structured-residual pass MUST still run — the
    /// guard must not over-trigger and suppress a real whitened refit.
    #[test]
    fn residual_bearing_fit_still_runs_structured_pass() {
        let target = with_noise(circle_target(7.0), 0.1);
        let report = run_primary(target, 1.0, 2);
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
