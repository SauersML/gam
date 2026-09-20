use super::*;
use crate::pirls::PirlsWorkspace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GeometryBackendKind {
    DenseSpectral,
    SparseExactSpd,
}

impl<'a> RemlState<'a> {
    /// Upper-triangle density of the penalized Hessian above which the sparse
    /// exact-SPD backend loses its advantage and we fall back to dense: once
    /// >10% of entries are nonzero, sparse factorization fill-in and bookkeeping
    /// cost more than a dense Cholesky of the same dimension.
    pub(crate) const SPARSE_HESSIAN_MAX_DENSITY: f64 = 0.10;

    pub(super) fn select_reml_geometry(
        &self,
        rho: &Array1<f64>,
    ) -> Result<SparseRemlDecision, EstimationError> {
        let lambdas =
            Array1::from_vec(gam_problem::checked_exp_log_strengths(rho.iter().copied())?);
        let p = self.p;
        let has_dense_constraints =
            self.linear_constraints.is_some() || self.coefficient_lower_bounds.is_some();
        let x_sparse = self.x.as_sparse();
        let nnz_x = x_sparse.map(|s| s.val().len());
        let dense_backend =
            |reason: &'static str,
             nnz_h_upper_est: Option<usize>,
             density_h_upper_est: Option<f64>| SparseRemlDecision {
                geometry: RemlGeometry::DenseSpectral,
                reason,
                p,
                nnz_x,
                nnz_h_upper_est,
                density_h_upper_est,
            };

        if self.config.firth_bias_reduction {
            // Route ALL Firth-active fits (Logit or otherwise) through the
            // dense-spectral backend.  The sparse bundle assembly at
            // `prepare_sparse_eval_bundlewithkey` factors
            // `X'WX + S_λ + barrier`, WITHOUT subtracting the Jeffreys
            // Hessian `H_φ`.  The dense path at
            // `prepare_dense_eval_bundlewithkey` (runtime.rs:2175-2194)
            // explicitly subtracts `H_φ` from `h_total` before caching,
            // so `bundle.h_total = X'WX + S_λ − H_φ (+ barrier)`.
            //
            // The `FirthAwareGlmDerivatives` provider returns
            // `dH/dρ = A_k + D_β(X'WX − H_φ)[v_k]`, including the
            // Firth correction term `−D(H_φ)[B_k]`.  If the sparse
            // logdet operator is factored from `X'WX + S_λ` but the
            // derivative provider differentiates `X'WX + S_λ − H_φ`,
            // then `log|H|` and `trace_logdet(dH/dρ)` live on different
            // Hessian surfaces — the REML cost and its ρ-gradient
            // would no longer be consistent (see `reml_laml_evaluate`
            // in unified.rs, where both `log|H|` and
            // `trace_logdet(dH/dρ)` come from the same `hop`, yet
            // `dH/dρ` is assembled via `effective_deriv
            // .hessian_derivative_correction_result(&neg_v_i)`).
            //
            // Subtracting `H_φ` inside the sparse factoring path would
            // require materialising `H_φ` densely (Firth H_φ is a
            // generally-dense p×p matrix — it's defined on the
            // identifiable column-space of `X`), which would destroy
            // the sparsity that justified going sparse in the first
            // place.  Routing to dense is the only cost-gradient-
            // consistent option.
            return Ok(dense_backend("firth_bias_reduction_active", None, None));
        }
        if has_dense_constraints {
            return Ok(dense_backend("constraints_present", None, None));
        }
        let Some(x_sparse) = x_sparse else {
            return Ok(dense_backend("design_not_sparse", None, None));
        };
        let block_count = self.sparse_penalty_block_count;

        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p);
        Ok(
            match workspace.sparse_penalized_system_stats(x_sparse, &s_lambda) {
                Ok(stats)
                    if stats.density_upper < Self::SPARSE_HESSIAN_MAX_DENSITY
                        && block_count > 0 =>
                {
                    SparseRemlDecision {
                        geometry: RemlGeometry::SparseExactSpd,
                        reason: "sparse_exact_spd",
                        p,
                        nnz_x,
                        nnz_h_upper_est: Some(stats.nnz_h_upper),
                        density_h_upper_est: Some(stats.density_upper),
                    }
                }
                Ok(stats) => dense_backend(
                    "penalized_hessian_too_dense",
                    Some(stats.nnz_h_upper),
                    Some(stats.density_upper),
                ),
                Err(_) => dense_backend("sparse_stats_failed", None, None),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The canonical Gaussian-identity fixture spec is owned by the REML module's
    // own test tree. `use super::*` reaches the REML module itself, which does
    // not re-export its `#[cfg(test)] mod tests` items, so the helper has to be
    // named explicitly — by relative path, because this file is `#[path]`-
    // included and therefore has no stable absolute module path.
    use super::super::tests::gaussian_identity_glm_spec;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    #[test]
    fn small_sparse_design_routes_by_penalized_hessian_structure_2413() {
        let n = 8;
        let p = 4;
        let triplets: Vec<_> = (0..n)
            .flat_map(|row| {
                (0..p)
                    .filter(move |&col| col != row % p)
                    .map(move |col| Triplet::new(row, col, 1.0))
            })
            .collect();
        let x = SparseColMat::try_new_from_triplets(n, p, &triplets)
            .expect("sparse design should build");
        let y = Array1::<f64>::zeros(n);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let config = RemlConfig::external(gaussian_identity_glm_spec(), 1e-8, false);
        let penalty = gam_terms::construction::CanonicalPenalty::from_dense_root(Array2::eye(p), p);
        let state = RemlState::newwith_offset(
            y.view(),
            x,
            weights.view(),
            offset.view(),
            vec![penalty],
            p,
            &config,
            Some(vec![0]),
            None,
            None,
        )
        .expect("REML state should build");

        let decision = state
            .select_reml_geometry(&array![0.0])
            .expect("geometry decision should succeed");

        assert!(matches!(decision.geometry, RemlGeometry::DenseSpectral));
        assert_eq!(decision.reason, "penalized_hessian_too_dense");
        assert_eq!(decision.nnz_x, Some(triplets.len()));
        assert!(
            decision
                .density_h_upper_est
                .is_some_and(|density| density > RemlState::SPARSE_HESSIAN_MAX_DENSITY),
            "the decision must come from measured Hessian structure"
        );
        assert!(decision.nnz_h_upper_est.is_some());
    }

    /// A many-level factor beside a smooth that carries two penalties on one
    /// coefficient range (`s(x)`'s wiggliness penalty and null-space ridge) is
    /// structurally sparse: its penalized Hessian is an arrow of a few dense
    /// rows over a diagonal. Shared penalty ranges change nothing about that
    /// structure, so the model must take the sparse exact route.
    #[test]
    fn shared_penalty_ranges_beside_a_many_level_factor_route_sparse_exact() {
        const LEVELS: usize = 160;
        let smooth = 1..3;
        let p = smooth.end + LEVELS;
        let n = 2 * LEVELS;
        let triplets: Vec<_> = (0..n)
            .flat_map(|row| {
                let t = (row as f64 + 0.5) / n as f64;
                [
                    Triplet::new(row, 0, 1.0),
                    Triplet::new(row, 1, t),
                    Triplet::new(row, 2, t * t),
                    Triplet::new(row, 3 + row % LEVELS, 1.0),
                ]
            })
            .collect();
        let x = SparseColMat::try_new_from_triplets(n, p, &triplets)
            .expect("sparse design should build");
        let y = Array1::<f64>::zeros(n);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let config = RemlConfig::external(gaussian_identity_glm_spec(), 1e-8, false);
        use crate::estimate::PenaltySpec;
        use gam_terms::smooth::BlockwisePenalty;
        let specs = vec![
            PenaltySpec::from_blockwise(BlockwisePenalty::new(
                smooth.clone(),
                array![[1.0, -1.0], [-1.0, 1.0]],
            )),
            PenaltySpec::from_blockwise(BlockwisePenalty::new(
                smooth.clone(),
                array![[0.5, 0.5], [0.5, 0.5]],
            )),
            PenaltySpec::from_blockwise(BlockwisePenalty::new(
                smooth.end..p,
                Array2::eye(LEVELS),
            )),
        ];
        let (canonical, _) =
            gam_terms::construction::canonicalize_penalty_specs(&specs, &[1, 1, 0], p, "test")
                .expect("canonicalize");
        let state = RemlState::newwith_offset(
            y.view(),
            x,
            weights.view(),
            offset.view(),
            canonical,
            p,
            &config,
            Some(vec![1, 1, 0]),
            None,
            None,
        )
        .expect("REML state should build");

        let decision = state
            .select_reml_geometry(&array![0.0, 0.0, 0.0])
            .expect("geometry decision should succeed");

        assert!(
            matches!(decision.geometry, RemlGeometry::SparseExactSpd),
            "shared penalty ranges must not force the dense route: {}",
            decision.basis()
        );
        assert_eq!(decision.reason, "sparse_exact_spd");
    }

    /// #2465 instance 4: the routing verdict rides on the bundle, so every
    /// emitter of the `backend=` label can print the quantities the label was
    /// decided from — and a route that decided BEFORE measuring any structure
    /// must say so rather than render a zero that reads like a measurement.
    #[test]
    fn a_routing_verdict_renders_the_quantities_it_was_decided_from_2465() {
        let measured = SparseRemlDecision {
            geometry: RemlGeometry::DenseSpectral,
            reason: "penalized_hessian_too_dense",
            p: 111,
            nnz_x: Some(35_520),
            nnz_h_upper_est: Some(4_096),
            density_h_upper_est: Some(0.6654),
        };
        let rendered = measured.basis();
        assert!(
            rendered.contains("reason=penalized_hessian_too_dense"),
            "the basis must name which route decided: {rendered}"
        );
        assert!(
            rendered.contains("density_h_est=0.6654") && rendered.contains("nnz_h_est=4096"),
            "a measured density must appear as the measurement it is: {rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "threshold={:.4}",
                RemlState::SPARSE_HESSIAN_MAX_DENSITY
            )),
            "the measurement is only falsifiable beside the threshold it was compared with: {rendered}"
        );

        let unmeasured = SparseRemlDecision {
            reason: "design_not_sparse",
            nnz_x: None,
            nnz_h_upper_est: None,
            density_h_upper_est: None,
            ..measured
        };
        let rendered = unmeasured.basis();
        assert!(
            rendered.contains("nnz_h_est=na") && rendered.contains("density_h_est=na"),
            "a route that measured no structure must report the absence, not a value: {rendered}"
        );
        assert!(
            !rendered.contains("density_h_est=0"),
            "an unmeasured density must never render as a number: {rendered}"
        );
        // `design_not_sparse` is precisely the route with no sparse
        // representation to count nonzeros from, and `nnz_x=0` reads as a
        // measured emptiness. Measured on #2569: the same fit that logged
        // `design_not_sparse … nnz_x=0` counted 583,674 design nonzeros one
        // gate later, because the design was a lazy conditioning wrapper over
        // a sparse matrix rather than a dense one.
        assert!(
            rendered.contains("nnz_x=na"),
            "a route with no sparse representation must report the absence of a \
             nonzero count, not a zero: {rendered}"
        );
        assert!(
            !rendered.contains("nnz_x=0"),
            "an uncounted design must never render as an empty one: {rendered}"
        );
    }
}
