use super::*;
use crate::pirls::PirlsWorkspace;
use crate::types::LinkFunction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GeometryBackendKind {
    DenseSpectral,
    SparseExactSpd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HessianEvalStrategyKind {
    SpectralExact,
    DiagnosticNumeric,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HessianStrategyDecision {
    pub(super) strategy: HessianEvalStrategyKind,
    pub(super) reason: &'static str,
}

impl<'a> RemlState<'a> {
    pub(super) fn selecthessian_strategy_policy(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> HessianStrategyDecision {
        if self.usesobjective_consistentfdgradient(rho) {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::DiagnosticNumeric,
                reason: "objective_consistent_numericgradient",
            };
        }
        // When the sparse-exact backend produced the PIRLS result, prefer
        // the sparse Hessian path for consistency (avoids dense→sparse
        // round-trip that loses sparsity structure).
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::SpectralExact,
                reason: "sparse_exact_backend_consistency",
            };
        }
        HessianStrategyDecision {
            strategy: HessianEvalStrategyKind::SpectralExact,
            reason: "exact_preferred",
        }
    }

    pub(super) fn select_reml_geometry(&self, rho: &Array1<f64>) -> SparseRemlDecision {
        let p = self.p;
        let has_dense_constraints =
            self.linear_constraints.is_some() || self.coefficient_lower_bounds.is_some();
        let x_sparse = self.x.as_sparse();
        let nnz_x = x_sparse.map(|s| s.val().len()).unwrap_or(0);
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

        if self.config.firth_bias_reduction
            && !matches!(self.config.link_function(), LinkFunction::Logit)
        {
            return dense_backend("firth_non_logit", None, None);
        }
        if p < 256 {
            return dense_backend("p_below_threshold", None, None);
        }
        if has_dense_constraints {
            return dense_backend("constraints_present", None, None);
        }
        let Some(x_sparse) = x_sparse else {
            return dense_backend("design_not_sparse", None, None);
        };
        let Some(blocks) = self.sparse_penalty_blocks.as_ref() else {
            return dense_backend("penalty_blocks_not_separable", None, None);
        };

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        match workspace.sparse_penalized_system_stats(x_sparse, &s_lambda) {
            Ok(stats) if stats.density_upper < 0.10 && !blocks.is_empty() => SparseRemlDecision {
                geometry: RemlGeometry::SparseExactSpd,
                reason: "sparse_exact_spd",
                p,
                nnz_x,
                nnz_h_upper_est: Some(stats.nnz_h_upper),
                density_h_upper_est: Some(stats.density_upper),
            },
            Ok(stats) => dense_backend(
                "penalized_hessian_too_dense",
                Some(stats.nnz_h_upper),
                Some(stats.density_upper),
            ),
            Err(_) => dense_backend("sparse_stats_failed", None, None),
        }
    }
}
