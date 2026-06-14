//! Test-only helper exposing the internal outer objective evaluator as a
//! dense (objective, gradient, Hessian, warm-start) tuple for finite-difference
//! checks. Kept as a sibling `#[cfg(test)] mod test_support` so the test module
//! reaches it via `super::test_support::...` exactly as before.

    use super::*;
    use ndarray::{Array1, Array2};

    pub(crate) fn outerobjectivegradienthessian<F: CustomFamily + Clone + Send + Sync + 'static>(
        family: &F,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
        penalty_counts: &[usize],
        rho: &Array1<f64>,
        warm_start: Option<&ConstrainedWarmStart>,
        eval_mode: EvalMode,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
        let result = super::outerobjectivegradienthessian_internal(
            family,
            specs,
            options,
            penalty_counts,
            rho,
            warm_start,
            crate::types::RhoPrior::Flat,
            eval_mode,
        )?;
        Ok((
            result.objective,
            result.gradient,
            result.outer_hessian.materialize_dense()?,
            result.warm_start,
        ))
    }
}
