use super::*;

#[cfg(test)]
#[path = "checkpoint_price_2953_tests.rs"]
mod checkpoint_price_2953_tests;

/// Why a stored checkpoint could not be priced at full inner fidelity (#2953).
#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointPriceRefusal {
    /// The objective refused the evaluation as an infeasible trial.
    Refused(String),
    /// The inner solve did not converge with the search-time inner cap lifted.
    InnerUnconverged,
    /// The evaluation returned a non-finite value.
    NonFinite(f64),
}

impl std::fmt::Display for CheckpointPriceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(error) => write!(f, "the objective refused it: {error}"),
            Self::InnerUnconverged => {
                f.write_str("its inner solve did not converge with the search-time cap lifted")
            }
            Self::NonFinite(value) => write!(f, "it evaluated to {value}"),
        }
    }
}

/// A stored checkpoint that beat the published optimum on its stored value but could not be
/// priced at full inner fidelity, so it did not defeat it: an unpriceable state cannot defeat a
/// certified one (#2953).
#[derive(Clone, Debug, PartialEq)]
pub struct UnpriceableCheckpoint {
    /// Where the checkpoint sits.
    pub rho: Array1<f64>,
    /// Its stored value, where a search stopped.
    pub stored_value: f64,
    /// Why it could not be priced.
    pub refusal: CheckpointPriceRefusal,
}

/// What a stored checkpoint is worth under the criterion that judges it (#2953).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CheckpointPrice {
    /// Its value at full inner fidelity.
    Priced(f64),
    /// It has no full-fidelity value.
    Unpriceable(CheckpointPriceRefusal),
}

/// Price a stored checkpoint at its own `rho` before it may outrank a certified optimum (#2953).
///
/// The price follows the protocol that priced every stored value it is compared with. The inner
/// solve starts from the objective's reset state, which is where each seed's search started,
/// with the literal seed's cached inner state installed only at that seed's own `rho`. The
/// search-time inner cap is lifted, as the finalize installation and the terminal certificate
/// lift it, because a stored value may come from a capped solve that is no evidence about the
/// profiled criterion. A state the objective refuses there, whose inner solve does not converge
/// there, or which evaluates non-finite, is unpriceable. Any other failure propagates.
pub(crate) fn price_checkpoint(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    rho: &Array1<f64>,
    context: &str,
) -> Result<CheckpointPrice, EstimationError> {
    let full_fidelity = config
        .outer_inner_cap
        .as_ref()
        .map(FullFidelityInnerCapGuard::lift);
    obj.reset();
    install_matching_initial_inner_seed(obj, config, rho, context)?;
    let value = obj.eval_cost(rho);
    // Read before anything else can solve, so it describes this evaluation.
    let inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
    drop(full_fidelity);
    obj.reset();
    match value {
        Ok(_) if !inner_converged => Ok(CheckpointPrice::Unpriceable(
            CheckpointPriceRefusal::InnerUnconverged,
        )),
        Ok(value) if value.is_finite() => Ok(CheckpointPrice::Priced(value)),
        Ok(value) => Ok(CheckpointPrice::Unpriceable(CheckpointPriceRefusal::NonFinite(
            value,
        ))),
        Err(error) if error.is_trial_point_infeasible() => Ok(CheckpointPrice::Unpriceable(
            CheckpointPriceRefusal::Refused(error.to_string()),
        )),
        Err(error) => Err(error),
    }
}
