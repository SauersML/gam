//! Shared latent-interval engine backing both `LatentSurvivalFamily` and
//! `LatentBinaryFamily`.
//!
//! Both families consume the same per-row interval shape — a delayed-entry
//! `(age_entry, age_exit)` window, an event target in `{0, 1}`, a non-negative
//! weight, a monotone unloaded-mass decomposition, and a structural time block
//! whose entry/exit/derivative designs all share `n` rows and `p_time`
//! columns. The only model-specific knobs are:
//!
//! * how the frailty spec resolves to `(scale, loading)` — the survival model
//!   permits a learned scale, while the binary model demands a fixed one;
//! * whether the per-row unloaded *hazard* (`unloaded_hazard_exit`) participates
//!   — the survival model carries it (it feeds the exact-event loaded/unloaded
//!   split), the binary model never observes an exact event so it has none.
//!
//! Those two knobs are captured by [`LatentIntervalModel`]; the shared
//! [`validate_latent_interval_inputs`] driver owns every common check so the
//! survival and binary validators are thin adapters that differ only in the
//! descriptor they hand it.

use crate::survival::location_scale::TimeBlockInput;
use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use ndarray::{Array1, ArrayView2};

/// Outcome of resolving a [`FrailtySpec`] for a latent-interval model: the
/// typed latent scale and the hazard loading. The binary model rejects
/// [`FrailtyScale::Learned`] in its `frailty_policy`.
#[derive(Clone, Copy, Debug)]
pub struct LatentFrailtyResolution {
    pub scale: FrailtyScale,
    pub loading: HazardLoading,
}

/// The portions of a per-row dataset that the shared validator needs but that
/// live under different field names on the two term specs. A model assembles
/// this once from its spec; the validator then iterates it generically.
pub struct LatentIntervalRowView<'a> {
    pub frailty: &'a FrailtySpec,
    pub age_entry: &'a Array1<f64>,
    pub age_exit: &'a Array1<f64>,
    pub event_target: &'a Array1<u8>,
    pub weights: &'a Array1<f64>,
    pub unloaded_mass_entry: &'a Array1<f64>,
    pub unloaded_mass_exit: &'a Array1<f64>,
    /// Present only for the survival model: the per-row unloaded baseline
    /// hazard at exit. `None` for the binary model, which never evaluates an
    /// exact event and therefore carries no hazard component.
    pub unloaded_hazard_exit: Option<&'a Array1<f64>>,
    pub mean_offset: &'a Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: &'a TimeBlockInput,
}

/// Model-specific policy for a latent-interval family. The survival and binary
/// families implement this; everything else (per-row and time-block
/// validation) is owned by [`validate_latent_interval_inputs`].
pub trait LatentIntervalModel {
    /// Lower-case diagnostic prefix used verbatim in every validation error
    /// (`"latent-survival"` / `"latent-binary"`) so external messages are
    /// byte-identical to the pre-unification per-family validators.
    fn context() -> &'static str;

    /// Resolve the supplied frailty spec into a scale / loading pair, or reject
    /// it. The survival policy permits a learned scale; the binary policy
    /// requires a finite fixed sigma.
    fn frailty_policy(
        frailty: &FrailtySpec,
    ) -> Result<LatentFrailtyResolution, crate::survival::latent::LatentSurvivalError>;

    /// Whether this model accepts interval-censored rows (the reserved
    /// `LATENT_SURVIVAL_EVENT_INTERVAL` event code). Survival does; binary never
    /// observes an event window, so it rejects the interval code as an invalid
    /// event target. Defaults to `false`.
    fn allows_interval() -> bool {
        false
    }
}

/// Shared validation driver for the latent-interval families.
///
/// Checks, in order: frailty resolution, non-empty data, per-spec length
/// agreement (including `unloaded_hazard_exit` when the model carries it),
/// the derivative guard, an atomic whole-vector weight preflight, every active
/// per-row interval/event/unloaded-mass invariant, and the time block's
/// row/column/offset shape. Exact-zero rows are dormant after the weight
/// preflight and their response geometry is never inspected. Returns the
/// resolved typed latent scale on success.
pub fn validate_latent_interval_inputs<M: LatentIntervalModel>(
    data: ArrayView2<'_, f64>,
    row: &LatentIntervalRowView<'_>,
) -> Result<FrailtyScale, crate::survival::latent::LatentSurvivalError> {
    use crate::survival::latent::{LatentSurvivalError, validate_unloaded_components_for_loading};

    let context = M::context();
    let resolution = M::frailty_policy(row.frailty)?;
    let LatentFrailtyResolution { scale, loading } = resolution;
    let n = data.nrows();
    if n == 0 {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!("{context} requires a non-empty dataset"),
        });
    }
    let hazard_lengths_match = match row.unloaded_hazard_exit {
        Some(hazard) => hazard.len() == n,
        None => true,
    };
    if row.age_entry.len() != n
        || row.age_exit.len() != n
        || row.event_target.len() != n
        || row.weights.len() != n
        || row.unloaded_mass_entry.len() != n
        || row.unloaded_mass_exit.len() != n
        || !hazard_lengths_match
        || row.mean_offset.len() != n
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: size_mismatch_reason(context, n, row),
        });
    }
    if !row.derivative_guard.is_finite() || row.derivative_guard < 0.0 {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "{context} derivative_guard must be finite and >= 0, got {}",
                row.derivative_guard
            ),
        });
    }
    for (i, &weight) in row.weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has invalid weight {}; expected a finite non-negative weight",
                    i + 1,
                    weight
                ),
            });
        }
    }
    for i in 0..n {
        let entry = row.age_entry[i];
        let exit = row.age_exit[i];
        let event = row.event_target[i];
        let weight = row.weights[i];
        let unloaded_entry = row.unloaded_mass_entry[i];
        let unloaded_exit = row.unloaded_mass_exit[i];
        let unloaded_hazard = row.unloaded_hazard_exit.map(|hazard| hazard[i]);
        if weight == 0.0 {
            continue;
        }
        if !entry.is_finite() || !exit.is_finite() {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has non-finite entry/exit ages: entry={}, exit={}",
                    i + 1,
                    entry,
                    exit
                ),
            });
        }
        if entry < 0.0 || exit < entry {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has invalid delayed-entry bounds: entry={}, exit={}",
                    i + 1,
                    entry,
                    exit
                ),
            });
        }
        let is_interval = event == crate::survival::latent::LATENT_SURVIVAL_EVENT_INTERVAL;
        if is_interval && !M::allows_interval() {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has the interval-censoring event code but {context} does not \
                     support interval-censored rows",
                    i + 1
                ),
            });
        }
        if event > 1 && !is_interval {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has invalid event target {}; expected 0 or 1",
                    i + 1,
                    event
                ),
            });
        }
        let masses_invalid = !unloaded_entry.is_finite()
            || !unloaded_exit.is_finite()
            || unloaded_entry < 0.0
            || unloaded_exit < unloaded_entry;
        let hazard_invalid =
            unloaded_hazard.is_some_and(|hazard| !hazard.is_finite() || hazard < 0.0);
        if masses_invalid || hazard_invalid {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: unloaded_decomposition_reason(
                    context,
                    i,
                    unloaded_entry,
                    unloaded_exit,
                    unloaded_hazard,
                ),
            });
        }
        validate_unloaded_components_for_loading(
            context,
            i,
            loading,
            unloaded_entry,
            unloaded_exit,
            unloaded_hazard,
        )?;
    }
    validate_latent_interval_time_block(context, n, row.time_block)?;
    Ok(scale)
}

/// The size-mismatch diagnostic. The survival variant carries the
/// `unloaded_hazard` length, the binary variant omits it, so the message text
/// stays byte-identical to the pre-unification per-family validators.
fn size_mismatch_reason(context: &str, n: usize, row: &LatentIntervalRowView<'_>) -> String {
    match row.unloaded_hazard_exit {
        Some(hazard) => format!(
            "{context} size mismatch: data has {n} rows, entry={}, exit={}, event={}, weights={}, unloaded_entry={}, unloaded_exit={}, unloaded_hazard={}, offset={}",
            row.age_entry.len(),
            row.age_exit.len(),
            row.event_target.len(),
            row.weights.len(),
            row.unloaded_mass_entry.len(),
            row.unloaded_mass_exit.len(),
            hazard.len(),
            row.mean_offset.len()
        ),
        None => format!(
            "{context} size mismatch: data has {n} rows, entry={}, exit={}, event={}, weights={}, unloaded_entry={}, unloaded_exit={}, offset={}",
            row.age_entry.len(),
            row.age_exit.len(),
            row.event_target.len(),
            row.weights.len(),
            row.unloaded_mass_entry.len(),
            row.unloaded_mass_exit.len(),
            row.mean_offset.len()
        ),
    }
}

/// The invalid-unloaded-decomposition diagnostic. The survival variant reports
/// the exit hazard, the binary variant reports only the two masses, matching
/// the pre-unification per-family messages exactly.
fn unloaded_decomposition_reason(
    context: &str,
    row_index: usize,
    unloaded_entry: f64,
    unloaded_exit: f64,
    unloaded_hazard: Option<f64>,
) -> String {
    match unloaded_hazard {
        Some(hazard) => format!(
            "{context} row {} has invalid unloaded hazard decomposition: entry_mass={}, exit_mass={}, exit_hazard={}",
            row_index + 1,
            unloaded_entry,
            unloaded_exit,
            hazard
        ),
        None => format!(
            "{context} row {} has invalid unloaded mass decomposition: entry_mass={}, exit_mass={}",
            row_index + 1,
            unloaded_entry,
            unloaded_exit,
        ),
    }
}

/// Shared time-block row/column/offset validation. The survival and binary
/// time blocks are structurally identical (entry/exit/derivative designs over
/// `n` rows and a common `p_time`), so this owns the full check.
fn validate_latent_interval_time_block(
    context: &str,
    n: usize,
    time_block: &TimeBlockInput,
) -> Result<(), crate::survival::latent::LatentSurvivalError> {
    use crate::survival::latent::LatentSurvivalError;
    let p_time = time_block.design_exit.ncols();
    if time_block.design_entry.nrows() != n
        || time_block.design_exit.nrows() != n
        || time_block.design_derivative_exit.nrows() != n
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "{context} time block row mismatch: n={}, entry_rows={}, exit_rows={}, derivative_rows={}",
                n,
                time_block.design_entry.nrows(),
                time_block.design_exit.nrows(),
                time_block.design_derivative_exit.nrows()
            ),
        });
    }
    if time_block.design_entry.ncols() != p_time
        || time_block.design_derivative_exit.ncols() != p_time
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "{context} time block column mismatch: entry_cols={}, exit_cols={}, derivative_cols={}",
                time_block.design_entry.ncols(),
                time_block.design_exit.ncols(),
                time_block.design_derivative_exit.ncols()
            ),
        });
    }
    if time_block.offset_entry.len() != n
        || time_block.offset_exit.len() != n
        || time_block.derivative_offset_exit.len() != n
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "{context} time block offset mismatch: n={}, entry_offset={}, exit_offset={}, derivative_offset={}",
                n,
                time_block.offset_entry.len(),
                time_block.offset_exit.len(),
                time_block.derivative_offset_exit.len()
            ),
        });
    }
    Ok(())
}
