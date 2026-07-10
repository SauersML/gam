//! Rung-3 intervention shard — the validated data contract between the Python
//! patch runner (the model-interaction boundary) and the Rust calibration fit.
//!
//! See `RUNG3_INTERVENTIONS_DESIGN.md` (§6) in this directory. One record per
//! executed intervention: `(token row, atom, dose Δt)` with the *predicted*
//! nats (`ν̂₁` from the Rung-1 behavioral-Fisher metric, `ν̂₂` from the Rung-2
//! behavior decoder when a y-block exists) and the *measured* realized KL. The
//! `.npz` I/O lives at the Python boundary (mirroring the harvest-shard
//! discipline of `gamfit/torch/harvest.py` / `gamfit/torch/interventions.py`);
//! this type owns validation and the **G2 eval-forever split**.  The typed
//! calibration plan in this module also owns every policy that turns those
//! records into a Rung-3 fit and turns fitted predictions into chart
//! re-speeds; language bindings only marshal the plan into their fitting
//! surface.
//!
//! # The G2 split is part of the contract
//!
//! Guard G2 of the design: the held-out intervention set is never trained on,
//! ever, across refits. That only holds if the split is a *deterministic pure
//! function of (group id, seed)* — independent of record order, of which other
//! groups happen to be present, and of how many times the shard is reloaded.
//! [`InterventionShard::eval_forever_split`] therefore hashes each group id
//! through SplitMix64 with the caller's seed and assigns by parity: adding new
//! groups later can never move an existing group across the fence. The Python
//! calibration driver consumes the plan produced here, so there is no second
//! implementation of the split or any other calibration policy.

use std::collections::BTreeMap;
use std::fmt;

/// One shard of executed interventions. All per-record vectors share length
/// `m`; `dose` is row-major `(m, d_dose)`.
#[derive(Clone, Debug)]
pub struct InterventionShard {
    /// Corpus row (token) each intervention was applied at.
    pub row_id: Vec<i64>,
    /// Atom index `k` whose chart was moved.
    pub atom: Vec<i64>,
    /// Applied coordinate move `Δt`, row-major `(m, d_dose)`. All-zero rows
    /// are the Δt = 0 control splices (guard G3's measurement null).
    pub dose: Vec<f64>,
    /// Latent dose dimensionality `d`.
    pub d_dose: usize,
    /// Rung-1 predicted nats `½ Δxᵀ G_n Δx` (behavioral-Fisher metric).
    pub nu_hat_1: Vec<f64>,
    /// Rung-2 predicted nats (behavior decoder); `None` when the fit carried
    /// no y-block. When present it must be finite and non-negative.
    pub nu_hat_2: Option<Vec<f64>>,
    /// Measured realized KL(clean ‖ patched), nats.
    pub nu_measured: Vec<f64>,
    /// Document/question id — the G2 split unit.
    pub group: Vec<i64>,
    /// Whether the record is a Δt = 0 control splice.
    pub is_control: Vec<bool>,
    /// Hook layer the splice ran at.
    pub layer: i64,
    /// Seed of the sampling plan that produced the records.
    pub seed: u64,
}

/// The G2 manifest: which groups are train, which are eval-forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalForeverSplit {
    /// Groups the calibration fit may use. Sorted ascending, deduplicated.
    pub train_groups: Vec<i64>,
    /// Groups reserved for evaluation forever. Sorted ascending, deduplicated.
    pub eval_groups: Vec<i64>,
}

/// The one production calibration model.  Keeping the model description next
/// to the design builder makes the Rust library, CLI, and Python binding
/// consume one contract instead of spelling model policy in each front-end.
pub const CHART_CALIBRATION_FORMULA: &str = "log_nu ~ s(log_nu_hat) + re(atom)";
pub const CHART_CALIBRATION_SMOOTH_TERM: &str = "s(log_nu_hat)";
pub const CHART_CALIBRATION_SMOOTH_CONSTRAINT: &str = "monotone_increasing";

/// Which typed predicted-nats channel the Rung-3 calibration consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictedNats {
    Rung1,
    Rung2,
}

/// Caller decisions needed to build a Rung-3 chart calibration design from a
/// validated [`InterventionShard`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterventionCalibrationSpec {
    pub prediction: PredictedNats,
    pub split_seed: u64,
    /// Caller-selected one-sided evidence quantile for the G3 control floor.
    pub floor_quantile: f64,
}

/// Fully prepared calibration design.  Numeric transforms, selectors, and
/// reference/evaluation rows have already been decided by the Rust core.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionCalibrationPlan {
    /// Response for the calibration fit, `log(max(nu_measured, floor))`.
    pub train_log_nu: Vec<f64>,
    /// Predictor for the calibration fit, `log(nu_hat)`.
    pub train_log_nu_hat: Vec<f64>,
    pub train_atom: Vec<i64>,
    /// One reference predictor row per measurable atom, used to isolate the
    /// centered random intercept and hence the chart re-speed.
    pub reference_log_nu_hat: f64,
    pub measurable_atoms: Vec<i64>,
    /// Held-out response and predictors.  These groups never enter the fit.
    pub eval_log_nu: Vec<f64>,
    pub eval_log_nu_hat: Vec<f64>,
    pub eval_atom: Vec<i64>,
    pub below_measurement_floor_atoms: Vec<i64>,
    pub no_training_intervention_atoms: Vec<i64>,
    pub floor_nats: f64,
}

/// Final chart-safe calibration output.  It deliberately contains only
/// coordinate re-speeds and diagnostics (guard G1), never a value that can
/// enter a fit criterion.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionCalibrationResult {
    pub respeed: Vec<(i64, f64)>,
    pub below_measurement_floor: Vec<i64>,
    pub no_training_intervention: Vec<i64>,
    pub floor_nats: f64,
    pub heldout_rmse_lognats: Option<f64>,
    pub n_train: usize,
    pub n_eval: usize,
}

/// Typed failures from calibration design construction or fitted-prediction
/// reduction.  Front-ends may map this to their native error hierarchy without
/// having to parse strings or repeat validation.
#[derive(Clone, Debug, PartialEq)]
pub enum InterventionCalibrationError {
    InvalidShard(String),
    Rung2Unavailable,
    InvalidFloorQuantile(f64),
    NoTrainingControls,
    NonPositiveControlFloor(f64),
    NoUsableTrainingRecords,
    PredictionLengthMismatch {
        phase: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinitePrediction {
        phase: &'static str,
        index: usize,
        value: f64,
    },
    NonRepresentableRespeed {
        atom: i64,
        centered_log_speed: f64,
    },
}

impl fmt::Display for InterventionCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShard(message) => {
                write!(f, "intervention calibration: invalid shard: {message}")
            }
            Self::Rung2Unavailable => write!(
                f,
                "intervention calibration: Rung-2 predicted nats were requested but the shard has no Rung-2 channel"
            ),
            Self::InvalidFloorQuantile(q) => write!(
                f,
                "intervention calibration: floor_quantile must be finite and in (0, 1); got {q}"
            ),
            Self::NoTrainingControls => write!(
                f,
                "intervention calibration: no control records occur in the train split; the G3 floor must be estimated from controls"
            ),
            Self::NonPositiveControlFloor(value) => write!(
                f,
                "intervention calibration: the train-control quantile is {value}; log-scale calibration requires a strictly positive measured resolution"
            ),
            Self::NoUsableTrainingRecords => write!(
                f,
                "intervention calibration: no measurable, positive-prediction intervention records remain in the train split"
            ),
            Self::PredictionLengthMismatch {
                phase,
                expected,
                got,
            } => write!(
                f,
                "intervention calibration: {phase} predictions have length {got}; expected {expected}"
            ),
            Self::NonFinitePrediction {
                phase,
                index,
                value,
            } => write!(
                f,
                "intervention calibration: {phase} prediction {index} is not finite ({value})"
            ),
            Self::NonRepresentableRespeed {
                atom,
                centered_log_speed,
            } => write!(
                f,
                "intervention calibration: atom {atom} re-speed is not representable from centered log speed {centered_log_speed}"
            ),
        }
    }
}

impl std::error::Error for InterventionCalibrationError {}

/// SplitMix64 — the split's hash. A fixed, well-known mixing function so the
/// group→side assignment is reproducible across languages and releases.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The G2 per-group predicate: group `g` is eval-forever under a split whose
/// seed hashes to `seed_mix` iff `splitmix64(g ^ seed_mix)` is odd. The single
/// place the split's membership is decided — shared by
/// [`InterventionShard::eval_forever_split`] and [`eval_forever_mask`].
#[inline]
fn group_is_eval_forever(g: i64, seed_mix: u64) -> bool {
    splitmix64((g as u64) ^ seed_mix) & 1 == 1
}

/// Per-record eval-forever mask: `mask[i]` is true iff record `i`'s `group[i]`
/// is an eval-forever group under `seed` (guard G2). A pure per-group function
/// of `(group id, seed)` — record order, shard composition, and refit history
/// cannot move a record across the fence. This is the single source of truth
/// the Python `intervention_calibration._eval_forever_mask` binding wraps, so
/// the SplitMix64 split stays bit-identical across the language boundary.
pub fn eval_forever_mask(group: &[i64], seed: u64) -> Vec<bool> {
    let seed_mix = splitmix64(seed);
    group
        .iter()
        .map(|&g| group_is_eval_forever(g, seed_mix))
        .collect()
}

/// NumPy-compatible inclusive linear-interpolation quantile over a non-empty,
/// finite sample.  Callers validate the preconditions before entering this
/// small common kernel.
fn inclusive_quantile(mut values: Vec<f64>, q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let h = q * (values.len() as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    values[lo] * (1.0 - frac) + values[hi] * frac
}

/// Build the single-source Rung-3 calibration design.
///
/// This function owns guards G2/G3 and every numerical transform used by the
/// calibration fit: the permanent split, train-control floor, measurable-atom
/// screen, log/floor transform, fit selector, reference predictor, and held-out
/// rows.  A front-end only has to fit [`CHART_CALIBRATION_FORMULA`] to the three
/// `train_*` vectors and request predictions at the prepared reference/eval
/// rows.
pub fn prepare_intervention_calibration(
    shard: &InterventionShard,
    spec: InterventionCalibrationSpec,
) -> Result<InterventionCalibrationPlan, InterventionCalibrationError> {
    shard
        .validate()
        .map_err(InterventionCalibrationError::InvalidShard)?;
    let nu_hat = match spec.prediction {
        PredictedNats::Rung1 => shard.nu_hat_1.as_slice(),
        PredictedNats::Rung2 => shard
            .nu_hat_2
            .as_deref()
            .ok_or(InterventionCalibrationError::Rung2Unavailable)?,
    };
    let n = shard.n_records();
    if !(spec.floor_quantile.is_finite()
        && spec.floor_quantile > 0.0
        && spec.floor_quantile < 1.0)
    {
        return Err(InterventionCalibrationError::InvalidFloorQuantile(
            spec.floor_quantile,
        ));
    }

    let eval = eval_forever_mask(&shard.group, spec.split_seed);
    let train_controls: Vec<f64> = (0..n)
        .filter(|&i| !eval[i] && shard.is_control[i])
        .map(|i| shard.nu_measured[i])
        .collect();
    if train_controls.is_empty() {
        return Err(InterventionCalibrationError::NoTrainingControls);
    }
    let floor_nats = inclusive_quantile(train_controls, spec.floor_quantile);
    if !(floor_nats.is_finite() && floor_nats > 0.0) {
        return Err(InterventionCalibrationError::NonPositiveControlFloor(
            floor_nats,
        ));
    }

    // Sorted map makes both the Rust API and every binding deterministic.
    // Every atom with at least one train intervention is classified exactly
    // once; controls and eval-forever rows cannot influence measurability.
    let mut atom_is_measurable: BTreeMap<i64, Option<bool>> = shard
        .atom
        .iter()
        .copied()
        .map(|atom| (atom, None))
        .collect();
    for i in 0..n {
        if !eval[i] && !shard.is_control[i] {
            let measurable = shard.nu_measured[i] > floor_nats;
            atom_is_measurable
                .entry(shard.atom[i])
                .and_modify(|seen| *seen = Some(seen.unwrap_or(false) || measurable))
                .or_insert(Some(measurable));
        }
    }
    let measurable_atoms: Vec<i64> = atom_is_measurable
        .iter()
        .filter_map(|(&atom, &measurable)| (measurable == Some(true)).then_some(atom))
        .collect();
    let below_measurement_floor_atoms: Vec<i64> = atom_is_measurable
        .iter()
        .filter_map(|(&atom, &measurable)| (measurable == Some(false)).then_some(atom))
        .collect();
    let no_training_intervention_atoms: Vec<i64> = atom_is_measurable
        .iter()
        .filter_map(|(&atom, &measurable)| measurable.is_none().then_some(atom))
        .collect();

    let mut train_log_nu = Vec::new();
    let mut train_log_nu_hat = Vec::new();
    let mut train_atom = Vec::new();
    let mut eval_log_nu = Vec::new();
    let mut eval_log_nu_hat = Vec::new();
    let mut eval_atom = Vec::new();
    for i in 0..n {
        if shard.is_control[i]
            || nu_hat[i] <= 0.0
            || !matches!(atom_is_measurable.get(&shard.atom[i]), Some(Some(true)))
        {
            continue;
        }
        let log_nu = shard.nu_measured[i].max(floor_nats).ln();
        let log_nu_hat = nu_hat[i].ln();
        if eval[i] {
            eval_log_nu.push(log_nu);
            eval_log_nu_hat.push(log_nu_hat);
            eval_atom.push(shard.atom[i]);
        } else {
            train_log_nu.push(log_nu);
            train_log_nu_hat.push(log_nu_hat);
            train_atom.push(shard.atom[i]);
        }
    }
    if train_log_nu.is_empty() {
        return Err(InterventionCalibrationError::NoUsableTrainingRecords);
    }
    let reference_log_nu_hat = inclusive_quantile(train_log_nu_hat.clone(), 0.5);

    Ok(InterventionCalibrationPlan {
        train_log_nu,
        train_log_nu_hat,
        train_atom,
        reference_log_nu_hat,
        measurable_atoms,
        eval_log_nu,
        eval_log_nu_hat,
        eval_atom,
        below_measurement_floor_atoms,
        no_training_intervention_atoms,
        floor_nats,
    })
}

impl InterventionCalibrationPlan {
    /// Convert predictions from the fitted calibration model into the only
    /// chart mutation calibration may emit (G1) and its held-out diagnostic.
    pub fn finish(
        &self,
        reference_eta: &[f64],
        eval_eta: &[f64],
    ) -> Result<InterventionCalibrationResult, InterventionCalibrationError> {
        if reference_eta.len() != self.measurable_atoms.len() {
            return Err(
                InterventionCalibrationError::PredictionLengthMismatch {
                    phase: "reference",
                    expected: self.measurable_atoms.len(),
                    got: reference_eta.len(),
                },
            );
        }
        if eval_eta.len() != self.eval_log_nu.len() {
            return Err(
                InterventionCalibrationError::PredictionLengthMismatch {
                    phase: "held-out",
                    expected: self.eval_log_nu.len(),
                    got: eval_eta.len(),
                },
            );
        }
        for (phase, values) in [("reference", reference_eta), ("held-out", eval_eta)] {
            for (index, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(InterventionCalibrationError::NonFinitePrediction {
                        phase,
                        index,
                        value,
                    });
                }
            }
        }

        // Online mean avoids overflow from summing many individually finite
        // linear predictors.  A non-representable center is a typed refusal,
        // never a silently saturated chart update.
        let mut mean_eta = 0.0_f64;
        for (i, &eta) in reference_eta.iter().enumerate() {
            mean_eta += (eta - mean_eta) / (i + 1) as f64;
        }
        let mut respeed = Vec::with_capacity(reference_eta.len());
        for (&atom, &eta) in self.measurable_atoms.iter().zip(reference_eta) {
            let centered_log_speed = eta - mean_eta;
            let value = (0.5 * centered_log_speed).exp();
            if !(value.is_finite() && value > 0.0) {
                return Err(InterventionCalibrationError::NonRepresentableRespeed {
                    atom,
                    centered_log_speed,
                });
            }
            respeed.push((atom, value));
        }

        let heldout_rmse_lognats = if eval_eta.is_empty() {
            None
        } else {
            // Hypot accumulation computes the Euclidean norm without squaring
            // overflow, then division by sqrt(n) gives RMSE.
            let residual_norm = eval_eta
                .iter()
                .zip(&self.eval_log_nu)
                .fold(0.0_f64, |norm, (&predicted, &observed)| {
                    norm.hypot(predicted - observed)
                });
            Some(residual_norm / (eval_eta.len() as f64).sqrt())
        };

        Ok(InterventionCalibrationResult {
            respeed,
            below_measurement_floor: self.below_measurement_floor_atoms.clone(),
            no_training_intervention: self.no_training_intervention_atoms.clone(),
            floor_nats: self.floor_nats,
            heldout_rmse_lognats,
            n_train: self.train_log_nu.len(),
            n_eval: self.eval_log_nu.len(),
        })
    }
}

impl InterventionShard {
    /// Validate the shard invariants. Errors carry the first offending record.
    ///
    /// Invariants:
    /// * equal record counts across all per-record vectors, `dose` of shape
    ///   `(m, d_dose)`;
    /// * every numeric entry finite; predictions and measurements
    ///   non-negative (KL and quadratic forms are);
    /// * `is_control[i]` ⇔ `dose` row `i` is all-zero — the G3 null is defined
    ///   by the dose actually applied, so a mislabeled control is a hard error,
    ///   not a warning.
    pub fn validate(&self) -> Result<(), String> {
        let m = self.row_id.len();
        if self.d_dose == 0 {
            return Err("InterventionShard: d_dose must be >= 1".to_string());
        }
        let checks: [(&str, usize); 5] = [
            ("atom", self.atom.len()),
            ("nu_hat_1", self.nu_hat_1.len()),
            ("nu_measured", self.nu_measured.len()),
            ("group", self.group.len()),
            ("is_control", self.is_control.len()),
        ];
        for (name, len) in checks {
            if len != m {
                return Err(format!(
                    "InterventionShard: {name} has {len} records but row_id has {m}"
                ));
            }
        }
        if self.dose.len() != m * self.d_dose {
            return Err(format!(
                "InterventionShard: dose has {} entries; expected m*d = {}*{} = {}",
                self.dose.len(),
                m,
                self.d_dose,
                m * self.d_dose
            ));
        }
        if let Some(nu2) = &self.nu_hat_2 {
            if nu2.len() != m {
                return Err(format!(
                    "InterventionShard: nu_hat_2 has {} records but row_id has {m}",
                    nu2.len()
                ));
            }
        }
        for i in 0..m {
            let d_row = &self.dose[i * self.d_dose..(i + 1) * self.d_dose];
            if !d_row.iter().all(|v| v.is_finite()) {
                return Err(format!("InterventionShard: record {i}: non-finite dose"));
            }
            let zero_dose = d_row.iter().all(|&v| v == 0.0);
            if zero_dose != self.is_control[i] {
                return Err(format!(
                    "InterventionShard: record {i}: is_control={} but dose is {}zero \
                     (the G3 null is defined by the applied dose)",
                    self.is_control[i],
                    if zero_dose { "" } else { "non-" }
                ));
            }
            for (name, v) in [
                ("nu_hat_1", self.nu_hat_1[i]),
                ("nu_measured", self.nu_measured[i]),
            ] {
                if !(v.is_finite() && v >= 0.0) {
                    return Err(format!(
                        "InterventionShard: record {i}: {name} must be finite and >= 0; got {v}"
                    ));
                }
            }
            if let Some(nu2) = &self.nu_hat_2 {
                if !(nu2[i].is_finite() && nu2[i] >= 0.0) {
                    return Err(format!(
                        "InterventionShard: record {i}: nu_hat_2 must be finite and >= 0; got {}",
                        nu2[i]
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of records.
    pub fn n_records(&self) -> usize {
        self.row_id.len()
    }

    /// The G2 eval-forever split: each distinct group id goes to eval iff
    /// `splitmix64(group_id ^ splitmix64(seed))` is odd. A pure per-group
    /// function — record order, shard composition, and refit history cannot
    /// move a group across the fence, which is what makes "eval forever" a
    /// property of the *function* rather than of bookkeeping.
    pub fn eval_forever_split(&self, seed: u64) -> EvalForeverSplit {
        let mut train: Vec<i64> = Vec::new();
        let mut eval: Vec<i64> = Vec::new();
        let mut groups: Vec<i64> = self.group.clone();
        groups.sort_unstable();
        groups.dedup();
        let seed_mix = splitmix64(seed);
        for g in groups {
            if group_is_eval_forever(g, seed_mix) {
                eval.push(g);
            } else {
                train.push(g);
            }
        }
        EvalForeverSplit {
            train_groups: train,
            eval_groups: eval,
        }
    }

    /// Guard G3's measurement floor: the `q`-quantile (0 < q < 1, caller
    /// supplies the same one-sided evidence quantile the certificates use) of
    /// `nu_measured` over the Δt = 0 control records. Errors when the shard
    /// carries no controls — a floor from zero controls would be a fabricated
    /// number, and the design requires the null to be *estimated*.
    pub fn control_floor_nats(&self, q: f64) -> Result<f64, String> {
        if !(q > 0.0 && q < 1.0) {
            return Err(format!(
                "control_floor_nats: quantile must be in (0, 1); got {q}"
            ));
        }
        let nulls: Vec<f64> = self
            .nu_measured
            .iter()
            .zip(self.is_control.iter())
            .filter_map(|(&v, &c)| c.then_some(v))
            .collect();
        if nulls.is_empty() {
            return Err(
                "control_floor_nats: shard has no Δt = 0 control records; the G3 floor \
                 must be estimated from controls, never assumed"
                    .to_string(),
            );
        }
        if nulls.iter().any(|value| !value.is_finite()) {
            return Err(
                "control_floor_nats: control measurements must be finite before taking a quantile"
                    .to_string(),
            );
        }
        // Inclusive linear-interpolation quantile (the same convention as
        // numpy's default), on the validated finite sample.
        Ok(inclusive_quantile(nulls, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_shard() -> InterventionShard {
        InterventionShard {
            row_id: vec![0, 1, 2, 3],
            atom: vec![0, 0, 1, 1],
            dose: vec![0.1, 0.0, -0.2, 0.0],
            d_dose: 1,
            nu_hat_1: vec![0.5, 0.0, 0.8, 0.0],
            nu_hat_2: None,
            nu_measured: vec![0.45, 1e-6, 0.7, 2e-6],
            group: vec![10, 10, 20, 20],
            is_control: vec![false, true, false, true],
            layer: 17,
            seed: 0,
        }
    }

    #[test]
    fn valid_shard_passes() {
        assert!(tiny_shard().validate().is_ok());
    }

    #[test]
    fn mislabeled_control_is_a_hard_error() {
        let mut s = tiny_shard();
        s.is_control[0] = true; // dose 0.1 but claimed control
        assert!(s.validate().unwrap_err().contains("is_control"));
    }

    #[test]
    fn zero_dose_without_control_flag_is_a_hard_error() {
        let mut s = tiny_shard();
        s.is_control[1] = false; // dose 0.0 but not flagged
        assert!(s.validate().unwrap_err().contains("is_control"));
    }

    #[test]
    fn negative_measured_kl_rejected() {
        let mut s = tiny_shard();
        s.nu_measured[0] = -0.1;
        assert!(s.validate().unwrap_err().contains("nu_measured"));
    }

    #[test]
    fn split_is_deterministic_and_partitions_groups() {
        let s = tiny_shard();
        let a = s.eval_forever_split(7);
        let b = s.eval_forever_split(7);
        assert_eq!(a, b);
        let mut all: Vec<i64> = a
            .train_groups
            .iter()
            .chain(a.eval_groups.iter())
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, vec![10, 20]);
    }

    #[test]
    fn split_is_per_group_stable_under_shard_growth() {
        // Adding new groups must never move an existing group across the
        // fence — the "eval forever" property.
        let s = tiny_shard();
        let before = s.eval_forever_split(3);
        let mut grown = s.clone();
        grown.row_id.extend([4, 5]);
        grown.atom.extend([2, 2]);
        grown.dose.extend([0.3, 0.0]);
        grown.nu_hat_1.extend([0.2, 0.0]);
        grown.nu_measured.extend([0.15, 1e-6]);
        grown.group.extend([30, 30]);
        grown.is_control.extend([false, true]);
        grown.validate().unwrap();
        let after = grown.eval_forever_split(3);
        for g in &before.train_groups {
            assert!(after.train_groups.contains(g), "group {g} left train");
        }
        for g in &before.eval_groups {
            assert!(after.eval_groups.contains(g), "group {g} left eval");
        }
    }

    #[test]
    fn control_floor_is_a_control_quantile() {
        let s = tiny_shard();
        // Controls are {1e-6, 2e-6}: the median is 1.5e-6.
        let f = s.control_floor_nats(0.5).unwrap();
        assert!((f - 1.5e-6).abs() < 1e-12);
    }

    #[test]
    fn control_floor_requires_controls() {
        let mut s = tiny_shard();
        s.is_control = vec![false; 4];
        s.dose = vec![0.1, 0.2, -0.2, 0.4];
        s.validate().unwrap();
        assert!(s.control_floor_nats(0.5).unwrap_err().contains("control"));
    }

    #[test]
    fn splitmix_reference_values_pin_the_cross_language_contract() {
        // A change here moves permanent train/eval membership and is a
        // calibration-contract break, not a refactor.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(1), 0x910A_2DEC_8902_5CC1);
    }

    fn group_on_side(seed: u64, eval: bool) -> i64 {
        (0_i64..)
            .find(|&group| eval_forever_mask(&[group], seed)[0] == eval)
            .expect("an infinite sequence contains a group on each hash parity")
    }

    fn calibration_shard_and_spec() -> (InterventionShard, InterventionCalibrationSpec) {
        let seed = 19;
        let train_group = group_on_side(seed, false);
        let eval_group = group_on_side(seed, true);
        let shard = InterventionShard {
            // Two train controls establish floor=2.  Atom 10 is measurable;
            // atom 20 never clears the floor and is reported, not fitted.
            row_id: (0..6).collect(),
            atom: vec![10, 10, 10, 20, 10, 10],
            dose: vec![0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
            d_dose: 1,
            nu_hat_1: vec![0.0, 0.0, 1.0, 4.0, 2.0, 3.0],
            nu_hat_2: None,
            nu_measured: vec![1.0, 3.0, 4.0, 1.0, 2.0, 8.0],
            group: vec![
                train_group,
                train_group,
                train_group,
                train_group,
                train_group,
                eval_group,
            ],
            is_control: vec![true, true, false, false, false, false],
            layer: 4,
            seed: 7,
        };
        let spec = InterventionCalibrationSpec {
            prediction: PredictedNats::Rung1,
            split_seed: seed,
            floor_quantile: 0.5,
        };
        (shard, spec)
    }

    #[test]
    fn calibration_plan_owns_split_floor_screen_and_log_transforms() {
        let (shard, spec) = calibration_shard_and_spec();
        let plan = prepare_intervention_calibration(&shard, spec).unwrap();
        assert_eq!(plan.floor_nats, 2.0);
        assert_eq!(plan.measurable_atoms, vec![10]);
        assert_eq!(plan.below_measurement_floor_atoms, vec![20]);
        assert_eq!(plan.no_training_intervention_atoms, Vec::<i64>::new());
        assert_eq!(plan.train_atom, vec![10, 10]);
        assert_eq!(plan.train_log_nu, vec![4.0_f64.ln(), 2.0_f64.ln()]);
        assert_eq!(plan.train_log_nu_hat, vec![0.0, 2.0_f64.ln()]);
        assert_eq!(plan.reference_log_nu_hat, 2.0_f64.ln() / 2.0);
        assert_eq!(plan.eval_atom, vec![10]);
        assert_eq!(plan.eval_log_nu, vec![8.0_f64.ln()]);
        assert_eq!(plan.eval_log_nu_hat, vec![3.0_f64.ln()]);
    }

    #[test]
    fn calibration_finish_computes_only_respeeds_and_heldout_diagnostic() {
        let (mut shard, spec) = calibration_shard_and_spec();
        // Make atom 20 measurable so centering of two reference predictions is
        // observable in the chart updates.
        shard.nu_measured[3] = 5.0;
        let plan = prepare_intervention_calibration(&shard, spec).unwrap();
        let result = plan.finish(&[1.0, 3.0], &[8.0_f64.ln() + 0.25]).unwrap();
        assert_eq!(result.below_measurement_floor, Vec::<i64>::new());
        assert_eq!(result.no_training_intervention, Vec::<i64>::new());
        assert_eq!(result.n_train, 3);
        assert_eq!(result.n_eval, 1);
        assert!((result.respeed[0].1 - (-0.5_f64).exp()).abs() < 1.0e-12);
        assert!((result.respeed[1].1 - 0.5_f64.exp()).abs() < 1.0e-12);
        assert_eq!(result.heldout_rmse_lognats, Some(0.25));
    }

    #[test]
    fn calibration_requires_positive_control_resolution_for_log_model() {
        let (mut shard, spec) = calibration_shard_and_spec();
        shard.nu_measured[0] = 0.0;
        shard.nu_measured[1] = 0.0;
        assert_eq!(
            prepare_intervention_calibration(&shard, spec).unwrap_err(),
            InterventionCalibrationError::NonPositiveControlFloor(0.0)
        );
    }

    #[test]
    fn calibration_finish_rejects_prediction_shape_drift() {
        let (shard, spec) = calibration_shard_and_spec();
        let plan = prepare_intervention_calibration(&shard, spec).unwrap();
        assert!(matches!(
            plan.finish(&[], &[]),
            Err(InterventionCalibrationError::PredictionLengthMismatch {
                phase: "reference",
                expected: 1,
                got: 0,
            })
        ));
    }

    #[test]
    fn calibration_prediction_channel_is_typed() {
        let (mut shard, mut spec) = calibration_shard_and_spec();
        spec.prediction = PredictedNats::Rung2;
        assert_eq!(
            prepare_intervention_calibration(&shard, spec).unwrap_err(),
            InterventionCalibrationError::Rung2Unavailable
        );
        shard.nu_hat_2 = Some(shard.nu_hat_1.iter().map(|value| value * 2.0).collect());
        let plan = prepare_intervention_calibration(&shard, spec).unwrap();
        assert_eq!(plan.train_log_nu_hat[0], 2.0_f64.ln());
    }
}
