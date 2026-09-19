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
//! [`eval_forever_mask`] therefore hashes each group id
//! through SplitMix64 with the caller's seed and assigns by parity: adding new
//! groups later can never move an existing group across the fence. The Python
//! calibration driver consumes the plan produced here, so there is no second
//! implementation of the split or any other calibration policy.
//!
//! # Intervention experiments (#2946 Stage D)
//!
//! [`InterventionExperimentPlan`] generalizes one record per atom and dose to
//! experiments that apply several typed changes together in one patched forward
//! pass and read several declared responses: KL at the edited positions, KL at the
//! positions that follow, token log-probabilities, and executed rows of a later
//! site. Where replacement rows come from is part of their type. Only
//! [`GaussianLoadingLaw`] rows are randomized, and only this module can draw them,
//! so designed or observed rows can never reach an estimator as a randomized dose.
//! Every experiment falls on one side of the same G2 split, and observed rows
//! taken across the fence are refused.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::inference::steering::SteerPlan;
use gam_math::probability::standard_normal_from_uniform_bits;
use gam_linalg::faer_ndarray::FaerQr;
use gam_linalg::roundoff::factor_rank_partition;
use ndarray::{Array2, ArrayView2};

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
    /// Measured realized KL(clean ‖ patched), nats, as computed. Roundoff can make
    /// it negative by at most [`kl_evaluation_band_nats`].
    pub nu_measured: Vec<f64>,
    /// Largest `|logit|` over each record's clean and patched logits.
    pub logit_max_abs: Vec<f64>,
    /// Largest `|patched − clean|` logit change of each record.
    pub logit_max_abs_change: Vec<f64>,
    /// Document/question id — the G2 split unit.
    pub group: Vec<i64>,
    /// Whether the record is a Δt = 0 control splice.
    pub is_control: Vec<bool>,
    /// Hook layer the splice ran at.
    pub layer: i64,
    /// Seed of the sampling plan that produced the records.
    pub seed: u64,
    /// Format of the logits every KL was measured from.
    pub logit_format: LogitFormat,
    /// Vocabulary size of those logits.
    pub vocab_size: usize,
}

/// Floating-point format of the logits a KL was measured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitFormat {
    Float16,
    BFloat16,
    Float32,
    Float64,
}

impl LogitFormat {
    /// Fraction bits `f`: values in `[2^e, 2^(e+1))` are spaced `2^(e − f)` apart.
    pub const fn fraction_bits(self) -> i32 {
        match self {
            Self::Float16 => 10,
            Self::BFloat16 => 7,
            Self::Float32 => 23,
            Self::Float64 => 52,
        }
    }

    /// Exponent of the smallest normal value. Below it the spacing stays at the
    /// subnormal spacing.
    pub const fn min_normal_exponent(self) -> i32 {
        match self {
            Self::Float16 => -14,
            Self::BFloat16 | Self::Float32 => -126,
            Self::Float64 => -1022,
        }
    }

    /// `ulp_F(m)`: the spacing of this format's representable values at magnitude
    /// `m`. It is `2^(e − f)` for `|m| ∈ [2^e, 2^(e+1))`, and the subnormal spacing
    /// below the smallest normal value. It is built from bits, so it is exact.
    pub fn spacing_at(self, magnitude: f64) -> f64 {
        let exponent = if magnitude.is_normal() {
            ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023
        } else {
            self.min_normal_exponent()
        };
        let power = exponent.max(self.min_normal_exponent()) - self.fraction_bits();
        if power >= -1022 {
            f64::from_bits(((power + 1023) as u64) << 52)
        } else {
            f64::from_bits(1_u64 << (power + 1074))
        }
    }
}

impl std::str::FromStr for LogitFormat {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            other => Err(format!(
                "logit format must be float16, bfloat16, float32 or float64; got {other:?}"
            )),
        }
    }
}

/// First-order bound on the float64 evaluation error of the runner's KL:
/// `log_softmax` of both logit vectors, `exp` of the clean one, and one sum of
/// `p_v·(log p_v − log q_v)` over `V` logits.
///
/// With `ε = 2⁻⁵²`, `M` the largest `|logit|` and `Δ` the largest `|patched − clean|`:
/// - log-sum-exp (shift by the max, `V` exponentials, one sum, one `ln`, one add):
///   `λ = ε(V + 3M + 2 ln V)`;
/// - each log-probability: `η = λ + ε(2M + ln V)`;
/// - the probabilities, differences, products and the `V`-term sum add
///   `Σ_v p_v[2η + |d_v|(η + 3ε)] + (V − 1)ε·Σ_v p_v|d_v|`, where `d_v` is the
///   log-probability difference and `Σ_v p_v|d_v| ≤ 2Δ`, because log-sum-exp is
///   1-Lipschitz in the max norm.
///
/// Total: `E₆₄ = 2η + 2Δ(η + (V + 2)ε)`. A computed KL below `−E₆₄` is not roundoff.
pub fn kl_evaluation_band_nats(
    vocab_size: usize,
    logit_max_abs: f64,
    logit_max_abs_change: f64,
) -> f64 {
    let epsilon = f64::EPSILON;
    let vocab = vocab_size as f64;
    let ln_vocab = vocab.ln();
    let log_sum_exp_error = epsilon * (vocab + 3.0 * logit_max_abs + 2.0 * ln_vocab);
    let log_probability_error = log_sum_exp_error + epsilon * (2.0 * logit_max_abs + ln_vocab);
    2.0 * log_probability_error
        + 2.0 * logit_max_abs_change * (log_probability_error + (vocab + 2.0) * epsilon)
}

/// The KL a measurement cannot tell apart from its own arithmetic, in nats:
/// `B = ½·ulp_F(M)² + E₆₄`.
///
/// Rounding in format `F` moves each logit difference by at most `u = ulp_F(M)`.
/// For the logit difference `δ`, `KL(p‖q) = log E_p[e^δ] − E_p[δ]`, and Hoeffding's
/// lemma with every `δ_v ∈ [−u, u]` gives `KL ≤ u²/2` exactly. A measured KL at or
/// below `B` is therefore within what rounding of the logits and the float64
/// evaluation ([`kl_evaluation_band_nats`]) can produce by themselves. The band
/// covers the measurement from logits to KL, not an edit rounded away upstream
/// inside the forward pass.
pub fn kl_measurement_band_nats(
    format: LogitFormat,
    vocab_size: usize,
    logit_max_abs: f64,
    logit_max_abs_change: f64,
) -> f64 {
    let spacing = format.spacing_at(logit_max_abs);
    0.5 * spacing * spacing
        + kl_evaluation_band_nats(vocab_size, logit_max_abs, logit_max_abs_change)
}

/// The one production calibration model.  Keeping the model description next
/// to the design builder makes the Rust library, CLI, and Python binding
/// consume one contract instead of spelling model policy in each front-end.
pub const CHART_CALIBRATION_FORMULA: &str = "log_nu ~ s(log_nu_hat) + group(atom)";
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
    /// Caller-selected one-sided evidence quantile of the train-control
    /// measurements. Where controls are stochastic it raises each record's floor
    /// above the record's derived measurement band.
    pub floor_quantile: f64,
}

/// Fully prepared calibration design.  Numeric transforms, selectors, and
/// reference/evaluation rows have already been decided by the Rust core.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionCalibrationPlan {
    /// Response for the calibration fit, `log(max(nu_measured, floor))`, with
    /// each record's own floor `max(B, control_quantile_nats)`.
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
    /// The `floor_quantile` of the train-control measurements. It is exactly 0 on
    /// a deterministic model.
    pub control_quantile_nats: f64,
    /// The largest derived measurement band over the train interventions.
    pub measurement_band_nats_max: f64,
}

/// Final chart-safe calibration output.  It deliberately contains only
/// coordinate re-speeds and diagnostics (guard G1), never a value that can
/// enter a fit criterion.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionCalibrationResult {
    pub respeed: Vec<(i64, f64)>,
    pub below_measurement_floor: Vec<i64>,
    pub no_training_intervention: Vec<i64>,
    pub control_quantile_nats: f64,
    pub measurement_band_nats_max: f64,
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
                "intervention calibration: no control records occur in the train split; the G3 null and the control quantile are read from controls"
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
    gam_linalg::utils::splitmix64_hash(x)
}

/// The G2 per-group predicate: group `g` is eval-forever under a split whose
/// seed hashes to `seed_mix` iff `splitmix64(g ^ seed_mix)` is odd. The single
/// place the split's membership is decided, called by [`eval_forever_mask`].
#[inline]
fn group_is_eval_forever(g: i64, seed_mix: u64) -> bool {
    splitmix64((g as u64) ^ seed_mix) & 1 == 1
}

/// Per-record eval-forever mask: `mask[i]` is true iff record `i`'s `group[i]`
/// is an eval-forever group under `seed` (guard G2). A pure per-group function
/// of `(group id, seed)` — record order, shard composition, and refit history
/// cannot move a record across the fence. This is the single source of truth
/// for the split; it is Rust-only today (no `gam-pyffi` export wraps it), so
/// any future Python or CLI surface must call through here rather than
/// reimplement the SplitMix64 split.
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
    if !(spec.floor_quantile.is_finite() && spec.floor_quantile > 0.0 && spec.floor_quantile < 1.0)
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
    // A deterministic model's controls re-splice the unchanged row and measure
    // exactly 0, so their quantile is no measurement floor. Each record's floor
    // is its derived measurement band, raised by the control quantile where
    // controls are stochastic. The band is strictly positive, so every floor is.
    let control_quantile_nats = inclusive_quantile(train_controls, spec.floor_quantile);
    let band: Vec<f64> = (0..n)
        .map(|i| {
            kl_measurement_band_nats(
                shard.logit_format,
                shard.vocab_size,
                shard.logit_max_abs[i],
                shard.logit_max_abs_change[i],
            )
        })
        .collect();
    let floor: Vec<f64> = band
        .iter()
        .map(|&record_band| record_band.max(control_quantile_nats))
        .collect();
    let measurement_band_nats_max = (0..n)
        .filter(|&i| !eval[i] && !shard.is_control[i])
        .map(|i| band[i])
        .fold(0.0_f64, f64::max);

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
            let measurable = shard.nu_measured[i] > floor[i];
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
        let log_nu = shard.nu_measured[i].max(floor[i]).ln();
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
        control_quantile_nats,
        measurement_band_nats_max,
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
            return Err(InterventionCalibrationError::PredictionLengthMismatch {
                phase: "reference",
                expected: self.measurable_atoms.len(),
                got: reference_eta.len(),
            });
        }
        if eval_eta.len() != self.eval_log_nu.len() {
            return Err(InterventionCalibrationError::PredictionLengthMismatch {
                phase: "held-out",
                expected: self.eval_log_nu.len(),
                got: eval_eta.len(),
            });
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
            control_quantile_nats: self.control_quantile_nats,
            measurement_band_nats_max: self.measurement_band_nats_max,
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
    /// * every numeric entry finite; predictions non-negative (quadratic forms
    ///   are);
    /// * each measured KL at least `−E₆₄`, its derived float64 evaluation band
    ///   ([`kl_evaluation_band_nats`]): a true KL is non-negative, and only
    ///   roundoff within that band can make a computed one negative;
    /// * logit extents non-negative, with the change at most twice the largest
    ///   `|logit|` (the triangle inequality);
    /// * `is_control[i]` ⇔ `dose` row `i` is all-zero — the G3 null is defined
    ///   by the dose actually applied, so a mislabeled control is a hard error,
    ///   not a warning.
    pub fn validate(&self) -> Result<(), String> {
        let m = self.row_id.len();
        if self.d_dose == 0 {
            return Err("InterventionShard: d_dose must be >= 1".to_string());
        }
        if self.vocab_size == 0 {
            return Err("InterventionShard: vocab_size must be >= 1".to_string());
        }
        let checks: [(&str, usize); 7] = [
            ("atom", self.atom.len()),
            ("nu_hat_1", self.nu_hat_1.len()),
            ("nu_measured", self.nu_measured.len()),
            ("logit_max_abs", self.logit_max_abs.len()),
            ("logit_max_abs_change", self.logit_max_abs_change.len()),
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
            let max_abs = self.logit_max_abs[i];
            let change = self.logit_max_abs_change[i];
            if !(max_abs.is_finite()
                && max_abs >= 0.0
                && change.is_finite()
                && change >= 0.0
                && change <= 2.0 * max_abs)
            {
                return Err(format!(
                    "InterventionShard: record {i}: logit extents must be finite and non-negative \
                     with change <= 2*max_abs; got max_abs {max_abs}, change {change}"
                ));
            }
            for (name, v, lower) in [
                ("nu_hat_1", self.nu_hat_1[i], 0.0),
                (
                    "nu_measured",
                    self.nu_measured[i],
                    -kl_evaluation_band_nats(self.vocab_size, max_abs, change),
                ),
            ] {
                if !(v.is_finite() && v >= lower) {
                    return Err(format!(
                        "InterventionShard: record {i}: {name} must be finite and >= {lower}; got {v}"
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
    pub(crate) fn n_records(&self) -> usize {
        self.row_id.len()
    }
}

/// A hookable site of the executed model. The runner resolves `module_path`;
/// Rust never sees the model, only the width of the rows written and read there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterventionSite {
    pub module_path: String,
    pub width: usize,
}

/// One input sequence of the corpus. `group` is the G2 split unit, and `length`
/// bounds every position an experiment on this unit may touch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExperimentUnit {
    pub group: i64,
    pub sequence: i64,
    pub length: usize,
}

/// The stream key of one Gaussian-law draw. Two replacements sharing a key would
/// share their draws, so a plan refuses a repeated key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DrawKey {
    pub seed: u64,
    pub stream: u64,
}

/// The #2946 declared law `h = h0 + L z` with `z ~ N(0, I_rank)`, together with the
/// coupled draws `z′ = P z + (I − P) z̃` of R4's estimator, drawn by this module.
///
/// The law records `z`, an independent copy `z̃`, and orthonormal frames `Q` with
/// `P = Q Qᵀ`. `P Z` and `(I − P) Z̃` are independent, so `Z′ ~ N(0, I)` and it shares
/// exactly `P Z` with `Z`. The fields are private, so [`GaussianLoadingLaw::draw`] is
/// the only way to hold one: hand-picked or correlated rows cannot pass as randomized.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLoadingLaw {
    baseline: Vec<f64>,
    loading: Vec<f64>,
    rank: usize,
    key: DrawKey,
    draws: Vec<f64>,
    independent_draws: Vec<f64>,
    frames: Vec<Array2<f64>>,
}

/// Which rows of a [`GaussianLoadingLaw`] an experiment executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LawArm {
    /// `h0 + L z`.
    Draw,
    /// `h0 + L z′`, with `z′ = P z + (I − P) z̃` for the law's frame `frame`. A frame of
    /// rank 0 executes `z̃`.
    Coupled { frame: usize },
}

impl GaussianLoadingLaw {
    /// Draw `draws` rows of the law. `baseline` is `h0` (its length is the site width),
    /// `loading` is `L`, row-major `(width, rank)`, and each frame basis has shape
    /// `(rank, r)`.
    ///
    /// `z` and then `z̃` come from [`standard_normal_from_uniform_bits`], the one draw
    /// owner, applied to successive words of the SplitMix64 stream keyed by `key`,
    /// row-major. The draws are recorded in the law, so consumers read them and never
    /// re-derive them: on another platform a re-derivation would match only to libm
    /// rounding.
    ///
    /// A frame basis is refused when the owned rank predicate [`factor_rank_partition`]
    /// resolves fewer directions than the basis has columns. Otherwise the thin-QR owner
    /// orthonormalizes it and the law records that `Q`, so `P = Q Qᵀ` is the
    /// experiment's projector with no orthonormality tolerance.
    pub fn draw(
        baseline: Vec<f64>,
        loading: Vec<f64>,
        rank: usize,
        key: DrawKey,
        draws: usize,
        frames: &[Array2<f64>],
    ) -> Result<Self, InterventionPlanError> {
        let width = baseline.len();
        if width == 0 || rank == 0 || loading.len() != width * rank {
            return Err(InterventionPlanError::InvalidGaussianLaw(format!(
                "the baseline has width {width}, the rank is {rank} and the loading has {} entries; \
                 expected width >= 1, rank >= 1 and width*rank loading entries",
                loading.len()
            )));
        }
        if !baseline.iter().chain(&loading).all(|value| value.is_finite()) {
            return Err(InterventionPlanError::InvalidGaussianLaw(
                "the baseline and loading must be finite".to_string(),
            ));
        }
        let mut orthonormal_frames: Vec<Array2<f64>> = Vec::with_capacity(frames.len());
        for (frame, basis) in frames.iter().enumerate() {
            let (rows, columns) = basis.dim();
            if rows != rank || !basis.iter().all(|value| value.is_finite()) {
                return Err(InterventionPlanError::InvalidFrame {
                    frame,
                    reason: format!(
                        "the basis has shape ({rows}, {columns}) for a law of rank {rank} and must be finite"
                    ),
                });
            }
            if columns == 0 {
                orthonormal_frames.push(Array2::zeros((rank, 0)));
                continue;
            }
            let partition = factor_rank_partition(basis).map_err(|error| {
                InterventionPlanError::InvalidFrame {
                    frame,
                    reason: error.to_string(),
                }
            })?;
            if partition.rank < columns {
                return Err(InterventionPlanError::RankDeficientFrame {
                    frame,
                    rank: partition.rank,
                    columns,
                });
            }
            let orthonormal = basis
                .qr()
                .map_err(|error| InterventionPlanError::InvalidFrame {
                    frame,
                    reason: error.to_string(),
                })?
                .0;
            if orthonormal.dim() != (rank, columns) {
                return Err(InterventionPlanError::InvalidFrame {
                    frame,
                    reason: format!("thin QR returned shape {:?}", orthonormal.dim()),
                });
            }
            orthonormal_frames.push(orthonormal);
        }
        let mut state = splitmix64(key.seed ^ splitmix64(key.stream));
        let mut next =
            || standard_normal_from_uniform_bits(gam_linalg::utils::splitmix64(&mut state));
        let z = std::iter::repeat_with(&mut next)
            .take(draws * rank)
            .collect::<Result<Vec<f64>, String>>()
            .map_err(InterventionPlanError::InvalidGaussianLaw)?;
        let z_tilde = std::iter::repeat_with(&mut next)
            .take(draws * rank)
            .collect::<Result<Vec<f64>, String>>()
            .map_err(InterventionPlanError::InvalidGaussianLaw)?;
        Ok(Self {
            baseline,
            loading,
            rank,
            key,
            draws: z,
            independent_draws: z_tilde,
            frames: orthonormal_frames,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn width(&self) -> usize {
        self.baseline.len()
    }

    pub fn key(&self) -> DrawKey {
        self.key
    }

    /// The number `n` of recorded draws.
    pub fn draw_count(&self) -> usize {
        self.draws.len() / self.rank
    }

    /// The drawn `z`, row-major `(n, rank)`.
    pub fn draws(&self) -> &[f64] {
        &self.draws
    }

    /// The independent copy `z̃`, row-major `(n, rank)`.
    pub fn independent_draws(&self) -> &[f64] {
        &self.independent_draws
    }

    /// The recorded orthonormal frames `Q`, each of shape `(rank, r)`.
    pub fn frames(&self) -> &[Array2<f64>] {
        &self.frames
    }

    /// `z′ = P z + (z̃ − P z̃)` for frame `frame`, row-major `(n, rank)`, or `None` for an
    /// unknown frame. The residual `z̃ − P z̃` is formed first. Where `P` reproduces its
    /// input exactly, as identity-column frames do, a full frame therefore gives `z′ = z`
    /// and a rank-0 frame gives `z′ = z̃`, bit for bit.
    pub fn coupled_draws(&self, frame: usize) -> Option<Vec<f64>> {
        let q = self.frames.get(frame)?;
        let shape = (self.draw_count(), self.rank);
        let z = ArrayView2::from_shape(shape, &self.draws).ok()?;
        let z_tilde = ArrayView2::from_shape(shape, &self.independent_draws).ok()?;
        let residual = &z_tilde - &z_tilde.dot(q).dot(&q.t());
        let coupled = z.dot(q).dot(&q.t()) + residual;
        Some(coupled.iter().copied().collect())
    }

    /// The replacement rows `h0 + L z` (arm `Draw`) or `h0 + L z′` (arm `Coupled`),
    /// row-major `(n, width)`: row `i` is draw `i`. `None` for an unknown frame.
    pub fn rows(&self, arm: LawArm) -> Option<Vec<f64>> {
        match arm {
            LawArm::Draw => self.affine_rows(&self.draws),
            LawArm::Coupled { frame } => self.affine_rows(&self.coupled_draws(frame)?),
        }
    }

    /// `h0 + L x` for each row-major latent row `x` of width `rank`.
    fn affine_rows(&self, latent: &[f64]) -> Option<Vec<f64>> {
        let latent = ArrayView2::from_shape((latent.len() / self.rank, self.rank), latent).ok()?;
        let loading = ArrayView2::from_shape((self.width(), self.rank), &self.loading).ok()?;
        let mut rows = latent.dot(&loading.t());
        for mut row in rows.rows_mut() {
            for (value, &h0) in row.iter_mut().zip(&self.baseline) {
                *value += h0;
            }
        }
        Some(rows.iter().copied().collect())
    }
}

/// Where an ambient replacement's rows come from. This is the fence between an
/// experiment and correlated observational inputs: only [`GaussianLoadingLaw`]
/// rows are randomized.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplacementRows {
    /// Rows the caller declared deterministically, such as a dose ladder, row-major
    /// `(positions, width)`. Unit-level effects are exact; population averages
    /// cover only the declared design.
    Designed(Vec<f64>),
    /// Natural rows read from `source` at `source_positions`, row-major
    /// `(positions, width)`. The unit-level effect is exact, but the rows carry every
    /// feature correlated with their source, so no estimator may attribute the
    /// response to one declared coordinate.
    Observed {
        source: ExperimentUnit,
        source_positions: Vec<usize>,
        rows: Vec<f64>,
    },
    /// Rows of the plan's Gaussian law `law` for arm `arm`: draw `i` replaces the
    /// change's `i`-th position.
    GaussianLaw { law: usize, arm: LawArm },
}

/// One declared change. Every kind is a do-operator on the executed network that
/// does not depend on the patched state, so the runner applies it without
/// computing anything itself.
#[derive(Clone, Debug, PartialEq)]
pub enum InterventionChange {
    /// Set the site's rows at `positions`.
    AmbientReplacement {
        site: usize,
        positions: Vec<usize>,
        rows: ReplacementRows,
    },
    /// Add `delta`, row-major `(positions, width)`, to the site's rows. The
    /// matched-norm controls of #2234 are additions.
    AmbientAddition {
        site: usize,
        positions: Vec<usize>,
        delta: Vec<f64>,
    },
    /// Move one atom's chart coordinate by adding the chord `plan.delta`, priced at
    /// the fitted row. The whole [`SteerPlan`] travels with the move, so the dose
    /// read back is the dose of the delta applied.
    ChartCoordinateMove {
        site: usize,
        position: usize,
        plan: SteerPlan,
    },
    /// Hold coordinate `component` of the site's rows at `value`.
    ComponentWriteClamp {
        site: usize,
        positions: Vec<usize>,
        component: usize,
        value: f64,
    },
    /// Add the rank-`rank` product `left · rightᵀ` (`left` row-major `(rows, rank)`,
    /// `right` row-major `(cols, rank)`) to one named parameter, at the reads `scope`
    /// names.
    ParameterEdit {
        parameter: String,
        rows: usize,
        cols: usize,
        rank: usize,
        left: Vec<f64>,
        right: Vec<f64>,
        scope: ParameterEditScope,
    },
}

/// Which reads of a parameter an [`InterventionChange::ParameterEdit`] reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterEditScope {
    /// Every read of the tensor in the pass. The shared tensor changes for the whole
    /// batch, so the clean response needs its own forward pass.
    Global,
    /// One read of the tensor: the `ordinal`-th read (0-based, forward order) of the
    /// parameter's storage on the plan's declared forward path, seen only by query
    /// `positions` (every position when `None`).
    ///
    /// `read_module` and `read_op` name that read as the executing framework's
    /// discovery reports it: the innermost executing module (`""` for the root
    /// module's own forward, as torch names the root) and the op. The ordinal
    /// addresses the read, and the names validate it, because a (module, call) pair
    /// cannot address a functional read outside a module call, such as a tied
    /// `lm_head` weight read through `F.linear`. An executed experiment is accepted
    /// only when its forward made the ordinal's read by these names
    /// ([`InterventionChange::check_executed_reads`]). The runner applies the edit to
    /// the patched copy alone. The clean copy therefore stays in the same batch, and
    /// under the causal mask every position before the first query position reads
    /// back exactly clean.
    UseSite {
        ordinal: usize,
        read_module: String,
        read_op: String,
        positions: Option<Vec<usize>>,
    },
}

impl InterventionChange {
    /// Check a use-site parameter edit against the reads of the forward that executed
    /// it ([`ParameterEditScope::check_executed_reads`]). Every other change names no
    /// read and passes.
    pub fn check_executed_reads(
        &self,
        executed: &ExecutedParameterReads,
    ) -> Result<(), ParameterReadRefusal> {
        match self {
            InterventionChange::ParameterEdit { parameter, scope, .. } => {
                scope.check_executed_reads(parameter, executed)
            }
            _ => Ok(()),
        }
    }
}

impl ParameterEditScope {
    /// Check this scope of an edit of `parameter` against the reads of the forward that
    /// executed it. A use-site scope passes only when that forward read the parameter's
    /// storage at its ordinal, by the module and op it names; a read by any other names
    /// means the ordinal addressed another read. A global scope names no read and
    /// passes.
    pub fn check_executed_reads(
        &self,
        parameter: &str,
        executed: &ExecutedParameterReads,
    ) -> Result<(), ParameterReadRefusal> {
        let ParameterEditScope::UseSite {
            ordinal,
            read_module,
            read_op,
            ..
        } = self
        else {
            return Ok(());
        };
        let reads = executed
            .reads
            .iter()
            .filter(|read| read.parameter == parameter)
            .count();
        let read = executed
            .reads
            .iter()
            .find(|read| read.parameter == parameter && read.ordinal == *ordinal)
            .ok_or_else(|| ParameterReadRefusal::Unread {
                parameter: parameter.to_string(),
                ordinal: *ordinal,
                reads,
            })?;
        if read.read_module != *read_module || read.read_op != *read_op {
            return Err(ParameterReadRefusal::LabelMismatch {
                parameter: parameter.to_string(),
                ordinal: *ordinal,
                declared_module: read_module.clone(),
                declared_op: read_op.clone(),
                executed_module: read.read_module.clone(),
                executed_op: read.read_op.clone(),
            });
        }
        Ok(())
    }
}

/// One read of a stored parameter in an executed forward, as the executing framework
/// reports it: the storage read (whichever alias the code used), the read's ordinal
/// among that storage's reads in execution order, and the module and op that made it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutedParameterRead {
    pub parameter: String,
    pub ordinal: usize,
    pub read_module: String,
    pub read_op: String,
}

/// Every parameter read of one executed forward, in execution order. The field is
/// private, so every value has passed [`ExecutedParameterReads::new`], and each
/// storage's reads are numbered `0, 1, 2, …` in the order they executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutedParameterReads {
    reads: Vec<ExecutedParameterRead>,
}

impl ExecutedParameterReads {
    /// Refuse a read that names no parameter or no op. Also refuse reads of one storage
    /// that are not numbered in execution order, because under another numbering an
    /// ordinal addresses another read.
    pub fn new(reads: Vec<ExecutedParameterRead>) -> Result<Self, ParameterReadRefusal> {
        let mut next: BTreeMap<&str, usize> = BTreeMap::new();
        for (index, read) in reads.iter().enumerate() {
            if read.parameter.is_empty() || read.read_op.is_empty() {
                return Err(ParameterReadRefusal::Unnamed { index });
            }
            let expected = next.entry(read.parameter.as_str()).or_insert(0);
            if read.ordinal != *expected {
                return Err(ParameterReadRefusal::OutOfOrder {
                    index,
                    parameter: read.parameter.clone(),
                    ordinal: read.ordinal,
                    expected: *expected,
                });
            }
            *expected += 1;
        }
        Ok(Self { reads })
    }

    pub fn reads(&self) -> &[ExecutedParameterRead] {
        &self.reads
    }
}

/// Typed refusals of executed parameter reads, and of a use-site edit checked against
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterReadRefusal {
    /// Read `index` names no parameter or no op.
    Unnamed { index: usize },
    /// Read `index` is not the next read of its storage, so the forward numbered its
    /// reads in another order.
    OutOfOrder {
        index: usize,
        parameter: String,
        ordinal: usize,
        expected: usize,
    },
    /// The executed forward read `parameter` only `reads` times, so it made no read at
    /// the edit's ordinal.
    Unread {
        parameter: String,
        ordinal: usize,
        reads: usize,
    },
    /// The executed read at the edit's ordinal was made by another module or op, so
    /// the ordinal addressed another read.
    LabelMismatch {
        parameter: String,
        ordinal: usize,
        declared_module: String,
        declared_op: String,
        executed_module: String,
        executed_op: String,
    },
}

impl fmt::Display for ParameterReadRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unnamed { index } => write!(
                f,
                "parameter reads: read {index} names no parameter or no op"
            ),
            Self::OutOfOrder {
                index,
                parameter,
                ordinal,
                expected,
            } => write!(
                f,
                "parameter reads: read {index} is {parameter}#{ordinal}, but the next read of {parameter} in execution order is #{expected}"
            ),
            Self::Unread {
                parameter,
                ordinal,
                reads,
            } => write!(
                f,
                "parameter reads: the executed forward read {parameter} {reads} times, so it made no read #{ordinal}"
            ),
            Self::LabelMismatch {
                parameter,
                ordinal,
                declared_module,
                declared_op,
                executed_module,
                executed_op,
            } => write!(
                f,
                "parameter reads: the edit names {parameter}#{ordinal} as ({declared_module:?}, {declared_op:?}), but the executed forward made that read as ({executed_module:?}, {executed_op:?}), so the ordinal addressed another read"
            ),
        }
    }
}

impl std::error::Error for ParameterReadRefusal {}

/// The positions a KL readout reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KlPositions {
    /// Every position some change writes.
    Edited,
    /// Up to `horizon` positions after the last edited one, within the unit. A
    /// causal model's response to an edit lives only there.
    Following { horizon: usize },
    /// Declared positions.
    Declared(Vec<usize>),
}

/// One declared response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readout {
    /// `KL(p_clean ‖ p_patched)` in nats at each resolved position.
    Kl(KlPositions),
    /// Clean and patched log-probabilities of `token_ids` at `positions`.
    TokenLogProb {
        token_ids: Vec<u32>,
        positions: Vec<usize>,
    },
    /// Rows of a declared site at `positions`, as executed in the patched pass.
    Activation { site: usize, positions: Vec<usize> },
}

/// How the clean response is executed. It follows from the changes and is never
/// an option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanPass {
    /// The clean and patched copies share one batched forward pass, so every
    /// position before the earliest edit must read back exactly the clean response.
    SameBatch,
    /// A global parameter edit reaches every position, so the clean pass is its own
    /// forward pass.
    SeparateForward,
}

/// A side of the permanent G2 split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitSide {
    Train,
    EvalForever,
}

/// Several changes applied together in one patched forward pass of one unit,
/// with the responses read from that pass.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionExperiment {
    pub unit: ExperimentUnit,
    pub changes: Vec<InterventionChange>,
    pub readouts: Vec<Readout>,
}

impl InterventionExperiment {
    /// `SeparateForward` exactly when a global parameter edit changes the shared tensor
    /// for the whole batch. Every other change, a use-site edit included, is applied to
    /// the patched copy alone.
    pub fn clean_pass(&self) -> CleanPass {
        if self.changes.iter().any(|change| {
            matches!(
                change,
                InterventionChange::ParameterEdit {
                    scope: ParameterEditScope::Global,
                    ..
                }
            )
        }) {
            CleanPass::SeparateForward
        } else {
            CleanPass::SameBatch
        }
    }

    /// Sorted positions some change writes. A global parameter edit writes no single
    /// position and contributes none. A use-site edit contributes its query positions,
    /// or every position of the unit when it declares none.
    pub fn edited_positions(&self) -> Vec<usize> {
        let mut edited = BTreeSet::new();
        for change in &self.changes {
            match change {
                InterventionChange::AmbientReplacement { positions, .. }
                | InterventionChange::AmbientAddition { positions, .. }
                | InterventionChange::ComponentWriteClamp { positions, .. } => {
                    edited.extend(positions.iter().copied());
                }
                InterventionChange::ChartCoordinateMove { position, .. } => {
                    edited.insert(*position);
                }
                InterventionChange::ParameterEdit { scope, .. } => match scope {
                    ParameterEditScope::Global => {}
                    ParameterEditScope::UseSite {
                        positions: Some(query_positions),
                        ..
                    } => {
                        edited.extend(query_positions.iter().copied());
                    }
                    ParameterEditScope::UseSite {
                        positions: None, ..
                    } => {
                        edited.extend(0..self.unit.length);
                    }
                },
            }
        }
        edited.into_iter().collect()
    }

    /// The positions a KL readout resolves to on this experiment.
    pub fn kl_positions(&self, positions: &KlPositions) -> Vec<usize> {
        match positions {
            KlPositions::Edited => self.edited_positions(),
            KlPositions::Following { horizon } => match self.edited_positions().last() {
                Some(&last) => {
                    (last + 1..self.unit.length.min(last.saturating_add(1).saturating_add(*horizon)))
                        .collect()
                }
                None => Vec::new(),
            },
            KlPositions::Declared(declared) => declared.clone(),
        }
    }
}

/// A validated set of experiments. The fields are private, so every plan a
/// consumer holds has passed [`InterventionExperimentPlan::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionExperimentPlan {
    sites: Vec<InterventionSite>,
    laws: Vec<GaussianLoadingLaw>,
    experiments: Vec<InterventionExperiment>,
    split_seed: u64,
    forward_path: Option<String>,
}

/// Typed refusals of an experiment plan.
#[derive(Clone, Debug, PartialEq)]
pub enum InterventionPlanError {
    InvalidSite {
        site: usize,
    },
    DuplicateSite {
        site: usize,
    },
    NoExperiments,
    EmptyExperiment {
        experiment: usize,
    },
    InvalidChange {
        experiment: usize,
        change: usize,
        reason: String,
    },
    /// Two writes at one site and position do not commute (a replacement with
    /// anything, or a clamp with an addition or the same clamp), so their order
    /// would decide the experiment.
    ConflictingWrites {
        experiment: usize,
        site: usize,
        position: usize,
    },
    /// Observed rows come from a unit on the other side of the G2 split, which
    /// would leak eval-forever information into train.
    StraddlesSplit {
        experiment: usize,
        change: usize,
    },
    /// Two laws share a draw key, so their draws would not be independent.
    RepeatedDrawKey {
        law: usize,
        key: DrawKey,
    },
    /// One arm of one law is executed twice, so its responses could not be paired.
    RepeatedLawArm {
        experiment: usize,
        law: usize,
        arm: LawArm,
    },
    InvalidReadout {
        experiment: usize,
        readout: usize,
        reason: String,
    },
    InvalidGaussianLaw(String),
    InvalidFrame {
        frame: usize,
        reason: String,
    },
    /// The frame basis resolves fewer directions than it has columns, so no projector
    /// of its declared rank exists.
    RankDeficientFrame {
        frame: usize,
        rank: usize,
        columns: usize,
    },
    /// A use-site edit names a read by its ordinal, which only means something on a
    /// declared forward path: KV-cache decode and a full forward can order reads
    /// differently.
    MissingForwardPath {
        experiment: usize,
        change: usize,
    },
}

impl fmt::Display for InterventionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSite { site } => write!(
                f,
                "intervention plan: site {site} needs a module path and a width >= 1"
            ),
            Self::DuplicateSite { site } => {
                write!(f, "intervention plan: site {site} repeats a module path")
            }
            Self::NoExperiments => write!(f, "intervention plan: no experiments"),
            Self::EmptyExperiment { experiment } => write!(
                f,
                "intervention plan: experiment {experiment} needs a change, a readout and a unit length >= 1"
            ),
            Self::InvalidChange {
                experiment,
                change,
                reason,
            } => write!(
                f,
                "intervention plan: experiment {experiment} change {change}: {reason}"
            ),
            Self::ConflictingWrites {
                experiment,
                site,
                position,
            } => write!(
                f,
                "intervention plan: experiment {experiment} writes site {site} at position {position} with changes that do not commute"
            ),
            Self::StraddlesSplit { experiment, change } => write!(
                f,
                "intervention plan: experiment {experiment} change {change} copies rows from a unit on the other side of the G2 split"
            ),
            Self::RepeatedDrawKey { law, key } => write!(
                f,
                "intervention plan: law {law} repeats draw key (seed {}, stream {}), so its draws would not be independent",
                key.seed, key.stream
            ),
            Self::RepeatedLawArm {
                experiment,
                law,
                arm,
            } => write!(
                f,
                "intervention plan: experiment {experiment} executes arm {arm:?} of law {law} a second time, so its responses could not be paired"
            ),
            Self::InvalidReadout {
                experiment,
                readout,
                reason,
            } => write!(
                f,
                "intervention plan: experiment {experiment} readout {readout}: {reason}"
            ),
            Self::InvalidGaussianLaw(reason) => {
                write!(f, "intervention plan: invalid Gaussian law: {reason}")
            }
            Self::InvalidFrame { frame, reason } => {
                write!(f, "intervention plan: invalid frame {frame}: {reason}")
            }
            Self::RankDeficientFrame {
                frame,
                rank,
                columns,
            } => write!(
                f,
                "intervention plan: frame {frame} resolves {rank} directions for {columns} columns, so no projector of its declared rank exists"
            ),
            Self::MissingForwardPath { experiment, change } => write!(
                f,
                "intervention plan: experiment {experiment} change {change} edits a parameter read by ordinal, but the plan declares no forward path the ordinal was recorded on"
            ),
        }
    }
}

impl std::error::Error for InterventionPlanError {}

/// The writes one experiment has made at one (site, position).
#[derive(Default)]
struct SiteWrites {
    replaced: bool,
    added: bool,
    clamped: BTreeSet<usize>,
}

enum WriteKind {
    Replace,
    Add,
    Clamp(usize),
}

/// Record one write, or return false when it does not commute with an earlier one.
fn record_write(
    writes: &mut BTreeMap<(usize, usize), SiteWrites>,
    at: (usize, usize),
    kind: WriteKind,
) -> bool {
    let entry = writes.entry(at).or_default();
    let conflicts = entry.replaced
        || match kind {
            WriteKind::Replace => entry.added || !entry.clamped.is_empty(),
            WriteKind::Add => !entry.clamped.is_empty(),
            WriteKind::Clamp(component) => entry.added || entry.clamped.contains(&component),
        };
    if conflicts {
        return false;
    }
    match kind {
        WriteKind::Replace => entry.replaced = true,
        WriteKind::Add => entry.added = true,
        WriteKind::Clamp(component) => {
            entry.clamped.insert(component);
        }
    }
    true
}

/// Positions must be non-empty, inside the unit and unrepeated: a repeated index
/// in one batched write keeps only its last value.
fn check_positions(positions: &[usize], length: usize) -> Result<(), String> {
    if positions.is_empty() {
        return Err("no positions are declared".to_string());
    }
    let mut seen = BTreeSet::new();
    for &position in positions {
        if position >= length {
            return Err(format!(
                "position {position} is outside the unit of length {length}"
            ));
        }
        if !seen.insert(position) {
            return Err(format!("position {position} is repeated"));
        }
    }
    Ok(())
}

fn check_values(name: &str, values: &[f64], expected: usize) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!(
            "{name} have {} entries; expected {expected}",
            values.len()
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    Ok(())
}

impl InterventionExperimentPlan {
    /// Validate sites, laws and experiments under the permanent split `split_seed`.
    /// `forward_path` identifies the forward on which parameter-read ordinals were
    /// recorded, and every use-site parameter edit requires one.
    pub fn new(
        sites: Vec<InterventionSite>,
        laws: Vec<GaussianLoadingLaw>,
        experiments: Vec<InterventionExperiment>,
        split_seed: u64,
        forward_path: Option<String>,
    ) -> Result<Self, InterventionPlanError> {
        let mut paths = BTreeSet::new();
        for (index, site) in sites.iter().enumerate() {
            if site.module_path.is_empty() || site.width == 0 {
                return Err(InterventionPlanError::InvalidSite { site: index });
            }
            if !paths.insert(site.module_path.as_str()) {
                return Err(InterventionPlanError::DuplicateSite { site: index });
            }
        }
        if experiments.is_empty() {
            return Err(InterventionPlanError::NoExperiments);
        }
        let mut draw_keys = BTreeSet::new();
        for (index, law) in laws.iter().enumerate() {
            if !draw_keys.insert(law.key) {
                return Err(InterventionPlanError::RepeatedDrawKey {
                    law: index,
                    key: law.key,
                });
            }
        }
        let has_forward_path = forward_path.as_ref().is_some_and(|path| !path.is_empty());
        let seed_mix = splitmix64(split_seed);
        let mut law_arms = BTreeSet::new();
        for (index, experiment) in experiments.iter().enumerate() {
            validate_experiment(
                index,
                experiment,
                &sites,
                &laws,
                seed_mix,
                has_forward_path,
                &mut law_arms,
            )?;
        }
        Ok(Self {
            sites,
            laws,
            experiments,
            split_seed,
            forward_path,
        })
    }

    pub fn sites(&self) -> &[InterventionSite] {
        &self.sites
    }

    pub fn laws(&self) -> &[GaussianLoadingLaw] {
        &self.laws
    }

    pub fn experiments(&self) -> &[InterventionExperiment] {
        &self.experiments
    }

    /// The forward path on which parameter-read ordinals were recorded.
    pub fn forward_path(&self) -> Option<&str> {
        self.forward_path.as_deref()
    }

    /// Indices of the experiments on `side` of the permanent split, in plan order.
    /// Membership is [`eval_forever_mask`]'s predicate on the unit's group.
    pub fn experiments_on(&self, side: SplitSide) -> Vec<usize> {
        let seed_mix = splitmix64(self.split_seed);
        let want_eval = side == SplitSide::EvalForever;
        (0..self.experiments.len())
            .filter(|&index| {
                group_is_eval_forever(self.experiments[index].unit.group, seed_mix) == want_eval
            })
            .collect()
    }
}

fn validate_experiment(
    index: usize,
    experiment: &InterventionExperiment,
    sites: &[InterventionSite],
    laws: &[GaussianLoadingLaw],
    seed_mix: u64,
    has_forward_path: bool,
    law_arms: &mut BTreeSet<(usize, LawArm)>,
) -> Result<(), InterventionPlanError> {
    let unit = experiment.unit;
    if experiment.changes.is_empty() || experiment.readouts.is_empty() || unit.length == 0 {
        return Err(InterventionPlanError::EmptyExperiment { experiment: index });
    }
    let unit_is_eval = group_is_eval_forever(unit.group, seed_mix);
    let site_width = |site: usize| sites.get(site).map(|found| found.width);
    let mut writes = BTreeMap::new();
    for (change_index, change) in experiment.changes.iter().enumerate() {
        let invalid = |reason: String| InterventionPlanError::InvalidChange {
            experiment: index,
            change: change_index,
            reason,
        };
        let conflict = |site: usize, position: usize| InterventionPlanError::ConflictingWrites {
            experiment: index,
            site,
            position,
        };
        match change {
            InterventionChange::AmbientReplacement {
                site,
                positions,
                rows,
            } => {
                let width = site_width(*site)
                    .ok_or_else(|| invalid(format!("site {site} is not in the plan")))?;
                check_positions(positions, unit.length).map_err(invalid)?;
                let expected = positions.len() * width;
                match rows {
                    ReplacementRows::Designed(values) => {
                        check_values("designed rows", values, expected).map_err(invalid)?;
                    }
                    ReplacementRows::Observed {
                        source,
                        source_positions,
                        rows: observed,
                    } => {
                        check_values("observed rows", observed, expected).map_err(invalid)?;
                        if source_positions.len() != positions.len()
                            || source_positions
                                .iter()
                                .any(|&position| position >= source.length)
                        {
                            return Err(invalid(format!(
                                "observed rows need one source position inside the source unit per target position; got {} for {}",
                                source_positions.len(),
                                positions.len()
                            )));
                        }
                        if group_is_eval_forever(source.group, seed_mix) != unit_is_eval {
                            return Err(InterventionPlanError::StraddlesSplit {
                                experiment: index,
                                change: change_index,
                            });
                        }
                    }
                    ReplacementRows::GaussianLaw { law, arm } => {
                        let recorded = laws
                            .get(*law)
                            .ok_or_else(|| invalid(format!("law {law} is not in the plan")))?;
                        if recorded.width() != width || recorded.draw_count() != positions.len() {
                            return Err(invalid(format!(
                                "law {law} has width {} and {} draws; the site has width {width} and the change has {} positions",
                                recorded.width(),
                                recorded.draw_count(),
                                positions.len()
                            )));
                        }
                        if let LawArm::Coupled { frame } = arm {
                            if *frame >= recorded.frames.len() {
                                return Err(invalid(format!("law {law} has no frame {frame}")));
                            }
                        }
                        if !law_arms.insert((*law, *arm)) {
                            return Err(InterventionPlanError::RepeatedLawArm {
                                experiment: index,
                                law: *law,
                                arm: *arm,
                            });
                        }
                    }
                }
                for &position in positions {
                    if !record_write(&mut writes, (*site, position), WriteKind::Replace) {
                        return Err(conflict(*site, position));
                    }
                }
            }
            InterventionChange::AmbientAddition {
                site,
                positions,
                delta,
            } => {
                let width = site_width(*site)
                    .ok_or_else(|| invalid(format!("site {site} is not in the plan")))?;
                check_positions(positions, unit.length).map_err(invalid)?;
                check_values("delta rows", delta, positions.len() * width).map_err(invalid)?;
                for &position in positions {
                    if !record_write(&mut writes, (*site, position), WriteKind::Add) {
                        return Err(conflict(*site, position));
                    }
                }
            }
            InterventionChange::ChartCoordinateMove {
                site,
                position,
                plan,
            } => {
                let width = site_width(*site)
                    .ok_or_else(|| invalid(format!("site {site} is not in the plan")))?;
                check_positions(&[*position], unit.length).map_err(invalid)?;
                if plan.delta.len() != width || !plan.delta.iter().all(|value| value.is_finite()) {
                    return Err(invalid(format!(
                        "the chart-move delta has {} entries for a site of width {width} and must be finite",
                        plan.delta.len()
                    )));
                }
                if !record_write(&mut writes, (*site, *position), WriteKind::Add) {
                    return Err(conflict(*site, *position));
                }
            }
            InterventionChange::ComponentWriteClamp {
                site,
                positions,
                component,
                value,
            } => {
                let width = site_width(*site)
                    .ok_or_else(|| invalid(format!("site {site} is not in the plan")))?;
                check_positions(positions, unit.length).map_err(invalid)?;
                if *component >= width || !value.is_finite() {
                    return Err(invalid(format!(
                        "clamp of component {component} to {value} needs a component below the site width {width} and a finite value"
                    )));
                }
                for &position in positions {
                    if !record_write(&mut writes, (*site, position), WriteKind::Clamp(*component))
                    {
                        return Err(conflict(*site, position));
                    }
                }
            }
            InterventionChange::ParameterEdit {
                parameter,
                rows,
                cols,
                rank,
                left,
                right,
                scope,
            } => {
                if parameter.is_empty() || *rank == 0 || *rank > (*rows).min(*cols) {
                    return Err(invalid(format!(
                        "parameter edit {parameter:?} of shape ({rows}, {cols}) needs a name and a rank in 1..=min(rows, cols); got rank {rank}"
                    )));
                }
                check_values("left factor entries", left, rows * rank).map_err(invalid)?;
                check_values("right factor entries", right, cols * rank).map_err(invalid)?;
                if let ParameterEditScope::UseSite { read_op, positions, .. } = scope {
                    // The module may be `""`: torch names the root module so, and a tied
                    // head read through `F.linear` in the root's own forward is made there.
                    if read_op.is_empty() {
                        return Err(invalid(
                            "a use-site parameter edit names the op of its read".to_string(),
                        ));
                    }
                    if let Some(query_positions) = positions {
                        check_positions(query_positions, unit.length).map_err(invalid)?;
                    }
                    if !has_forward_path {
                        return Err(InterventionPlanError::MissingForwardPath {
                            experiment: index,
                            change: change_index,
                        });
                    }
                }
            }
        }
    }
    let clean_pass = experiment.clean_pass();
    for (readout_index, readout) in experiment.readouts.iter().enumerate() {
        let invalid = |reason: String| InterventionPlanError::InvalidReadout {
            experiment: index,
            readout: readout_index,
            reason,
        };
        match readout {
            Readout::Kl(kl) => {
                if let KlPositions::Declared(declared) = kl {
                    check_positions(declared, unit.length).map_err(invalid)?;
                } else if clean_pass == CleanPass::SeparateForward {
                    return Err(invalid(
                        "a global parameter edit reaches every position, so its KL readout must declare positions"
                            .to_string(),
                    ));
                }
                if experiment.kl_positions(kl).is_empty() {
                    return Err(invalid("the KL readout resolves to no position".to_string()));
                }
            }
            Readout::TokenLogProb {
                token_ids,
                positions,
            } => {
                if token_ids.is_empty() {
                    return Err(invalid("no token ids are declared".to_string()));
                }
                check_positions(positions, unit.length).map_err(invalid)?;
            }
            Readout::Activation { site, positions } => {
                if site_width(*site).is_none() {
                    return Err(invalid(format!("site {site} is not in the plan")));
                }
                check_positions(positions, unit.length).map_err(invalid)?;
            }
        }
    }
    Ok(())
}

/// The measured response of one declared readout.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadoutMeasurement {
    /// One KL per resolved position, in nats, stored raw.
    Kl(Vec<f64>),
    /// Log-probabilities, row-major `(positions, token_ids)`.
    TokenLogProb { clean: Vec<f64>, patched: Vec<f64> },
    /// Executed rows of the readout site in the patched pass, row-major
    /// `(positions, width)`. They may come from an external execution, such as a torch
    /// run of the block, as long as row `i` answers the experiment's `i`-th position.
    Activation { rows: Vec<f64> },
}

/// What one executed experiment measured.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentMeasurement {
    /// One measurement per declared readout, in declaration order.
    pub readouts: Vec<ReadoutMeasurement>,
    /// For a same-batch experiment whose earliest edit is after position 0: the
    /// largest KL over the positions before that edit. `None` otherwise.
    pub pre_edit_kl_max: Option<f64>,
    /// Every parameter read the patched forward executed, when the runner recorded
    /// them. An experiment with a use-site parameter edit must carry them, so that each
    /// edit is accepted only where the read it names executed.
    pub parameter_reads: Option<ExecutedParameterReads>,
}

/// A validated plan together with what its execution measured.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutedInterventionExperiments {
    plan: InterventionExperimentPlan,
    measurements: Vec<ExperimentMeasurement>,
}

/// Paired executed responses of one law and frame: row `i` of `at_draw` answers
/// `h0 + L z_i` and row `i` of `at_coupled` answers `h0 + L z′_i`.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLawResponses<'a> {
    /// The law, which records `z`, `z̃`, the orthonormal frames and the draw key.
    pub law: &'a GaussianLoadingLaw,
    pub frame: usize,
    /// Width of the readout rows.
    pub width: usize,
    /// Readout rows at the draws, row-major `(n, width)`.
    pub at_draw: &'a [f64],
    /// Readout rows at the coupled draws, row-major `(n, width)`.
    pub at_coupled: &'a [f64],
}

/// Typed refusals of executed measurements and of their consumers.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutedInterventionError {
    MeasurementCountMismatch {
        expected: usize,
        got: usize,
    },
    InvalidMeasurement {
        experiment: usize,
        reason: String,
    },
    /// A same-batch position before the earliest edit moved. Causal attention
    /// cannot carry an edit backwards and the clean copy rides in the same batch,
    /// so the splice wrote the wrong row or position.
    PreEditResponse {
        experiment: usize,
        kl: f64,
    },
    UnknownSite {
        site: usize,
    },
    /// The plan has no such law, or the law has no such frame.
    UnknownLawFrame {
        law: usize,
        frame: usize,
    },
    /// No experiment on the requested side executes exactly this arm of this law and
    /// nothing else, so no randomized response exists for it.
    MissingLawArm {
        law: usize,
        arm: LawArm,
    },
    MissingActivationReadout {
        experiment: usize,
        site: usize,
    },
    /// An experiment with a use-site parameter edit reported no executed parameter
    /// reads, so nothing shows that its edits reached the reads they name.
    MissingParameterReads {
        experiment: usize,
    },
    /// A use-site parameter edit disagrees with the reads its forward executed.
    ParameterRead {
        experiment: usize,
        change: usize,
        refusal: ParameterReadRefusal,
    },
}

impl fmt::Display for ExecutedInterventionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeasurementCountMismatch { expected, got } => write!(
                f,
                "executed interventions: {got} measurements for {expected} experiments"
            ),
            Self::InvalidMeasurement { experiment, reason } => write!(
                f,
                "executed interventions: experiment {experiment}: {reason}"
            ),
            Self::PreEditResponse { experiment, kl } => write!(
                f,
                "executed interventions: experiment {experiment} moved a position before its earliest edit (KL {kl}); the splice wrote the wrong row or position"
            ),
            Self::UnknownSite { site } => {
                write!(f, "executed interventions: site {site} is not in the plan")
            }
            Self::UnknownLawFrame { law, frame } => write!(
                f,
                "executed interventions: the plan has no law {law} with a frame {frame}"
            ),
            Self::MissingLawArm { law, arm } => write!(
                f,
                "executed interventions: no experiment on this side executes exactly arm {arm:?} of law {law}, so no randomized response exists for it"
            ),
            Self::MissingActivationReadout { experiment, site } => write!(
                f,
                "executed interventions: experiment {experiment} has no activation readout at site {site} over its replaced positions"
            ),
            Self::MissingParameterReads { experiment } => write!(
                f,
                "executed interventions: experiment {experiment} edits a parameter at a use site but reported no executed parameter reads"
            ),
            Self::ParameterRead {
                experiment,
                change,
                refusal,
            } => write!(
                f,
                "executed interventions: experiment {experiment} change {change}: {refusal}"
            ),
        }
    }
}

impl std::error::Error for ExecutedInterventionError {}

impl ExecutedInterventionExperiments {
    /// Validate one measurement per experiment against its declared readouts, and each
    /// use-site parameter edit against the parameter reads its forward executed.
    pub fn new(
        plan: InterventionExperimentPlan,
        measurements: Vec<ExperimentMeasurement>,
    ) -> Result<Self, ExecutedInterventionError> {
        if measurements.len() != plan.experiments.len() {
            return Err(ExecutedInterventionError::MeasurementCountMismatch {
                expected: plan.experiments.len(),
                got: measurements.len(),
            });
        }
        for (index, (experiment, measurement)) in
            plan.experiments.iter().zip(&measurements).enumerate()
        {
            let invalid = |reason: String| ExecutedInterventionError::InvalidMeasurement {
                experiment: index,
                reason,
            };
            if measurement.readouts.len() != experiment.readouts.len() {
                return Err(invalid(format!(
                    "{} readout measurements for {} declared readouts",
                    measurement.readouts.len(),
                    experiment.readouts.len()
                )));
            }
            for (readout, measured) in experiment.readouts.iter().zip(&measurement.readouts) {
                match (readout, measured) {
                    (Readout::Kl(kl), ReadoutMeasurement::Kl(values)) => {
                        check_values("KL values", values, experiment.kl_positions(kl).len())
                            .map_err(invalid)?;
                    }
                    (
                        Readout::TokenLogProb {
                            token_ids,
                            positions,
                        },
                        ReadoutMeasurement::TokenLogProb { clean, patched },
                    ) => {
                        let expected = positions.len() * token_ids.len();
                        check_values("clean log-probabilities", clean, expected).map_err(invalid)?;
                        check_values("patched log-probabilities", patched, expected)
                            .map_err(invalid)?;
                    }
                    (
                        Readout::Activation { site, positions },
                        ReadoutMeasurement::Activation { rows },
                    ) => {
                        check_values(
                            "activation rows",
                            rows,
                            positions.len() * plan.sites[*site].width,
                        )
                        .map_err(invalid)?;
                    }
                    _ => {
                        return Err(invalid(
                            "a readout was measured as a different kind".to_string(),
                        ));
                    }
                }
            }
            match &measurement.parameter_reads {
                Some(executed) => {
                    for (change_index, change) in experiment.changes.iter().enumerate() {
                        change.check_executed_reads(executed).map_err(|refusal| {
                            ExecutedInterventionError::ParameterRead {
                                experiment: index,
                                change: change_index,
                                refusal,
                            }
                        })?;
                    }
                }
                None => {
                    if experiment.changes.iter().any(|change| {
                        matches!(
                            change,
                            InterventionChange::ParameterEdit {
                                scope: ParameterEditScope::UseSite { .. },
                                ..
                            }
                        )
                    }) {
                        return Err(ExecutedInterventionError::MissingParameterReads {
                            experiment: index,
                        });
                    }
                }
            }
            let has_pre_edit_positions = experiment.clean_pass() == CleanPass::SameBatch
                && experiment
                    .edited_positions()
                    .first()
                    .is_some_and(|&earliest| earliest > 0);
            match (has_pre_edit_positions, measurement.pre_edit_kl_max) {
                (true, Some(kl)) => {
                    if kl != 0.0 {
                        return Err(ExecutedInterventionError::PreEditResponse {
                            experiment: index,
                            kl,
                        });
                    }
                }
                (false, None) => {}
                (true, None) => {
                    return Err(invalid(
                        "a same-batch experiment edited after position 0 must report its pre-edit KL"
                            .to_string(),
                    ));
                }
                (false, Some(kl)) => {
                    return Err(invalid(format!(
                        "a pre-edit KL of {kl} was reported for an experiment with no same-batch pre-edit positions"
                    )));
                }
            }
        }
        Ok(Self { plan, measurements })
    }

    pub fn plan(&self) -> &InterventionExperimentPlan {
        &self.plan
    }

    pub fn measurements(&self) -> &[ExperimentMeasurement] {
        &self.measurements
    }

    /// The paired executed responses of law `law` at its draws and at frame `frame`'s
    /// coupled draws, on `side` of the split, read at the activation readout of
    /// `readout_site`. Only an experiment that is exactly one Gaussian-law replacement
    /// executes an arm, so designed or observed rows never enter an estimator as
    /// randomized.
    pub fn gaussian_law_responses(
        &self,
        side: SplitSide,
        readout_site: usize,
        law: usize,
        frame: usize,
    ) -> Result<GaussianLawResponses<'_>, ExecutedInterventionError> {
        let width = self
            .plan
            .sites
            .get(readout_site)
            .map(|site| site.width)
            .ok_or(ExecutedInterventionError::UnknownSite { site: readout_site })?;
        let recorded = self
            .plan
            .laws
            .get(law)
            .filter(|recorded| frame < recorded.frames.len())
            .ok_or(ExecutedInterventionError::UnknownLawFrame { law, frame })?;
        Ok(GaussianLawResponses {
            law: recorded,
            frame,
            width,
            at_draw: self.law_arm_rows(side, readout_site, law, LawArm::Draw)?,
            at_coupled: self.law_arm_rows(side, readout_site, law, LawArm::Coupled { frame })?,
        })
    }

    /// The executed readout rows of the experiment on `side` that executes exactly arm
    /// `arm` of law `law`, read at `readout_site` over the replaced positions.
    fn law_arm_rows(
        &self,
        side: SplitSide,
        readout_site: usize,
        law: usize,
        arm: LawArm,
    ) -> Result<&[f64], ExecutedInterventionError> {
        for index in self.plan.experiments_on(side) {
            let experiment = &self.plan.experiments[index];
            let positions = match experiment.changes.as_slice() {
                [
                    InterventionChange::AmbientReplacement {
                        positions,
                        rows:
                            ReplacementRows::GaussianLaw {
                                law: executed_law,
                                arm: executed_arm,
                            },
                        ..
                    },
                ] if *executed_law == law && *executed_arm == arm => positions,
                _ => continue,
            };
            return experiment
                .readouts
                .iter()
                .zip(&self.measurements[index].readouts)
                .find_map(|(readout, measured)| match (readout, measured) {
                    (
                        Readout::Activation {
                            site,
                            positions: read,
                        },
                        ReadoutMeasurement::Activation { rows },
                    ) if *site == readout_site && read == positions => Some(rows.as_slice()),
                    _ => None,
                })
                .ok_or(ExecutedInterventionError::MissingActivationReadout {
                    experiment: index,
                    site: readout_site,
                });
        }
        Err(ExecutedInterventionError::MissingLawArm { law, arm })
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
            logit_max_abs: vec![12.0; 4],
            logit_max_abs_change: vec![0.5, 0.0, 0.6, 0.0],
            group: vec![10, 10, 20, 20],
            is_control: vec![false, true, false, true],
            layer: 17,
            seed: 0,
            logit_format: LogitFormat::Float32,
            vocab_size: 32,
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
            logit_max_abs: vec![10.0; 6],
            logit_max_abs_change: vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
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
            logit_format: LogitFormat::Float32,
            vocab_size: 16,
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
        assert_eq!(plan.control_quantile_nats, 2.0);
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
    fn deterministic_controls_are_floored_by_the_derived_measurement_band() {
        // Controls on a deterministic model re-splice the unchanged row and
        // measure exactly 0. Before the fix their quantile was the floor, so this
        // fixture was refused with NonPositiveControlFloor(0.0). Each record's
        // floor is now its derived measurement band.
        let (mut shard, spec) = calibration_shard_and_spec();
        shard.nu_measured[0] = 0.0;
        shard.nu_measured[1] = 0.0;
        let band = kl_measurement_band_nats(LogitFormat::Float32, 16, 10.0, 1.0);
        let plan = prepare_intervention_calibration(&shard, spec).unwrap();
        assert_eq!(plan.control_quantile_nats, 0.0);
        assert_eq!(plan.measurement_band_nats_max, band);
        assert!(band > 0.0 && band < 1.0e-12, "band={band}");
        assert_eq!(plan.measurable_atoms, vec![10, 20]);
        assert_eq!(plan.train_atom, vec![10, 20, 10]);
        assert_eq!(plan.train_log_nu, vec![4.0_f64.ln(), 0.0, 2.0_f64.ln()]);

        // A record whose KL does not clear its band is screened out.
        shard.nu_measured[3] = 0.5 * band;
        let screened = prepare_intervention_calibration(&shard, spec).unwrap();
        assert_eq!(screened.measurable_atoms, vec![10]);
        assert_eq!(screened.below_measurement_floor_atoms, vec![20]);
    }

    /// The runner's float64 KL: `log_softmax` of both vectors, `exp` of the
    /// clean one, one sum.
    fn float64_kl(clean: &[f64], patched: &[f64]) -> f64 {
        let log_softmax = |logits: &[f64]| {
            let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = shift
                + logits
                    .iter()
                    .map(|&logit| (logit - shift).exp())
                    .sum::<f64>()
                    .ln();
            logits
                .iter()
                .map(|&logit| logit - log_sum_exp)
                .collect::<Vec<f64>>()
        };
        log_softmax(clean)
            .iter()
            .zip(log_softmax(patched))
            .map(|(&log_p, log_q)| log_p.exp() * (log_p - log_q))
            .sum()
    }

    #[test]
    fn one_spacing_of_logit_rounding_stays_within_the_band_and_a_real_change_exceeds_it() {
        let vocab = 64;
        let clean: Vec<f64> = (0..vocab)
            .map(|v| f64::from((10.0 * (v as f64).sin()) as f32))
            .collect();
        let max_abs = clean.iter().fold(0.0_f64, |largest, &x| largest.max(x.abs()));
        assert!((8.0..16.0).contains(&max_abs), "max_abs={max_abs}");
        assert_eq!(LogitFormat::Float32.spacing_at(max_abs), 2.0_f64.powi(-20));
        let band_of = |patched: &[f64]| {
            let extent = clean
                .iter()
                .chain(patched)
                .fold(0.0_f64, |largest, &x| largest.max(x.abs()));
            let change = clean
                .iter()
                .zip(patched)
                .fold(0.0_f64, |largest, (&a, &b)| largest.max((b - a).abs()));
            kl_measurement_band_nats(LogitFormat::Float32, vocab, extent, change)
        };

        // Worst-case rounding: every logit moved by one f32 spacing, in
        // alternating directions.
        let rounded: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(v, &x)| {
                let x32 = x as f32;
                f64::from(if v % 2 == 0 { x32.next_up() } else { x32.next_down() })
            })
            .collect();
        let rounding_kl = float64_kl(&clean, &rounded);
        let rounding_band = band_of(&rounded);
        assert!(
            rounding_kl <= rounding_band,
            "one-spacing rounding KL {rounding_kl} exceeds its band {rounding_band}"
        );

        // Positive control: a real change of 40 spacings on the largest logit
        // exceeds the band, so the band does not swallow real responses.
        let top = (0..vocab).fold(0, |best, v| if clean[v] > clean[best] { v } else { best });
        let mut changed = clean.clone();
        changed[top] += 40.0 * 2.0_f64.powi(-20);
        let real_kl = float64_kl(&clean, &changed);
        let real_band = band_of(&changed);
        assert!(
            real_kl > real_band,
            "a 40-spacing change KL {real_kl} is inside its band {real_band}"
        );
    }

    #[test]
    fn logit_formats_parse_by_torch_name_and_space_values_exactly() {
        assert_eq!("bfloat16".parse::<LogitFormat>(), Ok(LogitFormat::BFloat16));
        assert!("int8".parse::<LogitFormat>().is_err());
        assert_eq!(LogitFormat::BFloat16.spacing_at(30.0), 0.125);
        assert_eq!(LogitFormat::Float16.spacing_at(30.0), 2.0_f64.powi(-6));
        assert_eq!(LogitFormat::Float64.spacing_at(1.0), f64::EPSILON);
        assert_eq!(
            LogitFormat::Float32.spacing_at(0.0),
            f64::from(f32::from_bits(1))
        );
    }

    #[test]
    fn a_measured_kl_is_refused_only_below_its_evaluation_band() {
        let mut shard = tiny_shard();
        let evaluation_band = kl_evaluation_band_nats(32, 12.0, 0.5);
        shard.nu_measured[0] = -0.5 * evaluation_band;
        assert!(shard.validate().is_ok());
        shard.nu_measured[0] = -2.0 * evaluation_band;
        assert!(shard.validate().unwrap_err().contains("nu_measured"));
        // Extents that break the triangle inequality are refused.
        shard.nu_measured[0] = 0.45;
        shard.logit_max_abs_change[0] = 30.0;
        assert!(shard.validate().unwrap_err().contains("logit extents"));
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

    fn experiment_sites() -> Vec<InterventionSite> {
        vec![
            InterventionSite {
                module_path: "gpt_neox.layers.4.mlp".to_string(),
                width: 2,
            },
            InterventionSite {
                module_path: "gpt_neox.layers.6".to_string(),
                width: 2,
            },
        ]
    }

    fn unit_in(group: i64) -> ExperimentUnit {
        ExperimentUnit {
            group,
            sequence: 3,
            length: 6,
        }
    }

    fn chart_move(delta: [f64; 2]) -> SteerPlan {
        SteerPlan {
            atom: 1,
            atom_name: "month".to_string(),
            t_from: vec![0.1],
            t_to: vec![0.2],
            amplitude: 1.0,
            metric_row: 9,
            delta: ndarray::Array1::from(delta.to_vec()),
            predicted_nats: None,
            predicted_nats_kind: crate::inference::steering::FisherDoseKind::Unavailable,
            fisher_mass_captured: None,
            fisher_mass_residual: None,
            fisher_mass_residual_fraction: None,
            off_manifold_norm: 0.0,
            metric_provenance: gam_problem::MetricProvenance::Euclidean,
        }
    }

    fn two_by_two_law(stream: u64, draws: usize, frames: &[Array2<f64>]) -> GaussianLoadingLaw {
        GaussianLoadingLaw::draw(
            vec![0.5, -0.5],
            vec![1.0, 0.0, 0.0, 2.0],
            2,
            DrawKey { seed: 3, stream },
            draws,
            frames,
        )
        .unwrap()
    }

    #[test]
    fn experiment_carries_several_typed_changes_and_derives_its_clean_pass() {
        let seed = 19;
        let train_group = group_on_side(seed, false);
        let same_batch = InterventionExperiment {
            unit: unit_in(train_group),
            changes: vec![
                InterventionChange::AmbientReplacement {
                    site: 0,
                    positions: vec![2],
                    rows: ReplacementRows::GaussianLaw {
                        law: 0,
                        arm: LawArm::Draw,
                    },
                },
                InterventionChange::AmbientAddition {
                    site: 1,
                    positions: vec![2, 3],
                    delta: vec![0.1, 0.0, 0.0, 0.1],
                },
                InterventionChange::ChartCoordinateMove {
                    site: 1,
                    position: 3,
                    plan: chart_move([0.2, -0.2]),
                },
                InterventionChange::ComponentWriteClamp {
                    site: 1,
                    positions: vec![4],
                    component: 1,
                    value: 0.0,
                },
            ],
            readouts: vec![
                Readout::Kl(KlPositions::Edited),
                Readout::Kl(KlPositions::Following { horizon: 8 }),
                Readout::TokenLogProb {
                    token_ids: vec![11, 12],
                    positions: vec![5],
                },
                Readout::Activation {
                    site: 1,
                    positions: vec![2, 3],
                },
            ],
        };
        let parameter_edit = InterventionExperiment {
            unit: unit_in(train_group),
            changes: vec![InterventionChange::ParameterEdit {
                parameter: "gpt_neox.layers.4.mlp.dense_4h_to_h.weight".to_string(),
                rows: 2,
                cols: 3,
                rank: 1,
                left: vec![1.0, -1.0],
                right: vec![0.5, 0.0, 0.5],
                scope: ParameterEditScope::Global,
            }],
            readouts: vec![Readout::Kl(KlPositions::Declared(vec![0, 5]))],
        };
        let plan = InterventionExperimentPlan::new(
            experiment_sites(),
            vec![two_by_two_law(0, 1, &[])],
            vec![same_batch, parameter_edit],
            seed,
            None,
        )
        .unwrap();
        let experiments = plan.experiments();
        assert_eq!(experiments[0].clean_pass(), CleanPass::SameBatch);
        assert_eq!(experiments[1].clean_pass(), CleanPass::SeparateForward);
        assert_eq!(experiments[0].edited_positions(), vec![2, 3, 4]);
        assert!(experiments[1].edited_positions().is_empty());
        assert_eq!(
            experiments[0].kl_positions(&KlPositions::Following { horizon: 8 }),
            vec![5]
        );
        assert_eq!(plan.experiments_on(SplitSide::Train), vec![0, 1]);
        assert!(plan.experiments_on(SplitSide::EvalForever).is_empty());
    }

    #[test]
    fn non_commuting_writes_at_one_site_and_position_are_refused() {
        let addition = |position: usize| InterventionChange::AmbientAddition {
            site: 1,
            positions: vec![position],
            delta: vec![0.1, 0.2],
        };
        let plan_with = |second: InterventionChange| {
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: unit_in(4),
                    changes: vec![addition(2), second],
                    readouts: vec![Readout::Kl(KlPositions::Edited)],
                }],
                0,
                None,
            )
        };
        // Additions commute, and a clamp at another site does not overlap.
        assert!(plan_with(addition(2)).is_ok());
        assert!(
            plan_with(InterventionChange::ComponentWriteClamp {
                site: 0,
                positions: vec![2],
                component: 0,
                value: 1.0,
            })
            .is_ok()
        );
        let conflict = InterventionPlanError::ConflictingWrites {
            experiment: 0,
            site: 1,
            position: 2,
        };
        assert_eq!(
            plan_with(InterventionChange::AmbientReplacement {
                site: 1,
                positions: vec![2],
                rows: ReplacementRows::Designed(vec![0.0, 0.0]),
            })
            .unwrap_err(),
            conflict
        );
        assert_eq!(
            plan_with(InterventionChange::ComponentWriteClamp {
                site: 1,
                positions: vec![2],
                component: 0,
                value: 1.0,
            })
            .unwrap_err(),
            conflict
        );
    }

    #[test]
    fn observed_rows_from_across_the_split_are_refused() {
        let seed = 19;
        let train_group = group_on_side(seed, false);
        let eval_group = group_on_side(seed, true);
        let patch_from = |source_group: i64| {
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: unit_in(train_group),
                    changes: vec![InterventionChange::AmbientReplacement {
                        site: 0,
                        positions: vec![1],
                        rows: ReplacementRows::Observed {
                            source: unit_in(source_group),
                            source_positions: vec![4],
                            rows: vec![0.3, -0.3],
                        },
                    }],
                    readouts: vec![Readout::Kl(KlPositions::Following { horizon: 2 })],
                }],
                seed,
                None,
            )
        };
        assert!(patch_from(train_group).is_ok());
        assert_eq!(
            patch_from(eval_group).unwrap_err(),
            InterventionPlanError::StraddlesSplit {
                experiment: 0,
                change: 0,
            }
        );
    }

    /// An experiment that replaces positions 0 and 1 of site 0 with `rows` and reads
    /// site 1 back over the same positions.
    fn law_arm_experiment(group: i64, rows: ReplacementRows) -> InterventionExperiment {
        InterventionExperiment {
            unit: unit_in(group),
            changes: vec![InterventionChange::AmbientReplacement {
                site: 0,
                positions: vec![0, 1],
                rows,
            }],
            readouts: vec![Readout::Activation {
                site: 1,
                positions: vec![0, 1],
            }],
        }
    }

    fn activation_rows(rows: [f64; 4]) -> ExperimentMeasurement {
        ExperimentMeasurement {
            readouts: vec![ReadoutMeasurement::Activation {
                rows: rows.to_vec(),
            }],
            pre_edit_kl_max: None,
            parameter_reads: None,
        }
    }

    #[test]
    fn gaussian_law_responses_pair_draw_and_coupled_rows_and_refuse_designed_rows() {
        let train_group = group_on_side(0, false);
        let frame = ndarray::array![[1.0], [1.0]];
        let law = || two_by_two_law(0, 2, std::slice::from_ref(&frame));
        let draw_arm = || ReplacementRows::GaussianLaw {
            law: 0,
            arm: LawArm::Draw,
        };
        let coupled_arm = || ReplacementRows::GaussianLaw {
            law: 0,
            arm: LawArm::Coupled { frame: 0 },
        };
        let plan = InterventionExperimentPlan::new(
            experiment_sites(),
            vec![law()],
            vec![
                law_arm_experiment(train_group, draw_arm()),
                law_arm_experiment(train_group, coupled_arm()),
            ],
            0,
            None,
        )
        .unwrap();
        let executed = ExecutedInterventionExperiments::new(
            plan,
            vec![
                activation_rows([1.0, 2.0, 3.0, 4.0]),
                activation_rows([5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        let paired = executed
            .gaussian_law_responses(SplitSide::Train, 1, 0, 0)
            .unwrap();
        assert_eq!(paired.width, 2);
        assert_eq!(paired.frame, 0);
        assert_eq!(paired.law, &law());
        assert_eq!(paired.at_draw, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(paired.at_coupled, &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            executed
                .gaussian_law_responses(SplitSide::EvalForever, 1, 0, 0)
                .unwrap_err(),
            ExecutedInterventionError::MissingLawArm {
                law: 0,
                arm: LawArm::Draw,
            }
        );
        assert_eq!(
            executed
                .gaussian_law_responses(SplitSide::Train, 1, 0, 1)
                .unwrap_err(),
            ExecutedInterventionError::UnknownLawFrame { law: 0, frame: 1 }
        );

        // Positive control: identical rows declared as designed execute no law arm, so
        // the fence is the type, not the values.
        let designed = InterventionExperimentPlan::new(
            experiment_sites(),
            vec![law()],
            vec![
                law_arm_experiment(
                    train_group,
                    ReplacementRows::Designed(law().rows(LawArm::Draw).unwrap()),
                ),
                law_arm_experiment(train_group, coupled_arm()),
            ],
            0,
            None,
        )
        .unwrap();
        let executed_designed = ExecutedInterventionExperiments::new(
            designed,
            vec![
                activation_rows([1.0, 2.0, 3.0, 4.0]),
                activation_rows([5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        assert_eq!(
            executed_designed
                .gaussian_law_responses(SplitSide::Train, 1, 0, 0)
                .unwrap_err(),
            ExecutedInterventionError::MissingLawArm {
                law: 0,
                arm: LawArm::Draw,
            }
        );

        // An arm executed twice could not be paired, and two laws sharing a key would
        // share their draws.
        assert_eq!(
            InterventionExperimentPlan::new(
                experiment_sites(),
                vec![law()],
                vec![
                    law_arm_experiment(train_group, draw_arm()),
                    law_arm_experiment(train_group, draw_arm()),
                ],
                0,
                None,
            )
            .unwrap_err(),
            InterventionPlanError::RepeatedLawArm {
                experiment: 1,
                law: 0,
                arm: LawArm::Draw,
            }
        );
        assert_eq!(
            InterventionExperimentPlan::new(
                experiment_sites(),
                vec![law(), law()],
                vec![law_arm_experiment(train_group, draw_arm())],
                0,
                None,
            )
            .unwrap_err(),
            InterventionPlanError::RepeatedDrawKey {
                law: 1,
                key: DrawKey { seed: 3, stream: 0 },
            }
        );
    }

    #[test]
    fn coupled_draws_copy_retained_coordinates_and_take_the_rest_from_the_independent_copy() {
        let frames = [
            Array2::eye(2),
            ndarray::array![[1.0], [0.0]],
            Array2::zeros((2, 0)),
        ];
        let law = two_by_two_law(7, 64, &frames);
        assert_ne!(law.draws(), law.independent_draws());
        // The full frame reproduces z bit for bit.
        assert_eq!(law.coupled_draws(0).unwrap(), law.draws());
        assert_eq!(
            law.rows(LawArm::Coupled { frame: 0 }),
            law.rows(LawArm::Draw)
        );
        // The rank-0 frame executes the independent copy bit for bit.
        assert_eq!(law.coupled_draws(2).unwrap(), law.independent_draws());
        // The first-axis frame keeps z's first coordinate and z̃'s second.
        let partial = law.coupled_draws(1).unwrap();
        for ((coupled, z), z_tilde) in partial
            .chunks_exact(2)
            .zip(law.draws().chunks_exact(2))
            .zip(law.independent_draws().chunks_exact(2))
        {
            assert_eq!(coupled, &[z[0], z_tilde[1]]);
        }
        assert!(law.coupled_draws(3).is_none());
    }

    #[test]
    fn a_rank_deficient_or_misshaped_frame_is_refused() {
        let draw_with = |frame: Array2<f64>| {
            GaussianLoadingLaw::draw(
                vec![0.5, -0.5],
                vec![1.0, 0.0, 0.0, 2.0],
                2,
                DrawKey { seed: 3, stream: 0 },
                4,
                &[frame],
            )
        };
        assert_eq!(
            draw_with(ndarray::array![[1.0, 0.0], [1.0, 0.0]]).unwrap_err(),
            InterventionPlanError::RankDeficientFrame {
                frame: 0,
                rank: 1,
                columns: 2,
            }
        );
        assert!(matches!(
            draw_with(Array2::ones((3, 1))),
            Err(InterventionPlanError::InvalidFrame { frame: 0, .. })
        ));
        // Positive control: the same columns made independent are accepted.
        assert!(draw_with(ndarray::array![[1.0, 0.0], [1.0, 1.0]]).is_ok());
    }

    /// The Kolmogorov distance between the empirical law of `values` and the standard
    /// normal, with the Dvoretzky-Kiefer-Wolfowitz bound (Massart's constant) it must
    /// stay within at false-failure probability 1e-9.
    fn kolmogorov_distance_and_dkw_bound(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let distance = sorted
            .iter()
            .enumerate()
            .fold(0.0_f64, |distance, (index, &value)| {
                let cdf = gam_math::probability::normal_cdf(value);
                distance
                    .max(cdf - index as f64 / n)
                    .max((index + 1) as f64 / n - cdf)
            });
        (distance, ((2.0 / 1.0e-9_f64).ln() / (2.0 * n)).sqrt())
    }

    #[test]
    fn coupled_draws_are_standard_normal_and_share_the_frame_component() {
        let n = 20_000;
        let law = two_by_two_law(
            11,
            n,
            &[ndarray::array![[1.0], [1.0]], Array2::zeros((2, 0))],
        );
        let coupled = law.coupled_draws(0).unwrap();
        for axis in 0..2 {
            let marginal: Vec<f64> = coupled.iter().skip(axis).step_by(2).copied().collect();
            let (distance, bound) = kolmogorov_distance_and_dkw_bound(&marginal);
            assert!(
                distance <= bound,
                "coupled axis {axis} is not standard normal: {distance} > {bound}"
            );
        }
        // Positive control: adding z̃ without removing its frame component gives variance
        // 3/2 along each axis, which the same bound detects.
        let q = &law.frames()[0];
        let unremoved: Vec<f64> = law
            .draws()
            .chunks_exact(2)
            .zip(law.independent_draws().chunks_exact(2))
            .map(|(z, z_tilde)| q[[0, 0]] * (q[[0, 0]] * z[0] + q[[1, 0]] * z[1]) + z_tilde[0])
            .collect();
        let (distance, bound) = kolmogorov_distance_and_dkw_bound(&unremoved);
        assert!(
            distance > bound,
            "an unremoved frame component was not detected: {distance} <= {bound}"
        );

        // Z′ shares exactly P Z with Z, so on axis 0 its correlation with z is P₀₀ = 1/2.
        // The sample correlation must match within the normal-approximation standard
        // error (1 − ρ²)/√n times the two-sided 1e-9 quantile.
        let axis_zero = |values: &[f64]| values.iter().step_by(2).copied().collect::<Vec<f64>>();
        let sample_correlation = |a: &[f64], b: &[f64]| {
            let count = a.len() as f64;
            let mean_a = a.iter().sum::<f64>() / count;
            let mean_b = b.iter().sum::<f64>() / count;
            let (cross, square_a, square_b) = a.iter().zip(b).fold(
                (0.0_f64, 0.0_f64, 0.0_f64),
                |(cross, square_a, square_b), (&x, &y)| {
                    let (dx, dy) = (x - mean_a, y - mean_b);
                    (cross + dx * dy, square_a + dx * dx, square_b + dy * dy)
                },
            );
            cross / (square_a * square_b).sqrt()
        };
        let quantile = gam_math::probability::standard_normal_quantile(1.0 - 0.5e-9).unwrap();
        let tolerance = quantile * (1.0 - 0.25) / (n as f64).sqrt();
        let draws_axis_zero = axis_zero(law.draws());
        let shared = sample_correlation(&axis_zero(&coupled), &draws_axis_zero);
        assert!(
            (shared - 0.5).abs() <= tolerance,
            "the coupled draws do not share the frame component: correlation {shared}, tolerance {tolerance}"
        );
        // Positive control: the rank-0 frame shares nothing, and the same tolerance
        // detects it.
        let independent = sample_correlation(
            &axis_zero(&law.coupled_draws(1).unwrap()),
            &draws_axis_zero,
        );
        assert!(
            (independent - 0.5).abs() > tolerance,
            "an independent copy was not detected: correlation {independent}, tolerance {tolerance}"
        );
    }

    #[test]
    fn gaussian_law_rows_are_baseline_plus_loading_times_standard_normal_draws() {
        let law = two_by_two_law(5, 20_000, &[]);
        for (row, z) in law
            .rows(LawArm::Draw)
            .unwrap()
            .chunks_exact(2)
            .zip(law.draws().chunks_exact(2))
        {
            assert_eq!(row[0], 0.5 + z[0]);
            assert_eq!(row[1], -0.5 + 2.0 * z[1]);
        }
        let (distance, bound) = kolmogorov_distance_and_dkw_bound(law.draws());
        assert!(
            distance <= bound,
            "the draws are not standard normal: Kolmogorov distance {distance} > {bound}"
        );
        // Positive control: the same draws shifted by a tenth of a standard deviation
        // exceed the bound, so the bound can detect a wrong stream.
        let shifted: Vec<f64> = law.draws().iter().map(|z| z + 0.1).collect();
        let (shifted_distance, shifted_bound) = kolmogorov_distance_and_dkw_bound(&shifted);
        assert!(
            shifted_distance > shifted_bound,
            "a 0.1 shift was not detected: {shifted_distance} <= {shifted_bound}"
        );
    }

    #[test]
    fn a_same_batch_response_before_the_earliest_edit_is_refused() {
        let plan = || {
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: unit_in(4),
                    changes: vec![InterventionChange::AmbientAddition {
                        site: 1,
                        positions: vec![3],
                        delta: vec![0.1, 0.2],
                    }],
                    readouts: vec![
                        Readout::Kl(KlPositions::Edited),
                        Readout::Kl(KlPositions::Following { horizon: 2 }),
                    ],
                }],
                0,
                None,
            )
            .unwrap()
        };
        let measured = |pre_edit_kl_max: Option<f64>| {
            vec![ExperimentMeasurement {
                readouts: vec![
                    ReadoutMeasurement::Kl(vec![0.4]),
                    ReadoutMeasurement::Kl(vec![0.02, 0.01]),
                ],
                pre_edit_kl_max,
                parameter_reads: None,
            }]
        };
        assert!(ExecutedInterventionExperiments::new(plan(), measured(Some(0.0))).is_ok());
        assert_eq!(
            ExecutedInterventionExperiments::new(plan(), measured(Some(1.0e-12))).unwrap_err(),
            ExecutedInterventionError::PreEditResponse {
                experiment: 0,
                kl: 1.0e-12,
            }
        );
        assert!(matches!(
            ExecutedInterventionExperiments::new(plan(), measured(None)),
            Err(ExecutedInterventionError::InvalidMeasurement { experiment: 0, .. })
        ));
        // A future-position readout measured at the wrong length is refused.
        let mut short = measured(Some(0.0));
        short[0].readouts[1] = ReadoutMeasurement::Kl(vec![0.02]);
        assert!(matches!(
            ExecutedInterventionExperiments::new(plan(), short),
            Err(ExecutedInterventionError::InvalidMeasurement { experiment: 0, .. })
        ));
    }

    #[test]
    fn a_parameter_edit_must_declare_its_kl_positions() {
        let edit = |readout: Readout| {
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: unit_in(4),
                    changes: vec![InterventionChange::ParameterEdit {
                        parameter: "embed_out.weight".to_string(),
                        rows: 2,
                        cols: 2,
                        rank: 1,
                        left: vec![1.0, 0.0],
                        right: vec![0.0, 1.0],
                        scope: ParameterEditScope::Global,
                    }],
                    readouts: vec![readout],
                }],
                0,
                None,
            )
        };
        assert!(edit(Readout::Kl(KlPositions::Declared(vec![5]))).is_ok());
        assert!(matches!(
            edit(Readout::Kl(KlPositions::Edited)),
            Err(InterventionPlanError::InvalidReadout {
                experiment: 0,
                readout: 0,
                ..
            })
        ));
    }

    #[test]
    fn a_use_site_parameter_edit_keeps_the_clean_prefix_in_the_batch() {
        let experiment = |scope: ParameterEditScope, readout: Readout| InterventionExperiment {
            unit: unit_in(4),
            changes: vec![InterventionChange::ParameterEdit {
                parameter: "gpt_neox.embed_out.weight".to_string(),
                rows: 2,
                cols: 2,
                rank: 1,
                left: vec![1.0, 0.0],
                right: vec![0.0, 1.0],
                scope,
            }],
            readouts: vec![readout],
        };
        let use_site = |positions: Option<Vec<usize>>| ParameterEditScope::UseSite {
            ordinal: 1,
            read_module: "embed_out".to_string(),
            read_op: "F.linear".to_string(),
            positions,
        };
        let forward_path = || Some("full-forward:len6".to_string());
        let following = || Readout::Kl(KlPositions::Following { horizon: 2 });
        let plan = InterventionExperimentPlan::new(
            experiment_sites(),
            Vec::new(),
            vec![experiment(use_site(Some(vec![3])), following())],
            0,
            forward_path(),
        )
        .unwrap();
        assert_eq!(plan.experiments()[0].clean_pass(), CleanPass::SameBatch);
        assert_eq!(plan.experiments()[0].edited_positions(), vec![3]);
        assert_eq!(plan.forward_path(), Some("full-forward:len6"));

        // Positions before the first query position must read back exactly clean.
        let executed = ExecutedParameterReads::new(vec![
            ExecutedParameterRead {
                parameter: "gpt_neox.embed_out.weight".to_string(),
                ordinal: 0,
                read_module: "gpt_neox.embed_in".to_string(),
                read_op: "F.embedding".to_string(),
            },
            ExecutedParameterRead {
                parameter: "gpt_neox.embed_out.weight".to_string(),
                ordinal: 1,
                read_module: "embed_out".to_string(),
                read_op: "F.linear".to_string(),
            },
        ])
        .unwrap();
        let measured = |pre_edit_kl_max: Option<f64>| {
            vec![ExperimentMeasurement {
                readouts: vec![ReadoutMeasurement::Kl(vec![0.02, 0.01])],
                pre_edit_kl_max,
                parameter_reads: Some(executed.clone()),
            }]
        };
        assert!(ExecutedInterventionExperiments::new(plan.clone(), measured(Some(0.0))).is_ok());
        assert_eq!(
            ExecutedInterventionExperiments::new(plan, measured(Some(1.0e-12))).unwrap_err(),
            ExecutedInterventionError::PreEditResponse {
                experiment: 0,
                kl: 1.0e-12,
            }
        );

        // A read ordinal means nothing without the forward path it was recorded on.
        assert_eq!(
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![experiment(use_site(Some(vec![3])), following())],
                0,
                None,
            )
            .unwrap_err(),
            InterventionPlanError::MissingForwardPath {
                experiment: 0,
                change: 0,
            }
        );

        // A use-site edit read at every position edits every position.
        let everywhere = InterventionExperimentPlan::new(
            experiment_sites(),
            Vec::new(),
            vec![experiment(use_site(None), Readout::Kl(KlPositions::Edited))],
            0,
            forward_path(),
        )
        .unwrap();
        assert_eq!(
            everywhere.experiments()[0].edited_positions(),
            (0..6).collect::<Vec<usize>>()
        );

        // Negative control: a global edit of the same tensor changes the whole batch.
        let global = InterventionExperimentPlan::new(
            experiment_sites(),
            Vec::new(),
            vec![experiment(
                ParameterEditScope::Global,
                Readout::Kl(KlPositions::Declared(vec![5])),
            )],
            0,
            None,
        )
        .unwrap();
        assert_eq!(
            global.experiments()[0].clean_pass(),
            CleanPass::SeparateForward
        );
    }

    #[test]
    fn a_use_site_edit_is_accepted_only_where_its_forward_made_the_named_read() {
        // A tied embedding: read #0 by the embedding module, read #1 as the head through
        // F.linear in the root module's own forward, which torch names "".
        let tied = "embed.weight";
        let read = |parameter: &str, ordinal: usize, module: &str, op: &str| {
            ExecutedParameterRead {
                parameter: parameter.to_string(),
                ordinal,
                read_module: module.to_string(),
                read_op: op.to_string(),
            }
        };
        let linear = "torch.nn.functional.linear";
        let forward = || {
            vec![
                read(tied, 0, "embed", "torch.nn.functional.embedding"),
                read("block.mlp.weight", 0, "block.mlp", linear),
                read(tied, 1, "", linear),
            ]
        };
        let plan = |scope: ParameterEditScope, readout: Readout| {
            InterventionExperimentPlan::new(
                experiment_sites(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: unit_in(4),
                    changes: vec![InterventionChange::ParameterEdit {
                        parameter: tied.to_string(),
                        rows: 2,
                        cols: 2,
                        rank: 1,
                        left: vec![1.0, 0.0],
                        right: vec![0.0, 1.0],
                        scope,
                    }],
                    readouts: vec![readout],
                }],
                0,
                Some("full-forward:len6".to_string()),
            )
        };
        let head = |ordinal: usize, module: &str, op: &str| ParameterEditScope::UseSite {
            ordinal,
            read_module: module.to_string(),
            read_op: op.to_string(),
            positions: None,
        };
        // Every position is edited, so the KL readout has one value per position.
        let execute = |scope: ParameterEditScope, reads: Option<Vec<ExecutedParameterRead>>| {
            ExecutedInterventionExperiments::new(
                plan(scope, Readout::Kl(KlPositions::Edited)).unwrap(),
                vec![ExperimentMeasurement {
                    readouts: vec![ReadoutMeasurement::Kl(vec![0.1; 6])],
                    pre_edit_kl_max: None,
                    parameter_reads: reads
                        .map(|reads| ExecutedParameterReads::new(reads).unwrap()),
                }],
            )
        };
        let mismatch = |ordinal: usize, declared: (&str, &str), executed: (&str, &str)| {
            ExecutedInterventionError::ParameterRead {
                experiment: 0,
                change: 0,
                refusal: ParameterReadRefusal::LabelMismatch {
                    parameter: tied.to_string(),
                    ordinal,
                    declared_module: declared.0.to_string(),
                    declared_op: declared.1.to_string(),
                    executed_module: executed.0.to_string(),
                    executed_op: executed.1.to_string(),
                },
            }
        };

        // The head read, named as the forward made it, root module "" included.
        assert!(execute(head(1, "", linear), Some(forward())).is_ok());
        // Another op, or another module, at that ordinal is another read.
        assert_eq!(
            execute(head(1, "", "torch.matmul"), Some(forward())).unwrap_err(),
            mismatch(1, ("", "torch.matmul"), ("", linear))
        );
        assert_eq!(
            execute(head(1, "lm_head", linear), Some(forward())).unwrap_err(),
            mismatch(1, ("lm_head", linear), ("", linear))
        );
        // An ordinal off by one addresses the embedding read, which the names catch.
        assert_eq!(
            execute(head(0, "", linear), Some(forward())).unwrap_err(),
            mismatch(0, ("", linear), ("embed", "torch.nn.functional.embedding"))
        );
        // The forward read the tied tensor twice, so it made no read #2.
        assert_eq!(
            execute(head(2, "", linear), Some(forward())).unwrap_err(),
            ExecutedInterventionError::ParameterRead {
                experiment: 0,
                change: 0,
                refusal: ParameterReadRefusal::Unread {
                    parameter: tied.to_string(),
                    ordinal: 2,
                    reads: 2,
                },
            }
        );
        // Without the executed reads, nothing shows the edit reached its read.
        assert_eq!(
            execute(head(1, "", linear), None).unwrap_err(),
            ExecutedInterventionError::MissingParameterReads { experiment: 0 }
        );
        // Negative control: a global edit names no read, so it needs no reads.
        assert!(ExecutedInterventionExperiments::new(
            plan(
                ParameterEditScope::Global,
                Readout::Kl(KlPositions::Declared(vec![5]))
            )
            .unwrap(),
            vec![ExperimentMeasurement {
                readouts: vec![ReadoutMeasurement::Kl(vec![0.1])],
                pre_edit_kl_max: None,
                parameter_reads: None,
            }],
        )
        .is_ok());
        // The op is required even where the module is the root's "".
        assert!(matches!(
            plan(head(1, "", ""), Readout::Kl(KlPositions::Edited)),
            Err(InterventionPlanError::InvalidChange {
                experiment: 0,
                change: 0,
                ..
            })
        ));

        // Each storage is numbered in execution order on its own, so the interleaved
        // forward is accepted. A read out of order, a repeated ordinal and a read with
        // no op are refused.
        assert!(ExecutedParameterReads::new(forward()).is_ok());
        assert_eq!(
            ExecutedParameterReads::new(vec![
                read(tied, 1, "", linear),
                read(tied, 0, "embed", linear),
            ])
            .unwrap_err(),
            ParameterReadRefusal::OutOfOrder {
                index: 0,
                parameter: tied.to_string(),
                ordinal: 1,
                expected: 0,
            }
        );
        assert_eq!(
            ExecutedParameterReads::new(vec![
                read(tied, 0, "embed", linear),
                read(tied, 0, "", linear),
            ])
            .unwrap_err(),
            ParameterReadRefusal::OutOfOrder {
                index: 1,
                parameter: tied.to_string(),
                ordinal: 0,
                expected: 1,
            }
        );
        assert_eq!(
            ExecutedParameterReads::new(vec![read(tied, 0, "embed", "")]).unwrap_err(),
            ParameterReadRefusal::Unnamed { index: 0 }
        );
    }
}
