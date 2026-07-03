//! Rung-3 intervention shard — the validated data contract between the Python
//! patch runner (the model-interaction boundary) and the Rust calibration fit.
//!
//! See `RUNG3_INTERVENTIONS_DESIGN.md` (§6) in this directory. One record per
//! executed intervention: `(token row, atom, dose Δt)` with the *predicted*
//! nats (`ν̂₁` from the Rung-1 behavioral-Fisher metric, `ν̂₂` from the Rung-2
//! behavior decoder when a y-block exists) and the *measured* realized KL. The
//! `.npz` I/O lives at the Python boundary (mirroring the harvest-shard
//! discipline of `gamfit/torch/harvest.py` / `gamfit/torch/interventions.py`);
//! this type owns validation and the **G2 eval-forever split**.
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
//! calibration driver implements the *same* function
//! (`gamfit/intervention_calibration.py::_splitmix64_parity`) so the two sides
//! agree on the manifest without shipping one.

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

/// SplitMix64 — the split's hash. A fixed, well-known mixing function so the
/// group→side assignment is reproducible across languages and releases.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
            if splitmix64((g as u64) ^ seed_mix) & 1 == 1 {
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
        let mut nulls: Vec<f64> = self
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
        nulls.sort_by(|a, b| a.partial_cmp(b).expect("validated finite"));
        // Inclusive linear-interpolation quantile (the same convention as
        // numpy's default), on the validated finite sample.
        let h = q * (nulls.len() as f64 - 1.0);
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        let frac = h - lo as f64;
        Ok(nulls[lo] * (1.0 - frac) + nulls[hi] * frac)
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
        // The Python driver mirrors these exact values
        // (gamfit/intervention_calibration.py); a change here is a contract
        // break, not a refactor.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(1), 0x910A_2DEC_8902_5CC1);
    }
}
