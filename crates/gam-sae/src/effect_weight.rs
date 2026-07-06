//! Effect-weighted atom retention and fit-quality reporting.
//!
//! Reconstruction EV is a distributional currency: an atom that fires rarely can
//! explain almost no variance even when ablating it changes the downstream
//! distribution sharply. This module keeps the two ledgers separate. The
//! variance/rank-charge decision remains available, and a behavioral-effect
//! decision is added beside it. An atom is retained when either ledger pays.

use crate::inference::intervention_shard::InterventionShard;

/// Per-atom evidence in the existing reconstruction currency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceChargeEvidence {
    /// Reconstruction deviance reduction claimed by the atom, in nats.
    pub delta_deviance: f64,
    /// Realised-rank evidence price, in nats.
    pub charge: f64,
}

impl VarianceChargeEvidence {
    pub fn margin(self) -> f64 {
        self.delta_deviance - self.charge
    }

    pub fn retains(self) -> bool {
        self.margin() > 0.0
    }
}

/// Behavioral effect evidence for one atom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectDoseEvidence {
    /// Atom index.
    pub atom: usize,
    /// Mean realised KL over non-control interventions for this atom.
    pub mean_realized_kl_nats: f64,
    /// Largest realised KL over non-control interventions for this atom.
    pub max_realized_kl_nats: f64,
    /// Number of non-control interventions that touched this atom.
    pub n_interventions: usize,
    /// Derived discovery threshold in nats. This is the one-degree BIC price for
    /// the intervention sample size: 0.5 * ln(max(n_interventions, 2)).
    pub threshold_nats: f64,
}

impl EffectDoseEvidence {
    pub fn margin(self) -> f64 {
        self.mean_realized_kl_nats - self.threshold_nats
    }

    pub fn retains(self) -> bool {
        self.margin() > 0.0
    }
}

/// Full retention verdict for one atom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomRetentionEvidence {
    pub atom: usize,
    pub variance: Option<VarianceChargeEvidence>,
    pub effect: Option<EffectDoseEvidence>,
    pub retained_by_variance: bool,
    pub retained_by_effect: bool,
    pub retained: bool,
}

/// Compute the effect ledger from realised Rung-3 KL measurements.
///
/// Controls are excluded: they estimate measurement floor elsewhere and do not
/// represent an ablation/interchange dose. The retained effect score is the mean
/// realised KL per executed non-control intervention for the atom, so rare atoms
/// are judged by the behavioral swing when they are actually probed rather than
/// diluted by corpus frequency.
pub fn effect_dose_from_interventions(
    atom_count: usize,
    shard: &InterventionShard,
) -> Result<Vec<Option<EffectDoseEvidence>>, String> {
    shard.validate()?;
    let mut sums = vec![0.0_f64; atom_count];
    let mut maxes = vec![0.0_f64; atom_count];
    let mut counts = vec![0_usize; atom_count];
    for i in 0..shard.n_records() {
        if shard.is_control[i] {
            continue;
        }
        let atom = usize::try_from(shard.atom[i]).map_err(|err| {
            format!(
                "effect_dose_from_interventions: record {i} atom id {} is invalid: {err}",
                shard.atom[i]
            )
        })?;
        if atom >= atom_count {
            return Err(format!(
                "effect_dose_from_interventions: record {i} atom {atom} out of range for {atom_count} atoms"
            ));
        }
        let dose = shard.nu_measured[i];
        sums[atom] += dose;
        maxes[atom] = maxes[atom].max(dose);
        counts[atom] += 1;
    }
    let mut out = Vec::with_capacity(atom_count);
    for atom in 0..atom_count {
        if counts[atom] == 0 {
            out.push(None);
            continue;
        }
        let n = counts[atom];
        let threshold_nats = bic_one_degree_threshold_nats(n);
        out.push(Some(EffectDoseEvidence {
            atom,
            mean_realized_kl_nats: sums[atom] / n as f64,
            max_realized_kl_nats: maxes[atom],
            n_interventions: n,
            threshold_nats,
        }));
    }
    Ok(out)
}

/// Combine reconstruction and behavioral ledgers. Retention is an OR: an atom
/// that pays either in variance/charge or in causal-effect dose survives.
pub fn effect_weighted_retention(
    variance: &[Option<VarianceChargeEvidence>],
    effect: &[Option<EffectDoseEvidence>],
) -> Result<Vec<AtomRetentionEvidence>, String> {
    if variance.len() != effect.len() {
        return Err(format!(
            "effect_weighted_retention: variance has {} atoms but effect has {}",
            variance.len(),
            effect.len()
        ));
    }
    let mut out = Vec::with_capacity(variance.len());
    for atom in 0..variance.len() {
        if let Some(e) = effect[atom] {
            if e.atom != atom {
                return Err(format!(
                    "effect_weighted_retention: effect entry for slot {atom} names atom {}",
                    e.atom
                ));
            }
        }
        let retained_by_variance = variance[atom].is_some_and(VarianceChargeEvidence::retains);
        let retained_by_effect = effect[atom].is_some_and(EffectDoseEvidence::retains);
        out.push(AtomRetentionEvidence {
            atom,
            variance: variance[atom],
            effect: effect[atom],
            retained_by_variance,
            retained_by_effect,
            retained: retained_by_variance || retained_by_effect,
        });
    }
    Ok(out)
}

/// Convenience wrapper for a full Rung-3 shard plus existing variance evidence.
pub fn effect_weighted_retention_from_interventions(
    variance: &[Option<VarianceChargeEvidence>],
    shard: &InterventionShard,
) -> Result<Vec<AtomRetentionEvidence>, String> {
    let effect = effect_dose_from_interventions(variance.len(), shard)?;
    effect_weighted_retention(variance, &effect)
}

/// Primary fit-quality report. Interchange accuracy is deliberately the headline
/// because coordinates are useful only if interventions in those coordinates
/// land in the intended downstream behavior; reconstruction EV is secondary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectWeightedFitReport {
    pub headline: FitQualityMetric,
    pub interchange_accuracy: f64,
    pub explained_variance: f64,
}

impl EffectWeightedFitReport {
    pub fn new(interchange_accuracy: f64, explained_variance: f64) -> Result<Self, String> {
        validate_unit_interval("interchange_accuracy", interchange_accuracy)?;
        validate_unit_interval("explained_variance", explained_variance)?;
        Ok(Self {
            headline: FitQualityMetric::InterchangeAccuracy(interchange_accuracy),
            interchange_accuracy,
            explained_variance,
        })
    }

    pub fn headline_line(self) -> String {
        match self.headline {
            FitQualityMetric::InterchangeAccuracy(v) => {
                format!(
                    "headline: interchange_accuracy={v:.6}; secondary: explained_variance={:.6}",
                    self.explained_variance
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitQualityMetric {
    InterchangeAccuracy(f64),
}

fn bic_one_degree_threshold_nats(n_interventions: usize) -> f64 {
    0.5 * (n_interventions.max(2) as f64).ln()
}

fn validate_unit_interval(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "EffectWeightedFitReport: {name} must be finite and in [0, 1], got {value}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATOMS: usize = 2;
    const RARE_ATOM: usize = 0;
    const DENSE_ATOM: usize = 1;

    fn rare_effect_shard() -> InterventionShard {
        InterventionShard {
            row_id: vec![7, 8, 9, 10],
            atom: vec![
                RARE_ATOM as i64,
                RARE_ATOM as i64,
                DENSE_ATOM as i64,
                DENSE_ATOM as i64,
            ],
            dose: vec![1.0, 0.0, 1.0, 0.0],
            d_dose: 1,
            nu_hat_1: vec![8.0, 0.0, 0.0, 0.0],
            nu_hat_2: None,
            nu_measured: vec![8.0, 0.0, 0.0, 0.0],
            group: vec![70, 70, 90, 90],
            is_control: vec![false, true, false, true],
            layer: 18,
            seed: 19,
        }
    }

    #[test]
    fn rare_high_effect_atom_is_retained_when_variance_only_would_drop_it() {
        let variance = vec![
            Some(VarianceChargeEvidence {
                delta_deviance: 0.01,
                charge: 4.0,
            }),
            Some(VarianceChargeEvidence {
                delta_deviance: 20.0,
                charge: 4.0,
            }),
        ];
        assert!(!variance[RARE_ATOM].unwrap().retains());
        assert!(variance[DENSE_ATOM].unwrap().retains());

        let effect = effect_dose_from_interventions(ATOMS, &rare_effect_shard()).unwrap();
        let retained = effect_weighted_retention(&variance, &effect).unwrap();

        assert!(retained[RARE_ATOM].retained_by_effect);
        assert!(!retained[RARE_ATOM].retained_by_variance);
        assert!(retained[RARE_ATOM].retained);
        assert!(!retained[DENSE_ATOM].retained_by_effect);
        assert!(retained[DENSE_ATOM].retained_by_variance);
        assert!(retained[DENSE_ATOM].retained);
    }

    #[test]
    fn report_headline_is_interchange_accuracy_before_ev() {
        let report = EffectWeightedFitReport::new(0.875, 0.992).unwrap();
        assert_eq!(
            report.headline,
            FitQualityMetric::InterchangeAccuracy(0.875)
        );
        let line = report.headline_line();
        let interchange_pos = line.find("interchange_accuracy").unwrap();
        let ev_pos = line.find("explained_variance").unwrap();
        assert!(interchange_pos < ev_pos);
    }
}
