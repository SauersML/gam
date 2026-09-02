//! Fisher-effect-weighted atom retention and fit-quality reporting.
//!
//! Reconstruction EV is a distributional currency: an atom that fires rarely can
//! explain almost no variance even when ablating it changes the downstream
//! distribution sharply. This module keeps the two ledgers separate. The
//! variance/rank-charge decision remains available, and a Fisher local-KL
//! effect decision is added beside it. Realized intervention KL is retained as
//! an empirical validation ledger, not as the derived Fisher effect weight.

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

/// Empirical intervention KL ledger for one atom.
///
/// This is a validation report for executed Rung-3 interventions. It is not the
/// Fisher effect weight used for retention, because measured realized KL can
/// include finite-dose and measurement effects outside the local quadratic
/// approximation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealizedKlValidationEvidence {
    /// Atom index.
    pub atom: usize,
    /// Mean measured KL over non-control interventions for this atom.
    pub mean_empirical_realized_kl_nats: f64,
    /// Largest measured KL over non-control interventions for this atom.
    pub max_empirical_realized_kl_nats: f64,
    /// Number of non-control interventions that touched this atom.
    pub n_interventions: usize,
}

/// Streaming Fisher local-KL evidence for one atom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherEffectEvidence {
    /// Atom index.
    pub atom: usize,
    /// Mean Fisher quadratic local-KL, `0.5 * Δθᵀ I Δθ`, over ablated firings.
    pub mean_fisher_quadratic_kl_nats: f64,
    /// Total Fisher quadratic local-KL over the `n_firings` observations. BIC is
    /// an additive evidence price, so this — not the per-firing mean — is the
    /// quantity compared with [`Self::threshold_nats`].
    pub total_fisher_quadratic_kl_nats: f64,
    /// Largest per-firing Fisher quadratic local-KL for this atom.
    pub max_fisher_quadratic_kl_nats: f64,
    /// Number of ablated firings accumulated for this atom.
    pub n_firings: usize,
    /// Derived discovery threshold in nats. This is the one-degree BIC price for
    /// the firing sample size: 0.5 * ln(max(n_firings, 2)).
    pub threshold_nats: f64,
    /// Optional measured-KL validation ledger for the same atom.
    pub realized_kl_validation: Option<RealizedKlValidationEvidence>,
}

impl FisherEffectEvidence {
    pub fn margin(self) -> f64 {
        self.total_fisher_quadratic_kl_nats - self.threshold_nats
    }

    pub fn retains(self) -> bool {
        self.margin() > 0.0
    }
}

/// Streaming per-firing Fisher accumulator.
///
/// Callers may either pass an already-computed local quadratic KL term, or pass
/// a score vector and an ablation vector. The latter streams the quadratic form
/// without materializing any token-by-atom design matrix: each firing contributes
/// `0.5 * (scoreᵀ Δθ)^2`, the empirical-score form of `0.5 * Δθᵀ I Δθ`.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingFisherEffectAccumulator {
    atom_count: usize,
    fisher_sums: Vec<f64>,
    fisher_maxes: Vec<f64>,
    firing_counts: Vec<usize>,
    realized_sums: Vec<f64>,
    realized_maxes: Vec<f64>,
    realized_counts: Vec<usize>,
}

impl StreamingFisherEffectAccumulator {
    pub fn new(atom_count: usize) -> Self {
        Self {
            atom_count,
            fisher_sums: vec![0.0; atom_count],
            fisher_maxes: vec![0.0; atom_count],
            firing_counts: vec![0; atom_count],
            realized_sums: vec![0.0; atom_count],
            realized_maxes: vec![0.0; atom_count],
            realized_counts: vec![0; atom_count],
        }
    }

    pub fn accumulate_firing_local_kl(
        &mut self,
        atom: usize,
        fisher_quadratic_kl_nats: f64,
    ) -> Result<(), String> {
        self.validate_atom(atom, "accumulate_firing_local_kl")?;
        validate_nonnegative_finite(
            "accumulate_firing_local_kl",
            "fisher_quadratic_kl_nats",
            fisher_quadratic_kl_nats,
        )?;
        self.fisher_sums[atom] += fisher_quadratic_kl_nats;
        self.fisher_maxes[atom] = self.fisher_maxes[atom].max(fisher_quadratic_kl_nats);
        self.firing_counts[atom] += 1;
        Ok(())
    }

    pub fn finish(self) -> Vec<Option<FisherEffectEvidence>> {
        let mut out = Vec::with_capacity(self.atom_count);
        for atom in 0..self.atom_count {
            let n_firings = self.firing_counts[atom];
            if n_firings == 0 {
                out.push(None);
                continue;
            }
            let realized_kl_validation = if self.realized_counts[atom] == 0 {
                None
            } else {
                let n_interventions = self.realized_counts[atom];
                Some(RealizedKlValidationEvidence {
                    atom,
                    mean_empirical_realized_kl_nats: self.realized_sums[atom]
                        / n_interventions as f64,
                    max_empirical_realized_kl_nats: self.realized_maxes[atom],
                    n_interventions,
                })
            };
            out.push(Some(FisherEffectEvidence {
                atom,
                mean_fisher_quadratic_kl_nats: self.fisher_sums[atom] / n_firings as f64,
                total_fisher_quadratic_kl_nats: self.fisher_sums[atom],
                max_fisher_quadratic_kl_nats: self.fisher_maxes[atom],
                n_firings,
                threshold_nats: bic_one_degree_threshold_nats(n_firings),
                realized_kl_validation,
            }));
        }
        out
    }

    fn validate_atom(&self, atom: usize, caller: &str) -> Result<(), String> {
        if atom < self.atom_count {
            Ok(())
        } else {
            Err(format!(
                "{caller}: atom {atom} out of range for {} atoms",
                self.atom_count
            ))
        }
    }
}

/// Full retention verdict for one atom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomRetentionEvidence {
    pub atom: usize,
    pub variance: Option<VarianceChargeEvidence>,
    pub effect: Option<FisherEffectEvidence>,
    pub retained_by_variance: bool,
    pub retained_by_effect: bool,
    pub retained: bool,
}

/// Combine reconstruction and behavioral ledgers. Retention is an OR: an atom
/// that pays either in variance/charge or in Fisher local-KL effect survives.
pub fn effect_weighted_retention(
    variance: &[Option<VarianceChargeEvidence>],
    effect: &[Option<FisherEffectEvidence>],
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
        let retained_by_effect = effect[atom].is_some_and(FisherEffectEvidence::retains);
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

fn validate_nonnegative_finite(caller: &str, name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "{caller}: {name} must be finite and >= 0; got {value}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bic_price_is_compared_with_total_not_mean_kl() {
        let mut accumulator = StreamingFisherEffectAccumulator::new(1);
        for _ in 0..100 {
            accumulator.accumulate_firing_local_kl(0, 0.1).unwrap();
        }
        let evidence = accumulator.finish()[0].unwrap();
        assert!((evidence.mean_fisher_quadratic_kl_nats - 0.1).abs() < 1e-12);
        assert!((evidence.total_fisher_quadratic_kl_nats - 10.0).abs() < 1e-12);
        assert!(
            evidence.retains(),
            "total KL 10 must exceed the one-dof BIC price"
        );
    }

}
