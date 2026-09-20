//! Fisher-effect-weighted atom retention.
//!
//! Reconstruction EV is a distributional currency: an atom that fires rarely can
//! explain almost no variance even when ablating it changes the downstream
//! distribution sharply. This module keeps the two ledgers separate. The
//! variance/rank-charge decision remains available, and a Fisher local-KL
//! effect decision is added beside it.

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
/// Each ablated firing contributes its already-computed local quadratic KL term
/// `0.5 * Δθᵀ I Δθ`, so no token-by-atom design matrix is ever materialized.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingFisherEffectAccumulator {
    atom_count: usize,
    fisher_sums: Vec<f64>,
    fisher_maxes: Vec<f64>,
    firing_counts: Vec<usize>,
}

impl StreamingFisherEffectAccumulator {
    pub fn new(atom_count: usize) -> Self {
        Self {
            atom_count,
            fisher_sums: vec![0.0; atom_count],
            fisher_maxes: vec![0.0; atom_count],
            firing_counts: vec![0; atom_count],
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
            out.push(Some(FisherEffectEvidence {
                atom,
                mean_fisher_quadratic_kl_nats: self.fisher_sums[atom] / n_firings as f64,
                total_fisher_quadratic_kl_nats: self.fisher_sums[atom],
                max_fisher_quadratic_kl_nats: self.fisher_maxes[atom],
                n_firings,
                threshold_nats: bic_one_degree_threshold_nats(n_firings),
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

fn bic_one_degree_threshold_nats(n_firings: usize) -> f64 {
    0.5 * (n_firings.max(2) as f64).ln()
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
