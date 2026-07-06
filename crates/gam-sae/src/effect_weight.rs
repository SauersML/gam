//! Fisher-effect-weighted atom retention and fit-quality reporting.
//!
//! Reconstruction EV is a distributional currency: an atom that fires rarely can
//! explain almost no variance even when ablating it changes the downstream
//! distribution sharply. This module keeps the two ledgers separate. The
//! variance/rank-charge decision remains available, and a Fisher local-KL
//! effect decision is added beside it. Realized intervention KL is retained as
//! an empirical validation ledger, not as the derived Fisher effect weight.

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
        self.mean_fisher_quadratic_kl_nats - self.threshold_nats
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

    pub fn accumulate_firing_score_vector(
        &mut self,
        atom: usize,
        ablation_delta_theta: &[f64],
        score_vector: &[f64],
    ) -> Result<f64, String> {
        if ablation_delta_theta.len() != score_vector.len() {
            return Err(format!(
                "accumulate_firing_score_vector: ablation_delta_theta has length {} but score_vector has length {}",
                ablation_delta_theta.len(),
                score_vector.len()
            ));
        }
        let mut score_dot_delta = 0.0_f64;
        for j in 0..ablation_delta_theta.len() {
            let delta = ablation_delta_theta[j];
            let score = score_vector[j];
            if !delta.is_finite() {
                return Err(format!(
                    "accumulate_firing_score_vector: ablation_delta_theta[{j}] is non-finite"
                ));
            }
            if !score.is_finite() {
                return Err(format!(
                    "accumulate_firing_score_vector: score_vector[{j}] is non-finite"
                ));
            }
            score_dot_delta += score * delta;
        }
        let fisher_quadratic_kl_nats = 0.5 * score_dot_delta * score_dot_delta;
        self.accumulate_firing_local_kl(atom, fisher_quadratic_kl_nats)?;
        Ok(fisher_quadratic_kl_nats)
    }

    pub fn record_realized_kl_validation(
        &mut self,
        atom: usize,
        empirical_realized_kl_nats: f64,
    ) -> Result<(), String> {
        self.validate_atom(atom, "record_realized_kl_validation")?;
        validate_nonnegative_finite(
            "record_realized_kl_validation",
            "empirical_realized_kl_nats",
            empirical_realized_kl_nats,
        )?;
        self.realized_sums[atom] += empirical_realized_kl_nats;
        self.realized_maxes[atom] = self.realized_maxes[atom].max(empirical_realized_kl_nats);
        self.realized_counts[atom] += 1;
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

/// Compute the Fisher effect ledger from Rung-3 local-KL predictions.
///
/// Controls are excluded: they estimate measurement floor elsewhere and do not
/// represent an ablation/interchange dose. The retained effect score is the
/// streaming mean Fisher quadratic local-KL per executed non-control firing for
/// the atom; realized KL is attached only as empirical validation.
pub fn fisher_effect_from_interventions(
    atom_count: usize,
    shard: &InterventionShard,
) -> Result<Vec<Option<FisherEffectEvidence>>, String> {
    shard.validate()?;
    let mut accumulator = StreamingFisherEffectAccumulator::new(atom_count);
    for i in 0..shard.n_records() {
        if shard.is_control[i] {
            continue;
        }
        let atom = usize::try_from(shard.atom[i]).map_err(|err| {
            format!(
                "fisher_effect_from_interventions: record {i} atom id {} is invalid: {err}",
                shard.atom[i]
            )
        })?;
        if atom >= atom_count {
            return Err(format!(
                "fisher_effect_from_interventions: record {i} atom {atom} out of range for {atom_count} atoms"
            ));
        }
        accumulator.accumulate_firing_local_kl(atom, shard.nu_hat_1[i])?;
        accumulator.record_realized_kl_validation(atom, shard.nu_measured[i])?;
    }
    Ok(accumulator.finish())
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

/// Convenience wrapper for a full Rung-3 shard plus existing variance evidence.
pub fn fisher_effect_weighted_retention_from_interventions(
    variance: &[Option<VarianceChargeEvidence>],
    shard: &InterventionShard,
) -> Result<Vec<AtomRetentionEvidence>, String> {
    let effect = fisher_effect_from_interventions(variance.len(), shard)?;
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
    fn rare_high_fisher_effect_atom_is_retained_when_variance_only_would_drop_it() {
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

        let effect = fisher_effect_from_interventions(ATOMS, &rare_effect_shard()).unwrap();
        let rare_effect = effect[RARE_ATOM].unwrap();
        assert_eq!(rare_effect.mean_fisher_quadratic_kl_nats, 8.0);
        assert_eq!(
            rare_effect
                .realized_kl_validation
                .unwrap()
                .mean_empirical_realized_kl_nats,
            8.0
        );
        let retained = effect_weighted_retention(&variance, &effect).unwrap();

        assert!(retained[RARE_ATOM].retained_by_effect);
        assert!(!retained[RARE_ATOM].retained_by_variance);
        assert!(retained[RARE_ATOM].retained);
        assert!(!retained[DENSE_ATOM].retained_by_effect);
        assert!(retained[DENSE_ATOM].retained_by_variance);
        assert!(retained[DENSE_ATOM].retained);
    }

    #[test]
    fn streaming_score_vector_accumulates_fisher_quadratic_per_firing() {
        let mut accumulator = StreamingFisherEffectAccumulator::new(ATOMS);
        let local = accumulator
            .accumulate_firing_score_vector(RARE_ATOM, &[0.02, -0.01], &[3.0, 4.0])
            .unwrap();
        assert!((local - 0.0002).abs() <= 1e-15);
        accumulator
            .record_realized_kl_validation(RARE_ATOM, 0.000201)
            .unwrap();

        let evidence = accumulator.finish();
        let rare = evidence[RARE_ATOM].unwrap();
        assert!((rare.mean_fisher_quadratic_kl_nats - 0.0002).abs() <= 1e-15);
        assert_eq!(rare.n_firings, 1);
        assert!(evidence[DENSE_ATOM].is_none());
        let validation = rare.realized_kl_validation.unwrap();
        assert_eq!(validation.n_interventions, 1);
        assert!((validation.mean_empirical_realized_kl_nats - 0.000201).abs() <= 1e-15);
    }

    #[test]
    fn fisher_quadratic_matches_realized_kl_in_small_dose_limit() {
        let p = 0.37_f64;
        let eta = (p / (1.0 - p)).ln();
        let fisher = p * (1.0 - p);
        let mut accumulator = StreamingFisherEffectAccumulator::new(1);
        for eps in [0.001_f64, -0.0015] {
            let local_fisher_kl = 0.5 * fisher * eps * eps;
            let q = logistic(eta + eps);
            let realized_kl = bernoulli_kl(p, q);
            accumulator
                .accumulate_firing_local_kl(0, local_fisher_kl)
                .unwrap();
            accumulator
                .record_realized_kl_validation(0, realized_kl)
                .unwrap();
        }

        let evidence = accumulator.finish();
        let atom = evidence[0].unwrap();
        let realized = atom.realized_kl_validation.unwrap();
        let rel_err = (atom.mean_fisher_quadratic_kl_nats
            - realized.mean_empirical_realized_kl_nats)
            .abs()
            / realized.mean_empirical_realized_kl_nats;
        eprintln!(
            "small-dose Fisher/realized KL: fisher_mean={:.12e} realized_mean={:.12e} rel_err={:.6e}",
            atom.mean_fisher_quadratic_kl_nats,
            realized.mean_empirical_realized_kl_nats,
            rel_err
        );
        assert!(rel_err <= 0.002, "relative error {rel_err}");
    }

    fn logistic(eta: f64) -> f64 {
        1.0 / (1.0 + (-eta).exp())
    }

    fn bernoulli_kl(p: f64, q: f64) -> f64 {
        p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
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
