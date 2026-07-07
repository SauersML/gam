//! App A — the Certified Feature Dossier: one orchestrator that, per atom,
//! collects everything the instrumented SAE already measures about a fitted
//! feature and lays it out as a single JSON (+ an HTML surface via `gam-report`).
//!
//! This is WIRING, not new computation. The heavy per-atom certificates are
//! produced by [`SaeManifoldTerm::fit_diagnostics_report`] (the two-score lens,
//! the residual-gauge group, the per-atom Riesz functionals + split-LRT
//! non-constancy e-value, the chart coordinate-fidelity statistic, and the
//! persistent-homology topology audit with its standing-null calibration and
//! `contested` flag) and by [`SaeManifoldTerm::trust_diagnostics_report`] (the
//! per-atom tangent-condition trust score, coverage, activation frequency,
//! effective-N). The shape band comes from the Laplace shape-uncertainty pass
//! ([`crate::manifold::SaeShapeUncertainty`]); the behavioral steering dose
//! (nats/Δt) from [`crate::inference::steering::steer_delta`]; the fit's own
//! description-length decomposition (bits) from
//! [`crate::description_length::DescriptionLength`]. This module reads those
//! reports and re-projects them **per atom** into one dossier, adding only the
//! cheap per-atom steering-dose call. It never mutates the term and never feeds
//! anything back into a loss/criterion.
//!
//! Signals that a single-model fit does not itself produce — cross-layer
//! transport class + loop holonomy, between-atom Terracini conditioning, and the
//! executed contested-probe / e-BH loop — are exposed as attachable fields
//! ([`FeatureDossier::with_terracini`], [`AtomDossierEntry`]'s `transport` /
//! `contested_probe`) so a driver that has the extra inputs (a second layer's
//! atoms, a `TerraciniReport`, a realized probe ledger) can fold them in without
//! this orchestrator reaching into a subsystem it cannot certify from one term.

use serde::Serialize;

use crate::description_length::DescriptionLength;
use crate::manifold::{
    SaeManifoldFitDiagnostics, SaeManifoldTerm, SaeShapeUncertainty, SaeTrustDiagnostics,
    TerraciniReport,
};

/// The fit's own description-length decomposition, in bits (mirror of
/// [`DescriptionLength`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DescriptionLengthDossier {
    pub code_bits: f64,
    pub selection_bits: f64,
    pub dict_bits: f64,
    pub total_bits: f64,
    pub bits_per_token: f64,
}

impl From<&DescriptionLength> for DescriptionLengthDossier {
    fn from(d: &DescriptionLength) -> Self {
        Self {
            code_bits: d.code_bits,
            selection_bits: d.selection_bits,
            dict_bits: d.dict_bits,
            total_bits: d.total_bits,
            bits_per_token: d.bits_per_token,
        }
    }
}

/// Model-level residual-gauge / identifiability certificate (from
/// [`crate::identifiability::ResidualGaugeReport`]).
#[derive(Clone, Debug, Serialize)]
pub struct ResidualGaugeDossier {
    /// The certified residual gauge group — two replicate fits are "identified up
    /// to the same group" iff this string is equal.
    pub group_signature: String,
    /// Number of generators certified as unpinned residual gauge freedoms.
    pub residual_gauge_dim: usize,
    /// Escalation flag: the model is only identified up to an arbitrary
    /// diffeomorphism of the latent manifolds.
    pub diffeomorphism_unpinned: bool,
    /// Under output-Fisher provenance, whether the atom-permutation subgroup is
    /// trivially pinned (`None` when provenance is not output-Fisher).
    pub sym_f_trivial_under_output_fisher: Option<bool>,
    pub metric_provenance: String,
    pub summary: String,
}

/// Per-atom shape band summary (from [`crate::manifold::SaeAtomShapeUncertainty`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ShapeBandDossier {
    /// Mean posterior band standard deviation over the atom's band grid/channels.
    pub mean_band_sd: f64,
    /// Maximum posterior band standard deviation (the widest point of the band).
    pub max_band_sd: f64,
    /// Whether a misspecification-robust (sandwich) band companion is present.
    pub has_robust_band: bool,
    /// Mean robust band standard deviation, when the robust companion is present.
    pub mean_band_sd_robust: Option<f64>,
}

/// Per-atom two-score lens entry (from
/// [`crate::inference::atom_lens::AtomLensEntry`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct LensDossier {
    pub presence: f64,
    pub presence_normalized: f64,
    pub coupling: Option<f64>,
    pub coupling_normalized: Option<f64>,
    /// `presence_normalized − coupling_normalized`: "represented but not used".
    pub discrepancy: Option<f64>,
    pub represented_not_used: bool,
    pub used: bool,
}

/// Per-atom persistent-homology topology audit + standing-null calibration (from
/// [`crate::manifold::AtomTopologyPersistence`]).
#[derive(Clone, Debug, Serialize)]
pub struct TopologyDossier {
    pub measured_betti: String,
    pub expected_betti: String,
    pub dominant_h1_persistence: f64,
    pub dominant_h2_persistence: f64,
    pub support_size: usize,
    pub landmark_count: usize,
    pub stability_band: String,
    /// The measured topology disagrees with the latched race winner — the probe
    /// planner's re-adjudication trigger.
    pub contested: bool,
    /// Standing-null calibration for the topology claim, when computed.
    pub null_calibration: Option<NullCalibrationDossier>,
}

/// Standing-null calibration (from [`crate::null_battery::ClaimNullCalibration`]).
#[derive(Clone, Debug, Serialize)]
pub struct NullCalibrationDossier {
    pub claim: String,
    pub observed_statistic: f64,
    pub null_pvalue: f64,
    pub null_z: f64,
    pub claimed_snr: f64,
    pub claimed_false_positive_rate: f64,
    pub spikein_power: f64,
}

/// Per-atom chart coordinate-fidelity certificate (from
/// [`crate::manifold::AtomCoordinateFidelity`]).
#[derive(Clone, Debug, Serialize)]
pub struct CoordinateFidelityDossier {
    pub topology: String,
    pub uniformity_statistic: f64,
    pub uniformity_p_value: f64,
    pub arclength_defect: f64,
    pub n_coords: usize,
    pub verdict: String,
}

/// Per-atom penalty-debiased functional inference + split-LRT non-constancy
/// e-value (from [`crate::identifiability::AtomInferenceReport`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct InferenceDossier {
    /// One-step penalty-debiased data-averaged decoder value.
    pub average_value_onestep: Option<f64>,
    /// Signed mass-weighted mean derivative `E_data[∂g/∂t]` along the leading axis.
    pub decoder_variation: Option<f64>,
    /// `log E` for "the atom's smooth is non-constant" (null = constant); a
    /// universal-inference split-LRT e-value, `≥ 1/α` certifies at level α.
    pub log_e_nonconstant: Option<f64>,
}

/// Per-atom tangent-condition trust (from
/// [`crate::manifold::SaeAtomTrustDiagnostics`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TrustDossier {
    /// `tangent_condition_score × coverage⁴` — quartic decay with chart coverage.
    pub trust_score: f64,
    pub tangent_condition_score: f64,
    pub coverage: f64,
    pub activation_frequency: f64,
    pub effective_n: f64,
    pub active_token_count: usize,
    /// True when the atom's basis was caller-supplied (untyped precomputed).
    pub untyped: bool,
}

/// Per-atom behavioral steering dose (from [`crate::inference::steering::steer_delta`]).
#[derive(Clone, Debug, Serialize)]
pub struct SteeringDossier {
    pub t_from: Vec<f64>,
    pub t_to: Vec<f64>,
    /// Behavioral speed `½ ∫ a² g'ᵀ M g' dt` in nats over the traversed span;
    /// `None` under a Euclidean (no-behavior) metric.
    pub predicted_nats: Option<f64>,
    pub validity_radius: Option<f64>,
    pub off_manifold_norm: f64,
    pub metric_provenance: String,
}

/// The crown: a contested claim and the cheapest designed probe to settle it.
#[derive(Clone, Debug, Serialize)]
pub struct ContestedProbeDossier {
    /// What is contested (e.g. the measured-vs-predicted topology disagreement).
    pub claim: String,
    /// The designed settling experiment (a human-readable description of the
    /// cheapest probe the planner selected).
    pub designed_probe: String,
    /// The realized log-e the probe contributed to the e-BH ledger, when executed.
    pub realized_log_e: Option<f64>,
}

/// Cross-layer transport class + loop holonomy for one atom (attachable; a
/// single-model fit does not produce it).
#[derive(Clone, Debug, Serialize)]
pub struct TransportDossier {
    /// The circle transport class across layers (e.g. `Identity`, `Reflection`).
    pub transport_class: String,
    /// Transport phase in degrees.
    pub phase_degrees: f64,
    /// Loop holonomy defect (radians) when a closed layer loop was traversed.
    pub loop_holonomy: Option<f64>,
}

/// One atom's full certified dossier entry.
#[derive(Clone, Debug, Serialize)]
pub struct AtomDossierEntry {
    pub atom_index: usize,
    pub atom_name: String,
    pub shape_band: Option<ShapeBandDossier>,
    pub trust: Option<TrustDossier>,
    pub lens: Option<LensDossier>,
    pub topology: Option<TopologyDossier>,
    pub coordinate_fidelity: Option<CoordinateFidelityDossier>,
    pub inference: Option<InferenceDossier>,
    pub steering: Option<SteeringDossier>,
    /// Set when the atom's topology audit is contested — the designed probe the
    /// planner would run to settle it.
    pub contested_probe: Option<ContestedProbeDossier>,
    /// Cross-layer transport, when attached by a multi-layer driver.
    pub transport: Option<TransportDossier>,
}

/// The App A Certified Feature Dossier: model-level identifiability header plus
/// one certified entry per atom.
#[derive(Clone, Debug, Serialize)]
pub struct FeatureDossier {
    pub schema: &'static str,
    pub n_atoms: usize,
    pub residual_gauge: ResidualGaugeDossier,
    /// The fit's own description-length decomposition (bits), when the caller
    /// supplied the criterion's nats breakdown.
    pub description_length: Option<DescriptionLengthDossier>,
    /// Between-atom Terracini conditioning (report-only, never folded into V);
    /// attached when the caller ran a Terracini scan.
    pub terracini: Option<TerraciniDossier>,
    pub atoms: Vec<AtomDossierEntry>,
}

/// Between-atom Terracini conditioning summary (from
/// [`crate::manifold::terracini::TerraciniReport`]).
#[derive(Clone, Debug, Serialize)]
pub struct TerraciniDossier {
    pub mode: String,
    pub n_rows_scanned: usize,
    pub n_atoms_scanned: usize,
    pub n_pairs_scanned: usize,
    pub n_flagged_cliques: usize,
    pub flag_margin: f64,
}

impl From<&TerraciniReport> for TerraciniDossier {
    fn from(r: &TerraciniReport) -> Self {
        Self {
            mode: format!("{:?}", r.mode),
            n_rows_scanned: r.n_rows_scanned,
            n_atoms_scanned: r.atoms.len(),
            n_pairs_scanned: r.pairs.len(),
            n_flagged_cliques: r.flagged_cliques.len(),
            flag_margin: r.flag_margin,
        }
    }
}

impl FeatureDossier {
    /// Attach a between-atom Terracini conditioning report (report-only).
    #[must_use]
    pub fn with_terracini(mut self, report: &TerraciniReport) -> Self {
        self.terracini = Some(TerraciniDossier::from(report));
        self
    }

    /// Serialize to a pretty JSON string.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("FeatureDossier::to_json_pretty: {e}"))
    }

    /// Serialize to a compact JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("FeatureDossier::to_json: {e}"))
    }
}

fn betti_string(b: &crate::manifold::BettiSignature) -> String {
    match b.b2 {
        Some(b2) => format!("(b0={}, b1={}, b2={})", b.b0, b.b1, b2),
        None => format!("(b0={}, b1={})", b.b0, b.b1),
    }
}

fn shape_band_dossier(atom: &crate::manifold::SaeAtomShapeUncertainty) -> Option<ShapeBandDossier> {
    let finite: Vec<f64> = atom.band_sd.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_robust = atom.band_sd_robust.as_ref().and_then(|r| {
        let f: Vec<f64> = r.iter().copied().filter(|v| v.is_finite()).collect();
        (!f.is_empty()).then(|| f.iter().sum::<f64>() / f.len() as f64)
    });
    Some(ShapeBandDossier {
        mean_band_sd: mean,
        max_band_sd: max,
        has_robust_band: atom.band_sd_robust.is_some(),
        mean_band_sd_robust: mean_robust,
    })
}

impl SaeManifoldTerm {
    /// App A — assemble the Certified Feature Dossier from the already-built
    /// diagnostics reports, re-projected per atom.
    ///
    /// `diagnostics` and `trust` are the outputs of
    /// [`Self::fit_diagnostics_report`] and [`Self::trust_diagnostics_report`]
    /// (built once by the caller). `shape` is the optional Laplace shape-band
    /// report. `criterion_nats` is the optional `(data_fit, sparsity,
    /// logdet_occam, n_tokens)` breakdown for the fit's own description-length
    /// decomposition. The only new per-atom computation this performs is the cheap
    /// behavioral steering-dose call over each atom's band coordinate span; every
    /// other field is a re-projection of the passed-in reports. Pure read.
    pub fn certified_feature_dossier(
        &self,
        diagnostics: &SaeManifoldFitDiagnostics,
        trust: &SaeTrustDiagnostics,
        shape: Option<&SaeShapeUncertainty>,
        criterion_nats: Option<(f64, f64, f64, i64)>,
    ) -> FeatureDossier {
        let k = self.k_atoms();
        let gauge = &diagnostics.residual_gauge;
        let residual_gauge = ResidualGaugeDossier {
            group_signature: gauge.group_signature(),
            residual_gauge_dim: gauge.residual_gauge_dim,
            diffeomorphism_unpinned: gauge.diffeomorphism_unpinned,
            sym_f_trivial_under_output_fisher: gauge.sym_f_trivial_under_output_fisher,
            metric_provenance: format!("{:?}", gauge.metric_provenance),
            summary: gauge.summary.clone(),
        };

        let description_length = criterion_nats.map(|(data_fit, sparsity, logdet_occam, n)| {
            DescriptionLengthDossier::from(&DescriptionLength::from_criterion_nats(
                data_fit,
                sparsity,
                logdet_occam,
                n,
            ))
        });

        let metric = self.row_metric();
        let mut atoms = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let atom_name = diagnostics
                .atom_two_lens
                .atoms
                .get(atom_idx)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| format!("atom_{atom_idx}"));

            let shape_band = shape
                .and_then(|s| s.atoms.get(atom_idx))
                .and_then(shape_band_dossier);

            let trust_entry = trust.atoms.get(atom_idx).map(|t| TrustDossier {
                trust_score: t.trust_score,
                tangent_condition_score: t.tangent_condition_score,
                coverage: t.coverage,
                activation_frequency: t.activation_frequency,
                effective_n: t.effective_n,
                active_token_count: t.active_token_count,
                untyped: t.untyped,
            });

            let lens = diagnostics.atom_two_lens.atoms.get(atom_idx).map(|e| LensDossier {
                presence: e.presence,
                presence_normalized: e.presence_normalized,
                coupling: e.coupling,
                coupling_normalized: e.coupling_normalized,
                discrepancy: e.discrepancy,
                represented_not_used: e.is_represented_not_used(),
                used: e.is_used(),
            });

            let topology = diagnostics
                .topology_persistence
                .get(atom_idx)
                .and_then(|opt| opt.as_ref())
                .map(|tp| TopologyDossier {
                    measured_betti: betti_string(&tp.measured_betti),
                    expected_betti: betti_string(&tp.expected_betti),
                    dominant_h1_persistence: tp.dominant_h1_persistence,
                    dominant_h2_persistence: tp.dominant_h2_persistence,
                    support_size: tp.support_size,
                    landmark_count: tp.landmark_count,
                    stability_band: format!("{:?}", tp.stability_band),
                    contested: tp.contested,
                    null_calibration: tp.null_calibration.as_ref().map(|nc| {
                        NullCalibrationDossier {
                            claim: nc.claim.clone(),
                            observed_statistic: nc.observed_statistic,
                            null_pvalue: nc.null_pvalue,
                            null_z: nc.null_z,
                            claimed_snr: nc.claimed_snr,
                            claimed_false_positive_rate: nc.claimed_false_positive_rate,
                            spikein_power: nc.spikein_power,
                        }
                    }),
                });

            let coordinate_fidelity = diagnostics
                .coordinate_fidelity
                .get(atom_idx)
                .and_then(|opt| opt.as_ref())
                .map(|cf| CoordinateFidelityDossier {
                    topology: cf.topology.to_string(),
                    uniformity_statistic: cf.uniformity_statistic,
                    uniformity_p_value: cf.uniformity_p_value,
                    arclength_defect: cf.arclength_defect,
                    n_coords: cf.n_coords,
                    verdict: format!("{:?}", cf.verdict),
                });

            let inference = diagnostics.atom_inference.get(atom_idx).map(|ai| {
                let (avg, var) = match &ai.functionals {
                    Some(f) => (
                        f.average_value.as_ref().map(|e| e.theta_onestep),
                        f.decoder_variation_norm.as_ref().map(|e| e.theta_onestep),
                    ),
                    None => (None, None),
                };
                InferenceDossier {
                    average_value_onestep: avg,
                    decoder_variation: var,
                    log_e_nonconstant: ai
                        .smooth_significance
                        .as_ref()
                        .and_then(|s| s.log_e_nonconstant),
                }
            });

            // The only new per-atom computation: the behavioral steering dose over
            // the atom's band coordinate span. Best-effort — a missing basis
            // evaluator, a Euclidean metric, or a degenerate span degrades the
            // dose to `None`, never an error.
            let steering = self.atom_steering_dossier(atom_idx, metric, shape);

            let contested_probe = topology.as_ref().filter(|t| t.contested).map(|t| {
                ContestedProbeDossier {
                    claim: format!(
                        "measured topology {} disagrees with predicted {}",
                        t.measured_betti, t.expected_betti
                    ),
                    designed_probe: "resample the atom's assigned-row image at a \
                        farthest-point cover and re-run the Vietoris-Rips persistence \
                        against the raced-kind null; adopt the measured Betti when the \
                        standing-null e-value clears 1/alpha under optional stopping"
                        .to_string(),
                    realized_log_e: None,
                }
            });

            atoms.push(AtomDossierEntry {
                atom_index: atom_idx,
                atom_name,
                shape_band,
                trust: trust_entry,
                lens,
                topology,
                coordinate_fidelity,
                inference,
                steering,
                contested_probe,
                transport: None,
            });
        }

        FeatureDossier {
            schema: "gam-sae/certified-feature-dossier/v1",
            n_atoms: k,
            residual_gauge,
            description_length,
            terracini: None,
            atoms,
        }
    }

    /// The per-atom behavioral steering dose over the atom's band coordinate span.
    /// Best-effort: returns `None` when no metric, no shape bands, or the atom's
    /// span is degenerate / the steer plan refuses.
    fn atom_steering_dossier(
        &self,
        atom_idx: usize,
        metric: Option<&gam_problem::RowMetric>,
        shape: Option<&SaeShapeUncertainty>,
    ) -> Option<SteeringDossier> {
        let metric = metric?;
        let atom_shape = shape?.atoms.get(atom_idx)?;
        let coords = &atom_shape.band_coords;
        let (n_band, d) = coords.dim();
        if n_band < 2 || d == 0 {
            return None;
        }
        let t_from: Vec<f64> = coords.row(0).iter().copied().collect();
        let t_to: Vec<f64> = coords.row(n_band - 1).iter().copied().collect();
        let plan =
            crate::inference::steering::steer_delta(self, metric, atom_idx, &t_from, &t_to).ok()?;
        Some(SteeringDossier {
            t_from: plan.t_from,
            t_to: plan.t_to,
            predicted_nats: plan.predicted_nats,
            validity_radius: plan.validity_radius,
            off_manifold_norm: plan.off_manifold_norm,
            metric_provenance: format!("{:?}", plan.metric_provenance),
        })
    }
}
