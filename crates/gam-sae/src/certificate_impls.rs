//! [`Certificate`] implementations for the gam-sae certificate zoo (task #16;
//! descended #1521).
//!
//! These `impl Certificate for …` blocks were relocated out of the monolith
//! root (`gam::inference::certificate_impls`) to satisfy the coherence orphan
//! rule: the [`Certificate`] trait now lives in the neutral `gam-problem` crate
//! and the implemented types ([`EncodeResult`], [`ResidualGaugeReport`],
//! [`CertificateInputs`]) are owned here in `gam-sae`, so the impls must be
//! defined in the type's home crate. The bodies are byte-identical to the
//! monolith originals: each [`Certificate::verdict`] is still defined in terms
//! of the type's own (unchanged) decision rule, so there remains exactly one
//! source of truth for each verdict.

use gam_problem::topology_certificates::{Certificate, Claim, Evidence, Verdict};

use crate::encode::EncodeResult;
use crate::identifiability::ResidualGaugeReport;
use crate::manifold::{CertificateInputs, CoordinateFidelityCertificate, GlobalOptimalityVerdict};

/// Helper: insert a scalar only when finite, else record it as text "n/a" so the
/// evidence is explicit about a missing quantity (never a silent 0.0).
fn put_finite(evidence: &mut Evidence, key: &'static str, value: f64) {
    if value.is_finite() {
        evidence.insert(key, value.into());
    } else {
        evidence.insert(key, "n/a".into());
    }
}

// ── 4. Kantorovich encode atlas (#1010) ──────────────────────────────────────

impl Certificate for EncodeResult {
    fn claim(&self) -> Claim {
        Claim::new(
            "encode-atlas",
            "each encoded row carries a per-row Newton–Kantorovich certificate \
             (h = β·η·L ≤ ½ at the start point); certified rows converge \
             quadratically into the unique root, and uncertified rows are flagged \
             for the exact multi-start fallback — never silently encoded wrong",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        let n = self.certified.len();
        let certified = n - self.encode_uncertified_count;
        e.insert("rows", n.into());
        e.insert("certified_rows", certified.into());
        e.insert(
            "encode_uncertified_count",
            self.encode_uncertified_count.into(),
        );
        let frac = if n > 0 {
            certified as f64 / n as f64
        } else {
            f64::NAN
        };
        put_finite(&mut e, "certified_fraction", frac);
        e
    }

    fn verdict(&self) -> Verdict {
        // Conservative batch roll-up: the whole encode certifies only when EVERY
        // row certified. One flagged row makes the batch `Insufficient` (the
        // flagged rows must route to the exact fallback). An empty batch
        // certifies nothing → `Unavailable`.
        if self.certified.is_empty() {
            Verdict::Unavailable
        } else if self.encode_uncertified_count == 0 {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

// ── 5. Exact-orbit residual-gauge report (#980/#998/#1008) ───────────────────

impl Certificate for ResidualGaugeReport {
    fn claim(&self) -> Claim {
        Claim::new(
            "residual-gauge",
            "the fit is identified up to a named residual gauge group: every \
             generator was curvature-tested in the fit's own metric, the pinning \
             span rank is reported, and any surviving (unpinned) freedom is \
             enumerated rather than silently absorbed",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        e.insert(
            "metric_provenance",
            format!("{:?}", self.metric_provenance).into(),
        );
        e.insert("group_signature", self.group_signature().into());
        e.insert("pinning_rank", self.pinning_rank.into());
        e.insert("residual_gauge_dim", self.residual_gauge_dim.into());
        e.insert(
            "diffeomorphism_unpinned",
            self.diffeomorphism_unpinned.into(),
        );
        e.insert("generator_count", self.generators.len().into());
        match self.sym_f_trivial_under_output_fisher {
            Some(t) => e.insert("sym_f_trivial_under_output_fisher", t.into()),
            None => e.insert("sym_f_trivial_under_output_fisher", "n/a".into()),
        };
        e.insert("summary", self.summary.clone().into());
        e
    }

    fn verdict(&self) -> Verdict {
        // The report ALWAYS makes a claim once computed (every generator is
        // tested), so it is never `Unavailable` here. The conservative reading:
        // the identifiability claim is `Certified` when the model is pinned down
        // to a discrete (zero-dimensional) gauge group and the diffeomorphism
        // pin is active; a positive residual gauge dimension or an inactive
        // diffeomorphism pin is the escalation flag → `Insufficient`. Under
        // OutputFisher provenance a surviving atom-permutation is a certificate
        // violation → also `Insufficient`.
        let pinned = self.residual_gauge_dim == 0
            && !self.diffeomorphism_unpinned
            && self.sym_f_trivial_under_output_fisher != Some(false);
        if pinned {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

// ── 6. Dictionary incoherence / global optimality (#1008) ────────────────────

impl Certificate for CertificateInputs {
    fn claim(&self) -> Claim {
        Claim::new(
            "global-optimality",
            "the fitted dictionary's basin stationary point is the unique global \
             optimum up to the residual gauge group: a conservative sufficient \
             condition on mutual coherence, per-atom curvature, activity floors, \
             and reconstruction SNR holds with positive margin",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(&mut e, "mu_hat", self.mu_hat);
        put_finite(&mut e, "mean_activity_floor", self.mean_activity_floor);
        put_finite(&mut e, "peak_activity_floor", self.peak_activity_floor);
        put_finite(&mut e, "snr_proxy", self.snr_proxy);
        put_finite(&mut e, "dispersion", self.dispersion);
        put_finite(
            &mut e,
            "global_optimality_margin",
            self.global_optimality.margin(),
        );
        e.insert(
            "global_optimality",
            if self.global_optimality.is_certified() {
                "certified_global"
            } else {
                "uncertified"
            }
            .into(),
        );
        e.insert("atom_count", self.per_atom_mean_activity.len().into());
        e.insert("note", self.note.clone().into());
        e
    }

    fn verdict(&self) -> Verdict {
        // The unchanged decision rule is `GlobalOptimalityVerdict::is_certified`:
        // a `CertifiedGlobal { margin > 0 }` is never wrong (conservative
        // sufficient condition), an `Uncertified` is "cannot decide" — not
        // "non-unique" — so it maps to `Insufficient`, never a false pass.
        match self.global_optimality {
            GlobalOptimalityVerdict::CertifiedGlobal { .. } => Verdict::Certified,
            GlobalOptimalityVerdict::Uncertified { .. } => Verdict::Insufficient,
        }
    }
}

// ── 7. Chart coordinate fidelity (#2081) ─────────────────────────────────────

impl<'a> Certificate for CoordinateFidelityCertificate<'a> {
    fn claim(&self) -> Claim {
        Claim::new(
            "coordinate-fidelity",
            "every eligible d=1 SAE chart reports a faithful coordinate reading: \
             either the raw chart is already arc-length or the payload supplies \
             the pure-read arc-length coordinate; degenerate charts are exposed \
             as refusals rather than silently treated as angles",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        let mut eligible = 0_usize;
        let mut certified = 0_usize;
        let mut degenerate = 0_usize;
        let mut max_arclength = f64::NEG_INFINITY;
        let mut max_raw_rms = f64::NEG_INFINITY;
        let mut max_raw_max = f64::NEG_INFINITY;
        let mut max_uniformity = f64::NEG_INFINITY;
        let mut min_p = f64::INFINITY;
        let mut worst = "unavailable";

        for atom in self.atoms.iter().flatten() {
            eligible += 1;
            if atom.certified {
                certified += 1;
                if worst == "unavailable" {
                    worst = atom.verdict.label();
                }
            } else {
                degenerate += 1;
                worst = atom.verdict.label();
            }
            if atom.arclength_defect.is_finite() {
                max_arclength = max_arclength.max(atom.arclength_defect);
            }
            if atom.raw_arclength_defect_rms.is_finite() {
                max_raw_rms = max_raw_rms.max(atom.raw_arclength_defect_rms);
            }
            if atom.raw_arclength_defect_max.is_finite() {
                max_raw_max = max_raw_max.max(atom.raw_arclength_defect_max);
            }
            if atom.uniformity_statistic.is_finite() {
                max_uniformity = max_uniformity.max(atom.uniformity_statistic);
            }
            if atom.uniformity_p_value.is_finite() {
                min_p = min_p.min(atom.uniformity_p_value);
            }
        }

        e.insert("atom_count", self.atoms.len().into());
        e.insert("eligible_d1_atoms", eligible.into());
        e.insert("certified_d1_atoms", certified.into());
        e.insert("degenerate_d1_atoms", degenerate.into());
        e.insert("worst_coordinate_verdict", worst.into());
        put_finite(&mut e, "max_arclength_defect", max_arclength);
        put_finite(&mut e, "max_raw_arclength_defect_rms", max_raw_rms);
        put_finite(&mut e, "max_raw_arclength_defect_max", max_raw_max);
        put_finite(&mut e, "max_uniformity_statistic", max_uniformity);
        put_finite(&mut e, "min_uniformity_p_value", min_p);
        e
    }

    fn verdict(&self) -> Verdict {
        let mut saw_eligible = false;
        for atom in self.atoms.iter().flatten() {
            saw_eligible = true;
            if !atom.certified {
                return Verdict::Insufficient;
            }
        }
        if saw_eligible {
            Verdict::Certified
        } else {
            Verdict::Unavailable
        }
    }
}
