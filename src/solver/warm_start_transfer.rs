//! Cross-fit warm-start *transfer*: build a starting iterate (block β and ρ)
//! for a new fit from a structurally-matching parent [`FitArtifact`].
//!
//! Phase 1 implements the **ρ half** of the transfer (the marquee LOSO win):
//! for each new term, if a parent term shares its [`TermIdentityKey`], copy
//! the parent's converged log-smoothing parameters into the new ρ layout —
//! clamped out of the saturated box. Unmatched / brand-new terms fall back to
//! the new fit's penalty-label default. β stays cold (zeros) in Phase 1; the
//! function-space β projection is Phase 2.
//!
//! Every path is exactness-preserving: a warm ρ only changes where the outer
//! optimizer *starts*; it still runs to its KKT/REML certificate, so the
//! converged optimum is identical to a cold start within tolerance. On any
//! anomaly — missing parent, length mismatch, non-finite payload — the build
//! returns an all-cold result (never an error), so a misfired transfer can
//! never fail a fit.

use crate::solver::warm_start_artifact::{
    FitArtifact, FitDescriptor, TermIdentityKey, TransferProvenance, RHO_SATURATION,
};
use ndarray::Array1;

/// Per-term context for the new (about-to-run) fit. Carries everything the
/// transfer needs to lay β and ρ out in the new fit's coordinate system
/// without reaching back into the solver internals.
#[derive(Clone, Debug)]
pub struct TermBuildContext {
    /// Structural identity of this new-fit term.
    pub identity: TermIdentityKey,
    /// Reduced (post-identifiability) block width — the length of the cold β
    /// vector this term contributes to the inner solve.
    pub reduced_block_width: usize,
    /// Indices into the new fit's outer ρ vector that this term's penalties
    /// occupy (after label de-duplication). Empty for an unpenalized term.
    pub rho_slots: Vec<usize>,
}

/// Result of a warm-start build: one cold-or-warm β per new block, the new ρ
/// vector, and a per-term provenance trace for logging / tests.
#[derive(Clone, Debug)]
pub struct TransferResult {
    pub block_beta: Vec<Array1<f64>>,
    pub rho: Array1<f64>,
    pub provenance: Vec<TransferProvenance>,
}

/// Configuration knobs for the transfer. Defaults are the magic path; there
/// are no user-facing flags.
#[derive(Clone, Copy, Debug)]
pub struct TransferConfig {
    /// Magnitude past which a copied ρ coordinate is treated as pinned at the
    /// optimizer box and is NOT transferred (the new default is used instead).
    pub rho_saturation: f64,
    /// Interior clamp magnitude applied to every transferred ρ coordinate, so
    /// a near-saturated-but-finite parent value still seeds inside the box.
    pub rho_interior_clamp: f64,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            rho_saturation: RHO_SATURATION,
            // Clamp transferred coordinates to a comfortable interior; the
            // outer optimizer expands back out if the data wants it. Mirrors
            // the `[CACHE] hit-clamp` interior policy.
            rho_interior_clamp: RHO_SATURATION - 1.0,
        }
    }
}

/// Errors that abort a transfer *build*. Callers treat any error as "use the
/// all-cold result" — a transfer must never fail a fit. The error variant is
/// retained so tests can assert which guard fired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransferError {
    /// The parent artifact failed its finite/schema usability guard.
    ParentUnusable,
    /// The descriptor keys disagree (the matcher handed us the wrong parent).
    DescriptorMismatch,
}

/// Build an all-cold result for the given new-fit layout: zero β at each
/// reduced block width, ρ set to the new fit's label-layout default.
pub fn cold_result(new_terms: &[TermBuildContext], rho_default: &Array1<f64>) -> TransferResult {
    TransferResult {
        block_beta: new_terms
            .iter()
            .map(|t| Array1::<f64>::zeros(t.reduced_block_width))
            .collect(),
        rho: rho_default.clone(),
        provenance: vec![TransferProvenance::Cold; new_terms.len()],
    }
}

/// Build a warm-start iterate for `new_descriptor` from `parent`.
///
/// Phase 1 contract:
///   - β: always cold (zeros at the new reduced block widths).
///   - ρ: per new term, if a parent term shares the [`TermIdentityKey`],
///     copy its converged `rho_for_term` into the term's ρ slots, clamped
///     into the interior and skipping any saturated parent coordinate;
///     otherwise leave the new fit's default in those slots.
///
/// On any anomaly the all-cold result is returned (with a `TransferError`
/// so callers can log which guard fired); the caller still gets a usable
/// behavior-neutral iterate.
pub fn build_warm_start(
    new_descriptor: &FitDescriptor,
    new_terms: &[TermBuildContext],
    rho_default: &Array1<f64>,
    parent: &FitArtifact,
    cfg: TransferConfig,
) -> Result<TransferResult, TransferError> {
    // Finite-guard the parent before reading any of its numbers.
    if !parent.is_usable() {
        return Err(TransferError::ParentUnusable);
    }
    // Defend against a mis-routed parent: descriptor keys must agree.
    if parent.descriptor.descriptor_key() != new_descriptor.descriptor_key() {
        return Err(TransferError::DescriptorMismatch);
    }

    // β stays cold in Phase 1.
    let block_beta: Vec<Array1<f64>> = new_terms
        .iter()
        .map(|t| Array1::<f64>::zeros(t.reduced_block_width))
        .collect();

    let mut rho = rho_default.clone();
    let mut provenance = vec![TransferProvenance::Cold; new_terms.len()];

    for (term_idx, new_term) in new_terms.iter().enumerate() {
        // Match this new term to a parent term by structural identity.
        let Some(parent_term) = parent
            .terms
            .iter()
            .find(|p| p.identity == new_term.identity)
        else {
            // Brand-new / unmatched term: keep the cold default in its slots.
            continue;
        };

        // The parent's per-term ρ must line up 1:1 with this term's ρ slots.
        // If the penalty layout differs (a structural change the identity
        // key did not capture), skip the term rather than mis-assign.
        if parent_term.rho_for_term.len() != new_term.rho_slots.len() {
            continue;
        }

        let mut copied_any = false;
        for (slot, &parent_rho) in new_term.rho_slots.iter().zip(parent_term.rho_for_term.iter()) {
            if *slot >= rho.len() {
                // Out-of-range slot index: defensive skip (never panic).
                continue;
            }
            if !parent_rho.is_finite() {
                continue;
            }
            // Saturation gate: a coordinate pinned at the optimizer box is
            // poor seed material — leave the default in this slot.
            if parent_rho.abs() >= cfg.rho_saturation {
                continue;
            }
            // Interior clamp: pull near-box-but-finite values inside.
            let clamped = parent_rho.clamp(-cfg.rho_interior_clamp, cfg.rho_interior_clamp);
            rho[*slot] = clamped;
            copied_any = true;
        }

        provenance[term_idx] = if copied_any {
            TransferProvenance::RhoOnly
        } else {
            TransferProvenance::Cold
        };
    }

    Ok(TransferResult {
        block_beta,
        rho,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::warm_start_artifact::{
        term_identity, GlobalFitSummary, ResponseSig, SerializableBasisMeta, TermArtifact, TermRole,
        FIT_ARTIFACT_SCHEMA,
    };
    use crate::terms::basis::types::{BasisMetadata, DuchonNullspaceOrder};
    use ndarray::{Array1, Array2};

    fn duchon_meta(n_centers: usize) -> BasisMetadata {
        BasisMetadata::Duchon {
            centers: Array2::<f64>::zeros((n_centers, 2)),
            length_scale: None,
            periodic: None,
            power: 1.0,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability_transform: None,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
        }
    }

    fn parent_with(
        identity: TermIdentityKey,
        meta: &BasisMetadata,
        rho_for_term: Vec<f64>,
    ) -> FitArtifact {
        FitArtifact {
            schema: FIT_ARTIFACT_SCHEMA,
            created_unix_secs: 0,
            descriptor: FitDescriptor {
                family_kind: "gaussian".to_string(),
                term_identities: vec![identity],
                response_signature: ResponseSig {
                    family_kind: "gaussian".to_string(),
                    n_response_channels: 1,
                },
                row_population: None,
            },
            terms: vec![TermArtifact {
                identity,
                role: TermRole::Mean,
                basis_meta: SerializableBasisMeta::from_metadata(meta),
                joint_null_rotation: None,
                raw_beta: vec![0.0; 5],
                rho_for_term,
            }],
            global: GlobalFitSummary {
                outer_objective: -10.0,
                converged: true,
                n_rows: 1000,
            },
        }
    }

    fn new_descriptor(identity: TermIdentityKey) -> FitDescriptor {
        FitDescriptor {
            family_kind: "gaussian".to_string(),
            term_identities: vec![identity],
            response_signature: ResponseSig {
                family_kind: "gaussian".to_string(),
                n_response_channels: 1,
            },
            row_population: None,
        }
    }

    #[test]
    fn matched_term_copies_parent_rho() {
        let meta = duchon_meta(10);
        let id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        let parent = parent_with(id, &meta, vec![2.5]);
        let new_terms = vec![TermBuildContext {
            identity: id,
            reduced_block_width: 9, // a different (fold) reduced width — still matches
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(res.rho[0], 2.5, "matched term must inherit parent ρ");
        assert_eq!(res.provenance[0], TransferProvenance::RhoOnly);
        // β stays cold at the NEW reduced width.
        assert_eq!(res.block_beta[0].len(), 9);
        assert!(res.block_beta[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn unmatched_term_keeps_default() {
        let meta = duchon_meta(10);
        let parent_id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        // The new term has a DIFFERENT identity (different variable), so even
        // though the descriptor keys are forced to agree, the per-term match
        // fails and the new default ρ is retained.
        let new_id = term_identity(TermRole::Mean, &["z".to_string()], &meta);
        let new_terms = vec![TermBuildContext {
            identity: new_id,
            reduced_block_width: 9,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![-1.3]);
        // Parent whose descriptor key matches the new fit (so the build is not
        // rejected up front) but whose single term carries `parent_id`, which
        // does not match `new_id` — isolating the per-term unmatched path.
        let mut parent = parent_with(new_id, &meta, vec![2.5]);
        parent.terms[0].identity = parent_id;
        let res = build_warm_start(
            &new_descriptor(new_id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(res.rho[0], -1.3, "unmatched term keeps the new default ρ");
        assert_eq!(res.provenance[0], TransferProvenance::Cold);
    }

    #[test]
    fn saturated_parent_rho_not_copied() {
        let meta = duchon_meta(10);
        let id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        // Parent ρ at the box: must NOT be copied.
        let parent = parent_with(id, &meta, vec![12.0]);
        let new_terms = vec![TermBuildContext {
            identity: id,
            reduced_block_width: 9,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.7]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(res.rho[0], 0.7, "saturated parent ρ must not be copied");
        assert_eq!(res.provenance[0], TransferProvenance::Cold);
    }

    #[test]
    fn near_box_parent_rho_is_interior_clamped() {
        let meta = duchon_meta(10);
        let id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        // Finite but near the box (below saturation): copied, then clamped.
        let parent = parent_with(id, &meta, vec![8.7]);
        let new_terms = vec![TermBuildContext {
            identity: id,
            reduced_block_width: 9,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.0]);
        let cfg = TransferConfig::default();
        let res = build_warm_start(&new_descriptor(id), &new_terms, &rho_default, &parent, cfg)
            .expect("transfer builds");
        assert!(res.rho[0] <= cfg.rho_interior_clamp);
        assert_eq!(res.rho[0], cfg.rho_interior_clamp);
        assert_eq!(res.provenance[0], TransferProvenance::RhoOnly);
    }

    #[test]
    fn nonfinite_parent_falls_back_cold() {
        let meta = duchon_meta(10);
        let id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        let mut parent = parent_with(id, &meta, vec![2.0]);
        parent.terms[0].raw_beta[0] = f64::NAN; // corrupt the parent
        let new_terms = vec![TermBuildContext {
            identity: id,
            reduced_block_width: 9,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.42]);
        let err = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, TransferError::ParentUnusable);
        // The caller's cold fallback yields the default ρ untouched.
        let cold = cold_result(&new_terms, &rho_default);
        assert_eq!(cold.rho[0], 0.42);
        assert!(cold.block_beta[0].iter().all(|&v| v == 0.0));
        assert_eq!(cold.provenance, vec![TransferProvenance::Cold]);
    }

    #[test]
    fn beta_is_always_cold_in_phase1() {
        // Exactness/no-op guard: regardless of what ρ transfers, β must remain
        // exactly zeros at the NEW reduced widths — a warm ρ only moves the
        // outer optimizer's starting iterate, it never seeds β, so it cannot
        // change the converged optimum (which still runs to the KKT/REML
        // certificate). This is the behavior-neutrality contract.
        let meta = duchon_meta(10);
        let id = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        let parent = parent_with(id, &meta, vec![3.3]);
        let new_terms = vec![TermBuildContext {
            identity: id,
            reduced_block_width: 12,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        // β identical to the all-cold result.
        let cold = cold_result(&new_terms, &rho_default);
        assert_eq!(res.block_beta.len(), cold.block_beta.len());
        for (w, c) in res.block_beta.iter().zip(cold.block_beta.iter()) {
            assert_eq!(w, c, "β must equal the cold-start β");
        }
        // ρ DID warm-start (that's the win), but β did not move.
        assert_eq!(res.rho[0], 3.3);
    }

    #[test]
    fn empty_terms_yields_empty_cold() {
        let res = cold_result(&[], &Array1::from_vec(vec![]));
        assert!(res.block_beta.is_empty());
        assert!(res.rho.is_empty());
        assert!(res.provenance.is_empty());
    }

    #[test]
    fn descriptor_mismatch_rejected() {
        let meta = duchon_meta(10);
        let id_a = term_identity(TermRole::Mean, &["x".to_string()], &meta);
        let id_b = term_identity(TermRole::Mean, &["z".to_string()], &meta);
        let parent = parent_with(id_a, &meta, vec![2.0]);
        let new_terms = vec![TermBuildContext {
            identity: id_b,
            reduced_block_width: 9,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![0.0]);
        let err = build_warm_start(
            &new_descriptor(id_b),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, TransferError::DescriptorMismatch);
    }
}
