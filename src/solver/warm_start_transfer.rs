//! Cross-fit warm-start *transfer*: build a starting ρ iterate for a new fit
//! from a structurally-matching parent [`FitArtifact`].
//!
//! This is the **ρ half** of the transfer (the marquee LOSO win): for each new
//! term, if a parent term shares its [`TermIdentityKey`], copy the parent's
//! converged log-smoothing parameters into the new ρ layout — clamped out of
//! the saturated box. Unmatched / brand-new terms fall back to the new fit's
//! penalty-label default. β always stays cold (zeros at the new reduced block
//! widths, seeded by the caller); only ρ transfers.
//!
//! Every path is exactness-preserving: a warm ρ only changes where the outer
//! optimizer *starts*; it still runs to its KKT/REML certificate, so the
//! converged optimum is identical to a cold start within tolerance. On any
//! anomaly — missing parent, descriptor mismatch, non-finite payload — the
//! build returns an error and the caller cold-starts, so a misfired transfer
//! can never fail a fit.

use crate::solver::warm_start_artifact::{
    FitArtifact, FitDescriptor, RHO_SATURATION, TermIdentityKey, TransferProvenance,
};
use ndarray::Array1;

/// Per-term context for the new (about-to-run) fit. Carries everything the
/// transfer needs to lay ρ out in the new fit's coordinate system without
/// reaching back into the solver internals.
#[derive(Clone, Debug)]
pub struct TermBuildContext {
    /// Structural identity of this new-fit term.
    pub identity: TermIdentityKey,
    /// Indices into the new fit's outer ρ vector that this term's penalties
    /// occupy (after label de-duplication). Empty for an unpenalized term.
    pub rho_slots: Vec<usize>,
}

/// Result of a warm-start build: the new ρ vector and a per-term provenance
/// trace for logging / tests. β stays cold (the new fit's reduced block
/// widths, all zeros) and is not materialized here — only ρ transfers.
#[derive(Clone, Debug)]
pub struct TransferResult {
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

/// Build a warm-start iterate for `new_descriptor` from `parent`.
///
/// Phase 1 contract:
///   - β: always cold (zeros at the new reduced block widths).
///   - ρ: per new term, if a parent term shares the [`TermIdentityKey`],
///     copy its converged `rho_for_term` into the term's ρ slots, clamped
///     into the interior and skipping any saturated parent coordinate;
///     otherwise leave the new fit's default in those slots.
///
/// On any anomaly a `TransferError` is returned so callers can log which
/// guard fired and cold-start; a misfired transfer can never fail a fit.
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

    // β stays cold in Phase 1: only ρ transfers, so β is never materialized
    // here — the caller seeds zeros at the new reduced block widths.
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
        for (slot, &parent_rho) in new_term
            .rho_slots
            .iter()
            .zip(parent_term.rho_for_term.iter())
        {
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

    Ok(TransferResult { rho, provenance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::warm_start_artifact::{
        FIT_ARTIFACT_SCHEMA, GlobalFitSummary, ResponseSig, SerializableBasisMeta, TermArtifact,
        TermRole, term_identity_from_block,
    };
    use ndarray::Array1;

    /// Block-layer term identity (the surviving, fold-invariant identity API),
    /// one unlabeled penalty over a 1-dim nullspace.
    fn block_id(block_name: &str) -> TermIdentityKey {
        term_identity_from_block(TermRole::Mean, block_name, &[None], &[1])
    }

    /// Minimal serializable basis-meta stub, as produced at the block-spec
    /// capture layer.
    fn basis_meta_stub() -> SerializableBasisMeta {
        SerializableBasisMeta {
            kind: "block-spec".to_string(),
            degree: None,
            num_knots: None,
            n_centers: Some(5),
            nullspace_order: None,
            matern_nu: None,
            periodic: false,
        }
    }

    fn parent_with(identity: TermIdentityKey, rho_for_term: Vec<f64>) -> FitArtifact {
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
                basis_meta: basis_meta_stub(),
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
        let id = block_id("s(x)");
        let parent = parent_with(id, vec![2.5]);
        let new_terms = vec![TermBuildContext {
            identity: id,
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
    }

    #[test]
    fn unmatched_term_keeps_default() {
        let parent_id = block_id("s(x)");
        // The new term has a DIFFERENT identity (different block), so even
        // though the descriptor keys are forced to agree, the per-term match
        // fails and the new default ρ is retained.
        let new_id = block_id("s(z)");
        let new_terms = vec![TermBuildContext {
            identity: new_id,
            rho_slots: vec![0],
        }];
        let rho_default = Array1::from_vec(vec![-1.3]);
        // Parent whose descriptor key matches the new fit (so the build is not
        // rejected up front) but whose single term carries `parent_id`, which
        // does not match `new_id` — isolating the per-term unmatched path.
        let mut parent = parent_with(new_id, vec![2.5]);
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
        let id = block_id("s(x)");
        // Parent ρ at the box: must NOT be copied.
        let parent = parent_with(id, vec![12.0]);
        let new_terms = vec![TermBuildContext {
            identity: id,
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
        let id = block_id("s(x)");
        // Finite but near the box (below saturation): copied, then clamped.
        let parent = parent_with(id, vec![8.7]);
        let new_terms = vec![TermBuildContext {
            identity: id,
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
    fn nonfinite_parent_is_rejected() {
        let id = block_id("s(x)");
        let mut parent = parent_with(id, vec![2.0]);
        parent.terms[0].raw_beta[0] = f64::NAN; // corrupt the parent
        let new_terms = vec![TermBuildContext {
            identity: id,
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
    }

    #[test]
    fn rho_only_transfer_leaves_unrelated_slots_at_default() {
        // Behavior-neutrality contract: only matched ρ slots move; everything
        // else keeps the new fit's default. A warm ρ merely shifts the outer
        // optimizer's starting iterate; it still runs to the KKT/REML
        // certificate, so the converged optimum is unchanged.
        let id = block_id("s(x)");
        let parent = parent_with(id, vec![3.3]);
        let new_terms = vec![TermBuildContext {
            identity: id,
            rho_slots: vec![0],
        }];
        // Slot 1 belongs to no transferred term and must stay at its default.
        let rho_default = Array1::from_vec(vec![0.0, -2.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(res.rho[0], 3.3, "matched slot warm-starts");
        assert_eq!(res.rho[1], -2.0, "unrelated slot keeps the default");
    }

    #[test]
    fn descriptor_mismatch_rejected() {
        let id_a = block_id("s(x)");
        let id_b = block_id("s(z)");
        let parent = parent_with(id_a, vec![2.0]);
        let new_terms = vec![TermBuildContext {
            identity: id_b,
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
