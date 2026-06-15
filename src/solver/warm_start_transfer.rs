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

use crate::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use crate::solver::warm_start_artifact::{
    FitArtifact, FitDescriptor, RHO_SATURATION, TermIdentityKey, TransferProvenance,
};
use faer::Side;
use ndarray::{Array1, Array2};

/// Magnitude past which a *projected* reduced-coordinate β is treated as a
/// numerical blow-up and the block falls back to cold. A function-space warm
/// start only seeds the inner Newton's starting iterate, so a wild seed is
/// never a correctness hazard — but it is poor seed material, so we reject it.
const PROJECTED_BETA_CLAMP: f64 = 1.0e6;

/// Per-term context for the new (about-to-run) fit. Carries everything the
/// transfer needs to lay ρ out in the new fit's coordinate system, plus the
/// per-block gauge lift `T_b : reduced → raw` used to project the parent's
/// RAW β into this fold's reduced coordinates, without reaching back into the
/// solver internals.
#[derive(Clone, Debug)]
pub struct TermBuildContext {
    /// Structural identity of this new-fit term.
    pub identity: TermIdentityKey,
    /// Indices into the new fit's outer ρ vector that this term's penalties
    /// occupy (after label de-duplication). Empty for an unpenalized term.
    pub rho_slots: Vec<usize>,
    /// Reduced (post-identifiability) width of this term's block in the NEW
    /// fit — i.e. `spec.design.ncols()`. The cold β for this block is a zero
    /// vector of this length.
    pub reduced_width: usize,
    /// This block's slice of the NEW fit's gauge lift `T : reduced → raw`
    /// (shape `raw_width × reduced_width`, `β_raw = T · θ`). When present and
    /// its raw row count matches the parent term's `raw_beta` length, the
    /// parent's RAW β is least-squares projected onto this fold's reduced
    /// subspace via `θ = (TᵀT + εI)⁻¹ Tᵀ β_raw_parent`. `None` (or a raw-width
    /// mismatch) ⇒ β stays cold for this block.
    pub gauge_t_block: Option<Array2<f64>>,
}

/// Result of a warm-start build: the new ρ vector, the per-block warm β (in
/// the NEW fit's reduced coordinates), and a per-term provenance trace for
/// logging / tests.
///
/// `block_beta[b]` is always present with length `new_terms[b].reduced_width`:
/// it is the parent's RAW β projected into this fold's reduced subspace when
/// the term matched and the gauge made the projection well-defined, and a cold
/// zero vector otherwise. A warm β only seeds the inner Newton's starting
/// iterate; the solve still runs to its KKT certificate, so the converged
/// answer is identical to cold within tolerance.
#[derive(Clone, Debug)]
pub struct TransferResult {
    pub rho: Array1<f64>,
    pub block_beta: Vec<Array1<f64>>,
    pub provenance: Vec<TransferProvenance>,
}

/// Least-squares project a parent term's RAW β onto a new fold's reduced
/// subspace via the new gauge lift `T : reduced → raw` (`β_raw = T · θ`):
///
///   θ = argmin_θ ‖ T·θ − β_raw_parent ‖²  =  (TᵀT + εI)⁻¹ Tᵀ β_raw_parent.
///
/// This is the principled cross-fit coefficient-space transfer for the LOSO
/// case: the RAW basis (block name, #centers/knots, nullspace order) is
/// fold-invariant, so the parent's RAW β lives in the same raw column space as
/// the new fold even when the reduced width differs (e.g. p=37 vs p=35,
/// because the identifiability reduction `T` drops a fold-dependent number of
/// columns). Projecting through the NEW gauge lands a θ whose lifted raw β best
/// reproduces the parent's fitted function in the new reduced coordinates.
///
/// Returns `None` (cold fallback) on ANY anomaly — raw-width mismatch,
/// non-finite input, factorization failure, non-finite or blown-up output —
/// because a warm start can never error or distort a converged fit.
fn project_raw_beta_to_reduced(
    t_block: &Array2<f64>,
    raw_beta_parent: &[f64],
    reduced_width: usize,
) -> Option<Array1<f64>> {
    let (raw_rows, red_cols) = t_block.dim();
    if red_cols != reduced_width || raw_rows != raw_beta_parent.len() {
        return None;
    }
    if reduced_width == 0 {
        return Some(Array1::zeros(0));
    }
    if raw_beta_parent.iter().any(|v| !v.is_finite()) || t_block.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Normal equations with a small relative ridge so the system is SPD even
    // when `T` is rank-deficient in reduced space.
    let mut gram = fast_ata(t_block); // TᵀT, shape (red, red)
    let trace: f64 = (0..reduced_width).map(|i| gram[[i, i]]).sum();
    let eps = (1.0e-8 * trace / (reduced_width as f64)).max(1.0e-12);
    for i in 0..reduced_width {
        gram[[i, i]] += eps;
    }
    let rhs_col = Array2::from_shape_vec((raw_rows, 1), raw_beta_parent.to_vec()).ok()?;
    let rhs = fast_atb(t_block, &rhs_col); // Tᵀ β_raw, shape (red, 1)
    let rhs_vec = rhs.column(0).to_owned();
    let factor = gram.cholesky(Side::Lower).ok()?;
    let theta = factor.solvevec(&rhs_vec);
    if theta.len() != reduced_width
        || theta
            .iter()
            .any(|v| !v.is_finite() || v.abs() > PROJECTED_BETA_CLAMP)
    {
        return None;
    }
    Some(theta)
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
/// Contract:
///   - ρ: per new term, if a parent term shares the [`TermIdentityKey`],
///     copy its converged `rho_for_term` into the term's ρ slots, clamped
///     into the interior and skipping any saturated parent coordinate;
///     otherwise leave the new fit's default in those slots.
///   - β: per matched term, least-squares project the parent's RAW β onto this
///     fold's reduced subspace via the per-block gauge lift `T_b` (see
///     [`project_raw_beta_to_reduced`]). This delivers the cross-width LOSO
///     transfer (parent reduced width ≠ new reduced width). Any anomaly —
///     unmatched term, missing/mismatched gauge block, non-finite or blown-up
///     projection — falls back to a cold zero β for that block. β is in the
///     NEW fit's reduced (post-T) coordinates, the inner solver's working
///     space.
///
/// `block_beta[b]` is always present with length `new_terms[b].reduced_width`.
/// On any *whole-artifact* anomaly a `TransferError` is returned so callers can
/// log which guard fired and cold-start; a misfired transfer can never fail a
/// fit.
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

    let mut rho = rho_default.clone();
    let mut provenance = vec![TransferProvenance::Cold; new_terms.len()];
    // β defaults to cold (zeros at the new reduced block widths); a matched +
    // projectable term overwrites its block below.
    let mut block_beta: Vec<Array1<f64>> = new_terms
        .iter()
        .map(|t| Array1::<f64>::zeros(t.reduced_width))
        .collect();

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

        // β projection: reconstruct the parent term's fitted function in this
        // fold's reduced coordinates. Independent of the ρ-slot layout, so a
        // term whose penalty count drifted can still transfer β.
        let mut beta_projected = false;
        if let Some(t_block) = new_term.gauge_t_block.as_ref()
            && let Some(theta) =
                project_raw_beta_to_reduced(t_block, &parent_term.raw_beta, new_term.reduced_width)
        {
            block_beta[term_idx] = theta;
            beta_projected = true;
        }

        // The parent's per-term ρ must line up 1:1 with this term's ρ slots.
        // If the penalty layout differs (a structural change the identity
        // key did not capture), skip the ρ copy rather than mis-assign.
        let mut copied_any = false;
        if parent_term.rho_for_term.len() == new_term.rho_slots.len() {
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
        }

        provenance[term_idx] = if beta_projected {
            TransferProvenance::Projected
        } else if copied_any {
            TransferProvenance::RhoOnly
        } else {
            TransferProvenance::Cold
        };
    }

    Ok(TransferResult {
        rho,
        block_beta,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::warm_start_artifact::{
        FIT_ARTIFACT_SCHEMA, GlobalFitSummary, ResponseSig, SerializableBasisMeta, TermArtifact,
        TermRole, term_identity_from_block,
    };
    use ndarray::{Array1, Array2};

    /// Block-layer term identity (the surviving, fold-invariant identity API),
    /// one unlabeled penalty over a 1-dim nullspace.
    fn block_id(block_name: &str) -> TermIdentityKey {
        term_identity_from_block(TermRole::Mean, block_name, &[None], &[1])
    }

    /// A ρ-only term context (no gauge block): β stays cold, only ρ transfers.
    fn rho_only_ctx(identity: TermIdentityKey, rho_slots: Vec<usize>) -> TermBuildContext {
        TermBuildContext {
            identity,
            rho_slots,
            reduced_width: 0,
            gauge_t_block: None,
        }
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
        let new_terms = vec![rho_only_ctx(id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(new_id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(id, vec![0])];
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
        let new_terms = vec![rho_only_ctx(id_b, vec![0])];
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

    /// Build a parent artifact carrying an explicit RAW β of the given length.
    fn parent_with_raw_beta(
        identity: TermIdentityKey,
        raw_beta: Vec<f64>,
        rho_for_term: Vec<f64>,
    ) -> FitArtifact {
        let mut p = parent_with(identity, rho_for_term);
        p.terms[0].raw_beta = raw_beta;
        p
    }

    fn beta_ctx(
        identity: TermIdentityKey,
        rho_slots: Vec<usize>,
        reduced_width: usize,
        t_block: Array2<f64>,
    ) -> TermBuildContext {
        TermBuildContext {
            identity,
            rho_slots,
            reduced_width,
            gauge_t_block: Some(t_block),
        }
    }

    #[test]
    fn beta_projects_to_reduced_width() {
        // Identity gauge (raw width == reduced width == 3): the projection is
        // ≈ identity, so the projected θ reproduces the parent raw β.
        let id = block_id("s(x)");
        let raw = vec![1.0, -2.0, 3.5];
        let parent = parent_with_raw_beta(id, raw.clone(), vec![1.0]);
        let t = Array2::<f64>::eye(3);
        let new_terms = vec![beta_ctx(id, vec![0], 3, t)];
        let rho_default = Array1::from_vec(vec![0.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(res.block_beta[0].len(), 3, "β must be at the reduced width");
        for (got, want) in res.block_beta[0].iter().zip(raw.iter()) {
            assert!((got - want).abs() < 1e-6, "identity projection ≈ parent β");
        }
        assert_eq!(res.provenance[0], TransferProvenance::Projected);
    }

    #[test]
    fn cross_width_loso_case_transfers_beta() {
        // The marquee cross-fit win: the parent's RAW width (4) exceeds the new
        // fold's REDUCED width (2). The new gauge T (4×2) drops two raw columns
        // for this fold; projection lands a length-2 θ rather than skipping.
        let id = block_id("s(x)");
        let raw = vec![0.5, 0.5, 1.0, -1.0];
        let parent = parent_with_raw_beta(id, raw, vec![1.0]);
        // T sends reduced coord 0 -> raw 0,1 and reduced 1 -> raw 2,3, a
        // legitimate fold reduction (raw width 4, reduced width 2).
        let t =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        let new_terms = vec![beta_ctx(id, vec![0], 2, t)];
        let rho_default = Array1::from_vec(vec![0.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(
            res.block_beta[0].len(),
            2,
            "cross-width LOSO must project to the new reduced width, not skip"
        );
        assert!(res.block_beta[0].iter().all(|v| v.is_finite()));
        assert_eq!(res.provenance[0], TransferProvenance::Projected);
    }

    #[test]
    fn beta_dimension_anomaly_falls_back_to_cold() {
        // The gauge block's raw rows (3) disagree with the parent raw β length
        // (5): the projection is undefined, so β must fall back to cold zeros.
        let id = block_id("s(x)");
        let parent = parent_with_raw_beta(id, vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.0]);
        let t = Array2::<f64>::eye(3); // raw rows = 3 ≠ parent raw_beta len 5
        let new_terms = vec![beta_ctx(id, vec![0], 3, t)];
        let rho_default = Array1::from_vec(vec![0.0]);
        let res = build_warm_start(
            &new_descriptor(id),
            &new_terms,
            &rho_default,
            &parent,
            TransferConfig::default(),
        )
        .expect("transfer builds");
        assert_eq!(
            res.block_beta[0].len(),
            3,
            "cold β still at the reduced width"
        );
        assert!(
            res.block_beta[0].iter().all(|&v| v == 0.0),
            "dimension anomaly must yield cold zeros"
        );
        // ρ still transferred, so the term is RhoOnly, not Projected.
        assert_eq!(res.provenance[0], TransferProvenance::RhoOnly);
    }

    #[test]
    fn beta_nonfinite_parent_is_globally_rejected() {
        // A non-finite raw β fails the whole-artifact usability guard before any
        // per-block projection runs — the build errors and the caller cold-starts.
        let id = block_id("s(x)");
        let mut parent = parent_with_raw_beta(id, vec![1.0, 0.0, 3.0], vec![1.0]);
        parent.terms[0].raw_beta[1] = f64::NAN;
        let t = Array2::<f64>::eye(3);
        let new_terms = vec![beta_ctx(id, vec![0], 3, t)];
        let rho_default = Array1::from_vec(vec![0.0]);
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
    fn projection_helper_identity_is_exact() {
        let raw = vec![2.0, -1.0, 0.0, 4.0];
        let t = Array2::<f64>::eye(4);
        let theta = project_raw_beta_to_reduced(&t, &raw, 4).expect("projects");
        for (g, w) in theta.iter().zip(raw.iter()) {
            assert!((g - w).abs() < 1e-7);
        }
    }
}
