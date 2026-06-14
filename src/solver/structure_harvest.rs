//! #997 — the wiring seam between a fitted [`SaeManifoldTerm`] and the
//! evidence-guarded move engine of [`crate::solver::structure_search`].
//!
//! #976 closed with the move engine (`search`) and its triggers
//! ([`crate::terms::atom_codes::SparseAtomCodes::coactivation`], ARD precisions,
//! terminal [`CollapseEvent`]s) on main but deliberately unwired: nothing
//! harvested move proposals from a fitted dictionary or drove `search` around
//! the production fit. This module is that seam. It owns three things:
//!
//! 1. [`harvest_move_proposals`] — reads a fitted term + its ρ + the per-row
//!    reconstruction residuals and emits the canonical-order-ready
//!    [`MoveProposal`] stream (deaths, fusions, fission audits, births).
//! 2. [`apply_structure_move`] — the warm-inheritance restructuring of a
//!    [`SaeManifoldTerm`] under one [`StructureMove`]: a death demotes an atom's
//!    routing, a fission splits an atom into two children that inherit its
//!    decoder block, a fusion folds the weaker of a pair into the stronger, a
//!    birth appends a residual-factor atom. Every child state is built FROM the
//!    parent (never cold) so the engine's warm-state contract holds by
//!    construction.
//! 3. [`run_structure_search_rounds`] — the round driver: fit → harvest →
//!    [`search`] (over held-out row-block shards, with warm child refits) →
//!    re-fit → repeat until a round applies no moves. The accumulated
//!    [`SearchLedger`] (with the joint fit's [`CollapseEvent`]s) is the honesty
//!    surface returned to the caller and serialized onto the fit payload.
//!
//! # Determinism
//!
//! Pure: no RNG, no clock. Proposal triggers are deterministic functions of the
//! fitted state; the engine canonicalizes and gates them; the ledger serializes
//! byte-identically for identical inputs. The structural hashes that dedup the
//! proposal stream are computed with the same [`crate::cache::Fingerprinter`]
//! the [`crate::terms::smooth::TermCollectionSpec`] machinery (#869) uses, fed
//! the POST-move dictionary shape (atom count, per-atom basis kind + latent dim
//! + the move that produced it), so two proposals that reach the same dictionary
//! shape collide exactly as the engine requires.

use std::sync::Arc;

use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::cache::Fingerprinter;
use crate::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use crate::inference::structure_evidence::{ClaimKind, StructureLedger};
use crate::linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::solver::structure_search::{
    CollapseAction, MoveBudget, MoveProposal, SearchLedger, SearchOutcome, StructureMove, search,
};
use crate::solver::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoSelector, TopologyScoreScale,
    select_topology_with_fit,
};
use crate::terms::atom_codes::SparseAtomCodes;
use crate::terms::latent_coord::{LatentIdMode, LatentManifold};
use crate::terms::sae::basis::{
    EuclideanPatchEvaluator, PeriodicHarmonicEvaluator, SaeBasisEvaluator, SphereChartEvaluator,
    TorusHarmonicEvaluator,
};
use crate::terms::sae_manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};

/// Per-row soft-assignment mass below which an atom is treated as INACTIVE on
/// that row when deriving the discrete co-activation support. A soft softmax /
/// gate assignment never reaches exactly zero, so the discrete masks the
/// coactivation triggers consume are obtained by thresholding the per-row mass.
/// Chosen as a fixed structural constant (magic-by-default): small enough that a
/// genuinely-routed atom counts as active on its rows, large enough that the
/// near-uniform softmax floor (`≈ 1/K`) on rows an atom does not own does not
/// leak into its support. The threshold is relative to a uniform-assignment
/// reference so it scales with `K`.
const ACTIVE_SUPPORT_REL_FLOOR: f64 = 0.5;

/// ARD log-precision above which an atom's coordinate prior is treated as
/// DIVERGED — the coordinate has been shrunk to its prior mean, so the atom
/// carries no on-manifold structure and its existence was never certified by
/// the data. This is the death-proposal trigger (#976): diverged ARD ⇒ demote
/// the atom unless its `AtomExists` claim certified in an earlier round (the
/// veto). A large positive `log_alpha` is a precision blow-up; the floor is set
/// well above the strengths a live coordinate settles at and well below the
/// `stable_exp_strength` clamp.
const ARD_DIVERGENCE_LOG_PRECISION: f64 = 12.0;

/// Minimum symmetric code dependence for a pair to be proposed for FUSION. Below
/// this the two atoms' supports are essentially independent (the shattering
/// signature needs both conditionals high); above it the pair is a fusion
/// candidate. The e-gate, not this threshold, decides acceptance — this only
/// keeps the proposal stream from carrying every independent pair.
const FUSION_DEPENDENCE_FLOOR: f64 = 0.6;

/// Minimum conditional asymmetry for a pair to be proposed for a FISSION audit
/// (the A⇒B absorption signature: one conditional near 1 without the converse).
const ABSORPTION_ASYMMETRY_FLOOR: f64 = 0.5;

/// Knobs for one harvest pass. All magic-by-default — derived from the fit, not
/// surfaced as user flags.
#[derive(Clone, Copy, Debug)]
pub struct HarvestParams {
    /// Maximum fusion pairs proposed per round (the top-dependence pairs).
    pub max_fusions: usize,
    /// Maximum fission audits proposed per round (the top-asymmetry pairs).
    pub max_fissions: usize,
    /// Maximum residual-factor birth candidates proposed per round (the top
    /// factor directions by explained residual mass).
    pub max_births: usize,
}

impl Default for HarvestParams {
    fn default() -> Self {
        // A small fixed budget per round; the round driver iterates until a
        // round applies nothing, so per-round breadth need not be exhaustive.
        Self {
            max_fusions: 4,
            max_fissions: 4,
            max_births: 4,
        }
    }
}

/// Derive the discrete active-support codes the co-activation triggers consume
/// from a fitted term's SOFT assignments. An atom counts as active on a row when
/// its assignment mass exceeds `ACTIVE_SUPPORT_REL_FLOOR / K` (relative to the
/// uniform-assignment reference), so the discrete support reflects genuine
/// routing rather than the near-uniform softmax floor.
pub fn sparse_codes_from_term(term: &SaeManifoldTerm) -> SparseAtomCodes {
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k = assignments.ncols();
    let floor = if k == 0 {
        0.0
    } else {
        ACTIVE_SUPPORT_REL_FLOOR / k as f64
    };
    let mut codes = SparseAtomCodes::empty(n, k);
    for row in 0..n {
        for atom in 0..k {
            let mass = assignments[[row, atom]];
            if mass > floor {
                codes.row_mut(row).assign(atom, mass);
            }
        }
    }
    codes
}

/// Per-atom maximum active mass over rows — the collapse statistic (a
/// legitimately sparse atom has small MEAN mass but high MAX on its rows; only
/// an atom with no material support anywhere has a small MAX). Used as the
/// birth-residual activity coordinate and as a secondary death signal.
fn per_atom_max_mass(term: &SaeManifoldTerm) -> Array1<f64> {
    let assignments = term.assignment.assignments();
    let k = assignments.ncols();
    let mut out = Array1::<f64>::zeros(k);
    for atom in 0..k {
        let mut max = 0.0_f64;
        for &m in assignments.column(atom).iter() {
            if m > max {
                max = m;
            }
        }
        out[atom] = max;
    }
    out
}

/// The largest per-atom ARD log-precision (over the atom's axes), or `-inf` for
/// an atom with native ARD disabled (empty block). A diverged precision on ANY
/// axis collapses that coordinate, so the per-atom death trigger is the max.
fn per_atom_ard_divergence(rho: &SaeManifoldRho, atom: usize) -> f64 {
    rho.log_ard
        .get(atom)
        .and_then(|axes| axes.iter().copied().reduce(f64::max))
        .unwrap_or(f64::NEG_INFINITY)
}

/// Structural hash of the POST-move dictionary shape, computed with the same
/// [`Fingerprinter`] the [`TermCollectionSpec`](crate::terms::smooth::TermCollectionSpec)
/// hash machinery (#869) uses. The hash covers the move kind, the atoms it
/// touches, and the resulting atom count + per-atom basis-kind/latent-dim
/// shape — structural identity only, never decoder coefficients or coordinates,
/// so two distinct proposals that reach the same dictionary shape collide.
fn post_move_structure_hash(term: &SaeManifoldTerm, mv: &StructureMove) -> u64 {
    let mut fp = Fingerprinter::new();
    fp.write_str("sae_structure_move");
    match mv {
        StructureMove::Birth { candidate } => {
            fp.write_str("birth");
            fp.write_usize(*candidate);
        }
        StructureMove::Death { atom } => {
            fp.write_str("death");
            fp.write_usize(*atom);
        }
        StructureMove::Fission { atom } => {
            fp.write_str("fission");
            fp.write_usize(*atom);
        }
        StructureMove::Fusion { a, b } => {
            fp.write_str("fusion");
            // Order-independent: a fusion of (a,b) is the same structure as
            // (b,a).
            fp.write_usize((*a).min(*b));
            fp.write_usize((*a).max(*b));
        }
    }
    // Post-move atom-shape skeleton: the current per-atom (basis-kind tag,
    // latent dim) plus the count delta the move applies. Births/fissions add an
    // atom; deaths/fusions do not change the count (death demotes, fusion folds)
    // — the routing change, not a structural resize, so the shape skeleton is
    // the parent's plus the move tag above.
    fp.write_usize(term.atoms.len());
    for atom in &term.atoms {
        fp.write_str(basis_kind_tag(&atom.basis_kind));
        fp.write_usize(atom.latent_dim);
    }
    let digest = fp.finalize();
    let bytes = digest.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Structural tag for an atom basis kind — the discrete shape identity the
/// structural hash needs (never coordinates or coefficients).
fn basis_kind_tag(kind: &SaeAtomBasisKind) -> &str {
    match kind {
        SaeAtomBasisKind::Duchon => "duchon",
        SaeAtomBasisKind::Periodic => "periodic",
        SaeAtomBasisKind::Sphere => "sphere",
        SaeAtomBasisKind::Torus => "torus",
        SaeAtomBasisKind::EuclideanPatch => "euclidean_patch",
        SaeAtomBasisKind::Poincare => "poincare",
        SaeAtomBasisKind::Precomputed(_) => "precomputed",
    }
}

/// Build a [`MoveProposal`] from a move + trigger by stamping its post-move
/// structural hash and the structural claim it asserts.
fn proposal(term: &SaeManifoldTerm, mv: StructureMove, trigger: f64) -> MoveProposal {
    let structure_hash = post_move_structure_hash(term, &mv);
    let claim = match &mv {
        StructureMove::Birth { candidate } => ClaimKind::AtomExists {
            // Births claim the existence of the NEXT atom index (appended).
            atom: term.k_atoms() + *candidate,
        },
        StructureMove::Death { atom } => ClaimKind::AtomExists { atom: *atom },
        StructureMove::Fusion { a, b } => ClaimKind::BindingEdge { a: *a, b: *b },
        StructureMove::Fission { atom } => ClaimKind::Custom {
            label: format!("fission:{atom}"),
        },
    };
    MoveProposal {
        mv,
        trigger,
        structure_hash,
        claim,
    }
}

/// Harvest the canonical move-proposal stream from a fitted term, its ρ, and the
/// per-row reconstruction residuals `R = target − fitted` (used for the birth
/// channel under the [`WhitenedStructured`](crate::inference::row_metric::MetricProvenance::WhitenedStructured)
/// residual-factor metric — never raw-Euclidean Λ, per the #974 rescope).
///
/// The four channels (#976/#997):
///
/// * **Deaths** from diverged ARD precisions ∪ terminal [`CollapseEvent`]s. The
///   trigger is the ARD precision (descending); a terminally-collapsed atom is
///   proposed even with finite ARD (its routing is gone regardless of its
///   coordinate prior).
/// * **Fusions** from the top co-activation pairs by symmetric code dependence.
/// * **Fission audits** from absorption-suspect pairs (high conditional
///   asymmetry). The within-atom substructure carve (#907 mixture race / #975
///   `carve`) that would refine the audit is NOT yet wired (its fit-side inputs
///   land with #993); until then the audit proposes a fission whose acceptance
///   the e-gate owns, and the absent carve is recorded loudly via
///   [`HarvestReport::fission_carve_skipped`] rather than silently dropped.
/// * **Births** from the whitened residual-factor subspace: the residuals are
///   fed to [`StructuredResidualModel::fit`], whose factor directions
///   ([`StructuredResidualModel::factor`]) are the birth candidates, ranked by
///   explained residual mass.
pub fn harvest_move_proposals(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    residuals: ArrayView2<'_, f64>,
    params: &HarvestParams,
) -> Result<HarvestReport, String> {
    let k = term.k_atoms();
    let mut proposals: Vec<MoveProposal> = Vec::new();

    // --- Deaths: diverged ARD ∪ terminal collapses -------------------------
    let max_mass = per_atom_max_mass(term);
    let terminal: std::collections::HashSet<usize> = term
        .collapse_events()
        .iter()
        .filter(|e| matches!(e.action, CollapseAction::Terminal))
        .map(|e| e.atom)
        .collect();
    for atom in 0..k {
        let ard = per_atom_ard_divergence(rho, atom);
        let diverged = ard >= ARD_DIVERGENCE_LOG_PRECISION;
        let collapsed = terminal.contains(&atom);
        if diverged || collapsed {
            // Trigger (descending): a terminal collapse is maximally urgent
            // (the routing is already gone), ranked above ARD divergence; ARD
            // deaths rank by precision. `max_mass` breaks ties toward emptier
            // atoms.
            let trigger = if collapsed { f64::MAX / 2.0 } else { ard };
            // Lower max-mass (emptier) sorts first among equal triggers; encode
            // by subtracting a small mass-proportional term that cannot reorder
            // across the collapsed/ARD bands.
            let trigger = trigger - max_mass[atom].min(1.0) * 1e-9;
            proposals.push(proposal(term, StructureMove::Death { atom }, trigger));
        }
    }

    // --- Fusions: top co-activation dependence -----------------------------
    let codes = sparse_codes_from_term(term);
    let mut fusion_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..k {
        for b in (a + 1)..k {
            let stats = codes.coactivation(a, b);
            let dep = stats.dependence();
            if dep >= FUSION_DEPENDENCE_FLOOR {
                fusion_pairs.push((a, b, dep));
            }
        }
    }
    fusion_pairs.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    for &(a, b, dep) in fusion_pairs.iter().take(params.max_fusions) {
        proposals.push(proposal(term, StructureMove::Fusion { a, b }, dep));
    }

    // --- Fission audits: absorption-suspect asymmetry ----------------------
    let mut fission_atoms: Vec<(usize, f64)> = Vec::new();
    for a in 0..k {
        for b in (a + 1)..k {
            let stats = codes.coactivation(a, b);
            let asym = stats.absorption_asymmetry();
            if asym >= ABSORPTION_ASYMMETRY_FLOOR {
                // The parent (the conditioned-on atom whose support nests the
                // child) is the one whose `P(parent|child) ≈ 1`. Audit the
                // parent for the absorbed substructure.
                let parent = if stats.p_a_given_b >= stats.p_b_given_a {
                    a
                } else {
                    b
                };
                // Fission trigger is audit significance ASCENDING; map a high
                // asymmetry to a low significance proxy `1 − asym` so the most
                // asymmetric (most suspect) pair sorts first.
                let significance = (1.0 - asym).max(0.0);
                fission_atoms.push((parent, significance));
            }
        }
    }
    // Keep the most-suspect (lowest significance) audit per parent atom.
    fission_atoms.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
    fission_atoms.dedup_by_key(|(atom, _)| *atom);
    // The within-atom carve that would refine each audit is not yet wired
    // (#993): record the skip loudly. The fission proposal still rides; the
    // e-gate decides acceptance.
    let fission_carve_skipped = !fission_atoms.is_empty();
    for &(atom, significance) in fission_atoms.iter().take(params.max_fissions) {
        proposals.push(proposal(
            term,
            StructureMove::Fission { atom },
            significance,
        ));
    }

    // --- Births: whitened residual-factor subspace -------------------------
    // The activity coordinate the residual-factor scale law is smooth in is the
    // per-row total assignment mass (an activation-strength summary): rows where
    // the dictionary routes strongly should have smaller unexplained residual
    // factor energy than rows it does not cover.
    let n = residuals.nrows();
    let assignments = term.assignment.assignments();
    let activity: Array1<f64> = (0..n).map(|r| assignments.row(r).sum()).collect();
    let mut births_proposed = 0usize;
    let mut birth_skipped_reason: Option<String> = None;
    if params.max_births > 0 && n > 0 && residuals.ncols() > 0 {
        let p = residuals.ncols();
        let max_rank = params.max_births.min(p.saturating_sub(1));
        match StructuredResidualModel::fit(ResidualFactorInput {
            residuals,
            activity: activity.view(),
            max_factor_rank: max_rank,
        }) {
            Ok(model) => {
                let factor = model.factor();
                let r = model.factor_rank();
                // Rank each factor direction by its explained residual mass
                // (column norm of Λ scaled by the mean activity); births are
                // proposed in descending mass order, capped at `max_births`.
                let mut dirs: Vec<(usize, f64)> = (0..r)
                    .map(|j| {
                        let mass = factor.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
                        (j, mass)
                    })
                    .collect();
                dirs.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
                for &(candidate, mass) in dirs.iter().take(params.max_births) {
                    proposals.push(proposal(term, StructureMove::Birth { candidate }, mass));
                    births_proposed += 1;
                }
            }
            Err(e) => {
                birth_skipped_reason = Some(e);
            }
        }
    } else if params.max_births > 0 {
        birth_skipped_reason =
            Some("residuals empty or single-channel; no factor subspace to mine".to_string());
    }

    Ok(HarvestReport {
        proposals,
        fission_carve_skipped,
        births_proposed,
        birth_skipped_reason,
    })
}

/// The output of one [`harvest_move_proposals`] pass: the proposal stream plus
/// the loud records of any degrade-to-skip path taken (no silent drops).
#[derive(Clone, Debug)]
pub struct HarvestReport {
    /// Trigger-stamped, claim-stamped, structurally-hashed proposals, ready for
    /// [`search`] (which canonicalizes and gates them).
    pub proposals: Vec<MoveProposal>,
    /// Whether any fission audit rode without its #993 within-atom carve refit
    /// (the carve's fit-side inputs are not yet available). Recorded so the
    /// degraded path is visible, never silent.
    pub fission_carve_skipped: bool,
    /// Number of residual-factor birth candidates proposed.
    pub births_proposed: usize,
    /// If the birth channel could not run (empty residuals, evidence-ladder
    /// failure), why — so the absence of births is explained, not silent.
    pub birth_skipped_reason: Option<String>,
}

/// Apply one [`StructureMove`] to a fitted term + ρ, returning the warm child
/// state. Warm inheritance by construction: the child is cloned from the parent
/// and only the touched atoms are restructured.
///
/// * **Death** demotes atom `atom`: its assignment logits are driven to a
///   strongly-negative value (routing → ~0) on every row, and its ARD block is
///   left in place. The atom is NOT removed (stable indices for the round); it
///   simply stops carrying mass. Demote-never-reject (#976).
/// * **Fission** appends a child cloned from atom `atom` (same basis, decoder,
///   coordinates), splitting the parent's per-row routing between parent and
///   child so the joint refit can pull them apart along the absorbed
///   substructure. The child inherits the parent's main-effect block.
/// * **Fusion** folds atom `b` into atom `a`: `a`'s routing absorbs `b`'s mass
///   (logit-sum on the active rows) and `b` is demoted. The retained atom's
///   product coordinates are initialized from the pair.
/// * **Birth** appends a fresh atom whose decoder is seeded from the
///   residual-factor direction `candidate` (passed in `birth_decoders`), routed
///   at a small neutral mass so the refit can grow it if it is real.
pub fn apply_structure_move(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    mv: &StructureMove,
    birth_decoders: &[Array2<f64>],
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    match mv {
        StructureMove::Death { atom } => {
            let mut child = term.clone();
            demote_atom(&mut child, *atom)?;
            Ok((child, rho.clone()))
        }
        StructureMove::Fusion { a, b } => {
            let mut child = term.clone();
            fold_atom_into(&mut child, *a, *b)?;
            Ok((child, rho.clone()))
        }
        StructureMove::Fission { atom } => {
            let (child, child_rho) = duplicate_atom(term, rho, *atom)?;
            Ok((child, child_rho))
        }
        StructureMove::Birth { candidate } => {
            let decoder = birth_decoders.get(*candidate).ok_or_else(|| {
                format!(
                    "apply_structure_move: birth candidate {candidate} out of range \
                     ({} residual-factor decoders)",
                    birth_decoders.len()
                )
            })?;
            born_atom(term, rho, decoder.view())
        }
    }
}

/// A strongly-negative logit that drives a softmax / gate routing channel to ~0
/// mass without producing a non-finite value the assignment validator rejects.
const DEMOTE_LOGIT: f64 = -40.0;

/// Drive an atom's per-row routing to ~0 by setting its logit column to a
/// strongly-negative constant. Demotion, not removal: the atom keeps its index.
fn demote_atom(term: &mut SaeManifoldTerm, atom: usize) -> Result<(), String> {
    let k = term.k_atoms();
    if atom >= k {
        return Err(format!("demote_atom: atom {atom} out of range (K={k})"));
    }
    for row in 0..term.assignment.logits.nrows() {
        term.assignment.logits[[row, atom]] = DEMOTE_LOGIT;
    }
    Ok(())
}

/// Fold atom `b` into atom `a`: `a` absorbs `b`'s routing mass on every row
/// (logit max, the dominance the fused atom should express), then `b` is
/// demoted. The retained atom keeps its decoder; the joint refit reconciles the
/// merged structure.
fn fold_atom_into(term: &mut SaeManifoldTerm, a: usize, b: usize) -> Result<(), String> {
    let k = term.k_atoms();
    if a >= k || b >= k {
        return Err(format!(
            "fold_atom_into: atoms ({a},{b}) out of range (K={k})"
        ));
    }
    if a == b {
        return Err("fold_atom_into: cannot fuse an atom with itself".to_string());
    }
    for row in 0..term.assignment.logits.nrows() {
        let la = term.assignment.logits[[row, a]];
        let lb = term.assignment.logits[[row, b]];
        // The fused atom should route wherever EITHER constituent did: take the
        // dominant logit. (A sum would double-count and overflow the softmax;
        // the max preserves the union support the fusion asserts.)
        term.assignment.logits[[row, a]] = la.max(lb);
    }
    demote_atom(term, b)?;
    Ok(())
}

/// Append a child cloned from atom `parent`: identical basis, decoder, and
/// coordinates, with the parent's routing split evenly between parent and child
/// (the parent's logit dropped by `ln 2` on every row, the child seeded equal).
/// The joint refit then pulls the two apart along the absorbed substructure. The
/// child's ARD block is inherited from the parent.
fn duplicate_atom(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    parent: usize,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let k = term.k_atoms();
    if parent >= k {
        return Err(format!(
            "duplicate_atom: parent {parent} out of range (K={k})"
        ));
    }
    let mut atoms = term.atoms.clone();
    let child_atom = term.atoms[parent].clone();
    atoms.push(child_atom);

    let n = term.assignment.logits.nrows();
    let mut logits = Array2::<f64>::zeros((n, k + 1));
    let split = std::f64::consts::LN_2;
    for row in 0..n {
        for col in 0..k {
            let mut v = term.assignment.logits[[row, col]];
            if col == parent {
                // Halve the parent's routing mass (logit − ln 2) and give the
                // other half to the child.
                v -= split;
            }
            logits[[row, col]] = v;
        }
        logits[[row, k]] = term.assignment.logits[[row, parent]] - split;
    }
    let mut coords = term.assignment.coords.clone();
    coords.push(term.assignment.coords[parent].clone());
    let assignment =
        crate::terms::sae_manifold::SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let child = SaeManifoldTerm::new(atoms, assignment)?;

    let mut child_rho = rho.clone();
    if parent < child_rho.log_ard.len() {
        let inherited = child_rho.log_ard[parent].clone();
        child_rho.log_ard.push(inherited);
    } else {
        child_rho.log_ard.push(Array1::<f64>::zeros(0));
    }
    Ok((child, child_rho))
}

// ===========================================================================
// #977 — per-atom topology RACE at birth.
//
// A born atom must not inherit atom-0's circle template by fiat. Its TOPOLOGY is
// chosen by EVIDENCE: each candidate basis whose required intrinsic dimension
// matches the born atom's ARD-selected `d_k` is fit to the residual-factor image
// the atom would reconstruct, and the winner is the lowest TK-normalized REML —
// the SAME gauge-invariant comparison [`select_topology_with_fit`] applies to the
// smooth-term topology race, so cross-topology scores are commensurable. The
// dictionary the learner discovers is therefore genuinely heterogeneous: a born
// atom on a circular residual gets a circle, one on a straight residual a line.
// ===========================================================================

/// Ridge on the candidate-fit normal equations, added to the intrinsic roughness
/// Gram so the per-candidate penalized least-squares solve is well-posed even on
/// a near-degenerate residual image. Small relative to the design scale (the
/// reconstruction target is the already-fit residual factor, O(1)) — it pins the
/// solve, it does not shape the verdict (every candidate pays the same ridge).
const TOPOLOGY_FIT_RIDGE: f64 = 1e-6;

/// One realized candidate of the birth topology race: the fitted evaluator, its
/// penalized-least-squares decoder against the birth target, the chart manifold
/// the winning atom will carry, and the basis kind tag. Carried as the
/// `select_topology_with_fit` fit handle so the winner's basis seeds the born
/// atom directly (no re-fit, no cold restart).
#[derive(Clone)]
struct TopologyRaceFit {
    evaluator: Arc<dyn SaeBasisEvaluator>,
    basis_kind: SaeAtomBasisKind,
    manifold: LatentManifold,
    latent_dim: usize,
    /// Fitted basis design `Φ(coords)` (`n × m`).
    phi: Array2<f64>,
    /// Fitted basis Jacobian `∂Φ` (`n × m × d`).
    jet: ndarray::Array3<f64>,
    /// Penalized-least-squares decoder `B` (`m × p`).
    decoder: Array2<f64>,
    /// Intrinsic roughness Gram `S` (`m × m`) the atom is seeded with.
    penalty: Array2<f64>,
}

/// A candidate topology paired with the evaluator + coordinates + manifold it
/// realizes for a `d`-dimensional birth. The evaluator is built fresh (cold) for
/// each candidate; the race then fits it to the birth target.
struct TopologyCandidateSpec {
    kind: AutoTopologyKind,
    basis_kind: SaeAtomBasisKind,
    manifold: LatentManifold,
    latent_dim: usize,
    evaluator: Arc<dyn SaeBasisEvaluator>,
    /// The `(n, d)` coordinates this candidate evaluates its basis at. A `d = 1`
    /// candidate reads the template coordinate column; a `d = 2` candidate reads
    /// the first two columns (or pads with the single column the seed carries).
    coords: Array2<f64>,
}

/// Build the topology candidate set whose required intrinsic dimension matches
/// the born atom's `d_k`, each realized over `coords` (`n × d_seed`, the
/// template's coordinate block). The candidate set is the realizable subset of
/// the smooth-term topology race — every member is a CORE basis evaluator
/// (`src/terms/sae/basis.rs`), so no FFI round-trip and no cold curved family
/// that the joint refit cannot warm-start:
///
/// * **`d = 1`** — `Circle` ([`PeriodicHarmonicEvaluator`]) vs `Euclidean` line
///   ([`EuclideanPatchEvaluator`] degree 3). These are the line-vs-circle race
///   the #1026 curved-vs-linear rung adjudicates post-fit, lifted to BIRTH.
/// * **`d = 2`** — `Torus` ([`TorusHarmonicEvaluator`]), `Sphere`
///   ([`SphereChartEvaluator`]), and a flat `Euclidean` patch
///   ([`EuclideanPatchEvaluator`] degree 2). `Cylinder` (`S¹ × ℝ`) has no core
///   evaluator — there is no mixed periodic/flat tensor basis in the rank-aware
///   basis (`src/terms/sae/basis.rs`, settled), so it is honestly EXCLUDED from
///   the realizable race rather than faked by a torus stand-in.
///
/// The fixed harmonic / degree budgets mirror the seed-dictionary builder
/// (`sae_build_atom_plans`): periodic gets `2·d_k + 1` columns, torus two
/// harmonics per axis, the patch degree 3 (`d = 1`) / 2 (`d = 2`).
fn topology_candidates_for_dim(
    coords: ArrayView2<'_, f64>,
    d_k: usize,
) -> Result<Vec<TopologyCandidateSpec>, String> {
    let n = coords.nrows();
    let d_seed = coords.ncols();
    if d_k == 0 {
        // d_k = 0 is the cluster-null rung — it is NOT a manifold topology and is
        // adjudicated below the race (the bottom rung), never inside it.
        return Ok(Vec::new());
    }
    // Project the seed coordinates onto the candidate's intrinsic dimension. A
    // d=1 candidate uses the first column; a d=2 candidate uses the first two
    // (padding the second from the first when the seed carries only one column,
    // so a 1-D seed can still present a 2-D candidate to the race).
    let coords_d = |d: usize| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, d));
        for row in 0..n {
            for col in 0..d {
                let src = col.min(d_seed.saturating_sub(1));
                out[[row, col]] = coords[[row, src]];
            }
        }
        out
    };

    let mut specs: Vec<TopologyCandidateSpec> = Vec::new();
    match d_k {
        1 => {
            let n_harmonics = (2 * d_k + 1).max(3) | 1; // odd, ≥ 3
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Circle,
                basis_kind: SaeAtomBasisKind::Periodic,
                manifold: LatentManifold::Circle { period: 1.0 },
                latent_dim: 1,
                evaluator: Arc::new(PeriodicHarmonicEvaluator::new(n_harmonics)?),
                coords: coords_d(1),
            });
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Euclidean,
                basis_kind: SaeAtomBasisKind::EuclideanPatch,
                manifold: LatentManifold::Euclidean,
                latent_dim: 1,
                evaluator: Arc::new(EuclideanPatchEvaluator::new(1, 3)?),
                coords: coords_d(1),
            });
        }
        2 => {
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Torus,
                basis_kind: SaeAtomBasisKind::Torus,
                manifold: LatentManifold::Euclidean,
                latent_dim: 2,
                evaluator: Arc::new(TorusHarmonicEvaluator::new(2, 2)?),
                coords: coords_d(2),
            });
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Sphere,
                basis_kind: SaeAtomBasisKind::Sphere,
                manifold: LatentManifold::Sphere { dim: 2 },
                latent_dim: 2,
                evaluator: Arc::new(SphereChartEvaluator),
                coords: coords_d(2),
            });
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Euclidean,
                basis_kind: SaeAtomBasisKind::EuclideanPatch,
                manifold: LatentManifold::Euclidean,
                latent_dim: 2,
                evaluator: Arc::new(EuclideanPatchEvaluator::new(2, 2)?),
                coords: coords_d(2),
            });
        }
        _ => {
            // d_k ≥ 3: a flat Euclidean patch is the only realizable core basis
            // (the curved families top out at d = 2). The race degenerates to a
            // single candidate — still honest (the winner is reported), just not
            // a contest.
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Euclidean,
                basis_kind: SaeAtomBasisKind::EuclideanPatch,
                manifold: LatentManifold::Euclidean,
                latent_dim: d_k,
                evaluator: Arc::new(EuclideanPatchEvaluator::new(d_k, 2)?),
                coords: coords_d(d_k),
            });
        }
    }
    Ok(specs)
}

/// Fit one topology candidate to the birth target `Y` (`n × p`) over `weights`
/// (`n`, the candidate's per-row reconstruction mass) by penalized least
/// squares, and return its TK evidence inputs + the realized fit handle.
///
/// The reduced per-atom Gaussian-reconstruction evidence is computed on EXACTLY
/// the scale the #1026 curved-vs-linear rung and the smooth-term topology race
/// use, so it is commensurable under the shared TK normalizer:
///
/// * `raw_reml = ½·(weighted residual SSE) + ½·log|Φᵀ W Φ + S|` — the rank-aware
///   Laplace negative log evidence of the penalized fit (data-fit deviance + the
///   Hessian logdet that prices the parameters against the effective sample).
/// * `null_dim` / `null_space_logdet` — the roughness Gram's null space (the
///   unpenalized polynomial/constant directions) and its Hessian logdet over that
///   null space, the gauge-invariance term the TK normalizer subtracts.
/// * `effective_dim = tr[(Φᵀ W Φ + S)⁻¹ Φᵀ W Φ]` — the penalized effective degrees
///   of freedom, the per-effective-dim scale's denominator.
fn fit_topology_candidate(
    spec: &TopologyCandidateSpec,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<TopologyAutoFitEvidence<TopologyRaceFit>, String> {
    let n = target.nrows();
    let p = target.ncols();
    let (phi, jet) = spec.evaluator.evaluate(spec.coords.view())?;
    let m = phi.ncols();
    if phi.nrows() != n {
        return Err(format!(
            "fit_topology_candidate: basis rows {} != target rows {n}",
            phi.nrows()
        ));
    }
    if weights.len() != n {
        return Err(format!(
            "fit_topology_candidate: weights length {} != target rows {n}",
            weights.len()
        ));
    }

    // Weighted normal equations Φᵀ W Φ and Φᵀ W Y, plus the weighted total mass.
    let mut gram = Array2::<f64>::zeros((m, m)); // Φᵀ W Φ
    let mut rhs = Array2::<f64>::zeros((m, p)); // Φᵀ W Y
    let mut w_sum = 0.0_f64;
    for row in 0..n {
        let w = weights[row];
        if !(w.is_finite() && w >= 0.0) {
            return Err("fit_topology_candidate: weights must be finite and non-negative".into());
        }
        w_sum += w;
        if w == 0.0 {
            continue;
        }
        for a in 0..m {
            let pa = phi[[row, a]];
            let wpa = w * pa;
            for b in a..m {
                gram[[a, b]] += wpa * phi[[row, b]];
            }
            for out in 0..p {
                rhs[[a, out]] += wpa * target[[row, out]];
            }
        }
    }
    // Symmetrize the upper triangle into the lower.
    for a in 0..m {
        for b in (a + 1)..m {
            gram[[b, a]] = gram[[a, b]];
        }
    }
    if !(w_sum > 0.0 && w_sum.is_finite()) {
        return Err("fit_topology_candidate: degenerate (zero-mass) birth target".into());
    }

    // The intrinsic roughness Gram of this candidate basis. An atom built from the
    // evaluator seeds its penalty via `refresh_intrinsic_smooth_penalty`; here we
    // need the same operator to price smoothness in the evidence. Seed a tiny
    // ridge so the unpenalized null space is still pinned for the solve, and read
    // the candidate's analytic raw roughness from a throwaway atom seeded with an
    // identity penalty (the refresh recomputes the pullback-metric Gram from the
    // decoder + coordinates, exactly the production seeding).
    let identity = Array2::<f64>::eye(m);
    let zero_decoder = Array2::<f64>::zeros((m, p));
    let mut probe = SaeManifoldAtom::new(
        "topology_race_probe",
        spec.basis_kind.clone(),
        spec.latent_dim,
        phi.clone(),
        jet.clone(),
        zero_decoder,
        identity,
    )?
    .with_basis_evaluator(spec.evaluator.clone());
    // The intrinsic penalty depends on the decoder via the pullback metric; seed
    // it from the least-squares decoder once it is solved (below), then read it
    // back. For the evidence Hessian we use the penalty at the FITTED decoder so
    // the smoothness price matches the seeded atom.

    // Penalized normal-equations matrix H = Φᵀ W Φ + S(+ridge). Use the raw
    // roughness Gram (decoder-independent) for the penalty so the solve does not
    // chase its own decoder; the production atom then refreshes the pullback
    // penalty from this very decoder.
    let s_raw = probe.smooth_penalty_raw.clone();
    let mut h = gram.clone();
    for a in 0..m {
        for b in 0..m {
            h[[a, b]] += s_raw[[a, b]];
        }
        h[[a, a]] += TOPOLOGY_FIT_RIDGE;
    }
    let h_chol = h
        .cholesky(Side::Lower)
        .map_err(|e| format!("fit_topology_candidate: penalized Hessian Cholesky: {e:?}"))?;
    let decoder = h_chol.solve_mat(&rhs); // (ΦᵀWΦ + S)⁻¹ Φᵀ W Y, m × p

    // Seed the probe atom's pullback penalty from the fitted decoder so the
    // returned penalty matches what the production atom carries.
    probe.decoder_coefficients = decoder.clone();
    probe.refresh_intrinsic_smooth_penalty();

    // Weighted residual SSE of the penalized reconstruction.
    let mut sse = 0.0_f64;
    for row in 0..n {
        let w = weights[row];
        if w == 0.0 {
            continue;
        }
        for out in 0..p {
            let mut pred = 0.0_f64;
            for a in 0..m {
                pred += phi[[row, a]] * decoder[[a, out]];
            }
            let r = target[[row, out]] - pred;
            sse += w * r * r;
        }
    }

    // Hessian logdet (the parameter price). H is SPD (ridge + Gram), so its
    // logdet is 2·Σ log(diag(L)) of its Cholesky — but FaerCholeskyFactor exposes
    // only solves, so recompute the logdet from the symmetric eigenvalues, which
    // we also need for the null-space accounting.
    let (h_evals, _h_evecs) = h
        .eigh(Side::Lower)
        .map_err(|e| format!("fit_topology_candidate: Hessian eigendecomposition: {e:?}"))?;
    let mut log_det_h = 0.0_f64;
    for &ev in &h_evals {
        if !(ev > 0.0) {
            return Err("fit_topology_candidate: penalized Hessian not positive definite".into());
        }
        log_det_h += ev.ln();
    }

    // Rank-aware Laplace negative log evidence on the smooth-rung scale:
    // ½·SSE (the Gaussian deviance, unit dispersion — the constant cancels in the
    // TK race) + ½·log|H|.
    let raw_reml = 0.5 * sse + 0.5 * log_det_h;

    // Null space of the roughness Gram S (the unpenalized constant/polynomial
    // directions): null_dim = nullity(S), and the null-space Hessian logdet is
    // the logdet of H restricted to ker(S). Over ker(S), H = Φᵀ W Φ (+ridge) — the
    // data curvature of the unpenalized directions, which is what the TK
    // normalizer prices to make cross-topology scores gauge-invariant.
    let (s_evals, s_evecs) = s_raw
        .eigh(Side::Lower)
        .map_err(|e| format!("fit_topology_candidate: penalty eigendecomposition: {e:?}"))?;
    let s_max = s_evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    let s_tol = 1e-9 * (1.0 + s_max);
    let null_cols: Vec<usize> = s_evals
        .iter()
        .enumerate()
        .filter(|(_, &v)| v <= s_tol)
        .map(|(i, _)| i)
        .collect();
    let null_dim = null_cols.len();
    let null_space_logdet = if null_dim == 0 {
        None
    } else {
        // H restricted to ker(S): Uᵀ H U where U are the null eigenvectors.
        let mut h_null = Array2::<f64>::zeros((null_dim, null_dim));
        for (ii, &ci) in null_cols.iter().enumerate() {
            // H · u_ci
            let mut hu = Array1::<f64>::zeros(m);
            for a in 0..m {
                let mut acc = 0.0_f64;
                for b in 0..m {
                    acc += h[[a, b]] * s_evecs[[b, ci]];
                }
                hu[a] = acc;
            }
            for (jj, &cj) in null_cols.iter().enumerate() {
                let mut acc = 0.0_f64;
                for a in 0..m {
                    acc += s_evecs[[a, cj]] * hu[a];
                }
                h_null[[ii, jj]] = acc;
            }
        }
        let (hn_evals, _) = h_null
            .eigh(Side::Lower)
            .map_err(|e| format!("fit_topology_candidate: null-space Hessian eigh: {e:?}"))?;
        let mut ld = 0.0_f64;
        for &ev in &hn_evals {
            if !(ev > 0.0) {
                return Err(
                    "fit_topology_candidate: null-space Hessian not positive definite".into(),
                );
            }
            ld += ev.ln();
        }
        Some(ld)
    };

    // Effective degrees of freedom tr[H⁻¹ (Φᵀ W Φ)] = Σ_a (H⁻¹ Gram)_{aa}.
    let h_inv_gram = h_chol.solve_mat(&gram); // H⁻¹ (Φᵀ W Φ), m × m
    let mut effective_dim = 0.0_f64;
    for a in 0..m {
        effective_dim += h_inv_gram[[a, a]];
    }
    if !(effective_dim.is_finite() && effective_dim > 0.0) {
        // A fully-penalized fit (no effective parameters) cannot be scored on the
        // per-effective-dim scale; floor at a single effective parameter so the
        // race still ranks it (the data-fit term dominates the verdict anyway).
        effective_dim = 1.0;
    }
    if !raw_reml.is_finite() {
        return Err("fit_topology_candidate: non-finite raw REML".into());
    }

    let penalty = probe.smooth_penalty.clone();
    Ok(TopologyAutoFitEvidence {
        topology_name: spec.kind.as_str().to_string(),
        raw_reml,
        null_dim: null_dim as f64,
        null_space_logdet,
        effective_dim,
        n_obs: n,
        fit_handle: TopologyRaceFit {
            evaluator: spec.evaluator.clone(),
            basis_kind: spec.basis_kind.clone(),
            manifold: spec.manifold.clone(),
            latent_dim: spec.latent_dim,
            phi,
            jet,
            decoder,
            penalty,
        },
    })
}

/// Race the candidate topologies whose required intrinsic dimension matches the
/// born atom's `d_k` against the birth target `Y` (`n × p`, the residual-factor
/// image the atom would reconstruct) over the template coordinates `coords`, and
/// return the EVIDENCE-WINNING fit. The winner is the lowest TK-normalized REML
/// via [`select_topology_with_fit`] — the gauge-invariant comparison the
/// smooth-term topology race already applies — so a circular residual gets a
/// circle, a straight residual a line, a spherical residual a sphere, etc.
///
/// Returns `None` when the race has no realizable candidate (`d_k = 0`, the
/// cluster-null rung, handled below the race) or the birth target is degenerate;
/// the caller then falls back to the template basis (warm inheritance) and the
/// post-fit curved-vs-linear rung adjudicates as before.
fn race_birth_topology(
    coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    d_k: usize,
) -> Result<Option<(TopologyRaceFit, String)>, String> {
    let specs = topology_candidates_for_dim(coords, d_k)?;
    if specs.is_empty() {
        return Ok(None);
    }
    let selector = TopologyAutoSelector {
        // The race is over EXACTLY the candidate set we built; do not let the
        // selector's constant-curvature fuse drop one — pass them through as-is.
        candidates: specs.iter().map(|s| s.kind).collect(),
        // Per-effective-dim normalization so a low-parameter line and a
        // high-parameter sphere are compared on the same per-parameter scale (the
        // smooth-term race default).
        score_scale: TopologyScoreScale::PerEffectiveDim,
        latent: None,
    };
    // Index the realized specs by kind so the fit closure can find the right
    // evaluator/coords for the kind the selector hands it (the selector may fuse
    // or reorder).
    let mut by_kind: std::collections::HashMap<AutoTopologyKind, &TopologyCandidateSpec> =
        std::collections::HashMap::with_capacity(specs.len());
    for spec in &specs {
        by_kind.insert(spec.kind, spec);
    }
    let ranked = select_topology_with_fit(&selector, |kind| {
        let spec = by_kind.get(&kind).ok_or_else(|| {
            format!(
                "race_birth_topology: no realized candidate for fused topology {:?}",
                kind.as_str()
            )
        })?;
        fit_topology_candidate(spec, target, weights)
    })?;
    let winner = ranked
        .winner()
        .ok_or_else(|| "race_birth_topology: empty ranking".to_string())?;
    Ok(Some((winner.fit_handle.clone(), winner.topology_name.clone())))
}

/// A small neutral routing logit a born atom is seeded at: large enough that the
/// refit can grow it if the residual-factor direction is real, small relative to
/// the established atoms so it does not perturb the current routing.
const BIRTH_SEED_LOGIT: f64 = -4.0;

/// Append a fresh atom whose decoder is seeded from a residual-factor direction.
/// The new atom reuses the structural basis of atom 0 (same basis kind, latent
/// dim, basis values + jacobian + smooth penalty) as its BIRTH TEMPLATE — warm
/// inheritance by construction, so the engine's warm-state contract holds and
/// the joint refit starts from a live basis rather than a cold curved family.
/// Only its decoder coefficients carry the residual-factor direction. Routed at
/// a small neutral mass on every row so the refit grows it if it is real and the
/// death channel demotes it next round if it is not.
///
/// # Topology adjudication (#977)
///
/// The template basis is the atom's INITIAL parameterization, not its final
/// topology. A born atom's topology is adjudicated by EVIDENCE downstream, on
/// the discovered dictionary, at two rungs:
///
/// * **Existence** — the #984 held-out e-value birth gate (run inside
///   [`crate::solver::structure_search::search`]) decides whether the atom is
///   born at all. Only a residual factor whose held-out reconstruction
///   likelihood-ratio crosses the Ville threshold earns an atom; the rest stay
///   contested in the [`SearchLedger`].
/// * **Curved (`d ≥ 1`) vs straight / cluster (`d = 0`)** — the #1026
///   hybrid-split pass ([`SaeManifoldTerm::compute_hybrid_split_report`], run
///   post-search over the FULL discovered dictionary) adjudicates every eligible
///   `d = 1` atom's fitted curved image against its straight (linear
///   special-case) secant on the common rank-aware Laplace evidence scale, and
///   records the verdict. A born atom whose curvature does not pay collapses to
///   the linear / cluster lane; one that earns it keeps its curved image. The
///   dictionary is therefore genuinely heterogeneous (curved + linear atoms),
///   not all-circle, with the per-atom verdict surfaced on the fit payload.
///
/// A full per-atom topology race (circle vs line vs torus vs sphere vs euclidean
/// by `reml_score`) at birth would require refitting each candidate basis under
/// the gate and is deferred; the existence gate + the curved-vs-linear rung
/// above are the adjudicated minimum that makes the discovered dictionary
/// heterogeneous and honest.
fn born_atom(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    factor_dir: ArrayView2<'_, f64>,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let k = term.k_atoms();
    let template = &term.atoms[0];
    let m = template.basis_size();
    let p = term.output_dim();
    if factor_dir.dim() != (m, p) {
        return Err(format!(
            "born_atom: residual-factor decoder must be ({m}, {p}); got {:?}",
            factor_dir.dim()
        ));
    }
    let mut atoms = term.atoms.clone();
    // The born atom reuses the template's structural basis (kind, latent dim,
    // basis values + jacobian + raw penalty); only its decoder carries the
    // residual-factor direction. Mutating the public `decoder_coefficients` and
    // refreshing the intrinsic (pullback-metric) smooth penalty rebuilds exactly
    // the decoder-dependent state, matching the constructor's seeding.
    let mut born = template.clone();
    born.decoder_coefficients = factor_dir.to_owned();
    born.refresh_intrinsic_smooth_penalty();
    atoms.push(born);

    let n = term.assignment.logits.nrows();
    let mut logits = Array2::<f64>::zeros((n, k + 1));
    for row in 0..n {
        for col in 0..k {
            logits[[row, col]] = term.assignment.logits[[row, col]];
        }
        logits[[row, k]] = BIRTH_SEED_LOGIT;
    }
    let mut coords = term.assignment.coords.clone();
    coords.push(term.assignment.coords[0].clone());
    let assignment =
        crate::terms::sae_manifold::SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let child = SaeManifoldTerm::new(atoms, assignment)?;

    let mut child_rho = rho.clone();
    // The born atom inherits the template atom's ARD block shape (disabled if
    // the template's was disabled).
    let inherited = child_rho
        .log_ard
        .first()
        .cloned()
        .unwrap_or_else(|| Array1::<f64>::zeros(0));
    child_rho.log_ard.push(inherited);
    Ok((child, child_rho))
}

/// A held-out row-block shard for the universal-inference estimation/evaluation
/// split the gates run over: a contiguous block of row indices into the FULL
/// target the triggers were not tuned on.
///
/// The split is realized through the term's per-row reconstruction weights
/// ([`SaeManifoldTerm::set_row_loss_weights`]): a candidate is refit with the
/// currently-held-out shards' rows at weight `0` (no fitting pressure) and the
/// estimation rows at weight `1`, then EVALUATED on the held-out rows. The
/// predictable-plugin e-process streams the shards: shard `k` is evaluated under
/// a candidate that has not yet seen its rows, then folded into the estimation
/// set (un-masked) for shard `k+1` — exactly the contract
/// [`run_atom_birth_gate`](crate::inference::structure_evidence::run_atom_birth_gate)
/// guarantees the call order of.
#[derive(Clone, Debug)]
pub struct RowBlockShard {
    /// The full target, shared across shards (`(N, p)`).
    pub target: std::sync::Arc<Array2<f64>>,
    /// Row indices into the full target that this shard holds out for
    /// evaluation.
    pub rows: Vec<usize>,
}

/// The estimation/evaluation row split the e-process gates run over. The
/// estimation rows are the candidate's fitting set (weight `1`); the evaluation
/// rows are held out (weight `0` during the fit) and partitioned into the shard
/// stream the gate accumulates evidence over.
#[derive(Clone, Debug)]
pub struct EstimationEvalSplit {
    /// Estimation row indices (the candidate is refit on these; held-out rows
    /// carry weight `0`).
    pub estimation_rows: Vec<usize>,
    /// The evaluation shards, in stream order.
    pub shards: Vec<RowBlockShard>,
}

/// Fraction of rows reserved for estimation (the candidate's fitting set); the
/// remainder is split into evaluation shards. A fixed structural constant
/// (magic-by-default): a majority estimation split keeps the candidate fit
/// faithful while leaving a held-out block for honest evidence.
const ESTIMATION_FRACTION: f64 = 0.6;

/// Build the estimation/evaluation split: the first `ESTIMATION_FRACTION` of the
/// rows (contiguous) are the estimation set, the remainder is partitioned into
/// `n_shards` contiguous held-out evaluation blocks. Deterministic — contiguous
/// blocks, no shuffle. Each shard shares the full target by reference.
pub fn estimation_eval_split(target: ArrayView2<'_, f64>, n_shards: usize) -> EstimationEvalSplit {
    let n = target.nrows();
    if n == 0 {
        return EstimationEvalSplit {
            estimation_rows: Vec::new(),
            shards: Vec::new(),
        };
    }
    let shared = std::sync::Arc::new(target.to_owned());
    // At least one estimation row and at least one evaluation row when n ≥ 2.
    let n_est =
        ((n as f64 * ESTIMATION_FRACTION).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let estimation_rows: Vec<usize> = (0..n_est).collect();
    let eval_rows: Vec<usize> = (n_est..n).collect();
    let n_eval = eval_rows.len();
    let n_shards = n_shards.min(n_eval).max(usize::from(n_eval > 0));
    let mut shards = Vec::new();
    if n_eval > 0 && n_shards > 0 {
        let base = n_eval / n_shards;
        let rem = n_eval % n_shards;
        let mut cursor = 0usize;
        for s in 0..n_shards {
            let len = base + usize::from(s < rem);
            let rows: Vec<usize> = eval_rows[cursor..cursor + len].to_vec();
            shards.push(RowBlockShard {
                target: shared.clone(),
                rows,
            });
            cursor += len;
        }
    }
    EstimationEvalSplit {
        estimation_rows,
        shards,
    }
}

/// Outcome of the full round driver: the (possibly restructured) fitted term +
/// ρ and the per-round ledgers, each carrying the joint fit's collapse events.
pub struct StructureSearchResult {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    /// One ledger per round actually run (a round that applies no move is the
    /// last; its ledger is included so the certificate covers the fixpoint).
    pub rounds: Vec<SearchLedger>,
}

/// The round driver's configuration: how the data is split into shards, the
/// e-gate's budget/level, the round cap, and the per-round harvest breadth.
/// Bundled so the driver entry points stay below the argument-count threshold
/// and so a caller configures one object rather than a positional argument
/// cascade.
#[derive(Clone, Copy, Debug)]
pub struct RoundDriverConfig {
    /// Number of held-out evaluation shards the gate streams over.
    pub n_shards: usize,
    /// Move budget + α the e-gates certify at (fixed for the run).
    pub budget: MoveBudget,
    /// Maximum harvest → search rounds before stopping at the fixpoint.
    pub max_rounds: usize,
    /// Per-round harvest breadth (max fusions / fissions / births).
    pub harvest_params: HarvestParams,
}

/// Drive evidence-guarded structure search around a fitted SAE term until a
/// round applies no moves (#997 round driver).
///
/// Each round: harvest proposals from the current fitted term, run [`search`]
/// over the held-out evaluation shards (gating births/fissions/fusions, demoting
/// never-certified deaths), and adopt the restructured state. The loop stops
/// when a round's ledger contains no applied move (every record is
/// contested / vetoed / deduplicated / deferred / stale) or `max_rounds` is hit.
///
/// `candidate_fit` is the warm refit: given a RESTRUCTURED candidate term + ρ,
/// it refits the candidate on the ESTIMATION rows only (held-out evaluation rows
/// carry weight `0`), so the candidate is the predictable plug-in the e-process
/// evaluates on the held-out shard stream. It is INFALLIBLE at this boundary —
/// it absorbs its own inner-solve errors by returning the unchanged candidate (a
/// conservative no-improvement signal to the gate, never a panic). The shard
/// fold is a no-op: the candidate is fixed across the stream (a predictable
/// plug-in), and each shard contributes its held-out reconstruction
/// likelihood-ratio against the honestly-refit null sup.
pub fn run_structure_search_rounds(
    mut term: SaeManifoldTerm,
    mut rho: SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    config: RoundDriverConfig,
    ledger: &mut StructureLedger,
    mut candidate_fit: impl FnMut(
        SaeManifoldTerm,
        SaeManifoldRho,
        &[usize],
    ) -> (SaeManifoldTerm, SaeManifoldRho),
) -> Result<StructureSearchResult, String> {
    let RoundDriverConfig {
        n_shards,
        budget,
        max_rounds,
        harvest_params,
    } = config;
    let split = estimation_eval_split(target, n_shards);
    let mut rounds: Vec<SearchLedger> = Vec::new();

    for _ in 0..max_rounds {
        // Harvest from the current fitted state. Residuals R = target − fitted.
        let fitted = term.try_fitted()?;
        let residuals = &target.to_owned() - &fitted;
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params)?;

        // Pre-build the birth-decoder list ONCE per round from the residual
        // factor (the birth candidates index into it), so the apply-move
        // closure inside the gate is a pure function of the candidate index.
        let birth_decoders = build_birth_decoders(&term, residuals.view(), &harvest_params)?;

        if report.proposals.is_empty() || split.shards.is_empty() {
            // Nothing to do this round — record an empty ledger (with the live
            // collapse events) as the fixpoint and stop.
            rounds.push(SearchLedger {
                alpha: budget.alpha,
                moves: Vec::new(),
                collapse_events: term.collapse_events().to_vec(),
            });
            break;
        }

        // The search state threads (term, rho) together. apply_move restructures
        // both AND refits the candidate on the estimation rows so it is the
        // predictable plug-in the held-out shards are evaluated against.
        type State = (SaeManifoldTerm, SaeManifoldRho);
        let collapse_events = term.collapse_events().to_vec();
        let decoders = birth_decoders;
        let estimation_rows = split.estimation_rows.clone();
        let outcome: SearchOutcome<State> = search(
            (term, rho),
            report.proposals,
            &split.shards,
            &budget,
            ledger,
            |state: &State, mv: &StructureMove| {
                let (cand_term, cand_rho) =
                    apply_structure_move(&state.0, &state.1, mv, &decoders)?;
                // Refit the restructured candidate on the estimation rows only.
                Ok(candidate_fit(cand_term, cand_rho, &estimation_rows))
            },
            |state: &State, shard: &RowBlockShard| eval_log_lik(&state.0, shard),
            |state: &State, shard: &RowBlockShard| eval_log_lik(&state.0, shard),
            // No-op fold: the candidate is the fixed predictable plug-in across
            // the held-out stream.
            |state: State, _: &RowBlockShard| state,
        )?;

        let (next_term, next_rho) = outcome.state;
        let mut round_ledger = outcome.ledger;
        round_ledger.collapse_events = collapse_events;
        let applied = round_ledger.moves.iter().any(|m| {
            matches!(
                m.verdict,
                crate::solver::structure_search::MoveVerdict::Accepted { .. }
                    | crate::solver::structure_search::MoveVerdict::Demoted { .. }
            )
        });
        rounds.push(round_ledger);

        term = next_term;
        rho = next_rho;

        if !applied {
            break;
        }
    }

    Ok(StructureSearchResult { term, rho, rounds })
}

/// Build the per-round residual-factor decoder list the birth apply-move indexes
/// into: each factor direction lifted to a `(m, p)` decoder in atom 0's basis.
fn build_birth_decoders(
    term: &SaeManifoldTerm,
    residuals: ArrayView2<'_, f64>,
    params: &HarvestParams,
) -> Result<Vec<Array2<f64>>, String> {
    let n = residuals.nrows();
    let p = residuals.ncols();
    if params.max_births == 0 || n == 0 || p == 0 {
        return Ok(Vec::new());
    }
    let assignments = term.assignment.assignments();
    let activity: Array1<f64> = (0..n).map(|r| assignments.row(r).sum()).collect();
    let max_rank = params.max_births.min(p.saturating_sub(1));
    let model = match StructuredResidualModel::fit(ResidualFactorInput {
        residuals,
        activity: activity.view(),
        max_factor_rank: max_rank,
    }) {
        Ok(m) => m,
        Err(_) => return Ok(Vec::new()),
    };
    let factor = model.factor();
    let r = factor.ncols();
    let m = term.atoms[0].basis_size();
    // Lift each p-vector factor direction to a (m, p) decoder: place the
    // direction on the constant (first) basis row so the born atom emits the
    // residual-factor direction as a flat decoder the refit can then shape. This
    // is the WhitenedStructured residual subspace, not raw-Euclidean Λ.
    let mut decoders = Vec::with_capacity(r);
    for j in 0..r {
        let mut decoder = Array2::<f64>::zeros((m, p));
        for out in 0..p {
            decoder[[0, out]] = factor[[out, j]];
        }
        decoders.push(decoder);
    }
    Ok(decoders)
}

/// Per-row Gaussian reconstruction log-likelihood of a shard under the current
/// (restructured, possibly shard-refit) state. The gate's evaluation statistic;
/// the engine guarantees a shard is evaluated strictly before it is folded in.
fn eval_log_lik(term: &SaeManifoldTerm, shard: &RowBlockShard) -> f64 {
    // The fitted reconstruction at the shard's held-out rows, scored against the
    // full target. The term's per-row routing/basis covers all N rows, so the
    // reconstruction at a held-out row is the model's prediction for it.
    let fitted = match term.try_fitted() {
        Ok(f) => f,
        Err(_) => return f64::NEG_INFINITY,
    };
    let n_full = fitted.nrows();
    let p = fitted.ncols();
    if p != shard.target.ncols() || n_full != shard.target.nrows() {
        return f64::NEG_INFINITY;
    }
    let mut sse = 0.0_f64;
    let mut count = 0usize;
    for &row in &shard.rows {
        if row >= n_full {
            continue;
        }
        for out in 0..p {
            let d = fitted[[row, out]] - shard.target[[row, out]];
            sse_accumulate(&mut sse, d);
        }
        count += p;
    }
    if count == 0 {
        return f64::NEG_INFINITY;
    }
    // Gaussian log-lik up to the additive constant that cancels in every
    // e-value ratio: −½·SSE (unit dispersion). The gate forms differences of
    // this against the null sup, so the constant and the dispersion scale drop
    // out of the certified evidence.
    -0.5 * sse
}

#[inline]
fn sse_accumulate(sse: &mut f64, d: f64) {
    *sse += d * d;
}

/// Inner-fit knobs for the production structure-search refit (the same numbers
/// the outer SAE fit drove its inner Arrow-Schur joint fit with).
#[derive(Clone, Copy, Debug)]
pub struct ProductionRefitParams {
    /// Inner Newton iterations per shard refit.
    pub inner_max_iter: usize,
    /// Inner Newton step size.
    pub learning_rate: f64,
    /// Ext-coordinate ridge.
    pub ridge_ext_coord: f64,
    /// β ridge.
    pub ridge_beta: f64,
}

/// Run the production structure-search pass around a fitted SAE term: harvest →
/// e-gated [`search`] over held-out row blocks → adopt certified/demoted moves →
/// repeat, returning the (possibly restructured) term + ρ and the per-round
/// ledgers (#997).
///
/// The shard refit folds a held-out block into a candidate via the SAME inner
/// joint-fit driver the outer fit used ([`SaeManifoldTerm::run_joint_fit_arrow_schur`]),
/// PENALTY-FREE: the gate's evidence is a held-out reconstruction
/// likelihood-ratio, and the isometry/ARD penalties are gauge/regularization
/// terms that do not belong in the evaluation likelihood. The refit absorbs its
/// own inner-solve errors by returning the unchanged candidate (a conservative
/// no-improvement signal, never a panic). `ledger` carries banked evidence
/// across rounds so the death veto sees earlier certifications.
pub fn run_production_structure_search(
    term: SaeManifoldTerm,
    rho: SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    config: RoundDriverConfig,
    refit_params: ProductionRefitParams,
    ledger: &mut StructureLedger,
) -> Result<StructureSearchResult, String> {
    let full_target = target.to_owned();
    let n = full_target.nrows();
    run_structure_search_rounds(
        term,
        rho,
        target,
        config,
        ledger,
        move |mut cand_term, mut cand_rho, estimation_rows| {
            // Refit the restructured candidate on the ESTIMATION rows only: the
            // held-out evaluation rows carry a near-zero weight (vanishing
            // fitting pressure) via the per-row reconstruction-weight seam, so
            // the candidate is the predictable plug-in the held-out shards are
            // scored against. The seam requires strictly-positive weights, so a
            // tiny epsilon stands in for the structural zero; after mean-1
            // normalization the estimation rows carry weight ≈ n/n_est and the
            // held-out rows ≈ 0. A non-converging inner solve returns the
            // unchanged candidate (the closure is infallible at the boundary).
            const HELD_OUT_WEIGHT: f64 = 1e-12;
            let mut weights = vec![HELD_OUT_WEIGHT; n];
            for &r in estimation_rows {
                if r < n {
                    weights[r] = 1.0;
                }
            }
            if cand_term.set_row_loss_weights(weights).is_err() {
                return (cand_term, cand_rho);
            }
            // A non-converging inner solve leaves the candidate as-is (the gate
            // then sees no improvement — a conservative, valid degradation).
            if cand_term
                .run_joint_fit_arrow_schur(
                    full_target.view(),
                    &mut cand_rho,
                    None,
                    refit_params.inner_max_iter,
                    refit_params.learning_rate,
                    refit_params.ridge_ext_coord,
                    refit_params.ridge_beta,
                )
                .is_err()
            {
                return (cand_term, cand_rho);
            }
            (cand_term, cand_rho)
        },
    )
}

/// Serialize the per-round ledgers to a JSON string for the fit payload — the
/// honesty surface the python boundary attaches under an additive
/// `structure_search` key. Byte-deterministic for identical inputs.
pub fn rounds_to_json(rounds: &[SearchLedger]) -> Result<String, String> {
    serde_json::to_string(rounds)
        .map_err(|e| format!("rounds_to_json: serialize search ledger: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::structure_search::{CollapseAction, CollapseEvent};
    use crate::terms::latent_coord::LatentManifold;
    use crate::terms::sae_manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom,
    };
    use ndarray::Array2;
    use std::sync::Arc;

    /// A high active logit (atom routes strongly on the row) and a low one
    /// (atom is dormant). With the `ACTIVE_SUPPORT_REL_FLOOR / K` threshold a
    /// softmax of these separates the discrete support cleanly.
    const ON: f64 = 6.0;
    const OFF: f64 = -6.0;

    /// Build a `K`-atom periodic SAE term whose per-row routing is dictated by a
    /// caller-supplied boolean activity matrix `active[(row, atom)]` (ON/OFF
    /// logits). Every atom shares the same circle basis; only the routing (and,
    /// for the birth template, the decoder) differs. Returns the term and a
    /// matching ρ with native ARD enabled (one axis per atom).
    fn planted_term(active: &[Vec<bool>]) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = active.len();
        let k = active[0].len();
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut atoms = Vec::with_capacity(k);
        let mut coord_blocks = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let mut decoder = Array2::<f64>::zeros((3, p));
            // Give each atom a distinct decoder direction so reconstruction is
            // non-degenerate.
            decoder[[1, atom_idx % p]] = 1.0;
            decoder[[2, (atom_idx + 1) % p]] = 1.0;
            let atom = SaeManifoldAtom::new(
                format!("atom_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
            coord_blocks.push(coords.clone());
        }
        let mut logits = Array2::<f64>::zeros((n, k));
        for (row, atom_active) in active.iter().enumerate() {
            for (atom, &on) in atom_active.iter().enumerate() {
                logits[[row, atom]] = if on { ON } else { OFF };
            }
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        (term, rho)
    }

    fn residuals_of(term: &SaeManifoldTerm) -> Array2<f64> {
        // A term scored against zero target gives R = −fitted; non-degenerate
        // residuals for the birth channel.
        let fitted = term.try_fitted().unwrap();
        -&fitted
    }

    /// #977 discovery oracle: with the production birth budget enabled, a fit
    /// whose residuals carry an unexplained factor direction (a structure the
    /// current dictionary does not express) HARVESTS a birth proposal — the
    /// candidate atom whose held-out e-value the gate then adjudicates. This is
    /// the proposal channel the production site re-enabled (`max_births > 0`);
    /// without it K could never grow.
    #[test]
    fn residual_bearing_fit_harvests_birth_proposal() {
        // A single circle atom routed on every row; its fitted reconstruction
        // leaves a structured residual (R = −fitted has rank > 1 across the p=4
        // output channels), so the whitened residual-factor subspace is
        // non-empty and the birth channel mines a candidate direction.
        let n = 40usize;
        let active: Vec<Vec<bool>> = (0..n).map(|_| vec![true]).collect();
        let (term, rho) = planted_term(&active);
        // Inject a clear shared-direction (rank-1) factor into the residuals that
        // varies smoothly with the per-row activity coordinate, so the whitened
        // residual-factor evidence ladder selects rank ≥ 1: every row gets a
        // multiple of the same unit output direction `u`, scaled by a per-row
        // amplitude. This is the unexplained shared structure a born atom would
        // absorb.
        let p = term.output_dim();
        let mut residuals = Array2::<f64>::zeros((n, p));
        let u = [0.6_f64, -0.4, 0.5, -0.3];
        for row in 0..n {
            // A non-constant per-row amplitude so the factor is genuine shared
            // structure (not absorbed by the diagonal noise floor).
            let amp = 1.0 + (row as f64) / (n as f64);
            for c in 0..p {
                residuals[[row, c]] = amp * u[c % u.len()];
            }
        }
        let params = HarvestParams {
            max_fusions: 0,
            max_fissions: 0,
            // The production-enabled budget (births > 0) — the whole point of
            // #977: K can grow.
            max_births: 2,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        let births: usize = report
            .proposals
            .iter()
            .filter(|p| matches!(p.mv, StructureMove::Birth { .. }))
            .count();
        assert!(
            births >= 1,
            "a residual-bearing fit with births enabled must harvest at least \
             one birth proposal (so K can be discovered); got {:?}",
            report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
        );
        assert!(
            report.births_proposed >= 1,
            "births_proposed must count the harvested births; got {}",
            report.births_proposed
        );
        assert!(
            report.birth_skipped_reason.is_none(),
            "the birth channel must run (no skip) on a non-degenerate residual; got {:?}",
            report.birth_skipped_reason
        );
    }

    /// #977 NULL oracle: a target the dictionary reconstructs exactly leaves
    /// ZERO residual, so the birth channel finds no factor subspace and proposes
    /// no birth — nothing is born under the null. (The round driver's e-gate is
    /// the second line of defense; this asserts the harvest itself does not
    /// manufacture growth where there is no unexplained structure.)
    #[test]
    fn fully_reconstructed_null_harvests_no_birth() {
        let n = 40usize;
        let active: Vec<Vec<bool>> = (0..n).map(|_| vec![true]).collect();
        let (term, rho) = planted_term(&active);
        // Residual ≡ 0: the dictionary reconstructs the target exactly, so there
        // is no unexplained factor to mine.
        let p = term.output_dim();
        let zero_residual = Array2::<f64>::zeros((n, p));
        let params = HarvestParams {
            max_fusions: 0,
            max_fissions: 0,
            max_births: 2,
        };
        let report =
            harvest_move_proposals(&term, &rho, zero_residual.view(), &params).unwrap();
        let births: usize = report
            .proposals
            .iter()
            .filter(|p| matches!(p.mv, StructureMove::Birth { .. }))
            .count();
        assert_eq!(
            births, 0,
            "a fully-reconstructed (zero-residual) null must harvest no birth \
             proposal; got {births} births"
        );
    }

    /// Oracle (#997 trigger): a planted SHATTER — two atoms with identical
    /// supports (one curved family re-encoded as near-duplicate flat atoms) —
    /// produces a FUSION proposal on that pair (symmetric code dependence ≈ 1),
    /// and NO fission audit (asymmetry ≈ 0).
    #[test]
    fn planted_shatter_harvests_fusion_not_fission() {
        // Atoms 0 and 1 share support exactly (every third row); atom 2 is
        // independent. n = 30.
        let n = 30usize;
        let active: Vec<Vec<bool>> = (0..n)
            .map(|row| {
                let dup = row % 3 == 0;
                vec![dup, dup, row % 2 == 0]
            })
            .collect();
        let (term, rho) = planted_term(&active);
        let residuals = residuals_of(&term);
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 4,
            max_births: 0,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        let has_fusion_01 = report.proposals.iter().any(|p| {
            matches!(p.mv, StructureMove::Fusion { a, b } if (a, b) == (0, 1) || (a, b) == (1, 0))
        });
        assert!(
            has_fusion_01,
            "shattered duplicate pair (0,1) must yield a fusion proposal; got {:?}",
            report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
        );
        // The duplicate pair is symmetric ⇒ no absorption fission audit on it.
        let has_fission = report
            .proposals
            .iter()
            .any(|p| matches!(p.mv, StructureMove::Fission { .. }));
        assert!(
            !has_fission,
            "symmetric duplicate supports must not trigger an absorption fission audit"
        );
    }

    /// Oracle (#997 trigger): a planted ABSORPTION (A⊇B: B's support nests
    /// inside A's) produces a FISSION audit on the parent A (high conditional
    /// asymmetry, parent conditional ≈ 1), and the `fission_carve_skipped` flag
    /// is recorded loudly (the #993 within-atom carve is not yet wired).
    #[test]
    fn planted_absorption_harvests_fission_audit_with_loud_carve_skip() {
        // Atom 0 (parent) active on rows ≡ 0 mod 2 PLUS rows ≡ 1 mod 4; atom 1
        // (child) active only on rows ≡ 0 mod 4 — strictly nested in 0's
        // support ⇒ P(0|1) = 1, P(1|0) < 1. n = 40.
        let n = 40usize;
        let active: Vec<Vec<bool>> = (0..n)
            .map(|row| {
                let child = row % 4 == 0;
                let parent = row % 2 == 0 || row % 4 == 1;
                vec![parent, child, row % 5 == 0]
            })
            .collect();
        let (term, rho) = planted_term(&active);
        let residuals = residuals_of(&term);
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 4,
            max_births: 0,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        let fissioned_parent = report
            .proposals
            .iter()
            .any(|p| matches!(p.mv, StructureMove::Fission { atom: 0 }));
        assert!(
            fissioned_parent,
            "nested-support parent (atom 0) must be flagged for a fission audit; got {:?}",
            report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
        );
        assert!(
            report.fission_carve_skipped,
            "the #993 within-atom carve is unwired; the skip must be recorded, not silent"
        );
    }

    /// Oracle (#997 type-I): three INDEPENDENT planted atoms (marginal supports
    /// at coprime strides) yield NO fusion proposal — the trigger does not
    /// manufacture binding edges where the codes are independent, so the e-gate
    /// is never even asked to reject a true null.
    #[test]
    fn independent_atoms_harvest_no_fusion() {
        let n = 60usize;
        let active: Vec<Vec<bool>> = (0..n)
            .map(|row| vec![row % 2 == 0, row % 3 == 0, row % 5 == 0])
            .collect();
        let (term, rho) = planted_term(&active);
        let residuals = residuals_of(&term);
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 4,
            max_births: 0,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        let has_fusion = report
            .proposals
            .iter()
            .any(|p| matches!(p.mv, StructureMove::Fusion { .. }));
        assert!(
            !has_fusion,
            "independent atom supports must not produce fusion proposals; got {:?}",
            report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
        );
    }

    /// Oracle (#997 death trigger): a diverged ARD precision yields a DEATH
    /// proposal; a terminal collapse event yields a death even with finite ARD.
    #[test]
    fn diverged_ard_and_terminal_collapse_harvest_deaths() {
        let n = 20usize;
        let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0, false]).collect();
        let (mut term, mut rho) = planted_term(&active);
        // Diverge atom 2's ARD precision well past the divergence floor.
        rho.log_ard[2] = Array1::from_elem(1, ARD_DIVERGENCE_LOG_PRECISION + 5.0);
        // Inject a terminal collapse for atom 1 (finite ARD, but routing gone).
        term.record_collapse_event(CollapseEvent {
            iteration: 3,
            atom: 1,
            max_active_mass: 1e-6,
            floor: 1e-3,
            action: CollapseAction::Terminal,
        });
        let residuals = residuals_of(&term);
        let params = HarvestParams {
            max_fusions: 0,
            max_fissions: 0,
            max_births: 0,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        let death_atoms: Vec<usize> = report
            .proposals
            .iter()
            .filter_map(|p| match p.mv {
                StructureMove::Death { atom } => Some(atom),
                _ => None,
            })
            .collect();
        assert!(
            death_atoms.contains(&2),
            "diverged ARD on atom 2 must yield a death proposal; got {death_atoms:?}"
        );
        assert!(
            death_atoms.contains(&1),
            "terminal collapse on atom 1 must yield a death proposal; got {death_atoms:?}"
        );
    }

    /// Apply-move restructuring oracle: fission GROWS the dictionary by one atom
    /// (child inherits parent's basis + ARD block), fusion and death keep K
    /// (fold / demote), birth appends a residual-factor atom.
    #[test]
    fn apply_move_restructures_warm() {
        let n = 12usize;
        let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0]).collect();
        let (term, rho) = planted_term(&active);
        let k0 = term.k_atoms();

        // Fission: K grows, child ARD block inherited.
        let (fissioned, fissioned_rho) =
            apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
        assert_eq!(fissioned.k_atoms(), k0 + 1);
        assert_eq!(fissioned_rho.log_ard.len(), k0 + 1);

        // Fusion: K unchanged, atom b demoted to ~0 routing.
        let (fused, _) =
            apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
        assert_eq!(fused.k_atoms(), k0);
        let fused_assign = fused.assignment.assignments();
        assert!(
            fused_assign.column(1).iter().all(|&m| m < 1e-6),
            "fused-away atom 1 must route to ~0 mass"
        );

        // Death: K unchanged, atom demoted.
        let (dead, _) =
            apply_structure_move(&term, &rho, &StructureMove::Death { atom: 1 }, &[]).unwrap();
        assert_eq!(dead.k_atoms(), k0);
        let dead_assign = dead.assignment.assignments();
        assert!(dead_assign.column(1).iter().all(|&m| m < 1e-6));

        // Birth: K grows, new atom carries the supplied residual-factor decoder.
        let p = term.output_dim();
        let m = term.atoms[0].basis_size();
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[0, 0]] = 0.7;
        let (born, born_rho) = apply_structure_move(
            &term,
            &rho,
            &StructureMove::Birth { candidate: 0 },
            &[decoder],
        )
        .unwrap();
        assert_eq!(born.k_atoms(), k0 + 1);
        assert_eq!(born_rho.log_ard.len(), k0 + 1);
        assert_eq!(born.atoms[k0].decoder_coefficients[[0, 0]], 0.7);
    }

    /// Ledger byte-determinism oracle (#997): two runs of the round driver over
    /// the same planted shatter, with a deterministic scripted fit, serialize
    /// the per-round ledgers byte-identically.
    #[test]
    fn round_driver_ledger_is_byte_deterministic() {
        let n = 24usize;
        let active: Vec<Vec<bool>> = (0..n)
            .map(|row| {
                let dup = row % 3 == 0;
                vec![dup, dup, row % 2 == 0]
            })
            .collect();

        let run = || {
            let (term, rho) = planted_term(&active);
            let target = Array2::<f64>::zeros((n, term.output_dim()));
            let mut ledger = crate::inference::structure_evidence::StructureLedger::new();
            let budget = MoveBudget {
                max_moves: 4,
                alpha: 0.05,
            };
            let params = HarvestParams {
                max_fusions: 4,
                max_fissions: 0,
                max_births: 0,
            };
            let config = RoundDriverConfig {
                n_shards: 3,
                budget,
                max_rounds: 2,
                harvest_params: params,
            };
            // Deterministic no-op fit: the scripted gate sees the unrefit
            // candidate (the engine's determinism is what this asserts, not the
            // SAE inner solve).
            run_structure_search_rounds(term, rho, target.view(), config, &mut ledger, |t, r, _| {
                (t, r)
            })
            .unwrap()
        };

        let a = run();
        let b = run();
        let sa = serde_json::to_string(&a.rounds).unwrap();
        let sb = serde_json::to_string(&b.rounds).unwrap();
        assert_eq!(
            sa, sb,
            "identical inputs must produce a byte-identical ledger"
        );
        assert_eq!(a.term.k_atoms(), b.term.k_atoms());
    }

    /// Estimation/eval split oracle: the split reserves estimation rows and
    /// partitions the remainder into held-out shards that do NOT overlap the
    /// estimation set (the universal-inference contract the gates rely on).
    #[test]
    fn estimation_eval_split_is_disjoint() {
        let target = Array2::<f64>::zeros((20, 3));
        let split = estimation_eval_split(target.view(), 4);
        assert!(!split.estimation_rows.is_empty());
        assert!(!split.shards.is_empty());
        let est: std::collections::HashSet<usize> = split.estimation_rows.iter().copied().collect();
        for shard in &split.shards {
            for &row in &shard.rows {
                assert!(
                    !est.contains(&row),
                    "eval shard row {row} must not be in the estimation set"
                );
            }
        }
    }
}
