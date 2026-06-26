//! #997 — the wiring seam between a fitted [`SaeManifoldTerm`] and the
//! evidence-guarded move engine of [`crate::structure_search`].
//!
//! #976 closed with the move engine (`search`) and its triggers
//! ([`crate::terms::sae::atom_codes::SparseAtomCodes::coactivation`], ARD precisions,
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
//!    birth appends a residual-factor atom whose TOPOLOGY is chosen by EVIDENCE
//!    (#977): [`race_birth_topology`] races the candidate bases matched to the
//!    atom's intrinsic dim (`d = 1`: circle vs line; `d = 2`: torus vs
//!    sphere/constant-curvature vs euclidean vs cylinder) by TK-normalized REML and seeds the
//!    born atom from the winner, so the discovered dictionary is genuinely
//!    heterogeneous rather than all-circle. Every child state is built FROM the
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
//! proposal stream are computed with the same [`gam_runtime::warm_start::Fingerprinter`]
//! the [`gam_terms::smooth::TermCollectionSpec`] machinery (#869) uses, fed
//! the POST-move dictionary shape (atom count, per-atom basis kind + latent dim
//! + the move that produced it), so two proposals that reach the same dictionary
//! shape collide exactly as the engine requires.

use std::sync::Arc;

use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use crate::inference::structure_evidence::{ClaimKind, StructureLedger};
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::structure_search::{
    CollapseAction, MoveBudget, MoveProposal, SearchLedger, SearchOutcome, StructureMove, search,
};
use crate::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoSelector, TopologyScoreScale,
    select_topology_with_fit,
};
use gam_terms::latent::{LatentIdMode, LatentManifold};
use crate::terms::sae::atom_codes::SparseAtomCodes;
use crate::terms::sae::basis::{
    CylinderHarmonicEvaluator, EuclideanPatchEvaluator, PeriodicHarmonicEvaluator,
    SaeBasisSecondJet, SphereChartEvaluator, TorusHarmonicEvaluator,
};
use crate::terms::sae::manifold::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::structure::anova_atom::{
    CarveReport, FissionDecision, carve, carve_input_from_fitted_atom, fission_decision,
};
use gam_runtime::warm_start::Fingerprinter;

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

/// Level at which the within-atom representational carve (#993) calls binding
/// PROVEN (blocking a fission). The harvest carve is a PROPOSAL filter, not the
/// final certificate — the downstream held-out e-gate owns acceptance — so this
/// is the conventional 0.05 screening level, deliberately not the stricter
/// certificate level; a carve that fails to reject here still rides as a
/// fission proposal for the e-gate to adjudicate on held-out shards.
const WITHIN_ATOM_CARVE_ALPHA: f64 = 0.05;

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
/// [`Fingerprinter`] the [`TermCollectionSpec`](gam_terms::smooth::TermCollectionSpec)
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
        SaeAtomBasisKind::Linear => "linear",
        SaeAtomBasisKind::EuclideanPatch => "euclidean_patch",
        SaeAtomBasisKind::Poincare => "poincare",
        SaeAtomBasisKind::Cylinder => "cylinder",
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
///   asymmetry). For each candidate that is a `d = 2` product atom the
///   within-atom functional-ANOVA carve (#975 / #993) RUNS on the atom's own
///   fitted decoder via [`run_within_atom_carve`]: a carve that proves binding
///   blocks the fission (the atom is irreducible), an additive carve rides as a
///   fission proposal ranked by its interaction fraction, and every outcome is
///   recorded on [`HarvestReport::fission_carve_results`]. A non-product
///   candidate (no factor split) rides on the co-activation audit and is
///   counted in [`HarvestReport::fission_carve_unavailable_count`] — never a
///   silent drop. The held-out e-gate still owns final acceptance.
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

    // #993: run the within-atom functional-ANOVA carve on each fission
    // candidate that is a genuine `d = 2` product atom. The carve adjudicates
    // the representational binding question (is the surface ONE bound product
    // atom or TWO superposed factors?) on the atom's OWN fitted decoder, on the
    // same empirical code measure. A carve that PROVES binding (the interaction
    // is significant, or energetically non-negligible) blocks the fission — the
    // atom stays whole and contested; the e-gate never sees a fission proposal
    // for a bound atom. A carve that does NOT prove binding rides as a fission
    // proposal whose trigger is the carve's interaction fraction (ascending —
    // the most-separable atom sorts first), and whose binding evidence is the
    // carve's `edge_p_value`, recorded for the ledger.
    //
    // A candidate that is NOT a recoverable product atom (single-axis, sphere
    // chart, monomial patch — `factor_basis_sizes() == None`), or whose carve
    // could not run (degenerate sample, non-separable basis), is recorded
    // loudly via `fission_carve_unavailable` rather than silently dropped: its
    // fission audit still rides on the co-activation significance, exactly the
    // pre-#993 behavior, but the absence of the carve is now an explicit,
    // counted signal instead of a blanket skip.
    let mut carve_results: Vec<FissionCarveResult> = Vec::new();
    let mut fission_carve_ran_count = 0usize;
    let mut fission_carve_unavailable_count = 0usize;
    let mut fission_carve_blocked_count = 0usize;
    let mut gated_fissions: Vec<(usize, f64)> = Vec::new();
    for &(atom, significance) in fission_atoms.iter().take(params.max_fissions) {
        match run_within_atom_carve(term, atom) {
            Some(Ok(report)) => {
                fission_carve_ran_count += 1;
                let decision = fission_decision(&report, None);
                let edge_p = report.edge_p_value;
                let interaction = report.interaction_fraction;
                carve_results.push(FissionCarveResult {
                    atom,
                    edge_p_value: edge_p,
                    interaction_fraction: interaction,
                    decision,
                });
                match decision {
                    FissionDecision::Keep => {
                        // Binding proven (or interaction non-negligible): the
                        // atom is irreducible. Do NOT propose a fission.
                        fission_carve_blocked_count += 1;
                        log::debug!(
                            "[structure-harvest] #993 carve KEEPS atom {atom}: binding proven \
                             (edge_p={edge_p:?}, interaction_fraction={interaction:.3e}); no fission proposed",
                        );
                    }
                    FissionDecision::SplitReconstructionOnly
                    | FissionDecision::SplitCertifiedJoint => {
                        // Separable: propose the fission, ranked by interaction
                        // fraction (ascending — most-additive first).
                        gated_fissions.push((atom, interaction));
                    }
                }
            }
            Some(Err(err)) => {
                fission_carve_unavailable_count += 1;
                log::debug!(
                    "[structure-harvest] #993 carve could not run on atom {atom}: {err}; \
                     fission audit rides on co-activation significance, e-gate owns acceptance",
                );
                gated_fissions.push((atom, significance));
            }
            None => {
                // Not a recoverable product atom — the within-atom carve is not
                // defined here. Ride the co-activation audit, count it loudly.
                fission_carve_unavailable_count += 1;
                gated_fissions.push((atom, significance));
            }
        }
    }

    for &(atom, trigger) in &gated_fissions {
        proposals.push(proposal(term, StructureMove::Fission { atom }, trigger));
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
        fission_carve_results: carve_results,
        fission_carve_ran_count,
        fission_carve_unavailable_count,
        fission_carve_blocked_count,
        births_proposed,
        birth_skipped_reason,
    })
}

/// One within-atom carve outcome on a fission candidate (#993). Recorded on
/// the [`HarvestReport`] so the binding decision and its evidence are visible
/// — including the `edge_p_value` the dictionary certificate's `BindingEdge`
/// claim reads — never silent.
#[derive(Clone, Debug)]
pub struct FissionCarveResult {
    /// The audited product atom.
    pub atom: usize,
    /// Edge-level representational binding p-value (the carve's joint Wald over
    /// the gauge-projected interaction block). `None` when the test degenerated.
    pub edge_p_value: Option<f64>,
    /// Fraction of centered surface energy carried by the interaction
    /// (0 = perfectly additive / separable, 1 = pure interaction).
    pub interaction_fraction: f64,
    /// The carve's representational fission verdict.
    pub decision: FissionDecision,
}

/// Run the within-atom representational carve on one fitted atom (#993).
///
/// Returns:
/// * `None` — the atom is not a recoverable `d = 2` product atom (no factor
///   split); the within-atom carve is undefined here.
/// * `Some(Err(_))` — the atom is a product atom but the carve could not run
///   (degenerate sample, non-separable basis, REML fit failure).
/// * `Some(Ok(report))` — the carve ran; the report carries the binding
///   verdict and evidence.
///
/// The factor sizes come from the atom's basis evaluator
/// ([`SaeBasisEvaluator::factor_basis_sizes`]); the carve inputs are built from
/// the atom's FUSED basis and decoder by
/// [`gam_terms::structure::anova_atom::carve_input_from_fitted_atom`], which
/// verifies the Kronecker separability before fitting.
fn run_within_atom_carve(
    term: &SaeManifoldTerm,
    atom: usize,
) -> Option<Result<CarveReport, String>> {
    let a = &term.atoms[atom];
    if a.latent_dim != 2 {
        return None;
    }
    let evaluator = a.basis_evaluator.as_ref()?;
    let (m_a, m_b) = evaluator.factor_basis_sizes()?;
    let build = carve_input_from_fitted_atom(
        a.basis_values.view(),
        a.decoder_coefficients.view(),
        m_a,
        m_b,
    );
    let bundle = match build {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let input = bundle.representational_carve_input();
    Some(carve(&input, WITHIN_ATOM_CARVE_ALPHA))
}

/// The output of one [`harvest_move_proposals`] pass: the proposal stream plus
/// the loud records of any degrade-to-skip path taken (no silent drops).
#[derive(Clone, Debug)]
pub struct HarvestReport {
    /// Trigger-stamped, claim-stamped, structurally-hashed proposals, ready for
    /// [`search`] (which canonicalizes and gates them).
    pub proposals: Vec<MoveProposal>,
    /// The within-atom carve outcomes (#993): one entry per fission candidate
    /// (within the `max_fissions` cap) that is a recoverable `d = 2` product
    /// atom whose carve RAN. Carries the representational binding verdict and
    /// `edge_p_value` — the dictionary certificate's `BindingEdge` evidence —
    /// so a fission's binding decision is visible, never silent. A carve that
    /// KEEPS the atom (binding proven) blocked its fission proposal; the
    /// remaining entries' atoms each have a corresponding `Fission` proposal.
    pub fission_carve_results: Vec<FissionCarveResult>,
    /// How many fission candidates had the #993 within-atom carve actually run.
    pub fission_carve_ran_count: usize,
    /// How many fission candidates could NOT be carved (not a product atom, or
    /// the carve failed) and rode on the co-activation audit instead — the
    /// precise, never-silent record of the residual degrade path.
    pub fission_carve_unavailable_count: usize,
    /// How many fission candidates the carve BLOCKED (binding proven → atom kept
    /// whole, no fission proposed). These never reach the e-gate.
    pub fission_carve_blocked_count: usize,
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
    let assignment = crate::terms::sae::manifold::SaeAssignment::with_mode(
        logits,
        coords,
        term.assignment.mode,
    )?;
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
    evaluator: Arc<dyn SaeBasisSecondJet>,
    basis_kind: SaeAtomBasisKind,
    manifold: LatentManifold,
    latent_dim: usize,
    /// The `(n × d)` coordinates the winning basis was evaluated at — the born
    /// atom's coordinate block, dimension-matched to the winning evaluator.
    coords: Array2<f64>,
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
    evaluator: Arc<dyn SaeBasisSecondJet>,
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
///   ([`SphereChartEvaluator`]), a flat `Euclidean` patch
///   ([`EuclideanPatchEvaluator`] degree 2), and `Cylinder` `S¹ × ℝ`
///   ([`CylinderHarmonicEvaluator`]: a periodic circle axis tensored with a flat
///   line axis). The cylinder is now a first-class d=2 candidate (the basis
///   landed in `src/terms/sae/basis.rs`), so a residual that is periodic along
///   one axis and unbounded-linear along the other earns a cylinder rather than
///   being forced into a torus stand-in (which would wrap the linear axis
///   spuriously) or a flat patch (which would lose the periodicity).
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
                // T² = S¹ × S¹: each axis is a unit-period circle (the
                // fraction-of-period convention `TorusHarmonicEvaluator` shares
                // with the periodic 1-D atom). This MUST match the production
                // seeding (`AtomTopology::Torus` → Product[Circle, Circle] in
                // `sae::manifold::atom`); a flat `Euclidean` manifold would leave
                // the born atom's angles un-wrapped and the joint refit would
                // retract on the wrong geometry.
                manifold: LatentManifold::Product(vec![
                    LatentManifold::Circle { period: 1.0 },
                    LatentManifold::Circle { period: 1.0 },
                ]),
                latent_dim: 2,
                evaluator: Arc::new(TorusHarmonicEvaluator::new(2, 2)?),
                coords: coords_d(2),
            });
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Sphere,
                basis_kind: SaeAtomBasisKind::Sphere,
                // The `SphereChartEvaluator` is a (lat, lon) intrinsic chart, so
                // the latent manifold is the 2-D product of a bounded latitude
                // interval and a wrapped longitude circle — NOT
                // `LatentManifold::Sphere { dim: 2 }`, which would demand ambient
                // unit 3-vectors the chart never produces. This matches the
                // production seeding (`AtomTopology::Sphere` →
                // Product[Interval(-π/2, π/2), Circle(τ)] in `sae::manifold::atom`).
                manifold: LatentManifold::Product(vec![
                    LatentManifold::Interval {
                        lo: -std::f64::consts::FRAC_PI_2,
                        hi: std::f64::consts::FRAC_PI_2,
                    },
                    LatentManifold::Circle {
                        period: std::f64::consts::TAU,
                    },
                ]),
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
            specs.push(TopologyCandidateSpec {
                kind: AutoTopologyKind::Cylinder,
                basis_kind: SaeAtomBasisKind::Cylinder,
                // Cylinder S¹ × ℝ: axis 0 is a unit-period circle (the
                // fraction-of-period convention `CylinderHarmonicEvaluator` shares
                // with the periodic / torus atoms), axis 1 is the unbounded flat
                // line (`Euclidean`). This MUST match the production seeding
                // (`SaeAtomBasisKind::Cylinder` → Product[Circle(1.0), Euclidean]
                // in `sae::manifold::atom`); a flat `Euclidean` manifold would leave
                // the born atom's phase axis un-wrapped, and a torus stand-in would
                // wrap the linear axis spuriously. The harmonic / degree budget
                // mirrors the torus (2 circle harmonics) and the patch (degree 2)
                // so the cross-topology design widths stay commensurable.
                manifold: LatentManifold::Product(vec![
                    LatentManifold::Circle { period: 1.0 },
                    LatentManifold::Euclidean,
                ]),
                latent_dim: 2,
                evaluator: Arc::new(CylinderHarmonicEvaluator::new(2, 2)?),
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

    // The candidate basis's raw roughness Gram, the smoothness operator the
    // topology evidence prices. The gauge-invariant, basis-AGNOSTIC roughness
    // every candidate here can present analytically is the total second-derivative
    // (curvature) energy `S = Σ_n Σ_{a,c} Φ''_{·,a,c}(t_n)ᵀ Φ''_{·,a,c}(t_n)` — the
    // thin-plate / Reinsch penalty — read off each evaluator's analytic second jet
    // `Φ''[n, μ, a, c]`. A flat (line / patch) basis has a small curvature Gram
    // (its low-degree monomials are barely curved), a periodic / sphere basis a
    // large one for its high harmonics: exactly the smoothness price the race must
    // weigh against data fit. Computed identically for every candidate so the
    // cross-topology comparison stays commensurable.
    let second_jet = spec.evaluator.second_jet(spec.coords.view())?; // (n, m, d, d)
    let d = spec.latent_dim;
    let mut s_raw = Array2::<f64>::zeros((m, m));
    for row in 0..n {
        for a in 0..d {
            for c in 0..d {
                // Outer product of the (a,c) second-derivative column over basis
                // functions, accumulated into the roughness Gram.
                for mu in 0..m {
                    let hmu = second_jet[[row, mu, a, c]];
                    if hmu == 0.0 {
                        continue;
                    }
                    for nu in mu..m {
                        s_raw[[mu, nu]] += hmu * second_jet[[row, nu, a, c]];
                    }
                }
            }
        }
    }
    for mu in 0..m {
        for nu in (mu + 1)..m {
            s_raw[[nu, mu]] = s_raw[[mu, nu]];
        }
    }

    // Penalized normal-equations matrix H = Φᵀ W Φ + S(+ridge). The roughness Gram
    // is decoder-independent (a property of the basis), so the solve does not
    // chase its own decoder.
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
    //
    // NOTE (#1374, the `−½·log|S_pen|+` penalty pseudo-determinant is
    // INTENTIONALLY OMITTED): a within-model marginal likelihood also carries
    // `−½·log|S_pen|+` (see `solver::evidence::laplace_evidence`). It is left out
    // HERE because this `raw_reml` is consumed ONLY by the cross-BASIS born-atom
    // topology race (`select_topology_with_fit` → `tk_normalized_score`,
    // `PerEffectiveDim`-normalized), never as a single-model evidence.
    // `log|S_pen|+` is computed per-candidate from each basis's OWN penalty, whose
    // eigenvalue scale is basis-arbitrary (a circle's curvature penalty is not on
    // the same scale as a line's), so it is NOT commensurable across competing
    // topologies — adding it flips the cross-basis winner (it broke
    // `birth_topology_race_assigns_circle_vs_line_by_evidence`). Cross-candidate
    // complexity is already priced by the per-effective-dim scale, so the correct
    // race score on this path is `½·SSE + ½·log|H|`.
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
        .filter(|&(_, &v)| v <= s_tol)
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

    // The born atom is seeded with the RAW roughness Gram; `SaeManifoldAtom::new`
    // installs it as `smooth_penalty_raw` and `refresh_intrinsic_smooth_penalty`
    // recomputes the pullback-metric `smooth_penalty` from it + the fitted decoder
    // (the production seeding path).
    let penalty = s_raw.clone();
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
            coords: spec.coords.clone(),
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
) -> Result<Option<TopologyRaceFit>, String> {
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
    // evaluator/coords for the kind the selector hands it.
    //
    // #944 stage 4: `select_topology_with_fit` FUSES the fixed simply-connected
    // constant-curvature forms (Euclidean κ = 0 ∪ Sphere κ > 0) into ONE
    // estimated-κ `ConstantCurvature` candidate when both are present (the d = 2
    // case: euclidean-patch + sphere). That fusion is correct — euclidean-vs-sphere
    // IS a curvature estimation, not two discrete topologies — so the fused
    // `ConstantCurvature` candidate is realized by the CURVED (sphere) basis, the
    // simply-connected form that can express both flat and positively-curved
    // images under its fitted decoder. The race then adjudicates that one
    // constant-curvature form against the genuinely non-homotopic `Torus`. For
    // d = 1 no fusion fires (Circle is not simply connected), so circle-vs-line
    // races as two discrete candidates.
    let mut by_kind: std::collections::HashMap<AutoTopologyKind, &TopologyCandidateSpec> =
        std::collections::HashMap::with_capacity(specs.len() + 1);
    for spec in &specs {
        by_kind.insert(spec.kind, spec);
    }
    if !by_kind.contains_key(&AutoTopologyKind::ConstantCurvature) {
        // Resolve the fused candidate to the curved simply-connected realization
        // (sphere) when present, else the flat patch (euclidean) — whichever the
        // realizable set carries.
        if let Some(sphere) = specs.iter().find(|s| s.kind == AutoTopologyKind::Sphere) {
            by_kind.insert(AutoTopologyKind::ConstantCurvature, sphere);
        } else if let Some(euclid) = specs.iter().find(|s| s.kind == AutoTopologyKind::Euclidean) {
            by_kind.insert(AutoTopologyKind::ConstantCurvature, euclid);
        }
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
    Ok(Some(winner.fit_handle.clone()))
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
///   [`crate::structure_search::search`]) decides whether the atom is
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
/// # The race (#977)
///
/// The born atom's topology is now chosen by EVIDENCE at birth, not inherited.
/// The residual-factor direction `factor_dir` is expressed as a per-row image
/// `Y = Φ_template(coords) · factor_dir` (the structure the atom would
/// reconstruct), and [`race_birth_topology`] fits each candidate basis whose
/// intrinsic dimension matches the template's `d_k` (`d = 1`: circle vs line;
/// `d = 2`: torus vs sphere vs euclidean-patch) to `Y` by penalized least
/// squares, ranking them by TK-normalized REML — the gauge-invariant comparison
/// the smooth-term topology race applies. The WINNING topology's evaluator,
/// decoder, manifold, and roughness penalty seed the born atom, so the discovered
/// dictionary is genuinely heterogeneous: different atoms get different topologies
/// by evidence. The post-fit curved-vs-linear hybrid-split rung remains the
/// second line of defense (an atom whose curvature does not pay over the FULL
/// dictionary still collapses linear), and the held-out e-value birth gate
/// decides whether the atom is born at all. When the race finds no realizable
/// candidate (`d_k = 0` cluster-null, or a degenerate image) the born atom falls
/// back to the template basis (warm inheritance), exactly the prior behavior.
fn born_atom(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    factor_dir: ArrayView2<'_, f64>,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let k = term.k_atoms();
    if term.atoms.is_empty() {
        return Err(
            "born_atom: cannot birth from an empty dictionary (no template atom to seed the \
             coordinate block / basis from)"
                .to_string(),
        );
    }
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

    // The per-row birth target the topology race adjudicates: the residual-factor
    // direction expressed as a reconstruction image over the template
    // coordinates. A born atom seeded with `factor_dir` in the template basis
    // would emit exactly `Y = Φ_template · factor_dir`; racing topologies asks
    // which geometry parameterizes that image most parsimoniously.
    let template_coords = term.assignment.coords[0].as_matrix();
    let birth_target = template.basis_values.dot(&factor_dir); // (n, p)
    // Uniform per-row mass: at birth the routing is neutral (the atom does not yet
    // own any rows), so every row contributes equally to the topology evidence.
    let weights = Array1::<f64>::ones(birth_target.nrows());

    // Race the candidate topologies matched to the template's intrinsic dim. On a
    // win, seed the born atom from the winning evaluator + penalized decoder; on
    // no realizable candidate (cluster-null d_k, degenerate image), fall back to
    // the template basis (warm inheritance), and let the post-fit curved-vs-linear
    // rung adjudicate as before.
    let raced = race_birth_topology(
        template_coords.view(),
        birth_target.view(),
        weights.view(),
        template.latent_dim,
    )?;
    // The born atom + its coordinate block. The race-won path carries the winning
    // topology's coordinate block (dimension-matched to its evaluator, manifold
    // set to the winning chart); the fallback path reuses the template block.
    let (born, born_coord_block) = match raced {
        Some(fit) => {
            // Build the born atom directly from the winning topology's realized
            // basis: its evaluator, penalized decoder, and roughness penalty. The
            // intrinsic (pullback-metric) penalty is then refreshed from the
            // seeded decoder so the atom carries exactly the production seeding.
            let mut atom = SaeManifoldAtom::new(
                format!("atom_born_{k}"),
                fit.basis_kind.clone(),
                fit.latent_dim,
                fit.phi.clone(),
                fit.jet.clone(),
                fit.decoder.clone(),
                fit.penalty.clone(),
            )?
            .with_basis_second_jet(fit.evaluator.clone());
            atom.refresh_intrinsic_smooth_penalty();
            // Coordinate block matched to the winning evaluator's intrinsic dim,
            // carrying the winning chart manifold so the joint refit retracts on
            // the right geometry.
            let coord_block = gam_terms::latent::LatentCoordValues::from_matrix_with_manifold(
                fit.coords.view(),
                LatentIdMode::None,
                fit.manifold.clone(),
            );
            (atom, coord_block)
        }
        None => {
            // The born atom reuses the template's structural basis (kind, latent
            // dim, basis values + jacobian + raw penalty); only its decoder carries
            // the residual-factor direction.
            let mut atom = template.clone();
            atom.decoder_coefficients = factor_dir.to_owned();
            atom.refresh_intrinsic_smooth_penalty();
            (atom, term.assignment.coords[0].clone())
        }
    };
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
    coords.push(born_coord_block);
    let assignment = crate::terms::sae::manifold::SaeAssignment::with_mode(
        logits,
        coords,
        term.assignment.mode,
    )?;
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

impl StructureSearchResult {
    /// `true` iff at least one structure-changing move LANDED across the rounds —
    /// an `Accepted` move (certified birth / fission / fusion that restructured
    /// the dictionary and triggered a warm refit) or a `Demoted` death (an atom
    /// folded to ~0 routing). Both mutate the returned `term`/`rho` away from the
    /// pre-search joint fit, so any shape uncertainty assembled from the
    /// PRE-search joint Hessian is stale and must be recomputed from the final
    /// post-search per-atom inner fits (#1230). When this is `false`, every round
    /// was contested / vetoed / deduplicated / deferred / stale, the term/rho are
    /// byte-for-byte the pre-search fit, and the exact joint-Hessian bands remain
    /// valid.
    #[must_use]
    pub fn structure_changed(&self) -> bool {
        use crate::structure_search::MoveVerdict;
        self.rounds.iter().any(|round| {
            round.moves.iter().any(|record| {
                matches!(
                    record.verdict,
                    MoveVerdict::Accepted { .. } | MoveVerdict::Demoted { .. }
                )
            })
        })
    }
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
    mut finalize_round: impl FnMut(
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

        // #993 item 3: BANK the within-atom carve binding evidence in the
        // ledger. The carve ran on each `d = 2` product-atom fission candidate
        // (`harvest_move_proposals` → `run_within_atom_carve`) and reported a
        // representational binding p-value; absorb it as a `BindingEdge` claim
        // on the atom's OWN two factors (a self-edge `{atom, atom}`: the carve
        // asks whether THIS atom's two latent factors are bound). A small
        // `edge_p_value` (interaction proven) calibrates to strong positive
        // evidence FOR the binding claim via `log_e_from_p_calibrator`; a
        // p ≈ 1 (additive) absorbs evidence AGAINST it. This makes the binding
        // verdict not merely observable on the `HarvestReport` but BANKED in
        // the persisted ledger, so the dictionary certificate covers it and the
        // evidence resumes across corpus shards. A `None` p-value (the Wald
        // test degenerated) is skipped — no fabricated evidence.

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
                crate::structure_search::MoveVerdict::Accepted { .. }
                    | crate::structure_search::MoveVerdict::Demoted { .. }
            )
        });
        rounds.push(round_ledger);

        if applied {
            // The adopted winner reached its restructured form through the cheap
            // capped-iteration SCORING refit; re-refit it at the full inner
            // budget (same estimation-row weighting) before it becomes the next
            // round's parent and the returned dictionary, so the cap is a
            // scoring-only economy and the adopted state matches a direct
            // full-iter refit (the inner solve is convergent; the capped score
            // was only a worse starting iterate). When no move landed, the state
            // is byte-identical to the unrefit pre-search parent and needs no
            // polish.
            let (polished_term, polished_rho) =
                finalize_round(next_term, next_rho, &split.estimation_rows);
            term = polished_term;
            rho = polished_rho;
        } else {
            term = next_term;
            rho = next_rho;
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
    let reconstruction = -0.5 * sse;

    // Occam-priced gate-block evidence (#1016/#1218). The split-LR difference
    // this gate forms is between the K+1 candidate (alternative) and the K null;
    // the gate/assignment-logit block is the weakest-Gaussian piece of the SAE
    // evidence and is mispriced by a plain Laplace quadratic near a birth. The
    // deterministic Pólya–Gamma gate-block marginal supplies the correct
    // normalizer, whose `−½·d_g·log(2π)` term scales with the gate dimension
    // `d_g` (one coordinate per atom). Because the candidate carries one more
    // gate coordinate than the null, that `d_g`-dependent normalizer does NOT
    // cancel in the K-vs-(K+1) difference — it is exactly the per-coordinate
    // `log(2π)` Occam term #1218 corrects the sign of. Folding it into the
    // evaluation likelihood is what makes the corrected sign reach the live
    // gate decision (the unit test alone never touched this path).
    let gate_evidence = gate_block_log_evidence(term, shard);

    reconstruction + gate_evidence
}

/// The deterministic Pólya–Gamma gate-block marginal log-evidence of the
/// candidate's per-atom logistic gates on a shard's held-out rows (#1016/#1218).
///
/// Each atom carries one free per-atom gate logit, so the gate block is a stack
/// of `K` one-dimensional logistic gates: design `X_g = 1` (the per-atom gate
/// coordinate), tilt `ψ̂ =` the atom's per-row logit, binomial response `y =`
/// the binarized activation (`b = 1`), under a unit ridge gate prior. The
/// returned value is the log-evidence `−neg_log_evidence` from
/// [`crate::inference::pg_gate_evidence::pg_gate_evidence`], summed over atoms,
/// so the K-dependent `−½·d_g·log(2π)` normalizer enters the gate's split-LR.
///
/// A degenerate/non-PD gate block contributes `0` (no gate evidence) rather than
/// poisoning the reconstruction likelihood — a conservative, valid degradation.
fn gate_block_log_evidence(term: &SaeManifoldTerm, shard: &RowBlockShard) -> f64 {
    use crate::inference::pg_gate_evidence::{GateBlock, pg_gate_evidence};

    let logits = &term.assignment.logits;
    let n_full = logits.nrows();
    let k = logits.ncols();
    if k == 0 {
        return 0.0;
    }
    // Restrict to the shard's held-out rows; an empty / out-of-range shard
    // carries no gate evidence.
    let rows: Vec<usize> = shard.rows.iter().copied().filter(|&r| r < n_full).collect();
    let m = rows.len();
    if m == 0 {
        return 0.0;
    }

    // Unit gate design (one gate coordinate per atom) and a unit ridge gate
    // prior; the PG block is solved per atom and summed, so `d_g = K` overall.
    let design = Array2::<f64>::ones((m, 1));
    let b = Array1::<f64>::ones(m);
    let penalty = Array2::<f64>::eye(1);

    let mut total = 0.0_f64;
    for atom in 0..k {
        let mut psi = Array1::<f64>::zeros(m);
        let mut y = Array1::<f64>::zeros(m);
        for (i, &row) in rows.iter().enumerate() {
            let logit = logits[[row, atom]];
            if !logit.is_finite() {
                return 0.0;
            }
            psi[i] = logit;
            // Binarized activation: the gate is ON when its logit is positive.
            y[i] = if logit > 0.0 { 1.0 } else { 0.0 };
        }
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: Some(psi.view()),
            penalty: Some(penalty.view()),
            hess_rest: None,
            h_rest: None,
        };
        match pg_gate_evidence(&block) {
            // `neg_log_evidence` is `−log p(gate block)`; the log-likelihood the
            // split-LR consumes is its negation.
            Ok(ev) => total -= ev.neg_log_evidence,
            // A non-PD / degenerate gate block contributes no evidence.
            Err(_) => return 0.0,
        }
    }
    total
}

#[inline]
fn sse_accumulate(sse: &mut f64, d: f64) {
    *sse += d * d;
}

/// Inner-fit knobs for the production structure-search refit (the same numbers
/// the outer SAE fit drove its inner Arrow-Schur joint fit with).
#[derive(Clone, Copy, Debug)]
pub struct ProductionRefitParams {
    /// Inner Newton iterations for the FULL refit that produces the adopted
    /// state (the round winner that becomes the next round's parent and the
    /// returned dictionary). Quality-bearing — kept at the outer fit's budget.
    pub inner_max_iter: usize,
    /// Inner Newton iterations for the per-candidate SCORING refit only (the
    /// many rejected / contested candidates the e-gate ranks each round). A
    /// structural move yields a WARM child — the parent's converged dictionary
    /// with one atom restructured (see [`apply_structure_move`]) — so only the
    /// touched atom must re-equilibrate before the held-out evidence gate can
    /// rank it; a small budget suffices. This caps the dominant cost of the
    /// search (≈ K·rounds full-dictionary refits) WITHOUT changing the adopted
    /// state: every round's winner is re-refit at the full `inner_max_iter`
    /// budget before it is adopted, so cap-then-polish reaches the same inner
    /// optimum as a direct full-iter refit (the inner solve is convergent; the
    /// capped score is only a worse starting iterate for the polish). The e-gate
    /// DECISION reads the capped score, so this is the one quantity that can in
    /// principle differ from a full-iter search; it is held to a tight
    /// match-or-equivalent bar (#1026).
    pub scoring_inner_max_iter: usize,
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
    let n = target.nrows();
    // Refit a restructured candidate on the ESTIMATION rows only at a chosen
    // inner-iteration budget: the held-out evaluation rows carry a near-zero
    // weight (vanishing fitting pressure) via the per-row reconstruction-weight
    // seam, so the candidate is the predictable plug-in the held-out shards are
    // scored against. The seam requires strictly-positive weights, so a tiny
    // epsilon stands in for the structural zero; after mean-1 normalization the
    // estimation rows carry weight ≈ n/n_est and the held-out rows ≈ 0. A
    // non-converging inner solve returns the unchanged candidate (infallible at
    // the boundary). `inner_max_iter` is the only knob that varies between the
    // cheap per-candidate scoring pass and the full-iter polish of the adopted
    // winner; `full_target` is borrowed by reference so the helper holds no owned
    // capture and can be called from both closures below.
    let refit_at = |full_target: ArrayView2<'_, f64>,
                    mut cand_term: SaeManifoldTerm,
                    mut cand_rho: SaeManifoldRho,
                    estimation_rows: &[usize],
                    inner_max_iter: usize|
     -> (SaeManifoldTerm, SaeManifoldRho) {
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
        if cand_term
            .run_joint_fit_arrow_schur(
                full_target,
                &mut cand_rho,
                None,
                inner_max_iter,
                refit_params.learning_rate,
                refit_params.ridge_ext_coord,
                refit_params.ridge_beta,
            )
            .is_err()
        {
            return (cand_term, cand_rho);
        }
        (cand_term, cand_rho)
    };
    let scoring_iters = refit_params.scoring_inner_max_iter;
    let full_iters = refit_params.inner_max_iter;
    let full_target_score = target.to_owned();
    let full_target_polish = target.to_owned();
    run_structure_search_rounds(
        term,
        rho,
        target,
        config,
        ledger,
        // Per-candidate SCORING refit: capped (warm child, few iters).
        move |cand_term, cand_rho, estimation_rows| {
            refit_at(
                full_target_score.view(),
                cand_term,
                cand_rho,
                estimation_rows,
                scoring_iters,
            )
        },
        // Full-iter POLISH of each round's adopted winner before it becomes the
        // next round's parent / the returned dictionary, so the cap is a
        // scoring-only economy and the adopted state matches a full-iter refit.
        move |adopted_term, adopted_rho, estimation_rows| {
            refit_at(
                full_target_polish.view(),
                adopted_term,
                adopted_rho,
                estimation_rows,
                full_iters,
            )
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
    use crate::structure_search::{CollapseAction, CollapseEvent};
    use gam_terms::latent::LatentManifold;
    use crate::terms::sae::manifold::{
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
    /// #1230 — `StructureSearchResult::structure_changed()` is the trigger the
    /// FFI uses to decide whether the pre-search joint-Hessian shape bands are
    /// stale and must be recomputed from the final post-search model.
    ///
    /// It must report `true` iff at least one move LANDED and mutated the
    /// returned `term`/`rho`: an `Accepted` move (certified birth / fission /
    /// fusion + warm refit) or a `Demoted` death. It must report `false` when
    /// every round was contested / vetoed (the term/rho are byte-for-byte the
    /// pre-search fit, so the exact joint-Hessian bands stay valid), and when no
    /// round ran at all. A false negative leaves seed atoms with stale bands
    /// (the #1230 bug); a false positive needlessly discards exact bands.
    #[test]
    fn structure_changed_is_true_only_when_a_move_lands() {
        use crate::structure_search::{MoveRecord, MoveVerdict};

        fn ledger_with(verdicts: Vec<MoveVerdict>) -> SearchLedger {
            SearchLedger {
                alpha: 0.05,
                moves: verdicts
                    .into_iter()
                    .enumerate()
                    .map(|(i, verdict)| MoveRecord {
                        mv: StructureMove::Death { atom: i },
                        trigger: 0.0,
                        structure_hash: i as u64,
                        claim: ClaimKind::AtomExists { atom: i },
                        verdict,
                    })
                    .collect(),
                collapse_events: Vec::new(),
            }
        }

        // No rounds ran at all: nothing changed.
        let (term0, rho0) = planted_term(&[vec![true], vec![true]]);
        let empty = StructureSearchResult {
            term: term0.clone(),
            rho: rho0.clone(),
            rounds: Vec::new(),
        };
        assert!(
            !empty.structure_changed(),
            "no rounds ⇒ the term/rho are the pre-search fit ⇒ structure_changed() must be false"
        );

        // Every move contested or vetoed: the dictionary is byte-for-byte the
        // pre-search fit, so the exact joint-Hessian bands remain valid.
        let no_landed = StructureSearchResult {
            term: term0.clone(),
            rho: rho0.clone(),
            rounds: vec![ledger_with(vec![
                MoveVerdict::Contested { log_e: -1.0 },
                MoveVerdict::Vetoed { log_e: -2.0 },
            ])],
        };
        assert!(
            !no_landed.structure_changed(),
            "all-contested/vetoed rounds leave the model unchanged ⇒ structure_changed() must be false"
        );

        // An Accepted move landed (certified restructuring + warm refit): the
        // returned model differs from the pre-search fit ⇒ bands are stale.
        let accepted = StructureSearchResult {
            term: term0.clone(),
            rho: rho0.clone(),
            rounds: vec![ledger_with(vec![
                MoveVerdict::Contested { log_e: -1.0 },
                MoveVerdict::Accepted { log_e: 3.0 },
            ])],
        };
        assert!(
            accepted.structure_changed(),
            "a landed Accepted move mutates term/rho ⇒ structure_changed() must be true (recompute bands)"
        );

        // A Demoted death is also a landed structure change.
        let demoted = StructureSearchResult {
            term: term0.clone(),
            rho: rho0.clone(),
            rounds: vec![ledger_with(vec![MoveVerdict::Demoted { log_e: -1.0 }])],
        };
        assert!(
            demoted.structure_changed(),
            "a landed Demoted death folds an atom to ~0 routing ⇒ structure_changed() must be true"
        );
    }

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
        let report = harvest_move_proposals(&term, &rho, zero_residual.view(), &params).unwrap();
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
    /// asymmetry, parent conditional ≈ 1). The planted atoms are 1-D `Periodic`
    /// (NOT a `d = 2` product), so the #993 within-atom carve is undefined on
    /// them and the candidate rides on the co-activation audit — recorded
    /// loudly via `fission_carve_unavailable_count`, never silent.
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
        assert_eq!(
            report.fission_carve_ran_count, 0,
            "1-D periodic atoms are not a product manifold; the within-atom carve cannot run"
        );
        assert!(
            report.fission_carve_unavailable_count >= 1,
            "the non-product fission candidate must be recorded as carve-unavailable, not silent"
        );
        assert!(
            report.fission_carve_results.is_empty(),
            "no carve ran, so there are no carve results to report"
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
            run_structure_search_rounds(
                term,
                rho,
                target.view(),
                config,
                &mut ledger,
                |t, r, _| (t, r),
                // No-op polish: this determinism oracle scripts the gate and
                // never runs the SAE inner solve.
                |t, r, _| (t, r),
            )
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

    /// #1026 move-equivalence oracle for the candidate-SCORING iteration cap.
    ///
    /// The production structure search caps the per-candidate scoring refit's
    /// inner iterations (`scoring_inner_max_iter`) well below the outer fit's
    /// `inner_max_iter`, then re-refits each round's adopted winner at the full
    /// budget. The economy is sound only if it does NOT change WHICH moves the
    /// e-gate accepts (the gate ranks the capped-score candidate) NOR the adopted
    /// dictionary (the winner is polished to the full-iter optimum). This runs
    /// the real `run_production_structure_search` driver — same residual-bearing
    /// target, seed, splits, budgets — twice: a full-iter REFERENCE
    /// (`scoring_inner_max_iter == inner_max_iter`) and the CAPPED production
    /// path, and asserts the accepted-move ledger is byte-identical and the
    /// final fitted reconstruction matches to a tight tolerance. On a tractable
    /// K/n where the full-iter reference completes, equivalence here certifies
    /// the cap is a pure scoring-cost economy (the #1026 perf fix is quality- and
    /// decision-preserving).
    #[test]
    fn scoring_iter_cap_preserves_moves_and_adopted_fit() {
        // A residual-bearing single-atom fit so the birth channel mines a real
        // shared-structure candidate the gate can certify (mirrors
        // `residual_bearing_fit_harvests_birth_proposal`'s planted factor).
        let n = 40usize;
        let active: Vec<Vec<bool>> = (0..n).map(|_| vec![true]).collect();
        let p = 4usize;
        let u = [0.6_f64, -0.4, 0.5, -0.3];
        let mut target = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let amp = 1.0 + (row as f64) / (n as f64);
            for c in 0..p {
                target[[row, c]] = amp * u[c % u.len()];
            }
        }
        let config = RoundDriverConfig {
            n_shards: 4,
            budget: MoveBudget {
                max_moves: 4,
                alpha: 0.05,
            },
            max_rounds: 2,
            harvest_params: HarvestParams {
                max_fusions: 2,
                max_fissions: 2,
                max_births: 2,
            },
        };
        let full_iters = 24usize;
        let run = |scoring_inner_max_iter: usize| {
            let (term, rho) = planted_term(&active);
            let mut ledger = StructureLedger::new();
            let refit_params = ProductionRefitParams {
                inner_max_iter: full_iters,
                scoring_inner_max_iter,
                learning_rate: 1.0,
                ridge_ext_coord: 1e-6,
                ridge_beta: 1e-6,
            };
            let result = run_production_structure_search(
                term,
                rho,
                target.view(),
                config,
                refit_params,
                &mut ledger,
            )
            .unwrap();
            let fitted = result.term.try_fitted().unwrap();
            (result, fitted)
        };

        // Full-iter reference: scoring budget == full budget (no economy).
        let (reference, ref_fitted) = run(full_iters);
        // Production cap: a warm child re-equilibrates in very few iters.
        let (capped, cap_fitted) = run(4);

        // The accepted-move trajectory — the per-round `moves` (each carrying the
        // proposed `mv`, its `trigger`, `structure_hash`, `claim`, and the e-gate
        // `verdict`) — must be identical: the cap must not flip a single e-gate
        // decision. We compare ONLY `moves`, NOT the per-round `collapse_events`.
        // The collapse-event log is the #976 active-mass guard's per-INNER-iteration
        // diagnostic trail: a full-iter refit runs more inner Newton steps than a
        // capped one, so it legitimately records more reseed/terminal guard fires
        // for the SAME structural outcome. Those events are a refit-trajectory
        // diagnostic, not an e-gate decision — the move verdicts already encode
        // their structural effect — so comparing them would assert a property the
        // cap is not meant to preserve (and the adopted-fit check below is the
        // real guarantee that the converged state matches).
        let round_moves = |rounds: &[SearchLedger]| -> String {
            serde_json::to_string(&rounds.iter().map(|r| &r.moves).collect::<Vec<_>>()).unwrap()
        };
        assert_eq!(
            round_moves(&reference.rounds),
            round_moves(&capped.rounds),
            "scoring-iteration cap changed the accepted-move trajectory — the e-gate \
             decisions are NOT cap-invariant (the #1026 economy is unsound)"
        );
        assert_eq!(
            reference.term.k_atoms(),
            capped.term.k_atoms(),
            "scoring cap changed the discovered dictionary size"
        );

        // The adopted dictionary's reconstruction must match: the full-iter
        // polish lands the capped path on the same inner optimum.
        assert_eq!(ref_fitted.dim(), cap_fitted.dim());
        let mut max_abs = 0.0_f64;
        for (a, b) in ref_fitted.iter().zip(cap_fitted.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-6,
            "capped-scoring adopted fit diverged from the full-iter reference by \
             {max_abs:.3e} (> 1e-6); the polish did not reach the same optimum"
        );
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

    /// #977 per-atom topology RACE oracle: two birth targets — one tracing a
    /// CIRCLE in output space as the coordinate sweeps, the other a straight
    /// LINE — must be assigned DIFFERENT topologies by evidence. A genuine
    /// dictionary learner does not stamp every born atom with atom-0's circle
    /// template: the circular residual earns a Periodic (circle) basis, the
    /// straight residual a EuclideanPatch (line). This is the heterogeneous,
    /// evidence-chosen dictionary the issue demands.
    #[test]
    fn birth_topology_race_assigns_circle_vs_line_by_evidence() {
        use std::f64::consts::TAU;

        let n = 80usize;
        // A monotone 1-D latent coordinate the residual image is parameterized by.
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);

        // CIRCLE target: γ(t) = (cos 2πt, sin 2πt) — full revolution, strong
        // turning a straight line cannot express. Two output channels carry the
        // circle; the rest are zero.
        let p = 4usize;
        let mut circle_target = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let t = coords[[row, 0]];
            circle_target[[row, 0]] = (TAU * t).cos();
            circle_target[[row, 1]] = (TAU * t).sin();
        }

        // LINE target: γ(t) = t·u — a straight ray, zero turning. The circle basis
        // has no parsimony advantage; the cheaper line wins on evidence.
        let mut line_target = Array2::<f64>::zeros((n, p));
        let u = [0.7_f64, -0.4, 0.5, -0.2];
        for row in 0..n {
            let t = coords[[row, 0]];
            for c in 0..p {
                line_target[[row, c]] = t * u[c];
            }
        }

        let weights = Array1::<f64>::ones(n);

        let circle_fit =
            race_birth_topology(coords.view(), circle_target.view(), weights.view(), 1)
                .expect("circle race runs")
                .expect("circle race has a realizable candidate");
        let line_fit = race_birth_topology(coords.view(), line_target.view(), weights.view(), 1)
            .expect("line race runs")
            .expect("line race has a realizable candidate");

        assert_eq!(
            circle_fit.basis_kind,
            SaeAtomBasisKind::Periodic,
            "a circular birth residual must win the circle (Periodic) topology"
        );
        assert_eq!(
            line_fit.basis_kind,
            SaeAtomBasisKind::EuclideanPatch,
            "a straight birth residual must win the line (EuclideanPatch) topology"
        );
        // The crux: the two atoms get DIFFERENT topologies by evidence — the
        // dictionary is heterogeneous, not all-circle.
        assert_ne!(
            circle_fit.basis_kind, line_fit.basis_kind,
            "the discovery must assign DIFFERENT topologies to the circle and line \
             atoms (evidence-chosen, not inherited)"
        );
    }

    /// #977 d=2 topology-race COMPLETENESS: the candidate set includes the
    /// Cylinder kind, and a birth target that is genuinely cylindrical — periodic
    /// along one latent axis and unbounded-linear along the other — is adjudicated
    /// to the Cylinder topology, not forced into a torus (which would wrap the
    /// linear axis spuriously) or a flat patch (which would lose the periodicity).
    /// This is the realizable d=2 race the issue demands: torus / sphere /
    /// euclidean / cylinder, evidence-chosen.
    #[test]
    fn birth_topology_race_d2_includes_and_selects_cylinder() {
        use std::f64::consts::TAU;

        // The d=2 candidate set must literally CONTAIN the cylinder candidate.
        let n = 120usize;
        let coords = Array2::<f64>::from_shape_fn((n, 2), |(row, axis)| {
            // axis 0: a phase that completes ~2 revolutions over the rows;
            // axis 1: a monotone unbounded coordinate.
            if axis == 0 {
                (row as f64 / n as f64) * 2.0
            } else {
                (row as f64 / n as f64) * 3.0 - 1.5
            }
        });
        let specs = topology_candidates_for_dim(coords.view(), 2).expect("d=2 candidates build");
        let has_cylinder = specs
            .iter()
            .any(|s| s.basis_kind == SaeAtomBasisKind::Cylinder);
        assert!(
            has_cylinder,
            "the d=2 topology-race candidate set MUST include the Cylinder kind; got {:?}",
            specs.iter().map(|s| &s.basis_kind).collect::<Vec<_>>()
        );
        let has_torus = specs
            .iter()
            .any(|s| s.basis_kind == SaeAtomBasisKind::Torus);
        let has_sphere = specs
            .iter()
            .any(|s| s.basis_kind == SaeAtomBasisKind::Sphere);
        let has_patch = specs
            .iter()
            .any(|s| s.basis_kind == SaeAtomBasisKind::EuclideanPatch);
        assert!(
            has_torus && has_sphere && has_patch,
            "the d=2 race must be COMPLETE (torus + sphere + euclidean + cylinder)"
        );

        // CYLINDER target: periodic along axis 0 (cos/sin of the phase) AND
        // linearly growing along axis 1 (a magnitude ramp). A torus would have to
        // wrap the magnitude axis (no periodicity there); a flat patch cannot
        // express the full revolution; the cylinder expresses both exactly.
        let p = 4usize;
        let mut cyl_target = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let phase = coords[[row, 0]];
            let mag = coords[[row, 1]];
            cyl_target[[row, 0]] = (TAU * phase).cos();
            cyl_target[[row, 1]] = (TAU * phase).sin();
            // The linear-axis structure: a magnitude ramp on a third channel.
            cyl_target[[row, 2]] = mag;
        }
        let weights = Array1::<f64>::ones(n);
        let cyl_fit = race_birth_topology(coords.view(), cyl_target.view(), weights.view(), 2)
            .expect("cylinder race runs")
            .expect("cylinder race has a realizable candidate");
        assert_eq!(
            cyl_fit.basis_kind,
            SaeAtomBasisKind::Cylinder,
            "a cylindrical birth residual (periodic along one axis, linear along the \
             other) must win the Cylinder topology by evidence; got {:?}",
            cyl_fit.basis_kind
        );
    }

    /// #977 BORN-ATOM UNCERTAINTY: a structure-search-born atom (grown past the
    /// seed K the joint Schur factor was assembled at) must report a FINITE
    /// shape-uncertainty band, computed from its OWN fitted penalized inner
    /// Hessian — never a silently-missing band. This is the completed deferred
    /// gap: `complete_born_atom_shape_bands` fills the born atom's band so no
    /// post-search atom is reported without honest uncertainty.
    #[test]
    fn born_atom_reports_finite_uncertainty_band() {
        let n = 48usize;
        // A K=1 seed dictionary, routed on every row.
        let active: Vec<Vec<bool>> = (0..n).map(|_| vec![true]).collect();
        let (term, rho) = planted_term(&active);
        let k_seed = term.k_atoms();

        // Grow the dictionary by one BORN atom (index k_seed) seeded from a
        // residual-factor decoder. This is the atom the pre-search Schur factor
        // never covered.
        let p = term.output_dim();
        let m = term.atoms[0].basis_size();
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[1, 0]] = 0.9;
        decoder[[2, 1]] = -0.6;
        let (mut born, born_rho) = apply_structure_move(
            &term,
            &rho,
            &StructureMove::Birth { candidate: 0 },
            &[decoder],
        )
        .expect("birth applies");
        assert_eq!(born.k_atoms(), k_seed + 1, "the birth grows K by one");

        // Build a reconstruction target the born dictionary fits, then harvest the
        // per-atom inner fits at the settled state (this populates the per-atom
        // penalized inner Hessian for EVERY atom, born included).
        let target = born.try_fitted().expect("born term reconstructs");
        let dispersion = 1.0e-2_f64;
        born.set_atom_inner_fits(target.view(), &born_rho, dispersion)
            .expect("inner fits build");

        // The pre-search Schur factor only covered the SEED atoms: emulate the
        // production path by starting from a band list that is missing the born
        // atom (the no-decoder-covariance fallback over the seed atoms), then
        // completing it.
        let mut unc = born.shape_uncertainty_without_decoder_covariance(dispersion);
        // Truncate to the seed atoms to emulate a Schur factor assembled at the
        // seed K (the born atom has NO entry yet).
        unc.atoms.truncate(k_seed);
        assert_eq!(
            unc.atoms.len(),
            k_seed,
            "seed-K Schur band omits the born atom"
        );

        born.complete_born_atom_shape_bands(&mut unc)
            .expect("born-atom band completes");

        // Every post-search atom now has a band slot.
        assert_eq!(
            unc.atoms.len(),
            born.k_atoms(),
            "completion must grow the band list to the post-search atom count"
        );
        let born_band = &unc.atoms[k_seed];
        assert!(
            born_band.band_sd.nrows() > 0 && born_band.band_sd.ncols() == p,
            "the born atom's band must be shaped (G>0, p)"
        );
        // The crux: the born atom's band is FINITE and non-negative everywhere —
        // an honest uncertainty, not a silently-missing (or NaN) band.
        let mut any_positive = false;
        for &sd in born_band.band_sd.iter() {
            assert!(
                sd.is_finite() && sd >= 0.0,
                "born-atom band sd must be finite and non-negative; got {sd}"
            );
            if sd > 0.0 {
                any_positive = true;
            }
        }
        assert!(
            any_positive,
            "a born atom with a non-degenerate inner Hessian must report a strictly \
             positive uncertainty somewhere (a finite band, never all-zero / missing)"
        );
    }

    /// #1218 PRODUCTION-GATE wiring proof: the corrected PG gate-block
    /// normalizer is consumed by the live per-shard likelihood the K-vs-(K+1)
    /// birth gate forms its split-LR from — not just by the isolated unit test.
    ///
    /// `eval_log_lik` is the exact `alternative_log_lik` / `null_sup_log_lik`
    /// closure `run_atom_birth_gate` accumulates (see [`run_structure_search_rounds`]),
    /// so it is the production gate's evaluation statistic. We score the SAME
    /// shard under a K-atom null and a (K+1)-atom candidate and isolate the
    /// gate-block contribution: growing the dictionary by one atom adds exactly
    /// one gate coordinate, so the `−½·d_g·log(2π)` normalizer (the term #1218
    /// fixed the sign of) does NOT cancel in the gate difference. With the
    /// corrected (subtracted) sign it is an Occam PENALTY that resists the
    /// extra atom; the buggy (added) sign would flip it into a spurious REWARD.
    #[test]
    fn production_gate_consumes_corrected_pg_normalizer() {
        let n = 32usize;
        // K=2 null and a K=3 candidate, every atom routed on every row so the
        // gate logits are well-defined and finite.
        let null_active: Vec<Vec<bool>> = (0..n).map(|_| vec![true, true]).collect();
        let cand_active: Vec<Vec<bool>> = (0..n).map(|_| vec![true, true, true]).collect();
        let (null_term, _) = planted_term(&null_active);
        let (cand_term, _) = planted_term(&cand_active);
        assert_eq!(null_term.k_atoms(), 2);
        assert_eq!(cand_term.k_atoms(), 3, "candidate grows K by one atom");

        // One held-out shard: the row block the gate accumulates evidence over.
        let p = null_term.output_dim();
        let target = Arc::new(Array2::<f64>::zeros((n, p)));
        let shard = RowBlockShard {
            target: target.clone(),
            rows: (0..n).collect(),
        };

        // The gate-block contribution alone (private helper the live
        // `eval_log_lik` adds in): the corrected normalizer is reachable here.
        let null_gate = gate_block_log_evidence(&null_term, &shard);
        let cand_gate = gate_block_log_evidence(&cand_term, &shard);
        assert!(
            null_gate.is_finite() && cand_gate.is_finite(),
            "gate-block evidence must be finite on a well-posed gate block"
        );

        // The Occam normalizer per added gate coordinate. The candidate carries
        // K+1 gate coordinates, the null K, so the gate-difference includes one
        // extra `−½·log(2π)` normalizer that must NOT cancel.
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let gate_delta = cand_gate - null_gate;

        // Corrected sign ⇒ the per-coordinate normalizer SUBTRACTS, so the
        // extra atom's gate-block log-evidence is pushed DOWN by ≈ ½·log(2π)
        // relative to a no-normalizer baseline. The decisive, sign-sensitive
        // assertion: the extra-coordinate normalizer is the *negative*
        // ½·log(2π) Occam term, never the positive (buggy) one. Compare against
        // the per-atom evidence WITHOUT the normalizer to isolate it.
        let per_atom_no_norm = |term: &SaeManifoldTerm| -> f64 {
            // Re-derive the gate evidence with the normalizer ADDED back (the
            // pre-fix sign) to recover the unnormalized quadratic/logdet part.
            // `gate_block_log_evidence` already SUBTRACTS ½·d_g·log(2π); adding
            // it back yields the normalizer-free score, and the difference
            // between candidate and null of THAT isolates everything except the
            // one extra normalizer.
            let dg = term.k_atoms() as f64; // one gate coordinate per atom
            gate_block_log_evidence(term, &shard) + 0.5 * dg * log_2pi
        };
        let no_norm_delta = per_atom_no_norm(&cand_term) - per_atom_no_norm(&null_term);
        let normalizer_in_delta = gate_delta - no_norm_delta;

        // The normalizer contribution to the K→K+1 gate difference must be
        // exactly `−½·log(2π)` (one extra gate coordinate, corrected sign).
        assert!(
            (normalizer_in_delta + 0.5 * log_2pi).abs() < 1e-9,
            "the gate-block normalizer in the K→K+1 difference must be the \
             corrected −½·log(2π) Occam penalty, got {normalizer_in_delta} \
             (buggy +½·log(2π) = {})",
            0.5 * log_2pi
        );

        // And the full production statistic carries it: the gate-block evidence
        // is a real, finite addend on top of the reconstruction likelihood.
        let full = eval_log_lik(&cand_term, &shard);
        let recon_only = {
            // Reconstruction-only baseline (what the path returned BEFORE the
            // wiring): −½·SSE over the shard rows.
            let fitted = cand_term.try_fitted().unwrap();
            let mut sse = 0.0;
            for &row in &shard.rows {
                for out in 0..p {
                    let d = fitted[[row, out]] - shard.target[[row, out]];
                    sse += d * d;
                }
            }
            -0.5 * sse
        };
        assert!(
            (full - (recon_only + cand_gate)).abs() < 1e-9,
            "the live per-shard likelihood must equal reconstruction + the \
             PG gate-block evidence (so the corrected normalizer reaches the gate)"
        );
    }
}
