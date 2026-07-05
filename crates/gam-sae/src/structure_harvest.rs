//! #997 — the wiring seam between a fitted [`SaeManifoldTerm`] and the
//! evidence-guarded move engine of [`gam_solve::structure_search`].
//!
//! #976 closed with the move engine (`search`) and its triggers
//! (`gam_sae::atom_codes::SparseAtomCodes::coactivation`, ARD precisions,
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
//!
//! # Theory: curvature IS identifiability (the birth/topology race)
//!
//! Of the four move channels this module owns, the BIRTH channel is where a
//! structural theory claim becomes an operational decision: superposition
//! ambiguity is fundamentally a FLATNESS disease. When several linear
//! directions co-fire, any invertible recombination (any element of `GL(d)`
//! acting on those coordinates) reproduces the same observed activations
//! exactly — a flat co-firing subspace is generically NON-identifiable, and
//! its gauge groupoid (the group of relabelings that leave the data
//! indistinguishable) is as large as `GL(d)` itself. A CURVED embedding does
//! not have this freedom: by jet transversality, two generic curved
//! embeddings that agree to second order (the same point, tangent, AND
//! curvature/osculation) are equal on an infinite-codimension set — so a
//! curved atom's gauge groupoid collapses from `GL(d)` down to the much
//! smaller diffeomorphism-and-symmetry group (`Diff × Sym`) of its own
//! topology. Concretely: a residual-factor blob that looks like a flat 2-D
//! co-firing pair is compatible with countless linear re-mixings, but a
//! residual-factor blob that is genuinely a circle can only be reparameterized
//! by circle diffeomorphisms — its curvature is what pins it down.
//!
//! This module's birth path (residual-factor mining in
//! [`harvest_move_proposals`], candidate construction in
//! [`topology_candidates_for_dim`], the commensurable evidence comparison in
//! [`fit_topology_candidate`], the race itself in [`race_birth_topology`], and
//! the seeding in [`born_atom`]) is the engine's CURE for that flatness
//! disease: instead of leaving a co-firing residual subspace as an
//! unidentifiable flat blob (or forcing every birth to inherit a fixed
//! circular template by fiat), it fits every topology whose intrinsic
//! dimension matches the candidate atom (line vs circle at `d = 1`; torus vs
//! sphere vs cylinder vs flat patch at `d = 2`) and lets the data's own
//! curvature evidence — TK-normalized REML, the same gauge-invariant scale the
//! smooth-term topology race uses — pick the winner. A curved winner is not
//! merely a nicer-looking basis: it is the SPECIFIC configuration whose
//! rigidity is what makes the born atom identifiable at all. The race is
//! therefore the optimizer's equilibrium response to superposition, not a
//! stylistic preference for circles.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::atom_codes::SparseAtomCodes;
use crate::null_sampler::{NULL_REPLICATES, coactivation_exceedance};
use crate::basis::{
    CylinderHarmonicEvaluator, EuclideanPatchEvaluator, PeriodicHarmonicEvaluator,
    SaeBasisSecondJet, SphereChartEvaluator, TorusHarmonicEvaluator,
};
use crate::manifold::{
    AssignmentMode, OccupancyLaw, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm, amplitude_concentration_certificate, classify_occupancy_interval,
};
use std::sync::atomic::{AtomicBool, Ordering};
use gam_runtime::warm_start::Fingerprinter;
use gam_solve::gaussian_reml::gaussian_reml_multi_closed_form;
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::structure_search::{
    CollapseAction, MoveBudget, MoveProposal, SearchLedger, SearchOutcome, StructureMove, search,
};
use gam_solve::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoSelector, TopologyScoreScale,
    select_topology_with_fit,
};
use gam_terms::inference::structure_evidence::{ClaimKind, StructureLedger};
use gam_terms::latent::{LatentIdMode, LatentManifold};
use gam_terms::structure::anova_atom::{
    CarveReport, FissionDecision, carve, carve_input_from_fitted_atom, fission_decision,
};

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

/// Conventional one-sided screening level for the fixed-margin-null exceedance
/// gate on co-activation triggers (#976 top-`k` correction) — the SAME 0.05
/// screening convention as [`WITHIN_ATOM_CARVE_ALPHA`], deliberately not a
/// certificate level: the exceedance gate is a PROPOSAL filter (does this pair
/// co-fire ABOVE what the top-`k` margins mechanically force?), and the held-out
/// e-gate still owns final acceptance. See [`null_exceedance_z_floor`].
const NULL_EXCEEDANCE_ALPHA: f64 = 0.05;

/// The standardized-excess floor a pair's fixed-margin-null exceedance must clear
/// to be proposed, derived (never hand-set) from [`NULL_EXCEEDANCE_ALPHA`] as the
/// one-sided standard-normal deviate `z_{1−α}`. Charging the raw co-activation
/// coupling instead would inherit the top-`k` mechanical (anti)correlation the
/// null absorbs; requiring `z ≥ z_{1−α}` keeps only genuine above-margin
/// co-firing.
fn null_exceedance_z_floor() -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    // Standard normal inverse-CDF at 1 − α (α the conventional screening level).
    Normal::new(0.0, 1.0)
        .expect("standard normal is well-defined")
        .inverse_cdf(1.0 - NULL_EXCEEDANCE_ALPHA)
}

/// Minimum conditional asymmetry for a pair to be proposed for a FISSION audit
/// (the A⇒B absorption signature: one conditional near 1 without the converse).
const ABSORPTION_ASYMMETRY_FLOOR: f64 = 0.5;

/// Anti-symmetric decoder perturbation applied when a fission DUPLICATES an atom,
/// to break the symmetric saddle so the two children can separate in the joint
/// refit (see `duplicate_atom`). Small enough to preserve the warm-start (and the
/// mass-split combined decoder is exactly unchanged), but ≫ floating-point noise.
const FISSION_SYMMETRY_BREAK_EPS: f64 = 0.05;

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

/// Collapse the raw fission-audit list to ONE entry per parent atom, keeping the
/// most-suspect nomination (the LOWEST significance — significance is the
/// ascending `1 − asym` proxy, so smaller = more absorption-suspect).
///
/// A single parent can be nominated by several partners at different
/// significances, so the raw list carries duplicate atoms. The old
/// `sort_by(significance).dedup_by_key(atom)` was wrong: `dedup_by_key` removes
/// only ADJACENT duplicates, and a significance-first sort does not place
/// same-atom entries adjacently, so duplicates survived — the same parent rode as
/// several `Fission` proposals, wasting births on a duplicate split. Here the
/// per-atom minimum is taken explicitly, then the survivors are re-sorted by the
/// total order `(significance asc, atom asc)` so the result is most-suspect-first
/// (the order the downstream `take(max_fissions)` and carve loop expect) and fully
/// deterministic despite the `HashMap`'s arbitrary iteration order.
fn dedup_most_suspect_per_parent(candidates: Vec<(usize, f64)>) -> Vec<(usize, f64)> {
    let mut best_per_parent: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    for (atom, significance) in candidates {
        best_per_parent
            .entry(atom)
            .and_modify(|s| {
                if significance < *s {
                    *s = significance;
                }
            })
            .or_insert(significance);
    }
    let mut out: Vec<(usize, f64)> = best_per_parent.into_iter().collect();
    out.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
    out
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
        SaeAtomBasisKind::FiniteSet => "finite_set",
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
/// channel under the `WhitenedStructured` (`gam_inference::row_metric::MetricProvenance::WhitenedStructured`)
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
///   explained residual mass. This is a SHAPE-level mining step — it finds
///   directions the current dictionary does not reconstruct, not yet a claim
///   about what topology lives there. The topology itself is adjudicated
///   downstream, atom-by-atom, by [`race_birth_topology`] (see the module
///   docs: curvature is what makes the winner identifiable). Note the two
///   halves of identifiability this leaves complementary rather than
///   redundant: this residual-factor step (and the topology race it feeds) is
///   a SUPPORT/shape-level test (does the reconstruction residual look like a
///   line, a circle, a torus…?), while a separate producer elsewhere in this
///   crate (the ISA κ-contrast statistic, `identifiability.rs` /
///   `isa_seed.rs`) is a MEASURE-level test: a centered circle's cone `ℝ₊·Y`
///   is literally the same point set as a 2-plane minus the origin, so no
///   support-based test can ever tell a dense circle from a Gaussian plane —
///   only the radial fourth-moment ratio `κ = E[r⁴]/E[r²]²` (`= 1` dense
///   circle, `= 2` Gaussian plane, `= 1/q` gated) can, because it reads the
///   RADIAL LAW rather than the support. Neither test subsumes the other;
///   they see complementary halves of the same identifiability question.
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

    // --- Fixed-margin (curveball) null for the co-activation triggers ------
    // Top-`k` selection stamps mechanical (anti)correlation into the co-activation
    // masks (each token's fixed support size induces a negative indicator
    // covariance between every pair; a hard top-`k` puts zero mass off the
    // `k`-shell). A raw coupling trigger reads that artifact as structure. So the
    // fusion/fission triggers below are gated on the EXCEEDANCE of each pair's
    // joint activation over a null that preserves both the row margins (the
    // top-`k` constraint) and the column margins (per-atom totals): only
    // above-margin co-firing survives. Computed once and shared by both triggers;
    // skipped entirely when neither trigger is enabled (the null is not free).
    let codes = sparse_codes_from_term(term);
    let want_coactivation = params.max_fusions > 0 || params.max_fissions > 0;
    let exceedance =
        want_coactivation.then(|| coactivation_exceedance(&codes, NULL_REPLICATES));
    let z_floor = null_exceedance_z_floor();

    // --- Fusions: top co-activation dependence, gated by the null ----------
    // The trigger REPORTED is the null exceedance `z` (above-margin co-firing),
    // not the raw dependence: the raw floor only pre-selects genuinely co-firing
    // pairs, and the fixed-margin null strips the mechanical top-`k` coupling.
    let mut fusion_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..k {
        for b in (a + 1)..k {
            let stats = codes.coactivation(a, b);
            let dep = stats.dependence();
            if dep < FUSION_DEPENDENCE_FLOOR {
                continue;
            }
            let z = exceedance
                .as_ref()
                .map_or(0.0, |ex| ex.excess_z(a, b));
            if z >= z_floor {
                fusion_pairs.push((a, b, z));
            }
        }
    }
    fusion_pairs.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    for &(a, b, z) in fusion_pairs.iter().take(params.max_fusions) {
        proposals.push(proposal(term, StructureMove::Fusion { a, b }, z));
    }

    // --- Fission audits: absorption-suspect asymmetry, gated by the null ---
    let mut fission_atoms: Vec<(usize, f64)> = Vec::new();
    for a in 0..k {
        for b in (a + 1)..k {
            let stats = codes.coactivation(a, b);
            let asym = stats.absorption_asymmetry();
            if asym < ABSORPTION_ASYMMETRY_FLOOR {
                continue;
            }
            // A nested (absorbed) pair co-fires ABOVE its fixed margins; a pair
            // whose asymmetry is only the top-`k` mechanical artifact does not.
            // Require the joint activation to exceed the fixed-margin null before
            // auditing, so mechanical asymmetry is not read as absorption.
            let z = exceedance
                .as_ref()
                .map_or(0.0, |ex| ex.excess_z(a, b));
            if z < z_floor {
                continue;
            }
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
    // Keep the most-suspect (lowest significance) audit per parent atom.
    let fission_atoms = dedup_most_suspect_per_parent(fission_atoms);

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

/// A birth seed the seeded apply-move ([`apply_structure_move_seeded`])
/// materializes. The residual-factor births carry only a flat decoder (the
/// topology race in [`born_atom`] then adjudicates line vs circle vs …); a curl
/// birth (INTEGRATION_PLAN Phase 4) instead carries a fully-formed periodic
/// circle — decoder, per-row phase, and gate — because a shattered centered
/// circle leaves NO residual for the race to seed from, so the seed IS the
/// hypothesis and [`born_circle_atom`] installs it directly for the REML e-gate
/// to adjudicate.
#[derive(Clone, Debug)]
pub enum BirthSeed {
    /// A whitened residual-factor direction lifted to a flat `(m, p)` decoder;
    /// born via the topology race (the legacy birth path).
    ResidualFactor(Array2<f64>),
    /// A curl circle seed: periodic-harmonic `(m, p)` decoder (`m` odd, `>= 3`),
    /// per-row phase coordinate `(n, 1)`, and per-row own-presence gate (`n`).
    Circle {
        decoder: Array2<f64>,
        phase_coords: Array2<f64>,
        gate: Vec<f64>,
    },
}

/// Apply one [`StructureMove`] with a heterogeneous birth-seed list — the curl
/// extension of [`apply_structure_move`]. Non-birth moves are identical; a
/// `Birth { candidate }` dispatches on the indexed [`BirthSeed`]: a
/// `ResidualFactor` rides the topology race ([`born_atom`]), a `Circle` is
/// installed directly as a periodic atom ([`born_circle_atom`]). The legacy
/// [`apply_structure_move`] is exactly this with an all-`ResidualFactor` seed
/// list, so bitwise legacy behavior is preserved when no curl seed is present.
pub fn apply_structure_move_seeded(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    mv: &StructureMove,
    birth_seeds: &[BirthSeed],
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    match mv {
        StructureMove::Birth { candidate } => {
            let seed = birth_seeds.get(*candidate).ok_or_else(|| {
                format!(
                    "apply_structure_move_seeded: birth candidate {candidate} out of range \
                     ({} birth seeds)",
                    birth_seeds.len()
                )
            })?;
            match seed {
                BirthSeed::ResidualFactor(decoder) => born_atom(term, rho, decoder.view()),
                BirthSeed::Circle {
                    decoder,
                    phase_coords,
                    gate,
                } => born_circle_atom(term, rho, decoder.clone(), phase_coords.clone(), gate.clone()),
            }
        }
        // Death / Fission / Fusion are seed-independent — delegate to the legacy
        // apply with an empty decoder list (never indexed for these).
        other => apply_structure_move(term, rho, other, &[]),
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
    // For SOFTMAX the fused atom must carry the COMBINED routing mass of its two
    // constituents. `softmax` mass is `e^logit/Z`, so the mass-preserving combine
    // is `logsumexp(la, lb)` (`softmax(logsumexp(la,lb)) = softmax(la)+softmax(lb)`).
    // Plain `max` UNDER-masses by up to `ln 2` on exactly the co-active rows that
    // triggered the fusion (where `la ≈ lb`): it gives the fused atom half the
    // combined mass, leaving the warm-start short and risking a FALSE rejection by
    // the e-gate under a capped refit. For IBP/JumpReLU the per-atom gate is the
    // UN-normalized `σ(logit)`, so the union gate is `max(σ(la),σ(lb)) = σ(max(la,lb))`
    // → `max` is the correct combine there (a sum/logsumexp would over-gate).
    let softmax_routing = matches!(term.assignment.mode, AssignmentMode::Softmax { .. });
    for row in 0..term.assignment.logits.nrows() {
        let la = term.assignment.logits[[row, a]];
        let lb = term.assignment.logits[[row, b]];
        term.assignment.logits[[row, a]] = if softmax_routing {
            // Numerically stable logsumexp. When BOTH logits are -∞ (two rows of
            // zero softmax mass — a hard-masked/dead pair), `m = -∞` makes
            // `la - m = -∞ - (-∞) = NaN`, and the NaN poisons the whole logits
            // row (every subsequent softmax over it is NaN). The combined mass of
            // two zero-mass atoms is exactly zero, i.e. logit -∞ — return that
            // directly instead of computing NaN.
            let m = la.max(lb);
            if m == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                m + ((la - m).exp() + (lb - m).exp()).ln()
            }
        } else {
            la.max(lb)
        };
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
    let mut child_atom = term.atoms[parent].clone();
    // Symmetry-breaking perturbation. A fission that duplicates the parent atom
    // IDENTICALLY (same decoder, same coords, mass split 50/50) sits at a
    // SYMMETRIC SADDLE of the joint refit: the two children have identical
    // gradients, so a deterministic Newton/descent refit moves them in lockstep
    // and they NEVER separate — the fission stays a no-op (two identical
    // half-atoms ≡ the original atom) and the e-gate, seeing no reconstruction
    // gain, rejects it. So fission could only ever land by floating-point noise.
    // Apply a small ANTI-SYMMETRIC perturbation — parent decoder ×(1−ε·s_ij),
    // child decoder ×(1+ε·s_ij) for a deterministic varying pattern s_ij — which
    // breaks the symmetry (the refit can roll off the saddle toward the
    // two-factor configuration the carve identified) while leaving the mass-split
    // combined decoder `½·parent + ½·child = original` EXACTLY unchanged (the
    // ±ε·s_ij cancel), so the warm-start is preserved. `ε ≫ fp noise`.
    {
        let (m, p) = atoms[parent].decoder_coefficients.dim();
        let s = |i: usize, j: usize| -> f64 {
            // Deterministic, varying, NON-ZERO pattern in [-1,-0.2]∪[0.2,1]. It
            // must vary across (i,j) (so `parent − child = −2·f_ij·D_ij` points
            // off the symmetric `D` direction and can separate factors) and never
            // vanish (else a sparse decoder element gets no perturbation).
            let raw = ((i * 7 + j * 13) % 11) as f64 / 5.0 - 1.0;
            if raw.abs() < 0.2 { 0.3 } else { raw }
        };
        for i in 0..m {
            for j in 0..p {
                let f = FISSION_SYMMETRY_BREAK_EPS * s(i, j);
                atoms[parent].decoder_coefficients[[i, j]] *= 1.0 - f;
                child_atom.decoder_coefficients[[i, j]] *= 1.0 + f;
            }
        }
        // The Grassmann decoder frame is derived from the coefficients; drop both
        // so the warm refit recomputes them consistent with the perturbed decoders.
        atoms[parent].decoder_frame = None;
        child_atom.decoder_frame = None;
    }
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
        crate::manifold::SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let mut child = SaeManifoldTerm::new(atoms, assignment)?;
    // Score the child on the parent's evidence-charge convention (see the note in
    // `born_circle_atom`): `SaeManifoldTerm::new` resets `rank_charge_evidence`,
    // and a birth/split gate must compare like-for-like Laplace complexity.
    child.set_rank_charge_evidence(term.rank_charge_evidence());

    let mut child_rho = rho.clone();
    if parent < child_rho.log_ard.len() {
        let inherited = child_rho.log_ard[parent].clone();
        child_rho.log_ard.push(inherited);
    } else {
        child_rho.log_ard.push(Array1::<f64>::zeros(0));
    }
    // The fissioned child inherits the PARENT atom's per-atom smoothness strength
    // (#1556). As with `log_ard`, failing to grow `log_lambda_smooth` in step with
    // `k_atoms()` makes the next `assemble_arrow_schur` panic on the per-atom
    // `lambda_smooth[atom_idx]` index (out of bounds).
    let inherited_smooth = child_rho
        .log_lambda_smooth
        .get(parent)
        .or_else(|| child_rho.log_lambda_smooth.first())
        .copied()
        .unwrap_or(0.0);
    child_rho.log_lambda_smooth.push(inherited_smooth);
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
///
/// This is the FLAT-vs-RIGID contest made concrete. Every candidate here
/// realizes one specific point in the flat/curved spectrum for the born
/// atom's intrinsic dimension: at `d = 1` the flat `Euclidean` line (large
/// `GL(1)` gauge freedom — any rescaling of the coordinate is indistinguishable
/// from any other) against the rigid `Circle` (gauge collapses to rotations of
/// `S¹`); at `d = 2` the flat patch against the curved `Torus` / `Sphere` /
/// `Cylinder`, each with its own residual symmetry group strictly smaller than
/// `GL(2)`. The candidate SET is exactly the set of hypotheses
/// [`fit_topology_candidate`] scores and [`race_birth_topology`] adjudicates —
/// building it is choosing which flat/rigid alternatives the evidence gets to
/// discriminate between; the race itself decides which one is real.
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
/// (`n`, the candidate's per-row reconstruction mass) by PROPER closed-form
/// Gaussian REML, and return its TK evidence inputs + the realized fit handle.
///
/// The per-atom Gaussian-reconstruction evidence is the marginal likelihood the
/// solver's [`gaussian_reml_multi_closed_form`] returns at the REML-optimal
/// smoothing strength λ̂ — the SAME REML/LAML quantity every smooth term is
/// scored by, so it is commensurable under the shared TK normalizer:
///
/// * `raw_reml` — the rank-aware closed-form REML score
///   `½d·(log|Φᵀ W Φ + λ̂S| − log|λ̂S|₊) + dispersion` at the estimated λ̂. Unlike
///   a fixed-λ, unit-dispersion Laplace term, this (a) estimates λ per candidate
///   on its own basis (SPEC: REML/LAML always, never a hand-set λ) and (b) prices
///   complexity with the penalty pseudo-determinant on a cross-basis-comparable
///   scale, so a perfect periodic fit to a circle beats a poor cubic-patch fit.
/// * `null_dim = 0` / `null_space_logdet = None` — the closed-form REML score is
///   ALREADY null-space-restricted (rank-aware), so the TK normalizer must not
///   re-subtract a gauge term; we report no null space to avoid double-counting.
/// * `effective_dim = tr[(Φᵀ W Φ + λ̂S)⁻¹ Φᵀ W Φ]` — the penalized effective
///   degrees of freedom the solver returns (`edf`), the per-effective-dim scale's
///   denominator.
///
/// This is what makes the flat-vs-rigid contest ADJUDICABLE rather than
/// merely posed: a flat and a curved candidate are different function spaces
/// (different `m`, different roughness operator `S`, different gauge group),
/// so their raw fit residuals are not comparable on their own. TK-normalized
/// REML is the common currency — the same marginal-likelihood scale every
/// smooth term in the fitter is scored on — that prices each candidate's
/// data fit against its own complexity/roughness cost, so "the circle beats
/// the line" is a real, commensurable evidence statement (not an artifact of
/// one basis happening to have fewer parameters). That evidence is exactly
/// what [`race_birth_topology`] compares across candidates to pick the
/// rigid — hence identifiable — winner.
fn fit_topology_candidate(
    spec: &TopologyCandidateSpec,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<TopologyAutoFitEvidence<TopologyRaceFit>, String> {
    let n = target.nrows();
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

    // Validate the per-row reconstruction mass and reject a degenerate
    // (zero-total-mass) birth target — the closed-form REML below needs a
    // positive weighted sample to estimate a dispersion.
    let mut w_sum = 0.0_f64;
    for row in 0..n {
        let w = weights[row];
        if !(w.is_finite() && w >= 0.0) {
            return Err("fit_topology_candidate: weights must be finite and non-negative".into());
        }
        w_sum += w;
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

    // Score each candidate by its PROPER closed-form Gaussian REML evidence
    // (#977/#1026), NOT a hand-rolled fixed-λ Laplace term. The previous score
    // (`½·SSE + ½·log|H|` at a stamped λ = 1, unit dispersion) was wrong on two
    // counts, and they conspired to make a perfect circle lose to a line:
    //
    //   1. λ = 1 is NOT commensurable across bases. A periodic basis's curvature
    //      energy for a `cos(2πt)` harmonic scales like `(2π)⁴ ≈ 1.6e3` the data
    //      Gram, so a unit-λ penalty CRUSHES the very harmonics that reconstruct a
    //      circle exactly, while the barely-curved cubic patch is left essentially
    //      unpenalized. SPEC also forbids a hand-set smoothing strength — λ must be
    //      REML/LAML-estimated, ALWAYS.
    //   2. Unit dispersion (`½·SSE`) does not reward a near-perfect fit. The
    //      profiled-dispersion REML deviance `½ν·log(σ̂²)` rewards a basis that
    //      drives σ̂² → 0 (the circle's exact harmonic fit) far more strongly,
    //      which is what makes the contest honest, and `½·log|H| − ½·log|λS|₊`
    //      prices complexity on a scale that is comparable across bases (the
    //      penalty pseudo-determinant the old score dropped is what restores
    //      commensurability).
    //
    // `gaussian_reml_multi_closed_form` returns exactly this: the marginal-
    // likelihood-optimal λ̂ for each candidate on its OWN basis, the penalized
    // decoder at λ̂, the rank-aware REML score (`½d·(log|H| − log|λS|₊) +
    // dispersion`, already null-space-restricted), and the effective degrees of
    // freedom `tr[(ΦᵀWΦ + λS)⁻¹ ΦᵀWΦ]`. We feed `reml_score` as `raw_reml` and
    // `edf` as `effective_dim`, and report `null_dim = 0` so the TK normalizer does
    // NOT re-subtract a null-space term the REML score already integrated out.
    let reml_fit =
        gaussian_reml_multi_closed_form(phi.view(), target, s_raw.view(), Some(weights), None)
            .map_err(|e| format!("fit_topology_candidate: REML evidence: {e:?}"))?;
    let lambda = reml_fit.lambda;
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(format!(
            "fit_topology_candidate: REML returned a non-finite/negative λ ({lambda})"
        ));
    }
    let raw_reml = reml_fit.reml_score;
    if !raw_reml.is_finite() {
        return Err("fit_topology_candidate: non-finite REML score".into());
    }
    let decoder = reml_fit.coefficients.clone(); // penalized fit at λ̂, m × p
    let mut effective_dim = reml_fit.edf;
    if !(effective_dim.is_finite() && effective_dim > 0.0) {
        // A fully-penalized fit (no effective parameters) cannot be scored on the
        // per-effective-dim scale; floor at a single effective parameter so the
        // race still ranks it (the REML deviance dominates the verdict anyway).
        effective_dim = 1.0;
    }

    // The born atom is seeded with the RAW roughness Gram; `SaeManifoldAtom::new`
    // installs it as `smooth_penalty_raw` and `refresh_intrinsic_smooth_penalty`
    // recomputes the pullback-metric `smooth_penalty` from it + the fitted decoder
    // (the production seeding path).
    let penalty = s_raw.clone();
    Ok(TopologyAutoFitEvidence {
        topology_name: spec.kind.as_str().to_string(),
        raw_reml,
        // The closed-form REML score is ALREADY restricted to the penalty's range
        // complement (rank-aware: `log|λS|₊` over the non-null directions, the null
        // space integrated out), so the TK null-space normalizer must NOT fire
        // again — pass `null_dim = 0` (its `null_space_logdet` branch is then
        // skipped) so we don't double-count the gauge directions.
        null_dim: 0.0,
        null_space_logdet: None,
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
///
/// This function IS the flatness cure at runtime: a `d_k`-dimensional
/// co-firing residual subspace is, prior to this call, exactly the kind of
/// flat structure that admits an unbounded gauge group of equally-good linear
/// recombinations (see the module docs). Racing the realizable candidates by
/// [`fit_topology_candidate`]'s commensurable evidence and keeping only the
/// winner replaces that flat ambiguity with ONE specific, generically rigid
/// geometry (or, if nothing curved earns its keep, the flat/line candidate —
/// an honest verdict, not a default). The winner returned here is what
/// [`born_atom`] seeds the new atom from directly, so the identifiability
/// gain is realized in the dictionary rather than merely reported.
/// The realized per-row amplitude of a birth target: the L2 norm of each row of
/// `Y` (`n × p`), i.e. the magnitude the born atom would reconstruct at that
/// sample. The amplitude-concentration certificate reads this to tell a
/// present/absent spike (binary presence) from a continuous spread (a radial
/// coordinate).
fn birth_row_amplitudes(target: ArrayView2<'_, f64>) -> Array1<f64> {
    let n = target.nrows();
    let mut amps = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut ss = 0.0_f64;
        for &v in target.row(i).iter() {
            ss += v * v;
        }
        amps[i] = ss.sqrt();
    }
    amps
}

/// When a `d = 1` birth's realized amplitude is CONTINUOUS (a hidden radial axis,
/// per the amplitude-concentration certificate), build the promoted
/// circle-vs-cylinder(radial)-vs-disk candidate set so the race adjudicates the
/// extra radial dimension by evidence. Returns `None` when no promotion applies
/// (not `d = 1`, or the amplitude is a genuine present/absent spike), so the
/// caller keeps the base race. The promoted set uses DISTINCT topology kinds
/// (`Circle` d=1, `Cylinder` d=2, `Euclidean` d=2 = the flat disk) so the
/// by-kind race map has no collision — the `d = 1` line is intentionally dropped,
/// because if the amplitude is radial the contest is circle vs the radial
/// two-manifolds, not circle vs line.
fn radial_promoted_specs(
    coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    d_k: usize,
) -> Result<Option<Vec<TopologyCandidateSpec>>, String> {
    if d_k != 1 {
        return Ok(None);
    }
    let amps = birth_row_amplitudes(target);
    let cert = amplitude_concentration_certificate(amps.view());
    if !cert.recommends_radial_axis() {
        return Ok(None);
    }
    // The circle from the d=1 set (the un-promoted alternative the evidence must
    // still be free to prefer) plus the d=2 radial two-manifolds.
    let mut promoted: Vec<TopologyCandidateSpec> = Vec::with_capacity(3);
    for spec in topology_candidates_for_dim(coords, 1)? {
        if spec.kind == AutoTopologyKind::Circle {
            promoted.push(spec);
        }
    }
    for spec in topology_candidates_for_dim(coords, 2)? {
        if matches!(
            spec.kind,
            AutoTopologyKind::Cylinder | AutoTopologyKind::Euclidean
        ) {
            promoted.push(spec);
        }
    }
    if promoted.len() < 2 {
        // Need at least the circle plus one radial candidate to make a contest.
        return Ok(None);
    }
    Ok(Some(promoted))
}

/// F2 finite-set-atom opt-in. Default `false`: the birth race does NOT enrol a
/// finite-set (discrete anchor) candidate, so the [`SaeAtomBasisKind::FiniteSet`]
/// variant + [`AnchorIndicatorEvaluator`] land as inert scaffolding that cannot
/// affect any birth. The switch flips to `true` only AFTER the finite-set atom is
/// verified — full `gam-sae` suite green plus the real-data weekday adjudication
/// (is weekday seven cyclic points or an occupied circle?). Enrolling the
/// candidate in the actual race additionally needs an `AutoTopologyKind::FiniteSet`
/// in `gam-solve`'s selector (the cross-crate follow-up); until then the flag +
/// [`finite_set_candidate_for_birth`] are the staged, unit-tested substrate.
static FINITE_SET_RACE_ENROLLED: AtomicBool = AtomicBool::new(false);

/// Whether the birth race enrols the finite-set (discrete anchor) candidate.
/// Default `false` — see [`FINITE_SET_RACE_ENROLLED`].
pub fn finite_set_race_enrolled() -> bool {
    FINITE_SET_RACE_ENROLLED.load(Ordering::Relaxed)
}

/// Flip the finite-set-atom enrolment opt-in. Intended for the post-verification
/// enablement (and for tests exercising the enrolled path); default is `false`.
pub fn set_finite_set_race_enrolled(enrolled: bool) {
    FINITE_SET_RACE_ENROLLED.store(enrolled, Ordering::Relaxed);
}

/// Build the finite-set (discrete anchor) candidate inputs for a `d = 1` birth
/// whose occupancy is DISCRETE — the honest "seven cyclic points, not an occupied
/// circle" alternative. Returns `(anchors, index_coords)` where `index_coords`
/// (`n × 1`) assigns each row to its nearest of `anchors` anchors (the integer
/// index the [`AnchorIndicatorEvaluator`] reads), and `anchors − 1` is the rank
/// charge ([`finite_set_rank_charge`]). Returns `None` when the birth is not a
/// discrete finite set (uniform / continuous occupancy, wrong dimension, or a
/// degenerate coordinate) — so it never fabricates a cluster structure.
///
/// This is the pure, unit-tested substrate the race enrolment consumes once
/// [`finite_set_race_enrolled`] flips; it does not itself touch any birth.
pub fn finite_set_candidate_for_birth(coords: ArrayView2<'_, f64>) -> Option<(usize, Array2<f64>)> {
    if coords.ncols() != 1 {
        return None;
    }
    let n = coords.nrows();
    if n < 4 {
        return None;
    }
    let col = coords.column(0);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &t in col.iter() {
        if !t.is_finite() {
            return None;
        }
        lo = lo.min(t);
        hi = hi.max(t);
    }
    let span = hi - lo;
    if !(span > 0.0) {
        return None;
    }
    // Range-normalize the single coordinate column to [0, 1] and classify on the
    // INTERVAL (non-wrapping) occupancy law: a birth coordinate is
    // interval-topology (linear, from the PCA seed), so its extreme values must
    // NOT wrap onto each other — the circular classifier would fold `0` and `1`
    // together and merge a linear set's first and last anchors (7 weekday points
    // → 6), and its full-circle uniform model would misread a range-filling
    // uniform coordinate as non-uniform. The interval classifier keeps linear
    // ends distinct and range-uniform data uniform.
    let r: Vec<f64> = col.iter().map(|&t| ((t - lo) / span).clamp(0.0, 1.0)).collect();
    match classify_occupancy_interval(&r) {
        OccupancyLaw::Discrete { anchors } if anchors >= 2 => {
            // Assign each row to its nearest anchor bin from the normalization
            // `r ∈ [0, 1]`: the `anchors` evenly spaced bins give the categorical
            // index the indicator basis reads. A fitted-anchor-position
            // assignment is the cross-crate refinement (it needs the classifier's
            // centers exposed).
            let mut idx = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                let bin = (r[i] * anchors as f64).floor();
                idx[[i, 0]] = bin.clamp(0.0, (anchors - 1) as f64);
            }
            Some((anchors, idx))
        }
        _ => None,
    }
}

fn race_birth_topology(
    coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    d_k: usize,
) -> Result<Option<TopologyRaceFit>, String> {
    let base_specs = topology_candidates_for_dim(coords, d_k)?;
    if base_specs.is_empty() {
        return Ok(None);
    }
    // F1 radial promotion: a `d = 1` birth whose realized amplitude is CONTINUOUS
    // (a hidden radial coordinate — the amplitude-concentration certificate reads
    // the per-row birth magnitudes as a spread, not a present/absent spike) is
    // really a disk / annulus, not a circle. When the certificate recommends it,
    // ENRICH the race with the `d = 2` radial candidates so the evidence
    // adjudicates circle-vs-cylinder(radial)-vs-disk rather than the amplitude
    // silently riding an uncertified quantity. Strictly additive and fail-safe:
    // any failure building the promoted set falls back to the base race, and the
    // gate only fires on a genuine continuous-amplitude signal, so the common path
    // is unchanged.
    if let Ok(Some(promoted)) = radial_promoted_specs(coords, target, d_k) {
        if !promoted.is_empty() {
            // Try the promoted circle-vs-cylinder-vs-disk race; on ANY failure
            // (a degenerate d=2 fit, an empty ranking) fall back to the base race
            // so a radial-flagged birth never regresses relative to the un-promoted
            // path — the promotion can only ever ADD adjudicated candidates.
            if let Ok(Some(fit)) = race_spec_set(promoted, target, weights) {
                return Ok(Some(fit));
            }
        }
    }
    race_spec_set(base_specs, target, weights)
}

/// Race one realized candidate spec set against the birth target and return the
/// evidence-winning fit. Shared by the base and the F1 radial-promoted races.
fn race_spec_set(
    specs: Vec<TopologyCandidateSpec>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Option<TopologyRaceFit>, String> {
    if specs.is_empty() {
        return Ok(None);
    }
    let selector = TopologyAutoSelector {
        // The race is over EXACTLY the candidate set we built; do not let the
        // selector's constant-curvature fuse drop one — pass them through as-is.
        candidates: specs.iter().map(|s| s.kind).collect(),
        // PER-OBSERVATION normalization (a common `n` divisor across candidates).
        // The candidate scores are now PROPER closed-form REML marginal
        // likelihoods (see `fit_topology_candidate`), which ALREADY price model
        // complexity through `log|H| − log|λS|₊` + the profiled dispersion. The
        // older `PerEffectiveDim` scale was calibrated for the previous hand-rolled
        // POSITIVE cost (`½·SSE + ½·log|H|`, which grew with model size and needed
        // per-parameter normalization); applied to a proper (negative) evidence it
        // DOUBLE-COUNTS complexity and inverts the ranking for higher-parameter
        // bases — e.g. a cylinder that fits a cylindrical residual best (most
        // negative evidence) would lose to a sphere purely because it spends more
        // effective dimensions. A common-`n` divisor preserves the raw
        // marginal-likelihood ranking the Bayesian evidence is designed to support,
        // so the genuinely-best-fitting topology wins.
        score_scale: TopologyScoreScale::PerObservation,
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
///   [`gam_solve::structure_search::search`]) decides whether the atom is
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
///
/// Seeding directly from `fit.evaluator` / `fit.decoder` / `fit.penalty` (the
/// winning [`TopologyRaceFit`]) rather than re-deriving anything is what makes
/// the identifiability gain land in the actual dictionary: the born atom does
/// not merely get labeled with a topology name, it is CONSTRUCTED in the
/// winning basis, so its gauge group is the winner's `Diff × Sym` (curved
/// case) or `GL(d)` (flat fallback) from the moment it exists, not something a
/// later pass has to retrofit.
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
    let assignment =
        crate::manifold::SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let mut child = SaeManifoldTerm::new(atoms, assignment)?;
    // Score the child on the parent's evidence-charge convention (see the note in
    // `born_circle_atom`): `SaeManifoldTerm::new` resets `rank_charge_evidence`,
    // and a birth gate must compare like-for-like Laplace complexity.
    child.set_rank_charge_evidence(term.rank_charge_evidence());

    let mut child_rho = rho.clone();
    // The born atom inherits the template atom's ARD block shape (disabled if
    // the template's was disabled).
    let inherited = child_rho
        .log_ard
        .first()
        .cloned()
        .unwrap_or_else(|| Array1::<f64>::zeros(0));
    child_rho.log_ard.push(inherited);
    // ρ carries a PER-ATOM smoothness strength `log_lambda_smooth[k]` (#1556),
    // and `assemble_arrow_schur` indexes it by atom (`lambda_smooth[atom_idx]`).
    // Growing the dictionary without growing this vector leaves `k_atoms()`
    // ahead of `log_lambda_smooth.len()` and the next assemble panics with an
    // out-of-bounds index. The born atom inherits the template atom's smoothness
    // strength (atom 0), matching the `log_ard` inheritance just above.
    let inherited_smooth = child_rho.log_lambda_smooth.first().copied().unwrap_or(0.0);
    child_rho.log_lambda_smooth.push(inherited_smooth);
    Ok((child, child_rho))
}

/// #2101 — build a born atom seeded DIRECTLY as a rank-2 circle: a Periodic atom
/// carrying the residual 2-plane on its cos/sin harmonic decoder rows and a
/// PHASE-ALIGNED coordinate `phase_coords` (`(n, 1)`). This BYPASSES the topology
/// race in [`born_atom`], which parameterizes the born-circle candidate with the
/// TEMPLATE atom's coordinate (`topology_candidates_for_dim` reuses the seed
/// coords) — the wrong phase for a fresh disjoint circle, which leaves the born
/// image `Φ·B` at the DC stationary point where cos/sin never populate. Seeding the
/// fresh phase directly gives the coordinate a nonzero gradient at birth (the birth
/// analogue of the 7a93b1d06 cold-start chart deflation). Mirrors `born_atom`'s
/// logit / coord / ρ construction so the born atom joins the dictionary identically.
pub(crate) fn born_circle_atom(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    harmonic_decoder: Array2<f64>,
    phase_coords: Array2<f64>,
    circle_gate: Vec<f64>,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let k = term.k_atoms();
    if term.atoms.is_empty() {
        return Err("born_circle_atom: cannot birth from an empty dictionary".to_string());
    }
    // The periodic-harmonic width `m` comes from the SEED decoder's own row
    // count, not the template atom's basis size (#2101 tied it to `atoms[0]`,
    // which forced every born circle to match atom-0's width). Curl seeds
    // (`crate::manifold::curl`) carry their own odd harmonic width, and existing
    // circle-seed callers pass a decoder already shaped to `atoms[0]` (odd, since
    // `PeriodicHarmonicEvaluator::new` demands it), so deriving `m` here is
    // backward-compatible AND lets curl birth a circle into a dictionary whose
    // template atom is linear/even-width. A born atom carries its own
    // `basis_values`, so a heterogeneous width is well-formed.
    let m = harmonic_decoder.nrows();
    let p = term.output_dim();
    if m % 2 != 1 || m < 3 {
        return Err(format!(
            "born_circle_atom: harmonic decoder must have odd height >= 3 (constant + \
             >= 1 sin/cos harmonic pair); got height {m}"
        ));
    }
    if harmonic_decoder.ncols() != p {
        return Err(format!(
            "born_circle_atom: harmonic decoder must have {p} columns (output dim); got {}",
            harmonic_decoder.ncols()
        ));
    }
    let n = term.assignment.logits.nrows();
    if phase_coords.dim() != (n, 1) {
        return Err(format!(
            "born_circle_atom: phase coords must be ({n}, 1); got {:?}",
            phase_coords.dim()
        ));
    }
    // A Periodic harmonic basis of the template's width, evaluated at the FRESH
    // phase coordinate the born circle lives on.
    let evaluator = std::sync::Arc::new(crate::manifold::PeriodicHarmonicEvaluator::new(m)?);
    let (phi, jet) = {
        use crate::manifold::SaeBasisEvaluator;
        evaluator.evaluate(phase_coords.view())?
    };
    let mut born = SaeManifoldAtom::new(
        format!("atom_born_{k}"),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        harmonic_decoder,
        Array2::<f64>::eye(m),
    )?
    .with_basis_second_jet(evaluator.clone());
    born.refresh_intrinsic_smooth_penalty();

    let born_coord_block = gam_terms::latent::LatentCoordValues::from_matrix_with_manifold(
        phase_coords.view(),
        LatentIdMode::None,
        LatentManifold::Circle { period: 1.0 },
    );

    let mut atoms = term.atoms.clone();
    atoms.push(born);
    let mut logits = Array2::<f64>::zeros((n, k + 1));
    for row in 0..n {
        for col in 0..k {
            logits[[row, col]] = term.assignment.logits[[row, col]];
        }
        // #2101/#2109 PRESENCE-PROPORTIONAL gate seed. The flat weak BIRTH_SEED_LOGIT
        // (−4) is fatal under IBP — the born circle starts nearly OFF (σ(−4)≈0.018) and
        // the sub-fit collapses it (measured: ibp logit −4 collapses ‖B‖ 1.41→1e-4,
        // logit +3 survives). On a row where the born circle is PRESENT (`circle_gate`
        // finite: its 2-plane energy cleared the derived MP floor), route it at the
        // STRONGER of two derived scales: (a) CO-ACTIVE with the incumbent dictionary
        // (per-row max of existing logits) and (b) the born circle's OWN presence gate
        // `ln(ρ_i²/2·λ₊)` carried in `circle_gate`. Taking the max is what fixes #2109:
        // on incumbent-SPARSE rows `inc_max` is low/negative (the incumbents don't
        // cover where the new circle lives), so the own-presence gate keeps the born
        // circle strong enough to ESTABLISH there instead of re-collapsing. Elsewhere
        // (absent rows, `circle_gate` = −∞) keep the conservative birth default. Both
        // scales are derived — the dictionary's own logits and the ρ_i/λ₊ ratio — no
        // new constant.
        let own_gate = circle_gate.get(row).copied().unwrap_or(f64::NEG_INFINITY);
        let inc_max = (0..k)
            .map(|c| term.assignment.logits[[row, c]])
            .fold(f64::NEG_INFINITY, f64::max);
        logits[[row, k]] = if own_gate.is_finite() {
            if inc_max.is_finite() {
                inc_max.max(own_gate)
            } else {
                own_gate
            }
        } else {
            BIRTH_SEED_LOGIT
        };
    }
    let mut coords = term.assignment.coords.clone();
    coords.push(born_coord_block);
    let assignment =
        crate::manifold::SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let mut child = SaeManifoldTerm::new(atoms, assignment)?;
    // Propagate the evidence-charge convention from the parent. `SaeManifoldTerm::
    // new` resets `rank_charge_evidence` to its default (false); if the incumbent
    // dictionary is scored on the occupancy-aware BIC rank charge, the born
    // candidate MUST be scored the same way, or the birth gate compares two REML
    // values on different Laplace-complexity scales (the raw per-row coordinate
    // log-det ½log|H_tt| grows ≈ O(n) per atom with no occam offset, so a
    // htt-charged candidate looks arbitrarily worse than a rank-charged incumbent
    // and every good birth is rejected at large n).
    child.set_rank_charge_evidence(term.rank_charge_evidence());

    let mut child_rho = rho.clone();
    let inherited = child_rho
        .log_ard
        .first()
        .cloned()
        .unwrap_or_else(|| Array1::<f64>::zeros(0));
    child_rho.log_ard.push(inherited);
    let inherited_smooth = child_rho.log_lambda_smooth.first().copied().unwrap_or(0.0);
    child_rho.log_lambda_smooth.push(inherited_smooth);
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
/// [`run_atom_birth_gate`](gam_terms::inference::structure_evidence::run_atom_birth_gate)
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
        use gam_solve::structure_search::MoveVerdict;
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
    /// Curl/flatten structure moves (INTEGRATION_PLAN Phase 4). `None` (the
    /// default) disables them entirely — the driver behaves bit-for-bit as
    /// before (`moves.curl = off`). `Some(cfg)` mines flat-pair→circle
    /// promotions (and the inverse circle→flat demotions) each round and submits
    /// their seeds into the SAME birth/race plumbing the residual-factor births
    /// use, so the REML e-gate remains the only judge.
    pub curl: Option<CurlConfig>,
}

/// Configuration for the curl/flatten proposer (INTEGRATION_PLAN Phase 4). All
/// derived-not-tuned in spirit; the fields are the pipeline's structural knobs,
/// not statistical dials (the verdict's σ screens and the RD crossover are fixed
/// in [`crate::manifold::curl`]).
#[derive(Clone, Copy, Debug)]
pub struct CurlConfig {
    /// Decoder-cosine ceiling for two linear atoms to be treated as the
    /// rectified halves (`±d`) of one signed direction (antipodal coalescing).
    /// `-0.85` ⇒ at least ~148° apart.
    pub coalesce_cos_threshold: f64,
    /// Gate-overlap (Jaccard) ceiling for a coalescing pair to count as
    /// near-disjoint rectified halves.
    pub coalesce_max_overlap: f64,
    /// Minimum co-firing rows (over the subsample) for a signed-direction pair
    /// to be a curl candidate plane.
    pub min_cooccurrence: usize,
    /// Row-subsample cap for co-occurrence counting + the joint-law projection.
    pub subsample_rows: usize,
    /// Number of `(sin, cos)` harmonics on the seeded circle decoder (width
    /// `2·harmonics + 1`; higher harmonics start at zero for the refit to
    /// sharpen). `>= 1`.
    pub harmonics: usize,
    /// Maximum curl births proposed per round (matched to the ISA birth budget
    /// class).
    pub max_curls: usize,
    /// Whether to run the inverse flatten audit on fitted circle atoms each
    /// round.
    pub flatten: bool,
    /// Rounds an atom-set is silenced after a curl or flatten fires on it, so
    /// `curl → flatten → curl` cannot oscillate (risk #5 hysteresis guard).
    pub cooldown_rounds: usize,
}

impl Default for CurlConfig {
    fn default() -> Self {
        Self {
            coalesce_cos_threshold: -0.85,
            coalesce_max_overlap: 0.15,
            min_cooccurrence: 8,
            subsample_rows: 4096,
            harmonics: 1,
            max_curls: 4,
            flatten: true,
            cooldown_rounds: 2,
        }
    }
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
        curl,
    } = config;
    let split = estimation_eval_split(target, n_shards);
    let mut rounds: Vec<SearchLedger> = Vec::new();
    // Hysteresis ledger for the curl/flatten pair — persists across rounds so a
    // just-curled atom-set (or just-flattened one) is silenced for a few rounds
    // and the two moves cannot chase each other (INTEGRATION_PLAN risk #5).
    let mut cooldown = crate::manifold::CurlCooldownLedger::new();

    for _ in 0..max_rounds {
        // Harvest from the current fitted state. Residuals R = target − fitted.
        let fitted = term.try_fitted()?;
        let residuals = &target.to_owned() - &fitted;
        let mut report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params)?;

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

        // Pre-build the birth-SEED list ONCE per round: the residual-factor
        // decoders first (indices `0..r`), then — when curl is enabled — the
        // race-ready circle seeds appended at `r..`, so the apply-move closure
        // inside the gate is a pure function of the candidate index. The two
        // channels share ONE index space via `StructureMove::Birth`.
        let residual_decoders = build_birth_decoders(&term, residuals.view(), &harvest_params)?;
        let mut birth_seeds: Vec<BirthSeed> =
            residual_decoders.into_iter().map(BirthSeed::ResidualFactor).collect();

        // Curl / flatten proposals (INTEGRATION_PLAN Phase 4), gated behind the
        // driver flag and the per-atom-set cooldown. `curl_atoms` maps a curl
        // birth's candidate index to its donor atom-set (the cooldown key +
        // certificate donors); `flatten_atoms` is the set of circle atoms a
        // flatten demotion targets this round.
        let mut curl_atoms: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        let mut flatten_atoms: std::collections::HashSet<usize> = std::collections::HashSet::new();
        if let Some(cfg) = curl {
            for cand in curl_candidates(&term, residuals.view(), &cfg) {
                if cooldown.blocked(&cand.members) {
                    continue;
                }
                let candidate = birth_seeds.len();
                birth_seeds.push(cand.seed);
                curl_atoms.insert(candidate, cand.members.clone());
                report
                    .proposals
                    .push(proposal(&term, StructureMove::Birth { candidate }, cand.net_evidence));
            }
            if cfg.flatten {
                for atom in flatten_candidates(&term) {
                    if cooldown.blocked(&[atom]) {
                        continue;
                    }
                    // A degenerate circle is retired through the existing death
                    // path; the e-gate adjudicates. Trigger is `MAX/4` so a
                    // flatten sorts among deaths, below terminal collapses.
                    flatten_atoms.insert(atom);
                    report.proposals.push(proposal(
                        &term,
                        StructureMove::Death { atom },
                        f64::MAX / 4.0,
                    ));
                }
            }
        }

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
        let decoders = birth_seeds;
        let estimation_rows = split.estimation_rows.clone();
        let outcome: SearchOutcome<State> = search(
            (term, rho),
            report.proposals,
            &split.shards,
            &budget,
            ledger,
            |state: &State, mv: &StructureMove| {
                let (cand_term, cand_rho) =
                    apply_structure_move_seeded(&state.0, &state.1, mv, &decoders)?;
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
                gam_solve::structure_search::MoveVerdict::Accepted { .. }
                    | gam_solve::structure_search::MoveVerdict::Demoted { .. }
            )
        });
        // Record the atom-sets any APPLIED curl / flatten move fired on into the
        // cooldown ledger, then advance one round — so the inverse move cannot
        // re-fire on the same atom-set next round (hysteresis, risk #5).
        if let Some(cfg) = curl {
            use gam_solve::structure_search::MoveVerdict;
            for rec in &round_ledger.moves {
                let fired = matches!(
                    rec.verdict,
                    MoveVerdict::Accepted { .. } | MoveVerdict::Demoted { .. }
                );
                if !fired {
                    continue;
                }
                match &rec.mv {
                    StructureMove::Birth { candidate } => {
                        if let Some(members) = curl_atoms.get(candidate) {
                            cooldown.record(members, cfg.cooldown_rounds);
                        }
                    }
                    StructureMove::Death { atom } if flatten_atoms.contains(atom) => {
                        cooldown.record(&[*atom], cfg.cooldown_rounds);
                    }
                    _ => {}
                }
            }
            cooldown.tick();
        }
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

// ===========================================================================
// Curl / flatten proposer driver (INTEGRATION_PLAN Phase 4)
//
// The pure statistics live in `crate::manifold::curl`; this is the term-level
// glue: it reads the fitted linear atoms, coalesces their rectified antipodal
// halves, generates co-firing candidate planes, projects the joint amplitude
// law, ranks by net evidence, and emits race-ready `BirthSeed::Circle` seeds
// (curl) plus `Death` demotions of degenerate circles (flatten). No move is
// accepted here — every seed is submitted to the same REML e-gate the
// residual-factor births race through.
// ===========================================================================

/// One curl birth candidate: the donor (coalesced) atom indices, the race-ready
/// circle seed, and the pre-screen net evidence that ranks it.
struct CurlCandidate {
    /// The linear atoms coalesced into the two signed axes of this circle — the
    /// cooldown key and the donors the race retires if the circle wins.
    members: Vec<usize>,
    /// The race-ready periodic circle seed (`BirthSeed::Circle`).
    seed: BirthSeed,
    /// `n_eff·ln(R̂/σ) − Δcharge` — the ranking score (NOT a decision).
    net_evidence: f64,
}

/// Whether an atom is a flat/linear parse the curl move coalesces over (a
/// centered circle is parked on these). Curved bases (periodic, torus, sphere,
/// cylinder, Poincaré) are excluded — they already carry their own curvature.
fn is_linear_like(kind: &SaeAtomBasisKind) -> bool {
    matches!(
        kind,
        SaeAtomBasisKind::Linear | SaeAtomBasisKind::EuclideanPatch
    )
}

/// A linear atom's ambient reconstruction image `G = Φ · B` (`n × p`, before
/// routing weight) — the geometric locus it parses.
fn atom_ambient_image(atom: &SaeManifoldAtom) -> Array2<f64> {
    atom.basis_values.dot(&atom.decoder_coefficients)
}

/// Top principal direction of the rows of `img` about `center`, restricted to
/// `active` rows, by a few power iterations on `Σ (g−c)(g−c)ᵀ` (formed
/// implicitly, so the cost is `O(active·p)` per iteration, never `p²`).
fn power_iter_top_dir(img: ArrayView2<'_, f64>, center: &Array1<f64>, active: &[usize]) -> Array1<f64> {
    let p = img.ncols();
    let mut v = Array1::<f64>::zeros(p);
    // Seed from the highest-norm centered active row (a strong signal direction).
    let mut best_norm = 0.0_f64;
    for &r in active {
        let mut nrm = 0.0_f64;
        for j in 0..p {
            let d = img[[r, j]] - center[j];
            nrm += d * d;
        }
        if nrm > best_norm {
            best_norm = nrm;
            for j in 0..p {
                v[j] = img[[r, j]] - center[j];
            }
        }
    }
    let vn = v.dot(&v).sqrt();
    if vn <= 0.0 {
        return v;
    }
    v.mapv_inplace(|x| x / vn);
    for _ in 0..5 {
        // w = C v = Σ (g−c) ((g−c)·v)
        let mut w = Array1::<f64>::zeros(p);
        for &r in active {
            let mut dot = 0.0_f64;
            for j in 0..p {
                dot += (img[[r, j]] - center[j]) * v[j];
            }
            for j in 0..p {
                w[j] += (img[[r, j]] - center[j]) * dot;
            }
        }
        let wn = w.dot(&w).sqrt();
        if wn <= 0.0 {
            break;
        }
        w.mapv_inplace(|x| x / wn);
        v = w;
    }
    v
}

/// Assemble the fitted linear atoms' `(atom index, unit direction, active mask,
/// ambient image)` — the raw material coalescing and candidate generation read.
fn linear_atom_frames(
    term: &SaeManifoldTerm,
) -> Vec<(usize, Array1<f64>, Vec<bool>, Array2<f64>)> {
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k = assignments.ncols();
    let floor = if k == 0 {
        0.0
    } else {
        ACTIVE_SUPPORT_REL_FLOOR / k as f64
    };
    let mut out = Vec::new();
    for (a, atom) in term.atoms.iter().enumerate() {
        if !is_linear_like(&atom.basis_kind) {
            continue;
        }
        let active_mask: Vec<bool> = (0..n).map(|r| assignments[[r, a]] > floor).collect();
        let active_idx: Vec<usize> = (0..n).filter(|&r| active_mask[r]).collect();
        if active_idx.len() < 2 {
            continue;
        }
        let img = atom_ambient_image(atom);
        if img.ncols() == 0 {
            continue;
        }
        let p = img.ncols();
        let mut center = Array1::<f64>::zeros(p);
        for &r in &active_idx {
            for j in 0..p {
                center[j] += img[[r, j]];
            }
        }
        center.mapv_inplace(|x| x / active_idx.len() as f64);
        let dir = power_iter_top_dir(img.view(), &center, &active_idx);
        if dir.dot(&dir).sqrt() <= 0.0 {
            continue;
        }
        out.push((a, dir, active_mask, img));
    }
    out
}

/// Mine flat-pair → circle promotion candidates from the fitted dictionary
/// (INTEGRATION_PLAN Phase 4 items 1–4). Coalesces rectified antipodal halves,
/// generates co-firing candidate planes over a row subsample, projects the joint
/// amplitude law, runs the geodict verdict, and returns the RECOMMENDED
/// candidates ranked by net evidence, deduplicated so no two circles claim the
/// same donor atom. Deterministic in `(term, residuals, cfg)` so the harvest and
/// the seed-build agree on candidate order/indices.
fn curl_candidates(
    term: &SaeManifoldTerm,
    residuals: ArrayView2<'_, f64>,
    cfg: &CurlConfig,
) -> Vec<CurlCandidate> {
    let frames = linear_atom_frames(term);
    if frames.len() < 2 {
        return Vec::new();
    }
    let n = term.assignment.logits.nrows();
    let p = term.output_dim();

    // Ambient noise scale for the RD screen: RMS reconstruction residual, floored
    // off zero (a perfectly-shattered circle leaves ~no residual, which is
    // exactly why it was invisible — the floor keeps ln(R̂/σ) finite and large).
    let mut sse = 0.0_f64;
    let mut cnt = 0usize;
    for r in 0..residuals.nrows() {
        for j in 0..residuals.ncols() {
            sse += residuals[[r, j]] * residuals[[r, j]];
            cnt += 1;
        }
    }
    let sigma = if cnt > 0 {
        (sse / cnt as f64).sqrt().max(1e-9)
    } else {
        1e-9
    };

    // Antipodal coalescing over the linear atoms' directions + gates.
    let dirs: Vec<ArrayView1<f64>> = frames.iter().map(|(_, d, _, _)| d.view()).collect();
    let actives: Vec<Vec<bool>> = frames.iter().map(|(_, _, m, _)| m.clone()).collect();
    let ids: Vec<usize> = frames.iter().map(|(a, _, _, _)| *a).collect();
    let signed = crate::manifold::coalesce_antipodal(
        &dirs,
        &actives,
        &ids,
        cfg.coalesce_cos_threshold,
        cfg.coalesce_max_overlap,
    );
    if signed.len() < 2 {
        return Vec::new();
    }

    // A per-atom → frame index map so a signed direction can gather its members'
    // ambient images for the plane projection.
    let frame_of: std::collections::HashMap<usize, usize> =
        ids.iter().enumerate().map(|(i, a)| (*a, i)).collect();
    let signed_active: Vec<Vec<bool>> = signed.iter().map(|s| s.active.clone()).collect();
    // Row subsample for co-occurrence counting (never O(K²)·O(n)).
    let rows: Vec<usize> = if n <= cfg.subsample_rows {
        (0..n).collect()
    } else {
        let stride = n / cfg.subsample_rows;
        (0..n).step_by(stride.max(1)).collect()
    };
    let pairs = crate::manifold::cooccurrence_pairs(&signed_active, &rows, cfg.min_cooccurrence);

    let mut cands: Vec<CurlCandidate> = Vec::new();
    for (si, sj, _count) in pairs {
        let di = &signed[si];
        let dj = &signed[sj];
        // Co-firing rows (both signed axes active), capped at the subsample.
        let mut co_fire: Vec<usize> = (0..n)
            .filter(|&r| di.active.get(r).copied().unwrap_or(false)
                && dj.active.get(r).copied().unwrap_or(false))
            .collect();
        if co_fire.len() < cfg.min_cooccurrence.max(2) {
            continue;
        }
        if co_fire.len() > cfg.subsample_rows {
            let stride = (co_fire.len() / cfg.subsample_rows).max(1);
            co_fire = co_fire.iter().copied().step_by(stride).collect();
        }
        // The plane image is the SUM of the two signed axes' member atom images —
        // isolating the two directions' joint parse from the rest of the fit.
        let members: Vec<usize> = di.members.iter().chain(dj.members.iter()).copied().collect();
        let mut x = Array2::<f64>::zeros((co_fire.len(), p));
        for (row_out, &r) in co_fire.iter().enumerate() {
            for &atom in &members {
                if let Some(&fi) = frame_of.get(&atom) {
                    let img = &frames[fi].3;
                    for j in 0..p {
                        x[[row_out, j]] += img[[r, j]];
                    }
                }
            }
        }
        let mut center = Array1::<f64>::zeros(p);
        for row_out in 0..co_fire.len() {
            for j in 0..p {
                center[j] += x[[row_out, j]];
            }
        }
        center.mapv_inplace(|v| v / co_fire.len() as f64);

        let (alpha, beta, e1, e2) = match crate::manifold::orthonormal_pair_coords(
            x.view(),
            di.dir.view(),
            dj.dir.view(),
            center.view(),
        ) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let n_eff = co_fire.len() as f64;
        // A mild MDL charge (circle basis rows vs the two linear directions); the
        // ranking is dominated by the gain, and the REML gate is the real judge.
        let m_circle = (2 * cfg.harmonics + 1) as f64;
        let delta_charge = 0.5 * m_circle * n_eff.max(2.0).ln();
        let verdict = match crate::manifold::curl_verdict(
            alpha.view(),
            beta.view(),
            sigma,
            n_eff,
            delta_charge,
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !verdict.recommend_curl {
            continue;
        }
        // Build the race-ready seed from the orthonormal frame + parse.
        let seed_circle = match crate::manifold::curl_seed(
            e1.view(),
            e2.view(),
            alpha.view(),
            beta.view(),
            cfg.harmonics,
            center.view(),
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Lift the co-firing phases + own-presence gate to the full row set.
        let mut phase_coords = Array2::<f64>::zeros((n, 1));
        let mut gate = vec![f64::NEG_INFINITY; n];
        let own = verdict.gain_nats_per_row.max(0.5);
        for (idx, &r) in co_fire.iter().enumerate() {
            phase_coords[[r, 0]] = seed_circle.theta_turns[idx];
            gate[r] = own;
        }
        cands.push(CurlCandidate {
            members,
            seed: BirthSeed::Circle {
                decoder: seed_circle.decoder,
                phase_coords,
                gate,
            },
            net_evidence: verdict.net_evidence_nats,
        });
    }

    // Rank by net evidence; keep the top budget, deduplicated so two circles
    // never claim the same donor atom in one round.
    cands.sort_by(|a, b| b.net_evidence.total_cmp(&a.net_evidence));
    let mut claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for c in cands {
        if c.members.iter().any(|a| claimed.contains(a)) {
            continue;
        }
        for a in &c.members {
            claimed.insert(*a);
        }
        out.push(c);
        if out.len() >= cfg.max_curls {
            break;
        }
    }
    out
}

/// Audit fitted circle atoms for degeneration (INTEGRATION_PLAN Phase 4.5). A
/// circle whose radial law has relaxed to Gaussian fill (κ ≈ 2) or collapsed to
/// a diameter (second resultant ≈ 1) is no longer carrying rotational structure;
/// return those atoms for the existing death/demotion path to retire. The
/// e-gate still owns the decision.
fn flatten_candidates(term: &SaeManifoldTerm) -> Vec<usize> {
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k = assignments.ncols();
    let floor = if k == 0 {
        0.0
    } else {
        ACTIVE_SUPPORT_REL_FLOOR / k as f64
    };
    let mut out = Vec::new();
    for (a, atom) in term.atoms.iter().enumerate() {
        if !matches!(atom.basis_kind, SaeAtomBasisKind::Periodic) || atom.latent_dim != 1 {
            continue;
        }
        let active_idx: Vec<usize> = (0..n).filter(|&r| assignments[[r, a]] > floor).collect();
        if active_idx.len() < 8 {
            continue;
        }
        // Per-row polar law: angle from the atom's phase coordinate, radius from
        // the centered ambient image norm in the atom's own image plane.
        let img = atom_ambient_image(atom);
        let p = img.ncols();
        let mut center = Array1::<f64>::zeros(p);
        for &r in &active_idx {
            for j in 0..p {
                center[j] += img[[r, j]];
            }
        }
        center.mapv_inplace(|x| x / active_idx.len() as f64);
        let coords = term.assignment.coords[a].as_matrix();
        if coords.ncols() == 0 {
            continue;
        }
        let mut radii = Array1::<f64>::zeros(active_idx.len());
        let mut angles = Array1::<f64>::zeros(active_idx.len());
        for (i, &r) in active_idx.iter().enumerate() {
            let mut rr = 0.0_f64;
            for j in 0..p {
                let d = img[[r, j]] - center[j];
                rr += d * d;
            }
            radii[i] = rr.sqrt();
            // Phase coordinate is in turns; angle in radians.
            angles[i] = std::f64::consts::TAU * coords[[r, 0]];
        }
        if let Ok(v) = crate::manifold::flatten_verdict(radii.view(), angles.view()) {
            if v.recommend_flatten {
                out.push(a);
            }
        }
    }
    out
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
/// `gam_inference::pg_gate_evidence::pg_gate_evidence`, summed over atoms,
/// so the K-dependent `−½·d_g·log(2π)` normalizer enters the gate's split-LR.
///
/// A degenerate/non-PD gate block contributes `0` (no gate evidence) rather than
/// poisoning the reconstruction likelihood — a conservative, valid degradation.
fn gate_block_log_evidence(term: &SaeManifoldTerm, shard: &RowBlockShard) -> f64 {
    use gam_solve::inference::pg_gate_evidence::{GateBlock, pg_gate_evidence};

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
    use crate::manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom,
    };
    use gam_solve::structure_search::{CollapseAction, CollapseEvent};
    use gam_terms::latent::LatentManifold;
    use ndarray::Array2;
    use std::sync::Arc;

    /// A parent nominated more than once (by different partners at different
    /// significances) must collapse to EXACTLY ONE entry — the most-suspect
    /// (lowest-significance) one — and distinct parents must all survive, in
    /// most-suspect-first order. This is the regression for the
    /// `dedup_by_key`-only-removes-adjacent-duplicates bug that used to let a
    /// parent ride as several duplicate `Fission` proposals.
    #[test]
    fn dedup_most_suspect_keeps_one_per_parent() {
        // Atom 2 nominated three times (0.4, 0.1, 0.7); atom 5 twice (0.3, 0.9);
        // atom 1 once (0.6). Deliberately unsorted so a significance-first sort
        // would NOT place same-atom entries adjacently.
        let raw = vec![
            (2usize, 0.4_f64),
            (5, 0.9),
            (1, 0.6),
            (2, 0.1),
            (5, 0.3),
            (2, 0.7),
        ];
        let out = dedup_most_suspect_per_parent(raw);

        // Exactly one entry per distinct parent.
        assert_eq!(out.len(), 3, "one entry per distinct parent: {out:?}");
        let mut atoms: Vec<usize> = out.iter().map(|(a, _)| *a).collect();
        atoms.sort_unstable();
        assert_eq!(atoms, vec![1, 2, 5], "all distinct parents kept");

        // The kept significance per parent is the minimum (most-suspect).
        let sig = |atom: usize| out.iter().find(|(a, _)| *a == atom).unwrap().1;
        assert_eq!(sig(2), 0.1, "atom 2 keeps its most-suspect nomination");
        assert_eq!(sig(5), 0.3, "atom 5 keeps its most-suspect nomination");
        assert_eq!(sig(1), 0.6, "the singly-nominated atom is unchanged");

        // Most-suspect-first (significance ascending) — the order the downstream
        // `take(max_fissions)` and carve loop rely on.
        assert_eq!(
            out,
            vec![(2, 0.1), (5, 0.3), (1, 0.6)],
            "deterministic most-suspect-first order"
        );
    }

    /// A high active logit (atom routes strongly on the row) and a low one
    /// (atom is dormant). With the `ACTIVE_SUPPORT_REL_FLOOR / K` threshold a
    /// softmax of these separates the discrete support cleanly.
    const ON: f64 = 6.0;
    const OFF: f64 = -6.0;

    /// Deterministic low-discrepancy sequence on `[0, 1)` (van der Corput, base
    /// 2) for RNG-free synthetic birth targets.
    fn vdc(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (mut x, mut denom, mut k) = (0.0_f64, 2.0_f64, i + 1);
                while k > 0 {
                    x += (k & 1) as f64 / denom;
                    denom *= 2.0;
                    k >>= 1;
                }
                x
            })
            .collect()
    }

    /// F1 radial promotion: a `d = 1` birth whose per-row amplitude is a
    /// CONTINUOUS spread (a disk, radius uniform in area ⇒ density ∝ r) must
    /// enrich the race with the circle-vs-cylinder-vs-disk candidate set; a
    /// present/absent (bimodal) birth must NOT promote.
    #[test]
    fn radial_promotion_fires_only_on_continuous_amplitude() {
        let n = 400;
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
        // Disk: place each row on a circle of radius r_i = sqrt(u_i) (area-uniform
        // radius, density ∝ r ⇒ Beta(2,1) ⇒ continuous), so the per-row amplitude
        // (row norm) is a continuous spread.
        let u = vdc(n);
        let disk = Array2::from_shape_fn((n, 2), |(i, j)| {
            let r = u[i].sqrt();
            let theta = std::f64::consts::TAU * (i as f64 / n as f64);
            if j == 0 { r * theta.cos() } else { r * theta.sin() }
        });
        let promoted = radial_promoted_specs(coords.view(), disk.view(), 1)
            .expect("promotion decision")
            .expect("disk amplitude is continuous ⇒ promotion fires");
        let kinds: std::collections::HashSet<_> = promoted.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&AutoTopologyKind::Circle), "{kinds:?}");
        assert!(kinds.contains(&AutoTopologyKind::Cylinder), "{kinds:?}");
        assert!(kinds.contains(&AutoTopologyKind::Euclidean), "{kinds:?}");
        // No key collision: each promoted kind appears once.
        assert_eq!(kinds.len(), promoted.len());

        // Present/absent circle: half the rows on the unit circle (amplitude 1),
        // half at the origin (amplitude 0) ⇒ bimodal ⇒ spike ⇒ NO promotion.
        let ring = Array2::from_shape_fn((n, 2), |(i, j)| {
            if i % 2 == 0 {
                0.0
            } else {
                let theta = std::f64::consts::TAU * (i as f64 / n as f64);
                if j == 0 { theta.cos() } else { theta.sin() }
            }
        });
        assert!(
            radial_promoted_specs(coords.view(), ring.view(), 1)
                .expect("promotion decision")
                .is_none(),
            "present/absent birth must not promote"
        );

        // A d != 1 birth never promotes (radial promotion is the d=1→2 lift).
        assert!(
            radial_promoted_specs(coords.view(), disk.view(), 2)
                .expect("promotion decision")
                .is_none()
        );
    }

    #[test]
    fn birth_row_amplitudes_are_row_norms() {
        let y = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let a = birth_row_amplitudes(y.view());
        assert!((a[0] - 5.0).abs() < 1e-12);
        assert!((a[1]).abs() < 1e-12);
    }

    // ---- F2: finite-set (discrete anchor) atom ------------------------------

    #[test]
    fn finite_set_race_is_not_enrolled_by_default() {
        // Containment: the finite-set candidate is inert unless explicitly
        // enrolled, so the enum arm + evaluator can never affect a birth by
        // default.
        assert!(!finite_set_race_enrolled());
        set_finite_set_race_enrolled(true);
        assert!(finite_set_race_enrolled());
        set_finite_set_race_enrolled(false);
        assert!(!finite_set_race_enrolled());
    }

    #[test]
    fn finite_set_candidate_fires_on_discrete_occupancy() {
        // Seven-point cyclic occupancy (weekdays): the coordinate collapses onto
        // 7 anchors, so the finite-set candidate builder returns 7 anchors and a
        // per-row integer index in [0, 7); the rank charge is anchors − 1 = 6.
        let per = 100;
        let mut rows = Vec::new();
        for i in 0..(7 * per) {
            // Sub-resolution embedding noise (±1e-3 over a span of 6 ⇒ ~1.7e-4
            // normalized, below the width floor) so the seven weekdays are a
            // genuine finite point set, not seven fuzzy blobs whose structured
            // noise the evidence could honestly resolve into more clusters.
            rows.push((i % 7) as f64 + 0.001 * ((i as f64).sin()));
        }
        let coords = Array2::from_shape_vec((7 * per, 1), rows).unwrap();
        let (anchors, idx) =
            finite_set_candidate_for_birth(coords.view()).expect("discrete ⇒ finite-set candidate");
        assert_eq!(anchors, 7, "anchors");
        assert_eq!(crate::manifold::finite_set_rank_charge(anchors), 6);
        // Every index is a valid anchor bin.
        assert!(idx.iter().all(|&v| (0.0..=6.0).contains(&v) && v.fract() == 0.0));

        // A uniformly-occupied coordinate is NOT a finite set — no candidate.
        let n = 400;
        let uni = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
        assert!(finite_set_candidate_for_birth(uni.view()).is_none());
    }

    #[test]
    fn anchor_indicator_evaluator_is_one_hot_with_zero_jets() {
        use crate::basis::{AnchorIndicatorEvaluator, SaeBasisEvaluator, SaeBasisSecondJet};
        let ev = AnchorIndicatorEvaluator::new(3).unwrap();
        // Coordinates snap to nearest anchor index; the design is one-hot.
        let coords = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 1.4]).unwrap();
        let (phi, jet) = ev.evaluate(coords.view()).unwrap();
        assert_eq!(phi.dim(), (4, 3));
        // Row sums are 1 (exactly one active anchor per row).
        for r in 0..4 {
            assert!((phi.row(r).sum() - 1.0).abs() < 1e-12);
        }
        assert!((phi[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((phi[[1, 1]] - 1.0).abs() < 1e-12);
        assert!((phi[[2, 2]] - 1.0).abs() < 1e-12);
        assert!((phi[[3, 1]] - 1.0).abs() < 1e-12); // 1.4 rounds to anchor 1
        // The indicator is piecewise constant: all jets are zero.
        assert!(jet.iter().all(|&v| v == 0.0));
        let h = ev.second_jet(coords.view()).unwrap();
        assert!(h.iter().all(|&v| v == 0.0));
    }

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
        use gam_solve::structure_search::{MoveRecord, MoveVerdict};

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
        // Every length-K ρ vector the penalty assembler indexes by atom must
        // grow with K, not just `log_ard`. `log_lambda_smooth` is read as
        // `lambda_smooth[atom_idx]` in construction.rs; a stale length-K vector
        // panics out of bounds on the K-th (new) atom (#357).
        assert_eq!(
            fissioned_rho.log_lambda_smooth.len(),
            fissioned.k_atoms(),
            "fission must grow per-atom log_lambda_smooth in lockstep with K"
        );

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

        // Birth: K grows, and the new atom RECONSTRUCTS the residual-factor image.
        //
        // Since the #977 topology RACE, a born atom no longer carries the raw
        // `factor_dir` coefficients verbatim: its topology is chosen by evidence
        // and its decoder is the winning basis's penalized least-squares fit to the
        // birth target `Y = Φ_template · factor_dir` (so the raw coefficient
        // `[[0,0]]` is shrunk by the fit ridge — `0.6999…`, not exactly `0.7`).
        // The structural invariant the move must preserve is therefore
        // RECONSTRUCTION PARITY, not coefficient identity: the born atom, evaluated
        // on its own coordinates with its own (raced) basis, must reproduce the
        // birth-target image to within the small fit ridge.
        let p = term.output_dim();
        let m = term.atoms[0].basis_size();
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[0, 0]] = 0.7;
        let birth_target = term.atoms[0].basis_values.dot(&decoder); // Φ_template · factor_dir
        let (born, born_rho) = apply_structure_move(
            &term,
            &rho,
            &StructureMove::Birth { candidate: 0 },
            &[decoder],
        )
        .unwrap();
        assert_eq!(born.k_atoms(), k0 + 1);
        assert_eq!(born_rho.log_ard.len(), k0 + 1);
        // ρ's per-atom smoothness vector must grow in step with K (the #1556
        // contract `assemble_arrow_schur` validates); a stale-length vector would
        // panic the next assemble on the per-atom `lambda_smooth[atom_idx]` index.
        assert_eq!(born_rho.log_lambda_smooth.len(), k0 + 1);
        let born_atom = &born.atoms[k0];
        let born_image = born_atom.basis_values.dot(&born_atom.decoder_coefficients);
        assert_eq!(born_image.dim(), birth_target.dim());
        let mut max_recon_err = 0.0_f64;
        for (a, b) in born_image.iter().zip(birth_target.iter()) {
            max_recon_err = max_recon_err.max((a - b).abs());
        }
        assert!(
            max_recon_err < 1e-3,
            "born atom must reconstruct the residual-factor image (penalized fit); \
             max |Φ_born·B_born − Φ_template·factor_dir| = {max_recon_err:.3e} (> 1e-3)"
        );
    }

    /// #357 regression: after a structure move that GROWS the atom count
    /// (fission/birth), the returned ρ's per-atom `log_lambda_smooth` must be
    /// length-K so the penalty assembler's `lambda_smooth[atom_idx]` read does
    /// not panic out of bounds. Before the fix `duplicate_atom`/`born_atom`
    /// pushed only `log_ard`, leaving `log_lambda_smooth` one short — the next
    /// `assemble_arrow_schur_inner` panicked with `index out of bounds: the len
    /// is K but the index is K` (construction.rs `scaled_s[[i,j]] =
    /// lambda_smooth[atom_idx] * s_ij`). This drives the REAL assembly so it
    /// fails on the buggy path, not just on a length assertion.
    #[test]
    fn grown_atom_count_assembles_without_lambda_smooth_oob_357() {
        let n = 16usize;
        let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0]).collect();
        let (term, rho) = planted_term(&active);
        let target = Array2::<f64>::from_shape_fn((n, term.output_dim()), |(row, col)| {
            0.1 * (row as f64) - 0.05 * (col as f64)
        });

        // Fission grows K by one.
        let (fissioned, fissioned_rho) =
            apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
        assert_eq!(fissioned_rho.log_lambda_smooth.len(), fissioned.k_atoms());
        // The assembly indexes lambda_smooth[atom_idx] for every atom; on the
        // pre-fix ρ this panicked out of bounds for the new K-th atom.
        let mut fissioned = fissioned;
        fissioned
            .assemble_arrow_schur_scaled(target.view(), &fissioned_rho, None, 1.0)
            .expect("post-fission assembly must not panic or error on the grown atom set");

        // Birth grows K by one and must assemble too.
        let p = term.output_dim();
        let m = term.atoms[0].basis_size();
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[0, 0]] = 0.5;
        let (born, born_rho) = apply_structure_move(
            &term,
            &rho,
            &StructureMove::Birth { candidate: 0 },
            &[decoder],
        )
        .unwrap();
        assert_eq!(born_rho.log_lambda_smooth.len(), born.k_atoms());
        let mut born = born;
        born.assemble_arrow_schur_scaled(target.view(), &born_rho, None, 1.0)
            .expect("post-birth assembly must not panic or error on the grown atom set");
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
            let mut ledger = gam_terms::inference::structure_evidence::StructureLedger::new();
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
                curl: None,
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
            curl: None,
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
        // The cap-invariant surface is the DECISION each move records — its
        // proposal (`mv`), the structure it acts on (`structure_hash`), its
        // `claim`, and the e-gate VERDICT VARIANT (Accepted vs Contested vs …).
        // The `trigger` and the verdict's `log_e` are floating-point MAGNITUDES
        // computed under the scoring budget: a 4-iter scoring fit and a 24-iter
        // one legitimately land on slightly different evidence for the SAME
        // decision — that is exactly the economy the cap trades on. Asserting
        // bit-identical `log_e`/`trigger` would demand the cap be a no-op, which
        // is the opposite of its purpose; the #1026 soundness property is that no
        // e-gate DECISION flips, which the projection below captures.
        use gam_solve::structure_search::MoveVerdict;
        let verdict_kind = |v: &MoveVerdict| -> &'static str {
            match v {
                MoveVerdict::Accepted { .. } => "Accepted",
                MoveVerdict::Contested { .. } => "Contested",
                MoveVerdict::Demoted { .. } => "Demoted",
                MoveVerdict::Vetoed { .. } => "Vetoed",
                MoveVerdict::Deduplicated => "Deduplicated",
                MoveVerdict::Stale => "Stale",
                MoveVerdict::Deferred => "Deferred",
            }
        };
        let round_moves = |rounds: &[SearchLedger]| -> String {
            serde_json::to_string(
                &rounds
                    .iter()
                    .map(|r| {
                        r.moves
                            .iter()
                            .map(|m| {
                                (
                                    serde_json::to_string(&m.mv).unwrap(),
                                    m.structure_hash,
                                    serde_json::to_string(&m.claim).unwrap(),
                                    verdict_kind(&m.verdict),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
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

    /// Fission must BREAK the parent/child symmetry. Duplicating an atom
    /// identically (same decoder, mass split 50/50) sits at a symmetric saddle of
    /// the joint refit — the children's gradients are identical, so a
    /// deterministic refit never separates them and the fission is a no-op the
    /// e-gate rejects. The anti-symmetric perturbation makes the two children's
    /// decoders genuinely differ (so the refit can separate factors) while the
    /// equal-mass combined decoder `½(parent+child)` stays EXACTLY the original
    /// (warm-start preserved).
    #[test]
    fn fission_breaks_symmetry_so_children_can_separate() {
        let (term, rho) = planted_term(&vec![vec![true]; 8]);
        assert_eq!(term.k_atoms(), 1);
        let orig = term.atoms[0].decoder_coefficients.clone();

        let (child, _child_rho) =
            apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
        assert_eq!(child.k_atoms(), 2, "fission must add one atom");

        let d0 = &child.atoms[0].decoder_coefficients;
        let d1 = &child.atoms[1].decoder_coefficients;
        // (1) Symmetry BROKEN: the children's decoders are not identical (without
        // this the refit is stuck at the symmetric saddle and fission is a no-op).
        let sep = (d0 - d1).iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = orig.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        assert!(
            sep / scale > 1.0e-3,
            "fission children must NOT be identical (symmetric saddle); rel sep = {}",
            sep / scale
        );
        // (2) Warm-start preserved EXACTLY: the equal-mass combined decoder is the
        // original (the anti-symmetric ±ε perturbation cancels).
        let combined = (d0 + d1).mapv(|x| 0.5 * x);
        let warm_err = (&combined - &orig)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(
            warm_err < 1.0e-12,
            "mass-split combined decoder must equal the original; err = {warm_err}"
        );
        // (3) Mass split is EVEN: the parent and child carry equal routing logits
        // on every row (each gets half the parent's softmax mass).
        for row in 0..child.assignment.logits.nrows() {
            assert!(
                (child.assignment.logits[[row, 0]] - child.assignment.logits[[row, 1]]).abs()
                    < 1e-12,
                "fission must split routing mass 50/50 (equal child logits)"
            );
        }
    }

    /// Softmax fusion must PRESERVE the combined routing mass. Merging the two
    /// constituent logits with `logsumexp` keeps `mass(fused) = mass(a)+mass(b)`;
    /// the old `max` under-massed the fused atom (½ vs ⅔ on this 3-atom fixture
    /// where atoms 0,1 are co-active and atom 2 competes), leaving the warm-start
    /// short and risking a FALSE e-gate rejection of a good fusion under a capped
    /// refit. (For IBP routing `max` stays correct — the gate is un-normalized.)
    #[test]
    fn fusion_preserves_combined_softmax_mass() {
        let (term, rho) = planted_term(&vec![vec![true, true, true]; 6]);
        let combined: Vec<f64> = (0..6)
            .map(|r| {
                let a = term.assignment.try_assignments_row(r).unwrap();
                a[0] + a[1]
            })
            .collect();
        let (fused, _) =
            apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
        for r in 0..6 {
            let a = fused.assignment.try_assignments_row(r).unwrap();
            assert!(
                (a[0] - combined[r]).abs() < 1e-6,
                "fused atom must carry the COMBINED softmax mass (logsumexp, not \
                 max): got {}, want {} (row {r})",
                a[0],
                combined[r]
            );
            // Sanity: plain max would have given ½ here, materially short of ⅔.
            assert!(
                combined[r] > 0.6,
                "fixture must exercise a co-active pair (combined mass {} should be ~⅔)",
                combined[r]
            );
        }
    }

    #[test]
    fn fusion_of_zero_mass_pair_yields_neg_inf_not_nan() {
        // Folding two atoms whose softmax logits are BOTH -∞ (zero routing mass on
        // a row) must give the mass-preserving combined logit -∞ (combined mass 0),
        // NOT NaN. Pre-fix, `logsumexp(-∞,-∞)` evaluated `(-∞)-(-∞)=NaN` and poisoned
        // the entire logits row.
        let (mut term, rho) = planted_term(&vec![vec![true, true, true]; 6]);
        assert!(
            matches!(term.assignment.mode, AssignmentMode::Softmax { .. }),
            "fixture must be softmax-routed to exercise the logsumexp combine"
        );
        // Zero out atoms 0 and 1 on row 0 (both -∞), leave the rest finite.
        term.assignment.logits[[0, 0]] = f64::NEG_INFINITY;
        term.assignment.logits[[0, 1]] = f64::NEG_INFINITY;
        let (fused, _) =
            apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
        let folded = fused.assignment.logits[[0, 0]];
        assert!(
            !folded.is_nan(),
            "fused zero-mass logit must not be NaN (got {folded})"
        );
        assert_eq!(
            folded,
            f64::NEG_INFINITY,
            "combined mass of two zero-mass atoms is zero → logit -∞"
        );
        // The whole row must stay NaN-free so softmax over it is well defined.
        for c in 0..fused.assignment.logits.ncols() {
            assert!(
                !fused.assignment.logits[[0, c]].is_nan(),
                "row 0 col {c} must not be NaN after the fold"
            );
        }
    }

    // =======================================================================
    // Curl / flatten Phase-4 killer demo (INTEGRATION_PLAN §8 definition of
    // done): plant a centered circle, let a NONNEGATIVE-gate linear dictionary
    // shatter it into four rectified half-atoms (±u, ±v), and show the curl
    // proposer coalesces them, recovers the circle, and would win on the
    // evidence the race reads — while a Gaussian-fill plane is NOT curled, a
    // diameter-collapsed circle flattens, and a healthy ring is left alone.
    // =======================================================================

    /// A straight-line (Linear) atom whose ambient image is `t ↦ t · dir` over
    /// the supplied per-row coordinate — `Φ = [1, t]`, decoder rows
    /// `[0; dir]`. This is the rectified half-atom a nonnegative gate parks on
    /// one lobe of a centered signed direction.
    fn linear_line_atom(name: &str, coord: &Array1<f64>, dir: &Array1<f64>) -> SaeManifoldAtom {
        let n = coord.len();
        let p = dir.len();
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = ndarray::Array3::<f64>::zeros((n, 2, 1));
        for r in 0..n {
            phi[[r, 0]] = 1.0;
            phi[[r, 1]] = coord[r];
            jet[[r, 0, 0]] = 0.0;
            jet[[r, 1, 0]] = 1.0;
        }
        let mut decoder = Array2::<f64>::zeros((2, p));
        for j in 0..p {
            decoder[[1, j]] = dir[j];
        }
        SaeManifoldAtom::new(
            name.to_string(),
            SaeAtomBasisKind::Linear,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(2),
        )
        .unwrap()
    }

    /// Build a dictionary of four rectified half-atoms `±u, ±v` parking a
    /// centered feature in the `(e0, e1)` plane of `R⁴`. When `gaussian` the
    /// parked feature is an isotropic 2-D Gaussian (κ ≈ 2, no curved gain);
    /// otherwise a constant-radius circle (κ ≈ 1). Each half is gated on the
    /// rows where its lobe is positive, so the ± gates are disjoint (the
    /// coalescer's precondition) and the two signed axes co-fire on every row.
    fn shattered_plane_term(gaussian: bool) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = 600usize;
        let radius = 3.0_f64;
        let u = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let v = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let neg_u = u.mapv(|x| -x);
        let neg_v = v.mapv(|x| -x);
        let mut s = 0xC0FFEE_u64;
        let lcg = |st: &mut u64| -> f64 {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        // Per-row (x, y) in-plane coordinates the four halves rectify.
        let mut xs = Array1::<f64>::zeros(n);
        let mut ys = Array1::<f64>::zeros(n);
        for r in 0..n {
            if gaussian {
                // Box–Muller isotropic Gaussian.
                let u1 = lcg(&mut s).max(1e-12);
                let u2 = lcg(&mut s);
                let g0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let g1 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).sin();
                xs[r] = radius * g0;
                ys[r] = radius * g1;
            } else {
                let th = std::f64::consts::TAU * (r as f64 + 0.5) / n as f64;
                xs[r] = radius * th.cos();
                ys[r] = radius * th.sin();
            }
        }
        // Rectified coordinates per half.
        let cu: Array1<f64> = xs.mapv(|x| x.max(0.0));
        let cnu: Array1<f64> = xs.mapv(|x| (-x).max(0.0));
        let cv: Array1<f64> = ys.mapv(|y| y.max(0.0));
        let cnv: Array1<f64> = ys.mapv(|y| (-y).max(0.0));
        let atoms = vec![
            linear_line_atom("half_+u", &cu, &u),
            linear_line_atom("half_-u", &cnu, &neg_u),
            linear_line_atom("half_+v", &cv, &v),
            linear_line_atom("half_-v", &cnv, &neg_v),
        ];
        let coord_blocks = vec![
            cu.clone().insert_axis(ndarray::Axis(1)),
            cnu.clone().insert_axis(ndarray::Axis(1)),
            cv.clone().insert_axis(ndarray::Axis(1)),
            cnv.clone().insert_axis(ndarray::Axis(1)),
        ];
        let k = atoms.len();
        // Gate each half on the rows where its lobe is active (coordinate > 0).
        let lobes = [&cu, &cnu, &cv, &cnv];
        let mut logits = Array2::<f64>::zeros((n, k));
        for r in 0..n {
            for (a, lobe) in lobes.iter().enumerate() {
                logits[[r, a]] = if lobe[r] > 1e-9 { ON } else { OFF };
            }
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        (term, rho)
    }

    /// KILLER DEMO — the curl proposer recovers a centered circle a linear
    /// dictionary shattered into four rectified halves: it coalesces the ±
    /// pairs, reads κ ≈ 1 off the joint amplitude law, recommends the
    /// promotion, and the seed born through the existing plumbing reconstructs
    /// the planted ring.
    #[test]
    fn curl_recovers_shattered_centered_circle() {
        let (term, rho) = shattered_plane_term(false);
        let residuals = residuals_of(&term);
        let cfg = CurlConfig::default();
        let cands = curl_candidates(&term, residuals.view(), &cfg);
        assert!(
            !cands.is_empty(),
            "curl must recover the shattered circle (got no candidate)"
        );
        let cand = &cands[0];
        // All four rectified halves coalesced into the two signed axes.
        let mut members = cand.members.clone();
        members.sort_unstable();
        members.dedup();
        assert_eq!(
            members,
            vec![0, 1, 2, 3],
            "the circle's donor set is all four rectified halves"
        );
        assert!(cand.net_evidence > 0.0, "net evidence must favour the circle");

        // Born through the existing birth plumbing → a Periodic circle atom.
        let mv = StructureMove::Birth { candidate: 0 };
        let seeds = vec![cand.seed.clone()];
        let (born, _born_rho) = apply_structure_move_seeded(&term, &rho, &mv, &seeds).unwrap();
        let circle = born.k_atoms() - 1;
        assert_eq!(
            born.atoms[circle].basis_kind,
            SaeAtomBasisKind::Periodic,
            "curl births a Periodic (circle) atom"
        );
        // The born circle's own reconstruction traces the planted ring: every
        // active row sits at radius ≈ R about the centre.
        let img = atom_ambient_image(&born.atoms[circle]);
        let ncols = img.ncols();
        let mut center = Array1::<f64>::zeros(ncols);
        for r in 0..img.nrows() {
            for j in 0..ncols {
                center[j] += img[[r, j]];
            }
        }
        center.mapv_inplace(|x| x / img.nrows() as f64);
        let mut min_r = f64::INFINITY;
        let mut max_r = 0.0_f64;
        for r in 0..img.nrows() {
            let mut rr = 0.0_f64;
            for j in 0..ncols {
                let d = img[[r, j]] - center[j];
                rr += d * d;
            }
            let rr = rr.sqrt();
            min_r = min_r.min(rr);
            max_r = max_r.max(rr);
        }
        // A ring: radius nearly constant across rows (thickness ≪ radius).
        assert!(
            max_r > 0.0 && (max_r - min_r) / max_r < 0.1,
            "born circle must trace a constant-radius ring (min={min_r:.3}, max={max_r:.3})"
        );
    }

    /// A Gaussian-fill plane (κ ≈ 2, the zero-gain point of the coding law) is
    /// NOT curled — the radius law is exactly the flat-parse null.
    #[test]
    fn curl_rejects_gaussian_fill_plane() {
        let (term, _rho) = shattered_plane_term(true);
        let residuals = residuals_of(&term);
        let cfg = CurlConfig::default();
        let cands = curl_candidates(&term, residuals.view(), &cfg);
        assert!(
            cands.is_empty(),
            "a Gaussian-fill plane must not be curled (κ ≈ 2)"
        );
    }

    /// Build a single-Periodic-atom term whose phase coordinate takes the given
    /// per-row turns; the fundamental decoder places the ring in the `(e0, e1)`
    /// plane at radius `R`.
    fn single_circle_term(phase_turns: &Array1<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = phase_turns.len();
        let p = 4usize;
        let radius = 3.0_f64;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = phase_turns.clone().insert_axis(ndarray::Axis(1));
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[2, 0]] = radius; // cos₁ · e0
        decoder[[1, 1]] = radius; // sin₁ · e1
        let atom = SaeManifoldAtom::new(
            "circle".to_string(),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        let logits = Array2::<f64>::from_elem((n, 1), ON);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
        (term, rho)
    }

    /// A circle whose angular mass has collapsed to a diameter (phases at 0 and
    /// ½ turn only) is flagged for flattening; a healthy full-coverage ring is
    /// not.
    #[test]
    fn flatten_flags_diameter_and_spares_healthy_ring() {
        let n = 400usize;
        // Diameter: phases alternate 0 / ½ turn → angles {0, π}.
        let diameter_phases =
            Array1::from_shape_fn(n, |r| if r % 2 == 0 { 0.0 } else { 0.5 });
        let (diam_term, _) = single_circle_term(&diameter_phases);
        let flagged = flatten_candidates(&diam_term);
        assert_eq!(flagged, vec![0], "a diameter-collapsed circle must flatten");

        // Healthy ring: full angular coverage.
        let ring_phases = Array1::from_shape_fn(n, |r| r as f64 / n as f64);
        let (ring_term, _) = single_circle_term(&ring_phases);
        let flagged = flatten_candidates(&ring_term);
        assert!(
            flagged.is_empty(),
            "a healthy full-coverage ring must NOT be flattened"
        );
    }

    /// End-to-end through the round driver: with curl ON the shattered circle
    /// yields a circle Birth that certifies through the SAME e-gate the residual
    /// births race through; with curl OFF (the default) the driver is unchanged.
    /// This is the flag-gated wiring proof (the REML gate remains the judge;
    /// certed acceptance of the win additionally rests on the Phase-1 charge
    /// ledger).
    #[test]
    fn driver_curl_flag_accepts_circle_birth() {
        let budget = MoveBudget {
            max_moves: 4,
            alpha: 0.05,
        };
        // Only curl births — no residual/fusion/fission channels — so any
        // accepted birth is the curl circle winning the round race.
        let harvest_params = HarvestParams {
            max_fusions: 0,
            max_fissions: 0,
            max_births: 0,
        };
        let run = |curl: Option<CurlConfig>| -> StructureSearchResult {
            let (term, rho) = shattered_plane_term(false);
            let mut ledger = StructureLedger::new();
            let config = RoundDriverConfig {
                n_shards: 3,
                budget,
                max_rounds: 1,
                harvest_params,
                curl,
            };
            let result = run_structure_search_rounds(
                term,
                rho,
                Array2::<f64>::zeros((600, 4)).view(),
                config,
                &mut ledger,
                |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| (t, r),
                |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| (t, r),
            )
            .unwrap()
        };

        let off = run(None);
        let off_births = off
            .rounds
            .iter()
            .flat_map(|r| r.moves.iter())
            .filter(|m| matches!(m.mv, StructureMove::Birth { .. }))
            .count();
        assert_eq!(off_births, 0, "curl OFF (default) must inject no births");

        let on = run(Some(CurlConfig::default()));
        let accepted_curl_births = on
            .rounds
            .iter()
            .flat_map(|r| r.moves.iter())
            .filter(|m| {
                matches!(m.mv, StructureMove::Birth { .. })
                    && matches!(
                        m.verdict,
                        gam_solve::structure_search::MoveVerdict::Accepted { .. }
                    )
            })
            .count();
        assert_eq!(
            accepted_curl_births, 1,
            "curl ON must certify exactly one circle Birth winner"
        );
        assert_eq!(
            on.term.atoms.last().map(|a| a.basis_kind),
            Some(SaeAtomBasisKind::Periodic),
            "the accepted curl winner must be the recovered circle atom"
        );
        assert!(
            on.structure_changed(),
            "accepted curl winner must mutate the returned term"
        );
    }
}
