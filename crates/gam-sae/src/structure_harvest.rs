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
//! curvature evidence — the same declared reference-function scale the
//! smooth-term topology race uses — pick the winner. A curved winner is not
//! merely a nicer-looking basis: it is the SPECIFIC configuration whose
//! rigidity is what makes the born atom identifiable at all. The race is
//! therefore the optimizer's equilibrium response to superposition, not a
//! stylistic preference for circles.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::atom_codes::SparseAtomCodes;
use crate::basis::{
    CylinderHarmonicEvaluator, DuchonCoordinateEvaluator, EuclideanPatchEvaluator,
    MobiusHarmonicEvaluator, PeriodicHarmonicEvaluator, SaeBasisSecondJet, SphereChartEvaluator,
    TorusHarmonicEvaluator,
};
use crate::description_length::{BirthMdlPrescreen, predicted_birth_dl_bits};
use crate::frames::GrassmannFrame;
use crate::manifold::{
    AssignmentMode, AtlasSeamKind, GraphStructureSelection, LearnedGraphAtom, OccupancyLaw,
    SAE_MAX_PERIODIC_HARMONICS, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    SaeReferenceRoughness, SphereChartTransition, UnitSpeedChartTransition,
    amplitude_concentration_certificate, classify_occupancy_interval,
};
use crate::migration_ledger::SaeMigrationLedger;
use crate::null_sampler::{NULL_REPLICATES, coactivation_exceedance_for_pairs};
use gam_linalg::faer_ndarray::FaerSvd;
use gam_runtime::warm_start::Fingerprinter;
use gam_solve::gaussian_reml::gaussian_reml_multi_closed_form;
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::structure_search::{
    ChartGlueOutcome, CollapseAction, MoveBudget, MoveProposal, SearchLedger, SearchOutcome,
    StructureMove, search,
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
use std::sync::atomic::{AtomicBool, Ordering};

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
/// outer log-strength domain endpoint.
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

/// Participation ratio `(Σλ)²/Σλ²` of a non-negative spectrum — the effective
/// number of significant directions (the #2233 span estimate `ŝ` when the spectrum
/// is the residual factor-energy set). `1.0` for a single direction (or an
/// all-but-one-zero spectrum); `0.0` for an empty / all-zero spectrum.
fn participation_ratio(spectrum: &[f64]) -> f64 {
    let sum: f64 = spectrum.iter().map(|&e| e.max(0.0)).sum();
    let sum_sq: f64 = spectrum.iter().map(|&e| e.max(0.0) * e.max(0.0)).sum();
    if sum_sq > 0.0 {
        (sum * sum) / sum_sq
    } else {
        0.0
    }
}

/// The curved topology `(d, m)` the #2233 pre-screen matches to an estimated
/// ambient span `ŝ`, so the dictionary surcharge is priced against the realizable
/// curved atom a span-`ŝ` residual would be raced into. The basis budgets mirror
/// [`topology_candidates_for_dim`] (the downstream birth topology race), not a new
/// menu: circle `2·1+1 = 3`, sphere chart `7`, torus `(2·2+1)² = 25`. The curved
/// families top out at `d = 2`, so a span `≥ 4` residual is priced against the
/// richest curved atom (the torus); the e-gate, never this map, owns acceptance.
fn curved_topology_for_span(span: f64) -> (usize, usize) {
    match span.round().max(1.0) as usize {
        0 | 1 | 2 => (1, 3), // circle (PeriodicHarmonicEvaluator, 2·d+1 harmonics)
        3 => (2, 7),         // sphere chart (SphereChartEvaluator)
        _ => (2, 25),        // torus (TorusHarmonicEvaluator, (2H+1)² at H=2)
    }
}

/// Mean active atoms per token `L0` — the support-budget denominator for the
/// #2233 pre-screen's `log₂(G/L0)` term. An atom counts as active on a row by the
/// SAME `ACTIVE_SUPPORT_REL_FLOOR / K` discrete-support threshold
/// [`sparse_codes_from_term`] uses (no new constant), floored at `1.0` so the
/// support term is well-defined even on a degenerate all-inactive round.
fn mean_active_atoms(assignments: ArrayView2<'_, f64>) -> f64 {
    let n = assignments.nrows();
    let k = assignments.ncols();
    if n == 0 || k == 0 {
        return 1.0;
    }
    let floor = ACTIVE_SUPPORT_REL_FLOOR / k as f64;
    let mut total_active = 0usize;
    for row in 0..n {
        for atom in 0..k {
            if assignments[[row, atom]] > floor {
                total_active += 1;
            }
        }
    }
    (total_active as f64 / n as f64).max(1.0)
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
        StructureMove::Glue { a, b, outcome } => {
            // A fuse reaches the same physical dictionary shape as Fusion and
            // deliberately shares its tag.  Atlas registration is a different
            // post-move structure: K local charts stay, while the pair becomes
            // one semantic atom with a persisted transition cocycle.
            fp.write_str(match outcome {
                ChartGlueOutcome::Fuse => "fusion",
                ChartGlueOutcome::RegisterAtlas => "atlas_register",
            });
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
        SaeAtomBasisKind::Mobius => "mobius",
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
        // #1890: the seam-glue claim is its own ledger entry, distinct from the
        // fusion `BindingEdge` on the same pair (glue asserts "these two charts
        // tile ONE manifold within an isometry tolerance", a strictly stronger
        // claim than "these two atoms co-fire and bind"). Carried as a labeled
        // `Custom` — an ordered `(min,max)` key so `Glue{a,b}` and `Glue{b,a}`
        // dedup — rather than a new `ClaimKind` variant, to avoid a cross-crate
        // enum change whose exhaustive-match fallout the design does not need.
        StructureMove::Glue { a, b, .. } => ClaimKind::Custom {
            // Atom indices are only stable within one dictionary epoch.  A
            // certified glue physically compacts the atom columns at the round
            // boundary, so the next round's `(0, 1)` can denote a DIFFERENT
            // pair than this round's `(0, 1)`.  Scope the running e-process by
            // the proposal's stamped structural hash (which includes the
            // current dictionary skeleton): contested seams in an unchanged
            // dictionary still resume their evidence, while a compaction can
            // never lend already-certified evidence to a newly re-indexed pair.
            label: format!(
                "seam_glue:{structure_hash:016x}:{}:{}",
                (*a).min(*b),
                (*a).max(*b)
            ),
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
    let coactive_pairs = if want_coactivation {
        codes.coactive_pair_stats()
    } else {
        Vec::new()
    };
    let coactive_pair_keys: Vec<(usize, usize)> =
        coactive_pairs.iter().map(|(a, b, _)| (*a, *b)).collect();
    let exceedance_z = if want_coactivation {
        coactivation_exceedance_for_pairs(&codes, &coactive_pair_keys, NULL_REPLICATES)
    } else {
        Vec::new()
    };
    let z_floor = null_exceedance_z_floor();

    // --- Fusions: top co-activation dependence, gated by the null ----------
    // The trigger REPORTED is the null exceedance `z` (above-margin co-firing),
    // not the raw dependence: the raw floor only pre-selects genuinely co-firing
    // pairs, and the fixed-margin null strips the mechanical top-`k` coupling.
    let mut fusion_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (pair_idx, &(a, b, stats)) in coactive_pairs.iter().enumerate() {
        let dep = stats.dependence();
        if dep < FUSION_DEPENDENCE_FLOOR {
            continue;
        }
        let z = exceedance_z[pair_idx];
        if z >= z_floor {
            fusion_pairs.push((a, b, z));
        }
    }
    fusion_pairs.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    for &(a, b, z) in fusion_pairs.iter().take(params.max_fusions) {
        proposals.push(proposal(term, StructureMove::Fusion { a, b }, z));
    }

    // --- Glue: chart-gluing over-tiling detector (#1890) -------------------
    // A SECOND, independent proposal lane, blind to the co-activation currency
    // the fusion lane above runs on. Atoms over-tiling ONE manifold have
    // DISJOINT supports (each owns its own arc), hence anti-correlated codes, so
    // no such pair EVER clears the co-activation floor — the fusion lane is
    // structurally blind to them. This lane screens pairs GEOMETRICALLY instead:
    // a d=1 periodic pair whose decoder AMBIENT spans align (small principal
    // angles via the Grassmann frame) and whose supports are disjoint is a
    // candidate. Acceptance is NOT decided here — the seam equivalence e-value
    // from `unit_speed_glue_certificate` is carried on the proposal's trigger and
    // the engine's Glue arm banks it against the churn null. The pre-screen only
    // RANKS, so the budget spends its e-value evaluations on the most-aligned
    // pairs; it carries no acceptance threshold of its own (magic-free).
    let mut certified_glues = Vec::new();
    let (glues_proposed, glue_candidates_screened) = harvest_glue_proposals(
        term,
        residuals,
        params.max_fusions,
        &mut proposals,
        &mut certified_glues,
    );

    // --- Fission audits: absorption-suspect asymmetry, gated by the null ---
    let mut fission_atoms: Vec<(usize, f64)> = Vec::new();
    for (pair_idx, &(a, b, stats)) in coactive_pairs.iter().enumerate() {
        let asym = stats.absorption_asymmetry();
        if asym < ABSORPTION_ASYMMETRY_FLOOR {
            continue;
        }
        // A nested (absorbed) pair co-fires ABOVE its fixed margins; a pair
        // whose asymmetry is only the top-`k` mechanical artifact does not.
        // Require the joint activation to exceed the fixed-margin null before
        // auditing, so mechanical asymmetry is not read as absorption.
        let z = exceedance_z[pair_idx];
        if z < z_floor {
            continue;
        }
        // The parent (the conditioned-on atom whose support nests the child) is
        // the one whose `P(parent|child) ≈ 1`. Audit the parent for the absorbed
        // substructure.
        let parent = if stats.p_a_given_b >= stats.p_b_given_a {
            a
        } else {
            b
        };
        // Fission trigger is audit significance ASCENDING; map a high asymmetry
        // to a low significance proxy `1 − asym` so the most asymmetric (most
        // suspect) pair sorts first.
        let significance = (1.0 - asym).max(0.0);
        fission_atoms.push((parent, significance));
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
    let mut birth_predictions: Vec<(usize, f64)> = Vec::new();
    let mut births_deferred = 0usize;
    let mut deferred_predicted_bits = 0.0_f64;
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
                let diagonal = model.diagonal();
                let r = model.factor_rank();
                // #2233 closed-form MDL birth pre-screen. Every quantity below is
                // read from the structured residual-factor fit already computed —
                // no candidate refit runs here. Per-proposal crossover inputs:
                //  * ŝ_j = LOCAL ambient span (participation ratio of the residual's
                //    factor-coordinate energies on candidate j's OWN active rows),
                //  * (d_j, m_j) = the curved topology matched to ŝ_j,
                //  * G = current dictionary size, L0 = mean active atoms/token,
                //  * N = tokens, P = channels.
                let energies: Vec<f64> = (0..r)
                    .map(|j| factor.column(j).iter().map(|v| v * v).sum::<f64>())
                    .collect();
                let norms: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();
                // Per-row projection onto each UNIT factor direction, computed once
                // (`unit_proj[[i, l]] = r_i · u_l`, `u_l = col_l/‖col_l‖`). A proposal's
                // ambient span is then read LOCALLY from these coordinates on its own
                // active rows — never one global participation ratio of the whole
                // residual spectrum, which conflates every co-mined factor and biases
                // admit/defer inconsistently (up when the spectrum is rich, down when
                // it is peaked).
                let mut unit_proj = Array2::<f64>::zeros((n, r));
                for row in 0..n {
                    let res_row = residuals.row(row);
                    for l in 0..r {
                        if norms[l] > 0.0 {
                            let col = factor.column(l);
                            let mut proj = 0.0_f64;
                            for out in 0..p {
                                proj += res_row[out] * col[out];
                            }
                            unit_proj[[row, l]] = proj / norms[l];
                        }
                    }
                }
                let g_dict = term.k_atoms();
                let l0 = mean_active_atoms(assignments.view());
                let n_tokens = n as f64;
                // Score every factor direction; a positive predicted ΔMDL rides as a
                // proposal (ordered by the prediction), a non-positive one is
                // DEFERRED (not proposed this round — a soft defer, never a kill).
                let mut scored: Vec<(usize, f64)> = Vec::with_capacity(r);
                for j in 0..r {
                    let energy = energies[j];
                    if !(energy > 0.0) {
                        // A zero-energy direction carries no residual structure to
                        // birth from — defer it (no finite prediction to bank).
                        births_deferred += 1;
                        continue;
                    }
                    let col = factor.column(j);
                    let norm = norms[j];
                    // Per-direction idiosyncratic-noise floor δ_j = u_jᵀ D u_j (the
                    // residual diagonal projected onto the unit birth direction) —
                    // derived from the fitted noise model, not a hand-set floor.
                    let mut noise_floor = 0.0_f64;
                    for out in 0..p {
                        let u = col[out] / norm;
                        noise_floor += u * u * diagonal[out];
                    }
                    // ρ̂_j (fraction of tokens above the noise floor on u_j) AND the
                    // local factor-coordinate energies on j's active rows, in ONE pass.
                    let mut active = 0usize;
                    let mut local_energy = vec![0.0_f64; r];
                    for row in 0..n {
                        let proj_j = unit_proj[[row, j]];
                        if proj_j * proj_j > noise_floor {
                            active += 1;
                            for l in 0..r {
                                let v = unit_proj[[row, l]];
                                local_energy[l] += v * v;
                            }
                        }
                    }
                    let rho = active as f64 / n_tokens;
                    // ŝ_j: the LOCAL ambient span — participation ratio of the residual's
                    // factor-coordinate energies where THIS candidate fires (a circle
                    // living in a 2-plane ⇒ ≈2; an isolated direction ⇒ ≈1, priced as
                    // linear). (d_j, m_j) follow from ŝ_j, so the dictionary/support
                    // terms are matched to the atom this candidate would actually race.
                    let span = participation_ratio(&local_energy);
                    let (intrinsic_dim, basis_size) = curved_topology_for_span(span);
                    let predicted = predicted_birth_dl_bits(&BirthMdlPrescreen {
                        rho,
                        span,
                        intrinsic_dim,
                        basis_size,
                        signal_var: energy,
                        noise_floor,
                        n_tokens,
                        p_out: p,
                        g_dict,
                        l0,
                    });
                    if predicted.is_finite() && predicted > 0.0 {
                        scored.push((j, predicted));
                    } else {
                        births_deferred += 1;
                        if predicted.is_finite() {
                            deferred_predicted_bits += predicted;
                        }
                    }
                }
                // Order the survivors by predicted ΔMDL (descending), tie-break by
                // index, and cap at `max_births`; the overflow is deferred too.
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for &(candidate, predicted) in scored.iter().take(params.max_births) {
                    proposals.push(proposal(
                        term,
                        StructureMove::Birth { candidate },
                        predicted,
                    ));
                    birth_predictions.push((candidate, predicted));
                    births_proposed += 1;
                }
                for &(_, predicted) in scored.iter().skip(params.max_births) {
                    births_deferred += 1;
                    deferred_predicted_bits += predicted;
                }
                if births_deferred > 0 {
                    log::debug!(
                        "[structure-harvest] #2233 MDL pre-screen deferred {births_deferred} \
                         birth(s) (total predicted ΔMDL {deferred_predicted_bits:.1} bits; \
                         per-proposal local span) of {r} residual factors; proposed \
                         {births_proposed} ordered by predicted ΔMDL",
                    );
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
        birth_predictions,
        births_deferred,
        deferred_predicted_bits,
        birth_skipped_reason,
        glues_proposed,
        glue_candidates_screened,
        certified_glues,
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

/// The exact geometric object which earned a chart-glue proposal's equivalence
/// e-value.  Glue acceptance is based on that harvest-time certificate, so the
/// round driver carries it unchanged to the adoption boundary instead of trying
/// to infer the seam again from a proposal-scoring refit.
#[derive(Clone, Debug)]
enum CertifiedGlueTransition {
    UnitSpeed {
        transition: UnitSpeedChartTransition,
        /// B's support at certification time.  A destructive fusion transplants
        /// precisely these coordinates before B is physically removed.
        rows_b: Vec<usize>,
    },
    Sphere(SphereChartTransition),
}

/// Harvest-time certificate paired one-to-one with an emitted glue proposal.
#[derive(Clone, Debug)]
struct CertifiedGlue {
    a: usize,
    b: usize,
    outcome: ChartGlueOutcome,
    transition: CertifiedGlueTransition,
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
    /// #2233 closed-form MDL pre-screen: `(candidate index, predicted ΔMDL bits)`
    /// for every residual-factor birth that was PROPOSED (predicted saving > 0).
    /// The candidate index is the factor direction the birth seeds from — the same
    /// index the [`StructureMove::Birth`] carries — so the round driver threads
    /// each prediction into the unified [`SaeMigrationLedger`] record the post-refit
    /// verdict fills in (the predicted-vs-realized calibration curve).
    pub birth_predictions: Vec<(usize, f64)>,
    /// #2233: number of residual-factor births DEFERRED this round — non-positive
    /// predicted ΔMDL, so not proposed (a soft defer: they may return next round
    /// once the residual changes; never a hard kill).
    pub births_deferred: usize,
    /// #2233: total predicted ΔMDL (bits) summed over the deferred births — the
    /// round-cadence honesty figure logged alongside the deferred count.
    pub deferred_predicted_bits: f64,
    /// If the birth channel could not run (empty residuals, evidence-ladder
    /// failure), why — so the absence of births is explained, not silent.
    pub birth_skipped_reason: Option<String>,
    /// #1890: number of chart-gluing proposals emitted (pairs whose seam
    /// equivalence e-value was finite and carried to the engine's Glue gate).
    pub glues_proposed: usize,
    /// #1890: number of disjoint-support d=1 periodic pairs the glue lane
    /// GEOMETRICALLY screened (passed the ambient-frame / disjoint-support
    /// pre-screen) before ranking under budget — the loud denominator for the
    /// glues actually proposed.
    pub glue_candidates_screened: usize,
    /// Exact seam transitions paired with the emitted glue proposals.  Private
    /// because these are adoption capabilities, not an additional public
    /// proposal/evidence surface.
    certified_glues: Vec<CertifiedGlue>,
}

// ===========================================================================
// #1890 — chart-gluing lane: the geometric over-tiling detector.
//
// Atoms over-tiling ONE manifold are CHARTS, and the merge test for charts is
// GLUING, not co-activation fusion. The co-activation lane (above) fires on
// DEPENDENT codes; over-tiling atoms have DISJOINT supports (each owns its own
// arc) and hence anti-correlated codes, so the fusion lane is structurally
// blind to them. This lane screens pairs GEOMETRICALLY — a d=1 periodic pair
// whose decoder AMBIENT spans align (small principal angles) and whose supports
// are no more co-active than chance — and its acceptance is an EQUIVALENCE
// e-value on the decoded seam (the two charts coincide within an isometry
// tolerance) against the churn-null scatter, NOT a fit-improvement gate (a
// clean glue leaves EV tied, so a likelihood-ratio gate could never accept it).
// ===========================================================================

/// Fallback intrinsic period of a d=1 periodic atom's latent coordinate when the
/// atom's `Circle { period }` manifold does not report one. The periodic
/// harmonic evaluator sweeps a full circle over `t ∈ [0, 1)` (`angle = 2π·h·t`),
/// so the coordinate period is `1.0` unless the manifold overrides it — which is
/// the value read per-atom by [`atom_axis_period`].
const GLUE_DEFAULT_PERIOD: f64 = 1.0;

/// Finite clamp on the seam log-e-value so a perfect (zero-residual) synthetic
/// glue banks a large-but-finite certificate rather than `+∞` — the engine
/// rejects non-finite triggers, and a banked e-value only needs to clear the
/// ledger threshold `ln(1/α) ≈ 3`, not diverge. Kept well under `ln(f64::MAX)`
/// so the ledger never overflows when it exponentiates the banked log-e.
const GLUE_LOG_E_CLAMP: f64 = 50.0;

/// A fitted seam transition between two d=1 charts A, B of one manifold under
/// the unit-speed gauge: `t_A = sign · t_B + offset` (mod the `2π` period), with
/// the seam equivalence e-value that certifies the two decoded charts coincide
/// within an isometry tolerance against the churn null.
///
/// `sign = +1` is a plain over-tile (two arcs of one ORIENTED circle → a single
/// periodic atom covers the union: the fuse outcome, Increment 1). `sign = -1`
/// is ORIENTATION-REVERSING — the sphere-pole / Möbius signature no single
/// orientable chart can represent, which must instead be REGISTERED as a
/// partition-of-unity atlas atom (Increment 2, in the atom/construction types
/// this lane does not own). The sign is the detector either way.
#[derive(Clone, Copy, Debug)]
pub struct ChartTransition {
    /// `+1` orientation-preserving (fuse), `-1` orientation-reversing (register).
    pub sign: i8,
    /// Latent offset `c` in `t_A = sign·t_B + c`, wrapped into `[0, 2π)`.
    pub offset: f64,
    /// Seam equivalence e-value (log scale) against the churn-null scatter. Large
    /// positive ⇒ the two charts coincide within the reconstruction band beyond
    /// what independent curves would; carried on the proposal trigger and banked
    /// by the engine's Glue gate.
    pub log_e_value: f64,
}

/// The geometric half of a chart glue (no e-value): the fitted sign + offset and
/// the decoded seam clouds/curves, shared by the acceptance e-value
/// ([`unit_speed_glue_certificate`]) and the warm-start coordinate transplant
/// ([`transplant_glued_coords`]).
struct SeamTransition {
    sign: f64,
    offset: f64,
    /// Intrinsic period of atom A's latent coordinate (the transplant wraps into
    /// `[0, period_a)`).
    period: f64,
    rows_a: Vec<usize>,
    points_a: Array2<f64>,
    rows_b: Vec<usize>,
    points_b: Array2<f64>,
    /// A decoded exactly at the transition-mapped coordinates of B.
    mapped_b_to_a: Array2<f64>,
    /// B decoded exactly at the inverse-transition coordinates of A.
    mapped_a_to_b: Array2<f64>,
}

/// Intrinsic period of a d=1 atom's latent coordinate, read from its
/// `Circle { period }` manifold (falling back to [`GLUE_DEFAULT_PERIOD`] when the
/// manifold reports no axis period).
fn atom_axis_period(term: &SaeManifoldTerm, atom: usize) -> f64 {
    let coords = &term.assignment.coords;
    if atom < coords.len() {
        if let Some(Some(p)) = coords[atom].effective_axis_periods().first().copied() {
            if p.is_finite() && p > 0.0 {
                return p;
            }
        }
    }
    GLUE_DEFAULT_PERIOD
}

/// Active rows of `atom` — the discrete support the disjoint-support signature
/// is read from, thresholded at the same relative floor as
/// [`sparse_codes_from_term`].
fn atom_active_rows(term: &SaeManifoldTerm, atom: usize) -> Vec<usize> {
    let assignments = term.assignment.assignments();
    let k = assignments.ncols();
    let floor = if k == 0 {
        0.0
    } else {
        ACTIVE_SUPPORT_REL_FLOOR / k as f64
    };
    (0..assignments.nrows())
        .filter(|&r| assignments[[r, atom]] > floor)
        .collect()
}

/// Decode an atom's ambient points `x_i = Φ_k(t_i) · B_k` at the given rows,
/// reading the atom's already-evaluated basis values (no re-evaluate). Returns
/// `(rows.len() × p)`.
fn decoded_points_at(atom: &SaeManifoldAtom, rows: &[usize]) -> Array2<f64> {
    let phi_sub = atom.basis_values.select(Axis(0), rows);
    phi_sub.dot(&atom.decoder_coefficients)
}

/// Decode a standard periodic-harmonic coefficient block at explicit
/// coordinates.  This evaluates the analytic family at the requested points;
/// no sampled curve or nearest-grid approximation is involved.
fn periodic_decoded_points(
    decoder: ArrayView2<'_, f64>,
    coordinates: &[f64],
) -> Option<Array2<f64>> {
    let m = decoder.nrows();
    if m == 0 || m % 2 == 0 {
        return None;
    }
    let p = decoder.ncols();
    let harmonics = (m - 1) / 2;
    let mut points = Array2::<f64>::zeros((coordinates.len(), p));
    for (row, &coordinate) in coordinates.iter().enumerate() {
        for output in 0..p {
            let mut value = decoder[[0, output]];
            for harmonic in 1..=harmonics {
                let angle = std::f64::consts::TAU * harmonic as f64 * coordinate;
                value += angle.sin() * decoder[[2 * harmonic - 1, output]]
                    + angle.cos() * decoder[[2 * harmonic, output]];
            }
            points[[row, output]] = value;
        }
    }
    Some(points)
}

/// Decoder of A expressed in B's coordinate under
/// `t_A = sign*t_B + offset`.  Harmonic addition identities make this action
/// exact for every represented harmonic.
fn periodic_decoder_under_transition(
    decoder_a: ArrayView2<'_, f64>,
    sign: i8,
    offset: f64,
) -> Option<Array2<f64>> {
    if !matches!(sign, -1 | 1) || decoder_a.nrows() == 0 || decoder_a.nrows() % 2 == 0 {
        return None;
    }
    let mut mapped = decoder_a.to_owned();
    let harmonics = (decoder_a.nrows() - 1) / 2;
    for harmonic in 1..=harmonics {
        let angle = std::f64::consts::TAU * harmonic as f64 * offset;
        let (cosine, sine) = (angle.cos(), angle.sin());
        for output in 0..decoder_a.ncols() {
            let a_sin = decoder_a[[2 * harmonic - 1, output]];
            let a_cos = decoder_a[[2 * harmonic, output]];
            if sign == 1 {
                mapped[[2 * harmonic - 1, output]] = cosine * a_sin - sine * a_cos;
                mapped[[2 * harmonic, output]] = sine * a_sin + cosine * a_cos;
            } else {
                mapped[[2 * harmonic - 1, output]] = -cosine * a_sin + sine * a_cos;
                mapped[[2 * harmonic, output]] = sine * a_sin + cosine * a_cos;
            }
        }
    }
    Some(mapped)
}

/// Closed-form registration of two periodic harmonic decoders.  The first
/// non-zero harmonic identifies a finite set of phase roots analytically for
/// each of the only two unit-speed slopes (`+1`, `-1`); the complete coefficient
/// block then chooses the root/sign by exact represented-function residual.
/// There is no coordinate scan, optimizer, finite difference, or sampled
/// nearest-point proxy.
fn fit_periodic_transition_from_decoders(
    decoder_a: ArrayView2<'_, f64>,
    decoder_b: ArrayView2<'_, f64>,
) -> Option<(i8, f64)> {
    if decoder_a.dim() != decoder_b.dim() || decoder_a.nrows() < 3 || decoder_a.nrows() % 2 == 0 {
        return None;
    }
    let dot = |left_row: usize, right_row: usize| -> f64 {
        (0..decoder_a.ncols())
            .map(|output| decoder_b[[left_row, output]] * decoder_a[[right_row, output]])
            .sum()
    };
    let harmonics = (decoder_a.nrows() - 1) / 2;
    let mut candidates = Vec::new();
    for sign in [1_i8, -1_i8] {
        for harmonic in 1..=harmonics {
            let sin_row = 2 * harmonic - 1;
            let cos_row = 2 * harmonic;
            let (cos_score, sin_score) = if sign == 1 {
                (
                    dot(sin_row, sin_row) + dot(cos_row, cos_row),
                    -dot(sin_row, cos_row) + dot(cos_row, sin_row),
                )
            } else {
                (
                    -dot(sin_row, sin_row) + dot(cos_row, cos_row),
                    dot(sin_row, cos_row) + dot(cos_row, sin_row),
                )
            };
            if cos_score.hypot(sin_score) > 0.0 {
                let harmonic_phase = sin_score.atan2(cos_score).rem_euclid(std::f64::consts::TAU);
                // `h*delta = harmonic_phase (mod 2π)` has exactly h roots.
                for branch in 0..harmonic {
                    let phase =
                        (harmonic_phase + std::f64::consts::TAU * branch as f64) / harmonic as f64;
                    candidates.push((sign, phase));
                }
                break;
            }
        }
    }

    candidates
        .into_iter()
        .filter_map(|(sign, angle)| {
            let offset = angle / std::f64::consts::TAU;
            let mapped = periodic_decoder_under_transition(decoder_a, sign, offset)?;
            let residual = mapped
                .iter()
                .zip(decoder_b.iter())
                .map(|(predicted, observed)| (predicted - observed).powi(2))
                .sum::<f64>();
            residual.is_finite().then_some((sign, offset, residual))
        })
        .min_by(|left, right| {
            left.2.total_cmp(&right.2).then_with(|| {
                // Exact ties are gauge-ambiguous; canonicalize to +1 so an
                // isotropic circle is not spuriously called a half-twist.
                right.0.cmp(&left.0)
            })
        })
        .map(|(sign, offset, _)| (sign, offset))
}

/// Fit the geometric seam transition (sign + offset) between two d=1 charts and
/// carry the decoded clouds/curves the e-value and transplant reuse. `None`
/// unless both atoms are d=1 standard periodic-harmonic charts with the same
/// period/width, a shared ambient dim, and non-empty active supports.
fn fit_seam_transition(term: &SaeManifoldTerm, a: usize, b: usize) -> Option<SeamTransition> {
    let k = term.k_atoms();
    if a >= k || b >= k || a == b {
        return None;
    }
    let atom_a = &term.atoms[a];
    let atom_b = &term.atoms[b];
    if atom_a.latent_dim != 1
        || atom_b.latent_dim != 1
        || !matches!(atom_a.basis_kind, SaeAtomBasisKind::Periodic)
        || !matches!(atom_b.basis_kind, SaeAtomBasisKind::Periodic)
    {
        return None;
    }
    let p = atom_a.decoder_coefficients.ncols();
    if p == 0 || p != atom_b.decoder_coefficients.ncols() {
        return None;
    }
    let rows_a = atom_active_rows(term, a);
    let rows_b = atom_active_rows(term, b);
    if rows_a.is_empty() || rows_b.is_empty() {
        return None;
    }
    let coords = &term.assignment.coords;
    if b >= coords.len() || coords[b].latent_dim() < 1 {
        return None;
    }
    let period_a = atom_axis_period(term, a);
    let period_b = atom_axis_period(term, b);
    // `PeriodicHarmonicEvaluator` is exactly one-periodic in its stored raw
    // coordinate.  A different retraction period does not describe this basis,
    // so refuse rather than silently rescale or approximate it.
    if period_a.to_bits() != 1.0_f64.to_bits() || period_b.to_bits() != period_a.to_bits() {
        return None;
    }
    let decoder_a = atom_a.full_width_decoder();
    let decoder_b = atom_b.full_width_decoder();
    let (sign, offset) = fit_periodic_transition_from_decoders(decoder_a.view(), decoder_b.view())?;
    let points_a = decoded_points_at(atom_a, &rows_a);
    let points_b = decoded_points_at(atom_b, &rows_b);
    let mapped_b_coords: Vec<f64> = rows_b
        .iter()
        .map(|&row| (sign as f64 * coords[b].row(row)[0] + offset).rem_euclid(period_a))
        .collect();
    let mapped_a_coords: Vec<f64> = rows_a
        .iter()
        .map(|&row| {
            // Inverse of `t_a = sign*t_b + offset`; sign^{-1} = sign.
            (sign as f64 * (coords[a].row(row)[0] - offset)).rem_euclid(period_a)
        })
        .collect();
    let mapped_b_to_a = periodic_decoded_points(decoder_a.view(), &mapped_b_coords)?;
    let mapped_a_to_b = periodic_decoded_points(decoder_b.view(), &mapped_a_coords)?;
    Some(SeamTransition {
        sign: sign as f64,
        offset,
        period: period_a,
        rows_a,
        points_a,
        rows_b,
        points_b,
        mapped_b_to_a,
        mapped_a_to_b,
    })
}

/// The seam equivalence e-value (#1890): does chart A's arc lie on chart B's
/// curve AND vice versa, within the reconstruction band, beyond the pooled
/// independent-scatter (churn-scatter reference null) scale?
///
/// Per point the statistic is the Gaussian likelihood ratio
/// `N(x; other_chart(transition(t)), σ_band²) /
/// N(x; pooled centroid, σ_pool²)`, both directions, summed to a log-e-value.
/// The numerator is evaluated at the exact analytic affine transition, not at a
/// nearest point on a sampled grid:
///
/// * `σ_band²` — per-coordinate reconstruction noise floor (the isometry
///   TOLERANCE), the mean squared dictionary residual over the pair's rows,
///   floored only at the representation's machine-resolution scale.
/// * `σ_pool²` — per-coordinate scatter of the pooled decoded points about their
///   centroid (the reference-null currency: how far apart INDEPENDENT curves'
///   points sit). A pair whose pooled scatter is no larger than the band cannot
///   be discriminated and yields no e-value.
///
/// # Honesty: this is a genuine e-value, by SAMPLE SPLITTING
///
/// `σ_band²`, `σ_pool²` and the centroid `μ` are estimated on the EVEN-indexed
/// boundary points and the per-point ratios are evaluated ONLY on the
/// ODD-indexed points. The reference-null density and the numerator variance are
/// therefore independent of the points they are scored on, so under the null
/// (odd points drawn from the isotropic churn-scatter reference `N(μ, σ_pool²I)`)
/// `E_null[∏ q/p] = ∏ E_null[q/p] = 1` — a bona-fide e-value, not the plug-in LR
/// a same-data estimate would give (whose optimism has no `E[e]≤1` guarantee).
/// The e-value is stated against the isotropic-Gaussian reference null at the
/// pooled scale; that is the "independent decode scatter" the churn currency
/// stands in for.
///
/// Two arcs of ONE circle decode ONTO each other's curve (`e_glue ≈ σ_band ≪
/// σ_pool`) so the ratio is large positive; two DISTINCT circles decode FAR from
/// each other's curve (`e_glue ~ σ_pool ≫ σ_band`) so the `−e_glue/(2σ_band²)`
/// term drives the ratio large negative — the tied-EV-cannot-win property the
/// issue requires.
fn unit_speed_glue_certificate(
    term: &SaeManifoldTerm,
    residuals: ArrayView2<'_, f64>,
    a: usize,
    b: usize,
) -> Option<(ChartTransition, CertifiedGlue)> {
    let seam = fit_seam_transition(term, a, b)?;
    let log_e = seam_equivalence_log_e(
        residuals,
        &seam.rows_a,
        &seam.points_a,
        &seam.mapped_a_to_b,
        &seam.rows_b,
        &seam.points_b,
        &seam.mapped_b_to_a,
    )?;
    let chart_transition = ChartTransition {
        sign: seam.sign as i8,
        offset: seam.offset,
        log_e_value: log_e,
    };
    let outcome = if chart_transition.sign == 1 {
        ChartGlueOutcome::Fuse
    } else {
        ChartGlueOutcome::RegisterAtlas
    };
    let transition = UnitSpeedChartTransition::new(
        b,
        a,
        chart_transition.sign,
        chart_transition.offset,
        seam.period,
        AtlasSeamKind::Regular,
    )
    .ok()?;
    Some((
        chart_transition,
        CertifiedGlue {
            a,
            b,
            outcome,
            transition: CertifiedGlueTransition::UnitSpeed {
                transition,
                rows_b: seam.rows_b,
            },
        },
    ))
}

/// The sample-split equivalence log-e-value shared by the 1-D
/// ([`unit_speed_glue_certificate`]) and sphere ([`sphere_glue_pair_evalue`]) seam
/// certifiers.  `points_*` are the decoded ambient clouds of each chart's active
/// rows; `mapped_*` are the SAME points carried through the fitted transition to
/// the other chart's coordinate; `rows_*` index `residuals` for the
/// reconstruction band.
///
/// The reference-null centroid/scatter and the reconstruction band are estimated
/// on the EVEN-indexed points and the per-point Gaussian likelihood ratio
/// `N(x; other_chart(transition(t)), σ_band²) / N(x; pooled centroid, σ_pool²)`
/// is scored ONLY on the ODD-indexed points, so under the isotropic churn-scatter
/// reference null `E_null[∏ q/p] = 1` — a bona-fide e-value, not a plug-in LR.
fn seam_equivalence_log_e(
    residuals: ArrayView2<'_, f64>,
    rows_a: &[usize],
    points_a: &Array2<f64>,
    mapped_a_to_b: &Array2<f64>,
    rows_b: &[usize],
    points_b: &Array2<f64>,
    mapped_b_to_a: &Array2<f64>,
) -> Option<f64> {
    let p = points_a.ncols();
    if p == 0 || residuals.ncols() != p || points_b.ncols() != p {
        return None;
    }
    let na = points_a.nrows();
    let nb = points_b.nrows();
    if na != rows_a.len() || nb != rows_b.len() {
        return None;
    }
    if mapped_a_to_b.dim() != (na, p) || mapped_b_to_a.dim() != (nb, p) {
        return None;
    }
    // Sample split needs at least one estimation and one evaluation point on
    // each side (even/odd parity), so ≥ 2 active points per atom.
    if na < 2 || nb < 2 {
        return None;
    }
    let a_est: Vec<usize> = (0..na).filter(|i| i % 2 == 0).collect();
    let a_eval: Vec<usize> = (0..na).filter(|i| i % 2 == 1).collect();
    let b_est: Vec<usize> = (0..nb).filter(|i| i % 2 == 0).collect();
    let b_eval: Vec<usize> = (0..nb).filter(|i| i % 2 == 1).collect();
    let n_est = a_est.len() + b_est.len();
    let n_eval = a_eval.len() + b_eval.len();
    if n_est == 0 || n_eval == 0 {
        return None;
    }

    // --- Reference-null centroid + scatter, ESTIMATED on the even points ------
    let mut mu = vec![0.0_f64; p];
    for &i in &a_est {
        for c in 0..p {
            mu[c] += points_a[[i, c]];
        }
    }
    for &i in &b_est {
        for c in 0..p {
            mu[c] += points_b[[i, c]];
        }
    }
    for c in 0..p {
        mu[c] /= n_est as f64;
    }
    let point_null_sq =
        |pt: ArrayView1<'_, f64>| -> f64 { (0..p).map(|c| (pt[c] - mu[c]).powi(2)).sum::<f64>() };
    let mut pool_acc = 0.0_f64;
    for &i in &a_est {
        pool_acc += point_null_sq(points_a.row(i));
    }
    for &i in &b_est {
        pool_acc += point_null_sq(points_b.row(i));
    }
    let pool_sq = pool_acc / (n_est as f64 * p as f64);
    if !(pool_sq.is_finite() && pool_sq > 0.0) {
        return None;
    }

    // --- Reconstruction band (tolerance), ESTIMATED on the even rows ----------
    let mut band_acc = 0.0_f64;
    let mut band_rows = 0usize;
    for &i in &a_est {
        let r = rows_a[i];
        for c in 0..p {
            band_acc += residuals[[r, c]].powi(2);
        }
        band_rows += 1;
    }
    for &i in &b_est {
        let r = rows_b[i];
        for c in 0..p {
            band_acc += residuals[[r, c]].powi(2);
        }
        band_rows += 1;
    }
    let band_raw = if band_rows == 0 {
        0.0
    } else {
        band_acc / (band_rows as f64 * p as f64)
    };
    // The transition is analytic, so there is no discretization tolerance.  A
    // perfect synthetic band still needs a positive normal density; the only
    // floor is f64's own relative resolution at the observed pooled scale.
    let band_sq = band_raw.max(pool_sq * f64::EPSILON);
    // A pooled scatter no larger than the band cannot separate "same manifold"
    // from "independent blob" — no e-value.
    if !(pool_sq > band_sq) {
        return None;
    }

    // --- Per-point likelihood ratio, EVALUATED on the odd points --------------
    let norm_term = (p as f64 / 2.0) * (pool_sq / band_sq).ln();
    let mut log_e = 0.0_f64;
    for &i in &b_eval {
        let e_glue: f64 = (0..p)
            .map(|c| (points_b[[i, c]] - mapped_b_to_a[[i, c]]).powi(2))
            .sum();
        let e_null = point_null_sq(points_b.row(i));
        log_e += norm_term - e_glue / (2.0 * band_sq) + e_null / (2.0 * pool_sq);
    }
    for &i in &a_eval {
        let e_glue: f64 = (0..p)
            .map(|c| (points_a[[i, c]] - mapped_a_to_b[[i, c]]).powi(2))
            .sum();
        let e_null = point_null_sq(points_a.row(i));
        log_e += norm_term - e_glue / (2.0 * band_sq) + e_null / (2.0 * pool_sq);
    }
    if !log_e.is_finite() {
        log_e = if log_e < 0.0 {
            -GLUE_LOG_E_CLAMP
        } else {
            GLUE_LOG_E_CLAMP
        };
    }
    Some(log_e.clamp(-GLUE_LOG_E_CLAMP, GLUE_LOG_E_CLAMP))
}

// ===========================================================================
// #1890 Increment 2 — SPHERE POLE seams (the d=2 register emitter).
//
// A sphere pole seam is TWO `SphereChartEvaluator` (lat/lon, `latent_dim = 2`)
// charts covering ONE ambient sphere with their poles in each other's INTERIOR:
// neither lat/lon chart alone covers both poles (`cos(lat) → 0` gauge collapse),
// so the cover is irreducibly an atlas — a single chart cannot represent it.
// The transition relating two such charts is an ambient rotation `R ∈ SO(3)` on
// the intrinsic unit vector `u = [x, y, z]`, NOT a 1-D affine map; the 1-D seam
// fit ([`fit_seam_transition`]) is structurally blind to it (it short-circuits
// on non-`Periodic`, `latent_dim ≠ 1` atoms). This lane fits that rotation by
// exact orthogonal Procrustes on the two decoders' linear `[x, y, z]` blocks,
// classifies pole-vs-regular by whether each chart's pole falls strictly inside
// the OTHER chart's active latitude span (data-driven, no magic angle), and
// certifies the overlap with the SAME sample-split equivalence e-value the 1-D
// lane uses. A pole seam always REGISTERS (keeps both charts as one
// partition-of-unity atlas atom); a sphere is orientable, so its proper-rotation
// transition carries `sign = +1` in the cocycle.
// ===========================================================================

/// The linear `[x, y, z]` decoder block of a `SphereChartEvaluator` atom: rows
/// `1..4` of the `(7, p)` decoder (`[1, x, y, z, xy, yz, xz]`), returned as the
/// `3 × p` ambient frame the sphere's unit vector maps through.
fn sphere_linear_block(atom: &SaeManifoldAtom) -> Option<Array2<f64>> {
    let decoder = atom.full_width_decoder();
    if decoder.nrows() != 7 {
        return None;
    }
    Some(decoder.slice(ndarray::s![1..4, ..]).to_owned())
}

/// Exact inverse of a `3×3` matrix (Cramer). `None` if singular.
fn inverse_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(inv)
}

/// Nearest orthogonal matrix to `m` (the orthogonal polar factor `U Vᵀ` of its
/// SVD) via Higham's polar iteration `Q ← ½(Q + Q⁻ᵀ)`.  This is the orthogonal
/// Procrustes solution `argmin_{QᵀQ=I} ‖Q − M‖_F`; convergence is quadratic once
/// near the factor, so a few dozen iterations reach machine precision.  `None`
/// if any iterate is singular (a degenerate, non-invertible frame product).
fn nearest_orthogonal_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut q = m;
    for _ in 0..128 {
        let inv = inverse_3x3(&q)?;
        let mut next = [[0.0; 3]; 3];
        let mut diff = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                // inv transpose: (Q⁻ᵀ)[i][j] = inv[j][i].
                next[i][j] = 0.5 * (q[i][j] + inv[j][i]);
                diff += (next[i][j] - q[i][j]).abs();
            }
        }
        q = next;
        if diff < 1e-15 {
            break;
        }
    }
    if q.iter().flatten().any(|x| !x.is_finite()) {
        return None;
    }
    Some(q)
}

/// Decode a sphere chart's `(7, p)` decoder at explicit intrinsic unit vectors
/// `u = [x, y, z]` (on `S²`), evaluating the analytic basis
/// `[1, x, y, z, xy, yz, xz]` — no sampled-grid nearest-point proxy.
fn sphere_decoded_points_at_units(
    decoder: ArrayView2<'_, f64>,
    units: &[[f64; 3]],
) -> Option<Array2<f64>> {
    if decoder.nrows() != 7 {
        return None;
    }
    let p = decoder.ncols();
    let mut points = Array2::<f64>::zeros((units.len(), p));
    for (row, &[x, y, z]) in units.iter().enumerate() {
        let phi = [1.0, x, y, z, x * y, y * z, x * z];
        for output in 0..p {
            let mut value = 0.0;
            for (basis, &phi_b) in phi.iter().enumerate() {
                value += phi_b * decoder[[basis, output]];
            }
            points[[row, output]] = value;
        }
    }
    Some(points)
}

/// A fitted sphere pole-seam transition: the ambient rotation `R` (`b -> a`, so
/// `u_a = R u_b`), its pole-vs-regular classification, and the decoded /
/// rotation-mapped point clouds the shared equivalence e-value scores.
struct SphereSeamTransition {
    rotation: [[f64; 3]; 3],
    seam_kind: AtlasSeamKind,
    rows_a: Vec<usize>,
    points_a: Array2<f64>,
    rows_b: Vec<usize>,
    points_b: Array2<f64>,
    /// A decoded at the rotation-mapped unit vectors of B's active rows.
    mapped_b_to_a: Array2<f64>,
    /// B decoded at the inverse-rotation-mapped unit vectors of A's active rows.
    mapped_a_to_b: Array2<f64>,
}

/// The intrinsic unit vector `[x, y, z]` of a sphere atom's active row, read from
/// its already-evaluated basis values (`phi = [1, x, y, z, ...]`).
fn sphere_row_unit(atom: &SaeManifoldAtom, row: usize) -> [f64; 3] {
    [
        atom.basis_values[[row, 1]],
        atom.basis_values[[row, 2]],
        atom.basis_values[[row, 3]],
    ]
}

/// Apply a `3×3` rotation to a unit vector.
fn rotate_unit(r: &[[f64; 3]; 3], u: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * u[0] + r[0][1] * u[1] + r[0][2] * u[2],
        r[1][0] * u[0] + r[1][1] * u[1] + r[1][2] * u[2],
        r[2][0] * u[0] + r[2][1] * u[1] + r[2][2] * u[2],
    ]
}

/// Whether two atoms are both `SphereChartEvaluator` local charts.
fn is_sphere_pair(term: &SaeManifoldTerm, a: usize, b: usize) -> bool {
    let k = term.k_atoms();
    if a >= k || b >= k || a == b {
        return false;
    }
    let sa = &term.atoms[a];
    let sb = &term.atoms[b];
    sa.latent_dim == 2
        && sb.latent_dim == 2
        && matches!(sa.basis_kind, SaeAtomBasisKind::Sphere)
        && matches!(sb.basis_kind, SaeAtomBasisKind::Sphere)
}

/// Fit the ambient-rotation seam transition between two sphere charts and
/// classify pole-vs-regular. `None` unless both atoms are `latent_dim = 2` sphere
/// charts with a shared ambient dim, non-empty active supports, and an invertible
/// frame product.
fn fit_sphere_seam_transition(
    term: &SaeManifoldTerm,
    a: usize,
    b: usize,
) -> Option<SphereSeamTransition> {
    if !is_sphere_pair(term, a, b) {
        return None;
    }
    let atom_a = &term.atoms[a];
    let atom_b = &term.atoms[b];
    let p = atom_a.decoder_coefficients.ncols();
    if p == 0 || p != atom_b.decoder_coefficients.ncols() {
        return None;
    }
    let l_a = sphere_linear_block(atom_a)?;
    let l_b = sphere_linear_block(atom_b)?;
    // Orthogonal Procrustes: R = argmin ‖L_a − R L_b‖ ⇒ R is the orthogonal
    // factor of M = L_a L_bᵀ (3×3). Then u_a = R u_b matches the ambient images.
    let m_prod = l_a.dot(&l_b.t());
    let mut m = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = m_prod[[i, j]];
        }
    }
    let rotation = nearest_orthogonal_3x3(m)?;
    let rows_a = atom_active_rows(term, a);
    let rows_b = atom_active_rows(term, b);
    if rows_a.is_empty() || rows_b.is_empty() {
        return None;
    }
    // Active latitudes (asin z) of each chart, and each chart's pole mapped into
    // the OTHER chart's coordinate: pole-vs-regular is decided by whether the
    // mapped pole falls strictly inside the other chart's active latitude span.
    let lat_of = |u: [f64; 3]| -> f64 { u[2].clamp(-1.0, 1.0).asin() };
    let (mut a_lat_lo, mut a_lat_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &r in &rows_a {
        let lat = lat_of(sphere_row_unit(atom_a, r));
        a_lat_lo = a_lat_lo.min(lat);
        a_lat_hi = a_lat_hi.max(lat);
    }
    let (mut b_lat_lo, mut b_lat_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &r in &rows_b {
        let lat = lat_of(sphere_row_unit(atom_b, r));
        b_lat_lo = b_lat_lo.min(lat);
        b_lat_hi = b_lat_hi.max(lat);
    }
    // B's north pole u_b = [0,0,1] into A's coordinate; A's north pole into B's.
    let b_pole_in_a = lat_of(rotate_unit(&rotation, [0.0, 0.0, 1.0]));
    let inv_rotation = inverse_3x3(&rotation)?;
    let a_pole_in_b = lat_of(rotate_unit(&inv_rotation, [0.0, 0.0, 1.0]));
    let b_pole_interior_to_a = b_pole_in_a > a_lat_lo && b_pole_in_a < a_lat_hi;
    let a_pole_interior_to_b = a_pole_in_b > b_lat_lo && a_pole_in_b < b_lat_hi;
    let seam_kind = if b_pole_interior_to_a && a_pole_interior_to_b {
        AtlasSeamKind::Pole
    } else {
        AtlasSeamKind::Regular
    };
    // Decoded clouds + rotation-mapped clouds for the equivalence e-value.
    let units_a: Vec<[f64; 3]> = rows_a.iter().map(|&r| sphere_row_unit(atom_a, r)).collect();
    let units_b: Vec<[f64; 3]> = rows_b.iter().map(|&r| sphere_row_unit(atom_b, r)).collect();
    let points_a = sphere_decoded_points_at_units(atom_a.full_width_decoder().view(), &units_a)?;
    let points_b = sphere_decoded_points_at_units(atom_b.full_width_decoder().view(), &units_b)?;
    // B's rows carried into A: rotate u_b -> u_a, decode through A.
    let mapped_b_units: Vec<[f64; 3]> =
        units_b.iter().map(|&u| rotate_unit(&rotation, u)).collect();
    let mapped_b_to_a =
        sphere_decoded_points_at_units(atom_a.full_width_decoder().view(), &mapped_b_units)?;
    // A's rows carried into B: rotate u_a by R⁻¹ -> u_b, decode through B.
    let mapped_a_units: Vec<[f64; 3]> = units_a
        .iter()
        .map(|&u| rotate_unit(&inv_rotation, u))
        .collect();
    let mapped_a_to_b =
        sphere_decoded_points_at_units(atom_b.full_width_decoder().view(), &mapped_a_units)?;
    Some(SphereSeamTransition {
        rotation,
        seam_kind,
        rows_a,
        points_a,
        rows_b,
        points_b,
        mapped_b_to_a,
        mapped_a_to_b,
    })
}

/// The sphere pole-seam equivalence certificate (#1890 Increment 2). Returns the
/// exact ambient-rotation transition (`b -> a`) plus its equivalence log-e-value,
/// or `None` if the pair is not an identifiable pole seam.  Only genuine POLE
/// seams (each pole interior to the other chart) are certified for registration;
/// a regular sphere overlap has no register/fuse outcome wired in this lane and
/// yields `None`.
fn sphere_glue_pair_evalue(
    term: &SaeManifoldTerm,
    residuals: ArrayView2<'_, f64>,
    a: usize,
    b: usize,
) -> Option<(SphereChartTransition, f64)> {
    let seam = fit_sphere_seam_transition(term, a, b)?;
    if !matches!(seam.seam_kind, AtlasSeamKind::Pole) {
        return None;
    }
    let log_e = seam_equivalence_log_e(
        residuals,
        &seam.rows_a,
        &seam.points_a,
        &seam.mapped_a_to_b,
        &seam.rows_b,
        &seam.points_b,
        &seam.mapped_b_to_a,
    )?;
    let transition = SphereChartTransition::new(b, a, seam.rotation, AtlasSeamKind::Pole).ok()?;
    Some((transition, log_e))
}

/// Emit the chart-gluing proposal lane (#1890) into `proposals`, ranked under
/// `budget`. Returns `(glues_proposed, candidates_screened)`.
///
/// Pre-screen (GEOMETRIC, code-blind): a pair `(a, b)` is a candidate iff both
/// atoms are d=1 periodic with a live decoder ambient frame, their supports are
/// no more co-active than independence would give (`inter ≤ na·nb/N` — the
/// complement of the fusion lane's positive-dependence trigger, so the two lanes
/// partition the pair space with NO tuned overlap threshold), and their ambient
/// spans are comparable (principal-angle alignment, the ranking key). The
/// pre-screen only RANKS; the seam equivalence e-value owns acceptance, so no
/// magic alignment cutoff is imposed here.
fn harvest_glue_proposals(
    term: &SaeManifoldTerm,
    residuals: ArrayView2<'_, f64>,
    budget: usize,
    proposals: &mut Vec<MoveProposal>,
    certified_glues: &mut Vec<CertifiedGlue>,
) -> (usize, usize) {
    let k = term.k_atoms();
    if k < 2 || budget == 0 {
        return (0, 0);
    }
    let assignments = term.assignment.assignments();
    let n_rows = assignments.nrows();
    if n_rows == 0 {
        return (0, 0);
    }
    let floor = ACTIVE_SUPPORT_REL_FLOOR / k as f64;
    let supports: Vec<Vec<bool>> = (0..k)
        .map(|atom| {
            (0..n_rows)
                .map(|r| assignments[[r, atom]] > floor)
                .collect()
        })
        .collect();
    let support_sizes: Vec<usize> = supports
        .iter()
        .map(|s| s.iter().filter(|&&x| x).count())
        .collect();
    // Ambient decoder frames — only for d=1 periodic atoms (the over-tiling
    // signature); every other atom is a `None` and never a glue endpoint.
    let frames: Vec<Option<GrassmannFrame>> = (0..k)
        .map(|atom| {
            let at = &term.atoms[atom];
            if at.latent_dim == 1 && matches!(at.basis_kind, SaeAtomBasisKind::Periodic) {
                GrassmannFrame::from_decoder_row_space(at.decoder_coefficients.view())
            } else {
                None
            }
        })
        .collect();

    let mut screened = 0usize;
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..k {
        let fa = match &frames[a] {
            Some(f) => f,
            None => continue,
        };
        if support_sizes[a] == 0 {
            continue;
        }
        for b in (a + 1)..k {
            // A registered pair is already one semantic atom.  Re-proposing its
            // seam would double-bank the same geometric fact and, worse, could
            // later route it through the destructive fuse outcome.
            if term.charts_share_atlas(a, b) {
                continue;
            }
            let fb = match &frames[b] {
                Some(f) => f,
                None => continue,
            };
            if support_sizes[b] == 0 {
                continue;
            }
            // Disjoint-support gate: keep only pairs that co-fire no MORE than
            // independence predicts (the anti-correlated / disjoint signature the
            // fusion lane cannot see). Positively co-active pairs are fusion's.
            let inter = (0..n_rows)
                .filter(|&r| supports[a][r] && supports[b][r])
                .count();
            let expected = support_sizes[a] as f64 * support_sizes[b] as f64 / n_rows as f64;
            if inter as f64 > expected {
                continue;
            }
            // Ambient-span alignment (ranking key): cos of the largest principal
            // angle between the two decoder frames — 1 for a shared plane.
            let alignment = match fa.max_principal_angle(fb.frame()) {
                Ok(theta) => theta.cos(),
                Err(_) => continue,
            };
            if !alignment.is_finite() {
                continue;
            }
            screened += 1;
            candidates.push((a, b, alignment));
        }
    }
    candidates.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));

    let mut proposed = 0usize;
    for &(a, b, _score) in candidates.iter().take(budget) {
        if let Some((tr, certificate)) = unit_speed_glue_certificate(term, residuals, a, b) {
            proposals.push(proposal(
                term,
                StructureMove::Glue {
                    a,
                    b,
                    outcome: certificate.outcome,
                },
                tr.log_e_value,
            ));
            certified_glues.push(certificate);
            proposed += 1;
        }
    }

    // --- Sphere POLE-seam pass (#1890 Increment 2, the d=2 register emitter) ---
    // Two `SphereChartEvaluator` charts whose poles sit in each other's interior
    // are an irreducible atlas; the transition is an ambient rotation the 1-D
    // lane cannot see. Screened with the SAME disjoint-support gate (over-tiling
    // charts anti-correlate), certified by the ambient-rotation pole e-value, and
    // always REGISTERED (a pole seam is never single-chart-coverable).
    let mut sphere_candidates: Vec<(usize, usize)> = Vec::new();
    for a in 0..k {
        if support_sizes[a] == 0 || !matches!(term.atoms[a].basis_kind, SaeAtomBasisKind::Sphere) {
            continue;
        }
        for b in (a + 1)..k {
            if support_sizes[b] == 0
                || !matches!(term.atoms[b].basis_kind, SaeAtomBasisKind::Sphere)
                || term.charts_share_atlas(a, b)
            {
                continue;
            }
            let inter = (0..n_rows)
                .filter(|&r| supports[a][r] && supports[b][r])
                .count();
            let expected = support_sizes[a] as f64 * support_sizes[b] as f64 / n_rows as f64;
            if inter as f64 > expected {
                continue;
            }
            sphere_candidates.push((a, b));
        }
    }
    for &(a, b) in sphere_candidates.iter().take(budget) {
        screened += 1;
        if let Some((transition, log_e)) = sphere_glue_pair_evalue(term, residuals, a, b) {
            proposals.push(proposal(
                term,
                StructureMove::Glue {
                    a,
                    b,
                    outcome: ChartGlueOutcome::RegisterAtlas,
                },
                log_e,
            ));
            certified_glues.push(CertifiedGlue {
                a,
                b,
                outcome: ChartGlueOutcome::RegisterAtlas,
                transition: CertifiedGlueTransition::Sphere(transition),
            });
            proposed += 1;
        }
    }
    (proposed, screened)
}

/// Warm the glued atom `a`'s chart to cover the union of both arcs by
/// transplanting `b`'s per-row latent coordinate through the certified seam
/// transition `t_a = sign·t_b + offset` (mod `2π`), so the joint refit starts
/// from a full-manifold chart rather than re-discovering `b`'s arc cold.
fn transplant_glued_coords(
    term: &mut SaeManifoldTerm,
    a: usize,
    b: usize,
    transition: &UnitSpeedChartTransition,
    rows_b: &[usize],
) -> Result<(), String> {
    if transition.from_chart != b || transition.to_chart != a {
        return Err(format!(
            "transplant_glued_coords: transition {}->{} does not match glue ({a},{b})",
            transition.from_chart, transition.to_chart
        ));
    }
    let coords = &mut term.assignment.coords;
    if a >= coords.len() || b >= coords.len() {
        return Err(format!(
            "transplant_glued_coords: glue ({a},{b}) outside {} coordinate blocks",
            coords.len()
        ));
    }
    let da = coords[a].latent_dim();
    let db = coords[b].latent_dim();
    if da < 1 || db < 1 || coords[a].n_obs() != coords[b].n_obs() {
        return Err(format!(
            "transplant_glued_coords: incompatible coordinate blocks for glue ({a},{b})"
        ));
    }
    // Read B's flat coords, then write A's transplanted rows through the fitted
    // transition. The read is cloned first, so the mutable borrow of A's coords
    // never aliases B's.
    let flat_b = coords[b].as_flat().to_owned();
    let mut flat_a = coords[a].as_flat().to_owned();
    let n = coords[b].n_obs();
    for &r in rows_b {
        if r >= n {
            return Err(format!(
                "transplant_glued_coords: certified row {r} outside n={n} for glue ({a},{b})"
            ));
        }
        let t_b = flat_b[r * db];
        flat_a[r * da] = transition.apply(t_b);
    }
    coords[a].set_flat(flat_a.view());
    Ok(())
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
        StructureMove::Glue { a, b, outcome } => {
            // Sphere pole seam (d=2): the transition is an ambient rotation, not a
            // 1-D affine map. Register it (keep both charts) — a pole seam has no
            // destructive fuse outcome, since neither lat/lon chart covers both
            // poles alone.
            if is_sphere_pair(term, *a, *b) {
                let mut child = term.clone();
                match outcome {
                    ChartGlueOutcome::RegisterAtlas => {
                        let seam = fit_sphere_seam_transition(term, *a, *b).ok_or_else(|| {
                            format!(
                                "apply_structure_move: sphere seam ({a},{b}) is no longer identifiable"
                            )
                        })?;
                        if !matches!(seam.seam_kind, AtlasSeamKind::Pole) {
                            return Err(format!(
                                "apply_structure_move: sphere seam ({a},{b}) is not a pole seam"
                            ));
                        }
                        child.register_sphere_chart_transition(SphereChartTransition::new(
                            *b,
                            *a,
                            seam.rotation,
                            AtlasSeamKind::Pole,
                        )?)?;
                    }
                    ChartGlueOutcome::Fuse => {
                        return Err(format!(
                            "apply_structure_move: sphere pole seam ({a},{b}) cannot be destructively fused"
                        ));
                    }
                }
                return Ok((child, rho.clone()));
            }
            let seam = fit_seam_transition(term, *a, *b).ok_or_else(|| {
                format!("apply_structure_move: chart seam ({a},{b}) is no longer identifiable")
            })?;
            let mut child = term.clone();
            match outcome {
                ChartGlueOutcome::Fuse => {
                    if seam.sign != 1.0 {
                        return Err(format!(
                            "apply_structure_move: refusing to fuse orientation-reversing seam ({a},{b})"
                        ));
                    }
                    // Index-stable within the round: fold/demote now, physically
                    // excise at [`compact_glued_atoms`] before polish.
                    let transition = UnitSpeedChartTransition::new(
                        *b,
                        *a,
                        1,
                        seam.offset,
                        seam.period,
                        AtlasSeamKind::Regular,
                    )?;
                    fold_atom_into(&mut child, *a, *b)?;
                    transplant_glued_coords(&mut child, *a, *b, &transition, &seam.rows_b)?;
                }
                ChartGlueOutcome::RegisterAtlas => {
                    if seam.sign != -1.0 {
                        return Err(format!(
                            "apply_structure_move: atlas registration requires an orientation-reversing seam, got sign {}",
                            seam.sign
                        ));
                    }
                    // Keep both numerical charts. Their existing routing masses
                    // are exactly atlas activation × partition-of-unity; only the
                    // semantic quotient and transition cocycle are new state.
                    child.register_chart_transition(UnitSpeedChartTransition::new(
                        *b,
                        *a,
                        -1,
                        seam.offset,
                        seam.period,
                        AtlasSeamKind::Regular,
                    )?)?;
                }
            }
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
                } => born_circle_atom(
                    term,
                    rho,
                    decoder.clone(),
                    phase_coords.clone(),
                    gate.clone(),
                ),
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
    // the e-gate under a capped refit. For ordered Beta--Bernoulli/ThresholdGate the per-atom gate is the
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

/// Physically REMOVE the atoms in `remove` from `term`/`rho` — the TRUE-FUSION
/// tail that a demotion alone cannot deliver. A [`fold_atom_into`] combines the
/// folded atom's routing mass into its survivor and drops its logit column to
/// [`DEMOTE_LOGIT`], but a demoted-not-removed atom keeps a full decoder at ~0
/// mass, and the joint refit's #976/#1003 active-mass guard
/// ([`SaeManifoldTerm::enforce_active_mass_guard`]) runs at fit ENTRY after
/// `collapse_events` is cleared: it reads that atom's max gate as below the trust
/// floor and RESEEDS its logits to per-row winner parity, resurrecting the atom
/// the glue just retired (the effective atom count never falls, #1890). So for a
/// certified glue we excise the folded atoms outright — decoder atom,
/// routing-logit column, latent coordinate block, per-atom `ungated`/frozen slot,
/// and per-atom ρ (smoothness + ARD) blocks — so BOTH the raw and the active
/// dictionary size fall, no zero-mass atom survives for the guard to revive, and
/// each survivor is forced to carry the absorbed arc through the refit.
///
/// Every atom is dropped in ONE pass (a shared keep-mask), so removing several
/// atoms needs no descending-index bookkeeping. Each survivor must already hold
/// its folded partner's mass (call this AFTER the folds); any per-atom diagnostic
/// caches are reset so the post-glue refit rebuilds them against the reduced
/// dictionary rather than indexing a stale length-`K`.
fn remove_atoms(
    term: &mut SaeManifoldTerm,
    rho: &mut SaeManifoldRho,
    remove: &std::collections::BTreeSet<usize>,
) -> Result<(), String> {
    let k = term.k_atoms();
    if let Some(&bad) = remove.iter().find(|&&j| j >= k) {
        return Err(format!("remove_atoms: atom {bad} out of range (K={k})"));
    }
    if remove.len() >= k {
        return Err("remove_atoms: cannot remove every atom".to_string());
    }
    if remove.is_empty() {
        return Ok(());
    }
    // Validate every atom-indexed container BEFORE mutating any of them.  The
    // normal constructors maintain these invariants, but this function is the
    // variable-K boundary and must return a useful error rather than partially
    // compacting a malformed warm state and then panicking on an indexed gather.
    let n = term.assignment.logits.nrows();
    if term.assignment.logits.ncols() != k
        || term.assignment.coords.len() != k
        || term.assignment.ungated.len() != k
    {
        return Err(format!(
            "remove_atoms: atom-indexed assignment shape mismatch: atoms={k}, \
             logits={:?}, coords={}, ungated={}",
            term.assignment.logits.dim(),
            term.assignment.coords.len(),
            term.assignment.ungated.len()
        ));
    }
    if let Some(frozen) = term.assignment.frozen_logits.as_ref() {
        if frozen.dim() != (n, k) {
            return Err(format!(
                "remove_atoms: frozen logits shape {:?} must equal ({n}, {k})",
                frozen.dim()
            ));
        }
    }
    if rho.log_lambda_smooth.len() != k || rho.log_ard.len() != k {
        return Err(format!(
            "remove_atoms: rho per-atom lengths (smooth {}, ard {}) must equal K={k}",
            rho.log_lambda_smooth.len(),
            rho.log_ard.len()
        ));
    }
    let keep: Vec<usize> = (0..k).filter(|j| !remove.contains(j)).collect();
    // Registered atlas endpoints are atom indices.  A seam-bearing chart may
    // never be deleted by ordinary compaction; surviving atlases are remapped
    // through the same keep permutation before the atom arrays move.
    let mut old_to_new = vec![None; k];
    for (new, &old) in keep.iter().enumerate() {
        old_to_new[old] = Some(new);
    }
    term.remap_chart_atlases(&old_to_new)?;
    // Rebuild the atom list, coord blocks, ungated flags, and ρ blocks keeping
    // only the surviving indices (descending removal on the Vecs would also work,
    // but the keep-mask keeps atoms/coords/logits/ρ provably in lock-step).
    term.atoms = keep.iter().map(|&j| term.atoms[j].clone()).collect();
    // `ndarray::select(Axis(1), ..)` may retain a column-major/non-standard
    // stride layout. The Newton driver updates logits row-wise and requires each
    // row to be a contiguous mutable slice, so materialize the compacted router
    // explicitly in row-major order at this variable-K boundary.
    let compacted_logits = Array2::from_shape_fn((n, keep.len()), |(row, new_atom)| {
        term.assignment.logits[[row, keep[new_atom]]]
    });
    term.assignment.logits = compacted_logits;
    term.assignment.coords = keep
        .iter()
        .map(|&j| term.assignment.coords[j].clone())
        .collect();
    term.assignment.ungated = keep.iter().map(|&j| term.assignment.ungated[j]).collect();
    // A frozen router was trained against the old dictionary.  Merely slicing
    // its columns would not encode the fold's log-sum-exp mass transfer and
    // would keep routing permanently frozen to an invalid model.  The reduced
    // dictionary must refit its routing from the physically folded logits.
    term.assignment.frozen_logits = None;
    // Per-atom ρ blocks (the block-relevance `log_lambda_block` is per-output-block,
    // not per-atom, so it is untouched).
    rho.log_lambda_smooth = keep.iter().map(|&j| rho.log_lambda_smooth[j]).collect();
    rho.log_ard = keep.iter().map(|&j| rho.log_ard[j].clone()).collect();
    // Drop every K-dependent cache and optimization ledger.  Compaction changes
    // both the column order and the quotient dimension, so retaining any of the
    // old assembly layout, frozen pair gates, evidence-deflation anchor, or
    // per-atom diagnostic reports would make the polish refit interpret old-K
    // state as if it described the reduced dictionary.
    term.collapse_events.clear();
    term.last_row_layout = None;
    term.last_frames_active = false;
    term.fixed_decoder_assembly = false;
    term.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
    term.decoder_repulsion_gate = None;
    term.barrier_coactivation_gate = None;
    term.streaming_gates_frozen = false;
    term.curvature_walk_report = None;
    term.expected_criterion_gauge_deflated_directions = None;
    term.criterion_gauge_deflation_reanchors = 0;
    term.criterion_gauge_deflation_last_delta_sign = 0;
    term.dictionary_cocollapse_reseeds = 0;
    term.structural_cocollapse_reseeds = 0;
    term.atom_inner_fits = None;
    term.oos_linear_images = None;
    term.hybrid_split_report = None;
    term.best_cocollapse_incumbent = None;
    term.best_fit_incumbent = None;
    Ok(())
}

/// Round-boundary chart-glue adoption (#1890).  A glue is a harvest-certified
/// equivalence move, not a likelihood-scored numerical candidate: the exact seam
/// transition which earned its e-value is carried in `certified_glues`, and the
/// search defers materializing the move until its index-sensitive proposal chain
/// is complete.  This boundary then applies every accepted certificate exactly
/// once: an orientation-preserving seam folds/transplants/removes B, while an
/// irreducible reversing or pole seam registers both charts as one atlas atom.
///
/// The whole accepted matching and all proposal/certificate pairings are checked
/// before mutation.  Materialization runs on a cloned child and commits only on
/// success, so malformed resumed/direct ledgers cannot leave half-folded state.
/// The search engine's `touched` guard guarantees accepted glues share no atom;
/// checking it again here makes that adoption contract explicit.  No accepted
/// glue is a no-op.
fn compact_glued_atoms(
    term: &mut SaeManifoldTerm,
    rho: &mut SaeManifoldRho,
    round_ledger: &SearchLedger,
    certified_glues: &[CertifiedGlue],
) -> Result<usize, String> {
    use gam_solve::structure_search::MoveVerdict;
    let accepted_glues: Vec<(usize, usize, ChartGlueOutcome)> = round_ledger
        .moves
        .iter()
        .filter_map(|rec| {
            if let (StructureMove::Glue { a, b, outcome }, MoveVerdict::Accepted { .. }) =
                (&rec.mv, &rec.verdict)
            {
                Some((*a, *b, *outcome))
            } else {
                None
            }
        })
        .collect();
    if accepted_glues.is_empty() {
        return Ok(0);
    }
    // Validate the whole accepted matching before the first fold.  The search
    // engine enforces this through its `touched` set; checking again here keeps
    // the variable-K boundary transactional even for resumed/deserialized or
    // directly-constructed ledgers.
    let k = term.k_atoms();
    let mut touched = std::collections::BTreeSet::new();
    for &(a, b, _) in &accepted_glues {
        if a >= k || b >= k || a == b {
            return Err(format!(
                "compact_glued_atoms: accepted glue ({a},{b}) out of range or self-gluing (K={k})"
            ));
        }
        if !touched.insert(a) || !touched.insert(b) {
            return Err(format!(
                "compact_glued_atoms: accepted glues are not an atom-disjoint matching; \
                 atom reused by ({a},{b})"
            ));
        }
    }

    // Pair every accepted ledger record with exactly one harvest certificate.
    // A proposal without its geometric object cannot be adopted: trying to
    // reconstruct it here from a scoring/polish-mutated state is precisely the
    // ordering bug this boundary forbids.
    let mut adopted: Vec<CertifiedGlue> = Vec::with_capacity(accepted_glues.len());
    for &(a, b, outcome) in &accepted_glues {
        let mut matches = certified_glues
            .iter()
            .filter(|certificate| certificate.a == a && certificate.b == b);
        let certificate = matches.next().ok_or_else(|| {
            format!("compact_glued_atoms: accepted glue ({a},{b}) has no harvest-time certificate")
        })?;
        if matches.next().is_some() {
            return Err(format!(
                "compact_glued_atoms: accepted glue ({a},{b}) has duplicate harvest-time certificates"
            ));
        }
        if certificate.outcome != outcome {
            return Err(format!(
                "compact_glued_atoms: accepted glue ({a},{b}) outcome {outcome:?} does not match certified {:?}",
                certificate.outcome
            ));
        }
        match (&certificate.transition, outcome) {
            (CertifiedGlueTransition::UnitSpeed { transition, .. }, ChartGlueOutcome::Fuse)
                if transition.from_chart == b
                    && transition.to_chart == a
                    && transition.sign == 1
                    && matches!(transition.seam_kind, AtlasSeamKind::Regular) => {}
            (
                CertifiedGlueTransition::UnitSpeed { transition, .. },
                ChartGlueOutcome::RegisterAtlas,
            ) if transition.from_chart == b
                && transition.to_chart == a
                && transition.sign == -1
                && matches!(transition.seam_kind, AtlasSeamKind::Regular) => {}
            (CertifiedGlueTransition::Sphere(transition), ChartGlueOutcome::RegisterAtlas)
                if transition.from_chart == b
                    && transition.to_chart == a
                    && matches!(transition.seam_kind, AtlasSeamKind::Pole) => {}
            _ => {
                return Err(format!(
                    "compact_glued_atoms: accepted glue ({a},{b}) is incompatible with its certified transition"
                ));
            }
        }
        adopted.push(certificate.clone());
    }

    let mut child_term = term.clone();
    let mut child_rho = rho.clone();
    let mut to_remove: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for certificate in adopted {
        let (a, b) = (certificate.a, certificate.b);
        match certificate.transition {
            CertifiedGlueTransition::UnitSpeed { transition, rows_b }
                if matches!(certificate.outcome, ChartGlueOutcome::Fuse) =>
            {
                fold_atom_into(&mut child_term, a, b)?;
                transplant_glued_coords(&mut child_term, a, b, &transition, &rows_b)?;
                to_remove.insert(b);
            }
            CertifiedGlueTransition::UnitSpeed { transition, .. } => {
                child_term.register_chart_transition(transition)?;
            }
            CertifiedGlueTransition::Sphere(transition) => {
                child_term.register_sphere_chart_transition(transition)?;
            }
        }
    }
    remove_atoms(&mut child_term, &mut child_rho, &to_remove)?;
    *term = child_term;
    *rho = child_rho;
    Ok(to_remove.len())
}

/// Refit every production-registered seam on terminal polished charts. Atlas
/// state is persisted model state, so after a genuinely numerical move it must
/// describe the returned fit rather than the pre-polish warm start. A seam that
/// ceases to be identifiable or changes kind/orientation is a structural-fit
/// failure: returning a stale atlas would be worse than failing loudly. Pure
/// registration rounds never call this function because they do not polish.
fn refresh_registered_atlas_transitions(term: &mut SaeManifoldTerm) -> Result<(), String> {
    let registered: Vec<UnitSpeedChartTransition> = term
        .chart_atlases()
        .iter()
        .flat_map(|atlas| atlas.transitions().iter().copied())
        .filter(|transition| matches!(transition.seam_kind, AtlasSeamKind::Regular))
        .collect();
    for transition in registered {
        // `fit_seam_transition(a,b)` returns the map b -> a.
        let seam = fit_seam_transition(term, transition.to_chart, transition.from_chart)
            .ok_or_else(|| {
                format!(
                    "terminal atlas seam {}->{} is no longer identifiable",
                    transition.from_chart, transition.to_chart
                )
            })?;
        if seam.sign as i8 != transition.sign {
            return Err(format!(
                "terminal atlas seam {}->{} changed orientation ({} -> {})",
                transition.from_chart, transition.to_chart, transition.sign, seam.sign as i8
            ));
        }
        term.refresh_chart_transition(UnitSpeedChartTransition::new(
            transition.from_chart,
            transition.to_chart,
            transition.sign,
            seam.offset,
            seam.period,
            transition.seam_kind,
        )?)?;
    }
    let registered_spheres: Vec<SphereChartTransition> = term
        .chart_atlases()
        .iter()
        .flat_map(|atlas| atlas.sphere_transitions().iter().copied())
        .collect();
    for transition in registered_spheres {
        // `fit_sphere_seam_transition(a,b)` likewise returns the map b -> a.
        let seam = fit_sphere_seam_transition(term, transition.to_chart, transition.from_chart)
            .ok_or_else(|| {
                format!(
                    "terminal sphere atlas seam {}->{} is no longer identifiable",
                    transition.from_chart, transition.to_chart
                )
            })?;
        if seam.seam_kind != transition.seam_kind {
            return Err(format!(
                "terminal sphere atlas seam {}->{} changed kind ({:?} -> {:?})",
                transition.from_chart, transition.to_chart, transition.seam_kind, seam.seam_kind
            ));
        }
        term.refresh_sphere_chart_transition(SphereChartTransition::new(
            transition.from_chart,
            transition.to_chart,
            seam.rotation,
            transition.seam_kind,
        )?)?;
    }
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
    let child = SaeManifoldTerm::new(atoms, assignment)?;

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
    /// Declared reference-function Gram `S_ref` (`m × m`) the atom is seeded with.
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
    // d=1 candidate uses the first column; a d=2 candidate uses the first two.
    // When the caller has only a 1-D seed, this generic builder repeats the seed
    // column; radial promotion overwrites that second coordinate with the
    // standardized log-amplitude below so promoted cylinder/disk candidates carry
    // the radial signal rather than a duplicate angle.
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

    // The candidate basis's declared reference-function Gram, the smoothness
    // operator the topology evidence prices. Every candidate is compared under
    // the same fixed reference measure and second-derivative seminorm
    // `S_ref = Σ_n Σ_{a,c} Φ''_{·,a,c}(t_n)ᵀ Φ''_{·,a,c}(t_n)`, read off its
    // analytic second jet `Φ''[n, μ, a, c]`. A flat (line / patch) basis has a small Gram
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

    // The born atom carries the topology candidate's declared function-space
    // roughness Gram unchanged. Decoder fitting never rewrites that objective.
    let penalty = s_raw.clone();
    Ok(TopologyAutoFitEvidence {
        topology_name: spec.kind.display_name(),
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

fn standardized_log_birth_amplitudes(amps: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let n = amps.len();
    if n == 0 {
        return None;
    }
    let mut logs = Array1::<f64>::zeros(n);
    for (i, &amp) in amps.iter().enumerate() {
        if !amp.is_finite() || amp < 0.0 {
            return None;
        }
        logs[i] = amp.max(f64::MIN_POSITIVE).ln();
    }
    let mean = logs.sum() / n as f64;
    let mut var = 0.0_f64;
    for &value in logs.iter() {
        let centered = value - mean;
        var += centered * centered;
    }
    let std = (var / n as f64).sqrt();
    if !std.is_finite() || std <= 0.0 {
        return None;
    }
    for value in logs.iter_mut() {
        *value = (*value - mean) / std;
    }
    Some(logs)
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
    let log_amp_coord = standardized_log_birth_amplitudes(amps.view())
        .ok_or_else(|| "radial_promoted_specs: degenerate log-amplitude spread".to_string())?;
    // The circle from the d=1 set (the un-promoted alternative the evidence must
    // still be free to prefer) plus the d=2 radial two-manifolds.
    let mut promoted: Vec<TopologyCandidateSpec> = Vec::with_capacity(3);
    for spec in topology_candidates_for_dim(coords, 1)? {
        if spec.kind == AutoTopologyKind::Circle {
            promoted.push(spec);
        }
    }
    for mut spec in topology_candidates_for_dim(coords, 2)? {
        if matches!(
            spec.kind,
            AutoTopologyKind::Cylinder | AutoTopologyKind::Euclidean
        ) {
            for row in 0..spec.coords.nrows() {
                spec.coords[[row, 1]] = log_amp_coord[row];
            }
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
    let r: Vec<f64> = col
        .iter()
        .map(|&t| ((t - lo) / span).clamp(0.0, 1.0))
        .collect();
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

/// A graph birth candidate enrolled in structure search.
///
/// The candidate edge set is the derived anchor-kNN graph; REML per-edge losses
/// decide survival, and the selection currency is the SUM of surviving one-edge
/// charges. Named shapes are only certified compressions of the learned graph.
#[derive(Clone, Debug)]
pub struct GraphBirthCandidate {
    pub atom: LearnedGraphAtom,
    pub selection: GraphStructureSelection,
}

/// Build the graph-atom birth candidate that the structure search scores. This
/// is the graph counterpart to the fixed topology menu: the caller supplies
/// REML per-edge deletion losses for the kNN edge set, and selection is paid in
/// summed edge charge rather than by promoting to a circle/line first.
pub fn graph_birth_candidate_for_structure_search(
    anchor_embeddings: ArrayView2<'_, f64>,
    row_coordinates: &[f64],
    n_eff: f64,
    edge_precisions: &[f64],
    edge_delta_loss: &[f64],
) -> Result<GraphBirthCandidate, String> {
    let atom = LearnedGraphAtom::from_reml_knn_edges(
        anchor_embeddings,
        row_coordinates,
        n_eff,
        edge_precisions,
        edge_delta_loss,
    )?;
    let selection = atom.structure_selection();
    Ok(GraphBirthCandidate { atom, selection })
}

fn race_birth_topology(
    coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    d_k: usize,
) -> Result<Option<TopologyRaceFit>, String> {
    // The PCA/template-coordinate race is the cheaper DEFAULT: the born atom
    // inherits the template atom's coordinate block, and the topology candidates
    // are adjudicated on those linear-seed coordinates.
    let template_winner = race_template_coords(coords, target, weights, d_k)?;
    // Intrinsic-metric CHALLENGER (#2240/#2280): on a FOLDED residual (a swiss
    // roll, a creased sheet) the template coordinates are self-overlapping in the
    // ambient metric, so every topology candidate fits a crumpled image. Re-race
    // the SAME candidate set on the geodesic (Isomap) embedding of the birth
    // image, which unrolls the fold. The challenger enters under the IDENTICAL
    // REML evidence and wins only when it scores strictly better; on a non-fold it
    // ties the linear seed and the default (template) is kept. Fail-safe: any
    // embedding/race failure leaves the template winner untouched.
    //
    // GATED to a FLAT template verdict (EuclideanPatch): a genuinely curved born
    // atom (circle/torus/sphere/cylinder) already wins its specialized chart on the
    // template coords, and re-racing on the geodesic embedding must not let a
    // flexible flat/patch fit override that true topology. Only a flat verdict — the
    // least-bad chart for a folded plane — can be a fold worth unrolling.
    let template_is_sheet = matches!(
        template_winner.as_ref().map(|(fit, _)| &fit.basis_kind),
        Some(SaeAtomBasisKind::EuclideanPatch)
    );
    let intrinsic_winner = if template_is_sheet {
        race_intrinsic_coords(target, weights, d_k).unwrap_or(None)
    } else {
        None
    };
    // Lower TK/REML cost wins (issue #396 sign convention); the template keeps
    // ties, so PCA stays default and intrinsic only supplants it by evidence.
    let winner = match (template_winner, intrinsic_winner) {
        (Some((t_fit, t_score)), Some((i_fit, i_score))) => {
            if i_score < t_score {
                Some(i_fit)
            } else {
                Some(t_fit)
            }
        }
        (Some((t_fit, _)), None) => Some(t_fit),
        (None, Some((i_fit, _))) => Some(i_fit),
        (None, None) => None,
    };
    Ok(winner)
}

/// The PCA/template-coordinate topology race: the historical born-atom path,
/// returning the winning fit AND its TK-normalized evidence so the intrinsic
/// challenger in [`race_birth_topology`] can be compared on the same scale.
fn race_template_coords(
    coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    d_k: usize,
) -> Result<Option<(TopologyRaceFit, f64)>, String> {
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

/// The intrinsic-metric challenger race: embed the birth image `target`
/// (`n × p`) into `d_k` dimensions by classical Landmark-Isomap (geodesic MDS),
/// min-max normalize each geodesic axis to the flat `[-0.5, 0.5]` convention the
/// template coordinates use (so the Euclidean-patch design is scale-commensurable
/// with the template race), and race the SAME topology candidate set on those
/// unfolded coordinates. Returns the winning fit and its TK evidence, or `None`
/// when the embedding is degenerate or no candidate is realizable.
fn race_intrinsic_coords(
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    d_k: usize,
) -> Result<Option<(TopologyRaceFit, f64)>, String> {
    // Folds are a d ≥ 2 story: a 1-D manifold has no ambient fold a geodesic
    // embedding could unroll that a line/circle basis does not already capture,
    // and the geodesic 1-D embedding of a closed loop is degenerate. Restricting
    // the challenger to d ≥ 2 also leaves the d = 1 circle-vs-line race untouched.
    if d_k < 2 || target.nrows() < 3 {
        return Ok(None);
    }
    let embed = crate::manifold::intrinsic_geodesic_embedding(target, d_k)?;
    let n = embed.nrows();
    let d = embed.ncols();
    if n == 0 || d == 0 {
        return Ok(None);
    }
    // Per-axis min-max to [-0.5, 0.5]; a collapsed axis (zero span) means the
    // geodesic embedding found no intrinsic spread there — the challenger is not
    // realizable, so bail and keep the template winner.
    let mut coords = Array2::<f64>::zeros((n, d));
    for col in 0..d {
        let (lo, hi) = (0..n).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
            let v = embed[[r, col]];
            (lo.min(v), hi.max(v))
        });
        let span = hi - lo;
        if !(span > 0.0) || !span.is_finite() {
            return Ok(None);
        }
        for r in 0..n {
            coords[[r, col]] = (embed[[r, col]] - lo) / span - 0.5;
        }
    }
    let specs = topology_candidates_for_dim(coords.view(), d_k)?;
    if specs.is_empty() {
        return Ok(None);
    }
    race_spec_set(specs, target, weights)
}

/// Race one realized candidate spec set against the birth target and return the
/// evidence-winning fit. Shared by the base and the F1 radial-promoted races.
fn race_spec_set(
    specs: Vec<TopologyCandidateSpec>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Option<(TopologyRaceFit, f64)>, String> {
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
                kind.display_name()
            )
        })?;
        fit_topology_candidate(spec, target, weights)
    })?;
    let winner = ranked
        .winner()
        .ok_or_else(|| "race_birth_topology: empty ranking".to_string())?;
    Ok(Some((winner.fit_handle.clone(), winner.tk_score)))
}

/// A primary-atom topology choice discovered by the fit-entry evidence race
/// (#2238/#2239): the basis kind the seed dictionary should build for the atom
/// and the latent dimension that kind carries.
pub struct PrimaryTopologyChoice {
    pub basis_kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    /// Evidence-selected harmonic resolution for a periodic (circle) winner
    /// (#2243): the number of Fourier harmonics the seed circle carries, chosen
    /// by REML marginal likelihood rather than the historical fixed budget.
    /// `None` for every non-periodic kind, whose chart resolution is either a
    /// different knob (a torus winner carries its per-axis order in
    /// `n_torus_harmonics`; a flat/Duchon-sheet winner carries data-scaled
    /// centers in `n_duchon_centers`) or genuinely fixed (the sphere lat/lon
    /// chart and the Möbius double-cover basis are fixed-degree constructions).
    pub n_harmonics: Option<usize>,
    /// Evidence-selected thin-plate center count for a Duchon-sheet winner
    /// (#2240, the #2243 resolution-growth pattern lifted to 2-D): the number
    /// of Duchon centers the seeded sheet should carry, chosen by the same
    /// REML marginal likelihood the topology race scores with. `None` for
    /// every other kind (including a flat `EuclideanPatch` winner, which is
    /// installed as a duchon seed at the builder's default center budget).
    pub n_duchon_centers: Option<usize>,
    /// Evidence-selected per-axis harmonic order for a torus winner (#2243, the
    /// resolution-growth pattern lifted to the tensor-product torus): the number
    /// of Fourier harmonics per circle factor the seeded torus should carry
    /// (basis size `(2H+1)^d`), chosen by the same REML marginal likelihood the
    /// topology race scores with rather than the fixed `SAE_DEFAULT_TORUS_HARMONICS`
    /// budget. `None` for every other kind.
    pub n_torus_harmonics: Option<usize>,
    /// The `(n, latent_dim)` seed chart to install for this atom when the
    /// INTRINSIC-metric seed won the primary race (#2240/#2280). `Some` only when
    /// the geodesic (Isomap) embedding beat the PCA chart on evidence: a
    /// swiss-roll-class fold must be seeded UNFOLDED, so the winning intrinsic
    /// coordinates ride out of the race rather than being re-creased by the linear
    /// PCA seed downstream. `None` when the PCA seed won (the default seed path
    /// already reproduces its chart) — every non-auto atom carries `None`.
    pub coords: Option<Array2<f64>>,
}

/// Per-atom topology discovery for the PRIMARY seed dictionary (#2238/#2239).
///
/// [`race_birth_topology`] adjudicates topology by evidence, but it only ever
/// runs on residual births — the K primary atoms created at fit entry kept the
/// pinned default (a 1-D circle), hard-capping every intrinsically 2-D factor
/// at R² ≈ 0.5. This lifts the SAME evidence race to fit entry: each atom
/// races a circle, a torus, a sphere and a flat 2-D patch — every candidate
/// seeded with its own NATURAL chart of the atom's cluster (phase angles for
/// the periodic forms, (lat, lon) for the sphere, standardized principal
/// projections for the patch) so no candidate is handicapped by a chart built
/// for a rival — and the proper REML marginal likelihood picks the winner.
///
/// `labels` assigns each observation to its seed cluster (the same
/// output-energy labels the periodic seed refinement uses); the race for atom
/// `k` weights exactly its cluster's rows. `max_dims[k]` caps the intrinsic
/// dimension enrolled for atom `k`, so `d_atom = 1` keeps the race
/// one-dimensional. Auto discovery is an explicit contract: every requested
/// atom must produce an evidence-backed winner. Invalid inputs, undersupported
/// clusters, or numerical failures are returned to the caller instead of
/// silently substituting a different topology.
pub fn discover_primary_atom_topologies(
    target: ArrayView2<'_, f64>,
    labels: &[usize],
    k_atoms: usize,
    max_dims: &[usize],
) -> Result<Vec<PrimaryTopologyChoice>, String> {
    let n_obs = target.nrows();
    let p_out = target.ncols();
    if labels.len() != n_obs {
        return Err(format!(
            "discover_primary_atom_topologies: labels must have N={n_obs} entries; got {}",
            labels.len()
        ));
    }
    if max_dims.len() != k_atoms {
        return Err(format!(
            "discover_primary_atom_topologies: max_dims must have K={k_atoms} entries; got {}",
            max_dims.len()
        ));
    }
    if p_out < 2 {
        return Err(format!(
            "discover_primary_atom_topologies: evidence racing needs at least two output dimensions; got P={p_out}"
        ));
    }
    (0..k_atoms)
        .map(|atom_idx| -> Result<PrimaryTopologyChoice, String> {
            let rows: Vec<usize> =
                (0..n_obs).filter(|&row| labels[row] == atom_idx).collect();
            // Too few rows to score a 2-candidate race honestly.
            if rows.len() < 16 {
                return Err(format!(
                    "discover_primary_atom_topologies: auto atom {atom_idx} has only {} seed-cluster rows; at least 16 are required for an evidence race (name an explicit topology when discovery is not identifiable)",
                    rows.len()
                ));
            }
            // Cluster-local principal frame: up to 4 components of the atom's
            // rows, then every observation projected into that frame (the race
            // weights select the cluster; out-of-cluster rows carry weight 0).
            let mut mean = vec![0.0_f64; p_out];
            for &row in &rows {
                for col in 0..p_out {
                    mean[col] += target[[row, col]];
                }
            }
            let inv_count = 1.0 / rows.len() as f64;
            for value in &mut mean {
                *value *= inv_count;
            }
            let mut local = Array2::<f64>::zeros((rows.len(), p_out));
            for (out_row, &src_row) in rows.iter().enumerate() {
                for col in 0..p_out {
                    local[[out_row, col]] = target[[src_row, col]] - mean[col];
                }
            }
            let (_u, _s, vt_opt) = local.svd(false, true).map_err(|error| {
                format!(
                    "discover_primary_atom_topologies: SVD failed for auto atom {atom_idx}: {error}"
                )
            })?;
            let vt = vt_opt.ok_or_else(|| {
                format!(
                    "discover_primary_atom_topologies: SVD returned no right-singular frame for auto atom {atom_idx}"
                )
            })?;
            let n_pcs = vt.nrows().min(4);
            if n_pcs < 2 {
                return Err(format!(
                    "discover_primary_atom_topologies: auto atom {atom_idx} has principal rank {n_pcs}; at least two directions are required"
                ));
            }
            let mut proj = Array2::<f64>::zeros((n_obs, n_pcs));
            for row in 0..n_obs {
                for pc in 0..n_pcs {
                    let mut acc = 0.0_f64;
                    for col in 0..p_out {
                        acc += (target[[row, col]] - mean[col]) * vt[[pc, col]];
                    }
                    proj[[row, pc]] = acc;
                }
            }
            // In-cluster standard deviation per component (the projections are
            // already centered at the cluster mean), so the flat patch sees
            // O(1) coordinates.
            let cluster_sd = |pc: usize| -> f64 {
                let mut acc = 0.0_f64;
                for &row in &rows {
                    acc += proj[[row, pc]] * proj[[row, pc]];
                }
                (acc * inv_count).sqrt().max(1e-12)
            };
            let phase = |a: f64, b: f64| -> f64 {
                let frac = b.atan2(a) / std::f64::consts::TAU;
                frac - frac.floor()
            };
            let mut specs: Vec<TopologyCandidateSpec> = Vec::with_capacity(4);
            // Circle: phase of the leading principal pair (unit-period
            // convention, matching the periodic seed refinement). The phase
            // coordinate is retained so that, if the circle wins the topology
            // race, its harmonic RESOLUTION can be selected by evidence (#2243)
            // on the same coordinate the topology race discriminated on.
            let circle_coords = {
                let mut coords = Array2::<f64>::zeros((n_obs, 1));
                for row in 0..n_obs {
                    coords[[row, 0]] = phase(proj[[row, 0]], proj[[row, 1]]);
                }
                let n_harmonics = 3;
                let evaluator = PeriodicHarmonicEvaluator::new(n_harmonics).map_err(|error| {
                    format!(
                        "discover_primary_atom_topologies: circle evaluator failed for auto atom {atom_idx}: {error}"
                    )
                })?;
                specs.push(TopologyCandidateSpec {
                    kind: AutoTopologyKind::Circle,
                    basis_kind: SaeAtomBasisKind::Periodic,
                    manifold: LatentManifold::Circle { period: 1.0 },
                    latent_dim: 1,
                    evaluator: Arc::new(evaluator),
                    coords: coords.clone(),
                });
                coords
            };
            let mut sheet_coords: Option<Array2<f64>> = None;
            let mut torus_coords: Option<Array2<f64>> = None;
            if max_dims[atom_idx] >= 2 {
                // Flat 2-D patch: standardized leading principal projections.
                let (sd0, sd1) = (cluster_sd(0), cluster_sd(1));
                let mut coords = Array2::<f64>::zeros((n_obs, 2));
                for row in 0..n_obs {
                    coords[[row, 0]] = proj[[row, 0]] / sd0;
                    coords[[row, 1]] = proj[[row, 1]] / sd1;
                }
                let evaluator = EuclideanPatchEvaluator::new(2, 2).map_err(|error| {
                    format!(
                        "discover_primary_atom_topologies: flat evaluator failed for auto atom {atom_idx}: {error}"
                    )
                })?;
                specs.push(TopologyCandidateSpec {
                    kind: AutoTopologyKind::Euclidean,
                    basis_kind: SaeAtomBasisKind::EuclideanPatch,
                    manifold: LatentManifold::Euclidean,
                    latent_dim: 2,
                    evaluator: Arc::new(evaluator),
                    coords: coords.clone(),
                });
                // #2240 — flexible thin-plate (Duchon) sheet over the SAME
                // standardized 2-PC chart, with adaptive in-cluster centers:
                // the rich 2-D candidate for swiss-roll-class factors a
                // degree-2 patch cannot follow. It races as its OWN kind
                // (`DuchonSheet`): it is not a fixed constant-curvature form,
                // so the #944 Euclidean/Sphere fusion cannot absorb it —
                // without it, any race that also carried a sphere candidate
                // fused the flat patch away and left a rolled sheet NO
                // admissible chart at all. A cluster too small to identify the
                // thin-plate nullspace simply skips the candidate (the flat
                // patch above stays as the sheet fallback).
                if let Some(centers) =
                    duchon_sheet_centers(&coords, &rows, duchon_sheet_race_center_budget(rows.len()))
                {
                    let evaluator = DuchonCoordinateEvaluator::new(centers, DUCHON_SHEET_M)
                        .map_err(|error| {
                            format!(
                                "discover_primary_atom_topologies: duchon-sheet evaluator failed for auto atom {atom_idx}: {error}"
                            )
                        })?;
                    specs.push(TopologyCandidateSpec {
                        kind: AutoTopologyKind::DuchonSheet,
                        basis_kind: SaeAtomBasisKind::Duchon,
                        manifold: LatentManifold::Euclidean,
                        latent_dim: 2,
                        evaluator: Arc::new(evaluator),
                        coords: coords.clone(),
                    });
                }
                sheet_coords = Some(coords);
                if n_pcs >= 3 {
                    // Sphere: (lat, lon) of the unit-normalized leading 3-frame.
                    let mut coords = Array2::<f64>::zeros((n_obs, 2));
                    for row in 0..n_obs {
                        let (x, y, z) = (proj[[row, 0]], proj[[row, 1]], proj[[row, 2]]);
                        let norm = (x * x + y * y + z * z).sqrt().max(1e-12);
                        coords[[row, 0]] = (z / norm).clamp(-1.0, 1.0).asin();
                        coords[[row, 1]] = y.atan2(x);
                    }
                    specs.push(TopologyCandidateSpec {
                        kind: AutoTopologyKind::Sphere,
                        basis_kind: SaeAtomBasisKind::Sphere,
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
                        coords,
                    });
                }
                if n_pcs >= 3 {
                    // Möbius band (#2240): recover one fundamental domain of the
                    // period-two double cover plus the SIGNED band width from the
                    // radial/transverse half-angle vector. The deck-invariant basis
                    // makes width-odd structure carry half-period angular factors —
                    // the non-orientable signature no other candidate can express.
                    if let (Ok(coords), Ok(evaluator)) = (
                        crate::manifold::mobius_double_cover_coords_from_projection(
                            proj.view(),
                            &rows,
                        ),
                        MobiusHarmonicEvaluator::new(3, 2),
                    ) {
                        specs.push(TopologyCandidateSpec {
                            kind: AutoTopologyKind::Mobius,
                            basis_kind: SaeAtomBasisKind::Mobius,
                            manifold: LatentManifold::Product(vec![
                                LatentManifold::Circle { period: 2.0 },
                                LatentManifold::Interval { lo: -1.0, hi: 1.0 },
                            ]),
                            latent_dim: 2,
                            evaluator: Arc::new(evaluator),
                            coords,
                        });
                    }
                }
                if n_pcs >= 4 {
                    // Torus: independent phases of the two leading principal
                    // pairs (fraction-of-period convention on both axes).
                    let mut coords = Array2::<f64>::zeros((n_obs, 2));
                    for row in 0..n_obs {
                        coords[[row, 0]] = phase(proj[[row, 0]], proj[[row, 1]]);
                        coords[[row, 1]] = phase(proj[[row, 2]], proj[[row, 3]]);
                    }
                    let evaluator = TorusHarmonicEvaluator::new(2, 2).map_err(|error| {
                        format!(
                            "discover_primary_atom_topologies: torus evaluator failed for auto atom {atom_idx}: {error}"
                        )
                    })?;
                    specs.push(TopologyCandidateSpec {
                        kind: AutoTopologyKind::Torus,
                        basis_kind: SaeAtomBasisKind::Torus,
                        manifold: LatentManifold::Product(vec![
                            LatentManifold::Circle { period: 1.0 },
                            LatentManifold::Circle { period: 1.0 },
                        ]),
                        latent_dim: 2,
                        evaluator: Arc::new(evaluator),
                        coords: coords.clone(),
                    });
                    torus_coords = Some(coords);
                }
            }
            if specs.is_empty() {
                return Err(format!(
                    "discover_primary_atom_topologies: auto atom {atom_idx} produced no realizable candidates"
                ));
            }
            let mut weights = Array1::<f64>::zeros(n_obs);
            for &row in &rows {
                weights[row] = 1.0;
            }
            // PCA/linear race is the cheaper DEFAULT.
            let pca_winner = race_spec_set(specs, target, weights.view()).map_err(|error| {
                format!(
                    "discover_primary_atom_topologies: evidence race failed for auto atom {atom_idx}: {error}"
                )
            })?;
            // Intrinsic-metric CHALLENGER (#2240/#2280): re-race the fold-sensitive
            // d=2 candidates on the geodesic embedding of the birth image, which
            // unrolls a swiss-roll-class fold the PCA chart creases. It enters under
            // the SAME REML evidence and wins only by a strictly lower TK cost; a
            // win also swaps in its unfolded chart for the Duchon-center growth.
            // Fail-safe: any embedding/race failure leaves the PCA winner untouched.
            //
            // GATED to a FLAT PCA verdict (EuclideanPatch/Duchon): the challenger
            // only ever offers the raw-coordinate flat + thin-plate charts, and a
            // thin-plate sheet is flexible enough to out-fit a SPECIALIZED curved
            // chart (sphere/torus/circle) on evidence even when that curved chart is
            // the true topology (the #2238/#2239 contract). A genuinely curved factor
            // gets a curved PCA winner; only a SHEET verdict (flat/Duchon — the
            // least-bad chart for a folded plane) can be a fold worth unrolling. So
            // the challenger runs iff PCA already said "sheet", never overriding a
            // discovered sphere/torus/circle.
            let pca_is_sheet = matches!(
                pca_winner.as_ref().map(|(fit, _)| &fit.basis_kind),
                Some(SaeAtomBasisKind::EuclideanPatch) | Some(SaeAtomBasisKind::Duchon)
            );
            let intrinsic_challenger = if pca_is_sheet {
                match build_intrinsic_primary_specs(target, &rows, max_dims[atom_idx]) {
                    Ok(Some((int_specs, int_chart))) => {
                        match race_spec_set(int_specs, target, weights.view()) {
                            Ok(Some((int_fit, int_score))) => Some((int_fit, int_score, int_chart)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
            // When the intrinsic seed wins, its unfolded chart both drives the
            // Duchon-center growth (`sheet_coords`) AND must be installed as the
            // atom's final seed coordinates (`winning_intrinsic_chart`) so the
            // downstream PCA seed does not re-crease the fold.
            let mut winning_intrinsic_chart: Option<Array2<f64>> = None;
            let fit = match (pca_winner, intrinsic_challenger) {
                (Some((p_fit, p_score)), Some((i_fit, i_score, i_chart))) => {
                    if i_score < p_score {
                        sheet_coords = Some(i_chart.clone());
                        winning_intrinsic_chart = Some(i_chart);
                        i_fit
                    } else {
                        p_fit
                    }
                }
                (Some((p_fit, _)), None) => p_fit,
                (None, Some((i_fit, _, i_chart))) => {
                    sheet_coords = Some(i_chart.clone());
                    winning_intrinsic_chart = Some(i_chart);
                    i_fit
                }
                (None, None) => {
                    return Err(format!(
                        "discover_primary_atom_topologies: evidence race returned no winner for auto atom {atom_idx}"
                    ));
                }
            };
            // #2243 — for a circle winner, GROW the harmonic resolution by the
            // same REML evidence: the topology race ran the circle at a fixed low
            // budget only to discriminate topology, but a genuinely 1-D factor's
            // fidelity is capped by that budget. Every other kind carries a chart
            // whose resolution is not a harmonic count, so it selects none.
            let n_harmonics = if fit.basis_kind == SaeAtomBasisKind::Periodic {
                Some(select_periodic_resolution(
                    circle_coords.view(),
                    target,
                    weights.view(),
                    rows.len(),
                )?)
            } else {
                None
            };
            // #2240 — for a Duchon-sheet winner, GROW the center count by the
            // same REML evidence (the #2243 pattern lifted from harmonics to
            // thin-plate centers): the race ran the sheet at the seed-economy
            // budget only to discriminate topology; a tightly rolled sheet's
            // fidelity is capped by that budget.
            let n_duchon_centers = if fit.basis_kind == SaeAtomBasisKind::Duchon {
                let coords = sheet_coords.as_ref().ok_or_else(|| {
                    format!(
                        "discover_primary_atom_topologies: duchon-sheet winner without a 2-D chart for auto atom {atom_idx}"
                    )
                })?;
                Some(select_duchon_sheet_resolution(
                    coords,
                    target,
                    weights.view(),
                    &rows,
                )?)
            } else {
                None
            };
            // #2243 — for a torus winner, GROW the per-axis harmonic order by
            // the same REML evidence (the circle pattern lifted to the tensor-
            // product torus): the race ran the torus at a fixed low order only
            // to discriminate topology, but a genuinely toroidal factor with
            // high-frequency angular content on either circle factor is capped
            // by that order.
            let n_torus_harmonics = if fit.basis_kind == SaeAtomBasisKind::Torus {
                let coords = torus_coords.as_ref().ok_or_else(|| {
                    format!(
                        "discover_primary_atom_topologies: torus winner without a 2-D chart for auto atom {atom_idx}"
                    )
                })?;
                Some(select_torus_resolution(
                    coords.view(),
                    target,
                    weights.view(),
                    rows.len(),
                )?)
            } else {
                None
            };
            // Install the winning intrinsic chart, projected to the resolved
            // latent dimension, only when the intrinsic seed actually won.
            let coords = winning_intrinsic_chart.map(|chart| {
                let d = fit.latent_dim.min(chart.ncols());
                let mut out = Array2::<f64>::zeros((chart.nrows(), fit.latent_dim));
                for row in 0..chart.nrows() {
                    for col in 0..d {
                        out[[row, col]] = chart[[row, col]];
                    }
                }
                out
            });
            Ok(PrimaryTopologyChoice {
                basis_kind: fit.basis_kind,
                latent_dim: fit.latent_dim,
                n_harmonics,
                n_duchon_centers,
                n_torus_harmonics,
                coords,
            })
        })
        .collect()
}

/// Intrinsic-metric CHALLENGER spec set for the primary discovery race
/// (#2240/#2280). On a FOLDED residual factor (a swiss roll, a creased sheet) the
/// PCA 2-PC chart the primary race builds is self-overlapping, so even the
/// flexible Duchon sheet fits a crumpled image. This embeds the birth image
/// `target` by its geodesic (Isomap) metric — which unrolls the fold — standardizes
/// each intrinsic axis to unit in-cluster SD (so the Euclidean-patch / Duchon
/// design is scale-commensurable with the PCA flat candidate), and offers the two
/// FOLD-SENSITIVE `d = 2` candidates (a flat patch and a thin-plate sheet) on those
/// unfolded coordinates. The caller races this set against the PCA set under the
/// SAME REML evidence and keeps the PCA winner on ties, so the intrinsic seed
/// supplants the linear one only when its unfolding earns strictly higher evidence.
///
/// Returns `(specs, chart)` where `chart` is the standardized intrinsic 2-D chart
/// the winning sheet's resolution growth reads. `None` when the atom is `d < 2`
/// (folds are a sheet story; `d = 1` line/circle is served by the PCA race), the
/// cluster is too small for a geodesic graph, or the embedding is degenerate.
fn build_intrinsic_primary_specs(
    target: ArrayView2<'_, f64>,
    rows: &[usize],
    max_dim: usize,
) -> Result<Option<(Vec<TopologyCandidateSpec>, Array2<f64>)>, String> {
    if max_dim < 2 || rows.len() < 3 {
        return Ok(None);
    }
    let n_obs = target.nrows();
    let embed = crate::manifold::intrinsic_geodesic_embedding(target, 2)?;
    if embed.ncols() < 2 {
        return Ok(None);
    }
    // Standardize each intrinsic axis to unit in-cluster SD (the PCA flat patch's
    // O(1)-coordinate convention). A collapsed axis (no intrinsic spread) means the
    // geodesic embedding found no second dimension — the challenger is not
    // realizable, so bail and keep the PCA winner.
    let inv_count = 1.0 / rows.len().max(1) as f64;
    let mut coords = Array2::<f64>::zeros((n_obs, 2));
    for col in 0..2 {
        let mut acc = 0.0_f64;
        for &row in rows {
            acc += embed[[row, col]] * embed[[row, col]];
        }
        let sd = (acc * inv_count).sqrt();
        if !(sd > 1e-12) || !sd.is_finite() {
            return Ok(None);
        }
        for row in 0..n_obs {
            coords[[row, col]] = embed[[row, col]] / sd;
        }
    }
    let mut specs: Vec<TopologyCandidateSpec> = Vec::with_capacity(2);
    let flat = EuclideanPatchEvaluator::new(2, 2)
        .map_err(|error| format!("intrinsic primary flat evaluator failed: {error}"))?;
    specs.push(TopologyCandidateSpec {
        kind: AutoTopologyKind::Euclidean,
        basis_kind: SaeAtomBasisKind::EuclideanPatch,
        manifold: LatentManifold::Euclidean,
        latent_dim: 2,
        evaluator: Arc::new(flat),
        coords: coords.clone(),
    });
    if let Some(centers) =
        duchon_sheet_centers(&coords, rows, duchon_sheet_race_center_budget(rows.len()))
    {
        let sheet = DuchonCoordinateEvaluator::new(centers, DUCHON_SHEET_M)
            .map_err(|error| format!("intrinsic primary duchon evaluator failed: {error}"))?;
        specs.push(TopologyCandidateSpec {
            kind: AutoTopologyKind::DuchonSheet,
            basis_kind: SaeAtomBasisKind::Duchon,
            manifold: LatentManifold::Euclidean,
            latent_dim: 2,
            evaluator: Arc::new(sheet),
            coords: coords.clone(),
        });
    }
    Ok(Some((specs, coords)))
}

/// Duchon nullspace order `m` for the raced 2-D thin-plate sheet (#2240) —
/// DERIVED (identity with the seed builder): mirrors the FFI seed path's
/// `sae_duchon_atom_m(dim) = dim/2 + 2`, i.e. `m = 3` (degree-2 polynomial
/// nullspace) at `dim = 2`, so the race scores the same function space the
/// winning seed will build.
const DUCHON_SHEET_M: usize = 3;

/// Polynomial-nullspace dimension of the `DUCHON_SHEET_M` thin-plate sheet in
/// 2-D — DERIVED: monomials of total degree ≤ 2 in two variables, `C(2+2, 2) =
/// 6`. The center count must clear this for the kernel block to have positive
/// rank.
const DUCHON_SHEET_NULLSPACE_DIM: usize = 6;

/// Race-time center budget for the 2-D Duchon-sheet candidate (#2240) —
/// mirrors the seed builder's economy band (`sae_build_atom_plans`: floor
/// `nullspace + d + 1`, dense ceiling 32) so the race scores the exact chart a
/// default seed would build; the evidence ladder
/// ([`select_duchon_sheet_resolution`]) then grows a WINNER past this budget.
/// Returns 0 (no realizable candidate) when the cluster cannot identify the
/// thin-plate nullspace.
fn duchon_sheet_race_center_budget(n_cluster: usize) -> usize {
    let floor = DUCHON_SHEET_NULLSPACE_DIM + 2 + 1;
    if n_cluster <= floor {
        return 0;
    }
    n_cluster.min(32).max(floor)
}

/// Deterministic adaptive centers for the Duchon-sheet candidate: `n_centers`
/// evenly-strided IN-CLUSTER rows of the standardized 2-PC chart, so the
/// thin-plate kernel is anchored where the factor's data actually lies
/// (knot-at-data placement). `None` when the cluster cannot supply the
/// requested count (the candidate is skipped, not degraded).
fn duchon_sheet_centers(
    coords: &Array2<f64>,
    rows: &[usize],
    n_centers: usize,
) -> Option<Array2<f64>> {
    if n_centers == 0 || rows.len() < n_centers {
        return None;
    }
    let mut centers = Array2::<f64>::zeros((n_centers, 2));
    for i in 0..n_centers {
        // Even stride over the cluster's rows; i·len/n is strictly increasing
        // in i for n ≤ len, so the selected rows are distinct.
        let row = rows[i * rows.len() / n_centers];
        centers[[i, 0]] = coords[[row, 0]];
        centers[[i, 1]] = coords[[row, 1]];
    }
    Some(centers)
}

/// Evidence-driven center count for a Duchon-sheet primary winner (#2240 — the
/// #2243 resolution-growth pattern lifted from circle harmonics to thin-plate
/// centers). The topology race scored the sheet at the seed-economy budget
/// only to discriminate topology; a swiss-roll-class factor's fidelity is
/// capped by that budget, so the winner's center count is selected by the SAME
/// proper closed-form REML marginal likelihood the race scores with
/// (`fit_topology_candidate` → `raw_reml`, complexity-priced, lower is
/// better), taking the GLOBAL evidence minimum over a dyadic ladder — the
/// evidence need not be unimodal in resolution.
///
/// The ladder is bounded by two hard, data-derived limits (no tuned
/// resolution constant): the seed-economy floor below and the identifiability
/// ceiling `n_centers < n_cluster` (the Duchon design has one column per
/// center, and a weighted REML cannot be identified with as many columns as
/// the cluster has observations). The ceiling rung is always included so a
/// near-noiseless factor can reach full resolution.
fn select_duchon_sheet_resolution(
    sheet_coords: &Array2<f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    rows: &[usize],
) -> Result<usize, String> {
    let floor = duchon_sheet_race_center_budget(rows.len());
    if floor == 0 {
        return Err(
            "select_duchon_sheet_resolution: cluster too small to identify the thin-plate nullspace"
                .to_string(),
        );
    }
    let ceiling = rows.len().saturating_sub(1).max(floor);
    let mut ladder: Vec<usize> = Vec::new();
    let mut c = floor;
    while c < ceiling {
        ladder.push(c);
        c = c.saturating_mul(2);
    }
    ladder.push(ceiling);
    let mut best_c = 0usize;
    let mut best_score = f64::INFINITY;
    for &n_centers in &ladder {
        let Some(centers) = duchon_sheet_centers(sheet_coords, rows, n_centers) else {
            continue;
        };
        let evaluator = match DuchonCoordinateEvaluator::new(centers, DUCHON_SHEET_M) {
            Ok(evaluator) => evaluator,
            Err(_) => continue,
        };
        let spec = TopologyCandidateSpec {
            kind: AutoTopologyKind::DuchonSheet,
            basis_kind: SaeAtomBasisKind::Duchon,
            manifold: LatentManifold::Euclidean,
            latent_dim: 2,
            evaluator: Arc::new(evaluator),
            coords: sheet_coords.clone(),
        };
        // `raw_reml` is the proper REML evidence (lower is better) on a common
        // `n_obs`, so comparing it directly selects the same resolution the
        // race machinery would (see `select_periodic_resolution`).
        let score = match fit_topology_candidate(&spec, target, weights) {
            Ok(evidence) => evidence.raw_reml,
            Err(_) => continue,
        };
        if score.is_finite() && score < best_score {
            best_score = score;
            best_c = n_centers;
        }
    }
    if best_c == 0 {
        return Err(
            "select_duchon_sheet_resolution: no fittable center count for the duchon-sheet winner"
                .to_string(),
        );
    }
    Ok(best_c)
}

/// Measured spectral noise floor for evidence-driven resolution selection
/// (#2243). A resolution knob is a BANDWIDTH question, not a smoothing one:
/// include every harmonic carrying real, above-noise energy and let the fit's
/// own REML-selected λ shrink the unsupported ones — over-provisioning is
/// harmless (an empty harmonic's roughness penalty drives its coefficient to
/// zero), while under-provisioning structurally caps reconstruction. The floor
/// is measured from the periodogram itself, with no tuned smoothing constant:
///
/// * a numerical-zero guard `peak · 1e-12` — a harmonic that small is
///   indistinguishable from roundoff, never real signal. It dominates on clean
///   (near-noiseless) data, where the median energy collapses to ~0;
/// * a noise-level guard `median · log2(K)` — under band-limited signal the
///   per-harmonic energies are dominated by the noise floor, whose robust
///   center is the median (real harmonics are sparse outliers that do not move
///   it). `log2(K) = ln K / ln 2` is the expected value of the maximum of `K`
///   exponential-tailed noise energies in units of the median, i.e. the
///   Bonferroni expected-one-false-alarm bound over the `K` tested harmonics —
///   a harmonic above it is a genuine spectral outlier, not the largest of `K`
///   noise draws. It grows only logarithmically in the band size, so it stays
///   sensitive to real structure on large clusters. It dominates under real
///   noise.
///
/// This replaces the earlier REML-argmin-over-resolutions ladder, which
/// UNDER-resolved exactly-fittable clean data (the #2243 disease it was meant
/// to cure): the closed-form REML dispersion reward is floored at
/// `MIN_DEVIANCE`, so an exact fit's evidence gain is capped while its
/// complexity term `½d·(log|H| − log|λS|₊)` diverges as the REML λ→0 (needed to
/// admit a high harmonic against its `∝ h⁴` roughness penalty); at modest
/// cluster sizes the complexity term wins and the argmin stops below the real
/// bandwidth. Bandwidth selection prices resolution on the spectrum, where it
/// belongs, and leaves smoothing to the fit's own REML λ.
fn spectral_noise_floor(energies: &[f64], peak_energy: f64) -> f64 {
    let numerical = peak_energy * 1e-12;
    let k = energies.len();
    if k == 0 {
        return numerical;
    }
    let mut sorted: Vec<f64> = energies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if k % 2 == 1 {
        sorted[k / 2]
    } else {
        0.5 * (sorted[k / 2 - 1] + sorted[k / 2])
    };
    let bonferroni = (k as f64).max(2.0).log2();
    numerical.max(median * bonferroni)
}

/// Evidence-driven harmonic resolution for a periodic (circle) primary atom
/// (#2243). The historical seed budget (`2·d_atom + 1` harmonics, i.e. 2
/// harmonics at the default `d_atom = 2`) under-resolves genuinely 1-D factors
/// with real high-frequency content, capping reconstruction below the fidelity
/// the data supports even once the topology is right. The resolution is the
/// weighted angular periodogram's BANDWIDTH — the highest harmonic whose energy
/// clears the measured [`spectral_noise_floor`] — bounded by the
/// identifiability limit `2H + 1 < n_cluster` (the weighted fit cannot be
/// identified with more basis columns than the cluster has observations). A
/// target with no angular energy returns an error (the caller surfaces it as a
/// discovery failure rather than silently pinning a resolution).
fn select_periodic_resolution(
    circle_coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    n_cluster: usize,
) -> Result<usize, String> {
    let n_obs = target.nrows();
    let p_out = target.ncols();
    // 2H + 1 basis columns must stay strictly below the cluster sample count for
    // the weighted fit to be identifiable.
    let ident_ceiling = (n_cluster.saturating_sub(2) / 2).max(1);
    // Weighted angular periodogram energy per harmonic, over the cluster rows the
    // weights select.
    let mut peak_energy = 0.0_f64;
    let mut energies = Vec::with_capacity(ident_ceiling);
    for h in 1..=ident_ceiling {
        let mut energy = 0.0_f64;
        for col in 0..p_out {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for row in 0..n_obs {
                let w = weights[row];
                if w == 0.0 {
                    continue;
                }
                let angle = std::f64::consts::TAU * h as f64 * circle_coords[[row, 0]];
                re += w * target[[row, col]] * angle.cos();
                im += w * target[[row, col]] * angle.sin();
            }
            energy += re * re + im * im;
        }
        peak_energy = peak_energy.max(energy);
        energies.push(energy);
    }
    if !(peak_energy > 0.0) {
        return Err(
            "select_periodic_resolution: the circle winner carries no angular energy".to_string(),
        );
    }
    let floor = spectral_noise_floor(&energies, peak_energy);
    let bandwidth = energies
        .iter()
        .rposition(|&energy| energy > floor)
        .map(|idx| idx + 1)
        .unwrap_or(1);
    Ok(bandwidth.min(ident_ceiling).max(1))
}

/// Evidence-driven per-axis harmonic order for a torus primary winner (#2243 —
/// the circle resolution-growth pattern lifted to the tensor-product torus).
/// The historical fixed order (`SAE_DEFAULT_TORUS_HARMONICS = 3`) under-resolves
/// a genuinely toroidal factor whose angular content on either circle factor
/// runs above third order, capping reconstruction below the fidelity the data
/// supports even once the topology is right. The per-axis order is the joint
/// angular periodogram's BANDWIDTH — the largest per-axis order of any joint
/// harmonic `(h₀, h₁)` whose energy clears the measured [`spectral_noise_floor`]
/// — bounded by the identifiability limit `(2H + 1)^2 < n_cluster` (the
/// tensor-product design has `(2H+1)^2` columns, and the weighted fit cannot be
/// identified with more columns than the cluster has observations) intersected
/// with the seed builder's dense guard `(2H+1)^2 ≤ 4·SAE_MAX_PERIODIC_HARMONICS`,
/// so the selected order always builds. Same spectral criterion as the circle
/// (see [`select_periodic_resolution`]): over-provisioning is smoothed away by
/// the fit's own REML λ. A target with no angular energy returns an error (the
/// caller surfaces it as a discovery failure rather than silently pinning a
/// resolution).
fn select_torus_resolution(
    torus_coords: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    n_cluster: usize,
) -> Result<usize, String> {
    let n_obs = target.nrows();
    let p_out = target.ncols();
    // Solve (2H+1)^2 < n_cluster for the identifiability ceiling on the per-axis
    // order, intersected with the seed builder's dense guard.
    let axis_ceiling = |limit: f64| -> usize {
        let root = limit.sqrt();
        if root <= 1.0 {
            1
        } else {
            (((root - 1.0) / 2.0).floor() as usize).max(1)
        }
    };
    let ident_ceiling = axis_ceiling(n_cluster as f64);
    let dense_ceiling = axis_ceiling((SAE_MAX_PERIODIC_HARMONICS * 4) as f64);
    let hard_ceiling = ident_ceiling.min(dense_ceiling).max(1);
    // Weighted joint angular periodogram over the bounded (h₀, h₁) grid; keep
    // the per-axis order of every cell so the bandwidth prune can be resolved
    // once the peak is known.
    let mut peak_energy = 0.0_f64;
    let mut cells: Vec<(usize, f64)> = Vec::new();
    for h0 in 0..=hard_ceiling {
        for h1 in 0..=hard_ceiling {
            if h0 == 0 && h1 == 0 {
                continue;
            }
            let mut energy = 0.0_f64;
            for col in 0..p_out {
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for row in 0..n_obs {
                    let w = weights[row];
                    if w == 0.0 {
                        continue;
                    }
                    let angle = std::f64::consts::TAU
                        * (h0 as f64 * torus_coords[[row, 0]] + h1 as f64 * torus_coords[[row, 1]]);
                    re += w * target[[row, col]] * angle.cos();
                    im += w * target[[row, col]] * angle.sin();
                }
                energy += re * re + im * im;
            }
            peak_energy = peak_energy.max(energy);
            cells.push((h0.max(h1), energy));
        }
    }
    if !(peak_energy > 0.0) {
        return Err(
            "select_torus_resolution: the torus winner carries no angular energy".to_string(),
        );
    }
    // Per-axis resolution = the joint periodogram's bandwidth (the largest
    // per-axis order of any cell clearing the measured noise floor), bounded by
    // the identifiability/dense ceiling. Same spectral-bandwidth criterion as
    // the circle (see [`spectral_noise_floor`] and [`select_periodic_resolution`]):
    // over-provisioning is smoothed away by the fit's own REML λ, so this prices
    // resolution on the spectrum rather than under-resolving via a REML argmin.
    let cell_energies: Vec<f64> = cells.iter().map(|(_, energy)| *energy).collect();
    let floor = spectral_noise_floor(&cell_energies, peak_energy);
    let bandwidth = cells
        .iter()
        .filter(|(_, energy)| *energy > floor)
        .map(|(order, _)| *order)
        .max()
        .unwrap_or(1);
    Ok(bandwidth.min(hard_ceiling).max(1))
}

/// Resolve every `"auto"` entry of a primary seed dictionary to the concrete
/// basis-kind string + latent dimension the fit-entry evidence race selects
/// (#2238/#2239). This is the SINGLE place the auto policy lives — the FFI
/// layer only plumbs arrays through (SPEC: pyffi stays thin). Policy:
///
/// * torus / sphere winners keep their kind and carry `latent_dim = 2`;
/// * a flat 2-D winner builds the expressive thin-plate (`duchon`) chart
///   rather than the degree-2 patch the race scored with — same topology,
///   strictly richer basis for the seeded atom;
/// * a Duchon-sheet winner (#2240, the rich swiss-roll-class chart raced as
///   its own candidate) installs as `duchon` and carries its evidence-selected
///   center count in the returned per-atom override vector, so the seed
///   builder grows the thin-plate resolution REML picked rather than its
///   fixed economy budget;
/// * a circle winner carries the harmonic resolution the fit-entry evidence
///   race selected (#2243), installed as the periodic atom's `d_atom` (the seed
///   builder's harmonic-count knob) so discovery grows resolution rather than
///   pinning the caller's default budget;
/// * any discovery failure is returned. Auto mode never silently substitutes
///   the old periodic default; callers that require a fixed topology must name
///   that topology explicitly.
///
/// Returns `(resolution_overrides, coord_overrides)`, both aligned with
/// `atom_basis`. `resolution_overrides[k]` is the per-atom basis-native
/// resolution knob (`None` unless evidence-grown), interpreted per the resolved
/// basis kind — Duchon center count for a flat/Duchon-sheet winner (#2240),
/// per-axis harmonic order for a torus winner (#2243). `coord_overrides[k]` is the
/// UNFOLDED geodesic seed chart to install when the intrinsic-metric seed won the
/// primary race (#2280), `None` when the PCA seed won (the common path) — the
/// caller overrides that atom's coordinate block with it so a folded factor is
/// seeded unrolled rather than re-creased by the linear PCA seed.
pub fn resolve_auto_primary_atoms(
    target: ArrayView2<'_, f64>,
    labels: &[usize],
    atom_basis: &mut [String],
    atom_dim: &mut [usize],
) -> Result<(Vec<Option<usize>>, Vec<Option<Array2<f64>>>), String> {
    let k_atoms = atom_basis.len();
    if atom_dim.len() != k_atoms {
        return Err(format!(
            "resolve_auto_primary_atoms: atom_basis and atom_dim must both have K={k_atoms} entries; atom_dim has {}",
            atom_dim.len()
        ));
    }
    let mut resolution_overrides: Vec<Option<usize>> = vec![None; k_atoms];
    // Per-atom seed-chart overrides: `Some` only where the intrinsic-metric seed
    // won the primary race (#2240/#2280), so the caller installs the UNFOLDED
    // geodesic chart instead of the PCA seed for that atom.
    let mut coord_overrides: Vec<Option<Array2<f64>>> = vec![None; k_atoms];
    if !atom_basis.iter().any(|basis| basis == "auto") {
        return Ok((resolution_overrides, coord_overrides));
    }
    let choices = discover_primary_atom_topologies(target, labels, k_atoms, atom_dim)?;
    for atom_idx in 0..k_atoms {
        if atom_basis[atom_idx] != "auto" {
            continue;
        }
        let choice = &choices[atom_idx];
        match choice.basis_kind {
            SaeAtomBasisKind::Torus => {
                // #2243 — the latent dimension stays the manifold dimension; the
                // evidence-selected per-axis harmonic order rides the resolution
                // override so the seed builder grows the torus past its fixed
                // `SAE_DEFAULT_TORUS_HARMONICS` budget. `None` cannot occur for a
                // torus winner (discovery always selects an order), but a missing
                // value conservatively leaves the builder's default.
                atom_basis[atom_idx] = "torus".to_string();
                atom_dim[atom_idx] = choice.latent_dim;
                resolution_overrides[atom_idx] = choice.n_torus_harmonics;
            }
            SaeAtomBasisKind::Sphere => {
                atom_basis[atom_idx] = "sphere".to_string();
                atom_dim[atom_idx] = choice.latent_dim;
            }
            SaeAtomBasisKind::Mobius => {
                atom_basis[atom_idx] = "mobius".to_string();
                atom_dim[atom_idx] = choice.latent_dim;
            }
            SaeAtomBasisKind::EuclideanPatch => {
                atom_basis[atom_idx] = "duchon".to_string();
                atom_dim[atom_idx] = choice.latent_dim;
                // If the intrinsic seed won, install its unfolded chart (#2280).
                coord_overrides[atom_idx] = choice.coords.clone();
            }
            SaeAtomBasisKind::Duchon => {
                // #2240 — the rich thin-plate sheet won the race outright.
                // Install as a duchon seed and carry the evidence-selected
                // center count so the seed builder grows the resolution REML
                // picked (the #2243 pattern in 2-D).
                atom_basis[atom_idx] = "duchon".to_string();
                atom_dim[atom_idx] = choice.latent_dim;
                resolution_overrides[atom_idx] = choice.n_duchon_centers;
                // If the intrinsic seed won, install its unfolded chart (#2280)
                // so the thin-plate sheet is anchored on the unrolled fold.
                coord_overrides[atom_idx] = choice.coords.clone();
            }
            SaeAtomBasisKind::Periodic => {
                atom_basis[atom_idx] = "periodic".to_string();
                // #2243 — install the evidence-selected harmonic resolution as
                // the periodic atom's `d_atom` (the seed builder routes `d_atom`
                // into the Fourier harmonic count for a periodic basis), so the
                // seeded circle carries the resolution REML picked rather than
                // the caller's default budget. `None` cannot occur for a
                // periodic winner (discovery always selects a resolution), but a
                // missing value conservatively leaves the caller's budget.
                if let Some(n_harmonics) = choice.n_harmonics {
                    atom_dim[atom_idx] = n_harmonics;
                }
            }
            ref unexpected => {
                return Err(format!(
                    "resolve_auto_primary_atoms: evidence race selected unsupported primary basis {unexpected:?} for auto atom {atom_idx}"
                ));
            }
        }
    }
    Ok((resolution_overrides, coord_overrides))
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
            // basis: its evaluator, penalized decoder, and declared reference
            // roughness. Decoder fitting does not redefine that seminorm.
            let reference_roughness = if matches!(fit.basis_kind, SaeAtomBasisKind::Poincare) {
                SaeReferenceRoughness::PoincareConformalDirichlet {
                    reference_coords: fit.coords.clone(),
                }
            } else {
                SaeReferenceRoughness::ProvidedFunctionGram(fit.penalty.clone())
            };
            let atom = SaeManifoldAtom::new(
                format!("atom_born_{k}"),
                fit.basis_kind.clone(),
                fit.latent_dim,
                fit.phi.clone(),
                fit.jet.clone(),
                fit.decoder.clone(),
                reference_roughness,
            )?
            .with_basis_second_jet(fit.evaluator.clone());
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
            // dim, basis values + Jacobian + reference Gram); only its decoder carries
            // the residual-factor direction.
            let mut atom = template.clone();
            atom.decoder_coefficients = factor_dir.to_owned();
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
    let born = SaeManifoldAtom::new_with_provided_function_gram(
        format!("atom_born_{k}"),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        harmonic_decoder,
        Array2::<f64>::eye(m),
    )?
    .with_basis_second_jet(evaluator.clone());

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
        // (−4) is fatal under ordered Beta--Bernoulli — the born circle starts nearly OFF (σ(−4)≈0.018) and
        // the sub-fit collapses it (measured: ordered_beta_bernoulli logit −4 collapses ‖B‖ 1.41→1e-4,
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
    let child = SaeManifoldTerm::new(atoms, assignment)?;

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
    /// The per-round move stream folded into the ONE unified accounting currency
    /// ([`SaeMigrationLedger`], sae-unification Increment 3): every adjudicated
    /// birth / death / refusal priced in the shared `dl_bits` description-length
    /// unit (the e-process `log_e` banked as bits). This is a read-out of the
    /// e-process verdicts in [`Self::rounds`] — the e-BH gating is untouched and
    /// still owns acceptance — and carries the `pc_reseed_events == 0` invariant
    /// (structure births seed from the residual-factor pool, never a PC).
    pub migration: SaeMigrationLedger,
}

impl StructureSearchResult {
    /// Assemble a result from the fitted term/ρ and the per-round e-process
    /// ledgers, folding those rounds into the unified [`SaeMigrationLedger`]
    /// currency ([`Self::migration`]) in one place so producers cannot drift.
    #[must_use]
    pub fn from_rounds(
        term: SaeManifoldTerm,
        rho: SaeManifoldRho,
        rounds: Vec<SearchLedger>,
    ) -> Self {
        Self::from_rounds_with_predictions(term, rho, rounds, &[])
    }

    /// Assemble a result AND thread the #2233 closed-form birth pre-screen
    /// predictions into the unified ledger. `birth_predictions[i]` maps a birth
    /// candidate index to its predicted ΔMDL (bits) for round `i` (parallel to
    /// `rounds`; a missing / short entry is treated as "no predictions"), so each
    /// proposed residual-factor birth's `predicted_dl_bits` sits on the same
    /// migration record its post-refit verdict fills in.
    #[must_use]
    pub fn from_rounds_with_predictions(
        term: SaeManifoldTerm,
        rho: SaeManifoldRho,
        rounds: Vec<SearchLedger>,
        birth_predictions: &[std::collections::HashMap<usize, f64>],
    ) -> Self {
        let mut migration = SaeMigrationLedger::new();
        let empty = std::collections::HashMap::new();
        for (round_idx, round_ledger) in rounds.iter().enumerate() {
            let preds = birth_predictions.get(round_idx).unwrap_or(&empty);
            migration.record_search_round(round_idx, round_ledger, preds);
        }
        Self {
            term,
            rho,
            rounds,
            migration,
        }
    }

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
/// e-gate's budget/level and the per-round harvest breadth.
/// Bundled so the driver entry points stay below the argument-count threshold
/// and so a caller configures one object rather than a positional argument
/// cascade.
#[derive(Clone, Copy, Debug)]
pub struct RoundDriverConfig {
    /// Number of held-out evaluation shards the gate streams over.
    pub n_shards: usize,
    /// Move budget + α the e-gates certify at (fixed for the run).
    pub budget: MoveBudget,
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
/// only when a round's ledger contains no applied move (every record is
/// contested / vetoed / deduplicated / deferred / stale).
///
/// `candidate_fit` is the warm refit: given a RESTRUCTURED candidate term + ρ,
/// it refits the candidate on the ESTIMATION rows only (held-out evaluation rows
/// carry weight `0`), so the candidate is the predictable plug-in the e-process
/// evaluates on the held-out shard stream. A non-converged candidate has no
/// valid likelihood score and therefore aborts the search. The shard
/// fold is a no-op: the candidate is fixed across the stream (a predictable
/// plug-in), and each shard contributes its held-out reconstruction
/// likelihood-ratio against the null state that `null_fit` independently refits
/// on that shard to obtain the honest constrained supremum.
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
    ) -> Result<(SaeManifoldTerm, SaeManifoldRho), String>,
    mut null_fit: impl FnMut(
        SaeManifoldTerm,
        SaeManifoldRho,
        &[usize],
    ) -> Result<(SaeManifoldTerm, SaeManifoldRho), String>,
    mut finalize_round: impl FnMut(
        SaeManifoldTerm,
        SaeManifoldRho,
        &[usize],
    ) -> Result<(SaeManifoldTerm, SaeManifoldRho), String>,
) -> Result<StructureSearchResult, String> {
    let RoundDriverConfig {
        n_shards,
        budget,
        harvest_params,
        curl,
    } = config;
    let split = estimation_eval_split(target, n_shards);
    let mut rounds: Vec<SearchLedger> = Vec::new();
    // #2233: per-round birth-pre-screen predictions (candidate index → predicted
    // ΔMDL bits), pushed in lock-step with `rounds` so the unified ledger can pair
    // each proposed birth's prediction with its post-refit verdict.
    let mut round_predictions: Vec<std::collections::HashMap<usize, f64>> = Vec::new();
    // Hysteresis ledger for the curl/flatten pair — persists across rounds so a
    // just-curled atom-set (or just-flattened one) is silenced for a few rounds
    // and the two moves cannot chase each other (INTEGRATION_PLAN risk #5).
    let mut cooldown = crate::manifold::CurlCooldownLedger::new();

    loop {
        // Harvest from the current fitted state. Residuals R = target − fitted.
        let fitted = term.try_fitted_target_aware(target, None)?;
        let residuals = &target.to_owned() - &fitted;
        let mut report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params)?;
        // Capture the pre-screen predictions before `report.proposals` is consumed
        // by the search; curl births (appended below) carry no prediction.
        let birth_predictions: std::collections::HashMap<usize, f64> =
            report.birth_predictions.iter().copied().collect();

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
        let mut birth_seeds: Vec<BirthSeed> = residual_decoders
            .into_iter()
            .map(BirthSeed::ResidualFactor)
            .collect();

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
                report.proposals.push(proposal(
                    &term,
                    StructureMove::Birth { candidate },
                    cand.net_evidence,
                ));
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
            round_predictions.push(birth_predictions);
            break;
        }

        // The search state threads (term, rho) together. Numerical moves are
        // restructured and refit on the estimation rows so they are predictable
        // plug-ins for held-out scoring. Glue is different: its trigger already
        // IS the sample-split equivalence e-value, and the engine never consults
        // a fit-improvement score for that arm. Keep its state index-stable and
        // materialize the harvest-certified transition exactly once at the round
        // boundary after the proposal chain is complete.
        type State = (SaeManifoldTerm, SaeManifoldRho);
        let collapse_events = term.collapse_events().to_vec();
        let decoders = birth_seeds;
        let estimation_rows = split.estimation_rows.clone();
        let certified_glues = std::mem::take(&mut report.certified_glues);
        let proposals = std::mem::take(&mut report.proposals);
        let outcome: SearchOutcome<State> = search(
            (term, rho),
            proposals,
            &split.shards,
            &budget,
            ledger,
            |state: &State, mv: &StructureMove| {
                if matches!(mv, StructureMove::Glue { .. }) {
                    return Ok(state.clone());
                }
                let (cand_term, cand_rho) =
                    apply_structure_move_seeded(&state.0, &state.1, mv, &decoders)?;
                // Refit the restructured candidate on the estimation rows only.
                candidate_fit(cand_term, cand_rho, &estimation_rows)
            },
            |state: &State, shard: &RowBlockShard| eval_log_lik(&state.0, shard),
            |state: &State, shard: &RowBlockShard| {
                let (null_term, _null_rho) =
                    null_fit(state.0.clone(), state.1.clone(), &shard.rows)?;
                eval_log_lik(&null_term, shard)
            },
            // No-op fold: the candidate is the fixed predictable plug-in across
            // the held-out stream.
            |state: State, _: &RowBlockShard| Ok(state),
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
        // A certified Glue is an exact geometric quotient, whether it registers
        // an atlas or physically removes an over-tile: the carried transition
        // transplants the removed chart and the fold preserves routing mass.
        // Re-optimizing a glue-only round would turn an evidence-certified image
        // equivalence into a new numerical fit on only the estimation split.
        // Mixed rounds still polish because their non-Glue moves are numerical.
        let requires_polish = round_ledger.moves.iter().any(|record| {
            let fired = matches!(
                record.verdict,
                gam_solve::structure_search::MoveVerdict::Accepted { .. }
                    | gam_solve::structure_search::MoveVerdict::Demoted { .. }
            );
            fired && !matches!(record.mv, StructureMove::Glue { .. })
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
        round_predictions.push(birth_predictions);

        if applied {
            // #1890 — adopt each accepted harvest certificate exactly once, now
            // that the search's index-sensitive chain is complete. Destructive
            // glues physically compact their partner before polish; atlas glues
            // register an image-exact quotient and retain both numerical charts.
            let (mut next_term, mut next_rho) = (next_term, next_rho);
            compact_glued_atoms(
                &mut next_term,
                &mut next_rho,
                rounds.last().expect("round ledger pushed above"),
                &certified_glues,
            )?;
            if requires_polish {
                // Numerical winners reached their restructured form through the
                // cheap capped scoring refit. Refit at the full inner budget and
                // then refresh any registered regular seams against that terminal
                // numerical state.
                let (mut polished_term, polished_rho) =
                    finalize_round(next_term, next_rho, &split.estimation_rows)?;
                refresh_registered_atlas_transitions(&mut polished_term)?;
                term = polished_term;
                rho = polished_rho;
            } else {
                // A pure certified-Glue round is already terminal: optimizer work
                // here would violate the carried quotient's image-equivalence
                // contract (and would fit only the estimation split after the
                // sample-split verdict).
                term = next_term;
                rho = next_rho;
            }
        } else {
            term = next_term;
            rho = next_rho;
            break;
        }
    }

    // Fold the per-round e-process verdicts into the ONE unified migration
    // currency (Increment 3): a read-out, not a second gate.
    Ok(StructureSearchResult::from_rounds_with_predictions(
        term,
        rho,
        rounds,
        &round_predictions,
    ))
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
    // Propagate a genuine fit failure instead of degrading to "no births".
    // The evidence ladder already includes the rank-0 rung, so a true "no
    // structure to harvest" outcome returns `Ok` (an empty/zero-rank factor);
    // an `Err` here signals a numerical/degenerate failure (non-finite inputs,
    // an empty ladder, a broken alternation), and swallowing it into
    // `Ok(Vec::new())` would silently paper over that non-convergence
    // (the #2069/#2070 accept-on-failure genus). Surface it.
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals,
        activity: activity.view(),
        max_factor_rank: max_rank,
    })
    .map_err(|e| format!("build_birth_decoders: structured-residual fit failed: {e}"))?;
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
    /// `n_eff·½·ln(3R̂²/(π²σ²)) − Δcharge` — the ranking score (NOT a decision).
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
fn power_iter_top_dir(
    img: ArrayView2<'_, f64>,
    center: &Array1<f64>,
    active: &[usize],
) -> Array1<f64> {
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
fn linear_atom_frames(term: &SaeManifoldTerm) -> Vec<(usize, Array1<f64>, Vec<bool>, Array2<f64>)> {
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
    // exactly why it was invisible — the floor keeps the per-row coding gain
    // ½·ln(3R̂²/(π²σ²)) finite and large).
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
            .filter(|&r| {
                di.active.get(r).copied().unwrap_or(false)
                    && dj.active.get(r).copied().unwrap_or(false)
            })
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
        let members: Vec<usize> = di
            .members
            .iter()
            .chain(dj.members.iter())
            .copied()
            .collect();
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
        // Own-presence gate logit: the per-row coding gain ½·ln(3R̂²/(π²σ²)),
        // floored at 0.5 nats so a barely-paying circle still opens its gate for
        // the race to adjudicate. The gain carries the circle shape constant
        // −ln(π/√3) ≈ −0.595 nats/row, so the floor binds for R̂ ≲ 3.3σ (the
        // radius where the gain reaches 0.5).
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
fn eval_log_lik(term: &SaeManifoldTerm, shard: &RowBlockShard) -> Result<f64, String> {
    // The fitted reconstruction at the shard's held-out rows, scored against the
    // full target. The term's per-row routing/basis covers all N rows, so the
    // reconstruction at a held-out row is the model's prediction for it.
    let fitted = term.try_fitted_target_aware(shard.target.view(), None)?;
    let n_full = fitted.nrows();
    let p = fitted.ncols();
    if p != shard.target.ncols() || n_full != shard.target.nrows() {
        return Err(format!(
            "structure-search fitted shape {:?} does not match target {:?}",
            fitted.dim(),
            shard.target.dim()
        ));
    }
    let mut sse = 0.0_f64;
    let mut count = 0usize;
    for &row in &shard.rows {
        if row >= n_full {
            return Err(format!(
                "structure-search evaluation row {row} is out of range for {n_full} rows"
            ));
        }
        for out in 0..p {
            let d = fitted[[row, out]] - shard.target[[row, out]];
            sse_accumulate(&mut sse, d);
        }
        count += p;
    }
    if count == 0 {
        return Err("structure-search evaluation shard must contain rows".to_string());
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

    Ok(reconstruction + gate_evidence?)
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
/// An undefined or non-PD gate block is a fit failure. It cannot be omitted
/// without changing the model-selection scalar, so the error is propagated.
fn gate_block_log_evidence(term: &SaeManifoldTerm, shard: &RowBlockShard) -> Result<f64, String> {
    use gam_solve::inference::pg_gate_evidence::{GateBlock, pg_gate_evidence};

    let logits = &term.assignment.logits;
    let n_full = logits.nrows();
    let k = logits.ncols();
    if k == 0 {
        return Ok(0.0);
    }
    // Restrict to the shard's held-out rows; an empty / out-of-range shard
    // carries no gate evidence.
    if let Some(row) = shard.rows.iter().copied().find(|&row| row >= n_full) {
        return Err(format!(
            "gate-block evidence row {row} is out of range for {n_full} rows"
        ));
    }
    let rows: Vec<usize> = shard.rows.clone();
    let m = rows.len();
    if m == 0 {
        return Err("gate-block evidence shard must contain rows".to_string());
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
                return Err(format!(
                    "gate-block evidence encountered non-finite logit at row {row}, atom {atom}"
                ));
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
        let evidence = pg_gate_evidence(&block)
            .map_err(|error| format!("gate-block evidence failed for atom {atom}: {error}"))?;
        // `neg_log_evidence` is `−log p(gate block)`; the log-likelihood the
        // split-LR consumes is its negation.
        total -= evidence.neg_log_evidence;
    }
    Ok(total)
}

#[inline]
fn sse_accumulate(sse: &mut f64, d: f64) {
    *sse += d * d;
}

/// Inner-fit knobs for the production structure-search refit (the same numbers
/// the outer SAE fit drove its inner Arrow-Schur joint fit with).
#[derive(Clone, Copy, Debug)]
pub struct ProductionRefitParams {
    /// Inner Newton iterations available to every candidate and adopted-state
    /// solve. The solver must certify convergence before the state is scored or
    /// returned; this is a numerical ceiling, not an acceptance criterion.
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
/// repeat to a no-move fixpoint, returning the (possibly restructured) term + ρ
/// and the per-round ledgers (#997).
///
/// The shard refit folds a held-out block into a candidate via the SAME inner
/// joint-fit driver the outer fit used ([`SaeManifoldTerm::run_joint_fit_arrow_schur`]),
/// PENALTY-FREE: the gate's evidence is a held-out reconstruction
/// likelihood-ratio, and the isometry/ARD penalties are gauge/regularization
/// terms that do not belong in the evaluation likelihood. Every candidate and
/// adopted state must reach the inner solver's convergence certificate; a fit
/// error aborts the structure search rather than being converted into a
/// no-improvement score. `ledger` carries banked evidence across rounds so the
/// death veto sees earlier certifications.
pub fn run_production_structure_search(
    term: SaeManifoldTerm,
    rho: SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    config: RoundDriverConfig,
    refit_params: ProductionRefitParams,
    ledger: &mut StructureLedger,
) -> Result<StructureSearchResult, String> {
    let n = target.nrows();
    // Refit a restructured candidate on the ESTIMATION rows only: held-out
    // evaluation rows carry exact zero weight, so the candidate is the
    // predictable plug-in scored on the held-out shards. A non-converged solve
    // is an error; an unfitted candidate is never assigned an evidence score.
    // `full_target` is borrowed by reference so the helper holds no owned capture
    // and can be called from both closures below.
    let refit_at = |full_target: ArrayView2<'_, f64>,
                    mut cand_term: SaeManifoldTerm,
                    mut cand_rho: SaeManifoldRho,
                    estimation_rows: &[usize],
                    inner_max_iter: usize|
     -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
        let mut weights = vec![0.0; n];
        for &r in estimation_rows {
            if r >= n {
                return Err(format!(
                    "structure-search estimation row {r} is out of range for {n} rows"
                ));
            }
            weights[r] = 1.0;
        }
        cand_term.set_row_loss_weights(weights)?;
        cand_term.run_joint_fit_arrow_schur(
            full_target,
            &mut cand_rho,
            None,
            inner_max_iter,
            refit_params.learning_rate,
            refit_params.ridge_ext_coord,
            refit_params.ridge_beta,
        )?;
        Ok((cand_term, cand_rho))
    };
    let full_iters = refit_params.inner_max_iter;
    let full_target_score = target.to_owned();
    let full_target_null = target.to_owned();
    let full_target_polish = target.to_owned();
    let candidate_refit = refit_at;
    let null_refit = refit_at;
    let final_refit = refit_at;
    run_structure_search_rounds(
        term,
        rho,
        target,
        config,
        ledger,
        // Per-candidate scoring refit. It must converge before the held-out
        // likelihood is evaluated.
        move |cand_term, cand_rho, estimation_rows| {
            candidate_refit(
                full_target_score.view(),
                cand_term,
                cand_rho,
                estimation_rows,
                full_iters,
            )
        },
        // Honest constrained null supremum: refit the current K-atom state on
        // exactly the shard being scored. Under-fitting this side would inflate
        // the e-value.
        move |null_term, null_rho, shard_rows| {
            null_refit(
                full_target_null.view(),
                null_term,
                null_rho,
                shard_rows,
                full_iters,
            )
        },
        // Refit each adopted winner on all rows before it becomes the next
        // round's parent or the returned dictionary.
        //
        // #1890 — the polish refits on ALL rows, NOT the held-out estimation
        // split the per-candidate scoring uses. The estimation/eval split is
        // CONTIGUOUS (`estimation_rows = 0..n_est`), and a disjoint-support
        // dictionary (e.g. the over-tiled / orientation-reversing arcs of the
        // chart-glue fixtures) can place an entire atom's support inside the
        // held-out shard. Refitting the polish on estimation-only then weights
        // that atom's rows at ~0, leaving its decoder UNCONSTRAINED: it drifts
        // or blows up (the reversing pin's `try_fitted()` diverges to ~1.9) or is
        // demoted for lack of support (the over-tile partner). The held-out split
        // exists solely so the per-candidate SCORING refit is an honest
        // predictable plug-in for the e-process; the adopted winner's polish is
        // the RETURNED dictionary and must fit every row it will be evaluated on.
        move |adopted_term, adopted_rho, _estimation_rows| {
            let all_rows: Vec<usize> = (0..n).collect();
            final_refit(
                full_target_polish.view(),
                adopted_term,
                adopted_rho,
                &all_rows,
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
        AssignmentMode, PeriodicHarmonicEvaluator, SAE_DEFAULT_TORUS_HARMONICS, SaeAssignment,
        SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
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

    /// #2238/#2239 — `auto` is an evidence-discovery request, not an alias for
    /// the old periodic default. An undersupported cluster must fail loudly and
    /// leave the caller's unresolved state untouched.
    #[test]
    fn auto_primary_topology_never_falls_back_on_race_failure_2238_2239() {
        let target = Array2::<f64>::zeros((15, 2));
        let labels = vec![0usize; 15];
        let mut basis = vec!["auto".to_string()];
        let mut dims = vec![2usize];

        let error = resolve_auto_primary_atoms(target.view(), &labels, &mut basis, &mut dims)
            .expect_err("an undersupported automatic race must be rejected");

        assert!(error.contains("auto atom 0"), "unexpected error: {error}");
        assert!(error.contains("at least 16"), "unexpected error: {error}");
        assert_eq!(basis, vec!["auto"], "failure must not install a fallback");
        assert_eq!(dims, vec![2], "failure must not rewrite latent dimension");
    }

    /// #2238 — a genuinely two-dimensional primary factor must not be pinned to
    /// the old one-dimensional circle. A full 8x8 planar grid is represented
    /// exactly by the flat 2-D candidate, while phase alone discards radius.
    #[test]
    fn auto_primary_topology_selects_two_dimensional_factor_2238() {
        let side = 8usize;
        let target = Array2::<f64>::from_shape_fn((side * side, 2), |(row, col)| {
            let i = row / side;
            let j = row % side;
            if col == 0 {
                i as f64 - 0.5 * (side - 1) as f64
            } else {
                j as f64 - 0.5 * (side - 1) as f64
            }
        });
        let labels = vec![0usize; target.nrows()];
        let choices = discover_primary_atom_topologies(target.view(), &labels, 1, &[2])
            .expect("the supported planar race must produce a winner");

        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].latent_dim, 2);
        assert_eq!(choices[0].basis_kind, SaeAtomBasisKind::EuclideanPatch);
    }

    /// #2238/#2239 — a genuinely CURVED 2-D primary factor (a 2-sphere) must be
    /// discovered by the fit-entry evidence race as a d=2 sphere chart, not
    /// pinned to the 1-D circle default. This is the manifold-zoo plateau and
    /// its fix in one test: the circle chart is a function of longitude alone,
    /// so it structurally discards latitude and caps the recovery near the
    /// observed plateau, while the raced sphere chart reconstructs the planted
    /// factor almost exactly. The race both SELECTS the curved chart (over the
    /// circle and the flat patch) and, fitted, strictly BEATS the circle-pinned
    /// recovery — the two claims the companion issues turn on.
    #[test]
    fn auto_primary_topology_selects_curved_sphere_and_beats_circle_2238_2239() {
        use crate::basis::SphereChartEvaluator;
        use gam_solve::AutoTopologyKind;
        use ndarray::Array1;

        // Deterministic (lat, lon) grid on the OPEN sphere (poles excluded so no
        // chart row is degenerate), embedded as the unit 2-sphere in R³.
        let (n_lat, n_lon) = (12usize, 14usize);
        let n = n_lat * n_lon;
        let mut lat = Vec::with_capacity(n);
        let mut lon = Vec::with_capacity(n);
        let mut target = Array2::<f64>::zeros((n, 3));
        for i in 0..n_lat {
            let theta = -std::f64::consts::FRAC_PI_2
                + std::f64::consts::PI * (i as f64 + 1.0) / (n_lat as f64 + 1.0);
            for j in 0..n_lon {
                let phi = std::f64::consts::TAU * j as f64 / n_lon as f64;
                let row = i * n_lon + j;
                target[[row, 0]] = theta.cos() * phi.cos();
                target[[row, 1]] = theta.cos() * phi.sin();
                target[[row, 2]] = theta.sin();
                lat.push(theta);
                lon.push(phi);
            }
        }

        // The primary-atom race (single cluster) must pick the d=2 sphere chart.
        let labels = vec![0usize; n];
        let choices = discover_primary_atom_topologies(target.view(), &labels, 1, &[2])
            .expect("the supported sphere race must produce a winner");
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0].basis_kind,
            SaeAtomBasisKind::Sphere,
            "the curved 2-sphere factor must be discovered as a sphere chart, not a circle/patch"
        );
        assert_eq!(choices[0].latent_dim, 2, "a sphere is intrinsically 2-D");

        // Reconstruction proof of "beats the circle-pinned recovery": fit the
        // circle chart (longitude only) and the sphere chart (lat, lon) to the
        // SAME planted factor through the same REML candidate fitter the race
        // uses, and compare the explained variance each achieves.
        let weights = Array1::<f64>::ones(n);
        let recon_r2 = |spec: &TopologyCandidateSpec| -> f64 {
            let fit = fit_topology_candidate(spec, target.view(), weights.view())
                .expect("candidate fit")
                .fit_handle;
            let recon = fit.phi.dot(&fit.decoder);
            let mut means = [0.0_f64; 3];
            for col in 0..3 {
                let mut acc = 0.0;
                for row in 0..n {
                    acc += target[[row, col]];
                }
                means[col] = acc / n as f64;
            }
            let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
            for row in 0..n {
                for col in 0..3 {
                    let r = target[[row, col]] - recon[[row, col]];
                    ss_res += r * r;
                    let c = target[[row, col]] - means[col];
                    ss_tot += c * c;
                }
            }
            1.0 - ss_res / ss_tot.max(1e-12)
        };

        let mut circle_coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            circle_coords[[row, 0]] = lon[row] / std::f64::consts::TAU;
        }
        let circle_spec = TopologyCandidateSpec {
            kind: AutoTopologyKind::Circle,
            basis_kind: SaeAtomBasisKind::Periodic,
            manifold: LatentManifold::Circle { period: 1.0 },
            latent_dim: 1,
            evaluator: Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator")),
            coords: circle_coords,
        };

        let mut sphere_coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            sphere_coords[[row, 0]] = lat[row];
            sphere_coords[[row, 1]] = lon[row];
        }
        let sphere_spec = TopologyCandidateSpec {
            kind: AutoTopologyKind::Sphere,
            basis_kind: SaeAtomBasisKind::Sphere,
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
            coords: sphere_coords,
        };

        let circle_r2 = recon_r2(&circle_spec);
        let sphere_r2 = recon_r2(&sphere_spec);
        eprintln!(
            "[topology-2238] planted 2-sphere R²: circle-pinned={circle_r2:.4} sphere-chart={sphere_r2:.4}"
        );
        assert!(
            sphere_r2 > 0.9,
            "the discovered sphere chart must recover the planted 2-sphere (R²={sphere_r2:.4})"
        );
        assert!(
            circle_r2 < 0.75,
            "the 1-D circle default structurally caps the 2-sphere recovery (R²={circle_r2:.4})"
        );
        assert!(
            sphere_r2 > circle_r2 + 0.2,
            "discovery must strictly beat the circle-pinned recovery (sphere={sphere_r2:.4} vs circle={circle_r2:.4})"
        );
    }

    /// DIAGNOSTIC (temporary #2240): fit the PCA-folded Duchon sheet and the
    /// intrinsic-unfolded Duchon sheet directly and print their in-sample REML
    /// scores, to see why the intrinsic challenger loses the race.
    #[test]
    fn diag_swiss_roll_duchon_reml_scores() {
        use ndarray::Array1;
        let (n_t, n_h) = (45usize, 10usize);
        let n = n_t * n_h;
        let mut target = Array2::<f64>::zeros((n, 3));
        for ti in 0..n_t {
            let t = 1.2 * std::f64::consts::PI
                + (3.2 - 1.2) * std::f64::consts::PI * ti as f64 / (n_t - 1) as f64;
            for hi in 0..n_h {
                let h = 10.0 * hi as f64 / (n_h - 1) as f64;
                let row = ti * n_h + hi;
                target[[row, 0]] = t * t.cos();
                target[[row, 1]] = h;
                target[[row, 2]] = t * t.sin();
            }
        }
        let rows: Vec<usize> = (0..n).collect();
        let weights = Array1::<f64>::ones(n);

        // PCA standardized 2-PC chart, exactly as discover_primary_atom_topologies.
        let mut mean = vec![0.0_f64; 3];
        for &row in &rows {
            for col in 0..3 {
                mean[col] += target[[row, col]];
            }
        }
        for v in &mut mean {
            *v /= rows.len() as f64;
        }
        let mut local = Array2::<f64>::zeros((rows.len(), 3));
        for (o, &r) in rows.iter().enumerate() {
            for col in 0..3 {
                local[[o, col]] = target[[r, col]] - mean[col];
            }
        }
        let (_u, _s, vt) = local.svd(false, true).unwrap();
        let vt = vt.unwrap();
        let n_pcs = vt.nrows().min(4);
        let mut proj = Array2::<f64>::zeros((n, n_pcs));
        for row in 0..n {
            for pc in 0..n_pcs {
                let mut acc = 0.0;
                for col in 0..3 {
                    acc += (target[[row, col]] - mean[col]) * vt[[pc, col]];
                }
                proj[[row, pc]] = acc;
            }
        }
        let sd = |pc: usize| -> f64 {
            let mut acc = 0.0;
            for &r in &rows {
                acc += proj[[r, pc]] * proj[[r, pc]];
            }
            (acc / rows.len() as f64).sqrt().max(1e-12)
        };
        let (sd0, sd1) = (sd(0), sd(1));
        let mut pca_coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            pca_coords[[row, 0]] = proj[[row, 0]] / sd0;
            pca_coords[[row, 1]] = proj[[row, 1]] / sd1;
        }
        let budget = duchon_sheet_race_center_budget(rows.len());
        let pca_centers = duchon_sheet_centers(&pca_coords, &rows, budget).unwrap();
        let pca_duchon = TopologyCandidateSpec {
            kind: AutoTopologyKind::DuchonSheet,
            basis_kind: SaeAtomBasisKind::Duchon,
            manifold: LatentManifold::Euclidean,
            latent_dim: 2,
            evaluator: Arc::new(
                DuchonCoordinateEvaluator::new(pca_centers, DUCHON_SHEET_M).unwrap(),
            ),
            coords: pca_coords.clone(),
        };
        let pca_flat = TopologyCandidateSpec {
            kind: AutoTopologyKind::Euclidean,
            basis_kind: SaeAtomBasisKind::EuclideanPatch,
            manifold: LatentManifold::Euclidean,
            latent_dim: 2,
            evaluator: Arc::new(EuclideanPatchEvaluator::new(2, 2).unwrap()),
            coords: pca_coords.clone(),
        };
        let pca_duchon_reml = fit_topology_candidate(&pca_duchon, target.view(), weights.view())
            .unwrap()
            .raw_reml;
        let pca_flat_reml = fit_topology_candidate(&pca_flat, target.view(), weights.view())
            .unwrap()
            .raw_reml;

        // Intrinsic challenger specs on the geodesic-unfolded chart.
        let (int_specs, int_chart) =
            build_intrinsic_primary_specs(target.view(), &rows, 2).unwrap().unwrap();
        let mut int_lines = String::new();
        for spec in &int_specs {
            let reml = fit_topology_candidate(spec, target.view(), weights.view())
                .unwrap()
                .raw_reml;
            int_lines.push_str(&format!(" [{:?} reml={reml}]", spec.basis_kind));
        }
        // In-sample reconstruction R² of each chart (proof of fit quality).
        let recon_r2 = |spec: &TopologyCandidateSpec| -> f64 {
            let fit = fit_topology_candidate(spec, target.view(), weights.view())
                .unwrap()
                .fit_handle;
            let recon = fit.phi.dot(&fit.decoder);
            let (mut sr, mut st) = (0.0_f64, 0.0_f64);
            for row in 0..n {
                for col in 0..3 {
                    let r = target[[row, col]] - recon[[row, col]];
                    sr += r * r;
                    let c = target[[row, col]] - mean[col];
                    st += c * c;
                }
            }
            1.0 - sr / st
        };
        let int_duchon = int_specs
            .iter()
            .find(|s| s.basis_kind == SaeAtomBasisKind::Duchon)
            .unwrap();
        let int_chart_span = {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for row in 0..n {
                for c in 0..2 {
                    lo[c] = lo[c].min(int_chart[[row, c]]);
                    hi[c] = hi[c].max(int_chart[[row, c]]);
                }
            }
            [hi[0] - lo[0], hi[1] - lo[1]]
        };
        panic!(
            "DIAG2: pca_duchon_reml={pca_duchon_reml} pca_flat_reml={pca_flat_reml} int={int_lines} \
             pca_duchon_insample_r2={} int_duchon_insample_r2={} int_chart_span={:?}",
            recon_r2(&pca_duchon),
            recon_r2(int_duchon),
            int_chart_span
        );
    }

    /// #2243 — a circle winner's harmonic RESOLUTION is grown by evidence, not
    /// pinned to the historical fixed budget (2 harmonics at the default
    /// `d_atom = 2`). The planted 1-D factor carries energy at the fundamental
    /// AND the 4th harmonic with a GAP between (harmonics 2, 3 are absent), which
    /// (a) the 2-harmonic default structurally cannot represent — half the energy
    /// lives at 4f — and (b) defeats a naive "stop at the first non-improving
    /// resolution" rule, exercising the global-argmin robustness of the selector.
    #[test]
    fn select_periodic_resolution_grows_past_default_over_harmonic_gap_2243() {
        use gam_solve::AutoTopologyKind;
        use ndarray::Array1;

        // Angular signal in R⁴: fundamental in (col0, col1), 4th harmonic in
        // (col2, col3), nothing at harmonics 2 or 3.
        let n = 240usize;
        let mut coords = Array2::<f64>::zeros((n, 1));
        let mut target = Array2::<f64>::zeros((n, 4));
        for row in 0..n {
            let t = row as f64 / n as f64;
            let angle = std::f64::consts::TAU * t;
            coords[[row, 0]] = t;
            target[[row, 0]] = angle.cos();
            target[[row, 1]] = angle.sin();
            target[[row, 2]] = (4.0 * angle).cos();
            target[[row, 3]] = (4.0 * angle).sin();
        }
        let weights = Array1::<f64>::ones(n);

        let selected = select_periodic_resolution(coords.view(), target.view(), weights.view(), n)
            .expect("resolution selection must succeed on a supported periodic signal");
        assert!(
            selected >= 4,
            "the 4th-harmonic content (past a gap) requires at least 4 harmonics; the fixed \
             2-harmonic default under-resolves it (selected={selected})"
        );

        // Reconstruction: the selected resolution recovers the whole signal while
        // the historical 2-harmonic default cannot touch the 4f half of the energy.
        // `h` is a harmonic ORDER (what `select_periodic_resolution` returns); the
        // periodic basis for order `h` is the `2h + 1` columns `{1, cos, sin, …,
        // cos hθ, sin hθ}`, and `PeriodicHarmonicEvaluator::new` takes that (odd)
        // width, not the order.
        let circle_r2 = |h: usize| -> f64 {
            let spec = TopologyCandidateSpec {
                kind: AutoTopologyKind::Circle,
                basis_kind: SaeAtomBasisKind::Periodic,
                manifold: LatentManifold::Circle { period: 1.0 },
                latent_dim: 1,
                evaluator: Arc::new(
                    PeriodicHarmonicEvaluator::new(2 * h + 1).expect("periodic evaluator"),
                ),
                coords: coords.clone(),
            };
            let fit = fit_topology_candidate(&spec, target.view(), weights.view())
                .expect("candidate fit")
                .fit_handle;
            let recon = fit.phi.dot(&fit.decoder);
            let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
            for col in 0..4 {
                let mut mean = 0.0;
                for row in 0..n {
                    mean += target[[row, col]];
                }
                mean /= n as f64;
                for row in 0..n {
                    let r = target[[row, col]] - recon[[row, col]];
                    ss_res += r * r;
                    let c = target[[row, col]] - mean;
                    ss_tot += c * c;
                }
            }
            1.0 - ss_res / ss_tot.max(1e-12)
        };
        let default_r2 = circle_r2(2);
        let selected_r2 = circle_r2(selected);
        eprintln!(
            "[resolution-2243] circle R²: default(2 harmonics)={default_r2:.4} selected({selected})={selected_r2:.4}"
        );
        assert!(
            selected_r2 > 0.99,
            "the evidence-selected resolution must recover the signal (R²={selected_r2:.4})"
        );
        assert!(
            default_r2 < 0.75,
            "the 2-harmonic default cannot represent the 4th-harmonic half of the energy (R²={default_r2:.4})"
        );
    }

    /// #2243 — a torus winner's per-axis harmonic ORDER is grown by evidence,
    /// not pinned to the fixed `SAE_DEFAULT_TORUS_HARMONICS = 3` budget. The
    /// planted toroidal factor carries the fundamental on axis 0 and the 5th
    /// harmonic on axis 1 with a GAP (orders 2, 3, 4 absent on that axis), which
    /// (a) the order-3 default structurally cannot represent — half the energy
    /// lives at 5f — and (b) defeats a naive "stop at the first non-improving
    /// order" rule, exercising the global-argmin robustness of the selector.
    #[test]
    fn select_torus_resolution_grows_past_default_over_harmonic_gap_2243() {
        use gam_solve::AutoTopologyKind;
        use ndarray::Array1;

        // Full g×g angular grid so the integer harmonics below Nyquist (g/2) are
        // exactly resolvable: fundamental on axis 0, 5th harmonic on axis 1.
        let g = 20usize;
        let n = g * g;
        let mut coords = Array2::<f64>::zeros((n, 2));
        let mut target = Array2::<f64>::zeros((n, 4));
        for i in 0..g {
            for j in 0..g {
                let row = i * g + j;
                let t0 = i as f64 / g as f64;
                let t1 = j as f64 / g as f64;
                coords[[row, 0]] = t0;
                coords[[row, 1]] = t1;
                let a0 = std::f64::consts::TAU * t0;
                let a1 = std::f64::consts::TAU * t1;
                target[[row, 0]] = a0.cos();
                target[[row, 1]] = a0.sin();
                target[[row, 2]] = (5.0 * a1).cos();
                target[[row, 3]] = (5.0 * a1).sin();
            }
        }
        let weights = Array1::<f64>::ones(n);

        let selected = select_torus_resolution(coords.view(), target.view(), weights.view(), n)
            .expect("resolution selection must succeed on a supported toroidal signal");
        assert!(
            selected >= 5,
            "the 5th-harmonic content (past a gap) requires at least order 5; the fixed \
             order-3 default under-resolves it (selected={selected})"
        );

        // Reconstruction: the selected order recovers the whole signal while the
        // fixed order-3 default cannot touch the 5f half of the energy.
        let torus_r2 = |h: usize| -> f64 {
            let spec = TopologyCandidateSpec {
                kind: AutoTopologyKind::Torus,
                basis_kind: SaeAtomBasisKind::Torus,
                manifold: LatentManifold::Product(vec![
                    LatentManifold::Circle { period: 1.0 },
                    LatentManifold::Circle { period: 1.0 },
                ]),
                latent_dim: 2,
                evaluator: Arc::new(TorusHarmonicEvaluator::new(2, h).expect("torus evaluator")),
                coords: coords.clone(),
            };
            let fit = fit_topology_candidate(&spec, target.view(), weights.view())
                .expect("candidate fit")
                .fit_handle;
            let recon = fit.phi.dot(&fit.decoder);
            let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
            for col in 0..4 {
                let mut mean = 0.0;
                for row in 0..n {
                    mean += target[[row, col]];
                }
                mean /= n as f64;
                for row in 0..n {
                    let r = target[[row, col]] - recon[[row, col]];
                    ss_res += r * r;
                    let c = target[[row, col]] - mean;
                    ss_tot += c * c;
                }
            }
            1.0 - ss_res / ss_tot.max(1e-12)
        };
        let default_r2 = torus_r2(SAE_DEFAULT_TORUS_HARMONICS);
        let selected_r2 = torus_r2(selected);
        eprintln!(
            "[resolution-2243] torus R²: default(order {SAE_DEFAULT_TORUS_HARMONICS})={default_r2:.4} selected({selected})={selected_r2:.4}"
        );
        assert!(
            selected_r2 > 0.99,
            "the evidence-selected order must recover the signal (R²={selected_r2:.4})"
        );
        assert!(
            default_r2 < 0.75,
            "the order-3 default cannot represent the 5th-harmonic half of the energy (R²={default_r2:.4})"
        );
    }

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
            if j == 0 {
                r * theta.cos()
            } else {
                r * theta.sin()
            }
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
        let expected_radial =
            standardized_log_birth_amplitudes(birth_row_amplitudes(disk.view()).view())
                .expect("disk log-amplitude spread");
        for spec in promoted.iter().filter(|spec| {
            matches!(
                spec.kind,
                AutoTopologyKind::Cylinder | AutoTopologyKind::Euclidean
            )
        }) {
            for row in 0..n {
                assert!(
                    (spec.coords[[row, 1]] - expected_radial[row]).abs() < 1.0e-12,
                    "promoted {:?} row {row} axis 1 must be standardized log-amplitude",
                    spec.kind
                );
            }
        }

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

    fn topology_fit_sse(fit: &TopologyRaceFit, target: ArrayView2<'_, f64>) -> f64 {
        let fitted = fit.phi.dot(&fit.decoder);
        let mut sse = 0.0_f64;
        for row in 0..target.nrows() {
            for col in 0..target.ncols() {
                let err = target[[row, col]] - fitted[[row, col]];
                sse += err * err;
            }
        }
        sse
    }

    #[test]
    fn radial_promotion_seed_coordinate_expresses_annulus_radius() {
        let n_angles = 16;
        let n_radii = 25;
        let n = n_angles * n_radii;
        let radial = vdc(n_radii);
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| {
            let angle_idx = row / n_radii;
            angle_idx as f64 / n_angles as f64
        });
        let annulus = Array2::from_shape_fn((n, 2), |(row, col)| {
            let angle_idx = row / n_radii;
            let radius_idx = row % n_radii;
            let theta = std::f64::consts::TAU * (angle_idx as f64 / n_angles as f64);
            let radius = 0.3 + 0.7 * radial[radius_idx].sqrt();
            if col == 0 {
                radius * theta.cos()
            } else {
                radius * theta.sin()
            }
        });
        let promoted = radial_promoted_specs(coords.view(), annulus.view(), 1)
            .expect("promotion decision")
            .expect("annulus radius spread promotes a radial axis");
        let circle = promoted
            .iter()
            .find(|spec| spec.kind == AutoTopologyKind::Circle)
            .expect("promoted race includes the circle alternative");
        let cylinder = promoted
            .iter()
            .find(|spec| spec.kind == AutoTopologyKind::Cylinder)
            .expect("promoted race includes the cylinder alternative");
        let weights = Array1::<f64>::ones(n);
        let circle_fit =
            fit_topology_candidate(circle, annulus.view(), weights.view()).expect("circle fit");
        let cylinder_fit =
            fit_topology_candidate(cylinder, annulus.view(), weights.view()).expect("cylinder fit");
        let circle_sse = topology_fit_sse(&circle_fit.fit_handle, annulus.view());
        let cylinder_sse = topology_fit_sse(&cylinder_fit.fit_handle, annulus.view());
        assert!(
            cylinder_sse < 0.75 * circle_sse,
            "radial seed should let cylinder express radius variation: cylinder_sse={cylinder_sse}, circle_sse={circle_sse}"
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
        assert!(
            idx.iter()
                .all(|&v| (0.0..=6.0).contains(&v) && v.fract() == 0.0)
        );

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
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        let empty = StructureSearchResult::from_rounds(term0.clone(), rho0.clone(), Vec::new());
        assert!(
            !empty.structure_changed(),
            "no rounds ⇒ the term/rho are the pre-search fit ⇒ structure_changed() must be false"
        );

        // Every move contested or vetoed: the dictionary is byte-for-byte the
        // pre-search fit, so the exact joint-Hessian bands remain valid.
        let no_landed = StructureSearchResult::from_rounds(
            term0.clone(),
            rho0.clone(),
            vec![ledger_with(vec![
                MoveVerdict::Contested { log_e: -1.0 },
                MoveVerdict::Vetoed { log_e: -2.0 },
            ])],
        );
        assert!(
            !no_landed.structure_changed(),
            "all-contested/vetoed rounds leave the model unchanged ⇒ structure_changed() must be false"
        );

        // An Accepted move landed (certified restructuring + warm refit): the
        // returned model differs from the pre-search fit ⇒ bands are stale.
        let accepted = StructureSearchResult::from_rounds(
            term0.clone(),
            rho0.clone(),
            vec![ledger_with(vec![
                MoveVerdict::Contested { log_e: -1.0 },
                MoveVerdict::Accepted { log_e: 3.0 },
            ])],
        );
        assert!(
            accepted.structure_changed(),
            "a landed Accepted move mutates term/rho ⇒ structure_changed() must be true (recompute bands)"
        );

        // A Demoted death is also a landed structure change.
        let demoted = StructureSearchResult::from_rounds(
            term0.clone(),
            rho0.clone(),
            vec![ledger_with(vec![MoveVerdict::Demoted { log_e: -1.0 }])],
        );
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
        // An OVERCOMPLETE dictionary (G = 8 atoms, exactly one active per token ⇒
        // L0 = 1) so the #2233 birth pre-screen the harvest channel now embeds
        // (`predicted_birth_dl_bits`) sees a real support saving log₂(G/L0) = 3
        // bits/token — the term that funds a curved birth. A single-atom dictionary
        // (G = 1, log₂(G/L0) = 0) offers ZERO overcompleteness and a lone LINEAR
        // residual direction earns no code saving either, so the pre-screen
        // correctly refuses to birth there: growth pays only once a curved atom
        // spares the extra active slots a flat span would spend.
        let n = 40usize;
        let k = 8usize;
        let active: Vec<Vec<bool>> = (0..n)
            .map(|row| (0..k).map(|atom| atom == row % k).collect())
            .collect();
        let (term, rho) = planted_term(&active);
        // Inject a genuinely CURVED (rank-2) residual: a zero-mean circle living in
        // the 2-plane spanned by two ORTHONORMAL output directions `u ⟂ v`. Its two
        // factor coordinates carry balanced, common-mode-free energy (an isotropic
        // 2-D cloud, not a dominant shared direction), so the LOCAL ambient span the
        // pre-screen measures on the candidate's own firing rows is ≈ 2 ⇒ priced as
        // a circle (d = 1, code term 0) with a positive support saving. This is the
        // curved structure a born atom absorbs — the birth the theorem admits.
        let p = term.output_dim();
        let mut residuals = Array2::<f64>::zeros((n, p));
        let u = [0.6_f64, -0.4, 0.5, -0.3];
        let v = [0.4_f64, 0.6, 0.3, 0.5]; // u · v = 0, ‖u‖ = ‖v‖
        let un: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        let vn: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for row in 0..n {
            let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
            let (s, c) = theta.sin_cos();
            for out in 0..p {
                residuals[[row, out]] = 2.0 * (c * u[out] / un + s * v[out] / vn);
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

    /// #1890 verification fixture: plant ONE circle tiled into `k` arc atoms with
    /// DISJOINT supports. Row `r` (of `n`) sits at circle phase `2π·r/n`; arc atom
    /// `j` owns the contiguous row block `[j·n/k, (j+1)·n/k)` (routed ON there,
    /// OFF elsewhere). Every atom shares the SAME periodic basis and a unit-circle
    /// decoder (harmonic cols 1,2 → ambient dims 0,1) scaled by `decoder_scale[j]`,
    /// so a shared scale makes all arcs decode onto ONE closed curve — the
    /// over-tiling signature the co-activation fusion lane is structurally blind to
    /// (disjoint supports ⇒ anti-correlated codes). A per-atom scale plants a
    /// CONCENTRIC circle of a different radius (a genuinely distinct manifold —
    /// the negative control).
    fn tiled_circle_term(
        n: usize,
        k: usize,
        decoder_scale: &[f64],
    ) -> (SaeManifoldTerm, SaeManifoldRho) {
        assert_eq!(decoder_scale.len(), k, "one decoder scale per arc atom");
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let mut atoms = Vec::with_capacity(k);
        let mut coord_blocks = Vec::with_capacity(k);
        for (j, &scale) in decoder_scale.iter().enumerate() {
            // Unit-circle decoder: first cos/sin harmonic → ambient {0,1}, radius
            // `scale`. Shared scale ⇒ ONE circle (glue); distinct scale ⇒ a
            // concentric distinct circle (no glue).
            let mut decoder = Array2::<f64>::zeros((m, p));
            decoder[[1, 0]] = scale;
            decoder[[2, 1]] = scale;
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
                format!("arc_{j}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
            coord_blocks.push(coords.clone());
        }
        let mut logits = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let owner = (row * k) / n; // contiguous arc ownership
            for j in 0..k {
                logits[[row, j]] = if j == owner { ON } else { OFF };
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

    /// #1890 negative control (type-I): two CONCENTRIC circles of different radii —
    /// a shared plane (so the frames align and the geometric pre-screen DOES
    /// nominate the pair), disjoint adjacent arc supports, but a NON-isometric
    /// transition onto a genuinely distinct manifold. Because the pair is screened,
    /// this exercises the seam EQUIVALENCE e-value itself, not the pre-screen: it
    /// must REJECT (negative log-e), so the engine's positive-evidence e-gate never
    /// accepts the glue. Clusters that DON'T glue must not be forced together.
    #[test]
    fn distinct_concentric_circles_do_not_glue() {
        let n = 40usize;
        let (term, rho) = tiled_circle_term(n, 2, &[1.0, 2.0]); // radius 1 vs radius 2
        let residuals = Array2::<f64>::zeros((n, 4));
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 4,
            max_births: 0,
        };
        let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();

        // No co-activation fusion (disjoint supports), same as the tiling case.
        assert!(
            !report
                .proposals
                .iter()
                .any(|p| matches!(p.mv, StructureMove::Fusion { .. })),
            "disjoint supports must yield no co-activation fusion"
        );

        // The shared plane aligns the frames, so the pair IS screened — this is a
        // genuine test of the equivalence e-value, which must reject the distinct
        // manifold.
        assert!(
            report.glue_candidates_screened >= 1,
            "the shared-plane pair must be geometrically screened"
        );
        let (e_distinct, _) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
            .expect("the aligned pair yields a seam e-value and transition certificate");
        assert!(
            e_distinct.log_e_value < 0.0,
            "distinct concentric circles must NOT glue (negative log-e), got {}",
            e_distinct.log_e_value
        );

        // Any emitted glue proposal therefore carries negative evidence — rejected
        // by the engine's e-gate (which certifies only positive log-e).
        for p in &report.proposals {
            if matches!(p.mv, StructureMove::Glue { .. }) {
                assert!(
                    p.trigger < 0.0,
                    "the distinct-circle glue must carry negative evidence, got {}",
                    p.trigger
                );
            }
        }
    }

    #[test]
    fn closed_form_transition_uses_first_nonzero_harmonic_without_scanning() {
        let mut decoder_a = Array2::<f64>::zeros((5, 3));
        decoder_a[[3, 0]] = 2.0;
        decoder_a[[4, 1]] = 1.0;
        decoder_a[[0, 2]] = 0.25;
        let decoder_b = periodic_decoder_under_transition(decoder_a.view(), -1, 0.125).unwrap();
        let (sign, offset) =
            fit_periodic_transition_from_decoders(decoder_a.view(), decoder_b.view()).unwrap();
        assert_eq!(sign, -1);
        let recovered = periodic_decoder_under_transition(decoder_a.view(), sign, offset).unwrap();
        for (actual, expected) in recovered.iter().zip(decoder_b.iter()) {
            assert!((actual - expected).abs() < 32.0 * f64::EPSILON);
        }
    }

    /// An orientation-reversing seam is a valid equivalence, but it is NOT a
    /// license to erase either local chart.  The production proposal must select
    /// the atlas-register outcome, and applying it must preserve the numerical
    /// chart count and fitted image while reducing the semantic atom count.
    #[test]
    fn orientation_reversing_seam_registers_atlas_without_destructive_fusion() {
        let n = 40usize;
        let (mut term, rho) = tiled_circle_term(n, 2, &[1.0, 1.0]);
        // `sin` changes sign under t -> -t; `cos` does not.  B therefore traces
        // exactly A's image with the orientation-reversing transition t_A=-t_B.
        term.atoms[1].decoder_coefficients[[1, 0]] = -1.0;
        let residuals = Array2::<f64>::zeros((n, 4));
        let (transition, _) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
            .expect("reflected charts have an exact certified seam");
        assert_eq!(transition.sign, -1);
        assert!(transition.log_e_value > 5.0);

        let report = harvest_move_proposals(
            &term,
            &rho,
            residuals.view(),
            &HarvestParams {
                max_fusions: 4,
                max_fissions: 0,
                max_births: 0,
            },
        )
        .unwrap();
        let mv = report
            .proposals
            .iter()
            .find_map(|proposal| match proposal.mv {
                StructureMove::Glue {
                    a,
                    b,
                    outcome: ChartGlueOutcome::RegisterAtlas,
                } => Some(StructureMove::Glue {
                    a,
                    b,
                    outcome: ChartGlueOutcome::RegisterAtlas,
                }),
                _ => None,
            })
            .expect("negative seam must propose atlas registration");

        let fitted_before = term.try_fitted().unwrap();
        let (registered, _) = apply_structure_move(&term, &rho, &mv, &[]).unwrap();
        assert_eq!(registered.k_atoms(), 2, "both local charts must survive");
        assert_eq!(registered.semantic_atom_count(), 1);
        assert_eq!(registered.chart_atlases().len(), 1);
        assert_eq!(registered.chart_atlases()[0].transitions()[0].sign, -1);
        assert_eq!(
            registered.try_fitted().unwrap(),
            fitted_before,
            "atlas registration is an image-exact quotient"
        );

        let assignments = registered.assignment.assignments();
        for row in 0..n {
            let (activation, partition) = registered
                .atlas_partition_of_unity(0, assignments.row(row))
                .unwrap();
            assert!((partition.sum() - 1.0).abs() < 8.0 * f64::EPSILON);
            for (slot, &chart) in registered.chart_atlases()[0].charts().iter().enumerate() {
                assert!(
                    (activation * partition[slot] - assignments[[row, chart]]).abs()
                        < 8.0 * f64::EPSILON
                );
            }
        }
    }

    /// #1890 over-birth reassembly — the PHYSICAL-excision resurrection fix, on the
    /// private primitives the `chart_gluing_1890.rs` integration test cannot reach.
    /// A circle over-tiled into 4 disjoint arcs proposes a spanning set of glues
    /// (each arc pair lies on ONE circle, so `unit_speed_glue_certificate`
    /// certifies with large positive log-e) with ZERO co-activation fusions. A
    /// round that accepts a matching of those glues then EXCISES the folded
    /// partners for real ([`compact_glued_atoms`]): folding + demoting alone leaves
    /// a zero-mass atom the #976/#1003 active-mass guard revives on the next refit,
    /// so the effective count never falls; removal drops both the raw and active
    /// size, and the wider-arc survivors STILL glue — the round sequence converges
    /// toward the single reassembled chart K=1.
    #[test]
    fn over_tiling_physical_excision_reduces_k_toward_one() {
        use gam_solve::structure_search::{MoveRecord, MoveVerdict};

        let n = 32usize;
        let (mut term, mut rho) = tiled_circle_term(n, 4, &[1.0; 4]);
        assert_eq!(term.k_atoms(), 4);

        // The primitive certifies every arc pair as ONE circle (large positive
        // log-e), and the co-activation lane stays silent on the disjoint supports.
        let residuals0 = Array2::<f64>::zeros((n, 4));
        let (e_arc, _) = unit_speed_glue_certificate(&term, residuals0.view(), 0, 2)
            .expect("a d=1 aligned disjoint pair yields a certified seam e-value");
        assert!(
            e_arc.log_e_value > 5.0,
            "e_glue must certify two arcs of one circle, got {}",
            e_arc.log_e_value
        );
        let params0 = HarvestParams {
            max_fusions: 16,
            max_fissions: 0,
            max_births: 0,
        };
        let report0 = harvest_move_proposals(&term, &rho, residuals0.view(), &params0).unwrap();
        assert!(
            !report0
                .proposals
                .iter()
                .any(|p| matches!(p.mv, StructureMove::Fusion { .. })),
            "disjoint tiling must yield no co-activation fusion"
        );
        assert!(
            report0.glues_proposed >= 3,
            "a spanning set (≥ k−1 = 3 edges) must reassemble the 4 arcs, got {}",
            report0.glues_proposed
        );
        let first_epoch_glue_claims: Vec<ClaimKind> = report0
            .proposals
            .iter()
            .filter(|proposal| matches!(proposal.mv, StructureMove::Glue { .. }))
            .map(|proposal| proposal.claim.clone())
            .collect();
        assert!(!first_epoch_glue_claims.is_empty());

        // Seed old-K state that MUST NOT survive a physical dictionary resize.
        // These are all legitimate transient states immediately after a fit;
        // the compactor owns invalidating them before the reduced-K polish.
        term.assignment.frozen_logits = Some(term.assignment.logits.clone());
        term.last_frames_active = true;
        term.fixed_decoder_assembly = true;
        term.border_hbb_workspace = Array2::<f64>::ones((3, 3));
        term.decoder_repulsion_gate = Some(vec![(0, 1, 1.0)]);
        term.streaming_gates_frozen = true;
        term.expected_criterion_gauge_deflated_directions = Some(7);
        term.criterion_gauge_deflation_reanchors = 2;
        term.criterion_gauge_deflation_last_delta_sign = -1;
        term.dictionary_cocollapse_reseeds = 3;
        term.structural_cocollapse_reseeds = 4;
        let accepted_glue = |a: usize, b: usize| MoveRecord {
            mv: StructureMove::Glue {
                a,
                b,
                outcome: ChartGlueOutcome::Fuse,
            },
            trigger: 40.0,
            structure_hash: 0,
            claim: ClaimKind::Custom {
                label: format!("seam_glue:{a}:{b}"),
            },
            verdict: MoveVerdict::Accepted { log_e: 40.0 },
        };
        // A matching (shares no atom) of disjoint glues — the within-round `touched`
        // guard admits exactly this shape in one round.
        let ledger = SearchLedger {
            alpha: 0.05,
            moves: vec![accepted_glue(0, 1), accepted_glue(2, 3)],
            collapse_events: Vec::new(),
        };
        let removed =
            compact_glued_atoms(&mut term, &mut rho, &ledger, &report0.certified_glues).unwrap();
        assert_eq!(
            removed, 2,
            "both folded partners must be physically excised"
        );
        assert_eq!(
            term.k_atoms(),
            2,
            "physical excision must reduce K from 4 to 2 (no active-mass resurrection)"
        );
        assert!(
            term.assignment
                .logits
                .rows()
                .into_iter()
                .all(|row| row.as_slice().is_some()),
            "compaction must materialize a row-contiguous router for the polish refit"
        );
        assert_eq!(
            rho.log_ard.len(),
            2,
            "ρ ARD blocks must fall in lock-step with the atoms"
        );
        assert_eq!(
            rho.log_lambda_smooth.len(),
            2,
            "ρ smoothness blocks must fall in lock-step with the atoms"
        );
        assert!(
            term.assignment.frozen_logits.is_none(),
            "an old-K frozen router cannot survive compaction"
        );
        assert!(!term.last_frames_active);
        assert!(!term.fixed_decoder_assembly);
        assert_eq!(term.border_hbb_workspace.dim(), (0, 0));
        assert!(term.decoder_repulsion_gate.is_none());
        assert!(!term.streaming_gates_frozen);
        assert_eq!(term.expected_criterion_gauge_deflated_directions, None);
        assert_eq!(term.criterion_gauge_deflation_reanchors, 0);
        assert_eq!(term.criterion_gauge_deflation_last_delta_sign, 0);
        assert_eq!(term.dictionary_cocollapse_reseeds, 0);
        assert_eq!(term.structural_cocollapse_reseeds, 0);
        // The two survivors each now cover a half-circle and STILL glue — the round
        // sequence converges toward the single reassembled chart (K=1).
        let residuals = Array2::<f64>::zeros((n, 4));
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 0,
            max_births: 0,
        };
        let report2 = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
        assert!(
            report2.glues_proposed >= 1,
            "the two reassembled half-circle survivors must still glue toward K=1"
        );
        for proposal in report2
            .proposals
            .iter()
            .filter(|proposal| matches!(proposal.mv, StructureMove::Glue { .. }))
        {
            assert!(
                !first_epoch_glue_claims.contains(&proposal.claim),
                "a reduced dictionary must not reuse old atom-index evidence: {:?}",
                proposal.claim
            );
        }
    }

    /// The compactor must adopt the transition measured while the retired chart
    /// still had live support. Folding first demotes B and makes seam fitting
    /// impossible, so the harvest certificate carries both the transition and
    /// B's certified support rows to the round boundary.
    #[test]
    fn physical_excision_transplants_coords_from_the_live_seam() {
        use gam_solve::structure_search::{MoveRecord, MoveVerdict};

        let (mut term, mut rho) = tiled_circle_term(32, 2, &[1.0; 2]);
        assert!(term.assignment.frozen_logits.is_none());
        let seam = fit_seam_transition(&term, 0, 1).expect("live pair has a seam");
        let residuals = Array2::<f64>::zeros((32, 4));
        let (_, certificate) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
            .expect("live pair has a harvest-time glue certificate");
        let da = term.assignment.coords[0].latent_dim();
        let db = term.assignment.coords[1].latent_dim();
        let flat_b = term.assignment.coords[1].as_flat().to_owned();
        let expected: Vec<(usize, f64)> = seam
            .rows_b
            .iter()
            .map(|&row| {
                let mapped = (seam.sign * flat_b[row * db] + seam.offset).rem_euclid(seam.period);
                (row, mapped)
            })
            .collect();
        // Poison only A's INACTIVE coordinates on B's rows.  A correct
        // transplant overwrites every sentinel through the measured seam.
        let mut flat_a = term.assignment.coords[0].as_flat().to_owned();
        for &(row, mapped) in &expected {
            flat_a[row * da] = (mapped + 0.37).rem_euclid(seam.period);
        }
        term.assignment.coords[0].set_flat(flat_a.view());
        // Simulate the scoring-refit drift that caused #1890's production red:
        // the terminal state can no longer identify a decoder transition.  The
        // accepted structural object is nevertheless fully specified by the
        // live harvest certificate above and must not be inferred again here.
        term.atoms[1].decoder_coefficients.fill(0.0);
        assert!(
            fit_seam_transition(&term, 0, 1).is_none(),
            "post-harvest fixture must make seam re-fitting impossible"
        );

        let ledger = SearchLedger {
            alpha: 0.05,
            moves: vec![MoveRecord {
                mv: StructureMove::Glue {
                    a: 0,
                    b: 1,
                    outcome: ChartGlueOutcome::Fuse,
                },
                trigger: 40.0,
                structure_hash: 0,
                claim: ClaimKind::Custom {
                    label: "test-live-seam".to_string(),
                },
                verdict: MoveVerdict::Accepted { log_e: 40.0 },
            }],
            collapse_events: Vec::new(),
        };
        compact_glued_atoms(&mut term, &mut rho, &ledger, &[certificate]).unwrap();

        assert_eq!(term.k_atoms(), 1);
        let survivor = term.assignment.coords[0].as_flat();
        for (row, mapped) in expected {
            assert!(
                (survivor[row * da] - mapped).abs() < 1.0e-12,
                "row {row}: survivor coordinate {} != seam-mapped {mapped}",
                survivor[row * da]
            );
        }
    }

    /// Variable-K compaction is transactional: malformed paired ρ state returns
    /// an error before changing the term, instead of panicking halfway through a
    /// gather and leaving atom/assignment arrays at different widths.
    #[test]
    fn physical_excision_validation_is_transactional() {
        use gam_solve::structure_search::{MoveRecord, MoveVerdict};

        let (mut term, mut rho) = tiled_circle_term(16, 3, &[1.0; 3]);
        let atoms_before = term.k_atoms();
        let logits_before = term.assignment.logits.clone();
        rho.log_ard.pop();
        let remove = std::collections::BTreeSet::from([1usize]);

        let err = remove_atoms(&mut term, &mut rho, &remove).unwrap_err();
        assert!(
            err.contains("rho per-atom lengths"),
            "unexpected error: {err}"
        );
        assert_eq!(term.k_atoms(), atoms_before);
        assert_eq!(term.assignment.logits, logits_before);

        // Restore ρ and feed an impossible overlapping matching.  The compactor
        // must reject the whole ledger before the first fold mutates any logit.
        rho.log_ard.push(Array1::zeros(1));
        let accepted = |a: usize, b: usize| MoveRecord {
            mv: StructureMove::Glue {
                a,
                b,
                outcome: ChartGlueOutcome::Fuse,
            },
            trigger: 40.0,
            structure_hash: 0,
            claim: ClaimKind::Custom {
                label: format!("test-glue:{a}:{b}"),
            },
            verdict: MoveVerdict::Accepted { log_e: 40.0 },
        };
        let overlapping = SearchLedger {
            alpha: 0.05,
            moves: vec![accepted(0, 1), accepted(1, 2)],
            collapse_events: Vec::new(),
        };
        let err = compact_glued_atoms(&mut term, &mut rho, &overlapping, &[]).unwrap_err();
        assert!(
            err.contains("not an atom-disjoint matching"),
            "unexpected error: {err}"
        );
        assert_eq!(term.k_atoms(), atoms_before);
        assert_eq!(term.assignment.logits, logits_before);

        // A structurally valid accepted record without its harvest certificate
        // is also rejected transactionally.  Re-fitting a replacement seam at
        // this boundary would recreate the scoring/adoption ordering bug.
        let missing_certificate = SearchLedger {
            alpha: 0.05,
            moves: vec![accepted(0, 1)],
            collapse_events: Vec::new(),
        };
        let err = compact_glued_atoms(&mut term, &mut rho, &missing_certificate, &[]).unwrap_err();
        assert!(
            err.contains("no harvest-time certificate"),
            "unexpected error: {err}"
        );
        assert_eq!(term.k_atoms(), atoms_before);
        assert_eq!(term.assignment.logits, logits_before);
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
                |t, r, _| Ok((t, r)),
                |t, r, _| Ok((t, r)),
                // No-op polish: this determinism oracle scripts the gate and
                // never runs the SAE inner solve.
                |t, r, _| Ok((t, r)),
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
        let null_gate = gate_block_log_evidence(&null_term, &shard).unwrap();
        let cand_gate = gate_block_log_evidence(&cand_term, &shard).unwrap();
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
            gate_block_log_evidence(term, &shard).unwrap() + 0.5 * dg * log_2pi
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
        let full = eval_log_lik(&cand_term, &shard).unwrap();
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
    /// refit. (For ordered Beta--Bernoulli routing `max` stays correct — the gate is un-normalized.)
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
        SaeManifoldAtom::new_with_provided_function_gram(
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
        assert!(
            cand.net_evidence > 0.0,
            "net evidence must favour the circle"
        );

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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        let diameter_phases = Array1::from_shape_fn(n, |r| if r % 2 == 0 { 0.0 } else { 0.5 });
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

    /// KILLER DEMO — end-to-end through the round driver: with curl ON the
    /// shattered centered circle yields a circle Birth that certifies through
    /// the same e-gate the residual births race through; with curl OFF (the
    /// default) the driver is unchanged. The paired null checks pin the
    /// structural boundary: Gaussian fill is not curled, and a diameter
    /// collapse flattens to rank 1.
    #[test]
    fn curl_killer_demo_planted_circle_wins_race() {
        let (term, _rho) = shattered_plane_term(false);
        let residuals = residuals_of(&term);
        let cands = curl_candidates(&term, residuals.view(), &CurlConfig::default());
        assert!(
            !cands.is_empty(),
            "curl must recover the shattered circle before the race"
        );
        let mut members = cands[0].members.clone();
        members.sort_unstable();
        members.dedup();
        assert_eq!(
            members,
            vec![0, 1, 2, 3],
            "the recovered circle must claim all four rectified halves"
        );

        let budget = MoveBudget {
            max_moves: 4,
            alpha: 0.05,
        };
        let harvest_params = HarvestParams {
            max_fusions: 0,
            max_fissions: 0,
            max_births: 0,
        };
        let run = |curl: Option<CurlConfig>| -> StructureSearchResult {
            let (term, rho) = shattered_plane_term(false);
            let target = 2.0 * term.try_fitted().unwrap();
            let mut ledger = StructureLedger::new();
            let config = RoundDriverConfig {
                n_shards: 3,
                budget,
                harvest_params,
                curl,
            };
            run_structure_search_rounds(
                term,
                rho,
                target.view(),
                config,
                &mut ledger,
                |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
                |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
                |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
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
            on.term.atoms.last().map(|a| &a.basis_kind),
            Some(&SaeAtomBasisKind::Periodic),
            "the accepted curl winner must be the recovered circle atom"
        );
        assert!(
            on.structure_changed(),
            "accepted curl winner must mutate the returned term"
        );

        let (gauss_term, _) = shattered_plane_term(true);
        let gauss_residuals = residuals_of(&gauss_term);
        let gauss_cands =
            curl_candidates(&gauss_term, gauss_residuals.view(), &CurlConfig::default());
        assert!(
            gauss_cands.is_empty(),
            "a Gaussian-fill plane must not be curled"
        );

        let n = 400usize;
        let radii = Array1::<f64>::from_elem(n, 3.0);
        let angles = Array1::<f64>::from_shape_fn(n, |r| {
            if r % 2 == 0 {
                0.0
            } else {
                std::f64::consts::PI
            }
        });
        let flatten = crate::manifold::flatten_verdict(radii.view(), angles.view()).unwrap();
        assert!(flatten.recommend_flatten, "diameter must flatten");
        assert_eq!(flatten.residual_rank, 1, "diameter must flatten to rank 1");
    }
}
