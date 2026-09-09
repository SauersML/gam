//! Atlas-first topology discovery (#2280): what manifold the local charts and
//! their transition holonomy SAY the data is, rather than which entry of a fixed
//! menu fits it best.
//!
//! # The construction
//!
//! A [`LocalAtlas`] is a finite open cover `𝒰 = {U_0, …, U_{m−1}}` of the sampled
//! rows by overlapping neighborhoods, each carrying a certified injective local
//! chart `φ_i : U_i → ℝ^d` (the patch's local-PCA frame) and, on every certified
//! overlap, the fitted transition `g_ij = φ_j ∘ φ_i^{-1}` with its orthogonal part
//! `R_ij ∈ O(d)`. Two objects are read off that cover.
//!
//! **The nerve.** `N(𝒰)` is the simplicial complex whose `k`-simplices are the
//! `(k+1)`-subsets `S` with `⋂_{i∈S} U_i ≠ ∅`. By the nerve theorem, when every
//! non-empty finite intersection is contractible, `N(𝒰) ≃ M`; its `GF(2)` homology
//! and its Euler characteristic are then the manifold's. Two things this module
//! refuses to get wrong, both of them measured mistakes recorded on the issue:
//! the 1-skeleton alone is the OVERLAP GRAPH and its cycle rank is a property of
//! the cover, not of `M`; and truncating `χ` at triples is wrong for any data
//! cover, because 4-, 5- and 6-way overlaps are routinely non-empty. So `b₁` is
//! read from `H₁(N)` — graph cycles quotiented by the 2-cells — and `χ` is the
//! FULL alternating sum `Σ_k (−1)^k N_{k+1}` over every cardinality.
//!
//! **The holonomy class.** `(R_ij)` is a Čech 1-cochain valued in `O(d)`. Because
//! `det : O(d) → {±1}` is a group homomorphism, the signs
//! `s_ij = (1 − det R_ij)/2 ∈ GF(2)` form a `GF(2)` 1-cochain whose coboundary on
//! a triangle is `s_ij + s_jk + s_ik`. When that vanishes on every 2-simplex, `s`
//! is a genuine cocycle and its class `[s] ∈ H¹(N; GF(2))` is the first
//! Stiefel–Whitney class `w₁(TM)` pulled back along the nerve equivalence:
//!
//! * `[s] = 0` ⟺ `s = δf` for a vertex 0-cochain `f` ⟺ the chart frames admit one
//!   globally coherent choice of sign ⟺ `M` is orientable;
//! * `[s] ≠ 0` ⟺ some 1-cycle carries an odd number of orientation reversals —
//!   the Möbius obstruction, evaluated by `⟨s, z⟩` loop by loop.
//!
//! This is the sense in which the holonomy *is* the topological content: it is a
//! cohomology class on the nerve, not a scalar score, and nothing about it refers
//! to a candidate model.
//!
//! # Why the SIGN cocycle is the trust gate and the `O(d)` cocycle is not
//!
//! The composed rotation around a triangle, `R_ca R_bc R_ab`, is the parallel
//! transport of the tangent frame around that loop. On a CURVED manifold it is
//! not the identity — its defect is the integrated curvature over the loop,
//! `‖R_ca R_bc R_ab − I‖ ≈ |K| · area`. Gating the atlas's trustworthiness on that
//! defect being small therefore penalizes curvature, which is the very thing a
//! sphere or a torus is made of, and it silently abstains exactly where the
//! readout matters most.
//!
//! The `GF(2)` reduction has no such term: `Z/2` carries no curvature, so the SIGN
//! cocycle closes EXACTLY whenever the cover is fine enough that consecutive
//! tangent planes are mutually non-degenerate. Its closure is therefore both the
//! honest precondition for `w₁` to exist and a constant-free, data-derived
//! good-cover probe: it is checked, not thresholded.
//!
//! # What the charts add that the nerve cannot
//!
//! A circle and a cylinder are homotopy equivalent — same `b₀, b₁, b₂`, same `χ`,
//! same trivial `w₁` — so no invariant of the nerve separates them. The atlas
//! supplies the missing datum the nerve does not have: the LOCAL CHART RANK `d`,
//! certified per patch by the local-PCA rank gate. Charts and nerve together are
//! strictly stronger than either alone, and the classification below dispatches
//! on `d` first.
//!
//! # Ambient-embedding blindness
//!
//! Every input to the verdict is a transition between charts, and a transition is
//! intrinsic. A trefoil knot in `ℝ³` is a circle whose three ambient principal
//! directions have comparable spread, so no global-linear seed can recognize it;
//! its atlas is a cycle of 1-dimensional tangent charts with trivial holonomy, and
//! this readout calls it a circle. That gap — between what a global linear seed
//! can see and what the transitions see — is the whole point of the construction.
//!
//! # Authority
//!
//! Every quantity here is named `observed_*` for the same reason
//! [`LocalAtlas::observed_orientability`] is: this module has no sampling model
//! and no error probability, so it may PROPOSE a topology but never promote one.
//! A population claim continues to route through
//! [`crate::inference::atlas_holonomy::AtlasHolonomyCertificate`]. What the
//! verdict is authorized to do is put the recognized manifold in front of the
//! evidence race, which then adjudicates it on the same REML scale as every other
//! candidate.

use super::local_charts::sorted_intersection;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::AtlasOrientability;
use super::graph_atom::GraphCompressionKind;
use super::local_charts::LocalAtlas;
use super::persistence::BettiSignature;
use crate::inference::atlas_nerve::{compute_betti, enumerate_full_nerve, surface_from_invariants};

/// Why the atlas declined to name a topology.
///
/// A refusal is never a verdict of "no structure": it says the cover this atlas
/// realized cannot support a topological claim, and the caller should proceed as
/// if no atlas had been built.
#[derive(Clone, Debug, PartialEq)]
pub enum TopologyRefusal {
    /// The atlas certified no charts at all.
    EmptyAtlas,
    /// The certified charts fall into more than one connected overlap component
    /// (`b₀ > 1`), so they cover several disjoint pieces rather than one manifold.
    /// Naming a topology would name only whichever piece the table happened to
    /// match.
    DisconnectedCover { components: usize },
    /// Some pair of patches overlaps in two or more components whose fitted
    /// transitions disagree in sign. The intersection is then not connected, let
    /// alone contractible, so the nerve theorem does not apply and `s` is not even
    /// well defined as a function of the pair.
    IncoherentOverlap { pairs: usize },
    /// The `GF(2)` orientation cochain is not a cocycle: `δs ≠ 0` on this many of
    /// the nerve's 2-simplices whose three edges all carry a RESOLVED sign. The
    /// cover is too coarse for the tangent planes to compose, so `w₁` does not
    /// exist on it.
    OrientationCocycleOpen { open_triangles: usize, total: usize },
    /// `w₁` is a class on the NERVE, `[s] ∈ H¹(N; GF(2))`, but `s` is defined only
    /// on the subcomplex `S ⊆ N` of edges whose handedness is resolved. Reading the
    /// class off `S` is legitimate only while `S` sees the same 1-cycles `N` does:
    /// if `S` is more disconnected than `N` the relative orientation across the gap
    /// is undetermined (and an orientable reading is vacuous — no cycle crosses the
    /// gap, so none can contradict it), and if `S` carries EXTRA 1-cycles that `N`
    /// fills with 2-cells, a sign pattern can be non-trivial on `S` for a reason
    /// that is not the manifold's — which is how a torus gets called a Klein
    /// bottle. Both are refused here, on the measured Betti numbers of the two
    /// complexes rather than on a proxy count of dropped edges.
    OrientationSubcomplexIncomplete {
        signed_b0: usize,
        signed_b1: usize,
        nerve_b0: usize,
        nerve_b1: usize,
    },
    /// The nerve of a `d ≥ 2` cover carries no 2-simplex, so it is a bare graph
    /// and its cycle rank is a property of the cover rather than of the manifold.
    NoTwoCells,
    /// Some row lies in far more patches than the cover was built to overlap.
    ///
    /// Lebesgue's covering-dimension theorem says a good cover of a `d`-manifold
    /// refines to one of multiplicity at most `d + 1`; the atlas builder in turn
    /// sizes its patches so the cover's MEAN multiplicity is its own overlap
    /// design point. A row covered more than `d + 1` times as densely as that mean
    /// is not a point of a `d`-manifold cover but a pile-up — collapsed rows, a
    /// duplicated block, a degenerate direction — and its nerve is a simplex on
    /// every patch through it rather than a chart intersection. Both bounds are
    /// measured on the atlas itself; nothing here is a chosen budget.
    CoverMultiplicityTooHigh {
        observed: usize,
        admissible: usize,
        mean: f64,
    },
    /// The measured invariants are internally consistent but match no compact
    /// manifold of this chart rank.
    UnclassifiedInvariants,
}

impl fmt::Display for TopologyRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAtlas => write!(f, "the atlas certified no charts"),
            Self::DisconnectedCover { components } => write!(
                f,
                "the certified charts cover {components} disconnected components, not one manifold"
            ),
            Self::IncoherentOverlap { pairs } => write!(
                f,
                "{pairs} patch pairs overlap in sign-disagreeing components, so the intersection is not connected"
            ),
            Self::OrientationCocycleOpen {
                open_triangles,
                total,
            } => write!(
                f,
                "the orientation cochain is not a cocycle: it fails to close on {open_triangles} of {total} fully-signed nerve triangles"
            ),
            Self::OrientationSubcomplexIncomplete {
                signed_b0,
                signed_b1,
                nerve_b0,
                nerve_b1,
            } => write!(
                f,
                "the resolved-sign subcomplex has (b0, b1) = ({signed_b0}, {signed_b1}) against the \
                 nerve's ({nerve_b0}, {nerve_b1}), so it does not see the same 1-cycles and the \
                 orientation class it carries is not the manifold's"
            ),
            Self::NoTwoCells => write!(
                f,
                "the nerve of this surface cover has no 2-cell, so its cycles are cover artifacts"
            ),
            Self::CoverMultiplicityTooHigh {
                observed,
                admissible,
                mean,
            } => write!(
                f,
                "some row lies in {observed} patches, above the {admissible} a d-manifold cover of \
                 mean multiplicity {mean:.2} admits"
            ),
            Self::UnclassifiedInvariants => write!(
                f,
                "the measured invariants match no compact manifold of this chart rank"
            ),
        }
    }
}

/// The exact invariants measured on one atlas: the nerve's `GF(2)` homology, its
/// full-nerve Euler characteristic, and the orientation class of the transition
/// signs.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedTopologyInvariants {
    /// Certified local chart rank `d` — the datum the nerve does not have.
    pub intrinsic_dim: usize,
    /// `GF(2)` Betti numbers of the full nerve.
    pub betti: BettiSignature,
    /// `χ = Σ_k (−1)^k N_{k+1}` over EVERY cardinality of co-firing patch set.
    pub euler_characteristic: i128,
    /// Number of admitted nerve simplices at each cardinality (index zero counts
    /// vertices).
    pub simplex_counts: Vec<usize>,
    /// The orientation class `[s] ∈ H¹(N; GF(2))`: `NonOrientable` iff some
    /// 1-cycle carries an odd number of chart-orientation reversals.
    pub orientation_class: AtlasOrientability,
    /// Whether `δs = 0` on every nerve 2-simplex whose three edges all carry a
    /// resolved sign, i.e. whether `s` is a cocycle wherever it is defined. `w₁` is
    /// only defined when this holds.
    pub orientation_cocycle_closes: bool,
    /// Fully-signed nerve triangles on which `δs ≠ 0`.
    pub open_orientation_triangles: usize,
    /// Nerve triangles carrying NO cocycle constraint because at least one of their
    /// edges joins two charts whose tangent planes are orthogonal to within their
    /// own estimation resolution. These are neither closed nor open: `s` is not
    /// defined on them, and counting them as failures would read a chart-conditioning
    /// fact as a topological one.
    pub unsigned_orientation_triangles: usize,
    /// `GF(2)` Betti numbers of the SUBCOMPLEX carrying the sign cochain: the nerve
    /// simplices all of whose edges have a resolved handedness. The orientation
    /// class is read on this complex, so it is a statement about the manifold only
    /// while these match [`ObservedTopologyInvariants::betti`].
    pub signed_subcomplex_betti: BettiSignature,
    /// Patch pairs whose overlap components disagree in sign.
    pub incoherent_overlap_pairs: usize,
    /// Largest number of patches containing a single row. This is exactly the
    /// maximum cardinality any nerve simplex can reach.
    pub max_cover_multiplicity: usize,
    /// Mean number of patches per covered row, `Σ_i |U_i| / |⋃_i U_i|` — the
    /// overlap density the atlas builder actually realized.
    pub mean_cover_multiplicity: f64,
    /// Certified charts that entered the nerve.
    pub chart_count: usize,
    /// Farthest-point centers the atlas had to drop before this readout.
    pub dropped_center_count: usize,
}

/// What the atlas says the data is, plus the invariants it says it from.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasTopologyReadout {
    invariants: ObservedTopologyInvariants,
    orientation_gauge: Vec<i8>,
    twisted_edges: Vec<(usize, usize)>,
    verdict: Result<GraphCompressionKind, TopologyRefusal>,
}

impl AtlasTopologyReadout {
    /// The measured invariants, present whether or not a topology was named.
    #[must_use]
    pub fn invariants(&self) -> &ObservedTopologyInvariants {
        &self.invariants
    }

    /// The orientation gauge: one sign per chart, propagated over a spanning
    /// forest of the nerve's 1-skeleton.
    ///
    /// A single chart's frame sign is arbitrary — flipping it flips the raw sign of
    /// every edge at that chart — so RAW transition signs say nothing on their own.
    /// Only the gauge-corrected sign `o_a · s_ab · o_b` is meaningful, and it is
    /// `+1` on every edge of the spanning forest by construction.
    #[must_use]
    pub fn orientation_gauge(&self) -> &[i8] {
        &self.orientation_gauge
    }

    /// The edges that CANNOT be made orientation-preserving by any choice of chart
    /// signs: the ones whose gauge-corrected sign is `−1`.
    ///
    /// This is `w₁` localized. It is empty exactly when `[s] = 0` (the manifold is
    /// orientable), and on a Möbius band it is the handful of edges that close the
    /// half-twist. WHICH edges appear depends on the spanning forest, but whether
    /// the list is empty does not, and neither does the class it witnesses.
    #[must_use]
    pub fn twisted_edges(&self) -> &[(usize, usize)] {
        &self.twisted_edges
    }

    /// The recognized manifold, or `None` when the atlas refused.
    #[must_use]
    pub fn observed_manifold(&self) -> Option<GraphCompressionKind> {
        self.verdict.as_ref().ok().copied()
    }

    /// Why no topology was named, or `None` when one was.
    #[must_use]
    pub fn refusal(&self) -> Option<&TopologyRefusal> {
        self.verdict.as_ref().err()
    }

    /// Whether the recognized manifold is non-orientable — POSITIVE evidence of a
    /// twist, distinct from the absence of evidence an orientable reading carries.
    ///
    /// [`LocalAtlas::observed_orientability`] returns `Orientable` vacuously when
    /// there are no well-conditioned edges to contradict it, so an orientable
    /// reading may only ever fail to raise the twisted candidates; it must never
    /// veto them. Requiring a NAMED non-orientable manifold makes that asymmetry
    /// structural: the twist has to have survived cocycle closure, connectivity,
    /// and the classification table before it counts as observed.
    #[must_use]
    pub fn observes_non_orientable(&self) -> bool {
        matches!(
            self.observed_manifold(),
            Some(
                GraphCompressionKind::MobiusStrip
                    | GraphCompressionKind::KleinBottle
                    | GraphCompressionKind::ProjectivePlane
            )
        )
    }
}

impl fmt::Display for AtlasTopologyReadout {
    /// One legible line for the fit diagnostic: what was recognized (or why not)
    /// and the invariants it rests on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inv = &self.invariants;
        match &self.verdict {
            Ok(kind) => write!(f, "atlas observes {}", observed_manifold_name(*kind))?,
            Err(refusal) => write!(f, "atlas named no topology ({refusal})")?,
        }
        write!(
            f,
            ": d={} charts={} b0={} b1={} b2={} chi={} orientation={} \
             open_tri={} unsigned_tri={} signed_b0={} signed_b1={} dropped_centers={}",
            inv.intrinsic_dim,
            inv.chart_count,
            inv.betti.b0,
            inv.betti.b1,
            inv.betti
                .b2
                .map_or_else(|| "?".to_string(), |b2| b2.to_string()),
            inv.euler_characteristic,
            match inv.orientation_class {
                AtlasOrientability::Orientable => "trivial",
                AtlasOrientability::NonOrientable => "twisted",
            },
            inv.open_orientation_triangles,
            inv.unsigned_orientation_triangles,
            inv.signed_subcomplex_betti.b0,
            inv.signed_subcomplex_betti.b1,
            inv.dropped_center_count,
        )
    }
}

/// Stable snake-case name of a recognized manifold, for diagnostics.
#[must_use]
pub fn observed_manifold_name(kind: GraphCompressionKind) -> &'static str {
    match kind {
        GraphCompressionKind::Circle => "circle",
        GraphCompressionKind::Interval => "interval",
        GraphCompressionKind::FiniteSet => "finite_set",
        GraphCompressionKind::Disk => "disk",
        GraphCompressionKind::Cylinder => "cylinder",
        GraphCompressionKind::MobiusStrip => "mobius_strip",
        GraphCompressionKind::Torus => "torus",
        GraphCompressionKind::Sphere => "sphere",
        GraphCompressionKind::ProjectivePlane => "projective_plane",
        GraphCompressionKind::KleinBottle => "klein_bottle",
        GraphCompressionKind::Graph => "graph",
    }
}

/// Read the topology the atlas's charts and transition holonomy determine.
///
/// Pure and deterministic: the nerve is enumerated in canonical vertex order over
/// sorted patch memberships, the homology is exact `GF(2)` linear algebra, and the
/// orientation class comes from the atlas's own well-conditioned sign cocycle. No
/// fit, no RNG, no threshold.
#[must_use]
pub fn observe_atlas_topology(atlas: &LocalAtlas) -> Result<AtlasTopologyReadout, String> {
    let chart_count = atlas.chart_count();
    let intrinsic_dim = atlas.intrinsic_dim();
    let dropped_center_count = atlas.rejected_centers().len();

    if chart_count == 0 {
        return Ok(AtlasTopologyReadout {
            invariants: ObservedTopologyInvariants {
                intrinsic_dim,
                betti: BettiSignature {
                    b0: 0,
                    b1: 0,
                    b2: Some(0),
                },
                euler_characteristic: 0,
                simplex_counts: Vec::new(),
                orientation_class: AtlasOrientability::Orientable,
                orientation_cocycle_closes: true,
                open_orientation_triangles: 0,
                unsigned_orientation_triangles: 0,
                signed_subcomplex_betti: BettiSignature {
                    b0: 0,
                    b1: 0,
                    b2: Some(0),
                },
                incoherent_overlap_pairs: 0,
                max_cover_multiplicity: 0,
                mean_cover_multiplicity: 0.0,
                chart_count,
                dropped_center_count,
            },
            orientation_gauge: Vec::new(),
            twisted_edges: Vec::new(),
            verdict: Err(TopologyRefusal::EmptyAtlas),
        });
    }

    // TWO DIFFERENT OBJECTS, on purpose.
    //
    // The NERVE is a combinatorial fact about the cover: patches `a` and `b` are
    // joined iff their supports meet. A fitted transition additionally requires
    // enough shared rows; that sample requirement cannot erase an intersection
    // (#2280: on `swiss_roll(80, 16)` the transition list omitted two singleton
    // intersections that the membership nerve restores). Nothing about how well
    // the two tangent planes are resolved changes which patches intersect, so
    // nothing about it may change the nerve's homology — otherwise a
    // chart-conditioning artifact would move `b₁`.
    //
    // The SIGN COCHAIN `s` lives on top of that nerve and is defined only on the
    // edges whose handedness is resolved (see
    // [`super::local_charts::TransitionConditioning`]). On a cover coarse enough
    // for some patch pair's tangent planes to be orthogonal to within their own
    // estimation resolution, `s` has a genuine hole there — and a hole in `s` is
    // not the same event as `δs ≠ 0`, which is why the two are counted separately
    // below.
    //
    // A pair whose several overlap components disagree in sign has a disconnected
    // intersection: the nerve theorem's hypothesis fails outright. Count it and
    // refuse the topology claim, but retain the observed combinatorial nerve.
    // Only the undefined sign is removed from the sign cochain.
    let mut pair_sign: BTreeMap<(usize, usize), i8> = BTreeMap::new();
    let mut incoherent: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (a, b, _, sign) in atlas.observed_signed_edges() {
        let key = (a.min(b), a.max(b));
        match pair_sign.get(&key) {
            Some(&existing) if existing != sign => {
                incoherent.insert(key);
            }
            Some(_) => {}
            None => {
                pair_sign.insert(key, sign);
            }
        }
    }
    for key in &incoherent {
        pair_sign.remove(key);
    }

    // Patch memberships are sorted row-index lists, so "the intersection is
    // non-empty" is an exact sorted-set predicate — no mass, no tolerance.
    let members: Vec<&[usize]> = atlas
        .patches()
        .iter()
        .map(|patch| patch.members.as_slice())
        .collect();
    if members.len() != chart_count {
        return Err(format!(
            "atlas has {chart_count} charts but {} patches",
            members.len()
        ));
    }

    // Cover multiplicity, measured before anything is enumerated: the maximum is
    // exactly the largest simplex cardinality the nerve can reach, so a pile-up is
    // caught as a statement about the COVER rather than discovered as an
    // enumeration that never terminates.
    let mut row_charts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (chart, rows) in members.iter().enumerate() {
        for &row in *rows {
            row_charts.entry(row).or_default().push(chart);
        }
    }
    let covered_rows = row_charts.len();
    let ambient_max_cover_multiplicity = row_charts.values().map(Vec::len).max().unwrap_or(0);
    let ambient_mean_cover_multiplicity = if covered_rows == 0 {
        0.0
    } else {
        row_charts.values().map(Vec::len).sum::<usize>() as f64 / covered_rows as f64
    };
    let (intrinsic_max, intrinsic_mean) = atlas.intrinsic_cover_multiplicity();
    let (max_cover_multiplicity, mean_cover_multiplicity) =
        if intrinsic_max > ambient_max_cover_multiplicity {
            (intrinsic_max, intrinsic_mean)
        } else {
            (
                ambient_max_cover_multiplicity,
                ambient_mean_cover_multiplicity,
            )
        };
    // Lebesgue covering dimension: a good cover of a d-manifold refines to
    // multiplicity ≤ d + 1. Applied to the realized mean rather than to 1, because
    // this cover is deliberately unrefined — the builder grows every patch past its
    // Voronoi cell precisely so neighbors overlap, and that design point is the
    // measured mean.
    let admissible_multiplicity =
        (mean_cover_multiplicity * (intrinsic_dim.max(1) + 1) as f64).ceil() as usize;
    if max_cover_multiplicity > admissible_multiplicity {
        return Ok(AtlasTopologyReadout {
            invariants: ObservedTopologyInvariants {
                intrinsic_dim,
                betti: BettiSignature {
                    b0: 0,
                    b1: 0,
                    b2: None,
                },
                euler_characteristic: 0,
                simplex_counts: Vec::new(),
                orientation_class: atlas.observed_orientability(),
                orientation_cocycle_closes: false,
                open_orientation_triangles: 0,
                unsigned_orientation_triangles: 0,
                signed_subcomplex_betti: BettiSignature {
                    b0: 0,
                    b1: 0,
                    b2: Some(0),
                },
                incoherent_overlap_pairs: incoherent.len(),
                max_cover_multiplicity,
                mean_cover_multiplicity,
                chart_count,
                dropped_center_count,
            },
            orientation_gauge: Vec::new(),
            twisted_edges: Vec::new(),
            verdict: Err(TopologyRefusal::CoverMultiplicityTooHigh {
                observed: max_cover_multiplicity,
                admissible: admissible_multiplicity,
                mean: mean_cover_multiplicity,
            }),
        });
    }

    // Every co-firing pair is an edge, irrespective of whether its overlap can
    // fit a transition. Built from the sparse membership incidence already
    // measured above: work scales with actual row multiplicities, without an
    // all-patch-pairs intersection scan. The pile-up guard runs before this.
    let mut adjacency = vec![BTreeSet::<usize>::new(); chart_count];
    for charts in row_charts.values() {
        for (position, &a) in charts.iter().enumerate() {
            for &b in &charts[(position + 1)..] {
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }
    }

    let nonempty = |simplex: &[usize]| -> bool {
        let Some((&first, rest)) = simplex.split_first() else {
            return false;
        };
        let mut shared: Vec<usize> = members[first].to_vec();
        for &next in rest {
            shared = sorted_intersection(&shared, members[next]);
            if shared.is_empty() {
                return false;
            }
        }
        !shared.is_empty()
    };

    let inventory = enumerate_full_nerve(chart_count, &nonempty, &adjacency, None)?;
    let betti = compute_betti(
        &inventory.vertices,
        &inventory.edges,
        &inventory.triangles,
        &inventory.tetrahedra,
    );

    // δs on every 2-simplex of the nerve. The sign cochain closes exactly when the
    // three pairwise frame determinants compose, which is the Z/2 reduction of the
    // O(d) cocycle condition and carries no curvature term.
    let mut open_orientation_triangles = 0usize;
    let mut unsigned_orientation_triangles = 0usize;
    for triangle in &inventory.triangles {
        let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
        match (
            pair_sign.get(&(a, b)),
            pair_sign.get(&(a, c)),
            pair_sign.get(&(b, c)),
        ) {
            (Some(&ab), Some(&ac), Some(&bc)) => {
                if ab * ac * bc != 1 {
                    open_orientation_triangles += 1;
                }
            }
            // At least one edge's handedness is unresolved, so this triangle states
            // nothing about `s` — neither that it closes nor that it fails to.
            _ => unsigned_orientation_triangles += 1,
        }
    }

    // Reduce `s` to a gauge: propagate a coherent chart sign over a spanning forest
    // of the 1-skeleton, so every forest edge is orientation-preserving by
    // construction and the remaining reversals are irreducible. This is
    // `LocalAtlas::observed_orientability`'s statement refined from a bit to the
    // WITNESS — which edges carry the obstruction — and the two are asserted to
    // agree by `the_gauge_and_the_atlas_agree_on_orientability_2280`.
    // The subcomplex the sign cochain actually lives on: every nerve simplex all of
    // whose edges have a resolved handedness. Its homology is what decides whether
    // the orientation class read off `s` is the manifold's — see
    // [`TopologyRefusal::OrientationSubcomplexIncomplete`].
    let all_edges_signed = |simplex: &Vec<usize>| -> bool {
        simplex.iter().enumerate().all(|(i, &a)| {
            simplex[(i + 1)..]
                .iter()
                .all(|&b| pair_sign.contains_key(&(a.min(b), a.max(b))))
        })
    };
    let signed_edges: Vec<Vec<usize>> = inventory
        .edges
        .iter()
        .filter(|simplex| all_edges_signed(simplex))
        .cloned()
        .collect();
    let signed_triangles: Vec<Vec<usize>> = inventory
        .triangles
        .iter()
        .filter(|simplex| all_edges_signed(simplex))
        .cloned()
        .collect();
    let signed_tetrahedra: Vec<Vec<usize>> = inventory
        .tetrahedra
        .iter()
        .filter(|simplex| all_edges_signed(simplex))
        .cloned()
        .collect();
    let signed_subcomplex_betti = compute_betti(
        &inventory.vertices,
        &signed_edges,
        &signed_triangles,
        &signed_tetrahedra,
    );

    let mut signed_adjacency = vec![BTreeSet::<usize>::new(); chart_count];
    for &(a, b) in pair_sign.keys() {
        signed_adjacency[a].insert(b);
        signed_adjacency[b].insert(a);
    }
    let (orientation_gauge, twisted_edges) =
        orientation_gauge(chart_count, &signed_adjacency, &pair_sign);
    let orientation_class = if twisted_edges.is_empty() {
        AtlasOrientability::Orientable
    } else {
        AtlasOrientability::NonOrientable
    };

    let invariants = ObservedTopologyInvariants {
        intrinsic_dim,
        betti,
        euler_characteristic: inventory.euler_characteristic,
        simplex_counts: inventory.counts.clone(),
        orientation_class,
        orientation_cocycle_closes: open_orientation_triangles == 0,
        open_orientation_triangles,
        unsigned_orientation_triangles,
        signed_subcomplex_betti,
        incoherent_overlap_pairs: incoherent.len(),
        max_cover_multiplicity,
        mean_cover_multiplicity,
        chart_count,
        dropped_center_count,
    };

    let verdict = classify(&invariants, inventory.triangles.len());
    Ok(AtlasTopologyReadout {
        invariants,
        orientation_gauge,
        twisted_edges,
        verdict,
    })
}

/// Reduce the sign cochain to a gauge: one coherent sign per chart plus the edges
/// no gauge can fix.
///
/// Breadth-first over a spanning forest of the SIGNED 1-skeleton, roots taken in
/// chart order and neighbours in sorted order, so the forest — and therefore the
/// returned witness set — is deterministic. An edge to an already-oriented chart
/// whose sign contradicts the assignment cannot be repaired by any relabelling
/// reachable from this root, and is returned as a twisted edge. The gauge is left
/// untouched by such an edge, so one twist does not cascade into a spurious second
/// one.
///
/// Whether the gauge it produces is a statement about the manifold is decided
/// separately, by comparing the signed subcomplex's homology with the nerve's; see
/// [`TopologyRefusal::OrientationSubcomplexIncomplete`].
fn orientation_gauge(
    chart_count: usize,
    adjacency: &[BTreeSet<usize>],
    pair_sign: &BTreeMap<(usize, usize), i8>,
) -> (Vec<i8>, Vec<(usize, usize)>) {
    let mut gauge = vec![0i8; chart_count];
    let mut twisted: BTreeSet<(usize, usize)> = BTreeSet::new();
    for root in 0..chart_count {
        if gauge[root] != 0 {
            continue;
        }
        gauge[root] = 1;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(chart) = queue.pop_front() {
            let here = gauge[chart];
            for &next in &adjacency[chart] {
                let key = (chart.min(next), chart.max(next));
                let Some(&sign) = pair_sign.get(&key) else {
                    continue;
                };
                let required = here * sign;
                if gauge[next] == 0 {
                    gauge[next] = required;
                    queue.push_back(next);
                } else if gauge[next] != required {
                    twisted.insert(key);
                }
            }
        }
    }
    (gauge, twisted.into_iter().collect())
}

/// Name the manifold the invariants determine, or say why they cannot.
///
/// The order of the refusals is the order of the preconditions the nerve theorem
/// needs, strongest first: a cover that is disconnected or whose intersections are
/// disconnected fails before any homology is worth interpreting; an open sign
/// cochain means `w₁` is not defined; a 2-cell-free nerve means `b₁` is measuring
/// the overlap graph rather than the manifold. Only what survives all of them
/// reaches the classification table.
fn classify(
    invariants: &ObservedTopologyInvariants,
    triangle_count: usize,
) -> Result<GraphCompressionKind, TopologyRefusal> {
    if invariants.incoherent_overlap_pairs > 0 {
        return Err(TopologyRefusal::IncoherentOverlap {
            pairs: invariants.incoherent_overlap_pairs,
        });
    }
    if invariants.betti.b0 != 1 {
        return Err(TopologyRefusal::DisconnectedCover {
            components: invariants.betti.b0,
        });
    }
    if !invariants.orientation_cocycle_closes {
        return Err(TopologyRefusal::OrientationCocycleOpen {
            open_triangles: invariants.open_orientation_triangles,
            total: triangle_count - invariants.unsigned_orientation_triangles,
        });
    }
    // A cocycle that closes only because it is defined nowhere states nothing, and
    // one defined on a complex with cycles the nerve does not have states something
    // about the wrong space. Both are the same check: the subcomplex carrying `s`
    // must see the nerve's 1-cycles and no others.
    if invariants.signed_subcomplex_betti.b0 != invariants.betti.b0
        || invariants.signed_subcomplex_betti.b1 != invariants.betti.b1
    {
        return Err(TopologyRefusal::OrientationSubcomplexIncomplete {
            signed_b0: invariants.signed_subcomplex_betti.b0,
            signed_b1: invariants.signed_subcomplex_betti.b1,
            nerve_b0: invariants.betti.b0,
            nerve_b1: invariants.betti.b1,
        });
    }
    match invariants.intrinsic_dim {
        // One-manifolds. A connected 1-manifold is the circle or the line, and the
        // two are separated by whether the cover closes up: `b₁ = 1, χ = 0` is
        // `S¹`, `b₁ = 0, χ = 1` is the interval. There is no orientation case —
        // every 1-manifold whose charts certify is orientable — and no 2-cell
        // precondition, because the nerve of an arc cover IS a graph and its cycle
        // is the manifold's, not an artifact.
        1 => match (invariants.betti.b1, invariants.euler_characteristic) {
            (1, 0) => Ok(GraphCompressionKind::Circle),
            (0, 1) => Ok(GraphCompressionKind::Interval),
            _ => Err(TopologyRefusal::UnclassifiedInvariants),
        },
        // Surfaces. `(χ, orientability)` is a complete invariant, but only once the
        // nerve has 2-cells to quotient the overlap graph's spurious cycles by —
        // the sphere's overlap graph is full of cycles that all bound.
        2 => {
            if triangle_count == 0 {
                return Err(TopologyRefusal::NoTwoCells);
            }
            surface_from_invariants(
                invariants.betti,
                invariants.euler_characteristic,
                invariants.orientation_class,
            )
            .map(|(kind, _)| kind)
            .ok_or(TopologyRefusal::UnclassifiedInvariants)
        }
        // `d ≥ 3`: the classification of manifolds is not a finite table, so there
        // is nothing honest to name. The invariants are still reported.
        _ => Err(TopologyRefusal::UnclassifiedInvariants),
    }
}

#[cfg(test)]
#[path = "tests_nerve_membership_2280.rs"]
mod tests_nerve_membership_2280;

#[cfg(test)]
mod tests_2280 {
    use super::*;
    use crate::manifold::LocalAtlas;
    use crate::manifold::local_charts::LocalAtlasConfig;
    use crate::manifold::tests_topology_fixtures::{
        circle, cylinder_strip, embedded_plane, mobius_strip, open_arc, sphere, spherical_band,
        swiss_roll, torus, trefoil_knot,
    };
    use ndarray::{Array2, ArrayView2};

    /// Build the atlas at the builder's own derived configuration and read out the
    /// topology. Every test goes through the production entry points.
    fn read(z: ArrayView2<'_, f64>, d: usize) -> AtlasTopologyReadout {
        let config = LocalAtlasConfig::balanced(z.nrows(), d);
        let atlas = LocalAtlas::build(z, config).expect("fixture atlas must build");
        observe_atlas_topology(&atlas).expect("fixture readout must not error")
    }

    /// A round circle is `S¹`: one 1-cycle in the nerve's `GF(2)` homology, `χ = 0`,
    /// trivial orientation class.
    #[test]
    fn circle_is_recognized_at_chart_rank_one_2280() {
        let readout = read(circle(200, 2.0).view(), 1);
        assert_eq!(
            readout.observed_manifold(),
            Some(GraphCompressionKind::Circle),
            "{readout}"
        );
        let inv = readout.invariants();
        assert_eq!((inv.betti.b0, inv.betti.b1), (1, 1), "{readout}");
        assert_eq!(inv.euler_characteristic, 0, "{readout}");
        assert!(inv.orientation_cocycle_closes, "{readout}");
    }

    /// An open arc is an interval: no cycle, `χ = 1`. This is the `b₁` half of the
    /// one-manifold table — the cover of an arc is a chain, the cover of a circle
    /// closes up, and nothing else about them differs.
    #[test]
    fn open_arc_is_an_interval_2280() {
        let readout = read(open_arc(200, 2.0).view(), 1);
        assert_eq!(
            readout.observed_manifold(),
            Some(GraphCompressionKind::Interval),
            "{readout}"
        );
        let inv = readout.invariants();
        assert_eq!(inv.betti.b1, 0, "{readout}");
        assert_eq!(inv.euler_characteristic, 1, "{readout}");
    }

    /// Regression for the sampled-cover hole localized in the issue thread.  With
    /// ambient-distance patches the four inner-end charts centered at rows
    /// 0/13/131/143 formed a real `H1` representative even though the underlying
    /// sheet is contractible.  The independent intrinsic realization exposes a
    /// cover pile-up at that fold, so promotion is honestly refused instead of
    /// treating the ambient cover's accidental cycle as a cylinder.
    #[test]
    fn dense_swiss_roll_is_never_promoted_as_cylinder_2280() {
        let roll = swiss_roll(80, 16);
        let first = read(roll.view(), 2);
        let second = read(roll.view(), 2);

        assert_eq!(
            first, second,
            "atlas construction and readout must be deterministic"
        );
        assert_ne!(
            first.observed_manifold(),
            Some(GraphCompressionKind::Cylinder),
            "a rolled contractible sheet may be named a disk or honestly refused, but must never be promoted as a cylinder: {first}"
        );
    }

    /// THE ambient-embedding test. A trefoil knot is a smooth `S¹` whose three
    /// ambient principal directions all carry comparable spread, so no global-linear
    /// seed recovers the loop — and the atlas returns the SAME verdict for it as for
    /// a round circle, because every input to that verdict is a transition between
    /// overlapping charts and a transition is intrinsic.
    #[test]
    fn trefoil_knot_is_a_circle_and_pca_cannot_see_it_2280() {
        let knot = trefoil_knot(600, 1.0);

        // The global-linear seed's view: all three singular values comparable.
        let mut centered = knot.clone();
        let mean: Vec<f64> = (0..3)
            .map(|c| knot.column(c).iter().sum::<f64>() / knot.nrows() as f64)
            .collect();
        for row in 0..centered.nrows() {
            for c in 0..3 {
                centered[[row, c]] -= mean[c];
            }
        }
        let (_, singular, _) = gam_linalg::faer_ndarray::FaerSvd::svd(&centered, false, false)
            .expect("trefoil SVD must succeed");
        let spread = singular[2] / singular[0];
        assert!(
            spread > 0.4,
            "the trefoil's third ambient direction must carry comparable spread \
             (sigma3/sigma1 = {spread}), else a linear seed could recover the loop"
        );

        let knot_readout = read(knot.view(), 1);
        assert_eq!(
            knot_readout.observed_manifold(),
            Some(GraphCompressionKind::Circle),
            "the trefoil is intrinsically S¹: {knot_readout}"
        );
        let round_readout = read(circle(600, 2.0).view(), 1);
        assert_eq!(
            knot_readout.observed_manifold(),
            round_readout.observed_manifold(),
            "the ambient knotting must not change the intrinsic verdict: \
             knot={knot_readout} round={round_readout}"
        );
        assert_eq!(
            (
                knot_readout.invariants().betti.b1,
                knot_readout.invariants().euler_characteristic
            ),
            (
                round_readout.invariants().betti.b1,
                round_readout.invariants().euler_characteristic
            ),
            "the knot and the round circle must agree invariant for invariant"
        );
    }

    /// A flat sheet is a disk: contractible, `χ = 1`, no closed 2-cycle. This is the
    /// `χ` half of the simply-connected pair — the discriminant that separates it
    /// from the sphere.
    #[test]
    fn flat_sheet_is_a_disk_2280() {
        let readout = read(embedded_plane(24, 24).view(), 2);
        assert_eq!(
            readout.observed_manifold(),
            Some(GraphCompressionKind::Disk),
            "{readout}"
        );
        let inv = readout.invariants();
        assert_eq!((inv.betti.b1, inv.betti.b2), (0, Some(0)), "{readout}");
        assert_eq!(inv.euler_characteristic, 1, "{readout}");
    }

    /// The closed 2-sphere is recognized by `χ = 2` and its closed 2-cycle, at three
    /// independent sample sizes. Holonomy is blind here — `S²` is simply connected,
    /// so EVERY loop is contractible and the orientation class is trivial no matter
    /// what — which is exactly why the nerve's `χ` has to carry this case.
    #[test]
    fn sphere_is_recognized_by_its_euler_characteristic_2280() {
        for n in [400usize, 900, 1600] {
            let readout = read(sphere(n).view(), 2);
            assert_eq!(
                readout.observed_manifold(),
                Some(GraphCompressionKind::Sphere),
                "n={n}: {readout}"
            );
            let inv = readout.invariants();
            assert_eq!(inv.euler_characteristic, 2, "n={n}: {readout}");
            assert_eq!(
                (inv.betti.b1, inv.betti.b2),
                (0, Some(1)),
                "n={n}: {readout}"
            );
        }
    }

    /// THE holonomy test. A cylinder and a Möbius band have IDENTICAL `GF(2)`
    /// homology and identical `χ` — every invariant of the nerve agrees — and are
    /// separated only by whether the transition signs admit a globally coherent
    /// choice. The half-twist is a discrete cohomology class, and it is the only
    /// thing that tells them apart.
    #[test]
    fn cylinder_and_mobius_differ_only_by_the_holonomy_class_2280() {
        let cyl = read(cylinder_strip(40, 10).view(), 2);
        let mob = read(mobius_strip(40, 10).view(), 2);

        let (ci, mi) = (cyl.invariants(), mob.invariants());
        assert_eq!(
            (
                ci.betti.b0,
                ci.betti.b1,
                ci.betti.b2,
                ci.euler_characteristic
            ),
            (
                mi.betti.b0,
                mi.betti.b1,
                mi.betti.b2,
                mi.euler_characteristic
            ),
            "the cylinder and the Möbius band must be homologically indistinguishable: \
             cyl={cyl} mob={mob}"
        );
        assert!(ci.orientation_cocycle_closes && mi.orientation_cocycle_closes);
        assert_eq!(
            ci.orientation_class,
            AtlasOrientability::Orientable,
            "{cyl}"
        );
        assert_eq!(
            mi.orientation_class,
            AtlasOrientability::NonOrientable,
            "{mob}"
        );
        assert_eq!(
            cyl.observed_manifold(),
            Some(GraphCompressionKind::Cylinder),
            "{cyl}"
        );
        assert_eq!(
            mob.observed_manifold(),
            Some(GraphCompressionKind::MobiusStrip),
            "{mob}"
        );
        assert!(!cyl.observes_non_orientable() && mob.observes_non_orientable());
    }

    /// A band cut out of a sphere is an annulus — a cylinder with curvature — and
    /// the readout says so. It is the guard against reading ambient curvature as
    /// topology: the surface is as curved as a sphere everywhere, and none of that
    /// enters the verdict, because removing the two polar caps removes the two
    /// 2-cells that make `χ = 2`.
    #[test]
    fn a_spherical_band_is_a_cylinder_not_a_sphere_2280() {
        let readout = read(spherical_band(20, 26).view(), 2);
        assert_eq!(
            readout.observed_manifold(),
            Some(GraphCompressionKind::Cylinder),
            "a pole-free spherical band is an annulus: {readout}"
        );
        assert_eq!(readout.invariants().euler_characteristic, 0, "{readout}");
    }

    /// The torus is NAMED once its cover resolves the tangent planes, and refuses
    /// honestly until then — with its homology exact the whole way.
    ///
    /// Three things are pinned here, and none of them holds before the derived sign
    /// gate ([`super::local_charts`]'s `frame_angular_resolution`) and the
    /// nerve/cochain separation in [`observe_atlas_topology`]:
    ///
    /// 1. **`b₁ = 2`, `b₂ = 1`, `χ = 0` at every resolution.** The nerve is the
    ///    cover's intersection pattern, so no chart-conditioning verdict may move
    ///    it. Gating the sign WITHOUT separating the two objects measures
    ///    `b₁ = 35` on the coarsest arm — the homology of whichever edges happened
    ///    to resolve.
    /// 2. **The fine cover names `Torus`, orientation trivial.** Under the retired
    ///    `|det| > 1e-6` floor this arm refused with 172 open triangles: the floor
    ///    admitted a handedness from tangent planes meeting at `89.4°`, and those
    ///    fabricated signs are what failed to compose.
    /// 3. **The coarse covers refuse, and refuse for the right reason.** Their
    ///    charts capture only `~0.92` of their neighborhood variance, so each frame
    ///    is pinned to about `16°` and a pair of them to `~31°`; roughly half the
    ///    edges carry no resolvable handedness, the subcomplex `s` lives on has
    ///    `b₁ = 35` against the nerve's `2`, and an orientation class read there is
    ///    not the manifold's. Naming it anyway produces `KleinBottle` — measured,
    ///    not hypothesised, which is why the subcomplex-completeness gate exists.
    #[test]
    fn torus_is_named_once_its_cover_resolves_the_tangent_planes_2280() {
        for (n_u, n_v, major, minor) in [
            (48usize, 20usize, 2.0, 0.8),
            (60, 26, 2.0, 0.8),
            (90, 45, 3.0, 1.5),
        ] {
            let readout = read(torus(n_u, n_v, major, minor).view(), 2);
            let inv = readout.invariants();
            assert_eq!(
                (
                    inv.betti.b0,
                    inv.betti.b1,
                    inv.betti.b2,
                    inv.euler_characteristic
                ),
                (1, 2, Some(1), 0),
                "the torus's homology must be measured exactly at every resolution: {readout}"
            );
        }

        let fine = read(torus(90, 45, 3.0, 1.5).view(), 2);
        assert_eq!(
            fine.observed_manifold(),
            Some(GraphCompressionKind::Torus),
            "the resolved cover must name the torus: {fine}"
        );
        assert_eq!(
            fine.invariants().orientation_class,
            AtlasOrientability::Orientable,
            "{fine}"
        );
        assert!(
            fine.twisted_edges().is_empty(),
            "an orientable surface admits a coherent gauge: {fine}"
        );

        for (n_u, n_v) in [(48usize, 20usize), (60, 26)] {
            let coarse = read(torus(n_u, n_v, 2.0, 0.8).view(), 2);
            let inv = coarse.invariants();
            assert!(
                inv.unsigned_orientation_triangles > 0,
                "the coarse cover must have triangles `s` is not defined on: {coarse}"
            );
            assert!(
                inv.signed_subcomplex_betti.b1 > inv.betti.b1,
                "the signed subcomplex must carry cycles the nerve fills: {coarse}"
            );
            assert!(
                matches!(
                    coarse.refusal(),
                    Some(TopologyRefusal::OrientationSubcomplexIncomplete { .. })
                ),
                "the coarse cover must refuse on the subcomplex, not name a surface: {coarse}"
            );
        }
    }

    /// Measurement, not a gate: what the torus's sign gate actually sees, edge by
    /// edge. Prints `σ_min(F_toᵀ F_from)` against the two charts' combined angular
    /// resolution, plus what the retired `|det| > 1e-6` floor would have admitted,
    /// so the difference between the two gates is a table rather than a claim.
    #[test]
    fn tests_zz_measure_2280_torus_sign_gate() {
        for (n_u, n_v) in [(48usize, 20usize), (60, 26)] {
            let z = torus(n_u, n_v, 2.0, 0.8);
            let config = LocalAtlasConfig::balanced(z.nrows(), 2);
            let atlas = LocalAtlas::build(z.view(), config).expect("torus atlas must build");
            let admitted = atlas.observed_signed_edges().len();
            let mut margins: Vec<(f64, f64, f64, usize, usize)> = atlas
                .transitions()
                .iter()
                .map(|t| {
                    (
                        t.smallest_principal_cosine,
                        t.sign_resolution_budget,
                        t.residual,
                        t.from_patch,
                        t.to_patch,
                    )
                })
                .collect();
            margins.sort_by(|a, b| {
                (a.0 - a.1)
                    .partial_cmp(&(b.0 - b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let det_floor_admitted = margins
                .iter()
                .filter(|(cos_min, _, _, _, _)| *cos_min > 1.0e-6)
                .count();
            eprintln!(
                "[#2280 torus {n_u}x{n_v}] charts={} transitions={} admitted_now={admitted} \
                 admitted_under_1e-6_cosine_floor={det_floor_admitted}",
                atlas.chart_count(),
                atlas.transitions().len(),
            );
            for (cos_min, budget, residual, a, b) in margins.iter().take(12) {
                eprintln!(
                    "  edge {a:>3}-{b:<3} sigma_min={cos_min:.6e} budget={budget:.6e} \
                     margin={:+.3e} procrustes_residual={residual:.3e}",
                    cos_min - budget
                );
            }
            let caps: Vec<f64> = atlas
                .charts()
                .iter()
                .map(|c| c.certificate.captured_variance_fraction)
                .collect();
            let worst = caps.iter().copied().fold(1.0_f64, f64::min);
            let best = caps.iter().copied().fold(0.0_f64, f64::max);
            eprintln!("  captured_variance_fraction: min={worst:.9} max={best:.9}");
            let readout = observe_atlas_topology(&atlas).expect("readout");
            eprintln!("  readout: {readout}");
        }
    }

    /// The property that must hold for every fixture, and must keep holding as the
    /// cover improves: the readout either names the manifold the fixture IS, or
    /// names nothing. It is never allowed to name a different one.
    ///
    /// A verdict is consumed as a proposal-time prior over a candidate menu, so a
    /// refusal costs only the prior while a WRONG name would put the wrong
    /// hypothesis in front of the evidence. This is the asymmetry the gate encodes.
    #[test]
    fn the_readout_never_misnames_a_known_manifold_2280() {
        let cases: Vec<(&str, Array2<f64>, usize, GraphCompressionKind)> = vec![
            ("circle", circle(200, 2.0), 1, GraphCompressionKind::Circle),
            (
                "trefoil",
                trefoil_knot(600, 1.0),
                1,
                GraphCompressionKind::Circle,
            ),
            (
                "trefoil_coarse",
                trefoil_knot(400, 1.0),
                1,
                GraphCompressionKind::Circle,
            ),
            (
                "open_arc",
                open_arc(200, 2.0),
                1,
                GraphCompressionKind::Interval,
            ),
            (
                "plane",
                embedded_plane(24, 24),
                2,
                GraphCompressionKind::Disk,
            ),
            (
                "cylinder",
                cylinder_strip(40, 10),
                2,
                GraphCompressionKind::Cylinder,
            ),
            (
                "cylinder_big",
                cylinder_strip(60, 14),
                2,
                GraphCompressionKind::Cylinder,
            ),
            (
                "spherical_band",
                spherical_band(20, 26),
                2,
                GraphCompressionKind::Cylinder,
            ),
            (
                "mobius",
                mobius_strip(40, 10),
                2,
                GraphCompressionKind::MobiusStrip,
            ),
            (
                "mobius_big",
                mobius_strip(60, 14),
                2,
                GraphCompressionKind::MobiusStrip,
            ),
            ("sphere", sphere(900), 2, GraphCompressionKind::Sphere),
            (
                "torus",
                torus(48, 20, 2.0, 0.8),
                2,
                GraphCompressionKind::Torus,
            ),
            (
                "torus_fat",
                torus(60, 30, 3.0, 1.5),
                2,
                GraphCompressionKind::Torus,
            ),
        ];
        for (label, z, d, truth) in cases {
            let readout = read(z.view(), d);
            match readout.observed_manifold() {
                None => {
                    assert!(
                        readout.refusal().is_some(),
                        "{label}: an unnamed readout must carry its reason"
                    );
                }
                Some(named) => assert_eq!(
                    named, truth,
                    "{label}: the readout named a manifold the fixture is not: {readout}"
                ),
            }
        }
    }

    /// Two well-separated sheets are two manifolds, and the readout refuses instead
    /// of naming whichever component the table happens to match.
    #[test]
    fn a_disconnected_cover_refuses_2280() {
        let sheet = embedded_plane(14, 14);
        let n = sheet.nrows();
        let mut z = Array2::<f64>::zeros((2 * n, 4));
        for row in 0..n {
            for c in 0..4 {
                z[[row, c]] = sheet[[row, c]];
                // A second copy displaced far beyond any patch radius.
                z[[n + row, c]] = sheet[[row, c]] + if c == 0 { 1.0e4 } else { 0.0 };
            }
        }
        let readout = read(z.view(), 2);
        assert_eq!(readout.observed_manifold(), None, "{readout}");
        assert!(
            matches!(
                readout.refusal(),
                Some(TopologyRefusal::DisconnectedCover { components }) if *components >= 2
            ),
            "{readout}"
        );
    }

    /// The gauge is a refinement of the atlas's own orientability bit, not a second
    /// opinion: on every fixture the two agree, and the twisted-edge witness is
    /// non-empty exactly when the class is non-trivial.
    ///
    /// It also pins the gauge's meaning. A single chart's frame sign is arbitrary,
    /// so RAW transition signs are reversing all over an orientable atlas; only
    /// after the gauge is fixed does a remaining reversal mean anything. The
    /// cylinder must end with ZERO twisted edges even though its raw sign cochain
    /// does not.
    #[test]
    fn the_gauge_and_the_atlas_agree_on_orientability_2280() {
        for (label, z, d) in [
            ("cylinder", cylinder_strip(40, 10), 2usize),
            ("mobius", mobius_strip(40, 10), 2),
            ("sphere", sphere(400), 2),
            ("circle", circle(200, 2.0), 1),
        ] {
            let config = LocalAtlasConfig::balanced(z.nrows(), d);
            let atlas = LocalAtlas::build(z.view(), config).expect("fixture atlas must build");
            let readout = observe_atlas_topology(&atlas).expect("readout must not error");
            assert_eq!(
                readout.invariants().orientation_class,
                atlas.observed_orientability(),
                "{label}: the gauge and the atlas primitive must agree"
            );
            assert_eq!(
                readout.twisted_edges().is_empty(),
                readout.invariants().orientation_class == AtlasOrientability::Orientable,
                "{label}: the witness must be empty exactly when the class is trivial"
            );
            assert!(
                readout.orientation_gauge().iter().all(|&sign| sign != 0),
                "{label}: every chart of a connected cover must receive a gauge sign"
            );
        }
    }

    /// The readout is a pure function of the atlas: same rows in, bit-identical
    /// invariants out. No RNG, no hashing, no float-order ambiguity.
    #[test]
    fn the_readout_is_bit_identical_run_to_run_2280() {
        let z = mobius_strip(40, 10);
        let first = read(z.view(), 2);
        let second = read(z.view(), 2);
        assert_eq!(first, second);
        assert_eq!(
            first.invariants().mean_cover_multiplicity.to_bits(),
            second.invariants().mean_cover_multiplicity.to_bits()
        );
    }
}

#[cfg(test)]
mod tests_zz_measure_2280 {
    use super::*;
    use crate::manifold::local_charts::LocalAtlasConfig;
    use crate::manifold::tests_topology_fixtures::{
        circle, cylinder_strip, embedded_plane, mobius_strip, open_arc, sphere, spherical_band,
        swiss_roll, torus, trefoil_knot,
    };
    use ndarray::Array2;

    /// Build the fixture's atlas, read its topology out, PRINT the measurement --
    /// and hand the caller a verdict it cannot discard.
    ///
    /// This used to swallow BOTH error arms: a bare `return` on a failed build
    /// and a bare `eprintln!` on a failed readout. All twenty-one zoo fixtures
    /// could therefore fail to build AND fail to read out with the test still
    /// reporting green, which is strictly worse than a skip -- a skip at least
    /// announces itself. Both arms are now errors the caller collects.
    ///
    /// Deliberately NOT asserted here: that the readout's intrinsic dimension
    /// equals the `d` the fixture passes in. `LocalAtlas::build` stores
    /// `config.intrinsic_dim` and `observe_atlas_topology` reads it straight
    /// back, so that comparison is a config round-trip that cannot fail on any
    /// input -- adding it would be the same species of defect this change
    /// exists to remove. A real dimension check needs an ESTIMATED rank (from
    /// the local-PCA spectra), which the readout does not currently carry.
    fn report(label: &str, z: &Array2<f64>, d: usize) -> Result<(), String> {
        let config = LocalAtlasConfig::balanced(z.nrows(), d);
        let atlas = crate::manifold::LocalAtlas::build(z.view(), config)
            .map_err(|error| format!("{label}: atlas build failed: {error}"))?;
        match observe_atlas_topology(&atlas) {
            Ok(readout) => eprintln!(
                "{label}: n={} patches={} size={} | {} | counts={:?} open_tri={} incoherent={} \
                 mult_max={} mult_mean={:.2}",
                z.nrows(),
                config.patch_count,
                config.patch_size,
                readout,
                readout.invariants().simplex_counts,
                readout.invariants().open_orientation_triangles,
                readout.invariants().incoherent_overlap_pairs,
                readout.invariants().max_cover_multiplicity,
                readout.invariants().mean_cover_multiplicity,
            ),
            Err(error) => return Err(format!("{label}: readout failed: {error}")),
        }
        Ok(())
    }

    /// Dump the recovered atlas of one fixture in a line format a plotting script
    /// can parse: the ambient points, which patch each center sits on, and the
    /// nerve's signed edges.
    fn dump(label: &str, z: &Array2<f64>, d: usize) {
        let config = LocalAtlasConfig::balanced(z.nrows(), d);
        let atlas = crate::manifold::LocalAtlas::build(z.view(), config)
            .expect("plot fixture must build an atlas");
        let readout = observe_atlas_topology(&atlas).expect("plot fixture must read out");
        for row in 0..z.nrows() {
            eprintln!(
                "PLOT {label} POINT {row} {:.6} {:.6} {:.6}",
                z[[row, 0]],
                z[[row, 1]],
                if z.ncols() > 2 { z[[row, 2]] } else { 0.0 }
            );
        }
        for (patch_idx, patch) in atlas.patches().iter().enumerate() {
            eprintln!("PLOT {label} CENTER {patch_idx} {}", patch.center);
            for &member in &patch.members {
                eprintln!("PLOT {label} MEMBER {patch_idx} {member}");
            }
        }
        // Edges carry their GAUGE-CORRECTED sign. The raw sign is meaningless on its
        // own — flipping one chart's frame flips every raw sign at that chart — so a
        // figure drawn from raw signs would paint an orientable atlas full of
        // reversals. After the gauge, a reversal is the obstruction itself.
        let gauge = readout.orientation_gauge();
        let twisted: std::collections::BTreeSet<(usize, usize)> =
            readout.twisted_edges().iter().copied().collect();
        for (a, b, _, sign) in atlas.observed_signed_edges() {
            let corrected = if twisted.contains(&(a.min(b), a.max(b))) {
                -1
            } else {
                i32::from(gauge[a]) * i32::from(sign) * i32::from(gauge[b])
            };
            eprintln!(
                "PLOT {label} EDGE {} {} {corrected}",
                atlas.patches()[a].center,
                atlas.patches()[b].center
            );
        }
        eprintln!("PLOT {label} VERDICT {readout}");
    }

    #[test]
    fn zz_measure_atlas_plot_dump_2280() {
        dump("trefoil", &trefoil_knot(600, 1.0), 1);
        dump("circle", &circle(200, 2.0), 1);
        dump("cylinder", &cylinder_strip(40, 10), 2);
        dump("mobius", &mobius_strip(40, 10), 2);
        dump("sphere", &sphere(400), 2);
    }

    /// The contract is in the name: a "known topology zoo" is a set of fixtures
    /// whose topology is known, so each must at minimum BUILD an atlas and READ
    /// OUT. The body only printed, so the name was unbacked and all twenty-one
    /// fixtures could fail silently.
    ///
    /// Every fixture is still attempted -- one failure must not hide the other
    /// twenty -- and the assertion names each one that failed.
    #[test]
    fn zz_measure_known_topology_zoo_2280() {
        let mut failures: Vec<String> = Vec::new();
        for outcome in [
            report("circle", &circle(200, 2.0), 1),
            report("circle_big", &circle(400, 2.0), 1),
            report("trefoil_400", &trefoil_knot(400, 1.0), 1),
            report("trefoil_600", &trefoil_knot(600, 1.0), 1),
            report("trefoil_900", &trefoil_knot(900, 1.0), 1),
            report("open_arc", &open_arc(200, 2.0), 1),
            report("plane", &embedded_plane(24, 24), 2),
            report("swiss_roll", &swiss_roll(40, 12), 2),
            report("cylinder", &cylinder_strip(40, 10), 2),
            report("cylinder_big", &cylinder_strip(60, 14), 2),
            report("spherical_band", &spherical_band(20, 26), 2),
            report("mobius", &mobius_strip(40, 10), 2),
            report("mobius_big", &mobius_strip(60, 14), 2),
            report("sphere_400", &sphere(400), 2),
            report("sphere_900", &sphere(900), 2),
            report("sphere_1600", &sphere(1600), 2),
            report("torus_48x20", &torus(48, 20, 2.0, 0.8), 2),
            report("torus_60x26", &torus(60, 26, 2.0, 0.8), 2),
            report("torus_80x34", &torus(80, 34, 2.0, 0.8), 2),
            report("torus_fat_60x30", &torus(60, 30, 3.0, 1.5), 2),
            report("torus_fat_90x45", &torus(90, 45, 3.0, 1.5), 2),
        ] {
            if let Err(failure) = outcome {
                failures.push(failure);
            }
        }
        assert!(
            failures.is_empty(),
            "#2280: {} of 21 known-topology-zoo fixtures failed to build an atlas \
             or read its topology out:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
