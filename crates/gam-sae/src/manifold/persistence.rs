//! Persistent-homology audit of a raced atom's topology (reviewer F3).
//!
//! The topology race (`Periodic`/`Torus`/`Sphere`/`Cylinder`/`EuclideanPatch`/…)
//! is *model selection among a fixed library*: it always crowns a winner, even
//! when every candidate is misspecified. This module turns topology back into a
//! **measured property**. For each accepted atom it reads the atom's assigned
//! rows' in-atom residual-space positions — the decoded image points
//! `g_k(t_{ik}) = Φ_k(t_{ik}) B_k` of the rows this atom explains — and computes
//! their distance-to-measure weighted Vietoris–Rips persistent homology (H₀ =
//! connected components, H₁ = loops, H₂ = shells for sphere/torus candidates).
//! The measured diagram is then confronted with what the *raced* type predicts:
//!
//! * a circle / cylinder should show Betti `(b₀=1, b₁=1)`;
//! * a torus should show Betti `(b₀=1, b₁=2, b₂=1)`;
//! * a sphere should show Betti `(b₀=1, b₁=0, b₂=1)`;
//! * a line / Duchon patch / Euclidean patch / sphere-chart should show Betti
//!   `(b₀=1, b₁=0)`;
//! * a set that is really `c` clusters (e.g. a 7-point ring forced through a
//!   circle fit) shows `c` persistent H₀ bars — disagreeing with *every*
//!   connected candidate in the library.
//!
//! Disagreement raises a first-class [`AtomTopologyPersistence::contested`]
//! certificate flag, which the probe planner reads to schedule a re-adjudication
//! rather than trusting the latched winner.
//!
//! # No magic constants (SPEC.md)
//!
//! The filtration values ARE exact DTM-weighted pairwise distances (no scale
//! grid, no bucketing), so the persistence diagram is computed exactly. "How
//! many components / loops / shells" is decided by a **dominant-gap** test on
//! the bar lengths (a single log-gap that outweighs all the others combined),
//! which carries no threshold. The only compute-side ceiling is the
//! farthest-point subsample cap [`PERSISTENCE_MAX_POINTS`] — a budget on the
//! simplex enumeration, above the covering number of any modest atom, mirroring
//! the in-tree [`crate::manifold::shape_uncertainty::SHAPE_BAND_MAX_POINTS`]
//! band ceiling. Topology is invariant to it above the covering number.

use super::*;
use crate::inference::atlas_nerve::AtlasCoveringSide;
use crate::null_battery::ClaimNullCalibration;
use std::collections::HashMap;

/// Compute ceiling on the number of points fed to the Vietoris–Rips filtration.
///
/// The H₂ audit includes `O(m⁴)` tetrahedra for sphere/torus candidates; the
/// boundary reduction is quadratic in the simplex count, so an uncapped atom
/// (tens of thousands of assigned rows) is intractable and — more importantly —
/// pointless: the persistent topology of a point cloud is fixed once the sample
/// covers the manifold, and a farthest-point subsample to a few dozen landmarks
/// already covers any modest atom. This is a compute budget, not a model knob:
/// raising it only spends more time confirming the same diagram. Kept in the
/// same spirit (and rough magnitude class, scaled down for the cubic cost) as
/// [`crate::manifold::shape_uncertainty::SHAPE_BAND_MAX_POINTS`].
pub const PERSISTENCE_MAX_POINTS: usize = 48;

/// Landmark budget for the H₁ (loop) audit, decoupled from [`PERSISTENCE_MAX_POINTS`].
///
/// Counting loops needs only the triangles of the filtration (`C(m, 3)`, cubic in
/// the landmark count), whereas the H₂ shell audit needs tetrahedra (`C(m, 4)`,
/// quartic) — which is why H₂ is capped at [`PERSISTENCE_MAX_POINTS`]. Resolving
/// the two independent generators of a torus robustly needs a cover at the
/// manifold's *covering number*, far above 48: on the coarse 48-landmark
/// farthest-point cover the two loops are not simultaneously resolved and the
/// second generator collapses (measured `b₁ = 0` on a clean torus, #2159). Since
/// the triangle enumeration is only cubic, H₁ can afford this larger cover at a
/// cost comparable to the quartic H₂ budget. Atoms whose positive support exceeds
/// this fall back to a farthest-point cover for H₁ as well; density-stable
/// homology on a subsampled cover is a Vietoris–Rips limitation whose principled
/// fix is an alpha/witness complex (tracked follow-up), not a larger cover.
pub const PERSISTENCE_H1_MAX_POINTS: usize = 256;

/// A persistence bar `[birth, death)` in a Vietoris–Rips filtration. An
/// *essential* class (one that never dies within the filtration — e.g. the
/// single connected component, or a loop whose disk is never filled) has
/// `death = f64::INFINITY`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PersistenceBar {
    pub birth: f64,
    pub death: f64,
}

impl PersistenceBar {
    /// Bar length `death − birth` (`+∞` for an essential class).
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Whether the class never dies within the filtration.
    pub fn is_essential(&self) -> bool {
        !self.death.is_finite()
    }
}

/// The H₀/H₁/H₂ persistence diagram of a point cloud. `h0` always contains
/// exactly one essential bar (VR on a finite cloud is connected at its
/// diameter); the finite H₀ bars are the merge events. `h1` holds the loops
/// (finite loops die when their disk fills), and `h2` holds shells when the
/// caller asks for sphere/torus homology.
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
    pub h0: Vec<PersistenceBar>,
    pub h1: Vec<PersistenceBar>,
    pub h2: Vec<PersistenceBar>,
}

/// Betti signature used by the topology audit. `b2 = None` means H₂ was not
/// computed for this raced topology because it has no H₂ claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BettiSignature {
    pub b0: usize,
    pub b1: usize,
    pub b2: Option<usize>,
}

impl BettiSignature {
    fn matches_expected(self, expected: Self) -> bool {
        self.b0 == expected.b0
            && self.b1 == expected.b1
            && expected.b2.map_or(true, |b2| self.b2 == Some(b2))
    }
}

/// Which side of the persistence landmark cap this atom's sampled support is
/// on. At the cap, the audit reads a fixed-size farthest-point cover; below it,
/// every positive-support row is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceStabilityBand {
    BelowLandmarkCap,
    AtLandmarkCap,
}

impl PersistenceStabilityBand {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BelowLandmarkCap => "below_landmark_cap",
            Self::AtLandmarkCap => "at_landmark_cap",
        }
    }
}

/// The per-atom topology audit certificate: the measured persistence summary of
/// an atom's assigned-row image points, the topology the raced type predicts,
/// and the `contested` verdict when they disagree.
#[derive(Clone, Debug)]
pub struct AtomTopologyPersistence {
    /// The raced basis/topology tag whose prediction is under audit.
    pub raced_kind: SaeAtomBasisKind,
    /// Number of positive-support rows before persistence subsampling.
    pub support_size: usize,
    /// Number of image points actually persisted (post-subsample).
    pub landmark_count: usize,
    /// Which side of the landmark cap this atom's support lies on.
    pub stability_band: PersistenceStabilityBand,
    /// Whether the atom support is below or at/above the topology covering
    /// count used by the atlas-nerve honesty band.
    pub covering_side: AtlasCoveringSide,
    /// Soft occupancy mass `Σ_i w_i` from the shared atom support measure.
    pub support_mass: f64,
    /// Reconstruction-information effective count `Σ_i w_i²` from the shared
    /// atom support measure.
    pub effective_n: f64,
    /// Kish effective support `(Σ_i w_i)² / Σ_i w_i²`, the number of equally
    /// weighted rows represented by this atom's support distribution.
    pub support_ess: f64,
    /// Measured Betti signature.
    pub measured_betti: BettiSignature,
    /// Betti signature predicted by the raced topology.
    pub expected_betti: BettiSignature,
    /// Standing-null and spike-in calibration for the scalar topology claim.
    pub null_calibration: Option<ClaimNullCalibration>,
    /// Persistence of the most persistent measured loop (`+∞` for an essential
    /// loop, `0` when none).
    pub dominant_h1_persistence: f64,
    /// Persistence of the most persistent measured H₂ shell (`+∞` for an
    /// essential class, `0` when H₂ was not computed or none was measured).
    pub dominant_h2_persistence: f64,
    /// The measured H₀ bars (essential bar included).
    pub h0: Vec<PersistenceBar>,
    /// The measured H₁ bars.
    pub h1: Vec<PersistenceBar>,
    /// The measured H₂ bars. Empty unless the raced topology is sphere/torus.
    pub h2: Vec<PersistenceBar>,
    /// The certificate flag: the measured topology disagrees with the raced
    /// type's expected Betti signature. Fed to the probe planner.
    pub contested: bool,
    /// Human-readable summary.
    pub note: String,
}

/// Aggregate certificate adapter for the unified certificate ledger.
///
/// The full per-atom persistence records stay in the typed
/// `topology_persistence` payload. This adapter contributes the conservative
/// dictionary-level claim to the shared certificate ledger: every audited atom's
/// measured persistent topology must agree with its raced topology.
#[derive(Debug, Clone, Copy)]
pub struct TopologyPersistenceCertificate<'a> {
    pub atoms: &'a [Option<AtomTopologyPersistence>],
}

impl<'a> TopologyPersistenceCertificate<'a> {
    pub fn new(atoms: &'a [Option<AtomTopologyPersistence>]) -> Self {
        Self { atoms }
    }
}

/// Betti signature predicted by the raced topology. `finite_set_components`
/// supplies the anchor count for [`SaeAtomBasisKind::FiniteSet`], whose enum tag
/// is intentionally unit-shaped; caller-supplied precomputed bases carry no
/// library prediction to audit against.
fn expected_betti_signature(
    kind: &SaeAtomBasisKind,
    finite_set_components: Option<usize>,
) -> Option<BettiSignature> {
    match kind {
        SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Cylinder => Some(BettiSignature {
            b0: 1,
            b1: 1,
            b2: None,
        }),
        SaeAtomBasisKind::Torus => Some(BettiSignature {
            b0: 1,
            b1: 2,
            b2: Some(1),
        }),
        SaeAtomBasisKind::Sphere => Some(BettiSignature {
            b0: 1,
            b1: 0,
            b2: Some(1),
        }),
        SaeAtomBasisKind::Linear
        | SaeAtomBasisKind::Duchon
        | SaeAtomBasisKind::EuclideanPatch
        | SaeAtomBasisKind::Poincare => Some(BettiSignature {
            b0: 1,
            b1: 0,
            b2: None,
        }),
        SaeAtomBasisKind::FiniteSet => finite_set_components.map(|b0| BettiSignature {
            b0,
            b1: 0,
            b2: None,
        }),
        SaeAtomBasisKind::Precomputed(_) => None,
    }
}

fn point_distance(points: ArrayView2<'_, f64>, i: usize, j: usize) -> f64 {
    let mut acc = 0.0_f64;
    for col in 0..points.ncols() {
        let d = points[[i, col]] - points[[j, col]];
        acc += d * d;
    }
    acc.sqrt()
}

/// Deterministic farthest-point subsample to `target` landmarks (seeded at row
/// `0`), returning the chosen row indices. Returns all rows when `n <= target`.
fn farthest_point_subsample(points: ArrayView2<'_, f64>, target: usize) -> Vec<usize> {
    farthest_point_subsample_weighted(points, None, target)
}

fn farthest_point_subsample_weighted(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    target: usize,
) -> Vec<usize> {
    let n = points.nrows();
    if n <= target {
        return (0..n).collect();
    }
    let mut chosen = Vec::with_capacity(target);
    let mut first = 0usize;
    if let Some(w) = weights {
        let mut best_w = f64::NEG_INFINITY;
        for (row, &weight) in w.iter().enumerate() {
            if weight > best_w {
                best_w = weight;
                first = row;
            }
        }
    }
    chosen.push(first);
    let mean_weight = match weights {
        Some(w) => w.iter().copied().sum::<f64>() / w.len().max(1) as f64,
        None => 1.0,
    };
    let mut min_dist: Vec<f64> = (0..n).map(|i| point_distance(points, i, first)).collect();
    while chosen.len() < target {
        let mut best = 0usize;
        let mut best_score = -1.0_f64;
        for (i, &d) in min_dist.iter().enumerate() {
            let weight_factor = weights.map(|w| w[i] / mean_weight).unwrap_or(1.0);
            let score = d * weight_factor;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        chosen.push(best);
        for i in 0..n {
            let d = point_distance(points, i, best);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }
    chosen
}

/// Landmark weights carrying the FULL support measure after a farthest-point
/// subsample.
///
/// # Math correction (P2): distance-to-measure is a property of the WHOLE measure
///
/// The farthest-point subsample keeps only the `landmarks` rows. Taking the
/// landmark masses as `w[landmark]` alone DISCARDS the mass of every dropped row,
/// so the distance-to-measure weighting — which normalises by the total mass
/// (`target_mass = Σ w / m` inside [`dtm_radii`]) — is then evaluated on a
/// truncated measure and the DTM radii are biased low. The empirical
/// distance-to-measure is defined on the *entire* sample, not on the retained
/// landmarks. We therefore push the measure forward onto the cover: each row folds
/// its mass into its nearest retained landmark (a single-pass nearest-landmark
/// accumulation), so `Σ folded == Σ w` exactly and the DTM is computed on the full
/// measure. When nothing is discarded (`target ≥ n`), every row is its own nearest
/// landmark and this reduces to the identity `folded[k] = w[k]`.
fn fold_mass_to_landmarks(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    landmarks: &[usize],
) -> Array1<f64> {
    let mut folded = Array1::<f64>::zeros(landmarks.len());
    for r in 0..points.nrows() {
        let mut nearest = 0usize;
        let mut nearest_dist = f64::INFINITY;
        for (li, &l) in landmarks.iter().enumerate() {
            let d = point_distance(points, r, l);
            if d < nearest_dist {
                nearest_dist = d;
                nearest = li;
            }
        }
        let mass = match weights {
            Some(w) => w[r],
            None => 1.0,
        };
        if mass.is_finite() && mass > 0.0 {
            folded[nearest] += mass;
        }
    }
    folded
}

/// One simplex in the Vietoris–Rips filtration: its sorted vertex set, its
/// filtration value (max pairwise distance among its vertices), and dimension.
struct Simplex {
    verts: Vec<usize>,
    filt: f64,
    dim: usize,
}

/// Distance-to-measure radii for the selected support. The mass source is this
/// atom's assignment-mass vector today; this function is the single swap point
/// for a first-class support measure carrying density-corrected row masses.
fn dtm_radii(points: ArrayView2<'_, f64>, weights: Option<ArrayView1<'_, f64>>) -> Vec<f64> {
    let m = points.nrows();
    if m <= 1 {
        return vec![0.0; m];
    }
    let local_weights = weights
        .map(|w| w.to_owned())
        .unwrap_or_else(|| Array1::<f64>::ones(m));
    let total = local_weights.iter().copied().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return vec![0.0; m];
    }
    let target_mass = total / m as f64;
    let mut radii = vec![0.0_f64; m];
    for i in 0..m {
        let mut neighbors: Vec<(f64, f64)> = Vec::with_capacity(m);
        for j in 0..m {
            let weight = local_weights[j];
            if weight.is_finite() && weight > 0.0 {
                let distance = if i == j { 0.0 } else { point_distance(points, i, j) };
                neighbors.push((distance, weight));
            }
        }
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut mass = 0.0_f64;
        let mut moment = 0.0_f64;
        for (dist, weight) in neighbors {
            let take = (target_mass - mass).min(weight);
            if take > 0.0 {
                moment += take * dist * dist;
                mass += take;
            }
            if mass >= target_mass {
                break;
            }
        }
        if mass > 0.0 {
            radii[i] = (moment / mass).sqrt();
        }
    }
    radii
}

/// DTM-weighted pairwise distances together with the per-vertex DTM radii that
/// define them. The two outputs are the full data of the weighted Vietoris–Rips
/// filtration: for the standard `p = ∞` DTM convention a vertex is born at its
/// own DTM radius `w_i = dtm[i]` and an edge at `max(‖x_i − x_j‖, w_i, w_j)`.
fn dtm_weighted_distances_and_radii(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> (Array2<f64>, Vec<f64>) {
    let m = points.nrows();
    let dtm = dtm_radii(points, weights);
    let mut dist = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in (i + 1)..m {
            let d = point_distance(points, i, j).max(dtm[i]).max(dtm[j]);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    (dist, dtm)
}

fn dtm_weighted_distances(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Array2<f64> {
    dtm_weighted_distances_and_radii(points, weights).0
}

/// Exact DTM-weighted Vietoris–Rips persistent homology up to H₁ (needs
/// 2-simplices to kill loops). Filtration values are exact weighted pairwise
/// distances — no scale grid.
///
/// Reduction is the standard GF(2) boundary-matrix reduction: simplices are
/// ordered by `(filtration, dimension)`, each column is reduced against earlier
/// pivots, and a persistence pair `(birth-face, death-simplex)` is emitted when
/// a column's lowest surviving entry matches an existing pivot.
pub fn vietoris_rips_persistence(points: ArrayView2<'_, f64>) -> PersistenceDiagram {
    dtm_vietoris_rips_persistence(points, None, 1)
}

fn dtm_vietoris_rips_persistence(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    max_homology_dim: usize,
) -> PersistenceDiagram {
    let m = points.nrows();
    let mut h0 = Vec::new();
    let mut h1 = Vec::new();
    let mut h2 = Vec::new();
    if m == 0 {
        return PersistenceDiagram { h0, h1, h2 };
    }
    if m == 1 {
        // A single point has DTM radius 0 (`dtm_radii` returns 0 for `m <= 1`), so
        // its DTM-weighted vertex birth is 0 — historical behavior preserved.
        h0.push(PersistenceBar {
            birth: 0.0,
            death: f64::INFINITY,
        });
        return PersistenceDiagram { h0, h1, h2 };
    }

    let (dist, dtm) = dtm_weighted_distances_and_radii(points, weights);

    // Build simplices up to the coface dimension needed by the requested
    // homology: H₁ needs triangles, H₂ needs tetrahedra.
    let max_simplex_dim = (max_homology_dim + 1).min(3);
    let mut simplices: Vec<Simplex> = Vec::new();
    // Standard `p = ∞` DTM-weighted Vietoris–Rips convention: a vertex is born at
    // its own DTM radius `w_i = dtm[i]`, NOT at 0. Edges/higher simplices already
    // carry `max(d_ij, w_i, w_j)` (see `dtm_weighted_distances_and_radii`), which
    // is `≥` each face's DTM birth, so face-before-coface ordering is preserved.
    for i in 0..m {
        simplices.push(Simplex {
            verts: vec![i],
            filt: dtm[i],
            dim: 0,
        });
    }
    for i in 0..m {
        for j in (i + 1)..m {
            simplices.push(Simplex {
                verts: vec![i, j],
                filt: dist[[i, j]],
                dim: 1,
            });
        }
    }
    if max_simplex_dim >= 2 {
        for i in 0..m {
            for j in (i + 1)..m {
                for k in (j + 1)..m {
                    let filt = dist[[i, j]].max(dist[[i, k]]).max(dist[[j, k]]);
                    simplices.push(Simplex {
                        verts: vec![i, j, k],
                        filt,
                        dim: 2,
                    });
                }
            }
        }
    }
    if max_simplex_dim >= 3 {
        for i in 0..m {
            for j in (i + 1)..m {
                for k in (j + 1)..m {
                    for l in (k + 1)..m {
                        let filt = dist[[i, j]]
                            .max(dist[[i, k]])
                            .max(dist[[i, l]])
                            .max(dist[[j, k]])
                            .max(dist[[j, l]])
                            .max(dist[[k, l]]);
                        simplices.push(Simplex {
                            verts: vec![i, j, k, l],
                            filt,
                            dim: 3,
                        });
                    }
                }
            }
        }
    }

    // Filtration order: ascending filtration, then ascending dimension (a face
    // must precede its coface), then lexicographic vertices for a total order.
    let mut order: Vec<usize> = (0..simplices.len()).collect();
    order.sort_by(|&a, &b| {
        let sa = &simplices[a];
        let sb = &simplices[b];
        sa.filt
            .partial_cmp(&sb.filt)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(sa.dim.cmp(&sb.dim))
            .then(sa.verts.cmp(&sb.verts))
    });
    // Global filtration index of each simplex, and a vertex-set -> index map.
    let mut filt_index = vec![0usize; simplices.len()];
    let mut key_to_index: HashMap<Vec<usize>, usize> = HashMap::with_capacity(simplices.len());
    for (fi, &orig) in order.iter().enumerate() {
        filt_index[orig] = fi;
        key_to_index.insert(simplices[orig].verts.clone(), fi);
    }

    // Ordered simplices (indexed by filtration position) with their boundaries
    // (as filtration indices of their codim-1 faces).
    let mut ordered_filt = vec![0.0_f64; simplices.len()];
    let mut ordered_dim = vec![0usize; simplices.len()];
    let mut boundary: Vec<Vec<usize>> = vec![Vec::new(); simplices.len()];
    for &orig in &order {
        let s = &simplices[orig];
        let fi = filt_index[orig];
        ordered_filt[fi] = s.filt;
        ordered_dim[fi] = s.dim;
        if s.dim == 0 {
            continue;
        }
        let mut faces = Vec::with_capacity(s.verts.len());
        for drop in 0..s.verts.len() {
            let mut face = Vec::with_capacity(s.verts.len() - 1);
            for (idx, &v) in s.verts.iter().enumerate() {
                if idx != drop {
                    face.push(v);
                }
            }
            if let Some(&face_fi) = key_to_index.get(&face) {
                faces.push(face_fi);
            }
        }
        faces.sort_unstable();
        boundary[fi] = faces;
    }

    // GF(2) reduction. `reduced[j]` holds the reduced column (sorted). `pivot`
    // maps a low-index to the column that owns it. `paired_birth` marks faces
    // that have been consumed as a birth (so leftover empty columns are the
    // essential classes).
    let n = simplices.len();
    let mut reduced: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pivot: HashMap<usize, usize> = HashMap::new();
    let mut paired_birth = vec![false; n];

    for j in 0..n {
        let mut col = boundary[j].clone();
        while let Some(&low) = col.last() {
            if let Some(&owner) = pivot.get(&low) {
                col = symmetric_difference(&col, &reduced[owner]);
            } else {
                break;
            }
        }
        if let Some(&low) = col.last() {
            pivot.insert(low, j);
            reduced[j] = col;
            paired_birth[low] = true;
            // Persistence pair: face `low` born, simplex `j` kills it.
            let birth = ordered_filt[low];
            let death = ordered_filt[j];
            let bar = PersistenceBar { birth, death };
            match ordered_dim[low] {
                0 => {
                    if death > birth {
                        h0.push(bar);
                    }
                }
                1 => {
                    if death > birth {
                        h1.push(bar);
                    }
                }
                2 => {
                    if max_homology_dim >= 2 && death > birth {
                        h2.push(bar);
                    }
                }
                _ => {}
            }
        }
    }

    // Essential classes: fully reduced zero columns that were never consumed as
    // a birth.
    for j in 0..n {
        if reduced[j].is_empty() && !paired_birth[j] {
            let bar = PersistenceBar {
                birth: ordered_filt[j],
                death: f64::INFINITY,
            };
            match ordered_dim[j] {
                0 => h0.push(bar),
                1 => h1.push(bar),
                2 if max_homology_dim >= 2 => h2.push(bar),
                _ => {}
            }
        }
    }

    PersistenceDiagram { h0, h1, h2 }
}

fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ia = 0;
    let mut ib = 0;
    while ia < a.len() && ib < b.len() {
        match a[ia].cmp(&b[ib]) {
            std::cmp::Ordering::Less => {
                out.push(a[ia]);
                ia += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[ib]);
                ib += 1;
            }
            std::cmp::Ordering::Equal => {
                ia += 1;
                ib += 1;
            }
        }
    }
    out.extend_from_slice(&a[ia..]);
    out.extend_from_slice(&b[ib..]);
    out
}

/// Number of connected components read off the finite H₀ merge bars by the
/// dominant-gap test, plus the within-component spacing scale used to qualify a
/// loop as persistent.
///
/// The finite H₀ deaths are the merge scales. On a connected manifold they are
/// all the *local* point spacing (a narrow band); a `c`-cluster set instead has
/// `c − 1` merges at the *inter-cluster* gap — orders of magnitude larger than
/// the within-cluster spacing. So there is a single dominant multiplicative gap
/// in the sorted deaths exactly when the cloud is genuinely clustered. The cut
/// is accepted only when that one log-gap outweighs *all* the others combined
/// AND clears a floor of `ln 2`.
///
/// The `ln 2` floor is *derived*, not a magic constant. The deaths are computed
/// on a farthest-point subsample (see [`PERSISTENCE_MAX_POINTS`]); farthest-point
/// sampling of a connected d-manifold is a covering/packing construction in which
/// each new landmark sits at the current covering radius, so consecutive landmark
/// (merge) scales are guaranteed to stay within a factor of 2 of one another.
/// Hence *within* a single connected manifold every consecutive log-gap is
/// `≤ ln 2` by construction. Without this floor the sum-only rule is a tie
/// degeneracy: on a clean, near-noiseless atom the FPS spacings are nearly
/// uniform, all gaps are tiny and nearly equal, and the largest of them trivially
/// exceeds the (equally tiny) sum of the rest — so a genuinely connected circle is
/// declared multi-component. That false positive is *worst* on the atoms that
/// reconstruct cleanest (lowest residual noise breaks the ties least). A real
/// cluster separation produces a log-gap `≫ ln 2`, so the floor removes the
/// degeneracy while leaving genuine splits untouched.
///
/// A declared split is additionally rejected when it would isolate a lone
/// outlier — a component holding fewer than 2 landmarks. That is a single stray
/// row (a chart artifact), not a second sampled manifold component. The clusters
/// the dominant cut separates are exactly the single-linkage components just
/// below the smallest inter-cluster merge scale `deaths[gmax_idx]` (VR H₀ deaths
/// ARE the single-linkage merge heights), so the guard reuses the same
/// DTM-weighted landmark distances as the filtration.
///
/// Returns `(n_components, within_component_scale)`.
fn components_and_scale(finite_h0: &[PersistenceBar], distances: &Array2<f64>) -> (usize, f64) {
    let mut deaths: Vec<f64> = finite_h0
        .iter()
        .map(|b| b.death)
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    if deaths.is_empty() {
        return (1, 0.0);
    }
    deaths.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let l = deaths.len();
    if l == 1 {
        // A single merge cannot distinguish a two-cluster split from ordinary
        // within-manifold sampling; treat as connected.
        return (1, deaths[0]);
    }
    let logs: Vec<f64> = deaths.iter().map(|d| d.ln()).collect();
    let gaps: Vec<f64> = (0..l - 1).map(|i| logs[i] - logs[i + 1]).collect();
    let mut gmax = f64::NEG_INFINITY;
    let mut gmax_idx = 0usize;
    for (i, &g) in gaps.iter().enumerate() {
        if g > gmax {
            gmax = g;
            gmax_idx = i;
        }
    }
    let sum_others: f64 = gaps.iter().sum::<f64>() - gmax;
    // Split only when the dominant gap outweighs every other gap combined AND
    // clears the FPS covering-bound floor `ln 2` (consecutive farthest-point
    // landmark scales stay within a factor of 2 within a connected manifold, so
    // within-manifold log-gaps are `≤ ln 2` by construction). The floor kills the
    // near-uniform-spacing tie degeneracy that otherwise fires on clean atoms.
    let split_floor = sum_others.max(std::f64::consts::LN_2);
    if gmax > split_floor && smallest_linkage_component(distances, deaths[gmax_idx]) >= 2 {
        // Bars 0..=gmax_idx are inter-cluster merges: gmax_idx + 1 of them, so
        // gmax_idx + 2 components (the extra essential one). The within-cluster
        // scale is the largest death below the cut. The linkage guard above has
        // confirmed no side of the cut is a lone outlier.
        let within = deaths[gmax_idx + 1];
        (gmax_idx + 2, within)
    } else {
        // Connected (or a rejected lone-outlier cut): the coarsest merge is the
        // within-component spacing.
        (1, deaths[0])
    }
}

/// Size (in landmarks) of the smallest single-linkage component at `scale`:
/// landmark pairs closer than `scale` are unioned, and the least-populated
/// resulting component is returned. Used to reject a dominant-gap split that
/// would carve off a lone outlier (a component of one landmark).
fn smallest_linkage_component(distances: &Array2<f64>, scale: f64) -> usize {
    let m = distances.nrows();
    if m == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..m).collect();
    for i in 0..m {
        for j in (i + 1)..m {
            if distances[[i, j]] < scale {
                let ri = nerve_find(&mut parent, i);
                let rj = nerve_find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for x in 0..m {
        let r = nerve_find(&mut parent, x);
        *sizes.entry(r).or_insert(0) += 1;
    }
    sizes.values().copied().min().unwrap_or(0)
}

fn dominant_persistence(bars: &[PersistenceBar]) -> f64 {
    bars.iter().map(|b| b.persistence()).fold(0.0_f64, f64::max)
}

/// Count the significant H₁ generators as the number of loops alive on the
/// **dominant persistence plateau**, on a cover dense enough to resolve them.
///
/// # Math correction (#2159): read the homology RANK, never merge by birth value
///
/// The first Betti number of the Vietoris–Rips complex at scale `ε` is *exactly*
/// `β₁(ε) = #{ bars with birth < ε < death }`. The GF(2) boundary reduction emits
/// one persistence bar per independent 1-cycle, and distinct surviving bars are, by
/// construction, independent homology classes: the reduction never conflates two
/// classes into one bar, nor splits one class across two bars. So the honest
/// generator count at a scale is simply the number of bars alive there.
///
/// The previous rule instead grouped the alive bars by *birth-value proximity* (a
/// `~1e-6` band) and counted one generator per birth-cluster. That is wrong on a
/// **symmetric grid**: a square Clifford torus (`nu == nv`) is *born-degenerate* —
/// both of its genuinely-independent generators appear at the *same* filtration
/// value by symmetry — so birth-proximity merged the two into one and reported
/// `b₁ = 1` with a spurious `contested` flag (#2159). Birth coincidence cannot tell
/// "two independent generators born together by symmetry" apart from "one homologous
/// family": both share filtration values to numerical precision. The reduction had
/// already answered the question (two independent bars ⇒ rank 2); the downstream
/// birth-merge discarded that answer. Distinct generators must be separated by the
/// classes themselves — which the reduction does — not by birth-value nearness.
///
/// The resolution carries no magic constant and no dedup. (1) A class is real iff
/// its lifetime exceeds the **local sampling resolution** — the *median
/// nearest-neighbour distance* of the landmark cover (a measured covering scale, the
/// same family as the `ln 2` H₀ floor); within-manifold noise is born and filled
/// within one point-spacing, whereas a genuine cycle survives from the spacing scale
/// up to its hole size. (2) Among the survivors, read the rank `β₁` on the **widest
/// log-scale plateau** of the Betti curve — the topology that persists over the
/// largest range of scales — exactly as [`shell_plateau_bar_count`] reads the H₂
/// shell count. A structured grid's spurious per-cell loops are born at the edge
/// scale and filled at the (nearby) diagonal scale, so they occupy only a *narrow*
/// plateau and never dominate; the two genuine torus generators span from the
/// spacing scale to the hole scale and own the widest plateau, so both survive the
/// symmetry degeneracy and are counted (→ 2). The sphere's transient loops die
/// within the spacing (→ 0) and a circle keeps its single generator (→ 1).
fn spacing_floor_bar_count(bars: &[PersistenceBar], distances: &Array2<f64>) -> usize {
    let essential = bars.iter().filter(|b| b.is_essential()).count();
    // Covering-scale floor: the median nearest-neighbour distance of the landmark
    // cover. A cycle that persists past this outlives the sampling resolution and
    // is a genuine generator; within-manifold noise dies within one spacing.
    let m = distances.nrows();
    let mut nn: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        let mut best = f64::INFINITY;
        for j in 0..m {
            if i != j {
                let d = distances[[i, j]];
                if d > 0.0 && d < best {
                    best = d;
                }
            }
        }
        if best.is_finite() {
            nn.push(best);
        }
    }
    if nn.is_empty() {
        return essential;
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let spacing = nn[nn.len() / 2];
    let finite: Vec<PersistenceBar> = bars
        .iter()
        .copied()
        .filter(|b| {
            b.birth.is_finite()
                && b.death.is_finite()
                && b.death > b.birth
                && b.persistence() > spacing
        })
        .collect();
    if finite.is_empty() {
        return essential;
    }

    // Read β₁ on the widest log-scale plateau. The count of bars alive at a scale
    // IS the homology rank there (each bar is an independent GF(2) class), so there
    // is no birth-proximity merge — that merge collapsed the symmetry-degenerate
    // torus generators (#2159). Ties in width break to the smaller (coarser) count,
    // matching `shell_plateau_bar_count`.
    let mut critical = Vec::with_capacity(finite.len() * 2);
    for bar in &finite {
        critical.push(bar.birth);
        critical.push(bar.death);
    }
    critical.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    critical.dedup_by(|a, b| *a == *b);
    let mut best = 0usize;
    let mut best_width = f64::NEG_INFINITY;
    for window in critical.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if !(hi > lo) {
            continue;
        }
        let probe = if lo > 0.0 { (lo * hi).sqrt() } else { hi / 2.0 };
        let count = finite
            .iter()
            .filter(|bar| bar.birth < probe && probe < bar.death)
            .count();
        if count == 0 {
            continue;
        }
        let width = if lo > 0.0 { (hi / lo).ln() } else { hi - lo };
        if width > best_width || (width == best_width && count < best) {
            best_width = width;
            best = count;
        }
    }
    essential + best
}

fn shell_plateau_bar_count(bars: &[PersistenceBar]) -> usize {
    let essential = bars.iter().filter(|b| b.is_essential()).count();
    let finite: Vec<PersistenceBar> = bars
        .iter()
        .copied()
        .filter(|b| b.birth.is_finite() && b.death.is_finite() && b.death > b.birth)
        .collect();
    if finite.is_empty() {
        return essential;
    }
    let mut critical = Vec::with_capacity(finite.len() * 2);
    for bar in &finite {
        critical.push(bar.birth);
        critical.push(bar.death);
    }
    critical.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    critical.dedup_by(|a, b| *a == *b);
    let mut best = essential;
    let mut best_width = f64::NEG_INFINITY;
    for window in critical.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if !(hi > lo) {
            continue;
        }
        let probe = if lo > 0.0 { (lo * hi).sqrt() } else { hi / 2.0 };
        let count = essential
            + finite
                .iter()
                .filter(|bar| bar.birth < probe && probe < bar.death)
                .count();
        if count == 0 {
            continue;
        }
        let width = if lo > 0.0 { (hi / lo).ln() } else { hi - lo };
        if width > best_width || (width == best_width && count < best) {
            best_width = width;
            best = count;
        }
    }
    best
}

fn support_summary(weights: Option<ArrayView1<'_, f64>>, full: usize) -> (f64, f64, f64) {
    match weights {
        Some(w) => {
            let mut mass = 0.0_f64;
            let mut fisher_n = 0.0_f64;
            for &weight in w.iter() {
                if weight.is_finite() && weight > 0.0 {
                    mass += weight;
                    fisher_n += weight * weight;
                }
            }
            let ess = if fisher_n > 0.0 {
                (mass * mass) / fisher_n
            } else {
                0.0
            };
            (mass, fisher_n, ess)
        }
        None => {
            let n = full as f64;
            (n, n, n)
        }
    }
}

/// Audit a raced atom's topology against the persistent homology of a point
/// cloud (its assigned-row image points). Returns `None` when the raced type
/// carries no library prediction ([`SaeAtomBasisKind::Precomputed`]) or the
/// cloud is too small to resolve H₁ (fewer than four points — a triangle plus
/// one is the minimum that can kill a loop).
pub fn topology_persistence_verdict(
    points: ArrayView2<'_, f64>,
    raced_kind: &SaeAtomBasisKind,
) -> Option<AtomTopologyPersistence> {
    topology_persistence_verdict_impl(points, None, raced_kind, None)
}

fn topology_persistence_verdict_impl(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    raced_kind: &SaeAtomBasisKind,
    finite_set_components: Option<usize>,
) -> Option<AtomTopologyPersistence> {
    let expected_betti = expected_betti_signature(raced_kind, finite_set_components)?;
    let full = points.nrows();
    if full < 4 {
        return None;
    }
    // H₀/H₁ (components and loops) read a LARGER farthest-point cover than the H₂
    // shell audit: loop counting is only cubic in the landmark count, so it can
    // afford the manifold's covering number of samples, whereas the coarse
    // PERSISTENCE_MAX_POINTS cover (sized for the quartic H₂ tetrahedra) fails to
    // resolve a torus's second generator (#2159).
    let h1_landmarks = farthest_point_subsample_weighted(points, weights, PERSISTENCE_H1_MAX_POINTS);
    let h1_sub = points.select(ndarray::Axis(0), &h1_landmarks);
    // P2: fold every discarded row's mass into its nearest retained landmark so the
    // DTM weighting sees the FULL support measure, not just the landmark rows' mass
    // (see `fold_mass_to_landmarks`). A no-op when nothing was subsampled.
    let h1_weights = fold_mass_to_landmarks(points, weights, &h1_landmarks);
    let h1_diagram =
        dtm_vietoris_rips_persistence(h1_sub.view(), Some(h1_weights.view()), 1);
    let h1_distances = dtm_weighted_distances(h1_sub.view(), Some(h1_weights.view()));

    // H₂ shells (sphere/torus voids) need the quartic tetrahedron enumeration, so
    // they stay on the compute-bounded PERSISTENCE_MAX_POINTS cover.
    let h2: Vec<PersistenceBar> = if expected_betti.b2.is_some() {
        let landmarks = farthest_point_subsample_weighted(points, weights, PERSISTENCE_MAX_POINTS);
        let sub = points.select(ndarray::Axis(0), &landmarks);
        // P2: fold discarded mass into the nearest landmark. The H₂ cover is the
        // small PERSISTENCE_MAX_POINTS budget, so it almost always subsamples and
        // the truncated-measure bias would otherwise be largest here.
        let sub_weights = fold_mass_to_landmarks(points, weights, &landmarks);
        dtm_vietoris_rips_persistence(sub.view(), Some(sub_weights.view()), 2).h2
    } else {
        Vec::new()
    };

    let (support_mass, effective_n, support_ess) = support_summary(weights, full);

    let finite_h0: Vec<PersistenceBar> = h1_diagram
        .h0
        .iter()
        .copied()
        .filter(|b| !b.is_essential())
        .collect();
    let (n_components, _) = components_and_scale(&finite_h0, &h1_distances);

    let measured_betti = BettiSignature {
        b0: n_components,
        b1: spacing_floor_bar_count(&h1_diagram.h1, &h1_distances),
        b2: expected_betti.b2.map(|expected_h2| {
            let counted = shell_plateau_bar_count(&h2);
            if expected_h2 == 0 && counted == 0 {
                0
            } else {
                counted
            }
        }),
    };
    let dominant_h1_persistence = dominant_persistence(&h1_diagram.h1);
    let dominant_h2_persistence = dominant_persistence(&h2);
    let contested = !measured_betti.matches_expected(expected_betti);

    let note = if contested {
        let mut reasons = Vec::new();
        if measured_betti.b0 != expected_betti.b0 {
            reasons.push(format!(
                "measured b0={} but raced type predicts b0={}",
                measured_betti.b0, expected_betti.b0
            ));
        }
        if measured_betti.b1 != expected_betti.b1 {
            reasons.push(format!(
                "measured b1={} but raced type predicts b1={}",
                measured_betti.b1, expected_betti.b1
            ));
        }
        if let Some(expected_h2) = expected_betti.b2 {
            if measured_betti.b2 != Some(expected_h2) {
                reasons.push(format!(
                    "measured b2={} but raced type predicts b2={expected_h2}",
                    measured_betti.b2.unwrap_or(0)
                ));
            }
        }
        format!("CONTESTED topology: {}", reasons.join("; "))
    } else {
        format!(
            "topology agrees: measured Betti {:?} matches raced Betti {:?}",
            measured_betti, expected_betti
        )
    };
    let stability_band = if full > PERSISTENCE_MAX_POINTS {
        PersistenceStabilityBand::AtLandmarkCap
    } else {
        PersistenceStabilityBand::BelowLandmarkCap
    };
    let covering_side = if full >= PERSISTENCE_MAX_POINTS {
        AtlasCoveringSide::AtOrAboveCoveringNumber
    } else {
        AtlasCoveringSide::BelowCoveringNumber
    };

    Some(AtomTopologyPersistence {
        raced_kind: raced_kind.clone(),
        support_size: full,
        landmark_count: h1_landmarks.len(),
        stability_band,
        covering_side,
        support_mass,
        effective_n,
        support_ess,
        measured_betti,
        expected_betti,
        null_calibration: None,
        dominant_h1_persistence,
        dominant_h2_persistence,
        h0: h1_diagram.h0,
        h1: h1_diagram.h1,
        h2,
        contested,
        note,
    })
}

/// Gather one atom's positive-support image points and audit its raced topology.
///
/// The row weights come from the shared [`SupportMeasure`] (`w_i = a_ik`), so
/// the persistence audit reads the same atom support as coordinate fidelity,
/// trust diagnostics, and rank charge. Each supported row's in-atom
/// residual-space position is the decoded image `g_k(t_{ik}) = Φ_k(t_{ik}) B_k`.
/// Returns `None` when the atom's topology is caller-supplied or too few
/// positive-support rows are present to resolve H₁.
pub fn atom_topology_persistence(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Option<AtomTopologyPersistence> {
    let atom = term.atoms.get(atom_idx)?;
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k = assignments.ncols();
    if n == 0 || atom_idx >= k {
        return None;
    }
    let mut supported_rows = Vec::new();
    let mut support_weights = Vec::new();
    for row in 0..n {
        let mass = assignments[[row, atom_idx]];
        if mass.is_finite() && mass > 0.0 {
            supported_rows.push(row);
            support_weights.push(mass);
        }
    }
    if supported_rows.len() < 4 || supported_rows.iter().any(|&r| r >= atom.n_obs()) {
        return None;
    }
    let p = atom.output_dim();
    let mut points = Array2::<f64>::zeros((supported_rows.len(), p));
    let mut weights = Array1::<f64>::zeros(supported_rows.len());
    for (i, &row) in supported_rows.iter().enumerate() {
        let image = atom.decoded_row(row);
        for col in 0..p {
            points[[i, col]] = image[col];
        }
        weights[i] = support_weights[i];
    }
    let finite_set_components = if matches!(atom.basis_kind, SaeAtomBasisKind::FiniteSet) {
        Some(atom.basis_size())
    } else {
        None
    };
    topology_persistence_verdict_impl(
        points.view(),
        Some(weights.view()),
        &atom.basis_kind,
        finite_set_components,
    )
}

/// The topology read off a typed-free local-chart atlas by its NERVE (reviewer
/// F3, atlas-first inversion).
///
/// The default pipeline *assumes-then-races* — it picks a topology up front and
/// fits it. The inversion *measures-then-imposes*: cover the atom's points with
/// overlapping local charts (Duchon patches glued by the in-tree
/// [`crate::chart_transfer`] pulled-back operators in production; here the
/// overlap is read geometrically from the point cloud), then read the topology
/// from the nerve of the cover — the graph whose vertices are charts and whose
/// edges join charts that overlap. A cyclic nerve is a circle; a path nerve is
/// an arc. The nerve's first Betti number `b₁ = E − V + C` is the loop count,
/// so `b₁ = 1, C = 1` recovers `S¹` and `b₁ = 0, C = 1` an arc — a topology that
/// was *measured*, not latched, and can then be imposed as the typed refit.
#[derive(Clone, Debug)]
pub struct AtlasNerveReport {
    /// Number of local charts (landmark cover elements).
    pub n_charts: usize,
    /// Number of nerve edges (overlapping chart pairs).
    pub n_edges: usize,
    /// Connected components of the nerve.
    pub n_components: usize,
    /// First Betti number `b₁ = E − V + C` (the graph cycle rank): the number
    /// of independent loops in the recovered manifold.
    pub b1: i64,
    /// Whether the sampled support is below or at/above the atlas covering
    /// count. Below-covering reports are measurements, but marked under-resolved.
    pub covering_side: AtlasCoveringSide,
}

impl AtlasNerveReport {
    /// Whether the nerve recovers a single circle `S¹` (one component, one loop).
    pub fn is_circle(&self) -> bool {
        self.n_components == 1 && self.b1 == 1
    }
    /// Whether the nerve recovers a single arc / path (one component, no loop).
    pub fn is_arc(&self) -> bool {
        self.n_components == 1 && self.b1 == 0
    }
}

fn nerve_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Build the nerve of a landmark-chart atlas over a point cloud and read its
/// topology. The chart count is data-derived (`⌈√n⌉` landmarks, floored at 3 —
/// the nerve is invariant to it above the covering number); charts are
/// farthest-point landmarks. The nerve edges are the **witness-complex**
/// 1-skeleton: each point witnesses an edge between its two nearest landmark
/// charts (the two charts whose regions overlap where it sits). This is exactly
/// the Voronoi adjacency of the atlas — adjacent charts only, no radius, no
/// magic constant.
pub fn atlas_nerve(points: ArrayView2<'_, f64>) -> AtlasNerveReport {
    let n = points.nrows();
    if n == 0 {
        return AtlasNerveReport {
            n_charts: 0,
            n_edges: 0,
            n_components: 0,
            b1: 0,
            covering_side: AtlasCoveringSide::BelowCoveringNumber,
        };
    }
    let n_charts = ((n as f64).sqrt().ceil() as usize).max(3).min(n);
    let landmarks = farthest_point_subsample(points, n_charts);
    let v = landmarks.len();
    let mut adj = vec![vec![false; v]; v];
    for i in 0..n {
        let mut best = (f64::INFINITY, 0usize);
        let mut second = (f64::INFINITY, 0usize);
        for (ci, &l) in landmarks.iter().enumerate() {
            let d = point_distance(points, i, l);
            if d < best.0 {
                second = best;
                best = (d, ci);
            } else if d < second.0 {
                second = (d, ci);
            }
        }
        if second.0.is_finite() && best.1 != second.1 {
            adj[best.1][second.1] = true;
            adj[second.1][best.1] = true;
        }
    }
    let mut n_edges = 0usize;
    let mut parent: Vec<usize> = (0..v).collect();
    for a in 0..v {
        for b in (a + 1)..v {
            if adj[a][b] {
                n_edges += 1;
                let ra = nerve_find(&mut parent, a);
                let rb = nerve_find(&mut parent, b);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut roots = std::collections::HashSet::new();
    for x in 0..v {
        let r = nerve_find(&mut parent, x);
        roots.insert(r);
    }
    let n_components = roots.len();
    let b1 = n_edges as i64 - v as i64 + n_components as i64;
    let covering_side = if n >= v {
        AtlasCoveringSide::AtOrAboveCoveringNumber
    } else {
        AtlasCoveringSide::BelowCoveringNumber
    };
    AtlasNerveReport {
        n_charts: v,
        n_edges,
        n_components,
        b1,
        covering_side,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// A Clifford torus: the product of two unit circles in ℝ⁴, sampled on a
    /// `nu × nv` grid. When `nu == nv` the grid is symmetric and both H₁
    /// generators are born at the *same* filtration value — the #2159 degeneracy.
    fn clifford_torus(nu: usize, nv: usize) -> Array2<f64> {
        let mut pts = Array2::<f64>::zeros((nu * nv, 4));
        let mut row = 0usize;
        for i in 0..nu {
            let u = std::f64::consts::TAU * (i as f64) / (nu as f64);
            for j in 0..nv {
                let v = std::f64::consts::TAU * (j as f64) / (nv as f64);
                pts[[row, 0]] = u.cos();
                pts[[row, 1]] = u.sin();
                pts[[row, 2]] = v.cos();
                pts[[row, 3]] = v.sin();
                row += 1;
            }
        }
        pts
    }

    /// A circle of `n` equally spaced points at radius `r` in ℝ².
    fn circle(n: usize, r: f64) -> Array2<f64> {
        let mut pts = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
            pts[[i, 0]] = r * theta.cos();
            pts[[i, 1]] = r * theta.sin();
        }
        pts
    }

    /// #2159 regression pin. A *symmetric* Clifford grid (`nu == nv`) is
    /// born-degenerate: both genuinely-independent H₁ generators appear at the
    /// same filtration value by symmetry. The old birth-proximity dedup merged
    /// them and reported `b₁ = 1` with a spurious `contested` flag; the
    /// dominant-plateau homology-rank reading must recover `b₁ = 2` cleanly.
    #[test]
    fn symmetric_torus_measures_two_generators_uncontested() {
        let pts = clifford_torus(8, 8);
        let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Torus)
            .expect("torus verdict");
        assert_eq!(verdict.measured_betti.b0, 1, "a torus is connected");
        assert_eq!(
            verdict.measured_betti.b1, 2,
            "a torus has TWO independent loops; symmetric grids must not merge them (#2159)"
        );
        assert_eq!(
            verdict.measured_betti.b2,
            Some(1),
            "a torus bounds exactly one void"
        );
        assert!(
            !verdict.contested,
            "measured (1,2,1) must agree with the raced Torus prediction: {}",
            verdict.note
        );
    }

    /// An *asymmetric* torus (`nu != nv`): the two generators are born at
    /// different scales. It must also measure `b₁ = 2` — the plateau reading is
    /// not special-cased to the degenerate case.
    #[test]
    fn asymmetric_torus_still_measures_two_generators() {
        let pts = clifford_torus(10, 8);
        let landmarks =
            farthest_point_subsample_weighted(pts.view(), None, PERSISTENCE_H1_MAX_POINTS);
        let sub = pts.select(ndarray::Axis(0), &landmarks);
        let weights = fold_mass_to_landmarks(pts.view(), None, &landmarks);
        let diagram = dtm_vietoris_rips_persistence(sub.view(), Some(weights.view()), 1);
        let distances = dtm_weighted_distances(sub.view(), Some(weights.view()));
        assert_eq!(
            spacing_floor_bar_count(&diagram.h1, &distances),
            2,
            "an asymmetric torus still has two independent loops"
        );
    }

    /// A circle keeps its single generator: the dedup removal must not
    /// over-split a shape with genuine multiplicity one.
    #[test]
    fn circle_still_measures_one_generator() {
        let pts = circle(40, 2.0);
        let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic)
            .expect("circle verdict");
        assert_eq!(
            verdict.measured_betti.b1, 1,
            "a circle has exactly one loop; the plateau reading must not over-split"
        );
        assert!(
            !verdict.contested,
            "measured circle topology must agree with the raced Periodic prediction: {}",
            verdict.note
        );
    }

    /// P2 regression pin. A cloud larger than the H₁ landmark cap forces a real
    /// farthest-point subsample; the folded landmark weights must still sum to the
    /// FULL support mass, i.e. no discarded row's mass is dropped from the DTM.
    #[test]
    fn folded_landmark_mass_preserves_full_measure() {
        let n = 400usize;
        let pts = circle(n, 1.0);
        let mut weights = Array1::<f64>::zeros(n);
        for i in 0..n {
            weights[i] = 0.5 + (i as f64) * 0.01;
        }
        let total: f64 = weights.iter().sum();
        let landmarks = farthest_point_subsample_weighted(
            pts.view(),
            Some(weights.view()),
            PERSISTENCE_H1_MAX_POINTS,
        );
        assert!(
            landmarks.len() < n,
            "the cap must actually drop rows for this test to exercise the fold"
        );
        let folded = fold_mass_to_landmarks(pts.view(), Some(weights.view()), &landmarks);
        let folded_total: f64 = folded.iter().sum();
        assert!(
            (folded_total - total).abs() <= 1e-9 * total,
            "folded landmark mass {folded_total} must equal the full support mass {total}"
        );
    }

    fn bars(deaths: &[f64]) -> Vec<PersistenceBar> {
        deaths
            .iter()
            .map(|&d| PersistenceBar {
                birth: 0.0,
                death: d,
            })
            .collect()
    }

    /// A 1-D landmark cloud from the given coordinates (one point per row). The
    /// dominant-gap logic reads the gaps off `bars`; the lone-outlier guard reads
    /// single-linkage component sizes off these coordinates, so the two are set
    /// independently in each test to exercise the guard in isolation.
    fn line_points(xs: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).unwrap()
    }

    /// The sum-only split rule (`g_max > Σ other gaps`) as it stood before the
    /// `ln 2` floor — used to demonstrate that the false positive it produced is
    /// now suppressed.
    fn old_rule_splits(deaths: &[f64]) -> bool {
        let mut d: Vec<f64> = deaths.to_vec();
        d.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if d.len() < 2 {
            return false;
        }
        let logs: Vec<f64> = d.iter().map(|x| x.ln()).collect();
        let gaps: Vec<f64> = (0..d.len() - 1).map(|i| logs[i] - logs[i + 1]).collect();
        let gmax = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_others: f64 = gaps.iter().sum::<f64>() - gmax;
        gmax > sum_others && gmax > 0.0
    }

    #[test]
    fn near_uniform_clean_spacing_is_connected() {
        // A clean, near-noiseless atom: farthest-point spacings are nearly tied.
        // With only two merges the "sum of other gaps" is zero, so the sum-only
        // rule ALWAYS declared a split; with three nearly-equal gaps whichever is
        // marginally largest still trips it. Both are genuine single manifolds.
        for deaths in [vec![1.0, 0.98], vec![1.0, 0.98, 0.96]] {
            assert!(
                old_rule_splits(&deaths),
                "sum-only rule should have (wrongly) split {deaths:?}"
            );
            // The floor rejects the split before the guard is consulted; points
            // are a well-separated cloud so they cannot be what keeps it connected.
            let pts = line_points(
                &(0..deaths.len() + 1)
                    .map(|i| i as f64 * 100.0)
                    .collect::<Vec<_>>(),
            );
            let distances = dtm_weighted_distances(pts.view(), None);
            let (n, _) = components_and_scale(&bars(&deaths), &distances);
            assert_eq!(n, 1, "ln2 floor should keep {deaths:?} connected");
        }
    }

    #[test]
    fn genuine_two_cluster_split_survives_floor() {
        // One inter-cluster merge at scale 10 with within-cluster spacing near
        // 0.1: the dominant log-gap is ≫ ln 2. Landmarks form two groups (3 + 2)
        // that stay separate below the cut scale 10, so both sides clear the
        // ≥2-landmark guard and the split is preserved.
        let deaths = vec![10.0, 0.1, 0.09, 0.08];
        assert!(old_rule_splits(&deaths));
        let pts = line_points(&[0.0, 0.1, 0.2, 50.0, 50.1]);
        let distances = dtm_weighted_distances(pts.view(), None);
        let (n, within) = components_and_scale(&bars(&deaths), &distances);
        assert_eq!(
            n, 2,
            "a real inter-cluster gap with ≥2 per side must still split"
        );
        assert!(
            (within - 0.1).abs() < 1e-12,
            "within-scale is the coarsest sub-cut merge"
        );
    }

    #[test]
    fn lone_outlier_cut_is_not_split() {
        // Same dominant log-gap ≫ ln 2, but the cut would carve off a SINGLE
        // stray landmark (x=50) from a 3-landmark cluster. Below the cut scale 10
        // the outlier is its own component of size 1, so the guard rejects the
        // split and the atom is reported connected — a chart artifact, not a
        // second manifold component.
        let deaths = vec![10.0, 0.1, 0.09];
        assert!(old_rule_splits(&deaths));
        let pts = line_points(&[0.0, 0.1, 0.2, 50.0]);
        let distances = dtm_weighted_distances(pts.view(), None);
        let (n, _) = components_and_scale(&bars(&deaths), &distances);
        assert_eq!(n, 1, "a cut isolating one landmark must be rejected");
    }
}
