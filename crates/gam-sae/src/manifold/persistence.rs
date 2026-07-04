//! Persistent-homology audit of a raced atom's topology (reviewer F3).
//!
//! The topology race (`Periodic`/`Torus`/`Sphere`/`Cylinder`/`EuclideanPatch`/…)
//! is *model selection among a fixed library*: it always crowns a winner, even
//! when every candidate is misspecified. This module turns topology back into a
//! **measured property**. For each accepted atom it reads the atom's assigned
//! rows' in-atom residual-space positions — the decoded image points
//! `g_k(t_{ik}) = Φ_k(t_{ik}) B_k` of the rows this atom explains — and computes
//! their Vietoris–Rips persistent homology (H₀ = connected components, H₁ =
//! loops). The measured diagram is then confronted with what the *raced* type
//! predicts:
//!
//! * a circle / torus / cylinder should show ≥1 persistent H₁ loop and exactly
//!   one connected component;
//! * a line / Duchon patch / Euclidean patch / sphere-chart should show no
//!   persistent H₁ and one component;
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
//! The filtration values ARE the exact pairwise distances (no scale grid, no
//! bucketing), so the persistence diagram is computed exactly. "How many
//! components / is there a loop" is decided by a **dominant-gap** test on the
//! bar lengths (a single log-gap that outweighs all the others combined), which
//! carries no threshold. The only compute-side ceiling is the farthest-point
//! subsample cap [`PERSISTENCE_MAX_POINTS`] — a budget on the `O(m³)` triangle
//! enumeration, above the covering number of any modest atom, mirroring the
//! in-tree [`crate::manifold::shape_uncertainty::SHAPE_BAND_MAX_POINTS`] band
//! ceiling. Topology is invariant to it above the covering number.

use super::*;
use std::collections::HashMap;

/// Compute ceiling on the number of points fed to the Vietoris–Rips filtration.
///
/// The VR-2 complex has `O(m³)` triangles; the boundary reduction is quadratic
/// in the simplex count, so an uncapped atom (tens of thousands of assigned
/// rows) is intractable and — more importantly — pointless: the persistent
/// topology of a point cloud is fixed once the sample covers the manifold, and
/// a farthest-point subsample to a few dozen landmarks already covers any
/// modest atom. This is a compute budget, not a model knob: raising it only
/// spends more time confirming the same diagram. Kept in the same spirit (and
/// rough magnitude class, scaled down for the cubic cost) as
/// [`crate::manifold::shape_uncertainty::SHAPE_BAND_MAX_POINTS`].
pub const PERSISTENCE_MAX_POINTS: usize = 48;

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

/// The H₀/H₁ persistence diagram of a point cloud. `h0` always contains exactly
/// one essential bar (VR on a finite cloud is connected at its diameter); the
/// finite H₀ bars are the merge events. `h1` holds the loops (finite loops die
/// when their disk fills; an essential loop is never filled at any sampled
/// scale).
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
    pub h0: Vec<PersistenceBar>,
    pub h1: Vec<PersistenceBar>,
}

/// The per-atom topology audit certificate: the measured persistence summary of
/// an atom's assigned-row image points, the topology the raced type predicts,
/// and the `contested` verdict when they disagree.
#[derive(Clone, Debug)]
pub struct AtomTopologyPersistence {
    /// The raced basis/topology tag whose prediction is under audit.
    pub raced_kind: SaeAtomBasisKind,
    /// Number of image points actually persisted (post-subsample).
    pub n_points: usize,
    /// Measured number of connected components (dominant-gap partition of the
    /// finite H₀ bars, plus the one essential component). A connected manifold
    /// predicts `1`; a `c`-cluster set measures `c`.
    pub n_components: usize,
    /// Whether a persistent H₁ loop was measured (a bar whose persistence
    /// exceeds the within-component point-spacing scale, or an essential loop).
    pub has_loop: bool,
    /// Whether the raced type predicts a loop.
    pub expected_loop: bool,
    /// Persistence of the most persistent measured loop (`+∞` for an essential
    /// loop, `0` when none).
    pub dominant_h1_persistence: f64,
    /// The measured H₀ bars (essential bar included).
    pub h0: Vec<PersistenceBar>,
    /// The measured H₁ bars.
    pub h1: Vec<PersistenceBar>,
    /// The certificate flag: the measured topology disagrees with the raced
    /// type (extra components, a missing predicted loop, or a loop where none
    /// was predicted). Fed to the probe planner.
    pub contested: bool,
    /// Human-readable summary.
    pub note: String,
}

/// Whether the raced topology tag predicts a persistent H₁ loop. `None` for a
/// [`SaeAtomBasisKind::Precomputed`] atom whose topology is caller-supplied and
/// carries no library prediction to audit against.
fn kind_expects_loop(kind: &SaeAtomBasisKind) -> Option<bool> {
    match kind {
        // Circle / torus / cylinder charts carry non-trivial H₁.
        SaeAtomBasisKind::Periodic
        | SaeAtomBasisKind::Torus
        | SaeAtomBasisKind::Cylinder => Some(true),
        // The (lat, lon) sphere chart is topologically S², whose H₁ is trivial;
        // the contractible patch/curve charts likewise carry no loop.
        SaeAtomBasisKind::Sphere
        | SaeAtomBasisKind::Linear
        | SaeAtomBasisKind::Duchon
        | SaeAtomBasisKind::EuclideanPatch
        | SaeAtomBasisKind::Poincare => Some(false),
        // Caller-supplied basis: no library prediction, nothing to contest.
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
    let n = points.nrows();
    if n <= target {
        return (0..n).collect();
    }
    let mut chosen = Vec::with_capacity(target);
    chosen.push(0usize);
    let mut min_dist: Vec<f64> = (0..n).map(|i| point_distance(points, i, 0)).collect();
    while chosen.len() < target {
        let mut best = 0usize;
        let mut best_dist = -1.0_f64;
        for (i, &d) in min_dist.iter().enumerate() {
            if d > best_dist {
                best_dist = d;
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

/// One simplex in the Vietoris–Rips filtration: its sorted vertex set, its
/// filtration value (max pairwise distance among its vertices), and dimension.
struct Simplex {
    verts: Vec<usize>,
    filt: f64,
    dim: usize,
}

/// Exact Vietoris–Rips persistent homology up to H₁ (needs 2-simplices to kill
/// loops). Filtration values are the exact pairwise distances — no scale grid.
///
/// Reduction is the standard GF(2) boundary-matrix reduction: simplices are
/// ordered by `(filtration, dimension)`, each column is reduced against earlier
/// pivots, and a persistence pair `(birth-face, death-simplex)` is emitted when
/// a column's lowest surviving entry matches an existing pivot.
pub fn vietoris_rips_persistence(points: ArrayView2<'_, f64>) -> PersistenceDiagram {
    let m = points.nrows();
    let mut h0 = Vec::new();
    let mut h1 = Vec::new();
    if m == 0 {
        return PersistenceDiagram { h0, h1 };
    }
    if m == 1 {
        h0.push(PersistenceBar {
            birth: 0.0,
            death: f64::INFINITY,
        });
        return PersistenceDiagram { h0, h1 };
    }

    // Pairwise distances.
    let mut dist = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in (i + 1)..m {
            let d = point_distance(points, i, j);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }

    // Build simplices up to dimension 2.
    let mut simplices: Vec<Simplex> = Vec::new();
    for i in 0..m {
        simplices.push(Simplex {
            verts: vec![i],
            filt: 0.0,
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
                _ => {}
            }
        }
    }

    // Essential classes: empty (fully reduced) columns that were never consumed
    // as a birth.
    for j in 0..n {
        if boundary[j].is_empty() && reduced[j].is_empty() && !paired_birth[j] {
            let bar = PersistenceBar {
                birth: ordered_filt[j],
                death: f64::INFINITY,
            };
            match ordered_dim[j] {
                0 => h0.push(bar),
                1 => h1.push(bar),
                _ => {}
            }
        }
    }

    PersistenceDiagram { h0, h1 }
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
/// is accepted only when that one log-gap outweighs *all* the others combined —
/// a parameter-free "one gap dominates" rule that never fires on the roughly
/// uniform spacings of a single connected manifold.
///
/// Returns `(n_components, within_component_scale)`.
fn components_and_scale(finite_h0: &[PersistenceBar]) -> (usize, f64) {
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
    if gmax > sum_others && gmax > 0.0 {
        // Bars 0..=gmax_idx are inter-cluster merges: gmax_idx + 1 of them, so
        // gmax_idx + 2 components (the extra essential one). The within-cluster
        // scale is the largest death below the cut.
        let within = deaths[gmax_idx + 1];
        (gmax_idx + 2, within)
    } else {
        // Connected: the coarsest merge is the within-component spacing.
        (1, deaths[0])
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
    latent_dim: usize,
) -> Option<AtomTopologyPersistence> {
    let _ = latent_dim;
    let expected_loop = kind_expects_loop(raced_kind)?;
    let full = points.nrows();
    if full < 4 {
        return None;
    }
    let landmarks = farthest_point_subsample(points, PERSISTENCE_MAX_POINTS);
    let sub = points.select(ndarray::Axis(0), &landmarks);
    let diagram = vietoris_rips_persistence(sub.view());

    let finite_h0: Vec<PersistenceBar> = diagram
        .h0
        .iter()
        .copied()
        .filter(|b| !b.is_essential())
        .collect();
    let (n_components, within_scale) = components_and_scale(&finite_h0);

    // A measured loop is persistent when it outlives the within-component point
    // spacing (or is essential — an unfilled loop at every sampled scale).
    let dominant_h1_persistence = diagram
        .h1
        .iter()
        .map(|b| b.persistence())
        .fold(0.0_f64, f64::max);
    let has_loop = diagram
        .h1
        .iter()
        .any(|b| b.is_essential() || b.persistence() > within_scale);

    let contested = n_components != 1 || has_loop != expected_loop;

    let note = if contested {
        let mut reasons = Vec::new();
        if n_components != 1 {
            reasons.push(format!(
                "measured {n_components} components (raced type is a connected manifold)"
            ));
        }
        if has_loop && !expected_loop {
            reasons.push("measured a loop the raced type does not predict".to_string());
        }
        if !has_loop && expected_loop {
            reasons.push("no persistent loop where the raced type predicts one".to_string());
        }
        format!("CONTESTED topology: {}", reasons.join("; "))
    } else {
        format!(
            "topology agrees: {n_components} component(s), loop={has_loop} (predicted {expected_loop})"
        )
    };

    Some(AtomTopologyPersistence {
        raced_kind: raced_kind.clone(),
        n_points: landmarks.len(),
        n_components,
        has_loop,
        expected_loop,
        dominant_h1_persistence,
        h0: diagram.h0,
        h1: diagram.h1,
        contested,
        note,
    })
}

/// Gather one atom's assigned-row image points and audit its raced topology.
///
/// A row is *assigned* to an atom when that atom carries the row's largest
/// assignment mass (hard argmax) — the rows the atom actually explains. Each
/// assigned row's in-atom residual-space position is the decoded image
/// `g_k(t_{ik}) = Φ_k(t_{ik}) B_k`. Returns `None` when the atom's topology is
/// caller-supplied or too few rows are assigned to resolve H₁.
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
    // Hard argmax assignment: the rows this atom owns.
    let mut assigned_rows: Vec<usize> = Vec::new();
    for row in 0..n {
        let mut best = 0usize;
        let mut best_mass = f64::NEG_INFINITY;
        for col in 0..k {
            let mass = assignments[[row, col]];
            if mass > best_mass {
                best_mass = mass;
                best = col;
            }
        }
        if best == atom_idx && best_mass > 0.0 {
            assigned_rows.push(row);
        }
    }
    if assigned_rows.len() < 4 || assigned_rows.iter().any(|&r| r >= atom.n_obs()) {
        return None;
    }
    let p = atom.output_dim();
    let mut points = Array2::<f64>::zeros((assigned_rows.len(), p));
    for (i, &row) in assigned_rows.iter().enumerate() {
        let image = atom.decoded_row(row);
        for col in 0..p {
            points[[i, col]] = image[col];
        }
    }
    topology_persistence_verdict(points.view(), &atom.basis_kind, atom.latent_dim)
}
