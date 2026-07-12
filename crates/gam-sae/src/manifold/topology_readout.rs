//! Atlas topology readout for the SAE manifold fit (#2280).
//!
//! A DIAGNOSTIC that runs alongside the intrinsic-metric seed race
//! ([`super::intrinsic_geodesic_embedding`]) and reports the coarse topology of an
//! atom's output-energy cluster: is the chart a closed sphere, a closed torus, a
//! non-orientable Möbius band, or a bounded flat sheet? The seed race remains the
//! infallible arbiter (REML evidence adjudicates the chart family); this readout
//! *informs* topology recognition where the race's fixed candidate menu can miss a
//! shape — a half-twist, or a genuinely closed curved surface.
//!
//! # Why this construction (a falsification-driven design)
//!
//! Two natural readouts were built in numpy and FALSIFIED before this one:
//!
//!   * NAÏVE NERVE χ + holonomy — a farthest-point *membership* cover gives a
//!     meaningless Euler characteristic (multi-way overlaps are noise-dominated and
//!     not contractible; the Nerve theorem's good-cover hypothesis fails) and an
//!     over-fragmented nerve with no essential cycle, so a Möbius band reads as
//!     orientable. Dead end.
//!   * WITNESS / DELAUNAY TRIANGULATION HOMOLOGY — χ and b₁ read off a landmark
//!     triangulation are hair-trigger sensitive to the handful of spurious
//!     triangulation holes a landmark cover always leaves; no dependency-free
//!     hole-fill closes them without occasionally welding a false handle (a torus χ
//!     flipped to an impossible odd −1). Triangulation homology is not robust from
//!     a landmark cover.
//!
//! What survives — and is validated stable across landmark density, noise, and
//! seed — is recognition from three LOCAL, per-landmark primitives that need no
//! globally consistent triangulation:
//!
//!   1. ORIENTABILITY — a local tangent `d`-frame per landmark (SVD of its Voronoi
//!      cell) is transported across every Voronoi-adjacent landmark pair by
//!      orthogonal Procrustes; the signed graph is NON-ORIENTABLE iff it cannot be
//!      consistently 2-gauged (an odd −1 cycle). The transition sign is trusted
//!      only where the two tangent planes genuinely coincide (Procrustes minimum
//!      singular value ≥ [`RELIABLE_SVAL`]), so one curved or degenerate edge
//!      cannot flip an orientable surface. A Möbius twist rides well-aligned
//!      interior edges, so its essential flip survives.
//!   2. BOUNDARY PRESENCE — the fraction of landmarks whose tangent-plane neighbour
//!      fan has an angular gap ≥ π (a half-disk neighbourhood = a manifold edge).
//!      ~0 for a closed surface, a whole ring for a bounded one.
//!   3. INTERIOR GAUSS–BONNET χ — the sum over interior landmarks of the ambient
//!      angle deficit `2π − Σθ` (θ the 3-D corner angles between consecutive
//!      neighbours), divided by `2π`. By Gauss–Bonnet this integrates the interior
//!      Gaussian curvature: `+2` for the sphere, `~0` for the developable torus.
//!
//! # Recognition (the four #2280 targets, plus the cylinder control)
//!
//! ```text
//!   non-orientable                    → Möbius (bounded) / Klein (closed)
//!   orientable, closed, χ ≥ 1         → sphere
//!   orientable, closed, χ < 1         → torus
//!   orientable, bounded               → flat sheet / disk (or cylinder)
//! ```
//!
//! DOCUMENTED LIMITATION (a falsification, not a defect): disk / swiss-roll sheet
//! vs cylinder — two orientable BOUNDED surfaces differing only by `b₁` (0 vs 1) —
//! is not separated by any robust local invariant here (both are developable so
//! interior χ ≈ 0; boundary-loop counting fragments; the full Gauss–Bonnet boundary
//! term drifts with density). The distinction is real but has no cheap robust
//! scalar, exactly as with the fold diagnostic. The SAE chart race already splits a
//! flat Duchon sheet from a cylinder by REML evidence; this readout's job is the
//! topology the menu can miss — non-orientability and closed curvature — which it
//! recognizes robustly.
//!
//! Determinism is fleet law: farthest-point landmarks seed from row 0, the geodesic
//! primitives break ties by index, and every traversal here iterates in index
//! order. Same input ⇒ bit-identical readout run-to-run. No RNG.

use super::{deterministic_knn_graph, farthest_point_landmarks, landmark_geodesics};
use gam_linalg::faer_ndarray::FaerSvd;
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

/// A Procrustes transition sign is a trustworthy orientation constraint only when
/// the two tangent planes actually coincide. `M = Fₐ Fᵦᵀ` is exactly orthogonal
/// (all singular values 1) when the planes match; a thin or highly-curved cell
/// gives a degenerate frame and a small minimum singular value. Excluding
/// transitions below this floor keeps one noisy edge from falsely flagging an
/// orientable surface, while a Möbius twist (well-aligned interior edges, min
/// singular value ≈ 1) is retained. `0.7` sits well below the ≈1 of a real chart
/// transition and well above the collapse of a degenerate overlap.
pub const RELIABLE_SVAL: f64 = 0.7;

/// A surface has a boundary when at least this fraction of landmarks sit on a
/// manifold edge (their tangent neighbour fan is a half-disk). Closed surfaces
/// (sphere/torus) score ~0; bounded ones (disk/sheet/cylinder/Möbius) score a whole
/// ring's worth (measured 0.26–0.47), so `0.12` cleanly separates them.
pub const BOUNDARY_FRAC: f64 = 0.12;

/// For an orientable closed surface, an interior Gauss–Bonnet χ at or above this
/// splits the sphere (χ ≈ 2, measured 1.9–2.1) from the torus (χ ≈ 0, measured
/// −0.9…−0.2). The margin is > 1, so the midpoint `1.0` is a safe, untuned split.
pub const GB_CHI_SPLIT: f64 = 1.0;

/// Recognized coarse topology of an atom's output-energy cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasTopology {
    /// Closed, orientable, positive integrated curvature (χ ≈ 2).
    Sphere,
    /// Closed, orientable, developable (χ ≈ 0).
    Torus,
    /// Non-orientable with a boundary (a half-twisted band).
    Mobius,
    /// Non-orientable and closed (χ ≈ 0).
    Klein,
    /// Orientable with a boundary — a flat sheet / disk (a cylinder folds in here;
    /// see the module note).
    SheetOrDisk,
    /// Too few landmarks / cells to certify any of the above.
    Indeterminate,
}

/// The raw invariants behind an [`AtlasTopology`] verdict, exposed so the seed race
/// can log or threshold them itself.
#[derive(Clone, Copy, Debug)]
pub struct AtlasReadout {
    /// The orientation-holonomy verdict.
    pub nonorientable: bool,
    /// Whether the manifold has a boundary (`frac_boundary ≥ `[`BOUNDARY_FRAC`]).
    pub has_boundary: bool,
    /// Fraction of landmarks flagged as boundary (angular gap ≥ π).
    pub frac_boundary: f64,
    /// Interior Gauss–Bonnet Euler characteristic estimate.
    pub gb_chi: f64,
    /// Number of reliable transition edges frustrated under the best 2-gauge
    /// (> 1 ⇒ non-orientable; a lone frustrated edge is treated as noise).
    pub frustrated: usize,
    /// Number of landmark cells that carried a usable tangent frame.
    pub n_cells: usize,
    /// The recognized topology.
    pub topology: AtlasTopology,
}

/// kNN degree for the readout graph — the same derived rule the intrinsic seed uses
/// (`max(2d+1, ⌈log₂ n⌉)`): enough neighbours to span the local tangent star and to
/// keep the graph (almost surely) connected before the primitives bridge it.
fn readout_knn(n: usize, d: usize) -> usize {
    let tangent = 2 * d + 1;
    let connectivity = (n.max(2) as f64).log2().ceil() as usize;
    tangent.max(connectivity).max(2)
}

/// Landmark count for the cover: `⌈1.5√n⌉` (a coverage net that grows sublinearly),
/// floored at `2(d+1)` so even small clusters over-determine the frames, capped at
/// `n`. Sparser than the seed's `4√n` because topology recognition wants larger,
/// smoother Voronoi cells (stable frames and neighbour fans) rather than a fine
/// metric embedding.
fn readout_landmark_count(n: usize, d: usize) -> usize {
    let coverage = (1.5 * (n as f64).sqrt()).ceil() as usize;
    coverage.max(2 * (d + 1)).min(n)
}

/// Top-`d` tangent frame at a cell: the leading `d` right singular vectors of the
/// mean-centered cell rows, as a `(d, p)` row-orthonormal matrix. `None` if the cell
/// is too small or the SVD fails.
fn cell_frame(z: ArrayView2<'_, f64>, rows: &[usize], d: usize) -> Option<Array2<f64>> {
    let m = rows.len();
    let p = z.ncols();
    if m < d + 1 || p == 0 {
        return None;
    }
    let mut x = Array2::<f64>::zeros((m, p));
    for (ri, &r) in rows.iter().enumerate() {
        for c in 0..p {
            x[[ri, c]] = z[[r, c]];
        }
    }
    for c in 0..p {
        let mean = (0..m).map(|ri| x[[ri, c]]).sum::<f64>() / m as f64;
        for ri in 0..m {
            x[[ri, c]] -= mean;
        }
    }
    let (_u, _s, vt) = x.svd(false, true).ok()?;
    let vt = vt?;
    if vt.nrows() < d {
        return None;
    }
    let mut frame = Array2::<f64>::zeros((d, p));
    for a in 0..d {
        for c in 0..p {
            frame[[a, c]] = vt[[a, c]];
        }
    }
    Some(frame)
}

/// Determinant of a small square matrix via Gaussian elimination with partial
/// pivoting (the readout only ever calls this on `d × d`, `d ≤ 3`). The sign is all
/// the orientation transport needs.
fn small_det(m: &Array2<f64>) -> f64 {
    let d = m.nrows();
    let mut a = m.clone();
    let mut det = 1.0;
    for col in 0..d {
        // partial pivot
        let mut piv = col;
        let mut best = a[[col, col]].abs();
        for r in (col + 1)..d {
            if a[[r, col]].abs() > best {
                best = a[[r, col]].abs();
                piv = r;
            }
        }
        if best == 0.0 {
            return 0.0;
        }
        if piv != col {
            for c in 0..d {
                let tmp = a[[col, c]];
                a[[col, c]] = a[[piv, c]];
                a[[piv, c]] = tmp;
            }
            det = -det;
        }
        det *= a[[col, col]];
        for r in (col + 1)..d {
            let f = a[[r, col]] / a[[col, col]];
            for c in col..d {
                a[[r, c]] -= f * a[[col, c]];
            }
        }
    }
    det
}

/// Orientation transport sign for a chart transition and its reliability (minimum
/// singular value of `M = Fₐ Fᵦᵀ`). `sign(det M) = sign(det R)` for the orthogonal
/// Procrustes rotation `R`, so the determinant sign is the transported orientation;
/// the minimum singular value certifies the two tangent planes actually coincide.
fn transition_sign(fa: &Array2<f64>, fb: &Array2<f64>) -> (f64, f64) {
    let d = fa.nrows();
    let mut m = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            let mut acc = 0.0;
            for c in 0..fa.ncols() {
                acc += fa[[a, c]] * fb[[b, c]];
            }
            m[[a, b]] = acc;
        }
    }
    let sign = if small_det(&m) >= 0.0 { 1.0 } else { -1.0 };
    let smin = match m.svd(false, false) {
        Ok((_, s, _)) => s.iter().cloned().fold(f64::INFINITY, f64::min),
        Err(_) => 0.0,
    };
    (sign, smin)
}

/// Compute the atlas topology readout for an ambient point cloud `z` (rows = the
/// atom's output-energy observations, columns = ambient dimension) assumed to lie
/// on a `d`-manifold. See the module docs for the mechanism and its falsification
/// history.
pub fn atlas_topology_readout(z: ArrayView2<'_, f64>, d: usize) -> AtlasReadout {
    let n = z.nrows();
    let indeterminate = AtlasReadout {
        nonorientable: false,
        has_boundary: false,
        frac_boundary: 0.0,
        gb_chi: 0.0,
        frustrated: 0,
        n_cells: 0,
        topology: AtlasTopology::Indeterminate,
    };
    if n < 3 * (d + 1) || z.ncols() == 0 || d == 0 {
        return indeterminate;
    }
    for v in z.iter() {
        if !v.is_finite() {
            return indeterminate;
        }
    }

    let k = readout_knn(n, d).min(n - 1);
    let adj = deterministic_knn_graph(z, k);
    let n_landmarks = readout_landmark_count(n, d);
    let landmarks = farthest_point_landmarks(z, n_landmarks);
    let l = landmarks.len();
    if l < d + 2 {
        return indeterminate;
    }
    let geo = landmark_geodesics(&adj, &landmarks); // (L, n)

    // Nearest landmark (Voronoi cell) per row; ties by lowest landmark index.
    let mut cell_of = vec![0usize; n];
    for r in 0..n {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for li in 0..l {
            let dv = geo[[li, r]];
            if dv < best_d {
                best_d = dv;
                best = li;
            }
        }
        cell_of[r] = best;
    }
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); l];
    for r in 0..n {
        cells[cell_of[r]].push(r);
    }

    // Voronoi adjacency (chart transitions): a graph edge crossing two cells.
    let mut vor: Vec<Vec<usize>> = vec![Vec::new(); l];
    {
        let mut seen = vec![std::collections::BTreeSet::<usize>::new(); l];
        for r in 0..n {
            let cr = cell_of[r];
            for &(s, _) in &adj[r] {
                let cs = cell_of[s];
                if cr != cs && seen[cr].insert(cs) {
                    vor[cr].push(cs);
                }
            }
        }
        for nbrs in vor.iter_mut() {
            nbrs.sort_unstable();
        }
    }

    // Per-cell tangent frame.
    let mut frames: Vec<Option<Array2<f64>>> = Vec::with_capacity(l);
    for li in 0..l {
        frames.push(cell_frame(z, &cells[li], d));
    }
    let n_cells = frames.iter().filter(|f| f.is_some()).count();

    // Boundary flag + interior Gauss–Bonnet deficit from each landmark's neighbour
    // fan in its own tangent plane.
    let mut n_framed = 0usize;
    let mut n_boundary = 0usize;
    let mut gb_sum = 0.0_f64;
    let two_pi = std::f64::consts::TAU;
    for i in 0..l {
        let fi = match &frames[i] {
            Some(f) => f,
            None => continue,
        };
        // Neighbours that also carry a frame.
        let nb: Vec<usize> = vor[i].iter().copied().filter(|&j| frames[j].is_some()).collect();
        n_framed += 1;
        if nb.len() < 2 {
            n_boundary += 1;
            continue;
        }
        // Tangent-plane bearing (frame axes 0,1) and ambient direction of each nb.
        let ci = landmarks[i];
        let mut bearings: Vec<(f64, usize)> = Vec::with_capacity(nb.len());
        for &j in &nb {
            let cj = landmarks[j];
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            for c in 0..z.ncols() {
                let diff = z[[cj, c]] - z[[ci, c]];
                t0 += fi[[0, c]] * diff;
                t1 += fi[[1, c]] * diff;
            }
            bearings.push((t1.atan2(t0), j));
        }
        bearings.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let m = bearings.len();
        // Angular gaps between consecutive bearings (wrapping).
        let mut max_gap = 0.0_f64;
        let mut gaps = vec![0.0_f64; m];
        for t in 0..m {
            let a = bearings[t].0;
            let b = bearings[(t + 1) % m].0;
            let mut g = (b - a) % two_pi;
            if g < 0.0 {
                g += two_pi;
            }
            gaps[t] = g;
            if g > max_gap {
                max_gap = g;
            }
        }
        if max_gap >= std::f64::consts::PI {
            n_boundary += 1;
            continue;
        }
        // Interior landmark: 3-D corner-angle sum over consecutive neighbours.
        let mut theta = 0.0_f64;
        for t in 0..m {
            let ja = landmarks[bearings[t].1];
            let jb = landmarks[bearings[(t + 1) % m].1];
            let mut dot = 0.0;
            let mut na = 0.0;
            let mut nbn = 0.0;
            for c in 0..z.ncols() {
                let va = z[[ja, c]] - z[[ci, c]];
                let vb = z[[jb, c]] - z[[ci, c]];
                dot += va * vb;
                na += va * va;
                nbn += vb * vb;
            }
            let cos = (dot / (na.sqrt() * nbn.sqrt() + 1.0e-12)).clamp(-1.0, 1.0);
            theta += cos.acos();
        }
        gb_sum += two_pi - theta;
    }
    let frac_boundary = if n_framed > 0 {
        n_boundary as f64 / n_framed as f64
    } else {
        0.0
    };
    let gb_chi = gb_sum / two_pi;

    // Orientability: reliability-gated Procrustes transport, odd-cycle (frustration)
    // test under a BFS 2-gauge in landmark-index order.
    let mut sign_edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..l {
        let fi = match &frames[i] {
            Some(f) => f,
            None => continue,
        };
        for &j in &vor[i] {
            if j <= i {
                continue;
            }
            if let Some(fj) = &frames[j] {
                let (sign, smin) = transition_sign(fi, fj);
                if smin >= RELIABLE_SVAL {
                    sign_edges.push((i, j, sign));
                }
            }
        }
    }
    let mut nadj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); l];
    for &(a, b, s) in &sign_edges {
        nadj[a].push((b, s));
        nadj[b].push((a, s));
    }
    let mut gauge = vec![0.0_f64; l]; // 0 = unvisited, ±1 = assigned
    for start in 0..l {
        if gauge[start] != 0.0 || nadj[start].is_empty() {
            continue;
        }
        gauge[start] = 1.0;
        let mut q = VecDeque::new();
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            for &(v, s) in &nadj[u] {
                if gauge[v] == 0.0 {
                    gauge[v] = gauge[u] * s;
                    q.push_back(v);
                }
            }
        }
    }
    let mut frustrated = 0usize;
    for &(a, b, s) in &sign_edges {
        if gauge[a] != 0.0 && gauge[b] != 0.0 && gauge[a] * gauge[b] * s < 0.0 {
            frustrated += 1;
        }
    }
    let nonorientable = frustrated > 1;
    let has_boundary = frac_boundary >= BOUNDARY_FRAC;

    let topology = if n_cells < d + 2 {
        AtlasTopology::Indeterminate
    } else if nonorientable {
        if has_boundary {
            AtlasTopology::Mobius
        } else {
            AtlasTopology::Klein
        }
    } else if !has_boundary {
        if gb_chi >= GB_CHI_SPLIT {
            AtlasTopology::Sphere
        } else {
            AtlasTopology::Torus
        }
    } else {
        AtlasTopology::SheetOrDisk
    };

    AtlasReadout {
        nonorientable,
        has_boundary,
        frac_boundary,
        gb_chi,
        frustrated,
        n_cells,
        topology,
    }
}
