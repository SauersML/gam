//! #2280: sampled witnesses for intersections missed by the membership nerve.
//!
//! The Sep-7 measurement localized the Swiss-roll membership nerve's spurious
//! H1 class to patches {0, 18, 45, 48} of the ambient farthest-point cover. The
//! fixture rows supplied triangles `(0,18,45)` and `(0,45,48)`, but neither
//! `(0,18,48)` nor `(18,45,48)`, although sampled surface points lay inside all
//! three patch balls of each missing triple. Those inner-tip cells held 33–63
//! rows against a mean occupancy of 17.8, so the row-denominated patch budget
//! starved them. The atlas now places its centers in the occupancy-normalized
//! metric (`local_charts::occupancy_normalized_centers`), and the two nerves agree.
//!
//! Extend each patch to the closed ambient ball through its farthest member,
//! intersected with the analytic fixture surface: `U_i = B(c_i, r_i) ∩ M`.
//! Evaluate these domains on a grid refined fourfold in each parameter, with
//! the original fixture rows appended exactly. This preserves every original
//! membership intersection, including boundary rows without a sqrt roundtrip.
//! The production enumerator and Betti calculation compare the two predicates.
//!
//! Every membership simplex must co-fire on the grid (a shared row is a sheet
//! point within every patch radius). The converse does not follow: a finite grid
//! can miss intersections and uncovered regions between its points. Zero
//! uncovered grid points is only sampled coverage, and the witness nerve's Betti
//! numbers describe that finite nerve. These measurements do not certify a
//! continuous cover. A good cover requires every nonempty finite intersection to
//! be contractible; sampled domain connectivity alone cannot establish that.
//!
//! The Swiss-roll regression pins membership (1,0,0) and witness (1,0,0). Torus
//! and Möbius controls pin their known signatures. Connectivity is reported on
//! the refinement lattice with the fixture's actual seam identifications,
//! separately from the witness nerve.

use super::*;
use crate::inference::atlas_nerve::SimplexInventory;
use crate::manifold::local_charts::LocalAtlasConfig;
use crate::manifold::tests_topology_fixtures::{mobius_strip, swiss_roll, torus};
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

/// One cover's continuous geometry: per-chart center and radius, plus the
/// witness grid as per-point chart bitsets.
struct WitnessCover {
    centers: Vec<usize>,
    /// Patch radius `r_i` (for reporting).
    radii: Vec<f64>,
    words: usize,
    /// `contains[p]` holds the membership bits of grid point `p` (`words`
    /// u64 words, chart `c` in word `c / 64` bit `c % 64`).
    contains: Vec<Vec<u64>>,
}

impl WitnessCover {
    fn build(atlas: &LocalAtlas, data: ArrayView2<'_, f64>, grid: ArrayView2<'_, f64>) -> Self {
        let charts = atlas.chart_count();
        let words = charts.div_ceil(64);
        let mut centers = Vec::with_capacity(charts);
        let mut radii = Vec::with_capacity(charts);
        let mut radii2 = Vec::with_capacity(charts);
        for patch in atlas.patches() {
            centers.push(patch.center);
            let radius2 = patch
                .members
                .iter()
                .map(|&row| {
                    (0..data.ncols())
                        .map(|column| (data[[row, column]] - data[[patch.center, column]]).powi(2))
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max);
            radii2.push(radius2);
            radii.push(radius2.sqrt());
        }
        // `radii2` above holds each patch's max member SQUARED distance
        // exactly as accumulated — the containment test below compares the
        // same expression, so a farthest-member row compares EQUAL rather
        // than failing through a `sqrt(x)² ⪆ x` roundtrip.
        let mut contains = vec![vec![0_u64; words]; grid.nrows()];
        for (point, row) in grid.outer_iter().enumerate() {
            for chart in 0..charts {
                let distance2 = (0..data.ncols())
                    .map(|column| (row[column] - data[[centers[chart], column]]).powi(2))
                    .sum::<f64>();
                if distance2 <= radii2[chart] {
                    contains[point][chart / 64] |= 1_u64 << (chart % 64);
                }
            }
        }
        WitnessCover {
            centers,
            radii,
            words,
            contains,
        }
    }

    fn has(&self, point: usize, chart: usize) -> bool {
        self.contains[point][chart / 64] & (1_u64 << (chart % 64)) != 0
    }

    /// Does any witness-grid point lie in every chart's continuous domain?
    fn co_fires(&self, simplex: &[usize]) -> bool {
        let mut mask = vec![0_u64; self.words];
        for &chart in simplex {
            mask[chart / 64] |= 1_u64 << (chart % 64);
        }
        self.contains
            .iter()
            .any(|bits| bits.iter().zip(&mask).all(|(b, m)| b & m == *m))
    }

    /// Witness nerve at every cardinality, enumerated by the production
    /// enumerator with the sampled ball-intersection predicate.
    fn nerve(&self) -> SimplexInventory {
        let count = self.centers.len();
        let mut adjacency = vec![BTreeSet::new(); count];
        for a in 0..count {
            for b in (a + 1)..count {
                if self.co_fires(&[a, b]) {
                    adjacency[a].insert(b);
                    adjacency[b].insert(a);
                }
            }
        }
        enumerate_full_nerve(count, &|simplex| self.co_fires(simplex), &adjacency, None)
            .expect("witness nerve must enumerate")
    }

    /// Cover multiplicity of one grid point: how many continuous domains
    /// contain it.
    fn multiplicity(&self, point: usize) -> u32 {
        self.contains[point]
            .iter()
            .map(|word| word.count_ones())
            .sum()
    }
}

/// The full MEMBERSHIP nerve of an atlas, at every cardinality, through the
/// same enumerator (the production predicate: non-empty patch-membership
/// intersection).
fn membership_nerve(atlas: &LocalAtlas) -> SimplexInventory {
    let count = atlas.chart_count();
    let mut adjacency = vec![BTreeSet::new(); count];
    for a in 0..count {
        for b in (a + 1)..count {
            if !sorted_intersection(&atlas.patches()[a].members, &atlas.patches()[b].members)
                .is_empty()
            {
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }
    }
    enumerate_full_nerve(
        count,
        &|simplex| {
            let mut shared = atlas.patches()[simplex[0]].members.clone();
            for &chart in &simplex[1..] {
                shared = sorted_intersection(&shared, &atlas.patches()[chart].members);
            }
            !shared.is_empty()
        },
        &adjacency,
        None,
    )
    .expect("membership nerve must enumerate")
}

/// Append the exact fixture rows to the refined sample of the same surface.
/// Connectivity uses only the refinement; all rows participate in the nerve.
fn witness_grid(fine: Array2<f64>, data: &Array2<f64>) -> Array2<f64> {
    let mut grid = Array2::<f64>::zeros((fine.nrows() + data.nrows(), data.ncols()));
    for row in 0..fine.nrows() {
        for column in 0..fine.ncols() {
            grid[[row, column]] = fine[[row, column]];
        }
    }
    for row in 0..data.nrows() {
        for column in 0..data.ncols() {
            grid[[fine.nrows() + row, column]] = data[[row, column]];
        }
    }
    grid
}

#[derive(Clone, Copy)]
enum GridTopology {
    Rectangle,
    Torus,
    Mobius,
}

/// Count components of each domain's occupied REFINEMENT vertices, with
/// four-neighbor edges and the fixture's seam identifications. Torus coordinates
/// are periodic in both directions; the Möbius u seam reverses the width index.
/// The appended fixture rows are deliberately excluded: they need not coincide
/// with refinement vertices, and arbitrary attachments can create false splits
/// or joins. These graph counts do not certify continuous domain connectivity.
fn domain_components(
    witness: &WitnessCover,
    grid_shape: (usize, usize),
    topology: GridTopology,
) -> Vec<usize> {
    let (n_t, n_h) = grid_shape;
    assert!(n_t > 0 && n_h > 0, "refinement dimensions must be positive");
    let charts = witness.centers.len();
    let lattice = n_t * n_h;
    assert!(lattice <= witness.contains.len());
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); lattice];
    // Add each undirected edge once, including the seam edges.
    for it in 0..n_t {
        for ih in 0..n_h {
            let point = it * n_h + ih;
            let next_t = if it + 1 < n_t {
                Some((it + 1) * n_h + ih)
            } else {
                match topology {
                    GridTopology::Rectangle => None,
                    GridTopology::Torus => Some(ih),
                    GridTopology::Mobius => Some(n_h - 1 - ih),
                }
            };
            let next_h = if ih + 1 < n_h {
                Some(point + 1)
            } else if matches!(topology, GridTopology::Torus) {
                Some(it * n_h)
            } else {
                None
            };
            for next in [next_t, next_h].into_iter().flatten() {
                adjacency[point].push(next);
                adjacency[next].push(point);
            }
        }
    }
    let mut components = Vec::with_capacity(charts);
    for chart in 0..charts {
        let mut seen = vec![false; lattice];
        let mut count = 0;
        for start in 0..lattice {
            if seen[start] || !witness.has(start, chart) {
                continue;
            }
            count += 1;
            let mut queue = VecDeque::from([start]);
            seen[start] = true;
            while let Some(point) = queue.pop_front() {
                for &next in &adjacency[point] {
                    if !seen[next] && witness.has(next, chart) {
                        seen[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }
        components.push(count);
    }
    components
}

/// Everything `check` measured, so callers can assert exactly what they know
/// and report the rest.
struct WitnessReadout {
    witness_betti: BettiSignature,
    membership_betti: BettiSignature,
    uncovered: usize,
}

fn check(
    label: &str,
    atlas: &LocalAtlas,
    witness: &WitnessCover,
    grid_shape: (usize, usize),
    topology: GridTopology,
) -> WitnessReadout {
    let mut uncovered = 0;
    let mut min_multiplicity = usize::MAX;
    for point in 0..witness.contains.len() {
        let multiplicity = witness.multiplicity(point);
        min_multiplicity = min_multiplicity.min(multiplicity as usize);
        if multiplicity == 0 {
            uncovered += 1;
        }
    }

    let membership = membership_nerve(atlas);
    let nerve = witness.nerve();
    let membership_betti = compute_betti(
        &membership.vertices,
        &membership.edges,
        &membership.triangles,
        &membership.tetrahedra,
    );
    let witness_betti = compute_betti(
        &nerve.vertices,
        &nerve.edges,
        &nerve.triangles,
        &nerve.tetrahedra,
    );

    // Containment: every membership intersection is a continuous one (a shared
    // row is a sheet point within both patch radii), so every membership
    // simplex must co-fire on the witness grid.
    let membership_edges: BTreeSet<_> = membership.edges.iter().cloned().collect();
    let witness_edges: BTreeSet<_> = nerve.edges.iter().cloned().collect();
    let missing: Vec<_> = membership_edges.difference(&witness_edges).collect();
    assert!(
        missing.is_empty(),
        "{label}: membership edges absent from the witness nerve: {missing:?}"
    );
    let membership_triangles: BTreeSet<_> = membership.triangles.iter().cloned().collect();
    let witness_triangles: BTreeSet<_> = nerve.triangles.iter().cloned().collect();
    let missing_tris: Vec<_> = membership_triangles
        .difference(&witness_triangles)
        .collect();
    assert!(
        missing_tris.is_empty(),
        "{label}: membership triangles absent from the witness nerve: {missing_tris:?}"
    );

    let components = domain_components(witness, grid_shape, topology);
    let max_components = components.iter().copied().max().unwrap_or(0);
    eprintln!(
        "#2280 witness {label}: charts={} membership=({},{},{:?}) witness=({},{},{:?}) \
         added_edges={} added_triangles={} uncovered_grid_points={} min_multiplicity={} max_lattice_components={}",
        atlas.chart_count(),
        membership_betti.b0,
        membership_betti.b1,
        membership_betti.b2,
        witness_betti.b0,
        witness_betti.b1,
        witness_betti.b2,
        witness_edges.len() - membership_edges.len(),
        witness_triangles.len() - membership_triangles.len(),
        uncovered,
        min_multiplicity,
        max_components,
    );
    for chart in 0..components.len() {
        if components[chart] > 1 {
            eprintln!(
                "#2280 witness {label}: multiple lattice components chart={chart} center={} radius={:.4} components={}",
                witness.centers[chart], witness.radii[chart], components[chart]
            );
        }
    }
    WitnessReadout {
        witness_betti,
        membership_betti,
        uncovered,
    }
}

#[test]
fn torus_witness_nerve_reads_the_true_topology_2280() {
    let data = torus(60, 26, 2.0, 0.8);
    let grid = witness_grid(torus(240, 104, 2.0, 0.8), &data);
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .expect("torus must build an atlas");
    let witness = WitnessCover::build(&atlas, data.view(), grid.view());
    let readout = check(
        "torus_60x26",
        &atlas,
        &witness,
        (240, 104),
        GridTopology::Torus,
    );
    assert_eq!(
        readout.uncovered, 0,
        "torus must cover all witness-grid points"
    );
    for betti in [readout.membership_betti, readout.witness_betti] {
        assert_eq!((betti.b0, betti.b1, betti.b2), (1, 2, Some(1)));
    }
}

#[test]
fn mobius_witness_nerve_reads_the_true_topology_2280() {
    let data = mobius_strip(40, 10);
    let grid = witness_grid(mobius_strip(160, 40), &data);
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .expect("Möbius strip must build an atlas");
    let witness = WitnessCover::build(&atlas, data.view(), grid.view());
    let readout = check(
        "mobius_40x10",
        &atlas,
        &witness,
        (160, 40),
        GridTopology::Mobius,
    );
    assert_eq!(
        readout.uncovered, 0,
        "Möbius must cover all witness-grid points"
    );
    for betti in [readout.membership_betti, readout.witness_betti] {
        assert_eq!((betti.b0, betti.b1, betti.b2), (1, 1, Some(0)));
    }
}

#[test]
fn swiss_roll_witness_fills_missing_membership_intersections_2280() {
    let data = swiss_roll(80, 16);
    let grid = witness_grid(swiss_roll(320, 64), &data);
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .expect("Swiss roll must build an atlas");
    let witness = WitnessCover::build(&atlas, data.view(), grid.view());
    let readout = check(
        "swiss_roll_80x16",
        &atlas,
        &witness,
        (320, 64),
        GridTopology::Rectangle,
    );
    let membership = readout.membership_betti;
    assert_eq!(
        (membership.b0, membership.b1, membership.b2),
        (1, 0, Some(0)),
        "the occupancy-normalized cover's membership nerve must be the contractible sheet"
    );
    let sampled = readout.witness_betti;
    assert_eq!(
        (sampled.b0, sampled.b1, sampled.b2),
        (1, 0, Some(0)),
        "the refined witness nerve must read the contractible sheet"
    );
    assert_eq!(
        readout.uncovered, 0,
        "Swiss roll must cover all witness-grid points"
    );
}

#[test]
fn lattice_components_respect_surface_seams_2280() {
    let shape = (4, 5);
    let make_witness = |occupied: &[usize]| WitnessCover {
        centers: vec![0],
        radii: vec![1.0],
        words: 1,
        contains: (0..shape.0 * shape.1)
            .map(|point| vec![u64::from(occupied.contains(&point))])
            .collect(),
    };

    // One patch crosses BOTH torus seams. Cutting the parameter rectangle
    // manufactures four components from its four corner samples.
    let torus = make_witness(&[0, 4, 15, 19]);
    assert_eq!(
        domain_components(&torus, shape, GridTopology::Rectangle),
        vec![4]
    );
    assert_eq!(
        domain_components(&torus, shape, GridTopology::Torus),
        vec![1]
    );

    // A Möbius seam connects the last u row to the first with reversed width.
    let mobius = make_witness(&[1, 18]);
    assert_eq!(
        domain_components(&mobius, shape, GridTopology::Rectangle),
        vec![2]
    );
    assert_eq!(
        domain_components(&mobius, shape, GridTopology::Torus),
        vec![2]
    );
    assert_eq!(
        domain_components(&mobius, shape, GridTopology::Mobius),
        vec![1]
    );

    // Equal-width seam points are NOT neighbors on the Möbius strip.
    let untwisted = make_witness(&[1, 16]);
    assert_eq!(
        domain_components(&untwisted, shape, GridTopology::Mobius),
        vec![2]
    );
    assert_eq!(
        domain_components(&untwisted, shape, GridTopology::Torus),
        vec![1]
    );
}

#[test]
fn lattice_components_exclude_appended_fixture_rows_2280() {
    let mut witness = WitnessCover {
        centers: vec![0],
        radii: vec![1.0],
        words: 1,
        contains: vec![vec![1]; 4],
    };
    assert_eq!(
        domain_components(&witness, (2, 2), GridTopology::Rectangle),
        vec![1]
    );
    // A fixture row has no lattice edge. Counting it as an isolated vertex
    // would manufacture a component even when its position duplicates a sample.
    witness.contains.push(vec![1]);
    assert_eq!(
        domain_components(&witness, (2, 2), GridTopology::Rectangle),
        vec![1]
    );
    assert!(
        witness.has(4, 0),
        "appended rows still participate in the witness predicate"
    );
}
