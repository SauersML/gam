//! #2280, stage 3: decide, with a continuous witness, whether the erroneous
//! swiss-roll `b₁ = 1` is a DISCRETE-nerve artifact (the continuous chart-domain
//! intersections are real but no SAMPLED row witnesses them) or whether the
//! continuous domains themselves leave a hole.
//!
//! The Sep-7 measurement localized the spurious class to the four inner-tip
//! patches {0, 18, 45, 48}: the full MEMBERSHIP nerve carries triangles
//! `(0,18,45)` (common row 88) and `(0,45,48)` (common row 87) but not
//! `(0,18,48)` or `(18,45,48)`, so the `K₄` 1-skeleton on those vertices keeps
//! one unfilled cycle. Whether a continuous chart-domain intersection is being
//! missed, or the actual domains leave a gap, was left open — this module is
//! that evidence.
//!
//! ## The witness
//!
//! A patch's `members` are its center's distance-order prefix, so the continuous
//! domain the patch IMPLIES is the closed ambient ball through its farthest
//! member, intersected with the sheet: `U_i = B(c_i, r_i) ∩ M` with
//! `r_i = max_{m ∈ members} ‖x_m − x_{c_i}‖` — the same `patch_radius` the
//! membership diagnostic already reads. Sampling those balls on a 4×-finer
//! grid of the SAME analytic sheet (the fixture generators ARE the sheets)
//! turns every nerve question into a decidable one:
//!
//! * a chart set co-fires continuously iff a witness-grid point lies in all
//!   its balls (grid spacing ~0.25 in `h` and ~0.12 in arc against patch radii
//!   ~1–2.6; the margin probe below reports how close each call is);
//! * the WITNESS nerve is enumerated with production's own
//!   [`enumerate_full_nerve`] and its Betti numbers read with
//!   [`compute_betti`], so the comparison is between two predicates on one
//!   enumerator, not between two implementations.
//!
//! Every membership intersection is a continuous one (a shared row is a sheet
//! point inside both balls), so the witness complex contains the membership
//! complex — asserted, not assumed. What the witness ADDS is exactly the
//! intersections no sampled row falls into. If the added triangles kill the
//! class, the hole is a sampling artifact of the discrete predicate; if the
//! class survives with the sheet fully covered, the continuous intersections
//! themselves fail to fill it; if the sheet is NOT fully covered, the domains
//! leave a real gap. The swiss-roll test pins whichever branch the witness
//! measures; torus/Möbius controls pin the machinery on known answers.

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
    /// Patch radius `r_i` (for reporting and margin arithmetic).
    radii: Vec<f64>,
    words: usize,
    /// `contains[p]` holds the membership bits of grid point `p` (`words`
    /// u64 words, chart `c` in word `c / 64` bit `c % 64`).
    contains: Vec<Vec<u64>>,
}

impl WitnessCover {
    fn build(atlas: &LocalAtlas, data: ArrayView2<'_, f64>, grid: Array2<f64>) -> Self {
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
    /// enumerator with the continuous predicate.
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

    /// Signed margin of a grid point against one chart's ball:
    /// `‖x − c‖ − r` (negative = inside the domain).
    fn margin(
        &self,
        data: ArrayView2<'_, f64>,
        grid: ArrayView2<'_, f64>,
        point: usize,
        chart: usize,
    ) -> f64 {
        let center = self.centers[chart];
        let distance = (0..data.ncols())
            .map(|column| {
                let delta = grid[[point, column]] - data[[center, column]];
                delta * delta
            })
            .sum::<f64>()
            .sqrt();
        distance - self.radii[chart]
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

/// The witness grid for a fixture: the fixture's own rows PLUS a 4×-refinement
/// of the same analytic sheet. Including the fixture rows exactly is what
/// makes containment provable (a shared row is a sheet point inside both patch
/// balls AND on the grid), and the refinement supplies the intersections no
/// sampled row falls into. `grid_shape` then names the refinement's extent for
/// the domain-connectivity flood fill; fixture rows are appended after it.
fn witness_grid(fine: Array2<f64>, data: &Array2<f64>, grid_shape: (usize, usize)) -> Array2<f64> {
    let _ = grid_shape;
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

/// Connected components of each continuous domain, counted on the witness
/// grid's REFINEMENT lattice (4-neighbor adjacency over the first
/// `grid_shape.0 * grid_shape.1` points; the appended fixture rows only add
/// points, never remove them, so a domain connected on the refinement is
/// connected on the full grid). A domain with more than one component breaks
/// the good-cover hypothesis the nerve theorem needs, so it is reported — the
/// witness Betti numbers are only a statement about the SHEET while every
/// domain is connected.
fn domain_components(
    witness: &WitnessCover,
    grid_shape: (usize, usize),
    fixture_shape: (usize, usize),
) -> Vec<usize> {
    let (n_t, n_h) = grid_shape;
    let (f_t, f_h) = fixture_shape;
    let charts = witness.centers.len();
    let lattice = n_t * n_h;
    let total = lattice + f_t * f_h;
    // Lattice adjacency (4-neighbor); each appended fixture row attaches to
    // the refinement point at its parametrization position and that point's
    // four lattice neighbors, so fixture rows can only ADD connectivity.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); total];
    for it in 0..n_t {
        for ih in 0..n_h {
            let point = it * n_h + ih;
            for (jt, jh) in [
                (it + 1, ih),
                (it.saturating_sub(1), ih),
                (it, ih + 1),
                (it, ih.saturating_sub(1)),
            ] {
                if jt < n_t && jh < n_h {
                    let next = jt * n_h + jh;
                    adjacency[point].push(next);
                }
            }
        }
    }
    for it in 0..f_t {
        for ih in 0..f_h {
            let fixture = lattice + it * f_h + ih;
            let near_t =
                ((it as f64 / (f_t as f64 - 1.0).max(1.0)) * (n_t - 1) as f64).round() as usize;
            let near_h =
                ((ih as f64 / (f_h as f64 - 1.0).max(1.0)) * (n_h - 1) as f64).round() as usize;
            let near = (near_t.min(n_t - 1)) * n_h + near_h.min(n_h - 1);
            adjacency[fixture].push(near);
            adjacency[near].push(fixture);
        }
    }
    let mut components = Vec::with_capacity(charts);
    for chart in 0..charts {
        let mut seen = vec![false; total];
        let mut count = 0;
        for start in 0..total {
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
    min_multiplicity: usize,
    added_edges: usize,
    added_triangles: usize,
    max_domain_components: usize,
}

fn check(
    label: &str,
    data: &Array2<f64>,
    fine: Array2<f64>,
    grid_shape: (usize, usize),
    fixture_shape: (usize, usize),
) -> WitnessReadout {
    let grid = witness_grid(fine, data, grid_shape);
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .unwrap_or_else(|e| panic!("{label} must build an atlas: {e}"));
    let witness = WitnessCover::build(&atlas, data.view(), grid.clone());

    let mut uncovered = 0;
    let mut min_multiplicity = usize::MAX;
    for point in 0..grid.nrows() {
        let multiplicity = witness.multiplicity(point);
        min_multiplicity = min_multiplicity.min(multiplicity as usize);
        if multiplicity == 0 {
            uncovered += 1;
        }
    }

    let membership = membership_nerve(&atlas);
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

    let components = domain_components(&witness, grid_shape, fixture_shape);
    let max_components = components.iter().copied().max().unwrap_or(0);
    eprintln!(
        "#2280 witness {label}: charts={} membership=({},{},{:?}) witness=({},{},{:?}) \
         added_edges={} added_triangles={} uncovered={} min_multiplicity={} max_domain_components={}",
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
                "#2280 witness {label}: DISCONNECTED domain chart={chart} center={} radius={:.4} components={}",
                witness.centers[chart], witness.radii[chart], components[chart]
            );
        }
    }
    WitnessReadout {
        witness_betti,
        membership_betti,
        uncovered,
        min_multiplicity,
        added_edges: witness_edges.len() - membership_edges.len(),
        added_triangles: witness_triangles.len() - membership_triangles.len(),
        max_domain_components: max_components,
    }
}

/// Margin of the best witness-grid point against a chart triple: the smallest
/// over grid points of the WORST ball margin, `min_p max_i (‖x_p − c_i‖ − r_i)`
/// (negative = a sheet point exists inside all three domains). Reported for
/// the two triples whose absent triangles carry the Sep-7 class, gated on the
/// atlas reproducing the named centers (rows 0, 143, 13, 131 for patches 0,
/// 18, 45, 48 respectively).
fn triple_margin(
    witness: &WitnessCover,
    data: &Array2<f64>,
    grid: &Array2<f64>,
    triple: [usize; 3],
    expected_centers: [usize; 3],
) -> Option<f64> {
    let actual: Vec<_> = triple.iter().map(|&c| witness.centers[c]).collect();
    if actual != expected_centers {
        eprintln!(
            "#2280 witness: triple {triple:?} centers drifted {actual:?} != {expected_centers:?}; margin skipped"
        );
        return None;
    }
    let mut best = f64::INFINITY;
    for point in 0..grid.nrows() {
        let worst = triple
            .iter()
            .map(|&chart| witness.margin(data.view(), grid.view(), point, chart))
            .fold(f64::NEG_INFINITY, f64::max);
        best = best.min(worst);
    }
    eprintln!(
        "#2280 witness: triple {triple:?} continuous margin {best:.4} (negative = nonempty intersection)"
    );
    Some(best)
}

#[test]
fn torus_witness_nerve_reads_the_true_topology_2280() {
    let readout = check(
        "torus_60x26",
        &torus(60, 26, 2.0, 0.8),
        torus(240, 104, 2.0, 0.8),
        (240, 104),
        (60, 26),
    );
    assert_eq!(readout.uncovered, 0, "torus domains must cover the sheet");
    assert_eq!(
        (
            readout.witness_betti.b0,
            readout.witness_betti.b1,
            readout.witness_betti.b2
        ),
        (1, 2, Some(1)),
        "torus witness nerve must read the true (1,2,1)"
    );
}

#[test]
fn mobius_witness_nerve_reads_the_true_topology_2280() {
    let readout = check(
        "mobius_40x10",
        &mobius_strip(40, 10),
        mobius_strip(160, 40),
        (160, 40),
        (40, 10),
    );
    assert_eq!(readout.uncovered, 0, "möbius domains must cover the sheet");
    assert_eq!(
        (
            readout.witness_betti.b0,
            readout.witness_betti.b1,
            readout.witness_betti.b2
        ),
        (1, 1, Some(0)),
        "möbius witness nerve must read the true (1,1,0)"
    );
}

#[test]
fn swiss_roll_witness_nerve_decides_the_continuous_question_2280() {
    let data = swiss_roll(80, 16);
    let fine = swiss_roll(320, 64);
    let readout = check("swiss_roll_80x16", &data, fine, (320, 64), (80, 16));
    let grid = witness_grid(swiss_roll(320, 64), &data, (320, 64));

    // The margins on the two triples whose missing triangles carry the Sep-7
    // class: {0,18,48} and {18,45,48} (patches 18 and 48 sit at grid rows
    // 8·16+15 = 143 and 8·16+3 = 131).
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .expect("swiss roll must build an atlas");
    let witness = WitnessCover::build(&atlas, data.view(), grid.clone());
    let first = triple_margin(&witness, &data, &grid, [0, 18, 48], [0, 143, 8 * 16 + 3]);
    let second = triple_margin(
        &witness,
        &data,
        &grid,
        [18, 45, 48],
        [8 * 16 + 15, 13, 8 * 16 + 3],
    );

    // Pre-registered fork, pinned by measurement. The sheet is contractible,
    // so the true signature is (1,0,0) and the membership nerve's (1,1,0) is
    // known-wrong. Whichever branch the witness reads is a FACT about the
    // continuous domains, and it names the fix lane:
    //   * (1,0,0): the continuous intersections exist and close the hole — the
    //     membership predicate under-witnesses them, fix the predicate;
    //   * (1,1,0) with uncovered > 0: the domains leave a real gap — fix
    //     coverage (density-aware budget), not the predicate;
    //   * (1,1,0) with uncovered == 0: the domains cover the sheet yet their
    //     intersections do not fill the cycle — the good-cover hypothesis
    //     itself is failing (disconnected intersections), and no predicate or
    //     budget fix can save the nerve readout without addressing it.
    eprintln!(
        "#2280 witness summary: membership=({},{},{:?}) witness=({},{},{:?}) added_edges={} \
         added_triangles={} min_multiplicity={} max_domain_components={}",
        readout.membership_betti.b0,
        readout.membership_betti.b1,
        readout.membership_betti.b2,
        readout.witness_betti.b0,
        readout.witness_betti.b1,
        readout.witness_betti.b2,
        readout.added_edges,
        readout.added_triangles,
        readout.min_multiplicity,
        readout.max_domain_components,
    );
    match (
        readout.witness_betti.b0,
        readout.witness_betti.b1,
        readout.uncovered,
    ) {
        (1, 0, 0) => {
            assert_eq!(
                readout.witness_betti.b2,
                Some(0),
                "contractible witness cover"
            );
            for (name, margin) in [("0-18-48", first), ("18-45-48", second)] {
                if let Some(margin) = margin {
                    assert!(
                        margin < 0.0,
                        "{name}: continuous intersection must exist when the witness kills the class"
                    );
                }
            }
            eprintln!("#2280 witness: SAMPLING ARTIFACT — continuous intersections close the hole");
        }
        (1, 1, uncovered) => {
            if uncovered == 0 {
                eprintln!(
                    "#2280 witness: GOOD-COVER FAILURE — domains cover the sheet yet the cycle survives; \
                     disconnected intersections are the suspect (max_domain_components={})",
                    readout.max_domain_components
                );
            } else {
                eprintln!(
                    "#2280 witness: REAL GAP — {uncovered} sheet points lie in no continuous domain; \
                     the fix lane is coverage (density-aware budget), not the predicate"
                );
            }
        }
        other => panic!("swiss-roll witness signature drifted to {other:?}"),
    }
}
