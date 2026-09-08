//! Reconstruct the cover independently of whether an overlap fits a transition.
//! Diagnostic cycles are graph cycles reduced modulo every triangular boundary,
//! so they identify actual H1 classes rather than spanning-tree chords alone.

use super::*;
use crate::inference::atlas_nerve::SimplexInventory;
use crate::manifold::local_charts::LocalAtlasConfig;
use crate::manifold::tests_topology_fixtures::{embedded_plane, mobius_strip, swiss_roll, torus};
use ndarray::Array2;
use std::collections::VecDeque;

fn inventory(atlas: &LocalAtlas, transition_edges_only: bool) -> SimplexInventory {
    let count = atlas.chart_count();
    let mut adjacency = vec![BTreeSet::new(); count];
    let transition_pairs: BTreeSet<_> = atlas
        .transitions()
        .iter()
        .map(|transition| {
            (
                transition.from_patch.min(transition.to_patch),
                transition.from_patch.max(transition.to_patch),
            )
        })
        .collect();
    for a in 0..count {
        for b in (a + 1)..count {
            let shared =
                sorted_intersection(&atlas.patches()[a].members, &atlas.patches()[b].members);
            if !shared.is_empty() && (!transition_edges_only || transition_pairs.contains(&(a, b)))
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
    .expect("fixture membership nerve must enumerate")
}

fn reduce_and_insert(
    mut chain: BTreeSet<usize>,
    basis: &mut BTreeMap<usize, BTreeSet<usize>>,
) -> bool {
    while let Some(&pivot) = chain.last() {
        if let Some(prior) = basis.get(&pivot) {
            chain = chain.symmetric_difference(prior).copied().collect();
        } else {
            basis.insert(pivot, chain);
            return true;
        }
    }
    false
}

fn h1_representatives(nerve: &SimplexInventory) -> Vec<Vec<(usize, usize)>> {
    let edge_indices: BTreeMap<_, _> = nerve
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| ((edge[0], edge[1]), index))
        .collect();
    let mut boundaries = BTreeMap::new();
    for triangle in &nerve.triangles {
        reduce_and_insert(
            [
                edge_indices[&(triangle[0], triangle[1])],
                edge_indices[&(triangle[0], triangle[2])],
                edge_indices[&(triangle[1], triangle[2])],
            ]
            .into_iter()
            .collect(),
            &mut boundaries,
        );
    }
    let mut adjacency = vec![Vec::new(); nerve.vertices.len()];
    for (&(a, b), &edge) in &edge_indices {
        adjacency[a].push((b, edge));
        adjacency[b].push((a, edge));
    }
    let mut root_paths: Vec<Option<BTreeSet<usize>>> = vec![None; adjacency.len()];
    for root in 0..adjacency.len() {
        if root_paths[root].is_some() {
            continue;
        }
        root_paths[root] = Some(BTreeSet::new());
        let mut queue = VecDeque::from([root]);
        while let Some(a) = queue.pop_front() {
            for &(b, edge) in &adjacency[a] {
                if root_paths[b].is_none() {
                    let mut path = root_paths[a].as_ref().expect("queued vertex has a root path").clone();
                    path.insert(edge);
                    root_paths[b] = Some(path);
                    queue.push_back(b);
                }
            }
        }
    }
    let mut representatives = Vec::new();
    for (&(a, b), &edge) in &edge_indices {
        let mut cycle: BTreeSet<_> = root_paths[a]
            .as_ref()
            .expect("edge source has a root path")
            .symmetric_difference(root_paths[b].as_ref().expect("edge destination has a root path"))
            .copied()
            .collect();
        if !cycle.insert(edge) {
            cycle.remove(&edge);
        }
        if reduce_and_insert(cycle.clone(), &mut boundaries) {
            representatives.push(
                cycle
                    .into_iter()
                    .map(|index| (nerve.edges[index][0], nerve.edges[index][1]))
                    .collect(),
            );
        }
    }
    let betti = compute_betti(
        &nerve.vertices,
        &nerve.edges,
        &nerve.triangles,
        &nerve.tetrahedra,
    );
    assert_eq!(representatives.len(), betti.b1, "cycle basis must span H1");
    representatives
}

fn check_cover(label: &str, data: Array2<f64>, expected: (usize, usize, Option<usize>)) {
    let atlas = LocalAtlas::build(data.view(), LocalAtlasConfig::balanced(data.nrows(), 2))
        .expect("known surface must build an atlas");
    let observed = observe_atlas_topology(&atlas).expect("known surface must read out");
    let full = inventory(&atlas, false);
    let fitted = inventory(&atlas, true);
    let full_betti = compute_betti(
        &full.vertices,
        &full.edges,
        &full.triangles,
        &full.tetrahedra,
    );
    let fitted_betti = compute_betti(
        &fitted.vertices,
        &fitted.edges,
        &fitted.triangles,
        &fitted.tetrahedra,
    );
    eprintln!(
        "#2280 {label}: observed={observed}; membership={full_betti:?}; fitted={fitted_betti:?}"
    );
    let full_edges: BTreeSet<_> = full.edges.iter().cloned().collect();
    let fitted_edges: BTreeSet<_> = fitted.edges.iter().cloned().collect();
    let mut owned_rows = vec![Vec::new(); atlas.chart_count()];
    for row in 0..data.nrows() {
        let distance = |chart: usize| {
            let center = atlas.patches()[chart].center;
            (0..data.ncols())
                .map(|column| (data[[row, column]] - data[[center, column]]).powi(2))
                .sum::<f64>()
        };
        let owner = (0..atlas.chart_count())
            .min_by(|&a, &b| distance(a).total_cmp(&distance(b)).then_with(|| a.cmp(&b)))
            .expect("atlas has a chart to own the row");
        owned_rows[owner].push(row);
    }
    for (chart, rows) in owned_rows.iter().enumerate() {
        let patch = &atlas.patches()[chart];
        let missing: Vec<_> = rows
            .iter()
            .copied()
            .filter(|row| patch.members.binary_search(row).is_err())
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "#2280 {label} patch={chart} center={} owns={} members={} missing_own_rows={missing:?}",
                patch.center, rows.len(), patch.members.len()
            );
        }
    }
    for edge in full_edges.difference(&fitted_edges) {
        let a = &atlas.patches()[edge[0]];
        let b = &atlas.patches()[edge[1]];
        eprintln!(
            "#2280 {label} missing_edge={edge:?} centers=({}, {}) overlap={:?}",
            a.center,
            b.center,
            sorted_intersection(&a.members, &b.members)
        );
    }
    for (cycle, edges) in h1_representatives(&fitted).iter().enumerate() {
        let patches: BTreeSet<_> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        for chart in patches {
            let patch = &atlas.patches()[chart];
            let radius = |rows: &[usize]| {
                rows.iter()
                    .map(|&row| {
                        (0..data.ncols())
                            .map(|column| {
                                (data[[row, column]] - data[[patch.center, column]]).powi(2)
                            })
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max)
                    .sqrt()
            };
            eprintln!(
                "#2280 {label} cycle={cycle} patch={chart} center={} owns={} own_radius={:e} patch_radius={:e} members={:?}",
                patch.center,
                owned_rows[chart].len(),
                radius(&owned_rows[chart]),
                radius(&patch.members),
                patch.members
            );
        }
        for &(a, b) in edges {
            let pa = &atlas.patches()[a];
            let pb = &atlas.patches()[b];
            eprintln!(
                "#2280 {label} cycle={cycle} edge=({a},{b}) centers=({},{}) overlap={:?}",
                pa.center,
                pb.center,
                sorted_intersection(&pa.members, &pb.members)
            );
        }
    }
    assert_eq!(
        (full_betti.b0, full_betti.b1, full_betti.b2),
        expected,
        "{label}: membership nerve"
    );
    assert_eq!(
        observed.invariants().betti,
        full_betti,
        "{label}: transition fitting must not change the nerve"
    );
}

#[test]
fn swiss_roll_nerve_is_the_contractible_membership_cover_2280() {
    check_cover("swiss_roll_80x16", swiss_roll(80, 16), (1, 0, Some(0)));
}

#[test]
fn transition_sample_requirement_cannot_change_the_membership_nerve_2280() {
    let data = embedded_plane(24, 24);
    let config = LocalAtlasConfig::balanced(data.nrows(), 2);
    let fitted = LocalAtlas::build(data.view(), config).unwrap();
    let no_transitions = LocalAtlas::build(
        data.view(),
        LocalAtlasConfig {
            min_overlap: config.patch_size + 1,
            ..config
        },
    )
    .unwrap();
    assert_eq!(fitted.patches(), no_transitions.patches());
    assert!(!fitted.transitions().is_empty());
    assert!(no_transitions.transitions().is_empty());
    let baseline = observe_atlas_topology(&fitted).unwrap();
    let unresolved = observe_atlas_topology(&no_transitions).unwrap();
    assert_eq!(baseline.invariants().betti.b0, 1);
    assert_eq!(baseline.invariants().betti.b1, 0);
    assert_eq!(unresolved.invariants().betti, baseline.invariants().betti);
    assert_eq!(unresolved.invariants().simplex_counts, baseline.invariants().simplex_counts);
    assert_eq!(unresolved.invariants().euler_characteristic, baseline.invariants().euler_characteristic);
    assert_eq!(unresolved.invariants().signed_subcomplex_betti.b0, no_transitions.chart_count());
    assert!(matches!(unresolved.refusal(), Some(TopologyRefusal::OrientationSubcomplexIncomplete { .. })));
}

#[test]
fn torus_membership_cycles_survive_transition_conditioning_2280() {
    check_cover("torus_60x26", torus(60, 26, 2.0, 0.8), (1, 2, Some(1)));
}

#[test]
fn mobius_membership_cycle_survives_transition_conditioning_2280() {
    check_cover("mobius_40x10", mobius_strip(40, 10), (1, 1, Some(0)));
}
