//! Connected components of an undirected graph by union–find.

use std::collections::BTreeMap;

fn union_root(parent: &mut [usize], mut vertex: usize) -> usize {
    while parent[vertex] != vertex {
        parent[vertex] = parent[parent[vertex]];
        vertex = parent[vertex];
    }
    vertex
}

/// The connected components of the graph on `0..count` with the given edges. Each component is
/// sorted, and components are ordered by their smallest vertex. The one component owner for every
/// banded-edge graph in the workspace: the caller decides which edges are resolved, this only joins
/// them.
pub fn connected_components(
    count: usize,
    edges: impl IntoIterator<Item = (usize, usize)>,
) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..count).collect();
    for (i, j) in edges {
        let root_i = union_root(&mut parent, i);
        let root_j = union_root(&mut parent, j);
        if root_i != root_j {
            parent[root_i.max(root_j)] = root_i.min(root_j);
        }
    }
    let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for vertex in 0..count {
        let root = union_root(&mut parent, vertex);
        blocks.entry(root).or_default().push(vertex);
    }
    blocks.into_values().collect()
}
