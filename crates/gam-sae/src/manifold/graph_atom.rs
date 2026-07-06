use super::*;

/// One undirected candidate edge between graph anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    pub a: usize,
    pub b: usize,
}

impl GraphEdge {
    pub fn new(a: usize, b: usize) -> Result<Self, String> {
        if a == b {
            return Err(format!("GraphEdge cannot join vertex {a} to itself"));
        }
        Ok(if a < b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        })
    }
}

/// Per-edge rank charge for a learned graph edge.
///
/// This is the tiered spine's BIC/Laplace currency applied to one edge:
/// `0.5 * d_eff * ln(n_eff)`. For a graph atom edge, `d_eff` is the number of
/// fiber channels coupled by that edge.
pub fn graph_edge_rank_charge(n_eff: f64, fiber_rank: usize) -> f64 {
    0.5 * fiber_rank as f64 * n_eff.max(2.0).ln()
}

/// Exact read-out of the surviving graph topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTopologyReadout {
    pub vertices: usize,
    pub surviving_edges: usize,
    pub b0: usize,
    pub b1: usize,
}

/// Named-shape compression certified after the graph has been learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCompressionKind {
    Circle,
    Interval,
    FiniteSet,
    Cylinder,
    Torus,
    Sphere,
    Graph,
}

/// MDL read-out for whether the learned edge set earns a standard name.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphCompressionReport {
    pub kind: GraphCompressionKind,
    pub name: &'static str,
    pub generic_edge_bits: f64,
    pub named_bits: f64,
    pub bits_saved: f64,
}

/// The structure-search birth currency for a graph atom.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStructureSelection {
    pub selected: bool,
    pub total_edge_delta_loss: f64,
    pub total_edge_charge: f64,
    pub margin: f64,
    pub topology: GraphTopologyReadout,
    pub occupancy: OccupancyLaw,
    pub compression: GraphCompressionReport,
}

impl GraphCompressionReport {
    pub fn certified(
        kind: GraphCompressionKind,
        name: &'static str,
        generic_edge_bits: f64,
        named_bits: f64,
    ) -> Self {
        Self {
            kind,
            name,
            generic_edge_bits,
            named_bits,
            bits_saved: generic_edge_bits - named_bits,
        }
    }

    pub fn unnamed(generic_edge_bits: f64) -> Self {
        Self {
            kind: GraphCompressionKind::Graph,
            name: "structure without a standard name",
            generic_edge_bits,
            named_bits: generic_edge_bits,
            bits_saved: 0.0,
        }
    }

    pub fn earns_standard_name(&self) -> bool {
        self.kind != GraphCompressionKind::Graph && self.bits_saved > 0.0
    }
}

/// A canonical learned graph atom: anchors with a learned subset of a derived kNN
/// candidate edge set.
///
/// The smoothness penalty is `beta^T (L_W kron I_r) beta`, where `L_W` is the
/// weighted graph Laplacian of the surviving graph and `r` is the fiber rank.
/// Edge survival is read from the same rank-charge discipline used by tiered
/// births/deaths: an edge survives only when the REML loss increase from
/// removing it is greater than its one-edge charge. Betti read-out is exact on
/// surviving edges; named shapes are secondary MDL compressions of the graph.
#[derive(Debug, Clone)]
pub struct LearnedGraphAtom {
    anchor_embeddings: Array2<f64>,
    candidate_edges: Vec<GraphEdge>,
    edge_precisions: Vec<f64>,
    edge_delta_loss: Vec<f64>,
    surviving_edges: Vec<bool>,
    n_eff: f64,
    occupancy: OccupancyLaw,
}

impl LearnedGraphAtom {
    /// Derived k for the anchor-kNN candidate graph. The graph atom is expected to
    /// learn edge survival by ARD; k only supplies a sparse local superset.
    pub fn derived_knn_k(anchors: usize) -> usize {
        anchors.saturating_sub(1).min(2)
    }

    /// Build the derived undirected kNN candidate edge set from anchor embeddings.
    pub fn knn_candidate_edges(
        anchor_embeddings: ArrayView2<'_, f64>,
    ) -> Result<Vec<GraphEdge>, String> {
        validate_anchor_embeddings(anchor_embeddings)?;
        let anchors = anchor_embeddings.nrows();
        let k = Self::derived_knn_k(anchors);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut edges = Vec::new();
        for a in 0..anchors {
            let mut distances = Vec::<(f64, usize)>::with_capacity(anchors.saturating_sub(1));
            for b in 0..anchors {
                if a == b {
                    continue;
                }
                let mut dist2 = 0.0_f64;
                for c in 0..anchor_embeddings.ncols() {
                    let d = anchor_embeddings[[a, c]] - anchor_embeddings[[b, c]];
                    dist2 += d * d;
                }
                distances.push((dist2, b));
            }
            distances.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.1.cmp(&right.1))
            });
            for &(_, b) in distances.iter().take(k) {
                let edge = GraphEdge::new(a, b)?;
                if !edges.contains(&edge) {
                    edges.push(edge);
                }
            }
        }
        edges.sort_by_key(|edge| (edge.a, edge.b));
        Ok(edges)
    }

    /// Build a learned graph atom from the derived kNN candidate edge set.
    pub fn from_reml_knn_edges(
        anchor_embeddings: ArrayView2<'_, f64>,
        row_coordinates: &[f64],
        n_eff: f64,
        edge_precisions: &[f64],
        edge_delta_loss: &[f64],
    ) -> Result<Self, String> {
        let candidate_edges = Self::knn_candidate_edges(anchor_embeddings)?;
        Self::from_reml_candidate_edges(
            anchor_embeddings,
            row_coordinates,
            n_eff,
            &candidate_edges,
            edge_precisions,
            edge_delta_loss,
        )
    }

    /// Build a graph atom from a caller-supplied candidate edge set.
    ///
    /// Production births should pass the kNN set from [`Self::knn_candidate_edges`].
    /// Tests and certified imports use this form to make the candidate superset
    /// explicit while preserving the same ARD survival rule.
    pub fn from_reml_candidate_edges(
        anchor_embeddings: ArrayView2<'_, f64>,
        row_coordinates: &[f64],
        n_eff: f64,
        candidate_edges: &[GraphEdge],
        edge_precisions: &[f64],
        edge_delta_loss: &[f64],
    ) -> Result<Self, String> {
        validate_anchor_embeddings(anchor_embeddings)?;
        let anchors = anchor_embeddings.nrows();
        let fiber_rank = anchor_embeddings.ncols();
        if candidate_edges.is_empty() {
            return Err("LearnedGraphAtom requires at least one candidate edge".to_string());
        }
        if edge_precisions.len() != candidate_edges.len() {
            return Err(format!(
                "LearnedGraphAtom edge_precisions length {} must equal candidate edges {}",
                edge_precisions.len(),
                candidate_edges.len()
            ));
        }
        if edge_delta_loss.len() != candidate_edges.len() {
            return Err(format!(
                "LearnedGraphAtom edge_delta_loss length {} must equal candidate edges {}",
                edge_delta_loss.len(),
                candidate_edges.len()
            ));
        }
        if !(n_eff.is_finite() && n_eff > 0.0) {
            return Err(format!(
                "LearnedGraphAtom n_eff must be finite and positive; got {n_eff}"
            ));
        }
        let mut normalized = Vec::<GraphEdge>::with_capacity(candidate_edges.len());
        for (idx, edge) in candidate_edges.iter().enumerate() {
            if edge.a >= anchors || edge.b >= anchors || edge.a == edge.b {
                return Err(format!(
                    "LearnedGraphAtom candidate edge {idx} = ({}, {}) is invalid for {anchors} anchors",
                    edge.a, edge.b
                ));
            }
            let edge = GraphEdge::new(edge.a, edge.b)?;
            if normalized.contains(&edge) {
                return Err(format!(
                    "LearnedGraphAtom candidate edge {idx} duplicates ({}, {})",
                    edge.a, edge.b
                ));
            }
            normalized.push(edge);
        }
        for (edge, &precision) in edge_precisions.iter().enumerate() {
            if !(precision.is_finite() && precision >= 0.0) {
                return Err(format!(
                    "LearnedGraphAtom edge {edge} precision must be finite and nonnegative; got {precision}"
                ));
            }
        }
        for (edge, &delta) in edge_delta_loss.iter().enumerate() {
            if !delta.is_finite() {
                return Err(format!(
                    "LearnedGraphAtom edge {edge} deletion loss must be finite; got {delta}"
                ));
            }
        }

        let charge = graph_edge_rank_charge(n_eff, fiber_rank);
        let surviving_edges = edge_precisions
            .iter()
            .zip(edge_delta_loss.iter())
            .map(|(&precision, &delta)| precision > 0.0 && delta > charge)
            .collect();

        Ok(Self {
            anchor_embeddings: anchor_embeddings.to_owned(),
            candidate_edges: normalized,
            edge_precisions: edge_precisions.to_vec(),
            edge_delta_loss: edge_delta_loss.to_vec(),
            surviving_edges,
            n_eff,
            occupancy: classify_occupancy(row_coordinates),
        })
    }

    pub fn anchors(&self) -> usize {
        self.anchor_embeddings.nrows()
    }

    pub fn fiber_rank(&self) -> usize {
        self.anchor_embeddings.ncols()
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    pub fn one_edge_charge(&self) -> f64 {
        graph_edge_rank_charge(self.n_eff, self.fiber_rank())
    }

    pub fn summed_edge_charge(&self) -> f64 {
        self.one_edge_charge() * self.topology_readout().surviving_edges as f64
    }

    pub fn occupancy(&self) -> OccupancyLaw {
        self.occupancy
    }

    pub fn candidate_edges(&self) -> &[GraphEdge] {
        &self.candidate_edges
    }

    pub fn edge_precisions(&self) -> &[f64] {
        &self.edge_precisions
    }

    pub fn edge_delta_loss(&self) -> &[f64] {
        &self.edge_delta_loss
    }

    pub fn surviving_edges(&self) -> &[bool] {
        &self.surviving_edges
    }

    /// Vertex degrees in the surviving graph.
    pub fn surviving_degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0usize; self.anchors()];
        for (idx, edge) in self.candidate_edges.iter().enumerate() {
            if self.surviving_edges[idx] {
                degrees[edge.a] += 1;
                degrees[edge.b] += 1;
            }
        }
        degrees
    }

    /// Weighted graph Laplacian `L_W` over all non-retired edges.
    pub fn surviving_laplacian(&self) -> Array2<f64> {
        self.weighted_laplacian_from_mask(&self.surviving_edges)
    }

    /// Weighted graph Laplacian `L_W` before edge retirement.
    pub fn full_laplacian(&self) -> Array2<f64> {
        let all_edges = vec![true; self.candidate_edges.len()];
        self.weighted_laplacian_from_mask(&all_edges)
    }

    /// Smoothness value `beta^T (L_W kron I_r) beta =
    /// sum_e w_e ||beta_i-beta_j||^2` over the surviving edge set.
    pub fn surviving_smoothness_value(&self) -> f64 {
        self.smoothness_value_from_mask(&self.surviving_edges)
    }

    /// Exact `O(E)` topology read-out from the surviving edge set. No persistence
    /// and no topology menu are involved.
    pub fn topology_readout(&self) -> GraphTopologyReadout {
        let vertices = self.anchors();
        let mut parent: Vec<usize> = (0..vertices).collect();
        let mut surviving_edges = 0usize;

        for (idx, edge) in self.candidate_edges.iter().enumerate() {
            if self.surviving_edges[idx] {
                surviving_edges += 1;
                graph_union(&mut parent, edge.a, edge.b);
            }
        }

        let mut roots = Vec::with_capacity(vertices);
        for vertex in 0..vertices {
            let root = graph_find(&mut parent, vertex);
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        let b0 = roots.len();
        let b1 = surviving_edges + b0 - vertices;
        GraphTopologyReadout {
            vertices,
            surviving_edges,
            b0,
            b1,
        }
    }

    /// Certified named compression of the learned graph. A positive bit saving
    /// returns the named shape; otherwise the structure remains an unnamed graph.
    pub fn certified_compression(&self) -> GraphCompressionReport {
        let readout = self.topology_readout();
        let degrees = self.surviving_degrees();
        let max_edges = readout.vertices.saturating_mul(readout.vertices.saturating_sub(1)) / 2;
        let generic = crate::description_length::selection_bits(
            max_edges as i64,
            readout.surviving_edges as i64,
        );
        let log_vertices = (readout.vertices.max(2) as f64).log2();
        let named = if readout.b0 == 1
            && readout.b1 == 1
            && degrees.iter().all(|&d| d == 2)
            && self.surviving_edge_weights_are_uniform()
        {
            Some((GraphCompressionKind::Circle, "circle", log_vertices))
        } else if readout.b0 == 1
            && readout.b1 == 0
            && readout.vertices >= 2
            && degrees.iter().filter(|&&d| d == 1).count() == 2
            && degrees.iter().filter(|&&d| d == 2).count() == readout.vertices.saturating_sub(2)
            && self.surviving_edge_weights_are_uniform()
        {
            Some((GraphCompressionKind::Interval, "interval", 2.0 * log_vertices))
        } else if matches!(
            self.occupancy,
            OccupancyLaw::Discrete { anchors } if anchors == readout.vertices
        ) && readout.b1 == 0
        {
            Some((GraphCompressionKind::FiniteSet, "finite_set", log_vertices))
        } else {
            None
        };
        if let Some((kind, name, named_bits)) = named {
            let report = GraphCompressionReport::certified(kind, name, generic, named_bits);
            if report.bits_saved > 0.0 {
                return report;
            }
        }
        GraphCompressionReport::unnamed(generic)
    }

    /// Birth-selection readout: graph existence is paid for by the sum of the
    /// surviving one-edge charges, not by a fixed topology menu.
    pub fn structure_selection(&self) -> GraphStructureSelection {
        let topology = self.topology_readout();
        let total_edge_delta_loss = self
            .edge_delta_loss
            .iter()
            .zip(self.surviving_edges.iter())
            .filter_map(|(&delta, &survives)| survives.then_some(delta))
            .sum::<f64>();
        let total_edge_charge = self.one_edge_charge() * topology.surviving_edges as f64;
        let margin = total_edge_delta_loss - total_edge_charge;
        GraphStructureSelection {
            selected: topology.surviving_edges > 0 && margin > 0.0,
            total_edge_delta_loss,
            total_edge_charge,
            margin,
            topology,
            occupancy: self.occupancy,
            compression: self.certified_compression(),
        }
    }

    pub fn surviving_penalty_op(
        &self,
        global_offset: usize,
        beta_dim: usize,
    ) -> Arc<dyn gam_solve::arrow_schur::BetaPenaltyOp> {
        Arc::new(IdentityRightKroneckerPenaltyOp {
            factor_a: self.surviving_laplacian(),
            p: self.fiber_rank(),
            global_offset,
            k: beta_dim,
        })
    }

    fn weighted_laplacian_from_mask(&self, active_edges: &[bool]) -> Array2<f64> {
        let anchors = self.anchors();
        let mut laplacian = Array2::<f64>::zeros((anchors, anchors));
        for (idx, edge) in self.candidate_edges.iter().enumerate() {
            if !active_edges[idx] {
                continue;
            }
            let w = self.edge_precisions[idx];
            if w == 0.0 {
                continue;
            }
            laplacian[[edge.a, edge.a]] += w;
            laplacian[[edge.b, edge.b]] += w;
            laplacian[[edge.a, edge.b]] -= w;
            laplacian[[edge.b, edge.a]] -= w;
        }
        laplacian
    }

    fn smoothness_value_from_mask(&self, active_edges: &[bool]) -> f64 {
        let fiber_rank = self.fiber_rank();
        let mut value = 0.0_f64;
        for (idx, edge) in self.candidate_edges.iter().enumerate() {
            if !active_edges[idx] {
                continue;
            }
            let w = self.edge_precisions[idx];
            if w == 0.0 {
                continue;
            }
            for channel in 0..fiber_rank {
                let diff = self.anchor_embeddings[[edge.a, channel]]
                    - self.anchor_embeddings[[edge.b, channel]];
                value += w * diff * diff;
            }
        }
        value
    }

    fn surviving_edge_weights_are_uniform(&self) -> bool {
        let mut min_weight = f64::INFINITY;
        let mut max_weight = f64::NEG_INFINITY;
        let mut count = 0usize;
        for (idx, survives) in self.surviving_edges.iter().enumerate() {
            if *survives {
                let weight = self.edge_precisions[idx];
                min_weight = min_weight.min(weight);
                max_weight = max_weight.max(weight);
                count += 1;
            }
        }
        if count == 0 {
            return false;
        }
        let scale = max_weight.abs().max(min_weight.abs()).max(1.0);
        max_weight - min_weight <= f64::EPSILON * scale * count as f64
    }
}

fn validate_anchor_embeddings(anchor_embeddings: ArrayView2<'_, f64>) -> Result<(), String> {
    let anchors = anchor_embeddings.nrows();
    let fiber_rank = anchor_embeddings.ncols();
    if anchors < 2 {
        return Err(format!(
            "LearnedGraphAtom requires at least 2 anchors; got {anchors}"
        ));
    }
    if fiber_rank == 0 {
        return Err("LearnedGraphAtom requires fiber_rank >= 1".to_string());
    }
    if anchor_embeddings.iter().any(|v| !v.is_finite()) {
        return Err("LearnedGraphAtom anchor_embeddings contain a non-finite value".to_string());
    }
    Ok(())
}

fn graph_find(parent: &mut [usize], x: usize) -> usize {
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

fn graph_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = graph_find(parent, a);
    let rb = graph_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}
