use super::*;

/// Per-edge rank charge for a learned cycle-graph edge.
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

/// A first-slice graph atom: `m` anchors initialized as the cycle `C_m`, with a
/// nonnegative ARD precision per cycle edge.
///
/// The smoothness penalty is `beta^T (L_W kron I_r) beta`, where `L_W` is the
/// weighted graph Laplacian of the cycle and `r` is the fiber rank. Edge survival
/// is read from the same rank-charge discipline used by tiered births/deaths:
/// an edge survives only when the REML loss increase from removing it is greater
/// than its one-edge charge.
#[derive(Debug, Clone)]
pub struct CycleGraphAtom {
    anchor_embeddings: Array2<f64>,
    edge_precisions: Vec<f64>,
    edge_delta_loss: Vec<f64>,
    surviving_edges: Vec<bool>,
    n_eff: f64,
    occupancy: OccupancyLaw,
}

impl CycleGraphAtom {
    /// Build a cycle graph atom from REML-selected per-edge precisions and
    /// per-edge deletion losses.
    ///
    /// `anchor_embeddings` is `(m, r)`, one fiber vector per anchor. The cycle
    /// edges are `(0,1), (1,2), ..., (m-2,m-1), (m-1,0)`, so
    /// `edge_precisions` and `edge_delta_loss` must both have length `m`.
    /// `row_coordinates` are folded onto the unit circle and sent through the
    /// existing coordinate-fidelity occupancy classifier.
    pub fn from_reml_cycle_edges(
        anchor_embeddings: ArrayView2<'_, f64>,
        row_coordinates: &[f64],
        n_eff: f64,
        edge_precisions: &[f64],
        edge_delta_loss: &[f64],
    ) -> Result<Self, String> {
        let anchors = anchor_embeddings.nrows();
        let fiber_rank = anchor_embeddings.ncols();
        if anchors < 3 {
            return Err(format!(
                "CycleGraphAtom requires at least 3 anchors for C_m; got {anchors}"
            ));
        }
        if fiber_rank == 0 {
            return Err("CycleGraphAtom requires fiber_rank >= 1".to_string());
        }
        if edge_precisions.len() != anchors {
            return Err(format!(
                "CycleGraphAtom edge_precisions length {} must equal anchors {anchors}",
                edge_precisions.len()
            ));
        }
        if edge_delta_loss.len() != anchors {
            return Err(format!(
                "CycleGraphAtom edge_delta_loss length {} must equal anchors {anchors}",
                edge_delta_loss.len()
            ));
        }
        if !(n_eff.is_finite() && n_eff > 0.0) {
            return Err(format!(
                "CycleGraphAtom n_eff must be finite and positive; got {n_eff}"
            ));
        }
        if anchor_embeddings.iter().any(|v| !v.is_finite()) {
            return Err("CycleGraphAtom anchor_embeddings contain a non-finite value".to_string());
        }
        for (edge, &precision) in edge_precisions.iter().enumerate() {
            if !(precision.is_finite() && precision >= 0.0) {
                return Err(format!(
                    "CycleGraphAtom edge {edge} precision must be finite and nonnegative; got {precision}"
                ));
            }
        }
        for (edge, &delta) in edge_delta_loss.iter().enumerate() {
            if !delta.is_finite() {
                return Err(format!(
                    "CycleGraphAtom edge {edge} deletion loss must be finite; got {delta}"
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

    pub fn occupancy(&self) -> OccupancyLaw {
        self.occupancy
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

    /// Weighted cycle Laplacian `L_W` over all non-retired edges.
    pub fn surviving_laplacian(&self) -> Array2<f64> {
        self.weighted_laplacian_from_mask(&self.surviving_edges)
    }

    /// Weighted cycle Laplacian `L_W` before edge retirement.
    pub fn full_laplacian(&self) -> Array2<f64> {
        let all_edges = vec![true; self.anchors()];
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

        for edge in 0..vertices {
            if self.surviving_edges[edge] {
                surviving_edges += 1;
                let a = edge;
                let b = (edge + 1) % vertices;
                graph_union(&mut parent, a, b);
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
        for edge in 0..anchors {
            if !active_edges[edge] {
                continue;
            }
            let w = self.edge_precisions[edge];
            if w == 0.0 {
                continue;
            }
            let a = edge;
            let b = (edge + 1) % anchors;
            laplacian[[a, a]] += w;
            laplacian[[b, b]] += w;
            laplacian[[a, b]] -= w;
            laplacian[[b, a]] -= w;
        }
        laplacian
    }

    fn smoothness_value_from_mask(&self, active_edges: &[bool]) -> f64 {
        let anchors = self.anchors();
        let fiber_rank = self.fiber_rank();
        let mut value = 0.0_f64;
        for edge in 0..anchors {
            if !active_edges[edge] {
                continue;
            }
            let w = self.edge_precisions[edge];
            if w == 0.0 {
                continue;
            }
            let a = edge;
            let b = (edge + 1) % anchors;
            for channel in 0..fiber_rank {
                let diff =
                    self.anchor_embeddings[[a, channel]] - self.anchor_embeddings[[b, channel]];
                value += w * diff * diff;
            }
        }
        value
    }
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
