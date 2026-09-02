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
    /// A contractible bounded surface (`χ = 1`, orientable, `b₁ = 0`, `b₂ = 0`):
    /// the topological type of a sampled sheet or disk. This is what a swiss roll
    /// glues to — the fold unrolls into one flat chart with no handle and no
    /// closed 2-cycle (#2280 acceptance: "swiss roll → sheet").
    Disk,
    Cylinder,
    /// The non-orientable bounded surface with one boundary loop (`χ = 0`,
    /// non-orientable, `b₁ = 1`, `b₂ = 0`): a cylinder's orientation cocycle with
    /// a single sign reversal around the loop. Recognized by the orientability
    /// certificate — the "half-twist as a discrete sign" (#2280 acceptance:
    /// "Möbius band → holonomy sign detected").
    MobiusStrip,
    Torus,
    Sphere,
    ProjectivePlane,
    KleinBottle,
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
    n_eff: f64,
    occupancy: OccupancyLaw,
}

impl LearnedGraphAtom {

    pub fn anchors(&self) -> usize {
        self.anchor_embeddings.nrows()
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    pub fn occupancy(&self) -> OccupancyLaw {
        self.occupancy
    }

}

// ===========================================================================
// SPECTRAL DECODE — the open-world atom's basis and out-of-sample coordinate.
//
// The learned graph atom certifies and prices topology (Betti read-out, named-
// shape MDL) but by itself cannot *reconstruct*: it has no basis `Φ` and no
// per-row coordinate. The spectral decode closes that gap without leaving the
// currency the atom already uses:
//
//   * BASIS — the leading `q` non-trivial eigenvectors of the SAME survived,
//     ARD-weighted Laplacian `L_W` that assembles the smoothness penalty. In
//     that eigenbasis the Dirichlet form IS the penalty: `Φᵀ L_W Φ = diag(λ)`
//     (a diagonalisation, not a parallel computation) — see
//     [`GraphSpectralBasis::penalty`].
//   * COORDINATE — a differentiable Nyström (geometric-harmonics) extension of
//     those eigenvectors to any out-of-sample row `z`, with an analytic jet.
//   * RACE — a decodable candidate ([`SpectralGraphRaceCandidate`]) presenting
//     the same {basis eval, penalty, jet, rank charge} interface the typed
//     atoms hand the birth topology race.
// ===========================================================================

/// Upper cap on the spectral decode dimension `q`. The eigengap rule
/// (`select_spectral_q`) never keeps more than this many non-trivial modes:
/// a decode coordinate is meant to be a *small* intrinsic chart (a circle is
/// `q = 2`, a torus `q = 4`), and the pricing charges every kept mode
/// ([`spectral_decode_rank_charge`]), so an unbounded `q` would both defeat the
/// compression story and make the Nyström jet needlessly wide.
pub const SPECTRAL_DECODE_MAX_Q: usize = 8;

/// The eigengap-selected spectral decode basis of a learned graph atom: the
/// leading `q` non-trivial eigenvectors of the survived weighted Laplacian
/// `L_W`, evaluated at the graph vertices, together with the eigenvalues that
/// ARE the Dirichlet penalty in this basis and the Gaussian-affinity Nyström
/// data that extends it out of sample.
#[derive(Debug, Clone)]
pub struct GraphSpectralBasis {
    /// `Φ` at the graph vertices, shape `(anchors × q)`. Column `k` is the
    /// unit-norm eigenvector `v_k` of `L_W` at the `k`-th smallest non-trivial
    /// eigenvalue.
    basis_values: Array2<f64>,
    /// `λ_1 ≤ … ≤ λ_q`, the non-trivial Laplacian eigenvalues. `diag(λ)` is
    /// literally `Φᵀ L_W Φ` — the Dirichlet form of the basis columns and the
    /// decode penalty are the SAME object.
    eigenvalues: Vec<f64>,
    /// Training-vertex embeddings `x_i`, shape `(anchors × r)`; the Nyström
    /// anchors the out-of-sample kernel weights against.
    anchor_embeddings: Array2<f64>,
    /// Gaussian affinity bandwidth `ε` — the median squared length of the
    /// surviving graph edges (the same median-neighbour-distance rule the
    /// Laplacian-eigenmap seed uses, `gam_geometry::latent_seed`).
    bandwidth: f64,
}

impl GraphSpectralBasis {
    /// Selected decode dimension `q`.
    pub fn selected_q(&self) -> usize {
        self.eigenvalues.len()
    }

    /// The kept non-trivial Laplacian eigenvalues `λ_1 ≤ … ≤ λ_q`.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Nyström Gaussian bandwidth `ε`.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The decode penalty in the spectral basis: `diag(λ)`, shape `(q × q)`.
    ///
    /// This is not a second computation of the roughness — it is exactly the
    /// atom's Dirichlet form `Φᵀ L_W Φ` read in the eigenbasis, where `L_W` is
    /// the *same* survived weighted Laplacian [`LearnedGraphAtom::surviving_laplacian`]
    /// whose Kronecker lift `L_W ⊗ I_r` is the atom's smoothness penalty
    /// [`LearnedGraphAtom::surviving_penalty_op`]. Diagonalising `L_W` on its
    /// own eigenvectors returns `diag(λ)`, so the penalty a decode consumer
    /// reads here and the penalty the graph atom prices are one operator.
    pub fn penalty(&self) -> Array2<f64> {
        let q = self.selected_q();
        let mut penalty = Array2::<f64>::zeros((q, q));
        for k in 0..q {
            penalty[[k, k]] = self.eigenvalues[k];
        }
        penalty
    }

    /// Batched Nyström extension of the decode basis to arbitrary out-of-sample
    /// rows `points` (`n × r`).
    ///
    /// The coordinate is the affinity-weighted (Nadaraya–Watson / geometric-
    /// harmonics) average of the training eigenvectors,
    ///
    /// ```text
    ///   φ_k(z) = N_k(z) / S(z),
    ///   N_k(z) = Σ_i w_i(z) Φ[i, k],   S(z) = Σ_i w_i(z),
    ///   w_i(z) = exp(−‖z − x_i‖² / ε),
    /// ```
    ///
    /// with `ε` the graph's edge-length bandwidth [`Self::bandwidth`]. At a
    /// training vertex the Gaussian mass concentrates on that vertex, so
    /// `φ_k(x_j) ≈ Φ[j, k]`, and between vertices it interpolates smoothly — the
    /// standard Nyström / geometric-harmonics extension of a graph eigenmap.
    ///
    /// The jet is analytic (no finite differences). With
    /// `∂w_i/∂z_c = w_i · (−2 (z_c − x_{i,c}) / ε)`,
    ///
    /// ```text
    ///   ∂φ_k/∂z_c = ( (∂N_k/∂z_c)·S − N_k·(∂S/∂z_c) ) / S²,
    ///   ∂N_k/∂z_c = Σ_i (∂w_i/∂z_c) Φ[i, k],   ∂S/∂z_c = Σ_i ∂w_i/∂z_c.
    /// ```
    ///
    /// Returns `(φ, ∂φ/∂z)` with shapes `(n × q)` and `(n × q × r)`.
    pub fn nystrom_coordinates(
        &self,
        points: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array3<f64>), String> {
        let anchors = self.anchor_embeddings.nrows();
        let r = self.anchor_embeddings.ncols();
        let q = self.selected_q();
        if points.ncols() != r {
            return Err(format!(
                "GraphSpectralBasis::nystrom_coordinates: query has {} features but graph anchors have {r}",
                points.ncols()
            ));
        }
        if points.iter().any(|v| !v.is_finite()) {
            return Err(
                "GraphSpectralBasis::nystrom_coordinates: query contains a non-finite value".into(),
            );
        }
        let n = points.nrows();
        let eps = self.bandwidth;
        if !(eps > 0.0 && eps.is_finite()) {
            return Err(format!(
                "GraphSpectralBasis::nystrom_coordinates: non-positive bandwidth {eps}"
            ));
        }
        let mut phi = Array2::<f64>::zeros((n, q));
        let mut jet = Array3::<f64>::zeros((n, q, r));
        let mut w = vec![0.0_f64; anchors];
        let mut n_k = vec![0.0_f64; q];
        let mut dw = vec![0.0_f64; anchors];
        for row in 0..n {
            // Affinities and the normaliser S(z).
            let mut s = 0.0_f64;
            for i in 0..anchors {
                let mut d2 = 0.0_f64;
                for c in 0..r {
                    let d = points[[row, c]] - self.anchor_embeddings[[i, c]];
                    d2 += d * d;
                }
                let wi = (-d2 / eps).exp();
                w[i] = wi;
                s += wi;
            }
            if !(s > 0.0 && s.is_finite()) {
                return Err(
                    "GraphSpectralBasis::nystrom_coordinates: query point has vanishing affinity \
                     to every anchor (bandwidth underflow)"
                        .into(),
                );
            }
            // Numerators N_k and the decode coordinate φ_k = N_k / S.
            for k in 0..q {
                let mut acc = 0.0_f64;
                for i in 0..anchors {
                    acc += w[i] * self.basis_values[[i, k]];
                }
                n_k[k] = acc;
                phi[[row, k]] = acc / s;
            }
            // Analytic jet, one ambient channel c at a time.
            for c in 0..r {
                let mut ds_c = 0.0_f64;
                for i in 0..anchors {
                    let dwi =
                        w[i] * (-2.0 * (points[[row, c]] - self.anchor_embeddings[[i, c]]) / eps);
                    dw[i] = dwi;
                    ds_c += dwi;
                }
                for k in 0..q {
                    let mut dn_kc = 0.0_f64;
                    for i in 0..anchors {
                        dn_kc += dw[i] * self.basis_values[[i, k]];
                    }
                    jet[[row, k, c]] = (dn_kc * s - n_k[k] * ds_c) / (s * s);
                }
            }
        }
        Ok((phi, jet))
    }

    /// The Nyström extension as a first-class [`SaeBasisEvaluator`], so the
    /// decode presents the exact {basis values, jet} interface the typed atoms'
    /// evaluators do. Its input coordinates are the ambient row embeddings `z`
    /// (`r` features), its output the `q`-dimensional decode coordinate.
    pub fn evaluator(&self) -> Arc<dyn SaeBasisEvaluator> {
        Arc::new(NystromSpectralEvaluator {
            basis: self.clone(),
        })
    }
}

/// [`SaeBasisEvaluator`] adapter over a [`GraphSpectralBasis`]: `evaluate`
/// returns the Nyström decode coordinate `φ(z)` and its analytic first jet
/// `∂φ/∂z` at each queried ambient row `z`. It declares no analytic second /
/// third jet (`None`): the spectral decode's roughness is the graph Dirichlet
/// form `diag(λ)` carried directly on [`GraphSpectralBasis::penalty`], not a
/// second-jet curvature Gram, so no consumer needs a Nyström Hessian and the
/// honest capability declaration is absence.
#[derive(Debug, Clone)]
pub struct NystromSpectralEvaluator {
    basis: GraphSpectralBasis,
}

impl SaeBasisEvaluator for NystromSpectralEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        self.basis.nystrom_coordinates(coords)
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        // A mismatched query width is a caller bug, not a missing capability:
        // surface it as an error exactly like `nystrom_coordinates` would,
        // instead of a silent "no jet" that sends the caller down a fallback.
        let r = self.basis.anchor_embeddings.ncols();
        if coords.ncols() != r {
            return Some(Err(format!(
                "NystromSpectralEvaluator::second_jet_dyn: query has {} features but graph anchors have {r}",
                coords.ncols()
            )));
        }
        None
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Option<Result<ndarray::Array5<f64>, String>> {
        let r = self.basis.anchor_embeddings.ncols();
        if coords.ncols() != r {
            return Some(Err(format!(
                "NystromSpectralEvaluator::third_jet_dyn: query has {} features but graph anchors have {r}",
                coords.ncols()
            )));
        }
        None
    }
}

/// A spectral-graph decode candidate shaped for the birth topology race.
///
/// This mirrors the private `TopologyRaceFit` shape the typed candidates carry
/// in [`crate::structure_harvest`] — evaluator, basis kind, latent manifold,
/// decode design `Φ`, jet `∂Φ`, penalized decoder `B`, roughness penalty — but
/// for the open-world graph atom, whose penalty is the graph Dirichlet form
/// `diag(λ)` rather than a basis second-jet Gram.
///
/// # Where this plugs into the race
///
/// The single call site is
/// `crate::structure_harvest::topology_candidates_for_dim` (crate-internal,
/// concurrently edited elsewhere, so it is NOT touched here). After that
/// function builds the typed `TopologyCandidateSpec`s for the born atom's `d_k`,
/// a spectral candidate is appended when the born atom already carries a
/// [`LearnedGraphAtom`], by calling
/// [`LearnedGraphAtom::spectral_race_candidate`] with the birth target and the
/// per-row ambient embeddings; its `{phi, jet, penalty, rank_charge_dof}` feed
/// the SAME `TopologyAutoFitEvidence` inputs `fit_topology_candidate` produces —
/// with `penalty` supplied directly instead of re-derived from a second jet —
/// and its `evaluator` seeds the born atom's out-of-sample decode. Everything
/// up to that append (basis, penalty, decoder, jet, charge, evaluator) is
/// implemented here; only the one `specs.push(...)` line lives across the seam.
pub struct SpectralGraphRaceCandidate {
    /// Basis-kind tag; a precomputed decode basis with no closed-form typed
    /// evaluator family.
    pub basis_kind: SaeAtomBasisKind,
    /// The flat `q`-coordinate decode chart the atom carries.
    pub manifold: LatentManifold,
    /// Decode dimension `q`.
    pub latent_dim: usize,
    /// The ambient row embeddings `z` (`n × r`) the Nyström evaluator reads.
    pub row_coords: Array2<f64>,
    /// Decode design `Φ(z)` at the rows (`n × q`).
    pub phi: Array2<f64>,
    /// Decode design jet `∂Φ/∂z` (`n × q × r`).
    pub jet: Array3<f64>,
    /// Dirichlet-penalized decoder `B` (`q × p`) at the REML-optimal `λ̂` on the
    /// spectral penalty — fit through the SAME closed-form entry point the typed
    /// candidates use, so the decode is priced commensurably.
    pub decoder: Array2<f64>,
    /// The decode penalty `diag(λ)` (`q × q`) — the graph Dirichlet form.
    pub penalty: Array2<f64>,
    /// The Nyström out-of-sample decode map.
    pub evaluator: Arc<dyn SaeBasisEvaluator>,
    /// The decode rank charge `0.5·q·ln(n_eff)`.
    pub rank_charge_dof: f64,
}

impl LearnedGraphAtom {

}
