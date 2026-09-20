//! Cellular-sheaf consistency penalty.
//!
//! Given a directed graph `G = (V, E)` with per-vertex stalk vectors
//! `s_v ∈ R^{d_v}` and per-edge linear restriction maps
//! `R_e^{(u→e)}: R^{d_u} → R^{d_e}`, the **coboundary**
//!
//! ```text
//! δs[e] = R_e^{(u→e)}(s_{u_e}) − R_e^{(v→e)}(s_{v_e})
//! ```
//!
//! lifts each edge into a per-edge "discrepancy" vector. The
//! **sheaf Laplacian** `L = δᵀ δ` is sparse PSD on the stacked stalk
//! space `R^{Σ d_v}`. Globally consistent sections live in `ker L`;
//! `dim ker L` ("number of harmonic modes") generalises the
//! connected-component count of a graph Laplacian to sheaves.
//!
//! The penalty value is
//!
//! ```text
//! P(s) = ½ · weight · sᵀ L s = ½ · weight · ∑_e ‖δs[e]‖².
//! ```
//!
//! References:
//!   * Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves",
//!     J. Appl. Comput. Topol. 3 (2019).
//!   * Bodnar, Di Giovanni, Chamberlain, Lió, Bronstein,
//!     "Neural Sheaf Diffusion" (NeurIPS 2022).
//!
//! Design choices in this module:
//!   * The Laplacian is **never materialised**. All operations route through
//!     two matvecs (`δ` and `δᵀ`).
//!   * Restriction maps are `(R_uv, Option<R_vu>)` pairs. If the second is
//!     `None` it defaults to the identity (`δs[e] = R_uv·s_u − s_v`), which
//!     is the "single-restriction edge" convention common in sheaf-diffusion
//!     networks.
//!   * `harmonic_modes()` counts exactly at every size: `L` is block
//!     diagonal over the connected components of the edge graph, and each
//!     component block is counted by faer's self-adjoint eigendecomposition
//!     (`gam_linalg::faer_ndarray::FaerEigh`) under a memory-ledger charge.

use faer::Side;
use ndarray::{Array1, Array2, ArrayView1};

use crate::analytic_penalties::{AnalyticPenalty, PenaltyTier};
use gam_linalg::faer_ndarray::FaerEigh;

/// A single edge's pair of restriction operators.
///
/// * `r_uv` maps the tail stalk into the edge stalk.
/// * `r_vu` maps the head stalk into the edge stalk; `None` means "identity"
///   (which forces `d_e == d_v`).
#[derive(Debug, Clone)]
pub struct EdgeRestriction {
    pub r_uv: Array2<f64>,
    pub r_vu: Option<Array2<f64>>,
}

impl EdgeRestriction {
    /// Both endpoints have an explicit restriction map.
    #[must_use]
    pub fn paired(r_uv: Array2<f64>, r_vu: Array2<f64>) -> Self {
        Self {
            r_uv,
            r_vu: Some(r_vu),
        }
    }

    /// Tail-only restriction; the head side is implicitly identity.
    #[must_use]
    pub fn single(r_uv: Array2<f64>) -> Self {
        Self { r_uv, r_vu: None }
    }

    /// Output (edge-stalk) dimension `d_e` for this edge.
    pub(crate) fn edge_dim(&self) -> usize {
        self.r_uv.nrows()
    }
}

/// Cellular-sheaf consistency penalty over a fixed directed graph + restriction
/// maps. The stacked stalk space layout is row-major over vertices:
/// vertex `v` occupies `stalk_offsets[v] .. stalk_offsets[v] + stalk_dims[v]`.
#[derive(Debug, Clone)]
pub struct SheafConsistencyPenalty {
    edges: Vec<(usize, usize)>,
    restrictions: Vec<EdgeRestriction>,
    weight: f64,
    stalk_offsets: Vec<usize>,
    stalk_dims: Vec<usize>,
}

impl SheafConsistencyPenalty {
    /// Construct a sheaf-consistency penalty.
    ///
    /// * `edges` — directed edges as `(u, v)` pairs, vertex indices `0..K`.
    /// * `restrictions` — same length as `edges`; per-edge `EdgeRestriction`.
    /// * `weight` — finite, positive scalar penalty weight.
    /// * `stalk_dims` — per-vertex stalk dimensions `d_v`.
    ///
    /// Validates: dim agreement (`r_uv.ncols == d_u`, `r_vu.ncols == d_v`,
    /// `r_uv.nrows == r_vu.nrows`), vertex indices in range, finite entries.
    #[must_use = "build error must be handled"]
    pub fn new(
        edges: Vec<(usize, usize)>,
        restrictions: Vec<EdgeRestriction>,
        weight: f64,
        stalk_dims: Vec<usize>,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "SheafConsistencyPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if edges.len() != restrictions.len() {
            return Err(format!(
                "SheafConsistencyPenalty::new edge count {} != restriction count {}",
                edges.len(),
                restrictions.len()
            ));
        }
        if stalk_dims.is_empty() {
            return Err("SheafConsistencyPenalty::new requires at least one vertex".into());
        }
        for (v, &d) in stalk_dims.iter().enumerate() {
            if d == 0 {
                return Err(format!(
                    "SheafConsistencyPenalty::new stalk dim at vertex {v} is zero"
                ));
            }
        }
        for (e, ((u, v), restriction)) in edges.iter().zip(restrictions.iter()).enumerate() {
            if *u >= stalk_dims.len() || *v >= stalk_dims.len() {
                return Err(format!(
                    "SheafConsistencyPenalty::new edge {e} = ({u}, {v}) references vertex \
                     out of range (K = {})",
                    stalk_dims.len()
                ));
            }
            let d_u = stalk_dims[*u];
            let d_v = stalk_dims[*v];
            let d_e = restriction.r_uv.nrows();
            if restriction.r_uv.ncols() != d_u {
                return Err(format!(
                    "SheafConsistencyPenalty::new edge {e}: r_uv has {} cols, expected d_u = {d_u}",
                    restriction.r_uv.ncols()
                ));
            }
            match &restriction.r_vu {
                Some(r_vu) => {
                    if r_vu.ncols() != d_v {
                        return Err(format!(
                            "SheafConsistencyPenalty::new edge {e}: r_vu has {} cols, \
                             expected d_v = {d_v}",
                            r_vu.ncols()
                        ));
                    }
                    if r_vu.nrows() != d_e {
                        return Err(format!(
                            "SheafConsistencyPenalty::new edge {e}: r_vu has {} rows, \
                             expected d_e = {d_e}",
                            r_vu.nrows()
                        ));
                    }
                }
                None => {
                    if d_e != d_v {
                        return Err(format!(
                            "SheafConsistencyPenalty::new edge {e}: r_vu is identity but \
                             d_e ({d_e}) != d_v ({d_v})"
                        ));
                    }
                }
            }
            if !restriction.r_uv.iter().all(|x| x.is_finite()) {
                return Err(format!(
                    "SheafConsistencyPenalty::new edge {e}: r_uv contains non-finite entries"
                ));
            }
            if let Some(r_vu) = &restriction.r_vu
                && !r_vu.iter().all(|x| x.is_finite())
            {
                return Err(format!(
                    "SheafConsistencyPenalty::new edge {e}: r_vu contains non-finite entries"
                ));
            }
        }
        let mut stalk_offsets = Vec::with_capacity(stalk_dims.len() + 1);
        let mut acc = 0usize;
        for &d in &stalk_dims {
            stalk_offsets.push(acc);
            acc = acc.checked_add(d).ok_or_else(|| {
                "SheafConsistencyPenalty::new stalk offsets overflow usize".to_string()
            })?;
        }
        stalk_offsets.push(acc);
        Ok(Self {
            edges,
            restrictions,
            weight,
            stalk_offsets,
            stalk_dims,
        })
    }

    /// Total dimension of the stacked stalk space `Σ d_v`.
    pub fn total_dim(&self) -> usize {
        *self.stalk_offsets.last().expect("offsets non-empty")
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of vertices `K`.
    pub fn num_vertices(&self) -> usize {
        self.stalk_dims.len()
    }

    /// Per-vertex stalk dimensions (clone of internal vector).
    pub fn stalk_dims(&self) -> &[usize] {
        &self.stalk_dims
    }

    /// Penalty weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    fn vertex_slice<'a>(&self, s: ArrayView1<'a, f64>, v: usize) -> ArrayView1<'a, f64> {
        let start = self.stalk_offsets[v];
        let end = self.stalk_offsets[v + 1];
        s.slice_move(ndarray::s![start..end])
    }

    /// Apply `δ` to a stacked-stalk vector `s`. Returns a `Vec<Array1<f64>>`
    /// with one entry per edge containing `δs[e] ∈ R^{d_e}`.
    fn delta(&self, s: ArrayView1<'_, f64>) -> Vec<Array1<f64>> {
        assert_eq!(
            s.len(),
            self.total_dim(),
            "stacked stalk vector has wrong length",
        );
        let mut out = Vec::with_capacity(self.edges.len());
        for (e, &(u, v)) in self.edges.iter().enumerate() {
            let s_u = self.vertex_slice(s, u);
            let s_v = self.vertex_slice(s, v);
            let restriction = &self.restrictions[e];
            // R_uv · s_u
            let mut delta_e = restriction.r_uv.dot(&s_u);
            // − R_vu · s_v   (identity if r_vu is None)
            match &restriction.r_vu {
                Some(r_vu) => {
                    let r_vu_s_v = r_vu.dot(&s_v);
                    delta_e.scaled_add(-1.0, &r_vu_s_v);
                }
                None => {
                    delta_e.scaled_add(-1.0, &s_v);
                }
            }
            out.push(delta_e);
        }
        out
    }

    /// Apply `δᵀ` to per-edge discrepancies `y`. Returns the stacked-stalk
    /// vector `δᵀ y ∈ R^{Σ d_v}`.
    fn delta_transpose(&self, y: &[Array1<f64>]) -> Array1<f64> {
        assert_eq!(
            y.len(),
            self.edges.len(),
            "delta_transpose edge count mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.total_dim());
        for (e, &(u, v)) in self.edges.iter().enumerate() {
            let restriction = &self.restrictions[e];
            let y_e = &y[e];
            assert_eq!(y_e.len(), restriction.edge_dim(), "edge dim mismatch");
            // R_uvᵀ · y_e → vertex u
            let contrib_u = restriction.r_uv.t().dot(y_e);
            let u_start = self.stalk_offsets[u];
            let u_end = self.stalk_offsets[u + 1];
            {
                let mut out_u = out.slice_mut(ndarray::s![u_start..u_end]);
                out_u.scaled_add(1.0, &contrib_u);
            }
            // −R_vuᵀ · y_e → vertex v   (identity if r_vu is None)
            let v_start = self.stalk_offsets[v];
            let v_end = self.stalk_offsets[v + 1];
            match &restriction.r_vu {
                Some(r_vu) => {
                    let contrib_v = r_vu.t().dot(y_e);
                    let mut out_v = out.slice_mut(ndarray::s![v_start..v_end]);
                    out_v.scaled_add(-1.0, &contrib_v);
                }
                None => {
                    let mut out_v = out.slice_mut(ndarray::s![v_start..v_end]);
                    out_v.scaled_add(-1.0, y_e);
                }
            }
        }
        out
    }

    /// Apply the sheaf Laplacian `L = δᵀ δ` to a stacked-stalk vector `s`.
    /// Cost: two matvecs per edge; never materialises `L`.
    pub(crate) fn laplacian_apply(&self, s: ArrayView1<'_, f64>) -> Array1<f64> {
        let ds = self.delta(s);
        self.delta_transpose(&ds)
    }

    /// Penalty value `½ · weight · ‖δs‖²`. Quadratic in `s`.
    pub fn value(&self, s: ArrayView1<'_, f64>) -> f64 {
        let ds = self.delta(s);
        let mut sq = 0.0;
        for de in &ds {
            for &x in de.iter() {
                sq += x * x;
            }
        }
        0.5 * self.weight * sq
    }

    /// Gradient `∂P/∂s = weight · L s`. Length `Σ d_v`.
    pub fn gradient(&self, s: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut g = self.laplacian_apply(s);
        g *= self.weight;
        g
    }

    /// Hessian diagonal `diag(weight · L)`. Independent of `s` because `L` is
    /// constant. For a **distinct-vertex** edge `(u, v)` (`u ≠ v`) the coboundary
    /// `C = [R_uv | −R_vu]` acts on disjoint stalk blocks, so
    ///   * tail (u-side): `Σ_j R_uv[j, i_local]²`
    ///   * head (v-side, single-restriction): `1.0` per incident edge
    ///   * head (v-side, paired): `Σ_j R_vu[j, i_local]²`
    /// For a **self-loop** edge `(u, u)` both sides share one block and the
    /// coboundary collapses to `(R_uv − R_vu)·s_u`, so the correct contribution
    /// is `colnorm²(R_uv − R_vu)` — NOT the sum of the two separate norms.
    pub fn hessian_diag(&self, s: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(
            s.len(),
            self.total_dim(),
            "stacked stalk vector has wrong length",
        );
        // L is constant in s; the argument is retained only for trait-style symmetry
        // (other penalties take target as the first arg). The shape assertion above
        // exercises that input.
        let mut diag = Array1::<f64>::zeros(self.total_dim());
        for (e, &(u, v)) in self.edges.iter().enumerate() {
            let restriction = &self.restrictions[e];
            let u_start = self.stalk_offsets[u];
            let v_start = self.stalk_offsets[v];
            let r_uv = &restriction.r_uv;

            if u == v {
                // Self-loop: δ(s)[e] = (R_uv − R_vu)·s_u, so the edge's
                // contribution to diag(L) is colnorm²(R_uv − R_vu), NOT the
                // sum of the two separate squared column norms (which would
                // double-count on the shared stalk block). The distinct-vertex
                // path below is not reached for self-loops.
                match &restriction.r_vu {
                    Some(r_vu) => {
                        for col in 0..r_uv.ncols() {
                            let mut s2 = 0.0;
                            for row in 0..r_uv.nrows() {
                                let diff = r_uv[[row, col]] - r_vu[[row, col]];
                                s2 += diff * diff;
                            }
                            diag[u_start + col] += s2;
                        }
                    }
                    None => {
                        // r_vu = I; contribution is colnorm²(R_uv − I).
                        let d = self.stalk_dims[u];
                        for col in 0..d {
                            let mut s2 = 0.0;
                            for row in 0..r_uv.nrows() {
                                let identity_entry = if row == col { 1.0 } else { 0.0 };
                                let diff = r_uv[[row, col]] - identity_entry;
                                s2 += diff * diff;
                            }
                            diag[u_start + col] += s2;
                        }
                    }
                }
            } else {
                // Distinct-vertex path: u_start ≠ v_start, so u-side and v-side
                // accumulations land on disjoint index ranges. Diagonal of Cᵀ C
                // with C = [R_uv | −R_vu] decomposes cleanly into the two blocks.
                for col in 0..r_uv.ncols() {
                    let mut s2 = 0.0;
                    for row in 0..r_uv.nrows() {
                        let a = r_uv[[row, col]];
                        s2 += a * a;
                    }
                    diag[u_start + col] += s2;
                }
                match &restriction.r_vu {
                    Some(r_vu) => {
                        for col in 0..r_vu.ncols() {
                            let mut s2 = 0.0;
                            for row in 0..r_vu.nrows() {
                                let a = r_vu[[row, col]];
                                s2 += a * a;
                            }
                            diag[v_start + col] += s2;
                        }
                    }
                    None => {
                        let d_v = self.stalk_dims[v];
                        for col in 0..d_v {
                            diag[v_start + col] += 1.0;
                        }
                    }
                }
            }
        }
        diag *= self.weight;
        diag
    }

    /// Hessian-vector product `H v = weight · L v`. Two matvecs, no
    /// materialisation. The `_s` argument is unused (L is constant); it
    /// matches the trait-style `(target, v)` signature other penalties use.
    pub fn hvp(&self, s: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(
            s.len(),
            self.total_dim(),
            "stacked stalk vector has wrong length",
        );
        assert_eq!(v.len(), self.total_dim(), "hvp direction has wrong length");
        let mut hv = self.laplacian_apply(v);
        hv *= self.weight;
        hv
    }

    /// The vertex sets of the connected components of the edge graph, each in
    /// increasing vertex order. No edge joins two components, so `L` is block
    /// diagonal over them up to a permutation.
    fn components(&self) -> Vec<Vec<usize>> {
        fn find(parent: &mut [usize], mut vertex: usize) -> usize {
            while parent[vertex] != vertex {
                parent[vertex] = parent[parent[vertex]];
                vertex = parent[vertex];
            }
            vertex
        }
        let k = self.stalk_dims.len();
        let mut parent: Vec<usize> = (0..k).collect();
        for &(u, v) in &self.edges {
            let root_u = find(&mut parent, u);
            let root_v = find(&mut parent, v);
            if root_u != root_v {
                parent[root_u.max(root_v)] = root_u.min(root_v);
            }
        }
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for vertex in 0..k {
            let root = find(&mut parent, vertex);
            members[root].push(vertex);
        }
        members.into_iter().filter(|vertices| !vertices.is_empty()).collect()
    }

    /// The block of `L = δᵀδ` over one component's edges, in that component's
    /// stacked-stalk coordinates (`local_offset[v]` is vertex `v`'s first
    /// coordinate inside its component).
    ///
    /// `δs[e] = R_uv s_u − R_vu s_v`, so each edge adds `R_uvᵀR_uv` to the `u`
    /// block, `R_vuᵀR_vu` to the `v` block, and `−R_uvᵀR_vu` and its transpose to
    /// the two off-diagonal blocks. For a self-loop all four land in one block and
    /// sum to `(R_uv − R_vu)ᵀ(R_uv − R_vu)`, matching `laplacian_apply`.
    fn component_laplacian(&self, edges: &[usize], local_offset: &[usize], dim: usize) -> Array2<f64> {
        let mut block = Array2::<f64>::zeros((dim, dim));
        for &edge in edges {
            let (u, v) = self.edges[edge];
            let restriction = &self.restrictions[edge];
            let tail = &restriction.r_uv;
            let head_identity;
            let head: &Array2<f64> = match &restriction.r_vu {
                Some(r_vu) => r_vu,
                None => {
                    head_identity = Array2::<f64>::eye(self.stalk_dims[v]);
                    &head_identity
                }
            };
            let (u_start, d_u) = (local_offset[u], self.stalk_dims[u]);
            let (v_start, d_v) = (local_offset[v], self.stalk_dims[v]);
            let tail_tail = tail.t().dot(tail);
            let head_head = head.t().dot(head);
            let tail_head = tail.t().dot(head);
            {
                let mut uu =
                    block.slice_mut(ndarray::s![u_start..u_start + d_u, u_start..u_start + d_u]);
                uu += &tail_tail;
            }
            {
                let mut vv =
                    block.slice_mut(ndarray::s![v_start..v_start + d_v, v_start..v_start + d_v]);
                vv += &head_head;
            }
            {
                let mut uv =
                    block.slice_mut(ndarray::s![u_start..u_start + d_u, v_start..v_start + d_v]);
                uv -= &tail_head;
            }
            {
                let mut vu =
                    block.slice_mut(ndarray::s![v_start..v_start + d_v, u_start..u_start + d_u]);
                vu -= &tail_head.t();
            }
        }
        block
    }

    /// Number of harmonic modes of the unweighted Laplacian `L`: the dimension of
    /// its null space, the global sections of the sheaf. The penalty weight is
    /// **not** folded in: harmonic modes are an intrinsic property of `δ`.
    ///
    /// `L` is block diagonal over the connected components of the edge graph, so
    /// the count is the sum over the component blocks, and it is exact at every
    /// total stalk dimension. A block's null eigenvalues are counted within its
    /// eigendecomposition's backward-error band `dim·ε·λ_max`: a backward-stable
    /// self-adjoint eigensolver moves every eigenvalue by at most that much
    /// (Weyl), so an exact zero is counted and a resolved positive eigenvalue is
    /// not. An edge-free component has a zero block, so all of its coordinates
    /// are harmonic. Each dense block and its eigenvectors are charged on the
    /// memory ledger, and a component too large for the ledger returns an error
    /// instead of a count.
    pub fn harmonic_modes(&self) -> Result<usize, String> {
        let components = self.components();
        let k = self.stalk_dims.len();
        let mut component_of = vec![0usize; k];
        let mut local_offset = vec![0usize; k];
        let mut component_dims = Vec::with_capacity(components.len());
        for (index, vertices) in components.iter().enumerate() {
            let mut dim = 0usize;
            for &vertex in vertices {
                component_of[vertex] = index;
                local_offset[vertex] = dim;
                dim += self.stalk_dims[vertex];
            }
            component_dims.push(dim);
        }
        let mut component_edges: Vec<Vec<usize>> = vec![Vec::new(); components.len()];
        for (edge, &(u, _)) in self.edges.iter().enumerate() {
            component_edges[component_of[u]].push(edge);
        }
        let governor = gam_runtime::resource::MemoryGovernor::global();
        let mut count = 0usize;
        for (index, &dim) in component_dims.iter().enumerate() {
            if component_edges[index].is_empty() {
                count += dim;
                continue;
            }
            let reservation = governor
                .try_reserve_dense_f64_copies(
                    dim,
                    dim,
                    2,
                    "SheafConsistencyPenalty::harmonic_modes component Laplacian",
                )
                .map_err(|error| {
                    format!(
                        "SheafConsistencyPenalty::harmonic_modes: refusing a {dim}x{dim} component \
                         Laplacian: {error}"
                    )
                })?;
            let block = self.component_laplacian(&component_edges[index], &local_offset, dim);
            let (eigenvalues, _) = block.eigh(Side::Lower).map_err(|error| {
                format!(
                    "SheafConsistencyPenalty::harmonic_modes: eigendecomposition of a {dim}x{dim} \
                     component Laplacian failed: {error:?}"
                )
            })?;
            let max_abs = eigenvalues
                .iter()
                .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
            let band = dim as f64 * f64::EPSILON * max_abs;
            count += eigenvalues.iter().filter(|&&value| value <= band).count();
            drop(reservation);
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// AnalyticPenalty trait bridge.
// ---------------------------------------------------------------------------
//
// Wires `SheafConsistencyPenalty` into the analytic-penalty registry so it is
// reachable from REML / PIRLS / CLI callers exactly like ARDPenalty,
// BlockOrthogonalityPenalty, etc. `target` is the stacked-stalk vector
// (treated as a ψ-tier flat block); `rho` is unused — this penalty is
// quadratic with a fixed scalar weight set at construction. The
// `harmonic_modes` query and the per-vertex layout helpers remain available
// as inherent methods for callers that want the cellular-sheaf-specific
// diagnostics.

impl AnalyticPenalty for SheafConsistencyPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        assert!(
            rho.iter().all(|x| x.is_finite()),
            "SheafConsistencyPenalty: rho must be finite (got {rho:?})",
        );
        SheafConsistencyPenalty::value(self, target)
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        assert!(
            rho.iter().all(|x| x.is_finite()),
            "SheafConsistencyPenalty: rho must be finite (got {rho:?})",
        );
        SheafConsistencyPenalty::gradient(self, target)
    }

    // `hessian_diag` keeps the trait default `None`: the Hessian `weight·L`
    // couples every edge's endpoints, so it is not diagonal. That `None` is
    // also what routes `psd_majorizer_hvp` to the exact Laplacian `hvp`,
    // which is the PSD majorizer of this convex quadratic; a `Some(diag(L))`
    // would make the trait default apply `diag(L) ⊙ v` and drop every edge
    // coupling, and `diag(L)` is not `⪰ L` on any edge.
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert!(
            rho.iter().all(|x| x.is_finite()),
            "SheafConsistencyPenalty: rho must be finite (got {rho:?})",
        );
        SheafConsistencyPenalty::hvp(self, target, v)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // No learnable hyperparameter axes: rho_count == 0.
        assert_eq!(
            rho.len(),
            0,
            "SheafConsistencyPenalty: rho_count is 0 but rho has length {}",
            rho.len(),
        );
        assert_eq!(
            target.len(),
            self.total_dim(),
            "SheafConsistencyPenalty: target length {} != total stalk dim {}",
            target.len(),
            self.total_dim(),
        );
        Array1::<f64>::zeros(0)
    }

    fn rho_count(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "SheafConsistencyPenalty"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn identity(d: usize) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            m[[i, i]] = 1.0;
        }
        m
    }

    /// The dense Laplacian `L` (no weight) from `total_dim` matvecs against the
    /// standard basis: the matvec oracle for the assembled blocks and counts.
    fn dense_laplacian(pen: &SheafConsistencyPenalty) -> Array2<f64> {
        let n = pen.total_dim();
        let mut l = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            let col = pen.laplacian_apply(e.view());
            for i in 0..n {
                l[[i, j]] = col[i];
            }
            e[j] = 0.0;
        }
        l
    }

    /// Dense-eigh count of the matvec Laplacian's null eigenvalues, within the
    /// dense eigendecomposition's backward-error band `dim·ε·λ_max`.
    fn dense_null_count(pen: &SheafConsistencyPenalty) -> usize {
        let (evals, _) = dense_laplacian(pen)
            .eigh(Side::Lower)
            .expect("dense Laplacian eigendecomposition");
        let max_abs = evals.iter().fold(0.0_f64, |acc, &value| acc.max(value.abs()));
        let band = pen.total_dim() as f64 * f64::EPSILON * max_abs;
        evals.iter().filter(|&&value| value <= band).count()
    }

    // #2900: the component blocks must reassemble the matvec Laplacian exactly on a
    // connected sheaf mixing paired, single-restriction and self-loop edges.
    #[test]
    fn component_laplacian_matches_the_matvec_laplacian_2900() {
        let edges = vec![(0usize, 1usize), (1usize, 2usize), (2usize, 2usize)];
        let restrictions = vec![
            EdgeRestriction::paired(array![[0.9_f64, 0.1], [-0.2, 0.7]], identity(2)),
            EdgeRestriction::single(array![[0.5_f64, -0.3], [0.4, 0.8]]),
            EdgeRestriction::paired(array![[0.6_f64, 0.2], [0.1, 1.1]], array![[0.3_f64, 0.0], [0.5, 0.2]]),
        ];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![2, 2, 2]).expect("build");
        let local_offset = vec![0usize, 2, 4];
        let block = pen.component_laplacian(&[0, 1, 2], &local_offset, pen.total_dim());
        let reference = dense_laplacian(&pen);
        for i in 0..pen.total_dim() {
            for j in 0..pen.total_dim() {
                assert_abs_diff_eq!(block[[i, j]], reference[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // #2900: on a sheaf with several components (one with a self-loop, one edge-free
    // vertex, one rank-deficient restriction), the component count must equal the
    // dense null count of the whole matvec Laplacian. The smallest resolved
    // eigenvalue sits many orders above both bands, so neither eigensolver's
    // roundoff can move a mode across one.
    #[test]
    fn harmonic_modes_matches_the_dense_count_across_components_2900() {
        let edges = vec![(0usize, 1usize), (1usize, 1usize), (3usize, 4usize)];
        let restrictions = vec![
            EdgeRestriction::paired(array![[1.0_f64, 0.0], [0.0, 0.0]], identity(2)),
            EdgeRestriction::paired(array![[0.4_f64, 0.3], [0.2, 0.9]], array![[0.1_f64, 0.0], [0.0, 0.2]]),
            EdgeRestriction::single(array![[0.8_f64, 0.6], [-0.6, 0.8]]),
        ];
        let pen = SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![2, 2, 3, 2, 2])
            .expect("build");
        let (evals, _) = dense_laplacian(&pen).eigh(Side::Lower).expect("eigh");
        let top = evals.iter().copied().fold(0.0_f64, f64::max);
        let band = pen.total_dim() as f64 * f64::EPSILON * top;
        let smallest_resolved = evals
            .iter()
            .copied()
            .filter(|&value| value > band)
            .fold(f64::INFINITY, f64::min);
        assert!(
            smallest_resolved > 1.0e6 * band,
            "the fixture must have a resolved gap above the null band: {smallest_resolved:e} vs {band:e}"
        );
        assert_eq!(
            pen.harmonic_modes().expect("harmonic modes"),
            dense_null_count(&pen)
        );
    }

    // The harmonic-mode count is a property of the sheaf's kernel, so scaling
    // every restriction by a constant must not move it. With restrictions
    // 2^-17·I, each identity pair's block is 2^-34·[[1, -1], [-1, 1]], with
    // eigenvalues {0, 2^-33 ≈ 1.2e-10}. The retired default tolerance 1e-8
    // counted both as harmonic. The scale is a power of two, so the scaled
    // block's rounding is the unscaled block's, bit for bit.
    #[test]
    fn harmonic_mode_count_does_not_depend_on_the_restriction_scale() {
        let pairs = 64usize;
        for scale in [1.0_f64, 2.0_f64.powi(-17)] {
            let edges: Vec<(usize, usize)> = (0..pairs).map(|p| (2 * p, 2 * p + 1)).collect();
            let restrictions =
                std::iter::repeat_with(|| EdgeRestriction::paired(identity(1) * scale, identity(1) * scale))
                    .take(pairs)
                    .collect();
            let pen = SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![1; 2 * pairs])
                .expect("build");
            assert_eq!(
                pen.harmonic_modes().expect("harmonic modes"),
                pairs,
                "restriction scale {scale:e}"
            );
        }
    }

    // #2900: 2100 disconnected identity-restricted pairs with one-dimensional stalks
    // (total dimension 4200, above the retired 4096-dimension Lanczos switch) have
    // exactly one harmonic mode per pair. A single-vector Lanczos sees that
    // eigenvalue only once, so it could never report 2100.
    #[test]
    fn harmonic_modes_counts_multiplicity_at_every_size_2900() {
        let pairs = 2100usize;
        let edges: Vec<(usize, usize)> = (0..pairs).map(|p| (2 * p, 2 * p + 1)).collect();
        let restrictions = (0..pairs)
            .map(|_| EdgeRestriction::paired(identity(1), identity(1)))
            .collect();
        let pen = SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![1; 2 * pairs])
            .expect("build");
        assert_eq!(pen.total_dim(), 4200);
        assert_eq!(pen.harmonic_modes().expect("harmonic modes"), pairs);
    }

    #[test]
    fn single_edge_identity_restriction_value() {
        // K=2, d_0 = d_1 = 3, R_uv = R_vu = I.
        // s_0 = (1,0,0), s_1 = (0,1,0). δs = (1,-1,0). ‖·‖² = 2. Value = ½·1·2 = 1.
        let edges = vec![(0usize, 1usize)];
        let restrictions = vec![EdgeRestriction::paired(identity(3), identity(3))];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![3, 3]).expect("build");
        let s = array![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = pen.value(s.view());
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn gradient_matches_finite_difference_k2_random() {
        // K=2 with arbitrary restrictions; FD-check the gradient.
        let r_uv = array![[0.7_f64, -0.1, 0.3], [0.2, 0.9, -0.4]];
        let r_vu = array![[1.0_f64, 0.5], [-0.3, 0.8]];
        let edges = vec![(0usize, 1usize)];
        let restrictions = vec![EdgeRestriction::paired(r_uv, r_vu)];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 0.3, vec![3, 2]).expect("build");
        let s = array![0.4_f64, -1.1, 0.2, 0.6, -0.7];
        let g = pen.gradient(s.view());
        let eps = 1e-7;
        for i in 0..s.len() {
            let mut sp = s.clone();
            let mut sm = s.clone();
            sp[i] += eps;
            sm[i] -= eps;
            let fd = (pen.value(sp.view()) - pen.value(sm.view())) / (2.0 * eps);
            assert_abs_diff_eq!(g[i], fd, epsilon = 1e-6);
        }
    }

    #[test]
    fn hvp_matches_reconstructed_laplacian_chain_k3() {
        // K=3 chain: edges (0,1) and (1,2), each with explicit 2x2 restrictions.
        // d_0 = d_1 = d_2 = 2.
        let r01_uv = array![[0.9_f64, 0.1], [-0.2, 0.7]];
        let r01_vu = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r12_uv = array![[0.5_f64, -0.3], [0.4, 0.8]];
        let r12_vu = array![[0.6_f64, 0.0], [0.1, 1.1]];
        let edges = vec![(0usize, 1usize), (1usize, 2usize)];
        let restrictions = vec![
            EdgeRestriction::paired(r01_uv, r01_vu),
            EdgeRestriction::paired(r12_uv, r12_vu),
        ];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![2, 2, 2]).expect("build");
        // Reconstruct L densely via 6 matvecs.
        let l_dense = dense_laplacian(&pen);
        let n = pen.total_dim();
        let s = array![0.1_f64, -0.2, 0.3, 0.4, -0.5, 0.6];
        let v = array![0.7_f64, 0.2, -0.1, 0.5, 0.3, -0.4];
        let hv = pen.hvp(s.view(), v.view());
        // Reference: L · v (weight = 1)
        let mut lv = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += l_dense[[i, j]] * v[j];
            }
            lv[i] = acc;
        }
        for i in 0..n {
            assert_abs_diff_eq!(hv[i], lv[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn harmonic_modes_two_components_identity_restrictions() {
        // Two disconnected vertices (no edges), d = 3 each → ker L = R^{6}, all 6 modes.
        let pen = SheafConsistencyPenalty::new(vec![], vec![], 1.0, vec![3, 3]).expect("build");
        let h = pen.harmonic_modes().expect("harmonic modes");
        assert_eq!(h, 6);

        // K=4, two connected components: (0-1) and (2-3) with identity restrictions, d=2 each.
        // Each component's sheaf-Laplacian kernel has dim d (the "constant sections").
        // Total ker dim = 2·d = 4.
        let edges = vec![(0usize, 1usize), (2usize, 3usize)];
        let restrictions = vec![
            EdgeRestriction::paired(identity(2), identity(2)),
            EdgeRestriction::paired(identity(2), identity(2)),
        ];
        let pen2 = SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![2, 2, 2, 2])
            .expect("build");
        let h2 = pen2.harmonic_modes().expect("harmonic modes");
        assert_eq!(h2, 4);
    }

    #[test]
    fn value_psd_and_zero_iff_kernel() {
        // Random s on a non-trivial sheaf: value ≥ 0.
        let r01_uv = array![[0.9_f64, 0.1], [-0.2, 0.7]];
        let r01_vu = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let edges = vec![(0usize, 1usize)];
        let restrictions = vec![EdgeRestriction::paired(r01_uv.clone(), r01_vu.clone())];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 0.5, vec![2, 2]).expect("build");

        // Several random-ish stalks: non-negative value.
        let samples = [
            array![0.0_f64, 0.0, 0.0, 0.0],
            array![1.0_f64, 2.0, -0.5, 0.3],
            array![-1.3_f64, 0.7, 0.2, -0.9],
        ];
        for s in &samples {
            let v = pen.value(s.view());
            assert!(v >= 0.0, "value must be non-negative, got {v}");
        }
        // Zero stalk → zero value.
        let z = Array1::<f64>::zeros(4);
        assert_abs_diff_eq!(pen.value(z.view()), 0.0, epsilon = 1e-15);
        // A kernel direction: pick s_0 arbitrary then set s_1 = r_vu⁻¹ · r_uv · s_0.
        // r_vu = I, so s_1 = r_uv · s_0.
        let s0 = array![0.3_f64, -1.1];
        let s1 = r01_uv.dot(&s0);
        let mut s = Array1::<f64>::zeros(4);
        s[0] = s0[0];
        s[1] = s0[1];
        s[2] = s1[0];
        s[3] = s1[1];
        assert_abs_diff_eq!(pen.value(s.view()), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn hessian_diag_matches_diag_of_dense_laplacian() {
        let r_uv = array![[0.7_f64, -0.1, 0.3], [0.2, 0.9, -0.4]];
        let r_vu = array![[1.0_f64, 0.5], [-0.3, 0.8]];
        let edges = vec![(0usize, 1usize)];
        let restrictions = vec![EdgeRestriction::paired(r_uv, r_vu)];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 0.3, vec![3, 2]).expect("build");
        let n = pen.total_dim();
        let s = Array1::<f64>::zeros(n);
        let diag = pen.hessian_diag(s.view());
        let l = dense_laplacian(&pen);
        for i in 0..n {
            assert_abs_diff_eq!(diag[i], 0.3 * l[[i, i]], epsilon = 1e-12);
        }
    }

    #[test]
    fn hessian_diag_matches_dense_laplacian_on_self_loop_paired() {
        // Self-loop (0,0) with two distinct paired restrictions: the diagonal
        // must equal diag(weight·L) built from the matrix-free operator, i.e.
        // colnorm²(R_uv − R_vu), NOT colnorm²(R_uv) + colnorm²(R_vu).
        let r_uv = array![[0.9_f64, 0.1], [-0.2, 0.7]];
        let r_vu = array![[1.0_f64, 0.5], [-0.3, 0.8]];
        let edges = vec![(0usize, 0usize)];
        let restrictions = vec![EdgeRestriction::paired(r_uv, r_vu)];
        let pen = SheafConsistencyPenalty::new(edges, restrictions, 0.7, vec![2]).expect("build");
        let n = pen.total_dim();
        let s = Array1::<f64>::zeros(n);
        let diag = pen.hessian_diag(s.view());
        let l = dense_laplacian(&pen);
        for i in 0..n {
            assert_abs_diff_eq!(diag[i], 0.7 * l[[i, i]], epsilon = 1e-12);
        }
    }

    #[test]
    fn hessian_diag_matches_dense_laplacian_on_self_loop_single() {
        // Self-loop (0,0) with a single-restriction edge: R_vu is the identity,
        // so the coboundary is (R_uv − I)·s_0. The drop-cross-term bug would
        // report colnorm²(R_uv) + 1 per column; the correct diagonal is
        // colnorm²(R_uv − I). d_e == d_v == d_u = 2 (single-edge requirement).
        // This single-restriction self-loop path is exercised by neither the
        // committed repro (paired only) nor the landing fix's tests.
        let r_uv = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let edges = vec![(0usize, 0usize)];
        let restrictions = vec![EdgeRestriction::single(r_uv)];
        let pen = SheafConsistencyPenalty::new(edges, restrictions, 1.3, vec![2]).expect("build");
        let n = pen.total_dim();
        let s = Array1::<f64>::zeros(n);
        let diag = pen.hessian_diag(s.view());
        let l = dense_laplacian(&pen);
        for i in 0..n {
            assert_abs_diff_eq!(diag[i], 1.3 * l[[i, i]], epsilon = 1e-12);
        }
        // Spot the closed form: C = R_uv − I = [[0,2],[3,3]].
        // col 0: 0² + 3² = 9; col 1: 2² + 3² = 13. ×weight 1.3 → [11.7, 16.9].
        assert_abs_diff_eq!(diag[0], 1.3 * 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 1.3 * 13.0, epsilon = 1e-12);
    }

    #[test]
    fn hessian_diag_matches_dense_laplacian_mixed_self_loop_and_cross_edge() {
        // A self-loop on vertex 0 AND a distinct edge (0,1) both touch vertex 0.
        // The two edges' diagonal contributions must accumulate independently:
        // the self-loop contributes colnorm²(R0 − R0b) on block 0, while the
        // cross edge contributes colnorm²(R1_uv) on block 0 and colnorm²(R1_vu)
        // on block 1. Checked against the operator-built dense Laplacian.
        let r0_uv = array![[0.5_f64, -0.4], [0.3, 0.9]];
        let r0_vu = array![[0.2_f64, 0.1], [-0.6, 0.7]];
        let r1_uv = array![[1.1_f64, 0.2], [0.0, -0.5]];
        let r1_vu = array![[0.8_f64, -0.1], [0.4, 1.0]];
        let edges = vec![(0usize, 0usize), (0usize, 1usize)];
        let restrictions = vec![
            EdgeRestriction::paired(r0_uv, r0_vu),
            EdgeRestriction::paired(r1_uv, r1_vu),
        ];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 0.5, vec![2, 2]).expect("build");
        let n = pen.total_dim();
        let s = Array1::<f64>::zeros(n);
        let diag = pen.hessian_diag(s.view());
        let l = dense_laplacian(&pen);
        for i in 0..n {
            assert_abs_diff_eq!(diag[i], 0.5 * l[[i, i]], epsilon = 1e-12);
        }
    }

    #[test]
    fn single_restriction_edge_form() {
        // δs = R·s_0 − s_1 (single-restriction form). d_0 = 2, d_e = d_1 = 2.
        let r = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let edges = vec![(0usize, 1usize)];
        let restrictions = vec![EdgeRestriction::single(r.clone())];
        let pen =
            SheafConsistencyPenalty::new(edges, restrictions, 2.0, vec![2, 2]).expect("build");
        // s_0 = (1, 0) → R·s_0 = (1, 3). s_1 = (1, 3) → δs = 0. Value = 0.
        let s = array![1.0_f64, 0.0, 1.0, 3.0];
        assert_abs_diff_eq!(pen.value(s.view()), 0.0, epsilon = 1e-12);
        // Now break consistency: s_1 = (0, 0). δs = (1, 3). Value = ½·2·(1+9) = 10.
        let s2 = array![1.0_f64, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(pen.value(s2.view()), 10.0, epsilon = 1e-12);
    }

    // The penalty is a convex quadratic, so its PSD majorizer must be the exact
    // Hessian `weight·L`, not its diagonal. `diag(L)` is not a majorizer of `L`:
    // on the single edge below, `diag(L) − L` has a negative eigenvalue.
    #[test]
    fn sheaf_psd_majorizer_is_the_exact_weighted_laplacian() {
        let edges = vec![(0usize, 1usize), (1usize, 2usize), (2usize, 2usize)];
        let restrictions = vec![
            EdgeRestriction::paired(array![[0.9_f64, 0.1], [-0.2, 0.7]], identity(2)),
            EdgeRestriction::single(array![[0.5_f64, -0.3], [0.4, 0.8]]),
            EdgeRestriction::paired(
                array![[0.6_f64, 0.2], [0.1, 1.1]],
                array![[0.3_f64, 0.0], [0.5, 0.2]],
            ),
        ];
        let weight = 1.7_f64;
        let pen = SheafConsistencyPenalty::new(edges, restrictions, weight, vec![2, 2, 2])
            .expect("build");
        let n = pen.total_dim();
        let s = Array1::from_shape_fn(n, |i| 0.3 * i as f64 - 0.5);
        let rho = Array1::<f64>::zeros(0);
        assert!(
            AnalyticPenalty::hessian_diag(&pen, s.view(), rho.view()).is_none(),
            "a non-diagonal Hessian has no diagonal to report"
        );
        assert!(
            AnalyticPenalty::psd_majorizer_diag(&pen, s.view(), rho.view()).is_none(),
            "a non-diagonal Hessian has no diagonal majorizer to report"
        );
        let l = dense_laplacian(&pen);
        let mut b = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            let col = pen.psd_majorizer_hvp(s.view(), rho.view(), e.view());
            let exact = AnalyticPenalty::hvp(&pen, s.view(), rho.view(), e.view());
            for i in 0..n {
                b[[i, j]] = col[i];
                assert_abs_diff_eq!(col[i], exact[i], epsilon = 1e-12);
                assert_abs_diff_eq!(col[i], weight * l[[i, j]], epsilon = 1e-12);
            }
            e[j] = 0.0;
        }
        // The diagonal the trait default used to apply is not a majorizer: the
        // matrix `diag(B) − B` must have a clearly negative eigenvalue here.
        let mut gap = -b.clone();
        for i in 0..n {
            gap[[i, i]] += b[[i, i]];
        }
        let (gap_evals, _) = gap.eigh(Side::Lower).expect("eigh");
        let min_gap = gap_evals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            min_gap < -0.1,
            "diag(weight·L) − weight·L should be indefinite, min eigenvalue {min_gap}"
        );
    }
}
