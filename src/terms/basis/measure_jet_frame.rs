//! Measure-jet frame synthesis function space.
//!
//! The current representer basis (one Gaussian RBF per center) is
//! *value-coupled*: its columns reach across the web with a confident
//! parametric backbone, so it cannot represent an ambient-affine function
//! without paying an `O(τ)` toll and it does not give a clean per-scale
//! diagonal prior. This module realizes the frame function space described in
//! `docs/measure_jet_frame.md` §1: the per-level `ε/2`-nets that
//! `measure_jet_smooth::assemble_weighted_forms` already constructs are PROMOTED
//! from outer-quadrature points to the index set of a multiscale frame.
//!
//! The synthesis operator `S` maps a coefficient vector laid out as
//! `[ head | innovations ]` to nodal values on the centers (the web):
//!
//! * The **head** is an unpenalized global polynomial block of degree `< r`
//!   (`r = 2` ⇒ `{1, x_1, …, x_d}`, the ambient-affine space). It passes
//!   ambient-affine functions through EXACTLY at any `τ`, structurally killing
//!   the `O(τ)` affine toll — this is the property the Gaussian-representer basis
//!   lacked, and oracle (1) pins it.
//!
//! * Each **innovation** lives at one `(level ℓ, net node a)`: a Gaussian bump
//!   at scale `ε_ℓ` centered at the net node, modulated by the local jet
//!   monomials `{1, frame coords}` with frame coordinate `(x − c_a)/ε_ℓ`. The
//!   coarse-to-fine PREDICTION (lifting) step projects each level's monomial
//!   action onto the span of the head plus all coarser levels and keeps only the
//!   residual, so every innovation column is orthogonal — in the mass inner
//!   product `Sᵀ M_μ S` — to the polynomial head and to every coarser
//!   innovation. That is vanishing moments by construction (oracle (2)): a degree `< r` polynomial
//!   evaluated on the web is reproduced entirely by the head and leaves the
//!   innovations untouched.
//!
//! The prior is DIAGONAL by coordinates: the head is unpenalized, each level's
//! innovations are an independent block with precision `λ_ℓ = ε_ℓ^{-2s}`
//! (prior variance `ε_ℓ^{2s}` up to the global amplitude — the multiscale per-scale-candidate mode
//! made structural). The whitened synthesis `S D^{1/2}` therefore has a
//! value-space frame ratio
//! `A · value_energy ≤ Σ_ℓ ε_ℓ^{−2s} ‖d_ℓ‖² ≤ B · value_energy`; this module
//! estimates `A`, `B` at runtime via a few power iterations on the whitened
//! value Gram (oracle (4) checks that `B/A` is finite and at least one on a
//! quasi-uniform net), so every fit can ship its own frame-ratio certificate.
//!
//! Everything here is deterministic (no RNG): the nets are greedy on a fixed
//! index order, the power iterations start from a fixed all-ones seed, and the
//! lifting is a sequence of pseudo-inverse solves with a frozen rank tolerance.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use faer::Side;

use super::BasisError;
use crate::faer_ndarray::FaerEigh;

/// Default jet-frame order: `r = 2` ⇒ the head is `{1, x_1, …, x_d}`, the
/// ambient-affine space (charter §6, "r = 2 default"). Degree `< r` polynomials
/// pass through the head exactly.
pub const MEASURE_JET_FRAME_DEFAULT_ORDER_R: usize = 2;

/// `ε/2`-net radius factor: a node claims points within `ε/2`, matching the
/// outer-net coarsening `net_radius² = 0.25 · ε²` used by
/// `assemble_weighted_forms` (charter §1: the nets are reused verbatim).
pub(crate) const MEASURE_JET_FRAME_NET_RADIUS_FACTOR: f64 = 0.5;

/// Relative rank tolerance for the lifting / whitening pseudo-inverse solves,
/// in units of the largest singular value. The frozen value makes the
/// coarse-to-fine prediction reproducible across builds.
pub(crate) const MEASURE_JET_FRAME_RANK_RTOL: f64 = 1e-10;

/// Power-iteration budget for the frame-ratio certificate. Five iterations on a
/// well-clustered whitened Gram resolve the extreme Rayleigh quotients to the
/// precision the certificate budgets; deterministic from a fixed seed.
pub(crate) const MEASURE_JET_FRAME_POWER_ITERS: usize = 64;

/// A coarse-to-fine ascending scale band's worth of innovation atoms, plus the
/// unpenalized polynomial head, realized as a single synthesis operator on the
/// web (the seed centers).
///
/// Coefficients are laid out as `[ head | level_0 | level_1 | … ]`; the prior is
/// diagonal: the `head_dim` head coordinates are unpenalized, and the
/// `level_dims[ℓ]` coordinates of level `ℓ` share the precision `λ_ℓ = ε_ℓ^{-2s}`
/// (stored as `level_precisions[ℓ]`).
pub struct MeasureJetFrame {
    /// Number of web nodes (rows of `S`): the seed centers.
    pub(crate) n_nodes: usize,
    /// Web-node quadrature masses defining the value inner product.
    pub(crate) masses: Array1<f64>,
    /// Unpenalized polynomial-head width (`= number of degree < r monomials`).
    pub(crate) head_dim: usize,
    /// Per-level innovation widths, ascending in scale.
    pub(crate) level_dims: Vec<usize>,
    /// Per-level prior precision `λ_ℓ = ε_ℓ^{-2s}` (the diagonal whitening).
    pub(crate) level_precisions: Vec<f64>,
    /// The dense synthesis matrix `S` (`n_nodes × total_dim`): `value = S · coef`.
    /// Column `0 .. head_dim` is the head; the remainder are the lifted, value-
    /// orthogonalized innovation columns in `[level_0 | level_1 | …]` order.
    pub(crate) synthesis: Array2<f64>,
    /// Realized ascending scale band `ε_ℓ` (one per level).
    pub(crate) eps_band: Vec<f64>,
    /// Smoothness order `s` baked into `level_precisions`.
    pub(crate) order_s: f64,
}

/// Runtime value-space frame-ratio certificate: estimates of the equivalence
/// constants `A ≤ coefficient_energy / value_energy ≤ B`, where
/// `value_energy = zᵀ(SD^{1/2})ᵀM_μ(SD^{1/2})z`. `ratio = B / A ≥ 1` is the
/// condition number; finiteness witnesses that the innovations form a Riesz
/// frame for their span under the mass value inner product.
#[derive(Clone, Copy, Debug)]
pub struct MeasureJetFrameRatioCertificate {
    /// Lower frame constant `A = 1/λ_max` for coefficient-energy / value-energy.
    pub lower_a: f64,
    /// Upper frame constant `B = 1/λ_min` for coefficient-energy / value-energy.
    pub upper_b: f64,
    /// Condition number `B / A ≥ 1`; non-finite iff the frame is degenerate.
    pub ratio: f64,
    /// Number of power-iteration sweeps actually run (for the audit trail).
    pub iterations: usize,
}

impl MeasureJetFrame {
    /// Total coefficient width `head_dim + Σ_ℓ level_dims[ℓ]`.
    pub fn total_dim(&self) -> usize {
        self.head_dim + self.level_dims.iter().sum::<usize>()
    }

    /// Number of web nodes (rows of the synthesis operator).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Unpenalized polynomial-head width.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Per-level innovation widths (ascending in scale).
    pub fn level_dims(&self) -> &[usize] {
        &self.level_dims
    }

    /// Per-level diagonal prior precision `λ_ℓ = ε_ℓ^{-2s}`.
    pub fn level_precisions(&self) -> &[f64] {
        &self.level_precisions
    }

    /// Realized ascending scale band.
    pub fn eps_band(&self) -> &[f64] {
        &self.eps_band
    }

    /// Smoothness order baked into the prior.
    pub fn order_s(&self) -> f64 {
        self.order_s
    }

    /// The dense synthesis matrix (`n_nodes × total_dim`).
    pub fn synthesis_matrix(&self) -> ArrayView2<'_, f64> {
        self.synthesis.view()
    }

    /// Synthesis `S`: map innovation+head coefficients to nodal web values.
    /// `coef.len()` must equal [`total_dim`](Self::total_dim).
    pub fn synthesize(&self, coef: ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError> {
        if coef.len() != self.total_dim() {
            crate::bail_dim_basis!(
                "measure-jet frame synthesis expects {} coefficients, got {}",
                self.total_dim(),
                coef.len()
            );
        }
        Ok(self.synthesis.dot(&coef))
    }

    /// Analysis `Sᵀ`: map nodal web values to the coefficient dual.
    /// `values.len()` must equal [`n_nodes`](Self::n_nodes). This is the EXACT
    /// f64 adjoint of [`synthesize`](Self::synthesize) under the mass value
    /// inner product `uᵀM_μv` (oracle (3)).
    pub fn analyze(&self, values: ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError> {
        if values.len() != self.n_nodes {
            crate::bail_dim_basis!(
                "measure-jet frame analysis expects {} nodal values, got {}",
                self.n_nodes,
                values.len()
            );
        }
        let weighted = &values * &self.masses;
        Ok(self.synthesis.t().dot(&weighted))
    }

    /// The diagonal whitening vector `D^{1/2}` over the coefficient layout: the
    /// head block is `1` (unpenalized) and each level `ℓ` block is
    /// `λ_ℓ^{−1/2} = ε_ℓ^{s}` (so that whitened coefficients have unit prior
    /// variance, and the whitened energy is `Σ_ℓ ε_ℓ^{−2s} ‖d_ℓ‖²` of §1).
    pub fn whitening_sqrt(&self) -> Array1<f64> {
        let mut w = Array1::<f64>::ones(self.total_dim());
        let mut off = self.head_dim;
        for (&dim, &lambda) in self.level_dims.iter().zip(self.level_precisions.iter()) {
            let inv_sqrt = 1.0 / lambda.sqrt();
            for j in off..off + dim {
                w[j] = inv_sqrt;
            }
            off += dim;
        }
        w
    }

    /// Estimate the value-space frame-ratio certificate
    /// `A · value_energy ≤ Σ_ℓ ε_ℓ^{−2s} ‖d_ℓ‖² ≤ B · value_energy` by a few deterministic power iterations on the whitened
    /// value-Gram `M = (S D^{1/2})ᵀ M_μ (S D^{1/2})` restricted to the innovation
    /// block (the head is unpenalized and excluded from the frame energy). Since
    /// the documented quotient is coefficient energy divided by value energy,
    /// `A = 1/λ_max(M)` and `B = 1/λ_min(M)`.
    pub fn frame_ratio_certificate(&self) -> MeasureJetFrameRatioCertificate {
        let p = self.total_dim();
        let n_innov = p - self.head_dim;
        if n_innov == 0 {
            // Degenerate-but-valid: a pure-head frame trivially satisfies the
            // equivalence with A = B = 1 (no innovation energy to bound).
            return MeasureJetFrameRatioCertificate {
                lower_a: 1.0,
                upper_b: 1.0,
                ratio: 1.0,
                iterations: 0,
            };
        }
        // Whitened synthesis restricted to the innovation columns.
        let w = self.whitening_sqrt();
        let mut s_white = Array2::<f64>::zeros((self.n_nodes, n_innov));
        for j in 0..n_innov {
            let col = self.head_dim + j;
            let scale = w[col];
            for i in 0..self.n_nodes {
                s_white[(i, j)] = self.synthesis[(i, col)] * scale;
            }
        }
        // Gram of the whitened innovation block: M = Sᵀ M_μ S (n_innov × n_innov).
        let mut mass_s_white = s_white.clone();
        for (i, mut row) in mass_s_white.outer_iter_mut().enumerate() {
            row.mapv_inplace(|v| v * self.masses[i]);
        }
        let m = s_white.t().dot(&mass_s_white);
        let top = power_top_eigenvalue(&m, MEASURE_JET_FRAME_POWER_ITERS);
        let bottom = power_bottom_eigenvalue(&m, top, MEASURE_JET_FRAME_POWER_ITERS);
        let lower_a = if top > 0.0 { 1.0 / top } else { f64::INFINITY };
        let upper_b = if bottom > 0.0 {
            1.0 / bottom
        } else {
            f64::INFINITY
        };
        let ratio = if lower_a.is_finite() && lower_a > 0.0 {
            upper_b / lower_a
        } else {
            f64::INFINITY
        };
        MeasureJetFrameRatioCertificate {
            lower_a,
            upper_b,
            ratio,
            iterations: MEASURE_JET_FRAME_POWER_ITERS,
        }
    }
}

/// Build the jet-frame synthesis space on `centers` with masses `masses` over
/// the ascending scale band `eps_band`, smoothness order `order_s`, and head
/// order `r` (degree `< r` polynomial head). The synthesis operator's value
/// domain is the web (the `centers`).
///
/// Construction (charter §1):
///   1. Head = degree `< r` global monomials evaluated on the centers, then
///      `QR`-orthonormalized in the mass value inner product (so the head columns are
///      a mass-orthonormal basis of the ambient-affine span — exact
///      pass-through).
///   2. Per level (coarse→fine): build the `ε_ℓ/2`-net on the centers (greedy,
///      deterministic), form one atom per net node × local jet monomial
///      `{1, (x−c)/ε_ℓ}` as a Gaussian-bump-weighted column on the web, then
///      LIFT: subtract the value-projection onto the head plus all already-built
///      (coarser) innovation columns, and drop columns whose residual norm falls
///      below the rank tolerance. The survivors are the level-`ℓ` innovations.
///   3. Prior precision `λ_ℓ = ε_ℓ^{-2s}` per level.
pub fn build_measure_jet_frame(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    eps_band: &[f64],
    order_s: f64,
    order_r: usize,
) -> Result<MeasureJetFrame, BasisError> {
    let n = centers.nrows();
    let d = centers.ncols();
    if n == 0 || d == 0 {
        crate::bail_invalid_basis!("measure-jet frame needs nonempty centers");
    }
    if masses.len() != n {
        crate::bail_dim_basis!(
            "measure-jet frame needs one mass per center: {} masses, {} centers",
            masses.len(),
            n
        );
    }
    if eps_band.is_empty() {
        crate::bail_invalid_basis!("measure-jet frame needs a nonempty scale band");
    }
    for (l, pair) in eps_band.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            crate::bail_invalid_basis!(
                "measure-jet frame band must be strictly ascending: eps[{l}] = {} vs eps[{}] = {}",
                pair[0],
                l + 1,
                pair[1]
            );
        }
    }
    if eps_band.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet frame band scales must be finite and positive");
    }
    if !(order_s.is_finite() && order_s > 0.0) {
        crate::bail_invalid_basis!("measure-jet frame order_s must be finite and positive");
    }
    if order_r == 0 {
        crate::bail_invalid_basis!("measure-jet frame order_r must be at least 1");
    }
    if centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("measure-jet frame centers must be finite");
    }
    if masses.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        crate::bail_invalid_basis!("measure-jet frame masses must be finite and nonnegative");
    }

    // ---- 1. Unpenalized polynomial head: degree < r monomials on the web ----
    let head_exponents = enumerate_monomials_below_degree(d, order_r);
    let head_raw = monomial_design(centers, &head_exponents);
    // Value-orthonormalize the head via the symmetric value Gram: head columns
    // become an orthonormal basis of the ambient-affine value span, so a degree
    // < r polynomial is reproduced by the head and leaves the innovations zero.
    let head = orthonormalize_columns(&head_raw, masses, MEASURE_JET_FRAME_RANK_RTOL)?;
    let head_dim = head.ncols();

    // The growing pool of mass-orthonormal columns: head first, then the lifted
    // innovations as they are constructed level by level.
    let mut ortho_pool: Vec<Array1<f64>> =
        (0..head_dim).map(|j| head.column(j).to_owned()).collect();
    let mut innovation_cols: Vec<Array1<f64>> = Vec::new();
    let mut level_dims: Vec<usize> = Vec::with_capacity(eps_band.len());
    let mut level_precisions: Vec<f64> = Vec::with_capacity(eps_band.len());

    // ---- 2. Coarse-to-fine innovations with lifting ----
    // Iterate coarsest (largest ε) to finest (smallest ε): the lifting predicts
    // fine detail from already-built coarse detail, so the finest level carries
    // only the highest-frequency innovation. eps_band is ascending, so reverse.
    for (rev_idx, &eps) in eps_band.iter().enumerate().rev() {
        let net = build_eps_half_net(centers, eps);
        let mut level_count = 0usize;
        for &node in net.iter() {
            let atom_block = jet_atom_columns(centers, node, eps, d);
            for raw in atom_block.into_iter() {
                // Lift: subtract the value-projection onto the orthonormal pool
                // (head + coarser innovations). The residual is the innovation;
                // vanishing moments hold because the head is in the pool.
                let mut resid = raw;
                for basis_col in ortho_pool.iter() {
                    let proj = mass_dot(basis_col, &resid, masses);
                    resid.scaled_add(-proj, basis_col);
                }
                let norm = mass_dot(&resid, &resid, masses).sqrt();
                if norm > MEASURE_JET_FRAME_RANK_RTOL {
                    resid /= norm;
                    ortho_pool.push(resid.clone());
                    innovation_cols.push(resid);
                    level_count += 1;
                }
            }
        }
        // Recorded in reverse (coarse→fine); reversed back to band order below.
        level_dims.push(level_count);
        level_precisions.push(eps.powf(-2.0 * order_s));
    }
    // level_dims / level_precisions were filled coarse→fine (reverse of band).
    // Re-order to ascending-scale (band) order so the public layout matches
    // eps_band and the per-level prior precisions line up with eps_band[ℓ].
    level_dims.reverse();
    level_precisions.reverse();

    // ---- Assemble the synthesis matrix [ head | innovations(ascending) ] ----
    // innovation_cols were appended coarse→fine; rebuild them in ascending-scale
    // order so column blocks match level_dims (band order).
    let mut total_dim = head_dim;
    for &dim in &level_dims {
        total_dim += dim;
    }
    let mut synthesis = Array2::<f64>::zeros((n, total_dim));
    for j in 0..head_dim {
        let mut col = synthesis.column_mut(j);
        col.assign(&head.column(j));
    }
    // innovation_cols index k corresponds to coarse→fine order; map it to the
    // ascending-scale column position. Build a per-coarse-level offset table.
    let mut coarse_to_fine_dims: Vec<usize> = level_dims.clone();
    coarse_to_fine_dims.reverse(); // back to coarse→fine, matching innovation_cols
    let mut ascending_offsets = vec![head_dim; level_dims.len()];
    {
        let mut acc = head_dim;
        for (l, &dim) in level_dims.iter().enumerate() {
            ascending_offsets[l] = acc;
            acc += dim;
        }
    }
    // Walk innovation_cols in coarse→fine blocks; level (coarse index) c maps to
    // ascending level (eps_band.len() - 1 - c).
    {
        let mut k = 0usize;
        for (c, &dim) in coarse_to_fine_dims.iter().enumerate() {
            let ascending_level = level_dims.len() - 1 - c;
            let mut write = ascending_offsets[ascending_level];
            for _ in 0..dim {
                let mut col = synthesis.column_mut(write);
                col.assign(&innovation_cols[k]);
                k += 1;
                write += 1;
            }
        }
    }

    Ok(MeasureJetFrame {
        n_nodes: n,
        masses: masses.to_owned(),
        head_dim,
        level_dims,
        level_precisions,
        synthesis,
        eps_band: eps_band.to_vec(),
        order_s,
    })
}

/// Enumerate all monomial exponent multi-indices on `d` variables with total
/// degree strictly less than `order_r`, in deterministic ascending-degree,
/// lexicographic order. For `r = 2` this is `{(0,…,0)} ∪ {e_k}` — the constant
/// plus the `d` linear terms (the ambient-affine span).
pub(crate) fn enumerate_monomials_below_degree(d: usize, order_r: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let max_total = order_r - 1;
    // Degree-by-degree so the constant comes first, then linears, etc.
    for total in 0..=max_total {
        let mut current = vec![0usize; d];
        enumerate_fixed_total(d, total, 0, &mut current, &mut out);
    }
    out
}

/// Recursive helper: enumerate exponent vectors over coordinates `pos..d` whose
/// remaining-degree budget is exactly `remaining`, appending completed vectors.
pub(crate) fn enumerate_fixed_total(
    d: usize,
    remaining: usize,
    pos: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if pos == d - 1 {
        current[pos] = remaining;
        out.push(current.clone());
        current[pos] = 0;
        return;
    }
    for e in 0..=remaining {
        current[pos] = e;
        enumerate_fixed_total(d, remaining - e, pos + 1, current, out);
    }
    current[pos] = 0;
}

/// Evaluate the monomial design `[ ∏_k x_{i,k}^{α_k} ]` for each exponent vector
/// `α` at each center `x_i`. Returns an `n × |exponents|` matrix.
pub(crate) fn monomial_design(
    centers: ArrayView2<'_, f64>,
    exponents: &[Vec<usize>],
) -> Array2<f64> {
    let n = centers.nrows();
    let mut out = Array2::<f64>::zeros((n, exponents.len()));
    for (j, alpha) in exponents.iter().enumerate() {
        for i in 0..n {
            let mut v = 1.0;
            for (k, &e) in alpha.iter().enumerate() {
                if e > 0 {
                    v *= centers[(i, k)].powi(e as i32);
                }
            }
            out[(i, j)] = v;
        }
    }
    out
}

/// Build the `ε/2`-net on the centers: a greedy, deterministic farthest-skip
/// cover in fixed index order. A node claims every not-yet-claimed center within
/// radius `ε/2`; the survivors (nodes) index the level's atoms. Returns the node
/// indices into `centers` in ascending order.
pub(crate) fn build_eps_half_net(centers: ArrayView2<'_, f64>, eps: f64) -> Vec<usize> {
    let n = centers.nrows();
    let radius = MEASURE_JET_FRAME_NET_RADIUS_FACTOR * eps;
    let radius2 = radius * radius;
    let mut claimed = vec![false; n];
    let mut nodes: Vec<usize> = Vec::new();
    for i in 0..n {
        if claimed[i] {
            continue;
        }
        nodes.push(i);
        let ci = centers.row(i);
        for j in i..n {
            if claimed[j] {
                continue;
            }
            let cj = centers.row(j);
            let mut dd = 0.0;
            for k in 0..centers.ncols() {
                let diff = ci[k] - cj[k];
                dd += diff * diff;
            }
            if dd <= radius2 {
                claimed[j] = true;
            }
        }
    }
    nodes
}

/// Build the local jet-atom columns for a single net node at `node` and scale
/// `eps`: a pointwise Gaussian bump `w_i = exp(−‖x_i − c‖²/(2ε²))` modulated by
/// the local frame monomials `{1, (x_{i,k} − c_k)/ε}` for each coordinate `k`.
/// Returns `1 + d` raw value-columns over the web (one per local monomial); the
/// caller lifts them against the coarser pool.
///
/// The Gaussian-bump and frame-coordinate conventions match
/// `measure_jet_smooth::assemble_weighted_forms`: kernel `exp(−d²/(2ε²))`,
/// frame coordinate `(x − c)/ε`; masses enter the discrete inner product, not
/// these pointwise values.
pub(crate) fn jet_atom_columns(
    centers: ArrayView2<'_, f64>,
    node: usize,
    eps: f64,
    d: usize,
) -> Vec<Array1<f64>> {
    let n = centers.nrows();
    let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
    let inv_eps = 1.0 / eps;
    let c = centers.row(node);
    // Gaussian bump weight per web node.
    let mut bump = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = centers.row(i);
        let mut dd = 0.0;
        for k in 0..d {
            let diff = xi[k] - c[k];
            dd += diff * diff;
        }
        bump[i] = (-dd * inv_two_eps2).exp();
    }
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(1 + d);
    // Monomial {1}: the bump itself.
    cols.push(bump.clone());
    // Frame coordinates {(x_k − c_k)/ε}: bump × local linear coordinate.
    for k in 0..d {
        let mut col = Array1::<f64>::zeros(n);
        for i in 0..n {
            let frame = (centers[(i, k)] - c[k]) * inv_eps;
            col[i] = bump[i] * frame;
        }
        cols.push(col);
    }
    cols
}

/// Orthonormalize the columns of `a` in the mass value inner product via
/// modified Gram–Schmidt with a frozen rank tolerance; drop columns
/// whose post-orthogonalization norm falls below `rtol · max_norm`. Returns an
/// `n × rank` matrix with orthonormal columns spanning `col(a)`.
pub(crate) fn orthonormalize_columns(
    a: &Array2<f64>,
    masses: ArrayView1<'_, f64>,
    rtol: f64,
) -> Result<Array2<f64>, BasisError> {
    let n = a.nrows();
    let p = a.ncols();
    if p == 0 {
        crate::bail_invalid_basis!("measure-jet frame head must have at least one column");
    }
    // Reference scale: the largest raw column norm, for a relative drop test.
    let mut max_norm = 0.0_f64;
    for j in 0..p {
        let col = a.column(j).to_owned();
        let nj = mass_dot(&col, &col, masses).sqrt();
        max_norm = max_norm.max(nj);
    }
    let drop_below = (rtol * max_norm).max(f64::MIN_POSITIVE);
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for j in 0..p {
        let mut v = a.column(j).to_owned();
        for q in basis.iter() {
            let proj = mass_dot(q, &v, masses);
            v.scaled_add(-proj, q);
        }
        let norm = mass_dot(&v, &v, masses).sqrt();
        if norm > drop_below {
            v /= norm;
            basis.push(v);
        }
    }
    if basis.is_empty() {
        crate::bail_invalid_basis!("measure-jet frame head collapsed to rank zero");
    }
    let rank = basis.len();
    let mut out = Array2::<f64>::zeros((n, rank));
    for (j, col) in basis.into_iter().enumerate() {
        out.column_mut(j).assign(&col);
    }
    Ok(out)
}

#[inline]
pub(crate) fn mass_dot(u: &Array1<f64>, v: &Array1<f64>, masses: ArrayView1<'_, f64>) -> f64 {
    u.iter()
        .zip(v.iter())
        .zip(masses.iter())
        .map(|((&ui, &vi), &mi)| mi * ui * vi)
        .sum()
}

/// Top eigenvalue of a symmetric PSD matrix `m` by deterministic power
/// iteration from the all-ones seed. Falls back to the exact symmetric
/// eigensolver if the iteration stalls (tiny / degenerate spectra), so the
/// certificate is robust on small nets.
pub(crate) fn power_top_eigenvalue(m: &Array2<f64>, iters: usize) -> f64 {
    let p = m.nrows();
    if p == 0 {
        return 0.0;
    }
    let mut v = Array1::<f64>::ones(p);
    let mut norm = v.dot(&v).sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    v /= norm;
    let mut lambda = 0.0;
    for _ in 0..iters {
        let w = m.dot(&v);
        lambda = v.dot(&w);
        norm = w.dot(&w).sqrt();
        if !(norm > 0.0) {
            break;
        }
        v = w / norm;
    }
    // Exact backstop for ill-conditioned tiny problems.
    if let Ok((eigs, _)) = m.eigh(Side::Lower) {
        if let Some(max) = eigs.iter().cloned().fold(None, |acc: Option<f64>, e| {
            Some(acc.map_or(e, |a| a.max(e)))
        }) {
            return max.max(lambda).max(0.0);
        }
    }
    lambda.max(0.0)
}

/// Bottom eigenvalue of a symmetric PSD matrix `m` via shifted power iteration
/// on `top·I − m` (its top eigenvalue is `top − λ_min`). Seeded from all-ones;
/// backed by the exact symmetric eigensolver for robustness.
pub(crate) fn power_bottom_eigenvalue(m: &Array2<f64>, top: f64, iters: usize) -> f64 {
    let p = m.nrows();
    if p == 0 {
        return 0.0;
    }
    // Exact path first: the symmetric eigensolver is the source of truth for the
    // smallest eigenvalue on these modest innovation Grams.
    if let Ok((eigs, _)) = m.eigh(Side::Lower) {
        if let Some(min) = eigs.iter().cloned().fold(None, |acc: Option<f64>, e| {
            Some(acc.map_or(e, |a| a.min(e)))
        }) {
            return min.max(0.0);
        }
    }
    // Power-iteration fallback on the shifted operator.
    let mut shifted = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            shifted[(i, j)] = if i == j { top - m[(i, j)] } else { -m[(i, j)] };
        }
    }
    let shifted_top = power_top_eigenvalue(&shifted, iters);
    (top - shifted_top).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// A deterministic quasi-uniform grid of `side × side` centers in the unit
    /// square, with uniform masses. No RNG.
    pub(crate) fn grid_centers(side: usize) -> (Array2<f64>, Array1<f64>) {
        let n = side * side;
        let mut centers = Array2::<f64>::zeros((n, 2));
        let step = 1.0 / (side as f64 - 1.0).max(1.0);
        let mut idx = 0;
        for i in 0..side {
            for j in 0..side {
                centers[(idx, 0)] = i as f64 * step;
                centers[(idx, 1)] = j as f64 * step;
                idx += 1;
            }
        }
        let masses = Array1::<f64>::from_elem(n, 1.0 / n as f64);
        (centers, masses)
    }

    /// Three-level ascending band on the grid, deterministic.
    pub(crate) fn grid_band() -> Vec<f64> {
        vec![0.15, 0.30, 0.60]
    }

    /// Evaluate a degree < r polynomial on the centers (for affine = r=2 this is
    /// `a + b·x + c·y`).
    pub(crate) fn affine_values(
        centers: ArrayView2<'_, f64>,
        a: f64,
        b: f64,
        c: f64,
    ) -> Array1<f64> {
        let n = centers.nrows();
        let mut v = Array1::<f64>::zeros(n);
        for i in 0..n {
            v[i] = a + b * centers[(i, 0)] + c * centers[(i, 1)];
        }
        v
    }

    /// ORACLE (1): ambient-affine functions live in the EXACT null space of the
    /// innovations at ANY τ (here: at the default band): the innovation analysis
    /// of an affine field is ~0 (energy ≤ 1e-10 × the rough field energy), the
    /// property the Gaussian-representer basis lacked. The head reproduces the affine
    /// field exactly.
    #[test]
    pub(crate) fn affine_functions_in_exact_null_space() {
        let (centers, masses) = grid_centers(7);
        let band = grid_band();
        let frame = build_measure_jet_frame(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            MEASURE_JET_FRAME_DEFAULT_ORDER_R,
        )
        .expect("frame builds");

        let f = affine_values(centers.view(), 0.4, -1.3, 2.1);
        let rough = mass_dot(&f, &f, masses.view());
        // Analysis: the dual coefficients. The innovation block must vanish.
        let dual = frame.analyze(f.view()).expect("analyze");
        let head_dim = frame.head_dim();
        let mut innov_energy = 0.0;
        for j in head_dim..frame.total_dim() {
            innov_energy += dual[j] * dual[j];
        }
        assert!(
            innov_energy <= 1e-10 * rough.max(1.0),
            "affine field must leave innovations empty: innov_energy={innov_energy:.3e}, rough={rough:.3e}"
        );

        // The head alone reproduces the affine field exactly: project f onto the
        // head columns and resynthesize the head block; residual ~0.
        let mut head_coef = Array1::<f64>::zeros(frame.total_dim());
        for j in 0..head_dim {
            head_coef[j] = dual[j];
        }
        let recon = frame.synthesize(head_coef.view()).expect("synthesize");
        let mut resid = 0.0;
        for i in 0..centers.nrows() {
            let e = recon[i] - f[i];
            resid += e * e;
        }
        assert!(
            resid <= 1e-10 * rough.max(1.0),
            "head must reproduce the affine field exactly: resid={resid:.3e}"
        );
    }

    /// ORACLE (2): vanishing moments — the synthesis innovations annihilate
    /// degree < r polynomials at EVERY level. Equivalently, every innovation
    /// column is mass-orthogonal to the polynomial head, so `SᵀM_μ` of any affine
    /// field has a zero innovation block (checked here across a basis of the
    /// degree<r space, level by level).
    #[test]
    pub(crate) fn innovations_have_vanishing_moments() {
        let (centers, masses) = grid_centers(7);
        let band = grid_band();
        let frame = build_measure_jet_frame(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            MEASURE_JET_FRAME_DEFAULT_ORDER_R,
        )
        .expect("frame builds");
        let s = frame.synthesis_matrix();
        let head_dim = frame.head_dim();

        // A spanning set of the degree<r (affine) space.
        let affine_basis = [
            affine_values(centers.view(), 1.0, 0.0, 0.0),
            affine_values(centers.view(), 0.0, 1.0, 0.0),
            affine_values(centers.view(), 0.0, 0.0, 1.0),
        ];

        // Per level, every innovation column's inner product with every affine
        // field is ~0 (vanishing moments by construction).
        let mut col = head_dim;
        for (l, &dim) in frame.level_dims().iter().enumerate() {
            for _ in 0..dim {
                let column = s.column(col);
                for (bidx, basis_field) in affine_basis.iter().enumerate() {
                    let ip = mass_dot(&column.to_owned(), basis_field, masses.view());
                    assert!(
                        ip.abs() <= 1e-9,
                        "level {l} innovation column {col} couples to affine basis {bidx}: <c,p>={ip:.3e}"
                    );
                }
                col += 1;
            }
        }
    }

    /// ORACLE (3): `S` and `Sᵀ` are exact f64 adjoints: `⟨S x, y⟩ = ⟨x, Sᵀ y⟩`
    /// for deterministic `x` (coefficient space) and `y` (value space).
    #[test]
    pub(crate) fn synthesis_and_analysis_are_exact_adjoints() {
        let (centers, masses) = grid_centers(6);
        let band = grid_band();
        let frame = build_measure_jet_frame(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            MEASURE_JET_FRAME_DEFAULT_ORDER_R,
        )
        .expect("frame builds");

        let p = frame.total_dim();
        let n = frame.n_nodes();
        // Deterministic, no RNG: structured ramps.
        let x = Array1::<f64>::from_shape_fn(p, |i| ((i % 7) as f64) - 3.0 + 0.25 * i as f64);
        let y = Array1::<f64>::from_shape_fn(n, |i| 1.0 - 0.5 * (i as f64) + ((i % 5) as f64));

        let sx = frame.synthesize(x.view()).expect("synthesize");
        let sty = frame.analyze(y.view()).expect("analyze");
        let lhs = mass_dot(&sx, &y, masses.view());
        let rhs = x.dot(&sty);
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            (lhs - rhs).abs() <= 1e-10 * scale,
            "S and Sᵀ must be exact adjoints: <Sx,y>={lhs:.6e}, <x,Sᵀy>={rhs:.6e}"
        );
    }

    /// ORACLE (4): the frame ratio `B/A` is finite and `≥ 1` on a quasi-uniform
    /// net — the runtime equivalence certificate witnesses a genuine Riesz frame.
    #[test]
    pub(crate) fn frame_ratio_is_finite_and_at_least_one() {
        let (centers, masses) = grid_centers(7);
        let band = grid_band();
        let frame = build_measure_jet_frame(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            MEASURE_JET_FRAME_DEFAULT_ORDER_R,
        )
        .expect("frame builds");

        let cert = frame.frame_ratio_certificate();
        assert!(
            cert.lower_a > 0.0,
            "lower frame constant A must be positive on a quasi-uniform net: A={:.3e}",
            cert.lower_a
        );
        assert!(
            cert.upper_b.is_finite() && cert.upper_b >= cert.lower_a,
            "upper frame constant B must be finite and ≥ A: B={:.3e}, A={:.3e}",
            cert.upper_b,
            cert.lower_a
        );
        assert!(
            cert.ratio.is_finite() && cert.ratio >= 1.0 - 1e-12,
            "frame ratio B/A must be finite and ≥ 1: ratio={:.3e}",
            cert.ratio
        );
    }

    /// Guard: ambient-affine null space holds at the DEFAULT band too (the
    /// "any τ including the default" clause of oracle (1)) — re-checked with the
    /// auto order_s default so the property is not band-specific.
    #[test]
    pub(crate) fn affine_null_space_holds_at_default_order() {
        let (centers, masses) = grid_centers(8);
        let band = grid_band();
        let frame = build_measure_jet_frame(
            centers.view(),
            masses.view(),
            &band,
            // Default smoothness order s = 1.5 (charter default).
            1.5,
            MEASURE_JET_FRAME_DEFAULT_ORDER_R,
        )
        .expect("frame builds");
        let f = affine_values(centers.view(), -2.0, 3.5, -0.75);
        let rough = mass_dot(&f, &f, masses.view());
        let dual = frame.analyze(f.view()).expect("analyze");
        let mut innov_energy = 0.0;
        for j in frame.head_dim()..frame.total_dim() {
            innov_energy += dual[j] * dual[j];
        }
        assert!(
            innov_energy <= 1e-10 * rough.max(1.0),
            "default-band affine null space failed: innov_energy={innov_energy:.3e}"
        );
    }
}
