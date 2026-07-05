use gam_solve::arrow_schur::{BetaBlockId, BetaPenaltyOp};
use gam_runtime::warm_start::Fingerprinter;
use ndarray::{Array2, ArrayView1};
use std::ops::Range;
use std::sync::Arc;

/// Kronecker-factored per-row beta Jacobian primitive for SAE-manifold.
///
/// The per-row beta Jacobian has exact Kronecker form
///
/// ```text
/// J_{β,i} = φ_i^T ⊗ I_p
/// ```
///
/// where `φ_i ∈ ℝ^{m_i}` (active per-row atom·basis scalar weights, the
/// `a_k * phi` products in the assembly loop) and `p` is the decoder output
/// dimension.  The four trait methods implement the four operations that the
/// Arrow-Schur solver needs without ever forming the dense `(q × K·p)` block:
///
/// * `apply_jbeta`:   `u = J_β x`   (gather along active support)
/// * `scatter_jbeta_t`: `y += J_βᵀ u`  (scatter)
/// * `apply_l`:       `w = L u`     (q × p Jacobian apply)
/// * `apply_l_t`:     `u += Lᵀ v`   (q × p Jacobian transpose-accumulate)
///
/// The inner Schur row contribution
///
/// ```text
/// S_i = J_{β,i}^T (I - L_i^T A_i^{-1} L_i) J_{β,i}
/// ```
///
/// is applied in `O(m_i p + q p + q²)` per row per PCG iteration using
/// the five-step sequence:
///
/// ```text
/// u_p        = Σ_s φ_i[s] * x_β[s, :]    // gather (apply_jbeta)
/// w_q        = L_i u_p                    // q × p apply (apply_l)
/// v_q        = A_i^{-1} w_q               // existing per-row factor
/// u_p       -= L_i^T v_q                  // q × p apply-t (apply_l_t)
/// y_β[s, :] += φ_i[s] * u_p              // scatter (scatter_jbeta_t)
/// ```
pub trait SaeKroneckerRow {
    /// `u_out[j] = Σ_s φ_i[s] * x_beta[s * p + j]` for `j in 0..p`.
    ///
    /// Gather step: projects the full `K·p` beta vector down to the `p`-dimensional
    /// decoded output space using the active per-row support weights.
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]);

    /// `y_beta[s * p + j] += φ_i[s] * u[j]` for each active `(s, j)`.
    ///
    /// Scatter step: distributes the `p`-dimensional residual back into the
    /// full `K·p` beta gradient using the active per-row support weights.
    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]);

    /// `w_out[c] = Σ_j L[c, j] * u[j]` — apply the `q × p` local Jacobian.
    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]);

    /// `u_out[j] += Σ_c L[c, j] * v[c]` — accumulate `Lᵀ v` into `u_out`.
    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]);
}

/// Per-row Kronecker data for the SAE-manifold beta Jacobian.
///
/// Each row `i` stores:
/// * `a_phi_row`: sparse support — `(beta_base_idx, scalar_weight)` pairs,
///   one entry per active `(atom, basis_col)` combination.
/// * `local_jac_row`: the `(q × p)` assignment + coordinate Jacobian `L_i`
///   (same matrix written into `block.htt` via `local_jac` in the assembler).
///
/// Together these implement `J_β = φᵀ ⊗ I_p` without materializing the dense
/// `(q × K·p)` block.  Storage is `O(m_i · q · p)` per row rather than
/// `O(q · K · p)`.
#[derive(Debug, Clone)]
pub struct SaeKroneckerRows {
    /// Decoder output dimension `p`.
    pub(crate) p: usize,
    /// Per-row sparse support: `a_phi[i]` is a `Vec<(beta_base_idx, weight)>`.
    ///
    /// #1033 large-n: held as `Arc<[…]>` so the SAME backing allocation is
    /// shared with the solver's [`DeviceSaePcgData`] instead of cloned a second
    /// time (`O(n·k_active)` saved on the always-resident CPU non-frames path).
    pub(crate) a_phi: std::sync::Arc<[Vec<(usize, f64)>]>,
    /// Per-row local Jacobian `L_i`, shape `(q_i × p)` flattened row-major.
    ///
    /// Element `(c, j)` is at `local_jac[i][c * p + j]`.
    /// For heterogeneous (active-set) systems, each row may have a different
    /// `q_i = local_jac[i].len() / p`. Shared (`Arc<[…]>`) with the solver's
    /// `DeviceSaePcgData.local_jac` — the dominant `O(n·q·p)` resident slab.
    pub(crate) local_jac: std::sync::Arc<[Vec<f64>]>,
    /// #974 likelihood-whitening: the per-row output metric `M_n = U_n U_nᵀ`
    /// installed when the fit whitens the likelihood (`BehavioralFisher` /
    /// `WhitenedStructured`). The cross-block Jacobian factors as
    /// `J_β = φᵀ ⊗ I_p`, and the correct metric-aware cross-block is
    /// `H_tβ = L_i M_n J_β` — so applying `M_n` to the p-space intermediate
    /// (`apply_output_metric_row`) between the `J_β` gather and the `L_i` apply
    /// gives the exact whitened operator without ever touching `local_jac` or
    /// the isotropic `htt`/`jac_white` assembly. `None` ⇒ `M_n = I_p`, so every
    /// apply is bit-for-bit the historical isotropic path.
    pub(crate) output_metric: Option<gam_problem::RowMetric>,
}

impl SaeKroneckerRows {
    /// Build from per-row data collected during `assemble_arrow_schur`. The
    /// row count is implicit in `a_phi.len()` and `local_jac.len()`; the
    /// constructor asserts they agree so callers cannot pass mismatched rows.
    pub fn new(
        p: usize,
        a_phi: std::sync::Arc<[Vec<(usize, f64)>]>,
        local_jac: std::sync::Arc<[Vec<f64>]>,
    ) -> Self {
        assert_eq!(
            a_phi.len(),
            local_jac.len(),
            "SaeKroneckerRows: a_phi rows ({}) != local_jac rows ({})",
            a_phi.len(),
            local_jac.len(),
        );
        Self {
            p,
            a_phi,
            local_jac,
            output_metric: None,
        }
    }

    /// Install the per-row likelihood-whitening output metric `M_n = U_n U_nᵀ`
    /// (#974). `None` leaves the operator on the isotropic `M_n = I_p` path
    /// (bit-for-bit identical applies). Builder form so the isotropic
    /// constructor `new` stays a two-argument drop-in for existing callers.
    pub fn with_output_metric(mut self, metric: Option<gam_problem::RowMetric>) -> Self {
        self.output_metric = metric;
        self
    }

    /// Apply the per-row output metric `M_n x = U_n(U_nᵀ x) ∈ ℝ^p` in place to a
    /// length-`p` vector. No-op (leaves `u` untouched) when no whitening metric
    /// is installed (`M_n = I_p`), so the isotropic cross-block / β-Gram applies
    /// are bit-for-bit unchanged. This is the single site the whitening metric
    /// enters the matrix-free Kronecker operators: `H_tβ = L_i M_n J_β` and
    /// `H_ββ = Σ_i J_βᵀ M_n J_β` both apply `M_n` to the p-space intermediate.
    pub(crate) fn apply_output_metric_row(&self, row: usize, u: &mut [f64]) {
        if let Some(metric) = self.output_metric.as_ref() {
            let applied = metric.apply_metric_row(row, ArrayView1::from(&u[..]));
            u.copy_from_slice(&applied);
        }
    }

    /// The output-metric diagonal `M_n[oc, oc]` for one row (length `p`), used
    /// by the whitened β-Gram Jacobi preconditioner diagonal. Identity metric
    /// returns all-ones. `M_n[oc, oc] = Σ_k U_n[oc, k]²` from the low-rank
    /// factor (criterion-facing, `δ`-free).
    pub(crate) fn metric_output_diagonal(&self, row: usize) -> Vec<f64> {
        match self.output_metric.as_ref() {
            None => vec![1.0_f64; self.p],
            Some(metric) => {
                let rank = metric.metric_rank();
                let mut out = vec![0.0_f64; self.p];
                for (oc, slot) in out.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for k in 0..rank {
                        let u = metric.factor_entry(row, oc, k);
                        acc += u * u;
                    }
                    *slot = acc;
                }
                out
            }
        }
    }
}

impl SaeKroneckerRow for SaeKroneckerRows {
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]) {
        for val in u_out.iter_mut() {
            *val = 0.0;
        }
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += phi * x_beta[beta_base + j];
            }
        }
    }

    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]) {
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                y_beta[beta_base + j] += phi * u[j];
            }
        }
    }

    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        // Per-row q_i = jac.len() / p (supports heterogeneous active-set layouts).
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let mut acc = 0.0_f64;
            for j in 0..self.p {
                acc += jac[c * self.p + j] * u[j];
            }
            w_out[c] = acc;
        }
    }

    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += jac[c * self.p + j] * vc;
            }
        }
    }
}

/// #974 whitened data-fit Gauss–Newton β-Hessian operator.
///
/// The isotropic path collapses the per-row basis Gram into per-atom-pair
/// blocks and installs `H_ββ = G ⊗ I_p` ([`gam_solve::arrow_schur::SparseBlockKroneckerPenaltyOp`]).
/// That factorization is EXACT only when the reconstruction likelihood is
/// isotropic (`M_n = I_p`). When the fit whitens the likelihood
/// (`RowMetric::whitens_likelihood`) the per-row β-Jacobian is
/// `J_{β,n} = φ_nᵀ ⊗ I_p` and the true Gauss–Newton β-Hessian is
///
/// ```text
/// H_ββ = Σ_n J_{β,n}ᵀ M_n J_{β,n} = Σ_n (φ_n φ_nᵀ) ⊗ M_n
/// ```
///
/// which does NOT factor out of the basis Gram because `M_n` varies per row.
/// This operator applies that sum matrix-free, reusing the row support and the
/// low-rank metric on the shared [`SaeKroneckerRows`]: each row's contribution
/// is `gather (J_{β,n})` → `apply M_n` → `scatter (J_{β,n}ᵀ)`, the same p-space
/// intermediate the cross-block uses. Storage and per-matvec cost stay
/// `O(Σ_n k_active·p + n·p·r)` — the low-rank `M_n = U_n U_nᵀ` is never
/// materialized `p × p`. With `M_n = I_p` it reproduces the isotropic
/// `SparseBlockKroneckerPenaltyOp` exactly (pinned by the parity test), so it is
/// only installed on the whitening path.
pub struct WhitenedRowGramPenaltyOp {
    /// Shared per-row support `φ_n` (`a_phi`) + output metric `M_n`, reused from
    /// the cross-block operator so the two β-tier objects cannot drift apart.
    pub(crate) kron: Arc<SaeKroneckerRows>,
    /// Full β dimension `K = m_total · p`.
    pub(crate) k: usize,
}

impl WhitenedRowGramPenaltyOp {
    pub fn new(kron: Arc<SaeKroneckerRows>, k: usize) -> Self {
        Self { kron, k }
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.kron.a_phi.len()
    }
}

impl BetaPenaltyOp for WhitenedRowGramPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        // y += Σ_n J_{β,n}ᵀ M_n J_{β,n} x. `apply_jbeta` zeroes `u` internally,
        // so the p-space intermediate is rebuilt fresh each row.
        let p = self.kron.p;
        let mut u = vec![0.0_f64; p];
        for row in 0..self.n_rows() {
            self.kron.apply_jbeta(row, x, &mut u);
            self.kron.apply_output_metric_row(row, &mut u);
            self.kron.scatter_jbeta_t(row, &u, y);
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        // diag[(μ, oc)] += Σ_n φ_n[μ]² M_n[oc, oc].
        let p = self.kron.p;
        for row in 0..self.n_rows() {
            let mdiag = self.kron.metric_output_diagonal(row);
            for &(base, phi) in self.kron.a_phi[row].iter() {
                if phi == 0.0 {
                    continue;
                }
                let phi2 = phi * phi;
                for oc in 0..p {
                    diag[base + oc] += phi2 * mdiag[oc];
                }
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        // Dense per-atom β block for the block-Jacobi preconditioner. Only
        // support entries whose global β base falls inside this atom's range
        // contribute; within the block the local index of `(basis, oc)` is
        // `(base − range.start) + oc`. Metric columns `M_n[:, oc2]` are formed
        // factored via `apply_output_metric_row` on unit vectors.
        let range = &offsets[id.0];
        let p = self.kron.p;
        for row in 0..self.n_rows() {
            let entries: Vec<(usize, f64)> = self.kron.a_phi[row]
                .iter()
                .copied()
                .filter(|&(base, phi)| phi != 0.0 && base >= range.start && base < range.end)
                .collect();
            if entries.is_empty() {
                continue;
            }
            let mcols = metric_columns(&self.kron, row, p);
            for &(base_a, phi_a) in entries.iter() {
                let la = base_a - range.start;
                for &(base_b, phi_b) in entries.iter() {
                    let lb = base_b - range.start;
                    let w = phi_a * phi_b;
                    for oc2 in 0..p {
                        let col = &mcols[oc2];
                        for oc1 in 0..p {
                            out[[la + oc1, lb + oc2]] += w * col[oc1];
                        }
                    }
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        // Honest `O(K²)` materialization — used only by Direct-mode / small-K
        // fixtures. The matrix-free `matvec`/`diagonal` carry the production path.
        let p = self.kron.p;
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for row in 0..self.n_rows() {
            let entries = &self.kron.a_phi[row];
            let mcols = metric_columns(&self.kron, row, p);
            for &(base_a, phi_a) in entries.iter() {
                if phi_a == 0.0 {
                    continue;
                }
                for &(base_b, phi_b) in entries.iter() {
                    let w = phi_a * phi_b;
                    if w == 0.0 {
                        continue;
                    }
                    for oc2 in 0..p {
                        let col = &mcols[oc2];
                        for oc1 in 0..p {
                            out[[base_a + oc1, base_b + oc2]] += w * col[oc1];
                        }
                    }
                }
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("whitened-row-gram-penalty-op-v1");
        hasher.write_usize(self.k);
        hasher.write_usize(self.kron.p);
        hasher.write_usize(self.n_rows());
        for row in self.kron.a_phi.iter() {
            hasher.write_usize(row.len());
            for &(base, phi) in row.iter() {
                hasher.write_usize(base);
                hasher.write_f64(phi);
            }
        }
        // Fold the metric identity + its per-row Fisher-mass traces so a changed
        // whitening metric (fixed within a fit, but distinct across metrics)
        // invalidates the evidence cache.
        match self.kron.output_metric.as_ref() {
            None => hasher.write_str("metric:identity"),
            Some(metric) => {
                hasher.write_str("metric:whitened");
                hasher.write_usize(metric.metric_rank());
                for tr in metric.row_traces().iter() {
                    hasher.write_f64(*tr);
                }
            }
        }
    }
}

/// #974 — the frame-factored data-fit β-Hessian UNDER a likelihood-whitening
/// metric, matrix-free. This is the metric-aware generalization of
/// [`gam_solve::arrow_schur::FactoredFrameKroneckerOp`], which applies the
/// SEPARABLE `G_{ij} ⊗ (U_iᵀU_j)` — exact ONLY for the isotropic likelihood.
/// Under an active per-row metric `M_n = U_n U_nᵀ` the true reduced-space block
/// is `Σ_n (a_iφ_i)_n (a_jφ_j)_n (U_iᵀ M_n U_j)`, and `U_iᵀ M_n U_j` is
/// row-dependent so it does NOT pull out of the basis Gram — the separable form
/// is wrong. The exact operator is the per-row sandwich `Σ_n Φ_nᵀ M_n Φ_n`,
/// where `Φ_n` expands the ACTIVE atoms' factored coordinates `C` into the
/// `p`-dim decoded output through each atom's frame `U_k` (`I_p` for an unframed
/// atom). Each row: expand `u_p = Σ_active a·φ · (U_k · x_k)`, whiten
/// `u_p ← M_n u_p`, project back `y_k += a·φ · (U_kᵀ u_p)`. Cost
/// `O(n · k_active · M · p · r)`, memory `O(p)` scratch — NEVER the dense
/// `(M·p)²` per-atom covariance. With `M_n = I_p` it reduces bit-for-bit to the
/// isotropic factored operator (pinned by the reduction test), so it is only
/// installed on the whitening path.
pub struct WhitenedFactoredFrameOp {
    /// Decoder output dimension `p`.
    p: usize,
    /// Total reduced border dimension `Σ_k M_k · r_k`.
    dim: usize,
    /// Per-atom factored border offset (start of atom `k`'s `C_k` block).
    c_offsets: Vec<usize>,
    /// Per-atom frame rank `r_k` (equals `p` for an unframed atom).
    ranks: Vec<usize>,
    /// Per-atom basis size `M_k`.
    basis_sizes: Vec<usize>,
    /// Per-atom output frame `U_k` (`p × r_k`), or `None` for an unframed atom
    /// (`U_k = I_p`, the full-`p` block riding the identity special case).
    frames: Vec<Option<Array2<f64>>>,
    /// Per-row active support: `(atom, basis_col, weight = a_k·φ_k)`.
    support: Arc<[Vec<(usize, usize, f64)>]>,
    /// The likelihood-whitening metric `M_n` (always whitening here).
    metric: gam_problem::RowMetric,
}

impl WhitenedFactoredFrameOp {
    pub fn new(
        p: usize,
        dim: usize,
        c_offsets: Vec<usize>,
        ranks: Vec<usize>,
        basis_sizes: Vec<usize>,
        frames: Vec<Option<Array2<f64>>>,
        support: Arc<[Vec<(usize, usize, f64)>]>,
        metric: gam_problem::RowMetric,
    ) -> Self {
        Self {
            p,
            dim,
            c_offsets,
            ranks,
            basis_sizes,
            frames,
            support,
            metric,
        }
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.support.len()
    }

    /// Expand the active atoms' factored coordinates in row `row` into the
    /// `p`-dim decoded output: `u_p = Σ_active weight · (U_k · x_k[basis, :])`.
    fn expand_row(&self, row: usize, x: &[f64], u_p: &mut [f64]) {
        for v in u_p.iter_mut() {
            *v = 0.0;
        }
        for &(atom, basis, w) in self.support[row].iter() {
            if w == 0.0 {
                continue;
            }
            let r = self.ranks[atom];
            let cb = self.c_offsets[atom] + basis * r;
            match &self.frames[atom] {
                Some(u) => {
                    for j in 0..self.p {
                        let mut acc = 0.0;
                        for a in 0..r {
                            acc += u[[j, a]] * x[cb + a];
                        }
                        u_p[j] += w * acc;
                    }
                }
                None => {
                    // U_k = I_p ⇒ r = p; the block is the full-p decoder slice.
                    for j in 0..self.p {
                        u_p[j] += w * x[cb + j];
                    }
                }
            }
        }
    }

    /// Project the whitened p-space vector back into each active atom's factored
    /// coordinates: `y_k += weight · (U_kᵀ mu_p)`.
    fn project_row(&self, row: usize, mu_p: &[f64], y: &mut [f64]) {
        for &(atom, basis, w) in self.support[row].iter() {
            if w == 0.0 {
                continue;
            }
            let r = self.ranks[atom];
            let cb = self.c_offsets[atom] + basis * r;
            match &self.frames[atom] {
                Some(u) => {
                    for a in 0..r {
                        let mut acc = 0.0;
                        for j in 0..self.p {
                            acc += u[[j, a]] * mu_p[j];
                        }
                        y[cb + a] += w * acc;
                    }
                }
                None => {
                    for j in 0..self.p {
                        y[cb + j] += w * mu_p[j];
                    }
                }
            }
        }
    }

    /// Per-row reduced output Gram `U_kᵀ M_n U_k` (`r_k × r_k`) for one atom,
    /// used by the diagonal and dense-block preconditioner paths. For an
    /// unframed atom this is the dense `p × p` `M_n`.
    fn frame_gram_row(&self, atom: usize, row: usize) -> Array2<f64> {
        let r = self.ranks[atom];
        match &self.frames[atom] {
            Some(u) => {
                // MU[:, a] = M_n U[:, a]; Gram[a,b] = U[:,a]·MU[:,b].
                let mut mu = Array2::<f64>::zeros((self.p, r));
                for a in 0..r {
                    let col: Vec<f64> = (0..self.p).map(|j| u[[j, a]]).collect();
                    let m_col = self
                        .metric
                        .apply_metric_row(row, ndarray::aview1(&col));
                    for j in 0..self.p {
                        mu[[j, a]] = m_col[j];
                    }
                }
                let mut g = Array2::<f64>::zeros((r, r));
                for a in 0..r {
                    for b in 0..r {
                        let mut acc = 0.0;
                        for j in 0..self.p {
                            acc += u[[j, a]] * mu[[j, b]];
                        }
                        g[[a, b]] = acc;
                    }
                }
                g
            }
            None => {
                // U = I_p: Gram = M_n (p×p).
                let mut g = Array2::<f64>::zeros((r, r));
                for b in 0..r {
                    let mut e = vec![0.0_f64; self.p];
                    e[b] = 1.0;
                    let m_col = self
                        .metric
                        .apply_metric_row(row, ndarray::aview1(&e));
                    for a in 0..r {
                        g[[a, b]] = m_col[a];
                    }
                }
                g
            }
        }
    }
}

impl BetaPenaltyOp for WhitenedFactoredFrameOp {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        // y += Σ_n Φ_nᵀ M_n Φ_n x, matrix-free through a length-p scratch.
        let mut u_p = vec![0.0_f64; self.p];
        for row in 0..self.n_rows() {
            self.expand_row(row, x, &mut u_p);
            let mu = self
                .metric
                .apply_metric_row(row, ndarray::aview1(&u_p));
            self.project_row(row, &mu, y);
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        // diag[c_off[k] + basis·r + a] += Σ_n weight² (U_kᵀ M_n U_k)[a,a].
        for row in 0..self.n_rows() {
            // Cache the per-atom reduced Gram once per (row, atom) touched.
            let mut seen: std::collections::BTreeMap<usize, Array2<f64>> =
                std::collections::BTreeMap::new();
            for &(atom, basis, w) in self.support[row].iter() {
                if w == 0.0 {
                    continue;
                }
                let g = seen
                    .entry(atom)
                    .or_insert_with(|| self.frame_gram_row(atom, row));
                let r = self.ranks[atom];
                let cb = self.c_offsets[atom] + basis * r;
                let w2 = w * w;
                for a in 0..r {
                    diag[cb + a] += w2 * g[[a, a]];
                }
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        // Dense per-atom block for the block-Jacobi preconditioner. The block
        // ranges are per-atom `[c_off[k] .. c_off[k] + M_k·r_k]`; find which atom
        // this range starts at and accumulate its whitened factored block.
        let range = &offsets[id.0];
        let atom = match self.c_offsets.iter().position(|&o| o == range.start) {
            Some(k) => k,
            None => return,
        };
        let r = self.ranks[atom];
        for row in 0..self.n_rows() {
            // Support entries for THIS atom in this row: (basis, weight).
            let entries: Vec<(usize, f64)> = self.support[row]
                .iter()
                .filter(|&&(a, _, w)| a == atom && w != 0.0)
                .map(|&(_, basis, w)| (basis, w))
                .collect();
            if entries.is_empty() {
                continue;
            }
            let g = self.frame_gram_row(atom, row);
            for &(basis_a, wa) in entries.iter() {
                for &(basis_b, wb) in entries.iter() {
                    let w = wa * wb;
                    if w == 0.0 {
                        continue;
                    }
                    let base_a = basis_a * r;
                    let base_b = basis_b * r;
                    for a in 0..r {
                        for b in 0..r {
                            out[[base_a + a, base_b + b]] += w * g[[a, b]];
                        }
                    }
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        // Honest O(dim²) materialization — Direct/small-K fixtures only; the
        // matrix-free matvec/diagonal carry the production path.
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        let mut u_p = vec![0.0_f64; self.p];
        for col in 0..self.dim {
            let mut e = vec![0.0_f64; self.dim];
            e[col] = 1.0;
            let mut y = vec![0.0_f64; self.dim];
            // One column of the operator = matvec on a unit vector.
            for row in 0..self.n_rows() {
                self.expand_row(row, &e, &mut u_p);
                let mu = self
                    .metric
                    .apply_metric_row(row, ndarray::aview1(&u_p));
                self.project_row(row, &mu, &mut y);
            }
            for r in 0..self.dim {
                out[[r, col]] = y[r];
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("whitened-factored-frame-op-v1");
        hasher.write_usize(self.dim);
        hasher.write_usize(self.p);
        hasher.write_usize(self.n_rows());
        for (k, r) in self.ranks.iter().enumerate() {
            hasher.write_usize(*r);
            hasher.write_usize(self.basis_sizes[k]);
            hasher.write_usize(self.c_offsets[k]);
            if let Some(u) = &self.frames[k] {
                for &v in u.iter() {
                    hasher.write_f64(v);
                }
            }
        }
        for row in self.support.iter() {
            hasher.write_usize(row.len());
            for &(atom, basis, w) in row.iter() {
                hasher.write_usize(atom);
                hasher.write_usize(basis);
                hasher.write_f64(w);
            }
        }
        hasher.write_usize(self.metric.metric_rank());
        for tr in self.metric.row_traces().iter() {
            hasher.write_f64(*tr);
        }
    }
}

/// The `p` metric columns `M_n[:, oc]` for one row, formed factored via
/// `apply_output_metric_row` on unit vectors: `mcols[oc][oc1] = M_n[oc1, oc]`.
fn metric_columns(kron: &SaeKroneckerRows, row: usize, p: usize) -> Vec<Vec<f64>> {
    let mut mcols = Vec::with_capacity(p);
    for oc in 0..p {
        let mut e = vec![0.0_f64; p];
        e[oc] = 1.0;
        kron.apply_output_metric_row(row, &mut e);
        mcols.push(e);
    }
    mcols
}

#[cfg(test)]
mod tests {
    use super::{
        SaeKroneckerRow, SaeKroneckerRows, WhitenedFactoredFrameOp, WhitenedRowGramPenaltyOp,
    };
    use gam_solve::arrow_schur::{BetaBlockId, BetaPenaltyOp, DeviceSaePcgData};
    use std::sync::Arc;

    /// Dense reference for the frame-factored whitened β-Hessian: build each row's
    /// full `p × p` metric `M_n` from the low-rank factor, then sum
    /// `Φ_nᵀ M_n Φ_n` where `Φ_n` decodes the factored coords through the frames.
    /// One framed atom (`M` basis fns, frame `U` = `p × r`).
    fn dense_framed_whitened_reference(
        p: usize,
        r: usize,
        m: usize,
        u: &ndarray::Array2<f64>,           // p × r frame
        factor: &ndarray::Array2<f64>,      // n_rows × (p·rank) metric factor
        rank: usize,
        support: &[Vec<(usize, usize, f64)>], // per row (atom, basis, weight)
    ) -> ndarray::Array2<f64> {
        let dim = m * r;
        let mut h = ndarray::Array2::<f64>::zeros((dim, dim));
        for (row, entries) in support.iter().enumerate() {
            // M_n (p×p) from the factor row: M[i,j] = Σ_k f[i·rank+k] f[j·rank+k].
            let mut mn = ndarray::Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    let mut acc = 0.0;
                    for k in 0..rank {
                        acc += factor[[row, i * rank + k]] * factor[[row, j * rank + k]];
                    }
                    mn[[i, j]] = acc;
                }
            }
            // Uᵀ M_n U (r×r).
            let mut umu = ndarray::Array2::<f64>::zeros((r, r));
            for a in 0..r {
                for b in 0..r {
                    let mut acc = 0.0;
                    for i in 0..p {
                        for j in 0..p {
                            acc += u[[i, a]] * mn[[i, j]] * u[[j, b]];
                        }
                    }
                    umu[[a, b]] = acc;
                }
            }
            // H[(ba,a),(bb,b)] += w[ba] w[bb] (UᵀM_nU)[a,b].
            for &(_, ba, wa) in entries.iter() {
                for &(_, bb, wb) in entries.iter() {
                    let w = wa * wb;
                    for a in 0..r {
                        for b in 0..r {
                            h[[ba * r + a, bb * r + b]] += w * umu[[a, b]];
                        }
                    }
                }
            }
        }
        h
    }

    /// #974 core: the frame-factored whitened β-Hessian operator must equal the
    /// dense `Σ_n Φ_nᵀ M_n Φ_n` reference (to_dense + matvec + diagonal + block),
    /// with the metric genuinely NOT proportional to identity.
    #[test]
    fn whitened_factored_frame_op_matches_dense_reference() {
        let p = 3usize;
        let r = 2usize;
        let m = 2usize;
        let rank = 2usize;
        // Frame U (p×r), arbitrary (not orthonormal — the op must not assume it).
        let u = ndarray::Array2::from_shape_vec(
            (p, r),
            vec![1.0, 0.0, 0.5, 1.0, -0.3, 0.7],
        )
        .unwrap();
        // Metric factor rows (n_rows × p·rank), anisotropic per row.
        let factor = ndarray::Array2::from_shape_vec(
            (3, p * rank),
            vec![
                1.0, 0.2, 0.0, 1.0, 0.5, 0.0, // row 0
                0.3, 1.0, 1.0, 0.1, 0.0, 0.4, // row 1
                0.9, 0.0, 0.2, 0.6, 1.0, 0.0, // row 2
            ],
        )
        .unwrap();
        let support: Vec<Vec<(usize, usize, f64)>> = vec![
            vec![(0, 0, 1.0), (0, 1, 0.5)],
            vec![(0, 0, -0.3), (0, 1, 0.8)],
            vec![(0, 0, 0.2), (0, 1, -0.4)],
        ];
        let metric =
            gam_problem::RowMetric::behavioral_fisher(Arc::new(factor.clone()), p, rank).unwrap();

        let dim = m * r;
        let op = WhitenedFactoredFrameOp::new(
            p,
            dim,
            vec![0],
            vec![r],
            vec![m],
            vec![Some(u.clone())],
            Arc::from(support.clone().into_boxed_slice()),
            metric,
        );

        let reference =
            dense_framed_whitened_reference(p, r, m, &u, &factor, rank, &support);

        let dense = op.to_dense();
        for a in 0..dim {
            for b in 0..dim {
                assert!(
                    (dense[[a, b]] - reference[[a, b]]).abs() < 1e-10,
                    "to_dense mismatch at ({a},{b}): {} vs {}",
                    dense[[a, b]],
                    reference[[a, b]]
                );
            }
        }
        // matvec on a fixed vector equals reference·x.
        let x = vec![0.7, -0.2, 1.1, 0.3];
        let mut y = vec![0.0; dim];
        op.matvec(&x, &mut y);
        for a in 0..dim {
            let mut expect = 0.0;
            for b in 0..dim {
                expect += reference[[a, b]] * x[b];
            }
            assert!((y[a] - expect).abs() < 1e-10, "matvec[{a}] {} vs {expect}", y[a]);
        }
        // diagonal matches reference diagonal.
        let mut d = vec![0.0; dim];
        op.diagonal(&mut d);
        for a in 0..dim {
            assert!(
                (d[a] - reference[[a, a]]).abs() < 1e-10,
                "diag[{a}] {} vs {}",
                d[a],
                reference[[a, a]]
            );
        }
        // block (single atom, range 0..dim) equals the full reference block.
        let mut blk = ndarray::Array2::<f64>::zeros((dim, dim));
        op.block(BetaBlockId(0), &[0..dim], &mut blk);
        for a in 0..dim {
            for b in 0..dim {
                assert!(
                    (blk[[a, b]] - reference[[a, b]]).abs() < 1e-10,
                    "block mismatch at ({a},{b})"
                );
            }
        }
    }

    /// #974 reduction: with an IDENTITY metric and an ORTHONORMAL frame the
    /// factored whitened op collapses to the isotropic `g ⊗ I_r` block (a
    /// whitening-OFF fit is unchanged).
    #[test]
    fn whitened_factored_frame_op_reduces_to_g_kron_ir_under_identity_metric() {
        let p = 3usize;
        let r = 2usize;
        let m = 1usize;
        // Orthonormal frame columns e0, e1.
        let u = ndarray::Array2::from_shape_vec(
            (p, r),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        // Identity metric per row: factor = I_p (rank p).
        let mut factor = ndarray::Array2::<f64>::zeros((2, p * p));
        for row in 0..2 {
            for i in 0..p {
                factor[[row, i * p + i]] = 1.0;
            }
        }
        let support: Vec<Vec<(usize, usize, f64)>> =
            vec![vec![(0, 0, 1.5)], vec![(0, 0, 0.25)]];
        let metric =
            gam_problem::RowMetric::behavioral_fisher(Arc::new(factor), p, p).unwrap();
        let dim = m * r;
        let op = WhitenedFactoredFrameOp::new(
            p,
            dim,
            vec![0],
            vec![r],
            vec![m],
            vec![Some(u)],
            Arc::from(support.clone().into_boxed_slice()),
            metric,
        );
        // g = Σ_rows w² = 1.5² + 0.25² = 2.3125; block should be g·I_r.
        let g: f64 = support.iter().map(|row| row[0].2 * row[0].2).sum();
        let dense = op.to_dense();
        for a in 0..dim {
            for b in 0..dim {
                let expect = if a == b { g } else { 0.0 };
                assert!(
                    (dense[[a, b]] - expect).abs() < 1e-12,
                    "g⊗I_r mismatch at ({a},{b}): {} vs {expect}",
                    dense[[a, b]]
                );
            }
        }
    }

    /// #974 counterexample (the team-lead's spec): one row, one basis fn, `p = 2`,
    /// `M = diag(100, 0)`, `a·φ = 1`. The correct whitened data-fit β-Hessian is
    /// `H_ββ = Σ_n φ_n² M_n = M = diag(100, 0)` — curvature 100 in the behaviorally
    /// weighted channel and ZERO in the behaviorally-null one. The old isotropic
    /// `G ⊗ I_p` assembly gives `I_2` — the exact inversion this fix targets.
    #[test]
    fn whitened_beta_gram_matches_metric_not_identity() {
        let p = 2usize;
        // φ support: atom0/basis0 at β base 0 with weight a·φ = 1.
        let a_phi: Arc<[Vec<(usize, f64)>]> =
            Arc::from(vec![vec![(0usize, 1.0f64)]].into_boxed_slice());
        // local_jac unused by the β-Gram op; supply a matching dummy row.
        let jac: Arc<[Vec<f64>]> = Arc::from(vec![vec![0.0f64; p]].into_boxed_slice());
        // M = U Uᵀ = diag(100, 0): U ∈ ℝ^{p×1}, U[0,0]=10, U[1,0]=0
        // (row-major `u[i*probes + k]`).
        let u = ndarray::Array2::from_shape_vec((1, p * 1), vec![10.0, 0.0]).unwrap();
        let metric = gam_problem::RowMetric::behavioral_fisher(Arc::new(u), p, 1).unwrap();
        let kron = Arc::new(SaeKroneckerRows::new(p, a_phi, jac).with_output_metric(Some(metric)));
        let k = p; // beta_dim = m_total·p = 1·2
        let op = WhitenedRowGramPenaltyOp::new(kron, k);

        let dense = op.to_dense();
        assert!((dense[[0, 0]] - 100.0).abs() < 1e-9, "weighted curvature {}", dense[[0, 0]]);
        assert!(dense[[0, 1]].abs() < 1e-12);
        assert!(dense[[1, 0]].abs() < 1e-12);
        assert!(
            dense[[1, 1]].abs() < 1e-12,
            "behaviorally-null direction must carry ZERO curvature (isotropic G⊗I would give 1), got {}",
            dense[[1, 1]]
        );
        // matvec in the null direction is zero (isotropic G⊗I would return [0,1]).
        let mut y = vec![0.0; k];
        op.matvec(&[0.0, 1.0], &mut y);
        assert!(y[0].abs() < 1e-12 && y[1].abs() < 1e-12, "null-direction matvec {y:?}");
        // Jacobi diagonal matches M's diagonal.
        let mut d = vec![0.0; k];
        op.diagonal(&mut d);
        assert!((d[0] - 100.0).abs() < 1e-9 && d[1].abs() < 1e-12, "diag {d:?}");
    }

    /// #974 reduction: with an IDENTITY metric (`U = I_p`, `s = p` ⇒ `M_n = I_p`)
    /// the whitened β-Gram op must reproduce the isotropic `G ⊗ I_p` operator
    /// exactly (bit-for-bit at the values, so a whitening-OFF fit is unchanged).
    #[test]
    fn whitened_beta_gram_reduces_to_g_kron_ip_under_identity_metric() {
        let p = 2usize;
        // Two rows, two single-basis atoms (β bases 0 and p). Distinct φ weights.
        let a_phi: Arc<[Vec<(usize, f64)>]> = Arc::from(
            vec![
                vec![(0usize, 1.5f64), (p, -0.5f64)],
                vec![(0usize, 0.25f64)],
            ]
            .into_boxed_slice(),
        );
        let jac: Arc<[Vec<f64>]> =
            Arc::from(vec![vec![0.0f64; p], vec![0.0f64; p]].into_boxed_slice());
        // Identity metric per row: U = I_p (rank p).
        let mut u = ndarray::Array2::<f64>::zeros((2, p * p));
        for row in 0..2 {
            for i in 0..p {
                u[[row, i * p + i]] = 1.0;
            }
        }
        let metric = gam_problem::RowMetric::behavioral_fisher(Arc::new(u), p, p).unwrap();
        let kron =
            Arc::new(SaeKroneckerRows::new(p, a_phi.clone(), jac).with_output_metric(Some(metric)));
        let k = 2 * p; // m_total = 2 atoms × 1 basis
        let op = WhitenedRowGramPenaltyOp::new(kron, k);
        let dense = op.to_dense();

        // Reference: G[μ,μ'] = Σ_rows φ[μ]φ[μ'] with μ = base/p, then G ⊗ I_p.
        let m_total = 2usize;
        let mut g = vec![vec![0.0f64; m_total]; m_total];
        for row in a_phi.iter() {
            for &(bi, wi) in row.iter() {
                for &(bj, wj) in row.iter() {
                    g[bi / p][bj / p] += wi * wj;
                }
            }
        }
        for mi in 0..m_total {
            for mj in 0..m_total {
                for oc1 in 0..p {
                    for oc2 in 0..p {
                        let expect = if oc1 == oc2 { g[mi][mj] } else { 0.0 };
                        let got = dense[[mi * p + oc1, mj * p + oc2]];
                        assert!(
                            (got - expect).abs() < 1e-12,
                            "G⊗I mismatch at ({mi},{oc1})/({mj},{oc2}): {got} vs {expect}"
                        );
                    }
                }
            }
        }
    }

    /// #974 cross-block: the metric-threaded `SaeKroneckerRows` forward apply must
    /// equal the dense `H_tβ = L_i M_n J_β`, and its transpose leg must be the
    /// exact adjoint `H_βt = J_βᵀ M_n Lᵀ` (so the bordered system stays symmetric).
    #[test]
    fn whitened_cross_block_is_l_m_jbeta_with_exact_adjoint() {
        let p = 3usize;
        let q = 2usize;
        let r = 2usize;
        // One atom, one basis at β base 0, weight a·φ = 2.
        let a_phi: Arc<[Vec<(usize, f64)>]> =
            Arc::from(vec![vec![(0usize, 2.0f64)]].into_boxed_slice());
        // L_i (q×p) row-major.
        let l = vec![1.0, 0.0, -1.0, 0.5, 2.0, 0.0];
        let jac: Arc<[Vec<f64>]> = Arc::from(vec![l.clone()].into_boxed_slice());
        // M = U Uᵀ, U rows [[1,0],[0,1],[1,1]] (row-major u[i*r+k]).
        let u = ndarray::Array2::from_shape_vec(
            (1, p * r),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let metric =
            gam_problem::RowMetric::behavioral_fisher(Arc::new(u.clone()), p, r).unwrap();
        let kron = SaeKroneckerRows::new(p, a_phi, jac).with_output_metric(Some(metric));

        // Reference dense arithmetic.
        let mut m = [[0.0f64; 3]; 3];
        for (i, mrow) in m.iter_mut().enumerate() {
            for (j, mij) in mrow.iter_mut().enumerate() {
                let mut s = 0.0;
                for kk in 0..r {
                    s += u[[0, i * r + kk]] * u[[0, j * r + kk]];
                }
                *mij = s;
            }
        }
        let x = vec![1.0f64, -2.0, 0.5]; // k = p (one atom, one basis)
        // u_p = J_β x = 2·x; M u_p; out = L (M u_p).
        let up = [2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
        let mut mup = [0.0f64; 3];
        for (i, mup_i) in mup.iter_mut().enumerate() {
            for (j, up_j) in up.iter().enumerate() {
                *mup_i += m[i][j] * up_j;
            }
        }
        let mut expect = [0.0f64; 2];
        for (c, ec) in expect.iter_mut().enumerate() {
            for j in 0..3 {
                *ec += l[c * 3 + j] * mup[j];
            }
        }

        let mut u_p = vec![0.0; p];
        kron.apply_jbeta(0, &x, &mut u_p);
        kron.apply_output_metric_row(0, &mut u_p);
        let mut out = vec![0.0; q];
        kron.apply_l(0, &u_p, &mut out);
        for c in 0..q {
            assert!((out[c] - expect[c]).abs() < 1e-9, "c={c}: {} vs {}", out[c], expect[c]);
        }

        // Adjoint: ⟨H_tβ x, y⟩ == ⟨x, H_βt y⟩.
        let y = vec![0.7f64, -1.3];
        let mut lt = vec![0.0; p];
        kron.apply_l_t(0, &y, &mut lt);
        kron.apply_output_metric_row(0, &mut lt);
        let mut back = vec![0.0; p];
        kron.scatter_jbeta_t(0, &lt, &mut back);
        let lhs: f64 = out.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(back.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9, "adjoint mismatch {lhs} vs {rhs}");
    }

    /// #974 identity no-op: without a metric the cross-block apply is bit-for-bit
    /// the raw `L_i J_β` (the historical isotropic path).
    #[test]
    fn cross_block_without_metric_is_bit_identical_raw() {
        let p = 3usize;
        let q = 2usize;
        let a_phi: Arc<[Vec<(usize, f64)>]> =
            Arc::from(vec![vec![(0usize, 2.0f64)]].into_boxed_slice());
        let l = vec![1.0, 0.0, -1.0, 0.5, 2.0, 0.0];
        let jac: Arc<[Vec<f64>]> = Arc::from(vec![l.clone()].into_boxed_slice());
        let kron = SaeKroneckerRows::new(p, a_phi, jac); // no metric
        let x = vec![1.0f64, -2.0, 0.5];
        let mut u_p = vec![0.0; p];
        kron.apply_jbeta(0, &x, &mut u_p);
        let before = u_p.clone();
        kron.apply_output_metric_row(0, &mut u_p); // must be a no-op
        assert_eq!(before, u_p, "identity metric must leave u_p untouched");
        let mut out = vec![0.0; q];
        kron.apply_l(0, &u_p, &mut out);
        // Raw L J_β: u_p = 2x, out = L·(2x).
        let up = [2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
        for c in 0..q {
            let mut e = 0.0;
            for j in 0..p {
                e += l[c * p + j] * up[j];
            }
            assert!((out[c] - e).abs() < 1e-12);
        }
    }

    /// #1033 large-n sharing invariant (cross-crate half). The assembler hands
    /// BOTH the host matrix-free row operator (`SaeKroneckerRows`, this crate)
    /// and the solver's `DeviceSaePcgData` (`gam-solve`) the SAME `Arc<[…]>`
    /// backing allocation for `a_phi`/`local_jac` rather than a second full
    /// `O(n·q·p)` clone — the production path at `construction.rs`'s
    /// `set_device_sae_pcg_data` does exactly this. This pins the no-second-copy
    /// contract via `Arc::ptr_eq` across the crate boundary; the solver-internal
    /// `a_phi_shared()` half is covered in `gam-solve`
    /// (`device_a_phi_shared_is_refcount_bump_not_clone_1033`). A regression that
    /// reverts either side to a `Vec` deep-clone would double the always-resident
    /// per-row Jacobian footprint at the LLM shape (p≈5120) and fail here, even
    /// though every matvec stays numerically equal.
    #[test]
    fn device_and_kron_rows_share_backing_alloc_1033() {
        let p = 6usize;
        let a_phi: Arc<[Vec<(usize, f64)>]> =
            Arc::from(vec![vec![(0usize, 2.0f64), (12, 1.0)], vec![(0, 0.5)]].into_boxed_slice());
        let jac: Arc<[Vec<f64>]> =
            Arc::from(vec![vec![1.0; 4 * p], vec![2.0; 4 * p]].into_boxed_slice());
        // Both consumers built from refcount bumps of the same allocation.
        let host = SaeKroneckerRows::new(p, Arc::clone(&a_phi), Arc::clone(&jac));
        let device = DeviceSaePcgData {
            p,
            beta_dim: 6,
            a_phi: Arc::clone(&a_phi),
            local_jac: Arc::clone(&jac),
            smooth_blocks: Vec::new(),
            sparse_g_blocks: Vec::new(),
            frame: None,
        };
        // Host operator and device data point at the identical backing buffers.
        assert!(
            Arc::ptr_eq(&host.local_jac, &device.local_jac),
            "host SaeKroneckerRows and DeviceSaePcgData must share one local_jac alloc"
        );
        assert!(
            Arc::ptr_eq(&host.a_phi, &device.a_phi),
            "host SaeKroneckerRows and DeviceSaePcgData must share one a_phi alloc"
        );
        // strong_count = original + host + device — a deep clone would instead
        // leave the count at the lower no-share value.
        assert_eq!(
            Arc::strong_count(&jac),
            3,
            "exactly three references (original, host, device) share the Jacobian"
        );
    }
}
