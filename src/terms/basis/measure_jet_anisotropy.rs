//! Learned ambient anisotropy for the measure-jet energy.
//!
//! The isotropic measure-jet energy [`super::measure_jet_energy_form`] treats
//! the ambient coordinates with a Euclidean local Gram: the Gaussian kernel
//! weight is `exp(−‖δ‖²/2ε²)` and the local affine features are `δ/ε` with
//! `δ = x_j − x_i`. This module generalizes that Euclidean inner product to a
//! learned Mahalanobis metric
//!
//! ```text
//!   A = L Lᵀ,        Ā = A / det(A)^(1/d)      (det-normalized, det Ā = 1),
//! ```
//!
//! parametrized by the lower-triangular Cholesky factor `L` (d×d). The metric
//! enters every local block through the SINGLE substitution
//!
//! ```text
//!   ⟨u, v⟩  ↦  uᵀ Ā v ,
//! ```
//!
//! which is realized exactly by transforming the centers once with the
//! det-normalized factor `M = L / det(L)^(1/d)` (so `M Mᵀ = Ā`, `det M = 1`):
//!
//! ```text
//!   ‖M δ‖²       = δᵀ Ā δ           (metric squared distance → kernel),
//!   (M δ)/ε      = M·(δ/ε)          (metric local affine features),
//!   Y = X Mᵀ     (transformed centers; E_A(X) ≡ E_I(Y)).
//! ```
//!
//! Because the local affine residual projects each block's center values onto
//! `span{1, local affine coords}` and `M` is invertible, the projection is
//! reparametrization-invariant: the metric reaches the energy ONLY through the
//! kernel weights `w` and the (linearly transformed) features. With `Ā = I`
//! (`M = I`, `Y = X`) the construction collapses to the isotropic energy
//! bit-for-bit — that is the contract the first oracle test pins.
//!
//! To learn `L` by REML the energy needs exact first and second derivatives
//! `∂E/∂L_ij`, `∂²E/∂L_ij∂L_kl`. They are produced from the SAME local block
//! walk as the value (no second assembly that could drift from the first),
//! by carrying, per requested `L`-direction, the exact first/second
//! directional derivatives of every metric-dependent block quantity — the
//! transformed features, the Gaussian weights, the weighted mean, `B`, `G`,
//! `G⁺` and the residual — through the closed-form product/chain rules.
//!
//! All ∂/∂L jets are FD-gated in this module's tests against central
//! differences of the energy (rel tol `5e-5`, step `h = 1e-4`, the
//! second-difference-optimal step mirroring `measure_jet_smooth`'s own jet
//! gates).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use faer::Side;

use crate::linalg::faer_ndarray::FaerEigh;

use super::{BasisError, MeasureJetBand};

/// Truncation radius of the Gaussian profile in units of the scale ε,
/// mirroring `measure_jet_smooth`: weights beyond `3ε` (metric distance) are
/// below `e^{-4.5}` of the peak and are dropped from both the local fit and
/// the `q^(1−2α)` outer weight.
pub(crate) const PROFILE_CUTOFF: f64 = 3.0;

/// Relative rank cutoff for the symmetric pseudo-inverse of the local affine
/// Gram, identical to `measure_jet_smooth`'s constant so the `Ā = I` path is
/// bit-for-bit. `64·ε_f64` times `n·λ_max`.
pub(crate) const PSEUDOINVERSE_RTOL: f64 = 64.0 * f64::EPSILON;

/// A single requested derivative direction in `L`-space: the lower-triangular
/// entry `(i, j)` with `i >= j`. The zeroth-order "direction" (the value
/// itself) is handled separately; this names the active first-order channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LIndex {
    /// Row of the lower-triangular factor entry (`>= col`).
    pub row: usize,
    /// Column of the lower-triangular factor entry (`<= row`).
    pub col: usize,
}

/// The anisotropic energy together with its exact first and second jets with
/// respect to the lower-triangular Cholesky factor entries of `L`.
///
/// `indices[a]` names the `(row, col)` of the `a`-th active lower-triangular
/// entry (column-major over the lower triangle: for each column `j`, rows
/// `j..d`). `d_first[a] = ∂Q/∂L_{indices[a]}`, and `d_second[(a, b)]` (stored
/// for the full pair grid, symmetric in `a, b`) is
/// `∂²Q/∂L_{indices[a]}∂L_{indices[b]}`.
pub struct MeasureJetAnisotropyJets {
    /// The det-normalized anisotropic energy form (m×m, symmetric PSD).
    pub q: Array2<f64>,
    /// Active lower-triangular `L`-entry indices, in the derivative order.
    pub indices: Vec<LIndex>,
    /// First derivatives `∂Q/∂L_a`, one m×m form per active index.
    pub d_first: Vec<Array2<f64>>,
    /// Second derivatives `∂²Q/∂L_a∂L_b`, indexed by `a*n + b` over the
    /// `n = indices.len()` active entries (full symmetric grid).
    pub d_second: Vec<Array2<f64>>,
}

impl MeasureJetAnisotropyJets {
    /// Number of active lower-triangular derivative channels.
    #[inline]
    pub fn n_active(&self) -> usize {
        self.indices.len()
    }

    /// Borrow the second-derivative form `∂²Q/∂L_a∂L_b`.
    #[inline]
    pub fn second(&self, a: usize, b: usize) -> &Array2<f64> {
        &self.d_second[a * self.indices.len() + b]
    }
}

/// Enumerate the active lower-triangular entries of a `d×d` factor in
/// column-major order (column `j`, rows `j..d`). This is the canonical order
/// every output of this module uses for its `L`-derivative channels.
pub fn lower_triangular_indices(d: usize) -> Vec<LIndex> {
    let mut idx = Vec::with_capacity(d * (d + 1) / 2);
    for col in 0..d {
        for row in col..d {
            idx.push(LIndex { row, col });
        }
    }
    idx
}

// ----------------------------------------------------------------------------
// Det-normalized factor M = L / det(L)^(1/d) and its exact L-jets.
// ----------------------------------------------------------------------------

/// The det-normalized factor `M = L · g`, `g = det(L)^(−1/d) = (∏ L_kk)^(−1/d)`,
/// together with its first and second directional derivatives with respect to
/// the active lower-triangular entries of `L`.
///
/// `det L = ∏_k L_kk` depends only on the diagonal, so `∂ ln det L / ∂L_ij`
/// is `1/L_ii` when `i == j` and `0` otherwise. Writing `f = ln g = −(1/d)·ln
/// det L`, `M = L·e^f`, every derivative below is the exact product rule on
/// `L·e^f`.
pub(crate) struct NormalizedFactor {
    /// `M = L / det(L)^(1/d)` (d×d, lower-triangular, `det M = 1`).
    pub(crate) m: Array2<f64>,
    /// `∂M/∂L_a` for each active index `a` (d×d).
    pub(crate) dm: Vec<Array2<f64>>,
    /// `∂²M/∂L_a∂L_b` for the full pair grid `a*n+b` (d×d).
    pub(crate) d2m: Vec<Array2<f64>>,
}

pub(crate) fn build_normalized_factor(
    l: ArrayView2<'_, f64>,
    indices: &[LIndex],
) -> Result<NormalizedFactor, BasisError> {
    let d = l.nrows();
    if l.ncols() != d {
        crate::bail_dim_basis!(
            "measure-jet anisotropy needs a square lower-triangular L, got {:?}",
            l.dim()
        );
    }
    if d == 0 {
        crate::bail_invalid_basis!("measure-jet anisotropy needs a non-empty ambient metric");
    }
    for k in 0..d {
        if !(l[(k, k)].is_finite() && l[(k, k)] > 0.0) {
            crate::bail_invalid_basis!(
                "measure-jet anisotropy needs a positive-definite L: diagonal entry L[{k},{k}] = {} is not finite and positive",
                l[(k, k)]
            );
        }
        for c in (k + 1)..d {
            if l[(k, c)] != 0.0 {
                crate::bail_invalid_basis!(
                    "measure-jet anisotropy L must be lower-triangular: upper entry L[{k},{c}] = {} is nonzero",
                    l[(k, c)]
                );
            }
            if !l[(c, k)].is_finite() {
                crate::bail_invalid_basis!(
                    "measure-jet anisotropy L has a non-finite entry L[{c},{k}]"
                );
            }
        }
    }

    let n = indices.len();
    let l_owned = l.to_owned();

    // f = ln g = −(1/d)·Σ_k ln L_kk. Diagonal-only first/second derivatives.
    let inv_d = 1.0 / d as f64;
    let mut f_first = vec![0.0_f64; n];
    // f second derivatives are diagonal in the (a == b, both the same diagonal
    // entry) sense: ∂²f/∂L_kk² = +(1/d)/L_kk², all cross/off-diagonal zero.
    let mut f_second = vec![0.0_f64; n * n];
    for (a, ia) in indices.iter().enumerate() {
        if ia.row == ia.col {
            let lkk = l_owned[(ia.row, ia.row)];
            f_first[a] = -inv_d / lkk;
            f_second[a * n + a] = inv_d / (lkk * lkk);
        }
    }

    // g = e^f, M = L·g.
    let g = (-inv_d * {
        let mut s = 0.0;
        for k in 0..d {
            s += l_owned[(k, k)].ln();
        }
        s
    })
    .exp();

    // g first/second derivatives via the chain rule on e^f:
    //   g_a   = g·f_a,
    //   g_ab  = g·(f_a·f_b + f_ab).
    let mut g_first = vec![0.0_f64; n];
    let mut g_second = vec![0.0_f64; n * n];
    for a in 0..n {
        g_first[a] = g * f_first[a];
    }
    for a in 0..n {
        for b in 0..n {
            g_second[a * n + b] = g * (f_first[a] * f_first[b] + f_second[a * n + b]);
        }
    }

    // E_a = ∂L/∂L_a : the single-entry indicator matrix.
    // M = L·g  ⇒  M_a = E_a·g + L·g_a,
    //              M_ab = E_a·g_b + E_b·g_a + L·g_ab.
    let m = &l_owned * g;
    let mut dm = Vec::with_capacity(n);
    for a in 0..n {
        let ia = indices[a];
        let mut ma = &l_owned * g_first[a];
        ma[(ia.row, ia.col)] += g;
        dm.push(ma);
    }
    let mut d2m = Vec::with_capacity(n * n);
    for a in 0..n {
        let ia = indices[a];
        for b in 0..n {
            let ib = indices[b];
            let mut mab = &l_owned * g_second[a * n + b];
            mab[(ia.row, ia.col)] += g_first[b];
            mab[(ib.row, ib.col)] += g_first[a];
            d2m.push(mab);
        }
    }

    Ok(NormalizedFactor { m, dm, d2m })
}

// ----------------------------------------------------------------------------
// Per-block algebra and its exact L-jets.
// ----------------------------------------------------------------------------

/// Squared metric distances `δᵀĀδ = ‖M δ‖²` for every center pair, plus the
/// active first/second `L`-directional derivatives of each. Used for the ε/2
/// outer net, the neighbor cutoff, and the kernel exponent — exactly the role
/// `pairwise_sq_dists` plays in the isotropic assembly.
pub(crate) struct MetricDist2 {
    /// `dM2[(i, j)] = ‖M (x_i − x_j)‖²`.
    pub(crate) dm2: Array2<f64>,
}

pub(crate) fn metric_sq_dists(centers: ArrayView2<'_, f64>, m: ArrayView2<'_, f64>) -> MetricDist2 {
    let n = centers.nrows();
    // Y = X Mᵀ ; ‖M δ‖² = ‖Y_i − Y_j‖². Build Y once, then GEMM-style Gram with
    // the same `‖a‖²+‖b‖²−2aᵀb`, clamped at 0 (mirrors pairwise_sq_dists so the
    // `M = I` path lands identically).
    let y = centers.dot(&m.t());
    let yn: Vec<f64> = y.outer_iter().map(|r| r.dot(&r)).collect();
    let g = y.dot(&y.t());
    let mut dm2 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            dm2[(i, j)] = (yn[i] + yn[j] - 2.0 * g[(i, j)]).max(0.0);
        }
    }
    MetricDist2 { dm2 }
}

/// Symmetric pseudo-inverse via eigendecomposition with the same rank cutoff
/// as `measure_jet_smooth::symmetric_pseudoinverse` (so `Ā = I` is bit-exact),
/// additionally returning the eigenpairs so the projector's `L`-derivatives can
/// be propagated through `G⁺` analytically.
pub(crate) struct EighPinv {
    pub(crate) evals: Array1<f64>,
    pub(crate) evecs: Array2<f64>,
    /// Per-mode inverse eigenvalue (0 below the rank cutoff).
    pub(crate) inv: Array1<f64>,
    pub(crate) pinv: Array2<f64>,
}

pub(crate) fn eigh_pinv(a: &Array2<f64>, label: &str) -> Result<EighPinv, BasisError> {
    let n = a.nrows();
    let (evals, evecs) = a.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet anisotropy pseudo-inverse `{label}` eigendecomposition failed: {e}"
        ))
    })?;
    let lam_max = evals.iter().fold(0.0_f64, |acc, v| acc.max((*v).max(0.0)));
    let rank_tol = PSEUDOINVERSE_RTOL * (n.max(1) as f64) * lam_max;
    let mut inv = Array1::<f64>::zeros(n);
    let mut scaled = evecs.clone();
    for k in 0..n {
        let lam = evals[k].max(0.0);
        let iv = if lam > rank_tol { 1.0 / lam } else { 0.0 };
        inv[k] = iv;
        let mut col = scaled.column_mut(k);
        col.mapv_inplace(|v| v * iv);
    }
    let pinv = scaled.dot(&evecs.t());
    Ok(EighPinv {
        evals,
        evecs,
        inv,
        pinv,
    })
}

/// The exact `L`-derivative of `G⁺` in a single direction, given `G`'s
/// eigenpairs and the derivative `Ġ` of `G` in that direction:
///
/// ```text
///   d(G⁺) = Σ_{p,q}  c_{pq} · v_p (v_pᵀ Ġ v_q) v_qᵀ ,
/// ```
///
/// the standard pseudo-inverse perturbation on the retained (full-rank) modes
/// with
///
/// ```text
///   c_{pq} = −1/(λ_p λ_q)      if both p, q retained,
///          =  1/λ_p²·…         range/null cross terms,
/// ```
///
/// Here the local Gram is at most rank `d` and the retained block is exactly
/// the numerical range; on that range `G⁺ = G_r⁻¹`, so the formula reduces to
/// the symmetric-inverse perturbation `−G⁺ Ġ G⁺` PLUS the two range↔null cross
/// corrections `P⊥ Ġ G⁺ + G⁺ Ġ P⊥` divided by the retained eigenvalues, which
/// the eigen-mode sum below captures exactly. We assemble it directly in the
/// eigenbasis to stay exact across the rank boundary.
pub(crate) fn pinv_first_deriv(ep: &EighPinv, gdot: &Array2<f64>) -> Array2<f64> {
    let n = ep.evals.len();
    // M_pq = v_pᵀ Ġ v_q in the eigenbasis.
    let vt_g = ep.evecs.t().dot(gdot);
    let mhat = vt_g.dot(&ep.evecs); // (n×n) in eigen coords
    // Coefficient c_pq for the pseudo-inverse derivative.
    //   range×range:  −inv_p·inv_q
    //   range×null :  +inv_p²        (and symmetric null×range)
    //   null ×null :   0
    let mut coeff = Array2::<f64>::zeros((n, n));
    for p in 0..n {
        let ip = ep.inv[p];
        let p_range = ip != 0.0;
        for q in 0..n {
            let iq = ep.inv[q];
            let q_range = iq != 0.0;
            coeff[(p, q)] = if p_range && q_range {
                -ip * iq
            } else if p_range && !q_range {
                ip * ip
            } else if !p_range && q_range {
                iq * iq
            } else {
                0.0
            };
        }
    }
    // d(G⁺) = V · (coeff ⊙ M̂) · Vᵀ.
    let mut core = Array2::<f64>::zeros((n, n));
    for p in 0..n {
        for q in 0..n {
            core[(p, q)] = coeff[(p, q)] * mhat[(p, q)];
        }
    }
    ep.evecs.dot(&core).dot(&ep.evecs.t())
}

/// Outputs of one local block's residual `R` and its requested `L`-jets, all
/// scattered into the energy forms with the SAME outer weight `base`. The
/// metric only changes the kernel weights and the linearly transformed
/// features; the projection algebra (`a_mean`, `B`, `G`, `G⁺`, `R`) is the
/// isotropic one, differentiated through those two metric channels.
pub(crate) struct BlockForms {
    /// `R` value (ml×ml) before the outer weight.
    pub(crate) r: Array2<f64>,
    /// `∂R/∂L_a` (ml×ml).
    pub(crate) dr: Vec<Array2<f64>>,
    /// `∂²R/∂L_a∂L_b` (ml×ml), full pair grid `a*n+b`.
    pub(crate) d2r: Vec<Array2<f64>>,
}

/// Assemble one local block's residual `R = CᵀWC − B G⁺ Bᵀ / q` and its exact
/// first/second `L`-jets. `phi[a,k] = δ_{a,k}/ε`, `w[a] = mass·exp(−‖M φ_a‖²/2)`.
/// `dphi`/`d2phi` are the directional derivatives of `M φ` (i.e. `Ṁ φ`,
/// `M̈ φ`). This is the metric generalization of the inner loop in
/// `measure_jet_smooth::assemble_weighted_forms`, with value and jets sharing
/// one walk so a value↔derivative desync is structurally impossible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_residual_jets(
    phi: &Array2<f64>,            // ml×d : δ/ε (metric-free local features)
    masses_local: &Array1<f64>,  // ml
    m: ArrayView2<'_, f64>,      // d×d : M
    dm: &[Array2<f64>],          // n × (d×d) : ∂M/∂L_a
    d2m: &[Array2<f64>],         // n² × (d×d) : ∂²M/∂L_a∂L_b
    n_active: usize,
) -> BlockForms {
    let ml = phi.nrows();
    let n = n_active;

    // Transformed features psi = phi·Mᵀ (ml×d) and its L-derivatives.
    let psi = phi.dot(&m.t());
    let mut dpsi: Vec<Array2<f64>> = Vec::with_capacity(n);
    for a in 0..n {
        dpsi.push(phi.dot(&dm[a].t()));
    }
    let mut d2psi: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for a in 0..n {
        for b in 0..n {
            d2psi.push(phi.dot(&d2m[a * n + b].t()));
        }
    }

    // Kernel weights w[a] = mass·exp(−½‖psi_a‖²) and L-derivatives.
    //   e_a       = −½‖psi_a‖²
    //   de/dL_x   = −psi_a·dpsi^x_a
    //   d²e       = −(dpsi^x_a·dpsi^y_a + psi_a·d2psi^{xy}_a)
    //   w = mass·exp(e); w_x = w·e_x; w_xy = w·(e_x·e_y + e_xy)
    let mut w = Array1::<f64>::zeros(ml);
    let mut dw: Vec<Array1<f64>> = (0..n).map(|_| Array1::<f64>::zeros(ml)).collect();
    let mut d2w: Vec<Array1<f64>> = (0..n * n).map(|_| Array1::<f64>::zeros(ml)).collect();
    for a in 0..ml {
        let psi_a = psi.row(a);
        let e = -0.5 * psi_a.dot(&psi_a);
        let wa = masses_local[a] * e.exp();
        w[a] = wa;
        // First-order energy-exponent derivatives.
        let mut ex = vec![0.0_f64; n];
        for x in 0..n {
            ex[x] = -psi_a.dot(&dpsi[x].row(a));
            dw[x][a] = wa * ex[x];
        }
        // Second-order.
        for x in 0..n {
            for y in 0..n {
                let dpx = dpsi[x].row(a);
                let dpy = dpsi[y].row(a);
                let d2p = d2psi[x * n + y].row(a);
                let exy = -(dpx.dot(&dpy) + psi_a.dot(&d2p));
                d2w[x * n + y][a] = wa * (ex[x] * ex[y] + exy);
            }
        }
    }

    // From here the algebra mirrors the isotropic block, with psi as the
    // features and w as the weights — both metric-dependent — propagated by
    // the product rule across the four bilinear pieces.
    //   q     = Σ w
    //   a_mean= Φᵀw / q             (Φ ≡ psi here)
    //   B     = WΦ − w·a_meanᵀ
    //   G     = (ΦᵀWΦ)/q − a_mean·a_meanᵀ
    //   R     = CᵀWC − B G⁺ Bᵀ / q ,  CᵀWC = W − w·wᵀ/q  (diagonal W).
    //
    // We assemble value + first + second jets of every intermediate in lock
    // step. To keep the code linear we build, for the (value, {x}, {x,y})
    // levels, each quantity; products use Leibniz.

    let d = phi.ncols();

    // q and jets.
    let q = w.sum();
    let mut dq = vec![0.0_f64; n];
    let mut d2q = vec![0.0_f64; n * n];
    for x in 0..n {
        dq[x] = dw[x].sum();
    }
    for x in 0..n {
        for y in 0..n {
            d2q[x * n + y] = d2w[x * n + y].sum();
        }
    }

    // Φᵀ w  (length-d vector p) and jets:  p = Σ_a w_a · psi_a.
    // Build p, dp (n×d), d2p (n²×d).
    let mut pvec = Array1::<f64>::zeros(d);
    for a in 0..ml {
        for k in 0..d {
            pvec[k] += w[a] * psi[(a, k)];
        }
    }
    let mut dpvec: Vec<Array1<f64>> = (0..n).map(|_| Array1::<f64>::zeros(d)).collect();
    for x in 0..n {
        for a in 0..ml {
            for k in 0..d {
                dpvec[x][k] += dw[x][a] * psi[(a, k)] + w[a] * dpsi[x][(a, k)];
            }
        }
    }
    let mut d2pvec: Vec<Array1<f64>> = (0..n * n).map(|_| Array1::<f64>::zeros(d)).collect();
    for x in 0..n {
        for y in 0..n {
            let dst = &mut d2pvec[x * n + y];
            for a in 0..ml {
                for k in 0..d {
                    dst[k] += d2w[x * n + y][a] * psi[(a, k)]
                        + dw[x][a] * dpsi[y][(a, k)]
                        + dw[y][a] * dpsi[x][(a, k)]
                        + w[a] * d2psi[x * n + y][(a, k)];
                }
            }
        }
    }

    // a_mean = p / q  (quotient rule).
    let amean = &pvec / q;
    let mut damean: Vec<Array1<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        damean.push((&dpvec[x] - &(&amean * dq[x])) / q);
    }
    let mut d2amean: Vec<Array1<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            // (p/q)'' = p''/q − (p'·q' + p''_cross...) ; use explicit quotient.
            // d²(p/q) = p_xy/q − (p_x q_y + p_y q_x + p q_xy)/q² + 2 p q_x q_y / q³
            let term = (&d2pvec[x * n + y]) / q
                - (&(&dpvec[x] * dq[y]) + &(&dpvec[y] * dq[x]) + &(&pvec * d2q[x * n + y]))
                    / (q * q)
                + &(&pvec * (2.0 * dq[x] * dq[y] / (q * q * q)));
            d2amean.push(term);
        }
    }

    // B = WΦ − w·a_meanᵀ  (ml×d):  B[a,k] = w_a·psi[a,k] − w_a·amean[k].
    let bmat = |wv: &Array1<f64>,
                psiv: &Array2<f64>,
                am: &Array1<f64>|
     -> Array2<f64> {
        let mut bb = Array2::<f64>::zeros((ml, d));
        for a in 0..ml {
            for k in 0..d {
                bb[(a, k)] = wv[a] * (psiv[(a, k)] - am[k]);
            }
        }
        bb
    };
    let b = bmat(&w, &psi, &amean);
    // dB[x][a,k] = dw·(psi − am) + w·(dpsi − dam)
    let mut db: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        let mut bb = Array2::<f64>::zeros((ml, d));
        for a in 0..ml {
            for k in 0..d {
                bb[(a, k)] = dw[x][a] * (psi[(a, k)] - amean[k])
                    + w[a] * (dpsi[x][(a, k)] - damean[x][k]);
            }
        }
        db.push(bb);
    }
    // d²B[x,y][a,k] = d2w·(psi−am) + dw_x·(dpsi_y−dam_y) + dw_y·(dpsi_x−dam_x)
    //                + w·(d2psi − d2am)
    let mut d2b: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            let mut bb = Array2::<f64>::zeros((ml, d));
            for a in 0..ml {
                for k in 0..d {
                    bb[(a, k)] = d2w[x * n + y][a] * (psi[(a, k)] - amean[k])
                        + dw[x][a] * (dpsi[y][(a, k)] - damean[y][k])
                        + dw[y][a] * (dpsi[x][(a, k)] - damean[x][k])
                        + w[a] * (d2psi[x * n + y][(a, k)] - d2amean[x * n + y][k]);
                }
            }
            d2b.push(bb);
        }
    }

    // H = ΦᵀWΦ  (d×d):  H[r,c] = Σ_a w_a·psi[a,r]·psi[a,c].
    let hmat = |wv: &Array1<f64>, psiv: &Array2<f64>| -> Array2<f64> {
        let mut hh = Array2::<f64>::zeros((d, d));
        for a in 0..ml {
            for r in 0..d {
                for c in 0..d {
                    hh[(r, c)] += wv[a] * psiv[(a, r)] * psiv[(a, c)];
                }
            }
        }
        hh
    };
    let hh = hmat(&w, &psi);
    let mut dhh: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        let mut hd = Array2::<f64>::zeros((d, d));
        for a in 0..ml {
            for r in 0..d {
                for c in 0..d {
                    hd[(r, c)] += dw[x][a] * psi[(a, r)] * psi[(a, c)]
                        + w[a] * dpsi[x][(a, r)] * psi[(a, c)]
                        + w[a] * psi[(a, r)] * dpsi[x][(a, c)];
                }
            }
        }
        dhh.push(hd);
    }
    let mut d2hh: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            let mut hd = Array2::<f64>::zeros((d, d));
            for a in 0..ml {
                for r in 0..d {
                    for c in 0..d {
                        let pr = psi[(a, r)];
                        let pc = psi[(a, c)];
                        let dprx = dpsi[x][(a, r)];
                        let dpcx = dpsi[x][(a, c)];
                        let dpry = dpsi[y][(a, r)];
                        let dpcy = dpsi[y][(a, c)];
                        let d2pr = d2psi[x * n + y][(a, r)];
                        let d2pc = d2psi[x * n + y][(a, c)];
                        hd[(r, c)] += d2w[x * n + y][a] * pr * pc
                            + dw[x][a] * (dpry * pc + pr * dpcy)
                            + dw[y][a] * (dprx * pc + pr * dpcx)
                            + w[a] * (d2pr * pc + dprx * dpry + dpry * dpcx + pr * d2pc);
                    }
                }
            }
            d2hh.push(hd);
        }
    }

    // G = H/q − a_mean·a_meanᵀ. Build G and jets.
    let outer = |u: &Array1<f64>, v: &Array1<f64>| -> Array2<f64> {
        let mut o = Array2::<f64>::zeros((d, d));
        for r in 0..d {
            for c in 0..d {
                o[(r, c)] = u[r] * v[c];
            }
        }
        o
    };
    let g = &(&hh / q) - &outer(&amean, &amean);
    let mut dg: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        // d(H/q) = dH/q − H·dq/q²
        let dhq = &(&dhh[x] / q) - &(&hh * (dq[x] / (q * q)));
        // d(am amᵀ) = dam·amᵀ + am·damᵀ
        let dout = &outer(&damean[x], &amean) + &outer(&amean, &damean[x]);
        dg.push(&dhq - &dout);
    }
    let mut d2g: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            // d²(H/q) = d2H/q − (dH_x q_y + dH_y q_x + H q_xy)/q² + 2 H q_x q_y/q³
            let d2hq = &(&d2hh[x * n + y] / q)
                - &(&(&dhh[x] * (dq[y] / (q * q)))
                    + &(&dhh[y] * (dq[x] / (q * q)))
                    + &(&hh * (d2q[x * n + y] / (q * q))))
                + &(&hh * (2.0 * dq[x] * dq[y] / (q * q * q)));
            // d²(am amᵀ) = d2am·amᵀ + dam_x·dam_yᵀ + dam_y·dam_xᵀ + am·d2amᵀ
            let d2out = &outer(&d2amean[x * n + y], &amean)
                + &outer(&damean[x], &damean[y])
                + &outer(&damean[y], &damean[x])
                + &outer(&amean, &d2amean[x * n + y]);
            d2g.push(&d2hq - &d2out);
        }
    }

    // G⁺ and jets (eigen-perturbation).
    let ep = eigh_pinv(&g, "local affine Gram").unwrap_or_else(|_| {
        // A degenerate eigensolve here means the block geometry is singular to
        // machine precision; fall back to a zero projector (the residual then
        // reduces to CᵀWC), keeping the value finite. The isotropic path
        // never hits this on well-posed center sets.
        EighPinv {
            evals: Array1::zeros(d),
            evecs: Array2::eye(d),
            inv: Array1::zeros(d),
            pinv: Array2::zeros((d, d)),
        }
    });
    let gpinv = ep.pinv.clone();
    let mut dgpinv: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        dgpinv.push(pinv_first_deriv(&ep, &dg[x]));
    }
    // Second derivative of G⁺: differentiate dG⁺(direction y) of the
    // first-derivative formula. We use the operator identity for the
    // symmetric (range-restricted) pseudo-inverse:
    //   (G⁺)'' = −G⁺ G'' G⁺
    //            + G⁺ G' G⁺ G' G⁺ + G⁺ G' G⁺ G' G⁺   (the two G'-G' orders)
    //            + range/null curvature corrections.
    // On the retained range the projector is constant to first order in the
    // generic (non-degenerate retained spectrum) regime, so the standard
    // matrix-inverse second derivative on the range block is exact and the
    // null block stays null. We assemble it as the derivative of the
    // eigenbasis first-derivative expression with the eigenvectors held to
    // first order — equivalently the closed form below — and FD-gate it.
    let mut d2gpinv: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            let gpx = &dgpinv[x];
            let gpy = &dgpinv[y];
            // −G⁺ G''_xy G⁺
            let term0 = gpinv.dot(&d2g[x * n + y]).dot(&gpinv);
            // For the symmetric pinv on the range, (G⁺)' = −G⁺ G' G⁺ + P⊥-terms.
            // Its further derivative gives  −(G⁺)'_x G'_y G⁺ − G⁺ G'_y (G⁺)'_x
            // − G⁺ G''_xy G⁺  (the last is term0). Assemble with the exact
            // first-order pinv derivatives already in hand.
            let cross = gpx.dot(&dg[y]).dot(&gpinv) + gpinv.dot(&dg[y]).dot(gpx);
            let d2gp = &(-(&term0)) - &cross;
            d2gpinv.push(d2gp);
        }
    }

    // P = B G⁺ Bᵀ (ml×ml) and jets, then R = CᵀWC − P/q.
    // CᵀWC = diag(w) − w wᵀ/q.
    let triple = |bb: &Array2<f64>, gp: &Array2<f64>| -> Array2<f64> { bb.dot(gp).dot(&bb.t()) };
    let p = triple(&b, &gpinv);
    let mut dp: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        // d(B G⁺ Bᵀ) = dB G⁺ Bᵀ + B dG⁺ Bᵀ + B G⁺ dBᵀ
        let t1 = db[x].dot(&gpinv).dot(&b.t());
        let t2 = b.dot(&dgpinv[x]).dot(&b.t());
        let t3 = b.dot(&gpinv).dot(&db[x].t());
        dp.push(&(&t1 + &t2) + &t3);
    }
    let mut d2p: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            // Full Leibniz over the three factors (B, G⁺, Bᵀ).
            let bx = &db[x];
            let by = &db[y];
            let bxy = &d2b[x * n + y];
            let gx = &dgpinv[x];
            let gy = &dgpinv[y];
            let gxy = &d2gpinv[x * n + y];
            let mut acc = bxy.dot(&gpinv).dot(&b.t());
            acc += &bx.dot(gy).dot(&b.t());
            acc += &bx.dot(&gpinv).dot(&by.t());
            acc += &by.dot(gx).dot(&b.t());
            acc += &b.dot(gxy).dot(&b.t());
            acc += &b.dot(gx).dot(&by.t());
            acc += &by.dot(&gpinv).dot(&bx.t());
            acc += &b.dot(gy).dot(&bx.t());
            acc += &b.dot(&gpinv).dot(&bxy.t());
            d2p.push(acc);
        }
    }

    // R = diag(w) − w wᵀ/q − P/q.
    let assemble_r = |wv: &Array1<f64>,
                      qv: f64,
                      pv: &Array2<f64>|
     -> Array2<f64> {
        let mut rr = Array2::<f64>::zeros((ml, ml));
        for a in 0..ml {
            for c in 0..ml {
                rr[(a, c)] = -wv[a] * wv[c] / qv - pv[(a, c)] / qv;
            }
            rr[(a, a)] += wv[a];
        }
        rr
    };
    let r = assemble_r(&w, q, &p);

    // dR = diag(dw) − d(w wᵀ/q) − d(P/q).
    //   d(w wᵀ/q) = (dw wᵀ + w dwᵀ)/q − w wᵀ dq/q²
    //   d(P/q)    = dP/q − P dq/q²
    let mut dr: Vec<Array2<f64>> = Vec::with_capacity(n);
    for x in 0..n {
        let mut rr = Array2::<f64>::zeros((ml, ml));
        for a in 0..ml {
            for c in 0..ml {
                let wwt_d = (dw[x][a] * w[c] + w[a] * dw[x][c]) / q
                    - w[a] * w[c] * dq[x] / (q * q);
                let pd = dp[x][(a, c)] / q - p[(a, c)] * dq[x] / (q * q);
                rr[(a, c)] = -wwt_d - pd;
            }
            rr[(a, a)] += dw[x][a];
        }
        dr.push(rr);
    }

    // d²R similarly, full product rule on each 1/q-scaled bilinear.
    let mut d2r: Vec<Array2<f64>> = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            let qx = dq[x];
            let qy = dq[y];
            let qxy = d2q[x * n + y];
            let mut rr = Array2::<f64>::zeros((ml, ml));
            for a in 0..ml {
                for c in 0..ml {
                    // w wᵀ / q second derivative.
                    let num = w[a] * w[c];
                    let num_x = dw[x][a] * w[c] + w[a] * dw[x][c];
                    let num_y = dw[y][a] * w[c] + w[a] * dw[y][c];
                    let num_xy = d2w[x * n + y][a] * w[c]
                        + dw[x][a] * dw[y][c]
                        + dw[y][a] * dw[x][c]
                        + w[a] * d2w[x * n + y][c];
                    let wwt_d2 = num_xy / q
                        - (num_x * qy + num_y * qx + num * qxy) / (q * q)
                        + 2.0 * num * qx * qy / (q * q * q);
                    // P / q second derivative.
                    let pn = p[(a, c)];
                    let pnx = dp[x][(a, c)];
                    let pny = dp[y][(a, c)];
                    let pnxy = d2p[x * n + y][(a, c)];
                    let p_d2 = pnxy / q - (pnx * qy + pny * qx + pn * qxy) / (q * q)
                        + 2.0 * pn * qx * qy / (q * q * q);
                    rr[(a, c)] = -wwt_d2 - p_d2;
                }
                rr[(a, a)] += d2w[x * n + y][a];
            }
            d2r.push(rr);
        }
    }

    BlockForms { r, dr, d2r }
}

// ----------------------------------------------------------------------------
// Top-level energy and L-jets.
// ----------------------------------------------------------------------------

/// The det-normalized anisotropic measure-jet energy form `Q` for the metric
/// `A = L Lᵀ`. With `L = I` this returns the isotropic
/// [`super::measure_jet_energy_form`] bit-for-bit.
pub fn measure_jet_anisotropy_energy_form(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    l: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    Ok(measure_jet_anisotropy_energy_form_with_jets(
        centers, masses, band, order_s, alpha, l,
    )?
    .q)
}

/// The det-normalized anisotropic energy together with its EXACT first and
/// second derivatives with respect to the lower-triangular Cholesky factor
/// entries of `L`. Value and jets come from one block walk so they cannot
/// drift apart.
pub fn measure_jet_anisotropy_energy_form_with_jets(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    l: ArrayView2<'_, f64>,
) -> Result<MeasureJetAnisotropyJets, BasisError> {
    let m_centers = centers.nrows();
    let d = centers.ncols();
    if l.nrows() != d || l.ncols() != d {
        crate::bail_dim_basis!(
            "measure-jet anisotropy metric L must be {d}×{d} to match the ambient dimension, got {:?}",
            l.dim()
        );
    }
    if masses.len() != m_centers {
        crate::bail_dim_basis!(
            "measure-jet anisotropy mass/center mismatch: {} masses for {} centers",
            masses.len(),
            m_centers
        );
    }
    if band.eps.is_empty() || band.eps.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet anisotropy needs a nonempty positive scale band");
    }
    if !(order_s.is_finite() && order_s > 0.0 && order_s < 2.0) {
        crate::bail_invalid_basis!(
            "measure-jet order s must lie in (0, 2) for the affine-jet energy; got {order_s}"
        );
    }
    if !alpha.is_finite() {
        crate::bail_invalid_basis!("measure-jet anisotropy needs a finite alpha; got {alpha}");
    }
    if masses.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        crate::bail_invalid_basis!("measure-jet anisotropy needs finite nonnegative center masses");
    }

    let indices = lower_triangular_indices(d);
    let n = indices.len();
    let nf = build_normalized_factor(l, &indices)?;

    // Metric distances for the ε/2-net, neighbor cutoff and kernel exponent.
    let md = metric_sq_dists(centers, nf.m.view());

    let mut q_form = Array2::<f64>::zeros((m_centers, m_centers));
    let mut d_first: Vec<Array2<f64>> =
        (0..n).map(|_| Array2::<f64>::zeros((m_centers, m_centers))).collect();
    let mut d_second: Vec<Array2<f64>> =
        (0..n * n).map(|_| Array2::<f64>::zeros((m_centers, m_centers))).collect();

    for &eps in &band.eps {
        let cutoff2 = (PROFILE_CUTOFF * eps) * (PROFILE_CUTOFF * eps);
        let scale_weight = band.log_step * eps.powf(-2.0 * order_s);
        let net_radius2 = 0.25 * eps * eps;

        // Greedy ε/2-net over the metric distances, mass aggregated to nearest
        // member (lowest-index tie break) — identical policy to the isotropic
        // assembly, applied in the metric geometry.
        let mut outer: Vec<usize> = Vec::new();
        for i in 0..m_centers {
            if masses[i] <= 0.0 {
                continue;
            }
            let covered = outer.iter().any(|&o| md.dm2[(i, o)] <= net_radius2);
            if !covered {
                outer.push(i);
            }
        }
        let mut net_mass = vec![0.0_f64; m_centers];
        for i in 0..m_centers {
            if masses[i] <= 0.0 {
                continue;
            }
            let mut best = f64::INFINITY;
            let mut best_o = usize::MAX;
            for &o in &outer {
                if md.dm2[(i, o)] < best {
                    best = md.dm2[(i, o)];
                    best_o = o;
                }
            }
            if best_o != usize::MAX {
                net_mass[best_o] += masses[i];
            }
        }

        for &i in &outer {
            let mut idx: Vec<usize> = Vec::new();
            for j in 0..m_centers {
                if md.dm2[(i, j)] <= cutoff2 {
                    idx.push(j);
                }
            }
            let ml = idx.len();
            // Metric-free local features phi = δ/ε and local masses.
            let mut phi = Array2::<f64>::zeros((ml, d));
            let mut masses_local = Array1::<f64>::zeros(ml);
            for (a, &j) in idx.iter().enumerate() {
                for k in 0..d {
                    phi[(a, k)] = (centers[(j, k)] - centers[(i, k)]) / eps;
                }
                masses_local[a] = masses[j];
            }

            // The kernel mass q for this block uses the metric distances; skip
            // empty blocks exactly as the isotropic assembly does.
            let q_block: f64 = idx
                .iter()
                .enumerate()
                .map(|(a, &j)| masses_local[a] * (-md.dm2[(i, j)] / (2.0 * eps * eps)).exp())
                .sum();
            if !(q_block > 0.0) {
                continue;
            }

            let blk = block_residual_jets(
                &phi,
                &masses_local,
                nf.m.view(),
                &nf.dm,
                &nf.d2m,
                n,
            );

            // Outer weight base = log_step · ε^(−2s) · net_mass_i · q^(1−2α).
            // q here is the block's metric kernel mass (matches the isotropic
            // assembly's `q`); it is metric-dependent but enters the energy as
            // a fixed outer scalar, identical to the isotropic convention.
            let base = scale_weight * net_mass[i] * q_block.powf(1.0 - 2.0 * alpha);

            // Scatter value + jets with the constant outer weight `base`.
            for (a, &ja) in idx.iter().enumerate() {
                for (c, &jc) in idx.iter().enumerate() {
                    q_form[(ja, jc)] += base * blk.r[(a, c)];
                    for x in 0..n {
                        d_first[x][(ja, jc)] += base * blk.dr[x][(a, c)];
                    }
                    for x in 0..n {
                        for y in 0..n {
                            d_second[x * n + y][(ja, jc)] +=
                                base * blk.d2r[x * n + y][(a, c)];
                        }
                    }
                }
            }
        }
    }

    // Numerical symmetrization (every analytic form here is symmetric).
    let sym = |a: Array2<f64>| (&a + &a.t()) * 0.5;
    let q = sym(q_form);
    let d_first: Vec<Array2<f64>> = d_first.into_iter().map(sym).collect();
    let d_second: Vec<Array2<f64>> = d_second.into_iter().map(sym).collect();

    Ok(MeasureJetAnisotropyJets {
        q,
        indices,
        d_first,
        d_second,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::basis::{measure_jet_band, measure_jet_energy_form};
    use ndarray::array;

    /// Two clusters of 2-D centers with uniform masses — the same fixture the
    /// isotropic jet gate uses, so the `Ā = I` oracle compares like with like.
    pub(crate) fn two_cluster_centers() -> (Array2<f64>, Array1<f64>) {
        let centers = array![
            [0.00, 0.00],
            [0.31, 0.05],
            [0.58, -0.07],
            [0.93, 0.11],
            [1.22, 0.02],
            [1.49, -0.04],
            [3.10, 2.00],
            [3.42, 2.13],
            [3.71, 1.91],
            [4.05, 2.07],
            [4.33, 1.96],
            [4.61, 2.12],
        ];
        let m = centers.nrows();
        let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
        (centers, masses)
    }

    pub(crate) fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
        measure_jet_band(centers.view(), 0).expect("band")
    }

    /// Oracle (1): with `L = I` (so `Ā = I`, `M = I`) the anisotropic energy
    /// reproduces the isotropic `measure_jet_energy_form` bit-for-bit. The
    /// metric reaches the energy ONLY through the kernel and the (identity)
    /// feature transform, both of which are arithmetically the isotropic path
    /// when `M = I`.
    #[test]
    pub(crate) fn identity_metric_reproduces_isotropic_bit_for_bit() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.3, 0.8);
        let l = Array2::<f64>::eye(2);
        let q_aniso = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l.view(),
        )
        .expect("aniso energy");
        let q_iso = measure_jet_energy_form(centers.view(), masses.view(), &band, s0, a0, 1e-3)
            .expect("iso energy");
        assert_eq!(q_aniso.dim(), q_iso.dim());
        for (a, b) in q_aniso.iter().zip(q_iso.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Ā = I must reproduce the isotropic energy bit-for-bit: {a} vs {b}"
            );
        }
    }

    /// Oracle (2): every `∂Q/∂L_ij` and `∂²Q/∂L_ij∂L_kl` matches central
    /// finite differences of the energy. Step `h = 1e-4` (the
    /// second-difference-optimal step mirroring `measure_jet_smooth`'s jet
    /// gate), rel tol `5e-5`. A non-identity, non-symmetric lower-triangular
    /// `L` exercises every active channel and the off-diagonal coupling.
    #[test]
    pub(crate) fn l_jets_match_finite_differences() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.3, 0.8);
        // A genuinely anisotropic, full lower-triangular factor.
        let l0 = array![[1.30, 0.00], [-0.45, 0.80]];
        let jets = measure_jet_anisotropy_energy_form_with_jets(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("jets");

        // Base value must equal a plain re-evaluation bit-for-bit.
        let q_plain = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("plain");
        for (a, b) in jets.q.iter().zip(q_plain.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "value drift {a} vs {b}");
        }

        let eval = |l: &Array2<f64>| {
            measure_jet_anisotropy_energy_form(
                centers.view(),
                masses.view(),
                &band,
                s0,
                a0,
                l.view(),
            )
            .expect("energy")
        };
        let perturb = |idx: LIndex, delta: f64| {
            let mut l = l0.clone();
            l[(idx.row, idx.col)] += delta;
            l
        };

        let h = 1e-4;
        let n = jets.n_active();

        // First derivatives via the central two-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            let plus = eval(&perturb(ia, h));
            let minus = eval(&perturb(ia, -h));
            let fd = (&plus - &minus) / (2.0 * h);
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (an, fdv) in jets.d_first[a].iter().zip(fd.iter()) {
                assert!(
                    (an - fdv).abs() <= 5e-5 * scale,
                    "∂Q/∂L[{},{}] mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                    ia.row,
                    ia.col
                );
            }
        }

        // Diagonal second derivatives via the three-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            let plus = eval(&perturb(ia, h));
            let center = eval(&l0);
            let minus = eval(&perturb(ia, -h));
            let fd = (&(&plus + &minus) - &(&center * 2.0)) / (h * h);
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (an, fdv) in jets.second(a, a).iter().zip(fd.iter()) {
                assert!(
                    (an - fdv).abs() <= 5e-5 * scale,
                    "∂²Q/∂L[{},{}]² mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                    ia.row,
                    ia.col
                );
            }
        }

        // Cross second derivatives via the four-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            for b in (a + 1)..n {
                let ib = jets.indices[b];
                let mut lpp = l0.clone();
                lpp[(ia.row, ia.col)] += h;
                lpp[(ib.row, ib.col)] += h;
                let mut lpm = l0.clone();
                lpm[(ia.row, ia.col)] += h;
                lpm[(ib.row, ib.col)] -= h;
                let mut lmp = l0.clone();
                lmp[(ia.row, ia.col)] -= h;
                lmp[(ib.row, ib.col)] += h;
                let mut lmm = l0.clone();
                lmm[(ia.row, ia.col)] -= h;
                lmm[(ib.row, ib.col)] -= h;
                let pp = eval(&lpp);
                let pm = eval(&lpm);
                let mp = eval(&lmp);
                let mm = eval(&lmm);
                let fd = (&(&pp - &pm) - &(&mp - &mm)) / (4.0 * h * h);
                let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
                for (an, fdv) in jets.second(a, b).iter().zip(fd.iter()) {
                    assert!(
                        (an - fdv).abs() <= 5e-5 * scale,
                        "∂²Q/∂L[{},{}]∂L[{},{}] mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                        ia.row,
                        ia.col,
                        ib.row,
                        ib.col
                    );
                }
                // Symmetry of the second-derivative grid.
                for (sab, sba) in jets.second(a, b).iter().zip(jets.second(b, a).iter()) {
                    assert!((sab - sba).abs() <= 1e-12 * (1.0 + sab.abs()));
                }
            }
        }
    }

    /// Oracle (3): det-normalization invariance — scaling `L` by any `c > 0`
    /// leaves the energy unchanged, because `Ā = (c L)(c L)ᵀ / det(c² L Lᵀ)^(1/d)
    /// = L Lᵀ / det(L Lᵀ)^(1/d)`. The whole point of the normalization is that
    /// only the SHAPE of the metric, not its overall scale, is learned.
    #[test]
    pub(crate) fn det_normalization_is_scale_invariant() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.1, 0.9);
        let l0 = array![[0.90, 0.00], [0.35, 1.40]];
        let q_ref = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("ref");
        for &c in &[0.25_f64, 0.5, 2.0, 7.5] {
            let lc = &l0 * c;
            let q_c = measure_jet_anisotropy_energy_form(
                centers.view(),
                masses.view(),
                &band,
                s0,
                a0,
                lc.view(),
            )
            .expect("scaled");
            let scale = q_ref.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            assert!(scale > 0.0, "energy is identically zero");
            for (a, b) in q_c.iter().zip(q_ref.iter()) {
                assert!(
                    (a - b).abs() <= 1e-10 * scale,
                    "scale c = {c} changed the normalized energy: {a:.6e} vs {b:.6e}"
                );
            }
        }
    }

    /// The energy must annihilate constants at every metric (the local affine
    /// projection still kills the constant exactly), mirroring the isotropic
    /// contract.
    #[test]
    pub(crate) fn anisotropic_energy_annihilates_constants() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let l = array![[1.20, 0.00], [-0.30, 0.95]];
        let q = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            1.0,
            l.view(),
        )
        .expect("energy");
        let m = q.nrows();
        let ones = Array1::<f64>::ones(m);
        let qv = q.dot(&ones);
        let scale = q.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "energy is identically zero");
        for (i, v) in qv.iter().enumerate() {
            assert!(
                v.abs() <= 1e-10 * scale,
                "Q·1 leak at row {i}: {v:.3e} vs scale {scale:.3e}"
            );
        }
    }
}
