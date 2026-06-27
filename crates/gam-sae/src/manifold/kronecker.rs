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

#[cfg(test)]
mod tests {
    use super::SaeKroneckerRows;
    use gam_solve::arrow_schur::DeviceSaePcgData;
    use std::sync::Arc;

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
        let a_phi: Arc<[Vec<(usize, f64)>]> = Arc::from(
            vec![vec![(0usize, 2.0f64), (12, 1.0)], vec![(0, 0.5)]].into_boxed_slice(),
        );
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
