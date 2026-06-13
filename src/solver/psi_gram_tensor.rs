//! Certified Chebyshev-in-ψ Gram tensor: n-independent design-moving trials
//! (#1033 item b).
//!
//! ## Why
//!
//! When a design-moving hyperparameter ψ (= log κ for the radial families) is
//! searched by the outer loop, every trial today rebuilds the n×k design and
//! re-forms XᵀWX — an O(n·k) + O(n·k²) pass per trial. But along the trial
//! window every design entry `X(ψ)[i, j]` is an ANALYTIC function of ψ on a
//! compact interval (Matérn channels depend on (r, ℓ) only through κr and
//! κ-power prefactors; Duchon power blocks are ψ-free; partial-fraction
//! coefficients are analytic scalars), so the whole design admits a
//! geometrically-convergent Chebyshev expansion
//!
//! ```text
//!   X(ψ) = Σ_{d=0}^{D} X_d · T_d(ψ̃),     ψ̃ = affine map of ψ to [−1, 1],
//! ```
//!
//! with n×k coefficient slabs `X_d` computed ONCE from D+1 exact design
//! evaluations at Chebyshev nodes (a first-kind DCT). Precomputing the
//! expanded Gram `G̃[d][e] = X_dᵀ W X_e` and cross-products `c̃[d] = X_dᵀ W z`
//! in ONE pass over the data makes every subsequent trial n-free:
//!
//! ```text
//!   XᵀWX(ψ) = Σ_{d,e} T_d(ψ̃) T_e(ψ̃) G̃[d][e]          O(D²k²)
//!   XᵀWz(ψ) = Σ_d T_d(ψ̃) c̃[d]                          O(D k)
//!   ∂/∂ψ (XᵀWX) = Σ_{d,e} (T_d′T_e + T_dT_e′) G̃[d][e]   O(D²k²)
//! ```
//!
//! The ψ-gradient comes from the SAME representation as the value — one
//! source of truth, structurally immune to the objective↔gradient desync
//! class. `T_d′(ψ̃) = d·U_{d−1}(ψ̃) · dψ̃/dψ` is closed-form.
//!
//! ## Certification, not approximation-by-fiat
//!
//! Same discipline as [`crate::basis::radial_profile`]: [`PsiGramTensor::build`]
//! returns `None` (callers keep the exact per-trial path) unless BOTH
//! 1. the Chebyshev coefficient tail of the EXPANDED DESIGN decays below
//!    [`PSI_GRAM_CERT_RTOL`] of the design scale (geometric-decay certificate
//!    for analytic interpolands, with node-count escalation), and
//! 2. deterministic off-node spot checks of the ASSEMBLED Gram against an
//!    exactly rebuilt Gram agree to [`PSI_GRAM_SPOT_RTOL`].
//!
//! Trials outside `[psi_lo, psi_hi]` are the caller's signal to fall back to
//! the exact path ([`PsiGramTensor::contains`]).

use ndarray::{Array1, Array2, ArrayView1};

/// Relative ceiling on the per-column Chebyshev coefficient tail.
pub const PSI_GRAM_CERT_RTOL: f64 = 1.0e-12;

/// Relative agreement required at the off-node Gram spot checks.
pub const PSI_GRAM_SPOT_RTOL: f64 = 1.0e-10;

/// Relative agreement required of the analytic ψ-DERIVATIVE `dgram_dpsi`
/// against a central finite difference of the exactly rebuilt Gram, used to
/// certify the interior gradient sub-window. Set an order tighter than the
/// downstream outer-gradient comparison (1e-7 absolute) so the propagated
/// derivative-reconstruction error stays bit-tight in the gradient lane.
pub const PSI_GRAM_GRAD_SPOT_RTOL: f64 = 1.0e-9;

/// Number of equispaced scan points (per side) used to locate the interior
/// gradient sub-window where `dgram_dpsi` certifies.
pub const PSI_GRAM_GRAD_SCAN_POINTS: usize = 64;

/// Node-count escalation ladder for the expansion build (degree = nodes − 1).
///
/// The top rung sizes to WIDE trial windows: Chebyshev coefficients of the
/// Matérn-type channels decay like Bessel `I_d(σ)` with `σ ≈ s_max·halfwidth`
/// (s = κr), which only drops below the 1e-12 tail tolerance for `d ≳ 2σ` —
/// e.g. σ ≈ 9 (s_max ≈ 8, ±1.1 window) needs degree ≳ 40, so 33 nodes refuse
/// and 65 certify. Node counts stay trivially cheap (one design eval each).
pub const PSI_GRAM_NODE_LADDER: [usize; 4] = [9, 17, 33, 65];

/// Number of deterministic off-node spot-check ψ values.
pub const PSI_GRAM_SPOT_POINTS: usize = 3;

/// Certified Chebyshev-in-ψ expansion of a design-moving Gram (#1033b).
///
/// Holds the one-time n-pass products; every per-trial accessor is O(D²k²)
/// or cheaper and never touches n rows again.
pub struct PsiGramTensor {
    psi_lo: f64,
    psi_hi: f64,
    /// Interior sub-window `[grad_psi_lo, grad_psi_hi] ⊆ [psi_lo, psi_hi]` over
    /// which the ANALYTIC ψ-derivative `dgram_dpsi` reproduces the exact design
    /// derivative to [`PSI_GRAM_GRAD_SPOT_RTOL`] (#1033b gradient lane).
    ///
    /// The value reconstruction `gram_at` is certified over the FULL window
    /// (`T_d ≤ 1` everywhere), but the derivative reconstruction amplifies the
    /// coefficient-tail error by `T_d′ ∼ d²`, which blows up toward the window
    /// endpoints (the classic Chebyshev endpoint phenomenon). The gradient lane
    /// therefore only fires on this certified interior sub-window; near-edge
    /// trials keep the exact per-trial slab gradient. `contains` (value lane)
    /// still spans the full window.
    grad_psi_lo: f64,
    grad_psi_hi: f64,
    /// Number of Chebyshev coefficients (degree + 1).
    n_coeff: usize,
    k: usize,
    /// `gram[d * n_coeff + e]` = `X_dᵀ W X_e` (k×k); symmetric in (d, e) up to
    /// transpose: `gram[d][e] == gram[e][d]ᵀ`.
    gram: Vec<Array2<f64>>,
    /// `rhs[d]` = `X_dᵀ W z` (the caller's fixed weighted response/offset).
    rhs: Vec<Array1<f64>>,
    /// `zᵀWz` — ψ-free, captured at build so the Gaussian sufficient-statistic
    /// triple can be assembled per trial without any row access.
    zt_w_z: f64,
}

/// One ladder rung's outcome: a hard evaluation failure aborts the whole
/// build (no larger rung can fix a non-finite design), an uncertified tail
/// escalates to the next rung, and a candidate proceeds to the spot check.
enum BuildOutcome {
    EvalFailed,
    TailNotCertified,
    Candidate(PsiGramTensor),
}

/// Chebyshev values `T_0..T_{n−1}` at `x ∈ [−1, 1]`.
fn cheb_t(x: f64, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n];
    if n > 0 {
        t[0] = 1.0;
    }
    if n > 1 {
        t[1] = x;
    }
    for d in 2..n {
        t[d] = 2.0 * x * t[d - 1] - t[d - 2];
    }
    t
}

/// Chebyshev derivative values `T_0′..T_{n−1}′` at `x ∈ [−1, 1]` in the
/// MAPPED coordinate (multiply by `dx/dψ` for the ψ-derivative):
/// `T_d′ = d · U_{d−1}` with the Chebyshev-U recurrence.
fn cheb_t_prime(x: f64, n: usize) -> Vec<f64> {
    let mut u = vec![0.0; n.max(1)];
    // U_0 = 1, U_1 = 2x, U_d = 2x U_{d−1} − U_{d−2}.
    if !u.is_empty() {
        u[0] = 1.0;
    }
    if n > 1 {
        u[1] = 2.0 * x;
    }
    for d in 2..n {
        u[d] = 2.0 * x * u[d - 1] - u[d - 2];
    }
    let mut tp = vec![0.0; n];
    for d in 1..n {
        tp[d] = d as f64 * u[d - 1];
    }
    tp
}

impl PsiGramTensor {
    /// Build and certify the tensor over `psi ∈ [psi_lo, psi_hi]`.
    ///
    /// `eval_design(psi)` must return the EXACT n×k design at `psi` (the same
    /// builder the per-trial path uses — exactness of the expansion is judged
    /// against it). `weights` are the fixed observation weights, `z` the fixed
    /// weighted-response target (e.g. `y − offset`). Returns `None` when the
    /// window is degenerate, any evaluation fails/has non-finite entries, or
    /// no ladder rung certifies — callers then keep the exact per-trial path.
    pub fn build(
        mut eval_design: impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Option<Self> {
        if !(psi_lo.is_finite() && psi_hi.is_finite()) || psi_hi <= psi_lo {
            return None;
        }
        for &m in PSI_GRAM_NODE_LADDER.iter() {
            match Self::build_at(&mut eval_design, weights, z, psi_lo, psi_hi, m) {
                // An exact evaluation failed or was non-finite somewhere in
                // the window — no larger rung can fix that.
                BuildOutcome::EvalFailed => return None,
                // Tail not yet below the certificate at this rung: escalate.
                // (Conflating this with EvalFailed would kill the ladder at
                // its first — intentionally coarse — rung.)
                BuildOutcome::TailNotCertified => continue,
                BuildOutcome::Candidate(mut candidate) => {
                    if candidate.spot_check(&mut eval_design, weights) {
                        // Narrow the gradient sub-window to the certified
                        // interior (the value lane keeps the full window).
                        candidate.certify_gradient_window(&mut eval_design, weights);
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    fn build_at(
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
        m: usize,
    ) -> BuildOutcome {
        // First-kind Chebyshev nodes (no endpoints) and exact design slabs.
        let mut nodes_x = vec![0.0_f64; m];
        let mut designs: Vec<Array2<f64>> = Vec::with_capacity(m);
        for (i, x_slot) in nodes_x.iter_mut().enumerate() {
            let x = (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * m) as f64).cos();
            *x_slot = x;
            let psi = 0.5 * (psi_lo + psi_hi) + 0.5 * (psi_hi - psi_lo) * x;
            let Ok(design) = eval_design(psi) else {
                return BuildOutcome::EvalFailed;
            };
            if design.iter().any(|v| !v.is_finite()) {
                return BuildOutcome::EvalFailed;
            }
            designs.push(design);
        }
        let (n, k) = designs[0].dim();
        if designs.iter().any(|d| d.dim() != (n, k))
            || weights.len() != n
            || z.len() != n
            || n == 0
            || k == 0
        {
            return BuildOutcome::EvalFailed;
        }
        // First-kind discrete orthogonality: coefficient slabs
        //   X_d = (γ_d / m) Σ_i X(ψ_i) T_d(x_i),  γ_0 = 1, γ_d = 2.
        let t_at_nodes: Vec<Vec<f64>> = nodes_x.iter().map(|&x| cheb_t(x, m)).collect();
        let mut coeff_slabs: Vec<Array2<f64>> = Vec::with_capacity(m);
        for d in 0..m {
            let gamma = if d == 0 { 1.0 } else { 2.0 };
            let mut slab = Array2::<f64>::zeros((n, k));
            for (i, design) in designs.iter().enumerate() {
                let wgt = gamma / m as f64 * t_at_nodes[i][d];
                slab.scaled_add(wgt, design);
            }
            coeff_slabs.push(slab);
        }
        // Tail-decay certificate per design column: the trailing quarter of
        // the coefficient slabs must fall below rtol × column scale.
        let mut col_scale = vec![0.0_f64; k];
        for slab in &coeff_slabs {
            for (j, scale) in col_scale.iter_mut().enumerate() {
                for i in 0..n {
                    *scale = scale.max(slab[[i, j]].abs());
                }
            }
        }
        let tail_start = m - (m / 4).max(1);
        for slab in coeff_slabs.iter().skip(tail_start) {
            for (j, &scale) in col_scale.iter().enumerate() {
                let bound = PSI_GRAM_CERT_RTOL * scale.max(1e-300);
                for i in 0..n {
                    if slab[[i, j]].abs() > bound {
                        return BuildOutcome::TailNotCertified;
                    }
                }
            }
        }
        // One-time n-pass products: G̃[d][e] = X_dᵀ W X_e, c̃[d] = X_dᵀ W z.
        let mut weighted: Vec<Array2<f64>> = Vec::with_capacity(m);
        for slab in &coeff_slabs {
            let mut ws = slab.clone();
            for (mut row, &w) in ws.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            weighted.push(ws);
        }
        let mut wz = Array1::<f64>::zeros(z.len());
        let mut zt_w_z = 0.0_f64;
        for ((slot, &w), &zv) in wz.iter_mut().zip(weights.iter()).zip(z.iter()) {
            *slot = w * zv;
            zt_w_z += w * zv * zv;
        }
        let mut gram: Vec<Array2<f64>> = Vec::with_capacity(m * m);
        let mut rhs = Vec::with_capacity(m);
        for d in 0..m {
            for e in 0..m {
                if e < d {
                    // Symmetry: G̃[d][e] = G̃[e][d]ᵀ — reuse, don't recompute.
                    let g: Array2<f64> = gram[e * m + d].t().to_owned();
                    gram.push(g);
                } else {
                    gram.push(coeff_slabs[d].t().dot(&weighted[e]));
                }
            }
            rhs.push(coeff_slabs[d].t().dot(&wz));
        }
        BuildOutcome::Candidate(Self {
            psi_lo,
            psi_hi,
            // Provisional: `build` narrows these to the certified interior after
            // the value spot-check passes (`certify_gradient_window`).
            grad_psi_lo: psi_lo,
            grad_psi_hi: psi_hi,
            n_coeff: m,
            k,
            gram,
            rhs,
            zt_w_z,
        })
    }

    /// Off-node certification: the ASSEMBLED Gram must reproduce the exactly
    /// rebuilt Gram at deterministic interior ψ values.
    fn spot_check(
        &self,
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
    ) -> bool {
        for s in 0..PSI_GRAM_SPOT_POINTS {
            // Golden-ratio low-discrepancy interior points — never the nodes.
            let frac = ((s as f64 + 1.0) * 0.618_033_988_749_894_9).fract();
            let psi = self.psi_lo + frac * (self.psi_hi - self.psi_lo);
            let Ok(design) = eval_design(psi) else {
                return false;
            };
            let mut wd = design.clone();
            for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            let exact = design.t().dot(&wd);
            let assembled = self.gram_at(psi);
            let scale = exact
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1e-300);
            for (a, b) in assembled.iter().zip(exact.iter()) {
                if (a - b).abs() > PSI_GRAM_SPOT_RTOL * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Locate the largest centered interior interval where the analytic
    /// derivative `dgram_dpsi` reproduces a central finite difference of the
    /// exactly rebuilt Gram to [`PSI_GRAM_GRAD_SPOT_RTOL`], and store it as the
    /// gradient sub-window. Scans inward symmetrically from both endpoints; the
    /// value lane (`gram_at`) is unaffected. One-time cost: a handful of extra
    /// exact design evals (each cheap under the radial profile).
    fn certify_gradient_window(
        &mut self,
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
    ) {
        let span = self.psi_hi - self.psi_lo;
        // Central-FD step for the exact-derivative reference: small relative to
        // the window but comfortably above f64 cancellation at the Gram scale.
        let h = (span * 1e-4).max(1e-7);
        let exact_dgram = |psi: f64,
                           eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
         -> Option<Array2<f64>> {
            let weighted_gram = |p: f64,
                                 eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
             -> Option<Array2<f64>> {
                let design = eval(p).ok()?;
                let mut wd = design.clone();
                for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                    row.mapv_inplace(|v| v * w);
                }
                Some(design.t().dot(&wd))
            };
            let g_plus = weighted_gram(psi + h, eval)?;
            let g_minus = weighted_gram(psi - h, eval)?;
            Some((g_plus - g_minus) / (2.0 * h))
        };
        // True when the analytic derivative matches the exact FD at `psi`.
        let certifies = |me: &Self,
                             psi: f64,
                             eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
         -> bool {
            // Keep the FD stencil strictly inside the value window.
            if psi - h <= me.psi_lo || psi + h >= me.psi_hi {
                return false;
            }
            let Some(exact) = exact_dgram(psi, eval) else {
                return false;
            };
            let analytic = me.dgram_dpsi(psi);
            let scale = exact
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1e-300);
            analytic
                .iter()
                .zip(exact.iter())
                .all(|(a, b)| (a - b).abs() <= PSI_GRAM_GRAD_SPOT_RTOL * scale)
        };
        // Scan inward from each endpoint to the first certified point.
        let n = PSI_GRAM_GRAD_SCAN_POINTS;
        let mut lo = self.psi_hi;
        let mut hi = self.psi_lo;
        let mut found = false;
        for i in 0..=n {
            let psi = self.psi_lo + span * (i as f64) / (n as f64);
            if certifies(self, psi, eval_design) {
                lo = psi;
                found = true;
                break;
            }
        }
        for i in (0..=n).rev() {
            let psi = self.psi_lo + span * (i as f64) / (n as f64);
            if certifies(self, psi, eval_design) {
                hi = psi;
                break;
            }
        }
        if found && hi > lo {
            self.grad_psi_lo = lo;
            self.grad_psi_hi = hi;
        } else {
            // No certified interior: disable the gradient lane entirely
            // (empty sub-window) — callers keep the exact slab gradient.
            self.grad_psi_lo = f64::NAN;
            self.grad_psi_hi = f64::NAN;
        }
    }

    /// True when `psi` lies inside the certified window.
    pub fn contains(&self, psi: f64) -> bool {
        psi.is_finite() && psi >= self.psi_lo && psi <= self.psi_hi
    }

    /// True when `psi` lies inside the certified gradient sub-window — the
    /// region where the analytic ψ-derivative is bit-tight against the exact
    /// design derivative (#1033b). Outside it (near the window edges) callers
    /// must keep the exact slab gradient.
    pub fn contains_for_gradient(&self, psi: f64) -> bool {
        psi.is_finite()
            && self.grad_psi_lo.is_finite()
            && self.grad_psi_hi.is_finite()
            && psi >= self.grad_psi_lo
            && psi <= self.grad_psi_hi
    }

    fn mapped(&self, psi: f64) -> f64 {
        (2.0 * psi - (self.psi_lo + self.psi_hi)) / (self.psi_hi - self.psi_lo)
    }

    /// `XᵀWX(ψ)` assembled n-free in O(D²k²).
    pub fn gram_at(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.n_coeff);
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for d in 0..self.n_coeff {
            for e in 0..self.n_coeff {
                out.scaled_add(t[d] * t[e], &self.gram[d * self.n_coeff + e]);
            }
        }
        out
    }

    /// `XᵀWz(ψ)` assembled n-free in O(Dk).
    pub fn rhs_at(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.n_coeff);
        let mut out = Array1::<f64>::zeros(self.k);
        for (d, td) in t.iter().enumerate() {
            out.scaled_add(*td, &self.rhs[d]);
        }
        out
    }

    /// Exact `∂(XᵀWX)/∂ψ` from the SAME representation as the value — the
    /// structural cure for the objective↔gradient desync class on this
    /// channel. n-free, O(D²k²).
    pub fn dgram_dpsi(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let t = cheb_t(x, self.n_coeff);
        let tp = cheb_t_prime(x, self.n_coeff);
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for d in 0..self.n_coeff {
            for e in 0..self.n_coeff {
                out.scaled_add(
                    (tp[d] * t[e] + t[d] * tp[e]) * dx_dpsi,
                    &self.gram[d * self.n_coeff + e],
                );
            }
        }
        out
    }

    /// Exact `∂(XᵀWz)/∂ψ`, n-free.
    pub fn drhs_dpsi(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let tp = cheb_t_prime(x, self.n_coeff);
        let mut out = Array1::<f64>::zeros(self.k);
        for (d, tpd) in tp.iter().enumerate() {
            out.scaled_add(*tpd * dx_dpsi, &self.rhs[d]);
        }
        out
    }

    /// Assemble the Gaussian-identity sufficient-statistic cache at `psi`
    /// without touching a single data row — the bridge from this tensor into
    /// the inner PLS solver's fast path (#1033b → `GaussianFixedCache`).
    ///
    /// `(XᵀWX, XᵀWz, zᵀWz)` is everything the Gaussian penalized solve needs
    /// at any λ, so a ψ-trial that holds a certified tensor can hand the
    /// inner solver this cache instead of realizing the n×k design. The
    /// caller is responsible for `contains(psi)` (off-window trials fall back
    /// to the exact realizer path). Dense-path bridge only: the sparse
    /// scatter cache stays `None`.
    pub fn gaussian_fixed_cache_at(&self, psi: f64) -> crate::pirls::GaussianFixedCache {
        crate::pirls::GaussianFixedCache {
            xtwx_orig: self.gram_at(psi),
            xtwy_orig: self.rhs_at(psi),
            centered_weighted_y_sq: self.zt_w_z,
            xtwx_sparse_orig: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic Matérn-shaped synthetic design: entries g(e^{u_ij + ψ}) with
    /// g(s) = (1 + s)·exp(−s) (the ν = 3/2 Matérn shape) plus a ψ-free power
    /// column — the exact structural mix of the production radial designs.
    fn synth_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (n as f64 * k as f64) * 3.0;
                if j == k - 1 {
                    // ψ-free polynomial block column.
                    x[[i, j]] = r * r * r;
                } else {
                    let s = r * psi.exp();
                    x[[i, j]] = (1.0 + s) * (-s).exp();
                }
            }
        }
        Ok(x)
    }

    fn exact_gram(psi: f64, n: usize, k: usize, w: &Array1<f64>) -> Array2<f64> {
        let design = synth_design(psi, n, k).unwrap();
        let mut wd = design.clone();
        for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
            row.mapv_inplace(|v| v * wi);
        }
        design.t().dot(&wd)
    }

    /// #1033b primitive gate: the certified tensor must reproduce the exact
    /// Gram/rhs at arbitrary in-window ψ to certification accuracy, and its
    /// analytic ψ-derivative must match central finite differences of the
    /// exact Gram — value and gradient from one representation.
    #[test]
    fn psi_gram_tensor_matches_exact_gram_and_fd_gradient() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        for &psi in &[-1.1, -0.63, 0.0, 0.41, 0.97] {
            assert!(tensor.contains(psi));
            let exact = exact_gram(psi, n, k, &w);
            let fast = tensor.gram_at(psi);
            let scale = exact.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast.iter().zip(exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "gram mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // rhs check against the exact weighted cross-product.
            let design = synth_design(psi, n, k).unwrap();
            let mut wz = Array1::<f64>::zeros(n);
            for ((slot, &wi), &zi) in wz.iter_mut().zip(w.iter()).zip(z.iter()) {
                *slot = wi * zi;
            }
            let exact_rhs = design.t().dot(&wz);
            let fast_rhs = tensor.rhs_at(psi);
            let rscale = exact_rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast_rhs.iter().zip(exact_rhs.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * rscale,
                    "rhs mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // Analytic ψ-gradient vs central FD of the EXACT gram.
            let h = 1e-5;
            let g_plus = exact_gram(psi + h, n, k, &w);
            let g_minus = exact_gram(psi - h, n, k, &w);
            let dg = tensor.dgram_dpsi(psi);
            let dscale = dg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
            for ((a, p), m_) in dg.iter().zip(g_plus.iter()).zip(g_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-5 * dscale,
                    "dgram/dpsi mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
        }
        // Outside the window the caller must fall back to the exact path.
        assert!(!tensor.contains(psi_hi + 0.5));
        assert!(!tensor.contains(psi_lo - 0.5));

        // Bridge gate (#1033b → GaussianFixedCache): the n-free cache must
        // reproduce the exactly streamed sufficient statistics, and the
        // ridge-penalized solves through both must agree — the inner PLS
        // consumes nothing else, so this certifies the trial-loop handoff.
        for &psi in &[-0.9, 0.2, 0.8] {
            let cache = tensor.gaussian_fixed_cache_at(psi);
            let design = synth_design(psi, n, k).unwrap();
            let mut wd = design.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            let exact_gram = design.t().dot(&wd);
            let exact_rhs = wd.t().dot(&z);
            let exact_ztwz: f64 = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi * zi).sum();
            assert!(
                (cache.centered_weighted_y_sq - exact_ztwz).abs()
                    <= 1e-12 * exact_ztwz.abs().max(1e-300),
                "zᵀWz drift: cache={}, exact={exact_ztwz}",
                cache.centered_weighted_y_sq
            );
            // Ridge-penalized solve agreement: (G + I)β = r on both sides.
            let solve = |g: &Array2<f64>, r: &Array1<f64>| -> Array1<f64> {
                let mut a = g.clone();
                for i in 0..k {
                    a[[i, i]] += 1.0;
                }
                // Small dense Gauss elimination (k = 7 in this test).
                let mut aug = Array2::<f64>::zeros((k, k + 1));
                aug.slice_mut(ndarray::s![.., ..k]).assign(&a);
                aug.slice_mut(ndarray::s![.., k]).assign(r);
                for col in 0..k {
                    let piv = (col..k)
                        .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
                        .unwrap();
                    if piv != col {
                        for j in 0..=k {
                            let tmp = aug[[col, j]];
                            aug[[col, j]] = aug[[piv, j]];
                            aug[[piv, j]] = tmp;
                        }
                    }
                    let p = aug[[col, col]];
                    for row in 0..k {
                        if row == col {
                            continue;
                        }
                        let f = aug[[row, col]] / p;
                        for j in col..=k {
                            aug[[row, j]] -= f * aug[[col, j]];
                        }
                    }
                }
                Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]))
            };
            let beta_fast = solve(&cache.xtwx_orig, &cache.xtwy_orig);
            let beta_exact = solve(&exact_gram, &exact_rhs);
            let bscale = beta_exact
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in beta_fast.iter().zip(beta_exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-8 * bscale,
                    "penalized solve drift at psi={psi}: fast={a}, exact={b}"
                );
            }
        }
    }

    /// Certification negative: a NON-analytic (kinked) design must refuse to
    /// certify rather than silently approximate.
    #[test]
    fn psi_gram_tensor_refuses_non_analytic_design() {
        let (n, k) = (40usize, 3usize);
        let w = Array1::from_elem(n, 1.0);
        let z = Array1::from_elem(n, 0.5);
        let tensor = PsiGramTensor::build(
            |psi| {
                let mut x = Array2::<f64>::zeros((n, k));
                for i in 0..n {
                    for j in 0..k {
                        // |ψ| kink at 0 inside the window: not analytic.
                        x[[i, j]] = psi.abs() + (i + j) as f64 / (n + k) as f64;
                    }
                }
                Ok(x)
            },
            w.view(),
            z.view(),
            -1.0,
            1.0,
        );
        assert!(
            tensor.is_none(),
            "kinked design must fail the tail-decay/spot-check certificates"
        );
    }
}
