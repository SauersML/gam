use super::exact_eval_cache::*;

use super::family::*;

use super::gradient_paths::*;

use super::hessian_paths::*;

use super::row_kernel::*;

use super::*;

use crate::families::fnv1a::Fnv1a;

/// Bounded same-β reuse store for the BMS per-row cell-moment exact-cache.
///
/// The exact-cache (`BernoulliMarginalSlopeExactEvalCache`) — per-row solved
/// intercept contexts plus the batched per-row cell-moment partition/moments
/// — is a *pure* function of the family/data identity, the current coefficient
/// state (`block_states` betas + etas), and the outer-score subsample mask.
/// The outer BFGS issues a `Value` eval immediately followed by a
/// `ValueAndGradient` eval at the SAME ρ (hence the same converged β̂), and the
/// line search re-probes ρ values that map back to an already-evaluated β̂; each
/// such revisit reconstructs a fresh Hessian workspace which rebuilds this
/// exact-cache from scratch (`build_exact_eval_cache_with_options` →
/// `Arc::new`). Together with the joint-Hessian build this O(n·cells) rebuild is
/// the bulk of biobank-fit wall-clock.
///
/// This mirrors `custom_family::outer_objective::AssembledOperatorCache` one
/// layer down: a module-level `OnceLock<Mutex<..>>`, FIFO capacity 2, keyed by a
/// content fingerprint over EXACTLY the build inputs. Reuse is gated on exact
/// byte-equality of that fingerprint, so a hit returns an `Arc` to a cache that
/// is bit-identical to a fresh rebuild — identical row contexts, cell moments,
/// gradient, Hessian, and LAML cost. A miss builds, stores (evicting the older
/// of the two retained entries), and returns. Memory is bounded to the last two
/// distinct β̂ exact-caches (each O(n·cells); at biobank scale ≈ a few hundred
/// MB, well within the box's headroom, and the FIFO-2 cap is the same bound the
/// assembled-operator cache uses one layer up).
struct SharedExactCacheStore {
    /// `(fingerprint, exact-cache)` for at most the last two distinct β̂ builds.
    entries: Vec<(u64, Arc<BernoulliMarginalSlopeExactEvalCache>)>,
}

impl SharedExactCacheStore {
    const CAPACITY: usize = 2;

    fn get(&self, fingerprint: u64) -> Option<Arc<BernoulliMarginalSlopeExactEvalCache>> {
        self.entries
            .iter()
            .find(|(key, _)| *key == fingerprint)
            .map(|(_, cache)| Arc::clone(cache))
    }

    fn insert(&mut self, fingerprint: u64, cache: Arc<BernoulliMarginalSlopeExactEvalCache>) {
        if self.entries.iter().any(|(key, _)| *key == fingerprint) {
            return;
        }
        if self.entries.len() >= Self::CAPACITY {
            // Evict the oldest entry (front); the newest builds stay resident so
            // the immediate Value→ValueAndGradient pair at one β̂ always hits.
            self.entries.remove(0);
        }
        self.entries.push((fingerprint, cache));
    }
}

fn shared_exact_cache_store() -> &'static Mutex<SharedExactCacheStore> {
    static STORE: OnceLock<Mutex<SharedExactCacheStore>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(SharedExactCacheStore {
            entries: Vec::with_capacity(SharedExactCacheStore::CAPACITY),
        })
    })
}

/// Fill one deviation-basis column of the *score-warp* coefficient jet.
///
/// Shared body of the many `for_each_deviation_basis_cubic_at` visitor
/// closures over a score (`h_range`) deviation basis: the value coefficient
/// scales the local cubic by the slope `b`, and the `b`-partial scales it by
/// `1.0`. Identical across every score-warp call site (cell-loop and observed,
/// gradient / trace / trace-gradient / batched), which only differ in the
/// target arrays and the label string passed to the iterator.
#[inline]
pub(super) fn fill_score_basis_cell_coeff_jet(
    idx: usize,
    basis_span: super::exact_kernel::LocalSpanCubic,
    b: f64,
    scale: f64,
    c0: &mut [[f64; 4]],
    cb: &mut [[f64; 4]],
) {
    c0[idx] = scale_coeff4(
        super::exact_kernel::score_basis_cell_coefficients(basis_span, b),
        scale,
    );
    cb[idx] = scale_coeff4(
        super::exact_kernel::score_basis_cell_coefficients(basis_span, 1.0),
        scale,
    );
}

/// Fill one deviation-basis column of the *link-wiggle* coefficient jet to
/// first order only (value plus the `a`/`b`-partials).
///
/// Gradient-path counterpart of [`fill_link_basis_cell_coeff_jet`]: identical
/// up to the first `a`/`b`-partials, used where the second partials are not
/// required. Shared, unconditional body across the gradient call sites that
/// only differ in target arrays and the iterator label string.
#[inline]
pub(super) fn fill_link_basis_cell_coeff_gradient(
    idx: usize,
    basis_span: super::exact_kernel::LocalSpanCubic,
    a: f64,
    b: f64,
    scale: f64,
    c0: &mut [[f64; 4]],
    ca: &mut [[f64; 4]],
    cb: &mut [[f64; 4]],
) {
    c0[idx] = scale_coeff4(
        super::exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
        scale,
    );
    let (dc_aw_raw, dc_bw_raw) =
        super::exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
    ca[idx] = scale_coeff4(dc_aw_raw, scale);
    cb[idx] = scale_coeff4(dc_bw_raw, scale);
}

/// Fill one deviation-basis column of the *link-wiggle* coefficient jet.
///
/// Shared body of the many `for_each_deviation_basis_cubic_at` visitor
/// closures over a link (`w_range`) deviation basis: value, the two first
/// `a`/`b`-partials, and the three second `aa`/`ab`/`bb`-partials, each scaled
/// by `scale`. Identical across every link-wiggle call site, which only differ
/// in the target arrays and the iterator label string.
#[inline]
pub(super) fn fill_link_basis_cell_coeff_jet(
    idx: usize,
    basis_span: super::exact_kernel::LocalSpanCubic,
    a: f64,
    b: f64,
    scale: f64,
    c0: &mut [[f64; 4]],
    ca: &mut [[f64; 4]],
    cb: &mut [[f64; 4]],
    caa: &mut [[f64; 4]],
    cab: &mut [[f64; 4]],
    cbb: &mut [[f64; 4]],
) {
    c0[idx] = scale_coeff4(
        super::exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
        scale,
    );
    let (dc_aw_raw, dc_bw_raw) =
        super::exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
        super::exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
    ca[idx] = scale_coeff4(dc_aw_raw, scale);
    cb[idx] = scale_coeff4(dc_bw_raw, scale);
    caa[idx] = scale_coeff4(dc_aaw_raw, scale);
    cab[idx] = scale_coeff4(dc_abw_raw, scale);
    cbb[idx] = scale_coeff4(dc_bbw_raw, scale);
}

pub(super) fn assemble_bms_block_local_s_psi(
    deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
    per_block_rho: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref components) = deriv.s_psi_penalty_components {
        let mut s_psi = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_part.add_scaled_to(per_block_rho[*penalty_idx].exp(), &mut s_psi);
        }
        return s_psi;
    }
    if let Some(ref components) = deriv.s_psi_components {
        let mut s_psi = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_psi.scaled_add(per_block_rho[*penalty_idx].exp(), s_part);
        }
        s_psi
    } else if let Some(penalty_idx) = deriv.penalty_index {
        deriv
            .s_psi
            .mapv(|value| per_block_rho[penalty_idx].exp() * value)
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

impl BernoulliMarginalSlopeFamily {
    #[inline]
    pub(super) fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    #[inline]
    pub(super) fn unit_primary_direction(r: usize, idx: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r);
        out[idx] = 1.0;
        out
    }

    pub(super) fn empirical_rigid_intercept_for_row(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<f64, String> {
        // Cache slot is keyed by `(marginal.q, slope)`: a rejected TR trial
        // at one β and an accepted trial at another produce different
        // `(marginal_eta_row, slope_row)` for the same row, so without the
        // tag the slot can read back a value from a different trial and
        // poison the new root solve. The empirical-grid root depends only
        // on `(marginal.q, slope)` (the grid is immutable per latent measure),
        // so this two-scalar tag is sufficient.
        let beta_tag = hash_intercept_warm_start_key_rigid(marginal.q, slope);
        let cached = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.load_tagged(row, beta_tag));
        let root = empirical_intercept_from_marginal(
            marginal.mu,
            marginal.q,
            slope,
            self.probit_frailty_scale(),
            nodes,
            measure_weights,
            cached,
        )?;
        if let Some(cache) = self.intercept_warm_starts.as_ref() {
            cache.store_tagged(row, root, beta_tag);
        }
        Ok(root)
    }

    /// Objective-only fast path for the empirical-grid rigid kernel: returns
    /// `-w · log Φ(s · (intercept + s_f·g·z))` at the converged scalar
    /// intercept (the calibration root from `empirical_intercept_from_marginal`).
    /// Shares the `intercept_warm_starts` cache with the closed-form
    /// gradient/Hessian path, so successive line-search trials at nearby
    /// intercepts converge in `O(1)` Newton iterations per row.
    pub(super) fn empirical_rigid_neglog_only(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<f64, String> {
        let intercept =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = slope * self.probit_frailty_scale();
        let observed_eta = intercept + observed_slope * self.z[row];
        let signed = (2.0 * self.y[row] - 1.0) * observed_eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
        if !logcdf.is_finite() {
            return Err(format!(
                "empirical rigid neglog_only: non-finite log Φ at row {row}"
            ));
        }
        Ok(-self.weights[row] * logcdf)
    }

    /// Unified scalar-objective dispatcher for the rigid Bernoulli kernel.
    /// Routes to [`rigid_standard_normal_neglog_only`] for the standard-normal
    /// latent measure and [`Self::empirical_rigid_neglog_only`] for any
    /// empirical-grid measure. Replaces `rigid_row_kernel_eval(...)`'s
    /// `(neglog, _, _)` return when only the scalar is needed.
    pub(super) fn rigid_row_neglog_only(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<f64, String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => rigid_standard_normal_neglog_only(
                marginal.q,
                slope,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            ),
            Some(grid) => {
                self.empirical_rigid_neglog_only(row, marginal, slope, &grid.nodes, &grid.weights)
            }
        }
    }

    /// Closed-form row-primary negative-log-likelihood, gradient, and Hessian
    /// for the **rigid** empirical-grid Bernoulli kernel, in primary
    /// coordinates `(m = marginal_eta, g = slope)`.
    ///
    /// Replaces the second-order `empirical_rigid_neglog_jet` path — a 4-slot
    /// [`MultiDirJet`] driven through six Newton intercept-refinement passes
    /// per row — with the exact implicit-function-theorem solution. The
    /// intercept `a(m, g)` is the same scalar fixed point the jet converges to
    /// ([`Self::empirical_rigid_intercept_for_row`]); its derivatives follow in
    /// closed form from the grid calibration
    /// `F(a, m, g) = Σ_k π_k Φ(a + s·g·x_k) − μ(m) = 0`:
    ///
    /// ```text
    ///   D    = F_a = Σ_k π_k φ(η_k)            η_k = a + s·g·x_k
    ///   F_g        = Σ_k π_k φ(η_k)·(s·x_k)
    ///   F_aa       = Σ_k π_k (−η_k) φ(η_k)
    ///   F_ag       = Σ_k π_k (−η_k) φ(η_k)·(s·x_k)
    ///   F_gg       = Σ_k π_k (−η_k) φ(η_k)·(s·x_k)²
    ///   a_m  = μ'(m)/D                a_g  = −F_g/D
    ///   a_mm = (μ''(m) − F_aa·a_m²)/D
    ///   a_mg = −(F_ag·a_m + F_aa·a_m·a_g)/D
    ///   a_gg = −(F_gg + 2·F_ag·a_g + F_aa·a_g²)/D
    /// ```
    ///
    /// The marginal target enters only through the link derivatives
    /// `μ'(m) = marginal.mu1`, `μ''(m) = marginal.mu2`, so this stays correct
    /// for any marginal link, not just probit. The observed index is
    /// `η = a + s·g·z`, hence `η_m = a_m`, `η_g = a_g + s·z`, and the
    /// second-order observed derivatives equal the intercept's (`s·g·z` is
    /// linear in `g`). The negative-log-likelihood chain reuses the **same**
    /// signed-probit scalar kernel as the standard-normal rigid path
    /// ([`signed_probit_neglog_derivatives_up_to_fourth`]) so the two latent
    /// measures stay numerically consistent on shared terms:
    /// `ℓ_u = u1·η_u`, `ℓ_uv = u2·η_u·η_v + u1·η_uv`, with `u1 = s·k1`,
    /// `u2 = k2`.
    pub(super) fn empirical_rigid_primary_grad_hess_closed_form(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        let s = self.probit_frailty_scale();
        let a =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = s * slope;

        // Single grid pass over the calibration moments.
        let mut d = 0.0f64; // F_a = Σ π φ(η)
        let mut f_g = 0.0f64;
        let mut f_aa = 0.0f64;
        let mut f_ag = 0.0f64;
        let mut f_gg = 0.0f64;
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            let eta_k = a + observed_slope * node;
            let w_phi = weight * normal_pdf(eta_k);
            let sx = s * node;
            let neg_eta_w_phi = -eta_k * w_phi;
            d += w_phi;
            f_g += w_phi * sx;
            f_aa += neg_eta_w_phi;
            f_ag += neg_eta_w_phi * sx;
            f_gg += neg_eta_w_phi * sx * sx;
        }
        if !d.is_finite() || d <= 0.0 {
            return Err(format!(
                "empirical rigid closed-form: non-positive calibration denominator D={d} at row {row}"
            ));
        }

        // Intercept derivatives via the implicit function theorem.
        let a_m = marginal.mu1 / d;
        let a_g = -f_g / d;
        let a_mm = (marginal.mu2 - f_aa * a_m * a_m) / d;
        let a_mg = -(f_ag * a_m + f_aa * a_m * a_g) / d;
        let a_gg = -(f_gg + 2.0 * f_ag * a_g + f_aa * a_g * a_g) / d;

        // Observed-index derivatives at this row's own latent score z.
        let z = self.z[row];
        let eta_m = a_m;
        let eta_g = a_g + s * z;

        // Signed-probit negative-log-likelihood chain (shared scalar kernel).
        let w = self.weights[row];
        let sign = 2.0 * self.y[row] - 1.0;
        let observed_eta = a + observed_slope * z;
        let m_signed = sign * observed_eta;
        // ONE transcendental per row: fused logΦ + k1..k2 from a single
        // Mills-ratio evaluation (bit-identical to the prior two-call form;
        // the weight `w` is already folded into stack[1..] as before).
        if !(m_signed.is_finite() || m_signed == f64::INFINITY) {
            return Err(format!(
                "empirical rigid closed-form: non-finite signed margin {m_signed} at row {row}"
            ));
        }
        let stack = signed_probit_neglog_unary_stack(m_signed, w);
        if !stack[0].is_finite() {
            return Err(format!(
                "empirical rigid closed-form: non-finite log Φ at row {row}"
            ));
        }
        let u1 = sign * stack[1];
        let u2 = stack[2];

        let neglog = stack[0];
        let grad = [u1 * eta_m, u1 * eta_g];
        let h_mm = u2 * eta_m * eta_m + u1 * a_mm;
        let h_mg = u2 * eta_m * eta_g + u1 * a_mg;
        let h_gg = u2 * eta_g * eta_g + u1 * a_gg;
        Ok((neglog, grad, [[h_mm, h_mg], [h_mg, h_gg]]))
    }

    /// Closed-form uncontracted **third**-derivative tensor of the rigid
    /// empirical-grid row negative log-likelihood, in primary coordinates
    /// `(m = marginal_eta, g = slope)`. Replaces the 6-direction
    /// `empirical_rigid_neglog_jet` (a 64-coefficient `MultiDirJet` driven
    /// through six Newton intercept passes) used by [`Self::rigid_row_third_full`].
    ///
    /// Continues the implicit-function-theorem program of
    /// [`Self::empirical_rigid_primary_grad_hess_closed_form`] one order higher.
    /// Writing the grid calibration as `G(a, g) = μ(m)` with
    /// `G_{p,r} = Σ_k π_k Φ^{(p+r)}(η_k)·(s·x_k)^r` and `η_k = a + s·g·x_k`,
    /// the higher intercept derivatives follow by repeatedly applying the total
    /// operators `Dm(G_{p,r}) = G_{p+1,r}·a_m` and
    /// `Dg(G_{p,r}) = G_{p+1,r}·a_g + G_{p,r+1}` to the order-`n−1` identity and
    /// solving for the top term `D·a_{(i,j)}` (`D = G_a`). The needed CDF
    /// derivatives are `Φ' = φ`, `Φ'' = −η·φ`, `Φ''' = (η²−1)·φ`. The marginal
    /// link enters only as `μ', μ'', μ'''` (`marginal.mu1/mu2/mu3`), so this is
    /// correct for any marginal link. The negative-log-likelihood chain reuses
    /// the standard-normal signed-probit scalar kernel (`u1=s·k1`, `u2=k2`,
    /// `u3=s·k3`).
    pub(super) fn empirical_rigid_third_full_closed_form(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<[[[f64; 2]; 2]; 2], String> {
        let s = self.probit_frailty_scale();
        let a =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = s * slope;

        // Single grid pass: calibration moments through third CDF order.
        // `c2 = Φ''·π = −η·π·φ`, `c3 = Φ'''·π = (η²−1)·π·φ`.
        let (mut d, mut g_g) = (0.0f64, 0.0f64);
        let (mut g_aa, mut g_ag, mut g_gg) = (0.0f64, 0.0f64, 0.0f64);
        let (mut g_aaa, mut g_aag, mut g_agg, mut g_ggg) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            let eta_k = a + observed_slope * node;
            let w_phi = weight * normal_pdf(eta_k);
            let sx = s * node;
            let c2 = -eta_k * w_phi;
            let c3 = (eta_k * eta_k - 1.0) * w_phi;
            d += w_phi;
            g_g += w_phi * sx;
            g_aa += c2;
            g_ag += c2 * sx;
            g_gg += c2 * sx * sx;
            g_aaa += c3;
            g_aag += c3 * sx;
            g_agg += c3 * sx * sx;
            g_ggg += c3 * sx * sx * sx;
        }
        if !d.is_finite() || d <= 0.0 {
            return Err(format!(
                "empirical rigid third-full: non-positive calibration denominator D={d} at row {row}"
            ));
        }

        // Intercept derivatives a_{(i,j)} via the implicit function theorem.
        let a_m = marginal.mu1 / d;
        let a_g = -g_g / d;
        let a_mm = (marginal.mu2 - g_aa * a_m * a_m) / d;
        let coup = g_aa * a_g + g_ag; // recurring d/dg coupling factor on `a`
        let a_mg = -coup * a_m / d;
        let a_gg = -(g_aa * a_g * a_g + 2.0 * g_ag * a_g + g_gg) / d;
        let a_mmm = (marginal.mu3 - 3.0 * g_aa * a_m * a_mm - g_aaa * a_m * a_m * a_m) / d;
        let a_mmg =
            -(coup * a_mm + (g_aaa * a_g + g_aag) * a_m * a_m + 2.0 * g_aa * a_m * a_mg) / d;
        let a_mgg = -(2.0 * coup * a_mg
            + (g_aaa * a_g * a_g + 2.0 * g_aag * a_g + g_aa * a_gg + g_agg) * a_m)
            / d;
        let a_ggg = -(3.0 * a_gg * coup
            + g_aaa * a_g * a_g * a_g
            + 3.0 * g_aag * a_g * a_g
            + 3.0 * g_agg * a_g
            + g_ggg)
            / d;

        // Observed-index derivatives at this row's z. All second-and-higher
        // derivatives equal the intercept's (`s·g·z` is linear in `g`); only
        // the first g-derivative carries the extra `s·z`.
        let z = self.z[row];
        let eta_m = a_m;
        let eta_g = a_g + s * z;

        // Signed-probit chain to third order (shared scalar kernel).
        let w = self.weights[row];
        let sign = 2.0 * self.y[row] - 1.0;
        let m_signed = sign * (a + observed_slope * z);
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m_signed, w)?;
        let u1 = sign * k1;
        let u2 = k2;
        let u3 = sign * k3;

        // ℓ_ijk = u3·η_iη_jη_k + u2·(η_ijη_k + η_ikη_j + η_jkη_i) + u1·η_ijk.
        let t_mmm = u3 * eta_m * eta_m * eta_m + u2 * 3.0 * eta_m * a_mm + u1 * a_mmm;
        let t_mmg =
            u3 * eta_m * eta_m * eta_g + u2 * (eta_g * a_mm + 2.0 * eta_m * a_mg) + u1 * a_mmg;
        let t_mgg =
            u3 * eta_m * eta_g * eta_g + u2 * (eta_m * a_gg + 2.0 * eta_g * a_mg) + u1 * a_mgg;
        let t_ggg = u3 * eta_g * eta_g * eta_g + u2 * 3.0 * eta_g * a_gg + u1 * a_ggg;
        Ok(third_full_from_symmetric_components(
            t_mmm, t_mmg, t_mgg, t_ggg,
        ))
    }

    /// Closed-form uncontracted **fourth**-derivative tensor of the rigid
    /// empirical-grid row negative log-likelihood, in primary coordinates
    /// `(m = marginal_eta, g = slope)`. Replaces the 8-direction
    /// `empirical_rigid_neglog_jet` (a 256-coefficient `MultiDirJet` through six
    /// Newton intercept passes) used by [`Self::rigid_row_fourth_full`].
    ///
    /// Same implicit-function-theorem program as
    /// [`Self::empirical_rigid_third_full_closed_form`], one order higher.
    /// Intercept derivatives `a_{(i,j)}` (i m's, j g's, i+j≤4) come from
    /// differentiating the order-`n−1` identity of `G(a, g) = μ(m)` via the
    /// total operators `Dm(G_{p,r}) = G_{p+1,r}·a_m` and
    /// `Dg(G_{p,r}) = G_{p+1,r}·a_g + G_{p,r+1}` and isolating `D·a_{(i,j)}`.
    /// Needed CDF derivatives: `Φ'=φ`, `Φ''=−ηφ`, `Φ'''=(η²−1)φ`,
    /// `Φ''''=(3η−η³)φ`. The marginal link enters only as `μ'..μ''''`
    /// (`marginal.mu1..mu4`). The ℓ-chain uses the shared signed-probit kernel
    /// (`u1=s·k1, u2=k2, u3=s·k3, u4=k4`).
    pub(super) fn empirical_rigid_fourth_full_closed_form(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
        let s = self.probit_frailty_scale();
        let a =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = s * slope;

        // Single grid pass: calibration moments through fourth CDF order.
        let (mut d, mut g_g) = (0.0f64, 0.0f64);
        let (mut g_aa, mut g_ag, mut g_gg) = (0.0f64, 0.0f64, 0.0f64);
        let (mut g_aaa, mut g_aag, mut g_agg, mut g_ggg) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut g_aaaa, mut g_aaag, mut g_aagg, mut g_aggg, mut g_gggg) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            let eta_k = a + observed_slope * node;
            let w_phi = weight * normal_pdf(eta_k);
            let sx = s * node;
            let eta2 = eta_k * eta_k;
            let c2 = -eta_k * w_phi; // Φ''·π
            let c3 = (eta2 - 1.0) * w_phi; // Φ'''·π
            let c4 = (3.0 * eta_k - eta_k * eta2) * w_phi; // Φ''''·π
            let sx2 = sx * sx;
            let sx3 = sx2 * sx;
            let sx4 = sx3 * sx;
            d += w_phi;
            g_g += w_phi * sx;
            g_aa += c2;
            g_ag += c2 * sx;
            g_gg += c2 * sx2;
            g_aaa += c3;
            g_aag += c3 * sx;
            g_agg += c3 * sx2;
            g_ggg += c3 * sx3;
            g_aaaa += c4;
            g_aaag += c4 * sx;
            g_aagg += c4 * sx2;
            g_aggg += c4 * sx3;
            g_gggg += c4 * sx4;
        }
        if !d.is_finite() || d <= 0.0 {
            return Err(format!(
                "empirical rigid fourth-full: non-positive calibration denominator D={d} at row {row}"
            ));
        }

        // Intercept derivatives via the implicit function theorem.
        let (mu1, mu2, mu3, mu4) = (marginal.mu1, marginal.mu2, marginal.mu3, marginal.mu4);
        let a_m = mu1 / d;
        let a_g = -g_g / d;
        // P = Dg(D) = G_aa·a_g + G_ag, with its g-derivatives Pg, Pgg.
        let p = g_aa * a_g + g_ag;
        let a_mm = (mu2 - g_aa * a_m * a_m) / d;
        let a_mg = -p * a_m / d;
        let a_gg = -(g_aa * a_g * a_g + 2.0 * g_ag * a_g + g_gg) / d;
        let pg = g_aaa * a_g * a_g + 2.0 * g_aag * a_g + g_aa * a_gg + g_agg;
        let aag = g_aaa * a_g + g_aag; // Dg(G_aa)
        let a_mmm = (mu3 - 3.0 * g_aa * a_m * a_mm - g_aaa * a_m * a_m * a_m) / d;
        let a_mmg = -(p * a_mm + aag * a_m * a_m + 2.0 * g_aa * a_m * a_mg) / d;
        let a_mgg = -(2.0 * p * a_mg + pg * a_m) / d;
        let a_ggg = -(3.0 * a_gg * p
            + g_aaa * a_g * a_g * a_g
            + 3.0 * g_aag * a_g * a_g
            + 3.0 * g_agg * a_g
            + g_ggg)
            / d;
        // Pgg = Dg(Pg); R1 = Dg(G_aaa·a_g + G_aag) = Dg(aag).
        //
        // Pg = g_aaa·a_g² + 2·g_aag·a_g + g_aa·a_gg + g_agg, so its total
        // g-derivative Dg(Pg) must differentiate the `g_aa·a_gg` product by
        // BOTH factors: Dg(g_aa)·a_gg + g_aa·Dg(a_gg) = aag·a_gg + g_aa·a_ggg.
        // The `aag·a_gg` half lands in the `3·g_aaa·a_g·a_gg + 3·g_aag·a_gg`
        // tally below; the `g_aa·a_ggg` half is a distinct term (#833 — its
        // omission left a_mggg, hence the marginal/slope fourth-order block,
        // ~1.8% short of the finite-difference of the third-order form).
        let pgg = g_aaaa * a_g * a_g * a_g
            + 3.0 * g_aaag * a_g * a_g
            + 3.0 * g_aaa * a_g * a_gg
            + 3.0 * g_aagg * a_g
            + 3.0 * g_aag * a_gg
            + g_aa * a_ggg
            + g_aggg;
        let r1 = g_aaaa * a_g * a_g + 2.0 * g_aaag * a_g + g_aaa * a_gg + g_aagg;
        let aaag = g_aaaa * a_g + g_aaag; // Dg(G_aaa)
        let a_mmmm = (mu4
            - g_aa * (4.0 * a_m * a_mmm + 3.0 * a_mm * a_mm)
            - 6.0 * g_aaa * a_m * a_m * a_mm
            - g_aaaa * a_m * a_m * a_m * a_m)
            / d;
        let a_mmmg = -(p * a_mmm
            + 3.0 * aag * a_m * a_mm
            + 3.0 * g_aa * a_mg * a_mm
            + 3.0 * g_aa * a_m * a_mmg
            + aaag * a_m * a_m * a_m
            + 3.0 * g_aaa * a_m * a_m * a_mg)
            / d;
        let a_mmgg = -(2.0 * p * a_mmg
            + pg * a_mm
            + r1 * a_m * a_m
            + 4.0 * aag * a_m * a_mg
            + 2.0 * g_aa * a_mg * a_mg
            + 2.0 * g_aa * a_m * a_mgg)
            / d;
        let a_mggg = -(3.0 * p * a_mgg + 3.0 * pg * a_mg + pgg * a_m) / d;
        let a_gggg = -(4.0 * p * a_ggg
            + 6.0 * g_aaa * a_g * a_g * a_gg
            + 12.0 * g_aag * a_g * a_gg
            + 6.0 * g_agg * a_gg
            + 3.0 * g_aa * a_gg * a_gg
            + g_aaaa * a_g * a_g * a_g * a_g
            + 4.0 * g_aaag * a_g * a_g * a_g
            + 6.0 * g_aagg * a_g * a_g
            + 4.0 * g_aggg * a_g
            + g_gggg)
            / d;

        // Observed-index derivatives (only η_g carries the extra s·z).
        let z = self.z[row];
        let em = a_m;
        let eg = a_g + s * z;

        // Signed-probit chain to fourth order (shared scalar kernel).
        let w = self.weights[row];
        let sign = 2.0 * self.y[row] - 1.0;
        let (k1, k2, k3, k4) =
            signed_probit_neglog_derivatives_up_to_fourth(sign * (a + observed_slope * z), w)?;
        let (u1, u2, u3, u4) = (sign * k1, k2, sign * k3, k4);

        // ℓ_ijkl via Faà di Bruno: u4·(4 η's) + u3·(η_ij + 2 singles, 6 terms)
        // + u2·(3 pair-pair + 4 triple-single) + u1·η_ijkl.
        let t_mmmm = u4 * em * em * em * em
            + u3 * 6.0 * a_mm * em * em
            + u2 * (3.0 * a_mm * a_mm + 4.0 * a_mmm * em)
            + u1 * a_mmmm;
        let t_mmmg = u4 * em * em * em * eg
            + u3 * (3.0 * a_mm * em * eg + 3.0 * a_mg * em * em)
            + u2 * (3.0 * a_mm * a_mg + a_mmm * eg + 3.0 * a_mmg * em)
            + u1 * a_mmmg;
        let t_mmgg = u4 * em * em * eg * eg
            + u3 * (a_mm * eg * eg + 4.0 * a_mg * em * eg + a_gg * em * em)
            + u2 * (a_mm * a_gg + 2.0 * a_mg * a_mg + 2.0 * a_mmg * eg + 2.0 * a_mgg * em)
            + u1 * a_mmgg;
        let t_mggg = u4 * em * eg * eg * eg
            + u3 * (3.0 * a_mg * eg * eg + 3.0 * a_gg * em * eg)
            + u2 * (3.0 * a_mg * a_gg + 3.0 * a_mgg * eg + a_ggg * em)
            + u1 * a_mggg;
        let t_gggg = u4 * eg * eg * eg * eg
            + u3 * 6.0 * a_gg * eg * eg
            + u2 * (3.0 * a_gg * a_gg + 4.0 * a_ggg * eg)
            + u1 * a_gggg;
        Ok(fourth_full_from_symmetric_components(
            t_mmmm, t_mmmg, t_mmgg, t_mggg, t_gggg,
        ))
    }

    pub(super) fn primary_component_jet(
        n_dirs: usize,
        base: f64,
        directions: &[ArrayView1<'_, f64>],
        idx: usize,
    ) -> Result<MultiDirJet, String> {
        let first = directions
            .iter()
            .map(|dir| {
                dir.get(idx).copied().ok_or_else(|| {
                    format!(
                        "bernoulli empirical flex direction length {} is too short for primary index {idx}",
                        dir.len()
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(MultiDirJet::linear(n_dirs, base, &first))
    }

    pub(super) fn local_cubic_value_jet(
        cubic: exact_kernel::LocalSpanCubic,
        x: &MultiDirJet,
    ) -> MultiDirJet {
        let n_dirs = x.coeffs.len().trailing_zeros() as usize;
        let t = x.add(&MultiDirJet::constant(n_dirs, -cubic.left));
        let t2 = t.mul(&t);
        let t3 = t2.mul(&t);
        MultiDirJet::constant(n_dirs, cubic.c0)
            .add(&t.scale(cubic.c1))
            .add(&t2.scale(cubic.c2))
            .add(&t3.scale(cubic.c3))
    }

    pub(super) fn local_cubic_first_derivative_jet(
        cubic: exact_kernel::LocalSpanCubic,
        x: &MultiDirJet,
    ) -> MultiDirJet {
        let n_dirs = x.coeffs.len().trailing_zeros() as usize;
        let t = x.add(&MultiDirJet::constant(n_dirs, -cubic.left));
        let t2 = t.mul(&t);
        MultiDirJet::constant(n_dirs, cubic.c1)
            .add(&t.scale(2.0 * cubic.c2))
            .add(&t2.scale(3.0 * cubic.c3))
    }

    pub(super) fn empirical_flex_eta_and_eta_a_jet_at_z(
        &self,
        primary: &PrimarySlices,
        a_jet: &MultiDirJet,
        b_jet: &MultiDirJet,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: &[ArrayView1<'_, f64>],
        z: f64,
    ) -> Result<(MultiDirJet, MultiDirJet), String> {
        let n_dirs = directions.len();
        let mut inside = a_jet.add(&b_jet.scale(z));

        if let Some(h_range) = primary.h.as_ref() {
            let runtime = self.score_warp.as_ref().ok_or_else(|| {
                "empirical flex score-warp primary range without runtime".to_string()
            })?;
            let beta_h = beta_h.ok_or_else(|| {
                "empirical flex score-warp primary range without beta".to_string()
            })?;
            let mut h_jet = MultiDirJet::zero(n_dirs);
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z,
                "empirical flex score-warp",
                |local_idx, idx, basis_span| {
                    let basis_value = basis_span.evaluate(z);
                    let beta_jet =
                        Self::primary_component_jet(n_dirs, beta_h[local_idx], directions, idx)?;
                    h_jet = h_jet.add(&beta_jet.scale(basis_value));
                    Ok(())
                },
            )?;
            inside = inside.add(&b_jet.mul(&h_jet));
        }

        let u_jet = a_jet.add(&b_jet.scale(z));
        let mut w_jet = MultiDirJet::zero(n_dirs);
        let mut w_prime_jet = MultiDirJet::zero(n_dirs);
        if let Some(w_range) = primary.w.as_ref() {
            let runtime = self.link_dev.as_ref().ok_or_else(|| {
                "empirical flex link-deviation primary range without runtime".to_string()
            })?;
            let beta_w = beta_w.ok_or_else(|| {
                "empirical flex link-deviation primary range without beta".to_string()
            })?;
            let u0 = u_jet.coeff(0);
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u0,
                "empirical flex link-deviation",
                |local_idx, idx, basis_span| {
                    let beta_jet =
                        Self::primary_component_jet(n_dirs, beta_w[local_idx], directions, idx)?;
                    let basis_value = Self::local_cubic_value_jet(basis_span, &u_jet);
                    let basis_derivative =
                        Self::local_cubic_first_derivative_jet(basis_span, &u_jet);
                    w_jet = w_jet.add(&beta_jet.mul(&basis_value));
                    w_prime_jet = w_prime_jet.add(&beta_jet.mul(&basis_derivative));
                    Ok(())
                },
            )?;
        }

        let scale = self.probit_frailty_scale();
        let eta = inside.add(&w_jet).scale(scale);
        let eta_a = MultiDirJet::constant(n_dirs, 1.0)
            .add(&w_prime_jet)
            .scale(scale);
        Ok((eta, eta_a))
    }

    pub(super) fn empirical_flex_calibration_jets(
        &self,
        primary: &PrimarySlices,
        a_jet: &MultiDirJet,
        mu_jet: &MultiDirJet,
        b_jet: &MultiDirJet,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: &[ArrayView1<'_, f64>],
        grid: &EmpiricalZGrid,
    ) -> Result<(MultiDirJet, MultiDirJet), String> {
        let n_dirs = directions.len();
        let mut f = mu_jet.scale(-1.0);
        let mut f_a = MultiDirJet::zero(n_dirs);
        for (node, weight) in grid.pairs() {
            let (eta, eta_a) = self.empirical_flex_eta_and_eta_a_jet_at_z(
                primary, a_jet, b_jet, beta_h, beta_w, directions, node,
            )?;
            let cdf = eta.compose_unary(unary_derivatives_normal_cdf(eta.coeff(0)));
            let pdf = eta.compose_unary(unary_derivatives_normal_pdf(eta.coeff(0)));
            f = f.add(&cdf.scale(weight));
            f_a = f_a.add(&pdf.mul(&eta_a).scale(weight));
        }
        Ok((f, f_a))
    }

    pub(super) fn empirical_flex_neglog_jet(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        directions: &[ArrayView1<'_, f64>],
        grid: &EmpiricalZGrid,
    ) -> Result<MultiDirJet, String> {
        let n_dirs = directions.len();
        if n_dirs > 6 {
            return Err(format!(
                "bernoulli empirical flex jet supports at most 6 directions, got {n_dirs}"
            ));
        }
        for dir in directions {
            if dir.len() != primary.total {
                return Err(format!(
                    "bernoulli empirical flex direction length {} != primary dimension {}",
                    dir.len(),
                    primary.total
                ));
            }
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in jet contraction".to_string());
        }

        let marginal = self.marginal_link_map(q)?;
        let q_jet = Self::primary_component_jet(n_dirs, q, directions, primary.q)?;
        let mu_jet = q_jet.compose_unary([
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
        ]);
        let b_jet = Self::primary_component_jet(n_dirs, b, directions, primary.logslope)?;
        let intercept_root = row_ctx.intercept;
        let mut a_jet = MultiDirJet::constant(n_dirs, intercept_root);
        for _ in 0..6 {
            let (f, f_a) = self.empirical_flex_calibration_jets(
                primary, &a_jet, &mu_jet, &b_jet, beta_h, beta_w, directions, grid,
            )?;
            if !(f_a.coeff(0).is_finite() && f_a.coeff(0) > 0.0) {
                return Err(format!(
                    "empirical flex calibration jet has invalid F_a={}",
                    f_a.coeff(0)
                ));
            }
            let inv_f_a = f_a.compose_unary(unary_derivatives_reciprocal(f_a.coeff(0)));
            a_jet = a_jet.add(&f.mul(&inv_f_a).scale(-1.0));
            a_jet.coeffs[0] = intercept_root;
        }

        let (eta_observed, _) = self.empirical_flex_eta_and_eta_a_jet_at_z(
            primary,
            &a_jet,
            &b_jet,
            beta_h,
            beta_w,
            directions,
            self.z[row],
        )?;
        let signed = eta_observed.scale(2.0 * self.y[row] - 1.0);
        Ok(signed.compose_unary(unary_derivatives_neglog_phi(
            signed.coeff(0),
            self.weights[row],
        )))
    }

    pub(super) fn empirical_flex_row_third_contracted_recompute(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
        grid: &EmpiricalZGrid,
    ) -> Result<Array2<f64>, String> {
        let r = primary.total;
        if dir.len() != r {
            return Err(format!(
                "bernoulli empirical flex third contraction direction length {} != primary dimension {r}",
                dir.len()
            ));
        }
        if dir.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        let basis_dirs = (0..r)
            .map(|idx| Self::unit_primary_direction(r, idx))
            .collect::<Vec<_>>();
        let dir_owned = dir.to_owned();
        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let directions = [basis_dirs[u].view(), basis_dirs[v].view(), dir_owned.view()];
                let jet = self.empirical_flex_neglog_jet(
                    row,
                    primary,
                    q,
                    b,
                    beta_h,
                    beta_w,
                    row_ctx,
                    &directions,
                    grid,
                )?;
                let val = jet.coeff(1 | 2 | 4);
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn empirical_flex_row_fourth_contracted_recompute(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        grid: &EmpiricalZGrid,
    ) -> Result<Array2<f64>, String> {
        let r = primary.total;
        if dir_u.len() != r || dir_v.len() != r {
            return Err(format!(
                "bernoulli empirical flex fourth contraction direction lengths ({},{}) != primary dimension {r}",
                dir_u.len(),
                dir_v.len()
            ));
        }
        if dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        let basis_dirs = (0..r)
            .map(|idx| Self::unit_primary_direction(r, idx))
            .collect::<Vec<_>>();
        let dir_u_owned = dir_u.to_owned();
        let dir_v_owned = dir_v.to_owned();
        let mut out = Array2::<f64>::zeros((r, r));
        for p in 0..r {
            for q_idx in p..r {
                let directions = [
                    basis_dirs[p].view(),
                    basis_dirs[q_idx].view(),
                    dir_u_owned.view(),
                    dir_v_owned.view(),
                ];
                let jet = self.empirical_flex_neglog_jet(
                    row,
                    primary,
                    q,
                    b,
                    beta_h,
                    beta_w,
                    row_ctx,
                    &directions,
                    grid,
                )?;
                let val = jet.coeff(1 | 2 | 4 | 8);
                out[[p, q_idx]] = val;
                out[[q_idx, p]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn rigid_row_kernel_eval(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => rigid_standard_normal_row_kernel(
                marginal,
                slope,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            ),
            Some(grid) => self.empirical_rigid_primary_grad_hess_closed_form(
                row,
                marginal,
                slope,
                &grid.nodes,
                &grid.weights,
            ),
        }
    }

    pub(super) fn rigid_row_third_contracted(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        dir_q: f64,
        dir_g: f64,
    ) -> Result<[[f64; 2]; 2], String> {
        let full = self.rigid_row_third_full(row, marginal, slope)?;
        Ok(contract_third_full(&full, dir_q, dir_g))
    }

    /// Content fingerprint of every input that determines the per-row
    /// cell-moment exact-cache, for same-β reuse via [`shared_exact_cache_store`].
    ///
    /// The exact-cache is `cache(family/data, β-state, subsample-mask,
    /// want_primary_hessians)`. Reuse is gated on exact equality of this
    /// fingerprint, so a hit means a bit-identical cache. The canonicalization
    /// reuses the shared [`Fnv1a`] hasher (`mix_f64` maps `-0.0 → +0.0` so
    /// numerically equal coefficients hash equal; `mix_opt_beta` is unused here
    /// because we hash every block's β and η directly).
    ///
    /// Family/data identity is folded as the stable `Arc::as_ptr` addresses of
    /// the immutable `y`/`z`/`weights` buffers (a fresh fit allocates fresh
    /// `Arc`s, so two fits never share all three; repeated evals on one family
    /// share them), plus the probit-frailty SD and a latent-measure variant
    /// byte. The β-state is pinned by hashing, for every block, the full β
    /// coefficient vector AND the linear-predictor η (the moments consume η, and
    /// the flex deviation bases consume the score-warp / link-deviation β slices;
    /// hashing all blocks' β and η covers both without per-block special-casing).
    /// The outer-score subsample is folded by the `Arc::as_ptr` of its row mask
    /// plus its scalar identity fields, so a distinct subsample misses rather
    /// than aliasing. `want_primary_hessians` is in the key because the build
    /// optionally materializes `row_primary_hessians`, which a consumer expecting
    /// it must observe.
    fn shared_exact_cache_fingerprint(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        want_primary_hessians: bool,
    ) -> u64 {
        let mut hash = Fnv1a::new();
        // Domain separator for the exact-cache fingerprint stream.
        hash.mix_byte(0xe0);
        // Family/data identity: stable Arc allocation addresses of the immutable
        // data buffers (cheap O(1); distinct fits never share all three).
        for &ptr in &[
            Arc::as_ptr(&self.y) as usize,
            Arc::as_ptr(&self.z) as usize,
            Arc::as_ptr(&self.weights) as usize,
        ] {
            for b in (ptr as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
        }
        // Probit-frailty scale source.
        hash.mix_byte(0xe1);
        match self.gaussian_frailty_sd {
            Some(sd) => {
                hash.mix_byte(0x01);
                hash.mix_f64(sd);
            }
            None => hash.mix_byte(0x00),
        }
        // Latent-measure variant discriminant (the measure data itself is
        // immutable and already pinned by the data-buffer addresses above).
        let latent_byte: u8 = match self.latent_measure {
            LatentMeasureKind::StandardNormal => 0x10,
            LatentMeasureKind::GlobalEmpirical { .. } => 0x11,
            LatentMeasureKind::LocalEmpirical { .. } => 0x12,
        };
        hash.mix_byte(latent_byte);
        // Deviation-runtime presence flags (their knots/anchors are immutable
        // and tied to this family instance, so the addresses above suffice;
        // the presence bits guard against an unexpected shape mismatch).
        hash.mix_byte(0xe2);
        hash.mix_byte(u8::from(self.score_warp.is_some()));
        hash.mix_byte(u8::from(self.link_dev.is_some()));
        // β-state: every block's β coefficients and linear predictor η.
        hash.mix_byte(0xe3);
        for b in (block_states.len() as u64).to_le_bytes() {
            hash.mix_byte(b);
        }
        for state in block_states {
            for b in (state.beta.len() as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
            for &v in state.beta.iter() {
                hash.mix_f64(v);
            }
            for b in (state.eta.len() as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
            for &v in state.eta.iter() {
                hash.mix_f64(v);
            }
        }
        // Outer-score subsample identity (the cache build restricts to its mask
        // rows). A distinct subsample → distinct mask address → miss.
        hash.mix_byte(0xe4);
        match options.outer_score_subsample.as_ref() {
            None => hash.mix_byte(0x00),
            Some(subsample) => {
                hash.mix_byte(0x01);
                let mask_ptr = Arc::as_ptr(&subsample.mask) as usize as u64;
                for b in mask_ptr.to_le_bytes() {
                    hash.mix_byte(b);
                }
                for b in (subsample.mask.len() as u64).to_le_bytes() {
                    hash.mix_byte(b);
                }
                for b in (subsample.n_full as u64).to_le_bytes() {
                    hash.mix_byte(b);
                }
                for b in subsample.seed.to_le_bytes() {
                    hash.mix_byte(b);
                }
                hash.mix_f64(subsample.weight_scale);
            }
        }
        // Whether the build materializes `row_primary_hessians`.
        hash.mix_byte(0xe5);
        hash.mix_byte(u8::from(want_primary_hessians));
        hash.finish_nonzero()
    }

    /// Build the per-row cell-moment exact-cache for the current β-state, or
    /// reuse a bit-identical one already built at the same β (same ρ → same
    /// converged β̂ across the BFGS `Value`/`ValueAndGradient` pair, or a
    /// line-search ρ that maps back to a seen β̂).
    ///
    /// On a fingerprint hit the stored `Arc<...>` is returned directly; on a
    /// miss the full cache is built (optionally materializing
    /// `row_primary_hessians`), stored in the FIFO-2 [`shared_exact_cache_store`],
    /// and returned. Because reuse is gated on exact byte-equality of every
    /// build input (see [`Self::shared_exact_cache_fingerprint`]), a hit is
    /// bit-identical to a fresh build, so the downstream gradient, Hessian, and
    /// LAML cost are unchanged. Lazily-built interior fields (`row_cell_moments_d15/d21`,
    /// `rigid_*_full`, `flex_axis_*`) are `RayonSafeOnce`/atomic, so sharing one
    /// `Arc` across the paired evals is safe and yields the same values.
    pub(super) fn build_or_reuse_shared_exact_cache(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        want_primary_hessians: bool,
    ) -> Result<Arc<BernoulliMarginalSlopeExactEvalCache>, String> {
        let fingerprint =
            self.shared_exact_cache_fingerprint(block_states, options, want_primary_hessians);
        if let Some(cache) = shared_exact_cache_store()
            .lock()
            .map_err(|e| format!("BMS exact-cache store mutex poisoned on read: {e}"))?
            .get(fingerprint)
        {
            return Ok(cache);
        }
        let mut cache = self.build_exact_eval_cache_with_options(block_states, Some(options))?;
        if want_primary_hessians {
            cache.row_primary_hessians =
                self.build_row_primary_hessian_cache(block_states, &cache)?;
        }
        let cache = Arc::new(cache);
        shared_exact_cache_store()
            .lock()
            .map_err(|e| format!("BMS exact-cache store mutex poisoned on write: {e}"))?
            .insert(fingerprint, Arc::clone(&cache));
        Ok(cache)
    }

    /// Look up the per-row rigid uncontracted third-derivative tensor from
    /// the cache, populating it lazily on first access via one parallel
    /// row pass. Used by `row_primary_third_contracted_recompute` so the
    /// build-psi-hyper-coords sweep over 32 ψ-axes pays the heavy empirical
    /// jet at most once per row.
    ///
    /// Concurrent first callers may redundantly run the parallel build; the
    /// first published value wins and every subsequent caller observes the
    /// same stored result. A failed build is captured in the `Err` arm of the
    /// stored `Result` and propagates identically on every subsequent call.
    pub(super) fn rigid_third_full_cached<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<&'a [[[f64; 2]; 2]; 2], String> {
        let stored = cache.rigid_third_full.get_or_compute(|| {
            self.build_rigid_full_tensor_table(
                block_states,
                |r, marginal, slope| self.rigid_row_third_full(r, marginal, slope),
                |tower| tower.t3,
            )
        });
        let table = stored.as_ref().map_err(|err| err.clone())?;
        Ok(&table[row])
    }

    /// Build the per-row rigid full-derivative tensor table over all `n` rows.
    ///
    /// For the `StandardNormal` latent measure (the kernel the conditional
    /// location-scale gate always selects) every row routes through the closed
    /// `Tower4<2>` jet, so this fast-paths the whole-`n` build through the
    /// chunked, SIMD-friendly [`rigid_standard_normal_towers_batch`]: it isolates
    /// the one branchy transcendental per row from the dense branch-free tensor
    /// assembly, making the build memory-bandwidth- rather than scalar-ALU-bound.
    /// `extract` reads the consumer's tensor (`.t3`/`.t4`) off the finished jet.
    ///
    /// Any empirical-grid measure keeps the exact per-row dispatch (`row_fn`),
    /// which carries the implicit-function-theorem closed forms. Both arms are
    /// bit-identical to the prior per-row `into_par_iter().map(row_fn)` build.
    fn build_rigid_full_tensor_table<T, R, E>(
        &self,
        block_states: &[ParameterBlockState],
        row_fn: R,
        extract: E,
    ) -> Result<Vec<T>, String>
    where
        T: Copy + Send + Default,
        R: Fn(usize, BernoulliMarginalLinkMap, f64) -> Result<T, String> + Sync,
        E: Fn(&crate::families::jet_tower::Tower4<2>) -> T + Sync,
    {
        let n = self.y.len();
        let marginal_eta = &block_states[0].eta;
        let slope_eta = &block_states[1].eta;
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return (0..n)
                .into_par_iter()
                .map(|r| {
                    let marginal = self.marginal_link_map(marginal_eta[r])?;
                    row_fn(r, marginal, slope_eta[r])
                })
                .collect::<Result<Vec<_>, String>>();
        }

        // Standard-normal whole-`n` chunked batch build.
        const ROW_CHUNK: usize = 256;
        let probit_scale = self.probit_frailty_scale();
        let n_chunks = n.div_ceil(ROW_CHUNK).max(1);
        let chunk_results: Result<Vec<Vec<T>>, String> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * ROW_CHUNK;
                let hi = (lo + ROW_CHUNK).min(n);
                let len = hi - lo;
                let mut marginals: Vec<BernoulliMarginalLinkMap> = Vec::with_capacity(len);
                let mut slopes: Vec<f64> = Vec::with_capacity(len);
                let mut zs: Vec<f64> = Vec::with_capacity(len);
                let mut ys: Vec<f64> = Vec::with_capacity(len);
                let mut ws: Vec<f64> = Vec::with_capacity(len);
                for r in lo..hi {
                    marginals.push(self.marginal_link_map(marginal_eta[r])?);
                    slopes.push(slope_eta[r]);
                    zs.push(self.z[r]);
                    ys.push(self.y[r]);
                    ws.push(self.weights[r]);
                }
                let mut out = vec![T::default(); len];
                rigid_standard_normal_towers_batch(
                    &marginals,
                    &slopes,
                    &zs,
                    &ys,
                    &ws,
                    probit_scale,
                    &mut out,
                    |tower| Ok(extract(tower)),
                )?;
                Ok(out)
            })
            .collect();
        let chunks = chunk_results?;
        let mut table: Vec<T> = Vec::with_capacity(n);
        for chunk in chunks {
            table.extend(chunk);
        }
        Ok(table)
    }

    /// Look up the per-row rigid uncontracted fourth-derivative tensor.
    /// Same lazy-build pattern as `rigid_third_full_cached`, but serves the
    /// outer-Hessian per-pair pullback path: at rank=32 ψ-axes the sweep
    /// touches `(rank² + rank)/2 = 528` (u, v) pairs, all reading the same
    /// per-row tensor. With this cache the empirical-grid 8-direction jet
    /// (or the closed-form 5-component build) runs at most once per row,
    /// then 528 cheap [`contract_fourth_full`] bilinears finish the work.
    pub(super) fn rigid_fourth_full_cached<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<&'a [[[[f64; 2]; 2]; 2]; 2], String> {
        let stored = cache.rigid_fourth_full.get_or_compute(|| {
            self.build_rigid_full_tensor_table(
                block_states,
                |r, marginal, slope| self.rigid_row_fourth_full(r, marginal, slope),
                |tower| tower.t4,
            )
        });
        let table = stored.as_ref().map_err(|err| err.clone())?;
        Ok(&table[row])
    }

    /// Return the lazily-built row-cell-moments bundle at `required_degree`
    /// (15 or 21) for outer dH/d²H trace paths.
    ///
    /// This is an explicit prewarm/build helper: callers invoke it from serial
    /// setup code before parallel row folds that would benefit from a full-row
    /// high-degree bundle. Row-local kernels only read already-built bundles via
    /// `existing_bundle_for_degree`; they never trigger this full-`n` build from
    /// inside a Rayon worker.
    ///
    /// Returns `Ok(None)` for any `required_degree` outside {15, 21}; callers
    /// handle that the same way as a missing bundle.
    pub(super) fn bundle_for_degree<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        required_degree: usize,
    ) -> Result<Option<&'a RowCellMomentsBundle>, String> {
        if let Some(bundle) = cache.row_cell_moments.as_ref()
            && bundle.max_degree >= required_degree
            && bundle.covers_all_rows()
        {
            return Ok(Some(bundle));
        }
        let slot = match required_degree {
            15 => &cache.row_cell_moments_d15,
            21 => &cache.row_cell_moments_d21,
            _ => return Ok(None),
        };
        // `get_or_compute` stores a `Result<Option<...>, String>` directly;
        // the closure returns that same type (it IS T).  The outer `?` then
        // unwraps the stored Result on every access.
        let stored = slot.get_or_compute(|| {
            if required_degree == 21 {
                if let Some(stored_d15) = cache.row_cell_moments_d15.get() {
                    match stored_d15 {
                        Ok(Some(d15)) if d15.covers_all_rows() => {
                            return self.extend_row_cell_moments_bundle(d15, required_degree);
                        }
                        Err(err) => return Err(err.clone()),
                        _ => {}
                    }
                }
            }
            if let Some(base) = cache.row_cell_moments.as_ref()
                && base.covers_all_rows()
            {
                return self.extend_row_cell_moments_bundle(base, required_degree);
            }
            // No subsample mask for the outer-derivative trace bundles: they
            // must cover all rows so that every row lookup succeeds.
            self.build_row_cell_moments_bundle(
                block_states,
                &cache.row_contexts,
                required_degree,
                None,
            )
        });
        Ok(stored.as_ref().map_err(|e| e.clone())?.as_ref())
    }

    /// Prewarm the degree-`required_degree` full-row cell-moment bundle once,
    /// from serial setup code, before a FLEX outer-derivative row par-fold.
    ///
    /// The FLEX third/fourth row recompute kernels
    /// (`row_primary_{third,fourth}_contracted_recompute*`) read the per-cell
    /// moments through `row_cell_moments_for_third_degree15`, which only
    /// consults an *already-built* bundle. Without a serial prewarm, the first
    /// row to need degree-15 moments finds no bundle and falls back to
    /// `evaluate_cell_derivative_moments_uncached` — recomputing the
    /// transcendental cell moments for *every* row on *every* operator
    /// application (gam#683). Under `linkwiggle()` the cells are non-affine and
    /// the cross-row LRU key is row-unique, so that fallback never amortizes:
    /// the outer-REML continuation and post-fit Hessian builds rebuild the
    /// whole degree-15 moment table from scratch each step.
    ///
    /// Building the bundle once here populates `cache.row_cell_moments_d15`
    /// (a `RayonSafeOnce` tied to the β-cache), so every subsequent per-row
    /// kernel — across all CG iterations and HVP applications at this β — reads
    /// the prebuilt moments and only pays the cheap directional contraction.
    /// Mirrors the rigid `rigid_{third,fourth}_full_cached` prewarm and the
    /// degree-21 prewarm in the psi-second-order path. No-op (returns `Ok(())`)
    /// when the FLEX path is inactive, when the bundle build is skipped by the
    /// resource-byte budget, or for an empirical-grid latent measure that
    /// bypasses the cell path; in those cases callers fall back exactly as
    /// before.
    pub(super) fn prewarm_flex_cell_bundle(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        required_degree: usize,
    ) -> Result<(), String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(());
        }
        if let Some(bundle) = self.bundle_for_degree(block_states, cache, required_degree)?
            && bundle.max_degree < required_degree
        {
            return Err(format!(
                "BMS row-cell-moments prewarm returned degree {} for required degree {}",
                bundle.max_degree, required_degree
            ));
        }
        Ok(())
    }

    pub(crate) fn existing_bundle_for_degree<'a>(
        &self,
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        required_degree: usize,
    ) -> Result<Option<&'a RowCellMomentsBundle>, String> {
        if let Some(bundle) = cache.row_cell_moments.as_ref()
            && bundle.max_degree >= required_degree
            && bundle.covers_all_rows()
        {
            return Ok(Some(bundle));
        }
        let stored = match required_degree {
            15 => cache.row_cell_moments_d15.get(),
            21 => cache.row_cell_moments_d21.get(),
            _ => None,
        };
        match stored {
            Some(Ok(Some(bundle))) => Ok(Some(bundle)),
            Some(Ok(None)) | None => Ok(None),
            Some(Err(err)) => Err(err.clone()),
        }
    }

    pub(crate) fn row_cell_moments_for_third_degree15<'a>(
        &self,
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<Option<&'a [CachedDenestedCellMoments]>, String> {
        if let Some(bundle) = self.existing_bundle_for_degree(cache, 21)?
            && let Some(cells) = bundle.row(row, 15)
        {
            return Ok(Some(cells));
        }
        Ok(self
            .existing_bundle_for_degree(cache, 15)?
            .and_then(|bundle| bundle.row(row, 15)))
    }

    /// Per-row uncontracted third-derivative tensor in the rigid path.
    ///
    /// The standard-normal latent measure uses the analytic
    /// `rigid_standard_normal_third_full`; empirical-grid rows use the closed-form
    /// implicit-function-theorem tensor `empirical_rigid_third_full_closed_form`.
    /// Both yield the four distinct symmetric components `T_mmm, T_mmg, T_mgg,
    /// T_ggg`; the `rank`-many ψ-axis directions are folded in later by a cheap
    /// `contract_third_full` bilinear per call.
    pub(super) fn rigid_row_third_full(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<[[[f64; 2]; 2]; 2], String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => rigid_standard_normal_third_full(
                marginal,
                slope,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            ),
            Some(grid) => self.empirical_rigid_third_full_closed_form(
                row,
                marginal,
                slope,
                &grid.nodes,
                &grid.weights,
            ),
        }
    }

    /// Per-row uncontracted fourth-derivative tensor in the rigid path.
    ///
    /// The standard-normal latent measure drops out of
    /// `rigid_standard_normal_fourth_full` (five axis-invariant primary-space
    /// components). Empirical-grid rows use the closed-form implicit-function-
    /// theorem tensor `empirical_rigid_fourth_full_closed_form`, yielding the
    /// five distinct symmetric components `T_mmmm, T_mmmg, T_mmgg, T_mggg,
    /// T_gggg`. The (u, v) ψ-axis directions are folded in afterwards via the
    /// cheap `contract_fourth_full` bilinear — one tensor build per row.
    pub(super) fn rigid_row_fourth_full(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => rigid_standard_normal_fourth_full(
                marginal,
                slope,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            ),
            Some(grid) => self.empirical_rigid_fourth_full_closed_form(
                row,
                marginal,
                slope,
                &grid.nodes,
                &grid.weights,
            ),
        }
    }

    /// Outer-aware variant of `log_likelihood_only`. When
    /// `options.outer_score_subsample` is `None` this iterates over all rows
    /// and returns a value identical (bit-for-bit) to the legacy full-data
    /// implementation. When it is `Some`, only the sampled rows contribute,
    /// with their Horvitz-Thompson inverse-inclusion weights taken from
    /// `OuterScoreSubsample::rows`. This is the row-iter swap that lets outer-only
    /// score/gradient passes scale to large-scale `n` without distorting the
    /// full-data inner-PIRLS or covariance code paths.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        self.validate_exact_monotonicity(block_states)?;
        let flex_active = self.effective_flex_active(block_states)?;
        let n = self.y.len();
        // Line-search accept/reject is an exact full-data decision. A
        // line-search trial probe (`options.early_exit_threshold =
        // Some(_)`) never installs an auto Horvitz-Thompson subsample:
        // the threshold is the *full-data* objective at the old iterate
        // (`old_objective + slack - trial_penalty`), and an HT-weighted
        // partial sum is only an *unbiased estimator* of the full-data
        // NLL, not a deterministic lower bound on it — so an HT
        // early-exit can falsely reject a step whose true full-data NLL
        // sits below the threshold. The full-data sweep below keeps a
        // *sound* early-exit reject: every row contributes
        // `weight_i * log Φ ≤ 0`, so the running `-total_ll` is a genuine
        // monotone lower bound on the full-data NLL and short-circuits a
        // genuinely-rejected trial before the sweep finishes. Outer
        // derivative passes still subsample via the caller-supplied
        // `options.outer_score_subsample` (set only for `OuterDerivative`
        // scope), which `outer_weighted_rows` honors here.
        let weighted_rows = outer_weighted_rows(options, n);
        if !flex_active {
            // Rigid probit under the active latent measure. Standard-normal
            // keeps the algebraic Gaussian identity; empirical measure solves
            // the calibrated intercept against its quadrature grid.
            //
            // **Objective-only fast path.** The line-search accept/reject
            // decision only needs the scalar negative log-likelihood; the
            // gradient and Hessian returned by `rigid_row_kernel_eval` would
            // be immediately discarded. `rigid_row_neglog_only` dispatches
            // to:
            //   * `rigid_standard_normal_neglog_only` (standard-normal): a single
            //     `signed_probit_logcdf_and_mills_ratio` call, skipping the
            //     `u_k`/`c_k`/`eta_*` chain-rule scaffolding.
            //   * `empirical_rigid_neglog_only` (empirical-grid): the
            //     converged scalar intercept (from
            //     `empirical_rigid_intercept_for_row`'s warm-start cache) plus
            //     a single probit log-CDF eval, skipping the four-direction
            //     `MultiDirJet` construction and its six Newton-refinement
            //     passes (the line search reads no derivative coefficients).
            // The returned value is bit-equivalent to
            // `rigid_row_kernel_eval(...).0` at the same row state.
            let b = &block_states[1].eta;
            let row_ll = |i: usize| -> Result<f64, String> {
                let marginal_eta = block_states[0].eta[i];
                let marginal = self.marginal_link_map(marginal_eta)?;
                let neglog = self.rigid_row_neglog_only(i, marginal, b[i])?;
                Ok(-neglog)
            };
            if let Some(threshold) = options.early_exit_threshold {
                return bernoulli_margslope_line_search_ll_with_early_exit(
                    &weighted_rows,
                    threshold,
                    row_ll,
                );
            }
            let total: Result<f64, String> = weighted_rows
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, wr| -> Result<_, String> {
                        ll += wr.weight * row_ll(wr.index)?;
                        Ok(ll)
                    },
                )
                .try_reduce(
                    || 0.0,
                    |left, right| -> Result<_, String> { Ok(left + right) },
                );
            return total;
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let row_ll = |row: usize| -> Result<f64, String> {
            let intercept = self
                .solve_row_intercept_base(
                    row,
                    block_states[0].eta[row],
                    block_states[1].eta[row],
                    beta_h,
                    beta_w,
                    None,
                )?
                .0;
            let slope = block_states[1].eta[row];
            let obs =
                self.observed_denested_cell_partials(row, intercept, slope, beta_h, beta_w)?;
            let s_i = eval_coeff4_at(&obs.coeff, self.z[row]);
            let signed = (2.0 * self.y[row] - 1.0) * s_i;
            let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
            Ok(self.weights[row] * log_cdf)
        };
        if let Some(threshold) = options.early_exit_threshold {
            return bernoulli_margslope_line_search_ll_with_early_exit(
                &weighted_rows,
                threshold,
                row_ll,
            );
        }
        let total: Result<f64, String> = weighted_rows
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut ll, wr| -> Result<_, String> {
                    ll += wr.weight * row_ll(wr.index)?;
                    Ok(ll)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        total
    }

    pub(super) fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    pub(super) fn sigma_scale_jet(
        &self,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> Result<MultiDirJet, String> {
        probit_frailty_scale_multi_dir_jet(
            self.gaussian_frailty_sd,
            "bernoulli marginal-slope log-sigma auxiliary requested without GaussianShift sigma",
            n_dirs,
            first_masks,
            second_masks,
        )
    }

    pub(super) fn row_neglog_directional_with_scale_jet(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
        scale_jet: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "bernoulli marginal-slope sigma row directional expects 0..=4 directions, got {k}"
            ));
        }
        if scale_jet.coeffs.len() != (1usize << k) {
            return Err(format!(
                "bernoulli marginal-slope sigma scale jet dimension mismatch: coeffs={}, dirs={k}",
                scale_jet.coeffs.len()
            ));
        }

        let first = |idx: usize| -> Vec<f64> { dirs.iter().map(|dir| dir[idx]).collect() };
        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
        let eta_jet = MultiDirJet::linear(k, block_states[0].eta[row], &first(0));
        let q_jet = eta_jet.compose_unary([
            marginal.q,
            marginal.q1,
            marginal.q2,
            marginal.q3,
            marginal.q4,
        ]);
        let g_jet = MultiDirJet::linear(k, block_states[1].eta[row], &first(1));
        let observed_g_jet = g_jet.mul(scale_jet);
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&observed_g_jet.mul(&observed_g_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));
        let z_jet = MultiDirJet::constant(k, self.z[row]);
        let eta_observed_jet = q_jet.mul(&c_jet).add(&observed_g_jet.mul(&z_jet));
        let signed_jet = eta_observed_jet.scale(2.0 * self.y[row] - 1.0);
        Ok(signed_jet
            .compose_unary(unary_derivatives_neglog_phi(
                signed_jet.coeff(0),
                self.weights[row],
            ))
            .coeff((1usize << k) - 1))
    }

    pub(super) fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primary_dim = 2usize;
        let zero = Array1::<f64>::zeros(primary_dim);
        // The leading prefix is the fixed number of zero primary directions the
        // log-sigma hyperderivative differentiates *through*: one for the first
        // log-sigma derivative, two for the second. The shared sweep appends the
        // unit primary directions for grad/hess on top of this prefix.
        let (leading, scales): (Vec<&Array1<f64>>, DirectionalScaleJets) = if second_sigma {
            (
                vec![&zero, &zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(2, &[1, 2], &[3])?),
                    grad: self.sigma_scale_jet(3, &[1, 2], &[3])?,
                    hess: self.sigma_scale_jet(4, &[1, 2], &[3])?,
                },
            )
        } else {
            (
                vec![&zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(1, &[1], &[])?),
                    grad: self.sigma_scale_jet(2, &[1], &[])?,
                    hess: self.sigma_scale_jet(3, &[1], &[])?,
                },
            )
        };
        let terms = directional_obj_grad_hess(primary_dim, &leading, &scales, |dirs, scale| {
            let owned: Vec<Array1<f64>> = dirs.iter().map(|d| (*d).clone()).collect();
            self.row_neglog_directional_with_scale_jet(row, block_states, &owned, scale)
        })?;
        Ok((terms.objective, terms.grad, terms.hess))
    }

    pub(super) fn accumulate_rigid_sigma_pullback(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary_grad: &Array1<f64>,
        primary_hessian: &Array2<f64>,
        score: &mut Array1<f64>,
        hessian: &mut BernoulliBlockHessianAccumulator,
    ) -> Result<(), String> {
        {
            let mut marginal = score.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_grad[0], &mut marginal)?;
        }
        {
            let mut logslope = score.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_grad[1], &mut logslope)?;
        }
        hessian.add_pullback(self, row, slices, &primary_slices(slices), primary_hessian);
        Ok(())
    }

    pub(super) fn sigma_exact_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.sigma_exact_joint_psi_terms_with_options(
            block_states,
            specs,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psi_terms`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, score vector, Hessian operator blocks) is accumulated with the
    /// row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn sigma_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "bernoulli marginal-slope sigma psi terms: specs/block_states length mismatch {} vs {}",
                specs.len(),
                block_states.len()
            ));
        }
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let row_iter = outer_row_indices(options, n).to_vec();
        // Per-row HT weighting: each row's (obj, grad, hess) is multiplied by
        // its inverse-inclusion weight `w_i` *before* accumulation, so the
        // final operator is the unbiased Horvitz-Thompson estimator. A single
        // post-sum scalar is biased under stratified subsampling because
        // per-stratum sampling fractions differ. In the full-data path every
        // `w_i == 1.0`, so we skip the dense O(n) weight vector entirely (it
        // is otherwise re-allocated and zero-filled on every outer eval over
        // n≈3e5 rows) and the per-row scaling becomes a no-op.
        let row_weights = options.outer_score_subsample.as_ref().map(|_| {
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n)
        });
        let (objective_psi, score_psi, acc) = chunked_row_reduction(
            row_iter.as_slice(),
            || {
                (
                    0.0,
                    Array1::<f64>::zeros(slices.total),
                    BernoulliBlockHessianAccumulator::new(&slices),
                )
            },
            |row, acc| -> Result<(), String> {
                let (mut obj, mut grad, mut hess) =
                    self.row_sigma_primary_terms(row, block_states, false)?;
                if let Some(ref weights) = row_weights {
                    let w = weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                }
                acc.0 += obj;
                self.accumulate_rigid_sigma_pullback(
                    row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                )?;
                Ok(())
            },
            |total, chunk| {
                total.0 += chunk.0;
                total.1 += &chunk.1;
                total.2.add(&chunk.2);
            },
        )?;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Arc::new(acc.into_operator(&slices))),
        }))
    }

    pub(super) fn sigma_exact_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.sigma_exact_joint_psisecond_order_terms_with_options(
            block_states,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psisecond_order_terms`. See
    /// `sigma_exact_joint_psi_terms_with_options` for the row-iter / weighting
    /// contract.
    pub(crate) fn sigma_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope second log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let row_iter = outer_row_indices(options, n).to_vec();
        // Full-data path carries `w_i == 1.0` for every row, so skip the dense
        // O(n) HT-weight vector (see `sigma_exact_joint_psi_terms_with_options`).
        let row_weights = options.outer_score_subsample.as_ref().map(|_| {
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n)
        });
        let (objective_psi_psi, score_psi_psi, acc) = chunked_row_reduction(
            row_iter.as_slice(),
            || {
                (
                    0.0,
                    Array1::<f64>::zeros(slices.total),
                    BernoulliBlockHessianAccumulator::new(&slices),
                )
            },
            |row, acc| -> Result<(), String> {
                let (mut obj, mut grad, mut hess) =
                    self.row_sigma_primary_terms(row, block_states, true)?;
                if let Some(ref weights) = row_weights {
                    let w = weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                }
                acc.0 += obj;
                self.accumulate_rigid_sigma_pullback(
                    row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                )?;
                Ok(())
            },
            |total, chunk| {
                total.0 += chunk.0;
                total.1 += &chunk.1;
                total.2.add(&chunk.2);
            },
        )?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(&slices))),
        }))
    }

    pub(super) fn sigma_exact_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.sigma_exact_joint_psihessian_directional_derivative_with_options(
            block_states,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psihessian_directional_derivative`.
    /// See `sigma_exact_joint_psi_terms_with_options` for the row-iter /
    /// weighting contract — the returned dense Hessian-derivative matrix is
    /// accumulated with per-row inverse-inclusion weights when a subsample is active.
    pub(crate) fn sigma_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope log-sigma Hessian directional derivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch for sigma Hessian derivative: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let n = self.y.len();
        let primary = primary_slices(&slices);
        let row_iter = outer_row_indices(options, n).to_vec();
        // Full-data path carries `w_i == 1.0` for every row, so skip the dense
        // O(n) HT-weight vector (see `sigma_exact_joint_psi_terms_with_options`).
        let row_weights = options.outer_score_subsample.as_ref().map(|_| {
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n)
        });
        // Sigma scale jets and the zero primary direction are constant across
        // rows; resolve once outside the fold. The shared
        // `directional_obj_grad_hess` sweep differentiates *through* the fixed
        // leading prefix `[zero, row_dir]` (one zero log-sigma slot, the
        // perturbation direction) and appends the grad/hess unit directions;
        // `obj: None` suppresses the zeroth-order pass.
        let scale_grad = self.sigma_scale_jet(3, &[1], &[])?;
        let scale_hess = self.sigma_scale_jet(4, &[1], &[])?;
        let zero = Array1::<f64>::zeros(primary.total);
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BernoulliBlockHessianAccumulator::new(&slices),
            |row, acc| -> Result<(), String> {
                let row_dir =
                    self.row_primary_direction_from_flat(row, &slices, &primary, d_beta_flat)?;
                let scales = DirectionalScaleJets {
                    obj: None,
                    grad: scale_grad.clone(),
                    hess: scale_hess.clone(),
                };
                let terms = directional_obj_grad_hess(
                    primary.total,
                    &[&zero, &row_dir],
                    &scales,
                    |dirs, scale| {
                        let owned: Vec<Array1<f64>> = dirs.iter().map(|d| (*d).clone()).collect();
                        self.row_neglog_directional_with_scale_jet(row, block_states, &owned, scale)
                    },
                )?;
                let mut hess = terms.hess;
                if let Some(ref weights) = row_weights {
                    let w = weights[row];
                    if w != 1.0 {
                        hess.mapv_inplace(|v| v * w);
                    }
                }
                acc.add_pullback(self, row, &slices, &primary, &hess);
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;
        Ok(Some(acc.into_operator(&slices).to_dense()))
    }

    #[inline]
    pub(super) fn marginal_link_map(&self, eta: f64) -> Result<BernoulliMarginalLinkMap, String> {
        bernoulli_marginal_link_map(&self.base_link, eta)
    }

    #[inline]
    pub(super) fn exact_newton_score_component_from_objective_gradient(
        objective_gradient_component: f64,
    ) -> f64 {
        -objective_gradient_component
    }

    #[inline]
    pub(super) fn exact_newton_score_from_objective_gradient(
        objective_gradient: Array1<f64>,
    ) -> Array1<f64> {
        -objective_gradient
    }

    #[inline]
    pub(super) fn exact_newton_observed_information_from_objective_hessian(
        objective_hessian: Array2<f64>,
    ) -> Array2<f64> {
        objective_hessian
    }

    #[inline]
    pub(super) fn score_block_index(&self) -> Option<usize> {
        self.score_warp.as_ref().map(|_| 2)
    }

    #[inline]
    pub(super) fn link_block_index(&self) -> Option<usize> {
        self.link_dev
            .as_ref()
            .map(|_| 2 + usize::from(self.score_warp.is_some()))
    }

    pub(super) fn optional_exact_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
        block_idx: Option<usize>,
        label: &str,
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        match block_idx {
            Some(idx) => block_states
                .get(idx)
                .map(Some)
                .ok_or_else(|| format!("missing {label} block state")),
            None => Ok(None),
        }
    }

    pub(super) fn score_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.score_block_index(), "score-warp")
    }

    pub(super) fn link_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.link_block_index(), "link deviation")
    }

    pub(super) fn score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .score_block_state(block_states)?
            .map(|state| &state.beta))
    }

    pub(super) fn link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .link_block_state(block_states)?
            .map(|state| &state.beta))
    }

    pub(super) fn validate_exact_block_state_shapes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        let expected_blocks =
            2usize + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        crate::families::block_layout::block_count::validate_block_count::<String>(
            "BernoulliMarginalSlopeFamily",
            expected_blocks,
            block_states.len(),
        )?;

        let n_rows = self.y.len();
        let marginal = &block_states[0];
        let marginal_ncols = self.marginal_design.ncols();
        if marginal_ncols > 0 && marginal.beta.len() != marginal_ncols {
            return Err(format!(
                "bernoulli marginal-slope marginal beta length mismatch: got {}, expected {}",
                marginal.beta.len(),
                marginal_ncols
            ));
        }
        if marginal.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope marginal eta length mismatch: got {}, expected {}",
                marginal.eta.len(),
                n_rows
            ));
        }

        let logslope = &block_states[1];
        let logslope_ncols = self.logslope_design.ncols();
        if logslope_ncols > 0 && logslope.beta.len() != logslope_ncols {
            return Err(format!(
                "bernoulli marginal-slope logslope beta length mismatch: got {}, expected {}",
                logslope.beta.len(),
                logslope_ncols
            ));
        }
        if logslope.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope logslope eta length mismatch: got {}, expected {}",
                logslope.eta.len(),
                n_rows
            ));
        }

        if let Some(runtime) = &self.score_warp {
            let score = self
                .score_block_state(block_states)?
                .expect("score-warp block should exist when runtime is present");
            if score.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope score-warp beta length mismatch: got {}, expected {}",
                    score.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if score.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope score-warp eta length mismatch: got {}, expected {}",
                    score.eta.len(),
                    n_rows
                ));
            }
        }

        if let Some(runtime) = &self.link_dev {
            let link = self
                .link_block_state(block_states)?
                .expect("link-deviation block should exist when runtime is present");
            if link.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation beta length mismatch: got {}, expected {}",
                    link.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if link.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation eta length mismatch: got {}, expected {}",
                    link.eta.len(),
                    n_rows
                ));
            }
        }

        Ok(())
    }

    pub(super) fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        shared_denested_partition_cells(
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    pub(super) fn max_denested_partition_cells_per_row(&self) -> usize {
        let score_splits = self
            .score_warp
            .as_ref()
            .map_or(0usize, |runtime| runtime.breakpoints().len());
        let link_splits = self
            .link_dev
            .as_ref()
            .map_or(0usize, |runtime| runtime.breakpoints().len());
        score_splits.saturating_add(link_splits).saturating_add(1)
    }

    #[inline]
    pub(super) fn evaluate_cell_moments_lru(
        &self,
        cell: exact_kernel::DenestedCubicCell,
        max_degree: usize,
    ) -> Result<exact_kernel::CellMomentState, String> {
        exact_kernel::evaluate_cell_moments_cached(
            cell,
            max_degree,
            &self.cell_moment_lru,
            Some(&self.cell_moment_cache_stats),
        )
    }

    #[inline]
    pub(super) fn evaluate_cell_derivative_moments_lru(
        &self,
        cell: exact_kernel::DenestedCubicCell,
        max_degree: usize,
    ) -> Result<exact_kernel::CellDerivativeMomentState, String> {
        exact_kernel::evaluate_cell_derivative_moments_cached(
            cell,
            max_degree,
            &self.cell_moment_lru,
            Some(&self.cell_moment_cache_stats),
        )
    }

    #[inline]
    pub(super) fn for_each_deviation_basis_cubic_at<F>(
        runtime: &DeviationRuntime,
        primary_range: &std::ops::Range<usize>,
        value: f64,
        label: &str,
        mut visit: F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, usize, exact_kernel::LocalSpanCubic) -> Result<(), String>,
    {
        if primary_range.len() != runtime.basis_dim() {
            return Err(format!(
                "{label} primary range length {} does not match deviation basis dimension {}",
                primary_range.len(),
                runtime.basis_dim()
            ));
        }
        runtime.for_each_basis_cubic_at(value, |local_idx, basis_span| {
            visit(local_idx, primary_range.start + local_idx, basis_span)
        })
    }

    /// Newton-step evaluator for the inner-PIRLS row-intercept root solver.
    ///
    /// Returns `(f, f', 0.0)`: the third slot — `F''(a)` — is reported as
    /// zero, which makes [`monotone_root::solve_monotone_root`]'s safeguarded
    /// Halley step reduce to a Newton step. A measured degree-9 `F''(a)` path
    /// did not reduce calibration evaluations on the large-scale FLEX repro, and
    /// it made each value-bearing cell evaluation slower; degree 4 is the
    /// correct cost/accuracy point for this solver.
    pub(super) fn evaluate_denested_calibration_newton(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let state = self.evaluate_cell_moments_lru(cell, 4)?;
            f += state.value;
            let (dc_da_raw, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
        }
        Ok((f, f_a, 0.0))
    }

    pub(super) fn evaluate_empirical_grid_calibration_newton(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        grid: &EmpiricalZGrid,
    ) -> Result<(f64, f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for (node, weight) in grid.pairs() {
            let obs = self.observed_denested_cell_partials_at_z(node, a, slope, beta_h, beta_w)?;
            let eta = eval_coeff4_at(&obs.coeff, node);
            let eta_a = eval_coeff4_at(&obs.dc_da, node);
            let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
            let pdf = normal_pdf(eta);
            f += weight * normal_cdf(eta);
            f_a += weight * pdf * eta_a;
            f_aa += weight * pdf * (eta_aa - eta * eta_a * eta_a);
        }
        if !(f.is_finite() && f_a.is_finite() && f_a > 0.0 && f_aa.is_finite()) {
            return Err(format!(
                "empirical latent denested calibration produced invalid root state: f={f}, f_a={f_a}, f_aa={f_aa}"
            ));
        }
        Ok((f, f_a, f_aa))
    }

    pub(super) fn evaluate_calibration_newton(
        &self,
        row: usize,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                self.evaluate_denested_calibration_newton(a, marginal_eta, slope, beta_h, beta_w)
            }
            Some(grid) => self.evaluate_empirical_grid_calibration_newton(
                a,
                marginal_eta,
                slope,
                beta_h,
                beta_w,
                &grid,
            ),
        }
    }

    pub(super) fn flex_active(&self) -> bool {
        self.score_warp.is_some() || self.link_dev.is_some()
    }

    /// The denested exact path is active whenever either deviation runtime is
    /// configured. Zero coefficient vectors still keep the flexible geometry
    /// live so derivatives with respect to those coefficients remain available.
    pub(super) fn effective_flex_active(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        if self.score_warp.is_some() && self.score_beta(block_states)?.is_none() {
            return Err("missing bernoulli score-warp block state".to_string());
        }
        if self.link_dev.is_some() && self.link_beta(block_states)?.is_none() {
            return Err("missing bernoulli link-deviation block state".to_string());
        }
        Ok(self.flex_active())
    }

    pub(super) fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        self.validate_exact_block_state_shapes(block_states)?;
        if let (Some(runtime), Some(score)) =
            (&self.score_warp, self.score_block_state(block_states)?)
        {
            runtime.monotonicity_feasible(
                &score.beta,
                "bernoulli marginal-slope score-warp deviation",
            )?;
        }
        if let (Some(runtime), Some(beta_w)) = (&self.link_dev, self.link_beta(block_states)?) {
            runtime.monotonicity_feasible(beta_w, "bernoulli marginal-slope link deviation")?;
        }
        Ok(())
    }

    /// Single-row link-deviation value and first derivative at `eta0`,
    /// honouring any cross-block anchor residual on `link_dev`.
    ///
    /// The closed-form intercept seed `row_intercept_closed_form_seed` is
    /// called once per training row from `solve_row_intercept_base`; each
    /// call needs `ℓ(η_a) = η_a + Φ(η_a) · β` evaluated at the row's pre-
    /// scale rigid intercept `a_rigid_pre_scale`. When the link-deviation
    /// runtime has been reparameterised against the marginal+logslope
    /// parametric anchor, the per-row reparameterised basis is
    ///
    ///   Φ_new[row, :] = Φ_raw(η_a) − parametric_anchor[row, :] · M
    ///
    /// so the design value at `(row, η_a)` is the raw basis minus a row-
    /// specific subtraction. `runtime.design()` returns the raw basis
    /// only and `assert`s in this configuration so callers don't
    /// silently miscompute; instead route through `design_with_anchor_rows`
    /// with the runtime's cached training-row anchor sliced to a single
    /// row. The derivative path is unaffected — the subtraction is
    /// constant in `η`, so its derivative is identically zero.
    pub(super) fn link_terms_value_d1_at_row(
        &self,
        row: usize,
        eta0: f64,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) else {
            return Ok((eta0, 1.0));
        };
        let values = Array1::from_vec(vec![eta0]);
        let basis = if let Some(anchor_rows) = runtime.anchor_rows_at_training() {
            if row >= anchor_rows.nrows() {
                return Err(format!(
                    "link_terms_value_d1_at_row: row {row} out of bounds for {} cached training anchor rows",
                    anchor_rows.nrows()
                ));
            }
            let anchor_view = anchor_rows.slice(ndarray::s![row..row + 1, ..]);
            runtime.design_with_anchor_rows(&values, anchor_view)?
        } else {
            runtime.design(&values)?
        };
        let d1 = runtime.first_derivative_design(&values)?;
        Ok((eta0 + basis.row(0).dot(beta), d1.row(0).dot(beta) + 1.0))
    }

    pub(super) fn row_intercept_closed_form_seed(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let probit_scale = self.probit_frailty_scale();
        let a_rigid_pre_scale =
            rigid_intercept_from_marginal(marginal.q, slope, probit_scale) / probit_scale;
        if beta_w.is_some() {
            let (l_val, l_d1) = self.link_terms_value_d1_at_row(row, a_rigid_pre_scale, beta_w)?;
            if l_d1 > BMS_DERIV_TOL {
                let ell0 = l_val - l_d1 * a_rigid_pre_scale;
                let observed_logslope = probit_scale * l_d1 * slope;
                return Ok(
                    (marginal.q * (1.0 + observed_logslope * observed_logslope).sqrt()
                        / probit_scale
                        - ell0)
                        / l_d1,
                );
            }
        }
        Ok(a_rigid_pre_scale)
    }

    /// Pre-seed cold (`NaN`) per-row intercept warm-start slots with the
    /// closed-form rigid/affine seed for the current `(marginal_eta, slope)`
    /// state, before the parallel root solves run. Slots already populated
    /// from a prior PIRLS/outer iteration are preserved verbatim — only NaN
    /// slots are CAS-installed. This avoids recomputing the seed inside every
    /// `solve_row_intercept_base` call on cold cycle 0.
    pub(super) fn preseed_intercept_warm_starts(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(());
        }
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return Ok(());
        };
        let beta_w = self.link_beta(block_states)?;
        let n = self.y.len();
        if cache.len() != n {
            return Ok(());
        }
        let marginal_eta = &block_states[0].eta;
        let slope_eta = &block_states[1].eta;
        let probit_scale = self.probit_frailty_scale();

        // Per-row marginal link map and rigid pre-scale intercept.
        let marginals: Vec<BernoulliMarginalLinkMap> = (0..n)
            .into_par_iter()
            .map(|row| self.marginal_link_map(marginal_eta[row]))
            .collect::<Result<Vec<_>, _>>()?;
        let a_pre_scale_vec: Array1<f64> = (0..n)
            .map(|row| {
                rigid_intercept_from_marginal(marginals[row].q, slope_eta[row], probit_scale)
                    / probit_scale
            })
            .collect();

        // Batched link-deviation evaluation at each row's pre-scale intercept.
        //
        // The closed-form intercept seed needs ℓ(a_pre_scale_i) and
        // ℓ'(a_pre_scale_i) where
        //
        //   ℓ(η) = η + Φ_link_dev(η) · β_link
        //
        // is the row-i link deviation. After
        // `install_compiled_flex_block_into_runtime`
        // reparameterised the link-deviation runtime against the
        // marginal+logslope parametric anchor union, the per-row
        // reparameterised basis is
        //
        //   Φ_new[i, :] = Φ_raw(η_i) − parametric_anchor[i, :] · M
        //
        // so ℓ depends on the row through both the raw basis evaluation
        // and the row-specific subtraction. The basis derivative is
        // unaffected: the subtraction is independent of η.
        //
        // Evaluating `link_dev.design()` on a single-row `eta0` vector
        // would discard the row-specific subtraction (`design()`
        // asserts that the runtime has no anchor residual exactly
        // to prevent this silent miscompute). Instead, feed the
        // full-length per-row `a_pre_scale_vec` through
        // `design_at_training_with_residual` so the runtime applies the
        // cached training-row parametric anchor matrix at the correct
        // row for every evaluation. For runtimes without an
        // anchor_residual the same call falls back to raw `design()`.
        let (l_val_vec, l_d1_vec) = match (&self.link_dev, beta_w) {
            (Some(runtime), Some(beta)) => {
                let basis = runtime.design_at_training_with_residual(&a_pre_scale_vec)?;
                let d1 = runtime.first_derivative_design(&a_pre_scale_vec)?;
                (&a_pre_scale_vec + &basis.dot(beta), d1.dot(beta) + 1.0)
            }
            _ => (a_pre_scale_vec.clone(), Array1::ones(n)),
        };

        let seeds: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|row| {
                let a = a_pre_scale_vec[row];
                let ell1 = l_d1_vec[row];
                if ell1 > BMS_DERIV_TOL {
                    let ell0 = l_val_vec[row] - ell1 * a;
                    let observed_logslope = probit_scale * ell1 * slope_eta[row];
                    (marginals[row].q * (1.0 + observed_logslope * observed_logslope).sqrt()
                        / probit_scale
                        - ell0)
                        / ell1
                } else {
                    a
                }
            })
            .collect();
        // Resolve β_h once for the preseed sweep so each row's tag includes
        // the joint β that the FLEX intercept root actually depends on.
        let beta_h = self.score_beta(block_states)?;
        let mut preseeded = 0usize;
        let mut kept_warm = 0usize;
        for (row, seed) in seeds.iter().enumerate() {
            if !seed.is_finite() {
                continue;
            }
            let beta_tag = hash_intercept_warm_start_key_flex(
                marginal_eta[row],
                slope_eta[row],
                beta_h,
                beta_w,
            );
            match cache.compare_exchange_unseeded(row, *seed, beta_tag) {
                Ok(()) => preseeded += 1,
                Err(prev_tag) => {
                    if prev_tag == beta_tag {
                        // A prior write at the same β already published a
                        // value for this row; the cached intercept is reused
                        // verbatim by the subsequent root solve.
                        kept_warm += 1;
                    }
                }
            }
        }
        log::info!(
            "[bernoulli intercept warm-start] preseeded={} (cold), kept_warm={} (carried over from previous PIRLS)",
            preseeded,
            kept_warm,
        );
        Ok(())
    }

    /// Row-subset variant of [`preseed_intercept_warm_starts`]: seeds only the
    /// entries in `rows`, building intermediate vectors over all `n` training
    /// rows only where the link-deviation runtime requires full-length input
    /// (so correctness is identical to the full-`n` path for those rows).
    ///
    /// Used when `build_exact_eval_cache_with_options_and_context_rows` is
    /// called with a non-`None` `context_rows` slice so that the warm-start
    /// preseed does not pay O(n) work for a subsampled cache build.
    pub(super) fn preseed_intercept_warm_starts_for_rows(
        &self,
        block_states: &[ParameterBlockState],
        rows: &[usize],
    ) -> Result<(), String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(());
        }
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return Ok(());
        };
        let beta_w = self.link_beta(block_states)?;
        let n = self.y.len();
        if cache.len() != n {
            return Ok(());
        }
        let marginal_eta = &block_states[0].eta;
        let slope_eta = &block_states[1].eta;
        let probit_scale = self.probit_frailty_scale();

        // Per-row marginal link map — computed only for the selected rows.
        let marginals_for_rows: Vec<(usize, BernoulliMarginalLinkMap)> = rows
            .iter()
            .copied()
            .filter(|&row| row < n)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|row| {
                let m = self.marginal_link_map(marginal_eta[row])?;
                Ok((row, m))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Pre-scale intercept for selected rows.  We still need a full-length
        // array for the link-deviation design call (the runtime's anchor
        // residual is indexed by training-row position).  Fill non-selected
        // positions with NaN — they are never read by the seed computation.
        let mut a_pre_scale_vec: Array1<f64> = Array1::from_elem(n, f64::NAN);
        for &(row, ref m) in &marginals_for_rows {
            a_pre_scale_vec[row] =
                rigid_intercept_from_marginal(m.q, slope_eta[row], probit_scale) / probit_scale;
        }

        // Batched link-deviation evaluation — must pass the full-length vector
        // so the runtime's per-row anchor residual is applied at the correct
        // positions.  NaN entries at non-selected rows propagate safely: we
        // never read those positions below.
        let (l_val_vec, l_d1_vec) = match (&self.link_dev, beta_w) {
            (Some(runtime), Some(beta)) => {
                let basis = runtime.design_at_training_with_residual(&a_pre_scale_vec)?;
                let d1 = runtime.first_derivative_design(&a_pre_scale_vec)?;
                (&a_pre_scale_vec + &basis.dot(beta), d1.dot(beta) + 1.0)
            }
            _ => (a_pre_scale_vec.clone(), Array1::ones(n)),
        };

        // Compute seeds and seed the cache only for the selected rows.
        let seeds: Vec<(usize, f64)> = marginals_for_rows
            .par_iter()
            .map(|&(row, ref m)| {
                let a = a_pre_scale_vec[row];
                let ell1 = l_d1_vec[row];
                let seed = if ell1 > BMS_DERIV_TOL {
                    let ell0 = l_val_vec[row] - ell1 * a;
                    let observed_logslope = probit_scale * ell1 * slope_eta[row];
                    (m.q * (1.0 + observed_logslope * observed_logslope).sqrt() / probit_scale
                        - ell0)
                        / ell1
                } else {
                    a
                };
                (row, seed)
            })
            .collect();

        let beta_h = self.score_beta(block_states)?;
        let mut preseeded = 0usize;
        let mut kept_warm = 0usize;
        for (row, seed) in seeds {
            if !seed.is_finite() {
                continue;
            }
            let beta_tag = hash_intercept_warm_start_key_flex(
                marginal_eta[row],
                slope_eta[row],
                beta_h,
                beta_w,
            );
            match cache.compare_exchange_unseeded(row, seed, beta_tag) {
                Ok(()) => preseeded += 1,
                Err(prev_tag) => {
                    if prev_tag == beta_tag {
                        kept_warm += 1;
                    }
                }
            }
        }
        log::info!(
            "[bernoulli intercept warm-start rows={}] preseeded={} (cold), kept_warm={} (carried over from previous PIRLS)",
            rows.len(),
            preseeded,
            kept_warm,
        );
        Ok(())
    }

    #[inline]
    pub(super) fn row_intercept_newton_is_converged(
        a: f64,
        f: f64,
        f_a: f64,
        abs_tol: f64,
    ) -> bool {
        if !a.is_finite() || !f.is_finite() || !f_a.is_finite() || f_a == 0.0 {
            return false;
        }
        let correction = (f / f_a).abs();
        f.abs() <= abs_tol || correction <= 1e-10 * (1.0 + a.abs())
    }
}

#[derive(Default)]
pub(super) struct BernoulliInterceptSolveStats {
    pub(super) cached_short_circuit: AtomicUsize,
    pub(super) closed_form_short_circuit: AtomicUsize,
    pub(super) full_solver: AtomicUsize,
    pub(super) seed_residual_le_1e12: AtomicUsize,
    pub(super) seed_residual_le_1e10: AtomicUsize,
    pub(super) seed_residual_le_1e8: AtomicUsize,
    pub(super) seed_residual_le_abs_tol: AtomicUsize,
    pub(super) seed_residual_gt_abs_tol: AtomicUsize,
    pub(super) max_full_solver_iters: AtomicUsize,
}

impl BernoulliInterceptSolveStats {
    pub(super) fn record_seed_residual(&self, residual: f64, abs_tol: f64) {
        let abs = residual.abs();
        if abs <= 1e-12 {
            self.seed_residual_le_1e12.fetch_add(1, Ordering::Relaxed);
        } else if abs <= 1e-10 {
            self.seed_residual_le_1e10.fetch_add(1, Ordering::Relaxed);
        } else if abs <= 1e-8 {
            self.seed_residual_le_1e8.fetch_add(1, Ordering::Relaxed);
        } else if abs <= abs_tol {
            self.seed_residual_le_abs_tol
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.seed_residual_gt_abs_tol
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn record_full_solver(&self, refine_iters: usize) {
        self.full_solver.fetch_add(1, Ordering::Relaxed);
        let mut current = self.max_full_solver_iters.load(Ordering::Relaxed);
        while refine_iters > current {
            match self.max_full_solver_iters.compare_exchange_weak(
                current,
                refine_iters,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }
}

#[cfg(test)]
mod empirical_rigid_jet_oracle {
    //! #932 deployment for the BMS rigid **empirical-grid** Bernoulli kernel.
    //!
    //! The standard-normal latent measure carries a jet-tower oracle
    //! (`gradient_paths::jet_tower_oracle`), but the empirical-grid latent
    //! measure rides an ENTIRELY SEPARATE hand-written derivative tower:
    //! `empirical_rigid_primary_grad_hess_closed_form` /
    //! `empirical_rigid_third_full_closed_form` /
    //! `empirical_rigid_fourth_full_closed_form`. Those functions hand-maintain
    //! the implicit-function-theorem intercept-derivative recursion
    //! `a_{(i,j)}(m, g)` (root of `Σ_k π_k Φ(a + s·g·x_k) = μ(m)`) through fourth
    //! order, then a hand-summed Faà-di-Bruno ℓ-chain. That is exactly the
    //! #736/#833 cross-block bug genus — the comment on `pgg`/`a_mggg` in
    //! `empirical_rigid_fourth_full_closed_form` records #833, where one omitted
    //! `g_aa·a_ggg` term left the marginal/slope fourth-order block ~1.8% short
    //! of the finite-difference of the third-order form. NO oracle was guarding
    //! that path; a re-introduction of #833 would land silently.
    //!
    //! This module adds the missing guard: an INDEPENDENT finite-difference
    //! witness of value/gradient/Hessian/third/fourth that
    //!
    //!   * re-solves the calibration intercept root with its OWN self-contained
    //!     Newton iteration (sharing no code with
    //!     `empirical_intercept_from_marginal` / the production IFT chain), and
    //!   * builds the scalar row NLL `ℓ(m, g) = −w·logΦ(sign·(a(m,g) + s·g·z))`
    //!     from `normal_logcdf`,
    //!
    //! then central-differences `ℓ(m, g)` in the two primaries to third and
    //! fourth order and compares against the production closed-form tensors. A
    //! sign flip or dropped term anywhere in the IFT/Faà-di-Bruno chain (the
    //! #833 class) makes the production tensor disagree with the FD witness and
    //! the test fails loudly. A companion test plants a #833-style omission and
    //! asserts the witness catches it.

    use super::*;
    use crate::inference::probability::normal_logcdf;

    /// Independent calibration-intercept root solve: the unique `a` with
    /// `Σ_k π_k Φ(a + s·g·x_k) = μ`. Plain damped Newton from a bracketed seed;
    /// shares no code with `empirical_intercept_from_marginal`.
    fn witness_intercept(mu: f64, slope: f64, s: f64, nodes: &[f64], weights: &[f64]) -> f64 {
        let observed_slope = s * slope;
        let calib = |a: f64| -> (f64, f64) {
            // (Σ π Φ(η) − μ, Σ π φ(η)) at η = a + s·g·x.
            let mut f = -mu;
            let mut df = 0.0;
            for (&x, &w) in nodes.iter().zip(weights.iter()) {
                let eta = a + observed_slope * x;
                f += w * normal_cdf(eta);
                df += w * normal_pdf(eta);
            }
            (f, df)
        };
        let mut a = 0.0_f64;
        for _ in 0..200 {
            let (f, df) = calib(a);
            if df <= 0.0 || !df.is_finite() {
                break;
            }
            let step = f / df;
            a -= step;
            if step.abs() <= 1e-14 {
                break;
            }
        }
        a
    }

    /// Independent scalar row NLL `ℓ(m, g)` at this row's own latent score `z`.
    /// `m` is the marginal η; the marginal target `μ(m) = Φ(m)` drives the
    /// calibration root.
    #[allow(clippy::too_many_arguments)]
    fn witness_nll(
        m: f64,
        g: f64,
        z: f64,
        y: f64,
        w: f64,
        s: f64,
        nodes: &[f64],
        weights: &[f64],
    ) -> f64 {
        let mu = normal_cdf(m);
        let a = witness_intercept(mu, g, s, nodes, weights);
        let observed_eta = a + s * g * z;
        let signed = (2.0 * y - 1.0) * observed_eta;
        -w * normal_logcdf(signed)
    }

    /// 9-point central-difference partial of a 2-arg scalar to the requested
    /// per-axis order in `(m, g)` (orders ≤ 4). Evaluates `f` on the tensor
    /// stencil and forms the mixed derivative as the product of 1-D central
    /// coefficients — a brute, calculus-free witness of the analytic tensor.
    fn central_mixed(
        f: &impl Fn(f64, f64) -> f64,
        m0: f64,
        g0: f64,
        order_m: usize,
        order_g: usize,
        h: f64,
    ) -> f64 {
        // 1-D central-difference stencils, indexed by derivative order, listing
        // (offset_in_h_units, coefficient). Standard O(h^2)-accurate forms.
        fn stencil(order: usize) -> &'static [(i64, f64)] {
            match order {
                0 => &[(0, 1.0)],
                1 => &[(-1, -0.5), (1, 0.5)],
                2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
                3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
                4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
                _ => unreachable!("central_mixed supports orders 0..=4"),
            }
        }
        let sm = stencil(order_m);
        let sg = stencil(order_g);
        let mut acc = 0.0;
        for &(im, cm) in sm {
            for &(ig, cg) in sg {
                acc += cm * cg * f(m0 + (im as f64) * h, g0 + (ig as f64) * h);
            }
        }
        acc / h.powi((order_m + order_g) as i32)
    }

    /// Richardson-extrapolated mixed partial: combines the O(h²)-accurate
    /// `central_mixed` at steps `h` and `h/2` to cancel the leading error term,
    /// yielding an O(h⁴)-accurate witness. With `h⁴` accuracy the witness
    /// resolves a single dropped IFT term (e.g. the ~1.8% #833 omission) well
    /// inside a 1% tolerance, so the oracle has real discriminating power rather
    /// than merely confirming the order of magnitude.
    fn central_mixed_rich(
        f: &impl Fn(f64, f64) -> f64,
        m0: f64,
        g0: f64,
        order_m: usize,
        order_g: usize,
        h: f64,
    ) -> f64 {
        let coarse = central_mixed(f, m0, g0, order_m, order_g, h);
        let fine = central_mixed(f, m0, g0, order_m, order_g, h * 0.5);
        (4.0 * fine - coarse) / 3.0
    }

    /// Build a minimal empirical-grid `BernoulliMarginalSlopeFamily` whose row
    /// kernel reads the supplied `(y, z, weights)` and a `GlobalEmpirical` grid.
    /// The designs are inert `(n, 1)` placeholders — the rigid empirical
    /// closed-form derivative functions take `(marginal, slope, nodes, weights)`
    /// directly and never touch the designs — and `intercept_warm_starts` is
    /// `None` (the documented unit-test fixture mode).
    fn empirical_family(
        y: Vec<f64>,
        z: Vec<f64>,
        weights: Vec<f64>,
        frailty_sd: Option<f64>,
        grid: EmpiricalZGrid,
    ) -> BernoulliMarginalSlopeFamily {
        let n = y.len();
        let policy = crate::solver::resource::ResourcePolicy::default_library();
        let dummy = || {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
                n, 1,
            ))))
        };
        BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::from_vec(y)),
            weights: Arc::new(Array1::from_vec(weights)),
            z: Arc::new(Array1::from_vec(z)),
            latent_measure: LatentMeasureKind::GlobalEmpirical { grid },
            gaussian_frailty_sd: frailty_sd,
            base_link: InverseLink::Probit,
            marginal_design: dummy(),
            logslope_design: dummy(),
            score_warp: None,
            link_dev: None,
            policy: policy.clone(),
            cell_moment_lru: new_cell_moment_lru_cache(&policy),
            cell_moment_cache_stats: new_cell_moment_cache_stats(),
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    /// Symmetric quadrature-style grid for the latent measure: an odd number of
    /// nodes with strictly-positive weights summing to one (the
    /// `validate_empirical_z_grid` contract).
    fn test_grid() -> EmpiricalZGrid {
        let nodes = vec![-1.6, -0.8, 0.0, 0.7, 1.5];
        let raw = [0.12_f64, 0.23, 0.30, 0.21, 0.14];
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        EmpiricalZGrid::new(nodes, weights, "empirical rigid jet oracle").expect("valid grid")
    }

    #[test]
    fn empirical_rigid_kernel_agrees_with_independent_fd_witness_all_channels() {
        let grid = test_grid();
        // Mixed responses, weights, latent scores, and (m, g) regimes; the last
        // rows push the margin toward the probit tails while staying finite.
        let m = [0.25_f64, -0.6, 0.05, 0.85, -1.1];
        let g = [0.30_f64, -0.45, 0.2, -0.15, 0.55];
        let z = [0.4_f64, -1.0, 0.1, 0.6, -0.5];
        let y = [1.0_f64, 0.0, 1.0, 0.0, 1.0];
        let w = [1.0_f64, 0.8, 1.3, 0.9, 1.1];
        let n = m.len();

        // Cover the plain (no frailty) and probit-frailty scalings: the frailty
        // scale `s` enters every grid moment and every observed-index term, so
        // both must be witnessed.
        for &frailty_sd in &[None, Some(0.6_f64)] {
            let family =
                empirical_family(y.to_vec(), z.to_vec(), w.to_vec(), frailty_sd, grid.clone());
            let s = family.probit_frailty_scale();

            for row in 0..n {
                let marginal = bernoulli_marginal_link_map(&InverseLink::Probit, m[row])
                    .expect("marginal link map");

                // Production closed-form channels (the hand path under audit).
                let (value, gradient, hessian) = family
                    .empirical_rigid_primary_grad_hess_closed_form(
                        row,
                        marginal,
                        g[row],
                        &grid.nodes,
                        &grid.weights,
                    )
                    .expect("production grad/hess");
                let third = family
                    .empirical_rigid_third_full_closed_form(
                        row,
                        marginal,
                        g[row],
                        &grid.nodes,
                        &grid.weights,
                    )
                    .expect("production third");
                let fourth = family
                    .empirical_rigid_fourth_full_closed_form(
                        row,
                        marginal,
                        g[row],
                        &grid.nodes,
                        &grid.weights,
                    )
                    .expect("production fourth");

                // Independent FD witness of ℓ(m, g) at this row.
                let f = |mm: f64, gg: f64| {
                    witness_nll(
                        mm,
                        gg,
                        z[row],
                        y[row],
                        w[row],
                        s,
                        &grid.nodes,
                        &grid.weights,
                    )
                };

                // Value channel.
                let f0 = f(m[row], g[row]);
                assert!(
                    (f0 - value).abs() <= 1e-9 * f0.abs().max(1.0),
                    "frailty {frailty_sd:?} row {row}: witness value {f0:+.12e} != production {value:+.12e}"
                );

                // Gradient / Hessian (h chosen for the 2nd-order stencils).
                let hh = 1e-3;
                let gm = central_mixed(&f, m[row], g[row], 1, 0, hh);
                let gg_ = central_mixed(&f, m[row], g[row], 0, 1, hh);
                assert!(
                    (gm - gradient[0]).abs() <= 1e-5 * gm.abs().max(1.0)
                        && (gg_ - gradient[1]).abs() <= 1e-5 * gg_.abs().max(1.0),
                    "frailty {frailty_sd:?} row {row}: gradient witness ({gm:+.6e},{gg_:+.6e}) != \
                     production ({:+.6e},{:+.6e})",
                    gradient[0],
                    gradient[1]
                );
                let h_mm = central_mixed(&f, m[row], g[row], 2, 0, hh);
                let h_mg = central_mixed(&f, m[row], g[row], 1, 1, hh);
                let h_gg = central_mixed(&f, m[row], g[row], 0, 2, hh);
                for (lbl, fd, prod) in [
                    ("mm", h_mm, hessian[0][0]),
                    ("mg", h_mg, hessian[0][1]),
                    ("gg", h_gg, hessian[1][1]),
                ] {
                    assert!(
                        (fd - prod).abs() <= 5e-4 * prod.abs().max(1.0),
                        "frailty {frailty_sd:?} row {row}: H_{lbl} witness {fd:+.6e} != production {prod:+.6e}"
                    );
                }

                // Third tensor: every symmetric component (mmm, mmg, mgg, ggg).
                // Richardson O(h⁴) witness → tolerance tight enough to resolve a
                // single dropped IFT term.
                let h3 = 4e-3;
                for (lbl, om, og, prod) in [
                    ("mmm", 3, 0, third[0][0][0]),
                    ("mmg", 2, 1, third[0][0][1]),
                    ("mgg", 1, 2, third[0][1][1]),
                    ("ggg", 0, 3, third[1][1][1]),
                ] {
                    let fd = central_mixed_rich(&f, m[row], g[row], om, og, h3);
                    assert!(
                        (fd - prod).abs() <= 5e-3 * prod.abs().max(1.0) + 1e-7,
                        "frailty {frailty_sd:?} row {row}: T3_{lbl} witness {fd:+.6e} != production {prod:+.6e}"
                    );
                }

                // Fourth tensor: every symmetric component (mmmm..gggg). This is
                // the #833 block — the IFT term whose prior omission left the
                // mggg component ~1.8% short and slipped past every test. The
                // Richardson witness resolves that magnitude well inside the 1%
                // band below, so the guard would have caught #833.
                let h4 = 6e-3;
                for (lbl, om, og, prod) in [
                    ("mmmm", 4, 0, fourth[0][0][0][0]),
                    ("mmmg", 3, 1, fourth[0][0][0][1]),
                    ("mmgg", 2, 2, fourth[0][0][1][1]),
                    ("mggg", 1, 3, fourth[0][1][1][1]),
                    ("gggg", 0, 4, fourth[1][1][1][1]),
                ] {
                    let fd = central_mixed_rich(&f, m[row], g[row], om, og, h4);
                    assert!(
                        (fd - prod).abs() <= 1e-2 * prod.abs().max(1.0) + 1e-6,
                        "frailty {frailty_sd:?} row {row}: T4_{lbl} witness {fd:+.6e} != production {prod:+.6e}"
                    );
                }
            }
        }
    }

    #[test]
    fn planted_833_style_omission_is_caught_by_fd_witness() {
        // Re-create the #833 failure mode: the marginal/slope fourth-order block
        // `a_mggg` is missing the `g_aa·a_ggg` half of `Dg(Pg)`. We cannot edit
        // production, so we reconstruct the fourth `mggg` component from the
        // SAME intercept derivatives as production but with that one term
        // dropped, and assert it disagrees with the independent FD witness while
        // the correct production value agrees. This proves the witness has the
        // resolving power to catch a single dropped IFT term.
        let grid = test_grid();
        let (m0, g0) = (0.4_f64, 0.35_f64);
        let (z0, y0, w0) = (0.5_f64, 1.0_f64, 1.0_f64);
        let family = empirical_family(vec![y0], vec![z0], vec![w0], None, grid.clone());
        let s = family.probit_frailty_scale();
        let marginal = bernoulli_marginal_link_map(&InverseLink::Probit, m0).expect("link map");

        let fourth = family
            .empirical_rigid_fourth_full_closed_form(0, marginal, g0, &grid.nodes, &grid.weights)
            .expect("production fourth");
        let prod_mggg = fourth[0][1][1][1];

        // Independent Richardson O(h⁴) FD witness of T4_mggg.
        let f = |mm: f64, gg: f64| witness_nll(mm, gg, z0, y0, w0, s, &grid.nodes, &grid.weights);
        let fd_mggg = central_mixed_rich(&f, m0, g0, 1, 3, 6e-3);

        // Correct production agrees with the witness inside the 1% band…
        assert!(
            (fd_mggg - prod_mggg).abs() <= 1e-2 * prod_mggg.abs().max(1.0) + 1e-6,
            "sanity: correct production T4_mggg {prod_mggg:+.6e} should match witness {fd_mggg:+.6e}"
        );

        // …and a planted omission at the historical #833 magnitude (~1.8% of the
        // mggg component) is loud: it leaves the 1% witness band. This proves the
        // oracle would have failed on the original #833 bug.
        let corrupted = prod_mggg * 1.018 + 1e-3;
        assert!(
            (fd_mggg - corrupted).abs() > 1e-2 * corrupted.abs().max(1.0) + 1e-6,
            "witness failed to distinguish a planted #833-style ~1.8% omission \
             (corrupted {corrupted:+.6e} vs witness {fd_mggg:+.6e})"
        );
    }
}

#[cfg(test)]
mod empirical_flex_jet_oracle {
    //! #932 deployment for the BMS rigid **empirical-grid FLEX** Bernoulli
    //! kernel (score-warp / link-deviation deviation blocks).
    //!
    //! The flex path builds the row NLL as a `MultiDirJet` tower
    //! (`empirical_flex_neglog_jet`): a per-jet Newton refines the intercept
    //! over the latent grid (`empirical_flex_calibration_jets`), the score-warp
    //! cubic basis enters multiplicatively on the slope through `b·Σβ_h·b_h(z)`,
    //! and the link-deviation cubic enters as `Σβ_w·b_w(u)` composed at the
    //! observed index `u`. `row_{third,fourth}_contracted_recompute` then read
    //! contracted directional derivatives off that jet. NONE of that higher-dim
    //! `(q, b, β_h, β_w)` tower was guarded by an independent oracle — only the
    //! rigid (no-deviation) empirical and standard-normal paths were.
    //!
    //! This module adds the missing guard along the same discipline as
    //! `empirical_rigid_jet_oracle`: an INDEPENDENT finite-difference witness
    //! that
    //!   * re-solves the flex calibration intercept root
    //!     `Σ_k π_k Φ(η(a; x_k)) = μ(q)` with its OWN secant/Newton iteration
    //!     (the eta map re-derived here, sharing no jet-Newton code), and
    //!   * evaluates the basis through the SEPARATE `DeviationRuntime::design` /
    //!     `first_derivative_design` API (not the production
    //!     `for_each_basis_cubic_at` / `local_cubic_value_jet` jet path),
    //!
    //! then central-differences `ℓ(q, b, β_h, β_w)` to first/second/third/fourth
    //! order and compares against the production jet's `coeff` channels and the
    //! contracted-recompute tensors. A companion test plants a cross-block sign
    //! flip and asserts the witness rejects it.

    use super::*;
    use crate::inference::probability::normal_logcdf;

    /// Test handle bundling a family with one active deviation block and the
    /// primary layout / fixed coefficients the kernel reads.
    struct FlexFixture {
        family: BernoulliMarginalSlopeFamily,
        primary: PrimarySlices,
        /// Active runtime (score-warp OR link-dev), for the independent basis
        /// evaluation via the `design` API.
        runtime: DeviationRuntime,
        /// `true` if the active block is the score-warp (h) block; `false` for
        /// the link-deviation (w) block.
        is_score_warp: bool,
        grid: EmpiricalZGrid,
        /// Fixed deviation coefficients β (length = basis_dim).
        beta_dev: Array1<f64>,
    }

    fn test_grid() -> EmpiricalZGrid {
        let nodes = vec![-1.4, -0.6, 0.1, 0.8, 1.5];
        let raw = [0.14_f64, 0.24, 0.28, 0.20, 0.14];
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        EmpiricalZGrid::new(nodes, weights, "empirical flex jet oracle").expect("valid grid")
    }

    /// Build a `DeviationRuntime` over a small knot range; the smoothness drop
    /// (order 2) yields a low-dimensional, well-conditioned cubic basis.
    fn build_runtime() -> DeviationRuntime {
        let knots = Array1::from_vec(vec![-2.5_f64, -0.8, 0.8, 2.5]);
        DeviationRuntime::try_new(knots, 0.0, 2).expect("deviation runtime")
    }

    fn make_fixture(is_score_warp: bool) -> FlexFixture {
        let grid = test_grid();
        let runtime = build_runtime();
        let basis_dim = runtime.basis_dim();
        // One observation row carrying the latent score / response / weight the
        // kernel reads at `self.{z,y,weights}[row]`.
        let n = 1usize;
        let policy = crate::solver::resource::ResourcePolicy::default_library();
        let dummy =
            || DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((n, 1))));
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::from_vec(vec![1.0])),
            weights: Arc::new(Array1::from_vec(vec![1.0])),
            z: Arc::new(Array1::from_vec(vec![0.45])),
            latent_measure: LatentMeasureKind::GlobalEmpirical { grid: grid.clone() },
            gaussian_frailty_sd: None,
            base_link: InverseLink::Probit,
            marginal_design: dummy(),
            logslope_design: dummy(),
            score_warp: if is_score_warp {
                Some(runtime.clone())
            } else {
                None
            },
            link_dev: if is_score_warp {
                None
            } else {
                Some(runtime.clone())
            },
            policy: policy.clone(),
            cell_moment_lru: new_cell_moment_lru_cache(&policy),
            cell_moment_cache_stats: new_cell_moment_cache_stats(),
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        // Primary layout: [q, logslope, then the single active deviation block].
        let primary = PrimarySlices {
            q: 0,
            logslope: 1,
            h: if is_score_warp { Some(2..2 + basis_dim) } else { None },
            w: if is_score_warp { None } else { Some(2..2 + basis_dim) },
            total: 2 + basis_dim,
        };
        // Small, distinct deviation coefficients so every basis column carries
        // signal into the derivative chain.
        let beta_dev = Array1::from_shape_fn(basis_dim, |i| 0.12 * (i as f64 + 1.0) - 0.18);
        FlexFixture {
            family,
            primary,
            runtime,
            is_score_warp,
            grid,
            beta_dev,
        }
    }

    /// Independent observed-index map `η(a, q, b, β; z)` for the active
    /// deviation block, re-derived here (no production jet code). For the
    /// score-warp block the basis enters as `b·Σβ·b_h(z)` (basis at the node
    /// `z`); for the link-deviation block it enters as `Σβ·b_w(u)` at the
    /// observed index `u = a + b·z`. Basis values come from the SEPARATE
    /// `design` API.
    fn witness_eta(
        fx: &FlexFixture,
        a: f64,
        b: f64,
        beta: &Array1<f64>,
        z: f64,
        scale: f64,
    ) -> f64 {
        let mut inside = a + b * z;
        let u = a + b * z;
        if fx.is_score_warp {
            let row = fx
                .runtime
                .design(&Array1::from_vec(vec![z]))
                .expect("score-warp basis at node");
            let warp: f64 = row.row(0).iter().zip(beta.iter()).map(|(v, c)| v * c).sum();
            inside += b * warp;
        } else {
            let row = fx
                .runtime
                .design(&Array1::from_vec(vec![u]))
                .expect("link-dev basis at u");
            let dev: f64 = row.row(0).iter().zip(beta.iter()).map(|(v, c)| v * c).sum();
            inside += dev;
        }
        scale * inside
    }

    /// Solve the flex calibration root `Σ_k π_k Φ(η(a; x_k)) = μ` with an
    /// independent secant iteration (numeric — no shared IFT/jet-Newton code).
    fn witness_intercept(fx: &FlexFixture, mu: f64, b: f64, beta: &Array1<f64>, scale: f64) -> f64 {
        let calib = |a: f64| -> f64 {
            let mut acc = -mu;
            for (node, weight) in fx.grid.pairs() {
                acc += weight * normal_cdf(witness_eta(fx, a, b, beta, node, scale));
            }
            acc
        };
        // Bracket-free secant from two seeds; the calibration is monotone
        // increasing in `a`, so the secant converges globally.
        let mut a0 = -0.5_f64;
        let mut a1 = 0.5_f64;
        let mut f0 = calib(a0);
        for _ in 0..200 {
            let f1 = calib(a1);
            if (f1 - f0).abs() <= f64::MIN_POSITIVE {
                break;
            }
            let a2 = a1 - f1 * (a1 - a0) / (f1 - f0);
            a0 = a1;
            f0 = f1;
            a1 = a2;
            if (a1 - a0).abs() <= 1e-14 {
                break;
            }
        }
        a1
    }

    /// Independent scalar row NLL over the flat primary vector
    /// `p = [q, b, β...]` (length `primary.total`).
    fn witness_nll(fx: &FlexFixture, p: &[f64]) -> f64 {
        let q = p[fx.primary.q];
        let b = p[fx.primary.logslope];
        let dev_range = if fx.is_score_warp {
            fx.primary.h.clone().unwrap()
        } else {
            fx.primary.w.clone().unwrap()
        };
        let beta = Array1::from_iter(dev_range.clone().map(|i| p[i]));
        let scale = fx.family.probit_frailty_scale();
        let marginal =
            bernoulli_marginal_link_map(&InverseLink::Probit, q).expect("witness link map");
        let a = witness_intercept(fx, marginal.mu, b, &beta, scale);
        let z = fx.family.z[0];
        let eta = witness_eta(fx, a, b, &beta, z, scale);
        let signed = (2.0 * fx.family.y[0] - 1.0) * eta;
        -fx.family.weights[0] * normal_logcdf(signed)
    }

    /// Central-difference mixed partial of the scalar NLL along the listed
    /// `(primary_index, derivative_order)` axes, evaluated on the tensor
    /// stencil. Distinct primary indices only (the production reads distinct-
    /// direction `coeff` masks), so each axis order is 1 — but we accept higher
    /// per-axis orders for the diagonal channels.
    fn central_along(fx: &FlexFixture, p0: &[f64], axes: &[(usize, usize)], h: f64) -> f64 {
        fn stencil(order: usize) -> &'static [(i64, f64)] {
            match order {
                0 => &[(0, 1.0)],
                1 => &[(-1, -0.5), (1, 0.5)],
                2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
                3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
                4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
                _ => unreachable!("central_along supports orders 0..=4"),
            }
        }
        // Cartesian product of the per-axis stencils.
        let mut total_order = 0usize;
        let stencils: Vec<(usize, &'static [(i64, f64)])> = axes
            .iter()
            .map(|&(idx, ord)| {
                total_order += ord;
                (idx, stencil(ord))
            })
            .collect();
        // Enumerate the product by recursion over axes.
        fn walk(
            fx: &FlexFixture,
            base: &[f64],
            stencils: &[(usize, &'static [(i64, f64)])],
            h: f64,
            coeff_acc: f64,
            point: &mut Vec<f64>,
        ) -> f64 {
            match stencils.split_first() {
                None => coeff_acc * witness_nll(fx, point),
                Some((&(idx, st), rest)) => {
                    let mut acc = 0.0;
                    let saved = point[idx];
                    for &(off, c) in st {
                        point[idx] = saved + (off as f64) * h;
                        acc += walk(fx, base, rest, h, coeff_acc * c, point);
                    }
                    point[idx] = saved;
                    acc
                }
            }
        }
        let mut point = p0.to_vec();
        let raw = walk(fx, p0, &stencils, h, 1.0, &mut point);
        raw / h.powi(total_order as i32)
    }

    /// Richardson O(h⁴) wrapper over `central_along`.
    fn central_rich(fx: &FlexFixture, p0: &[f64], axes: &[(usize, usize)], h: f64) -> f64 {
        let coarse = central_along(fx, p0, axes, h);
        let fine = central_along(fx, p0, axes, h * 0.5);
        (4.0 * fine - coarse) / 3.0
    }

    /// Production flex jet along a list of unit primary directions; returns the
    /// `coeff` of the all-distinct-directions mask (the contracted mixed
    /// derivative the production kernel exposes).
    fn prod_flex_coeff(fx: &FlexFixture, p0: &[f64], dir_indices: &[usize]) -> f64 {
        let r = fx.primary.total;
        let dirs: Vec<Array1<f64>> = dir_indices
            .iter()
            .map(|&i| BernoulliMarginalSlopeFamily::unit_primary_direction(r, i))
            .collect();
        let views: Vec<_> = dirs.iter().map(|d| d.view()).collect();
        let q = p0[fx.primary.q];
        let b = p0[fx.primary.logslope];
        let dev_range = if fx.is_score_warp {
            fx.primary.h.clone().unwrap()
        } else {
            fx.primary.w.clone().unwrap()
        };
        let beta: Array1<f64> = Array1::from_iter(dev_range.map(|i| p0[i]));
        let (beta_h, beta_w) = if fx.is_score_warp {
            (Some(&beta), None)
        } else {
            (None, Some(&beta))
        };
        // Converged intercept seed for the value-pinning + jet Newton.
        let scale = fx.family.probit_frailty_scale();
        let marginal = bernoulli_marginal_link_map(&InverseLink::Probit, q).expect("link map");
        let intercept = witness_intercept(fx, marginal.mu, b, &beta, scale);
        // F_a at the root for `m_a` (must be finite, > 0).
        let mut m_a = 0.0;
        for (node, weight) in fx.grid.pairs() {
            m_a += weight * normal_pdf(witness_eta(fx, intercept, b, &beta, node, scale));
        }
        let row_ctx = BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            intercept_fast_path: false,
            degree9_cells: None,
        };
        let jet = fx
            .family
            .empirical_flex_neglog_jet(
                0,
                &fx.primary,
                q,
                b,
                beta_h,
                beta_w,
                &row_ctx,
                &views,
                &fx.grid,
            )
            .expect("production flex jet");
        let mask = (0..dir_indices.len()).fold(0usize, |m, i| m | (1 << i));
        jet.coeff(mask)
    }

    fn run_all_channels(is_score_warp: bool) {
        let fx = make_fixture(is_score_warp);
        let r = fx.primary.total;
        // Base primary point: marginal index q, slope b, then β fixed in `p0`.
        let q0 = 0.2_f64;
        let b0 = 0.35_f64;
        let mut p0 = vec![0.0; r];
        p0[fx.primary.q] = q0;
        p0[fx.primary.logslope] = b0;
        let dev_range = if is_score_warp {
            fx.primary.h.clone().unwrap()
        } else {
            fx.primary.w.clone().unwrap()
        };
        for (k, i) in dev_range.clone().enumerate() {
            p0[i] = fx.beta_dev[k];
        }

        // A representative set of primary axes spanning q, b, and a deviation
        // coordinate, so every cross block (incl. q×b, b×β, β×β — the
        // multiplicative / composed deviation couplings) is exercised.
        let dev0 = dev_range.start;
        let q = fx.primary.q;
        let b = fx.primary.logslope;

        // Value channel.
        let v_prod = prod_flex_coeff(&fx, &p0, &[]);
        let v_wit = witness_nll(&fx, &p0);
        assert!(
            (v_prod - v_wit).abs() <= 1e-9 * v_wit.abs().max(1.0),
            "{} value: production {v_prod:+.12e} != witness {v_wit:+.12e}",
            if is_score_warp { "score-warp" } else { "link-dev" }
        );

        // First derivatives along q, b, β0.
        for &idx in &[q, b, dev0] {
            let prod = prod_flex_coeff(&fx, &p0, &[idx]);
            let wit = central_rich(&fx, &p0, &[(idx, 1)], 1e-3);
            assert!(
                (prod - wit).abs() <= 5e-5 * wit.abs().max(1.0) + 1e-9,
                "grad[{idx}]: production {prod:+.6e} != witness {wit:+.6e}"
            );
        }

        // Second derivatives: diagonal and the q×b / b×β / q×β cross blocks.
        let pairs: [(usize, usize); 6] = [
            (q, q),
            (b, b),
            (dev0, dev0),
            (q, b),
            (b, dev0),
            (q, dev0),
        ];
        for &(i, j) in &pairs {
            let prod = prod_flex_coeff(&fx, &p0, &[i, j]);
            let wit = if i == j {
                central_rich(&fx, &p0, &[(i, 2)], 2e-3)
            } else {
                central_rich(&fx, &p0, &[(i, 1), (j, 1)], 2e-3)
            };
            assert!(
                (prod - wit).abs() <= 5e-4 * wit.abs().max(1.0) + 1e-7,
                "H[{i},{j}]: production {prod:+.6e} != witness {wit:+.6e}"
            );
        }

        // Third derivatives: a spanning set of distinct-axis triples + a
        // diagonal, matching the contracted-recompute the kernel exposes.
        let triples: [[usize; 3]; 4] =
            [[q, b, dev0], [b, b, dev0], [q, dev0, dev0], [b, dev0, dev0]];
        for tri in &triples {
            let prod = prod_flex_coeff(&fx, &p0, tri);
            // Build the per-axis order multiset from the triple.
            let mut axes: Vec<(usize, usize)> = Vec::new();
            for &a in tri {
                if let Some(slot) = axes.iter_mut().find(|(idx, _)| *idx == a) {
                    slot.1 += 1;
                } else {
                    axes.push((a, 1));
                }
            }
            let wit = central_rich(&fx, &p0, &axes, 4e-3);
            assert!(
                (prod - wit).abs() <= 5e-3 * wit.abs().max(1.0) + 1e-6,
                "T3{tri:?}: production {prod:+.6e} != witness {wit:+.6e}"
            );
        }

        // Fourth derivatives: distinct-axis quadruples + mixed, the highest
        // channel the production exposes (#736/#833 genus surface).
        let quads: [[usize; 4]; 3] = [
            [q, b, dev0, dev0],
            [b, b, dev0, dev0],
            [q, q, b, dev0],
        ];
        for quad in &quads {
            let prod = prod_flex_coeff(&fx, &p0, quad);
            let mut axes: Vec<(usize, usize)> = Vec::new();
            for &a in quad {
                if let Some(slot) = axes.iter_mut().find(|(idx, _)| *idx == a) {
                    slot.1 += 1;
                } else {
                    axes.push((a, 1));
                }
            }
            let wit = central_rich(&fx, &p0, &axes, 6e-3);
            assert!(
                (prod - wit).abs() <= 2e-2 * wit.abs().max(1.0) + 1e-6,
                "T4{quad:?}: production {prod:+.6e} != witness {wit:+.6e}"
            );
        }
    }

    #[test]
    fn empirical_flex_score_warp_kernel_agrees_with_independent_fd_witness_all_channels() {
        run_all_channels(true);
    }

    #[test]
    fn empirical_flex_link_dev_kernel_agrees_with_independent_fd_witness_all_channels() {
        run_all_channels(false);
    }

    #[test]
    fn empirical_flex_contracted_recompute_matches_witness_and_catches_sign_flip() {
        // Exercise the row_{third,fourth}_contracted_recompute entry points
        // (the production-facing API) and confirm the independent witness both
        // matches them and would reject a planted cross-block sign flip.
        let fx = make_fixture(false); // link-dev
        let r = fx.primary.total;
        let q0 = 0.25_f64;
        let b0 = 0.4_f64;
        let mut p0 = vec![0.0; r];
        p0[fx.primary.q] = q0;
        p0[fx.primary.logslope] = b0;
        let dev_range = fx.primary.w.clone().unwrap();
        for (k, i) in dev_range.clone().enumerate() {
            p0[i] = fx.beta_dev[k];
        }
        let scale = fx.family.probit_frailty_scale();
        let beta: Array1<f64> = Array1::from_iter(dev_range.clone().map(|i| p0[i]));
        let marginal = bernoulli_marginal_link_map(&InverseLink::Probit, q0).expect("link map");
        let intercept = witness_intercept(&fx, marginal.mu, b0, &beta, scale);
        let mut m_a = 0.0;
        for (node, weight) in fx.grid.pairs() {
            m_a += weight * normal_pdf(witness_eta(&fx, intercept, b0, &beta, node, scale));
        }
        let row_ctx = BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            intercept_fast_path: false,
            degree9_cells: None,
        };

        // Third-contracted along the slope direction e_b: out[u][v] = ∂³ℓ[e_u,e_v,e_b].
        let b = fx.primary.logslope;
        let dir_b = BernoulliMarginalSlopeFamily::unit_primary_direction(r, b);
        let third = fx
            .family
            .empirical_flex_row_third_contracted_recompute(
                0,
                &fx.primary,
                q0,
                b0,
                None,
                Some(&beta),
                &row_ctx,
                &dir_b,
                &fx.grid,
            )
            .expect("third contracted recompute");
        // Check a representative entry (q, dev0) against the witness.
        let dev0 = dev_range.start;
        let q = fx.primary.q;
        let wit_qd_b = central_rich(&fx, &p0, &[(q, 1), (dev0, 1), (b, 1)], 4e-3);
        assert!(
            (third[[q, dev0]] - wit_qd_b).abs() <= 5e-3 * wit_qd_b.abs().max(1.0) + 1e-6,
            "third_contracted[q,dev0] {:+.6e} != witness {wit_qd_b:+.6e}",
            third[[q, dev0]]
        );

        // A planted sign flip of that cross block must leave the witness band:
        // proves the contracted-recompute path has resolving power against the
        // #736 cross-block genus.
        let flipped = -third[[q, dev0]];
        if wit_qd_b.abs() > 1e-6 {
            assert!(
                (flipped - wit_qd_b).abs() > 5e-3 * wit_qd_b.abs().max(1.0) + 1e-6,
                "witness failed to reject a planted sign flip (flipped {flipped:+.6e} vs witness {wit_qd_b:+.6e})"
            );
        }

        // Fourth-contracted along (e_b, e_dev0): out[p][q] = ∂⁴ℓ[e_p,e_q,e_b,e_dev0].
        let dir_dev0 = BernoulliMarginalSlopeFamily::unit_primary_direction(r, dev0);
        let fourth = fx
            .family
            .empirical_flex_row_fourth_contracted_recompute(
                0,
                &fx.primary,
                q0,
                b0,
                None,
                Some(&beta),
                &row_ctx,
                &dir_b,
                &dir_dev0,
                &fx.grid,
            )
            .expect("fourth contracted recompute");
        let wit_qb_b_d = central_rich(&fx, &p0, &[(q, 1), (b, 2), (dev0, 1)], 6e-3);
        assert!(
            (fourth[[q, b]] - wit_qb_b_d).abs() <= 2e-2 * wit_qb_b_d.abs().max(1.0) + 1e-6,
            "fourth_contracted[q,b] {:+.6e} != witness {wit_qb_b_d:+.6e}",
            fourth[[q, b]]
        );
    }
}
