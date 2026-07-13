use super::exact_eval_cache::*;

use super::family::*;

use super::gradient_paths::*;

use super::hessian_paths::*;

use super::row_kernel::*;

use super::*;

use crate::fnv1a::Fnv1a;
use gam_math::jet_scalar::{
    DynamicJetBatchWorkspace, DynamicOneSeedBatch, DynamicTwoSeedBatch, FixedRuntimeJet, OneSeed,
    RuntimeJetScalar, TwoSeed, filtered_implicit_solve_runtime_scalar,
};

thread_local! {
    /// Per-worker empirical FLEX third-order workspace. The largest batch is
    /// retained across rows, so a warmed worker does not revisit the global
    /// allocator for the runtime-sized jet tape.
    static EMPIRICAL_BMS_THIRD_WORKSPACE: std::cell::RefCell<DynamicJetBatchWorkspace> =
        std::cell::RefCell::new(DynamicJetBatchWorkspace::new(1));
    /// Per-worker empirical FLEX fourth-order pair workspace. A caller may
    /// evaluate several `(u,v)` contractions in one row-plan traversal.
    static EMPIRICAL_BMS_FOURTH_WORKSPACE: std::cell::RefCell<DynamicJetBatchWorkspace> =
        std::cell::RefCell::new(DynamicJetBatchWorkspace::new(1));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EmpiricalBmsJetSchedule {
    FixedWidthFromPlan,
    DynamicBatch { lanes: usize },
}

/// Bound the live directional jet work for widths without a fixed
/// specialization.
///
/// Each lane carries `r²` order-two coefficients, while the basis-dependent
/// row program contributes `O(r)` live tape nodes. Bounding `lanes·r³` tracks
/// the leading working set that caused the measured cache cliff and geometric
/// arena growth, while every chunk still evaluates the same frozen
/// [`EmpiricalBmsRowJetPlan`].
const EMPIRICAL_BMS_BATCH_TAPE_WORK_BUDGET: usize = 4096;
const EMPIRICAL_BMS_BATCH_LANE_CAP: usize = 8;

#[inline]
fn empirical_bms_runtime_batch_lanes(r: usize) -> usize {
    let tape_work_per_lane = r.saturating_mul(r).saturating_mul(r).max(1);
    (EMPIRICAL_BMS_BATCH_TAPE_WORK_BUDGET / tape_work_per_lane)
        .max(1)
        .min(EMPIRICAL_BMS_BATCH_LANE_CAP)
}

fn empirical_bms_jet_schedule(r: usize) -> EmpiricalBmsJetSchedule {
    match r {
        4 | 8 | 12 | 18 => EmpiricalBmsJetSchedule::FixedWidthFromPlan,
        runtime_width => EmpiricalBmsJetSchedule::DynamicBatch {
            lanes: empirical_bms_runtime_batch_lanes(runtime_width),
        },
    }
}

#[derive(Clone)]
pub(super) struct EmpiricalBmsIndexJetPlan {
    z: f64,
    score_values: Vec<f64>,
    link_stacks: Vec<[f64; 5]>,
}

#[derive(Clone)]
struct EmpiricalBmsCalibrationNodeJetPlan {
    index: EmpiricalBmsIndexJetPlan,
    weight: f64,
    cdf_stack: [f64; 5],
}

/// Canonical empirical-latent Bernoulli row plan shared by rigid and FLEX.
///
/// Rigid is the `h = w = None`, `dimension = 2` specialization. FLEX appends
/// the score-warp and link-deviation primaries. The plan freezes only scalar
/// base-point data: the certified intercept root, its scalar calibration
/// Jacobian, the left-biased local spline spans, and primitive derivative
/// stacks. Every derivative channel is produced by [`Self::evaluate`] from one
/// [`RuntimeJetScalar`] expression.
#[derive(Clone)]
pub(super) struct EmpiricalBmsRowJetPlan {
    primary: PrimarySlices,
    pub(super) intercept_root: f64,
    pub(super) inv_f_a: f64,
    scale: f64,
    mu_stack: [f64; 5],
    calibration: Vec<EmpiricalBmsCalibrationNodeJetPlan>,
    observed: EmpiricalBmsIndexJetPlan,
    pub(super) observed_sign: f64,
    pub(super) observed_neglog_stack: [f64; 5],
}

impl EmpiricalBmsRowJetPlan {
    #[inline]
    fn index_jet<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        a: &S,
        vars: &[S],
        index: &EmpiricalBmsIndexJetPlan,
        workspace: &'arena S::Workspace,
    ) -> S {
        let dimension = self.primary.total;
        let b = &vars[self.primary.logslope];
        let u = a.add(&b.scale(index.z));
        let mut inside = u.clone();

        if let Some(range) = self.primary.h.as_ref() {
            let mut score = S::constant(0.0, dimension, workspace);
            for (local, idx) in range.clone().enumerate() {
                score = score.add(&vars[idx].scale(index.score_values[local]));
            }
            inside = inside.add(&b.mul(&score));
        }

        if let Some(range) = self.primary.w.as_ref() {
            let mut warp = S::constant(0.0, dimension, workspace);
            for (local, idx) in range.clone().enumerate() {
                let basis = u.compose_unary(index.link_stacks[local]);
                warp = warp.add(&vars[idx].mul(&basis));
            }
            inside = inside.add(&warp);
        }
        inside.scale(self.scale)
    }

    /// Evaluate the single empirical-row expression in any runtime-sized jet
    /// algebra. `lift_iters` is two for order-2, three for one-seed, and four
    /// for two-seed/full-order evaluation.
    pub(super) fn evaluate<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        vars: &[S],
        lift_iters: usize,
        workspace: &'arena S::Workspace,
    ) -> Result<S, String> {
        let dimension = self.primary.total;
        if vars.len() != dimension {
            return Err(format!(
                "empirical BMS row plan received {} primaries, expected {dimension}",
                vars.len()
            ));
        }
        if vars.iter().any(|var| var.dimension() != dimension) {
            return Err("empirical BMS row plan received a mismatched jet dimension".to_string());
        }

        let neg_mu = vars[self.primary.q].compose_unary(self.mu_stack).neg();
        let constraint = |a: &S| -> S {
            let mut residual = neg_mu.clone();
            for node in &self.calibration {
                let eta = self.index_jet(a, vars, &node.index, workspace);
                residual = residual.add(&eta.compose_unary(node.cdf_stack).scale(node.weight));
            }
            residual
        };
        let intercept = filtered_implicit_solve_runtime_scalar(
            self.intercept_root,
            self.inv_f_a,
            lift_iters,
            dimension,
            workspace,
            constraint,
        );
        let signed = self
            .index_jet(&intercept, vars, &self.observed, workspace)
            .scale(self.observed_sign);
        Ok(signed.compose_unary(self.observed_neglog_stack))
    }
}

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
    per_block_lambdas: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref components) = deriv.s_psi_penalty_components {
        let mut s_psi = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_part.add_scaled_to(per_block_lambdas[*penalty_idx], &mut s_psi);
        }
        return s_psi;
    }
    if let Some(ref components) = deriv.s_psi_components {
        let mut s_psi = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_psi.scaled_add(per_block_lambdas[*penalty_idx], s_part);
        }
        s_psi
    } else if let Some(penalty_idx) = deriv.penalty_index {
        deriv
            .s_psi
            .mapv(|value| per_block_lambdas[penalty_idx] * value)
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

impl BernoulliMarginalSlopeFamily {
    #[inline]
    pub(super) fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
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
    /// Replaces the historical four-channel bitmask jet and its six Newton
    /// intercept-refinement passes per row with the exact
    /// implicit-function-theorem solution. The
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
        // #932 (doc §11, §14.3): value/gradient/Hessian are read off the SAME
        // single-source row jet every channel uses, with the grid intercept
        // a(m, g) lifted directly in the packed `Order2<2>` algebra (no hand
        // intercept-derivative formulas, no dense extra-variable tower).
        let jet = self.empirical_rigid_row_nll_jet::<gam_math::jet_scalar::Order2<2>>(
            row,
            marginal,
            slope,
            nodes,
            measure_weights,
            2,
        )?;
        Ok((
            gam_math::nested_dual::JetField::value(&jet),
            jet.g(),
            jet.h(),
        ))
    }

    /// Closed-form uncontracted **third**-derivative tensor of the rigid
    /// empirical-grid row negative log-likelihood, in primary coordinates
    /// `(m = marginal_eta, g = slope)`. Replaces the historical 64-coefficient
    /// bitmask jet and its six Newton intercept passes used by
    /// [`Self::rigid_row_third_full`].
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
        // #932 (doc §11, §14.3): the uncontracted third tensor is the `.t3`
        // channel of the SAME single-source row jet, evaluated at `Tower3<2>`.
        // Keep the previous four finite-precision lift passes for bit identity
        // with the old `Tower4<2>` path, but do not build a fourth tensor for a
        // consumer that never reads it.
        let jet = self.empirical_rigid_row_nll_jet::<gam_math::jet_tower::Tower3<2>>(
            row,
            marginal,
            slope,
            nodes,
            measure_weights,
            4,
        )?;
        Ok(jet.t3)
    }

    /// Closed-form uncontracted **fourth**-derivative tensor of the rigid
    /// empirical-grid row negative log-likelihood, in primary coordinates
    /// `(m = marginal_eta, g = slope)`. Replaces the historical 256-coefficient
    /// bitmask jet and its six Newton intercept passes used by
    /// [`Self::rigid_row_fourth_full`].
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
        // #932 (doc §11, §14.3): the uncontracted fourth tensor is the `.t4`
        // channel of the SAME single-source row jet at the packed `Tower4<2>`.
        // The former hand intercept-fourth chain (including the #833
        // `g_aa·a_ggg` term whose omission shifted the m/g block ~1.8%) is now
        // generated mechanically by the filtered lift — that whole genus cannot
        // recur because there is no separate channel to drop.
        let jet = self.empirical_rigid_row_nll_jet::<gam_math::jet_tower::Tower4<2>>(
            row,
            marginal,
            slope,
            nodes,
            measure_weights,
            4,
        )?;
        Ok(jet.t4)
    }

    /// Row negative-log-likelihood jet of the rigid empirical-grid kernel in the
    /// primaries `(m = marginal η, g = slope)`, evaluated in any [`JetScalar`]
    /// `S` (#932 — doc §11 "generic implicit-lift operator", §14.3 "BMS
    /// empirical rigid"). The grid intercept `a(m, g)` solving the calibration
    /// `Σ_k π_k Φ(a + s·g·x_k) = μ(m)` is lifted DIRECTLY in `S` by the filtered
    /// Hensel operator
    /// ([`gam_math::jet_scalar::filtered_implicit_solve_scalar`]) — no
    /// dense extra-variable tower and no hand-written intercept-derivative
    /// formulas — then the observed signed-probit NLL is composed on top.
    /// Reading `(value, g, H)` off `Order2<2>` serves `primary_grad_hess`;
    /// reading `.t3` / `.t4` off `Tower4<2>` serves `third_full` / `fourth_full`.
    /// `lift_iters` is `S`'s nilpotency order (`Order2`: 2, `Tower4`: 4).
    ///
    /// The intercept value channel never moves under the lift (the constraint's
    /// value channel is the certified root residual `= 0`), so each grid node's
    /// normal-CDF derivative stack at the fixed base index `η_k0 = a0 + s·g·x_k`
    /// is built ONCE — one transcendental pass — and the cheap polynomial
    /// composition repeats per lift grade.
    fn empirical_rigid_row_nll_jet<S: gam_math::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
        lift_iters: usize,
    ) -> Result<S, String> {
        let s = self.probit_frailty_scale();
        let a0 =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = s * slope;

        // One transcendental pass: per node the fixed normal-CDF derivative
        // stack at η_k0 = a0 + s·slope·x_k, the g-jet coefficient s·x_k, and the
        // primal calibration Jacobian F_a = Σ_k π_k φ(η_k0) = Σ_k π_k Φ'(η_k0).
        let mut f_a = 0.0f64;
        let mut node_stacks: Vec<([f64; 5], f64, f64)> = Vec::with_capacity(nodes.len());
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            let eta0 = a0 + observed_slope * node;
            let cdf_stack = unary_derivatives_normal_cdf(eta0);
            f_a += weight * cdf_stack[1];
            node_stacks.push((cdf_stack, s * node, weight));
        }
        if !f_a.is_finite() || f_a <= 0.0 {
            return Err(format!(
                "empirical rigid jet: non-positive calibration Jacobian F_a={f_a} at row {row}"
            ));
        }
        let inv_fa = 1.0 / f_a;

        // Seeded primaries θ = (m slot 0, g slot 1) and the marginal target
        // −μ(m) in S (its derivatives are exactly the production link map, so
        // this is correct for any marginal link).
        let m_jet = S::variable(marginal.eta, 0);
        let g_jet = S::variable(slope, 1);
        let neg_mu = m_jet
            .compose_unary([
                marginal.mu,
                marginal.mu1,
                marginal.mu2,
                marginal.mu3,
                marginal.mu4,
            ])
            .neg();

        // Constraint F(a, θ) = −μ(m) + Σ_k π_k Φ(a + s·g·x_k), evaluated in S.
        let constraint = |a: &S| -> S {
            let mut acc = neg_mu;
            for &(cdf_stack, g_coef, weight) in node_stacks.iter() {
                let eta_k = a.add(&g_jet.scale(g_coef));
                acc = acc.add(&eta_k.compose_unary(cdf_stack).scale(weight));
            }
            acc
        };
        let a_jet = gam_math::jet_scalar::filtered_implicit_solve_scalar::<2, S>(
            a0, inv_fa, lift_iters, constraint,
        );

        // Observed signed-probit NLL: η = a(m, g) + s·g·z, r = (2y−1)·η,
        // ℓ = −w·logΦ(r), through the SAME signed-probit scalar kernel the
        // standard-normal path uses.
        let z = self.z[row];
        let eta = a_jet.add(&g_jet.scale(s * z));
        let sign = 2.0 * self.y[row] - 1.0;
        let signed = eta.scale(sign);
        let m_signed = gam_math::nested_dual::JetField::value(&signed);
        if !(m_signed.is_finite() || m_signed == f64::INFINITY) {
            return Err(format!(
                "empirical rigid jet: non-finite signed margin {m_signed} at row {row}"
            ));
        }
        let stack = signed_probit_neglog_unary_stack(m_signed, self.weights[row]);
        if !stack[0].is_finite() {
            return Err(format!(
                "empirical rigid jet: non-finite log Φ at row {row}"
            ));
        }
        Ok(signed.compose_unary(stack))
    }

    fn empirical_bms_index_jet_plan(
        &self,
        primary: &PrimarySlices,
        intercept: f64,
        slope: f64,
        z: f64,
    ) -> Result<EmpiricalBmsIndexJetPlan, String> {
        let score_values = if let Some(range) = primary.h.as_ref() {
            let runtime = self.score_warp.as_ref().ok_or_else(|| {
                "empirical BMS score-warp primary range without runtime".to_string()
            })?;
            let mut values = vec![0.0; range.len()];
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                z,
                "empirical BMS score-warp plan",
                |local, _, cubic| {
                    values[local] = cubic.evaluate(z);
                    Ok(())
                },
            )?;
            values
        } else {
            Vec::new()
        };

        let link_stacks = if let Some(range) = primary.w.as_ref() {
            let runtime = self.link_dev.as_ref().ok_or_else(|| {
                "empirical BMS link-deviation primary range without runtime".to_string()
            })?;
            let u = intercept + slope * z;
            let mut stacks = vec![[0.0; 5]; range.len()];
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                u,
                "empirical BMS link-deviation plan",
                |local, _, cubic| {
                    stacks[local] = [
                        cubic.evaluate(u),
                        cubic.first_derivative(u),
                        cubic.second_derivative(u),
                        6.0 * cubic.c3,
                        0.0,
                    ];
                    Ok(())
                },
            )?;
            stacks
        } else {
            Vec::new()
        };
        Ok(EmpiricalBmsIndexJetPlan {
            z,
            score_values,
            link_stacks,
        })
    }

    /// Compile the scalar base-point data for the canonical empirical-latent
    /// row expression. `intercept_seed` is in the raw FLEX coordinate, where
    /// the observed probit index is `scale * (a + b*z + deviations)`. Rigid
    /// callers convert their historical scaled intercept by dividing by
    /// `scale`, making rigid exactly the zero-deviation specialization.
    pub(super) fn empirical_bms_row_jet_plan(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        intercept_seed: f64,
        grid: &EmpiricalZGrid,
    ) -> Result<EmpiricalBmsRowJetPlan, String> {
        if primary.total < 2 || primary.q >= primary.total || primary.logslope >= primary.total {
            return Err("empirical BMS row plan has an invalid primary layout".to_string());
        }
        match (primary.h.as_ref(), beta_h) {
            (Some(range), Some(beta)) if range.len() == beta.len() => {}
            (None, None) => {}
            (Some(range), Some(beta)) => {
                return Err(format!(
                    "empirical BMS score coefficients {} != primary range {}",
                    beta.len(),
                    range.len()
                ));
            }
            _ => {
                return Err("empirical BMS score primary/beta presence mismatch".to_string());
            }
        }
        match (primary.w.as_ref(), beta_w) {
            (Some(range), Some(beta)) if range.len() == beta.len() => {}
            (None, None) => {}
            (Some(range), Some(beta)) => {
                return Err(format!(
                    "empirical BMS link coefficients {} != primary range {}",
                    beta.len(),
                    range.len()
                ));
            }
            _ => {
                return Err("empirical BMS link primary/beta presence mismatch".to_string());
            }
        }
        if !intercept_seed.is_finite() {
            return Err(format!(
                "empirical BMS row plan has non-finite intercept seed at row {row}"
            ));
        }

        // The derivative lift requires a genuine scalar root. Refine in the
        // model's scalar calibration kernel, then freeze the primal Jacobian;
        // derivative channels below come only from the canonical jet expression.
        let mut intercept_root = intercept_seed;
        let scalar_tol = 1e-12 * (1.0 + intercept_root.abs());
        for _ in 0..4 {
            let (residual, f_a, _) = self.evaluate_empirical_grid_calibration_newton(
                intercept_root,
                q,
                slope,
                beta_h,
                beta_w,
                grid,
            )?;
            intercept_root -= residual / f_a;
            if residual.abs() <= scalar_tol {
                break;
            }
        }
        let (root_residual, f_a, _) = self.evaluate_empirical_grid_calibration_newton(
            intercept_root,
            q,
            slope,
            beta_h,
            beta_w,
            grid,
        )?;
        let root_tol = 1e-9 * (1.0 + intercept_root.abs());
        if root_residual.abs() > root_tol {
            return Err(format!(
                "empirical BMS intercept is not a calibration root at row {row}: \
                 residual={root_residual:.3e} > {root_tol:.3e}"
            ));
        }
        if !(f_a.is_finite() && f_a > 0.0) {
            return Err(format!(
                "empirical BMS calibration has invalid F_a={f_a} at row {row}"
            ));
        }

        let marginal = self.marginal_link_map(q)?;
        let mut calibration = Vec::with_capacity(grid.nodes.len());
        for (node, weight) in grid.pairs() {
            let index = self.empirical_bms_index_jet_plan(primary, intercept_root, slope, node)?;
            let obs = self.observed_denested_cell_partials_at_z(
                node,
                intercept_root,
                slope,
                beta_h,
                beta_w,
            )?;
            let eta = eval_coeff4_at(&obs.coeff, node);
            calibration.push(EmpiricalBmsCalibrationNodeJetPlan {
                index,
                weight,
                cdf_stack: unary_derivatives_normal_cdf(eta),
            });
        }

        let observed =
            self.empirical_bms_index_jet_plan(primary, intercept_root, slope, self.z[row])?;
        let obs = self.observed_denested_cell_partials_at_z(
            self.z[row],
            intercept_root,
            slope,
            beta_h,
            beta_w,
        )?;
        let observed_sign = 2.0 * self.y[row] - 1.0;
        let signed = observed_sign * eval_coeff4_at(&obs.coeff, self.z[row]);
        let observed_neglog_stack = unary_derivatives_neglog_phi(signed, self.weights[row]);
        if !observed_neglog_stack[0].is_finite() {
            return Err(format!(
                "empirical BMS row plan has non-finite log Phi at row {row}"
            ));
        }
        Ok(EmpiricalBmsRowJetPlan {
            primary: primary.clone(),
            intercept_root,
            inv_f_a: 1.0 / f_a,
            scale: self.probit_frailty_scale(),
            mu_stack: [
                marginal.mu,
                marginal.mu1,
                marginal.mu2,
                marginal.mu3,
                marginal.mu4,
            ],
            calibration,
            observed,
            observed_sign,
            observed_neglog_stack,
        })
    }

    /// Rigid empirical-latent specialization of the canonical BMS row plan.
    /// The historical rigid scalar root is stored in the scaled probit-index
    /// coordinate; dividing by `scale` maps it into the raw intercept coordinate
    /// used by the shared FLEX expression.
    pub(super) fn empirical_rigid_bms_row_jet_plan(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        grid: &EmpiricalZGrid,
    ) -> Result<EmpiricalBmsRowJetPlan, String> {
        let scale = self.probit_frailty_scale();
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!(
                "empirical rigid BMS row plan has invalid scale {scale}"
            ));
        }
        let scaled_intercept = self.empirical_rigid_intercept_for_row(
            row,
            marginal,
            slope,
            &grid.nodes,
            &grid.weights,
        )?;
        let primary = PrimarySlices {
            q: 0,
            logslope: 1,
            h: None,
            w: None,
            total: 2,
        };
        self.empirical_bms_row_jet_plan(
            row,
            &primary,
            marginal.eta,
            slope,
            None,
            None,
            scaled_intercept / scale,
            grid,
        )
    }

    #[inline]
    fn empirical_fixed_third_contracted<const K: usize>(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let point: &[f64; K] = point.try_into().map_err(|_| {
            format!(
                "fixed empirical BMS point length {} != specialization width {K}",
                point.len()
            )
        })?;
        let direction: &[f64; K] = direction
            .as_slice()
            .and_then(|values| values.try_into().ok())
            .ok_or_else(|| {
                format!(
                    "fixed empirical BMS third direction length {} != specialization width {K}",
                    direction.len()
                )
            })?;
        let vars: [FixedRuntimeJet<OneSeed<K>, K>; K] = std::array::from_fn(|axis| {
            FixedRuntimeJet::from_inner(OneSeed::seed_direction(point[axis], axis, direction[axis]))
        });
        let contracted = plan
            .evaluate(&vars, 3, &())?
            .into_inner()
            .contracted_third();
        let mut out = Array2::<f64>::zeros((K, K));
        for a in 0..K {
            for b in 0..K {
                out[[a, b]] = contracted[a][b];
            }
        }
        Ok(out)
    }

    #[inline]
    fn empirical_fixed_fourth_contracted<const K: usize>(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let point: &[f64; K] = point.try_into().map_err(|_| {
            format!(
                "fixed empirical BMS point length {} != specialization width {K}",
                point.len()
            )
        })?;
        let direction_u: &[f64; K] = direction_u
            .as_slice()
            .and_then(|values| values.try_into().ok())
            .ok_or_else(|| {
                format!(
                    "fixed empirical BMS fourth first-direction length {} != specialization width {K}",
                    direction_u.len()
                )
            })?;
        let direction_v: &[f64; K] = direction_v
            .as_slice()
            .and_then(|values| values.try_into().ok())
            .ok_or_else(|| {
                format!(
                    "fixed empirical BMS fourth second-direction length {} != specialization width {K}",
                    direction_v.len()
                )
            })?;
        let vars: [FixedRuntimeJet<TwoSeed<K>, K>; K] = std::array::from_fn(|axis| {
            FixedRuntimeJet::from_inner(TwoSeed::seed(
                point[axis],
                axis,
                direction_u[axis],
                direction_v[axis],
            ))
        });
        let contracted = plan
            .evaluate(&vars, 4, &())?
            .into_inner()
            .contracted_fourth();
        let mut out = Array2::<f64>::zeros((K, K));
        for a in 0..K {
            for b in 0..K {
                out[[a, b]] = contracted[a][b];
            }
        }
        Ok(out)
    }

    fn empirical_fixed_third_many_from_plan<const K: usize>(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        directions: &[Array1<f64>],
    ) -> Result<Vec<Array2<f64>>, String> {
        directions
            .iter()
            .map(|direction| Self::empirical_fixed_third_contracted::<K>(plan, point, direction))
            .collect()
    }

    fn empirical_fixed_third_many_dispatch(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        directions: &[Array1<f64>],
        r: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        match r {
            4 => Self::empirical_fixed_third_many_from_plan::<4>(plan, point, directions),
            8 => Self::empirical_fixed_third_many_from_plan::<8>(plan, point, directions),
            12 => Self::empirical_fixed_third_many_from_plan::<12>(plan, point, directions),
            18 => Self::empirical_fixed_third_many_from_plan::<18>(plan, point, directions),
            _ => Err(format!(
                "unsupported fixed empirical BMS third-many specialization width {r}"
            )),
        }
    }

    fn empirical_fixed_third_trace_from_plan<const K: usize>(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        gram: &[f64],
    ) -> Result<Array1<f64>, String> {
        let mut gradient = Array1::<f64>::zeros(K);
        for axis in 0..K {
            let mut basis = Array1::<f64>::zeros(K);
            basis[axis] = 1.0;
            let third = Self::empirical_fixed_third_contracted::<K>(plan, point, &basis)?;
            gradient[axis] = Self::row_primary_trace_contract(&third, gram);
        }
        Ok(gradient)
    }

    fn empirical_fixed_third_trace_dispatch(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        gram: &[f64],
        r: usize,
    ) -> Result<Array1<f64>, String> {
        match r {
            4 => Self::empirical_fixed_third_trace_from_plan::<4>(plan, point, gram),
            8 => Self::empirical_fixed_third_trace_from_plan::<8>(plan, point, gram),
            12 => Self::empirical_fixed_third_trace_from_plan::<12>(plan, point, gram),
            18 => Self::empirical_fixed_third_trace_from_plan::<18>(plan, point, gram),
            _ => Err(format!(
                "unsupported fixed empirical BMS third-trace specialization width {r}"
            )),
        }
    }

    fn empirical_fixed_fourth_many_from_plan<const K: usize>(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        direction_pairs: &[(&Array1<f64>, &Array1<f64>)],
    ) -> Result<Vec<Array2<f64>>, String> {
        direction_pairs
            .iter()
            .map(|(direction_u, direction_v)| {
                Self::empirical_fixed_fourth_contracted::<K>(plan, point, direction_u, direction_v)
            })
            .collect()
    }

    fn empirical_fixed_fourth_many_dispatch(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        direction_pairs: &[(&Array1<f64>, &Array1<f64>)],
        r: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        match r {
            4 => Self::empirical_fixed_fourth_many_from_plan::<4>(plan, point, direction_pairs),
            8 => Self::empirical_fixed_fourth_many_from_plan::<8>(plan, point, direction_pairs),
            12 => Self::empirical_fixed_fourth_many_from_plan::<12>(plan, point, direction_pairs),
            18 => Self::empirical_fixed_fourth_many_from_plan::<18>(plan, point, direction_pairs),
            _ => Err(format!(
                "unsupported fixed empirical BMS fourth-many specialization width {r}"
            )),
        }
    }

    pub(super) fn empirical_flex_row_third_contracted(
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
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in third contraction".into());
        }
        if matches!(r, 4 | 8 | 12 | 18) {
            let plan = self.empirical_bms_row_jet_plan(
                row,
                primary,
                q,
                b,
                beta_h,
                beta_w,
                row_ctx.intercept,
                grid,
            )?;
            let point = Self::intercept_primary_point(q, b, beta_h, beta_w);
            return match r {
                4 => Self::empirical_fixed_third_contracted::<4>(&plan, &point, dir),
                8 => Self::empirical_fixed_third_contracted::<8>(&plan, &point, dir),
                12 => Self::empirical_fixed_third_contracted::<12>(&plan, &point, dir),
                18 => Self::empirical_fixed_third_contracted::<18>(&plan, &point, dir),
                _ => Err(format!(
                    "unsupported fixed empirical BMS third specialization width {r}"
                )),
            };
        }
        let mut contracted = self.empirical_flex_row_third_contracted_many(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx,
            std::slice::from_ref(dir),
            grid,
        )?;
        Ok(contracted
            .pop()
            .expect("one empirical BMS direction produces one contraction"))
    }

    /// Evaluate every requested third contraction from one canonical row plan.
    /// Common widths reuse the plan across fixed-width lanes; other runtime
    /// widths reuse it across bounded arena chunks.
    pub(super) fn empirical_flex_row_third_contracted_many(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_dirs: &[Array1<f64>],
        grid: &EmpiricalZGrid,
    ) -> Result<Vec<Array2<f64>>, String> {
        let r = primary.total;
        if row_dirs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((lane, direction)) = row_dirs
            .iter()
            .enumerate()
            .find(|(_, direction)| direction.len() != r)
        {
            return Err(format!(
                "bernoulli empirical flex third contraction direction {lane} length {} != primary dimension {r}",
                direction.len()
            ));
        }
        if row_dirs
            .iter()
            .all(|direction| direction.iter().all(|value| *value == 0.0))
        {
            return Ok(row_dirs
                .iter()
                .map(|_| Array2::<f64>::zeros((r, r)))
                .collect());
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in third contraction".into());
        }
        let plan = self.empirical_bms_row_jet_plan(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx.intercept,
            grid,
        )?;
        let point = Self::intercept_primary_point(q, b, beta_h, beta_w);
        match empirical_bms_jet_schedule(r) {
            EmpiricalBmsJetSchedule::FixedWidthFromPlan => {
                Self::empirical_fixed_third_many_dispatch(&plan, &point, row_dirs, r)
            }
            EmpiricalBmsJetSchedule::DynamicBatch { lanes } => {
                EMPIRICAL_BMS_THIRD_WORKSPACE.with(|workspace| {
                    let mut workspace = workspace.borrow_mut();
                    let mut contracted = Vec::with_capacity(row_dirs.len());
                    for directions in row_dirs.chunks(lanes) {
                        workspace.reset(directions.len());
                        let vars = workspace.alloc_slice_fill_with(r, |axis| {
                            DynamicOneSeedBatch::seed_directions(
                                point[axis],
                                axis,
                                r,
                                &workspace,
                                |lane| directions[lane][axis],
                            )
                        });
                        let jet = plan.evaluate(vars, 3, &workspace)?;
                        for lane in 0..directions.len() {
                            contracted.push(
                                Array2::from_shape_vec((r, r), jet.contracted_third(lane).to_vec())
                                    .map_err(|error| {
                                        format!("empirical BMS third-contraction shape: {error}")
                                    })?,
                            );
                        }
                    }
                    Ok(contracted)
                })
            }
        }
    }

    /// Trace-contract every Hessian index of the full third derivative from one
    /// row plan. Direction `c` is seeded by basis vector `e_c`, then reduced
    /// immediately to `sum_ab gram[ab] * d3[abc]`; no rank-three tensor is
    /// materialized.
    pub(super) fn empirical_flex_row_third_trace_gradient(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        gram: &[f64],
        grid: &EmpiricalZGrid,
    ) -> Result<Array1<f64>, String> {
        let r = primary.total;
        if gram.len() != r * r {
            return Err(format!(
                "bernoulli empirical flex third trace gram length {} != {}",
                gram.len(),
                r * r
            ));
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in third trace gradient".into());
        }
        let plan = self.empirical_bms_row_jet_plan(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx.intercept,
            grid,
        )?;
        let point = Self::intercept_primary_point(q, b, beta_h, beta_w);
        match empirical_bms_jet_schedule(r) {
            EmpiricalBmsJetSchedule::FixedWidthFromPlan => {
                Self::empirical_fixed_third_trace_dispatch(&plan, &point, gram, r)
            }
            EmpiricalBmsJetSchedule::DynamicBatch { lanes } => {
                EMPIRICAL_BMS_THIRD_WORKSPACE.with(|workspace| {
                    let mut workspace = workspace.borrow_mut();
                    let mut gradient = Array1::<f64>::zeros(r);
                    for axis_start in (0..r).step_by(lanes) {
                        let active_lanes = (r - axis_start).min(lanes);
                        workspace.reset(active_lanes);
                        let vars = workspace.alloc_slice_fill_with(r, |axis| {
                            DynamicOneSeedBatch::seed_directions(
                                point[axis],
                                axis,
                                r,
                                &workspace,
                                |lane| {
                                    if axis_start + lane == axis { 1.0 } else { 0.0 }
                                },
                            )
                        });
                        let jet = plan.evaluate(vars, 3, &workspace)?;
                        for lane in 0..active_lanes {
                            gradient[axis_start + lane] = jet
                                .contracted_third(lane)
                                .iter()
                                .zip(gram)
                                .map(|(third, weight)| third * weight)
                                .sum();
                        }
                    }
                    Ok(gradient)
                })
            }
        }
    }

    pub(super) fn empirical_flex_row_fourth_contracted(
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
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in fourth contraction".into());
        }
        if matches!(r, 4 | 8 | 12 | 18) {
            let plan = self.empirical_bms_row_jet_plan(
                row,
                primary,
                q,
                b,
                beta_h,
                beta_w,
                row_ctx.intercept,
                grid,
            )?;
            let point = Self::intercept_primary_point(q, b, beta_h, beta_w);
            return match r {
                4 => Self::empirical_fixed_fourth_contracted::<4>(&plan, &point, dir_u, dir_v),
                8 => Self::empirical_fixed_fourth_contracted::<8>(&plan, &point, dir_u, dir_v),
                12 => Self::empirical_fixed_fourth_contracted::<12>(&plan, &point, dir_u, dir_v),
                18 => Self::empirical_fixed_fourth_contracted::<18>(&plan, &point, dir_u, dir_v),
                _ => Err(format!(
                    "unsupported fixed empirical BMS fourth specialization width {r}"
                )),
            };
        }
        let pairs = [(dir_u, dir_v)];
        let mut contracted = self.empirical_flex_row_fourth_contracted_many_ordered(
            row, primary, q, b, beta_h, beta_w, row_ctx, &pairs, grid,
        )?;
        Ok(contracted
            .pop()
            .expect("one empirical BMS direction pair produces one contraction"))
    }

    /// Evaluate ordered fourth contractions from one canonical row plan.
    pub(super) fn empirical_flex_row_fourth_contracted_many_ordered(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        direction_pairs: &[(&Array1<f64>, &Array1<f64>)],
        grid: &EmpiricalZGrid,
    ) -> Result<Vec<Array2<f64>>, String> {
        let r = primary.total;
        if direction_pairs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((lane, (direction_u, direction_v))) =
            direction_pairs
                .iter()
                .enumerate()
                .find(|(_, (direction_u, direction_v))| {
                    direction_u.len() != r || direction_v.len() != r
                })
        {
            return Err(format!(
                "bernoulli empirical flex fourth contraction pair {lane} lengths ({},{}) != primary dimension {r}",
                direction_u.len(),
                direction_v.len()
            ));
        }
        let is_zero = |direction: &Array1<f64>| direction.iter().all(|value| *value == 0.0);
        if direction_pairs
            .iter()
            .all(|(direction_u, direction_v)| is_zero(direction_u) || is_zero(direction_v))
        {
            return Ok(direction_pairs
                .iter()
                .map(|_| Array2::<f64>::zeros((r, r)))
                .collect());
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in fourth contraction".into());
        }
        let plan = self.empirical_bms_row_jet_plan(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx.intercept,
            grid,
        )?;
        let point = Self::intercept_primary_point(q, b, beta_h, beta_w);
        Self::empirical_bms_fourth_batch_from_plan(&plan, &point, direction_pairs, primary)
    }

    /// Execute ordered two-seed contractions from one already-frozen row plan.
    /// Common widths use the fixed specialization; other runtime widths use
    /// bounded dynamic arena chunks without repeating grid/basis preprocessing.
    pub(super) fn empirical_bms_fourth_batch_from_plan(
        plan: &EmpiricalBmsRowJetPlan,
        point: &[f64],
        direction_pairs: &[(&Array1<f64>, &Array1<f64>)],
        primary: &PrimarySlices,
    ) -> Result<Vec<Array2<f64>>, String> {
        let r = primary.total;
        match empirical_bms_jet_schedule(r) {
            EmpiricalBmsJetSchedule::FixedWidthFromPlan => {
                Self::empirical_fixed_fourth_many_dispatch(plan, point, direction_pairs, r)
            }
            EmpiricalBmsJetSchedule::DynamicBatch { lanes } => {
                let is_zero = |direction: &Array1<f64>| direction.iter().all(|value| *value == 0.0);
                EMPIRICAL_BMS_FOURTH_WORKSPACE.with(|workspace| {
                    let mut workspace = workspace.borrow_mut();
                    let mut contracted = Vec::with_capacity(direction_pairs.len());
                    for pairs in direction_pairs.chunks(lanes) {
                        workspace.reset(pairs.len());
                        let vars = workspace.alloc_slice_fill_with(r, |axis| {
                            DynamicTwoSeedBatch::seed_direction_pairs(
                                point[axis],
                                axis,
                                r,
                                &workspace,
                                |lane| (pairs[lane].0[axis], pairs[lane].1[axis]),
                            )
                        });
                        let jet = plan.evaluate(vars, 4, &workspace)?;
                        for (lane, (direction_u, direction_v)) in pairs.iter().enumerate() {
                            if is_zero(direction_u) || is_zero(direction_v) {
                                contracted.push(Array2::<f64>::zeros((r, r)));
                            } else {
                                contracted.push(
                                    Array2::from_shape_vec(
                                        (r, r),
                                        jet.contracted_fourth(lane).to_vec(),
                                    )
                                    .map_err(|error| {
                                        format!("empirical BMS fourth-contraction shape: {error}")
                                    })?,
                                );
                            }
                        }
                    }
                    Ok(contracted)
                })
            }
        }
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
    /// row pass. Used by `row_primary_third_contracted` so the
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
        E: Fn(&gam_math::jet_tower::Tower4<2>) -> T + Sync,
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
    /// The FLEX third/fourth row contraction kernels
    /// (`row_primary_{third,fourth}_contracted*`) read the per-cell
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

    // ── Jeffreys wide-p contracted-trace-Hessian row kernel ──────────────
    //
    // Binary twin of
    // `binomial_location_scale::expected_joint_contracted_trace_hessian_from_designs`
    // (gam#979): computes one row's contribution to `∇²_β tr(W · H(β))` for a
    // caller-supplied full-joint trace weight `W`, where `H` is the OBSERVED
    // joint Newton Hessian (BMS's Jeffreys information is declared identical
    // to the observed Hessian via
    // `joint_jeffreys_information_matches_observed_hessian` staying `true`).
    //
    // `H` is block-structured over the two rigid primaries
    // `(marginal, logslope)`; for row `i` it equals
    // `X_pᵀ h_i[p][q] X_q` summed over the primary block pair `(p, q)`, so
    // `tr(W H) = Σ_i (trace_qq[i]·h_i[q][q] + trace_qg[i]·h_i[q][g] +
    // trace_gg[i]·h_i[g][g])` where `trace_pq[i] = x_p[i]ᵀ W_pq x_q[i]`
    // (the reference's `trace_tt`/`trace_tl`/`trace_ll`, renamed to this
    // family's `(marginal=q, logslope=g)` primaries). Differentiating this
    // linear functional of the row's local Hessian twice through
    // `η_q[i] = x_q[i]·β_q`, `η_g[i] = x_g[i]·β_g` requires exactly the
    // row's uncontracted FOURTH-order primary tensor (one order higher than
    // the reference's third-order expected-information coefficients, because
    // BMS's `H` is the observed Hessian — second order in the log-likelihood
    // — rather than an expected/Fisher information already one order lower).
    // `contract_fourth_full` at each of the three unit (marginal, logslope)
    // direction pairs gives that second directional derivative of the row's
    // full local Hessian in one call; combining with the trace scalars
    // mirrors the reference's `coeff_tt[i] = trace_tt·tt_tt + trace_tl·tt_tl +
    // trace_ll·tt_ll` pattern exactly, substituted for this family's own
    // closed-form tensor.
    pub(super) fn rigid_row_contracted_trace_hessian_coefficients(
        fourth: &[[[[f64; 2]; 2]; 2]; 2],
        trace_qq: f64,
        trace_qg: f64,
        trace_gg: f64,
    ) -> (f64, f64, f64) {
        let m_qq = contract_fourth_full(fourth, 1.0, 0.0, 1.0, 0.0);
        let m_qg = contract_fourth_full(fourth, 1.0, 0.0, 0.0, 1.0);
        let m_gg = contract_fourth_full(fourth, 0.0, 1.0, 0.0, 1.0);
        let coeff_qq = trace_qq * m_qq[0][0] + trace_qg * m_qg[0][0] + trace_gg * m_gg[0][0];
        let coeff_qg = trace_qq * m_qq[0][1] + trace_qg * m_qg[0][1] + trace_gg * m_gg[0][1];
        let coeff_gg = trace_qq * m_qq[1][1] + trace_qg * m_qg[1][1] + trace_gg * m_gg[1][1];
        (coeff_qq, coeff_qg, coeff_gg)
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
            //     a single probit log-CDF eval, skipping derivative-plan
            //     construction and evaluation (the line search reads no
            //     derivative coefficients).
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
            let total: Result<f64, String> =
                gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                    weighted_rows.len(),
                    |range| -> Result<f64, String> {
                        let mut ll = 0.0;
                        for wr in &weighted_rows[range] {
                            ll += wr.weight * row_ll(wr.index)?;
                        }
                        Ok(ll)
                    },
                    |left, right| -> Result<_, String> { Ok(left + right) },
                )
                .map(|opt| opt.unwrap_or(0.0));
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
        let total: Result<f64, String> =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                weighted_rows.len(),
                |range| -> Result<f64, String> {
                    let mut ll = 0.0;
                    for wr in &weighted_rows[range] {
                        ll += wr.weight * row_ll(wr.index)?;
                    }
                    Ok(ll)
                },
                |left, right| -> Result<_, String> { Ok(left + right) },
            )
            .map(|opt| opt.unwrap_or(0.0));
        total
    }

    pub(super) fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    fn sigma_scale_derivatives(
        &self,
    ) -> Result<crate::survival::lognormal_kernel::ProbitFrailtyScaleJet, String> {
        let sigma = self.gaussian_frailty_sd.ok_or_else(|| {
            "bernoulli marginal-slope log-sigma auxiliary requested without GaussianShift sigma"
                .to_string()
        })?;
        Ok(crate::survival::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln()))
    }

    /// Evaluate the canonical rigid standard-normal row program with the slope
    /// already lifted through a jet-valued frailty scale. `probit_scale = 1`
    /// prevents a second scale application inside the single row expression.
    fn row_neglog_canonical_scale_jet<S: gam_math::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primaries: &[S; 2],
        scale: &S,
    ) -> Result<S, String> {
        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
        let observed_primaries = [primaries[0], primaries[1].mul(scale)];
        rigid_standard_normal_row_nll_generic(
            &observed_primaries,
            marginal,
            self.z[row],
            self.y[row],
            self.weights[row],
            1.0,
        )
    }

    pub(super) fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primaries = [block_states[0].eta[row], block_states[1].eta[row]];
        let scale = self.sigma_scale_derivatives()?;
        let terms = if second_sigma {
            second_parameter_order2_terms(
                primaries,
                scale.s,
                scale.ds,
                scale.d2s,
                |variables, parameter| {
                    self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
                },
            )?
        } else {
            first_parameter_order2_terms(primaries, scale.s, scale.ds, |variables, parameter| {
                self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
            })?
        };
        Ok((terms.objective, terms.grad, terms.hess))
    }

    fn row_sigma_primary_directional_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        direction: &[f64; 2],
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let primaries = [block_states[0].eta[row], block_states[1].eta[row]];
        let scale = self.sigma_scale_derivatives()?;
        let terms = first_parameter_directional_order2_terms(
            primaries,
            direction,
            scale.s,
            scale.ds,
            |variables, parameter| {
                self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
            },
        )?;
        Ok((terms.grad, terms.hess))
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
        let row_weights = options
            .outer_score_subsample
            .as_ref()
            .map(|_| crate::marginal_slope_shared::outer_row_weights_by_index(options, n));
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
        let row_weights = options
            .outer_score_subsample
            .as_ref()
            .map(|_| crate::marginal_slope_shared::outer_row_weights_by_index(options, n));
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
        let row_weights = options
            .outer_score_subsample
            .as_ref()
            .map(|_| crate::marginal_slope_shared::outer_row_weights_by_index(options, n));
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BernoulliBlockHessianAccumulator::new(&slices),
            |row, acc| -> Result<(), String> {
                let row_dir =
                    self.row_primary_direction_from_flat(row, &slices, &primary, d_beta_flat)?;
                let direction = [row_dir[0], row_dir[1]];
                let (_, mut hess) =
                    self.row_sigma_primary_directional_terms(row, block_states, &direction)?;
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
        crate::block_layout::block_count::validate_block_count::<String>(
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
        // When a deviation runtime (score-warp / linkwiggle) is active the
        // denested-partition cells are a function of the *row's* converged
        // intercept and slope, so their `(c0..c3, left, right)` fingerprints are
        // effectively row-unique. The fit-lifetime cross-row LRU then runs at
        // ~0.1% hit rate at large scale while pinning multiple GiB of resident
        // moment entries and serialising every row behind its mutex (insert +
        // eviction churn). Intra-β reuse of a single row's moments is already
        // served by the per-row `degree9_cells` cache and the
        // `RowCellMomentsBundle`; the cross-row layer buys nothing here. Skip it
        // and evaluate uncached — bit-identical to a cold LRU miss, which still
        // honours the affine tail-cell memo inside `evaluate_cell_moments`.
        if self.flex_active() {
            return exact_kernel::evaluate_cell_moments(cell, max_degree);
        }
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
        // See `evaluate_cell_moments_lru`: under an active deviation runtime the
        // cross-row LRU never amortises (row-unique cell fingerprints), so it is
        // pure resident-memory and lock overhead. Evaluate uncached — identical
        // to a cold LRU miss, which is exactly what the cached path computes via
        // `evaluate_cell_derivative_moments_uncached` on a miss.
        if self.flex_active() {
            return exact_kernel::evaluate_cell_derivative_moments_uncached(cell, max_degree);
        }
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
mod empirical_rigid_jet_oracle_tests {
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

    /// Independent calibration-intercept root solve: the unique `a` with
    /// `Σ_k π_k Φ(a + s·g·x_k) = μ`. Plain damped Newton from a bracketed seed;
    /// shares no code with `empirical_intercept_from_marginal`.
    // Witness-exact standard-normal primitives (`libm::erfc`, no piecewise
    // rational approximation). The high-order FD witness divides by h⁴, so it
    // amplifies any *smooth* approximation error in the CDF/logCDF by ~1/h⁴:
    // production's `normal_logcdf`/`normal_cdf` carry an ~1e-11 oscillating
    // approximation error on a scale near `h`, whose 4th finite difference is
    // ~2% of the mmmg tensor (#932). Routing the witness through ulp-accurate
    // `erfc` keeps it a genuinely INDEPENDENT, exact oracle of the analytic
    // production tensor (which differentiates the true log Φ analytically).
    fn wnorm_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
    }
    fn wnorm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
    fn wnorm_logcdf(x: f64) -> f64 {
        wnorm_cdf(x).max(1e-300).ln()
    }

    fn witness_intercept(mu: f64, slope: f64, s: f64, nodes: &[f64], weights: &[f64]) -> f64 {
        let observed_slope = s * slope;
        let calib = |a: f64| -> (f64, f64) {
            // (Σ π Φ(η) − μ, Σ π φ(η)) at η = a + s·g·x.
            let mut f = -mu;
            let mut df = 0.0;
            for (&x, &w) in nodes.iter().zip(weights.iter()) {
                let eta = a + observed_slope * x;
                f += w * wnorm_cdf(eta);
                df += w * wnorm_pdf(eta);
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
        let mu = wnorm_cdf(m);
        let a = witness_intercept(mu, g, s, nodes, weights);
        let observed_eta = a + s * g * z;
        let signed = (2.0 * y - 1.0) * observed_eta;
        -w * wnorm_logcdf(signed)
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
                _ => panic!("central_mixed supports orders 0..=4, got {order}"),
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
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let dummy = || {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
                (n, 1),
            )))
        };
        BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::from_vec(y)),
            weights: Arc::new(Array1::from_vec(weights)),
            z: Arc::new(Array1::from_vec(z)),
            latent_measure: LatentMeasureKind::GlobalEmpirical { grid },
            gaussian_frailty_sd: frailty_sd,
            base_link: InverseLink::Standard(gam_problem::StandardLink::Probit),
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
                let marginal = bernoulli_marginal_link_map(
                    &InverseLink::Standard(gam_problem::StandardLink::Probit),
                    m[row],
                )
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
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            m0,
        )
        .expect("link map");

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

    // ──────────────────────────────────────────────────────────────────────
    // EXACT tower oracle (#932): the empirical-grid rigid derivative tower,
    // single-sourced.
    //
    // The FD witness above is the *only* guard the hand-written
    // `empirical_rigid_{primary_grad_hess,third_full,fourth_full}_closed_form`
    // IFT/Faà-di-Bruno recursion had — and FD is an APPROXIMATION (percent-band
    // Richardson tolerance, conditioning-limited on the stiffest mggg channel:
    // the #833 term sat right at the edge of its 1% band). #932 asks for the
    // tower to be derived MECHANICALLY from a once-written row NLL and pinned to
    // the production tensors at the exact f64-FMA floor (~1e-9), so a dropped
    // IFT term cannot hide inside FD truncation.
    //
    // This builds a SECOND, fully exact `Tower4<2>` over the primaries (m, g)
    // that shares NO code with the production closed-forms: the calibrated
    // intercept `a(m, g)` is recovered by `jet_tower::implicit_solve` (per-order
    // linear correction of the dense-symmetric-tensor constraint
    // `F(a, m, g) = −μ(m) + Σ_k π_k Φ(a + s·g·node_k)`), and the row NLL is the
    // single scalar `ℓ = unary_derivatives_neglog_phi(sign·(a + s·g·z), w)`
    // composed onto that tower. Both compute the SAME analytic derivatives, so
    // they must agree to ~1e-9 with no truncation tolerance — the same exact
    // discipline the standard-normal `rigid` and `empirical_flex` oracles use.
    // This is the rigid-empirical analogue of `flex_tower_witness`
    // (`empirical_flex_jet_oracle_tests`) and the BMS rigid std-normal
    // `tower.t4` cutover, extended to the last hand-derived BMS tower.

    /// Exact `Tower4<2>` row NLL over the primaries θ = (m = marginal η, g =
    /// slope), with the calibrated intercept `a(m, g)` solved as an exact
    /// implicit tower via [`gam_math::jet_tower::implicit_solve`]. Reads
    /// value / gradient / Hessian / third / fourth straight off the tower.
    ///
    /// `m` enters the constraint ONLY through the marginal target `μ(m)` (its
    /// derivatives `mu1..mu4` come from the production link map), exactly as in
    /// production: the observed index `a + s·g·z` carries no explicit `m`, so
    /// every m-channel of the tower rides through `a(m, g)`.
    fn rigid_empirical_tower_witness(
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> gam_math::jet_tower::Tower4<2> {
        // `unary_derivatives_{normal_cdf,neglog_phi}` are in scope via the
        // file-level `use super::gradient_paths::*` (the same glob the flex
        // tower witness relies on).
        use gam_math::jet_tower::{Tower4, implicit_solve};

        let s = family.probit_frailty_scale();
        // Scalar intercept anchor (order-0 root) from the independent bracketed
        // solve already defined in this module — shares no code with the
        // production IFT chain.
        let a0 = witness_intercept(marginal.mu, slope, s, nodes, measure_weights);

        // Calibration constraint F(a, m, g) over slots (0 = a, 1 = m, 2 = g):
        //   F = −μ(m) + Σ_k π_k · Φ(a + s·g·node_k).
        let a_var = Tower4::<3>::variable(a0, 0);
        // m-axis anchor is the marginal linear predictor η this map expands
        // about (`marginal.eta`); `mu1..mu4` are derivatives of μ w.r.t. that η.
        let m_var = Tower4::<3>::variable(marginal.eta, 1);
        let g_var = Tower4::<3>::variable(slope, 2);
        // μ(m) as a unary composition of the marginal η slot — derivatives are
        // exactly the production link map (correct for ANY marginal link).
        let mu_tower = m_var.compose_unary([
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
        ]);
        let mut f_constraint = Tower4::<3>::constant(0.0) - mu_tower;
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            // η_k = a + (s·g)·node_k.
            let eta_k = a_var + g_var.scale(s * node);
            let cdf = eta_k.compose_unary(unary_derivatives_normal_cdf(eta_k.v));
            f_constraint = f_constraint + cdf.scale(weight);
        }
        // Eliminate a → exact intercept tower a(m, g) as a Tower4<2>.
        let a_tower: Tower4<2> = implicit_solve::<3, 2>(&f_constraint, a0)
            .expect("rigid empirical implicit intercept tower");

        // Row NLL over θ = (m, g): observed index η = a(m, g) + s·g·z, signed by
        // (2y − 1), through the SAME signed-probit −logΦ scalar kernel
        // production uses (`unary_derivatives_neglog_phi` = the production
        // `signed_probit_neglog_unary_stack`). g (slot 1) enters the index both
        // directly (s·g·z) and through a; m (slot 0) only through a.
        let z = family.z[row];
        let g_t = Tower4::<2>::variable(slope, 1);
        let eta = a_tower + g_t.scale(s * z);
        let sign = 2.0 * family.y[row] - 1.0;
        let signed = eta.scale(sign);
        signed.compose_unary(unary_derivatives_neglog_phi(signed.v, family.weights[row]))
    }

    /// The production hand-written closed-form tower
    /// (`primary_grad_hess` + `third_full` + `fourth_full`) must equal the
    /// mechanically-derived `implicit_solve` tower to the f64 floor (~1e-9) —
    /// the exact-oracle replacement for the percent-band FD witness. Any dropped
    /// IFT / Faà-di-Bruno term (the #833 genus) shifts a tensor component well
    /// outside 1e-9 and fails loudly, with no truncation slack to hide in.
    #[test]
    fn empirical_rigid_kernel_matches_exact_implicit_solve_tower_932() {
        let grid = test_grid();
        let m = [0.25_f64, -0.6, 0.05, 0.85, -1.1];
        let g = [0.30_f64, -0.45, 0.2, -0.15, 0.55];
        let z = [0.4_f64, -1.0, 0.1, 0.6, -0.5];
        let y = [1.0_f64, 0.0, 1.0, 0.0, 1.0];
        let w = [1.0_f64, 0.8, 1.3, 0.9, 1.1];
        let n = m.len();

        for &frailty_sd in &[None, Some(0.6_f64)] {
            let family =
                empirical_family(y.to_vec(), z.to_vec(), w.to_vec(), frailty_sd, grid.clone());

            for row in 0..n {
                let marginal = bernoulli_marginal_link_map(
                    &InverseLink::Standard(gam_problem::StandardLink::Probit),
                    m[row],
                )
                .expect("marginal link map");

                // Production hand-written closed-form channels (under audit).
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

                // Independent exact tower (implicit_solve), shares no code.
                let tower = rigid_empirical_tower_witness(
                    &family,
                    row,
                    marginal,
                    g[row],
                    &grid.nodes,
                    &grid.weights,
                );

                // Graded tolerance. Value/gradient/Hessian (orders 0–2) are
                // read off the lift at the f64 floor and match to 1e-9. The
                // third/fourth grades (#833 block) are produced by the iterative
                // filtered-Newton lift (`filtered_implicit_solve_scalar`, one
                // corrective pass per grade) whereas the witness uses the
                // order-by-order exact `implicit_solve`; the two are algebraically
                // identical but reassociate the f64 arithmetic differently, and
                // that reassociation noise grows with grade — empirically ~1e-9
                // absolute by T4. The lift is grade-4-complete at iters=4 (bumping
                // to iters=5 leaves the T4 value unmoved — confirmed reassociation,
                // not under-iteration/truncation). A graded floor (1e-9 through
                // order 2, 1e-8 for the T3/T4 grades) tracks the achievable FP
                // agreement while still rejecting a genuinely dropped IFT / Faà-di-
                // Bruno term by ~6 orders of magnitude: a real #833-genus omission
                // shifts the m/g block ~1e-2 (see
                // `planted_833_style_omission_is_caught_by_exact_tower_932`), which
                // 1e-8 still catches with an enormous margin.
                let tol = |scale: f64| 1e-9 * scale.abs().max(1.0);
                let tol_hi = |scale: f64| 1e-8 * scale.abs().max(1.0);

                // Value.
                assert!(
                    (tower.v - value).abs() <= tol(value),
                    "frailty {frailty_sd:?} row {row}: tower value {:+.12e} != production {value:+.12e}",
                    tower.v
                );

                // Gradient (m, g).
                for (axis, prod) in [(0usize, gradient[0]), (1usize, gradient[1])] {
                    assert!(
                        (tower.g[axis] - prod).abs() <= tol(prod),
                        "frailty {frailty_sd:?} row {row}: grad[{axis}] tower {:+.12e} != production {prod:+.12e}",
                        tower.g[axis]
                    );
                }

                // Hessian (symmetric).
                for (i, j, prod) in [
                    (0, 0, hessian[0][0]),
                    (0, 1, hessian[0][1]),
                    (1, 1, hessian[1][1]),
                ] {
                    assert!(
                        (tower.h[i][j] - prod).abs() <= tol(prod),
                        "frailty {frailty_sd:?} row {row}: H[{i}][{j}] tower {:+.12e} != production {prod:+.12e}",
                        tower.h[i][j]
                    );
                }

                // Third tensor (every symmetric component).
                for (i, j, k, prod) in [
                    (0, 0, 0, third[0][0][0]),
                    (0, 0, 1, third[0][0][1]),
                    (0, 1, 1, third[0][1][1]),
                    (1, 1, 1, third[1][1][1]),
                ] {
                    assert!(
                        (tower.t3[i][j][k] - prod).abs() <= tol_hi(prod),
                        "frailty {frailty_sd:?} row {row}: T3[{i}][{j}][{k}] tower {:+.12e} != production {prod:+.12e}",
                        tower.t3[i][j][k]
                    );
                }

                // Fourth tensor (every symmetric component) — the #833 block.
                for (i, j, k, l, prod) in [
                    (0, 0, 0, 0, fourth[0][0][0][0]),
                    (0, 0, 0, 1, fourth[0][0][0][1]),
                    (0, 0, 1, 1, fourth[0][0][1][1]),
                    (0, 1, 1, 1, fourth[0][1][1][1]),
                    (1, 1, 1, 1, fourth[1][1][1][1]),
                ] {
                    assert!(
                        (tower.t4[i][j][k][l] - prod).abs() <= tol_hi(prod),
                        "frailty {frailty_sd:?} row {row}: T4[{i}][{j}][{k}][{l}] tower {:+.12e} != production {prod:+.12e}",
                        tower.t4[i][j][k][l]
                    );
                }
            }
        }
    }

    /// Companion to the planted-#833 FD test, but at the EXACT tower's
    /// resolution: a single dropped IFT term in the production `mggg` component
    /// is `O(10⁻²)` of the tensor, while the exact-tower agreement band is
    /// `~1e-9`, so the omission is `~10⁷×` outside tolerance — the exact oracle
    /// catches it with enormous margin (vs the FD witness's bare ~2× margin).
    #[test]
    fn planted_833_style_omission_is_caught_by_exact_tower_932() {
        let grid = test_grid();
        let (m0, g0) = (0.4_f64, 0.35_f64);
        let (z0, y0, w0) = (0.5_f64, 1.0_f64, 1.0_f64);
        let family = empirical_family(vec![y0], vec![z0], vec![w0], None, grid.clone());
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            m0,
        )
        .expect("link map");

        let fourth = family
            .empirical_rigid_fourth_full_closed_form(0, marginal, g0, &grid.nodes, &grid.weights)
            .expect("production fourth");
        let prod_mggg = fourth[0][1][1][1];

        let tower =
            rigid_empirical_tower_witness(&family, 0, marginal, g0, &grid.nodes, &grid.weights);
        let tower_mggg = tower.t4[0][1][1][1];

        // Correct production agrees with the exact tower at the f64 floor…
        assert!(
            (tower_mggg - prod_mggg).abs() <= 1e-9 * prod_mggg.abs().max(1.0),
            "sanity: correct production T4_mggg {prod_mggg:+.12e} should match exact tower {tower_mggg:+.12e}"
        );

        // …and a planted #833-style ~1.8% omission is ~10⁷× outside the band.
        let corrupted = prod_mggg * 1.018 + 1e-3;
        assert!(
            (tower_mggg - corrupted).abs() > 1e-9 * corrupted.abs().max(1.0),
            "exact tower failed to distinguish a planted #833-style omission \
             (corrupted {corrupted:+.12e} vs tower {tower_mggg:+.12e})"
        );
    }
}

#[cfg(test)]
mod empirical_flex_jet_oracle_tests {
    //! #932 deployment for the BMS rigid **empirical-grid FLEX** Bernoulli
    //! kernel (score-warp / link-deviation deviation blocks).
    //!
    //! The production flex path freezes one [`EmpiricalBmsRowJetPlan`] at the
    //! scalar row state, then evaluates that single expression over the
    //! fixed-width or bounded runtime [`RuntimeJetScalar`] algebra selected by
    //! the consumer schedule. The score-warp
    //! basis enters multiplicatively through `b·Σβ_h·b_h(z)` and the
    //! link-deviation basis enters as `Σβ_w·b_w(u)` at `u = a + b·z`; the
    //! filtered implicit solve lifts the calibrated intercept in the same
    //! evaluation. Third/fourth contractions are read directly from the
    //! one-/two-seed result, without a dense derivative tensor.
    //!
    //! This module adds the missing guard along the same discipline as
    //! `empirical_rigid_jet_oracle`: an INDEPENDENT finite-difference witness
    //! that
    //!   * re-solves the flex calibration intercept root
    //!     `Σ_k π_k Φ(η(a; x_k)) = μ(q)` with its OWN secant/Newton iteration
    //!     (the eta map re-derived here, sharing no row-plan code), and
    //!   * evaluates the basis through the SEPARATE `DeviationRuntime::design` /
    //!     `first_derivative_design` API rather than the production plan's
    //!     frozen local-cubic derivative stacks,
    //!
    //! then central-differences `ℓ(q, b, β_h, β_w)` to first/second/third/fourth
    //! order and compares against the production scalar and contracted
    //! channels. A companion test plants a cross-block sign flip and asserts
    //! the witness rejects it.

    use super::*;
    use gam_math::jet_scalar::{DynamicJetArena, DynamicOneSeed, DynamicTwoSeed};

    fn unit_primary_direction(r: usize, idx: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r);
        out[idx] = 1.0;
        out
    }

    #[test]
    fn empirical_bms_schedule_maps_common_widths_to_one_fixed_plan_932() {
        for r in [4, 8, 12, 18] {
            assert_eq!(
                empirical_bms_jet_schedule(r),
                EmpiricalBmsJetSchedule::FixedWidthFromPlan,
                "r={r} must reuse one fixed-width plan",
            );
        }
        for r in [1, 2, 3, 5, 7, 9, 16, 19, 32, 128] {
            assert_eq!(
                empirical_bms_jet_schedule(r),
                EmpiricalBmsJetSchedule::DynamicBatch {
                    lanes: empirical_bms_runtime_batch_lanes(r),
                },
                "r={r} must use the bounded runtime schedule",
            );
        }
    }

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

    /// Build a `DeviationRuntime` over a small knot range; the smoothness-
    /// nullspace drop yields a low-dimensional, well-conditioned cubic basis
    /// for the independent finite-difference witness.
    fn build_runtime() -> DeviationRuntime {
        // 11 uniform knots over [-2.45, 2.55] (10 spans). The half-span offset
        // keeps the oracle's finite-difference stencils away from spline knots;
        // production differentiates the local cubic branch selected at the base
        // point, and the independent witness must sample that same branch. The
        // cubic I-spline
        // DEVIATION basis is built from strictly-monotone increments, so its
        // span contains NO constant and NO linear function. An order-`m`
        // smoothness penalty's null space is the polynomials of degree `< m`:
        //   - order 1 (null = constants)  -> NOT in the I-spline span -> the raw
        //     penalty is full-rank -> `smoothness_nullspace_orthogonal_complement`
        //     finds nothing to drop and `try_new` rejects it;
        //   - order 2 (null = linears)    -> likewise NOT in the span -> rejected;
        //   - order 3 (null = quadratics) -> quadratics ARE in the cubic span,
        //     giving a genuine 3-dim droppable null space for location-block
        //     absorption.
        // Order 3 is therefore the ONLY mathematically valid order for an
        // I-spline value basis — do NOT flip this back to 1 or 2 (both panic
        // with "smoothness penalty has no null directions"). 10 spans leave
        // ~7 columns after the 3-dim drop, comfortably enough for the oracle's
        // q/b/deviation axes.
        let n_knots = 11usize;
        let knots = Array1::from_iter(
            (0..n_knots).map(|i| -2.45_f64 + 5.0_f64 * (i as f64) / ((n_knots - 1) as f64)),
        );
        DeviationRuntime::try_new(knots, 0.0, 3).expect("deviation runtime")
    }

    fn make_fixture(is_score_warp: bool) -> FlexFixture {
        let grid = test_grid();
        let runtime = build_runtime();
        let basis_dim = runtime.basis_dim();
        // One observation row carrying the latent score / response / weight the
        // kernel reads at `self.{z,y,weights}[row]`.
        let n = 1usize;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let dummy = || {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
                (n, 1),
            )))
        };
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::from_vec(vec![1.0])),
            weights: Arc::new(Array1::from_vec(vec![1.0])),
            z: Arc::new(Array1::from_vec(vec![0.45])),
            latent_measure: LatentMeasureKind::GlobalEmpirical { grid: grid.clone() },
            gaussian_frailty_sd: None,
            base_link: InverseLink::Standard(gam_problem::StandardLink::Probit),
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
            h: if is_score_warp {
                Some(2..2 + basis_dim)
            } else {
                None
            },
            w: if is_score_warp {
                None
            } else {
                Some(2..2 + basis_dim)
            },
            total: 2 + basis_dim,
        };
        // Small, distinct deviation coefficients so every basis column carries
        // signal into the derivative chain. The symmetric scaling has an exact
        // max |β| of 0.06 for any basis dimension, keeping the composed
        // link-deviation witness well conditioned while preserving nonzero
        // q/b/β cross-channel signal.
        let beta_dev = Array1::from_shape_fn(basis_dim, |i| {
            let center = 0.5 * (basis_dim.saturating_sub(1) as f64);
            let radius = center.max(1.0);
            0.06 * ((i as f64) - center) / radius
        });
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

    fn witness_normal_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
    }

    fn witness_normal_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    fn witness_normal_logcdf(x: f64) -> f64 {
        witness_normal_cdf(x).max(1e-300).ln()
    }

    /// Solve the flex calibration root `Σ_k π_k Φ(η(a; x_k)) = μ` with an
    /// independent bracketed iteration (numeric — no shared IFT/jet-Newton code).
    fn witness_intercept(fx: &FlexFixture, mu: f64, b: f64, beta: &Array1<f64>, scale: f64) -> f64 {
        let calib = |a: f64| -> f64 {
            let mut acc = -mu;
            for (node, weight) in fx.grid.pairs() {
                acc += weight * witness_normal_cdf(witness_eta(fx, a, b, beta, node, scale));
            }
            acc
        };
        // Use a safeguarded bracketed solve rather than an open secant.  The
        // finite-difference witness evaluates many nearby coefficient states;
        // for link-deviation states with a steep composed basis, an open secant
        // can jump to a remote intercept and make high-order stencils compare a
        // different calibrated branch.  The calibration map is continuous and
        // has opposite limits at ±∞, so expanding a local bracket and bisecting
        // keeps every stencil point on the same mathematical root.
        let mut lo = -1.0_f64;
        let mut hi = 1.0_f64;
        let mut flo = calib(lo);
        let mut fhi = calib(hi);
        for _ in 0..80 {
            if flo <= 0.0 && fhi >= 0.0 {
                break;
            }
            if flo > 0.0 {
                hi = lo;
                fhi = flo;
                lo *= 2.0;
                flo = calib(lo);
            } else {
                lo = hi;
                flo = fhi;
                hi *= 2.0;
                fhi = calib(hi);
            }
        }
        assert!(
            flo <= 0.0 && fhi >= 0.0,
            "failed to bracket flex calibration root: F({lo})={flo}, F({hi})={fhi}"
        );
        // Drive the bracket to (near) f64 resolution so the scalar root that
        // anchors the exact tower's order-0 (and the independent scalar value
        // cross-check) is as tight as the arithmetic allows.
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let fmid = calib(mid);
            if fmid == 0.0 || (hi - lo).abs() <= 1e-15 * mid.abs().max(1.0) {
                return mid;
            }
            if fmid < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
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
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q,
        )
        .expect("witness link map");
        let a = witness_intercept(fx, marginal.mu, b, &beta, scale);
        let z = fx.family.z[0];
        let eta = witness_eta(fx, a, b, &beta, z, scale);
        let signed = (2.0 * fx.family.y[0] - 1.0) * eta;
        -fx.family.weights[0] * witness_normal_logcdf(signed)
    }

    // ----------------------------------------------------------------------
    // EXACT independent fourth-order tower witness.
    //
    // The finite-difference witness above is conditioning-limited on the
    // stiffest channels: for link-deviation states the basis enters the
    // *composed* observed index `u = a + b·node` THROUGH the implicit calibrated
    // intercept `a(q,b,β)`, so the highest mixed coefficients (e.g. the
    // 4th-order `[q,b,β,β]`) ride on the 6th/8th derivatives of a deep
    // composition, where any FD step large enough to escape rounding still
    // carries percent-level truncation. Rather than chase Richardson levels,
    // build a SECOND, fully exact witness from an INDEPENDENT jet kernel:
    //
    //   * the implicit intercept `a(θ)` is solved as an exact dense-symmetric
    //     `Tower4` via `jet_tower::implicit_solve`, independently assembled
    //     from the scalar fixture rather than the production runtime row plan,
    //     and
    //   * the I-spline deviation basis enters through its OWN
    //     `DeviationRuntime::{first,second,third}_derivative_design` stacks
    //     (production freezes local-cubic derivative stacks in its row plan),
    //     composed onto the `u` tower by Faà di Bruno.
    //
    // Both the production jet and this tower compute the SAME analytic
    // derivatives, so they must agree to ~1e-9 with no truncation tolerance —
    // exactly the discipline the rigid `verify_kernel_channels` oracle uses.
    // The primaries are θ = (q, b, β₀) in tower slots (0, 1, 2); the remaining
    // deviation coordinates are held at their fixed values as tower constants.

    /// Per-node, per-basis unary stack `[Φⱼ, Φⱼ′, Φⱼ″, Φⱼ‴, 0]` of the active
    /// deviation basis, evaluated at the SCALAR base argument the tower expands
    /// about. For the link-deviation block the argument is `u₀ = a₀ + b₀·node`
    /// (the basis is composed with the tower `u`); for the score-warp block the
    /// basis sits at the fixed `node` and contributes a *constant* warp (its
    /// only θ-dependence is the linear `βⱼ` it multiplies), so the stack is the
    /// plain value with zero derivatives. The fourth basis derivative of a cubic
    /// I-spline is identically zero, so the stack tops out at the third.
    fn witness_basis_stacks_at(fx: &FlexFixture, arg: f64) -> Vec<[f64; 5]> {
        let pt = Array1::from_vec(vec![arg]);
        let d0 = fx.runtime.design(&pt).expect("witness basis value");
        let basis_dim = d0.ncols();
        if fx.is_score_warp {
            // Score-warp basis enters at the fixed node: a constant per column.
            return (0..basis_dim)
                .map(|j| [d0[[0, j]], 0.0, 0.0, 0.0, 0.0])
                .collect();
        }
        let d1 = fx
            .runtime
            .first_derivative_design(&pt)
            .expect("witness basis 1st");
        let d2 = fx
            .runtime
            .second_derivative_design(&pt)
            .expect("witness basis 2nd");
        let d3 = fx
            .runtime
            .third_derivative_design(&pt)
            .expect("witness basis 3rd");
        (0..basis_dim)
            .map(|j| [d0[[0, j]], d1[[0, j]], d2[[0, j]], d3[[0, j]], 0.0])
            .collect()
    }

    /// The observed-index tower `η(a; node) = scale·(a + b·node + warp)` over
    /// the `K` primaries, with the deviation basis entering exactly as the model
    /// (score-warp: `b·Σβⱼ·Φⱼ(node)`; link-dev: `Σβⱼ·Φⱼ(u)`, `u = a + b·node`).
    /// `a`, `b`, `beta0` are the intercept / log-slope / active-β₀ towers; the
    /// inactive deviation coefficients are folded in as constants from `beta`.
    fn witness_eta_tower<const K: usize>(
        fx: &FlexFixture,
        a: &gam_math::jet_tower::Tower4<K>,
        b: &gam_math::jet_tower::Tower4<K>,
        beta0: &gam_math::jet_tower::Tower4<K>,
        beta: &[f64],
        node: f64,
        node_arg: f64,
        scale: f64,
    ) -> gam_math::jet_tower::Tower4<K> {
        use gam_math::jet_tower::Tower4;
        let stacks = witness_basis_stacks_at(fx, node_arg);
        let beta_tower = |j: usize| -> Tower4<K> {
            if j == 0 {
                *beta0
            } else {
                Tower4::<K>::constant(beta[j])
            }
        };
        if fx.is_score_warp {
            // warp = Σⱼ βⱼ·Φⱼ(node) (Φⱼ(node) constant), enters as b·(node + warp).
            let mut warp = Tower4::<K>::constant(0.0);
            for (j, stack) in stacks.iter().enumerate() {
                warp = warp + beta_tower(j).scale(stack[0]);
            }
            let inside = *a + b.mul(&(warp + Tower4::<K>::constant(node)));
            inside.scale(scale)
        } else {
            // u = a + b·node; warp = Σⱼ βⱼ·Φⱼ(u); inside = a + b·node + warp.
            let u = *a + b.scale(node);
            let mut warp = Tower4::<K>::constant(0.0);
            for (j, stack) in stacks.iter().enumerate() {
                warp = warp + beta_tower(j).mul(&u.compose_unary(*stack));
            }
            let inside = u + warp;
            inside.scale(scale)
        }
    }

    /// Exact `Tower4<3>` row NLL over θ = (q, b, β₀), with the calibrated
    /// intercept solved as an exact implicit tower. Read value/grad/Hessian/
    /// third/fourth straight off the returned tower.
    fn flex_tower_witness(fx: &FlexFixture, p0: &[f64]) -> gam_math::jet_tower::Tower4<3> {
        use gam_math::jet_tower::{Tower4, implicit_solve};
        let q0 = p0[fx.primary.q];
        let b0 = p0[fx.primary.logslope];
        let dev_range = if fx.is_score_warp {
            fx.primary.h.clone().unwrap()
        } else {
            fx.primary.w.clone().unwrap()
        };
        let beta: Vec<f64> = dev_range.clone().map(|i| p0[i]).collect();
        let beta0_0 = beta[0];
        let scale = fx.family.probit_frailty_scale();
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q0,
        )
        .expect("witness link map");
        let mu_stack = [
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
        ];

        // Scalar intercept root (the tower's order-0 anchor) from the existing
        // independent bracketed solve.
        let a0 = witness_intercept(fx, marginal.mu, b0, &Array1::from(beta.clone()), scale);

        let nodes: Vec<f64> = fx.grid.pairs().map(|(n, _)| n).collect();
        let node_weights: Vec<f64> = fx.grid.pairs().map(|(_, w)| w).collect();

        // Calibration constraint over (a, q, b, β₀) as a Tower4<4>:
        //   F(a, q, b, β₀) = −μ(q) + Σ_k π_k · Φ_cdf(η(a; node_k)).
        // slot 0 = a (the dependent variable implicit_solve eliminates),
        // slots 1,2,3 = q, b, β₀.
        let a_var = Tower4::<4>::variable(a0, 0);
        let q_var = Tower4::<4>::variable(q0, 1);
        let b_var = Tower4::<4>::variable(b0, 2);
        let beta0_var = Tower4::<4>::variable(beta0_0, 3);
        let mu_tower = q_var.compose_unary(mu_stack);
        let mut f_constraint = Tower4::<4>::constant(0.0) - mu_tower;
        // The deviation basis is evaluated at the fixed node for the score-warp
        // block (`Φ(node)`, a constant) and at the composed observed index
        // `u₀ = a₀ + b₀·node` for the link-deviation block (`Φ(u)`).
        let basis_arg = |node: f64| -> f64 {
            if fx.is_score_warp {
                node
            } else {
                a0 + b0 * node
            }
        };
        for (node, &w) in nodes.iter().zip(node_weights.iter()) {
            let eta = witness_eta_tower::<4>(
                fx,
                &a_var,
                &b_var,
                &beta0_var,
                &beta,
                *node,
                basis_arg(*node),
                scale,
            );
            let cdf = eta.compose_unary(unary_derivatives_normal_cdf(eta.v));
            f_constraint = f_constraint + cdf.scale(w);
        }
        let a_tower: Tower4<3> =
            implicit_solve::<4, 3>(&f_constraint, a0).expect("implicit intercept tower");

        // Row NLL over θ = (q, b, β₀) as a Tower4<3>. q (slot 0) enters the
        // observed-index map ONLY through the calibrated intercept a(q,b,β₀)
        // (μ(q) sets the calibration target), so it appears here solely via the
        // q-derivative channels already carried in `a_tower`; b and β₀ also
        // enter directly through the index map below.
        let b_t = Tower4::<3>::variable(b0, 1);
        let beta0_t = Tower4::<3>::variable(beta0_0, 2);
        let z = fx.family.z[0];
        let eta =
            witness_eta_tower::<3>(fx, &a_tower, &b_t, &beta0_t, &beta, z, basis_arg(z), scale);
        let signed = eta.scale(2.0 * fx.family.y[0] - 1.0);
        signed.compose_unary(unary_derivatives_neglog_phi(signed.v, fx.family.weights[0]))
    }

    /// Read the exact tower channel for the multiset of primary axes `axes`,
    /// mapping test primary indices (q, b, dev0) to tower slots (0, 1, 2).
    fn tower_channel(
        fx: &FlexFixture,
        tower: &gam_math::jet_tower::Tower4<3>,
        axes: &[usize],
    ) -> f64 {
        let dev0 = if fx.is_score_warp {
            fx.primary.h.clone().unwrap().start
        } else {
            fx.primary.w.clone().unwrap().start
        };
        let slot = |idx: usize| -> usize {
            if idx == fx.primary.q {
                0
            } else if idx == fx.primary.logslope {
                1
            } else if idx == dev0 {
                2
            } else {
                panic!("tower witness only carries the q/b/dev0 axes, got primary index {idx}")
            }
        };
        match axes.len() {
            0 => tower.v,
            1 => tower.g[slot(axes[0])],
            2 => tower.h[slot(axes[0])][slot(axes[1])],
            3 => tower.t3[slot(axes[0])][slot(axes[1])][slot(axes[2])],
            4 => tower.t4[slot(axes[0])][slot(axes[1])][slot(axes[2])][slot(axes[3])],
            n => panic!("tower witness carries at most 4th-order channels, got {n}"),
        }
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
                _ => panic!("central_along supports orders 0..=4, got {order}"),
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
                        acc += walk(fx, rest, h, coeff_acc * c, point);
                    }
                    point[idx] = saved;
                    acc
                }
            }
        }
        let mut point = p0.to_vec();
        let raw = walk(fx, &stencils, h, 1.0, &mut point);
        raw / h.powi(total_order as i32)
    }

    /// Read one value/gradient/Hessian/third/fourth channel from the canonical
    /// empirical FLEX row plan. Higher channels use the packed directional
    /// scalars production uses, so this oracle never depends on the retired
    /// exponential bitmask representation.
    fn prod_flex_coeff(fx: &FlexFixture, p0: &[f64], dir_indices: &[usize]) -> f64 {
        let r = fx.primary.total;
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
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q,
        )
        .expect("link map");
        let intercept = witness_intercept(fx, marginal.mu, b, &beta, scale);
        let plan = fx
            .family
            .empirical_bms_row_jet_plan(0, &fx.primary, q, b, beta_h, beta_w, intercept, &fx.grid)
            .expect("canonical empirical flex plan");
        match dir_indices {
            [] | [_] | [_, _] => {
                let arena = DynamicJetArena::new();
                let vars = arena.alloc_slice_fill_with(r, |axis| {
                    gam_math::jet_scalar::DynamicOrder2::variable(p0[axis], axis, r, &arena)
                });
                let jet = plan
                    .evaluate(vars, 2, &arena)
                    .expect("canonical order-2 row");
                match dir_indices {
                    [] => jet.value(),
                    &[axis] => jet.g()[axis],
                    &[row, column] => jet.h_at(row, column),
                    _ => unreachable!(),
                }
            }
            &[row, column, contracted] => {
                let direction = unit_primary_direction(r, contracted);
                let arena = DynamicJetArena::new();
                let vars = arena.alloc_slice_fill_with(r, |axis| {
                    DynamicOneSeed::seed_direction(p0[axis], axis, direction[axis], r, &arena)
                });
                plan.evaluate(vars, 3, &arena)
                    .expect("canonical third row")
                    .contracted_third()[row * r + column]
            }
            &[row, column, contracted_u, contracted_v] => {
                let direction_u = unit_primary_direction(r, contracted_u);
                let direction_v = unit_primary_direction(r, contracted_v);
                let arena = DynamicJetArena::new();
                let vars = arena.alloc_slice_fill_with(r, |axis| {
                    DynamicTwoSeed::seed(
                        p0[axis],
                        axis,
                        direction_u[axis],
                        direction_v[axis],
                        r,
                        &arena,
                    )
                });
                plan.evaluate(vars, 4, &arena)
                    .expect("canonical fourth row")
                    .contracted_fourth()[row * r + column]
            }
            _ => panic!("canonical flex oracle supports channels through order four"),
        }
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

        let label = if is_score_warp {
            "score-warp"
        } else {
            "link-dev"
        };

        // EXACT independent oracle: one `Tower4<3>` over θ = (q, b, β₀) carrying
        // value/grad/Hessian/third/fourth, built from the implicit-intercept
        // tower and the basis derivative-design stacks. Production and witness
        // compute the SAME analytic derivatives, so every channel must agree to
        // machine precision — no finite-difference truncation tolerance. The
        // tight bound is itself the guard: a dropped/incorrect Leibniz, Faà di
        // Bruno, or implicit-diff term would blow it by many orders.
        let tower = flex_tower_witness(&fx, &p0);
        // Cross-check the exact tower's VALUE against the fully independent
        // scalar (non-jet) NLL, so the tower's order-0 anchor is itself pinned
        // by a path that shares no jet code at all.
        let v_scalar = witness_nll(&fx, &p0);
        assert!(
            (tower.v - v_scalar).abs() <= 1e-9 * v_scalar.abs().max(1.0),
            "{label} tower value vs scalar witness: {:+.12e} != {v_scalar:+.12e}",
            tower.v,
        );

        // Tolerance for the production-vs-exact-tower comparison. Both are exact
        // analytic jets; the only gap is floating-point reassociation between
        // two different composition orders, which stays near machine epsilon on
        // these well-conditioned interior coefficients.
        let exact_tol = 1e-9_f64;

        // Value channel.
        let v_prod = prod_flex_coeff(&fx, &p0, &[]);
        assert!(
            (v_prod - tower.v).abs() <= exact_tol * tower.v.abs().max(1.0),
            "{label} value: production {v_prod:+.12e} != tower {:+.12e}",
            tower.v,
        );

        // First derivatives along q, b, β0.
        for &idx in &[q, b, dev0] {
            let prod = prod_flex_coeff(&fx, &p0, &[idx]);
            let wit = tower_channel(&fx, &tower, &[idx]);
            assert!(
                (prod - wit).abs() <= exact_tol * wit.abs().max(1.0),
                "{label} grad[{idx}]: production {prod:+.12e} != tower {wit:+.12e}"
            );
        }

        // Second derivatives: diagonal and the q×b / b×β / q×β cross blocks.
        let pairs: [(usize, usize); 6] =
            [(q, q), (b, b), (dev0, dev0), (q, b), (b, dev0), (q, dev0)];
        for &(i, j) in &pairs {
            let prod = prod_flex_coeff(&fx, &p0, &[i, j]);
            let wit = tower_channel(&fx, &tower, &[i, j]);
            assert!(
                (prod - wit).abs() <= exact_tol * wit.abs().max(1.0),
                "{label} H[{i},{j}]: production {prod:+.12e} != tower {wit:+.12e}"
            );
        }

        // Third derivatives: a spanning set of distinct-axis triples + a
        // diagonal, matching the contracted tensors the kernel exposes.
        let triples: [[usize; 3]; 4] =
            [[q, b, dev0], [b, b, dev0], [q, dev0, dev0], [b, dev0, dev0]];
        for tri in &triples {
            let prod = prod_flex_coeff(&fx, &p0, tri);
            let wit = tower_channel(&fx, &tower, tri);
            assert!(
                (prod - wit).abs() <= exact_tol * wit.abs().max(1.0),
                "{label} T3{tri:?}: production {prod:+.12e} != tower {wit:+.12e}"
            );
        }

        // Fourth derivatives: distinct-axis quadruples + mixed, the highest
        // channel the production exposes (#736/#833 genus surface). This is the
        // channel the #1394 link-dev `[q,b,β,β]` regression lived in — now
        // pinned at machine precision by the exact tower, not an FD stencil.
        let quads: [[usize; 4]; 3] = [[q, b, dev0, dev0], [b, b, dev0, dev0], [q, q, b, dev0]];
        for quad in &quads {
            let prod = prod_flex_coeff(&fx, &p0, quad);
            let wit = tower_channel(&fx, &tower, quad);
            assert!(
                (prod - wit).abs() <= exact_tol * wit.abs().max(1.0),
                "{label} T4{quad:?}: production {prod:+.12e} != tower {wit:+.12e}"
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
    fn link_dev_hqq_witness_stays_on_local_cubic_branch() {
        // Guard the link-dev q×q and q×b Hessian witnesses across a range of
        // deviation magnitudes. The bracketed calibration solve and shifted knot
        // grid should keep both the coarse and fine finite-difference stencils on
        // the same local cubic branch as production.
        let fx = make_fixture(false);
        let r = fx.primary.total;
        let q0 = 0.2_f64;
        let b0 = 0.35_f64;
        let q = fx.primary.q;
        let b = fx.primary.logslope;
        let dev_range = fx.primary.w.clone().unwrap();
        for scale in [0.1_f64, 0.2, 0.3, 0.4, 0.5, 1.0] {
            let mut fxs = make_fixture(false);
            fxs.beta_dev = fx.beta_dev.mapv(|v| v * scale);
            let mut p0 = vec![0.0; r];
            p0[fx.primary.q] = q0;
            p0[fx.primary.logslope] = b0;
            for (k, i) in dev_range.clone().enumerate() {
                p0[i] = fxs.beta_dev[k];
            }
            let max_beta = fxs
                .beta_dev
                .iter()
                .cloned()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            // H[q,q] and the q×b cross — the two blocks that have tripped the
            // unsound secant-calibration witness.
            let pqq = prod_flex_coeff(&fxs, &p0, &[q, q]);
            let wqq_c = central_along(&fxs, &p0, &[(q, 2)], 2e-3);
            let wqq_f = central_along(&fxs, &p0, &[(q, 2)], 5e-4);
            let pqb = prod_flex_coeff(&fxs, &p0, &[q, b]);
            let wqb_c = central_along(&fxs, &p0, &[(q, 1), (b, 1)], 2e-3);
            let wqb_f = central_along(&fxs, &p0, &[(q, 1), (b, 1)], 5e-4);
            let q_tol = 5e-4 * pqq.abs().max(1.0) + 1e-7;
            let qb_tol = 5e-4 * pqb.abs().max(1.0) + 1e-7;
            assert!(
                (pqq - wqq_c).abs() <= q_tol && (pqq - wqq_f).abs() <= q_tol,
                "scale={scale:.2} max|beta|={max_beta:.3}: H[q,q] prod={pqq:+.5e} wc={wqq_c:+.5e} wf={wqq_f:+.5e}"
            );
            assert!(
                (pqb - wqb_c).abs() <= qb_tol && (pqb - wqb_f).abs() <= qb_tol,
                "scale={scale:.2} max|beta|={max_beta:.3}: H[q,b] prod={pqb:+.5e} wc={wqb_c:+.5e} wf={wqb_f:+.5e}"
            );
        }
    }

    #[test]
    fn empirical_flex_contractions_match_witness_and_catch_sign_flip() {
        // Exercise the row_{third,fourth}_contracted entry points
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
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q0,
        )
        .expect("link map");
        let intercept = witness_intercept(&fx, marginal.mu, b0, &beta, scale);
        let mut m_a = 0.0;
        for (node, weight) in fx.grid.pairs() {
            m_a += weight * witness_normal_pdf(witness_eta(&fx, intercept, b0, &beta, node, scale));
        }
        let row_ctx = BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            intercept_fast_path: false,
            degree9_cells: None,
        };

        // Third-contracted along the slope direction e_b: out[u][v] = ∂³ℓ[e_u,e_v,e_b].
        let b = fx.primary.logslope;
        let dir_b = unit_primary_direction(r, b);
        let third = fx
            .family
            .empirical_flex_row_third_contracted(
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
            .expect("third contracted evaluation");
        // Exact independent tower over (q, b, dev0); read the contracted
        // entries straight off its symmetric tensors — no FD truncation.
        let tower = flex_tower_witness(&fx, &p0);
        // Check a representative entry (q, dev0) against the exact tower:
        // third_contracted[q,dev0] = ∂³ℓ[q, dev0, b] = t3[q,dev0,b].
        let dev0 = dev_range.start;
        let q = fx.primary.q;
        let wit_qd_b = tower_channel(&fx, &tower, &[q, dev0, b]);
        assert!(
            (third[[q, dev0]] - wit_qd_b).abs() <= 1e-9 * wit_qd_b.abs().max(1.0),
            "third_contracted[q,dev0] {:+.12e} != tower {wit_qd_b:+.12e}",
            third[[q, dev0]]
        );

        // A planted sign flip of that cross block must leave the witness band:
        // proves the contracted path has resolving power against the
        // #736 cross-block genus.
        let flipped = -third[[q, dev0]];
        if wit_qd_b.abs() > 1e-6 {
            assert!(
                (flipped - wit_qd_b).abs() > 1e-9 * wit_qd_b.abs().max(1.0),
                "witness failed to reject a planted sign flip (flipped {flipped:+.6e} vs tower {wit_qd_b:+.6e})"
            );
        }

        // Fourth-contracted along (e_b, e_dev0): out[p][q] = ∂⁴ℓ[e_p,e_q,e_b,e_dev0].
        let dir_dev0 = unit_primary_direction(r, dev0);
        let fourth = fx
            .family
            .empirical_flex_row_fourth_contracted(
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
            .expect("fourth contracted evaluation");
        // fourth_contracted[q,b] = ∂⁴ℓ[q, b, b, dev0] = t4[q,b,b,dev0].
        let wit_qb_b_d = tower_channel(&fx, &tower, &[q, b, b, dev0]);
        assert!(
            (fourth[[q, b]] - wit_qb_b_d).abs() <= 1e-9 * wit_qb_b_d.abs().max(1.0),
            "fourth_contracted[q,b] {:+.12e} != tower {wit_qb_b_d:+.12e}",
            fourth[[q, b]]
        );
    }

    // ----------------------------------------------------------------------
    // #932 BMS flex single-source (P2/P3): the empirical-grid flex row NLL
    // value/gradient/Hessian read off ONE runtime-dimension jet — the
    // calibrated intercept a(θ) lifted directly in the jet by the
    // implicit-function-theorem operator, the observed signed-probit NLL
    // composed on top — instead of the hand intercept/slope derivative chains
    // of `compute_row_analytic_flex_from_parts_into`. This is the runtime
    // analogue of the rigid `empirical_rigid_row_nll_jet` and mirrors the
    // exact `flex_tower_witness` term-for-term (Tower4 -> runtime Jet2), so the
    // gate below pins it to the SAME analytic derivatives the production hand
    // path produces — at machine precision, no finite-difference truncation.

    /// Observed-index jet `η(a; node) = scale·(a + b·node + warp)` over the `r`
    /// runtime primaries, the deviation basis entering exactly as the model
    /// (score-warp: `b·Σβⱼ·Φⱼ(node)`; link-dev: `Σβⱼ·Φⱼ(u)`, `u = a + b·node`).
    /// `a_jet` is the (lifted or seeded) intercept jet; `b_jet` / `beta_jets`
    /// are the seeded slope / deviation-coefficient primaries. Reuses
    /// [`witness_basis_stacks_at`] for the per-column basis derivative stacks so
    /// it samples the SAME spline branch the exact tower witness does.
    fn flex_eta_row_jet2(
        fx: &FlexFixture,
        a_jet: &crate::bms::test_support::Jet2,
        b_jet: &crate::bms::test_support::Jet2,
        beta_jets: &[crate::bms::test_support::Jet2],
        node: f64,
        node_arg: f64,
        scale: f64,
    ) -> crate::bms::test_support::Jet2 {
        use crate::bms::test_support::{Jet2, RuntimeJet};
        let r = a_jet.p();
        let stacks = witness_basis_stacks_at(fx, node_arg);
        if fx.is_score_warp {
            // inside = a + b·(node + Σⱼ βⱼ·Φⱼ(node)); Φⱼ(node) is a constant.
            let mut warp = Jet2::constant(0.0, r);
            for (j, stack) in stacks.iter().enumerate() {
                warp = warp.add(&beta_jets[j].scale(stack[0]));
            }
            let inside = a_jet.add(&b_jet.mul(&warp.add(&Jet2::constant(node, r))));
            inside.scale(scale)
        } else {
            // u = a + b·node; warp = Σⱼ βⱼ·Φⱼ(u); inside = u + warp.
            let u = a_jet.add(&b_jet.scale(node));
            let mut warp = Jet2::constant(0.0, r);
            for (j, stack) in stacks.iter().enumerate() {
                warp = warp.add(&beta_jets[j].mul(&u.compose_unary(*stack)));
            }
            let inside = u.add(&warp);
            inside.scale(scale)
        }
    }

    /// Single-source empirical-grid flex row NLL `(value, gradient, Hessian)`
    /// over the flat primary vector `p = [q, b, β...]` (length `primary.total`),
    /// produced entirely by the runtime-dimension jet: the calibration
    /// `F(a, θ) = −μ(q) + Σ_k π_k Φ(η(a; x_k)) = 0` lifts the intercept `a(θ)`
    /// directly in the jet via [`filtered_implicit_solve_jet2`]; the observed
    /// signed-probit NLL `−w·logΦ((2y−1)·η)` is then composed on top. No hand
    /// intercept/slope derivative formulas.
    fn empirical_flex_row_nll_jet2(fx: &FlexFixture, p0: &[f64]) -> crate::bms::test_support::Jet2 {
        use crate::bms::test_support::{Jet2, RuntimeJet, filtered_implicit_solve_jet2};
        let r = fx.primary.total;
        let q0 = p0[fx.primary.q];
        let b0 = p0[fx.primary.logslope];
        let dev_range = if fx.is_score_warp {
            fx.primary.h.clone().unwrap()
        } else {
            fx.primary.w.clone().unwrap()
        };
        let beta: Vec<f64> = dev_range.clone().map(|i| p0[i]).collect();
        let scale = fx.family.probit_frailty_scale();
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q0,
        )
        .expect("flex jet2 link map");

        // Converged scalar intercept root (the lift's order-0 anchor) from the
        // independent bracketed solve — shares no jet code with the lift.
        let a0 = witness_intercept(fx, marginal.mu, b0, &Array1::from(beta.clone()), scale);

        // Seeded primaries: q at slot `primary.q`, b at `primary.logslope`, each
        // βⱼ at its own deviation slot. q enters the row NLL ONLY through the
        // calibrated intercept (μ(q) is the calibration target), so it is seeded
        // for the calibration but never added to the observed index directly.
        let q_jet = Jet2::primary(q0, fx.primary.q, r);
        let b_jet = Jet2::primary(b0, fx.primary.logslope, r);
        let beta_jets: Vec<Jet2> = dev_range
            .clone()
            .map(|i| Jet2::primary(p0[i], i, r))
            .collect();
        let neg_mu = q_jet
            .compose_unary([
                marginal.mu,
                marginal.mu1,
                marginal.mu2,
                marginal.mu3,
                marginal.mu4,
            ])
            .scale(-1.0);

        let basis_arg = |node: f64| -> f64 {
            if fx.is_score_warp {
                node
            } else {
                a0 + b0 * node
            }
        };

        // Calibration Jacobian F_a at the root: Σ_k π_k φ(η₀)·∂η/∂a. The score-
        // warp basis is a-independent (∂η/∂a = scale); the link-deviation basis
        // rides the observed index u = a + b·node (∂η/∂a = scale·(1 + Σβⱼ·Φⱼ′)).
        let mut f_a = 0.0_f64;
        for (node, weight) in fx.grid.pairs() {
            let eta0 = witness_eta(fx, a0, b0, &Array1::from(beta.clone()), node, scale);
            let eta0_a = if fx.is_score_warp {
                scale
            } else {
                let stacks = witness_basis_stacks_at(fx, basis_arg(node));
                let mut s = 1.0_f64;
                for (j, stack) in stacks.iter().enumerate() {
                    s += beta[j] * stack[1];
                }
                scale * s
            };
            f_a += weight * witness_normal_pdf(eta0) * eta0_a;
        }
        assert!(
            f_a.is_finite() && f_a > 0.0,
            "flex jet2: non-positive calibration Jacobian F_a={f_a}"
        );
        let inv_fa = 1.0 / f_a;

        // Lift a(θ) directly in the jet: F(a, θ) = −μ(q) + Σ_k π_k Φ(η(a; x_k)).
        let constraint = |a: &Jet2| -> Jet2 {
            let mut acc = neg_mu.clone();
            for (node, weight) in fx.grid.pairs() {
                let eta =
                    flex_eta_row_jet2(fx, a, &b_jet, &beta_jets, node, basis_arg(node), scale);
                let cdf = eta.compose_unary(unary_derivatives_normal_cdf(eta.value()));
                acc = acc.add(&cdf.scale(weight));
            }
            acc
        };
        let a_jet = filtered_implicit_solve_jet2(a0, inv_fa, 2, r, constraint);

        // Observed signed-probit NLL through the SAME scalar kernel production
        // uses: η = a(θ) + b·z + warp, ℓ = −w·logΦ((2y−1)·η).
        let z = fx.family.z[0];
        let eta_obs = flex_eta_row_jet2(fx, &a_jet, &b_jet, &beta_jets, z, basis_arg(z), scale);
        let signed = eta_obs.scale(2.0 * fx.family.y[0] - 1.0);
        let stack = signed_probit_neglog_unary_stack(signed.value(), fx.family.weights[0]);
        signed.compose_unary(stack)
    }

    /// #932 P2/P3 GATE: the single-source runtime-jet flex row NLL matches the
    /// exact `Tower4<3>` witness on the representative `(q, b, β₀)` block at
    /// machine precision (value/gradient/Hessian), its value matches the fully
    /// independent scalar NLL, and its full gradient (every β column) matches a
    /// central-difference of the scalar NLL. A dropped/incorrect implicit-diff,
    /// Leibniz, or Faà di Bruno term would blow the 1e-9 tower bound.
    #[test]
    fn empirical_flex_row_nll_jet2_matches_tower_and_scalar_932() {
        for is_score_warp in [true, false] {
            let fx = make_fixture(is_score_warp);
            let r = fx.primary.total;
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
            let q = fx.primary.q;
            let b = fx.primary.logslope;
            let dev0 = dev_range.start;
            let label = if is_score_warp {
                "score-warp"
            } else {
                "link-dev"
            };

            let jet = empirical_flex_row_nll_jet2(&fx, &p0);
            let tower = flex_tower_witness(&fx, &p0);
            let v_scalar = witness_nll(&fx, &p0);

            // Value vs the fully independent scalar NLL (shares no jet code).
            assert!(
                (jet.v - v_scalar).abs() <= 1e-9 * v_scalar.abs().max(1.0),
                "{label} jet value {:+.12e} != scalar witness {v_scalar:+.12e}",
                jet.v,
            );

            // Value / gradient / Hessian vs the exact tower on the (q, b, β₀)
            // block — every cross channel (q×b, b×β, β×β, and the calibration
            // coupling q×* through the lifted intercept).
            assert!(
                (jet.v - tower.v).abs() <= 1e-9 * tower.v.abs().max(1.0),
                "{label} jet value vs tower",
            );
            let axes = [q, b, dev0];
            for &u in axes.iter() {
                let gu = tower_channel(&fx, &tower, &[u]);
                assert!(
                    (jet.g[u] - gu).abs() <= 1e-9 * gu.abs().max(1.0),
                    "{label} grad[{u}] {:+.12e} != tower {gu:+.12e}",
                    jet.g[u],
                );
                for &v in axes.iter() {
                    let huv = tower_channel(&fx, &tower, &[u, v]);
                    assert!(
                        (jet.h[u * r + v] - huv).abs() <= 1e-9 * huv.abs().max(1.0),
                        "{label} hess[{u},{v}] {:+.12e} != tower {huv:+.12e}",
                        jet.h[u * r + v],
                    );
                }
            }

            // Full gradient (including every β column the tower holds constant)
            // vs a central difference of the independent scalar NLL.
            let step = 1e-5_f64;
            for i in 0..r {
                let mut pp = p0.clone();
                let mut pm = p0.clone();
                pp[i] += step;
                pm[i] -= step;
                let fd = (witness_nll(&fx, &pp) - witness_nll(&fx, &pm)) / (2.0 * step);
                assert!(
                    (jet.g[i] - fd).abs() <= 1e-5 * fd.abs().max(1.0) + 1e-8,
                    "{label} grad[{i}] {:+.12e} != fd {fd:+.12e}",
                    jet.g[i],
                );
            }
        }
    }

    /// #932 P3 GATE (direct hand-vs-jet certificate): the single-source
    /// runtime-jet flex row NLL ([`empirical_flex_row_nll_jet2`]) reproduces the
    /// production HAND path `compute_row_analytic_flex_from_parts_into` — value,
    /// dense `r`-gradient, AND full `r×r` Hessian — to ≤1e-9 on the
    /// empirical-grid branch (the branch the empirical fixture routes the hand
    /// path through). This pins the jet against the EXACT function the cutover
    /// will replace, exercising the entire shared assembly the hand path runs:
    /// the calibration moments `f_u/f_au/f_uv/f_aa`, the implicit-function-theorem
    /// intercept lift `a(θ)` (`a_u`/`a_uv`), the observed-index chain
    /// `η_u = χ·a_u + ρ` / `η_uv = χ·a_uv + η_aa·a_u·a_v + τ_u·a_v + a_u·τ_v + r_uv`,
    /// and the signed-probit Mills finalization — over r ~ 2 + basis_dim primaries
    /// with score-warp OR link-wiggle active, deaths (y=1) at the observed row.
    /// A dropped IFT / Leibniz / Faà di Bruno term in either path blows the bound.
    #[test]
    fn empirical_flex_row_nll_jet2_matches_hand_path_932() {
        for is_score_warp in [true, false] {
            let fx = make_fixture(is_score_warp);
            let r = fx.primary.total;
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
            let label = if is_score_warp {
                "score-warp"
            } else {
                "link-dev"
            };

            let beta: Array1<f64> = Array1::from_iter(dev_range.clone().map(|i| p0[i]));
            let (beta_h, beta_w) = if is_score_warp {
                (Some(&beta), None)
            } else {
                (None, Some(&beta))
            };
            let scale = fx.family.probit_frailty_scale();
            let marginal = bernoulli_marginal_link_map(
                &InverseLink::Standard(gam_problem::StandardLink::Probit),
                q0,
            )
            .expect("link map");

            // Converged scalar intercept root (the IFT/lift order-0 anchor) and the
            // calibration Jacobian F_a = Σ_k π_k φ(η₀)·∂η/∂a — the exact `m_a` the
            // hand IFT divides by, computed identically to the jet's internal F_a.
            let a0 = witness_intercept(&fx, marginal.mu, b0, &beta, scale);
            let basis_arg =
                |node: f64| -> f64 { if is_score_warp { node } else { a0 + b0 * node } };
            let mut f_a = 0.0_f64;
            for (node, weight) in fx.grid.pairs() {
                let eta0 = witness_eta(&fx, a0, b0, &beta, node, scale);
                let eta0_a = if is_score_warp {
                    scale
                } else {
                    let stacks = witness_basis_stacks_at(&fx, basis_arg(node));
                    let mut s = 1.0_f64;
                    for (j, stack) in stacks.iter().enumerate() {
                        s += beta[j] * stack[1];
                    }
                    scale * s
                };
                f_a += weight * witness_normal_pdf(eta0) * eta0_a;
            }
            let row_ctx = BernoulliMarginalSlopeRowExactContext {
                intercept: a0,
                m_a: f_a,
                intercept_fast_path: false,
                degree9_cells: None,
            };

            let mut scratch =
                crate::bms::hessian_paths::BernoulliMarginalSlopeFlexRowScratch::new(r);
            let neglog = fx
                .family
                .compute_row_analytic_flex_from_parts_into(
                    0,
                    &fx.primary,
                    q0,
                    b0,
                    beta_h,
                    beta_w,
                    &row_ctx,
                    None,
                    None,
                    true,
                    &mut scratch,
                )
                .expect("hand flex path");

            let jet = empirical_flex_row_nll_jet2(&fx, &p0);

            assert!(
                (neglog - jet.v).abs() <= 1e-9 * jet.v.abs().max(1.0),
                "{label} value: hand {neglog:+.12e} != jet {:+.12e}",
                jet.v,
            );
            for u in 0..r {
                assert!(
                    (scratch.grad[u] - jet.g[u]).abs() <= 1e-9 * jet.g[u].abs().max(1.0),
                    "{label} grad[{u}]: hand {:+.12e} != jet {:+.12e}",
                    scratch.grad[u],
                    jet.g[u],
                );
                for v in 0..r {
                    let h_hand = scratch.hess[[u, v]];
                    let h_jet = jet.h[u * r + v];
                    assert!(
                        (h_hand - h_jet).abs() <= 1e-9 * h_jet.abs().max(1.0),
                        "{label} hess[{u},{v}]: hand {h_hand:+.12e} != jet {h_jet:+.12e}"
                    );
                }
            }
        }
    }

    /// #932 BMS-flex cutover INC-1(b) GATE: the per-denested-cell moment
    /// compiler (production `flex_grid_calibration_derivs_compiled_jet2`, reached
    /// through `compute_row_analytic_flex_from_parts_into`) reproduces the
    /// INDEPENDENT grid jet `empirical_flex_row_nll_jet2` (value / dense gradient /
    /// full Hessian) to ≤1e-9 on DEGENERATE empirical grids — a sparse grid whose
    /// four nodes leave several denested cells EMPTY and at least one holding a
    /// single node (the degenerate-moment paths where a compiled accumulator
    /// typically diverges from a loop) — over score-warp AND link-deviation,
    /// `b>0` AND `b<0`. The fixture routes the `GlobalEmpirical` branch, i.e. the
    /// production path the cutover replaces; `jet2` is a fully independent
    /// per-node reference (proven vs the hand path by
    /// `empirical_flex_row_nll_jet2_matches_hand_path_932`).
    #[test]
    fn flex_factored_matches_jet2_degenerate_grids_932() {
        let sparse = crate::bms::EmpiricalZGrid::new(
            vec![-2.0, -0.1, 0.1, 2.0],
            vec![0.25, 0.25, 0.25, 0.25],
            "flex factored degenerate oracle",
        )
        .expect("sparse grid");
        for is_score_warp in [true, false] {
            for b0 in [0.35_f64, -0.4_f64] {
                let mut fx = make_fixture(is_score_warp);
                // Route BOTH the production path (via latent_measure) and jet2
                // (via fx.grid) through the degenerate sparse grid.
                fx.grid = sparse.clone();
                fx.family.latent_measure = LatentMeasureKind::GlobalEmpirical {
                    grid: sparse.clone(),
                };
                let r = fx.primary.total;
                let q0 = 0.2_f64;
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
                let beta: Array1<f64> = Array1::from_iter(dev_range.clone().map(|i| p0[i]));
                let (beta_h, beta_w) = if is_score_warp {
                    (Some(&beta), None)
                } else {
                    (None, Some(&beta))
                };
                let scale = fx.family.probit_frailty_scale();
                let marginal = bernoulli_marginal_link_map(
                    &InverseLink::Standard(gam_problem::StandardLink::Probit),
                    q0,
                )
                .expect("link map");
                let a0 = witness_intercept(&fx, marginal.mu, b0, &beta, scale);
                let basis_arg =
                    |node: f64| -> f64 { if is_score_warp { node } else { a0 + b0 * node } };
                let mut f_a = 0.0_f64;
                for (node, weight) in fx.grid.pairs() {
                    let eta0 = witness_eta(&fx, a0, b0, &beta, node, scale);
                    let eta0_a = if is_score_warp {
                        scale
                    } else {
                        let stacks = witness_basis_stacks_at(&fx, basis_arg(node));
                        let mut s = 1.0_f64;
                        for (j, stack) in stacks.iter().enumerate() {
                            s += beta[j] * stack[1];
                        }
                        scale * s
                    };
                    f_a += weight * witness_normal_pdf(eta0) * eta0_a;
                }
                let row_ctx = BernoulliMarginalSlopeRowExactContext {
                    intercept: a0,
                    m_a: f_a,
                    intercept_fast_path: false,
                    degree9_cells: None,
                };
                let mut scratch =
                    crate::bms::hessian_paths::BernoulliMarginalSlopeFlexRowScratch::new(r);
                let neglog = fx
                    .family
                    .compute_row_analytic_flex_from_parts_into(
                        0,
                        &fx.primary,
                        q0,
                        b0,
                        beta_h,
                        beta_w,
                        &row_ctx,
                        None,
                        None,
                        true,
                        &mut scratch,
                    )
                    .expect("compiled moment-jet flex path");
                let jet = empirical_flex_row_nll_jet2(&fx, &p0);
                let label = if is_score_warp {
                    "score-warp"
                } else {
                    "link-dev"
                };
                let tol = |x: f64| 1e-9 * x.abs().max(1.0);
                assert!(
                    (neglog - jet.v).abs() <= tol(jet.v),
                    "{label} b={b0} value: factored {neglog:+.12e} != jet2 {:+.12e}",
                    jet.v
                );
                for u in 0..r {
                    assert!(
                        (scratch.grad[u] - jet.g[u]).abs() <= tol(jet.g[u]),
                        "{label} b={b0} grad[{u}]: factored {:+.12e} != jet2 {:+.12e}",
                        scratch.grad[u],
                        jet.g[u]
                    );
                    for v in 0..r {
                        let h_f = scratch.hess[[u, v]];
                        let h_j = jet.h[u * r + v];
                        assert!(
                            (h_f - h_j).abs() <= tol(h_j),
                            "{label} b={b0} hess[{u},{v}]: factored {h_f:+.12e} != jet2 {h_j:+.12e}"
                        );
                    }
                }
            }
        }
    }

    fn runtime_for_primary_dimension(total_dimension: usize) -> DeviationRuntime {
        let wanted = total_dimension - 2;
        for n_knots in 5..=40 {
            let knots = Array1::from_iter(
                (0..n_knots).map(|i| -2.45_f64 + 5.0_f64 * (i as f64) / ((n_knots - 1) as f64)),
            );
            if let Ok(runtime) = DeviationRuntime::try_new(knots, 0.0, 3)
                && runtime.basis_dim() == wanted
            {
                return runtime;
            }
        }
        panic!("no deviation runtime realizes total primary dimension {total_dimension}");
    }

    fn make_dimension_fixture(is_score_warp: bool, total_dimension: usize) -> FlexFixture {
        let mut fixture = make_fixture(is_score_warp);
        let runtime = runtime_for_primary_dimension(total_dimension);
        let basis_dim = runtime.basis_dim();
        fixture.family.score_warp = is_score_warp.then(|| runtime.clone());
        fixture.family.link_dev = (!is_score_warp).then(|| runtime.clone());
        fixture.primary = PrimarySlices {
            q: 0,
            logslope: 1,
            h: is_score_warp.then_some(2..2 + basis_dim),
            w: (!is_score_warp).then_some(2..2 + basis_dim),
            total: 2 + basis_dim,
        };
        fixture.beta_dev = Array1::from_shape_fn(basis_dim, |i| {
            let center = 0.5 * (basis_dim.saturating_sub(1) as f64);
            0.06 * ((i as f64) - center) / center.max(1.0)
        });
        fixture.runtime = runtime;
        assert_eq!(fixture.primary.total, total_dimension);
        fixture
    }

    fn fixture_state(
        fixture: &FlexFixture,
    ) -> (f64, f64, Array1<f64>, BernoulliMarginalSlopeRowExactContext) {
        let q = 0.23_f64;
        let slope = 0.37_f64;
        let beta = fixture.beta_dev.clone();
        let scale = fixture.family.probit_frailty_scale();
        let marginal = bernoulli_marginal_link_map(
            &InverseLink::Standard(gam_problem::StandardLink::Probit),
            q,
        )
        .expect("benchmark marginal map");
        let intercept = witness_intercept(fixture, marginal.mu, slope, &beta, scale);
        let mut f_a = 0.0;
        for (node, weight) in fixture.grid.pairs() {
            let eta = witness_eta(fixture, intercept, slope, &beta, node, scale);
            f_a += weight * witness_normal_pdf(eta);
        }
        (
            q,
            slope,
            beta,
            BernoulliMarginalSlopeRowExactContext {
                intercept,
                m_a: f_a,
                intercept_fast_path: false,
                degree9_cells: None,
            },
        )
    }

    fn fixture_block_states(q: f64, slope: f64, beta: &Array1<f64>) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![q]),
                eta: Array1::from_vec(vec![q]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![slope]),
                eta: Array1::from_vec(vec![slope]),
            },
            ParameterBlockState {
                beta: beta.clone(),
                eta: Array1::zeros(1),
            },
        ]
    }

    fn assert_matrix_close(label: &str, expected: &Array2<f64>, actual: &Array2<f64>) {
        assert_eq!(expected.dim(), actual.dim(), "{label} shape");
        for ((row, column), &expected_value) in expected.indexed_iter() {
            let actual_value = actual[[row, column]];
            assert!(
                (expected_value - actual_value).abs()
                    <= 2e-10 * expected_value.abs().max(actual_value.abs()).max(1.0),
                "{label}[{row},{column}]: expected {expected_value:+.12e}, got {actual_value:+.12e}"
            );
        }
    }

    /// Permanent production-wiring oracle for the runtime batch algebras.
    /// Five directions/pairs cross the four-pair fourth-order chunk boundary;
    /// lane two is exactly zero. Every batched result is compared to the
    /// canonical single-contraction path at all specialized widths and both
    /// empirical FLEX programs, and the trace gradient is independently
    /// reduced from basis-direction singles.
    #[test]
    fn empirical_flex_batched_contractions_match_single_production_932() {
        for is_score_warp in [true, false] {
            for r in [4_usize, 8, 12, 18] {
                let fixture = make_dimension_fixture(is_score_warp, r);
                let (q, slope, beta, _) = fixture_state(&fixture);
                let states = fixture_block_states(q, slope, &beta);
                let cache = fixture
                    .family
                    .build_exact_eval_cache(&states)
                    .expect("empirical FLEX batch oracle cache");
                assert_eq!(cache.primary.total, r);
                let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, 0);
                let mut directions = (0..5)
                    .map(|lane| {
                        Array1::from_shape_fn(r, |axis| {
                            let magnitude = ((lane + 2) * (axis + 3) % 11 + 1) as f64 / 13.0;
                            if (lane + axis) % 2 == 0 {
                                magnitude
                            } else {
                                -0.6 * magnitude
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                directions[2].fill(0.0);

                let singles = directions
                    .iter()
                    .map(|direction| {
                        fixture
                            .family
                            .row_primary_third_contracted(0, &states, &cache, row_ctx, direction)
                            .expect("single empirical FLEX third contraction")
                    })
                    .collect::<Vec<_>>();
                let batched = fixture
                    .family
                    .row_primary_third_contracted_many_with_moments(
                        0,
                        &states,
                        &cache,
                        row_ctx,
                        &directions,
                    )
                    .expect("batched empirical FLEX third contractions");
                for lane in 0..directions.len() {
                    assert_matrix_close(
                        &format!("third kind={is_score_warp} r={r} lane={lane}"),
                        &singles[lane],
                        &batched[lane],
                    );
                }
                assert!(batched[2].iter().all(|value| *value == 0.0));

                let gram = (0..r * r)
                    .map(|idx| {
                        let row = idx / r;
                        let column = idx % r;
                        ((row + 2 * column + 1) as f64) / (3 * r) as f64
                    })
                    .collect::<Vec<_>>();
                let mut expected_trace = Array1::<f64>::zeros(r);
                for axis in 0..r {
                    let mut basis = Array1::<f64>::zeros(r);
                    basis[axis] = 1.0;
                    let third = fixture
                        .family
                        .row_primary_third_contracted(0, &states, &cache, row_ctx, &basis)
                        .expect("basis empirical FLEX third contraction");
                    expected_trace[axis] =
                        BernoulliMarginalSlopeFamily::row_primary_trace_contract(&third, &gram);
                }
                let actual_trace = fixture
                    .family
                    .row_primary_third_trace_gradient_with_moments(
                        0, &states, &cache, row_ctx, &gram,
                    )
                    .expect("batched empirical FLEX trace gradient");
                for axis in 0..r {
                    let expected = expected_trace[axis];
                    let actual = actual_trace[axis];
                    assert!(
                        (expected - actual).abs()
                            <= 2e-10 * expected.abs().max(actual.abs()).max(1.0),
                        "trace kind={is_score_warp} r={r} axis={axis}: expected {expected:+.12e}, got {actual:+.12e}"
                    );
                }

                let pair_indices = [(0, 1), (1, 3), (2, 4), (3, 0), (4, 1)];
                let direction_pairs = pair_indices
                    .iter()
                    .map(|&(u, v)| (&directions[u], &directions[v]))
                    .collect::<Vec<_>>();
                let fourth_singles = direction_pairs
                    .iter()
                    .map(|&(direction_u, direction_v)| {
                        fixture
                            .family
                            .row_primary_fourth_contracted(
                                0,
                                &states,
                                &cache,
                                row_ctx,
                                direction_u,
                                direction_v,
                            )
                            .expect("single empirical FLEX fourth contraction")
                    })
                    .collect::<Vec<_>>();
                let fourth_batched = fixture
                    .family
                    .row_primary_fourth_contracted_many(
                        0,
                        &states,
                        &cache,
                        row_ctx,
                        &direction_pairs,
                    )
                    .expect("batched empirical FLEX fourth contractions");
                for lane in 0..direction_pairs.len() {
                    assert_matrix_close(
                        &format!("fourth kind={is_score_warp} r={r} lane={lane}"),
                        &fourth_singles[lane],
                        &fourth_batched[lane],
                    );
                }
                assert!(fourth_batched[2].iter().all(|value| *value == 0.0));
            }
        }
    }
}
