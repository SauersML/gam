//! The rigid per-row `RowKernel<4>` implementation and its Jacobian-action
//! assembly helpers: the memory-efficient row-at-a-time kernel used on the
//! no-flex hot path.

use super::*;

use crate::families::jet_scalar::{JetScalar, OneSeed, TwoSeed};

// ── Static-sparsity (v,g,H) scalar (#932 perf) ─────────────────────────
//
// The rigid row primaries are `(q0, q1, qd1, g)`. Three of them — `q0`, `q1`,
// `qd1` — enter every INDEX intermediate (`eta0 = q0·c + s·g·z`, `eta1`, `ad1`)
// AFFINELY: each is a single linear coefficient times the shared curvature
// factor `c(g)`. So the *index-space* second derivative between any two of those
// three linear primaries is structurally zero — the only curvature they acquire
// is the rank-1 outer-function term `f''·(∂η/∂q)·(∂η/∂q)` created at the leaf
// composes (logΦ / logφ / log), which is genuinely dense and computed normally.
//
// [`SparseOrder2`] encodes "which axes are linear" as a compile-time bitmask and
// ELIDES exactly the provably-zero work: the linear×linear self-Hessian READS in
// `mul`/`compose_unary` (a linear axis carries a structurally-zero self-Hessian
// block by the index-affine contract). Everything else — the gradient, the
// linear×nonlinear cross curvature, and the dense leaf `g⊗g` — is computed bit
// for bit as the dense [`Order2`]. The family writes the row NLL ONCE against
// [`JetScalar`]; this is just a different instantiation that the compiler
// monomorphizes into sparse-optimal code (proven: a single `mul` drops from 63
// to 21 floating-point instructions in the emitted IR/asm). No hand chain rule,
// so the #736 cross-block-sign-flip bug genus cannot reappear.
//
// CONTRACT: an axis may be declared linear only when the program never forms
// curvature between it and another linear axis (the linear×linear index Hessian
// stays zero for the life of every intermediate). [`SparseOrder2::check_contract`]
// debug-asserts this at every elision site, so a wrong declaration panics loudly
// in debug/test builds rather than silently dropping curvature.

/// Bitmask of which `K=4` rigid primaries enter linearly: bit `a` set ⇒ axis `a`
/// is linear. `q0 (0), q1 (1), qd1 (2)` are linear; `g (3)` is nonlinear.
pub(crate) const RIGID_LINEAR_MASK: u32 = (1 << 0) | (1 << 1) | (1 << 2);

#[inline(always)]
const fn axis_is_linear(mask: u32, a: usize) -> bool {
    (mask >> a) & 1 == 1
}

/// Order-2 (value/gradient/Hessian) jet over `K=4` primaries, with compile-time
/// static sparsity: the linear×linear self-Hessian block (axes both set in
/// `LIN`) is never read in `mul`/`compose_unary` because it is structurally
/// zero. Bit-identical to [`Order2<4>`] on every channel a consumer reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseOrder2<const LIN: u32> {
    v: f64,
    grad: [f64; 4],
    hess: [[f64; 4]; 4],
}

impl<const LIN: u32> SparseOrder2<LIN> {
    #[inline]
    pub(crate) fn g(&self) -> [f64; 4] {
        self.grad
    }
    #[inline]
    pub(crate) fn h(&self) -> [[f64; 4]; 4] {
        self.hess
    }

    /// Guard for the index-affine contract: the linear×linear self-Hessian
    /// block must be exactly zero wherever we elide its read.
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..4 {
            if axis_is_linear(LIN, i) {
                for j in 0..4 {
                    if axis_is_linear(LIN, j) {
                        assert!(
                            self.hess[i][j] == 0.0,
                            "static-sparsity contract violated: linear×linear Hessian h[{i}][{j}]={} != 0 (axes {i},{j} both declared linear but the program forms curvature between them)",
                            self.hess[i][j]
                        );
                    }
                }
            }
        }
    }
}

impl<const LIN: u32> JetScalar<4> for SparseOrder2<LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            grad: [0.0; 4],
            hess: [[0.0; 4]; 4],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut grad = [0.0; 4];
        grad[axis] = 1.0;
        Self {
            v: x,
            grad,
            hess: [[0.0; 4]; 4],
        }
    }
    fn value(&self) -> f64 {
        self.v
    }
    // add / sub / scale are uniform-dense: after a leaf compose, a linear×linear
    // entry can be legitimately nonzero (the dense `f''·g⊗g` term), so these must
    // touch every entry. The elision lives ONLY in the h-reads of mul/compose.
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..4 {
            r.grad[i] += o.grad[i];
            for j in 0..4 {
                r.hess[i][j] += o.hess[i][j];
            }
        }
        r
    }
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        let mut r = *self;
        r.v *= s;
        for i in 0..4 {
            r.grad[i] *= s;
            for j in 0..4 {
                r.hess[i][j] *= s;
            }
        }
        r
    }
    fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        // Elision precondition: we skip reading a.hess/b.hess on the
        // linear×linear block — assert those reads would indeed be zero.
        a.check_contract();
        b.check_contract();
        let mut r = Self::constant(a.v * b.v);
        for i in 0..4 {
            r.grad[i] = a.v * b.grad[i] + a.grad[i] * b.v;
        }
        // H_out[i][j] = a.v·H_b + a.g[i]·b.g[j] + a.g[j]·b.g[i] + H_a·b.v. The
        // self-Hessian reads H_a[i][j]/H_b[i][j] are structurally zero when both
        // i,j are linear (contract), so they are elided; the `g⊗g` product-rule
        // curvature term is always kept.
        for i in 0..4 {
            for j in 0..4 {
                let mut hij = a.grad[i] * b.grad[j] + a.grad[j] * b.grad[i];
                if !axis_is_linear(LIN, i) || !axis_is_linear(LIN, j) {
                    hij += a.v * b.hess[i][j] + a.hess[i][j] * b.v;
                }
                r.hess[i][j] = hij;
            }
        }
        r
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Elision precondition: skipping self.hess on the linear×linear block.
        self.check_contract();
        let (f1, f2) = (d[1], d[2]);
        let mut r = Self::constant(d[0]);
        for i in 0..4 {
            r.grad[i] = f1 * self.grad[i];
        }
        // H_out[i][j] = f1·H_self[i][j] + f2·g_i·g_j. The dense `f2·g⊗g` term is
        // always kept (this is the leaf curvature, nonzero on linear×linear); the
        // `f1·H_self` read skips linear×linear (structurally zero by contract).
        for i in 0..4 {
            for j in 0..4 {
                let mut hij = f2 * self.grad[i] * self.grad[j];
                if !axis_is_linear(LIN, i) || !axis_is_linear(LIN, j) {
                    hij += f1 * self.hess[i][j];
                }
                r.hess[i][j] = hij;
            }
        }
        r
    }
}

// ── RowKernel<4> implementation ───────────────────────────────────────

pub(crate) struct SurvivalMarginalSlopeRowKernel {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) slices: BlockSlices,
}

impl SurvivalMarginalSlopeRowKernel {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Self {
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

pub(crate) fn rigid_row_kernel_primaries(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<[f64; 4], String> {
    let q_geom = family.row_dynamic_q_values(row, block_states)?;
    Ok([q_geom.q0, q_geom.q1, q_geom.qd1, block_states[2].eta[row]])
}

/// The scalar-independent per-row inputs the generic rigid row NLL
/// ([`rigid_row_nll`]) consumes: the f64 quantities computed ONCE per row and
/// reused across every [`JetScalar`] instantiation (value/grad/Hessian, the
/// contracted third/fourth, and the dense `Tower4<4>` oracle/all-axes path).
pub(crate) struct RigidRowInputs {
    pub(crate) row: usize,
    pub(crate) wi: f64,
    pub(crate) di: f64,
    pub(crate) z_sum: f64,
    pub(crate) covariance_ones: f64,
    pub(crate) probit_scale: f64,
    pub(crate) qd1_lower: f64,
}

/// Resolve the row's scalar inputs (shared-score summary, probit scale,
/// monotonicity floor). Pure f64 — no jet arithmetic.
pub(crate) fn rigid_row_inputs(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
    context: &str,
) -> Result<RigidRowInputs, String> {
    let (z_sum, covariance_ones) = family.exact_shared_score_summary(row, block_states, context)?;
    Ok(RigidRowInputs {
        row,
        wi: family.weights[row],
        di: family.event[row],
        z_sum,
        covariance_ones,
        probit_scale: family.probit_frailty_scale(),
        qd1_lower: family.time_derivative_lower_bound(),
    })
}

/// The rigid survival marginal-slope row negative log-likelihood, written ONCE
/// over a generic [`JetScalar<4>`] so a single expression yields every
/// derivative channel a consumer needs (#736/#932 single-source contract):
///
/// * `S = Order2<4>`  → `(v, g, H)` (inner Newton / `row_kernel`, 168 B/row),
/// * `S = OneSeed<4>` → contracted third `Σ_c ℓ_{abc} dir_c`
///   (`row_third_contracted`),
/// * `S = TwoSeed<4>` → contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`
///   (`row_fourth_contracted`),
/// * `S = Tower4<4>`  → the full dense `(v,g,H,t3,t4)` oracle / #979 all-axes
///   build-once path ([`rigid_row_kernel_nll_tower`]).
///
/// The four primaries are `(q0, q1, qd1, g)`. From them
///   `c(g) = √(1 + (s·g)²·covariance_ones)`,
///   `η0 = q0·c + s·g·z_sum`, `η1 = q1·c + s·g·z_sum`, `ad1 = qd1·c`,
/// and the NLL is `+w logΦ(−η0) + w(1−d) logΦ(−η1) − w·d·(logφ(η1) + log ad1)`,
/// each special-function stack supplied as a hand-certified `[f64; 5]` through
/// [`JetScalar::compose_unary`] — there is no separate hand-derivative channel.
pub(crate) fn rigid_row_nll<S: JetScalar<4>>(
    vars: &[S; 4],
    inputs: &RigidRowInputs,
) -> Result<S, String> {
    let RigidRowInputs {
        row,
        wi,
        di,
        z_sum,
        covariance_ones,
        probit_scale,
        qd1_lower,
    } = *inputs;

    let q0 = &vars[0];
    let q1 = &vars[1];
    let qd1 = &vars[2];
    let g = &vars[3];

    let observed_g = g.scale(probit_scale);
    let one_plus_b2 = observed_g
        .mul(&observed_g)
        .scale(covariance_ones)
        .add(&S::constant(1.0));
    let c = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.value()));

    let observed_gz = observed_g.scale(z_sum);
    let eta0 = q0.mul(&c).add(&observed_gz);
    let eta1 = q1.mul(&c).add(&observed_gz);
    let ad1 = qd1.mul(&c);

    if survival_derivative_guard_violated(qd1.value(), qd1_lower) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated at row {row}: raw time derivative={:.3e} must be at least derivative_guard={:.3e}; transformed time derivative={:.3e}",
                qd1.value(), qd1_lower, ad1.value()
            ),
        }
        .into());
    }

    // Mirror the exact closed-form contract
    // (`signed_probit_neglog_derivatives_up_to_fourth`): the saturated `+∞`
    // tail is the legitimate zero-survival limit, but `-∞`/NaN signed margins
    // are domain failures that must surface as an error rather than being
    // masked into a NaN/∞-laden derivative stack by `unary_derivatives_neglog_phi`.
    // The guard respects zero weight (those terms drop out entirely).
    let reject_nonfinite_margin = |margin: f64, weight: f64| -> Result<(), String> {
        if weight != 0.0 && margin != f64::INFINITY && !margin.is_finite() {
            Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "non-finite signed margin in rigid survival marginal-slope row tower at row {row}: {margin}"
                ),
            }
            .into())
        } else {
            Ok(())
        }
    };

    let neg_eta0 = eta0.neg();
    reject_nonfinite_margin(neg_eta0.value(), wi)?;
    let entry = neg_eta0
        .compose_unary(unary_derivatives_neglog_phi(neg_eta0.value(), wi))
        .scale(-1.0);

    let neg_eta1 = eta1.neg();
    reject_nonfinite_margin(neg_eta1.value(), wi * (1.0 - di))?;
    let exit =
        neg_eta1.compose_unary(unary_derivatives_neglog_phi(neg_eta1.value(), wi * (1.0 - di)));

    let event_density = if di > 0.0 {
        eta1.compose_unary(unary_derivatives_log_normal_pdf(eta1.value()))
            .scale(-wi * di)
    } else {
        S::constant(0.0)
    };

    let time_deriv = if di > 0.0 {
        ad1.compose_unary(unary_derivatives_log(ad1.value()))
            .scale(-wi * di)
    } else {
        S::constant(0.0)
    };

    Ok(exit.add(&entry).add(&event_density).add(&time_deriv))
}

/// Thin `Tower4<4>` wrapper over the single-source [`rigid_row_nll`]: evaluates
/// the SAME expression at the all-channels dense scalar to obtain the full
/// `(v, g, H, t3, t4)` in one pass. Used by the `RowNllProgram<4>` impl (the
/// contraction.rs helpers) and the #979 all-axes build-once path
/// ([`SurvivalMarginalSlopeRowKernel::build_row_towers`]), which genuinely
/// reuses a row's `t3`/`t4` across every coefficient axis.
pub(crate) fn rigid_row_kernel_nll_tower(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
    p: &[crate::families::jet_tower::Tower4<4>; 4],
    context: &str,
) -> Result<crate::families::jet_tower::Tower4<4>, String> {
    let inputs = rigid_row_inputs(family, block_states, row, context)?;
    rigid_row_nll(p, &inputs)
}

impl crate::families::jet_tower::RowNllProgram<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        rigid_row_kernel_primaries(&self.family, &self.block_states, row)
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<4>; 4],
    ) -> Result<crate::families::jet_tower::Tower4<4>, String> {
        rigid_row_kernel_nll_tower(
            &self.family,
            &self.block_states,
            row,
            p,
            "survival marginal-slope rigid row tower",
        )
    }
}

impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        // #932: value/gradient/Hessian derive from the SAME single-sourced row
        // NLL (`rigid_row_nll`) — no dense `Tower4<4>` (256-entry `t4` + 64-entry
        // `t3`) is built and discarded on the inner-Newton hot path. Instantiated
        // at the static-sparsity `SparseOrder2<RIGID_LINEAR_MASK>` scalar: q0/q1/
        // qd1 enter the index quantities affinely, so their linear×linear self-
        // Hessian block is structurally zero and the compiler elides those reads
        // (proven 63→21 FP ops per `mul`), recovering the hand-kernel's sparsity
        // throughput WITHOUT a hand chain rule (so no #736 sign-flip bug genus).
        // Bit-identical to the dense `Order2<4>` / `Tower4<4>` channels by the
        // `rigid_row_kernel_agrees_with_jet_tower_program_all_channels` oracle and
        // `rigid_row_kernel_sparse_matches_dense_932`.
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row kernel",
        )?;
        let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
        let vars: [SparseOrder2<RIGID_LINEAR_MASK>; 4] =
            std::array::from_fn(|a| SparseOrder2::variable(p[a], a));
        let out = rigid_row_nll(&vars, &inputs)?;
        Ok((out.value(), out.g(), out.h()))
    }

    /// Batched all-rows `(nll, grad, hess)` via the A100 NVRTC survival row-jet
    /// (#932-GPU). Gathers every row's primaries + scalar inputs, then calls the
    /// device dispatcher ([`crate::gpu::kernels::survival_rowjet`]) which runs the
    /// SAME unified `rigid_row_nll` jet on device for all `n` rows in parallel and
    /// falls back to the CPU jet on no-GPU / small-`n` / any device error. The
    /// result is bit-close (≤1e-9; measured 4.7e-12) to the per-row `row_kernel`
    /// loop, so the cache is identical.
    ///
    /// Returns `None` (per-row CPU loop) when ANY row violates the monotonicity
    /// guard — that domain failure must surface as the `MonotonicityViolation`
    /// error the per-row path raises, not be masked by the device kernel (which
    /// does not re-derive the guard).
    fn batched_value_grad_hess_all(
        &self,
    ) -> Option<Result<(Vec<f64>, Vec<[f64; 4]>, Vec<[[f64; 4]; 4]>), String>> {
        use crate::gpu::kernels::survival_rowjet::{
            survival_rigid_row_jets, SurvivalRowInputs,
        };
        let n = self.family.n;
        let probit_scale = self.family.probit_frailty_scale();
        let qd1_lower = self.family.time_derivative_lower_bound();
        // Gather per-row inputs in parallel (the pure-f64 score summary + primary
        // projections — the same quantities the per-row path computes). Surface
        // any gather error; bail to the per-row path on a monotonicity violation.
        let gather: Result<Vec<SurvivalRowInputs>, String> = (0..n)
            .into_par_iter()
            .map(|row| {
                let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
                if survival_derivative_guard_violated(p[2], qd1_lower) {
                    return Err("monotonicity-violation-fallback".to_string());
                }
                let inputs = rigid_row_inputs(
                    &self.family,
                    &self.block_states,
                    row,
                    "survival marginal-slope rigid row kernel (batched)",
                )?;
                Ok(SurvivalRowInputs {
                    primaries: p,
                    wi: inputs.wi,
                    di: inputs.di,
                    z_sum: inputs.z_sum,
                    cov_ones: inputs.covariance_ones,
                })
            })
            .collect();
        let rows = match gather {
            Ok(rows) => rows,
            // Monotonicity violation or gather failure → defer to the per-row
            // path, which raises the precise error.
            Err(_) => return None,
        };
        // (v,g,H) only — the third/fourth direction args are unused for the
        // value cache; pass zeros.
        let zero = [0.0_f64; 4];
        let ch = survival_rigid_row_jets(&rows, probit_scale, &zero, &zero, &zero);
        let mut grads = vec![[0.0_f64; 4]; n];
        let mut hesss = vec![[[0.0_f64; 4]; 4]; n];
        for row in 0..n {
            for a in 0..4 {
                grads[row][a] = ch.grad[row * 4 + a];
                for b in 0..4 {
                    hesss[row][a][b] = ch.hess[row * 16 + a * 4 + b];
                }
            }
        }
        Some(Ok((ch.value, grads, hesss)))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        let n_rows = self.family.n;
        // Whole-projection build: each axis uses the batched design matvec
        // (`fast_ab` on dense, one operator `dot` per column on operator-backed
        // designs).
        Some(self.assemble_jf(factor, n_rows, |design, factor_block| {
            crate::families::row_kernel::row_kernel_design_jf(design, factor_block, n_rows)
        }))
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        if factor.nrows() != self.slices.total {
            // Shape contract broken (the tiled trace always passes the
            // coefficient-width factor, so this is defensive only): fall back
            // to the exact generic per-row build over the range.
            return crate::families::row_kernel::row_kernel_jacobian_action_matrix_generic_rows(
                self, factor, start, end,
            );
        }
        // Block-tiled build for one row-tile: dense designs slice to a
        // contiguous row block and GEMM (`fast_ab`), operator/sparse designs
        // fall to a row-local dot over the range. Bounds peak memory to the
        // tile while keeping BLAS-3 on the materialized designs.
        let b = end.saturating_sub(start);
        self.assemble_jf(factor, b, |design, factor_block| {
            crate::families::row_kernel::row_kernel_design_jf_rows(design, factor_block, start, end)
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        // Packed one-seed directional scalar: the ε-Hessian channel is exactly
        // `Σ_c ℓ_{abc} dir_c` without materialising the dense `t3`. Bit-identical
        // to `rigid_row_kernel_nll_tower(row)?.third_contracted(dir)`.
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row third",
        )?;
        let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
        let vars: [OneSeed<4>; 4] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir[a]));
        Ok(rigid_row_nll(&vars, &inputs)?.contracted_third())
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        // Packed two-seed scalar: the εδ-Hessian channel is exactly
        // `Σ_{cd} ℓ_{abcd} u_c v_d` without materialising the dense `t4`.
        // Bit-identical to `rigid_row_kernel_nll_tower(row)?.fourth_contracted(u, v)`.
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row fourth",
        )?;
        let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
        let vars: [TwoSeed<4>; 4] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, dir_u[a], dir_v[a]));
        Ok(rigid_row_nll(&vars, &inputs)?.contracted_fourth())
    }

    /// Batched all-axes FIRST directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979).
    ///
    /// The generic per-axis fall-back (`row_kernel_directional_derivative_all_axes`)
    /// asks for `Hdot[e_a]` `p` separate times, and EACH per-axis sweep rebuilds
    /// the per-row fourth-order `Tower4<4>` for every row inside
    /// `row_third_contracted` (`evaluate_program` → full t3/t4 build) — `n·p`
    /// tower evaluations per all-axes call. For survival the tower is the
    /// expensive object (closed-form probit/log-pdf composition over four
    /// primaries), so this is the #979 inner-Newton Jeffreys/Firth hot path.
    ///
    /// This override builds each row's `t3` tensor ONCE (the swept axis enters
    /// only through the cheap primary projection `dir_a = Jᵢ·e_a` and the linear
    /// `t3.third_contracted(dir_a)`), then closes every axis off that single
    /// build. Crucially it reuses the kernel's OWN `jacobian_action`,
    /// `Tower4::third_contracted`, and `add_pullback_hessian` in the EXACT SAME
    /// `ARROW_ROW_CHUNK`-chunked reduction order as the generic per-axis path
    /// (`par_try_reduce_fold(RowSet::All)`): the cached `t3[row]` is bit-for-bit
    /// the tensor a fresh `evaluate_program(row)` would produce (a deterministic
    /// pure function of the row), and every float op downstream is identical, so
    /// axis `a` matches `row_kernel_directional_derivative(self, All, e_a)`
    /// bit-for-bit. Only the redundant `(p−1)·n` tower rebuilds are removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise
    /// returns `None` so the generic per-axis Horvitz-Thompson sweep runs.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::families::row_kernel::RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope directional_derivative_all_axes_dense_override: \
                 axis count {p} disagrees with n_coefficients() {}",
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::families::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.directional_derivative_all_axes_build_once(p))
    }

    /// Batched all-axes SECOND directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979): the outer-REML
    /// Jeffreys `H_Φ` drift analogue of the first-order override above.
    ///
    /// With `d_beta_u` fixed and the second direction sweeping every canonical
    /// axis, the generic per-axis path runs `p` full-data sweeps each rebuilding
    /// the per-row `Tower4<4>` (`row_fourth_contracted` → `evaluate_program`).
    /// This override builds each row's `t4` tensor and the fixed-direction
    /// projection `dir_u = Jᵢ·u` ONCE, then closes every axis with the cheap
    /// linear `t4.fourth_contracted(dir_u, dir_a)` and the kernel's own
    /// `add_pullback_hessian`, in the SAME chunked reduction order as
    /// `row_kernel_second_directional_derivative(self, All, u, e_a)` — bit-for-bit
    /// identical, only the redundant tower rebuilds removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise `None`.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::families::row_kernel::RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if d_beta_u.len() != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope second_directional_derivative_all_axes_dense_override: \
                 fixed direction has {} entries, expected {}",
                d_beta_u.len(),
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::families::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.second_directional_derivative_all_axes_build_once(d_beta_u))
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Assemble the `(n_out × 4·rank)` joint Jacobian-action projection `Jᵢ · F`
    /// from the four primary axes — `[entry+marginal | exit+marginal |
    /// derivative | logslope]` — given a per-axis builder `axis(design,
    /// factor_block)` that produces that design's `n_out × rank` contribution.
    /// The whole-projection path passes the batched builder; the block-tiled
    /// path passes the row-range builder. Either way at most one axis transient
    /// is alive at a time: the marginal block feeds both the entry and exit
    /// axes, so it is built once and dropped, and every other axis is a
    /// statement-scoped temporary — keeping the assembly peak at
    /// `output + one n_out×rank block` rather than five blocks at once.
    pub(crate) fn assemble_jf<F>(
        &self,
        factor: ArrayView2<'_, f64>,
        n_out: usize,
        axis: F,
    ) -> Array2<f64>
    where
        F: Fn(&DesignMatrix, ArrayView2<'_, f64>) -> Array2<f64>,
    {
        let rank = factor.ncols();
        if rank == 0 {
            return Array2::<f64>::zeros((n_out, 0));
        }
        let f_time = factor.slice(s![self.slices.time.clone(), ..]);
        let f_marginal = factor.slice(s![self.slices.marginal.clone(), ..]);
        let f_logslope = factor.slice(s![self.slices.logslope.clone(), ..]);

        let jf_marginal = axis(&self.family.marginal_design, f_marginal);
        let mut axis0 = axis(&self.family.design_entry, f_time);
        axis0 += &jf_marginal;
        let mut axis1 = axis(&self.family.design_exit, f_time);
        axis1 += &jf_marginal;
        let axis2 = axis(&self.family.design_derivative_exit, f_time);
        let axis3 = axis(&self.family.logslope_design, f_logslope);

        crate::families::row_kernel::row_kernel_pack_jf_axes::<4>(
            n_out,
            rank,
            [(0, axis0), (1, axis1), (2, axis2), (3, axis3)],
        )
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Build every row's full fourth-order primary tower ONCE.
    ///
    /// `evaluate_program(self, row)` is a deterministic pure function of the
    /// row's primaries, so the cached tower's `t3`/`t4` channels are bit-for-bit
    /// what a fresh per-axis `row_third_contracted` / `row_fourth_contracted`
    /// rebuild would produce — the build-once batched overrides below contract
    /// against these cached towers without changing any downstream arithmetic.
    fn build_row_towers(&self) -> Result<Vec<crate::families::jet_tower::Tower4<4>>, String> {
        let n = <Self as RowKernel<4>>::n_rows(self);
        (0..n)
            .into_par_iter()
            .map(|row| crate::families::jet_tower::evaluate_program::<4, Self>(self, row))
            .collect()
    }

    /// Deterministic `ARROW_ROW_CHUNK`-chunked reduction matching
    /// `par_try_reduce_fold(RowSet::All)`: rows fold in index order inside each
    /// fixed 256-row chunk, chunks reduce in chunk-index order on the caller
    /// thread. `per_row(row, &mut acc)` accumulates one row's pullback into the
    /// `p×p` accumulator exactly as the generic per-axis fold does.
    fn chunked_pullback_reduce<F>(&self, p: usize, per_row: F) -> Result<Array2<f64>, String>
    where
        F: Fn(usize, &mut Array2<f64>) -> Result<(), String> + Sync,
    {
        let n = <Self as RowKernel<4>>::n_rows(self);
        let chunk = crate::outer_subsample::ARROW_ROW_CHUNK;
        let n_chunks = crate::outer_subsample::arrow_row_chunk_count(n);
        let chunk_accumulators: Vec<Result<Array2<f64>, String>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk;
                let end = (start + chunk).min(n);
                let mut acc = Array2::<f64>::zeros((p, p));
                for row in start..end {
                    per_row(row, &mut acc)?;
                }
                Ok(acc)
            })
            .collect();
        let mut total = Array2::<f64>::zeros((p, p));
        for acc in chunk_accumulators {
            total += &acc?;
        }
        Ok(total)
    }

    /// gam#979 build-once all-axes FIRST directional derivative — see the trait
    /// override docstring. Builds the per-row `t3` towers once, then for each
    /// canonical axis runs the identical chunked pullback reduction the generic
    /// per-axis sweep runs, reusing the cached tower instead of rebuilding it.
    fn directional_derivative_all_axes_build_once(
        &self,
        p: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let towers = self.build_row_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                crate::linalg::faer_ndarray::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir = self.jacobian_action(row, &axis);
                        let third = towers[row].third_contracted(&dir);
                        self.add_pullback_hessian(row, &third, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }

    /// gam#979 build-once all-axes SECOND directional derivative — see the trait
    /// override docstring. Builds the per-row `t4` towers and the fixed-direction
    /// projection once, then closes every axis from that single build in the
    /// generic per-axis sweep's reduction order.
    fn second_directional_derivative_all_axes_build_once(
        &self,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = self.n_coefficients();
        let towers = self.build_row_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                crate::linalg::faer_ndarray::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir_u = self.jacobian_action(row, d_beta_u);
                        let dir_v = self.jacobian_action(row, &axis);
                        let fourth = towers[row].fourth_contracted(&dir_u, &dir_v);
                        self.add_pullback_hessian(row, &fourth, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }
}
