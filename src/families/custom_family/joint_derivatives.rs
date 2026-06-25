//! Joint second-derivative correction, the `HessianDerivativeProvider`
//! implementations (borrowed / owned / Jeffreys-aware), scaled hyper-
//! operators, and the ext-coord bundle, split out of `outer_objective.rs`
//! by concern (#1145). Re-exported via `custom_family`.

use super::*;

/// Shared `(term1, term2)` second-derivative correction assembly used by both
/// the borrowed and owned joint derivative providers. `compute_dh` supplies the
/// drift derivative `D_β H[u_kl]` (term1) and `compute_d2h` the mixed second
/// derivative `D²_β H[−v_l, −v_k]` (term2); the two are fused into a single
/// `CompositeHyperOperator`. Returns `None` as soon as either term is absent.
pub(crate) fn joint_second_derivative_correction_result(
    compute_dh: &dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    compute_d2h: &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    v_k: &Array1<f64>,
    v_l: &Array1<f64>,
    u_kl: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, String> {
    let Some(term1) = compute_dh(u_kl)? else {
        return Ok(None);
    };
    let neg_v_k = -v_k;
    let neg_v_l = -v_l;
    let Some(term2) = compute_d2h(&neg_v_l, &neg_v_k)? else {
        return Ok(None);
    };
    let op = CompositeHyperOperator {
        dense: None,
        operators: vec![term1.into_operator(), term2.into_operator()],
        dim_hint: u_kl.len(),
    };
    Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
}

impl HessianDerivativeProvider for BorrowedJointDerivProvider<'_> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_derivative_correction_result(v_k)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v = -v_k;
        (self.compute_dh)(&neg_v)
    }

    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        let neg_vs: Vec<Array1<f64>> = v_ks.iter().map(|v_k| -v_k).collect();
        if let Some(compute_dh_many) = self.compute_dh_many {
            compute_dh_many(&neg_vs)
        } else {
            neg_vs
                .iter()
                .map(|neg_v| (self.compute_dh)(neg_v))
                .collect()
        }
    }

    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.compute_dh_many.is_some()
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        joint_second_derivative_correction_result(self.compute_dh, self.compute_d2h, v_k, v_l, u_kl)
    }

    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        // Fast path: family supplied a batched D²H callback that fuses the
        // per-row scan across all K(K+1)/2 (v_k, v_l, u_kl) triples in one
        // pass. Pair it with the (also potentially batched) `compute_dh`
        // term1 walk over `u_kl` directions to keep the (term1, term2)
        // CompositeHyperOperator semantics that the singular hook produces.
        if let Some(compute_d2h_many) = self.compute_d2h_many {
            let u_kls: Vec<Array1<f64>> = triples.iter().map(|(_, _, u_kl)| u_kl.clone()).collect();
            let term1s = self.hessian_derivative_corrections_result(
                &u_kls.iter().map(|u| -u).collect::<Vec<_>>(),
            )?;
            let pairs: Vec<(Array1<f64>, Array1<f64>)> =
                triples.iter().map(|(v_k, v_l, _)| (-v_l, -v_k)).collect();
            let term2s = compute_d2h_many(&pairs)?;
            triples
                .iter()
                .enumerate()
                .map(|(idx, (_, _, u_kl))| match (&term1s[idx], &term2s[idx]) {
                    (Some(t1), Some(t2)) => {
                        let op = CompositeHyperOperator {
                            dense: None,
                            operators: vec![t1.clone().into_operator(), t2.clone().into_operator()],
                            dim_hint: u_kl.len(),
                        };
                        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
                    }
                    _ => Ok(None),
                })
                .collect()
        } else {
            triples
                .iter()
                .map(|(v_k, v_l, u_kl)| {
                    self.hessian_second_derivative_correction_result(v_k, v_l, u_kl)
                })
                .collect()
        }
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.compute_d2h_many.is_some()
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::OuterHessianOperator>> {
        self.family_outer_hessian_operator.clone()
    }
}

pub(crate) struct OwnedJointDerivProvider {
    pub(crate) compute_dh:
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    pub(crate) compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
    >,
    pub(crate) compute_d2h: Arc<
        dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
            + Send
            + Sync,
    >,
    /// Optional batched second-derivative callback. See the matching field on
    /// `BorrowedJointDerivProvider` for the dispatch contract.
    pub(crate) compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
                + Send
                + Sync,
        >,
    >,
    pub(crate) family_outer_hessian_operator: Option<Arc<dyn gam_problem::OuterHessianOperator>>,
}

impl HessianDerivativeProvider for OwnedJointDerivProvider {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_derivative_correction_result(v_k)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v = -v_k;
        (self.compute_dh)(&neg_v)
    }

    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        let neg_vs: Vec<Array1<f64>> = v_ks.iter().map(|v_k| -v_k).collect();
        if let Some(compute_dh_many) = self.compute_dh_many.as_ref() {
            compute_dh_many(&neg_vs)
        } else {
            neg_vs
                .iter()
                .map(|neg_v| (self.compute_dh)(neg_v))
                .collect()
        }
    }

    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.compute_dh_many.is_some()
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        joint_second_derivative_correction_result(
            &*self.compute_dh,
            &*self.compute_d2h,
            v_k,
            v_l,
            u_kl,
        )
    }

    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        if let Some(compute_d2h_many) = self.compute_d2h_many.as_ref() {
            let u_kls: Vec<Array1<f64>> = triples.iter().map(|(_, _, u_kl)| u_kl.clone()).collect();
            let term1s = self.hessian_derivative_corrections_result(
                &u_kls.iter().map(|u| -u).collect::<Vec<_>>(),
            )?;
            let pairs: Vec<(Array1<f64>, Array1<f64>)> =
                triples.iter().map(|(v_k, v_l, _)| (-v_l, -v_k)).collect();
            let term2s = compute_d2h_many(&pairs)?;
            triples
                .iter()
                .enumerate()
                .map(|(idx, (_, _, u_kl))| match (&term1s[idx], &term2s[idx]) {
                    (Some(t1), Some(t2)) => {
                        let op = CompositeHyperOperator {
                            dense: None,
                            operators: vec![t1.clone().into_operator(), t2.clone().into_operator()],
                            dim_hint: u_kl.len(),
                        };
                        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
                    }
                    _ => Ok(None),
                })
                .collect()
        } else {
            triples
                .iter()
                .map(|(v_k, v_l, u_kl)| {
                    self.hessian_second_derivative_correction_result(v_k, v_l, u_kl)
                })
                .collect()
        }
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.compute_d2h_many.is_some()
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        Some(OuterHessianDerivativeKernel::Callback {
            first: Arc::clone(&self.compute_dh),
            second: Arc::clone(&self.compute_d2h),
        })
    }

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::OuterHessianOperator>> {
        self.family_outer_hessian_operator.clone()
    }
}

/// BATCHED Jeffreys-`H_Φ` mode-response drift over MANY directions at once.
///
/// PERF (the biobank #979 outer-gradient black hole). The β-fixed base of the
/// drift — the reduced-information eigendecomposition AND the `p` per-axis first
/// directional derivatives `Hdot[e_a]` (each an `O(n)` n≈348k row-stream) — is
/// IDENTICAL across every mode-response direction `δβ = −v_k` at fixed `β̂(ρ)`.
/// A per-direction drift that rebuilt the base on every call would re-stream the
/// whole dataset `k·p` extra times per outer gradient eval. This batched form
/// prepares the base ONCE (via [`JeffreysHphiDriftBase`]) and then applies it to
/// every direction, so the only per-direction cost is that direction's own
/// `Hdot[δ]` and `p` second-directional `H²dot[δ,e_a]` passes. The per-direction
/// result is byte-identical to the divided-difference drift it amortizes.
///
/// The closure expects the actual perturbation directions `δβ` (NOT the raw `v_k`
/// the trait hands the provider); the [`JeffreysHphiAwareJointDerivatives`]
/// wrapper negates `v_k → δβ = −v_k` before calling. A `None` entry (gated-out
/// term / missing exact derivative on some axis) leaves the inner likelihood
/// drift unchanged for that direction.
pub(crate) type JeffreysHphiDriftBatchFn =
    Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<Array2<f64>>>, String> + Send + Sync>;

/// Jeffreys-`H_Φ`-aware joint derivative provider.
///
/// Wraps an inner Tier-B joint provider (which supplies the likelihood-Hessian
/// drift `D_β H_L[v_k]`) and ADDS the Jeffreys-curvature drift `D_β H_Φ[v_k]` to
/// the first-order trace corrections. This closes the bug where the Tier-B outer
/// LAML gradient omitted `H_Φ`'s ρ-dependence (through β̂): the objective folds
/// `H_Φ` into `½ log|H + S_λ + H_Φ|`, so its exact gradient
///   `½ tr[(H+S_λ+H_Φ)⁻¹ (∂_ρ S_λ + D_β H_L[v_k] + D_β H_Φ[v_k])]`
/// MUST include the `D_β H_Φ[v_k]` term. It is the exact analogue of the Tier-A
/// `FirthAwareGlmDerivatives` (`unified.rs`) `−D(Hφ)[B_k]` first-order term, and
/// of `BarrierDerivativeProvider`'s additive-correction composition pattern.
///
/// SIGN. The trait passes `v_k = H⁻¹(A_kβ̂)`; the mode response is `δβ = −v_k`.
/// We negate before invoking the drift closure, so `corr = + D_β H_Φ[δβ]` is
/// added on top of the inner provider's already-correct likelihood drift.
pub(crate) struct JeffreysHphiAwareJointDerivatives<'a> {
    pub(crate) inner: Box<dyn HessianDerivativeProvider + 'a>,
    pub(crate) drift: JeffreysHphiDriftBatchFn,
    pub(crate) p: usize,
}

impl<'a> JeffreysHphiAwareJointDerivatives<'a> {
    pub(crate) fn new(
        inner: Box<dyn HessianDerivativeProvider + 'a>,
        drift: JeffreysHphiDriftBatchFn,
        p: usize,
    ) -> Self {
        Self { inner, drift, p }
    }

    /// `D_β H_Φ[δβ]` for MANY mode-response directions at once, with the trait's
    /// `v_k → δβ = −v_k` convention. The batched drift prepares the β-fixed base
    /// (reduced-information eigendecomposition + the `p` per-axis first directional
    /// derivatives `Hdot[e_a]`, each an `O(n)` row-stream) ONCE and reuses it for
    /// every direction — collapsing the released `k·p` redundant full-data passes
    /// (the biobank #979 outer-gradient black hole) to a single `p`-axis sweep plus
    /// the genuinely per-direction `Hdot[δ]` / `H²dot[δ,e_a]` work. Per-direction
    /// output is byte-identical to the singular hook.
    pub(crate) fn hphi_drifts(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<Array2<f64>>>, String> {
        let deltas: Vec<Array1<f64>> = v_ks.iter().map(|v| v.mapv(|value| -value)).collect();
        (self.drift)(&deltas)
    }

    /// `D_β H_Φ[δβ]` for a SINGLE mode-response direction. Routes through the
    /// batched closure with a one-element slice so the singular trait methods reuse
    /// the identical arithmetic; the dominant outer-gradient path goes through
    /// [`Self::hphi_drifts`] where the base is amortized across all `k` directions.
    pub(crate) fn hphi_drift(&self, v_k: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        let delta = v_k.mapv(|value| -value);
        let mut out = (self.drift)(std::slice::from_ref(&delta))?;
        Ok(out.pop().flatten())
    }
}

impl HessianDerivativeProvider for JeffreysHphiAwareJointDerivatives<'_> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let inner = self.inner.hessian_derivative_correction(v_k)?;
        let drift = self.hphi_drift(v_k)?;
        Ok(match (inner, drift) {
            (Some(mut ic), Some(d)) => {
                ic += &d;
                Some(ic)
            }
            (Some(ic), None) => Some(ic),
            (None, Some(d)) => Some(d),
            (None, None) => None,
        })
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let inner = self.inner.hessian_derivative_correction_result(v_k)?;
        let drift = self.hphi_drift(v_k)?;
        Ok(match (inner, drift) {
            (Some(DriftDerivResult::Dense(mut dense)), Some(d)) => {
                dense += &d;
                Some(DriftDerivResult::Dense(dense))
            }
            (Some(DriftDerivResult::Operator(operator)), Some(d)) => Some(
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: Some(d),
                    operators: vec![operator],
                    dim_hint: self.p,
                })),
            ),
            (Some(other), None) => Some(other),
            (None, Some(d)) => Some(DriftDerivResult::Dense(d)),
            (None, None) => None,
        })
    }

    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        // Delegate the (possibly batched) inner walk, then fold the per-direction
        // H_Φ drift into each result so the batched path stays consistent with the
        // singular one. The H_Φ drift is computed for ALL `k` directions in ONE
        // batched call so the β-fixed base (reduced eigendecomposition + the `p`
        // per-axis first directional derivatives, each an `O(n)` row-stream) is
        // prepared ONCE rather than recomputed `k` times — the biobank #979
        // outer-gradient black hole. Per-direction values are byte-identical.
        let inner = self.inner.hessian_derivative_corrections_result(v_ks)?;
        let drifts = self.hphi_drifts(v_ks)?;
        if drifts.len() != inner.len() {
            return Err(format!(
                "JeffreysHphiAwareJointDerivatives: batched H_Φ drift returned {} results for {} directions",
                drifts.len(),
                inner.len()
            ));
        }
        inner
            .into_iter()
            .zip(drifts.into_iter())
            .map(|(inner_result, drift)| {
                Ok(match (inner_result, drift) {
                    (Some(DriftDerivResult::Dense(mut dense)), Some(d)) => {
                        dense += &d;
                        Some(DriftDerivResult::Dense(dense))
                    }
                    (Some(DriftDerivResult::Operator(operator)), Some(d)) => Some(
                        DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                            dense: Some(d),
                            operators: vec![operator],
                            dim_hint: self.p,
                        })),
                    ),
                    (Some(other), None) => Some(other),
                    (None, Some(d)) => Some(DriftDerivResult::Dense(d)),
                    (None, None) => None,
                })
            })
            .collect()
    }

    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.inner.has_batched_hessian_derivative_corrections()
    }

    // SECOND-ORDER (outer Hessian) RESIDUAL GAP. The full second-order Jeffreys
    // drift `D²_β H_Φ[v_k, v_l]` (the analogue of Tier-A's
    // `−D(Hφ)[B_{kl}] − D²(Hφ)[B_k, B_l]`) is NOT yet folded in here: the
    // second-derivative methods delegate to the inner likelihood drift only. This
    // leaves the OUTER HESSIAN's Jeffreys contribution first-order-incomplete, but
    // the FIRST-ORDER outer GRADIENT — the term the line search and KKT
    // certification actually consume — is now exact. ARC/Newton on the outer
    // problem still gets a consistent gradient; the Hessian is a (PD) curvature
    // surrogate as before.
    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.inner
            .hessian_second_derivative_correction(v_k, v_l, u_kl)
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.inner
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)
    }

    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.inner
            .hessian_second_derivative_corrections_result(triples)
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.inner
            .has_batched_hessian_second_derivative_corrections()
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        // Delegate to the inner provider so the matrix-free outer-HESSIAN route
        // (the `Callback { first, second }` kernel) is preserved. This kernel
        // feeds ONLY the outer Hessian, never the gradient (the gradient's
        // first-order trace flows through `hessian_derivative_correction_result`,
        // which IS wrapped above). The H_Φ SECOND-order drift is the documented
        // residual gap; routing the kernel unchanged keeps the Hessian a
        // consistent PD curvature surrogate without forcing dense assembly.
        self.inner.outer_hessian_derivative_kernel()
    }

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::OuterHessianOperator>> {
        self.inner.family_outer_hessian_operator()
    }
}

/// Optional bundle of extended (ψ) hyperparameter coordinate data to attach
/// to an `InnerSolution` before calling the unified evaluator.
pub(crate) struct ExtCoordBundle {
    pub(crate) coords: Vec<HyperCoord>,
    pub(crate) ext_ext_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    pub(crate) rho_ext_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    pub(crate) drift_fn: Option<FixedDriftDerivFn>,
    /// Direction-contracted ψψ second-order hook (#740). When `Some`, the
    /// outer-Hessian operator builder skips the `K²` per-pair ψψ assembly
    /// (`ext_ext_fn`) and applies this once per matvec. `ext_ext_fn` is still
    /// kept as the documented fallback for the dense `compute_outer_hessian`
    /// path and for outer evaluations that do not build the matrix-free
    /// operator.
    pub(crate) contracted_psi_fn: Option<ContractedPsiSecondOrderFn>,
}

pub(crate) struct ScaledHyperOperator {
    pub(crate) inner: Arc<dyn HyperOperator>,
    pub(crate) scale: f64,
}

impl HyperOperator for ScaledHyperOperator {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.inner.mul_vec(v).mapv(|value| self.scale * value)
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.scale * self.inner.bilinear(v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.inner.to_dense().mapv(|value| self.scale * value)
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

pub(crate) fn scale_hypercoord_drift(mut drift: HyperCoordDrift, scale: f64) -> HyperCoordDrift {
    if scale == 1.0 {
        return drift;
    }
    if let Some(ref mut dense) = drift.dense {
        *dense *= scale;
    }
    if let Some(ref mut block_local) = drift.block_local {
        block_local.local *= scale;
    }
    if let Some(operator) = drift.operator.take() {
        drift.operator = Some(Arc::new(ScaledHyperOperator {
            inner: operator,
            scale,
        }));
    }
    drift
}

pub(crate) fn scale_hypercoord(mut coord: HyperCoord, scale: f64) -> HyperCoord {
    if scale == 1.0 {
        return coord;
    }
    coord.g *= scale;
    if let Some(firth_g) = coord.firth_g.as_mut() {
        *firth_g *= scale;
    }
    if let Some(tk_eta_fixed) = coord.tk_eta_fixed.as_mut() {
        *tk_eta_fixed *= scale;
    }
    if let Some(tk_x_fixed) = coord.tk_x_fixed.as_mut() {
        *tk_x_fixed *= scale;
    }
    coord.drift = scale_hypercoord_drift(coord.drift, scale);
    coord
}

pub(crate) fn scale_hypercoord_pair(mut pair: HyperCoordPair, scale: f64) -> HyperCoordPair {
    if scale == 1.0 {
        return pair;
    }
    pair.g *= scale;
    pair.b_mat *= scale;
    if let Some(operator) = pair.b_operator.take() {
        pair.b_operator = Some(Box::new(ScaledHyperOperator {
            inner: Arc::from(operator),
            scale,
        }));
    }
    pair
}

pub(crate) fn scale_drift_deriv_result(result: DriftDerivResult, scale: f64) -> DriftDerivResult {
    if scale == 1.0 {
        return result;
    }
    match result {
        DriftDerivResult::Dense(mut dense) => {
            dense *= scale;
            DriftDerivResult::Dense(dense)
        }
        DriftDerivResult::Operator(operator) => {
            DriftDerivResult::Operator(Arc::new(ScaledHyperOperator {
                inner: operator,
                scale,
            }))
        }
    }
}

impl ExtCoordBundle {
    pub(crate) fn scaled(self, scale: f64) -> Self {
        if scale == 1.0 {
            return self;
        }
        let coords = self
            .coords
            .into_iter()
            .map(|coord| scale_hypercoord(coord, scale))
            .collect();
        let ext_ext_fn = self.ext_ext_fn.map(|callback| {
            Box::new(move |i: usize, j: usize| scale_hypercoord_pair(callback(i, j), scale))
                as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
        });
        let rho_ext_fn = self.rho_ext_fn.map(|callback| {
            Box::new(move |i: usize, j: usize| scale_hypercoord_pair(callback(i, j), scale))
                as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
        });
        let drift_fn = self.drift_fn.map(|callback| {
            Box::new(move |ext_idx: usize, direction: &Array1<f64>| {
                callback(ext_idx, direction).map(|result| scale_drift_deriv_result(result, scale))
            }) as FixedDriftDerivFn
        });
        // The contracted ψψ hook is a (scaled) linear functional of the same
        // family curvature `ext_ext_fn` reproduces, so the `rho_curvature_scale`
        // applies term-for-term: objective/score/ld_s by `scale`, and each
        // `hessian[i]` drift via `scale_drift_deriv_result` (matching how
        // `scale_hypercoord_pair` scales the per-pair `b_mat`/`b_operator`).
        let contracted_psi_fn = self.contracted_psi_fn.map(|callback| {
            Arc::new(move |alpha_psi: &[f64]| {
                callback(alpha_psi).map(|opt| {
                    opt.map(|contracted| ContractedPsiSecondOrder {
                        objective: contracted.objective.mapv(|v| scale * v),
                        score: contracted.score.mapv(|v| scale * v),
                        hessian: contracted
                            .hessian
                            .into_iter()
                            .map(|drift| scale_drift_deriv_result(drift, scale))
                            .collect(),
                        ld_s: contracted.ld_s.mapv(|v| scale * v),
                    })
                })
            }) as ContractedPsiSecondOrderFn
        });
        Self {
            coords,
            ext_ext_fn,
            rho_ext_fn,
            drift_fn,
            contracted_psi_fn,
        }
    }
}
