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
    compute_dh: &dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError>,
    compute_d2h: &dyn Fn(
        &Array1<f64>,
        &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, CustomFamilyError>,
    v_k: &Array1<f64>,
    v_l: &Array1<f64>,
    u_kl: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, CustomFamilyError> {
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

/// Fold an optional dense Jeffreys drift into an optional inner drift result,
/// preserving the inner result's shape (dense stays dense, operator gains a
/// dense companion instead of being materialized).
fn compose_drift(
    inner: Option<DriftDerivResult>,
    drift: Option<Array2<f64>>,
    dim_hint: usize,
) -> Option<DriftDerivResult> {
    match (inner, drift) {
        (Some(DriftDerivResult::Dense(mut dense)), Some(d)) => {
            dense += &d;
            Some(DriftDerivResult::Dense(dense))
        }
        (Some(DriftDerivResult::Operator(operator)), Some(d)) => Some(DriftDerivResult::Operator(
            Arc::new(CompositeHyperOperator {
                dense: Some(d),
                operators: vec![operator],
                dim_hint,
            }),
        )),
        (Some(other), None) => Some(other),
        (None, Some(d)) => Some(DriftDerivResult::Dense(d)),
        (None, None) => None,
    }
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
        // Display boundary: `HessianDerivativeProvider` (gam-solve) is a
        // `String`-erroring trait, so the typed error is rendered HERE and
        // visibly, rather than by a silent blanket `From` (gam#2689).
        (self.compute_dh)(&neg_v).map_err(|error| error.to_string())
    }

    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        let neg_vs: Vec<Array1<f64>> = v_ks.iter().map(|v_k| -v_k).collect();
        if let Some(compute_dh_many) = self.compute_dh_many {
            compute_dh_many(&neg_vs).map_err(|error| error.to_string())
        } else {
            neg_vs
                .iter()
                .map(|neg_v| (self.compute_dh)(neg_v).map_err(|error| error.to_string()))
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
            .map_err(|error| error.to_string())
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
            let term2s = compute_d2h_many(&pairs).map_err(|error| error.to_string())?;
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

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        self.family_outer_hessian_operator.clone()
    }
}

pub(crate) struct OwnedJointDerivProvider {
    pub(crate) compute_dh: Arc<
        dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError> + Send + Sync,
    >,
    pub(crate) compute_dh_many: Option<
        Arc<
            dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    pub(crate) compute_d2h: Arc<
        dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError>
            + Send
            + Sync,
    >,
    /// Optional batched second-derivative callback. See the matching field on
    /// `BorrowedJointDerivProvider` for the dispatch contract.
    pub(crate) compute_d2h_many: Option<
        Arc<
            dyn Fn(
                    &[(Array1<f64>, Array1<f64>)],
                ) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    pub(crate) family_outer_hessian_operator: Option<Arc<dyn gam_problem::HessianOperator>>,
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
        // Display boundary: `HessianDerivativeProvider` (gam-solve) is a
        // `String`-erroring trait, so the typed error is rendered HERE and
        // visibly, rather than by a silent blanket `From` (gam#2689).
        (self.compute_dh)(&neg_v).map_err(|error| error.to_string())
    }

    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        let neg_vs: Vec<Array1<f64>> = v_ks.iter().map(|v_k| -v_k).collect();
        if let Some(compute_dh_many) = self.compute_dh_many.as_ref() {
            compute_dh_many(&neg_vs).map_err(|error| error.to_string())
        } else {
            neg_vs
                .iter()
                .map(|neg_v| (self.compute_dh)(neg_v).map_err(|error| error.to_string()))
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
        .map_err(|error| error.to_string())
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
            let term2s = compute_d2h_many(&pairs).map_err(|error| error.to_string())?;
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
        // Display boundary: `OuterHessianDerivativeKernel::Callback` (gam-solve)
        // is declared over `Result<_, String>`, so the adapters render the typed
        // error explicitly instead of leaning on a blanket `From` (gam#2689).
        let first = Arc::clone(&self.compute_dh);
        let second = Arc::clone(&self.compute_d2h);
        Some(OuterHessianDerivativeKernel::Callback {
            first: Arc::new(move |v: &Array1<f64>| first(v).map_err(|error| error.to_string())),
            second: Arc::new(move |v: &Array1<f64>, u: &Array1<f64>| {
                second(v, u).map_err(|error| error.to_string())
            }),
        })
    }

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::HessianOperator>> {
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
/// wrapper negates `v_k → δβ = −v_k` before calling. A `None` entry denotes an
/// inactive term. Missing derivatives of active curvature must return an error.
#[derive(Clone)]
pub(crate) struct JeffreysHphiDriftBatchFn {
    pub(crate) completion_beta: Arc<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>, CustomFamilyError> + Send + Sync>,
    pub(crate) completion_psi: Option<CompletionPsiAction>,
    pub(crate) response_scale: f64,
    pub(crate) first: Arc<
        dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<Array2<f64>>>, CustomFamilyError> + Send + Sync,
    >,
    pub(crate) second: Arc<
        dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Array2<f64>>, CustomFamilyError>
            + Send
            + Sync,
    >,
}

impl std::ops::Deref for JeffreysHphiDriftBatchFn {
    type Target =
        dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<Array2<f64>>>, CustomFamilyError> + Send + Sync;

    fn deref(&self) -> &Self::Target {
        self.first.as_ref()
    }
}

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
    ) -> Result<Vec<Option<Array2<f64>>>, CustomFamilyError> {
        let deltas: Vec<Array1<f64>> = v_ks.iter().map(|v| v.mapv(|value| -value)).collect();
        (self.drift)(&deltas)
    }

    /// `D_β H_Φ[δβ]` for a SINGLE mode-response direction. Routes through the
    /// batched closure with a one-element slice so the singular trait methods reuse
    /// the identical arithmetic; the dominant outer-gradient path goes through
    /// [`Self::hphi_drifts`] where the base is amortized across all `k` directions.
    pub(crate) fn hphi_drift(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, CustomFamilyError> {
        let delta = v_k.mapv(|value| -value);
        self.hphi_drift_along(&delta)
    }

    /// `D_β H_Φ[δ]` along a direction the caller has ALREADY put in `δβ`
    /// convention, with no sign flip applied here.
    ///
    /// The second-order correction needs this: `joint_second_derivative_correction_result`
    /// consumes the second mode response `u_kl` UNNEGATED (only the first-order
    /// `v_k`, `v_l` are flipped), so routing `u_kl` through [`Self::hphi_drift`]
    /// would silently price `−u_kl`.
    pub(crate) fn hphi_drift_along(
        &self,
        delta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, CustomFamilyError> {
        let mut out = (self.drift)(std::slice::from_ref(delta))?;
        Ok(out.pop().flatten())
    }
}

impl HessianDerivativeProvider for JeffreysHphiAwareJointDerivatives<'_> {
    fn mode_response_rhs_correction(&self) -> Option<gam_solve::estimate::reml::reml_outer_engine::ModeResponseRhsCorrectionFn> {
        let beta = Arc::clone(&self.drift.completion_beta);
        let psi = self.drift.completion_psi.clone();
        let scale = self.drift.response_scale;
        Some(Arc::new(move |i, j, vi, vj| {
            let mut result = beta(&(-vj), vi).map_err(|e| e.to_string())?;
            if let Some(psi) = &psi {
                if let Some(j) = j {
                    result += &psi(j, vi).map_err(|e| e.to_string())?;
                }
                if let Some(i) = i {
                    result += &psi(i, vj).map_err(|e| e.to_string())?;
                }
            }
            Ok(result * scale)
        }))
    }

    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let inner = self.inner.hessian_derivative_correction(v_k)?;
        // Display boundary: `HessianDerivativeProvider` is `String`-erroring (gam#2689).
        let drift = self.hphi_drift(v_k).map_err(|error| error.to_string())?;
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
        // Display boundary: `HessianDerivativeProvider` is `String`-erroring (gam#2689).
        let drift = self.hphi_drift(v_k).map_err(|error| error.to_string())?;
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
        let drifts = self.hphi_drifts(v_ks).map_err(|error| error.to_string())?;
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

    // SECOND-ORDER (outer Hessian) JEFFREYS DRIFT (#2612).
    //
    // The inner provider's second-order correction is
    //   `D_β H[u_kl] + D²_β H[−v_l, −v_k]`,
    // and `H` here is the criterion's `H + S_λ + H_Φ`, so BOTH terms owe a
    // Jeffreys contribution. The first — `D_β H_Φ[u_kl]`, the drift along the
    // SECOND mode response — is exactly the object the first-order wrapper
    // already builds, evaluated along a different direction, so it is folded in
    // below and costs one more direction through the same amortized base.
    //
    // The second, `D²_β H_Φ[−v_l, −v_k]`, consumes THIRD information
    // derivatives through the family's explicit fifth-likelihood contract.
    // Both terms must be present before the provider can claim exact curvature.
    //
    // SIGN. `u_kl` arrives in `δβ` convention already (the inner provider passes
    // it to `compute_dh` unnegated, unlike `v_k`/`v_l`), so it goes through
    // `hphi_drift_along`, not `hphi_drift`.
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
        let inner = self
            .inner
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?;
        // Display boundary: `HessianDerivativeProvider` is `String`-erroring (gam#2689).
        let mut drift = self
            .hphi_drift_along(u_kl)
            .map_err(|error| error.to_string())?;
        // D²Hφ[-v_k,-v_l] has two minus signs, so the bilinear callback can
        // consume the original response vectors directly.
        let mut mixed = (self.drift.second)(&[(v_k.clone(), v_l.clone())])
            .map_err(|error| error.to_string())?;
        if mixed.len() != 1 {
            return Err("Jeffreys mixed drift did not return exactly one matrix".to_string());
        }
        let mixed = mixed.pop().expect("checked single mixed drift");
        match &mut drift {
            Some(matrix) => *matrix += &mixed,
            None => drift = Some(mixed),
        }
        Ok(compose_drift(inner, drift, self.p))
    }

    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        let inner = self
            .inner
            .hessian_second_derivative_corrections_result(triples)?;
        // ONE batched call over every triple's `u_kl`, so the β-fixed base (the
        // reduced eigendecomposition and the `p` per-axis `Hdot[e_a]` row-streams)
        // is prepared once for the whole outer-Hessian assembly rather than once
        // per pair — the same amortization the first-order path relies on.
        let deltas: Vec<Array1<f64>> = triples.iter().map(|(_, _, u_kl)| u_kl.clone()).collect();
        let mut drifts = (self.drift)(&deltas).map_err(|error| error.to_string())?;
        let pairs: Vec<_> = triples
            .iter()
            .map(|(u, v, _)| (u.clone(), v.clone()))
            .collect();
        let mixed = (self.drift.second)(&pairs).map_err(|error| error.to_string())?;
        if mixed.len() != drifts.len() {
            return Err("Jeffreys mixed drift batch length mismatch".to_string());
        }
        for (drift, mixed) in drifts.iter_mut().zip(mixed) {
            match drift {
                Some(matrix) => *matrix += &mixed,
                None => *drift = Some(mixed),
            }
        }
        if drifts.len() != inner.len() {
            return Err(format!(
                "JeffreysHphiAwareJointDerivatives: batched second-order H_Φ drift returned {} \
                 results for {} triples",
                drifts.len(),
                inner.len()
            ));
        }
        Ok(inner
            .into_iter()
            .zip(drifts)
            .map(|(inner_result, drift)| compose_drift(inner_result, drift, self.p))
            .collect())
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        // The batched arm above is always available: it drives the inner
        // provider's own (batched or per-triple) walk and folds one batched H_Φ
        // sweep on top, so it is correct whatever the inner provider supports.
        true
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        let OuterHessianDerivativeKernel::Callback { first, second } =
            self.inner.outer_hessian_derivative_kernel()?
        else {
            return None;
        };
        let drift_first = self.drift.clone();
        let drift_second = self.drift.clone();
        let p = self.p;
        Some(OuterHessianDerivativeKernel::Callback {
            first: Arc::new(move |direction| {
                let inner = first(direction)?;
                let mut drift = (drift_first)(std::slice::from_ref(direction))
                    .map_err(|error| error.to_string())?;
                if drift.len() != 1 {
                    return Err("Jeffreys first callback batch length mismatch".to_string());
                }
                Ok(compose_drift(inner, drift.pop().flatten(), p))
            }),
            second: Arc::new(move |u, v| {
                let inner = second(u, v)?;
                let mut drift = (drift_second.second)(&[(u.clone(), v.clone())])
                    .map_err(|error| error.to_string())?;
                if drift.len() != 1 {
                    return Err("Jeffreys second callback batch length mismatch".to_string());
                }
                Ok(compose_drift(inner, drift.pop(), p))
            }),
        })
    }

    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        // The family operator knows only its own curvature. The unified
        // callback above composes the active Jeffreys term into the exact Hv.
        None
    }
}

/// Optional bundle of extended (ψ) hyperparameter coordinate data to attach
/// to an `InnerSolution` before calling the unified evaluator.
pub(crate) type CompletionPsiAction = Arc<dyn Fn(usize, &Array1<f64>) -> Result<Array1<f64>, CustomFamilyError> + Send + Sync>;

pub(crate) struct ExtCoordBundle {
    pub(crate) completion_psi: Option<CompletionPsiAction>,
    pub(crate) coords: Vec<HyperCoord>,
    pub(crate) ext_ext_fn: Option<
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>,
    >,
    pub(crate) rho_ext_fn: Option<
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>,
    >,
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
        pair.b_operator = Some(Arc::new(ScaledHyperOperator {
            inner: operator,
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
            Box::new(move |i: usize, j: usize| {
                callback(i, j).map(|pair| scale_hypercoord_pair(pair, scale))
            })
                as Box<
                    dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync,
                >
        });
        let rho_ext_fn = self.rho_ext_fn.map(|callback| {
            Box::new(move |i: usize, j: usize| {
                callback(i, j).map(|pair| scale_hypercoord_pair(pair, scale))
            })
                as Box<
                    dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync,
                >
        });
        let drift_fn = self.drift_fn.map(|callback| {
            Box::new(move |ext_idx: usize, direction: &Array1<f64>| {
                callback(ext_idx, direction)
                    .map(|result| result.map(|result| scale_drift_deriv_result(result, scale)))
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
            completion_psi: self.completion_psi.map(|callback| Arc::new(move |i, v: &Array1<f64>| callback(i, v).map(|value| value * scale)) as CompletionPsiAction),
            coords,
            ext_ext_fn,
            rho_ext_fn,
            drift_fn,
            contracted_psi_fn,
        }
    }
}

#[cfg(test)]
mod jeffreys_drift_composition_tests {
    //! #2612: the Jeffreys wrapper must fold `H_Φ`'s drift into BOTH orders of
    //! the trace correction, and along the direction each order is actually
    //! evaluated at.
    //!
    //! The sign convention is the trap. `HessianDerivativeProvider` hands the
    //! first-order hook the raw mode response `v_k`, and the perturbation
    //! direction is `δβ = −v_k`; but the second-order hook's third argument
    //! `u_kl` is ALREADY the perturbation (the inner provider passes it to
    //! `compute_dh` unnegated, while it negates `v_k`/`v_l`). Routing `u_kl`
    //! through the first-order helper would silently price `−u_kl` — a term of
    //! the right magnitude and the wrong sign, which is worse than the omission
    //! it replaced.

    use super::*;
    use ndarray::array;

    /// An inner provider that contributes nothing, so the composed result IS the
    /// Jeffreys drift and the test reads it directly rather than by subtraction.
    struct SilentInner;

    impl HessianDerivativeProvider for SilentInner {
        fn hessian_derivative_correction(
            &self,
            mode_response: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(mode_response.iter().all(|v| v.is_finite()));
            Ok(None)
        }

        fn hessian_second_derivative_correction(
            &self,
            v_k: &Array1<f64>,
            v_l: &Array1<f64>,
            u_kl: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(v_k.len() == v_l.len() && v_l.len() == u_kl.len());
            Ok(None)
        }

        fn has_corrections(&self) -> bool {
            true
        }
    }

    /// `D_β H_Φ[δ] = diag(δ)`: linear, so the direction it was evaluated at is
    /// readable straight off the returned matrix.
    fn identity_drift() -> JeffreysHphiDriftBatchFn {
        JeffreysHphiDriftBatchFn {
            completion_beta: Arc::new(|u, _| Ok(Array1::zeros(u.len()))),
            completion_psi: None,
            response_scale: 1.0,
            first: Arc::new(|deltas: &[Array1<f64>]| {
                Ok(deltas
                    .iter()
                    .map(|delta| Some(Array2::from_diag(delta)))
                    .collect())
            }),
            second: Arc::new(|pairs| {
                Ok(pairs
                    .iter()
                    .map(|(u, _)| Array2::zeros((u.len(), u.len())))
                    .collect())
            }),
        }
    }

    fn wrapper() -> JeffreysHphiAwareJointDerivatives<'static> {
        JeffreysHphiAwareJointDerivatives::new(Box::new(SilentInner), identity_drift(), 2)
    }

    #[test]
    fn first_order_correction_prices_the_negated_mode_response_2612() {
        let v_k = array![0.25, -1.5];
        let correction = wrapper()
            .hessian_derivative_correction(&v_k)
            .expect("the composed correction evaluates")
            .expect("the drift contributes");
        assert_eq!(correction, Array2::from_diag(&array![-0.25, 1.5]));
    }

    #[test]
    fn second_order_correction_prices_the_second_mode_response_unnegated_2612() {
        let (v_k, v_l, u_kl) = (array![0.25, -1.5], array![-0.75, 0.5], array![2.0, -3.0]);
        let correction = wrapper()
            .hessian_second_derivative_correction(&v_k, &v_l, &u_kl)
            .expect("the composed correction evaluates")
            .expect("the drift contributes");
        assert_eq!(
            correction,
            Array2::from_diag(&u_kl),
            "the second-order Jeffreys drift must be taken along `u_kl` itself; \
             `-u_kl` would be the same magnitude with the wrong sign"
        );
    }

    #[test]
    fn second_order_correction_includes_bilinear_jeffreys_motion_979() {
        // Hphi(beta)=diag(beta^2)/2 at beta=ones has D Hphi[u]=diag(u)
        // and D² Hphi[u,v]=diag(u*v). The independent polynomial identity
        // exposes a dropped mixed term or a single incorrect response sign.
        let mut drift = identity_drift();
        drift.second = Arc::new(|pairs| {
            Ok(pairs
                .iter()
                .map(|(u, v)| Array2::from_diag(&(u * v)))
                .collect())
        });
        let provider = JeffreysHphiAwareJointDerivatives::new(Box::new(SilentInner), drift, 2);
        let (u, v, second) = (array![0.25, -1.5], array![-0.75, 0.5], array![2.0, -3.0]);
        let expected = Array2::from_diag(&(&second + &(&u * &v)));
        let scalar = provider
            .hessian_second_derivative_correction(&u, &v, &second)
            .unwrap()
            .unwrap();
        assert_eq!(scalar, expected);
        let batch = provider
            .hessian_second_derivative_corrections_result(&[(u, v, second)])
            .unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(
            batch[0].clone().unwrap().into_operator().to_dense(),
            expected
        );
    }

    #[test]
    fn batched_second_order_corrections_match_the_singular_ones_2612() {
        let triples = vec![
            (array![0.25, -1.5], array![-0.75, 0.5], array![2.0, -3.0]),
            (array![1.0, 0.0], array![0.0, 1.0], array![-0.5, 0.125]),
        ];
        let wrapper = wrapper();
        let batched = wrapper
            .hessian_second_derivative_corrections_result(&triples)
            .expect("the batched walk evaluates");
        assert_eq!(batched.len(), triples.len());
        for (result, (v_k, v_l, u_kl)) in batched.into_iter().zip(triples.iter()) {
            let singular = wrapper
                .hessian_second_derivative_correction(v_k, v_l, u_kl)
                .expect("the singular walk evaluates")
                .expect("the drift contributes");
            let batched_dense = result
                .expect("the drift contributes")
                .into_operator()
                .to_dense();
            assert_eq!(batched_dense, singular);
        }
    }
}
