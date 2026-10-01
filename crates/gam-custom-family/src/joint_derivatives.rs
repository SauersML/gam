//! Joint second-derivative correction, the `HessianDerivativeProvider`
//! implementations (borrowed / owned / Jeffreys-aware), scaled hyper-
//! operators, and the ext-coord bundle, split out of `outer_objective.rs`
//! by concern (#1145). Re-exported via `custom_family`.

use super::*;
use gam_solve::estimate::reml::reml_outer_engine::formed_second_derivative_corrections_applied;

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

/// The corrections `D_βH[u_kl] + D²_βH[v_l, v_k]` of
/// [`joint_second_derivative_correction_result`], each applied to one vector
/// `x`, from `1 + K` drifts instead of two per triple (`K` distinct `v_k`).
///
/// `H` is the exact coefficient Hessian of one scalar objective
/// (`ExactNewtonJointHessianWorkspace`), so `D_βH` and `D²_βH` are its third
/// and fourth derivative tensors, symmetric in every slot:
/// `D_βH[u]x = D_βH[x]u` and `D²_βH[v_l, v_k]x = D²_βH[x, v_k]v_l`. One drift
/// `D_βH[x]` and one `D²_βH[x, v_k]` per distinct `v_k` then give every
/// triple, where the operator form costs a full row pass per triple for each
/// term. `None` when a drift is unavailable, so the caller forms the
/// operators exactly as before.
fn symmetric_second_derivative_corrections_applied(
    compute_dh: &crate::joint_newton::DriftDerivFn<'_>,
    compute_d2h: &crate::joint_newton::DriftSecondDerivFn<'_>,
    compute_d2h_many: Option<&crate::joint_newton::DriftSecondDerivManyFn<'_>>,
    triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    x: &Array1<f64>,
) -> Result<Option<Vec<Option<Array1<f64>>>>, CustomFamilyError> {
    let Some(drift_at_x) = compute_dh(x)? else {
        return Ok(None);
    };
    // Each triple's first response, by its index among the distinct ones.
    let mut firsts: Vec<&Array1<f64>> = Vec::new();
    let first_of: Vec<usize> = triples
        .iter()
        .map(|(v_k, _, _)| {
            firsts.iter().position(|first| *first == v_k).unwrap_or_else(|| {
                firsts.push(v_k);
                firsts.len() - 1
            })
        })
        .collect();
    let second_drifts: Vec<Option<DriftDerivResult>> = match compute_d2h_many {
        Some(many) => {
            let pairs: Vec<(Array1<f64>, Array1<f64>)> =
                firsts.iter().map(|v_k| (x.clone(), (*v_k).clone())).collect();
            many(&pairs)?
        }
        None => {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
            firsts
                .par_iter()
                .map(|v_k| gam_problem::with_nested_parallel(|| compute_d2h(x, v_k)))
                .collect::<Result<_, CustomFamilyError>>()?
        }
    };
    if second_drifts.len() != firsts.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "second-derivative drifts: {} returned for {} directions",
            second_drifts.len(),
            firsts.len()
        )));
    }
    let Some(second_drifts) = second_drifts.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    Ok(Some(
        triples
            .iter()
            .zip(&first_of)
            .map(|((_, v_l, u_kl), &first)| {
                Some(drift_at_x.apply(u_kl) + second_drifts[first].apply(v_l))
            })
            .collect(),
    ))
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

    fn hessian_second_derivative_corrections_applied(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
        x: &Array1<f64>,
    ) -> Result<Vec<Option<Array1<f64>>>, String> {
        match symmetric_second_derivative_corrections_applied(
            self.compute_dh,
            self.compute_d2h,
            self.compute_d2h_many,
            triples,
            x,
        )
        .map_err(|error| error.to_string())?
        {
            Some(applied) => Ok(applied),
            None => formed_second_derivative_corrections_applied(self, triples, x),
        }
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
    /// The second-order corrections' logdet traces from the workspace's row
    /// kernels (gam#2922), for the drifts `compute_dh` and `compute_d2h` form.
    pub(crate) second_correction_traces: Option<Arc<DriftSecondCorrectionTracesFn>>,
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

    fn hessian_second_derivative_corrections_applied(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
        x: &Array1<f64>,
    ) -> Result<Vec<Option<Array1<f64>>>, String> {
        match symmetric_second_derivative_corrections_applied(
            &*self.compute_dh,
            &*self.compute_d2h,
            self.compute_d2h_many.as_deref(),
            triples,
            x,
        )
        .map_err(|error| error.to_string())?
        {
            Some(applied) => Ok(applied),
            None => formed_second_derivative_corrections_applied(self, triples, x),
        }
    }

    fn hessian_second_derivative_correction_traces(
        &self,
        factor: &Array2<f64>,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Option<Vec<f64>>, String> {
        // Display boundary: `HessianDerivativeProvider` is `String`-erroring (gam#2689).
        match self.second_correction_traces.as_ref() {
            Some(traces) => traces(factor, triples).map_err(|error| error.to_string()),
            None => Ok(None),
        }
    }

    fn has_hessian_second_derivative_correction_traces(&self) -> bool {
        self.second_correction_traces.is_some()
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
    /// `D_β completion[δ]` along each direction, present exactly when the criterion
    /// prices the complete Jeffreys curvature `H_Φ + completion` (gam#2894).
    pub(crate) completion_first: Option<CompletionDriftFn>,
    /// `D² completion[u, v]` for each pair, present with `completion_first`.
    pub(crate) completion_second: Option<CompletionSecondDriftFn>,
    /// Whether the mode response's operator carries a completion at all. Without one the
    /// stationarity operator is the log-determinant operator, so their difference is constant and
    /// no right-hand-side correction exists (gam#2765).
    pub(crate) completion_present: bool,
    /// Whether the family supplies the third information derivative `completion_beta` reads, so a
    /// present, unpriced completion's motion can be priced (gam#2765).
    pub(crate) completion_derivatives_supplied: bool,
}

/// `D_β completion[δ]` for many directions (gam#2894).
pub(crate) type CompletionDriftFn =
    Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Array2<f64>>, CustomFamilyError> + Send + Sync>;

/// `D² completion[u, v]` for many pairs (gam#2894).
pub(crate) type CompletionSecondDriftFn = Arc<
    dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Array2<f64>>, CustomFamilyError>
        + Send
        + Sync,
>;

impl JeffreysHphiDriftBatchFn {
    /// Drift of the criterion's Jeffreys curvature along each `δβ`: `D_β H_Φ[δ]`, plus
    /// `D_β completion[δ]` when the criterion prices the completion (gam#2894).
    pub(crate) fn criterion_first(
        &self,
        deltas: &[Array1<f64>],
    ) -> Result<Vec<Option<Array2<f64>>>, CustomFamilyError> {
        let mut drifts = (self.first)(deltas)?;
        if let Some(completion_first) = self.completion_first.as_ref() {
            let completion = completion_first(deltas)?;
            if completion.len() != drifts.len() {
                return Err(CustomFamilyError::trial_point(format!(
                    "priced Jeffreys completion drift returned {} results for {} directions",
                    completion.len(),
                    drifts.len()
                )));
            }
            for (drift, completion) in drifts.iter_mut().zip(completion) {
                match drift {
                    Some(matrix) => *matrix += &completion,
                    None => *drift = Some(completion),
                }
            }
        }
        Ok(drifts)
    }

    /// Mixed second drift of the criterion's Jeffreys curvature for each pair: `D² H_Φ`,
    /// plus `D² completion` when the criterion prices the completion (gam#2894).
    pub(crate) fn criterion_second(
        &self,
        pairs: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Array2<f64>>, CustomFamilyError> {
        let mut drifts = (self.second)(pairs)?;
        if let Some(completion_second) = self.completion_second.as_ref() {
            let completion = completion_second(pairs)?;
            if completion.len() != drifts.len() {
                return Err(CustomFamilyError::trial_point(format!(
                    "priced Jeffreys completion second drift returned {} results for {} pairs",
                    completion.len(),
                    drifts.len()
                )));
            }
            for (drift, completion) in drifts.iter_mut().zip(completion) {
                *drift += &completion;
            }
        }
        Ok(drifts)
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
        self.drift.criterion_first(&deltas)
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
        let mut out = self.drift.criterion_first(std::slice::from_ref(delta))?;
        Ok(out.pop().flatten())
    }
}

impl HessianDerivativeProvider for JeffreysHphiAwareJointDerivatives<'_> {
    fn mode_response_rhs_correction(&self) -> Option<gam_solve::estimate::reml::reml_outer_engine::ModeResponseRhsCorrectionFn> {
        // The correction is `D(C)` for `C = M_stationarity − M_logdet`. A criterion that
        // prices the completion already carries it in `M_logdet`: `criterion_first` folds
        // `D_β completion` into every `hessian_derivative_correction`, so `C = 0` and the
        // pair right-hand side moves the completion through `h_k·v` already. Installing it
        // here as well counted it twice (the survival marginal-slope outer Hessian read
        // 0.40957 at [1,1] against a central difference of 0.37341; gam#2894). Without a
        // completion the two operators are one, so `C = 0` as well (gam#2765).
        if self.drift.completion_first.is_some() || !self.drift.completion_present {
            return None;
        }
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

    /// The completion's motion needs the family's third information derivative (gam#2765).
    fn mode_response_rhs_correction_supplied(&self) -> bool {
        self.drift.completion_derivatives_supplied
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
        let mut mixed = self.drift.criterion_second(&[(v_k.clone(), v_l.clone())])
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
        let mut drifts = self.drift.criterion_first(&deltas).map_err(|error| error.to_string())?;
        let pairs: Vec<_> = triples
            .iter()
            .map(|(u, v, _)| (u.clone(), v.clone()))
            .collect();
        let mixed = self.drift.criterion_second(&pairs).map_err(|error| error.to_string())?;
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

    /// The inner provider's products (its own fast route where it has one) plus
    /// the `H_Φ` drifts of [`Self::hessian_second_derivative_corrections_result`]
    /// applied to `x`. Those drifts are not derivatives of the likelihood
    /// Hessian, so they are formed as before; only the inner share changes route.
    fn hessian_second_derivative_corrections_applied(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
        x: &Array1<f64>,
    ) -> Result<Vec<Option<Array1<f64>>>, String> {
        let inner = self
            .inner
            .hessian_second_derivative_corrections_applied(triples, x)?;
        let deltas: Vec<Array1<f64>> = triples.iter().map(|(_, _, u_kl)| u_kl.clone()).collect();
        let drifts = self.drift.criterion_first(&deltas).map_err(|error| error.to_string())?;
        let pairs: Vec<_> = triples
            .iter()
            .map(|(u, v, _)| (u.clone(), v.clone()))
            .collect();
        let mixed = self.drift.criterion_second(&pairs).map_err(|error| error.to_string())?;
        if drifts.len() != triples.len() || mixed.len() != triples.len() || inner.len() != triples.len() {
            return Err(format!(
                "JeffreysHphiAwareJointDerivatives: applied second-order corrections returned \
                 {}/{}/{} results for {} triples",
                inner.len(),
                drifts.len(),
                mixed.len(),
                triples.len()
            ));
        }
        Ok(inner
            .into_iter()
            .zip(drifts)
            .zip(mixed)
            .map(|((inner, drift), mixed)| {
                let jeffreys = match drift {
                    Some(matrix) => (matrix + &mixed).dot(x),
                    None => mixed.dot(x),
                };
                Some(match inner {
                    Some(applied) => applied + &jeffreys,
                    None => jeffreys,
                })
            })
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
                let mut drift = drift_first.criterion_first(std::slice::from_ref(direction))
                    .map_err(|error| error.to_string())?;
                if drift.len() != 1 {
                    return Err("Jeffreys first callback batch length mismatch".to_string());
                }
                Ok(compose_drift(inner, drift.pop().flatten(), p))
            }),
            second: Arc::new(move |u, v| {
                let inner = second(u, v)?;
                let mut drift = drift_second.criterion_second(&[(u.clone(), v.clone())])
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

/// The explicit ψ partial of the Jeffreys completion `∂C/∂ψ|_β` as a matrix, per ψ axis (gam#2930).
pub(crate) type CompletionPsiPartial = Arc<dyn Fn(usize) -> Result<Array2<f64>, CustomFamilyError> + Send + Sync>;

/// The second explicit ψ partial of the Jeffreys completion `∂²C/∂ψ_i∂ψ_j|_β`, per ψ pair (gam#2930).
pub(crate) type CompletionPsiPair = Arc<dyn Fn(usize, usize) -> Result<Array2<f64>, CustomFamilyError> + Send + Sync>;

/// The mixed drift of the Jeffreys completion `∂_ψ D_β C[v]`, per ψ axis and coefficient direction
/// (gam#2930).
pub(crate) type CompletionBetaPsi = Arc<dyn Fn(usize, &Array1<f64>) -> Result<Array2<f64>, CustomFamilyError> + Send + Sync>;

pub(crate) struct ExtCoordBundle {
    pub(crate) completion_psi: Option<CompletionPsiAction>,
    pub(crate) completion_psi_partial: Option<CompletionPsiPartial>,
    pub(crate) completion_psi_pair: Option<CompletionPsiPair>,
    pub(crate) completion_beta_psi: Option<CompletionBetaPsi>,
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
            completion_psi_partial: self.completion_psi_partial.map(|callback| {
                Arc::new(move |psi: usize| callback(psi).map(|partial| partial * scale))
                    as CompletionPsiPartial
            }),
            completion_psi_pair: self.completion_psi_pair.map(|callback| {
                Arc::new(move |psi_i: usize, psi_j: usize| {
                    callback(psi_i, psi_j).map(|pair| pair * scale)
                }) as CompletionPsiPair
            }),
            completion_beta_psi: self.completion_beta_psi.map(|callback| {
                Arc::new(move |psi: usize, direction: &Array1<f64>| {
                    callback(psi, direction).map(|drift| drift * scale)
                }) as CompletionBetaPsi
            }),
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
            completion_first: None,
            completion_second: None,
            completion_present: false,
            completion_derivatives_supplied: false,
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

    /// gam#2894: a criterion priced on the complete Jeffreys curvature adds the completion's
    /// β-drift to `D_β H_Φ` along the same negated mode response and the same unnegated second
    /// response, and adds its second drift to the mixed Jeffreys drift.
    #[test]
    fn priced_completion_drifts_ride_with_the_jeffreys_drifts_2894() {
        let mut drift = identity_drift();
        let completion_first: CompletionDriftFn = Arc::new(|deltas: &[Array1<f64>]| {
            Ok(deltas
                .iter()
                .map(|delta| Array2::from_diag(&delta.mapv(|value| 2.0 * value)))
                .collect())
        });
        let completion_second: CompletionSecondDriftFn =
            Arc::new(|pairs: &[(Array1<f64>, Array1<f64>)]| {
                Ok(pairs.iter().map(|(u, v)| Array2::from_diag(&(u * v))).collect())
            });
        drift.completion_first = Some(completion_first);
        drift.completion_second = Some(completion_second);
        let composed = JeffreysHphiAwareJointDerivatives::new(Box::new(SilentInner), drift, 2);
        let v_k = array![0.25, -1.5];
        let first = composed
            .hessian_derivative_correction(&v_k)
            .expect("the composed correction evaluates")
            .expect("the drifts contribute");
        assert_eq!(first, Array2::from_diag(&array![-0.75, 4.5]));
        let (v_l, u_kl) = (array![-0.75, 0.5], array![2.0, -3.0]);
        let second = composed
            .hessian_second_derivative_correction(&v_k, &v_l, &u_kl)
            .expect("the composed correction evaluates")
            .expect("the drifts contribute");
        assert_eq!(second, Array2::from_diag(&array![5.8125, -9.75]));
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

    /// gam#2765: without a completion the stationarity operator is the log-determinant operator, so
    /// the provider names no right-hand-side correction and the fold record's `t₃` is complete by
    /// contract. `D_β H_Φ[δ] = diag(δ)` at `δ = −v` gives `t₃ = Σ vᵢ³` along `v`.
    #[test]
    fn a_drift_without_a_completion_names_no_right_hand_side_correction_2765() {
        let provider = wrapper();
        assert!(
            provider.mode_response_rhs_correction().is_none(),
            "no completion, so no correction exists"
        );
        let direction = array![0.6, 0.8];
        let (third, completion) =
            gam_solve::estimate::reml::reml_outer_engine::inner_mode_third_derivative(&provider, &direction)
                .expect("t3 along the direction");
        assert_eq!(
            completion,
            gam_solve::estimate::reml::reml_outer_engine::CompletionShare::Priced
        );
        let expected = 0.6_f64.powi(3) + 0.8_f64.powi(3);
        assert!((third - expected).abs() <= 1e-12 * expected, "t3={third} expected={expected}");
    }

    /// gam#2765: a completion that is present but whose derivatives the family does not supply is
    /// recorded as not supplied, and its motion is never requested. With the derivatives supplied,
    /// the same completion's motion `D_β C[−v]·v` enters `t₃`.
    #[test]
    fn a_present_completion_is_priced_only_where_its_derivatives_are_supplied_2765() {
        let direction = array![0.6, 0.8];
        let likelihood_share = 0.6_f64.powi(3) + 0.8_f64.powi(3);

        let mut unsupplied = identity_drift();
        unsupplied.completion_present = true;
        unsupplied.completion_beta = Arc::new(|_, _| {
            panic!("a completion declared without derivatives is never asked for its motion")
        });
        let provider = JeffreysHphiAwareJointDerivatives::new(Box::new(SilentInner), unsupplied, 2);
        assert!(
            provider.mode_response_rhs_correction().is_some(),
            "the completion moves, so the correction exists"
        );
        assert!(
            !provider.mode_response_rhs_correction_supplied(),
            "its derivatives are declared absent"
        );
        let (third, completion) =
            gam_solve::estimate::reml::reml_outer_engine::inner_mode_third_derivative(&provider, &direction)
                .expect("t3 along the direction");
        assert_eq!(
            completion,
            gam_solve::estimate::reml::reml_outer_engine::CompletionShare::NotSupplied
        );
        assert!(
            (third - likelihood_share).abs() <= 1e-12 * likelihood_share,
            "t3={third} carries the log-determinant operator's share {likelihood_share} alone"
        );

        let mut supplied = identity_drift();
        supplied.completion_present = true;
        supplied.completion_derivatives_supplied = true;
        // `completion_beta(δ, u) = δ ⊙ u`, so the correction at `δ = −v` adds `−v²` to the image.
        supplied.completion_beta = Arc::new(|delta, u| Ok(delta * u));
        let provider = JeffreysHphiAwareJointDerivatives::new(Box::new(SilentInner), supplied, 2);
        let (third, completion) =
            gam_solve::estimate::reml::reml_outer_engine::inner_mode_third_derivative(&provider, &direction)
                .expect("t3 along the direction");
        assert_eq!(
            completion,
            gam_solve::estimate::reml::reml_outer_engine::CompletionShare::Priced
        );
        let complete = 2.0 * likelihood_share;
        assert!(
            (third - complete).abs() <= 1e-12 * complete,
            "t3={third} carries the completion's motion: expected {complete}"
        );
    }
}

#[cfg(test)]
mod applied_second_corrections_tests {
    //! #4564: the cone term's outer Hessian reads each pair's second-order
    //! correction only against one vector, and the joint providers form those
    //! products from `D_βH[x]` and one `D²_βH[x, v_k]` per direction by the
    //! symmetry of an exact Hessian's derivatives. Here `H` is the Hessian of
    //! `f(β) = Σ_i w_i·g(a_iᵀβ)` with `g''' = sinh`, `g'''' = cosh`, so
    //! `D_βH[u] = Σ w_i sinh(t_i)(a_iᵀu)a_ia_iᵀ` and
    //! `D²_βH[u, v] = Σ w_i cosh(t_i)(a_iᵀu)(a_iᵀv)a_ia_iᵀ` exactly.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const P: usize = 5;
    const ROWS: usize = 9;

    fn design() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let a = Array2::from_shape_fn((ROWS, P), |(i, j)| {
            ((1 + i * 7 + j * 3) as f64 * 0.37).sin() + 0.2 * (j as f64 - 2.0)
        });
        let w = Array1::from_shape_fn(ROWS, |i| 0.5 + 0.1 * i as f64);
        let beta = Array1::from_shape_fn(P, |j| 0.3 * (j as f64 - 2.0));
        (a, w, beta)
    }

    fn vector(seed: usize) -> Array1<f64> {
        Array1::from_shape_fn(P, |j| ((seed * 11 + j * 5 + 1) as f64 * 0.73).cos())
    }

    fn weighted_outer(a: &Array2<f64>, weight: impl Fn(usize) -> f64) -> Array2<f64> {
        let mut out = Array2::zeros((P, P));
        for i in 0..ROWS {
            let row = a.row(i);
            let scale = weight(i);
            for r in 0..P {
                for c in 0..P {
                    out[[r, c]] += scale * row[r] * row[c];
                }
            }
        }
        out
    }

    fn provider(
        first_calls: Arc<AtomicUsize>,
        second_calls: Arc<AtomicUsize>,
    ) -> OwnedJointDerivProvider {
        let (a, w, beta) = design();
        let t = a.dot(&beta);
        let (a1, w1, t1) = (a.clone(), w.clone(), t.clone());
        let (a2, w2, t2) = (a, w, t);
        OwnedJointDerivProvider {
            compute_dh: Arc::new(move |u: &Array1<f64>| {
                first_calls.fetch_add(1, Ordering::Relaxed);
                let au = a1.dot(u);
                Ok(Some(DriftDerivResult::Dense(weighted_outer(&a1, |i| {
                    w1[i] * t1[i].sinh() * au[i]
                }))))
            }),
            compute_dh_many: None,
            compute_d2h: Arc::new(move |u: &Array1<f64>, v: &Array1<f64>| {
                second_calls.fetch_add(1, Ordering::Relaxed);
                let (au, av) = (a2.dot(u), a2.dot(v));
                Ok(Some(DriftDerivResult::Dense(weighted_outer(&a2, |i| {
                    w2[i] * t2[i].cosh() * au[i] * av[i]
                }))))
            }),
            compute_d2h_many: None,
            second_correction_traces: None,
            family_outer_hessian_operator: None,
        }
    }

    #[test]
    fn applied_corrections_match_the_formed_operators_from_k_plus_one_drifts_4564() {
        let directions = 4;
        let responses: Vec<Array1<f64>> = (0..directions).map(vector).collect();
        let mut triples = Vec::new();
        for i in 0..directions {
            for j in i..directions {
                triples.push((
                    responses[i].clone(),
                    responses[j].clone(),
                    vector(100 + 10 * i + j),
                ));
            }
        }
        let x = vector(999);

        let (formed_first, formed_second) =
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let formed = formed_second_derivative_corrections_applied(
            &provider(Arc::clone(&formed_first), Arc::clone(&formed_second)),
            &triples,
            &x,
        )
        .expect("formed corrections");
        // Premise: the formed route prices two drifts per triple.
        assert_eq!(formed_first.load(Ordering::Relaxed), triples.len());
        assert_eq!(formed_second.load(Ordering::Relaxed), triples.len());

        let (first, second) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let applied = provider(Arc::clone(&first), Arc::clone(&second))
            .hessian_second_derivative_corrections_applied(&triples, &x)
            .expect("applied corrections");
        assert_eq!(first.load(Ordering::Relaxed), 1, "one D_βH[x]");
        assert_eq!(second.load(Ordering::Relaxed), directions, "one D²_βH[x, v_k] per direction");

        // Both routes sum the same products of exact inputs in different orders: each entry is
        // a dot product over P of row sums over ROWS of products of `a_iᵀ·` (2P operations each),
        // `w_i`, `g(t_i)` and two row entries, so each side carries at most
        // k = ROWS + 3P + 8 rounded operations per term and the two differ by at most
        // 2γ_k times the entry's absolute-value majorant (Higham, Lemma 3.1).
        let (a, w, beta) = design();
        let t = a.dot(&beta);
        let abs_a = a.mapv(f64::abs);
        let growth = 2.0 * gam_linalg::roundoff::accumulation_growth(ROWS + 3 * P + 8);
        assert_eq!(applied.len(), formed.len());
        for (index, ((fast, slow), (v_k, v_l, u_kl))) in
            applied.iter().zip(&formed).zip(&triples).enumerate()
        {
            let (fast, slow) = (fast.as_ref().expect("fast"), slow.as_ref().expect("formed"));
            let reach = |v: &Array1<f64>| abs_a.dot(&v.mapv(f64::abs));
            let (ru, rk, rl, rx) = (reach(u_kl), reach(v_k), reach(v_l), reach(&x));
            let weights = Array1::from_shape_fn(ROWS, |i| {
                w[i].abs() * rx[i] * (t[i].sinh().abs() * ru[i] + t[i].cosh() * rk[i] * rl[i])
            });
            let majorant = abs_a.t().dot(&weights);
            for (entry, ((f, s), m)) in fast.iter().zip(slow.iter()).zip(majorant.iter()).enumerate() {
                let band = growth * m;
                assert!(
                    (f - s).abs() <= band,
                    "triple {index} entry {entry}: {f:e} vs {s:e} (band {band:e})"
                );
            }
            // Premise: the correction is not vanishing, so the comparison is not vacuous.
            assert!(slow.iter().any(|value| value.abs() > 1e3 * growth * majorant.sum()));
        }
    }
}
