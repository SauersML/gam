//! The λ→∞ limit of the REML/LAML criterion, built analytically (#2348 Inc 5;
//! LAML extension per #2349 rounds 9–10b).
//!
//! [`RemlState::rail_face_limit`] answers one question exactly: *what is this
//! fit when the smoothing parameters on a rail face are literally infinite?*
//! It does not evaluate the criterion at a huge finite λ — at ρ ≈ 30 the
//! logdet pair `½log|H| − ½log|S_λ|₊` is cancelling `r_F·log λ ≈ 400` against
//! itself and its ρ-derivative is instrument noise. Instead it forms the limit
//! objects directly:
//!
//! * the **limit fit** — the model restricted to `N = ⋂_{j∈F} null(S_j)`,
//!   solved with the surviving penalties. This is the λ=∞ model, and its
//!   criterion value is `V_∞` (the divergent `log|QᵀS_FQ|` cancels exactly
//!   between the two logdets, so the limit is finite);
//! * the **limit score** `g_c = ∇ℓ(β̂_∞) − S_Rβ̂_∞`, the Lagrange force the
//!   constraint carries, computed from `O(1)` quantities rather than read off
//!   a `λ⁻¹`-scale coefficient tail;
//! * the **first-order form** `C` on the released subspace, whose positive
//!   definiteness is the whole face certificate (see
//!   [`crate::rho_optimizer::rail_face`] for both derivations and what
//!   `C ≻ 0` proves).
//!
//! # Scope: two closed forms, one dispatch
//!
//! *Profiled-Gaussian REML* (identity link, dense design): the working weights
//! do not move with `β̂`, the limit fit is a linear solve, and
//! `C = Schur_Z(K) − Schur_Z(S_R) − g_Qg_Qᵀ/φ̂`.
//!
//! *Fixed-unit-dispersion LAML* (binomial, poisson; dense design): the working
//! weights move with `β̂`, so the Laplace logdet contributes the symmetric
//! rank-2 drift term `g_Q d_Qᵀ + d_Q g_Qᵀ` with `d = ½Xᵀ(c ⊙ a)` built from
//! the limit fit's own third-derivative array and leverage. The limit fit is
//! an ordinary P-IRLS solve of the null-space-restricted model, run by the
//! same inner engine as every other fit, so the closed form expands exactly
//! the criterion production minimizes.
//!
//! Everything else declines, TYPED: `OutsideClosedForm` says this fit's
//! criterion is not one the forms model (a future form may still apply), while
//! `FaceUnavailable` is a statement about the face that no other form rescues.
//! Either way the caller keeps whatever authority it already had.

use super::*;
use crate::rho_optimizer::rail_face::{
    LamlFaceParts, RailFaceLimitOutcome, face_release_bases, gaussian_rail_face_limit,
    laml_rail_face_limit, released_rank, split_face_penalties,
};
use crate::rho_optimizer::zero_smoothing_face::{
    ZeroSmoothingFaceOutcome, gaussian_zero_smoothing_face,
};

impl RemlState<'_> {
    /// Build the analytic λ→∞ limit data for the rail face `face` (ρ-block
    /// indices), at the smoothing parameters `rho`.
    ///
    /// This is the gate; the algebra lives in
    /// [`crate::rho_optimizer::rail_face`], which any caller holding the same
    /// parts can use directly. A decline is never an error, and it is TYPED
    /// (see the module doc).
    pub(crate) fn rail_face_limit(
        &self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        // Scope clauses shared by BOTH closed forms: each names a criterion
        // term (or a geometry) neither form models.
        let outside = if self.linear_constraints.is_some()
            || self.coefficient_lower_bounds.is_some()
        {
            Some("the fit carries coefficient constraints, so the limit is a constrained optimum")
        } else if self.runtime_mixture_link_state.is_some() || self.runtime_sas_link_state.is_some()
        {
            Some(
                "the link carries runtime state, so the criterion is not one the closed forms model",
            )
        } else if !self
            .rho_prior
            .upper_tail_gradient_vanishes_everywhere(rho.len())
        {
            // Not `matches!(prior, Flat)`: `Gamma(1, 0)` is the same flat
            // coordinate spelled differently, and an `Independent` prior can be
            // flat on the face and configured elsewhere. The question is the
            // one the λ=∞ law actually asks — does the prior's gradient survive
            // into the tail — and `RhoPrior` carries the answer (#2450/#2427).
            Some(
                "the rho-prior's gradient survives into the lambda -> infinity tail, so this \
                 criterion has no lambda = infinity face for the closed form to describe",
            )
        } else if !matches!(self.x, DesignMatrix::Dense(_)) {
            Some("the design is not dense")
        } else {
            None
        };
        if let Some(reason) = outside {
            return Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: reason.to_string(),
            });
        }
        let design = self
            .x
            .try_to_dense_by_chunks("rail face limit")
            .map_err(EstimationError::RemlOptimizationFailed)?;
        if design.ncols() != self.p {
            return Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: format!(
                    "the densified design has {} columns against a coefficient layout of {}",
                    design.ncols(),
                    self.p
                ),
            });
        }

        if reml_is_gaussian_identity(&self.config.likelihood) {
            // The builder wants the response already net of any offset.
            let mut response = self.y.to_owned();
            if self.offset.len() == response.len() {
                response -= &self.offset;
            }
            return Ok(gaussian_rail_face_limit(
                design.view(),
                response.view(),
                self.weights,
                self.canonical_penalties.as_slice(),
                rho,
                face,
            ));
        }

        // ── the LAML closed form ────────────────────────────────────────
        // Restricted to families whose dispersion is exactly 1, so the
        // criterion is `−ℓ + ½β̂ᵀSβ̂ + ½log|H| − ½log|S|₊` with `−ℓ` and `H`
        // in the same units and no profiled/estimated scale anywhere.
        if !matches!(
            self.config.likelihood.spec.response,
            ResponseFamily::Binomial | ResponseFamily::Poisson
        ) {
            return Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "the LAML closed form is derived for fixed-unit-dispersion families \
                         (binomial, poisson); this family's dispersion enters the criterion \
                         through terms the form does not carry"
                    .to_string(),
            });
        }
        if self.config.firth_bias_reduction {
            return Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "the Jeffreys/Firth term is armed: the criterion carries a logdet \
                         prior this closed form does not model"
                    .to_string(),
            });
        }
        self.laml_rail_face_limit_via_limit_fit(&design, rho, face)
    }

    /// Build the analytic λ→0 (zero-smoothing) face law for `face` (ρ-block
    /// indices) at `rho`: the exact slopes `c′_j = ∂V/∂λ_j` at `λ_face = 0`
    /// (see [`crate::rho_optimizer::zero_smoothing_face`]).
    ///
    /// Profiled-Gaussian REML only; every other criterion declines TYPED. The
    /// ρ-prior must be exactly linear in `λ` on each face coordinate, which is
    /// the only way it can leave a finite slope at `λ = 0`.
    pub(crate) fn zero_smoothing_face(
        &self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<ZeroSmoothingFaceOutcome, EstimationError> {
        let outside = if self.linear_constraints.is_some()
            || self.coefficient_lower_bounds.is_some()
        {
            Some("the fit carries coefficient constraints, so the limit is a constrained optimum")
        } else if self.runtime_mixture_link_state.is_some() || self.runtime_sas_link_state.is_some()
        {
            Some(
                "the link carries runtime state, so the criterion is not one the closed form models",
            )
        } else if !reml_is_gaussian_identity(&self.config.likelihood) {
            Some(
                "the zero-smoothing closed form is derived for profiled-Gaussian REML; this \
                 family's criterion carries terms the form does not",
            )
        } else if !matches!(self.x, DesignMatrix::Dense(_)) {
            Some("the design is not dense")
        } else {
            None
        };
        if let Some(reason) = outside {
            return Ok(ZeroSmoothingFaceOutcome::OutsideClosedForm {
                reason: reason.to_string(),
            });
        }
        let mut prior_rates = Vec::with_capacity(face.len());
        for &coordinate in face.iter() {
            match self.rho_prior.lower_tail_linear_rate(coordinate) {
                Some(rate) => prior_rates.push(rate),
                None => {
                    return Ok(ZeroSmoothingFaceOutcome::OutsideClosedForm {
                        reason: format!(
                            "the rho-prior on coordinate {coordinate} is not linear in lambda, so \
                             its gradient does not vanish into the lambda -> 0 tail and there is \
                             no zero-smoothing face for the closed form to describe"
                        ),
                    });
                }
            }
        }
        let design = self
            .x
            .try_to_dense_by_chunks("zero-smoothing face")
            .map_err(EstimationError::RemlOptimizationFailed)?;
        if design.ncols() != self.p {
            return Ok(ZeroSmoothingFaceOutcome::OutsideClosedForm {
                reason: format!(
                    "the densified design has {} columns against a coefficient layout of {}",
                    design.ncols(),
                    self.p
                ),
            });
        }
        let mut response = self.y.to_owned();
        if self.offset.len() == response.len() {
            response -= &self.offset;
        }
        Ok(gaussian_zero_smoothing_face(
            design.view(),
            response.view(),
            self.weights,
            self.canonical_penalties.as_slice(),
            rho,
            face,
            &prior_rates,
        ))
    }

    /// Solve the λ=∞ limit model — the fit restricted to the face's common
    /// null space with the surviving penalties, through the SAME inner engine
    /// as every production fit — and hand its converged row bundle to the
    /// LAML face form.
    fn laml_rail_face_limit_via_limit_fit(
        &self,
        design: &Array2<f64>,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        let split =
            match split_face_penalties(self.canonical_penalties.as_slice(), rho, face, self.p) {
                Ok(split) => split,
                Err(outcome) => return Ok(outcome),
            };
        let bases = match face_release_bases(&split.s_face_unit) {
            Ok(bases) => bases,
            Err(outcome) => return Ok(outcome),
        };
        let pinned = bases.z_basis.ncols();
        if pinned == 0 {
            return Ok(RailFaceLimitOutcome::FaceUnavailable {
                reason:
                    "the face releases every direction: the lambda=infinity limit model is empty"
                        .to_string(),
            });
        }

        // The limit model's penalties: `ZᵀS_jZ` for each survivor at its own
        // λ. A survivor whose penalty vanishes on the null space constrains
        // only released directions and drops out of the limit model.
        let reduced_design = design.dot(&bases.z_basis);
        let mut reduced_specs = Vec::new();
        let mut reduced_null_dims = Vec::new();
        let mut reduced_rho = Vec::new();
        for (j, penalty) in self.canonical_penalties.iter().enumerate() {
            if split.face_sorted.contains(&j) {
                continue;
            }
            let mut full = Array2::<f64>::zeros((self.p, self.p));
            let cols = penalty.col_range.clone();
            for (li, gi) in cols.clone().enumerate() {
                for (lj, gj) in cols.clone().enumerate() {
                    full[[gi, gj]] += penalty.local[[li, lj]];
                }
            }
            let reduced = bases.z_basis.t().dot(&full).dot(&bases.z_basis);
            let rank = match released_rank(&reduced) {
                Ok(rank) => rank,
                Err(reason) => return Ok(RailFaceLimitOutcome::FaceUnavailable { reason }),
            };
            if rank == 0 {
                continue;
            }
            reduced_specs.push(crate::estimate::PenaltySpec::Dense(reduced));
            reduced_null_dims.push(pinned - rank);
            reduced_rho.push(rho[j]);
        }
        let (reduced_penalties, _) = gam_terms::construction::canonicalize_penalty_specs(
            &reduced_specs,
            &reduced_null_dims,
            pinned,
            "laml rail-face limit fit",
        )?;
        if reduced_penalties.len() != reduced_rho.len() {
            // We filtered rank-0 blocks ourselves, so canonicalization must
            // keep every spec; a drop here means the two rank scans disagree
            // and the reduced ρ vector would be misaligned.
            return Ok(RailFaceLimitOutcome::FaceUnavailable {
                reason: "reduced-penalty canonicalization dropped a block the rank scan kept"
                    .to_string(),
            });
        }

        let mut pirls_config = self.config.as_pirls_config();
        pirls_config.link_kind = self.runtime_inverse_link();
        let problem = crate::pirls::PirlsProblem {
            x: reduced_design,
            offset: self.offset.view(),
            y: self.y,
            priorweights: self.weights,
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        };
        let penalty = crate::pirls::PenaltyConfig {
            canonical_penalties: &reduced_penalties,
            reparam_invariant: None,
            p: pinned,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
        };
        let reduced_rho = Array1::from(reduced_rho);
        let rho_view = gam_problem::LogSmoothingParamsView::new(reduced_rho.view())?;
        let limit_fit = match crate::pirls::fit_model_for_fixed_rho(
            rho_view,
            problem,
            penalty,
            &pirls_config,
            None,
        ) {
            Ok((result, _working)) => result,
            Err(err) => {
                // The λ=∞ model failing to solve is a statement about the
                // face (LIMIT_INNER_DIVERGED in the design's taxonomy), not
                // an error in the caller's fit: decline and leave the
                // measured-tail path its authority.
                return Ok(RailFaceLimitOutcome::FaceUnavailable {
                    reason: format!("the lambda=infinity limit fit did not solve: {err}"),
                });
            }
        };
        if !matches!(limit_fit.status, crate::pirls::PirlsStatus::Converged) {
            return Ok(RailFaceLimitOutcome::FaceUnavailable {
                reason: format!(
                    "the lambda=infinity limit fit did not converge: {:?}",
                    limit_fit.status
                ),
            });
        }
        if limit_fit.derivatives_unsupported {
            return Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "this family does not expose the third-derivative curvature array the \
                         LAML form's drift term needs"
                    .to_string(),
            });
        }
        // Both honest labels are inside the form. `ObservedExact` is the
        // exact Laplace curvature; `ExpectedInformationSurrogate` on these
        // families means either a canonical link (Observed ≡ Fisher, so the
        // "surrogate" IS exact) or a by-design Fisher family — and in every
        // case the criterion's `½log|H|` and this row bundle come from the
        // SAME exported state, so the form expands the criterion as shipped.
        // Only an indefinite observed Hessian is a real refusal: its logdet
        // is not trustworthy by the exporter's own diagnosis.
        if let crate::pirls::ExportedLaplaceCurvature::InvalidObservedCurvature { .. } =
            limit_fit.exported_laplace_curvature
        {
            return Ok(RailFaceLimitOutcome::FaceUnavailable {
                reason: format!(
                    "the limit fit's observed curvature is invalid ({:?}); its Laplace \
                     logdet is not trustworthy",
                    limit_fit.exported_laplace_curvature
                ),
            });
        }

        // Reduced coefficients back to the reduced-original basis, then to
        // the model basis: `β̂_∞ = Z · (Qs · β_transformed)`.
        let alpha = limit_fit
            .reparam_result
            .qs
            .dot(limit_fit.beta_transformed.as_ref());
        let limit_beta = bases.z_basis.dot(&alpha);
        // `∇ℓ = XᵀW_s(z − η)`: the score-side identity, on the limit fit's
        // own converged rows. Row order is the design's, so these arrays are
        // valid against the FULL design too — the reduced model differs only
        // in its coefficient basis.
        let n = design.nrows();
        let mut score_residuals = Array1::<f64>::zeros(n);
        for i in 0..n {
            score_residuals[i] = limit_fit.solveweights[i]
                * (limit_fit.solveworking_response[i] - limit_fit.final_eta[i]);
        }
        Ok(laml_rail_face_limit(
            design.view(),
            self.canonical_penalties.as_slice(),
            rho,
            face,
            LamlFaceParts {
                limit_beta,
                working_weights: limit_fit.finalweights.view(),
                score_residuals: score_residuals.view(),
                weight_eta_derivatives: limit_fit.solve_c_array.view(),
                convergence_tolerance: pirls_config.convergence_tolerance,
            },
        ))
    }
}

#[cfg(test)]
mod rail_face_limit_tests {
    use super::*;
    use crate::rho_optimizer::OuterEvalOrder;
    use crate::rho_optimizer::rail_face::{RailFaceLimit, RailFaceVerdict, certify_rail_face};

    /// Unwrap an outcome that the fixture guarantees is available, carrying
    /// the TYPED decline reason into the panic instead of discarding it.
    fn expect_available(outcome: RailFaceLimitOutcome, what: &str) -> RailFaceLimit {
        match outcome {
            RailFaceLimitOutcome::Available(limit) => *limit,
            other => panic!("{what}: expected an available limit, got {other:?}"),
        }
    }

    /// A cubic design `[1, t, t², t³]` whose truth is linear. Penalty 0 hits the
    /// quadratic+cubic columns (the block that can rail to λ=∞), penalty 1 hits
    /// the linear column (the survivor that stays finite).
    ///
    /// `curvature` sets how much genuine quadratic signal the data carries —
    /// the whole point of the face certificate is that it separates "the
    /// released directions are not worth their Occam cost" from "they are".
    /// The residual is a deterministic Nyquist alternation, which no cubic can
    /// chase, so the dispersion is real and `curvature` alone controls the
    /// score for releasing the face.
    /// Self-adjoint eigendecomposition of the small design gram, used only to
    /// project the fixture's residual out of the design's column space.
    fn design_gram_eigh(gram: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
        let n = gram.nrows();
        let mut sym = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sym[[i, j]] = 0.5 * (gram[[i, j]] + gram[[j, i]]);
            }
        }
        sym.eigh(Side::Lower)
            .expect("the cubic design gram is well posed")
    }

    fn cubic_fixture(curvature: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        cubic_fixture_sized(curvature, 96)
    }

    fn cubic_fixture_sized(curvature: f64, n: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut x = Array2::<f64>::zeros((n, 4));
        let mut raw_noise = Array1::<f64>::zeros(n);
        let mut signal = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            x[[i, 3]] = t * t * t;
            raw_noise[i] = if i % 2 == 0 { 0.35 } else { -0.35 };
            // NO true slope: a penalized survivor coordinate shrinks a real
            // slope, and that shrinkage bias is itself signal the released
            // directions can absorb — a genuine reason to come off the face,
            // and not the one this fixture is about.
            signal[i] = 0.8 + curvature * t * t;
        }
        // Project the deterministic residual out of the design's column space,
        // so the limit fit's score in the released directions is EXACTLY the
        // curvature the fixture asked for — the quantity the certificate weighs
        // against those directions' Occam cost.
        let gram = x.t().dot(&x);
        let rhs = x.t().dot(&raw_noise);
        let (values, vectors) = design_gram_eigh(&gram);
        let rotated = vectors.t().dot(&rhs);
        let mut scaled = Array1::<f64>::zeros(values.len());
        for i in 0..values.len() {
            scaled[i] = rotated[i] / values[i];
        }
        let residual = &raw_noise - &x.dot(&vectors.dot(&scaled));
        let y = &signal + &residual;
        (y, Array1::<f64>::ones(n), x)
    }

    /// A truth with no material curvature: releasing the bend block buys
    /// almost no fit against a real Occam cost, so `C ≻ 0` and λ=∞ is the
    /// optimum.
    const NULL_CURVATURE: f64 = 0.004;
    /// A truth the released directions genuinely explain: the score for
    /// releasing them beats their cost, `C` is indefinite, and the face must be
    /// refused.
    const SIGNAL_CURVATURE: f64 = 6.0;

    fn cubic_penalties() -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[2, 2]] = 1.0;
        bend[[3, 3]] = 2.0;
        let mut slope = Array2::<f64>::zeros((p, p));
        slope[[1, 1]] = 1.0;
        gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(bend),
                crate::estimate::PenaltySpec::Dense(slope),
            ],
            &[2, 3],
            p,
            "rail_face_limit_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the rail-face fixture penalties")
    }

    fn gaussian_config() -> RemlConfig {
        RemlConfig::external(
            GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            1e-10,
            false,
        )
    }

    /// THE VALIDATION. The analytic pencil constant is a prediction about the
    /// PRODUCTION criterion: on the tail, `∂V/∂ρ_j = −c_j·e^{−ρ_j}` with
    /// `c_j = ½tr((QᵀS_jQ)⁻¹QᵀCQ)` — a number built entirely from limit
    /// objects, with no probe, no finite difference, and no evaluation at a
    /// large λ anywhere in it.
    ///
    /// Measure the real `∂V/∂ρ_0` at a MODERATE `ρ_0 = 12` (deep enough that
    /// the `O(e^{−2ρ})` correction is ~1e-5 relative, shallow enough that the
    /// logdet pair has not started cancelling) and require the analytic
    /// constant to reproduce it. If the closed form had the wrong dispersion
    /// convention, a missing Schur term, or a sign error, this test would miss
    /// by orders of magnitude rather than by a rounding.
    /// One `(n, ρ_0)` row: the analytic constant, the production pencil
    /// `−e^{ρ_0}·∂V/∂ρ_0`, and the pencil's rounding band. The band is Wilkinson's
    /// `γ_k·Σ|terms|` with `k` the three `dim × dim` accumulations that form the
    /// gradient entry, scaled by the pencil's `e^{ρ_0}`. On the face the entry is
    /// what is left of the log-determinant pair `½tr(H⁻¹λS) → rank/2` against
    /// `½∂log|λS|₊ = rank/2` after they cancel, so the terms are that pair
    /// (`rank`) plus the audited parts (`fixed_beta`, `logdet_h`, `logdet_s` and
    /// the KKT remainder). The audited parts alone are already `O(e^{−ρ_0})`, and a
    /// band over them left an allowance of 3.0e-11 against a measured ratio
    /// deviation of 1.7e-6.
    fn measured_against_analytic(n: usize, rho_0: f64) -> (f64, f64, f64) {
        let (y, weights, x) = cubic_fixture_sized(NULL_CURVATURE, n);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the rail-face fixture state");

        let rho = Array1::from(vec![rho_0, 1.0]);
        let limit = state
            .rail_face_limit(&rho, &[0])
            .expect("the analytic face limit must not error")
            .available()
            .expect("a Gaussian dense fixture is inside the closed form");
        let analytic = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof.tail_constants[0],
            RailFaceVerdict::Refused { reason } => {
                panic!("the fixture face should certify at n={n}, rho_0={rho_0}: {reason}")
            }
        };
        crate::estimate::outer_eval_capture::enable_rho_outer_audit();
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        let audit = crate::estimate::outer_eval_capture::take_rho_outer_audit()
            .expect("the ρ-block audit was armed for this evaluation");
        let parts = audit
            .parts
            .first()
            .copied()
            .expect("the production gradient publishes its coordinate-0 parts");
        let kkt = parts.total - (parts.fixed_beta + parts.logdet_h + parts.logdet_s);
        let absolute_sum = parts.rank as f64
            + parts.fixed_beta.abs()
            + parts.logdet_h.abs()
            + parts.logdet_s.abs()
            + kkt.abs();
        let band = rho_0.exp()
            * gam_linalg::roundoff::accumulation_growth(3 * parts.dim * parts.dim)
            * absolute_sum;
        (analytic, -(rho_0.exp()) * eval.gradient[0], band)
    }

    /// The historical soft ρ-guard barrier, `w·log cosh(a·ρ)` with `w = 1e-6`
    /// and `a = 4/30`, which the criterion no longer carries (#2902 row 8).
    /// The tail test re-adds its gradient to the pencil as a positive control.
    const RETIRED_BARRIER_WEIGHT: f64 = 1.0e-6;
    const RETIRED_BARRIER_SHARPNESS: f64 = 4.0 / 30.0;

    /// On the tail the measured pencil approaches the analytic constant as
    /// `c − m(ρ_0) = C·e^{−ρ_0} + O(e^{−2ρ_0})`, so the signed gap contracts by
    /// exactly `e^{−Δ}` across a depth step `Δ`. The allowance is the two pencils'
    /// rounding bands relative to their gaps. A gap below its own band carries no
    /// sign or magnitude, so it is refused rather than compared. Returns the
    /// measured ratio.
    fn gap_contracts_like_the_tail(
        analytic: f64,
        (rho_shallow, measured_shallow, band_shallow): (f64, f64, f64),
        (rho_deep, measured_deep, band_deep): (f64, f64, f64),
    ) -> Result<f64, String> {
        let (gap_shallow, gap_deep) = (analytic - measured_shallow, analytic - measured_deep);
        if !(gap_shallow.abs() > band_shallow && gap_deep.abs() > band_deep) {
            return Err(format!(
                "a gap lies inside its rounding band: {gap_shallow:.4e} (band {band_shallow:.4e}) \
                 at rho_0={rho_shallow}, {gap_deep:.4e} (band {band_deep:.4e}) at rho_0={rho_deep}"
            ));
        }
        let ratio = gap_deep / gap_shallow;
        let expected = (rho_shallow - rho_deep).exp();
        let allowance = expected * (band_deep / gap_deep.abs() + band_shallow / gap_shallow.abs());
        if (ratio - expected).abs() <= allowance {
            Ok(ratio)
        } else {
            Err(format!(
                "the gap ratio {ratio:.6e} is not the tail contraction e^(-{:.1}) = {expected:.6e} \
                 within {allowance:.3e}: gaps {gap_shallow:.6e} at rho_0={rho_shallow} and \
                 {gap_deep:.6e} at rho_0={rho_deep}",
                rho_deep - rho_shallow
            ))
        }
    }

    #[test]
    fn analytic_face_constant_reproduces_the_production_rho_gradient() {
        let (analytic_shallow, measured_shallow, band_shallow) =
            measured_against_analytic(96, 10.0);
        let (analytic_deep, measured_deep, band_deep) = measured_against_analytic(96, 14.0);
        let (analytic_wide, measured_wide, _) = measured_against_analytic(384, 10.0);
        for (label, measured) in [
            ("n=96 rho_0=10", measured_shallow),
            ("n=96 rho_0=14", measured_deep),
            ("n=384 rho_0=10", measured_wide),
        ] {
            assert!(
                measured > 0.0,
                "{label}: the fixture must be on an upper tail, measured pencil {measured:.6e}"
            );
        }

        // Where the production gradient is still a trustworthy instrument, the
        // analytic constant reproduces it.
        let shallow = (analytic_shallow - measured_shallow).abs() / measured_shallow;
        let wide = (analytic_wide - measured_wide).abs() / measured_wide;
        assert!(
            shallow < 5.0e-3 && wide < 5.0e-3,
            "the analytic face constant must reproduce the production tail law: \
             n=96 analytic={analytic_shallow:.9e} measured={measured_shallow:.9e} \
             rel={shallow:.4e}; n=384 analytic={analytic_wide:.9e} \
             measured={measured_wide:.9e} rel={wide:.4e}"
        );

        // The analytic constant is a property of the LIMIT, so it does not
        // depend on where the coordinate happens to sit — bitwise, not to a
        // tolerance: nothing in it is evaluated at ρ.
        assert_eq!(
            analytic_shallow, analytic_deep,
            "the analytic pencil constant must not depend on rho at all"
        );

        // The criterion carries no soft ρ-guard barrier (#2902 row 8), so the
        // measured pencil at depth is the criterion's own tail law, and its gap to
        // the analytic constant is that law's truncated tail. Measured before the
        // restatement: 1.79e-4 relative at ρ₀ = 10 and 3.28e-6 at ρ₀ = 14, a ratio
        // of 54.6 = e⁴. The divergence this test used to assert was the barrier's
        // saturated gradient `w·a·tanh(a·ρ)` times `e^ρ`.
        let shallow_row = (10.0, measured_shallow, band_shallow);
        let deep_row = (14.0, measured_deep, band_deep);
        let contraction = gap_contracts_like_the_tail(analytic_shallow, shallow_row, deep_row);
        assert!(
            contraction.is_ok(),
            "the production pencil must approach the analytic face constant as the tail law \
             does: {contraction:?} (measured {measured_shallow:.9e} then {measured_deep:.9e} \
             against a constant analytic {analytic_shallow:.9e})"
        );

        // Positive control: the same pencils with the retired barrier's gradient
        // put back. Its term grows like `e^ρ·w·a` instead of contracting, so the
        // contraction check must refuse it.
        let with_barrier = |(rho_0, measured, band): (f64, f64, f64)| {
            let guard = RETIRED_BARRIER_WEIGHT
                * RETIRED_BARRIER_SHARPNESS
                * (RETIRED_BARRIER_SHARPNESS * rho_0).tanh();
            (rho_0, measured - rho_0.exp() * guard, band)
        };
        let barrier_contraction = gap_contracts_like_the_tail(
            analytic_shallow,
            with_barrier(shallow_row),
            with_barrier(deep_row),
        );
        assert!(
            barrier_contraction.is_err(),
            "the contraction check must refuse a pencil carrying the retired barrier, but it \
             accepted ratio {barrier_contraction:?}"
        );
    }

    /// The value gap is the criterion improvement still available by running
    /// the face to λ=∞. Its prediction is testable directly against production
    /// VALUES — no derivative, no cancellation: `V(ρ_0) − V(ρ_0 + Δ)` must
    /// equal `c·(e^{−ρ_0} − e^{−ρ_0−Δ})`.
    #[test]
    fn analytic_value_gap_predicts_the_measured_criterion_drop() {
        let (y, weights, x) = cubic_fixture(NULL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the rail-face fixture state");

        let near = Array1::from(vec![10.0, 1.0]);
        let far = Array1::from(vec![14.0, 1.0]);
        let limit = state
            .rail_face_limit(&near, &[0])
            .expect("the analytic face limit must not error")
            .available()
            .expect("a Gaussian dense fixture is inside the closed form");
        let near_gap = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };
        let far_limit = state
            .rail_face_limit(&far, &[0])
            .expect("the analytic face limit must not error")
            .available()
            .expect("a Gaussian dense fixture is inside the closed form");
        let far_gap = match certify_rail_face(&far_limit) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };

        let near_value = state.compute_cost(&near).expect("criterion at rho_0=10");
        let far_value = state.compute_cost(&far).expect("criterion at rho_0=14");
        let measured_drop = near_value - far_value;
        let predicted_drop = near_gap - far_gap;

        assert!(
            predicted_drop > 0.0,
            "running the face outward must lower the criterion; predicted {predicted_drop:.6e}"
        );
        let relative = (predicted_drop - measured_drop).abs() / measured_drop.abs();
        assert!(
            relative < 1.0e-2,
            "the analytic value gap must predict the measured criterion drop: \
             predicted={predicted_drop:.9e} measured={measured_drop:.9e} rel={relative:.3e}"
        );
    }

    /// THE REFUSAL IS ALSO A MEASUREMENT. Give the same design a truth the
    /// released directions genuinely explain. The face's pencil constant goes
    /// negative and the certificate refuses — and it is RIGHT to: the production criterion
    /// at the same point is still *rising* in `ρ_0`, so the descent runs INWARD
    /// and λ=∞ is not the optimum at all. A certificate that minted here would
    /// be shipping a fit the optimizer was still trying to leave.
    #[test]
    fn face_refuses_when_the_released_directions_earn_their_cost() {
        let (y, weights, x) = cubic_fixture(SIGNAL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the signal-bearing fixture state");

        let rho = Array1::from(vec![12.0, 1.0]);
        let limit = state
            .rail_face_limit(&rho, &[0])
            .expect("the analytic face limit must not error")
            .available()
            .expect("a Gaussian dense fixture is inside the closed form");
        match certify_rail_face(&limit) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("lowers the criterion"),
                "the refusal must name the descending coordinate: {reason}"
            ),
            RailFaceVerdict::Certified(proof) => panic!(
                "a face the data wants released must NOT certify; got λ_min(C)={:.3e}",
                proof.min_curvature
            ),
        }

        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        assert!(
            eval.gradient[0] > 0.0,
            "the refusal must agree with the criterion: dV/drho_0={:.6e} should be POSITIVE \
             (descent runs inward, away from the rail)",
            eval.gradient[0]
        );
    }

    // ═════════════ multi-coordinate faces (#2348 Inc 5 build-out) ═════════════

    /// The cubic fixture's bend block split into its two eigen-directions, so a
    /// face can rail them as TWO ρ coordinates instead of one, plus the same
    /// finite slope survivor.
    ///
    /// `S_0 + S_1` is the single `bend` of [`cubic_penalties`] exactly
    /// (canonicalization stores `RᵀR`, which reproduces the block on its range
    /// without rescaling), and `M_p = p − Σ_j rank(S_j)` is 4 − 3 in both
    /// layouts — three rank-1 blocks against one rank-2 plus one rank-1. So at
    /// `ρ_0 = ρ_1` the two layouts are the SAME model with the same `S_λ` and
    /// the same criterion, which is what makes the cross-layout comparison an
    /// identity rather than an approximation.
    fn split_bend_penalties() -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut quadratic = Array2::<f64>::zeros((p, p));
        quadratic[[2, 2]] = 1.0;
        let mut cubic = Array2::<f64>::zeros((p, p));
        cubic[[3, 3]] = 2.0;
        let mut slope = Array2::<f64>::zeros((p, p));
        slope[[1, 1]] = 1.0;
        gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(quadratic),
                crate::estimate::PenaltySpec::Dense(cubic),
                crate::estimate::PenaltySpec::Dense(slope),
            ],
            &[3, 3, 3],
            p,
            "rail_face_limit_split_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the split-bend fixture penalties")
    }

    /// The same released subspace reached by a face whose two coordinates
    /// OVERLAP: `S_0` is the full rank-2 bend and `S_1` is a rank-1 penalty
    /// along `(e₂+e₃)/√2`, a direction `S_0` already penalizes. This is the
    /// coalesced geometry the face form exists for — the case a per-coordinate
    /// marginal law cannot describe, because coordinate 1 releases nothing once
    /// coordinate 0 is at `λ = ∞`.
    fn overlapping_face_penalties() -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[2, 2]] = 1.0;
        bend[[3, 3]] = 2.0;
        let mut coalesced = Array2::<f64>::zeros((p, p));
        for (i, j) in [(2, 2), (2, 3), (3, 2), (3, 3)] {
            coalesced[[i, j]] = 0.5;
        }
        let mut slope = Array2::<f64>::zeros((p, p));
        slope[[1, 1]] = 1.0;
        gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(bend),
                crate::estimate::PenaltySpec::Dense(coalesced),
                crate::estimate::PenaltySpec::Dense(slope),
            ],
            &[2, 3, 3],
            p,
            "rail_face_limit_overlap_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the overlapping-face fixture penalties")
    }

    /// THE MULTI-COORDINATE JOINT LAW, against production.
    ///
    /// Release the whole face along its diagonal: with every face `λ_j = e^ρ`,
    /// `V(ρ) = V_∞ + e^{−ρ}·c_joint` exactly, so
    /// `Σ_{j∈F} ∂V/∂ρ_j = −c_joint·e^{−ρ}`. That is the `|F| > 1` analogue of
    /// the single-coordinate pencil law, and it holds whether or not the face
    /// penalties overlap, because it differentiates the one form `C` along the
    /// ray rather than assembling marginals.
    ///
    /// This is the first test in this module to build a face limit at all at
    /// `|F| > 1`. The proof module has always accepted a multi-coordinate face
    /// and has unit tests on hand-built `RailFaceLimit` values; every limit
    /// this module built for it, however, was built for `&[0]` — so the
    /// production assembly at `|F| > 1` (the Q/Z split of a multi-penalty face,
    /// the per-coordinate released blocks, the rank bookkeeping against a
    /// survivor) was covered only by the run-loop, which passes whatever the
    /// active box face is.
    #[test]
    fn two_coordinate_face_joint_law_reproduces_the_production_gradient() {
        let (y, weights, x) = cubic_fixture(NULL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            split_bend_penalties(),
            4,
            &config,
            Some(vec![3, 3, 3]),
            None,
            None,
        )
        .expect("build the split-bend fixture state");

        let rho_face = 10.0_f64;
        let rho = Array1::from(vec![rho_face, rho_face, 1.0]);
        let limit = expect_available(
            state
                .rail_face_limit(&rho, &[0, 1])
                .expect("the analytic face limit must not error"),
            "two-coordinate face",
        );
        assert_eq!(
            limit.face,
            vec![0, 1],
            "the limit must carry BOTH railed coordinates, not collapse to one"
        );
        assert_eq!(
            limit.released_penalties.len(),
            2,
            "each face coordinate needs its own released block"
        );

        let proof = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof,
            RailFaceVerdict::Refused { reason } => {
                panic!("the two-coordinate face should certify on a null truth: {reason}")
            }
        };
        assert_eq!(
            proof.coordinate_kinds,
            vec![
                crate::rho_optimizer::rail_face::FaceCoordinateKind::StrictOutward,
                crate::rho_optimizer::rail_face::FaceCoordinateKind::StrictOutward
            ],
            "disjoint face penalties each release a direction of their own"
        );

        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        let measured_joint = -(rho_face.exp()) * (eval.gradient[0] + eval.gradient[1]);
        assert!(
            measured_joint > 0.0,
            "the fixture must sit on an upper tail in both face coordinates; \
             measured joint pencil {measured_joint:.6e}"
        );
        let joint_rel = (proof.joint_tail_constant - measured_joint).abs() / measured_joint;
        assert!(
            joint_rel < 5.0e-3,
            "the analytic JOINT constant must reproduce the production gradient summed \
             over the face: analytic={:.9e} measured={measured_joint:.9e} rel={joint_rel:.4e}",
            proof.joint_tail_constant
        );

        // The per-coordinate law, one coordinate at a time. These face
        // penalties have orthogonal ranges, so releasing j alone and releasing
        // it alongside the rest agree, and each `c_j` is separately measurable
        // against its own production partial.
        for (idx, &measured) in [
            -(rho_face.exp()) * eval.gradient[0],
            -(rho_face.exp()) * eval.gradient[1],
        ]
        .iter()
        .enumerate()
        {
            let rel = (proof.tail_constants[idx] - measured).abs() / measured;
            assert!(
                rel < 5.0e-3,
                "face coordinate {idx}: analytic c={:.9e} against measured {measured:.9e} \
                 (rel {rel:.4e})",
                proof.tail_constants[idx]
            );
        }

        // With disjoint released blocks the joint constant is the sum of the
        // marginals — an internal identity of the assembly, not a fixture
        // accident: `½tr((ΣA)⁻¹C)` splits when the `A_j` have orthogonal
        // ranges. It is asserted here because it FAILS for the overlapping
        // face below, which is what makes that case the interesting one.
        let additive: f64 = proof.tail_constants.iter().sum();
        let additive_rel = (additive - proof.joint_tail_constant).abs() / proof.joint_tail_constant;
        assert!(
            additive_rel < 1.0e-10,
            "disjoint face blocks must make the joint constant additive: \
             Σc_j={additive:.12e} joint={:.12e} rel={additive_rel:.3e}",
            proof.joint_tail_constant
        );
    }

    /// ONE GEOMETRY, TWO PENALTY LAYOUTS, ONE CONSTANT.
    ///
    /// The rank-2 bend railed as a SINGLE coordinate and the same bend split
    /// into two rank-1 coordinates railed TOGETHER at equal `ρ` are the same
    /// `S_λ`, the same `M_p`, and therefore the same criterion — so the face
    /// they describe is the same face and the analytic constant must be the
    /// same number. Nothing here is measured against production: this is an
    /// identity between two paths through the assembly, and it fails if the
    /// `|F| > 1` path double-counts a penalty, builds the wrong `Q`, or gets
    /// the released/surviving rank split wrong — none of which a
    /// single-coordinate test can see.
    #[test]
    fn a_face_split_into_two_coordinates_is_the_same_face() {
        let (y, weights, x) = cubic_fixture(NULL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let rho_face = 10.0_f64;

        let single_state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the single-coordinate fixture state");
        let single = expect_available(
            single_state
                .rail_face_limit(&Array1::from(vec![rho_face, 1.0]), &[0])
                .expect("the analytic face limit must not error"),
            "single-coordinate face",
        );
        let single_proof = match certify_rail_face(&single) {
            RailFaceVerdict::Certified(proof) => proof,
            RailFaceVerdict::Refused { reason } => panic!("single face should certify: {reason}"),
        };

        let split_state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            split_bend_penalties(),
            4,
            &config,
            Some(vec![3, 3, 3]),
            None,
            None,
        )
        .expect("build the split-bend fixture state");
        let split = expect_available(
            split_state
                .rail_face_limit(&Array1::from(vec![rho_face, rho_face, 1.0]), &[0, 1])
                .expect("the analytic face limit must not error"),
            "two-coordinate face",
        );
        let split_proof = match certify_rail_face(&split) {
            RailFaceVerdict::Certified(proof) => proof,
            RailFaceVerdict::Refused { reason } => panic!("split face should certify: {reason}"),
        };

        // The two layouts release the same subspace, so the forms are the same
        // size and carry the same spectrum extremes.
        assert_eq!(
            single.first_order_form.nrows(),
            split.first_order_form.nrows(),
            "the two layouts must release the same subspace"
        );
        let constant_rel = (split_proof.joint_tail_constant - single_proof.joint_tail_constant)
            .abs()
            / single_proof.joint_tail_constant;
        assert!(
            constant_rel < 1.0e-9,
            "one geometry must give one constant: single-coordinate={:.15e} \
             two-coordinate joint={:.15e} rel={constant_rel:.3e}",
            single_proof.joint_tail_constant,
            split_proof.joint_tail_constant
        );

        // The pricing of the shipped point is the same statement, and it is
        // what the certificate reports to the fit — so it has to agree too.
        let gap_rel =
            (split_proof.value_gap - single_proof.value_gap).abs() / single_proof.value_gap;
        assert!(
            gap_rel < 1.0e-9,
            "the value gap must not depend on how the face is coordinatized: \
             single={:.15e} split={:.15e} rel={gap_rel:.3e}",
            single_proof.value_gap,
            split_proof.value_gap
        );
        let travel_rel = (split_proof.estimand_travel - single_proof.estimand_travel).abs()
            / single_proof.estimand_travel;
        assert!(
            travel_rel < 1.0e-9,
            "the estimand travel must not depend on the coordinatization: \
             single={:.15e} split={:.15e} rel={travel_rel:.3e}",
            single_proof.estimand_travel,
            split_proof.estimand_travel
        );
    }

    /// AN OVERLAPPING FACE COORDINATE IS UNIDENTIFIED, AND MUST NOT BLOCK THE
    /// FACE.
    ///
    /// Coordinate 1's penalty acts only along a direction coordinate 0 already
    /// pins at `λ = ∞`, so releasing it alone frees nothing: `V` is exactly
    /// independent of `λ_1` there. The certificate has to type that as
    /// [`FaceCoordinateKind::Unidentified`] with `c_1 = 0` and still certify —
    /// demanding a strict outward derivative from a coordinate that has none
    /// would refuse a face that is genuinely optimal.
    ///
    /// This is the coalesced/overlapping-penalty geometry (#2349's multinomial
    /// family, the all-on Hilbert scale) reaching the production assembly
    /// rather than a hand-built form, and the joint law is the ONLY law that
    /// describes it: the marginals do not add up here, which the test asserts.
    #[test]
    fn an_overlapping_face_coordinate_is_unidentified_and_still_certifies() {
        let (y, weights, x) = cubic_fixture(NULL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            overlapping_face_penalties(),
            4,
            &config,
            Some(vec![2, 3, 3]),
            None,
            None,
        )
        .expect("build the overlapping-face fixture state");

        let rho_face = 10.0_f64;
        let rho = Array1::from(vec![rho_face, rho_face, 1.0]);
        let limit = expect_available(
            state
                .rail_face_limit(&rho, &[0, 1])
                .expect("the analytic face limit must not error"),
            "overlapping two-coordinate face",
        );
        let proof = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof,
            RailFaceVerdict::Refused { reason } => panic!(
                "an unidentified face coordinate must not block a face that is otherwise \
                 proven: {reason}"
            ),
        };

        use crate::rho_optimizer::rail_face::FaceCoordinateKind;
        assert_eq!(
            proof.coordinate_kinds,
            vec![
                FaceCoordinateKind::StrictOutward,
                FaceCoordinateKind::Unidentified
            ],
            "the coalesced coordinate releases nothing once the bend is at lambda=infinity"
        );
        assert_eq!(
            proof.tail_constants[1], 0.0,
            "an unidentified coordinate carries no outward derivative at all"
        );
        assert!(
            proof.tail_constants[0] > 0.0,
            "the identified coordinate must still carry its own strict constant"
        );

        // The joint law still holds — and it is not the sum of the marginals
        // here, which is exactly why a face certificate cannot be assembled
        // coordinate-wise.
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        let measured_joint = -(rho_face.exp()) * (eval.gradient[0] + eval.gradient[1]);
        let joint_rel = (proof.joint_tail_constant - measured_joint).abs() / measured_joint;
        assert!(
            joint_rel < 5.0e-3,
            "the joint law must hold on an OVERLAPPING face: analytic={:.9e} \
             measured={measured_joint:.9e} rel={joint_rel:.4e}",
            proof.joint_tail_constant
        );
        let marginal_sum: f64 = proof.tail_constants.iter().sum();
        assert!(
            (marginal_sum - proof.joint_tail_constant).abs() / proof.joint_tail_constant > 1.0e-3,
            "the overlapping face must NOT be describable by its marginals: \
             Σc_j={marginal_sum:.9e} against joint {:.9e} — if these agree the fixture has \
             stopped exercising the coalesced geometry",
            proof.joint_tail_constant
        );
    }

    /// THE PROFILED SCALE ON AN OVERLAPPING FACE USES THE JOINT PENALTY RANK.
    ///
    /// The criterion's `M_p = p − rank(Σ_k S_k)` is the joint structural rank
    /// its penalty pseudo-logdet reports. On the overlapping fixture the
    /// per-penalty ranks sum to `2 + 1 + 1 = 4 = p` while the joint rank is 3
    /// (the coalesced penalty lives inside the bend's range), so a Σ-rank
    /// bookkeeping gives `M_p = 0` against the criterion's 1, and the limit's
    /// profiled scale `D_p/(n − M_p)` is off by the factor `(n−1)/n`.
    ///
    /// The overlap test above cannot see that: at `NULL_CURVATURE` the fit term
    /// `g_Qg_Qᵀ/φ̂` of `C` is negligible, so `φ̂` barely enters. Here the truth
    /// carries enough curvature that the fit term is a sizable share of the joint
    /// constant, and `n` is small enough that `(n−1)/n` moves that share well
    /// outside the pencil's tail correction. The joint pencil
    /// `−e^ρ(∂V/∂ρ_0 + ∂V/∂ρ_1)` from the production gradient is then the
    /// arbiter of which `M_p` the criterion uses.
    ///
    /// The same truth also makes the JOINT release descend: the joint law
    /// `½tr((A_0 + A_1)⁻¹C)` is negative although no single coordinate's own
    /// law is, so the face is refuted on the release simplex — and the
    /// production gradient's joint pencil agrees in sign and size.
    #[test]
    fn overlapping_face_profiled_scale_uses_the_joint_penalty_rank() {
        const N: usize = 12;
        const CURVATURE: f64 = 0.5;
        let (y, weights, x) = cubic_fixture_sized(CURVATURE, N);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            overlapping_face_penalties(),
            4,
            &config,
            Some(vec![2, 3, 3]),
            None,
            None,
        )
        .expect("build the overlapping-face fixture state");

        let rho_face = 12.0_f64;
        let rho = Array1::from(vec![rho_face, rho_face, 1.0]);
        let limit = expect_available(
            state
                .rail_face_limit(&rho, &[0, 1])
                .expect("the analytic face limit must not error"),
            "overlapping two-coordinate face at small n",
        );
        match certify_rail_face(&limit) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("ranges overlap") && reason.contains("lowers the criterion"),
                "the jointly descending overlapping face must be refuted on its simplex: {reason}"
            ),
            RailFaceVerdict::Certified(proof) => panic!(
                "a face whose joint release descends must NOT certify; got {:?} statistic {:.3e}",
                proof.route, proof.statistic
            ),
        }

        // The joint law `c_joint = ½tr((A_0 + A_1)⁻¹C)`, and the fit term's
        // share of it, `½g_Qᵀ(ΣA_j)⁻¹g_Q/φ̂`: the lever the scale
        // mis-statement acts through.
        let joint_released = &limit.released_penalties[0] + &limit.released_penalties[1];
        let (values, vectors) = design_gram_eigh(&joint_released);
        let joint_tail_constant: f64 = 0.5
            * (0..values.len())
                .map(|a| {
                    let u = vectors.column(a);
                    u.dot(&limit.first_order_form.dot(&u)) / values[a]
                })
                .sum::<f64>();
        assert!(
            joint_tail_constant < 0.0,
            "the fixture's joint release must descend: c_joint={joint_tail_constant:.6e}"
        );
        let rotated_score = vectors.t().dot(&limit.released_score);
        let fit_term: f64 = 0.5
            * rotated_score
                .iter()
                .zip(values.iter())
                .map(|(g, sigma)| g * g / sigma)
                .sum::<f64>()
            / limit.limit_dispersion;
        let fit_share = (fit_term / joint_tail_constant).abs();
        let scale_shift = fit_share / (N as f64 - 1.0);
        assert!(
            scale_shift > 1.0e-2,
            "the fixture must make the M_p bookkeeping visible: the fit term is \
             {fit_share:.4} of the joint constant, so an (n−1)/n scale error moves it by \
             only {scale_shift:.3e}"
        );

        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        let measured_joint = -(rho_face.exp()) * (eval.gradient[0] + eval.gradient[1]);
        let joint_rel = (joint_tail_constant - measured_joint).abs() / measured_joint.abs();
        assert!(
            joint_rel < 0.1 * scale_shift,
            "the face limit's profiled scale must use the criterion's joint penalty rank: \
             analytic={:.9e} measured={measured_joint:.9e} rel={joint_rel:.4e} against the \
             {scale_shift:.3e} a Σ-rank M_p would cost",
            joint_tail_constant
        );
    }

    // ═══════════════════ the LAML closed form (#2349) ═══════════════════

    /// Binomial sibling of the cubic fixture: logistic truth `η = a + b·t +
    /// c·t²`, responses produced by error-diffusion dithering, and — unlike
    /// the Gaussian fixture — NO survivor penalty. Three deliberate choices:
    ///
    /// * **Error diffusion, not thresholding.** A quasi-random threshold
    ///   leaves CLT-scale `O(√n)` noise in the released-direction score, and
    ///   the score test genuinely rejects the face on it (measured:
    ///   λ_min(C) = −8.0 against ‖C‖ = 8.0 on the thresholded ancestor of
    ///   this fixture). The diffusion carry keeps every prefix sum of
    ///   `y − μ` bounded by one, so the realized noise carries `O(1)` score
    ///   in ANY smooth direction while the information grows like `n` — the
    ///   binomial analogue of projecting the residual out of the design's
    ///   column space.
    /// * **No survivor penalty.** With nothing shrinking it, a real slope
    ///   creates no shrinkage bias for the released directions to absorb —
    ///   the trap the Gaussian fixture avoids by having no slope at all.
    /// * **A real slope, so `μ` crosses ½.** The drift array
    ///   `c_i = μ(1−μ)(1−2μ)` then CHANGES SIGN mid-domain. That texture is
    ///   what the K-oblique reduction cannot absorb into the pinned
    ///   directions, making the rank-2 drift term a measurable fraction of
    ///   the face constant — which the discriminator arm of the validation
    ///   requires. (A near-constant `μ` gives a near-smooth `c⊙a`, whose
    ///   drift is almost entirely oblique shadow: measured at 0.14% of the
    ///   constant, below the instrument's own 0.1% floor.)
    fn binomial_cubic_fixture(curvature: f64, n: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut x = Array2::<f64>::zeros((n, 4));
        let mut y = Array1::<f64>::zeros(n);
        let mut carry = 0.0_f64;
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            x[[i, 3]] = t * t * t;
            let eta = -0.9 + 1.8 * t + curvature * t * t;
            let mu = 1.0 / (1.0 + (-eta).exp());
            carry += mu;
            if carry >= 0.5 {
                y[i] = 1.0;
                carry -= 1.0;
            } else {
                y[i] = 0.0;
            }
        }
        (y, Array1::<f64>::ones(n), x)
    }

    /// The bend block alone: the face penalty with no survivor, so the limit
    /// model is the unpenalized `[1, t]` logistic fit.
    fn bend_only_penalty() -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[2, 2]] = 1.0;
        bend[[3, 3]] = 2.0;
        gam_terms::construction::canonicalize_penalty_specs(
            &[crate::estimate::PenaltySpec::Dense(bend)],
            &[2],
            p,
            "laml_rail_face_bend_only",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the bend-only penalty")
    }

    fn binomial_config(firth: bool) -> RemlConfig {
        RemlConfig::external(
            GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            )),
            1e-10,
            firth,
        )
    }

    fn binomial_state<'a>(
        y: &'a Array1<f64>,
        weights: &'a Array1<f64>,
        x: &Array2<f64>,
        offset: &Array1<f64>,
        config: &'a RemlConfig,
    ) -> RemlState<'a> {
        RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            bend_only_penalty(),
            4,
            config,
            Some(vec![2]),
            None,
            None,
        )
        .expect("build the binomial rail-face fixture state")
    }

    /// `½·tr(A⁻¹C)` for the single-coordinate face with its 2×2 released
    /// block, in closed form — the raw first-order constant of the expansion
    /// `∂V/∂ρ → −c₀·e^{−ρ}`, which holds whether or not `C ≻ 0` (the sign of
    /// `C` decides OPTIMALITY, not the validity of the expansion). Tied to
    /// production's own computation by the certified null arm, where it must
    /// agree with `proof.tail_constants[0]`.
    fn face_constant_from(limit: &RailFaceLimit) -> f64 {
        let a = &limit.released_penalties[0];
        let c = &limit.first_order_form;
        assert_eq!(
            a.nrows(),
            2,
            "the bend face releases exactly two directions"
        );
        let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
        0.5 * (a[[1, 1]] * c[[0, 0]] - a[[0, 1]] * c[[1, 0]] - a[[1, 0]] * c[[0, 1]]
            + a[[0, 0]] * c[[1, 1]])
            / det
    }

    /// The same limit with the rank-2 drift term subtracted — the Gaussian
    /// form misapplied to a moving-weight criterion. The limit records `d̃_Q`
    /// precisely so this subtraction is exact.
    fn strip_drift(limit: &RailFaceLimit) -> RailFaceLimit {
        let mut stripped = limit.clone();
        let drift = stripped
            .released_curvature_drift
            .take()
            .expect("the LAML form records its curvature drift");
        for a in 0..stripped.first_order_form.nrows() {
            for b in 0..stripped.first_order_form.ncols() {
                stripped.first_order_form[[a, b]] -=
                    stripped.released_score[a] * drift[b] + drift[a] * stripped.released_score[b];
            }
        }
        stripped
    }

    /// One `(n, ρ_0)` row of the LAML validation: (analytic with drift,
    /// analytic with the rank-2 drift term subtracted, measured pencil).
    fn laml_measured_against_analytic(n: usize, rho_0: f64) -> (f64, f64, f64) {
        let (y, weights, x) = binomial_cubic_fixture(0.02, n);
        let offset = Array1::<f64>::zeros(y.len());
        let config = binomial_config(false);
        let state = binomial_state(&y, &weights, &x, &offset, &config);

        let rho = Array1::from(vec![rho_0]);
        let limit = state
            .rail_face_limit(&rho, &[0])
            .expect("the analytic face limit must not error");
        let limit = expect_available(limit, "binomial LAML fixture");
        let analytic = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof.tail_constants[0],
            RailFaceVerdict::Refused { reason } => {
                panic!("the LAML fixture face should certify at n={n}, rho_0={rho_0}: {reason}")
            }
        };
        // The closed-form helper must agree with production's own
        // per-coordinate computation wherever the face certifies.
        let direct = face_constant_from(&limit);
        assert!(
            (direct - analytic).abs() <= 1.0e-10 * analytic.abs(),
            "face_constant_from must reproduce the certified constant: {direct} vs {analytic}"
        );
        let analytic_no_drift = face_constant_from(&strip_drift(&limit));
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production LAML gradient must evaluate");
        (
            analytic,
            analytic_no_drift,
            -(rho_0.exp()) * eval.gradient[0],
        )
    }

    /// The LAML face constant against the production LAML ρ-gradient — the
    /// same validation the Gaussian form landed with, on a criterion whose
    /// working weights genuinely move with `β̂`.
    #[test]
    fn laml_face_constant_reproduces_the_production_rho_gradient() {
        let (analytic_shallow, no_drift_shallow, measured_shallow) =
            laml_measured_against_analytic(192, 9.0);
        let (analytic_deep, _, _) = laml_measured_against_analytic(192, 12.0);
        let (analytic_wide, _, measured_wide) = laml_measured_against_analytic(384, 9.0);

        assert!(
            measured_shallow > 0.0 && measured_wide > 0.0,
            "the fixture must be on an upper tail: measured {measured_shallow:.6e} / {measured_wide:.6e}"
        );

        let shallow = (analytic_shallow - measured_shallow).abs() / measured_shallow;
        let wide = (analytic_wide - measured_wide).abs() / measured_wide;
        assert!(
            shallow < 5.0e-3 && wide < 5.0e-3,
            "the LAML face constant must reproduce the production tail law: \
             n=192 analytic={analytic_shallow:.9e} measured={measured_shallow:.9e} \
             rel={shallow:.4e}; n=384 analytic={analytic_wide:.9e} \
             measured={measured_wide:.9e} rel={wide:.4e}"
        );

        // The constant is a property of the limit: bitwise ρ-independent.
        assert_eq!(
            analytic_shallow, analytic_deep,
            "the analytic pencil constant must not depend on rho at all"
        );

        // On a comfortably-certifying null fixture the rank-2 drift term is
        // structurally SMALL — it is `∝ g_Q`, and certification requires a
        // small `g_Q`. Assert only that including it does not hurt; the
        // load-bearing discrimination lives on the signal fixture below,
        // where `g_Q` is large and the expansion is still falsifiable.
        let no_drift_miss = (no_drift_shallow - measured_shallow).abs() / measured_shallow;
        assert!(
            shallow < no_drift_miss + 5.0e-3,
            "the drift term must not degrade the null-arm match: with drift \
             rel={shallow:.4e}, without rel={no_drift_miss:.4e}"
        );
    }

    /// THE DRIFT TERM IS LOAD-BEARING — proven where it is visible. The
    /// rank-2 term is `∝ g_Q`, so a comfortably-certifying face (small
    /// `g_Q` by construction) cannot discriminate it from zero. The
    /// curvature-bearing fixture can: `g_Q` is large there, and the
    /// first-order law `∂V/∂ρ_0 → −c₀·e^{−ρ_0}` is valid — and measurable
    /// against the production gradient — whether or not the face is optimal.
    /// The certificate rightly refuses to MINT there; the expansion is still
    /// a falsifiable prediction, and stripping the drift term must break it.
    #[test]
    fn laml_drift_term_is_load_bearing_on_the_curved_fixture() {
        let (y, weights, x) = binomial_cubic_fixture(5.0, 192);
        let offset = Array1::<f64>::zeros(y.len());
        let config = binomial_config(false);
        let state = binomial_state(&y, &weights, &x, &offset, &config);

        let rho_0 = 9.0_f64;
        let rho = Array1::from(vec![rho_0]);
        let limit = expect_available(
            state
                .rail_face_limit(&rho, &[0])
                .expect("the analytic face limit must not error"),
            "curved binomial LAML fixture",
        );
        let c_with = face_constant_from(&limit);
        let c_without = face_constant_from(&strip_drift(&limit));
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production LAML gradient must evaluate");
        let measured = -(rho_0.exp()) * eval.gradient[0];
        assert!(
            measured < 0.0,
            "on the curved fixture the descent runs INWARD, so the pencil is negative: {measured:.6e}"
        );
        let with_rel = ((c_with - measured) / measured).abs();
        let without_rel = ((c_without - measured) / measured).abs();
        assert!(
            without_rel > 3.0 * with_rel,
            "stripping the curvature-drift term must visibly break the match where g_Q \
             is large: with drift rel={with_rel:.4e}, without rel={without_rel:.4e} \
             (analytic {c_with:.9e} vs stripped {c_without:.9e} vs measured {measured:.9e})"
        );
        // The bound is looser than the null arm's 5e-3 for a reason worth
        // keeping on record: on the curved fixture the constant sits near the
        // form's own sign change (information ≈ score), and nothing carries
        // relative precision across its own root — the ABSOLUTE miss here
        // (9.0e-4) is the same instrument floor the null arm shows at 3.5e-3
        // relative on a constant four times larger.
        assert!(
            with_rel < 2.5e-2,
            "the full LAML constant must reproduce the production pencil on the curved \
             fixture: analytic={c_with:.9e} measured={measured:.9e} rel={with_rel:.4e}"
        );
    }

    /// Value-domain falsification for the LAML form: the predicted remaining
    /// improvement from running the face out must match the measured
    /// criterion drop between two depths. No derivative, no cancellation.
    #[test]
    fn laml_value_gap_predicts_the_measured_criterion_drop() {
        let (y, weights, x) = binomial_cubic_fixture(0.02, 192);
        let offset = Array1::<f64>::zeros(y.len());
        let config = binomial_config(false);
        let state = binomial_state(&y, &weights, &x, &offset, &config);

        let near = Array1::from(vec![9.0]);
        let far = Array1::from(vec![13.0]);
        let near_gap = match certify_rail_face(&expect_available(
            state.rail_face_limit(&near, &[0]).expect("no error"),
            "binomial LAML near depth",
        )) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };
        let far_gap = match certify_rail_face(&expect_available(
            state.rail_face_limit(&far, &[0]).expect("no error"),
            "binomial LAML far depth",
        )) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };

        let near_value = state.compute_cost(&near).expect("criterion at rho_0=9");
        let far_value = state.compute_cost(&far).expect("criterion at rho_0=13");
        let measured_drop = near_value - far_value;
        let predicted_drop = near_gap - far_gap;

        assert!(
            predicted_drop > 0.0,
            "running the face outward must lower the criterion; predicted {predicted_drop:.6e}"
        );
        let relative = (predicted_drop - measured_drop).abs() / measured_drop.abs();
        assert!(
            relative < 2.0e-2,
            "the LAML value gap must predict the measured criterion drop: \
             predicted={predicted_drop:.9e} measured={measured_drop:.9e} rel={relative:.3e}"
        );
    }

    /// A binomial truth with real curvature: the released directions earn
    /// their Occam cost, the face's pencil constant goes negative, the
    /// certificate refuses —
    /// and the production criterion agrees (its gradient still runs inward).
    #[test]
    fn laml_face_refuses_when_the_released_directions_earn_their_cost() {
        let (y, weights, x) = binomial_cubic_fixture(5.0, 192);
        let offset = Array1::<f64>::zeros(y.len());
        let config = binomial_config(false);
        let state = binomial_state(&y, &weights, &x, &offset, &config);

        let rho = Array1::from(vec![11.0]);
        let limit = expect_available(
            state
                .rail_face_limit(&rho, &[0])
                .expect("the analytic face limit must not error"),
            "binomial LAML signal fixture",
        );
        match certify_rail_face(&limit) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("lowers the criterion"),
                "the refusal must name the descending coordinate: {reason}"
            ),
            RailFaceVerdict::Certified(proof) => panic!(
                "a face the data wants released must NOT certify; got λ_min(C)={:.3e}",
                proof.min_curvature
            ),
        }
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production LAML gradient must evaluate");
        assert!(
            eval.gradient[0] > 0.0,
            "the refusal must agree with the criterion: dV/drho_0={:.6e} should be POSITIVE",
            eval.gradient[0]
        );
    }

    /// The gates are gates: a family whose dispersion is estimated is outside
    /// the LAML form, and an armed Firth term changes the criterion — both
    /// must decline BY NAME rather than return a wrong limit.
    #[test]
    fn criteria_outside_both_closed_forms_decline_by_name() {
        // Gamma: the dispersion enters the criterion.
        let (y_raw, weights, x) = cubic_fixture(NULL_CURVATURE);
        let y: Array1<f64> = y_raw.mapv(|v| v.abs() + 0.5);
        let offset = Array1::<f64>::zeros(y.len());
        let config = RemlConfig::external(
            GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            )),
            1e-10,
            false,
        );
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the gamma fixture state");
        let rho = Array1::from(vec![12.0, 1.0]);
        match state
            .rail_face_limit(&rho, &[0])
            .expect("the gate must decline, not error")
        {
            RailFaceLimitOutcome::OutsideClosedForm { reason } => assert!(
                reason.contains("dispersion"),
                "the decline must name the clause that failed: {reason}"
            ),
            other => panic!("a gamma fit is outside both closed forms, got {other:?}"),
        }

        // Firth-armed binomial: the Jeffreys logdet is part of the criterion.
        let (yb, wb, xb) = binomial_cubic_fixture(0.02, 96);
        let offset_b = Array1::<f64>::zeros(yb.len());
        let config_b = binomial_config(true);
        let state_b = binomial_state(&yb, &wb, &xb, &offset_b, &config_b);
        let rho_b = Array1::from(vec![12.0]);
        match state_b
            .rail_face_limit(&rho_b, &[0])
            .expect("the gate must decline, not error")
        {
            RailFaceLimitOutcome::OutsideClosedForm { reason } => assert!(
                reason.contains("Firth"),
                "the decline must name the armed Jeffreys term: {reason}"
            ),
            other => panic!("a Firth-armed criterion is outside the LAML form, got {other:?}"),
        }
    }

    // ── the λ → 0 end (#2348 Inc 5, lower face) ─────────────────────────

    /// A COVERED zero-smoothing geometry: the face is the bend block on the
    /// quadratic+cubic columns, and the survivor is a ridge on the slope,
    /// quadratic and cubic columns, so `range(S_bend) ⊆ range(S_ridge)` and
    /// `λ_bend → 0` changes no rank.
    fn covered_penalties(bend_cols: (usize, usize)) -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[bend_cols.0, bend_cols.0]] = 1.0;
        bend[[bend_cols.1, bend_cols.1]] = 2.0;
        let mut ridge = Array2::<f64>::zeros((p, p));
        for c in 1..p {
            ridge[[c, c]] = 1.0;
        }
        gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(bend),
                crate::estimate::PenaltySpec::Dense(ridge),
            ],
            &[2, 1],
            p,
            "zero_smoothing_face_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the zero-smoothing fixture penalties")
    }

    fn gaussian_state_for<'a>(
        y: &'a Array1<f64>,
        weights: &'a Array1<f64>,
        x: &Array2<f64>,
        offset: &Array1<f64>,
        penalties: Vec<gam_terms::construction::CanonicalPenalty>,
        nullspace_dims: Vec<usize>,
        config: &'a RemlConfig,
    ) -> RemlState<'a> {
        RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            penalties,
            4,
            config,
            Some(nullspace_dims),
            None,
            None,
        )
        .expect("build the zero-smoothing fixture state")
    }

    /// THE VALIDATION for the λ → 0 law. On a covered face the criterion is
    /// analytic in `λ_0`, so the production pencil `e^{−ρ_0}·∂V/∂ρ_0` equals
    /// `∂V/∂λ_0` and approaches the analytic slope `c′_0` as
    /// `c′_0 + q·e^{ρ_0} + O(e^{2ρ_0})`: the signed gap contracts by exactly
    /// `e^{−Δ}` across a depth step `Δ`. A wrong dispersion convention, a
    /// missing trace term or a sign error misses by orders of magnitude, not
    /// by a factor that contracts like the law.
    #[test]
    fn covered_zero_smoothing_slope_reproduces_the_production_gradient() {
        let (y, weights, x) = cubic_fixture_sized(SIGNAL_CURVATURE, 96);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = gaussian_state_for(
            &y,
            &weights,
            &x,
            &offset,
            covered_penalties((2, 3)),
            vec![2, 1],
            &config,
        );
        let mut rows = Vec::new();
        for rho_0 in [-10.0_f64, -14.0] {
            let rho = Array1::from(vec![rho_0, 1.0]);
            let law = match state
                .zero_smoothing_face(&rho, &[0])
                .expect("the zero-smoothing gate must not error")
            {
                ZeroSmoothingFaceOutcome::Available(law) => *law,
                other => panic!("a covered Gaussian face is inside the closed form: {other:?}"),
            };
            let eval = state
                .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
                .expect("the production REML gradient must evaluate");
            let pencil = (-rho_0).exp() * eval.gradient[0];
            rows.push((rho_0, law.slopes[0], law.slope_bands[0], pencil));
        }
        let (_, slope, band, pencil_shallow) = rows[0];
        let (_, slope_deep, _, pencil_deep) = rows[1];
        println!(
            "zero-smoothing law: c'={slope:.12e} band={band:.3e}; pencil(-10)={pencil_shallow:.12e} \
             pencil(-14)={pencil_deep:.12e}"
        );
        // The slope is a property of the λ=0 limit, so it is the same number
        // from either depth.
        assert!(
            (slope - slope_deep).abs() <= band,
            "the λ=0 slope moved with the certified depth: {slope:.12e} vs {slope_deep:.12e}"
        );
        assert!(slope > band, "the SIGNAL fixture must prove the face: c'={slope:.6e} band={band:.3e}");
        let gap_shallow = pencil_shallow - slope;
        let gap_deep = pencil_deep - slope;
        let ratio = gap_deep / gap_shallow;
        let expected = (-4.0_f64).exp();
        println!(
            "gap(-10)={gap_shallow:.6e} gap(-14)={gap_deep:.6e} ratio={ratio:.6e} expected={expected:.6e}"
        );
        assert!(
            gap_shallow.abs() < 1.0e-3 * slope.abs(),
            "the production pencil does not approach the analytic slope: gap {gap_shallow:.3e} \
             against c'={slope:.6e}"
        );
        assert!(
            // The next order moves the ratio by O(λ_shallow) = O(e^{−10})
            // relative; measured 2.9e-5.
            (ratio - expected).abs() <= 1.0e-3 * expected,
            "the gap does not contract like the O(λ) law: ratio {ratio:.6e} vs e^-4={expected:.6e}"
        );
    }

    /// Adding the bend to a model whose survivor never touches the bend
    /// columns adds rank: `log|S|₊` gains `2·log λ_0` and `V → +∞` at λ_0 = 0.
    /// That corner is a barrier, and the law must refuse it as a face.
    #[test]
    fn uncovered_zero_smoothing_face_is_refused_as_a_barrier() {
        let (y, weights, x) = cubic_fixture_sized(SIGNAL_CURVATURE, 96);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = gaussian_state_for(
            &y,
            &weights,
            &x,
            &offset,
            cubic_penalties(),
            vec![2, 3],
            &config,
        );
        let rho = Array1::from(vec![-12.0, 1.0]);
        match state
            .zero_smoothing_face(&rho, &[0])
            .expect("the gate must decline, not error")
        {
            ZeroSmoothingFaceOutcome::FaceUnavailable { reason } => assert!(
                reason.contains("barrier"),
                "the refusal must name the barrier: {reason}"
            ),
            other => panic!("an uncovered face is a barrier, got {other:?}"),
        }
    }

    /// A response the survivor-penalized model fits EXACTLY leaves no
    /// residual, `D_p⁰ = 0` and `V → −∞` at λ_0 = 0: no minimizer, no face.
    #[test]
    fn exact_fit_zero_smoothing_face_is_refused() {
        let n = 48usize;
        let (_, weights, x) = cubic_fixture_sized(0.0, n);
        let y: Array1<f64> = (0..n).map(|i| 0.8 + 0.3 * x[[i, 1]]).collect();
        let offset = Array1::<f64>::zeros(n);
        let config = gaussian_config();
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[2, 2]] = 1.0;
        bend[[3, 3]] = 2.0;
        let mut ridge = Array2::<f64>::zeros((p, p));
        ridge[[2, 2]] = 1.0;
        ridge[[3, 3]] = 1.0;
        let penalties = gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(bend),
                crate::estimate::PenaltySpec::Dense(ridge),
            ],
            &[2, 2],
            p,
            "zero_smoothing_exact_fit_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the exact-fit fixture penalties");
        let state = gaussian_state_for(&y, &weights, &x, &offset, penalties, vec![2, 2], &config);
        let rho = Array1::from(vec![-12.0, 1.0]);
        match state
            .zero_smoothing_face(&rho, &[0])
            .expect("the gate must decline, not error")
        {
            ZeroSmoothingFaceOutcome::FaceUnavailable { reason } => assert!(
                reason.contains("exact"),
                "the refusal must name the exact fit: {reason}"
            ),
            other => panic!("an exact fit has no zero-smoothing face, got {other:?}"),
        }
    }
}
