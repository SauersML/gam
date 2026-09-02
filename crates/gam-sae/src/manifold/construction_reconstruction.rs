//! Reconstruction-dispersion and shape-uncertainty methods, split out of the
//! tail of `construction.rs` to keep that tracked file under the #780 10k-line
//! gate. Holds the contiguous trailing `impl SaeManifoldTerm` block:
//! `reconstruction_dispersion` (the Gaussian dispersion `φ̂` estimator),
//! `assemble_shape_uncertainty`, `recompute_joint_shape_uncertainty`, and the
//! explicit streaming-unavailable shape report. All are reached bare by
//! callers through `use super::*`, so their visibility is unchanged.

use super::*;

/// Assembly scale `‖H‖₂` of the whole arrow, from the Cholesky factors the
/// cache already carries.
///
/// The arrow factorizes as `H = L Lᵀ` with `L` block-lower: the per-row
/// coordinate factors on the diagonal, the border rows below them, and the
/// β-Schur factor last. So `‖H‖₂ ≤ ‖L‖₂² ≤ ‖L‖_F²`, and `‖L‖_F²` is the SUM of
/// its blocks' squared Frobenius norms — not their maximum. The distinction
/// matters: the ARD trace reads the latent diagonal of `H⁻¹`, which is
/// `A_i⁻¹ + G_i S⁻¹ G_iᵀ`, so both the row-local and the border-through-Schur
/// paths contribute, and the n-fold accumulation across rows is already
/// carried by the `shrinkage` factor in the tolerance rather than by this
/// scale.
///
/// The border rows themselves are an operator (`apply_htbeta_row`), not a
/// stored matrix, so this sums the two block families the cache materializes
/// and omits the border's own contribution. `‖L‖_F²` is therefore
/// under-counted, which makes this an observable scale rather than a proved
/// bound on `‖H‖₂`. The certificate does not rest on tightness: roundoff at
/// this scale and a genuinely indefinite block are separated by ten orders of
/// magnitude on every state measured (`1.97e-9` against `5.5e1` on the same
/// fixture), so what the scale has to get right is the exponent, not the
/// constant.
pub(super) fn undamped_row_curvature_scale(cache: &ArrowFactorCache) -> f64 {
    let frobenius_squared =
        |factor: ArrayView2<'_, f64>| -> f64 { factor.iter().map(|entry| entry * entry).sum() };
    let mut scale = 0.0_f64;
    for row in 0..cache.undamped_factor_count() {
        scale += frobenius_squared(cache.undamped_factor(row));
    }
    if let Some(schur) = cache.schur_factor.as_ref() {
        scale += frobenius_squared(schur.view());
    }
    scale
}

/// Forward-error constant for the ARD trace. Matches the constant the interval
/// certificate has always used; the change below is the SCALE it multiplies,
/// not its size.
const ARD_EDF_FORWARD_ERROR_FACTOR: f64 = 64.0;

/// Certify the ARD identity
/// `edf = n_active - alpha * shrinkage_trace` against its exact
/// `[0, n_active]` interval.  A tiny excursion at the forward-error scale of
/// the accumulated trace is snapped to the boundary; a material excursion is a
/// failed trace certificate, not an EDF that may be silently projected into
/// another model.
///
/// `shrinkage_trace` is [`SaeManifoldTerm::ard_shrinkage_traces`], NOT
/// [`SaeManifoldTerm::ard_inverse_traces`]: it carries the per-row
/// dimensionless prior-curvature factor `f = softplus_{τ₀}(cos κt)` so that
/// `alpha * shrinkage_trace` is `Σ_i P_i·[H⁻¹]_{ss}` with `P_i` the curvature
/// the arrow was actually assembled with. On a Euclidean axis `f ≡ 1` and the
/// two traces are the same number; on a periodic axis they are not, and only
/// this one is bounded (#2499 — see that function's derivation).
///
/// # Why the tolerance carries the block's conditioning
///
/// The interval is exact for `H = C + P` with data curvature `C ⪰ 0` and
/// diagonal prior curvature `P ⪰ 0`: then `H ⪰ P`, so every slot has
/// `P_s·[H⁻¹]_ss ≤ 1` (and `= 0` where `P_s = 0`), and the sum lands in
/// `[0, n_active]`. The certificate is therefore asking whether the ASSEMBLED
/// curvature is PSD, and it must tell a state whose curvature is genuinely
/// indefinite from one whose curvature is zero on this axis and whose assembled
/// copy carries a roundoff-scale negative eigenvalue.
///
/// For `Ĥ = H + E`,
/// `|α·tr(Ĥ⁻¹) − α·tr(H⁻¹)| ≤ α‖E‖₂·tr(H⁻²) ≤ (‖E‖₂/λ_min(H))·α·tr(H⁻¹)`,
/// and on the Euclidean axis this derivation was written for, `λ_min(H) ≥ α`,
/// so with `‖E‖₂ ≤ c·ε·‖H‖₂` the excursion is bounded by
/// `c·ε·(‖H‖₂/α)·shrinkage`. The factor `‖H‖₂/α` is the block's conditioning,
/// and it is exactly what a tolerance written against `n_active + shrinkage`
/// omits: on a `d`-dimensional axis pinned at the prior, `shrinkage = n_active`
/// identically, so that form measures the answer's magnitude rather than the
/// accuracy with which it was reached. Measured on the shared two-atom fixture
/// it was short by ~700x (excursion `1.97e-9` against a `2.84e-12` bound) at a
/// state where the exact EDF is zero, which refused an ordinary dispersion.
/// A genuinely indefinite state on the same fixture missed the interval by
/// `17.3` on `n_active = 10` — ten orders of magnitude clear of either bound,
/// so the conditioning factor costs the certificate no discrimination. On a
/// periodic axis `λ_min(H) ≥ α` does not hold, so `‖H‖₂/α` is an observable
/// scale there rather than a proved bound — the same standing the scale it
/// multiplies already has (see [`undamped_row_curvature_scale`]), and the two
/// regimes this certificate separates are ten orders apart on every state
/// measured.
pub(super) fn certified_ard_axis_edf(
    n_active: f64,
    alpha: f64,
    inverse_trace: f64,
    curvature_scale: f64,
    atom: usize,
    axis: usize,
) -> Result<f64, String> {
    if !(n_active.is_finite() && n_active >= 0.0) {
        return Err(format!(
            "reconstruction_dispersion: ARD active count at atom {atom}, axis {axis} \
             must be finite and non-negative; got {n_active}"
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0 && inverse_trace.is_finite()) {
        return Err(format!(
            "reconstruction_dispersion: ARD precision/trace at atom {atom}, axis {axis} \
             must be finite with positive precision; got alpha={alpha}, trace={inverse_trace}"
        ));
    }
    let shrinkage = alpha * inverse_trace;
    let raw = n_active - shrinkage;
    if !shrinkage.is_finite() || !raw.is_finite() {
        return Err(format!(
            "reconstruction_dispersion: ARD EDF arithmetic is unrepresentable at atom \
             {atom}, axis {axis} (n_active={n_active}, alpha={alpha}, trace={inverse_trace})"
        ));
    }
    if !(curvature_scale.is_finite() && curvature_scale >= 0.0) {
        return Err(format!(
            "reconstruction_dispersion: ARD curvature scale at atom {atom}, axis {axis} \
             must be finite and non-negative; got {curvature_scale}"
        ));
    }
    // `‖H‖₂/α ≥ 1` always, since `H ⪰ αI`; a scale that says otherwise is a
    // block whose largest eigenvalue is the prior itself.
    let conditioning = (curvature_scale / alpha).max(1.0);
    let tolerance = ARD_EDF_FORWARD_ERROR_FACTOR
        * f64::EPSILON
        * conditioning
        * shrinkage.abs().max(n_active).max(1.0);
    // #2499 — one message cannot serve two failure modes whose remedies are
    // opposite. A roundoff excursion (just past a correctly-derived tolerance)
    // is answered by re-deriving the tolerance; a STRUCTURAL excursion — one
    // whose magnitude is orders past any admissible tolerance — is answered by
    // suspecting the assembly, and widening is categorically wrong. The
    // discriminator is already computed: the excursion measured in units of the
    // tolerance. Ten-plus orders means no admissible widening reaches it, so
    // the two must not be one grep-able string.
    if raw < -tolerance || raw > n_active + tolerance {
        let excursion = if raw < 0.0 { -raw } else { raw - n_active };
        let over_tolerance = excursion / tolerance;
        let regime = if over_tolerance > 1.0e3 {
            "STRUCTURAL: the assembled quantity is not the EDF this interval was \
             derived for — widening the tolerance cannot reach it. An EDF is a \
             trace of a projection-like operator; check that the prior curvature \
             the shrinkage trace carries is the curvature the arrow was assembled \
             with (a periodic axis majorizes it to α·softplus(cos κt), NOT α)"
        } else {
            "ROUNDOFF: the excursion is at the forward-error scale of the \
             accumulated trace, so the tolerance's derivation is the suspect, \
             not the assembly"
        };
        return Err(format!(
            "reconstruction_dispersion: ARD EDF at atom {atom}, axis {axis} is \
             {raw:.6e}, outside certified [0, {n_active}] by {excursion:.6e} = \
             {over_tolerance:.3e}x the roundoff tolerance {tolerance:.6e} at \
             conditioning {conditioning:.6e}; alpha={alpha:.6e}, \
             shrinkage_trace={inverse_trace:.6e}. {regime}"
        ));
    }
    Ok(raw.clamp(0.0, n_active))
}

/// Project a Hutchinson ARD-EDF estimate onto its known parameter space.
///
/// Individual grouped diagonal estimates can leave `[0, n_active]` by sampling
/// noise even when the exact trace is valid. Euclidean projection onto this
/// closed interval is the constrained estimator: for every true EDF in the
/// interval it cannot increase squared error. This is deliberately distinct
/// from the exact-trace certificate above and is used only on the declared
/// massive-K stochastic trace lane.
///
/// `shrinkage_trace_estimate` is the stochastic
/// [`SaeManifoldTerm::ard_shrinkage_traces`] — carrying the per-row prior
/// curvature factor, so `alpha * estimate` is `Σ_i P_i·[H⁻¹]_ii` and the
/// interval it is projected onto is the one the quantity actually lives in
/// (#2499). Projecting an `α·tr(H⁻¹)` estimate here would clamp a structurally
/// out-of-range value to a boundary and report it as sampling noise.
fn projected_hutchinson_ard_axis_edf(
    n_active: f64,
    alpha: f64,
    shrinkage_trace_estimate: f64,
    atom: usize,
    axis: usize,
) -> Result<f64, String> {
    if !(n_active.is_finite()
        && n_active >= 0.0
        && alpha.is_finite()
        && alpha > 0.0
        && shrinkage_trace_estimate.is_finite())
    {
        return Err(format!(
            "reconstruction_dispersion: stochastic ARD EDF inputs at atom {atom}, axis \
             {axis} must be finite with non-negative active count and positive precision; \
             got n_active={n_active}, alpha={alpha}, trace={shrinkage_trace_estimate}"
        ));
    }
    let estimate = n_active - alpha * shrinkage_trace_estimate;
    if !estimate.is_finite() {
        return Err(format!(
            "reconstruction_dispersion: stochastic ARD EDF estimate is unrepresentable at \
             atom {atom}, axis {axis}"
        ));
    }
    Ok(estimate.clamp(0.0, n_active))
}

#[cfg(test)]
mod ard_edf_certificate_tests {
    use super::{certified_ard_axis_edf, projected_hutchinson_ard_axis_edf};

    #[test]
    fn snaps_only_trace_roundoff_at_the_ard_edf_faces() {
        let n = 8.0;
        let tiny = 8.0 * f64::EPSILON;
        assert_eq!(certified_ard_axis_edf(n, 1.0, -tiny, 1.0, 0, 0).unwrap(), n);
        assert_eq!(
            certified_ard_axis_edf(n, 1.0, n + tiny, 1.0, 0, 0).unwrap(),
            0.0
        );
    }

    #[test]
    fn refuses_material_or_nonfinite_ard_edf_excursions() {
        for trace in [-1.0e-8, 8.0 + 1.0e-8, f64::NAN, f64::INFINITY] {
            assert!(certified_ard_axis_edf(8.0, 1.0, trace, 1.0, 2, 3).is_err());
        }
    }

    /// The tolerance must track the block's conditioning `‖H‖₂/α`, not the
    /// magnitude of the answer.
    ///
    /// The state below is the one measured on the shared two-atom fixture: an
    /// ARD axis pinned at the prior, so the exact EDF is zero and the trace is
    /// `n_active/α` exactly. Its assembled copy overshoots by `1.97e-9`, which
    /// is `c·ε·(‖H‖₂/α)·shrinkage` for an assembly scale of order `10³` — an
    /// ordinary state, not an indefinite one. A tolerance written against
    /// `n_active + shrinkage` alone rejects it, so this pins the scale.
    #[test]
    fn ard_edf_tolerance_admits_pinned_axis_roundoff_and_still_refuses_a_saddle() {
        let n_active = 10.0_f64;
        let alpha = 2.478752e-3_f64;
        let pinned_trace = n_active / alpha;
        // The measured excursion, expressed back in trace units.
        let overshoot = 1.973920e-9 / alpha;
        let curvature_scale = 1.0e3;
        assert_eq!(
            certified_ard_axis_edf(
                n_active,
                alpha,
                pinned_trace + overshoot,
                curvature_scale,
                1,
                0
            )
            .unwrap(),
            0.0,
            "a pinned ARD axis whose assembled trace overshoots at the block's \
             own forward-error scale is EDF zero, not a failed certificate"
        );
        // The same fixture's genuinely indefinite state: `α·tr = 27.3` against
        // `n_active = 10`. Ten orders of magnitude clear of the bound above.
        let saddle_trace = 27.27_f64 / alpha;
        assert!(
            certified_ard_axis_edf(n_active, alpha, saddle_trace, curvature_scale, 1, 0).is_err(),
            "coordinate curvature far below the prior is an indefinite block; the \
             conditioning factor must not launder it"
        );
        // Conditioning cannot buy an arbitrary excursion: a scale large enough
        // to admit the saddle is not one this certificate ever sees, but the
        // bound must still be the derived product rather than a free pass.
        assert!(
            certified_ard_axis_edf(n_active, alpha, pinned_trace + overshoot, 0.0, 1, 0).is_err(),
            "at unit conditioning the pinned-axis overshoot is far outside the bound"
        );
    }

    #[test]
    fn stochastic_trace_lane_uses_the_declared_constrained_estimator() {
        assert_eq!(
            projected_hutchinson_ard_axis_edf(8.0, 1.0, -2.0, 0, 0).unwrap(),
            8.0
        );
        assert_eq!(
            projected_hutchinson_ard_axis_edf(8.0, 1.0, 10.0, 0, 0).unwrap(),
            0.0
        );
    }
}

/// Reconstruct a persisted SAE-manifold atom set from frozen coordinates,
/// assignment masses, and decoder blocks.
///
/// This is the stateless counterpart to [`SaeManifoldTerm::try_fitted`]: Python
/// artifacts that intentionally dropped the full term still carry enough
/// persisted atom state to materialize `Σ_k a_ik · Φ_k(t_ik)B_k`. Keeping the
/// basis evaluation, GEMM, and weighted atom sum here prevents the Python facade
/// from becoming a second decoder implementation.
pub fn reconstruct_persisted_atom_set(
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    p_out: usize,
) -> Result<Array2<f64>, String> {
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(format!(
            "reconstruct_persisted_atom_set: decoder and coordinate counts must equal \
             geometry-plan count K={k_atoms} (decoder_blocks={}, coords={})",
            decoder_blocks.len(),
            coords.len()
        ));
    }
    let n_rows = assignments.nrows();
    if assignments.ncols() != k_atoms {
        return Err(format!(
            "reconstruct_persisted_atom_set: assignments {:?} must have K={k_atoms} columns",
            assignments.dim()
        ));
    }
    if p_out == 0 {
        return Err("reconstruct_persisted_atom_set: p_out must be positive".to_string());
    }
    let mut out = Array2::<f64>::zeros((n_rows, p_out));
    for atom_idx in 0..k_atoms {
        let plan = &geometry_plans[atom_idx];
        let basis_width = plan.basis_size()?;
        let decoder = decoder_blocks[atom_idx];
        if decoder.dim() != (basis_width, p_out) {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} decoder shape {:?} must \
                 equal plan-derived ({basis_width}, {p_out})",
                decoder.dim()
            ));
        }
        let atom_coords = coords[atom_idx];
        if atom_coords.nrows() != n_rows {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} coords rows {} != {n_rows}",
                atom_coords.nrows()
            ));
        }
        if atom_coords.ncols() != plan.latent_dim() {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} coordinate width {} must \
                 equal plan latent_dim {}",
                atom_coords.ncols(),
                plan.latent_dim()
            ));
        }
        let (phi, _) = plan.build_evaluator()?.evaluate(atom_coords)?;
        if phi.dim() != (n_rows, basis_width) {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} basis {:?} != ({n_rows}, {basis_width})",
                phi.dim()
            ));
        }
        let decoded = phi.dot(&decoder);
        for row in 0..n_rows {
            let gate = assignments[[row, atom_idx]];
            if gate == 0.0 {
                continue;
            }
            for col in 0..p_out {
                out[[row, col]] += gate * decoded[[row, col]];
            }
        }
    }
    Ok(out)
}

impl SaeManifoldTerm {
    fn reconstruction_residual_sum_squares(
        &self,
        loss: &SaeManifoldLoss,
        residual: Option<ArrayView2<'_, f64>>,
    ) -> Result<f64, String> {
        if let Some(residual) = residual.as_ref() {
            if residual.dim() != (self.n_obs(), self.output_dim()) {
                return Err(format!(
                    "reconstruction residual shape {:?} does not match ({}, {})",
                    residual.dim(),
                    self.n_obs(),
                    self.output_dim(),
                ));
            }
            if residual.iter().any(|value| !value.is_finite()) {
                return Err("reconstruction residual must be finite".to_string());
            }
        }
        let metric_whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let rss = if metric_whitens {
            residual
                .as_ref()
                .map(|values| values.iter().map(|value| value * value).sum::<f64>())
                .unwrap_or(2.0 * loss.data_fit)
        } else {
            2.0 * loss.data_fit
        };
        if rss.is_finite() && rss >= 0.0 {
            Ok(rss)
        } else {
            Err(format!(
                "reconstruction residual sum of squares must be finite and non-negative; got {rss}"
            ))
        }
    }

    /// Gaussian reconstruction dispersion `φ̂`, the scale that turns the
    /// unscaled inverse-Hessian β-block `S_β⁻¹` into a posterior covariance
    /// `Cov(β) = φ̂·S_β⁻¹` — the same `Vb = φ·H⁻¹` convention the main GAM
    /// inference path uses.
    ///
    /// `RSS = Σ_{i,c} (z_{ic} − ẑ_{ic})² = 2·data_fit` (the loss stores the
    /// half-sum `½Σr²`). The residual degrees of freedom subtract the effective
    /// parameter count from the `N·p` scalar observations:
    ///   * decoder β: `beta_dim − tr(λ_smooth · S_β⁻¹ · ⊕_k S_k⊗I_p)`, the
    ///     smoothness effective-dof already assembled for the Fellner-Schall
    ///     step (penalty-shrunk directions do not cost a full parameter);
    ///   * latent coordinates: enabled ARD axes use the ARD-shrunk trace
    ///     `Σ_k Σ_j (n_active_k − α_{kj}·tr_{kj}(H⁻¹))`; atoms with disabled
    ///     native ARD charge the full active coordinate count because those
    ///     latent variables are estimated without an ARD precision.
    ///
    /// Below the declared massive-K threshold the coordinate term is the exact
    /// ARD-shrunk effective dof of the latent block: along axis `(k,j)` the
    /// MacKay/Fellner-Schall edf is
    /// `n_active_k − α_{kj}·τ_{kj}`, the well-determined-direction count
    /// after the ARD prior `α_{kj}` shrinks each coordinate. `τ_{kj}` is the
    /// SHRINKAGE trace [`Self::ard_shrinkage_traces`], NOT the posterior-variance
    /// trace [`Self::ard_inverse_traces`] the EFS ARD step consumes: the two
    /// coincide on a Euclidean axis and differ on a periodic one, where the prior
    /// curvature the arrow carries is the PSD majorizer `α·softplus(cos κt) ≤ α`
    /// rather than `α` (#2499). Only the former makes `α·τ ∈ [0, n_active_k]`.
    /// The per-axis scalar count `n_active_k` must match the support the trace sums
    /// over: `n` for the dense full-support layout, or the number of rows where
    /// atom `k` is active for the compact active-set layout (inactive
    /// prior-dominated coordinates contribute 0 to both the trace and the
    /// count, hence 0 edf). At massive K the selected-inverse diagonal is the
    /// declared Hutchinson estimate; its grouped EDF is projected onto the exact
    /// `[0,n_active_k]` parameter space, which cannot increase squared error.
    /// The residual dof is floored at 1 so `φ̂` stays finite and positive.
    /// `residual` is the per-row reconstruction residual `f(θ̂) − y` (n×p) at the
    /// same state that produced `cache`. When supplied it engages the #2133 SURE
    /// within-basin second-order deflation correction
    /// ([`Self::coordinate_sure_deflation_correction`]) — the exact-Newton
    /// completion of the Gauss-Newton `coord_edf`, which removes the
    /// incidental-parameters under-dispersion of the per-row coordinate MAP.
    /// `None` reproduces the historical Gauss-Newton dispersion exactly (used by
    /// callers with no residual in hand — the correction is then simply absent).
    pub(crate) fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        residual: Option<ArrayView2<'_, f64>>,
    ) -> Result<f64, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n = self.n_obs();
        let p = self.output_dim();
        // Design-honesty weights are normalized to mean one, so they redistribute
        // residual mass without changing the scalar observation count.
        let n_scalar = (n * p) as f64;
        // FRAME CONSISTENCY (#2228/#2258 tier-0 root cause): under an active
        // WHITENING row metric the likelihood's `loss.data_fit` is the
        // WHITENED residual energy — ≈ n·p BY CONSTRUCTION (whitening
        // normalizes residuals to unit scale) — so a φ̂ built from it prices
        // the noise floor at ~n·p/resid_dof ≈ 2 REGARDLESS of the actual fit
        // quality. Every consumer of this dispersion lives in the RAW output
        // frame: the rank-charge MP edge compares against the unwhitened
        // reconstruction Gram (measured veto: R=2.16 vs top signal 1.01 on a
        // fitted EV=0.998 circle → rank_eff=0 → categorical +∞ → 'infeasible
        // at the requested rho' for every structured pass), and the shape
        // bands are φ-scaled output-frame covariances. Price φ from the RAW
        // residual whenever the caller supplied it and the metric whitens.
        let rss = self.reconstruction_residual_sum_squares(loss, residual)?;
        let smooth_edf: f64 = self
            .decoder_smoothness_effective_dof_per_atom(cache, &rho.lambda_smooth_vec()?)
            .map_err(|e| format!("reconstruction_dispersion: smooth edf: {e}"))?
            .iter()
            .sum();
        // #972 / #977 T1: the raw decoder-parameter count is `beta_dim` on the
        // full-`B` path, but when frames are active the estimated decoder freedom
        // is the factored border `Σ M_k·r_k` PLUS the `Σ r_k·(p−r_k)` Grassmann
        // frame degrees profiled out (both are genuinely estimated), which the
        // smoothness shrinkage `smooth_edf` (taken over the factored border) then
        // discounts. On the full-`B` path `factored_border_dim == beta_dim` and
        // `grassmann_evidence_dimension == 0`, so this is exactly `beta_dim`.
        let raw_decoder_dof = if self.frames_active() {
            (self.factored_border_dim() + self.grassmann_evidence_dimension()) as f64
        } else {
            self.beta_dim() as f64
        };
        let beta_edf = (raw_decoder_dof - smooth_edf).max(0.0);
        // ARD-shrunk latent-coordinate EDF, reusing the EFS trace cache.
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let stochastic_ard_trace = self.k_atoms() >= Self::ARD_TRACE_HUTCHINSON_MIN_ATOMS;
        let traces = self
            .ard_shrinkage_traces(cache)
            .map_err(|e| format!("reconstruction_dispersion: ARD shrinkage traces: {e}"))?;
        // The scale at which those traces' forward error lives (see
        // `certified_ard_axis_edf`). One pass over factors the cache already
        // holds.
        let curvature_scale = undamped_row_curvature_scale(cache);
        let mut coord_edf = 0.0_f64;
        for (k, atom) in self.atoms.iter().enumerate() {
            let d_k = atom.latent_dim();
            if traces[k].len() != d_k {
                return Err(format!(
                    "reconstruction_dispersion: trace shape mismatch at atom {k} \
                     (traces={}, d_k={d_k})",
                    traces[k].len()
                ));
            }
            let ard_len = rho.log_ard[k].len();
            // Scalar count matched to the trace support (see fn doc).
            let n_active_k = match self.last_row_layout {
                Some(ref layout) => layout
                    .active_atoms
                    .iter()
                    .filter(|active| active.contains(&k))
                    .count() as f64,
                None => n as f64,
            };
            if ard_len == 0 {
                coord_edf += n_active_k * d_k as f64;
                continue;
            }
            for j in 0..d_k {
                let alpha = ard_precisions[k][j];
                let edf_kj = if stochastic_ard_trace {
                    projected_hutchinson_ard_axis_edf(n_active_k, alpha, traces[k][j], k, j)?
                } else {
                    certified_ard_axis_edf(n_active_k, alpha, traces[k][j], curvature_scale, k, j)?
                };
                coord_edf += edf_kj;
            }
        }
        // #2133 — restore the second-order residual-curvature term the
        // Gauss-Newton `coord_edf` above drops, turning the per-row GN divergence
        // into the exact within-basin SURE divergence of the coordinate MAP. Pure
        // additive readout; only engaged when the caller supplies the residual.
        if let Some(residual) = residual {
            coord_edf = (coord_edf + self.coordinate_sure_deflation_correction(residual, rho)?)
                .clamp(0.0, n_scalar);
            // #2133 — the basin-SELECTION (search) deflation dof: the boundary
            // Stein term the within-basin correction above omits. The per-row charge
            // depends on σ̂ = √φ̂, so seed it with the within-basin-corrected but
            // search-UNcorrected φ̂ and take ONE monotone fixed-point pass (the charge
            // is decreasing in σ̂ through the margin z, so one pass contracts). It is
            // identically 0 for single-basin / hard-frozen / genuinely-soft rows, so
            // w=None + non-selecting fits are bit-for-bit today's φ̂.
            let phi_seed = rss / (n_scalar - beta_edf - coord_edf).max(1.0);
            let df_search = self.basin_selection_deflation_correction(residual, phi_seed)?;
            coord_edf = (coord_edf + df_search).clamp(0.0, n_scalar);
        }
        let resid_dof = (n_scalar - beta_edf - coord_edf).max(1.0);
        let phi = rss / resid_dof;
        if !phi.is_finite() || phi < 0.0 {
            return Err(format!(
                "reconstruction_dispersion: non-finite/negative φ̂={phi} \
                 (RSS={rss}, resid_dof={resid_dof}, beta_edf={beta_edf}, coord_edf={coord_edf})"
            ));
        }
        Ok(phi.max(f64::MIN_POSITIVE))
    }

    /// Posterior covariance and ambient shape band for every atom — the
    /// user-facing uncertainty of the fitted manifold shapes.
    ///
    /// For atom `k` with decoder-block range `r_k` (see
    /// [`Self::beta_block_offsets`]), `Cov(β_k) = φ·S_β⁻¹[r_k, r_k]` is the
    /// φ-scaled posterior covariance of its decoder coefficients with the
    /// latent coordinates marginalized out. The ambient point at a coordinate
    /// `t` is `m_k(t) = Φ_k(t)·B_k`, *linear* in `β_k`, so its per-channel
    /// posterior variance is the closed form
    /// `Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]`
    /// — no sampling. The band is evaluated at up to [`SHAPE_BAND_MAX_POINTS`]
    /// evenly-strided of the atom's own on-atom coordinates, reusing the basis
    /// values already stored on the atom, so it reports uncertainty exactly
    /// where the data lives and needs no basis-kind-specific grid.
    ///
    /// A near-degenerate atom has a near-singular Schur block, so `Cov(β_k)` —
    /// and the band — fans out automatically: the band width is a
    /// per-coordinate visual of how well each atom is identified.
    pub fn assemble_shape_uncertainty(
        &self,
        cache: &ArrowFactorCache,
        dispersion: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache β block is the FACTORED border when frames
        // are active, so each atom's Schur inverse block is the `(M_k·r_k)`
        // coordinate covariance `Cov(vec C_k)`. We LIFT it to the full
        // `(M_k·p)` decoder covariance `Cov(vec B_k) = (I_{M_k} ⊗ U_k) Cov(vec
        // C_k)(I_{M_k} ⊗ U_k)ᵀ` (since `B_k = C_k U_kᵀ`) so the downstream band
        // code — which reads the `b·p + c` flat layout — is unchanged. On the
        // full-`B` path the block is already `(M_k·p)` and the lift is skipped.
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if frames_active {
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect::<Vec<_>>()
        } else {
            self.beta_block_offsets().to_vec()
        };
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let cov_block = cache
                .schur_inverse_block(block_ranges[k].clone())
                .map_err(|e| format!("assemble_shape_uncertainty: atom {k}: {e}"))?;
            let n_rows = atom.n_obs();
            let d = atom.latent_dim();
            // Evenly-strided evaluation rows bound the band cost.
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let mut band_sd = Array2::<f64>::zeros((g, p));
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }

            let framed = frames_active && atom.decoder_frame.is_some();
            let dense_entries = (m * p).saturating_mul(m * p);
            let cov = if framed && dense_entries > SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES {
                // LLM-scale ambient `p`: the dense `(M_k·p)²` lift would be
                // gigabytes per atom and exists only to export the full
                // covariance. Compute the band variance EXACTLY from the
                // factored frame covariance instead: with `B_k = C_k·U_kᵀ`,
                //   Var_c(t) = (φ ⊗ u_c)ᵀ Cov(vec C_k) (φ ⊗ u_c)
                // which is the r×r quadratic form `u_cᵀ Y u_c` with
                //   Y = Σ_{b1,b2} φ[b1] φ[b2] Cov(C)[(b1,·),(b2,·)].
                let mut cov_c = cov_block;
                cov_c.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let basis = atom.basis_values.row(row);
                    for c in 0..p {
                        let var = frame_projection.output_variance(k, cov_c.view(), basis, c);
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                None
            } else {
                // Lift the factored `(M_k·r_k)` coordinate covariance to the
                // full `(M_k·p)` decoder covariance through this atom's frame;
                // identity (a plain scaled copy) on the un-framed full-`B` path.
                let mut cov = if framed {
                    frame_projection.lift_block(k, cov_block.view())
                } else {
                    cov_block
                };
                cov.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    // Var_c = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]; the flat
                    // decoder index is basis·p + channel (row-major (M_k, p)).
                    for c in 0..p {
                        let var = frame_projection.full_output_variance(
                            k,
                            cov.view(),
                            atom.basis_values.row(row),
                            c,
                        );
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                Some(cov)
            };
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords: Some(band_coords),
                band_mean: Some(band_mean),
                band_sd: Some(band_sd),
                band_sd_robust: None,
            });
        }
        Ok(SaeShapeUncertainty { dispersion, atoms })
    }

    /// Recompute the JOINT inverse-Hessian shape bands at the CURRENT (final)
    /// term + ρ state — the same joint covariance
    /// [`Self::assemble_shape_uncertainty`] forms, but rebuilt AFTER a
    /// structure-changing or finalization move invalidated the pre-search Schur
    /// factor.
    ///
    /// [`super::SaeManifoldOuterObjective::decoder_shape_uncertainty`] reads the
    /// joint factor off the outer objective BEFORE `into_fitted` consumes it, so
    /// the bands it returns describe the PRE-search dictionary at the settled ρ.
    /// When evidence-guarded structure search grows / re-converges the whole
    /// dictionary (a certified birth / fission / fusion or a demoted death), or a
    /// finalization fallback swaps the settled basin / canonicalizes charts, that
    /// factor no longer describes the returned model. This rebuilds the undamped
    /// Direct joint-Hessian factor from THIS (final) term at `rho` — the exact
    /// factor the penalized quasi-Laplace criterion forms at the inner optimum — and reads the
    /// per-atom covariance and bands off its Schur factor, scaling by the
    /// reconstruction dispersion `φ̂`. The result is the DOCUMENTED joint
    /// covariance: it carries the cross-atom covariance and the decoder-coordinate
    /// Schur couplings, and its per-channel band varies across output channels.
    /// Every atom is covered because the factor is assembled at the final
    /// dictionary's `k_atoms()`.
    ///
    /// The term is already at its optimum, so the inner re-solve converges
    /// immediately. When the streaming plan cannot expose the exact Direct
    /// factor, the returned atom entries carry explicit `None` bands. Call before
    /// `Self::into_fitted` has run is not required; it takes the fitted
    /// `term`/`rho` directly.
    pub fn recompute_joint_shape_uncertainty(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            // No exact Direct Schur factor at this scale: report explicit
            // unavailability rather than substituting a different covariance.
            let loss = self.loss(target, rho)?;
            let n_scalar = (self.n_obs().saturating_mul(self.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self.unavailable_shape_uncertainty(dispersion));
        }
        let (_cost, loss, cache) = self
            .penalized_quasi_laplace_criterion_with_cache(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .map_err(|error| error.to_string())?;
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion =
            self.reconstruction_dispersion(&loss, &cache, rho, Some(residual.view()))?;
        self.assemble_shape_uncertainty(&cache, dispersion)
    }

    /// Explicitly unavailable joint shape uncertainty for a streaming fit whose
    /// execution plan cannot expose the exact joint Schur factor.
    pub(crate) fn unavailable_shape_uncertainty(&self, dispersion: f64) -> SaeShapeUncertainty {
        let atoms = self
            .atoms
            .iter()
            .map(|_| SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords: None,
                band_mean: None,
                band_sd: None,
                band_sd_robust: None,
            })
            .collect();
        SaeShapeUncertainty { dispersion, atoms }
    }

}

#[cfg(test)]
mod persisted_reconstruct_tests {
    use super::*;

    // Exercises `reconstruct_persisted_atom_set`: a stateless K=1 periodic-atom
    // round trip that must equal `a_i · (Φ(t_i) · B)` computed directly from the
    // exact evaluator declared by the persisted geometry plan.
    #[test]
    fn reconstruct_persisted_periodic_atom_matches_direct_decode() {
        let n_rows = 4usize;
        let p_out = 2usize;
        let width = 3usize; // odd decoder width required for periodic
        let coords = Array2::from_shape_vec((n_rows, 1), vec![0.1, 0.7, 1.9, 2.8]).unwrap();
        let decoder =
            Array2::from_shape_vec((width, p_out), vec![0.5, -0.2, 0.3, 0.9, -0.4, 0.1]).unwrap();
        let assignments = Array2::from_shape_vec((n_rows, 1), vec![1.0, 0.5, 0.8, 0.2]).unwrap();

        let plan = SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Periodic,
            1,
            SaeBasisResolution::PeriodicHarmonics { order: 1 },
            SaeReferenceMetricPlan::UnitCircle,
        )
        .unwrap();
        let out = reconstruct_persisted_atom_set(
            &[plan],
            &[decoder.view()],
            &[coords.view()],
            assignments.view(),
            p_out,
        )
        .expect("reconstruct persisted periodic atom");
        assert_eq!(out.dim(), (n_rows, p_out));

        let evaluator = PeriodicHarmonicEvaluator::new(width).unwrap();
        let (phi, _jet) = evaluator.evaluate(coords.view()).unwrap();
        let decoded = phi.dot(&decoder);
        for i in 0..n_rows {
            for j in 0..p_out {
                let expected = assignments[[i, 0]] * decoded[[i, j]];
                assert!(
                    (out[[i, j]] - expected).abs() < 1.0e-9,
                    "row {i} col {j}: got {} expected {expected}",
                    out[[i, j]]
                );
            }
        }
    }
}
