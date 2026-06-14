//! Post-fit *inference* for smooth terms: curvature (effective-df) inference and
//! the smooth-term likelihood-ratio correction.

use super::*;

pub struct CurvatureInference {
    /// Smooth-term index of the `curv(...)` term this report is about.
    pub term_idx: usize,
    /// The fitted signed sectional curvature κ̂ (the outer optimiser's argmin of
    /// the profiled REML/LAML criterion over κ).
    pub kappa_hat: f64,
    /// Profile-likelihood CI for κ and the geometry verdict from its sign.
    pub ci: crate::geometry::curvature_estimand::KappaProfileCi,
    /// Interior-point κ = 0 likelihood-ratio flatness test (full χ²₁, no
    /// half-χ² boundary correction — κ = 0 is an interior point of the
    /// `S^d ← ℝ^d → H^d` family).
    pub flatness: crate::geometry::curvature_estimand::FlatnessTest,
}


/// Compute the #944 curvature inference for the constant-curvature smooth at
/// `term_idx`, given the already-fitted resolved spec (carrying κ̂) and the same
/// fit inputs used to produce it.
///
/// The profiled criterion `V_p(κ) = max_{ρ} V(κ, ρ)` is evaluated as an oracle:
/// for each probe κ, pin the term's curvature to κ, fit with κ-optimisation
/// **disabled** (so only the smoothing parameters ρ are profiled), and read the
/// resulting `reml_score` (the negative-log-evidence the outer loop minimises,
/// so κ̂ is its argmin). The exact same criterion the joint κ-fit minimised —
/// the only difference is which coordinates move — so κ̂ is a genuine stationary
/// point of this oracle. The statistics (profile-CI walk, interior κ=0 LR test)
/// are then the principled likelihood-set / Wilks constructions in
/// [`crate::geometry::curvature_estimand`].
///
/// `v_pp` (the initial Wald step size) is taken from a central finite difference
/// of `V_p` at κ̂; the CI itself is the exact χ²₁ likelihood crossing, not the
/// Wald ellipsoid, so this only sizes the first bracket step.
#[allow(clippy::too_many_arguments)]
pub fn curvature_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    family: LikelihoodSpec,
    options: &FitOptions,
    level: f64,
) -> Result<CurvatureInference, EstimationError> {
    let kappa_hat = get_constant_curvature_kappa(resolvedspec, term_idx).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "curvature_inference_forspec: term {term_idx} is not a constant-curvature smooth"
        ))
    })?;
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);

    // Profiled criterion oracle V_p(κ): pin κ, fit with κ-optimisation OFF so
    // only ρ is profiled, return the REML/LAML negative-log-evidence. Disabling
    // κ-opt routes `fit_term_collectionwith_spatial_length_scale_optimization`
    // straight to `fit_term_collection_forspec` at the spec's κ.
    let fixed_kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: false,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let v_p = |kappa: f64| -> Result<f64, String> {
        if !kappa.is_finite() {
            return Err(format!("V_p probed a non-finite κ = {kappa}"));
        }
        let mut probe_spec = resolvedspec.clone();
        match probe_spec
            .smooth_terms
            .get_mut(term_idx)
            .map(|t| &mut t.basis)
        {
            Some(SmoothBasisSpec::ConstantCurvature { spec, .. }) => spec.kappa = kappa,
            _ => {
                return Err(format!(
                    "V_p oracle: term {term_idx} is not a constant-curvature smooth"
                ));
            }
        }
        let fit = fit_term_collectionwith_spatial_length_scale_optimization(
            data,
            y.to_owned(),
            weights.to_owned(),
            offset.to_owned(),
            &probe_spec,
            family.clone(),
            options,
            &fixed_kappa_options,
        )
        .map_err(|e| format!("V_p fixed-κ fit at κ={kappa} failed: {e}"))?;
        let score = fit_score(&fit.fit);
        if score.is_finite() {
            Ok(score)
        } else {
            Err(format!(
                "V_p fixed-κ fit at κ={kappa} returned a non-finite score"
            ))
        }
    };

    // Wald step seed: central FD of V_p at κ̂ (only sizes the first bracket; the
    // CI is the exact likelihood crossing). Step a small fraction of the κ
    // window so the FD straddles κ̂ without leaving the chart.
    let h = (1e-3 * (kappa_max - kappa_min)).max(1e-4);
    let v_pp = match (v_p(kappa_hat + h), v_p(kappa_hat), v_p(kappa_hat - h)) {
        (Ok(vp), Ok(v0), Ok(vm)) => (vp - 2.0 * v0 + vm) / (h * h),
        _ => f64::NAN, // profile_ci_walk falls back to a default step
    };

    let ci = crate::geometry::curvature_estimand::profile_ci_walk(
        &v_p, kappa_hat, v_pp, kappa_min, kappa_max, level, 1e-4,
    )
    .map_err(EstimationError::InvalidInput)?;
    let flatness = crate::geometry::curvature_estimand::flatness_lr_test(&v_p, kappa_hat)
        .map_err(EstimationError::InvalidInput)?;

    Ok(CurvatureInference {
        term_idx,
        kappa_hat,
        ci,
        flatness,
    })
}


/// Provenance tag for the smooth-term significance correction (#1063): which
/// statistic the reported p-value is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrCorrection {
    /// A per-term likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)` that has
    /// been Bartlett-corrected with the exact Lawley factor `c = E[W]/d`
    /// (`W* = W/c`, referenced against `χ²_d`): second-order accurate.
    LawleyLr,
    /// No second-order correction was applied — either the family has no
    /// closed-form Lawley cumulant jets or the null refit did not converge — so
    /// the uncorrected `χ²_d` of the raw LR statistic stands.
    None,
}


impl SmoothLrCorrection {
    /// The serialized provenance label surfaced in the summary table.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrCorrection::LawleyLr => "lawley_lr",
            SmoothLrCorrection::None => "none",
        }
    }
}


/// The Bartlett-corrected per-term significance report for one penalized smooth
/// term (#1063). Unlike the summary table's Wood rank-truncated **Wald**
/// statistic, this is a genuine **likelihood-ratio** statistic from a
/// constrained refit (the smooth dropped), so the exact Lawley LR Bartlett
/// factor corrects the right quantity.
#[derive(Clone, Debug)]
pub struct SmoothTermLrInference {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// The uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)`,
    /// floored at zero (a non-negative LR by construction).
    pub statistic_lr: f64,
    /// Reference degrees of freedom `d` (the Wood truncation `tr(F)²/tr(F²)` on
    /// the term's influence block, falling back to the term EDF).
    pub ref_df: f64,
    /// Lawley LR Bartlett factor `c = E[W]/d = 1 + Δε/d` when computable, else
    /// `1.0` (no correction).
    pub bartlett_factor: f64,
    /// Bartlett-corrected statistic `W* = W / c`.
    pub statistic_corrected: f64,
    /// Uncorrected p-value `P(χ²_d > W)`.
    pub p_value_uncorrected: f64,
    /// Corrected p-value `P(χ²_d > W*)`; equals the uncorrected value when no
    /// correction was applied.
    pub p_value_corrected: f64,
    /// Which statistic the corrected p-value is built from.
    pub correction: SmoothLrCorrection,
}


/// The end-to-end per-term likelihood-ratio significance report for every
/// penalized (shape-unconstrained) smooth term in a fitted model, magically
/// Bartlett-corrected when the family carries closed-form Lawley cumulant jets
/// (#1063, follow-up to #939).
///
/// # Why an LR statistic (not the summary Wald)
///
/// The summary table's `wood_smooth_test` is Wood's rank-truncated **Wald**
/// statistic `T = β̂'Σ̂⁻β̂`. Lawley's ε corrects the **likelihood-ratio**
/// statistic, and under penalization the Wald form is already a weighted χ²
/// whose second-order mean is *not* `d + Δε` — dividing `T` by the LR factor
/// would correct the wrong statistic. The principled route (#1063 Option 1) is
/// to compute a real per-term LR statistic by a constrained refit and correct
/// *that*:
///
/// ```text
/// W = 2(ℓ_full − ℓ_null),   W* = W / c,   c = 1 + Δε/d,   p = P(χ²_d > W*).
/// ```
///
/// # Method
///
/// 1. Fit the full model and read `ℓ_full` and the per-term coefficient ranges /
///    EDF / influence block. The full design's column layout fixes the tested
///    block for the Lawley factor.
/// 2. For each penalized smooth term, refit a null model with that term dropped
///    from the spec; `W = max(2(ℓ_full − ℓ_null), 0)`.
/// 3. The reference d.f. `d` is the Wood truncation `tr(F)²/tr(F²)` on the
///    term's influence block (falling back to the term EDF) — the same `ref_df`
///    the summary Wald row reports.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`crate::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets, or a null refit that did not converge) the
///    uncorrected `χ²_d` stands with provenance `none` — never weakened.
///
/// Random-effect smooths and shape-constrained smooths are skipped (their tests
/// are not a central-χ² LR), matching the summary table's policy.
#[allow(clippy::too_many_arguments)]
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrInference>, EstimationError> {
    use crate::inference::lawley::{
        LAWLEY_PAIR_MATRIX_MAX_ROWS, known_scale_expected_jets_with_dispersion,
        lawley_lr_bartlett_factor,
    };

    let n = data.nrows();
    // Full fit: ℓ_full, the per-term coefficient ranges/EDF/influence, and the
    // full design whose column layout fixes each tested block for Lawley.
    let full = fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        family.clone(),
        options,
    )?;
    let ll_full = full.fit.log_likelihood;
    let p_total = full.design.design.ncols();
    let s_lambda = weighted_blockwise_penalty_sum(
        &full.design.penalties,
        full.fit.lambdas.as_slice().ok_or_else(|| {
            EstimationError::InvalidInput(
                "smooth_term_lr_inference: non-contiguous lambda vector".to_string(),
            )
        })?,
        p_total,
    );
    // Full design as a dense n×p array for the Lawley pair-matrix reduction.
    let full_design_dense = full.design.design.to_dense();
    let influence = full.fit.coefficient_influence();
    let edf_blocks = full.fit.edf_by_block().to_vec();
    let family_disp = lawley_dispersion_for_family(&family, &full.fit);

    // The penalty-block cursor walks the same block order the summary table
    // uses: random-effect ranges first (skipped here), then smooth terms.
    let mut penalty_cursor = full.design.random_effect_ranges.len();
    let mut out = Vec::<SmoothTermLrInference>::new();
    for (term_idx, design_term) in full.design.smooth.terms.iter().enumerate() {
        let k = design_term.penalties_local.len();
        let block_start = penalty_cursor;
        penalty_cursor += k;
        // Shape-constrained smooths get no central-χ² LR (cone-projected
        // boundary test); the summary table skips them too.
        if design_term.shape != ShapeConstraint::None {
            continue;
        }
        let coeff_range = design_term.coeff_range.clone();
        if coeff_range.start >= coeff_range.end || coeff_range.end > p_total {
            continue;
        }
        let edf = edf_blocks
            .get(block_start..block_start + k)
            .map(|block: &[f64]| block.iter().sum::<f64>())
            .unwrap_or(0.0);
        let ref_df = wood_reference_df(influence, &coeff_range).unwrap_or(edf.max(1e-12));
        if !(ref_df.is_finite() && ref_df > 0.0) {
            continue;
        }

        // Null model: drop this smooth term from the spec and refit. The term's
        // name pins which spec entry to remove (design and spec share names).
        let mut null_spec = resolvedspec.clone();
        let Some(spec_pos) = null_spec
            .smooth_terms
            .iter()
            .position(|t| t.name == design_term.name)
        else {
            continue;
        };
        null_spec.smooth_terms.remove(spec_pos);
        let null_fit = fit_term_collection_forspec(
            data,
            y,
            weights,
            offset,
            &null_spec,
            family.clone(),
            options,
        );
        let (statistic_lr, eta_null) = match null_fit {
            Ok(null) if null.fit.log_likelihood.is_finite() => {
                let w = (2.0 * (ll_full - null.fit.log_likelihood)).max(0.0);
                // η at the null fit: X_null β_null + offset (per-row linear
                // predictor; design-layout independent — Lawley reads it on the
                // full design rows).
                let mut eta = null.design.design.dot(&null.fit.beta);
                eta += &offset;
                (w, Some(eta))
            }
            _ => (f64::NAN, None),
        };

        let chi2 = statrs::distribution::ChiSquared::new(ref_df).ok();
        let p_uncorrected = match (chi2.as_ref(), statistic_lr.is_finite()) {
            (Some(dist), true) => {
                use statrs::distribution::ContinuousCDF;
                (1.0 - dist.cdf(statistic_lr)).clamp(0.0, 1.0)
            }
            _ => f64::NAN,
        };

        // Magic Bartlett correction: only when the LR statistic is finite, the
        // family has closed-form jets, n is in the resolvable regime, and the
        // factor is computable. Otherwise the uncorrected χ² stands.
        let mut bartlett_factor = 1.0;
        let mut statistic_corrected = statistic_lr;
        let mut p_corrected = p_uncorrected;
        let mut correction = SmoothLrCorrection::None;
        if let (Some(eta), true, true) = (
            eta_null.as_ref(),
            statistic_lr.is_finite(),
            n <= LAWLEY_PAIR_MATRIX_MAX_ROWS,
        ) {
            let kappas: Option<Vec<_>> = (0..n)
                .map(|i| {
                    known_scale_expected_jets_with_dispersion(&family, eta[i], family_disp)
                        .and_then(|jets| jets.kappas().ok())
                })
                .collect();
            if let (Some(kappas), Some(dist)) = (kappas, chi2.as_ref()) {
                if let Ok(c) = lawley_lr_bartlett_factor(
                    full_design_dense.view(),
                    &kappas,
                    Some(s_lambda.view()),
                    coeff_range.clone(),
                    ref_df,
                ) {
                    if c.is_finite() && c > 0.0 {
                        use statrs::distribution::ContinuousCDF;
                        bartlett_factor = c;
                        statistic_corrected = statistic_lr / c;
                        p_corrected = (1.0 - dist.cdf(statistic_corrected)).clamp(0.0, 1.0);
                        correction = SmoothLrCorrection::LawleyLr;
                    }
                }
            }
        }

        out.push(SmoothTermLrInference {
            name: design_term.name.clone(),
            term_idx,
            statistic_lr,
            ref_df,
            bartlett_factor,
            statistic_corrected,
            p_value_uncorrected: p_uncorrected,
            p_value_corrected: p_corrected,
            correction,
        });
    }
    Ok(out)
}


/// The dispersion `φ` Lawley needs for the family's cumulant scaling: Gaussian
/// `σ̂²`, Gamma `1/shape`, and `1` for the scale-free Poisson/Binomial.
fn lawley_dispersion_for_family(family: &LikelihoodSpec, fit: &UnifiedFitResult) -> f64 {
    match family.response {
        crate::types::ResponseFamily::Gaussian => {
            let sd = fit.standard_deviation;
            (sd * sd).max(f64::MIN_POSITIVE)
        }
        crate::types::ResponseFamily::Gamma => {
            let shape = fit.standard_deviation;
            if shape.is_finite() && shape > 0.0 {
                1.0 / shape
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}


/// Wood's rank-corrected reference d.f. `tr(F_jj)² / tr(F_jj²)` on the
/// coefficient-influence block `F = H⁻¹ X'WX` restricted to `coeff_range`. This
/// is the same reference the summary Wald row uses, so the corrected LR and the
/// Wald test reference the *same* `χ²_d`. Returns `None` when the influence
/// block is unavailable or degenerate.
fn wood_reference_df(influence: Option<&Array2<f64>>, coeff_range: &Range<usize>) -> Option<f64> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]);
    let tr = (0..block.nrows()).map(|i| block[[i, i]]).sum::<f64>();
    let tr2 = block.dot(&block).diag().sum();
    (tr.is_finite() && tr2.is_finite() && tr > 0.0 && tr2 > 0.0).then(|| (tr * tr / tr2).max(1e-12))
}

