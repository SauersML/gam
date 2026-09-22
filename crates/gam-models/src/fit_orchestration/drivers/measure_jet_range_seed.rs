// #2750 — the measure-jet representer range is SCREENED against the response
// before the outer ψ search refines it.
//
// `include!`d into `drivers/mod.rs` exactly like `constant_curvature_profile.rs`,
// whose machinery this file reuses: the λ-profiled Gaussian REML ψ-jet, driven
// by the workspace's outer engine over a derived window.
//
// ## The defect this closes
//
// `ℓ` is the ONE design-moving coordinate of a measure-jet term: it decides
// which span the representers occupy, and λ cannot move a span. #2761 made it
// REML-selected, but the selection is a local descent seeded at a pure geometry
// heuristic (the median nearest-node spacing), and the profiled criterion in
// `ln ℓ` is not unimodal — as `ℓ` grows past a few node spacings the Gaussian
// columns become collinear, the rank-revealing identifiability section drops
// columns, and the criterion steps. Measured end-to-end on
// `measure_jet_formula_fit_robustness_sweep` seed 1:
//
// ```text
//   ℓ (orig)   profiled V     held-out RMSE
//   0.0204      -234.5           0.0173     <- auto seed: LOCAL minimum
//   0.0345      -231.1           0.0185     <- barrier (+3.4)
//   0.0757      -236.4           0.0161
//   0.2163      -246.4           0.0110
//   0.8030      -256.3           0.0084     <- GLOBAL minimum, 21.7 deeper
//   1.0438      -198.5           0.0538     <- past the diameter: block collapses
//   s(x, bs="tps")  -247.4        0.0123
// ```
//
// The free search lands at `V = -234.6`: it never leaves the first basin. The
// criterion's ranking tracks the truth at every node, so the criterion is right
// and the search is caged — and the λ that comes back is a faithful readout of
// a range nothing could move. That is the "1-D fits select a too-large λ" of
// gam#2750, and the same cage is what leaves the term behind a same-size
// Matérn in gam#2761.
//
// ## What this does, and what it deliberately does NOT do
//
// The screen replaces a heuristic SEED with a data-chosen one. It does not
// replace the outer search, does not add a coordinate, and does not fire when
// the user pinned `length_scale=` (an explicit range is a request, not a seed —
// the same mgcv-`sp=` convention the range already follows).
//
// The screening criterion is the closed-form profiled Gaussian REML of the term
// ALONE against the response, with the double-penalty component off — the same
// object `constant_curvature_psi_profile_jet` differentiates, for the same
// reason: a full-collection multi-ρ solve at every trial range would multiply
// the cost of every measure-jet fit by the number of trials. It is minimized
// from each node of the term's scale band by the outer engine on its exact
// `ln ℓ` jet, and only certified searches compete (see
// `screen_measure_jet_range`). Two consequences are accepted openly:
//
//   * on a multi-term formula the screen ranks the term's own fit to `y`, not
//     its fit to the partial residual. It is a seed; the joint ψ/ρ search that
//     follows is the estimator.
//   * on a non-Gaussian family the screen is Gaussian-REML on the response
//     scale. Still strictly more informed than the geometry-only heuristic it
//     replaces, which never looks at `y` at all.

/// The screening criterion and its exact `ln ℓ` derivatives at one range:
/// `(V, V′, V″)` for the profiled Gaussian REML of `[1 | X(ℓ)]` with the term's
/// single jet-energy penalty, with λ profiled out.
///
/// `data` holds the term's feature columns in the STANDARDIZED frame the basis is
/// realized in, and `ln_ell` is a standardized log range, so this never has to
/// reason about the input-frame conversion the term-collection builder owns.
///
/// The design and penalty jets come from
/// [`gam_terms::basis::build_measure_jet_basis_psi_derivatives`], realized in the
/// basis chart at this ℓ. [`profiled_gaussian_reml_psi_jet`] turns them into
/// the λ-profiled jet: the envelope gradient and the Schur-complemented
/// curvature. That routine carries two coordinates `(κ, η)`. The screen has only
/// `η = ln ℓ`, so the κ blocks are exact zeros and only the η entries are read.
fn measure_jet_range_screen_jet(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    spec: &gam_terms::basis::MeasureJetBasisSpec,
    ln_ell: f64,
) -> Result<(f64, f64, f64), EstimationError> {
    let ell = ln_ell.exp();
    if !(ell.is_finite() && ell > 0.0) {
        crate::bail_invalid_estim!(
            "measure-jet range screen probed a non-representable range ln ℓ = {ln_ell}"
        );
    }
    let mut screen = spec.clone();
    screen.length_scale = ell;
    // The screen ranks SPANS. The null-component candidate is a second REML
    // coordinate, not a property of the span, and carrying it would make every
    // evaluation a multi-ρ solve; the shipped fit still gets it.
    screen.double_penalty = false;
    // Enrols the `ln ℓ` coordinate in the jet producer. The realized basis is
    // the same either way.
    screen.learn_length_scale = true;
    let basis = gam_terms::basis::build_measure_jet_basis(data, &screen)
        .map_err(EstimationError::from)?;
    if basis.active_penalties.len() != 1 {
        crate::bail_invalid_estim!(
            "measure-jet range screen expected one active penalty; got {}",
            basis.active_penalties.len()
        );
    }
    let jets = gam_terms::basis::build_measure_jet_basis_psi_derivatives(data, &screen)
        .map_err(EstimationError::from)?;
    let candidate = basis.active_penalties[0].info.original_index;
    let (Some(design_first), Some(design_second), Some(penalty_first), Some(penalty_second)) = (
        jets.design_first.first(),
        jets.design_second_diag.first(),
        jets.penalties_first
            .first()
            .and_then(|blocks| blocks.get(candidate)),
        jets.penalties_second_diag
            .first()
            .and_then(|blocks| blocks.get(candidate)),
    ) else {
        crate::bail_invalid_estim!(
            "measure-jet range screen: the ψ jets carry no ln ℓ coordinate for the primary penalty"
        );
    };
    let smooth_design = basis.design.to_dense();
    let (n, p) = smooth_design.dim();
    if n != y.len()
        || p == 0
        || design_first.dim() != (n, p)
        || design_second.dim() != (n, p)
        || penalty_first.dim() != (p, p)
        || penalty_second.dim() != (p, p)
    {
        crate::bail_invalid_estim!("measure-jet range screen: design and jet shapes disagree");
    }
    // The intercept column is ℓ-free, so it borders every jet with zeros.
    let border_design = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, p + 1));
        out.slice_mut(s![.., 1..]).assign(block);
        out
    };
    let border_penalty = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p + 1, p + 1));
        out.slice_mut(s![1.., 1..]).assign(block);
        out
    };
    let mut design = border_design(&smooth_design);
    design.column_mut(0).fill(1.0);
    let mut first = border_design(design_first);
    let mut second = border_design(design_second);
    let mut response = y.to_owned();
    // `√w` on every row: weighted Gaussian REML is ordinary REML on the scaled
    // rows, up to a `log|W|` constant that does not move with ℓ.
    if let Some(weights) = weights {
        for (row, &weight) in weights.iter().enumerate() {
            let root = weight.sqrt();
            design.row_mut(row).mapv_inplace(|value| value * root);
            first.row_mut(row).mapv_inplace(|value| value * root);
            second.row_mut(row).mapv_inplace(|value| value * root);
            response[row] *= root;
        }
    }
    let transform = whiten_to_identifiable_subspace(&design).ok_or_else(|| {
        EstimationError::InvalidInput(
            "measure-jet range screen: the representer design identifies no direction".to_string(),
        )
    })?;
    let chart = |block: &Array2<f64>| fast_ab(block, &transform);
    // The congruence is symmetric in exact arithmetic; make it so in floating
    // point as well, because the evaluator's spectral classification refuses a
    // matrix that is not exactly self-adjoint.
    let congruence = |block: &Array2<f64>| -> Array2<f64> {
        let half = fast_ab(block, &transform);
        let full = fast_atb(&transform, &half);
        (&full + &full.t()) * 0.5
    };
    let design = chart(&design);
    let first = chart(&first);
    let second = chart(&second);
    let penalty = congruence(&border_penalty(&basis.active_penalties[0].matrix));
    let penalty_first = congruence(&border_penalty(penalty_first));
    let penalty_second = congruence(&border_penalty(penalty_second));
    let kept = transform.ncols();
    let zero_design = Array2::<f64>::zeros((n, kept));
    let zero_penalty = Array2::<f64>::zeros((kept, kept));
    let jet = profiled_gaussian_reml_psi_jet(
        &design,
        &penalty,
        &PsiCoordinateBlocks {
            design_first: [&zero_design, &first],
            design_second: [&zero_design, &zero_design, &second],
            penalty_first: [&zero_penalty, &penalty_first],
            penalty_second: [&zero_penalty, &zero_penalty, &penalty_second],
        },
        response.view(),
    )?;
    Ok((jet.value, jet.gradient[1], jet.hessian[1][1]))
}

/// The chart `T` (`p × k`) that restricts a design to the subspace it actually
/// identifies, with the Gram the identity there: `(XT)ᵀ(XT) = I`.
///
/// ## Why this is needed at all
///
/// A Gaussian representer design LOSES RANK as the range grows — that is the
/// whole reason the range is worth screening — and the profiled evaluator
/// classifies the penalty in the `XᵀWX` metric (`L⁻¹ S L⁻ᵀ`, `L` the Cholesky
/// of the Gram). On a rank-deficient design that congruence turns an
/// exactly-PSD penalty into one with a negative eigenvalue and the evaluation
/// REFUSES: measured on the gam#2750 fixture, `-8.9e-8` at `ℓ = 0.166` and
/// `-2.8e-7` at `ℓ = 0.298`. A screen that refuses at exactly the ranges it
/// exists to reach would report the seed basin as the optimum for the second
/// time, which is the defect rather than a measurement of it.
///
/// ## Why it is free
///
/// For an INVERTIBLE `T`, `X → XT`, `S → TᵀST` leaves the profiled criterion
/// exactly unchanged: `log|Tᵀ(XᵀWX + λS)T|` and `log|λTᵀST|₊` both pick up
/// `2 ln|det T|` and the deviance is invariant, so the `2 ln|det T|` cancels in
/// the difference. Whitening is therefore a free change of chart, not a change
/// of model — and it makes the Gram exactly `I`, so the congruence above is the
/// identity and the evaluator sees the penalty as it was assembled. The same
/// `T` carries the ℓ-jets, `∂X → ∂X·T` and `∂S → Tᵀ∂S T`: on a stretch of ranges
/// where the identified rank does not change the criterion does not depend on
/// which chart of the identified subspace it is read in.
///
/// The map is only non-invertible where it drops directions, and dropping is
/// the honest reading there: the collection's own realization drops columns at
/// the same ranges. The cut is at `√ε` of the leading Gram eigenvalue — the
/// half-mantissa bar, i.e. the point past which a direction cannot survive
/// being squared into a Gram and inverted back out with any significant digits.
fn whiten_to_identifiable_subspace(design: &Array2<f64>) -> Option<Array2<f64>> {
    let p = design.ncols();
    if p == 0 {
        return None;
    }
    let gram = gam_linalg::faer_ndarray::fast_ata(design);
    let (values, vectors) = gam_linalg::faer_ndarray::strict_symmetric_eigh(
        &gram,
        // `fast_ata` accumulates one triangle and mirrors it.
        gam_linalg::roundoff::SymmetricAssembly::Mirrored,
        faer::Side::Lower,
    )
    .ok()?;
    let leading = values.iter().copied().fold(0.0_f64, |a, v| a.max(v));
    if !(leading.is_finite() && leading > 0.0) {
        return None;
    }
    let cut = leading * f64::EPSILON.sqrt();
    let kept: Vec<usize> = (0..p).filter(|&i| values[i] > cut).collect();
    if kept.is_empty() {
        return None;
    }
    let mut transform = Array2::<f64>::zeros((p, kept.len()));
    for (column, &index) in kept.iter().enumerate() {
        let inverse_root = values[index].sqrt().recip();
        for row in 0..p {
            transform[(row, column)] = vectors[(row, index)] * inverse_root;
        }
    }
    Some(transform)
}

/// The screened range for ONE measure-jet term, in standardized units, or
/// `None` when no search converged.
///
/// The range is searched over the term's own `ln ℓ` window
/// ([`gam_terms::basis::measure_jet_ln_range_window`]: the node-spacing floor and
/// the feasibility ceiling) by the workspace's outer engine, on the exact
/// criterion jet [`measure_jet_range_screen_jet`] supplies. The criterion is not
/// unimodal in `ln ℓ` (see the module docs), so one search from one seed is
/// caged in its first basin. So the engine starts from each node of the term's
/// realized scale band ([`gam_terms::basis::measure_jet_range_bracket`]): lengths
/// the basis already derived, not a lattice this screen chose. Only a search the
/// engine certifies converged, at an interior stationary point or at a face of
/// the window, may compete, and the lowest certified criterion wins. No node's
/// value is ever the answer.
///
/// This replaces a scan of the band nodes, an upward walk at the band's log step
/// while the value improved, and one parabolic step through three values
/// (#2902: SPEC rules 18 and 19).
fn screen_measure_jet_range(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    spec: &gam_terms::basis::MeasureJetBasisSpec,
) -> Option<f64> {
    use gam_problem::{Derivative, HessianValue, OuterEval};
    use gam_solve::rho_optimizer::OuterProblem;
    let bracket = gam_terms::basis::measure_jet_range_bracket(data, spec).ok()?;
    let (lower, upper) = gam_terms::basis::measure_jet_ln_range_window(data, spec).ok()?;
    if !(lower.is_finite() && upper.is_finite() && upper > lower) {
        return None;
    }
    // A range whose basis or jet cannot be realized is a property of that trial,
    // so the search retreats from it instead of abandoning the screen.
    let refuse = |error: EstimationError| EstimationError::TrialPointRefused {
        reason: error.to_string(),
    };
    // The screen's REML is over `[1 | X(ℓ)]`, and the Gaussian kernel keeps
    // `X(ℓ)` at one column per center for every ℓ > 0, so the coefficient
    // count the engine sizes its resolution by is fixed across the window.
    let p_coefficients = {
        let mut sizing = spec.clone();
        sizing.length_scale = bracket.nodes.iter().copied().find(|node| node.is_finite() && *node > 0.0)?;
        sizing.double_penalty = false;
        gam_terms::basis::build_measure_jet_basis(data, &sizing).ok()?.design.ncols() + 1
    };
    let mut best: Option<(f64, f64)> = None;
    for &node in &bracket.nodes {
        let start = node.ln();
        if !start.is_finite() {
            continue;
        }
        let problem = OuterProblem::new(1)
            .with_problem_size(y.len(), p_coefficients)
            .with_gradient(Derivative::Analytic)
            .with_hessian(gam_problem::DeclaredHessianForm::Dense)
            .with_bounds(Array1::from_vec(vec![lower]), Array1::from_vec(vec![upper]))
            .with_initial_rho(Array1::from_vec(vec![start.clamp(lower, upper)]));
        let mut objective = problem.build_objective(
            (),
            |_: &mut (), rho: &Array1<f64>| {
                measure_jet_range_screen_jet(data, y, weights, spec, rho[0])
                    .map(|(value, _, _)| value)
                    .map_err(refuse)
            },
            |_: &mut (), rho: &Array1<f64>| {
                let (cost, gradient, curvature) =
                    measure_jet_range_screen_jet(data, y, weights, spec, rho[0]).map_err(refuse)?;
                Ok(OuterEval {
                    cost,
                    gradient: Array1::from_vec(vec![gradient]),
                    hessian: HessianValue::Dense(Array2::from_elem((1, 1), curvature)),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError>>,
        );
        let Ok(result) = problem.run(&mut objective, "measure-jet representer range screen") else {
            continue;
        };
        if !result.converged() {
            continue;
        }
        let (ln_ell, value) = (result.rho[0], result.final_value);
        if ln_ell.is_finite()
            && value.is_finite()
            && best.is_none_or(|(_, incumbent)| value < incumbent)
        {
            best = Some((ln_ell, value));
        }
    }
    best.map(|(ln_ell, _)| ln_ell.exp())
}

/// The screening response for the SLOPE surface of a marginal-slope family.
///
/// The marginal-slope construction is `η_i = α(x_i) + β(x_i)·z_i` with a
/// binomial link `F`, so the slope surface `β` never appears in `E[y | x]`
/// and screening its span against `y` would rank spans by how well they carry
/// the MARGINAL surface — the wrong function. What `β` does appear in is the
/// conditional covariance of the response with the latent driver: for
/// `z ⟂ x` with `E[z] = 0`, `Var(z) = 1`,
///
/// ```text
///   Cov(y, z | x) = E[ z·F(α(x) + β(x)·z) ]
///                 = F'(α(x))·β(x)·E[z²] + O(β³·E[z⁴])
///                 = F'(α(x))·β(x) + O(β³),
/// ```
///
/// by expanding `F` about `α(x)` (the odd moments of `z` kill the even terms).
/// So the empirical cross-product `s_i = (y_i − ȳ)·(z_i − z̄)` has conditional
/// mean `F'(α(x))·β(x)` to first order: the planted slope surface times a
/// strictly positive, smooth modulation. A span that represents `β` well
/// represents `F'(α)·β` well, which is exactly the ranking a SEED needs — the
/// joint ψ/ρ search that follows is the estimator.
///
/// Two properties make this usable as-is rather than as an approximation to be
/// corrected: the profiled Gaussian REML the screen ranks with is invariant to
/// a global rescaling of its response (a rescale shifts the criterion by a
/// constant and moves `argmin` nowhere), so the unknown `E[z²]` factor and the
/// `F'` scale are both free; and `ȳ`/`z̄` are the weighted means, so a
/// weighted fit screens on its own measure.
///
/// The `z ⟂ x` step is the one assumption. When the latent driver correlates
/// with the covariates the surrogate picks up `(E[y|x] − ȳ)(E[z|x] − z̄)`,
/// which is a marginal-surface term. That is a bias in a SEED's ranking, not in
/// an estimand — the alternative on offer is screening against `y`, which is
/// that same wrong function with none of the right one added, or not screening
/// at all, which is the pure-geometry heuristic gam#2750 measured landing in
/// the wrong basin.
pub(crate) fn marginal_slope_screen_response(
    y: ArrayView1<'_, f64>,
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Option<Array1<f64>> {
    let n = y.len();
    if n == 0 || z.len() != n || weights.len() != n {
        return None;
    }
    if !y.iter().chain(z.iter()).all(|v| v.is_finite()) {
        return None;
    }
    let total: f64 = weights.iter().filter(|w| w.is_finite()).sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let mean = |v: ArrayView1<'_, f64>| -> f64 {
        v.iter()
            .zip(weights.iter())
            .map(|(a, w)| if w.is_finite() { a * w } else { 0.0 })
            .sum::<f64>()
            / total
    };
    let (y_bar, z_bar) = (mean(y), mean(z));
    let surrogate =
        Array1::from_iter((0..n).map(|i| (y[i] - y_bar) * (z[i] - z_bar)));
    // A degenerate driver (no variation left after centering) carries no
    // slope signal at all; screening on a constant would rank every span
    // identically and is better declined than reported. The surrogate mean is
    // hoisted: this runs once per fit on every row, so recomputing it inside the
    // scan would make a linear check quadratic.
    let surrogate_mean = surrogate.sum() / n as f64;
    let spread = surrogate
        .iter()
        .map(|v| (v - surrogate_mean).abs())
        .fold(0.0_f64, f64::max);
    (spread > 0.0).then_some(surrogate)
}

/// Screen every AUTO measure-jet representer range in `spec` against the
/// response and write the winner back, in the spec's own ORIGINAL input units.
///
/// Returns the number of terms whose range moved. A term is eligible only when
/// its range is the auto sentinel (`length_scale == 0.0`) and its quadrature is
/// not frozen: an explicit `length_scale=` is a request, and a frozen term is a
/// replay with nothing left to seed.
///
/// Failure to screen is never an error. Every refusal path leaves the term at
/// the geometry heuristic, which is exactly the pre-#2750 behaviour.
///
/// # Where this is reached from, and where it is not (#2754/#2761)
///
/// `length_scale == 0.0` has ONE resolver, and the whole point of that sentence
/// is that it holds no matter which family entry point a model takes. The
/// builder's geometry heuristic is what makes a miss silent — a path that never
/// screens still fits, just to a different span — so the reached/unreached
/// inventory belongs here, in the resolver, where it can be read in one place:
///
/// | entry point | screening response | status |
/// |---|---|---|
/// | `fit_standard_model` | `y` | screened (#2750) |
/// | `fit_bernoulli_marginal_slope_terms` | marginal: `y`; slope: `(y−ȳ)(z−z̄)` | screened (#2754) |
/// | `fit_transformation_normal` | `response` | screened (#2754) |
/// | the CTN cross-fit Stage-1 builder | full `response`, BEFORE the freeze | screened (#2754) |
/// | `fit_survival_marginal_slope_terms` | — | **not derived** |
/// | `fit_latent_survival_terms`, `fit_latent_binary_terms` | — | **not derived** |
/// | the `*_location_scale` families | — | **not derived** |
///
/// "Not derived" is a statement about the screening TARGET, not an oversight
/// left unexamined. Each of those families' surfaces enters a likelihood in
/// which the response is not a direct readout of the surface — a survival
/// marginal-slope block is modulated by the risk set carried in
/// `age_entry`/`age_exit`, and a location-scale scale block enters through a
/// variance rather than a mean — so screening them against the raw response
/// would rank spans by their fit to a function the surface is not. Inventing
/// one per family without a fixture that can grade it would be landing an
/// unmeasured modelling choice in five places at once; the honest state is that
/// they still take the geometry heuristic, and that this table says so.
pub(crate) fn seed_measure_jet_auto_ranges(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    spec: &mut TermCollectionSpec,
) -> usize {
    let n = data.nrows();
    if y.len() != n || weights.len() != n || n == 0 {
        return 0;
    }
    let positive_weights = weights.iter().all(|w| w.is_finite() && *w > 0.0);
    let mut seeded = 0usize;
    for term in spec.smooth_terms.iter_mut() {
        let SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec: mj,
            input_scale,
        } = &mut term.basis
        else {
            continue;
        };
        if mj.length_scale != 0.0 || mj.frozen_quadrature.is_some() || input_scale.is_some() {
            continue;
        }
        let Ok(columns) = select_columns(data, feature_cols) else {
            continue;
        };
        // The basis is realized in the auto-standardized frame; screen there,
        // then hand the winner back in original units because a FRESH spec's
        // `length_scale` is an original-units request (the scale contract's
        // asymmetric fresh/replay rule).
        let Ok(scale) =
            gam_terms::smooth::input_standardization::estimate_isotropic_scale(columns.view())
        else {
            continue;
        };
        let mut standardized = columns;
        scale.standardize(&mut standardized);
        let mut screen_spec = mj.clone();
        // Center selection happens in the standardized frame here, so a
        // resolved (already-standardized-by-the-builder) strategy would be
        // double-converted. Only auto strategies reach this path; anything
        // carrying explicit coordinates is left to the builder.
        if matches!(
            screen_spec.center_strategy,
            gam_terms::basis::CenterStrategy::UserProvided(_)
        ) {
            continue;
        }
        screen_spec.identifiability = gam_terms::basis::MeasureJetIdentifiability::CenterSumToZero;
        let screened = screen_measure_jet_range(
            standardized.view(),
            y,
            positive_weights.then_some(weights),
            &screen_spec,
        );
        let Some(ell) = screened else {
            continue;
        };
        let original = scale
            .to_original_units(gam_terms::StandardizedUnits::new(ell))
            .original_value();
        if original.is_finite() && original > 0.0 {
            mj.length_scale = original;
            seeded += 1;
        }
    }
    seeded
}

#[cfg(test)]
mod marginal_slope_screen_response_tests {
    use super::*;
    use gam_math::probability::{normal_cdf, normal_pdf};

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(state: &mut u64) -> f64 {
        ((splitmix(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(state: &mut u64) -> f64 {
        // `unit` draws from the open interval (0, 1), so `ln u1` is finite.
        let u1 = unit(state);
        let u2 = unit(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let mut sab = 0.0;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        for (x, y) in a.iter().zip(b) {
            sab += (x - ma) * (y - mb);
            saa += (x - ma) * (x - ma);
            sbb += (y - mb) * (y - mb);
        }
        sab / (saa.sqrt() * sbb.sqrt())
    }

    /// The derivation the slope screening surrogate rests on, checked
    /// against a probit sample rather than asserted: binning `s = (y−ȳ)(z−z̄)`
    /// by `x` must recover `F'(α(x))·β(x)`, NOT `α(x)`.
    ///
    /// This is the property that makes the surrogate the right ranking target
    /// for the slope span. Screening against `y` — the only other response
    /// on hand at that point in the fit — recovers `α` instead, which is a
    /// different function; the test pins the separation by scoring the binned
    /// surrogate against BOTH candidate truths on the same bins.
    #[test]
    fn slope_screen_surrogate_tracks_the_slope_surface_not_the_marginal_2754() {
        const N: usize = 200_000;
        const BINS: usize = 20;
        let alpha_true = |x: f64| -0.2 + 0.7 * (std::f64::consts::PI * x).sin();
        let beta_true = |x: f64| 0.2 + 0.9 * x;

        let mut state = 0x2754_2026_0811_0001_u64;
        let mut xs = vec![0.0; N];
        let mut zs = vec![0.0; N];
        let mut ys = vec![0.0; N];
        for i in 0..N {
            let x = unit(&mut state);
            let z = normal(&mut state);
            let p = normal_cdf(alpha_true(x) + beta_true(x) * z);
            xs[i] = x;
            zs[i] = z;
            ys[i] = f64::from(unit(&mut state) < p);
        }
        let weights = Array1::<f64>::ones(N);
        let y = Array1::from(ys.clone());
        let z = Array1::from(zs.clone());
        let surrogate =
            marginal_slope_screen_response(y.view(), z.view(), weights.view())
                .expect("a non-degenerate driver must produce a surrogate");
        assert_eq!(surrogate.len(), N);

        // Bin by x and average, so what is compared is the CONDITIONAL mean the
        // derivation is about rather than the per-row noise it sits under.
        let mut bin_sum = vec![0.0; BINS];
        let mut bin_count = vec![0.0; BINS];
        let mut bin_x = vec![0.0; BINS];
        for i in 0..N {
            let b = ((xs[i] * BINS as f64) as usize).min(BINS - 1);
            bin_sum[b] += surrogate[i];
            bin_x[b] += xs[i];
            bin_count[b] += 1.0;
        }
        let binned: Vec<f64> = (0..BINS).map(|b| bin_sum[b] / bin_count[b]).collect();
        let centers: Vec<f64> = (0..BINS).map(|b| bin_x[b] / bin_count[b]).collect();
        // The derivation's predicted conditional mean, and the response the
        // surrogate exists to avoid.
        let predicted: Vec<f64> = centers
            .iter()
            .map(|&x| normal_pdf(alpha_true(x)) * beta_true(x))
            .collect();
        let marginal: Vec<f64> = centers.iter().map(|&x| normal_cdf(alpha_true(x))).collect();

        let to_predicted = pearson(&binned, &predicted);
        let to_marginal = pearson(&binned, &marginal);
        println!(
            "[#2754 surrogate] corr(binned s, F'(alpha)*beta)={to_predicted:.4} \
             corr(binned s, marginal E[y|x])={to_marginal:.4}"
        );
        assert!(
            to_predicted > 0.95,
            "the slope screening surrogate must track F'(alpha)*beta (got {to_predicted:.4}); \
             the derivation behind `marginal_slope_screen_response` is what the screen's \
             ranking rests on"
        );
        assert!(
            to_predicted > to_marginal + 0.2,
            "the surrogate must separate the slope surface from the marginal one: \
             corr to F'(alpha)*beta = {to_predicted:.4} vs corr to E[y|x] = {to_marginal:.4}"
        );
    }

    /// A driver with no variation carries no slope signal, so the surrogate
    /// declines instead of handing the screen a constant every span fits equally.
    #[test]
    fn slope_screen_surrogate_declines_a_degenerate_driver_2754() {
        let y = Array1::from(vec![0.0, 1.0, 1.0, 0.0]);
        let z = Array1::from(vec![0.5, 0.5, 0.5, 0.5]);
        let w = Array1::<f64>::ones(4);
        assert!(
            marginal_slope_screen_response(y.view(), z.view(), w.view()).is_none(),
            "a constant latent driver must not be screened against"
        );
        // Weighted means, not arithmetic ones: a weighted fit screens on its own
        // measure. With all mass on rows 0 and 1 the centering must use those.
        let w2 = Array1::from(vec![1.0, 1.0, 0.0, 0.0]);
        let z2 = Array1::from(vec![-1.0, 1.0, 7.0, -7.0]);
        let s = marginal_slope_screen_response(y.view(), z2.view(), w2.view())
            .expect("a varying driver must produce a surrogate");
        // y_bar = 0.5, z_bar = 0.0 under w2.
        assert!((s[0] - 0.5).abs() < 1e-12, "s[0]={} != 0.5", s[0]);
        assert!((s[1] - 0.5).abs() < 1e-12, "s[1]={} != 0.5", s[1]);
    }
}

#[cfg(test)]
mod range_screen_tests {
    use super::*;

    /// A 1-D scatter with a deterministic irregular spacing, so the median nearest
    /// node spacing is a real median rather than a constant grid step.
    fn chart() -> Array2<f64> {
        Array2::from_shape_fn((240, 1), |(i, _)| {
            let t = i as f64 / 239.0;
            t + 0.04 * (7.0 * t).sin()
        })
    }

    fn response(data: &Array2<f64>) -> Array1<f64> {
        Array1::from_shape_fn(data.nrows(), |i| {
            (4.0 * data[(i, 0)]).sin() + 0.1 * (i as f64 * 1.618).sin()
        })
    }

    /// The fixture's center count, named once so the rounding floor below can
    /// state the criterion's log-determinant length without re-deriving it from
    /// the spec by a match the ban list does not allow a fallback arm for.
    const SCREEN_CENTERS: usize = 40;

    fn spec() -> gam_terms::basis::MeasureJetBasisSpec {
        gam_terms::basis::MeasureJetBasisSpec {
            center_strategy: gam_terms::basis::CenterStrategy::FarthestPoint {
                num_centers: SCREEN_CENTERS,
            },
            ..gam_terms::basis::MeasureJetBasisSpec::default()
        }
    }

    /// The band a central difference of `g` at step `h` may miss `g′` by, read
    /// off the differences themselves rather than chosen (#2902, SPEC rule 23).
    ///
    /// A central difference carries two errors and no others:
    ///
    /// * TRUNCATION. `D(h) = g′ + C·h² + O(h⁴)` with `C = g‴/6`. The same
    ///   difference at `2h` gives `D(2h) = g′ + 4C·h² + O(h⁴)`, so the observed
    ///   gap `D(2h) − D(h)` IS `3C·h²` and the truncation of `D(h)` is exactly a
    ///   third of it. `C` is never named: it is measured here, on this fixture,
    ///   at this range.
    /// * ROUNDING. The two values differ in their leading digits by `O(h)`, so
    ///   whatever band they carry is amplified by `1/(2h)`. Their band is
    ///   Wilkinson's `γ_k` for the criterion's longest accumulation — the
    ///   `n`-term deviance sum plus the `p`-term log-determinant of the evaluator
    ///   this screen calls — against the values' own scale.
    ///
    /// The returned band is the WHOLE gap plus the rounding floor, not the third
    /// of it that truncation actually costs. That factor of three is not a safety
    /// margin someone picked either: it is the `(2h)²/h²` scaling of the
    /// differencing error itself, so a miss outside this band is not explained by
    /// an `h²` truncation at any of the three steps, nor by the arithmetic.
    fn central_difference_band(
        near: f64,
        far: f64,
        values: [f64; 2],
        step: f64,
        rows: usize,
        columns: usize,
    ) -> f64 {
        let truncation = (far - near).abs();
        let scale = values[0].abs() + values[1].abs();
        let rounding = gam_linalg::roundoff::accumulation_growth(rows + columns) * scale
            / (2.0 * step.abs());
        truncation + rounding
    }

    /// #2902: the screen's exact `ln ℓ` jet against central differences of its own
    /// value and gradient, at ranges a few node spacings above the window floor
    /// where the identified rank does not change under the step. The certified
    /// multi-start screen then returns a range inside the window.
    ///
    /// Both comparisons are made at two steps, `h` and `2h`, because a single
    /// step cannot say how much of a miss is the difference's own truncation. The
    /// bar is then [`central_difference_band`] at each point rather than a
    /// relative tolerance: a fixed `1e-3·(1 + |V′|)` accepted the frozen-`Z` jet's
    /// `1.5e-3` relative error at one range while refusing nothing at the other,
    /// so it measured the criterion's scale and not the jet.
    #[test]
    fn range_screen_jet_matches_central_differences_2902() {
        let data = chart();
        let y = response(&data);
        let spec = spec();
        let rows = data.nrows();
        let columns = SCREEN_CENTERS;
        let (lower, upper) =
            gam_terms::basis::measure_jet_ln_range_window(data.view(), &spec).expect("window");
        let jet = |ln_ell: f64| {
            measure_jet_range_screen_jet(data.view(), y.view(), None, &spec, ln_ell)
                .expect("the screen jet at a representable range")
        };
        // A PROBE step, not an acceptance bar. Nothing is accepted against it:
        // the band below is computed FROM the probe, so a step too large shows up
        // as a large measured truncation and a step too small as a large
        // amplified rounding floor. Either way the bar widens rather than a
        // defect hiding. `1e-4` in `ln ℓ` keeps `[at − 2h, at + 2h]` inside one
        // identified rank on this fixture, which is what the derivative exists on
        // at all.
        let step = 1e-4;
        for offset in [0.4_f64, 1.0] {
            let at = lower + offset;
            let (_, first, second) = jet(at);
            let (up_value, up_first, _) = jet(at + step);
            let (down_value, down_first, _) = jet(at - step);
            let (far_up_value, far_up_first, _) = jet(at + 2.0 * step);
            let (far_down_value, far_down_first, _) = jet(at - 2.0 * step);
            let first_difference = (up_value - down_value) / (2.0 * step);
            let first_difference_far = (far_up_value - far_down_value) / (4.0 * step);
            let second_difference = (up_first - down_first) / (2.0 * step);
            let second_difference_far = (far_up_first - far_down_first) / (4.0 * step);
            let first_band = central_difference_band(
                first_difference,
                first_difference_far,
                [up_value, down_value],
                step,
                rows,
                columns,
            );
            let second_band = central_difference_band(
                second_difference,
                second_difference_far,
                [up_first, down_first],
                step,
                rows,
                columns,
            );
            assert!(
                (first - first_difference).abs() <= first_band,
                "ln ℓ = {at}: V′ {first} vs central difference {first_difference} \
                 (at 2h: {first_difference_far}), band {first_band:e}"
            );
            assert!(
                (second - second_difference).abs() <= second_band,
                "ln ℓ = {at}: V″ {second} vs central difference {second_difference} \
                 (at 2h: {second_difference_far}), band {second_band:e}"
            );
        }
        let screened = screen_measure_jet_range(data.view(), y.view(), None, &spec)
            .expect("at least one certified range search");
        let ln_screened = screened.ln();
        // The screen answers in ℓ and the window is stated in `ln ℓ`, so the only
        // slack a containment check is owed is that round trip: `exp` then `ln`,
        // each correctly rounded, against the bound's own formation. Eight
        // roundings is a generous count of that path, and lands about five
        // decades under the `1e-9` this replaces, which stood for no arithmetic
        // at all.
        let round_trip =
            gam_linalg::roundoff::accumulation_growth(8) * (1.0 + ln_screened.abs().max(1.0));
        assert!(
            ln_screened >= lower - round_trip && ln_screened <= upper + round_trip,
            "the screened range ln ℓ = {ln_screened} left the window [{lower}, {upper}] \
             by more than the {round_trip:e} the ℓ round trip costs"
        );
    }
}
