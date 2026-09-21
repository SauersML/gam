/// Per-iteration BFGS step budget on the ψ axes (log κ and the anisotropy log-scales):
/// one doubling of the kernel length scale. `opt`'s per-axis trust budget exists because
/// ρ (natural step ≈ 5 in log λ) and ψ step on very different natural magnitudes, and a
/// single radius either starves ρ or lets the kernel scale jump orders of magnitude per
/// iteration (#1053/#1066/#1069).
const SPATIAL_PSI_BFGS_STEP_CAP: f64 = std::f64::consts::LN_2;

// The ψ hyper-direction builders and latent/analytic-penalty objective terms (moved verbatim, line limit).
include!("spatial_hyper_dirs.rs");

// The single-block exact-joint and latent-coordinate design caches (moved verbatim, line limit).
include!("spatial_single_block_caches.rs");

/// The λ-selection domain of the joint spatial search, derived per ρ
/// coordinate from the term's own design-relative penalty spectrum (#2812):
/// `[ln(ε γ_min), ln(γ_max / ε)]` over the positive generalized eigenvalues
/// `γ_j` of the block's design Gram against its penalty, intersected with the
/// representable log-strength range. Below the lower edge the penalty is under
/// the round-off of the data curvature (the term is unpenalized to working
/// precision); above the upper edge the data curvature is under the round-off
/// of the penalty (the term sits on its null space to working precision). A
/// search that reaches an edge has found a structural result, not a wall.
///
/// This replaces the `±12` prior box and its per-coordinate fallback to the
/// engine's `±30` rail. MEASURED on the #2760 ladder (noiseless 1-D Duchon
/// `y = sin(t)`, 12 centers, 5 penalties): REML drives `λ̂` down as `n` grows
/// and the incumbents crossed `−12` one at a time, 4 of 5 pasted onto the wall
/// at `n = 1 000 … 8 000` and all 5 at `n = 16 000` (coordinate 0 at
/// `−12.347`), where the joint gradient at the wall was `+1.484` against a
/// stationarity bound of `1.030` — a feasible-set defect reported as a
/// length-scale failure. A domain read off the spectrum has no such wall: the
/// incumbent of a fit inside its own domain is inside this one, and the only
/// edges are the two structural limits.
///
/// A coordinate whose penalty geometry cannot be projected (no positive
/// penalty eigenvalue, or a design that does not reach its range) keeps the
/// precision box `[ln ε, ln(1/ε)]` around unit strength; so does a coordinate
/// beyond the penalties the design carries, with a warning, since the layout
/// then disagrees with the fit.
pub(crate) fn joint_rho_resolvability_domain(
    design: &DesignMatrix,
    penalties: &[gam_terms::smooth::BlockwisePenalty],
    rho_dim: usize,
) -> (Array1<f64>, Array1<f64>) {
    let precision_box = gam_solve::estimate::rho_domain::precision_box();
    let mut lower = Array1::<f64>::from_elem(rho_dim, precision_box.0);
    let mut upper = Array1::<f64>::from_elem(rho_dim, precision_box.1);
    if rho_dim > penalties.len() {
        log::warn!(
            "[spatial-kappa] joint rho domain: {rho_dim} coordinates but {} penalty blocks; \
             the coordinates past the blocks keep the precision box",
            penalties.len()
        );
    }
    let dense = design.to_dense();
    for (k, penalty) in penalties.iter().take(rho_dim).enumerate() {
        let range = penalty.col_range.clone();
        if range.end > dense.ncols() || range.start >= range.end {
            continue;
        }
        let block = dense.slice(ndarray::s![.., range.start..range.end]);
        let gram = block.t().dot(&block);
        let Some(gammas) = gam_custom_family::penalty_range_gammas_from_gram(&gram, &penalty.local)
        else {
            continue;
        };
        let Some((lo, hi)) = gam_custom_family::resolvability_interval(&gammas) else {
            continue;
        };
        lower[k] = lo.max(gam_problem::LOG_STRENGTH_MIN);
        upper[k] = hi.min(gam_problem::LOG_STRENGTH_MAX);
    }
    (lower, upper)
}

/// The ρ domain of an exact-joint search over a custom family's blocks realized
/// at the seed: the #2812 law `fit_custom_family` applies to those same blocks
/// ([`per_block_resolvability_rho_domain`](crate::custom_family::per_block_resolvability_rho_domain)),
/// so a coordinate is searched on one domain whichever route fits it. Every
/// block that owns a ρ coordinate is realized, including the ones the driver's
/// term collections never carry (a survival time or link-wiggle block, an
/// influence absorber, a noise ridge). A layout whose free penalties do not
/// number `rho_dim` is refused rather than given a box (#2902 item 15).
pub(crate) fn realized_blocks_rho_domain(
    blocks: &[crate::custom_family::ParameterBlockSpec],
    options: &crate::custom_family::BlockwiseFitOptions,
    rho_dim: usize,
) -> Result<(Array1<f64>, Array1<f64>), FitFailure> {
    let (lower, upper) = crate::custom_family::per_block_resolvability_rho_domain(blocks, options)
        .map_err(|error| FitFailure::from(error).context("exact-joint rho resolvability domain"))?;
    if lower.len() != rho_dim || upper.len() != rho_dim {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            format!(
                "exact-joint rho resolvability domain: the realized blocks carry {} free \
                 penalties for {rho_dim} rho coordinates",
                lower.len()
            ),
        ));
    }
    Ok((lower, upper))
}

/// The per-coordinate ρ domain of one penalized block given as a design and
/// its penalties as full-block matrices (a survival time block, a prepared
/// score-warp or link-deviation block): each penalty's resolvability interval
/// against the block's Gram, intersected with the representable strength
/// range; a penalty without design curvature keeps the precision box.
pub(crate) fn penalized_block_rho_domain<'a>(
    design: &DesignMatrix,
    penalties: impl IntoIterator<Item = &'a Array2<f64>>,
) -> (Vec<f64>, Vec<f64>) {
    let dense = design.to_dense();
    let gram = dense.t().dot(&dense);
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for penalty in penalties {
        let interval = (penalty.nrows() == gram.nrows() && penalty.ncols() == gram.ncols())
            .then(|| gam_custom_family::penalty_range_gammas_from_gram(&gram, penalty))
            .flatten()
            .and_then(|gammas| gam_custom_family::resolvability_interval(&gammas));
        let (lo, hi) = gam_solve::estimate::rho_domain::coordinate_domain(interval, None);
        lower.push(lo);
        upper.push(hi);
    }
    (lower, upper)
}

/// What the joint `[rho, psi]` spatial route did, as three answers rather than
/// two (#2748).
///
/// The route used to return `Option<FittedTermCollectionWithSpec>`, and `None`
/// carried two facts that call for opposite responses:
///
/// * the route could not be BUILT — no `psi` hyper-directions exist for these
///   terms, so a caller that requires kappa optimisation has nothing; and
/// * the route RAN, produced a candidate, graded it against the shipped
///   scalar-route score and correctly DECLINED it. Its own log line says
///   "keeping the incumbent fit and treating joint kappa optimization as a
///   no-op for this fit" -- a successful decision, taken by the one routine
///   that holds both candidates.
///
/// The sole caller mapped `None` to
/// `"spatial kappa optimization is unavailable for one or more eligible spatial
/// terms"` and failed the whole fit, so the second case killed fits the route
/// had just decided were fine. Measured on `geo_disease_eas_matern_k6`,
/// `papuan_oce4_matern_k6` and `papuan_oce_matern_k12`: after #2748 cleared the
/// rho-Hessian refusal, this is what they died of instead.
///
/// Same species as #2578 (a verdict channel whose absence-of-observation was
/// read as an observation) and #2737 (a timeout branch and an error branch that
/// both ended in a bare `exit 1`): one channel, two verdicts, and the consumer
/// reading the wrong one.
enum JointSpatialKappaOutcome {
    /// The joint route ran and its candidate improved the shipped score.
    Optimized(Box<FittedTermCollectionWithSpec>),
    /// The joint route ran to completion and declined its own candidate. The
    /// incumbent scalar-route fit is the better of the two, and is what ships.
    DeclinedKeepIncumbent {
        baseline_score: f64,
        optimized_score: f64,
        /// The κ trial's own timing. The route ran, so its cost is reported
        /// whichever fit ships.
        kappa_timing: SpatialLengthScaleOptimizationTiming,
    },
    /// The joint route could not be built for these terms at all.
    Unavailable,
}

fn try_exact_joint_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    spatial_terms: &[usize],
) -> Result<JointSpatialKappaOutcome, EstimationError> {
    if spatial_terms.is_empty() {
        return Ok(JointSpatialKappaOutcome::Unavailable);
    }
    // Fail loud on nonsensical κ options rather than letting them propagate
    // silent NaNs (e.g. inverted min/max inverts the BFGS window, negative
    // scales produce NaN logs). This is the first function on every outer-κ
    // path; downstream paths assume validated options.
    kappa_options
        .validate()
        .map_err(EstimationError::InvalidInput)?;

    if try_build_spatial_log_kappa_hyper_dirs(data, resolvedspec, &best.design, spatial_terms)?
        .is_none()
    {
        if !constant_curvature_term_indices(resolvedspec).is_empty() {
            log::info!(
                "[#1464-trace] try_exact_joint RETURNED None (hyper_dirs unavailable); \
                 κ̂ comes from a NON-joint path"
            );
        }
        return Ok(JointSpatialKappaOutcome::Unavailable);
    }
    if !constant_curvature_term_indices(resolvedspec).is_empty() {
        log::info!(
            "[#1464-trace] try_exact_joint ENTERED for {} spatial term(s); CC present",
            spatial_terms.len()
        );
    }

    let ExactJointSpatialSeed {
        kind,
        rho_dim,
        dims_per_term,
        theta0,
        lower,
        upper,
    } = exact_joint_spatial_seed(data, resolvedspec, best, spatial_terms)?;
    let has_constant_curvature_term = !constant_curvature_term_indices(resolvedspec).is_empty();

    // ───────────────────────────────────────────────────────────────────────
    //  Both coordinate kinds drive the SAME exact joint optimizer
    //  (`run_exact_joint_spatial_optimization`): the unified REML evaluator with
    //  ext_coords for joint [ρ, ψ] optimization, with analytic gradient +
    //  Hessian flowing through the
    //  AnisoBasisPsiDerivatives / SpatialPsiDerivative → DirectionalHyperParam →
    //  HyperCoord pipeline for Newton/BFGS quadratic convergence. The only
    //  difference is the coordinate kind: anisotropic carries one ψ per axis per
    //  term, isotropic one log-κ per term. `outer_strategy` handles the
    //  centralized degradation path when the analytic Hessian is unavailable.
    // ───────────────────────────────────────────────────────────────────────
    let (theta_star, joint_final_value, joint_seed_value, kappa_timing) = run_exact_joint_spatial_optimization(
        kind,
        data,
        y,
        weights,
        offset,
        resolvedspec,
        &best.design,
        family.clone(),
        options,
        spatial_terms,
        &dims_per_term,
        &theta0,
        &lower,
        &upper,
        rho_dim,
        kappa_options,
    )?;

    let baseline_score = fit_score(&best.fit);

    // Compare the joint optimizer's certified cost (final_value at theta*)
    // against the baseline. Tolerance ≥ options.tol because both endpoints
    // are outer-BFGS approximations accurate to options.tol; a tighter
    // gate would reject true improvements due to floating-point noise.
    let accept_tol = options.tol.max(1e-8 * baseline_score.abs()).max(1e-12);
    // The monotonicity certificate used to be ONE comparison —
    // `joint_final_value <= baseline_score + accept_tol` — spanning TWO
    // independent facts, and it therefore could not say which of them had
    // failed. `joint_final_value` is this route's criterion at θ*;
    // `baseline_score` is the scalar-ρ route's `fit_score` at θ0. A refusal
    // could mean either "the optimizer ended above where it started" (a solver
    // regression) or "the two routes disagree about the criterion at the SAME
    // point" (a criterion inconsistency, which no amount of optimizer work can
    // fix). `run_exact_joint_spatial_optimization` already evaluates its own
    // criterion at θ0 to prime the evaluator, so both facts are available; state
    // them separately so the refusal names the defect it found.
    //
    // The route-agreement bound is the SAME derived quantity as the acceptance
    // bound — no second tolerance is introduced. It is two-sided because a route
    // disagreement is a disagreement in either direction, whereas the descent
    // contract is one-sided by construction.
    //
    // MEASURED (2026-07-31, while working #2644). This half fires on
    // `misc::broad_sweep_batch_h::matern_low_n_does_not_crash` and the numbers
    // say the disagreement is SYSTEMATIC, not noise:
    //
    //   run 30602192415  joint_seed=2.787395886872e0  baseline=2.787395850137e0
    //   run 30619084852  joint_seed=2.787290435812e0  baseline=2.787290399076e0
    //   local, 4e7fd2ae1 joint_seed=2.787290435812e0  baseline=2.787290399076e0
    //
    // `gap = 3.674e-8` on all three — bit-identical across two nightlies on
    // different runners AND a local run, at two different `theta0`. A
    // deterministic `1.318e-8` RELATIVE offset against a `1e-8` relative
    // tolerance, i.e. this refusal misses by 1.32x and would miss by 1.32x every
    // time. So it is NOT the `O(ε·κ)` criterion-conditioning family #2644 turned
    // out to be (that one is scattered and moves run to run); the two routes are
    // evaluating slightly different functions, and the difference is reproducible
    // enough to bisect directly by differencing the two criteria term by term at
    // `theta0`. It survived the #2644 root-scale log-determinant work unchanged,
    // which rules that mechanism out rather than leaving it open.
    if !joint_seed_value.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "exact joint spatial optimization could not evaluate its own criterion at the \
             seed (seed_value={joint_seed_value:.6e}), so neither its descent nor its \
             agreement with the scalar-rho route is checkable; baseline={baseline_score:.6e}"
        )));
    }
    // The gap, emitted UNCONDITIONALLY rather than only when it happens to
    // exceed (gam#2760, the same reasoning as `[CERTIFICATE-BOUND]`). The gate
    // is a RELATIVE `1e-8` on a criterion whose magnitude grows with `n`, so
    // whether it fires is a question about a trend, and a number a reader can
    // only see on the run that already failed cannot show a trend. Measured on
    // the #2760 ladder: `5.965e-8` relative at `n = 8 000`, i.e. the gap is
    // itself above the `√ε ≈ 1.49e-8` forward-error scale this file's own
    // `outer_arithmetic_gradient_floor` calls the resolution of a
    // matrix-factorization REML score — so it is not roundoff, and reading it at
    // every `n` is how the residual half of #2671 gets bisected.
    log::info!(
        "[spatial-kappa] route agreement at theta0: joint_seed={joint_seed_value:.12e} \
         baseline={baseline_score:.12e} gap={:.6e} ({:.6e} relative) \
         agreement_tolerance={accept_tol:.6e} ({}) sqrt_eps_scale={:.6e}",
        joint_seed_value - baseline_score,
        (joint_seed_value - baseline_score) / baseline_score.abs().max(f64::MIN_POSITIVE),
        if (joint_seed_value - baseline_score).abs() > accept_tol {
            "REFUSES"
        } else {
            "admits"
        },
        baseline_score.abs() * f64::EPSILON.sqrt(),
    );
    // WARNS, and no longer REFUSES (gam#2760). The gate's own complaint is
    // right — "the joint search is minimizing a different function than the one
    // its result is graded against" — and the response to it is to grade the
    // result on the function it will SHIP with, which this routine can do
    // exactly (see the acceptance comparison after the accept-fit below), not
    // to refuse a whole REML fit on a cross-route scalar comparison no fixed
    // relative constant can denominate.
    //
    // Why no constant can. `joint_seed_value` and `baseline_score` are two
    // INDEPENDENT assemblies of a REML criterion whose forward error is the
    // `O(ε·κ)` conditioning family #2644 named, and `κ` here is the penalized
    // Hessian's. MEASURED on the #2760 ladder, same fixture, five rungs:
    //
    //   n =  1000   gap = −1.386e-13 relative     baseline rho: one coordinate at −RHO_BOUND
    //   n =  2000   gap = −1.667e-13 relative     one coordinate at −RHO_BOUND
    //   n =  4000   gap = +5.475e-13 relative     one coordinate at −RHO_BOUND
    //   n =  8000   gap = +5.965e-08 relative     TWO coordinates at −RHO_BOUND
    //   n = 16000   gap = +5.968e-08 relative     TWO coordinates at −RHO_BOUND
    //
    // Five orders in one step, and the step is not in `n`: it is the rung at
    // which a SECOND penalty block reaches `λ = e^−30 ≈ 9.4e-14` and stops
    // contributing to `H = XᵀWX + S_λ` at working precision. `log|H|` is then a
    // sum of logs across the raw Duchon Gram's ~1e15 spectrum and the two
    // assemblies part company at exactly the scale `ε·κ` predicts. A `1e-8`
    // relative demand cannot be met there by any correct implementation, and a
    // constant loose enough to admit it would no longer catch the formula
    // difference #2671 found (`5.047e-5` relative) that this gate exists for.
    //
    // So the number keeps its full decomposition and its loudness, and the
    // REFUSAL moves to a comparison both sides of which come from ONE route.
    if (joint_seed_value - baseline_score).abs() > accept_tol {
        log::warn!(
            "[spatial-kappa] the joint and scalar-rho routes disagree about the criterion AT \
             THE SAME POINT theta0: joint_seed={joint_seed_value:.12e}, \
             baseline={baseline_score:.12e}, gap={:.3e} ({:.3e} relative) against a \
             {accept_tol:.3e} agreement tolerance. Two independent assemblies of one \
             criterion; their forward error is O(eps*kappa) in the penalized Hessian, so this \
             is only evidence of a formula difference when it exceeds what the conditioning \
             explains. The joint result is graded on the SHIPPED scalar-route score below, \
             which is a like-for-like comparison; this line is the record that the two \
             assemblies parted company (joint_final={joint_final_value:.12e}, \
             theta_checkpoint={:?}).",
            joint_seed_value - baseline_score,
            (joint_seed_value - baseline_score) / baseline_score.abs().max(f64::MIN_POSITIVE),
            theta_star.to_vec(),
        );
    }
    // Descent contract. Measured on `b8745892a`, this is the half that actually
    // fires (`seed=6.613467e1, final=6.613469e1, initial=6.613467e1` on the
    // binomial-logit Matérn fixture): the two routes agree at θ0 to every
    // printed digit, and the joint search ends ABOVE the point it started from.
    //
    // The optimizer's certificate is a LOCAL, possibly boundary, stationarity
    // statement at θ* (`theta_checkpoint=[30.0, …]` sits on `RHO_BOUND`), so it
    // says nothing about θ0 — a certified stationary point of a nonconvex
    // criterion is routinely worse than a different feasible point. Refusing the
    // whole REML fit here treated "the search moved to a worse local optimum" as
    // an internal failure, when this routine has already EVALUATED both
    // candidates and can simply return the better one. `run_exact_joint_…`
    // returns its terminal iterate, not its best, so the driver is the first
    // place that holds both numbers.
    //
    // Keeping the better candidate makes the routine's own contract — "joint
    // κ optimization never returns a point worse than its seed" — true by
    // construction rather than checked after the fact. The regression is still
    // a solver defect and must stay visible, so it is logged with both values
    // and the rejected checkpoint rather than silently absorbed.
    let (theta_star, joint_final_value) = if joint_final_value > joint_seed_value + accept_tol {
        log::warn!(
            "[spatial-kappa] the exact joint search terminated ABOVE its own seed \
             (seed={joint_seed_value:.12e}, final={joint_final_value:.12e}, \
             regression={:.3e}, acceptance_tolerance={accept_tol:.3e}); its terminal \
             certificate is local/boundary at theta={:?} and does not dominate the seed, \
             so the seed is kept and joint kappa optimization is a no-op for this fit. \
             A descent method returning a point worse than its start is a solver defect \
             in its own right and this line is the record of it.",
            joint_final_value - joint_seed_value,
            theta_star.to_vec(),
        );
        (theta0.clone(), joint_seed_value)
    } else {
        (theta_star, joint_final_value)
    };

    // The refit reads its warm start in log λ (#1340); a physical λ there starts
    // the search at the wrong point, clamped onto a domain face when λ is large.
    let selected_log_lambdas = theta_star.slice(s![..rho_dim]).to_owned();
    gam_problem::validate_log_strengths(selected_log_lambdas.iter().copied()).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "selected joint spatial smoothing coordinate is outside the canonical log-strength domain: {error}"
        ))
    })?;
    let log_kappa_star =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta_star, rho_dim, dims_per_term);
    // #1464 diagnostic (ban-clean): the joint solver's CONVERGED ψ-tail κ for each
    // CC term — the value BEFORE any spec write-back / freeze / readback. If this
    // is negative for the hyperbolic dataset but `get_constant_curvature_kappa`
    // later returns +1.08, the railing is a POST-SOLVE clamp/readback, not the
    // optimiser. If this is itself +1.08, the joint solver railed past the pin.
    if has_constant_curvature_term {
        let star = log_kappa_star.as_array();
        let dims = log_kappa_star.dims_per_term();
        for (slot, &term_idx) in spatial_terms.iter().enumerate() {
            if constant_curvature_term_spec(resolvedspec, term_idx).is_some() {
                let off: usize = dims[..slot].iter().sum();
                log::info!(
                    "[#1464-trace] term {term_idx}: joint solver CONVERGED ψ-tail κ = {} \
                     (this is the optimised candidate; joint_final_value={joint_final_value})",
                    star[off]
                );
            }
        }
    }
    let optimized_spec = log_kappa_star.apply_tospec(resolvedspec, spatial_terms)?;
    let optimized = fit_term_collection_forspecwith_heuristic_log_lambdas(
        data,
        y,
        weights,
        offset,
        &optimized_spec,
        selected_log_lambdas.as_slice(),
        family.clone(),
        options,
    )?;

    // THE ACCEPTANCE COMPARISON (gam#2760). Both sides are `fit_score` of a
    // scalar-route fit — the incumbent at `theta0` and the accept-fit at `θ*` —
    // so this is the one comparison the two routes can make like for like, in
    // one arithmetic, on the quantity that actually ships. The cross-route
    // comparison at `theta0` above states whether the two assemblies agree; THIS
    // states whether optimizing κ improved the fit, which is what the routine
    // promises. A joint search that lands somewhere the shipped score does not
    // like is a no-op, exactly as the descent contract above treats a search
    // that lands above its own seed — and for the same reason: this routine
    // holds both candidates and can simply return the better one.
    let optimized_score = fit_score(&optimized.fit);
    if optimized_score > baseline_score + accept_tol {
        log::warn!(
            "[spatial-kappa] joint kappa optimization did not improve the SHIPPED scalar-route \
             score (baseline={baseline_score:.12e}, at theta_star={optimized_score:.12e}, \
             regression={:.3e}, acceptance_tolerance={accept_tol:.3e}); keeping the incumbent \
             fit and treating joint kappa optimization as a no-op for this fit. Both numbers \
             are `fit_score` of a scalar-route fit, so unlike the theta0 cross-route line this \
             comparison is like-for-like and a regression here is a real one.",
            optimized_score - baseline_score,
        );
        return Ok(JointSpatialKappaOutcome::DeclinedKeepIncumbent {
            baseline_score,
            optimized_score,
            kappa_timing,
        });
    }

    // Stamp reml_score with joint_final_value so downstream consumers see a
    // score consistent with the gate decision; the refit serves as a
    // β/inference harvester at the certified (ρ*, ψ*).
    let mut fit = optimized.fit;
    fit.set_criterion(Some(joint_final_value));
    let optimized_result = FittedTermCollectionWithSpec {
        fit,
        design: optimized.design,
        resolvedspec: optimized_spec,
        kappa_timing: Some(kappa_timing),
    };

    Ok(JointSpatialKappaOutcome::Optimized(Box::new(
        optimized_result,
    )))
}

/// The joint [ρ, ψ] route's seed and search box, derived from the scalar-ρ
/// incumbent: its log strengths and length scales, projected into their derived
/// domains.
struct ExactJointSpatialSeed {
    kind: SpatialHyperKind,
    rho_dim: usize,
    dims_per_term: Vec<usize>,
    theta0: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
}

fn exact_joint_spatial_seed(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    spatial_terms: &[usize],
) -> Result<ExactJointSpatialSeed, EstimationError> {
    let rho_dim = best.fit.lambdas.len();

    // #1464 used to widen the over-smoothing side of a hand `±12` box to the
    // engine's `±30` rail when a constant-curvature `curv()` term was present,
    // because that kernel's REML optimum at the +κ side is a large smoothing
    // λ. The joint ρ domain is now derived per coordinate from the term's own
    // spectrum (`joint_rho_resolvability_domain`, #2812), so that basin is
    // inside the domain whenever the data resolve it, for every term alike.

    // Compute per-term dimensionality for anisotropic terms.
    let dims_per_term = spatial_dims_per_term(resolvedspec, spatial_terms);
    let use_aniso = has_aniso_terms(resolvedspec, spatial_terms);

    // Build initial ψ values and bounds, using aniso-aware constructors
    // when any term has d > 1 axes. Bounds are tied to each term's center
    // geometry (r_min, r_max) so κ cannot saturate at an upper bound that
    // has no relationship to the data's distance scale.
    let log_kappa0 = if use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(resolvedspec, spatial_terms)
    } else {
        SpatialLogKappaCoords::from_length_scales(resolvedspec, spatial_terms)
    };
    // If the user/spec did not set a length_scale, re-seed ψ at the midpoint
    // of the data-derived window instead of the constructors' placeholder.
    let mut log_kappa0 = log_kappa0
        .reseed_from_data(data, resolvedspec, spatial_terms)
        .map_err(EstimationError::BasisError)?;
    // Constant curvature is selected once, continuously, before the baseline
    // fit. The full joint solve therefore profiles only nuisance ρ (and any
    // non-curvature spatial coordinates) at that certified κ. User-pinned and
    // estimated values share the same fixed-coordinate treatment, including κ=0.
    let has_constant_curvature_term = !constant_curvature_term_indices(resolvedspec).is_empty();
    let mut cc_profiled_values: Vec<(usize, f64)> = Vec::new();
    if has_constant_curvature_term {
        for (slot, &term_idx) in spatial_terms.iter().enumerate() {
            if constant_curvature_term_spec(resolvedspec, term_idx).is_none() {
                continue;
            }
            let kappa = get_constant_curvature_kappa(resolvedspec, term_idx)
                .expect("constant-curvature term exposes its kappa");
            log_kappa0.set_scalar_slot(slot, kappa);
            cc_profiled_values.push((slot, kappa));
        }
    }
    let log_kappa_lower = if use_aniso {
        SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            data,
            resolvedspec,
            spatial_terms,
            &dims_per_term,
        )
    } else {
        SpatialLogKappaCoords::lower_bounds_from_data(data, resolvedspec, spatial_terms)
    }
    .map_err(EstimationError::BasisError)?;
    let log_kappa_upper = if use_aniso {
        SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            data,
            resolvedspec,
            spatial_terms,
            &dims_per_term,
        )
    } else {
        SpatialLogKappaCoords::upper_bounds_from_data(data, resolvedspec, spatial_terms)
    }
    .map_err(EstimationError::BasisError)?;
    let mut log_kappa_lower = log_kappa_lower;
    let mut log_kappa_upper = log_kappa_upper;
    for &(slot, kappa) in &cc_profiled_values {
        log_kappa_lower.set_scalar_slot(slot, kappa);
        log_kappa_upper.set_scalar_slot(slot, kappa);
        log::info!("[spatial-kappa] slot {slot}: profiling rho at certified kappa={kappa}");
    }
    // Project seed onto data-derived bounds; spec.length_scale is a hint,
    // not a hard constraint. BFGS requires theta0 ∈ [lower, upper].
    // `{lower,upper}_bounds*_from_data` build the SEARCH box, which already
    // contains the incumbent length scale (#2454), and `reseed_from_data` puts
    // every unscaled term at the window midpoint, so for a log-κ slot this
    // projection has nothing to do.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    // #2726: ASSERT the `AT THE SAME POINT theta0` premise instead of stating it
    // in prose. The monotonicity certificate below grades this route's criterion
    // at θ0 against `fit_score(&best.fit)`, the scalar-ρ incumbent — a comparison
    // that only means anything if the ψ half of θ0 is the ψ `best` was realized
    // at. It once was not: the seed constructors projected `length_scale` onto a
    // caller window while `best` was fit from the raw value, so the two routes
    // sat `ln 10` apart and the refusal reported a criterion defect for a
    // feasible-set mismatch. `resolvedspec` is frozen from `best.design`, so its
    // `length_scale` IS the incumbent's realized scale, and with no caller window
    // (#2902) the seed is −ln of that scale. This check therefore passes by
    // construction, and any projection site reintroduced between the incumbent
    // and the seed fails here instead of twelve orders of magnitude downstream.
    for (slot, &term_idx) in spatial_terms.iter().enumerate() {
        if constant_curvature_term_spec(resolvedspec, term_idx).is_some()
            || measure_jet_term_spec(resolvedspec, term_idx).is_some()
        {
            continue;
        }
        let Some(incumbent) = get_spatial_length_scale(resolvedspec, term_idx) else {
            // No explicit incumbent scale: `reseed_from_data` owns this seed and
            // there is no realized ψ for it to be equal to.
            continue;
        };
        if !(incumbent.is_finite() && incumbent > 0.0) {
            continue;
        }
        let psi_incumbent = -incumbent.ln();
        let axes = log_kappa0.term_slice(slot);
        if axes.is_empty() {
            continue;
        }
        let psi_bar = axes.iter().sum::<f64>() / axes.len() as f64;
        // Forward-error bound for the arithmetic actually performed: the d-term
        // mean above (η_a are centered, so ψ̄ is exact for a scalar axis and
        // accumulates only summation roundoff otherwise), plus one rounding for
        // the box projection, which can move the seed to the nearest
        // representable edge when the incumbent sits exactly on a face. Not a
        // tolerance knob — the failure it guards against is a whole projection
        // step, `ln 10` in the measured case, some 5e14x above this bound.
        let max_abs_axis = axes.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let mean_roundoff =
            (axes.len() as f64 + 1.0) * f64::EPSILON * (max_abs_axis + psi_incumbent.abs());
        if (psi_bar - psi_incumbent).abs() > mean_roundoff {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "exact joint spatial optimization would grade its criterion at a psi the \
                 scalar-rho incumbent was never realized at (term {term_idx}): \
                 seed_psi_bar={psi_bar:.17e}, incumbent_psi={psi_incumbent:.17e}, \
                 delta={:.6e}, incumbent_length_scale={incumbent:.17e}. theta0 is \
                 not shared, so the monotonicity certificate below would compare two \
                 different functions (#2726).",
                psi_bar - psi_incumbent,
            )));
        }
    }

    // The joint ρ domain is derived from the incumbent's own design and
    // penalties (#2812), and the setup projects the incumbent's seed into it.
    let rho_seed = best.fit.lambdas.mapv(f64::ln);
    let (rho_lower, rho_upper) =
        joint_rho_resolvability_domain(&best.design.design, &best.design.penalties, rho_dim);
    let setup = ExactJointHyperSetup::new(
        rho_seed,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();
    log::info!(
        "[spatial-kappa] joint rho domain per coordinate: lower={:.3?} upper={:.3?} seed={:.3?}",
        lower.iter().take(rho_dim).copied().collect::<Vec<_>>(),
        upper.iter().take(rho_dim).copied().collect::<Vec<_>>(),
        theta0.iter().take(rho_dim).copied().collect::<Vec<_>>(),
    );
    let kind = if use_aniso {
        SpatialHyperKind::Anisotropic
    } else {
        SpatialHyperKind::Isotropic
    };
    Ok(ExactJointSpatialSeed {
        kind,
        rho_dim,
        dims_per_term,
        theta0,
        lower,
        upper,
    })
}

/// Coordinate kind for the exact joint spatial hyperparameter optimizer.
///
/// Anisotropic and isotropic spatial terms drive the *same* joint `[ρ, ψ]`
/// optimizer: identical outer-Hessian policy, identical
/// `ExternalJointHyperEvaluator` wiring, identical convergence processing, and
/// an identical `eval_full / eval_efs / eval_cost`
/// inner loop that routes ψ through `try_build_spatial_log_kappa_hyper_dirs`.
/// The coordinate *kind* distinguishes per-axis log scales (ψ_a) from one
/// log-κ per term and selects diagnostic labels. It also tells the startup
/// policy when an isotropic Matérn point has already won the explicit certified
/// endpoint comparison, in which case that point owns the sole joint start.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpatialHyperKind {
    Anisotropic,
    Isotropic,
}

impl SpatialHyperKind {
    /// Stable diagnostic prefix used in every `log::*` line and as the
    /// `ExternalJointHyperEvaluator` / cost-only label root.
    fn label(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "spatial-aniso-joint",
            SpatialHyperKind::Isotropic => "spatial-iso-joint",
        }
    }

    /// Human-readable adjective for error strings ("anisotropic" / "isotropic").
    fn adjective(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "anisotropic",
            SpatialHyperKind::Isotropic => "isotropic",
        }
    }

    /// Name of the directional coordinate being optimized ("psi" / "kappa"),
    /// used only in hyper-direction construction error messages.
    fn coord_name(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "psi",
            SpatialHyperKind::Isotropic => "kappa",
        }
    }
}

/// Shared context for the exact joint spatial optimizer's closures. Holds the
/// realized-design cache and the joint REML evaluator, plus the coordinate
/// `kind` whose only effect is the diagnostic label routed into the cost-only
/// evaluation path. The `eval_full / eval_efs / eval_cost` methods are the
/// single source of truth for both anisotropic and isotropic spatial terms.
struct SpatialFrozenGlmInputs {
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    family: LikelihoodSpec,
}

/// True when the frozen-weight GLM ψ-tensor (#1111 / #1033 mechanism (c)) is a
/// faithful first-Fisher-step provider for this family.
///
/// The mechanism freezes the working weight `w = w(η_warm)` and working response
/// `z = z(η_warm)` once per outer ψ-sweep, so it is exact for ANY family whose
/// per-iteration PIRLS reduces to a Gaussian working model with a SINGLE
/// canonical Fisher weight at a FIXED dispersion — i.e. the one-parameter
/// exponential families Binomial, Poisson, Gamma, and Negative-Binomial (the
/// θ-fixed running-seed weight `W = μθ/(θ+μ)` is a clean per-row Fisher weight).
/// These are precisely the "Poisson/Binomial/etc" families the issue names.
///
/// Tweedie and Beta jointly estimate an extra dispersion parameter that moves
/// the working weight outside the frozen snapshot, so the frozen-W stand-in is
/// not faithful for them and they keep the exact per-trial PIRLS rebuild.
/// Gaussian-identity is served by the (exact, converged) `PsiGramTensor` lane,
/// and Royston-Parmar is the survival path, neither of which routes here.
fn frozen_glm_tensor_eligible_family(family: &LikelihoodSpec) -> bool {
    !family.is_gaussian_identity()
        && matches!(
            &family.response,
            ResponseFamily::Binomial
                | ResponseFamily::Poisson
                | ResponseFamily::Gamma
                | ResponseFamily::NegativeBinomial { .. }
        )
}

struct SpatialJointContext<'d> {
    data: ArrayView2<'d, f64>,
    rho_dim: usize,
    kind: SpatialHyperKind,
    cache: SingleBlockExactJointDesignCache<'d>,
    evaluator: gam_solve::estimate::ExternalJointHyperEvaluator<'d>,
    frozen_glm_inputs: Option<SpatialFrozenGlmInputs>,
    frozen_glm_psi_bounds: Option<(f64, f64)>,
    frozen_glm_tensor: Option<gam_solve::glm_sufficient_lane::FrozenWeightGramTensor>,
    frozen_glm_tensor_attempted: bool,
    /// #1033: memo of the frozen-W trial Fisher weights keyed on the warm β that
    /// produced them. `stage_frozen_glm_trial_statistics` runs on EVERY κ trial
    /// (every cost / gradient probe), and the only β-dependent quantity it needs
    /// is the current Fisher weight vector `W(η)` (η = Xβ + offset) for the
    /// drift check and the n-free gradient soundness gate. Computing `W` is an
    /// O(n·p) GEMV + O(n) family evaluation; β only changes when the inner solve
    /// re-converges (after an accepted outer step), so recomputing it on every
    /// same-β probe was a redundant per-trial n-touch. Cache `(β, W)` and reuse
    /// `W` whenever β is unchanged — the GEMV runs once per distinct β, i.e.
    /// O(outer steps), not O(trials). `None` until the first compute / when no
    /// frozen-W inputs are installed.
    frozen_glm_weight_memo: Option<(Array1<f64>, Array1<f64>)>,
    /// #2481: failed value-probe attempts, split by the stage that refused.
    /// Recoverable trial-point failures remain ordinary `Ok(+∞)` domain refusals;
    /// every other failure is propagated through the typed outer-objective seam.
    /// The counters retain stage attribution for successful runs that encountered
    /// recoverable walls before converging.
    value_realization_failures: usize,
    value_evaluation_failures: usize,
    /// `Some((slow_path_resets, nfree_skip_row_touches))` read at the instant
    /// `begin_exact_polish` retired the #1033b n-free surrogate; `None` while
    /// the SEARCH is still running (gam#2760).
    ///
    /// The `[KAPPA-PHASE-SUMMARY]` counters — and the #1868 / #1264 gates that
    /// consume them — are statements about the SEARCH: "an in-window
    /// hyperparameter TRIAL touches only k×k objects", "the exact-lane fallback
    /// COUNT is n-independent". The exact polish is not a trial phase; it is the
    /// deliberate, once-per-fit transition onto the exact streamed criterion,
    /// and every one of its evaluations takes the O(n) lane BY CONSTRUCTION.
    /// Charging those to the search's counters would make a correctness repair
    /// read as a broken skip. Splitting them here keeps both facts reportable
    /// and neither hidden: the search's counts stay exactly what they measured
    /// before, and the polish's own O(n) cost is published beside them.
    nfree_polish_boundary: Option<(u64, u64)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NfreeSkipGateStatus {
    shape: bool,
    value: bool,
    gradient: bool,
    penalty: bool,
    revision: bool,
    second_order: bool,
}

impl NfreeSkipGateStatus {
    fn would_skip(self, require_gradient: bool) -> bool {
        self.shape
            && self.value
            && (!require_gradient || self.gradient)
            && self.penalty
            && self.revision
            && !self.second_order
    }
}

fn nfree_skip_gate_status_from_parts(
    shape: bool,
    covers_value: bool,
    covers_skip: bool,
    covers_gradient: bool,
    penalty: bool,
    revision: bool,
    allow_second_order: bool,
    require_gradient: bool,
) -> NfreeSkipGateStatus {
    NfreeSkipGateStatus {
        shape,
        // A value-only cost probe consumes only the Chebyshev Gram value; it
        // does not expose a beta/row-space object, so the #1264 reduced-basis
        // skip witness is not part of the value soundness certificate. Requiring
        // `covers_skip` here forces harmless cost probes across basis-rotation
        // seams onto `reset_surface`, reintroducing an O(n) pass into the κ
        // trial loop. Gradient probes still require the skip witness because
        // they return a stationary beta/gradient in the frozen reduced basis.
        value: shape && covers_value && (!require_gradient || covers_skip),
        gradient: shape && (!require_gradient || covers_gradient),
        penalty,
        revision,
        second_order: allow_second_order,
    }
}

/// Give the value and derivative lanes one typed answer for a failed trial.
///
/// A refusal at this trial travels as an error whose variant answers
/// `is_trial_point_infeasible()`, so every outer consumer classifies it by
/// variant and the refusal's reason reaches the outer log (#2735). A bare
/// `BasisError` — the design cannot be built at this hyperparameter — is a
/// refusal here but graded fatal by `is_trial_point_infeasible`, so it is carried
/// as `TrialPointRefused` with its own message. Every other error is returned
/// unchanged and stays fatal.
fn classify_spatial_value_probe_failure(error: EstimationError) -> EstimationError {
    if error.is_trial_point_infeasible() || !is_recoverable_trial_point_error(&error) {
        error
    } else {
        EstimationError::TrialPointRefused {
            reason: format!("the design cannot be realized at this trial point: {error}"),
        }
    }
}

impl<'d> SpatialJointContext<'d> {
    fn nfree_skip_gate_status(
        &self,
        theta: &Array1<f64>,
        allow_second_order: bool,
        require_gradient: bool,
    ) -> NfreeSkipGateStatus {
        let shape = theta.len() == self.rho_dim + 1;
        let (covers_value, covers_skip, covers_gradient) = if shape {
            let psi = theta[self.rho_dim];
            (
                self.evaluator.psi_gram_tensor_covers(psi),
                self.evaluator.psi_gram_tensor_covers_skip(psi),
                self.evaluator.psi_gram_tensor_covers_gradient(psi),
            )
        } else {
            (false, false, false)
        };
        nfree_skip_gate_status_from_parts(
            shape,
            covers_value,
            covers_skip,
            covers_gradient,
            self.evaluator.supports_nfree_penalty_rekey(),
            self.evaluator.nfree_fast_path_revision().is_some(),
            allow_second_order,
            require_gradient,
        )
    }

    fn frozen_glm_working_state(
        &self,
        beta: &Array1<f64>,
    ) -> Result<Option<(Array1<f64>, Array1<f64>)>, EstimationError> {
        let Some(inputs) = self.frozen_glm_inputs.as_ref() else {
            return Ok(None);
        };
        if beta.len() != self.cache.design().design.ncols() {
            return Ok(None);
        }
        let mut eta = self.cache.design().design.matrixvectormultiply(beta);
        if eta.len() != inputs.offset.len() {
            crate::bail_invalid_estim!(
                "frozen GLM tensor warm-state row mismatch: eta={}, offset={}",
                eta.len(),
                inputs.offset.len()
            );
        }
        eta += &inputs.offset;
        let obs = evaluate_standard_familyobservations(
            inputs.family.clone(),
            None,
            None,
            None,
            &inputs.y,
            &inputs.weights,
            &eta,
        )?;
        let mut working_response = obs.eta.clone();
        for i in 0..working_response.len() {
            // A row with positive Fisher weight has the working response
            // `η + score/W` exactly. A row with none enters every weighted sum
            // with zero weight, so its working response stays `η`.
            let wi = obs.fisherweight[i];
            if wi > 0.0 {
                working_response[i] += obs.score[i] / wi;
            }
        }
        Ok(Some((obs.fisherweight, working_response)))
    }

    /// #1033: the trial Fisher weight vector `W(η)` for `beta`, memoized on
    /// `beta`. `stage_frozen_glm_trial_statistics` consults `W` on EVERY κ trial
    /// (drift check + n-free gradient soundness gate) but `W` is a deterministic
    /// function of β (η = Xβ + offset), and β only changes when the inner solve
    /// re-converges — many cost / gradient probes share one β. Recompute the
    /// O(n·p) working state only when β differs from the memoized key; otherwise
    /// return the cached weights. Returns `None` exactly when
    /// `frozen_glm_working_state` does (no frozen-W inputs / β shape mismatch).
    fn frozen_glm_trial_weights(
        &mut self,
        beta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        if let Some((memo_beta, memo_w)) = self.frozen_glm_weight_memo.as_ref()
            && memo_beta.len() == beta.len()
            && memo_beta
                .iter()
                .zip(beta.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(Some(memo_w.clone()));
        }
        match self.frozen_glm_working_state(beta)? {
            Some((current_w, _)) => {
                self.frozen_glm_weight_memo = Some((beta.clone(), current_w.clone()));
                Ok(Some(current_w))
            }
            None => Ok(None),
        }
    }

    fn ensure_frozen_glm_tensor(
        &mut self,
        theta: &Array1<f64>,
        warm_beta: Option<&Array1<f64>>,
    ) -> Result<(), EstimationError> {
        if self.frozen_glm_tensor.is_some() || self.frozen_glm_tensor_attempted {
            return Ok(());
        }
        let Some((psi_lo, psi_hi)) = self.frozen_glm_psi_bounds else {
            return Ok(());
        };
        if theta.len() != self.rho_dim + 1 {
            self.frozen_glm_tensor_attempted = true;
            return Ok(());
        }
        let Some(beta) = warm_beta else {
            return Ok(());
        };
        let Some((frozen_w, working_z)) = self.frozen_glm_working_state(beta)? else {
            self.frozen_glm_tensor_attempted = true;
            return Ok(());
        };
        let theta_probe_base = theta.clone();
        let rho_dim = self.rho_dim;
        // Build through the evaluator so the frozen-W Gram is assembled in the
        // SAME conditioned `x_fit` column frame the inner PIRLS solve uses
        // (the evaluator owns the ψ-invariant parametric conditioning). Disjoint
        // mutable borrows of `cache` (in the realizer) and `evaluator` (the
        // build host) — both fields of `self` — exactly as the Gaussian
        // `build_and_set_psi_gram_tensor` site does.
        let Self {
            cache, evaluator, ..
        } = self;
        let tensor = evaluator.build_frozen_glm_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                cache.ensure_theta(&theta_probe).map_err(|e| e.to_string())?;
                Ok(cache.design().design.clone())
            },
            frozen_w.view(),
            working_z.view(),
            psi_lo,
            psi_hi,
        );
        self.cache
            .ensure_theta(theta)?;
        self.frozen_glm_tensor_attempted = true;
        if let Some(tensor) = tensor {
            self.frozen_glm_tensor = Some(tensor);
            log::info!(
                "[STAGE] {} certified frozen-W GLM ψ tensor over [{psi_lo:.3}, {psi_hi:.3}]",
                self.kind.label(),
            );
        } else {
            log::info!(
                "[STAGE] {} frozen-W GLM ψ tensor did not certify over [{psi_lo:.3}, {psi_hi:.3}]",
                self.kind.label(),
            );
        }
        Ok(())
    }

    fn stage_frozen_glm_trial_statistics(
        &mut self,
        theta: &Array1<f64>,
        warm_beta: Option<&Array1<f64>>,
        allow_gradient: bool,
    ) -> Result<(), EstimationError> {
        let kind = self.kind;
        let mut staged_gram: Option<Array2<f64>> = None;
        let mut staged_deriv: Option<(Array2<f64>, Array1<f64>)> = None;
        if theta.len() == self.rho_dim + 1 {
            let psi = theta[self.rho_dim];
            // Compute the β-memoized trial Fisher weights up front (mutable
            // self borrow) so the immutable `self.frozen_glm_tensor` borrow
            // below does not alias it. `frozen_glm_trial_weights` recomputes the
            // O(n·p) working state only on a β change, so a same-β probe pays
            // nothing here (#1033). Only proceed when a tensor is installed and
            // covers this ψ — otherwise skip the weight compute entirely.
            let tensor_covers = self
                .frozen_glm_tensor
                .as_ref()
                .is_some_and(|t| t.contains(psi));
            let current_w = if tensor_covers {
                match warm_beta {
                    Some(beta) => self.frozen_glm_trial_weights(beta)?,
                    None => None,
                }
            } else {
                None
            };
            if let (Some(tensor), Some(current_w)) =
                (self.frozen_glm_tensor.as_ref(), current_w.as_ref())
            {
                const FROZEN_GLM_WEIGHT_DRIFT_RTOL: f64 = 1e-3;
                if tensor.weight_drift_within(current_w.view(), FROZEN_GLM_WEIGHT_DRIFT_RTOL) {
                    staged_gram = Some(tensor.gram_at(psi));
                    log::debug!(
                        "[STAGE] {} trial at psi={psi:.6}: serving frozen-W GLM \
                         first-Fisher-step XᵀWX n-free (weight drift within tol)",
                        kind.label(),
                    );
                }
                if allow_gradient
                    && tensor.contains_for_gradient(psi)
                    && let Some((dgram_dpsi, drhs_dpsi)) =
                        tensor.gradient_pair_if_sound(psi, current_w.view())
                {
                    staged_deriv = Some((dgram_dpsi, drhs_dpsi));
                    log::debug!(
                        "[STAGE] {} trial at psi={psi:.6}: serving frozen-W GLM \
                         ψ-gradient (∂G/∂ψ, ∂b/∂ψ) n-free (gradient weight drift within \
                         tight tol); B_j stays exact",
                        kind.label(),
                    );
                }
            }
        }
        self.evaluator.stage_glm_first_step_gram(staged_gram);
        self.evaluator.stage_glm_psi_gram_deriv(staged_deriv);
        Ok(())
    }

    /// Full evaluation on the current realized design + hyper_dirs.
    fn eval_full(
        &mut self,
        theta: &Array1<f64>,
        order: gam_solve::rho_optimizer::OuterEvalOrder,
        analytic_outer_hessian_available: bool,
    ) -> Result<(f64, Array1<f64>, gam_problem::HessianValue), EstimationError> {
        use gam_solve::rho_optimizer::OuterEvalOrder;
        // A rejected value trial never needs spatial derivative slabs. Keep
        // its cache entry value-only so a later accepted-point request still
        // computes the gradient rather than reading a placeholder (#2735).
        if matches!(order, OuterEvalOrder::Value) {
            return self.eval_cost(theta).map(|cost| {
                (cost, Array1::zeros(theta.len()), gam_problem::HessianValue::Unavailable)
            });
        }
        let allow_second_order = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        if let Some(eval) = self.cache.memoized_eval(theta) {
            let cached_satisfies_order = !allow_second_order || eval.2.is_analytic();
            if cached_satisfies_order {
                return Ok(eval);
            }
        }
        let kind = self.kind;
        // #1033: the per-trial n×k design re-realization (`ensure_theta` →
        // `apply_log_kappa`) plus the downstream n-row reconditioning
        // (`reset_surface`) are the LAST n-passes in the certified κ loop. They
        // are redundant on the Gaussian-identity certified path: the inner
        // Gaussian PLS reads its `XᵀWX(ψ)/XᵀW(y−offset)(ψ)` entirely from the
        // ψ-keyed `GaussianFixedCache` the certified tensor installs (zero row
        // access), and the ψ-gradient HyperCoord is served from the k-space
        // `(∂G/∂ψ, ∂b/∂ψ)` tensor derivatives — never the n×k ∂X/∂ψ slab. So when
        //   (a) this is the single design-moving ψ coordinate (`rho_dim + 1`),
        //   (b) the certified ψ-Gram tensor covers ψ for BOTH the value lane
        //       (`psi_gram_tensor_covers`) AND the gradient window
        //       (`psi_gram_tensor_covers_gradient`) — so neither channel reads
        //       the realized rows,
        //   (c) this eval is gradient-only (`!allow_second_order`) — the exact
        //       outer-Hessian `B_j` path DOES read the slab, so a Hessian trial
        //       must keep a faithful (freshly realized) design, and
        //   (d) the evaluator has a pinned canonical slow-path revision — i.e.
        //       a prior slow-path eval already built a faithful reference surface,
        //       which `prepare_eval_state` will reuse while re-installing the
        //       ψ-keyed cache,
        // we SKIP `ensure_theta`. The realizer revision then does not advance, so
        // `prepare_eval_state` takes its design-revision fast path by receiving
        // that pinned revision back: it skips `reset_surface` + the n×k
        // `apply_to_design`, keeps the reference surface, and re-keys the
        // `GaussianFixedCache` to this ψ. The hyper_dirs built below are a pure
        // function of (data, frozen spec, column layout) — ψ-invariant — so they
        // are bit-identical whether or not the design was re-realized, and the
        // tensor branch never reads their n×k slab anyway. Net: criterion +
        // gradient + inner solve come from k-space statistics only, with no
        // per-trial O(n·k) pass.
        //
        // When ANY gate clause fails (non-Gaussian, off-window, off the gradient
        // sub-window, a Hessian eval, or no pinned canonical surface yet) we
        // realize the design as before so the slow path rebuilds a faithful
        // surface — the existing exact lane runs unchanged.
        let nfree_fast_path_revision = self.evaluator.nfree_fast_path_revision();
        let skip_design_realization = !allow_second_order && theta.len() == self.rho_dim + 1 && {
            let psi = theta[self.rho_dim];
            self.evaluator.psi_gram_tensor_covers(psi)
                    // #1033 gradient coverage: the skip serves the ψ-gradient n-free
                    // only where the analytic Chebyshev derivative is CERTIFIED.
                    // The kappa sufficient-statistic outer loop is routed here only
                    // when the certified gradient window spans the entire optimizer
                    // bounds, so a measured trial cannot pay an edge streamed
                    // ∂X/∂ψ pass after the initial priming eval.
                    && self.evaluator.psi_gram_tensor_covers_gradient(psi)
                    // #1264 (RESTORED) reduced-basis-rotation soundness precondition.
                    // The Gaussian inner penalized solve `(QsᵀGQs+S)β=b` runs in the
                    // CONDITIONED reduced basis. On the near-singular production
                    // Duchon Gram (κ(G)≈9.5e14) that basis ROTATES with ψ, and the
                    // skip installs the Chebyshev-interpolated `gram_at(ψ)` (≤1e-10
                    // vs streamed exact). When the trial-ψ basis differs from the
                    // reference surface's, the κ-amplified round-off moves β̂ by
                    // ~1.7e-5 — 17× the issue's 1e-6 bar — EVEN at a ψ the n-free
                    // VALUE window admits (cluster: β̂rel=1.749e-5 at ψ=2.803). The
                    // "stale-penalty-not-stale-basis" theory that dropped this gate
                    // was empirically refuted. So the skip is β̂-sound ONLY where the
                    // gauge-invariant range projector is unchanged vs the pinning ψ:
                    // `reduced_basis_equal(psi_ref, psi)`. Value coverage is NOT
                    // sufficient. This forces the exact O(n) `reset_surface` fallback
                    // across a basis rotation — correctness over n-independence
                    // (#1033 is frontier-blocked on rotating Duchon geometry).
                    && self.evaluator.psi_gram_tensor_covers_skip(psi)
                    // #1033 penalty lane: ψ moves S(ψ) too, and the skip leaves
                    // `reset_surface` un-run; only skip when the penalty can be
                    // rebuilt EXACTLY and n-free on the fast path, else the inner
                    // solve would pair XᵀWX(ψ_new) with the stale S(ψ_old).
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
        };
        // #1868: the #1033 n-free design-realization skip is armed above. A prior
        // debug override (`TEMP-SKIPOFF-1122`) hard-forced `skip_design_realization`
        // to `false` here to test whether the n-free ψ-Gram Chebyshev interpolant
        // was the source of the #1122 H-side FD-vs-analytic gap. That override was
        // never removed, so every in-window κ `eval_full` trial fell through to the
        // O(n) `ensure_theta` → `apply_log_kappa` + `reset_surface` lane — the O(n)
        // per-callback regression #1868 reports. The skip is already gated on
        // `!allow_second_order`, so it never fires on the H (Hessian) trials the
        // #1122 diagnostic was probing; the override only ever suppressed the
        // n-free gradient/value lane. Removing it routes the gradient eval through
        // the k-space `GaussianFixedCache` + ψ-derivative tensor as intended.
        if skip_design_realization {
            log::debug!(
                "[STAGE] {} eval_full at psi={:.6}: skipping n×k design re-realization \
                 + reconditioning — criterion/gradient/inner-solve served n-free from \
                 the certified ψ-gram tensor (GaussianFixedCache + k-space ψ-derivatives)",
                kind.label(),
                theta[self.rho_dim],
            );
        } else {
            self.cache
                .ensure_theta(theta)?;
        }
        let warm_beta = self.evaluator.current_beta();
        self.ensure_frozen_glm_tensor(theta, warm_beta.as_ref())?;
        // #1033 / #1111: stage the GLM frozen-W first-step Gram and conditioned
        // ψ-gradient whenever the certified frozen-weight tensor covers this
        // trial's ψ. The provider applies its drift guards, so misses clear the
        // staged slots and the exact streamed path runs.
        //
        // Stage through a shared helper because cost-only line-search probes use
        // the same first-Fisher-step Gram; they simply pass `allow_gradient=false`.
        self.stage_frozen_glm_trial_statistics(theta, warm_beta.as_ref(), !allow_second_order)?;
        // #1033: on the certified Gaussian skip path the value and ψ-gradient
        // are both served by k-space tensor statistics, so the row-wise X_ψ slab
        // is dead. Build only the exact n-free S_ψ components from frozen
        // geometry and attach a zero-storage design derivative placeholder.
        // Edge-gradient/Hessian/non-certified trials keep the exact row-wise
        // builder, because those lanes genuinely consume X_ψ.
        let hyper_dirs = if skip_design_realization {
            self.cache.nfree_tensor_gradient_hyper_dirs(theta)?
        } else {
            self.cache.hyper_dirs_for_current_design(self.data, kind)?
        };

        let design_revision = if skip_design_realization {
            nfree_fast_path_revision
        } else {
            Some(self.cache.design_revision())
        };
        // #1033 penalty lane: stage the EXACT n-free `S(ψ)` for this trial so the
        // evaluator's design-revision fast path can re-key the kept reference
        // surface without `reset_surface`. Built from the FROZEN basis geometry
        // (centers + identifiability transform + operator collocation points) at
        // the trial length-scale — no data rows — so it is valid even on the
        // design-realization skip path (where the design was not re-realized). The
        // caller (holding `cache`) computes it and hands the owned result to the
        // evaluator, sidestepping a `&mut cache` borrow alias. On the slow path
        // the evaluator ignores + clears the staged value (it rebuilds S from the
        // realized design). A build error here clears the stage; if the skip
        // already fired (fast path), the evaluator then hard-errors rather than
        // pairing a stale S — the safe outcome, since a rebuild from frozen
        // geometry should never fail in practice.
        if self.evaluator.supports_nfree_penalty_rekey() {
            match self
                .cache
                .canonical_penalties_at(theta, self.evaluator.frozen_penalty_ranks())
            {
                Ok(penalty) => self.evaluator.stage_fast_path_penalty(Some(penalty)),
                Err(e) => {
                    log::warn!(
                        "[STAGE] {} eval_full at psi={:.6}: exact n-free S(ψ) rebuild failed \
                         ({e}); clearing stage (eval falls to slow path)",
                        kind.label(),
                        theta[self.rho_dim],
                    );
                    self.evaluator.stage_fast_path_penalty(None);
                }
            }
        }
        // Warm-start PIRLS from the previous outer step's converged β. This is
        // especially impactful for GLM families (Poisson, NB, Binomial) that
        // cannot use the Gaussian Gram tensor n-free shortcut: without the warm
        // β every outer step cold-solves a full PIRLS from β=0, paying the full
        // O(n·p²) cost × PIRLS-iters × outer-iters budget. With the warm β the
        // inner solve typically converges in 1-2 Newton steps instead of 4-8.
        let eval = evaluate_joint_reml_outer_eval_at_theta(
            &mut self.evaluator,
            self.cache.design(),
            theta,
            self.rho_dim,
            hyper_dirs,
            warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
            if allow_second_order {
                order
            } else {
                OuterEvalOrder::ValueAndGradient
            },
            design_revision,
        );
        if let Ok(ref value) = eval {
            self.cache.store_eval_at(theta, value.clone());
        }
        eval
    }

    fn eval_efs(&mut self, theta: &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError> {
        self.cache
            .ensure_theta(theta)?;
        let kind = self.kind;
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            self.data,
            self.cache.spec(),
            self.cache.design(),
            &self.cache.spatial_terms,
        )?
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "failed to build {} hyper_dirs for exact-joint EFS",
                kind.adjective(),
            ))
        })?;
        let design_revision = Some(self.cache.design_revision());
        let warm_beta = self.evaluator.current_beta();
        evaluate_joint_reml_efs_at_theta(
            &mut self.evaluator,
            self.cache.design(),
            theta,
            self.rho_dim,
            hyper_dirs,
            warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
            design_revision,
        )
    }

    /// Cost-only evaluation. BFGS line-search probes route through the
    /// evaluator's true value-only path so they neither construct
    /// `try_build_spatial_log_kappa_hyper_dirs` nor assemble a gradient that
    /// the line search will discard. Split-borrow on `self.cache` +
    /// `self.evaluator` matches the pattern already used by `eval_full`.
    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        if let Some(cost) = self.cache.memoized_cost(theta) {
            return Ok(cost);
        }
        // #1029: a BFGS line-search VALUE probe. It converges the inner PIRLS to
        // the SAME tolerance the accepted-point full eval uses (NOT a capped
        // surrogate — a cap returns ∞ for a feasible point and re-imports the
        // #787/#808 outer stall), so probe and incumbent values live in ONE
        // refinement regime (measure-consistent Armijo). It is cheaper only
        // because it skips the gradient / hyper-dir assembly. Time the inner
        // cost-only solve and report it alongside the trial-θ distance from the
        // last evaluated point so this convergence-critical regression class is
        // visible in the STAGE trace (the spatial REML lane has no PROGRESS-
        // EXTENDED refine multiplier — that knob is SAE-only — so there is no
        // extended polish to strip from a probe here).
        //
        // Capture the previous evaluated θ BEFORE `ensure_theta` overwrites it,
        // so the logged distance reflects the backtracking step rather than 0.
        let probe_start = std::time::Instant::now();
        let psi_distance = self
            .cache
            .current_theta
            .as_ref()
            .filter(|reference| reference.len() == theta.len())
            .map(|reference| {
                reference
                    .iter()
                    .zip(theta.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .unwrap_or(f64::NAN);
        // #1033: a VALUE-only line-search probe needs only the certified ψ-Gram
        // tensor's value lane (`XᵀWX(ψ)/XᵀW(y−offset)(ψ)`), which the inner
        // Gaussian PLS reads n-free from the ψ-keyed `GaussianFixedCache`. So when
        // the single design-moving ψ is covered for the VALUE lane and the
        // evaluator has a pinned canonical slow-path revision, skip the n×k
        // design re-realization: `evaluate_cost_only` receives that pinned
        // revision, takes its `prepare_eval_state_cost_only` fast path (which
        // skips `reset_surface` + the n×k `apply_to_design` and re-keys the cache
        // to this probe's ψ), and the probe cost comes from k-space statistics
        // only. Line-search probes are the bulk of the κ-loop per-trial work, so
        // this is the dominant n-flat lever. Any miss (non-Gaussian, off-window,
        // missing penalty re-key support, or no pinned surface yet) realizes the
        // design and runs the exact streamed probe unchanged.
        let nfree_fast_path_revision = self.evaluator.nfree_fast_path_revision();
        let skip_value_realization = theta.len() == self.rho_dim + 1 && {
            let psi = theta[self.rho_dim];
            self.evaluator.psi_gram_tensor_covers(psi)
                    // #1868: a VALUE-only line-search probe does NOT need the
                    // #1264 `reduced_basis_equal` (`covers_skip`) soundness gate the
                    // ACCEPTED gradient eval (`eval_full`, still gated) requires. That
                    // gate exists because the design-realization skip freezes the
                    // conditioned reduced basis at the pinning ψ, and on the near-
                    // singular Duchon Gram (κ(G)≈9.5e14) a ψ-rotation makes the
                    // frozen basis interpolate β̂ with a κ-amplified round-off of
                    // β̂rel≈1.7e-5 — which matters for the RETURNED coefficients/
                    // gradient. A cost probe returns only the scalar REML criterion
                    // for the line search, and that criterion is STATIONARY in β̂ at
                    // the inner minimizer (envelope theorem): a β̂ perturbation δβ
                    // moves the data-fit+penalty term by O(δβ²) and leaves the
                    // `log|H|` term (built from the EXACT tensor Gram G(ψ), not β̂)
                    // untouched, so the RELATIVE cost error is ~δβ² ≈ 3e-10 — orders
                    // below the line search's 1e-5 Armijo tolerance. So the probe
                    // cannot be mis-ranked, and the converged κ/β̂ (pinned by the
                    // covers_skip-gated `eval_full` at accepted iterates) is
                    // unchanged. Gating the probe on `covers_skip` instead forced the
                    // O(n) `reset_surface` lane for every line-search step that
                    // overshoots the (n-drifting) reduced-basis-stable band — the
                    // #1868 per-callback reset climb: the band's rotation rate dP/dψ
                    // grows with n (sample-std standardization), so more probes fall
                    // just past PSI_GRAM_SKIP_PROJ_ATOL as n grows, defeating the
                    // n-independence the tensor lane was built for. The evaluator's
                    // own value-probe fast path (`prepare_eval_state_cost_only`) is
                    // gated on VALUE coverage exactly for this reason; aligning the
                    // driver here lets the probe cost come from the n-free k-space
                    // Gram/penalty statistics across the rotation.
                    //
                    // #1033 penalty lane: the value-probe fast path also skips
                    // `reset_surface`, so the probe must be able to re-key S(ψ)
                    // EXACTLY and n-free; otherwise its cost would use the stale
                    // S(ψ_old) and mis-rank the line search.
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
        };
        if theta.len() == self.rho_dim + 1
            && self.evaluator.has_psi_gram_tensor()
            && !self.evaluator.psi_gram_tensor_covers(theta[self.rho_dim])
        {
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "psi={:.6e} lies outside the certified psi-Gram tensor window",
                    theta[self.rho_dim]
                ),
            });
        }
        // #2481: preserve the derivative-lane contract. A basis or inner-solve
        // refusal at this trial is a recoverable domain wall; layout, topology,
        // and arbitrary invalid-input failures are fatal evaluation failures.
        if !skip_value_realization && let Err(error) = self.cache.ensure_theta(theta) {
            self.value_realization_failures += 1;
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, self.rho_dim);
            if !is_recoverable_trial_point_error(&error) {
                log::warn!(
                    "[STAGE] {} value-probe: design realization FAILED fatally at theta_norm={:.4e} log_kappa_norm={:.4e} ({error}); propagating",
                    self.kind.label(), theta_norm, log_kappa_norm,
                );
            }
            return Err(classify_spatial_value_probe_failure(error));
        }
        // #1033 penalty lane: stage the EXACT n-free `S(ψ)` for this probe's ψ so
        // the cost-only fast path re-keys the kept surface without `reset_surface`
        // (built from frozen geometry — valid even when the design was not
        // re-realized). The slow path clears it. A rebuild failure clears the
        // stage; the evaluator then takes the slow path or hard-errors (safe).
        if self.evaluator.supports_nfree_penalty_rekey() {
            match self
                .cache
                .canonical_penalties_at(theta, self.evaluator.frozen_penalty_ranks())
            {
                Ok(penalty) => self.evaluator.stage_fast_path_penalty(Some(penalty)),
                Err(_) => self.evaluator.stage_fast_path_penalty(None),
            }
        }
        let warm_beta = self.evaluator.current_beta();
        if let Err(err) = self.ensure_frozen_glm_tensor(theta, warm_beta.as_ref()) {
            log::warn!(
                "[STAGE] {} value-probe at psi={:.6}: frozen-W GLM tensor setup failed ({err}); \
                 falling back to exact streamed Gram",
                self.kind.label(),
                if theta.len() > self.rho_dim {
                    theta[self.rho_dim]
                } else {
                    f64::NAN
                },
            );
            self.evaluator.stage_glm_first_step_gram(None);
            self.evaluator.stage_glm_psi_gram_deriv(None);
        } else if let Err(err) =
            self.stage_frozen_glm_trial_statistics(theta, warm_beta.as_ref(), false)
        {
            log::warn!(
                "[STAGE] {} value-probe at psi={:.6}: frozen-W GLM staging failed ({err}); \
                 falling back to exact streamed Gram",
                self.kind.label(),
                if theta.len() > self.rho_dim {
                    theta[self.rho_dim]
                } else {
                    f64::NAN
                },
            );
            self.evaluator.stage_glm_first_step_gram(None);
            self.evaluator.stage_glm_psi_gram_deriv(None);
        }
        let design_revision = if skip_value_realization {
            nfree_fast_path_revision
        } else {
            Some(self.cache.design_revision())
        };
        let cost_label = self.kind.label();
        let result = {
            let design = self.cache.design();
            self.evaluator.evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                self.rho_dim,
                warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
                cost_label,
                design_revision,
            )
        };
        match result {
            Ok(cost) => {
                log::debug!(
                    "[STAGE] {cost_label} value-probe (order=Value): elapsed={:.3}s \
                     cost={cost:.6e} trial_theta_distance={psi_distance:.3e}",
                    probe_start.elapsed().as_secs_f64(),
                );
                self.cache.store_cost_at(theta, cost);
                Ok(cost)
            }
            // #2481: cost-evaluator failures use the same classifier as
            // design realization and the derivative-bearing lane.
            Err(error) => {
                self.value_evaluation_failures += 1;
                let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, self.rho_dim);
                if !is_recoverable_trial_point_error(&error) {
                    log::warn!(
                        "[STAGE] {cost_label} value-probe: cost evaluation FAILED fatally at theta_norm={theta_norm:.4e} log_kappa_norm={log_kappa_norm:.4e} ({error}); propagating",
                    );
                }
                Err(classify_spatial_value_probe_failure(error))
            }
        }
    }

    fn reset(&mut self) {
        self.cache.current_theta = None;
        self.cache.last_eval_theta = None;
        self.cache.last_cost = None;
        self.cache.last_eval = None;
    }
}

/// Exact joint `[ρ, ψ]` optimization for spatial terms using analytic
/// derivatives through the unified REML evaluator. This is the single shared
/// engine for both the anisotropic and isotropic coordinate kinds (selected by
/// `kind`).
///
/// At each outer iteration, the frozen term topology is reused and only the
/// spatial realized blocks affected by the current ψ are refreshed before the
/// unified evaluator returns cost + gradient + Hessian for the full
/// θ = [ρ, ψ] vector. The ψ derivatives flow through:
///
///   `AnisoBasisPsiDerivatives` / `SpatialPsiDerivative` → `DirectionalHyperParam`
///     → `build_tau_unified_objects` → `HyperCoord` ext_coords → unified evaluator
///
/// This gives Newton/BFGS quadratic convergence on the length-scale /
/// anisotropy parameters while jointly optimizing the smoothing parameters.
///
/// The ψ coordinates are parameterized as unconstrained log-scales. For the
/// anisotropic kind the decomposition into isotropic scale (ψ̄ = mean(ψ_a)) and
/// anisotropy (η_a = ψ_a − ψ̄, with Ση_a = 0) happens only on writeback via
/// `SpatialLogKappaCoords::apply_tospec`; the all-ones direction in ψ-space is
/// NOT a gauge direction — it controls the identifiable isotropic scale
/// κ = exp(ψ̄). The isotropic kind carries one log-κ coordinate per term. In
/// neither case is a sum-to-zero constraint enforced during optimization.
/// The ψ tail of `theta`, SIGNED and per-coordinate.
///
/// [`kphase_log_norms`] reports `‖ψ‖`, which is the right summary for a
/// multi-axis anisotropy block and the wrong one for a single signed coordinate:
/// measure-jet's ψ is `ln ℓ`, so `‖ψ‖ = 0.718` is consistent with a trial at
/// `ℓ = 2.05` and with one at `ℓ = 0.49`, and only the second is outside the
/// term's own geometry window. Reading the trajectory of a design-moving
/// coordinate out of the log requires the sign (gam#2750), so the per-trial
/// record carries the coordinates themselves alongside the norm.
fn kphase_psi_display(theta: &Array1<f64>, rho_dim: usize) -> String {
    let mut out = String::from("[");
    for (offset, value) in theta.iter().skip(rho_dim).enumerate() {
        if offset > 0 {
            out.push(',');
        }
        out.push_str(&format!("{value:+.4e}"));
    }
    out.push(']');
    out
}

fn kphase_log_norms(theta: &Array1<f64>, rho_dim: usize) -> (f64, f64) {
    let theta_norm = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let log_kappa_norm = theta
        .iter()
        .skip(rho_dim)
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    (theta_norm, log_kappa_norm)
}

fn run_exact_joint_spatial_optimization(
    kind: SpatialHyperKind,
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    baseline_design: &TermCollectionDesign,
    family: LikelihoodSpec,
    options: &FitOptions,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<(Array1<f64>, f64, f64, SpatialLengthScaleOptimizationTiming), EstimationError> {
    let label = kind.label();
    let inputs =
        exact_joint_spatial_inputs(label, y, weights, offset, baseline_design, &family, options)?;
    // #2671: condition the response through the SAME gate and the SAME
    // arithmetic the scalar-ρ route uses before it builds its `RemlState`
    // (#1000 centering / #1127 scaling). This route used to hand `y` to
    // `ExternalJointHyperEvaluator::new` VERBATIM, so while PIRLS charged a fixed
    // stabilization ridge `delta*||beta||^2` against a target of zero (removed in
    // #2901 V22) the two routes minimized penalized problems differing by
    // `delta*(2*c*beta0 + c^2)` on the intercept axis — and
    // `try_exact_joint_spatial_length_scale_optimization` then grades
    // `joint_seed_value` against the scalar route's `fit_score`.
    //
    // MEASURED at `517b6303f` on `mk_1d(15, t^2, 0.05, 7)` / `y ~ matern(x,
    // nu=5/2)`, one run, three arms, against the registered law
    // `gap = (n/2)/D_p * delta * ((beta0 + m)^2 - beta0^2)`:
    //
    //   mean(y) = -3.70e-17 (pre-centered)  gap ~ 0        fit ACCEPTED
    //   mean(y) =  2.130e-1 (as-is)         gap 3.674e-8   REFUSED
    //   mean(y) =  1.0213e1 (y + 10)        gap 5.047e-5   REFUSED, 1374x worse
    //
    // against `agreement_tolerance = 2.787e-8`. Whether the fit shipped depended
    // on where the origin of the user's response units happened to sit. The
    // scalar route moved 4.085e-14 under the same +10 shift (separation 1.24e9).
    //
    // This is the SEARCH response only. `theta_star` selects `(λ̂, ψ̂)` and the
    // caller's accept-fit re-fits the ORIGINAL `y` at that point, exactly as the
    // scalar route's accept-fit does, so no reported coefficient, fitted value or
    // dispersion moves. Off the identity-link Gaussian path the helper returns
    // `None` and `y` is borrowed verbatim — no allocation, no behavioural change.
    //
    // `y` is shadowed rather than threaded so that ALL THREE consumers below take
    // the conditioned response together: the evaluator, the frozen-GLM inputs,
    // and the `z = y − offset` vector the certified ψ-Gram tensor is built from.
    // A partial application would pair an n-free fast path with a differently
    // conditioned slow path.
    let y = inputs.response(y);
    let offset = inputs.offset.view();
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};
    use gam_solve::rho_optimizer::OuterEvalOrder;

    let theta_dim = theta0.len();
    let coord_dim = theta_dim - rho_dim;
    let PreparedExactJointSpatialRoute {
        mut ctx,
        analytic_outer_hessian_available,
        suppress_outer_hessian_for_nfree,
        psi_rank_stable_floor,
        psi_rank_stable_ceiling,
    } = prepare_exact_joint_spatial_route(
        kind,
        data,
        y,
        weights,
        offset,
        &inputs.external_opts,
        resolvedspec,
        baseline_design,
        &family,
        options,
        spatial_terms,
        dims_per_term,
        theta0,
        lower,
        upper,
        rho_dim,
    )?;

    // Priming is part of search, so it must stop at the order-three gradient
    // lane. The only `ValueGradientHessian` request belongs to the mint audit.
    let kphase_prime_order = OuterEvalOrder::ValueAndGradient;
    let kphase_prime_start = std::time::Instant::now();
    // The priming eval is the joint criterion AT THE SEED, and it is the only
    // number that can tell a solver regression apart from a cross-route
    // criterion disagreement at the acceptance gate downstream: that gate grades
    // `final_value` (this evaluator, at θ*) against `fit_score(&best.fit)` (the
    // scalar-ρ route, at θ0). Discarding it forced the two questions into one
    // refusal, so a route difference of a few ulps-relative was reported as
    // "the optimizer made the score worse". Keep it and let the caller state the
    // two contracts separately.
    let seed_value = ctx
        .eval_full(theta0, kphase_prime_order, analytic_outer_hessian_available)?
        .0;
    log::info!(
        "[KAPPA-PHASE-PRIME] n_rows={} order={:?} seed_value={seed_value:.12e} elapsed_s={:.4} slow_path_resets_total={} design_revision={}",
        data.nrows(),
        kphase_prime_order,
        kphase_prime_start.elapsed().as_secs_f64(),
        ctx.evaluator.slow_path_reset_count(),
        ctx.cache.design_revision(),
    );

    let kphase_cost_calls = std::cell::Cell::new(0usize);
    let kphase_eval_calls = std::cell::Cell::new(0usize);
    let kphase_efs_calls = std::cell::Cell::new(0usize);
    let kphase_cost_total_s = std::cell::Cell::new(0.0);
    let kphase_eval_total_s = std::cell::Cell::new(0.0);
    let kphase_efs_total_s = std::cell::Cell::new(0.0);
    let kphase_nfree_miss_shape = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_value = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_gradient = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_penalty = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_revision = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_second_order = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_other = std::cell::Cell::new(0u64);
    let kphase_optim_start = std::time::Instant::now();
    let kphase_log_kappa_dim = coord_dim;
    let kphase_slow_resets_start = ctx.evaluator.slow_path_reset_count();
    let kphase_design_revision_start = ctx.cache.design_revision();
    // #1868: snapshot the deterministic n-free skip-path row-touch accumulator
    // AFTER the one-time priming eval above, so the reported delta measures only
    // the per-trial inner-synthesis row work across the κ-optimisation phase.
    let kphase_nfree_skip_touches_start = gam_solve::pirls::nfree_skip_row_element_touches();

    // #1033: lift the ψ (log-κ) lower bound to the n-free rank-stable floor so the
    // optimizer never line-searches into the rank-deficient sliver where the
    // design-realization skip is soundly refused (→ O(n) reset_surface). The lift
    // touches ONLY the single design-moving ψ coordinate at `rho_dim`; all ρ
    // bounds are untouched. `psi_rank_stable_floor` is already constrained to lie
    // strictly inside `(psi_lo, theta0[rho_dim])`, so theta0 stays feasible.
    let lower_effective: std::borrow::Cow<'_, Array1<f64>> = match psi_rank_stable_floor {
        Some(floor) if coord_dim == 1 && floor > lower[rho_dim] => {
            let mut lifted = lower.clone();
            lifted[rho_dim] = floor;
            std::borrow::Cow::Owned(lifted)
        }
        _ => std::borrow::Cow::Borrowed(lower),
    };
    let lower = lower_effective.as_ref();

    // #1033: clamp the ψ (log-κ) upper bound DOWN to the n-free rank-stable ceiling
    // so the optimizer never line-searches into the high-edge rank-deficient sliver
    // where the design-realization skip is soundly refused (→ O(n) reset_surface,
    // plus a second reset from the deficient pinning ψ). Touches ONLY the single
    // design-moving ψ coordinate at `rho_dim`; all ρ bounds are untouched.
    // `psi_rank_stable_ceiling` is already constrained to lie strictly inside
    // `(theta0[rho_dim], psi_hi)`, so theta0 stays feasible.
    let upper_effective: std::borrow::Cow<'_, Array1<f64>> = match psi_rank_stable_ceiling {
        Some(ceiling) if coord_dim == 1 && ceiling < upper[rho_dim] => {
            let mut clamped = upper.clone();
            clamped[rho_dim] = ceiling;
            std::borrow::Cow::Owned(clamped)
        }
        _ => std::borrow::Cow::Borrowed(upper),
    };
    let upper = upper_effective.as_ref();

    let problem = exact_joint_multistart_outer_problem(
        theta0,
        lower,
        upper,
        rho_dim,
        coord_dim,
        theta_dim,
        Derivative::Analytic,
        if analytic_outer_hessian_available && !suppress_outer_hessian_for_nfree {
            // `Either` even when the #1033 n-free ψ-lane is armed (gam#2760).
            //
            // The suppression used to force `Unavailable` here, on the stated
            // grounds that "the planner then selects BFGS instead of ARC". It
            // does not need to: `with_prefer_gradient_only(true)` below is
            // unconditional, and `capability::plan` reads
            // `(Analytic, Analytic) if prefer_gradient_only -> S::Bfgs` BEFORE
            // the ARC arm. Gradient-only ROUTING was already secured; the
            // declaration was not what secured it.
            //
            // What `Unavailable` actually did was erase the ONE terminal
            // curvature evaluation the mint is entitled to — the arrangement
            // `with_prefer_gradient_only`'s own doc describes three lines below
            // ("Hessian availability is a terminal-certification capability, not
            // a warrant to rebuild that tower at every accepted iterate";
            // "reserve it for that one terminal evaluation"). With curvature
            // gone, this lane silently forfeits FOUR certification mechanisms
            // that every other outer route has: the `curvature-resolvability`
            // rung (the only bound in the ladder derived from the criterion's
            // own resolution), the #2348 asymptote-rail certificate, the
            // curvature-scaled flat-valley widening, and the #2299 large-step
            // flatness certificate. Its refusals then rest entirely on a raw
            // gradient-magnitude band, which is how a converged fit at the
            // criterion's noise floor reads as `NOT STATIONARY` (gam#2760: the
            // n = 4000 rung refuses at `|Pg| = 7.677e-1` against `2.566e-1`
            // after a line search that spent 48 consecutive probes below the
            // fifth digit of θ without improving the objective — the signature
            // of a remaining decrement under the criterion's own resolution,
            // which is exactly the question `curvature-resolvability` answers
            // and this lane could not ask).
            //
            // This costs ONE O(n) evaluation per minted candidate. #1033's
            // invariant is per-TRIAL cost — "an in-window hyperparameter trial
            // touches only k×k objects" — and a terminal certification is not a
            // trial; the fit already pays O(n) for its final PIRLS assembly.
            // The per-trial skip is untouched: BFGS still issues only
            // `ValueAndGradient`, so every in-window trial stays n-free.
            //
            // This is the same shape as #2706's repair one flag over, where
            // `suppress_outer_hessian_for_nfree` was also answering both "how
            // should the SEARCH route?" and a second question it had no
            // business answering (there, `with_require_measured_psd`).
            //
            // NOT RESTORED HERE, AND THE REASON IS A MEASUREMENT. Restoring it
            // makes `exact_spatial_joint_engine_aniso_iso_parity_1d` refuse —
            // and the refusal is honest, which is exactly why it cannot ride in
            // on this issue:
            //
            //   aniso-psi joint REML: |Pg| = 5.143e-3 vs bound 8.100e-3 (STATIONARY)
            //   hessian_psd=NO curvature_source=terminal-analytic
            //   INDEFINITE CURVATURE AT INTERIOR OPTIMUM (curvature floor did not clear)
            //   [interior lambda_min = -1.585e-3, gradient_floor = 3.061e-3]
            //
            // That fit is at a stationary point (`|Pg|` a third of its bound)
            // whose interior curvature is measurably indefinite, with `ψ` railed
            // at its own box edge. Before this it shipped with
            // `curvature_source=unavailable` — nobody had checked. Turning "not
            // checked" into "refused, with the eigenvalue that refused it" is
            // the right direction, and it is a real finding about this lane's
            // terminal geometry. It is not, however, #2760's defect, and #2760's
            // repair does not need it: the ladder is green at all five rungs
            // WITHOUT the restoration, because what fixed the line search was
            // retiring the ψ-Gram surrogate at the polish, not the mint's
            // curvature. Restoring curvature was the INSTRUMENT that found the
            // surrogate — the value-agreement guard only fires when the mint
            // asks for the analytic lane — and an instrument is not a fix.
            //
            // So it stays off here, with the measurement written down, and the
            // indefinite-curvature question gets its own issue rather than
            // arriving as a side effect of this one.
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        },
        // Single-block spatial path: penalty-like rho + spatial psi.
        // EFS/HybridEFS remain eligible (the Wood-Fasiolo PSD structure holds
        // for single-block families with β-independent joint H_L) UNLESS the
        // n-free Gaussian ψ-lane is armed (#1033): HybridEFS forms the trace Gram
        // `tr(H⁻¹ B_d H⁻¹ B_e)` from the n-dependent curvature slab `B_d`, so it
        // realizes O(n) per step exactly like a Hessian eval. Disabling the
        // fixed-point lane there forces the planner to BFGS (`(Analytic,
        // Unavailable)` → `S::Bfgs`), keeping every in-window κ-trial on the
        // n-free `ValueAndGradient` skip even when `n_params` exceeds the small-
        // BFGS threshold (aniso / multi-ψ).
        suppress_outer_hessian_for_nfree,
        seed_risk_profile_for_likelihood_family(&family),
        kappa_options.rel_tol,
        kappa_options.max_outer_iter.max(1),
        // Rho-axis BFGS cap: log-λ's natural step is ≈ 5. Anything tighter
        // throttles BFGS on flat REML valleys.
        Some(5.0),
        // Psi-axis BFGS cap: one doubling of the kernel length scale per iteration.
        Some(SPATIAL_PSI_BFGS_STEP_CAP),
        None,
        // Calibrate the outer to the n-scaled profiled REML/LAML objective for
        // every family — the iso-κ non-convergence cure (#1053 1-D Matérn,
        // #1066 2-D binomial geo, #1069 GP/kriging). p = baseline design column
        // count.
        Some((data.nrows(), baseline_design.design.ncols())),
        // The scalar Matérn endpoint comparison has already selected and
        // certified the range basin. Give its explicit theta0 the only joint
        // start; anisotropic and non-Matérn paths keep their established seed
        // policy.
        kind == SpatialHyperKind::Isotropic
            && constant_curvature_term_indices(resolvedspec).is_empty()
            && spatial_terms.iter().any(|&term_idx| {
                matches!(
                    resolvedspec
                        .smooth_terms
                        .get(term_idx)
                        .map(|term| &term.basis),
                    Some(SmoothBasisSpec::Matern { .. })
                )
            }),
    )?;

    let eval_outer = |ctx: &mut &mut SpatialJointContext<'_>,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let t0 = std::time::Instant::now();
        let allow_second_order_for_call = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        let requires_gradient = !matches!(order, OuterEvalOrder::Value);
        let gate = ctx.nfree_skip_gate_status(theta, allow_second_order_for_call, requires_gradient);
        let resets_before = ctx.evaluator.slow_path_reset_count();
        let raw = ctx.eval_full(theta, order, analytic_outer_hessian_available);
        let reset_delta = ctx
            .evaluator
            .slow_path_reset_count()
            .saturating_sub(resets_before);
        if reset_delta > 0 {
            if !gate.shape {
                kphase_nfree_miss_shape.set(kphase_nfree_miss_shape.get() + reset_delta);
            }
            if gate.shape && !gate.value {
                kphase_nfree_miss_value.set(kphase_nfree_miss_value.get() + reset_delta);
            }
            if gate.shape && gate.value && !gate.gradient {
                kphase_nfree_miss_gradient.set(kphase_nfree_miss_gradient.get() + reset_delta);
            }
            if gate.shape && gate.value && gate.gradient && !gate.penalty {
                kphase_nfree_miss_penalty.set(kphase_nfree_miss_penalty.get() + reset_delta);
            }
            if gate.shape && gate.value && gate.gradient && gate.penalty && !gate.revision {
                kphase_nfree_miss_revision.set(kphase_nfree_miss_revision.get() + reset_delta);
            }
            if gate.shape
                && gate.value
                && gate.gradient
                && gate.penalty
                && gate.revision
                && gate.second_order
            {
                kphase_nfree_miss_second_order
                    .set(kphase_nfree_miss_second_order.get() + reset_delta);
            }
            if gate.would_skip(requires_gradient) {
                kphase_nfree_miss_other.set(kphase_nfree_miss_other.get() + reset_delta);
            }
        }
        let elapsed_s = t0.elapsed().as_secs_f64();
        kphase_eval_calls.set(kphase_eval_calls.get() + 1);
        kphase_eval_total_s.set(kphase_eval_total_s.get() + elapsed_s);
        let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
        log::info!(
            "[KAPPA-PHASE] phase=eval_outer call={} order={:?} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} psi={} elapsed_s={:.4}",
            kphase_eval_calls.get(),
            order,
            Some(ctx.cache.design_revision()),
            theta_norm,
            log_kappa_norm,
            kphase_psi_display(theta, rho_dim),
            elapsed_s,
        );
        match raw {
            Ok((cost, grad, hess)) => Ok(OuterEval {
                cost,
                gradient: grad,
                hessian: hess,
                inner_beta_hint: None,
            }),
            // A trial hyperparameter at which the spatial kernel design, its
            // ψ-derivatives or the criterion refuse is an infeasible point, not
            // a fatal error. The refusal travels typed, through the same
            // classifier as the value lane, so a single bad probe — e.g. an
            // anisotropy that overflows the Duchon radial kernel — makes the
            // search retreat and its reason reaches the outer log (#2735).
            Err(err) => {
                let err = classify_spatial_value_probe_failure(err);
                if err.is_trial_point_infeasible() {
                    // Each refusal costs the line search a halving and this call's
                    // work; a run that crawls on refusals must say why (#2735).
                    log::info!("[{label}] trial point refused at theta={theta:?}: {err}; retreating");
                }
                Err(err)
            }
        }
    };

    let obj = problem.build_objective_with_eval_order(
        &mut ctx,
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            let t0 = std::time::Instant::now();
            let gate = ctx.nfree_skip_gate_status(theta, false, false);
            let resets_before = ctx.evaluator.slow_path_reset_count();
            let cost = ctx.eval_cost(theta);
            let reset_delta = ctx
                .evaluator
                .slow_path_reset_count()
                .saturating_sub(resets_before);
            if reset_delta > 0 {
                if !gate.shape {
                    kphase_nfree_miss_shape.set(kphase_nfree_miss_shape.get() + reset_delta);
                }
                if gate.shape && !gate.value {
                    kphase_nfree_miss_value.set(kphase_nfree_miss_value.get() + reset_delta);
                }
                if gate.shape && gate.value && !gate.penalty {
                    kphase_nfree_miss_penalty.set(kphase_nfree_miss_penalty.get() + reset_delta);
                }
                if gate.shape && gate.value && gate.penalty && !gate.revision {
                    kphase_nfree_miss_revision.set(kphase_nfree_miss_revision.get() + reset_delta);
                }
                if gate.would_skip(false) {
                    kphase_nfree_miss_other.set(kphase_nfree_miss_other.get() + reset_delta);
                }
            }
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_cost_calls.set(kphase_cost_calls.get() + 1);
            kphase_cost_total_s.set(kphase_cost_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
            log::info!(
                "[KAPPA-PHASE] phase=cost call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_cost_calls.get(),
                Some(ctx.cache.design_revision()),
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            cost
        },
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            eval_outer(
                ctx,
                theta,
                // The legacy gradient bridge is first-order by definition.
                // Exact curvature is reachable only through the order-aware hook
                // below, which terminal certification invokes once.
                OuterEvalOrder::ValueAndGradient,
            )
        },
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(ctx, theta, order)
        },
        Some(|ctx: &mut &mut SpatialJointContext<'_>| {
            ctx.reset();
        }),
        Some(|ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            let t0 = std::time::Instant::now();
            let eval = ctx.eval_efs(theta);
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_efs_calls.set(kphase_efs_calls.get() + 1);
            kphase_efs_total_s.set(kphase_efs_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
            log::info!(
                "[KAPPA-PHASE] phase=efs call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_efs_calls.get(),
                Some(ctx.cache.design_revision()),
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            eval
        }),
    );

    // #2676: publish the criterion's EXACT invariance so the outer certificate
    // deflates it instead of judging a chain-rule term against its own absolute
    // value. The closure speaks rho; `ClosureObjective` embeds it into
    // `theta = [rho, psi]` with exact zeros in the psi block from the declared
    // layout, because the invariance lives entirely in the penalty map.
    //
    // Read at the CURRENT psi, not once per fit: on this route the penalty map
    // is rebuilt at every psi, so the redundancy's coefficient
    // (`S_2 = c(psi) S_0`) moves and a snapshot would deflate a direction the
    // criterion is no longer flat along.
    let mut obj = obj
        .with_criterion_invariance(
            |ctx: &mut &mut SpatialJointContext<'_>, rho: &Array1<f64>| {
                ctx.evaluator.criterion_invariant_directions(rho)
            },
        )
        // gam#2760: the #1033b ψ-Gram tensor is a certified n-free SURROGATE for
        // the criterion, not the criterion. It is certified on the GRAM
        // (`PSI_GRAM_CERT_RTOL = 1e-9`) and on the reduced-basis SUBSPACE
        // (`PSI_GRAM_SKIP_PROJ_ATOL = 1e-7`); nothing in it bounds the scalar the
        // optimizer ranks, and the weakly-penalized inner solve amplifies a Gram
        // residual by the radial-kernel conditioning. MEASURED at `n = 2000` on
        // the #2760 ladder, at the point the search stopped: the surrogate and
        // the exact lane price the criterion as `-1.2781058170149880e4` and
        // `-1.2781006804748626e4`, a `5.137e-2` gap against a `1.905e-4` roundoff
        // envelope — `270×` `outer_value_agreement_bound`, `4e-6` relative where
        // `√ε` is the contract.
        //
        // That makes the surrogate an optimization stage, never a certifiable
        // measure, so it gets an exact exit. `run_outer` calls this once, after
        // the search has
        // converged, and then re-runs the optimizer from that checkpoint on
        // whatever measure the objective now prices — here the exact streamed
        // criterion — before the mandatory analytic certificate. The n-free
        // property is kept where it pays (every in-window TRIAL of the search)
        // and dropped where it cannot be certified (the terminal polish).
        .with_exact_polish(|ctx: &mut &mut SpatialJointContext<'_>| {
            if !ctx.evaluator.retire_psi_gram_tensor() {
                return false;
            }
            // Objective memoization is theta-only, so a surrogate value at the
            // warm checkpoint must not alias the exact value at the same theta.
            ctx.cache.forget_eval_memo();
            // The SEARCH's n-independence counters stop here; the polish's own
            // O(n) work is measured from this boundary and reported beside them.
            ctx.nfree_polish_boundary = Some((
                ctx.evaluator.slow_path_reset_count(),
                gam_solve::pirls::nfree_skip_row_element_touches(),
            ));
            log::info!(
                "[KAPPA-PHASE-POLISH] the certified n-free psi-Gram surrogate is retired at \
                 the search checkpoint; the optimizer continues and certifies on the exact \
                 streamed criterion (gam#2760)"
            );
            true
        });

    let run_label = match kind {
        SpatialHyperKind::Anisotropic => "aniso-psi joint REML",
        SpatialHyperKind::Isotropic => "iso-kappa joint REML",
    };
    let result = problem.run(&mut obj, run_label)?;
    if !result.converged() {
        crate::bail_invalid_estim!(
            "{} did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
            run_label,
            result.iterations,
            result.final_value,
            result.final_grad_norm_report(),
        );
    }
    drop(obj);
    let kphase_total_s = kphase_optim_start.elapsed().as_secs_f64();
    let slow_resets_end = ctx.evaluator.slow_path_reset_count();
    let skip_touches_end = gam_solve::pirls::nfree_skip_row_element_touches();
    // gam#2760: the SEARCH's counters stop at the exact-polish boundary. Every
    // polish evaluation takes the O(n) lane by construction — that IS the
    // repair — so charging them to gates whose subject is "an in-window
    // hyperparameter TRIAL touches only k×k objects" would report a correctness
    // fix as a broken skip. Both halves are published; neither is hidden.
    let (search_slow_resets_end, search_skip_touches_end) =
        ctx.nfree_polish_boundary.unwrap_or((slow_resets_end, skip_touches_end));
    let kphase_slow_resets = search_slow_resets_end.saturating_sub(kphase_slow_resets_start);
    let kphase_polish_slow_resets = slow_resets_end.saturating_sub(search_slow_resets_end);
    let kphase_design_revision_delta = ctx
        .cache
        .design_revision()
        .saturating_sub(kphase_design_revision_start);
    let kphase_nfree_skip_touches =
        search_skip_touches_end.saturating_sub(kphase_nfree_skip_touches_start);
    let kphase_polish_skip_touches = skip_touches_end.saturating_sub(search_skip_touches_end);
    log::info!(
        "[KAPPA-PHASE-POLISH-SUMMARY] n_rows={} exact_polish_ran={} polish_slow_path_resets={} polish_nfree_skip_row_touches={}",
        data.nrows(),
        ctx.nfree_polish_boundary.is_some(),
        kphase_polish_slow_resets,
        kphase_polish_skip_touches,
    );
    log::info!(
        "[KAPPA-PHASE-SUMMARY] n_rows={} log_kappa_dim={} n_cost={} cost_total_s={:.4} n_eval={} eval_total_s={:.4} n_efs={} efs_total_s={:.4} value_realization_failures={} value_evaluation_failures={} slow_path_resets={} design_revision_delta={} nfree_skip_row_touches={} nfree_miss_shape={} nfree_miss_value={} nfree_miss_gradient={} nfree_miss_penalty={} nfree_miss_revision={} nfree_miss_second_order={} nfree_miss_other={} optim_total_s={:.4}",
        data.nrows(),
        kphase_log_kappa_dim,
        kphase_cost_calls.get(),
        kphase_cost_total_s.get(),
        kphase_eval_calls.get(),
        kphase_eval_total_s.get(),
        kphase_efs_calls.get(),
        kphase_efs_total_s.get(),
        ctx.value_realization_failures,
        ctx.value_evaluation_failures,
        kphase_slow_resets,
        kphase_design_revision_delta,
        kphase_nfree_skip_touches,
        kphase_nfree_miss_shape.get(),
        kphase_nfree_miss_value.get(),
        kphase_nfree_miss_gradient.get(),
        kphase_nfree_miss_penalty.get(),
        kphase_nfree_miss_revision.get(),
        kphase_nfree_miss_second_order.get(),
        kphase_nfree_miss_other.get(),
        kphase_total_s,
    );
    let timing = SpatialLengthScaleOptimizationTiming {
        log_kappa_dim: kphase_log_kappa_dim,
        cost_calls: kphase_cost_calls.get(),
        cost_total_s: kphase_cost_total_s.get(),
        eval_calls: kphase_eval_calls.get(),
        eval_total_s: kphase_eval_total_s.get(),
        efs_calls: kphase_efs_calls.get(),
        efs_total_s: kphase_efs_total_s.get(),
        slow_path_resets: kphase_slow_resets,
        design_revision_delta: kphase_design_revision_delta,
        nfree_skip_row_touches: kphase_nfree_skip_touches,
        nfree_miss_shape: kphase_nfree_miss_shape.get(),
        nfree_miss_value: kphase_nfree_miss_value.get(),
        nfree_miss_gradient: kphase_nfree_miss_gradient.get(),
        nfree_miss_penalty: kphase_nfree_miss_penalty.get(),
        nfree_miss_revision: kphase_nfree_miss_revision.get(),
        nfree_miss_second_order: kphase_nfree_miss_second_order.get(),
        nfree_miss_other: kphase_nfree_miss_other.get(),
        exact_polish_ran: ctx.nfree_polish_boundary.is_some(),
        polish_slow_path_resets: kphase_polish_slow_resets,
        polish_nfree_skip_row_touches: kphase_polish_skip_touches,
        optim_total_s: kphase_total_s,
    };
    log::trace!(
        "[{}] converged in {} iterations, final_value={:.6e}, grad_norm={}",
        label,
        result.iterations,
        result.final_value,
        result.final_grad_norm_report(),
    );
    // No sum-to-zero enforcement needed: ψ coordinates are unconstrained during
    // optimization. For the anisotropic kind the decomposition into (ψ̄, η)
    // happens later in apply_tospec.
    let theta_star = result.rho;
    Ok((theta_star, result.final_value, seed_value, timing))
}

/// The joint [ρ, ψ] route's owned inputs: the composed offset, the search
/// response conditioned through the scalar-ρ route's gate (#2671), and the
/// evaluator options.
struct ExactJointSpatialInputs {
    offset: Array1<f64>,
    conditioned_y: Option<Array1<f64>>,
    external_opts: ExternalOptimOptions,
}

impl ExactJointSpatialInputs {
    /// The response the search evaluates: the conditioned one when the gate
    /// conditioned it, otherwise the caller's.
    fn response<'a>(&'a self, y: ArrayView1<'a, f64>) -> ArrayView1<'a, f64> {
        self.conditioned_y
            .as_ref()
            .map_or(y, |conditioned| conditioned.view())
    }
}

fn exact_joint_spatial_inputs(
    label: &str,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    baseline_design: &TermCollectionDesign,
    family: &LikelihoodSpec,
    options: &FitOptions,
) -> Result<ExactJointSpatialInputs, EstimationError> {
    let offset = baseline_design
        .compose_offset(offset, "spatial joint fit")
        .map_err(EstimationError::BasisError)?;
    let external_opts = external_opts_for_design(family, baseline_design, options);
    let conditioned_y = gam_solve::estimate::gaussian_identity_outer_response_conditioning(
        &baseline_design.design,
        &baseline_design.penalties,
        &external_opts,
        y,
        weights,
        offset.view(),
    )?;
    if conditioned_y.is_some() {
        log::info!(
            "[{label}] outer response conditioned for the joint [rho, psi] search (#2671): the \
             criterion is now formed in the same coordinates as the scalar-rho route it is \
             graded against"
        );
    }
    Ok(ExactJointSpatialInputs {
        offset,
        conditioned_y,
        external_opts,
    })
}

/// The joint [ρ, ψ] route as the search receives it: the evaluation context with
/// any certified ψ-Gram lane attached, and the routing facts the search reads.
/// `y` is the search response from [`ExactJointSpatialInputs::response`] and
/// `offset` its composed offset.
struct PreparedExactJointSpatialRoute<'d> {
    ctx: SpatialJointContext<'d>,
    analytic_outer_hessian_available: bool,
    suppress_outer_hessian_for_nfree: bool,
    psi_rank_stable_floor: Option<f64>,
    psi_rank_stable_ceiling: Option<f64>,
}

fn prepare_exact_joint_spatial_route<'d>(
    kind: SpatialHyperKind,
    data: ArrayView2<'d, f64>,
    y: ArrayView1<'d, f64>,
    weights: ArrayView1<'d, f64>,
    offset: ArrayView1<'_, f64>,
    external_opts: &ExternalOptimOptions,
    resolvedspec: &TermCollectionSpec,
    baseline_design: &TermCollectionDesign,
    family: &LikelihoodSpec,
    options: &FitOptions,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
) -> Result<PreparedExactJointSpatialRoute<'d>, EstimationError> {
    let label = kind.label();
    // Use bounds and design metadata for validation.
    assert!(
        lower.len() == theta0.len() && upper.len() == theta0.len(),
        "spatial hyperparameter bounds must match theta length: lower_len={}, upper_len={}, theta_len={}",
        lower.len(),
        upper.len(),
        theta0.len()
    );
    assert!(
        baseline_design.smooth.terms.len() >= spatial_terms.len(),
        "baseline design must have at least one smooth term per spatial term: baseline_terms={}, spatial_terms={}",
        baseline_design.smooth.terms.len(),
        spatial_terms.len()
    );

    let theta_dim = theta0.len();
    // Directional-coordinate dimension: psi-per-axis (anisotropic) or
    // kappa-per-term (isotropic). The numerics below are identical either way.
    let coord_dim = theta_dim - rho_dim;
    // Capability records the exact Hessian even though #2359 reserves it for
    // the terminal certificate. Search uses the analytic gradient and therefore
    // stops at the third-order family channel; minting alone consumes the
    // fourth-order spatial contractions.
    let analytic_outer_hessian_available =
        exact_joint_spatial_outer_hessian_available(&family, baseline_design);
    if !analytic_outer_hessian_available {
        log::info!(
            "[{label}] analytic outer Hessian unavailable for family/design; routing without second-order geometry (coord_dim={coord_dim})"
        );
    }
    // #1033: set when the n-free Gaussian ψ-lane arms below. It keeps the SEARCH
    // gradient-only — the outer Hessian curvature slab `B_j` is irreducibly
    // n-dependent, so a `ValueGradientHessian` eval forces the O(n) design
    // re-realization and an in-window κ-trial must never issue one. It also
    // disables the EFS/HybridEFS fixed-point lane, whose trace Gram
    // `tr(H⁻¹ B_d H⁻¹ B_e)` realizes the same slab.
    //
    // It no longer suppresses the DECLARED Hessian (gam#2760). Declaring
    // `Unavailable` never was what routed the search to BFGS —
    // `with_prefer_gradient_only(true)` is — and erasing the declaration cost
    // the mint the one terminal curvature evaluation #2359 reserves for it,
    // together with every certificate rung that reads curvature. See the
    // `DeclaredHessianForm` argument at the `exact_joint_multistart_outer_problem`
    // call below.
    let mut suppress_outer_hessian_for_nfree = false;

    log::trace!(
        "[{}] starting analytic optimization: rho_dim={}, coord_dim={}, dims_per_term={:?}",
        label,
        rho_dim,
        coord_dim,
        dims_per_term,
    );

    let mut ctx = SpatialJointContext {
        data,
        rho_dim,
        kind,
        value_realization_failures: 0,
        value_evaluation_failures: 0,
        nfree_polish_boundary: None,
        cache: SingleBlockExactJointDesignCache::new_with_policy(
            data,
            resolvedspec.clone(),
            baseline_design.clone(),
            spatial_terms.to_vec(),
            rho_dim,
            dims_per_term.to_vec(),
            &options.resource_policy,
        )
        .map_err(EstimationError::InvalidInput)?,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y,
            weights,
            &baseline_design.design,
            offset,
            &baseline_design.penalties,
            &external_opts,
            label,
        )?,
        frozen_glm_inputs: if coord_dim == 1 && frozen_glm_tensor_eligible_family(&family) {
            Some(SpatialFrozenGlmInputs {
                y: y.to_owned(),
                weights: weights.to_owned(),
                offset: offset.to_owned(),
                family: family.clone(),
            })
        } else {
            None
        },
        frozen_glm_psi_bounds: if coord_dim == 1 && frozen_glm_tensor_eligible_family(&family) {
            Some((lower[rho_dim], upper[rho_dim]))
        } else {
            None
        },
        frozen_glm_tensor: None,
        frozen_glm_tensor_attempted: false,
        frozen_glm_weight_memo: None,
    };

    // #1033b: single isotropic design-moving coordinate on a Gaussian-identity
    // fit — build the certified Chebyshev-in-ψ Gram tensor ONCE over the
    // optimizer's ψ window and hand it to the evaluator. Every in-window trial
    // then receives its Gaussian sufficient statistics (XᵀWX(ψ), XᵀW(y−offset),
    // (y−offset)ᵀW(y−offset)) assembled n-free instead of paying the per-trial
    // O(n·p²) Gram re-stream after the design rebuild. The realizer closure
    // returns the RAW realized design; the evaluator threads it through its
    // own (fixed, ψ-invariant) parametric column conditioning so the tensor
    // lives in the same frame as the streamed Gram. Certification failure,
    // off-window trials, or any other ineligibility silently keep the exact
    // streamed path (same numbers, the tensor is certified to
    // PSI_GRAM_SPOT_RTOL against the exact rebuild).
    // #1033 (rank-stable κ-floor): set to the lowest ψ at which the certified
    // tensor's conditioned Gram holds maximal numerical rank. Below it the
    // reduced basis collapses/rotates and the design-realization skip is SOUNDLY
    // refused (→ O(n) reset_surface); the κ window floor lands inside that
    // degenerate sliver and DRIFTS with n through the sample-std
    // standardization, so n=2000's line search re-enters the slow lane while
    // n=1000's does not. Lifting the optimizer's lower bound to this n-FREE
    // (k-space) floor keeps every in-window trial on the fast path for all n,
    // and only excludes long length scales at which the conditioned Gram has
    // already lost a direction.
    let mut psi_rank_stable_floor: Option<f64> = None;
    // #1033 (rank-stable κ-ceiling): symmetric twin of the floor. The conditioned
    // Gram is rank-deficient at the HIGH window edge too (the longest-frequency
    // radial mode goes collinear), so a line-search overshoot above the maximal-
    // rank band soundly refuses the design-realization skip → O(n) reset_surface,
    // and the deficient pinning ψ it records makes the NEXT in-band trial reset a
    // second time. Clamping the optimizer's UPPER bound to this n-free k-space
    // ceiling keeps every trial inside the band. The κ-optimum lives well inside
    // it, so the clamp only excludes over-fit (too-short) length scales.
    let mut psi_rank_stable_ceiling: Option<f64> = None;
    let nfree_penalty_capable =
        coord_dim == 1 && family.is_gaussian_identity() && ctx.cache.supports_nfree_penalty_rekey();
    if nfree_penalty_capable {
        let psi_lo = lower[rho_dim];
        let psi_hi = upper[rho_dim];
        let z = Array1::from_iter(y.iter().zip(offset.iter()).map(|(yi, oi)| yi - oi));
        let theta_probe_base = theta0.clone();
        // Disjoint mutable borrows of `cache` (in the realizer) and
        // `evaluator` (the build target) — both fields of `ctx`.
        let SpatialJointContext {
            cache, evaluator, ..
        } = &mut ctx;
        let attached = evaluator.build_and_set_psi_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                cache.ensure_theta(&theta_probe).map_err(|e| e.to_string())?;
                Ok(cache.design().design.clone())
            },
            weights,
            z.view(),
            psi_lo,
            psi_hi,
        );
        if attached {
            log::info!(
                "[{label}] certified ψ-gram tensor over [{psi_lo:.3}, {psi_hi:.3}]: \
                 in-window trials assemble Gaussian sufficient statistics n-free"
            );
            // #1033: read the n-free rank-stable κ-floor off the k-space tensor.
            // Only lift INTO the window (never below psi_lo, never above the seed
            // ψ — the seed is the geometric-mean midpoint and is well clear of the
            // degenerate band), so the optimizer never starts outside its bounds.
            let psi_anchor = theta0[rho_dim];
            // #2448: the band search and the skip witness both decide on the
            // anchor's range projector, so its Davis–Kahan bar is what says whether
            // an edge that came back AT the anchor means "the band is that narrow"
            // or "the instrument could not resolve the question and everything
            // soundly refused". Read once and log it alongside the edge.
            let psi_projector_bar = evaluator.psi_gram_projector_error_bar(psi_anchor);
            // One bisection, not two: each `rank_stable_psi_floor` call is a
            // 64-step search with an O(k³) eigendecomposition per step.
            let psi_rank_stable_floor_raw = evaluator.psi_gram_rank_stable_floor(psi_anchor);
            psi_rank_stable_floor = psi_rank_stable_floor_raw
                .filter(|&f| f.is_finite() && f > psi_lo && f < psi_anchor);
            log::info!(
                "[KAPPA-PHASE-FLOOR] n_rows={} psi_lo={psi_lo:.6} psi_anchor={psi_anchor:.6} \
                 rank_stable_floor={psi_rank_stable_floor_raw:?} lifted={} \
                 projector_error_bar={psi_projector_bar:?}",
                data.nrows(),
                psi_rank_stable_floor.is_some(),
            );
            if let Some(floor) = psi_rank_stable_floor {
                log::info!(
                    "[{label}] rank-stable κ-floor ψ_floor={floor:.6} > window floor \
                     ψ_lo={psi_lo:.6}: lifting the optimizer lower bound to keep every \
                     in-window trial on the n-free design-realization skip (#1033). The \
                     conditioned Gram is rank-deficient below ψ_floor (longest-length-scale \
                     radial mode collapses into the nullspace), where the skip is soundly \
                     refused. The SEARCH is n-free — O(iters·k³) off the k-space tensor, \
                     zero row access — but the EDGE IS NOT AN n-INVARIANT CONSTANT of the \
                     design (#2408): the tensor is built from n rows, so its Gram is an \
                     O(1/n) relative perturbation of the continuum Gram, which moves the \
                     rank margin additively and displaces this root by \
                     sup|δ margin| / inf|d margin/dψ|. A steep cliff pins it to machine \
                     precision; a grazing crossing does not. Treat it as a clamp carrying \
                     that transport bound, not as the n-independent answer."
                );
            }
            // #1033: read the n-free rank-stable κ-CEILING (symmetric twin of the
            // floor). Only clamp INTO the window (strictly below psi_hi, strictly
            // above the seed ψ — the seed is the geometric-mean midpoint, well
            // inside the maximal-rank band), so the optimizer never starts outside
            // its bounds. This is the fix for the n=16000 fast-ladder resets: the
            // line search overshot to ψ≈1.0 (rank 11→10 at the high edge), tripping
            // two O(n) reset_surface calls; clamping the upper bound keeps the
            // search inside the band where the n-free skip stays sound.
            let psi_rank_stable_ceiling_raw = evaluator.psi_gram_rank_stable_ceiling(psi_anchor);
            psi_rank_stable_ceiling = psi_rank_stable_ceiling_raw
                .filter(|&c| c.is_finite() && c < psi_hi && c > psi_anchor);
            log::info!(
                "[KAPPA-PHASE-CEIL] n_rows={} psi_hi={psi_hi:.6} psi_anchor={psi_anchor:.6} \
                 rank_stable_ceiling={psi_rank_stable_ceiling_raw:?} clamped={} \
                 projector_error_bar={psi_projector_bar:?}",
                data.nrows(),
                psi_rank_stable_ceiling.is_some(),
            );
            if let Some(ceiling) = psi_rank_stable_ceiling {
                log::info!(
                    "[{label}] rank-stable κ-ceiling ψ_ceil={ceiling:.6} < window ceiling \
                     ψ_hi={psi_hi:.6}: clamping the optimizer upper bound to keep every \
                     in-window trial on the n-free design-realization skip (#1033). The \
                     conditioned Gram is rank-deficient above ψ_ceil (longest-frequency \
                     radial mode goes collinear), where the skip is soundly refused; a \
                     line-search overshoot there trips the O(n) reset_surface lane (and the \
                     deficient pinning ψ it records resets the next in-band trial too)."
                );
            }
            // #2448: when the anchor's range projector is not resolved to the
            // subspace tolerance, `reduced_basis_equal` refuses EVERY non-trivial
            // pair, so both band edges collapse onto the anchor and get filtered
            // out above — indistinguishable in the log from "the band already
            // covers the window". It is not the same thing at all: the n-free
            // design-realization skip is dead for the whole fit and every trial
            // falls to the O(n) exact path. Say so once, loudly, so the resulting
            // wall-clock is attributable to the geometry rather than mysterious.
            if let Some(bar) = psi_projector_bar
                && bar > gam_solve::psi_gram_tensor::PSI_GRAM_SKIP_PROJ_ATOL
            {
                log::warn!(
                    "[{label}] ψ-gram range projector at the anchor ψ={psi_anchor:.6} is \
                     UNRESOLVED: Davis–Kahan bar {bar:.3e} exceeds the {:.3e} subspace \
                     tolerance the design-revision skip gates on (#2448). The conditioned \
                     Gram has no kept/dropped eigen-gap wide enough to decide subspace \
                     identity at double precision here — its spectrum decays smoothly \
                     through the rank cutoff instead of cliffing — so the skip witness \
                     soundly refuses every trial and the n-free fast path will not fire \
                     at all. Results are unaffected (the exact O(n) path runs); the cost \
                     is the fast path. The lever is the geometry (basis size / centers) \
                     or the rank cutoff, not this clamp.",
                    gam_solve::psi_gram_tensor::PSI_GRAM_SKIP_PROJ_ATOL
                );
            }
            let gradient_covers_full_window = evaluator.psi_gram_tensor_covers_gradient(psi_lo)
                && evaluator.psi_gram_tensor_covers_gradient(psi_hi);
            if gradient_covers_full_window {
                log::info!(
                    "[{label}] certified ψ-gram tensor gradient lane covers the full \
                     optimizer window [{psi_lo:.3}, {psi_hi:.3}]"
                );
            } else {
                log::info!(
                    "[{label}] ψ-gram tensor value lane certified, but the gradient lane \
                     does not cover the full optimizer window [{psi_lo:.3}, {psi_hi:.3}]; \
                     keeping exact streamed kappa routing"
                );
            }
            // #1033 penalty lane: ψ also moves the penalty `S(ψ)` (the
            // Duchon/ThinPlate Hilbert scale is an analytic function of the
            // length-scale, built from the FROZEN basis CENTERS — not the data
            // rows). The design-revision fast path that the Gram tensor enables
            // SKIPS `reset_surface`, the only place the canonical penalty surface
            // is rebuilt; without re-keying, the inner solve would pair
            // `XᵀWX(ψ_new)` with the stale `S(ψ_old)` and converge to the wrong
            // β̂ / κ-optimum. Rather than interpolate `S(ψ)`, the fast path rebuilds
            // it EXACTLY and n-free per trial from the frozen geometry via
            // `cache.canonical_penalties_at(theta)` (the SAME
            // `canonicalize_penalty_specs` pipeline the slow `reset_surface` runs).
            // Here we only DECLARE the capability to the evaluator; the per-trial
            // staging happens in `eval_full` / `eval_cost`. The skip is enabled
            // exactly when the single spatial term's frozen metadata
            // (Duchon/ThinPlate) admits the exact rebuild. Matérn deliberately
            // does not enter this block: mixing tensor value probes with exact
            // streamed gradients/Hessians changed its selected κ enough to miss
            // the truth-recovery quality gate, so Matérn stays on one exact
            // streamed objective for value, gradient, and Hessian.
            evaluator.set_supports_nfree_penalty_rekey(true);
            log::info!(
                "[{label}] exact n-free ψ-penalty re-key enabled over [{psi_lo:.3}, \
                 {psi_hi:.3}]: in-window fast-path trials rebuild S(ψ) n-free from frozen \
                 geometry (no reset_surface)"
            );
        } else {
            log::info!(
                "[{label}] ψ-gram tensor did not certify over [{psi_lo:.3}, {psi_hi:.3}]; \
                 keeping the exact per-trial path"
            );
        }
        // #1033 (n-independent outer loop): with the n-free Gaussian lane fully
        // armed (Gram tensor attached + exact n-free penalty re-key), the design-
        // realization skip serves the criterion AND the ψ-gradient `(a_j, g_j)`
        // n-free for every in-window trial — but ONLY a `ValueAndGradient` eval
        // takes that skip. A `ValueGradientHessian` eval sets `allow_second_order`,
        // which forces `ensure_theta` → `reset_surface` (the O(n) design re-
        // realization) because the outer Hessian curvature `B_j` is the exact
        // n-dependent slab. So second-order outer steps are the LAST O(n) per-trial
        // cost in the κ search, and they make the outer loop scale with n. Route
        // gradient-only here: the spatial length-scale objective is smooth and the
        // budget policy already establishes that gradient-only quasi-Newton
        // converges to the same optimum strictly cheaper per eval past the pair-
        // Hessian budget — and with the tensor, the realized Hessian is the only
        // remaining expensive operation, so the same argument applies for ANY n
        // once the lane is armed. This keeps every in-window κ-trial on the n-free
        // `ValueAndGradient` skip, delivering the n-independent outer loop. The
        // exact second-order geometry is preserved whenever the lane is NOT armed
        // for gradient-only routing (non-Gaussian, multi-term, Matérn, or an
        // uncertified window), where it still pays O(n) per Hessian but keeps the
        // quality-sensitive exact second-order path.
        if attached
            && evaluator.psi_gram_tensor_covers_gradient(psi_lo)
            && evaluator.psi_gram_tensor_covers_gradient(psi_hi)
            && evaluator.supports_nfree_penalty_rekey()
            && cache.supports_nfree_gradient_only_routing()
        {
            suppress_outer_hessian_for_nfree = true;
            log::info!(
                "[{label}] n-free Gaussian ψ-lane armed; routing the SEARCH gradient-only \
                 (BFGS, fixed-point lane off) so no in-window κ-trial realizes the O(n) \
                 second-order slab — n-independent outer loop (#1033). The terminal \
                 certificate keeps its one exact curvature evaluation (gam#2760)."
            );
        }
    } else if coord_dim == 1 && family.is_gaussian_identity() {
        log::info!(
            "[{label}] exact n-free ψ-penalty re-key unavailable; skipping ψ-gram tensor \
             attachment so value, gradient, and Hessian remain on the same exact streamed \
             objective"
        );
    }
    Ok(PreparedExactJointSpatialRoute {
        ctx,
        analytic_outer_hessian_available,
        suppress_outer_hessian_for_nfree,
        psi_rank_stable_floor,
        psi_rank_stable_ceiling,
    })
}

/// Apply a length scale to a single `SmoothTermSpec` (independent of any
/// outer `TermCollectionSpec`). Mirrors `set_spatial_length_scale` but on a
/// term in isolation; used by the incremental realizer's cached planned spec.
fn set_single_term_spatial_length_scale(
    term: &mut SmoothTermSpec,
    length_scale: f64,
) -> Result<(), EstimationError> {
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale.set_resolved(length_scale);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.length_scale = Some(length_scale);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not expose a spatial length scale",
            term.name
        ))),
    }
}

/// Apply anisotropy contrasts to a single `SmoothTermSpec`. Mirrors
/// `set_spatial_aniso_log_scales` but on a term in isolation; used by the
/// incremental realizer's cached planned spec.
fn set_single_term_spatial_aniso_log_scales(
    term: &mut SmoothTermSpec,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let eta = center_aniso_log_scales(&eta);
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}

/// Freeze the design-moving representer length-scale dial on every measure-jet
/// term in `spec` (sets `learn_length_scale = false`), so ℓ stays at its
/// realized auto value with no outer REML enrollment.
///
/// Used by COUPLED-block families (bernoulli marginal-slope: a shared mjs
/// surface feeds both the marginal mean and the slope). In that coupling a
/// design-moving kernel-scale dial on the shared covariates is an
/// identifiability hazard: the outer search can reach a sharp ℓ at which a
/// marginal smooth direction trades off against the slope into a
/// separation-scale runaway (#1116). A single Gaussian surface has no such
/// coupling and keeps ℓ learnable. Returns the number of terms frozen.
/// The signed sectional curvature κ of a constant-curvature smooth at
/// `term_idx`, or `None` if that term is not a `curv(...)` smooth. After a fit
/// with κ-optimization enabled this reads the **fitted κ̂** out of the resolved
/// spec (`freeze_term_collection_from_design` writes the optimized κ back into
/// the spec, and `BasisMetadata::ConstantCurvature.kappa` carries the same
/// value). This is the headline #944 estimand accessor — the κ̂ in
/// "κ̂ = −1.8 (95% CI …)". Mirrors [`get_spatial_length_scale`].
pub fn get_constant_curvature_kappa(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    constant_curvature_term_spec(spec, term_idx).map(|cc| cc.kappa)
}

/// `true` when `term_idx` is a `curv(...)` smooth whose user PINNED the
/// sectional curvature with an explicit `kappa=` (the mgcv-`sp=` convention,
/// gam#2152). A pinned κ is a fixed geometry: the outer loop must hold it
/// constant and never run the continuous curvature profile optimizer on
/// that term. Non-CC terms and CC terms whose `kappa=` was omitted (κ free,
/// #944/#1464 estimation) return `false`.
pub fn constant_curvature_kappa_is_fixed(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    constant_curvature_term_spec(spec, term_idx).is_some_and(|cc| cc.kappa_fixed)
}

/// `true` when `term_idx` is a `curv(...)` smooth whose user PINNED the kernel
/// range with an explicit `length_scale=` — the same mgcv-`sp=` convention
/// [`constant_curvature_kappa_is_fixed`] reports for the curvature (gam#2747).
/// A pinned range is a fixed kernel resolution: the curvature profile must hold
/// it constant, and the κ̂ it then reports is conditional on that choice. CC
/// terms whose `length_scale=` was omitted (range free, estimated) and non-CC
/// terms return `false`.
pub(crate) fn constant_curvature_length_scale_is_fixed(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> bool {
    constant_curvature_term_spec(spec, term_idx).is_some_and(|cc| cc.length_scale_fixed)
}

/// Indices of every constant-curvature (`curv(...)`) smooth term in `spec`.
pub(crate) fn constant_curvature_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    (0..spec.smooth_terms.len())
        .filter(|&idx| constant_curvature_term_spec(spec, idx).is_some())
        .collect()
}

#[derive(Debug, Clone)]
struct SingleSmoothTermRealization {
    design_local: DesignMatrix,
    term: SmoothTerm,
}

/// Wrap a fresh `LocalSmoothTermBuild` (produced by `build_single_local_smooth_term`)
/// into a `SingleSmoothTermRealization`. Mirrors the single-term portion of
/// `build_smooth_design_withworkspace_unvalidated`, but skips the joint center
/// planner and per-term workspace fork — the realizer drives κ-only rebuilds
/// directly with its persistent workspace so basis caches survive across BFGS
/// κ proposals.
fn wrap_local_build_as_realization(
    mut local: LocalSmoothTermBuild,
    termspec: &SmoothTermSpec,
) -> Result<SingleSmoothTermRealization, String> {
    let p_local = local.dim;
    let lb_local = if local.box_reparam {
        shape_lower_bounds_local(termspec.shape, p_local)
    } else {
        None
    };

    // Stage-2 joint-null absorption rotation, same logic as the main
    // aggregation loop in `build_smooth_design_withworkspace_unvalidated`:
    // apply Q when Some AND the smooth has no shape constraints.
    let applied_rotation: Option<gam_terms::basis::JointNullRotation> = match (
        local.joint_null_rotation.take(),
        lb_local.is_some(),
        local.linear_constraints.is_some(),
    ) {
        (Some(rot), false, false) => {
            let q = &rot.rotation;
            local.design =
                apply_smooth_transform_to_design(local.design.clone(), q, &termspec.name).map_err(
                    |e| {
                        format!(
                            "joint-null absorption rotation failed for term '{}': {}",
                            termspec.name, e
                        )
                    },
                )?;
            for penalty in &mut local.active_penalties {
                let qt_s = gam_linalg::faer_ndarray::fast_atb(q, &penalty.matrix);
                penalty.matrix = gam_linalg::faer_ndarray::fast_ab(&qt_s, q);
                penalty.null_eigenvectors = penalty
                    .null_eigenvectors
                    .as_ref()
                    .map(|basis| gam_linalg::faer_ndarray::fast_atb(q, basis));
                // Same transport the aggregation loop this mirrors performs:
                // `Q` is orthogonal, so `null(Qᵀ S Q) = Qᵀ null(S)` and a
                // declared structural null frame moves with it. Leaving it in
                // pre-rotation coordinates would hand the double-penalty
                // rebuild a frame for the wrong chart (#2761).
                penalty.info.structural_null_frame = penalty
                    .info
                    .structural_null_frame
                    .as_ref()
                    .map(|frame| gam_linalg::faer_ndarray::fast_atb(q, frame));
                // `QᵀAᵀAQ = (AQ)ᵀ(AQ)`: the energy factor moves by right
                // multiplication.
                penalty.info.energy_factor = penalty
                    .info
                    .energy_factor
                    .as_ref()
                    .map(|factor| gam_linalg::faer_ndarray::fast_ab(factor, q));
                penalty.op = None;
                penalty.info.kronecker_factors = None;
            }
            Some(rot)
        }
        (Some(_), _, _) => None,
        (None, _, _) => None,
    };

    let smooth_term = SmoothTerm {
        parametric_residualization: None,
        // A single-term realization decides no gauge. The caller splices this
        // into a collection design and re-applies THAT collection's gauge
        // (#2747); it must never claim one of its own.
        collection_gauge: None,
        name: termspec.name.clone(),
        coeff_range: 0..p_local,
        shape: termspec.shape,
        active_penalties: local.active_penalties.clone(),
        dropped_penalties: local.dropped_penalties.clone(),
        metadata: local.metadata.clone(),
        lower_bounds_local: lb_local,
        linear_constraints_local: local.linear_constraints.clone(),
        joint_null_rotation: applied_rotation,
        // Single-term realizations never run the global ownership pass, so
        // there is no overlap residualization to export here (#978).
        unabsorbed_global_orthogonality: None,
    };

    Ok(SingleSmoothTermRealization {
        design_local: local.design,
        term: smooth_term,
    })
}

/// Extract the κ-invariant pieces of a freshly-built spatial basis — center
/// cloud (in standardized coords) and `input_scale` — and bake them into a
/// `SmoothTermSpec` whose `center_strategy` becomes `UserProvided` and whose
/// `input_scale` is `Some`. Subsequent rebuilds driven from this cached spec
/// will short-circuit `select_centers_by_strategy` (KMeans / FarthestPoint /
/// EqualMass cluster searches over n×d data) and isotropic scale estimation,
/// leaving only the κ-dependent kernel
/// values and basis assembly. Returns `None` for non-spatial families or when
/// the metadata does not yet expose the required pieces (for instance when a
/// ThinPlate request was auto-promoted to Duchon during the build).
fn freeze_geometry_from_metadata(
    termspec: &SmoothTermSpec,
    metadata: &BasisMetadata,
) -> Option<SmoothTermSpec> {
    let mut frozen = termspec.clone();
    match (&mut frozen.basis, metadata) {
        (
            SmoothBasisSpec::Matern {
                spec,
                input_scale: spec_scale,
                ..
            },
            BasisMetadata::Matern {
                centers,
                input_scale: metadata_scale,
                identifiability_transform,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            *spec_scale = Some(*metadata_scale);
            // Freeze the cold-build coefficient chart. Double-penalty topology
            // is structural (the explicit intercept only), so no numerical
            // nullspace decision needs to be carried across κ trials.
            if let Some(transform) = identifiability_transform.clone() {
                spec.identifiability = MaternIdentifiability::FrozenTransform { transform };
            }
            Some(frozen)
        }
        (
            SmoothBasisSpec::Duchon {
                spec,
                input_scale: spec_scale,
                ..
            },
            BasisMetadata::Duchon {
                centers,
                input_scale: metadata_scale,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            *spec_scale = Some(*metadata_scale);
            // The #1355 data-metric radial chart `V` is NOT re-frozen here, and
            // that is deliberate (gam#2760): the realizer's replay spec already
            // carries it from `freeze_term_collection_from_design`, which copies
            // `radial_reparam` off the same metadata. Setting it a second time
            // would give one fact two owners — the defect class this file is
            // otherwise removing. Measured while chasing this issue's blocker:
            // a rebuild that re-DERIVES `V` prunes radial modes as the kernel
            // flattens (12 → 10 → 7 → 5 columns over ℓ = 1 … 100 on the
            // `kappa_loop_n_scaling` spec), so the replay is load-bearing — it
            // is simply already in force by the time this runs.
            Some(frozen)
        }
        (
            SmoothBasisSpec::ThinPlate {
                spec,
                input_scale: spec_scale,
                ..
            },
            BasisMetadata::ThinPlate {
                centers,
                input_scale: metadata_scale,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            *spec_scale = Some(*metadata_scale);
            Some(frozen)
        }
        // Family mismatch (e.g. ThinPlate auto-promotion to Duchon) leaves the
        // cache empty; we'll retry materialization on the next κ apply.
        _ => None,
    }
}

/// Put a replay spec's identifiability back to the TERM-LOCAL chart the
/// collection gauge was derived on (gam#2760).
///
/// Used for exactly one thing: a term whose collection applied a
/// [`gam_terms::smooth::SmoothCollectionGauge`], whose `(T, R)` pair is re-derived
/// at every ψ rebuild and must therefore not ALSO arrive frozen inside the spec.
/// `Some(z)` replays the term's own chart verbatim (a center sum-to-zero frame,
/// a linear-orthogonality frame, a caller's frozen chart — all ψ-independent);
/// `None` states that the local build applied none, which is the radial families'
/// ordinary case, where `OrthogonalToParametric` defers to the gauge entirely.
///
/// Only the families the spatial outer search rebuilds are listed. A gauged term
/// of any other family is never re-realized by this realizer, so its replay spec
/// is left exactly as the freeze wrote it.
fn restore_local_identifiability_chart(
    replay: &mut SmoothBasisSpec,
    local_chart: Option<&Array2<f64>>,
) {
    let spatial = |chart: Option<&Array2<f64>>| match chart {
        Some(transform) => SpatialIdentifiability::FrozenTransform {
            transform: transform.clone(),
        },
        None => SpatialIdentifiability::None,
    };
    if let SmoothBasisSpec::Duchon { spec, .. } = &mut *replay {
        spec.identifiability = spatial(local_chart);
    }
    if let SmoothBasisSpec::ThinPlate { spec, .. } = &mut *replay {
        spec.identifiability = spatial(local_chart);
    }
    if let SmoothBasisSpec::Matern { spec, .. } = &mut *replay {
        spec.identifiability = match local_chart {
            Some(transform) => MaternIdentifiability::FrozenTransform {
                transform: transform.clone(),
            },
            None => MaternIdentifiability::None,
        };
    }
    // These two families have no "no chart" policy — their local build always
    // applies a center sum-to-zero section — so a `None` here would be a claim
    // the enum cannot express. It is left alone instead of invented, and a
    // gauged term of theirs whose metadata carried no transform keeps whatever
    // the freeze wrote (which is that same `CenterSumToZero` default).
    if let (SmoothBasisSpec::ConstantCurvature { spec, .. }, Some(transform)) =
        (&mut *replay, local_chart)
    {
        spec.identifiability = gam_terms::basis::ConstantCurvatureIdentifiability::FrozenTransform {
            transform: transform.clone(),
        };
    }
    if let (SmoothBasisSpec::MeasureJet { spec, .. }, Some(transform)) = (&mut *replay, local_chart)
    {
        spec.identifiability = gam_terms::basis::MeasureJetIdentifiability::FrozenTransform {
            transform: transform.clone(),
        };
    }
}

/// Shape of the frozen radial chart a rebuild spec carries, for diagnostics.
fn spatial_frozen_radial_chart_shape(termspec: &SmoothTermSpec) -> Option<(usize, usize)> {
    match &termspec.basis {
        SmoothBasisSpec::Duchon { spec, .. } => spec.radial_reparam.as_ref().map(|v| v.dim()),
        SmoothBasisSpec::ThinPlate { spec, .. } => spec.radial_reparam.as_ref().map(|v| v.dim()),
        _ => None,
    }
}

/// Shape of the radial chart a realized basis reports, for diagnostics.
fn spatial_realized_radial_chart_shape(metadata: &BasisMetadata) -> Option<(usize, usize)> {
    match metadata {
        BasisMetadata::Duchon { radial_reparam, .. } => radial_reparam.as_ref().map(|v| v.dim()),
        BasisMetadata::ThinPlate { radial_reparam, .. } => radial_reparam.as_ref().map(|v| v.dim()),
        _ => None,
    }
}

fn rebuild_smooth_auxiliary_state(
    smooth: &mut SmoothDesign,
    dropped_penaltyinfo_by_term: &[Vec<DroppedPenaltyBlockInfo>],
) -> Result<(), String> {
    if dropped_penaltyinfo_by_term.len() != smooth.terms.len() {
        return Err(SmoothError::dimension_mismatch(format!(
            "smooth dropped-penalty cache mismatch: terms={}, dropped_sets={}",
            smooth.terms.len(),
            dropped_penaltyinfo_by_term.len()
        ))
        .into());
    }

    let total_p = smooth.total_smooth_cols();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for term in &smooth.terms {
        let range = term.coeff_range.clone();
        if let Some(lb_local) = term.lower_bounds_local.as_ref() {
            if lb_local.len() != range.len() {
                return Err(SmoothError::dimension_mismatch(format!(
                    "smooth lower-bound cache mismatch for term '{}': bounds={}, coeffs={}",
                    term.name,
                    lb_local.len(),
                    range.len()
                ))
                .into());
            }
            coefficient_lower_bounds
                .slice_mut(s![range.clone()])
                .assign(lb_local);
            any_bounds = true;
        }
        if let Some(lin_local) = term.linear_constraints_local.as_ref() {
            if lin_local.a.ncols() != range.len() {
                return Err(SmoothError::dimension_mismatch(format!(
                    "smooth linear-constraint cache mismatch for term '{}': cols={}, coeffs={}",
                    term.name,
                    lin_local.a.ncols(),
                    range.len()
                ))
                .into());
            }
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![range.clone()]).assign(&lin_local.a.row(r));
                linear_constraintrows.push(row);
                linear_constraint_b.push(lin_local.b[r]);
            }
        }
    }

    smooth.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    smooth.linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), total_p));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    smooth.dropped_penaltyinfo = dropped_penaltyinfo_by_term
        .iter()
        .flat_map(|infos| infos.iter().cloned())
        .collect();
    Ok(())
}

fn rebuild_term_collection_auxiliary_state(
    spec: &TermCollectionSpec,
    design: &mut TermCollectionDesign,
) -> Result<(), String> {
    if spec.linear_terms.len() != design.linear_ranges.len() {
        return Err(SmoothError::dimension_mismatch(format!(
            "term-collection linear bookkeeping mismatch: spec_terms={}, design_ranges={}",
            spec.linear_terms.len(),
            design.linear_ranges.len()
        ))
        .into());
    }

    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for (linear, (_, range)) in spec.linear_terms.iter().zip(design.linear_ranges.iter()) {
        if range.len() != 1 {
            return Err(SmoothError::dimension_mismatch(format!(
                "linear term '{}' expected one coefficient column, found {}",
                linear.name,
                range.len()
            ))
            .into());
        }
        let col = range.start;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
    }

    if let Some(lb_smooth) = design.smooth.coefficient_lower_bounds.as_ref() {
        if lb_smooth.len() != design.smooth.total_smooth_cols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "smooth lower-bound width mismatch: bounds={}, smooth_cols={}",
                lb_smooth.len(),
                design.smooth.total_smooth_cols()
            ))
            .into());
        }
        coefficient_lower_bounds
            .slice_mut(s![
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = design.smooth.linear_constraints.as_ref() {
        if lin_smooth.a.ncols() != design.smooth.total_smooth_cols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "smooth linear-constraint width mismatch: cols={}, smooth_cols={}",
                lin_smooth.a.ncols(),
                design.smooth.total_smooth_cols()
            ))
            .into());
        }
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        a_global
            .slice_mut(s![
                ..,
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };

    design.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    design.linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints)
            .map_err(|error| error.to_string())?;
    design.dropped_penaltyinfo = design.smooth.dropped_penaltyinfo.clone();
    Ok(())
}

fn theta_values_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(&l, &r)| l.to_bits() == r.to_bits())
}

fn latent_values_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    theta_values_match(left, right)
}

fn spatial_aniso_matches(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(&x, &y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}

fn spatial_length_scale_matches(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}

struct FrozenTermCollectionIncrementalRealizer<'d> {
    data: ArrayView2<'d, f64>,
    spec: TermCollectionSpec,
    design: TermCollectionDesign,
    fixed_blocks: Vec<DesignBlock>,
    dropped_penaltyinfo_by_term: Vec<Vec<DroppedPenaltyBlockInfo>>,
    smooth_penalty_ranges: Vec<Range<usize>>,
    full_penalty_ranges: Vec<Range<usize>>,
    /// Persistent workspace for basis cache reuse across κ proposals.
    /// Distance matrices are cached here so they're computed once and
    /// reused across repeated `apply_log_kappa_to_term` calls.
    basisworkspace: gam_terms::basis::BasisWorkspace,
    /// Per-term cached realization geometry for incremental κ updates.
    ///
    /// On the first κ-driven rebuild of term `i`, this slot is populated with a
    /// `SmoothTermSpec` whose κ-invariant geometry — center cloud (as
    /// `CenterStrategy::UserProvided`) and `input_scale` — has been frozen
    /// out of the realized basis metadata. Subsequent
    /// `apply_log_kappa_to_term` calls reuse this spec, mutating only the
    /// κ / aniso fields. This short-circuits `select_centers_by_strategy`
    /// (KMeans / FarthestPoint / EqualMass cluster searches over the n×d data
    /// matrix) and isotropic scale estimation over n rows on every BFGS
    /// κ-eval, leaving the kernel-value pass and
    /// basis assembly as the only work.
    spatial_realization_geometry: Vec<Option<SmoothTermSpec>>,
    /// Monotonic counter incremented every time `apply_log_kappa` actually
    /// rebuilds the realized design / smooth penalties. Read by the
    /// design-revision-counter fast path in `ExternalJointHyperEvaluator`
    /// to skip redundant canonical-penalty rebuilds and cache wipes when
    /// the outer BFGS loop probes the same ψ twice in a row.
    design_revision: u64,
}

impl<'d> std::fmt::Debug for FrozenTermCollectionIncrementalRealizer<'d> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenTermCollectionIncrementalRealizer")
            .field("data_shape", &(self.data.nrows(), self.data.ncols()))
            .field("fixed_blocks", &self.fixed_blocks.len())
            .finish_non_exhaustive()
    }
}

/// Translate the authoritative emitted global penalty layout into the two
/// coordinate systems the incremental realizer updates.
///
/// The model-global ranges come directly from `TermCollectionDesign`; the
/// smooth-local ranges are their exact translation past the recorded leading
/// penalty prefix. Keeping this outside the constructor makes the layout
/// invariant independently testable without constructing any κ-specific
/// spatial caches.
fn emitted_smooth_penalty_ranges(
    design: &TermCollectionDesign,
) -> Result<(Vec<Range<usize>>, Vec<Range<usize>>), String> {
    let leading = design.leading_penalty_blocks_before_smooth();
    let mut smooth_penalty_ranges = Vec::with_capacity(design.smooth.terms.len());
    let mut full_penalty_ranges = Vec::with_capacity(design.smooth.terms.len());
    let mut smooth_cursor = 0usize;
    for term_idx in 0..design.smooth.terms.len() {
        let full_range = design.smooth_term_penalty_range(term_idx)?;
        match full_range {
            Some(full_range) => {
                let local_start = full_range.start.checked_sub(leading).ok_or_else(|| {
                    "incremental realizer smooth penalty range precedes the emitted smooth prefix"
                        .to_string()
                })?;
                let local_end = full_range.end.checked_sub(leading).ok_or_else(|| {
                    "incremental realizer smooth penalty range precedes the emitted smooth prefix"
                        .to_string()
                })?;
                if local_start != smooth_cursor {
                    return Err(format!(
                        "incremental realizer non-contiguous emitted smooth layout at term {term_idx}: expected local start {smooth_cursor}, got {local_start}"
                    ));
                }
                smooth_cursor = local_end;
                smooth_penalty_ranges.push(local_start..local_end);
                full_penalty_ranges.push(full_range);
            }
            None => {
                smooth_penalty_ranges.push(smooth_cursor..smooth_cursor);
                let global_cursor = leading.checked_add(smooth_cursor).ok_or_else(|| {
                    "incremental realizer empty smooth penalty range overflow".to_string()
                })?;
                full_penalty_ranges.push(global_cursor..global_cursor);
            }
        }
    }
    if smooth_cursor != design.smooth.penalties.len() {
        return Err(format!(
            "incremental realizer smooth penalty mismatch: ranged={}, actual={}",
            smooth_cursor,
            design.smooth.penalties.len()
        ));
    }
    Ok((smooth_penalty_ranges, full_penalty_ranges))
}

impl<'d> FrozenTermCollectionIncrementalRealizer<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
    ) -> Result<Self, String> {
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        Self::new_with_policy(data, spec, design, &policy)
    }

    fn new_with_policy(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Self, String> {
        if spec.smooth_terms.len() != design.smooth.terms.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer smooth term mismatch: spec_terms={}, design_terms={}",
                spec.smooth_terms.len(),
                design.smooth.terms.len()
            ))
            .into());
        }

        // Cache the exact ranges reported by the emitted global layout. Do not
        // reconstruct a second global offset from term specs or coefficient
        // blocks: unpenalized fixed/random effects own columns but emit no
        // penalty, and multi-penalty smooths own more than one coordinate.
        let (smooth_penalty_ranges, full_penalty_ranges) = emitted_smooth_penalty_ranges(&design)?;
        // The emitted collection design is also the authority for the replay
        // specification. In particular, global smooth identifiability can
        // restrict a source term's coefficient chart and eliminate a dependent
        // double-penalty ridge. Retaining the caller's raw pre-assembly spec
        // would let the first κ proposal rebuild in that obsolete chart even
        // though every cached range below describes the emitted chart (#2433).
        //
        // Freeze once at this ownership boundary so value rebuilds, analytic
        // derivatives, and the geometry cache all start from the same centers,
        // scaling, identifiability transform, and penalty topology.
        //
        // EXCEPT the half a collection GAUGE owns (gam#2760). The gauge carries
        // the fixed row space `C` AND the fixed reference coefficient chart
        // `T0`; a moving-psi value is represented canonically as
        // `P_C X_local(psi) T0`. The freeze above writes the metadata's composed
        // `z_local*T0` into the replay spec. Leaving that composition in place
        // would apply `T0` once in the local rebuild and once again when the
        // gauge performs its fixed-chart placement.
        //
        // MEASURED on the `kappa_loop_n_scaling` fixture's own spec
        // (`examples/probe_2760_replay_gauge_double_apply`, 12 centers, n = 600,
        // one Duchon term, `arm=Delete`, `C = [1]`, replay chart
        // `FrozenTransform(12, 11)`) — the orthogonality residual of the
        // singly-charted rebuild against the gauge's own block:
        //
        //   ℓ = 1.0 (the fit's own)   1.5e-12   the frozen chart is right here
        //   ℓ = 0.5                   9.0e-1    and stale everywhere else
        //   ℓ = 2.0                   9.5e-1
        //
        // so the gauge resolves a direction and DELETES one, at every ψ
        // including the seed. The term then reaches the splice one column short
        // (11 → 10 against a cached 11) and every κ fixture on a Duchon term
        // refuses in 0.2 s with `incremental realizer width mismatch`. On the
        // `Residualize` arm the second application is idempotent (`P² = P`),
        // which is why the Matérn fixture the gauge work was verified on stayed
        // green while every Duchon one went red.
        //
        // So a gauged term's replay spec is put back into the TERM-LOCAL chart
        // the fixed `T0` was derived on. The gauge carries both pieces because
        // the composition in metadata cannot be decomposed after the fact.
        // Everything else the freeze decides (centers, input scale, radial
        // chart, penalty topology) is psi-invariant and is kept.
        //
        // The caller's own spec is NOT the source: by the time it reaches this
        // realizer it has already been frozen at least once upstream, so its
        // policy is itself a composed transform.
        let mut spec = freeze_term_collection_from_design(&spec, &design)
            .map_err(|e| format!("failed to freeze incremental replay specification: {e}"))?;
        for (term_idx, term) in design.smooth.terms.iter().enumerate() {
            let Some(gauge) = term.collection_gauge.as_ref() else {
                continue;
            };
            let Some(replay) = spec.smooth_terms.get_mut(term_idx) else {
                continue;
            };
            restore_local_identifiability_chart(
                &mut replay.basis,
                gauge.local_identifiability_transform.as_ref(),
            );
            // The rotation `Q` the collection applied BEFORE it derived `T0`
            // (gam#2760). The freeze copied the term's own `joint_null_rotation`,
            // which the collection cleared once it composed `Q·T0` into the
            // metadata, and the chart restored above is the pre-`Q` one — so
            // without this the replay put an unrotated block through a chart
            // derived on the rotated one. The local build honours a persisted
            // rotation instead of re-deriving one, and
            // `wrap_local_build_as_realization` applies it before the gauge
            // applies `T0`: the collection's own order.
            replay.joint_null_rotation = gauge.joint_null_rotation.clone();
        }
        let spec = spec;
        let fixed_blocks = build_term_collection_fixed_blocks(data, &spec)
            .map_err(|e| format!("failed to cache fixed term-collection blocks: {e}"))?;

        // The collection design is the authority for the realized coefficient
        // chart and penalty topology. Do not rebuild each source term in
        // isolation here: that bypasses `apply_global_smooth_identifiability`,
        // whose constrained-primary analysis can legitimately eliminate a
        // double-penalty ridge. Re-deriving the term therefore manufactured a
        // second, incompatible topology before the incremental realizer even
        // received its first κ proposal (#2433).
        //
        // Carry the collection's certified dropped-penalty facts directly.
        // Later κ rebuilds still pass through `replace_term_realization`, whose
        // topology guard compares each new realization with the authoritative
        // emitted penalty range and continues to hard-fail a genuine topology
        // change.
        let dropped_penaltyinfo_by_term: Vec<Vec<DroppedPenaltyBlockInfo>> = design
            .smooth
            .terms
            .iter()
            .map(|term| {
                term.dropped_penalties
                    .iter()
                    .cloned()
                    .map(|penalty| DroppedPenaltyBlockInfo {
                        termname: Some(term.name.clone()),
                        penalty,
                    })
                    .collect()
            })
            .collect();

        let geometry_slots = spec.smooth_terms.len();
        Ok(Self {
            data,
            spec,
            design,
            fixed_blocks,
            dropped_penaltyinfo_by_term,
            smooth_penalty_ranges,
            full_penalty_ranges,
            basisworkspace: gam_terms::basis::BasisWorkspace::with_policy(policy.clone()),
            spatial_realization_geometry: vec![None; geometry_slots],
            design_revision: 0,
        })
    }

    fn design_revision(&self) -> u64 {
        self.design_revision
    }

    fn spec(&self) -> &TermCollectionSpec {
        &self.spec
    }

    fn design(&self) -> &TermCollectionDesign {
        &self.design
    }

    /// True when this realizer carries exactly ONE spatial smooth term whose
    /// frozen basis geometry (`BasisMetadata::Duchon`/`ThinPlate`)
    /// admits an EXACT, n-free penalty rebuild at a new length-scale (#1033).
    /// The κ-loop fast path gates its design-realization skip on this: the skip
    /// leaves `reset_surface` un-run, so it is only sound when `S(ψ_new)` can be
    /// re-keyed n-free from the frozen geometry (centers + identifiability
    /// transform + operator collocation points), never from the data rows, AND
    /// the re-keyed penalty's block topology is IDENTICAL to the one the frozen
    /// design carries.
    ///
    /// Matérn stays on the exact slow re-key path here, but NOT for the reason
    /// #1270 originally pinned. The operator-triplet penalty re-key (#1274) IS
    /// fully landed: `canonical_penalties_at_psi` and
    /// `canonical_penalty_derivatives_at_psi` both rebuild the realized Matérn
    /// `{mass, tension, stiffness}` triplet (and its analytic ψ-derivative)
    /// n-free from the frozen collocation geometry, routed through the SAME
    /// shared `matern_operator_penalty_triplet_at_length_scale` builder the
    /// design uses — so the block topology is ψ-stable by construction and the
    /// surface is byte-identical to the slow path across the ψ window (pinned
    /// to <1e-10 by `matern_nfree_rekey_topology_tests`). The historical
    /// "the re-key cannot reproduce the operator triplet" rationale is resolved.
    ///
    /// Re-admission is nonetheless withheld because it is net-negative on the
    /// CURRENT architecture, for two independent reasons the #1274 acceptance
    /// gates surface:
    ///   1. NO SPEED WIN. Even with the penalty re-keyed, the #1264
    ///      reduced-basis-rotation soundness gate (`psi_gram_tensor_covers_skip`)
    ///      refuses Matérn's rotating collocation geometry, so the design-
    ///      realization skip still falls to the exact O(n) `reset_surface`
    ///      re-realization every trial — admitting the penalty rekey alone buys
    ///      no n-independence. Closing this needs an n-free re-key of the Matérn
    ///      *design* (Chebyshev-in-ψ Gram over the rotating basis), which is the
    ///      remaining design-scope work, not a flag flip.
    ///   2. QUALITY REGRESSION. Re-admitting Matérn (as #1033 `6a5a2e1` did,
    ///      reverted by `feb0eb5`) perturbs the selected fit enough to miss the
    ///      mgcv/GP truth-recovery bar (`matern_nu_sweep_*`) — slower AND worse.
    ///
    /// So Matérn is deliberately "slow-but-right". Duchon/ThinPlate are the
    /// #1033 acceptance lane. `matern_nfree_rekey_topology_tests` test (b) pins
    /// this negative admission contract: a flip must first re-clear both gates.
    fn supports_nfree_penalty_rekey(&self, spatial_terms: &[usize]) -> bool {
        if spatial_terms.len() != 1 {
            return false;
        }
        let term_idx = spatial_terms[0];
        matches!(
            self.design.smooth.terms.get(term_idx).map(|t| &t.metadata),
            Some(BasisMetadata::Duchon { .. } | BasisMetadata::ThinPlate { .. })
        )
    }

    /// True when the armed n-free Gaussian lane should suppress exact outer
    /// Hessians and route κ search through gradient-only BFGS.
    ///
    /// This is deliberately narrower than [`Self::supports_nfree_penalty_rekey`]:
    /// Matérn has an exact n-free operator-triplet `S(ψ)` re-key (#1274), but its
    /// quality gate still depends on the exact second-order outer route. Duchon
    /// and ThinPlate are the #1033 n-independent acceptance lane where the exact
    /// Hessian slab is the remaining O(n) per-trial cost.
    fn supports_nfree_gradient_only_routing(&self, spatial_terms: &[usize]) -> bool {
        if spatial_terms.len() != 1 {
            return false;
        }
        let term_idx = spatial_terms[0];
        matches!(
            self.design.smooth.terms.get(term_idx).map(|t| &t.metadata),
            Some(BasisMetadata::Duchon { .. } | BasisMetadata::ThinPlate { .. })
        )
    }

    /// Rebuild the EXACT canonical penalty surface `S(ψ)` at the length-scale
    /// implied by `psi`, entirely n-free (#1033). Reuses the FROZEN basis
    /// geometry from the single spatial term's `BasisMetadata` (centers,
    /// identifiability transform, operator collocation points — all `k × d`, no
    /// data rows) and the spec's `(power, nullspace_order, operator_penalties,
    /// nu, …)`; only the length-scale moves. The reconstructed term-local
    /// penalty matrices replace the `local` of the FROZEN
    /// `design.penalties` templates (whose `col_range` / `prior_mean` /
    /// `structure_hint` / `op` are ψ-invariant), so the resulting
    /// `PenaltySpec`s are bit-identical in topology to the slow path's; running
    /// them through the SAME `canonicalize_penalty_specs` pipeline yields the
    /// canonical list the kept reference surface must be re-keyed with.
    fn canonical_penalties_at_psi(
        &mut self,
        spatial_terms: &[usize],
        psi: &[f64],
        frozen_penalty_ranks: &[usize],
    ) -> Result<(Vec<gam_terms::construction::CanonicalPenalty>, Vec<usize>), String> {
        let rebuild_started = std::time::Instant::now();
        if spatial_terms.len() != 1 {
            return Err(format!(
                "n-free penalty re-key requires exactly one spatial term, found {}",
                spatial_terms.len()
            ));
        }
        let term_idx = spatial_terms[0];
        // Decode ψ with the same chart used by the slow rebuild path. For
        // Matérn, per-axis ψ entries are REML hyper-coordinates, so the n-free
        // penalty rebuild must consume the trial η contrasts as well as the
        // scalar length scale. Duchon keeps η as fixed geometry and continues
        // to use frozen metadata below.
        let (ls_opt, aniso_from_psi) = spatial_term_psi_to_length_scale_and_aniso(psi);
        // Pull the spec-level penalty configuration (which operator orders are
        // active / double_penalty) — ψ-invariant, frozen at construction.
        let termspec =
            self.spec.smooth_terms.get(term_idx).ok_or_else(|| {
                format!("spatial term {term_idx} out of range for n-free penalty")
            })?;
        let term = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("realized smooth term {term_idx} out of range"))?;
        // The per-term penalties live contiguously in the collection penalty
        // list at the term's `coeff_range` (single-spatial-term collection).
        let p_total = self.design.design.ncols();
        // The trial realization's energy factors, block by block, for the bases
        // whose builder carries one. A template's factor belongs to the build ψ,
        // so it is never reused with a trial `local`.
        let mut rekey_energy_factors: Vec<Option<Array2<f64>>> = Vec::new();
        let (locals, nullspace_dims): (Vec<Array2<f64>>, Vec<usize>) = match &term.metadata {
            BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                power,
                nullspace_order,
                aniso_log_scales,
                input_scale,
                radial_reparam,
                ..
            } => {
                let operator_penalties = match &termspec.basis {
                    SmoothBasisSpec::Duchon { spec, .. } => spec.operator_penalties.clone(),
                    _ => gam_terms::basis::DuchonOperatorPenaltySpec::default(),
                };
                // Slow-path Duchon realization stores centers/collocation points
                // in standardized coordinates and compensates the user-facing
                // length_scale by the scalar input frame before building penalties. The n-free
                // re-key must use the same effective length scale, or the fast
                // path pairs G(ψ_new) with an S(ψ_new) from a different
                // coordinate scale.
                let effective_ls = ls_opt.map(|length| {
                    input_scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length))
                        .standardized_value()
                });
                gam_terms::basis::duchon_penalties_at_length_scale(
                    centers.view(),
                    identifiability_transform.as_ref(),
                    operator_collocation_points.as_ref().map(|p| p.view()),
                    &operator_penalties,
                    *power,
                    *nullspace_order,
                    aniso_log_scales.as_deref(),
                    radial_reparam.as_ref(),
                    effective_ls,
                    &mut self.basisworkspace,
                )
                .map_err(|e| e.to_string())?
            }
            BasisMetadata::Matern {
                centers,
                periodic,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                input_scale,
                ..
            } => {
                // `spatial_term_psi_to_length_scale_and_aniso` decodes ψ to a
                // length scale in ORIGINAL data coordinates — exactly what the
                // slow-path rebuild writes into `spec.length_scale` before
                // `matern_operator_penalty_triplet_from_metadata` compensates it
                // by the scalar input frame. Compensate identically here so the n-free re-key
                // reproduces the slow-path penalty surface byte-for-byte (#706).
                let ls = ls_opt.ok_or_else(|| {
                    "Matérn n-free penalty re-key requires a finite length-scale".to_string()
                })?;
                let effective_ls = input_scale
                    .to_standardized_units(gam_terms::OriginalUnits::new(ls))
                    .standardized_value();
                let aniso_for_penalty = aniso_from_psi.as_deref().or(aniso_log_scales.as_deref());
                // Route through the SAME canonical operator-triplet builder the
                // realized design uses (`matern_operator_penalty_triplet_from_
                // metadata`). The Matérn design ALWAYS uses this {mass, tension,
                // stiffness} triplet (see the Matérn penalty selection in
                // term_specs.rs; #1074 confirmed by MSI measurement that the RKHS
                // kernel penalty does not improve recovery and regresses the
                // high-frequency guard), so re-keying via the kernel path would
                // produce a 1-block surface against a 3-block frozen design — the
                // topology desync #1270 hard-errored on. Sharing the builder
                // makes the block count ψ-stable by construction.
                let filtered = matern_operator_penalty_triplet_at_length_scale(
                    centers.view(),
                    periodic.as_deref(),
                    identifiability_transform.as_ref(),
                    *nu,
                    *include_intercept,
                    aniso_for_penalty,
                    effective_ls,
                )
                .map_err(|e| e.to_string())?;
                rekey_energy_factors = filtered
                    .active
                    .iter()
                    .map(|penalty| penalty.info.energy_factor.clone())
                    .collect();
                let locals = filtered
                    .active
                    .iter()
                    .map(|penalty| penalty.matrix.clone())
                    .collect();
                let nullspace_dims = filtered
                    .active
                    .iter()
                    .map(|penalty| penalty.nullity)
                    .collect();
                (locals, nullspace_dims)
            }
            BasisMetadata::ThinPlate {
                centers,
                identifiability_transform,
                radial_reparam,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "thin-plate n-free penalty re-key requires a finite length-scale".to_string()
                })?;
                let double_penalty = match &termspec.basis {
                    SmoothBasisSpec::ThinPlate { spec, .. } => spec.double_penalty,
                    _ => false,
                };
                gam_terms::basis::thin_plate_penalties_at_length_scale(
                    centers.view(),
                    identifiability_transform.as_ref(),
                    radial_reparam.as_ref(),
                    ls,
                    double_penalty,
                    &mut self.basisworkspace,
                )
                .map_err(|e| e.to_string())?
            }
            other => {
                return Err(format!(
                    "n-free penalty re-key unsupported for basis metadata {:?}",
                    std::mem::discriminant(other)
                ));
            }
        };
        // The frozen collection penalties for THIS term are the templates whose
        // ψ-invariant structure (col_range / prior_mean / structure_hint / op)
        // we keep, swapping only the numeric `local`. For a single-spatial-term
        // collection the term owns the whole penalty list.
        let templates = &self.design.penalties;
        if templates.len() != locals.len() {
            return Err(format!(
                "n-free penalty re-key produced {} blocks but the frozen design carries {} \
                 — penalty topology is not ψ-stable",
                locals.len(),
                templates.len()
            ));
        }
        let specs: Vec<gam_solve::estimate::PenaltySpec> = templates
            .iter()
            .zip(locals.into_iter())
            .enumerate()
            .map(|(block, (tmpl, local))| gam_solve::estimate::PenaltySpec::Block {
                local,
                col_range: tmpl.col_range.clone(),
                prior_mean: tmpl.prior_mean.clone(),
                structure_hint: match rekey_energy_factors.get(block).cloned().flatten() {
                    Some(factor) => {
                        Some(gam_terms::smooth::PenaltyStructureHint::EnergyFactor(factor))
                    }
                    None => match &tmpl.structure_hint {
                        Some(gam_terms::smooth::PenaltyStructureHint::EnergyFactor(_)) => None,
                        other => other.clone(),
                    },
                },
                op: tmpl.op.clone(),
            })
            .collect();
        let canonical = gam_terms::construction::canonicalize_penalty_specs_at_frozen_ranks(
            &specs,
            &nullspace_dims,
            frozen_penalty_ranks,
            p_total,
            "nfree-psi-penalty",
        )
        .map_err(|e| e.to_string())?;
        log::info!(
            "[STAGE] n-free S(psi) rebuild: {} penalty block(s), p={p_total}, psi_dim={}, elapsed={:.3}s",
            canonical.0.len(),
            psi.len(),
            rebuild_started.elapsed().as_secs_f64(),
        );
        Ok(canonical)
    }

    fn canonical_penalty_derivatives_at_psi(
        &mut self,
        spatial_terms: &[usize],
        psi: &[f64],
    ) -> Result<(Range<usize>, usize, Vec<Array2<f64>>), String> {
        if spatial_terms.len() != 1 {
            return Err(format!(
                "n-free penalty derivative re-key requires exactly one spatial term, found {}",
                spatial_terms.len()
            ));
        }
        let term_idx = spatial_terms[0];
        let (ls_opt, aniso_from_psi) = spatial_term_psi_to_length_scale_and_aniso(psi);
        let termspec = self.spec.smooth_terms.get(term_idx).ok_or_else(|| {
            format!("spatial term {term_idx} out of range for n-free penalty derivative")
        })?;
        let term = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("realized smooth term {term_idx} out of range"))?;
        let p_total = self.design.design.ncols();
        let smooth_start = p_total.saturating_sub(self.design.smooth.total_smooth_cols());
        let global_range =
            (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);

        let locals = match &term.metadata {
            BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                power,
                nullspace_order,
                aniso_log_scales,
                input_scale,
                radial_reparam,
                ..
            } => {
                let mut spec = match &termspec.basis {
                    SmoothBasisSpec::Duchon { spec, .. } => spec.clone(),
                    _ => {
                        return Err(
                            "Duchon n-free penalty derivative requires a Duchon term spec"
                                .to_string(),
                        );
                    }
                };
                let effective_ls = ls_opt.map(|length| {
                    input_scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length))
                        .standardized_value()
                });
                spec.length_scale = effective_ls;
                spec.power = *power;
                spec.nullspace_order = *nullspace_order;
                spec.aniso_log_scales = aniso_log_scales.clone();
                // #1355: replay the frozen data-metric reparam so the n-free
                // penalty ψ-derivative matches the rotated forward penalty.
                spec.radial_reparam = radial_reparam.clone();
                if spec.length_scale.is_none() {
                    return Err(
                        "Duchon n-free penalty derivative requires a hybrid length-scale"
                            .to_string(),
                    );
                }
                let collocation = operator_collocation_points
                    .as_ref()
                    .map(|points| points.view())
                    .unwrap_or_else(|| centers.view());
                let (_native_sources, mut first, _native_second) =
                    gam_terms::basis::build_duchon_native_penalty_psi_derivatives(
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                let (_operator_sources, operator_first, _operator_second) =
                    gam_terms::basis::build_duchon_operator_penalty_psi_derivatives(
                        collocation,
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                first.extend(operator_first);
                first
            }
            BasisMetadata::Matern {
                centers,
                periodic,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                input_scale,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "Matérn n-free penalty derivative requires a finite length-scale".to_string()
                })?;
                let effective_ls = input_scale
                    .to_standardized_units(gam_terms::OriginalUnits::new(ls))
                    .standardized_value();
                let penalty_centers = gam_terms::basis::expand_periodic_centers(
                    &centers.to_owned(),
                    periodic.as_deref(),
                )
                .map_err(|e| e.to_string())?;
                let aniso_for_penalty = aniso_from_psi.as_deref().or(aniso_log_scales.as_deref());
                let (first, _second) =
                    gam_terms::basis::build_matern_operator_penalty_psi_derivatives(
                        penalty_centers.view(),
                        effective_ls,
                        *nu,
                        *include_intercept,
                        identifiability_transform.as_ref(),
                        aniso_for_penalty,
                    )
                    .map_err(|e| e.to_string())?;
                first
            }
            BasisMetadata::ThinPlate {
                centers,
                identifiability_transform,
                radial_reparam,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "thin-plate n-free penalty derivative requires a finite length-scale"
                        .to_string()
                })?;
                let mut spec = match &termspec.basis {
                    SmoothBasisSpec::ThinPlate { spec, .. } => spec.clone(),
                    _ => {
                        return Err(
                            "thin-plate n-free penalty derivative requires a ThinPlate term spec"
                                .to_string(),
                        );
                    }
                };
                spec.length_scale = ls;
                if spec.radial_reparam.is_none() {
                    spec.radial_reparam = radial_reparam.clone();
                }
                let (primary, _primary_second, nullspace, _nullspace_second) =
                    gam_terms::basis::build_thin_plate_penalty_psi_derivativeswithworkspace(
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                if self.design.penalties.len() > 1 {
                    vec![primary, nullspace]
                } else {
                    vec![primary]
                }
            }
            other => {
                return Err(format!(
                    "n-free penalty derivative re-key unsupported for basis metadata {:?}",
                    std::mem::discriminant(other)
                ));
            }
        };
        if locals.len() != self.design.penalties.len() {
            return Err(format!(
                "n-free penalty derivative re-key produced {} blocks but the frozen design carries {} \
                 — penalty topology is not ψ-stable",
                locals.len(),
                self.design.penalties.len()
            ));
        }
        Ok((global_range, p_total, locals))
    }

    /// Realize a new ψ on every named term.
    ///
    /// Typed, not stringly (gam#2760): a trial ψ at which the collection's model
    /// cannot be realized is a DOMAIN WALL the outer search retreats from, while
    /// a rebuild that is not the basis the collection gauged is a defect that
    /// must abort. `EstimationError` is the type that already carries that
    /// distinction (`TrialPointRefused` vs the rest), and flattening it to a
    /// `String` here is what erased it — every realization failure reached
    /// `eval_cost` as `InvalidInput`, i.e. fatal.
    fn apply_log_kappa(
        &mut self,
        log_kappa: &SpatialLogKappaCoords,
        term_indices: &[usize],
    ) -> Result<(), EstimationError> {
        if term_indices.len() != log_kappa.dims_per_term().len() {
            return Err(EstimationError::InvalidInput(
                SmoothError::dimension_mismatch(format!(
                    "incremental realizer log-kappa term mismatch: term_indices={}, dims_per_term={}",
                    term_indices.len(),
                    log_kappa.dims_per_term().len()
                ))
                .to_string(),
            ));
        }

        let mut any_changed = false;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            any_changed |= self.apply_log_kappa_to_term(term_idx, log_kappa.term_slice(slot))?;
        }

        if any_changed {
            self.refresh_full_design_operator()
                .map_err(EstimationError::InvalidInput)?;
            rebuild_smooth_auxiliary_state(
                &mut self.design.smooth,
                &self.dropped_penaltyinfo_by_term,
            )
            .map_err(EstimationError::InvalidInput)?;
            rebuild_term_collection_auxiliary_state(&self.spec, &mut self.design)
                .map_err(EstimationError::InvalidInput)?;
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        Ok(())
    }

    fn apply_log_kappa_to_term(
        &mut self,
        term_idx: usize,
        psi: &[f64],
    ) -> Result<bool, EstimationError> {
        if !spatial_term_supports_hyper_optimization(&self.spec, term_idx) {
            return Err(EstimationError::InvalidInput(
                SmoothError::invalid_config(format!(
                    "incremental realizer term {term_idx} does not expose spatial hyperparameters"
                ))
                .to_string(),
            ));
        }
        // Measure-jet ψ slots are dial coordinates, not log-κ (dial docs:
        // the MEASURE_JET_PSI_* bounds block); route through the dial setter
        // so the κ-translation below never misreads them as log-scales.
        let measure_jet_term = measure_jet_term_spec(&self.spec, term_idx).is_some();
        // Constant-curvature ψ is the raw signed curvature κ, NOT a log-scale;
        // route through the κ setter so `spatial_term_psi_to_length_scale_and_aniso`
        // never misreads it (and never hits the "no length scale" rejection).
        let constant_curvature_term = constant_curvature_term_spec(&self.spec, term_idx).is_some();
        let mut next_length_scale = None;
        let mut next_aniso: Option<Vec<f64>> = None;
        if measure_jet_term {
            if !set_measure_jet_psi_dials(&mut self.spec, term_idx, psi)
                ?
            {
                return Ok(false);
            }
        } else if constant_curvature_term {
            if !set_constant_curvature_kappa(&mut self.spec, term_idx, psi)
                ?
            {
                return Ok(false);
            }
        } else {
            let current_length_scale = get_spatial_length_scale(&self.spec, term_idx);
            let current_aniso = get_spatial_aniso_log_scales(&self.spec, term_idx);
            let (ls, eta) = spatial_term_psi_to_length_scale_and_aniso(psi);
            next_length_scale = ls;
            next_aniso = eta;
            let same_length = spatial_length_scale_matches(current_length_scale, next_length_scale);
            let same_aniso = spatial_aniso_matches(current_aniso.as_deref(), next_aniso.as_deref());
            if same_length && same_aniso {
                return Ok(false);
            }
            if let Some(length_scale) = next_length_scale {
                set_spatial_length_scale(&mut self.spec, term_idx, length_scale)
                    ?;
            }
            if let Some(eta) = next_aniso.clone() {
                set_spatial_aniso_log_scales(&mut self.spec, term_idx, eta)
                    ?;
            }
        }

        // Pick the spec to drive the rebuild. If the per-term geometry cache
        // is populated, it carries already-resolved centers
        // (`CenterStrategy::UserProvided`) and frozen `input_scale`; reusing
        // it short-circuits `select_centers_by_strategy` (KMeans /
        // FarthestPoint / EqualMass cluster searches) and
        // isotropic scale estimation over n rows in
        // the family builders. Centers in the cached spec live in
        // standardized coordinates (matching the cached `input_scale`), so
        // the same standardization + kernel path runs without recomputation
        // of the geometry.
        let geometry_slot = self
            .spatial_realization_geometry
            .get(term_idx)
            .ok_or_else(|| EstimationError::InvalidInput(format!("incremental realizer geometry slot {term_idx} out of range")))?;
        let geometry_cached = geometry_slot.is_some();
        let mut build_spec = match geometry_slot {
            Some(cached) => cached.clone(),
            None => self
                .spec
                .smooth_terms
                .get(term_idx)
                .ok_or_else(|| EstimationError::InvalidInput(format!("incremental realizer smooth term {term_idx} out of range")))?
                .clone(),
        };
        if measure_jet_term {
            // The cached build spec carries the frozen geometry (UserProvided
            // barycenter nodes, frozen quadrature + transform); only the
            // dials move per trial.
            set_single_term_measure_jet_psi_dials(&mut build_spec, psi)
                ?;
        } else if constant_curvature_term {
            // The cached build spec carries the κ-fixed geometry (UserProvided
            // centers, frozen ℓ and constraint transform); only κ moves per
            // trial, written through the raw-κ setter to match the collection
            // write-back above.
            set_single_term_constant_curvature_kappa(&mut build_spec, psi)
                ?;
        } else {
            if let Some(length_scale) = next_length_scale {
                set_single_term_spatial_length_scale(&mut build_spec, length_scale)
                    ?;
            }
            if let Some(eta) = next_aniso {
                set_single_term_spatial_aniso_log_scales(&mut build_spec, eta)
                    ?;
            }
        }

        let termname = build_spec.name.clone();
        let t_build = std::time::Instant::now();
        let local = build_single_local_smooth_term(
            self.data,
            &build_spec,
            &mut self.basisworkspace,
        )
        .map_err(|e| {
            // A penalty Gram with an eigenvalue below its own rounding band is not
            // PSD at this psi, so the model does not exist at the trial: the search
            // retreats instead of aborting the fit.
            if matches!(e, gam_terms::basis::BasisError::IndefinitePenalty { .. }) {
                EstimationError::TrialPointRefused {
                    reason: format!("smooth term '{termname}' has no PSD penalty at this psi: {e}"),
                }
            } else {
                EstimationError::InvalidInput(format!(
                    "failed to rebuild smooth term '{termname}' during incremental κ realization: {e}"
                ))
            }
        })?;

        // Populate the geometry cache from the realized metadata on first use.
        // Family auto-promotion (ThinPlate -> Duchon) is detected as a basis /
        // metadata mismatch in `freeze_geometry_from_metadata` and leaves the
        // cache empty so the next call re-tries with the (now stable) family.
        if self.spatial_realization_geometry[term_idx].is_none()
            && let Some(frozen) = freeze_geometry_from_metadata(&build_spec, &local.metadata)
        {
            // Mirror the frozen identifiability (pinned `Z` + double-penalty
            // nullspace-shrinkage decision, #787/#860/#1122) back onto the
            // collection spec the analytic ψ-gradient reads
            // (`try_build_spatial_log_kappa_hyper_dirs(self.spec(), …)`). The
            // value rebuild consumes the cached `build_spec`, so without this
            // copy the gradient would keep re-running the κ-DEPENDENT spectral
            // test on the un-frozen collection spec while the value uses the
            // frozen decision — re-introducing the very objective↔gradient
            // desync the freeze removes. Pinning both to the same frozen
            // transform keeps the per-trial value and its analytic gradient on
            // one fixed `Z` and one fixed null dimension `r`.
            if let (
                SmoothBasisSpec::Matern {
                    spec: frozen_spec, ..
                },
                Some(SmoothBasisSpec::Matern {
                    spec: live_spec, ..
                }),
            ) = (
                &frozen.basis,
                self.spec
                    .smooth_terms
                    .get_mut(term_idx)
                    .map(|t| &mut t.basis),
            ) {
                live_spec.identifiability = frozen_spec.identifiability.clone();
                live_spec.center_strategy = frozen_spec.center_strategy.clone();
            }
            self.spatial_realization_geometry[term_idx] = Some(frozen);
        }

        // What this trial was rebuilt FROM, so a shape refusal downstream names
        // the trial rather than only its arithmetic (gam#2760).
        let trial_report = format!(
            "psi={psi:?}, length_scale={next_length_scale:?}, geometry_cached={geometry_cached}, \
             frozen_radial_chart={:?}, realized_radial_chart={:?}, local_cols={}",
            spatial_frozen_radial_chart_shape(&build_spec),
            spatial_realized_radial_chart_shape(&local.metadata),
            local.design.ncols(),
        );
        // The n×k realization is the trial's dominant cost and had no timer of
        // its own: the only per-trial number reported was the splice's, under a
        // label that named the rebuild (measured on the 6-D isotropic Duchon
        // fit at n=50 000, k=500: 16.6 s per κ trial, of which the splice the
        // old line reported was 1.7 s).
        log::info!(
            "[STAGE] smooth term realization (term {term_idx}, '{termname}', local_cols={}): {:.3}s",
            local.design.ncols(),
            t_build.elapsed().as_secs_f64(),
        );
        let realization = wrap_local_build_as_realization(local, &build_spec)
            .map_err(EstimationError::InvalidInput)?;
        self.replace_term_realization(term_idx, realization, &trial_report)?;
        Ok(true)
    }

    fn replace_term_realization(
        &mut self,
        term_idx: usize,
        realization: SingleSmoothTermRealization,
        trial_report: &str,
    ) -> Result<(), EstimationError> {
        let t_replace = std::time::Instant::now();
        let SingleSmoothTermRealization { design_local, term } = realization;
        let SmoothTerm {
            name,
            active_penalties,
            dropped_penalties,
            metadata,
            lower_bounds_local,
            linear_constraints_local,
            joint_null_rotation,
            ..
        } = term;
        // THE GAUGE IS THE COLLECTION'S (#2747). This rebuild is TERM-LOCAL, so
        // it cannot see the constraint block `[1 | owned linear axes | owner
        // smooths]` the collection made this term orthogonal to — the same
        // blindness the penalty-topology note below records for #2750, on the
        // design instead of on the penalty set.
        //
        // Before this, the splice wrote a term-local design and chart into the
        // slot while leaving the collection's `R` behind, and `R` is a function
        // of the design, hence of the ψ this realizer exists to move. The fit
        // then shipped `X(ψ̂)·Z − C·R(ψ₀)`: measured at `‖XᵀC‖/(‖X‖‖C‖) =
        // 4.15e-1` on `y ~ x1 + matern(x1, x2)` against the `1e-8` bar the
        // global step asserts whenever it applies a transform, with `2.39e-14`
        // for the same spec and rows when the pair is derived rather than
        // replayed. The κ search was therefore also minimizing a criterion for
        // a model the fit did not ship.
        //
        // `C`, the arm, and `T0` are psi-INDEPENDENT and travel on the term.
        // Placement recomputes only the left-projection correction `R(psi)`,
        // through the same entry point the collection build itself uses. This
        // makes the value replay identical to the analytic fixed-chart jet and
        // prevents penalty normalization from acquiring arbitrary RRQR motion.
        let collection_gauge = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .and_then(|target| target.collection_gauge.clone());
        // Everything the width check below decides on, captured BEFORE the gauge
        // consumes the local build (gam#2760). A refusal that reports only the
        // two widths cannot say which of the two halves moved — the local basis
        // dimension or the gauge's deletion — and those have different causes and
        // different repairs.
        let pre_gauge_cols = design_local.ncols();
        let gauge_report = match collection_gauge.as_ref() {
            Some(gauge) => format!(
                "arm={:?}, constraint_block={}x{}, owner_terms={:?}, local_columns={}",
                gauge.arm,
                gauge.constraint_block.nrows(),
                gauge.constraint_block.ncols(),
                gauge.owner_terms,
                gauge.local_columns,
            ),
            None => "none".to_string(),
        };
        let had_collection_gauge = collection_gauge.is_some();
        let (
            design_local,
            metadata,
            active_penalties,
            dropped_penalties,
            linear_constraints_local,
            joint_null_rotation,
            regauged_residualization,
        ) = match collection_gauge {
            Some(gauge) => {
                let placed = gam_terms::smooth::place_term_in_collection_gauge(
                    &gauge,
                    gam_terms::smooth::LocalTermRealization {
                        design: design_local,
                        metadata: &metadata,
                        active_penalties: &active_penalties,
                        dropped_penalties,
                        linear_constraints_local: linear_constraints_local.as_ref(),
                        joint_null_rotation: joint_null_rotation.as_ref(),
                        duchon_operator_penalties: self
                            .spec
                            .smooth_terms
                            .get(term_idx)
                            .and_then(gam_terms::smooth::duchon_operator_penalty_request),
                        termname: &name,
                    },
                )
                .map_err(|e| collection_gauge_placement_error(&name, trial_report, e))?;
                (
                    placed.design,
                    placed.metadata,
                    placed.active_penalties,
                    placed.dropped_penalties,
                    placed.linear_constraints_local,
                    // Folded into `metadata` above, exactly as a collection-built
                    // term reports it.
                    None,
                    Some(placed.parametric_residualization),
                )
            }
            None => (
                design_local,
                metadata,
                active_penalties,
                dropped_penalties,
                linear_constraints_local,
                joint_null_rotation,
                None,
            ),
        };
        // The gauge can add drops (a penalty that becomes vacuous under the
        // congruence), so the per-term dropped-block report is restated from the
        // post-gauge set rather than from the local build's.
        let dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo> = dropped_penalties
            .iter()
            .map(|info| DroppedPenaltyBlockInfo {
                termname: Some(name.clone()),
                penalty: info.clone(),
            })
            .collect();
        let coeff_range = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| EstimationError::InvalidInput(format!("incremental realizer smooth term {term_idx} out of range")))?
            .coeff_range
            .clone();
        if design_local.ncols() != coeff_range.len() {
            // A collection-gauged term has a fixed coefficient transform. Its
            // local-input width is checked inside placement and its output width
            // is `T0.ncols()`, so reaching this branch with a gauge is an
            // invariant violation, never a moving-rank domain wall. An ungauged
            // radial basis may still lose a numerical chart direction at this
            // psi; that remains a recoverable trial refusal.
            let reason = format!(
                "incremental realizer width mismatch for term {term_idx} ('{name}'): rebuilt_cols={}, \
                 cached_cols={}; the local rebuild produced {pre_gauge_cols} column(s) before the \
                 collection gauge ({gauge_report}) and {} after it. Trial: {trial_report}",
                design_local.ncols(),
                coeff_range.len(),
                design_local.ncols(),
            );
            if had_collection_gauge {
                return Err(EstimationError::InvalidInput(format!(
                    "{reason}. A fixed collection coefficient chart cannot change width; the \
                     replay is not the collection-gauged model (gam#2760)"
                )));
            }
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "{reason}. This ungauged realized basis loses a numerical chart direction at \
                     this psi, so the model does not exist at the trial (gam#2760)"
                ),
            });
        }
        if design_local.nrows() != self.design.design.nrows() {
            return Err(EstimationError::InvalidInput(SmoothError::dimension_mismatch(format!(
                "incremental realizer row mismatch for term {}: rebuilt_rows={}, design_rows={}",
                term_idx,
                design_local.nrows(),
                self.design.design.nrows()
            )).to_string()));
        }

        let smooth_penalty_range = self
            .smooth_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                EstimationError::InvalidInput(format!("incremental realizer missing smooth penalty range for term {term_idx}"))
            })?
            .clone();
        let full_penalty_range = self
            .full_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                EstimationError::InvalidInput(format!("incremental realizer missing full penalty range for term {term_idx}"))
            })?
            .clone();
        // TOPOLOGY IS FROZEN WITH THE CHART (#2750). A ψ trial may move penalty
        // VALUES; it may not move the penalty SET. The cached topology is the
        // COLLECTION's — it was decided after the global parametric
        // orthogonalization — while this rebuild is TERM-LOCAL and cannot see
        // that gauge, so a candidate the collection dropped as vacuous can
        // reappear here. Measured on `measure_jet_formula_fit_robustness_sweep`
        // seed 4: the cold collection drops `DoublePenaltyNullspace` with
        // reason `ZeroMatrix` (the parametric block had already absorbed the
        // affine head) and a trial rebuild at a different representer range
        // emits it again, aborting the outer search mid-flight.
        //
        // Align by `(original_index, source)` (`align_rebuilt_penalties`): keep the
        // candidates the cached topology kept, in cached order, and record the
        // rest as dropped. Anything the cache holds that the rebuild did NOT
        // produce, or produced as a different kind of block, is a real
        // inconsistency — a ρ coordinate with no matrix behind it, or a matrix
        // of another penalty family — and still refuses, naming both topologies
        // (#2953).
        let cached_penalties: Vec<(usize, gam_terms::basis::PenaltySource)> = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .map(|term| {
                term.active_penalties
                    .iter()
                    .map(|active| (active.info.original_index, active.info.source.clone()))
                    .collect()
            })
            .unwrap_or_default();
        // A rebuild with the same NUMBER of blocks is aligned the same way. Pairing
        // by position alone let a count-1 Primary cache take an OperatorMass rebuild
        // without a refusal. An identical topology aligns to the identity.
        let (active_penalties, dropped_penalties) = if cached_penalties.len()
            == smooth_penalty_range.len()
        {
            let rebuilt_active: Vec<(usize, gam_terms::basis::PenaltySource)> = active_penalties
                .iter()
                .map(|active| (active.info.original_index, active.info.source.clone()))
                .collect();
            let rebuilt_dropped: Vec<(
                usize,
                gam_terms::basis::PenaltySource,
                gam_terms::basis::PenaltyDropReason,
            )> = dropped_penalties
                .iter()
                .map(|info| (info.original_index, info.source.clone(), info.reason.clone()))
                .collect();
            match align_rebuilt_penalties(&cached_penalties, &rebuilt_active, &rebuilt_dropped) {
                PenaltyAlignment::Aligned { slots } => {
                    let mut rebuilt: Vec<Option<gam_terms::basis::ActivePenalty>> =
                        active_penalties.into_iter().map(Some).collect();
                    let kept = slots
                        .iter()
                        .map(|&slot| {
                            rebuilt[slot]
                                .take()
                                .expect("align_rebuilt_penalties hands out each rebuilt slot once")
                        })
                        .collect::<Vec<_>>();
                    let mut dropped = dropped_penalties;
                    dropped.extend(rebuilt.into_iter().flatten().map(|active| {
                        gam_terms::basis::DroppedPenaltyInfo {
                            source: active.info.source.clone(),
                            original_index: active.info.original_index,
                            reason: gam_terms::basis::PenaltyDropReason::ZeroMatrix,
                            normalization_scale: active.info.normalization_scale,
                        }
                    }));
                    (kept, dropped)
                }
                // A cached block the rebuild DROPPED at this psi is a trial whose
                // rho coordinate has no matrix behind it, not a broken invariant.
                // The Matérn collocation Grams of odd derivative order are exactly
                // zero once every off-diagonal kernel value underflows, which a
                // length scale far below the center spacing produces. MSI job
                // 602008 (`y ~ matern(x, periodic=true, period=2π)`, n = 400)
                // reached this branch at an ARC trial with psi = 9.43, where the
                // rebuild kept originals [0, 2] of the 4 cached, and the InvalidInput
                // aborted the whole fit. The model does not exist at that trial, so
                // it is refused and the search shortens its step or rejects the seed.
                PenaltyAlignment::DroppedAtTrial {
                    original_index,
                    source,
                    reason,
                } => {
                    return Err(EstimationError::TrialPointRefused {
                        reason: format!(
                            "incremental realizer: cached penalty {original_index} ({source:?}) of \
                             term '{name}' was dropped as {reason:?} at this psi and the rebuild \
                             produced {rebuilt_active:?}, so its rho coordinate has no matrix at the \
                             trial. Trial: {trial_report}"
                        ),
                    });
                }
                PenaltyAlignment::Inconsistent { detail } => {
                    return Err(EstimationError::InvalidInput(SmoothError::dimension_mismatch(format!(
                        "incremental realizer penalty topology for term '{name}' does not match the \
                         cached topology: {detail}. Trial: {trial_report}"
                    )).to_string()));
                }
            }
        } else {
            (active_penalties, dropped_penalties)
        };
        if active_penalties.len() != smooth_penalty_range.len() {
            return Err(EstimationError::InvalidInput(SmoothError::dimension_mismatch(format!(
                "incremental realizer topology changed for term '{}': active_penalties={}, cached_penalties={}",
                name,
                active_penalties.len(),
                smooth_penalty_range.len()
            )).to_string()));
        }

        self.design.smooth.term_designs[term_idx] = design_local;

        for (offset, active_penalty) in active_penalties.iter().enumerate() {
            let smooth_penalty_idx = smooth_penalty_range.start + offset;
            let full_penalty_idx = full_penalty_range.start + offset;
            let penalty_local = &active_penalty.matrix;

            if penalty_local.nrows() != coeff_range.len()
                || penalty_local.ncols() != coeff_range.len()
            {
                return Err(EstimationError::InvalidInput(
                    SmoothError::dimension_mismatch(format!(
                        "incremental realizer penalty shape mismatch for term '{}' penalty {}: \
                         penalty is {}x{} but coeff_range has {} columns",
                        name,
                        offset,
                        penalty_local.nrows(),
                        penalty_local.ncols(),
                        coeff_range.len()
                    ))
                    .to_string(),
                ));
            }

            let smooth_penalty = self
                .design
                .smooth
                .penalties
                .get_mut(smooth_penalty_idx)
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "incremental realizer smooth penalty {} out of range for term {}",
                        smooth_penalty_idx, term_idx
                    ))
                })?;
            // With per-term block-local penalties, col_range already targets
            // this specific term, so .local is p_k × p_k.
            smooth_penalty.local.assign(penalty_local);
            smooth_penalty.op = active_penalty.op.clone();

            let full_bp = self
                .design
                .penalties
                .get_mut(full_penalty_idx)
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "incremental realizer full penalty {} out of range for term {}",
                        full_penalty_idx, term_idx
                    ))
                })?;
            // With per-term block-local penalties, col_range already targets
            // this specific term, so .local is p_k × p_k.
            full_bp.local.assign(penalty_local);
            full_bp.op = active_penalty.op.clone();

            self.design.smooth.nullspace_dims[smooth_penalty_idx] = active_penalty.nullity;
            self.design.nullspace_dims[full_penalty_idx] = active_penalty.nullity;

            self.design.smooth.penaltyinfo[smooth_penalty_idx].global_index = smooth_penalty_idx;
            self.design.smooth.penaltyinfo[smooth_penalty_idx].termname = Some(name.clone());
            self.design.smooth.penaltyinfo[smooth_penalty_idx].penalty =
                active_penalty.info.clone();

            self.design.penaltyinfo[full_penalty_idx].global_index = full_penalty_idx;
            self.design.penaltyinfo[full_penalty_idx].termname = Some(name.clone());
            self.design.penaltyinfo[full_penalty_idx].penalty = active_penalty.info.clone();
        }

        let target_term = self.design.smooth.terms.get_mut(term_idx).ok_or_else(|| {
            EstimationError::InvalidInput(format!("incremental realizer smooth term {term_idx} disappeared during replacement"))
        })?;
        target_term.active_penalties = active_penalties;
        target_term.dropped_penalties = dropped_penalties;
        target_term.metadata = metadata;
        target_term.lower_bounds_local = lower_bounds_local;
        target_term.linear_constraints_local = linear_constraints_local;
        target_term.joint_null_rotation = joint_null_rotation;
        // `R` moves with the design it was derived from, or the freeze ships a
        // pair that describes two different models (#2747). `None` on the
        // `Delete` arm is the right answer there, not a missing one.
        if let Some(chart) = regauged_residualization {
            target_term.parametric_residualization = chart;
        }
        self.dropped_penaltyinfo_by_term[term_idx] = dropped_penaltyinfo;
        log::info!(
            "[STAGE] collection-gauge placement + splice (term {}, '{}', cols={}): {:.3}s",
            term_idx,
            target_term.name,
            coeff_range.len(),
            t_replace.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    fn refresh_full_design_operator(&mut self) -> Result<(), String> {
        let mut blocks = Vec::<DesignBlock>::with_capacity(
            self.fixed_blocks.len() + self.design.smooth.term_designs.len(),
        );
        blocks.extend(self.fixed_blocks.iter().cloned());
        for term_design in &self.design.smooth.term_designs {
            blocks.push(DesignBlock::from(term_design));
        }
        self.design.design = assemble_term_collection_design_matrix(blocks)
            .map_err(|e| format!("failed to refresh term-collection design: {e}"))?;
        Ok(())
    }
}

fn build_term_collection_fixed_blocks(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<Vec<DesignBlock>, BasisError> {
    let mut blocks = Vec::<DesignBlock>::new();
    if !term_collection_has_anchored_bspline(spec) {
        blocks.push(DesignBlock::Intercept(data.nrows()));
    }

    if !spec.linear_terms.is_empty() {
        let mut linear_block = Array2::<f64>::zeros((data.nrows(), spec.linear_terms.len()));
        for (j, linear) in spec.linear_terms.iter().enumerate() {
            // Single shared realizer: numeric product gated by any
            // categorical-level indicators (factor-aware `:` interaction),
            // mirroring `build_term_collection_design_inner`.
            let column = linear
                .realized_design_column(data)
                .map_err(BasisError::InvalidInput)?;
            linear_block.column_mut(j).assign(&column);
        }
        blocks.push(DesignBlock::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(linear_block),
        ));
    }

    for term in &spec.random_effect_terms {
        let block = build_random_effect_block(data, term)?;
        let re_op = RandomEffectOperator::new(block.group_ids, block.num_groups);
        blocks.push(DesignBlock::RandomEffect(Arc::new(re_op)));
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// N-block spatial length-scale optimizer.
// ---------------------------------------------------------------------------

pub struct SpatialLengthScaleOptimizationResult<FitOut> {
    pub resolved_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    pub fit: FitOut,
    pub certified_outer: Option<gam_solve::rho_optimizer::CertifiedOuterResult>,
    pub timing: Option<SpatialLengthScaleOptimizationTiming>,
}

/// One exact outer-objective evaluation together with the owned coefficient
/// mode that produced it.
///
/// `mode` is deliberately generic and move-only.  The spatial driver never
/// interprets or clones it; it retains the carrier that belongs to the latest
/// successful evaluation and transfers that exact ownership into final fit
/// assembly after the outer certificate has been issued.
pub struct ExactJointEvaluation<M> {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub hessian: gam_problem::HessianValue,
    pub mode: M,
}

/// One exact fixed-point evaluation and the owned coefficient mode that
/// produced its value and update equations.
pub struct ExactJointEfsEvaluation<M> {
    pub evaluation: gam_problem::EfsEval,
    pub mode: M,
}

pub enum SpatialFitProvenance<'a, M> {
    NoOuterOptimization,
    Certified {
        outer: &'a gam_solve::rho_optimizer::CertifiedOuterResult,
        mode: M,
    },
}

/// Exact-joint hyper-parameter setup for N-block spatial length-scale optimization.
#[derive(Debug, Clone)]
pub struct ExactJointHyperSetup {
    rho0: Array1<f64>,
    /// The per-coordinate ρ domain its builder derived over every block that
    /// owns a ρ coordinate (#2812). It is the ρ half of the θ box, and no
    /// coordinate is searched without one (#2902 item 15).
    rho_lower: Array1<f64>,
    rho_upper: Array1<f64>,
    log_kappa0: SpatialLogKappaCoords,
    log_kappa_lower: SpatialLogKappaCoords,
    log_kappa_upper: SpatialLogKappaCoords,
    auxiliary0: Array1<f64>,
    auxiliary_lower: Array1<f64>,
    auxiliary_upper: Array1<f64>,
}

impl ExactJointHyperSetup {
    /// A ρ seed inside its derived domain. A finite seed is projected onto the
    /// domain, and a non-finite one takes the domain's midpoint, the geometric
    /// mean of the two edge strengths.
    fn project_rho_seed(
        rho0: Array1<f64>,
        lower: &Array1<f64>,
        upper: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::from_iter(rho0.iter().zip(lower.iter().zip(upper.iter())).map(
            |(&value, (&lo, &hi))| {
                if value.is_finite() {
                    value.clamp(lo, hi)
                } else {
                    0.5 * (lo + hi)
                }
            },
        ))
    }

    /// An auxiliary seed inside the box its caller derived for it.
    fn sanitize_auxiliary_seed(
        seed: Array1<f64>,
        lower: &Array1<f64>,
        upper: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::from_iter(seed.iter().enumerate().map(|(idx, &value)| {
            let lo = lower[idx];
            let hi = upper[idx];
            if value.is_finite() {
                value.clamp(lo, hi)
            } else {
                0.0_f64.clamp(lo, hi)
            }
        }))
    }

    /// The setup of a ρ seed on the domain its builder derived for it, and of
    /// the κ coordinates on their bounds. The seed is projected into the domain.
    pub(crate) fn new(
        rho0: Array1<f64>,
        rho_lower: Array1<f64>,
        rho_upper: Array1<f64>,
        log_kappa0: SpatialLogKappaCoords,
        log_kappa_lower: SpatialLogKappaCoords,
        log_kappa_upper: SpatialLogKappaCoords,
    ) -> Self {
        assert_eq!(rho_lower.len(), rho0.len(), "rho domain lower length mismatch");
        assert_eq!(rho_upper.len(), rho0.len(), "rho domain upper length mismatch");
        assert!(
            rho_lower.iter().zip(rho_upper.iter()).all(|(lo, hi)| lo <= hi),
            "rho domain must be ordered: lower={rho_lower:?} upper={rho_upper:?}"
        );
        let rho0 = Self::project_rho_seed(rho0, &rho_lower, &rho_upper);
        Self {
            rho0,
            rho_lower,
            rho_upper,
            log_kappa0,
            log_kappa_lower,
            log_kappa_upper,
            auxiliary0: Array1::zeros(0),
            auxiliary_lower: Array1::zeros(0),
            auxiliary_upper: Array1::zeros(0),
        }
    }

    pub(crate) fn with_auxiliary(
        mut self,
        auxiliary0: Array1<f64>,
        auxiliary_lower: Array1<f64>,
        auxiliary_upper: Array1<f64>,
    ) -> Self {
        assert_eq!(
            auxiliary0.len(),
            auxiliary_lower.len(),
            "auxiliary lower bound length mismatch"
        );
        assert_eq!(
            auxiliary0.len(),
            auxiliary_upper.len(),
            "auxiliary upper bound length mismatch"
        );
        self.auxiliary0 =
            Self::sanitize_auxiliary_seed(auxiliary0, &auxiliary_lower, &auxiliary_upper);
        self.auxiliary_lower = auxiliary_lower;
        self.auxiliary_upper = auxiliary_upper;
        self
    }

    pub(crate) fn rho_dim(&self) -> usize {
        self.rho0.len()
    }

    pub(crate) fn log_kappa_dim(&self) -> usize {
        self.log_kappa0.len()
    }

    pub(crate) fn auxiliary_dim(&self) -> usize {
        self.auxiliary0.len()
    }

    pub(crate) fn theta0(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho0);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa0.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary0);
        out
    }

    /// The θ box: the derived ρ domain, the κ bounds, then the auxiliary box.
    pub(crate) fn lower(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_lower);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa_lower.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary_lower);
        out
    }

    pub(crate) fn upper(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_upper);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa_upper.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary_upper);
        out
    }

    /// Per-term dimensionality layout for the psi block.
    pub(crate) fn log_kappa_dims_per_term(&self) -> Vec<usize> {
        self.log_kappa0.dims_per_term().to_vec()
    }
}

/// N-block design cache for exact-joint spatial length-scale optimization.
///
/// Each block owns a `FrozenTermCollectionIncrementalRealizer` and a list of
/// spatial term indices within that block's spec. The cache splits the
/// combined psi vector into per-block slices using precomputed offsets.
struct ExactJointDesignCache<'d> {
    realizers: Vec<FrozenTermCollectionIncrementalRealizer<'d>>,
    block_term_indices: Vec<Vec<usize>>,
    current_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(f64, Array1<f64>, gam_problem::HessianValue)>,
    rho_dim: usize,
    all_dims: Vec<usize>,
    log_kappa_dim: usize,
    block_term_counts: Vec<usize>,
}

impl<'d> ExactJointDesignCache<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)>,
        rho_dim: usize,
        all_dims: Vec<usize>,
    ) -> Result<Self, String> {
        let n_blocks = blocks.len();
        let mut realizers = Vec::with_capacity(n_blocks);
        let mut block_term_indices = Vec::with_capacity(n_blocks);
        let mut block_term_counts = Vec::with_capacity(n_blocks);

        for (spec, design, terms) in blocks {
            block_term_counts.push(terms.len());
            block_term_indices.push(terms);
            realizers.push(FrozenTermCollectionIncrementalRealizer::new(
                data, spec, design,
            )?);
        }

        Ok(Self {
            realizers,
            block_term_indices,
            current_theta: None,
            last_cost: None,
            last_eval: None,
            rho_dim,
            log_kappa_dim: all_dims.iter().sum(),
            all_dims,
            block_term_counts,
        })
    }

    /// Realize every block at `theta`. The realizers' typed errors pass through
    /// unchanged, so a trial one of them refuses (`TrialPointRefused`, e.g. a
    /// cached penalty the rebuild dropped at this ψ) stays a refusal the outer
    /// search can retreat from (#2953).
    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), EstimationError> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }

        let t_ensure = std::time::Instant::now();
        let kappa_theta_len = self.rho_dim + self.log_kappa_dim;
        if theta.len() < kappa_theta_len {
            return Err(EstimationError::InvalidInput(
                SmoothError::dimension_mismatch(format!(
                    "exact-joint theta length mismatch: got {}, expected at least {} (rho_dim={}, log_kappa_dim={})",
                    theta.len(),
                    kappa_theta_len,
                    self.rho_dim,
                    self.log_kappa_dim
                ))
                .to_string(),
            ));
        }
        let theta_kappa = theta.slice(s![..kappa_theta_len]).to_owned();
        let full_log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            &theta_kappa,
            self.rho_dim,
            self.all_dims.clone(),
        );

        // Split the full log_kappa into per-block sub-coords using split_at.
        // We split from the front iteratively: after extracting block 0..N-2,
        // the remainder is the last block.
        let n = self.realizers.len();
        let mut remaining = full_log_kappa;
        for block_idx in 0..n {
            let count = self.block_term_counts[block_idx];
            if block_idx < n - 1 {
                let (block_lk, rest) = remaining.split_at(count);
                self.realizers[block_idx]
                    .apply_log_kappa(&block_lk, &self.block_term_indices[block_idx])?;
                remaining = rest;
            } else {
                // Last block gets the remainder.
                self.realizers[block_idx]
                    .apply_log_kappa(&remaining, &self.block_term_indices[block_idx])?;
            }
        }

        log::info!(
            "[STAGE] ensure_theta (n-block, {} blocks, {} realizers): {:.3}s",
            n,
            self.realizers.len(),
            t_ensure.elapsed().as_secs_f64(),
        );
        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    impl_exact_joint_theta_memo!();

    /// Cache a cost-only result. Called after `ensure_theta(theta)` for
    /// literal-seed and line-search cost probes. We
    /// intentionally do not populate `last_eval` because no gradient was
    /// computed; the next outer evaluation at this θ will recompute
    /// (V, ∇V) via `evaluate_with_order` if the optimizer asks for it.
    fn store_cost_only(&mut self, theta: &Array1<f64>, cost: f64) {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_cost = Some(cost);
        }
    }

    /// Revoke objective values when the row measure changes while retaining
    /// the realized design at the current theta.
    fn invalidate_objective_memo(&mut self) {
        self.last_cost = None;
        self.last_eval = None;
    }

    fn specs(&self) -> Vec<&TermCollectionSpec> {
        self.realizers.iter().map(|r| r.spec()).collect()
    }

    fn designs(&self) -> Vec<&TermCollectionDesign> {
        self.realizers.iter().map(|r| r.design()).collect()
    }

    /// Combined monotonic design revision across all per-block realizers.
    ///
    /// Mirrors `SingleBlockExactJointDesignCache::design_revision` for the
    /// n-block exact-joint path. Each realizer's `design_revision` counter
    /// advances iff `apply_log_kappa` actually rebuilt that block's realized
    /// design / smooth penalties; the wrapping sum therefore changes iff
    /// *any* block rebuilt. Equal values across two calls imply no realizer
    /// has been rebuilt in between, which is the invariant the
    /// `ExternalJointHyperEvaluator` canonical-penalty fast path needs.
    fn design_revision(&self) -> u64 {
        self.realizers
            .iter()
            .fold(0u64, |acc, r| acc.wrapping_add(r.design_revision()))
    }
}

pub(crate) fn seed_risk_profile_for_likelihood_family(
    family: &LikelihoodSpec,
) -> gam_problem::SeedRiskProfile {
    match &family.response {
        ResponseFamily::Gaussian => gam_problem::SeedRiskProfile::Gaussian,
        ResponseFamily::RoystonParmar => gam_problem::SeedRiskProfile::Survival,
        ResponseFamily::Binomial
        | ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma => gam_problem::SeedRiskProfile::GeneralizedLinear,
    }
}

fn exact_joint_seed_config(
    risk_profile: gam_problem::SeedRiskProfile,
    auxiliary_dim: usize,
    initial_seed_only: bool,
) -> gam_problem::SeedConfig {
    let mut config = gam_problem::SeedConfig {
        risk_profile,
        num_auxiliary_trailing: auxiliary_dim,
        ..Default::default()
    };
    match risk_profile {
        gam_problem::SeedRiskProfile::Gaussian
        | gam_problem::SeedRiskProfile::GaussianLocationScale => {
            config.max_seeds = 4;
            config.seed_budget = 2;
        }
        gam_problem::SeedRiskProfile::GeneralizedLinear => {
            // Bernoulli marginal-slope Matérn fits use the exact-joint spatial
            // driver rather than the family-local BMS outer. Mirror BMS proper:
            // screen one principled heuristic seed deeply enough to reach the
            // KKT basin instead of spending minutes screening equivalent starts.
            config.max_seeds = 1;
            config.seed_budget = 1;
            config.screen_max_inner_iterations = 8;
        }
        gam_problem::SeedRiskProfile::Survival => {
            // Survival marginal-slope has an additional time/hazard block and
            // is the most sensitive Matérn startup regime. Keep more of the
            // coherent SPDE candidate manifold alive through truncation and
            // validate enough starts that one bad transient does not report
            // "no candidate seeds" before reaching a viable basin.
            config.max_seeds = 8;
            config.seed_budget = 4;
            config.screen_max_inner_iterations = 8;
        }
    }
    if initial_seed_only {
        // The isotropic Matérn path has already compared and fully profiled its
        // two geometry-derived range basins. Its winning [rho, psi] point is an
        // explicit certified initial point, so launching another heuristic seed
        // would repeat basin selection inside the local joint solve. A budget of
        // one gives that explicit initial point sole ownership of the run-plan
        // slot (run_plan inserts it at slot zero and skips seed screening).
        config.max_seeds = 1;
        config.seed_budget = 1;
        config.over_smoothing_probe_rho = None;
    }
    config
}

/// A term-local rebuild's penalty blocks against the realizer's cached topology
/// (#2750, #2953).
#[derive(Debug)]
enum PenaltyAlignment {
    /// `slots[i]` indexes the rebuilt active block kept for cached block `i`.
    Aligned { slots: Vec<usize> },
    /// The rebuild recorded a cached block as dropped at this psi.
    DroppedAtTrial {
        original_index: usize,
        source: gam_terms::basis::PenaltySource,
        reason: gam_terms::basis::PenaltyDropReason,
    },
    /// The rebuild's blocks are not the cached topology.
    Inconsistent { detail: String },
}

/// Align a term-local rebuild's penalty blocks to the cached topology by
/// `(original_index, source)`.
///
/// An `original_index` is a candidate's position in the build that produced it,
/// so two builds agree on a block only when they agree on its position AND on
/// what the block is. Matching the index alone pairs blocks of different penalty
/// families that both number from zero, such as a Primary block with an
/// OperatorMass one.
fn align_rebuilt_penalties(
    cached: &[(usize, gam_terms::basis::PenaltySource)],
    rebuilt_active: &[(usize, gam_terms::basis::PenaltySource)],
    rebuilt_dropped: &[(
        usize,
        gam_terms::basis::PenaltySource,
        gam_terms::basis::PenaltyDropReason,
    )],
) -> PenaltyAlignment {
    let topologies = || {
        format!(
            "cached active {cached:?}, rebuilt active {rebuilt_active:?}, rebuilt dropped \
             {rebuilt_dropped:?}"
        )
    };
    let mut taken = vec![false; rebuilt_active.len()];
    let mut slots = Vec::with_capacity(cached.len());
    for (original, source) in cached {
        let at_index = rebuilt_active
            .iter()
            .enumerate()
            .find(|(slot, (index, _))| !taken[*slot] && index == original);
        if let Some((slot, (_, rebuilt_source))) = at_index {
            if rebuilt_source != source {
                return PenaltyAlignment::Inconsistent {
                    detail: format!(
                        "cached penalty {original} is {source:?} but the rebuild's block {original} \
                         is {rebuilt_source:?} ({})",
                        topologies()
                    ),
                };
            }
            taken[slot] = true;
            slots.push(slot);
            continue;
        }
        return match rebuilt_dropped.iter().find(|(index, _, _)| index == original) {
            Some((_, dropped_source, reason)) if dropped_source == source => {
                PenaltyAlignment::DroppedAtTrial {
                    original_index: *original,
                    source: source.clone(),
                    reason: reason.clone(),
                }
            }
            Some((_, dropped_source, _)) => PenaltyAlignment::Inconsistent {
                detail: format!(
                    "cached penalty {original} is {source:?} but the rebuild dropped block {original} \
                     as {dropped_source:?} ({})",
                    topologies()
                ),
            },
            None => PenaltyAlignment::Inconsistent {
                detail: format!(
                    "cached penalty {original} ({source:?}) is neither an active nor a dropped block \
                     of the rebuild ({})",
                    topologies()
                ),
            },
        };
    }
    PenaltyAlignment::Aligned { slots }
}

#[cfg(test)]
mod penalty_alignment_2953_tests {
    use super::*;
    use gam_terms::basis::{PenaltyDropReason, PenaltySource};

    /// md:550's shape: three cached operator blocks, a rebuild of two, and nothing
    /// recorded as dropped. The refusal names every block on both sides.
    #[test]
    fn a_cached_block_the_rebuild_never_produced_names_both_topologies_2953() {
        let cached = [
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
            (2, PenaltySource::OperatorStiffness),
        ];
        let rebuilt = [
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
        ];
        let PenaltyAlignment::Inconsistent { detail } = align_rebuilt_penalties(&cached, &rebuilt, &[])
        else {
            panic!("a cached block missing from both rebuilt lists must be inconsistent");
        };
        assert!(
            detail.contains("cached penalty 2 (OperatorStiffness)"),
            "{detail}"
        );
        assert!(
            detail.contains("rebuilt active [(0, OperatorMass), (1, OperatorTension)]"),
            "{detail}"
        );
    }

    /// A rebuild on another penalty family numbers its blocks from zero too.
    /// Matching the index alone pairs Primary with OperatorMass; matching the
    /// source refuses and names both.
    #[test]
    fn a_family_switch_refuses_instead_of_pairing_primary_with_operator_mass_2953() {
        let cached = [
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
            (2, PenaltySource::OperatorStiffness),
        ];
        let rebuilt = [
            (0, PenaltySource::Primary),
            (1, PenaltySource::DoublePenaltyNullspace),
        ];
        let PenaltyAlignment::Inconsistent { detail } = align_rebuilt_penalties(&cached, &rebuilt, &[])
        else {
            panic!("a penalty-family switch must be inconsistent");
        };
        assert!(
            detail.contains("cached penalty 0 is OperatorMass but the rebuild's block 0 is Primary"),
            "{detail}"
        );
    }

    /// A cached block the rebuild recorded as dropped at this psi is a trial the
    /// search steps away from, not an inconsistency (MSI 602008).
    #[test]
    fn a_cached_block_recorded_as_dropped_is_a_trial_refusal_2953() {
        let cached = [
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
            (2, PenaltySource::OperatorStiffness),
            (3, PenaltySource::OperatorThirdOrder),
        ];
        let rebuilt = [
            (0, PenaltySource::OperatorMass),
            (2, PenaltySource::OperatorStiffness),
        ];
        let dropped = [
            (1, PenaltySource::OperatorTension, PenaltyDropReason::ZeroMatrix),
            (3, PenaltySource::OperatorThirdOrder, PenaltyDropReason::ZeroMatrix),
        ];
        assert!(matches!(
            align_rebuilt_penalties(&cached, &rebuilt, &dropped),
            PenaltyAlignment::DroppedAtTrial {
                original_index: 1,
                source: PenaltySource::OperatorTension,
                reason: PenaltyDropReason::ZeroMatrix,
            }
        ));
    }

    /// A rebuild that re-emits a block the cached collection dropped keeps the
    /// cached blocks in cached order (#2750). The caller records the extra block
    /// as dropped.
    #[test]
    fn a_rebuild_re_emitting_a_cached_drop_keeps_the_cached_blocks_2953() {
        let cached = [(0, PenaltySource::Primary)];
        let rebuilt = [
            (0, PenaltySource::Primary),
            (1, PenaltySource::DoublePenaltyNullspace),
        ];
        let PenaltyAlignment::Aligned { slots } = align_rebuilt_penalties(&cached, &rebuilt, &[]) else {
            panic!("the cached block is present with its own source, so the rebuild aligns");
        };
        assert_eq!(slots, vec![0]);
    }

    /// The same NUMBER of blocks is no license to pair by position. A cached
    /// double-penalty Matérn without an intercept is [Primary 0]. A rebuild on the
    /// operator branch at ν = 1/2 is [OperatorMass 0]. The source mismatch refuses.
    #[test]
    fn a_same_count_family_switch_refuses_2953() {
        let cached = [(0, PenaltySource::Primary)];
        let rebuilt = [(0, PenaltySource::OperatorMass)];
        let PenaltyAlignment::Inconsistent { detail } = align_rebuilt_penalties(&cached, &rebuilt, &[])
        else {
            panic!("a same-count penalty-family switch must be inconsistent");
        };
        assert!(
            detail.contains("cached penalty 0 is Primary but the rebuild's block 0 is OperatorMass"),
            "{detail}"
        );
    }

    /// An identical same-count topology aligns to the identity, so routing the
    /// equal-count path through the alignment changes nothing where the builds agree.
    #[test]
    fn an_identical_same_count_topology_aligns_to_the_identity_2953() {
        let blocks = [
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
            (2, PenaltySource::OperatorStiffness),
        ];
        let PenaltyAlignment::Aligned { slots } = align_rebuilt_penalties(&blocks, &blocks, &[]) else {
            panic!("an identical topology must align");
        };
        assert_eq!(slots, vec![0, 1, 2]);
    }
}

#[cfg(test)]
mod exact_joint_seed_config_tests {
    use super::*;

    #[test]
    fn exact_joint_marginal_slope_profiles_get_deeper_startup_validation() {
        let bms =
            exact_joint_seed_config(gam_problem::SeedRiskProfile::GeneralizedLinear, 2, false);
        assert_eq!(bms.max_seeds, 1);
        assert_eq!(bms.seed_budget, 1);
        assert_eq!(bms.screen_max_inner_iterations, 8);
        assert_eq!(bms.num_auxiliary_trailing, 2);

        let survival = exact_joint_seed_config(gam_problem::SeedRiskProfile::Survival, 3, false);
        assert_eq!(survival.max_seeds, 8);
        assert_eq!(survival.seed_budget, 4);
        assert_eq!(survival.screen_max_inner_iterations, 8);
        assert_eq!(survival.num_auxiliary_trailing, 3);
    }

    #[test]
    fn exact_joint_gaussian_keeps_tight_historical_multistart_budget() {
        let gaussian = exact_joint_seed_config(gam_problem::SeedRiskProfile::Gaussian, 1, false);
        assert_eq!(gaussian.max_seeds, 4);
        assert_eq!(gaussian.seed_budget, 2);
        assert_eq!(
            gaussian.screen_max_inner_iterations,
            gam_problem::SeedConfig::default().screen_max_inner_iterations
        );
        assert_eq!(gaussian.num_auxiliary_trailing, 1);
    }

    #[test]
    fn certified_matern_basin_owns_the_only_joint_start() {
        let gaussian = exact_joint_seed_config(gam_problem::SeedRiskProfile::Gaussian, 1, true);
        assert_eq!(gaussian.max_seeds, 1);
        assert_eq!(gaussian.seed_budget, 1);
        assert_eq!(gaussian.over_smoothing_probe_rho, None);
        assert_eq!(gaussian.num_auxiliary_trailing, 1);
    }
}

/// The property #2760 is about, asserted on [`joint_rho_resolvability_domain`]
/// directly: the domain the joint search is handed must contain the incumbent
/// it will be GRADED against strictly inside it, so no coordinate begins the
/// search as an active constraint the criterion wants to cross — and a domain
/// read off the term's own spectrum has no wall for an incumbent to sit on.
#[cfg(test)]
mod joint_rho_resolvability_domain_tests {
    use super::*;

    fn two_column_block(gamma: f64) -> (DesignMatrix, Vec<gam_terms::smooth::BlockwisePenalty>) {
        let c = gamma.sqrt();
        let design = DesignMatrix::from(ndarray::array![[c, 0.0], [0.0, c]]);
        let penalty =
            gam_terms::smooth::BlockwisePenalty::new(0..2, ndarray::array![[1.0, 0.0], [0.0, 1.0]]);
        (design, vec![penalty])
    }

    /// The domain is the resolvability interval of the block's own spectrum:
    /// `[ln(εγ), ln(γ/ε)]` for two directions of equal design-relative
    /// curvature γ.
    #[test]
    fn the_domain_is_the_blocks_resolvability_interval_2812() {
        for gamma in [1.0e-6_f64, 1.0, 3.0e4] {
            let (design, penalties) = two_column_block(gamma);
            let (lower, upper) = joint_rho_resolvability_domain(&design, &penalties, 1);
            let expected_lower = 0.5 * f64::EPSILON.ln() + gamma.ln();
            let expected_upper = gamma.ln() - 0.5 * f64::EPSILON.ln();
            assert!(
                (lower[0] - expected_lower).abs() < 1e-9
                    && (upper[0] - expected_upper).abs() < 1e-9,
                "γ={gamma}: domain [{}, {}] must be [{expected_lower}, {expected_upper}]",
                lower[0],
                upper[0]
            );
        }
    }

    /// The #2760 incumbents that used to sit on the `±12` wall are interior
    /// points of the derived domain of a unit-curvature term: a fit inside
    /// its own domain seeds the joint search inside this one. The one
    /// incumbent a fit produced at `−24.13` — under the first cut of this
    /// domain, whose edge sat at the effective degrees of freedom's
    /// resolution rather than the gradient's — lies past the edge the
    /// gradient can resolve, and the driver projects such a seed onto it.
    #[test]
    fn the_2760_incumbents_are_interior_2812() {
        let (design, penalties) = two_column_block(1.0);
        let (lower, upper) = joint_rho_resolvability_domain(&design, &penalties, 1);
        for seed in [-12.347_446_785_500_143, 11.9, 17.5] {
            assert!(
                lower[0] < seed && seed < upper[0],
                "incumbent {seed} must be strictly inside [{}, {}]",
                lower[0],
                upper[0]
            );
        }
        let past_the_edge = -24.126_016_487_917_27;
        assert!(
            past_the_edge < lower[0] && lower[0] < -18.0,
            "an incumbent at {past_the_edge} sits past the gradient-resolution floor {}",
            lower[0]
        );
    }

    /// A coordinate the design carries no penalty block for keeps the
    /// precision box around unit strength, and says so.
    #[test]
    fn a_coordinate_past_the_blocks_keeps_the_precision_box_2812() {
        let (design, penalties) = two_column_block(1.0);
        let (lower, upper) = joint_rho_resolvability_domain(&design, &penalties, 2);
        let (box_lo, box_hi) = gam_solve::estimate::rho_domain::precision_box();
        assert_eq!(lower[1], box_lo);
        assert_eq!(upper[1], box_hi);
        assert!(lower[0] <= lower[1] && upper[1] <= upper[0]);
    }
}

pub(crate) fn exact_joint_multistart_outer_problem(
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    auxiliary_dim: usize,
    n_params: usize,
    gradient: gam_problem::Derivative,
    hessian: gam_problem::DeclaredHessianForm,
    disable_fixed_point: bool,
    risk_profile: gam_problem::SeedRiskProfile,
    tolerance: f64,
    max_iter: usize,
    // BFGS step caps split by parameter type. `bfgs_step_cap` (rho-axis cap)
    // bounds first-trial moves on log-λ; documented natural step is ≈ 5.
    // `bfgs_step_cap_psi` bounds moves on the trailing `auxiliary_dim`
    // psi-axes (kappa / aniso-log-scales), where ≈ ln 2 keeps the kernel
    // scale from oscillating across orders of magnitude per iter. Using a
    // single uniform cap (the old API) starved rho on the survival-marg-slope
    // joint solver because the psi-calibrated value (`ln 2 ≈ 0.69`) was
    // applied to log-λ, where |d|≈5 is the natural quasi-Newton magnitude.
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    screening_cap: Option<Arc<AtomicUsize>>,
    // `Some((n_obs, p_cols))` calibrates the outer solver to the n-scaled
    // profiled REML/LAML criterion exactly as the primary REML outer
    // (`solver/estimate.rs`) does. The profiled criterion is a sum over the n
    // observations, so its magnitude is O(n) (|f| ~ thousands at n ~ 10³) for
    // EVERY family — Gaussian, binomial, GP/kriging alike. A scale-blind outer
    // takes the bare `tolerance` (≈1e-6) as the *absolute* projected-gradient
    // floor, which is hopelessly tight against an n-scaled gradient: in-basin
    // iterates (e.g. ‖g‖≈7e-2 at |f|≈17, or single-digit ‖g‖ at |f|≈1.3e3)
    // never clear it and the fit bails at the iteration cap. Worse, ARC's
    // trust-region reduction ratios and default initial regularization are
    // referenced against the wrong curvature magnitude, so the first step can
    // overshoot and diverge (the ‖g‖≈½|f| blow-ups in #1053/#1066). Threading
    // the scale (→ absolute floor = max(tol, n·1e-9)) plus a warm ARC
    // regularization (σ₀ = 0.25) and operator trust radius (4.0) makes the
    // spatial exact-joint outer converge as robustly as the primary REML outer
    // across 1-D Matérn (#1053), 2-D binomial geo (#1066), and GP/kriging
    // (#1069). This is NOT a loosening of the `τ·(1+|f|)` REML acceptance gate
    // — that relative-to-cost criterion is unchanged; only the nonsensical
    // scale-free *absolute* floor and the solver's curvature reference are
    // corrected. `None` preserves the prior scale-free calibration.
    profiled_objective_size: Option<(usize, usize)>,
    // `true` only after the isotropic Matérn endpoint profiler has certified a
    // winning range basin. The explicit theta0 then owns the sole joint-start
    // budget; generic multi-block and latent-coordinate callers retain their
    // family-specific multistart policies.
    initial_seed_only: bool,
) -> Result<gam_solve::rho_optimizer::OuterProblem, EstimationError> {
    if rho_dim > theta0.len() {
        crate::bail_invalid_estim!(
            "exact joint outer problem declares {rho_dim} smoothing coordinates for theta length {}",
            theta0.len(),
        );
    }
    gam_problem::validate_log_strengths(theta0.iter().take(rho_dim).copied()).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "exact joint initial smoothing coordinate is outside the canonical log-strength domain: {error}"
        ))
    })?;
    // The seed lattice reads its anchor in the outer coordinate, log λ (#1340).
    // An exp(ρ₀) anchor clamps to the domain's upper face, and every lattice point
    // built around it inherits that face (#2902 row 9, #2765).
    let seed_heuristic = theta0.to_vec();
    let mut problem = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_gradient(gradient)
        .with_hessian(hessian)
        // Exact REML/LAML curvature consumes the fourth-order family tower,
        // while BFGS search needs only exact gradients. Hessian availability is
        // a terminal-certification capability, not a warrant to rebuild that
        // tower at every accepted iterate (#979). Keep the Hessian declared so
        // the mint still requires exact curvature, but reserve it for that one
        // terminal evaluation.
        .with_prefer_gradient_only(true)
        // Exact joint spatial callers publish a selected coefficient mode as a
        // certified local minimum. Declare that second-order requirement here,
        // at the actual outer-problem construction boundary, so a raw-negative
        // terminal Hessian enters saddle recovery instead of surviving the
        // generic gradient-residue floor and failing later in fit assembly.
        //
        // BUT ONLY WHERE A HESSIAN WAS DECLARED. This used to be an unconditional
        // `true`, while the Hessian's availability arrives as a caller-supplied
        // parameter — so a caller passing `DeclaredHessianForm::Unavailable` built
        // a problem that both suppressed the analytic Hessian and required a
        // measured one. `run.rs` detects exactly that and refuses the mint by its
        // own words: "CONFIGURATION CONTRADICTION: the same outer problem both
        // suppressed the analytic Hessian and required a measured one. No
        // optimizer result can satisfy this — fix the construction ... rather than
        // the search". This is that fix.
        //
        // The caller in question is the #1033 n-free Gaussian ψ-lane
        // (`suppress_outer_hessian_for_nfree`), which declares `Unavailable`
        // BECAUSE the planner routes on the pair `(Analytic, Unavailable) ->
        // S::Bfgs` — that declaration is how the lane forces gradient-only search
        // and keeps every in-window κ-trial on the n-free design-realization skip.
        // So the suppression cannot simply be removed; the requirement is what has
        // to become conditional. One flag was answering two different questions —
        // "how should the SEARCH route?" and "must the MINT measure curvature?" —
        // and they have different answers on this lane.
        //
        // This cannot weaken any fit that mints today: a problem that never
        // declared a Hessian could never have had a measured one, so every fit
        // currently reaching the PSD requirement declared `Either` or `Analytic`
        // and is unaffected. What changes is only that a lane which today cannot
        // produce a fit at all can produce one.
        .with_require_measured_psd(!matches!(
            hessian,
            gam_problem::DeclaredHessianForm::Unavailable
        ))
        .with_disable_fixed_point(disable_fixed_point)
        // Re-enable the automatic fallback ladder for exact joint spatial
        // problems. It was previously `Disabled` to suppress a geo-bench
        // fallback bug where HybridEFS ψ stagnation degraded silently to
        // BfgsApprox on a Charbonnier surface. With the ψ-stagnation guard
        // in OuterFixedPointBridge (`MAX_CONSECUTIVE_PSI_STAGNATION`) the
        // bridge now surfaces `EFS_FIRST_ORDER_FALLBACK_MARKER` when ψ
        // stationarity cannot be enforced, so the ladder routes correctly
        // to a joint gradient-based solver instead of grinding HybridEFS
        // for thousands of iterations.
        .with_fallback_policy(gam_solve::rho_optimizer::FallbackPolicy::Automatic)
        .with_psi_dim(auxiliary_dim)
        .with_tolerance(tolerance)
        .with_max_iter(max_iter)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(theta0.clone())
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_bfgs_step_cap_psi(bfgs_step_cap_psi)
        // gam#1464: no explicit over-smoothing probe for constant-curvature terms.
        // The probe seeds the joint [ρ, ψ] solve at the collapsed-kernel corner
        // where the geodesic exponential exp(−d_κ/L) degenerates to a
        // near-constant. There the criterion is flat in κ (the kernel no longer
        // resolves curvature) and reduces to the monotone log-det Occam term, so
        // keep-best adopts the low-Occam collapsed null regardless of the true κ
        // sign: the bit-identical κ̂ → +chart-bound rail for both ±κ datasets (the
        // headline #1464 sign-blindness). Curvature is instead chosen once by the
        // sign-correct continuous likelihood-profile solve before this joint
        // nuisance optimization, and its coordinate is pinned here. The seed
        // lattice spans the declared domain `lower`/`upper` for every fit (#2902
        // row 9), so legitimate over-smoothing stays reachable by the analytic
        // gradient solve without pre-pinning a start at the collapsed corner.
        .with_seed_config(exact_joint_seed_config(
            risk_profile,
            auxiliary_dim,
            initial_seed_only,
        ))
        .with_heuristic_log_lambdas(seed_heuristic);
    if let Some((n_obs, p_cols)) = profiled_objective_size {
        // Calibrate to the n-scaled profiled criterion (see the param doc).
        // This is the scale the spatial exact-joint path was missing relative
        // to the primary REML outer; without it the iso-κ length-scale fit
        // stalls as |f| grows with n (#1053 / #1066 / #1069).
        problem = problem
            .with_objective_scale(Some(n_obs as f64))
            .with_problem_size(n_obs, p_cols);
    }
    if let Some(screening_cap) = screening_cap {
        problem = problem
            .with_screening_cap(screening_cap)
            .with_screen_initial_rho(true);
    }
    Ok(problem)
}

/// [`optimize_spatial_length_scale_exact_joint_typed`] for callers whose final
/// coefficient fit still reports text. Their fit failures cross as prose, which
/// the typed driver records as unclassified; its own failures are rendered to
/// the text these callers used to receive (#2937).
pub fn optimize_spatial_length_scale_exact_joint<FitOut, Mode, FitFn, ExactFn, ExactEfsFn, SeedFn>(
    data: ArrayView2<'_, f64>,
    block_specs: &[TermCollectionSpec],
    block_term_indices: &[Vec<usize>],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    joint_setup: &ExactJointHyperSetup,
    seed_risk_profile: gam_problem::SeedRiskProfile,
    analytic_joint_gradient_available: bool,
    analytic_joint_hessian_available: bool,
    disable_fixed_point: bool,
    screening_cap: Option<Arc<AtomicUsize>>,
    walk_signals: Option<crate::exact_mode_branch::OuterWalkSignals>,
    outer_derivative_policy: gam_model_api::families::custom_family::OuterDerivativePolicy,
    mut fit_fn: FitFn,
    exact_fn: ExactFn,
    exact_efs_fn: ExactEfsFn,
    seed_inner_beta_fn: SeedFn,
) -> Result<SpatialLengthScaleOptimizationResult<FitOut>, String>
where
    FitFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        SpatialFitProvenance<'_, Mode>,
    ) -> Result<FitOut, String>,
    ExactFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        gam_solve::estimate::reml::reml_outer_engine::EvalMode,
        Option<Mode>,
    ) -> Result<ExactJointEvaluation<Mode>, String>,
    ExactEfsFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<ExactJointEfsEvaluation<Mode>, String>,
    SeedFn: FnMut(&Array1<f64>) -> Result<gam_solve::rho_optimizer::SeedOutcome, EstimationError>,
{
    optimize_spatial_length_scale_exact_joint_typed(
        data,
        block_specs,
        block_term_indices,
        kappa_options,
        joint_setup,
        seed_risk_profile,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        disable_fixed_point,
        screening_cap,
        walk_signals,
        outer_derivative_policy,
        |theta: &Array1<f64>,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         provenance: SpatialFitProvenance<'_, Mode>| {
            fit_fn(theta, specs, designs, provenance).map_err(FitFailure::from)
        },
        exact_fn,
        exact_efs_fn,
        seed_inner_beta_fn,
    )
    .map_err(|failure| failure.to_string())
}

/// The custom-family refusal a typed fit failure carries, through the context and notes layers
/// placed in front of it (gam#2938).
pub(crate) fn custom_family_refusal_in(failure: &FitFailure) -> Option<&gam_problem::CustomFamilyError> {
    match failure {
        FitFailure::CustomFamily(refusal) => Some(refusal),
        FitFailure::Estimation(error) => match error.as_ref() {
            EstimationError::CustomFamily(refusal) => Some(refusal),
            _ => None,
        },
        FitFailure::Context { source, .. } | FitFailure::Annotated { source, .. } => {
            custom_family_refusal_in(source)
        }
        FitFailure::SurvivalMarginalSlope(_) | FitFailure::Workflow(_) | FitFailure::Raised { .. } => {
            None
        }
    }
}

/// The n-block exact-joint spatial driver. Its final coefficient fit and the
/// outer search's own verdict both reach the caller as a typed [`FitFailure`]
/// (#2937).
pub fn optimize_spatial_length_scale_exact_joint_typed<
    FitOut,
    Mode,
    FitFn,
    ExactFn,
    ExactErr,
    ExactEfsFn,
    SeedFn,
>(
    data: ArrayView2<'_, f64>,
    block_specs: &[TermCollectionSpec],
    block_term_indices: &[Vec<usize>],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    joint_setup: &ExactJointHyperSetup,
    seed_risk_profile: gam_problem::SeedRiskProfile,
    analytic_joint_gradient_available: bool,
    analytic_joint_hessian_available: bool,
    disable_fixed_point: bool,
    screening_cap: Option<Arc<AtomicUsize>>,
    walk_signals: Option<crate::exact_mode_branch::OuterWalkSignals>,
    outer_derivative_policy: gam_model_api::families::custom_family::OuterDerivativePolicy,
    mut fit_fn: FitFn,
    mut exact_fn: ExactFn,
    mut exact_efs_fn: ExactEfsFn,
    mut seed_inner_beta_fn: SeedFn,
) -> Result<SpatialLengthScaleOptimizationResult<FitOut>, FitFailure>
where
    FitFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        SpatialFitProvenance<'_, Mode>,
    ) -> Result<FitOut, FitFailure>,
    ExactFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        gam_solve::estimate::reml::reml_outer_engine::EvalMode,
        Option<Mode>,
    ) -> Result<ExactJointEvaluation<Mode>, ExactErr>,
    ExactErr: Into<FitFailure>,
    ExactEfsFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<ExactJointEfsEvaluation<Mode>, String>,
    SeedFn: FnMut(&Array1<f64>) -> Result<gam_solve::rho_optimizer::SeedOutcome, EstimationError>,
{
    let n_blocks = block_specs.len();
    if block_term_indices.len() != n_blocks {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            SmoothError::dimension_mismatch(format!(
                "block_specs ({}) and block_term_indices ({}) length mismatch",
                n_blocks,
                block_term_indices.len()
            ))
            .to_string(),
        ));
    }

    let log_kappa_dim = joint_setup.log_kappa_dim();

    log::trace!(
        "[spatial-exact-joint] driver entry: aux_dim={} log_kappa_dim={} kappa_enabled={} rho_dim={} theta0_len={}",
        joint_setup.auxiliary_dim(),
        log_kappa_dim,
        kappa_options.enabled,
        joint_setup.rho_dim(),
        joint_setup.theta0().len()
    );

    // -----------------------------------------------------------------------
    // Fast path: kappa disabled or no spatial terms — build designs once.
    // -----------------------------------------------------------------------
    if joint_setup.auxiliary_dim() == 0 && (!kappa_options.enabled || log_kappa_dim == 0) {
        log::trace!(
            "[spatial-exact-joint] taking fast path (no outer theta optimization in this driver)"
        );
        let (designs, resolved_specs) = build_term_collection_designs_and_freeze_joint(
            data, block_specs,
        )
        .map_err(|e| {
            format!("failed to build and freeze joint block designs during exact joint kappa optimization: {e}")
        })?;
        let theta0 = joint_setup.theta0();

        // Build temporary owned slices for the closure call.
        let spec_refs: Vec<TermCollectionSpec> = resolved_specs.clone();
        let design_refs: Vec<TermCollectionDesign> = designs.clone();
        let fit = fit_fn(
            &theta0,
            &spec_refs,
            &design_refs,
            SpatialFitProvenance::NoOuterOptimization,
        )?;
        return Ok(SpatialLengthScaleOptimizationResult {
            resolved_specs,
            designs,
            fit,
            certified_outer: None,
            timing: None,
        });
    }

    // -----------------------------------------------------------------------
    // Full optimization path.
    // -----------------------------------------------------------------------
    let theta0 = joint_setup.theta0();
    let lower = joint_setup.lower();
    let upper = joint_setup.upper();
    if theta0.len() < log_kappa_dim || lower.len() != theta0.len() || upper.len() != theta0.len() {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            SmoothError::dimension_mismatch(format!(
                "invalid exact joint theta setup: theta0={}, lower={}, upper={}, required_log_kappa_dim={}",
                theta0.len(),
                lower.len(),
                upper.len(),
                log_kappa_dim
            ))
            .to_string(),
        ));
    }
    let rho_dim = joint_setup.rho_dim();
    let all_dims = joint_setup.log_kappa_dims_per_term();

    // Build bootstrap designs and frozen specs for each block.
    let (boot_designs, best_specs) = build_term_collection_designs_and_freeze_joint(
        data,
        block_specs,
    )
    .map_err(|e| {
        format!(
            "failed to build and freeze joint block designs during exact joint kappa bootstrap: {e}"
        )
    })?;
    // The ρ half of the θ box is the domain the setup's builder derived over
    // every block that owns a ρ coordinate (#2812, #2902 item 15).
    log::info!(
        "[spatial-exact-joint] joint rho domain per coordinate: lower={:.3?} upper={:.3?} seed={:.3?}",
        lower.iter().take(rho_dim).copied().collect::<Vec<_>>(),
        upper.iter().take(rho_dim).copied().collect::<Vec<_>>(),
        theta0.iter().take(rho_dim).copied().collect::<Vec<_>>(),
    );
    // Capability vs realized policy: the family may *advertise* an exact
    // analytic outer Hessian, but at this realized (n, psi_dim, rho_dim,
    // p_total) the predicted per-eval cost can still exceed the universal
    // outer-Hessian work budget. In that regime we route the outer optimizer
    // through gradient-only BFGS / L-BFGS, which is **convergent** to the
    // exact MLE — it just takes more line-search iterations. This is **not**
    // a feature drop: quasi-Newton picks up curvature from successive
    // analytic gradients, and the per-eval cost saving (`O(p)` instead of
    // `O(p²)`) more than pays for the iteration overhead at large scale.
    let policy_hessian_form = outer_derivative_policy.declared_hessian_form();
    let analytic_outer_hessian_available = analytic_joint_hessian_available
        && matches!(
            policy_hessian_form,
            gam_problem::DeclaredHessianForm::Either
                | gam_problem::DeclaredHessianForm::Dense
                | gam_problem::DeclaredHessianForm::Operator { .. }
        );
    let theta_dim = theta0.len();
    let psi_dim = theta_dim - rho_dim;

    // Build the cache with one realizer per block.
    let cache_blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)> = best_specs
        .iter()
        .zip(boot_designs.iter())
        .zip(block_term_indices.iter())
        .map(|((spec, design), terms)| (spec.clone(), design.clone(), terms.clone()))
        .collect();

    struct NBlockExactJointState<'d, M> {
        cache: ExactJointDesignCache<'d>,
        terminal_mode: Option<(Array1<f64>, f64, M)>,
    }

    impl<M> NBlockExactJointState<'_, M> {
        /// Realize `theta`. An invalid input gains this driver's context, and every
        /// other variant passes through typed. A trial the realizer refuses stays a
        /// `TrialPointRefused` the outer search retreats from, not an InvalidInput
        /// that aborts the fit (#2953).
        fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), EstimationError> {
            let theta_changed = !self
                .cache
                .current_theta
                .as_ref()
                .is_some_and(|current| theta_values_match(current, theta));
            if theta_changed {
                self.terminal_mode = None;
            }
            self.cache.ensure_theta(theta).map_err(|error| match error {
                EstimationError::InvalidInput(message) => EstimationError::InvalidInput(format!(
                    "n-block exact-joint spatial design realization failed: {message}"
                )),
                other => other,
            })
        }

        fn install_terminal_mode(&mut self, theta: &Array1<f64>, objective: f64, mode: M) {
            self.terminal_mode = Some((theta.clone(), objective, mode));
        }

        fn terminal_mode_matches(&self, theta: &Array1<f64>, objective: f64) -> bool {
            self.terminal_mode
                .as_ref()
                .is_some_and(|(mode_theta, mode_objective, _)| {
                    theta_values_match(mode_theta, theta)
                        && mode_objective.to_bits() == objective.to_bits()
                })
        }

        fn take_terminal_mode(&mut self, theta: &Array1<f64>) -> Option<M> {
            if self
                .terminal_mode
                .as_ref()
                .is_some_and(|(mode_theta, _, _)| theta_values_match(mode_theta, theta))
            {
                self.terminal_mode.take().map(|(_, _, mode)| mode)
            } else {
                None
            }
        }
    }

    let mut state = NBlockExactJointState {
        cache: ExactJointDesignCache::new(data, cache_blocks, rho_dim, all_dims.clone())?,
        terminal_mode: None,
    };

    let n_total = data.nrows();

    let exact_fn_cell = std::cell::RefCell::new(&mut exact_fn);
    // The typed custom-family refusal of the last exact evaluation that refused. The outer
    // search sees each refusal as text; its verdict carries this one typed (gam#2938).
    let last_custom_family_refusal = std::cell::RefCell::new(None::<gam_problem::CustomFamilyError>);
    let exact_efs_fn_cell = std::cell::RefCell::new(&mut exact_efs_fn);

    // ── κ-optimization scaling instrumentation ──
    //
    // Per-phase wall-clock counters for the three kinds of evaluator
    // invocation the κ outer drives: cost-only line-search probes,
    // value-and-gradient(/Hessian) evaluations at accepted iterates, and
    // EFS fixed-point evaluations. Each invocation emits one
    // `[KAPPA-PHASE]` log line with a per-call elapsed time, plus the
    // running call counter and a summary `theta_norm` /
    // `log_kappa_norm` so the bench runner can attribute cost to
    // particular trajectory regions. A single `[KAPPA-PHASE-SUMMARY]`
    // line is emitted on optimization exit. Grepping these is the
    // production-fit κ-scaling probe (task #32) — measurement happens
    // in real large-scale fits rather than a synthetic harness, so the
    // scaling law reflects the actual workload.
    use std::cell::Cell;
    let kphase_cost_calls: Cell<usize> = Cell::new(0);
    let kphase_cost_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_eval_calls: Cell<usize> = Cell::new(0);
    let kphase_eval_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_efs_calls: Cell<usize> = Cell::new(0);
    let kphase_efs_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_optim_start = std::time::Instant::now();
    let kphase_log_kappa_dim = log_kappa_dim;
    let kphase_log_norms = |theta: &Array1<f64>| -> (f64, f64) {
        let theta_norm = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let log_kappa_norm = if kphase_log_kappa_dim > 0 && theta.len() >= kphase_log_kappa_dim {
            let start = theta.len() - kphase_log_kappa_dim;
            theta.iter().skip(start).map(|v| v * v).sum::<f64>().sqrt()
        } else {
            0.0
        };
        (theta_norm, log_kappa_norm)
    };

    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};
    use gam_solve::rho_optimizer::OuterEvalOrder;

    // Joint design width across blocks → the `p` reported to the outer solver's
    // operator-vs-dense Hessian crossover. `n_total` is the load-bearing
    // profiled-objective scale (see `exact_joint_multistart_outer_problem`).
    let joint_p_cols: usize = boot_designs
        .iter()
        .map(|d| d.design.ncols())
        .sum::<usize>()
        .max(1);

    let problem = exact_joint_multistart_outer_problem(
        &theta0,
        &lower,
        &upper,
        rho_dim,
        psi_dim,
        theta_dim,
        if analytic_joint_gradient_available {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        },
        if analytic_outer_hessian_available {
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        },
        disable_fixed_point,
        seed_risk_profile,
        kappa_options.rel_tol,
        kappa_options.max_outer_iter.max(1),
        // Rho-axis cap: log-λ natural step ≈ 5.
        Some(5.0),
        // Psi-axis cap: one doubling of the kernel length scale per iteration.
        Some(SPATIAL_PSI_BFGS_STEP_CAP),
        screening_cap.clone(),
        // n-scaled profiled-criterion calibration for every family (#1053 /
        // #1066 / #1069 iso-κ non-convergence cure).
        Some((n_total, joint_p_cols)),
        // Multi-block optimization has no preceding scalar Matérn endpoint
        // certificate, so retain its family-specific seed cascade.
        false,
    )?;
    let problem =
        crate::exact_mode_branch::OuterWalkSignals::subscribe(walk_signals.as_ref(), problem);

    // Helper: collect specs and designs from cache into owned Vecs for closure calls.
    fn collect_specs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionSpec> {
        cache.specs().into_iter().cloned().collect()
    }
    fn collect_designs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionDesign> {
        cache.designs().into_iter().cloned().collect()
    }

    let result = {
        let eval_outer = |ctx: &mut &mut NBlockExactJointState<'_, Mode>,
                          theta: &Array1<f64>,
                          order: OuterEvalOrder|
         -> Result<OuterEval, EstimationError> {
            if let Some((cost, grad, hess)) = ctx.cache.memoized_eval(theta)
                && ctx.terminal_mode_matches(theta, cost)
            {
                let cached_satisfies_order = match order {
                    OuterEvalOrder::Value => true,
                    OuterEvalOrder::ValueAndGradient => grad.len() == theta.len(),
                    OuterEvalOrder::ValueGradientHessian => {
                        grad.len() == theta.len() && hess.is_analytic()
                    }
                };
                if cached_satisfies_order {
                    if !cost.is_finite() {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    // Symmetric with the non-finite-cost guard above: a non-finite
                    // gradient marks this θ as infeasible just as a non-finite cost
                    // does (e.g. degenerate tied / zero-gap survival times drive the
                    // analytic exact-joint gradient channel to NaN/Inf). Return the
                    // bounded infeasible sentinel so the outer optimizer rejects the
                    // step and shrinks its trust region — instead of hard-failing the
                    // entire REML fit and handing the driver an unbroken stream of
                    // objective failures whose recovery path deepens once per outer
                    // step until the worker stack overflows (the survival
                    // location-scale path is the one that routes through this analytic
                    // gradient, which is why it crashed where the cost-only paths only
                    // stall).
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    return Ok(OuterEval {
                        cost,
                        gradient: grad,
                        hessian: hess,
                        inner_beta_hint: None,
                    });
                }
            }
            ctx.ensure_theta(theta)?;
            let design_revision = Some(ctx.cache.design_revision());
            let specs = collect_specs(&ctx.cache);
            let designs = collect_designs(&ctx.cache);
            // Clamp the requested order against the realized outer
            // derivative policy. The capability-aware
            // `analytic_outer_hessian_available` already encodes the
            // policy gate; re-checking through `order_for_evaluation`
            // here keeps the per-eval branch in lockstep with the
            // top-of-function declaration so the optimizer and the
            // evaluator never disagree on what was requested.
            let clamped = outer_derivative_policy.order_for_evaluation(order);
            let value_only = matches!(clamped, OuterEvalOrder::Value);
            let need_hessian = matches!(clamped, OuterEvalOrder::ValueGradientHessian)
                && analytic_outer_hessian_available;
            let eval_mode = if value_only {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueOnly
            } else if need_hessian {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            } else {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient
            };
            let owned_value_mode = if value_only {
                None
            } else {
                ctx.take_terminal_mode(theta)
            };
            let t0 = std::time::Instant::now();
            let result = (*exact_fn_cell.borrow_mut())(
                theta,
                &specs,
                &designs,
                eval_mode,
                owned_value_mode,
            );
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_eval_calls.set(kphase_eval_calls.get() + 1);
            kphase_eval_total_s.set(kphase_eval_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
            log::info!(
                "[KAPPA-PHASE] phase=eval_outer call={} order={:?} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_eval_calls.get(),
                order,
                design_revision,
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            match result {
                Ok(ExactJointEvaluation {
                    objective: cost,
                    gradient: grad,
                    hessian: hess,
                    mode,
                }) => {
                    last_custom_family_refusal.replace(None);
                    ctx.install_terminal_mode(theta, cost, mode);
                    if value_only {
                        ctx.cache.store_cost_only(theta, cost);
                    } else {
                        ctx.cache.store_eval((cost, grad.clone(), hess.clone()));
                    }
                    if !cost.is_finite() {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    // Symmetric with the non-finite-cost guard above: a non-finite
                    // gradient marks this θ as infeasible just as a non-finite cost
                    // does (e.g. degenerate tied / zero-gap survival times drive the
                    // analytic exact-joint gradient channel to NaN/Inf). Return the
                    // bounded infeasible sentinel so the outer optimizer rejects the
                    // step and shrinks its trust region — instead of hard-failing the
                    // entire REML fit and handing the driver an unbroken stream of
                    // objective failures whose recovery path deepens once per outer
                    // step until the worker stack overflows (the survival
                    // location-scale path is the one that routes through this analytic
                    // gradient, which is why it crashed where the cost-only paths only
                    // stall).
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    Ok(OuterEval {
                        cost,
                        gradient: grad,
                        hessian: hess,
                        inner_beta_hint: None,
                    })
                }
                // A refusal from the exact-joint evaluator is a refusal AT
                // THIS theta -- the same class the sibling `SpatialJointContext`
                // objective in this file already retreats from via
                // `is_recoverable_trial_point_error`. Reported as
                // `RemlOptimizationFailed`, `is_trial_point_infeasible`
                // answered false and `into_objective_error` graded it Fatal,
                // aborting the fit instead of the trial (#2627).
                Err(err) => {
                    let failure: FitFailure = err.into();
                    last_custom_family_refusal.replace(custom_family_refusal_in(&failure).cloned());
                    Err(EstimationError::TrialPointRefused {
                        reason: format!("n-block exact-joint spatial evaluation failed: {failure}"),
                    })
                }
            }
        };

        let obj = problem.build_objective_with_eval_order(
            &mut state,
            |ctx: &mut &mut NBlockExactJointState<'_, Mode>, theta: &Array1<f64>| {
                if let Some(cost) = ctx.cache.memoized_cost(theta)
                    && ctx.terminal_mode_matches(theta, cost)
                {
                    return Ok(cost);
                }
                ctx.ensure_theta(theta)?;
                let design_revision = Some(ctx.cache.design_revision());
                let specs = collect_specs(&ctx.cache);
                let designs = collect_designs(&ctx.cache);
                // Cost-only line-search probe: pass `ValueOnly` so the closure
                // skips gradient and Hessian assembly. This is the principled
                // fix for the N-block joint optimization V+G-per-probe waste —
                // gradient construction (≈ 6.5·10⁹ FLOPs per CTN step at
                // n=320 000, n_grid=293, p_resp=32, p_cov=23) is now paid only
                // when the outer evaluator actually requests it.
                let t0 = std::time::Instant::now();
                let result = (*exact_fn_cell.borrow_mut())(
                    theta,
                    &specs,
                    &designs,
                    gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueOnly,
                    None,
                );
                let elapsed_s = t0.elapsed().as_secs_f64();
                kphase_cost_calls.set(kphase_cost_calls.get() + 1);
                kphase_cost_total_s.set(kphase_cost_total_s.get() + elapsed_s);
                let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
                log::info!(
                    "[KAPPA-PHASE] phase=cost call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                    kphase_cost_calls.get(),
                    design_revision,
                    theta_norm,
                    log_kappa_norm,
                    elapsed_s,
                );
                match result {
                    Ok(ExactJointEvaluation {
                        objective: cost,
                        mode,
                        ..
                    }) => {
                        last_custom_family_refusal.replace(None);
                        ctx.install_terminal_mode(theta, cost, mode);
                        // Don't `store_eval`: that path is only valid when the
                        // closure produced a real gradient. The next outer-eval
                        // call will recompute (V, ∇V) at this θ if needed; the
                        // memoized_cost path covers the common case where the
                        // line search returns to an accepted iterate.
                        ctx.cache.store_cost_only(theta, cost);
                        Ok(cost)
                    }
                    Err(err) => {
                        let failure: FitFailure = err.into();
                        last_custom_family_refusal.replace(custom_family_refusal_in(&failure).cloned());
                        Err(EstimationError::TrialPointRefused {
                            reason: format!(
                                "n-block exact-joint spatial cost evaluation failed: {failure}"
                            ),
                        })
                    }
                }
            },
            |ctx: &mut &mut NBlockExactJointState<'_, Mode>, theta: &Array1<f64>| {
                // Search's legacy derivative bridge is first-order. The
                // order-aware hook below owns the terminal curvature request.
                eval_outer(ctx, theta, OuterEvalOrder::ValueAndGradient)
            },
            |ctx: &mut &mut NBlockExactJointState<'_, Mode>,
             theta: &Array1<f64>,
             order: OuterEvalOrder| { eval_outer(ctx, theta, order) },
            walk_signals
                .as_ref()
                .map(|signals| signals.reset_counter::<&mut NBlockExactJointState<'_, Mode>>()),
            Some(
                |ctx: &mut &mut NBlockExactJointState<'_, Mode>, theta: &Array1<f64>| {
                    ctx.ensure_theta(theta)?;
                    let design_revision = Some(ctx.cache.design_revision());
                    let specs = collect_specs(&ctx.cache);
                    let designs = collect_designs(&ctx.cache);
                    let t0 = std::time::Instant::now();
                    let eval_result = (*exact_efs_fn_cell.borrow_mut())(
                        theta,
                        &specs,
                        &designs,
                    );
                    let elapsed_s = t0.elapsed().as_secs_f64();
                    kphase_efs_calls.set(kphase_efs_calls.get() + 1);
                    kphase_efs_total_s.set(kphase_efs_total_s.get() + elapsed_s);
                    let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
                    log::info!(
                        "[KAPPA-PHASE] phase=efs call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                        kphase_efs_calls.get(),
                        design_revision,
                        theta_norm,
                        log_kappa_norm,
                        elapsed_s,
                    );
                    let ExactJointEfsEvaluation { evaluation, mode } =
                        eval_result.map_err(|reason| EstimationError::TrialPointRefused {
                            reason,
                        })?;
                    // An EFS solve can select a different coefficient mode at
                    // the same theta.  Revoke any derivative memo assembled
                    // from the previous mode before installing this carrier;
                    // a later analytic certification must then re-evaluate and
                    // replace both the derivative payload and the owned mode
                    // atomically.
                    ctx.cache.invalidate_objective_memo();
                    ctx.cache.store_cost_only(theta, evaluation.cost);
                    ctx.install_terminal_mode(theta, evaluation.cost, mode);
                    Ok(evaluation)
                },
            ),
        );
        let mut obj = obj
            .with_seed_inner_state(
                move |_: &mut &mut NBlockExactJointState<'_, Mode>, beta: &Array1<f64>| {
                    (seed_inner_beta_fn)(beta)
                },
            )
            // Declare the terminal evaluation order, which is what makes this
            // objective OWN its terminal coefficient mode.
            //
            // `ClosureObjective::owns_terminal_coefficient_mode()` is exactly
            // `terminal_eval_order.is_some()`. Without this call it answered
            // false, so `finalize_outer_result` fell through to `eval_efs` and
            // the terminal owner became the Fellner-Schall mode installed by the
            // EFS closure -- which has just invalidated the derivative memo.
            // Certification then missed the memo on BOTH its value lane and its
            // derivative lane and ran two fresh inner solves, warm-started off
            // whatever preceded them. Three coefficient modes at one theta, and
            // the published projected-gradient norm belonged to none of the
            // states any other consumer reads.
            //
            // Mode discrimination was never the gap -- every memo read is already
            // ANDed with `terminal_mode_matches`, which compares theta AND the
            // mode objective bitwise, so a stale-mode memo cannot be served. What
            // was missing is the declaration that routes finalization through
            // `eval_outer`, where `install_terminal_mode` and `store_eval` sit
            // together as the atomic pair the EFS closure's comment promises.
            //
            // The order must match what certification requests or the memo is
            // missed anyway: `certify_outer_optimality_at_terminal_fidelity` asks
            // for `ValueGradientHessian` iff the capability reports an analytic
            // Hessian, and this problem declares `DeclaredHessianForm::Either`
            // from the same `analytic_outer_hessian_available` flag.
            //
            // The `reset_fn` above only advances the walk-reset counter. `reset()`
            // fires AFTER finalization, so a reset that dropped the memo would
            // re-open the hole this closes.
            .with_terminal_eval_order(if analytic_outer_hessian_available {
                OuterEvalOrder::ValueGradientHessian
            } else {
                OuterEvalOrder::ValueAndGradient
            });

        // The outer search's verdict stays typed: a search whose every seed was
        // refused is not the same failure as one that started and stalled.
        problem
            .run_certified(&mut obj, "n-block exact-joint spatial")
            .map_err(|outer_error| match last_custom_family_refusal.take() {
                Some(refusal) => FitFailure::from(gam_problem::CustomFamilyError::OuterSmoothingFailed {
                    route: gam_problem::OuterSearchRoute::SpatialExactJoint,
                    reason: outer_error.to_string(),
                    last_refusal: Some(Box::new(refusal)),
                    // This driver keeps no record of an uncertified inner solve apart from
                    // its last refusal, so the fit boundary falls back to that (#2943).
                    search_inner_refusal: None,
                    outer_error: Arc::new(outer_error),
                }),
                None => FitFailure::from(outer_error),
            })?
    }; // obj dropped here, releasing mutable borrow on state

    // ── κ-optimization scaling summary ──
    //
    // Single line summarizing all per-call wall-clock counters
    // accumulated above. The bench runner / scaling-law analyzer
    // can pivot on this directly without parsing the per-call
    // [KAPPA-PHASE] markers (which remain available for
    // attribution).
    let kphase_total_s = kphase_optim_start.elapsed().as_secs_f64();
    log::info!(
        "[KAPPA-PHASE-SUMMARY] log_kappa_dim={} n_cost={} cost_total_s={:.4} n_eval={} eval_total_s={:.4} n_efs={} efs_total_s={:.4} optim_total_s={:.4}",
        kphase_log_kappa_dim,
        kphase_cost_calls.get(),
        kphase_cost_total_s.get(),
        kphase_eval_calls.get(),
        kphase_eval_total_s.get(),
        kphase_efs_calls.get(),
        kphase_efs_total_s.get(),
        kphase_total_s,
    );
    let timing = SpatialLengthScaleOptimizationTiming {
        log_kappa_dim: kphase_log_kappa_dim,
        cost_calls: kphase_cost_calls.get(),
        cost_total_s: kphase_cost_total_s.get(),
        eval_calls: kphase_eval_calls.get(),
        eval_total_s: kphase_eval_total_s.get(),
        efs_calls: kphase_efs_calls.get(),
        efs_total_s: kphase_efs_total_s.get(),
        slow_path_resets: 0,
        design_revision_delta: 0,
        nfree_skip_row_touches: 0,
        nfree_miss_shape: 0,
        nfree_miss_value: 0,
        nfree_miss_gradient: 0,
        nfree_miss_penalty: 0,
        nfree_miss_revision: 0,
        nfree_miss_second_order: 0,
        nfree_miss_other: 0,
        // The N-block driver never arms the #1033b ψ-Gram surrogate, so it has
        // no surrogate to retire.
        exact_polish_ran: false,
        polish_slow_path_resets: 0,
        polish_nfree_skip_row_touches: 0,
        optim_total_s: kphase_total_s,
    };

    let certified_outer = result;
    let theta_star = certified_outer.rho().clone();

    // The returned theta and certificate belong to the certified optimum. No
    // separate probe may mutate that certified identity before the final
    // coefficient fit.
    state.ensure_theta(&theta_star)?;
    let (mode_theta, mode_objective, mode) = state.terminal_mode.take().ok_or_else(|| {
        FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            "n-block exact-joint spatial optimization produced a certificate without retaining the owned terminal coefficient mode",
        )
    })?;
    if !theta_values_match(&mode_theta, &theta_star) {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            "n-block exact-joint spatial terminal coefficient mode does not bitwise match the certified hyperparameter vector",
        ));
    }
    if mode_objective.to_bits() != certified_outer.final_value().to_bits() {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            format!(
                "n-block exact-joint spatial terminal coefficient mode objective does not bitwise match the certified objective: mode={mode_objective:.17e}, certified={:.17e}",
                certified_outer.final_value(),
            ),
        ));
    }

    let resolved_specs: Vec<TermCollectionSpec> = collect_specs(&state.cache);
    let designs: Vec<TermCollectionDesign> = collect_designs(&state.cache);

    let fit = fit_fn(
        &theta_star,
        &resolved_specs,
        &designs,
        SpatialFitProvenance::Certified {
            outer: &certified_outer,
            mode,
        },
    )?;

    for spec in &resolved_specs {
        log_spatial_aniso_scales(spec);
    }

    Ok(SpatialLengthScaleOptimizationResult {
        resolved_specs,
        designs,
        fit,
        certified_outer: Some(certified_outer),
        timing: Some(timing),
    })
}

fn try_exact_joint_latent_coord_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodSpec,
    options: &FitOptions,
    latent: &StandardLatentCoordConfig,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};
    use gam_solve::rho_optimizer::OuterEvalOrder;

    let rho_dim = best.fit.lambdas.len();
    let latent_flat_dim = latent.values.len();
    if latent_flat_dim == 0 {
        crate::bail_invalid_estim!(
            "latent-coordinate optimization requires a non-empty latent block"
        );
    }
    let direct_hypers =
        latent_coord_initial_direct_hypers(latent.values.id_mode(), latent.values.latent_dim())?;
    let analytic_rho_count = latent
        .analytic_penalties
        .as_ref()
        .map_or(0, |registry| registry.total_rho_count());
    let latent_coord_ext_dim = latent_flat_dim + analytic_rho_count + direct_hypers.len();

    let mut theta0 = Array1::<f64>::zeros(rho_dim + latent_coord_ext_dim);
    theta0
        .slice_mut(s![..rho_dim])
        .assign(&best.fit.lambdas.mapv(f64::ln));
    theta0
        .slice_mut(s![rho_dim..rho_dim + latent_flat_dim])
        .assign(latent.values.as_flat());
    if !direct_hypers.is_empty() {
        let direct_start = rho_dim + latent_flat_dim + analytic_rho_count;
        theta0
            .slice_mut(s![direct_start..direct_start + direct_hypers.len()])
            .assign(&direct_hypers);
    }

    let mut lower = Array1::<f64>::from_elem(theta0.len(), -12.0);
    let mut upper = Array1::<f64>::from_elem(theta0.len(), 12.0);
    let latent_bound = latent
        .values
        .as_flat()
        .iter()
        .fold(1.0_f64, |acc, &v| acc.max(v.abs()))
        + 10.0;
    for axis in rho_dim..rho_dim + latent_flat_dim {
        lower[axis] = -latent_bound;
        upper[axis] = latent_bound;
    }
    if let Some(registry) = latent.analytic_penalties.as_ref() {
        let (domain_lower, domain_upper) = registry
            .rho_domain_bounds()
            .map_err(EstimationError::InvalidInput)?;
        let start = rho_dim + latent_flat_dim;
        for local in 0..analytic_rho_count {
            lower[start + local] = lower[start + local].max(domain_lower[local]);
            upper[start + local] = upper[start + local].min(domain_upper[local]);
            if lower[start + local] >= upper[start + local] {
                return Err(EstimationError::InvalidInput(format!(
                    "analytic-penalty rho domain has no searchable interval at coordinate {local}: lower={}, upper={}",
                    lower[start + local],
                    upper[start + local]
                )));
            }
        }
    }

    struct LatentJointContext<'d> {
        rho_dim: usize,
        cache: SingleBlockLatentCoordDesignCache,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator<'d>,
    }

    impl<'d> LatentJointContext<'d> {
        fn eval_full(
            &mut self,
            theta: &Array1<f64>,
            order: OuterEvalOrder,
        ) -> Result<(f64, Array1<f64>, gam_problem::HessianValue), EstimationError> {
            if let Some(eval) = self.cache.memoized_eval(theta) {
                return Ok(eval);
            }
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = self
                .cache
                .hyper_dirs()
                .map_err(EstimationError::InvalidInput)?;
            let design_revision = Some(self.cache.design_revision());
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let mut eval = evaluate_joint_reml_outer_eval_at_theta(
                &mut self.evaluator,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                order,
                design_revision,
            )?;
            let latent = self.cache.latent().map_err(EstimationError::InvalidInput)?;
            if let Some(registry) = registry_for_key {
                add_analytic_penalty_objective_to_eval(
                    theta,
                    self.rho_dim,
                    latent.as_ref(),
                    registry.as_ref(),
                    &mut eval,
                )?;
            }
            add_latent_id_objective_to_eval(
                theta,
                self.rho_dim,
                self.cache.analytic_penalty_rho_count(),
                latent.as_ref(),
                &mut eval,
            )?;
            self.cache.store_eval(eval.clone());
            Ok(eval)
        }

        fn eval_efs(
            &mut self,
            theta: &Array1<f64>,
        ) -> Result<gam_problem::EfsEval, EstimationError> {
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = self
                .cache
                .hyper_dirs()
                .map_err(EstimationError::InvalidInput)?;
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let mut efs = evaluate_joint_reml_efs_at_theta(
                &mut self.evaluator,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                Some(self.cache.design_revision()),
            )?;
            if let Some(registry) = registry_for_key {
                let latent = self.cache.latent().map_err(EstimationError::InvalidInput)?;
                let contribution = analytic_penalty_objective_contribution(
                    theta,
                    self.rho_dim,
                    latent.as_ref(),
                    registry.as_ref(),
                )?;
                efs.cost += contribution.cost;
                if let (Some(psi_gradient), Some(psi_indices)) =
                    (efs.psi_gradient.as_mut(), efs.psi_indices.as_ref())
                {
                    if psi_gradient.len() != psi_indices.len() {
                        crate::bail_invalid_estim!(
                            "latent-coordinate analytic penalty EFS psi gradient length mismatch: gradient={}, indices={}",
                            psi_gradient.len(),
                            psi_indices.len()
                        );
                    }
                    for (local_idx, &theta_idx) in psi_indices.iter().enumerate() {
                        psi_gradient[local_idx] += contribution.gradient[theta_idx];
                    }
                }
            }
            Ok(efs)
        }

        fn eval_cost(&mut self, theta: &Array1<f64>) -> f64 {
            if let Some(cost) = self.cache.memoized_cost(theta) {
                return cost;
            }
            if self.cache.ensure_theta(theta).is_err() {
                return f64::INFINITY;
            }
            let design_revision = Some(self.cache.design_revision());
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let result = {
                let design = self.cache.design();
                self.evaluator.evaluate_cost_only(
                    &design.design,
                    &design.penalties,
                    &design.nullspace_dims,
                    design.linear_constraints.clone(),
                    theta,
                    self.rho_dim,
                    None,
                    "latent-coordinate-joint cost-only",
                    design_revision,
                )
            };
            match result {
                Ok(cost) => {
                    let latent = match self.cache.latent() {
                        Ok(latent) => latent,
                        Err(_) => return f64::INFINITY,
                    };
                    let contribution = match latent_id_objective_contribution(
                        theta,
                        self.rho_dim,
                        self.cache.analytic_penalty_rho_count(),
                        latent.as_ref(),
                    ) {
                        Ok(contribution) => contribution,
                        Err(_) => return f64::INFINITY,
                    };
                    let cost = cost + contribution.cost;
                    let cost = if let Some(registry) = registry_for_key {
                        match analytic_penalty_objective_contribution(
                            theta,
                            self.rho_dim,
                            latent.as_ref(),
                            registry.as_ref(),
                        ) {
                            Ok(contribution) => cost + contribution.cost,
                            Err(_) => return f64::INFINITY,
                        }
                    } else {
                        cost
                    };
                    self.cache.store_cost(cost);
                    cost
                }
                Err(_) => f64::INFINITY,
            }
        }
    }

    let effective_offset = best
        .design
        .compose_offset(offset, "latent-coordinate joint fit")
        .map_err(EstimationError::BasisError)?;
    let mut ctx = LatentJointContext {
        rho_dim,
        cache: SingleBlockLatentCoordDesignCache::new(
            data.to_owned(),
            resolvedspec.clone(),
            best.design.clone(),
            latent,
            rho_dim,
        )
        .map_err(EstimationError::InvalidInput)?,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y,
            weights,
            &best.design.design,
            effective_offset.view(),
            &best.design.penalties,
            &external_opts_for_design(&family, &best.design, options),
            "latent-coordinate-joint",
        )?,
    };
    let registry_for_key = ctx.cache.analytic_penalties();
    ctx.evaluator
        .set_analytic_penalty_registry(registry_for_key.as_deref());
    ctx.evaluator
        .set_persistent_latent_values_fingerprint(latent.values.id_mode());
    if let Some(cached_t) = ctx
        .evaluator
        .load_persistent_latent_values(latent.values.n_obs(), latent.values.latent_dim())
    {
        let cached_t: Array2<f64> = cached_t;
        for (dst, src) in theta0
            .slice_mut(s![rho_dim..rho_dim + latent_flat_dim])
            .iter_mut()
            .zip(cached_t.iter())
        {
            *dst = *src;
        }
    }

    let problem = exact_joint_multistart_outer_problem(
        &theta0,
        &lower,
        &upper,
        rho_dim,
        latent_coord_ext_dim,
        theta0.len(),
        Derivative::Analytic,
        DeclaredHessianForm::Unavailable,
        false,
        seed_risk_profile_for_likelihood_family(&family),
        options.tol,
        options.max_iter.max(1),
        Some(5.0),
        Some(0.5),
        None,
        // n-scaled profiled-criterion calibration (same absolute-gradient-floor
        // correction as the spatial paths; #1053 / #1066 / #1069).
        Some((data.nrows(), best.design.design.ncols().max(1))),
        // Latent-coordinate optimization is not a profiled Matérn range solve.
        false,
    )?;

    let eval_outer = |ctx: &mut &mut LatentJointContext<'_>,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let (cost, gradient, hessian) = ctx.eval_full(theta, order)?;
        Ok(OuterEval {
            cost,
            gradient,
            hessian,
            inner_beta_hint: None,
        })
    };

    let result = {
        let obj = problem.build_objective_with_eval_order(
            &mut ctx,
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| Ok(ctx.eval_cost(theta)),
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| {
                eval_outer(ctx, theta, OuterEvalOrder::ValueAndGradient)
            },
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>, order: OuterEvalOrder| {
                eval_outer(ctx, theta, order)
            },
            Some(|ctx: &mut &mut LatentJointContext<'_>| {
                ctx.cache.reset();
            }),
            Some(|ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| ctx.eval_efs(theta)),
        );
        // #2676: same invariance hook as the iso-kappa arm — this route also
        // runs through `exact_joint_multistart_outer_problem`, which sets
        // `require_measured_psd`, so its certificate reaches the same curvature
        // verdict on the same kind of penalty map.
        let mut obj = obj.with_criterion_invariance(
            |ctx: &mut &mut LatentJointContext<'_>, rho: &Array1<f64>| {
                ctx.evaluator.criterion_invariant_directions(rho)
            },
        );

        problem
            .run(&mut obj, "latent-coordinate joint REML")
            .map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "latent-coordinate joint optimization failed after exhausting strategy fallbacks: {e}"
                ))
            })?
    };
    if !result.converged() {
        crate::bail_invalid_estim!(
            "latent-coordinate joint optimization did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
            result.iterations,
            result.final_value,
            result.final_grad_norm_report(),
        );
    }

    let theta_star = result.rho;
    let selected_log_lambdas = theta_star.slice(s![..rho_dim]).to_owned();
    gam_problem::validate_log_strengths(selected_log_lambdas.iter().copied()).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "selected latent-coordinate smoothing coordinate is outside the canonical log-strength domain: {error}"
        ))
    })?;
    let mut final_data = data.to_owned();
    let flat_t = theta_star
        .slice(s![rho_dim..rho_dim + latent_flat_dim])
        .to_owned();
    let mut fitted_latent_values =
        Array2::<f64>::zeros((latent.values.n_obs(), latent.values.latent_dim()));
    for n in 0..latent.values.n_obs() {
        for axis in 0..latent.values.latent_dim() {
            let value = flat_t[n * latent.values.latent_dim() + axis];
            fitted_latent_values[[n, axis]] = value;
            final_data[[n, latent.feature_cols[axis]]] = value;
        }
    }
    let optimized = fit_term_collection_forspecwith_heuristic_log_lambdas(
        final_data.view(),
        y,
        weights,
        offset,
        resolvedspec,
        selected_log_lambdas.as_slice(),
        family,
        options,
    )?;
    ctx.evaluator
        .store_persistent_latent_values(&fitted_latent_values);
    let mut fit = optimized.fit;
    fit.set_criterion(Some(result.final_value));
    Ok(FittedTermCollectionWithSpec {
        fit,
        design: optimized.design,
        resolvedspec: resolvedspec.clone(),
        kappa_timing: None,
    })
}

pub(crate) fn fit_term_collectionwith_latent_coord_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    latent: &StandardLatentCoordConfig,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        crate::bail_invalid_estim!(
            "fit_term_collectionwith_latent_coord_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        );
    }
    let best = fit_term_collection_forspec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        spec,
        family.clone(),
        options,
    )?;
    let resolvedspec = freeze_term_collection_from_design(spec, &best.design)?;
    try_exact_joint_latent_coord_optimization(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        &best,
        family,
        options,
        latent,
    )
}

/// Resolve the two physically distinct isotropic Matérn range basins before the
/// local joint `[rho, psi]` solve.
///
/// The short/rich basin is represented by the ordinary cold fit at the
/// observation-density seed. The competing long-range basin has one canonical,
/// data-derived representative: the rotation-invariant fill distance of the
/// reduced-rank center set, `extent_rot / sqrt(k)`. Profile all smoothing
/// parameters at that endpoint once, and retain it only when its certified REML
/// objective is strictly lower. The subsequent joint optimizer therefore runs
/// exactly once, from the winning basin.
///
/// This is a closed endpoint comparison, not a lattice, sweep, or collection of
/// joint restarts. Both profiles pass through `fit_term_collection_forspec`, so
/// either produces a fully certified fit or the error is surfaced; there is no
/// best-effort fallback. Pairwise-distance bounds and the incumbent Matérn seed
/// are Euclidean invariants, so the decision is unchanged by rigid rotations.
fn select_isotropic_matern_range_basin(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    mut resolvedspec: TermCollectionSpec,
    mut best: FittedTermCollection,
    family: &LikelihoodSpec,
    options: &FitOptions,
    spatial_terms: &[usize],
) -> Result<(TermCollectionSpec, FittedTermCollection), EstimationError> {
    // Per-axis anisotropy and signed curvature have dedicated geometry
    // estimators. Their outer coordinates are not a scalar Matérn range and
    // therefore do not participate in this two-basin decision.
    if has_aniso_terms(&resolvedspec, spatial_terms)
        || !constant_curvature_term_indices(&resolvedspec).is_empty()
    {
        return Ok((resolvedspec, best));
    }

    let mut best_score = fit_score(&best.fit);
    if !best_score.is_finite() {
        crate::bail_invalid_estim!(
            "isotropic Matérn basin selection received a non-finite incumbent profile"
        );
    }

    for &term_idx in spatial_terms {
        let Some(SmoothBasisSpec::Matern {
            feature_cols,
            spec: matern,
            ..
        }) = resolvedspec
            .smooth_terms
            .get(term_idx)
            .map(|term| &term.basis)
        else {
            continue;
        };
        let num_centers = gam_terms::basis::center_strategy_num_centers(&matern.center_strategy)
            .ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "resolved isotropic Matérn term {term_idx} has no finite center count"
                ))
            })?;
        let companion_length_scale = matern_low_rank_center_resolution_length_scale(
            data,
            feature_cols,
            num_centers,
        )
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "resolved isotropic Matérn term {term_idx} has no finite center-resolution range"
            ))
        })?;
        let (psi_long_bound, psi_short_bound) = spatial_term_psi_bounds(data, &resolvedspec, term_idx)
            .map_err(EstimationError::BasisError)?;
        let psi_long = (-companion_length_scale.ln()).clamp(psi_long_bound, psi_short_bound);
        let long_length_scale = (-psi_long).exp();
        if !(long_length_scale.is_finite() && long_length_scale > 0.0) {
            crate::bail_invalid_estim!(
                "isotropic Matérn term {term_idx} produced an invalid long-range endpoint from psi={psi_long}"
            );
        }
        if get_spatial_length_scale(&resolvedspec, term_idx)
            .is_some_and(|current| current == long_length_scale)
        {
            continue;
        }

        let mut endpoint_spec = resolvedspec.clone();
        set_spatial_length_scale(&mut endpoint_spec, term_idx, long_length_scale)?;
        // Profile rho at the competing geometry by starting the ordinary outer
        // optimizer literally at the already certified incumbent rho. This is
        // still a full standard REML
        // solve (including its ordinary seed certification), but it avoids
        // throwing away the exact smoothing optimum immediately before a
        // deliberately coarser center-resolution geometry move. The incumbent
        // lambdas provide the well-scaled starting chart needed for that profile
        // to reach its KKT certificate rather than exhausting its startup plans
        // a few ulps above stationarity.
        let endpoint = refit_term_collection_at_fitted_strengths(
            data,
            y,
            weights,
            offset,
            &endpoint_spec,
            &best.fit,
            family.clone(),
            options,
        )?;
        let endpoint_score = fit_score(&endpoint.fit);
        if !endpoint_score.is_finite() {
            crate::bail_invalid_estim!(
                "isotropic Matérn term {term_idx} long-range endpoint returned a non-finite profiled REML score"
            );
        }

        if endpoint_score < best_score {
            log::info!(
                "[spatial-kappa] term {term_idx} selected certified long-range basin: \
                 length_scale={long_length_scale:.6}, profiled REML {endpoint_score:.6} \
                 < short-basin {best_score:.6}"
            );
            resolvedspec = freeze_term_collection_from_design(&endpoint_spec, &endpoint.design)?;
            best = endpoint;
            best_score = endpoint_score;
        } else {
            log::info!(
                "[spatial-kappa] term {term_idx} retained certified short-range basin: \
                 profiled REML {best_score:.6} <= long-endpoint {endpoint_score:.6} \
                 at length_scale={long_length_scale:.6}"
            );
        }
    }

    Ok((resolvedspec, best))
}

pub fn fit_term_collectionwith_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    // Spatial hyperparameters change kernel geometry nonlinearly, so each
    // proposal rebuilds the spatial basis. Hybrid/isotropic terms expose a
    // scalar κ (= 1/length_scale); pure Duchon anisotropy exposes only
    // per-axis shape coordinates.
    //
    // When exact derivative information is available for the rebuilt basis and
    // penalty, kappa is promoted to a first-class outer hyperparameter beside
    // rho = log(lambda). In that mode this routine runs a joint outer solve in
    // theta = [rho, psi], where psi = log(kappa) = -log(length_scale), and the
    // optimizer is expected to consume a real joint Hessian. ARC is not meant
    // to run on a gradient-only surrogate here.
    //
    // Any eligible spatial smooth participates in this outer solve. If an
    // eligible spatial basis does not expose derivative information, that is
    // now a hard error.
    let (resolvedspec, best, spatial_terms, initial_score) = match spatial_kappa_incumbent(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        spec,
        &family,
        options,
        kappa_options,
    )? {
        SpatialKappaIncumbent::Final(fitted) => return Ok(fitted),
        SpatialKappaIncumbent::Joint {
            resolvedspec,
            best,
            spatial_terms,
            initial_score,
        } => (resolvedspec, best, spatial_terms, initial_score),
    };
    let exact_joint = match try_exact_joint_spatial_length_scale_optimization(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        &best,
        family.clone(),
        options,
        kappa_options,
        &spatial_terms,
    )? {
        JointSpatialKappaOutcome::Optimized(optimized) => *optimized,
        JointSpatialKappaOutcome::DeclinedKeepIncumbent {
            baseline_score,
            optimized_score,
            kappa_timing,
        } => {
            // The route ran, graded its own candidate against the shipped
            // score and declined it. Shipping the incumbent is what the
            // decline MEANS -- its own log line promises exactly that -- so
            // the fit continues at the incumbent κ, which is the same thing
            // that happens when there is no eligible spatial term at all
            // (the branch above). It is not an unavailability, and turning it
            // into one killed fits the route had just decided were fine
            // (#2748).
            log::info!(
                "[spatial-kappa] joint kappa optimization DECLINED its own candidate                  (incumbent={baseline_score:.12e}, candidate={optimized_score:.12e},                  regression={:.3e}); shipping the incumbent scalar-route fit at the                  incumbent κ, which is what the decline means. Not an unavailability.",
                optimized_score - baseline_score,
            );
            let fitted = refit_term_collection_at_fitted_strengths(
                data,
                y.view(),
                weights.view(),
                offset.view(),
                &resolvedspec,
                &best.fit,
                family,
                options,
            )?;
            return Ok(FittedTermCollectionWithSpec {
                fit: fitted.fit,
                design: fitted.design,
                resolvedspec,
                kappa_timing: Some(kappa_timing),
            });
        }
        JointSpatialKappaOutcome::Unavailable => {
            return Err(EstimationError::RemlOptimizationFailed(
                "spatial kappa optimization is unavailable for one or more eligible spatial                  terms"
                    .to_string(),
            ));
        }
    };
    let exact_joint = require_available_spatial_optimization_result(Ok(Some(exact_joint)))?;
    let exact_score = fit_score(&exact_joint.fit);

    // Keep whichever of the two SCORED fits is better (#2748). κ optimization
    // is a refinement of a fit that already exists, so "the refinement did not
    // improve on the incumbent" is an argument for shipping the incumbent, not
    // for destroying it — which is what this site did, on a bar of
    // `max(1e-6, |score|·1e-8)`, until `geo_disease_eas_matern_k6` lost all
    // four of its non-flexible benchmark lanes to a `1.267594e3 → 1.267595e3`
    // regression. It is also exactly the conclusion the sibling
    // `DeclinedKeepIncumbent` arm above reaches when the joint route grades its
    // own candidate one level in; two graders of one comparison must not reach
    // opposite responses.
    //
    // An `argmin` over two measured numbers needs no tolerance and admits no
    // drift argument: it cannot ship something worse than what it was handed.
    // A tie goes to the candidate, because the refinement is what was asked
    // for and a tied score means the two fits are equally supported.
    if exact_score.is_finite() && exact_score <= initial_score {
        log_spatial_aniso_scales(&exact_joint.resolvedspec);
        return Ok(exact_joint);
    }
    log::info!(
        "[spatial-kappa] the optimized-κ fit scores {exact_score:.12e} against the incumbent's \
         {initial_score:.12e} (regression {:.3e}); shipping the INCUMBENT, which is the better \
         of the two fits this call has in hand. A refinement that does not improve on the fit \
         it refines is not a reason to have no fit (#2748).",
        exact_score - initial_score,
    );
    let fitted = refit_term_collection_at_fitted_strengths(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        &best.fit,
        family,
        options,
    )?;
    Ok(FittedTermCollectionWithSpec {
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec,
        kappa_timing: exact_joint.kappa_timing,
    })
}

/// What the spatial length-scale pipeline hands the joint κ route: a fit that
/// needs no joint route, or the scalar-ρ incumbent the route refines.
enum SpatialKappaIncumbent {
    /// κ optimization is disabled, or no spatial coordinate is left to optimize.
    Final(FittedTermCollectionWithSpec),
    /// The frozen spec, its scalar-ρ fit, the spatial terms the joint route
    /// enrolls, and that fit's score.
    Joint {
        resolvedspec: TermCollectionSpec,
        best: FittedTermCollection,
        spatial_terms: Vec<usize>,
        initial_score: f64,
    },
}

/// The spatial length-scale pipeline up to the joint κ route: the length-scale
/// projection, the anisotropy and curvature seeds, the scalar-ρ baseline fit,
/// the freeze and the Matérn range basin. The gam#2895 gradient gate starts the
/// route from here as well, so it grades the route at production's own θ0.
fn spatial_kappa_incumbent(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    family: &LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SpatialKappaIncumbent, EstimationError> {
    let mut resolvedspec = spec.clone();
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        crate::bail_invalid_estim!(
            "fit_term_collectionwith_spatial_length_scale_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        );
    }
    // #2750: choose the measure-jet representer range's SEED from the response
    // rather than from the node spacing alone. The profiled criterion in `ln ℓ`
    // is not unimodal, so the local descent that follows cannot leave the basin
    // it starts in; the screen picks the basin and the search still owns the
    // range inside it. Runs before `spatial_length_scale_term_indices` only for
    // readability — the enrollment predicate does not read `length_scale`.
    // Skipped for pinned, frozen and already-standardized terms; see
    // `seed_measure_jet_auto_ranges`.
    seed_measure_jet_auto_ranges(data, y.view(), weights.view(), &mut resolvedspec);
    let spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
    if !kappa_options.enabled || spatial_terms.is_empty() {
        let out = fit_term_collection_forspec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            family.clone(),
            options,
        )?;
        let resolvedspec = freeze_term_collection_from_design(&resolvedspec, &out.design)?;
        return Ok(SpatialKappaIncumbent::Final(FittedTermCollectionWithSpec {
            fit: out.fit,
            design: out.design,
            resolvedspec,
            kappa_timing: None,
        }));
    }
    if kappa_options.max_outer_iter == 0 {
        crate::bail_invalid_estim!("spatial kappa optimization requires max_outer_iter >= 1");
    }
    // #1376: the geometry-only anisotropy seed (`initial_aniso_contrasts`, from
    // per-axis knot-coordinate spread) is blind to the response, so a signal
    // axis and a nuisance axis with equal coordinate spread both seed to ~0 and
    // the κ optimizer can stall at the symmetric point (it found a weak/flat
    // antisymmetric gradient, amplified by double-penalty nullspace shrinkage).
    // Add a bounded, response-aware per-axis nudge so the optimizer starts in
    // the correct basin.
    apply_response_aware_anisotropy_seed(data, y.view(), &mut resolvedspec, &spatial_terms);

    // Select every free constant-curvature coordinate once from its continuous,
    // analytically differentiated likelihood profile before fitting the baseline.
    // That profile is the sole owner of BOTH of the smooth's coordinates — the
    // signed curvature and the log range (#2747) — so a later joint REML solve
    // must not enroll either of them against a different objective, and the
    // ordinary fixed-geometry fit below already profiles rho at the certified
    // pair.
    //
    // A PINNED `kappa=` takes the term out of the CURVATURE search — fixed
    // geometry is the whole contract of `kappa=` (gam#2152) — but not out of the
    // range one. It used to: `20bde053f` reverted the pinned-κ/free-range
    // enrollment because the range criterion was "monotone in `ℓ` all the way to
    // its asymptote … `ℓ̂` ran to 1.5e6, a readout of the box rather than of the
    // data", and asked for "a derived stopping rule for a criterion that
    // converges rather than turning over".
    //
    // Both halves of that are now answered rather than deferred (gam#2747).
    // The monotone descent past `ℓ ≈ 10⁶` was not the criterion converging, it
    // was the criterion FABRICATED — the `exp(−d/ℓ)` gauge put every bit of the
    // range's information into `K − 1` and formed it by subtraction, so the
    // value fell ~100 nats per decade into its own cancellation and `edf` railed
    // at `p`. The contrast gauge removes that, the chart's top is now derived
    // from where the model stops moving (the kernel IS the geodesic distance to
    // within `√ε`), and arriving there is a DECLARED outcome rather than a rail.
    // A criterion that converges to a member of its own family does not need a
    // stopping rule; it needs its limit to be a point of the chart.
    //
    // So: κ free ⇒ both coordinates from the profile. κ pinned, range free ⇒ the
    // range alone, at that κ, from the SAME inner solve. Range pinned ⇒ neither.
    // The pinned-κ arm is skipped rather than refused when the profile's
    // Gaussian-identity/unit-weight precondition does not hold: the range is a
    // nuisance coordinate there and the auto `ℓ_ref` is a valid fallback, while
    // for a free κ the profile IS the estimand and there is nothing to fall back
    // to.
    let free_curvature_terms: Vec<usize> = constant_curvature_term_indices(&resolvedspec)
        .into_iter()
        .filter(|&term_idx| !constant_curvature_kappa_is_fixed(&resolvedspec, term_idx))
        .collect();
    let pinned_kappa_free_range_terms: Vec<usize> =
        constant_curvature_term_indices(&resolvedspec)
            .into_iter()
            .filter(|&term_idx| {
                constant_curvature_kappa_is_fixed(&resolvedspec, term_idx)
                    && !constant_curvature_length_scale_is_fixed(&resolvedspec, term_idx)
            })
            .collect();
    if !free_curvature_terms.is_empty() {
        validate_constant_curvature_profile_inputs(weights.view(), offset.view(), &family)?;
    }
    if !pinned_kappa_free_range_terms.is_empty()
        && validate_constant_curvature_profile_inputs(weights.view(), offset.view(), &family)
            .is_ok()
    {
        for term_idx in pinned_kappa_free_range_terms {
            let length_scale_hat =
                constant_curvature_range_only_optimum(data, y.view(), &resolvedspec, term_idx)?;
            if let Some(SmoothBasisSpec::ConstantCurvature { spec: cc, .. }) = resolvedspec
                .smooth_terms
                .get_mut(term_idx)
                .map(|term| &mut term.basis)
            {
                // `length_scale_fixed` stays as the user left it, for the same
                // reason the free-κ arm leaves it alone: a realized value frozen
                // into the spec must not be mistaken for a pin on a later fit.
                cc.length_scale = length_scale_hat;
            }
        }
    }
    for term_idx in free_curvature_terms {
        let psi_hat = constant_curvature_kappa_profile_optimum(
            data,
            y.view(),
            &resolvedspec,
            term_idx,
            options,
        )?;
        if let Some(SmoothBasisSpec::ConstantCurvature { spec: cc, .. }) = resolvedspec
            .smooth_terms
            .get_mut(term_idx)
            .map(|term| &mut term.basis)
        {
            cc.kappa = psi_hat.kappa;
            // Write the fitted range back too (#2747). `length_scale_fixed` is
            // left alone: it records whether the USER pinned the range, and a
            // realized value frozen into the spec must not be mistaken for a
            // pin on a later fit of the same spec.
            cc.length_scale = psi_hat.length_scale;
        }
    }

    let baseline_options = superseded_fit_options(options);
    let best = fit_term_collection_forspec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        family.clone(),
        &baseline_options,
    )?;
    resolvedspec = freeze_term_collection_from_design(&resolvedspec, &best.design)?;
    // The freeze step can rewrite a term's basis variant — most notably when
    // `build_thin_plate_basis_with_workspace` auto-promotes an infeasible
    // canonical-TPS request to a pure Duchon spline (length_scale = None,
    // no anisotropy). The pre-fit eligibility list was computed against the
    // ThinPlate spec, which has length_scale set, so it included that term.
    // After the rewrite the same term is a *pure* Duchon basis with no free
    // length-scale parameter to optimize, and the downstream kappa solver
    // (which assumes hybrid Duchon for log-κ derivatives) errors out. Refresh
    // the index list so it reflects the post-freeze spec.
    // Constant curvature is no longer a joint-REML coordinate at this point.
    // A free κ was just certified by the curvature profile; a user-pinned
    // κ is fixed geometry. In both cases `best` has already profiled rho at that
    // exact κ. Keeping the term in the generic spatial list would manufacture a
    // degenerate ψ axis (`lower == upper`) and re-profile rho through a second
    // evaluator, despite there being no spatial coordinate left to optimize.
    // Besides doing dead work, that gave κ two objective owners and made the
    // scalar and joint routes disagree at the identical seed on flat data.
    let spatial_terms: Vec<usize> = spatial_length_scale_term_indices(&resolvedspec)
        .into_iter()
        .filter(|&term_idx| constant_curvature_term_spec(&resolvedspec, term_idx).is_none())
        .collect();
    let (next_spec, best) = select_isotropic_matern_range_basin(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        resolvedspec,
        best,
        &family,
        &baseline_options,
        &spatial_terms,
    )?;
    resolvedspec = next_spec;
    // Sync knot-cloud-derived aniso contrasts from the basis metadata back
    // into the spec so the optimizer starts from the geometry-informed η values
    // rather than the zero sentinel from --scale-dimensions.
    sync_aniso_contrasts_from_metadata(&mut resolvedspec, &best.design.smooth);
    if spatial_terms.is_empty() {
        let fitted = refit_term_collection_at_fitted_strengths(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            &best.fit,
            family.clone(),
            options,
        )?;
        return Ok(SpatialKappaIncumbent::Final(FittedTermCollectionWithSpec {
            fit: fitted.fit,
            design: fitted.design,
            resolvedspec,
            kappa_timing: None,
        }));
    }
    let initial_score = fit_score(&best.fit);
    if !initial_score.is_finite() {
        crate::bail_invalid_estim!(
            "spatial kappa optimization received a non-finite initial profiled score"
        );
    }
    Ok(SpatialKappaIncumbent::Joint {
        resolvedspec,
        best,
        spatial_terms,
        initial_score,
    })
}

// Curvature inference for `curv(...)` smooths and the driver's trailing unit tests (moved verbatim, line limit).
include!("spatial_curvature_inference.rs");
