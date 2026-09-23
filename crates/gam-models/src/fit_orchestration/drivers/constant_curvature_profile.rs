// #2747 — the constant-curvature smooth's outer objective, in its OWN two
// coordinates `ψ = (κ, η = ln ℓ)`.
//
// Extracted from `spatial_optimization.rs` for the same reason
// `constant_curvature_kappa_jet.rs` was: that file sits at the 10,000-line ban
// and this machinery grew when the range became an estimated coordinate rather
// than a heuristic. `include!`d into `drivers/mod.rs` exactly like the sibling
// files, so the flat module namespace and every private-item reference are
// unchanged.
//
// Everything the curvature estimand is built on lives here and nowhere else:
// the value-only criterion the inner bracket screens with, the full ψ jet the
// Newton refines with, the profile object that owns both, and the bounded outer
// solve that mints κ̂. One owner, because the point estimate, the profile CI and
// the flatness LR have to be extrema of the same object — this subsystem
// already carries the scar from the last time one coordinate had two.

/// The profile's VALUE alone at one `(κ, η)`, with no derivative blocks built.
///
/// The bracketing scan calls this and the Newton refinement calls the full jet.
/// The split is worth its own function because the two costs are not close: the
/// value needs one kernel pass (`distance`), the jet needs the Tower2 κ-jet of
/// every pair plus five more `n×p` blocks. A thirteen-point deterministic
/// bracket at jet cost would multiply a production `curv(...)` fit's outer work
/// by an order of magnitude for information the bracket does not use.
fn constant_curvature_psi_profile_value(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    spec: &gam_terms::basis::ConstantCurvatureBasisSpec,
) -> Result<f64, EstimationError> {
    // ONE penalty, because this criterion is a single-λ closed form. That is a
    // restriction on the MODEL, not a formatting choice, so
    // `ConstantCurvatureProfile::new` refuses a `double_penalty=` term outright
    // rather than letting this line quietly score a different one — see the
    // argument there. By the time control reaches here the flag is already
    // false; the assignment stands so a future caller that bypasses the
    // constructor cannot silently get two penalties and one λ.
    let mut profile_spec = spec.clone();
    profile_spec.double_penalty = false;
    let basis = gam_terms::basis::build_constant_curvature_basis(data, &profile_spec)
        .map_err(EstimationError::from)?;
    if basis.active_penalties.len() != 1 {
        crate::bail_invalid_estim!(
            "constant-curvature profile expected exactly one primary penalty; got {}",
            basis.active_penalties.len()
        );
    }
    let smooth_design = basis.design.to_dense();
    let (n, p) = smooth_design.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth_design);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix);
    let response_2d = y.insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response_2d.view(),
        penalty.view(),
        None,
        None,
    )?;
    // A ρ̂ on a face of `fit.rho_domain` is a λ-profile value like any other; see
    // `ConstantCurvatureProfile::evaluate_value` for why it is comparable.
    Ok(fit.reml_score)
}

/// Value, exact gradient and exact Hessian of the continuously
/// smoothing-profiled Gaussian REML negative log evidence used for curvature
/// inference, in the smooth's TWO outer coordinates `ψ = (κ, η)`, `η = ln ℓ`.
///
/// The likelihood-ratio statistic must compare values of this one likelihood.
/// Subtracting a second REML fit to a response-dependent radial smoother would
/// produce neither a likelihood nor a calibrated likelihood ratio: the
/// subtraction can manufacture curvature signal even when the response is
/// constant plus noise.
///
/// The range enters as a coordinate rather than as a heuristic because it is
/// confounded with the curvature (#2747): pinning ℓ makes κ absorb the range
/// error, and the criterion then rails, inverts the reported sign, or invents
/// curvature from flat data. The exact second derivatives are what let this
/// route run the SAME stationarity certificate every other route runs (#2458).
fn constant_curvature_psi_profile_jet(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    spec: &gam_terms::basis::ConstantCurvatureBasisSpec,
) -> Result<ProfiledRemlPsiJet, EstimationError> {
    if y.len() != data.nrows() || y.is_empty() {
        crate::bail_invalid_estim!(
            "constant-curvature profile needs one non-empty response per row: data={}, response={}",
            data.nrows(),
            y.len(),
        );
    }

    // One penalty; see `constant_curvature_psi_profile_value` and the refusal in
    // `ConstantCurvatureProfile::new`.
    let mut profile_spec = spec.clone();
    profile_spec.double_penalty = false;
    let basis = gam_terms::basis::build_constant_curvature_basis(data, &profile_spec)
        .map_err(EstimationError::from)?;
    let jets =
        gam_terms::basis::build_constant_curvature_basis_psi_derivatives(data, &profile_spec)
            .map_err(EstimationError::from)?;
    let penalty_block_counts = [
        basis.active_penalties.len(),
        jets.penalties_kappa.len(),
        jets.penalties_eta.len(),
        jets.penalties_kappa2.len(),
        jets.penalties_kappa_eta.len(),
        jets.penalties_eta2.len(),
    ];
    if penalty_block_counts.iter().any(|&count| count != 1) {
        crate::bail_invalid_estim!(
            "constant-curvature profile expected exactly one primary penalty in every block; got {penalty_block_counts:?}"
        );
    }

    let smooth_design = basis.design.to_dense();
    let n = smooth_design.nrows();
    let p = smooth_design.ncols();
    let smooth_penalty = &basis.active_penalties[0].matrix;
    let smooth_design_blocks = [
        &jets.design_kappa,
        &jets.design_eta,
        &jets.design_kappa2,
        &jets.design_kappa_eta,
        &jets.design_eta2,
    ];
    let smooth_penalty_blocks = [
        &jets.penalties_kappa[0],
        &jets.penalties_eta[0],
        &jets.penalties_kappa2[0],
        &jets.penalties_kappa_eta[0],
        &jets.penalties_eta2[0],
    ];
    if smooth_penalty.dim() != (p, p)
        || smooth_design_blocks.iter().any(|m| m.dim() != (n, p))
        || smooth_penalty_blocks.iter().any(|m| m.dim() != (p, p))
    {
        crate::bail_invalid_estim!(
            "constant-curvature ψ derivative bundle does not match its value basis"
        );
    }

    // The unpenalized intercept column is ψ-independent, so it contributes zero
    // to every ψ-derivative and its coordinate stays in the penalty null space
    // at all ψ — the ψ-fixed-null-space premise the jet verifies.
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth_design);
    let bordered_design = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, p + 1));
        out.slice_mut(s![.., 1..]).assign(block);
        out
    };
    let bordered_penalty = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p + 1, p + 1));
        out.slice_mut(s![1.., 1..]).assign(block);
        out
    };
    let penalty = bordered_penalty(smooth_penalty);
    let design_blocks: Vec<Array2<f64>> = smooth_design_blocks
        .iter()
        .map(|block| bordered_design(block))
        .collect();
    let penalty_blocks: Vec<Array2<f64>> = smooth_penalty_blocks
        .iter()
        .map(|block| bordered_penalty(block))
        .collect();

    profiled_gaussian_reml_psi_jet(
        &design,
        &penalty,
        &PsiCoordinateBlocks {
            design_first: [&design_blocks[0], &design_blocks[1]],
            design_second: [&design_blocks[2], &design_blocks[3], &design_blocks[4]],
            penalty_first: [&penalty_blocks[0], &penalty_blocks[1]],
            penalty_second: [&penalty_blocks[2], &penalty_blocks[3], &penalty_blocks[4]],
        },
        y,
    )
}

/// The constant-curvature smooth's outer objective in its own two coordinates.
///
/// `ψ = (κ, η)` with `η = ln ℓ`: the signed sectional curvature and the log
/// kernel range. Both move the design and the penalty, both are estimated, and
/// the reason the second one exists is that it is confounded with the first
/// (#2747) — a κ optimized at a pinned ℓ measures the range error, not the
/// curvature.
///
/// This type is the SINGLE owner of the criterion. The point estimate, the
/// profile CI and the flatness LR all read [`Self::evaluate`], the
/// range-profiled κ jet, so they cannot be extrema of different objects.
struct ConstantCurvatureProfile<'a> {
    data: ArrayView2<'a, f64>,
    response: ArrayView1<'a, f64>,
    spec: gam_terms::basis::ConstantCurvatureBasisSpec,
    /// Derived `[ln ℓ_lo, ln ℓ_hi]` evaluability box; `None` when the user
    /// pinned the range, in which case η is not a coordinate at all.
    eta_bounds: Option<(f64, f64)>,
    /// `[ln d_min⁺, ln d_max]` over the pairs the kernel evaluates — where the
    /// inner search BRACKETS, as opposed to where it is walled.
    eta_bracket: (f64, f64),
    /// `η` seed — the auto rule's realized `ℓ_ref`, in logs.
    eta_seed: f64,
    /// Coefficients of the profiled REML fit, `[1 | X]`: one representer per
    /// realized center plus the intercept. Sizes the outer engine's resolution.
    p_coefficients: usize,
    cache: std::cell::RefCell<std::collections::HashMap<(u64, u64), ProfiledRemlPsiJet>>,
    /// Value-only cache for the bracketing scan; see [`Self::evaluate_value`].
    value_cache: std::cell::RefCell<std::collections::HashMap<(u64, u64), f64>>,
}

/// Identity of the profile, WITHOUT the caller's data or the memo tables.
///
/// Written out rather than derived because the two things a derive would print
/// are the two things a reader never wants: `data`/`response` are borrowed
/// views of the caller's whole design and response — a `Result::expect_err` on
/// a 120-row fixture would dump every row into the panic message — and the two
/// memo tables are keyed by the bit patterns of `(κ, η)`, which say nothing
/// about the object and everything about which points a search happened to
/// visit. What identifies a profile is its SPEC and the η geometry derived from
/// it, so that is what this prints; the views and the caches contribute their
/// shapes only.
impl std::fmt::Debug for ConstantCurvatureProfile<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `try_borrow`: a `Debug` reached from inside `evaluate` (a panic while
        // the memo table is mutably borrowed, or a debugger) must not panic a
        // second time on the borrow it cannot get. `None` then means "held", not
        // "empty".
        fn cached<V>(
            c: &std::cell::RefCell<std::collections::HashMap<(u64, u64), V>>,
        ) -> Option<usize> {
            c.try_borrow().map(|t| t.len()).ok()
        }
        f.debug_struct("ConstantCurvatureProfile")
            .field("rows", &self.data.nrows())
            .field("cols", &self.data.ncols())
            .field("response_len", &self.response.len())
            .field("spec", &self.spec)
            .field("eta_bounds", &self.eta_bounds)
            .field("eta_bracket", &self.eta_bracket)
            .field("eta_seed", &self.eta_seed)
            .field("jet_cache_len", &cached(&self.cache))
            .field("value_cache_len", &cached(&self.value_cache))
            .finish()
    }
}

/// How the inner range solve at one κ terminated.
///
/// The variants are not shades of one answer. Each is a different claim about
/// `dη̂/dκ`, and that derivative is what decides which reduction
/// [`ConstantCurvatureProfile::evaluate`] may apply — so a variant that is
/// wrong about it hands the outer solver a gradient that is not the gradient of
/// the value beside it. The claim has teeth: `η̂` moves steeply with κ on real
/// geometry, measured on the coverage fixture's own cloud as `ℓ̂` sweeping
/// `0.68 → 34 000` across the κ box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeSolveOutcome {
    /// `V_ηη > 0` at an η strictly inside the box that the outer engine's
    /// stationarity certificate accepted and did not report as railed.
    /// The profile is read at the Newton minimizer of the jet's quadratic in
    /// η, so the residual `V_η` the certificate leaves costs `V_p′` nothing to
    /// first order (gam#3426, see
    /// [`ProfiledRemlPsiJet::eta_profiled_kappa_jet`]), and the Schur reduction
    /// gives `V_p″`: the term `−V_κη²/V_ηη` is non-positive, so an interior
    /// minimum misfiled as anything else has its profile curvature OVERSTATED,
    /// and that curvature is what the outer solve's terminal stationarity
    /// certificate is denominated in (#2458).
    InteriorMinimum,
    /// The user pinned the range with an explicit `length_scale=`, so η is not
    /// a coordinate at all: `η̂(κ) ≡ η_pinned` and `dη̂/dκ = 0` identically.
    Pinned,
    /// `η̂` is at the BOTTOM of the range chart — the Gram-resolvability wall —
    /// with the criterion still descending toward it. The bound is ACTIVE, so
    /// `η̂(κ) ≡ lo` while it stays active and `dη̂/dκ = 0` there.
    EvaluabilityWall,
    /// `η̂` reached the TOP of the range chart, where the kernel has become the
    /// geodesic-distance kernel to within `√ε` in every design entry
    /// (`constant_curvature_length_scale_bounds`). `dη̂/dκ = 0` for the same
    /// reason as at the bottom wall — the bound is active — but the two are not
    /// the same STATEMENT, and conflating them is the whole of gam#2747 on this
    /// coordinate.
    ///
    /// A wall says the estimator was stopped. This says it ARRIVED: `k → −d_κ`
    /// as `ℓ → ∞`, and `−d_κ` is conditionally positive definite on all three
    /// space forms, so the far face of the range is an ordinary non-degenerate
    /// model rather than a degeneracy. `V(ℓ)` converging monotonically to it is
    /// therefore an answer — "the range is at or beyond the point where the
    /// kernel IS the geodesic distance" — and not the "readout of the box"
    /// `20bde053f` reverted the free-range enrollment over. Nothing past the top
    /// is a different model, so nothing past it is worth searching, and the
    /// stopping rule that comment asked for is a consequence of the chart rather
    /// than a rule.
    ///
    /// Declared rather than inferable, exactly as `146f9232d` made
    /// `KappaEstimateSupport` for the curvature coordinate: a consumer that
    /// reads `ℓ̂` alone cannot tell an arrival from a truncation, and the two
    /// support very different claims about the magnitude.
    DistanceKernelLimit,
    /// The inner solve neither certified an interior stationary point nor
    /// reached a face of the chart, or it certified one whose η-curvature is
    /// non-positive — a maximum or a saddle in η, not the minimum the profile
    /// is defined as.
    ///
    /// **There is no η̂ FUNCTION here, so there is no `dη̂/dκ` to assume.** The
    /// three variants above each earn `dη̂/dκ = 0` from a theorem: a pin makes η
    /// constant by construction, and an active bound makes it constant while the
    /// bound stays active. This one earns nothing — the η it returns is a point
    /// of a trajectory that stopped, and the next κ's trajectory stops somewhere
    /// else. Taking the plain κ slice there reports `V_κ` as the derivative of
    /// `V(κ, η̂(κ))`, which is wrong by exactly `V_η·η̂′` — and `V_η` is not small,
    /// because not being small is what "uncertified" means.
    ///
    /// So [`ConstantCurvatureProfile::evaluate`] REFUSES on this variant rather
    /// than substituting a derivative, the same way
    /// [`ProfiledRemlPsiJet::eta_profiled_kappa_jet`] refuses a non-positive
    /// `V_ηη` instead of dividing by it. It reports as `LocallyFixed` on the
    /// user-facing [`gam_geometry::curvature_estimand::RangeEstimateSupport`],
    /// whose contract already covers "otherwise not certified as an interior
    /// minimizer".
    Uncertified,
}

impl RangeSolveOutcome {
    /// The published provenance of `ℓ̂`, for the report surfaces.
    fn support(self) -> gam_geometry::curvature_estimand::RangeEstimateSupport {
        use gam_geometry::curvature_estimand::RangeEstimateSupport;
        match self {
            Self::InteriorMinimum => RangeEstimateSupport::Interior,
            Self::DistanceKernelLimit => RangeEstimateSupport::DistanceKernelLimit,
            // Three internal states share one published one, and that is the
            // published enum's own contract: "pinned by an explicit
            // `length_scale=`, parked at the evaluability wall, or otherwise not
            // certified as an interior minimizer". They are distinguished HERE
            // because they make different claims about `dη̂/dκ`, which is a
            // solver question rather than a reporting one.
            Self::Pinned | Self::EvaluabilityWall | Self::Uncertified => {
                RangeEstimateSupport::LocallyFixed
            }
        }
    }
}

impl<'a> ConstantCurvatureProfile<'a> {
    /// Construct the curvature-estimation profile in its fit-time constraint
    /// frame.
    ///
    /// A frozen transform is a predict-time replay artifact: it is the global
    /// identifiability frame realized at one particular fitted ψ. Reusing that
    /// fixed frame while this profile varies ψ changes the objective and omits
    /// the frame's ψ derivative. Inference must instead use the same local
    /// center-sum-to-zero quotient that produced the point estimate. Realized
    /// centers remain a valid frozen representation of a deterministic fit-time
    /// choice, so only the ψ-anchored transform is removed.
    fn new(
        data: ArrayView2<'a, f64>,
        response: ArrayView1<'a, f64>,
        mut spec: gam_terms::basis::ConstantCurvatureBasisSpec,
    ) -> Result<Self, EstimationError> {
        if response.len() != data.nrows() || response.is_empty() {
            crate::bail_invalid_estim!(
                "constant-curvature profile needs one non-empty response per row: data={}, response={}",
                data.nrows(),
                response.len(),
            );
        }
        // A `double_penalty=` term is not a model this profile can score, and
        // saying so is the difference between an estimate and a number.
        //
        // The criterion is `gaussian_reml_multi_closed_form` on ONE design and
        // ONE penalty, so it carries one λ. `double_penalty = true` makes
        // `build_constant_curvature_basis` emit TWO active penalties — the RKHS
        // Gram and a ridge `I` — which the fit gives two independent smoothing
        // parameters. Both profile entry points therefore forced the flag off,
        // and the effect of that was silent: κ̂ and ℓ̂ were selected against the
        // one-penalty model and the fit then realized the two-penalty one, so
        // the reported curvature was an estimate for a model nobody fits, with
        // a CI and a flatness p-value to match.
        //
        // The Matérn sibling makes the same assignment (`spatial_optimization.rs`,
        // "Honoring `double_penalty: true` instead returned the kernel-Gram
        // double-penalty ψ-derivatives — a penalty the design does NOT carry"),
        // and there it is CORRECT because the term-collection assembler
        // overrides the basis-level penalty with the operator triplet anyway, so
        // `false` reproduces the realized design exactly (verified to ~1e-9 by
        // FD). No such override exists here: what the basis emits is what the
        // fit penalizes, so dropping the ridge drops a penalty the fit carries.
        //
        // Refusing rather than honoring is deliberate. Honoring it means a
        // two-λ profile, and #1464 measured what the ridge does to this
        // estimand — "the curvature-blind ridge `I` absorbs the data fit
        // independently of κ and rails the fitted curvature to the +chart bound
        // (hyperbolic truth recovered as spherical)" — which is why `curv`
        // defaults to no ridge and only an EXPLICIT `double_penalty=` turns it
        // on. A user who set it has asked for a model whose curvature this
        // machinery cannot estimate, and the two ways out are both one edit:
        // drop `double_penalty=`, or pin `kappa=` and take fixed geometry.
        if spec.double_penalty {
            crate::bail_invalid_estim!(
                "constant-curvature curvature/range estimation is unavailable for a \
                 `double_penalty=` term: the profile criterion carries ONE smoothing \
                 parameter and this basis emits two penalties (RKHS Gram + ridge), so a κ̂ \
                 selected here would be an estimate for a model the fit does not realize. \
                 Either drop `double_penalty=` (the default, and what #1464 recommends — \
                 the κ-blind ridge absorbs the data fit and rails κ̂ to the +chart bound), \
                 or pin `kappa=` and `length_scale=` and take fixed geometry."
            );
        }
        spec.identifiability = gam_terms::basis::ConstantCurvatureIdentifiability::CenterSumToZero;
        // Box, bracket and seed are all read from the REALIZED center set — the
        // one the basis builder itself will use — and in the κ = 0 chart gauge,
        // so all three are κ-FIXED and none of them moves while the optimizer
        // walks κ.
        //
        // They are DERIVED, not configured, and deliberately do not consult
        // `SpatialLengthScaleOptimizationOptions`: the κ box beside them is
        // derived the same way (the half-margin to the antipodal fold), and the
        // curvature-inference entry point has no access to those options at
        // all. A box visible to the fit but not to the profile CI would put the
        // point estimate and its interval on two different parameter spaces.
        let centers = gam_terms::basis::constant_curvature_realized_centers(data, &spec)
            .map_err(EstimationError::from)?;
        // The SEED is derived too, and that takes one line of work rather than
        // none (gam#2747).
        //
        // `realized_constant_curvature_length_scale` returns an explicit
        // positive `length_scale` VERBATIM and falls back to the derived median
        // only on the `0.0` auto sentinel — and by the time this profile is
        // built for inference, `spec.length_scale` is no longer a request. Both
        // of the fit's write-backs have overwritten it with `ℓ̂`: the free-κ arm
        // in `spatial_optimization.rs` (`cc.length_scale = psi_hat.length_scale`)
        // and `freeze_term_collection_from_design` (`s.length_scale =
        // *length_scale` off `BasisMetadata::ConstantCurvature`).
        //
        // `ℓ̂` is the range this criterion profiled to AT κ̂. Seeding the inner
        // solve with it is a warm start from ONE κ, and
        // [`Self::minimize_over_eta`] states why that is not allowed: a `V_p`
        // that depends on where the search has already been is not a function of
        // its own argument, and the CI walk and the flatness LR both compare
        // values of `V_p(κ)` across κ. The point estimate would then be the
        // argmin of one object and the interval a level set of another.
        //
        // This is the same argument the line above makes about
        // `identifiability`, and it has the same answer. A fitted range is a
        // realized artifact of one particular fitted ψ, exactly as a frozen
        // constraint transform is, so the profile un-freezes both. A USER pin is
        // different in kind — it is a request, not an artifact — and it is
        // honored: `length_scale_fixed` takes η out of the coordinate set
        // entirely, and then the pinned value is the only η there is.
        let seed_request = if spec.length_scale_fixed {
            spec.length_scale
        } else {
            0.0
        };
        let ell_seed = gam_terms::basis::realized_constant_curvature_length_scale(
            centers.view(),
            seed_request,
        )
        .map_err(EstimationError::from)?;
        let (span_lo, span_hi) =
            gam_terms::basis::constant_curvature_evaluated_scale_span(data, centers.view())
                .map_err(EstimationError::from)?;
        let eta_bounds = if spec.length_scale_fixed {
            None
        } else {
            let (lo, hi) =
                gam_terms::basis::constant_curvature_length_scale_bounds(data, centers.view())
                    .map_err(EstimationError::from)?;
            Some((lo.ln(), hi.ln()))
        };
        let eta_seed = match eta_bounds {
            Some((lo, hi)) => ell_seed.ln().clamp(lo, hi),
            None => ell_seed.ln(),
        };
        Ok(Self {
            data,
            response,
            spec,
            eta_bounds,
            eta_bracket: (span_lo.ln(), span_hi.ln()),
            eta_seed,
            p_coefficients: centers.nrows() + 1,
            cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            value_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// A κ-free ceiling on the profile: `V_p(κ) ≤ value_ceiling()` at EVERY κ
    /// and η of the chart (gam#3509).
    ///
    /// At any `(κ, η)` the closed form is, in the penalty's whitened spectrum
    /// `δ_j` (`t_j = e^ρ δ_j`, rank `r`, `dp(ρ) = r0 + Σ_j c_j² t_j/(1+t_j)`),
    ///
    /// ```text
    ///   V(ρ) = ½[log|XᵀX| + Σ_j log(1+t_j) − log|S|₊ − rρ] + ½ν·(1 + log(2π·dp/ν)) + const.
    /// ```
    ///
    /// The penalty's null space is exactly the ψ-free intercept column (the
    /// kernel Gram is strictly positive definite on the sum-to-zero frame), so
    /// `ν = n − 1` and, as `ρ → ∞`, `log|XᵀX + λS| − log|λS|₊ → log n` and
    /// `dp → Σ(y − ȳ)²`: `V_∞` is the REML value of the intercept-only model, the
    /// same number at every ψ. The finite-ρ excess is
    /// `½Σ_j log(1 + 1/t_j) + ½ν·log(dp(ρ)/dp_∞)`, whose second part is `≤ 0`
    /// (`dp` rises to `dp_∞`). At the upper face of the resolvability domain,
    /// `ρ_up = −log √ε − log δ_min`, every `1/t_j ≤ √ε·δ_min/δ_j ≤ √ε`, so
    /// `V(ρ_up) ≤ V_∞ + ½r·log(1 + √ε)`. The ρ selector evaluates that face and
    /// returns a value no larger, and the η profile then takes a minimum over
    /// such values, so the bound holds for `V_p(κ)` itself. `r ≤ p_coefficients − 1`.
    fn value_ceiling(&self) -> Result<f64, EstimationError> {
        let n = self.response.len();
        let intercept = Array2::<f64>::ones((n, 1));
        let no_penalty = Array2::<f64>::zeros((1, 1));
        let response_2d = self.response.insert_axis(ndarray::Axis(1));
        let smooth_absent = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
            intercept.view(),
            response_2d.view(),
            no_penalty.view(),
            None,
            None,
        )?;
        let penalized_modes = self.p_coefficients.saturating_sub(1) as f64;
        let face_excess_per_mode = gam_solve::estimate::rho_domain::log_gradient_resolution()
            .exp()
            .ln_1p();
        Ok(smooth_absent.reml_score + 0.5 * penalized_modes * face_excess_per_mode)
    }

    /// The profile VALUE at one point of the plane, without derivative blocks.
    ///
    /// Shares the jet cache: a point already evaluated at full order answers
    /// from there, so the bracket never re-pays for a point the Newton has
    /// visited and vice versa.
    ///
    /// **A ρ̂ on a face of its domain is a λ-profile value, and is compared like
    /// one** (gam#3509). The domain is not a box: it is the #2812 resolvability
    /// interval `[ln(√ε·γ_min), ln(γ_max/√ε)]` of the pair's own penalty
    /// spectrum (`gam_solve::estimate::rho_domain`). In each penalized direction
    /// the ρ-gradient of `V` is carried by `γ_j/(γ_j + λ)` (and, in the
    /// deviance, by the same factor times that direction's squared score), so
    /// past the upper face every direction contributes `≤ γ_j/λ ≤ √ε·e^{ρ_up−ρ}`
    /// and the whole tail `∫_{ρ_up}^{∞} |V_ρ| dρ` is bounded by the same `√ε`
    /// per direction — the value on the face IS `lim_{λ→∞} V`, the term switched
    /// off, to the gradient's own resolution. The lower face is the mirror image
    /// (`λ/γ_j ≤ √ε`: the term unpenalized). So a railed value is not "a
    /// truncated minimum": it is the infimum along the ρ ray, which is exactly
    /// what the λ-profile is, and it is the SAME model value at every ψ the rail
    /// is reached from (at λ = ∞ only the ψ-free intercept survives). Refusing it
    /// made the profile undefined on the one dataset where that limit is the
    /// truth — constant mean plus noise, `κ⋆ = 0` — and turned the flatness test
    /// into an error instead of a flat profile and `LR ≈ 0`.
    ///
    /// The derivative side already agrees: `profiled_gaussian_reml_psi_jet`
    /// drops the `V_ρψ²/V_ρρ` Schur term on a face, because `dρ̂/dψ = 0` while
    /// the face is active (envelope theorem at an active bound).
    fn evaluate_value(&self, kappa: f64, eta: f64) -> Result<f64, EstimationError> {
        if !(kappa.is_finite() && eta.is_finite()) {
            crate::bail_invalid_estim!(
                "constant-curvature profile probed a non-finite ψ = ({kappa}, {eta})"
            );
        }
        let key = (kappa.to_bits(), eta.to_bits());
        if let Some(cached) = self.cache.borrow().get(&key) {
            return Ok(cached.value);
        }
        if let Some(&cached) = self.value_cache.borrow().get(&key) {
            return Ok(cached);
        }
        let mut probe_spec = self.spec.clone();
        probe_spec.kappa = kappa;
        probe_spec.length_scale = eta.exp();
        let value = constant_curvature_psi_profile_value(self.data, self.response, &probe_spec)?;
        self.value_cache.borrow_mut().insert(key, value);
        Ok(value)
    }

    /// The full `(κ, η)` jet at one point of the plane.
    fn evaluate_psi(&self, kappa: f64, eta: f64) -> Result<ProfiledRemlPsiJet, EstimationError> {
        if !(kappa.is_finite() && eta.is_finite()) {
            crate::bail_invalid_estim!(
                "constant-curvature profile probed a non-finite ψ = ({kappa}, {eta})"
            );
        }
        let key = (kappa.to_bits(), eta.to_bits());
        if let Some(cached) = self.cache.borrow().get(&key) {
            return Ok(cached.clone());
        }
        let mut probe_spec = self.spec.clone();
        probe_spec.kappa = kappa;
        probe_spec.length_scale = eta.exp();
        let sample = constant_curvature_psi_profile_jet(self.data, self.response, &probe_spec)?;
        self.cache.borrow_mut().insert(key, sample.clone());
        Ok(sample)
    }

    /// The profiled range `η̂(κ) = argmin_η V(κ, η)` on the evaluability domain,
    /// found by the workspace's outer engine: one coordinate carrying the
    /// analytic ψ-gradient and curvature the jet already provides, the engine's
    /// seed cascade and stationarity certificate, and the domain `[lo, hi]`
    /// derived at construction (the distance-kernel limit above, the
    /// evaluability wall below). The certificate's railed coordinate names a
    /// wall outcome; a certified interior point with positive curvature is an
    /// interior minimum; a point the certificate cannot vouch for is
    /// `Uncertified`, and a search the engine could not certify at all is an
    /// error. This replaced a 13-point scan of the bracket seeding a Newton
    /// with a `√ε` resolution, a `1e-9` relative stationarity test, a
    /// quarter-width fallback step and two hand budgets that returned its last
    /// iterate (#2469, #2670: SPEC forbids grid search and hand bounds).
    fn minimize_over_eta(
        &self,
        kappa: f64,
    ) -> Result<(f64, ProfiledRemlPsiJet, RangeSolveOutcome), EstimationError> {
        let Some((lo, hi)) = self.eta_bounds else {
            let jet = self.evaluate_psi(kappa, self.eta_seed)?;
            return Ok((self.eta_seed, jet, RangeSolveOutcome::Pinned));
        };
        use gam_problem::{Derivative, HessianValue, OuterEval};
        use gam_solve::rho_optimizer::OuterProblem;
        let context = format!("constant-curvature range solve at κ = {kappa}");
        // A kernel that cannot be evaluated at a trial η is a property of that
        // trial, so the search retreats from it instead of abandoning the
        // profile.
        let refuse = |error: EstimationError| EstimationError::TrialPointRefused {
            reason: error.to_string(),
        };
        let problem = OuterProblem::new(1)
            .with_problem_size(self.response.len(), self.p_coefficients)
            .with_gradient(Derivative::Analytic)
            .with_hessian(gam_problem::DeclaredHessianForm::Dense)
            .with_bounds(Array1::from_vec(vec![lo]), Array1::from_vec(vec![hi]))
            .with_initial_rho(Array1::from_vec(vec![self.eta_seed.clamp(lo, hi)]));
        let mut objective = problem.build_objective(
            (),
            |_: &mut (), rho: &Array1<f64>| self.evaluate_value(kappa, rho[0]).map_err(refuse),
            |_: &mut (), rho: &Array1<f64>| {
                let jet = self.evaluate_psi(kappa, rho[0]).map_err(refuse)?;
                Ok(OuterEval {
                    cost: jet.value,
                    gradient: Array1::from_vec(vec![jet.gradient[1]]),
                    hessian: HessianValue::Dense(Array2::from_elem((1, 1), jet.hessian[1][1])),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError>>,
        );
        let result = problem.run(&mut objective, &context)?;
        let eta = result.rho[0];
        let jet = self.evaluate_psi(kappa, eta)?;
        let certificate = result.criterion_certificate.as_ref();
        // A railed coordinate sits on one of the two walls; which one is read
        // off the fact's own box, not off a resolution constant.
        let railed_at_upper = certificate.is_some_and(|c| {
            c.railed_facts
                .iter()
                .any(|fact| fact.theta > 0.5 * (fact.lower + fact.upper))
        });
        let railed_at_lower = certificate.is_some_and(|c| {
            c.railed_facts
                .iter()
                .any(|fact| fact.theta <= 0.5 * (fact.lower + fact.upper))
        });
        let curvature = jet.hessian[1][1];
        let outcome = if railed_at_upper {
            RangeSolveOutcome::DistanceKernelLimit
        } else if railed_at_lower {
            RangeSolveOutcome::EvaluabilityWall
        } else if certificate.and_then(|c| c.hessian_psd()) != Some(false)
            && curvature.is_finite()
            && curvature > 0.0
        {
            RangeSolveOutcome::InteriorMinimum
        } else {
            RangeSolveOutcome::Uncertified
        };
        // A certified η̂ whose slope still points at a face is not the minimum over
        // the derived range when that face is no higher: the profile is then the
        // face's value, reached (#1464). On the κ*=−2 fixture the solve stopped at
        // ln ℓ = 14.6 with V_η/V_ηη = −1 — the exponential approach to the
        // distance-kernel face at ln ℓ = 18.3 — and reported an interior minimum at
        // ℓ = 2.3e6, whose value was the face's to 1.8e-4. One value evaluation at
        // the face decides it, with nothing to tune.
        if outcome == RangeSolveOutcome::InteriorMinimum {
            let toward = if jet.gradient[1] < 0.0 {
                Some((hi, RangeSolveOutcome::DistanceKernelLimit))
            } else if jet.gradient[1] > 0.0 {
                Some((lo, RangeSolveOutcome::EvaluabilityWall))
            } else {
                None
            };
            if let Some((face, face_outcome)) = toward
                && self.evaluate_value(kappa, face)? <= jet.value
            {
                return Ok((face, self.evaluate_psi(kappa, face)?, face_outcome));
            }
        }
        Ok((eta, jet, outcome))
    }

    /// `(V_p(κ), V_p′(κ), V_p″(κ))` with the range PROFILED out — the
    /// one-dimensional likelihood the point estimate, the CI and the flatness
    /// test all consume.
    ///
    /// At a certified interior η̂ the profile is read at the Newton minimizer of
    /// the jet's quadratic in η: `V_p′ = V_κ − V_κη·V_η/V_ηη`, the envelope
    /// theorem with the certificate's residual `V_η` carried rather than
    /// dropped, and the Schur complement `V_p″ = V_κκ − V_κη²/V_ηη`.
    ///
    /// Otherwise the reduction is absent and what replaces it depends on WHY,
    /// which is the whole content of [`RangeSolveOutcome`]. Where `dη̂/dκ = 0` is
    /// a theorem — η pinned, or a chart bound active — the plain κ slice IS the
    /// total derivative and is returned. Where it is not a theorem, this
    /// refuses. It used to return the plain slice there too, on the stated
    /// premise that "η̂ is locally constant in κ", and that premise is false at a
    /// stalled iterate: `ℓ̂` sweeps `0.68 → 34 000` across the κ box on the
    /// coverage fixture's own geometry, so `η̂′` is order tens per unit κ and
    /// `V_κ` is short of `dV/dκ` by `V_η·η̂′` — with `V_η` not small, because not
    /// being small is exactly what failing the certificate means. The value was
    /// right and the gradient was not, which is the shape of desync that costs
    /// the most to find.
    fn evaluate(&self, kappa: f64) -> Result<(f64, f64, f64), EstimationError> {
        let (eta, jet, outcome) = self.minimize_over_eta(kappa)?;
        match outcome {
            RangeSolveOutcome::InteriorMinimum => jet.eta_profiled_kappa_jet(),
            // `dη̂/dκ = 0` is a theorem on all three of these — η is not a
            // coordinate, or the bound it sits on is active — so the total
            // derivative of `V(κ, η̂(κ))` IS the partial `V_κ`.
            RangeSolveOutcome::Pinned
            | RangeSolveOutcome::EvaluabilityWall
            | RangeSolveOutcome::DistanceKernelLimit => Ok(jet.kappa_slice()),
            // And it is a theorem on none of the rest. See
            // `RangeSolveOutcome::Uncertified`: the returned η is a point of a
            // trajectory that stopped, so `V_κ` is wrong by `V_η·η̂′` with both
            // factors unknown and `V_η` demonstrably not small.
            RangeSolveOutcome::Uncertified => {
                crate::bail_invalid_estim!(
                    "constant-curvature profile has no derivative at κ = {kappa}: the inner \
                     range solve stopped at ln ℓ = {eta} without certifying a stationary point \
                     (V_η = {:.6e}, V_ηη = {:.6e}), so η̂ is not a differentiable function of κ \
                     there and neither the envelope reduction nor the plain κ slice is V_p′",
                    jet.gradient[1],
                    jet.hessian[1][1],
                )
            }
        }
    }
}

/// `ℓ̂` at a PINNED κ — the range half of the profile, run on its own.
///
/// A pinned `kappa=` fixes the geometry (gam#2152) and takes the term out of the
/// curvature search. It does not fix the RANGE, and the two were coupled here
/// only because one function owned both: `20bde053f` reverted the pinned-κ /
/// free-range enrollment because the range criterion "is monotone in ell all
/// the way to its asymptote … a readout of the box rather than of the data".
/// That reading was correct about the symptom and wrong about the cause — past
/// `ℓ ≈ 10⁶` the old kernel gauge's criterion was fabricated, descending ~100
/// nats per decade into its own cancellation — and both halves are fixed:
/// the criterion is now a function of the data across the whole chart, and the
/// chart's top is the geodesic-distance face, an arrival the solve DECLARES
/// (see [`RangeSolveOutcome::DistanceKernelLimit`]).
///
/// So the range is estimated whenever the user did not pin it, at whatever κ the
/// term carries. This runs the same inner solve
/// [`ConstantCurvatureProfile::minimize_over_eta`] the full profile runs at each
/// trial κ — one owner, one objective — rather than a second range search with
/// its own bracket.
fn constant_curvature_range_only_optimum(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
) -> Result<f64, EstimationError> {
    let (feature_cols, base_spec) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|term| &term.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => {
            crate::bail_invalid_estim!(
                "constant-curvature range optimum requested for non-curvature term {term_idx}"
            )
        }
    };
    let pinned_kappa = base_spec.kappa;
    let x_term = select_columns(data, feature_cols).map_err(EstimationError::from)?;
    let profile = ConstantCurvatureProfile::new(x_term.view(), y, base_spec)?;
    let (eta_hat, _, outcome) = profile.minimize_over_eta(pinned_kappa)?;
    let length_scale_hat = eta_hat.exp();
    log::debug!(
        "[spatial-kappa] pinned kappa={pinned_kappa:.6}: range profiled to \
         length_scale_hat={length_scale_hat:.6} ({outcome:?}) for term {term_idx}",
    );
    Ok(length_scale_hat)
}

/// The preconditions under which the profile criterion IS the fitted model's
/// criterion for the curvature and range of `curv(...)` term `term_idx`.
///
/// The profile's design is `[1 | curv block]` and nothing else (see
/// [`constant_curvature_psi_profile_value`]). A κ̂ and ℓ̂ selected on it, and
/// the CI and flatness p-value read off it, describe the fitted model only
/// when that model is exactly the intercept plus this one term. Any other
/// parametric, random-effect or smooth term (or a removed intercept) makes
/// them estimates for a model the fit does not realize, which is the same
/// reason `ConstantCurvatureProfile::new` refuses `double_penalty=` (gam#3763).
fn validate_constant_curvature_profile_inputs(
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: &LikelihoodSpec,
) -> Result<(), EstimationError> {
    let sole_curvature_term = resolvedspec.linear_terms.is_empty()
        && resolvedspec.random_effect_terms.is_empty()
        && resolvedspec.smooth_terms.len() == 1
        && term_idx == 0
        && matches!(resolvedspec.level, gam_terms::smooth::ModelLevel::Intercept);
    if !sole_curvature_term {
        crate::bail_invalid_estim!(
            "curvature-as-an-estimand profile for term {term_idx} requires the model to be \
             exactly `y ~ curv(...)` (intercept plus this one term): its criterion carries \
             only the intercept and the curvature block, so with {} linear, {} random-effect \
             and {} smooth terms (level {:?}) the κ̂, ℓ̂, CI and flatness p-value would \
             describe a model the fit does not realize. Pin `kappa=` and `length_scale=` \
             to take fixed geometry inside a larger model.",
            resolvedspec.linear_terms.len(),
            resolvedspec.random_effect_terms.len(),
            resolvedspec.smooth_terms.len(),
            resolvedspec.level,
        );
    }
    if *family != LikelihoodSpec::gaussian_identity() {
        crate::bail_invalid_estim!(
            "curvature-as-an-estimand profile currently requires Gaussian identity likelihood"
        );
    }
    let input_tolerance = f64::EPSILON.sqrt();
    if weights
        .iter()
        .any(|&weight| (weight - 1.0).abs() > input_tolerance)
        || offset.iter().any(|&value| value.abs() > input_tolerance)
    {
        crate::bail_invalid_estim!(
            "curvature-as-an-estimand profile requires unit weights and zero offset"
        );
    }
    Ok(())
}

/// The constant-curvature smooth's fitted outer coordinates.
#[derive(Clone, Copy, Debug)]
struct ConstantCurvatureOptimum {
    /// Signed sectional curvature κ̂.
    kappa: f64,
    /// Kernel range ℓ̂ = exp(η̂(κ̂)) — the range the criterion profiles to at the
    /// fitted curvature. Equals the pinned value when the user set
    /// `length_scale=`.
    length_scale: f64,
}

/// The κ-profile outer problem: one coordinate on the chart-valid interval,
/// sized by the `n_obs` rows the evidence sums, held to the fit's own `tol`
/// exactly as every other outer route is. The exact `d²V_p/dκ²` and the
/// declared size are what let the certificate resolve κ̂; the tolerance is not
/// widened on this route's behalf.
fn constant_curvature_kappa_problem(
    n_obs: usize,
    profile: &ConstantCurvatureProfile<'_>,
    options: &FitOptions,
    kappa_min: f64,
    kappa_max: f64,
) -> gam_solve::rho_optimizer::OuterProblem {
    let initial_kappa = profile.spec.kappa.clamp(kappa_min, kappa_max);
    gam_solve::rho_optimizer::OuterProblem::new(1)
        .with_problem_size(n_obs, profile.p_coefficients)
        .with_gradient(gam_problem::Derivative::Analytic)
        // #2458: the κ profile supplies an EXACT d²V_p/dκ², so this route runs
        // the same curvature-denominated stationarity certificate every other
        // route runs. It previously declared `Unavailable` — not because the
        // curvature was unavailable, but because this call site never asked the
        // basis bundle for the seconds it already ships.
        .with_hessian(gam_problem::DeclaredHessianForm::Dense)
        // #3201: the search runs ARC on that curvature. Every derivative-bearing
        // evaluation of the profile forms `V_p″` with its gradient, so a
        // gradient-only search would discard curvature it has already paid for,
        // and a line search cannot resolve ‖g‖ below `2√(L·ε_f)`, where ARC on
        // the exact second derivative can.
        .with_disable_fixed_point(true)
        .with_fallback_policy(gam_solve::rho_optimizer::FallbackPolicy::Disabled)
        .with_psi_dim(1)
        .with_tolerance(options.tol)
        .with_bounds(
            Array1::from_vec(vec![kappa_min]),
            Array1::from_vec(vec![kappa_max]),
        )
        .with_initial_rho(Array1::from_vec(vec![initial_kappa]))
}

/// Minimize the RANGE-PROFILED, continuously smoothing-profiled Gaussian REML
/// evidence `V_p(κ) = min_{η,ρ} V(κ, η, ρ)` on the chart-valid κ interval, with
/// the shared bounded analytic outer solver — so every accepted result has
/// passed the solver's final box-KKT projected-gradient certificate. No sampled
/// point is ever returned as the estimate: samples are only line-search probes
/// for the continuous solve.
///
/// # Why the range is profiled rather than searched jointly
///
/// The range has to be estimated at all because it is confounded with the
/// curvature (#2747): the two enter `exp(−d_κ/ℓ)` through one exponent, so a κ
/// optimized against a pinned ℓ reports the range error rather than the
/// curvature — measured, it rails, inverts the sign, or invents curvature from
/// flat data.
///
/// But it must be profiled, not co-searched, because **the point estimate and
/// the interval have to be extrema of the SAME object**. A joint search over
/// `(κ, η)` returns a local stationary point of `V(κ, η)`, while the profile CI
/// and the flatness LR compare values of `V_p(κ) = min_η V(κ, η)`; where the
/// two disagree the reported κ̂ is not the argmin of its own interval's
/// criterion. This file already carries the scar from the last time one
/// coordinate had two objective owners — see the `spatial_terms` filter, which
/// exists because that "made the scalar and joint routes disagree at the
/// identical seed on flat data". So there is one owner: `ConstantCurvatureProfile`,
/// whose inner range solve is deterministic and globally bracketed, and whose
/// κ jet is the exact envelope/Schur reduction of it.
///
/// A user who pins `length_scale=` gets the same one-dimensional κ search at
/// that range, exactly as a user who pins `kappa=` gets fixed geometry.
fn constant_curvature_kappa_profile_optimum(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    options: &FitOptions,
) -> Result<ConstantCurvatureOptimum, EstimationError> {
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);
    if !(kappa_min.is_finite() && kappa_max.is_finite() && kappa_max > kappa_min) {
        crate::bail_invalid_estim!(
            "constant-curvature term {term_idx} has invalid kappa bounds [{kappa_min}, {kappa_max}]"
        );
    }
    let (feature_cols, base_spec) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|term| &term.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => {
            crate::bail_invalid_estim!(
                "constant-curvature optimum requested for non-curvature term {term_idx}"
            )
        }
    };
    let x_term = select_columns(data, feature_cols).map_err(EstimationError::from)?;
    let profile = ConstantCurvatureProfile::new(x_term.view(), y, base_spec)?;
    solve_constant_curvature_kappa_profile(
        y.len(),
        profile,
        options,
        kappa_min,
        kappa_max,
        term_idx,
    )
}

/// The κ solve of [`constant_curvature_kappa_profile_optimum`] on an already
/// constructed profile over `n_obs` rows.
fn solve_constant_curvature_kappa_profile(
    n_obs: usize,
    profile: ConstantCurvatureProfile<'_>,
    options: &FitOptions,
    kappa_min: f64,
    kappa_max: f64,
    term_idx: usize,
) -> Result<ConstantCurvatureOptimum, EstimationError> {
    let problem = constant_curvature_kappa_problem(n_obs, &profile, options, kappa_min, kappa_max);
    let mut objective = problem.build_objective(
        profile,
        |profile: &mut ConstantCurvatureProfile<'_>, theta: &Array1<f64>| {
            profile.evaluate(theta[0]).map(|(value, _, _)| value)
        },
        |profile: &mut ConstantCurvatureProfile<'_>, theta: &Array1<f64>| {
            let (cost, derivative, curvature) = profile.evaluate(theta[0])?;
            Ok(gam_problem::OuterEval {
                cost,
                gradient: Array1::from_vec(vec![derivative]),
                hessian: gam_problem::HessianValue::Dense(
                    Array2::from_shape_vec((1, 1), vec![curvature]).expect("1x1 from one element"),
                ),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ConstantCurvatureProfile<'_>)>,
        None::<
            fn(
                &mut ConstantCurvatureProfile<'_>,
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, EstimationError>,
        >,
    );
    let result = problem.run(
        &mut objective,
        &format!("constant-curvature likelihood profile term {term_idx}"),
    )?;
    if !result.converged() {
        crate::bail_invalid_estim!(
            "constant-curvature likelihood-profile κ optimization did not converge for term {} after {} iterations (negative_log_evidence={:.6e}, final_grad_norm={})",
            term_idx,
            result.iterations,
            result.final_value,
            result.final_grad_norm_report(),
        );
    }
    let kappa_hat = result.rho[0];
    // Read ℓ̂ off the SAME profile object the solve just used, so the reported
    // range is the one the accepted κ̂ was profiled against (and replays from its
    // cache rather than re-solving).
    let (eta_hat, _, range_outcome) = objective.state.minimize_over_eta(kappa_hat)?;
    let length_scale_hat = eta_hat.exp();
    // The range's support, said rather than left to be read off the magnitude
    // (gam#2747). `DistanceKernelLimit` is not a rail: the kernel has become
    // `−d_κ`, which is the model, so `ℓ̂` there is a lower bound with a meaning
    // and not a readout of a box.
    let range_support = match range_outcome {
        RangeSolveOutcome::InteriorMinimum => "interior",
        RangeSolveOutcome::DistanceKernelLimit => "at the geodesic-distance limit",
        RangeSolveOutcome::Pinned => "pinned by the caller",
        RangeSolveOutcome::EvaluabilityWall => "at the evaluability wall",
        RangeSolveOutcome::Uncertified => "UNCERTIFIED (no stationarity claim)",
    };
    log::debug!(
        "[spatial-kappa] continuous likelihood-profile optimum kappa_hat={:.6} \
         length_scale_hat={:.6} ({range_support}) \
         (negative_log_evidence={:.6e}, projected_gradient={}) for term {term_idx}",
        kappa_hat,
        length_scale_hat,
        result.final_value,
        result.final_grad_norm_report(),
    );
    Ok(ConstantCurvatureOptimum {
        kappa: kappa_hat,
        length_scale: length_scale_hat,
    })
}

#[cfg(test)]
mod profile_model_contract_tests {
    use super::*;
    use crate::fit_orchestration::{FitConfig,FitRequest,materialize};

    #[test]
    fn curvature_and_range_profiles_require_the_actual_model() {
        let headers=["x","z","y"].into_iter().map(String::from).collect();
        let rows=(0..80).map(|i| {
            let x=0.25*(i as f64*0.71).sin();
            let z=0.25*(i as f64*0.53).cos();
            csv::StringRecord::from(vec![x.to_string(),z.to_string(),(x*z+0.1*z).to_string()])
        }).collect();
        let data=gam_data::encode_recordswith_inferred_schema(headers,rows).unwrap();
        let config=FitConfig {family:Some("gaussian".into()),..FitConfig::default()};
        for (formula,accepted) in [
            ("y ~ curv(x, z, centers=20)",true),
            ("y ~ x + curv(x, z, centers=20)",false),
            ("y ~ x + curv(x, z, kappa=1, centers=20)",false),
            ("y ~ 0 + curv(x, z, kappa=1, centers=20)",false),
        ] {
            let FitRequest::Standard(request)=materialize(formula,&data,&config).unwrap().request else {panic!("standard request")};
            let verdict=validate_constant_curvature_profile_inputs(&request.spec,0,request.weights.view(),request.offset.view(),&request.family);
            if accepted {verdict.unwrap();} else {assert!(verdict.unwrap_err().to_string().contains("exactly `y ~ curv(...)`"),"{formula}");}
        }
    }
}

/// κ recovery on draws from the model's own family (#1464).
///
/// `curv()` estimates the κ of the kernel family it fits, `exp(−d_κ/ℓ)` closed by
/// its derived faces (the Gram-resolvability wall and the distance-kernel face
/// `−d_κ`, #2747). A contract can only ask it to recover κ from data that family
/// generates, so the fixtures here are combinations of the family's own kernel at
/// κ* = ±2, ℓ* = 1, at the term's realized centres, over the #1464 disc (radius
/// 0.68, 600 points, noise 0.02). The old generator `2e^{−d_hyp} − 1` is kept as a documented case: it is
/// not a draw from the family, and within the family the spherical distance kernel
/// explains it better than any hyperbolic fit, which is the #2747 behaviour working
/// as designed and is pinned as such.
#[cfg(test)]
mod kappa_recovery_1464_tests {
    use super::*;
    use crate::fit_orchestration::request::StandardFitRequest;
    use crate::fit_orchestration::{FitConfig, FitRequest, materialize};
    use gam_geometry::manifolds::ConstantCurvature;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal, Uniform};

    const RADIUS: f64 = 0.68;
    const ROWS: usize = 600;
    const NOISE: f64 = 0.02;

    /// Uniform points on the disc of radius [`RADIUS`], each with its noise draw.
    fn disc_sample() -> Vec<([f64; 2], f64)> {
        let mut rng = StdRng::seed_from_u64(1);
        let unit = Uniform::new(-1.0_f64, 1.0).expect("uniform");
        let noise = Normal::new(0.0, NOISE).expect("normal");
        let mut sample = Vec::with_capacity(ROWS);
        while sample.len() < ROWS {
            let (a, b) = (unit.sample(&mut rng), unit.sample(&mut rng));
            if a * a + b * b > 1.0 {
                continue;
            }
            sample.push(([a * RADIUS, b * RADIUS], noise.sample(&mut rng)));
        }
        sample
    }

    fn encode(rows: impl Iterator<Item = ([f64; 2], f64)>) -> gam_data::EncodedDataset {
        let rows = rows
            .map(|([u, v], y)| csv::StringRecord::from(vec![y.to_string(), u.to_string(), v.to_string()]))
            .collect();
        let headers = ["y", "x1", "x2"].into_iter().map(String::from).collect();
        gam_data::encode_recordswith_inferred_schema(headers, rows).expect("encode")
    }

    /// The disc sample with response `response(x) + noise`.
    fn disc_dataset(response: impl Fn([f64; 2]) -> f64) -> gam_data::EncodedDataset {
        encode(disc_sample().into_iter().map(|(x, noise)| (x, response(x) + noise)))
    }

    /// A draw from the family `curv(x1, x2, centers=10)` fits: the kernel
    /// `exp(−d_κ*(x, c_j)/ℓ*)`, ℓ* = 1, at the term's own realized centres `c_j`,
    /// with coefficients that sum to zero as the term's `CenterSumToZero`
    /// constraint requires, and the geodesic distance the basis evaluates.
    ///
    /// Both halves are what makes it in-family. A combination at centres the
    /// basis does not hold, or with coefficients off the constraint, is not in the
    /// span at any κ, and its best approximation need not be at κ*: with five
    /// hand-placed centres and coefficients summing to 1.5, κ* = −2 was published
    /// at κ̂ = +2.02, and with the sum made zero at κ̂ = +2.13.
    fn kernel_draw(kappa_star: f64) -> gam_data::EncodedDataset {
        const WEIGHTS: [f64; 10] = [2.0, -1.5, 1.2, -1.0, 0.8, -0.6, 0.5, -0.4, 0.3, -1.3];
        let sample = disc_sample();
        // The centres depend on the covariates alone, so any response realizes them.
        let covariates = encode(sample.iter().copied());
        let request = request(&covariates);
        let (feature_cols, spec) = match &request.spec.smooth_terms[0].basis {
            SmoothBasisSpec::ConstantCurvature { feature_cols, spec, .. } => (feature_cols, spec),
            _ => panic!("curv term"),
        };
        let x_term = select_columns(request.data.view(), feature_cols).expect("columns");
        let realized = gam_terms::basis::constant_curvature_realized_centers(x_term.view(), spec)
            .expect("realized centres");
        assert_eq!(realized.nrows(), WEIGHTS.len(), "centers=10 realizes ten centres");
        // The coefficients are assigned in lexicographic centre order, so the draw
        // does not depend on the order the selector lists its centres in.
        let mut centres: Vec<[f64; 2]> = realized.outer_iter().map(|c| [c[0], c[1]]).collect();
        centres.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        let manifold = ConstantCurvature::new(2, kappa_star);
        encode(sample.into_iter().map(|(x, noise)| {
            let point = ndarray::array![x[0], x[1]];
            let signal: f64 = centres
                .iter()
                .zip(WEIGHTS)
                .map(|(centre, weight)| {
                    let d = manifold
                        .distance(point.view(), ndarray::array![centre[0], centre[1]].view())
                        .expect("every point and centre is inside the chart");
                    weight * (-d).exp()
                })
                .sum();
            (x, signal + noise)
        }))
    }

    /// The pre-#1464-ruling generator: a radial response `2e^{−d} − 1` in the
    /// geodesic distance from the origin at κ*. Not a draw from the family.
    fn saturating_radial(kappa_star: f64) -> gam_data::EncodedDataset {
        let root = kappa_star.abs().sqrt();
        disc_dataset(|[u, v]| {
            let r = u.hypot(v);
            let d = if kappa_star < 0.0 {
                2.0 * (root * r).min(1.0 - 1e-9).atanh() / root
            } else if kappa_star > 0.0 {
                2.0 * (root * r).atan() / root
            } else {
                2.0 * r
            };
            2.0 * (-d).exp() - 1.0
        })
    }

    fn request(data: &gam_data::EncodedDataset) -> StandardFitRequest<'_> {
        let config = FitConfig {
            family: Some("gaussian".into()),
            ..FitConfig::default()
        };
        let FitRequest::Standard(request) =
            materialize("y ~ curv(x1, x2, centers=10)", data, &config).expect("materialize").request
        else {
            panic!("standard request");
        };
        request
    }

    fn published(request: &StandardFitRequest<'_>, label: &str) -> ConstantCurvatureOptimum {
        let optimum = constant_curvature_kappa_profile_optimum(
            request.data.view(),
            request.y.view(),
            &request.spec,
            0,
            &request.options,
        )
        .expect("the κ profile certifies an optimum");
        eprintln!(
            "[1464-KAPPA] {label}: kappa_hat={:+.6} range={:.6e}",
            optimum.kappa, optimum.length_scale
        );
        optimum
    }

    /// The curvature report the Python contract reads (`FittedModel.curvature`):
    /// [`curvature_inference_forspec`] on the spec the fit publishes, κ̂ and ℓ̂
    /// written back as `spatial_kappa_incumbent` writes them. It must certify
    /// κ̂, rail or not, and report it unchanged.
    fn curvature_report(
        request: &StandardFitRequest<'_>,
        optimum: &ConstantCurvatureOptimum,
        label: &str,
    ) -> CurvatureInference {
        let mut spec = request.spec.clone();
        let Some(SmoothBasisSpec::ConstantCurvature { spec: cc, .. }) =
            spec.smooth_terms.get_mut(0).map(|term| &mut term.basis)
        else {
            panic!("curv term");
        };
        cc.kappa = optimum.kappa;
        cc.length_scale = optimum.length_scale;
        let report = curvature_inference_forspec(
            request.data.view(),
            request.y.view(),
            request.weights.view(),
            request.offset.view(),
            &spec,
            0,
            request.family.clone(),
            0.95,
        )
        .expect("curvature inference certifies the published κ̂");
        eprintln!(
            "[1464-KAPPA] {label}: report kappa_hat={:+.6} ci=({:+.4},{:+.4}) support={} range={:.6e}",
            report.kappa_hat,
            report.ci.ci_lo,
            report.ci.ci_hi,
            report.ci.kappa_hat_support.label(),
            report.length_scale_hat
        );
        assert_eq!(report.kappa_hat, optimum.kappa, "{label}: the report restates κ̂");
        report
    }

    #[test]
    fn a_hyperbolic_kernel_draw_is_recovered_hyperbolic_1464() {
        let data = kernel_draw(-2.0);
        let request = request(&data);
        let label = "kernel draw, kappa*=-2";
        let optimum = published(&request, label);
        assert!(optimum.kappa < 0.0, "κ* = −2 must be recovered hyperbolic, got κ̂ = {}", optimum.kappa);
        curvature_report(&request, &optimum, label);
    }

    #[test]
    fn a_spherical_kernel_draw_is_recovered_spherical_1464() {
        let data = kernel_draw(2.0);
        let request = request(&data);
        let label = "kernel draw, kappa*=+2";
        let optimum = published(&request, label);
        assert!(optimum.kappa > 0.0, "κ* = +2 must be recovered spherical, got κ̂ = {}", optimum.kappa);
        curvature_report(&request, &optimum, label);
    }

    /// The saturating radial profile generated at κ* = −2 is explained better by
    /// the spherical distance kernel than by every certified hyperbolic fit: the
    /// published κ̂ is spherical and its profiled criterion is below the best
    /// criterion a κ search confined to the hyperbolic half of the chart certifies
    /// from either of that half's ends.
    #[test]
    fn a_saturating_radial_profile_is_explained_best_by_the_spherical_distance_kernel_1464() {
        let data = saturating_radial(-2.0);
        let request = request(&data);
        let optimum = published(&request, "saturating radial, kappa*=-2");
        let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(request.data.view(), &request.spec, 0);
        let (feature_cols, base_spec) = match &request.spec.smooth_terms[0].basis {
            SmoothBasisSpec::ConstantCurvature { feature_cols, spec, .. } => (feature_cols.clone(), spec.clone()),
            _ => panic!("curv term"),
        };
        let x_term = select_columns(request.data.view(), &feature_cols).expect("columns");
        let profile_from = |start: f64| {
            let mut spec = base_spec.clone();
            spec.kappa = start;
            ConstantCurvatureProfile::new(x_term.view(), request.y.view(), spec).expect("profile")
        };
        let published_value = profile_from(optimum.kappa).evaluate(optimum.kappa).expect("profile at κ̂").0;
        let hyperbolic: Vec<(f64, f64)> = [kappa_min, 0.0]
            .into_iter()
            .map(|start| {
                let hyperbolic = solve_constant_curvature_kappa_profile(
                    ROWS,
                    profile_from(start),
                    &request.options,
                    kappa_min,
                    0.0,
                    0,
                )
                .expect("a κ search on the hyperbolic half certifies");
                let value = profile_from(hyperbolic.kappa).evaluate(hyperbolic.kappa).expect("profile").0;
                (hyperbolic.kappa, value)
            })
            .collect();
        eprintln!(
            "[1464-KAPPA] saturating radial: published kappa_hat={:+.6} V={published_value:.6} chart=[{kappa_min:+.6},{kappa_max:+.6}] hyperbolic certified (kappa, V)={hyperbolic:?}",
            optimum.kappa
        );
        assert!(optimum.kappa > 0.0, "the published κ̂ is the spherical distance kernel's, got {}", optimum.kappa);
        // It rests on the fold wall, and the curvature report certifies it there by
        // local box-KKT rather than refusing it on the whole-box model. Its range
        // is the distance-kernel face, reached and reported as such rather than
        // as an interior minimum short of it.
        let report = curvature_report(&request, &optimum, "saturating radial, kappa*=-2");
        assert_eq!(
            report.length_scale_support,
            gam_geometry::curvature_estimand::RangeEstimateSupport::DistanceKernelLimit,
            "the range at the published κ̂ is the distance-kernel face"
        );
        for (kappa, value) in hyperbolic {
            assert!(
                published_value < value,
                "the published V = {published_value} must beat the certified hyperbolic fit at κ = {kappa}, V = {value}"
            );
        }
    }
}
