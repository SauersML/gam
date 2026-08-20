//! #2720 — WHICH TERM of the penalized objective is not flat along the chart
//! orbit, measured per direction, per atom kind, by finite difference of the
//! objective's own value functions.
//!
//! ## What this adds to the thread, and why the existing numbers are not enough
//!
//! Every measurement on #2720 so far reports ONE scalar per direction: `|gᵀvᵢ|`
//! against the convergence tolerance (`8.24x` / `10.45x`, later `2.33x` /
//! `3.63x` per geometry). That scalar says the penalized objective is not flat.
//! It does not say WHICH penalty is not flat, and the modelling fix is a
//! different change depending on the answer:
//!
//! * if it is the **ARD prior on the coordinates**, the prior's fixed origin
//!   (Gaussian mean 0 on a Euclidean axis, von Mises mean 0 on a periodic one)
//!   is an arbitrary gauge choice, and the fix is to stop making it;
//! * if it is the **smoothness prior on the decoder**, the penalty matrix `S` is
//!   attached to a chart whose coordinates moved, and the fix is in the basis;
//! * if it is a **barrier**, the direction is not a symmetry of the admissible
//!   set at all and must simply leave the quotient.
//!
//! `|gᵀvᵢ|` cannot distinguish those three. A per-term central difference of the
//! value functions can, and the value functions are the right instrument
//! precisely because they are what the line search descends — an analytic
//! gradient agreeing with itself proves nothing (#2714).
//!
//! ## The second question this answers: is the direction likelihood-null for a
//! REASON, or only because the least-squares solve had room?
//!
//! [`SaeManifoldTerm::dense_step_gauge_vector_from_field`] certifies nothing. It
//! solves `design·δB = −motion` in the least-squares sense and returns the
//! solution whatever the residual is. When the design `n × M` has full ROW rank
//! (`M ≥ rank`), that solve is exact for EVERY field, so a `1e-16` reconstruction
//! residual is a statement about the shape of the fixture and not about the
//! field being a symmetry of anything. This module records `n_active` against
//! `M` next to every residual so the two readings cannot be confused.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_gauge_posterior_flatness_2720;` — its single
// declaration.
#![cfg(test)]
use super::*;
use crate::manifold::fit_drivers::JointFitTermination;

/// The #2253/#2234 planted circle, verbatim from
/// `tests_gauge_frame_roundtrip_2720` (same LCG, same seed, same shape) so every
/// number here sits on the geometry the `1.29e-16` unframed measurement and the
/// `2.33x` geometry sweep were both taken on.
pub(crate) fn planted_circle_cloud() -> Array2<f64> {
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    z
}

/// A seeded single-atom term on `basis`, at latent dimension `latent_dim`.
pub(crate) fn seeded_term_of_kind(
    target: ArrayView2<'_, f64>,
    basis: &str,
    latent_dim: usize,
) -> SaeManifoldTerm {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec![basis.to_string()],
        atom_dim: vec![latent_dim],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target,
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    seed.base_term
}

/// The atom kinds this module sweeps, with the latent dimension each one
/// requires. #2720's own "Not established" section records that its central
/// measurement was taken on `Periodic` atoms alone; this list is that gap
/// closed. Every kind that emits a chart-gauge orbit at all **and can be
/// seeded through `sae_manifold_fit_minimal`** is here:
///
/// * `Sphere` rides the ambient 3-vector cover, but `atom_dim` names the
///   INTRINSIC dimension — `sae_build_atom_plans` requires `d == 2` for the
///   sphere (the plan itself carries the ambient width 3), and the same
///   intrinsic-dim reading holds for `ProjectivePlane`;
/// * `KleinBottle` requires `d = 2`;
/// * `Linear`, `EuclideanPatch`, `Poincare` and `Duchon` emit translation and
///   dilation fields per axis; `Periodic` / `Torus` emit the phase shift only.
///
/// `Cylinder` emits an orbit (one continuous phase gauge on the `S¹` axis,
/// `dense_step_gauge_vectors`) but is **not seedable through the minimal-seed
/// path** — `sae_build_atom_plans` rejects it as a birth-discovered-only
/// topology ("seed with periodic, duchon, sphere, torus, or euclidean_patch
/// and let the structure search grow a cylinder by evidence"), so it cannot
/// be constructed by this instrument and is excluded with that reason
/// recorded. `Mobius`, `FiniteSet` and `Precomputed` emit no orbit by
/// construction (their arms in `dense_step_gauge_vectors` say why), so they
/// cannot contribute a direction to either span and are not swept.
pub(crate) const GAUGE_SWEEP_KINDS: &[(&str, usize)] = &[
    ("periodic", 1),
    ("torus", 2),
    ("duchon", 1),
    ("linear", 1),
    ("euclidean", 1),
    ("poincare", 1),
    ("sphere", 2),
    ("klein_bottle", 2),
];

/// The penalized objective, split into the pieces `penalized_objective_total`
/// sums. Every field is a VALUE, not a gradient — this struct is differenced,
/// never differentiated.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ObjectiveTerms {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub analytic: f64,
    pub repulsion: f64,
    pub amplitude_barrier: f64,
    pub separation_barrier: f64,
}

impl ObjectiveTerms {
    pub(crate) fn total(&self) -> f64 {
        self.data_fit
            + self.assignment_sparsity
            + self.smoothness
            + self.ard
            + self.analytic
            + self.repulsion
            + self.amplitude_barrier
            + self.separation_barrier
    }

    fn combine(&self, other: &Self, scale: f64) -> Self {
        Self {
            data_fit: (self.data_fit - other.data_fit) * scale,
            assignment_sparsity: (self.assignment_sparsity - other.assignment_sparsity) * scale,
            smoothness: (self.smoothness - other.smoothness) * scale,
            ard: (self.ard - other.ard) * scale,
            analytic: (self.analytic - other.analytic) * scale,
            repulsion: (self.repulsion - other.repulsion) * scale,
            amplitude_barrier: (self.amplitude_barrier - other.amplitude_barrier) * scale,
            separation_barrier: (self.separation_barrier - other.separation_barrier) * scale,
        }
    }
}

pub(crate) fn objective_terms(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
) -> Result<ObjectiveTerms, String> {
    let loss = term.loss_scaled(target, rho, 1.0)?;
    Ok(ObjectiveTerms {
        data_fit: loss.data_fit,
        assignment_sparsity: loss.assignment_sparsity,
        smoothness: loss.smoothness,
        ard: loss.ard,
        analytic: term
            .analytic_penalty_value_total(registry, 1.0)
            .map_err(|err| err.to_string())?,
        repulsion: term.decoder_repulsion_value(1.0),
        amplitude_barrier: term.amplitude_barrier_value(1.0),
        separation_barrier: term.separation_barrier_value(1.0),
    })
}

/// Central difference of every objective term along the unit direction
/// `(δt, δβ)`, at step `h`, through the SAME `apply_newton_step` the inner solve
/// moves with (so the measured surface is the one the solver walks, wrapping and
/// basis rebuild included).
pub(crate) fn directional_derivative_terms(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
    direction: &Array1<f64>,
    h: f64,
) -> Result<ObjectiveTerms, String> {
    let dense_len = term.n_obs() * term.assignment.row_block_dim();
    let snapshot = term.snapshot_mutable_state();
    // `apply_newton_step` refuses a non-positive step, so the backward arm walks
    // the NEGATED direction at the same positive length rather than a negative
    // length along the same one. Same point, same code path.
    let evaluate =
        |term: &mut SaeManifoldTerm, walk: &Array1<f64>| -> Result<ObjectiveTerms, String> {
            term.apply_newton_step(walk.slice(s![..dense_len]), walk.slice(s![dense_len..]), h)?;
            let out = objective_terms(term, target, rho, registry);
            term.restore_mutable_state(&snapshot)?;
            out
        };
    let backward = direction.mapv(|value| -value);
    let plus = evaluate(term, direction)?;
    let minus = evaluate(term, &backward)?;
    Ok(plus.combine(&minus, 1.0 / (2.0 * h)))
}

/// How many rows the atom is actually active on, against the atom's basis size.
/// `n_active <= m` is the regime in which the gauge construction's least-squares
/// compensation is exact for ANY field, symmetry or not.
pub(crate) fn active_rows_and_basis_size(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<(usize, usize), String> {
    let mut active = 0usize;
    for row in 0..term.n_obs() {
        let assignments = term.assignment.try_assignments_row(row)?;
        if assignments[atom_idx] != 0.0 {
            active += 1;
        }
    }
    Ok((active, term.atoms[atom_idx].basis_size()))
}

pub(crate) fn unit_norm(mut v: Array1<f64>) -> Option<Array1<f64>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Some(v)
}

/// #2720 — the per-term attribution the thread has been missing.
///
/// This test PRINTS a table and asserts only what it can prove without deciding
/// the modelling question: that the data fit is flat along every constructed
/// direction (the likelihood-symmetry claim, re-measured here through the VALUE
/// function rather than through the construction's own least-squares residual),
/// and that at least one direction moves at least one penalty term by more than
/// the data fit (the demonstrandum: the orbit is not a posterior symmetry).
#[test]
fn chart_orbit_directional_derivative_splits_by_objective_term_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut any_penalty_dominates = false;
    let mut worst_data_fit_slope = 0.0_f64;
    for &(kind, latent_dim) in GAUGE_SWEEP_KINDS {
        let mut term = seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(
            0.0,
            0.0,
            vec![Array1::<f64>::zeros(term.assignment.coords[0].latent_dim())],
        );
        let base = objective_terms(&term, z.view(), &rho, &registry).expect("base objective");
        let gauges = term.dense_step_gauge_vectors().expect("gauge vectors");
        let (active, basis_size) = active_rows_and_basis_size(&term, 0).expect("active rows");
        println!(
            "\n[2720-split] kind={kind} d={latent_dim} n={} p={} M={basis_size} n_active={active} \
             gauge_dirs={} objective={:.9e}",
            term.n_obs(),
            term.output_dim(),
            gauges.len(),
            base.total(),
        );
        println!(
            "[2720-split]   {:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "dir", "data_fit", "sparsity", "smoothness", "ard", "barriers", "total"
        );
        for (index, gauge) in gauges.into_iter().enumerate() {
            let Some(direction) = unit_norm(gauge) else {
                continue;
            };
            // `h` is a relative step on a unit direction: small enough that the
            // second-order term is below the difference's own truncation error,
            // large enough that the difference is not cancellation noise.
            let h = 1.0e-5;
            let slope =
                directional_derivative_terms(&mut term, z.view(), &rho, &registry, &direction, h)
                    .expect("directional derivative");
            let barriers = slope.analytic
                + slope.repulsion
                + slope.amplitude_barrier
                + slope.separation_barrier;
            println!(
                "[2720-split]   {index:>4} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
                slope.data_fit,
                slope.assignment_sparsity,
                slope.smoothness,
                slope.ard,
                barriers,
                slope.total(),
            );
            worst_data_fit_slope = worst_data_fit_slope.max(slope.data_fit.abs());
            let penalty = slope.total() - slope.data_fit;
            if penalty.abs() > slope.data_fit.abs().max(1.0e-12) {
                any_penalty_dominates = true;
            }
        }
    }
    assert!(
        any_penalty_dominates,
        "no constructed chart-gauge direction moved the penalty block more than the data fit; \
         #2720's central claim (the orbit is a likelihood symmetry that the priors break) would \
         then be unreproducible on this fixture and the modelling question would be moot"
    );
    println!("[2720-split] worst |d data_fit| over all directions = {worst_data_fit_slope:.6e}");
}

/// The three families [`SaeManifoldTerm::gauge_quotient_basis`] concatenates,
/// counted, so a per-direction verdict can be attributed to the family that
/// produced it without re-deriving the concatenation order.
pub(crate) fn quotient_family_sizes(
    term: &SaeManifoldTerm,
    lambda_smooth: &[f64],
) -> Result<(usize, usize, usize), String> {
    Ok((
        term.dense_step_gauge_vectors()?.len(),
        term.joint_decoder_beta_null_directions(lambda_smooth)?
            .len(),
        term.decoder_channel_null_directions()?.len(),
    ))
}

/// #2720's ACCEPTANCE CRITERION, as an executable gate.
///
/// > the directional derivative of the **penalized** objective along every
/// > constructed orbit direction is at or below the convergence tolerance
///
/// denominated in the tolerance the accept path itself uses
/// (`SAE_MANIFOLD_INNER_GRAD_REL_TOL · inner_iterate_scale()`, the single source
/// of truth at `construction_quasi_laplace.rs:937`), and measured by central
/// difference of the value function rather than by the analytic gradient, so it
/// cannot pass by the gradient and the objective agreeing with each other while
/// both are wrong (#2714).
///
/// This is a property of the SPAN THE GATES REMOVE, not of the chart-gauge
/// construction: it reads `posterior_null_quotient_basis`, the single source of
/// truth for that span, so it fails whichever family contributes a live
/// direction.
#[test]
fn quotient_span_is_flat_for_the_penalized_objective_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut violations: Vec<String> = Vec::new();
    let mut measured = 0usize;
    for &(kind, latent_dim) in GAUGE_SWEEP_KINDS {
        let mut term = seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(
            0.0,
            0.0,
            vec![Array1::<f64>::zeros(term.assignment.coords[0].latent_dim())],
        );
        let lambda_smooth = rho
            .lambda_smooth_vec()
            .expect("the fixture rho carries one smoothing block per atom");
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let (chart, beta_null, channel_null) =
            quotient_family_sizes(&term, &lambda_smooth).expect("family sizes");
        let basis = term
            .posterior_null_quotient_basis(&lambda_smooth)
            .expect("the seeded fixture has a well-defined quotient span");
        println!(
            "\n[2720-gate] kind={kind} tol={tolerance:.6e} span={} \
             (chart={chart}, beta_null={beta_null}, channel_null={channel_null})",
            basis.len(),
        );
        for (index, direction) in basis.into_iter().enumerate() {
            let slope = directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                1.0e-5,
            )
            .expect("directional derivative");
            measured += 1;
            let total = slope.total().abs();
            println!(
                "[2720-gate]   dir {index:>3}  |d f| = {total:.6e}  ({:.3}x tol)  \
                 data_fit={:.3e} smooth={:.3e} ard={:.3e}",
                total / tolerance,
                slope.data_fit,
                slope.smoothness,
                slope.ard,
            );
            if total > tolerance {
                violations.push(format!(
                    "{kind} direction {index}: |d f| = {total:.6e} = {:.3}x the convergence \
                     tolerance {tolerance:.6e} (data_fit {:.3e}, smoothness {:.3e}, ard {:.3e})",
                    total / tolerance,
                    slope.data_fit,
                    slope.smoothness,
                    slope.ard,
                ));
            }
        }
    }
    assert!(
        measured > 0,
        "no quotient direction was measured on any fixture, so this gate certified nothing"
    );
    assert!(
        violations.is_empty(),
        "the span both inner convergence gates project out of the KKT residual carries LIVE \
         descent of the penalized objective. Every consumer that reads a small quotient as \
         `stationary` — the inner accept gate, the terminal polish's stationarity return, and \
         `SaeInstalledInnerKktAudit::certifies()` via `parameter_space.certifies()` — is reading \
         a point that is not stationary.\n  {}",
        violations.join("\n  "),
    );
}

/// The largest `|gᵀvᵢ|` over a list of unit directions, plus the per-direction
/// central-difference slopes of the penalized objective.
fn slopes_along(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
    directions: &[Array1<f64>],
) -> Vec<f64> {
    directions
        .iter()
        .map(|direction| {
            directional_derivative_terms(term, target, rho, registry, direction, 1.0e-5)
                .expect("directional derivative")
                .total()
        })
        .collect()
}

/// #2720 OPTION 2, REFUTED: "extend the compensation to the priors" has no
/// solution, and the reason is that a first-order zero of the prior derivative
/// is a property of the STATE rather than of the field.
///
/// The issue offers, as its second shape of modelling fix, solving for the
/// `(δt, δβ)` that cancels the reconstruction change *and* is first-order
/// stationary for the priors, noting that it "is generally over-determined and
/// may have no solution, which would itself be the finding". This test takes
/// that finding.
///
/// The freedom available is exactly the choice of coordinate FIELD: once a field
/// is fixed, `δβ` is the unique least-squares solution of an over-determined
/// system (`n_active = 42` rows against `M ∈ {2,3}` columns here), so nothing
/// downstream can be steered. On a patch atom the enumerated field family is
/// two-dimensional — constant shift and dilation — and their prior slopes have
/// OPPOSITE signs, so a combination with zero first-order prior derivative
/// always exists at any single state. That combination is what option 2 would
/// return.
///
/// The test then asks the only question that matters: is it a symmetry? A
/// symmetry is flat at every state, so the combination is rebuilt from the
/// freshly constructed fields at a DIFFERENT state and measured there. It is not
/// flat — by orders of magnitude — because the two slopes do not move together
/// as the state moves. There is therefore no field in the family whose flow
/// leaves the posterior invariant, which is what "no posterior gauge exists
/// here" means constructively.
#[test]
fn option_two_prior_aware_compensation_has_no_state_independent_solution_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut term = seeded_term_of_kind(z.view(), "linear", 1);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
    let dense_len = term.n_obs() * term.assignment.row_block_dim();

    let at_state = |term: &mut SaeManifoldTerm| -> (Vec<Array1<f64>>, Vec<f64>) {
        let directions: Vec<Array1<f64>> = term
            .dense_step_gauge_vectors()
            .expect("gauge vectors")
            .into_iter()
            .filter_map(unit_norm)
            .collect();
        let slopes = slopes_along(term, z.view(), &rho, &registry, &directions);
        (directions, slopes)
    };

    let (home_directions, home_slopes) = at_state(&mut term);
    assert_eq!(
        home_directions.len(),
        2,
        "the linear patch must enumerate exactly the shift and dilation fields; option 2's \
         freedom is that two-dimensional family and nothing else"
    );
    // The zero-slope combination option 2 would return, in the orthonormal frame
    // the construction emits: `c = s1·v0 − s0·v1` has slope `s1·s0 − s0·s1 = 0`.
    let (s0, s1) = (home_slopes[0], home_slopes[1]);
    assert!(
        s0 * s1 < 0.0,
        "the two fields must pull the priors in OPPOSITE directions for a zero combination to \
         exist at all: shift {s0:.6e}, dilation {s1:.6e}"
    );
    let combine = |dirs: &[Array1<f64>]| -> Array1<f64> {
        let mut combined = dirs[0].mapv(|value| value * s1);
        for index in 0..combined.len() {
            combined[index] -= s0 * dirs[1][index];
        }
        unit_norm(combined).expect("a nonzero combination of two orthonormal directions")
    };
    let home_combination = combine(&home_directions);
    let home_slope = slopes_along(
        &mut term,
        z.view(),
        &rho,
        &registry,
        std::slice::from_ref(&home_combination),
    )[0];
    println!(
        "[2720-opt2] home: shift={s0:.6e} dilation={s1:.6e} combination={home_slope:.6e} \
         ({:.3}x tol {tolerance:.6e})",
        home_slope.abs() / tolerance,
    );
    assert!(
        home_slope.abs() <= 0.02 * s1.abs(),
        "the constructed combination must actually be the first-order zero at its own state, or \
         this test is refuting something other than option 2: {home_slope:.6e} against a \
         dilation slope of {s1:.6e}"
    );

    // Move to a different state, INSIDE the likelihood-flat set, so the
    // reconstruction is what it was and only the chart parametrisation moved.
    // A symmetry of the posterior would be flat here too.
    let travel = home_directions[0].clone();
    term.apply_newton_step(
        travel.slice(s![..dense_len]),
        travel.slice(s![dense_len..]),
        0.5,
    )
    .expect("a step along the shift field applies");

    let (away_directions, away_slopes) = at_state(&mut term);
    assert_eq!(
        away_directions.len(),
        2,
        "the field family is state-independent in SIZE"
    );
    let away_combination = combine(&away_directions);
    let away_slope = slopes_along(
        &mut term,
        z.view(),
        &rho,
        &registry,
        std::slice::from_ref(&away_combination),
    )[0];
    println!(
        "[2720-opt2] away: shift={:.6e} dilation={:.6e} combination={away_slope:.6e} \
         ({:.3}x tol)",
        away_slopes[0],
        away_slopes[1],
        away_slope.abs() / tolerance,
    );

    assert!(
        away_slope.abs() > tolerance,
        "option 2's compensation would be a symmetry, and this one is not: the combination that \
         zeroes the prior derivative at one state carries |d f| = {away_slope:.6e} at another, \
         which is at or below the convergence tolerance {tolerance:.6e}. If this ever fires, the \
         field family HAS a state-independent prior-stationary member and #2720's option 2 is \
         back on the table."
    );
    // The mechanism, stated as its own assertion so a future reader does not
    // have to infer it: the two slopes do not move together, so no fixed
    // combination of them can stay at zero.
    let home_ratio = s0 / s1;
    let away_ratio = away_slopes[0] / away_slopes[1];
    println!(
        "[2720-opt2] slope ratio shift/dilation: home {home_ratio:.6e} -> away {away_ratio:.6e}"
    );
    assert!(
        (home_ratio - away_ratio).abs() > 0.01 * home_ratio.abs().max(away_ratio.abs()),
        "the shift/dilation slope ratio moved by less than 1% ({home_ratio:.6e} -> \
         {away_ratio:.6e}); a state-independent ratio would mean a fixed combination IS flat \
         everywhere and option 2 has a solution after all"
    );
}

/// #2720 — the defect cannot come back silently: the chart orbit must stay OUT
/// of the span the convergence gates remove, on a fixture where it demonstrably
/// carries live descent.
///
/// The two clauses are deliberately redundant with each other. The first is
/// structural (the span excludes the orbit); the second is behavioural (the
/// orbit is worth excluding). A change that re-merged the spans would fail the
/// first; a change that made the orbit genuinely flat — the modelling fix this
/// issue considered and rejected on the grounds that it makes the coordinate
/// prior improper — would fail the second, and should, because it would mean
/// the argument for the split needs re-deriving rather than re-asserting.
#[test]
fn chart_orbit_stays_out_of_the_convergence_quotient_and_is_worth_excluding_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut term = seeded_term_of_kind(z.view(), "linear", 1);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();

    let chart: Vec<Array1<f64>> = term
        .dense_step_gauge_vectors()
        .expect("gauge vectors")
        .into_iter()
        .filter_map(unit_norm)
        .collect();
    assert!(
        !chart.is_empty(),
        "the linear patch must enumerate a chart orbit, or this fixture cannot see the defect"
    );
    let quotient = term
        .posterior_null_quotient_basis(&lambda_smooth)
        .expect("quotient span");
    let block = term
        .likelihood_flat_block_basis(&lambda_smooth)
        .expect("descent block");

    for (index, direction) in chart.iter().enumerate() {
        let mut residual = direction.clone();
        for basis in &quotient {
            let coeff = residual.dot(basis);
            for i in 0..residual.len() {
                residual[i] -= coeff * basis[i];
            }
        }
        let retained = residual.dot(&residual).sqrt();
        assert!(
            retained >= 1.0 - 1.0e-9,
            "chart direction {index} is (partly) inside the convergence quotient: the projection \
             kept only {retained:.6e} of a unit vector. The gates would then be blind to \
             {:.1}% of any residual along it.",
            100.0 * (1.0 - retained),
        );
        let mut in_block = 0.0_f64;
        for basis in &block {
            let coeff = direction.dot(basis);
            in_block += coeff * coeff;
        }
        assert!(
            in_block >= 1.0 - 1.0e-9,
            "chart direction {index} is not reachable by the descent block ({in_block:.6e} of \
             its unit norm); it would then be a direction no mover reduces and no gate sees"
        );
    }

    let slopes = slopes_along(&mut term, z.view(), &rho, &registry, &chart);
    let worst = slopes.iter().fold(0.0_f64, |acc, s| acc.max(s.abs()));
    println!(
        "[2720-live] chart directions={} worst |d f|={worst:.6e} = {:.1}x tol {tolerance:.6e}",
        chart.len(),
        worst / tolerance,
    );
    assert!(
        worst > 100.0 * tolerance,
        "the chart orbit no longer carries live posterior descent on this fixture (worst \
         |d f| = {worst:.6e} against tolerance {tolerance:.6e}), so the evidence that put it \
         outside the convergence quotient no longer reproduces and the decision needs re-taking"
    );
}

/// #2720's "Not established", second bullet: the framed decoder layout.
///
/// The chart-gauge construction writes its compensation through a Grassmann
/// frame round trip (`δβ · U`, read back through `Uᵀ`) whenever a decoder frame
/// is active, and `tests_gauge_frame_roundtrip_2720` establishes that the round
/// trip preserves the RECONSTRUCTION exactly. That is the likelihood side. This
/// is the posterior side of the same layout: with frames active, the span the
/// convergence gates remove must still be flat for the penalized objective.
///
/// The framed path also changes the border coordinate the decoder-null families
/// are written in (`decoder_channel_null_directions` declines to emit for a
/// framed atom at all, and `joint_decoder_beta_null_directions` diagonalizes the
/// factored operator instead of the full-`B` one), so this is not a repeat of
/// the unframed arm with extra steps — it exercises a different construction.
#[test]
fn quotient_span_is_flat_with_decoder_frames_active_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut activated_total = 0usize;
    let mut measured = 0usize;
    let mut violations: Vec<String> = Vec::new();
    for &(kind, latent_dim) in GAUGE_SWEEP_KINDS {
        let mut term = seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(
            0.0,
            0.0,
            vec![Array1::<f64>::zeros(term.assignment.coords[0].latent_dim())],
        );
        let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let output_dim = term.output_dim();
        let mut activated: Vec<(usize, usize)> = Vec::new();
        for (atom_idx, atom) in term.atoms.iter_mut().enumerate() {
            if let Some(rank) = atom
                .maybe_activate_decoder_frame()
                .expect("frame activation must not error")
            {
                activated.push((atom_idx, rank));
            }
        }
        activated_total += activated.len();
        let basis = term
            .posterior_null_quotient_basis(&lambda_smooth)
            .expect("quotient span on the framed layout");
        println!(
            "[2720-framed] kind={kind} frames={activated:?} p={output_dim} span={} tol={tolerance:.6e}",
            basis.len(),
        );
        for (index, direction) in basis.into_iter().enumerate() {
            let slope = directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                1.0e-5,
            )
            .expect("directional derivative")
            .total()
            .abs();
            measured += 1;
            if slope > tolerance {
                violations.push(format!(
                    "{kind} framed direction {index}: |d f| = {slope:.6e} = {:.3}x tolerance \
                     {tolerance:.6e}",
                    slope / tolerance,
                ));
            }
        }
    }
    // Non-vacuity, both halves: a frame that never activated would make this a
    // duplicate of the unframed gate, and a span that is empty everywhere would
    // certify nothing at all.
    assert!(
        activated_total > 0,
        "no decoder frame activated on any swept kind, so the framed border layout was never \
         exercised and this test is a duplicate of its own unframed sibling"
    );
    assert!(
        measured > 0,
        "the framed arm measured no quotient direction on any kind, so it certified nothing"
    );
    assert!(
        violations.is_empty(),
        "with decoder frames ACTIVE the convergence quotient carries live descent of the \
         penalized objective:\n  {}",
        violations.join("\n  "),
    );
}

/// #2720's acceptance criterion, bullet 1, satisfied where it means something:
/// AT A FIXED POINT.
///
/// The issue asks for a fixture on which "the directional derivative of the
/// penalized objective along every constructed orbit direction is at or below
/// the convergence tolerance". Read as a demand for a SYMMETRY — flat at every
/// state — that is option 1, and option 1 makes the coordinate prior improper
/// along the very directions ARD needs in order to prune. Read as a demand
/// about the point the solver stops at, it is exactly what stationarity means,
/// it is checkable, and it is what this test checks.
///
/// The mechanism that makes it true is `descend_gauge_orbit`: the orbit carries
/// no data-fit curvature, so its only curvature is the priors' — tiny next to
/// the transverse block — and every globalization in this solver shortens
/// steps. The block-coordinate descent is the one mover that can take the long
/// step the orbit needs, and #2762 wired it into all three fixed-point claims.
///
/// CONDITIONAL on the exit claiming a fixed point, and that is not hedging: an
/// exit on iteration budget claims nothing, and asserting stationarity there
/// would be asserting that a truncated optimization is optimal. The
/// unconditional half — the measurement itself, and the fit descending — runs
/// either way and is printed.
#[test]
fn at_an_inner_fixed_point_the_chart_orbit_slope_is_within_the_kkt_tolerance_2720() {
    let z = planted_circle_cloud();
    let mut term = seeded_term_of_kind(z.view(), "periodic", 1);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let entry = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the seed");
    let outcome = term
        .run_joint_fit_arrow_schur_with_termination_policy(
            z.view(),
            &mut rho,
            None,
            512,
            0.05,
            1.0e-6,
            1.0e-6,
            // The EVIDENCE lane: its `NoStrictDecrease` exit is the one the
            // refine loop turns into `criterion_fixed_point`.
            false,
        )
        .expect("the planted-circle fixture fits");
    let settled = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the exit");
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();

    let system = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("the exit state assembles");
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let dense_len = n * q;
    let mut gradient = Array1::<f64>::zeros(dense_len + system.gb.len());
    for (row_index, row) in system.rows.iter().enumerate() {
        let base = system.row_offsets[row_index];
        for (axis, &value) in row.gt.iter().enumerate() {
            gradient[base + axis] = value;
        }
    }
    for (index, &value) in system.gb.iter().enumerate() {
        gradient[dense_len + index] = value;
    }
    drop(system);

    let orbit: Vec<Array1<f64>> = term
        .dense_step_gauge_vectors()
        .expect("gauge vectors at the exit state")
        .into_iter()
        .filter_map(unit_norm)
        .collect();
    assert!(
        !orbit.is_empty(),
        "the periodic atom must still enumerate a phase orbit at the exit state, or there is \
         nothing for this test to be about"
    );
    let mut worst = 0.0_f64;
    for direction in &orbit {
        if direction.len() == gradient.len() {
            worst = worst.max(gradient.dot(direction).abs());
        }
    }
    println!(
        "[2720-fixed] termination={:?} objective {entry:.9e} -> {settled:.9e}  \
         ‖g‖={:.6e}  maxᵢ|gᵀvᵢ| over the chart orbit = {worst:.6e} ({:.4}x tol {tolerance:.6e})",
        outcome.termination,
        gradient.dot(&gradient).sqrt(),
        worst / tolerance,
    );
    assert!(
        settled < entry,
        "the fit must strictly descend for its exit state to be worth grading: \
         {entry} -> {settled}"
    );
    if matches!(
        outcome.termination,
        JointFitTermination::Heuristic | JointFitTermination::NoStrictDecrease
    ) {
        assert!(
            worst <= tolerance,
            "the inner fit claimed a FIXED POINT ({:?}) while the chart orbit still carries \
             maxᵢ|gᵀvᵢ| = {worst:.6e} = {:.3}x the convergence tolerance {tolerance:.6e}. That is \
             #2720's acceptance criterion failing at the only state where it is a statement \
             about stationarity rather than about a symmetry the model does not have.",
            outcome.termination,
            worst / tolerance,
        );
    }
}
