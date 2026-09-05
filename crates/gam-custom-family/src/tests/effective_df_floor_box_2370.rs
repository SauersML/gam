//! Regression (#2370): the per-term effective-df-floor ρ *upper* bound must
//! never fall below the ρ box's own *lower* bound.
//!
//! `fit_custom_family_with_rho_prior` boxes the outer ρ = log λ with
//!
//! ```text
//! lower = options.rho_lower_bound                        (default -10.0)
//! upper = effective_df_floor_rho_upper_bounds(.., RhoBox { lower, ceiling })
//! ```
//!
//! The upper-bound derivation used to bisect the `edf(ρ) = 1` crossing on
//! `[-ceiling, ceiling] = [-12, 12]` and reject a crossing only at or below
//! `-ceiling`. That lower reference was a MIRROR of the ceiling, decoupled from
//! the box's actual floor. While the ceiling was 10.0 the two coincided and the
//! defect was latent; #2356 raised the ceiling to 12.0 and opened the window
//! `(-12, -10)`. A term whose structural `edf = 1` crossing lands in that
//! window yielded an upper bound below the real floor -10, so the optimizer
//! received the inverted box `[-10, -11.855…]` and `f64::clamp(min, max)`
//! panicked with `min > max` across the FFI boundary.
//!
//! The fix anchors the pre-check, the bisection endpoint, and the acceptance
//! guard on the caller's true lower wall, so the below-box case routes to the
//! "floor not enforceable inside the box" arm (`edf(lower) ≤ 1` ⇒ since edf
//! decreases in ρ, `edf ≤ 1` everywhere in the box) and the term keeps the
//! uniform ceiling. The emitted bound is then strictly above the floor by
//! construction.

use super::*;

/// A 2-column term with `X = c·I₂` and penalty `S = I₂` has design Gram
/// `G = XᵀX = c²I₂` and structural generalized eigenvalues `γ = [c², c²]`, so
///
/// ```text
/// edf(ρ) = Σ_j γ_j/(γ_j + e^ρ) = 2c²/(c² + e^ρ),
/// ```
///
/// which equals the `EFFECTIVE_DF_FLOOR` of 1 exactly when `e^ρ = c²`. Setting
/// `c = exp(rho_star/2)` therefore places the `edf = 1` crossing at `rho_star`.
fn two_dir_term(rho_star: f64) -> (Vec<ParameterBlockSpec>, PenaltyLabelLayout) {
    let c = (0.5 * rho_star).exp(); // c² = e^{rho_star}
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [c, 0.0],
        [0.0, c]
    ]));
    let spec = ParameterBlockSpec {
        name: "wiggle".to_string(),
        design,
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0, 0.0], [0.0, 1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0, 0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let layout = PenaltyLabelLayout {
        penalty_counts: vec![1],
        physical_to_outer: vec![Some(0)],
        fixed_log_lambdas: vec![None],
        initial_rho: array![0.0],
        joint_specs: std::sync::Arc::new(Vec::new()),
        joint_roots: std::sync::Arc::new(Vec::new()),
        joint_to_outer: Vec::new(),
    };
    (vec![spec], layout)
}

/// The production floor, so this tracks `rho_lower_bound` if it ever moves
/// rather than pinning a private copy of the literal.
fn production_lower() -> f64 {
    BlockwiseFitOptions::default().rho_lower_bound
}

/// Single call site for the bound derivation.
///
/// The two walls used to be adjacent bare `f64` parameters, so transposing
/// them compiled silently and surfaced only as a wrong answer — a real earlier
/// revision of this very file did exactly that. They now travel inside
/// [`RhoBox`], whose per-wall newtypes make the transposition a type error and
/// whose constructor validates the ordering once.
fn upper_bounds_for(
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    ceiling: f64,
    lower: f64,
) -> Result<Array1<f64>, CustomFamilyError> {
    let rho_box = RhoBox::new(RhoLowerWall(lower), RhoCeiling(ceiling))?;
    effective_df_floor_rho_upper_bounds(specs, layout, 1, rho_box)
}

#[test]
fn crossing_between_neg_ceiling_and_the_box_floor_keeps_the_ceiling_2370() {
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    // The decoupled window opened by #2356: strictly below the box floor, but
    // strictly above the old `-ceiling` reference. This is the exact geometry
    // that inverted the box.
    assert!(
        -ceiling < -11.0 && -11.0 < lower,
        "fixture must sit in the (-ceiling, lower) window: ceiling={ceiling}, lower={lower}"
    );
    let (specs, layout) = two_dir_term(-11.0);
    let upper =
        upper_bounds_for(&specs, &layout, ceiling, lower).expect("bounds derivation must succeed");

    // The invariant #2370 forbids violating. Pre-fix this returned -11.0.
    assert!(
        upper[0] >= lower,
        "effective-df-floor upper {} fell below the rho box lower bound {lower} \
         (an inverted box reaches f64::clamp and panics)",
        upper[0],
    );
    // And the correct disposition is NOT a pin at the floor: the edf floor is
    // unenforceable anywhere in this box, so the term keeps the uniform ceiling.
    assert_eq!(
        upper[0], ceiling,
        "a term whose edf=1 crossing lies below the box must keep the uniform ceiling",
    );
}

#[test]
fn interior_crossing_still_tightens_the_upper_bound_2370() {
    // Positive control: the fix must not disable legitimate tightening. A
    // crossing at ρ = 0 is well inside [-10, 12], so the bound tracks it.
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    let (specs, layout) = two_dir_term(0.0);
    let upper =
        upper_bounds_for(&specs, &layout, ceiling, lower).expect("bounds derivation must succeed");
    assert!(
        upper[0] > lower && upper[0] < ceiling,
        "an interior crossing must yield an interior upper bound, got {}",
        upper[0],
    );
    assert!(
        upper[0].abs() < 1e-3,
        "upper bound must track the edf=1 crossing at rho=0, got {}",
        upper[0],
    );
}

#[test]
fn derived_upper_bound_never_inverts_the_box_across_the_crossing_range_2370() {
    // The invariant itself, swept across crossings from far below the box to
    // far above it: the derivation may only ever return a bound the box can
    // actually contain.
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    for step in 0..=40 {
        let rho_star = -20.0 + f64::from(step);
        let (specs, layout) = two_dir_term(rho_star);
        let upper = upper_bounds_for(&specs, &layout, ceiling, lower)
            .expect("bounds derivation must succeed");
        assert!(
            upper[0] > lower && upper[0] <= ceiling,
            "crossing at rho*={rho_star} produced upper={} outside the box ({lower}, {ceiling}]",
            upper[0],
        );
    }
}

#[test]
fn a_pinned_box_yields_a_well_ordered_single_point_box_2370() {
    // The derivation needs no special case for a pinned box: no tightening is
    // possible (the term keeps the uniform ceiling), so the emitted upper bound
    // equals the floor and the box the optimizer receives is the single point
    // the caller pinned — never inverted.
    let pinned_at = EFFECTIVE_DF_CEILING;
    let (specs, layout) = two_dir_term(0.0);
    let upper = upper_bounds_for(&specs, &layout, pinned_at, pinned_at)
        .expect("a pinned rho box must be accepted");
    assert_eq!(
        upper[0], pinned_at,
        "a pinned box must emit its own wall as the upper bound, got {}",
        upper[0],
    );
}

#[test]
fn the_rho_box_constructor_rejects_an_inverted_pair_but_accepts_a_pinned_one_2370() {
    let ceiling = EFFECTIVE_DF_CEILING;
    // Inverted: floor above the ceiling.
    let inverted = RhoBox::new(RhoLowerWall(13.0), RhoCeiling(ceiling))
        .expect_err("an inverted pair must be rejected at construction");
    let message = inverted.to_string();
    assert!(
        message.contains("13") && message.contains("12"),
        "the refusal must name both offending walls, got: {message}"
    );
    // A PINNED box (lower == ceiling) is legal, not degenerate: the caller has
    // fixed λ. This mirrors the outer optimizer's
    // `pinned_equal_rho_bounds_are_accepted_2370`; the two layers must agree on
    // what the admissible set is, or they can drift apart exactly as the two
    // #2370 constants did.
    let pinned = RhoBox::new(RhoLowerWall(ceiling), RhoCeiling(ceiling))
        .expect("a pinned rho box must be accepted, matching the outer optimizer");
    assert_eq!(pinned.lower(), pinned.ceiling());
    // Non-finite walls are rejected as admissible log-strengths.
    assert!(
        RhoBox::new(RhoLowerWall(f64::NAN), RhoCeiling(ceiling)).is_err(),
        "a non-finite floor must be rejected"
    );
    // And the production pair is accepted, with the walls readable back in the
    // order they were supplied.
    let ok = RhoBox::new(RhoLowerWall(production_lower()), RhoCeiling(ceiling))
        .expect("the production rho box must be valid");
    assert_eq!(ok.lower(), production_lower());
    assert_eq!(ok.ceiling(), ceiling);
}

#[test]
fn a_caller_box_that_is_already_inverted_is_a_typed_error_2370() {
    // If the CALLER hands in a lower wall at or above the ceiling the box is
    // empty or degenerate before any per-term tightening; that must be a typed
    // refusal carrying both bounds, not a panic.
    let (specs, layout) = two_dir_term(0.0);
    let error = upper_bounds_for(&specs, &layout, EFFECTIVE_DF_CEILING, 13.0)
        .expect_err("an inverted caller box must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("13") && message.contains("12"),
        "typed error must name both offending bounds, got: {message}"
    );
}

/// #2608 (zz_measure): how far would a RELATIVE floor move the wall for the
/// rank-1 term the absolute floor cannot touch?
///
/// `effective_df_floor_bound` opens with `if !(edf_max > EFFECTIVE_DF_FLOOR)`,
/// and the floor is exactly `1.0`. A rank-1 penalty has a one-dimensional range
/// space, so `edf(ρ) = γ/(γ + e^ρ)` ranges over `(0, 1]` and reaches 1 only as
/// `λ → 0`: the test is false for every rank-1 term and the term is skipped. The
/// null-space half of a Marra–Wood double penalty IS rank-1, so the linear
/// direction of every smooth is unprotected — measured on penguins at
/// `edf ≈ 4.6e-5`, i.e. dead.
///
/// The proposed repair is a floor RELATIVE to what the term can attain: require
/// `edf ≥ f · edf_max` for some `f < 1`. The objection to it is that on
/// near-separable data the Fisher weights `p(1−p)` collapse and the REML surface
/// flattens regardless, so a relative floor might only move a wall rather than
/// restore data-driven selection. That objection deserves a number.
///
/// For rank 1 the crossing is CLOSED FORM, so no bisection or fitting is needed
/// to get it. From `γ/(γ + e^ρ) = f`:
///
/// ```text
/// ρ*(f) = ln γ + ln((1 − f)/f)
/// ```
///
/// which is finite for every `f ∈ (0, 1)` — so the relative floor is well posed
/// exactly where the absolute floor is not. This prints `ρ*(f)` against the
/// uniform ceiling the term currently keeps, so the size of the tightening is a
/// number rather than an intuition. It does NOT settle whether the tightening
/// helps the fit; that needs penguins. It settles that there is something real
/// to tighten TO.
///
/// The closed form is checked against the quantity it claims to invert, because
/// a printed table that asserts nothing passes for every behaviour of the code
/// it calls (#2818). `rho*(f) = ln γ + ln((1-f)/f)` is the ρ at which a rank-1
/// term retains exactly the fraction `f`, so `γ/(γ + e^(ρ*)) = f` identically —
/// the "check on the closed form" the loop below already computes and printed
/// without reading. Finiteness of `rho*` for every `f` in (0,1) is the other
/// claim this table exists to support, and it is asserted rather than eyeballed.
#[test]
fn zz_measure_2608_relative_floor_wall_movement() {
    let ceiling = EFFECTIVE_DF_CEILING;
    let mut checked = 0usize;
    eprintln!(
        "#2608: rank-1 term, absolute floor {EFFECTIVE_DF_FLOOR} is UNREACHABLE \
         (edf_max = 1.0 exactly, attained only as lambda -> 0) => term exempt, \
         keeps uniform ceiling {ceiling}"
    );
    eprintln!("#2608: gamma   f      rho*(f)=ln(gamma)+ln((1-f)/f)   tightening vs ceiling");
    for gamma in [1.0e-2_f64, 1.0, 1.0e2, 1.0e4] {
        for f in [0.25_f64, 0.5, 0.75, 0.9] {
            let rho_star = gamma.ln() + ((1.0 - f) / f).ln();
            // edf actually retained at that rho, as a check on the closed form.
            let retained = gamma / (gamma + rho_star.exp());
            eprintln!(
                "#2608: {gamma:>7.1e}  {f:>4.2}   {rho_star:>12.6}                  \
                 {:>8.4} nats   (edf there = {retained:.6})",
                ceiling - rho_star,
            );
            assert!(
                rho_star.is_finite(),
                "rho*({f}) must be finite for every f in (0,1) at gamma={gamma}; \
                 that finiteness is the whole reason the RELATIVE floor is well \
                 posed exactly where the absolute one is not"
            );
            // The identity is exact in the reals; the only slack is the roundoff
            // of the log-exp round trip, whose absolute error enters as
            // `eps*|rho*|` inside the exponent. The bound is that residue in
            // roundoff units, not a number read off the output.
            let roundoff = 64.0 * f64::EPSILON * (1.0 + rho_star.abs());
            assert!(
                (retained - f).abs() <= roundoff,
                "the closed form must invert the edf it claims to: at gamma={gamma} \
                 f={f} it retains {retained}, off by {} against a {roundoff} roundoff \
                 allowance",
                (retained - f).abs()
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked, 16,
        "the 4x4 (gamma, f) table must be swept in full; a loop that silently \
         ran zero rows prints the same nothing as one that ran"
    );
    eprintln!(
        "#2608: reading -- rho*(f) is FINITE for every f in (0,1), so the relative \
         floor is well posed exactly where the absolute one is not. Whether the \
         tightening improves the FIT is a penguins question, not this one."
    );
}

/// A rank-1 term whose single generalized eigenvalue is `γ = c²`.
///
/// `X = [[c]]`, `S = [[1]]`, so `G = XᵀX = c²` and the pencil `(G, S)` on
/// `range(S)` has the one eigenvalue `c²`.
fn one_dir_term(gamma: f64) -> (Vec<ParameterBlockSpec>, PenaltyLabelLayout) {
    let c = gamma.sqrt();
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[c]]));
    let spec = ParameterBlockSpec {
        name: "linear".to_string(),
        design,
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let layout = PenaltyLabelLayout {
        penalty_counts: vec![1],
        physical_to_outer: vec![Some(0)],
        fixed_log_lambdas: vec![None],
        initial_rho: array![0.0],
        joint_specs: std::sync::Arc::new(Vec::new()),
        joint_roots: std::sync::Arc::new(Vec::new()),
        joint_to_outer: Vec::new(),
    };
    (vec![spec], layout)
}

/// The rank-1 relative floor is a LOGIT, and the bisection recovers it (#2615).
///
/// `EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION` reached its third value in one day,
/// each read off a held-out log-loss curve, and the code gave no way to say what
/// the number means. It has an exact closed form. For a rank-1 penalty,
/// `edf(ρ) = γ/(γ + e^ρ)`, and `edf_max` is evaluated at `λ = 0`, where every
/// positive `γ_j` contributes exactly `1` — so `edf_max = 1` with NO design
/// dependence, the relative target is just `f`, and
///
/// ```text
///   γ/(γ + e^{ρ*}) = f    ⟺    ρ* = ln γ + ln((1 − f)/f).
/// ```
///
/// `f = 0.90` gives `ln(1/9) = −2.1972`, which is the `ρ* = ln γ − 2.20` the
/// constant's own doc records — that agreement is the evidence the form is
/// right, and this test makes it executable against the PRODUCTION bisection
/// rather than against a comment.
///
/// The reason to pin it: `edf = γ/(γ + λ)` is the data's share of the posterior
/// precision, so `edf ≥ f` is exactly `λ/γ ≤ (1 − f)/f` — a ceiling on the
/// prior-to-data precision odds. Anyone changing `f` is changing those odds
/// (0.90 is 1:9; 0.50 is 1:1), and this test is where that reading is anchored.
#[test]
fn the_rank_one_relative_floor_is_a_logit_of_the_prior_to_data_odds_2615() {
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    let f = EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION;
    let odds = (1.0 - f) / f;

    for gamma in [1.0e-2_f64, 1.0, 1.0e2] {
        let expected = gamma.ln() + odds.ln();
        assert!(
            lower < expected && expected < ceiling,
            "fixture must place ρ* strictly inside the box: γ={gamma:e}, ρ*={expected}, \
             box=({lower}, {ceiling})"
        );
        let (specs, layout) = one_dir_term(gamma);
        let upper = upper_bounds_for(&specs, &layout, ceiling, lower)
            .expect("bounds derivation must succeed");
        // Bisection on a smooth monotone scalar, so agreement is limited by the
        // bisection's own halting width, not by the model.
        assert!(
            (upper[0] - expected).abs() <= 1e-9,
            "the bisected rank-1 bound must equal ln γ + ln((1−f)/f): γ={gamma:e}, \
             f={f}, odds={odds:e}, expected={expected}, got={}",
            upper[0],
        );
        // The retained edf at the bound IS f, by construction. Asserted
        // separately because it is the statement a reader cares about: the
        // data keeps fraction f of the posterior precision there.
        let retained = gamma / (gamma + upper[0].exp());
        assert!(
            (retained - f).abs() <= 1e-9,
            "at ρ* the data's share of posterior precision must be exactly f: \
             γ={gamma:e}, retained={retained}, f={f}"
        );
    }
}

/// `EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION` is NOT inert, and no single fixture
/// can show that it is (#2612).
///
/// #2612 measured `Δ = 0.0016` nats of held-out log-loss between `f = 0.50` and
/// `f = 0.90` on penguins and read it as "this constant is inert". The reading
/// does not follow, and the same doc block says why: this floor manufactures an
/// UPPER ρ bound, and on that split four of the eight live null-space λ sit
/// exactly on the box's LOWER wall. REML is railed at the least-smoothed value
/// the box allows and pushing further, so a ceiling above it cannot bind. A
/// fixture where the bound provably cannot bind measures the FIXTURE, not the
/// constant — the same shape as a null fixture being asked to carry a bar.
///
/// So measure the constant where it acts, through the production edf curve.
/// For a rank-1 penalty `edf(ρ) = γ/(γ + e^ρ)` and the target is exactly `f`, so
/// the wall sits at `ρ*(f) = ln γ + ln((1 − f)/f)` and
///
/// ```text
///   ρ*(0.50) − ρ*(0.90) = ln(1/1) − ln(1/9) = ln 9 = 2.1972
/// ```
///
/// INDEPENDENTLY OF γ. That is the part that matters: the effect is scale-free,
/// so it cannot be made small by choosing a fixture, and a fixture that reports
/// it as small has reported something about itself. At those two walls the data
/// retains 50% versus 90% of the posterior precision along the linear direction
/// of every smooth — a 0.40 difference in retained edf, asserted below against
/// `unit_weight_term_edf`, the same function the bisection uses.
///
/// This test is also the gate the constant did not have. It reached its third
/// value in one day, each read off a held-out curve that later turned out not to
/// exist at the exact quadrature. Pinning the shipped value to its stated
/// meaning — prior-to-data precision odds of 1:9 — means the next change has to
/// arrive with a measurement that can actually discriminate, on a fixture whose
/// optimum is not railed at the opposite wall.
#[test]
fn the_relative_floor_is_scale_free_so_no_single_fixture_can_call_it_inert_2612() {
    let f_shipped = EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION;
    let f_control: f64 = 0.50;

    // The shipped value IS an odds statement; say which one, so a silent retune
    // fails here rather than in a log-loss table six months later.
    let odds = (1.0 - f_shipped) / f_shipped;
    assert!(
        (odds - 1.0 / 9.0).abs() <= 1e-12,
        "EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION = {f_shipped} encodes prior-to-data \
         precision odds of {odds:e}; the shipped meaning is 1:9. Changing it \
         changes an estimand-bearing wall by ln((1-f)/f) and must arrive with a \
         measurement on a fixture where an UPPER rho bound can bind — penguins \
         cannot, it is railed on the lower wall (#2612)."
    );

    let mut walls = Vec::new();
    for gamma in [1.0e-4_f64, 1.0e-2, 1.0, 1.0e2, 1.0e4] {
        let gammas = [gamma];
        let rho_shipped = gamma.ln() + ((1.0 - f_shipped) / f_shipped).ln();
        let rho_control = gamma.ln() + ((1.0 - f_control) / f_control).ln();

        // Production edf curve, not the closed form: this is what the bisection
        // in `effective_df_floor_bound` actually reads.
        let edf_shipped =
            unit_weight_term_edf(&gammas, rho_shipped).expect("edf at the shipped wall");
        let edf_control =
            unit_weight_term_edf(&gammas, rho_control).expect("edf at the control wall");

        assert!(
            (edf_shipped - f_shipped).abs() <= 1e-9 && (edf_control - f_control).abs() <= 1e-9,
            "the production edf curve must retain exactly f at the wall f selects: \
             gamma={gamma:e}, edf(rho*(0.90))={edf_shipped}, edf(rho*(0.50))={edf_control}"
        );

        // The measurement that refutes "inert": 40% of the direction's posterior
        // precision separates the two settings, at every scale tested.
        assert!(
            (edf_shipped - edf_control - 0.40).abs() <= 1e-9,
            "retained-precision gap between f=0.90 and f=0.50 must be 0.40: \
             gamma={gamma:e}, got {}",
            edf_shipped - edf_control
        );
        walls.push(rho_control - rho_shipped);
    }

    // Scale-free: the wall movement is ln 9 for every gamma, so a fixture cannot
    // shrink it. If this ever varies with gamma, the closed form above is wrong
    // and the odds reading of `f` goes with it.
    let ln9 = 9.0_f64.ln();
    for (shift, gamma) in walls.iter().zip([1.0e-4_f64, 1.0e-2, 1.0, 1.0e2, 1.0e4]) {
        assert!(
            (shift - ln9).abs() <= 1e-12,
            "rho*(0.50) - rho*(0.90) must equal ln 9 = {ln9} independently of gamma: \
             gamma={gamma:e}, got {shift}"
        );
    }
}
