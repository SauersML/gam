#![cfg(test)]
//! Gates for the follow-up-varying likelihood domain (#2765 / #2767).
//!
//! The family is a transformation model, so its row log-density carries
//! `log η′₁` and its DOMAIN is `η′₁ > 0` at every event row, with
//!
//! ```text
//!   η′₁(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz,    c = √(1 + bᵀΣb).
//! ```
//!
//! With a time-CONSTANT slope the last two terms vanish and `η′₁ = q′·c ≥ q′`,
//! so the time block's linear guard implies the condition and there is nothing
//! for a separate rule to do. A follow-up-varying slope breaks that implication
//! — `q·c′` and `b′ᵀz` carry no sign and read three blocks — and the condition
//! became a refusal at the row evaluator instead of a feasible set the solver
//! could steer by.
//!
//! These gates pin the two halves of putting it back where a solver can use it,
//! and pin the property that makes them trustworthy: the limiter and the row
//! program's own admission are the SAME arithmetic, so they cannot disagree
//! about which side of the boundary a coefficient is on.

use super::*;
use crate::custom_family::CustomFamily;
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;

const N_ROWS: usize = 16;

fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
            .expect("a 1x1 unit latent-score covariance"),
    )
}

/// The slope exit design: an intercept and one covariate, which is the
/// tensored shape the acceptance fixture's `slope_formula = "1"` produces
/// against a two-column time margin.
fn slope_exit_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 1.0,
            _ => t,
        }
    })
}

/// The entry channel: the same covariate chart read at the row's ENTRY time, so
/// it differs from the exit channel without being unrelated to it.
fn slope_entry_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 1.0,
            _ => 0.6 * t,
        }
    })
}

/// The exit-RATE channel `∂/∂t` of the exit design. It carries no offset — the
/// property that makes `β_g = 0` a provably interior point.
fn slope_rate_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 0.0,
            _ => 0.8 + 0.3 * t,
        }
    })
}

fn marginal_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 1), |(row, _)| {
        0.20 + 0.35 * (row as f64) / (N_ROWS as f64)
    })
}

/// Every row is an event, because a censored row's density has no `log η′₁`
/// factor and therefore imposes no domain condition at all — a fixture of
/// censored rows would gate nothing.
fn events() -> Array1<f64> {
    Array1::from_elem(N_ROWS, 1.0)
}

fn latent_scores() -> Array1<f64> {
    Array1::from_shape_fn(N_ROWS, |row| {
        -1.4 + 2.8 * (row as f64) / ((N_ROWS - 1) as f64)
    })
}

/// A family with a positive baseline time derivative and no time or baseline
/// chart of its own, so `q′` is exactly the offset below and the ONLY thing
/// that can drive `η′₁` negative is the slope's follow-up motion — which is
/// what these gates are about.
fn family(frame_is_follow_up_varying: bool) -> SurvivalMarginalSlopeFamily {
    let layout: SlopeLayout = DesignMatrix::from(slope_exit_design()).into();
    let slope_layout = if frame_is_follow_up_varying {
        layout
            .with_follow_up(
                DesignMatrix::from(slope_entry_design()),
                DesignMatrix::from(slope_rate_design()),
            )
            .expect("a shared slope layout accepts a follow-up margin")
    } else {
        layout
    };
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: N_ROWS,
        entry_at_origin: Arc::new(Array1::from_elem(N_ROWS, false)),
        event: Arc::new(events()),
        weights: Arc::new(Array1::from_elem(N_ROWS, 1.0)),
        z: Arc::new(latent_scores().insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::new(None, None)
            .expect("a family with no baseline chart and no learned sigma"),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        offset_entry: Arc::new(Array1::from_elem(N_ROWS, -0.9)),
        offset_exit: Arc::new(Array1::from_elem(N_ROWS, -0.3)),
        derivative_offset_exit: Arc::new(Array1::from_elem(N_ROWS, 1.0)),
        marginal_design: DesignMatrix::from(marginal_design()),
        slope_layout,
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
    }
}

fn states(family: &SurvivalMarginalSlopeFamily, slope_beta: Array1<f64>) -> Vec<ParameterBlockState> {
    let marginal_beta = ndarray::array![0.30];
    let marginal = family.marginal_design.to_dense().to_owned();
    let slope = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(family.n),
        },
        ParameterBlockState {
            eta: marginal.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: slope.dot(&slope_beta),
            beta: slope_beta,
        },
    ]
}

/// An interior slope coefficient: a moderate level and a mild follow-up
/// trend.
fn interior_slope_beta() -> Array1<f64> {
    ndarray::array![0.20, -0.05]
}

/// The step direction that leaves the domain: it drives the slope's follow-up
/// RATE hard, which is the channel a time-constant slope does not have.
fn exiting_direction() -> Array1<f64> {
    // (time: 0 coefficients | marginal: 1 | slope: 2)
    ndarray::array![0.0, 0.0, -6.0]
}

/// Evaluate the row program's OWN admission at a coefficient, the way the
/// likelihood does. This is the oracle the limiter is graded against: the point
/// of the design is that these two never disagree.
fn row_program_admits(
    family: &SurvivalMarginalSlopeFamily,
    states: &[ParameterBlockState],
) -> Result<(), String> {
    for row in 0..family.n {
        let inputs = rigid_row_inputs(family, states, row, "follow-up domain gate")?;
        let primaries = rigid_row_kernel_primaries::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            family, states, row,
        )?;
        let [neg_eta0, neg_eta1, adjusted_derivative] =
            rigid_row_admission_witnesses::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                &primaries, &inputs,
            );
        validate_rigid_row_admission::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            primaries[PRIMARY_QD1],
            &inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
    }
    Ok(())
}

fn margin_along(
    family: &SurvivalMarginalSlopeFamily,
    base: &[ParameterBlockState],
    direction: &Array1<f64>,
    alpha: f64,
) -> f64 {
    let moved = family
        .displaced_block_states(base, direction, alpha)
        .expect("the fixture's blocks and direction agree in width");
    family
        .follow_up_domain_margin(&moved)
        .expect("the fixture is on the follow-up-varying frame")
        .expect("the follow-up frame reports a margin")
}

/// The fixture has to actually exercise the thing: the base point is interior
/// and the whole step is not. Asserted separately so a later edit that makes
/// the fixture trivial fails HERE, naming the fixture, rather than making the
/// gates below vacuously true.
/// The likelihood the trust region scores a trial on (`log_likelihood_only`)
/// must be the likelihood whose gradient and Hessian the step was built from.
/// On the follow-up-varying frame the row kernel's `η′₁` carries `q·c′` and
/// `b′ᵀz`; a value-only path that reads the time-constant closed form drops
/// both, and then no step that varies the slope can be rewarded (#2765). The
/// time-constant frame is the control: there the closed form IS the kernel.
#[test]
fn the_value_only_likelihood_is_the_frame_kernels_likelihood_2765() {
    for frame_is_follow_up_varying in [false, true] {
        let family = family(frame_is_follow_up_varying);
        let base = states(&family, interior_slope_beta());
        let direction = exiting_direction();
        // The second probe moves the slope along the exiting direction but
        // stays inside the follow-up domain: halve the step until the
        // follow-up frame reports a positive margin (the time-constant frame
        // has no margin to report and takes the same step).
        let mut interior_t = 0.25_f64;
        if frame_is_follow_up_varying {
            let mut halvings = 0;
            while margin_along(&family, &base, &direction, interior_t) <= 0.0 {
                interior_t *= 0.5;
                halvings += 1;
                assert!(halvings < 20, "no interior probe found along the exiting direction");
            }
        }
        for t in [0.0_f64, interior_t] {
            let point = family
                .displaced_block_states(&base, &direction, t)
                .expect("width agreement");
            if frame_is_follow_up_varying {
                assert!(
                    margin_along(&family, &base, &direction, t) > 0.0,
                    "the probe must stay interior at t={t}"
                );
            }
            let value_only = family
                .log_likelihood_only(&point)
                .expect("value-only likelihood");
            let mut kernel = 0.0_f64;
            for row in 0..N_ROWS {
                let (nll, _, _) = family
                    .compute_row_primary_gradient_hessian_uncached(row, &point)
                    .expect("frame kernel row");
                kernel -= nll;
            }
            let scale = value_only.abs().max(kernel.abs()).max(1.0);
            assert!(
                (value_only - kernel).abs() <= 1e-10 * scale,
                "follow_up_varying={frame_is_follow_up_varying} t={t}: value-only likelihood \
                 {value_only:.12e} differs from the frame kernel's {kernel:.12e}"
            );
        }
    }
}

/// A row entering at the time origin has `S(0) = 1` and no entry factor
/// (gnomon#2336). The value-only likelihood must drop it on every frame, exactly
/// as the frame kernel does, or the trust region scores a trial with the factor
/// against a step built without it.
#[test]
fn the_value_only_likelihood_drops_the_origin_entry_factor_like_the_frame_kernel_2336() {
    for frame_is_follow_up_varying in [false, true] {
        let delayed = family(frame_is_follow_up_varying);
        let mut landmarked = family(frame_is_follow_up_varying);
        landmarked.entry_at_origin = Arc::new(Array1::from_shape_fn(N_ROWS, |row| row % 2 == 0));
        let point = states(&landmarked, interior_slope_beta());
        let value_only = landmarked
            .log_likelihood_only(&point)
            .expect("value-only likelihood");
        let mut kernel = 0.0_f64;
        for row in 0..N_ROWS {
            let (nll, _, _) = landmarked
                .compute_row_primary_gradient_hessian_uncached(row, &point)
                .expect("frame kernel row");
            kernel -= nll;
        }
        let scale = value_only.abs().max(kernel.abs()).max(1.0);
        assert!(
            (value_only - kernel).abs() <= 1e-10 * scale,
            "follow_up_varying={frame_is_follow_up_varying}: value-only likelihood \
             {value_only:.12e} differs from the frame kernel's {kernel:.12e}"
        );
        // The entry factor adds `−log Φ(−η₀) > 0` to a delayed row's
        // log-likelihood, so dropping it must lower the total.
        let delayed_value = delayed
            .log_likelihood_only(&point)
            .expect("value-only likelihood");
        assert!(
            value_only < delayed_value,
            "follow_up_varying={frame_is_follow_up_varying}: the origin rows kept their entry \
             factor; landmarked {value_only:.12e}, delayed {delayed_value:.12e}"
        );
    }
}

#[test]
fn the_fixture_step_really_leaves_the_domain_2765() {
    let family = family(true);
    let base = states(&family, interior_slope_beta());
    let direction = exiting_direction();
    let base_margin = margin_along(&family, &base, &direction, 0.0);
    let full_margin = margin_along(&family, &base, &direction, 1.0);
    assert!(
        base_margin > 0.0,
        "the base coefficient must be interior; min η′₁ = {base_margin:.6e}"
    );
    assert!(
        full_margin < 0.0,
        "the whole step must leave the domain; min η′₁ = {full_margin:.6e}"
    );
    row_program_admits(&family, &base).expect("the row program admits the interior base");
    let outside = family
        .displaced_block_states(&base, &direction, 1.0)
        .expect("width agreement");
    let error = row_program_admits(&family, &outside)
        .expect_err("the row program must refuse the exterior endpoint");
    assert!(
        error.contains("transformed time derivative must be positive"),
        "the refusal must be the follow-up domain condition: {error}"
    );
}

#[test]
fn follow_up_value_program_preserves_the_newton_value_without_derivative_work_2765() {
    use std::hint::black_box;
    use std::time::Instant;

    let family = family(true);
    let state = states(&family, interior_slope_beta());
    let rows: Vec<_> = (0..N_ROWS)
        .map(|row| {
            let inputs = rigid_row_inputs(&family, &state, row, "value program regression")
                .expect("row inputs");
            let primaries = rigid_row_kernel_primaries::<
                DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry,
            >(&family, &state, row).expect("dynamic primaries");
            (primaries, inputs)
        })
        .collect();
    for (primaries, inputs) in &rows {
        let scalar = rigid_row_value::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            primaries, inputs,
        ).expect("scalar row");
        let (newton, _, _) = rigid_row_order2::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            primaries, inputs,
        ).expect("Newton row");
        assert!((scalar - newton).abs() <= 16.0 * f64::EPSILON * newton.abs().max(1.0),
            "the value carrier must evaluate the Newton objective: {scalar:e} versus {newton:e}");
    }

    // Report both implementations in one binary on identical inputs. Timing is
    // evidence for iteration cost, while the assertions above grade correctness
    // independently of host load and compiler profile.
    let start = Instant::now();
    for _ in 0..1024 {
        for (primaries, inputs) in &rows {
            let (value, _, _) = rigid_row_order2::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                black_box(primaries), black_box(inputs),
            ).expect("Newton row");
            black_box(value);
        }
    }
    let newton_elapsed = start.elapsed();
    let start = Instant::now();
    for _ in 0..1024 {
        for (primaries, inputs) in &rows {
            black_box(rigid_row_value::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                black_box(primaries), black_box(inputs),
            ).expect("scalar row"));
        }
    }
    let scalar_elapsed = start.elapsed();
    eprintln!("[2765-VALUE] rows={} Newton={newton_elapsed:?} scalar={scalar_elapsed:?} speedup={:.3}",
        1024 * N_ROWS, newton_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64());
}

/// The limiter returns a point the ROW PROGRAM admits, at the row barrier's
/// damped fraction and strictly inside the crossing an independent scan finds.
/// Returning an admitted point is the whole purpose: a limiter whose endpoint the
/// likelihood then refuses has moved the problem rather than solved it. The
/// damped fraction is checked against rates measured independently of the
/// limiter's jet, by a test-only central difference of each row's `η′₁` along
/// this non-affine step. The exact margin it keeps is pinned on an affine
/// fixture below.
#[test]
fn the_joint_step_limit_lands_on_a_point_the_row_program_admits_2765() {
    let family = family(true);
    let base = states(&family, interior_slope_beta());
    let direction = exiting_direction();

    let alpha = family
        .max_feasible_follow_up_joint_step(&base, &direction)
        .expect("the follow-up frame answers")
        .expect("a step that leaves the domain must be limited");
    assert!(
        (0.0..1.0).contains(&alpha),
        "a limited step is a fraction strictly below one; got {alpha:.6e}"
    );
    assert!(alpha > 0.0, "the base is interior, so some step is feasible");

    let landed = family
        .displaced_block_states(&base, &direction, alpha)
        .expect("width agreement");
    row_program_admits(&family, &landed).unwrap_or_else(|error| {
        panic!("the row program refused the limiter's own endpoint α={alpha:.9e}: {error}")
    });

    let step = 1e-5;
    let along = |alpha: f64| {
        row_margins(
            &family,
            &family
                .displaced_block_states(&base, &direction, alpha)
                .expect("width agreement"),
        )
    };
    let (start, ahead, behind) = (along(0.0), along(step), along(-step));
    let max_rate = (0..family.n)
        .map(|row| (-(ahead[row] - behind[row]) / (2.0 * step)).max(0.0) / start[row])
        .fold(0.0_f64, f64::max);
    let damped = 1.0 / (1.0 + max_rate);
    assert!(
        margin_along(&family, &base, &direction, damped) > 0.0,
        "the fixture's damped point must be inside the domain"
    );
    assert!(
        (alpha - damped).abs() <= 1e-6 * damped,
        "the limited step must be the damped 1/(1 + max r) = {damped:.12e} from the \
         measured rates; got {alpha:.12e}"
    );

    // Measured against an independent scan rather than asserted from the
    // limiter's internals: the first grid point at which the margin turns
    // non-positive bounds the true crossing from above.
    const GRID: usize = 4096;
    let mut crossing = 1.0_f64;
    for step in 1..=GRID {
        let probe = step as f64 / GRID as f64;
        if !(margin_along(&family, &base, &direction, probe) > 0.0) {
            crossing = probe;
            break;
        }
    }
    assert!(
        alpha < crossing,
        "the limiter must stay strictly inside the scanned crossing: α={alpha:.9e}, \
         crossing={crossing:.9e}"
    );
}

/// Every event row's `η′₁` from the row program's own admission witness.
fn row_margins(family: &SurvivalMarginalSlopeFamily, states: &[ParameterBlockState]) -> Vec<f64> {
    (0..family.n)
        .map(|row| {
            let inputs = rigid_row_inputs(family, states, row, "follow-up row margins")
                .expect("row inputs");
            let primaries = rigid_row_kernel_primaries::<
                DYNAMIC_SLOPE_PRIMARIES,
                DynamicSlopeGeometry,
            >(family, states, row)
            .expect("dynamic primaries");
            let [_, _, adjusted_derivative] =
                rigid_row_admission_witnesses::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                    &primaries, &inputs,
                );
            adjusted_derivative
        })
        .collect()
}

/// The row barrier's damping on rows that are affine along the step (#2765,
/// #2627 docs/marginal-slope.md:205). A marginal-only step moves `q₁` and
/// nothing `c₁` reads, so every row's `η′₁` is affine along it and its rate
/// `r_i = max(0, η′₁,i(0) − η′₁,i(1)) / η′₁,i(0)` is exact from two
/// evaluations, independently of the limiter's jet. The limited step must be
/// `1/(1 + max_i r_i)` and keep every row at `η′₁,i ≥ η′₁,i(0)/(1 + r_i)`. The
/// same holds for a feasible step scaled to `max r = 0.9`, above the
/// self-concordant full-step bound, which must be damped to `1/1.9`, not taken.
///
/// Negative control: the undamped rule, the feasible end of a bisection of
/// `[0, 1]` to the step's representability, which the limiter used before.
/// Its endpoint sits within roundoff of the edge and breaks the bound.
#[test]
fn the_joint_step_limit_keeps_every_affine_row_a_damped_share_of_its_margin_2765() {
    let family = family(true);
    let base = states(&family, interior_slope_beta());
    let direction = ndarray::array![400.0, 0.0, 0.0];
    let moved = |alpha: f64| {
        family
            .displaced_block_states(&base, &direction, alpha)
            .expect("width agreement")
    };
    let start = row_margins(&family, &base);
    let whole = row_margins(&family, &moved(1.0));
    let half = row_margins(&family, &moved(0.5));
    for row in 0..family.n {
        let chord = 0.5 * (start[row] + whole[row]);
        assert!(
            (half[row] - chord).abs() <= 1e-12 * start[row].abs().max(whole[row].abs()),
            "the fixture's rows must be affine along the step: row {row} has η′₁ {:.15e} \
             at the midpoint against the chord's {chord:.15e}",
            half[row]
        );
    }
    assert!(
        start.iter().all(|&margin| margin > 0.0),
        "the base must be interior at every row"
    );
    assert!(
        whole.iter().any(|&margin| margin < 0.0),
        "the whole step must leave the domain at some row"
    );
    let rates: Vec<f64> = start
        .iter()
        .zip(whole.iter())
        .map(|(&from, &to)| (from - to).max(0.0) / from)
        .collect();
    let max_rate = rates.iter().copied().fold(0.0_f64, f64::max);

    let alpha = family
        .max_feasible_follow_up_joint_step(&base, &direction)
        .expect("the follow-up frame answers")
        .expect("a step that leaves the domain must be limited");
    let damped = 1.0 / (1.0 + max_rate);
    assert!(
        (alpha - damped).abs() <= 1e-12 * damped,
        "the limited step must be the damped 1/(1 + max r) = {damped:.15e}; got {alpha:.15e}"
    );
    let landed = row_margins(&family, &moved(alpha));
    for row in 0..family.n {
        let floor = start[row] / (1.0 + rates[row]);
        assert!(
            landed[row] >= floor * (1.0 - 1e-12),
            "row {row} must keep η′₁ ≥ η′₁(0)/(1 + r) = {floor:.6e}; got {:.6e}",
            landed[row]
        );
    }

    let step_scale = direction.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let beta_scale = base
        .iter()
        .flat_map(|state| state.beta.iter())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let resolution = beta_scale * f64::EPSILON / step_scale;
    let (mut feasible, mut infeasible) = (0.0_f64, 1.0_f64);
    while infeasible - feasible > resolution {
        let midpoint = 0.5 * (feasible + infeasible);
        if midpoint <= feasible || midpoint >= infeasible {
            break;
        }
        if margin_along(&family, &base, &direction, midpoint) > 0.0 {
            feasible = midpoint;
        } else {
            infeasible = midpoint;
        }
    }
    // A FEASIBLE whole step is damped too once it consumes more of a row's margin
    // than the self-concordant full-step bound `(3 − √5)/2`: at `max r = 0.9`
    // taking it would leave that row at a tenth of its margin, and the next
    // iterate would start at the edge.
    let scale = 0.9 / max_rate;
    let consuming = &direction * scale;
    let consumed = family
        .displaced_block_states(&base, &consuming, 1.0)
        .expect("width agreement");
    assert!(
        row_margins(&family, &consumed).iter().all(|&margin| margin > 0.0),
        "the scaled step must stay inside the domain"
    );
    let consuming_alpha = family
        .max_feasible_follow_up_joint_step(&base, &consuming)
        .expect("the follow-up frame answers")
        .expect("a feasible step that consumes 0.9 of a row's margin must be damped");
    assert!(
        (consuming_alpha - 1.0 / 1.9).abs() <= 1e-12,
        "the damped fraction of a max-rate-0.9 step is 1/1.9; got {consuming_alpha:.15e}"
    );
    let consumed_landed = row_margins(
        &family,
        &family
            .displaced_block_states(&base, &consuming, consuming_alpha)
            .expect("width agreement"),
    );
    for row in 0..family.n {
        let floor = start[row] / (1.0 + scale * rates[row]);
        assert!(
            consumed_landed[row] >= floor * (1.0 - 1e-12),
            "row {row} must keep η′₁ ≥ η′₁(0)/(1 + r) = {floor:.6e} on the feasible step; \
             got {:.6e}",
            consumed_landed[row]
        );
    }

    let edge = row_margins(&family, &moved(feasible));
    let broken = (0..family.n).any(|row| edge[row] < start[row] / (1.0 + rates[row]));
    let edge_margin = edge.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        broken && edge_margin < 1e-9 * start.iter().copied().fold(f64::INFINITY, f64::min),
        "negative control: the undamped feasible end must sit at the edge and break the \
         damped bound; min η′₁ there = {edge_margin:.6e}"
    );
}

/// The backstop damps by the chord to the exact crossing when a row falls faster
/// than its tangent (#2765, #2627 docs/marginal-slope.md:205). A marginal step with
/// a slope-level step makes `q₁·b₁` quadratic along it, so the rows are concave and
/// cross before the tangent's damped point, which the exact witness then refuses.
/// The step must be `α_c/(1 + α_c)`, with `α_c` the exact crossing, and every row
/// concave on `[0, α_c]` keeps `η′₁ ≥ η′₁(0)·α_c/(1 + α_c)`. Negative control: the
/// crossing itself, the old backstop's answer, sits at the edge.
#[test]
fn the_joint_step_limit_damps_a_concave_row_by_its_chord_to_the_crossing_2765() {
    let family = family(true);
    let base = states(&family, interior_slope_beta());
    let direction = ndarray::array![400.0, 2.0, 0.0];
    let along = |alpha: f64| {
        row_margins(
            &family,
            &family
                .displaced_block_states(&base, &direction, alpha)
                .expect("width agreement"),
        )
    };
    let step = 1e-5;
    let (start, ahead, behind) = (along(0.0), along(step), along(-step));
    let max_rate = (0..family.n)
        .map(|row| (-(ahead[row] - behind[row]) / (2.0 * step)).max(0.0) / start[row])
        .fold(0.0_f64, f64::max);
    let tangent_damped = 1.0 / (1.0 + max_rate);
    assert!(
        margin_along(&family, &base, &direction, tangent_damped) <= 0.0,
        "the fixture's rows must cross before the tangent's damped point α={tangent_damped:.6e}"
    );

    let step_scale = direction.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let beta_scale = base
        .iter()
        .flat_map(|state| state.beta.iter())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let resolution = beta_scale * f64::EPSILON / step_scale;
    let (mut crossing, mut outside) = (0.0_f64, tangent_damped);
    while outside - crossing > resolution {
        let midpoint = 0.5 * (crossing + outside);
        if midpoint <= crossing || midpoint >= outside {
            break;
        }
        if margin_along(&family, &base, &direction, midpoint) > 0.0 {
            crossing = midpoint;
        } else {
            outside = midpoint;
        }
    }
    let at_crossing = along(crossing);
    let half = along(0.5 * crossing);
    for row in 0..family.n {
        assert!(
            half[row] >= 0.5 * (start[row] + at_crossing[row]),
            "the fixture's row {row} must be concave on [0, α_c]"
        );
    }

    let alpha = family
        .max_feasible_follow_up_joint_step(&base, &direction)
        .expect("the follow-up frame answers")
        .expect("a step that leaves the domain must be limited");
    let chord = crossing / (1.0 + crossing);
    assert!(
        (alpha - chord).abs() <= 1e-9 * chord,
        "the backstop must damp by the chord to the crossing, α_c/(1 + α_c) = {chord:.12e}; \
         got {alpha:.12e}"
    );
    let landed = along(alpha);
    for row in 0..family.n {
        let floor = start[row] * crossing / (1.0 + crossing);
        assert!(
            landed[row] >= floor * (1.0 - 1e-9),
            "row {row} must keep η′₁ ≥ η′₁(0)·α_c/(1 + α_c) = {floor:.6e}; got {:.6e}",
            landed[row]
        );
    }
    let edge = at_crossing.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        edge < 1e-9 * start.iter().copied().fold(f64::INFINITY, f64::min),
        "negative control: the crossing itself sits at the edge; min η′₁ there = {edge:.6e}"
    );
}

/// The backstop repeats instead of returning a bisection end (#2765). The margin
/// dips below zero on `(0.1, 0.45)` and comes back until `0.8`. The first bisection
/// of `[0, 1]` finds the LATER sign change at `0.8`, and its chord point `0.8/1.8`
/// lies in the dip and is refused. The second round finds the first crossing at
/// `0.1` and damps by its chord, `0.1/1.1`, which is admitted with a strictly
/// positive margin.
#[test]
fn the_backstop_repeats_below_a_refused_chord_point_2765() {
    let (a, b, c) = (0.1_f64, 0.45_f64, 0.8_f64);
    let margin = |alpha: f64| -> Result<f64, String> {
        Ok(-(alpha - a) * (alpha - b) * (alpha - c) / (a * b * c))
    };
    assert!(margin(1.0).expect("margin") <= 0.0, "the right end must be refused");
    assert!(
        margin(0.5).expect("margin") > 0.0,
        "the margin must come back above zero past the dip"
    );
    assert!(
        margin(c / (1.0 + c)).expect("margin") < 0.0,
        "the later crossing's chord point must lie in the dip"
    );
    let (alpha, rounds) =
        SurvivalMarginalSlopeFamily::chord_damped_domain_fraction(&margin, 1.0, f64::EPSILON)
            .expect("the backstop answers");
    assert_eq!(rounds, 2, "the first chord point is refused, so a second round must run");
    assert!(
        (alpha - a / (1.0 + a)).abs() <= 1e-12,
        "the second round must damp by the chord to the first crossing, {:.12e}; got \
         {alpha:.12e}",
        a / (1.0 + a)
    );
    assert!(
        margin(alpha).expect("margin") > 0.0,
        "the answer must be admitted with a strictly positive margin"
    );
}

/// A time-constant slope answers `None` on every direction: the rule declines
/// where the time block's own linear guard already implies the domain, so the
/// static path keeps the exact feasible set it always had.
#[test]
fn a_time_constant_slope_declines_the_joint_step_limit_2765() {
    let family = family(false);
    let base = states(&family, interior_slope_beta());
    for scale in [-9.0_f64, -1.0, 0.0, 1.0, 9.0] {
        let direction = &exiting_direction() * scale;
        assert!(
            family
                .max_feasible_follow_up_joint_step(&base, &direction)
                .expect("the static frame answers")
                .is_none(),
            "the static frame must not limit a step (scale {scale})"
        );
        assert!(
            family
                .max_feasible_joint_step_size(&base, &direction)
                .expect("the trait hook answers")
                .is_none(),
            "the trait hook must agree with the rule it delegates to (scale {scale})"
        );
    }
}

/// A step that stays inside, and consumes no row's margin beyond the
/// self-concordant full-step bound `(3 − √5)/2`, is not limited at all. A limiter
/// that answered `Some(1.0)` here would be indistinguishable from one that
/// answered `None` in its effect, but it would route every cycle through the
/// `Scaled` arm and the log line that goes with it; `None` is the honest answer
/// and the gate says so. The rates are measured by a test-only central
/// difference of each row's `η′₁` along the step.
#[test]
fn an_interior_step_is_not_limited_2765() {
    let family = family(true);
    let base = states(&family, interior_slope_beta());
    let inward = &exiting_direction() * -0.02;
    assert!(
        margin_along(&family, &base, &inward, 1.0) > 0.0,
        "the fixture's inward step must stay interior"
    );
    let step = 1e-5;
    let along = |alpha: f64| {
        row_margins(
            &family,
            &family
                .displaced_block_states(&base, &inward, alpha)
                .expect("width agreement"),
        )
    };
    let (start, ahead, behind) = (along(0.0), along(step), along(-step));
    let max_rate = (0..family.n)
        .map(|row| (-(ahead[row] - behind[row]) / (2.0 * step)).max(0.0) / start[row])
        .fold(0.0_f64, f64::max);
    assert!(
        max_rate > 0.0 && max_rate < 0.5 * (3.0 - 5.0_f64.sqrt()),
        "the fixture's inward step must lower some row, by less than the full-step bound; \
         max row rate {max_rate:.6e}"
    );
    assert!(
        family
            .max_feasible_follow_up_joint_step(&base, &inward)
            .expect("the follow-up frame answers")
            .is_none(),
        "an interior step must not be limited"
    );
}

/// The warm-start restoration: a seed outside the domain is retreated toward the
/// slope block's origin until the row program admits it, and a seed already
/// inside is left bit-identical.
#[test]
fn an_exterior_warm_start_is_retreated_into_the_domain_2765() {
    let family = family(true);
    // The seed the outer step left behind: the interior base plus the whole
    // exiting step, i.e. exactly the exterior point the previous gate pinned.
    let exterior_beta = &interior_slope_beta() + &ndarray::array![0.0, -6.0];
    let mut blocks = blocks_for(&family, exterior_beta.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the retreat answers on the follow-up frame");
    assert!(
        fraction > 0.0 && fraction <= 1.0,
        "an exterior seed must be retreated by a real fraction; got {fraction:.6e}"
    );
    let restored = blocks[2]
        .initial_beta
        .as_ref()
        .expect("the retreat writes the seed it restored")
        .clone();
    assert!(
        restored
            .iter()
            .zip(exterior_beta.iter())
            .any(|(a, b)| a != b),
        "the retreat must move the seed it declared exterior"
    );
    row_program_admits(&family, &states(&family, restored.clone())).unwrap_or_else(|error| {
        panic!("the retreated seed is still outside the domain: {error}")
    });
    // The retreat is toward the ORIGIN of the slope block, so every
    // coordinate shrinks by the same factor and none changes sign.
    for (restored_value, seed_value) in restored.iter().zip(exterior_beta.iter()) {
        assert!(
            restored_value.abs() <= seed_value.abs() + f64::EPSILON,
            "a retreat toward the origin cannot grow a coordinate: {restored_value} vs {seed_value}"
        );
        assert!(
            restored_value * seed_value >= 0.0,
            "a retreat toward the origin cannot cross zero: {restored_value} vs {seed_value}"
        );
    }
}

/// The retreat stops at the time block's own guard, not at the domain boundary
/// (#2627, docs/marginal-slope.md:205). The smallest retreat with `η′₁ > 0` is
/// interior by the bisection's resolution only, and `−log η′₁` then starts the
/// inner solve where its gradient scales as `1/η′₁` and its curvature as
/// `1/η′₁²`. The md:205 seeds started at `min η′₁ = 1.9e-14`, and every one was
/// refused at the reduced-face KKT check.
#[test]
fn an_exterior_warm_start_is_retreated_to_the_derivative_guard_2627() {
    let family = family(true);
    let exterior_beta = &interior_slope_beta() + &ndarray::array![0.0, -6.0];
    let seed_margin = family
        .follow_up_domain_margin(&states(&family, exterior_beta.clone()))
        .expect("the margin evaluates")
        .expect("the follow-up frame has a margin");
    assert!(
        seed_margin < 0.0,
        "the fixture seed must be outside the domain; min η′₁ = {seed_margin:.6e}"
    );
    let mut blocks = blocks_for(&family, exterior_beta.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the retreat answers on the follow-up frame");
    let restored = blocks[2]
        .initial_beta
        .as_ref()
        .expect("the retreat writes the seed it restored")
        .clone();
    let restored_margin = family
        .follow_up_domain_margin(&states(&family, restored))
        .expect("the margin evaluates")
        .expect("the follow-up frame has a margin");
    assert!(
        restored_margin >= family.derivative_guard,
        "the retreated seed must meet the derivative guard, not merely the boundary: \
         min η′₁ = {restored_margin:.6e}, guard = {:.6e}",
        family.derivative_guard
    );
    assert!(
        fraction > 0.0 && fraction < 1.0,
        "the guard is met short of the origin, so the retreat keeps part of the warm \
         start; got fraction {fraction:.6e}"
    );
}

#[test]
fn an_interior_warm_start_is_left_alone_2765() {
    let family = family(true);
    let seed = interior_slope_beta();
    let mut blocks = blocks_for(&family, seed.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the retreat answers on the follow-up frame");
    assert_eq!(
        fraction, 0.0,
        "an interior seed is not retreated at all"
    );
    let untouched = blocks[2]
        .initial_beta
        .as_ref()
        .expect("the fixture seeded this block");
    assert_eq!(
        untouched.to_vec(),
        seed.to_vec(),
        "an interior seed must be left bit-identical"
    );
}

/// The static frame never retreats, whatever the seed.
#[test]
fn a_time_constant_warm_start_is_never_retreated_2765() {
    let family = family(false);
    let seed = &interior_slope_beta() + &ndarray::array![0.0, -6.0];
    let mut blocks = blocks_for(&family, seed.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the static frame answers");
    assert_eq!(fraction, 0.0, "the static frame declines the retreat");
    assert_eq!(
        blocks[2]
            .initial_beta
            .as_ref()
            .expect("the fixture seeded this block")
            .to_vec(),
        seed.to_vec(),
        "the static frame must leave the seed bit-identical"
    );
}

fn blocks_for(
    family: &SurvivalMarginalSlopeFamily,
    slope_beta: Array1<f64>,
) -> Vec<ParameterBlockSpec> {
    let state = states(family, slope_beta);
    let designs: [DesignMatrix; 3] = [
        family.design_exit.clone(),
        family.marginal_design.clone(),
        family.slope_layout.coefficient_design().clone(),
    ];
    state
        .into_iter()
        .zip(designs)
        .enumerate()
        .map(|(index, (block_state, design))| ParameterBlockSpec {
            name: format!("follow_up_domain_{index}"),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(
                design.to_dense().to_owned(),
            )),
            offset: Array1::zeros(N_ROWS),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(block_state.beta),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect()
}

/// The Jeffreys reference values the candidate objectives below are built from, mirrored from
/// gam-solve `reml::jeffreys_subspace`, where they are crate-private: `REDUCED_INFO_ABSOLUTE_FLOOR`,
/// `REDUCED_INFO_RELATIVE_FLOOR`, and `CONDITIONING_GATE_ABSOLUTE_CLEAR`, the top saturation `Λ`
/// that `jeffreys_cap` reads.
const JEFFREYS_ABSOLUTE_FLOOR: f64 = 1e-12;
const JEFFREYS_RELATIVE_FLOOR: f64 = 1e-10;
const JEFFREYS_TOP_SATURATION: f64 = 16.0;

/// `g(λ)` on `jeffreys_antiderivative`'s four branches, with the bottom floor and the top
/// saturation passed in so one spectrum can be scored under each candidate objective.
/// `cap = +∞` removes the top saturation.
fn jeffreys_candidate_value(lambda: f64, floor: f64, cap: f64) -> f64 {
    if lambda >= cap {
        cap.ln() + 1.0 - cap / lambda
    } else if lambda >= floor {
        lambda.ln()
    } else if lambda >= 0.0 {
        lambda / floor + floor.ln() - 1.0
    } else {
        floor.ln() - 1.0 + lambda / (floor - lambda)
    }
}

/// `g′(λ)` on the same four branches.
fn jeffreys_candidate_slope(lambda: f64, floor: f64, cap: f64) -> f64 {
    if lambda >= cap {
        cap / (lambda * lambda)
    } else if lambda >= floor {
        1.0 / lambda
    } else if lambda >= 0.0 {
        1.0 / floor
    } else {
        floor / ((floor - lambda) * (floor - lambda))
    }
}

/// `Φ = ½ Σ g(λ_i)` for one candidate objective, and the roundoff one eigensolve leaves on it.
/// Every `λ_i` is known only to `m·ε·λ_max`, and `g′` carries that into `Φ`; this is the bound
/// `JointJeffreysPlan::value_roundoff_bound` states for the production value.
fn jeffreys_candidate(spectrum: &[f64], floor: f64, cap: f64) -> (f64, f64) {
    let lambda_max = spectrum.iter().fold(0.0_f64, |acc, &lambda| acc.max(lambda));
    let perturbation = spectrum.len() as f64 * f64::EPSILON * lambda_max.max(floor);
    let phi = 0.5
        * spectrum
            .iter()
            .map(|&lambda| jeffreys_candidate_value(lambda, floor, cap))
            .sum::<f64>();
    let sensitivity = 0.5
        * spectrum
            .iter()
            .map(|&lambda| jeffreys_candidate_slope(lambda, floor, cap).abs())
            .sum::<f64>();
    (phi, perturbation * sensitivity)
}

/// One point of a walk into the follow-up face, read the way a fit reads it.
struct FacePoint {
    margin: f64,
    neg_log_likelihood: f64,
    spectrum: Vec<f64>,
    production_phi: f64,
    production_roundoff: f64,
    production_gate: f64,
}

fn face_point(
    family: &SurvivalMarginalSlopeFamily,
    base: &[ParameterBlockState],
    direction: &Array1<f64>,
    alpha: f64,
) -> FacePoint {
    let moved = family
        .displaced_block_states(base, direction, alpha)
        .expect("the fixture's blocks and direction agree in width");
    let margin = family
        .follow_up_domain_margin(&moved)
        .expect("the fixture is on the follow-up-varying frame")
        .expect("the follow-up frame reports a margin");
    let neg_log_likelihood = -family
        .log_likelihood_only(&moved)
        .expect("an interior point has a likelihood");
    let specs = blocks_for(family, moved[2].beta.clone());
    let information = family
        .joint_jeffreys_information_with_specs(&moved, &specs)
        .expect("an interior point has a joint information")
        .expect("the family serves its joint information");
    // This family states neither a span basis nor an aggregate penalty, so
    // `build_joint_jeffreys_subspace` gives the Jeffreys term the whole coefficient space.
    let span = Array2::<f64>::eye(information.nrows());
    let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        information.view(),
        span.view(),
    )
    .expect("the reduced information has a spectrum");
    let strength = family.joint_jeffreys_term_strength();
    let basis = plan.ambient_eigenbasis();
    let mut spectrum: Vec<f64> = (0..basis.ncols())
        .map(|index| {
            let column = basis.column(index);
            column.dot(&information.dot(&column))
        })
        .collect();
    spectrum.sort_by(f64::total_cmp);
    FacePoint {
        margin,
        neg_log_likelihood,
        spectrum,
        production_phi: plan.value() * strength,
        production_roundoff: plan.value_roundoff_bound() * strength.abs(),
        production_gate: plan.conditioning_gate_weight(),
    }
}

/// `dV/d(−ln ε)` with `V = −ℓ − Φ`, read between the third-last and the last decade that
/// `admitted` accepts, returned with those two decades' indices. `None` when fewer than three
/// decades are admitted.
fn barrier_rate(
    points: &[FacePoint],
    phi: &[f64],
    admitted: &dyn Fn(usize) -> bool,
) -> Option<(f64, usize, usize)> {
    let decades: Vec<usize> = (0..points.len()).filter(|&index| admitted(index)).collect();
    if decades.len() < 3 {
        return None;
    }
    let first = decades[decades.len() - 3];
    let last = decades[decades.len() - 1];
    let objective = |index: usize| points[index].neg_log_likelihood - phi[index];
    let rate = (objective(last) - objective(first))
        / (points[first].margin.ln() - points[last].margin.ln());
    Some((rate, first, last))
}

/// The armed Jeffreys objective keeps the likelihood's barrier at the follow-up face (#2765).
///
/// This is the face-boundedness gate on slice 3 of the #979 Jeffreys ruling (b): `G ≡ 1` inside
/// an armed fit, with the gate, its ramps and the relative floor deleted.
///
/// An event row carries `−ln ε` with `ε = η′₁`, and its curvature `uuᵀ/ε² − B/ε`
/// (`u = ∇η′₁`, `B = ∇²η′₁`) drives the reduced information without limit as a walk reaches
/// the face. One eigenvalue grows like `|u|²/ε²`. On `u⊥` the Schur complement is
/// `−B⊥/ε + O(1)`, so `r` eigenvalues grow like `|μ|/ε`, one for each negative eigenvalue `μ` of
/// `B⊥`, and the rest stay `O(1)` or fall like `−μ/ε`. The objective `V = −ℓ − Φ` with
/// `Φ = ½ Σ g(λ_i)` keeps its barrier exactly when `Φ` grows slower than `−ln ε`. Each
/// objective's rate `dV/d(−ln ε)`:
///
/// * armed: `G ≡ 1`, absolute floor, top saturation `Λ`. `g ≤ ln Λ + 1`, so `Φ` is bounded and
///   the rate is `1`;
/// * no floor at all, `Λ` kept: the same top bound, so the rate is `1` wherever the value exists
///   (a positive spectrum);
/// * production: `G·½Σg` with the relative floor and the floor-collapse factor. `G = 0` past the
///   band, so the rate is `1`;
/// * control, `Λ` removed: `½ ln λ₁` cancels the barrier and the rate is `−r/2`;
/// * control, relative floor `1e-10·λ_max` with `G ≡ 1` (#2919): past `floor = Λ` every `g` is
///   `ln floor ± 1`, and the rate is `1 − m`.
///
/// The controls must fail the rate-½ predicate the other objectives pass, or the predicate
/// measures nothing. Each rate is read over the last three decades of `ε` at which one
/// eigensolve's roundoff on that objective's `Φ` stays below a twentieth of a decade of barrier;
/// the relative-floor control also has to be past `floor = Λ` there. The table prints before any
/// assertion.
#[test]
fn the_armed_jeffreys_objective_keeps_the_follow_up_barrier_2765() {
    const DECADES: usize = 10;
    const GRID: usize = 4096;
    let family = family(true);
    assert!(
        family
            .jeffreys_span_basis()
            .expect("the span basis hook answers")
            .is_none()
            && family
                .jeffreys_span_aggregate_penalty()
                .expect("the aggregate penalty hook answers")
                .is_none(),
        "the walk scores the whole coefficient space, which is this family's Jeffreys span only \
         while it states neither a span basis nor an aggregate penalty"
    );
    let base = states(&family, interior_slope_beta());
    let direction = exiting_direction();
    let base_margin = margin_along(&family, &base, &direction, 0.0);
    // The first grid point outside the domain brackets the face the walk approaches.
    let mut outside = 1.0_f64;
    for step in 1..=GRID {
        let probe = step as f64 / GRID as f64;
        if !(margin_along(&family, &base, &direction, probe) > 0.0) {
            outside = probe;
            break;
        }
    }
    let points: Vec<FacePoint> = (1..=DECADES)
        .map(|decade| {
            let target = base_margin * 10.0_f64.powi(-(decade as i32));
            let mut inside = 0.0_f64;
            let mut beyond = outside;
            for _ in 0..128 {
                let middle = 0.5 * (inside + beyond);
                if margin_along(&family, &base, &direction, middle) > target {
                    inside = middle;
                } else {
                    beyond = middle;
                }
            }
            face_point(&family, &base, &direction, inside)
        })
        .collect();

    let armed: Vec<(f64, f64)> = points
        .iter()
        .map(|point| {
            jeffreys_candidate(&point.spectrum, JEFFREYS_ABSOLUTE_FLOOR, JEFFREYS_TOP_SATURATION)
        })
        .collect();
    let no_floor: Vec<(f64, f64)> = points
        .iter()
        .map(|point| {
            if point.spectrum[0] > 0.0 {
                jeffreys_candidate(&point.spectrum, 0.0, JEFFREYS_TOP_SATURATION)
            } else {
                (f64::NAN, f64::NAN)
            }
        })
        .collect();
    let no_cap: Vec<(f64, f64)> = points
        .iter()
        .map(|point| jeffreys_candidate(&point.spectrum, JEFFREYS_ABSOLUTE_FLOOR, f64::INFINITY))
        .collect();
    let relative_floors: Vec<f64> = points
        .iter()
        .map(|point| {
            let lambda_max = point
                .spectrum
                .iter()
                .fold(0.0_f64, |acc, &lambda| acc.max(lambda));
            (JEFFREYS_RELATIVE_FLOOR * lambda_max).max(JEFFREYS_ABSOLUTE_FLOOR)
        })
        .collect();
    let relative_floor: Vec<(f64, f64)> = points
        .iter()
        .zip(relative_floors.iter())
        .map(|(point, &floor)| {
            jeffreys_candidate(&point.spectrum, floor, JEFFREYS_TOP_SATURATION.max(floor))
        })
        .collect();

    for (index, point) in points.iter().enumerate() {
        let spectrum = point
            .spectrum
            .iter()
            .map(|lambda| format!("{lambda:.3e}"))
            .collect::<Vec<String>>()
            .join(",");
        let decade = index + 1;
        let eps = point.margin;
        let nll = point.neg_log_likelihood;
        let gate = point.production_gate;
        let production = point.production_phi;
        let production_roundoff = point.production_roundoff;
        let (armed_value, armed_roundoff) = armed[index];
        let no_floor_value = no_floor[index].0;
        let (no_cap_value, no_cap_roundoff) = no_cap[index];
        let relative_value = relative_floor[index].0;
        let floor = relative_floors[index];
        eprintln!(
            "[2765-FACE] decade={decade} eps={eps:.3e} nll={nll:.9e} spectrum=[{spectrum}] \
             G={gate:.3e} phi_production={production:.6e}+-{production_roundoff:.1e} \
             phi_armed={armed_value:.6e}+-{armed_roundoff:.1e} phi_no_floor={no_floor_value:.6e} \
             phi_no_cap={no_cap_value:.6e}+-{no_cap_roundoff:.1e} \
             phi_relative_floor={relative_value:.6e} relative_floor={floor:.3e}"
        );
    }

    let resolved = 0.05 * std::f64::consts::LN_10;
    let phi_of = |readings: &[(f64, f64)]| {
        readings
            .iter()
            .map(|reading| reading.0)
            .collect::<Vec<f64>>()
    };
    let armed_phi = phi_of(&armed);
    let no_floor_phi = phi_of(&no_floor);
    let no_cap_phi = phi_of(&no_cap);
    let relative_floor_phi = phi_of(&relative_floor);
    let production_phi: Vec<f64> = points.iter().map(|point| point.production_phi).collect();
    let barrier_only = vec![0.0_f64; points.len()];

    let barrier = barrier_rate(&points, &barrier_only, &|index: usize| index < points.len());
    let armed_rate = barrier_rate(&points, &armed_phi, &|index: usize| {
        armed[index].1 <= resolved
    });
    let no_floor_rate = barrier_rate(&points, &no_floor_phi, &|index: usize| {
        no_floor[index].1 <= resolved
    });
    let production_rate = barrier_rate(&points, &production_phi, &|index: usize| {
        points[index].production_roundoff <= resolved
    });
    let no_cap_rate = barrier_rate(&points, &no_cap_phi, &|index: usize| {
        no_cap[index].1 <= resolved
    });
    let relative_floor_rate = barrier_rate(&points, &relative_floor_phi, &|index: usize| {
        relative_floors[index] >= JEFFREYS_TOP_SATURATION && relative_floor[index].1 <= resolved
    });

    let report = |label: &str, reading: Option<(f64, usize, usize)>| match reading {
        Some((rate, first, last)) => {
            let first_decade = first + 1;
            let last_decade = last + 1;
            let first_eps = points[first].margin;
            let last_eps = points[last].margin;
            eprintln!(
                "[2765-FACE-RATE] {label} rate={rate:.4} decades={first_decade}..{last_decade} \
                 eps={first_eps:.3e}..{last_eps:.3e}"
            );
        }
        None => eprintln!("[2765-FACE-RATE] {label} unresolved: fewer than three decades admitted"),
    };
    report("barrier(-l)", barrier);
    report("armed", armed_rate);
    report("no_floor", no_floor_rate);
    report("production", production_rate);
    report("control_no_cap", no_cap_rate);
    report("control_relative_floor", relative_floor_rate);
    if let Some((rate, first, last)) = armed_rate {
        let decades = points[first].margin.log10() - points[last].margin.log10();
        let exponents = points[first]
            .spectrum
            .iter()
            .zip(points[last].spectrum.iter())
            .map(|(early, late)| {
                format!("{:.3}", (late.abs().log10() - early.abs().log10()) / decades)
            })
            .collect::<Vec<String>>()
            .join(",");
        let first_decade = first + 1;
        let last_decade = last + 1;
        eprintln!(
            "[2765-FACE-SPECTRUM] armed window decades {first_decade}..{last_decade} \
             rate={rate:.4}: growth exponent of |lambda| in 1/eps along the ascending spectrum \
             [{exponents}] (2: |u|^2/eps^2, 1: |mu|/eps, 0: O(1))"
        );
    }

    let predicate = 0.5;
    let barrier = barrier.expect("every decade is admitted for the likelihood alone");
    assert!(
        barrier.0 >= 0.9,
        "the walk must reach the face: -l rose at rate {:.4} per e-fold of the margin, below the \
         one -ln(eta1') an event row carries",
        barrier.0
    );
    let armed_rate =
        armed_rate.expect("the armed objective must be resolved over three decades of the walk");
    assert!(
        armed_rate.0 >= predicate,
        "the armed objective (G = 1, absolute floor, top saturation kept) lost the follow-up \
         barrier: rate {:.4}",
        armed_rate.0
    );
    let production_rate = production_rate
        .expect("the production objective must be resolved over three decades of the walk");
    assert!(
        production_rate.0 >= predicate,
        "the production objective lost the follow-up barrier: rate {:.4}",
        production_rate.0
    );
    if let Some(no_floor_rate) = no_floor_rate {
        assert!(
            no_floor_rate.0 >= predicate,
            "the floorless objective with the top saturation kept lost the follow-up barrier: \
             rate {:.4}",
            no_floor_rate.0
        );
    }
    let no_cap_rate =
        no_cap_rate.expect("the control without a top saturation must be resolved over three decades");
    assert!(
        no_cap_rate.0 < predicate,
        "positive control: removing the top saturation must lose the barrier, rate {:.4}",
        no_cap_rate.0
    );
    let relative_floor_rate = relative_floor_rate.expect(
        "the relative-floor control must be past floor = top saturation over three resolved decades",
    );
    assert!(
        relative_floor_rate.0 < predicate,
        "positive control: the relative floor with G = 1 must lose the barrier, rate {:.4}",
        relative_floor_rate.0
    );
}
