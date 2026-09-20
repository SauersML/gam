//! gam#2926: a re-solve on another law starts from its converged fit's
//! coefficients. The cold-start pilot seeds only the blocks the re-solve drops.

use super::*;

fn filled(width: usize, value: f64) -> Array1<f64> {
    Array1::from_elem(width, value)
}

#[test]
fn a_re_solve_keeps_its_converged_blocks_and_the_pilot_seeds_only_those_it_drops_2926() {
    let widths = [5, 3, 2];
    let pilot = [filled(5, 0.1), filled(3, 0.2), filled(2, 0.3)];
    let converged_time = filled(5, -1.5);
    let converged_marginal = filled(3, 2.5);

    // The moving-law re-solve onto an arm on another axis carries the time and
    // marginal blocks and drops the slope, which lives on the latent axis.
    let mut hints = ThetaHints {
        time_beta: Some(converged_time.clone()),
        marginal_beta: Some(converged_marginal.clone()),
        ..ThetaHints::default()
    };
    let seeded = seed_uncarried_blocks(&mut hints, [&pilot[0], &pilot[1], &pilot[2]], widths);
    assert_eq!(seeded, [false, false, true]);
    assert_eq!(hints.time_beta.as_ref(), Some(&converged_time));
    assert_eq!(hints.marginal_beta.as_ref(), Some(&converged_marginal));
    assert_eq!(hints.slope_beta.as_ref(), Some(&pilot[2]));

    // The closed-form re-solve carries all three: nothing is reseeded.
    let converged_slope = filled(2, 0.7);
    let mut hints = ThetaHints {
        time_beta: Some(converged_time.clone()),
        marginal_beta: Some(converged_marginal.clone()),
        slope_beta: Some(converged_slope.clone()),
        ..ThetaHints::default()
    };
    let seeded = seed_uncarried_blocks(&mut hints, [&pilot[0], &pilot[1], &pilot[2]], widths);
    assert_eq!(seeded, [false; 3]);
    assert_eq!(hints.slope_beta.as_ref(), Some(&converged_slope));

    // A fresh fit carries nothing, so the pilot seeds every block whose width
    // is its design's.
    let mut fresh = ThetaHints::default();
    let seeded = seed_uncarried_blocks(&mut fresh, [&pilot[0], &pilot[1], &filled(4, 0.3)], widths);
    assert_eq!(seeded, [true, true, false]);
    assert_eq!(fresh.time_beta.as_ref(), Some(&pilot[0]));
    assert!(fresh.slope_beta.is_none());
}
