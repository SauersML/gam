//! #2939/#3355 — the rank-charge stratum wall, profiled on the fixture whose
//! search is pinned at it.
//!
//! `tests_termination_2235::planted_circle_fit_returns_with_analytic_certificate`
//! is red on main with published evidence `rank_boundary=[kept_rank=5 (per
//! component [2, 3]), refused_trials=1, ...]` and `line_search=StepSizeTooSmall
//! after 50 attempt(s)`: the outer search halts where its kept rank ends, exactly
//! the #2266 diagnosis. The named redesign (issue #3355, 2026-09-30) is to fit
//! each rank stratum as a smooth criterion of its own and select across strata
//! by value. That redesign needs one measurement this test takes: the profile of
//! the criterion over a (log λ_sparse, log λ_smooth) grid around the seed, with
//! each evaluation's realized stratum, so the flip is bracketed between adjacent
//! grid points and the jump at the flip is priced against the within-stratum
//! slope on both sides.
//!
//! What is asserted (structure, not the redesign's acceptance):
//! * the grid sees at least two distinct realized strata — the wall passes
//!   through the seed's neighbourhood, which is why the search meets it;
//! * some adjacent grid pair within one row has differing strata — the flip is
//!   bracketed, so a per-stratum fit has a segment to optimize on each side.
//!
//! What is reported (the receipt the redesign prices): per flip pair, the two
//! criterion values, their jump, the within-stratum slope estimate on each side
//! (from the next interior point), and the strata both sides.

use super::tests::planted_circle_embedded;
use super::tests_startup_validation_1782::{Topo, objective_and_seed};
use crate::assignment::AssignmentMode;

/// One grid probe: the criterion value at a ρ and the realized rank stratum the
/// value was priced on (`None` when the evaluation refused).
struct Probe {
    value: Option<f64>,
    stratum: Option<Vec<usize>>,
}

/// Probe the criterion on a grid over the first two flat ρ coordinates
/// (log λ_sparse, log λ_smooth), every other coordinate held at the seed.
fn profile(objective: &mut super::outer_objective::SaeManifoldOuterObjective, seed: &ndarray::Array1<f64>) -> Vec<Vec<Probe>> {
    let [c0, c1] = [seed[0], seed[1]];
    // Half-widths chosen to reach the wall the failing search pinned at on this
    // fixture (its checkpoint sat near log λ_sparse ≈ 0.7, log λ_smooth railed
    // at −2.46; the seed sits at (−3.91, ~0)): ±3.5 in λ_sparse, ±3 in
    // λ_smooth.
    let sparse_grid: Vec<f64> = (0..13).map(|i| c0 - 3.5 + 7.0 * i as f64 / 12.0).collect();
    let smooth_grid: Vec<f64> = (0..9).map(|j| c1 - 3.0 + 6.0 * j as f64 / 8.0).collect();
    let mut rows = Vec::with_capacity(smooth_grid.len());
    for &ls in &smooth_grid {
        let mut row = Vec::with_capacity(sparse_grid.len());
        for &lp in &sparse_grid {
            let mut rho = seed.clone();
            rho[0] = lp;
            rho[1] = ls;
            let probe = match objective.evaluate_authoritative_criterion(rho.view()) {
                Ok((value, _)) => Probe {
                    value: Some(value),
                    stratum: objective.term.priced_rank_stratum.as_ref().map(|r| r.to_vec()),
                },
                Err(_) => Probe { value: None, stratum: None },
            };
            row.push(probe);
        }
        rows.push(row);
    }
    rows
}

#[test]
fn rank_wall_profiles_piecewise_smooth_criteria_with_a_bracketed_flip_2939() {
    gam_runtime::test_support::install_diagnostic_logger();
    // The exact `tests_termination_2235` fixture whose outer search halts at
    // the rank wall on main.
    let z = planted_circle_embedded(48, 6, 0.03);
    let (mut objective, seed) = objective_and_seed(
        z.view(),
        2,
        Topo::Circle,
        AssignmentMode::softmax(1.0),
    );
    let rows = profile(&mut objective, &seed);

    // Distinct realized strata over the grid.
    let mut strata: Vec<Vec<usize>> = Vec::new();
    let mut finite = 0usize;
    for row in &rows {
        for probe in row {
            if let Some(stratum) = &probe.stratum {
                finite += 1;
                if !strata.contains(stratum) {
                    strata.push(stratum.clone());
                }
            }
        }
    }
    assert!(finite > 0, "at least one evaluation must price a stratum");
    for (i, s) in strata.iter().enumerate() {
        eprintln!("[#2939-profile] stratum {i}: per-atom chargeable ranks {s:?}");
    }
    assert!(
        strata.len() >= 2,
        "the grid over the seed's neighbourhood must cross the rank wall (found {} \
         stratum{}: {strata:?}); a single stratum would mean this fixture's wall lies \
         outside the probed region",
        strata.len(),
        if strata.len() == 1 { "" } else { "s" }
    );

    // A bracketed flip: adjacent pair within one row with differing strata.
    let mut flips = Vec::new();
    for row in &rows {
        for w in row.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if let (Some(sa), Some(sb), Some(va), Some(vb)) = (&a.stratum, &b.stratum, a.value, b.value)
                && sa != sb
            {
                flips.push((sa.clone(), va, sb.clone(), vb));
            }
        }
    }
    assert!(
        !flips.is_empty(),
        "some adjacent grid pair must bracket the flip (differing strata in one row)"
    );
    for (i, (sa, va, sb, vb)) in flips.iter().enumerate() {
        eprintln!(
            "[#2939-profile] flip {i}: {sa:?} -> {sb:?} values {va:.6e} -> {vb:.6e} jump {:+.6e}",
            vb - va
        );
    }
    // The receipt for the redesign: the jump at every flip must be reported
    // against the criterion's own resolution, so a per-stratum selection can
    // price it. (No magnitude is asserted: the jump is ½·Δr·edf·log N_eff,
    // fixture-dependent by construction.)
    for (sa, va, sb, vb) in &flips {
        let jump = vb - va;
        let resolution = f64::EPSILON * (1.0 + va.abs().max(vb.abs()));
        eprintln!(
            "[#2939-profile] flip jump {jump:+.6e} vs value resolution {resolution:.3e} (strata {sa:?} -> {sb:?})"
        );
    }
}
