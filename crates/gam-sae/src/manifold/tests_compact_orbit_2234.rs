//! #2234 — the compact-orbit integral: the trapezoid quadrature against the von Mises closed form,
//! doubled-node agreement with the complement coupling on, and a positive control showing the
//! derived node count is not vacuously loose.

#![cfg(test)]

use super::compact_orbit::CircleOrbitIntegrand;
use gam_math::special::bessel_i0_centered_terms_from_log_abs;

/// `log ∫₀ᴾ exp(−c[cos φ̄ − cos(φ̄ + κs)]) ds = log P − c cos φ̄ + log I₀(c)`.
fn closed_form_log_integral(resultant_cos: f64, resultant_sin: f64, period: f64) -> f64 {
    let c = resultant_cos.hypot(resultant_sin);
    let centered = if c > 0.0 {
        bessel_i0_centered_terms_from_log_abs(c.ln()).0
    } else {
        0.0
    };
    period.ln() - resultant_cos + c + centered
}

/// The trapezoid sum at an explicit node count, the same arithmetic `integrate` runs.
fn trapezoid_log_integral(integrand: &CircleOrbitIntegrand, nodes: usize) -> f64 {
    let exponents: Vec<f64> = (0..nodes)
        .map(|j| integrand.exponent(std::f64::consts::TAU * j as f64 / nodes as f64))
        .collect();
    let normalizer = gam_math::categorical::log_sum_exp(&exponents).expect("finite exponents");
    (integrand.period / nodes as f64).ln() + normalizer
}

#[test]
fn circle_orbit_integral_matches_the_von_mises_closed_form_2234() {
    // Concentrations from the #2234 stall pin's railed ARD state (c ≈ 1.12e-6), an O(1) state and
    // E1's collapsed winner (c = 1689), at a phase off the minimum so `cos φ̄ ≠ 1`.
    for &(c, phase) in &[(1.12e-6, 0.3_f64), (1.0, -1.1), (1689.0, 0.02)] {
        let integrand = CircleOrbitIntegrand {
            resultant_cos: c * phase.cos(),
            resultant_sin: c * phase.sin(),
            coupling: [0.0, 0.0, 0.0],
            period: 1.0,
        };
        let integral = integrand.integrate().expect("orbit integral");
        let oracle = closed_form_log_integral(integrand.resultant_cos, integrand.resultant_sin, 1.0);
        let nodes = integral.angles.len();
        // The quadrature's own band: its node sum's accumulation growth plus the closed form's
        // log-magnitude rounding.
        let band = gam_linalg::roundoff::accumulation_growth(nodes)
            + 4.0 * f64::EPSILON * (oracle.abs() + c);
        eprintln!(
            "[#2234 orbit] c={c:e} nodes={nodes} quadrature={:.16e} closed_form={oracle:.16e} gap={:e} band={band:e}",
            integral.log_integral,
            (integral.log_integral - oracle).abs()
        );
        assert!(
            (integral.log_integral - oracle).abs() <= band,
            "c = {c}: trapezoid log integral {} misses the closed form {oracle} by {} against band {band}",
            integral.log_integral,
            (integral.log_integral - oracle).abs()
        );
        let total: f64 = integral.weights.iter().sum();
        assert!((total - 1.0).abs() <= gam_linalg::roundoff::accumulation_growth(nodes) * 2.0);
    }
}

#[test]
fn circle_orbit_node_count_is_tight_with_the_complement_coupling_on_2234() {
    // An O(1) concentration with a nonzero coupling, where no closed form exists: the derived count
    // must agree with twice as many nodes to the band, and a quarter of the count must miss it.
    let integrand = CircleOrbitIntegrand {
        resultant_cos: 2.5 * 0.4_f64.cos(),
        resultant_sin: 2.5 * 0.4_f64.sin(),
        coupling: [0.7, -0.2, 0.35],
        period: 1.0,
    };
    let nodes = integrand.node_count();
    let at_count = trapezoid_log_integral(&integrand, nodes);
    let doubled = trapezoid_log_integral(&integrand, 2 * nodes);
    let band = gam_linalg::roundoff::accumulation_growth(2 * nodes) + 4.0 * f64::EPSILON * at_count.abs();
    let quarter = (nodes / 4).max(1);
    let at_quarter = trapezoid_log_integral(&integrand, quarter);
    eprintln!(
        "[#2234 orbit coupling] nodes={nodes} at_count={at_count:.16e} doubled={doubled:.16e} \
         quarter_nodes={quarter} at_quarter={at_quarter:.16e} band={band:e}"
    );
    assert!(nodes >= 8, "the coupled integrand needs more than a handful of nodes, got {nodes}");
    assert!(
        (at_count - doubled).abs() <= band,
        "doubling the derived node count moved the integral by {} against band {band}",
        (at_count - doubled).abs()
    );
    assert!(
        (at_quarter - doubled).abs() > band,
        "a quarter of the derived node count ({quarter}) already agrees to the band, so the bound is loose"
    );
}

/// The derived node count terminates where the coupling growth outruns the concentration.
///
/// The demo's minted state carries log α = 6.012 (ad-sae2's defect 2, state 2). At the basis period
/// P = 1 that is η = αP²/(2π)² and a coupling scale ½η²κ², and the phase cloud is spread over the
/// demo's 160 rows (resultant η·√160). The complement forms `(a, b, d) = (0.05, 0.01, 0.05)` are
/// representative fixture inputs chosen to put Q above c, not measured values. That is the regime
/// where a strip width balancing only c·cosh σ against Nσ lets Q·cosh²σ outgrow the decay.
#[test]
fn circle_orbit_node_count_terminates_at_a_demo_like_coupling_2234() {
    let kappa = std::f64::consts::TAU;
    let eta = 6.012_f64.exp() / (kappa * kappa);
    let resultant = eta * 160.0_f64.sqrt();
    let scale = 0.5 * eta * eta * kappa * kappa;
    let forms = [0.05_f64, 0.01, 0.05];
    let integrand = CircleOrbitIntegrand {
        resultant_cos: resultant * 0.7_f64.cos(),
        resultant_sin: resultant * 0.7_f64.sin(),
        coupling: [forms[0] * scale, forms[1] * scale, forms[2] * scale],
        period: 1.0,
    };
    let c = integrand.concentration();
    let [qa, qb, qd] = integrand.coupling;
    let q = qa.abs() + 2.0 * qb.abs() + qd.abs();
    let nodes = integrand.node_count();
    let bound = integrand.relative_truncation_bound(nodes);
    let target = gam_linalg::roundoff::accumulation_growth(nodes);
    // Positive control: the superseded half-width σ = asinh(N/(c + 2Q)) on the same state. Its log
    // bound stays above every target and rises with N, so no node count meets it under that rule.
    let superseded: Vec<(usize, f64, f64)> = [1024usize, 4096, 16384, 65536]
        .iter()
        .map(|&count| {
            let sigma = (count as f64 / (c + 2.0 * q)).asinh();
            (
                count,
                integrand.log_relative_truncation_bound_at(count, sigma),
                gam_linalg::roundoff::accumulation_growth(count).ln(),
            )
        })
        .collect();
    eprintln!(
        "[#2234 orbit demo coupling] eta={eta:.6e} forms={forms:?} c={c:.6e} Q={q:.6e} nodes={nodes} bound={bound:e} \
         target={target:e} superseded (N, log bound, log target) {superseded:?}"
    );
    assert!(q > c, "the state must put the coupling growth above the concentration (Q = {q}, c = {c})");
    assert!(bound <= target, "the derived count {nodes} misses its own target: bound {bound} > {target}");
    assert!(
        superseded.iter().all(|&(_, log_bound, log_target)| log_bound > log_target)
            && superseded.windows(2).all(|pair| pair[1].1 > pair[0].1),
        "the superseded strip width must give a bound above its target that rises with N, got {superseded:?}"
    );
}

/// A closure-certified circle of `atom` whose tangent touches `support`.
fn separation_generator(atom: usize, support: &[usize], dim: usize) -> super::compact_orbit::CircleOrbitGenerator {
    let mut tangent = ndarray::Array1::<f64>::zeros(dim);
    for &slot in support {
        tangent[slot] = 1.0;
    }
    super::compact_orbit::CircleOrbitGenerator {
        atom,
        period: 1.0,
        kappa: std::f64::consts::TAU,
        eta: 1.0,
        tangent,
        prior_rows: Vec::new(),
        closure_residual: 0.0,
        closure_band: 0.0,
        closure: ndarray::Array2::<f64>::zeros((1, 1)),
        border_start: 0,
        basis_size: 1,
        border_rank: 1,
    }
}

/// Hard TopK, four rows selecting one atom each: rows 0-1 atom 0, rows 2-3 atom 1. The border holds
/// atom 0's two columns, atom 1's two and one column of a non-orbit atom 2. A column's pattern
/// couples its own atom's rows and border only.
fn separation_border_pattern() -> ndarray::Array2<f64> {
    let mut columns = ndarray::Array2::<f64>::zeros((9, 5));
    for (column, rows) in [(0, [0, 1]), (1, [0, 1]), (2, [2, 3]), (3, [2, 3])] {
        for row in rows {
            columns[[row, column]] = 0.5;
        }
    }
    for (column, borders) in [(0, [4, 5]), (1, [4, 5]), (2, [6, 7]), (3, [6, 7])] {
        for border in borders {
            columns[[border, column]] = 2.0;
        }
    }
    columns[[8, 4]] = 3.0;
    columns
}

fn separated_verdicts(
    operator: &ndarray::Array2<f64>,
    metric: &ndarray::Array2<f64>,
    carriers: &[(f64, Vec<(usize, f64)>)],
) -> Vec<String> {
    separated_verdicts_in(&[0, 1, 2, 3, 4], [&[0, 1, 4, 5], &[2, 3, 6, 7]], Some((operator, metric)), carriers)
}

/// The two orbits' verdicts over `row_offsets`, with tangents touching `supports`, from the row blocks and
/// carriers alone when `border` is absent.
fn separated_verdicts_in(
    row_offsets: &[usize],
    supports: [&[usize]; 2],
    border: Option<(&ndarray::Array2<f64>, &ndarray::Array2<f64>)>,
    carriers: &[(f64, Vec<(usize, f64)>)],
) -> Vec<String> {
    use super::compact_orbit::{CompactOrbitLaplaceReason, CompactOrbitPricing};
    let pricings = vec![
        CompactOrbitPricing::ExactCircle(separation_generator(0, supports[0], 9)),
        CompactOrbitPricing::ExactCircle(separation_generator(1, supports[1], 9)),
        CompactOrbitPricing::Laplace {
            atom: 2,
            reason: CompactOrbitLaplaceReason::NotAPeriodicChart,
        },
    ];
    super::SaeManifoldTerm::separate_coupled_compact_orbits(
        pricings,
        row_offsets,
        5,
        carriers,
        border.map(|(operator, metric)| (operator.view(), metric.view())),
    )
    .expect("the separation reads a consistent layout")
    .into_iter()
    .take(2)
    .map(|pricing| match pricing {
        CompactOrbitPricing::ExactCircle(generator) => format!("atom {} exact", generator.atom),
        CompactOrbitPricing::Laplace { atom, reason } => format!("atom {atom} Laplace {reason:?}"),
    })
    .collect()
}

#[test]
fn only_orbits_in_distinct_blocks_of_a_and_phi_are_integrated_one_by_one_2234() {
    let base = separation_border_pattern();
    let separated = separated_verdicts(&base, &base, &[]);
    eprintln!("[#2234 orbit separation] block-diagonal: {separated:?}");
    assert_eq!(separated, vec!["atom 0 exact", "atom 1 exact"], "hard-TopK rows over a block-diagonal border keep both orbits exact");

    // Each coupling source alone joins the two orbits' blocks, and each orbit then names the other.
    let coupled = vec![
        "atom 0 Laplace CoupledCompactOrbits { atoms: [1] }",
        "atom 1 Laplace CoupledCompactOrbits { atoms: [0] }",
    ];
    let mut operator = base.clone();
    operator[[6, 0]] = 1.0e-9;
    operator[[4, 2]] = 1.0e-9;
    let from_operator = separated_verdicts(&operator, &base, &[]);
    let mut metric = base.clone();
    metric[[7, 1]] = 1.0e-12;
    metric[[5, 3]] = 1.0e-12;
    let from_metric = separated_verdicts(&base, &metric, &[]);
    let from_carrier = separated_verdicts(&base, &base, &[(0.25, vec![(0, 1.0), (2, -1.0)])]);
    // Through the non-orbit atom 2's border column: the orbits share no entry, only a third block.
    let mut bridged = base.clone();
    bridged[[5, 4]] = 1.0;
    bridged[[7, 4]] = 1.0;
    let through_third = separated_verdicts(&bridged, &base, &[]);
    eprintln!(
        "[#2234 orbit separation] operator {from_operator:?} metric {from_metric:?} carrier {from_carrier:?} \
         third block {through_third:?}"
    );
    assert_eq!(from_operator, coupled, "a cross-atom entry of A couples the orbits");
    assert_eq!(from_metric, coupled, "a cross-atom entry of Φ alone couples the orbits");
    assert_eq!(from_carrier, coupled, "a mass carrier across the atoms' rows couples the orbits");
    assert_eq!(through_third, coupled, "a non-orbit block touching both orbits couples them");

    // The row-only pass. Dense rows (width 2) hold both atoms' coordinates, so the rows alone couple
    // the orbits and no border column is needed. Over compact rows the rows alone leave both exact,
    // so that pass cannot stand in for the border columns.
    let dense_rows = separated_verdicts_in(&[0, 2, 4], [&[0, 2, 4, 5], &[1, 3, 6, 7]], None, &[]);
    let compact_rows = separated_verdicts_in(&[0, 1, 2, 3, 4], [&[0, 1, 4, 5], &[2, 3, 6, 7]], None, &[]);
    eprintln!("[#2234 orbit separation] rows only: dense {dense_rows:?} compact {compact_rows:?}");
    assert_eq!(dense_rows, coupled, "a row holding both atoms couples their orbits without a border probe");
    assert_eq!(compact_rows, vec!["atom 0 exact", "atom 1 exact"], "compact rows alone couple nothing");
}
