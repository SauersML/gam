#![cfg(test)]
//! #2946 R7 pins for the exhaustive finite-grid FANOVA and its worst-case complement.

use super::*;
use gam_sae::response::interaction::total_interactions;
use ndarray::{Array2, array};

fn governor() -> &'static MemoryGovernor {
    MemoryGovernor::global()
}

/// `F(a, b)` tabulated row-major over `Z_p × Z_q`.
fn table(
    p: usize,
    q: usize,
    outputs: usize,
    f: impl Fn(usize, usize, usize) -> f64,
) -> Array2<f64> {
    Array2::from_shape_fn((p * q, outputs), |(cell, output)| {
        f(cell / q, cell % q, output)
    })
}

fn within(value: f64, target: f64, band: f64) -> bool {
    (value - target).abs() <= band
}

/// `xy` on `{−1, 1}²`: the rectangle difference is 4, so every additive `g` misses by at least 1, and
/// the pair-ANOVA residual attains 1, so the bounds meet: `Exact` with value 1 and the witness.
#[test]
fn xy_on_the_signed_square_has_rectangle_four_and_exact_bound_one() {
    let signs = [-1.0, 1.0];
    let responses = table(2, 2, 1, |x, y, _| signs[x] * signs[y]);
    let grid = FiniteGridResponse::new(vec![2, 2], responses.view(), None, governor())
        .expect("a full product grid");
    let complement = grid.rectangle_complement(0, 1).expect("a valid pair");
    assert_eq!(complement.max_difference, 4.0);
    let witness = complement.witness.clone().expect("a rectangle");
    assert_eq!(witness.difference, 4.0);
    assert!(within(
        complement.anova_residual_sup,
        1.0,
        complement.anova_upper - 1.0 + 1e-300
    ));
    match &complement.evidence {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            witness: Some(_),
            ..
        } => assert!(
            within(*value, 1.0, *numerical_error),
            "{:?}",
            complement.evidence
        ),
        other => panic!("expected Exact 1, got {other:?}"),
    }
    assert!(complement.evidence.lower_bound().expect("exact") <= 1.0);
    assert!(complement.evidence.upper_bound().expect("exact") >= 1.0);
    assert!(complement.reconstruction.is_none());

    let screen = total_interactions(&grid).expect("the grid screens");
    let interaction = screen.interaction(0, 1).expect("ports in range");
    assert!(
        within(interaction.value, 1.0, interaction.band),
        "{interaction:?}"
    );
    assert_eq!(screen.additive_blocks(), vec![vec![0, 1]]);
}

/// An exactly additive grid function returns `Exact{Exhaustive}` with value 0 and a reconstruction equal
/// to the table. Positive control: one cell moved by 1/2 is caught, with the moved cell on the witness
/// rectangle and a lower bound of 1/8.
#[test]
fn an_additive_grid_is_reconstructed_exactly_and_a_planted_defect_is_caught() {
    let row = [1.0, -2.0, 0.5];
    let column = [3.0, 0.0, -1.0, 2.5];
    let additive = table(3, 4, 2, |x, y, output| {
        (output as f64 + 1.0) * row[x] - column[y] * if output == 0 { 1.0 } else { -2.0 }
    });
    let grid = FiniteGridResponse::new(vec![3, 4], additive.view(), None, governor())
        .expect("a full product grid");
    let complement = grid.rectangle_complement(0, 1).expect("a valid pair");
    match &complement.evidence {
        EvidenceStatus::Exact {
            value,
            basis: ExactBasis::Exhaustive { cardinality },
            ..
        } => {
            assert_eq!(*value, 0.0);
            assert_eq!(*cardinality, 24);
        }
        other => panic!("expected Exact 0, got {other:?}"),
    }
    let reconstruction = complement.reconstruction.expect("additive");
    assert_eq!(reconstruction, additive);
    let screen = total_interactions(&grid).expect("the grid screens");
    assert_eq!(screen.additive_blocks(), vec![vec![0], vec![1]]);

    let mut defective = additive.clone();
    defective[[2 * 4 + 1, 1]] += 0.5;
    let grid = FiniteGridResponse::new(vec![3, 4], defective.view(), None, governor())
        .expect("a full product grid");
    let complement = grid.rectangle_complement(0, 1).expect("a valid pair");
    assert!(complement.reconstruction.is_none());
    assert_eq!(complement.max_difference, 0.5);
    let witness = complement.witness.expect("a rectangle");
    assert_eq!(witness.output, 1);
    let touches = (witness.corner == vec![2, 1])
        || (witness.anchor == (2, 1))
        || (witness.corner[0] == 2 && witness.anchor.1 == 1)
        || (witness.anchor.0 == 2 && witness.corner[1] == 1);
    assert!(touches, "the witness {witness:?} holds the moved cell");
    let lower = complement.evidence.lower_bound().expect("a lower bound");
    assert!(lower <= 0.125 && lower > 0.125 - 1e-12, "{lower}");
    let screen = total_interactions(&grid).expect("the grid screens");
    assert!(
        screen
            .interaction(0, 1)
            .expect("in range")
            .resolved_positive()
    );
}

/// A function of `a + b mod 7` alone: in `(a, b)` it has no main effects, so the interaction is all of the
/// energy, and no additive response comes within 1/2 of it. In `(s, d) = (a + b, a − b)` it is the `s`
/// main effect alone: the `d` total effect is zero, the blocks split, and the rectangles vanish.
#[test]
fn a_sum_mod_p_function_is_pure_interaction_in_ab_and_pure_main_effect_in_sd() {
    let p = 7;
    let responses = table(p, p, p, |a, b, output| {
        let s = (a + b) % p;
        if output == s {
            3.0
        } else {
            f64::from(u32::try_from((s * output) % 3).expect("small"))
        }
    });
    let factor = centring_factor(p);
    let grid = FiniteGridResponse::new(
        vec![p, p],
        responses.view(),
        Some(factor.view()),
        governor(),
    )
    .expect("a full product grid");
    let screen = total_interactions(&grid).expect("the grid screens");
    let total = screen.total_variance();
    let interaction = screen.interaction(0, 1).expect("in range");
    assert!(total.resolved_positive());
    assert!(
        within(
            interaction.value,
            total.value,
            interaction.band + total.band
        ),
        "I_ab {interaction:?} vs V {total:?}"
    );
    let complement = grid.rectangle_complement(0, 1).expect("a valid pair");
    assert!(complement.reconstruction.is_none());
    assert!(complement.evidence.lower_bound().expect("a lower bound") > 0.5);

    let sum_difference =
        |cell: &[usize]| vec![(cell[0] + cell[1]) % p, (cell[0] + p - cell[1]) % p];
    let rotated = grid
        .reindex(vec![p, p], sum_difference, governor())
        .expect("(a+b, a−b) is a bijection of Z_7²");
    let screen = total_interactions(&rotated).expect("the (s, d) grid screens");
    let total = screen.total_variance();
    let s_main = rotated.retained_variance(&[0]).expect("a retained set");
    assert!(within(s_main.value, total.value, s_main.band + total.band));
    let d_effect = screen.total_effect(1).expect("in range");
    assert!(!d_effect.resolved_positive(), "d-dependence {d_effect:?}");
    assert!(d_effect.value.abs() <= d_effect.band);
    assert_eq!(screen.additive_blocks(), vec![vec![0], vec![1]]);
    let complement = rotated.rectangle_complement(0, 1).expect("a valid pair");
    assert!(matches!(complement.evidence, EvidenceStatus::Exact { value, .. } if value == 0.0));
    let reconstruction = complement.reconstruction.expect("additive in (s, d)");
    for cell in 0..p * p {
        for output in 0..p {
            let (value, band) = rotated
                .value(&[cell / p, cell % p], output)
                .expect("in range");
            assert!(within(
                reconstruction[[cell, output]],
                value,
                complement.uniform_band + band
            ));
        }
    }
    // The s-only law E[G | s] reads the label at every one of the p² inputs.
    let law = rotated.conditional_mean(&[0]).expect("a retained set");
    for s in 0..p {
        let row = law.values.row(s);
        let best = (0..p)
            .max_by(|&l, &r| row[l].total_cmp(&row[r]))
            .expect("outputs");
        assert_eq!(best, s);
    }
}

/// The re-indexing is a declaration and is checked: `(a+b, a−b)` folds `Z_8²` two to one.
#[test]
fn a_non_bijective_reindexing_is_refused() {
    let p = 8;
    let responses = table(p, p, 1, |a, b, _| (a * b) as f64);
    let grid = FiniteGridResponse::new(vec![p, p], responses.view(), None, governor())
        .expect("a full product grid");
    let fold = |cell: &[usize]| vec![(cell[0] + cell[1]) % p, (cell[0] + p - cell[1]) % p];
    assert!(matches!(
        grid.reindex(vec![p, p], fold, governor()),
        Err(FiniteGridError::NotABijection { .. })
    ));
    assert!(matches!(
        grid.reindex(vec![p, p + 1], |cell: &[usize]| cell.to_vec(), governor()),
        Err(FiniteGridError::NotABijection { .. })
    ));
    // Control: the swap is a bijection and moves the interaction nowhere.
    let swapped = grid
        .reindex(
            vec![p, p],
            |cell: &[usize]| vec![cell[1], cell[0]],
            governor(),
        )
        .expect("a swap is a bijection");
    let before = total_interactions(&grid)
        .expect("screens")
        .interaction(0, 1)
        .expect("in range");
    let after = total_interactions(&swapped)
        .expect("screens")
        .interaction(0, 1)
        .expect("in range");
    assert!(within(before.value, after.value, before.band + after.band));
}

/// Concurvity: a correlated domain (`a ≤ b`) or a repeated cell is refused. The full product in any row
/// order is accepted and equals the row-major table.
#[test]
fn a_non_product_domain_is_refused() {
    let p = 4;
    let triangle: Vec<Vec<usize>> = (0..p)
        .flat_map(|a| (a..p).map(move |b| vec![a, b]))
        .collect();
    let responses = Array2::from_shape_fn((triangle.len(), 1), |(row, _)| row as f64);
    assert!(matches!(
        FiniteGridResponse::from_points(vec![p, p], &triangle, responses.view(), None, governor()),
        Err(FiniteGridError::NonProductDomain { .. })
    ));
    let mut repeated: Vec<Vec<usize>> = (0..p * p).map(|cell| vec![cell / p, cell % p]).collect();
    repeated[1] = vec![0, 0];
    let responses = Array2::from_shape_fn((p * p, 1), |(row, _)| row as f64);
    assert!(matches!(
        FiniteGridResponse::from_points(vec![p, p], &repeated, responses.view(), None, governor()),
        Err(FiniteGridError::NonProductDomain { .. })
    ));

    let reversed: Vec<Vec<usize>> = (0..p * p)
        .rev()
        .map(|cell| vec![cell / p, cell % p])
        .collect();
    let values = |a: usize, b: usize| (a * a + 3 * b + a * b) as f64;
    let reversed_responses = Array2::from_shape_fn((p * p, 1), |(row, _)| {
        values(reversed[row][0], reversed[row][1])
    });
    let from_points = FiniteGridResponse::from_points(
        vec![p, p],
        &reversed,
        reversed_responses.view(),
        None,
        governor(),
    )
    .expect("the full product");
    let direct = FiniteGridResponse::new(
        vec![p, p],
        table(p, p, 1, |a, b, _| values(a, b)).view(),
        None,
        governor(),
    )
    .expect("the full product");
    for a in 0..p {
        for b in 0..p {
            assert_eq!(from_points.value(&[a, b], 0), direct.value(&[a, b], 0));
        }
    }
}

/// Three ports `g(x₀, x₁) + h(x₂)`: blocks `{0,1}, {2}`, and the cross-block energy of the singletons is
/// `I_01`, since the response is purely pairwise.
#[test]
fn three_port_grid_splits_and_prices_partitions_unchanged() {
    let levels = vec![3, 2, 4];
    let responses = Array2::from_shape_fn((24, 2), |(cell, output)| {
        let (x0, x1, x2) = (cell / 8, (cell / 4) % 2, cell % 4);
        let pair = (x0 as f64 - 1.0) * (x1 as f64 * 2.0 - 1.0);
        let single = [0.0, 1.0, -1.0, 3.0][x2];
        if output == 0 {
            pair + single
        } else {
            2.0 * pair - single
        }
    });
    let metric_factor = array![[1.0, 0.0], [1.0, 1.0]];
    let grid = FiniteGridResponse::new(
        levels,
        responses.view(),
        Some(metric_factor.view()),
        governor(),
    )
    .expect("a full product grid");
    let screen = total_interactions(&grid).expect("the grid screens");
    assert_eq!(screen.additive_blocks(), vec![vec![0, 1], vec![2]]);
    let singletons = vec![vec![0], vec![1], vec![2]];
    let energy = screen
        .cross_block_energy(&grid, &singletons)
        .expect("a partition");
    let bound = screen.cross_block_bound(&singletons).expect("a partition");
    assert!(energy.resolved_positive());
    assert!(within(energy.value, bound.value, energy.band + bound.band));
    let across = grid.rectangle_complement(0, 2).expect("a valid pair");
    assert!(matches!(across.evidence, EvidenceStatus::Exact { value, .. } if value == 0.0));
    let within_block = grid.rectangle_complement(1, 0).expect("a valid pair");
    assert!(within_block.reconstruction.is_none());
    assert!(matches!(
        grid.rectangle_complement(1, 1),
        Err(FiniteGridError::InvalidPair { .. })
    ));
}
