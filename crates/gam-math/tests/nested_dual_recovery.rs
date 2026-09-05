//! #2818 recovery of the public nested-dual contracts behind #932.

use gam_math::jet_tower::Tower4;
use gam_math::nested_dual::{Dual2, Dual22, JetField};

fn smooth_program<J: JetField>(x: &J, y: &J) -> J {
    let exponential_argument = x.mul(y).add(&x.scale(0.3));
    let exponential = exponential_argument.value().exp();
    let logarithm_argument = x
        .constant_like(1.0)
        .add(&x.mul(x))
        .add(&y.mul(y).scale(0.5))
        .add(&x.mul(y).scale(0.2));
    let reciprocal = 1.0 / logarithm_argument.value();
    let logarithm = logarithm_argument.compose_unary([
        logarithm_argument.value().ln(),
        reciprocal,
        -reciprocal.powi(2),
        2.0 * reciprocal.powi(3),
        -6.0 * reciprocal.powi(4),
    ]);
    let difference = x.sub(y);
    exponential_argument
        .compose_unary([exponential; 5])
        .add(&logarithm)
        .sub(&difference.mul(&difference).scale(0.7))
}

fn directional_seed(value: f64, outer: f64, inner: f64) -> Dual22 {
    Dual2 {
        v: Dual2 {
            v: value,
            g: inner,
            h: 0.0,
        },
        g: Dual2::constant(outer),
        h: Dual2::constant(0.0),
    }
}

fn compare_channels(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual.iter().zip(expected).enumerate().map(|(channel, (&got, &want))| {
        let relative_error = (got - want).abs() / want.abs().max(1.0);
        assert!(relative_error <= 1e-12,
            "channel {channel}: got {got:.16e}, expected {want:.16e}, relative error {relative_error:.3e}");
        relative_error
    }).fold(0.0, f64::max)
}

#[test]
fn nested_dual2_reproduces_tower4_channels_932() {
    let mut maximum_error = 0.0_f64;
    let mut fourth_order_witness = 0.0_f64;
    for (x, y) in [(0.31, -0.42), (-0.85, 0.17), (0.05, 0.93), (1.2, -0.6)] {
        let tower = smooth_program(&Tower4::<2>::variable(x, 0), &Tower4::<2>::variable(y, 1));
        let nested = smooth_program(
            &Dual2::variable(Dual2::constant(x)),
            &Dual2::constant(Dual2::variable(y)),
        );
        let expected = [
            tower.v,
            tower.g[0],
            tower.g[1],
            tower.h[0][0],
            tower.h[0][1],
            tower.h[1][1],
            tower.t3[0][0][1],
            tower.t3[0][1][1],
            tower.t4[0][0][1][1],
        ];
        maximum_error = maximum_error.max(compare_channels(&nested.channels(), &expected));
        fourth_order_witness = fourth_order_witness.max(expected[8].abs());
    }
    assert!(
        fourth_order_witness > 0.1,
        "the mixed fourth derivative must be resolved"
    );
    eprintln!(
        "#932 nested/tower nine channels: maximum_error={maximum_error:.6e} fourth_order_witness={fourth_order_witness:.6e}"
    );
}

#[test]
fn nested_dual2_directional_matches_tower4_contraction_932() {
    let outer = [0.7, -0.3];
    let inner = [0.4, 0.9];
    let mut maximum_error = 0.0_f64;
    let mut fourth_order_witness = 0.0_f64;
    for (x, y) in [(0.31, -0.42), (-0.85, 0.17), (1.2, -0.6)] {
        let tower = smooth_program(&Tower4::<2>::variable(x, 0), &Tower4::<2>::variable(y, 1));
        let mut expected = [0.0; 4];
        for a in 0..2 {
            expected[0] += tower.g[a] * outer[a];
            expected[1] += tower.g[a] * inner[a];
            for b in 0..2 {
                expected[2] += tower.h[a][b] * outer[a] * inner[b];
                for c in 0..2 {
                    for d in 0..2 {
                        expected[3] +=
                            tower.t4[a][b][c][d] * outer[a] * outer[b] * inner[c] * inner[d];
                    }
                }
            }
        }
        let channels = smooth_program(
            &directional_seed(x, outer[0], inner[0]),
            &directional_seed(y, outer[1], inner[1]),
        )
        .channels();
        maximum_error = maximum_error.max(compare_channels(
            &[channels[1], channels[2], channels[4], channels[8]],
            &expected,
        ));
        fourth_order_witness = fourth_order_witness.max(expected[3].abs());
    }
    assert!(
        fourth_order_witness > 0.1,
        "the contracted fourth derivative must be resolved"
    );
    eprintln!(
        "#932 directional contractions: maximum_error={maximum_error:.6e} fourth_order_witness={fourth_order_witness:.6e}"
    );
}

#[test]
fn nested_dual2_seed_swap_symmetry_932() {
    let x = 0.4;
    let y = -0.55;
    let ab = smooth_program(
        &directional_seed(x, 1.0, 0.0),
        &directional_seed(y, 0.0, 1.0),
    )
    .channels();
    let ba = smooth_program(
        &directional_seed(x, 0.0, 1.0),
        &directional_seed(y, 1.0, 0.0),
    )
    .channels();
    let swapped = [
        ba[0], ba[2], ba[1], ba[5], ba[4], ba[3], ba[7], ba[6], ba[8],
    ];
    compare_channels(&ab, &swapped);
}

#[test]
fn nested_dual2_channels_follow_polynomial_derivative_order_932() {
    // The deleted from_channels convenience wrapper is not a live contract.
    // A polynomial with analytic derivatives 1..9 independently pins the public
    // channel order, including the factorial factors in repeated derivatives.
    let x = Dual2::variable(Dual2::constant(0.0));
    let y = Dual2::constant(Dual2::variable(0.0));
    let x_squared = x.mul(&x);
    let y_squared = y.mul(&y);
    let polynomial = x
        .constant_like(1.0)
        .add(&x.scale(2.0))
        .add(&y.scale(3.0))
        .add(&x_squared.scale(2.0))
        .add(&x.mul(&y).scale(5.0))
        .add(&y_squared.scale(3.0))
        .add(&x_squared.mul(&y).scale(3.5))
        .add(&x.mul(&y_squared).scale(4.0))
        .add(&x_squared.mul(&y_squared).scale(2.25));
    assert_eq!(
        polynomial.channels(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}
