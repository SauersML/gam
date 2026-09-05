use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use gam_row_macros::row_atom;

row_atom! {
    fn generated_gaussian [order2_at_zero, third_at_zero, fourth_at_zero](
        delta_mu,
        delta_eta;
        obs_weight: f64,
        standardized_residual: f64,
        inv_sigma: f64,
        kappa: f64
    ) {
        obs_weight * ln((1.0 - kappa) + kappa * exp(delta_eta))
            + 0.5
                * obs_weight
                * (standardized_residual - delta_mu * inv_sigma)
                * (standardized_residual - delta_mu * inv_sigma)
                / ((1.0 - kappa) + kappa * exp(delta_eta))
                / ((1.0 - kappa) + kappa * exp(delta_eta))
    }
}

type Channels = (f64, [f64; 2], [[f64; 2]; 2]);

#[derive(Clone, Copy)]
struct Row {
    weight: f64,
    residual: f64,
    inv_sigma: f64,
    kappa: f64,
    direction_u: [f64; 2],
    direction_v: [f64; 2],
}

#[inline(always)]
fn generated_order2(row: Row) -> Channels {
    let atom =
        generated_gaussian_order2_at_zero(row.weight, row.residual, row.inv_sigma, row.kappa);
    let gradient = atom.gradient();
    (
        atom.value(),
        gradient,
        [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ],
    )
}

#[inline(always)]
fn generated_third(row: Row) -> [[f64; 2]; 2] {
    generated_gaussian_third_contracted_at_zero(
        row.weight,
        row.residual,
        row.inv_sigma,
        row.kappa,
        &row.direction_u,
    )
}

#[inline(always)]
fn generated_fourth(row: Row) -> [[f64; 2]; 2] {
    generated_gaussian_fourth_contracted_at_zero(
        row.weight,
        row.residual,
        row.inv_sigma,
        row.kappa,
        &row.direction_u,
        &row.direction_v,
    )
}

#[inline(always)]
fn hand_order2(row: Row) -> Channels {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let wr = w * r;
    let wq = w * q;
    let h00 = wq * q;
    let h01 = 2.0 * wr * q * k;
    let h11 = w * (k - k2 + r * r * (-k + 3.0 * k2));
    (
        0.5 * wr * r,
        [-wq * r, w * k * (1.0 - r * r)],
        [[h00, h01], [h01, h11]],
    )
}

#[inline(always)]
fn hand_third(row: Row) -> [[f64; 2]; 2] {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let k3 = k2 * k;
    let c2 = -2.0 * k + 6.0 * k2;
    let c3 = -2.0 * k + 18.0 * k2 - 24.0 * k3;
    let l3 = k - 3.0 * k2 + 2.0 * k3;
    let xxe = -2.0 * w * q * q * k;
    let xee = -w * r * q * c2;
    let eee = w * (l3 + 0.5 * r * r * c3);
    let [u0, u1] = row.direction_u;
    [
        [xxe * u1, xxe * u0 + xee * u1],
        [xxe * u0 + xee * u1, xee * u0 + eee * u1],
    ]
}

#[inline(always)]
fn hand_fourth(row: Row) -> [[f64; 2]; 2] {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k2 * k2;
    let c2 = -2.0 * k + 6.0 * k2;
    let c3 = -2.0 * k + 18.0 * k2 - 24.0 * k3;
    let c4 = -2.0 * k + 42.0 * k2 - 144.0 * k3 + 120.0 * k4;
    let l4 = k - 7.0 * k2 + 12.0 * k3 - 6.0 * k4;
    let xxee = w * q * q * c2;
    let xeee = -w * r * q * c3;
    let eeee = w * (l4 + 0.5 * r * r * c4);
    let [u0, u1] = row.direction_u;
    let [v0, v1] = row.direction_v;
    let uv = u0 * v1 + u1 * v0;
    [
        [xxee * u1 * v1, xxee * uv + xeee * u1 * v1],
        [
            xxee * uv + xeee * u1 * v1,
            xxee * u0 * v0 + xeee * uv + eeee * u1 * v1,
        ],
    ]
}

fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            Row {
                weight: 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs(),
                residual: 1.4 * (x * 0.17 + 0.3).sin() - 0.5 * (x * 0.09).cos(),
                inv_sigma: (0.8 * (x * 0.11 + 0.2).sin() - 0.25 * (x * 0.07).cos())
                    .exp()
                    .recip(),
                kappa: 0.05 + 0.9 * (x * 0.13 + 0.7).cos().abs(),
                direction_u: [
                    0.7 * (x * 0.23 + 0.4).cos() - 0.2 * (x * 0.03).sin(),
                    -0.6 * (x * 0.29 + 0.1).sin() + 0.25 * (x * 0.15).cos(),
                ],
                direction_v: [
                    -0.5 * (x * 0.21 + 0.9).sin() + 0.3 * (x * 0.06).cos(),
                    0.8 * (x * 0.27 + 0.5).cos() - 0.15 * (x * 0.04).sin(),
                ],
            }
        })
        .collect()
}

fn close(got: f64, want: f64) {
    let tolerance = 2e-12 * got.abs().max(want.abs()).max(1.0);
    assert!(
        (got - want).abs() <= tolerance,
        "{got:+.16e} vs {want:+.16e}"
    );
}

fn assert_channels(got: Channels, want: Channels) {
    close(got.0, want.0);
    for axis in 0..2 {
        close(got.1[axis], want.1[axis]);
        for other in 0..2 {
            close(got.2[axis][other], want.2[axis][other]);
        }
    }
}

fn assert_matrix(got: [[f64; 2]; 2], want: [[f64; 2]; 2]) {
    for axis in 0..2 {
        for other in 0..2 {
            close(got[axis][other], want[axis][other]);
        }
    }
}

/// One pass over every row, folded to a scalar the paired harness accumulates:
/// each row perturbs the standardized residual by the nudge, so no row can be
/// hoisted or merged across iterations, and the rows stay independent of one
/// another as production's rows are.
fn channels_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> Channels) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.residual += nudge;
        let (value, gradient, hessian) = evaluate(perturbed);
        fold += value
            + gradient.iter().sum::<f64>()
            + hessian.iter().flat_map(|line| line.iter()).sum::<f64>();
    }
    fold
}

fn matrix_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> [[f64; 2]; 2]) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.residual += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

#[test]
fn generated_gaussian_matches_and_beats_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated_order2(*row), hand_order2(*row));
        assert_matrix(generated_third(*row), hand_third(*row));
        assert_matrix(generated_fourth(*row), hand_fourth(*row));
    }

    // Parity above runs in every build; the speed contract opens only in the
    // release profile (`SpeedGate::open` documents why) and takes one paired,
    // interleaved, order-randomised measurement per channel. Every channel is
    // `faster`: the generated lowering must beat the strongest hand schedule
    // of the same row. (This gate once kept a running minimum per arm over
    // seven alternating rounds and asserted in the dev lane as well; it is
    // now one of the derived population and measured by the one instrument.)
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("GAUSSIAN-JOINT-HAND-932");
    let reps = 15usize;
    let passes = 256usize;
    for (channel, timing) in [
        (
            "order2",
            paired_interleaved(
                reps,
                passes,
                0x932_0_6A02,
                |nudge| channels_pass(&rows, nudge, generated_order2),
                |nudge| channels_pass(&rows, nudge, hand_order2),
            ),
        ),
        (
            "third",
            paired_interleaved(
                reps,
                passes,
                0x932_0_6A03,
                |nudge| matrix_pass(&rows, nudge, generated_third),
                |nudge| matrix_pass(&rows, nudge, hand_third),
            ),
        ),
        (
            "fourth",
            paired_interleaved(
                reps,
                passes,
                0x932_0_6A04,
                |nudge| matrix_pass(&rows, nudge, generated_fourth),
                |nudge| matrix_pass(&rows, nudge, hand_fourth),
            ),
        ),
    ] {
        // `ns/iter` is nanoseconds per PASS over `rows.len()` rows; the ratio
        // the verdict rests on is unit-free. `median_ratio` is hand /
        // generated, so above 1 means the generated kernel is faster.
        gate.faster(
            &format!("channel={channel} rows={}", rows.len()),
            &timing,
            "generated",
            "strongest_hand",
        );
    }
    gate.finish();
}
