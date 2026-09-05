use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use gam_math::probability::normal_logcdf_derivatives;
use gam_row_macros::row_program;

fn observed_scale_stack(observed_slope: f64) -> [f64; 5] {
    let scale = (1.0 + observed_slope * observed_slope).sqrt();
    let inverse = scale.recip();
    let inverse_squared = inverse * inverse;
    let inverse_cubed = inverse_squared * inverse;
    let inverse_fifth = inverse_cubed * inverse_squared;
    let inverse_seventh = inverse_fifth * inverse_squared;
    [
        scale,
        observed_slope * inverse,
        inverse_cubed,
        -3.0 * observed_slope * inverse_fifth,
        (12.0 * observed_slope * observed_slope - 3.0) * inverse_seventh,
    ]
}

fn signed_probit_stack(signed_margin: f64, weight: f64) -> [f64; 5] {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return [0.0; 5];
    }
    if signed_margin.is_nan() {
        return [f64::NAN; 5];
    }
    let derivatives = normal_logcdf_derivatives(signed_margin);
    derivatives.map(|derivative| -weight * derivative)
}

row_program! {
    fn generated_rigid_bms(
        marginal_eta,
        slope;
        marginal_q,
        marginal_q1,
        marginal_q2,
        marginal_q3,
        marginal_q4,
        probit_scale,
        latent_score,
        outcome_sign: sign,
        weight
    )
    emit [generic, order2, third, fourth, full];
    leaves {
        // The marginal-link stack is supplied, exactly as the production
        // program (`rigid_standard_normal_program`) declares it.
        supplied_link => supplied,
        observed_scale => observed_scale_stack => observed_scale_stack_cuda,
        signed_probit => signed_probit_stack => signed_probit_stack_cuda,
    }
    witnesses [];
    {
        let q = compose(
            supplied_link,
            marginal_eta,
            marginal_q,
            marginal_q1,
            marginal_q2,
            marginal_q3,
            marginal_q4
        );
        let observed_slope = scale(slope, probit_scale);
        let observed_scale_value = compose(observed_scale, observed_slope);
        let latent_index = add(
            mul(q, observed_scale_value),
            scale(observed_slope, latent_score)
        );
        let signed_margin = scale(latent_index, outcome_sign);
        return compose(signed_probit, signed_margin, weight);
    }
}

type Channels = (f64, [f64; 2], [[f64; 2]; 2]);

#[derive(Clone, Copy)]
struct Row {
    marginal_eta: f64,
    slope: f64,
    marginal: [f64; 5],
    probit_scale: f64,
    latent_score: f64,
    outcome_sign: f64,
    weight: f64,
    direction_u: [f64; 2],
    direction_v: [f64; 2],
}

#[inline(always)]
fn generated(row: Row) -> Channels {
    let (value, gradient, hessian, []) = generated_rigid_bms_order2(
        row.marginal_eta,
        row.slope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    );
    (value, gradient, hessian)
}

#[inline(always)]
fn strongest_hand(row: Row) -> Channels {
    let observed_slope = row.probit_scale * row.slope;
    let scale = (1.0 + observed_slope * observed_slope).sqrt();
    let inverse_scale = scale.recip();
    let scale_first = row.probit_scale * observed_slope * inverse_scale;
    let scale_second =
        row.probit_scale * row.probit_scale * inverse_scale * inverse_scale * inverse_scale;
    let latent_index = row.marginal[0] * scale + observed_slope * row.latent_score;
    let signed_margin = row.outcome_sign * latent_index;
    let outer = signed_probit_stack(signed_margin, row.weight);
    let outer_first = row.outcome_sign * outer[1];
    let outer_second = outer[2];
    let eta_marginal = row.marginal[1] * scale;
    let eta_slope = row.marginal[0] * scale_first + row.probit_scale * row.latent_score;
    let gradient = [outer_first * eta_marginal, outer_first * eta_slope];
    let cross =
        outer_second * eta_marginal * eta_slope + outer_first * row.marginal[1] * scale_first;
    (
        outer[0],
        gradient,
        [
            [
                outer_second * eta_marginal * eta_marginal + outer_first * row.marginal[2] * scale,
                cross,
            ],
            [
                cross,
                outer_second * eta_slope * eta_slope + outer_first * row.marginal[0] * scale_second,
            ],
        ],
    )
}

#[inline(always)]
fn generated_third(row: Row) -> [[f64; 2]; 2] {
    generated_rigid_bms_third_contracted(
        row.marginal_eta,
        row.slope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
        &row.direction_u,
    )
}

#[inline(always)]
fn generated_fourth(row: Row) -> [[f64; 2]; 2] {
    generated_rigid_bms_fourth_contracted(
        row.marginal_eta,
        row.slope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
        &row.direction_u,
        &row.direction_v,
    )
}

#[inline(always)]
fn generated_third_full(row: Row) -> [[[f64; 2]; 2]; 2] {
    generated_rigid_bms_third_full(
        row.marginal_eta,
        row.slope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    )
}

#[inline(always)]
fn generated_fourth_full(row: Row) -> [[[[f64; 2]; 2]; 2]; 2] {
    generated_rigid_bms_fourth_full(
        row.marginal_eta,
        row.slope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    )
}

#[inline(always)]
fn margin_chain(row: Row) -> ([f64; 5], [f64; 2], [f64; 3], [f64; 4], [f64; 5]) {
    let observed_slope = row.probit_scale * row.slope;
    let observed_stack = observed_scale_stack(observed_slope);
    let mut scale_stack = [0.0; 5];
    let mut scale_power = 1.0;
    for order in 0..5 {
        scale_stack[order] = observed_stack[order] * scale_power;
        scale_power *= row.probit_scale;
    }
    let derivative = |order: usize, g_axes: usize| {
        let eta_axes = order - g_axes;
        let linear = if eta_axes == 0 && g_axes == 1 {
            row.probit_scale * row.latent_score
        } else {
            0.0
        };
        row.outcome_sign * (row.marginal[eta_axes] * scale_stack[g_axes] + linear)
    };
    let signed_margin = row.outcome_sign
        * (row.marginal[0] * observed_stack[0] + observed_slope * row.latent_score);
    (
        signed_probit_stack(signed_margin, row.weight),
        std::array::from_fn(|g_axes| derivative(1, g_axes)),
        std::array::from_fn(|g_axes| derivative(2, g_axes)),
        std::array::from_fn(|g_axes| derivative(3, g_axes)),
        std::array::from_fn(|g_axes| derivative(4, g_axes)),
    )
}

#[inline(always)]
fn dot2(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

#[inline(always)]
fn contract2(derivative: &[f64; 3], left: [f64; 2], right: [f64; 2]) -> f64 {
    derivative[0] * left[0] * right[0]
        + derivative[1] * (left[0] * right[1] + left[1] * right[0])
        + derivative[2] * left[1] * right[1]
}

#[inline(always)]
fn strongest_hand_third(row: Row) -> [[f64; 2]; 2] {
    let (outer, d1, d2, d3, _) = margin_chain(row);
    let margin_u = dot2(d1, row.direction_u);
    std::array::from_fn(|axis_a| {
        std::array::from_fn(|axis_b| {
            let margin_a = d1[axis_a];
            let margin_b = d1[axis_b];
            let margin_ab = d2[axis_a + axis_b];
            let margin_au = d2[axis_a] * row.direction_u[0] + d2[axis_a + 1] * row.direction_u[1];
            let margin_bu = d2[axis_b] * row.direction_u[0] + d2[axis_b + 1] * row.direction_u[1];
            let margin_abu = d3[axis_a + axis_b] * row.direction_u[0]
                + d3[axis_a + axis_b + 1] * row.direction_u[1];
            outer[3] * margin_u * margin_a * margin_b
                + outer[2] * (margin_au * margin_b + margin_a * margin_bu + margin_u * margin_ab)
                + outer[1] * margin_abu
        })
    })
}

#[inline(always)]
fn strongest_hand_fourth(row: Row) -> [[f64; 2]; 2] {
    let (outer, d1, d2, d3, d4) = margin_chain(row);
    let margin_u = dot2(d1, row.direction_u);
    let margin_v = dot2(d1, row.direction_v);
    let margin_uv = contract2(&d2, row.direction_u, row.direction_v);
    std::array::from_fn(|axis_a| {
        std::array::from_fn(|axis_b| {
            let margin_a = d1[axis_a];
            let margin_b = d1[axis_b];
            let margin_ab = d2[axis_a + axis_b];
            let margin_au = d2[axis_a] * row.direction_u[0] + d2[axis_a + 1] * row.direction_u[1];
            let margin_av = d2[axis_a] * row.direction_v[0] + d2[axis_a + 1] * row.direction_v[1];
            let margin_bu = d2[axis_b] * row.direction_u[0] + d2[axis_b + 1] * row.direction_u[1];
            let margin_bv = d2[axis_b] * row.direction_v[0] + d2[axis_b + 1] * row.direction_v[1];
            let margin_abu = d3[axis_a + axis_b] * row.direction_u[0]
                + d3[axis_a + axis_b + 1] * row.direction_u[1];
            let margin_abv = d3[axis_a + axis_b] * row.direction_v[0]
                + d3[axis_a + axis_b + 1] * row.direction_v[1];
            let margin_auv = d3[axis_a] * row.direction_u[0] * row.direction_v[0]
                + d3[axis_a + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d3[axis_a + 2] * row.direction_u[1] * row.direction_v[1];
            let margin_buv = d3[axis_b] * row.direction_u[0] * row.direction_v[0]
                + d3[axis_b + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d3[axis_b + 2] * row.direction_u[1] * row.direction_v[1];
            let margin_abuv = d4[axis_a + axis_b] * row.direction_u[0] * row.direction_v[0]
                + d4[axis_a + axis_b + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d4[axis_a + axis_b + 2] * row.direction_u[1] * row.direction_v[1];
            let second_chain = margin_au * margin_b + margin_a * margin_bu + margin_u * margin_ab;
            let second_chain_v = margin_auv * margin_b
                + margin_au * margin_bv
                + margin_av * margin_bu
                + margin_a * margin_buv
                + margin_uv * margin_ab
                + margin_u * margin_abv;
            outer[4] * margin_v * margin_u * margin_a * margin_b
                + outer[3]
                    * (margin_uv * margin_a * margin_b
                        + margin_u * margin_av * margin_b
                        + margin_u * margin_a * margin_bv
                        + margin_v * second_chain)
                + outer[2] * (second_chain_v + margin_v * margin_abu)
                + outer[1] * margin_abuv
        })
    })
}

#[inline(always)]
fn strongest_hand_third_full(row: Row) -> [[[f64; 2]; 2]; 2] {
    let (outer, d1, d2, d3, _) = margin_chain(row);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            std::array::from_fn(|c| {
                outer[3] * d1[a] * d1[b] * d1[c]
                    + outer[2] * (d2[a + b] * d1[c] + d2[a + c] * d1[b] + d2[b + c] * d1[a])
                    + outer[1] * d3[a + b + c]
            })
        })
    })
}

#[inline(always)]
fn strongest_hand_fourth_full(row: Row) -> [[[[f64; 2]; 2]; 2]; 2] {
    let (outer, d1, d2, d3, d4) = margin_chain(row);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            std::array::from_fn(|c| {
                std::array::from_fn(|d| {
                    outer[4] * d1[a] * d1[b] * d1[c] * d1[d]
                        + outer[3]
                            * (d2[a + b] * d1[c] * d1[d]
                                + d2[a + c] * d1[b] * d1[d]
                                + d2[a + d] * d1[b] * d1[c]
                                + d2[b + c] * d1[a] * d1[d]
                                + d2[b + d] * d1[a] * d1[c]
                                + d2[c + d] * d1[a] * d1[b])
                        + outer[2]
                            * (d3[a + b + c] * d1[d]
                                + d3[a + b + d] * d1[c]
                                + d3[a + c + d] * d1[b]
                                + d3[b + c + d] * d1[a]
                                + d2[a + b] * d2[c + d]
                                + d2[a + c] * d2[b + d]
                                + d2[a + d] * d2[b + c])
                        + outer[1] * d4[a + b + c + d]
                })
            })
        })
    })
}

fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            let marginal_eta = 0.9 * (x * 0.17 + 0.3).sin();
            let q = 0.8 * marginal_eta + 0.12 * marginal_eta * marginal_eta;
            Row {
                marginal_eta,
                slope: 0.7 * (x * 0.11 + 0.4).cos(),
                marginal: [q, 0.8 + 0.24 * marginal_eta, 0.24, 0.0, 0.0],
                probit_scale: 0.8,
                latent_score: 1.4 * (x * 0.13 + 0.2).sin(),
                outcome_sign: if index % 2 == 0 { 1.0 } else { -1.0 },
                weight: 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs(),
                direction_u: [0.7 * (x * 0.23 + 0.4).cos(), -0.6 * (x * 0.29 + 0.1).sin()],
                direction_v: [-0.5 * (x * 0.21 + 0.9).sin(), 0.8 * (x * 0.27 + 0.5).cos()],
            }
        })
        .collect()
}

fn close(got: f64, want: f64) {
    let tolerance = 3e-12 * got.abs().max(want.abs()).max(1.0);
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

fn assert_third_full(got: [[[f64; 2]; 2]; 2], want: [[[f64; 2]; 2]; 2]) {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                close(got[a][b][c], want[a][b][c]);
            }
        }
    }
}

fn assert_fourth_full(got: [[[[f64; 2]; 2]; 2]; 2], want: [[[[f64; 2]; 2]; 2]; 2]) {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    close(got[a][b][c][d], want[a][b][c][d]);
                }
            }
        }
    }
}

/// One pass over every row, folded to a scalar the paired harness accumulates:
/// each row perturbs the observed slope by the nudge, so no row can be hoisted
/// or merged across iterations, and the rows stay independent of one another
/// as production's rows are.
fn channels_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> Channels) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.slope += nudge;
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
        perturbed.slope += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

fn third_full_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> [[[f64; 2]; 2]; 2]) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.slope += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|plane| plane.iter())
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

fn fourth_full_pass(
    rows: &[Row],
    nudge: f64,
    evaluate: impl Fn(Row) -> [[[[f64; 2]; 2]; 2]; 2],
) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.slope += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|cube| cube.iter())
            .flat_map(|plane| plane.iter())
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

#[test]
fn generated_rigid_bms_matches_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated(*row), strongest_hand(*row));
        assert_matrix(generated_third(*row), strongest_hand_third(*row));
        assert_matrix(generated_fourth(*row), strongest_hand_fourth(*row));
        assert_third_full(generated_third_full(*row), strongest_hand_third_full(*row));
        assert_fourth_full(
            generated_fourth_full(*row),
            strongest_hand_fourth_full(*row),
        );
    }

    // Parity above runs in every build; the speed contract opens only in the
    // release profile (`SpeedGate::open` documents why) and takes one paired,
    // interleaved, order-randomised measurement per channel. Every channel
    // is `faster`: the generated lowering must beat the strongest direct
    // analytic schedule of the same row. (This gate once carried its own
    // seven-round harness and asserted in the dev lane as well, where the
    // codegen it measured is not the shipped one; it is now one of the
    // derived population and measured by the one instrument.)
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("RIGID-BMS-HAND-932");
    let reps = 15usize;
    let passes = 128usize;
    for (channel, timing) in [
        (
            "order2",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B002,
                |nudge| channels_pass(&rows, nudge, generated),
                |nudge| channels_pass(&rows, nudge, strongest_hand),
            ),
        ),
        (
            "third",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B003,
                |nudge| matrix_pass(&rows, nudge, generated_third),
                |nudge| matrix_pass(&rows, nudge, strongest_hand_third),
            ),
        ),
        (
            "fourth",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B004,
                |nudge| matrix_pass(&rows, nudge, generated_fourth),
                |nudge| matrix_pass(&rows, nudge, strongest_hand_fourth),
            ),
        ),
        (
            "third_full",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B013,
                |nudge| third_full_pass(&rows, nudge, generated_third_full),
                |nudge| third_full_pass(&rows, nudge, strongest_hand_third_full),
            ),
        ),
        (
            "fourth_full",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B014,
                |nudge| fourth_full_pass(&rows, nudge, generated_fourth_full),
                |nudge| fourth_full_pass(&rows, nudge, strongest_hand_fourth_full),
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
