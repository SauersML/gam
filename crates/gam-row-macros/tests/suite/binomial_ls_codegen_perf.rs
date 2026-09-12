use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use gam_row_macros::row_program;

// The binomial location-scale row program, declared exactly as production's
// `binomial_ls_row_program` (gam-models gamlss/binomial/kernel.rs): local
// coordinates `q(δ) = (q0 − δ_t/σ)·e^{−δ_ls}` around the row, the exponential's
// stack at `δ_ls = 0` supplied as ones, the q-space loss stack supplied, and an
// all-zero loss stack skipped instead of composed.
row_program! {
    fn generated_binomial_ls(
        delta_eta_t,
        delta_eta_ls;
        q0,
        inv_sigma,
        neg_ll,
        m1,
        m2,
        m3,
        m4
    )
    emit [order2, third, fourth];
    leaves {
        unit_exponential => supplied,
        loss => supplied,
    }
    witnesses [];
    {
        let neg_delta_eta_ls = neg(delta_eta_ls);
        let scale_ratio = compose(unit_exponential, neg_delta_eta_ls, 1.0, 1.0, 1.0, 1.0, 1.0);
        let shifted_q = add_constant(scale(delta_eta_t, -inv_sigma), q0);
        let q = mul(shifted_q, scale_ratio);
        let mut nll = zero();
        if (neg_ll != 0.0 || m1 != 0.0 || m2 != 0.0 || m3 != 0.0 || m4 != 0.0) {
            nll = compose(loss, q, neg_ll, m1, m2, m3, m4);
        }
        return nll;
    }
}

type Channels = ([f64; 2], [[f64; 2]; 2]);

#[derive(Clone, Copy)]
struct Row {
    q0: f64,
    inv_sigma: f64,
    stack: [f64; 4],
    direction_u: [f64; 2],
    direction_v: [f64; 2],
}

/// Production's order-2 call: the value entry and the entries the surface does
/// not read passed as literal zeros, as the row-coefficient builder does.
#[inline(always)]
fn generated_order2(row: Row) -> Channels {
    let [m1, m2, _, _] = row.stack;
    let (_, gradient, hessian, []) =
        generated_binomial_ls_order2(0.0, 0.0, row.q0, row.inv_sigma, 0.0, m1, m2, 0.0, 0.0);
    (gradient, hessian)
}

#[inline(always)]
fn generated_third(row: Row) -> [[f64; 2]; 2] {
    let [m1, m2, m3, _] = row.stack;
    generated_binomial_ls_third_contracted(
        0.0,
        0.0,
        row.q0,
        row.inv_sigma,
        0.0,
        m1,
        m2,
        m3,
        0.0,
        &row.direction_u,
    )
}

#[inline(always)]
fn generated_fourth(row: Row) -> [[f64; 2]; 2] {
    let [m1, m2, m3, m4] = row.stack;
    generated_binomial_ls_fourth_contracted(
        0.0,
        0.0,
        row.q0,
        row.inv_sigma,
        0.0,
        m1,
        m2,
        m3,
        m4,
        &row.direction_u,
        &row.direction_v,
    )
}

/// The retired inner-Newton closed form (`m2 r²`, `r (m1 + q m2)`,
/// `q (m1 + q m2)` with `r = 1/σ`), carrying the program's activity rule.
#[inline(always)]
fn strongest_hand_order2(row: Row) -> Channels {
    let [a, b, _, _] = row.stack;
    if a == 0.0 && b == 0.0 {
        return ([0.0; 2], [[0.0; 2]; 2]);
    }
    let q = row.q0;
    let r = row.inv_sigma;
    let u = a + q * b;
    let h_tl = r * u;
    (
        [-a * r, -a * q],
        [[b * r * r, h_tl], [h_tl, q * u]],
    )
}

/// The retired first-order psi coefficient drift `d_z h` along a predictor
/// direction `z`, carrying the program's activity rule.
#[inline(always)]
fn strongest_hand_third(row: Row) -> [[f64; 2]; 2] {
    let [a, b, c, _] = row.stack;
    if a == 0.0 && b == 0.0 && c == 0.0 {
        return [[0.0; 2]; 2];
    }
    let q = row.q0;
    let r = row.inv_sigma;
    let [z_t, z_ls] = row.direction_u;
    let q_z = -r * z_t - q * z_ls;
    let u = a + q * b;
    let dh_tt = r * r * (c * q_z - 2.0 * b * z_ls);
    let dh_tl = r * ((2.0 * b + c * q) * q_z - u * z_ls);
    let dh_ll = (a + 3.0 * q * b + q * q * c) * q_z;
    [[dh_tt, dh_tl], [dh_tl, dh_ll]]
}

/// The retired second-order psi coefficient drift `d_ij h` along two predictor
/// directions with no second drift, carrying the program's activity rule.
#[inline(always)]
fn strongest_hand_fourth(row: Row) -> [[f64; 2]; 2] {
    let [a, b, c, d] = row.stack;
    if a == 0.0 && b == 0.0 && c == 0.0 && d == 0.0 {
        return [[0.0; 2]; 2];
    }
    let q = row.q0;
    let r = row.inv_sigma;
    let [z_t_i, z_ls_i] = row.direction_u;
    let [z_t_j, z_ls_j] = row.direction_v;
    let q_i = -r * z_t_i - q * z_ls_i;
    let q_j = -r * z_t_j - q * z_ls_j;
    let q_ij = r * (z_t_i * z_ls_j + z_t_j * z_ls_i) + q * z_ls_i * z_ls_j;
    let u = a + q * b;
    let v = 2.0 * b + q * c;
    let cross = q_j * z_ls_i + q_i * z_ls_j;
    let dd_tt = r * r * (d * q_i * q_j + c * q_ij - 2.0 * c * cross + 4.0 * b * z_ls_i * z_ls_j);
    let dd_tl = r * ((3.0 * c + q * d) * q_j * q_i + v * q_ij - v * cross + u * z_ls_i * z_ls_j);
    let dd_ll = (4.0 * b + 5.0 * q * c + q * q * d) * q_i * q_j + (a + 3.0 * q * b + q * q * c) * q_ij;
    [[dd_tt, dd_tl], [dd_tl, dd_ll]]
}

/// Rows of a logistic threshold-scale fit: `q0 = −η_t/σ`, and the loss stack is
/// the exact derivatives of `−w[y log μ + (1−y) log(1−μ)]` in `q` at `μ = σ(q0)`.
fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            let eta_t = 1.6 * (x * 0.17 + 0.3).sin() - 0.4 * (x * 0.09).cos();
            let eta_ls = 0.8 * (x * 0.11 + 0.2).sin() - 0.25 * (x * 0.07).cos();
            let inv_sigma = (-eta_ls).exp();
            let q0 = -eta_t * inv_sigma;
            let weight = 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs();
            let y = if index % 2 == 0 { 1.0 } else { 0.0 };
            let mu = 1.0 / (1.0 + (-q0).exp());
            let variance = mu * (1.0 - mu);
            Row {
                q0,
                inv_sigma,
                stack: [
                    weight * (mu - y),
                    weight * variance,
                    weight * variance * (1.0 - 2.0 * mu),
                    weight * variance * (1.0 - 6.0 * variance),
                ],
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
    let tolerance = 3e-12 * got.abs().max(want.abs()).max(1.0);
    assert!(
        (got - want).abs() <= tolerance,
        "{got:+.16e} vs {want:+.16e}"
    );
}

fn assert_channels(got: Channels, want: Channels) {
    for axis in 0..2 {
        close(got.0[axis], want.0[axis]);
        for other in 0..2 {
            close(got.1[axis][other], want.1[axis][other]);
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
/// each row perturbs the threshold map by the nudge, so no row can be hoisted
/// or merged across iterations, and the rows stay independent of one another
/// as production's rows are.
fn channels_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> Channels) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.q0 += nudge;
        let (gradient, hessian) = evaluate(perturbed);
        fold += gradient.iter().sum::<f64>()
            + hessian.iter().flat_map(|line| line.iter()).sum::<f64>();
    }
    fold
}

fn matrix_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> [[f64; 2]; 2]) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.q0 += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

#[test]
fn generated_binomial_ls_matches_strongest_hand_932() {
    let rows = rows();
    let mut largest = 0.0_f64;
    for row in &rows {
        let hand = strongest_hand_order2(*row);
        assert_channels(generated_order2(*row), hand);
        assert_matrix(generated_third(*row), strongest_hand_third(*row));
        assert_matrix(generated_fourth(*row), strongest_hand_fourth(*row));
        largest = largest.max(hand.1[1][1].abs());
    }
    assert!(
        largest > 1.0e-2,
        "fixture never reached a resolvable log-scale curvature (largest {largest:e})"
    );

    // Parity above runs in every build; the speed contract opens only in the
    // release profile (`SpeedGate::open` documents why) and takes one paired,
    // interleaved, order-randomised measurement per channel. Every channel is
    // `faster`: the generated lowering must beat the retired production hand
    // schedule of the same contract.
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("BINOMIAL-LS-HAND-932");
    let reps = 15usize;
    let passes = 256usize;
    for (channel, timing) in [
        (
            "order2",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B1A2,
                |nudge| channels_pass(&rows, nudge, generated_order2),
                |nudge| channels_pass(&rows, nudge, strongest_hand_order2),
            ),
        ),
        (
            "third",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B1A3,
                |nudge| matrix_pass(&rows, nudge, generated_third),
                |nudge| matrix_pass(&rows, nudge, strongest_hand_third),
            ),
        ),
        (
            "fourth",
            paired_interleaved(
                reps,
                passes,
                0x932_0_B1A4,
                |nudge| matrix_pass(&rows, nudge, generated_fourth),
                |nudge| matrix_pass(&rows, nudge, strongest_hand_fourth),
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
