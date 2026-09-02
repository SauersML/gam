use gam_row_macros::row_program;

const K: usize = 9;

#[derive(Clone, Copy)]
struct Kernel {
    w: f64,
    d: f64,
    u0: [f64; 5],
    censored_u1: [f64; 5],
    event_u1: [f64; 5],
    event_g: [f64; 5],
}

#[derive(Clone, Copy)]
struct Plan {
    u0: [f64; 5],
    u1: Option<[f64; 5]>,
    g: Option<[f64; 5]>,
}

#[inline(always)]
fn add_scaled(target: &mut [f64; 5], source: [f64; 5], scale: f64) {
    for i in 0..5 {
        target[i] += scale * source[i];
    }
}

#[inline(always)]
fn outer_plan(kernel: &Kernel) -> Plan {
    let mut u0 = [0.0; 5];
    add_scaled(&mut u0, kernel.u0, kernel.w);

    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let mut u1 = [0.0; 5];
    if censored_weight != 0.0 {
        add_scaled(&mut u1, kernel.censored_u1, -censored_weight);
    }
    if event_weight != 0.0 {
        add_scaled(&mut u1, kernel.event_u1, -event_weight);
    }
    let g = (event_weight != 0.0).then(|| {
        let mut stack = [0.0; 5];
        add_scaled(&mut stack, kernel.event_g, -event_weight);
        stack
    });
    Plan {
        u0,
        u1: (censored_weight != 0.0 || event_weight != 0.0).then_some(u1),
        g,
    }
}

#[inline(always)]
fn exp_stack(value: f64) -> [f64; 5] {
    let exp = value.exp();
    [exp; 5]
}

#[inline(always)]
fn preserve_composition_domain(point: f64, stack: [f64; 5]) -> [f64; 5] {
    if point.is_nan() { [f64::NAN; 5] } else { stack }
}

#[inline(always)]
fn outer_stack(
    composition_point: f64,
    value: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
) -> [f64; 5] {
    preserve_composition_domain(composition_point, [value, first, second, third, fourth])
}

row_program! {
    fn generated_sls(
        h0,
        h1,
        hdot,
        eta_t_exit,
        eta_t_entry,
        eta_t_deriv,
        eta_ls_exit,
        eta_ls_entry,
        eta_ls_deriv;
        u0_value,
        u0_first,
        u0_second,
        u0_third,
        u0_fourth,
        u1_value,
        u1_first,
        u1_second,
        u1_third,
        u1_fourth,
        g_value,
        g_first,
        g_second,
        g_third,
        g_fourth
    )
    emit [generic, order2, third, fourth];
    leaves {
        exponential => exp_stack => exp_stack_cuda,
        outer => outer_stack => outer_stack_cuda,
    }
    witnesses [];
    {
        let neg_eta_ls_entry = neg(eta_ls_entry);
        let inv_sigma_entry = compose(exponential, neg_eta_ls_entry);
        let u0 = add(h0, neg(mul(eta_t_entry, inv_sigma_entry)));

        let neg_eta_ls_exit = neg(eta_ls_exit);
        let inv_sigma_exit = compose(exponential, neg_eta_ls_exit);
        let u1 = add(h1, neg(mul(eta_t_exit, inv_sigma_exit)));
        let event_inner = add(mul(eta_t_exit, eta_ls_deriv), neg(eta_t_deriv));
        let g = add(hdot, mul(inv_sigma_exit, event_inner));

        let mut nll = zero();
        if (u0_value != 0.0 || u0_first != 0.0 || u0_second != 0.0 || u0_third != 0.0 || u0_fourth != 0.0) {
            nll = compose(
                outer,
                u0,
                u0_value,
                u0_first,
                u0_second,
                u0_third,
                u0_fourth
            );
        }
        if (u1_value != 0.0 || u1_first != 0.0 || u1_second != 0.0 || u1_third != 0.0 || u1_fourth != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    u1,
                    u1_value,
                    u1_first,
                    u1_second,
                    u1_third,
                    u1_fourth
                )
            );
        }
        if (g_value != 0.0 || g_first != 0.0 || g_second != 0.0 || g_third != 0.0 || g_fourth != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    g,
                    g_value,
                    g_first,
                    g_second,
                    g_third,
                    g_fourth
                )
            );
        }
        return nll;
    }
}

/// The stack of a plan slot the planner may leave absent: an absent slot is
/// the all-zero stack, which the program's own activity condition (`||` over
/// the entries) reads as inactive. The program takes no activity flag: with a
/// flag, the caller's scans ran before the inlined program could issue its
/// first leaf call, and the runner's disassembly showed both `exp` calls
/// held behind them while the hand kernel issues its two `exp` calls first.
#[inline(always)]
fn presence(slot: Option<[f64; 5]>) -> [f64; 5] {
    slot.unwrap_or([0.0; 5])
}

#[inline(never)]
fn generated_third(p: &[f64; K], kernel: &Kernel, direction: &[f64; K]) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let u1 = presence(plan.u1);
    let g = presence(plan.g);
    generated_sls_third_contracted(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        direction,
    )
}

#[inline(never)]
fn generated_fourth(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let u1 = presence(plan.u1);
    let g = presence(plan.g);
    generated_sls_fourth_contracted(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        direction_u,
        direction_v,
    )
}

#[inline(never)]
fn jet_third(p: &[f64; K], kernel: &Kernel, direction: &[f64; K]) -> [[f64; K]; K] {
    use gam_math::jet_scalar::OneSeed;

    let plan = outer_plan(kernel);
    let u1 = presence(plan.u1);
    let g = presence(plan.g);
    let vars: [OneSeed<K>; K] =
        std::array::from_fn(|axis| OneSeed::seed_direction(p[axis], axis, direction[axis]));
    let (value, []) = generated_sls(
        &vars[0],
        &vars[1],
        &vars[2],
        &vars[3],
        &vars[4],
        &vars[5],
        &vars[6],
        &vars[7],
        &vars[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    value.contracted_third()
}

#[inline(never)]
fn jet_fourth(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    use gam_math::jet_scalar::TwoSeed;

    let plan = outer_plan(kernel);
    let u1 = presence(plan.u1);
    let g = presence(plan.g);
    let vars: [TwoSeed<K>; K] = std::array::from_fn(|axis| {
        TwoSeed::seed(p[axis], axis, direction_u[axis], direction_v[axis])
    });
    let (value, []) = generated_sls(
        &vars[0],
        &vars[1],
        &vars[2],
        &vars[3],
        &vars[4],
        &vars[5],
        &vars[6],
        &vars[7],
        &vars[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    value.contracted_fourth()
}

const PERMUTATIONS_3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

const PERMUTATIONS_4: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

#[inline(always)]
fn hand_analytic_term<const ORDER: usize, const N: usize>(
    output: &mut [[f64; K]; K],
    active: [usize; N],
    stack: [f64; 5],
    directions: (&[f64; K], &[f64; K]),
    derivatives: (impl Fn(usize) -> f64, impl Fn(usize, usize) -> f64),
    terms: (&[([usize; 3], f64)], &[([usize; 4], f64)]),
) {
    let (direction_u, direction_v) = directions;
    let (d1, d2) = derivatives;
    let (third_terms, fourth_terms) = terms;
    let mut first = [0.0; N];
    let mut second = [[0.0; N]; N];
    let mut second_u = [0.0; N];
    let mut second_v = [0.0; N];
    let mut third_u = [[0.0; N]; N];
    let mut third_v = [[0.0; N]; N];
    let mut third_uv = [0.0; N];
    let mut fourth_uv = [[0.0; N]; N];
    let mut zu = 0.0;
    let mut zv = 0.0;
    let mut zuv = 0.0;
    for i in 0..N {
        first[i] = d1(active[i]);
        zu += first[i] * direction_u[active[i]];
        if ORDER == 4 {
            zv += first[i] * direction_v[active[i]];
        }
        for j in 0..N {
            second[i][j] = d2(active[i], active[j]);
            second_u[i] += second[i][j] * direction_u[active[j]];
            if ORDER == 4 {
                second_v[i] += second[i][j] * direction_v[active[j]];
                zuv += second[i][j] * direction_u[active[i]] * direction_v[active[j]];
            }
        }
    }
    for &(indices, coefficient) in third_terms {
        let mut seen = [[usize::MAX; 3]; 6];
        let mut seen_count = 0;
        for permutation in PERMUTATIONS_3 {
            let ordered = [
                indices[permutation[0]],
                indices[permutation[1]],
                indices[permutation[2]],
            ];
            if seen[..seen_count].contains(&ordered) {
                continue;
            }
            seen[seen_count] = ordered;
            seen_count += 1;
            let [i, j, k] = ordered;
            third_u[i][j] += coefficient * direction_u[active[k]];
            if ORDER == 4 {
                third_v[i][j] += coefficient * direction_v[active[k]];
                third_uv[i] += coefficient * direction_u[active[j]] * direction_v[active[k]];
            }
        }
    }
    if ORDER == 4 {
        for &(indices, coefficient) in fourth_terms {
            let mut seen = [[usize::MAX; 4]; 24];
            let mut seen_count = 0;
            for permutation in PERMUTATIONS_4 {
                let ordered = [
                    indices[permutation[0]],
                    indices[permutation[1]],
                    indices[permutation[2]],
                    indices[permutation[3]],
                ];
                if seen[..seen_count].contains(&ordered) {
                    continue;
                }
                seen[seen_count] = ordered;
                seen_count += 1;
                let [i, j, k, l] = ordered;
                fourth_uv[i][j] += coefficient * direction_u[active[k]] * direction_v[active[l]];
            }
        }
    }

    for i in 0..N {
        let a = active[i];
        let za = first[i];
        for j in 0..N {
            let b = active[j];
            let zb = first[j];
            let zab = second[i][j];
            if ORDER == 3 {
                output[a][b] += stack[3] * zu * za * zb
                    + stack[2] * (second_u[i] * zb + za * second_u[j] + zu * zab)
                    + stack[1] * third_u[i][j];
            } else {
                let f2_hessian = stack[4] * za * zb + stack[3] * zab;
                let f2_gradient_a = stack[3] * za;
                let f2_gradient_b = stack[3] * zb;
                let f2_zu_zv_hessian = f2_hessian * zu * zv
                    + stack[2] * third_u[i][j] * zv
                    + stack[2] * zu * third_v[i][j]
                    + f2_gradient_a * second_u[j] * zv
                    + f2_gradient_b * second_u[i] * zv
                    + f2_gradient_a * zu * second_v[j]
                    + f2_gradient_b * zu * second_v[i]
                    + stack[2] * second_u[i] * second_v[j]
                    + stack[2] * second_u[j] * second_v[i];
                let f1_hessian = stack[3] * za * zb + stack[2] * zab;
                let f1_zuv_hessian = f1_hessian * zuv
                    + stack[1] * fourth_uv[i][j]
                    + stack[2] * za * third_uv[j]
                    + stack[2] * zb * third_uv[i];
                output[a][b] += f2_zu_zv_hessian + f1_zuv_hessian;
            }
        }
    }
}

#[inline(never)]
fn hand_analytic_contracted<const ORDER: usize>(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let mut output = [[0.0; K]; K];

    if !plan.u0.iter().all(|value| *value == 0.0) {
        let exponential = (-p[7]).exp();
        let product = p[4] * exponential;
        hand_analytic_term::<ORDER, 3>(
            &mut output,
            [0, 4, 7],
            plan.u0,
            (direction_u, direction_v),
            (
            |axis| match axis {
                0 => 1.0,
                4 => -exponential,
                7 => product,
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [4, 7] => exponential,
                [7, 7] => -product,
                _ => 0.0,
            },
            ),
            (
            &[([1, 2, 2], -exponential), ([2, 2, 2], product)],
            &[([1, 2, 2, 2], exponential), ([2, 2, 2, 2], -product)],
            )
        );
    }

    if let Some(stack) = plan.u1 {
        let exponential = (-p[6]).exp();
        let product = p[3] * exponential;
        hand_analytic_term::<ORDER, 3>(
            &mut output,
            [1, 3, 6],
            stack,
            (direction_u, direction_v),
            (
            |axis| match axis {
                1 => 1.0,
                3 => -exponential,
                6 => product,
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [3, 6] => exponential,
                [6, 6] => -product,
                _ => 0.0,
            },
            ),
            (
            &[([1, 2, 2], -exponential), ([2, 2, 2], product)],
            &[([1, 2, 2, 2], exponential), ([2, 2, 2, 2], -product)],
            )
        );
    }

    if let Some(stack) = plan.g {
        let exponential = (-p[6]).exp();
        let inner = p[3] * p[8] - p[5];
        let product = exponential * inner;
        hand_analytic_term::<ORDER, 5>(
            &mut output,
            [2, 3, 5, 6, 8],
            stack,
            (direction_u, direction_v),
            (
            |axis| match axis {
                2 => 1.0,
                3 => exponential * p[8],
                5 => -exponential,
                6 => -product,
                8 => exponential * p[3],
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [3, 6] => -exponential * p[8],
                [3, 8] => exponential,
                [5, 6] => exponential,
                [6, 6] => product,
                [6, 8] => -exponential * p[3],
                _ => 0.0,
            },
            ),
            (
            &[
                ([1, 3, 3], exponential * p[8]),
                ([1, 3, 4], -exponential),
                ([2, 3, 3], -exponential),
                ([3, 3, 3], -product),
                ([3, 3, 4], exponential * p[3]),
            ],
            &[
                ([1, 3, 3, 3], -exponential * p[8]),
                ([1, 3, 3, 4], exponential),
                ([2, 3, 3, 3], exponential),
                ([3, 3, 3, 3], product),
                ([3, 3, 3, 4], -exponential * p[3]),
            ],
            )
        );
    }

    output
}

fn fixture() -> ([f64; K], Kernel) {
    (
        [0.4, -0.7, 0.2, 0.8, -0.35, 0.11, -0.25, 0.31, -0.17],
        Kernel {
            w: 1.3,
            d: 1.0,
            u0: [-0.8, -0.7, 0.3, -0.12, 0.05],
            censored_u1: [-1.1, -0.9, 0.4, -0.18, 0.08],
            event_u1: [-1.4, -0.6, -1.0, 0.0, 0.0],
            event_g: [-0.2, 1.4, -1.96, 5.488, -23.0496],
        },
    )
}

fn assert_matrix_close(got: [[f64; K]; K], want: [[f64; K]; K]) {
    for i in 0..K {
        for j in 0..K {
            let tolerance = 2e-11 * got[i][j].abs().max(want[i][j].abs()).max(1.0);
            assert!(
                (got[i][j] - want[i][j]).abs() <= tolerance,
                "[{i}][{j}] {:+.16e} vs {:+.16e}",
                got[i][j],
                want[i][j],
            );
        }
    }
}

#[test]
fn generated_sls_contracted_orders_match_canonical_jets_932() {
    let (p, kernel) = fixture();
    let direction_u = [0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8];
    let direction_v = [-0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1];
    for d in [0.0, 1.0, 0.37] {
        let endpoint = Kernel { d, ..kernel };
        let hand_third = hand_analytic_contracted::<3>(&p, &endpoint, &direction_u, &direction_v);
        let hand_fourth = hand_analytic_contracted::<4>(&p, &endpoint, &direction_u, &direction_v);
        assert_matrix_close(
            generated_third(&p, &endpoint, &direction_u),
            jet_third(&p, &endpoint, &direction_u),
        );
        assert_matrix_close(generated_third(&p, &endpoint, &direction_u), hand_third);
        assert_matrix_close(
            generated_fourth(&p, &endpoint, &direction_u, &direction_v),
            jet_fourth(&p, &endpoint, &direction_u, &direction_v),
        );
        assert_matrix_close(
            generated_fourth(&p, &endpoint, &direction_u, &direction_v),
            hand_fourth,
        );
    }
}

