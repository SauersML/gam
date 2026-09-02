use gam_math::paired_timing::SpeedGate;
use gam_row_macros::row_program;

const K: usize = 9;

/// Rows evaluated inside ONE timed arm call.
///
/// `paired_interleaved` costs a closure call and a `black_box` per iteration.
/// A single SLS row is ~43 ns, so timing one row per call puts a fixed
/// overhead of the same order as the quantity being compared into BOTH arms.
/// Equal overhead can only compress a ratio toward 1, never flip it -- but it
/// also lets a small difference in how the two arms inline into the closure
/// dominate a few-percent codegen margin. Batching amortises the per-call cost
/// to under 1% of the arm, which is the regime the 512-row cause-specific
/// gate already measures in.
const ROWS_PER_ARM: usize = 64;

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
fn outer_plan_order2(kernel: &Kernel) -> Plan {
    let u0 = [
        kernel.w * kernel.u0[0],
        kernel.w * kernel.u0[1],
        kernel.w * kernel.u0[2],
        0.0,
        0.0,
    ];
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let u1 = (censored_weight != 0.0 || event_weight != 0.0).then(|| {
        let mut stack = [0.0; 5];
        if censored_weight != 0.0 {
            for i in 0..3 {
                stack[i] -= censored_weight * kernel.censored_u1[i];
            }
        }
        if event_weight != 0.0 {
            for i in 0..3 {
                stack[i] -= event_weight * kernel.event_u1[i];
            }
        }
        stack
    });
    let g = (event_weight != 0.0).then(|| {
        [
            -event_weight * kernel.event_g[0],
            -event_weight * kernel.event_g[1],
            -event_weight * kernel.event_g[2],
            0.0,
            0.0,
        ]
    });
    Plan { u0, u1, g }
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
        u0_active,
        u0_value,
        u0_first,
        u0_second,
        u0_third,
        u0_fourth,
        u1_active,
        u1_value,
        u1_first,
        u1_second,
        u1_third,
        u1_fourth,
        g_active,
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
        if (u0_active != 0.0) {
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
        if (u1_active != 0.0) {
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
        if (g_active != 0.0) {
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

type Channels = (f64, [f64; K], [[f64; K]; K]);

#[inline(always)]
/// The activity flag and stack of a plan slot the planner may leave absent.
///
/// The hand arm reads the `Option` discriminant (`if let Some(u1) = plan.u1`);
/// the generated arm must read the same bit, not flatten the slot to zeros
/// and rediscover its absence with a five-way compare scan. Both arms then
/// do the same scaffolding on the same plan, and the timing compares the
/// kernels.
fn presence(slot: Option<[f64; 5]>) -> (f64, [f64; 5]) {
    match slot {
        Some(stack) => (1.0, stack),
        None => (0.0, [0.0; 5]),
    }
}

fn stack_active(stack: &[f64; 5]) -> f64 {
    if stack.iter().all(|value| *value == 0.0) {
        0.0
    } else {
        1.0
    }
}

#[inline(never)]
fn generated(p: &[f64; K], kernel: &Kernel) -> Channels {
    let plan = outer_plan_order2(kernel);
    let (u1_active, u1) = presence(plan.u1);
    let (g_active, g) = presence(plan.g);
    let (value, gradient, hessian, []) = generated_sls_order2(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1_active,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g_active,
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    (value, gradient, hessian)
}

#[inline(never)]
fn generated_third(p: &[f64; K], kernel: &Kernel, direction: &[f64; K]) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let (u1_active, u1) = presence(plan.u1);
    let (g_active, g) = presence(plan.g);
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
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1_active,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g_active,
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
    let (u1_active, u1) = presence(plan.u1);
    let (g_active, g) = presence(plan.g);
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
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1_active,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g_active,
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
    let (u1_active, u1) = presence(plan.u1);
    let (g_active, g) = presence(plan.g);
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
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1_active,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g_active,
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
    let (u1_active, u1) = presence(plan.u1);
    let (g_active, g) = presence(plan.g);
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
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        u1_active,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        g_active,
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

    if stack_active(&plan.u0) != 0.0 {
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

/// The strongest hand order-2 kernel for the same row.
///
/// It consumes the same `outer_plan_order2` plan the generated arm consumes,
/// through the same `Option` discriminants, so the paired timing compares the
/// two kernels and not two planners. (An earlier hand arm rebuilt the stacks
/// inline as three scalars per slot and never materialised the plan; it was
/// stronger than the production hand kernel, which shares the planner with
/// the generated kernel, and the deficit it measured was the plan's round
/// trip through the stack, not the compiled program.)
#[inline(never)]
fn hand(p: &[f64; K], kernel: &Kernel) -> Channels {
    let plan = outer_plan_order2(kernel);
    let entry_exp = (-p[7]).exp();
    let exit_exp = (-p[6]).exp();

    let u0_point = p[0] - p[4] * entry_exp;
    let u0_active = !plan.u0.iter().all(|value| *value == 0.0);
    let u0_stack = if u0_active {
        preserve_composition_domain(u0_point, plan.u0)
    } else {
        plan.u0
    };
    let mut value = u0_stack[0];
    let u0_first = u0_stack[1];
    let u0_second = u0_stack[2];

    let u1_point = p[1] - p[3] * exit_exp;
    let u1_active = plan.u1.is_some();
    let [u1_value, u1_first, u1_second, _, _] = match plan.u1 {
        Some(stack) => preserve_composition_domain(u1_point, stack),
        None => [0.0; 5],
    };
    value += u1_value;

    let inner = p[3] * p[8] - p[5];
    let g_point = p[2] + exit_exp * inner;
    let g_active = plan.g.is_some();
    let [g_value, g_first, g_second, _, _] = match plan.g {
        Some(stack) => preserve_composition_domain(g_point, stack),
        None => [0.0; 5],
    };
    value += g_value;

    let u0_g4 = -entry_exp;
    let u0_g7 = p[4] * entry_exp;
    let u1_g3 = -exit_exp;
    let u1_g6 = p[3] * exit_exp;
    let g3 = exit_exp * p[8];
    let g5 = -exit_exp;
    let g6 = -exit_exp * inner;
    let g8 = exit_exp * p[3];

    let mut gradient = [0.0; K];
    if u0_active {
        gradient[0] = u0_first;
        gradient[4] = u0_first * u0_g4;
        gradient[7] = u0_first * u0_g7;
    }
    if u1_active {
        gradient[1] = u1_first;
        gradient[3] = u1_first * u1_g3;
        gradient[6] = u1_first * u1_g6;
    }
    if g_active {
        gradient[2] = g_first;
        gradient[3] += g_first * g3;
        gradient[5] = g_first * g5;
        gradient[6] += g_first * g6;
        gradient[8] = g_first * g8;
    }

    let mut hessian = [[0.0; K]; K];
    macro_rules! symmetric {
        ($i:expr, $j:expr, $channel:expr) => {{
            let channel = $channel;
            hessian[$i][$j] += channel;
            if $i != $j {
                hessian[$j][$i] += channel;
            }
        }};
    }

    if u0_active {
        symmetric!(0, 0, u0_second);
        symmetric!(0, 4, u0_second * u0_g4);
        symmetric!(0, 7, u0_second * u0_g7);
        symmetric!(4, 4, u0_second * u0_g4 * u0_g4);
        symmetric!(4, 7, u0_second * u0_g4 * u0_g7 + u0_first * entry_exp);
        symmetric!(7, 7, u0_second * u0_g7 * u0_g7 - u0_first * u0_g7);
    }

    if u1_active {
        symmetric!(1, 1, u1_second);
        symmetric!(1, 3, u1_second * u1_g3);
        symmetric!(1, 6, u1_second * u1_g6);
        symmetric!(3, 3, u1_second * u1_g3 * u1_g3);
        symmetric!(3, 6, u1_second * u1_g3 * u1_g6 + u1_first * exit_exp);
        symmetric!(6, 6, u1_second * u1_g6 * u1_g6 - u1_first * u1_g6);
    }

    if g_active {
        symmetric!(2, 2, g_second);
        symmetric!(2, 3, g_second * g3);
        symmetric!(2, 5, g_second * g5);
        symmetric!(2, 6, g_second * g6);
        symmetric!(2, 8, g_second * g8);
        symmetric!(3, 3, g_second * g3 * g3);
        symmetric!(3, 5, g_second * g3 * g5);
        symmetric!(3, 6, g_second * g3 * g6 - g_first * exit_exp * p[8]);
        symmetric!(3, 8, g_second * g3 * g8 + g_first * exit_exp);
        symmetric!(5, 5, g_second * g5 * g5);
        symmetric!(5, 6, g_second * g5 * g6 + g_first * exit_exp);
        symmetric!(5, 8, g_second * g5 * g8);
        symmetric!(6, 6, g_second * g6 * g6 + g_first * exit_exp * inner);
        symmetric!(6, 8, g_second * g6 * g8 - g_first * exit_exp * p[3]);
        symmetric!(8, 8, g_second * g8 * g8);
    }

    (value, gradient, hessian)
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

fn assert_close(got: Channels, want: Channels) {
    let close = |a: f64, b: f64| {
        let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
        assert!((a - b).abs() <= tolerance, "{a:+.16e} vs {b:+.16e}");
    };
    close(got.0, want.0);
    for i in 0..K {
        close(got.1[i], want.1[i]);
        for j in 0..K {
            close(got.2[i][j], want.2[i][j]);
        }
    }
}

fn assert_same_channels(got: Channels, want: Channels) {
    let same = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{a:+.16e} vs {b:+.16e}");
        } else {
            let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tolerance, "{a:+.16e} vs {b:+.16e}");
        }
    };
    same(got.0, want.0);
    for i in 0..K {
        same(got.1[i], want.1[i]);
        for j in 0..K {
            same(got.2[i][j], want.2[i][j]);
        }
    }
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

/// Times `ROWS_PER_ARM` value/gradient/Hessian rows per call, feeding the
/// running checksum back into the input so the batch cannot be hoisted.
fn channels_batch(
    p: &[f64; K],
    kernel: &Kernel,
    nudge: f64,
    evaluate: impl Fn(&[f64; K], &Kernel) -> Channels,
) -> f64 {
    let mut accumulated = 0.0;
    let mut perturbed = *p;
    perturbed[7] += nudge;
    for _ in 0..ROWS_PER_ARM {
        perturbed[3] += accumulated * 1e-18;
        let (value, gradient, hessian) = std::hint::black_box(evaluate(&perturbed, kernel));
        accumulated += value + gradient[4] + hessian[4][4] + hessian[4][7];
    }
    accumulated
}

/// Adapts one matrix-valued row evaluation into the `FnMut(f64) -> f64` arm the
/// paired harness times: the nudge keeps consecutive iterations from being
/// folded into one another, and the returned checksum is what the harness
/// accumulates and asserts finite.
fn matrix_arm<'a>(
    p: &'a [f64; K],
    kernel: &'a Kernel,
    mut evaluate: impl FnMut(&[f64; K], &Kernel) -> [[f64; K]; K] + 'a,
) -> impl FnMut(f64) -> f64 + 'a {
    move |nudge| {
        let mut accumulated = 0.0;
        let mut perturbed = *p;
        perturbed[7] += nudge;
        for _ in 0..ROWS_PER_ARM {
            perturbed[3] += accumulated * 1e-20;
            let matrix = std::hint::black_box(evaluate(&perturbed, kernel));
            accumulated += matrix.iter().flat_map(|row| row.iter()).copied().sum::<f64>();
        }
        accumulated
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

