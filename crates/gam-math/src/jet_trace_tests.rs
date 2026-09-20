//! gam#2998: [`DynamicTraceJet`] lane `k` carries `v_kᵀ H v_k` and
//! `D³f[v_k, v_k, ·]`. Both are checked against a one-seed batch seeded along
//! every axis, which carries the full contracted third tensor, on one program
//! that exercises every overridden primitive: seed and mixed linear
//! combinations, products, compositions, seed and non-seed weighted compose
//! sums, the affine composed sum and a filtered implicit solve.

use crate::jet_scalar::{
    DynamicJetBatchWorkspace, DynamicOneSeedBatch, RuntimeJetScalar,
    filtered_implicit_solve_runtime_scalar,
};
use crate::jet_trace::{DynamicTraceJet, TraceJetWorkspace};

fn sin_stack(y: f64) -> [f64; 5] {
    let (s, c) = y.sin_cos();
    [s, c, -s, -c, s]
}

fn exp_stack(y: f64) -> [f64; 5] {
    let e = y.exp();
    [e, e, e, e, e]
}

/// A row-program-shaped expression over primaries `x`.
fn program<'a, S: RuntimeJetScalar<'a>>(x: &[S], dimension: usize, ws: &'a S::Workspace) -> S {
    let weights: Vec<f64> = (0..x.len()).map(|i| 0.3 - 0.17 * i as f64).collect();
    let a = S::linear_combination(x, &weights, dimension, ws);
    let b = a.exp();
    let c = x[1].mul(&b).sub(&x[2].scale(0.3));
    let shifted = x[3].with_value(x[3].value() + 0.1);
    let d = x[0].multiply_add(&c, &shifted);
    let mixed = S::linear_combination(&[c.clone(), x[4].clone(), d.clone()], &[0.5, 0.8, -1.2], dimension, ws);
    let seed_stacks: Vec<[f64; 5]> = (0..x.len())
        .map(|i| sin_stack(mixed.value() * (1.0 + 0.2 * i as f64)))
        .collect();
    let f = S::weighted_compose_sum(x, &mixed, &seed_stacks, &d);
    let g = S::weighted_compose_sum(
        &[c.clone(), d.clone(), x[5].clone()],
        &a,
        &[sin_stack(a.value()), exp_stack(0.4 * a.value()), sin_stack(-a.value())],
        &f,
    );
    let scales = [0.7, -0.4];
    let h = S::affine_composed_sum(
        &[g.clone(), mixed.clone()],
        &scales,
        &[sin_stack(scales[0] * g.value()), exp_stack(scales[1] * mixed.value())],
        dimension,
        ws,
    );
    // Root of `a + a³/3 = h`.
    let rhs = h.value();
    let mut a0 = rhs;
    for _ in 0..60 {
        a0 -= (a0 + a0 * a0 * a0 / 3.0 - rhs) / (1.0 + a0 * a0);
    }
    let root = filtered_implicit_solve_runtime_scalar::<S>(
        a0,
        1.0 / (1.0 + a0 * a0),
        3,
        dimension,
        ws,
        |state| {
            let v = state.value();
            state
                .compose_unary([v + v * v * v / 3.0, 1.0 + v * v, 2.0 * v, 2.0, 0.0])
                .sub(&h)
        },
    );
    root.neg().mul(&root).add(&b.ln()).add(&b.recip()).add(&h.scale(0.25))
}

const POINT: [f64; 6] = [0.31, -0.52, 0.77, 0.12, -0.28, 0.45];

fn directions() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.3, -0.8, 0.2, 0.5, -0.1, 0.9],
        vec![-0.6, 0.1, 0.7, -0.4, 0.35, 0.05],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.2, 0.2, -0.9, 0.1, 0.6, -0.3],
    ]
}

fn close(actual: f64, expected: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1e-12 * scale,
        "{what}: trace jet {actual:e} vs one-seed batch {expected:e}"
    );
}

#[test]
fn trace_jet_lanes_match_contracted_third_tensor_2998() {
    let r = POINT.len();
    let dirs = directions();

    let batch_ws = DynamicJetBatchWorkspace::new(r);
    let batch_x: Vec<DynamicOneSeedBatch> = (0..r)
        .map(|axis| {
            DynamicOneSeedBatch::seed_directions(POINT[axis], axis, r, &batch_ws, |lane| {
                if lane == axis { 1.0 } else { 0.0 }
            })
        })
        .collect();
    let batch = program(&batch_x, r, &batch_ws);

    let trace_ws = TraceJetWorkspace::new(dirs.len());
    let trace_x: Vec<DynamicTraceJet> = (0..r)
        .map(|axis| {
            DynamicTraceJet::seed_directions(POINT[axis], axis, r, &trace_ws, |lane| {
                dirs[lane][axis]
            })
        })
        .collect();
    let trace = program(&trace_x, r, &trace_ws);

    close(trace.value(), batch.value(), "value");
    let quad = |m: &[f64], v: &[f64]| -> f64 {
        (0..r)
            .map(|a| (0..r).map(|b| v[a] * m[a * r + b] * v[b]).sum::<f64>())
            .sum()
    };
    let hessian = batch.base.h();
    let mut largest = 0.0f64;
    for (lane, v) in dirs.iter().enumerate() {
        close(
            trace.second_directional(lane),
            quad(hessian, v),
            &format!("lane {lane} vᵀHv"),
        );
        let gradient = trace.second_directional_gradient(lane);
        assert_eq!(gradient.len(), r);
        for c in 0..r {
            let expected = quad(batch.contracted_third(c), v);
            largest = largest.max(expected.abs());
            close(gradient[c], expected, &format!("lane {lane} D³[v,v,{c}]"));
        }
    }
    // The program has a live third derivative, so the comparison is not 0 = 0.
    assert!(largest > 1e-2, "third tensor is degenerate: {largest:e}");
}

/// Resetting the workspace to a new lane count reshapes every jet it makes.
#[test]
fn trace_jet_workspace_reset_changes_lane_count_2998() {
    let mut ws = TraceJetWorkspace::new(2);
    {
        let x = DynamicTraceJet::seed_directions(0.5, 0, 1, &ws, |lane| lane as f64);
        assert_eq!(x.lanes(), 2);
    }
    ws.reset(5);
    let x = DynamicTraceJet::seed_directions(0.5, 0, 1, &ws, |lane| lane as f64);
    let y = x.exp().mul(&x);
    assert_eq!(y.lanes(), 5);
    // f = x eˣ: f'' = (x + 2) eˣ, f''' = (x + 3) eˣ, lane k direction k.
    for lane in 0..5 {
        let k2 = (lane * lane) as f64;
        close(y.second_directional(lane), k2 * 2.5 * 0.5f64.exp(), "f''");
        close(y.second_directional_gradient(lane)[0], k2 * 3.5 * 0.5f64.exp(), "f'''");
    }
}
