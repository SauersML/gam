#![cfg(test)]
//! #2933 F24 — the ARD log-precision derivatives of an embedded sphere atom must
//! differentiate the Laplace block and the stationarity gradient the criterion
//! actually assembles.
//!
//! On `S²` the assembly writes the ambient ARD curvature `α_a` on axis `a` and the
//! ambient gradient `α_a x_a`, then converts the row to its Riemannian block
//! `P H_amb P − (xᵀg)·P + xxᵀ` and its gradient to `P g`, with `P = I − xxᵀ`. So
//! `∂H/∂log α_a = α_a [P e_a e_aᵀ P − x_a² P]` and `∂g/∂log α_a = α_a x_a·P e_a`,
//! not the ambient slot entries `α_a e_a e_aᵀ` and `α_a x_a e_a`. Every oracle below
//! is a fixed-state central difference of an assembled production object:
//!
//! * the majorizer `½ log|H|` the evidence factor reports (dense trace);
//! * the dense exact observed information `A` (the operator map the dense exact-A
//!   logdet channel contracts);
//! * the assembled KKT gradient `(g_t, g_β)` (the IFT right-hand side), on the dense
//!   layout and on a compact TopK layout with two three-axis sphere atoms.
//!
//! The bundle-sourced trace is pinned to the dense trace at full-basis probes.

use super::tests_recovery_split_780::{
    FiniteDifferenceStratumCertificate, certified_central_logdet_difference,
    fixed_state_logdet_sample,
};
use crate::manifold::tests_dense_solver_oracles::DeflatedArrowSolver;
use super::*;
use ndarray::array;

/// Ten well-separated unit vectors on a golden-angle spiral, rotated by `phase`.
fn spiral_rows(n: usize, phase: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, 3), |(row, axis)| {
        let z = 1.0 - (2.0 * row as f64 + 1.0) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = 2.399_963_229_728_653 * row as f64 + phase;
        match axis {
            0 => r * phi.cos(),
            1 => r * phi.sin(),
            _ => z,
        }
    })
}

fn sphere_atom(
    name: &str,
    coords: &Array2<f64>,
    p: usize,
    shift: f64,
    amplitude: f64,
) -> SaeManifoldAtom {
    let evaluator =
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).expect("degree-2 ambient harmonics"));
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("unit rows are valid ambient-harmonic inputs");
    let m = phi.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
        amplitude * (0.7 * basis_col as f64 + 1.3 * out_col as f64 + shift).sin()
    });
    SaeManifoldAtom::new_with_provided_function_gram(
        name,
        SaeAtomBasisKind::Sphere,
        3,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("sphere atom fixture: basis, jet, decoder and Gram shapes agree")
    .with_basis_second_jet(evaluator)
}

pub(crate) fn sphere_logdet_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10_usize;
    let p = 3_usize;
    let coords = spiral_rows(n, 0.0);
    let atom = sphere_atom("sphere", &coords, p, 0.4, 0.3);
    let target = atom.basis_values.dot(atom.decoder_coefficients())
        + Array2::from_shape_fn((n, p), |(row, out_col)| {
            0.05 * ((3 * row + out_col) as f64).cos()
        });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Sphere { dim: 3 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("sphere assignment fixture");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("sphere term fixture");
    // At this seed the Weingarten term `−(gᵀx)·P` outweighs the decoder's curvature and
    // the row block is indefinite; the oracles below run at the converged mode
    // (`converged_anchor`), where the evidence factor is positive definite.
    let rho = SaeManifoldRho::new(0.0, -2.0, vec![array![0.4, -0.3, 1.1]]);
    (term, target, rho)
}

/// Two sphere atoms under hard TopK support one, alternating by row, so the
/// compact layout places each active atom's three axes at row-local slots.
fn two_sphere_topk_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10_usize;
    let p = 3_usize;
    let coords_a = spiral_rows(n, 0.0);
    let coords_b = spiral_rows(n, 1.1);
    let atom_a = sphere_atom("sphere_a", &coords_a, p, 0.4, 1.0);
    let atom_b = sphere_atom("sphere_b", &coords_b, p, -0.9, 1.0);
    let logits = Array2::from_shape_fn((n, 2), |(row, atom)| {
        if (row + atom) % 2 == 0 { 0.8 } else { -0.8 }
    });
    // Each row decodes its selected atom, plus a small unreachable residual.
    let target = Array2::from_shape_fn((n, p), |(row, out_col)| {
        let selected = if row % 2 == 0 { &atom_a } else { &atom_b };
        selected.basis_values.row(row).dot(&selected.decoder_coefficients().column(out_col))
            + 0.05 * ((2 * row + 3 * out_col) as f64 * 0.37).sin()
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_a, coords_b],
        vec![LatentManifold::Sphere { dim: 3 }, LatentManifold::Sphere { dim: 3 }],
        AssignmentMode::top_k_support(1),
    )
    .expect("two-sphere TopK assignment fixture");
    let term =
        SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).expect("two-sphere term fixture");
    let rho = SaeManifoldRho::new(
        0.0,
        -2.0,
        vec![array![-1.6, -2.3, -1.2], array![-2.0, -1.4, -1.8]],
    );
    (term, target, rho)
}

fn topk_layout(term: &SaeManifoldTerm) -> SaeRowLayout {
    SaeRowLayout::from_topk_gates(
        &term
            .assignments_all_parallel(term.n_obs())
            .expect("TopK gates of the fixture"),
        1,
        vec![3, 3],
        term.assignment.coord_offsets(),
    )
    .expect("hard TopK support-one layout")
}

pub(crate) fn fixed_state_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldTerm, ArrowFactorCache) {
    let mut state = term.clone();
    let (_value, _loss, cache) = state
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("fixed-state sphere cache");
    (state, cache)
}

/// The fixture's seed is off the inner optimum, where both the majorizer row block
/// and the exact information are indefinite. Converge once at `rho` and hold that
/// mode fixed for every stencil point, so the oracles are fixed-state differences
/// at a positive-definite mode.
pub(crate) fn converged_anchor(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> SaeManifoldTerm {
    let mut anchor = term.clone();
    anchor
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the sphere fixture converges to a positive-definite inner mode");
    anchor
}

fn inverse_2x2(m: &Array2<f64>) -> Array2<f64> {
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    array![[m[[1, 1]], -m[[0, 1]]], [-m[[1, 0]], m[[0, 0]]]] / det
}

/// Adjugate inverse, independent of the production factorizations.
fn inverse_3x3(m: &Array2<f64>) -> Array2<f64> {
    let cofactor = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[[r0, c0]] * m[[r1, c1]] - m[[r0, c1]] * m[[r1, c0]]
    };
    let det: f64 = (0..3).map(|j| m[[0, j]] * cofactor(0, j)).sum();
    Array2::from_shape_fn((3, 3), |(i, j)| cofactor(j, i) / det)
}

/// A sphere-row trace must equal its intrinsic 2-D tangent-chart value. The row
/// block is `U H_T Uᵀ + xxᵀ` (tangent chart Hessian plus the unit normal pin).
/// For every ARD axis on the factor, `ard_sphere_log_precision_derivative` must
/// satisfy three conditions:
/// - it is tangent, `D x = 0`;
/// - in the chart it equals the log-precision derivative of the energy's Riemannian
///   Hessian, `u_iᵀ diag(α) u_j − (Σ α x²)·δ_ij`, which is `h·u_i[a]u_j[a] − g_a x_a·δ_ij`;
/// - so `tr(H_R⁻¹ D) = tr(H_T⁻¹ UᵀDU)`.
///
/// The control is the unprojected ambient slot `h·e_a e_aᵀ`. It leaks `h·x_a²`
/// through the pin and misses the connection term.
#[test]
fn sphere_ard_log_precision_derivative_is_the_tangent_chart_derivative_2933_f24() {
    // A four-axis block: one line axis, then an embedded S² at axes 1..4, placed at
    // row-local slot 2 of a 7-slot row.
    let raw = [0.3_f64, -0.5, 0.81];
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let x = raw.map(|v| v / norm);
    let point = [0.7, x[0], x[1], x[2]];
    let factors = [(1_usize, 3_usize)];
    let (block_start, q, start) = (2_usize, 7_usize, 3_usize);
    // Orthonormal tangent basis: u1 ∝ e_0 − x_0·x, u2 = x × u1.
    let u1_raw = [1.0 - x[0] * x[0], -x[0] * x[1], -x[0] * x[2]];
    let u1_norm = u1_raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let u1 = u1_raw.map(|v| v / u1_norm);
    let u2 = [
        x[1] * u1[2] - x[2] * u1[1],
        x[2] * u1[0] - x[0] * u1[2],
        x[0] * u1[1] - x[1] * u1[0],
    ];
    let basis = Array2::from_shape_fn((3, 2), |(i, j)| if j == 0 { u1[i] } else { u2[i] });
    let chart_hessian = array![[1.7, 0.3], [0.3, 0.9]];
    let row_block = basis.dot(&chart_hessian).dot(&basis.t())
        + Array2::from_shape_fn((3, 3), |(i, j)| x[i] * x[j]);
    let row_inverse = inverse_3x3(&row_block);
    let chart_inverse = inverse_2x2(&chart_hessian);
    let alpha = [2.5_f64, 0.7, 1.3];

    assert!(
        SaeManifoldTerm::ard_sphere_log_precision_derivative(&factors, &point, 0, block_start, q, 1.0, 1.0)
            .is_none(),
        "the line axis keeps its slot derivative"
    );
    for local in 0..3 {
        let curvature = alpha[local];
        let gradient = alpha[local] * x[local];
        let derivative = SaeManifoldTerm::ard_sphere_log_precision_derivative(
            &factors,
            &point,
            local + 1,
            block_start,
            q,
            curvature,
            gradient,
        )
        .expect("an axis on the sphere factor has a block derivative");
        for i in 0..q {
            for j in 0..q {
                let inside = (start..start + 3).contains(&i) && (start..start + 3).contains(&j);
                assert!(
                    inside || derivative[[i, j]] == 0.0,
                    "axis {local}: entry [{i},{j}] outside the factor's slots is nonzero"
                );
            }
        }
        let block = derivative
            .slice(ndarray::s![start..start + 3, start..start + 3])
            .to_owned();
        let normal_leak = block.dot(&Array1::from(x.to_vec())).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(normal_leak < 1.0e-14, "axis {local}: D x = {normal_leak:.3e}, not tangent");
        let expected_chart = Array2::from_shape_fn((2, 2), |(i, j)| {
            curvature * basis[[local, i]] * basis[[local, j]]
                - gradient * x[local] * if i == j { 1.0 } else { 0.0 }
        });
        let chart = basis.t().dot(&block).dot(&basis);
        let chart_gap = (&chart - &expected_chart).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            chart_gap < 1.0e-14,
            "axis {local}: UᵀDU differs from the tangent-chart derivative by {chart_gap:.3e}"
        );
        let row_trace = (&row_inverse * &block).sum();
        let chart_trace = (&chart_inverse * &expected_chart).sum();
        assert!(
            (row_trace - chart_trace).abs() < 1.0e-12,
            "axis {local}: sphere-row trace {row_trace:.15e} vs 2-D chart trace {chart_trace:.15e}"
        );

        // Control: the unprojected ambient slot fails both properties materially.
        let mut ambient = Array2::<f64>::zeros((3, 3));
        ambient[[local, local]] = curvature;
        let ambient_trace = (&row_inverse * &ambient).sum();
        let leak = ambient_trace - chart_trace;
        assert!(
            leak.abs() > 1.0e-3,
            "axis {local}: the ambient-slot control must miss the chart trace; gap {leak:.3e}"
        );
    }
}

#[test]
fn sphere_ard_logdet_trace_matches_fixed_state_fd_2933_f24() {
    let (term, target, rho) = sphere_logdet_fixture();
    let anchor = converged_anchor(&term, &target, &rho);
    let (state, cache) = fixed_state_cache(&anchor, &target, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = state
        .ard_log_precision_hessian_trace(&rho, &cache, &solver, EvidenceOperator::Majorizer)
        .expect("ARD log-precision trace on the sphere fixture");
    let h = 1.0e-5;
    let fd_stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    let mut report = Vec::new();
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        let mut rho_plus = rho.clone();
        let mut rho_minus = rho.clone();
        rho_plus.log_ard[0][axis] += h;
        rho_minus.log_ard[0][axis] -= h;
        let fd_half = 0.5
            * certified_central_logdet_difference(
                &format!("sphere ARD trace axis={axis}"),
                &fd_stratum,
                fixed_state_logdet_sample(anchor.clone(), &target, &rho_plus),
                fixed_state_logdet_sample(anchor.clone(), &target, &rho_minus),
                h,
            );
        let a = analytic[0][axis];
        let gap = (fd_half - a).abs() / (1.0 + fd_half.abs().max(a.abs()));
        worst = worst.max(gap);
        report.push(format!("axis {axis}: fd={fd_half:.10e} analytic={a:.10e} relgap={gap:.3e}"));
    }
    assert!(
        worst <= 1.0e-5,
        "sphere ARD log-precision trace must differentiate the Riemannian block: {}",
        report.join("; ")
    );
}

/// The bundle-sourced trace at full-basis probes `√k·e_j` (exact border average)
/// must equal the dense solver trace on the sphere fixture, so the sphere block
/// contraction is one computation on both routes.
#[test]
fn sphere_ard_logdet_trace_from_probes_matches_dense_2933_f24() {
    let (term, target, rho) = sphere_logdet_fixture();
    let anchor = converged_anchor(&term, &target, &rho);
    let (state, cache) = fixed_state_cache(&anchor, &target, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = state
        .ard_log_precision_hessian_trace(&rho, &cache, &solver, EvidenceOperator::Majorizer)
        .expect("dense sphere ARD trace");
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("exact dense Schur inverse apply")
        })
        .collect();
    let bundle = state
        .ard_log_precision_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("bundle sphere ARD trace");
    for axis in 0..3 {
        let (d, b) = (dense[0][axis], bundle[0][axis]);
        assert!(
            (d - b).abs() <= 1.0e-9 * d.abs().max(1.0),
            "axis {axis}: bundle trace {b:.12e} vs dense trace {d:.12e}"
        );
    }
}

/// `exact_stationarity_penalty_derivatives_by_flat`, the operator map the dense
/// exact-A route contracts for `½tr(A⁺ ∂A/∂ρ)`, must be a central difference of the
/// dense exact `A` on the t-block of the sphere fixture.
#[test]
fn sphere_ard_exact_a_operator_derivative_matches_fd_2933_f24() {
    let (term, target, rho) = sphere_logdet_fixture();
    let anchor = converged_anchor(&term, &target, &rho);
    let (state, cache) = fixed_state_cache(&anchor, &target, &rho);
    let derivatives = state
        .exact_stationarity_penalty_derivatives_by_flat(&rho, &cache)
        .expect("exact-A derivative map");
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    let total_t = cache.delta_t_len();
    let h = 1.0e-5;
    let dense_a = |r: &SaeManifoldRho| -> Array2<f64> {
        let (endpoint, endpoint_cache) = fixed_state_cache(&anchor, &target, r);
        stratum.assert_same_stratum(
            "sphere exact-A operator endpoint",
            &FiniteDifferenceStratumCertificate::from_arrow_cache(&endpoint_cache),
        );
        endpoint
            .materialize_exact_hessian_dense(r, target.view(), &endpoint_cache)
            .expect("dense exact A at the endpoint")
    };
    for axis in 0..3 {
        let flat = rho.ard_flat_index(0, axis);
        let analytic = derivatives
            .get(&flat)
            .expect("the sphere ARD axis owns a curvature operator");
        let mut rho_plus = rho.clone();
        let mut rho_minus = rho.clone();
        rho_plus.log_ard[0][axis] += h;
        rho_minus.log_ard[0][axis] -= h;
        let a_plus = dense_a(&rho_plus);
        let a_minus = dense_a(&rho_minus);
        let mut worst = 0.0_f64;
        let mut label = String::new();
        for i in 0..total_t {
            for j in 0..total_t {
                let fd = (a_plus[[i, j]] - a_minus[[i, j]]) / (2.0 * h);
                let normalized = (fd - analytic[[i, j]]).abs() / (1.0e-6 + 1.0e-4 * fd.abs());
                if normalized > worst {
                    worst = normalized;
                    label = format!("[{i},{j}] analytic={:.9e} fd={fd:.9e}", analytic[[i, j]]);
                }
            }
        }
        assert!(
            worst <= 1.0,
            "axis {axis}: dA/dlog α must be the Riemannian block derivative; worst normalized \
             error {worst:.3} at {label}"
        );
    }
}

/// The assembled KKT gradient `(g_t, g_β)` in `forced_layout`, concatenated in
/// cache order.
fn assembled_gradient(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    forced_layout: Option<SaeRowLayout>,
) -> (SaeManifoldTerm, ArrowSchurSystem, Array1<f64>) {
    let mut state = term.clone();
    let sys = state
        .assemble_arrow_schur_inner(
            target.view(),
            rho,
            None,
            1.0,
            SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM,
            Some(forced_layout),
        )
        .expect("arrow-Schur assembly at a fixed state");
    let total_t: usize = sys.rows.iter().map(|row| row.gt.len()).sum();
    let mut gradient = Array1::<f64>::zeros(total_t + sys.gb.len());
    let mut offset = 0;
    for row in &sys.rows {
        for (slot, &value) in row.gt.iter().enumerate() {
            gradient[offset + slot] = value;
        }
        offset += row.gt.len();
    }
    for (slot, &value) in sys.gb.iter().enumerate() {
        gradient[total_t + slot] = value;
    }
    (state, sys, gradient)
}

/// `outer_rho_gradient_ift_rhs` for an ARD coordinate must be the log-precision
/// derivative of the assembled gradient: the tangent projection on a sphere, at
/// every axis's own compact slot under TopK.
#[test]
fn sphere_ard_ift_rhs_matches_fd_of_the_assembled_gradient_2933_f24() {
    let arms: [(&str, (SaeManifoldTerm, Array2<f64>, SaeManifoldRho), bool); 2] = [
        ("dense softmax", sphere_logdet_fixture(), false),
        ("compact TopK", two_sphere_topk_fixture(), true),
    ];
    let h = 1.0e-5;
    for (label, (term, target, rho), compact) in arms {
        let layout = |t: &SaeManifoldTerm| compact.then(|| topk_layout(t));
        let (state, mut sys, _) = assembled_gradient(&term, &target, &rho, layout(&term));
        assert_eq!(
            state.last_row_layout.is_some(),
            compact,
            "{label}: the assembly must take the declared layout"
        );
        // The right-hand side reads only the cache's row layout and border width, so
        // any admitted factor of this assembly serves. The seed state is off the inner
        // optimum, so the discarded Newton step is damped and the evidence factor
        // unit-deflates what the seed leaves indefinite.
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut sys);
        let options = state.evidence_factor_options();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 1.0e3, 1.0e3, &options)
                .expect("admitted factor of the fixed-state assembly");
        for atom in 0..rho.log_ard.len() {
            for axis in 0..3 {
                let flat = rho.ard_flat_index(atom, axis);
                let rhs = state
                    .outer_rho_gradient_ift_rhs(&rho, flat, &cache)
                    .expect("ARD IFT right-hand side");
                let mut rho_plus = rho.clone();
                let mut rho_minus = rho.clone();
                rho_plus.log_ard[atom][axis] += h;
                rho_minus.log_ard[atom][axis] -= h;
                let (_, _, g_plus) = assembled_gradient(&term, &target, &rho_plus, layout(&term));
                let (_, _, g_minus) =
                    assembled_gradient(&term, &target, &rho_minus, layout(&term));
                let total_t = rhs.t.len();
                assert_eq!(g_plus.len(), total_t + rhs.beta.len());
                let mut worst = 0.0_f64;
                let mut worst_label = String::new();
                for slot in 0..g_plus.len() {
                    let fd = (g_plus[slot] - g_minus[slot]) / (2.0 * h);
                    let analytic =
                        if slot < total_t { rhs.t[slot] } else { rhs.beta[slot - total_t] };
                    let normalized = (fd - analytic).abs() / (1.0e-7 + 1.0e-5 * fd.abs());
                    if normalized > worst {
                        worst = normalized;
                        worst_label = format!("slot {slot}: analytic={analytic:.9e} fd={fd:.9e}");
                    }
                }
                assert!(
                    worst <= 1.0,
                    "{label} atom {atom} axis {axis}: dg/dlog α must be the assembled gradient's \
                     derivative; worst normalized error {worst:.3} at {worst_label}"
                );
            }
        }
    }
}
