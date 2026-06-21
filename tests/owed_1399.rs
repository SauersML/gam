//! Owed-work regression gate for issue #1399.
//!
//! #1399: the device-resident SAE inner solve was reported to diverge from the
//! production arrow-Schur Newton step (rel ≈ 3.55e-1), alongside two stale
//! dimension assertions and a multiplex-sweep timeout.
//!
//! Root cause of the only real numeric failure: the *parity assertion's sign
//! convention*, not the solver math. The resident inner loop converges to the
//! quadratic minimiser `z* = H⁻¹ g₀` and STORES it (`t += Δ` each iterate), so
//! `resident.t == +z*`. The production entry `solve_arrow_newton_step_core`
//! solves the Newton system `H Δ = −g₀` from `z = 0`, giving `Δ = −H⁻¹ g₀ = −z*`.
//! The correct invariant is therefore `Δ + resident == 0`. The pre-fix test
//! asserted `Δ − resident == 0` (same-sign compare), which evaluates to
//! `|−z* − z*| = |2 z*|` — exactly the reported rel ≈ 3.55e-1 on the fixture.
//! Commit `0a83b8e4e` flipped the comparison to `Δ + resident`.
//!
//! This gate pins that invariant at the integration level through the public
//! API only, on a hand-built strongly-PD bordered quadratic (no CUDA required):
//! it would re-fire (rel ≈ |2 z*|) if the sign convention regressed, and it
//! independently checks that the resident iterate is genuinely the stationary
//! point `H z* = g₀`, so a future change that merely made both halves agree on a
//! wrong value could not pass it.

use gam::gpu::kernels::sae_resident::{
    DeviceResidentArrowShape, DeviceResidentArrowSlabs, DeviceResidentArrowWorkspace,
    DeviceResidentInnerOptions,
};
use gam::solver::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_core};

/// Build a small, strongly diagonally-dominant resident frame whose dense
/// reference factorisation is well-conditioned (the bordered quadratic minimiser
/// `z* = H⁻¹ g₀` is reached in one accepted Newton step). The slab layout mirrors
/// `DeviceResidentArrowWorkspace`'s row-major contract.
fn pd_fixture() -> DeviceResidentArrowWorkspace {
    let shape = DeviceResidentArrowShape {
        n: 3,
        p: 4,
        basis_cols: 2,
        d: 2,
    };
    let target_x = vec![0.0_f64; shape.target_len()];
    let basis_values = vec![0.5_f64; shape.basis_len()];
    let gate_activations = vec![1.0_f64; shape.basis_len()];

    let mut row_hessian_slabs = vec![0.0_f64; shape.row_hessian_len()];
    let mut row_cross_slabs = vec![0.0_f64; shape.row_cross_len()];
    let mut row_gradient_slabs = vec![0.0_f64; shape.row_gradient_len()];
    for i in 0..shape.n {
        let h = i * shape.d * shape.d;
        // Symmetric PD 2×2 tip block, diagonally dominant.
        row_hessian_slabs[h] = 5.0 + 0.05 * (i as f64);
        row_hessian_slabs[h + 1] = 0.03;
        row_hessian_slabs[h + 2] = 0.03;
        row_hessian_slabs[h + 3] = 4.0 + 0.05 * (i as f64);
        // Small cross block keeps the Schur complement well away from singular.
        let b = i * shape.d * shape.p;
        for j in 0..shape.p {
            row_cross_slabs[b + j] = 0.01 * (1 + i + j) as f64 * 0.1;
            row_cross_slabs[b + shape.p + j] = -0.01 * (1 + j) as f64 * 0.1;
        }
        let g = i * shape.d;
        row_gradient_slabs[g] = 0.7 - 0.2 * (i as f64);
        row_gradient_slabs[g + 1] = -0.4 + 0.3 * (i as f64);
    }
    let mut border_hessian = vec![0.0_f64; shape.border_hessian_len()];
    for r in 0..shape.p {
        border_hessian[r * shape.p + r] = 6.0 + 0.1 * (r as f64);
    }
    let border_gradient: Vec<f64> = (0..shape.p).map(|r| 0.5 - 0.15 * (r as f64)).collect();

    DeviceResidentArrowWorkspace::new(
        shape,
        target_x,
        basis_values,
        gate_activations,
        DeviceResidentArrowSlabs {
            row_hessian_slabs,
            row_cross_slabs,
            row_gradient_slabs,
            border_hessian,
            border_gradient,
        },
    )
    .expect("PD resident fixture must validate")
}

/// #1399: the production arrow-core Newton step must equal `−(resident converged
/// iterate)` on the same bordered quadratic. This is the sign-convention
/// invariant whose violation produced the reported rel ≈ 3.55e-1.
#[test]
fn resident_iterate_is_negated_production_arrow_core_step() {
    let ws = pd_fixture();
    let opts = DeviceResidentInnerOptions::default();

    // Resident CPU reference loop: converges to z* = H⁻¹ g₀ and stores t = +z*.
    let resident = ws.cpu_reference_fit(&opts).expect("resident cpu fit");
    assert!(
        resident.converged,
        "resident reference must converge on the PD quadratic"
    );

    // Production arrow path: one Newton step from z = 0 on the SAME system.
    // `_core` is the device-aware entry; on a CPU box it runs the dense CPU
    // solve, the exact path a GPU host falls back to on device decline.
    let sys = ws.to_arrow_system();
    let (delta_t, delta_beta, _diag) = solve_arrow_newton_step_core(
        &sys,
        opts.initial_ridge_t,
        opts.initial_ridge_beta,
        &ArrowSolveOptions::direct(),
    )
    .expect("production arrow-core solve");

    let t_scale = resident.t.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
    let b_scale = resident.beta.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));

    // Δt + t* must be 0 (a t-block gap implicates the per-row factor / row
    // gradient); Δβ + β* must be 0 (a β-block gap implicates the border Schur).
    let mut max_rel_t = 0.0_f64;
    for (prod, res) in delta_t.iter().zip(resident.t.iter()) {
        max_rel_t = max_rel_t.max((prod + res).abs() / t_scale);
    }
    let mut max_rel_b = 0.0_f64;
    for (prod, res) in delta_beta.iter().zip(resident.beta.iter()) {
        max_rel_b = max_rel_b.max((prod + res).abs() / b_scale);
    }
    let max_rel = max_rel_t.max(max_rel_b);

    // A same-sign regression (the original #1399 bug) would land here at
    // |2 z*| / scale ≈ 0.36, well above the factorisation tolerance.
    assert!(
        max_rel < 1e-9,
        "production arrow-core step must be −(resident converged fit) on the same \
         quadratic: rel_t={max_rel_t:e}, rel_beta={max_rel_b:e}. A ~0.36 failure is \
         the #1399 same-sign parity regression (assert Δ−z* instead of Δ+z*)."
    );
}

/// #1399 independence guard: the resident converged iterate must be the genuine
/// stationary point `H z* = g₀`, assembled independently here from the arrow
/// system's dense blocks. Without this, a regression that made BOTH the resident
/// loop and the production step converge to the same WRONG value could slip past
/// the negation invariant above.
#[test]
fn resident_iterate_solves_dense_stationary_system() {
    use ndarray::{Array1, Array2};

    let ws = pd_fixture();
    let opts = DeviceResidentInnerOptions::default();
    let outcome = ws.cpu_reference_fit(&opts).expect("resident cpu fit");
    assert!(outcome.converged, "resident reference must converge");

    let sys = ws.to_arrow_system();
    let shape = ws.shape();
    let total = shape.n * shape.d + shape.p;

    // Assemble dense H and g₀ from the arrow blocks (independent of the solver).
    let mut h = Array2::<f64>::zeros((total, total));
    let mut g0 = Array1::<f64>::zeros(total);
    for i in 0..shape.n {
        let base = i * shape.d;
        for r in 0..shape.d {
            for c in 0..shape.d {
                h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
            }
            for c in 0..shape.p {
                let v = sys.rows[i].htbeta[[r, c]];
                h[[base + r, shape.n * shape.d + c]] = v;
                h[[shape.n * shape.d + c, base + r]] = v;
            }
            g0[base + r] = sys.rows[i].gt[r];
        }
    }
    for r in 0..shape.p {
        for c in 0..shape.p {
            h[[shape.n * shape.d + r, shape.n * shape.d + c]] = sys.hbb[[r, c]];
        }
        g0[shape.n * shape.d + r] = sys.gb[r];
    }

    let mut z = Array1::<f64>::zeros(total);
    for r in 0..shape.n * shape.d {
        z[r] = outcome.t[r];
    }
    for c in 0..shape.p {
        z[shape.n * shape.d + c] = outcome.beta[c];
    }
    let hz = h.dot(&z);
    let mut max_resid = 0.0_f64;
    for r in 0..total {
        max_resid = max_resid.max((hz[r] - g0[r]).abs());
    }
    assert!(
        max_resid < 1e-9,
        "resident converged iterate must solve H z* = g₀; residual {max_resid:e}"
    );
}
