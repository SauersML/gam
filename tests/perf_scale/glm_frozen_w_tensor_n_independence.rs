//! #1033 mechanism (c): the GLM design-moving ψ-sweep is n-INDEPENDENT.
//!
//! The architectural invariant the issue enforces is: n-dependent work happens
//! ONCE per fit (the sufficient-statistic build); the κ/ψ outer loop manipulates
//! only k×k objects whose per-trial cost is O(D²k²) — independent of n. For the
//! non-Gaussian GLM lane the carrier of that invariant is
//! [`FrozenWeightGramTensor`] (`solver/glm_sufficient_lane.rs`): at the warm β it
//! freezes the working weight `W` and builds the weighted-design Chebyshev-in-ψ
//! tensor once, after which every per-trial accessor — the value Gram `XᵀWX(ψ)`,
//! the RHS `XᵀWz(ψ)`, the gradient pair `(∂G/∂ψ, ∂b/∂ψ)`, and the Fisher Hessian
//! block `(∂²G/∂ψ², ∂²b/∂ψ²)` — is served n-free in k-space.
//!
//! This is the algebraic companion to the wall-clock `perf_kappa_loop_n_scaling`
//! measurement: rather than time a fit (which is noisy and gated behind the
//! iso-κ convergence path), it pins the invariant *exactly*. Replicate the SAME
//! `b` distinct base rows `m` times to form `n = m·b`. The weighted-design Gram
//! and all its ψ-derivatives are additive over rows, so the n-row tensor's
//! accessors equal EXACTLY `m ×` the base-row tensor's accessors at every ψ —
//! hence, after dividing by the replication factor, they are BIT-IDENTICAL as n
//! scales at fixed k. Equivalently: the k×k object the outer trial loop touches
//! does not change shape or content-per-unit-data as n grows; the only thing n
//! buys is a constant scale absorbed by the one-time build. That is exactly the
//! "cost/grad/Hessian identical as n scales at fixed k" acceptance for this lane.

use gam::solver::glm_sufficient_lane::FrozenWeightGramTensor;
use ndarray::{Array1, Array2};

/// Matérn-shaped synthetic design `g(r·e^ψ)`, `g(s) = (1+s)e^{−s}`, plus a
/// ψ-free cubic column — the structural mix of the radial spatial designs the
/// frozen-W lane actually serves. `r` is a deterministic function of the
/// (base-row, column) index so replication is exact: base row `i` reused at
/// global row `i + t·b` gets the IDENTICAL `r`, hence the identical design row.
fn base_design(psi: f64, base_rows: usize, k: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((base_rows, k));
    for i in 0..base_rows {
        for j in 0..k {
            let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (base_rows as f64 * k as f64) * 3.0;
            if j == k - 1 {
                x[[i, j]] = r * r * r;
            } else {
                let s = r * psi.exp();
                x[[i, j]] = (1.0 + s) * (-s).exp();
            }
        }
    }
    x
}

/// A non-trivial positive Fisher weight per base row (e.g. Bernoulli μ(1−μ)).
fn base_weights(base_rows: usize) -> Array1<f64> {
    Array1::from_shape_fn(base_rows, |i| {
        let p = 0.1 + 0.8 * ((i as f64 + 0.5) / base_rows as f64);
        p * (1.0 - p)
    })
}

fn base_z(base_rows: usize) -> Array1<f64> {
    Array1::from_shape_fn(base_rows, |i| ((i as f64 * 0.37).sin()) + 0.5)
}

/// Build a frozen-W tensor whose data is the `base` set replicated `reps` times
/// (so `n = reps · base_rows`). The design realizer tiles the base design rows;
/// the weights and working response tile in lockstep. Same `k`, same ψ window —
/// only `n` changes.
fn build_replicated(
    base_rows: usize,
    k: usize,
    reps: usize,
    psi_lo: f64,
    psi_hi: f64,
) -> FrozenWeightGramTensor {
    let bw = base_weights(base_rows);
    let bz = base_z(base_rows);
    let n = base_rows * reps;
    let mut w = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    for t in 0..reps {
        for i in 0..base_rows {
            w[t * base_rows + i] = bw[i];
            z[t * base_rows + i] = bz[i];
        }
    }
    FrozenWeightGramTensor::build(
        move |psi| {
            let base = base_design(psi, base_rows, k);
            let mut tiled = Array2::<f64>::zeros((n, k));
            for t in 0..reps {
                for i in 0..base_rows {
                    for j in 0..k {
                        tiled[[t * base_rows + i, j]] = base[[i, j]];
                    }
                }
            }
            Ok(tiled)
        },
        w.view(),
        z.view(),
        psi_lo,
        psi_hi,
    )
    .expect("frozen-W tensor must certify on the analytic Matérn-shaped design")
}

fn rel_err_mat(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1e-300);
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
        / scale
}

fn rel_err_vec(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let scale = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1e-300);
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
        / scale
}

/// The hard acceptance gate for #1033 on the GLM frozen-W lane: every per-trial
/// k-space accessor (value, gradient, Hessian) computed from `n = reps·b` rows
/// equals EXACTLY `reps ×` the same accessor computed from `b` rows — so the
/// per-unit-data k×k object is invariant in n. We sweep `reps ∈ {1,4,16,64}`
/// (a 64× span in n at FIXED k) and assert bit-tight agreement after dividing
/// out the replication factor, across value / gradient / Fisher-Hessian.
#[test]
fn glm_frozen_w_outer_objects_are_n_independent() {
    let (base_rows, k) = (96usize, 5usize);
    let (psi_lo, psi_hi) = (-0.5, 0.5);
    // Interior off-node ψ values (never the build nodes), inside the gradient
    // sub-window so all three channels are exercised.
    let psis: [f64; 3] = [
        psi_lo + 0.382 * (psi_hi - psi_lo),
        0.0,
        psi_lo + 0.618 * (psi_hi - psi_lo),
    ];

    // Reference at reps=1 (n = base_rows).
    let base = build_replicated(base_rows, k, 1, psi_lo, psi_hi);

    let reps_sweep = [4usize, 16, 64];
    // After the build, the per-trial accessor cost is the SAME k×k assembly for
    // every n — there is no n-shaped object left in the accessor. We assert the
    // CONTENT invariant (the strongest observable): the n-row objects are exact
    // scalar multiples of the base objects. Bit-tight bar (1e-10 rel): these are
    // the same Chebyshev recurrence evaluated on a scaled Gram, so the only
    // difference is floating-point summation order over replicated rows.
    const REL: f64 = 1e-10;
    for &reps in &reps_sweep {
        let big = build_replicated(base_rows, k, reps, psi_lo, psi_hi);
        let m = reps as f64;
        for &psi in &psis {
            assert!(base.contains(psi) && big.contains(psi));
            assert!(base.contains_for_gradient(psi) && big.contains_for_gradient(psi));

            // Value channel: XᵀWX(ψ), XᵀWz(ψ).
            let g_big = big.gram_at(psi);
            let g_ref = base.gram_at(psi).mapv(|v| v * m);
            let e = rel_err_mat(&g_big, &g_ref);
            assert!(
                e <= REL,
                "value Gram XᵀWX(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e} \
                 from {m}× the base-row Gram"
            );
            let r_big = big.rhs_at(psi);
            let r_ref = base.rhs_at(psi).mapv(|v| v * m);
            let e = rel_err_vec(&r_big, &r_ref);
            assert!(e <= REL, "RHS XᵀWz(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e}");

            // Gradient channel: ∂G/∂ψ, ∂b/∂ψ.
            let dg_big = big.dgram_dpsi(psi);
            let dg_ref = base.dgram_dpsi(psi).mapv(|v| v * m);
            let e = rel_err_mat(&dg_big, &dg_ref);
            assert!(
                e <= REL,
                "gradient ∂G/∂ψ(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e}"
            );
            let db_big = big.drhs_dpsi(psi);
            let db_ref = base.drhs_dpsi(psi).mapv(|v| v * m);
            let e = rel_err_vec(&db_big, &db_ref);
            assert!(
                e <= REL,
                "gradient ∂b/∂ψ(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e}"
            );

            // Fisher-Hessian channel: ∂²G/∂ψ², ∂²b/∂ψ².
            let h_big = big.d2gram_dpsi2(psi);
            let h_ref = base.d2gram_dpsi2(psi).mapv(|v| v * m);
            let e = rel_err_mat(&h_big, &h_ref);
            assert!(
                e <= REL,
                "Fisher-Hessian ∂²G/∂ψ²(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e}"
            );
            let hb_big = big.d2rhs_dpsi2(psi);
            let hb_ref = base.d2rhs_dpsi2(psi).mapv(|v| v * m);
            let e = rel_err_vec(&hb_big, &hb_ref);
            assert!(
                e <= REL,
                "Fisher-Hessian ∂²b/∂ψ²(ψ={psi}) not n-independent: reps={reps} rel-dev {e:.3e}"
            );
        }
    }
}

/// The accessor objects are FIXED k×k in shape regardless of n — there is no
/// n-shaped allocation in the per-trial path. This pins the structural half of
/// the invariant (the content half is in the test above): a future refactor
/// that smuggled an n-row object into an accessor would change these shapes.
#[test]
fn glm_frozen_w_accessor_shapes_are_fixed_k_across_n() {
    let (base_rows, k) = (80usize, 4usize);
    let (psi_lo, psi_hi) = (-0.4, 0.4);
    let psi = 0.0;
    for &reps in &[1usize, 8, 64] {
        let t = build_replicated(base_rows, k, reps, psi_lo, psi_hi);
        // Every per-trial object is k×k / k-long — never n.
        assert_eq!(t.gram_at(psi).dim(), (k, k));
        assert_eq!(t.dgram_dpsi(psi).dim(), (k, k));
        assert_eq!(t.d2gram_dpsi2(psi).dim(), (k, k));
        assert_eq!(t.rhs_at(psi).len(), k);
        assert_eq!(t.drhs_dpsi(psi).len(), k);
        assert_eq!(t.d2rhs_dpsi2(psi).len(), k);
        // The drift guard's frozen weight IS n-long (it is the one-time-built
        // sufficient-statistic source, not a per-trial object) — confirm it
        // tracks n so we are genuinely testing distinct n, not a silent no-op.
        assert_eq!(t.frozen_weights().len(), base_rows * reps);
    }
}
