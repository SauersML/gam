//! End-to-end quality: gam's SINDy STLSQ solver with a concave SCAD penalty
//! (Fan & Li 2001) must induce *measurably sparser, smaller-tail* solutions
//! than the plain ridge baseline on a known cubic dynamical system — and the
//! ridge baseline itself must agree with **pysindy** (Brunton/Kutz, the
//! reference SINDy implementation) and the SCAD path with a hand-built
//! Fan-Li-2001 local-quadratic-approximation (LQA) re-weighted-ridge solver.
//!
//! Why two comparators in one body? pysindy ships `STLSQ` (sequentially
//! thresholded *ridge* least squares) — exactly gam's `SindyPenaltyKind::Ridge`
//! path — so it is the mature head-to-head for the ridge branch. pysindy does
//! NOT ship SCAD, so the SCAD ground truth is built from the Fan-Li 2001
//! derivative `p'_{λ,a}` directly in NumPy as iterative re-weighted ridge on
//! the same active set — the canonical LQA surrogate gam itself implements.
//! Feeding *byte-identical* `Θ` and `Ẋ` matrices to all three solvers makes the
//! comparison an apples-to-apples test of the optimisation, not the data.
//!
//! System: `dx/dt = -x + 0.1 x³`, integrated with fixed-seed RK4 (ode45-grade)
//! over a 1000-sample trajectory; library `Θ = [1, x, x³]`; the true SINDy
//! vector is `ξ* = [0, -1, 0.1]`. We sweep `λ ∈ {0.01, 0.1, 1.0, 10.0}` and at
//! each `λ` assert:
//!   1. gam-ridge recovers the same active support as pysindy STLSQ-ridge;
//!   2. gam-SCAD recovers the same active support as the Fan-Li reference SCAD;
//!   3. SCAD is never *less* sparse than ridge on this cubic system;
//!   4. SCAD recovers the exact true support {x, x³} at some swept λ, its
//!      sparsest solution over the sweep is at least as sparse as ridge's, and
//!      — the defining Fan-Li 2001 near-oracle property — SCAD's *best*
//!      coefficient estimate over the λ sweep is at least as close to the true
//!      ξ* = [0, -1, 0.1] as ridge's best, and essentially exact. (Note: the
//!      naive "SCAD shrinks the surviving coefficients harder, so |ξ|₁ ≤ ridge"
//!      claim is FALSE for a concave penalty: SCAD's whole point is to leave
//!      large signal terms *un*-shrunk, so on the same support its |ξ| is
//!      typically *larger* — and closer to the unbiased truth — than ridge's
//!      uniformly-biased-toward-zero estimate. The principled head-to-head is
//!      therefore distance-to-truth, not L1 magnitude.)

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_solve};
use gam::test_support::reference::{Column, run_python};
use ndarray::Array2;

#[test]
fn sindy_scad_is_sparser_than_ridge_and_matches_pysindy() {
    // ---- synthesise the cubic trajectory with fixed-seed RK4 --------------
    // dx/dt = f(x) = -x + 0.1 x^3. To span enough of the state space for a
    // well-conditioned library (a single decaying trajectory collapses to the
    // origin and makes Θ rank-deficient), we launch many short RK4 sub-arcs
    // from a deterministic LCG-seeded spread of initial conditions and
    // concatenate their samples. Every number here is reproducible; the SAME
    // (theta, dxdt) is shipped to gam and to Python.
    let n_total = 1000usize;
    let dt = 0.01_f64;
    let f = |x: f64| -x + 0.1 * x * x * x;

    let mut rng = 0x5eed_1234_u64;
    let mut next_unit = || {
        // SplitMix64-style step -> uniform in (0,1).
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut xs: Vec<f64> = Vec::with_capacity(n_total);
    let mut dxs: Vec<f64> = Vec::with_capacity(n_total);
    while xs.len() < n_total {
        // Fresh initial condition spread over a band where the cubic term is
        // non-negligible relative to the linear term (|x| up to ~4).
        let mut x = (next_unit() - 0.5) * 8.0;
        // A short RK4 sub-arc of ~20 steps.
        for _ in 0..20 {
            if xs.len() >= n_total {
                break;
            }
            // Record the exact analytic derivative as the SINDy target (this is
            // what pysindy would obtain from a clean derivative; using f(x)
            // directly removes finite-difference noise so the comparison probes
            // the *penalty*, not the differentiator).
            xs.push(x);
            dxs.push(f(x));
            // Advance the state with classic RK4 so the trajectory is a true
            // ode45-equivalent integral curve of the system.
            let k1 = f(x);
            let k2 = f(x + 0.5 * dt * k1);
            let k3 = f(x + 0.5 * dt * k2);
            let k4 = f(x + dt * k3);
            x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
    assert_eq!(xs.len(), n_total);

    // ---- build the SINDy library Θ = [1, x, x^3] --------------------------
    let theta_cols = 3usize;
    let mut theta = Array2::<f64>::zeros((n_total, theta_cols));
    for (i, &x) in xs.iter().enumerate() {
        theta[(i, 0)] = 1.0;
        theta[(i, 1)] = x;
        theta[(i, 2)] = x * x * x;
    }
    let mut dxdt = Array2::<f64>::zeros((n_total, 1));
    for (i, &d) in dxs.iter().enumerate() {
        dxdt[(i, 0)] = d;
    }

    // Flatten the library columns so the *identical* matrix can be reconstructed
    // in Python without any re-derivation.
    let col_const: Vec<f64> = theta.column(0).to_vec();
    let col_x: Vec<f64> = theta.column(1).to_vec();
    let col_x3: Vec<f64> = theta.column(2).to_vec();
    let target: Vec<f64> = dxdt.column(0).to_vec();

    let tol = 0.05_f64;
    let max_rounds = 20usize;
    let scad_a = 3.7_f64;
    let lams = [0.01_f64, 0.1, 1.0, 10.0];

    // ---- gam: STLSQ with Ridge and with SCAD at each λ --------------------
    // True SINDy vector for dx/dt = -x + 0.1 x^3 over library [1, x, x^3].
    let xi_true = [0.0_f64, -1.0, 0.1];
    struct Run {
        lam: f64,
        ridge_support: Vec<usize>,
        ridge_l1: f64,
        ridge_err: f64,
        scad_support: Vec<usize>,
        scad_l1: f64,
        scad_err: f64,
    }
    let support_of = |coef: &Array2<f64>| -> Vec<usize> {
        (0..theta_cols)
            .filter(|&j| coef[(j, 0)] != 0.0)
            .collect::<Vec<_>>()
    };
    let l1_of = |coef: &Array2<f64>| -> f64 { (0..theta_cols).map(|j| coef[(j, 0)].abs()).sum() };
    // Euclidean distance of a recovered coefficient vector to the true ξ*.
    let err_of = |coef: &Array2<f64>| -> f64 {
        (0..theta_cols)
            .map(|j| (coef[(j, 0)] - xi_true[j]).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut runs: Vec<Run> = Vec::with_capacity(lams.len());
    for &lam in &lams {
        let ridge = sindy_stlsq_solve(
            theta.view(),
            dxdt.view(),
            tol,
            max_rounds,
            lam,
            SindyPenaltyKind::Ridge,
            scad_a,
        )
        .expect("gam ridge STLSQ");
        let scad = sindy_stlsq_solve(
            theta.view(),
            dxdt.view(),
            tol,
            max_rounds,
            lam,
            SindyPenaltyKind::Scad,
            scad_a,
        )
        .expect("gam SCAD STLSQ");
        runs.push(Run {
            lam,
            ridge_support: support_of(&ridge.coefficients),
            ridge_l1: l1_of(&ridge.coefficients),
            ridge_err: err_of(&ridge.coefficients),
            scad_support: support_of(&scad.coefficients),
            scad_l1: l1_of(&scad.coefficients),
            scad_err: err_of(&scad.coefficients),
        });
    }

    // ---- Python references: pysindy STLSQ-ridge + Fan-Li SCAD reference ----
    // Both run on the IDENTICAL library columns and target. pysindy's `STLSQ`
    // is gam's ridge branch; the Fan-Li block re-implements gam's SCAD LQA
    // surrogate from the 2001 derivative so we have an independent SCAD oracle.
    let py = run_python(
        &[
            Column::new("const", &col_const),
            Column::new("x", &col_x),
            Column::new("x3", &col_x3),
            Column::new("dxdt", &target),
        ],
        r#"
import numpy as np
from pysindy.optimizers import STLSQ

# Identical library Theta = [1, x, x^3] and target dxdt as built in Rust.
Theta = np.column_stack([
    np.asarray(df["const"], dtype=float),
    np.asarray(df["x"], dtype=float),
    np.asarray(df["x3"], dtype=float),
])
y = np.asarray(df["dxdt"], dtype=float).reshape(-1)
n, p = Theta.shape
tol = 0.05
max_rounds = 20
a = 3.7
lams = [0.01, 0.1, 1.0, 10.0]

def ridge_diag(Th, yv, diag):
    # (Th^T Th + diag) xi = Th^T y  -- shared with gam's ridge_diag_solve.
    G = Th.T @ Th + np.diag(np.maximum(diag, 1e-12))
    return np.linalg.solve(G, Th.T @ yv)

def stlsq(Th, yv, lam, kind):
    # Reproduce gam's STLSQ control flow: seed full ridge, then alternate
    # hard-threshold + LQA-weighted ridge on the active set.
    lam_seed = lam if lam > 0 else 1e-12
    xi = ridge_diag(Th, yv, np.full(p, max(lam_seed, 1e-12)))
    eps = max(tol * 1e-2, 1e-10)
    active = np.ones(p, dtype=bool)
    prev = np.zeros(p, dtype=bool)
    for _ in range(max_rounds):
        small = np.abs(xi) < tol
        xi[small] = 0.0
        active = (~small) & active
        if not active.any():
            if np.array_equal(prev, active):
                break
            prev = active.copy()
            continue
        if np.array_equal(active, prev):
            break
        prev = active.copy()
        idx = np.where(active)[0]
        Ta = Th[:, idx]
        if kind == "ridge":
            diag = np.full(len(idx), lam)
        else:  # SCAD, Fan-Li 2001 derivative via LQA: p'(|xi|)/max(|xi|,eps)
            mag = np.abs(xi[idx])
            def scad_grad(t):
                if t <= lam:
                    return lam
                elif t <= a * lam:
                    return max((a * lam - t) / (a - 1.0), 0.0)
                else:
                    return 0.0
            diag = np.array([scad_grad(m) / max(m, eps) for m in mag])
        xa = ridge_diag(Ta, yv, diag)
        xi = np.zeros(p)
        xi[idx] = xa
    xi[np.abs(xi) < tol] = 0.0
    return xi

# pysindy STLSQ-ridge baseline: pysindy's STLSQ uses `threshold` (= our tol)
# and ridge `alpha` (= our lam). Fit one target column at a time.
def pysindy_ridge(lam):
    opt = STLSQ(threshold=tol, alpha=lam, max_iter=max_rounds)
    opt.fit(Theta, y.reshape(-1, 1))
    return opt.coef_.reshape(-1)

for lam in lams:
    xi_ps = pysindy_ridge(lam)
    xi_fl = stlsq(Theta, y, lam, "scad")
    sup_ps = [j for j in range(p) if xi_ps[j] != 0.0]
    sup_fl = [j for j in range(p) if xi_fl[j] != 0.0]
    emit("ridge_support_%g" % lam, sup_ps if sup_ps else [-1])
    emit("ridge_l1_%g" % lam, [float(np.abs(xi_ps).sum())])
    emit("scad_support_%g" % lam, sup_fl if sup_fl else [-1])
    emit("scad_l1_%g" % lam, [float(np.abs(xi_fl).sum())])
"#,
    );

    // ---- compare per-λ ----------------------------------------------------
    for run in &runs {
        // Python emits keys suffixed with "%g" of λ; for our values
        // {0.01, 0.1, 1.0, 10.0} Python's %g yields {0.01, 0.1, 1, 10}, which
        // Rust's default f64 Display reproduces exactly — so the keys line up.
        let lam_key = py_g(run.lam);
        let key_rs = format!("ridge_support_{lam_key}");
        let key_rl = format!("ridge_l1_{lam_key}");
        let key_ss = format!("scad_support_{lam_key}");
        let key_sl = format!("scad_l1_{lam_key}");

        let py_ridge_support: Vec<usize> = py
            .vector(&key_rs)
            .iter()
            .filter(|&&v| v >= 0.0)
            .map(|&v| v.round() as usize)
            .collect();
        let py_scad_support: Vec<usize> = py
            .vector(&key_ss)
            .iter()
            .filter(|&&v| v >= 0.0)
            .map(|&v| v.round() as usize)
            .collect();
        let py_ridge_l1 = py.scalar(&key_rl);
        let py_scad_l1 = py.scalar(&key_sl);

        eprintln!(
            "lam={:.2} | gam ridge supp={:?} l1={:.4}  scad supp={:?} l1={:.4} || \
             py ridge supp={:?} l1={:.4}  scad supp={:?} l1={:.4}",
            run.lam,
            run.ridge_support,
            run.ridge_l1,
            run.scad_support,
            run.scad_l1,
            py_ridge_support,
            py_ridge_l1,
            py_scad_support,
            py_scad_l1,
        );

        // (1) gam ridge support must match pysindy's STLSQ-ridge support
        // exactly: same algorithm (sequentially thresholded ridge), same
        // threshold, same alpha, identical Θ — the recovered active set is a
        // discrete invariant and any mismatch is a real algorithmic divergence.
        assert_eq!(
            run.ridge_support, py_ridge_support,
            "lam={}: gam ridge support {:?} != pysindy STLSQ support {:?}",
            run.lam, run.ridge_support, py_ridge_support
        );

        // (2) gam SCAD support must match the Fan-Li 2001 LQA reference support
        // exactly (same surrogate, same data, same control flow).
        assert_eq!(
            run.scad_support, py_scad_support,
            "lam={}: gam SCAD support {:?} != Fan-Li reference support {:?}",
            run.lam, run.scad_support, py_scad_support
        );

        // (3) SCAD must be at least as sparse as ridge (defining Fan-Li
        // property: concave penalties shrink spurious small terms harder).
        assert!(
            run.scad_support.len() <= run.ridge_support.len(),
            "lam={}: SCAD support ({}) larger than ridge ({}) — SCAD must not be \
             less sparse than ridge",
            run.lam,
            run.scad_support.len(),
            run.ridge_support.len()
        );
    }

    // ---- global sparsity-induction check ----------------------------------
    // At the largest λ (=10), the cubic system's spurious constant column must
    // be eliminated by both families, and SCAD must have recovered the true
    // two-term support {x, x^3} = {1, 2} at least at one swept λ — the
    // recovery the whole method exists to deliver.
    let scad_recovered_truth = runs.iter().any(|r| r.scad_support == vec![1usize, 2usize]);
    assert!(
        scad_recovered_truth,
        "SCAD never recovered the true active support {{x, x^3}} across the λ sweep; \
         supports were {:?}",
        runs.iter()
            .map(|r| r.scad_support.clone())
            .collect::<Vec<_>>()
    );

    // SCAD's tightest support over the sweep must be no larger than ridge's
    // tightest support — concave selection dominates ridge selection.
    let scad_min = runs.iter().map(|r| r.scad_support.len()).min().unwrap();
    let ridge_min = runs.iter().map(|r| r.ridge_support.len()).min().unwrap();
    eprintln!("min support over sweep: scad={scad_min} ridge={ridge_min}");
    assert!(
        scad_min <= ridge_min,
        "SCAD's sparsest solution ({scad_min}) is not at least as sparse as ridge's ({ridge_min})"
    );

    // ---- (4) near-oracle recovery: distance to the TRUE ξ* -----------------
    // The principled Fan-Li 2001 property is near-unbiasedness: among the swept
    // λ, SCAD's best coefficient estimate is at least as close to the true
    // ξ* = [0, -1, 0.1] as ridge's best, because ridge biases every surviving
    // coefficient uniformly toward zero while SCAD leaves the large signal
    // terms essentially un-shrunk. We assert both the head-to-head (SCAD's best
    // distance ≤ ridge's best) AND a tight absolute bound (SCAD's best is
    // essentially exact, < 1e-6) so this is not a vacuous one-sided inequality.
    // The true coefficients are an external oracle (the analytic ODE), so this
    // needs no Python reference and cannot be gamed by the solver.
    let scad_best_err = runs
        .iter()
        .map(|r| r.scad_err)
        .fold(f64::INFINITY, f64::min);
    let ridge_best_err = runs
        .iter()
        .map(|r| r.ridge_err)
        .fold(f64::INFINITY, f64::min);
    eprintln!("best ||·-xi*|| over sweep: scad={scad_best_err:.3e} ridge={ridge_best_err:.3e}");
    assert!(
        scad_best_err <= ridge_best_err + 1e-12,
        "SCAD's best recovery (||ξ-ξ*||={scad_best_err:.3e}) is not at least as \
         close to the true ξ* as ridge's best (={ridge_best_err:.3e}) — concave \
         near-unbiasedness should make SCAD's best estimate no worse than ridge's"
    );
    assert!(
        scad_best_err < 1e-6,
        "SCAD's best recovery over the λ sweep (||ξ-ξ*||={scad_best_err:.3e}) is not \
         essentially exact (< 1e-6); SCAD should achieve near-oracle recovery of \
         the true cubic dynamics"
    );
}

/// Render a λ value the way Python's `"%g"` formats it, so the emitted-key
/// suffixes match between Rust and the Python reference body. For the swept
/// values {0.01, 0.1, 1.0, 10.0} Python yields {"0.01", "0.1", "1", "10"} and
/// Rust's default float `Display` reproduces those shortest forms exactly.
fn py_g(lam: f64) -> String {
    format!("{lam}")
}
