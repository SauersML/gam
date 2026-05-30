//! End-to-end OBJECTIVE quality: gam's STLSQ loop must **recover the true
//! governing equations** of a known dynamical system from data — correct sparse
//! support and accurate coefficients — not merely reproduce pysindy's fitted
//! output.
//!
//! Sparse Identification of Nonlinear Dynamics (Brunton, Proctor & Kutz, PNAS
//! 2016) exists to answer one question objectively: *given trajectory data, do
//! we recover the equations that actually generated it?* We therefore generate
//! data from a system whose coefficients are KNOWN exactly, and assert gam
//! recovers that ground truth. The reference (pysindy's canonical `STLSQ`) is
//! kept only as a **match-or-beat baseline on coefficient accuracy** — gam's
//! error to the true coefficients must be no worse than pysindy's times 1.10.
//! Matching pysindy's noisy mask is explicitly NOT a pass criterion.
//!
//! The known multi-scale linear system (state = [x, y, z]):
//!     dx/dt = -10 x + 0.1 y
//!     dy/dt =  100 z
//!     dz/dt =  0.01 x
//! Its true coefficient matrix Ξ ∈ ℝ^{p×d} over the library Θ = [const, x, y, z]
//! (p = 4 features, d = 3 outputs) has exactly four nonzero entries:
//!     Ξ[1,0] = -10   (x → dx)      Ξ[2,0] = 0.1  (y → dx)
//!     Ξ[3,1] = 100   (z → dy)      Ξ[1,2] = 0.01 (x → dz)
//! and zero everywhere else. The coefficients span four decades, which is what
//! stresses a hard threshold: which terms should survive at a given `tol` is a
//! sharp, *knowable* function of the truth.
//!
//! We feed identical data (the 1200-row RK4-integrated trajectory and its
//! analytic derivatives Ẋ) to both engines and assert, against GROUND TRUTH:
//!
//!   1. NO FALSE POSITIVES (structure). At every `tol`, every coefficient gam
//!      keeps is one of the four true governing terms. A surviving term that is
//!      truly zero is spurious identification — overfitting the algorithm must
//!      not do.
//!
//!   2. EXACT TRUTH RECOVERY at the finest threshold. With `tol = 0.001` (below
//!      every true magnitude, including the 0.01 term), gam's surviving support
//!      must be *exactly* the four-term true active set — no term dropped, none
//!      added.
//!
//!   3. COEFFICIENT ACCURACY vs truth (truth recovery + match-or-beat). At the
//!      finest threshold gam's recovered coefficients must track the true Ξ:
//!      relative-L2 error over all p·d entries is small in absolute terms, AND
//!      no worse than pysindy's error to the same truth times 1.10.
//!
//!   4. CONVERGENCE. The system is exactly representable in the library, so
//!      STLSQ must reach a stable support strictly inside the iteration cap at
//!      every tol; a cap-pinned or non-converged run signals a broken loop.
//!
//! A weak ridge (`lam = 1e-4`) keeps the hard threshold — not the L2 shrinkage —
//! in control of which terms survive; pysindy gets the matching `alpha = 1e-4`.

use gam::solver::sindy::{SindyPenaltyKind, sindy_stlsq_solve};
use gam::test_support::reference::{Column, relative_l2, run_python};
use ndarray::Array2;

/// Number of trajectory rows.
const N: usize = 1200;
/// RK4 step.
const DT: f64 = 1.0e-3;
/// Ridge weight: deliberately tiny so the hard threshold (not the L2 penalty)
/// governs which terms survive.
const LAM: f64 = 1.0e-4;
/// STLSQ iteration cap, shared by both engines.
const MAX_ROUNDS: usize = 30;
/// Thresholds swept, tight → loose.
const TOLS: [f64; 4] = [0.001, 0.01, 0.05, 0.1];
/// Library width (p = number of features) and output count (d = number of
/// state-derivative columns).
const P: usize = 4;
const DD: usize = 3;

/// One RK4 step of the multi-scale linear system. State = [x, y, z].
fn rhs(s: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = s;
    [
        -10.0 * x + 0.1 * y, // dx/dt
        100.0 * z,           // dy/dt
        0.01 * x,            // dz/dt
    ]
}

fn rk4_step(s: [f64; 3], dt: f64) -> [f64; 3] {
    let k1 = rhs(s);
    let s2 = [
        s[0] + 0.5 * dt * k1[0],
        s[1] + 0.5 * dt * k1[1],
        s[2] + 0.5 * dt * k1[2],
    ];
    let k2 = rhs(s2);
    let s3 = [
        s[0] + 0.5 * dt * k2[0],
        s[1] + 0.5 * dt * k2[1],
        s[2] + 0.5 * dt * k2[2],
    ];
    let k3 = rhs(s3);
    let s4 = [s[0] + dt * k3[0], s[1] + dt * k3[1], s[2] + dt * k3[2]];
    let k4 = rhs(s4);
    [
        s[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        s[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        s[2] + dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    ]
}

/// True coefficient matrix Ξ flattened in gam's layout (column-major over
/// outputs: index = c * P + j for output c, feature j). Four nonzero entries.
fn true_coeffs_flat() -> Vec<f64> {
    let mut t = vec![0.0f64; P * DD];
    let slot = |c: usize, j: usize| c * P + j;
    // output c=0 (dx): -10 x + 0.1 y
    t[slot(0, 1)] = -10.0; // x
    t[slot(0, 2)] = 0.1; // y
    // output c=1 (dy): 100 z
    t[slot(1, 3)] = 100.0; // z
    // output c=2 (dz): 0.01 x
    t[slot(2, 1)] = 0.01; // x
    t
}

#[test]
fn stlsq_recovers_true_governing_equations() {
    // ---- build the identical trajectory both engines will see -------------
    // Fixed deterministic seed=42 LCG perturbs only the *initial* condition so
    // the trajectory excites all three states; the dynamics themselves are the
    // exact multi-scale system above. The same x/y/z/derivative arrays are fed
    // to gam (as ndarray) and pysindy (as CSV columns) — no divergence.
    let mut seed = 42u64;
    let mut unit = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0 // in [-1, 1)
    };
    let mut s = [unit(), unit(), unit()];

    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut dx = Vec::with_capacity(N);
    let mut dy = Vec::with_capacity(N);
    let mut dz = Vec::with_capacity(N);
    for _ in 0..N {
        let d = rhs(s);
        x.push(s[0]);
        y.push(s[1]);
        z.push(s[2]);
        dx.push(d[0]);
        dy.push(d[1]);
        dz.push(d[2]);
        s = rk4_step(s, DT);
    }

    // Library Θ = [const, x, y, z]  (p = 4). Targets Ẋ = [dx, dy, dz] (d = 3).
    let mut theta = Array2::<f64>::zeros((N, P));
    let mut dzdt = Array2::<f64>::zeros((N, DD));
    for i in 0..N {
        theta[(i, 0)] = 1.0;
        theta[(i, 1)] = x[i];
        theta[(i, 2)] = y[i];
        theta[(i, 3)] = z[i];
        dzdt[(i, 0)] = dx[i];
        dzdt[(i, 1)] = dy[i];
        dzdt[(i, 2)] = dz[i];
    }

    let truth = true_coeffs_flat();
    // The four true governing slots (flattened c*P + j) — used to detect any
    // false-positive (spurious) term gam might keep.
    let true_support: Vec<usize> = (0..P * DD).filter(|&k| truth[k] != 0.0).collect();
    assert_eq!(true_support.len(), 4, "true system has exactly four terms");

    // ---- gam: run STLSQ at each tol; record coefficients, support, loop ----
    let mut gam_converged = [false; TOLS.len()];
    let mut gam_rounds = [0usize; TOLS.len()];
    let mut gam_flat: Vec<Vec<f64>> = Vec::with_capacity(TOLS.len());

    for (ti, &tol) in TOLS.iter().enumerate() {
        let res = sindy_stlsq_solve(
            theta.view(),
            dzdt.view(),
            tol,
            MAX_ROUNDS,
            LAM,
            SindyPenaltyKind::Ridge,
            3.7, // ignored for Ridge
        )
        .expect("gam STLSQ must succeed");

        // Flatten Ξ in gam's layout (column-major over outputs): index = c*P + j.
        let mut flat = vec![0.0f64; P * DD];
        for c in 0..DD {
            for j in 0..P {
                flat[c * P + j] = res.coefficients[(j, c)];
            }
        }
        let card = flat.iter().filter(|&&v| v != 0.0).count();
        gam_converged[ti] = res.converged;
        gam_rounds[ti] = res.rounds_used;
        gam_flat.push(flat);

        eprintln!(
            "gam tol={tol:>6}: rounds={} converged={} support_card={}",
            res.rounds_used, res.converged, card
        );
    }

    // ---- pysindy: identical data, identical thresholds (baseline) ---------
    // We drive pysindy's STLSQ optimizer directly with the same Θ and Ẋ so the
    // algorithm — not pysindy's feature generation — is what is compared. For
    // each tol we emit the fitted coefficient matrix flattened in gam's layout
    // (column-major over outputs => for c in 0..d, for j in 0..p). pysindy is a
    // BASELINE on accuracy-to-truth only; we never assert gam matches its mask.
    let tol_list: Vec<f64> = TOLS.to_vec();
    let ref_res = run_python(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("z", &z),
            Column::new("dx", &dx),
            Column::new("dy", &dy),
            Column::new("dz", &dz),
        ],
        &format!(
            r#"
import numpy as np
from pysindy.optimizers import STLSQ

# Library Theta = [const, x, y, z]; targets Xdot = [dx, dy, dz].
n = len(df["x"])
const = np.ones(n)
Theta = np.column_stack([const, np.asarray(df["x"]), np.asarray(df["y"]), np.asarray(df["z"])])
Xdot  = np.column_stack([np.asarray(df["dx"]), np.asarray(df["dy"]), np.asarray(df["dz"])])
p = Theta.shape[1]
d = Xdot.shape[1]

tols = {tols:?}
lam = {lam}
max_rounds = {max_rounds}

coef_all = []
for tol in tols:
    opt = STLSQ(threshold=tol, alpha=lam, max_iter=max_rounds)
    # pysindy fits coef_ of shape (n_targets, n_features) = (d, p).
    opt.fit(Theta, Xdot)
    coef = np.asarray(opt.coef_)          # (d, p)
    # Flatten in gam's layout: for c in 0..d, for j in 0..p -> c*p + j.
    coef_all.extend(coef.reshape(d * p).tolist())

emit("coef", coef_all)
"#,
            tols = tol_list,
            lam = LAM,
            max_rounds = MAX_ROUNDS,
        ),
    );

    let py_coef = ref_res.vector("coef");
    assert_eq!(
        py_coef.len(),
        TOLS.len() * P * DD,
        "pysindy coefficient layout mismatch"
    );

    // ---- assertion 1: NO FALSE POSITIVES at any tol (structure) -----------
    // Every coefficient gam keeps must be one of the four true governing terms.
    // A surviving slot that is truly zero is spurious identification.
    for (ti, &tol) in TOLS.iter().enumerate() {
        for k in 0..(P * DD) {
            if gam_flat[ti][k] != 0.0 {
                assert!(
                    truth[k] != 0.0,
                    "tol={tol}: gam kept spurious term at slot {k} (true coefficient is zero)"
                );
            }
        }
    }

    // ---- assertion 2: EXACT TRUTH RECOVERY at the finest threshold --------
    // tol=0.001 sits below every true magnitude (smallest is 0.01), so the
    // surviving support must be EXACTLY the four-term true active set.
    let fine = 0; // TOLS[0] = 0.001
    for &k in &true_support {
        assert!(
            gam_flat[fine][k] != 0.0,
            "tol={}: gam failed to recover true term at slot {k}",
            TOLS[fine]
        );
    }
    let fine_card = gam_flat[fine].iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        fine_card,
        true_support.len(),
        "tol={}: gam support cardinality {fine_card} != true {} (exact recovery required)",
        TOLS[fine],
        true_support.len()
    );

    // ---- assertion 3: COEFFICIENT ACCURACY vs truth (+ match-or-beat) -----
    // At the finest threshold gam's recovered coefficients must track the true
    // Ξ closely (relative-L2 over all p*d entries), and no worse than pysindy's
    // error to the same truth times 1.10. The PRIMARY claim is truth recovery;
    // pysindy is only a same-or-better baseline.
    let gam_rel_truth = relative_l2(&gam_flat[fine], &truth);
    let py_fine: Vec<f64> = py_coef[fine * P * DD..(fine + 1) * P * DD].to_vec();
    let py_rel_truth = relative_l2(&py_fine, &truth);
    eprintln!(
        "tol={}: rel_l2(gam, truth)={gam_rel_truth:.3e}  rel_l2(pysindy, truth)={py_rel_truth:.3e}",
        TOLS[fine]
    );
    // Absolute objective bar: the system is exactly linear-in-library and
    // analytically differentiated, so the recovered coefficients must be within
    // 1% (relative L2) of truth.
    assert!(
        gam_rel_truth <= 1.0e-2,
        "tol={}: gam relative-L2 error to true coefficients {gam_rel_truth:.3e} exceeds 1e-2",
        TOLS[fine]
    );
    // Match-or-beat pysindy on accuracy-to-truth.
    assert!(
        gam_rel_truth <= py_rel_truth * 1.10 + 1.0e-12,
        "tol={}: gam error-to-truth {gam_rel_truth:.3e} worse than pysindy {py_rel_truth:.3e} * 1.10",
        TOLS[fine]
    );

    // ---- assertion 4: gam reports a converged, bounded loop everywhere ----
    // The system is exactly representable in the library, so STLSQ must reach a
    // stable support well inside the cap; a non-converged or cap-pinned run
    // would signal a broken loop.
    for (ti, &tol) in TOLS.iter().enumerate() {
        assert!(
            gam_converged[ti],
            "tol={tol}: gam STLSQ failed to converge within {MAX_ROUNDS} rounds"
        );
        assert!(
            gam_rounds[ti] >= 1 && gam_rounds[ti] < MAX_ROUNDS,
            "tol={tol}: gam rounds_used={} out of expected (1, {MAX_ROUNDS}) range",
            gam_rounds[ti]
        );
    }
}
