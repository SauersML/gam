//! #1266 instrumentation: the B-spline double-penalty linear-data EDF gate
//! routes through the O(n) state-space spline scan (`fit_spline_scan`, order 2).
//! After the shrinkage-floor fix the per-seed EDF is `[2.0, 3.4, 2.6, 3.6, 2.0]`
//! — two seeds nail the linear null (EDF 2) while three retain spurious wiggle.
//!
//! This probe replays the EXACT gate DGP (StdRng, n=800, y = 2 + 3x + N(0,0.15))
//! and, per seed, prints the auto-selected `log λ` / EDF AND sweeps the restricted
//! criterion over a fine `log λ` grid. It distinguishes the two failure modes:
//!
//!   (a) GRID/LOCAL-MAX MISS — the criterion is (weakly) monotone toward high λ
//!       but the 25-point grid + ±1-step golden section settles on an interior
//!       point, so a finer/expanded search would recover EDF≈2. Then the fix is
//!       in the SELECTION (bracket/grid), not the criterion.
//!
//!   (b) GENUINE INTERIOR MAX — the criterion truly peaks at a finite λ for that
//!       noise draw (the swept maximum is interior and beats the high-λ tail).
//!       Then the scan REML criterion itself under-smooths vs mgcv and the fix is
//!       in the criterion (e.g. the diffuse dof / σ² profiling).
//!
//! Report-only (no hard gate): prints the discriminating numbers.

use gam::solver::spline_scan::{fit_spline_scan, fit_spline_scan_at};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn gate_dgp(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 / (n.saturating_sub(1).max(1)) as f64;
        x.push(xi);
        y.push(2.0 + 3.0 * xi + noise.sample(&mut rng));
    }
    let w = vec![1.0; n];
    (x, y, w)
}

#[test]
fn spline_scan_1266_lambda_probe() {
    let n = 800usize;
    let order = 2usize;

    eprintln!(
        "[scan-1266] {:>4} {:>12} {:>10} | swept argmax over fine logλ grid",
        "seed", "sel_logλ", "sel_edf"
    );

    for seed in 0..5u64 {
        let (x, y, w) = gate_dgp(seed, n);

        // Auto-selected fit (what the gate sees).
        let fit = fit_spline_scan(&x, &y, &w, order).expect("scan fit");
        let sel_ll = fit.log_lambda;
        let sel_edf = fit.edf();

        // Fine sweep of the restricted criterion (proxy: restricted_loglik at the
        // profiled σ², which differs from the concentrated criterion only by an
        // additive constant — same argmax).
        let lo = -10.0f64;
        let hi = 40.0f64; // well past the gate's hi_anchor so the tail is visible
        let steps = 200usize;
        let mut best_ll = lo;
        let mut best_crit = f64::NEG_INFINITY;
        let mut best_edf = f64::NAN;
        let mut tail_crit = f64::NEG_INFINITY; // criterion at the high-λ end
        let mut samples: Vec<(f64, f64, f64)> = Vec::new();
        for s in 0..=steps {
            let ll = lo + (hi - lo) * s as f64 / steps as f64;
            let f = match fit_spline_scan_at(&x, &y, &w, ll, None, order) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let crit = f.restricted_loglik;
            let edf = f.edf();
            if crit > best_crit {
                best_crit = crit;
                best_ll = ll;
                best_edf = edf;
            }
            tail_crit = crit; // last (highest λ) sample
            // Keep a coarse trace for printing.
            if s % 20 == 0 {
                samples.push((ll, crit, edf));
            }
        }

        let interior = best_ll < hi - 1e-6;
        let margin = best_crit - tail_crit;
        eprintln!(
            "[scan-1266] {seed:>4} {sel_ll:>12.4} {sel_edf:>10.4} | swept argmax logλ={best_ll:.3} \
             edf={best_edf:.4} crit={best_crit:.6} tail_crit={tail_crit:.6} \
             margin(best−tail)={margin:.3e} interior_max={interior}"
        );
        for (ll, crit, edf) in &samples {
            eprintln!("[scan-1266]        logλ={ll:>7.2}  crit={crit:>14.6}  edf={edf:>8.4}");
        }
    }

    eprintln!(
        "[scan-1266] READ: if swept argmax is INTERIOR (interior_max=true) with a positive \
         margin, the scan REML criterion genuinely peaks at finite λ → criterion under-smooths \
         (fix the criterion). If the swept argmax is at the high-λ tail but the AUTO sel_logλ is \
         much lower, the SELECTION (grid/bracket) is missing it (fix selection)."
    );
}
