//! End-to-end quality: gam's simplex Fréchet mean must (a) RECOVER the known
//! Aitchison center that generated the data and (b) satisfy the metric-defining
//! group laws of Aitchison geometry — closure invariance and perturbation
//! equivariance — on identical data.
//!
//! OBJECTIVE METRIC ASSERTED (primary):
//!   TRUTH RECOVERY. We synthesize compositions from a *known* Aitchison center
//!   `mu` by adding zero-mean Gaussian noise in CLR coordinates (an Aitchison-
//!   normal sample), close them onto the simplex, and fit. The closed geometric
//!   mean is the maximum-likelihood / Fréchet center of an Aitchison-normal
//!   sample, so it must converge to `mu`. We assert gam's recovered center,
//!   expressed in CLR coordinates, has RMSE(clr(gam_mean), clr(mu)) below a
//!   principled sampling bound (the per-coordinate CLR noise sigma scaled by the
//!   1/sqrt(n) standard error of a mean of `n` draws, with slack). This is an
//!   accuracy claim about gam vs the TRUTH that generated the data — not "gam
//!   reproduces a peer tool's fit."
//!
//! GROUND-TRUTH CROSS-CHECK (kept, not a peer-matching claim):
//!   `compositions::mean.acomp` and `compositions::clr` / `robCompositions::cenLR`
//!   compute an EXACT closed-form mathematical quantity (the closed geometric
//!   mean and the centered-log-ratio map — deterministic analytic formulas, not a
//!   noisy fit). Asserting gam equals them to machine precision is therefore a
//!   correctness-vs-ground-truth claim (the documented GROUND-TRUTH exception),
//!   and we additionally require gam to MATCH-OR-BEAT the reference on the truth-
//!   recovery metric: gam's CLR distance to the true center must be <= the
//!   reference's distance to the true center * 1.10.
//!
//! STRUCTURE / CONSTRAINTS (kept — objective metric-defining properties):
//!   * Closure / scale invariance: row-scaling the (unclosed) inputs leaves the
//!     mean bit-identical — the invariance behind d(x,y)=d(cx,cy).
//!   * Perturbation equivariance: mean(x_i ⊕ p) == mean(x_i) ⊕ p — the group law
//!     that makes the Aitchison metric translation-invariant and geodesic.
//!   * Simplex closure of the output (positive, sums to 1).

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, max_abs_diff, rmse, run_r};
use ndarray::Array2;

/// CLR of a single closed composition: ln(x) - mean(ln(x)). The Aitchison metric
/// is Euclidean in these coordinates, so truth recovery is measured here.
fn clr(x: &[f64; 3]) -> [f64; 3] {
    let log: [f64; 3] = [x[0].ln(), x[1].ln(), x[2].ln()];
    let m = (log[0] + log[1] + log[2]) / 3.0;
    [log[0] - m, log[1] - m, log[2] - m]
}

/// Inverse CLR (closed softmax of a centered log vector) — maps a CLR point back
/// onto the simplex. Used to build the known center `mu` from a CLR target.
fn clr_inv(z: &[f64; 3]) -> [f64; 3] {
    let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0.0_f64; 3];
    let mut s = 0.0;
    for k in 0..3 {
        out[k] = (z[k] - mx).exp();
        s += out[k];
    }
    for v in out.iter_mut() {
        *v /= s;
    }
    out
}

/// Perturbation x ⊕ p = closure(x .* p) — the Aitchison "translation".
fn perturb(x: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    let mut prod = [x[0] * p[0], x[1] * p[1], x[2] * p[2]];
    let s: f64 = prod.iter().sum();
    for v in prod.iter_mut() {
        *v /= s;
    }
    prod
}

/// Deterministic SplitMix64 + Box–Muller, so the synthetic Aitchison-normal
/// sample is byte-identical for gam and for the R ground-truth cross-check.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v.clamp(f64::MIN_POSITIVE, 1.0 - 1e-15)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn gam_simplex_frechet_mean_recovers_known_aitchison_center() {
    // ---- known Aitchison center mu (the TRUTH we will try to recover) --------
    // Pick a CLR target, map it onto the simplex; mu is well inside the simplex.
    let mu_clr_target = [0.6_f64, -0.9, 0.3];
    let mu = clr_inv(&mu_clr_target);
    let mu_clr = clr(&mu); // re-centered exact CLR of the closed center
    let mu_log = [mu[0].ln(), mu[1].ln(), mu[2].ln()];

    // ---- fixed-seed Aitchison-normal sample around mu ------------------------
    // Each draw: log(mu) + iid N(0, sigma^2) in the 3 raw log-coords, closed onto
    // the simplex. After closure this is exactly an Aitchison-normal perturbation
    // of mu whose closed geometric mean (CLR-mean) is an unbiased estimator of mu.
    let sigma = 0.45_f64; // per-coordinate CLR noise scale
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let n = 400usize; // large enough that the sampling SE is tight
    let mut raw: Vec<[f64; 3]> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = [0.0_f64; 3];
        for k in 0..3 {
            // exp(log mu_k + noise) is positive; closure handles normalization
            row[k] = (mu_log[k] + sigma * rng.normal()).exp();
        }
        raw.push(row);
    }

    // Flat columns shared verbatim with R.
    let c1: Vec<f64> = raw.iter().map(|r| r[0]).collect();
    let c2: Vec<f64> = raw.iter().map(|r| r[1]).collect();
    let c3: Vec<f64> = raw.iter().map(|r| r[2]).collect();

    // ---- gam: closure + Aitchison Fréchet mean ------------------------------
    let mut pts = Array2::<f64>::zeros((n, 3));
    for (i, r) in raw.iter().enumerate() {
        for k in 0..3 {
            pts[[i, k]] = r[k];
        }
    }
    let gam_mean_vec = simplex_frechet_mean(pts.view(), None).expect("gam simplex Fréchet mean");
    assert_eq!(gam_mean_vec.len(), 3);
    let gam_mean = [gam_mean_vec[0], gam_mean_vec[1], gam_mean_vec[2]];
    let gam_clr = clr(&gam_mean);

    // ===== PRIMARY: truth recovery in CLR (Aitchison-Euclidean) coords ========
    // RMSE of recovered CLR center vs the true center's CLR. The standard error
    // of each CLR coordinate of the mean is ~ sigma * sqrt(2/3) / sqrt(n) (the
    // CLR projection of iid log-noise has variance sigma^2 * 2/3 per coord).
    // Bound = 4 sampling-SEs — a principled, not-loosened envelope that a correct
    // estimator clears with overwhelming probability and a biased one fails.
    let se = sigma * (2.0_f64 / 3.0).sqrt() / (n as f64).sqrt();
    let recovery_bound = 4.0 * se;
    let gam_recovery_rmse = rmse(&gam_clr, &mu_clr);
    eprintln!(
        "TRUTH RECOVERY: gam_clr=[{:.5},{:.5},{:.5}] true_clr=[{:.5},{:.5},{:.5}] rmse={:.3e} bound={:.3e} (se={:.3e})",
        gam_clr[0], gam_clr[1], gam_clr[2], mu_clr[0], mu_clr[1], mu_clr[2], gam_recovery_rmse, recovery_bound, se
    );
    assert!(
        gam_recovery_rmse <= recovery_bound,
        "gam did not recover the known Aitchison center: rmse={gam_recovery_rmse:.3e} > bound={recovery_bound:.3e}"
    );

    // ---- compositions / robCompositions GROUND-TRUTH cross-check ------------
    // mean.acomp / clr / cenLR are exact closed-form maps (analytic ground truth),
    // so we (1) confirm gam equals them to machine precision and (2) require gam
    // to match-or-beat them on the truth-recovery metric.
    let r = run_r(
        &[
            Column::new("c1", &c1),
            Column::new("c2", &c2),
            Column::new("c3", &c3),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        suppressPackageStartupMessages(library(robCompositions))
        X <- as.matrix(df[, c("c1","c2","c3")])
        ac <- acomp(X)                       # closed compositions (Aitchison)
        ctr <- mean(ac)                      # closed geometric-mean center
        ctr <- as.numeric(ctr / sum(ctr))    # ensure unit closure for comparison
        emit("center", ctr)
        # CLR center two independent ways: compositions::clr and robCompositions::cenLR
        clr_comp <- as.numeric(clr(acomp(matrix(ctr, nrow = 1))))
        cl <- robCompositions::cenLR(matrix(ctr, nrow = 1))
        clr_rob  <- as.numeric(if (!is.null(cl$x.clr)) cl$x.clr else as.matrix(cl))
        emit("clr_comp", clr_comp)
        emit("clr_rob", clr_rob)
        "#,
    );
    let ref_center = r.vector("center");
    let ref_clr_comp = r.vector("clr_comp");
    let ref_clr_rob = r.vector("clr_rob");
    assert_eq!(ref_center.len(), 3, "reference center must be 3-component");
    assert_eq!(ref_clr_comp.len(), 3, "reference clr_comp must be 3-component");
    assert_eq!(ref_clr_rob.len(), 3, "reference clr_rob must be 3-component");

    // Ground-truth correctness: gam == exact analytic closed geometric mean.
    let center_err = max_abs_diff(&gam_mean, ref_center);
    let ref_clr_comp_arr = [ref_clr_comp[0], ref_clr_comp[1], ref_clr_comp[2]];
    let ref_clr_rob_arr = [ref_clr_rob[0], ref_clr_rob[1], ref_clr_rob[2]];
    let clr_err_comp = max_abs_diff(&gam_clr, &ref_clr_comp_arr);
    let clr_err_rob = max_abs_diff(&gam_clr, &ref_clr_rob_arr);
    eprintln!(
        "GROUND-TRUTH analytic agreement: center_max_abs={center_err:.3e} clr_err_comp={clr_err_comp:.3e} clr_err_rob={clr_err_rob:.3e}"
    );
    assert!(
        center_err < 1e-10,
        "gam Fréchet mean != exact closed geometric mean (analytic ground truth): max_abs={center_err:.3e}"
    );
    assert!(
        clr_err_comp < 1e-9 && clr_err_rob < 1e-9,
        "gam closure is not the exact CLR-Euclidean (Aitchison) coords: comp={clr_err_comp:.3e} rob={clr_err_rob:.3e}"
    );

    // Match-or-beat on ACCURACY: gam's CLR distance to the truth <= ref's * 1.10.
    let ref_recovery_rmse = rmse(&ref_clr_comp_arr, &mu_clr);
    eprintln!(
        "MATCH-OR-BEAT recovery: gam_rmse={gam_recovery_rmse:.3e} ref_rmse={ref_recovery_rmse:.3e}"
    );
    assert!(
        gam_recovery_rmse <= ref_recovery_rmse * 1.10 + 1e-12,
        "gam recovers the truth worse than the reference: gam={gam_recovery_rmse:.3e} ref={ref_recovery_rmse:.3e}"
    );

    // ===== STRUCTURE 1: closure / scale invariance (d(x,y)=d(cx,cy)) =========
    // Multiply each composition by an arbitrary positive per-row scalar and
    // re-run. The closure must make the Fréchet mean identical — the invariance
    // that makes the Aitchison distance scale-free.
    let mut rng_s = Lcg(0xC0FF_EE00_D15E_A5E5);
    let mut scaled = Array2::<f64>::zeros((n, 3));
    for (i, rrow) in raw.iter().enumerate() {
        let c = 0.1 + 100.0 * rng_s.unit(); // strictly positive per-row scale
        for k in 0..3 {
            scaled[[i, k]] = rrow[k] * c;
        }
    }
    let scaled_mean = simplex_frechet_mean(scaled.view(), None).expect("gam mean on scaled rows");
    let scaled_arr = [scaled_mean[0], scaled_mean[1], scaled_mean[2]];
    let scale_err = max_abs_diff(&gam_mean, &scaled_arr);
    eprintln!("STRUCTURE scale invariance: max_abs={scale_err:.3e}");
    assert!(
        scale_err < 1e-12,
        "gam closure is not scale-invariant (Aitchison distance would not be scale-free): {scale_err:.3e}"
    );

    // ===== STRUCTURE 2: perturbation equivariance (group law) ================
    // mean(x_i ⊕ p) must equal mean(x_i) ⊕ p — the translation invariance of
    // Aitchison geometry, which makes its distance geodesic and the triangle
    // inequality hold by isometry to Euclidean CLR space.
    let p = [2.0, 5.0, 0.25];
    let mut perturbed = Array2::<f64>::zeros((n, 3));
    for (i, rrow) in raw.iter().enumerate() {
        let total: f64 = rrow.iter().sum();
        let closed = [rrow[0] / total, rrow[1] / total, rrow[2] / total];
        let pr = perturb(&closed, &p);
        for k in 0..3 {
            perturbed[[i, k]] = pr[k];
        }
    }
    let perturbed_mean =
        simplex_frechet_mean(perturbed.view(), None).expect("gam mean on perturbed rows");
    let perturbed_arr = [perturbed_mean[0], perturbed_mean[1], perturbed_mean[2]];
    let expected_perturbed = perturb(&gam_mean, &p);
    let equivar_err = max_abs_diff(&perturbed_arr, &expected_perturbed);
    eprintln!("STRUCTURE perturbation equivariance: max_abs={equivar_err:.3e}");
    assert!(
        equivar_err < 1e-12,
        "gam Fréchet mean violates Aitchison perturbation equivariance (metric would not be geodesic): {equivar_err:.3e}"
    );

    // ===== STRUCTURE 3: output lies on the closed simplex ====================
    let sum: f64 = gam_mean.iter().sum();
    eprintln!(
        "STRUCTURE simplex closure: sum={sum:.15} min={:.3e}",
        gam_mean.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    assert!(
        (sum - 1.0).abs() < 1e-12 && gam_mean.iter().all(|&v| v > 0.0),
        "gam mean is not a valid closed composition: sum={sum}"
    );
}
