//! Aggressive marginal-preservation identity tests for the multi-z survival
//! marginal-slope likelihood.
//!
//! The identity under test is, for the marginal-preserving scale
//! `c = sqrt(1 + r' Σ r)` (with `r` the observed scaled slope `probit_scale * g`):
//!
//!     Φ(-q) == Φ(-q * c / sqrt(1 + r' Σ r))
//!
//! When `c` is exactly the preserving scale this collapses to `Φ(-q) == Φ(-q)`
//! up to floating point. Any subtle bug in the vector generalisation of `c(a)`
//! (e.g. dropping a `probit_scale`, mixing up `Full` vs `LowRank` quadratic
//! forms, permutation sensitivity) makes one or more of these tests blow past
//! the 2e-15 tolerance.

use gam::bernoulli_marginal_slope::{
    MarginalSlopeCovariance, marginal_slope_covariance_from_scores,
};
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::{
    survival_marginal_slope_vector_eta, survival_marginal_slope_vector_neglog,
    survival_marginal_slope_vector_scale,
};
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (splitmix64 -> f64 in [0,1)) so we do not pull in a
// new crate dependency.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform [0,1).
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    /// Uniform [-1,1).
    fn next_signed(&mut self) -> f64 {
        2.0 * self.next_unit() - 1.0
    }
    /// Approx standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        // Avoid log(0).
        let u1 = (self.next_unit()).max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        // hi exclusive
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_marginal_preservation(
    q: f64,
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
    tol: f64,
    label: &str,
) {
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)
        .unwrap_or_else(|err| panic!("{label}: scale failed: {err}"));
    let observed: Vec<f64> = slopes.iter().map(|&r| probit_scale * r).collect();
    let variance = covariance
        .quadratic_form(&observed)
        .unwrap_or_else(|err| panic!("{label}: quadratic form failed: {err}"));
    let lhs = normal_cdf(-q * c / (1.0 + variance).sqrt());
    let rhs = normal_cdf(-q);
    let diff = (lhs - rhs).abs();
    assert!(
        diff <= tol,
        "{label}: identity violated; lhs={lhs:.17e} rhs={rhs:.17e} diff={diff:.3e} tol={tol:.3e}"
    );
}

fn random_diagonal(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    let mut diag = Array1::<f64>::zeros(k);
    for i in 0..k {
        diag[i] = 0.01 + 2.0 * rng.next_unit();
    }
    MarginalSlopeCovariance::Diagonal(diag)
}

fn random_full(rng: &mut SplitMix64, k: usize) -> (MarginalSlopeCovariance, Array2<f64>) {
    // Σ = L Lᵀ + εI, build dense.
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            l[[i, j]] = rng.next_signed();
        }
    }
    let mut cov = l.dot(&l.t());
    for i in 0..k {
        cov[[i, i]] += 1e-3;
    }
    (MarginalSlopeCovariance::Full(cov.clone()), cov)
}

fn random_low_rank(rng: &mut SplitMix64, k: usize, r: usize) -> MarginalSlopeCovariance {
    let mut f = Array2::<f64>::zeros((k, r));
    for i in 0..k {
        for j in 0..r {
            f[[i, j]] = rng.next_signed();
        }
    }
    MarginalSlopeCovariance::LowRank(f)
}

fn random_slopes(rng: &mut SplitMix64, k: usize, norm: f64) -> Vec<f64> {
    let mut s: Vec<f64> = (0..k).map(|_| rng.next_normal()).collect();
    let n2: f64 = s.iter().map(|v| v * v).sum();
    let inv = if n2 > 0.0 { norm / n2.sqrt() } else { 0.0 };
    for v in &mut s {
        *v *= inv;
    }
    s
}

fn random_z(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.next_normal()).collect()
}

// ---------------------------------------------------------------------------
// 1. Randomised sweep across shapes and dimensions.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_random_sweep_all_shapes() {
    assert!(file!().ends_with(".rs"));
    let probit_scales = [1e-3_f64, 1.0, 1e2];
    let slope_norms = [1e-3_f64, 1.0, 10.0];
    let qs = [-3.0_f64, 0.0, 3.0];

    for seed in 0u64..220 {
        let mut rng = SplitMix64::new(0xDEAD_BEEF ^ seed);
        let k = rng.range(2, 9); // K in [2,8]
        // pick shape deterministically from seed
        let shape_pick = seed % 3;

        let cov = match shape_pick {
            0 => random_diagonal(&mut rng, k),
            1 => random_full(&mut rng, k).0,
            _ => {
                let r = rng.range(1, k); // r in [1, k-1]
                random_low_rank(&mut rng, k, r)
            }
        };

        let slope_norm = slope_norms[(seed as usize) % slope_norms.len()];
        let probit_scale = probit_scales[((seed as usize) / 3) % probit_scales.len()];
        let q = qs[((seed as usize) / 9) % qs.len()];

        let slopes = random_slopes(&mut rng, k, slope_norm);
        let label = format!(
            "sweep seed={seed} K={k} shape={:?} probit={probit_scale:.0e} norm={slope_norm:.0e} q={q}",
            cov.shape()
        );
        assert_marginal_preservation(q, &slopes, &cov, probit_scale, 2e-15, &label);
    }
}

// ---------------------------------------------------------------------------
// 2. Extreme magnitudes (explicit grid).
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_extreme_magnitudes_grid() {
    assert!(file!().ends_with(".rs"));
    let probit_scales = [1e-3_f64, 1.0, 1e2];
    let slope_norms = [1e-3_f64, 1.0, 10.0];
    let qs = [-3.0_f64, 0.0, 3.0];

    let mut rng = SplitMix64::new(0xCAFEBABE);
    let k = 5;
    for &ps in &probit_scales {
        for &sn in &slope_norms {
            for &q in &qs {
                // exercise each shape
                let diag = random_diagonal(&mut rng, k);
                let full = random_full(&mut rng, k).0;
                let lr = random_low_rank(&mut rng, k, 2);
                let slopes = random_slopes(&mut rng, k, sn);
                let label = format!("extreme ps={ps:.0e} sn={sn:.0e} q={q}");
                assert_marginal_preservation(
                    q,
                    &slopes,
                    &diag,
                    ps,
                    2e-15,
                    &(label.clone() + " diag"),
                );
                assert_marginal_preservation(
                    q,
                    &slopes,
                    &full,
                    ps,
                    2e-15,
                    &(label.clone() + " full"),
                );
                assert_marginal_preservation(q, &slopes, &lr, ps, 2e-15, &(label + " lr"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Near-degenerate covariance.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_near_degenerate_rank1_lowrank() {
    assert!(file!().ends_with(".rs"));
    // K=6, rank-1 low-rank factor (very ill-conditioned Σ).
    let mut rng = SplitMix64::new(0xFEEDFACE);
    let k = 6;
    let mut f = Array2::<f64>::zeros((k, 1));
    for i in 0..k {
        f[[i, 0]] = rng.next_signed();
    }
    let cov = MarginalSlopeCovariance::LowRank(f);
    // For 50 random (q, slopes, probit_scale) we expect the exact identity
    // (no SPD floor was added on the rank-1 path).
    for seed in 0u64..50 {
        let mut rng = SplitMix64::new(0xABCD_1234 ^ seed);
        let q = 4.0 * rng.next_signed();
        let probit_scale = 0.1 + 3.0 * rng.next_unit();
        let amp = 1.0 + 5.0 * rng.next_unit();
        let slopes = random_slopes(&mut rng, k, amp);
        assert_marginal_preservation(
            q,
            &slopes,
            &cov,
            probit_scale,
            2e-15,
            &format!("rank1 lowrank seed={seed}"),
        );
    }
}

#[test]
fn survival_multi_z_near_degenerate_full_one_tiny_eigenvalue() {
    // Build Σ via eigendecomp: random orthonormal Q (Gram-Schmidt on a random
    // square matrix), eigenvalues = [1, 0.5, 0.2, 1e-10]. ε=1e-10 means
    // Σ is barely SPD; the identity still holds in exact arithmetic but
    // accumulated rounding makes 2e-15 unrealistic. We relax to 5e-13:
    // r' Σ r can be on the order of |r|^2 * 1 ≈ a few, so the relative
    // accuracy of `c` is ~ ulp(c) ≈ 1e-16, and the identity is composed of
    // two Φ evaluations whose forward error each spends ~5 ulp; 5e-13 leaves
    // headroom even when q ≈ 3 amplifies things by a Mills-ratio factor.
    let k = 4;
    let mut rng = SplitMix64::new(0xBADC0DE);
    // Random 4x4 matrix to QR for orthonormal Q.
    let mut a = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            a[[i, j]] = rng.next_signed();
        }
    }
    // Modified Gram-Schmidt.
    let mut q_mat = Array2::<f64>::zeros((k, k));
    for col in 0..k {
        let mut v: Vec<f64> = (0..k).map(|row| a[[row, col]]).collect();
        for prev in 0..col {
            let mut dot = 0.0;
            for row in 0..k {
                dot += q_mat[[row, prev]] * v[row];
            }
            for row in 0..k {
                v[row] -= dot * q_mat[[row, prev]];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-8, "QR failed: degenerate column {col}");
        for row in 0..k {
            q_mat[[row, col]] = v[row] / norm;
        }
    }
    let evals = [1.0, 0.5, 0.2, 1e-10];
    let mut sigma = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for r in 0..k {
                s += q_mat[[i, r]] * evals[r] * q_mat[[j, r]];
            }
            sigma[[i, j]] = s;
        }
    }
    // Symmetrise to kill rounding asymmetries (validate requires symmetric).
    for i in 0..k {
        for j in (i + 1)..k {
            let avg = 0.5 * (sigma[[i, j]] + sigma[[j, i]]);
            sigma[[i, j]] = avg;
            sigma[[j, i]] = avg;
        }
    }
    let cov = MarginalSlopeCovariance::Full(sigma);

    for seed in 0u64..50 {
        let mut rng = SplitMix64::new(0x1357_2468 ^ seed);
        let q = 3.0 * rng.next_signed();
        let probit_scale = 0.5 + 1.5 * rng.next_unit();
        let amp = 0.5 + 2.0 * rng.next_unit();
        let slopes = random_slopes(&mut rng, k, amp);
        assert_marginal_preservation(
            q,
            &slopes,
            &cov,
            probit_scale,
            5e-13,
            &format!("near-singular Full seed={seed}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Bitwise scalar reduction (K=1, Diagonal[1.0]).
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_k1_eta_bitwise_matches_scalar() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0]);
    for seed in 0u64..100 {
        let mut rng = SplitMix64::new(0x9999_AAAA ^ seed);
        let q = 4.0 * rng.next_signed();
        let z = 3.0 * rng.next_signed();
        let slope = rng.next_signed();
        let probit_scale = 0.05 + 2.0 * rng.next_unit();
        let eta = survival_marginal_slope_vector_eta(q, &[z], &[slope], &covariance, probit_scale)
            .expect("eta");
        let observed = probit_scale * slope;
        let scalar = q * (1.0 + observed * observed).sqrt() + observed * z;
        assert_eq!(
            eta.to_bits(),
            scalar.to_bits(),
            "seed={seed} eta={eta:.17e} scalar={scalar:.17e} (q={q} z={z} slope={slope} ps={probit_scale})"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Permutation invariance.
// ---------------------------------------------------------------------------

fn random_permutation(rng: &mut SplitMix64, k: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..k).collect();
    for i in (1..k).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        p.swap(i, j);
    }
    p
}

fn permute_vec(v: &[f64], perm: &[usize]) -> Vec<f64> {
    perm.iter().map(|&i| v[i]).collect()
}

fn permute_full(cov: &Array2<f64>, perm: &[usize]) -> Array2<f64> {
    let k = perm.len();
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            out[[i, j]] = cov[[perm[i], perm[j]]];
        }
    }
    out
}

fn permute_lowrank_rows(factor: &Array2<f64>, perm: &[usize]) -> Array2<f64> {
    let (k, r) = factor.dim();
    let mut out = Array2::<f64>::zeros((k, r));
    for i in 0..k {
        for j in 0..r {
            out[[i, j]] = factor[[perm[i], j]];
        }
    }
    out
}

#[test]
fn survival_multi_z_permutation_invariance_full_and_lowrank() {
    let k = 5;
    for seed in 0u64..50 {
        let mut rng = SplitMix64::new(0x7777_BEEF ^ seed);
        let (cov_full, dense) = random_full(&mut rng, k);
        let lr_rank = 3;
        let mut lr_factor = Array2::<f64>::zeros((k, lr_rank));
        for i in 0..k {
            for j in 0..lr_rank {
                lr_factor[[i, j]] = rng.next_signed();
            }
        }
        let cov_lr = MarginalSlopeCovariance::LowRank(lr_factor.clone());

        let amp = 1.0 + rng.next_unit();
        let slopes = random_slopes(&mut rng, k, amp);
        let z = random_z(&mut rng, k);
        let probit_scale = 0.2 + 1.5 * rng.next_unit();
        let q = 2.0 * rng.next_signed();

        let perm = random_permutation(&mut rng, k);
        let slopes_p = permute_vec(&slopes, &perm);
        let z_p = permute_vec(&z, &perm);
        let cov_full_p = MarginalSlopeCovariance::Full(permute_full(&dense, &perm));
        let cov_lr_p = MarginalSlopeCovariance::LowRank(permute_lowrank_rows(&lr_factor, &perm));

        for (label, cov, cov_p) in [
            ("Full", &cov_full, &cov_full_p),
            ("LowRank", &cov_lr, &cov_lr_p),
        ] {
            let scale_orig =
                survival_marginal_slope_vector_scale(&slopes, cov, probit_scale).expect("scale");
            let scale_perm = survival_marginal_slope_vector_scale(&slopes_p, cov_p, probit_scale)
                .expect("scale perm");
            assert!(
                (scale_orig - scale_perm).abs() <= 1e-14,
                "{label} seed={seed}: scale not invariant; orig={scale_orig:.17e} perm={scale_perm:.17e}"
            );

            let eta_orig =
                survival_marginal_slope_vector_eta(q, &z, &slopes, cov, probit_scale).expect("eta");
            let eta_perm =
                survival_marginal_slope_vector_eta(q, &z_p, &slopes_p, cov_p, probit_scale)
                    .expect("eta perm");
            assert!(
                (eta_orig - eta_perm).abs() <= 1e-14,
                "{label} seed={seed}: eta not invariant; orig={eta_orig:.17e} perm={eta_perm:.17e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Diagonal-from-Full equivalence.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_diagonal_from_full_matches() {
    let k = 6;
    for seed in 0u64..30 {
        let mut rng = SplitMix64::new(0x5555_1111 ^ seed);
        let mut diag = Array1::<f64>::zeros(k);
        for i in 0..k {
            diag[i] = 0.05 + 2.0 * rng.next_unit();
        }
        let mut dense = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            dense[[i, i]] = diag[i];
        }
        let cov_diag = MarginalSlopeCovariance::Diagonal(diag);
        let cov_full = MarginalSlopeCovariance::Full(dense);

        let slopes = random_slopes(&mut rng, k, 1.0);
        let probit_scale = 0.3 + 1.0 * rng.next_unit();

        let s_diag = survival_marginal_slope_vector_scale(&slopes, &cov_diag, probit_scale)
            .expect("diag scale");
        let s_full = survival_marginal_slope_vector_scale(&slopes, &cov_full, probit_scale)
            .expect("full scale");
        assert!(
            (s_diag - s_full).abs() <= 1e-15,
            "seed={seed}: diag={s_diag:.17e} full={s_full:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. LowRank vs Full equivalence.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_lowrank_vs_full_equivalence() {
    for seed in 0u64..50 {
        let mut rng = SplitMix64::new(0xC0FFEE ^ seed);
        let k = rng.range(2, 8); // [2,7]
        let r = rng.range(1, k + 1); // [1,k]
        let mut factor = Array2::<f64>::zeros((k, r));
        for i in 0..k {
            for j in 0..r {
                factor[[i, j]] = rng.next_signed();
            }
        }
        let cov_lr = MarginalSlopeCovariance::LowRank(factor.clone());
        let cov_full = MarginalSlopeCovariance::Full(factor.dot(&factor.t()));

        let amp = 0.5 + rng.next_unit();
        let slopes = random_slopes(&mut rng, k, amp);
        let z = random_z(&mut rng, k);
        let probit_scale = 0.2 + 1.5 * rng.next_unit();
        let q = 2.0 * rng.next_signed();

        let s_lr =
            survival_marginal_slope_vector_scale(&slopes, &cov_lr, probit_scale).expect("lr scale");
        let s_full = survival_marginal_slope_vector_scale(&slopes, &cov_full, probit_scale)
            .expect("full scale");
        assert!(
            (s_lr - s_full).abs() <= 1e-13,
            "seed={seed} K={k} r={r}: s_lr={s_lr:.17e} s_full={s_full:.17e}"
        );

        let eta_lr = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_lr, probit_scale)
            .expect("lr eta");
        let eta_full = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_full, probit_scale)
            .expect("full eta");
        assert!(
            (eta_lr - eta_full).abs() <= 1e-13,
            "seed={seed} K={k} r={r}: eta_lr={eta_lr:.17e} eta_full={eta_full:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Extra: smoke-test the canonical builder so it is still exercised here.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_covariance_from_scores_smoke() {
    let k = 3;
    let n = 50;
    let mut rng = SplitMix64::new(0xA5A5_5A5A);
    let mut scores = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            scores[[i, j]] = rng.next_normal();
        }
    }
    let weights = Array1::<f64>::from_elem(n, 1.0);
    let cov = marginal_slope_covariance_from_scores(scores.view(), &weights)
        .expect("covariance from scores");
    // Must validate cleanly. The classifier may legitimately return
    // Diagonal for IID-normal scores when sample off-diagonals are within
    // the statistical-noise threshold; the invariant is marginal preservation.
    let slopes = random_slopes(&mut rng, k, 1.0);
    assert_marginal_preservation(0.7, &slopes, &cov, 1.0, 2e-15, "scores-derived");
    assert_eq!(cov.dim(), k);
}

// ---------------------------------------------------------------------------
// Negative tests.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_eta_rejects_z_slope_dimension_mismatch() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0]);
    // z and covariance agree (len 2), but slopes is length 3.
    let err =
        survival_marginal_slope_vector_eta(0.2, &[0.4, -0.8], &[0.3, 0.1, -0.2], &covariance, 1.0)
            .expect_err("dimension mismatch must fail");
    assert!(
        err.to_lowercase().contains("dimension mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn survival_multi_z_eta_rejects_z_covariance_dimension_mismatch() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0, 1.0]);
    let err = survival_marginal_slope_vector_eta(0.1, &[0.4, 0.5], &[0.3, 0.2], &covariance, 1.0)
        .expect_err("z/cov mismatch must fail");
    assert!(
        err.to_lowercase().contains("dimension mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn survival_multi_z_scale_rejects_nonfinite_slope() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0]);
    let err = survival_marginal_slope_vector_scale(&[0.3, f64::NAN], &covariance, 1.0)
        .expect_err("non-finite slope must fail");
    assert!(!err.is_empty(), "expected non-empty error, got: {err}");
}

#[test]
fn survival_multi_z_eta_rejects_nonfinite_z() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0]);
    let err = survival_marginal_slope_vector_eta(
        0.1,
        &[0.4, f64::INFINITY],
        &[0.3, 0.2],
        &covariance,
        1.0,
    )
    .expect_err("non-finite z must fail");
    assert!(!err.is_empty(), "expected non-empty error, got: {err}");
}

// ---------------------------------------------------------------------------
// Spot check: neglog stays finite over many random shapes (just guards
// against panics in the negative-log path under the marginal-preserving
// scale). Identity here is just that no `Err` is returned and the value is
// finite for sensible event=0 inputs.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_neglog_finite_under_random_shapes() {
    for seed in 0u64..40 {
        let mut rng = SplitMix64::new(0x4242_4242 ^ seed);
        let k = rng.range(2, 6);
        let cov = match seed % 3 {
            0 => random_diagonal(&mut rng, k),
            1 => random_full(&mut rng, k).0,
            _ => {
                let r = (rng.range(1, k + 1)).max(1);
                random_low_rank(&mut rng, k, r)
            }
        };
        let slopes = random_slopes(&mut rng, k, 0.5);
        let z = random_z(&mut rng, k);
        let q0 = rng.next_signed();
        let q1 = q0 + 0.5 + rng.next_unit();
        let qd1 = 0.1 + rng.next_unit();
        let value = survival_marginal_slope_vector_neglog(
            q0, q1, qd1, &slopes, &z, &cov, 1.0, 0.0, 1e-6, 1.0,
        )
        .expect("neglog");
        assert!(value.is_finite(), "seed={seed}: neglog not finite: {value}");
    }
}
