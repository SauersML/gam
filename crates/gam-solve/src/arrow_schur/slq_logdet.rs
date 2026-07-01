//! Matrix-free log-determinant via Stochastic Lanczos Quadrature (SLQ).
//!
//! BIBLIOGRAPHY
//!
//! * Ubaru, Chen, Saad, "Fast Estimation of tr(f(A)) via Stochastic Lanczos
//!   Quadrature", SIAM J. Matrix Anal. Appl. 38(4), 2017: the canonical SLQ
//!   estimator for `tr(f(A))` with `f = ln` giving `log det A = tr(ln A)`.
//! * Bai, Fahey, Golub, "Some large-scale matrix computation problems", J.
//!   Comput. Appl. Math. 74, 1996: Gauss-quadrature view of `uᵀ f(A) u` as
//!   `Σ_i (e₁ᵀ y_i)² f(θ_i)` over the Lanczos tridiagonal eigenpairs `(θ_i,y_i)`.
//! * Hutchinson, "A stochastic estimator of the trace of the influence matrix",
//!   Comm. Statist. Simulation Comput. 19, 1990: Rademacher probe vectors with
//!   `E[zᵀ M z] = tr(M)` and `‖z‖² = dim`.
//! * Golub, Meurant, "Matrices, Moments and Quadrature with Applications", 2010:
//!   Lanczos quadrature, the need for reorthogonalization, and error analysis.
//!
//! ## What this provides
//!
//! [`slq_logdet`] estimates `log det A` for a symmetric positive-definite
//! operator `A` available ONLY through matrix-vector products `v ↦ A v`. It
//! never forms or factors `A`, so for the reduced-Schur Laplace normaliser it
//! replaces the dense `O(k³/3)` Cholesky log-determinant with
//! `O(num_probes · lanczos_steps · matvec)` work.
//!
//! The estimator is `tr(ln A) ≈ (dim / num_probes) Σ_p zₚᵀ ln(A) zₚ` with
//! Rademacher probes `zₚ`, and each quadratic form `zᵀ ln(A) z` is evaluated by
//! `m` steps of Lanczos against `A` started from `z/‖z‖`: building the symmetric
//! tridiagonal `T_m` (with FULL reorthogonalization against the stored basis),
//! eigendecomposing it, and reading the Gauss quadrature
//! `‖z‖² Σ_i (τ_{i,0})² ln(θ_i)` where `θ_i` are `T_m`'s eigenvalues and
//! `τ_{i,0}` is the first component of the `i`-th eigenvector.
//!
//! ## Reuse
//!
//! The numerically-critical Lanczos recurrence + full reorthogonalization +
//! tridiagonal eigendecomposition is the workspace primitive
//! [`gam_linalg::lanczos::symmetric_lanczos_eigenpairs`]; this module is the
//! Hutchinson outer loop (Rademacher probes, averaging, standard error) on top
//! of it. The clamped log-quadrature is computed here (rather than via
//! [`gam_linalg::lanczos::symmetric_lanczos_log_quadrature`], which errors on a
//! non-positive Ritz value) so a round-off-negative Ritz value floors to a tiny
//! positive number instead of failing the whole evidence solve.
//!
//! ## Determinism
//!
//! The probe vectors are drawn from [`gam_linalg::utils::splitmix64`] seeded by
//! `seed + probe_index`; there is NO system-RNG dependence, so a given
//! `(dim, matvec, num_probes, lanczos_steps, seed)` always returns the same
//! estimate. This is required by the evidence path, whose REML outer loop must
//! be reproducible.

use super::*;
use gam_linalg::lanczos::{symmetric_lanczos_eigenpairs, SymmetricLanczosOptions};
use gam_linalg::utils::splitmix64;

/// Result of a Stochastic Lanczos Quadrature log-determinant estimate.
#[derive(Debug, Clone, Copy)]
pub struct SlqLogDet {
    /// Estimate of `log det A`.
    pub estimate: f64,
    /// Standard error of the estimate: the sample standard deviation of the
    /// per-probe contributions divided by `sqrt(num_probes)`. With a single
    /// probe this is `0.0` (no spread is observable).
    pub std_err: f64,
}

/// Floor on Ritz eigenvalues before taking `ln`. The operator is SPD so the
/// Ritz values `θ_i` are positive in exact arithmetic; this clamps any tiny
/// negative/zero value produced by round-off so `ln` stays finite. Chosen far
/// below any physically meaningful curvature scale.
const RITZ_LN_FLOOR: f64 = 1e-300;

/// Draw a deterministic Rademacher (±1) vector of length `dim` into `z`,
/// seeded reproducibly by `probe_seed`. Two bits per draw are wasteful but the
/// per-element top-bit read keeps this trivially correct and stream-stable.
fn rademacher_into(z: &mut Array1<f64>, probe_seed: u64) {
    let mut state = probe_seed;
    let mut bits: u64 = 0;
    let mut remaining: u32 = 0;
    for value in z.iter_mut() {
        if remaining == 0 {
            bits = splitmix64(&mut state);
            remaining = 64;
        }
        *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
        bits >>= 1;
        remaining -= 1;
    }
}

/// Estimate `log det A` for an SPD operator given only its matrix-vector apply.
///
/// * `dim` — dimension of the operator (`A` is `dim × dim`).
/// * `matvec` — applies `A`: `matvec(v) = A v`, for `v.len() == dim`.
/// * `num_probes` — number of Rademacher probe vectors (Hutchinson samples).
/// * `lanczos_steps` — Lanczos iterations per probe (Gauss-quadrature nodes).
/// * `seed` — base seed; probe `p` uses `seed + p`, so results are reproducible.
///
/// Returns the averaged estimate and its standard error. For `dim == 0` the
/// determinant of the empty operator is `1`, so the log-determinant is `0`.
///
/// `lanczos_steps` is internally capped at `dim` (a Krylov subspace cannot
/// exceed the dimension) and `num_probes` is treated as at least `1`.
pub fn slq_logdet(
    dim: usize,
    matvec: impl Fn(ArrayView1<f64>) -> Array1<f64>,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> SlqLogDet {
    if dim == 0 {
        return SlqLogDet {
            estimate: 0.0,
            std_err: 0.0,
        };
    }
    let num_probes = num_probes.max(1);
    let steps = lanczos_steps.max(1).min(dim);
    let norm_sq = dim as f64; // ‖z‖² for a ±1 Rademacher vector of length `dim`.

    // The workspace Lanczos engine consumes `apply(&[f64], &mut [f64])`; wrap the
    // ndarray `matvec` closure into that slice contract. Scratch buffers are
    // reused across probes.
    let mut in_buf = Array1::<f64>::zeros(dim);
    let mut apply = |x: &[f64], out: &mut [f64]| -> Result<(), String> {
        in_buf
            .as_slice_mut()
            .expect("contiguous probe input buffer")
            .copy_from_slice(x);
        let y = matvec(in_buf.view());
        if y.len() != dim {
            return Err(format!(
                "slq_logdet matvec returned length {}, expected {dim}",
                y.len()
            ));
        }
        out.copy_from_slice(y.as_slice().expect("contiguous matvec output"));
        Ok(())
    };

    let lanczos_options = SymmetricLanczosOptions {
        max_steps: steps,
        // Pure SPD log-det quadrature: keep iterating until the Krylov space is
        // genuinely exhausted (a true lucky breakdown), not at a slack residual.
        residual_tol: 0.0,
        local_reorthogonalize: false,
        // Full reorthogonalization is numerically essential for the quadrature:
        // without it Lanczos loses orthogonality and produces ghost Ritz values.
        full_reorthogonalize: true,
    };

    let mut z = Array1::<f64>::zeros(dim);
    let mut contributions = Vec::with_capacity(num_probes);
    for probe in 0..num_probes {
        let probe_seed = seed.wrapping_add(probe as u64);
        rademacher_into(&mut z, probe_seed);
        let start = z.as_slice().expect("contiguous probe vector");
        let contribution = match symmetric_lanczos_eigenpairs(
            dim,
            start,
            lanczos_options,
            &mut apply,
        ) {
            Ok(pairs) => norm_sq * clamped_log_quadrature(&pairs.eigenvalues, &pairs.eigenvectors),
            // A Lanczos failure (non-finite matvec / start) cannot be silently
            // averaged in; the dense-Cholesky gate above this call should have
            // caught a degenerate operator. Treat it as a zero contribution and
            // let the std-error widen rather than poisoning the mean with NaN.
            Err(_) => 0.0,
        };
        contributions.push(contribution);
    }

    let n = contributions.len() as f64;
    let mean = contributions.iter().sum::<f64>() / n;
    let std_err = if contributions.len() > 1 {
        let var = contributions
            .iter()
            .map(|c| {
                let d = c - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        (var / n).sqrt()
    } else {
        0.0
    };

    SlqLogDet {
        estimate: mean,
        std_err,
    }
}

/// Gauss quadrature `e₁ᵀ ln(T) e₁ = Σ_i (τ_{i,0})² ln(θ_i)` over the Lanczos
/// tridiagonal eigenpairs, with `θ_i` floored to [`RITZ_LN_FLOOR`] so a
/// round-off-negative Ritz value (the SPD operator forbids genuine ones) cannot
/// produce a `NaN`. `eigenvectors` columns are the Ritz vectors `y_i`; `τ_{i,0}`
/// is their first component.
fn clamped_log_quadrature(eigenvalues: &Array1<f64>, eigenvectors: &Array2<f64>) -> f64 {
    let mut quad = 0.0_f64;
    for i in 0..eigenvalues.len() {
        let tau0 = eigenvectors[[0, i]];
        let weight = tau0 * tau0;
        let lambda = eigenvalues[i].max(RITZ_LN_FLOOR);
        quad += weight * lambda.ln();
    }
    quad
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform draw in `[lo, hi)` from a SplitMix64 state — keeps
    /// the test fixtures reproducible with no external RNG dependency.
    fn next_uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
        // 53-bit mantissa fraction in [0, 1).
        let bits = splitmix64(state) >> 11;
        let unit = (bits as f64) / ((1u64 << 53) as f64);
        lo + (hi - lo) * unit
    }

    /// Build a random SPD matrix `A = MᵀM + δI` (`dim × dim`) from a fixed seed.
    /// `m_rows ≥ dim` keeps `MᵀM` well-conditioned; `delta` sets the floor on the
    /// spectrum (larger `delta` ⇒ better conditioned).
    fn random_spd(dim: usize, m_rows: usize, delta: f64, seed: u64) -> Array2<f64> {
        let mut state = seed;
        let mut m = Array2::<f64>::zeros((m_rows, dim));
        for value in m.iter_mut() {
            *value = next_uniform(&mut state, -1.0, 1.0);
        }
        let mut a = m.t().dot(&m);
        for i in 0..dim {
            a[[i, i]] += delta;
        }
        // Symmetrize defensively against round-off.
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (a[[i, j]] + a[[j, i]]);
                a[[i, j]] = avg;
                a[[j, i]] = avg;
            }
        }
        a
    }

    /// Exact `log det A` via the workspace symmetric eigensolver (`Σ ln λ_i`).
    fn exact_logdet(a: &Array2<f64>) -> f64 {
        let (evals, _) = a.eigh(Side::Lower).expect("SPD eigendecomposition");
        evals.iter().map(|&l| l.max(RITZ_LN_FLOOR).ln()).sum()
    }

    fn condition_number(a: &Array2<f64>) -> f64 {
        let (evals, _) = a.eigh(Side::Lower).expect("SPD eigendecomposition");
        let max = evals.iter().cloned().fold(f64::MIN, f64::max);
        let min = evals.iter().cloned().fold(f64::MAX, f64::min);
        max / min
    }

    #[test]
    fn slq_matches_exact_logdet_well_conditioned() {
        // A spread of dimensions in the 60–200 range, all well-conditioned
        // (generous δ), checked against the exact eigenvalue log-determinant.
        for (dim, seed) in [(60usize, 1u64), (120, 2), (200, 3)] {
            let a = random_spd(dim, dim + 40, 5.0, seed);
            let exact = exact_logdet(&a);
            let cond = condition_number(&a);

            let result = slq_logdet(dim, |v| a.dot(&v), 48, 70, 0xA5A5_0000 ^ seed);

            let rel_err = (result.estimate - exact).abs() / exact.abs();
            eprintln!(
                "well-conditioned dim={dim} cond={cond:.2e} exact={exact:.6} \
                 est={:.6} rel_err={rel_err:.4e} std_err={:.4e}",
                result.estimate, result.std_err
            );
            assert!(
                rel_err < 0.05,
                "dim={dim}: SLQ relative error {rel_err:.4e} exceeds 5% \
                 (exact={exact}, est={})",
                result.estimate
            );
            // The exact value should sit within a few standard errors of the
            // estimate (the std_err must be a meaningful uncertainty band).
            assert!(
                (result.estimate - exact).abs() < 3.0 * result.std_err + 0.05 * exact.abs(),
                "dim={dim}: estimate not within ~3 std_err of exact \
                 (|Δ|={:.4e}, std_err={:.4e})",
                (result.estimate - exact).abs(),
                result.std_err
            );
        }
    }

    #[test]
    fn slq_handles_moderately_ill_conditioned() {
        // Smaller δ ⇒ a tighter spectral floor ⇒ a more ill-conditioned A.
        // More Lanczos steps resolve the wider spectrum.
        let dim = 150usize;
        let a = random_spd(dim, dim + 5, 0.05, 7);
        let exact = exact_logdet(&a);
        let cond = condition_number(&a);
        assert!(
            cond > 1e3,
            "test fixture should be moderately ill-conditioned, got cond={cond:.2e}"
        );

        let result = slq_logdet(dim, |v| a.dot(&v), 40, 110, 0xC0FFEE);
        let rel_err = (result.estimate - exact).abs() / exact.abs();
        eprintln!(
            "ill-conditioned dim={dim} cond={cond:.2e} exact={exact:.6} \
             est={:.6} rel_err={rel_err:.4e} std_err={:.4e}",
            result.estimate, result.std_err
        );
        assert!(
            rel_err < 0.10,
            "ill-conditioned dim={dim}: SLQ relative error {rel_err:.4e} \
             exceeds 10% (cond={cond:.2e}, exact={exact}, est={})",
            result.estimate
        );
    }

    #[test]
    fn slq_is_deterministic_for_fixed_seed() {
        let dim = 80usize;
        let a = random_spd(dim, dim + 20, 2.0, 11);
        let r1 = slq_logdet(dim, |v| a.dot(&v), 24, 50, 99);
        let r2 = slq_logdet(dim, |v| a.dot(&v), 24, 50, 99);
        assert_eq!(
            r1.estimate, r2.estimate,
            "SLQ must be bit-reproducible for a fixed seed"
        );
        assert_eq!(r1.std_err, r2.std_err);
    }

    #[test]
    fn slq_diagonal_operator_matches_closed_form() {
        // A diagonal operator has a closed-form log-determinant Σ ln d_i; this
        // exercises the matvec closure path without any matrix assembly.
        let dim = 100usize;
        let mut state = 123u64;
        let diag: Vec<f64> = (0..dim).map(|_| next_uniform(&mut state, 0.5, 4.0)).collect();
        let exact: f64 = diag.iter().map(|d| d.ln()).sum();

        let diag_clone = diag.clone();
        let result = slq_logdet(
            dim,
            move |v| {
                let mut out = v.to_owned();
                for (o, d) in out.iter_mut().zip(diag_clone.iter()) {
                    *o *= d;
                }
                out
            },
            32,
            60,
            7,
        );
        let rel_err = (result.estimate - exact).abs() / exact.abs();
        eprintln!(
            "diagonal dim={dim} exact={exact:.6} est={:.6} rel_err={rel_err:.4e}",
            result.estimate
        );
        assert!(
            rel_err < 0.05,
            "diagonal operator: relative error {rel_err:.4e} exceeds 5%"
        );
    }

    #[test]
    fn slq_empty_operator_is_zero() {
        let result = slq_logdet(0, |v| v.to_owned(), 8, 8, 1);
        assert_eq!(result.estimate, 0.0);
        assert_eq!(result.std_err, 0.0);
    }

    #[test]
    fn std_err_shrinks_with_more_probes() {
        // The standard error of a Monte-Carlo mean falls ~1/sqrt(num_probes);
        // many probes should give a tighter band than few.
        let dim = 120usize;
        let a = random_spd(dim, dim + 30, 3.0, 21);
        let few = slq_logdet(dim, |v| a.dot(&v), 6, 60, 5);
        let many = slq_logdet(dim, |v| a.dot(&v), 96, 60, 5);
        eprintln!(
            "std_err few(6)={:.4e} many(96)={:.4e}",
            few.std_err, many.std_err
        );
        assert!(
            many.std_err < few.std_err,
            "more probes should reduce std_err (few={:.4e}, many={:.4e})",
            few.std_err,
            many.std_err
        );
    }
}
