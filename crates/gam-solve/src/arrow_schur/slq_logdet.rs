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
use gam_linalg::lanczos::{
    SymmetricLanczosEigenpairs, SymmetricLanczosOptions, symmetric_lanczos_eigenpairs,
};
use gam_linalg::utils::splitmix64;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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

/// The fixed Lanczos configuration every SLQ probe runs: `steps` Gauss nodes,
/// no early residual break (exhaust the Krylov space), and FULL
/// reorthogonalization (numerically essential — without it Lanczos loses
/// orthogonality and produces ghost Ritz values that poison the quadrature).
#[inline]
fn slq_lanczos_options(steps: usize) -> SymmetricLanczosOptions {
    SymmetricLanczosOptions {
        max_steps: steps,
        residual_tol: 0.0,
        local_reorthogonalize: false,
        full_reorthogonalize: true,
    }
}

/// Run one Rademacher-probe Lanczos and return the tridiagonal eigenpairs, or
/// `None` if the Lanczos run declines (non-finite matvec / start). Shared by the
/// plain [`slq_logdet`] and the unit-deflated [`slq_logdet_unit_deflated`]
/// estimators so both draw the IDENTICAL probe vector and build the IDENTICAL
/// Krylov space for a given `(dim, matvec, probe_seed, options)` — the two
/// estimators then differ ONLY in the spectral function applied to the shared
/// Ritz pairs. Each probe carries its own Rademacher vector and matvec input
/// scratch (no shared mutable state), so a `rayon` fan-out over probes is
/// bit-identical to the serial build.
fn probe_lanczos_eigenpairs(
    dim: usize,
    probe_seed: u64,
    options: SymmetricLanczosOptions,
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
) -> Option<SymmetricLanczosEigenpairs> {
    let mut z = Array1::<f64>::zeros(dim);
    rademacher_into(&mut z, probe_seed);
    // The workspace Lanczos engine consumes `apply(&[f64], &mut [f64])`; wrap the
    // ndarray `matvec` into that slice contract with a per-probe input buffer so
    // probes never share mutable scratch.
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
    let start = z.as_slice().expect("contiguous probe vector");
    symmetric_lanczos_eigenpairs(dim, start, options, &mut apply).ok()
}

/// Serial mean and standard error of the per-probe SLQ contributions. Runs over
/// the `into_par_iter().collect()` ORDERED buffer so the reduction is bit-for-bit
/// reproducible for a fixed probe set — the determinism the REML evidence outer
/// loop requires.
fn slq_mean_std_err(contributions: &[f64]) -> (f64, f64) {
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
    (mean, std_err)
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
    matvec: impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync,
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
    let lanczos_options = slq_lanczos_options(steps);

    // Each Hutchinson probe is a FULLY INDEPENDENT Lanczos run against the same
    // read-only (`Sync`) operator, so at the K=32k evidence scale — where SLQ
    // fires precisely because the operator is large (`num_probes`×`lanczos_steps`
    // matvecs of an `O(k²)` apply) — the probes fan out across rayon workers for
    // a near-`num_probes`× wall-clock cut on the dominant matvec work. The
    // contribution a probe computes depends only on `(dim, matvec, probe_seed,
    // options)`, so it is bit-identical to the serial build.
    // `into_par_iter().collect()` preserves probe order, and the mean/std-err
    // reduction runs SERIALLY over that ordered buffer, so the estimate and
    // std-error are bit-for-bit reproducible for a fixed `(dim, matvec,
    // num_probes, lanczos_steps, seed)` — the determinism the REML evidence outer
    // loop requires (see the module `Determinism` note).
    let matvec = &matvec;
    let contributions: Vec<f64> = (0..num_probes)
        .into_par_iter()
        .map(|probe| {
            let probe_seed = seed.wrapping_add(probe as u64);
            match probe_lanczos_eigenpairs(dim, probe_seed, lanczos_options, matvec) {
                Some(pairs) => {
                    norm_sq * clamped_log_quadrature(&pairs.eigenvalues, &pairs.eigenvectors)
                }
                // A Lanczos failure (non-finite matvec / start) cannot be silently
                // averaged in; the dense-Cholesky gate above this call should have
                // caught a degenerate operator. Treat it as a zero contribution and
                // let the std-error widen rather than poisoning the mean with NaN.
                None => 0.0,
            }
        })
        .collect();

    let (estimate, std_err) = slq_mean_std_err(&contributions);
    SlqLogDet { estimate, std_err }
}

/// Result of a unit-deflated SLQ log-determinant estimate.
///
/// The estimate is `tr(φ(A))` with the unit-deflation spectral function
/// `φ(θ) = θ ≥ deflate_floor ? ln θ : 0`, i.e. every eigenvalue at or below the
/// floor is pinned to unit stiffness and contributes `ln 1 = 0` — the exact
/// matrix-free analogue of the dense
/// [`ReducedSchurPolicy::EvidenceUnitDeflation`](super::reduced_solve) convention
/// (see #2308). `lambda_max_abs` and `deflate_floor` are reported so callers can
/// audit the scale at which deflation kicked in.
#[derive(Debug, Clone, Copy)]
pub struct SlqUnitDeflatedLogDet {
    /// Estimate of the unit-deflated log-determinant `Σ_{λ ≥ floor} ln λ`.
    pub estimate: f64,
    /// Standard error of the estimate across probes (`0.0` for a single probe).
    pub std_err: f64,
    /// Estimated spectral radius `max|λ|` — the largest `|Ritz value|` observed
    /// across every probe. The deflation floor is relative to this scale, so it
    /// is the matrix-free stand-in for the dense path's `max|λ|`.
    pub lambda_max_abs: f64,
    /// The absolute deflation floor actually applied:
    /// `relative_floor · lambda_max_abs · (1 − hysteresis)`.
    pub deflate_floor: f64,
}

impl SlqUnitDeflatedLogDet {
    /// View as a plain [`SlqLogDet`] (estimate + std-error), dropping the
    /// deflation metadata — for the evidence plumbing that only consumes the
    /// scalar log-determinant and its uncertainty band.
    #[inline]
    pub fn as_logdet(&self) -> SlqLogDet {
        SlqLogDet {
            estimate: self.estimate,
            std_err: self.std_err,
        }
    }
}

/// Estimate the UNIT-DEFLATED log-determinant of a symmetric operator `A`
/// available only through matvecs — the matrix-free counterpart of the dense
/// `EvidenceUnitDeflation` reduced-Schur policy (#2308).
///
/// SLQ estimates `tr(f(A))` for ANY spectral function `f` via the same Gauss
/// quadrature; the plain [`slq_logdet`] uses `f = ln`. Unit deflation is simply
/// a DIFFERENT `f`:
///
/// ```text
///   φ(θ) = ln θ,  θ ≥ deflate_floor
///   φ(θ) = 0,     θ < deflate_floor   (pinned to unit stiffness: ln 1 = 0)
/// ```
///
/// so `tr(φ(A)) = Σ_{λ_i ≥ floor} ln λ_i` — every collapsed / near-null / (round-off
/// or genuinely) negative-curvature direction contributes exactly `0` instead of
/// the plain estimator's `ln(RITZ_LN_FLOOR) ≈ −690` per deflated direction. This
/// matches the dense convention where a sub-floor eigenvalue is pinned to `λ̃ = 1`.
///
/// The floor is RELATIVE, exactly as in the dense path:
/// `deflate_floor = relative_floor · max|λ| · (1 − SPECTRAL_DEFLATION_HYSTERESIS_FRACTION)`,
/// with `max|λ|` estimated as the largest `|Ritz value|` over all probes — Lanczos
/// converges to the extreme eigenvalues first, so this is a sharp, deterministic
/// spectral-radius estimate. A single shared floor is computed BEFORE any
/// contribution so every probe deflates against the SAME threshold (a per-probe
/// floor would make the estimate non-linear in the probes and break determinism
/// of the deflation set).
///
/// Determinism, probe fan-out, and the `dim == 0 ⇒ 0` convention are identical to
/// [`slq_logdet`]; the two share [`probe_lanczos_eigenpairs`], so for a fixed
/// `(dim, matvec, num_probes, lanczos_steps, seed)` the two estimators build
/// bit-identical Krylov spaces and differ ONLY in the applied spectral function.
pub fn slq_logdet_unit_deflated(
    dim: usize,
    matvec: impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
    relative_floor: f64,
) -> SlqUnitDeflatedLogDet {
    if dim == 0 {
        return SlqUnitDeflatedLogDet {
            estimate: 0.0,
            std_err: 0.0,
            lambda_max_abs: 0.0,
            deflate_floor: 0.0,
        };
    }
    let num_probes = num_probes.max(1);
    let steps = lanczos_steps.max(1).min(dim);
    let norm_sq = dim as f64;
    let lanczos_options = slq_lanczos_options(steps);

    // Pass 1 — build every probe's Ritz pairs (the expensive matvec work), fanned
    // across rayon workers. The ordered buffer keeps the reduction reproducible.
    let matvec = &matvec;
    let per_probe: Vec<Option<SymmetricLanczosEigenpairs>> = (0..num_probes)
        .into_par_iter()
        .map(|probe| {
            let probe_seed = seed.wrapping_add(probe as u64);
            probe_lanczos_eigenpairs(dim, probe_seed, lanczos_options, matvec)
        })
        .collect();

    // A single shared spectral-radius estimate `max|λ|` over ALL probes' Ritz
    // values, and from it the ONE deflation floor every probe uses.
    let lambda_max_abs = per_probe
        .iter()
        .flatten()
        .flat_map(|pairs| pairs.eigenvalues.iter())
        .filter(|value| value.is_finite())
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if !(lambda_max_abs.is_finite() && lambda_max_abs > 0.0) {
        // No usable spectrum (empty / all-zero / every probe declined): the
        // unit-deflated determinant of a fully-deflated operator is `Σ 0 = 0`.
        return SlqUnitDeflatedLogDet {
            estimate: 0.0,
            std_err: 0.0,
            lambda_max_abs: 0.0,
            deflate_floor: 0.0,
        };
    }
    let deflate_floor =
        relative_floor * lambda_max_abs * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);

    // Pass 2 — apply the unit-deflation spectral function against the shared floor.
    let contributions: Vec<f64> = per_probe
        .iter()
        .map(|maybe_pairs| match maybe_pairs {
            Some(pairs) => {
                norm_sq
                    * deflated_log_quadrature(
                        &pairs.eigenvalues,
                        &pairs.eigenvectors,
                        deflate_floor,
                    )
            }
            None => 0.0,
        })
        .collect();

    let (estimate, std_err) = slq_mean_std_err(&contributions);
    SlqUnitDeflatedLogDet {
        estimate,
        std_err,
        lambda_max_abs,
        deflate_floor,
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

/// Unit-deflated Gauss quadrature `Σ_i (τ_{i,0})² φ(θ_i)` with the deflation
/// spectral function `φ(θ) = θ ≥ deflate_floor ? ln θ : 0`. A Ritz value at or
/// below the floor (collapsed / near-null / round-off- or genuinely-negative)
/// is pinned to unit stiffness and contributes `ln 1 = 0`; a kept Ritz value
/// (necessarily `> deflate_floor > 0`) contributes `ln θ`. This is the
/// matrix-free image of the dense evidence unit deflation (#2308).
fn deflated_log_quadrature(
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    deflate_floor: f64,
) -> f64 {
    let mut quad = 0.0_f64;
    for i in 0..eigenvalues.len() {
        let lambda = eigenvalues[i];
        if lambda < deflate_floor {
            // Deflated direction: pinned to λ̃ = 1, contributes ln 1 = 0.
            continue;
        }
        let tau0 = eigenvectors[[0, i]];
        let weight = tau0 * tau0;
        // `deflate_floor > 0`, so a kept Ritz value is strictly positive; the
        // `RITZ_LN_FLOOR` guard only defends against a round-off boundary case.
        quad += weight * lambda.max(RITZ_LN_FLOOR).ln();
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
        let diag: Vec<f64> = (0..dim)
            .map(|_| next_uniform(&mut state, 0.5, 4.0))
            .collect();
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

    /// Dense symmetric `A = H diag(λ) H` with `H = I − 2wwᵀ` (‖w‖=1) a Householder
    /// reflector — orthogonal AND symmetric, so `A`'s eigenvalues are EXACTLY the
    /// planted `λ` and its eigenvectors are the columns of `H`. Gives a genuinely
    /// non-diagonal operator with a known, hand-chosen spectrum (unlike
    /// `random_spd`, whose spectrum would have to be eigendecomposed to learn),
    /// so a deflation test can plant a specific collapsed direction.
    fn householder_spectrum_matrix(eigenvalues: &[f64], seed: u64) -> Array2<f64> {
        let dim = eigenvalues.len();
        let mut state = seed;
        let mut w = Array1::<f64>::zeros(dim);
        for value in w.iter_mut() {
            *value = next_uniform(&mut state, -1.0, 1.0);
        }
        let norm = w.dot(&w).sqrt();
        w.mapv_inplace(|v| v / norm);
        // H = I − 2 w wᵀ.
        let mut h = Array2::<f64>::eye(dim);
        for i in 0..dim {
            for j in 0..dim {
                h[[i, j]] -= 2.0 * w[i] * w[j];
            }
        }
        // A = (H D) H, with D = diag(λ). H is symmetric, so A = H D Hᵀ is symmetric
        // with eigenpairs (λ_j, H[:, j]).
        let mut hd = h.clone();
        for j in 0..dim {
            for i in 0..dim {
                hd[[i, j]] *= eigenvalues[j];
            }
        }
        hd.dot(&h)
    }

    /// #2308 — the matrix-free evidence log|S| MUST obey the same unit-deflation
    /// convention as the dense reduced-Schur factor: a collapsed / near-null /
    /// negative-curvature direction is pinned to unit stiffness and contributes
    /// `ln 1 = 0`, NOT the plain estimator's `ln(RITZ_LN_FLOOR) ≈ −690`.
    #[test]
    fn slq_unit_deflation_pins_collapsed_direction_to_unit_2308() {
        let dim = 48usize;
        let mut state = 0x2308_0001_u64;
        let mut eigenvalues = vec![0.0_f64; dim];
        for e in eigenvalues.iter_mut() {
            *e = next_uniform(&mut state, 0.5, 12.0);
        }
        // One collapsed direction: genuinely negative curvature, |λ| ≪ floor —
        // exactly the collapsed-decoder mode the evidence deflation targets.
        eigenvalues[dim - 1] = -3.0e-11;

        let a = householder_spectrum_matrix(&eigenvalues, 0x51A9);
        let max_abs = eigenvalues.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        // The dense convention's floor and the kept-eigenvalue reference log-det.
        let floor = SPECTRAL_DEFLATION_REL_FLOOR * max_abs
            * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
        let reference: f64 = eigenvalues
            .iter()
            .filter(|&&l| l >= floor)
            .map(|&l| l.ln())
            .sum();

        let deflated = slq_logdet_unit_deflated(
            dim,
            |v| a.dot(&v),
            48,
            dim,
            0xD1F,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        eprintln!(
            "unit-deflated est={:.6} reference={:.6} lambda_max_abs={:.6} floor={:.3e}",
            deflated.estimate, reference, deflated.lambda_max_abs, deflated.deflate_floor
        );
        // The spectral-radius estimate recovers max|λ|, and the floor is the dense
        // path's floor built from it.
        assert!(
            (deflated.lambda_max_abs - max_abs).abs() / max_abs < 1e-6,
            "lambda_max_abs {} should recover planted max|λ| {}",
            deflated.lambda_max_abs,
            max_abs
        );
        assert!(
            (deflated.deflate_floor - floor).abs() / floor < 1e-9,
            "deflate_floor {} should equal the dense relative floor {}",
            deflated.deflate_floor,
            floor
        );
        let rel = (deflated.estimate - reference).abs() / reference.abs().max(1.0);
        assert!(
            rel < 0.02,
            "unit-deflated SLQ {} must match the kept-eigenvalue reference {} (rel {rel:.3e})",
            deflated.estimate,
            reference
        );

        // Plain SLQ (no deflation) is dragged HUNDREDS of units below by the
        // collapsed direction's `ln(RITZ_LN_FLOOR)` contribution — the exact
        // ρ-dependent-Occam-reward bug the matrix-free unit deflation removes.
        let plain = slq_logdet(dim, |v| a.dot(&v), 48, dim, 0xD1F);
        eprintln!("plain est={:.6}", plain.estimate);
        assert!(
            deflated.estimate - plain.estimate > 100.0,
            "plain SLQ ({}) must sit far below the unit-deflated estimate ({})",
            plain.estimate,
            deflated.estimate
        );
    }

    /// #2308 — with NO sub-floor direction, unit deflation deflates nothing, so it
    /// is bit-identical to the plain estimator (same probe stream, every Ritz value
    /// kept) and equally close to the exact log-determinant.
    #[test]
    fn slq_unit_deflation_matches_plain_when_no_nulls_2308() {
        let dim = 120usize;
        let a = random_spd(dim, dim + 40, 5.0, 3);
        let exact = exact_logdet(&a);
        let deflated = slq_logdet_unit_deflated(
            dim,
            |v| a.dot(&v),
            48,
            70,
            0xA5A5,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        let plain = slq_logdet(dim, |v| a.dot(&v), 48, 70, 0xA5A5);
        assert_eq!(
            deflated.estimate.to_bits(),
            plain.estimate.to_bits(),
            "no deflation ⇒ unit-deflated estimate must be bit-identical to plain"
        );
        let rel = (deflated.estimate - exact).abs() / exact.abs();
        assert!(
            rel < 0.05,
            "unit-deflated SLQ rel err {rel:.3e} vs exact {exact}"
        );
    }

    /// #2308 — degenerate operators: the empty and the fully-collapsed (`A = 0`)
    /// operator both have a finite unit-deflated log-det of `0` (every direction
    /// pinned to unit), never `−∞`.
    #[test]
    fn slq_unit_deflation_empty_and_degenerate_2308() {
        let empty = slq_logdet_unit_deflated(
            0,
            |v| v.to_owned(),
            8,
            8,
            1,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        assert_eq!(empty.estimate, 0.0);
        assert_eq!(empty.lambda_max_abs, 0.0);

        let dim = 16usize;
        let zeros = slq_logdet_unit_deflated(
            dim,
            |v| Array1::<f64>::zeros(v.len()),
            8,
            dim,
            2,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        assert!(zeros.estimate.is_finite());
        assert_eq!(zeros.estimate, 0.0);
    }

    /// #2308 — the unit-deflated estimate (value AND floor) is bit-reproducible for
    /// a fixed `(dim, matvec, probes, steps, seed)`, as the REML evidence outer
    /// loop requires of a differentiated objective.
    #[test]
    fn slq_unit_deflation_is_deterministic_2308() {
        let dim = 40usize;
        let mut state = 9u64;
        let mut eigenvalues = vec![0.0_f64; dim];
        for e in eigenvalues.iter_mut() {
            *e = next_uniform(&mut state, 0.3, 8.0);
        }
        eigenvalues[0] = -1.0e-10;
        let a = householder_spectrum_matrix(&eigenvalues, 77);
        let r1 = slq_logdet_unit_deflated(
            dim,
            |v| a.dot(&v),
            24,
            dim,
            99,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        let r2 = slq_logdet_unit_deflated(
            dim,
            |v| a.dot(&v),
            24,
            dim,
            99,
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
        assert_eq!(r1.estimate.to_bits(), r2.estimate.to_bits());
        assert_eq!(r1.deflate_floor.to_bits(), r2.deflate_floor.to_bits());
    }
}
