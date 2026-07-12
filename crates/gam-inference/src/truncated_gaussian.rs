//! Exact posterior sampling for a Gaussian (Laplace) coefficient posterior
//! restricted to a feasible polytope `{β : A β ≥ b}`.
//!
//! # Why this module exists
//!
//! A model fit with linear *inequality* constraints on its coefficients —
//! `nonnegative()` / `linear(min, max)` box bounds on a parametric term
//! (#1507), or the monotone/convex/concave shape cone `γ_j ≥ 0` on a spline
//! (#1509) — has a feasible region that is a convex polytope in coefficient
//! space. The constrained P-IRLS fit pins the point estimate to that polytope
//! (often onto an active face), but the posterior *sampler* historically drew a
//! plain unconstrained Gaussian `N(mode, φ·H⁻¹)`: when a bound is active that
//! Gaussian is centred on the boundary, so ~half its mass lands on the
//! forbidden side. The reported draws were confidently wrong.
//!
//! The principled posterior of a coefficient constrained to a polytope is the
//! Laplace Gaussian *truncated to that polytope*:
//!
//! ```text
//!     β ~ N(mode, φ·H⁻¹)   subject to   A β ≥ b.
//! ```
//!
//! For a Gaussian-identity model this truncated Gaussian is the *exact*
//! posterior (the un-truncated posterior is itself exactly `N(mode, φ·H⁻¹)`).
//! For a non-Gaussian GLM it is the constraint-respecting Laplace
//! approximation — the same modelling choice the `bounded()` interval term
//! already makes (it samples the Laplace Gaussian on a latent logit scale and
//! pushes it through the interval map). Box / cone constraints have no single
//! global smooth reparameterisation, so we sample the truncated Gaussian
//! directly.
//!
//! # Method — exact Hamiltonian Monte Carlo for truncated Gaussians
//!
//! We use Pakman & Paninski (2014, *"Exact Hamiltonian Monte Carlo for
//! Truncated Multivariate Gaussians"*, J. Comput. Graph. Statist.). After
//! whitening to a standard normal target the Hamiltonian trajectory is the
//! exactly-integrable harmonic oscillator `z(t) = z₀ cos t + v₀ sin t`; the
//! particle travels along that arc and *reflects* specularly off each linear
//! wall it reaches. The map preserves the truncated Gaussian exactly — there is
//! no Metropolis correction, no rejection, and (unlike a Gibbs sampler) no slow
//! mixing along correlated constraint directions. Refreshing the velocity from
//! `N(0, I)` and travelling for a quarter period `T = π/2` between draws makes
//! successive draws essentially independent (with no wall in the way,
//! `z(π/2) = v₀`), so the returned draws behave like i.i.d. samples — matching
//! the `rhat ≈ 1`, `ess ≈ n` contract of the other Laplace posterior paths.
//!
//! # Whitening
//!
//! With `H = L Lᵀ` (lower Cholesky) the target covariance is
//! `Σ = φ·H⁻¹ = (√φ·L⁻ᵀ)(√φ·L⁻ᵀ)ᵀ`, so `β = center + √φ·L⁻ᵀ z` maps
//! `z ~ N(0, I)` to `β ~ N(center, Σ)`. The constraint `A β ≥ b` becomes
//! `F z + g ≥ 0` with `Fᵢ = √φ·L⁻¹ aᵢ` (forward solve) and
//! `gᵢ = aᵢᵀ center − bᵢ`.
//!
//! `center` is the UNCONSTRAINED Gaussian center of the local quadratic — a
//! truncated Gaussian stays centred at its pre-truncation mean, and that mean
//! is not the boundary KKT mode (#2245 finding 20: centring `N(0,1)·1{β≥0}`
//! at a boundary mode of a `N(−1,1)` quadratic reports the half-normal mean
//! `0.798` where the true truncated mean is `0.525`). The center may be
//! infeasible (`g` can be negative); each chain instead starts from the
//! caller-supplied feasible point (the constrained mode), whose whitened
//! image `z₀ = (1/√φ)·Lᵀ·(start − center)` satisfies every wall, with active
//! constraints sitting ON their wall (the bounce logic launches the particle
//! inward).

use ndarray::{Array1, Array2};
use rand::SeedableRng;

use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::triangular::{
    back_substitution_lower_transpose_guarded_into, forward_substitution_lower_matrix,
};
use gam_solve::pirls::LinearInequalityConstraints;

/// Quarter-period travel time between velocity refreshes. With no active wall,
/// `z(π/2) = v₀`, so consecutive draws decorrelate completely.
const TRAVEL_TIME: f64 = std::f64::consts::FRAC_PI_2;

/// A constraint whose amplitude `R = ‖(uᵢ, wᵢ)‖` is below this floor cannot
/// reach its boundary along the current arc — treated as never-hit.
const AMPLITUDE_FLOOR: f64 = 1e-300;

/// Slack below which a constraint is considered "on the wall" at the current
/// position (in whitened units). Used to launch active-face starts inward and
/// to suppress spurious re-hits of a wall just reflected from.
const WALL_SLACK_EPS: f64 = 1e-9;

/// A reflection budget per trajectory. A pointed feasible cone resolves a
/// vertex start in `O(#active rows)` bounces; this cap is a backstop against a
/// pathological grazing cycle. On exhaustion we keep the current (feasible)
/// position — never an out-of-bounds draw.
const MAX_BOUNCES_BASE: usize = 256;

/// Draw `n_samples · n_chains` posterior samples of `β ~ N(center, φ·H⁻¹)`
/// truncated to `{β : A β ≥ b}`, returned as a `(n_total, p)` matrix in the
/// same coefficient coordinate system as `center` / `penalized_hessian` / `A`.
///
/// * `center` — the UNCONSTRAINED Gaussian center of the local quadratic
///   (`H⁻¹X′Wz` at the converged working state). A Gaussian truncated to a
///   feasible set stays centred at its pre-truncation mean; the boundary KKT
///   mode is NOT that mean, and centring there samples a different law
///   (half-normal instead of the correct boundary-truncated Gaussian — #2245
///   finding 20). May be infeasible; only the start point must be feasible.
/// * `feasible_start` — a feasible point (`A·start ≥ b`, up to numeric
///   slack), normally the constrained fit's KKT mode. Used only to seed each
///   reflective chain.
/// * `penalized_hessian` — the *unscaled* penalised Hessian `H` (no φ).
/// * `sqrt_phi` — `√φ` (dispersion square root); `1.0` for fixed-scale
///   families (Binomial / Poisson). Scales the posterior covariance to
///   `φ·H⁻¹`, exactly as [`crate::sample::laplace_gaussian_fallback`].
/// * `constraints` — `A` (`m × p`) and `b` (`m`), meaning `A β ≥ b`.
pub fn sample_truncated_gaussian_posterior(
    center: &Array1<f64>,
    feasible_start: &Array1<f64>,
    penalized_hessian: &Array2<f64>,
    sqrt_phi: f64,
    constraints: &LinearInequalityConstraints,
    n_samples: usize,
    n_chains: usize,
    seed: u64,
) -> Result<Array2<f64>, String> {
    let p = center.len();
    if feasible_start.len() != p {
        return Err(format!(
            "truncated-Gaussian posterior: start point has {} coefficients, expected {p}",
            feasible_start.len(),
        ));
    }
    if p == 0 {
        return Err(
            "truncated-Gaussian posterior: cannot sample from an empty coefficient vector"
                .to_string(),
        );
    }
    if penalized_hessian.nrows() != p || penalized_hessian.ncols() != p {
        return Err(format!(
            "truncated-Gaussian posterior: penalised Hessian is {}x{}, expected {p}x{p}",
            penalized_hessian.nrows(),
            penalized_hessian.ncols(),
        ));
    }
    let a = &constraints.a;
    let b = &constraints.b;
    let m = a.nrows();
    if m != b.len() {
        return Err(format!(
            "truncated-Gaussian posterior: constraint row mismatch (A has {m} rows, b has {})",
            b.len(),
        ));
    }
    if m > 0 && a.ncols() != p {
        return Err(format!(
            "truncated-Gaussian posterior: constraint matrix has {} columns, expected {p}",
            a.ncols(),
        ));
    }
    if !sqrt_phi.is_finite() || sqrt_phi <= 0.0 {
        return Err(format!(
            "truncated-Gaussian posterior: non-positive or non-finite √φ ({sqrt_phi})"
        ));
    }

    // H = L Lᵀ.
    let chol = penalized_hessian
        .cholesky(faer::Side::Lower)
        .map_err(|err| {
            format!(
                "truncated-Gaussian posterior: Cholesky of the penalised Hessian failed: {err:?}"
            )
        })?;
    let l = chol.lower_triangular();

    // Whitened constraint rows Fᵢ = √φ · L⁻¹ aᵢ and slacks gᵢ = aᵢᵀ center − bᵢ
    // (possibly negative — the CENTER may be infeasible; only the start point
    // must satisfy the polytope). `F` is `m × p`;
    // `forward_substitution_lower_matrix` solves `L M = Aᵀ` column-by-column
    // giving `M = L⁻¹ Aᵀ` (`p × m`), so `F = √φ · Mᵀ`.
    let (f_rows, g, f_sq_norm) = if m == 0 {
        (
            Array2::<f64>::zeros((0, p)),
            Array1::<f64>::zeros(0),
            Vec::new(),
        )
    } else {
        let at = a.t().to_owned();
        let mut f = forward_substitution_lower_matrix(&l, &at).reversed_axes(); // m × p
        f.mapv_inplace(|v| v * sqrt_phi);
        let g = a.dot(center) - b;
        let f_sq_norm: Vec<f64> = (0..m).map(|i| f.row(i).dot(&f.row(i))).collect();
        (f, g, f_sq_norm)
    };

    // Whitened start `z₀ = (1/√φ)·Lᵀ·(start − center)`, validated feasible up
    // to reflective slack: the constrained KKT mode sits ON its active walls,
    // so tiny negative numeric slack is snapped by the wall logic, but a
    // genuinely infeasible start would corrupt every trajectory.
    let start_diff = feasible_start - center;
    let z0 = l.t().dot(&start_diff) / sqrt_phi;
    for i in 0..m {
        let slack = a.row(i).dot(feasible_start) - b[i];
        let scale = a.row(i).iter().map(|v| v.abs()).sum::<f64>().max(1.0)
            * feasible_start
                .iter()
                .map(|v| v.abs())
                .fold(1.0_f64, f64::max);
        if slack < -1e-8 * scale {
            return Err(format!(
                "truncated-Gaussian posterior: start point violates constraint row {i} \
                 (slack {slack:.3e}); the constrained mode must be feasible"
            ));
        }
    }

    let n_total = n_samples.saturating_mul(n_chains);
    let mut samples = Array2::<f64>::zeros((n_total, p));
    let max_bounces = MAX_BOUNCES_BASE + 8 * m;

    // Scratch buffers reused across draws.
    let mut z = Array1::<f64>::zeros(p);
    let mut v = Array1::<f64>::zeros(p);
    let mut beta = Array1::<f64>::zeros(p);

    for chain in 0..n_chains {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed ^ ((chain as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        // Each chain starts at the feasible start point (the constrained
        // mode), which in whitened coordinates is `z₀` — NOT the origin: the
        // origin is the unconstrained center, which may be infeasible.
        z.assign(&z0);
        for draw in 0..n_samples {
            // Refresh the velocity from N(0, I).
            for vi in v.iter_mut() {
                *vi = standard_normal(&mut rng);
            }
            simulate_constrained_trajectory(&mut z, &mut v, &f_rows, &g, &f_sq_norm, max_bounces);
            // Back-transform: β = center + √φ · L⁻ᵀ z.
            back_substitution_lower_transpose_guarded_into(&l, &z, &mut beta);
            let row = chain * n_samples + draw;
            for j in 0..p {
                samples[(row, j)] = center[j] + sqrt_phi * beta[j];
            }
        }
    }

    Ok(samples)
}

/// Advance `(z, v)` along the harmonic trajectory `z(t) = z cos t + v sin t`
/// for a total time [`TRAVEL_TIME`], reflecting specularly off every wall
/// `fᵢᵀ z + gᵢ = 0` it reaches. On return `z` is the new (feasible) position.
fn simulate_constrained_trajectory(
    z: &mut Array1<f64>,
    v: &mut Array1<f64>,
    f_rows: &Array2<f64>,
    g: &Array1<f64>,
    f_sq_norm: &[f64],
    max_bounces: usize,
) {
    let m = f_rows.nrows();
    let mut t_left = TRAVEL_TIME;
    let mut bounces = 0usize;

    loop {
        if t_left <= 0.0 {
            return;
        }
        // Find the first wall hit within (0, t_left].
        let mut hit_time = t_left;
        let mut hit_wall: Option<usize> = None;
        for i in 0..m {
            let fi = f_rows.row(i);
            let u = fi.dot(z); // fᵢᵀ z   (so cᵢ(0) = u + gᵢ)
            let w = fi.dot(v); // fᵢᵀ v
            if let Some(t) = first_wall_hit(u, w, g[i], hit_time) {
                if t < hit_time {
                    hit_time = t;
                    hit_wall = Some(i);
                } else if hit_wall.is_none() && t <= hit_time {
                    // Immediate (t == 0) outward bounce on an active face.
                    hit_time = t;
                    hit_wall = Some(i);
                }
            }
        }

        match hit_wall {
            None => {
                // No wall within the remaining arc: advance the full time.
                advance(z, v, t_left);
                return;
            }
            Some(j) => {
                advance(z, v, hit_time);
                t_left -= hit_time;
                // Specular reflection of the velocity about the wall normal fⱼ:
                //   v ← v − 2 (fⱼᵀ v / ‖fⱼ‖²) fⱼ,
                // which flips the outward normal velocity component to inward.
                let fj = f_rows.row(j);
                let denom = f_sq_norm[j];
                if denom > 0.0 {
                    let coeff = 2.0 * fj.dot(v) / denom;
                    for k in 0..v.len() {
                        v[k] -= coeff * fj[k];
                    }
                }
                bounces += 1;
                if bounces >= max_bounces {
                    // Backstop: keep the current feasible position. `z` already
                    // satisfies every constraint (we only ever advanced *to* a
                    // wall, never through it), so the draw is in-bounds.
                    return;
                }
            }
        }
    }
}

/// First time `t ∈ (0, t_max]` at which the constraint value
/// `c(t) = u cos t + w sin t + g` crosses zero *downward* (feasible → wall),
/// or `None` if the arc never reaches the wall within `t_max`.
///
/// `c(0) = u + g`. The mode is feasible so `g ≥ 0`; an active face has `g = 0`
/// and the particle may start exactly on the wall.
#[inline]
fn first_wall_hit(u: f64, w: f64, g: f64, t_max: f64) -> Option<f64> {
    let c0 = u + g;
    if c0 <= WALL_SLACK_EPS {
        // Currently on (or numerically at) the wall.
        if w < 0.0 {
            // Moving outward → reflect immediately.
            return Some(0.0);
        }
        if g >= 0.0 {
            // Feasible-center geometry: from the wall with inward (or
            // tangent) velocity, the next downward crossing sits at
            // ψ + acos(−g/r) ≥ ψ + π/2 > t_max, so there is no hit to
            // register within the arc.
            return None;
        }
        // INFEASIBLE center (g < 0): the harmonic pull points THROUGH this
        // wall, so an inward launch curves back and exits the polytope
        // within the arc whenever the launch speed is small (downward
        // crossing at ψ + acos(−g/r) with acos(·) < π/2). Taking the
        // feasible-center shortcut here let the trajectory pass straight
        // through the wall — the measured infeasible draw (β = −0.518 for
        // the β ≥ 0, center −1 half-normal fixture, whose start sits
        // exactly on its active wall). Compute the exact downward crossing
        // HERE rather than falling through: the interior path's sub-eps
        // wrap-correction assumes c0 > eps and would misread a genuine
        // near-tangent exit (t ≈ 0⁺) as a wrap artefact. A sub-eps root is
        // returned as an immediate bounce; the bounce cap backstops the
        // degenerate tangent case at a feasible on-wall position.
        let r = (u * u + w * w).sqrt();
        if r <= AMPLITUDE_FLOOR {
            return None;
        }
        let q = (-g / r).clamp(-1.0, 1.0);
        let t = (w.atan2(u) + q.acos()).rem_euclid(2.0 * std::f64::consts::PI);
        return if t <= t_max { Some(t) } else { None };
    }

    let r = (u * u + w * w).sqrt();
    if r <= AMPLITUDE_FLOOR {
        // c(t) ≈ g > 0 constant — never reaches the wall.
        return None;
    }
    // Need cos(t − ψ) = −g / r. With g ≥ 0 this is ≤ 0; if it is below −1 the
    // amplitude is too small to ever reach the wall.
    let q = -g / r;
    if q < -1.0 {
        return None;
    }
    let q = q.clamp(-1.0, 1.0);
    let psi = w.atan2(u);
    let alpha = q.acos(); // ∈ [0, π]
    // Downward crossings occur at t ≡ ψ + α (mod 2π); the upward ones at ψ − α.
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut t = (psi + alpha).rem_euclid(two_pi);
    if t <= WALL_SLACK_EPS {
        // We are interior (c0 > eps), so a near-zero root is a wrap artefact of
        // the principal value; the true first downward crossing is one period
        // on. (Beyond t_max ≤ π/2, hence discarded below.)
        t += two_pi;
    }
    if t <= t_max { Some(t) } else { None }
}

/// In-place harmonic advance: `z ← z cos t + v sin t`, `v ← −z sin t + v cos t`.
#[inline]
fn advance(z: &mut Array1<f64>, v: &mut Array1<f64>, t: f64) {
    if t == 0.0 {
        return;
    }
    let (st, ct) = t.sin_cos();
    for k in 0..z.len() {
        let zk = z[k];
        let vk = v[k];
        z[k] = zk * ct + vk * st;
        v[k] = -zk * st + vk * ct;
    }
}

/// Box–Muller standard-normal draw, matching the engine's other sampler RNG
/// paths (`sample.rs`, the bounded latent sampler).
#[inline]
fn standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    use rand::RngExt as _;
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn constraints(a: Array2<f64>, b: Array1<f64>) -> LinearInequalityConstraints {
        LinearInequalityConstraints::new(a, b).expect("valid constraints")
    }

    /// Every draw must satisfy `A β ≥ b` exactly (the reflective dynamics only
    /// ever advance *to* a wall, never through it).
    fn assert_all_feasible(samples: &Array2<f64>, c: &LinearInequalityConstraints) {
        for k in 0..samples.nrows() {
            let beta = samples.row(k).to_owned();
            let slack = c.a.dot(&beta) - &c.b;
            for (i, s) in slack.iter().enumerate() {
                assert!(
                    *s >= -1e-8,
                    "draw {k} violates constraint {i}: slack {s} (β = {beta})"
                );
            }
        }
    }

    /// With a *loose* (non-binding) constraint the truncated Gaussian must
    /// reproduce the un-truncated `N(mode, φ·H⁻¹)`: sample mean ≈ mode and
    /// sample covariance ≈ φ·H⁻¹.
    #[test]
    fn loose_constraint_recovers_unconstrained_gaussian() {
        // H = [[4, 1],[1, 3]], φ = 1 → Σ = H⁻¹ = 1/11 [[3,-1],[-1,4]].
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let mode = array![0.5, -0.3];
        // β₀ ≥ −1000: utterly non-binding at this mode/scale.
        let c = constraints(array![[1.0, 0.0]], array![-1000.0]);
        let n = 60_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 1, 20240613)
            .expect("sampler");
        assert_all_feasible(&s, &c);

        let mean = s.mean_axis(ndarray::Axis(0)).unwrap();
        assert!((mean[0] - 0.5).abs() < 0.02, "mean0 {} ", mean[0]);
        assert!((mean[1] + 0.3).abs() < 0.02, "mean1 {}", mean[1]);

        // Sample covariance vs Σ = H⁻¹.
        let det = 4.0 * 3.0 - 1.0;
        let sigma = array![[3.0 / det, -1.0 / det], [-1.0 / det, 4.0 / det]];
        let mut cov = Array2::<f64>::zeros((2, 2));
        for k in 0..n {
            let d0 = s[(k, 0)] - mean[0];
            let d1 = s[(k, 1)] - mean[1];
            cov[(0, 0)] += d0 * d0;
            cov[(0, 1)] += d0 * d1;
            cov[(1, 1)] += d1 * d1;
        }
        cov.mapv_inplace(|v| v / (n as f64 - 1.0));
        cov[(1, 0)] = cov[(0, 1)];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cov[(i, j)] - sigma[(i, j)]).abs() < 0.01,
                    "cov[{i},{j}] {} vs Σ {}",
                    cov[(i, j)],
                    sigma[(i, j)]
                );
            }
        }
    }

    /// Active lower bound `β ≥ 0` with the mode pinned to the boundary: the
    /// posterior is a half-normal. Check feasibility and the analytic half-normal
    /// moments E = σ√(2/π), Var = σ²(1 − 2/π).
    #[test]
    fn active_lower_bound_is_half_normal() {
        // Σ = φ·H⁻¹ = 1/h. Pick σ = 2 → h = 0.25, φ = 1.
        let sigma = 2.0_f64;
        let h = array![[1.0 / (sigma * sigma)]];
        let mode = array![0.0]; // pinned on the boundary (active constraint)
        let c = constraints(array![[1.0]], array![0.0]); // β ≥ 0
        let n = 200_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 1, 7)
            .expect("sampler");
        assert_all_feasible(&s, &c);

        let col = s.column(0);
        let mean = col.mean().unwrap();
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let two_over_pi = 2.0 / std::f64::consts::PI;
        let expect_mean = sigma * two_over_pi.sqrt();
        let expect_var = sigma * sigma * (1.0 - two_over_pi);
        assert!(
            (mean - expect_mean).abs() < 0.02,
            "half-normal mean {mean} vs {expect_mean}"
        );
        assert!(
            (var - expect_var).abs() < 0.05,
            "half-normal var {var} vs {expect_var}"
        );
        assert!(col.iter().all(|&v| v >= 0.0), "a draw escaped β ≥ 0");
    }

    /// #2245 finding 20: an INFEASIBLE unconstrained center `N(−1, 1)`
    /// truncated to `β ≥ 0` has mean `−1 + φ(1)/(1−Φ(1)) ≈ 0.52514`, not the
    /// half-normal `√(2/π) ≈ 0.79788` produced by re-centring at the boundary
    /// KKT mode. The feasible start is the boundary mode `0`.
    #[test]
    fn infeasible_center_matches_truncated_normal_mean() {
        let h = array![[1.0]]; // σ = 1
        let center = array![-1.0];
        let start = array![0.0];
        let c = constraints(array![[1.0]], array![0.0]); // β ≥ 0
        let n = 200_000;
        let s = sample_truncated_gaussian_posterior(&center, &start, &h, 1.0, &c, n, 1, 424242)
            .expect("sampler");
        assert_all_feasible(&s, &c);
        let mean = s.column(0).mean().unwrap();
        let expect = 0.525_135_7; // −1 + φ(1)/(1−Φ(1))
        assert!(
            (mean - expect).abs() < 0.02,
            "truncated-normal mean {mean} vs analytic {expect}"
        );
        let half_normal = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean - half_normal).abs() > 0.2,
            "mean {mean} matches the boundary-centred half-normal — center regression"
        );
    }

    /// √φ scales the posterior covariance: doubling φ (×4 covariance) widens the
    /// half-normal moments by the same factor.
    #[test]
    fn dispersion_scales_covariance() {
        let h = array![[1.0]];
        let mode = array![0.0];
        let c = constraints(array![[1.0]], array![0.0]);
        let n = 200_000;
        let sqrt_phi = 2.0; // φ = 4 → σ = sqrt(φ/h) = 2.
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, sqrt_phi, &c, n, 1, 99)
            .expect("sampler");
        let mean = s.column(0).mean().unwrap();
        let expect = 2.0 * (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean - expect).abs() < 0.03,
            "scaled mean {mean} vs {expect}"
        );
    }

    /// A monotone-cone style polytope: several coordinate lower bounds
    /// `γ_j ≥ 0` with a correlated Hessian. Every draw must lie in the cone.
    #[test]
    fn monotone_cone_draws_stay_feasible() {
        let p = 6;
        // SPD Hessian: coord 0 (the free coord) is decoupled so its marginal is
        // a clean unconstrained check; coords 1..p are tridiagonally correlated
        // *and* truncated, stressing the reflective dynamics under correlation.
        let mut h = Array2::<f64>::eye(p);
        for i in 0..p {
            h[(i, i)] = 3.0;
            if i >= 1 && i + 1 < p {
                h[(i, i + 1)] = 0.7;
                h[(i + 1, i)] = 0.7;
            }
        }
        // Mode: first coord free, the rest pinned to the cone vertex (active).
        let mode = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // γ_j ≥ 0 for j = 1..p (a monotone-increasing reparam cone).
        let mut a = Array2::<f64>::zeros((p - 1, p));
        for r in 0..(p - 1) {
            a[(r, r + 1)] = 1.0;
        }
        let c = constraints(a, Array1::zeros(p - 1));
        let n = 40_000;
        let chains = 2;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, chains, 31337)
            .expect("sampler");
        assert_eq!(s.dim(), (n * chains, p));
        assert_all_feasible(&s, &c);
        // The free coordinate is unconstrained → its sample mean tracks the mode.
        assert!((s.column(0).mean().unwrap() - 1.0).abs() < 0.05);
    }

    /// An interior two-sided box `min ≤ β ≤ max` with a *centred* mode: draws
    /// stay strictly inside, and by symmetry the truncated mean equals the mode.
    /// (A mode placed near a wall would pull the truncated mean off the mode —
    /// the truncation is then asymmetric — so the mode is centred here.)
    #[test]
    fn interior_box_keeps_draws_in_interval() {
        let h = array![[44.0]]; // σ ≈ 0.151 → both walls ~3.3σ away
        let mode = array![0.5];
        // β ≥ 0 and −β ≥ −1  ⟺  0 ≤ β ≤ 1.
        let c = constraints(array![[1.0], [-1.0]], array![0.0, -1.0]);
        let n = 80_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 1, 5)
            .expect("sampler");
        assert_all_feasible(&s, &c);
        assert!(s.column(0).iter().all(|&v| v > 0.0 && v < 1.0));
        // Symmetric truncation around the centred mode ⇒ mean ≈ 0.5.
        assert!((s.column(0).mean().unwrap() - 0.5).abs() < 0.01);
    }
}
