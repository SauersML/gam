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
//! `N(0, I)` and travelling for a quarter period `T = π/2` between draws gives
//! independent draws when no wall intervenes (`z(π/2) = v₀`). Reflections can
//! induce serial dependence; constrained draws still require chain diagnostics.
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
//! infeasible (`g` can be negative), so chains start from feasible points.
//!
//! # Chain starts and burn-in
//!
//! Chains that share one start give split R-hat no between-chain spread by
//! which to see that start's transient, and draws kept from the start carry
//! the transient into every reported moment (#3380). Each chain therefore starts from its own
//! independent unconstrained draw `N(center, φ·H⁻¹)` projected onto the
//! polytope in the posterior's own `H` metric — the constrained quadratic solve
//! the fit itself uses. A draw already inside the polytope is its own start, an
//! exact draw from the target. The chains then burn in by the engine's doubling
//! windows ([`crate::hmc_io::burn_in_until_mixed`]) until their draws meet the
//! convergence targets, and only the draws after burn-in are returned.

use std::collections::HashSet;

use ndarray::{Array1, Array2, Array3, ArrayViewMut1};
use rand::SeedableRng;

use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::triangular::{
    back_substitution_lower_transpose_guarded_into, forward_substitution_lower_matrix,
};
use gam_solve::active_set::solve_quadratic_with_linear_constraints;
use gam_solve::pirls::LinearInequalityConstraints;

use crate::hmc_io::{NUTS_CHAINS, burn_in_until_mixed};

/// Quarter-period travel time between velocity refreshes. With no active wall,
/// `z(π/2) = v₀`, so consecutive draws decorrelate completely.
const TRAVEL_TIME: f64 = std::f64::consts::FRAC_PI_2;

/// Posterior draws of the truncated Gaussian after burn-in.
#[derive(Debug)]
pub(crate) struct TruncatedGaussianDraws {
    /// `(NUTS_CHAINS, n_samples, p)` draws kept after burn-in.
    pub(crate) chains: Array3<f64>,
    /// Transitions each chain ran before its first kept draw.
    pub(crate) warmup_transitions: usize,
}

impl TruncatedGaussianDraws {
    /// The kept draws stacked chain-major as a `(NUTS_CHAINS·n_samples, p)`
    /// matrix, row `chain·n_samples + draw`.
    pub(crate) fn into_stacked(self) -> Array2<f64> {
        let (chains, n_samples, p) = self.chains.dim();
        self.chains
            .into_shape_with_order((chains * n_samples, p))
            .expect("a standard-layout (chain, draw, coefficient) array stacks chain-major")
    }
}

/// Draw `n_samples` posterior samples per chain of `β ~ N(center, φ·H⁻¹)`
/// truncated to `{β : A β ≥ b}` from [`NUTS_CHAINS`] burned-in chains, in the
/// same coefficient coordinate system as `center` / `penalized_hessian` / `A`.
///
/// * `center` — the UNCONSTRAINED Gaussian center of the local quadratic
///   (`H⁻¹X′Wz` at the converged working state). A Gaussian truncated to a
///   feasible set stays centred at its pre-truncation mean; the boundary KKT
///   mode is NOT that mean, and centring there samples a different law
///   (half-normal instead of the correct boundary-truncated Gaussian — #2245
///   finding 20). May be infeasible; only the start point must be feasible.
/// * `feasible_start` — a feasible point (`A·start ≥ b`, up to numeric
///   slack), normally the constrained fit's KKT mode. It warm-starts the
///   projections that place each chain's start.
/// * `penalized_hessian` — the *unscaled* penalised Hessian `H` (no φ).
/// * `sqrt_phi` — `√φ` (dispersion square root); `1.0` for fixed-scale
///   families (Binomial / Poisson). Scales the posterior covariance to
///   `φ·H⁻¹`, exactly as `crate::sample::laplace_gaussian_fallback`.
/// * `constraints` — `A` (`m × p`) and `b` (`m`), meaning `A β ≥ b`.
pub(crate) fn sample_truncated_gaussian_posterior(
    center: &Array1<f64>,
    feasible_start: &Array1<f64>,
    penalized_hessian: &Array2<f64>,
    sqrt_phi: f64,
    constraints: &LinearInequalityConstraints,
    n_samples: usize,
    seed: u64,
) -> Result<TruncatedGaussianDraws, String> {
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
        let mut g = a.dot(center) - b;
        // Equivalent positive row scalings must give the same reflections.
        // Normalize in two stages before squaring: raw row norms can overflow
        // or underflow even when every coefficient and the unit normal is finite.
        for i in 0..m {
            let row_scale = f.row(i).iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
            if row_scale > 0.0 {
                f.row_mut(i).mapv_inplace(|v| v / row_scale);
                g[i] /= row_scale;
                let row_norm = f.row(i).dot(&f.row(i)).sqrt();
                f.row_mut(i).mapv_inplace(|v| v / row_norm);
                g[i] /= row_norm;
            }
        }
        let f_sq_norm: Vec<f64> = (0..m).map(|i| f.row(i).dot(&f.row(i))).collect();
        (f, g, f_sq_norm)
    };

    // The start is the persisted feasible optimizer mode, which the constrained
    // solver certifies to a scaled violation `(aᵢᵀx − bᵢ)/‖aᵢ‖` of at most
    // `PRIMAL_FEASIBILITY_TOL`, a zero row being infeasible iff `bᵢ > 0`. Refuse
    // exactly what that contract refuses: a tighter or differently scaled test
    // rejects valid boundary modes (gam#2719, #2469). The projected chain
    // starts come from the same solver under the same contract, so tiny
    // negative slack is snapped by the wall logic, while a genuinely infeasible
    // point would corrupt every trajectory.
    check_feasible(feasible_start, constraints, "start point")?;

    // Each chain starts from its own unconstrained draw `z ~ N(0, I)`, i.e.
    // `β = center + √φ·L⁻ᵀ z`, moved to the nearest feasible point in the
    // posterior metric `H`: the minimizer of `½ xᵀHx − (Hβ)ᵀx` over the
    // polytope. Its whitened image `(1/√φ)·Lᵀ·(x − center)` is the chain's
    // position, with the projected start ON its active walls (the bounce logic
    // launches the particle inward).
    let mut rngs: Vec<rand::rngs::StdRng> = (0..NUTS_CHAINS)
        .map(|chain| {
            rand::rngs::StdRng::seed_from_u64(
                seed ^ ((chain as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            )
        })
        .collect();
    let mut positions = Vec::with_capacity(NUTS_CHAINS);
    let mut beta = Array1::<f64>::zeros(p);
    for rng in &mut rngs {
        let draw = Array1::from_shape_fn(p, |_| standard_normal(rng));
        back_substitution_lower_transpose_guarded_into(&l, &draw, &mut beta);
        let unconstrained = center + &(sqrt_phi * &beta);
        let start = if m == 0 {
            unconstrained
        } else {
            let (projected, _active) = solve_quadratic_with_linear_constraints(
                penalized_hessian,
                &penalized_hessian.dot(&unconstrained),
                feasible_start,
                constraints,
                None,
            )
            .map_err(|err| {
                format!(
                    "truncated-Gaussian posterior: projecting a chain start onto the \
                     constraints failed: {err}"
                )
            })?;
            check_feasible(&projected, constraints, "projected chain start")?;
            projected
        };
        positions.push(l.t().dot(&(&start - center)) / sqrt_phi);
    }

    // One transition of a chain: refresh the velocity from N(0, I), travel the
    // reflected harmonic arc, and back-transform β = center + √φ · L⁻ᵀ z.
    let mut v = Array1::<f64>::zeros(p);
    let mut transition = |chain: usize, mut draw: ArrayViewMut1<'_, f64>| -> Result<(), String> {
        let z = &mut positions[chain];
        for vi in v.iter_mut() {
            *vi = standard_normal(&mut rngs[chain]);
        }
        simulate_constrained_trajectory(z, &mut v, &f_rows, &g, &f_sq_norm)?;
        back_substitution_lower_transpose_guarded_into(&l, z, &mut beta);
        for j in 0..p {
            draw[j] = center[j] + sqrt_phi * beta[j];
        }
        Ok(())
    };

    let warmup_transitions = burn_in_until_mixed(
        NUTS_CHAINS,
        p,
        "truncated-Gaussian reflective HMC",
        "transitions",
        &mut transition,
    )?;
    let mut chains = Array3::<f64>::zeros((NUTS_CHAINS, n_samples, p));
    for chain in 0..NUTS_CHAINS {
        for t in 0..n_samples {
            transition(chain, chains.slice_mut(ndarray::s![chain, t, ..]))?;
        }
    }

    Ok(TruncatedGaussianDraws {
        chains,
        warmup_transitions,
    })
}

/// Refuse a point outside `A x ≥ b` beyond the solver's feasibility contract: a
/// scaled violation `(aᵢᵀx − bᵢ)/‖aᵢ‖` above `PRIMAL_FEASIBILITY_TOL`, a zero
/// row being infeasible iff `bᵢ > 0`.
fn check_feasible(
    point: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    what: &str,
) -> Result<(), String> {
    let (a, b) = (&constraints.a, &constraints.b);
    let feasibility_tol = gam_problem::PRIMAL_FEASIBILITY_TOL;
    for i in 0..a.nrows() {
        let row_scale = a.row(i).iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
        let scaled_slack = if row_scale > 0.0 {
            let unit_norm = a
                .row(i)
                .iter()
                .map(|&v| (v / row_scale) * (v / row_scale))
                .sum::<f64>()
                .sqrt();
            let unit_dot = a
                .row(i)
                .iter()
                .zip(point.iter())
                .map(|(&v, &x)| (v / row_scale) * x)
                .sum::<f64>();
            (unit_dot - b[i] / row_scale) / unit_norm
        } else if b[i] > 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        if !(scaled_slack >= -feasibility_tol) {
            return Err(format!(
                "truncated-Gaussian posterior: {what} violates constraint row {i} \
                 (scaled slack {scaled_slack:.3e}, contract tolerance {feasibility_tol:.3e})"
            ));
        }
    }
    Ok(())
}

/// Advance `(z, v)` along the harmonic trajectory `z(t) = z cos t + v sin t`
/// for a total time [`TRAVEL_TIME`], reflecting specularly off every wall
/// `fᵢᵀ z + gᵢ = 0` it reaches. On return `z` is the new (feasible) position.
///
/// Termination is a property of the time integration, not of a reflection
/// count: a harmonic trajectory meets finitely many linear walls in the finite
/// time [`TRAVEL_TIME`] almost surely, however many that is (thousands for a
/// shape cone whose mass is pressed into its apex, gam#3112). The one way the
/// loop can fail to finish is Zeno accumulation at a wall intersection, where
/// the time increments between hits reach zero. While the clock `t_left` does
/// not advance, each step is a deterministic map of the floating-point state
/// `(z, v)`, so such a run either leaves the intersection or revisits a state
/// exactly; the revisit is detected bit for bit and refused.
fn simulate_constrained_trajectory(
    z: &mut Array1<f64>,
    v: &mut Array1<f64>,
    f_rows: &Array2<f64>,
    g: &Array1<f64>,
    f_sq_norm: &[f64],
) -> Result<(), String> {
    let m = f_rows.nrows();
    let mut t_left = TRAVEL_TIME;
    // States visited since the clock last advanced (bit patterns of `z`, `v`).
    let mut stalled_states: HashSet<Vec<u64>> = HashSet::new();

    loop {
        if t_left <= 0.0 {
            return Ok(());
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
                return Ok(());
            }
            Some(j) => {
                advance(z, v, hit_time);
                let t_before = t_left;
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
                if t_left < t_before {
                    stalled_states.clear();
                } else {
                    let state: Vec<u64> = z.iter().chain(v.iter()).map(|x| x.to_bits()).collect();
                    if !stalled_states.insert(state) {
                        return Err(format!(
                            "truncated-Gaussian posterior: Zeno cycle at a wall intersection; \
                             reflections returned to an earlier state without advancing the \
                             travel clock ({t_left:.6} of {TRAVEL_TIME:.6} left), so the \
                             trajectory cannot complete its fixed travel time"
                        ));
                    }
                }
            }
        }
    }
}

/// First time `t ∈ (0, t_max]` at which the constraint value
/// `c(t) = u cos t + w sin t + g` crosses zero *downward* (feasible → wall),
/// or `None` if the arc never reaches the wall within `t_max`.
///
/// `c(0) = u + g ≥ 0` at a feasible position. The Gaussian center may be
/// infeasible (`g < 0`), and the particle may start exactly on a wall.
#[inline]
fn first_wall_hit(u: f64, w: f64, g: f64, t_max: f64) -> Option<f64> {
    // Positive rescaling of a constraint must not change its impact time.
    // Normalize before products so finite large row scales cannot overflow.
    let scale = u.abs().max(w.abs()).max(g.abs());
    if scale == 0.0 {
        return None;
    }
    let u = u / scale;
    let w = w / scale;
    let g = g / scale;
    let c0 = u + g;
    if c0 <= 0.0 && w < 0.0 {
        return Some(0.0);
    }

    // On our quarter-period arc, s = tan(t/2) is finite and nonnegative.
    // Multiplying c(t) by 1+s² gives
    //     (g-u)s² + 2ws + (u+g) = 0.
    // Select the root with negative derivative. Rationalizing it when w<0
    // avoids cancellation near t=0; adding atan2 and acos instead can erase
    // such a hit, while wrapping a small positive time skips a real impact.
    let a = g - u;
    let discriminant = w.mul_add(w, -a * c0);
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let s = if w < 0.0 {
        c0 / (-w + root)
    } else if a < 0.0 {
        (-w - root) / a
    } else {
        return None;
    };
    let t = 2.0 * s.max(0.0).atan();
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
    // `random::<f64>()` lies in [0, 1); its complement lies in (0, 1], so `ln u1` is finite.
    let u1 = 1.0 - rng.random::<f64>();
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

    #[test]
    fn wall_hit_keeps_positive_times_below_the_old_slack_threshold() {
        let expected = (1e-6_f64 / 1e8).atan();
        let actual = first_wall_hit(1e-6, -1e8, 0.0, TRAVEL_TIME).expect("early hit");
        assert!((actual / expected - 1.0).abs() < 1e-14);
    }

    #[test]
    fn wall_hit_is_invariant_to_positive_constraint_rescaling() {
        // cos(t) - sin(t) = 0 first crosses downward at pi/4.
        for scale in [1e-200, 1.0, 1e200] {
            let t = first_wall_hit(scale, -scale, 0.0, TRAVEL_TIME).expect("hit");
            assert!((t - std::f64::consts::FRAC_PI_4).abs() < 1e-14);
        }
    }

    #[test]
    fn wall_hit_handles_an_infeasible_center_and_an_inward_launch() {
        // Starting at z=1 with wall z>=1, the next return solves
        // cos(t) + v sin(t) = 1, hence t=2 atan(v).
        for velocity in [1e-14, 0.25, 0.5] {
            let t = first_wall_hit(1.0, velocity, -1.0, TRAVEL_TIME).expect("return hit");
            assert!((t / (2.0 * velocity.atan()) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn zeno_cycle_at_a_wall_intersection_is_refused() {
        // `z ≥ 0` and `z ≤ 0` meet in a single point: every reflection is
        // instantaneous and the velocity flips back and forth forever.
        let error = simulate_constrained_trajectory(
            &mut array![0.0],
            &mut array![-2.0],
            &array![[1.0], [-1.0]],
            &array![0.0, 0.0],
            &[1.0, 1.0],
        )
        .expect_err("a trajectory whose clock cannot advance never completes its travel time");
        assert!(error.contains("Zeno cycle"));
    }

    #[test]
    fn apex_of_a_narrow_wedge_completes_thousands_of_reflections() {
        // gam#3112: a shape cone with its unconstrained center far outside
        // presses the mass into the apex, where a billiard in a wedge of
        // angle θ reflects O(π/θ) times per excursion. A fixed reflection
        // budget refused these trajectories; the time integration completes
        // them and every draw stays inside the wedge.
        let theta: f64 = 1e-3;
        let apex = 30.0;
        // Walls through (apex, 0): z₂ ≥ 0 and z₁ sin θ − z₂ cos θ ≥ apex sin θ.
        let a = array![[0.0, 1.0], [theta.sin(), -theta.cos()]];
        let b = array![0.0, apex * theta.sin()];
        let draws = sample_truncated_gaussian_posterior(
            &array![0.0, 0.0],
            &array![apex, 0.0],
            &Array2::eye(2),
            1.0,
            &constraints(a.clone(), b.clone()),
            50,
            7,
        )
        .expect("apex-pressed wedge draws")
        .into_stacked();
        for row in draws.rows() {
            let slack = a.dot(&row) - &b;
            assert!(
                slack.iter().all(|&s| s >= -1e-9),
                "draw {row} left the wedge (slack {slack})"
            );
        }
    }

    #[test]
    fn posterior_draws_are_invariant_to_extreme_constraint_row_scales() {
        let draw = |scale| {
            sample_truncated_gaussian_posterior(
                &array![0.0],
                &array![0.0],
                &array![[1.0]],
                1.0,
                &constraints(array![[scale]], array![0.0]),
                16,
                734,
            )
            .expect("rescaled half-normal sampler")
            .into_stacked()
        };
        let expected = draw(1.0);
        for scale in [1e-200, 1e200] {
            assert_eq!(draw(scale), expected);
        }
        assert!(expected.iter().all(|&value| value >= 0.0));
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
        let n = 30_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 20240613)
            .expect("sampler")
            .into_stacked();
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
        let n = 100_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 7)
            .expect("sampler")
            .into_stacked();
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
        let n = 100_000;
        let s = sample_truncated_gaussian_posterior(&center, &start, &h, 1.0, &c, n, 424242)
            .expect("sampler")
            .into_stacked();
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
        let n = 100_000;
        let sqrt_phi = 2.0; // φ = 4 → σ = sqrt(φ/h) = 2.
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, sqrt_phi, &c, n, 99)
            .expect("sampler")
            .into_stacked();
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
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 31337)
            .expect("sampler")
            .into_stacked();
        assert_eq!(s.dim(), (n * NUTS_CHAINS, p));
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
        let n = 40_000;
        let s = sample_truncated_gaussian_posterior(&mode, &mode, &h, 1.0, &c, n, 5)
            .expect("sampler")
            .into_stacked();
        assert_all_feasible(&s, &c);
        assert!(s.column(0).iter().all(|&v| v > 0.0 && v < 1.0));
        // Symmetric truncation around the centred mode ⇒ mean ≈ 0.5.
        assert!((s.column(0).mean().unwrap() - 0.5).abs() < 0.01);
    }

    /// #3380: neither the chain starts nor the burn-in may leak into the kept
    /// draws. With the mode ON the bound of a 1-D truncated normal `N(−1, 1)`,
    /// `β ≥ 0`, and the fewest kept draws, the replicate draw means over
    /// independent seeds must average to the closed-form truncated mean
    /// `−1 + φ(1)/(1−Φ(1))`. Replicates are independent, so their own spread
    /// gives the standard error with no assumed autocorrelation time, and the
    /// two-sided z level is the one a correct sampler exceeds with probability
    /// `1/R` over `R` replicates. Chains kept from the boundary mode average
    /// about `0.35` here, some 17 standard errors low.
    #[test]
    fn boundary_mode_does_not_bias_the_mean_of_few_draws() {
        use statrs::distribution::{Continuous, ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).expect("standard normal");
        let expect = -1.0 + normal.pdf(1.0) / (1.0 - normal.cdf(1.0));
        let c = constraints(array![[1.0]], array![0.0]); // β ≥ 0
        let replicates = 400_u64;
        let means: Vec<f64> = (0..replicates)
            .map(|seed| {
                let draws = sample_truncated_gaussian_posterior(
                    &array![-1.0],
                    &array![0.0],
                    &array![[1.0]],
                    1.0,
                    &c,
                    4,
                    seed,
                )
                .expect("sampler");
                assert!(draws.warmup_transitions > 0, "the chains ran no burn-in");
                draws.chains.mean().expect("draws")
            })
            .collect();
        let r = replicates as f64;
        let grand = means.iter().sum::<f64>() / r;
        let spread = (means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (r - 1.0)).sqrt();
        let z = (grand - expect) / (spread / r.sqrt());
        let level = normal.inverse_cdf(1.0 - 0.5 / r);
        assert!(
            z.abs() < level,
            "mean of {replicates} four-draw replicates {grand} vs truncated mean {expect}: \
             z = {z} beyond the {level} level"
        );
    }
}
