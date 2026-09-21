//! The exact Hamiltonian flow of a whitened Gaussian inside a polytope, and the
//! split chain that samples a non-Gaussian density on the same polytope with it.
//!
//! # The flow
//!
//! Pakman & Paninski (2014, *"Exact Hamiltonian Monte Carlo for Truncated
//! Multivariate Gaussians"*, J. Comput. Graph. Statist.). After whitening to a
//! standard normal target the Hamiltonian trajectory is the exactly integrable
//! harmonic oscillator `z(t) = z₀ cos t + v₀ sin t`; the particle travels along
//! that arc and reflects specularly off each linear wall `fᵢᵀz + gᵢ ≥ 0` it
//! reaches. The map preserves the truncated Gaussian exactly: there is no step
//! size and no rejection. Travelling a quarter period [`TRAVEL_TIME`] between
//! velocity refreshes gives `z(π/2) = v₀` when no wall intervenes, an
//! independent draw. `gam-inference`'s truncated-Gaussian posterior sampler runs
//! this flow.
//!
//! # The split chain (#2901 V20)
//!
//! A cone-constrained posterior `π(β) ∝ exp(ℓ(β) − ½βᵀSβ)·1{Aβ ≥ b}` is not the
//! truncated Gaussian its Laplace expansion gives where the likelihood is skewed
//! near a wall. Whitening by the Laplace precision about its unconstrained centre
//! splits the potential `U(z) = −log π(β(z))` into the quadratic `½|z|²` and a
//! residual `R(z) = U(z) − ½|z|²`, constant exactly when the posterior is that
//! truncated Gaussian. The quadratic moves by the exact flow, the residual enters
//! as Strang kicks `v ← v − (τ/2)∇R` with `τ` one quarter period divided evenly
//! among the kicks, and a Metropolis step on the total energy `U + ½|v|²` keeps
//! the chain exact for any number of kicks (Shahbaba, Lan, Johnson & Neal 2014,
//! *"Split Hamiltonian Monte Carlo"*, Stat. Comput.).

use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2};

/// Quarter-period travel time between velocity refreshes. With no active wall,
/// `z(π/2) = v₀`, so consecutive draws decorrelate completely.
pub const TRAVEL_TIME: f64 = std::f64::consts::FRAC_PI_2;

/// A reflection budget per trajectory. A pointed feasible cone resolves a
/// vertex start in `O(#active rows)` bounces; this cap is a backstop against a
/// pathological grazing cycle. Exhaustion is an error: stopping at a wall
/// instead of completing the fixed travel time does not preserve the target.
const MAX_BOUNCES_BASE: usize = 256;

/// The polytope `{β : Aβ ≥ b}` in the whitened coordinates of a Gaussian
/// `N(centre, scale²·H⁻¹)`, `H = LLᵀ`: walls `fᵢᵀz + gᵢ ≥ 0` with
/// `fᵢ = scale·L⁻¹aᵢ` and `gᵢ = aᵢᵀcentre − bᵢ`, each row normalized to a unit
/// normal so that positive row rescalings give the same reflections. The centre
/// may be infeasible (`gᵢ < 0`); only a chain's position must satisfy every wall.
pub struct WhitenedPolytope {
    normals: Array2<f64>,
    offsets: Array1<f64>,
    normal_sq_norms: Vec<f64>,
    max_bounces: usize,
}

impl WhitenedPolytope {
    /// `lower` is the lower Cholesky factor `L` of `H`, `scale` the Gaussian's
    /// scale `√φ`, `centre` its pre-truncation mean.
    pub fn new(
        lower: &Array2<f64>,
        scale: f64,
        centre: &Array1<f64>,
        constraints: &LinearInequalityConstraints,
    ) -> Result<Self, String> {
        let p = centre.len();
        if lower.dim() != (p, p) {
            return Err(format!(
                "whitened polytope: the factor is {:?}, expected ({p}, {p})",
                lower.dim()
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "whitened polytope: non-positive or non-finite scale ({scale})"
            ));
        }
        let a = &constraints.a;
        let b = &constraints.b;
        let m = a.nrows();
        if m != b.len() {
            return Err(format!(
                "whitened polytope: constraint row mismatch (A has {m} rows, b has {})",
                b.len(),
            ));
        }
        if m > 0 && a.ncols() != p {
            return Err(format!(
                "whitened polytope: constraint matrix has {} columns, expected {p}",
                a.ncols(),
            ));
        }
        if m == 0 {
            return Ok(Self {
                normals: Array2::zeros((0, p)),
                offsets: Array1::zeros(0),
                normal_sq_norms: Vec::new(),
                max_bounces: MAX_BOUNCES_BASE,
            });
        }
        // `forward_substitution_lower_matrix` solves `L M = Aᵀ` column by column,
        // giving `M = L⁻¹Aᵀ` (`p × m`), so the whitened rows are `scale·Mᵀ`.
        let at = a.t().to_owned();
        let mut normals =
            gam_linalg::triangular::forward_substitution_lower_matrix(lower, &at).reversed_axes();
        normals.mapv_inplace(|value| value * scale);
        let mut offsets = a.dot(centre) - b;
        // Normalize in two stages before squaring: raw row norms can overflow or
        // underflow even when every coefficient and the unit normal is finite.
        for i in 0..m {
            let row_scale = normals.row(i).iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
            if row_scale > 0.0 {
                normals.row_mut(i).mapv_inplace(|v| v / row_scale);
                offsets[i] /= row_scale;
                let row_norm = normals.row(i).dot(&normals.row(i)).sqrt();
                normals.row_mut(i).mapv_inplace(|v| v / row_norm);
                offsets[i] /= row_norm;
            }
        }
        let normal_sq_norms = (0..m).map(|i| normals.row(i).dot(&normals.row(i))).collect();
        Ok(Self {
            normals,
            offsets,
            normal_sq_norms,
            max_bounces: MAX_BOUNCES_BASE + 8 * m,
        })
    }

    /// Whether `z` satisfies every wall.
    pub fn contains(&self, z: &Array1<f64>) -> bool {
        (0..self.normals.nrows()).all(|i| self.normals.row(i).dot(z) + self.offsets[i] >= 0.0)
    }

    /// Advance `(z, v)` along the harmonic trajectory for `time`, reflecting
    /// specularly off every wall it reaches, and return the reflection count. On
    /// return `z` is the new (feasible) position.
    pub fn flow(&self, z: &mut Array1<f64>, v: &mut Array1<f64>, time: f64) -> Result<usize, String> {
        let m = self.normals.nrows();
        let mut t_left = time;
        let mut bounces = 0usize;
        loop {
            if t_left <= 0.0 {
                return Ok(bounces);
            }
            // Find the first wall hit within (0, t_left].
            let mut hit_time = t_left;
            let mut hit_wall: Option<usize> = None;
            for i in 0..m {
                let fi = self.normals.row(i);
                let u = fi.dot(z); // fᵢᵀ z   (so cᵢ(0) = u + gᵢ)
                let w = fi.dot(v); // fᵢᵀ v
                if let Some(t) = first_wall_hit(u, w, self.offsets[i], hit_time) {
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
                    return Ok(bounces);
                }
                Some(j) => {
                    advance(z, v, hit_time);
                    t_left -= hit_time;
                    // Specular reflection of the velocity about the wall normal fⱼ:
                    //   v ← v − 2 (fⱼᵀ v / ‖fⱼ‖²) fⱼ,
                    // which flips the outward normal velocity component to inward.
                    let fj = self.normals.row(j);
                    let denom = self.normal_sq_norms[j];
                    if denom > 0.0 {
                        let coeff = 2.0 * fj.dot(v) / denom;
                        for k in 0..v.len() {
                            v[k] -= coeff * fj[k];
                        }
                    }
                    bounces += 1;
                    if bounces >= self.max_bounces && t_left > 0.0 {
                        return Err(format!(
                            "exact polytope flow: trajectory exhausted its {} reflection budget \
                             before completing its travel time",
                            self.max_bounces
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

/// The whitened potential `U(z) = −log π(β(z))` (up to a constant) and its
/// gradient `∇_z U`, or `None` where the density is zero, outside the
/// likelihood's support.
pub trait WhitenedPotential {
    fn evaluate(&mut self, z: &Array1<f64>) -> Result<Option<(f64, Array1<f64>)>, String>;
}

/// A split chain's position with its potential and the residual gradient
/// `∇R = ∇U − z`.
pub struct SplitChainState {
    z: Array1<f64>,
    potential: f64,
    residual_gradient: Array1<f64>,
}

impl SplitChainState {
    /// The chain at `z`, which must lie in the density's support.
    pub fn at<P: WhitenedPotential>(potential: &mut P, z: Array1<f64>) -> Result<Self, String> {
        let (value, gradient) = potential
            .evaluate(&z)?
            .ok_or_else(|| "split chain: the start is outside the density's support".to_string())?;
        let residual_gradient = &gradient - &z;
        Ok(Self {
            z,
            potential: value,
            residual_gradient,
        })
    }

    /// The chain's position.
    pub fn position(&self) -> &Array1<f64> {
        &self.z
    }
}

/// What one split trajectory did.
pub struct TrajectoryOutcome {
    pub accepted: bool,
    pub reflections: usize,
}

/// One split trajectory of total time [`TRAVEL_TIME`] with `kicks` Strang kicks
/// of the residual, accepted by Metropolis on the total energy. `velocity` is a
/// fresh standard-normal draw and `uniform` a uniform draw on `(0, 1]` for the
/// acceptance. A trajectory that leaves the density's support is rejected, and
/// the state is unchanged unless the trajectory is accepted.
pub fn split_trajectory<P: WhitenedPotential>(
    polytope: &WhitenedPolytope,
    potential: &mut P,
    state: &mut SplitChainState,
    kicks: usize,
    mut velocity: Array1<f64>,
    uniform: f64,
) -> Result<TrajectoryOutcome, String> {
    if kicks == 0 {
        return Err("split trajectory: at least one kick is needed".to_string());
    }
    let tau = TRAVEL_TIME / kicks as f64;
    let start_energy = state.potential + 0.5 * velocity.dot(&velocity);
    let mut z = state.z.clone();
    let mut residual_gradient = state.residual_gradient.clone();
    let mut value = state.potential;
    let mut reflections = 0usize;
    for _ in 0..kicks {
        velocity.scaled_add(-0.5 * tau, &residual_gradient);
        reflections += polytope.flow(&mut z, &mut velocity, tau)?;
        let Some((next_value, gradient)) = potential.evaluate(&z)? else {
            return Ok(TrajectoryOutcome {
                accepted: false,
                reflections,
            });
        };
        value = next_value;
        residual_gradient = &gradient - &z;
        velocity.scaled_add(-0.5 * tau, &residual_gradient);
    }
    let end_energy = value + 0.5 * velocity.dot(&velocity);
    let log_accept = start_energy - end_energy;
    let accepted = log_accept >= 0.0 || uniform.ln() < log_accept;
    if accepted {
        state.z = z;
        state.potential = value;
        state.residual_gradient = residual_gradient;
    }
    Ok(TrajectoryOutcome {
        accepted,
        reflections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
    fn reflection_budget_exhaustion_refuses_a_partial_trajectory() {
        let polytope = WhitenedPolytope {
            normals: array![[1.0], [-1.0]],
            offsets: array![0.0, 1.0],
            normal_sq_norms: vec![1.0, 1.0],
            max_bounces: 1,
        };
        let error = polytope
            .flow(&mut array![0.5], &mut array![2.0], TRAVEL_TIME)
            .expect_err("a partial arc would place spurious probability mass on its last wall");
        assert!(error.contains("reflection budget"));
    }

    /// The potential of `N(shift, I)` in whitened coordinates: its residual
    /// against the unit quadratic is linear, so a split chain must kick.
    struct ShiftedGaussian {
        shift: Array1<f64>,
    }

    impl WhitenedPotential for ShiftedGaussian {
        fn evaluate(&mut self, z: &Array1<f64>) -> Result<Option<(f64, Array1<f64>)>, String> {
            let deviation = z - &self.shift;
            Ok(Some((0.5 * deviation.dot(&deviation), deviation)))
        }
    }

    /// SplitMix64 uniforms and Box–Muller normals, deterministic.
    struct Draws(u64);

    impl Draws {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let radius = (-2.0 * self.uniform().ln()).sqrt();
            radius * (2.0 * std::f64::consts::PI * self.uniform()).cos()
        }
    }

    /// #2901 V20 positive control: a chain whitened about the wrong centre
    /// (residual `R` linear, not constant) still samples its target exactly.
    /// The target `N(0.8, 1)` truncated to `z ≥ 0.3` has mean
    /// `0.8 + φ(α)/(1 − Φ(α))` with `α = 0.3 − 0.8`; the chain's mean over its
    /// kept draws must sit within four of its own batch-means standard errors.
    #[test]
    fn a_split_chain_samples_a_truncated_gaussian_it_is_not_whitened_about_2901() {
        let lower = array![[1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.3]).expect("half-line");
        let polytope =
            WhitenedPolytope::new(&lower, 1.0, &array![0.0], &constraints).expect("polytope");
        let mut potential = ShiftedGaussian {
            shift: array![0.8],
        };
        let alpha = 0.3_f64 - 0.8;
        let density = (-0.5 * alpha * alpha).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let survival = 1.0 - gam_math::probability::normal_cdf(alpha);
        let truth = 0.8 + density / survival;
        for kicks in [1usize, 3] {
            let mut draws = Draws(0x2901_5B11 + kicks as u64);
            let mut state = SplitChainState::at(&mut potential, array![1.0]).expect("start");
            let batches = 40usize;
            let per_batch = 500usize;
            let mut batch_means = Vec::with_capacity(batches);
            let mut accepted = 0usize;
            for _ in 0..batches {
                let mut sum = 0.0;
                for _ in 0..per_batch {
                    let velocity = array![draws.normal()];
                    let uniform = draws.uniform();
                    let outcome = split_trajectory(
                        &polytope,
                        &mut potential,
                        &mut state,
                        kicks,
                        velocity,
                        uniform,
                    )
                    .expect("trajectory");
                    accepted += usize::from(outcome.accepted);
                    assert!(polytope.contains(state.position()), "a draw left the half-line");
                    sum += state.position()[0];
                }
                batch_means.push(sum / per_batch as f64);
            }
            let count = batch_means.len() as f64;
            let mean = batch_means.iter().sum::<f64>() / count;
            let variance = batch_means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (count - 1.0);
            let standard_error = (variance / count).sqrt();
            assert!(
                (mean - truth).abs() <= 4.0 * standard_error,
                "K={kicks}: chain mean {mean} against the truncated mean {truth} \
                 (standard error {standard_error:e}, acceptance {})",
                accepted as f64 / (batches * per_batch) as f64
            );
        }
    }
}
