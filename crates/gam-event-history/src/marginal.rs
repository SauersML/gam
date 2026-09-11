//! Marginal likelihood of one subject's event history over its latent chain,
//! with its gradient and Hessian in the coefficient space the subject sees:
//! the per-mark coefficients `β_d` (through the design rows of its nodes),
//! the loadings `a`, and the log-rates `ρ`.
//!
//! The log-intensity of mark `d` at a node is
//!
//! ```text
//! η_d(z) = η⁰_d − ½ Σ_k a_{dk}² + Σ_k a_{dk} z_k
//! ```
//!
//! The atoms are stationary and standard at every time, so
//! `E_z exp(Σ_k a_{dk} z_k) = exp(½ Σ_k a_{dk}²)` and the shift cancels it:
//! `exp(η⁰_d)` is the population-average intensity whatever the loadings,
//! and the latent term is the individual deviation from it. Without the
//! shift, raising the heterogeneity would raise the population rate unless
//! the baseline moved to compensate.
//!
//! The chain is marginalised by forward filtering on the adaptive product
//! Gauss-Hermite grid of [`super::chain`]; derivatives come from the two
//! identities that hold for any latent-variable model whose complete-data
//! log-likelihood `L_c` is a sum of node terms and gap terms:
//!
//! ```text
//! ∂ℓ/∂θ     = E[∂L_c/∂θ | y]                                   (Fisher)
//! ∂²ℓ/∂θ∂θᵀ = E[∂²L_c/∂θ∂θᵀ | y] + Cov(∂L_c/∂θ | y)             (Louis)
//! ```
//!
//! The covariance is of the complete-data score `g = Σ_n v_n(z_n) + Σ_g
//! t_g(z_g, z_{g+1})`, a sum of node and gap functions. Its second moment is
//! accumulated in one forward sweep: by the Markov property, the past of the
//! chain given `z_m` is independent of the data after `m`, so the running
//! conditional expectation `C_m(z) = E[Σ_{n≤m} v_n + Σ_{g<m} t_g | z_m = z,
//! y_{≤m}]` is enough to form `E[(Σ_{n<m} v_n) v_mᵀ | y]` as an expectation
//! under the smoothed marginal at `m`, and it propagates forward through
//! the same separable transition operators the filter uses:
//! `C_{m+1} = F[α_m C_m] / p̂_{m+1}`. Nothing of size `S × S` is ever stored
//! per gap and no pair table over nodes is formed: the cost is linear in the
//! node count and in the coefficient count, and the Hessian is assembled in
//! coefficient space directly.
//!
//! Every function this pass carries is bounded: a score, a smoothed
//! conditional expectation, or a polynomial moment of the transition. Only
//! the smoother residual `log β` is interpolated, on cubic splines, and only
//! its expectations under the smoothed marginal enter. The Hessian is
//! therefore Louis' identity evaluated by the same quadrature as the value:
//! it agrees with the second derivative of the computed value to the
//! quadrature error the fit's certificate bounds, while the gradient the
//! inner Newton uses is the exact derivative of the computed value (see
//! [`super::family`]).

use super::chain::{
    AtomTransition, GaussHermite, Grid, OperatorFamily, backward_axis_bases, forward_operators,
    interpolate_at_inner_points, log_sum_exp, normal_density,
};
use super::cohort::{EventHistoryError, SubjectNodes};
use super::scalar::{add_real, exp, ln, recip, sqrt, square};
use gam_math::nested_dual::JetField;
use ndarray::ArrayView2;
use std::collections::HashMap;

/// Everything one subject's marginal needs, in the caller's scalar type.
pub(crate) struct SubjectInputs<'a, S> {
    pub nodes: &'a SubjectNodes,
    /// Population node log-intensities `η⁰`, index `n * marks + d`.
    pub eta0: &'a [S],
    /// Loadings, index `d * atoms + k`.
    pub loadings: &'a [S],
    /// The dimensionless rate `ν = rate · time_scale` per atom. This is the
    /// coefficient the fit carries: near zero the likelihood is smooth in it
    /// with finite curvature, where its logarithm would be flat, so a static
    /// frailty is a wall the coefficient can sit on rather than a plateau it
    /// runs along.
    pub rates: &'a [S],
    pub time_scale: f64,
    pub gh: &'a GaussHermite,
    /// Elapsed time between a supplied filtered state and the first node;
    /// zero unless [`forward_filter`] continues from an earlier state.
    pub continuation_gap: f64,
    /// Per mark, the design rows of this subject's nodes (`n × p_d`), so the
    /// derivatives come out in coefficient space. `None` uses the identity:
    /// the node log-intensities themselves are the parameters.
    pub designs: Option<&'a [ArrayView2<'a, f64>]>,
    /// The risk-set normaliser `log M_d(t)` at every node, index
    /// `n * marks + d`. `None` centres on the stationary prior instead, the
    /// constant `½|a_d|²` (see `super::preserve`).
    pub log_normaliser: Option<&'a [S]>,
}

/// Marginal log-likelihood and its derivatives in the subject-local parameter
/// vector `[β_0 | … | β_{D−1} | a (marks × atoms) | ν (atoms)]`.
pub(crate) struct SubjectOutput<S> {
    pub loglik: S,
    /// The Fisher-identity gradient `E[∂L_c/∂θ | y]`; empty when derivatives
    /// were not requested.
    pub gradient: Vec<S>,
    /// Row-major `P × P` log-likelihood Hessian; empty when derivatives were
    /// not requested.
    pub hessian: Vec<S>,
}

/// A filtered or predicted density value below this fraction of its grid's
/// peak is interpolation noise (the Lagrange-based operator is a signed sum,
/// accurate to about `ε · peak` in absolute terms), and the future-likelihood
/// ratio that multiplies it in the smoothed marginal is exponentially large
/// exactly there. Such points carry no smoothed mass, and a smoothed
/// conditional expectation is not formed there (its denominator is the
/// predicted density). `1e-11` sits several orders above the noise of a
/// product grid of a few hundred points; the mass it discards is far below
/// the Gauss-Hermite certificate's tolerance, and the smoothed marginal is
/// renormalised after the cut so every expectation is under a probability.
const DENSITY_NOISE_RELATIVE: f64 = 1e-11;

/// The marker a lost-positivity failure carries. The grid representation of
/// a density is a signed interpolant, so an order too coarse for a posterior
/// can produce negative mass where the mass is; that is a failure of the
/// representation, not of the model, and the fit driver answers it by
/// raising the Gauss-Hermite order rather than by passing the number on.
pub(crate) const LOST_POSITIVITY: &str = "the density representation lost positivity";

/// The noise floor of a density on a grid: `DENSITY_NOISE_RELATIVE` of its
/// largest value.
fn density_floor<S: JetField>(values: &[S]) -> f64 {
    DENSITY_NOISE_RELATIVE * values.iter().map(|v| v.value()).fold(0.0, f64::max)
}

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// `−½ Σ_k a_{dk}²` for one mark's loadings: the shift that makes
/// `exp(η⁰_d)` the population-average intensity (see the module docs).
pub(crate) fn marginal_shift<S: JetField>(loadings_d: &[S], like: &S) -> S {
    loadings_d.iter().fold(like.constant_like(0.0), |acc, a| {
        acc.sub(&square(a).scale(0.5))
    })
}

/// The centred baseline of mark `d`: `η⁰ − log M_d`, with `log M_d` the
/// normaliser that makes `exp(η⁰)` the intensity averaged over the declared
/// population. Without one the population is the stationary prior and the
/// normaliser is the constant `½|a_d|²`; with one it is the risk set's, and
/// it is a function of time (see `super::preserve`).
pub(crate) fn centred_baseline<S: JetField>(
    eta0: &S,
    loadings_d: &[S],
    log_normaliser: Option<&S>,
) -> S {
    match log_normaliser {
        Some(shift) => eta0.sub(shift),
        None => eta0.add(&marginal_shift(loadings_d, eta0)),
    }
}

/// The log-intensity of mark `d` at latent state `z`: `η⁰ − log M_d + a_d · z`.
pub(crate) fn log_intensity<S: JetField>(
    eta0: &S,
    loadings_d: &[S],
    z: &[S],
    log_normaliser: Option<&S>,
) -> S {
    let mut eta = centred_baseline(eta0, loadings_d, log_normaliser);
    for (a, zk) in loadings_d.iter().zip(z.iter()) {
        eta = eta.add(&a.mul(zk));
    }
    eta
}

/// The transitions of every atom across a gap of `gap` time units.
///
/// A rate whose `κ = ν · gap / T` is not positive would make the innovation
/// variance zero and the transition a singular Gaussian; the fit keeps `ν`
/// inside the band its breakpoints resolve, so reaching it is a numerical failure
/// to report, not a limit to take. The transition's own derivative fields
/// are in the log-rate, the coordinate in which the gap scores stay bounded
/// as a gap shrinks; the marginal converts them to `ν` at the end.
pub(crate) fn transitions_across<S: JetField>(
    rates: &[S],
    gap: f64,
    time_scale: f64,
) -> Result<Vec<AtomTransition<S>>, EventHistoryError> {
    rates
        .iter()
        .map(|nu| {
            let kappa = nu.scale(gap / time_scale);
            if !(kappa.value().is_finite() && kappa.value() >= 0.0) {
                return Err(numerical(format!(
                    "atom transition across a gap of {gap}: rate · gap = {} is not nonnegative and finite (rate {})",
                    kappa.value(),
                    nu.value()
                )));
            }
            Ok(AtomTransition::new(&kappa))
        })
        .collect()
}

/// A bivariate polynomial in the start and end coordinates of one atom
/// across one gap, degree at most four in each variable.
#[derive(Clone)]
struct GapPolynomial<S> {
    /// `c[a * 5 + b]` multiplies `z^a u^b`.
    c: Vec<S>,
    /// Structural support: which coefficients were ever set. A coefficient
    /// whose value happens to be zero may still carry derivative channels,
    /// so sparsity is tracked by construction, never read off the value.
    present: Vec<bool>,
}

impl<S: JetField> GapPolynomial<S> {
    fn zero(like: &S) -> Self {
        Self {
            c: vec![like.constant_like(0.0); 25],
            present: vec![false; 25],
        }
    }

    fn set(&mut self, a: usize, b: usize, value: S) {
        self.c[a * 5 + b] = value;
        self.present[a * 5 + b] = true;
    }

    fn get(&self, a: usize, b: usize) -> &S {
        &self.c[a * 5 + b]
    }

    /// Whether the coefficient of `z^a u^b` is structurally absent.
    fn absent(&self, a: usize, b: usize) -> bool {
        !self.present[a * 5 + b]
    }

    fn scaled(&self, factor: &S) -> Self {
        Self {
            c: self.c.iter().map(|v| v.mul(factor)).collect(),
            present: self.present.clone(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            c: self
                .c
                .iter()
                .zip(other.c.iter())
                .map(|(x, y)| x.add(y))
                .collect(),
            present: self
                .present
                .iter()
                .zip(other.present.iter())
                .map(|(x, y)| x | y)
                .collect(),
        }
    }

    /// Product, exact when the total degree stays at most four per variable.
    fn mul(&self, other: &Self) -> Self {
        let zero = self.c[0].constant_like(0.0);
        let mut out = Self {
            c: vec![zero; 25],
            present: vec![false; 25],
        };
        for a in 0..5 {
            for b in 0..5 {
                if self.absent(a, b) {
                    continue;
                }
                let x = self.get(a, b);
                for a2 in 0..5 - a {
                    for b2 in 0..5 - b {
                        if other.absent(a2, b2) {
                            continue;
                        }
                        let y = other.get(a2, b2);
                        let idx = (a + a2) * 5 + (b + b2);
                        out.c[idx] = out.c[idx].add(&x.mul(y));
                        out.present[idx] = true;
                    }
                }
            }
        }
        out
    }
}

/// The score `t = ∂ ln p(z'|z) / ∂ρ` of one atom across one gap and its own
/// derivative `∂t/∂ρ`, as polynomials in the start state `z` and the
/// standardised innovation `u = (z' − φz)/√(1 − φ²)`.
///
/// In these coordinates every coefficient stays bounded as the gap shrinks
/// (`κ → 0`, `φ → 1`, `q → 0`): each is a product of a `1/q` power with the
/// matching power of `dφ/dρ = −κφ ∝ q`, and products lose no precision. The
/// monomial expansion in `(z, z')` carries coefficients of order `1/q²` that
/// cancel to `O(1)` and destroy the quadrature moments in floating point.
fn gap_score_polynomials<S: JetField>(
    transition: &AtomTransition<S>,
    like: &S,
) -> (GapPolynomial<S>, GapPolynomial<S>) {
    let phi = &transition.phi;
    let v = &transition.innovation;
    let inv_v = recip(v);
    let inv_root_v = sqrt(&inv_v);
    let inv_v2 = square(&inv_v);
    let phi2 = square(phi);
    // dL/dφ = (φ/v)(1 − u²) + z u / √v
    let mut dl = GapPolynomial::zero(like);
    dl.set(0, 0, phi.mul(&inv_v));
    dl.set(0, 2, phi.mul(&inv_v).neg());
    dl.set(1, 1, inv_root_v.clone());
    // d²L/dφ² = (1+φ²)/v² − z²/v + 4φ z u / v^{3/2} − (1+3φ²) u²/v²
    let mut d2l = GapPolynomial::zero(like);
    d2l.set(0, 0, add_real(&phi2, 1.0).mul(&inv_v2));
    d2l.set(2, 0, inv_v.neg());
    d2l.set(1, 1, phi.mul(&inv_v).mul(&inv_root_v).scale(4.0));
    d2l.set(0, 2, add_real(&phi2.scale(3.0), 1.0).mul(&inv_v2).neg());
    let t = dl.scaled(&transition.dphi);
    let dt = d2l
        .scaled(&square(&transition.dphi))
        .add(&dl.scaled(&transition.d2phi));
    (t, dt)
}

/// The score `∂ ln p(z'|z)/∂ρ` of one atom across a gap of dimensionless
/// length `kappa`, and its derivative in `ρ`, as flat coefficient vectors
/// `c[a * 5 + b]` multiplying `z^a u^b` with `u = (z' − φz)/√(1 − φ²)`.
pub fn transition_score_polynomials(kappa: f64) -> (Vec<f64>, Vec<f64>) {
    let transition = AtomTransition::new(&kappa);
    let (t, dt) = gap_score_polynomials(&transition, &0.0);
    (t.c, dt.c)
}

pub(crate) fn weighted_sum<S: JetField>(weights: &[S], values: &[S]) -> S {
    weights
        .iter()
        .zip(values.iter())
        .fold(weights[0].constant_like(0.0), |acc, (w, v)| {
            acc.add(&w.mul(v))
        })
}

/// Pairwise (tree) sum of `terms`: rounding error grows like `log₂ n`
/// rather than `n`, which keeps a log-likelihood summed over thousands of
/// node terms resolvable at the level a Newton acceptance test needs.
pub(crate) fn pairwise_sum<S: JetField>(terms: &[S], zero: &S) -> S {
    match terms.len() {
        0 => zero.clone(),
        1 => terms[0].clone(),
        2 => terms[0].add(&terms[1]),
        n => {
            let (left, right) = terms.split_at(n / 2);
            pairwise_sum(left, zero).add(&pairwise_sum(right, zero))
        }
    }
}

/// Node log-likelihood pieces at one grid.
pub(crate) struct NodeLikelihood<S> {
    /// The node term's score in `η_d` at every grid point, index
    /// `d * size + i`: `∂/∂η_d` of the mark's contribution, which for a
    /// counting mark is `y − w e^{η}`.
    pub score: Vec<S>,
    /// Its curvature in `η_d`, `−∂²/∂η_d²`, at every grid point: `w e^{η}`
    /// for a counting mark. Zero exactly where the mark contributes nothing
    /// at this node.
    pub curvature: Vec<S>,
    /// Whether each mark's curvature is anything but zero at this node, so
    /// the accumulation can skip the marks that carry no term.
    pub informative: Vec<bool>,
    /// `Σ_d` of the mark contributions at every grid point.
    pub ell: Vec<S>,
    /// `max_i ell[i].value()`.
    pub shift: f64,
}

pub(crate) fn node_likelihood<S: JetField>(
    grid: &Grid<S>,
    eta0: &[S],
    loadings: &[S],
    counts: &[f64],
    exposures: &[f64],
    compensated: Option<&[bool]>,
    log_normaliser: Option<&[S]>,
    marks: usize,
    atoms: usize,
    derivatives: bool,
) -> NodeLikelihood<S> {
    let size = grid.size();
    let zero = eta0[0].constant_like(0.0);
    // The score and curvature are read only by the accumulation that forms
    // the gradient and Hessian. A grid placement pass, or a filter run for a
    // forecast, needs the node's value alone — and these two vectors are
    // `marks × size` of the caller's scalar, which for a seeded dual is the
    // largest thing the node touches.
    let width = if derivatives { marks * size } else { 0 };
    let mut score = Vec::with_capacity(width);
    let mut curvature = Vec::with_capacity(width);
    let mut informative = Vec::with_capacity(marks);
    let mut ell = vec![zero.clone(); size];
    for d in 0..marks {
        let exposure = if compensated.is_none_or(|mask| mask[d]) {
            exposures[d]
        } else {
            0.0
        };
        let loadings_d = &loadings[d * atoms..(d + 1) * atoms];
        let base = centred_baseline(&eta0[d], loadings_d, log_normaliser.map(|m| &m[d]));
        let y = counts[d];
        informative.push(exposure != 0.0);
        // Cache the axis contributions in log space. Exponentiating them
        // separately can produce 0 * infinity for a finite combined rate.
        let latent: Option<Vec<Vec<S>>> = (exposure != 0.0).then(|| {
            (0..atoms)
                .map(|k| {
                    grid.axes[k]
                        .points
                        .iter()
                        .map(|z| loadings_d[k].mul(z))
                        .collect()
                })
                .collect()
        });
        for i in 0..size {
            if y != 0.0 {
                let mut eta = base.clone();
                for (k, a) in loadings_d.iter().enumerate() {
                    eta = eta.add(&a.mul(grid.coordinate(i, k)));
                }
                ell[i] = ell[i].add(&eta.scale(y));
            }
            // A mark with no exposure at this node has no compensator and so
            // no curvature: its intensity is never formed, which is the work
            // the risk sets save.
            match &latent {
                Some(tables) => {
                    let mut log_c = base.clone();
                    for (k, table) in tables.iter().enumerate() {
                        log_c = log_c.add(&table[grid.index(i, k)]);
                    }
                    let c = exp(&add_real(&log_c, exposure.ln()));
                    ell[i] = ell[i].sub(&c);
                    if derivatives {
                        score.push(add_real(&c.scale(-1.0), y));
                        curvature.push(c);
                    }
                }
                _ => {
                    if derivatives {
                        score.push(zero.constant_like(y));
                        curvature.push(zero.clone());
                    }
                }
            }
        }
    }
    let shift = ell
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);
    NodeLikelihood {
        score,
        curvature,
        informative,
        ell,
        shift,
    }
}

/// Posterior mean and variance of every atom under `alpha` on `grid`.
///
/// A non-positive variance means the grid representation of the density has
/// lost positivity where its mass is (a signed interpolant whose negative
/// lobes carry weight): that is a numerical failure of the representation,
/// reported as such rather than floored into a plausible small number.
fn posterior_moments<S: JetField>(
    grid: &Grid<S>,
    alpha: &[S],
    label: &str,
) -> Result<(Vec<S>, Vec<S>), EventHistoryError> {
    let atoms = grid.dimension();
    let mut means = Vec::with_capacity(atoms);
    let mut variances = Vec::with_capacity(atoms);
    for k in 0..atoms {
        let mut mean = alpha[0].constant_like(0.0);
        for i in 0..grid.size() {
            mean = mean.add(&grid.weights[i].mul(&alpha[i]).mul(grid.coordinate(i, k)));
        }
        let mut variance = alpha[0].constant_like(0.0);
        for i in 0..grid.size() {
            let centred = square(&grid.coordinate(i, k).sub(&mean));
            variance = variance.add(&grid.weights[i].mul(&alpha[i]).mul(&centred));
        }
        if !(variance.value() > 0.0) || !variance.value().is_finite() {
            return Err(numerical(format!(
                "{label}: posterior variance of atom {k} is {} on the grid; {LOST_POSITIVITY}",
                variance.value()
            )));
        }
        means.push(mean);
        variances.push(variance);
    }
    Ok((means, variances))
}

/// Multiply a predicted density on `grid` by the node factor
/// `exp(ell − shift)` and normalise; returns the filtered density and the
/// normaliser `c`.
pub(crate) fn condition<S: JetField>(
    grid: &Grid<S>,
    predicted: &[S],
    ell: &[S],
    shift: f64,
    label: &str,
) -> Result<(Vec<S>, S), EventHistoryError> {
    let mut raw: Vec<S> = predicted
        .iter()
        .zip(ell.iter())
        .map(|(p, e)| p.mul(&exp(&add_real(e, -shift))))
        .collect();
    let c = weighted_sum(&grid.weights, &raw);
    if !(c.value() > 0.0) || !c.value().is_finite() {
        return Err(numerical(format!(
            "{label}: normaliser is not positive ({})",
            c.value()
        )));
    }
    let inverse = recip(&c);
    for v in raw.iter_mut() {
        *v = v.mul(&inverse);
    }
    Ok((raw, c))
}

/// One filtered node: its grid, the operators that reached it, the predicted
/// and filtered densities on it, and the node's likelihood pieces.
pub(crate) struct FilteredNode<S> {
    pub grid: Grid<S>,
    /// Transitions across the gap that led here; empty at the first node.
    pub transitions: Vec<AtomTransition<S>>,
    /// Forward operators from the previous grid; empty at the first node.
    pub forward: OperatorFamily<S>,
    pub predicted: Vec<S>,
    pub alpha: Vec<S>,
    pub normaliser: S,
    pub likelihood: NodeLikelihood<S>,
}

/// Where a node's grid goes: at the posterior mean, with the predictive
/// spread.
///
/// The forward operator is exact for `envelope × polynomial`, so what it
/// interpolates is the filtered density divided by the grid's Gaussian
/// envelope. On a grid placed at the *predicted* moments that ratio is the
/// node's likelihood factor itself — after an event, an exponential tilt
/// `exp(a z)` — and a degree-`G−1` interpolant of an exponential across a
/// hull of `±x_max √2 σ` has truncation error that grows like
/// `(a · hull)^G / G!`, astronomically large at the orders the quadrature
/// certificate reaches. Re-centring the grid on the posterior mean absorbs
/// the tilt into the envelope (a tilted Gaussian is a shifted Gaussian).
///
/// The spread stays the predictive one. The Poisson node factor is
/// log-concave with at most linear growth in `z`, so the posterior is
/// dominated by a shifted Gaussian of the predictive variance: against that
/// envelope the ratio is a bounded, mild tilt, and both the interpolation and
/// the quadrature are benign at any order. Scaling the envelope down to the
/// posterior variance would leave the integrand's Gaussian tails outside it
/// and break the quadrature instead (a survival forecast above one is the
/// symptom). The variance still narrows across nodes, through the predictive
/// recursion. A node whose likelihood factor is much sharper than the
/// predictive spread (a large integrated intensity at one node) makes the
/// ratio a narrow bump that a polynomial resolves only at high order; the
/// mesh refinement of the fit is what keeps every node mildly informative.
pub(crate) fn filter_start<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    atoms: usize,
    node_terms: &dyn Fn(&Grid<S>, bool) -> NodeLikelihood<S>,
    derivatives: bool,
    label: &str,
) -> Result<FilteredNode<S>, EventHistoryError> {
    let zero: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
    let unit: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
    let prior_density = |grid: &Grid<S>| -> Vec<S> {
        (0..grid.size())
            .map(|i| {
                let mut density = like.constant_like(1.0);
                for k in 0..atoms {
                    density =
                        density.mul(&normal_density(grid.coordinate(i, k), &zero[k], &unit[k]));
                }
                density
            })
            .collect()
    };
    let prior_grid = Grid::new(gh, &zero, &unit, like);
    let rough = node_terms(&prior_grid, false);
    let (rough_alpha, _) = condition(
        &prior_grid,
        &prior_density(&prior_grid),
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&prior_grid, &rough_alpha, label)?;
    let grid = Grid::new(gh, &means, &unit, like);
    let predicted = prior_density(&grid);
    let likelihood = node_terms(&grid, derivatives);
    let (alpha, normaliser) =
        condition(&grid, &predicted, &likelihood.ell, likelihood.shift, label)?;
    Ok(FilteredNode {
        grid,
        transitions: Vec::new(),
        forward: OperatorFamily {
            per_axis: Vec::new(),
            order: gh.order,
        },
        predicted,
        alpha,
        normaliser,
        likelihood,
    })
}

/// The predictive grid across one gap (centre `φ · mean`, spread
/// `√(φ² var + q)`) and the predicted density on it.
pub(crate) fn predict<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    previous_grid: &Grid<S>,
    previous_alpha: &[S],
    transitions: &[AtomTransition<S>],
    label: &str,
) -> Result<(Grid<S>, Vec<S>), EventHistoryError> {
    let atoms = transitions.len();
    if atoms > 0 && transitions.iter().all(|t| t.innovation.value() == 0.0) {
        return Ok((previous_grid.clone(), previous_alpha.to_vec()));
    }
    let (means, variances) = posterior_moments(previous_grid, previous_alpha, label)?;
    let centres: Vec<S> = (0..atoms)
        .map(|k| transitions[k].phi.mul(&means[k]))
        .collect();
    let scales: Vec<S> = (0..atoms)
        .map(|k| {
            sqrt(
                &square(&transitions[k].phi)
                    .mul(&variances[k])
                    .add(&transitions[k].innovation),
            )
        })
        .collect();
    let predictive = Grid::new(gh, &centres, &scales, like);
    let forward = forward_operators(gh, previous_grid, &predictive, transitions, 0);
    let predicted = forward.plain(previous_alpha);
    Ok((predictive, predicted))
}

/// Predict the filtered state on `previous_grid` across one gap and condition
/// on a node, on a grid placed at the posterior mean with the predictive
/// spread (see [`filter_start`]).
pub(crate) fn filter_step<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    previous_grid: &Grid<S>,
    previous_alpha: &[S],
    transitions: Vec<AtomTransition<S>>,
    forward_power: u8,
    node_terms: &dyn Fn(&Grid<S>, bool) -> NodeLikelihood<S>,
    derivatives: bool,
    label: &str,
) -> Result<FilteredNode<S>, EventHistoryError> {
    if !transitions.is_empty() && transitions.iter().all(|t| t.innovation.value() == 0.0) {
        let likelihood = node_terms(previous_grid, derivatives);
        let (alpha, normaliser) = condition(previous_grid, previous_alpha, &likelihood.ell,
            likelihood.shift, label)?;
        return Ok(FilteredNode { grid: previous_grid.clone(), transitions,
            forward: OperatorFamily { per_axis: Vec::new(), order: gh.order },
            predicted: previous_alpha.to_vec(), alpha, normaliser, likelihood });
    }
    let (predictive, rough_predicted) =
        predict(gh, like, previous_grid, previous_alpha, &transitions, label)?;
    let rough = node_terms(&predictive, false);
    let (rough_alpha, _) = condition(
        &predictive,
        &rough_predicted,
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&predictive, &rough_alpha, label)?;
    let scales: Vec<S> = predictive
        .axes
        .iter()
        .map(|axis| axis.sigma.clone())
        .collect();
    let grid = Grid::new(gh, &means, &scales, like);
    let forward = forward_operators(gh, previous_grid, &grid, &transitions, forward_power);
    let predicted = forward.plain(previous_alpha);
    let likelihood = node_terms(&grid, derivatives);
    let (alpha, normaliser) =
        condition(&grid, &predicted, &likelihood.ell, likelihood.shift, label)?;
    Ok(FilteredNode {
        grid,
        transitions,
        forward,
        predicted,
        alpha,
        normaliser,
        likelihood,
    })
}

/// Coefficient layout of one subject's local parameter vector.
struct Layout {
    /// Offset of mark `d`'s coefficient block.
    offsets: Vec<usize>,
    atoms: usize,
    /// Offset of the loading slots.
    a0: usize,
    /// Offset of the log-rate slots.
    rho0: usize,
    total: usize,
}

impl Layout {
    fn a(&self, d: usize, k: usize) -> usize {
        self.a0 + d * self.atoms + k
    }
    fn rho(&self, k: usize) -> usize {
        self.rho0 + k
    }
}

/// Evaluate one subject's marginal log-likelihood and, when requested, its
/// Fisher gradient and Louis Hessian in coefficient space.
pub(crate) fn subject_marginal<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    derivatives: bool,
) -> Result<SubjectOutput<S>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    if n_nodes == 0 || marks == 0 {
        return Err(numerical(
            "subject marginal needs at least one node and one mark",
        ));
    }
    if inputs.eta0.len() != n_nodes * marks || inputs.loadings.len() != marks * atoms {
        return Err(numerical(
            "subject marginal received mismatched parameter slices",
        ));
    }
    if let Some(designs) = inputs.designs
        && (designs.len() != marks || designs.iter().any(|d| d.nrows() != n_nodes))
    {
        return Err(numerical(
            "subject marginal received design rows of the wrong shape",
        ));
    }
    let like = &inputs.eta0[0];
    let zero = like.constant_like(0.0);
    let counts_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.counts.row(n).to_vec()).collect();
    let exposure_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.exposure_row(n)).collect();

    // ---- forward filter ------------------------------------------------
    let filtered = filter_nodes(inputs, derivatives, &counts_rows, &exposure_rows)?;
    let node_loglik: Vec<S> = filtered
        .iter()
        .map(|node| ln(&node.normaliser).add(&like.constant_like(node.likelihood.shift)))
        .collect();
    let loglik = pairwise_sum(&node_loglik, &zero);
    if !loglik.value().is_finite() {
        return Err(numerical("subject marginal log-likelihood is not finite"));
    }
    if !derivatives {
        return Ok(SubjectOutput {
            loglik,
            gradient: Vec::new(),
            hessian: Vec::new(),
        });
    }

    // ---- layout ------------------------------------------------------------
    // With designs, the coefficients are the per-mark blocks in mark order.
    // Without, the parameters are the node log-intensities themselves, laid
    // out exactly as `eta0` is: node-major, `n * marks + d`.
    let mut offsets = Vec::with_capacity(marks);
    let mut acc = 0usize;
    match inputs.designs {
        Some(designs) => {
            for design in designs {
                offsets.push(acc);
                acc += design.ncols();
            }
        }
        None => {
            offsets.extend(0..marks);
            acc = n_nodes * marks;
        }
    }
    let layout = Layout {
        offsets,
        atoms,
        a0: acc,
        rho0: acc + marks * atoms,
        total: acc + marks * atoms + atoms,
    };
    let p_total = layout.total;
    // Design row `x_{n,d,·}` as (coefficient, value) pairs.
    let design_row = |n: usize, d: usize| -> Vec<(usize, f64)> {
        match inputs.designs {
            Some(designs) => designs[d]
                .row(n)
                .iter()
                .enumerate()
                .filter(|(_, x)| **x != 0.0)
                .map(|(j, x)| (layout.offsets[d] + j, *x))
                .collect(),
            None => vec![(n * marks + d, 1.0)],
        }
    };

    // ---- backward pass: smoother residual and innovation moments ----------
    let n_gaps = n_nodes.saturating_sub(1);
    let Smoothed {
        marginals: smoothed_all,
        innovation_moments,
    } = backward_smoother(inputs, &filtered, &counts_rows, &exposure_rows, true)?;

    // ---- forward sweep: Fisher mean and Louis second moment ---------------
    // `carried[q * size + i]` is `C_m(z_i)[q]`, the conditional expectation
    // given `z_m = z_i` and the data up to `m` of the complete-data score
    // accumulated over the nodes and gaps before `m`. `mean` is `E[g | y]`
    // and `second` is `E[g gᵀ | y]`, both in coefficient space; `curvature`
    // is `E[∂²L_c | y]`, accumulated as each term is met.
    //
    // The complete-data node term is `y η − w e^η` with
    // `η = η⁰ − log M_d + a_d · z`, so with the centred coordinate
    // `ζ_{dk} = ∂η/∂a_{dk}`:
    //   ∂L/∂η⁰ = s,  ∂L/∂a_{dk} = s ζ_{dk},
    //   ∂²L/∂η⁰² = −c,  ∂²L/∂η⁰∂a_{dk} = −c ζ_{dk},
    //   ∂²L/∂a_{dk}∂a_{dj} = −c ζ_{dk} ζ_{dj} + s ∂²η/∂a_{dk}∂a_{dj},
    // with `s = y − w e^η` the score and `c = w e^η` the curvature in `η`.
    //
    // Which centring is in force decides both. Under the stationary prior's
    // `log M_d = ½|a_d|²` the derivative is `ζ_{dk} = z_k − a_{dk}` and the
    // second derivative is `−δ_{kj}`, the curvature of that shift. Under a
    // supplied risk-set normaliser (`super::preserve`) the normaliser is held
    // as data over the solve, so `ζ_{dk} = z_k` and the second derivative
    // vanishes. Holding it costs no consistency: `log M` is predictable, so
    // its own score contribution `−Σ_events ∂log M + ∫ R λ ∂log M` has
    // expectation zero by the compensator identity, and the estimating
    // equation the held normaliser defines is unbiased.
    let prior_centred = inputs.log_normaliser.is_none();
    let mut mean = vec![zero.clone(); p_total];
    let mut second = vec![zero.clone(); p_total * p_total];
    let mut curvature = vec![zero.clone(); p_total * p_total];
    // Louis' identity is `E[∂²L_c | y] + Var[∂L_c | y]`. Without atoms there
    // is no latent state to be uncertain about: the complete-data likelihood
    // *is* the observed one, its score has no posterior spread, and the
    // variance term is identically zero. Everything that forms it — the
    // carried second moments, the same-node block over every ordered pair of
    // marks, and the subtraction of the mean's outer product — is then
    // arithmetic whose answer is known, and it is the part that grows with
    // the square of the mark count. The expected curvature and the mean are
    // still needed, and still formed.
    let latent_variance = atoms > 0;
    let mut carried: Vec<S> = if latent_variance {
        vec![zero.clone(); p_total * filtered[0].grid.size()]
    } else {
        Vec::new()
    };
    // Reused by every node and mark: the grid's size is `order^atoms`, the
    // same at every node, so this is one allocation rather than one per mark
    // per node of a vector as wide as the grid.
    let mut weighted: Vec<S> = vec![zero.clone(); filtered[0].grid.size()];
    for m in 0..n_nodes {
        let grid = &filtered[m].grid;
        let size = grid.size();
        let smoothed = &smoothed_all[m];
        // `W(i) = w_i s(i)`: the smoothed probability of grid point `i`.
        let w: Vec<S> = (0..size)
            .map(|i| grid.weights[i].mul(&smoothed[i]))
            .collect();
        let rows: Vec<Vec<(usize, f64)>> = (0..marks).map(|d| design_row(m, d)).collect();
        // Centred coordinates `ζ_{dk}(i) = ∂η_d/∂a_{dk}` per mark and atom.
        let centred: Vec<Vec<Vec<S>>> = (0..marks)
            .map(|d| {
                (0..atoms)
                    .map(|k| {
                        let a = &inputs.loadings[d * atoms + k];
                        (0..size)
                            .map(|i| {
                                let z = grid.coordinate(i, k);
                                if prior_centred { z.sub(a) } else { z.clone() }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        // The node's score and curvature in every mark's `η`, formed once
        // with the node factor itself.
        let node = &filtered[m].likelihood;
        let scores: Vec<&[S]> = (0..marks)
            .map(|d| &node.score[d * size..(d + 1) * size])
            .collect();
        // The three passes below must not overlap: the carried vector holds
        // the functions of nodes strictly before `m` (plus the gaps before
        // it) while every node function of `m` is contracted against it, and
        // only then does it absorb them. Absorbing mark `d` before mark `d'`
        // is contracted would count the same-node pair twice — once through
        // the carry and once through the same-node block below.
        let ws_all: Vec<Vec<S>> = (0..marks)
            .map(|d| (0..size).map(|i| w[i].mul(&scores[d][i])).collect())
            .collect();
        // ---- pass 1: this node's functions against everything carried ------
        for d in 0..marks {
            let ws = &ws_all[d];
            // B[q] = Σ_i W s_d C[q];  A_k[q] = Σ_i W s_d ζ_{dk} C[q]
            let mut b = vec![zero.clone(); if latent_variance { p_total } else { 0 }];
            let mut a_k = vec![vec![zero.clone(); p_total]; atoms];
            if latent_variance {
                // `W s_d C[q]` is formed once per grid point and reused for the
                // atom contractions, which is the same arithmetic in the same
                // order with the product taken once instead of once per atom.
                for q in 0..p_total {
                    let row = &carried[q * size..(q + 1) * size];
                    let mut acc = zero.clone();
                    for i in 0..size {
                        weighted[i] = ws[i].mul(&row[i]);
                        acc = acc.add(&weighted[i]);
                    }
                    b[q] = acc;
                    for k in 0..atoms {
                        let mut acc = zero.clone();
                        for i in 0..size {
                            acc = acc.add(&weighted[i].mul(&centred[d][k][i]));
                        }
                        a_k[k][q] = acc;
                    }
                }
            }
            // E[C v_dᵀ] and its transpose.
            if latent_variance {
                for q in 0..p_total {
                    for &(col, x) in &rows[d] {
                        let value = b[q].scale(x);
                        second[q * p_total + col] = second[q * p_total + col].add(&value);
                        second[col * p_total + q] = second[col * p_total + q].add(&value);
                    }
                    for k in 0..atoms {
                        let col = layout.a(d, k);
                        second[q * p_total + col] = second[q * p_total + col].add(&a_k[k][q]);
                        second[col * p_total + q] = second[col * p_total + q].add(&a_k[k][q]);
                    }
                }
            }
            // Mean of the node function.
            let s_mean = ws.iter().fold(zero.clone(), |acc, v| acc.add(v));
            for &(col, x) in &rows[d] {
                mean[col] = mean[col].add(&s_mean.scale(x));
            }
            for k in 0..atoms {
                let mut acc = zero.clone();
                for i in 0..size {
                    acc = acc.add(&ws[i].mul(&centred[d][k][i]));
                }
                mean[layout.a(d, k)] = mean[layout.a(d, k)].add(&acc);
            }
            // Expected curvature of the node term.
            let mut ec = zero.clone();
            let mut ecz = vec![zero.clone(); atoms];
            let mut eczz = vec![zero.clone(); atoms * atoms];
            if node.informative[d] {
                for i in 0..size {
                    let wc = w[i].mul(&node.curvature[d * size + i]);
                    ec = ec.add(&wc);
                    for k in 0..atoms {
                        let wcz = wc.mul(&centred[d][k][i]);
                        ecz[k] = ecz[k].add(&wcz);
                        for j in 0..atoms {
                            eczz[k * atoms + j] =
                                eczz[k * atoms + j].add(&wcz.mul(&centred[d][j][i]));
                        }
                    }
                }
            }
            if prior_centred {
                for k in 0..atoms {
                    // −s δ_{kj}: the curvature of the stationary prior's
                    // shift. A held risk-set normaliser has none.
                    eczz[k * atoms + k] = eczz[k * atoms + k].add(&s_mean);
                }
            }
            for &(c1, x1) in &rows[d] {
                for &(c2, x2) in &rows[d] {
                    curvature[c1 * p_total + c2] =
                        curvature[c1 * p_total + c2].sub(&ec.scale(x1 * x2));
                }
                for k in 0..atoms {
                    let c2 = layout.a(d, k);
                    let value = ecz[k].scale(x1);
                    curvature[c1 * p_total + c2] = curvature[c1 * p_total + c2].sub(&value);
                    curvature[c2 * p_total + c1] = curvature[c2 * p_total + c1].sub(&value);
                }
            }
            for k in 0..atoms {
                for j in 0..atoms {
                    let (c1, c2) = (layout.a(d, k), layout.a(d, j));
                    curvature[c1 * p_total + c2] =
                        curvature[c1 * p_total + c2].sub(&eczz[k * atoms + j]);
                }
            }
        }
        // ---- pass 2: the same-node block, every ordered pair of marks -------
        for d in 0..if latent_variance { marks } else { 0 } {
            let ws = &ws_all[d];
            for d2 in 0..marks {
                let mut m00 = zero.clone();
                let mut m0k = vec![zero.clone(); atoms];
                let mut mk0 = vec![zero.clone(); atoms];
                let mut mkj = vec![zero.clone(); atoms * atoms];
                for i in 0..size {
                    let ss = ws[i].mul(&scores[d2][i]);
                    m00 = m00.add(&ss);
                    for k in 0..atoms {
                        m0k[k] = m0k[k].add(&ss.mul(&centred[d2][k][i]));
                        let ssz = ss.mul(&centred[d][k][i]);
                        mk0[k] = mk0[k].add(&ssz);
                        for j in 0..atoms {
                            mkj[k * atoms + j] =
                                mkj[k * atoms + j].add(&ssz.mul(&centred[d2][j][i]));
                        }
                    }
                }
                for &(c1, x1) in &rows[d] {
                    for &(c2, x2) in &rows[d2] {
                        second[c1 * p_total + c2] =
                            second[c1 * p_total + c2].add(&m00.scale(x1 * x2));
                    }
                    for k in 0..atoms {
                        let c2 = layout.a(d2, k);
                        second[c1 * p_total + c2] =
                            second[c1 * p_total + c2].add(&m0k[k].scale(x1));
                    }
                }
                for k in 0..atoms {
                    let c1 = layout.a(d, k);
                    for &(c2, x2) in &rows[d2] {
                        second[c1 * p_total + c2] =
                            second[c1 * p_total + c2].add(&mk0[k].scale(x2));
                    }
                    for j in 0..atoms {
                        let c2 = layout.a(d2, j);
                        second[c1 * p_total + c2] =
                            second[c1 * p_total + c2].add(&mkj[k * atoms + j]);
                    }
                }
            }
        }
        // ---- pass 3: the node's functions join the carried vector ----------
        for d in 0..if latent_variance { marks } else { 0 } {
            for &(col, x) in &rows[d] {
                let row = &mut carried[col * size..(col + 1) * size];
                for i in 0..size {
                    row[i] = row[i].add(&scores[d][i].scale(x));
                }
            }
            for k in 0..atoms {
                let col = layout.a(d, k);
                let row = &mut carried[col * size..(col + 1) * size];
                for i in 0..size {
                    row[i] = row[i].add(&scores[d][i].mul(&centred[d][k][i]));
                }
            }
        }
        if m == n_gaps {
            break;
        }
        // The gap carries the latent moments across it and scores its own
        // rate. Without atoms there is neither: nothing is carried, no rate
        // is a coefficient, and the whole section is a step over a state that
        // does not exist.
        if !latent_variance {
            continue;
        }
        // ---- gap m: (m, m+1) ------------------------------------------------
        let next = &filtered[m + 1];
        let next_size = next.grid.size();
        let transitions = &next.transitions;
        let backward_moments = &innovation_moments[m];
        let polys: Vec<(GapPolynomial<S>, GapPolynomial<S>)> = (0..atoms)
            .map(|k| gap_score_polynomials(&transitions[k], like))
            .collect();
        let unit = |k: usize, b: u8| -> Vec<u8> {
            let mut e = vec![0u8; atoms];
            e[k] = b;
            e
        };
        // Powers of the start coordinate of each atom on grid m.
        let powers: Vec<Vec<Vec<S>>> = (0..atoms)
            .map(|k| {
                (0..=4usize)
                    .map(|a| {
                        (0..size)
                            .map(|i| {
                                let z = grid.coordinate(i, k);
                                let mut v = like.constant_like(1.0);
                                for _ in 0..a {
                                    v = v.mul(z);
                                }
                                v
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        // Σ_{a,b} c[a][b] z_k^a E[u_k^b | z, data] on grid m.
        let start_function = |k: usize, poly: &GapPolynomial<S>, max_degree: usize| -> Vec<S> {
            let mut out = vec![zero.clone(); size];
            for a in 0..=max_degree {
                for b in 0..=max_degree {
                    if poly.absent(a, b) {
                        continue;
                    }
                    let coefficient = poly.get(a, b);
                    let moment = &backward_moments[&unit(k, b as u8)];
                    for i in 0..size {
                        out[i] = out[i].add(&coefficient.mul(&powers[k][a][i]).mul(&moment[i]));
                    }
                }
            }
            out
        };
        for k in 0..atoms {
            let (t, dt) = &polys[k];
            let tk = start_function(k, t, 2);
            let dtk = start_function(k, dt, 2);
            let ttk = start_function(k, &t.mul(t), 4);
            let rho = layout.rho(k);
            let mut e_t = zero.clone();
            let mut e_dt = zero.clone();
            let mut e_tt = zero.clone();
            for i in 0..size {
                e_t = e_t.add(&w[i].mul(&tk[i]));
                e_dt = e_dt.add(&w[i].mul(&dtk[i]));
                e_tt = e_tt.add(&w[i].mul(&ttk[i]));
            }
            mean[rho] = mean[rho].add(&e_t);
            curvature[rho * p_total + rho] = curvature[rho * p_total + rho].add(&e_dt);
            second[rho * p_total + rho] = second[rho * p_total + rho].add(&e_tt);
            // E[C_m t_k]: the gap score against everything carried so far
            // (nodes ≤ m and gaps < m), by the Markov property through z_m.
            for q in 0..p_total {
                let row = &carried[q * size..(q + 1) * size];
                let mut acc = zero.clone();
                for i in 0..size {
                    acc = acc.add(&w[i].mul(&tk[i]).mul(&row[i]));
                }
                second[q * p_total + rho] = second[q * p_total + rho].add(&acc);
                second[rho * p_total + q] = second[rho * p_total + q].add(&acc);
            }
        }
        // Cross-atom same-gap products E[t_k t_j].
        for k in 0..atoms {
            for j in (k + 1)..atoms {
                let (tk, _) = &polys[k];
                let (tj, _) = &polys[j];
                let mut value = zero.clone();
                for a in 0..=2usize {
                    for b in 0..=2usize {
                        if tk.absent(a, b) {
                            continue;
                        }
                        let ck = tk.get(a, b);
                        for a2 in 0..=2usize {
                            for b2 in 0..=2usize {
                                if tj.absent(a2, b2) {
                                    continue;
                                }
                                let cj = tj.get(a2, b2);
                                let mut e = vec![0u8; atoms];
                                e[k] = b as u8;
                                e[j] = b2 as u8;
                                let moment = &backward_moments[&e];
                                let coefficient = ck.mul(cj);
                                for i in 0..size {
                                    value = value.add(
                                        &w[i]
                                            .mul(&coefficient)
                                            .mul(&powers[k][a][i])
                                            .mul(&powers[j][a2][i])
                                            .mul(&moment[i]),
                                    );
                                }
                            }
                        }
                    }
                }
                let (rk, rj) = (layout.rho(k), layout.rho(j));
                second[rk * p_total + rj] = second[rk * p_total + rj].add(&value);
                second[rj * p_total + rk] = second[rj * p_total + rk].add(&value);
            }
        }
        // ---- propagate the carried vector to grid m+1 -------------------------
        // C_{m+1}(z') = F[α_m C_m](z') / p̂_{m+1}(z'), plus the gap score's own
        // forward-smoothed expectation E[t_g | z', y_{≤ m+1}] = ã / p̂ in its
        // log-rate slot. Where the predicted density is below its noise floor
        // no ratio is formed; the smoothed marginal has no mass there.
        let floor = density_floor(&next.predicted);
        let inverse_predicted: Vec<Option<S>> = next
            .predicted
            .iter()
            .map(|density| {
                if density.value() > floor {
                    Some(recip(density))
                } else {
                    None
                }
            })
            .collect();
        let alpha = &filtered[m].alpha;
        let mut propagated = vec![zero.clone(); p_total * next_size];
        for q in 0..p_total {
            let row = &carried[q * size..(q + 1) * size];
            let weighted: Vec<S> = (0..size).map(|i| alpha[i].mul(&row[i])).collect();
            let moved = next.forward.plain(&weighted);
            let out = &mut propagated[q * next_size..(q + 1) * next_size];
            for i in 0..next_size {
                if let Some(inverse) = &inverse_predicted[i] {
                    out[i] = moved[i].mul(inverse);
                }
            }
        }
        for k in 0..atoms {
            let (t, _) = &polys[k];
            // Ã on grid m+1: Σ_{a,b} c[a][b] ∫ A(z'|z) u^b z^a α(z) dz.
            let mut a_tilde = vec![zero.clone(); next_size];
            for a in 0..=2usize {
                for b in 0..=2usize {
                    if t.absent(a, b) {
                        continue;
                    }
                    let coefficient = t.get(a, b);
                    let e = unit(k, b as u8);
                    let zk_power: Vec<S> =
                        (0..size).map(|i| alpha[i].mul(&powers[k][a][i])).collect();
                    let moment = next.forward.apply(&e, &zk_power);
                    for i in 0..next_size {
                        a_tilde[i] = a_tilde[i].add(&coefficient.mul(&moment[i]));
                    }
                }
            }
            let rho = layout.rho(k);
            let out = &mut propagated[rho * next_size..(rho + 1) * next_size];
            for i in 0..next_size {
                if let Some(inverse) = &inverse_predicted[i] {
                    out[i] = out[i].add(&a_tilde[i].mul(inverse));
                }
            }
        }
        carried = propagated;
    }

    // ---- assemble ----------------------------------------------------------
    let mut hessian = vec![zero.clone(); p_total * p_total];
    for q in 0..p_total {
        for r in q..p_total {
            let (value, mirror) = if latent_variance {
                (
                    curvature[q * p_total + r]
                        .add(&second[q * p_total + r])
                        .sub(&mean[q].mul(&mean[r])),
                    curvature[r * p_total + q]
                        .add(&second[r * p_total + q])
                        .sub(&mean[r].mul(&mean[q])),
                )
            } else {
                (
                    curvature[q * p_total + r].clone(),
                    curvature[r * p_total + q].clone(),
                )
            };
            let symmetric = value.add(&mirror).scale(0.5);
            hessian[q * p_total + r] = symmetric.clone();
            hessian[r * p_total + q] = symmetric;
        }
    }
    // ---- the rate slots, from the log-rate to the rate ---------------------
    // The gap scores are derivatives in `ρ = ln ν`, the coordinate in which
    // they stay bounded across a short gap. The coefficient is `ν`, so with
    // `dρ/dν = 1/ν` and `d²ρ/dν² = −1/ν²`:
    //   ∂ℓ/∂ν = ∂ℓ/∂ρ / ν,
    //   ∂²ℓ/∂ν² = (∂²ℓ/∂ρ² − ∂ℓ/∂ρ) / ν²,   ∂²ℓ/∂ν∂x = ∂²ℓ/∂ρ∂x / ν.
    // The factors are jets, so every derivative channel of the conversion
    // is carried along with the value.
    let inverse_rates: Vec<S> = inputs.rates.iter().map(recip).collect();
    let rate_gradients: Vec<S> = (0..atoms).map(|k| mean[layout.rho(k)].clone()).collect();
    for k in 0..atoms {
        let rho_k = layout.rho(k);
        let inv_k = &inverse_rates[k];
        for j in 0..atoms {
            let rho_j = layout.rho(j);
            let inv_j = &inverse_rates[j];
            let raw = hessian[rho_k * p_total + rho_j].clone();
            let converted = if k == j {
                raw.sub(&rate_gradients[k]).mul(inv_k).mul(inv_k)
            } else {
                raw.mul(inv_k).mul(inv_j)
            };
            hessian[rho_k * p_total + rho_j] = converted;
        }
        for q in 0..layout.rho0 {
            let value = hessian[rho_k * p_total + q].mul(inv_k);
            hessian[rho_k * p_total + q] = value.clone();
            hessian[q * p_total + rho_k] = value;
        }
        mean[rho_k] = rate_gradients[k].mul(inv_k);
    }
    Ok(SubjectOutput {
        loglik,
        gradient: mean,
        hessian,
    })
}

/// The forward filter over every node of a subject, each node keeping the
/// operators that reached it so a backward pass can be run on the result.
/// With `derivatives` the forward operators carry the innovation powers the
/// gap scores need.
fn filter_nodes<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    derivatives: bool,
    counts_rows: &[Vec<f64>],
    exposure_rows: &[Vec<f64>],
) -> Result<Vec<FilteredNode<S>>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let gh = inputs.gh;
    let like = &inputs.eta0[0];
    if atoms > 0 && inputs.rates.iter().all(|r| r.value() == 0.0) {
        if derivatives { return Err(numerical("static frailties use derivatives of the integrated objective")); }
        let pass = crate::static_state::filter(inputs, None, &vec![true; marks])?;
        return Ok((0..n_nodes).map(|n| {
            let likelihood = node_likelihood(&pass.grids[n], &inputs.eta0[n * marks..(n + 1) * marks],
                inputs.loadings, &counts_rows[n], &exposure_rows[n], None,
                inputs.log_normaliser.map(|m| &m[n * marks..(n + 1) * marks]), marks, atoms, false);
            FilteredNode { grid: pass.grids[n].clone(), transitions: Vec::new(),
                forward: OperatorFamily { per_axis: Vec::new(), order: gh.order },
                predicted: pass.predicted[n].clone(), alpha: pass.alpha[n].clone(),
                normaliser: exp(&add_real(&pass.log_normalisers[n], -likelihood.shift)), likelihood }
        }).collect());
    }
    let mut filtered: Vec<FilteredNode<S>> = Vec::with_capacity(n_nodes);
    let forward_power: u8 = if derivatives { 2 } else { 0 };
    let node_terms = |grid: &Grid<S>, n: usize, store: bool| -> NodeLikelihood<S> {
        node_likelihood(
            grid,
            &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings,
            &counts_rows[n],
            &exposure_rows[n],
            None,
            inputs
                .log_normaliser
                .map(|m| &m[n * marks..(n + 1) * marks]),
            marks,
            atoms,
            store,
        )
    };
    filtered.push(filter_start(
        gh,
        like,
        atoms,
        &|grid, store| node_terms(grid, 0, store),
        derivatives,
        "first node",
    )?);
    for n in 0..n_nodes - 1 {
        let transitions = transitions_across(inputs.rates, nodes.gaps[n], inputs.time_scale)?;
        let step = filter_step(
            gh,
            like,
            &filtered[n].grid,
            &filtered[n].alpha,
            transitions,
            forward_power,
            &|grid, store| node_terms(grid, n + 1, store),
            derivatives,
            &format!("node {}", n + 1),
        )?;
        filtered.push(step);
    }
    Ok(filtered)
}

/// What the backward pass yields: the smoothed marginal on every node's
/// grid (normalised to a probability), and the smoothed innovation moments
/// of every gap when they were asked for.
struct Smoothed<S> {
    marginals: Vec<Vec<S>>,
    innovation_moments: Vec<HashMap<Vec<u8>, Vec<S>>>,
}

/// The backward pass over a filtered chain.
///
/// The future likelihood `lik_{n+1} β_{n+1}` is an exponential in the state
/// after an event, so it is never interpolated as a value: its logarithm
/// is interpolated (relative accuracy), and what is carried backward is
/// `log β_n = log E[lik_{n+1} β_{n+1} / c_{n+1} | z_n]`, a bounded smooth
/// function, plus — for the derivatives — the smoothed innovation moments
/// `E[Π_k u_k^{e_k} | z_n, data]` of every gap, which the gap scores need.
/// Kernel rows are streamed and reduced to those moments; no `S × S`
/// allocation is formed.
///
/// `β` alone overflows on a wide hull (it is a future-likelihood ratio,
/// astronomically large where the filtered density is astronomically
/// small), so it is never exponentiated on its own: the smoothed marginal
/// `α β` is formed in log space, a point whose filtered density is below
/// the noise floor carries no smoothed mass, and the result is renormalised
/// so every expectation under it is under a probability.
fn backward_smoother<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    filtered: &[FilteredNode<S>],
    counts_rows: &[Vec<f64>],
    exposure_rows: &[Vec<f64>],
    with_innovation_moments: bool,
) -> Result<Smoothed<S>, EventHistoryError> {
    let n_nodes = filtered.len();
    if !inputs.rates.is_empty() && inputs.rates.iter().all(|r| r.value() == 0.0) {
        if with_innovation_moments { return Err(numerical("static frailties have no innovation scores")); }
        return Ok(Smoothed { marginals: vec![filtered[n_nodes - 1].alpha.clone(); n_nodes],
            innovation_moments: Vec::new() });
    }
    let marks = inputs.nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let gh = inputs.gh;
    let zero = inputs.eta0[0].constant_like(0.0);
    let n_gaps = n_nodes.saturating_sub(1);
    let inner_count = filtered[0].grid.size();
    let log_inner_weights: Vec<f64> = (0..inner_count)
        .map(|l| {
            let mut rest = l;
            let mut acc = 0.0;
            for _ in 0..atoms {
                acc += gh.normal_weights[rest % gh.order].ln();
                rest /= gh.order;
            }
            acc
        })
        .collect();
    let inner_innovation = |l: usize, e: &[u8]| -> f64 {
        let mut rest = l;
        let mut acc = 1.0;
        for &power in e.iter() {
            let u = std::f64::consts::SQRT_2 * gh.nodes[rest % gh.order];
            rest /= gh.order;
            acc *= u.powi(i32::from(power));
        }
        acc
    };
    let mut innovation_exponents: Vec<Vec<u8>> = Vec::new();
    if with_innovation_moments {
        for k in 0..atoms {
            for b in 0..=4u8 {
                let mut e = vec![0u8; atoms];
                e[k] = b;
                innovation_exponents.push(e);
            }
            for j in (k + 1)..atoms {
                for b in 1..=2u8 {
                    for b2 in 1..=2u8 {
                        let mut e = vec![0u8; atoms];
                        e[k] = b;
                        e[j] = b2;
                        innovation_exponents.push(e);
                    }
                }
            }
        }
    }
    let mut log_beta: Vec<Vec<S>> = vec![Vec::new(); n_nodes];
    log_beta[n_nodes - 1] = vec![zero.clone(); filtered[n_nodes - 1].grid.size()];
    let mut innovation_moments: Vec<HashMap<Vec<u8>, Vec<S>>> = vec![HashMap::new(); n_gaps];
    for n in (0..n_gaps).rev() {
        let grid = &filtered[n].grid;
        let next = &filtered[n + 1].grid;
        let size = grid.size();
        let transitions = &filtered[n + 1].transitions;
        // The node's log-likelihood is an explicit formula, so it is
        // evaluated exactly at every inner point; only the smoother residual
        // `log β_{n+1}` is interpolated.
        let bases = backward_axis_bases(gh, grid, next, transitions);
        let log_c = ln(&filtered[n + 1].normaliser);
        let shift = filtered[n + 1].likelihood.shift;
        let node_log_lik = |zeta: &[S]| -> S {
            let mut ell = zero.clone();
            for d in 0..marks {
                let eta = log_intensity(
                    &inputs.eta0[(n + 1) * marks + d],
                    &inputs.loadings[d * atoms..(d + 1) * atoms],
                    zeta,
                    inputs.log_normaliser.map(|m| &m[(n + 1) * marks + d]),
                );
                let y = counts_rows[n + 1][d];
                if y != 0.0 {
                    ell = ell.add(&eta.scale(y));
                }
                let exposure = exposure_rows[n + 1][d];
                if exposure != 0.0 {
                    ell = ell.sub(&exp(&eta).scale(exposure));
                }
            }
            add_real(&ell, -shift).sub(&log_c)
        };
        let spreads: Vec<S> = transitions
            .iter()
            .map(|t| if t.innovation.value() == 0.0 {
                t.innovation.constant_like(0.0)
            } else { sqrt(&t.innovation.scale(2.0)) })
            .collect();
        let mut log_beta_n = Vec::with_capacity(size);
        let mut moments: HashMap<Vec<u8>, Vec<S>> = innovation_exponents
            .iter()
            .map(|e| (e.clone(), Vec::with_capacity(size)))
            .collect();
        for i in 0..size {
            let at_inner = interpolate_at_inner_points(gh.order, &bases, &log_beta[n + 1], i);
            let terms: Vec<S> = (0..inner_count)
                .map(|l| {
                    let mut rest_i = i;
                    let mut rest_l = l;
                    let zeta: Vec<S> = (0..atoms)
                        .map(|k| {
                            let point = grid.axes[k].points[rest_i % gh.order].clone();
                            let x = gh.nodes[rest_l % gh.order];
                            rest_i /= gh.order;
                            rest_l /= gh.order;
                            transitions[k].phi.mul(&point).add(&spreads[k].scale(x))
                        })
                        .collect();
                    add_real(
                        &node_log_lik(&zeta).add(&at_inner[l]),
                        log_inner_weights[l],
                    )
                })
                .collect();
            let log_total = log_sum_exp(&terms);
            if with_innovation_moments {
                let weights: Vec<S> = terms
                    .iter()
                    .map(|term| exp(&term.sub(&log_total)))
                    .collect();
                for e in innovation_exponents.iter() {
                    let moment = weights
                        .iter()
                        .enumerate()
                        .fold(zero.clone(), |acc, (l, w)| {
                            acc.add(&w.scale(inner_innovation(l, e)))
                        });
                    moments
                        .get_mut(e)
                        .expect("registered exponent")
                        .push(moment);
                }
            }
            log_beta_n.push(log_total);
        }
        log_beta[n] = log_beta_n;
        innovation_moments[n] = moments;
    }
    let marginals: Vec<Vec<S>> = (0..n_nodes)
        .map(|n| {
            let alpha = &filtered[n].alpha;
            let floor = density_floor(alpha);
            let mut smoothed: Vec<S> = alpha
                .iter()
                .zip(log_beta[n].iter())
                .map(|(a, log_b)| {
                    if a.value() > floor {
                        exp(&ln(a).add(log_b))
                    } else {
                        zero.clone()
                    }
                })
                .collect();
            let mass = weighted_sum(&filtered[n].grid.weights, &smoothed);
            if !(mass.value() > 0.0) || !mass.value().is_finite() {
                return Err(numerical(format!(
                    "node {n}: smoothed marginal has mass {} on the grid",
                    mass.value()
                )));
            }
            let inverse = recip(&mass);
            for s in smoothed.iter_mut() {
                *s = s.mul(&inverse);
            }
            Ok(smoothed)
        })
        .collect::<Result<_, _>>()?;
    Ok(Smoothed {
        marginals,
        innovation_moments,
    })
}

/// The posterior mean and covariance of the latent state at every node of a
/// subject given its whole history: the moments of the smoothed marginal on
/// each node's grid. Per node, the mean over the atoms and the row-major
/// `atoms × atoms` covariance.
pub(crate) fn latent_state_moments(
    inputs: &SubjectInputs<'_, f64>,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    if n_nodes == 0 || marks == 0 {
        return Err(numerical(
            "latent state moments need at least one node and one mark",
        ));
    }
    if inputs.eta0.len() != n_nodes * marks || inputs.loadings.len() != marks * atoms {
        return Err(numerical(
            "latent state moments received mismatched parameter slices",
        ));
    }
    let counts_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.counts.row(n).to_vec()).collect();
    let exposure_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.exposure_row(n)).collect();
    let filtered = filter_nodes(inputs, false, &counts_rows, &exposure_rows)?;
    let smoothed = backward_smoother(inputs, &filtered, &counts_rows, &exposure_rows, false)?;
    Ok(filtered
        .iter()
        .zip(smoothed.marginals.iter())
        .map(|(node, density)| {
            let grid = &node.grid;
            let mut mean = vec![0.0; atoms];
            for i in 0..grid.size() {
                let w = grid.weights[i] * density[i];
                for k in 0..atoms {
                    mean[k] += w * grid.coordinate(i, k);
                }
            }
            let mut covariance = vec![0.0; atoms * atoms];
            for i in 0..grid.size() {
                let w = grid.weights[i] * density[i];
                for k in 0..atoms {
                    let dk = grid.coordinate(i, k) - mean[k];
                    for j in 0..atoms {
                        covariance[k * atoms + j] += w * dk * (grid.coordinate(i, j) - mean[j]);
                    }
                }
            }
            (mean, covariance)
        })
        .collect())
}

/// One completed forward filter: per-node grids, filtered densities, the
/// predicted (pre-update) densities, and the per-node log normalisers
/// `ln c_n + m_n`, whose running sum is the log predictive probability of the
/// observed counts.
pub(crate) struct ForwardPass<S> {
    pub grids: Vec<Grid<S>>,
    pub alpha: Vec<Vec<S>>,
    pub predicted: Vec<Vec<S>>,
    pub log_normalisers: Vec<S>,
}

/// Forward filter only, optionally continuing from a filtered state and
/// optionally restricting the compensator to a subset of marks (a forecast
/// conditions on the absorbing marks not having fired).
pub(crate) fn forward_filter<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    initial: Option<(&Grid<S>, &[S])>,
    compensated: &[bool],
) -> Result<ForwardPass<S>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let gh = inputs.gh;
    if n_nodes == 0 || marks == 0 || compensated.len() != marks {
        return Err(numerical(
            "forward filter needs nodes, marks and a compensator mask",
        ));
    }
    if atoms > 0 && inputs.rates.iter().all(|r| r.value() == 0.0) {
        return crate::static_state::filter(inputs, initial, compensated);
    }
    let like = &inputs.eta0[0];
    let mut grids: Vec<Grid<S>> = Vec::with_capacity(n_nodes);
    let mut alpha: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut predicted: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut log_normalisers: Vec<S> = Vec::with_capacity(n_nodes);
    let node_terms = |grid: &Grid<S>, n: usize| -> NodeLikelihood<S> {
        node_likelihood(
            grid,
            &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings,
            &nodes.counts.row(n).to_vec(),
            &nodes.exposure_row(n),
            Some(compensated),
            inputs
                .log_normaliser
                .map(|m| &m[n * marks..(n + 1) * marks]),
            marks,
            atoms,
            false,
        )
    };
    // Node 0: either the stationary prior or a continuation of a filtered state.
    let first = match initial {
        None => filter_start(
            gh,
            like,
            atoms,
            &|grid, _| node_terms(grid, 0),
            false,
            "forecast first node",
        )?,
        Some((grid, filtered)) => filter_step(
            gh,
            like,
            grid,
            filtered,
            transitions_across(inputs.rates, inputs.continuation_gap, inputs.time_scale)?,
            0,
            &|grid, _| node_terms(grid, 0),
            false,
            "forecast first node",
        )?,
    };
    log_normalisers.push(ln(&first.normaliser).add(&like.constant_like(first.likelihood.shift)));
    predicted.push(first.predicted);
    alpha.push(first.alpha);
    grids.push(first.grid);
    for n in 0..n_nodes - 1 {
        let step = filter_step(
            gh,
            like,
            &grids[n],
            &alpha[n],
            transitions_across(inputs.rates, nodes.gaps[n], inputs.time_scale)?,
            0,
            &|grid, _| node_terms(grid, n + 1),
            false,
            &format!("forecast node {}", n + 1),
        )?;
        log_normalisers.push(ln(&step.normaliser).add(&like.constant_like(step.likelihood.shift)));
        predicted.push(step.predicted);
        alpha.push(step.alpha);
        grids.push(step.grid);
    }
    Ok(ForwardPass {
        grids,
        alpha,
        predicted,
        log_normalisers,
    })
}

/// `E[λ_d(z)]` for every mark under a density on a grid: the expected
/// intensity of each mark at a node.
pub(crate) fn expected_intensities<S: JetField>(
    grid: &Grid<S>,
    density: &[S],
    eta0: &[S],
    loadings: &[S],
    log_normaliser: Option<&[S]>,
    marks: usize,
    atoms: usize,
) -> Vec<S> {
    let mut z = vec![eta0[0].constant_like(0.0); atoms];
    (0..marks)
        .map(|d| {
            let loadings_d = &loadings[d * atoms..(d + 1) * atoms];
            let mut acc = eta0[0].constant_like(0.0);
            for i in 0..grid.size() {
                for (k, zk) in z.iter_mut().enumerate() {
                    *zk = grid.coordinate(i, k).clone();
                }
                let eta = log_intensity(&eta0[d], loadings_d, &z, log_normaliser.map(|m| &m[d]));
                acc = acc.add(&grid.weights[i].mul(&density[i]).mul(&exp(&eta)));
            }
            acc
        })
        .collect()
}
