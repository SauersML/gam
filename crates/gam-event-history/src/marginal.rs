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
//! the same transition kernel the filter uses:
//! `C_{m+1}(z') = E[C_m(z) + t_m(z, z') | z_{m+1} = z', y_{≤m}]`, an
//! expectation under the kernel's normalised inner weights
//! ([`ForwardKernel`]). Nothing of size `S × S` is ever stored
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
    AtomTransition, FactorMark, ForwardKernel, GaussHermite, Grid, LogFactor, SplitDensity,
    backward_axis_bases, interpolate_at_inner_points, log_standard_prior, log_sum_exp,
};
use super::cohort::{EventHistoryError, SubjectNodes};
use super::scalar::{add_real, div, exp, ln, recip, sqrt, square};
use gam_math::nested_dual::JetField;
use gam_math::roundoff::accumulation_growth;
use ndarray::ArrayView2;
use std::collections::HashMap;
use std::sync::Arc;

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

/// What [`subject_marginal`] evaluates.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Evaluation {
    /// The marginal log-likelihood alone.
    Value,
    /// The log-likelihood with its Fisher gradient and Louis Hessian. Both
    /// read the backward smoother, whose every marginal is certified against
    /// `tolerance`, the relative accuracy the fit's quadrature is certified
    /// to (`super::family::EventHistorySpec::quadrature_tolerance`): the
    /// derivative pass inherits the forward pass's certificate instead of
    /// silently dropping mass ([`smoothed_marginal`]).
    Derivatives { tolerance: f64 },
}

/// The relative noise of a density carried on a grid it reached through
/// `operators` interpolating forward kernels.
///
/// A kernel interpolates `ln r` on the nodes of its source grid and averages
/// the interpolant under the row's normalised inner weights
/// ([`ForwardKernel`]). The nodal rounding of its operand is a relative `ε`
/// on the density, so an absolute `ε` on `ln r`; the interpolant amplifies
/// it by at most its sup-norm over the reach, the Lebesgue constant `Λ_G`
/// ([`GaussHermite::lebesgue_constant`]), and the inner average is a convex
/// combination, which amplifies nothing further. The filter renormalises
/// after every kernel, which removes the part of the error field common to
/// the whole grid, so each kernel leaves its own `Λ_G ε` and the
/// contributions add along the chain instead of compounding.
///
/// This is the model the fit already chooses its Gauss-Hermite order by:
/// `super::family`'s `certifiable` admits a rung only while that rung's
/// `Λ · ε · max_subject_nodes` stays at or under the quadrature tolerance.
/// The product is the same one, over a rule no coarser and a count no larger,
/// so a chain fitted at an admitted rung carries a noise this function reports
/// under that tolerance.
fn interpolation_noise_relative(gh: &GaussHermite, operators: usize) -> f64 {
    gh.lebesgue_constant * f64::EPSILON * operators as f64
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
    /// The same node term as an explicit function of the latent state, for
    /// the forward kernel to evaluate off the grid.
    pub factor: LogFactor<S>,
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
    let mut factor = LogFactor::zero(&zero);
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
        if y != 0.0 || exposure != 0.0 {
            factor.marks.push(FactorMark {
                count: y,
                log_exposure: (exposure != 0.0).then(|| exposure.ln()),
                base: base.clone(),
                loadings: loadings_d.to_vec(),
            });
        }
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
            // The tables hold exactly the products `a_k z_k` the linear
            // predictor sums, in the same order, so a node with exposure reads
            // `η` from them instead of forming every product a second time.
            let eta = match &latent {
                Some(tables) => {
                    let mut eta = base.clone();
                    for (k, table) in tables.iter().enumerate() {
                        eta = eta.add(&table[grid.index(i, k)]);
                    }
                    Some(eta)
                }
                None => None,
            };
            if y != 0.0 {
                let term = match &eta {
                    Some(eta) => eta.scale(y),
                    None => {
                        let mut eta = base.clone();
                        for (k, a) in loadings_d.iter().enumerate() {
                            eta = eta.add(&a.mul(grid.coordinate(i, k)));
                        }
                        eta.scale(y)
                    }
                };
                ell[i] = ell[i].add(&term);
            }
            // A mark with no exposure at this node has no compensator and so
            // no curvature: its intensity is never formed, which is the work
            // the risk sets save.
            match eta {
                Some(log_c) => {
                    let c = exp(&add_real(&log_c, exposure.ln()));
                    ell[i] = ell[i].sub(&c);
                    if derivatives {
                        score.push(add_real(&c.scale(-1.0), y));
                        curvature.push(c);
                    }
                }
                None => {
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
        factor,
    }
}

/// Posterior mean and variance of every atom under `alpha` on `grid`.
///
/// Every filtered density is the exponential of its log and every grid
/// weight is positive, so the exact variance of the quadrature law is
/// positive; a non-positive or non-finite one is a numerical failure,
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
        if !(variance.value().is_finite() && variance.value() > 0.0) {
            return Err(numerical(format!(
                "{label}: posterior variance of atom {k} is {} on the grid",
                variance.value()
            )));
        }
        means.push(mean);
        variances.push(variance);
    }
    Ok((means, variances))
}

/// A conditioned density on one grid, carried by its logarithm.
pub(crate) struct Conditioned<S> {
    pub log_alpha: Vec<S>,
    /// `exp(log_alpha)`, the density values the quadrature integrates.
    pub alpha: Vec<S>,
    /// `ln c`, the log normaliser of the node factor `exp(ell − shift)`.
    pub log_normaliser: S,
}

/// Multiply a predicted density on `grid`, given by its logarithm, by the node
/// factor `exp(ell − shift)` and normalise, all in the log domain:
/// `ln c = lse_i(ln p̂_i + ell_i − shift + ln w_i)` and
/// `ln α_i = ln p̂_i + ell_i − shift − ln c`. Nothing is formed that can
/// underflow to a zero or a subnormal before it is normalised.
pub(crate) fn condition<S: JetField>(
    grid: &Grid<S>,
    log_predicted: &[S],
    ell: &[S],
    shift: f64,
    label: &str,
) -> Result<Conditioned<S>, EventHistoryError> {
    let raw: Vec<S> = log_predicted
        .iter()
        .zip(ell.iter())
        .map(|(p, e)| add_real(&p.add(e), -shift))
        .collect();
    let terms: Vec<S> = raw
        .iter()
        .zip(grid.weights.iter())
        .map(|(r, w)| r.add(&ln(w)))
        .collect();
    let log_normaliser = log_sum_exp(&terms);
    if !log_normaliser.value().is_finite() {
        return Err(numerical(format!(
            "{label}: log normaliser is not finite ({})",
            log_normaliser.value()
        )));
    }
    let log_alpha: Vec<S> = raw.iter().map(|r| r.sub(&log_normaliser)).collect();
    let alpha = log_alpha.iter().map(exp).collect();
    Ok(Conditioned {
        log_alpha,
        alpha,
        log_normaliser,
    })
}

/// One filtered node: its grid, the transitions that reached it, the
/// predicted and filtered densities on it, and the node's likelihood pieces.
pub(crate) struct FilteredNode<S> {
    /// Shared, not copied: a static frailty's nodes all live on one grid.
    pub grid: Arc<Grid<S>>,
    /// Transitions across the gap that led here; empty at the first node.
    pub transitions: Vec<AtomTransition<S>>,
    pub log_predicted: Vec<S>,
    pub log_alpha: Vec<S>,
    pub alpha: Vec<S>,
    /// `ln c`, the log normaliser before the node's shift is added back.
    pub log_normaliser: S,
    pub likelihood: NodeLikelihood<S>,
    /// `log_alpha` split into its smooth part and the explicit node factors
    /// conditioned on since the last gap, for the kernel out of this node.
    pub density: SplitDensity<S>,
}

/// The split of a density conditioned on a node:
/// `ln α = s + f_prior + (ell − shift − ln c)`, with `s` the smooth part on
/// the node's grid, `f_prior` the factors the conditioned-on density already
/// carried explicitly (across a gap of zero length), and the node's own term
/// kept as the function it is.
pub(crate) fn conditioned_density<S: JetField>(
    smooth: Vec<S>,
    prior: Option<&LogFactor<S>>,
    likelihood: &NodeLikelihood<S>,
    log_normaliser: &S,
) -> SplitDensity<S> {
    let constant = add_real(&log_normaliser.scale(-1.0), -likelihood.shift);
    let node = likelihood.factor.shifted(&constant);
    let factor = match prior {
        Some(prior) => prior.joined(&node),
        None => node,
    };
    SplitDensity { smooth, factor }
}

/// Where a node's grid goes: at the posterior mean, with the predictive
/// spread.
///
/// The grid's Gauss-Hermite rule integrates the filtered density against the
/// grid's Gaussian envelope, exactly when their ratio is a polynomial of
/// degree below twice the order. The Poisson node factor is not: after an
/// event the log ratio is the log-intensity `a z` minus its exposure-weighted
/// exponential, and on a grid placed at the *predicted* moments the posterior
/// mass sits off the envelope's centre. Re-centring the grid on the posterior
/// mean absorbs the tilt into the envelope (a tilted Gaussian is a shifted
/// Gaussian) and leaves a ratio that is flat where the mass is. The forward
/// kernel out of the node interpolates only the smooth part of the density
/// and evaluates the node factor exactly ([`SplitDensity`]).
///
/// The spread stays the predictive one. The Poisson node factor is
/// log-concave with at most linear growth in `z`, so the posterior is
/// dominated by a shifted Gaussian of the predictive variance: against that
/// envelope the ratio is a bounded, mild tilt, and the quadrature is benign
/// at any order. Scaling the envelope down to the posterior variance would
/// leave the integrand's Gaussian tails outside it and break the quadrature
/// instead (a survival forecast above one is the symptom). The variance still
/// narrows across nodes, through the predictive recursion. A node whose
/// likelihood factor is much sharper than the predictive spread (a large
/// integrated intensity at one node) makes the ratio a narrow bump that the
/// rule resolves only at high order; the mesh refinement of the fit is what
/// keeps every node mildly informative.
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
    let prior_grid = Grid::new(gh, &zero, &unit, like);
    let rough = node_terms(&prior_grid, false);
    let rough_state = condition(
        &prior_grid,
        &log_standard_prior(&prior_grid, like),
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&prior_grid, &rough_state.alpha, label)?;
    let grid = Grid::new(gh, &means, &unit, like);
    let log_predicted = log_standard_prior(&grid, like);
    let likelihood = node_terms(&grid, derivatives);
    let state = condition(&grid, &log_predicted, &likelihood.ell, likelihood.shift, label)?;
    let density =
        conditioned_density(log_predicted.clone(), None, &likelihood, &state.log_normaliser);
    Ok(FilteredNode {
        grid: Arc::new(grid),
        transitions: Vec::new(),
        log_predicted,
        log_alpha: state.log_alpha,
        alpha: state.alpha,
        log_normaliser: state.log_normaliser,
        likelihood,
        density,
    })
}

/// The predictive grid across one gap (centre `φ · mean`, spread
/// `√(φ² var + q)`) and the log predicted density on it, from the filtered
/// log density on `previous_grid`.
pub(crate) fn predict<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    previous_grid: &Grid<S>,
    previous_log_alpha: &[S],
    previous_density: &SplitDensity<S>,
    transitions: &[AtomTransition<S>],
    label: &str,
) -> Result<(Grid<S>, Vec<S>), EventHistoryError> {
    let atoms = transitions.len();
    if atoms > 0 && transitions.iter().all(|t| t.innovation.value() == 0.0) {
        return Ok((previous_grid.clone(), previous_log_alpha.to_vec()));
    }
    let previous_alpha: Vec<S> = previous_log_alpha.iter().map(exp).collect();
    let (means, variances) = posterior_moments(previous_grid, &previous_alpha, label)?;
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
    let log_predicted =
        ForwardKernel::new(gh, previous_grid, previous_density, &predictive, transitions)
            .log_predicted(predictive.size());
    Ok((predictive, log_predicted))
}

/// Predict the filtered state on `previous_grid` across one gap and condition
/// on a node, on a grid placed at the posterior mean with the predictive
/// spread (see [`filter_start`]).
pub(crate) fn filter_step<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    previous_grid: &Grid<S>,
    previous_log_alpha: &[S],
    previous_density: &SplitDensity<S>,
    transitions: Vec<AtomTransition<S>>,
    node_terms: &dyn Fn(&Grid<S>, bool) -> NodeLikelihood<S>,
    derivatives: bool,
    label: &str,
) -> Result<FilteredNode<S>, EventHistoryError> {
    if !transitions.is_empty() && transitions.iter().all(|t| t.innovation.value() == 0.0) {
        let likelihood = node_terms(previous_grid, derivatives);
        let state = condition(previous_grid, previous_log_alpha, &likelihood.ell,
            likelihood.shift, label)?;
        // The state does not move: the density conditioned on is the previous
        // one, whose explicit factors stay explicit.
        let density = conditioned_density(previous_density.smooth.clone(),
            Some(&previous_density.factor), &likelihood, &state.log_normaliser);
        return Ok(FilteredNode { grid: Arc::new(previous_grid.clone()), transitions,
            log_predicted: previous_log_alpha.to_vec(), log_alpha: state.log_alpha,
            alpha: state.alpha, log_normaliser: state.log_normaliser, likelihood, density });
    }
    let (predictive, rough_predicted) =
        predict(gh, like, previous_grid, previous_log_alpha, previous_density, &transitions, label)?;
    let rough = node_terms(&predictive, false);
    let rough_state = condition(
        &predictive,
        &rough_predicted,
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&predictive, &rough_state.alpha, label)?;
    let scales: Vec<S> = predictive
        .axes
        .iter()
        .map(|axis| axis.sigma.clone())
        .collect();
    let grid = Grid::new(gh, &means, &scales, like);
    let log_predicted = ForwardKernel::new(gh, previous_grid, previous_density, &grid, &transitions)
        .log_predicted(grid.size());
    let likelihood = node_terms(&grid, derivatives);
    let state = condition(&grid, &log_predicted, &likelihood.ell, likelihood.shift, label)?;
    let density =
        conditioned_density(log_predicted.clone(), None, &likelihood, &state.log_normaliser);
    Ok(FilteredNode {
        grid: Arc::new(grid),
        transitions,
        log_predicted,
        log_alpha: state.log_alpha,
        alpha: state.alpha,
        log_normaliser: state.log_normaliser,
        likelihood,
        density,
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
    evaluation: Evaluation,
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
    let derivatives = matches!(evaluation, Evaluation::Derivatives { .. });
    let filtered = filter_nodes(inputs, derivatives, &counts_rows, &exposure_rows)?;
    let node_loglik: Vec<S> = filtered
        .iter()
        .map(|node| node.log_normaliser.add(&like.constant_like(node.likelihood.shift)))
        .collect();
    let loglik = pairwise_sum(&node_loglik, &zero);
    if !loglik.value().is_finite() {
        return Err(numerical("subject marginal log-likelihood is not finite"));
    }
    let tolerance = match evaluation {
        Evaluation::Value => {
            return Ok(SubjectOutput {
                loglik,
                gradient: Vec::new(),
                hessian: Vec::new(),
            });
        }
        Evaluation::Derivatives { tolerance } => tolerance,
    };

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
    // Static atoms put every node on the one whole-history grid of
    // `static_state::filter`, with no gap between nodes to score.
    let is_static = crate::static_state::is_static(inputs.rates);
    let n_gaps = n_nodes.saturating_sub(1);
    let smoothed_chain =
        backward_smoother(inputs, &filtered, &counts_rows, &exposure_rows, !is_static, tolerance)?;

    // ---- forward sweep: Fisher mean and Louis second moment ---------------
    // `carried[q * size + i]` is `C_m(z_i)[q]`, the conditional expectation
    // given `z_m = z_i` and the data up to `m` of the complete-data score
    // accumulated over the nodes and gaps before `m`, on node `m`'s filtered
    // grid; `carried_smoothed` is the same function on the node's smoothed
    // grid, where it has one (see [`backward_smoother`]). Every expectation
    // of node `m` is a sum over the smoothed grid, so it contracts against
    // `carried_smoothed`; `carried` is what moves to the next node, because
    // `C_{m+1} = F[α_m C_m] / p̂_{m+1}` needs its operand beside `α_m`, and
    // the forward operator then evaluates it on both of node `m+1`'s grids.
    // Neither is ever interpolated without the filtered envelope. `mean` is `E[g | y]`
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
    let mut carried_smoothed: Option<Vec<S>> = smoothed_chain.grids[0]
        .as_ref()
        .filter(|_| latent_variance)
        .map(|grid| vec![zero.clone(); p_total * grid.size()]);
    // `ζ_{dk}(i) = ∂η_d/∂a_{dk}` per mark and atom at every point of a grid.
    let centred_on = |grid: &Grid<S>| -> Vec<Vec<Vec<S>>> {
        (0..marks)
            .map(|d| {
                (0..atoms)
                    .map(|k| {
                        let a = &inputs.loadings[d * atoms + k];
                        (0..grid.size())
                            .map(|i| {
                                let z = grid.coordinate(i, k);
                                if prior_centred { z.sub(a) } else { z.clone() }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    };
    // Powers `z_k^a`, `a ≤ 4`, of each atom's coordinate at every point of a grid.
    let powers_on = |grid: &Grid<S>| -> Vec<Vec<Vec<S>>> {
        (0..atoms)
            .map(|k| {
                (0..=4usize)
                    .map(|a| {
                        (0..grid.size())
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
            .collect()
    };
    // Node `d`'s functions join a carried vector on `grid`: `s_d x` in every
    // design slot and `s_d ζ_{dk}` in every loading slot.
    let absorb = |target: &mut [S], rows: &[Vec<(usize, f64)>], likelihood: &NodeLikelihood<S>,
                  centred: &[Vec<Vec<S>>], size: usize| {
        for d in 0..marks {
            let scores = &likelihood.score[d * size..(d + 1) * size];
            for &(col, x) in &rows[d] {
                let row = &mut target[col * size..(col + 1) * size];
                for i in 0..size {
                    row[i] = row[i].add(&scores[i].scale(x));
                }
            }
            for k in 0..atoms {
                let col = layout.a(d, k);
                let row = &mut target[col * size..(col + 1) * size];
                for i in 0..size {
                    row[i] = row[i].add(&scores[i].mul(&centred[d][k][i]));
                }
            }
        }
    };
    // Reused by every node and mark: the grid's size is `order^atoms`, the
    // same at every node, so this is one allocation rather than one per mark
    // per node of a vector as wide as the grid.
    let mut weighted: Vec<S> = vec![zero.clone(); filtered[0].grid.size()];
    for m in 0..n_nodes {
        let grid = smoothed_chain.grid(m, &filtered);
        let size = grid.size();
        let smoothed = &smoothed_chain.marginals[m];
        // `W(i) = w_i s(i)`: the smoothed probability of grid point `i`.
        let w: Vec<S> = (0..size)
            .map(|i| grid.weights[i].mul(&smoothed[i]))
            .collect();
        let rows: Vec<Vec<(usize, f64)>> = (0..marks).map(|d| design_row(m, d)).collect();
        let centred = centred_on(grid);
        // The node's score and curvature in every mark's `η`, formed once
        // with the node factor itself, on the smoothed grid.
        let own_likelihood = smoothed_chain.grids[m].as_ref().map(|grid| {
            subject_node_likelihood(inputs, &counts_rows, &exposure_rows, grid, m, true)
        });
        let node = own_likelihood.as_ref().unwrap_or(&filtered[m].likelihood);
        let contracted: &[S] = carried_smoothed.as_deref().unwrap_or(&carried);
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
                    let row = &contracted[q * size..(q + 1) * size];
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
                // Design entries scale one at a time: each is exact data, and
                // their product would be a rounded factor.
                let ec1 = ec.scale(x1);
                for &(c2, x2) in &rows[d] {
                    curvature[c1 * p_total + c2] =
                        curvature[c1 * p_total + c2].sub(&ec1.scale(x2));
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
                    let m001 = m00.scale(x1);
                    for &(c2, x2) in &rows[d2] {
                        second[c1 * p_total + c2] =
                            second[c1 * p_total + c2].add(&m001.scale(x2));
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
        // ---- pass 3: the node's functions join the carried vectors ---------
        if latent_variance {
            match carried_smoothed.as_mut() {
                Some(on_smoothed) => {
                    absorb(on_smoothed, &rows, node, &centred, size);
                    let filtered_grid = &filtered[m].grid;
                    absorb(&mut carried, &rows, &filtered[m].likelihood,
                        &centred_on(filtered_grid), filtered_grid.size());
                }
                None => absorb(&mut carried, &rows, node, &centred, size),
            }
        }
        if m == n_gaps {
            break;
        }
        // The gap carries the latent moments across it and scores its own
        // rate. Without atoms there is neither: nothing is carried, no rate
        // is a coefficient, and the whole section is a step over a state that
        // does not exist. A static atom has no rate to score, and its state
        // does not move: on the one whole-history grid `z_{m+1} = z_m`, so
        // `E[C_m(z_m) | z_{m+1} = z]` is `C_m(z)` itself.
        if !latent_variance || is_static {
            continue;
        }
        // ---- gap m: (m, m+1) ------------------------------------------------
        let next = &filtered[m + 1];
        let transitions = &next.transitions;
        let backward_moments = &smoothed_chain.innovation_moments[m];
        let polys: Vec<(GapPolynomial<S>, GapPolynomial<S>)> = (0..atoms)
            .map(|k| gap_score_polynomials(&transitions[k], like))
            .collect();
        let unit = |k: usize, b: u8| -> Vec<u8> {
            let mut e = vec![0u8; atoms];
            e[k] = b;
            e
        };
        // Powers of the start coordinate of each atom on node m's smoothed
        // grid, where the gap's expectations are taken.
        let powers = powers_on(grid);
        let contracted: &[S] = carried_smoothed.as_deref().unwrap_or(&carried);
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
                let row = &contracted[q * size..(q + 1) * size];
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
        // ---- propagate the carried vector to node m+1's grids ------------------
        // C_{m+1}(z') = E[C_m(z) | z', y_{≤m}], plus the gap score's own
        // expectation E[t_g | z', y_{≤m}] in its log-rate slot, both under the
        // forward kernel's normalised inner weights at every target point: the
        // carried functions through their Lagrange interpolant on the SOURCE
        // grid, the gap polynomial exactly at the inner points. Every weight is
        // positive and sums to one, so each propagated value is a bounded
        // average of bounded functions wherever the predicted density is.
        //
        // #3013 slice 2 — which grid the carrier is read on is the same question
        // slice 1 answered for the marginal, and it has the same answer. The
        // conditional `p(z_m | z_{m+1}, y_{≤m}) ∝ α_m(z)·f(z'|z)` is what both
        // routes integrate; only the quadrature points differ. On a slow atom's
        // early nodes the filtered grid is as wide as the prior, and
        // `C_m(z) ≈ e^{a z}` read there is an envelope times a degree-`G−1`
        // polynomial whose truncation is `(a·hull)^G / G!` — which is why the
        // Louis Hessian stopped converging geometrically in `G` while the
        // marginal, moved onto its own grid by slice 1, did not. For a `z'` in
        // node m+1's smoothed region the mass of `α_m(z)·f(z'|z)` sits where
        // node m's SMOOTHED grid is, so that is where the carrier is read.
        //
        // The density the kernel is built from is the filtered one either way:
        // `ln α_m = ln p̂_m + f_m`, with the smooth half read at the source
        // grid's points (`Smoothed::predicted`) and `f_m` the explicit node
        // factor, which is a function and not grid values, so it travels
        // unchanged. Nothing about the conditional changes — only where it is
        // sampled.
        //
        // `carried` still moves on the filtered chain: the last node has no
        // smoothed grid of its own, and its expectations contract against that
        // carrier.
        //
        // A gap of zero length leaves the state where it is: `z' = z`, the
        // carried functions pass through unchanged, and the gap's score is
        // identically zero (its `κ` does not depend on the rate). The filtered
        // grid does not move across it, so `carried` stays as it is; node
        // m+1's smoothed grid is a grid of its own, and the kernel, whose
        // inner rule collapses onto `z = z'/φ` at `q = 0`, carries the
        // functions to it.
        let zero_gap = transitions.iter().all(|t| t.innovation.value() == 0.0);
        let propagate = |source_grid: &Grid<S>,
                         source_density: &SplitDensity<S>,
                         source_values: &[S],
                         target: &Grid<S>|
         -> Vec<S> {
            let source_size = source_grid.size();
            let kernel =
                ForwardKernel::new(inputs.gh, source_grid, source_density, target, transitions);
            let target_size = target.size();
            let mut propagated = vec![zero.clone(); p_total * target_size];
            for j in 0..target_size {
                let row = kernel.row(j);
                let transfer = kernel.transfer(j, &row.weights);
                for q in 0..p_total {
                    let carried_q = &source_values[q * source_size..(q + 1) * source_size];
                    propagated[q * target_size + j] = transfer
                        .iter()
                        .zip(carried_q.iter())
                        .fold(zero.clone(), |acc, (t, c)| acc.add(&t.mul(c)));
                }
                if zero_gap {
                    continue;
                }
                let marginals = kernel.axis_marginals(&row.weights);
                for k in 0..atoms {
                    let (t, _) = &polys[k];
                    let mut expected = zero.clone();
                    for a in 0..=2usize {
                        for b in 0..=2usize {
                            if t.absent(a, b) {
                                continue;
                            }
                            let moment = kernel.innovation_moment(j, &marginals[k], k, b, a);
                            expected = expected.add(&t.get(a, b).mul(&moment));
                        }
                    }
                    let slot = layout.rho(k) * target_size + j;
                    propagated[slot] = propagated[slot].add(&expected);
                }
            }
            propagated
        };
        let filtered_source = &filtered[m];
        // Node m's smoothed grid with the FILTERED density read on it: the
        // smooth half at this grid's points, the node factor unchanged.
        let smoothed_source = smoothed_chain.grids[m]
            .as_ref()
            .zip(smoothed_chain.predicted[m].as_ref())
            .map(|(grid, smooth)| {
                (
                    grid,
                    SplitDensity {
                        smooth: smooth.clone(),
                        factor: filtered_source.density.factor.clone(),
                    },
                )
            });
        // The carrier moves from the smoothed grid only where the node has one
        // AND the carrier is already on it; at the first node without one it
        // falls back to the filtered chain, which is where it has been all along.
        let next_smoothed = smoothed_chain.grids[m + 1].as_ref().map(|target| {
            match (smoothed_source.as_ref(), carried_smoothed.as_deref()) {
                (Some((grid, density)), Some(values)) => propagate(grid, density, values, target),
                _ => propagate(
                    &filtered_source.grid,
                    &filtered_source.density,
                    &carried,
                    target,
                ),
            }
        });
        carried_smoothed = next_smoothed;
        if !zero_gap {
            carried = propagate(
                &filtered_source.grid,
                &filtered_source.density,
                &carried,
                &next.grid,
            );
        }
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
    // is carried along with the value. A static atom's rate is held at zero
    // with no gap score, so its slots stay zero and `evaluate_generic` drops
    // them; converting them would form `0 · ∞`.
    let converted = if is_static { 0 } else { atoms };
    let inverse_rates: Vec<S> = inputs.rates[..converted].iter().map(recip).collect();
    let rate_gradients: Vec<S> = (0..converted).map(|k| mean[layout.rho(k)].clone()).collect();
    for k in 0..converted {
        let rho_k = layout.rho(k);
        let inv_k = &inverse_rates[k];
        for j in 0..converted {
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

/// Node `n`'s likelihood pieces on `grid`, for one subject.
fn subject_node_likelihood<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    counts_rows: &[Vec<f64>],
    exposure_rows: &[Vec<f64>],
    grid: &Grid<S>,
    n: usize,
    derivatives: bool,
) -> NodeLikelihood<S> {
    let marks = inputs.nodes.counts.ncols();
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
        inputs.rates.len(),
        derivatives,
    )
}

/// The forward filter over every node of a subject, each node keeping the
/// transitions that reached it so a backward pass can be run on the result.
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
    if crate::static_state::is_static(inputs.rates) {
        // Every node shares the one whole-history grid of `static_state::filter`,
        // so a node's scores live on the grid the carried vector already lives on,
        // and the likelihood the pass conditioned on is the node's own.
        let (pass, likelihoods) =
            crate::static_state::conditioned(inputs, None, &vec![true; marks], derivatives)?;
        let ForwardPass { grids, log_alpha, log_predicted, log_normalisers, densities } = pass;
        return Ok(grids.into_iter().zip(log_alpha).zip(log_predicted).zip(log_normalisers)
            .zip(densities).zip(likelihoods)
            .map(|(((((grid, log_alpha), log_predicted), log_normaliser), density), likelihood)| {
                let alpha = log_alpha.iter().map(exp).collect();
                FilteredNode { grid, transitions: Vec::new(), log_predicted, log_alpha, alpha,
                    log_normaliser: add_real(&log_normaliser, -likelihood.shift), likelihood,
                    density }
            })
            .collect());
    }
    let mut filtered: Vec<FilteredNode<S>> = Vec::with_capacity(n_nodes);
    let node_terms = |grid: &Grid<S>, n: usize, store: bool| -> NodeLikelihood<S> {
        subject_node_likelihood(inputs, counts_rows, exposure_rows, grid, n, store)
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
            &filtered[n].log_alpha,
            &filtered[n].density,
            transitions,
            &|grid, store| node_terms(grid, n + 1, store),
            derivatives,
            &format!("node {}", n + 1),
        )?;
        filtered.push(step);
    }
    Ok(filtered)
}

/// What the backward pass yields: the grid every node's smoothed marginal
/// lives on, that marginal (normalised to a probability), and the smoothed
/// innovation moments of every gap when they were asked for, on the grid of
/// the gap's start node.
struct Smoothed<S> {
    /// A node's own smoothed grid, or `None` where the smoothed marginal
    /// lives on the node's filtered grid: the last node, whose smoothed
    /// marginal is its filtered one, and every node of a static chain.
    grids: Vec<Option<Grid<S>>>,
    /// `ln p̂_n` on the node's own smoothed grid: the SMOOTH half of the
    /// filtered log density `ln α_n = ln p̂_n + f_n`, and the only half that is
    /// bound to a grid. `f_n` is a [`LogFactor`], the explicit node function the
    /// kernel evaluates wherever it needs it, so the filtered density on the
    /// smoothed grid is `SplitDensity { smooth: predicted[n], factor:
    /// filtered[n].density.factor }` — the same density the filtered grid
    /// carries, read at the smoothed grid's points. `None` exactly where
    /// `grids[n]` is.
    predicted: Vec<Option<Vec<S>>>,
    marginals: Vec<Vec<S>>,
    innovation_moments: Vec<HashMap<Vec<u8>, Vec<S>>>,
}

impl<S> Smoothed<S> {
    /// The grid node `n`'s smoothed marginal lives on.
    fn grid<'a>(&'a self, n: usize, filtered: &'a [FilteredNode<S>]) -> &'a Grid<S> {
        self.grids[n].as_ref().unwrap_or(&filtered[n].grid)
    }
}

/// A density `exp(log raw + log β)` on `grid`, normalised to a probability,
/// with the mass its unresolved points carry certified against `tolerance`.
///
/// `β` alone overflows on a wide hull (it is a future-likelihood ratio,
/// astronomically large where `raw` is astronomically small), so the product
/// is never formed as a value. Each point's mass `ln w_i + ln raw_i + ln β_i`
/// is summed by [`log_sum_exp`] and every point is divided by that total in
/// the log, which bounds each normalised value by `1/w_i` before anything is
/// exponentiated. Normalising in the value domain is what forced the old
/// relative floor on `raw`: there the product overflowed where `β` was large
/// and underflowed where `raw` was small, and a point cut for either reason
/// took its mass with it unmeasured.
///
/// What the log domain cannot repair is a point whose `raw` is the noise the
/// `operators` interpolating kernels left in the density
/// ([`interpolation_noise_relative`]): `β` there is finite and the product is
/// a number, but the number stands on nothing. A point is resolved when its
/// density stands `1/tolerance` above that noise, so that its own relative
/// error is inside the accuracy the fit certifies. The share of the
/// marginal's mass the unresolved points hold is measured, and a share above
/// `tolerance` is refused: a smoothed marginal most of whose mass is
/// roundoff is a typed failure, not a renormalised answer.
fn smoothed_marginal<S: JetField>(
    grid: &Grid<S>,
    log_raw: &[S],
    log_beta: &[S],
    gh: &GaussHermite,
    operators: usize,
    tolerance: f64,
    label: &str,
) -> Result<Vec<S>, EventHistoryError> {
    let log_peak = log_raw
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);
    if !log_peak.is_finite() {
        return Err(numerical(format!(
            "{label}: the smoothed marginal's filtered density peaks at exp({log_peak}) on the grid"
        )));
    }
    let log_mass: Vec<S> = grid
        .weights
        .iter()
        .zip(log_raw.iter())
        .zip(log_beta.iter())
        .map(|((w, a), b)| ln(w).add(a).add(b))
        .collect();
    let log_total = log_sum_exp(&log_mass);
    if !log_total.value().is_finite() {
        return Err(numerical(format!(
            "{label}: smoothed marginal has log mass {} on the grid",
            log_total.value()
        )));
    }
    let noise = interpolation_noise_relative(gh, operators);
    let resolved_above = log_peak + (noise / tolerance).ln();
    // Every share is at most one, since `log_total` dominates each term, so
    // the sum needs no shift and cannot overflow.
    let mut unresolved_points = 0usize;
    let mut unresolved_share = 0.0_f64;
    for (mass, density) in log_mass.iter().zip(log_raw.iter()) {
        if density.value() <= resolved_above {
            unresolved_points += 1;
            unresolved_share += (mass.value() - log_total.value()).exp();
        }
    }
    if !(unresolved_share <= tolerance) {
        return Err(numerical(format!(
            "{label}: {unresolved_share:.3e} of the smoothed marginal's mass sits on {unresolved_points} of \
             {} grid points whose filtered density is under {noise:.3e} of the grid's peak, the noise \
             {operators} interpolating forward kernels leave in it; the quadrature is certified to \
             {tolerance:.3e}",
            grid.size()
        )));
    }
    Ok(log_raw
        .iter()
        .zip(log_beta.iter())
        .map(|(a, b)| exp(&a.add(b).sub(&log_total)))
        .collect())
}

/// The backward pass over a filtered chain.
///
/// The future likelihood `lik_{n+1} β_{n+1}` is an exponential in the state
/// after an event, so it is never interpolated as a value: its logarithm
/// is interpolated (relative accuracy), and what is carried backward is
/// `log β_n = log E[lik_{n+1} β_{n+1} / c_{n+1} | z_n]`, a bounded smooth
/// function, plus — with `derivatives` — the smoothed innovation moments
/// `E[Π_k u_k^{e_k} | z_n, data]` of every gap, which the gap scores need.
/// Kernel rows are streamed and reduced to those moments; no `S × S`
/// allocation is formed.
///
/// Each node's smoothed marginal gets a grid of its own. The filtered grid
/// is placed where the data up to the node put the state; the smoothed
/// marginal is where the whole history puts it, and for a slow atom the two
/// part: the filtered density at an early node is as wide as the prior, the
/// smoothed one is as narrow as every later event makes it, and a
/// degree-`G−1` interpolant on the filtered grid resolves that narrow bump
/// only at high order. So the node's smoothed grid is placed one-shot at the
/// Rauch-Tung-Striebel moments — the node's filtered moments `m_f, v_f`
/// combined with the next node's smoothed moments `m', v'` through the gap's
/// `φ, q`:
///   `v̂ = φ² v_f + q`,  `J = φ v_f / v̂`,
///   `m_s = m_f + J (m' − φ m_f)`,  `v_s = v_f q / v̂ + J² v'`,
/// exact for a Gaussian chain and a close envelope otherwise. The density
/// `p̂_n lik_n β_n` is then evaluated on it: `p̂_n` through the forward
/// operator from the previous node's filtered grid (the prior at the first
/// node), `lik_n` exactly, and `β_n` by the inner Gauss-Hermite rule over
/// the innovation, with `log β_{n+1}` interpolated on the next node's
/// smoothed grid. The last node's smoothed marginal is its filtered one.
fn backward_smoother<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    filtered: &[FilteredNode<S>],
    counts_rows: &[Vec<f64>],
    exposure_rows: &[Vec<f64>],
    derivatives: bool,
    tolerance: f64,
) -> Result<Smoothed<S>, EventHistoryError> {
    let n_nodes = filtered.len();
    let last = &filtered[n_nodes - 1];
    if crate::static_state::is_static(inputs.rates) {
        if derivatives { return Err(numerical("static frailties have no innovation scores")); }
        return Ok(Smoothed { grids: vec![None; n_nodes], predicted: vec![None; n_nodes],
            marginals: vec![last.alpha.clone(); n_nodes], innovation_moments: Vec::new() });
    }
    let marks = inputs.nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let gh = inputs.gh;
    let like = &inputs.eta0[0];
    let zero = like.constant_like(0.0);
    let n_gaps = n_nodes.saturating_sub(1);
    // How many interpolating forward kernels each node's FILTERED density
    // came through, so the smoother can say what noise it carries
    // ([`interpolation_noise_relative`]). The first node's density is the
    // standard prior evaluated on its own grid, which interpolates nothing,
    // and a gap whose atoms all have zero innovation is one [`filter_step`]
    // carries the previous density across unchanged.
    let mut filter_operators = Vec::with_capacity(n_nodes);
    let mut through = 0usize;
    for node in filtered {
        if !node.transitions.is_empty()
            && node.transitions.iter().any(|t| t.innovation.value() != 0.0)
        {
            through += 1;
        }
        filter_operators.push(through);
    }
    let inner_count = last.grid.size();
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
    if derivatives {
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
    let mut grids: Vec<Option<Grid<S>>> = vec![None; n_nodes];
    let mut predicted: Vec<Option<Vec<S>>> = vec![None; n_nodes];
    let mut marginals: Vec<Vec<S>> = vec![Vec::new(); n_nodes];
    let mut log_beta: Vec<Vec<S>> = vec![Vec::new(); n_nodes];
    log_beta[n_nodes - 1] = vec![zero.clone(); last.grid.size()];
    // The last node's smoothed marginal is its filtered one, so it carries
    // the filter's own kernels and no further one.
    marginals[n_nodes - 1] = smoothed_marginal(&last.grid, &last.log_alpha, &log_beta[n_nodes - 1],
        gh, filter_operators[n_nodes - 1], tolerance, &format!("node {}", n_nodes - 1))?;
    let mut innovation_moments: Vec<HashMap<Vec<u8>, Vec<S>>> = vec![HashMap::new(); n_gaps];
    for n in (0..n_gaps).rev() {
        let label = format!("node {n}");
        let transitions = &filtered[n + 1].transitions;
        let next = grids[n + 1].as_ref().unwrap_or(&filtered[n + 1].grid);
        // ---- the node's smoothed grid, at the Rauch-Tung-Striebel moments --
        let (filtered_means, filtered_variances) =
            posterior_moments(&filtered[n].grid, &filtered[n].alpha, &label)?;
        let (next_means, next_variances) = posterior_moments(next, &marginals[n + 1], &label)?;
        let mut centres = Vec::with_capacity(atoms);
        let mut scales = Vec::with_capacity(atoms);
        for k in 0..atoms {
            let phi = &transitions[k].phi;
            let q = &transitions[k].innovation;
            let v_f = &filtered_variances[k];
            let predictive = square(phi).mul(v_f).add(q);
            let gain = div(&phi.mul(v_f), &predictive);
            centres.push(filtered_means[k].add(
                &gain.mul(&next_means[k].sub(&phi.mul(&filtered_means[k]))),
            ));
            scales.push(sqrt(
                &div(&v_f.mul(q), &predictive).add(&square(&gain).mul(&next_variances[k])),
            ));
        }
        let grid = Grid::new(gh, &centres, &scales, like);
        let size = grid.size();
        // ---- `p̂_n lik_n` on it -------------------------------------------
        // One more interpolating kernel than the previous node's filtered
        // density came through, except at node 0, where the prior is
        // evaluated on the smoothed grid exactly.
        let (log_predicted, operators) = if n == 0 {
            (log_standard_prior(&grid, like), 0)
        } else {
            let previous = &filtered[n - 1];
            (
                ForwardKernel::new(gh, &previous.grid, &previous.density, &grid,
                    &filtered[n].transitions)
                    .log_predicted(size),
                filter_operators[n - 1] + 1,
            )
        };
        let likelihood =
            subject_node_likelihood(inputs, counts_rows, exposure_rows, &grid, n, false);
        let log_raw: Vec<S> = log_predicted
            .iter()
            .zip(likelihood.ell.iter())
            .map(|(p, e)| p.add(&add_real(e, -likelihood.shift)))
            .collect();
        // `log_predicted` is kept whole below: it is the smooth half of this
        // node's filtered density, and the Louis carried vector is propagated
        // from this grid rather than the filtered one (#3013 slice 2).
        // ---- `log β_n` on it ----------------------------------------------
        // The node's log-likelihood is an explicit formula, so it is
        // evaluated exactly at every inner point; only the smoother residual
        // `log β_{n+1}` is interpolated.
        let bases = backward_axis_bases(gh, &grid, next, transitions);
        let log_c = filtered[n + 1].log_normaliser.clone();
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
            if derivatives {
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
        marginals[n] =
            smoothed_marginal(&grid, &log_raw, &log_beta_n, gh, operators, tolerance, &label)?;
        log_beta[n] = log_beta_n;
        innovation_moments[n] = moments;
        grids[n] = Some(grid);
        predicted[n] = Some(log_predicted);
    }
    Ok(Smoothed {
        grids,
        predicted,
        marginals,
        innovation_moments,
    })
}

/// The posterior mean and covariance of the latent state at every node of a
/// subject given its whole history: the moments of the smoothed marginal on
/// each node's smoothed grid. Per node, the mean over the atoms and the row-major
/// `atoms × atoms` covariance.
///
/// `tolerance` is the relative accuracy the fit's quadrature is certified to;
/// a node whose smoothed marginal rests on more than that share of
/// unresolved mass is refused rather than reported ([`smoothed_marginal`]).
pub(crate) fn latent_state_moments(
    inputs: &SubjectInputs<'_, f64>,
    tolerance: f64,
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
    let smoothed =
        backward_smoother(inputs, &filtered, &counts_rows, &exposure_rows, false, tolerance)?;
    Ok(smoothed
        .marginals
        .iter()
        .enumerate()
        .map(|(n, density)| {
            let grid = smoothed.grid(n, &filtered);
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

/// One completed forward filter: per-node grids, filtered log densities, the
/// predicted (pre-update) log densities, and the per-node log normalisers
/// `ln c_n + m_n`, whose running sum is the log predictive probability of the
/// observed counts. A static frailty's pass shares one whole-history grid
/// across the nodes (`super::static_state`): only its total and its final
/// state are resolved, so chronological quantities come from [`spells`].
pub(crate) struct ForwardPass<S> {
    pub grids: Vec<Arc<Grid<S>>>,
    pub log_alpha: Vec<Vec<S>>,
    pub log_predicted: Vec<Vec<S>>,
    pub log_normalisers: Vec<S>,
    /// Each node's filtered log density split for the kernel out of it
    /// ([`SplitDensity`]).
    pub densities: Vec<SplitDensity<S>>,
}

/// Forward filter only, optionally continuing from a filtered state and
/// optionally restricting the compensator to a subset of marks (a forecast
/// conditions on the absorbing marks not having fired).
pub(crate) fn forward_filter<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    initial: Option<(&Grid<S>, &[S], &SplitDensity<S>)>,
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
    if crate::static_state::is_static(inputs.rates) {
        return crate::static_state::filter(inputs, initial, compensated);
    }
    let like = &inputs.eta0[0];
    let mut grids: Vec<Arc<Grid<S>>> = Vec::with_capacity(n_nodes);
    let mut log_alpha: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut log_predicted: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut log_normalisers: Vec<S> = Vec::with_capacity(n_nodes);
    let mut densities: Vec<SplitDensity<S>> = Vec::with_capacity(n_nodes);
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
        Some((grid, filtered, density)) => filter_step(
            gh,
            like,
            grid,
            filtered,
            density,
            transitions_across(inputs.rates, inputs.continuation_gap, inputs.time_scale)?,
            &|grid, _| node_terms(grid, 0),
            false,
            "forecast first node",
        )?,
    };
    log_normalisers.push(first.log_normaliser.add(&like.constant_like(first.likelihood.shift)));
    log_predicted.push(first.log_predicted);
    log_alpha.push(first.log_alpha);
    densities.push(first.density);
    grids.push(first.grid);
    for n in 0..n_nodes - 1 {
        let step = filter_step(
            gh,
            like,
            &grids[n],
            &log_alpha[n],
            &densities[n],
            transitions_across(inputs.rates, nodes.gaps[n], inputs.time_scale)?,
            &|grid, _| node_terms(grid, n + 1),
            false,
            &format!("forecast node {}", n + 1),
        )?;
        log_normalisers.push(step.log_normaliser.add(&like.constant_like(step.likelihood.shift)));
        log_predicted.push(step.log_predicted);
        log_alpha.push(step.log_alpha);
        densities.push(step.density);
        grids.push(step.grid);
    }
    Ok(ForwardPass {
        grids,
        log_alpha,
        log_predicted,
        log_normalisers,
        densities,
    })
}

/// One spell of a subject's follow-up as a chronological diagnostic reads it:
/// from the previous event (or the entry) to the next event, or to the last
/// node for the open tail.
pub(crate) struct Spell {
    /// The node that closes the spell.
    pub node: usize,
    /// `ln P(no event across the spell | the history before it)`.
    pub log_survival: f64,
    /// The roundoff bound of [`Self::log_survival`] as it was formed, which
    /// scales with the log integrals or normalisers it is assembled from.
    pub log_survival_roundoff: f64,
    /// `E[λ_d(t) | the history before t]` for every mark when an event closes
    /// the spell at `t`; `None` for the open tail.
    pub intensities: Option<Vec<f64>>,
}

/// The spells of a subject's follow-up in time order: one per event node, and
/// the open tail when exposure follows the last event.
///
/// Every value conditions on the history before the spell's end alone, so
/// appending later nodes cannot change it. A dynamic factor's filter places
/// every node's grid from the history before it, so its running normalisers
/// and predicted densities are these quantities. A static factor's
/// [`forward_filter`] resolves only the whole history, so its spells are
/// ratios of prefix integrals (`super::static_state::spells`).
pub(crate) fn spells(
    inputs: &SubjectInputs<'_, f64>,
    compensated: &[bool],
) -> Result<Vec<Spell>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    if n_nodes == 0 || marks == 0 || compensated.len() != marks {
        return Err(numerical(
            "spells need nodes, marks and a compensator mask",
        ));
    }
    if crate::static_state::is_static(inputs.rates) {
        return crate::static_state::spells(inputs, compensated);
    }
    let pass = forward_filter(inputs, None, compensated)?;
    let mut spells = Vec::new();
    // A node's normaliser `ln c + shift` sums `size` positive terms (relative
    // error `γ_{size+2}` with the weight and exponential), and the logarithm,
    // the shift and the running sum each add `ε` of their magnitudes.
    let (mut log_survival, mut log_survival_roundoff) = (0.0_f64, 0.0_f64);
    let mut open = false;
    for n in 0..n_nodes {
        if !nodes.is_event(n) {
            let log_normaliser = pass.log_normalisers[n];
            log_survival += log_normaliser;
            log_survival_roundoff += accumulation_growth(pass.grids[n].size() + 2)
                + f64::EPSILON * (2.0 * log_normaliser.abs() + log_survival.abs());
            open = true;
            continue;
        }
        let intensities = expected_intensities(
            &pass.grids[n],
            &pass.log_predicted[n],
            &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings,
            inputs
                .log_normaliser
                .map(|m| &m[n * marks..(n + 1) * marks]),
            marks,
            atoms,
        );
        spells.push(Spell {
            node: n,
            log_survival,
            log_survival_roundoff,
            intensities: Some(intensities),
        });
        (log_survival, log_survival_roundoff) = (0.0, 0.0);
        open = false;
    }
    if open {
        spells.push(Spell {
            node: n_nodes - 1,
            log_survival,
            log_survival_roundoff,
            intensities: None,
        });
    }
    Ok(spells)
}

/// `E[λ_d(z)]` for every mark under a density on a grid, given by its
/// logarithm: the expected intensity of each mark at a node.
pub(crate) fn expected_intensities<S: JetField>(
    grid: &Grid<S>,
    log_density: &[S],
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
                acc = acc.add(&grid.weights[i].mul(&exp(&log_density[i].add(&eta))));
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relative accuracy a fit certifies its quadrature to, which the
    /// smoother reads its own certificate against.
    fn default_quadrature_tolerance() -> f64 {
        crate::family::EventHistorySpec::new(Vec::new()).quadrature_tolerance
    }

    /// A constant future-likelihood ratio cancels in the normalisation, so a
    /// smoothed marginal under `log β = c` is the one under `log β = 0`
    /// however large `c` is. That is the property the value-domain
    /// normalisation did not have: it formed `exp(log raw + log β)` before
    /// dividing, so at `c = 800` every point overflowed to infinity and the
    /// whole marginal was refused for having infinite mass, while the
    /// exponent `log raw + c − (c + ln Σ …)` is representable throughout
    /// (#4559).
    ///
    /// The bar is the arithmetic. Four roundings fall on an exponent of
    /// magnitude `c`: adding `c` to each point's log mass, adding it back
    /// after the log-sum-exp, and the same two on the returned point; the
    /// largest log mass carries its own into the shift and into the total,
    /// so six is the count, and the two marginals agree to `expm1(γ₆ c)`
    /// relatively.
    #[test]
    fn a_constant_smoother_residual_cancels_however_large_it_is_4559() {
        let gh = GaussHermite::new(9).expect("rule");
        let grid = Grid::new(&gh, &[0.0_f64], &[1.0_f64], &0.0_f64);
        let size = grid.size();
        // Any constant in `log_raw` cancels too, so the prior's normalising
        // constant is left off: this is the standard Gaussian's log density
        // up to it.
        let log_raw: Vec<f64> = (0..size)
            .map(|i| {
                let z = *grid.coordinate(i, 0);
                -0.5 * z * z
            })
            .collect();
        let residual = 800.0_f64;
        let tolerance = default_quadrature_tolerance();
        let unit_residual = vec![0.0; size];
        let large_residual = vec![residual; size];
        let plain =
            smoothed_marginal(&grid, &log_raw, &unit_residual, &gh, 1, tolerance, "plain")
                .expect("a Gaussian marginal under a unit residual");
        let shifted =
            smoothed_marginal(&grid, &log_raw, &large_residual, &gh, 1, tolerance, "shifted")
                .expect("the same marginal under a residual no value-domain product can hold");
        let band = (accumulation_growth(6) * residual).exp_m1();
        for (i, (p, s)) in plain.iter().zip(shifted.iter()).enumerate() {
            assert!(
                (s - p).abs() <= band * p,
                "point {i}: the marginal under log β = {residual} reads {s:e}, the one under \
                 log β = 0 reads {p:e}, apart by {:.3e} relative, past the {band:.3e} that six \
                 roundings of an exponent of magnitude {residual} carry",
                (s - p).abs() / p
            );
        }
        // The normalised marginal integrates to one over the grid. The
        // weights make the round trip `ln` then `exp` inside the normaliser,
        // each of the `size` log masses is exponentiated and summed, its
        // logarithm is taken, each point is exponentiated again, and this
        // sum multiplies and adds `size` terms of its own.
        let mass: f64 = grid.weights.iter().zip(plain.iter()).map(|(w, p)| w * p).sum();
        assert!(
            (mass - 1.0).abs() <= accumulation_growth(3 * size + 5),
            "the normalised marginal integrates to {mass}, not to one within the {:.3e} that \
             {} roundings carry",
            accumulation_growth(3 * size + 5),
            3 * size + 5
        );
    }

    /// The smoother refuses a marginal whose unresolved points carry more
    /// than the quadrature's own tolerance of its mass, and accepts the same
    /// mass distribution when those points are resolved (#4559, #3998).
    ///
    /// Both arms put one point of an otherwise flat grid at the peak and
    /// every other point at a density `f` of it; the residual `log β` at one
    /// off-peak point is set so that point carries exactly the peak's mass,
    /// which is half the total — ten times the tolerance. The arms differ
    /// only in `f`: a tenth of the resolution bar `noise / tolerance` in the
    /// first and ten times it in the second. So the refusal is about what the
    /// representation resolves, not about where the mass sits.
    #[test]
    fn a_marginal_resting_on_unresolved_points_is_refused_4559() {
        let gh = GaussHermite::new(9).expect("rule");
        let grid = Grid::new(&gh, &[0.0_f64], &[1.0_f64], &0.0_f64);
        let size = grid.size();
        let operators = 3;
        let tolerance = default_quadrature_tolerance();
        let bar = interpolation_noise_relative(&gh, operators) / tolerance;
        assert!(
            bar > 0.0 && bar < 1.0,
            "the resolution bar {bar:e} must sit between the grid's noise and its peak"
        );
        // Order nine puts the peak at the middle node; the carrier is the
        // first, which the fixture drives with the residual.
        let (peak, carrier) = (size / 2, 0);
        let mut refusal_message = None;
        for (factor, resolved) in [(0.1 * bar, false), (10.0 * bar, true)] {
            let log_factor = factor.ln();
            let log_raw: Vec<f64> = (0..size)
                .map(|i| if i == peak { 0.0 } else { log_factor })
                .collect();
            // The carrier point holds exactly the peak's mass:
            // `w_c f e^B = w_p`, so `B = ln(w_p / (w_c f))`.
            let carrier_residual = (grid.weights[peak] / (grid.weights[carrier] * factor)).ln();
            let log_beta: Vec<f64> = (0..size)
                .map(|i| if i == carrier { carrier_residual } else { 0.0 })
                .collect();
            let outcome =
                smoothed_marginal(&grid, &log_raw, &log_beta, &gh, operators, tolerance, "carrier");
            match (resolved, outcome) {
                (true, Ok(marginal)) => {
                    let carried = grid.weights[carrier] * marginal[carrier];
                    let at_peak = grid.weights[peak] * marginal[peak];
                    // Both masses are rebuilt from exponents of magnitude
                    // `|ln f| + |B|`, six roundings apart (as above).
                    let band = accumulation_growth(6) * (log_factor.abs() + carrier_residual.abs());
                    assert!(
                        (carried - at_peak).abs() <= band * at_peak,
                        "the resolved carrier holds {carried:e} of the mass against the peak's \
                         {at_peak:e}, apart by more than the {band:.3e} the fixture's own \
                         exponents carry; it is built to hold exactly as much"
                    );
                }
                (true, Err(error)) => panic!(
                    "a marginal whose every point stands above {bar:e} of the peak was refused: {error}"
                ),
                (false, Ok(_)) => panic!(
                    "half the marginal's mass sits on a point at {factor:e} of the peak, under the \
                     resolution bar {bar:e}, and it was accepted"
                ),
                (false, Err(error)) => refusal_message = Some(error.to_string()),
            }
        }
        let refusal = refusal_message.expect("the unresolved arm must refuse");
        assert!(
            refusal.contains("smoothed marginal's mass"),
            "the refusal must name the mass the unresolved points hold: {refusal}"
        );
    }

    #[test]
    fn transition_polynomials_are_exact_scores_of_the_log_density() {
        // Finite differences are permitted in tests: the gap polynomial must
        // equal the derivative of the log transition density in log-rate.
        let z = 0.4;
        let zp = -0.2;
        let gap = 0.6;
        let log_density = |rho: f64| {
            let phi = (-(rho.exp() * gap)).exp();
            let v = 1.0 - phi * phi;
            -0.5 * (2.0 * std::f64::consts::PI * v).ln() - (zp - phi * z).powi(2) / (2.0 * v)
        };
        let rho = -0.3;
        let h = 1e-5;
        let fd1 = (log_density(rho + h) - log_density(rho - h)) / (2.0 * h);
        let fd2 =
            (log_density(rho + h) - 2.0 * log_density(rho) + log_density(rho - h)) / (h * h);
        let kappa = rho.exp() * gap;
        let transition = AtomTransition::new(&kappa);
        let (t, dt) = gap_score_polynomials(&transition, &0.0);
        let phi = (-kappa).exp();
        let u = (zp - phi * z) / (1.0 - phi * phi).sqrt();
        let evaluate = |c: &[f64]| -> f64 {
            let mut total = 0.0;
            for a in 0..5 {
                for b in 0..5 {
                    total += c[a * 5 + b] * z.powi(a as i32) * u.powi(b as i32);
                }
            }
            total
        };
        assert!(
            (evaluate(&t.c) - fd1).abs() < 1e-7,
            "score {} vs fd {fd1}",
            evaluate(&t.c)
        );
        assert!(
            (evaluate(&dt.c) - fd2).abs() < 1e-5,
            "score derivative {} vs fd {fd2}",
            evaluate(&dt.c)
        );
    }

    /// The Louis sweep and the computed path's block sweep read one grid at
    /// every node (#2965). `subject_marginal` places a subject's grids through
    /// `filter_nodes`: the Louis sweep with derivatives at the scalar, the block
    /// sweep without them at its nested jet. The tests comparing the two routes
    /// take the grids' positions and weights as the rule both evaluate, so a
    /// route that placed its own grid would owe their rounding. For static and
    /// dynamic atoms at two orders, every node's positions and weights must be
    /// the same doubles.
    #[test]
    fn louis_and_block_sweeps_read_bit_identical_grids_2965() {
        use crate::scalar::{Rows, TANGENT_WIDTH};
        use ndarray::Array2;
        type Block = Rows<Rows<f64, TANGENT_WIDTH>, TANGENT_WIDTH>;
        fn grid_bits<S: JetField>(grid: &Grid<S>) -> Vec<u64> {
            let mut bits: Vec<u64> = grid.weights.iter().map(|w| w.value().to_bits()).collect();
            for axis in &grid.axes {
                bits.extend([axis.mu.value().to_bits(), axis.sigma.value().to_bits()]);
                bits.extend(axis.points.iter().chain(&axis.weights).map(|x| x.value().to_bits()));
            }
            bits
        }
        let times = [0.0, 0.3, 0.7, 1.0];
        let events = [0.0, 1.0, 0.0, 1.0];
        let n_nodes = times.len();
        let nodes = SubjectNodes {
            first_row: 0,
            times: times.to_vec(),
            gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
            weights: vec![0.25; n_nodes],
            exposures: Array2::from_elem((n_nodes, 1), 0.25),
            counts: Array2::from_shape_fn((n_nodes, 1), |(n, _)| events[n]),
            covariate_rows: vec![0; n_nodes],
        };
        let design = Array2::from_shape_fn((n_nodes, 2), |(n, j)| if j == 0 { 1.0 } else { times[n] });
        let views = [design.view()];
        let beta = [-0.4, 0.3];
        let loadings = [0.8, 0.5];
        let counts_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.counts.row(n).to_vec()).collect();
        let exposure_rows: Vec<Vec<f64>> = (0..n_nodes).map(|n| nodes.exposure_row(n)).collect();
        // Coordinate `q` of `[β | a]` seeded on both levels of the block jet;
        // a `q` past the width seeds a constant.
        let seed = |value: f64, q: usize| -> Block {
            let tangent: [f64; TANGENT_WIDTH] = std::array::from_fn(|k| f64::from(k == q));
            Rows::seed(Rows::seed(value, tangent), tangent)
        };
        let eta0: Vec<f64> = times.iter().map(|t| beta[0] + beta[1] * t).collect();
        let block_eta0: Vec<Block> = times.iter().zip(&eta0)
            .map(|(t, eta)| seed(beta[0], 0).add(&seed(beta[1], 1).scale(*t)).with_value(*eta))
            .collect();
        let block_loadings = [seed(loadings[0], 2), seed(loadings[1], 3)];
        let mut compared = 0;
        for rates in [[0.0, 0.0], [0.6, 1.1]] {
            let block_rates = [seed(rates[0], TANGENT_WIDTH), seed(rates[1], TANGENT_WIDTH)];
            for order in [9, 17] {
                let gh = GaussHermite::new(order).unwrap();
                let louis_inputs = SubjectInputs {
                    nodes: &nodes, eta0: &eta0, loadings: &loadings, rates: &rates, time_scale: 1.0,
                    gh: &gh, continuation_gap: 0.0, designs: Some(&views[..]), log_normaliser: None,
                };
                let block_inputs = SubjectInputs {
                    nodes: &nodes, eta0: &block_eta0, loadings: &block_loadings, rates: &block_rates,
                    time_scale: 1.0, gh: &gh, continuation_gap: 0.0, designs: None, log_normaliser: None,
                };
                let louis = filter_nodes(&louis_inputs, true, &counts_rows, &exposure_rows).unwrap();
                let block = filter_nodes(&block_inputs, false, &counts_rows, &exposure_rows).unwrap();
                for (n, (l, b)) in louis.iter().zip(&block).enumerate() {
                    let (l, b) = (grid_bits(&l.grid), grid_bits(&b.grid));
                    compared += l.len();
                    assert_eq!(l, b, "rates {rates:?}, order {order}, node {n}: the Louis and block sweeps read different grids");
                }
            }
        }
        eprintln!("compared {compared} grid positions and weights");
        assert!(compared > 0, "no grid was compared");
    }

    /// A slow atom's first smoothed marginal is resolved at the forward
    /// filter's own accuracy (#3013). The fixture is the issue's: 41 evenly
    /// spaced recurrent events on [0, 6], η = 0.53, loading 0.845, rate
    /// 2.6e-3. Node 0's filtered density there is essentially the N(0, 1)
    /// prior, while its smoothed marginal has seen every event and sits near
    /// z ≈ 1.95 with σ ≈ 0.21, two filtered σ off centre and five times
    /// narrower. The reference is a dense forward-backward pass on a uniform
    /// grid; the trapezoid rule is spectrally accurate for these analytic
    /// integrands, and the reference's own error is measured by doubling
    /// its spacing.
    ///
    /// The bar is derived, not chosen. The smoothed marginal at node 0 is
    /// built from the forward filter, whose own error at order G is visible
    /// at the last node, where smoothed and filtered coincide. So at every
    /// order node 0's smoothed mean and standard deviation must be within the
    /// larger of that last-node error and the reference's own error. On the
    /// filtered grid (the smoother before this issue) node 0 misses by
    /// orders of magnitude more than the filter's error.
    #[test]
    fn slow_atom_first_smoothed_marginal_is_resolved_at_the_filter_accuracy_3013() {
        use ndarray::Array2;
        let n_nodes = 41;
        let spacing = 0.15;
        let rate = 2.6e-3;
        let loading = 0.845;
        let eta0_value = 0.53;
        let times: Vec<f64> = (0..n_nodes).map(|n| n as f64 * spacing).collect();
        let nodes = SubjectNodes {
            first_row: 0,
            times: times.clone(),
            gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
            weights: vec![spacing; n_nodes],
            exposures: Array2::from_elem((n_nodes, 1), spacing),
            counts: Array2::from_elem((n_nodes, 1), 1.0),
            covariate_rows: vec![0; n_nodes],
        };
        let eta0 = vec![eta0_value; n_nodes];
        let loadings = [loading];
        let rates = [rate];
        let phi = (-rate * spacing).exp();
        let q = 1.0 - phi * phi;
        // (mean, sd) of node 0's smoothed marginal and of the last node's
        // filtered one, by dense forward-backward at spacing `h` over ±8.
        let dense = |h: f64| -> [(f64, f64); 2] {
            let half = (8.0 / h).round() as i64;
            let zs: Vec<f64> = (-half..=half).map(|i| i as f64 * h).collect();
            let size = zs.len();
            let likelihood: Vec<f64> = {
                let log: Vec<f64> = zs
                    .iter()
                    .map(|&z| {
                        let eta = log_intensity(&eta0_value, &loadings, &[z], None);
                        eta - spacing * eta.exp()
                    })
                    .collect();
                let top = log.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                log.iter().map(|l| (l - top).exp()).collect()
            };
            let normalise = |v: Vec<f64>| -> Vec<f64> {
                let top = v.iter().cloned().fold(0.0_f64, f64::max);
                v.into_iter().map(|x| x / top).collect()
            };
            // The kernel is negligible past 12 innovation σ.
            let reach = (12.0 * q.sqrt() / h).ceil() as i64;
            let kernel = |from: f64, to: f64| (-(to - phi * from).powi(2) / (2.0 * q)).exp();
            let moments = |density: &[f64]| -> (f64, f64) {
                let mass: f64 = density.iter().sum();
                let mean = density.iter().zip(&zs).map(|(d, z)| d * z).sum::<f64>() / mass;
                let variance = density
                    .iter()
                    .zip(&zs)
                    .map(|(d, z)| d * (z - mean).powi(2))
                    .sum::<f64>()
                    / mass;
                (mean, variance.sqrt())
            };
            let first: Vec<f64> = zs
                .iter()
                .zip(&likelihood)
                .map(|(&z, l)| (-0.5 * z * z).exp() * l)
                .collect();
            let mut forward = normalise(first.clone());
            let mut beta = vec![1.0; size];
            for _ in 1..n_nodes {
                // Forward: from `from` to every `to` in reach of φ·from.
                let mut predicted = vec![0.0; size];
                // Backward: to every `from` whose φ·from reaches `to`.
                let carried: Vec<f64> = likelihood.iter().zip(&beta).map(|(l, b)| l * b).collect();
                let mut backward = vec![0.0; size];
                for i in 0..size {
                    let centre = (phi * zs[i] / h).round() as i64 + half;
                    let lo = (centre - reach).max(0) as usize;
                    let hi = ((centre + reach) as usize).min(size - 1);
                    for j in lo..=hi {
                        let k = kernel(zs[i], zs[j]);
                        predicted[j] += forward[i] * k;
                        backward[i] += k * carried[j];
                    }
                }
                forward = normalise(predicted.iter().zip(&likelihood).map(|(p, l)| p * l).collect());
                beta = normalise(backward);
            }
            let smoothed: Vec<f64> = first.iter().zip(&beta).map(|(a, b)| a * b).collect();
            [moments(&smoothed), moments(&forward)]
        };
        let [smoothed, last] = dense(0.002);
        let [coarse_smoothed, coarse_last] = dense(0.004);
        let reference_error = [
            (smoothed.0 - coarse_smoothed.0).abs(),
            (smoothed.1 - coarse_smoothed.1).abs(),
            (last.0 - coarse_last.0).abs(),
            (last.1 - coarse_last.1).abs(),
        ]
        .into_iter()
        .fold(0.0_f64, f64::max);
        eprintln!(
            "dense: node 0 smoothed N({:.8}, {:.8}²), last filtered N({:.8}, {:.8}²), reference error {reference_error:.2e}",
            smoothed.0, smoothed.1, last.0, last.1
        );
        assert!(smoothed.0 > 1.5, "the fixture's smoothed marginal must sit off the prior");
        for order in [9, 17, 33] {
            let gh = GaussHermite::new(order).unwrap();
            let inputs = SubjectInputs {
                nodes: &nodes, eta0: &eta0, loadings: &loadings, rates: &rates, time_scale: 1.0,
                gh: &gh, continuation_gap: 0.0, designs: None, log_normaliser: None,
            };
            let moments = latent_state_moments(&inputs, default_quadrature_tolerance()).unwrap();
            let node_0 = (moments[0].0[0], moments[0].1[0].sqrt());
            let final_node = (moments[n_nodes - 1].0[0], moments[n_nodes - 1].1[0].sqrt());
            let filter_error = (final_node.0 - last.0).abs().max((final_node.1 - last.1).abs());
            let bar = filter_error.max(reference_error);
            let (mean_error, sd_error) = (node_0.0 - smoothed.0, node_0.1 - smoothed.1);
            eprintln!(
                "order {order}: node 0 mean error {mean_error:+.3e}, sd error {sd_error:+.3e}; bar {bar:.3e} (filter error {filter_error:.3e})"
            );
            assert!(
                mean_error.abs() <= bar && sd_error.abs() <= bar,
                "order {order}: node 0's smoothed marginal N({:.6}, {:.6}²) misses the dense N({:.6}, {:.6}²) by \
                 ({mean_error:+.3e}, {sd_error:+.3e}), past the forward filter's own error {bar:.3e}",
                node_0.0, node_0.1, smoothed.0, smoothed.1
            );
        }
    }
}
