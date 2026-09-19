//! The frozen-time limit of the survival marginal-slope objective (gam#3003).
//!
//! # The boundary the time block's prior cannot see
//!
//! The row negative log-likelihood (`row_math.rs`) is
//!
//! ```text
//!     ℓ_i = w_i [ (1 − d_i)·(−log Φ(−η₁)) + log Φ(−η₀) − d_i·log φ(η₁) − d_i·log η′₁ ].
//! ```
//!
//! Take any path along which every at-risk row index `η` enters the probit's
//! upper tail while the time block's coefficients shrink in proportion,
//! `γ = τ·γ̄` with `τ → 0` and a fixed shape `γ̄ ≥ 0`. The Mills expansion
//! `−log Φ(−η) = η²/2 + log η + ½·log 2π + O(η⁻²)` sends every row term to a
//! proportional-hazards term whose baseline hazard is the shape `γ̄ᵀX_D(t)`:
//!
//! ```text
//!     ℓ_i → w_i [ r_i·γ̄ᵀ(I(t₁) − I(t₀)) − d_i·log(r_i·γ̄ᵀX_D(t₁)) ],
//! ```
//!
//! with the product of `τ` and the index's growth absorbed into `γ̄`. The time
//! penalty is a quadratic form in `γ = τ·γ̄`, so it is exactly zero on this
//! boundary for every smoothing parameter and every shape. That is the defect
//! gam#3003 measured: nothing in the prior resists the boundary, and on few
//! events the likelihood can prefer it.
//!
//! Two coefficient paths reach the boundary on every law the family anchors on.
//!
//! * **Marginal level.** The marginal intercept goes to `+∞` with every other
//!   coefficient at zero. On a finite law the anchoring equation
//!   `Σ_k w_k Φ(−(α + b·u_k)) = Φ(−q)` gives `α = q − b·u_far + o(1)`, so
//!   `∂α/∂q → 1` and every relative risk `r_i → 1`. On the Gaussian closed form
//!   `α = q·c_i`, so `r_i = c_i² = 1 + b_i²` at the slope's offset `b_i`. Every
//!   penalty is zero there.
//! * **Slope hinge.** The slope intercept goes to `±∞` with `q` frozen at the
//!   fitted marginal index `m̂_i`. The anchoring equation then selects the node
//!   `u_k` whose cumulative weight straddles `Φ(−q)`, with
//!   `Φ(ζ) = (Φ(−q) − W_{k−1})/w_k`, and gives `r_i = K(q_i)·(z_i − u_k)₊` with
//!   `K = φ(q)/(w_k·φ(ζ))`, mirrored for a negative slope. On the Gaussian
//!   closed form `r_i = (q_i ± z_i)₊`. The marginal coefficients are kept, so the
//!   limit's penalty is the marginal block's.
//!
//! A row below the hinge has `η → −∞` and contributes nothing. An event row
//! with `r_i = 0`, or a row entering at the time origin (it has no entry
//! factor) inside the tail, makes that path's limit `+∞`.
//!
//! # The certificate
//!
//! Each limit value is the limit of the fit's own objective `−ℓ + penalty`
//! along an explicit coefficient path, so a fit whose objective is not below it
//! is not a mode below the frozen boundary:
//! [`FrozenTimeIdentification::NotIdentified`]. The converse is not certified.
//! A fit below every limit derived here may still be beaten by a limit this
//! module does not construct, so it is [`FrozenTimeUndetermined::NoLimitBelowFit`],
//! never "identified".
//!
//! The derivative guard literal truncates both paths at `q̇ ≥ guard`. The limit
//! is the model's own; the literal only stops a solver short of it.

use super::*;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::probability::{normal_cdf, normal_pdf, standard_normal_quantile};

/// Which side of the latent score a slope-hinge limit keeps at risk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopeHingeSide {
    /// The slope intercept goes to `+∞`: rows above the hinge stay at risk.
    Upper,
    /// The slope intercept goes to `−∞`: rows below the hinge stay at risk.
    Lower,
}

/// The coefficient path a frozen-time limit value is reached along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrozenTimePath {
    /// The marginal intercept goes to `+∞`: relative risk `1` (finite law) or
    /// `1 + b_i²` (Gaussian closed form) on every row.
    MarginalLevel,
    /// The slope intercept goes to `±∞` with `q` frozen at the fitted marginal
    /// index.
    SlopeHinge(SlopeHingeSide),
}

impl std::fmt::Display for FrozenTimePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MarginalLevel => f.write_str("the marginal intercept going to +inf"),
            Self::SlopeHinge(SlopeHingeSide::Upper) => {
                f.write_str("the slope intercept going to +inf at the fitted marginal index")
            }
            Self::SlopeHinge(SlopeHingeSide::Lower) => {
                f.write_str("the slope intercept going to -inf at the fitted marginal index")
            }
        }
    }
}

/// One frozen-time limit: the limit of the fit's objective `−ℓ + penalty`
/// along [`FrozenTimeLimit::path`], with the time block's shape re-optimised
/// on its cone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrozenTimeLimit {
    pub path: FrozenTimePath,
    pub objective: f64,
}

/// What the frozen-time certificate says about one fitted coefficient point.
#[derive(Clone, Debug, PartialEq)]
pub enum FrozenTimeIdentification {
    /// A frozen-time limit is below the fit's objective by more than the two
    /// values' rounding envelope: the fitted point is not a mode below the
    /// boundary on which the time trend vanishes.
    NotIdentified {
        fit_objective: f64,
        limit: FrozenTimeLimit,
    },
    /// The certificate does not refuse the point.
    Undetermined(FrozenTimeUndetermined),
}

/// Why the frozen-time certificate did not refuse a point.
#[derive(Clone, Debug, PartialEq)]
pub enum FrozenTimeUndetermined {
    /// Every derived limit is at or above the fit's objective, or unreachable.
    NoLimitBelowFit {
        fit_objective: f64,
        lowest_limit: Option<FrozenTimeLimit>,
    },
    /// The limits are not derived for this configuration.
    NotDerived { configuration: &'static str },
}

impl FrozenTimeIdentification {
    /// The reason this verdict refuses its coefficient point, when it is a
    /// refusal.
    pub(crate) fn refusal_reason(&self) -> Option<String> {
        match self {
            Self::NotIdentified {
                fit_objective,
                limit,
            } => Some(format!(
                "the fitted objective {fit_objective:.9e} is not below the frozen-time limit \
                 {:.9e} reached along {}: the likelihood prefers the boundary on which the \
                 baseline time trend vanishes (qdot -> 0, where the time penalty is zero), so \
                 these data do not identify the time trend at these smoothing parameters \
                 (gam#3003)",
                limit.objective, limit.path,
            )),
            Self::Undetermined(_) => None,
        }
    }
}

/// The configuration the limits above are derived for, or the first feature of
/// `family` they are not derived for.
fn frozen_time_scope(family: &SurvivalMarginalSlopeFamily) -> Result<(), &'static str> {
    if family.z.ncols() != 1 || family.slope_layout.is_per_score() {
        return Err("a latent score vector (K >= 2)");
    }
    if family
        .latent_law
        .as_ref()
        .is_some_and(|law| law.joint().is_some())
    {
        return Err("a joint latent law");
    }
    if family.score_warp.is_some() {
        return Err("a score-warp block");
    }
    if family.link_dev.is_some() {
        return Err("a link-deviation block");
    }
    if family.influence_absorber.is_some() {
        return Err("an influence absorber block");
    }
    if family.flex_timewiggle_active() {
        return Err("a time wiggle");
    }
    if family.slope_is_follow_up_varying() {
        return Err("a follow-up-varying slope");
    }
    if family.family_hyper.baseline_geometry.is_some() {
        return Err("a nonlinear baseline target");
    }
    if family.jeffreys_armed {
        return Err("an armed Jeffreys prior");
    }
    Ok(())
}

/// The time block's rows as the frozen-time limits read them. The time designs
/// do not move with the smoothing or length-scale coordinates, so one geometry
/// serves every evaluation of a fit.
pub(crate) struct FrozenTimeGeometry {
    /// `Err` names a time-design property the limits are not derived for.
    design_scope: Result<(), &'static str>,
    /// Rows with positive weight that enter after the time origin, their weights
    /// and their exposure rows `I(t₁) − I(t₀)`.
    delayed_rows: Vec<usize>,
    delayed_weight: Array1<f64>,
    exposure: Array2<f64>,
    /// Rows with positive weight that enter at the time origin.
    origin_rows: Vec<usize>,
    /// Rows with `w·d > 0`, their derivative-at-exit rows and `w·d`.
    event_rows: Vec<usize>,
    event_design: Array2<f64>,
    event_weight: Array1<f64>,
}

impl FrozenTimeGeometry {
    pub(crate) fn new(family: &SurvivalMarginalSlopeFamily) -> Result<Self, String> {
        const CHUNK_ROWS: usize = 1024;
        let n = family.n;
        let p = family.design_exit.ncols();
        let mut delayed_rows = Vec::new();
        let mut delayed_weight = Vec::new();
        let mut exposure = Vec::new();
        let mut origin_rows = Vec::new();
        let mut event_rows = Vec::new();
        let mut event_design = Vec::new();
        let mut event_weight = Vec::new();
        let mut design_scope = Ok(());
        let read = |design: &DesignMatrix, rows: std::ops::Range<usize>, label: &str| {
            design
                .try_row_chunk(rows)
                .map_err(|error| format!("frozen-time geometry {label} rows: {error}"))
        };
        let mut start = 0;
        while start < n {
            let end = (start + CHUNK_ROWS).min(n);
            let entry = read(&family.design_entry, start..end, "entry")?;
            let exit = read(&family.design_exit, start..end, "exit")?;
            let derivative = read(&family.design_derivative_exit, start..end, "derivative")?;
            for local in 0..end - start {
                let row = start + local;
                let weight = family.weights[row];
                if !(weight > 0.0) {
                    continue;
                }
                if family.entry_at_origin[row] {
                    origin_rows.push(row);
                } else {
                    delayed_rows.push(row);
                    delayed_weight.push(weight);
                    exposure.extend(
                        exit.row(local)
                            .iter()
                            .zip(entry.row(local).iter())
                            .map(|(&x1, &x0)| x1 - x0),
                    );
                }
                let event_mass = weight * family.event[row];
                if event_mass > 0.0 {
                    if derivative.row(local).iter().any(|&m| !(m >= 0.0)) {
                        design_scope = Err("a time derivative design with a negative entry");
                    }
                    event_rows.push(row);
                    event_weight.push(event_mass);
                    event_design.extend(derivative.row(local).iter().copied());
                }
            }
            start = end;
        }
        let shape = |rows: usize, values: Vec<f64>, label: &str| {
            Array2::from_shape_vec((rows, p), values)
                .map_err(|error| format!("frozen-time geometry {label}: {error}"))
        };
        Ok(Self {
            design_scope,
            exposure: shape(delayed_rows.len(), exposure, "exposure")?,
            delayed_weight: Array1::from_vec(delayed_weight),
            delayed_rows,
            origin_rows,
            event_design: shape(event_rows.len(), event_design, "event design")?,
            event_weight: Array1::from_vec(event_weight),
            event_rows,
        })
    }

    /// The limit of `Σ_i ℓ_i` along a path whose relative risks are `risk(row)`,
    /// with the baseline shape minimised over the cone `γ̄ ≥ 0`. `None` when the
    /// path's limit is `+∞` (an event row out of the tail, an origin row inside
    /// it) or when `risk` is undefined on some row (`Ok(None)` from `risk`).
    fn limit_negative_log_likelihood(
        &self,
        risk: impl Fn(usize) -> Result<Option<f64>, String>,
    ) -> Result<Option<f64>, String> {
        for &row in &self.origin_rows {
            let Some(r) = risk(row)? else {
                return Ok(None);
            };
            if r > 0.0 {
                return Ok(None);
            }
        }
        let mut exposure = Array1::<f64>::zeros(self.exposure.ncols());
        for (at, &row) in self.delayed_rows.iter().enumerate() {
            let Some(r) = risk(row)? else {
                return Ok(None);
            };
            if r > 0.0 {
                exposure.scaled_add(self.delayed_weight[at] * r, &self.exposure.row(at));
            }
        }
        let mut event_log_risk = 0.0;
        for (at, &row) in self.event_rows.iter().enumerate() {
            let Some(r) = risk(row)? else {
                return Ok(None);
            };
            if !(r > 0.0) {
                return Ok(None);
            }
            event_log_risk += self.event_weight[at] * r.ln();
        }
        Ok(cone_minimiser(&exposure, &self.event_design, &self.event_weight)?
            .map(|minimum| minimum.value - event_log_risk))
    }
}

/// `min_{γ ≥ 0} bᵀγ − Σ_e v_e·log(m_eᵀγ)`, by Newton's method projected on the
/// cone: a Newton step on the coordinates that are positive or descending, a
/// projected backtracking search with the Armijo condition, and a stop when an
/// accepted step lowers the value by no more than its rounding. The problem is
/// convex. Every iterate is feasible, so the returned value is attained and
/// bounds the minimum from above wherever the iteration stops. With no event
/// rows the infimum is `0`, at `γ = 0`. `Ok(None)` when no feasible point has a
/// finite value: an event row with no positive column, or a column that carries
/// event mass and no exposure (the value is then unbounded below).
fn cone_minimiser(
    b: &Array1<f64>,
    m: &Array2<f64>,
    v: &Array1<f64>,
) -> Result<Option<ConeMinimum>, String> {
    const ARMIJO: f64 = 1e-4;
    let p = b.len();
    if m.nrows() == 0 {
        return Ok(Some(ConeMinimum { value: 0.0 }));
    }
    let event_mass = m.t().dot(v);
    let live: Vec<usize> = (0..p).filter(|&k| event_mass[k] > 0.0).collect();
    if live.is_empty() || live.iter().any(|&k| !(b[k] > 0.0 && b[k].is_finite())) {
        return Ok(None);
    }
    let value_at = |gamma: &Array1<f64>| -> Option<f64> {
        let hazard = m.dot(gamma);
        let mut log_term = 0.0;
        for (&weight, &h) in v.iter().zip(hazard.iter()) {
            if !(h > 0.0) {
                return None;
            }
            log_term += weight * h.ln();
        }
        Some(b.dot(gamma) - log_term)
    };
    let live_exposure: f64 = live.iter().map(|&k| b[k]).sum();
    let mut gamma = Array1::<f64>::zeros(p);
    for &k in &live {
        gamma[k] = v.sum() / live_exposure;
    }
    let Some(mut value) = value_at(&gamma) else {
        return Ok(None);
    };
    loop {
        let hazard = m.dot(&gamma);
        let ratio = v / &hazard;
        let gradient = b - &m.t().dot(&ratio);
        let free: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&k| gamma[k] > 0.0 || gradient[k] < 0.0)
            .collect();
        if free.is_empty() {
            break;
        }
        let curvature = &ratio / &hazard;
        let mut hessian = Array2::<f64>::zeros((free.len(), free.len()));
        for (event, row) in m.outer_iter().enumerate() {
            for (a, &ka) in free.iter().enumerate() {
                let scaled = curvature[event] * row[ka];
                for (c, &kc) in free.iter().enumerate() {
                    hessian[[a, c]] += scaled * row[kc];
                }
            }
        }
        let (eigenvalues, eigenvectors) = FaerEigh::eigh(&hessian, faer::Side::Lower)
            .map_err(|error| format!("frozen-time shape Newton step: {error}"))?;
        let top = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        let rank_floor = free.len() as f64 * f64::EPSILON * top;
        let free_gradient = Array1::from_iter(free.iter().map(|&k| gradient[k]));
        let mut step = Array1::<f64>::zeros(free.len());
        for (j, &eigenvalue) in eigenvalues.iter().enumerate() {
            if eigenvalue > rank_floor {
                let direction = eigenvectors.column(j);
                step.scaled_add(-direction.dot(&free_gradient) / eigenvalue, &direction);
            }
        }
        let mut accepted = None;
        let mut length = 1.0_f64;
        while length > f64::EPSILON {
            let mut candidate = gamma.clone();
            for (a, &k) in free.iter().enumerate() {
                candidate[k] = (gamma[k] + length * step[a]).max(0.0);
            }
            if let Some(candidate_value) = value_at(&candidate) {
                let predicted = gradient.dot(&(&candidate - &gamma));
                if candidate_value <= value + ARMIJO * predicted {
                    accepted = Some((candidate, candidate_value));
                    break;
                }
            }
            length *= 0.5;
        }
        let Some((candidate, candidate_value)) = accepted else {
            break;
        };
        let decrease = value - candidate_value;
        gamma = candidate;
        value = candidate_value;
        if decrease <= 4.0 * f64::EPSILON * value.abs().max(1.0) {
            break;
        }
    }
    Ok(Some(ConeMinimum { value }))
}

/// The value [`cone_minimiser`] attains.
struct ConeMinimum {
    value: f64,
}

/// The relative risk row `z` keeps in the slope-hinge limit on a finite law,
/// at frozen marginal index `q`. `Ok(None)` when `Φ(−q)` sits exactly on a
/// cumulative weight, where the limit depends on the side it is approached
/// from.
fn grid_hinge_risk(
    q: f64,
    z: f64,
    side: SlopeHingeSide,
    grid: AnchorGrid<'_>,
) -> Result<Option<f64>, String> {
    let survival = normal_cdf(-q);
    let count = grid.nodes.len();
    let order = |at: usize| match side {
        SlopeHingeSide::Upper => at,
        SlopeHingeSide::Lower => count - 1 - at,
    };
    let mut below = 0.0;
    for at in 0..count {
        let k = order(at);
        let weight = grid.weights[k];
        if below + weight >= survival {
            let fraction = (survival - below) / weight;
            if !(fraction > 0.0 && fraction < 1.0) {
                return Ok(None);
            }
            let zeta = standard_normal_quantile(fraction)?;
            let scale = normal_pdf(q) / (weight * normal_pdf(zeta));
            let excess = match side {
                SlopeHingeSide::Upper => z - grid.nodes[k],
                SlopeHingeSide::Lower => grid.nodes[k] - z,
            };
            return Ok(Some(scale * excess.max(0.0)));
        }
        below += weight;
    }
    Ok(None)
}

/// The first coefficient of `spec` whose design column is identically one and
/// which the block's `S_λ` leaves unpenalized: the direction along which a
/// limit path moves the block without cost. Every component of `S_λ` is
/// positive semi-definite with a positive weight, so a zero column of the sum is
/// a zero column of every component.
fn unpenalized_constant_column(spec: &ParameterBlockSpec, s_lambda: &Array2<f64>) -> Option<usize> {
    let p = spec.design.ncols();
    if s_lambda.dim() != (p, p) {
        return None;
    }
    (0..p).find(|&j| {
        let mut unit = Array1::<f64>::zeros(p);
        unit[j] = 1.0;
        s_lambda.column(j).iter().all(|&entry| entry == 0.0)
            && spec
                .design
                .matrixvectormultiply(&unit)
                .iter()
                .all(|&entry| entry == 1.0)
    })
}

/// The frozen-time certificate at one converged coefficient point.
///
/// `blocks` are the fit's block specifications, `states` the point's coefficient
/// states and `s_lambdas` each block's `S_λ` at the point's smoothing
/// parameters; `fit_log_likelihood` and `fit_penalty` are the point's own `ℓ`
/// and `½·βᵀS_λβ`, so the objective compared is the one its solve minimised.
/// The per-block penalties must reproduce `fit_penalty` within the rounding
/// envelope of two independently assembled values, since the slope-hinge limit
/// keeps the marginal block's share of it.
pub(crate) fn frozen_time_identification(
    family: &SurvivalMarginalSlopeFamily,
    geometry: &FrozenTimeGeometry,
    blocks: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    fit_log_likelihood: f64,
    fit_penalty: f64,
) -> Result<FrozenTimeIdentification, String> {
    let not_derived = |configuration: &'static str| -> Result<FrozenTimeIdentification, String> {
        Ok(FrozenTimeIdentification::Undetermined(
            FrozenTimeUndetermined::NotDerived { configuration },
        ))
    };
    if let Err(configuration) = frozen_time_scope(family).and(geometry.design_scope) {
        return not_derived(configuration);
    }
    if blocks.len() != 3 || states.len() != 3 || s_lambdas.len() != 3 {
        return not_derived("a block layout other than time, marginal and slope");
    }
    let mut block_penalties = [0.0; 3];
    for (at, (state, s_lambda)) in states.iter().zip(s_lambdas.iter()).enumerate() {
        let p = state.beta.len();
        if s_lambda.dim() != (p, p) {
            return not_derived("a block whose S_lambda does not match its coefficients");
        }
        block_penalties[at] = 0.5 * state.beta.dot(&s_lambda.dot(&state.beta));
    }
    let penalty_sum: f64 = block_penalties.iter().sum();
    if (penalty_sum - fit_penalty).abs()
        > gam_solve::rho_optimizer::outer_value_agreement_bound(penalty_sum, fit_penalty)
    {
        return not_derived("a penalty that is not the sum of its block penalties");
    }
    let fit_objective = -fit_log_likelihood + fit_penalty;
    let mut limits = Vec::new();
    let law = family.latent_law.as_deref();
    if unpenalized_constant_column(&blocks[1], &s_lambdas[1]).is_some() {
        let probit_scale = family.probit_frailty_scale();
        let zero_slope = Array1::<f64>::zeros(blocks[2].design.ncols());
        let level_risk = |row: usize| -> Result<Option<f64>, String> {
            if law.is_some() {
                return Ok(Some(1.0));
            }
            let offset = family
                .slope_layout
                .row_channels_from_beta(row, zero_slope.view())?
                .exit;
            let observed = probit_scale * offset;
            Ok(Some(1.0 + observed * observed))
        };
        if let Some(objective) = geometry.limit_negative_log_likelihood(level_risk)? {
            limits.push(FrozenTimeLimit {
                path: FrozenTimePath::MarginalLevel,
                objective,
            });
        }
    }
    if unpenalized_constant_column(&blocks[2], &s_lambdas[2]).is_some() {
        let marginal_index = &states[1].eta;
        for side in [SlopeHingeSide::Upper, SlopeHingeSide::Lower] {
            let hinge_risk = |row: usize| -> Result<Option<f64>, String> {
                let q = marginal_index[row];
                let z = family.z[[row, 0]];
                match law {
                    Some(law) => grid_hinge_risk(q, z, side, law.row(row)),
                    None => Ok(Some(match side {
                        SlopeHingeSide::Upper => (q + z).max(0.0),
                        SlopeHingeSide::Lower => (q - z).max(0.0),
                    })),
                }
            };
            if let Some(nll) = geometry.limit_negative_log_likelihood(hinge_risk)? {
                limits.push(FrozenTimeLimit {
                    path: FrozenTimePath::SlopeHinge(side),
                    objective: nll + block_penalties[1],
                });
            }
        }
    }
    let lowest_limit = limits
        .into_iter()
        .filter(|limit| limit.objective.is_finite())
        .min_by(|a, b| a.objective.total_cmp(&b.objective));
    Ok(match lowest_limit {
        Some(limit)
            if limit.objective
                < fit_objective
                    - gam_solve::rho_optimizer::outer_value_agreement_bound(
                        fit_objective,
                        limit.objective,
                    ) =>
        {
            FrozenTimeIdentification::NotIdentified {
                fit_objective,
                limit,
            }
        }
        lowest_limit => FrozenTimeIdentification::Undetermined(
            FrozenTimeUndetermined::NoLimitBelowFit {
                fit_objective,
                lowest_limit,
            },
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::CustomFamily;
    use gam_linalg::matrix::DenseDesignMatrix;

    const ROWS: usize = 48;
    /// The time block's columns are `I_k(t) = 1 − exp(−t/s_k)`: increasing,
    /// with positive derivatives, the two properties the limits read.
    const SCALES: [f64; 3] = [0.5, 1.5, 4.0];
    /// A baseline shape that is deliberately not the cone's minimiser.
    const SHAPE: [f64; 3] = [0.9, 0.05, 0.3];

    fn times(row: usize) -> (f64, f64) {
        let entry = 0.2 + 0.05 * (row % 12) as f64;
        (entry, entry + 0.4 + 0.07 * ((row * 5) % 11) as f64)
    }

    fn column(t: f64, k: usize) -> f64 {
        1.0 - (-t / SCALES[k]).exp()
    }

    fn derivative(t: f64, k: usize) -> f64 {
        (-t / SCALES[k]).exp() / SCALES[k]
    }

    /// Scores spread over the law's support, off its nodes (`−2.5 + 0.15·k`).
    fn score(row: usize) -> f64 {
        -1.587 + 3.2 * ((row * 7 + 3) % ROWS) as f64 / ROWS as f64
    }

    /// Events sit high on the score, so the upper hinge keeps every one at risk.
    fn event(row: usize) -> f64 {
        if score(row) > 0.4 && row % 2 == 0 { 1.0 } else { 0.0 }
    }

    fn weight(row: usize) -> f64 {
        0.6 + 0.1 * (row % 5) as f64
    }

    fn covariate(row: usize) -> f64 {
        -0.8 + 1.6 * ((row * 11 + 5) % ROWS) as f64 / ROWS as f64
    }

    /// The skewed two-component law of `test_support`, which no Gaussian describes.
    fn skewed_law() -> Arc<SurvivalLatentLaw> {
        let grid = crate::test_support::skewed_grid();
        let law = crate::bms::EmpiricalZGrid::new(
            grid.nodes.clone(),
            grid.weights.clone(),
            "gam#3003 frozen-time law",
        )
        .expect("a valid law");
        Arc::new(
            SurvivalLatentLaw::from_kind(
                &crate::bms::LatentMeasureKind::GlobalEmpirical { grid: law },
                ROWS,
            )
            .expect("materialise the law")
            .expect("an empirical law is a law"),
        )
    }

    fn time_design(value: impl Fn(usize, usize) -> f64) -> DesignMatrix {
        DesignMatrix::from(Array2::from_shape_fn((ROWS, 3), |(row, k)| value(row, k)))
    }

    fn family(law: Option<Arc<SurvivalLatentLaw>>) -> SurvivalMarginalSlopeFamily {
        let marginal = Array2::from_shape_fn((ROWS, 2), |(row, col)| {
            if col == 0 { 1.0 } else { covariate(row) }
        });
        SurvivalMarginalSlopeFamily {
            jeffreys_armed: false,
            latent_law: law,
            n: ROWS,
            event: Arc::new(Array1::from_shape_fn(ROWS, event)),
            weights: Arc::new(Array1::from_shape_fn(ROWS, weight)),
            z: Arc::new(Array1::from_shape_fn(ROWS, score).insert_axis(Axis(1))),
            score_covariance: ScoreCovarianceField::pooled(
                MarginalSlopeCovariance::diagonal(ndarray::array![1.0]).expect("a unit covariance"),
            ),
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: f64::MIN_POSITIVE,
            design_entry: time_design(|row, k| column(times(row).0, k)),
            design_exit: time_design(|row, k| column(times(row).1, k)),
            design_derivative_exit: time_design(|row, k| derivative(times(row).1, k)),
            offset_entry: Arc::new(Array1::zeros(ROWS)),
            offset_exit: Arc::new(Array1::zeros(ROWS)),
            derivative_offset_exit: Arc::new(Array1::zeros(ROWS)),
            entry_at_origin: Arc::new(Array1::from_elem(ROWS, false)),
            marginal_design: DesignMatrix::from(marginal),
            slope_layout: (DesignMatrix::from(Array2::ones((ROWS, 1)))).into(),
            score_warp: None,
            link_dev: None,
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
        }
    }

    /// The coefficient states at `(β_time, β_marginal, β_slope)`.
    fn states(
        family: &SurvivalMarginalSlopeFamily,
        time: Array1<f64>,
        marginal: Array1<f64>,
        slope: f64,
    ) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                eta: family.design_exit.matrixvectormultiply(&time),
                beta: time,
            },
            ParameterBlockState {
                eta: family.marginal_design.matrixvectormultiply(&marginal),
                beta: marginal,
            },
            ParameterBlockState {
                eta: Array1::from_elem(ROWS, slope),
                beta: ndarray::array![slope],
            },
        ]
    }

    /// `Σ_i w_i [r_i·γ̄ᵀ(I(t₁) − I(t₀)) − d_i·log(r_i·γ̄ᵀX_D(t₁))]` at the fixed
    /// shape `SHAPE`: the limit the objective must reach along a path that
    /// keeps that shape.
    fn limit_at_shape(risk: &Array1<f64>) -> f64 {
        (0..ROWS)
            .map(|row| {
                let (t0, t1) = times(row);
                let exposure: f64 =
                    (0..3).map(|k| SHAPE[k] * (column(t1, k) - column(t0, k))).sum();
                let hazard: f64 = (0..3).map(|k| SHAPE[k] * derivative(t1, k)).sum();
                let r = risk[row];
                let mut term = r * exposure;
                if event(row) > 0.0 {
                    term -= (r * hazard).ln();
                }
                weight(row) * term
            })
            .sum()
    }

    /// The negative log-likelihood along `path(s)` converges to `limit` at the
    /// derived rate `s^−order`.
    ///
    /// On the Gaussian closed form and on the marginal level the first
    /// corrections are the Mills term `1/η²` and `q̇/q`, both `O(s⁻²)`, so the
    /// gap shrinks by more than five per decade over `s = 10²…10⁴` and is
    /// `≤ 10⁻³` relative at `s = 10⁴`. On a finite law the index is
    /// `η = s·(z − u_k) − ζ(q)`, so every tail row's hazard carries the relative
    /// correction `−ζ/(s·(z − u_k))`: the gap is `O(1/s)`, large for rows near
    /// their hinge. There the gap must shrink by at least five per decade over
    /// `s = 10³…10⁵`, and the first-order Richardson extrapolant
    /// `N(10⁵) + (N(10⁵) − N(10⁴))/9` must remove nine tenths of it: that is what
    /// separates convergence to `limit` from convergence to a nearby value.
    fn assert_path_converges(
        family: &SurvivalMarginalSlopeFamily,
        path: impl Fn(f64) -> Vec<ParameterBlockState>,
        limit: f64,
        order: i32,
        label: &str,
    ) {
        let scales = if order == 1 { [1e3, 1e4, 1e5] } else { [1e2, 1e3, 1e4] };
        let nll: Vec<f64> = scales
            .iter()
            .map(|&s| {
                -family
                    .log_likelihood_only(&path(s))
                    .expect("the family evaluates along the path")
            })
            .collect();
        let gaps: Vec<f64> = nll.iter().map(|&value| (value - limit).abs()).collect();
        assert!(
            gaps[1] <= gaps[0] / 5.0 && gaps[2] <= gaps[1] / 5.0,
            "{label}: the objective does not converge to the limit {limit}: gaps {gaps:?} at {scales:?}"
        );
        if order == 1 {
            let extrapolated = nll[2] + (nll[2] - nll[1]) / 9.0;
            assert!(
                (extrapolated - limit).abs() <= 0.1 * gaps[2],
                "{label}: the extrapolant {extrapolated} does not reach the limit {limit}: \
                 gaps {gaps:?} at {scales:?}"
            );
        } else {
            assert!(
                gaps[2] <= 1e-3 * (1.0 + limit.abs()),
                "{label}: the gap at s = 1e4 is {:.3e} against the limit {limit}",
                gaps[2]
            );
        }
    }

    fn shape_over(scale: f64) -> Array1<f64> {
        Array1::from_iter(SHAPE.iter().map(|&g| g / scale))
    }

    #[test]
    fn marginal_level_limit_is_the_objectives_own_limit_3003() {
        let family = family(Some(skewed_law()));
        let limit = limit_at_shape(&Array1::ones(ROWS));
        assert_path_converges(
            &family,
            |s| states(&family, shape_over(s), ndarray::array![s, 0.0], 0.0),
            limit,
            2,
            "marginal level, finite law",
        );
    }

    #[test]
    fn slope_hinge_limit_on_a_finite_law_is_the_objectives_own_limit_3003() {
        let law = skewed_law();
        let family = family(Some(Arc::clone(&law)));
        let marginal = ndarray::array![0.3, -0.2];
        let index = family.marginal_design.matrixvectormultiply(&marginal);
        let risk = Array1::from_shape_fn(ROWS, |row| {
            grid_hinge_risk(index[row], score(row), SlopeHingeSide::Upper, law.row(row))
                .expect("the hinge quantile")
                .expect("Φ(−q) is off the law's cumulative weights")
        });
        let limit = limit_at_shape(&risk);
        // On a finite law ∂α/∂q stays O(1), so q̇ shrinks like 1/b.
        assert_path_converges(
            &family,
            |s| states(&family, shape_over(s), marginal.clone(), s),
            limit,
            1,
            "slope hinge, finite law",
        );
    }

    #[test]
    fn slope_hinge_limit_on_the_gaussian_closed_form_is_the_objectives_own_limit_3003() {
        let family = family(None);
        let marginal = ndarray::array![0.3, -0.2];
        let index = family.marginal_design.matrixvectormultiply(&marginal);
        let limit = limit_at_shape(&Array1::from_shape_fn(ROWS, |row| (index[row] + score(row)).max(0.0)));
        // On the closed form ∂α/∂q = c ≈ b, so q̇ shrinks like 1/b².
        assert_path_converges(
            &family,
            |s| states(&family, shape_over(s * s), marginal.clone(), s),
            limit,
            2,
            "slope hinge, Gaussian closed form",
        );
    }

    #[test]
    fn cone_minimum_matches_an_independent_em_iteration_3003() {
        let family = family(None);
        let geometry = FrozenTimeGeometry::new(&family).expect("geometry");
        let exposure = geometry.exposure.t().dot(&geometry.delayed_weight);
        let minimum = cone_minimiser(&exposure, &geometry.event_design, &geometry.event_weight)
            .expect("the shape Newton step")
            .expect("a finite minimum");
        // EM for a Poisson mixture on the cone: monotone, and every iterate is
        // attained, so it bounds the minimum from above.
        let m = &geometry.event_design;
        let v = &geometry.event_weight;
        let mut gamma = Array1::from_elem(3, v.sum() / exposure.sum());
        let mut remaining = 200_000;
        while remaining > 0 {
            let ratio = v / &m.dot(&gamma);
            gamma = &gamma * &m.t().dot(&ratio) / &exposure;
            remaining -= 1;
        }
        let em_value = exposure.dot(&gamma)
            - v.iter().zip(m.dot(&gamma).iter()).map(|(&w, &h)| w * h.ln()).sum::<f64>();
        let scale = 1.0 + em_value.abs();
        assert!(
            minimum.value <= em_value + 1e-12 * scale,
            "Newton {} is above EM {em_value}",
            minimum.value
        );
        assert!(
            em_value - minimum.value <= 1e-9 * scale,
            "Newton {} and EM {em_value} disagree",
            minimum.value
        );
    }

    fn block(name: &str, design: DesignMatrix, penalties: Vec<PenaltyMatrix>, log_lambdas: Vec<f64>) -> ParameterBlockSpec {
        let cols = design.ncols();
        ParameterBlockSpec {
            name: name.to_string(),
            design,
            offset: Array1::zeros(ROWS),
            penalties,
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::from_vec(log_lambdas),
            initial_beta: Some(Array1::zeros(cols)),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    fn blocks(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockSpec> {
        vec![
            block(
                "time",
                family.design_exit.clone(),
                vec![PenaltyMatrix::Dense(Array2::eye(3))],
                vec![0.0],
            ),
            block(
                "marginal",
                family.marginal_design.clone(),
                vec![PenaltyMatrix::Diagonal(ndarray::array![0.0, 1.0])],
                vec![0.2],
            ),
            block(
                "slope",
                DesignMatrix::Dense(DenseDesignMatrix::from(Array2::ones((ROWS, 1)))),
                Vec::new(),
                Vec::new(),
            ),
        ]
    }

    /// Each block's `S_λ` for [`blocks`] at its smoothing parameter.
    fn s_lambdas() -> Vec<Array2<f64>> {
        vec![
            Array2::eye(3),
            Array2::from_diag(&ndarray::array![0.0, 0.2_f64.exp()]),
            Array2::zeros((1, 1)),
        ]
    }

    /// `½·Σ_b λ_b·β_bᵀS_bβ_b` for [`blocks`], written out.
    fn penalty(states: &[ParameterBlockState]) -> f64 {
        0.5 * states[0].beta.dot(&states[0].beta)
            + 0.5 * 0.2_f64.exp() * states[1].beta[1] * states[1].beta[1]
    }

    #[test]
    fn a_point_on_the_frozen_path_is_refused_by_name_3003() {
        let family = family(Some(skewed_law()));
        let geometry = FrozenTimeGeometry::new(&family).expect("geometry");
        let specs = blocks(&family);
        let at = states(&family, shape_over(1e3), ndarray::array![0.3, -0.2], 1e3);
        let log_likelihood = family.log_likelihood_only(&at).expect("evaluate");
        let verdict = frozen_time_identification(
            &family,
            &geometry,
            &specs,
            &at,
            &s_lambdas(),
            log_likelihood,
            penalty(&at),
        )
        .expect("the certificate evaluates");
        let FrozenTimeIdentification::NotIdentified { fit_objective, limit } = verdict else {
            panic!("a point on the frozen path must be refused, got {verdict:?}");
        };
        assert!(limit.objective < fit_objective);
        let reason = FrozenTimeIdentification::NotIdentified { fit_objective, limit }
            .refusal_reason()
            .expect("a refusal");
        assert!(reason.contains("frozen-time limit"), "{reason}");
    }

    #[test]
    fn the_solver_hook_refuses_a_mode_on_the_frozen_path_3003() {
        let family = family(Some(skewed_law()));
        let specs = blocks(&family);
        let at = states(&family, shape_over(1e3), ndarray::array![0.3, -0.2], 1e3);
        let log_likelihood = family.log_likelihood_only(&at).expect("evaluate");
        let reason = family
            .coefficient_mode_refusal(&specs, &at, log_likelihood, penalty(&at), &s_lambdas())
            .expect("the hook evaluates")
            .expect("the joint criterion must refuse this trial point");
        assert!(reason.contains("frozen-time limit"), "{reason}");
    }

    #[test]
    fn a_point_below_every_limit_is_not_refused_3003() {
        let family = family(Some(skewed_law()));
        let geometry = FrozenTimeGeometry::new(&family).expect("geometry");
        let specs = blocks(&family);
        let at = states(&family, shape_over(1e3), ndarray::array![0.3, -0.2], 1e3);
        // A log-likelihood far above the frozen boundary's: nothing may refuse it.
        let verdict = frozen_time_identification(
            &family,
            &geometry,
            &specs,
            &at,
            &s_lambdas(),
            1e6,
            penalty(&at),
        )
            .expect("the certificate evaluates");
        assert!(
            matches!(
                verdict,
                FrozenTimeIdentification::Undetermined(FrozenTimeUndetermined::NoLimitBelowFit { .. })
            ),
            "got {verdict:?}"
        );
    }

    #[test]
    fn the_certificate_names_a_configuration_it_is_not_derived_for_3003() {
        let mut family = family(Some(skewed_law()));
        family.jeffreys_armed = true;
        let geometry = FrozenTimeGeometry::new(&family).expect("geometry");
        let specs = blocks(&family);
        let at = states(&family, shape_over(1e3), ndarray::array![0.3, -0.2], 1e3);
        let verdict = frozen_time_identification(
            &family,
            &geometry,
            &specs,
            &at,
            &s_lambdas(),
            0.0,
            penalty(&at),
        )
            .expect("the certificate evaluates");
        assert_eq!(
            verdict,
            FrozenTimeIdentification::Undetermined(FrozenTimeUndetermined::NotDerived {
                configuration: "an armed Jeffreys prior"
            })
        );
    }
}
