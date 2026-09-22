//! Solver-side rows of the generic variance × link cells
//! ([`gam_problem::GenericEdmCell`]): the cells of the exponential-dispersion
//! families with no hand-written kernel (identity-Poisson, log-Gaussian,
//! sqrt links, the reciprocal links off their canonical family, log-binomial
//! relative risk, …).
//!
//! Every quantity here is read off [`gam_math::edm_row::EdmRow`], which
//! composes the family's variance-function jet with the link's inverse-link
//! jet once, as a [`gam_math::jet_tower::RowProgram`], so the Fisher working
//! state, the observed information and its η-derivatives are exact.
//!
//! A cell whose mean map leaves the family's mean domain carries the
//! feasibility set [`LikelihoodSpec::eta_feasibility`]. A row outside it
//! reports [`EstimationError::InverseLinkDomainViolation`], which the inner
//! solver classifies as an infeasible trial step and damps (step-halving back
//! into the set) — `η` is never projected and `μ` never floored, so every
//! accepted iterate evaluates the exact likelihood.

use super::*;
use gam_math::edm_row::{EdmRow, EdmVariance};
use gam_problem::EtaFeasibility;
pub(crate) use gam_problem::GenericEdmCell;

/// The variance function of a generic cell's response.
#[inline]
pub(crate) fn generic_edm_variance(cell: GenericEdmCell) -> EdmVariance {
    match cell {
        GenericEdmCell::GaussianLog
        | GenericEdmCell::GaussianSqrt
        | GenericEdmCell::GaussianInverseSquared => EdmVariance::Gaussian,
        GenericEdmCell::PoissonIdentity
        | GenericEdmCell::PoissonSqrt
        | GenericEdmCell::PoissonInverse
        | GenericEdmCell::PoissonInverseSquared => EdmVariance::Poisson,
        GenericEdmCell::GammaIdentity
        | GenericEdmCell::GammaSqrt
        | GenericEdmCell::GammaInverseSquared => EdmVariance::Gamma,
        GenericEdmCell::InverseGaussianIdentity
        | GenericEdmCell::InverseGaussianSqrt
        | GenericEdmCell::InverseGaussianInverse => EdmVariance::InverseGaussian,
        GenericEdmCell::BinomialLog => EdmVariance::Bernoulli,
    }
}

/// The feasibility set of a generic cell, `η ∈ (lower, upper)`.
#[inline]
pub(crate) fn generic_edm_feasibility(cell: GenericEdmCell) -> EtaFeasibility {
    LikelihoodSpec::new(generic_edm_response(cell), InverseLink::Standard(cell.link()))
        .eta_feasibility()
}

#[inline]
fn generic_edm_response(cell: GenericEdmCell) -> ResponseFamily {
    match generic_edm_variance(cell) {
        EdmVariance::Gaussian => ResponseFamily::Gaussian,
        EdmVariance::Poisson => ResponseFamily::Poisson,
        EdmVariance::Gamma => ResponseFamily::Gamma,
        EdmVariance::InverseGaussian => ResponseFamily::InverseGaussian,
        EdmVariance::Bernoulli => ResponseFamily::Binomial,
    }
}

/// Reject `η` outside the cell's feasibility set as a recoverable domain
/// violation.
#[inline]
pub(crate) fn require_generic_edm_feasible(
    cell: GenericEdmCell,
    eta: f64,
) -> Result<(), EstimationError> {
    let feasibility = generic_edm_feasibility(cell);
    if feasibility.admits(eta) {
        return Ok(());
    }
    let (lower, upper) = feasibility.interval();
    Err(EstimationError::InverseLinkDomainViolation {
        link: cell.link().name(),
        eta,
        lower: lower.max(-f64::MAX),
        upper: upper.min(f64::MAX),
    })
}

/// The inverse-link stack `[h, …, h⁽⁵⁾]` and the complement `1 − μ` of a
/// generic cell at a feasible `η`.
#[inline]
pub(crate) fn generic_edm_link_jet(
    cell: GenericEdmCell,
    row: usize,
    eta: f64,
) -> Result<([f64; 6], f64), EstimationError> {
    require_generic_edm_feasible(cell, eta)?;
    let link = crate::mixture_link::standard_ladder_link_jet6(cell.link(), eta)
        .expect("every generic cell's link is on the power/log ladder")?;
    // The one saturating mean here is the relative-risk `μ = e^η < 1`, whose
    // complement is formed without cancellation.
    let one_minus_mu = match cell {
        GenericEdmCell::BinomialLog => -eta.exp_m1(),
        _ => 1.0 - link[0],
    };
    let variance = generic_edm_variance(cell);
    if !variance.mean_in_domain(link[0], one_minus_mu) || !link.iter().all(|v| v.is_finite()) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "generic-link mean",
            eta,
            link[0],
        ));
    }
    Ok((link, one_minus_mu))
}

/// The response support of a generic cell: finite for the Gaussian, an exact
/// count for the Poisson, strictly positive for the Gamma and inverse
/// Gaussian, a proportion in `[0, 1]` for the binomial.
#[inline]
pub(crate) fn certify_generic_edm_response(
    cell: GenericEdmCell,
    row: usize,
    eta: f64,
    y: f64,
) -> Result<(), EstimationError> {
    let (in_support, quantity) = match generic_edm_variance(cell) {
        EdmVariance::Gaussian => (y.is_finite(), "Gaussian response"),
        EdmVariance::Poisson => (gam_spec::is_poisson_response(y), "Poisson response"),
        EdmVariance::Gamma => (y.is_finite() && y > 0.0, "Gamma response"),
        EdmVariance::InverseGaussian => (y.is_finite() && y > 0.0, "inverse-Gaussian response"),
        EdmVariance::Bernoulli => (y.is_finite() && (0.0..=1.0).contains(&y), "binomial response"),
    };
    if in_support {
        Ok(())
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, quantity, eta, y))
    }
}

/// One certified generic row with likelihood scale `s = w/φ`.
#[inline]
pub(crate) fn generic_edm_row(
    cell: GenericEdmCell,
    row: usize,
    y: f64,
    eta: f64,
    scale: f64,
) -> Result<EdmRow, EstimationError> {
    let (link, one_minus_mu) = generic_edm_link_jet(cell, row, eta)?;
    Ok(EdmRow {
        variance: generic_edm_variance(cell),
        y,
        scale,
        eta,
        link,
        one_minus_mu,
    })
}

#[inline]
fn certify_prior_weight(row: usize, eta: f64, prior_weight: f64) -> Result<(), EstimationError> {
    if prior_weight.is_finite() && prior_weight >= 0.0 {
        Ok(())
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, "prior weight", eta, prior_weight))
    }
}

#[inline]
fn certify_generic_dispersion(phi: f64) -> Result<(), EstimationError> {
    if phi.is_finite() && phi > 0.0 {
        Ok(())
    } else {
        crate::bail_invalid_estim!("dispersion phi must be finite and > 0; got {phi}")
    }
}

#[inline]
fn finite_generic_value(row: usize, quantity: &'static str, eta: f64, value: f64) -> Result<f64, EstimationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, quantity, eta, value))
    }
}

#[derive(Clone, Copy, Default)]
struct GenericFisherRow {
    mu: f64,
    weight: f64,
    z: f64,
    c: f64,
    d: f64,
    h1: f64,
    h2: f64,
    h3: f64,
}

fn generic_fisher_row(
    cell: GenericEdmCell,
    phi: f64,
    row: usize,
    y: Option<f64>,
    eta: f64,
    prior_weight: f64,
) -> Result<GenericFisherRow, EstimationError> {
    certify_prior_weight(row, eta, prior_weight)?;
    let (link, one_minus_mu) = generic_edm_link_jet(cell, row, eta)?;
    let [mu, h1, h2, h3, ..] = link;
    if prior_weight == 0.0 {
        return Ok(GenericFisherRow { mu, weight: 0.0, z: eta, c: 0.0, d: 0.0, h1, h2, h3 });
    }
    if h1 == 0.0 {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "dmu/deta", eta, h1));
    }
    let edm = EdmRow {
        variance: generic_edm_variance(cell),
        y: y.unwrap_or(mu),
        scale: prior_weight / phi,
        eta,
        link,
        one_minus_mu,
    };
    let fisher = edm.fisher_tower().map_err(|message| {
        EstimationError::InvalidInput(format!("{} Fisher tower at row {row}: {message}", cell.name()))
    })?;
    let weight = finite_generic_value(row, "Fisher weight", eta, fisher.w)?;
    let c = finite_generic_value(row, "dW/deta", eta, fisher.c)?;
    let d = finite_generic_value(row, "d2W/deta2", eta, fisher.d)?;
    let z = match y {
        Some(y) => {
            certify_generic_edm_response(cell, row, eta, y)?;
            finite_generic_value(row, "working response", eta, eta + edm.residual() / h1)?
        }
        None => eta,
    };
    Ok(GenericFisherRow { mu, weight, z, c, d, h1, h2, h3 })
}

/// Fisher working state of a generic cell: exact `μ`, Fisher weight
/// `W = w h′²/(φ V(μ))`, working response `z = η + (y − μ)/h′`, and the
/// optional curvature/link-jet carriers. Every row is certified before any
/// output buffer is written.
pub(crate) fn write_generic_edm_working_state(
    cell: GenericEdmCell,
    phi: f64,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    certify_generic_dispersion(phi)?;
    let rows: Vec<GenericFisherRow> = super::par_certified_rows(eta.len(), |i| {
        generic_fisher_row(cell, phi, i, Some(y[i]), eta[i], priorweights[i])
    })?;
    for (i, row) in rows.iter().enumerate() {
        mu[i] = row.mu;
        weights[i] = row.weight;
        z[i] = row.z;
    }
    if let Some(derivs) = derivatives {
        scatter_generic_curvature(&rows, derivs);
    }
    Ok(())
}

/// The curvature carriers of [`write_generic_edm_working_state`] alone, for
/// outer-derivative reconstruction.
pub(crate) fn write_generic_edm_eta_curvature(
    cell: GenericEdmCell,
    phi: f64,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    buffers: WorkingDerivativeBuffersMut<'_>,
) -> Result<(), EstimationError> {
    certify_generic_dispersion(phi)?;
    let rows: Vec<GenericFisherRow> = super::par_certified_rows(eta.len(), |i| {
        generic_fisher_row(cell, phi, i, None, eta[i], priorweights[i])
    })?;
    scatter_generic_curvature(&rows, buffers);
    Ok(())
}

fn scatter_generic_curvature(rows: &[GenericFisherRow], buffers: WorkingDerivativeBuffersMut<'_>) {
    for (i, row) in rows.iter().enumerate() {
        buffers.c[i] = row.c;
        buffers.d[i] = row.d;
        buffers.dmu_deta[i] = row.h1;
        buffers.d2mu_deta2[i] = row.h2;
        buffers.d3mu_deta3[i] = row.h3;
    }
}

/// The observed information `W_obs = −ℓ″` of one generic row and its first
/// two η-derivatives, from one dense evaluation of the score program.
pub(crate) fn generic_edm_observed_weight_jet(
    cell: GenericEdmCell,
    phi: f64,
    row: usize,
    y: f64,
    eta: f64,
    prior_weight: f64,
) -> Result<(f64, f64, f64), EstimationError> {
    certify_prior_weight(row, eta, prior_weight)?;
    if prior_weight == 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    certify_generic_edm_response(cell, row, eta, y)?;
    let edm = generic_edm_row(cell, row, y, eta, prior_weight / phi)?;
    let tower = edm.observed_tower().map_err(|message| {
        EstimationError::InvalidInput(format!("{} observed tower at row {row}: {message}", cell.name()))
    })?;
    Ok((
        finite_generic_value(row, "observed Hessian weight", eta, tower.w)?,
        finite_generic_value(row, "observed Hessian dW/deta", eta, tower.c)?,
        finite_generic_value(row, "observed Hessian d2W/deta2", eta, tower.d)?,
    ))
}

/// Half unit deviance `½ w d(y, μ)` and its η-score `w (μ − y) h′/V(μ)` of one
/// generic row on a measure weight `w`.
pub(crate) fn generic_edm_deviance_row(
    cell: GenericEdmCell,
    row: usize,
    y: f64,
    eta: f64,
    weight: f64,
) -> Result<(f64, f64), EstimationError> {
    certify_generic_edm_response(cell, row, eta, y)?;
    let edm = generic_edm_row(cell, row, y, eta, weight)?;
    let [mu, h1, ..] = edm.link;
    let variance = generic_edm_variance(cell).jet(mu, edm.one_minus_mu)[0];
    let half = finite_generic_value(row, "generic-link half-deviance", eta, 0.5 * weight * edm.unit_deviance())?;
    let score = finite_generic_value(row, "generic-link eta score", eta, -weight * edm.residual() * (h1 / variance))?;
    Ok((half, score))
}

/// The mean `μ = h(η)` of a generic row at a feasible `η`.
#[inline]
pub(crate) fn generic_edm_mean(cell: GenericEdmCell, row: usize, eta: f64) -> Result<f64, EstimationError> {
    generic_edm_link_jet(cell, row, eta).map(|(link, _)| link[0])
}
