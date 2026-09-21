//! Reference CPU evaluator (parity gate against the GPU kernel).
//!
//! These functions reproduce, byte-for-byte in f64, the formulas in
//! `src/solver/pirls.rs`'s `update_glmvectors` / `write_poisson_log_working_state`
//! / `write_gamma_log_working_state` / `write_identityworking_state`. Stage 1
//! parity tests compare the GPU buffers to these on the V100, and the host
//! replays them to type a device refusal. Both consumers exist only where the
//! CUDA launcher does. `pirls_row.rs` declares this file as
//! `#[cfg(target_os = "linux")] mod cpu_reference;`, and declaring the same scope
//! in-file makes that a claim the compiler enforces.
#![cfg(target_os = "linux")]

use gam_math::special::{bd0, bernoulli_kl_from_logits, expm1_minus_x, softplus};
use gam_problem::EstimationError;

use super::{CurvatureMode, PirlsRowFamily, RowOutput, status_codes};

/// Per-row inputs in scalar form.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowInput {
    pub eta: f64,
    pub y: f64,
    pub prior_weight: f64,
}

/// Reference CPU evaluator for one row, indexed so a device refusal reports the
/// correct row in its typed error. `mode` selects `w_hessian` curvature, and
/// `gamma_shape` (α > 0) is read only when `family == GammaLog`.
pub(crate) fn row_reweight_cpu_at(
    row: usize,
    family: PirlsRowFamily,
    mode: CurvatureMode,
    input: RowInput,
    gamma_shape: f64,
) -> Result<RowOutput, EstimationError> {
    match family {
        PirlsRowFamily::GaussianIdentity => row_gaussian_identity(row, input, mode),
        PirlsRowFamily::PoissonLog => row_poisson_log(row, input, mode),
        PirlsRowFamily::GammaLog => row_gamma_log(row, input, mode, gamma_shape),
        PirlsRowFamily::BernoulliLogit => row_bernoulli_logit(row, input, mode),
        PirlsRowFamily::BernoulliProbit => row_bernoulli_probit(row, input, mode),
        PirlsRowFamily::BernoulliCLogLog => row_bernoulli_cloglog(row, input, mode),
    }
}

/// Recover the typed error represented by a device status vector.  Device
/// threads write one code per row, so scanning in index order makes concurrent
/// failures deterministic.  The scalar CPU replay supplies the exact
/// quantity/value payload without expanding the hot GPU ABI.
pub(crate) fn replay_first_refusal(
    family: PirlsRowFamily,
    mode: CurvatureMode,
    gamma_shape: f64,
    eta: &[f64],
    y: &[f64],
    prior_weight: &[f64],
    status: &[u32],
) -> Result<(), EstimationError> {
    let n = eta.len();
    if y.len() != n || prior_weight.len() != n || status.len() != n {
        return Err(EstimationError::InvalidInput(format!(
            "GPU PIRLS refusal replay length mismatch: eta={n}, y={}, prior_weight={}, status={}",
            y.len(),
            prior_weight.len(),
            status.len(),
        )));
    }
    let Some((row, &code)) = status
        .iter()
        .enumerate()
        .find(|(_, code)| **code != status_codes::OK)
    else {
        return Ok(());
    };
    let input = RowInput {
        eta: eta[row],
        y: y[row],
        prior_weight: prior_weight[row],
    };
    match row_reweight_cpu_at(row, family, mode, input, gamma_shape) {
        Err(error) => Err(error),
        Ok(_) => Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            status_codes::quantity(code),
            input.eta,
            f64::from(code),
        )),
    }
}

/// Resolve `(w_fisher, observed_correction)` into the `w_hessian` value that
/// matches the selected curvature surface. Stage 1 returns `w_fisher` for both
/// modes (parity with the CPU PIRLS path that, today, uses Fisher weights
/// even for non-canonical links); Stage 5 will switch the `Observed` arm to
/// `w_fisher + observed_correction` and the call sites stay unchanged.
#[inline]
fn select_w_hessian(mode: CurvatureMode, w_fisher: f64, observed_correction: f64) -> f64 {
    match mode {
        CurvatureMode::Fisher => w_fisher,
        CurvatureMode::Observed => w_fisher + observed_correction,
    }
}

#[inline]
fn finite_eta(link: &'static str, eta: f64) -> Result<(), EstimationError> {
    if eta.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InverseLinkDomainViolation {
            link,
            eta,
            lower: -f64::MAX,
            upper: f64::MAX,
        })
    }
}

#[inline]
fn prior_weight(row: usize, input: RowInput) -> Result<f64, EstimationError> {
    if input.prior_weight.is_finite() && input.prior_weight >= 0.0 {
        Ok(input.prior_weight)
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "prior weight",
            input.eta,
            input.prior_weight,
        ))
    }
}

#[inline]
fn certify_output(row: usize, eta: f64, output: RowOutput) -> Result<RowOutput, EstimationError> {
    for (quantity, value) in [
        ("mean", output.mu),
        ("eta gradient", output.grad_eta),
        ("Fisher weight", output.w_fisher),
        ("observed Hessian weight", output.w_hessian),
        ("solver Hessian weight", output.w_solver),
        ("deviance contribution", output.deviance),
    ] {
        if !value.is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(row, quantity, eta, value));
        }
    }
    Ok(output)
}

/// Evaluate `a*b/c` while avoiding a false overflow/underflow caused solely by
/// operation order.  At least one of the product-first or quotient-first forms
/// is normally representable whenever the final positive f64 is; all three are
/// tried in a fixed order so CPU/device refusal and rounding stay deterministic.
#[inline]
fn positive_mul_div(a: f64, b: f64, c: f64) -> f64 {
    let product = a * b;
    if product.is_finite() && product > 0.0 {
        let value = product / c;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    let quotient_a = a / c;
    if quotient_a.is_finite() && quotient_a > 0.0 {
        let value = quotient_a * b;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    let quotient_b = b / c;
    if quotient_b.is_finite() && quotient_b > 0.0 {
        let value = quotient_b * a;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    product / c
}

/// `u - log1p(u)` without cancellation around zero.
#[inline]
fn gamma_unit_deviance_near_one(u: f64) -> f64 {
    if u.abs() > 0.125 {
        return u - u.ln_1p();
    }
    let mut power = u * u;
    let mut sum = 0.5 * power;
    for degree in 3..=32 {
        power *= u;
        let term = power / f64::from(degree);
        let next = if degree % 2 == 0 {
            sum + term
        } else {
            sum - term
        };
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

/// `(1+u)log1p(u)-u` without cancellation around zero.
#[inline]
fn poisson_unit_deviance_near_one(u: f64) -> f64 {
    if u.abs() > 0.125 {
        return (1.0 + u) * u.ln_1p() - u;
    }
    let mut power = u * u;
    let mut sum = 0.5 * power;
    for degree in 3..=32 {
        power *= u;
        let coefficient =
            if degree % 2 == 0 { 1.0 } else { -1.0 } / (f64::from(degree) * f64::from(degree - 1));
        let next = sum + coefficient * power;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

#[inline]
fn row_gaussian_identity(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard identity inverse link", input.eta)?;
    let w = prior_weight(row, input)?;
    let mu = input.eta;
    if w > 0.0 && !input.y.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Gaussian response", input.eta, input.y));
    }
    let resid = input.y - mu;
    let (grad_eta, dev) = if w == 0.0 {
        (0.0, 0.0)
    } else {
        (w * resid, w * resid * resid)
    };
    let w_hessian = select_w_hessian(mode, w, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher: w,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn row_poisson_log(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    let mu = crate::mixture_link::log_link_solver_exp(input.eta)?;
    let w_prior = prior_weight(row, input)?;
    if w_prior > 0.0 && !(input.y.is_finite() && input.y >= 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Poisson response", input.eta, input.y));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * mu;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Poisson Fisher weight", input.eta, w_fisher));
    }
    let grad_eta = w_prior * (input.y - mu);
    let u = (input.y - mu) / mu;
    let dev_base = if input.y == 0.0 {
        w_fisher
    } else {
        // Accurate around saturation. In either far tail this dimensionless
        // ratio can become non-finite before multiplication by a tiny weight;
        // only then switch to the algebraically identical absolute-coordinate
        // expression, whose products are balanced independently.
        let scaled_unit = w_fisher * poisson_unit_deviance_near_one(u);
        if scaled_unit.is_finite() && scaled_unit >= 0.0 {
            scaled_unit
        } else {
            let weighted_y = positive_mul_div(w_fisher, input.y, mu);
            weighted_y * (input.y.ln() - input.eta - 1.0) + w_fisher
        }
    };
    let dev = 2.0 * dev_base;
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn row_gamma_log(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
    shape: f64,
) -> Result<RowOutput, EstimationError> {
    let mu = crate::mixture_link::log_link_solver_exp(input.eta)?;
    if !(shape.is_finite() && shape > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Gamma shape", input.eta, shape));
    }
    let w_prior = prior_weight(row, input)?;
    if w_prior > 0.0 && !(input.y.is_finite() && input.y > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Gamma response", input.eta, input.y));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * shape;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "Gamma Fisher weight", input.eta, w_fisher));
    }
    let observed_ratio = match mode {
        CurvatureMode::Fisher => None,
        CurvatureMode::Observed => {
            // Ratio-first: at `y == mu` the observed weight must reproduce the
            // Fisher weight EXACTLY (`y/mu` is bit-for-bit 1.0), which the
            // product-first reordering destroys by one ulp. The reordering
            // fallback remains for tails where the direct form is not
            // representable.
            let direct = w_fisher * (input.y / mu);
            let weighted_ratio = if direct.is_finite() && direct > 0.0 {
                direct
            } else {
                positive_mul_div(w_fisher, input.y, mu)
            };
            if !(weighted_ratio.is_finite() && weighted_ratio > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "Gamma observed Hessian weight",
                    input.eta,
                    weighted_ratio,
                ));
            }
            Some(weighted_ratio)
        }
    };
    let w_hessian = observed_ratio.unwrap_or(w_fisher);
    if !w_hessian.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Gamma observed Hessian weight",
            input.eta,
            w_hessian,
        ));
    }
    let u = (input.y - mu) / mu;
    // `u` rounds to exactly -1 when y/mu is a representable but very small
    // ratio, and the dimensionless expression can overflow for the opposite
    // tail. Preserve the local-series path whenever it succeeds, then use the
    // weighted absolute-coordinate identity for those two tail cases.
    let scaled_unit = w_fisher * gamma_unit_deviance_near_one(u);
    let need_weighted_ratio = !u.is_finite() || !(scaled_unit.is_finite() && scaled_unit >= 0.0);
    let weighted_ratio = if need_weighted_ratio {
        observed_ratio.unwrap_or_else(|| positive_mul_div(w_fisher, input.y, mu))
    } else {
        0.0
    };
    let grad_eta = if u.is_finite() {
        w_fisher * u
    } else {
        weighted_ratio - w_fisher
    };
    let dev_base = if scaled_unit.is_finite() && scaled_unit >= 0.0 {
        scaled_unit
    } else {
        weighted_ratio - w_fisher * (1.0 + input.y.ln() - input.eta)
    };
    let dev = 2.0 * dev_base;
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn bernoulli_response(row: usize, input: RowInput, w: f64) -> Result<(), EstimationError> {
    if w == 0.0 || (input.y.is_finite() && (0.0..=1.0).contains(&input.y)) {
        Ok(())
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, "binomial response", input.eta, input.y))
    }
}

#[inline]
fn row_bernoulli_logit(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard logit inverse link", input.eta)?;
    let w_prior = prior_weight(row, input)?;
    bernoulli_response(row, input, w_prior)?;
    let tail = (-input.eta.abs()).exp();
    let denom = 1.0 + tail;
    let (mu, residual) = if input.eta >= 0.0 {
        let one_minus_mu = tail / denom;
        let residual = if input.y == 1.0 {
            one_minus_mu
        } else {
            (input.y - 1.0) + one_minus_mu
        };
        (1.0 / denom, residual)
    } else {
        let mu = tail / denom;
        (mu, input.y - mu)
    };
    let dmu_deta = tail / (denom * denom);
    if !(dmu_deta.is_finite() && dmu_deta >= 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "canonical-logit inverse-link jet",
            input.eta,
            dmu_deta,
        ));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    if dmu_deta == 0.0 {
        // Saturated row (`|eta|` past ~745.13, where `mu'` rounds to zero): a
        // consistent row's residual rounds to zero with it and its geometry is
        // the analytic zero-weight limit; an inconsistent one has no representable
        // weight for its unit-order score. Twin of `bernoulli_logit_geometry_from_jet`.
        if residual != 0.0 {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                row,
                "saturated canonical-logit row inconsistent with response",
                input.eta,
                residual,
            ));
        }
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                deviance: bernoulli_logit_deviance(input.y, input.eta, w_prior),
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * dmu_deta;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(row, "logit Fisher weight", input.eta, w_fisher));
    }
    let grad_eta = w_prior * residual;
    let dev = bernoulli_logit_deviance(input.y, input.eta, w_prior);
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

/// Probit's log-probability jet (gam#3329), the twin of the CUDA
/// `bernoulli_probit_body`: `μ = Φ(η)` and `μ̄ = Φ(−η)` each from `erfc` on its
/// own tail, the Mills ratios `a' = φ/Φ(η)` and `b' = −φ/Φ(−η)`, and, from
/// `φ' = −ηφ`, `a'' = −a'(η + a')` and `b'' = −b'(η + b')`.
#[inline]
fn row_bernoulli_probit(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard probit inverse link", input.eta)?;
    let eta = input.eta;
    let mu = standard_normal_cdf(eta);
    let mu_bar = standard_normal_cdf(-eta);
    let density = standard_normal_pdf(eta);
    let a1 = density / mu;
    let b1 = -density / mu_bar;
    row_bernoulli_log_jet(
        row,
        input,
        mode,
        BernoulliLogJet {
            mu,
            mu_bar,
            a1,
            a2: -a1 * (eta + a1),
            b1,
            b2: -b1 * (eta + b1),
        },
    )
}

/// Complementary log-log's log-probability jet (gam#3329), the twin of the
/// CUDA `bernoulli_cloglog_body`. With `t = e^η`, `b = log μ̄ = −t` exactly;
/// `a' = t μ̄ / μ` and `a'' = a'((1 − a') − t)`, whose gap
/// `1 − a' = (1 − (1 + t)e^{−t}) / μ` is summed from `e^{−t}·(e^t − 1 − t)`
/// inside the radius where [`expm1_minus_x`] sums its series and read as
/// `μ − t μ̄` beyond it.
#[inline]
fn row_bernoulli_cloglog(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard complementary-log-log inverse link", input.eta)?;
    let t = input.eta.exp();
    let mu = -(-t).exp_m1();
    let mu_bar = (-t).exp();
    let gap = if t <= 0.5 {
        mu_bar * expm1_minus_x(t)
    } else {
        mu - t * mu_bar
    };
    let a1 = t * mu_bar / mu;
    row_bernoulli_log_jet(
        row,
        input,
        mode,
        BernoulliLogJet {
            mu,
            mu_bar,
            a1,
            a2: a1 * (gap / mu - t),
            b1: -t,
            b2: -t,
        },
    )
}

/// A Bernoulli link's log-probability jet at one η: both probabilities, each
/// formed directly, and the first two η-derivatives of `a = log μ` and
/// `b = log μ̄`.
#[derive(Clone, Copy)]
struct BernoulliLogJet {
    mu: f64,
    mu_bar: f64,
    a1: f64,
    a2: f64,
    b1: f64,
    b2: f64,
}

/// The rows of a non-canonical Bernoulli link from its log-probability jet,
/// the twin of the CUDA `bernoulli_log_jet_rows`: score `y a' + (1 − y) b'`,
/// Fisher weight `μ a'² + μ̄ b'²`, observed weight `−(y a'' + (1 − y) b'')`.
/// No variance `μ(1 − μ)` divides anything, so a row is refused only where
/// one of the two probabilities underflows.
#[inline]
fn row_bernoulli_log_jet(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
    jet: BernoulliLogJet,
) -> Result<RowOutput, EstimationError> {
    let w_prior = prior_weight(row, input)?;
    bernoulli_response(row, input, w_prior)?;
    let BernoulliLogJet {
        mu,
        mu_bar,
        a1,
        a2,
        b1,
        b2,
    } = jet;
    if !(mu > 0.0
        && mu_bar > 0.0
        && a1.is_finite()
        && a2.is_finite()
        && b1.is_finite()
        && b2.is_finite())
    {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "inverse-link jet",
            input.eta,
            mu.min(mu_bar),
        ));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let fisher_per_prior = mu * a1 * a1 + mu_bar * b1 * b1;
    let w_fisher = w_prior * fisher_per_prior;
    if !(fisher_per_prior.is_finite()
        && fisher_per_prior > 0.0
        && w_fisher.is_finite()
        && w_fisher > 0.0)
    {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Bernoulli Fisher weight",
            input.eta,
            w_fisher,
        ));
    }
    let y = input.y;
    let w_hessian = match mode {
        CurvatureMode::Fisher => w_fisher,
        CurvatureMode::Observed => -w_prior * (y * a2 + (1.0 - y) * b2),
    };
    if !w_hessian.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Bernoulli observed Hessian weight",
            input.eta,
            w_hessian,
        ));
    }
    let grad_eta = w_prior * (y * a1 + (1.0 - y) * b1);
    let dev = 2.0 * w_prior * (bd0(y, mu) + bd0(1.0 - y, mu_bar));
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn bernoulli_logit_deviance(y: f64, eta: f64, w: f64) -> f64 {
    let unit = if y == 0.0 {
        softplus(eta)
    } else if y == 1.0 {
        softplus(-eta)
    } else {
        let response_logit = y.ln() - (-y).ln_1p();
        bernoulli_kl_from_logits(response_logit, eta)
    };
    2.0 * w * unit
}

/// Stable Φ(x) using the complementary error function with the same identity
/// `erfc(-x/√2)/2 = Φ(x)` used by libstd. Keeps mass at the tails accurate.
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * gam_gpu::numerics_host::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    const COEFF: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2π)
    COEFF * (-0.5 * x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_canonical_logit_row_matches_host_zero_weight_limit() {
        let fit = |eta: f64, y: f64| {
            row_bernoulli_logit(
                0,
                RowInput {
                    eta,
                    y,
                    prior_weight: 1.0,
                },
                CurvatureMode::Fisher,
            )
        };
        for (eta, y) in [(-746.0, 0.0), (746.0, 1.0)] {
            let out = fit(eta, y).expect("consistent saturated row is representable");
            assert_eq!(out.mu, y);
            assert_eq!(out.grad_eta, 0.0);
            assert_eq!(out.w_fisher, 0.0);
            assert_eq!(out.w_hessian, 0.0);
            assert_eq!(out.w_solver, 0.0);
            assert!(out.deviance.is_finite() && out.deviance >= 0.0);
        }
        assert!(fit(-746.0, 1.0).is_err());
        assert!(fit(746.0, 0.0).is_err());
    }
}
