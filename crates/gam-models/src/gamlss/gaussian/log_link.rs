// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub struct PoissonLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
}

impl PoissonLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

/// Certified per-row IRLS contribution for a single-parameter log-link family.
/// Every field is the exact `f64` evaluation of the declared likelihood at the
/// supplied predictor; there is no alternate clamped objective or weight floor.
pub(crate) struct DiagonalIrlsRow {
    /// Weighted contribution to ℓ at this row.
    pub(crate) log_lik_increment: f64,
    /// Exact observed Hessian weight.
    pub(crate) observed_weight: f64,
    /// Exact representable working response.
    pub(crate) working_response: f64,
}

/// Trait implemented by single-block log-link families that share the
/// diagonal IRLS structure (Poisson, Gamma). Each impl is responsible only
/// for the family-specific math: validating `y[i]` and producing the
/// per-row triple `(ℓ_increment, observed_weight, working_step)`.
trait LogLinkDiagonalIrlsFamily {

    /// Read access to the shared (y, prior weights) buffers.
    fn y(&self) -> &Array1<f64>;

    /// Family-specific row math.  A positive-weight row must either return a
    /// fully representable likelihood/score/curvature triple or refuse it.
    fn row_kernel(
        &self,
        row: usize,
        yi: f64,
        eta: f64,
        prior_w: f64,
    ) -> Result<DiagonalIrlsRow, String>;
}

impl LogLinkDiagonalIrlsFamily for PoissonLogFamily {
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    #[inline]
    fn row_kernel(
        &self,
        row: usize,
        yi: f64,
        eta: f64,
        prior_w: f64,
    ) -> Result<DiagonalIrlsRow, String> {
        if prior_w == 0.0 {
            return Ok(DiagonalIrlsRow {
                log_lik_increment: 0.0,
                observed_weight: 0.0,
                working_response: eta,
            });
        }
        let m = eta.exp();
        if !m.is_finite() || m <= 0.0 {
            return Err(row_geometry_error(row, "Poisson mean exp(eta)", eta, m));
        }
        let observed_weight = scaled_positive_product_quotient(prior_w, 1.0, m, 1.0);
        if !observed_weight.is_finite() || observed_weight <= 0.0 {
            return Err(row_geometry_error(
                row,
                "Poisson observed information",
                eta,
                observed_weight,
            ));
        }
        // Drop log(y!) constant. Form the weighted `y*eta` term with its
        // exponent carried separately: `y*eta` may overflow even though the
        // final prior-weighted contribution is representable.
        let weighted_y_eta = if yi == 0.0 || eta == 0.0 {
            0.0
        } else {
            scaled_positive_product_quotient(prior_w, yi, eta.abs(), 1.0).copysign(eta)
        };
        let log_lik_increment = weighted_y_eta - observed_weight;
        if !log_lik_increment.is_finite() {
            return Err(row_geometry_error(
                row,
                "Poisson log-likelihood contribution",
                eta,
                log_lik_increment,
            ));
        }
        let working_response = eta + (yi / m - 1.0);
        if !working_response.is_finite() {
            return Err(row_geometry_error(
                row,
                "Poisson working response",
                eta,
                working_response,
            ));
        }
        Ok(DiagonalIrlsRow {
            log_lik_increment,
            observed_weight,
            working_response,
        })
    }
}

impl CustomFamily for PoissonLogFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        // Assemble the exact score from the IRLS working set (X_bᵀ(w⊙(z−η))),
        // the same source of truth the inner joint-Newton RHS uses. For the
        // canonical Poisson log link observed = Fisher, so this is the exact
        // observed gradient (matches FD of the log-likelihood).
        let eval = self.evaluate(block_states)?;
        gamlss_joint_gradient_from_working_sets(&eval, specs, block_states).map(Some)
    }
}

impl CustomFamilyGenerative for PoissonLogFamily {
}

/// Built-in Gamma log-link family (single parameter block, fixed shape).
#[derive(Clone)]
pub struct GammaLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub shape: f64,
}

impl GammaLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl LogLinkDiagonalIrlsFamily for GammaLogFamily {
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    #[inline]
    fn row_kernel(
        &self,
        row: usize,
        yi: f64,
        eta: f64,
        prior_w: f64,
    ) -> Result<DiagonalIrlsRow, String> {
        if prior_w == 0.0 {
            return Ok(DiagonalIrlsRow {
                log_lik_increment: 0.0,
                observed_weight: 0.0,
                working_response: eta,
            });
        }
        let m = eta.exp();
        if !m.is_finite() || m <= 0.0 {
            return Err(row_geometry_error(row, "Gamma mean exp(eta)", eta, m));
        }
        // Gamma(shape=k, scale=mu/k), dropping eta-independent constants.
        // Form the two terms independently with exponent-balanced algebra:
        // `prior*k*y/m` may be finite even when the intermediate `y/m`
        // overflows, and `(prior*k)*eta` may be representable even when
        // `prior*k` underflows before multiplication by a large |eta|.
        let observed_weight = scaled_positive_product_quotient(prior_w, self.shape, yi, m);
        if !observed_weight.is_finite() || observed_weight <= 0.0 {
            return Err(row_geometry_error(
                row,
                "Gamma observed information",
                eta,
                observed_weight,
            ));
        }
        let eta_term = if eta == 0.0 {
            0.0
        } else {
            scaled_positive_product_quotient(prior_w, self.shape, eta.abs(), 1.0).copysign(eta)
        };
        let log_lik_increment = -observed_weight - eta_term;
        if !log_lik_increment.is_finite() {
            return Err(row_geometry_error(
                row,
                "Gamma log-likelihood contribution",
                eta,
                log_lik_increment,
            ));
        }
        // Gamma with log mean is non-canonical. Use the exact observed
        // η-space curvature -d²ℓ/dη² = prior_w * shape * y / μ, not the
        // Fisher weight prior_w * shape, so diagonal REML/LAML Hessians
        // use the true Laplace curvature instead of a PQL/Fisher surrogate.
        // score / information = (y/μ - 1)/(y/μ); keep the cancellation
        // analytic. In the y >= μ branch the equivalent `1 - μ/y` never
        // materializes an overflowing y/μ ratio.
        let working_step = if yi >= m {
            1.0 - m / yi
        } else {
            let ratio = yi / m;
            (ratio - 1.0) / ratio
        };
        let working_response = eta + working_step;
        if !working_response.is_finite() {
            return Err(row_geometry_error(
                row,
                "Gamma working response",
                eta,
                working_response,
            ));
        }
        Ok(DiagonalIrlsRow {
            log_lik_increment,
            observed_weight,
            working_response,
        })
    }
}

impl CustomFamily for GammaLogFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        // Assemble the exact score from the IRLS working set (X_bᵀ(w⊙(z−η))).
        // The Gamma log link is non-canonical, but the working step z is defined
        // relative to the row weight so w(z−η) is the exact observed ∂ℓ/∂η
        // regardless — the assembled gradient matches FD of the log-likelihood.
        let eval = self.evaluate(block_states)?;
        gamlss_joint_gradient_from_working_sets(&eval, specs, block_states).map(Some)
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_idx != Self::BLOCK_ETA {
            return Ok(None);
        }
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GammaLogFamily input size mismatch".to_string(),
            }
            .into());
        }
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }

        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(GamlssError::InvalidInput {
                    reason: format!("GammaLogFamily requires positive finite y; found y[{i}]={yi}"),
                }
                .into());
            }
            let e = eta[i];
            if !e.is_finite() || !d_eta[i].is_finite() {
                return Err(GamlssError::NonFinite {
                    reason: format!(
                        "GammaLogFamily directional geometry requires finite eta and direction at row {i}"
                    ),
                }
                .into());
            }
            let prior_w = self.weights[i];
            if !prior_w.is_finite() || prior_w < 0.0 {
                return Err(GamlssError::InvalidInput {
                    reason: format!(
                        "GammaLogFamily requires finite non-negative prior weights; found weight[{i}]={prior_w}"
                    ),
                }
                .into());
            }
            if prior_w == 0.0 {
                values.push(0.0);
                continue;
            }
            let row = self.row_kernel(i, yi, e, prior_w)?;
            let observed_weight = row.observed_weight;
            // d/dη [prior_weight * shape * y / exp(η)] = -W_obs.
            let derivative = -observed_weight * d_eta[i];
            if !derivative.is_finite() {
                return Err(row_geometry_error(
                    i,
                    "Gamma observed-information directional derivative",
                    e,
                    derivative,
                ));
            }
            values.push(derivative);
        }
        Ok(Some(Array1::from_vec(values)))
    }
}

impl CustomFamilyGenerative for GammaLogFamily {
}

/// Exact power-of-two decomposition `x = mantissa * 2^exponent` for a positive
/// finite `f64`, including subnormals. The mantissa lies in `[1, 2)`.
#[inline]
fn positive_frexp(x: f64) -> (f64, i32) {
    assert!(x.is_finite() && x > 0.0);
    let bits = x.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if raw_exp != 0 {
        let mantissa = f64::from_bits((1023_u64 << 52) | fraction);
        (mantissa, raw_exp - 1023)
    } else {
        let leading = 63_i32 - fraction.leading_zeros() as i32;
        let shift = 52_i32 - leading;
        let normalized = fraction << shift;
        let mantissa = f64::from_bits((1023_u64 << 52) | (normalized & ((1_u64 << 52) - 1)));
        (mantissa, -1022 - shift)
    }
}

#[inline]
fn scale_normalized_power_of_two(mut mantissa: f64, mut exponent: i32) -> f64 {
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    if exponent >= -1022 {
        let power = f64::from_bits(((exponent + 1023) as u64) << 52);
        return mantissa * power;
    }
    if exponent < -1075 {
        return 0.0;
    }
    // Scale in units of the least positive subnormal.  Keeping the small
    // power-of-two multiplier normal until the final operation lets IEEE
    // round the final subnormal once instead of underflowing an intermediate.
    let units = mantissa * 2.0_f64.powi(exponent + 1074);
    units * f64::from_bits(1)
}

/// Compute `a*b*c/d` for positive finite inputs while carrying the binary
/// exponent separately. Overflow/underflow therefore occurs only when the
/// final `f64` result itself is unrepresentable.
#[inline]
fn scaled_positive_product_quotient(a: f64, b: f64, c: f64, d: f64) -> f64 {
    assert!(a.is_finite() && a > 0.0);
    assert!(b.is_finite() && b > 0.0);
    assert!(c.is_finite() && c > 0.0);
    assert!(d.is_finite() && d > 0.0);
    let (ma, ea) = positive_frexp(a);
    let (mb, eb) = positive_frexp(b);
    let (mc, ec) = positive_frexp(c);
    let (md, ed) = positive_frexp(d);
    scale_normalized_power_of_two((ma * mb) * (mc / md), ea + eb + ec - ed)
}

#[cfg(test)]
mod exact_domain_tests {
    use super::*;

    #[test]
    fn scaled_product_quotient_avoids_false_intermediate_overflow_and_underflow() {
        let got = scaled_positive_product_quotient(1.0e-300, 1.0, 1.0e308, 1.0);
        assert!((got - 1.0e8).abs() <= 4.0 * f64::EPSILON * 1.0e8);
        let got = scaled_positive_product_quotient(1.0e-300, 1.0e-200, 1.0, 1.0e-300);
        assert!((got - 1.0e-200).abs() <= 4.0 * f64::EPSILON * 1.0e-200);
    }

    #[test]
    fn gamma_row_accepts_overflowing_raw_ratio_when_final_geometry_is_finite() {
        let family = GammaLogFamily {
            y: Array1::from_vec(vec![1.0e308]),
            weights: Array1::from_vec(vec![1.0e-300]),
            shape: 1.0,
        };
        let row = family
            .row_kernel(0, 1.0e308, 0.0, 1.0e-300)
            .expect("final Gamma geometry is representable");
        assert!(row.log_lik_increment.is_finite());
        assert!((row.observed_weight - 1.0e8).abs() <= 4.0 * f64::EPSILON * 1.0e8);
        assert_eq!(row.working_response, 1.0);
    }

    #[test]
    fn poisson_row_scales_y_eta_before_multiplication() {
        let family = PoissonLogFamily {
            y: Array1::from_vec(vec![1.0e308]),
            weights: Array1::from_vec(vec![1.0e-300]),
        };
        let row = family
            .row_kernel(0, 1.0e308, 2.0, 1.0e-300)
            .expect("weighted Poisson objective is representable");
        assert!(row.log_lik_increment.is_finite());
        assert!(row.observed_weight.is_normal());
    }
}
