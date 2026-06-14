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

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

/// Per-row IRLS contribution that a single-parameter log-link family must
/// produce. The shared driver `evaluate_log_link_diagonal_irls` consumes
/// these and assembles the full `FamilyEvaluation` so the three pieces of
/// code that previously lived inside each family — size validation, per-row
/// y validation + η clamping + saturated `exp`, the active-clamp w/z guard,
/// and the final return — exist in exactly one place.
pub(crate) struct DiagonalIrlsRow {
    /// Weighted contribution to ℓ at this row.
    log_lik_increment: f64,
    /// Unfloored observed Hessian weight (the driver applies `MIN_WEIGHT`).
    observed_weight: f64,
    /// Per-row Newton step on the working response: `z = e + working_step`.
    /// Each family computes this with its own (score, denominator); the
    /// driver only handles the active-clamp / zero-weight guard.
    working_step: f64,
}

/// Trait implemented by single-block log-link families that share the
/// diagonal IRLS structure (Poisson, Gamma). Each impl is responsible only
/// for the family-specific math: validating `y[i]` and producing the
/// per-row triple `(ℓ_increment, observed_weight, working_step)`.
trait LogLinkDiagonalIrlsFamily {
    /// Short, human-readable name used in size-mismatch errors.
    fn family_label(&self) -> &'static str;

    /// Read access to the shared (y, prior weights) buffers.
    fn y(&self) -> &Array1<f64>;
    fn prior_weights(&self) -> &Array1<f64>;

    /// Optional pre-loop validation hook for parameters outside the
    /// (y, weights, eta) triple (e.g. Gamma shape > 0).
    fn validate_self(&self) -> Result<(), String> {
        Ok(())
    }

    /// Validate `y[i]` and return an error message if rejected. Default
    /// implementation enforces only finiteness; concrete families override
    /// to add domain constraints.
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String>;

    /// Family-specific per-row math; `m = saturated_exp_eta(eta_clamped)`
    /// is computed by the driver and handed in.
    fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow;
}

/// Shared IRLS driver for [`LogLinkDiagonalIrlsFamily`]. Centralises the
/// size-check, η-clamp, saturated-exp, active-clamp guard, ll accumulation,
/// and `FamilyEvaluation` assembly so all log-link families with the diagonal
/// structure (Poisson, Gamma) cannot drift apart numerically.
pub(crate) fn evaluate_log_link_diagonal_irls<F: LogLinkDiagonalIrlsFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<FamilyEvaluation, String> {
    let label = family.family_label();
    let eta = &expect_single_block(block_states, label)?.eta;
    let y = family.y();
    let prior_weights = family.prior_weights();
    let n = y.len();
    if eta.len() != n || prior_weights.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{label} input size mismatch"),
        }
        .into());
    }
    family.validate_self()?;

    let mut ll = 0.0;
    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);

    for i in 0..n {
        let yi = y[i];
        family.validate_yi(yi, i)?;
        let e_raw = eta[i];
        let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
        let active_clamp = e != e_raw;
        let m = saturated_exp_eta(e_raw);
        let prior_w = prior_weights[i];
        let row = family.row_kernel(yi, e, m, prior_w);
        ll += row.log_lik_increment;
        if prior_w == 0.0 || active_clamp {
            w[i] = 0.0;
            z[i] = e_raw;
        } else {
            w[i] = floor_positiveweight(row.observed_weight, MIN_WEIGHT);
            z[i] = e + row.working_step;
        }
    }

    Ok(FamilyEvaluation {
        log_likelihood: ll,
        blockworking_sets: vec![BlockWorkingSet::diagonal_checked(z, w)?],
    })
}

impl LogLinkDiagonalIrlsFamily for PoissonLogFamily {
    fn family_label(&self) -> &'static str {
        "PoissonLogFamily"
    }
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi < 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "PoissonLogFamily requires non-negative finite y; found y[{idx}]={yi}"
                ),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow {
        // Drop log(y!) constant in objective.
        let log_lik_increment = prior_w * (yi * e_clamped - m);
        let dmu = m.max(MIN_DERIV);
        let var = m.max(MIN_PROB);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight: prior_w * (dmu * dmu / var),
            // (yi - m)/dmu, identical to the previous direct expression.
            working_step: (yi - m) / signedwith_floor(dmu, MIN_DERIV),
        }
    }
}

impl CustomFamily for PoissonLogFamily {
    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }
}

impl CustomFamilyGenerative for PoissonLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "PoissonLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Poisson,
        })
    }
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

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl LogLinkDiagonalIrlsFamily for GammaLogFamily {
    fn family_label(&self) -> &'static str {
        "GammaLogFamily"
    }
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    fn validate_self(&self) -> Result<(), String> {
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }
        Ok(())
    }
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi <= 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!("GammaLogFamily requires positive finite y; found y[{idx}]={yi}"),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow {
        assert!(e_clamped.is_finite());
        assert!((e_clamped.exp() - m).abs() <= 1.0e-8 * m.abs().max(1.0));
        // Gamma(shape=k, scale=mu/k), dropping eta-independent constants.
        let log_lik_increment = prior_w * (-self.shape * (yi / m + m.ln()));
        // Gamma with log mean is non-canonical. Use the exact observed
        // η-space curvature -d²ℓ/dη² = prior_w * shape * y / μ, not the
        // Fisher weight prior_w * shape, so diagonal REML/LAML Hessians
        // use the true Laplace curvature instead of a PQL/Fisher surrogate.
        let observed_weight = prior_w * self.shape * yi / m;
        let score = prior_w * self.shape * (yi / m - 1.0);
        // Mirror the pre-extraction formula z = e + score / w_floored exactly;
        // the driver applies MIN_WEIGHT *before* writing w[i], but the old
        // code divided by the already-floored w[i] for non-degenerate rows,
        // and the floor only activates on the degenerate `observed_weight <=
        // MIN_WEIGHT` tail. Reproduce that branch here to preserve bitwise
        // step shape on every row that used to hit the floor.
        let w_floored = observed_weight.max(MIN_WEIGHT);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight,
            working_step: score / w_floored,
        }
    }
}

impl CustomFamily for GammaLogFamily {
    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
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

        let mut dw = Array1::<f64>::zeros(n);
        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(GamlssError::InvalidInput {
                    reason: format!("GammaLogFamily requires positive finite y; found y[{i}]={yi}"),
                }
                .into());
            }
            let e_raw = eta[i];
            let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
            if self.weights[i] == 0.0 || e != e_raw {
                dw[i] = 0.0;
                continue;
            }
            let m = safe_exp(e).max(MIN_WEIGHT);
            let observed_weight = self.weights[i] * self.shape * yi / m;
            // d/dη [prior_weight * shape * y / exp(η)] = -W_obs.
            // If the positive floor is active, match the evaluated local piece.
            if observed_weight <= MIN_WEIGHT {
                dw[i] = 0.0;
            } else {
                dw[i] = -observed_weight * d_eta[i];
            }
        }
        Ok(Some(dw))
    }
}

impl CustomFamilyGenerative for GammaLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gamma { shape: self.shape },
        })
    }
}

/// Built-in binomial location-scale family with a configurable inverse link.
///
/// Parameters:
/// - Block 0: threshold/location T(covariates)
/// - Block 1: log-scale log σ(covariates)
#[derive(Clone)]
