use super::*;

/// Validate the survival baseline fields shared by every front end.
pub fn validate_survival_baseline_config(
    likelihood_mode: SurvivalLikelihoodMode,
    baseline_target: &str,
    baseline_scale: Option<f64>,
    baseline_shape: Option<f64>,
    baseline_rate: Option<f64>,
    baseline_makeham: Option<f64>,
) -> Result<(), String> {
    if likelihood_mode == SurvivalLikelihoodMode::Weibull {
        if baseline_rate.is_some() || baseline_makeham.is_some() {
            return Err(
                "survival likelihood 'weibull' does not use baseline_rate or baseline_makeham"
                    .to_string(),
            );
        }
        if !matches!(baseline_target, "linear" | "weibull") {
            return Err(
                "survival likelihood 'weibull' supports only baseline_target 'linear' or 'weibull'"
                    .to_string(),
            );
        }
        return Ok(());
    }

    match baseline_target {
        "linear" => {
            if baseline_scale.is_some()
                || baseline_shape.is_some()
                || baseline_rate.is_some()
                || baseline_makeham.is_some()
            {
                return Err("baseline_target 'linear' does not use baseline parameters".to_string());
            }
        }
        "weibull" => {
            if baseline_rate.is_some() || baseline_makeham.is_some() {
                return Err(
                    "baseline_target 'weibull' does not use baseline_rate or baseline_makeham"
                        .to_string(),
                );
            }
        }
        "gompertz" => {
            if baseline_scale.is_some() || baseline_makeham.is_some() {
                return Err(
                    "baseline_target 'gompertz' does not use baseline_scale or baseline_makeham"
                        .to_string(),
                );
            }
        }
        "gompertz-makeham" => {
            if baseline_scale.is_some() {
                return Err(
                    "baseline_target 'gompertz-makeham' does not use baseline_scale".to_string(),
                );
            }
        }
        other => {
            return Err(format!(
                "unsupported baseline_target '{other}'; use linear, weibull, gompertz, or gompertz-makeham"
            ));
        }
    }
    Ok(())
}

impl FitConfig {
    /// Normalize and validate the canonical configuration contract.
    ///
    /// CLI and JSON layers translate syntax only. Model-family legality and
    /// cross-field invariants live here so direct Rust callers cannot bypass
    /// the same rules enforced by application front ends.
    pub fn resolve(mut self) -> Result<Self, String> {
        self.family = match self.family {
            Some(value) if value.eq_ignore_ascii_case("auto") => None,
            Some(value) => Some(value),
            None => None,
        };
        self.survival_likelihood = self.survival_likelihood.trim().to_ascii_lowercase();
        self.baseline_target = self.baseline_target.trim().to_ascii_lowercase();

        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            return Err("ridge_lambda must be finite and >= 0".to_string());
        }
        if self.outer_max_iter == Some(0) {
            return Err("outer_max_iter must be >= 1".to_string());
        }
        let likelihood_mode = parse_survival_likelihood_mode(&self.survival_likelihood)?;
        validate_survival_baseline_config(
            likelihood_mode,
            &self.baseline_target,
            self.baseline_scale,
            self.baseline_shape,
            self.baseline_rate,
            self.baseline_makeham,
        )?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_normalizes_front_end_spellings() {
        let resolved = FitConfig {
            family: Some("AUTO".to_string()),
            survival_likelihood: " Transformation ".to_string(),
            baseline_target: " Linear ".to_string(),
            ..FitConfig::default()
        }
        .resolve()
        .unwrap();
        assert_eq!(resolved.family, None);
        assert_eq!(resolved.survival_likelihood, "transformation");
        assert_eq!(resolved.baseline_target, "linear");
    }

    #[test]
    fn resolve_rejects_invalid_shared_fields() {
        assert!(
            FitConfig {
                ridge_lambda: f64::NAN,
                ..FitConfig::default()
            }
            .resolve()
            .is_err()
        );
        assert!(
            FitConfig {
                outer_max_iter: Some(0),
                ..FitConfig::default()
            }
            .resolve()
            .is_err()
        );
    }
}
