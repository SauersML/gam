use super::*;

fn normalize_optional_column(value: Option<String>, field: &str) -> Result<Option<String>, String> {
    value
        .map(|value| {
            let value = value.trim();
            if value.is_empty() {
                Err(format!("{field} must be a non-empty column name"))
            } else {
                Ok(value.to_string())
            }
        })
        .transpose()
}

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
        self.family = self.family.and_then(|value| {
            let value = value.trim();
            (!value.eq_ignore_ascii_case("auto")).then(|| value.to_string())
        });
        self.survival_likelihood = self
            .survival_likelihood
            .map(|value| value.trim().to_ascii_lowercase());
        self.baseline_target = self.baseline_target.trim().to_ascii_lowercase();
        self.link = self.link.and_then(|value| {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        });
        self.offset_column = normalize_optional_column(self.offset_column, "offset_column")?;
        self.noise_offset_column =
            normalize_optional_column(self.noise_offset_column, "noise_offset_column")?;
        self.weight_column = normalize_optional_column(self.weight_column, "weight_column")?;
        self.z_column = normalize_optional_column(self.z_column, "z_column")?;

        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            return Err("ridge_lambda must be finite and >= 0".to_string());
        }
        if self.outer_max_iter == Some(0) {
            return Err("outer_max_iter must be >= 1".to_string());
        }
        self.frailty.validate().map_err(|error| error.to_string())?;
        self.spatial_optimization.validate()?;
        let likelihood_mode = parse_survival_likelihood_mode(self.resolved_survival_likelihood())?;
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

    /// The survival likelihood mode this config resolves to for a `Surv(...)`
    /// fit.
    ///
    /// `survival_likelihood` is `None` by default — there is no library-side
    /// string default (#2301). An explicit `Some(mode)` selects that mode; an
    /// unset `None` resolves to the single canonical default `"transformation"`
    /// (Royston-Parmar), the same default the CLI documents. This is the ONE
    /// resolution point: the `Surv(...)` materialization seam, the CLI survival
    /// path, and the pyffi survival path all consult it, so the default lives in
    /// exactly one place. A non-`Surv()` formula never calls this — `Some(_)` on
    /// a non-survival response is a typed configuration error rejected by
    /// [`reject_survival_likelihood_for_nonsurvival`], and `None` is unset.
    pub fn resolved_survival_likelihood(&self) -> &str {
        self.survival_likelihood
            .as_deref()
            .unwrap_or("transformation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_normalizes_front_end_spellings() {
        let resolved = FitConfig {
            family: Some(" AUTO ".to_string()),
            survival_likelihood: Some(" Transformation ".to_string()),
            baseline_target: " Linear ".to_string(),
            ..FitConfig::default()
        }
        .resolve()
        .unwrap();
        assert_eq!(resolved.family, None);
        assert_eq!(resolved.survival_likelihood.as_deref(), Some("transformation"));
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
        assert!(
            FitConfig {
                weight_column: Some("   ".to_string()),
                ..FitConfig::default()
            }
            .resolve()
            .is_err()
        );
    }
}
