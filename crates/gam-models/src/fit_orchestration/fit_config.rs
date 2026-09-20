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

/// The `latent_measure` spellings a request may carry. `None` is the default
/// estimated law (the default policy), so the caller need not construct one.
pub(crate) fn parse_latent_measure_spec(
    value: &str,
) -> Result<Option<crate::bms::LatentMeasureSpec>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "gaussian" | "standard-normal" | "standard_normal" => {
            Ok(Some(crate::bms::LatentMeasureSpec::StandardNormal))
        }
        "global-empirical" | "global_empirical" | "empirical" => {
            Ok(Some(crate::bms::LatentMeasureSpec::GlobalEmpirical {
                grid_size: crate::bms::DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
            }))
        }
        "conditional-location-scale" | "conditional_location_scale" => {
            Ok(Some(crate::bms::LatentMeasureSpec::ConditionalLocationScale {
                grid_size: crate::bms::DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
            }))
        }
        other => Err(format!(
            "unsupported latent_measure '{other}'; use auto (the estimated law of the score, the \
             default), gaussian, global-empirical, or conditional-location-scale"
        )),
    }
}

impl FitConfig {
    /// The declared latent law as the validated grid the survival
    /// marginal-slope family anchors on (gam#2923).
    pub(crate) fn declared_latent_law_grid(
        &self,
    ) -> Result<Option<crate::bms::EmpiricalZGrid>, String> {
        self.declared_latent_law
            .as_ref()
            .map(|law| {
                crate::bms::EmpiricalZGrid::new(
                    law.nodes.clone(),
                    law.weights.clone(),
                    "declared latent law",
                )
            })
            .transpose()
    }

    /// Whether this request is a marginal-slope fit, by the predicate
    /// materialization dispatches on: a survival marginal-slope likelihood, or
    /// the Bernoulli marginal-slope request `requests_bernoulli_marginal_slope`
    /// recognizes (an explicit family, `slope_formula`, `z_column` or
    /// `ctn_stage1`). The latent-measure controls are legal exactly where this
    /// holds, and it is the materializer's own predicate rather than a copy of
    /// it, so `resolve` cannot refuse a request the materializer would fit as
    /// marginal slope (gam#2956).
    pub(crate) fn requests_marginal_slope(&self) -> bool {
        self.survival_likelihood.as_deref() == Some("marginal-slope")
            || super::materialize::requests_bernoulli_marginal_slope(self)
    }

    pub(crate) fn marginal_slope_latent_policy(&self) -> crate::bms::LatentZPolicy {
        let mut policy = crate::bms::LatentZPolicy::default();
        if self.frozen_score {
            policy.latent_measure = crate::bms::LatentMeasureSpec::StandardNormal;
        }
        // Validated by `resolve`; an unresolved config keeps the gate rather
        // than failing on a spelling, and `"auto"` names the gate itself.
        if let Some(Ok(Some(spec))) = self
            .latent_measure
            .as_deref()
            .map(parse_latent_measure_spec)
        {
            policy.latent_measure = spec;
        }
        policy
    }

    /// Opt in to cross-process warm starts at the exact supplied root.
    ///
    /// The path is neither canonicalized nor relocated through temp/cache
    /// discovery. Opening remains lazy until a real fit performs its first
    /// persistence operation.
    pub fn with_persistent_warm_start_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.persistent_warm_start_store = Some(
            gam_solve::persistent_warm_start::configured_store(root.into()),
        );
        self
    }

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
        self.resolved_expectile_levels()?;
        self.offset_column = normalize_optional_column(self.offset_column, "offset_column")?;
        self.noise_offset_column =
            normalize_optional_column(self.noise_offset_column, "noise_offset_column")?;
        self.weight_column = normalize_optional_column(self.weight_column, "weight_column")?;
        self.z_column = normalize_optional_column(self.z_column, "z_column")?;
        self.residual_columns = normalize_residual_columns(
            std::mem::take(&mut self.residual_columns),
            self.z_column.as_deref(),
        )?;
        // The block lives inside the Bernoulli marginal-slope likelihood, which
        // front ends select either by name or, with the family left automatic,
        // by the score column; any other family or no score is refused rather
        // than dropping the features on a path that never reads them.
        if !self.residual_columns.is_empty() {
            let names_another_family = self.family.as_deref().is_some_and(|family| {
                let canonical = family.to_ascii_lowercase().replace('_', "-");
                canonical != "bernoulli-marginal-slope" && canonical != "binary-marginal-slope"
            });
            if names_another_family
                || self.z_column.is_none()
                || self.survival_likelihood.as_deref() == Some("marginal-slope")
            {
                return Err(
                    "residual_columns requires a Bernoulli marginal-slope fit with a z_column \
                     (gam#2924); the survival marginal-slope family takes it once gam#2923 lands"
                        .to_string(),
                );
            }
        }
        // The materializer's own predicate, so every spelling it fits as
        // transformation-normal also carries its config.
        if self.transformation_normal_config.is_some()
            && !(self.transformation_normal
                || family_requests_transformation_normal(self.family.as_deref()))
        {
            return Err("transformation_normal_config requires a transformation-normal fit".to_string());
        }
        if self.frozen_score
            && (self.z_column.is_none()
                || self.ctn_stage1.is_some()
                || !self.requests_marginal_slope())
        {
            return Err("frozen_score requires a marginal-slope fit with an explicit z_column and no integrated CTN recipe".to_string());
        }
        self.latent_measure = self
            .latent_measure
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty());
        if self.declared_latent_law.is_some() {
            if !self.requests_marginal_slope() {
                return Err("declared_latent_law applies to marginal-slope fits only".to_string());
            }
            if self.z_column.is_none() || self.ctn_stage1.is_some() {
                return Err(
                    "declared_latent_law is a statement about an explicit z_column and cannot be \
                     combined with an integrated CTN recipe"
                        .to_string(),
                );
            }
            if self.frozen_score || self.latent_measure.is_some() {
                return Err(
                    "declared_latent_law already fixes the latent measure; do not combine it with \
                     frozen_score or latent_measure"
                        .to_string(),
                );
            }
            self.declared_latent_law_grid()?;
        }
        if let Some(measure) = self.latent_measure.as_deref() {
            let spec = parse_latent_measure_spec(measure)?;
            if !self.requests_marginal_slope() {
                return Err("latent_measure applies to marginal-slope fits only".to_string());
            }
            // Compared by the law it names, so every spelling of the Gaussian
            // declaration agrees with frozen_score.
            if self.frozen_score && spec != Some(crate::bms::LatentMeasureSpec::StandardNormal) {
                return Err(format!(
                    "frozen_score declares the Gaussian latent law; it cannot be combined with latent_measure = '{measure}'"
                ));
            }
        }
        if self
            .persistent_warm_start_store
            .as_ref()
            .is_some_and(|store| store.root().as_os_str().is_empty())
        {
            return Err("persistent_warm_start_root must not be empty".to_string());
        }

        // Normalize the survival time-anchor override through its one validator,
        // so the CLI flag, a `--request` document, a `gamfit.fit` kwarg and a
        // direct Rust caller are all held to the same contract and report the
        // same message (#2631).
        self.survival_time_anchor = self
            .survival_time_anchor
            .map(crate::survival::validate_survival_time_anchor_override)
            .transpose()?;
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

    /// The expectile levels this config requests, if any.
    ///
    /// `Ok(Some(levels))` when `family` is `"expectile"` (optionally with
    /// inline levels, `"expectile(0.9)"` or `"expectile(0.1, 0.9)"`);
    /// `Ok(None)` for every other family with `expectile_tau` unset. The levels
    /// are a parameter of the expectile family and never select it, so
    /// `expectile_tau` with any other family (including an inferred one) is an
    /// error rather than an ignored field. `Err` also when an expectile request
    /// is malformed: a level outside `(0, 1)`, levels that are not strictly
    /// increasing, or inline levels that contradict `expectile_tau`. When
    /// neither spelling pins the levels, the single median level `[0.5]` (the
    /// ordinary mean fit) is the default.
    pub fn resolved_expectile_levels(&self) -> Result<Option<Vec<f64>>, String> {
        let trimmed = self.family.as_deref().map(str::trim).unwrap_or("");
        let lower = trimmed.to_ascii_lowercase();
        if !(lower == "expectile" || lower.starts_with("expectile(")) {
            return match &self.expectile_tau {
                None => Ok(None),
                Some(levels) => Err(format!(
                    "expectile_tau = {levels:?} requires family = \"expectile\"; got family = {}",
                    self.family
                        .as_deref()
                        .map_or_else(|| "auto".to_string(), |family| format!("\"{family}\""))
                )),
            };
        }
        // Optional inline levels: `expectile(0.9)` or `expectile(0.1, 0.5, 0.9)`.
        let inline_levels = match lower.strip_prefix("expectile(") {
            Some(rest) => {
                let inner = rest.strip_suffix(')').ok_or_else(|| {
                    format!(
                        "expectile family levels must be written as `expectile(τ)` or \
                         `expectile(τ₁, τ₂, …)`; got `{trimmed}`"
                    )
                })?;
                let levels = inner
                    .split(',')
                    .map(|item| {
                        item.trim().parse::<f64>().map_err(|_| {
                            format!("expectile level `{}` is not a finite number", item.trim())
                        })
                    })
                    .collect::<Result<Vec<f64>, _>>()?;
                Some(levels)
            }
            None => None,
        };
        let levels = match (inline_levels, self.expectile_tau.clone()) {
            (Some(a), Some(b)) if a != b => {
                return Err(format!(
                    "expectile levels given both inline (`{trimmed}`) and via expectile_tau \
                     ({b:?}); supply exactly one"
                ));
            }
            (Some(a), _) => a,
            (None, Some(b)) => b,
            (None, None) => vec![0.5],
        };
        if levels.is_empty() {
            return Err("expectile_tau must name at least one expectile level".to_string());
        }
        for &tau in &levels {
            if !(tau.is_finite() && tau > 0.0 && tau < 1.0) {
                return Err(format!(
                    "expectile level τ must be finite and strictly in (0, 1); got {tau}"
                ));
            }
        }
        if levels.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!(
                "expectile levels must be strictly increasing with no duplicates; got {levels:?}"
            ));
        }
        Ok(Some(levels))
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
    /// `reject_survival_only_config_for_nonsurvival`, and `None` is unset.
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
        assert_eq!(
            resolved.survival_likelihood.as_deref(),
            Some("transformation")
        );
        assert_eq!(resolved.baseline_target, "linear");
    }

    /// `expectile_tau` is a parameter of the expectile family and never selects
    /// it: with any other family it is refused rather than ignored, and `τ` is
    /// held to the open unit interval (pyGAM audit F10).
    #[test]
    fn resolve_holds_expectile_tau_to_the_expectile_family_and_the_open_unit_interval() {
        let config = |family: Option<&str>, levels: Option<&[f64]>| FitConfig {
            family: family.map(str::to_string),
            expectile_tau: levels.map(<[f64]>::to_vec),
            ..FitConfig::default()
        };
        for family in [None, Some("auto"), Some("gaussian"), Some("poisson")] {
            for levels in [&[0.9][..], &[0.1, 0.9]] {
                let error = config(family, Some(levels)).resolve().unwrap_err();
                assert!(
                    error.contains("requires family = \"expectile\""),
                    "{family:?} {levels:?}: {error}"
                );
            }
            assert!(config(family, None).resolve().is_ok());
        }
        for tau in [0.0, 1.0, 1.5, -0.1, f64::NAN, f64::INFINITY] {
            let error = config(Some("expectile"), Some(&[tau])).resolve().unwrap_err();
            assert!(error.contains("strictly in (0, 1)"), "{tau}: {error}");
            let error = config(Some("expectile"), Some(&[0.5, tau])).resolve().unwrap_err();
            assert!(error.contains("strictly"), "{tau}: {error}");
        }
        assert!(config(Some("expectile(1.5)"), None).resolve().is_err());
        assert!(config(Some("expectile(0.9)"), Some(&[0.8])).resolve().is_err());
        let resolved = config(Some("Expectile"), Some(&[0.9])).resolve().unwrap();
        assert_eq!(resolved.resolved_expectile_levels(), Ok(Some(vec![0.9])));
        let inline = config(Some("expectile(0.25, 0.75)"), None).resolve().unwrap();
        assert_eq!(inline.resolved_expectile_levels(), Ok(Some(vec![0.25, 0.75])));
        let median = config(Some("expectile"), None).resolve().unwrap();
        assert_eq!(median.resolved_expectile_levels(), Ok(Some(vec![0.5])));
    }

    /// `transformation_normal_config` is legal on exactly the requests the
    /// materializer fits as transformation-normal; family names are
    /// case-insensitive.
    #[test]
    fn transformation_normal_config_follows_the_materializer_family_predicate() {
        let request = |family: Option<&str>, flag: bool| FitConfig {
            family: family.map(str::to_string),
            transformation_normal: flag,
            transformation_normal_config: Some(TransformationNormalConfig::default()),
            ..FitConfig::default()
        };
        for family in [
            "transformation-normal",
            "Transformation-Normal",
            " TRANSFORMATION-NORMAL ",
        ] {
            assert!(
                family_requests_transformation_normal(Some(family)),
                "{family}: the materializer fits this spelling as transformation-normal"
            );
            request(Some(family), false)
                .resolve()
                .unwrap_or_else(|error| panic!("{family}: {error}"));
        }
        request(None, true).resolve().expect("the flag spelling");
        for family in [None, Some("gaussian"), Some("transformation")] {
            let error = request(family, false).resolve().unwrap_err();
            assert_eq!(
                error, "transformation_normal_config requires a transformation-normal fit",
                "{family:?}"
            );
        }
    }

    #[test]
    fn resolve_rejects_invalid_shared_fields() {
        assert!(
            FitConfig {
                weight_column: Some("   ".to_string()),
                ..FitConfig::default()
            }
            .resolve()
            .is_err()
        );
    }

    #[test]
    fn latent_measure_controls_follow_the_materialization_marginal_slope_predicate_2956() {
        // With the family left to inference, `slope_formula` and `z_column`
        // select a Bernoulli marginal-slope fit in materialization, so the
        // latent-measure controls must resolve on that request.
        let selected = FitConfig {
            slope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            latent_measure: Some("Global-Empirical".to_string()),
            ..FitConfig::default()
        };
        assert!(super::super::materialize::requests_bernoulli_marginal_slope(&selected));
        let resolved = selected
            .resolve()
            .expect("a z_column-selected marginal-slope fit accepts latent_measure");
        assert_eq!(resolved.latent_measure.as_deref(), Some("global-empirical"));
        FitConfig {
            z_column: Some("z".to_string()),
            frozen_score: true,
            ..FitConfig::default()
        }
        .resolve()
        .expect("a z_column-selected marginal-slope fit accepts frozen_score");

        // A request that is not a marginal-slope fit is still refused, by name.
        let refused = FitConfig {
            family: Some("gaussian".to_string()),
            latent_measure: Some("global-empirical".to_string()),
            ..FitConfig::default()
        }
        .resolve()
        .expect_err("latent_measure outside a marginal-slope fit");
        assert_eq!(refused, "latent_measure applies to marginal-slope fits only");
        let refused_frozen = FitConfig {
            frozen_score: true,
            ..FitConfig::default()
        }
        .resolve()
        .expect_err("frozen_score outside a marginal-slope fit");
        assert!(
            refused_frozen.starts_with("frozen_score requires a marginal-slope fit"),
            "{refused_frozen}"
        );
    }

    #[test]
    fn resolve_admits_residual_columns_only_on_a_bernoulli_marginal_slope_request() {
        fn request(
            family: Option<&str>,
            z_column: Option<&str>,
            survival_likelihood: Option<&str>,
        ) -> Result<FitConfig, String> {
            FitConfig {
                family: family.map(str::to_string),
                z_column: z_column.map(str::to_string),
                survival_likelihood: survival_likelihood.map(str::to_string),
                residual_columns: vec!["r1".to_string(), "r2".to_string()],
                ..FitConfig::default()
            }
            .resolve()
        }
        let refused = |result: Result<FitConfig, String>| {
            result.is_err_and(|reason| reason.contains("residual_columns requires"))
        };
        // Front ends select the family by the score column with the family
        // left automatic (the CLI has no family value for it), or by name.
        assert!(request(None, Some("z"), None).is_ok());
        assert!(request(Some("auto"), Some("z"), None).is_ok());
        assert!(request(Some("bernoulli-marginal-slope"), Some("z"), None).is_ok());
        assert!(refused(request(Some("binomial-probit"), Some("z"), None)));
        assert!(refused(request(None, None, None)));
        assert!(refused(request(None, Some("z"), Some("marginal-slope"))));
    }
}

/// Trim, reject empties and duplicates, and keep the score out of the residual
/// block: a residual column that IS the score would enter the drive twice.
fn normalize_residual_columns(
    columns: Vec<String>,
    z_column: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::with_capacity(columns.len());
    for raw in columns {
        let name = raw.trim();
        if name.is_empty() {
            return Err("residual_columns contains an empty column name".to_string());
        }
        if out.iter().any(|existing| existing == name) {
            return Err(format!("residual_columns names '{name}' more than once"));
        }
        if z_column == Some(name) {
            return Err(format!(
                "residual_columns names the score column '{name}'; the score enters through z_column"
            ));
        }
        out.push(name.to_string());
    }
    Ok(out)
}
