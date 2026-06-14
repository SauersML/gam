
fn reject_marginal_slope_controls_for_transformation_normal(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    let family_requests_marginal_slope = config.family.as_deref().is_some_and(|family| {
        let canonical = family.to_ascii_lowercase().replace('_', "-");
        canonical == "bernoulli-marginal-slope" || canonical == "binary-marginal-slope"
    });
    if family_requests_marginal_slope
        || config.logslope_formula.is_some()
        || config.z_column.is_some()
        || config.ctn_stage1.is_some()
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "transformation_normal cannot be combined with marginal-slope family controls"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject `timewiggle(...)` / `survmodel(...)` in a formula whose response is
/// not `Surv(...)`.
///
/// These two DSL controls only have meaning under the survival likelihood: a
/// `timewiggle(...)` term parameterizes the time-varying baseline-hazard /
/// log-cumulative-hazard surface, and `survmodel(...)` selects the survival
/// likelihood mode. Both are read exclusively by `materialize_survival`. When
/// the main formula has no `Surv(...)` response, leaving them unguarded means
/// the term is parsed and option-validated and then dropped on the floor —
/// the contract violation reported in #371. We error instead, with the same
/// "only supported in the main survival formula" phrasing the auxiliary-formula
/// path already uses.
fn reject_survival_only_terms_for_nonsurvival(parsed: &ParsedFormula) -> Result<(), WorkflowError> {
    if parsed.timewiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "timewiggle(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    if parsed.survivalspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survmodel(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject an *explicitly requested* `linkwiggle(...)` term when the resolved
/// response family is not binomial.
///
/// `linkwiggle(...)` adds a spline-flexible correction to the *link* function
/// (logit / probit / cloglog), which only carries meaning for a binomial mean
/// model — the standard and location-scale materializers wire `wiggle` into the
/// fit only inside their `family.is_binomial()` arm. For a Gaussian / Gamma /
/// Poisson / etc. response the term is built and then dropped on the floor,
/// the same silent-no-op contract violation as #371. We error here.
///
/// This guards only the *explicit* formula term (`parsed.linkwiggle`), not the
/// implicit wiggle auto-derived from a `Flexible` link choice: a flexible link
/// requested against a non-binomial family is a separate, already-handled link
/// concern, and silently declining to add a binomial-only correction there is
/// the intended behavior rather than a dropped user-authored term.
fn reject_explicit_linkwiggle_for_nonbinomial(
    parsed: &ParsedFormula,
    family: &LikelihoodSpec,
) -> Result<(), WorkflowError> {
    if parsed.linkwiggle.is_some() && !family.is_binomial() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) corrects the link function of a binomial mean model \
                     and is only supported for a binomial response; it is meaningless for \
                     the resolved non-binomial family and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

/// Detect whether a response column is binary (0/1 only).
pub fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Verify that the dataset has at least as many rows as the smooth terms in
/// `spec` need for their bases to be well-posed.
///
/// Each [`SmoothBasisSpec`] owns its own `min_sample_rows` lower bound — the
/// B-spline knot count, the *penalized* tensor-product floor (the sum of the
/// per-marginal column counts, not their Kronecker product, because a `te()`
/// is regularized and its effective dof is a small fraction of the column
/// count), the PCA matrix width — so this helper is a thin sum-and-compare:
/// the workflow has no per-basis-kind knowledge. Adding a new smooth kind
/// extends the basis `match` in `min_sample_rows`, not this gate.
///
/// Catches the README-quickstart failure mode (#309) where `n=4` against
/// `y ~ s(x)` would otherwise surface as an opaque `cached inner beta has
/// length 8` message from the inner-state seeding hook.
fn check_smooth_capacity(
    spec: &crate::terms::smooth::TermCollectionSpec,
    n_rows: usize,
    response_name: &str,
) -> Result<(), WorkflowError> {
    // Intercept + 1 dof for the smoothing-parameter optimizer.
    let mut required: usize = 2;
    let mut per_term: Vec<(String, usize)> = Vec::new();
    for term in &spec.smooth_terms {
        let need = term.basis.min_sample_rows();
        required = required.saturating_add(need);
        per_term.push((term.name.clone(), need));
    }
    if per_term.is_empty() || n_rows >= required {
        return Ok(());
    }
    let breakdown = per_term
        .iter()
        .map(|(name, k)| format!("{name}≥{k}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "not enough observations to fit the requested formula: dataset has n={n_rows} \
             rows but the smooth terms on response '{response_name}' need at least \
             {required} rows total ({breakdown}, plus intercept + smoothing-parameter dof) \
             before REML estimation is well-posed. \
             Fix: add more training rows, replace `s(x)` with a linear term, or pass a \
             smaller basis via `s(x, k=3)`."
        ),
    })
}

/// Project an ingest-layer [`ColumnKindTag`] (plus the column's level table)
/// onto the [`ResponseColumnKind`] consumed by the family layer.
///
/// `Categorical` carries the source-string levels through so the
/// auto-inference refusal can echo them; `Binary` short-circuits the
/// numeric scan inside [`ResponseFamily::infer_from_response`]; `Continuous`
/// maps to `Numeric` and the family layer scans `y` itself to decide
/// Gaussian vs. Binomial.
pub(crate) fn response_column_kind(data: &Dataset, y_col: usize) -> ResponseColumnKind {
    match data.column_kinds.get(y_col) {
        Some(ColumnKindTag::Categorical) => ResponseColumnKind::Categorical {
            levels: data
                .schema
                .columns
                .get(y_col)
                .map(|sc| sc.levels.clone())
                .unwrap_or_default(),
        },
        Some(ColumnKindTag::Binary) => ResponseColumnKind::Binary,
        Some(ColumnKindTag::Continuous) | None => ResponseColumnKind::Numeric,
    }
}

/// Legality of a `(response family, link)` pairing.
///
/// This is the single source of truth for which links a given response family
/// accepts. It is consulted only when the caller supplied an *explicit* family
/// together with a link (`family=..., link(type=...)`): the link must be
/// validated against that family rather than the family re-inferred from the
/// link. The legal pairings are:
///
/// * `Gaussian` + `Identity`
/// * `{Poisson, Gamma, Tweedie, NegativeBinomial}` + `Log`
/// * `Beta` + `Logit`
/// * `Binomial` + `{Logit, Probit, CLogLog, Sas, BetaLogistic}` (and the
///   Logit-shaped `Mixture`, handled by the caller via `mixture_components`)
///
/// `RoystonParmar` is a flexible-parametric survival family whose link is fixed
/// at construction and is never reached through the scalar link-choice path, so
/// it accepts no link override here.
fn link_legal_for_family(response: &ResponseFamily, link: LinkFunction) -> bool {
    match response {
        ResponseFamily::Gaussian => matches!(link, LinkFunction::Identity),
        ResponseFamily::Poisson
        | ResponseFamily::Gamma
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. } => matches!(link, LinkFunction::Log),
        ResponseFamily::Beta { .. } => matches!(link, LinkFunction::Logit),
        ResponseFamily::Binomial => matches!(
            link,
            LinkFunction::Logit
                | LinkFunction::Probit
                | LinkFunction::CLogLog
                | LinkFunction::Sas
                | LinkFunction::BetaLogistic
        ),
        ResponseFamily::RoystonParmar => false,
    }
}

/// Resolve a family from an optional name, optional link choice, and response data.
///
/// `y_kind` describes the *source* representation of the response column
/// (string-valued `Categorical`, numeric `Binary` short-circuit, or generic
/// `Numeric`). It is consulted only on the auto-detect path — explicit
/// `family=...` always wins — but is required there because the same numeric
/// `y = [0.0, 1.0, ...]` payload may come from a real binary outcome or from
/// a categorical column whose two levels happened to encode to those indices.
/// Routing the kind through [`ResponseFamily::infer_from_response`] is what
/// stops the auto-detector from silently inferring Binomial off encoded
/// strings (see `tests/issues/issue_304`).
pub fn resolve_family(
    family: Option<&str>,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    let nb_theta = negative_binomial_theta.unwrap_or(1.0);
    if !nb_theta.is_finite() || nb_theta <= 0.0 {
        return Err(format!(
            "negative-binomial theta must be finite and > 0; got {nb_theta}"
        ));
    }
    // `link_pinned = true` means the family name carried a specific link suffix
    // (e.g. "binomial-probit"); `false` means the user only declared the response
    // family (e.g. "binomial") and any link_choice may legally refine the link
    // without being treated as a contradiction.
    let explicit: Option<(LikelihoodSpec, bool)> = match family {
        Some(name) => {
            // Accept both '-' and '_' as separators so e.g. "binomial_logit" and
            // "negative-binomial" resolve identically. Canonicalize to '-'.
            let canonical = name.to_ascii_lowercase().replace('_', "-");
            let resolved = match canonical.as_str() {
                "gaussian" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    false,
                ),
                "binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "binomial-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "binomial-probit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Probit),
                    ),
                    true,
                ),
                "binomial-cloglog" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::CLogLog),
                    ),
                    true,
                ),
                "latent-cloglog-binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::LatentCLogLog(
                            LatentCLogLogState::new(1.0)
                                .map_err(|err| format!("latent cloglog default state: {err}"))?,
                        ),
                    ),
                    true,
                ),
                "poisson" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Poisson,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // #983: a user-supplied `--negative-binomial-theta` holds θ
                // fixed at exactly that value (`theta_fixed = true` →
                // `FixedNegBinTheta` scale → the PIRLS refresh gate
                // `negbin_theta_is_estimated()` stays closed). With no flag,
                // θ is the running ML estimate (the #802 default seed 1.0).
                "nb" | "negbin" | "negative-binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::NegativeBinomial {
                            theta: nb_theta,
                            theta_fixed: negative_binomial_theta.is_some(),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                "negative-binomial-log" => (
                    LikelihoodSpec::new(
                        ResponseFamily::NegativeBinomial {
                            theta: nb_theta,
                            theta_fixed: negative_binomial_theta.is_some(),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    true,
                ),
                "beta" | "beta-regression" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "beta-logit" | "beta-regression-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "gamma" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gamma,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // Royston-Parmar flexible-parametric survival and the
                // transformation-normal response model are CLI/formula families
                // whose materialization is dispatched before the scalar GLM
                // family resolver runs (survival via `Surv(...)`, transformation
                // via the dedicated transformation-normal path). They are listed
                // here so this resolver is the single total source of truth for
                // every family name the surface accepts: `royston-parmar` maps to
                // the canonical flexible-parametric likelihood, and
                // `transformation-normal` shares Gaussian-identity scalar
                // semantics (the transformation is learned outside this spec).
                "royston-parmar" => (LikelihoodSpec::royston_parmar(), true),
                "transformation-normal" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    true,
                ),
                // Tweedie compound-Poisson-Gamma family. The variance power
                // p must lie strictly in (1, 2); we default to mgcv's
                // canonical p = 1.5. The link is fixed to log (the only link
                // wired through the Tweedie working-response and dispersion
                // machinery). "tw" matches mgcv's family alias.
                "tweedie" | "tw" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie { p: 1.5 },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                "tweedie-log" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie { p: 1.5 },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    true,
                ),
                "multinomial" | "multinomial-logit" | "categorical" | "categorical-logit"
                | "softmax" => {
                    // Multinomial-logit is a vector-response family with K-1
                    // active linear predictors and a per-row dense Fisher
                    // block — it cannot be represented by the scalar
                    // `LikelihoodSpec` (one `ResponseFamily` × one
                    // `InverseLink`) that this entry point produces.
                    //
                    // The principled coefficient-space solver lives in
                    // `crate::families::multinomial::fit_penalized_multinomial`,
                    // which routes the canonical
                    // `MultinomialLogitLikelihood: VectorLikelihood` through
                    // `crate::pirls::dense_block_xtwx` in output-major
                    // coefficient ordering. The forthcoming
                    // `gamfit.fit_multinomial(...)` Python entry exposes that
                    // path with formula → design wiring; until that wrapper
                    // lands, callers reach the driver directly through the
                    // FFI surface.
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "family '{name}' is a vector-response family; use \
                             the dedicated multinomial entry point \
                             (`crate::families::multinomial::fit_penalized_multinomial` \
                             in Rust, or `gamfit.fit_multinomial(...)` in Python) \
                             rather than the scalar `fit(family=...)` path"
                        ),
                    }
                    .into());
                }
                _ => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!("unknown family '{name}'"),
                    }
                    .into());
                }
            };
            Some(resolved)
        }
        None => {
            if negative_binomial_theta.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "negative_binomial_theta requires family='negative-binomial'"
                        .to_string(),
                }
                .into());
            }
            None
        }
    };

    if let Some(choice) = link_choice {
        let from_link: LikelihoodSpec = if let Some(components) = choice.mixture_components.as_ref()
        {
            let n = components.len();
            let free = n.saturating_sub(1);
            let mix_spec = MixtureLinkSpec {
                components: components.clone(),
                initial_rho: Array1::<f64>::zeros(free),
            };
            let state = state_fromspec(&mix_spec)
                .map_err(|err| format!("mixture link initial state: {err}"))?;
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state))
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                ),
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodSpec::new(
                            ResponseFamily::Poisson,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    } else {
                        LikelihoodSpec::new(
                            ResponseFamily::Gamma,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    }
                }
                LinkFunction::Logit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                ),
                LinkFunction::Probit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                ),
                LinkFunction::CLogLog => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                ),
                LinkFunction::Sas => {
                    // The SAS initial state (epsilon, log_delta) is carried into
                    // the fit through `FitOptions.sas_link`, not the family spec:
                    // the standard path's `effective_sas_link_for_family` rebuilds
                    // the inverse link from that option, overriding whatever the
                    // family embeds here. The canonical zero seed is therefore the
                    // correct, link-only placeholder for family resolution.
                    let state = state_from_sasspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("SAS link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(state))
                }
                LinkFunction::BetaLogistic => {
                    let state = state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("Beta-Logistic link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(state))
                }
            }
        };
        if let Some((explicit_spec, link_pinned)) = explicit.as_ref() {
            // An explicit response family was supplied: never re-infer the
            // family from the link. Validate that the requested link is legal
            // for *this* family, then apply the link (carrying any embedded
            // Sas/BetaLogistic/Mixture state, which `from_link.link` already
            // holds) to the explicit family's response variant (preserving e.g.
            // NB theta, Tweedie p, Beta phi).
            let mixture_requested = choice.mixture_components.is_some();
            let legal = if mixture_requested {
                // The mixture link is a Binomial latent construct; it has no
                // legal pairing with any other response family.
                matches!(explicit_spec.response, ResponseFamily::Binomial)
            } else {
                link_legal_for_family(&explicit_spec.response, choice.link)
            };
            if !legal {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "link '{}' is not supported for family '{}'",
                        choice.link.name(),
                        explicit_spec.response.name()
                    ),
                }
                .into());
            }
            // A family name that pinned its own link (e.g. "binomial-probit")
            // may not be re-pointed at a different link by `link(type=...)`.
            if *link_pinned && explicit_spec.link.link_function() != from_link.link.link_function()
            {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "family '{}' pins link '{}', which conflicts with requested link '{}'",
                        explicit_spec.name(),
                        explicit_spec.link.link_function().name(),
                        choice.link.name(),
                    ),
                }
                .into());
            }
            return Ok(LikelihoodSpec::new(
                explicit_spec.response.clone(),
                from_link.link,
            ));
        }
        return Ok(from_link);
    }

    if let Some((spec, _)) = explicit {
        return Ok(spec);
    }

    // Auto-detect: delegate to `ResponseFamily::infer_from_response` so the
    // refusal policy for non-numeric response columns lives in one place
    // (the family layer), not duplicated across every entry point. The link
    // is derived from the inferred response: Binomial → Logit, Poisson → Log,
    // Gaussian → Identity. The link_choice branch above already covered the case where
    // the user pinned a link without a family.
    let response = ResponseFamily::infer_from_response(y, y_kind).map_err(|refusal| {
        let err: String = WorkflowError::InvalidConfig {
            reason: refusal.message_for(response_name),
        }
        .into();
        err
    })?;
    let link = match response {
        ResponseFamily::Binomial => InverseLink::Standard(StandardLink::Logit),
        ResponseFamily::Poisson => InverseLink::Standard(StandardLink::Log),
        _ => InverseLink::Standard(StandardLink::Identity),
    };
    Ok(LikelihoodSpec::new(response, link))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Canonical termspec lowering path: formula DSL builds the initial
/// `SmoothBasisSpec`, then any `gamfit.fit(..., smooths={...})` Python
/// override registry entry whose `feature_cols` match the term's column set
/// replaces the spec's kind-specific tunables in place (explicit center
/// coordinate matrices, knot vectors, kernel hyperparameters). When all
/// descriptor fields default to the same values the DSL would auto-pick, the
/// override is a no-op and the spec is bit-identical to the formula-only
/// path. Callers that don't have overrides simply pass `smooth_overrides =
/// None`.
pub(crate) fn build_termspec_with_geometry_and_overrides(
    terms: &[ParsedTerm],
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
    scale_dimensions: bool,
    policy: &crate::resource::ResourcePolicy,
    smooth_overrides: Option<&JsonValue>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes, policy)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    if let Some(overrides) = smooth_overrides {
        crate::terms::smooth_overrides::apply_smooth_overrides(
            &mut spec,
            overrides,
            data,
            inference_notes,
        )
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    }
    Ok(spec)
}

fn linear_term_training_column(
    data: &Dataset,
    term: &LinearTermSpec,
) -> Result<Array1<f64>, WorkflowError> {
    let cols = term.effective_feature_cols();
    if cols.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linear term '{}' has no feature columns; cannot build its training column",
                term.name
            ),
        });
    }
    let n = data.values.nrows();
    let mut out = Array1::<f64>::ones(n);
    for &col in &cols {
        if col >= data.values.ncols() {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "linear term '{}' feature column {} out of bounds for {} columns",
                    term.name,
                    col,
                    data.values.ncols()
                ),
            }
            .into());
        }
        for row in 0..n {
            out[row] *= data.values[[row, col]];
        }
    }
    Ok(out)
}

fn residualize_against_orthonormal_basis(
    column: &Array1<f64>,
    basis: &[Array1<f64>],
) -> Array1<f64> {
    let mut residual = column.clone();
    for q in basis {
        let coeff = residual.dot(q);
        residual.scaled_add(-coeff, q);
    }
    residual
}

fn l2_norm(column: &Array1<f64>) -> f64 {
    column.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn prune_unidentified_linear_terms_for_marginal_slope(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    label: &str,
    inference_notes: &mut Vec<String>,
) -> Result<(), WorkflowError> {
    if spec.linear_terms.is_empty() {
        return Ok(());
    }

    let n = data.values.nrows();
    if n == 0 {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: cannot rank-check scalar terms on zero rows"),
        });
    }

    let mut basis = Vec::<Array1<f64>>::new();
    let intercept = Array1::<f64>::ones(n);
    let intercept_norm = l2_norm(&intercept);
    if intercept_norm == 0.0 || !intercept_norm.is_finite() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: implicit intercept has invalid norm {intercept_norm}"),
        });
    }
    basis.push(intercept.mapv(|v| v / intercept_norm));

    let rank_alpha = crate::linalg::faer_ndarray::default_rrqr_rank_alpha();
    let mut scale = intercept_norm.max(1.0);
    let mut kept = Vec::<LinearTermSpec>::with_capacity(spec.linear_terms.len());
    let mut dropped = Vec::<String>::new();

    for term in &spec.linear_terms {
        let column = linear_term_training_column(data, term)?;
        let norm = l2_norm(&column);
        if !norm.is_finite() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!("{label}: linear term '{}' has non-finite norm", term.name),
            });
        }
        scale = scale.max(norm.max(1.0));
        let residual = residualize_against_orthonormal_basis(&column, &basis);
        let residual_norm = l2_norm(&residual);
        let tol = rank_alpha * f64::EPSILON * ((n + basis.len() + 1).max(1) as f64) * scale;
        let is_data_redundant = residual_norm <= tol;
        let has_constraints = term.coefficient_min.is_some() || term.coefficient_max.is_some();
        if is_data_redundant {
            if has_constraints {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: constrained linear term '{}' is redundant with the implicit \
                         intercept or earlier scalar terms; remove the constraint or the \
                         redundant term",
                        term.name
                    ),
                });
            }
            if term.double_penalty {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: explicitly penalized linear term '{}' is redundant with the \
                         implicit intercept or earlier scalar terms; remove the redundant term \
                         instead of relying on a ridge to identify a duplicate data direction",
                        term.name
                    ),
                });
            }
            dropped.push(format!(
                "{} (residual_norm={:.3e}, tol={:.3e})",
                term.name, residual_norm, tol
            ));
            continue;
        }
        if residual_norm > tol {
            basis.push(residual.mapv(|v| v / residual_norm));
        }
        kept.push(term.clone());
    }

    if !dropped.is_empty() {
        inference_notes.push(format!(
            "{label}: removed {} scalar term(s) that add no identifiable \
             direction beyond the implicit intercept and earlier scalar terms: {}",
            dropped.len(),
            dropped.join(", ")
        ));
        spec.linear_terms = kept;
    }
    Ok(())
}

fn standard_adaptive_regularization_options(
    config: &FitConfig,
) -> Option<AdaptiveRegularizationOptions> {
    let enabled = config.adaptive_regularization.unwrap_or(false);
    enabled.then(|| AdaptiveRegularizationOptions {
        enabled: true,
        ..AdaptiveRegularizationOptions::default()
    })
}

fn resolve_survival_marginal_slope_base_link(
    linkspec: Option<&crate::inference::formula_dsl::LinkFormulaSpec>,
) -> Result<InverseLink, String> {
    let Some(linkspec) = linkspec else {
        return Ok(InverseLink::Standard(StandardLink::Probit));
    };
    let choice = parse_link_choice(Some(&linkspec.link), false)?
        .ok_or_else(|| "invalid survival marginal-slope link".to_string())?;
    if choice.mixture_components.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survival marginal-slope currently supports only link(type=probit)".to_string(),
        }
        .into());
    }
    match choice.link {
        LinkFunction::Probit => Ok(InverseLink::Standard(StandardLink::Probit)),
        other => Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival marginal-slope currently supports only link(type=probit), got {other:?}"
            ),
        }
        .into()),
    }
}

/// Canonical baseline-time stack shared by the workflow materializer and the
/// CLI survival path (`crate::bin::main`-side `run_survival`). Both entry points
/// build the survival time block identically — baseline offsets, derivative
/// guard, optional baseline time-wiggle augmentation — so the assembly lives
/// here once and the CLI consumes it through a thin re-export rather than
/// reconstructing the same decision tree.
pub struct PreparedSurvivalTimeStack {
    pub eta_offset_entry: Array1<f64>,
    pub eta_offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub time_design_entry: crate::matrix::DesignMatrix,
    pub time_design_exit: crate::matrix::DesignMatrix,
    pub time_design_derivative_exit: crate::matrix::DesignMatrix,
    pub time_penalties: Vec<Array2<f64>>,
    pub time_nullspace_dims: Vec<usize>,
    pub timewiggle_build: Option<crate::families::survival_construction::SurvivalTimeWiggleBuild>,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
}

pub fn prepare_survival_time_stack(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &crate::families::survival_construction::SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
    inverse_link: Option<&InverseLink>,
    time_anchor: f64,
    derivative_guard: f64,
    time_build: &crate::families::survival_construction::SurvivalTimeBuildOutput,
    effective_timewiggle: Option<&LinkWiggleFormulaSpec>,
    latent_loading: Option<crate::families::lognormal_kernel::HazardLoading>,
) -> Result<PreparedSurvivalTimeStack, String> {
    let (
        mut eta_offset_entry,
        mut eta_offset_exit,
        mut derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    ) = if let Some(loading) = latent_loading {
        let offsets =
            build_latent_survival_baseline_offsets(age_entry, age_exit, baseline_cfg, loading)?;
        (
            offsets.loaded_eta_entry,
            offsets.loaded_eta_exit,
            offsets.loaded_derivative_exit,
            offsets.unloaded_mass_entry,
            offsets.unloaded_mass_exit,
            offsets.unloaded_hazard_exit,
        )
    } else {
        // Baseline-hazard barrier conditioning for the marginal-slope likelihood
        // (gam#797). That likelihood carries `-d·log(qd1)`, a log-barrier on the
        // baseline-hazard time derivative `qd1 = X_d·β_time + derivative_offset`.
        // The default `baseline-target=linear` is DEGENERATE for this barrier:
        // `evaluate_survival_baseline` returns `(0, 0)` for Linear, so the offset
        // collapses to `derivative_guard` (1e-6) and the I-spline time seed starts
        // at `qd1 ≈ 1e-6` — exactly ON the barrier boundary, where the
        // self-concordant Newton step is `∝ qd1` (intrinsically ~1e-4), the
        // barrier gradient/Hessian are ~1e6 / ~1e12, and the inner joint-Newton
        // crawls and never reaches the data-scale baseline within the cycle
        // budget — every outer seed is rejected and the fit hard-fails.
        //
        // Condition the COLD START by building the baseline OFFSET from a fixed,
        // data-seeded Weibull (scale = mean positive exit time, shape = 1) instead
        // of the zero-derivative Linear baseline, but ONLY for the offset: the
        // outer `baseline_cfg.target` stays `Linear`, so the
        // `baseline_cfg.target != Linear` optimize gate
        // (the gradient baseline optimizers) never fires and no baseline-shape
        // search is introduced. With shape = 1 the Weibull baseline-hazard
        // derivative is `1/age_exit` (the natural data hazard scale), so the seed
        // starts with `qd1` at O(1/T) interior — barrier gradient O(10-10²),
        // comparable to the marginal/logslope blocks — and `β_time ≈ 0`. This
        // changes only the STARTING point / offset split: the I-spline still learns
        // the data-driven deviation from this parametric baseline (the converged
        // fitted hazard is the same flexible family), so the fix is a pure
        // preconditioning of the cold start. Gated to MarginalSlope with a Linear
        // target so every other Linear-baseline survival path is byte-unchanged.
        let conditioning_cfg;
        let offset_cfg = if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope
            && baseline_cfg.target == SurvivalBaselineTarget::Linear
        {
            let scale =
                crate::families::survival_construction::positive_survival_time_seed(age_exit);
            conditioning_cfg = crate::families::survival_construction::SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(scale),
                shape: Some(1.0),
                rate: None,
                makeham: None,
            };
            &conditioning_cfg
        } else {
            baseline_cfg
        };
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                age_entry,
                age_exit,
                offset_cfg,
                likelihood_mode,
                inverse_link,
            )?;
        let n = age_entry.len();
        (
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::zeros(n),
        )
    };
    add_survival_time_derivative_guard_offset(
        age_entry,
        age_exit,
        time_anchor,
        derivative_guard,
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative_exit = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(wiggle) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative_exit,
            wiggle.ncols,
        );
        for (idx, penalty) in wiggle.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + wiggle.ncols, p_base + wiggle.ncols));
            embedded
                .slice_mut(s![
                    p_base..p_base + wiggle.ncols,
                    p_base..p_base + wiggle.ncols
                ])
                .assign(penalty);
            time_penalties.push(embedded);
            time_nullspace_dims.push(wiggle.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: wiggle.knots.clone(),
            degree: wiggle.degree,
            ncols: wiggle.ncols,
        });
    }
    Ok(PreparedSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative_exit,
        time_penalties,
        time_nullspace_dims,
        timewiggle_build,
        timewiggle_block,
    })
}

fn resolve_continuous_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: &str,
    role: &str,
) -> Result<Array1<f64>, WorkflowError> {
    let col_idx = resolve_role_col(col_map, column_name, role)?;
    let values = data.values.column(col_idx).to_owned();
    for (row_idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "{role} column '{column_name}' contains non-finite value at row {row_idx}: {value}"
                ),
            });
        }
    }
    Ok(values)
}

pub fn resolve_offset_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::zeros(data.values.nrows()));
    };
    resolve_continuous_column(data, col_map, column_name, "offset")
}

pub fn resolve_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::ones(data.values.nrows()));
    };
    let values = resolve_continuous_column(data, col_map, column_name, "weights")?;
    for (row_idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "weights column '{column_name}' must be non-negative; found {value} at row {row_idx}"
                ),
            });
        }
    }
    Ok(values)
}

const MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR: f64 = 1e-12;

fn validate_bernoulli_marginal_slope_z_column_variance(
    z_column: &str,
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<(), WorkflowError> {
    if z.len() != weights.len() {
        return Err(WorkflowError::SchemaMismatch {
            reason: format!(
                "z_column '{z_column}' length mismatch for bernoulli-marginal-slope: z={}, weights={}",
                z.len(),
                weights.len()
            ),
        });
    }
    let n = z.len();
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "z_column '{z_column}' cannot be weighted for bernoulli-marginal-slope because the fit data have non-positive or non-finite total weight"
            ),
        });
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let weighted_sd = var.sqrt();
    if weighted_sd.is_finite() && weighted_sd > MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR {
        return Ok(());
    }

    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() <= MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR);
    let unique_count = sorted.len();
    let value_summary = match sorted.as_slice() {
        [] => "no observed finite values".to_string(),
        [only] => format!("all {n} values ~= {only:.6}"),
        [first, second] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}")
        }
        [first, second, ..] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}, ...")
        }
    };
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "z_column '{z_column}' has zero weighted variance on the fit data ({value_summary}; weighted_sd={weighted_sd:.6e}, n={n}); bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score. Check the score column and fit population."
        ),
    })
}

#[derive(Clone)]
enum LatentInitSpec {
    Pca,
    Random,
    Explicit(Array2<f64>),
}

#[derive(Clone)]
struct LatentAuxPriorSpec {
    u: Array2<f64>,
    family: AuxPriorFamily,
    strength: AuxPriorStrength,
}

#[derive(Clone)]
struct LatentDimSelectionSpec {
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentAuxOutcomeSpec {
    family: crate::terms::behavioral_head::AuxOutcomeFamily,
    /// Behavioral labels, length `n`. Binomial: 0/1; Multinomial: class index.
    y: Array1<f64>,
    /// Optional per-row head weight (semi-supervised); `None` ⇒ all rows
    /// labeled with unit weight. `0.0` on a row excludes it from the head
    /// channel — the missing-label seam.
    row_weights: Option<Array1<f64>>,
    /// ARD log-precision seed composed with the head (length `d`).
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentManifoldSpec {
    manifold: LatentManifold,
    auto: bool,
}

#[derive(Clone)]
struct LatentSpec {
    target: String,
    n: usize,
    d: usize,
    init: LatentInitSpec,
    manifold: LatentManifoldSpec,
    retraction_registry: LatentRetractionRegistry,
    aux_prior: Option<LatentAuxPriorSpec>,
    dim_selection: Option<LatentDimSelectionSpec>,
    aux_outcome: Option<LatentAuxOutcomeSpec>,
    explicit_none_mode: bool,
}

fn json_array2(value: &JsonValue, context: &str) -> Result<Array2<f64>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a two-dimensional numeric array"))?;
    let n = rows.len();
    let first = rows
        .first()
        .and_then(|row| row.as_array())
        .ok_or_else(|| format!("{context} must contain array rows"))?;
    let d = first.len();
    let mut out = Array2::<f64>::zeros((n, d));
    for (i, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .ok_or_else(|| format!("{context} row {i} must be an array"))?;
        if row.len() != d {
            return Err(format!(
                "{context} row {i} has length {}, expected {d}",
                row.len()
            ));
        }
        for (j, cell) in row.iter().enumerate() {
            let value = cell
                .as_f64()
                .ok_or_else(|| format!("{context}[{i}][{j}] must be a finite number"))?;
            if !value.is_finite() {
                return Err(format!("{context}[{i}][{j}] must be finite"));
            }
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn json_array1(value: &JsonValue, context: &str) -> Result<Array1<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a numeric array"))?;
    let mut out = Array1::<f64>::zeros(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}[{idx}] must be finite"));
        }
        out[idx] = value;
    }
    Ok(out)
}

fn parse_latent_manifold(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentManifoldSpec, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    };
    if value
        .as_str()
        .is_some_and(|s| s.eq_ignore_ascii_case("auto"))
    {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    }
    let parse_named = |name: &str| -> Result<LatentManifold, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(LatentManifold::Euclidean),
            "circle" | "s1" | "periodic" => {
                let radians = LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                };
                if d == 1 {
                    Ok(radians)
                } else {
                    Ok(LatentManifold::Product(
                        (0..d).map(|_| radians.clone()).collect(),
                    ))
                }
            }
            "sphere" | "sn" => Ok(LatentManifold::Sphere { dim: d }),
            "torus" => Ok(LatentManifold::Product(
                (0..d)
                    .map(|_| LatentManifold::Circle {
                        period: std::f64::consts::TAU,
                    })
                    .collect(),
            )),
            "cylinder" => {
                if d < 2 {
                    return Err(format!("{context}='cylinder' requires d >= 2"));
                }
                let mut parts = Vec::with_capacity(d);
                parts.push(LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                });
                for _ in 1..d {
                    parts.push(LatentManifold::Euclidean);
                }
                Ok(LatentManifold::Product(parts))
            }
            other => Err(format!(
                "{context} must be 'auto', 'euclidean', 'circle', 'sphere', 'torus', or 'cylinder'; got '{other}'"
            )),
        }
    };
    let manifold = if let Some(name) = value.as_str() {
        parse_named(name)?
    } else if let Some(obj) = value.as_object() {
        let kind = obj
            .get("type")
            .or_else(|| obj.get("kind"))
            .and_then(JsonValue::as_str)
            .unwrap_or("euclidean");
        match kind.to_ascii_lowercase().as_str() {
            "auto" => {
                return Ok(LatentManifoldSpec {
                    manifold: LatentManifold::Euclidean,
                    auto: true,
                });
            }
            "interval" => {
                let lo = obj
                    .get("lo")
                    .or_else(|| obj.get("min"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.lo is required for interval"))?;
                let hi = obj
                    .get("hi")
                    .or_else(|| obj.get("max"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.hi is required for interval"))?;
                if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                    return Err(format!("{context} interval requires finite lo < hi"));
                }
                LatentManifold::Interval { lo, hi }
            }
            other => parse_named(other)?,
        }
    } else if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts
                .push(parse_latent_manifold(Some(item), 1, &format!("{context}[{idx}]"))?.manifold);
        }
        LatentManifold::Product(parts)
    } else {
        return Err(format!(
            "{context} must be a string, object, or product array"
        ));
    };
    if manifold.ambient_dim(d) != d {
        return Err(format!(
            "{context} ambient dimension {} does not match latent d={d}",
            manifold.ambient_dim(d)
        ));
    }
    Ok(LatentManifoldSpec {
        manifold,
        auto: false,
    })
}

fn parse_retraction_kind(
    value: &JsonValue,
    fallback_dim: usize,
    context: &str,
) -> Result<RetractionKind, String> {
    let parse_named = |name: &str| -> Result<RetractionKind, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(RetractionKind::euclidean(fallback_dim)),
            "circle" | "s1" | "periodic" => {
                if fallback_dim == 1 {
                    Ok(RetractionKind::Circle)
                } else {
                    Ok(RetractionKind::Product(ProductRetraction {
                        parts: (0..fallback_dim).map(|_| RetractionKind::Circle).collect(),
                    }))
                }
            }
            "sphere" | "sn" => Ok(RetractionKind::Sphere { dim: fallback_dim }),
            other => Err(format!(
                "{context} must be 'euclidean', 'circle', 'sphere', or a product; got '{other}'"
            )),
        }
    };
    if let Some(name) = value.as_str() {
        return parse_named(name);
    }
    if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts.push(parse_retraction_kind(
                item,
                1,
                &format!("{context}[{idx}]"),
            )?);
        }
        return Ok(RetractionKind::Product(ProductRetraction { parts }));
    }
    let obj = value
        .as_object()
        .ok_or_else(|| format!("{context} must be a string, object, or product array"))?;
    let kind = obj
        .get("type")
        .or_else(|| obj.get("kind"))
        .and_then(JsonValue::as_str)
        .unwrap_or("euclidean");
    match kind.to_ascii_lowercase().as_str() {
        "euclidean" | "r" | "real" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::euclidean(dim))
        }
        "circle" | "s1" | "periodic" => Ok(RetractionKind::Circle),
        "sphere" | "sn" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::Sphere { dim })
        }
        "product" => {
            let items = obj
                .get("parts")
                .or_else(|| obj.get("components"))
                .and_then(JsonValue::as_array)
                .ok_or_else(|| format!("{context}.parts is required for product retraction"))?;
            let mut parts = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                parts.push(parse_retraction_kind(
                    item,
                    1,
                    &format!("{context}.parts[{idx}]"),
                )?);
            }
            Ok(RetractionKind::Product(ProductRetraction { parts }))
        }
        other => parse_named(other),
    }
}

fn parse_latent_retraction(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentRetractionRegistry, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentRetractionRegistry::all_euclidean());
    };
    let kind = parse_retraction_kind(value, d, context)?;
    let registry = LatentRetractionRegistry::new(kind);
    registry.validate_dim(d, context)?;
    Ok(registry)
}

fn parse_latent_specs(payload: Option<&JsonValue>) -> Result<Vec<LatentSpec>, String> {
    let Some(payload) = payload.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = payload
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut specs = Vec::with_capacity(map.len());
    for (key, raw) in map {
        let obj = raw
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let target = obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or(key)
            .to_string();
        let n = obj
            .get("n")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].n is required"))? as usize;
        let d = obj
            .get("d")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].d is required"))? as usize;
        if n == 0 || d == 0 {
            return Err(format!("latents['{key}'] requires positive n and d"));
        }
        let manifold = parse_latent_manifold(
            obj.get("manifold"),
            d,
            &format!("latents['{key}'].manifold"),
        )?;
        let retraction_registry = parse_latent_retraction(
            obj.get("retraction"),
            d,
            &format!("latents['{key}'].retraction"),
        )?;
        let init = match obj.get("init") {
            None => LatentInitSpec::Pca,
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("pca")) =>
            {
                LatentInitSpec::Pca
            }
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("random")) =>
            {
                LatentInitSpec::Random
            }
            Some(value) => {
                LatentInitSpec::Explicit(json_array2(value, &format!("latents['{key}'].init"))?)
            }
        };
        let aux_prior = match obj.get("aux_prior").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                let aux = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_prior must be an object"))?;
                let u = json_array2(
                    aux.get("u")
                        .ok_or_else(|| format!("latents['{key}'].aux_prior.u is required"))?,
                    &format!("latents['{key}'].aux_prior.u"),
                )?;
                let family = match aux
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("ridge")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "ridge" => AuxPriorFamily::Ridge,
                    "linear" => AuxPriorFamily::Linear,
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_prior.family must be 'ridge' or 'linear', got '{other}'"
                        ));
                    }
                };
                let strength = match aux.get("strength") {
                    None => AuxPriorStrength::Fixed(1.0),
                    Some(value)
                        if value
                            .as_str()
                            .is_some_and(|s| s.eq_ignore_ascii_case("auto")) =>
                    {
                        AuxPriorStrength::Auto
                    }
                    Some(value) => {
                        let mu = value.as_f64().ok_or_else(|| {
                            format!(
                                "latents['{key}'].aux_prior.strength must be positive or 'auto'"
                            )
                        })?;
                        if !mu.is_finite() || mu <= 0.0 {
                            return Err(format!(
                                "latents['{key}'].aux_prior.strength must be positive"
                            ));
                        }
                        AuxPriorStrength::Fixed(mu)
                    }
                };
                Some(LatentAuxPriorSpec {
                    u,
                    family,
                    strength,
                })
            }
        };
        let dim_selection = match obj.get("dim_selection") {
            None | Some(JsonValue::Bool(false)) => None,
            Some(JsonValue::Bool(true)) => Some(LatentDimSelectionSpec {
                init_log_precision: None,
            }),
            Some(value) => {
                let dim = value.as_object().ok_or_else(|| {
                    format!("latents['{key}'].dim_selection must be a bool or object")
                })?;
                let init_log_precision = dim
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].dim_selection.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentDimSelectionSpec { init_log_precision })
            }
        };
        let aux_outcome = match obj.get("aux_outcome").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                use crate::terms::behavioral_head::AuxOutcomeFamily;
                let ao = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_outcome must be an object"))?;
                let family = match ao
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("binomial")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "binomial" => AuxOutcomeFamily::Binomial,
                    "multinomial" => {
                        let n_classes = ao
                            .get("n_classes")
                            .and_then(JsonValue::as_u64)
                            .ok_or_else(|| {
                                format!(
                                    "latents['{key}'].aux_outcome.n_classes is required for multinomial"
                                )
                            })? as usize;
                        AuxOutcomeFamily::Multinomial { n_classes }
                    }
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_outcome.family must be 'binomial' or 'multinomial', got '{other}'"
                        ));
                    }
                };
                let y = json_array1(
                    ao.get("y")
                        .ok_or_else(|| format!("latents['{key}'].aux_outcome.y is required"))?,
                    &format!("latents['{key}'].aux_outcome.y"),
                )?;
                if y.len() != n {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.y has length {}, expected n = {n}",
                        y.len()
                    ));
                }
                let row_weights = ao
                    .get("row_weights")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        json_array1(value, &format!("latents['{key}'].aux_outcome.row_weights"))
                    })
                    .transpose()?;
                if let Some(w) = row_weights.as_ref()
                    && w.len() != n
                {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.row_weights has length {}, expected n = {n}",
                        w.len()
                    ));
                }
                let init_log_precision = ao
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].aux_outcome.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentAuxOutcomeSpec {
                    family,
                    y,
                    row_weights,
                    init_log_precision,
                })
            }
        };
        if dim_selection.is_some() && aux_prior.is_none() && aux_outcome.is_none() {
            return Err(format!(
                "latents['{key}'] uses dim_selection without aux_prior or aux_outcome; ARD alone is not an identifiable latent-coordinate gauge"
            ));
        }
        if aux_outcome.is_some() && aux_prior.is_some() {
            return Err(format!(
                "latents['{key}'] specifies both aux_prior and aux_outcome; the auxiliary signal is either a prior (gauge-pin covariate) or a modeled outcome (behavioral head), not both"
            ));
        }
        let explicit_none_mode = obj
            .get("id_mode")
            .or_else(|| obj.get("mode"))
            .and_then(JsonValue::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("none"));
        if aux_prior.is_none()
            && dim_selection.is_none()
            && aux_outcome.is_none()
            && !explicit_none_mode
        {
            return Err(format!(
                "latents['{key}'] requires aux_prior or aux_outcome for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied"
            ));
        }
        specs.push(LatentSpec {
            target,
            n,
            d,
            init,
            manifold,
            retraction_registry,
            aux_prior,
            dim_selection,
            aux_outcome,
            explicit_none_mode,
        });
    }
    Ok(specs)
}

fn deterministic_unit(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn initial_latent_matrix(spec: &LatentSpec, y: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    match &spec.init {
        LatentInitSpec::Explicit(matrix) => {
            if matrix.nrows() != spec.n || matrix.ncols() != spec.d {
                return Err(format!(
                    "latent '{}' explicit init has shape {}x{}, expected {}x{}",
                    spec.target,
                    matrix.nrows(),
                    matrix.ncols(),
                    spec.n,
                    spec.d
                ));
            }
            Ok(matrix.clone())
        }
        LatentInitSpec::Random => {
            let mut seed = 0x9E3779B97F4A7C15_u64 ^ ((spec.n as u64) << 32) ^ spec.d as u64;
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            for value in out.iter_mut() {
                *value = deterministic_unit(&mut seed);
            }
            Ok(out)
        }
        LatentInitSpec::Pca => {
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            let mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            let var = y
                .iter()
                .map(|v| {
                    let centered = *v - mean;
                    centered * centered
                })
                .sum::<f64>()
                / y.len().max(1) as f64;
            let sd = var.sqrt().max(1e-12);
            for n in 0..spec.n {
                out[[n, 0]] = (y[n] - mean) / sd;
            }
            if spec.d > 1 {
                let mut seed = 0xD1B54A32D192ED03_u64 ^ ((spec.n as u64) << 16) ^ spec.d as u64;
                for n in 0..spec.n {
                    for axis in 1..spec.d {
                        out[[n, axis]] = deterministic_unit(&mut seed) - 0.5;
                    }
                }
            }
            Ok(out)
        }
    }
}

fn latent_id_mode(spec: &LatentSpec) -> Result<LatentIdMode, String> {
    if let Some(ao) = spec.aux_outcome.as_ref() {
        use crate::terms::behavioral_head::BehavioralHead;
        if let Some(init) = ao.init_log_precision.as_ref()
            && init.len() != spec.d
        {
            return Err(format!(
                "latent '{}' aux_outcome.init_log_precision has length {}, expected {}",
                spec.target,
                init.len(),
                spec.d
            ));
        }
        let head = match ao.row_weights.as_ref() {
            Some(w) => BehavioralHead::new(ao.family, ao.y.clone(), w.clone()),
            None => BehavioralHead::fully_supervised(ao.family, ao.y.clone()),
        }
        .map_err(|e| format!("latent '{}' aux_outcome head: {e}", spec.target))?;
        return Ok(LatentIdMode::AuxOutcome {
            head,
            init_log_precision: ao.init_log_precision.clone(),
        });
    }
    match (&spec.aux_prior, &spec.dim_selection) {
        (Some(aux), Some(dim)) => {
            if let Some(init) = dim.init_log_precision.as_ref()
                && init.len() != spec.d
            {
                return Err(format!(
                    "latent '{}' dim_selection.init_log_precision has length {}, expected {}",
                    spec.target,
                    init.len(),
                    spec.d
                ));
            }
            Ok(LatentIdMode::AuxPriorDimSelection {
                u: aux.u.clone(),
                family: aux.family,
                strength: aux.strength,
                init_log_precision: dim.init_log_precision.clone(),
            })
        }
        (Some(aux), None) => Ok(LatentIdMode::AuxPrior {
            u: aux.u.clone(),
            family: aux.family,
            strength: aux.strength,
        }),
        (None, None) if spec.explicit_none_mode => Ok(LatentIdMode::None),
        (None, None) => Err(format!(
            "latent '{}' requires aux_prior for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied",
            spec.target
        )),
        (None, Some(_)) => Err(format!(
            "latent '{}' dim_selection requires aux_prior for identifiability",
            spec.target
        )),
    }
}

fn prepare_standard_latent_coord(
    parsed: &ParsedFormula,
    data: &Dataset,
    y: ArrayView1<'_, f64>,
    config: &FitConfig,
) -> Result<Option<(Dataset, ParsedFormula, StandardLatentCoordConfig)>, String> {
    let specs = parse_latent_specs(config.latents.as_ref())?;
    let analytic_penalties = descriptors::build_analytic_penalty_registry_from_descriptors(
        config.latents.as_ref(),
        config.analytic_penalties.as_ref(),
    )?;
    if config.topology_auto_selector.is_some() && specs.is_empty() {
        return Err(
            "TopologyAutoSelector requires a Smooth with latent coords; pass latents={...}"
                .to_string(),
        );
    }
    if specs.is_empty() {
        return Ok(None);
    }
    if specs.len() != 1 {
        return Err(
            "standard latent-coordinate REML currently accepts exactly one latent smooth term"
                .to_string(),
        );
    }
    let spec = specs.into_iter().next().unwrap();
    if let Some(selector) = config.topology_auto_selector.as_ref()
        && let Some(requested) = selector.latent.as_ref()
        && requested != &spec.target
    {
        return Err(format!(
            "TopologyAutoSelector requested latent {requested:?}, but the formula path materialized latent {:?}",
            spec.target
        ));
    }
    if spec.n != data.values.nrows() || spec.n != y.len() {
        return Err(format!(
            "latent '{}' row count {} does not match data rows {}",
            spec.target,
            spec.n,
            data.values.nrows()
        ));
    }
    if let Some(aux) = spec.aux_prior.as_ref()
        && aux.u.nrows() != spec.n
    {
        return Err(format!(
            "latent '{}' aux_prior.u has {} rows, expected {}",
            spec.target,
            aux.u.nrows(),
            spec.n
        ));
    }

    let matrix = initial_latent_matrix(&spec, y)?;
    let id_mode = latent_id_mode(&spec)?;
    let latent_values = Arc::new(LatentCoordValues::from_matrix_with_manifold_and_retraction(
        matrix.view(),
        id_mode,
        spec.manifold.manifold.clone(),
        spec.retraction_registry.clone(),
    ));

    let base_cols = data.values.ncols();
    let mut values = Array2::<f64>::zeros((data.values.nrows(), base_cols + spec.d));
    values.slice_mut(s![.., ..base_cols]).assign(&data.values);
    let mut headers = data.headers.clone();
    let mut columns = data.schema.columns.clone();
    let mut column_kinds = data.column_kinds.clone();
    let mut synthetic_vars = Vec::with_capacity(spec.d);
    let mut feature_cols = Vec::with_capacity(spec.d);
    for axis in 0..spec.d {
        let name = format!("{}__latent{}", spec.target, axis);
        let col = base_cols + axis;
        values.column_mut(col).assign(&matrix.column(axis));
        headers.push(name.clone());
        columns.push(SchemaColumn {
            name: name.clone(),
            kind: ColumnKindTag::Continuous,
            levels: Vec::new(),
        });
        column_kinds.push(ColumnKindTag::Continuous);
        synthetic_vars.push(name);
        feature_cols.push(col);
    }
    let augmented = Dataset {
        headers,
        values,
        schema: DataSchema { columns },
        column_kinds,
    };

    let mut rewritten = parsed.clone();
    let mut matched = false;
    for term in &mut rewritten.terms {
        if let ParsedTerm::Smooth { vars, .. } = term
            && vars.len() == 1
            && vars[0] == spec.target
        {
            *vars = synthetic_vars.clone();
            matched = true;
        }
    }
    if !matched {
        return Err(format!(
            "latents provided '{}' but no formula smooth term s({}, ...) was found",
            spec.target, spec.target
        ));
    }

    Ok(Some((
        augmented,
        rewritten,
        StandardLatentCoordConfig {
            values: latent_values,
            term_index: crate::types::SmoothTermIdx::placeholder(),
            feature_cols,
            manifold: spec.manifold.manifold,
            manifold_auto: spec.manifold.auto,
            retraction_registry: spec.retraction_registry,
            analytic_penalties: (!analytic_penalties.penalties.is_empty())
                .then(|| Arc::new(analytic_penalties)),
        },
    )))
}

fn smooth_basis_feature_cols_for_latent(
    basis: &crate::smooth::SmoothBasisSpec,
) -> Option<Vec<usize>> {
    match basis {
        crate::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(vec![*feature_col]),
        crate::smooth::SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Sphere { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Matern { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Duchon { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Pca { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            Some(feature_cols.clone())
        }
        crate::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_basis_feature_cols_for_latent(smooth)
        }
        crate::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | crate::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_feature_cols_for_latent(inner)
        }
        crate::smooth::SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}

fn natural_latent_manifold_for_basis(
    basis: &crate::smooth::SmoothBasisSpec,
    d: usize,
) -> LatentManifold {
    match basis {
        crate::smooth::SmoothBasisSpec::BSpline1D { spec, .. } => {
            if let crate::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                &spec.knotspec
            {
                LatentManifold::Circle {
                    period: data_range.1 - data_range.0,
                }
            } else {
                LatentManifold::Euclidean
            }
        }
        crate::smooth::SmoothBasisSpec::Sphere { .. } => LatentManifold::Sphere { dim: d },
        crate::smooth::SmoothBasisSpec::Duchon { spec, .. }
            if spec.periodic.is_some() && d == 1 =>
        {
            let period = spec
                .periodic
                .as_ref()
                .and_then(|v| v.first().copied().flatten())
                .unwrap_or(std::f64::consts::TAU);
            LatentManifold::Circle { period }
        }
        crate::smooth::SmoothBasisSpec::TensorBSpline { spec, .. } => {
            let parts: Vec<LatentManifold> = spec
                .marginalspecs
                .iter()
                .map(|margin| {
                    if let crate::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                        &margin.knotspec
                    {
                        LatentManifold::Circle {
                            period: data_range.1 - data_range.0,
                        }
                    } else {
                        LatentManifold::Euclidean
                    }
                })
                .collect();
            if parts.iter().all(|part| part.is_euclidean()) {
                LatentManifold::Euclidean
            } else {
                LatentManifold::Product(parts)
            }
        }
        crate::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            natural_latent_manifold_for_basis(smooth, d)
        }
        crate::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | crate::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            natural_latent_manifold_for_basis(inner, d)
        }
        crate::smooth::SmoothBasisSpec::ThinPlate { .. }
        // ConstantCurvature: the chart coordinates are Euclidean-valued (any
        // finite point for κ ≥ 0; the latent optimizer's chart-validity is the
        // term's own concern), so the latent retraction stays Euclidean. A
        // κ-aware latent seed/retraction is part of the later ψ-channel stage.
        | crate::smooth::SmoothBasisSpec::ConstantCurvature { .. }
        | crate::smooth::SmoothBasisSpec::Matern { .. }
        | crate::smooth::SmoothBasisSpec::MeasureJet { .. }
        | crate::smooth::SmoothBasisSpec::Duchon { .. }
        | crate::smooth::SmoothBasisSpec::Pca { .. }
        | crate::smooth::SmoothBasisSpec::FactorSmooth { .. } => LatentManifold::Euclidean,
    }
}

fn materialize_standard<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if config.noise_offset_column.is_some() {
        return Err(
            "noise_offset_column requires a location-scale model with noise_formula"
                .to_string()
                .into(),
        );
    }
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation (#335 Gamma requires y > 0;
    // #337 Poisson/NegativeBinomial require y ≥ 0; mirrors the Beta
    // (0,1)-support check in the external-design GLM path). The family
    // itself owns the check — see `ResponseFamily::validate_response_support`
    // — so adding a new family that constrains its support is a single edit
    // on the type, not a coordinated update across every materializer.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331 all-0/all-1 Bernoulli,
    // #332 near-constant Gaussian). Symmetric to validate_response_support —
    // each `ResponseFamily` variant owns its own degeneracy classifier, the
    // workflow only forwards the column name.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let latent_prepared = prepare_standard_latent_coord(parsed, data, y.view(), config)?;
    let (latent_dataset, latent_parsed, mut latent_coord) = match latent_prepared {
        Some((dataset, parsed, coord)) => (Some(dataset), Some(parsed), Some(coord)),
        None => (None, None, None),
    };
    let term_data = latent_dataset.as_ref().unwrap_or(data);
    let term_parsed = latent_parsed.as_ref().unwrap_or(parsed);
    let term_col_map = term_data.column_map();

    let policy =
        resolved_resource_policy(config, term_data, crate::resource::ProblemHints::default());
    let spec = build_termspec_with_geometry_and_overrides(
        &term_parsed.terms,
        term_data,
        &term_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;

    // Sample size vs basis-rank gate (#309). Each smooth basis answers
    // `min_sample_rows()` for itself; this helper just sums and compares.
    // Runs *after* `build_termspec_with_geometry_and_overrides` so the lower bound is
    // computed on the fully resolved basis spec (e.g. tensor-product columns,
    // knot counts inferred at materialization time).
    check_smooth_capacity(&spec, y.len(), &parsed.response)?;
    if let Some(coord) = latent_coord.as_mut() {
        let resolved_idx = spec
            .smooth_terms
            .iter()
            .position(|term| {
                smooth_basis_feature_cols_for_latent(&term.basis)
                    .is_some_and(|cols| cols == coord.feature_cols)
            })
            .ok_or_else(|| {
                "latent-coordinate smooth term disappeared during formula materialization"
                    .to_string()
            })?;
        coord.term_index = crate::types::SmoothTermIdx::new(resolved_idx);
        if coord.manifold_auto {
            let inferred = natural_latent_manifold_for_basis(
                &spec.smooth_terms[coord.term_index.get()].basis,
                coord.feature_cols.len(),
            );
            coord.manifold = inferred.clone();
            coord.values = Arc::new(coord.values.with_manifold(inferred));
        }
    }

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let latent_cloglog = if family.is_latent_cloglog() {
        let sigma = match config.frailty.clone().unwrap_or(FrailtySpec::None) {
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: crate::families::lognormal_kernel::HazardLoading::Full,
            } => sigma,
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(_),
                loading,
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason: format!(
                        "latent-cloglog-binomial requires HazardLoading::Full, got {loading:?}"
                    ),
                }
                .into());
            }
            FrailtySpec::HazardMultiplier {
                sigma_fixed: None, ..
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial currently requires a fixed hazard-multiplier sigma"
                            .to_string(),
                }
                .into());
            }
            FrailtySpec::GaussianShift { .. } => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "latent-cloglog-binomial does not support GaussianShift frailty"
                        .to_string(),
                }
                .into());
            }
            FrailtySpec::None => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial requires config.frailty=HazardMultiplier with a fixed sigma"
                            .to_string(),
                }
                .into());
            }
        };
        Some(
            LatentCLogLogState::new(sigma)
                .map_err(|e| format!("invalid latent_cloglog state: {e}"))?,
        )
    } else {
        if config.frailty.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "config.frailty is not supported for standard family {:?}; use a frailty-aware family instead",
                    family
                ),
            }
            .into());
        }
        None
    };
    let options = FitOptions {
        latent_cloglog,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        // Formula/workflow fits are the interactive/default path. Keep
        // coefficient covariance and smoothing correction, but do not run the
        // optional live-rho posterior certificate/escalation here: escalation can
        // launch NUTS over rho and turns ordinary quality gates into sampler
        // benchmarks. Lower-level callers that explicitly need the rho posterior
        // can still opt in through `FitOptions`.
        skip_rho_posterior_inference: true,
        max_iter: config.outer_max_iter.unwrap_or(200),
        // Outer REML/LAML smoothing-selection tolerance. The outer convergence
        // test (`outer_gradient_tolerance`) uses a `rel_cost` criterion whose
        // effective projected-gradient threshold is ≈ `tol · (1 + |V(ρ)|)`. The
        // LAML cost grows like O(n), so at the old `1e-7` the effective gradient
        // threshold was ≈ `1e-7 · |V|` ≈ 1e-4 for a typical fit — far too coarse
        // to *resolve* the smoothing parameter: the descent halted while λ̂ could
        // still move several percent. That under-resolution is benign for a
        // single fit but breaks an exact invariance — for a fixed-dispersion
        // family a uniform prior weight `w = c` is exact `c`-fold replication, so
        // the two encodings share a byte-identical LAML surface and an identical
        // optimum, yet their (replication-equal) surfaces carry O(1e-7)
        // floating-point differences AWAY from the optimum. With the coarse
        // threshold the descent stopped at those encoding-dependent points,
        // systematically over-smoothing the weighted encoding (gam#893; up to a
        // ~22× λ ratio across seeds). Tightening to `1e-10` (effective gradient
        // threshold ≈ 1e-7, ~100× below the FP-noise floor) drives both
        // encodings to the shared optimum, restoring `w=c ⇔ c-fold replication`
        // in smoothing selection to optimiser precision. Max-iter is handled
        // best-effort (the optimiser returns its best ρ on budget exhaustion, it
        // does not hard-fail), so a harder problem that cannot reach 1e-10 in
        // `outer_max_iter` is no worse off than before — just better-resolved
        // when it can.
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: config.firth,
        adaptive_regularization: standard_adaptive_regularization_options(config),
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: config.persist_warm_start_disk,
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    let wiggle = effective_linkwiggle.as_ref().and_then(|cfg| {
        if !family.is_binomial() {
            return None;
        }
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                // linkwiggle is gated by `linkname_supports_joint_wiggle` which
                // rejects Sas / BetaLogistic upstream, so reaching this arm
                // means the gate was bypassed.
                Err(_) => return None,
            },
            None => {
                if let Some(state) = latent_cloglog {
                    InverseLink::LatentCLogLog(state)
                } else {
                    InverseLink::Standard(StandardLink::Logit)
                }
            }
        };
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders.clone(),
                double_penalty: cfg.double_penalty,
            },
            // The second-stage refit options live inside the wiggle config so
            // the pilot can't be configured without them (see
            // `StandardBinomialWiggleConfig` doc + #320). Magic-by-default:
            // no caller-supplied options are required for the Python /
            // formula-DSL path.
            refit_options: BlockwiseFitOptions::default(),
        })
    });

    Ok(MaterializedModel {
        request: FitRequest::Standard(StandardFitRequest {
            data: term_data.values.clone(),
            y,
            weights,
            offset,
            spec,
            family,
            options,
            kappa_options,
            wiggle,
            coefficient_groups: config.coefficient_groups.clone(),
            penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
            latent_coord,
            _marker: std::marker::PhantomData,
        }),
        inference_notes,
    })
}

fn materialize_bernoulli_marginal_slope<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();

    if !is_binary_response(y.view()) {
        return Err(WorkflowError::SchemaMismatch {
            reason: "Bernoulli marginal-slope requires a binary {0,1} response".to_string(),
        }
        .into());
    }
    if config.noise_formula.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope cannot also use noise_formula".to_string(),
        }
        .into());
    }

    let logslope_formula = config
        .logslope_formula
        .as_deref()
        .ok_or_else(|| "Bernoulli marginal-slope requires logslope_formula".to_string())?;
    // `z_column` is OPTIONAL when a CTN Stage-1 recipe is present: the calibrated
    // chain produces `z` out-of-fold from the cross-fitted CTN, so there is no
    // raw dose column to read (and no throwaway pre-fit column — that round-trip
    // is what the no-slop cutover removes, #461). Without a recipe, the primitive
    // standalone marginal-slope still requires a raw `z_column` dose.
    let z_column = config.z_column.as_deref();
    if z_column.is_none() && config.ctn_stage1.is_none() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope requires z_column (or a CTN Stage-1 recipe via \
                     ctn_stage1, which produces z by cross-fitting)"
                .to_string(),
        });
    }

    let (_, parsed_logslope) =
        parse_matching_auxiliary_formula(logslope_formula, &parsed.response, "logslope_formula")?;
    if parsed_logslope.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported inside logslope_formula".to_string(),
        }
        .into());
    }
    if let Some(z_column) = z_column {
        validate_marginal_slope_z_column_exclusion(
            parsed,
            &parsed_logslope,
            z_column,
            "Bernoulli marginal-slope",
            "logslope_formula",
        )?;
    }

    let mut inference_notes = Vec::new();
    // Bernoulli marginal-slope: structurally operator-only at large scale, so
    // flip the hint regardless of n to keep dense fallbacks blocked.
    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            marginal_slope_large_scale_active: true,
        },
    );
    // Alias `z` to the dose column only when a raw z_column is supplied; with a
    // CTN Stage-1 chain there is no dose column and the formulas reference only
    // the x covariates.
    let aliased_col_map = match z_column {
        Some(z_column) => column_map_with_alias(col_map, "z", z_column),
        None => col_map.clone(),
    };
    let mut marginalspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut marginalspec,
        data,
        "bernoulli marginal-slope marginal formula",
        &mut inference_notes,
    )?;
    let mut logslopespec = build_termspec_with_geometry_and_overrides(
        &parsed_logslope.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut logslopespec,
        data,
        "bernoulli marginal-slope logslope_formula",
        &mut inference_notes,
    )?;
    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let marginal_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let logslope_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let routing = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
    )?;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when a CTN
    // Stage-1 recipe is present (design §5). Cross-fitting yields out-of-fold `z`
    // (the calibrated dose, with no raw column read) and the score-influence
    // Jacobian `J`, absorbed by Stage-2 as the realized leakage-projection block.
    // With no CTN Stage-1 recipe, `z` is the raw dose column and the free-warp
    // `score_warp` is the fallback basis.
    let (z, score_influence_jacobian) =
        match crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
        {
            Some(calibration) => (calibration.z_oof, Some(calibration.jac_oof)),
            None => {
                // No recipe ⇒ a raw z_column is required (guarded above) and read here.
                let z_column = z_column.expect("z_column presence checked when ctn_stage1 is None");
                let z_idx = resolve_role_col(col_map, z_column, "z")?;
                let z = data.values.column(z_idx).to_owned();
                validate_bernoulli_marginal_slope_z_column_variance(
                    z_column,
                    z.view(),
                    weights.view(),
                )?;
                (z, None)
            }
        };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
        score_warp: routing.score_warp,
        link_dev: routing.link_dev,
        latent_z_policy: Default::default(),
        score_influence_jacobian,
    };

    Ok(MaterializedModel {
        request: FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
            data: data.values.view(),
            spec,
            options: BlockwiseFitOptions {
                compute_covariance: true,
                // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                // default for bernoulli marginal-slope — no flag to thread.
                ..Default::default()
            },
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            policy,
        }),
        inference_notes,
    })
}

fn materialize_survival<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
    entry_col: Option<&str>,
    exit_col: &str,
    event_col: &str,
    interval_right_col: Option<&str>,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let mut inference_notes = Vec::new();

    // Extract columns. `entry_col == None` is the right-censored shorthand
    // `Surv(time, event)`: every subject enters at time zero, so we
    // synthesize a constant-zero entry vector instead of resolving a column.
    let entry_idx = entry_col
        .map(|name| resolve_role_col(col_map, name, "entry"))
        .transpose()?;
    let exit_idx = resolve_role_col(col_map, exit_col, "exit")?;
    let event_idx = resolve_role_col(col_map, event_col, "event")?;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = data.values.nrows();
    let event = data.values.column(event_idx).to_owned();
    let event_codes = Array1::from_iter(
        event
            .iter()
            .copied()
            .enumerate()
            .map(|(i, value)| crate::survival::survival_event_code_from_value(value, i))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let entry_val = entry_idx.map_or(0.0, |idx| data.values[[i, idx]]);
            normalize_survival_time_pair(entry_val, data.values[[i, exit_idx]], i)
        })
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (e, x)) in pairs.into_iter().enumerate() {
        age_entry[i] = e;
        age_exit[i] = x;
    }

    // Interval-censored `SurvInterval(L, R, event)`: `exit_col` carried the
    // LEFT boundary `L` (resolved into `age_exit` above), and `interval_right_col`
    // carries the RIGHT boundary `R`. The kernel's interval contribution
    // `log[S(L) − S(R)]` requires a finite `R ≥ L` per row (`event >= 0.5`); a
    // row with `event < 0.5` is right-censored at `L` (its `R` is ignored). We
    // resolve `age_right` here so the downstream latent time stack can evaluate
    // the baseline at `R`.
    let age_right = if let Some(right_col) = interval_right_col {
        let right_idx = resolve_role_col(col_map, right_col, "interval right")?;
        let mut right = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = data.values[[i, right_idx]];
            let is_bracketed = data.values[[i, event_idx]] >= 0.5;
            if is_bracketed {
                if !(r.is_finite()) || r < age_exit[i] {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "SurvInterval(L, R, event) requires a finite R >= L on bracketed rows (event >= 1); row {} has L={}, R={r}",
                            i + 1,
                            age_exit[i]
                        ),
                    });
                }
                right[i] = r;
            } else {
                // Right-censored row: R is unused by the likelihood. Pin it to L
                // so the (ignored) right channel stays well-defined and the
                // `age_exit <= age_right` time-basis invariant holds.
                right[i] = age_exit[i];
            }
        }
        Some(right)
    } else {
        None
    };

    let survival_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    if age_right.is_some() && survival_mode != SurvivalLikelihoodMode::Latent {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "interval-censored SurvInterval(L, R, event) is only defined for the latent \
                 hazard-window survival likelihood (its kernel carries the log[S(L) − S(R)] \
                 interval contribution); got survival_likelihood='{}'",
                config.survival_likelihood
            ),
        });
    }
    // Fail fast on all-censored (zero-event) survival data for every survival
    // likelihood (#789B / construction-time fittability split). With no row
    // marking a target event, the survival likelihood has no event score: the
    // hazard direction is unidentified and the inner/outer solve either spins
    // on a flat landscape (marginal-slope) or returns a numerically degenerate
    // fit (other modes). This is the single chokepoint every survival fit
    // dispatcher routes through (Surv(...) responses + all FitConfig survival
    // modes), so catching it here keeps every downstream constructor —
    // `WorkingModelSurvival`, the Royston-Parmar wrapper, the marginal-slope
    // builders — free to materialize models on censored fixtures (which the
    // engine's structural unit tests rely on) without losing the user-facing
    // safety on real fits.
    if !event_codes.iter().any(|&code| code > 0) {
        let mode_label = match survival_mode {
            SurvivalLikelihoodMode::MarginalSlope => "survival marginal-slope",
            _ => "survival fit",
        };
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "{mode_label} requires at least one target event; all rows are censored, so the likelihood has no event score and cannot identify the hazard"
            ),
        });
    }
    let cause_count =
        crate::survival::cause_count_from_event_codes(event_codes.view()).into_workflow_result()?;
    if cause_count > 1
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "cause-specific competing risks with {cause_count} causes are currently supported for survival_likelihood='transformation' and 'weibull'; got '{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkwiggle.is_some()
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linkwiggle(...) is not defined for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkspec.is_some()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "link(...) is not implemented for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    // Hoist the survival marginal-slope z-column exclusion check above the
    // time-basis / termspec construction below.  Those downstream steps fail
    // fast on small or tightly-spaced time data (e.g. an I-spline of degree 3
    // cannot be supported by a 2-row fixture), which would otherwise swallow
    // the z-column misuse error and surface a knot-count error instead.
    // Checking here keeps the user-visible error tied to the actual config
    // problem the caller can fix (rename `z` or remove the alias) rather than
    // to an unrelated basis-shape failure further downstream.
    if matches!(survival_mode, SurvivalLikelihoodMode::MarginalSlope)
        && let Some(z_column) = config.z_column.as_deref()
    {
        let logslope_parsed_for_check = match config.logslope_formula.as_deref() {
            Some(ls_formula) => Some(
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?
                    .1,
            ),
            None => None,
        };
        let logslope_ref = logslope_parsed_for_check.as_ref().unwrap_or(parsed);
        validate_marginal_slope_z_column_exclusion(
            parsed,
            logslope_ref,
            z_column,
            "survival marginal-slope",
            "logslope_formula",
        )?;
    }
    let effective_timewiggle = parsed.timewiggle.clone();
    let baseline_target_raw = match survival_mode {
        SurvivalLikelihoodMode::Weibull if effective_timewiggle.is_some() => "weibull",
        SurvivalLikelihoodMode::Weibull => "linear",
        _ => &config.baseline_target,
    };
    let baseline_cfg = initial_survival_baseline_config_for_fit(
        baseline_target_raw,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent hazard-window families require a non-linear scalar baseline target; use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }
    let time_cfg = if effective_timewiggle.is_some() {
        // Match the CLI path: the parametric baseline plus timewiggle supplies
        // the time structure, so the base time basis is disabled.
        SurvivalTimeBasisConfig::None
    } else if survival_mode == SurvivalLikelihoodMode::Weibull {
        SurvivalTimeBasisConfig::Linear
    } else {
        parse_survival_time_basis_config(
            &config.time_basis,
            config.time_degree,
            config.time_num_internal_knots,
            config.time_smooth_lambda,
        )?
    };
    // Marginal-slope centers the baseline-hazard I-spline at a robust interior
    // exit-scale time (median exit) rather than the earliest entry age: under
    // left truncation the earliest entry is a positive left-tail point and
    // centering there inflates the unpenalized linear-trend column, blowing up
    // the time-block seed score so REML rejects every seed (issue #751).
    // Location-scale keeps the earliest-entry anchor.
    let time_anchor = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, None)?
    } else {
        resolve_survival_time_anchor_value(&age_entry, None)?
    };
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(survival_mode);

    // Build time basis
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg.clone(),
        Some((config.time_num_internal_knots, config.time_smooth_lambda)),
    )?;
    if survival_mode != SurvivalLikelihoodMode::Weibull && effective_timewiggle.is_none() {
        require_structural_survival_time_basis(&time_build.basisname, "workflow survival fitting")?;
    }
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    // Interval-censored data needs the SAME monotone time basis evaluated at the
    // RIGHT boundary `R` (so `q_right = X_time(R)·β_time + offset_right`). Rebuild
    // it from the FROZEN knots (`resolved_time_cfg`, carrying the knot vector the
    // exit basis just inferred) at `age_right` in the exit slot — no knot drift —
    // and anchor-center its exit design identically. The resulting `x_exit_time`
    // row is exactly the design at `R`. `time_build_right.x_entry_time` /
    // `x_derivative_time` are unused by the interval-right channel.
    let time_build_right = if let Some(age_right) = age_right.as_ref() {
        let mut build_right = build_survival_time_basis(
            &age_entry,
            age_right,
            resolved_time_cfg.clone(),
            Some((config.time_num_internal_knots, config.time_smooth_lambda)),
        )?;
        center_survival_time_designs_at_anchor(
            &mut build_right.x_entry_time,
            &mut build_right.x_exit_time,
            &time_anchor_row,
        )?;
        Some(build_right)
    } else {
        None
    };
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }

    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            // Survival marginal-slope shares the operator-only invariant with
            // the Bernoulli path; flag it as such so strict mode is selected
            // even at small n.
            marginal_slope_large_scale_active: survival_mode
                == SurvivalLikelihoodMode::MarginalSlope,
        },
    );
    // Alias `z` to the dose column for the marginal termspec only when a raw
    // z_column is supplied. With a CTN Stage-1 recipe there is no dose column
    // (z is produced out-of-fold by cross-fitting) and the marginal formula
    // references only the x covariates, so no alias is needed.
    let marginal_slope_aliased_col_map = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(z_column) => Some(column_map_with_alias(col_map, "z", z_column)),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let termspec_col_map = marginal_slope_aliased_col_map.as_ref().unwrap_or(col_map);
    let mut termspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        termspec_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        prune_unidentified_linear_terms_for_marginal_slope(
            &mut termspec,
            data,
            "survival marginal-slope marginal formula",
            &mut inference_notes,
        )?;
    }

    let residual_dist = parse_survival_distribution(&config.survival_distribution)?;
    let survival_inverse_link = residual_distribution_inverse_link(residual_dist);
    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());
    let effective_linkwiggle_cfg = effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let threshold_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let threshold_template = if let Some(k) = config.threshold_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.threshold_time_degree,
            "threshold",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigma_template = if let Some(k) = config.sigma_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.sigma_time_degree,
            "sigma",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigmaspec = if let Some(noise) = config.noise_formula.as_deref() {
        let mut noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());
        // Use the same aliased col_map as the main termspec — survival
        // marginal-slope reserves `z` as a placeholder for `--z-column`,
        // and the logslope/noise formula may reference it too.
        build_termspec_with_geometry_and_overrides(
            &noise_parsed.terms,
            data,
            termspec_col_map,
            &mut inference_notes,
            config.scale_dimensions,
            &policy,
            config.smooth_overrides.as_ref(),
        )?
    } else {
        // No `noise_formula` ⇒ default to an empty log-σ spec for every
        // survival likelihood (constant log-σ baseline owned by the family
        // adapter). The previous `LocationScale`-only branch cloned the
        // mean `termspec` here, which duplicated every threshold term onto
        // the log-σ block. For a smooth `s(x)` on the mean that was
        // structurally fatal: the canonical-gauge identifiability audit
        // saw the log-σ block as exact-aliased to threshold and (per the
        // descending priorities time=200 > threshold=150 > log_sigma=120,
        // issue #366) attributed/dropped every log-σ column, leaving the
        // solver's `ParameterBlockSpec` design at width 0 while the
        // family kept the un-audited `x_log_sigma` at the smooth's width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then errored "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        // The empty default routes through the same
        // `infer_non_intercept_start_design`/`design_column_tail`
        // contract every other mode uses (yielding a 0-column
        // `x_log_sigma` that matches the spec), so the family and spec
        // agree by construction.
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    // `z_column` is OPTIONAL for the survival marginal-slope when a CTN Stage-1
    // recipe is present: the calibrated chain produces the single `z` surface
    // out-of-fold from the cross-fitted CTN, so there is no raw dose column to
    // read (no throwaway pre-fit column — the no-slop cutover, #461). Without a
    // recipe, the primitive standalone survival marginal-slope still requires a
    // raw `z_column` dose.
    let marginal_z_column_name = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(name) => Some(name),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let (
        marginal_z,
        marginal_logslopespec,
        marginal_logslopespecs,
        marginal_slope_deviation_routing,
        marginal_slope_base_link,
    ) = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        let base_link = resolve_survival_marginal_slope_base_link(parsed.linkspec.as_ref())?;
        if marginal_z_column_name.is_none() {
            // Calibrated chain: the CTN Stage-1 recipe produces a SINGLE z surface
            // out-of-fold, so no dose column is read. Stand in an n×1 placeholder
            // surface (the cross-fit below overrides column 0) and build the
            // logslope surface from the formula (or the marginal termspec). The
            // single-surface invariant matches the cross-fit guard further down.
            let placeholder_z = Array2::<f64>::zeros((data.values.nrows(), 1));
            let (logslopespec, routing) = if let Some(ls_formula) =
                config.logslope_formula.as_deref()
            {
                let (_, ls_parsed) = parse_matching_auxiliary_formula(
                    ls_formula,
                    &parsed.response,
                    "logslope_formula",
                )?;
                if ls_parsed.linkspec.is_some() {
                    return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.timewiggle.is_some() {
                    return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.survivalspec.is_some() {
                    return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &ls_parsed.terms,
                    data,
                    col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                let routing = route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?;
                (spec, routing)
            } else {
                (
                    termspec.clone(),
                    route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                )
            };
            (
                Some(placeholder_z),
                Some(logslopespec.clone()),
                Some(vec![logslopespec]),
                routing,
                Some(base_link),
            )
        } else if let Some(ls_formula) = config.logslope_formula.as_deref() {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            validate_marginal_slope_z_column_exclusion(
                parsed,
                &ls_parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let surfaces = marginal_slope_logslope_surfaces(&ls_parsed, default_z_column)?;
            let mut z = Array2::<f64>::zeros((data.values.nrows(), surfaces.len()));
            let mut specs = Vec::with_capacity(surfaces.len());
            for (surface_idx, surface) in surfaces.iter().enumerate() {
                let z_idx = resolve_role_col(col_map, &surface.z_column, "z")?;
                z.column_mut(surface_idx).assign(&data.values.column(z_idx));
                let aliased_col_map = column_map_with_alias(col_map, "z", &surface.z_column);
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &surface.terms,
                    data,
                    &aliased_col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                specs.push(spec);
            }
            (
                Some(z),
                specs.first().cloned(),
                Some(specs),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?,
                Some(base_link),
            )
        } else {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            validate_marginal_slope_z_column_exclusion(
                parsed,
                parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let z_idx = resolve_role_col(col_map, default_z_column, "z")?;
            let z = data.values.column(z_idx).to_owned().insert_axis(Axis(1));
            (
                Some(z),
                Some(termspec.clone()),
                Some(vec![termspec.clone()]),
                route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                Some(base_link),
            )
        }
    } else {
        (
            None,
            None,
            None,
            MarginalSlopeDeviationRouting {
                score_warp: None,
                link_dev: None,
            },
            None,
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when the
    // survival marginal-slope `z` was generated by a CTN Stage-1 fit (design
    // §5). Computed once (it refits the CTN K times) — outside the per-baseline
    // request closure below. When active it replaces the (single) CTN-generated
    // z surface with its out-of-fold value and captures the score-influence
    // Jacobian `J` for Stage-2's leakage-projection block. With no CTN Stage-1
    // recipe, the raw z surfaces stand and `score_warp` is the fallback basis.
    let crossfit_calibration = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
    } else {
        None
    };
    let (marginal_z, marginal_slope_jac_oof) = match (marginal_z, crossfit_calibration) {
        (Some(mut z_surfaces), Some(calibration)) => {
            // A CTN Stage-1 chain produces exactly one latent score surface; the
            // OOF projection is defined against that single column.
            if z_surfaces.ncols() != 1 {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "cross-fitted score calibration applies to a single CTN-generated z \
                         surface, but the survival marginal-slope model has {} z surfaces; \
                         multi-surface logslope is incompatible with the CTN Stage-1 chain",
                        z_surfaces.ncols()
                    ),
                });
            }
            z_surfaces.column_mut(0).assign(&calibration.z_oof);
            (Some(z_surfaces), Some(calibration.jac_oof))
        }
        (z, _) => (z, None),
    };

    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes formula-level linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_score_warp.is_some() {
            inference_notes.push(
                "survival marginal-slope routes logslope_formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_link_dev.is_none() && marginal_slope_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }
    }
    let marginal_slope_frailty = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        Some(fixed_gaussian_shift_frailty_from_spec(
            config.frailty.as_ref().unwrap_or(&FrailtySpec::None),
            "survival marginal-slope",
        )?)
    } else {
        None
    };
    match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
            if config.frailty.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "frailty is not supported for transformation/weibull survival models"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::LocationScale if config.frailty.is_some() => {
            return Err(WorkflowError::InvalidConfig {
                reason: "config.frailty is not implemented for survival-likelihood=location-scale"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
            if effective_timewiggle.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "timewiggle is not implemented for latent survival/binary likelihoods"
                    .to_string(),
            }
            .into());
        }
        _ => {}
    }
    let latent_loading = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let frailty = config.frailty.as_ref().unwrap_or(&FrailtySpec::None);
        Some(latent_hazard_loading(
            frailty,
            "workflow latent survival/binary",
        )?)
    } else {
        None
    };

    let build_time_block =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                (survival_mode == SurvivalLikelihoodMode::LocationScale)
                    .then_some(&survival_inverse_link),
                time_anchor,
                exact_derivative_guard,
                &time_build,
                effective_timewiggle.as_ref(),
                None,
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let initial_beta = if survival_mode == SurvivalLikelihoodMode::LocationScale {
                None
            } else {
                Some(Array1::from_elem(time_p, 1e-4))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta,
            };
            Ok::<_, String>((prepared, time_block))
        };

    // Warm-start cache for the outer baseline-config optimization: each probe
    // runs a complete inner BFGS over ρ (log-smoothing) starting from zeros if cold; by
    // capturing the previous probe's converged ρ (threshold + log_sigma blocks) and
    // injecting it here, the next inner BFGS typically converges in 1-3 iterations
    // instead of ~10, cutting per-probe cost roughly 5-10× across the probes per fit.
    let location_scale_smoothing_warm_start: RefCell<Option<(Array1<f64>, Array1<f64>)>> =
        RefCell::new(None);
    let build_location_scale_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig,
         allow_inverse_link_optimization: bool| {
            let (prepared, time_block) = build_time_block(candidate)?;
            let (initial_threshold_log_lambdas, initial_log_sigma_log_lambdas) =
                match location_scale_smoothing_warm_start.borrow().as_ref() {
                    Some((thr, lsg)) => (Some(thr.clone()), Some(lsg.clone())),
                    None => (None, None),
                };
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                inverse_link: survival_inverse_link.clone(),
                derivative_guard: exact_derivative_guard,
                max_iter: 200,
                tol: 1e-7,
                time_block,
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block,
                linkwiggle_block: None,
                initial_threshold_log_lambdas,
                initial_log_sigma_log_lambdas,
                cache_session: None,
                cache_mirror_sessions: Vec::new(),
            };
            // During baseline-θ BFGS probes we hold the inverse-link state
            // fixed: otherwise every probe would trigger a nested
            // optimization over the SAS / BetaLogistic / Mixture link
            // parameters, defeating the BFGS speedup entirely. The final
            // fit (after baseline has converged) flips this back on, so
            // joint baseline + link optimization still happens — just
            // alternating instead of nested.
            let optimize_inverse_link = allow_inverse_link_optimization
                && survival_inverse_link_has_free_parameters(&spec.inverse_link);
            Ok::<_, String>(SurvivalLocationScaleFitRequest {
                data: data.values.view(),
                spec,
                wiggle: effective_linkwiggle_cfg.clone(),
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
                optimize_inverse_link,
                cache_session: None,
            })
        };

    let build_marginal_slope_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let (prepared, mut time_block) = build_time_block(candidate)?;
            time_block.time_monotonicity =
                crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByRowConstraint;
            Ok::<_, String>(SurvivalMarginalSlopeFitRequest {
                data: data.values.view(),
                spec: SurvivalMarginalSlopeTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.clone(),
                    weights: weights.clone(),
                    z: marginal_z.clone().ok_or_else(|| {
                        "marginal-slope survival requires z_column in FitConfig".to_string()
                    })?,
                    base_link: marginal_slope_base_link.clone().ok_or_else(|| {
                        "internal error: marginal-slope base link validation missing".to_string()
                    })?,
                    marginalspec: termspec.clone(),
                    marginal_offset: threshold_offset.clone(),
                    frailty: marginal_slope_frailty.clone().ok_or_else(|| {
                        "internal error: marginal-slope frailty validation missing".to_string()
                    })?,
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    timewiggle_block: prepared.timewiggle_block,
                    logslopespec: marginal_logslopespec.clone().ok_or_else(|| {
                        "marginal-slope survival is missing logslope spec".to_string()
                    })?,
                    logslopespecs: marginal_logslopespecs.clone(),
                    logslope_offset: log_sigma_offset.clone(),
                    score_warp: marginal_slope_score_warp.clone(),
                    link_dev: marginal_slope_link_dev.clone(),
                    latent_z_policy: Default::default(),
                    score_influence_jacobian: marginal_slope_jac_oof.clone(),
                },
                options: BlockwiseFitOptions {
                    compute_covariance: false,
                    // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                    // default for survival marginal-slope — no flag to thread.
                    ..Default::default()
                },
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            })
        };

    let build_latent_survival_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent survival loading missing after frailty validation"
                    .to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            // Interval-censored: build the matching time stack at the RIGHT
            // boundary `R` (the exit slot holds `age_right`, evaluated through the
            // frozen-knot `time_build_right`). Its exit channel is exactly the
            // `R`-evaluated design / offset / unloaded mass, which feed the
            // dedicated `_right` spec fields the kernel consumes for
            // `log[S(L) − S(R)]`. The `event_target` then marks bracketed rows
            // (`event >= 1`) with the `LATENT_SURVIVAL_EVENT_INTERVAL` sentinel
            // and leaves `event < 1` rows right-censored at `L`.
            let (time_design_right, time_offset_right, unloaded_mass_right, event_target) =
                if let (Some(age_right), Some(time_build_right)) =
                    (age_right.as_ref(), time_build_right.as_ref())
                {
                    let prepared_right = prepare_survival_time_stack(
                        &age_entry,
                        age_right,
                        candidate,
                        survival_mode,
                        None,
                        time_anchor,
                        exact_derivative_guard,
                        time_build_right,
                        None,
                        Some(loading),
                    )?;
                    if prepared_right.time_design_exit.ncols() != prepared.time_design_exit.ncols() {
                        return Err(format!(
                            "interval-censored right time design has {} columns but the left/exit design has {}; the right boundary basis must share the exit basis columns",
                            prepared_right.time_design_exit.ncols(),
                            prepared.time_design_exit.ncols()
                        ));
                    }
                    let event_target = event.mapv(|v| {
                        if v >= 0.5 {
                            crate::families::latent_survival::LATENT_SURVIVAL_EVENT_INTERVAL
                        } else {
                            0
                        }
                    });
                    (
                        Some(prepared_right.time_design_exit.clone()),
                        Some(prepared_right.eta_offset_exit.clone()),
                        prepared_right.unloaded_mass_exit.clone(),
                        event_target,
                    )
                } else {
                    (
                        None,
                        None,
                        Array1::zeros(0),
                        event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    )
                };
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentSurvivalFitRequest {
                data: data.values.view(),
                spec: LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target,
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    time_design_right,
                    time_offset_right,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    unloaded_mass_right,
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let build_latent_binary_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent binary loading missing after frailty validation".to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentBinaryFitRequest {
                data: data.values.view(),
                spec: LatentBinaryTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let baseline_cfg = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        baseline_cfg
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::MarginalSlope
    {
        optimize_survival_baseline_config_with_gradient(
            &baseline_cfg,
            "workflow survival marginal-slope baseline",
            |candidate| {
                let fit =
                    fit_survival_marginal_slope_model(build_marginal_slope_request(candidate)?)
                        .map_err(|e| format!("survival marginal-slope fit failed: {e}"))?;
                let gradient = marginal_slope_baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &fit.baseline_offset_residuals,
                )?
                .ok_or_else(|| {
                    "workflow survival marginal-slope baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                let hessian = marginal_slope_baseline_chain_rule_hessian(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &fit.baseline_offset_residuals,
                    &fit.baseline_offset_curvatures,
                )?
                .ok_or_else(|| {
                    "workflow survival marginal-slope baseline unexpectedly has no theta Hessian"
                        .to_string()
                })?;
                Ok((fit.fit.reml_score, gradient, hessian))
            },
        )?
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::LocationScale
    {
        // Analytic θ-gradient path. The baseline configuration enters the
        // location-scale fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t); at the converged β the
        // envelope theorem gives
        //
        //   d(NLL)/dθ_k = Σ_i r^(E)_i ∂o_E_i/∂θ_k
        //               + r^(X)_i ∂o_X_i/∂θ_k
        //               + r^(D)_i ∂o_D_i/∂θ_k
        //
        // where r^(*) are populated by
        // `SurvivalLocationScaleFamily::offset_channel_geometry` and the
        // partials by `baseline_offset_theta_partials`. When the inverse
        // link is probit/SAS/Mixture/etc., the location-scale family uses
        // the probit-channel baseline q(t) instead, so we contract against
        // `marginal_slope_baseline_offset_theta_partials` exactly as the
        // marginal-slope path does. BFGS w/ this analytic gradient
        // typically converges in ≲10 outer evaluations.
        let probit_channel =
            location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
        // Catch errors at the optimizer-call site so a single bad θ
        // candidate doesn't blow up the whole `gam.fit()` call. Specific
        // failure mode: when the inner ρ-ARC stalls on a near-flat REML
        // direction (smoothing param running to exp(20+) on an
        // under-identified covariate at small n), the subsequent inner
        // refit can produce a fit whose family methods see empty
        // `block_states` and crash with "expects 3 blocks, got 0". The
        // crash message originates from `validate_joint_states` and can
        // bubble up from any of ~9 callers in the survival family. Rather
        // than enumerating them all, catch the wrapper error here and
        // fall back to the seed baseline_cfg — the gradient path made no
        // progress, but the rest of the fit can proceed at the user's
        // initial GM (α, λ, γ).
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow survival location-scale baseline",
            |candidate| {
                let fit_result = fit_survival_location_scale_model(build_location_scale_request(
                    candidate, false,
                )?)
                .map_err(|e| format!("survival location-scale fit failed: {e}"))?;
                // Warm-start the next probe's threshold / log-σ smoothing parameters
                // at the converged values for this probe.
                let threshold_rho = fit_result.fit.fit.lambdas_threshold().mapv(f64::ln);
                let log_sigma_rho = fit_result.fit.fit.lambdas_log_sigma().mapv(f64::ln);
                *location_scale_smoothing_warm_start.borrow_mut() =
                    Some((threshold_rho, log_sigma_rho));
                let residuals = &fit_result.fit.baseline_offset_residuals;
                let gradient = if probit_channel {
                    marginal_slope_baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        residuals,
                    )?
                } else {
                    baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        residuals,
                    )?
                }
                .ok_or_else(|| {
                    "workflow survival location-scale baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                // The envelope-theorem residual contraction is the exact
                // θ-gradient of the *profile penalized NLL* −ℓ + ½βᵀSβ at
                // converged (β̂, ρ̂). Optimizing `reml_score` (which includes
                // ½ log|S_λ| − ½ log|H| LAML corrections) against this
                // gradient would mismatch the cost surface, because the
                // log-determinant terms have their own θ-dependence through
                // H(β̂, θ). Use the matching profile-NLL cost here; the final
                // model refit downstream still picks ρ via the full REML
                // surface at the converged baseline θ.
                let profile_cost = -fit_result.fit.fit.log_likelihood
                    + 0.5 * fit_result.fit.fit.stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(format!(
                        "workflow survival location-scale baseline: non-finite profile cost \
                         (log_likelihood={}, stable_penalty_term={}, cost={})",
                        fit_result.fit.fit.log_likelihood,
                        fit_result.fit.fit.stable_penalty_term,
                        profile_cost
                    ));
                }
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(e)
                if e.contains("expects 3 blocks, got 0")
                    || e.contains("expects 4 blocks, got 0")
                    || (e.contains("block_states") && e.contains("got 0"))
                    || e.contains("blockwise fit requires at least one block state")
                    || e.contains(SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER) =>
            {
                log::warn!(
                    "workflow survival location-scale baseline: gradient-only BFGS \
                     failed at an empty-block_states candidate ({e}); falling back \
                     to the seed baseline_cfg as-is"
                );
                baseline_cfg.clone()
            }
            Err(e) => return Err(e.into()),
        }
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        // Latent / LatentBinary baseline-θ. The baseline configuration enters
        // the inner latent fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t), so the envelope theorem at the
        // converged β̂ gives the exact θ-gradient of the *profile penalized NLL*
        //   V(θ) = −ℓ(β̂(θ)) + ½·β̂ᵀS β̂,
        //     dV/dθ_k = Σ_i Σ_ch r^ch_i ∂o^ch_i/∂θ_k,
        // with r^ch = LatentSurvivalFamily::offset_channel_residuals(β̂)
        // (`baseline_offset_residuals` on the fit result) contracted against
        // `baseline_offset_theta_partials` by `baseline_chain_rule_gradient`.
        // We optimize the profile-NLL — not the LAML `reml_score` whose
        // ½log|H+S_λ| term carries its own θ-dependence through H(β̂,θ) — and
        // the downstream final refit re-picks ρ on the full REML surface at the
        // converged baseline θ. BFGS converges in ≲10 outer evaluations.
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow latent survival baseline",
            |candidate| {
                let (log_likelihood, stable_penalty_term, residuals) = match survival_mode {
                    SurvivalLikelihoodMode::Latent => {
                        let request = build_latent_survival_request(candidate)?;
                        match fit_model(FitRequest::LatentSurvival(request)) {
                            Ok(FitResult::LatentSurvival(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err("internal latent survival workflow returned the wrong result variant".to_string());
                            }
                            Err(e) => return Err(format!("latent survival fit failed: {e}")),
                        }
                    }
                    SurvivalLikelihoodMode::LatentBinary => {
                        let request = build_latent_binary_request(candidate)?;
                        match fit_model(FitRequest::LatentBinary(request)) {
                            Ok(FitResult::LatentBinary(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err("internal latent binary workflow returned the wrong result variant".to_string());
                            }
                            Err(e) => return Err(format!("latent binary fit failed: {e}")),
                        }
                    }
                    SurvivalLikelihoodMode::Transformation
                    | SurvivalLikelihoodMode::Weibull
                    | SurvivalLikelihoodMode::LocationScale
                    | SurvivalLikelihoodMode::MarginalSlope => {
                        return Err(format!(
                            "internal: workflow latent baseline closure reached for non-latent mode {survival_mode:?}"
                        ));
                    }
                };
                let profile_cost = -log_likelihood + 0.5 * stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(format!(
                        "workflow latent baseline: non-finite profile cost \
                         (log_likelihood={log_likelihood}, \
                         stable_penalty_term={stable_penalty_term}, cost={profile_cost})"
                    ));
                }
                let gradient = baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "workflow latent baseline unexpectedly has no theta gradient".to_string()
                })?;
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(e)
                if e.contains("expects 3 blocks, got 0")
                    || e.contains("expects 4 blocks, got 0")
                    || (e.contains("block_states") && e.contains("got 0"))
                    || e.contains("blockwise fit requires at least one block state")
                    || e.contains(SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER) =>
            {
                log::warn!(
                    "workflow latent survival baseline: gradient-only BFGS failed at an \
                     empty-block_states candidate ({e}); falling back to the seed \
                     baseline_cfg as-is"
                );
                baseline_cfg.clone()
            }
            Err(e) => return Err(WorkflowError::InvalidConfig { reason: e }.into()),
        }
    } else {
        baseline_cfg
    };

    let request = match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
            if config.noise_offset_column.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason:
                        "noise_offset_column is supported only for survival location-scale or marginal-slope"
                            .to_string(),
                }
                .into());
            }
            let weibull_seed = if survival_mode == SurvivalLikelihoodMode::Weibull
                && effective_timewiggle.is_none()
            {
                let scale = config
                    .baseline_scale
                    .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
                let shape = config.baseline_shape.unwrap_or(1.0);
                if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                    return Err(WorkflowError::InvalidConfig {
                        reason:
                            "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                                .to_string(),
                    }
                    .into());
                }
                Some((scale, shape))
            } else {
                None
            };
            FitRequest::SurvivalTransformation(SurvivalTransformationFitRequest {
                data: data.values.view(),
                spec: SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_codes.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: threshold_offset.clone(),
                    baseline_cfg,
                    likelihood_mode: survival_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    ridge_lambda: config.ridge_lambda,
                    penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
                },
                cache_session: None,
            })
        }
        SurvivalLikelihoodMode::LocationScale => {
            FitRequest::SurvivalLocationScale(build_location_scale_request(&baseline_cfg, true)?)
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            FitRequest::SurvivalMarginalSlope(build_marginal_slope_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::Latent => {
            FitRequest::LatentSurvival(build_latent_survival_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::LatentBinary => {
            FitRequest::LatentBinary(build_latent_binary_request(&baseline_cfg)?)
        }
    };

    Ok(MaterializedModel {
        request,
        inference_notes,
    })
}

fn materialize_transformation_normal<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if parsed.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported for the transformation-normal family".to_string(),
        }
        .into());
    }
    if parsed.linkwiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) is not supported for the transformation-normal family"
                .to_string(),
        }
        .into());
    }
    if config.noise_offset_column.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "noise_offset_column is not supported for transformation-normal models"
                .to_string(),
        }
        .into());
    }
    if config.frailty.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "frailty is not supported for transformation-normal models".to_string(),
        }
        .into());
    }

    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let policy = resolved_resource_policy(config, data, marginal_slope_hints(config));
    let covariate_spec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;

    Ok(MaterializedModel {
        request: FitRequest::TransformationNormal(TransformationNormalFitRequest {
            data: data.values.view(),
            response: y,
            weights,
            offset,
            covariate_spec,
            config: TransformationNormalConfig::default(),
            options: BlockwiseFitOptions::default(),
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            warm_start: None,
        }),
        inference_notes,
    })
}

/// Apply basis parsimony to a *secondary* (distributional) predictor's smooths.
///
/// In a location-scale / GAMLSS fit the mean is identified directly by the
/// response and warrants the generous default basis, but the scale (log-σ) and
/// other distributional predictors are identified only through (noisy) squared
/// residuals. Handing their radial spatial smooths a basis sized for the mean
/// lets REML over-fit them (#501). For each spatial smooth (thin-plate /
/// Matérn / Duchon) the user did not size explicitly, cap the *default* center
/// count via the private [`SECONDARY_CENTER_CAP_OPTION`]. The cap lowers the
/// default while preserving the `Auto` center strategy, so the basis is still
/// softly reduced when the data can't support the count (rather than erroring
/// like an explicit count would). Smooths the user sized explicitly, and the
/// non-radial bases (B-spline, cyclic, tensor) which already default modestly
/// via knot counts, are deliberately left untouched.
fn apply_secondary_predictor_basis_parsimony(terms: &mut [ParsedTerm], n_rows: usize) {
    for term in terms.iter_mut() {
        if let ParsedTerm::Smooth {
            vars,
            kind,
            options,
            ..
        } = term
        {
            let canonical = resolve_smooth_type_name(*kind, vars.len(), options);
            if !smooth_type_uses_spatial_center_heuristic(&canonical)
                || has_explicit_countwith_basis_alias(options, "centers")
            {
                continue;
            }
            let cap = crate::terms::basis::conservative_secondary_centers(n_rows, vars.len());
            options.insert(SECONDARY_CENTER_CAP_OPTION.to_string(), cap.to_string());
        }
    }
}

fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    let mut noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;
    apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation, owned by the family type.
    // See `ResponseFamily::validate_response_support`.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331, #332). The
    // location-scale path has its own σ-model so a near-constant Gaussian
    // mean response is even more pathological here than in the standard
    // path; same typed check, same family-owned classifier.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let policy = resolved_resource_policy(config, data, crate::resource::ProblemHints::default());
    let meanspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    let log_sigmaspec = build_termspec_with_geometry_and_overrides(
        &noise_parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    // Sample size vs basis rank, summed across the mean and log-σ smooths
    // (#309). Both designs share the same n_rows.
    check_smooth_capacity(&meanspec, y.len(), &parsed.response)?;
    check_smooth_capacity(&log_sigmaspec, y.len(), &parsed.response)?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let mean_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let options = BlockwiseFitOptions::default();

    let wiggle_cfg = effective_linkwiggle.map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    if family.is_latent_cloglog() {
        return Err(WorkflowError::InvalidConfig {
            reason: "latent-cloglog-binomial is not implemented for location-scale fitting"
                .to_string(),
        }
        .into());
    }

    if family.is_binomial() {
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                Err(e) => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "binomial location-scale fitting cannot route link `{}` through `InverseLink::Standard`: {e}",
                            c.link.name()
                        ),
                    }
                    .into());
                }
            },
            None => InverseLink::Standard(StandardLink::Logit),
        };
        Ok(MaterializedModel {
            request: FitRequest::BinomialLocationScale(BinomialLocationScaleFitRequest {
                data: data.values.view(),
                spec: BinomialLocationScaleTermSpec {
                    y,
                    weights,
                    link_kind,
                    thresholdspec: meanspec,
                    log_sigmaspec,
                    threshold_offset: mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    } else if let Some(kind) = dispersion_location_scale_kind(&family.response) {
        // Genuine-dispersion mean families (NegativeBinomial / Gamma / Beta /
        // Tweedie): `noise_formula` models the overdispersion channel (#913).
        // A link-wiggle is mean-only and not defined here.
        if wiggle_cfg.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "link-wiggle is not supported for {} location-scale models",
                    kind.family_tag()
                ),
            }
            .into());
        }
        Ok(MaterializedModel {
            request: FitRequest::DispersionLocationScale(DispersionLocationScaleFitRequest {
                data: data.values.view(),
                spec: DispersionGlmLocationScaleTermSpec {
                    kind,
                    y,
                    weights,
                    meanspec,
                    log_dispspec: log_sigmaspec,
                    mean_offset,
                    log_disp_offset: noise_offset,
                },
                options,
                kappa_options,
            }),
            inference_notes,
        })
    } else {
        Ok(MaterializedModel {
            request: FitRequest::GaussianLocationScale(GaussianLocationScaleFitRequest {
                data: data.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y,
                    weights,
                    meanspec,
                    log_sigmaspec,
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    }
}

/// Map a [`ResponseFamily`] to the dispersion-GAM kind whose overdispersion
/// channel can carry a `noise_formula` (#913), or `None` for families handled
/// by the Gaussian/Binomial location-scale paths.
fn dispersion_location_scale_kind(response: &ResponseFamily) -> Option<DispersionFamilyKind> {
    match response {
        ResponseFamily::NegativeBinomial { .. } => Some(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Some(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Some(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Some(DispersionFamilyKind::Tweedie { p: *p }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{DuchonNullspaceOrder, minimum_duchon_power_for_operator_penalties};
    use crate::inference::data::load_dataset_projected;
    use crate::inference::formula_dsl::{
        default_linkwiggle_formulaspec, parse_linkwiggle_formulaspec,
    };
    use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
    use crate::smooth::SmoothBasisSpec;
    use crate::solver::outer_strategy::{HessianSource, OuterPlan, OuterResult, Solver};
    use ndarray::Array2;
    use std::fs;
    use tempfile::tempdir;

    fn load_survival_dataset() -> crate::inference::data::EncodedDataset {
        let td = tempdir().expect("tempdir");
        let data_path = td.path().join("survival.csv");
        fs::write(
            &data_path,
            "entry,exit,event,x,z\n0.0,1.0,1,0.2,-0.4\n0.3,1.6,0,-0.1,0.6\n",
        )
        .expect("write survival csv");
        load_dataset_projected(
            &data_path,
            &[
                "entry".to_string(),
                "exit".to_string(),
                "event".to_string(),
                "x".to_string(),
                "z".to_string(),
            ],
        )
        .expect("load survival dataset")
    }

    #[test]
    fn competing_risks_baseline_seed_replicates_to_match_cause_specific_beta_length() {
        // Regression for #378's downstream break: the cause-specific assembly in
        // `fit_cause_specific_survival_transformation_custom` requires exactly
        // `p * cause_count` initial coefficients (it slices `cause * p..(cause +
        // 1) * p` per cause). The pooled baseline working model returns a
        // length-`p` seed, so without per-cause replication every `cause_count >
        // 1` fit aborts with a `SchemaMismatch` length mismatch. This pins that
        // the replication helper produces the exact length the assembly checks
        // for, and seeds each cause from the same pooled baseline.
        let pooled = Array1::from_vec(vec![-1.5_f64, 0.8, 0.0]);
        let p = pooled.len();

        for cause_count in [1usize, 2, 3] {
            let flat = replicate_pooled_baseline_seed_per_cause(pooled.view(), cause_count);
            // The exact invariant the cause-specific length guard enforces.
            assert_eq!(
                flat.len(),
                p * cause_count,
                "replicated seed must satisfy the `p * cause_count` length contract"
            );
            // Every per-cause slice must equal the shared pooled baseline seed.
            for cause in 0..cause_count {
                let slice = flat.slice(s![cause * p..(cause + 1) * p]);
                assert_eq!(
                    slice.to_owned(),
                    pooled,
                    "cause {cause} block must be seeded from the pooled baseline"
                );
            }
        }
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_main_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("main formula should reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("main formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_logslope_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1 + z".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x", &data, &config)
            .err()
            .expect("logslope formula should reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("logslope_formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_when_logslope_defaults_to_main_spec() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("defaulted logslope spec should still reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("main formula"));
    }

    #[test]
    fn survival_marginal_slope_matern_logslope_penalties_keep_surface_width() {
        let n = 24usize;
        let mut values = Array2::<f64>::zeros((n, 8));
        for i in 0..n {
            let u = i as f64 / (n - 1) as f64;
            values[[i, 0]] = 0.0;
            values[[i, 1]] = 0.25 + 8.0 * u;
            values[[i, 2]] = if i % 3 == 0 { 1.0 } else { 0.0 };
            values[[i, 3]] = ((i * 17 % 23) as f64 - 11.0) / 7.0;
            values[[i, 4]] = (2.0 * std::f64::consts::PI * u).sin();
            values[[i, 5]] = (2.0 * std::f64::consts::PI * u).cos();
            values[[i, 6]] = 2.0 * u - 1.0;
            values[[i, 7]] = if i % 2 == 0 { 0.0 } else { 1.0 };
        }
        let data = Dataset {
            headers: vec![
                "t0".to_string(),
                "t1".to_string(),
                "event".to_string(),
                "z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
                "sex".to_string(),
            ],
            values,
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "t0".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "t1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "sex".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
            ],
        };
        for (case, formula) in [
            (
                "with parametric sex term",
                "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6) + sex",
            ),
            (
                "without parametric sex term",
                "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6)",
            ),
        ] {
            let config = FitConfig {
                survival_likelihood: "marginal-slope".to_string(),
                logslope_formula: Some("matern(PC1, PC2, PC3, centers=6)".to_string()),
                z_column: Some("z".to_string()),
                ..FitConfig::default()
            };

            let materialized = materialize(formula, &data, &config).unwrap_or_else(|err| {
                panic!(
                    "survival marginal-slope materialization should keep block-local penalties \
                     {case}: {err}"
                )
            });
            let FitRequest::SurvivalMarginalSlope(request) = materialized.request else {
                panic!("expected survival marginal-slope request for {case}");
            };
            let specs = vec![
                request.spec.marginalspec.clone(),
                request.spec.logslopespec.clone(),
            ];
            let (designs, frozen_specs) =
                crate::smooth::build_term_collection_designs_and_freeze_joint(
                    data.values.view(),
                    &specs,
                )
                .unwrap_or_else(|err| {
                    panic!("joint freeze should preserve per-block penalty geometry {case}: {err}")
                });
            let (rebuilt, _) = crate::smooth::build_term_collection_designs_and_freeze_joint(
                data.values.view(),
                &frozen_specs,
            )
            .unwrap_or_else(|err| {
                panic!("frozen rebuild should preserve per-block penalty geometry {case}: {err}")
            });

            for (label, design) in [
                ("raw marginal", &designs[0]),
                ("raw logslope", &designs[1]),
                ("frozen marginal", &rebuilt[0]),
                ("frozen logslope", &rebuilt[1]),
            ] {
                let width = design.design.ncols();
                assert!(
                    width > 2,
                    "{case}: {label} design should be surface-width, not sex/intercept-width; \
                     width={width}"
                );
                for (idx, penalty) in design.penalties_as_penalty_matrix().iter().enumerate() {
                    assert_eq!(
                        penalty.shape(),
                        (width, width),
                        "{case}: {label} penalty {idx} must be block-local at the surface width"
                    );
                }
            }
        }
    }

    fn workflow_test_dataset() -> Dataset {
        Dataset {
            headers: vec![
                "age_entry".to_string(),
                "age_exit".to_string(),
                "event".to_string(),
                "bmi".to_string(),
                "z".to_string(),
            ],
            values: Array2::from_shape_vec(
                (4, 5),
                vec![
                    40.0, 43.0, 1.0, 22.0, -1.0, 41.0, 46.0, 0.0, 24.0, -0.2, 42.0, 47.0, 1.0,
                    27.0, 0.3, 44.0, 49.0, 0.0, 29.0, 1.2,
                ],
            )
            .expect("workflow test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "age_entry".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "age_exit".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "bmi".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        }
    }

    #[test]
    fn issue_789_transformation_normal_rejects_marginal_slope_controls_before_dispatch() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            transformation_normal: true,
            family: Some("bernoulli-marginal-slope".to_string()),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };

        let err = materialize("event ~ bmi", &data, &config)
            .err()
            .expect("transformation_normal must not steal marginal-slope fits");

        assert!(
            err.to_string()
                .contains("transformation_normal cannot be combined with marginal-slope")
        );
    }

    #[test]
    fn survival_marginal_slope_rejects_zero_event_data_before_fit() {
        let mut data = workflow_test_dataset();
        data.values.column_mut(2).fill(0.0);
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };

        let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
            .err()
            .expect("zero-event survival marginal-slope data must fail before optimization");

        assert!(err.to_string().contains("at least one target event"));
    }

    fn workflow_test_outer_result(converged: bool, rho: Array1<f64>) -> OuterResult {
        let mut result = OuterResult::new(
            rho,
            1.25,
            7,
            converged,
            OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        );
        result.final_grad_norm = Some(0.5);
        result
    }

    fn duchon_workflow_dataset() -> Dataset {
        let n = 72usize;
        let mut values = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            values[[i, 0]] = 0.5 * t.sin() + 0.15 * (3.0 * t).cos();
            values[[i, 1]] = t.cos();
            values[[i, 2]] = t.sin();
        }
        Dataset {
            headers: vec!["y".to_string(), "ct".to_string(), "st".to_string()],
            values,
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "y".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "ct".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "st".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        }
    }

    #[test]
    fn materialize_standard_keeps_adaptive_regularization_off_by_default_for_duchon() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12)",
            &data,
            &FitConfig::default(),
        )
        .expect("Duchon standard materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        assert!(request.options.adaptive_regularization.is_none());
    }

    #[test]
    fn materialize_standard_honors_adaptive_regularization_enable() {
        let data = duchon_workflow_dataset();
        let config = FitConfig {
            adaptive_regularization: Some(true),
            ..FitConfig::default()
        };
        let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
            .expect("Duchon materialization should allow enabling adaptive regularization");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let opts = request
            .options
            .adaptive_regularization
            .expect("Duchon should enable adaptive regularization when requested");
        assert!(opts.enabled);
    }

    #[test]
    fn materialize_standard_honors_adaptive_regularization_disable() {
        let data = duchon_workflow_dataset();
        let config = FitConfig {
            adaptive_regularization: Some(false),
            ..FitConfig::default()
        };
        let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
            .expect("Duchon materialization should allow disabling adaptive regularization");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        assert!(request.options.adaptive_regularization.is_none());
    }

    #[test]
    fn materialize_standard_duchon_defaults_to_pure_scale_free_basis() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12)",
            &data,
            &FitConfig::default(),
        )
        .expect("Duchon materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
            panic!("expected Duchon smooth");
        };
        assert_eq!(spec.length_scale, None);
        assert!(matches!(spec.nullspace_order, DuchonNullspaceOrder::Linear));
        assert_eq!(spec.power, 0.5);
    }

    #[test]
    fn materialize_standard_duchon_length_scale_opts_into_hybrid_basis() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12, length_scale=1.0)",
            &data,
            &FitConfig::default(),
        )
        .expect("hybrid Duchon materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
            panic!("expected Duchon smooth");
        };
        assert_eq!(spec.length_scale, Some(1.0));
        assert_eq!(spec.nullspace_order, DuchonNullspaceOrder::Linear);
        // The hybrid Matérn-blended kernel requires an INTEGER power. The cubic
        // structural default's fractional s=(d-1)/2 = 0.5 (d=2) is resolved at the
        // request layer to the smallest admissible integer (here s=0, the d=2
        // thin-plate order) rather than carried in as 0.5 and silently truncated
        // to 0 by the basis builder (#750). The pure path above still keeps 0.5.
        assert_eq!(spec.power, 0.0);
    }

    #[test]
    fn workflow_survival_marginal_slope_routes_logslope_linkwiggle_into_score_warp_only() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some(
                "1 + linkwiggle(degree=5, internal_knots=7, penalty_order=\"2,3\")".to_string(),
            ),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ s(bmi) + linkwiggle(degree=4, internal_knots=9, penalty_order=\"1\")",
            &data,
            &config,
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel {
            request,
            inference_notes,
        } = materialized;
        let FitRequest::SurvivalMarginalSlope(request) = request else {
            panic!("expected survival marginal-slope request");
        };

        let link_dev = request.spec.link_dev.expect("main-formula link-dev");
        let score_warp = request.spec.score_warp.expect("logslope score-warp");
        assert_eq!(link_dev.degree, 4);
        assert_eq!(link_dev.num_internal_knots, 9);
        assert_eq!(link_dev.penalty_order, 1);
        assert_eq!(link_dev.penalty_orders, vec![1]);
        assert_eq!(score_warp.degree, 5);
        assert_eq!(score_warp.num_internal_knots, 7);
        assert_eq!(score_warp.penalty_order, 3);
        assert_eq!(score_warp.penalty_orders, vec![2, 3]);
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("link-deviation block")),
            "workflow notes should mention main-formula linkwiggle routing"
        );
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("score-warp block")),
            "workflow notes should mention logslope_formula linkwiggle routing"
        );
    }

    #[test]
    fn materialize_routes_bernoulli_marginal_slope_when_logslope_and_z_are_set() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize("event ~ bmi", &data, &config)
            .expect("Bernoulli marginal-slope materialization should succeed");
        assert!(matches!(
            materialized.request,
            FitRequest::BernoulliMarginalSlope(_)
        ));
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_prunes_redundant_scalar_term() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "x".to_string(),
                "constant_spline_col".to_string(),
                "prs_z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
            ],
            values: Array2::from_shape_vec(
                (6, 7),
                vec![
                    0.0, -2.0, 1.0, -1.2, -1.0, 0.2, 0.7, 1.0, -1.0, 1.0, -0.4, -0.4, -0.3, 0.5,
                    0.0, 0.0, 1.0, 0.1, 0.1, 0.4, -0.2, 1.0, 1.0, 1.0, 0.5, 0.7, -0.6, 0.3, 0.0,
                    2.0, 1.0, 1.1, 1.2, 0.9, 0.0, 1.0, 3.0, 1.0, 1.7, 1.6, -0.8, -0.4,
                ],
            )
            .expect("BMS redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "event ~ matern(PC1, PC2, PC3, centers=3) + x + constant_spline_col",
            &data,
            &config,
        )
        .expect("BMS materialization should prune the redundant scalar term");
        let MaterializedModel {
            request,
            inference_notes,
        } = materialized;
        let FitRequest::BernoulliMarginalSlope(request) = request else {
            panic!("expected Bernoulli marginal-slope request");
        };
        let kept: Vec<&str> = request
            .spec
            .marginalspec
            .linear_terms
            .iter()
            .map(|term| term.name.as_str())
            .collect();
        assert_eq!(kept, vec!["x"]);
        assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
        assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("constant_spline_col")),
            "materialization should report the removed redundant scalar term; notes={inference_notes:?}"
        );
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_prunes_binary_outcome_style_scalar_alias() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "sex".to_string(),
                "entry_age_z".to_string(),
                "current_age_ns_1".to_string(),
                "current_age_ns_2".to_string(),
                "current_age_ns_3".to_string(),
                "current_age_ns_4".to_string(),
                "prs_z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
            ],
            values: Array2::from_shape_vec(
                (8, 11),
                vec![
                    0.0, 0.0, -1.4, 1.0, -0.6, 0.36, -0.216, -1.3, -1.0, 0.2, 0.7, 1.0, 1.0, -0.9,
                    1.0, -0.2, 0.04, -0.008, -0.8, -0.5, -0.3, 0.5, 0.0, 0.0, -0.5, 1.0, 0.1, 0.01,
                    0.001, -0.2, 0.1, 0.4, -0.2, 1.0, 1.0, -0.1, 1.0, 0.4, 0.16, 0.064, 0.3, 0.7,
                    -0.6, 0.3, 0.0, 0.0, 0.3, 1.0, 0.7, 0.49, 0.343, 0.8, 1.2, 0.9, 0.0, 1.0, 1.0,
                    0.7, 1.0, 1.0, 1.0, 1.0, 1.2, 1.6, -0.8, -0.4, 0.0, 0.0, 1.1, 1.0, 1.3, 1.69,
                    2.197, 1.6, -1.4, 0.8, -0.9, 1.0, 1.0, 1.5, 1.0, 1.6, 2.56, 4.096, 2.0, 0.3,
                    -1.1, 0.6,
                ],
            )
            .expect("binary-outcome-style BMS scalar-alias test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "sex".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "entry_age_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_4".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "event ~ matern(PC1, PC2, PC3, centers=3) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
            &data,
            &config,
        )
        .expect("BMS materialization should prune the local-column-3 scalar alias");
        let FitRequest::BernoulliMarginalSlope(request) = materialized.request else {
            panic!("expected Bernoulli marginal-slope request");
        };
        let kept: Vec<&str> = request
            .spec
            .marginalspec
            .linear_terms
            .iter()
            .map(|term| term.name.as_str())
            .collect();
        assert_eq!(
            kept,
            vec![
                "sex",
                "entry_age_z",
                "current_age_ns_2",
                "current_age_ns_3",
                "current_age_ns_4"
            ]
        );
        assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
        assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
        assert!(
            materialized
                .inference_notes
                .iter()
                .any(|note| note.contains("current_age_ns_1")),
            "materialization should report the removed binary-outcome-style scalar alias; notes={:?}",
            materialized.inference_notes
        );
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_rejects_constrained_redundant_scalar_term() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "x".to_string(),
                "constant_spline_col".to_string(),
                "prs_z".to_string(),
            ],
            values: Array2::from_shape_vec(
                (6, 4),
                vec![
                    0.0, -2.0, 1.0, -1.2, 1.0, -1.0, 1.0, -0.4, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 1.0,
                    0.5, 0.0, 2.0, 1.0, 1.1, 1.0, 3.0, 1.0, 1.7,
                ],
            )
            .expect("BMS constrained redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let err = match materialize(
            "event ~ x + linear(constant_spline_col, min=0.0)",
            &data,
            &config,
        ) {
            Ok(_) => panic!("constrained duplicate scalar term must be rejected, not pruned"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("constrained linear term 'constant_spline_col' is redundant"),
            "error should explain that the constrained duplicate scalar cannot be pruned: {msg}"
        );
    }

    #[test]
    fn bernoulli_marginal_slope_prune_rejects_penalized_redundant_scalar_term() {
        let data = Dataset {
            headers: vec!["event".to_string(), "constant_spline_col".to_string()],
            values: Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
                .expect("BMS penalized redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![ColumnKindTag::Binary, ColumnKindTag::Continuous],
        };
        let mut spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "constant_spline_col".to_string(),
                feature_col: 1,
                feature_cols: vec![1],
                double_penalty: true,
                coefficient_geometry: crate::smooth::LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let mut notes = Vec::new();
        let err = prune_unidentified_linear_terms_for_marginal_slope(
            &mut spec,
            &data,
            "test BMS formula",
            &mut notes,
        )
        .expect_err("explicitly penalized duplicate scalar term must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("explicitly penalized linear term 'constant_spline_col' is redundant"),
            "error should reject ridge-identification of duplicate scalar directions: {msg}"
        );
        assert_eq!(spec.linear_terms.len(), 1);
        assert!(notes.is_empty());
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_names_constant_z_column() {
        let data = Dataset {
            headers: vec!["event".to_string(), "bmi".to_string(), "prs_z".to_string()],
            values: Array2::from_shape_vec(
                (4, 3),
                vec![
                    0.0, 22.0, -0.58, 1.0, 24.0, -0.58, 0.0, 27.0, -0.58, 1.0, 29.0, -0.58,
                ],
            )
            .expect("constant z test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "bmi".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };

        let err = match materialize("event ~ bmi", &data, &config) {
            Ok(_) => panic!("constant z_column should be rejected before BMS integration"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("z_column 'prs_z' has zero weighted variance"),
            "error should name the constant z_column and diagnose weighted variance: {msg}"
        );
        assert!(
            msg.contains("all 4 values ~= -0.580000"),
            "error should summarize the observed constant value: {msg}"
        );
        assert!(
            msg.contains("weighted_sd=0.000000e0") && msg.contains("n=4"),
            "error should report weighted_sd and n: {msg}"
        );
        assert!(
            msg.contains(
                "bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score"
            ),
            "error should explain why the input is invalid: {msg}"
        );
        assert!(
            !msg.contains("requires z with positive finite weighted standard deviation"),
            "workflow should surface the input-style message instead of the generic BMS normalization error: {msg}"
        );
    }

    #[test]
    fn linkwiggle_defaults_are_consistent_across_formula_and_runtime() {
        let parsed = parse_linkwiggle_formulaspec(&Default::default(), "linkwiggle()")
            .expect("default linkwiggle should parse");
        let formula_default = default_linkwiggle_formulaspec();
        let runtime_default = DeviationBlockConfig::default();
        assert_eq!(parsed.degree, formula_default.degree);
        assert_eq!(
            parsed.num_internal_knots,
            formula_default.num_internal_knots
        );
        assert_eq!(parsed.penalty_orders, formula_default.penalty_orders);
        assert_eq!(parsed.double_penalty, formula_default.double_penalty);
        assert_eq!(runtime_default.degree, formula_default.degree);
        assert_eq!(
            runtime_default.num_internal_knots,
            formula_default.num_internal_knots
        );
        assert_eq!(
            runtime_default.penalty_orders,
            formula_default.penalty_orders
        );
        assert_eq!(
            runtime_default.double_penalty,
            formula_default.double_penalty
        );
    }

    #[test]
    fn survival_marginal_slope_accepts_explicit_probit_link() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let ok = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + link(type=probit)",
            &data,
            &config,
        );
        assert!(ok.is_ok(), "explicit probit should be accepted");

        let err = match materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + link(type=logit)",
            &data,
            &config,
        ) {
            Ok(_) => panic!("non-probit link should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("only link(type=probit)"));
    }

    #[test]
    fn high_dimensional_duchon_default_power_is_admissible() {
        let dim = 16;
        let power = minimum_duchon_power_for_operator_penalties(dim, DuchonNullspaceOrder::Zero, 2);
        assert!(2 * (1 + power) > dim + 2);
    }

    #[test]
    fn survival_location_scale_wiggle_rejects_unsupported_inverse_link() {
        let data = workflow_test_dataset();
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + linkwiggle(degree=4, internal_knots=3, penalty_order=\"1\")",
            &data,
            &FitConfig::default(),
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel { request, .. } = materialized;
        let FitRequest::SurvivalLocationScale(mut request) = request else {
            panic!("expected survival location-scale request");
        };
        request.spec.inverse_link = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: 0.1,
                initial_log_delta: 0.0,
            })
            .expect("valid SAS state"),
        );
        request.optimize_inverse_link = false;

        let err = match fit_survival_location_scale_model(request) {
            Ok(_) => panic!("survival link wiggle should reject unsupported inverse links"),
            Err(e) => e,
        };

        assert!(err.contains("survival link wiggle"));
        assert!(err.contains("does not support"));
    }

    #[test]
    fn survival_inverse_link_result_requires_convergence() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(false, Array1::from_vec(vec![0.1, -0.2])),
            "survival inverse-link optimization (SAS, dim=2)",
            |_| Some(InverseLink::Standard(StandardLink::Logit)),
        )
        .expect_err("non-converged inverse-link search should fail");

        assert!(err.contains("did not converge"));
        assert!(err.contains("final_objective"));
    }

    #[test]
    fn survival_inverse_link_result_requires_recoverable_state() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(true, Array1::from_vec(vec![9.0, 8.0])),
            "survival inverse-link optimization (mixture, dim=2)",
            |_| None,
        )
        .expect_err("unrecoverable inverse-link state should fail");

        assert!(err.contains("produced an invalid inverse-link state"));
        assert!(err.contains("9.0"));
    }

    // #371: survival-only / binomial-only DSL controls must be *rejected* in a
    // non-survival main formula, not parsed-and-silently-dropped. The bug was
    // that `parsed.timewiggle` / `parsed.survivalspec` are consumed only by
    // `materialize_survival`, and an explicit `linkwiggle(...)` is wired into
    // the fit only on the binomial arm, so a Gaussian formula carrying any of
    // these accepted the term and then ignored it — the user got an ordinary
    // GAM while believing they had configured a time-varying / wiggled model.

    #[test]
    fn timewiggle_rejected_in_nonsurvival_main_formula() {
        // `bmi` is a continuous response -> Gaussian standard path, no Surv(...).
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + timewiggle(internal_knots=4)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("timewiggle in a non-survival formula must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("timewiggle(...)") && msg.contains("survival"),
            "error should explain timewiggle is survival-only, got: {msg}"
        );
    }

    #[test]
    fn survmodel_rejected_in_nonsurvival_main_formula() {
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + survmodel(spec=net)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("survmodel in a non-survival formula must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("survmodel(...)") && msg.contains("survival"),
            "error should explain survmodel is survival-only, got: {msg}"
        );
    }

    #[test]
    fn linkwiggle_rejected_for_nonbinomial_response() {
        // `bmi` is continuous -> Gaussian; an explicit `linkwiggle(...)` corrects
        // a binomial link and would otherwise be dropped on the floor here.
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + linkwiggle(internal_knots=4)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("linkwiggle on a non-binomial response must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("linkwiggle(...)") && msg.contains("binomial"),
            "error should explain linkwiggle is binomial-only, got: {msg}"
        );
    }

    #[test]
    fn timewiggle_still_accepted_in_survival_formula() {
        // Guard must not regress the legitimate survival path: a Surv(...)
        // response still consumes timewiggle(...) without hitting the
        // non-survival rejection. We assert it does not error with the
        // non-survival "only supported in the main survival formula" message.
        let data = load_survival_dataset();
        let result = materialize(
            "Surv(entry, exit, event) ~ x + timewiggle(internal_knots=2)",
            &data,
            &FitConfig::default(),
        );
        if let Err(err) = result {
            let msg = err.to_string();
            assert!(
                !(msg.contains("timewiggle(...)") && msg.contains("meaningless")),
                "survival timewiggle wrongly rejected by the non-survival guard: {msg}"
            );
        }
    }

    // ---- #430 location-scale wiggle-pilot unification: parity tests ---------
    //
    // The Gaussian and binomial location-scale model entry points are now thin
    // adapters over the single `fit_location_scale_with_optional_wiggle` engine.
    // The tests below pin that the unified engine reproduces, coefficient for
    // coefficient, the exact per-family reference sequence it replaced — both
    // with and without a wiggle config — so the deslop cannot silently change
    // any fitted result. The reference replays the *old* hand-rolled flow
    // (pilot fit → select link-wiggle basis from the pilot → refit with that
    // basis → extract `beta_link_wiggle` from block 2) directly against the
    // family functions, with no shared code path with the engine other than
    // those leaf family functions.

    fn gaussian_location_scale_dataset() -> Dataset {
        // A mildly heteroscedastic, monotone-in-x signal with enough rows for a
        // stable mean+scale fit and a small wiggle basis.
        let n = 48usize;
        let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
        for i in 0..n {
            let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            // Deterministic, smooth response; the σ-model is intercept-only so
            // the test stays small while still exercising both blocks.
            let y = 0.7 * x + 0.3 * (1.3 * x).sin();
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
            ]));
        }
        crate::inference::data::encode_recordswith_inferred_schema(
            vec!["y".to_string(), "x".to_string()],
            records,
        )
        .expect("encode gaussian location-scale dataset")
    }

    fn binomial_location_scale_dataset() -> Dataset {
        // Balanced 0/1 response with a clear monotone gradient in x so the
        // threshold/log-σ blocks are well posed.
        let n = 60usize;
        let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
        for i in 0..n {
            let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            let y = if i % 2 == 0 { 1.0 } else { 0.0 };
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
            ]));
        }
        crate::inference::data::encode_recordswith_inferred_schema(
            vec!["y".to_string(), "x".to_string()],
            records,
        )
        .expect("encode binomial location-scale dataset")
    }

    fn small_wiggle_cfg() -> LinkWiggleConfig {
        LinkWiggleConfig {
            degree: 3,
            num_internal_knots: 3,
            penalty_orders: vec![2],
            double_penalty: false,
        }
    }

    fn assert_block_states_match(label: &str, lhs: &UnifiedFitResult, rhs: &UnifiedFitResult) {
        assert_eq!(
            lhs.block_states.len(),
            rhs.block_states.len(),
            "{label}: block count mismatch (engine {} vs reference {})",
            lhs.block_states.len(),
            rhs.block_states.len()
        );
        for (i, (a, b)) in lhs
            .block_states
            .iter()
            .zip(rhs.block_states.iter())
            .enumerate()
        {
            assert_eq!(
                a.beta.len(),
                b.beta.len(),
                "{label}: block {i} coefficient length mismatch"
            );
            for (j, (&av, &bv)) in a.beta.iter().zip(b.beta.iter()).enumerate() {
                // The engine and reference share the same leaf family functions
                // and feed them identical inputs, so the fitted coefficients
                // must agree to full numerical precision — this is a refactor,
                // not an approximation. A loose tolerance here would let a real
                // orchestration bug slip through, so the bound stays at the
                // bit-noise floor of an exact replay.
                assert!(
                    (av - bv).abs() <= 1e-12 * (1.0 + bv.abs()),
                    "{label}: block {i} coef {j} diverged: engine {av:.17e} vs reference {bv:.17e}"
                );
            }
        }
    }

    fn assert_beta_link_wiggle_match(
        label: &str,
        engine: &Option<Vec<f64>>,
        reference: &Option<Vec<f64>>,
    ) {
        match (engine, reference) {
            (Some(e), Some(r)) => {
                assert_eq!(
                    e.len(),
                    r.len(),
                    "{label}: beta_link_wiggle length mismatch (engine {} vs reference {})",
                    e.len(),
                    r.len()
                );
                for (j, (&ev, &rv)) in e.iter().zip(r.iter()).enumerate() {
                    // Same exact-replay floor as the block-state comparison: the
                    // engine reads block 2 off the very fit the reference refit
                    // produced, so any divergence beyond bit noise is a bug.
                    assert!(
                        (ev - rv).abs() <= 1e-12 * (1.0 + rv.abs()),
                        "{label}: beta_link_wiggle coef {j} diverged: \
                         engine {ev:.17e} vs reference {rv:.17e}"
                    );
                }
            }
            (None, None) => {}
            (e, r) => panic!(
                "{label}: beta_link_wiggle presence mismatch (engine is_some={}, reference is_some={})",
                e.is_some(),
                r.is_some()
            ),
        }
    }

    #[test]
    fn gaussian_location_scale_engine_matches_reference_flow() {
        let data = gaussian_location_scale_dataset();
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        };
        let materialized =
            materialize("y ~ x", &data, &config).expect("gaussian location-scale materialization");
        let FitRequest::GaussianLocationScale(request) = materialized.request else {
            panic!("expected a Gaussian location-scale request");
        };
        let GaussianLocationScaleFitRequest {
            data: req_data,
            spec,
            options,
            kappa_options,
            ..
        } = request;

        // --- no-wiggle parity ------------------------------------------------
        let engine_plain = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: None,
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine gaussian no-wiggle fit");
        let reference_plain =
            fit_gaussian_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference gaussian no-wiggle fit");
        assert_block_states_match(
            "gaussian/no-wiggle",
            &engine_plain.fit.fit,
            &reference_plain.fit,
        );
        assert!(engine_plain.wiggle_knots.is_none());
        assert!(engine_plain.wiggle_degree.is_none());
        assert!(engine_plain.beta_link_wiggle.is_none());

        // --- wiggle parity ---------------------------------------------------
        let wiggle_cfg = small_wiggle_cfg();
        let engine_wiggle = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: Some(wiggle_cfg.clone()),
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine gaussian wiggle fit");

        // Reference: the exact pre-unification hand-rolled sequence.
        let ref_pilot =
            fit_gaussian_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference gaussian pilot");
        let ref_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            &ref_pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )
        .expect("reference gaussian wiggle basis selection");
        let ref_solved = fit_gaussian_location_scale_terms_with_selected_wiggle(
            req_data,
            spec.clone(),
            ref_basis,
            &options,
            &kappa_options,
        )
        .expect("reference gaussian wiggle refit");

        assert_block_states_match(
            "gaussian/wiggle",
            &engine_wiggle.fit.fit,
            &ref_solved.fit.fit,
        );
        assert_eq!(
            engine_wiggle.wiggle_degree,
            Some(ref_solved.wiggle_degree),
            "gaussian wiggle degree must match the reference refit"
        );
        let engine_knots = engine_wiggle
            .wiggle_knots
            .as_ref()
            .expect("engine gaussian wiggle knots present");
        assert_eq!(
            engine_knots.len(),
            ref_solved.wiggle_knots.len(),
            "gaussian wiggle knot count must match the reference refit"
        );
        for (k, (&ek, &rk)) in engine_knots
            .iter()
            .zip(ref_solved.wiggle_knots.iter())
            .enumerate()
        {
            assert!(
                (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
                "gaussian wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
            );
        }
        // `beta_link_wiggle` is block 2 of the refit; the engine must extract it
        // exactly as the reference would read it off the same fit.
        let ref_beta_link_wiggle = ref_solved
            .fit
            .fit
            .block_states
            .get(2)
            .map(|b| b.beta.to_vec());
        assert_beta_link_wiggle_match(
            "gaussian",
            &engine_wiggle.beta_link_wiggle,
            &ref_beta_link_wiggle,
        );
        assert!(
            engine_wiggle.beta_link_wiggle.is_some(),
            "a wiggle refit must populate beta_link_wiggle (block 2 present)"
        );
    }

    #[test]
    fn binomial_location_scale_engine_matches_reference_flow() {
        let data = binomial_location_scale_dataset();
        let config = FitConfig {
            family: Some("binomial".to_string()),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        };
        let materialized =
            materialize("y ~ x", &data, &config).expect("binomial location-scale materialization");
        let FitRequest::BinomialLocationScale(request) = materialized.request else {
            panic!("expected a binomial location-scale request");
        };
        let BinomialLocationScaleFitRequest {
            data: req_data,
            spec,
            options,
            kappa_options,
            ..
        } = request;

        // --- no-wiggle parity ------------------------------------------------
        let engine_plain = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: None,
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine binomial no-wiggle fit");
        let reference_plain =
            fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference binomial no-wiggle fit");
        assert_block_states_match(
            "binomial/no-wiggle",
            &engine_plain.fit.fit,
            &reference_plain.fit,
        );
        assert!(engine_plain.wiggle_knots.is_none());
        assert!(engine_plain.wiggle_degree.is_none());
        assert!(engine_plain.beta_link_wiggle.is_none());

        // --- wiggle parity ---------------------------------------------------
        let wiggle_cfg = small_wiggle_cfg();
        let engine_wiggle = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: Some(wiggle_cfg.clone()),
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine binomial wiggle fit");

        // Reference: the exact pre-unification hand-rolled sequence, including
        // the binomial-only link compatibility guard.
        require_inverse_link_supports_joint_wiggle(
            &spec.link_kind,
            "binomial location-scale link wiggle",
        )
        .expect("logit link supports joint wiggle");
        let ref_pilot =
            fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference binomial pilot");
        let ref_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            &ref_pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )
        .expect("reference binomial wiggle basis selection");
        let ref_solved = fit_binomial_location_scale_terms_with_selected_wiggle(
            req_data,
            spec.clone(),
            ref_basis,
            &options,
            &kappa_options,
        )
        .expect("reference binomial wiggle refit");

        assert_block_states_match(
            "binomial/wiggle",
            &engine_wiggle.fit.fit,
            &ref_solved.fit.fit,
        );
        assert_eq!(
            engine_wiggle.wiggle_degree,
            Some(ref_solved.wiggle_degree),
            "binomial wiggle degree must match the reference refit"
        );
        let engine_knots = engine_wiggle
            .wiggle_knots
            .as_ref()
            .expect("engine binomial wiggle knots present");
        assert_eq!(
            engine_knots.len(),
            ref_solved.wiggle_knots.len(),
            "binomial wiggle knot count must match the reference refit"
        );
        for (k, (&ek, &rk)) in engine_knots
            .iter()
            .zip(ref_solved.wiggle_knots.iter())
            .enumerate()
        {
            assert!(
                (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
                "binomial wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
            );
        }
        let ref_beta_link_wiggle = ref_solved
            .fit
            .fit
            .block_states
            .get(2)
            .map(|b| b.beta.to_vec());
        assert_beta_link_wiggle_match(
            "binomial",
            &engine_wiggle.beta_link_wiggle,
            &ref_beta_link_wiggle,
        );
        assert!(
            engine_wiggle.beta_link_wiggle.is_some(),
            "a wiggle refit must populate beta_link_wiggle (block 2 present)"
        );
    }
}
