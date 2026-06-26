use super::*;

pub(crate) fn family_arg_canonical_name(arg: FamilyArg) -> Option<&'static str> {
    match arg {
        FamilyArg::Auto => None,
        FamilyArg::Gaussian => Some("gaussian"),
        FamilyArg::BinomialLogit => Some("binomial-logit"),
        FamilyArg::BinomialProbit => Some("binomial-probit"),
        FamilyArg::BinomialCloglog => Some("binomial-cloglog"),
        FamilyArg::LatentCloglogBinomial => Some("latent-cloglog-binomial"),
        FamilyArg::PoissonLog => Some("poisson"),
        FamilyArg::NegativeBinomial => Some("negative-binomial"),
        FamilyArg::GammaLog => Some("gamma"),
        FamilyArg::Tweedie => Some("tweedie"),
        FamilyArg::Beta => Some("beta"),
        FamilyArg::RoystonParmar => Some("royston-parmar"),
        FamilyArg::TransformationNormal => Some("transformation-normal"),
    }
}

/// CLI adapter over the canonical family resolver.
///
/// The fit-routing contract — explicit family vs link-implied family, the
/// SAS/Beta-Logistic links, negative-binomial `theta`, and response
/// auto-inference — lives once in `gam::resolve_family`. The CLI keeps only the
/// surface-specific concerns: translating the typed `FamilyArg` into the
/// canonical name and enforcing the CLI flag rule that
/// `--negative-binomial-theta` is meaningful exclusively with
/// `--family negative-binomial`.
///
/// The user's `link(sas_init=...)` / `link(beta_logistic_init=...)` state is
/// not threaded through this resolver: family resolution produces the
/// link-only placeholder, and the standard fit picks up the actual initial
/// state from `FitOptions.sas_link` (see `effective_sas_link_for_family` in
/// `src/solver/estimate.rs`), which overrides the family-embedded link. Keeping
/// the resolver link-state-free leaves a single, narrow family-routing contract
/// shared verbatim with the workflow and PyFFI surfaces.
pub(crate) fn resolve_family(
    arg: FamilyArg,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    if negative_binomial_theta.is_some() && !matches!(arg, FamilyArg::NegativeBinomial) {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    gam::families::fit_orchestration::resolve_family(
        family_arg_canonical_name(arg),
        negative_binomial_theta,
        link_choice.as_ref(),
        y,
        y_kind,
        response_name,
    )
}

pub(crate) fn inverse_link_from_fitted_link_state(state: &FittedLinkState) -> Option<InverseLink> {
    match state {
        FittedLinkState::Standard(Some(link)) => Some(InverseLink::Standard(*link)),
        FittedLinkState::Standard(None) => None,
        FittedLinkState::LatentCLogLog { state } => Some(InverseLink::LatentCLogLog(*state)),
        FittedLinkState::Sas { state, .. } => Some(InverseLink::Sas(*state)),
        FittedLinkState::BetaLogistic { state, .. } => Some(InverseLink::BetaLogistic(*state)),
        FittedLinkState::Mixture { state, .. } => Some(InverseLink::Mixture(state.clone())),
    }
}

pub(crate) fn resolve_binomial_inverse_link_for_fit(
    family: LikelihoodSpec,
    effective_link: LinkFunction,
    mixture_linkspec: Option<&MixtureLinkSpec>,
    context: &str,
) -> Result<InverseLink, String> {
    if !family.is_binomial() {
        return Err(format!(
            "{context} is only available for binomial links, got {}",
            family.name()
        ));
    }
    match &family.link {
        InverseLink::Standard(StandardLink::Logit) => {
            let spec = mixture_linkspec
                .ok_or_else(|| format!("{context} requires link(type=blended(...))"))?;
            let state = state_fromspec(spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            Ok(InverseLink::Mixture(state))
        }
        // `resolve_family` already upgrades Sas / BetaLogistic to their
        // state-bearing variants; we only need to forward them here.
        InverseLink::Sas(state) => Ok(InverseLink::Sas(*state)),
        InverseLink::BetaLogistic(state) => Ok(InverseLink::BetaLogistic(*state)),
        InverseLink::Standard(StandardLink::CLogLog) => Err(format!(
            "{context} does not construct latent-cloglog links directly; use the latent-cloglog family path with explicit frailty"
        )),
        InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::Standard(StandardLink::Log)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Mixture(_) => Ok(InverseLink::Standard(
            crate::config_resolve::effective_link_to_standard(effective_link, context)?,
        )),
    }
}

pub(crate) fn binomial_mean_linkwiggle_supports_family(
    family: &LikelihoodSpec,
    link_choice: Option<&LinkChoice>,
) -> bool {
    let standard_binomial = family.is_binomial()
        && matches!(
            &family.link,
            InverseLink::Standard(StandardLink::Logit)
                | InverseLink::Standard(StandardLink::Probit)
                | InverseLink::Standard(StandardLink::CLogLog)
        );
    standard_binomial
        && !link_choice.is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
}

pub(crate) fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Project the CLI's `EncodedDataset` column-kind tag onto the
/// [`ResponseColumnKind`] consumed by the family layer. Mirrors the helper
/// of the same name in `workflow.rs` — having two tiny copies (one per
/// crate-internal entry point) is cleaner than threading the ingest enum
/// itself into the types layer.
pub(crate) fn response_column_kind_for_dataset(ds: &Dataset, y_col: usize) -> ResponseColumnKind {
    match ds.column_kinds.get(y_col) {
        Some(ColumnKindTag::Categorical) => ResponseColumnKind::Categorical {
            levels: ds
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
