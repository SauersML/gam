use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_link_choice;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::types::{InverseLink, ResponseColumnKind, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, materialize, resolve_family};
use ndarray::{Array2, array};

fn tiny_binary_dataset() -> EncodedDataset {
    EncodedDataset {
        headers: vec!["y".into(), "x".into()],
        values: Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 2.0])
            .expect("shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Binary, ColumnKindTag::Continuous],
    }
}

#[test]
fn resolve_family_accepts_binomial_logit_with_underscore_alias() {
    let y = array![0.0, 1.0, 1.0, 0.0];
    let resolved = resolve_family(
        Some("binomial_logit"),
        None,
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("binomial_logit should be recognized as a supported family alias");
    assert_eq!(
        resolved.response,
        ResponseFamily::Binomial,
        "binomial_logit should map to the binomial response family"
    );
    assert_eq!(
        resolved.link,
        InverseLink::Standard(StandardLink::Logit),
        "binomial_logit should map to the standard logit inverse-link"
    );
}

#[test]
fn resolve_family_binomial_with_explicit_link_overrides_default_logit() {
    // GitHub issue #155: family='binomial' + link='probit'/'cloglog' must be
    // accepted because the bare 'binomial' family name does not pin a link.
    let y = array![0.0, 1.0, 1.0, 0.0];
    for (link_name, expected) in [
        ("probit", StandardLink::Probit),
        ("cloglog", StandardLink::CLogLog),
        ("logit", StandardLink::Logit),
    ] {
        let link_choice = parse_link_choice(Some(link_name), false)
            .expect("link should parse")
            .expect("link choice should not be None when raw was provided");
        let resolved = resolve_family(
            Some("binomial"),
            None,
            Some(&link_choice),
            y.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .unwrap_or_else(|err| panic!("binomial + link={link_name} should be accepted, got: {err}"));
        assert_eq!(resolved.response, ResponseFamily::Binomial);
        assert_eq!(resolved.link, InverseLink::Standard(expected));
    }
}

#[test]
fn resolve_family_accepts_tweedie() {
    // GitHub issue #158: family='tweedie' must be a recognized family.
    let y = array![0.0, 1.0, 2.0, 0.0, 3.5, 0.0];
    let resolved = resolve_family(
        Some("tweedie"),
        None,
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("tweedie family should be recognized");
    match resolved.response {
        ResponseFamily::Tweedie { p } => {
            assert!(
                p > 1.0 && p < 2.0,
                "default tweedie p should lie in (1, 2); got {p}"
            );
        }
        other => panic!("expected Tweedie response family, got {other:?}"),
    }
    assert_eq!(resolved.link, InverseLink::Standard(StandardLink::Log));
}

#[test]
fn resolve_family_rejects_unknown_family_names() {
    let y = array![0.0, 1.0, 1.0, 0.0];
    let err = resolve_family(
        Some("definitely-not-a-family"),
        None,
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect_err("unknown family names should return an error instead of being silently coerced");
    assert!(
        err.contains("unknown family"),
        "unknown family error should include plain-language context"
    );
}

#[test]
fn materialize_standard_request_keeps_full_likelihood_spec_for_sas_link() {
    let data = tiny_binary_dataset();
    let mut cfg = FitConfig::default();
    cfg.family = Some("binomial".into());
    cfg.link = Some("sas".into());

    let mat = materialize("y ~ x", &data, &cfg).expect(
        "materialize should build a standard fit request for a non-survival binary formula",
    );

    let FitRequest::Standard(req) = mat.request else {
        panic!("materialize should route this configuration to FitRequest::Standard");
    };

    assert!(
        matches!(req.family.link, InverseLink::Sas(_)),
        "standard request should store the family as LikelihoodSpec with the SAS inverse-link state intact"
    );
}

/// Regression for GitHub issue #961: when an explicit response family is given
/// together with `link(type=...)`, the resolver must validate the link against
/// that family rather than re-inferring a (conflicting) family from the link.
/// These legal pairings used to be wrongly rejected because the link inferred
/// its own family (logit -> Binomial, log -> Poisson/Gamma) and the explicit
/// family's discriminant did not match.
#[test]
fn resolve_family_explicit_family_does_not_infer_conflicting_family_from_link() {
    // beta + logit: legal (logit-inferred Binomial must not displace Beta).
    let y_unit = array![0.1, 0.5, 0.9, 0.3];
    let logit = parse_link_choice(Some("logit"), false)
        .expect("logit parses")
        .expect("logit choice present");
    for family in ["beta", "beta-logit"] {
        let resolved = resolve_family(
            Some(family),
            None,
            Some(&logit),
            y_unit.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .unwrap_or_else(|err| panic!("{family} + logit must resolve, got: {err}"));
        assert!(
            matches!(resolved.response, ResponseFamily::Beta { .. }),
            "{family} + logit should keep the Beta response family, got {:?}",
            resolved.response
        );
        assert_eq!(resolved.link, InverseLink::Standard(StandardLink::Logit));
    }

    // tweedie + log: legal (log-inferred Poisson/Gamma must not displace Tweedie).
    let y_count = array![0.0, 1.0, 2.0, 3.0];
    let log = parse_link_choice(Some("log"), false)
        .expect("log parses")
        .expect("log choice present");
    for family in ["tweedie", "tweedie-log"] {
        let resolved = resolve_family(
            Some(family),
            None,
            Some(&log),
            y_count.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .unwrap_or_else(|err| panic!("{family} + log must resolve, got: {err}"));
        assert!(
            matches!(resolved.response, ResponseFamily::Tweedie { .. }),
            "{family} + log should keep the Tweedie response family, got {:?}",
            resolved.response
        );
        assert_eq!(resolved.link, InverseLink::Standard(StandardLink::Log));
    }

    // gamma + log: legal even when y is integer-like (the no-family log path
    // would infer Poisson, but an explicit gamma must win).
    let resolved = resolve_family(
        Some("gamma"),
        None,
        Some(&log),
        y_count.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("gamma + log must resolve even with integer-like y");
    assert_eq!(resolved.response, ResponseFamily::Gamma);
    assert_eq!(resolved.link, InverseLink::Standard(StandardLink::Log));
}

/// Regression for GitHub issue #961: genuinely-illegal (family, link) pairings
/// must still be rejected with a clear, family-specific message.
#[test]
fn resolve_family_rejects_genuinely_illegal_family_link_pairs() {
    let y_unit = array![0.1, 0.5, 0.9, 0.3];
    let log = parse_link_choice(Some("log"), false)
        .expect("log parses")
        .expect("log choice present");
    let err = resolve_family(
        Some("beta"),
        None,
        Some(&log),
        y_unit.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect_err("beta + log is illegal and must be rejected");
    assert!(
        err.contains("not supported for family 'beta'"),
        "beta + log rejection should name the family and link, got: {err}"
    );

    let y = array![0.0, 1.0, 1.0, 0.0];
    let logit = parse_link_choice(Some("logit"), false)
        .expect("logit parses")
        .expect("logit choice present");
    let err = resolve_family(
        Some("gaussian"),
        None,
        Some(&logit),
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect_err("gaussian + logit is illegal and must be rejected");
    assert!(
        err.contains("not supported for family 'gaussian'"),
        "gaussian + logit rejection should name the family and link, got: {err}"
    );

    // A family name that pins its own link may not be re-pointed at a different
    // link via `link(type=...)`.
    let err = resolve_family(
        Some("binomial-probit"),
        None,
        Some(&logit),
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect_err("binomial-probit + logit is a pinned-link conflict and must be rejected");
    assert!(
        err.contains("pins link"),
        "pinned-link conflict should explain the pin, got: {err}"
    );
}
