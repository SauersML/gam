use gam::inference::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::types::{InverseLink, LinkFunction, ResponseFamily};
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
    let resolved = resolve_family(Some("binomial_logit"), None, None, y.view())
        .expect("binomial_logit should be recognized as a supported family alias");
    assert_eq!(
        resolved.response,
        ResponseFamily::Binomial,
        "binomial_logit should map to the binomial response family"
    );
    assert_eq!(
        resolved.link,
        InverseLink::Standard(LinkFunction::Logit),
        "binomial_logit should map to the standard logit inverse-link"
    );
}

#[test]
fn resolve_family_rejects_unknown_family_names() {
    let y = array![0.0, 1.0, 1.0, 0.0];
    let err = resolve_family(Some("definitely-not-a-family"), None, None, y.view()).expect_err(
        "unknown family names should return an error instead of being silently coerced",
    );
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
