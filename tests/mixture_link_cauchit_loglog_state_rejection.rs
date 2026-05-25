use gam::mixture_link::state_fromspec;
use gam::types::{LinkComponent, MixtureLinkSpec};
use ndarray::arr1;

#[test]
fn state_fromspec_rejects_cauchit_and_loglog_components() {
    let spec = MixtureLinkSpec {
        components: vec![LinkComponent::Cauchit, LinkComponent::LogLog],
        initial_rho: arr1(&[0.0]),
    };

    let err = state_fromspec(&spec).expect_err(
        "MixtureLinkSpec with Cauchit/LogLog must be rejected until LinkFunction supports them",
    );

    assert!(
        err.to_ascii_lowercase().contains("cauchit")
            || err.to_ascii_lowercase().contains("loglog")
            || err.to_ascii_lowercase().contains("unsupported"),
        "error should clearly mention unsupported component(s), got: {err}"
    );
}
