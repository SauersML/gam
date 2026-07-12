//! #932 production-artifact policy.
//!
//! Integration tests compile `gam_models` as a dependency without `cfg(test)`.
//! These public imports and trait bounds therefore fail at compile time if a
//! canonical family row program is moved back under a unit-test gate or loses
//! its `RowProgram` implementation. The numerical parity/finite-difference
//! pins remain beside each specialized production path and instantiate these
//! same exported types through absolute paths.

use gam_math::jet_tower::RowProgram;
use gam_models::gamlss::{GaussianJointRowProgram, GaussianJointRowScalars};
use gam_models::survival::CauseSpecificRowProgram;
use gam_models::MultinomialLogitRowProgram;

struct ProductionSpecialization {
    family: &'static str,
    source: &'static str,
    production_type: &'static str,
    row_program_impl: &'static str,
    shared_generic_lowering: Option<&'static str>,
    parity_pin: &'static str,
    minimum_parity_pins: usize,
    forbidden_test_mirror: &'static str,
}

const PRODUCTION_SPECIALIZATIONS: &[ProductionSpecialization] = &[
    ProductionSpecialization {
        family: "multinomial",
        source: include_str!("../src/multinomial_reml.rs"),
        production_type: "pub struct MultinomialLogitRowProgram<const M: usize>",
        row_program_impl:
            "impl<const M: usize> gam_math::jet_tower::RowProgram<M> for MultinomialLogitRowProgram<M>",
        shared_generic_lowering: None,
        parity_pin:
            "crate::multinomial_reml::MultinomialLogitRowProgram::new(eta, obs, w)",
        minimum_parity_pins: 1,
        forbidden_test_mirror: "struct MultinomialJetRow",
    },
    ProductionSpecialization {
        family: "gaulss",
        source: include_str!("../src/gamlss/gaussian/joint_psi.rs"),
        production_type: "pub struct GaussianJointRowProgram<'a>",
        row_program_impl:
            "impl gam_math::jet_tower::RowProgram<2> for GaussianJointRowProgram<'_>",
        shared_generic_lowering:
            Some("fn gaussian_normalized_row [generic, order2, third, fourth]"),
        parity_pin: "crate::gamlss::GaussianJointRowProgram::new(&rows)",
        minimum_parity_pins: 2,
        forbidden_test_mirror: "struct GaulssJetRow",
    },
    ProductionSpecialization {
        family: "cause-specific survival",
        source: include_str!("../src/survival/base.rs"),
        production_type: "pub struct CauseSpecificRowProgram",
        row_program_impl: "impl gam_math::jet_tower::RowProgram<3> for CauseSpecificRowProgram",
        shared_generic_lowering:
            Some("fn cause_specific_row [generic, order2, third, fourth]"),
        parity_pin: "crate::survival::CauseSpecificRowProgram::new(",
        minimum_parity_pins: 1,
        forbidden_test_mirror: "struct CauseSpecificJetRow",
    },
];

fn require_row_program<const K: usize, P: RowProgram<K>>() {}

#[test]
fn canonical_row_programs_are_production_artifacts() {
    require_row_program::<2, MultinomialLogitRowProgram<2>>();
    require_row_program::<3, CauseSpecificRowProgram>();

    fn require_gaussian<'a>() {
        require_row_program::<2, GaussianJointRowProgram<'a>>();
    }
    require_gaussian();

    let _: for<'a> fn(&'a GaussianJointRowScalars) -> GaussianJointRowProgram<'a> =
        GaussianJointRowProgram::new;
}

#[test]
fn specialization_registry_owns_production_programs_and_parity_pins() {
    for specialization in PRODUCTION_SPECIALIZATIONS {
        let production_offset = specialization
            .source
            .find(specialization.production_type)
            .unwrap_or_else(|| {
                panic!(
                    "{} production row-program type is missing",
                    specialization.family
                )
            });
        let test_gate_offset = specialization
            .source
            .find("#[cfg(test)]")
            .unwrap_or_else(|| panic!("{} test gate is missing", specialization.family));
        assert!(
            production_offset < test_gate_offset,
            "{} row-program type moved behind cfg(test)",
            specialization.family
        );
        assert!(
            specialization
                .source
                .contains(specialization.row_program_impl),
            "{} production RowProgram implementation is missing",
            specialization.family
        );
        if let Some(lowering) = specialization.shared_generic_lowering {
            assert!(
                specialization.source.contains(lowering),
                "{} generic and specialized lowerings no longer share one row declaration",
                specialization.family
            );
        }
        assert!(
            specialization
                .source
                .matches(specialization.parity_pin)
                .count()
                >= specialization.minimum_parity_pins,
            "{} parity pins no longer instantiate the production row program",
            specialization.family
        );
        assert!(
            !specialization
                .source
                .contains(specialization.forbidden_test_mirror),
            "{} reintroduced a test-local row-program expression",
            specialization.family
        );
    }
}
