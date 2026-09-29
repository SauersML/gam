//! `secant`: the exact two-endpoint change operators of
//! [`crate::secant`] on the wire, each with the owner's
//! per-entry forward-error band.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::{MpdOutput, MpdResult, MpdSurfaceError, matrix, output, reserve, vector};
use crate::secant::{
    BandedMatrix, BandedVector, RmsNormSecant, SecantActivation, SoftmaxSecant,
    activation_divided_differences, bilinear_change, bilinear_vector_change, softmax_change,
};

/// One secant operator. Every array field is the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecantRequest {
    pub operator: SecantOperator,
}

/// The owner operators.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SecantOperator {
    /// `softmax(end) − softmax(start)` (`secant::softmax_change`).
    SoftmaxChange { start: String, end: String },
    /// `A δ` for the softmax secant operator between `start` and `end`
    /// (`SoftmaxSecant::apply`), with its logarithmic means.
    SoftmaxApply {
        start: String,
        end: String,
        direction: String,
    },
    /// `N(end) − N(start)` for gainless RMSNorm with offset `epsilon`
    /// (`RmsNormSecant::change`).
    RmsNormChange {
        start: String,
        end: String,
        epsilon: f64,
    },
    /// `g ⊙ (N(end) − N(start))` (`RmsNormSecant::change` then
    /// `BandedVector::gained`).
    GainedRmsNormChange {
        start: String,
        end: String,
        epsilon: f64,
        gain: String,
    },
    /// `B Δ` for the RMSNorm secant operator between `start` and `end`
    /// (`RmsNormSecant::apply`).
    RmsNormApply {
        start: String,
        end: String,
        epsilon: f64,
        direction: String,
    },
    /// `L′R′ − LR` (`secant::bilinear_change`).
    BilinearChange {
        left_start: String,
        left_end: String,
        right_start: String,
        right_end: String,
    },
    /// `W′h′ − Wh` (`secant::bilinear_vector_change`).
    BilinearVectorChange {
        weight_start: String,
        weight_end: String,
        input_start: String,
        input_end: String,
    },
    /// The divided difference of `activation` entry by entry
    /// (`secant::activation_divided_differences`).
    ActivationDividedDifferences {
        activation: SecantActivationWire,
        start: String,
        end: String,
    },
}

/// [`SecantActivation`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecantActivationWire {
    Silu,
    ExactGelu,
}

/// The operator's values and their per-entry bands on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SecantReport {
    /// Id of the values.
    pub values: String,
    /// Id of the bands, one per value.
    pub bands: String,
    /// For `softmax_apply`: id of the logarithmic means `ℓ`.
    pub logarithmic_means: Option<String>,
}

fn slice<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<&'a [f64], MpdSurfaceError> {
    vector(tensors, id)?
        .to_slice()
        .ok_or_else(|| MpdSurfaceError::TensorShape {
            tensor: id.to_string(),
            reason: "expected a contiguous vector".to_string(),
        })
}

pub(super) fn run(
    request: SecantRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let secant = MpdSurfaceError::Secant;
    let banded = |banded: BandedVector| (Array1::from(banded.values).into_dyn(), Array1::from(banded.bands).into_dyn());
    let matrix_banded = |banded: BandedMatrix| (banded.values.into_dyn(), banded.bands.into_dyn());
    let mut means = None;
    let (values, bands) = match request.operator {
        SecantOperator::SoftmaxChange { start, end } => {
            banded(softmax_change(slice(tensors, &start)?, slice(tensors, &end)?).map_err(secant)?)
        }
        SecantOperator::SoftmaxApply {
            start,
            end,
            direction,
        } => {
            let operator = SoftmaxSecant::between(slice(tensors, &start)?, slice(tensors, &end)?)
                .map_err(secant)?;
            let applied = operator.apply(slice(tensors, &direction)?).map_err(secant)?;
            means = Some(Array1::from(operator.logarithmic_means().to_vec()).into_dyn());
            banded(applied)
        }
        SecantOperator::RmsNormChange {
            start,
            end,
            epsilon,
        } => banded(
            RmsNormSecant::between(slice(tensors, &start)?, slice(tensors, &end)?, epsilon)
                .map_err(secant)?
                .change(),
        ),
        SecantOperator::GainedRmsNormChange {
            start,
            end,
            epsilon,
            gain,
        } => banded(
            RmsNormSecant::between(slice(tensors, &start)?, slice(tensors, &end)?, epsilon)
                .map_err(secant)?
                .change()
                .gained(slice(tensors, &gain)?)
                .map_err(secant)?,
        ),
        SecantOperator::RmsNormApply {
            start,
            end,
            epsilon,
            direction,
        } => banded(
            RmsNormSecant::between(slice(tensors, &start)?, slice(tensors, &end)?, epsilon)
                .map_err(secant)?
                .apply(slice(tensors, &direction)?)
                .map_err(secant)?,
        ),
        SecantOperator::BilinearChange {
            left_start,
            left_end,
            right_start,
            right_end,
        } => {
            let left = matrix(tensors, &left_start)?;
            let right = matrix(tensors, &right_start)?;
            // The two means, the two steps, their absolute values, and the values and
            // bands, at the larger of the operand and product shapes.
            let rows = left.nrows().max(right.nrows());
            let cols = left.ncols().max(right.ncols());
            let formed = reserve(governor, rows, cols, 10, "bilinear secant")?;
            let change = bilinear_change(
                left,
                matrix(tensors, &left_end)?,
                right,
                matrix(tensors, &right_end)?,
            )
            .map_err(secant);
            drop(formed);
            matrix_banded(change?)
        }
        SecantOperator::BilinearVectorChange {
            weight_start,
            weight_end,
            input_start,
            input_end,
        } => {
            let weight = matrix(tensors, &weight_start)?;
            let formed = reserve(governor, weight.nrows(), weight.ncols(), 6, "bilinear vector secant")?;
            let change = bilinear_vector_change(
                weight,
                matrix(tensors, &weight_end)?,
                vector(tensors, &input_start)?,
                vector(tensors, &input_end)?,
            )
            .map_err(secant);
            drop(formed);
            banded(change?)
        }
        SecantOperator::ActivationDividedDifferences {
            activation,
            start,
            end,
        } => {
            let activation = match activation {
                SecantActivationWire::Silu => SecantActivation::Silu,
                SecantActivationWire::ExactGelu => SecantActivation::ExactGelu,
            };
            banded(
                activation_divided_differences(activation, slice(tensors, &start)?, slice(tensors, &end)?)
                    .map_err(secant)?,
            )
        }
    };
    let mut arrays = BTreeMap::from([
        ("values".to_string(), values),
        ("bands".to_string(), bands),
    ]);
    let logarithmic_means = means.map(|means| {
        arrays.insert("logarithmic_means".to_string(), means);
        "logarithmic_means".to_string()
    });
    Ok(output(
        MpdResult::Secant(SecantReport {
            values: "values".to_string(),
            bands: "bands".to_string(),
            logarithmic_means,
        }),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::secant::{SecantError, activation_divided_difference};
    use crate::test_support::test_governor;
    use ndarray::{Array2, array};

    fn run_operator(operator: &str, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
        run_parameter_decomposition(
            &request_json(&format!(r#"{{"kind": "secant", "operator": {operator}}}"#)),
            tensors,
            test_governor(),
        )
    }

    fn inputs() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("u".to_string(), array![0.5, -2.0, 3.0].into_dyn()),
            ("v".to_string(), array![1.5, -1.0, -4.0].into_dyn()),
            ("d".to_string(), array![0.25, 1.0, -0.5].into_dyn()),
            ("g".to_string(), array![2.0, -1.0, 0.5].into_dyn()),
            ("l0".to_string(), array![[1.0, 2.0], [0.5, -1.0]].into_dyn()),
            ("l1".to_string(), array![[1.5, 2.0], [0.0, -1.0]].into_dyn()),
            ("r0".to_string(), array![[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5]].into_dyn()),
            ("r1".to_string(), array![[1.0, 0.5, 2.0], [-1.5, 3.0, 0.5]].into_dyn()),
            ("h0".to_string(), array![1.0, -1.0].into_dyn()),
            ("h1".to_string(), array![0.5, 2.0].into_dyn()),
        ])
    }

    fn assert_banded(output: &MpdOutput, owner: &BandedVector) {
        let MpdResult::Secant(report) = &output.report.result else {
            panic!("expected a secant report, got {:?}", output.report.result);
        };
        assert_eq!(output.arrays[&report.values], Array1::from(owner.values.clone()).into_dyn());
        assert_eq!(output.arrays[&report.bands], Array1::from(owner.bands.clone()).into_dyn());
    }

    #[test]
    fn secant_reports_are_the_owner_results_field_for_field() {
        let tensors = inputs();
        let (u, v, d, g) = ([0.5, -2.0, 3.0], [1.5, -1.0, -4.0], [0.25, 1.0, -0.5], [2.0, -1.0, 0.5]);

        let output = run_operator(r#"{"kind": "softmax_change", "start": "u", "end": "v"}"#, &tensors).expect("run");
        assert_banded(&output, &softmax_change(&u, &v).expect("owner"));

        let output = run_operator(
            r#"{"kind": "softmax_apply", "start": "u", "end": "v", "direction": "d"}"#,
            &tensors,
        )
        .expect("run");
        let operator = SoftmaxSecant::between(&u, &v).expect("owner");
        assert_banded(&output, &operator.apply(&d).expect("owner apply"));
        let MpdResult::Secant(report) = &output.report.result else {
            panic!("expected a secant report");
        };
        let means = report.logarithmic_means.as_ref().expect("means named");
        assert_eq!(
            output.arrays[means],
            Array1::from(operator.logarithmic_means().to_vec()).into_dyn()
        );

        let rms = RmsNormSecant::between(&u, &v, 1e-6).expect("owner");
        let output = run_operator(
            r#"{"kind": "rms_norm_change", "start": "u", "end": "v", "epsilon": 1e-6}"#,
            &tensors,
        )
        .expect("run");
        assert_banded(&output, &rms.change());
        let output = run_operator(
            r#"{"kind": "gained_rms_norm_change", "start": "u", "end": "v", "epsilon": 1e-6, "gain": "g"}"#,
            &tensors,
        )
        .expect("run");
        assert_banded(&output, &rms.change().gained(&g).expect("owner gain"));
        let output = run_operator(
            r#"{"kind": "rms_norm_apply", "start": "u", "end": "v", "epsilon": 1e-6, "direction": "d"}"#,
            &tensors,
        )
        .expect("run");
        assert_banded(&output, &rms.apply(&d).expect("owner apply"));

        let output = run_operator(
            r#"{"kind": "activation_divided_differences", "activation": "exact_gelu", "start": "u", "end": "v"}"#,
            &tensors,
        )
        .expect("run");
        let owner = activation_divided_differences(SecantActivation::ExactGelu, &u, &v).expect("owner");
        assert_banded(&output, &owner);
        assert_eq!(
            owner.values[0],
            activation_divided_difference(SecantActivation::ExactGelu, u[0], v[0]).expect("scalar").value
        );

        let as_matrix = |id: &str| matrix(&tensors, id).expect("matrix");
        let owner = bilinear_change(as_matrix("l0"), as_matrix("l1"), as_matrix("r0"), as_matrix("r1")).expect("owner");
        let output = run_operator(
            r#"{"kind": "bilinear_change", "left_start": "l0", "left_end": "l1", "right_start": "r0", "right_end": "r1"}"#,
            &tensors,
        )
        .expect("run");
        let MpdResult::Secant(report) = &output.report.result else {
            panic!("expected a secant report");
        };
        assert_eq!(output.arrays[&report.values], owner.values.clone().into_dyn());
        assert_eq!(output.arrays[&report.bands], owner.bands.clone().into_dyn());
        // Exactly `L′R′ − LR` here, since every entry is a dyadic rational of few bits.
        let exact: Array2<f64> = as_matrix("l1").dot(&as_matrix("r1")) - as_matrix("l0").dot(&as_matrix("r0"));
        assert_eq!(owner.values, exact);
        assert!(report.logarithmic_means.is_none());

        let output = run_operator(
            r#"{"kind": "bilinear_vector_change", "weight_start": "l0", "weight_end": "l1", "input_start": "h0", "input_end": "h1"}"#,
            &tensors,
        )
        .expect("run");
        let owner = bilinear_vector_change(
            as_matrix("l0"),
            as_matrix("l1"),
            vector(&tensors, "h0").expect("h0"),
            vector(&tensors, "h1").expect("h1"),
        )
        .expect("owner");
        assert_banded(&output, &owner);
        assert_eq!(output.arrays.len(), 2);
    }

    #[test]
    fn a_secant_request_the_owner_refuses_reaches_the_caller() {
        let mut tensors = inputs();
        assert!(run_operator(r#"{"kind": "softmax_change", "start": "u", "end": "v"}"#, &tensors).is_ok());
        assert!(matches!(
            run_operator(r#"{"kind": "softmax_change", "start": "u", "end": "h0"}"#, &tensors),
            Err(MpdSurfaceError::Secant(SecantError::Shape { .. }))
        ));
        assert!(matches!(
            run_operator(r#"{"kind": "rms_norm_change", "start": "u", "end": "v", "epsilon": 0.0}"#, &tensors),
            Err(MpdSurfaceError::Secant(SecantError::NonPositiveEpsilon { .. }))
        ));
        assert!(matches!(
            run_operator(
                r#"{"kind": "bilinear_change", "left_start": "l0", "left_end": "l1", "right_start": "l0", "right_end": "r1"}"#,
                &tensors
            ),
            Err(MpdSurfaceError::Secant(SecantError::Shape { .. }))
        ));
        tensors.insert("u".to_string(), array![0.5, f64::NAN, 3.0].into_dyn());
        assert!(matches!(
            run_operator(r#"{"kind": "softmax_change", "start": "u", "end": "v"}"#, &tensors),
            Err(MpdSurfaceError::Secant(SecantError::NonFinite { .. }))
        ));
        assert!(matches!(
            run_operator(r#"{"kind": "softmax_change", "start": "l0", "end": "v"}"#, &tensors),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        assert!(matches!(
            run_operator(r#"{"kind": "activation_divided_differences", "activation": "gelu_tanh", "start": "v", "end": "v"}"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            run_operator(r#"{"kind": "softmax_change", "start": "v", "end": "v", "temperature": 1.0}"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
