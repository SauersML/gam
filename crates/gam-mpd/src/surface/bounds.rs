//! `logit_bounds`: the logit-box KL and total-variation bounds and the attention-read
//! bound of [`crate::bounds`] on the wire.

use std::collections::BTreeMap;

use ndarray::{ArrayD, ArrayView2, Axis, Ix1, Ix2};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, matrix};
use crate::bounds::{
    AttentionReadRegion, HeadRead, KlBoundRegion, TotalVariationRegion, attention_read_bound,
    kl_over_logit_boxes, kl_supremum_over_logit_boxes, payload_diameter,
    softmax_total_variation_bound, total_variation_over_logit_boxes,
};

/// One bound. Logits and radii are ids of vectors (one row) or matrices (one bound per
/// row).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LogitBoundRequest {
    /// `KL(softmax ℓ_p ‖ softmax ℓ_q)` at the centres with the box shift
    /// (`bounds::kl_over_logit_boxes`).
    Kl { boxes: LogitBoxes },
    /// `sup KL` over the boxes (`bounds::kl_supremum_over_logit_boxes`).
    KlSupremum { boxes: LogitBoxes },
    /// `sup TV` over the boxes (`bounds::total_variation_over_logit_boxes`).
    TotalVariation { boxes: LogitBoxes },
    /// `tanh(w/4)` for a logit-error range `w` (`bounds::softmax_total_variation_bound`).
    SoftmaxTotalVariation { range: f64 },
    /// `2 · radius · ‖C‖_F` (`bounds::payload_diameter`).
    PayloadDiameter { payload_map: String, radius: f64 },
    /// `Σ_h diam_h · TV_h` (`bounds::attention_read_bound`).
    AttentionRead { heads: Vec<HeadReadRequest> },
}

/// Two logit centres and their per-entry radii.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogitBoxes {
    pub reference: String,
    pub reference_radius: String,
    pub perturbed: String,
    pub perturbed_radius: String,
}

/// [`HeadRead`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HeadReadRequest {
    pub total_variation: f64,
    pub diameter: f64,
}

/// The `logit_bounds` operation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogitBoundsRequest {
    pub bound: LogitBoundRequest,
}

/// [`KlBoundRegion`] and [`TotalVariationRegion`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogitBoxRegion {
    pub reference_radius: f64,
    pub perturbed_radius: f64,
}

/// [`AttentionReadRegion`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionReadRegionReport {
    pub heads: usize,
}

pub type BoxStatusWire = EvidenceStatusWire<(), LogitBoxRegion>;

/// The bound's result.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogitBoundsReport {
    /// One status per row of the boxes.
    Boxes { rows: Vec<BoxStatusWire> },
    Value { value: f64 },
    AttentionRead { status: EvidenceStatusWire<(), AttentionReadRegionReport> },
}

fn kl_region(region: KlBoundRegion) -> LogitBoxRegion {
    match region {
        KlBoundRegion::LogitBoxes {
            reference_radius,
            perturbed_radius,
        } => LogitBoxRegion {
            reference_radius,
            perturbed_radius,
        },
    }
}

fn tv_region(region: TotalVariationRegion) -> LogitBoxRegion {
    match region {
        TotalVariationRegion::LogitBoxes {
            reference_radius,
            perturbed_radius,
        } => LogitBoxRegion {
            reference_radius,
            perturbed_radius,
        },
    }
}

/// A vector (one row) or a matrix, as rows.
fn rows<'a>(tensors: &'a BTreeMap<String, ArrayD<f64>>, id: &str) -> Result<ArrayView2<'a, f64>, MpdSurfaceError> {
    let array = input(tensors, id)?;
    let shape_error = |error: ndarray::ShapeError| MpdSurfaceError::TensorShape {
        tensor: id.to_string(),
        reason: format!("expected a vector or a matrix, got shape {:?}: {error}", array.shape()),
    };
    if array.ndim() == 1 {
        Ok(array.view().into_dimensionality::<Ix1>().map_err(shape_error)?.insert_axis(Axis(0)))
    } else {
        array.view().into_dimensionality::<Ix2>().map_err(shape_error)
    }
}

pub(super) fn run(request: LogitBoundsRequest, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
    let bound = MpdSurfaceError::Bound;
    let report = match request.bound {
        LogitBoundRequest::Kl { boxes } => per_row(&boxes, tensors, |r, rr, p, pr| {
            EvidenceStatusWire::from_status_with(kl_over_logit_boxes(r, rr, p, pr).map_err(bound)?, |w| w, kl_region)
        })?,
        LogitBoundRequest::KlSupremum { boxes } => per_row(&boxes, tensors, |r, rr, p, pr| {
            EvidenceStatusWire::from_status_with(
                kl_supremum_over_logit_boxes(r, rr, p, pr).map_err(bound)?,
                |w| w,
                kl_region,
            )
        })?,
        LogitBoundRequest::TotalVariation { boxes } => per_row(&boxes, tensors, |r, rr, p, pr| {
            EvidenceStatusWire::from_status_with(
                total_variation_over_logit_boxes(r, rr, p, pr).map_err(bound)?,
                |w| w,
                tv_region,
            )
        })?,
        LogitBoundRequest::SoftmaxTotalVariation { range } => LogitBoundsReport::Value {
            value: finite("value", softmax_total_variation_bound(range).map_err(bound)?)?,
        },
        LogitBoundRequest::PayloadDiameter { payload_map, radius } => LogitBoundsReport::Value {
            value: finite("value", payload_diameter(matrix(tensors, &payload_map)?, radius).map_err(bound)?)?,
        },
        LogitBoundRequest::AttentionRead { heads } => {
            let heads: Vec<HeadRead> = heads
                .iter()
                .map(|head| HeadRead {
                    total_variation: head.total_variation,
                    diameter: head.diameter,
                })
                .collect();
            let status = attention_read_bound(&heads).map_err(bound)?;
            LogitBoundsReport::AttentionRead {
                status: EvidenceStatusWire::from_status_with(status, |w| w, |AttentionReadRegion { heads }| {
                    AttentionReadRegionReport { heads }
                })?,
            }
        }
    };
    Ok(super::output(MpdResult::LogitBounds(report), BTreeMap::new()))
}

fn per_row(
    boxes: &LogitBoxes,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    bound: impl Fn(
        ndarray::ArrayView1<'_, f64>,
        ndarray::ArrayView1<'_, f64>,
        ndarray::ArrayView1<'_, f64>,
        ndarray::ArrayView1<'_, f64>,
    ) -> Result<BoxStatusWire, MpdSurfaceError>,
) -> Result<LogitBoundsReport, MpdSurfaceError> {
    let sides = [
        (&boxes.reference, rows(tensors, &boxes.reference)?),
        (&boxes.reference_radius, rows(tensors, &boxes.reference_radius)?),
        (&boxes.perturbed, rows(tensors, &boxes.perturbed)?),
        (&boxes.perturbed_radius, rows(tensors, &boxes.perturbed_radius)?),
    ];
    let count = sides[0].1.nrows();
    if let Some((id, side)) = sides.iter().find(|(_, side)| side.nrows() != count) {
        return Err(MpdSurfaceError::TensorShape {
            tensor: (*id).clone(),
            reason: format!("{} rows for {count} reference rows", side.nrows()),
        });
    }
    let statuses = (0..count)
        .map(|row| bound(sides[0].1.row(row), sides[1].1.row(row), sides[2].1.row(row), sides[3].1.row(row)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LogitBoundsReport::Boxes { rows: statuses })
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::test_support::test_governor;
    use ndarray::{Array2, array};

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("p".to_string(), array![[1.0, 0.5, -0.25], [0.0, 2.0, 1.0]].into_dyn()),
            ("q".to_string(), array![[1.25, 0.5, -0.5], [0.5, 1.5, 1.0]].into_dyn()),
            ("r".to_string(), Array2::<f64>::from_elem((2, 3), 1e-3).into_dyn()),
            ("c".to_string(), array![[3.0, 4.0]].into_dyn()),
        ])
    }

    fn run(bound: &str, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
        run_parameter_decomposition(
            &request_json(&format!(r#"{{"kind": "logit_bounds", "bound": {bound}}}"#)),
            tensors,
            test_governor(),
        )
    }

    fn report(output: &MpdOutput) -> &LogitBoundsReport {
        let MpdResult::LogitBounds(report) = &output.report.result else {
            panic!("expected logit bounds, got {:?}", output.report.result);
        };
        report
    }

    const BOXES: &str = r#"{"reference": "p", "reference_radius": "r", "perturbed": "q", "perturbed_radius": "r"}"#;

    #[test]
    fn logit_bound_reports_are_the_owner_results() {
        let tensors = tensors();
        let m = |id: &str| matrix(&tensors, id).expect("matrix");
        for (kind, which) in [("kl", 0), ("kl_supremum", 1), ("total_variation", 2)] {
            let output = run(&format!(r#"{{"kind": "{kind}", "boxes": {BOXES}}}"#), &tensors).expect("surface run");
            let LogitBoundsReport::Boxes { rows } = report(&output) else {
                panic!("expected per-row statuses");
            };
            assert_eq!(rows.len(), 2);
            let (pm, qm, rm) = (m("p"), m("q"), m("r"));
            for (row, wire) in rows.iter().enumerate() {
                let (p, q, r) = (pm.row(row), qm.row(row), rm.row(row));
                let expected = match which {
                    0 => EvidenceStatusWire::from_status_with(kl_over_logit_boxes(p, r, q, r).expect("owner"), |w| w, kl_region),
                    1 => EvidenceStatusWire::from_status_with(kl_supremum_over_logit_boxes(p, r, q, r).expect("owner"), |w| w, kl_region),
                    _ => EvidenceStatusWire::from_status_with(total_variation_over_logit_boxes(p, r, q, r).expect("owner"), |w| w, tv_region),
                }
                .expect("wire");
                assert_eq!(wire, &expected);
            }
        }
        let output = run(r#"{"kind": "softmax_total_variation", "range": 2.0}"#, &tensors).expect("surface run");
        assert_eq!(report(&output), &LogitBoundsReport::Value { value: softmax_total_variation_bound(2.0).expect("owner") });
        let output = run(r#"{"kind": "payload_diameter", "payload_map": "c", "radius": 1.5}"#, &tensors).expect("surface run");
        assert_eq!(report(&output), &LogitBoundsReport::Value { value: payload_diameter(m("c"), 1.5).expect("owner") });
        let output = run(
            r#"{"kind": "attention_read", "heads": [{"total_variation": 0.25, "diameter": 2.0}, {"total_variation": 0.5, "diameter": 1.0}]}"#,
            &tensors,
        )
        .expect("surface run");
        let owner = attention_read_bound(&[
            HeadRead { total_variation: 0.25, diameter: 2.0 },
            HeadRead { total_variation: 0.5, diameter: 1.0 },
        ])
        .expect("owner");
        let LogitBoundsReport::AttentionRead { status } = report(&output) else {
            panic!("expected an attention read bound");
        };
        assert!(matches!(status, EvidenceStatusWire::UniformBound { upper, .. } if *upper == owner.upper_bound().expect("upper")));
        assert!(owner.upper_bound().is_some_and(|upper| upper >= 1.0));
    }

    #[test]
    fn a_bound_the_owner_refuses_reaches_the_caller() {
        let mut tensors = tensors();
        assert!(run(&format!(r#"{{"kind": "kl", "boxes": {BOXES}}}"#), &tensors).is_ok());
        assert!(matches!(
            run(r#"{"kind": "softmax_total_variation", "range": -1.0}"#, &tensors),
            Err(MpdSurfaceError::Bound(_))
        ));
        assert!(matches!(
            run(r#"{"kind": "attention_read", "heads": [{"total_variation": 1.5, "diameter": 1.0}]}"#, &tensors),
            Err(MpdSurfaceError::Bound(_))
        ));
        tensors.insert("r".to_string(), Array2::<f64>::from_elem((1, 3), 1e-3).into_dyn());
        assert!(matches!(
            run(&format!(r#"{{"kind": "kl", "boxes": {BOXES}}}"#), &tensors),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        assert!(matches!(
            run(r#"{"kind": "softmax_total_variation", "range": 1.0, "scale": 2}"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
