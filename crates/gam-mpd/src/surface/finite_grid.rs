//! `finite_grid`: the exhaustive FANOVA of a response on a declared finite product grid
//! ([`crate::finite_grid`]) and its pairwise screen
//! ([`gam_sae::response::interaction`]) on the wire.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output, reserve};
use crate::finite_grid::{
    AdditiveAcrossPair, FiniteGridResponse, Rectangle, RectangleComplement, centring_factor,
};
use gam_sae::response::interaction::{TotalInteractions, total_interactions};
use gam_sae::response::subspace::BandedEnergy;

/// A response tabulated on a declared product grid, and what to read off it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteGridRequest {
    /// Id of the responses `F` (`cells × outputs`).
    pub responses: String,
    /// The level count of each port.
    pub levels: Vec<usize>,
    /// Which cell each response row is.
    pub cells: GridCells,
    /// The output metric's factor `R`, `M = RᵀR`.
    pub output_factor: OutputFactor,
    /// A declared change of port frame, applied before every reading.
    pub reindex: Option<GridReindex>,
    /// Port pairs whose worst-case complement of additivity is wanted.
    pub pairs: Vec<[usize; 2]>,
    /// Port partitions whose exact cross-block energy is wanted.
    pub partitions: Vec<Vec<Vec<usize>>>,
}

/// The cell of each response row.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GridCells {
    /// Rows are row-major over `levels`, last port fastest
    /// (`FiniteGridResponse::new`).
    RowMajor {},
    /// Row `r` is the cell `points[r]`; refused unless every cell appears exactly once
    /// (`FiniteGridResponse::from_points`).
    Points { points: Vec<Vec<usize>> },
}

/// The output factor `R`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputFactor {
    /// Raw outputs.
    Identity {},
    /// `I − 11ᵀ/n` (`finite_grid::centring_factor`): logits up to a shift.
    Centring {},
    /// A declared factor (`k × outputs`).
    Declared { tensor: String },
}

/// `FiniteGridResponse::reindex` under a declared cell map.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GridReindex {
    /// The new grid's level counts.
    pub levels: Vec<usize>,
    /// `images[c]` is the new cell of old cell `c` (row-major over the old levels).
    pub images: Vec<Vec<usize>>,
}

/// The screen and the requested readings.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FiniteGridReport {
    /// The grid every reading is on (after any re-indexing).
    pub levels: Vec<usize>,
    pub cells: usize,
    pub outputs: usize,
    /// `V([p])`.
    pub total_variance: BandedEnergyReport,
    /// Ids of `I_ij` (`p × p`, total effects on the diagonal) and of its bands.
    pub interactions: String,
    pub interaction_bands: String,
    /// The finest additive blocks (`TotalInteractions::additive_blocks`).
    pub additive_blocks: Vec<Vec<usize>>,
    pub cross_block: Vec<CrossBlockReport>,
    pub rectangle_complements: Vec<RectangleComplementReport>,
}

/// [`BandedEnergy`] on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BandedEnergyReport {
    pub value: f64,
    pub band: f64,
}

/// One partition's cross-block energy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CrossBlockReport {
    pub partition: Vec<Vec<usize>>,
    /// `V([p]) − Σ_B V(B)` (`TotalInteractions::cross_block_energy`).
    pub energy: BandedEnergyReport,
    /// `Σ_{cross pairs} I_ij`, an upper bound (`TotalInteractions::cross_block_bound`).
    pub bound: BandedEnergyReport,
}

/// [`Rectangle`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RectangleReport {
    pub corner: Vec<usize>,
    pub anchor: [usize; 2],
    pub output: usize,
    pub difference: f64,
    pub band: f64,
}

/// [`AdditiveAcrossPair`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AdditiveAcrossPairReport {
    pub levels: Vec<usize>,
    pub pair: [usize; 2],
}

/// [`RectangleComplement`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RectangleComplementReport {
    pub pair: [usize; 2],
    pub witness: Option<RectangleReport>,
    pub max_difference: f64,
    pub uniform_band: f64,
    pub anova_residual_sup: f64,
    pub anova_upper: f64,
    /// `ε* = inf_{g additive across the pair} sup|G − g|`.
    pub evidence: EvidenceStatusWire<RectangleReport, AdditiveAcrossPairReport>,
    /// Id of the additive reconstruction (`cells × outputs`), present when every
    /// rectangle vanishes within the uniform band.
    pub reconstruction: Option<String>,
}

fn energy(field: &'static str, energy: BandedEnergy) -> Result<BandedEnergyReport, MpdSurfaceError> {
    Ok(BandedEnergyReport {
        value: finite(field, energy.value)?,
        band: finite(field, energy.band)?,
    })
}

fn rectangle(rectangle: Rectangle) -> RectangleReport {
    RectangleReport {
        corner: rectangle.corner,
        anchor: [rectangle.anchor.0, rectangle.anchor.1],
        output: rectangle.output,
        difference: rectangle.difference,
        band: rectangle.band,
    }
}

fn domain(domain: AdditiveAcrossPair) -> AdditiveAcrossPairReport {
    AdditiveAcrossPairReport {
        levels: domain.levels,
        pair: [domain.pair.0, domain.pair.1],
    }
}

/// Row-major index of `cell` over `levels`.
fn row_major(levels: &[usize], cell: &[usize]) -> usize {
    cell.iter().zip(levels).fold(0, |index, (&level, &count)| index * count + level)
}

pub(super) fn run(
    request: FiniteGridRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let responses = matrix(tensors, &request.responses)?;
    let grid_error = MpdSurfaceError::FiniteGrid;
    let centring;
    let factor = match &request.output_factor {
        OutputFactor::Identity {} => None,
        OutputFactor::Centring {} => {
            let outputs = responses.ncols();
            let formed = reserve(governor, outputs, outputs, 1, "finite grid: centring factor")?;
            centring = (centring_factor(outputs), formed);
            Some(centring.0.view())
        }
        OutputFactor::Declared { tensor } => Some(matrix(tensors, tensor)?),
    };
    let levels = request.levels.clone();
    let response = match &request.cells {
        GridCells::RowMajor {} => FiniteGridResponse::new(levels, responses, factor, governor),
        GridCells::Points { points } => {
            FiniteGridResponse::from_points(levels, points, responses, factor, governor)
        }
    }
    .map_err(grid_error)?;
    let response = match &request.reindex {
        None => response,
        Some(reindex) => {
            if reindex.images.len() != response.cells() {
                return Err(MpdSurfaceError::InvalidRequest(format!(
                    "reindex declares {} images for a grid of {} cells",
                    reindex.images.len(),
                    response.cells()
                )));
            }
            let old = request.levels.clone();
            response
                .reindex(
                    reindex.levels.clone(),
                    |cell| reindex.images[row_major(&old, cell)].clone(),
                    governor,
                )
                .map_err(grid_error)?
        }
    };
    let screen = total_interactions(&response).map_err(MpdSurfaceError::Interaction)?;
    project(&response, &screen, &request.pairs, &request.partitions)
}

pub(super) fn project(
    response: &FiniteGridResponse,
    screen: &TotalInteractions,
    pairs: &[[usize; 2]],
    partitions: &[Vec<Vec<usize>>],
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = BTreeMap::new();
    let (values, bands) = screen.matrix();
    arrays.insert("interactions/values".to_string(), values.into_dyn());
    arrays.insert("interactions/bands".to_string(), bands.into_dyn());
    let mut cross_block = Vec::with_capacity(partitions.len());
    for partition in partitions {
        cross_block.push(CrossBlockReport {
            partition: partition.clone(),
            energy: energy(
                "cross_block.energy",
                screen
                    .cross_block_energy(response, partition)
                    .map_err(MpdSurfaceError::Interaction)?,
            )?,
            bound: energy(
                "cross_block.bound",
                screen
                    .cross_block_bound(partition)
                    .map_err(MpdSurfaceError::Interaction)?,
            )?,
        });
    }
    let mut rectangle_complements = Vec::with_capacity(pairs.len());
    for (index, &[i, j]) in pairs.iter().enumerate() {
        let complement: RectangleComplement = response
            .rectangle_complement(i, j)
            .map_err(MpdSurfaceError::FiniteGrid)?;
        let reconstruction = complement.reconstruction.map(|reconstruction: Array2<f64>| {
            let id = format!("rectangle_complements/{index}/reconstruction");
            arrays.insert(id.clone(), reconstruction.into_dyn());
            id
        });
        rectangle_complements.push(RectangleComplementReport {
            pair: [complement.pair.0, complement.pair.1],
            witness: complement.witness.map(rectangle),
            max_difference: finite("max_difference", complement.max_difference)?,
            uniform_band: finite("uniform_band", complement.uniform_band)?,
            anova_residual_sup: finite("anova_residual_sup", complement.anova_residual_sup)?,
            anova_upper: finite("anova_upper", complement.anova_upper)?,
            evidence: EvidenceStatusWire::from_status_with(complement.evidence, rectangle, domain)?,
            reconstruction,
        });
    }
    Ok(output(
        MpdResult::FiniteGrid(FiniteGridReport {
            levels: response.levels().to_vec(),
            cells: response.cells(),
            outputs: response.outputs(),
            total_variance: energy("total_variance", screen.total_variance())?,
            interactions: "interactions/values".to_string(),
            interaction_bands: "interactions/bands".to_string(),
            additive_blocks: screen.additive_blocks(),
            cross_block,
            rectangle_complements,
        }),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::test_support::test_governor;
    use crate::finite_grid::FiniteGridError;
    use gam_sae::response::interaction::InteractionError;

    const P: usize = 5;

    /// `F(a, b) = [cos(2π(a+b)/p), a − 2b]` on `Z_5²`, row-major: the first output couples
    /// the ports, the second is additive.
    fn modular() -> Array2<f64> {
        Array2::from_shape_fn((P * P, 2), |(cell, output)| {
            let (a, b) = (cell / P, cell % P);
            if output == 0 {
                (2.0 * std::f64::consts::PI * ((a + b) % P) as f64 / P as f64).cos()
            } else {
                a as f64 - 2.0 * b as f64
            }
        })
    }

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([("f".to_string(), modular().into_dyn())])
    }

    fn request(output_factor: &str, reindex: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "finite_grid", "responses": "f", "levels": [5, 5], "cells": {{"kind": "row_major"}},
                "output_factor": {output_factor}, "reindex": {reindex}, "pairs": [[0, 1]], "partitions": [[[0], [1]]]}}"#
        ))
    }

    fn report(output: &MpdOutput) -> &FiniteGridReport {
        let MpdResult::FiniteGrid(report) = &output.report.result else {
            panic!("expected a finite grid report, got {:?}", output.report.result);
        };
        report
    }

    fn assert_projects(output: &MpdOutput, response: &FiniteGridResponse) {
        let screen = total_interactions(response).expect("owner screen");
        let report = report(output);
        assert_eq!(report.levels, response.levels());
        assert_eq!((report.cells, report.outputs), (response.cells(), response.outputs()));
        let total = screen.total_variance();
        assert_eq!(report.total_variance, BandedEnergyReport { value: total.value, band: total.band });
        let (values, bands) = screen.matrix();
        assert_eq!(output.arrays[&report.interactions], values.into_dyn());
        assert_eq!(output.arrays[&report.interaction_bands], bands.into_dyn());
        assert_eq!(report.additive_blocks, screen.additive_blocks());
        let partition = vec![vec![0], vec![1]];
        let owner_energy = screen.cross_block_energy(response, &partition).expect("energy");
        let owner_bound = screen.cross_block_bound(&partition).expect("bound");
        assert_eq!(
            report.cross_block[0],
            CrossBlockReport {
                partition: partition.clone(),
                energy: BandedEnergyReport { value: owner_energy.value, band: owner_energy.band },
                bound: BandedEnergyReport { value: owner_bound.value, band: owner_bound.band },
            }
        );
        let owner = response.rectangle_complement(0, 1).expect("owner complement");
        let wire = &report.rectangle_complements[0];
        assert_eq!(wire.pair, [0, 1]);
        assert_eq!(wire.witness, owner.witness.clone().map(rectangle));
        assert_eq!(wire.max_difference, owner.max_difference);
        assert_eq!(wire.uniform_band, owner.uniform_band);
        assert_eq!(wire.anova_residual_sup, owner.anova_residual_sup);
        assert_eq!(wire.anova_upper, owner.anova_upper);
        assert_eq!(
            wire.evidence,
            EvidenceStatusWire::from_status_with(owner.evidence.clone(), rectangle, domain).expect("wire")
        );
        assert_eq!(wire.reconstruction.is_some(), owner.reconstruction.is_some());
        if let (Some(id), Some(reconstruction)) = (&wire.reconstruction, owner.reconstruction) {
            assert_eq!(output.arrays[id], reconstruction.into_dyn());
        }
    }

    #[test]
    fn finite_grid_report_is_the_owner_result_field_for_field() {
        let output = run_parameter_decomposition(&request(r#"{"kind": "identity"}"#, "null"), &tensors(), test_governor())
            .expect("surface run");
        let response = FiniteGridResponse::new(vec![P, P], modular().view(), None, test_governor()).expect("owner grid");
        assert_projects(&output, &response);
        // The cosine of a + b couples the ports: one block and a certified complement.
        assert_eq!(report(&output).additive_blocks, vec![vec![0, 1]]);
        assert!(report(&output).rectangle_complements[0].reconstruction.is_none());

        // Under the centring factor, and after the declared frame (a, b) -> (a + b, a − b).
        let images: Vec<Vec<usize>> = (0..P * P)
            .map(|cell| {
                let (a, b) = (cell / P, cell % P);
                vec![(a + b) % P, (a + P - b) % P]
            })
            .collect();
        let reindex = format!(r#"{{"levels": [5, 5], "images": {}}}"#, serde_json::to_string(&images).expect("json"));
        let output = run_parameter_decomposition(&request(r#"{"kind": "centring"}"#, &reindex), &tensors(), test_governor())
            .expect("surface run");
        let centring = centring_factor(2);
        let native = FiniteGridResponse::new(vec![P, P], modular().view(), Some(centring.view()), test_governor())
            .expect("owner grid");
        let moved = native
            .reindex(vec![P, P], |cell| images[cell[0] * P + cell[1]].clone(), test_governor())
            .expect("owner reindex");
        assert_projects(&output, &moved);
    }

    #[test]
    fn an_additive_response_reports_its_reconstruction() {
        let additive = Array2::from_shape_fn((P * P, 1), |(cell, _)| (cell / P) as f64 - 2.0 * (cell % P) as f64);
        let tensors = BTreeMap::from([("f".to_string(), additive.clone().into_dyn())]);
        let output = run_parameter_decomposition(&request(r#"{"kind": "identity"}"#, "null"), &tensors, test_governor())
            .expect("surface run");
        let response = FiniteGridResponse::new(vec![P, P], additive.view(), None, test_governor()).expect("owner grid");
        assert_projects(&output, &response);
        let wire = &report(&output).rectangle_complements[0];
        assert!(matches!(wire.evidence, EvidenceStatusWire::Exact { value: 0.0, .. }));
        assert_eq!(output.arrays[wire.reconstruction.as_ref().expect("reconstruction")], additive.into_dyn());
        assert_eq!(report(&output).additive_blocks, vec![vec![0], vec![1]]);
    }

    #[test]
    fn a_finite_grid_request_the_owners_refuse_reaches_the_caller() {
        assert!(run_parameter_decomposition(&request(r#"{"kind": "identity"}"#, "null"), &tensors(), test_governor()).is_ok());
        // Two points on the same cell: not the product domain.
        let mut points: Vec<Vec<usize>> = (0..P * P).map(|cell| vec![cell / P, cell % P]).collect();
        points[1] = vec![0, 0];
        let duplicated = request(r#"{"kind": "identity"}"#, "null").replace(
            r#"{"kind": "row_major"}"#,
            &format!(r#"{{"kind": "points", "points": {}}}"#, serde_json::to_string(&points).expect("json")),
        );
        assert!(matches!(
            run_parameter_decomposition(&duplicated, &tensors(), test_governor()),
            Err(MpdSurfaceError::FiniteGrid(FiniteGridError::NonProductDomain { .. }))
        ));
        // (a, b) -> (a + b, a + b) is not a bijection.
        let collapsed: Vec<Vec<usize>> = (0..P * P).map(|cell| vec![(cell / P + cell % P) % P, 0]).collect();
        let reindex = format!(r#"{{"levels": [5, 5], "images": {}}}"#, serde_json::to_string(&collapsed).expect("json"));
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "identity"}"#, &reindex), &tensors(), test_governor()),
            Err(MpdSurfaceError::FiniteGrid(FiniteGridError::NotABijection { .. }))
        ));
        let short = r#"{"levels": [5, 5], "images": [[0, 0]]}"#;
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "identity"}"#, short), &tensors(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let wrong_levels = request(r#"{"kind": "identity"}"#, "null").replace("[5, 5]", "[5, 4]");
        assert!(matches!(
            run_parameter_decomposition(&wrong_levels, &tensors(), test_governor()),
            Err(MpdSurfaceError::FiniteGrid(FiniteGridError::RowCountMismatch { .. }))
        ));
        let bad_pair = request(r#"{"kind": "identity"}"#, "null").replace("[[0, 1]]", "[[1, 1]]");
        assert!(matches!(
            run_parameter_decomposition(&bad_pair, &tensors(), test_governor()),
            Err(MpdSurfaceError::FiniteGrid(FiniteGridError::InvalidPair { .. }))
        ));
        let bad_partition = request(r#"{"kind": "identity"}"#, "null").replace("[[[0], [1]]]", "[[[0]]]");
        assert!(matches!(
            run_parameter_decomposition(&bad_partition, &tensors(), test_governor()),
            Err(MpdSurfaceError::Interaction(InteractionError::InvalidPartition { .. }))
        ));
        let stray = request(r#"{"kind": "identity", "scale": 2}"#, "null");
        assert!(matches!(
            run_parameter_decomposition(&stray, &tensors(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
