//! Typed difference-smooth contrast and report orchestration.
//!
//! Front ends provide saved-model schema/term metadata and one design-builder
//! callback. This module owns grid construction, group/pair resolution, row
//! assembly, nuisance-column policy, contrast orientation, covariance-band
//! configuration, and final report rows.

use crate::effects::{
    self, BandOptions, CovarianceSource, PointwiseBandOptions, SimultaneousBandOptions,
};
use gam_data::{ColumnKindTag, DataSchema};
use gam_terms::smooth::TermCollectionSpec;
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DifferenceSmoothRequest {
    pub view: String,
    pub group: Option<String>,
    pub pairs: Option<Vec<(String, String)>>,
    pub n: usize,
    pub level: Option<f64>,
    pub simultaneous: bool,
    pub n_sim: Option<usize>,
    pub seed: Option<u64>,
    pub marginalise_random: bool,
    pub group_means: bool,
    pub template: Option<BTreeMap<String, String>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DifferenceSmoothRow {
    #[serde(flatten)]
    pub view_value: BTreeMap<String, f64>,
    pub group: String,
    pub level_1: String,
    pub level_2: String,
    pub diff: f64,
    pub se: f64,
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
    pub simultaneous: bool,
    pub critical: f64,
    pub covariance_kind: String,
    pub covariance_corrected: bool,
}

pub struct DifferenceSmoothInputs<'a> {
    pub schema: &'a DataSchema,
    pub training_feature_ranges: &'a [(f64, f64)],
    pub termspec: &'a TermCollectionSpec,
    pub beta: ArrayView1<'a, f64>,
    pub covariance: ArrayView2<'a, f64>,
    pub covariance_source: CovarianceSource,
}

pub fn difference_smooth_report(
    inputs: DifferenceSmoothInputs<'_>,
    request: DifferenceSmoothRequest,
    mut build_design: impl FnMut(&[String], &[Vec<String>]) -> Result<Array2<f64>, String>,
) -> Result<Vec<DifferenceSmoothRow>, String> {
    let level = request.level.unwrap_or(effects::DEFAULT_BAND_LEVEL);
    if !(0.0 < level && level < 1.0) {
        return Err("difference_smooth level must be in (0, 1)".to_string());
    }
    if request.n < 2 {
        return Err("difference_smooth n must be at least 2".to_string());
    }
    if inputs.schema.columns.len() != inputs.training_feature_ranges.len() {
        return Err(format!(
            "difference_smooth schema/range mismatch: {} columns but {} training ranges",
            inputs.schema.columns.len(),
            inputs.training_feature_ranges.len()
        ));
    }

    let headers: Vec<String> = inputs
        .schema
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect();
    let view_idx = headers
        .iter()
        .position(|name| name == &request.view)
        .ok_or_else(|| {
            format!(
                "view column {:?} not found in model schema: {:?}",
                request.view, headers
            )
        })?;
    let group = match request.group {
        Some(group) => group,
        None => inputs
            .schema
            .columns
            .iter()
            .find(|column| column.kind == ColumnKindTag::Categorical && column.name != request.view)
            .map(|column| column.name.clone())
            .ok_or_else(|| {
                "difference_smooth could not infer a categorical group column; pass group="
                    .to_string()
            })?,
    };
    let group_column = inputs
        .schema
        .columns
        .iter()
        .find(|column| column.name == group)
        .ok_or_else(|| format!("group column {group:?} not found in model schema: {headers:?}"))?;
    if group_column.levels.len() < 2 {
        return Err(format!(
            "group column {group:?} must have at least two saved levels"
        ));
    }
    let pairs = request.pairs.unwrap_or_else(|| {
        let mut pairs = Vec::new();
        for left in 0..group_column.levels.len() {
            for right in (left + 1)..group_column.levels.len() {
                pairs.push((
                    group_column.levels[left].clone(),
                    group_column.levels[right].clone(),
                ));
            }
        }
        pairs
    });
    if pairs.is_empty() {
        return Err("difference_smooth requires at least one level pair".to_string());
    }
    for (level_1, level_2) in &pairs {
        if level_1 == level_2 {
            return Err(format!(
                "difference_smooth pair levels must differ; got {level_1:?} twice"
            ));
        }
        for level in [level_1, level_2] {
            if !group_column.levels.contains(level) {
                return Err(format!(
                    "difference_smooth level {level:?} is not saved for group {group:?}"
                ));
            }
        }
    }

    let (lo, hi) = inputs.training_feature_ranges[view_idx];
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(format!(
            "difference_smooth view range for {:?} must be finite and increasing; got ({lo:?}, {hi:?})",
            request.view
        ));
    }
    let step = (hi - lo) / (request.n - 1) as f64;
    let grid: Vec<f64> = (0..request.n)
        .map(|index| lo + step * index as f64)
        .collect();
    let template = complete_template(
        request.template.unwrap_or_default(),
        inputs.schema,
        inputs.training_feature_ranges,
    )?;
    let (random_ranges, group_ranges) = random_effect_ranges(inputs.termspec, &group)?;
    let band_options = if request.simultaneous {
        BandOptions::Simultaneous(SimultaneousBandOptions {
            level,
            simulations: request.n_sim.unwrap_or(effects::DEFAULT_SIMULATIONS),
            seed: request.seed.unwrap_or(effects::DEFAULT_SIMULATION_SEED),
        })
    } else {
        BandOptions::Pointwise(PointwiseBandOptions { level })
    };
    let covariance_kind = inputs.covariance_source.to_string();
    let covariance_corrected = inputs.covariance_source == CovarianceSource::SmoothingCorrected;
    let mut output = Vec::with_capacity(pairs.len() * grid.len());

    for (level_1, level_2) in pairs {
        let rows_left = contrast_rows(&headers, &template, &request.view, &group, &level_1, &grid);
        let rows_right = contrast_rows(&headers, &template, &request.view, &group, &level_2, &grid);
        let left = build_design(&headers, &rows_left)?;
        let right = build_design(&headers, &rows_right)?;
        if left.raw_dim() != right.raw_dim() {
            return Err(format!(
                "difference_smooth candidate designs disagree in shape: {:?} vs {:?}",
                left.raw_dim(),
                right.raw_dim()
            ));
        }
        // Pair orientation is level_1 - level_2, matching the row labels.
        let mut contrast = &left - &right;
        if request.marginalise_random {
            let ranges = if request.group_means {
                subtract_ranges(&random_ranges, &group_ranges)
            } else {
                random_ranges.clone()
            };
            zero_ranges(&mut contrast, &ranges)?;
        }
        if !request.group_means {
            zero_ranges(&mut contrast, &group_ranges)?;
        }
        let report = effects::effect_report(
            inputs.beta,
            inputs.covariance,
            contrast.view(),
            band_options,
        )
        .map_err(|error| error.to_string())?;
        for (index, &x) in grid.iter().enumerate() {
            output.push(DifferenceSmoothRow {
                view_value: BTreeMap::from([(request.view.clone(), x)]),
                group: group.clone(),
                level_1: level_1.clone(),
                level_2: level_2.clone(),
                diff: report.center[index],
                se: report.se[index],
                lower: report.lower[index],
                upper: report.upper[index],
                level,
                simultaneous: request.simultaneous,
                critical: report.critical,
                covariance_kind: covariance_kind.clone(),
                covariance_corrected,
            });
        }
    }
    Ok(output)
}

fn complete_template(
    mut template: BTreeMap<String, String>,
    schema: &DataSchema,
    ranges: &[(f64, f64)],
) -> Result<BTreeMap<String, String>, String> {
    for (index, column) in schema.columns.iter().enumerate() {
        if template.contains_key(&column.name) {
            continue;
        }
        let value = match column.kind {
            ColumnKindTag::Categorical => column.levels.first().cloned().ok_or_else(|| {
                format!("categorical column {:?} has no saved levels", column.name)
            })?,
            ColumnKindTag::Binary => "0".to_string(),
            ColumnKindTag::Continuous => {
                let (lo, hi) = ranges[index];
                if !(lo.is_finite() && hi.is_finite()) {
                    return Err(format!(
                        "training range for {:?} must be finite",
                        column.name
                    ));
                }
                (0.5 * (lo + hi)).to_string()
            }
        };
        template.insert(column.name.clone(), value);
    }
    Ok(template)
}

fn contrast_rows(
    headers: &[String],
    template: &BTreeMap<String, String>,
    view: &str,
    group: &str,
    level: &str,
    grid: &[f64],
) -> Vec<Vec<String>> {
    grid.iter()
        .map(|x| {
            headers
                .iter()
                .map(|header| {
                    if header == view {
                        x.to_string()
                    } else if header == group {
                        level.to_string()
                    } else {
                        template
                            .get(header)
                            .expect("complete template must contain every header")
                            .clone()
                    }
                })
                .collect()
        })
        .collect()
}

fn random_effect_ranges(
    termspec: &TermCollectionSpec,
    group: &str,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    let mut column = 1 + termspec.linear_terms.len();
    let mut all = Vec::with_capacity(termspec.random_effect_terms.len());
    let mut selected = Vec::new();
    for term in &termspec.random_effect_terms {
        let levels = term.frozen_levels.as_ref().ok_or_else(|| {
            format!(
                "difference_smooth random effect {:?} has no frozen levels",
                term.name
            )
        })?;
        let range = (column, column + levels.len());
        all.push(range);
        if term.name == group {
            selected.push(range);
        }
        column += levels.len();
    }
    Ok((all, selected))
}

fn subtract_ranges(ranges: &[(usize, usize)], excluded: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut output = Vec::new();
    for &(start, end) in ranges {
        let mut segments = vec![(start, end)];
        for &(excluded_start, excluded_end) in excluded {
            let mut next = Vec::new();
            for (segment_start, segment_end) in segments {
                if excluded_end <= segment_start || excluded_start >= segment_end {
                    next.push((segment_start, segment_end));
                } else {
                    if segment_start < excluded_start {
                        next.push((segment_start, excluded_start));
                    }
                    if excluded_end < segment_end {
                        next.push((excluded_end, segment_end));
                    }
                }
            }
            segments = next;
        }
        output.extend(segments.into_iter().filter(|(start, end)| start < end));
    }
    output
}

fn zero_ranges(design: &mut Array2<f64>, ranges: &[(usize, usize)]) -> Result<(), String> {
    for &(start, end) in ranges {
        if start > end || end > design.ncols() {
            return Err(format!(
                "difference_smooth design range {start}..{end} exceeds {} columns",
                design.ncols()
            ));
        }
        if start < end {
            design.slice_mut(s![.., start..end]).fill(0.0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::SchemaColumn;
    use ndarray::{Array1, array};

    #[test]
    fn pair_orientation_and_report_are_owned_by_core() {
        let schema = DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                },
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["A".to_string(), "B".to_string()],
                },
            ],
        };
        let request = DifferenceSmoothRequest {
            view: "x".to_string(),
            group: Some("g".to_string()),
            pairs: Some(vec![("B".to_string(), "A".to_string())]),
            n: 2,
            level: Some(0.95),
            simultaneous: false,
            n_sim: None,
            seed: None,
            marginalise_random: false,
            group_means: true,
            template: None,
        };
        let termspec = TermCollectionSpec {
            linear_terms: Vec::new(),
            smooth_terms: Vec::new(),
            random_effect_terms: Vec::new(),
        };
        let beta = Array1::from_vec(vec![0.0, 1.5]);
        let covariance = array![[0.1, 0.0], [0.0, 0.1]];
        let rows = difference_smooth_report(
            DifferenceSmoothInputs {
                schema: &schema,
                training_feature_ranges: &[(0.0, 1.0), (0.0, 1.0)],
                termspec: &termspec,
                beta: beta.view(),
                covariance: covariance.view(),
                covariance_source: CovarianceSource::Conditional,
            },
            request,
            |_headers, rows| {
                Ok(Array2::from_shape_fn((rows.len(), 2), |(row, column)| {
                    if column == 0 {
                        1.0
                    } else if rows[row][1] == "B" {
                        1.0
                    } else {
                        0.0
                    }
                }))
            },
        )
        .expect("difference report");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| (row.diff - 1.5).abs() < 1.0e-12));
    }
}
