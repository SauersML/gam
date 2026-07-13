//! Canonical log-slope channel layout.
//!
//! The coefficient block owns one current-coordinate vector, but a vector
//! latent score owns one physical log-slope channel per score coordinate.
//! Raw coefficient ranges are therefore construction metadata only: after a
//! coefficient transform, every physical channel may depend on every current
//! coefficient.  This module keeps the raw partition and the raw-to-current
//! map together and emits full-width current-coordinate channel rows.

use super::*;

const CHANNEL_SCAN_ROWS: usize = 256;

#[derive(Clone)]
pub(crate) enum LogslopeTopology {
    Shared,
    PerScore {
        raw_ranges: Arc<[std::ops::Range<usize>]>,
    },
}

impl LogslopeTopology {
    pub(crate) fn shared() -> Self {
        Self::Shared
    }

    pub(crate) fn per_score(
        raw_ranges: Vec<std::ops::Range<usize>>,
        raw_width: usize,
    ) -> Result<Self, String> {
        validate_partition(&raw_ranges, raw_width, "logslope topology")?;
        if raw_ranges.len() < 2 {
            return Err(
                "per-score logslope topology requires at least two physical channels".to_string(),
            );
        }
        if raw_ranges.iter().any(std::ops::Range::is_empty) {
            return Err(
                "per-score logslope topology contains an empty physical channel".to_string(),
            );
        }
        Ok(Self::PerScore {
            raw_ranges: raw_ranges.into(),
        })
    }

    #[inline]
    pub(crate) fn is_per_score(&self) -> bool {
        matches!(self, Self::PerScore { .. })
    }

    pub(crate) fn score_count(&self) -> usize {
        match self {
            Self::Shared => 1,
            Self::PerScore { raw_ranges } => raw_ranges.len(),
        }
    }

    pub(crate) fn materialize_identity(
        &self,
        raw_design: DesignMatrix,
        common_offset: &Array1<f64>,
    ) -> Result<LogslopeLayout, String> {
        let width = raw_design.ncols();
        self.materialize_with_design(
            raw_design.clone(),
            raw_design,
            Array2::<f64>::eye(width),
            common_offset,
        )
    }

    pub(crate) fn materialize(
        &self,
        raw_design: DesignMatrix,
        current_from_raw: Array2<f64>,
        common_offset: &Array1<f64>,
    ) -> Result<LogslopeLayout, String> {
        let coefficient_design = wrap_design_with_transform(
            raw_design.clone(),
            &current_from_raw,
            "logslope layout coefficient design",
        )?;
        self.materialize_with_design(
            raw_design,
            coefficient_design,
            current_from_raw,
            common_offset,
        )
    }

    fn materialize_with_design(
        &self,
        raw_design: DesignMatrix,
        coefficient_design: DesignMatrix,
        current_from_raw: Array2<f64>,
        common_offset: &Array1<f64>,
    ) -> Result<LogslopeLayout, String> {
        if raw_design.nrows() != common_offset.len() {
            return Err(format!(
                "logslope layout offset length {} does not match design rows {}",
                common_offset.len(),
                raw_design.nrows(),
            ));
        }
        if current_from_raw.nrows() != raw_design.ncols() {
            return Err(format!(
                "logslope layout transform has {} raw rows but design has {} columns",
                current_from_raw.nrows(),
                raw_design.ncols(),
            ));
        }
        if current_from_raw.iter().any(|value| !value.is_finite()) {
            return Err("logslope layout transform contains a non-finite value".to_string());
        }
        if common_offset.iter().any(|value| !value.is_finite()) {
            return Err("logslope layout offset contains a non-finite value".to_string());
        }

        let nrows = raw_design.nrows();
        let current_width = current_from_raw.ncols();
        if coefficient_design.nrows() != nrows || coefficient_design.ncols() != current_width {
            return Err(format!(
                "logslope current design is {}x{} but raw-design transform emits {nrows}x{current_width}",
                coefficient_design.nrows(),
                coefficient_design.ncols(),
            ));
        }
        match self {
            Self::Shared => Ok(LogslopeLayout {
                coefficient_design,
                nrows,
                current_width,
                channels: LogslopeChannels::Shared {
                    offset: Arc::new(common_offset.clone()),
                },
            }),
            Self::PerScore { raw_ranges } => {
                validate_partition(raw_ranges, raw_design.ncols(), "per-score logslope layout")?;
                for (channel, range) in raw_ranges.iter().enumerate() {
                    certify_channel_nonzero(
                        &raw_design,
                        &current_from_raw,
                        range,
                        channel,
                    )?;
                }
                let mut offsets = Array2::<f64>::zeros((nrows, raw_ranges.len()));
                for mut column in offsets.columns_mut() {
                    column.assign(common_offset);
                }
                Ok(LogslopeLayout {
                    coefficient_design,
                    nrows,
                    current_width,
                    channels: LogslopeChannels::PerScore {
                        raw_design,
                        current_from_raw: Arc::new(current_from_raw),
                        raw_ranges: Arc::clone(raw_ranges),
                        offsets: Arc::new(offsets),
                    },
                })
            }
        }
    }
}

#[derive(Clone)]
pub(crate) enum LogslopeChannels {
    Shared {
        offset: Arc<Array1<f64>>,
    },
    PerScore {
        raw_design: DesignMatrix,
        current_from_raw: Arc<Array2<f64>>,
        raw_ranges: Arc<[std::ops::Range<usize>]>,
        offsets: Arc<Array2<f64>>,
    },
}

#[derive(Clone)]
pub(crate) struct LogslopeLayout {
    coefficient_design: DesignMatrix,
    nrows: usize,
    current_width: usize,
    channels: LogslopeChannels,
}

impl LogslopeLayout {
    /// Shared-channel constructor. Production paths use
    /// [`LogslopeTopology::materialize_identity`] so topology validation stays
    /// at the construction boundary.
    pub(crate) fn shared(coefficient_design: DesignMatrix, offset: Array1<f64>) -> Self {
        let nrows = coefficient_design.nrows();
        let current_width = coefficient_design.ncols();
        Self {
            coefficient_design,
            nrows,
            current_width,
            channels: LogslopeChannels::Shared {
                offset: Arc::new(offset),
            },
        }
    }

    #[inline]
    pub(crate) fn is_per_score(&self) -> bool {
        matches!(self.channels, LogslopeChannels::PerScore { .. })
    }

    pub(crate) fn score_count(&self) -> usize {
        match &self.channels {
            LogslopeChannels::Shared { .. } => 1,
            LogslopeChannels::PerScore { raw_ranges, .. } => raw_ranges.len(),
        }
    }

    #[inline]
    pub(crate) fn coefficient_design(&self) -> &DesignMatrix {
        &self.coefficient_design
    }

    pub(crate) fn validate_for(&self, score_dim: usize) -> Result<(), String> {
        if self.coefficient_design.nrows() != self.nrows
            || self.coefficient_design.ncols() != self.current_width
        {
            return Err("logslope layout coefficient-design invariant is broken".to_string());
        }
        if let LogslopeChannels::Shared { offset } = &self.channels
            && offset.len() != self.nrows
        {
            return Err(format!(
                "shared logslope offset has length {} but layout has {} rows",
                offset.len(),
                self.nrows,
            ));
        }
        if self.is_per_score() && self.score_count() != score_dim {
            return Err(format!(
                "per-score logslope layout has {} channels but latent score has dimension {score_dim}",
                self.score_count(),
            ));
        }
        Ok(())
    }

    pub(crate) fn row_workspace(&self, score_dim: usize) -> Result<LogslopeRowWorkspace, String> {
        if self.is_per_score() && self.score_count() != score_dim {
            return Err(format!(
                "cannot build logslope row workspace: {} channels for score dimension {score_dim}",
                self.score_count(),
            ));
        }
        let raw_width = match &self.channels {
            LogslopeChannels::Shared { .. } => 0,
            LogslopeChannels::PerScore { raw_design, .. } => raw_design.ncols(),
        };
        Ok(LogslopeRowWorkspace {
            raw_row: Array2::<f64>::zeros((1, raw_width)),
            channel_rows: Array2::<f64>::zeros((score_dim, self.current_width)),
            values: vec![0.0; score_dim],
        })
    }

    pub(crate) fn fill_shared_values(
        &self,
        value: f64,
        workspace: &mut LogslopeRowWorkspace,
    ) -> Result<(), String> {
        if self.is_per_score() {
            return Err("cannot fill shared values for a per-score logslope layout".to_string());
        }
        workspace.values.fill(value);
        Ok(())
    }

    pub(crate) fn fill_per_score_row(
        &self,
        row: usize,
        beta: &Array1<f64>,
        workspace: &mut LogslopeRowWorkspace,
    ) -> Result<(), String> {
        let LogslopeChannels::PerScore {
            raw_design,
            current_from_raw,
            raw_ranges,
            offsets,
        } = &self.channels
        else {
            return Err("per-score logslope row requested from a shared layout".to_string());
        };
        if row >= self.nrows {
            return Err(format!(
                "logslope row {row} is out of bounds for {} rows",
                self.nrows
            ));
        }
        if beta.len() != self.current_width {
            return Err(format!(
                "logslope beta length {} does not match current width {}",
                beta.len(),
                self.current_width,
            ));
        }
        if workspace.raw_row.dim() != (1, raw_design.ncols())
            || workspace.channel_rows.dim() != (raw_ranges.len(), self.current_width)
            || workspace.values.len() != raw_ranges.len()
        {
            return Err("logslope row workspace shape does not match layout".to_string());
        }

        raw_design
            .row_chunk_into(row..row + 1, workspace.raw_row.view_mut())
            .map_err(|error| format!("logslope layout row materialization failed: {error}"))?;
        workspace.channel_rows.fill(0.0);
        let raw_row = workspace.raw_row.row(0);
        for (channel, range) in raw_ranges.iter().enumerate() {
            for raw_col in range.clone() {
                let value = raw_row[raw_col];
                if value == 0.0 {
                    continue;
                }
                for current_col in 0..self.current_width {
                    workspace.channel_rows[[channel, current_col]] +=
                        value * current_from_raw[[raw_col, current_col]];
                }
            }
            workspace.values[channel] =
                workspace.channel_rows.row(channel).dot(beta) + offsets[[row, channel]];
        }
        Ok(())
    }

}

#[cfg(test)]
impl From<DesignMatrix> for LogslopeLayout {
    fn from(design: DesignMatrix) -> Self {
        let nrows = design.nrows();
        Self::shared(design, Array1::<f64>::zeros(nrows))
    }
}

pub(crate) struct LogslopeRowWorkspace {
    raw_row: Array2<f64>,
    channel_rows: Array2<f64>,
    values: Vec<f64>,
}

impl LogslopeRowWorkspace {
    #[inline]
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    #[inline]
    pub(crate) fn channel_rows(&self) -> ndarray::ArrayView2<'_, f64> {
        self.channel_rows.view()
    }
}

fn validate_partition(
    ranges: &[std::ops::Range<usize>],
    width: usize,
    context: &str,
) -> Result<(), String> {
    let mut start = 0usize;
    for (channel, range) in ranges.iter().enumerate() {
        if range.start != start || range.end < range.start || range.end > width {
            return Err(format!(
                "{context}: malformed channel {channel} range {range:?}; expected a contiguous range starting at {start} within 0..{width}",
            ));
        }
        start = range.end;
    }
    if start != width {
        return Err(format!(
            "{context}: channel ranges end at {start}, expected raw width {width}"
        ));
    }
    Ok(())
}

fn certify_channel_nonzero(
    raw_design: &DesignMatrix,
    current_from_raw: &Array2<f64>,
    range: &std::ops::Range<usize>,
    channel: usize,
) -> Result<(), String> {
    if range.is_empty() || current_from_raw.ncols() == 0 {
        return Err(format!(
            "per-score logslope channel {channel} has no current-coordinate derivative"
        ));
    }
    let mut chunk = Array2::<f64>::zeros((CHANNEL_SCAN_ROWS.min(raw_design.nrows()), raw_design.ncols()));
    for start in (0..raw_design.nrows()).step_by(CHANNEL_SCAN_ROWS) {
        let end = (start + CHANNEL_SCAN_ROWS).min(raw_design.nrows());
        let rows = end - start;
        if chunk.nrows() != rows {
            chunk = Array2::<f64>::zeros((rows, raw_design.ncols()));
        }
        raw_design
            .row_chunk_into(start..end, chunk.view_mut())
            .map_err(|error| format!("per-score logslope channel scan failed: {error}"))?;
        for local_row in 0..rows {
            for current_col in 0..current_from_raw.ncols() {
                let mut value = 0.0;
                for raw_col in range.clone() {
                    value += chunk[[local_row, raw_col]]
                        * current_from_raw[[raw_col, current_col]];
                }
                if !value.is_finite() {
                    return Err(format!(
                        "per-score logslope channel {channel} produced a non-finite current-coordinate row"
                    ));
                }
                if value != 0.0 {
                    return Ok(());
                }
            }
        }
    }
    Err(format!(
        "per-score logslope channel {channel} is identically zero after the coefficient transform"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_close(left: &Array2<f64>, right: &Array2<f64>) {
        assert_eq!(left.dim(), right.dim());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!((lhs - rhs).abs() <= 1e-12, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn unequal_raw_widths_emit_full_width_channel_rows_and_offsets() {
        let raw = array![[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]];
        let topology = LogslopeTopology::per_score(vec![0..1, 1..3], 3).unwrap();
        let layout = topology
            .materialize_identity(DesignMatrix::from(raw), &array![0.5, -0.25])
            .unwrap();
        let beta = array![17.0, 19.0, 23.0];
        let mut workspace = layout.row_workspace(2).unwrap();
        layout
            .fill_per_score_row(0, &beta, &mut workspace)
            .unwrap();

        assert_eq!(workspace.channel_rows(), array![[2.0, 0.0, 0.0], [0.0, 3.0, 5.0]]);
        assert_eq!(workspace.values(), &[34.5, 172.5]);
    }

    #[test]
    fn transformed_channels_preserve_value_gradient_and_hessian_pullback() {
        let raw = array![[2.0, 3.0, 5.0, 7.0]];
        let current_from_raw = array![
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.25],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0]
        ];
        let topology = LogslopeTopology::per_score(vec![0..1, 1..4], 4).unwrap();
        let layout = topology
            .materialize(
                DesignMatrix::from(raw.clone()),
                current_from_raw.clone(),
                &array![0.75],
            )
            .unwrap();
        let theta = array![11.0, 13.0, 17.0];
        let beta_raw = current_from_raw.dot(&theta);
        let raw_channels = array![[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 5.0, 7.0]];
        let expected_current_channels = raw_channels.dot(&current_from_raw);
        let mut workspace = layout.row_workspace(2).unwrap();
        layout
            .fill_per_score_row(0, &theta, &mut workspace)
            .unwrap();
        assert_close(
            &workspace.channel_rows().to_owned(),
            &expected_current_channels,
        );
        let expected_values = raw_channels.dot(&beta_raw) + 0.75;
        for (actual, expected) in workspace.values().iter().zip(expected_values.iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }

        let channel_gradient = array![0.25, -1.5];
        let channel_hessian = array![[2.0, 0.5], [0.5, 3.0]];
        let current_gradient = expected_current_channels.t().dot(&channel_gradient);
        let raw_gradient = raw_channels.t().dot(&channel_gradient);
        let expected_gradient = current_from_raw.t().dot(&raw_gradient);
        for (actual, expected) in current_gradient.iter().zip(expected_gradient.iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        let current_hessian = expected_current_channels
            .t()
            .dot(&channel_hessian)
            .dot(&expected_current_channels);
        let raw_hessian = raw_channels
            .t()
            .dot(&channel_hessian)
            .dot(&raw_channels);
        let expected_hessian = current_from_raw
            .t()
            .dot(&raw_hessian)
            .dot(&current_from_raw);
        assert_close(&current_hessian, &expected_hessian);
    }

    #[test]
    fn transformed_zero_physical_channel_is_rejected_exactly() {
        let topology = LogslopeTopology::per_score(vec![0..1, 1..2], 2).unwrap();
        let error = topology
            .materialize(
                DesignMatrix::from(array![[1.0, 2.0]]),
                array![[1.0], [0.0]],
                &array![0.0],
            )
            .err()
            .expect("second physical channel must be rejected");
        assert!(error.contains("channel 1 is identically zero"), "{error}");
    }
}
