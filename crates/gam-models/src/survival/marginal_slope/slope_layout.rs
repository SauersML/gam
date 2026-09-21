//! Canonical slope channel layout.
//!
//! The coefficient block owns one current-coordinate vector, but a vector
//! latent score owns one physical slope channel per score coordinate.
//! Raw coefficient ranges are therefore construction metadata only: after a
//! coefficient transform, every physical channel may depend on every current
//! coefficient.  This module keeps the raw partition and the raw-to-current
//! map together and emits full-width current-coordinate channel rows.

use super::*;

const CHANNEL_SCAN_ROWS: usize = 256;

#[derive(Clone)]
pub(crate) enum SlopeTopology {
    Shared,
    PerScore {
        raw_ranges: Arc<[std::ops::Range<usize>]>,
    },
}

impl SlopeTopology {
    pub(crate) fn shared() -> Self {
        Self::Shared
    }

    pub(crate) fn per_score(
        raw_ranges: Vec<std::ops::Range<usize>>,
        raw_width: usize,
    ) -> Result<Self, String> {
        validate_partition(&raw_ranges, raw_width, "slope topology")?;
        if raw_ranges.len() < 2 {
            return Err(
                "per-score slope topology requires at least two physical channels".to_string(),
            );
        }
        if raw_ranges.iter().any(std::ops::Range::is_empty) {
            return Err(
                "per-score slope topology contains an empty physical channel".to_string(),
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
    ) -> Result<SlopeLayout, String> {
        let mut channel_offsets = Array2::<f64>::zeros((common_offset.len(), self.score_count()));
        for mut column in channel_offsets.columns_mut() {
            column.assign(common_offset);
        }
        self.materialize_identity_with_channel_offsets(raw_design, channel_offsets)
    }

    /// Materialise the identity-coordinate layout with one offset column per
    /// physical channel. Column `j` is the fixed part of channel `j`'s slope,
    /// `s_j(x) = X_j(x) beta_j + offset_{., j}`. Distinct columns arise when
    /// the score axes carry distinct unit maps (gam#4331): a raw slope offset
    /// `o` on score `j` standardised by scale `c_j` is the standardised-axis
    /// offset `c_j o`, which differs across channels.
    pub(crate) fn materialize_identity_with_channel_offsets(
        &self,
        raw_design: DesignMatrix,
        channel_offsets: Array2<f64>,
    ) -> Result<SlopeLayout, String> {
        let width = raw_design.ncols();
        self.materialize_with_design(
            raw_design.clone(),
            raw_design,
            Array2::<f64>::eye(width),
            channel_offsets,
        )
    }

    fn materialize_with_design(
        &self,
        raw_design: DesignMatrix,
        coefficient_design: DesignMatrix,
        current_from_raw: Array2<f64>,
        channel_offsets: Array2<f64>,
    ) -> Result<SlopeLayout, String> {
        if raw_design.nrows() != channel_offsets.nrows() {
            return Err(format!(
                "slope layout offset length {} does not match design rows {}",
                channel_offsets.nrows(),
                raw_design.nrows(),
            ));
        }
        if channel_offsets.ncols() != self.score_count() {
            return Err(format!(
                "slope layout has {} offset channels but the topology has {} physical channels",
                channel_offsets.ncols(),
                self.score_count(),
            ));
        }
        if current_from_raw.nrows() != raw_design.ncols() {
            return Err(format!(
                "slope layout transform has {} raw rows but design has {} columns",
                current_from_raw.nrows(),
                raw_design.ncols(),
            ));
        }
        if current_from_raw.iter().any(|value| !value.is_finite()) {
            return Err("slope layout transform contains a non-finite value".to_string());
        }
        if channel_offsets.iter().any(|value| !value.is_finite()) {
            return Err("slope layout offset contains a non-finite value".to_string());
        }

        let nrows = raw_design.nrows();
        let current_width = current_from_raw.ncols();
        if coefficient_design.nrows() != nrows || coefficient_design.ncols() != current_width {
            return Err(format!(
                "slope current design is {}x{} but raw-design transform emits {nrows}x{current_width}",
                coefficient_design.nrows(),
                coefficient_design.ncols(),
            ));
        }
        match self {
            Self::Shared => {
                certify_channel_nonzero(
                    &raw_design,
                    &current_from_raw,
                    &(0..raw_design.ncols()),
                    0,
                    "shared slope",
                )?;
                Ok(SlopeLayout {
                    coefficient_design,
                    follow_up: None,
                    nrows,
                    current_width,
                    channels: SlopeChannels::Shared {
                        offset: Arc::new(channel_offsets.column(0).to_owned()),
                    },
                })
            }
            Self::PerScore { raw_ranges } => {
                validate_partition(raw_ranges, raw_design.ncols(), "per-score slope layout")?;
                for (channel, range) in raw_ranges.iter().enumerate() {
                    certify_channel_nonzero(
                        &raw_design,
                        &current_from_raw,
                        range,
                        channel,
                        "per-score slope",
                    )?;
                }
                Ok(SlopeLayout {
                    coefficient_design,
                    follow_up: None,
                    nrows,
                    current_width,
                    channels: SlopeChannels::PerScore {
                        raw_design,
                        current_from_raw: Arc::new(current_from_raw),
                        raw_ranges: Arc::clone(raw_ranges),
                        offsets: Arc::new(channel_offsets),
                    },
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum SlopeChannels {
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

/// The slope block's follow-up designs (gam#2765, gam#2767).
///
/// The coefficient design is the slope at the row's EXIT time — the same
/// convention the time block uses, so the block's `ParameterBlockSpec` eta is
/// already `g₁`. This carries the other two channels: the same coefficients
/// read against the basis at the row's ENTRY time, and against the exit-time
/// derivative of that basis.
///
/// `None` on a layout is the time-constant slope: `g₀ = g₁` and `ġ₁ = 0`
/// identically, which is what every model built before this existed asks for.
#[derive(Clone, Debug)]
pub(crate) struct SlopeFollowUpDesigns {
    pub(crate) entry: DesignMatrix,
    pub(crate) derivative_exit: DesignMatrix,
    /// The margin the covariate factor was tensored against, when the layout
    /// was built from one.
    pub(crate) time_margin: Option<SlopeTimeMargin>,
}

/// The follow-up margin a slope's covariate factor is tensored against: the
/// margin basis at each row's entry time and exit time, and its exit-time rate.
///
/// Each channel design is `X_cov ⊗ B_c` row by row in covariate-major column
/// order, so a covariate-width row lifts onto a channel by the same product.
/// That is what lets a hyperparameter that moves the covariate factor reach all
/// three slope primaries (gam#2767).
#[derive(Clone, Debug)]
pub(crate) struct SlopeTimeMargin {
    pub(crate) entry: Arc<Array2<f64>>,
    pub(crate) exit: Arc<Array2<f64>>,
    pub(crate) derivative_exit: Arc<Array2<f64>>,
}

impl SlopeTimeMargin {
    /// Columns in the margin basis.
    pub(crate) fn width(&self) -> usize {
        self.exit.ncols()
    }

    /// `x ⊗ b_c(row)` for the entry, exit and exit-rate channels, in primary
    /// order `(g₀, g₁, ġ₁)`.
    pub(crate) fn lift_row(&self, row: usize, covariate_row: &Array1<f64>) -> [Array1<f64>; 3] {
        let lift = |margin: &Array2<f64>| {
            let time_row = margin.row(row);
            let p_time = time_row.len();
            Array1::from_shape_fn(covariate_row.len() * p_time, |index| {
                covariate_row[index / p_time] * time_row[index % p_time]
            })
        };
        [
            lift(&self.entry),
            lift(&self.exit),
            lift(&self.derivative_exit),
        ]
    }
}

/// The slope block's `(primary, design)` channels for one layout.
///
/// A time-constant slope has exactly one; a follow-up-varying slope has three,
/// in primary order.
pub(crate) struct SlopeChannelDesigns<'a> {
    entries: [(usize, &'a DesignMatrix); 3],
    len: usize,
}

impl<'a> SlopeChannelDesigns<'a> {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[(usize, &'a DesignMatrix)] {
        &self.entries[..self.len]
    }
}

/// One row's slope index on its three follow-up channels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlopeRowChannels {
    pub(crate) entry: f64,
    pub(crate) exit: f64,
    pub(crate) rate: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct SlopeLayout {
    coefficient_design: DesignMatrix,
    follow_up: Option<SlopeFollowUpDesigns>,
    nrows: usize,
    current_width: usize,
    channels: SlopeChannels,
}

impl SlopeLayout {
    #[inline]
    pub(crate) fn is_per_score(&self) -> bool {
        matches!(self.channels, SlopeChannels::PerScore { .. })
    }

    #[inline]
    pub(crate) fn follow_up(&self) -> Option<&SlopeFollowUpDesigns> {
        self.follow_up.as_ref()
    }

    /// The coefficient design of a time-constant slope, whose single primary
    /// channel reads this design directly. A follow-up-varying layout feeds its
    /// channels from its own tensor designs and returns `None`.
    #[inline]
    pub(crate) fn static_coefficient_design(&self) -> Option<&DesignMatrix> {
        self.follow_up
            .is_none()
            .then_some(&self.coefficient_design)
    }

    /// Whether this layout lets the slope move along the follow-up axis.
    #[inline]
    pub(crate) fn is_follow_up_varying(&self) -> bool {
        self.follow_up.is_some()
    }

    /// Attach the entry and exit-derivative designs of a follow-up-varying
    /// slope.
    ///
    /// A per-score layout is refused rather than truncated: its physical
    /// channels are carved out of a raw design by coefficient range, and a
    /// time margin would have to be tensored per channel with a per-channel
    /// basis this type does not carry. Silently tensoring only the shared
    /// channel would fit a different model than the one asked for.
    pub(crate) fn with_follow_up(
        mut self,
        entry: DesignMatrix,
        derivative_exit: DesignMatrix,
    ) -> Result<Self, String> {
        if self.is_per_score() {
            return Err(
                "a follow-up-varying slope is not supported for a per-score slope \
                 topology: each physical score channel would need its own time margin"
                    .to_string(),
            );
        }
        for (name, design) in [("entry", &entry), ("derivative", &derivative_exit)] {
            if design.nrows() != self.nrows || design.ncols() != self.current_width {
                return Err(format!(
                    "slope follow-up {name} design is {}x{} but the layout is {}x{}",
                    design.nrows(),
                    design.ncols(),
                    self.nrows,
                    self.current_width,
                ));
            }
        }
        self.follow_up = Some(SlopeFollowUpDesigns {
            entry,
            derivative_exit,
            time_margin: None,
        });
        Ok(self)
    }

    /// Attach the margin the covariate factor was tensored against, so a
    /// hyperparameter that moves the covariate factor can be lifted onto the
    /// three channel designs (gam#2767).
    pub(crate) fn with_follow_up_time_margin(
        mut self,
        margin: SlopeTimeMargin,
    ) -> Result<Self, String> {
        let width = margin.width();
        for (name, basis) in [
            ("entry", &margin.entry),
            ("exit", &margin.exit),
            ("derivative", &margin.derivative_exit),
        ] {
            if basis.nrows() != self.nrows || basis.ncols() != width {
                return Err(format!(
                    "slope time-margin {name} basis is {}x{} but the layout has {} rows and a {width}-column margin",
                    basis.nrows(),
                    basis.ncols(),
                    self.nrows,
                ));
            }
        }
        if width == 0 || self.current_width % width != 0 {
            return Err(format!(
                "slope layout width {} is not a multiple of its {width}-column time margin",
                self.current_width,
            ));
        }
        let Some(follow_up) = self.follow_up.as_mut() else {
            return Err("a slope time margin needs a follow-up-varying layout".to_string());
        };
        follow_up.time_margin = Some(margin);
        Ok(self)
    }

    /// The follow-up margin, when the layout was built from one.
    #[inline]
    pub(crate) fn time_margin(&self) -> Option<&SlopeTimeMargin> {
        self.follow_up
            .as_ref()
            .and_then(|follow_up| follow_up.time_margin.as_ref())
    }

    /// The block's design channels paired with the primary each one
    /// differentiates.
    ///
    /// This is the single place that knows a follow-up-varying slope feeds
    /// three primaries and a time-constant one feeds a single primary, so every
    /// Jacobian / pullback / assembly site downstream is written once as a loop
    /// over channels instead of once per geometry.
    pub(crate) fn primary_channels(&self) -> SlopeChannelDesigns<'_> {
        match self.follow_up.as_ref() {
            None => SlopeChannelDesigns {
                entries: [
                    (PRIMARY_SLOPE, &self.coefficient_design),
                    (PRIMARY_SLOPE, &self.coefficient_design),
                    (PRIMARY_SLOPE, &self.coefficient_design),
                ],
                len: 1,
            },
            Some(follow_up) => SlopeChannelDesigns {
                entries: [
                    (PRIMARY_SLOPE, &follow_up.entry),
                    (PRIMARY_SLOPE_EXIT, &self.coefficient_design),
                    (PRIMARY_SLOPE_RATE, &follow_up.derivative_exit),
                ],
                len: 3,
            },
        }
    }

    /// The row's shared-channel offset. The slope offset is an external,
    /// time-constant slope contribution, so it enters `g₀` and `g₁` alike and
    /// contributes nothing to `ġ₁`.
    #[inline]
    pub(crate) fn shared_offset(&self, row: usize) -> Result<f64, String> {
        match &self.channels {
            SlopeChannels::Shared { offset } => Ok(offset[row]),
            SlopeChannels::PerScore { .. } => Err(
                "shared slope offset requested from a per-score layout".to_string(),
            ),
        }
    }

    /// The row's slope index on all three follow-up channels, computed from the
    /// coefficient vector alone.
    ///
    /// The shared-channel sibling of [`Self::row_channels`], for callers (the
    /// effective-Jacobian audit) that hold `β` but not the block's formed eta.
    #[inline]
    pub(crate) fn row_channels_from_beta(
        &self,
        row: usize,
        beta: ArrayView1<'_, f64>,
    ) -> Result<SlopeRowChannels, String> {
        let offset = self.shared_offset(row)?;
        let exit = self.coefficient_design.dot_row_view(row, beta) + offset;
        let Some(follow_up) = self.follow_up() else {
            return Ok(SlopeRowChannels {
                entry: exit,
                exit,
                rate: 0.0,
            });
        };
        Ok(SlopeRowChannels {
            entry: follow_up.entry.dot_row_view(row, beta) + offset,
            exit,
            rate: follow_up.derivative_exit.dot_row_view(row, beta),
        })
    }

    /// The row's slope index on all three follow-up channels, given the block's
    /// already-formed exit-time linear predictor.
    #[inline]
    pub(crate) fn row_channels(
        &self,
        row: usize,
        beta: &Array1<f64>,
        exit_eta: f64,
    ) -> Result<SlopeRowChannels, String> {
        let Some(follow_up) = self.follow_up() else {
            return Ok(SlopeRowChannels {
                entry: exit_eta,
                exit: exit_eta,
                rate: 0.0,
            });
        };
        Ok(SlopeRowChannels {
            entry: follow_up.entry.dot_row(row, beta) + self.shared_offset(row)?,
            exit: exit_eta,
            rate: follow_up.derivative_exit.dot_row(row, beta),
        })
    }

    pub(crate) fn score_count(&self) -> usize {
        match &self.channels {
            SlopeChannels::Shared { .. } => 1,
            SlopeChannels::PerScore { raw_ranges, .. } => raw_ranges.len(),
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
            return Err("slope layout coefficient-design invariant is broken".to_string());
        }
        if let SlopeChannels::Shared { offset } = &self.channels
            && offset.len() != self.nrows
        {
            return Err(format!(
                "shared slope offset has length {} but layout has {} rows",
                offset.len(),
                self.nrows,
            ));
        }
        if self.is_per_score() && self.score_count() != score_dim {
            return Err(format!(
                "per-score slope layout has {} channels but latent score has dimension {score_dim}",
                self.score_count(),
            ));
        }
        Ok(())
    }

    pub(crate) fn row_workspace(&self, score_dim: usize) -> Result<SlopeRowWorkspace, String> {
        if self.is_per_score() && self.score_count() != score_dim {
            return Err(format!(
                "cannot build slope row workspace: {} channels for score dimension {score_dim}",
                self.score_count(),
            ));
        }
        let raw_width = match &self.channels {
            SlopeChannels::Shared { .. } => 0,
            SlopeChannels::PerScore { raw_design, .. } => raw_design.ncols(),
        };
        Ok(SlopeRowWorkspace {
            raw_row: Array2::<f64>::zeros((1, raw_width)),
            channel_rows: Array2::<f64>::zeros((score_dim, self.current_width)),
            values: vec![0.0; score_dim],
        })
    }

    /// Materialize every physical slope channel at one coefficient vector.
    /// This is the sole batch boundary; row semantics remain owned by
    /// [`Self::fill_callback_row`].
    pub(crate) fn physical_values(
        &self,
        score_dim: usize,
        beta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.validate_for(score_dim)?;
        if beta.len() != self.current_width {
            return Err(format!(
                "slope physical-value beta length {} does not match current width {}",
                beta.len(),
                self.current_width,
            ));
        }
        let mut values = Array2::<f64>::zeros((self.nrows, score_dim));
        let mut workspace = self.row_workspace(score_dim)?;
        for row in 0..self.nrows {
            self.fill_callback_row(row, beta.view(), &mut workspace)?;
            for (channel, value) in workspace.values().iter().copied().enumerate() {
                values[[row, channel]] = value;
            }
        }
        Ok(values)
    }

    pub(crate) fn fill_shared_values(
        &self,
        value: f64,
        workspace: &mut SlopeRowWorkspace,
    ) -> Result<(), String> {
        if self.is_per_score() {
            return Err("cannot fill shared values for a per-score slope layout".to_string());
        }
        workspace.values.fill(value);
        Ok(())
    }

    pub(crate) fn fill_per_score_row(
        &self,
        row: usize,
        beta: ArrayView1<'_, f64>,
        workspace: &mut SlopeRowWorkspace,
    ) -> Result<(), String> {
        let SlopeChannels::PerScore {
            raw_design,
            current_from_raw,
            raw_ranges,
            offsets,
        } = &self.channels
        else {
            return Err("per-score slope row requested from a shared layout".to_string());
        };
        if row >= self.nrows {
            return Err(format!(
                "slope row {row} is out of bounds for {} rows",
                self.nrows
            ));
        }
        if beta.len() != self.current_width {
            return Err(format!(
                "slope beta length {} does not match current width {}",
                beta.len(),
                self.current_width,
            ));
        }
        if workspace.raw_row.dim() != (1, raw_design.ncols())
            || workspace.channel_rows.dim() != (raw_ranges.len(), self.current_width)
            || workspace.values.len() != raw_ranges.len()
        {
            return Err("slope row workspace shape does not match layout".to_string());
        }

        raw_design
            .row_chunk_into(row..row + 1, workspace.raw_row.view_mut())
            .map_err(|error| format!("slope layout row materialization failed: {error}"))?;
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
                workspace.channel_rows.row(channel).dot(&beta) + offsets[[row, channel]];
        }
        Ok(())
    }

    /// Fill physical slope values and their full-width
    /// current-coordinate rows directly from a coefficient vector. Unlike the
    /// likelihood's shared-channel fast path, this includes the layout-owned
    /// offset and is therefore the authoritative source for effective-Jacobian
    /// callbacks.
    pub(crate) fn fill_callback_row(
        &self,
        row: usize,
        beta: ArrayView1<'_, f64>,
        workspace: &mut SlopeRowWorkspace,
    ) -> Result<(), String> {
        match &self.channels {
            SlopeChannels::PerScore { .. } => self.fill_per_score_row(row, beta, workspace),
            SlopeChannels::Shared { offset } => {
                if row >= self.nrows {
                    return Err(format!(
                        "slope callback row {row} is out of bounds for {} rows",
                        self.nrows
                    ));
                }
                if beta.len() != self.current_width {
                    return Err(format!(
                        "slope callback beta length {} does not match current width {}",
                        beta.len(),
                        self.current_width,
                    ));
                }
                if workspace.channel_rows.ncols() != self.current_width
                    || workspace.values.len() != workspace.channel_rows.nrows()
                {
                    return Err(
                        "shared slope callback workspace shape does not match layout"
                            .to_string(),
                    );
                }
                self.coefficient_design
                    .row_chunk_into(row..row + 1, workspace.channel_rows.slice_mut(s![0..1, ..]))
                    .map_err(|error| {
                        format!("shared slope callback row materialization failed: {error}")
                    })?;
                for channel in 1..workspace.channel_rows.nrows() {
                    for col in 0..self.current_width {
                        workspace.channel_rows[[channel, col]] = workspace.channel_rows[[0, col]];
                    }
                }
                let value = self.coefficient_design.dot_row_view(row, beta) + offset[row];
                workspace.values.fill(value);
                Ok(())
            }
        }
    }
}

pub(crate) struct SlopeRowWorkspace {
    raw_row: Array2<f64>,
    channel_rows: Array2<f64>,
    values: Vec<f64>,
}

impl SlopeRowWorkspace {
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
    context: &str,
) -> Result<(), String> {
    if range.is_empty() || current_from_raw.ncols() == 0 {
        return Err(format!(
            "{context} channel {channel} has no current-coordinate derivative"
        ));
    }
    let mut chunk = Array2::<f64>::zeros((
        CHANNEL_SCAN_ROWS.min(raw_design.nrows()),
        raw_design.ncols(),
    ));
    for start in (0..raw_design.nrows()).step_by(CHANNEL_SCAN_ROWS) {
        let end = (start + CHANNEL_SCAN_ROWS).min(raw_design.nrows());
        let rows = end - start;
        if chunk.nrows() != rows {
            chunk = Array2::<f64>::zeros((rows, raw_design.ncols()));
        }
        raw_design
            .row_chunk_into(start..end, chunk.view_mut())
            .map_err(|error| format!("{context} channel scan failed: {error}"))?;
        for local_row in 0..rows {
            for current_col in 0..current_from_raw.ncols() {
                let mut value = 0.0;
                for raw_col in range.clone() {
                    value += chunk[[local_row, raw_col]] * current_from_raw[[raw_col, current_col]];
                }
                if !value.is_finite() {
                    return Err(format!(
                        "{context} channel {channel} produced a non-finite current-coordinate row"
                    ));
                }
                if value != 0.0 {
                    return Ok(());
                }
            }
        }
    }
    Err(format!(
        "{context} channel {channel} is identically zero after the coefficient transform"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    impl SlopeLayout {
        /// Test-only shared-channel constructor. Production paths use
        /// [`SlopeTopology::materialize_identity`] so topology validation
        /// stays at the construction boundary.
        pub(crate) fn shared(coefficient_design: DesignMatrix, offset: Array1<f64>) -> Self {
            let nrows = coefficient_design.nrows();
            let current_width = coefficient_design.ncols();
            Self {
                coefficient_design,
                follow_up: None,
                nrows,
                current_width,
                channels: SlopeChannels::Shared {
                    offset: Arc::new(offset),
                },
            }
        }

        pub(crate) fn replace_coefficient_design(&mut self, design: DesignMatrix) {
            assert_eq!(
                design.nrows(),
                self.nrows,
                "test replacement must preserve slope layout row count"
            );
            self.current_width = design.ncols();
            self.coefficient_design = design;
        }
    }

    impl From<DesignMatrix> for SlopeLayout {
        fn from(design: DesignMatrix) -> Self {
            let nrows = design.nrows();
            Self::shared(design, Array1::<f64>::zeros(nrows))
        }
    }

    #[test]
    fn unequal_raw_widths_emit_full_width_channel_rows_and_offsets() {
        let raw = array![[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]];
        let topology = SlopeTopology::per_score(vec![0..1, 1..3], 3).unwrap();
        let layout = topology
            .materialize_identity(DesignMatrix::from(raw), &array![0.5, -0.25])
            .unwrap();
        let beta = array![17.0, 19.0, 23.0];
        let mut workspace = layout.row_workspace(2).unwrap();
        layout
            .fill_per_score_row(0, beta.view(), &mut workspace)
            .unwrap();

        assert_eq!(
            workspace.channel_rows(),
            array![[2.0, 0.0, 0.0], [0.0, 3.0, 5.0]]
        );
        assert_eq!(workspace.values(), &[34.5, 172.5]);
    }

    #[test]
    fn distinct_channel_offsets_reach_their_own_channels() {
        let raw = array![[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]];
        let topology = SlopeTopology::per_score(vec![0..1, 1..3], 3).unwrap();
        let layout = topology
            .materialize_identity_with_channel_offsets(
                DesignMatrix::from(raw),
                array![[0.5, -4.0], [1.5, 8.0]],
            )
            .unwrap();
        let beta = array![17.0, 19.0, 23.0];
        let mut workspace = layout.row_workspace(2).unwrap();
        layout
            .fill_per_score_row(1, beta.view(), &mut workspace)
            .unwrap();
        // channel 0: 7*17 + 1.5; channel 1: 11*19 + 13*23 + 8.
        assert_eq!(workspace.values(), &[120.5, 516.0]);

        let error = topology
            .materialize_identity_with_channel_offsets(
                DesignMatrix::from(array![[1.0, 1.0, 1.0]]),
                array![[0.0]],
            )
            .err()
            .expect("a per-score layout needs one offset column per channel");
        assert!(error.contains("1 offset channels"), "{error}");
    }

    #[test]
    fn shared_zero_width_physical_channel_is_rejected_exactly() {
        let topology = SlopeTopology::shared();
        let raw_design = DesignMatrix::from(array![[1.0], [2.0]]);
        let error = topology
            .materialize_with_design(
                raw_design,
                DesignMatrix::from(Array2::<f64>::zeros((2, 0))),
                Array2::<f64>::zeros((1, 0)),
                array![[0.0], [0.0]],
            )
            .err()
            .expect("shared physical channel cannot have zero current width");
        assert!(
            error.contains("shared slope channel 0 has no current-coordinate derivative"),
            "{error}"
        );
    }
}
