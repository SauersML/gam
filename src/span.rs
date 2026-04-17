/// Require a breakpoint sequence suitable for span lookup: finite, strictly
/// increasing, and long enough to define at least one span.
pub(crate) fn validate_breakpoints(breakpoints: &[f64], label: &str) -> Result<(), String> {
    if breakpoints.len() < 2 {
        return Err(format!("{label} requires at least two breakpoints"));
    }
    if let Some((idx, window)) = breakpoints.windows(2).enumerate().find(|(_, window)| {
        !window[0].is_finite() || !window[1].is_finite() || window[0] >= window[1]
    }) {
        return Err(format!(
            "{label} requires strictly increasing finite breakpoints; breakpoints[{idx}]={:.6}, breakpoints[{}]={:.6}",
            window[0],
            idx + 1,
            window[1]
        ));
    }
    Ok(())
}

/// Deduplicate an ordered knot sequence into strictly increasing breakpoints.
pub(crate) fn breakpoints_from_knots(knots: &[f64], label: &str) -> Result<Vec<f64>, String> {
    let mut breakpoints = Vec::new();
    for &knot in knots {
        if breakpoints
            .last()
            .is_none_or(|prev: &f64| (knot - *prev).abs() > 1e-12)
        {
            breakpoints.push(knot);
        }
    }
    validate_breakpoints(&breakpoints, label)?;
    Ok(breakpoints)
}

/// Select the span containing `value`, using `[left, right)` for every span
/// except the final span, which is right-closed.
pub(crate) fn span_index_for_breakpoints(
    breakpoints: &[f64],
    value: f64,
    label: &str,
) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("{label} requires finite value, got {value}"));
    }
    if breakpoints.len() < 2 {
        return Err(format!("{label} requires at least two breakpoints"));
    }
    let last_idx = breakpoints.len() - 1;
    if value <= breakpoints[0] {
        return Ok(0);
    }
    if value >= breakpoints[last_idx] {
        return Ok(last_idx - 1);
    }
    let insertion_idx = breakpoints.partition_point(|point| *point <= value);
    Ok((insertion_idx - 1).min(last_idx - 1))
}

#[cfg(test)]
mod tests {
    use super::{breakpoints_from_knots, span_index_for_breakpoints};

    #[test]
    fn deduplicates_knots_into_breakpoints() {
        assert_eq!(
            breakpoints_from_knots(&[-2.0, -2.0, 0.0, 1.5, 1.5, 3.0], "test breakpoints").unwrap(),
            vec![-2.0, 0.0, 1.5, 3.0]
        );
    }

    /// Documents the `span_index_for_breakpoints` helper convention only.
    /// Specific design evaluators may override this for d2+ handling — see
    /// `deviation_runtime.rs:186` and `inference/model.rs:761`, which both
    /// apply a LEFT-bias at interior breakpoints because their C¹ basis
    /// has distinct left/right limits for d2 at interior breaks.
    #[test]
    fn internal_breakpoints_use_right_hand_span() {
        let breakpoints = [-1.5, -0.9, 0.4, 2.0];
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, -1.5, "test span lookup").unwrap(),
            0
        );
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, -0.9, "test span lookup").unwrap(),
            1
        );
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, 0.4, "test span lookup").unwrap(),
            2
        );
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, 2.0, "test span lookup").unwrap(),
            2
        );
    }
}
