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
    use super::span_index_for_breakpoints;

    /// Documents the `span_index_for_breakpoints` helper convention only.
    /// Specific design evaluators may override this for endpoint convention.
    /// Anchored deviation runtimes apply a LEFT-bias at interior breakpoints so
    /// span-local third derivatives are reported from the left span; their
    /// cubic basis is C², so value, first derivative, and second derivative are
    /// unaffected by that choice.
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
