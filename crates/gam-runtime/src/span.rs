/// Select the span containing `value`, using `[left, right)` for every span
/// except the final span, which is right-closed.
pub fn span_index_for_breakpoints(
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

    #[test]
    fn value_below_first_breakpoint_returns_span_zero() {
        let bp = [0.0, 1.0, 2.0];
        assert_eq!(span_index_for_breakpoints(&bp, -5.0, "t").unwrap(), 0);
    }

    #[test]
    fn value_above_last_breakpoint_returns_last_span() {
        let bp = [0.0, 1.0, 2.0];
        assert_eq!(span_index_for_breakpoints(&bp, 99.0, "t").unwrap(), 1);
    }

    #[test]
    fn two_breakpoints_only_one_span() {
        let bp = [0.0, 1.0];
        assert_eq!(span_index_for_breakpoints(&bp, 0.5, "t").unwrap(), 0);
        assert_eq!(span_index_for_breakpoints(&bp, 0.0, "t").unwrap(), 0);
        assert_eq!(span_index_for_breakpoints(&bp, 1.0, "t").unwrap(), 0);
    }

    #[test]
    fn non_finite_value_returns_error() {
        let bp = [0.0, 1.0, 2.0];
        assert!(span_index_for_breakpoints(&bp, f64::NAN, "t").is_err());
        assert!(span_index_for_breakpoints(&bp, f64::INFINITY, "t").is_err());
        assert!(span_index_for_breakpoints(&bp, f64::NEG_INFINITY, "t").is_err());
    }

    #[test]
    fn fewer_than_two_breakpoints_returns_error() {
        assert!(span_index_for_breakpoints(&[], 0.5, "t").is_err());
        assert!(span_index_for_breakpoints(&[1.0], 0.5, "t").is_err());
    }

    #[test]
    fn interior_midpoint_selects_correct_span() {
        let bp = [0.0, 1.0, 2.0, 3.0];
        // 0.5 is in [0,1) → span 0
        assert_eq!(span_index_for_breakpoints(&bp, 0.5, "t").unwrap(), 0);
        // 1.5 is in [1,2) → span 1
        assert_eq!(span_index_for_breakpoints(&bp, 1.5, "t").unwrap(), 1);
        // 2.5 is in [2,3) → span 2
        assert_eq!(span_index_for_breakpoints(&bp, 2.5, "t").unwrap(), 2);
    }

    #[test]
    fn error_message_contains_label() {
        let err = span_index_for_breakpoints(&[0.0], 0.5, "my_var").unwrap_err();
        assert!(err.contains("my_var"), "error should mention label, got: {err}");
    }
}
