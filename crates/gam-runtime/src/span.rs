/// Select the span containing `value`: span `i` is `(b_i, b_{i+1}]`, except the
/// first, which is closed `[b_0, b_1]`. An interior breakpoint therefore belongs
/// to the span on its LEFT. The anchored deviation runtimes evaluate a C² cubic
/// basis, so value, first and second derivative agree on both sides of a
/// breakpoint and only the span-local third derivative depends on this choice.
/// Values outside the band select the nearest end span.
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
    // `breakpoints[0] < value < breakpoints[last_idx]`, so the first breakpoint
    // at or above `value` has an index in `1..=last_idx`.
    Ok(breakpoints.partition_point(|point| *point < value) - 1)
}

#[cfg(test)]
mod tests {
    use super::span_index_for_breakpoints;

    /// An interior breakpoint belongs to the span on its left, the convention
    /// both anchored deviation runtimes report span-local third derivatives in.
    #[test]
    fn internal_breakpoints_use_left_hand_span() {
        let breakpoints = [-1.5, -0.9, 0.4, 2.0];
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, -1.5, "test span lookup").unwrap(),
            0
        );
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, -0.9, "test span lookup").unwrap(),
            0
        );
        assert_eq!(
            span_index_for_breakpoints(&breakpoints, 0.4, "test span lookup").unwrap(),
            1
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
        // 0.5 is in [0,1] → span 0
        assert_eq!(span_index_for_breakpoints(&bp, 0.5, "t").unwrap(), 0);
        // 1.5 is in (1,2] → span 1
        assert_eq!(span_index_for_breakpoints(&bp, 1.5, "t").unwrap(), 1);
        // 2.5 is in (2,3] → span 2
        assert_eq!(span_index_for_breakpoints(&bp, 2.5, "t").unwrap(), 2);
    }

    #[test]
    fn error_message_contains_label() {
        let err = span_index_for_breakpoints(&[0.0], 0.5, "my_var").unwrap_err();
        assert!(
            err.contains("my_var"),
            "error should mention label, got: {err}"
        );
    }
}
