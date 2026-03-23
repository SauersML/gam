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
