//! Terminal smooth visualizer: faithful unicode sparklines for fitted
//! partial-effect curves.
//!
//! Sometimes you just want to *see* the shape of a smooth without exporting a
//! frame to R and firing up a plotting backend. This module renders the
//! already-computed partial effect `ŝ_j(x)` of each smooth term as a compact
//! one-line sparkline using the eight block glyphs `▁▂▃▄▅▆▇█`, labelled with
//! the term name and its x- and y-ranges so the picture is quantitative, not
//! just pretty.
//!
//! ```text
//!   s(age):  ▁▂▃▅▆▇█▇▆▅▃▂▁    x∈[18.0, 84.0]  y∈[-0.42, 1.07]
//! ```
//!
//! The renderer is **pure and read-only**. It consumes a slice of `f64`
//! values that some upstream caller already computed (e.g. a term's fitted
//! contribution on a grid) and turns them into a `String`. It performs no
//! fitting, no prediction, no penalty arithmetic — it can no more change a
//! coefficient than a thermometer can change the weather. Every glyph is a
//! faithful, order-preserving image of an input value: the series minimum maps
//! to the lowest block, the maximum to the highest, everything else linearly
//! in between.

/// The eight vertical block glyphs, shortest to tallest. A value at the series
/// minimum renders as `BLOCKS[0]`, a value at the maximum as the last glyph.
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Glyph used for a value that is not finite (NaN or ±∞) so a single bad
/// sample is visible rather than silently distorting the ramp.
const NON_FINITE_GLYPH: char = '·';

/// Renders a single-row sparkline of `values` using the block ramp.
///
/// Faithfulness guarantees:
/// * The finite minimum maps to `▁`, the finite maximum to `█`, and every
///   finite value in between to a linearly-interpolated block. Order is
///   preserved: `a <= b` implies `glyph(a) <= glyph(b)` on the ramp.
/// * A *constant* finite series (max == min, including a single point) is
///   drawn as a flat mid-height line `▄`, since "no variation" is the honest
///   picture — there is no spread to stretch across the ramp.
/// * Non-finite samples render as `·` and are excluded from the min/max range,
///   so one NaN cannot rescale the whole curve.
/// * An empty series renders as the empty string.
///
/// The output has exactly one glyph per input value (it does not resample);
/// callers that want a fixed width should pre-thin or pre-interpolate their
/// grid.
pub fn render_sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return String::new();
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }

    // No finite samples at all: nothing to scale against.
    if !min.is_finite() || !max.is_finite() {
        return values.iter().map(|_| NON_FINITE_GLYPH).collect();
    }

    let span = max - min;
    let last = BLOCKS.len() - 1;

    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return NON_FINITE_GLYPH;
            }
            if span <= 0.0 {
                // Constant series: a flat, mid-height line is the honest image.
                return BLOCKS[last / 2];
            }
            // Linear map into [0, last], rounded to the nearest block. The
            // clamp guards only against floating-point round-off at the ends.
            let frac = (v - min) / span;
            let idx = (frac * last as f64).round() as usize;
            BLOCKS[idx.min(last)]
        })
        .collect()
}

/// Renders one labelled line for a named smooth term: its sparkline followed by
/// the x-range it was evaluated over and the y-range (partial-effect spread).
///
/// `xs` and `ys` are paired grid samples (same length); `ys` is the fitted
/// partial effect that gets drawn, `xs` only supplies the x-range label. If the
/// lengths differ the shorter is used so the label never lies about the curve.
/// Ranges ignore non-finite samples, matching [`render_sparkline`].
pub fn render_smooth_line(name: &str, xs: &[f64], ys: &[f64]) -> String {
    let n = xs.len().min(ys.len());
    let xs = &xs[..n];
    let ys = &ys[..n];

    let spark = render_sparkline(ys);
    let (xlo, xhi) = finite_range(xs);
    let (ylo, yhi) = finite_range(ys);

    format!(
        "  {name}:  {spark}    x∈[{xlo}, {xhi}]  y∈[{ylo}, {yhi}]",
        xlo = fmt(xlo),
        xhi = fmt(xhi),
        ylo = fmt(ylo),
        yhi = fmt(yhi),
    )
}

/// Finite (min, max) of a slice, or `(NaN, NaN)` when no finite sample exists.
fn finite_range(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Compact fixed-precision label for a range endpoint; renders a non-finite
/// endpoint as `n/a` rather than the misleading `NaN`.
fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "n/a".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{render_smooth_line, render_sparkline};

    #[test]
    fn ramp_maps_min_to_bottom_and_max_to_top() {
        // A monotone ramp of 8 evenly spaced values should walk the full ramp
        // bottom to top, exactly.
        let ys: Vec<f64> = (0..8).map(|i| i as f64).collect();
        assert_eq!(render_sparkline(&ys), "▁▂▃▄▅▆▇█");
    }

    #[test]
    fn symmetric_bump_is_faithful() {
        // A tent function rises then falls symmetrically.
        let ys = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        assert_eq!(render_sparkline(&ys), "▁▃▆█▆▃▁");
    }

    #[test]
    fn constant_series_is_a_flat_midline() {
        // No variation -> a flat mid-height line, not a stretched ramp.
        assert_eq!(render_sparkline(&[2.5, 2.5, 2.5, 2.5]), "▄▄▄▄");
        // A single point is "constant" by the same logic.
        assert_eq!(render_sparkline(&[42.0]), "▄");
    }

    #[test]
    fn empty_series_renders_empty() {
        assert_eq!(render_sparkline(&[]), "");
    }

    #[test]
    fn non_finite_samples_are_marked_and_do_not_rescale() {
        // The NaN renders as `·` and is excluded from the min/max, so the two
        // finite endpoints still map to the bottom and top of the ramp.
        assert_eq!(render_sparkline(&[0.0, f64::NAN, 1.0]), "▁·█");
        // All non-finite -> all markers, no panic from an empty range.
        assert_eq!(render_sparkline(&[f64::NAN, f64::INFINITY]), "··");
    }

    #[test]
    fn labelled_line_reports_true_ranges() {
        let xs = [18.0, 50.0, 84.0];
        let ys = [-0.42, 0.30, 1.07];
        let line = render_smooth_line("s(age)", &xs, &ys);
        assert_eq!(
            line,
            "  s(age):  ▁▄█    x∈[18.000, 84.000]  y∈[-0.420, 1.070]"
        );
    }

    #[test]
    fn labelled_line_handles_no_finite_y() {
        let xs = [0.0, 1.0];
        let ys = [f64::NAN, f64::NAN];
        let line = render_smooth_line("s(x)", &xs, &ys);
        assert_eq!(line, "  s(x):  ··    x∈[0.000, 1.000]  y∈[n/a, n/a]");
    }
}
