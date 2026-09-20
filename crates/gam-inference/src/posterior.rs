use gam_math::quantile::quantile_from_sorted;

/// Equal-tailed posterior credible interval of every coefficient, from a
/// row-major `n_draws × n_coeffs` matrix of coefficient draws.
///
/// A coefficient draw is a real number: a NaN or ±inf draw is a failed
/// sampler step, not a posterior value, and it is refused with its draw and
/// coefficient index. Sorting such a column would otherwise need a comparator
/// that is not a total order (NaN ties with every value), which leaves the
/// column in an arbitrary order and turns the reported quantiles into garbage
/// that still looks like an interval.
pub fn credible_interval(
    samples_flat: &[f64],
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<Vec<f64>, String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("interval level must lie in (0, 1); got {level}"));
    }
    if n_draws.checked_mul(n_coeffs) != Some(samples_flat.len()) {
        return Err(format!(
            "posterior_credible_interval samples shape mismatch: got {} floats, expected {} * {}",
            samples_flat.len(),
            n_draws,
            n_coeffs
        ));
    }
    if n_draws == 0 {
        return Err("posterior_credible_interval requires at least one posterior draw".to_string());
    }
    if let Some((index, value)) = samples_flat
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "posterior_credible_interval coefficient draw is not finite at draw {}, coefficient {}: {value}",
            index / n_coeffs,
            index % n_coeffs
        ));
    }
    let alpha = (1.0 - level) / 2.0;
    let mut out = Vec::with_capacity(2 * n_coeffs);
    let mut column = vec![0.0_f64; n_draws];
    for j in 0..n_coeffs {
        for k in 0..n_draws {
            column[k] = samples_flat[k * n_coeffs + j];
        }
        column.sort_by(f64::total_cmp);
        out.push(quantile_from_sorted(&column, alpha));
        out.push(quantile_from_sorted(&column, 1.0 - alpha));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_draws_rejects_credible_interval() {
        let err = credible_interval(&[], 0, 2, 0.95).expect_err("zero draws must fail");
        assert!(err.contains("requires at least one posterior draw"));
    }

    #[test]
    fn credible_interval_brackets_sample_mean_with_draws() {
        let n_draws = 5;
        let n_coeffs = 2;
        let samples = vec![
            -2.0, 1.0, //
            -1.0, 2.0, //
            0.0, 3.0, //
            1.0, 4.0, //
            2.0, 5.0, //
        ];
        let ci = credible_interval(&samples, n_draws, n_coeffs, 0.80).expect("interval");
        for j in 0..n_coeffs {
            let mean =
                (0..n_draws).map(|k| samples[k * n_coeffs + j]).sum::<f64>() / n_draws as f64;
            assert!(
                ci[j * 2] <= mean && mean <= ci[j * 2 + 1],
                "coefficient {j} mean {mean} must sit inside [{}, {}]",
                ci[j * 2],
                ci[j * 2 + 1]
            );
        }
    }

    /// A NaN or ±inf coefficient draw used to go through a comparator that
    /// ties NaN with every value, so the "sorted" column was in an arbitrary
    /// order and the interval was silent garbage. It is refused, naming the
    /// draw and coefficient.
    #[test]
    fn credible_interval_refuses_non_finite_draws_naming_the_cell() {
        let n_draws = 40;
        let n_coeffs = 2;
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut samples: Vec<f64> = (0..n_draws * n_coeffs).map(|i| i as f64).collect();
            samples[17 * n_coeffs + 1] = bad;
            let err = credible_interval(&samples, n_draws, n_coeffs, 0.9)
                .expect_err("a non-finite coefficient draw must be refused");
            assert!(
                err.contains("draw 17, coefficient 1"),
                "the error must name the offending cell: {err}"
            );
        }
    }

    /// A shape whose element count overflows `usize` is a mismatch, not a
    /// wrapped product that could accidentally equal the buffer length.
    #[test]
    fn credible_interval_refuses_overflowing_shape() {
        let err = credible_interval(&[0.0], usize::MAX, 2, 0.9)
            .expect_err("an overflowing shape must be refused");
        assert!(err.contains("shape mismatch"), "{err}");
    }

    /// Draws arriving in reverse order give the same interval as sorted draws.
    #[test]
    fn credible_interval_is_order_invariant() {
        let forward: Vec<f64> = (0..101).map(|i| i as f64 * 0.5).collect();
        let reversed: Vec<f64> = forward.iter().rev().copied().collect();
        let a = credible_interval(&forward, 101, 1, 0.9).expect("forward");
        let b = credible_interval(&reversed, 101, 1, 0.9).expect("reversed");
        assert_eq!(a, b);
        // numpy.quantile(linspace(0, 50, 101), [0.05, 0.95]) = [2.5, 47.5].
        assert!((a[0] - 2.5).abs() < 1e-9 && (a[1] - 47.5).abs() < 1e-9, "{a:?}");
    }
}
