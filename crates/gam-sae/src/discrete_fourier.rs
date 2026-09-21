//! Exact discrete Fourier transform in `O(n log n)`, for arbitrary `n`.
//!
//! # Why this module exists
//!
//! [`crate::null_battery::phase_randomized_surrogate`] needs a forward and an
//! inverse DFT per activation column. It used to spell both out as the textbook
//! double loop
//!
//! ```text
//! for k in 0..n { for t in 0..n { acc += x_t · e^{−2πi k t / n} } }
//! ```
//!
//! which is `O(n²)` with two transcendental calls in the inner body. That is a
//! correct transform and a catastrophic cost: the surrogate is drawn ONCE PER
//! MONTE-CARLO REPLICATE, and a phase-randomized null
//! ([`crate::null_battery`]) draws `B` replicates per screened candidate, so the
//! quadratic term is multiplied by `B · candidates`. Measured consequence at
//! `n = 2000`, `B = 600`, 3 pairs: a single screen spends hours inside the
//! transform. This is not an oversized fixture — the *statistic* is cheap and the
//! sample sizes are ordinary — it is a quadratic algorithm standing where an
//! `O(n log n)` one belongs.
//!
//! # What is implemented
//!
//! [`dft_in_place`] is the same mathematical object as the double loop, to
//! floating-point roundoff, for EVERY length:
//!
//! * `n` a power of two: iterative in-place radix-2 Cooley–Tukey.
//! * any other `n`: Bluestein's chirp-z algorithm, which rewrites the DFT as a
//!   linear convolution of length `≥ 2n − 1` and evaluates that convolution with
//!   power-of-two transforms. No length is special-cased away, and no length is
//!   left on a quadratic path.
//!
//! Both directions are UNNORMALIZED (`inverse` uses the `+2πi` kernel and applies
//! no `1/n`), matching the loops they replace: the caller owns the scaling, which
//! keeps the surrogate's guarded `1/n` multiply exactly where it was.
//!
//! # Accuracy
//!
//! Twiddle factors are evaluated directly from the angle at every butterfly
//! rather than propagated by complex recurrence, so error does not accumulate
//! across stages; the transcendental count drops from `2n²` to `O(n log n)`,
//! which is the whole point of the change, so paying for exact twiddles is free
//! in comparison. Bluestein's chirp angles are formed from `(m² mod 2n)` in
//! INTEGER arithmetic before the division by `n`, so the argument handed to
//! `sin`/`cos` stays in `[0, 2π)` instead of growing like `m²` and losing
//! low-order bits to argument reduction. The result is at least as accurate as
//! the naive loop, which summed `n` terms in one unblocked accumulator.

/// In-place discrete Fourier transform of the complex sequence `(re, im)`.
///
/// `inverse = false` applies the `e^{−2πi k t / n}` kernel, `inverse = true` the
/// `e^{+2πi k t / n}` kernel. Neither direction divides by `n`.
///
/// Errors when `re` and `im` have different lengths — the two halves of one
/// complex sequence cannot disagree on how many entries it has.
pub(crate) fn dft_in_place(re: &mut [f64], im: &mut [f64], inverse: bool) -> Result<(), String> {
    if re.len() != im.len() {
        return Err(format!(
            "discrete Fourier transform: real and imaginary parts must have equal length, got {} and {}",
            re.len(),
            im.len()
        ));
    }
    let n = re.len();
    if n <= 1 {
        return Ok(());
    }
    if n.is_power_of_two() {
        fft_radix2_in_place(re, im, inverse);
    } else {
        bluestein_in_place(re, im, inverse)?;
    }
    Ok(())
}

/// Iterative in-place radix-2 Cooley–Tukey. `re.len()` must be a power of two.
fn fft_radix2_in_place(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    // `assert!`, not the debug form. The scanner bans it and is right here: a
    // radix-2 butterfly on a non-power-of-two length does not fail loudly, it
    // silently computes the WRONG TRANSFORM -- the bit-reversal permutation
    // below is only a permutation when `n` is a power of two, so a release
    // build returns a plausible, wrong spectrum with no diagnostic. A check
    // that vanishes in release is absent exactly where it matters. One
    // `is_power_of_two()` against an O(n log n) body is not a hot path.
    assert!(
        n.is_power_of_two(),
        "fft_radix2_in_place requires a power-of-two length, got {n}; the caller \
         must route other lengths to bluestein_in_place"
    );

    // Decimation-in-time reordering: index `i` moves to its bit-reversal.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        // Angle step for this stage. Twiddles are evaluated from `step * k`
        // directly (not accumulated), so no stage inherits the previous stage's
        // rounding.
        let step = sign * std::f64::consts::TAU / len as f64;
        let mut base = 0usize;
        while base < n {
            for k in 0..half {
                let angle = step * k as f64;
                let (wi, wr) = angle.sin_cos();
                let lo = base + k;
                let hi = lo + half;
                let vr = re[hi] * wr - im[hi] * wi;
                let vi = re[hi] * wi + im[hi] * wr;
                let ur = re[lo];
                let ui = im[lo];
                re[lo] = ur + vr;
                im[lo] = ui + vi;
                re[hi] = ur - vr;
                im[hi] = ui - vi;
            }
            base += len;
        }
        len <<= 1;
    }
}

/// Bluestein's chirp-z transform for arbitrary `n`.
///
/// Uses `k·t = (k² + t² − (k − t)²) / 2` to turn the DFT into the linear
/// convolution of `a_t = x_t · e^{∓πi t²/n}` with `b_m = e^{±πi m²/n}`, which is
/// evaluated by three power-of-two transforms of length `m ≥ 2n − 1`.
fn bluestein_in_place(re: &mut [f64], im: &mut [f64], inverse: bool) -> Result<(), String> {
    let n = re.len();
    let conv_len = (2 * n - 1)
        .checked_next_power_of_two()
        .ok_or_else(|| format!("discrete Fourier transform: length {n} overflows the chirp-z convolution length"))?;

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    // Chirp angle `sign · π · m² / n`, with `m²` reduced modulo `2n` in integer
    // arithmetic first: `m²` reaches `n²` and would otherwise be reduced by
    // `sin`/`cos` at full magnitude, discarding low-order bits of the angle.
    let two_n = 2 * n;
    let chirp = |m: usize| -> (f64, f64) {
        let residue = (m % two_n) * (m % two_n) % two_n;
        let angle = sign * std::f64::consts::PI * residue as f64 / n as f64;
        angle.sin_cos()
    };

    // `a_t = x_t · e^{±πi t²/n}` — the chirp pre-multiply.
    let mut a_re = vec![0.0_f64; conv_len];
    let mut a_im = vec![0.0_f64; conv_len];
    for t in 0..n {
        let (s, c) = chirp(t);
        a_re[t] = re[t] * c - im[t] * s;
        a_im[t] = re[t] * s + im[t] * c;
    }

    // `b_m = e^{∓πi m²/n}` = conj(chirp), laid out so a negative lag `m − t`
    // wraps to `conv_len − (t − m)`; `b` is even in `m`, so both ends share one
    // evaluation.
    let mut b_re = vec![0.0_f64; conv_len];
    let mut b_im = vec![0.0_f64; conv_len];
    for m in 0..n {
        let (s, c) = chirp(m);
        b_re[m] = c;
        b_im[m] = -s;
        if m > 0 {
            b_re[conv_len - m] = c;
            b_im[conv_len - m] = -s;
        }
    }

    fft_radix2_in_place(&mut a_re, &mut a_im, false);
    fft_radix2_in_place(&mut b_re, &mut b_im, false);
    for idx in 0..conv_len {
        let pr = a_re[idx] * b_re[idx] - a_im[idx] * b_im[idx];
        let pi = a_re[idx] * b_im[idx] + a_im[idx] * b_re[idx];
        a_re[idx] = pr;
        a_im[idx] = pi;
    }
    // Inverse power-of-two transform of the product; the `1/conv_len` that the
    // unnormalized kernel omits is applied here, where the convolution — not the
    // caller's DFT — is what needs it.
    fft_radix2_in_place(&mut a_re, &mut a_im, true);
    let scale = 1.0 / conv_len as f64;

    // Post-multiply by the chirp again to recover `X_k`.
    for k in 0..n {
        let (s, c) = chirp(k);
        let cr = a_re[k] * scale;
        let ci = a_im[k] * scale;
        re[k] = cr * c - ci * s;
        im[k] = cr * s + ci * c;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The textbook `O(n²)` transform this module replaces, kept here as the
    /// oracle: the fast path is only allowed to be fast if it computes the same
    /// thing the double loop did.
    fn naive_dft(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut out_re = vec![0.0_f64; n];
        let mut out_im = vec![0.0_f64; n];
        for k in 0..n {
            for t in 0..n {
                let angle = sign * std::f64::consts::TAU * (k as f64) * (t as f64) / (n as f64);
                let (s, c) = angle.sin_cos();
                out_re[k] += re[t] * c - im[t] * s;
                out_im[k] += re[t] * s + im[t] * c;
            }
        }
        (out_re, out_im)
    }

    fn sequence(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let re: Vec<f64> = (0..n).map(|_| next()).collect();
        let im: Vec<f64> = (0..n).map(|_| next()).collect();
        (re, im)
    }

    /// Power-of-two, composite non-power-of-two, and PRIME lengths all have to
    /// agree with the oracle — the prime case is the one that exercises
    /// Bluestein rather than the radix-2 kernel.
    #[test]
    fn fast_transform_matches_the_naive_dft_at_every_length_class() {
        for &n in &[1usize, 2, 3, 4, 7, 8, 12, 13, 16, 31, 64, 96, 101, 128, 200] {
            for inverse in [false, true] {
                let (re0, im0) = sequence(n, 0xD1CE + n as u64);
                let (want_re, want_im) = naive_dft(&re0, &im0, inverse);
                let mut re = re0.clone();
                let mut im = im0.clone();
                dft_in_place(&mut re, &mut im, inverse).expect("equal-length halves");
                // The oracle itself sums `n` terms unblocked, so the comparison
                // band scales with `n` times the coefficient magnitude.
                let scale = want_re
                    .iter()
                    .chain(want_im.iter())
                    .fold(1.0_f64, |acc, v| acc.max(v.abs()));
                let tol = 1.0e-12 * scale * (n as f64);
                for k in 0..n {
                    assert!(
                        (re[k] - want_re[k]).abs() <= tol && (im[k] - want_im[k]).abs() <= tol,
                        "n={n} inverse={inverse} bin {k}: fast=({}, {}) naive=({}, {}) tol={tol:e}",
                        re[k],
                        im[k],
                        want_re[k],
                        want_im[k]
                    );
                }
            }
        }
    }

    /// Round-tripping must return the input scaled by `n`, because BOTH
    /// directions are unnormalized. This is the contract the surrogate relies on
    /// when it applies its own guarded `1/n`.
    #[test]
    fn forward_then_inverse_is_multiplication_by_the_length() {
        for &n in &[5usize, 16, 33, 97, 256, 300] {
            let (re0, im0) = sequence(n, 0xBEEF + n as u64);
            let mut re = re0.clone();
            let mut im = im0.clone();
            dft_in_place(&mut re, &mut im, false).expect("forward");
            dft_in_place(&mut re, &mut im, true).expect("inverse");
            for k in 0..n {
                let want_re = re0[k] * n as f64;
                let want_im = im0[k] * n as f64;
                let tol = 1.0e-10 * (n as f64);
                assert!(
                    (re[k] - want_re).abs() <= tol && (im[k] - want_im).abs() <= tol,
                    "n={n} index {k}: round trip gave ({}, {}), expected ({want_re}, {want_im})",
                    re[k],
                    im[k]
                );
            }
        }
    }

    #[test]
    fn mismatched_halves_are_refused() {
        let mut re = vec![0.0_f64; 4];
        let mut im = vec![0.0_f64; 3];
        assert!(dft_in_place(&mut re, &mut im, false).is_err());
    }
}
