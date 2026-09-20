//! Full spectrum of a symmetric matrix held as a PACKED upper triangle,
//! together with the projection of one vector onto its eigenbasis — without
//! ever materializing the eigenvector matrix.
//!
//! # Why this exists
//!
//! Some certified criteria need `(Θ, Vᵀw)` for `A = VΘVᵀ`: every eigenvalue,
//! and the coordinates of ONE vector in the eigenbasis. A general
//! eigendecomposition hands that over by building the whole `n × n` `V` (plus
//! whatever workspace its tridiagonalization allocates), so the caller's live
//! memory is several `n²` blocks when the mathematics needs one packed triangle
//! and `O(n)` vectors. Where the admissible problem width is DERIVED from a
//! memory budget — `gam_solve::residual_cascade`'s certified Schur spectrum is
//! the motivating case (#2758) — every one of those blocks is a `1/√blocks`
//! factor on the widest design that can be certified at all.
//!
//! # The identity the routine is built on
//!
//! Householder tridiagonalization gives `A = Q T Qᵀ` with `Q = H₀H₁⋯H_{n−3}`,
//! and the symmetric tridiagonal `T = W Θ Wᵀ`. So `V = QW` and
//!
//! ```text
//!     Vᵀw = Wᵀ(Qᵀw) = Wᵀ q,      q = H_{n−3}⋯H₀ w.
//! ```
//!
//! `q` is accumulated by applying each reflector to a single vector as it is
//! formed, and `Wᵀq` by applying every implicit-QL Givens rotation to that same
//! single vector instead of to an `n × n` accumulator — the classical
//! Golub–Welsch "keep one row of the eigenvector matrix" device, here with a
//! general start vector rather than `e₁`. Neither `Q` nor `W` is ever formed.
//!
//! # Cost
//!
//! Time is `O(n³)` (the tridiagonalization; `4n³/3` flops) plus `O(n²)` for the
//! QL sweep — the same order a dense eigendecomposition pays. Memory is the
//! caller's packed triangle, destroyed in place, plus `O(n)` working vectors
//! and `O(threads · log(n) · n)` reduction buffers.

use rayon::prelude::*;

/// Offset of the first stored entry of row `i` in a row-major packed UPPER
/// triangle of an `n × n` symmetric matrix. Row `i` stores columns `i..n`
/// contiguously, so `entry (i, j)` for `i <= j` lives at
/// `packed_upper_row_offset(n, i) + (j - i)`.
#[inline]
#[must_use]
pub const fn packed_upper_row_offset(n: usize, i: usize) -> usize {
    // i*n - i*(i-1)/2 ; written to stay exact in integer arithmetic.
    i * n - (i * i).wrapping_sub(i) / 2
}

/// Number of `f64` a row-major packed upper triangle of an `n × n` symmetric
/// matrix occupies.
#[inline]
#[must_use]
pub const fn packed_upper_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Rows of the trailing block below which the symmetric matrix-vector product
/// and rank-2 update stay serial. Rayon's fork/join and the per-task `O(m)`
/// reduction buffer cost more than the `O(m²)` kernel below this size.
const PARALLEL_MIN_ROWS: usize = 256;

/// Full spectrum of a packed-upper symmetric matrix together with `Vᵀw`.
///
/// * `n` — matrix dimension.
/// * `packed` — row-major packed UPPER triangle, `n(n+1)/2` entries.
///   **Destroyed**: it is the tridiagonalization's working store.
/// * `probe` — on entry the vector `w` (length `n`); on return `Vᵀw`, permuted
///   into the same ascending order as the returned eigenvalues.
///
/// Returns the eigenvalues in ASCENDING order. The pairing is exact: entry `i`
/// of `probe` is the coordinate of `w` along the unit eigenvector belonging to
/// eigenvalue `i`. Eigenvector SIGN is not determined (it never is), so only
/// sign-independent functionals of `probe` — squares, and sums of them — are
/// reproducible across implementations.
///
/// # Errors
///
/// * a length mismatch between `n`, `packed` and `probe`;
/// * a non-finite entry in `packed` or `probe`;
/// * an eigenvalue or projected probe component outside the finite `f64` range;
/// * QL non-convergence within the `30·n` sweeps
///   [`gam_math::quadrature::symmetric_tridiagonal_eigen_with_probe`] allows.
pub fn packed_symmetric_spectrum_with_probe(
    n: usize,
    packed: &mut [f64],
    probe: &mut [f64],
) -> Result<Vec<f64>, String> {
    if packed.len() != packed_upper_len(n) {
        return Err(format!(
            "packed symmetric spectrum: packed triangle has {} entries but dimension {n} needs {}",
            packed.len(),
            packed_upper_len(n)
        ));
    }
    if probe.len() != n {
        return Err(format!(
            "packed symmetric spectrum: probe has {} entries but dimension is {n}",
            probe.len()
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if let Some(bad) = packed.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "packed symmetric spectrum: packed entry {bad} is not finite ({})",
            packed[bad]
        ));
    }
    if let Some(bad) = probe.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "packed symmetric spectrum: probe entry {bad} is not finite ({})",
            probe[bad]
        ));
    }

    // Positive rescaling preserves the eigenvectors and their ordering. Work
    // at unit entry scale so QL's sums of adjacent diagonal magnitudes cannot
    // overflow and its shifts/rotations do not underflow on a tiny matrix.
    // In particular, diag=(1e308,1e308), offdiag=5e307 must not deflate merely
    // because the two diagonal magnitudes sum to infinity.
    let matrix_scale = packed.iter().fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if matrix_scale == 0.0 {
        // The identity eigenbasis is a valid choice for the zero matrix.
        return Ok(vec![0.0; n]);
    }
    for value in packed.iter_mut() {
        *value /= matrix_scale;
    }
    // A Householder update can overflow its intermediate dot product even
    // when every final coordinate of Qᵀw is finite. Normalize the probe
    // independently and restore its units after the same orthogonal maps.
    let probe_scale = probe.iter().fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if probe_scale > 0.0 {
        for value in probe.iter_mut() {
            *value /= probe_scale;
        }
    }

    let (diagonal, offdiagonal) = tridiagonalize_packed_with_probe(n, packed, probe);
    // Refuse a broken reduction here, where the input is known finite and the
    // fault is the Householder step's; the QL solver would refuse it too, but
    // only as an anonymous non-finite tridiagonal. Costs O(n) against O(n³).
    let broken = diagonal
        .iter()
        .chain(offdiagonal.iter())
        .chain(probe.iter())
        .position(|value| !value.is_finite());
    if let Some(index) = broken {
        return Err(format!(
            "packed symmetric spectrum: the Householder reduction of a finite {n}x{n} matrix \
             produced a non-finite tridiagonal entry (flat index {index} over d, e, probe)"
        ));
    }
    // `Wᵀ(Qᵀw)` by the workspace's single implicit-QL solver, which carries
    // the probe through its rotations and deflates at `ε‖T‖_∞`.
    let mut diagonal = gam_math::quadrature::symmetric_tridiagonal_eigen_with_probe(
        &diagonal,
        &offdiagonal[..n - 1],
        probe,
    )
    .map_err(|error| format!("packed symmetric spectrum: {error}"))?;
    sort_spectrum_ascending(&mut diagonal, probe);
    for value in diagonal.iter_mut() {
        *value *= matrix_scale;
    }
    if probe_scale > 0.0 {
        for value in probe.iter_mut() {
            *value *= probe_scale;
        }
    }
    if diagonal.iter().chain(probe.iter()).any(|value| !value.is_finite()) {
        return Err("packed symmetric spectrum: an eigenvalue or projected probe component is not representable as a finite f64".to_string());
    }
    Ok(diagonal)
}

/// Householder-reduce a packed-upper symmetric matrix to tridiagonal form,
/// applying every reflector to `probe` so it leaves holding `Qᵀw`.
///
/// Returns `(d, e)`: the diagonal (length `n`) and the sub/super-diagonal
/// (length `n`, with `e[n-1] = 0`; the QL solver takes the leading `n - 1`).
fn tridiagonalize_packed_with_probe(
    n: usize,
    packed: &mut [f64],
    probe: &mut [f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut diagonal = vec![0.0_f64; n];
    let mut offdiagonal = vec![0.0_f64; n];
    // Reflector, its image under the trailing block, and the rank-2 partner.
    // Allocated once at full width and used through their leading `m` entries.
    let mut reflector = vec![0.0_f64; n];
    let mut image = vec![0.0_f64; n];
    let mut partner = vec![0.0_f64; n];

    // `tail` is always the packed upper triangle of the ACTIVE block, whose
    // first row is the row being eliminated. That the trailing block is again
    // contiguous is a property of this layout, not a coincidence: row `r`
    // stores columns `r..n`, so the rows of the block `k+1..n` are exactly the
    // stored rows from `k+1` on, in order.
    let mut tail: &mut [f64] = packed;
    for k in 0..n.saturating_sub(1) {
        let m = n - 1 - k;
        diagonal[k] = tail[0];
        if m == 1 {
            // One off-diagonal entry left: already tridiagonal, no reflector.
            offdiagonal[k] = tail[1];
            let (_row, rest) = tail.split_at_mut(2);
            tail = rest;
            continue;
        }

        // `dlarfg` on x = A[k, k+1..n]: choose `beta`, `tau` and a unit-leading
        // reflector `v` with `(I - tau v vᵀ) x = beta e₁`.
        //
        // BUILT ON THE ROW NORMALIZED BY ITS OWN LARGEST ENTRY, which is not a
        // refinement — the unscaled form produces NaN and it did. `tau` and `v`
        // are invariant to a positive rescaling of `x`, but the intermediate
        // `1/(alpha - beta)` is not: on a row whose entries have decayed to the
        // denormal range — what the trailing block of a rank-deficient Gram
        // becomes after a thousand reductions, and this cascade's design is 89%
        // columns the data cannot pin — that reciprocal OVERFLOWS to infinity,
        // and `0 · inf` on the row's exact zeros writes NaN into the reflector.
        // The whole trailing block is NaN from there, the tridiagonal comes out
        // NaN, and QL then spins to its sweep limit on an eigenvalue that never
        // existed. Measured: `the_spectral_residual_carries_no_null_modes`, NaN
        // at index 1454 of 1722, reported as a non-convergence.
        //
        // After normalization `|alpha_s - beta_s| = |alpha_s| + hypot(...) >= 1`
        // by construction, so the reciprocal cannot overflow at any input scale,
        // and every `v` entry is bounded by 1.
        let x = &tail[1..=m];
        let largest = x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let (beta, tau) = if largest == 0.0 {
            (0.0, 0.0)
        } else {
            let alpha = x[0] / largest;
            let tail_norm = vector_norm_scaled(&x[1..], largest);
            if tail_norm == 0.0 {
                // Already in the required form; a zero `tau` is the exact
                // identity reflector, so no update is applied at all below.
                (x[0], 0.0)
            } else {
                let magnitude = alpha.hypot(tail_norm);
                // `beta` takes the sign OPPOSITE to `alpha` so that
                // `alpha - beta` is an addition of like-signed quantities: the
                // cancellation-free choice, and the reason `dlarfg` does the
                // same.
                let beta = if alpha >= 0.0 { -magnitude } else { magnitude };
                let tau = (beta - alpha) / beta;
                let scale = 1.0 / (alpha - beta);
                reflector[0] = 1.0;
                for i in 1..m {
                    reflector[i] = (x[i] / largest) * scale;
                }
                (beta * largest, tau)
            }
        };
        offdiagonal[k] = beta;

        // Row `k` is never read again — `diagonal[k]` and `offdiagonal[k]` hold
        // everything the tridiagonal form keeps of it — so the zeros the
        // reflector introduces there are not written.
        let (_row_k, rest) = tail.split_at_mut(m + 1);
        tail = rest;

        if tau != 0.0 {
            let v = &reflector[..m];
            let p = &mut image[..m];
            packed_symmetric_matvec(m, tail, v, p);
            for value in p.iter_mut() {
                *value *= tau;
            }
            let correction = -0.5 * tau * dot(p, v);
            for (target, (&pi, &vi)) in partner[..m].iter_mut().zip(p.iter().zip(v.iter())) {
                *target = pi + correction * vi;
            }
            packed_symmetric_rank2_downdate(m, tail, v, &partner[..m]);

            // `q := H q` on the same index range, which is what makes `probe`
            // hold `Qᵀw` when the loop ends.
            let block = &mut probe[k + 1..];
            let scale = tau * dot(v, block);
            for (target, &vi) in block.iter_mut().zip(v.iter()) {
                *target -= scale * vi;
            }
        }
    }
    diagonal[n - 1] = tail[0];
    (diagonal, offdiagonal)
}

/// `p := S v` for the symmetric `m × m` `S` held as a row-major packed upper
/// triangle. `p` is fully overwritten.
fn packed_symmetric_matvec(m: usize, packed: &[f64], v: &[f64], p: &mut [f64]) {
    if m < PARALLEL_MIN_ROWS {
        p.fill(0.0);
        serial_packed_symmetric_matvec(m, packed, v, p, 0, m);
        return;
    }
    // Every row scatters into columns to its right, so the partial products do
    // not partition by output index; each block of rows accumulates a
    // full-width partial and the partials are added over a fixed pairwise tree.
    // The blocks are a function of `m` alone, so the product's bits are the
    // same at every pool width (blocks sized from the pool width made the
    // spectrum follow `RAYON_NUM_THREADS`). The partials live along the fold's
    // join stacks — `O(threads · log(m) · m)` doubles against the `m²/2`
    // triangle the kernel is reading.
    let block = PARALLEL_MIN_ROWS;
    let sum = crate::pairwise_reduce::par_deterministic_block_fold_by_work(
        m.div_ceil(block),
        block,
        |blocks| {
            let lo = blocks.start * block;
            let hi = (blocks.end * block).min(m);
            let mut local = vec![0.0_f64; m];
            serial_packed_symmetric_matvec(m, packed, v, &mut local, lo, hi);
            local
        },
        |mut acc, other| {
            for (target, &value) in acc.iter_mut().zip(other.iter()) {
                *target += value;
            }
            acc
        },
    )
    .expect("m >= PARALLEL_MIN_ROWS leaves at least one block");
    p.copy_from_slice(&sum);
}

/// Accumulate rows `lo..hi` of the packed symmetric product into `p`.
fn serial_packed_symmetric_matvec(
    m: usize,
    packed: &[f64],
    v: &[f64],
    p: &mut [f64],
    lo: usize,
    hi: usize,
) {
    for i in lo..hi {
        let base = packed_upper_row_offset(m, i);
        let row = &packed[base..base + (m - i)];
        let vi = v[i];
        let mut accumulated = row[0] * vi;
        for (offset, &entry) in row.iter().enumerate().skip(1) {
            accumulated += entry * v[i + offset];
            p[i + offset] += entry * vi;
        }
        p[i] += accumulated;
    }
}

/// `S := S − v wᵀ − w vᵀ` on the packed upper triangle of the symmetric
/// `m × m` `S`.
fn packed_symmetric_rank2_downdate(m: usize, packed: &mut [f64], v: &[f64], w: &[f64]) {
    if m < PARALLEL_MIN_ROWS {
        serial_packed_symmetric_rank2_downdate(m, packed, v, w, 0);
        return;
    }
    // Row `i` occupies `m - i` contiguous entries, so the triangle splits into
    // disjoint per-row slices and the update is embarrassingly parallel.
    let mut rows: Vec<(usize, &mut [f64])> = Vec::with_capacity(m);
    let mut rest = packed;
    for i in 0..m {
        let (row, next) = rest.split_at_mut(m - i);
        rows.push((i, row));
        rest = next;
    }
    rows.into_par_iter().for_each(|(i, row)| {
        let vi = v[i];
        let wi = w[i];
        for (offset, entry) in row.iter_mut().enumerate() {
            *entry -= vi * w[i + offset] + wi * v[i + offset];
        }
    });
}

fn serial_packed_symmetric_rank2_downdate(
    m: usize,
    packed: &mut [f64],
    v: &[f64],
    w: &[f64],
    from_row: usize,
) {
    for i in from_row..m {
        let base = packed_upper_row_offset(m, i);
        let vi = v[i];
        let wi = w[i];
        for offset in 0..(m - i) {
            packed[base + offset] -= vi * w[i + offset] + wi * v[i + offset];
        }
    }
}

/// Sort `(eigenvalue, probe)` pairs ascending by eigenvalue, keeping the
/// pairing exact.
fn sort_spectrum_ascending(diagonal: &mut [f64], probe: &mut [f64]) {
    let n = diagonal.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        diagonal[a].total_cmp(&diagonal[b]).then(a.cmp(&b))
    });
    let sorted_diagonal: Vec<f64> = order.iter().map(|&i| diagonal[i]).collect();
    let sorted_probe: Vec<f64> = order.iter().map(|&i| probe[i]).collect();
    diagonal.copy_from_slice(&sorted_diagonal);
    probe.copy_from_slice(&sorted_probe);
}

/// `‖values / divisor‖`, with `divisor > 0`. Dividing first keeps the sum of
/// squares inside the exponent range whatever the row's magnitude is.
fn vector_norm_scaled(values: &[f64], divisor: f64) -> f64 {
    let mut sum_squares = 0.0_f64;
    for &value in values {
        let scaled = value / divisor;
        sum_squares += scaled * scaled;
    }
    sum_squares.sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_two_spectrum_and_probe_are_invariant_to_extreme_matrix_scale() {
        for scale in [1.0, 1.0e308, 1.0e-308, f64::from_bits(2)] {
            let mut packed = vec![scale, 0.5 * scale, scale];
            let mut probe = vec![3.0, 1.0];
            let eigenvalues = packed_symmetric_spectrum_with_probe(2, &mut packed, &mut probe)
                .unwrap();
            for (actual, expected) in eigenvalues.iter().zip([0.5, 1.5]) {
                assert!((actual / scale - expected).abs() < 4.0 * f64::EPSILON);
            }
            // The eigenvectors are (1,-1)/√2 and (1,1)/√2, up to sign.
            assert!((probe[0] * probe[0] - 2.0).abs() < 2.0e-14);
            assert!((probe[1] * probe[1] - 8.0).abs() < 2.0e-14);
        }
    }

    #[test]
    fn householder_spectrum_preserves_probe_mass_at_extreme_scales() {
        for scale in [1.0, 3.0e307, 1.0e-308] {
            // A/scale = I + 11ᵀ: spectrum (1,1,4), with top eigenvector
            // (1,1,1)/√3. Check the total mass of the tied lower eigenspace.
            let mut packed = vec![2.0 * scale, scale, scale, 2.0 * scale, scale, 2.0 * scale];
            let mut probe = vec![1.0, 2.0, 3.0];
            let eigenvalues = packed_symmetric_spectrum_with_probe(3, &mut packed, &mut probe)
                .unwrap();
            for (actual, expected) in eigenvalues.iter().zip([1.0, 1.0, 4.0]) {
                assert!((actual / scale - expected).abs() < 2.0e-14);
            }
            assert!((probe[0] * probe[0] + probe[1] * probe[1] - 2.0).abs() < 2.0e-13);
            assert!((probe[2] * probe[2] - 12.0).abs() < 2.0e-13);
        }
    }

    #[test]
    fn large_probe_is_scaled_without_changing_its_eigenbasis_coordinates() {
        let mut packed = vec![2.0, 1.0, 1.0, 2.0, 1.0, 2.0];
        let scale = 1.0e308;
        let mut probe = vec![scale; 3];
        packed_symmetric_spectrum_with_probe(3, &mut packed, &mut probe).unwrap();
        assert!((probe[0] / scale).abs() < 2.0e-14);
        assert!((probe[1] / scale).abs() < 2.0e-14);
        assert!((probe[2].abs() / scale - 3.0_f64.sqrt()).abs() < 2.0e-14);

        let mut zero = vec![0.0; 6];
        let original = probe.clone();
        assert_eq!(packed_symmetric_spectrum_with_probe(3, &mut zero, &mut probe).unwrap(),
            vec![0.0; 3]);
        assert_eq!(probe, original);
    }

    #[test]
    fn a_non_finite_input_is_refused_rather_than_decomposed() {
        let mut packed = vec![1.0, f64::NAN, 1.0];
        let mut probe = vec![1.0, 1.0];
        let error = packed_symmetric_spectrum_with_probe(2, &mut packed, &mut probe)
            .expect_err("a NaN entry must refuse");
        assert!(error.contains("not finite"), "unexpected error: {error}");

        let mut packed = vec![1.0, 0.0, 1.0];
        let mut probe = vec![1.0, f64::INFINITY];
        let error = packed_symmetric_spectrum_with_probe(2, &mut packed, &mut probe)
            .expect_err("a non-finite probe must refuse");
        assert!(error.contains("probe entry"), "unexpected error: {error}");
    }

    /// A rank-deficient Gram `F Fᵀ` delivers its null directions as a block
    /// of `d, e ≈ ε‖T‖` that the classical relative split test can never
    /// deflate; the shared QL solver's `ε‖T‖_∞` floor must, and the spectrum it
    /// returns must still be the Gram's: `rank` modes above the null floor,
    /// the trace, and the probe's mass.
    #[test]
    fn a_rank_deficient_gram_deflates_its_null_block() {
        let (n, rank) = (200usize, 100usize);
        let mut state = 0x5EED_u64;
        let mut factor = vec![0.0_f64; n * rank];
        for value in factor.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *value = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
        }
        let mut packed = Vec::with_capacity(packed_upper_len(n));
        let mut trace = 0.0_f64;
        for i in 0..n {
            for j in i..n {
                let entry = dot(
                    &factor[i * rank..(i + 1) * rank],
                    &factor[j * rank..(j + 1) * rank],
                );
                if i == j {
                    trace += entry;
                }
                packed.push(entry);
            }
        }
        let mut probe: Vec<f64> = (0..n).map(|i| (0.29 * i as f64).cos()).collect();
        let mass = dot(&probe, &probe);
        let eigenvalues = packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe)
            .expect("a rank-deficient Gram must decompose");
        let largest = eigenvalues[n - 1];
        // The exact zeros come back at `O(ε‖A‖)` and the smallest genuine mode
        // of this `200 × 100` factor sits near `‖A‖/30`; `√ε·θ_max` lies
        // geometrically between the two, so it separates them with room on
        // both sides.
        let null_floor = f64::EPSILON.sqrt() * largest;
        let above = eigenvalues.iter().filter(|&&value| value > null_floor).count();
        assert_eq!(above, rank, "the Gram has exactly {rank} non-null modes");
        let spectral_trace = eigenvalues.iter().sum::<f64>();
        let trace_tolerance = f64::EPSILON * (n * n) as f64 * largest;
        assert!(
            (spectral_trace - trace).abs() <= trace_tolerance,
            "{spectral_trace} vs {trace}"
        );
        assert!((dot(&probe, &probe) - mass).abs() <= f64::EPSILON * (n * n) as f64 * mass);
    }

    /// The packed matvec above `PARALLEL_MIN_ROWS` splits its rows into
    /// shape-sized blocks, so the spectrum and the probe carry the same words on
    /// every pool width.
    #[test]
    fn spectrum_words_do_not_depend_on_the_pool_width() {
        let m = 700usize;
        let mut state = 0x7ACC_u64;
        let mut packed = Vec::with_capacity(m * (m + 1) / 2);
        for i in 0..m {
            for j in i..m {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                packed.push(if i == j { 4.0 + unit } else { unit / m as f64 });
            }
        }
        let probe: Vec<f64> = (0..m).map(|i| (0.13 * i as f64).cos()).collect();
        let at_width = |width: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(width).build().expect("pool");
            pool.install(|| {
                let mut product = vec![0.0; m];
                packed_symmetric_matvec(m, &packed, &probe, &mut product);
                let mut reduced = packed.clone();
                let mut rotated = probe.clone();
                let values = packed_symmetric_spectrum_with_probe(m, &mut reduced, &mut rotated)
                    .expect("spectrum");
                values
                    .iter()
                    .chain(rotated.iter())
                    .chain(product.iter())
                    .map(|v| v.to_bits())
                    .collect::<Vec<u64>>()
            })
        };
        let single = at_width(1);
        for width in [2, 3, 8] {
            assert!(single == at_width(width), "pool width {width} changed the words");
        }
    }
}
