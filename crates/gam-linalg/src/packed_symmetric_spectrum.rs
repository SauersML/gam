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
//! and `O(threads · n)` reduction buffers.

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

/// Iterations the implicit-shift QL sweep may spend on ONE eigenvalue before
/// the routine reports non-convergence rather than returning an unconverged
/// diagonal.
///
/// The shifted QL iteration converges cubically on a symmetric tridiagonal and
/// the classical implementations (EISPACK `tql2`, LAPACK `dsteqr`) allow 30
/// sweeps per eigenvalue; 30 is therefore the number this shares with them, and
/// exceeding it is a failure to report, never a tolerance to widen.
const QL_MAX_SWEEPS_PER_EIGENVALUE: usize = 30;

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
/// * QL non-convergence within `QL_MAX_SWEEPS_PER_EIGENVALUE` per eigenvalue.
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

    let (mut diagonal, mut offdiagonal) = tridiagonalize_packed_with_probe(n, packed, probe);
    // The reduction is scale-invariant by construction and cannot manufacture a
    // non-finite entry from finite input; this says so out loud rather than
    // letting a NaN reach the QL sweep, where `NaN <= floor` is false forever
    // and the failure is reported as a non-convergence at an index that means
    // nothing. Costs `O(n)` against the `O(n³)` above.
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
    implicit_ql_with_probe(&mut diagonal, &mut offdiagonal, probe)?;
    sort_spectrum_ascending(&mut diagonal, probe);
    Ok(diagonal)
}

/// Householder-reduce a packed-upper symmetric matrix to tridiagonal form,
/// applying every reflector to `probe` so it leaves holding `Qᵀw`.
///
/// Returns `(d, e)`: the diagonal (length `n`) and the sub/super-diagonal
/// (length `n`, with `e[n-1] = 0` so the QL sweep can index it uniformly).
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
    if m < PARALLEL_MIN_ROWS || rayon::current_num_threads() < 2 {
        p.fill(0.0);
        serial_packed_symmetric_matvec(m, packed, v, p, 0, m);
        return;
    }
    // Every row scatters into columns to its right, so the partial products do
    // not partition by output index; each task accumulates a full-width partial
    // and the reduction adds them. `threads × m` doubles, against the `m²/2`
    // triangle the kernel is reading — accounted for in the caller's budget as
    // an `O(m)` term.
    let tasks = rayon::current_num_threads().min(m.div_ceil(PARALLEL_MIN_ROWS)).max(1);
    let chunk = m.div_ceil(tasks);
    let partials: Vec<Vec<f64>> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let lo = task * chunk;
            let hi = ((task + 1) * chunk).min(m);
            let mut local = vec![0.0_f64; m];
            if lo < hi {
                serial_packed_symmetric_matvec(m, packed, v, &mut local, lo, hi);
            }
            local
        })
        .collect();
    p.fill(0.0);
    for local in &partials {
        for (target, &value) in p.iter_mut().zip(local.iter()) {
            *target += value;
        }
    }
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
    if m < PARALLEL_MIN_ROWS || rayon::current_num_threads() < 2 {
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

/// Implicit-shift QL on a symmetric tridiagonal, accumulating every rotation
/// into `probe` (a single row of the eigenvector matrix's transpose) instead of
/// into an `n × n` accumulator.
///
/// On entry `probe` holds `Qᵀw`; on return it holds `Wᵀ(Qᵀw) = Vᵀw` in the
/// order the (unsorted) `diagonal` ends in.
fn implicit_ql_with_probe(
    diagonal: &mut [f64],
    offdiagonal: &mut [f64],
    probe: &mut [f64],
) -> Result<(), String> {
    let n = diagonal.len();
    if n <= 1 {
        return Ok(());
    }
    offdiagonal[n - 1] = 0.0;
    // Absolute deflation floor: `eps · ‖T‖_∞`.
    //
    // THE RELATIVE TEST ALONE DOES NOT TERMINATE, and the failure is not
    // exotic — it is what a rank-deficient Gram produces every time. On
    // `F Fᵀ` with `F` of `296 × 148` standard normals, `‖T‖ ≈ 9·10²` while the
    // 148 null directions arrive as `d ≈ 10⁻¹³`, `e ≈ 10⁻¹³`. The classical
    // criterion asks `|e_i| ⩽ ε(|d_i| + |d_{i+1}|) ≈ 4·10⁻²⁹` there, which the
    // plane rotations cannot reach: every sweep re-injects rounding of order
    // `ε‖T‖ ≈ 2·10⁻¹³`. The sweep count then runs out on an eigenvalue that was
    // already correct to every digit the arithmetic holds.
    //
    // Deflating at `ε‖T‖` perturbs `T` by exactly the amount its own
    // factorization already carries, so the eigenvalues move by no more than
    // the accuracy any backward-stable dense method delivers. What it forfeits
    // is RELATIVE accuracy on eigenvalues below that floor — which is not a
    // quantity this routine ever promised, and its certified consumer discards
    // every mode inside its own `ε·rank·θ_max` floor as a null direction, a
    // floor `rank` times WIDER than this one.
    //
    // The relative test is kept as well, and taken first: where it does apply
    // (a graded matrix whose small eigenvalues are determined to high relative
    // accuracy) it deflates earlier and gives up nothing.
    let mut norm = 0.0_f64;
    for i in 0..n {
        let row = diagonal[i].abs()
            + if i > 0 { offdiagonal[i - 1].abs() } else { 0.0 }
            + offdiagonal[i].abs();
        norm = norm.max(row);
    }
    let deflation_floor = f64::EPSILON * norm;
    for l in 0..n {
        let mut sweeps = 0usize;
        loop {
            // Split at the first negligible off-diagonal at or after `l`: the
            // classical "adding it to the neighbouring diagonal magnitudes does
            // not change them", or the absolute floor derived above.
            let mut split = l;
            while split + 1 < n {
                let scale = diagonal[split].abs() + diagonal[split + 1].abs();
                if offdiagonal[split].abs() + scale == scale
                    || offdiagonal[split].abs() <= deflation_floor
                {
                    break;
                }
                split += 1;
            }
            if split == l {
                break;
            }
            if sweeps == QL_MAX_SWEEPS_PER_EIGENVALUE {
                return Err(format!(
                    "packed symmetric spectrum: implicit QL did not deflate eigenvalue {l} in \
                     {QL_MAX_SWEEPS_PER_EIGENVALUE} sweeps (block {l}..={split})"
                ));
            }
            sweeps += 1;

            // Wilkinson shift, formed from the leading 2x2 of the active block.
            let mut g = (diagonal[l + 1] - diagonal[l]) / (2.0 * offdiagonal[l]);
            let mut r = g.hypot(1.0);
            g = diagonal[split] - diagonal[l]
                + offdiagonal[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut deflated_early = false;
            for i in (l..split).rev() {
                let mut f = s * offdiagonal[i];
                let b = c * offdiagonal[i];
                r = f.hypot(g);
                offdiagonal[i + 1] = r;
                if r == 0.0 {
                    // An exactly-zero rotation radius splits the block here;
                    // recover the shift and restart the sweep.
                    diagonal[i + 1] -= p;
                    offdiagonal[split] = 0.0;
                    deflated_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = diagonal[i + 1] - p;
                r = (diagonal[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diagonal[i + 1] = g + p;
                g = c * r - b;
                f = probe[i + 1];
                probe[i + 1] = s * probe[i] + c * f;
                probe[i] = c * probe[i] - s * f;
            }
            if deflated_early {
                continue;
            }
            diagonal[l] -= p;
            offdiagonal[l] = g;
            offdiagonal[split] = 0.0;
        }
    }
    Ok(())
}

/// Sort `(eigenvalue, probe)` pairs ascending by eigenvalue, keeping the
/// pairing exact.
fn sort_spectrum_ascending(diagonal: &mut [f64], probe: &mut [f64]) {
    let n = diagonal.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        diagonal[a]
            .partial_cmp(&diagonal[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
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
    use crate::utils::splitmix64;
    use ndarray::{Array1, Array2};

    /// Deterministic standard normals from the crate's own SplitMix64 step, so
    /// these gates need no dependency the crate does not already carry.
    struct Normals(u64);

    impl Normals {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn uniform(&mut self) -> f64 {
            // 53-bit mantissa in the open interval (0, 1): the Box-Muller log
            // below is undefined at exactly 0.
            let bits = splitmix64(&mut self.0) >> 11;
            (bits as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u = self.uniform();
            let v = self.uniform();
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Reference `(Θ, Vᵀw)` from a full eigendecomposition, for the agreement
    /// gates below.
    fn reference_spectrum(dense: &Array2<f64>, w: &Array1<f64>) -> (Vec<f64>, Vec<f64>) {
        use crate::faer_ndarray::FaerEigh;
        let (values, vectors) = dense.eigh(faer::Side::Lower).expect("reference eigh");
        let n = dense.nrows();
        let mut projected = vec![0.0_f64; n];
        for (j, target) in projected.iter_mut().enumerate() {
            let mut accumulated = 0.0;
            for i in 0..n {
                accumulated += vectors[(i, j)] * w[i];
            }
            *target = accumulated;
        }
        (values.to_vec(), projected)
    }

    fn pack_upper(dense: &Array2<f64>) -> Vec<f64> {
        let n = dense.nrows();
        let mut packed = vec![0.0_f64; packed_upper_len(n)];
        for i in 0..n {
            for j in i..n {
                packed[packed_upper_index(n, i, j)] = dense[(i, j)];
            }
        }
        packed
    }

    /// Compare two `(Θ, Vᵀw)` readings as the SPECTRAL MEASURE they represent,
    /// which is the only thing either of them determines.
    ///
    /// Eigenvectors inside a degenerate eigenspace are not unique — a
    /// rank-deficient Gram has a whole null space of them — so an entrywise
    /// comparison of `Vᵀw` compares two arbitrary bases and fails on correct
    /// output. What IS determined is the measure `μ = Σ_i (Vᵀw)_i² δ_{θ_i}`, and
    /// that is also exactly what the certified caller consumes: its residual
    /// moments are `∫(θ+λ)^{−k} dμ`. So the gate is on the measure — cluster
    /// masses, and the moments themselves over a wide λ sweep.
    fn assert_spectral_measures_agree(
        label: &str,
        values: &[f64],
        probe: &[f64],
        reference_values: &[f64],
        reference_probe: &[f64],
    ) {
        let scale = reference_values
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let energy: f64 = reference_probe.iter().map(|v| v * v).sum();
        // Cluster mass: every maximal run of eigenvalues within the resolution
        // of the decomposition is ONE eigenspace, and its mass is basis-free.
        let cluster_tolerance = 1e-9 * scale;
        let mut start = 0usize;
        while start < values.len() {
            let mut end = start + 1;
            while end < values.len() && values[end] - values[end - 1] <= cluster_tolerance {
                end += 1;
            }
            let got: f64 = probe[start..end].iter().map(|v| v * v).sum();
            let want: f64 = reference_probe[start..end].iter().map(|v| v * v).sum();
            assert!(
                (got - want).abs() <= 1e-7 * (energy + 1.0),
                "{label}: eigenspace {start}..{end} mass {got} vs {want}"
            );
            start = end;
        }
        // The moments the caller actually evaluates, across the λ range a
        // certified search sweeps.
        for exponent in [-6.0_f64, -2.0, 0.0, 2.0, 6.0] {
            let lambda = 10.0_f64.powf(exponent) * scale.max(1.0);
            for power in 1..=4 {
                let moment = |theta: &[f64], mass: &[f64]| -> f64 {
                    theta
                        .iter()
                        .zip(mass.iter())
                        .map(|(&t, &c)| c * c / (t + lambda).powi(power))
                        .sum()
                };
                let got = moment(values, probe);
                let want = moment(reference_values, reference_probe);
                assert!(
                    (got - want).abs() <= 1e-8 * want.abs().max(f64::MIN_POSITIVE),
                    "{label}: S_{power}(lambda={lambda:.3e}) {got} vs {want}"
                );
            }
        }
    }

    fn random_symmetric(rng: &mut Normals, n: usize, rank: usize) -> Array2<f64> {
        // `F Fᵀ` with `rank` factor columns: positive semidefinite, and rank
        // deficient on purpose whenever `rank < n` (the cascade's null modes).
        let mut factor = Array2::<f64>::zeros((n, rank));
        for value in factor.iter_mut() {
            *value = rng.normal();
        }
        factor.dot(&factor.t())
    }

    #[test]
    fn spectrum_and_probe_match_a_full_eigendecomposition() {
        let mut rng = Normals::new(0x2758_0001);
        for &n in &[1usize, 2, 3, 5, 17, 64, 129] {
            for &rank in &[n, n.div_ceil(2), 1] {
                let dense = random_symmetric(&mut rng, n, rank);
                let w = Array1::from_shape_fn(n, |_| rng.normal());
                let (reference_values, reference_probe) = reference_spectrum(&dense, &w);
                let mut packed = pack_upper(&dense);
                let mut probe = w.to_vec();
                let values = packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe)
                    .expect("packed spectrum");
                let scale = reference_values
                    .iter()
                    .fold(0.0_f64, |acc, v| acc.max(v.abs()))
                    .max(f64::MIN_POSITIVE);
                for (index, (&got, &want)) in values.iter().zip(reference_values.iter()).enumerate()
                {
                    assert!(
                        (got - want).abs() <= 1e-10 * scale,
                        "n={n} rank={rank} eigenvalue {index}: {got} vs {want}"
                    );
                }
                assert_spectral_measures_agree(
                    &format!("n={n} rank={rank}"),
                    &values,
                    &probe,
                    &reference_values,
                    &reference_probe,
                );
            }
        }
    }

    #[test]
    fn the_probe_is_an_isometry_and_the_spectrum_carries_the_matrix_invariants() {
        // Three identities that need no reference implementation: `V` is
        // orthogonal, so `‖Vᵀw‖ = ‖w‖` exactly; and the spectrum reproduces the
        // trace and the Frobenius norm of the matrix it came from.
        let mut rng = Normals::new(0x2758_0002);
        for &n in &[2usize, 9, 40, 137] {
            let dense = random_symmetric(&mut rng, n, n);
            let w = Array1::from_shape_fn(n, |_| rng.normal());
            let trace: f64 = (0..n).map(|i| dense[(i, i)]).sum();
            let frobenius: f64 = dense.iter().map(|v| v * v).sum();
            let energy: f64 = w.iter().map(|v| v * v).sum();
            let mut packed = pack_upper(&dense);
            let mut probe = w.to_vec();
            let values = packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe)
                .expect("packed spectrum");
            let spectral_trace: f64 = values.iter().sum();
            let spectral_frobenius: f64 = values.iter().map(|v| v * v).sum();
            let probe_energy: f64 = probe.iter().map(|v| v * v).sum();
            assert!(
                (spectral_trace - trace).abs() <= 1e-9 * (1.0 + trace.abs()),
                "n={n} trace {spectral_trace} vs {trace}"
            );
            assert!(
                (spectral_frobenius - frobenius).abs() <= 1e-9 * (1.0 + frobenius),
                "n={n} frobenius {spectral_frobenius} vs {frobenius}"
            );
            assert!(
                (probe_energy - energy).abs() <= 1e-9 * (1.0 + energy),
                "n={n} probe energy {probe_energy} vs {energy}"
            );
        }
    }

    #[test]
    fn ascending_order_pairs_every_eigenvalue_with_its_own_projection() {
        // A diagonal matrix makes the pairing checkable by inspection: the
        // projection onto eigenvalue `d_i` must be `±w_i`, whatever order the
        // sort puts them in.
        let n = 12;
        let diagonal: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.5 - 2.0).collect();
        let mut dense = Array2::<f64>::zeros((n, n));
        for (i, &value) in diagonal.iter().enumerate() {
            dense[(i, i)] = value;
        }
        let w = Array1::from_shape_fn(n, |i| 1.0 + i as f64);
        let mut packed = pack_upper(&dense);
        let mut probe = w.to_vec();
        let values =
            packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe).expect("spectrum");
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]), "not ascending: {values:?}");
        for (value, projection) in values.iter().zip(probe.iter()) {
            let source = diagonal
                .iter()
                .position(|d| (d - value).abs() <= 1e-12)
                .expect("every eigenvalue is one of the diagonal entries");
            assert!(
                (projection.abs() - w[source]).abs() <= 1e-12,
                "eigenvalue {value} paired with {projection} instead of ±{}",
                w[source]
            );
        }
    }

    #[test]
    fn a_zero_matrix_and_a_zero_probe_are_handled_exactly() {
        for n in [1usize, 4, 33] {
            let mut packed = vec![0.0_f64; packed_upper_len(n)];
            let mut probe = vec![0.0_f64; n];
            let values =
                packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe).expect("spectrum");
            assert_eq!(values.len(), n);
            assert!(values.iter().all(|v| *v == 0.0), "zero matrix: {values:?}");
            assert!(probe.iter().all(|v| *v == 0.0), "zero probe: {probe:?}");
        }
        // A zero probe against a nontrivial matrix must stay exactly zero: the
        // reflectors and rotations are linear in it.
        let mut rng = Normals::new(0x2758_0003);
        let dense = random_symmetric(&mut rng, 24, 24);
        let mut packed = pack_upper(&dense);
        let mut probe = vec![0.0_f64; 24];
        packed_symmetric_spectrum_with_probe(24, &mut packed, &mut probe).expect("spectrum");
        assert!(probe.iter().all(|v| *v == 0.0), "zero probe moved: {probe:?}");
    }

    #[test]
    fn extreme_magnitudes_do_not_overflow_the_reflector_norm() {
        // A row whose entries square to infinity in naive arithmetic: the
        // scaled norm keeps the reduction finite, which is the property the
        // certified caller needs when a whitened Schur block is badly scaled.
        let n = 6;
        let mut dense = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            dense[(i, i)] = 1.0e200;
            if i + 1 < n {
                dense[(i, i + 1)] = 3.0e199;
                dense[(i + 1, i)] = 3.0e199;
            }
        }
        let w = Array1::from_shape_fn(n, |i| 1.0 + i as f64);
        let (reference_values, _) = reference_spectrum(&dense, &w);
        let mut packed = pack_upper(&dense);
        let mut probe = w.to_vec();
        let values =
            packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe).expect("spectrum");
        assert!(values.iter().all(|v| v.is_finite()), "overflowed: {values:?}");
        for (&got, &want) in values.iter().zip(reference_values.iter()) {
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "scaled eigenvalue {got} vs {want}"
            );
        }
    }

    #[test]
    fn a_denormal_scale_row_next_to_an_ordinary_one_does_not_reduce_to_nan() {
        // The reflector's `1/(alpha - beta)` overflows on a row whose entries
        // have decayed to the denormal range, and `0 · inf` on that row's exact
        // zeros then writes NaN into the whole trailing block. This is that row,
        // planted beside an ordinary-scale block so the matrix norm cannot be
        // rescaled away by the caller — which is exactly the shape a
        // rank-deficient Gram reduces to after a thousand Householder steps.
        let n = 8;
        let tiny = f64::MIN_POSITIVE * 4.0;
        let mut dense = Array2::<f64>::zeros((n, n));
        for i in 0..4 {
            for j in i..4 {
                let value = 1.0 / (1.0 + (i + j) as f64);
                dense[(i, j)] = value;
                dense[(j, i)] = value;
            }
        }
        // Row 4 reaches into the denormal block with zeros interleaved: the
        // zeros are what turn an overflowed scale into NaN rather than infinity.
        for j in 5..n {
            let value = if j % 2 == 0 { tiny } else { 0.0 };
            dense[(4, j)] = value;
            dense[(j, 4)] = value;
        }
        for i in 4..n {
            dense[(i, i)] = tiny;
        }
        let w = Array1::from_shape_fn(n, |i| 1.0 + i as f64);
        let (reference_values, reference_probe) = reference_spectrum(&dense, &w);
        let mut packed = pack_upper(&dense);
        let mut probe = w.to_vec();
        let values = packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe)
            .expect("a denormal-scale block must reduce, not refuse");
        assert!(
            values.iter().all(|v| v.is_finite()) && probe.iter().all(|v| v.is_finite()),
            "reduction produced non-finite output: values {values:?} probe {probe:?}"
        );
        assert_spectral_measures_agree(
            "denormal",
            &values,
            &probe,
            &reference_values,
            &reference_probe,
        );
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

    #[test]
    fn the_parallel_and_serial_kernels_agree_on_a_block_wider_than_the_threshold() {
        // `PARALLEL_MIN_ROWS` switches the matvec and the rank-2 update to a
        // Rayon fan-out; the two must produce the same spectrum, so the width
        // at which a design is certified cannot depend on the thread count.
        let n = PARALLEL_MIN_ROWS + 40;
        let mut rng = Normals::new(0x2758_0004);
        let dense = random_symmetric(&mut rng, n, n / 2);
        let w = Array1::from_shape_fn(n, |_| rng.normal());
        let (reference_values, reference_probe) = reference_spectrum(&dense, &w);
        let mut packed = pack_upper(&dense);
        let mut probe = w.to_vec();
        let values =
            packed_symmetric_spectrum_with_probe(n, &mut packed, &mut probe).expect("spectrum");
        let scale = reference_values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        for (&got, &want) in values.iter().zip(reference_values.iter()) {
            assert!(
                (got - want).abs() <= 1e-9 * scale,
                "wide eigenvalue {got} vs {want}"
            );
        }
        assert_spectral_measures_agree(
            "wide",
            &values,
            &probe,
            &reference_values,
            &reference_probe,
        );
    }
}
