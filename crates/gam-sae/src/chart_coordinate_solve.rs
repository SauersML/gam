//! Batched rank-1 chart-coordinate E-step solver for the torch ManifoldSAE
//! trainer.
//!
//! Motivation
//! ----------
//! In the torch `softmax_topk` lane the per-atom on-manifold coordinate
//! `t_f(x)` is learned only through the encoder head, whose gradient reaches
//! the position through the basis jet times the (initially tiny) harmonic
//! decoder rows. At init that product is ~0, so the flat sub-model wins the
//! reconstruction and the positions freeze: decoded charts crumple to
//! filaments instead of opening onto the data manifold (see the BSF
//! manifold-zoo arena root cause). The cure is an explicit E-step: hold the
//! decoder fixed and solve each `(row, atom)` coordinate by projecting the
//! target onto the atom's *current* decoded curve, amplitude profiled out.
//! Those solved coordinates are E-step CONSTANTS on the torch tape (the
//! decoder gradient still flows through the basis evaluation at the solved
//! coordinate), and the encoder position head keeps learning by being pulled
//! toward the solved coordinates through a small alignment penalty.
//!
//! Math
//! ----
//! Atom `f` decodes the curve `m_f(t) = Σ_k φ_k(t) · D_f[k, :]` for a decoder
//! block `D_f ∈ R^{K×D}` and a period-1 basis `φ(t) ∈ R^K`. For a target
//! vector `y ∈ R^D` the amplitude-profiled reconstruction gain at coordinate
//! `t` (minimizing `‖y − a·m_f(t)‖²` over the SAE amplitude `a ≥ 0`) is
//!
//! ```text
//!     score_f(t) = max(⟨y, m_f(t)⟩, 0)² / ‖m_f(t)‖² .
//! ```
//!
//! The solver returns the global `argmax_t score_f(t)` by forming its analytic
//! trigonometric stationarity polynomial, enumerating every unit-circle root
//! through a companion-matrix eigenproblem, and comparing the score at all of
//! them. A first sweep uses `y = x` (the row); an optional second sweep uses the
//! leave-one-out residual target
//! `y_f = x − Σ_{g≠f} gate_g · m_g(t_g)` built from the previous sweep's
//! positions and gate weights, so co-active atoms carve complementary charts.

mod stationary_roots;

use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

pub(crate) use stationary_roots::PeriodicCurveExtrema;

/// Basis family the coordinate solver evaluates analytically. Periodic
/// (`n_harmonics`-harmonic Fourier) covers `atom_manifold="circle"` /
/// `atom_basis="fourier"` — the intrinsic-rank-1 default of the torch lane.
#[derive(Debug, Clone, Copy)]
pub enum ChartBasisKind {
    /// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]`, width
    /// `K = 2·H + 1`, identical to
    /// [`crate::basis::PeriodicHarmonicEvaluator`].
    Periodic { n_harmonics: usize },
}

impl ChartBasisKind {
    /// Number of basis columns `K` this kind produces (must match the decoder
    /// block's `K` dimension).
    pub(crate) fn width(&self) -> usize {
        match self {
            ChartBasisKind::Periodic { n_harmonics } => 2 * n_harmonics + 1,
        }
    }

    /// Evaluate the basis row `φ(t)` into `out` (length `K`).
    pub(crate) fn eval_into(&self, t: f64, out: &mut [f64]) {
        match self {
            ChartBasisKind::Periodic { n_harmonics } => {
                let two_pi = 2.0 * std::f64::consts::PI;
                out[0] = 1.0;
                for h in 1..=*n_harmonics {
                    let angle = two_pi * (h as f64) * t;
                    out[2 * h - 1] = angle.sin();
                    out[2 * h] = angle.cos();
                }
            }
        }
    }
}

/// Row-chunk size for the rayon parallel sweep over the `N` observations.
const ROW_CHUNK: usize = 256;
/// Below this many rows the serial path is used (thread spin-up not worth it).
const PARALLEL_ROW_MIN: usize = 512;

/// Solve the rank-1 chart coordinate `t_f(x)` for every `(row, atom)` pair.
///
/// * `x` — `(N, D)` observations (f64).
/// * `decoders` — `(F, K, D)` decoder blocks; row `k` of block `f` is the
///   ambient vector `D_f[k, :]`.
/// * `basis` — the chart basis; its [`ChartBasisKind::width`] must equal `K`.
/// * `prev_positions` / `gate_weights` — optional `(N, F)` matrices from the
///   previous sweep. When BOTH are supplied the solver targets the
///   leave-one-out residual `x − Σ_{g≠f} gate_g · m_g(t_g)`; when either is
///   `None` the plain target `x` is used.
///
/// Returns solved coordinates `(N, F)` in `[0, 1)`.
pub fn solve_chart_coordinates(
    x: ArrayView2<'_, f64>,
    decoders: ArrayView3<'_, f64>,
    basis: ChartBasisKind,
    prev_positions: Option<ArrayView2<'_, f64>>,
    gate_weights: Option<ArrayView2<'_, f64>>,
) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let d = x.ncols();
    let f = decoders.shape()[0];
    let k = decoders.shape()[1];
    let d_dec = decoders.shape()[2];
    if d_dec != d {
        return Err(format!(
            "solve_chart_coordinates: decoder ambient dim {d_dec} != x ambient dim {d}"
        ));
    }
    if basis.width() != k {
        return Err(format!(
            "solve_chart_coordinates: basis width {} != decoder K {k}",
            basis.width()
        ));
    }
    if f == 0 || n == 0 {
        return Ok(Array2::<f64>::zeros((n, f)));
    }

    // COEFFICIENT-SPACE restructure.  Since
    // `⟨y, m_f(t)⟩ = φ(t)ᵀ (D_f y)`, projecting each target once into every
    // atom's K-dimensional coefficient space (one GEMM, `N·D·F·K`) leaves the
    // exact stationary solve independent of ambient dimension.  Likewise,
    // `‖m_f(t)‖² = φ(t)ᵀ (D_f D_fᵀ) φ(t)` needs only the K×K per-atom Gram.
    //
    // dec_flat: (F·K, D) — atom-major flattening of the decoder blocks.
    let dec_flat = decoders
        .to_shape((f * k, d))
        .map_err(|e| format!("solve_chart_coordinates: decoder reshape failed: {e}"))?
        .to_owned();
    // Per-atom Grams G_f = D_f D_fᵀ (K, K), and their reusable Fourier
    // quadratic forms.  Only the linear Fourier term changes by row.
    let mut grams = Array3::<f64>::zeros((f, k, k));
    let mut extrema = Vec::with_capacity(f);
    for fi in 0..f {
        let block = decoders.index_axis(ndarray::Axis(0), fi); // (K, D)
        let gram = fast_abt(&block, &block);
        extrema.push(
            PeriodicCurveExtrema::from_gram(gram.view()).map_err(|error| {
                format!("solve_chart_coordinates: atom {fi} periodic Gram: {error}")
            })?,
        );
        grams.index_axis_mut(ndarray::Axis(0), fi).assign(&gram);
    }
    // Plain-sweep projections P[row, f·K + k] = ⟨x_row, D_f[k, :]⟩ — one GEMM.
    let proj_x = fast_abt(&x, &dec_flat); // (N, F·K)

    // Leave-one-out residual mode requires BOTH previous positions and gates.
    let loo = match (prev_positions, gate_weights) {
        (Some(p), Some(w)) => {
            if p.shape() != [n, f] {
                return Err(format!(
                    "solve_chart_coordinates: prev_positions shape {:?} != (N, F) = ({n}, {f})",
                    p.shape()
                ));
            }
            if w.shape() != [n, f] {
                return Err(format!(
                    "solve_chart_coordinates: gate_weights shape {:?} != (N, F) = ({n}, {f})",
                    w.shape()
                ));
            }
            Some((p, w))
        }
        _ => None,
    };

    // LOO sweep GEMMs. `W[row, f·K + kk] = gate_f · φ_kk(prev_f)` assembles the
    // previous reconstruction `full = W · dec_flat` in one GEMM; the residual
    // projections `⟨x − full, D_f[k, :]⟩` are a second GEMM; the add-back term
    // `⟨c_f, m_f(t)⟩ = gate_f · φ(prev_f)ᵀ G_f φ(t)` never touches ambient space.
    let proj_target = if let Some((p, w)) = loo {
        let mut weights = Array2::<f64>::zeros((n, f * k));
        {
            use rayon::prelude::*;
            weights
                .axis_chunks_iter_mut(ndarray::Axis(0), ROW_CHUNK)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk, mut block)| {
                    let start = chunk * ROW_CHUNK;
                    let mut phi_row = vec![0.0_f64; k];
                    for local in 0..block.nrows() {
                        let row = start + local;
                        for fi in 0..f {
                            let gate = w[[row, fi]];
                            basis.eval_into(p[[row, fi]], &mut phi_row);
                            for (kk, &phi_v) in phi_row.iter().enumerate() {
                                block[[local, fi * k + kk]] = gate * phi_v;
                            }
                        }
                    }
                });
        }
        let full = fast_ab(&weights, &dec_flat); // (N, D)
        let residual = &x.to_owned() - &full;
        fast_abt(&residual, &dec_flat) // (N, F·K)
    } else {
        proj_x
    };

    // Enumerate and score every analytic stationary point in coefficient space.
    let solve_rows = |start: usize, end: usize, out: &mut Array2<f64>| -> Result<(), String> {
        let mut phi_prev = vec![0.0_f64; k];
        let mut gram_phi = vec![0.0_f64; k];
        let mut linear = vec![0.0_f64; k];
        for row in start..end {
            let out_row = row - start;
            for fi in 0..f {
                // LOO add-back coefficients v = gate_f · G_f φ(prev_f), so
                // dot(g) = proj_target[row, f·K..] · φ_g + v · φ_g.
                let mut has_addback = false;
                if let Some((p, w)) = loo {
                    let gate = w[[row, fi]];
                    if gate != 0.0 {
                        has_addback = true;
                        basis.eval_into(p[[row, fi]], &mut phi_prev);
                        let gram = grams.index_axis(ndarray::Axis(0), fi);
                        for a in 0..k {
                            let mut acc = 0.0_f64;
                            for b in 0..k {
                                acc += gram[[a, b]] * phi_prev[b];
                            }
                            gram_phi[a] = gate * acc;
                        }
                    }
                }
                for kk in 0..k {
                    linear[kk] = proj_target[[row, fi * k + kk]]
                        + if has_addback { gram_phi[kk] } else { 0.0 };
                }
                let solution = extrema[fi]
                    .maximize_nonnegative_profiled_score(&linear)
                    .map_err(|error| {
                        format!(
                            "solve_chart_coordinates: row {row}, atom {fi} stationary solve: {error}"
                        )
                    })?;
                out[[out_row, fi]] = solution.coordinate;
            }
        }
        Ok(())
    };

    let mut solved = Array2::<f64>::zeros((n, f));
    if n >= PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
        use rayon::prelude::*;
        let n_chunks = n.div_ceil(ROW_CHUNK);
        let blocks: Vec<Result<(usize, usize, Array2<f64>), String>> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * ROW_CHUNK;
                let end = (start + ROW_CHUNK).min(n);
                let mut local = Array2::<f64>::zeros((end - start, f));
                solve_rows(start, end, &mut local)?;
                Ok((start, end, local))
            })
            .collect();
        for block in blocks {
            let (start, end, slice) = block?;
            solved.slice_mut(ndarray::s![start..end, ..]).assign(&slice);
        }
    } else {
        solve_rows(0, n, &mut solved)?;
    }

    Ok(solved)
}

/// Value and encoder-gradient of the period-1 coordinate alignment penalty.
///
/// The E-step solves the chart coordinates as tape CONSTANTS, so the encoder
/// position head receives no reconstruction gradient. This penalty pulls the
/// tape-connected encoder positions `e` toward the detached solved positions
/// `s` on the unit circle (period 1), keeping the head learning. For each
/// `(row, atom)` entry with `d = e − s`, the shortest signed circular offset
/// into `(−1/2, 1/2]` is
///
/// ```text
///     wrap(d) = d − round_ties_even(d) ,
/// ```
/// and the penalty is the mean of its square over all `M = N·F` entries:
///
/// ```text
///     P = (1/M) · Σ wrap(d)² .
/// ```
///
/// `round_ties_even` (round half to even) matches `torch.round`, so ties at
/// half-integer offsets break identically to the torch reference. `round` is
/// locally constant, so its derivative is zero almost everywhere and
///
/// ```text
///     ∂P/∂e_{ij} = (2/M) · wrap(d_{ij}) .
/// ```
///
/// Returns `(P, G)` where `G ∈ R^{N×F}` is `∂P/∂e`. `s` is a constant (the
/// detached solve), so no gradient flows to it. An empty input (`M = 0`)
/// returns `(0, [])`.
pub fn position_alignment_penalty(
    encoder: ArrayView2<f64>,
    solved: ArrayView2<f64>,
) -> Result<(f64, Array2<f64>), String> {
    if encoder.dim() != solved.dim() {
        return Err(format!(
            "position_alignment_penalty shape mismatch: encoder {:?} vs solved {:?}",
            encoder.dim(),
            solved.dim()
        ));
    }
    let (n, f) = encoder.dim();
    let total = n * f;
    let mut grad = Array2::<f64>::zeros((n, f));
    if total == 0 {
        return Ok((0.0, grad));
    }
    let inv = 1.0 / total as f64;
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..f {
            let d = encoder[[i, j]] - solved[[i, j]];
            let wrapped = d - d.round_ties_even();
            sum_sq += wrapped * wrapped;
            grad[[i, j]] = 2.0 * inv * wrapped;
        }
    }
    Ok((sum_sq * inv, grad))
}

#[cfg(test)]
mod position_alignment_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn wrap_takes_the_short_way_around_the_circle() {
        // Positions 0.95 and 0.05 are 0.10 apart across the period-1 seam, not
        // 0.90. wrap(0.90) = 0.90 − round(0.90) = 0.90 − 1 = −0.10.
        let enc = array![[0.95]];
        let solved = array![[0.05]];
        let (value, grad) = position_alignment_penalty(enc.view(), solved.view()).unwrap();
        assert!((value - 0.10_f64 * 0.10).abs() < 1e-12);
        // ∂P/∂e = 2·wrap/M = 2·(−0.10)/1 = −0.20.
        assert!((grad[[0, 0]] + 0.20).abs() < 1e-12);
    }

    #[test]
    fn identical_positions_have_zero_penalty_and_gradient() {
        let enc = array![[0.3, 0.7], [0.1, 0.9]];
        let (value, grad) = position_alignment_penalty(enc.view(), enc.view()).unwrap();
        assert_eq!(value, 0.0);
        assert!(grad.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn matches_hand_computed_multi_entry_mean() {
        // enc − solved = [[0.10, -0.10], [0.60, 0.00]]. Wrapping 0.60 → 0.60 − 1
        // = −0.40 (the short way). Squares: [0.01, 0.01, 0.16, 0.00]; mean over
        // M = 4 is 0.18/4 = 0.045.
        let enc = array![[0.10, 0.40], [0.60, 0.25]];
        let solved = array![[0.00, 0.50], [0.00, 0.25]];
        let (value, grad) = position_alignment_penalty(enc.view(), solved.view()).unwrap();
        assert!((value - 0.045).abs() < 1e-12);
        // Gradients: 2·wrap/4 = wrap/2. wrap = [0.10, -0.10, -0.40, 0.00].
        assert!((grad[[0, 0]] - 0.05).abs() < 1e-12);
        assert!((grad[[0, 1]] + 0.05).abs() < 1e-12);
        assert!((grad[[1, 0]] + 0.20).abs() < 1e-12);
        assert!((grad[[1, 1]] - 0.00).abs() < 1e-12);
    }

    #[test]
    fn half_integer_offset_breaks_ties_to_even_like_torch() {
        // d = 0.5 → round_ties_even(0.5) = 0 → wrap = 0.5. d = 1.5 →
        // round_ties_even(1.5) = 2 → wrap = -0.5. Both square to 0.25.
        let enc = array![[0.5, 1.5]];
        let solved = array![[0.0, 0.0]];
        let (value, _grad) = position_alignment_penalty(enc.view(), solved.view()).unwrap();
        assert!((value - 0.25).abs() < 1e-12);
    }

    #[test]
    fn shape_mismatch_errors() {
        let enc = array![[0.1, 0.2]];
        let solved = array![[0.1]];
        assert!(position_alignment_penalty(enc.view(), solved.view()).is_err());
    }
}

#[cfg(test)]
mod chart_coordinate_solve_tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Build a periodic decoder block whose first-harmonic (sin, cos) rows plant
    /// a unit circle in the ambient plane spanned by two orthonormal frame
    /// columns; all other rows/dims are zero. Returns `(F=1, K, D)` decoders.
    ///
    /// This curve is a PURE circle: `m(t+½) = −m(t)`. Profiling the model's
    /// non-negative amplitude selects the member whose inner product with the
    /// target is positive, so the coordinate remains identifiable modulo one.
    fn planted_circle_decoder(n_harmonics: usize, d: usize, ax: usize, ay: usize) -> Array3<f64> {
        let k = 2 * n_harmonics + 1;
        let mut dec = Array3::<f64>::zeros((1, k, d));
        // sin(2πt) row → axis ax, cos(2πt) row → axis ay (first harmonic).
        dec[[0, 1, ax]] = 1.0; // sin row
        dec[[0, 2, ay]] = 1.0; // cos row
        dec
    }

    /// Evaluate the single decoded curve `m(t) = Σ_k φ_k(t)·D_0[k,:]`.
    fn eval_curve(dec: &Array3<f64>, basis: ChartBasisKind, t: f64) -> Vec<f64> {
        let k = dec.shape()[1];
        let d = dec.shape()[2];
        let mut phi = vec![0.0_f64; k];
        basis.eval_into(t, &mut phi);
        let block = dec.index_axis(ndarray::Axis(0), 0);
        let mut m = vec![0.0_f64; d];
        for col in 0..d {
            let mut v = 0.0_f64;
            for kk in 0..k {
                v += phi[kk] * block[[kk, col]];
            }
            m[col] = v;
        }
        m
    }

    #[test]
    fn exactness_target_at_off_lattice_phase_returns_that_phase() {
        let n_harmonics = 3;
        let k = 2 * n_harmonics + 1;
        let d = 16;
        let basis = ChartBasisKind::Periodic { n_harmonics };

        // A GENERIC decoder (DC + all harmonics, distinct ambient axes per row)
        // breaks the pure-circle antipodal degeneracy, so the global score
        // maximum at a target sitting exactly on the curve is unique.
        let mut dec = Array3::<f64>::zeros((1, k, d));
        for kk in 0..k {
            for col in 0..d {
                dec[[0, kk, col]] =
                    ((kk * 7 + col * 3 + 1) as f64 * 0.31).sin() + 0.1 * (kk as f64 - col as f64);
            }
        }

        // Pick a deliberately off-lattice phase and place the target exactly on
        // the decoded curve there, scaled by an arbitrary amplitude.
        let tg = 0.173_205_080_756_887_73;
        let amp = 2.3_f64;
        let m = eval_curve(&dec, basis, tg);
        let mut target = Array2::<f64>::zeros((1, d));
        for col in 0..d {
            target[[0, col]] = amp * m[col];
        }

        let solved = solve_chart_coordinates(target.view(), dec.view(), basis, None, None).unwrap();
        assert!(
            (solved[[0, 0]] - tg).abs() < 1e-9,
            "solver returned {} but the target sits exactly at t = {tg}",
            solved[[0, 0]]
        );
    }

    #[test]
    fn recovers_better_reconstruction_than_random_or_encoder_init() {
        // Synthetic planted circle: rows are points a·(cos φ, sin φ) in a known
        // 2-plane of R^D at random amplitudes/phases, plus small ambient noise.
        let n = 400;
        let d = 16;
        let n_harmonics = 3;
        let (ax, ay) = (3usize, 9usize);
        let dec = planted_circle_decoder(n_harmonics, d, ax, ay);
        let basis = ChartBasisKind::Periodic { n_harmonics };
        let two_pi = 2.0 * std::f64::consts::PI;

        // Deterministic pseudo-random draws (LCG) — no external rng dependency.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        let mut x = Array2::<f64>::zeros((n, d));
        let mut true_t = vec![0.0_f64; n];
        let mut rand_t = vec![0.0_f64; n];
        for row in 0..n {
            let t = next();
            let amp = 0.5 + next();
            true_t[row] = t;
            rand_t[row] = next();
            let angle = two_pi * t;
            x[[row, ax]] = amp * angle.sin();
            x[[row, ay]] = amp * angle.cos();
            for col in 0..d {
                x[[row, col]] += 0.02 * (next() - 0.5);
            }
        }

        let solved = solve_chart_coordinates(x.view(), dec.view(), basis, None, None).unwrap();

        // Reconstruction error at a set of coordinates, amplitude profiled per
        // row (the honest E-step reconstruction: best amplitude at the coord).
        let recon_sse = |coords: &[f64]| -> f64 {
            let mut phi = vec![0.0_f64; 2 * n_harmonics + 1];
            let mut sse = 0.0_f64;
            for row in 0..n {
                basis.eval_into(coords[row], &mut phi);
                // curve m(t) for the single atom.
                let block = dec.index_axis(ndarray::Axis(0), 0);
                let mut m = vec![0.0_f64; d];
                let mut nrm2 = 0.0_f64;
                let mut dot = 0.0_f64;
                for col in 0..d {
                    let mut v = 0.0_f64;
                    for kk in 0..(2 * n_harmonics + 1) {
                        v += phi[kk] * block[[kk, col]];
                    }
                    m[col] = v;
                    nrm2 += v * v;
                    dot += x[[row, col]] * v;
                }
                let a = if nrm2 > 1e-30 {
                    (dot / nrm2).max(0.0)
                } else {
                    0.0
                };
                for col in 0..d {
                    let r = x[[row, col]] - a * m[col];
                    sse += r * r;
                }
            }
            sse
        };

        let sse_solved = recon_sse(&solved.column(0).to_vec());
        let sse_rand = recon_sse(&rand_t);
        assert!(
            sse_solved < 0.25 * sse_rand,
            "solved reconstruction SSE {sse_solved} should beat random-init SSE {sse_rand} by a wide margin"
        );

        // Coordinates should track the planted phase closely. The non-negative
        // amplitude rejects the antipode `m(t+½) = −m(t)`, so compare on the
        // model's full period rather than quotienting away the sign error.
        let mut circ_err = 0.0_f64;
        for row in 0..n {
            let raw = (solved[[row, 0]] - true_t[row]).abs().rem_euclid(1.0);
            let dt = raw.min(1.0 - raw);
            circ_err += dt;
        }
        circ_err /= n as f64;
        assert!(
            circ_err < 0.02,
            "mean circular coordinate error {circ_err} should be small on a clean planted circle"
        );
    }
}
