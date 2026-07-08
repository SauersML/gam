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
//! `t` (minimizing `‖y − a·m_f(t)‖²` over the free amplitude `a`) is
//!
//! ```text
//!     score_f(t) = ⟨y, m_f(t)⟩² / ‖m_f(t)‖² .
//! ```
//!
//! The solver returns `argmax_t score_f(t)` over a uniform grid of
//! `8 · K` points on `[0, 1)` (resolution derived from the basis width, not a
//! knob). A first sweep uses `y = x` (the row); an optional second sweep uses
//! the leave-one-out residual target
//! `y_f = x − Σ_{g≠f} gate_g · m_g(t_g)` built from the previous sweep's
//! positions and gate weights, so co-active atoms carve complementary charts.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

/// Basis family the coordinate solver evaluates on the grid. Periodic
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
    fn width(&self) -> usize {
        match self {
            ChartBasisKind::Periodic { n_harmonics } => 2 * n_harmonics + 1,
        }
    }

    /// Evaluate the basis row `φ(t)` into `out` (length `K`).
    fn eval_into(&self, t: f64, out: &mut [f64]) {
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

    // Grid resolution is DERIVED from the basis width (8 samples per basis
    // column), never a caller knob.
    let grid_res = 8 * k;

    // Precompute grid basis rows φ(t_g), the per-atom decoded curves
    // m_{f,g} = Σ_k φ_g[k]·D_f[k,:], and their squared norms ‖m_{f,g}‖².
    let mut grid_t = Vec::with_capacity(grid_res);
    let mut grid_phi = Array2::<f64>::zeros((grid_res, k));
    {
        let mut row = vec![0.0_f64; k];
        for g in 0..grid_res {
            let t = (g as f64) / (grid_res as f64);
            grid_t.push(t);
            basis.eval_into(t, &mut row);
            grid_phi.row_mut(g).assign(&Array1::from(row.clone()));
        }
    }
    // grid_curve[(f, g, :)] = m_{f,g}; grid_nrm2[(f, g)] = ‖m_{f,g}‖².
    let mut grid_curve = Array3::<f64>::zeros((f, grid_res, d));
    let mut grid_nrm2 = Array2::<f64>::zeros((f, grid_res));
    for fi in 0..f {
        let block = decoders.index_axis(ndarray::Axis(0), fi); // (K, D)
        for g in 0..grid_res {
            let phi = grid_phi.row(g);
            let mut acc = grid_curve.slice_mut(ndarray::s![fi, g, ..]);
            let mut nrm2 = 0.0_f64;
            for col in 0..d {
                let mut v = 0.0_f64;
                for row in 0..k {
                    v += phi[row] * block[[row, col]];
                }
                acc[col] = v;
                nrm2 += v * v;
            }
            grid_nrm2[[fi, g]] = nrm2;
        }
    }

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

    // Solve one contiguous block of rows into `out`, whose row `i` corresponds
    // to observation `start + i`.
    let solve_rows = |start: usize, end: usize, out: &mut Array2<f64>| {
        let mut phi_row = vec![0.0_f64; k];
        // Scratch for the leave-one-out per-atom contributions of one row.
        let mut contrib = Array2::<f64>::zeros((f, d));
        for row in start..end {
            let out_row = row - start;
            let xr = x.row(row);

            // In leave-one-out mode, precompute each atom's contribution
            // c_g = gate_g · m_g(t_g) at its previous coordinate, and the full
            // reconstruction Σ_g c_g, so target_f = x − full + c_f.
            let mut full = Array1::<f64>::zeros(d);
            if let Some((p, w)) = loo {
                for g in 0..f {
                    let tg = p[[row, g]];
                    let gate = w[[row, g]];
                    basis.eval_into(tg, &mut phi_row);
                    let block = decoders.index_axis(ndarray::Axis(0), g); // (K, D)
                    let mut cg = contrib.row_mut(g);
                    for col in 0..d {
                        let mut v = 0.0_f64;
                        for kk in 0..k {
                            v += phi_row[kk] * block[[kk, col]];
                        }
                        let scaled = gate * v;
                        cg[col] = scaled;
                        full[col] += scaled;
                    }
                }
            }

            for fi in 0..f {
                // Build the projection target for this atom.
                // Plain sweep: y = x. LOO sweep: y = x − (full − c_f).
                let mut best_score = f64::NEG_INFINITY;
                let mut best_t = 0.0_f64;
                for g in 0..grid_res {
                    let nrm2 = grid_nrm2[[fi, g]];
                    if nrm2 <= 1e-30 {
                        continue;
                    }
                    let curve = grid_curve.slice(ndarray::s![fi, g, ..]);
                    let mut dot = 0.0_f64;
                    if loo.is_some() {
                        let cf = contrib.row(fi);
                        for col in 0..d {
                            let y = xr[col] - full[col] + cf[col];
                            dot += y * curve[col];
                        }
                    } else {
                        for col in 0..d {
                            dot += xr[col] * curve[col];
                        }
                    }
                    let score = dot * dot / nrm2;
                    if score > best_score {
                        best_score = score;
                        best_t = grid_t[g];
                    }
                }
                out[[out_row, fi]] = best_t;
            }
        }
    };

    let mut solved = Array2::<f64>::zeros((n, f));
    if n >= PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
        use rayon::prelude::*;
        let n_chunks = n.div_ceil(ROW_CHUNK);
        let blocks: Vec<(usize, usize, Array2<f64>)> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * ROW_CHUNK;
                let end = (start + ROW_CHUNK).min(n);
                let mut local = Array2::<f64>::zeros((end - start, f));
                solve_rows(start, end, &mut local);
                (start, end, local)
            })
            .collect();
        for (start, end, slice) in blocks {
            solved.slice_mut(ndarray::s![start..end, ..]).assign(&slice);
        }
    } else {
        solve_rows(0, n, &mut solved);
    }

    Ok(solved)
}

#[cfg(test)]
mod chart_coordinate_solve_tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Build a periodic decoder block whose first-harmonic (sin, cos) rows plant
    /// a unit circle in the ambient plane spanned by two orthonormal frame
    /// columns; all other rows/dims are zero. Returns `(F=1, K, D)` decoders.
    ///
    /// This curve is a PURE circle: `m(t+½) = −m(t)`, so the amplitude-profiled
    /// (sign-free) projection score is identical at `t` and `t+½` — the
    /// coordinate is identifiable only modulo ½. Recovery is measured
    /// accordingly.
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
    fn exactness_target_on_grid_point_returns_that_grid_point() {
        let n_harmonics = 3;
        let k = 2 * n_harmonics + 1;
        let d = 16;
        let basis = ChartBasisKind::Periodic { n_harmonics };
        let grid_res = 8 * k;

        // A GENERIC decoder (DC + all harmonics, distinct ambient axes per row)
        // breaks the pure-circle antipodal degeneracy, so the grid argmax at a
        // target sitting exactly on the curve is unique. Deterministic entries.
        let mut dec = Array3::<f64>::zeros((1, k, d));
        for kk in 0..k {
            for col in 0..d {
                dec[[0, kk, col]] =
                    ((kk * 7 + col * 3 + 1) as f64 * 0.31).sin() + 0.1 * (kk as f64 - col as f64);
            }
        }

        // Pick a grid point t_g and place the target EXACTLY on the decoded
        // curve at that point, scaled by an arbitrary amplitude.
        let g = 17usize;
        let tg = (g as f64) / (grid_res as f64);
        let amp = 2.3_f64;
        let m = eval_curve(&dec, basis, tg);
        let mut target = Array2::<f64>::zeros((1, d));
        for col in 0..d {
            target[[0, col]] = amp * m[col];
        }

        let solved =
            solve_chart_coordinates(target.view(), dec.view(), basis, None, None).unwrap();
        assert!(
            (solved[[0, 0]] - tg).abs() < 1e-12,
            "solver returned {} but the target sits exactly on grid point t_g = {tg}",
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
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

        let solved =
            solve_chart_coordinates(x.view(), dec.view(), basis, None, None).unwrap();

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
                let a = if nrm2 > 1e-30 { dot / nrm2 } else { 0.0 };
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

        // Coordinates should track the planted phase closely. A PURE circle is
        // identifiable only up to the antipode (m(t+½) = −m(t) reconstructs
        // equally under the sign-free amplitude), so the recovered coordinate is
        // compared modulo ½ (period-½ wrap into [0, ¼]).
        let mut circ_err = 0.0_f64;
        for row in 0..n {
            let raw = (solved[[row, 0]] - true_t[row]).rem_euclid(0.5);
            let dt = raw.min(0.5 - raw);
            circ_err += dt;
        }
        circ_err /= n as f64;
        assert!(
            circ_err < 0.02,
            "mean half-period coordinate error {circ_err} should be small on a clean planted circle"
        );
    }
}
