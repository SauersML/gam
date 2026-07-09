//! Chunked-seed accumulation for the overcomplete (`K > P`) hard-TopK CURVED
//! lane (#2134 walls 1+2, #1893).
//!
//! The cold-start seed fits each atom's decoder by least squares of the gated
//! design `D_k = diag(a_·k)·Φ_k` (`N × M_k`) against the current residual
//! `R` (`N × P`): `β_k = D_k⁺ R`. The dense path
//! ([`super::SaeManifoldTerm::seed_cold_start_disjoint_charts`]) forms the whole
//! `(N × M_k)` design and solves it with a thin SVD
//! ([`super::solve_design_least_squares`]). At LLM scale (`N` in the millions)
//! that full-height design is exactly the resident intermediate the front door
//! refuses to let the driver build.
//!
//! This module builds the SAME per-atom seed from the NORMAL EQUATIONS
//! accumulated one row chunk at a time:
//!
//!   * `G_k = D_kᵀ D_k`   (`M_k × M_k`, symmetric PSD — the design Gram);
//!   * `B_k = D_kᵀ R`     (`M_k × P`      — the design/residual cross term).
//!
//! Both are plain sums of per-row rank-1 contributions, so summing a corpus's
//! chunks yields the identical `(G_k, B_k)` as the full-batch build regardless
//! of the chunk boundaries (addition is associative; the per-row terms are
//! accumulated in row order, so the result is BIT-for-bit chunk-invariant — see
//! [`tests::chunk_accumulation_is_bit_invariant`]). The seed is then the
//! pseudo-inverse solve `β_k = G_k⁺ B_k`, which — because `G_k = V Σ² Vᵀ` and
//! `B_k = V Σ Uᵀ R` share the design's right singular vectors — equals the thin
//! SVD solve `V Σ⁻¹ Uᵀ R` the dense path produces, to tolerance, at the SAME
//! rank cutoff (`σ ≤ σ_max·ε·max(N, M)`, applied here as `μ ≤ μ_max·(ε·max(N,
//! M))²` on the Gram eigenvalues `μ = σ²`; see
//! [`tests::chunked_normal_eq_matches_dense_svd_seed`]).
//!
//! Peak memory is the chunk window `O(chunk_rows · M_k)` plus the two
//! accumulators `O(M_k² + M_k·P)` — never `O(N · M_k)`.

use super::*;

/// Per-atom decoder normal equations `(G_k, B_k)` accumulated over row chunks.
///
/// `gram` is `M_k × M_k`, `cross` is `M_k × P`. Row contributions are summed in
/// row order (rank-1 updates), so the accumulated system is independent of the
/// chunk sizes it was streamed in.
#[derive(Clone, Debug)]
pub(crate) struct AtomDecoderNormalEq {
    /// `G_k = Σ_i d_i d_iᵀ` where `d_i` is row `i` of the gated design `D_k`.
    gram: Array2<f64>,
    /// `B_k = Σ_i d_i r_iᵀ` where `r_i` is row `i` of the residual `R`.
    cross: Array2<f64>,
    /// Rows accumulated so far — the `N` that sets the rank cutoff, matching the
    /// dense thin-SVD `max(N, M)` scaling.
    rows: usize,
}

impl AtomDecoderNormalEq {
    /// An empty `(G_k = 0, B_k = 0)` system for an `M`-column design against a
    /// `P`-wide residual.
    pub(crate) fn zeros(m: usize, p: usize) -> Self {
        Self {
            gram: Array2::<f64>::zeros((m, m)),
            cross: Array2::<f64>::zeros((m, p)),
            rows: 0,
        }
    }

    /// Add one row chunk's contribution: `G_k += D_chunkᵀ D_chunk`,
    /// `B_k += D_chunkᵀ R_chunk`. `design_chunk` is the already-gated design
    /// `diag(a)·Φ` over the chunk rows (`n_chunk × M`); `residual_chunk` is the
    /// matching residual rows (`n_chunk × P`).
    ///
    /// Accumulation is an explicit per-row rank-1 sweep (not a blocked GEMM) so
    /// the running sum is deterministic and bit-identical no matter how the rows
    /// were chunked — the property the streaming seed relies on to equal the
    /// full-batch build.
    pub(crate) fn accumulate_chunk(
        &mut self,
        design_chunk: ArrayView2<'_, f64>,
        residual_chunk: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let m = self.gram.nrows();
        let p = self.cross.ncols();
        if design_chunk.ncols() != m {
            return Err(format!(
                "AtomDecoderNormalEq::accumulate_chunk: design has {} cols, expected M={m}",
                design_chunk.ncols()
            ));
        }
        if residual_chunk.ncols() != p {
            return Err(format!(
                "AtomDecoderNormalEq::accumulate_chunk: residual has {} cols, expected P={p}",
                residual_chunk.ncols()
            ));
        }
        if design_chunk.nrows() != residual_chunk.nrows() {
            return Err(format!(
                "AtomDecoderNormalEq::accumulate_chunk: design rows {} != residual rows {}",
                design_chunk.nrows(),
                residual_chunk.nrows()
            ));
        }
        for row in 0..design_chunk.nrows() {
            let d = design_chunk.row(row);
            let r = residual_chunk.row(row);
            for i in 0..m {
                let di = d[i];
                if di == 0.0 {
                    continue;
                }
                let mut grow = self.gram.row_mut(i);
                for j in 0..m {
                    grow[j] += di * d[j];
                }
                let mut brow = self.cross.row_mut(i);
                for c in 0..p {
                    brow[c] += di * r[c];
                }
            }
        }
        self.rows += design_chunk.nrows();
        Ok(())
    }

    /// Solve the accumulated system for the decoder `β_k = G_k⁺ B_k` by the
    /// symmetric eigen pseudo-inverse of the Gram.
    ///
    /// Mirrors [`super::solve_design_least_squares`] exactly: eigenvectors of
    /// `G_k = V Σ² Vᵀ` are the design's right singular vectors `V`, so
    /// `β_k = Σ_{μ_i > cut} μ_i⁻¹ v_i (v_iᵀ B_k) = V Σ⁻¹ Uᵀ R`, with the cutoff
    /// carried across the squaring: the dense path drops `σ ≤ σ_max·ε·max(N,M)`,
    /// so we drop `μ ≤ μ_max·(ε·max(N,M))²`.
    pub(crate) fn solve(&self) -> Result<Array2<f64>, String> {
        let m = self.gram.nrows();
        let p = self.cross.ncols();
        // Symmetrise the accumulated Gram before the eigendecomposition (the
        // rank-1 sweep is symmetric in exact arithmetic; this fences rounding).
        let mut gram = self.gram.clone();
        for i in 0..m {
            for j in 0..i {
                let sym = 0.5 * (gram[[i, j]] + gram[[j, i]]);
                gram[[i, j]] = sym;
                gram[[j, i]] = sym;
            }
        }
        let (evals, evecs) = gram
            .eigh(Side::Lower)
            .map_err(|e| format!("AtomDecoderNormalEq::solve: Gram eigendecomposition failed: {e}"))?;
        let max_eig = evals
            .iter()
            .fold(0.0_f64, |acc, &v| if v.is_finite() { acc.max(v) } else { acc });
        if !(max_eig > 0.0) {
            return Err("AtomDecoderNormalEq::solve: design has zero numerical rank".to_string());
        }
        // Dense cutoff is on singular values `σ = √μ`: `σ_max·ε·max(N, M)`.
        // Squared, it is the eigenvalue cutoff `μ_max·(ε·max(N, M))²`.
        let sigma_scale = f64::EPSILON * (self.rows.max(m) as f64);
        let mu_cutoff = max_eig * sigma_scale * sigma_scale;
        // scaled = diag(f(μ)) · (Vᵀ B),  f(μ) = 1/μ above the cutoff else 0.
        let vt_cross = evecs.t().dot(&self.cross);
        let mut scaled = Array2::<f64>::zeros((m, p));
        for i in 0..m {
            let mu = evals[i];
            if mu.is_finite() && mu > mu_cutoff {
                let inv = 1.0 / mu;
                for c in 0..p {
                    scaled[[i, c]] = inv * vt_cross[[i, c]];
                }
            }
        }
        Ok(evecs.dot(&scaled))
    }
}

/// Build the per-atom decoder seed `β_k = D_k⁺ R` from a whole gated design
/// `D_k` (`N × M`) and residual `R` (`N × P`) by streaming row chunks of
/// `chunk_rows`. Returned `β` is `M × P`, equal to the dense thin-SVD solve to
/// tolerance.
///
/// Reference wrapper over [`AtomDecoderNormalEq`] for the parity tests: it takes
/// the full design so it can be compared directly against
/// [`super::solve_design_least_squares`] on the same input. The production
/// cold-start seed
/// ([`super::SaeManifoldTerm::seed_cold_start_disjoint_charts_streaming`]) does
/// NOT take a resident design — it forms each chunk's gated design on the fly
/// and feeds it straight into [`AtomDecoderNormalEq::accumulate_chunk`], so its
/// footprint is the chunk window plus the `M² + M·P` accumulators.
#[cfg(test)]
pub(crate) fn seed_atom_decoder_chunked(
    design: ArrayView2<'_, f64>,
    residual: ArrayView2<'_, f64>,
    chunk_rows: usize,
) -> Result<Array2<f64>, String> {
    if design.nrows() != residual.nrows() {
        return Err(format!(
            "seed_atom_decoder_chunked: design rows {} != residual rows {}",
            design.nrows(),
            residual.nrows()
        ));
    }
    let n = design.nrows();
    let m = design.ncols();
    let p = residual.ncols();
    let step = chunk_rows.max(1);
    let mut eq = AtomDecoderNormalEq::zeros(m, p);
    let mut start = 0usize;
    while start < n {
        let end = (start + step).min(n);
        eq.accumulate_chunk(
            design.slice(s![start..end, ..]),
            residual.slice(s![start..end, ..]),
        )?;
        start = end;
    }
    eq.solve()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A deterministic, well-conditioned gated design and residual.
    fn design_and_residual(n: usize, m: usize, p: usize) -> (Array2<f64>, Array2<f64>) {
        let mut design = Array2::<f64>::zeros((n, m));
        let mut resid = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let t = row as f64 / n as f64;
            for col in 0..m {
                // Gated monomial-ish columns: bounded, full column rank for n≥m.
                design[[row, col]] = ((col as f64 + 1.0) * (t + 0.3)).cos() + 0.1 * col as f64;
            }
            for c in 0..p {
                resid[[row, c]] = (t * (c as f64 + 1.0)).sin() - 0.2 * c as f64 + 0.05 * row as f64;
            }
        }
        (design, resid)
    }

    /// The accumulated `(G, B)` is BIT-for-bit identical no matter the chunk
    /// size: the per-row rank-1 sweep sums in row order, so chunk boundaries
    /// never reorder the additions. This is the exactness the streaming seed
    /// stands on.
    #[test]
    fn chunk_accumulation_is_bit_invariant() {
        let (design, resid) = design_and_residual(97, 4, 3);
        let m = design.ncols();
        let p = resid.ncols();

        // Full batch in one chunk.
        let mut full = AtomDecoderNormalEq::zeros(m, p);
        full.accumulate_chunk(design.view(), resid.view()).unwrap();

        // Same rows, streamed in uneven chunks.
        for &step in &[1usize, 2, 5, 13, 96, 97, 1000] {
            let mut streamed = AtomDecoderNormalEq::zeros(m, p);
            let mut start = 0usize;
            while start < design.nrows() {
                let end = (start + step).min(design.nrows());
                streamed
                    .accumulate_chunk(
                        design.slice(s![start..end, ..]),
                        resid.slice(s![start..end, ..]),
                    )
                    .unwrap();
                start = end;
            }
            assert_eq!(
                streamed.gram, full.gram,
                "Gram must be bit-identical at chunk step {step}"
            );
            assert_eq!(
                streamed.cross, full.cross,
                "cross must be bit-identical at chunk step {step}"
            );
            assert_eq!(streamed.rows, full.rows);
        }
    }

    /// The chunked normal-equation seed equals the dense thin-SVD seed
    /// ([`super::solve_design_least_squares`], the PRODUCTION dense solver) to
    /// tolerance, at every chunk size — including chunks far smaller than the
    /// full height. This is the dense/chunked parity the front door now relies
    /// on to admit the streaming lane.
    #[test]
    fn chunked_normal_eq_matches_dense_svd_seed() {
        let (design, resid) = design_and_residual(200, 5, 4);
        let dense = solve_design_least_squares(design.view(), resid.view())
            .expect("dense SVD seed");

        for &step in &[3usize, 7, 32, 199, 200, 4096] {
            let chunked = seed_atom_decoder_chunked(design.view(), resid.view(), step)
                .expect("chunked normal-equation seed");
            assert_eq!(chunked.dim(), dense.dim());
            let max_abs = chunked
                .iter()
                .zip(dense.iter())
                .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
            assert!(
                max_abs <= 1.0e-9,
                "chunked seed (step {step}) disagrees with dense SVD seed by {max_abs:.3e}"
            );
        }
    }

    /// The residual is reconstructed by the seed as well as the dense solve does
    /// (the LSQ optimality the seed exists to provide): on a design that spans
    /// the residual the fit is near-exact.
    #[test]
    fn chunked_seed_reconstructs_a_spanned_residual() {
        // Residual generated as design · β_true, so the LSQ recovers β_true.
        let (design, _) = design_and_residual(64, 3, 2);
        let beta_true = array![[0.5_f64, -1.2], [2.0, 0.3], [-0.7, 1.1]];
        let resid = design.dot(&beta_true);
        let beta = seed_atom_decoder_chunked(design.view(), resid.view(), 8).unwrap();
        let fit = design.dot(&beta);
        let max_abs = fit
            .iter()
            .zip(resid.iter())
            .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(max_abs <= 1.0e-9, "spanned residual not reconstructed: {max_abs:.3e}");
    }
}
