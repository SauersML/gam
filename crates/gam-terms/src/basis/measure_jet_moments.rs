//! Measure-jet frame data interface: per-cell frozen-weight polynomial
//! moment tables with a binomial-shift merge monoid
//! (`docs/measure_jet_frame.md`, §2 "Data interface: moments or
//! nothing").
//!
//! This module aggregates caller-computed weights into order-0..2 coordinate
//! moments. Those tables exactly determine polynomial couplings under the
//! same frozen weights, including the local affine sufficient statistics used
//! by `measure_jet_smooth.rs`. They do NOT exactly determine Gaussian
//! transforms at moved kernel centers: support curves, Gaussian Gram entries,
//! and Gaussian `XᵀWX` products need their own kernel pass or a separately
//! controlled approximation. Truncation does NOT live here either: the caller
//! computes the Gaussian weights `w_i` (mass × kernel profile, with whatever
//! cutoff its explicit `e^{−ρ²/2}` tolerance budget licenses) and this module
//! only aggregates what it is handed.
//!
//! # The monoid
//!
//! A table holds, per response channel `g` and per coordinate multi-index
//! `α` with `|α| ≤ 2`, the centered moment `μ_α = Σ_i w_i g_i (x_i − c)^α`
//! about the cell reference point `c`. The binomial shift
//!
//! ```text
//!   μ′_α = Σ_{β ≤ α}  C(α, β) (c − c′)^{α−β} μ_β
//! ```
//!
//! re-expresses the same frozen-weight polynomial table about any other
//! center `c′` exactly as a finite polynomial identity. It does not move the
//! Gaussian kernel center or recompute weights. Merging two tables with
//! already-compatible frozen weights is therefore "recenter to a common
//! reference, add componentwise":
//! an associative, commutative monoid whose identity is the empty (all-zero)
//! table at any center. Exact distributed fitting, exact online updates, and
//! bit-reproducibility under sorted reduction are corollaries of that one
//! algebraic fact ([`merge_moment_tables`] is a monoid homomorphism from
//! disjoint row sets under union to tables under ⊕).
//!
//! # Determinism / bit-exactness convention (sorted reduction)
//!
//! Floating-point addition is commutative but not associative, so the monoid
//! laws hold algebraically while bit-patterns depend on reduction ORDER.
//! This module pins one order everywhere:
//!
//! - [`accumulate_moment_table`] splits rows into fixed-size chunks
//!   ([`MEASURE_JET_MOMENT_CHUNK_ROWS`], never derived from thread count),
//!   accumulates each chunk sequentially in row order, and folds the chunk
//!   partials sequentially in chunk-index order — the sorted reduction. The
//!   result is bit-identical across runs, machines, and rayon pool sizes.
//! - [`recenter_moment_table`] evaluates the shift in ONE fixed expression
//!   order (documented at the site).
//! - [`merge_moment_tables`] canonically orients its operands by the
//!   lexicographic total order on centers (`f64::total_cmp` per coordinate),
//!   so `a ⊕ b` and `b ⊕ a` execute the SAME instruction stream and are
//!   bit-identical for arbitrary inputs.
//!
//! Cross-GROUPING bit-identity — `(A⊕B)⊕C` vs `A⊕(B⊕C)` — additionally
//! requires the moment arithmetic itself to be exact; the in-module tests
//! pin it on dyadic lattices (integer coordinates/channels, dyadic weights),
//! where every product and sum is exactly representable, and callers
//! reducing many chunks get run-to-run determinism by folding in chunk-index
//! order exactly as the accumulator does.
//!
//! # 1:1 contract with `assemble_weighted_forms`
//!
//! [`jet_sufficient_stats`] reproduces, in closed form from a stored table
//! whose weights were computed for the same center and scale, exactly the
//! local-fit quantities the current workhorse
//! (`measure_jet_smooth.rs::assemble_weighted_forms`) computes from raw
//! points per (center, scale) block: the kernel mass `q`, the dimensionless
//! weighted feature mean `a_mean`, the dimensionless slope Gram
//! `G = Φ̃ᵀWΦ̃/q`, the weighted channel mean `uᵀv`, and the exact-projection
//! right-hand side `Bᵀv/q` — so the substrate can later replace that
//! same-center point loop without changing a single number.

use ndarray::{Array1, Array2};

/// The local jet-fit sufficient statistics read off one table — exactly the
/// per-block quantities `assemble_weighted_forms` (measure_jet_smooth.rs)
/// computes from raw points when the table weights are frozen at the same
/// center and scale, reproduced in closed form from stored moments.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureJetJetStats {
    /// Kernel mass `q = Σ w_i` (unit-channel zeroth moment).
    pub q: f64,
    /// Weighted mean of the requested value channel: `uᵀv = m0[ch]/q`.
    pub mean: f64,
    /// Dimensionless slope Gram `G = Φ̃ᵀWΦ̃/q = m2[0]/(qε²) − ā·āᵀ` with
    /// `ā = m1[0]/(qε)` (`Φ` rows are `(x_i − c)/ε`).
    pub gram: Array2<f64>,
    /// Local-fit right-hand side `Bᵀv/q = m1[ch]/(qε) − ā·(m0[ch]/q)` — the
    /// vector the exact weighted affine projection consumes.
    pub cross: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    /// Closeness metric for the recenter-exactness gate: relative at scale,
    /// absolute `tol` below unit scale (`|x−y| ≤ tol·(1 + max(|x|,|y|))`).
    pub(crate) fn assert_tables_close(
        a: &MeasureJetMomentTable,
        b: &MeasureJetMomentTable,
        tol: f64,
    ) {
        let pairs = |xs: &[f64], ys: &[f64], label: &str| {
            assert_eq!(xs.len(), ys.len(), "{label}: length mismatch");
            for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
                let scale = 1.0 + x.abs().max(y.abs());
                assert!(
                    (x - y).abs() <= tol * scale,
                    "{label}[{i}]: {x} vs {y} differ beyond {tol} rel"
                );
            }
        };
        pairs(
            a.center.as_slice().expect("contiguous center"),
            b.center.as_slice().expect("contiguous center"),
            "center",
        );
        pairs(
            a.m0.as_slice().expect("contiguous m0"),
            b.m0.as_slice().expect("contiguous m0"),
            "m0",
        );
        pairs(
            a.m1.as_slice().expect("contiguous m1"),
            b.m1.as_slice().expect("contiguous m1"),
            "m1",
        );
        assert_eq!(a.m2.len(), b.m2.len(), "m2: channel count mismatch");
        for (ch, (x, y)) in a.m2.iter().zip(b.m2.iter()).enumerate() {
            pairs(
                x.as_slice().expect("contiguous m2"),
                y.as_slice().expect("contiguous m2"),
                &format!("m2[{ch}]"),
            );
        }
    }

    /// Bit-identity gate: every stored f64 must agree by `to_bits`.
    pub(crate) fn assert_tables_bit_identical(
        a: &MeasureJetMomentTable,
        b: &MeasureJetMomentTable,
    ) {
        let bits = |xs: &[f64], ys: &[f64], label: &str| {
            assert_eq!(xs.len(), ys.len(), "{label}: length mismatch");
            for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{label}[{i}]: {x} vs {y} differ bitwise"
                );
            }
        };
        bits(
            a.center.as_slice().expect("contiguous center"),
            b.center.as_slice().expect("contiguous center"),
            "center",
        );
        bits(
            a.m0.as_slice().expect("contiguous m0"),
            b.m0.as_slice().expect("contiguous m0"),
            "m0",
        );
        bits(
            a.m1.as_slice().expect("contiguous m1"),
            b.m1.as_slice().expect("contiguous m1"),
            "m1",
        );
        assert_eq!(a.m2.len(), b.m2.len(), "m2: channel count mismatch");
        for (ch, (x, y)) in a.m2.iter().zip(b.m2.iter()).enumerate() {
            bits(
                x.as_slice().expect("contiguous m2"),
                y.as_slice().expect("contiguous m2"),
                &format!("m2[{ch}]"),
            );
        }
    }

    /// Deterministic generic-float dataset (no RNG): low-discrepancy
    /// fractional parts, d = 3, with a unit channel and one value channel.
    pub(crate) fn float_dataset(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut coords = Array2::<f64>::zeros((n, 3));
        let mut weights = Array1::<f64>::zeros(n);
        let mut ones = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i + 1) as f64;
            coords[(i, 0)] = (t * 0.618034).fract() * 4.0 - 2.0;
            coords[(i, 1)] = (t * 0.414214).fract() * 3.0 - 1.0;
            coords[(i, 2)] = (t * 0.732051).fract() * 2.0 - 1.5;
            weights[i] = 0.05 + (t * 0.292893).fract();
            ones[i] = 1.0;
            y[i] = (t * 0.539345).fract() * 6.0 - 3.0;
        }
        (coords, weights, ones, y)
    }

    /// Dyadic-lattice dataset: integer coordinates and channel values,
    /// dyadic weights — every moment product and sum is exactly
    /// representable in f64, so the algebraic monoid laws become BIT
    /// identities and the tests below can pin them with `to_bits`.
    pub(crate) fn dyadic_dataset() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let coords = ndarray::array![
            [3.0, -2.0],
            [1.0, 4.0],
            [-5.0, 2.0],
            [2.0, 2.0],
            [4.0, -1.0],
            [0.0, 5.0],
            [-3.0, -4.0],
            [6.0, 1.0],
            [-1.0, 3.0],
            [5.0, -3.0],
            [2.0, 7.0],
            [-6.0, -2.0],
            [3.0, 3.0],
            [1.0, -5.0],
            [4.0, 6.0],
            [-2.0, -3.0],
        ];
        let weights = ndarray::array![
            0.5, 1.0, 2.0, 0.25, 1.5, 0.75, 1.0, 0.5, 2.5, 1.25, 0.5, 3.0, 0.75, 1.0, 1.75, 2.0
        ];
        let ones = Array1::<f64>::ones(16);
        let y = ndarray::array![
            2.0, -3.0, 5.0, 1.0, -4.0, 7.0, 2.0, -6.0, 3.0, 4.0, -2.0, 8.0, 1.0, -7.0, 5.0, -1.0
        ];
        (coords, weights, ones, y)
    }

    #[test]
    pub(crate) fn recenter_is_exact() {
        let (coords, weights, ones, y) = float_dataset(40);
        let channels = [ones.view(), y.view()];
        let c = ndarray::array![0.4, -0.3, 0.9];
        let c_prime = ndarray::array![-1.1, 0.25, 0.5];
        let at_c = accumulate_moment_table(coords.view(), weights.view(), &channels, c.view())
            .expect("accumulation about c");
        let shifted = recenter_moment_table(&at_c, c_prime.view());
        let direct =
            accumulate_moment_table(coords.view(), weights.view(), &channels, c_prime.view())
                .expect("accumulation about c'");
        assert_tables_close(&shifted, &direct, 1e-14);
        // Round trip back to c reproduces the original to the same gate.
        let back = recenter_moment_table(&shifted, c.view());
        assert_tables_close(&back, &at_c, 1e-14);
    }

    #[test]
    pub(crate) fn merge_is_associative_and_commutative_bitwise() {
        // Dyadic lattice ⇒ all moment/shift arithmetic is exact, so the
        // monoid laws hold BITWISE across groupings (the sorted-reduction
        // convention covers generic-float grouping determinism; see the
        // module docs).
        let (coords, weights, ones, y) = dyadic_dataset();
        let chunk = |rows: Range<usize>, center: &Array1<f64>| {
            let ones_c = ones.slice(s![rows.clone()]);
            let y_c = y.slice(s![rows.clone()]);
            accumulate_moment_table(
                coords.slice(s![rows.clone(), ..]),
                weights.slice(s![rows]),
                &[ones_c, y_c],
                center.view(),
            )
            .expect("chunk accumulation")
        };
        let c_a = ndarray::array![2.0, -1.0];
        let c_b = ndarray::array![0.0, 3.0];
        let c_c = ndarray::array![-4.0, 1.0];
        let a = chunk(0..5, &c_a);
        let b = chunk(5..9, &c_b);
        let c = chunk(9..14, &c_c);

        // Commutativity is bitwise for ARBITRARY inputs: the canonical
        // center orientation makes merge(a, b) and merge(b, a) execute
        // identical arithmetic. No recentering needed before comparing.
        let ab = merge_moment_tables(&a, &b).expect("a+b");
        let ba = merge_moment_tables(&b, &a).expect("b+a");
        assert_tables_bit_identical(&ab, &ba);
        // ... including on generic (non-dyadic) float data.
        let (fc, fw, fo, fy) = float_dataset(24);
        let fa = accumulate_moment_table(
            fc.slice(s![0..12, ..]),
            fw.slice(s![0..12]),
            &[fo.slice(s![0..12]), fy.slice(s![0..12])],
            ndarray::array![0.3, -0.7, 0.1].view(),
        )
        .expect("float chunk a");
        let fb = accumulate_moment_table(
            fc.slice(s![12..24, ..]),
            fw.slice(s![12..24]),
            &[fo.slice(s![12..24]), fy.slice(s![12..24])],
            ndarray::array![-0.9, 0.4, 0.6].view(),
        )
        .expect("float chunk b");
        assert_tables_bit_identical(
            &merge_moment_tables(&fa, &fb).expect("fa+fb"),
            &merge_moment_tables(&fb, &fa).expect("fb+fa"),
        );

        // Associativity, bitwise on the exact lattice.
        let ab_c = merge_moment_tables(&ab, &c).expect("(a+b)+c");
        let bc = merge_moment_tables(&b, &c).expect("b+c");
        let a_bc = merge_moment_tables(&a, &bc).expect("a+(b+c)");
        assert_tables_bit_identical(&ab_c, &a_bc);
        // And after recentering both to a common reference.
        let c_ref = ndarray::array![1.0, 2.0];
        assert_tables_bit_identical(
            &recenter_moment_table(&ab_c, c_ref.view()),
            &recenter_moment_table(&a_bc, c_ref.view()),
        );
    }

    #[test]
    pub(crate) fn jet_stats_match_assemble_weighted_forms_math() {
        // Small 2-D point set, replicating assemble_weighted_forms' local
        // loop verbatim from raw points: w_j = mass_j·exp(−d²/(2ε²)),
        // q = Σ w, Φ_{jk} = (x_{jk} − c_k)/ε, a = Φᵀw/q,
        // G = (ΦᵀWΦ)/q − a·aᵀ, uᵀv = wᵀv/q, Bᵀv/q with B = WΦ − w·aᵀ.
        let pts = ndarray::array![
            [0.0, 0.0],
            [0.45, -0.2],
            [-0.35, 0.4],
            [0.25, 0.55],
            [-0.5, -0.45],
            [0.6, 0.3]
        ];
        let masses = ndarray::array![0.22, 0.13, 0.19, 0.11, 0.2, 0.15];
        let y = ndarray::array![0.7, -1.3, 2.1, 0.4, -0.6, 1.9];
        let center = ndarray::array![0.0, 0.0];
        let eps = 0.75;
        let m = pts.nrows();
        let d = pts.ncols();

        // Kernel weights exactly as the workhorse forms them.
        let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
        let mut w = Array1::<f64>::zeros(m);
        let mut q = 0.0_f64;
        for j in 0..m {
            let mut dist2 = 0.0_f64;
            for k in 0..d {
                let dlt = pts[(j, k)] - center[k];
                dist2 += dlt * dlt;
            }
            w[j] = masses[j] * (-dist2 * inv_two_eps2).exp();
            q += w[j];
        }
        let mut phi = Array2::<f64>::zeros((m, d));
        for j in 0..m {
            for k in 0..d {
                phi[(j, k)] = (pts[(j, k)] - center[k]) / eps;
            }
        }
        let a_mean = phi.t().dot(&w) / q;
        let mut wphi = phi.clone();
        for (j, mut row) in wphi.outer_iter_mut().enumerate() {
            row.mapv_inplace(|v| v * w[j]);
        }
        let mut g_ref = phi.t().dot(&wphi);
        g_ref.mapv_inplace(|v| v / q);
        for r in 0..d {
            for c in 0..d {
                g_ref[(r, c)] -= a_mean[r] * a_mean[c];
            }
        }
        let mean_ref = w.dot(&y) / q;
        let mut cross_ref = Array1::<f64>::zeros(d);
        for k in 0..d {
            let mut acc = 0.0_f64;
            for j in 0..m {
                acc += (wphi[(j, k)] - w[j] * a_mean[k]) * y[j];
            }
            cross_ref[k] = acc / q;
        }

        // Substrate path: same caller-computed weights into a moment table.
        let ones = Array1::<f64>::ones(m);
        let table = accumulate_moment_table(
            pts.view(),
            w.view(),
            &[ones.view(), y.view()],
            center.view(),
        )
        .expect("moment table");
        let stats = jet_sufficient_stats(&table, eps, 1).expect("jet stats");

        let tol = 1e-14;
        let close = |x: f64, y_: f64, label: &str| {
            let scale = 1.0 + x.abs().max(y_.abs());
            assert!(
                (x - y_).abs() <= tol * scale,
                "{label}: {x} vs {y_} beyond {tol} rel"
            );
        };
        close(stats.q, q, "q");
        close(stats.mean, mean_ref, "mean");
        for k in 0..d {
            close(stats.cross[k], cross_ref[k], &format!("cross[{k}]"));
            for l in 0..d {
                close(stats.gram[(k, l)], g_ref[(k, l)], &format!("gram[{k},{l}]"));
            }
        }

        // Unit channel: exact constant annihilation at the moment level —
        // mean is exactly 1, cross is identically +0.0.
        let unit_stats = jet_sufficient_stats(&table, eps, 0).expect("unit-channel stats");
        assert_eq!(unit_stats.mean, 1.0, "unit-channel mean must be exactly 1");
        for k in 0..d {
            assert_eq!(
                unit_stats.cross[k], 0.0,
                "unit-channel cross[{k}] must be exactly zero"
            );
        }
    }

    /// LEVEL/TILT truth-recovery gate (#1041). The deficit pattern flagged in
    /// the 8-dataset benchmark — worst on pooled/pointwise risk (RMSE/Brier/R²)
    /// but only mid-pack on calibration SLOPE — is the fingerprint of a biased
    /// affine projection: a systematic shift in the recovered LEVEL `c₀` or a
    /// TILT in the recovered gradient `g`. The local affine sufficient statistic
    /// this module computes (`mean`, `G`, `cross`) is the exact object that
    /// projection consumes, so a bias there would surface here.
    ///
    /// Construct a channel value that is EXACTLY affine in the coordinates,
    /// `v(x) = c₀ + gᵀ(x − center)`, under ARBITRARY (non-symmetric) weights.
    /// The weighted affine projection must then recover `(c₀, g)` with ZERO
    /// residual — the curved/higher-order energy is empty, so any nonzero level
    /// or tilt error is pure projection bias, not a smoothing artifact. We
    /// assert this across SHRINKING kernel widths ε (concentrating the weights),
    /// the regime where a level/tilt bias in the centered second moment `G` or
    /// the centered cross `Bᵀv/q` would be amplified.
    #[test]
    pub(crate) fn affine_projection_recovers_level_and_tilt_without_bias() {
        // Asymmetric, off-center point cloud so the weighted barycenter does
        // NOT coincide with the reference center: this is exactly where a
        // mis-centered (biased) projection would leak the level into the tilt
        // and vice versa.
        let pts = ndarray::array![
            [0.10, -0.30],
            [0.62, 0.05],
            [-0.18, 0.44],
            [0.37, 0.51],
            [-0.46, -0.22],
            [0.71, 0.33],
            [0.05, 0.62],
            [-0.33, 0.14],
        ];
        // Strictly positive, deliberately uneven masses (no symmetry to lean on).
        let masses = ndarray::array![0.31, 0.07, 0.22, 0.05, 0.19, 0.11, 0.27, 0.13];
        let center = ndarray::array![0.05, 0.10];
        let m = pts.nrows();
        let d = pts.ncols();

        // Exact affine truth in ambient coordinates: level c0, gradient g.
        let c0 = 1.37_f64;
        let g = ndarray::array![-0.85_f64, 0.42_f64];
        let mut v = Array1::<f64>::zeros(m);
        for j in 0..m {
            let mut acc = c0;
            for k in 0..d {
                acc += g[k] * (pts[(j, k)] - center[k]);
            }
            v[j] = acc;
        }

        let ones = Array1::<f64>::ones(m);
        // Tighten the kernel across several scales: shrinking eps concentrates
        // the Gaussian weights and amplifies any centering/projection bias.
        for &eps in &[1.0_f64, 0.5, 0.25, 0.12, 0.06] {
            let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
            let mut w = Array1::<f64>::zeros(m);
            for j in 0..m {
                let mut dist2 = 0.0_f64;
                for k in 0..d {
                    let dlt = pts[(j, k)] - center[k];
                    dist2 += dlt * dlt;
                }
                w[j] = masses[j] * (-dist2 * inv_two_eps2).exp();
            }

            let table = accumulate_moment_table(
                pts.view(),
                w.view(),
                &[ones.view(), v.view()],
                center.view(),
            )
            .expect("moment table");
            let stats = jet_sufficient_stats(&table, eps, 1).expect("affine jet stats");

            // The weighted affine projection solves `G b̂ = cross` for the
            // ε-scaled slope; the ambient gradient is b̂/ε and the recovered
            // LEVEL is `mean − āᵀ b̂` (the weighted mean minus the slope's
            // contribution at the weighted barycenter). For an exactly affine
            // truth both must equal the truth with zero residual.
            //
            // Solve the 2×2 SPD system directly (no external solver) so the
            // test pins the projection math, not a library inverse.
            let g00 = stats.gram[(0, 0)];
            let g01 = stats.gram[(0, 1)];
            let g11 = stats.gram[(1, 1)];
            let det = g00 * g11 - g01 * g01;
            assert!(
                det > 1e-10,
                "centered slope Gram must stay nondegenerate at eps={eps}; det={det}"
            );
            let b0 = (g11 * stats.cross[0] - g01 * stats.cross[1]) / det;
            let b1 = (-g01 * stats.cross[0] + g00 * stats.cross[1]) / det;
            // Ambient gradient = scaled slope / eps (Φ rows are (x−c)/ε).
            let grad = [b0 / eps, b1 / eps];

            // Recovered weighted barycenter offset ā (ambient) = a_mean·ε.
            // Level at the reference center = mean − gradᵀ·(barycenter − center)
            //                               = mean − (b̂ᵀ ā).
            let a_mean0 = table.m1[(0, 0)] / (stats.q * eps);
            let a_mean1 = table.m1[(0, 1)] / (stats.q * eps);
            let level = stats.mean - (b0 * a_mean0 + b1 * a_mean1);

            // TILT: the recovered gradient must match the truth — no systematic
            // rotation/scaling of the slope channel.
            assert!(
                (grad[0] - g[0]).abs() <= 1e-9 && (grad[1] - g[1]).abs() <= 1e-9,
                "TILT bias at eps={eps}: recovered gradient {grad:?} vs truth {g:?}"
            );
            // LEVEL: the recovered intercept at the reference center must match
            // the truth — no systematic offset of the reconstructed surface.
            assert!(
                (level - c0).abs() <= 1e-9,
                "LEVEL bias at eps={eps}: recovered {level} vs truth {c0}"
            );
        }
    }

    #[test]
    pub(crate) fn streaming_chunked_accumulation_matches_single_pass() {
        // Four chunks, each accumulated about its OWN center, merged in
        // chunk-index order (the sorted reduction) — versus one pass about
        // the lexicographically smallest chunk center. Dyadic lattice ⇒ the
        // agreement is exact, pinned bitwise.
        let (coords, weights, ones, y) = dyadic_dataset();
        let centers = [
            ndarray::array![-3.0, 2.0], // lexicographic minimum: merge target
            ndarray::array![0.0, 0.0],
            ndarray::array![1.0, -5.0],
            ndarray::array![4.0, 1.0],
        ];
        let chunk = |rows: Range<usize>, center: &Array1<f64>| {
            let ones_c = ones.slice(s![rows.clone()]);
            let y_c = y.slice(s![rows.clone()]);
            accumulate_moment_table(
                coords.slice(s![rows.clone(), ..]),
                weights.slice(s![rows]),
                &[ones_c, y_c],
                center.view(),
            )
            .expect("chunk accumulation")
        };
        let t0 = chunk(0..4, &centers[0]);
        let t1 = chunk(4..8, &centers[1]);
        let t2 = chunk(8..12, &centers[2]);
        let t3 = chunk(12..16, &centers[3]);
        let merged = merge_moment_tables(
            &merge_moment_tables(&merge_moment_tables(&t0, &t1).expect("t0+t1"), &t2)
                .expect("(t0+t1)+t2"),
            &t3,
        )
        .expect("((t0+t1)+t2)+t3");
        let single = accumulate_moment_table(
            coords.view(),
            weights.view(),
            &[ones.view(), y.view()],
            centers[0].view(),
        )
        .expect("single pass");
        // The fold target is the lex-min center, so no final recentering is
        // even needed; pin that and the bitwise agreement.
        assert_tables_bit_identical(&merged, &single);
        // Merging the identity is a no-op.
        let with_zero =
            merge_moment_tables(&merged, &MeasureJetMomentTable::zero(centers[0].clone(), 2))
                .expect("merge with identity");
        assert_tables_bit_identical(&with_zero, &merged);
    }
}
